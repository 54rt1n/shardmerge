import argparse
import json
import os
from pathlib import Path
import torch
from safetensors import safe_open
from safetensors.torch import save_file
from tqdm import tqdm
import shutil
import re # For natural sorting if needed later, but transformer_sort_key is better

# --- Sorting Logic (adapted from convert_pytorch_to_hf.py) ---

def transformer_sort_key(tensor_name):
    """Provides a sort key for tensor names based on typical transformer architecture.

    Sorts layers numerically and orders components within layers logically:
    Input LayerNorm -> Self-Attention (Q, K, V, O) -> Post-Attention LayerNorm -> MLP (Gate, Up, Down).
    Embeddings come first, final LayerNorm and LM Head come last.
    Handles variations like Llama vs standard Transformer naming.
    """
    # Component Priorities (lower number = earlier)
    # Adapted slightly for common variations
    COMPONENT_ORDER = {
        "input_layernorm": 0,
        "self_attn.q_proj": 1,
        "self_attn.k_proj": 2,
        "self_attn.v_proj": 3,
        "self_attn.o_proj": 4,
        "post_attention_layernorm": 5,
        "mlp.gate_proj": 6,        # Llama-style MLP
        "mlp.up_proj": 7,          # Llama-style MLP
        "mlp.down_proj": 8,        # Llama-style MLP
        "mlp.fc_in": 6,            # Standard MLP
        "mlp.fc_out": 7,           # Standard MLP
        "attention.wq": 1,         # Alternative naming
        "attention.wk": 2,
        "attention.wv": 3,
        "attention.wo": 4,
        "ffn_norm": 5,             # Alternative naming
        "feed_forward.w1": 6,      # Alternative naming
        "feed_forward.w3": 7,      # Llama-style SwiGLU uses w1/w3
        "feed_forward.w2": 8,      # Alternative naming
    }
    # Layer Type Priorities
    LAYER_TYPE_ORDER = {
        "model.embed_tokens": 0,
        "tok_embeddings": 0,        # Alternative naming
        "model.layers": 1,
        "layers": 1,               # Alternative naming
        "model.norm": 2,
        "norm": 2,                 # Alternative naming
        "lm_head": 3,
        "output": 3,               # Alternative naming
    }
    MAX_LAYER_TYPE = max(LAYER_TYPE_ORDER.values()) + 1
    MAX_COMPONENT = max(COMPONENT_ORDER.values()) + 1

    parts = tensor_name.split('.')

    # --- Identify Layer Type ---
    layer_type_key = ""
    if parts[0] == "model":
        if parts[1] in ["embed_tokens", "layers", "norm"]:
            layer_type_key = f"model.{parts[1]}"
    elif parts[0] in LAYER_TYPE_ORDER: # Handle top-level keys like 'layers', 'norm', 'output', 'lm_head'
         layer_type_key = parts[0]
    else: # Default case if pattern doesn't match known prefixes
        layer_type_key = parts[0] # Use the first part as a basic grouping

    layer_type_priority = LAYER_TYPE_ORDER.get(layer_type_key, MAX_LAYER_TYPE) # Default to high priority if unknown


    # --- Identify Layer Number (if applicable) ---
    layer_num = -1
    component_parts_index = -1
    if "layers" in layer_type_key and len(parts) > (layer_type_key.count('.') + 1) and parts[layer_type_key.count('.') + 1].isdigit():
         layer_num = int(parts[layer_type_key.count('.') + 1])
         component_parts_index = layer_type_key.count('.') + 2 # Index where component name starts
    # Special case for handling potential non-standard naming like encoder.layer.0...
    elif len(parts) > 1 and parts[1] == "layer" and len(parts) > 2 and parts[2].isdigit():
         layer_num = int(parts[2])
         component_parts_index = 3


    # --- Identify Component within Layer ---
    component_priority = MAX_COMPONENT
    sub_component_name = ""
    if component_parts_index != -1 and len(parts) > component_parts_index:
        # Try matching known component patterns
        potential_component_key = ""
        # Check for patterns like "self_attn.q_proj", "mlp.gate_proj"
        current_key = ""
        for i in range(component_parts_index, len(parts) -1): # Iterate up to the second-to-last part
            current_key = ".".join(parts[component_parts_index:i+1])
            if current_key in COMPONENT_ORDER:
                 potential_component_key = current_key
                 # Continue checking in case a longer match exists (e.g., mlp vs mlp.gate_proj)

        if potential_component_key:
             component_priority = COMPONENT_ORDER[potential_component_key]
        else:
            # Fallback: use the first part of the component name if no known pattern matches
            sub_component_name = ".".join(parts[component_parts_index:]) # Include suffix like .weight
    elif layer_type_priority != MAX_LAYER_TYPE and layer_type_priority > 0: # Not embed/head, likely norm or other root level
         sub_component_name = ".".join(parts[layer_type_key.count('.') + 1:])


    # --- Identify the final part (usually 'weight' or 'bias') ---
    qualifier = parts[-1] if len(parts) > 1 else ""

    # Return a tuple for sorting:
    # (Layer Type Priority, Layer Num, Component Priority, SubComponent Name for fallback, Qualifier, Full Name)
    # Adding the last element (tensor_name) ensures stable sort if all other keys are equal
    return (layer_type_priority, layer_num, component_priority, sub_component_name, qualifier, tensor_name)


# --- Main Resharding Function ---

def reshard_and_reorder_safetensors(input_dir: str, output_dir: str, num_shards: int):
    """
    Reads tensors from an input directory, reorders them based on transformer architecture,
    and saves them into a new set of shards in the output directory.

    Args:
        input_dir: Directory containing the original model.safetensors files and index.
        output_dir: Directory to save the reordered and resharded model.
        num_shards: The desired number of output shards.
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    input_index_path = input_path / "model.safetensors.index.json"
    output_index_path = output_path / "model.safetensors.index.json"

    if not input_path.is_dir():
        print(f"Error: Input directory not found at {input_path}")
        return
    if not input_index_path.is_file():
        # Allow proceeding without index, but warn, as we'll load all .safetensors files directly
        print(f"Warning: Index file not found at {input_index_path}. Loading all *.safetensors files.")
        # We can reconstruct the weight map by loading all tensors

    if num_shards <= 0:
        print(f"Error: Number of shards must be positive.")
        return

    output_path.mkdir(parents=True, exist_ok=True)

    print(f"Loading all tensors from {input_path}...")
    all_tensors = {}
    tensor_sizes = {}
    total_size = 0

    # Load tensors directly from all shard files
    input_shard_files = sorted(list(input_path.glob("*.safetensors")))
    if not input_shard_files:
        print(f"Error: No .safetensors files found in {input_path}")
        return

    for shard_file in tqdm(input_shard_files, desc="Loading input shards"):
        try:
            with safe_open(shard_file, framework="pt", device="cpu") as f:
                for key in f.keys():
                    if key in all_tensors:
                        print(f"Warning: Duplicate tensor key '{key}' found. Overwriting.")
                    tensor = f.get_tensor(key)
                    # Ensure contiguous for accurate size calculation
                    if not tensor.is_contiguous():
                         tensor = tensor.contiguous()
                    all_tensors[key] = tensor
                    size = tensor.numel() * tensor.element_size()
                    tensor_sizes[key] = size
                    total_size += size
        except Exception as e:
            print(f"Error loading shard {shard_file}: {e}")
            return # Stop if a shard fails to load

    print(f"Loaded {len(all_tensors)} tensors. Total size: {total_size / (1024**3):.2f} GB")

    # --- Sort Tensors ---
    print("Sorting tensors according to transformer architecture...")
    sorted_tensor_names = sorted(all_tensors.keys(), key=transformer_sort_key)

    # --- Assign Tensors to New Shards ---
    target_shard_size = total_size / num_shards
    print(f"Target size per shard: {target_shard_size / (1024**2):.2f} MB")

    shard_assignments = [[] for _ in range(num_shards)]
    shard_sizes = [0] * num_shards
    # Removed tensor_to_new_shard_map, will build final_weight_map during saving loop
    current_shard_idx = 0

    # Pre-calculate index of the last shard for easier checking
    last_shard_idx = num_shards - 1

    for name in tqdm(sorted_tensor_names, desc="Assigning tensors to new shards"):
        size = tensor_sizes[name]

        # Check if the current shard is not empty, if adding the current tensor
        # would exceed the target size, and if we are not already forced onto the last shard.
        if (shard_sizes[current_shard_idx] > 0 and
            shard_sizes[current_shard_idx] + size > target_shard_size and
            current_shard_idx < last_shard_idx):
            # Move to the next shard *before* assigning the current tensor
            current_shard_idx += 1

        # Assign the tensor to the determined shard (current or the new one)
        shard_assignments[current_shard_idx].append(name)
        shard_sizes[current_shard_idx] += size
        # final_weight_map will be built during the saving loop based on final filenames

    # --- Save New Shards ---
    print(f"Saving reordered shards to {output_path} (aiming for {num_shards} shards initially)...")
    final_weight_map = {} # Maps tensor name to the final shard filename
    num_actual_shards = 0 # Count shards that actually get tensors
    # Store paths before renaming, mapping original index 'i' to temporary path
    temp_saved_shard_paths = {}

    for i in range(num_shards):
        shard_tensor_names = shard_assignments[i]
        if not shard_tensor_names:
            # This shard ended up empty, skip it entirely
            # Do not increment num_actual_shards here
            print(f"Note: Shard index {i+1} has no tensors assigned. Skipping generation.")
            continue

        # Only increment count and proceed if shard is non-empty
        num_actual_shards += 1
        shard_state_dict = {name: all_tensors[name] for name in shard_tensor_names}

        # Format shard number with leading zeros using the initial index 'i'
        shard_num_str = str(i + 1).zfill(5)
        # Use the initially requested num_shards for the placeholder name
        total_shards_str_placeholder = str(num_shards).zfill(5)
        safetensors_filename = f"model-{shard_num_str}-of-{total_shards_str_placeholder}.safetensors"
        safetensors_path = output_path / safetensors_filename
        temp_saved_shard_paths[i] = safetensors_path # Store temp path against original index

        actual_shard_size_mb = sum(tensor_sizes[name] for name in shard_tensor_names) / (1024**2)
        # Refer to the original index 'i' + 1 for user-facing message clarity
        print(f"  Saving shard index {i+1} ({len(shard_tensor_names)} tensors, {actual_shard_size_mb:.2f} MB) temporarily as {safetensors_filename}...")

        # Temporarily map tensor names to the placeholder filename
        # This map isn't strictly needed here anymore, but can be useful for debugging
        # for name in shard_tensor_names:
        #      final_weight_map[name] = safetensors_filename # Placeholder

        try:
            save_file(shard_state_dict, safetensors_path)
        except Exception as e:
             print(f"Error saving shard {safetensors_filename}: {e}")
             # Consider cleaning up partially created files on error
             return # Stop processing

        # Clean up memory for the shard
        del shard_state_dict

    # --- Refine shard filenames and build final weight map based on actual shards created ---
    print(f"Actually saved {num_actual_shards} non-empty shards. Renaming files...")
    final_shard_filenames = []
    refined_weight_map = {} # This will be the final map for the index
    current_actual_shard_index = 1 # 1-based index for the final filenames

    # Iterate through the original indices (0 to num_shards-1)
    # Only process indices that correspond to saved shards (keys in temp_saved_shard_paths)
    saved_indices = sorted(temp_saved_shard_paths.keys())

    for original_index in saved_indices: # original_index is 'i' from the saving loop
        old_path = temp_saved_shard_paths[original_index]
        original_filename = old_path.name # Get the placeholder filename like model-0000i-of-000NN.safetensors

        # Create the new filename using the actual count of saved shards
        new_shard_num_str = str(current_actual_shard_index).zfill(5)
        new_total_shards_str = str(num_actual_shards).zfill(5) # Use actual count
        new_filename = f"model-{new_shard_num_str}-of-{new_total_shards_str}.safetensors"
        new_path = output_path / new_filename

        # Rename the file
        if old_path.exists():
            if old_path != new_path:
                 print(f"  Renaming {original_filename} to {new_filename}")
                 old_path.rename(new_path)
            else:
                # This happens if num_actual_shards == num_shards and shard 'i' was the i-th actual shard
                print(f"  File {new_filename} already has the correct name.")
        elif not old_path.exists() and old_path == new_path:
             # Handle case where file might already exist with the correct name if script was interrupted/rerun
             print(f"  File {new_filename} already exists with correct name (possibly from previous run).")
        else: # File expected but not found
             print(f"  Warning: Expected temporary shard file {original_filename} not found at {old_path} for renaming.")
             continue # Skip this shard if the temp file is missing


        final_shard_filenames.append(new_filename)

        # Update the refined weight map: map tensors from this shard to the *new* final filename
        # Use shard_assignments[original_index] to get the list of tensor names
        for tensor_name in shard_assignments[original_index]:
             refined_weight_map[tensor_name] = new_filename

        current_actual_shard_index += 1 # Increment for the next actual shard

    # --- Create Sorted Index File ---
    print("Creating final sorted index file...")
    # Ensure the index map uses the sorted tensor names
    # The refined_weight_map should already contain all tensors mapped to their (potentially renamed) files
    # We just need to order it correctly.
    index_weight_map_sorted = {
        name: refined_weight_map[name]
        for name in sorted_tensor_names
        if name in refined_weight_map # Ensure tensor was actually assigned and saved
    }

    new_index_data = {
        "metadata": {
            "total_size": total_size,
            "num_shards_requested": num_shards,
            "num_shards_actual": num_actual_shards
        },
        "weight_map": index_weight_map_sorted
    }

    print(f"Saving final index to {output_index_path}...")
    with open(output_index_path, 'w') as f:
        json.dump(new_index_data, f, indent=2) # Use indent=2 for readability

    # --- Copy Auxiliary Files ---
    print(f"Copying auxiliary files from {input_path} to {output_path}...")
    files_to_copy = [
        "config.json",
        "generation_config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "tokenizer.model", # For SentencePiece
    ]
    for filename in files_to_copy:
        source_file = input_path / filename
        dest_file = output_path / filename
        if source_file.is_file():
            try:
                 shutil.copyfile(source_file, dest_file)
                 print(f"  Copied {filename}")
            except Exception as e:
                 print(f"  Error copying {filename}: {e}")
        else:
            print(f"  Skipping {filename} (not found in source)")


    # --- Verification (Optional) ---
    if len(index_weight_map_sorted) != len(all_tensors):
         missing_tensors = set(all_tensors.keys()) - set(index_weight_map_sorted.keys())
         print(f"Warning: Mismatch between loaded tensors ({len(all_tensors)}) and tensors in final index ({len(index_weight_map_sorted)}).")
         if missing_tensors:
              print(f"  Missing tensors from index: {missing_tensors}")


    print("Resharding and reordering complete.")
    print(f"Model saved to: {output_path}")
    print(f"Number of shards created: {num_actual_shards}")

# --- Command Line Interface ---

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Reorder and reshard SafeTensors model files based on transformer architecture."
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        required=True,
        help="Directory containing the input model.safetensors shards and optionally index.json.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Directory to save the reordered and resharded .safetensors shards and new index.json.",
    )
    parser.add_argument(
        "--num-shards",
        type=int,
        required=True,
        help="The desired number of output .safetensors shards.",
    )

    args = parser.parse_args()

    reshard_and_reorder_safetensors(
        args.input_dir, args.output_dir, args.num_shards
    )

# Example Usage:
# python rewrite_reorder.py --input-dir ./maldv/badger-nu --output-dir ./maldv/badger-nu-resharded --num-shards 11 
