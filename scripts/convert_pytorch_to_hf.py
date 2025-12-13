import argparse
import json
import os
from pathlib import Path
import torch
from safetensors.torch import save_file
from tqdm import tqdm
import math # Added for shard size calculation
import re # Added for natural sorting

def convert_pytorch_to_safetensors(model_dir: str):
    """
    Converts sharded PyTorch .bin model files to .safetensors format.

    Args:
        model_dir: The directory containing the pytorch_model.bin.index.json
                   and associated .bin files.
    """
    model_path = Path(model_dir)
    index_path = model_path / "pytorch_model.bin.index.json"
    output_index_path = model_path / "model.safetensors.index.json"

    if not index_path.is_file():
        print(f"Error: Index file not found at {index_path}")
        return

    print(f"Loading index from {index_path}...")
    with open(index_path, 'r') as f:
        index_data = json.load(f)

    weight_map = index_data.get("weight_map", {})
    if not weight_map:
        print("Error: Could not find 'weight_map' in the index file.")
        return

    # Group tensors by their source bin file
    bin_files = {}
    for tensor_name, bin_filename in weight_map.items():
        if bin_filename not in bin_files:
            bin_files[bin_filename] = []
        bin_files[bin_filename].append(tensor_name)

    new_weight_map = {}
    conversion_map = {} # Maps old bin names to new safetensors names

    print(f"Found {len(bin_files)} shard(s) to convert.")

    # Process each bin file
    for bin_filename in tqdm(sorted(bin_files.keys()), desc="Converting shards"):
        bin_path = model_path / bin_filename
        # Construct the new safetensors filename
        parts = bin_filename.split('.')
        if parts[0] == 'pytorch_model':
             # Standard HF naming convention model-0000x-of-0000y.safetensors
            safetensors_filename = f"model-{parts[0].split('-')[-2]}-of-{parts[0].split('-')[-1]}.safetensors"

        else:
             # Fallback for non-standard names, replace extension
            safetensors_filename = f"{'.'.join(parts[:-1])}.safetensors"


        safetensors_path = model_path / safetensors_filename
        conversion_map[bin_filename] = safetensors_filename

        if not bin_path.is_file():
            print(f"Warning: Skipping shard. File not found: {bin_path}")
            # Add placeholders to new weight map so index creation doesn't fail
            for tensor_name in bin_files[bin_filename]:
                 new_weight_map[tensor_name] = safetensors_filename
            continue

        # Load the state dict from the .bin file
        # Use map_location='cpu' to avoid GPU memory usage if not needed
        state_dict = torch.load(bin_path, map_location='cpu')

        # Save the state dict to .safetensors
        # Safetensors automatically handles metadata if it exists in state_dict
        # We only save the tensors listed for this shard in the index
        shard_state_dict = {k: state_dict[k] for k in bin_files[bin_filename] if k in state_dict}

        # Add tensors that might not be in the index but are in the file (e.g., optimizer states)
        # This might not be desired, depending on the use case. Comment out if needed.
        # for k, v in state_dict.items():
        #     if k not in shard_state_dict:
        #         shard_state_dict[k] = v

        save_file(shard_state_dict, safetensors_path)

        # Update the new weight map
        for tensor_name in shard_state_dict.keys(): # Use keys from actual saved dict
             new_weight_map[tensor_name] = safetensors_filename

        # Optional: Clean up memory
        del state_dict
        del shard_state_dict
        # torch.cuda.empty_cache() # If tensors were loaded to GPU

    # Create the new index file data
    new_index_data = {
        "metadata": index_data.get("metadata", {}), # Copy metadata
        "weight_map": new_weight_map
    }

    print(f"Saving new index to {output_index_path}...")
    with open(output_index_path, 'w') as f:
        json.dump(new_index_data, f, indent=2)

    print("Conversion complete.")
    print(f"Original .bin files are still present in {model_path}")


def transformer_sort_key(tensor_name):
    """Provides a sort key for tensor names based on typical transformer architecture.

    Sorts layers numerically and orders components within layers logically:
    Input LayerNorm -> Self-Attention (Q, K, V, O) -> Post-Attention LayerNorm -> MLP (Gate, Up, Down).
    Embeddings come first, final LayerNorm and LM Head come last.
    """
    # Component Priorities (lower number = earlier)
    COMPONENT_ORDER = {
        "input_layernorm": 0,
        "self_attn.q_proj": 1,
        "self_attn.k_proj": 2,
        "self_attn.v_proj": 3,
        "self_attn.o_proj": 4,
        "post_attention_layernorm": 5,
        "mlp.gate_proj": 6,
        "mlp.up_proj": 7,
        "mlp.down_proj": 8,
    }
    # Layer Type Priorities
    LAYER_TYPE_ORDER = {
        "model.embed_tokens": 0,
        "model.layers": 1,
        "model.norm": 2,
        "lm_head": 3,
    }
    MAX_LAYER_TYPE = max(LAYER_TYPE_ORDER.values()) + 1
    MAX_COMPONENT = max(COMPONENT_ORDER.values()) + 1

    parts = tensor_name.split('.')

    layer_type_priority = MAX_LAYER_TYPE
    layer_num = -1
    component_priority = MAX_COMPONENT
    sub_component_name = ""

    if parts[0] == "model":
        if parts[1] == "embed_tokens":
            layer_type_priority = LAYER_TYPE_ORDER["model.embed_tokens"]
        elif parts[1] == "layers" and len(parts) > 2 and parts[2].isdigit():
            layer_type_priority = LAYER_TYPE_ORDER["model.layers"]
            layer_num = int(parts[2])
            # Combine relevant parts to match COMPONENT_ORDER keys
            component_key = ".".join(parts[3:-1]) # e.g., "self_attn.q_proj"
            if component_key in COMPONENT_ORDER:
                 component_priority = COMPONENT_ORDER[component_key]
            else:
                # Fallback for unknown components within a layer
                sub_component_name = ".".join(parts[3:])
        elif parts[1] == "norm":
            layer_type_priority = LAYER_TYPE_ORDER["model.norm"]
    elif parts[0] == "lm_head":
         layer_type_priority = LAYER_TYPE_ORDER["lm_head"]

    # Return a tuple for sorting:
    # (Layer Type, Layer Num, Component Type, SubComponent Name/Qualifier (e.g. weight), Full Name)
    # Adding the last element (tensor_name) ensures stable sort if all other keys are equal
    qualifier = parts[-1] if len(parts) > 1 else ""
    return (layer_type_priority, layer_num, component_priority, sub_component_name, qualifier, tensor_name)


def convert_single_bin_to_sharded_safetensors(input_bin_path: str, output_dir: str, num_shards: int):
    """
    Converts a single PyTorch .bin model file to sharded .safetensors format,
    assigning tensors to shards sequentially based on transformer architecture order,
    and ensuring the output index.json is also sorted accordingly.

    Args:
        input_bin_path: Path to the input .bin file.
        output_dir: Directory to save the .safetensors shards and index file.
        num_shards: The number of shards to create.
    """
    input_path = Path(input_bin_path)
    output_path = Path(output_dir)

    if not input_path.is_file():
        print(f"Error: Input bin file not found at {input_path}")
        return

    if num_shards <= 0:
        print(f"Error: Number of shards must be positive.")
        return

    output_path.mkdir(parents=True, exist_ok=True)
    output_index_path = output_path / "model.safetensors.index.json"

    print(f"Loading model state_dict from {input_path}...")
    # Load the state dict onto CPU to avoid high GPU memory usage
    state_dict = torch.load(input_path, map_location='cpu')
    print(f"Loaded {len(state_dict)} tensors.")

    # Calculate total size and size per tensor
    total_size = 0
    tensor_sizes = {}
    for name, tensor in state_dict.items():
        # Ensure tensor is contiguous before checking size
        # Some formats might store non-contiguous tensors
        if not tensor.is_contiguous():
             tensor = tensor.contiguous()
             state_dict[name] = tensor # Update state_dict if we made it contiguous
        size = tensor.numel() * tensor.element_size()
        tensor_sizes[name] = size
        total_size += size

    print(f"Total model size: {total_size / (1024**3):.2f} GB")

    # Sort tensors using the transformer-specific key for ordering.
    sorted_tensor_names = sorted(state_dict.keys(), key=transformer_sort_key)

    # Assign tensors to shards sequentially based on the sorted order.
    target_shard_size = total_size / num_shards
    print(f"Target size per shard: {target_shard_size / (1024**2):.2f} MB")

    shard_assignments = [[] for _ in range(num_shards)]
    shard_sizes = [0] * num_shards
    tensor_to_shard_map = {}
    current_shard_idx = 0

    for name in tqdm(sorted_tensor_names, desc="Assigning tensors to shards"):
        # Add tensor to the current shard
        shard_assignments[current_shard_idx].append(name)
        shard_sizes[current_shard_idx] += tensor_sizes[name]
        tensor_to_shard_map[name] = current_shard_idx # Store assignment index

        # Move to the next shard if the current one exceeds the target size
        # and we are not already on the last shard.
        # Use a small tolerance (e.g., 1.05) or check if adding the *next* tensor would exceed
        # to prevent creating too many tiny shards. A simple check is often sufficient.
        if shard_sizes[current_shard_idx] >= target_shard_size and current_shard_idx < num_shards - 1:
             # Optional: Add a check here to see if the *next* tensor would also fit
             # without exceeding the target size by too much, to potentially keep layers together.
             # For simplicity, we just switch when the target is met.
            current_shard_idx += 1

    shard_filenames = []

    print(f"Saving {num_shards} shards to {output_path} based on sequential architectural order...")
    for i in range(num_shards):
        shard_tensors_names = shard_assignments[i]
        if not shard_tensors_names:
            # This might happen if num_shards is very large compared to tensor count/size
            print(f"Warning: Shard {i+1} has no tensors assigned. Skipping generation.")
            continue

        # Re-fetch tensors for this shard to ensure contiguity if changed earlier
        shard_state_dict = {name: state_dict[name] for name in shard_tensors_names}

        # Format shard number with leading zeros if needed (e.g., 00001)
        shard_num_str = str(i + 1).zfill(5)
        total_shards_str = str(num_shards).zfill(5)
        safetensors_filename = f"model-{shard_num_str}-of-{total_shards_str}.safetensors"
        safetensors_path = output_path / safetensors_filename
        shard_filenames.append(safetensors_filename)

        actual_shard_size_mb = sum(tensor_sizes[name] for name in shard_tensors_names) / (1024**2)
        print(f"  Saving shard {i+1}/{num_shards} ({len(shard_tensors_names)} tensors, {actual_shard_size_mb:.2f} MB) to {safetensors_filename}...")

        # Update tensor_to_shard_map to store the actual filename *after* deciding it
        for name in shard_tensors_names:
             tensor_to_shard_map[name] = safetensors_filename # Map name to filename

        try:
            save_file(shard_state_dict, safetensors_path)
        except Exception as e:
             print(f"\nError saving shard {safetensors_filename}: {e}")
             # Optionally, try saving tensor by tensor for debugging
             # for name, tensor in shard_state_dict.items():
             #     try:
             #         save_file({name: tensor}, output_path / f"DEBUG_{name}.safetensor")
             #     except Exception as inner_e:
             #         print(f"    Failed on tensor: {name}, Size: {tensor.shape}, Dtype: {tensor.dtype}, Error: {inner_e}")
             # return # Stop processing if a shard fails

        # Clean up memory for the shard
        del shard_state_dict

    # Create the final weight map, ensuring it is sorted by transformer architecture
    # The tensor_to_shard_map should now correctly reflect the sequential assignment
    print("Sorting weight map for index file according to transformer architecture...")
    # Sort the final map based on the initial transformer-sorted tensor names
    final_weight_map = {name: tensor_to_shard_map[name] for name in sorted_tensor_names if name in tensor_to_shard_map}

    # Create the index file data
    new_index_data = {
        "metadata": {
             "total_size": total_size # Add total size metadata
        },
        "weight_map": final_weight_map # Use the transformer-sorted map
    }

    print(f"Saving transformer-sorted index to {output_index_path}...")
    with open(output_index_path, 'w') as f:
        json.dump(new_index_data, f, indent=2, ensure_ascii=True)

    # Optional: Verify all original tensors were saved and mapped
    if len(final_weight_map) != len(state_dict):
         missing_tensors = set(state_dict.keys()) - set(final_weight_map.keys())
         print(f"Warning: Number of tensors in index ({len(final_weight_map)}) does not match original state_dict ({len(state_dict)}).")
         if missing_tensors:
              print(f"  Missing tensors: {missing_tensors}")
         else:
              # This case might indicate duplicate tensors or an issue in mapping
              print("  All original tensors seem present but count mismatch occurred.")


    print("Conversion complete.")
    num_saved_shards = len([s for s in shard_assignments if s]) # Count non-empty shards saved
    print(f"Saved {num_saved_shards} shards and transformer-sorted index file to {output_path}")
    print(f"Original .bin file at {input_path} was not modified.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert a single PyTorch .bin model to architecturally ordered, sharded .safetensors format."
    )
    parser.add_argument(
        "--input-bin",
        type=str,
        required=True,
        help="Path to the input PyTorch .bin model file.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Directory to save the output .safetensors shards and index.json.",
    )
    parser.add_argument(
        "--num-shards",
        type=int,
        required=True,
        help="The desired number of .safetensors shards to create.",
    )

    args = parser.parse_args()

    convert_single_bin_to_sharded_safetensors(
        args.input_bin, args.output_dir, args.num_shards
    )


# Example Usage:
# python convert_pytorch_to_hf.py --input-bin /path/to/pytorch_model.bin --output-dir /path/to/output_safetensors --num-shards 4
