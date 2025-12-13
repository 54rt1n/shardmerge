# shard/writer.py
# Copyright (C) 2024 Martin Bukowski
#
# This software is free software: you can redistribute it and/or
# modify it under the terms of the GNU Lesser General Public License as
# published by the Free Software Foundation, either version 3 of the
# License, or (at your option) any later version.
#
# This software is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
# Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with this program. If not, see http://www.gnu.org/licenses/.

from dataclasses import dataclass, field
from typing import Optional, Dict, Set, List, Tuple, Generator
import json
import logging
from pathlib import Path
import torch
from safetensors.torch import save_file
from safetensors import safe_open
from huggingface_hub import snapshot_download

from .constants import INPUT_LAYER, OUTPUT_LAYER

logger = logging.getLogger(__name__)


@dataclass
class ShardLayer:
    layer_order_idx: int
    shard_name: str
    layer_name: str
    written: bool

    @property
    def layer_number(self) -> int:
        # Models are named like "model.layers.0..."
        # Plus we have the input and output layers:
        # Inputs = "model.embed_tokens.weight"
        # Ouptuts "model.norm.weight", "lm_head.weight", 
        if self.layer_name.startswith("model.embed_tokens.weight"):
            return INPUT_LAYER
        if self.layer_name.startswith("model.norm.weight") or self.layer_name.startswith("lm_head.weight"):
            return OUTPUT_LAYER
        if self.layer_name.startswith("model.layers."):
            splits = self.layer_name.split(".")
            parsed = int(splits[2])
            if str(parsed) == splits[2]:
                return parsed
            else:
                raise ValueError(f"Unknown layer name: {self.layer_name}")
        else:
            raise ValueError(f"Unknown layer name: {self.layer_name}")
    
    
@dataclass
class ModelWriter:
    """Manages writing merged tensors into safetensor files that mirror base model structure"""
    base_index: dict  # Raw index.json from base model
    output_path: Path
    layer_order: list[str]
    output_astype: torch.dtype
    written_shard_layers: Set[tuple[str, str]] = field(default_factory=set)
    shard_to_tensors: Dict[str, Set[str]] = field(default_factory=dict)
    
    def __post_init__(self):
        """Initialize the writer and create necessary mappings"""
        self.output_path.mkdir(parents=True, exist_ok=True)

        # Create copy of index for output model
        self.index_path = self.output_path / "model.safetensors.index.json"
        if not self.index_path.exists():
            with open(self.index_path, "w") as f:
                json.dump(self.base_index, f, indent=2)
        else:
            logger.info(f"Index already exists: {self.index_path}")
            self.base_index = json.load(open(self.index_path))

        # Create mapping of shards to their tensor names
        self.shard_to_tensors = {}
        for tensor_name, shard_name in self.base_index["weight_map"].items():
            if shard_name not in self.shard_to_tensors:
                self.shard_to_tensors[shard_name] = set()
            self.shard_to_tensors[shard_name].add(tensor_name)
        
        # Check for existing shards
        self._check_existing_shards()

    def _check_existing_shards(self):
        """Check for already written shards and validate them"""
        for shard_name, tensor_names in self.shard_to_tensors.items():
            shard_path = self.output_path / shard_name
            missing_tensors = tensor_names.copy()
            if shard_path.exists():
                try:
                    # Verify shard contains all expected tensors
                    with safe_open(shard_path, framework="pt") as f:
                        for layer in f.keys():
                            if layer not in missing_tensors:
                                raise ValueError(f"Tensor {layer} found in {shard_path} but not in base model")
                            missing_tensors.remove(layer)
                            self.written_shard_layers.add((shard_name, layer))
                    #if missing_tensors:
                    #    raise ValueError(f"Missing tensors in {shard_path}: {missing_tensors}")
                except Exception as e:
                    logger.error(f"Error validating shard {shard_name}: {e}")
                    #if shard_path.exists():
                    #    shard_path.unlink()
                    raise e

    def add_tensor(self, layer_name: str, tensor: torch.Tensor):
        """Add a tensor by reading existing shard, updating it, and writing it back"""
        shard_name = self.base_index["weight_map"][layer_name]
        shard_path = self.output_path / shard_name

        if (shard_name, layer_name) in self.written_shard_layers:
            logger.info(f"Skipping {layer_name} as it's already in written shard {shard_name}")
            return

        # Read existing tensors from shard if it exists
        existing_tensors = {}
        if shard_path.exists():
            from safetensors import safe_open
            with safe_open(shard_path, framework="pt") as f:
                for existing_layer in f.keys():
                    existing_tensors[existing_layer] = f.get_tensor(existing_layer)

        # Add new tensor
        existing_tensors[layer_name] = tensor.clone().cpu().to(self.output_astype)

        # Create ordered dict following layer_order
        ordered_tensors = {}
        for name in self.layer_order:
            if name in existing_tensors:
                ordered_tensors[name] = existing_tensors[name]

        # Save updated shard
        try:
            save_file(ordered_tensors, str(shard_path), metadata={"format": "pt"})
            self.written_shard_layers.add((shard_name, layer_name))
            logger.info(f"Successfully wrote tensor {layer_name} to shard: {shard_name}")
        except Exception as e:
            logger.error(f"Error saving shard {shard_name}: {e}")
            if shard_path.exists():
                shard_path.unlink()

    def finalize(self):
        """Verify all shards were written completely"""
        missing_layers = []
        for shard_name, tensor_names in self.shard_to_tensors.items():
            for tensor_name in tensor_names:
                if (shard_name, tensor_name) not in self.written_shard_layers:
                    missing_layers.append((shard_name, tensor_name))

        if missing_layers:
            logger.error(f"Failed to write all layers. Missing: {missing_layers}")
            raise RuntimeError(f"Incomplete model output: missing {len(missing_layers)} layers")

    def shard_layers(self) -> Generator[List[ShardLayer], None, None]:
        """Iterate over all layers in all shards"""
        for shard_name, tensors in sorted(self.shard_to_tensors.items(), key=lambda x: x[0]):
            shard_layers = []
            logger.info(f"ShardLayers: {shard_name} - {tensors}")
            for layer_order_idx, name in sorted([(self.layer_order.index(name), name) for name in tensors], key=lambda x: x[0]):
                shard_layer = ShardLayer(layer_order_idx, shard_name, name, (shard_name, name) in self.written_shard_layers)
                layer_number = shard_layer.layer_number
                if layer_number == INPUT_LAYER:
                    meta = " Input Layer"
                elif layer_number == OUTPUT_LAYER:
                    meta = " Output Layer"
                else:
                    meta = ""

                logger.info(f"ShardLayer ({layer_order_idx}): {name} - {shard_name} - {meta}")
                shard_layers.append(shard_layer)
            yield shard_layers
            
    @classmethod
    def from_huggingface(cls, model_id: str, output_path: Path, layer_order: list[str], revision: str = "main"):
        """Initialize a ModelWriter by downloading configuration files from Hugging Face.
        
        Args:
            model_id: The Hugging Face model ID (e.g., 'facebook/opt-350m')
            output_path: Where to save the merged model
            layer_order: List of layer names in order
            revision: Model revision/tag to use
            
        Returns:
            ModelWriter instance initialized with the model's configuration
        """
        # Download only configuration files
        output_path.mkdir(parents=True, exist_ok=True)
        
        allow_patterns = ["*.json", "*.md"]
        ignore_patterns = ["*.bin", "*.safetensors", "*.msgpack"]
        
        snapshot_download(
            repo_id=model_id,
            revision=revision,
            #allow_patterns=allow_patterns,
            ignore_patterns=ignore_patterns,
            local_dir=output_path,
        )
        
        # Load the model index
        index_path = output_path / "model.safetensors.index.json"
        if not index_path.exists():
            raise FileNotFoundError(f"Model index not found at {index_path}")
            
        with open(index_path) as f:
            base_index = json.load(f)
            
        return cls(
            base_index=base_index,
            output_path=output_path,
            layer_order=layer_order
        )

    @classmethod
    def like_model(cls, model_path: Path, output_path: Path, output_astype: torch.dtype = torch.bfloat16):
        """Initalize a ModelWriter by reading a model and it's model.safetensors.index.json"""
        index_path = model_path / "model.safetensors.index.json"
        if not index_path.exists():
            raise FileNotFoundError(f"Model index not found at {index_path}")
            
        with open(index_path) as f:
            base_index = json.load(f)
        
        # For each file in the model path, read the file in order and use it to determine the layer order
        layer_order = []
        for file in model_path.glob("*.safetensors"):
            with safe_open(file, framework="pt") as f:
                for layer in f.keys():
                    layer_order.append(layer)
            
        return cls(
            base_index=base_index,
            output_path=output_path,
            layer_order=layer_order,
            output_astype=output_astype
        )
            