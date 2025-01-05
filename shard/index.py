# shard/index.py
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

from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple
import os
import json
import logging
import asyncio
import aiohttp
from pathlib import Path
import torch
from .download import DownloadManager, DownloadStatus

logger = logging.getLogger(__name__)

@dataclass
class ModelShard:
    """Represents a model weight shard and its metadata"""
    filename: str
    contained_keys: List[str]
    weight_map: Dict[str, str]
    local_path: Optional[Path] = None

class TensorPromise:
    """A promise for an eventual tensor"""
    def __init__(self, model_uri: str, tensor_name: str, device: str):
        self.model_uri = model_uri
        self.tensor_name = tensor_name
        self.device = device
        self._future: asyncio.Future[torch.Tensor] = asyncio.Future()

    async def get(self) -> torch.Tensor:
        """Await and return the tensor"""
        return (await self._future).to(self.device)

    def set_result(self, tensor: torch.Tensor):
        """Set the result tensor"""
        if not self._future.done():
            self._future.set_result(tensor)

    def set_exception(self, exc: Exception):
        """Set an exception if loading failed"""
        if not self._future.done():
            self._future.set_exception(exc)

class HFMultiModelIndex:
    """Manages index and weight file mappings for multiple HuggingFace models"""
    
    def __init__(
        self,
        download_manager: Optional[DownloadManager] = None,
        cache_path: Optional[Path] = None
    ):
        self.download_manager = download_manager
        
        # Default cache directory if none provided
        if cache_path:
            self.cache_path = cache_path
        else:
            self.cache_path = Path.home() / ".cache" / "shardmerge"
        self.cache_path.mkdir(parents=True, exist_ok=True)
        
        self.model_indexes: Dict[str, Dict] = {}
        self.model_shards: Dict[str, Dict[str, ModelShard]] = {}
        self._tensor_downloads: Dict[Tuple[str, str], torch.Tensor] = {}
        self._ordered_weights: Dict[str, List[str]] = {}
        
    async def add_model(self, model_uri: str, revision: str = "main"):
        """Add a model to the index"""
        if model_uri in self.model_indexes:
            return

        # Check and see if the model is already in storage:
        model_path = self.download_manager.storage_path / model_uri
        model_index_path = self.download_manager.storage_path / model_uri / "model.safetensors.index.json"
        model_path.mkdir(parents=True, exist_ok=True)
        
        if model_index_path.exists():
            logger.info(f"Model {model_uri} already in storage, loading from {model_index_path}")
            with open(model_index_path, "r") as f:
                index = json.load(f)
        else:
            # Fetch and parse index
            index_url = f"https://huggingface.co/{model_uri}/raw/{revision}/model.safetensors.index.json"
            async with aiohttp.ClientSession() as session:
                async with session.get(index_url) as response:
                    response.raise_for_status()
                    logger.info(f"Fetched index for model {model_uri}")
                    # Get text content and parse as JSON regardless of content type
                    text = await response.text()
                    with open(model_index_path, "w") as f:
                        f.write(text)
                    index = json.loads(text)
        
        self.model_indexes[model_uri] = index
        
        # Initialize shard mapping for this model
        shard_contents: Dict[str, List[str]] = {}
        for tensor_name, shard_file in index["weight_map"].items():
            if shard_file not in shard_contents:
                shard_contents[shard_file] = []
            shard_contents[shard_file].append(tensor_name)
            
        # Create ModelShard objects
        self.model_shards[model_uri] = {}
        for shard_file, tensor_keys in shard_contents.items():
            self.model_shards[model_uri][shard_file] = ModelShard(
                filename=shard_file,
                contained_keys=tensor_keys,
                weight_map={k: shard_file for k in tensor_keys},
                local_path=None
            )
            
        # Cache ordered weights for this model
        self._ordered_weights[model_uri] = self._get_ordered_weights(model_uri)
        logger.info(f"Initialized {len(shard_contents)} shards for model {model_uri}")

    def _get_ordered_weights(self, model_uri: str) -> List[str]:
        """Get weight keys in their exact order they appear in model shards"""
        logger.info(f"Getting ordered weights for model {model_uri}")
        if model_uri not in self.model_indexes:
            raise KeyError(f"Model {model_uri} not found in index")
            
        index = self.model_indexes[model_uri]
        
        # Define standard layer component order
        layer_component_order = [
            "input_layernorm.weight",
            "mlp.down_proj.weight", 
            "mlp.gate_proj.weight",
            "mlp.up_proj.weight",
            "post_attention_layernorm.weight",
            "self_attn.q_proj.bias",
            "self_attn.q_proj.weight",
            "self_attn.k_proj.bias",
            "self_attn.k_proj.weight",
            "self_attn.v_proj.bias",
            "self_attn.v_proj.weight",
            "self_attn.o_proj.weight",
        ]
        
        # Get all weight keys
        weights = [k for k in index["weight_map"].keys()]
        
        # Separate special weights and layer weights
        embed_weights = [w for w in weights if "embed_tokens" in w]
        layer_weights = [w for w in weights if "layers." in w]
        norm_weights = [w for w in weights if "model.norm.weight" in w]
        lm_head_weights = [w for w in weights if "lm_head" in w]
        other_weights = [w for w in weights if w not in (
            embed_weights + layer_weights + norm_weights + lm_head_weights
        )]
        
        # Sort layer weights
        sorted_layer_weights = []
        layer_nums = sorted(set(
            int(w.split("layers.")[1].split(".")[0]) 
            for w in layer_weights
        ))
        
        for layer_num in layer_nums:
            layer_prefix = f"model.layers.{layer_num}."
            for component in layer_component_order:
                weight_key = layer_prefix + component
                if weight_key in layer_weights:
                    sorted_layer_weights.append(weight_key)
                else:
                    logger.warning(f"Weight {weight_key} not found in layer {layer_num}: {layer_weights}")
                    raise Exception("Stop")
        
        # Combine all weights in order
        ordered_weights = (
            embed_weights + 
            sorted_layer_weights + 
            norm_weights + 
            lm_head_weights + 
            other_weights
        )

        logger.info(f"Ordered weights: {ordered_weights}")
        
        # Verify we haven't lost any weights
        ordered_weights_set = set(ordered_weights)
        weights_set = set(weights)
        if ordered_weights_set != weights_set:
            raise ValueError(f"Weight ordering lost some weights!: {weights_set - ordered_weights_set}")
        return ordered_weights

    def get_layer_order(self, model_uri: str) -> List[str]:
        """Get weight keys in their proper order for a model"""
        if model_uri not in self._ordered_weights:
            raise KeyError(f"Model {model_uri} not found in index")
        return self._ordered_weights[model_uri].copy()

    def get_tensor(self, model_uri: str, tensor_name: str, device: str = "cpu") -> TensorPromise:
        """Get a promise for a tensor by name from a specific model"""
        if model_uri not in self.model_indexes:
            raise KeyError(f"Model {model_uri} not found in index")
            
        index = self.model_indexes[model_uri]
        if tensor_name not in index["weight_map"]:
            raise KeyError(f"Tensor {tensor_name} not found in model {model_uri}")

        logger.debug(f"Fetching tensor {tensor_name} from model {model_uri} {device}")
            
        # Create promise for this tensor
        promise = TensorPromise(model_uri, tensor_name, device)
        
        # Check if tensor is already downloaded
        tensor_key = (model_uri, tensor_name)
        if tensor_key in self._tensor_downloads:
            promise.set_result(self._tensor_downloads[tensor_key].to(device))
            return promise

        # Get shard info
        shard_name = index["weight_map"][tensor_name]
        shard_key = (model_uri, shard_name)

        # Start async loading
        asyncio.create_task(self._load_tensor(promise, shard_key))
        return promise

    async def preload_tensor(self, model_uri: str, tensor_name: str):
        """Async task to load a tensor and fulfill its promise"""
        try:
            index = self.model_indexes[model_uri]
            if tensor_name not in index["weight_map"]:
                raise KeyError(f"Tensor {tensor_name} not found in model {model_uri}")

            shard_name = index["weight_map"][tensor_name]
            
            # Start download
            shard_url = f"https://huggingface.co/{model_uri}/resolve/main/{shard_name}?download=true"
            await self.download_manager.cache_file(model_uri, shard_url, no_claims=-1)
        except Exception as e:
            logger.exception(f"Failed to preload tensor {tensor_name} from {model_uri}")

    async def _load_tensor(self, promise: TensorPromise, shard_key: Tuple[str, str]):
        """Async task to load a tensor and fulfill its promise"""
        try:
            model_uri, shard_name = shard_key
            shard = self.model_shards[model_uri][shard_name]
            # Count tensors in shard for proper claim count
            tensors_in_shard = len(shard.contained_keys)
            
            # Start download
            shard_url = f"https://huggingface.co/{model_uri}/resolve/main/{shard_name}?download=true"
            await self.download_manager.cache_file(model_uri, shard_url, no_claims=tensors_in_shard)

            task = self.download_manager.downloads[(model_uri, shard_url)]

            async with task.lock:
                # Get path once download completes
                #logger.debug(f"Shard {shard_name} downloaded")
                path = await self.download_manager.get_file(model_uri, shard_url, claim=True)
                if not path:
                    raise RuntimeError(f"Failed to get shard {shard_name}")
                shard.local_path = path

            # Load tensor from shard
            from safetensors import safe_open
            with safe_open(shard.local_path, framework="pt") as f:
                tensor = f.get_tensor(promise.tensor_name)
                tensor_key = (model_uri, promise.tensor_name)
                self._tensor_downloads[tensor_key] = tensor
                promise.set_result(tensor.to(promise.device))

        except Exception as e:
            logger.exception(f"Failed to load tensor {promise.tensor_name} from {shard_key[0]}/{shard_key[1]}")
            promise.set_exception(e)

    def get_model_keys(self, model_uri: str) -> Set[str]:
        """Get all available tensor names for a model"""
        if model_uri not in self.model_indexes:
            raise KeyError(f"Model {model_uri} not found in index")
        return set(self.model_indexes[model_uri]["weight_map"].keys())