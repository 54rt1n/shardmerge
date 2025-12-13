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

        # Get all weight keys
        weights = list(index["weight_map"].keys())

        # Separate special weights and layer weights
        embed_weights = sorted([w for w in weights if "embed_tokens" in w])
        layer_weights = [w for w in weights if "layers." in w]
        norm_weights = sorted([w for w in weights if "model.norm.weight" in w])
        lm_head_weights = sorted([w for w in weights if "lm_head" in w])
        other_weights = sorted([w for w in weights if w not in (
            embed_weights + layer_weights + norm_weights + lm_head_weights
        )])

        # Discover layer components from layer 0
        layer_nums = sorted(set(
            int(w.split("layers.")[1].split(".")[0])
            for w in layer_weights
        ))

        layer_0_prefix = "model.layers.0."
        layer_0_weights = [w for w in layer_weights if w.startswith(layer_0_prefix)]
        components = sorted([w.replace(layer_0_prefix, "") for w in layer_0_weights])
        logger.info(f"Discovered {len(components)} layer components: {components}")

        # Sort layer weights by layer number, then by component order
        sorted_layer_weights = []
        for layer_num in layer_nums:
            layer_prefix = f"model.layers.{layer_num}."
            for component in components:
                weight_key = layer_prefix + component
                sorted_layer_weights.append(weight_key)

        # Combine all weights in order
        ordered_weights = (
            embed_weights +
            sorted_layer_weights +
            norm_weights +
            lm_head_weights +
            other_weights
        )

        # Verify we haven't lost any weights
        ordered_weights_set = set(ordered_weights)
        weights_set = set(weights)
        if ordered_weights_set != weights_set:
            missing = weights_set - ordered_weights_set
            extra = ordered_weights_set - weights_set
            raise ValueError(f"Weight ordering mismatch! Missing: {missing}, Extra: {extra}")
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

# New class for offline-only index management
class OfflineMultiModelIndex:
    """Manages index and weight file mappings for multiple HuggingFace models stored locally."""
    
    def __init__(self):
        self.model_paths: Dict[str, Path] = {} # Stores model_id -> model_directory_path
        self.model_indexes: Dict[str, Dict] = {}
        self.model_shards: Dict[str, Dict[str, ModelShard]] = {}
        self._tensor_downloads: Dict[Tuple[str, str], torch.Tensor] = {} # In-memory cache once loaded
        self._ordered_weights: Dict[str, List[str]] = {}
        logger.info("OfflineMultiModelIndex initialized.")

    def add_model(self, model_path: Path):
        """Add a model to the index from its local directory path."""
        if not model_path.is_dir():
            raise NotADirectoryError(f"Provided model path is not a directory: {model_path}")
            
        model_id = model_path.name # Use directory name as the unique ID
        if model_id in self.model_indexes:
            logger.warning(f"Model '{model_id}' already added. Skipping.")
            return

        model_index_path = model_path / "model.safetensors.index.json"
        
        if not model_index_path.exists():
            raise FileNotFoundError(f"Index file 'model.safetensors.index.json' not found in {model_path}")

        logger.info(f"Loading index for model '{model_id}' from: {model_index_path}")
        with open(model_index_path, "r") as f:
            try:
                index = json.load(f)
            except json.JSONDecodeError as e:
                raise ValueError(f"Failed to parse index file {model_index_path}: {e}")

        self.model_paths[model_id] = model_path
        self.model_indexes[model_id] = index
        
        # Initialize shard mapping for this model
        shard_contents: Dict[str, List[str]] = {}
        if "weight_map" not in index:
             raise ValueError(f"Index file for '{model_id}' missing 'weight_map' key.")
             
        for tensor_name, shard_file in index["weight_map"].items():
            if shard_file not in shard_contents:
                shard_contents[shard_file] = []
            shard_contents[shard_file].append(tensor_name)
            
        # Create ModelShard objects
        self.model_shards[model_id] = {}
        for shard_file, tensor_keys in shard_contents.items():
            # local_path is now implicitly model_path / shard_file, set during load
            self.model_shards[model_id][shard_file] = ModelShard(
                filename=shard_file,
                contained_keys=tensor_keys,
                weight_map={k: shard_file for k in tensor_keys},
                local_path=None # Will be set in _load_tensor if needed, or just used directly
            )
            
        # Cache ordered weights for this model
        try:
            self._ordered_weights[model_id] = self._get_ordered_weights(model_id)
        except Exception as e:
            logger.error(f"Failed to determine ordered weights for {model_id}: {e}. Removing model.")
            # Clean up partially added model
            del self.model_paths[model_id]
            del self.model_indexes[model_id]
            del self.model_shards[model_id]
            raise # Re-raise the exception
            
        logger.info(f"Initialized {len(shard_contents)} shards for model '{model_id}' from {model_path}")

    def _get_ordered_weights(self, model_id: str) -> List[str]:
        """Get weight keys in their exact order (best effort based on common patterns)."""
        logger.info(f"Getting ordered weights for model {model_id}")
        if model_id not in self.model_indexes:
            raise KeyError(f"Model {model_id} not found in index")

        index = self.model_indexes[model_id]

        # Get all weight keys
        weights = list(index["weight_map"].keys())

        # Separate special weights and layer weights
        embed_weights = sorted([w for w in weights if "embed_tokens" in w])
        layer_weights = [w for w in weights if "layers." in w]
        norm_weights = sorted([w for w in weights if "model.norm.weight" in w])
        lm_head_weights = sorted([w for w in weights if "lm_head" in w])
        other_weights = sorted([w for w in weights if w not in (
            embed_weights + layer_weights + norm_weights + lm_head_weights
        )])

        # Discover layer components from layer 0
        layer_nums = sorted(set(
            int(w.split("layers.")[1].split(".")[0])
            for w in layer_weights
        ))

        layer_0_prefix = "model.layers.0."
        layer_0_weights = [w for w in layer_weights if w.startswith(layer_0_prefix)]
        components = sorted([w.replace(layer_0_prefix, "") for w in layer_0_weights])
        logger.info(f"Discovered {len(components)} layer components: {components}")

        # Sort layer weights by layer number, then by component order
        sorted_layer_weights = []
        for layer_num in layer_nums:
            layer_prefix = f"model.layers.{layer_num}."
            for component in components:
                weight_key = layer_prefix + component
                sorted_layer_weights.append(weight_key)

        # Combine all weights in order
        ordered_weights = (
            embed_weights +
            sorted_layer_weights +
            norm_weights +
            lm_head_weights +
            other_weights
        )

        # Verify we haven't lost any weights
        ordered_weights_set = set(ordered_weights)
        weights_set = set(weights)
        if ordered_weights_set != weights_set:
            missing = weights_set - ordered_weights_set
            extra = ordered_weights_set - weights_set
            raise ValueError(f"Weight ordering mismatch! Missing: {missing}, Extra: {extra}")
        return ordered_weights

    def get_layer_order(self, model_id: str) -> List[str]:
        """Get weight keys in their determined order for a model."""
        if model_id not in self._ordered_weights:
            # Attempt to generate if not already done (e.g., if add_model failed during ordering)
             try:
                 self._ordered_weights[model_id] = self._get_ordered_weights(model_id)
             except Exception as e:
                 raise KeyError(f"Model {model_id} not found or ordering failed: {e}")
        return self._ordered_weights[model_id].copy()

    def get_tensor(self, model_id: str, tensor_name: str, device: str = "cpu") -> TensorPromise:
        """Get a promise for a tensor by name from a specific locally stored model."""
        # We use model_id (directory name) instead of model_uri here
        if model_id not in self.model_indexes:
            raise KeyError(f"Model ID '{model_id}' not found in index. Add the model using add_model(Path(...)) first.")
            
        index = self.model_indexes[model_id]
        if tensor_name not in index["weight_map"]:
            raise KeyError(f"Tensor '{tensor_name}' not found in model '{model_id}'")

        logger.debug(f"Requesting tensor '{tensor_name}' from model '{model_id}' on device '{device}'")
            
        # Create promise for this tensor
        # Pass model_id instead of model_uri to TensorPromise if its logic needs it
        promise = TensorPromise(model_id, tensor_name, device) 
        
        # Check if tensor is already loaded into memory cache
        tensor_key = (model_id, tensor_name)
        if tensor_key in self._tensor_downloads:
            logger.debug(f"Tensor '{tensor_name}' found in memory cache for model '{model_id}'.")
            promise.set_result(self._tensor_downloads[tensor_key].to(device))
            return promise

        # Get shard info
        shard_name = index["weight_map"][tensor_name]
        shard_key = (model_id, shard_name) # (model_id, shard_filename)

        # Start async task to load the tensor from disk
        logger.debug(f"Tensor '{tensor_name}' not in cache. Scheduling load from shard '{shard_name}' for model '{model_id}'.")
        asyncio.create_task(self._load_tensor(promise, shard_key))
        return promise

    async def _load_tensor(self, promise: TensorPromise, shard_key: Tuple[str, str]):
        """Async task to load a tensor from local disk and fulfill its promise."""
        model_id, shard_name = shard_key
        
        try:
            # Construct the full path to the shard file
            model_base_path = self.model_paths.get(model_id)
            if not model_base_path:
                 # This should not happen if add_model succeeded
                 raise RuntimeError(f"Internal error: Base path for model_id '{model_id}' not found.")
                 
            local_shard_path = model_base_path / shard_name
            
            if not local_shard_path.exists():
                raise FileNotFoundError(f"Shard file not found: {local_shard_path}")
                
            logger.debug(f"Loading tensor '{promise.tensor_name}' from local file: {local_shard_path}")

            # Load tensor from shard using safetensors
            # This operation itself is blocking disk I/O, but run within an async task
            # Consider using asyncio.to_thread for very large files if blocking becomes an issue
            from safetensors import safe_open
            
            # Using asyncio.to_thread to avoid blocking the event loop during file I/O
            def load_from_safetensors():
                with safe_open(local_shard_path, framework="pt", device="cpu") as f: # Load to CPU first
                    return f.get_tensor(promise.tensor_name)

            tensor = await asyncio.to_thread(load_from_safetensors)

            # Store in memory cache and fulfill the promise
            tensor_key = (model_id, promise.tensor_name)
            self._tensor_downloads[tensor_key] = tensor # Keep CPU copy in cache
            logger.debug(f"Successfully loaded tensor '{promise.tensor_name}' from '{shard_name}'.")
            promise.set_result(tensor.to(promise.device)) # Move to target device for the caller

        except Exception as e:
            logger.exception(f"Failed to load tensor '{promise.tensor_name}' from model '{model_id}', shard '{shard_name}' ({local_shard_path})")
            promise.set_exception(e)

    def get_model_keys(self, model_id: str) -> Set[str]:
        """Get all available tensor names for a locally stored model."""
        if model_id not in self.model_indexes:
            raise KeyError(f"Model ID '{model_id}' not found in index.")
        return set(self.model_indexes[model_id]["weight_map"].keys())

    def __contains__(self, model_id: str) -> bool:
        """Check if a model ID is present in the index."""
        return model_id in self.model_indexes

    def __len__(self) -> int:
        """Return the number of models in the index."""
        return len(self.model_indexes)

# Example Usage (conceptual):
# async def main():
#     index = OfflineMultiModelIndex()
#     try:
#         index.add_model(Path("./models/my-llama-model"))
#         index.add_model(Path("/path/to/another-model"))
#     except (FileNotFoundError, NotADirectoryError, ValueError) as e:
#         print(f"Error adding model: {e}")
#         return
        
#     if "my-llama-model" in index:
#         try:
#             tensor_promise = index.get_tensor("my-llama-model", "model.layers.0.self_attn.q_proj.weight", device="cuda:0")
#             tensor = await tensor_promise.get()
#             print("Tensor loaded:", tensor.shape, tensor.device)
            
#             layer_order = index.get_layer_order("my-llama-model")
#             print("Layer order sample:", layer_order[:5])
            
#         except KeyError as e:
#             print(f"Error getting tensor or layer order: {e}")
#         except Exception as e:
#             print(f"Error loading tensor: {e}")

# if __name__ == "__main__":
#    # Setup logging basic config if needed
#    logging.basicConfig(level=logging.INFO) 
#    # asyncio.run(main())