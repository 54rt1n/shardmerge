# shard/merge/base.py
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

from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Callable, Any, List
from typing import List, Optional
import asyncio
import logging
import queue
import threading
import time
import torch

from ..config import MergeConfig, MergeModel
from ..index import HFMultiModelIndex
from ..writer import ModelWriter, ShardLayer
from ..constants import INPUT_LAYER, OUTPUT_LAYER

logger = logging.getLogger(__name__)


class TaskRunner:
    def __init__(self, max_workers=3):
        self.queue = queue.Queue()
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.running = False
        self.lock = threading.Lock()
        self.loop = asyncio.get_event_loop()
        self.worker_futures = []

    # Task is an async callable
    async def _perform_task(self, task: Callable[..., Any]) -> Any:
        # Simulate an asynchronous operation
        print(f"Processing task: {task}")
        result = await task()
        print(f"Task {task} completed")
        return result

    def _worker(self):
        while True:
            try:
                task = self.queue.get(block=False)
                # Run the async task in the event loop from another thread
                future = asyncio.run_coroutine_threadsafe(self._perform_task(task), self.loop)
                future.add_done_callback(lambda f: self.queue.task_done())
                self.worker_futures.append(future)
            except queue.Empty:
                # Double-checked locking to check if we should stop
                if self.queue.empty():
                    with self.lock:
                        if self.queue.empty() and not self._has_pending_submissions():
                            break
                time.sleep(0.1)  # Small delay to avoid busy waiting

        # Clean up futures when worker stops
        for future in self.worker_futures:
            if not future.done():
                future.cancel()

    def _has_pending_submissions(self):
        # Check if there are any pending submissions in the executor
        return any(future.running() or not future.done() for future in self.executor._threads + self.worker_futures)

    def submit(self, task):
        self.queue.put(task)
        if not self.running:
            with self.lock:
                if not self.running:
                    self.running = True
                    self.executor.submit(self._worker)

    def __del__(self):
        # Ensure all tasks are completed before destruction
        self.queue.join()
        # Wait for all futures to complete
        for future in self.worker_futures:
            if not future.done():
                future.result()  # Wait for the result or raise an exception if the future was cancelled or failed
        self.executor.shutdown(wait=True)

class MergeTensorsBase(ABC):
    """Handles merging multiple finetuned models by computing and combining deltas"""
    
    def __init__(
        self,
        config: MergeConfig,
        index_manager: Optional[HFMultiModelIndex] = None
    ):
        self.config = config
        self.index_manager = index_manager or HFMultiModelIndex()

    @abstractmethod
    def get_readme(self) -> str:
        """Get README text for output model"""
        return "No readme defined"

    @abstractmethod
    async def _merge_layer(self, shard_layer: ShardLayer, device: str) -> torch.Tensor:
        """Perform the merge operation"""
        raise NotImplementedError

    async def get_base_output_tensor(self, shard_layer: ShardLayer, device: str) -> torch.Tensor:
        """Get base tensor for a given layer"""
        return (await (self.index_manager.get_tensor(self.config.output_base_model, shard_layer.layer_name, device=device)).get()).to(torch.float32)

    async def get_delta_for_models(self, models: List[MergeModel], shard_layer: ShardLayer, device: str, apply_alpha: bool = True) -> list[torch.Tensor]:
        """Get delta tensor for a given model"""
        # This is crazy inefficient, but we are optimizing for memory usage
        results = []
        model_tensors = {}
        for model in models:
            if model.base not in model_tensors:
                base_tensor = (await (self.index_manager.get_tensor(model.base, shard_layer.layer_name, device=device)).get()).to(torch.float32)
            else:
                base_tensor = model_tensors[model.base]
            model_tensor = (await (self.index_manager.get_tensor(model.model, shard_layer.layer_name, device=device)).get()).to(torch.float32)
            results.append((model_tensor - base_tensor).detach() * (model.alpha if apply_alpha else 1))
            del model_tensor
            model_tensors[model.base] = base_tensor

        del model_tensors
        return results
        
    async def initialize(self):
        """Initialize indexes for all models"""
        # Initialize base model
        await self.index_manager.add_model(self.config.output_base_model)
        self.index_doc = self.index_manager.model_indexes[self.config.output_base_model]
        
        # Initialize all finetune models
        for model in self.config.finetune_merge:
            await self.index_manager.add_model(model.base)
            await self.index_manager.add_model(model.model)
            
        # Validate all models have compatible architectures
        base_keys = self.index_manager.get_model_keys(self.config.output_base_model)
        for model in self.config.finetune_merge:
            model_keys = self.index_manager.get_model_keys(model.model)
            missing_keys = base_keys - model_keys
            extra_keys = model_keys - base_keys
            
            if missing_keys or extra_keys:
                raise ValueError(
                    f"Model {model.model} architecture mismatch with base model {self.config.base_model.model}\n"
                    f"Missing keys: {missing_keys}\n"
                    f"Extra keys: {extra_keys}"
                )

    def get_writer(self, layer_order: list[str]) -> ModelWriter:
        """Get a ModelWriter for writing output tensors"""
        return ModelWriter(
            base_index=self.index_doc,
            output_path=self.config.output_path,
            layer_order=layer_order,
            output_astype=self.config.output_astype
        )

    async def merge(self, device: str):
        """Perform the merge operation"""
        await self.initialize()
        
        logger.info(f"init complete")
        
        # Process tensors in order
        layer_order = self.index_manager.get_layer_order(self.config.output_base_model)

        # Setup writer for output tensors
        writer = self.get_writer(layer_order)
        
        # Create tasks for each shard layer
        # This could be done in parallel, but we would need to use multiprocessing
        # and throwing everything in a giant async pool causes all of the files to be downloaded
        # at once, which is a lot of pressure on the disk.
        # Is it possible to use loop.run_in_executor here?
        for shard_layers in writer.shard_layers():
            shard_layers = [
                shard_layer
                for shard_layer in shard_layers
                if not shard_layer.written
            ]
            await self._process_layers(writer, shard_layers, device)

        # Finalize output
        writer.finalize()
        
        # Generate README
        readme = self.get_readme()
        if readme is None:
            readme = "No README defined"
            logger.warning("No README defined. Using default.")
        
        with open(self.output_path / "README.md", "w") as f:
            f.write(readme)
            
        logger.info(f"Merge complete. Output saved to {self.output_path}")

    async def _process_layers(self, writer: ModelWriter, shard_layers: List[ShardLayer], device: str):
        """Helper method to process each layer"""
        try:
            for shard_layer in shard_layers:
                out_tensor = await self._merge_layer(shard_layer, device)
                # TODO we need a semaphore here to limit multiple concurrent writes if we want to multithread
                # or we can solve it with a thread-per-shard layer group
                writer.add_tensor(shard_layer.layer_name, out_tensor)
                del out_tensor
        except Exception as e:
            logger.error(f"Error processing {shard_layer.layer_name}: {e}")
            raise e
