# shard/merge/fast_fourier.py
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

import asyncio
import hashlib
import logging
from pathlib import Path
import torch
from typing import List, Dict, Optional
from .base import MergeTensorsBase, HFMultiModelIndex, ShardLayer, INPUT_LAYER, OUTPUT_LAYER
from ..tensor.functions import correlate_pairs, correlated_pairs, merge_tensors_fft2_slerp, task_arithmetic_fft2
from ..tensor.util import cuda_memory_profiler
from ..config import MergeConfig, MergeModel

logger = logging.getLogger(__name__)

def task_arithmetic(t0: torch.Tensor, t1: torch.Tensor) -> torch.Tensor:
    sign_agreement = torch.sign(t0) == torch.sign(t1)
    result = torch.where(sign_agreement, t0 + t1, t0)
    del sign_agreement
    return result

def name_hash(name: str) -> str:
    # split the name at each underscore and take the first 4 characters of each, recombining with underscores, then hash
    subnames = name.split("_")
    subnames = [n[:4] for n in subnames]
    subname = "_".join(subnames)
    return subname + "::" + hashlib.sha256(name.encode()).hexdigest()[:8]

def clamp(value: float, min_value: float, max_value: float) -> float:
    return max(min_value, min(value, max_value))

class TensorDiskCache:
    def __init__(self, cache_path: Path):
        self.cache_path = cache_path
        self.cache_path.mkdir(parents=True, exist_ok=True)

    def get(self, model: str, layer_name: str, device: str) -> torch.Tensor:
        # rename our model name to remove any path characters
        model = model.replace("/", "--")
        cache_file = self.cache_path / f"{model}_{layer_name}_{device}.pt"
        if cache_file.exists():
            with open(cache_file, "rb") as f:
                return torch.load(f, weights_only=True)
        return None

    def set(self, model: str, layer_name: str, device: str, tensor: torch.Tensor):
        # rename our model name to remove any path characters
        model = model.replace("/", "--")
        cache_file = self.cache_path / f"{model}_{layer_name}_{device}.pt"
        with open(cache_file, "wb") as f:
            torch.save(tensor, f)

    def remove(self, model: str, layer_name: str, device: str):
        # rename our model name to remove any path characters
        model = model.replace("/", "--")
        cache_file = self.cache_path / f"{model}_{layer_name}_{device}.pt"
        if cache_file.exists():
            cache_file.unlink()

    def clear(self):
        logger.info(f"Clearing cache: {self.cache_path}")
        for cache_file in self.cache_path.glob("*.pt"):
            cache_file.unlink()

class FourierMerge(MergeTensorsBase):
    def __init__(
        self,
        config: MergeConfig,
        task_add_models: Optional[List[str]] = None,
        target_norm_offset: float = 1e-10,
        cull_start_pct: float = 0.20,
        index_manager: Optional[HFMultiModelIndex] = None,
        **kwargs
    ):
        super().__init__(config, index_manager)
        self.task_add_models = task_add_models or []
        self.target_norm_offset = target_norm_offset
        self.cull_start_pct = cull_start_pct
        self.cache = TensorDiskCache(config.cache_path)

    def get_readme(self) -> str:
        models = "\n".join(f"- {m.model} (vs {m.base})" for m in self.config.finetune_merge)
        return f"""# SLERP-FFT Merged Model
Base: {self.config.output_base_model}
Models merged:
{models}
"""

    async def _merge_layer(self, shard_layer: ShardLayer, device: str) -> torch.Tensor:
        if shard_layer.layer_number == INPUT_LAYER:
            # Get input tensor
            input_model = next((model for model in self.config.finetune_merge if model.is_input), None)
            if input_model is None:
                input_model = MergeModel(model=self.config.output_base_model, base=self.config.output_base_model, is_input=True)
            logger.info(f"Passthrough - {shard_layer.layer_name} is an input layer, using {input_model.model} as input")
            input_tensor_promise = self.index_manager.get_tensor(
                input_model.model,
                shard_layer.layer_name,
                device=device
            )
            input_tensor = (await input_tensor_promise.get())
            return input_tensor

        if shard_layer.layer_number == OUTPUT_LAYER:
            # Get output tensor
            output_model = next((model for model in self.config.finetune_merge if model.is_output), None)
            if output_model is None:
                output_model = MergeModel(model=self.config.output_base_model, base=self.config.output_base_model, is_output=True)
            logger.info(f"Passthrough - {shard_layer.layer_name} is an output layer, using {output_model.model} as output")
            output_tensor_promise = self.index_manager.get_tensor(
                output_model.model,
                shard_layer.layer_name,
                device=device
            )
            output_tensor = (await output_tensor_promise.get())
            return output_tensor
        
        self.cache.clear()
        
        # Add deltas from each finetuned model
        models = [model for model in self.config.finetune_merge if model.use_layer_index(shard_layer.layer_number)]

        layer_norms : list[torch.Tensor] = []
        # our layer stack represents the current tensors we are operating on
        layer_stack : list[str] = []
        stack_weights : list[float] = []
        norm_models : list[MergeModel] = []

        preload = [self.index_manager.preload_tensor(model.model, shard_layer.layer_name) for model in models]
        logger.info(f"Preloading {len(preload)}")
        await asyncio.gather(*preload)

        for model in models:
            delta_tensor = await self.get_delta_for_models([model], shard_layer, device, apply_alpha=False)
            if len(delta_tensor) == 0:
                continue
            delta_tensor = delta_tensor[0]
            layer_norms.append(torch.norm(delta_tensor))
            self.cache.set(model.model, shard_layer.layer_name, device, delta_tensor)
            del delta_tensor
            layer_stack.append(model.model)
            stack_weights.append(model.alpha)
            if model.is_norm:
                norm_models.append(model)

        if len(norm_models) == 0:
            norm_models = models
        
        logger.debug(f"Loaded {len(layer_stack)} layers")

        target_norm = torch.tensor(layer_norms).mean().item() + self.target_norm_offset
        cull_pct = self.cull_start_pct

        next_stack : list[str] = []
        next_weights : list[float] = []

        while len(layer_stack) > 1:
            layer_names = [n for n in layer_stack]
            next_stack = []
            next_weights = []

            logger.info(f"Processing {len(layer_stack)} layers : {', '.join(layer_names)}")

            with cuda_memory_profiler(title="correlation", display=False):
                # Normalize the vectors
                vector = torch.stack(layer_norms)
                correlation = torch.zeros((len(layer_stack), len(layer_stack)), dtype=torch.float32, device=device)
                for i in range(len(layer_stack)):
                    for j in range(i + 1, len(layer_stack)):
                        correlation[i, j] = vector[i] * vector[j]
                
                correlation = correlation.to('cpu')

            for x, y, corr in correlated_pairs(correlation, way="least"):
                with cuda_memory_profiler(title="merger", display=False):
                    if y < 0:
                        next_stack.append(layer_stack[x])
                        next_weights.append(stack_weights[x])
                        continue


                    name_a = layer_names[x]
                    name_b = layer_names[y]

                    logger.info(f"Merging {x}, {y}: {name_a}, {name_b}, layer_stack: {layer_stack}, stack_weights: {stack_weights}")

                    a_model = layer_stack[x]
                    b_model = layer_stack[y]
                    a_weight = stack_weights[x]
                    b_weight = stack_weights[y]

                    a = self.cache.get(a_model, shard_layer.layer_name, device)
                    b = self.cache.get(b_model, shard_layer.layer_name, device)

                    norm_a = torch.norm(a).item()
                    norm_b = torch.norm(b).item()

                    if abs(norm_a) < abs(norm_b):
                        a, b = b, a
                        a_model, b_model = b_model, a_model
                        norm_a, norm_b = norm_b, norm_a

                    cnorm_a = abs(norm_a / target_norm)
                    cnorm_b = abs(norm_b / target_norm)
                    n_ratio = cnorm_b / (cnorm_a + 1e-10)

                    logger.debug(f"Merging {x}, {y} with {cnorm_a} and {cnorm_b} {n_ratio:.2f}")

                    if cnorm_a < 1e-6:
                        merged = a + b
                        logger.info(f"Merged {a_model} and {b_model} with weight {merged.abs().sum()}")
                    elif cnorm_b < 1e-6 or n_ratio < 0.1:
                        norm_scale = target_norm / norm_a
                        scaled_a = a * norm_scale
                        weight_scale = b_weight / (a_weight + 1e-10)
                        scaled_b = b * weight_scale * norm_scale
                        merged = task_arithmetic_fft2(scaled_a, scaled_b, t=1.0, agreement=True, device=device)
                        logger.info(f"Arithmetic-FFT Merged {b_model} x {weight_scale} on to {a_model} x {norm_scale} ({norm_a} -> {target_norm}) Energy {merged.abs().sum()}")
                    else:
                        a_prop = a_weight / (a_weight + b_weight)
                        merged, _, _ = merge_tensors_fft2_slerp(
                            a, b, 
                            t=a_prop,
                            t_sum=1.0,
                            cutoff_pct=0.08, # Arithmetic-FFT cutoff
                            cull_pct=cull_pct,
                            device=device,
                        )
                        merged = merged * target_norm
                        logger.info(f"SLERP-FFT Merged {a_model} and {b_model} with weight {a_prop} {merged.abs().sum()}")

                    name = name_hash(f"{a_model}_{b_model}")
                    next_stack.append(name)
                    next_weights.append((a_weight + b_weight) / 2.0)
                    self.cache.set(name, shard_layer.layer_name, device, merged)
                    del a, b, merged

            layer_stack = next_stack
            stack_weights = next_weights
            cull_pct = cull_pct / 2.0 # reduce cull percentage as we merge deeper

        result_key = layer_stack[0]
        result_tensor = self.cache.get(result_key, shard_layer.layer_name, device)

        # TODO: add back in
        """
        for model_name, ft_tensor in add_stack:
            ft_tensor = ft_tensor.to(device)
            merged = task_arithmetic_fft2(result_tensor, ft_tensor, t=1, agreement=False)
            logger.info(f"Arithmetic Merged {a_key} and {b_key} with weight {1} {merged.sum()}")
            del result_tensor, ft_tensor
            result_tensor = merged.detach()
        """

        result_tensor = await self.get_base_output_tensor(shard_layer=shard_layer, device=result_tensor.device) + result_tensor
        if torch.any(torch.isnan(result_tensor)):
            result_tensor[torch.isnan(result_tensor)] = 0.0

        if torch.any(torch.isinf(result_tensor)):
            raise ValueError(f"Inf in merged tensor for {shard_layer.layer_name}")

        return result_tensor.to(torch.bfloat16)