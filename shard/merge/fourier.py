# shard/merge/fourier.py
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
from typing import List, Dict, Optional
import logging
import torch
from .base import MergeTensorsBase, HFMultiModelIndex, ShardLayer, INPUT_LAYER, OUTPUT_LAYER
from ..tensor.functions import correlate_pairs, correlated_pairs, merge_tensors_fft2_slerp, task_arithmetic_fft2
from ..tensor.util import cuda_memory_profiler
from ..config import MergeConfig

logger = logging.getLogger(__name__)

def task_arithmetic(t0: torch.Tensor, t1: torch.Tensor) -> torch.Tensor:
    sign_agreement = torch.sign(t0) == torch.sign(t1)
    result = torch.where(sign_agreement, t0 + t1, t0)
    del sign_agreement
    return result


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

    def get_readme(self) -> str:
        models = "\n".join(f"- {m.model}" for m in self.finetune_models)
        return f"""# SLERP-FFT Merged Model
Base: {self.base_model.model}
Models merged:
{models}
"""

    async def _merge_layer(self, shard_layer: ShardLayer, device: str) -> torch.Tensor:
        if shard_layer.layer_number == INPUT_LAYER:
            # Get input tensor
            input_model = next((model for model in self.config.finetune_merge if model.is_input), None)
            if input_model is None:
                raise ValueError("No input model found")
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
                raise ValueError("No output model found")
            logger.info(f"Passthrough - {shard_layer.layer_name} is an output layer, using {output_model.model} as output")
            output_tensor_promise = self.index_manager.get_tensor(
                output_model.model,
                shard_layer.layer_name,
                device=device
            )
            output_tensor = (await output_tensor_promise.get())
            return output_tensor
        
        # Get base tensor
        base_tensor_promise = self.index_manager.get_tensor(
            self.base_model.model,
            shard_layer.layer_name,
            device=device
        )

        # Add deltas from each finetuned model
        ft_promises = []
        for model in self.config.finetune_merge:
            if not model.use_layer(shard_layer.layer_name):
                continue
            # Get finetuned tensor
            ft_tensor_promise = self.index_manager.get_tensor(
                model.model,
                shard_layer.layer_name,
                device=device
            )
            ft_promises.append(ft_tensor_promise.get())
            
        # Wait for all tensors to load
        base_tensor = (await base_tensor_promise.get())
        layer_stack : list[torch.Tensor] = []
        add_stack : list[torch.Tensor] = []
        mean_norms : list[torch.Tensor] = []
        for i, ft_tensor in enumerate(await asyncio.gather(*ft_promises)):
            ft_tensor -= base_tensor
            model = self.finetune_models[i]
            if model.model in self.task_add_models:
                add_stack.append((model.model, ft_tensor.cpu().detach()))
            else:
                norm = torch.norm(ft_tensor).item()
                mean_norms.append(norm)
                layer_stack.append((model.model, ft_tensor.cpu().detach()))
                
        logger.debug(f"Loaded {len(layer_stack)} layers")

        target_norm = torch.tensor(mean_norms).median().item() + self.target_norm_offset
        cull_pct = self.cull_start_pct

        while len(layer_stack) > 1:
            layer_names = [n for n, _ in layer_stack]
            logger.info(f"Processing {len(layer_stack)} layers : {', '.join(layer_names)}")

            with cuda_memory_profiler(title="correlation", display=False):
                correlation = correlate_pairs(torch.stack([t for _, t in layer_stack], dim=0), store_device="cpu", work_device=device)
            next_stack = []
            
            logger.debug(f"Found {len(correlation)} correlations")

            for x, y, corr in correlated_pairs(correlation, way="least"):
                with cuda_memory_profiler(title="merger", display=False):
                    if y < 0:
                        next_stack.append(layer_stack[x])
                        continue

                    a, a_key = layer_stack[x][1].to(device), layer_stack[x][0]
                    b, b_key = layer_stack[y][1].to(device), layer_stack[y][0]

                    norm_a = torch.norm(a).item()
                    norm_b = torch.norm(b).item()

                    if abs(norm_a) < abs(norm_b):
                        a, b = b, a
                        a_key, b_key = b_key, a_key
                        norm_a, norm_b = norm_b, norm_a

                    cnorm_a = abs(norm_a / target_norm)
                    cnorm_b = abs(norm_b / target_norm)
                    n_ratio = cnorm_b / (cnorm_a + 1e-10)

                    logger.debug(f"Merging {x}, {y} with {cnorm_a} and {cnorm_b} {n_ratio:.2f}")

                    if cnorm_a < 1e-6:
                        merged = a + b
                        logger.info(f"Merged {a_key} and {b_key} with weight {merged.abs().sum()}")
                    elif cnorm_b < 1e-6 or n_ratio < 0.1:
                        scaled_a = a * target_norm / norm_a
                        merged = task_arithmetic_fft2(scaled_a, b, t=1.0, agreement=True, device=device)
                        logger.info(f"Arithmetic-FFT Merged {a_key} and {b_key} with norm {norm_a} -> {target_norm} {merged.abs().sum()}")
                    else:
                        # TODO verify that x and y are in the same order as self.finetune_models
                        a_weight = self.config.finetune_models[x].alpha
                        b_weight = self.config.finetune_models[y].alpha
                        a_prop = a_weight / (a_weight + b_weight)
                        merged, _, _ = merge_tensors_fft2_slerp(
                            a, b, 
                            t=a_prop,
                            t_sum=1.0,
                            cutoff_pct=0.08,
                            cull_pct=cull_pct,
                            device=device,
                        )
                        merged = merged * target_norm
                        logger.info(f"SLERP-FFT Merged {a_key} and {b_key} with weight {a_prop} {merged.abs().sum()}")

                    next_stack.append((f"{a_key}_{b_key}", merged.cpu().detach()))
                    del a, b, merged

            layer_stack = next_stack
            cull_pct = cull_pct / 2.0

        result_tensor = layer_stack[0][1].to(device)

        for model_name, ft_tensor in add_stack:
            ft_tensor = ft_tensor.to(device)
            merged = task_arithmetic_fft2(result_tensor, ft_tensor, t=1, agreement=False)
            logger.info(f"Arithmetic Merged {a_key} and {b_key} with weight {1} {merged.sum()}")
            del result_tensor, ft_tensor
            result_tensor = merged.detach()

        result_tensor = base_tensor + result_tensor
        if torch.any(torch.isnan(result_tensor)):
            result_tensor[torch.isnan(result_tensor)] = 0.0

        if torch.any(torch.isinf(result_tensor)):
            raise ValueError(f"Inf in merged tensor for {shard_layer.layer_name}")

        return result_tensor