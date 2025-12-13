# shard/merge/taskaddition.py
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
import logging
import torch

from .base import MergeTensorsBase, ShardLayer
from ..config import MergeModel

logger = logging.getLogger(__name__)


class TaskAdditionMerge(MergeTensorsBase):
    """Addition merge operation, using sign agreement"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def get_readme(self) -> str:
        return f"""# Merged Model

Base Model: {self.config.output_base_model}
Finetuned Models:
{chr(10).join('- ' + model.model for model in self.config.finetune_merge)}

This model was created by computing and combining the delta weights
from each finetuned model relative to the base model, using sign agreement.
"""

    async def _merge_layer(self, shard_layer: ShardLayer, device: str) -> torch.Tensor:
        """Perform the merge operation"""
        logging.info(f"Processing layer: {shard_layer.layer_name}")

        # Get base tensor
        base_tensor_promise = self.index_manager.get_tensor(
            self.config.output_base_model,
            shard_layer.layer_name,
            device=device
        )

        # Add deltas from each finetuned model
        ft_promises = []
        for model in self.config.finetune_merge:
            # Get finetuned tensor
            ft_tensor_promise = self.index_manager.get_tensor(
                model.model,
                shard_layer.layer_name,
                device=device
            )
            ft_promises.append(ft_tensor_promise.get())
            
        # Wait for all tensors to load
        base_tensor = (await base_tensor_promise.get())
        ft_tensors = (await asyncio.gather(*ft_promises))
        ft_tensors = torch.stack([t - base_tensor for t in ft_tensors], dim=0).cpu()
        del base_tensor
        # Compute the sign agreement
        sign_agreement = torch.sign(ft_tensors)
        # we need to sum the sign agreement down our stacked tensors
        sign_weight = torch.sum(sign_agreement, dim=0).sign()
        # mask our ft_tensors with the sign agreement and the sign weight, all nonmatching values will be 0
        mask = (sign_agreement == sign_weight.unsqueeze(0))
        ft_tensors = ft_tensors * mask
        # Sum our remaining values
        ft_tensors = torch.sum(ft_tensors, dim=0)

        torch.cuda.empty_cache()

        return ft_tensors