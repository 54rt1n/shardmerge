# shard/merge/addition.py
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

from .base import MergeTensorsBase
from ..config import MergeModel

logger = logging.getLogger(__name__)


class AdditionMerge(MergeTensorsBase):
    """Addition merge operation"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def get_readme(self) -> str:
        return f"""# Merged Model

Base Model: {self.base_model.model}
Finetuned Models: 
{chr(10).join('- ' + model.model for model in self.finetune_merge)}

This model was created by computing and combining the delta weights
from each finetuned model relative to the base model.
"""

    async def _merge_layer(self, layer_name: str, device: str = "cuda") -> torch.Tensor:
        """Perform the merge operation"""
        logging.info(f"Processing layer: {layer_name}")
        
        # Get base tensor
        base_tensor_promise = self.index_manager.get_tensor(
            self.base_model.model,
            layer_name,
            device=device
        )
        
        # Add deltas from each finetuned model
        ft_promises = []
        for model in self.finetune_merge:
            # Get finetuned tensor
            ft_tensor_promise = self.index_manager.get_tensor(
                model.model,
                layer_name,
                device=device
            )
            ft_promises.append(ft_tensor_promise.get())
            
        # Wait for all tensors to load
        base_tensor = await base_tensor_promise.get()
        ft_tensors = await asyncio.gather(*ft_promises)

        out_tensor = torch.zeros_like(base_tensor)

        for ft_tensor in ft_tensors:
            # Add delta to merged tensor
            delta = ft_tensor - base_tensor
            out_tensor += delta
            
            # Free up memory
            del ft_tensor

        del base_tensor
        torch.cuda.empty_cache()

        return out_tensor