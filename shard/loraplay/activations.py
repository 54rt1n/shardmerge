# shard/loraplay/activations.py
# Copyright (C) 2026 Martin Bukowski
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

"""Harvest per-layer input second moments ``H = XᵀX`` from a base model.

One hooked forward pass over calibration text accumulates, for every targeted
Linear, the Gram of its *inputs* (never storing the activations themselves).
The resulting ``{weight_name: H}`` is the calibration artifact: it is both the
true activation metric for projection and the weighting for scoring.

The :class:`GramAccumulator` is pure and unit-testable; :func:`harvest_model`
is the thin glue that loads an HF model and drives the pass.
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional
import logging

import torch

logger = logging.getLogger(__name__)


class GramAccumulator:
    """Accumulates ``H = Σ xᵀx`` for a set of named Linear modules via hooks."""

    def __init__(self, target_suffixes: List[str], compute_dtype: torch.dtype = torch.float32):
        self.target_suffixes = tuple(target_suffixes)
        self.compute_dtype = compute_dtype
        self.gram: Dict[str, torch.Tensor] = {}
        self.counts: Dict[str, int] = {}
        self._handles: list = []

    def is_target(self, module_name: str) -> bool:
        return any(module_name.endswith(s) for s in self.target_suffixes)

    def observe(self, name: str, x: torch.Tensor) -> None:
        """Fold one input batch ``x`` (``[..., in]``) into the running Gram."""
        flat = x.reshape(-1, x.shape[-1]).to(self.compute_dtype)
        g = flat.T @ flat
        if name in self.gram:
            self.gram[name] += g
            self.counts[name] += flat.shape[0]
        else:
            self.gram[name] = g
            self.counts[name] = flat.shape[0]

    def weight_name(self, module_name: str) -> str:
        """Map a module name to its weight-tensor name (``…q_proj`` -> ``…q_proj.weight``)."""
        return f"{module_name}.weight"

    def register(self, model) -> None:
        """Attach forward-pre-hooks to every targeted ``nn.Linear`` in ``model``."""
        import torch.nn as nn

        for name, module in model.named_modules():
            if isinstance(module, nn.Linear) and self.is_target(name):
                def hook(mod, inputs, _name=name):
                    self.observe(_name, inputs[0].detach())
                self._handles.append(module.register_forward_pre_hook(hook))
        logger.info(f"Registered Gram hooks on {len(self._handles)} linear modules")

    def remove(self) -> None:
        for h in self._handles:
            h.remove()
        self._handles = []

    def result(self) -> Dict[str, torch.Tensor]:
        """Return ``{weight_name: H}`` (mean-normalized by token count)."""
        return {
            self.weight_name(name): (g / max(self.counts[name], 1)).cpu()
            for name, g in self.gram.items()
        }


def harvest_model(
    model,
    batches: Iterable,
    target_suffixes: List[str],
    device: Optional[str] = None,
    compute_dtype: torch.dtype = torch.float32,
) -> Dict[str, torch.Tensor]:
    """Run a hooked forward pass over ``batches`` and return per-layer ``H``.

    ``batches`` yields tensors or dicts suitable for ``model(**batch)`` /
    ``model(batch)``. The model is run under ``torch.no_grad`` in eval mode.
    """
    acc = GramAccumulator(target_suffixes, compute_dtype=compute_dtype)
    acc.register(model)
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            for batch in batches:
                if isinstance(batch, dict):
                    if device:
                        batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
                    model(**batch)
                else:
                    if device and torch.is_tensor(batch):
                        batch = batch.to(device)
                    model(batch)
    finally:
        acc.remove()
        if was_training:
            model.train()
    return acc.result()


def save_grams(grams: Dict[str, torch.Tensor], path) -> None:
    from safetensors.torch import save_file
    from pathlib import Path
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    save_file({k: v.contiguous().cpu() for k, v in grams.items()}, str(path),
              metadata={"format": "pt", "content": "loraplay_input_gram"})


def load_grams(path) -> Dict[str, torch.Tensor]:
    from safetensors import safe_open
    out = {}
    with safe_open(str(path), framework="pt") as f:
        for k in f.keys():
            out[k] = f.get_tensor(k)
    return out
