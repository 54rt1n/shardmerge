# shard/loraplay/player.py
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

"""Play a PEFT adapter against a live base model and score it in one pass.

Instead of harvesting the input Gram ``H`` (which is in×in per layer — up to
~1.8 GB for ``down_proj``), this hooks each targeted Linear and, using the live
input ``X``, accumulates the activation reconstruction error directly:

    resid  += ‖ X·(Δ_true − Δ_lora)ᵀ ‖²        (output error of base+lora vs true)
    signal += ‖ X·Δ_trueᵀ ‖²                    (output change the finetune makes)

Only two scalars per layer are kept; activations and ``H`` are never stored.
This is the *local* (one-step) activation error on the base trajectory — the
GPTQ/AWQ objective — not the K-step trajectory metric.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Callable, Dict, List, Optional
import logging

import torch

from .adapter import LoadedAdapter
from .scoring import LayerScore, ScoreReport

logger = logging.getLogger(__name__)

# given (weight_name, base_weight) -> Δ_true = W_ft - W_base, shape [out, in]
TrueDeltaProvider = Callable[[str, torch.Tensor], Optional[torch.Tensor]]


class ActivationErrorAccumulator:
    """Running per-layer activation residual / signal energy."""

    def __init__(self):
        self.resid_sq: Dict[str, float] = defaultdict(float)
        self.signal_sq: Dict[str, float] = defaultdict(float)
        self.tokens: Dict[str, int] = defaultdict(int)
        self.rank: Dict[str, int] = {}

    def observe(self, name: str, X: torch.Tensor, d_true: torch.Tensor, d_lora: torch.Tensor) -> None:
        """Fold one input batch into the running activation-error energy."""
        x = X.reshape(-1, X.shape[-1]).to(torch.float32)
        E = (d_true - d_lora).to(torch.float32)
        self.resid_sq[name] += (x @ E.T).pow(2).sum().item()
        self.signal_sq[name] += (x @ d_true.to(torch.float32).T).pow(2).sum().item()
        self.tokens[name] += x.shape[0]

    def rel_error(self, name: str) -> float:
        s = self.signal_sq[name]
        return (self.resid_sq[name] / s) ** 0.5 if s > 0 else 0.0

    def report(self, ranks: Dict[str, int], shapes: Dict[str, tuple]) -> ScoreReport:
        layers: List[LayerScore] = []
        for name in self.signal_sq:
            out_f, in_f = shapes.get(name, (0, 0))
            r = ranks.get(name, 0)
            layers.append(LayerScore(
                layer_name=name, rank=r, out_features=out_f, in_features=in_f,
                raw_rel_error=float("nan"),  # raw error not measured on this path
                act_rel_error=self.rel_error(name),
                nbytes=r * (out_f + in_f) * 2,
            ))
        return ScoreReport(layers)


class AdapterPlayer:
    """Registers hooks that play ``adapter`` against a base model and score it."""

    def __init__(
        self,
        adapter: LoadedAdapter,
        true_delta_provider: TrueDeltaProvider,
        weight_name_of: Callable[[str], str] = lambda m: f"{m}.weight",
    ):
        self.adapter = adapter
        self.provider = true_delta_provider
        self.weight_name_of = weight_name_of
        self.acc = ActivationErrorAccumulator()
        self.shapes: Dict[str, tuple] = {}
        self._handles: list = []

    def _scored_weight_names(self) -> set:
        return set(self.adapter.factors)

    def register(self, model) -> None:
        import torch.nn as nn
        scored = self._scored_weight_names()
        for name, module in model.named_modules():
            if not isinstance(module, nn.Linear):
                continue
            wname = self.weight_name_of(name)
            if wname not in scored:
                continue

            def hook(mod, inputs, _wname=wname):
                X = inputs[0].detach()
                base_w = mod.weight.detach()
                d_true = self.provider(_wname, base_w)
                if d_true is None:
                    return
                d_lora = self.adapter.delta(_wname).to(base_w.device, base_w.dtype)
                self.acc.observe(_wname, X, d_true.to(base_w.device), d_lora)
                self.shapes[_wname] = tuple(base_w.shape)
            self._handles.append(module.register_forward_pre_hook(hook))
        logger.info(f"Playing adapter through {len(self._handles)} scored linears")

    def remove(self) -> None:
        for h in self._handles:
            h.remove()
        self._handles = []

    def report(self) -> ScoreReport:
        return self.acc.report(self.adapter.rank, self.shapes)


class FinetuneDeltaProvider:
    """Streams ``W_ft`` per layer from a finetune dir; returns ``W_ft - W_base``.

    The base weight is supplied live by the hook (it is the running model's
    weight), so only the finetune tensor is read from disk, one at a time.
    """

    def __init__(self, finetune_dir):
        from pathlib import Path
        import json
        self.dir = Path(finetune_dir)
        self.weight_map = json.load(open(self.dir / "model.safetensors.index.json"))["weight_map"]
        self._handles: dict = {}

    def __call__(self, weight_name: str, base_weight: torch.Tensor) -> Optional[torch.Tensor]:
        if weight_name not in self.weight_map:
            return None
        from safetensors import safe_open
        shard = self.weight_map[weight_name]
        if shard not in self._handles:
            self._handles[shard] = safe_open(str(self.dir / shard), framework="pt")
        w_ft = self._handles[shard].get_tensor(weight_name).to(base_weight.device, torch.float32)
        return w_ft - base_weight.to(torch.float32)


def play_adapter(model, batches, adapter: LoadedAdapter, true_delta_provider: TrueDeltaProvider,
                 device: Optional[str] = None) -> ScoreReport:
    """One hooked forward pass: play ``adapter`` and return activation scores."""
    player = AdapterPlayer(adapter, true_delta_provider)
    player.register(model)
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
        player.remove()
        if was_training:
            model.train()
    return player.report()
