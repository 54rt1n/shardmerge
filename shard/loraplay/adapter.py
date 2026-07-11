# shard/loraplay/adapter.py
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

"""Load a PEFT LoRA adapter into per-layer dense deltas for scoring.

Inverts the key convention in :mod:`shard.exlora.peft`: adapter tensors are
``base_model.model.<module>.lora_{A,B}.weight`` and the applied update is
``scaling · B @ A`` with ``scaling = lora_alpha / r`` (resolved per layer via
``rank_pattern`` / ``alpha_pattern``).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterator, Tuple
import json

import torch
from safetensors import safe_open

from ..exlora.functions import lora_delta

_PREFIX = "base_model.model."


def parse_lora_key(key: str) -> Tuple[str, str]:
    """Return ``(weight_name, which)`` for a LoRA adapter tensor key.

    e.g. ``base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight`` ->
    (``model.layers.0.self_attn.q_proj.weight``, ``"A"``).
    """
    if not key.startswith(_PREFIX):
        raise ValueError(f"not a PEFT LoRA key: {key!r}")
    body = key[len(_PREFIX):]
    for which in ("A", "B"):
        suffix = f".lora_{which}.weight"
        if body.endswith(suffix):
            return body[: -len(suffix)] + ".weight", which
    raise ValueError(f"not a lora_A/lora_B tensor: {key!r}")


@dataclass
class LoadedAdapter:
    """A PEFT adapter resolved to per-module factors and scalings."""

    factors: Dict[str, Dict[str, torch.Tensor]]  # weight_name -> {"A":.., "B":..}
    scaling: Dict[str, float]                     # weight_name -> alpha/r
    rank: Dict[str, int]                          # weight_name -> r
    config: dict

    def layers(self) -> Iterator[str]:
        return iter(self.factors)

    def delta(self, weight_name: str, compute_dtype: torch.dtype = torch.float32) -> torch.Tensor:
        f = self.factors[weight_name]
        return lora_delta(f["A"], f["B"], scale=self.scaling[weight_name], compute_dtype=compute_dtype)


def load_adapter(adapter_dir: Path) -> LoadedAdapter:
    adapter_dir = Path(adapter_dir)
    config = json.loads((adapter_dir / "adapter_config.json").read_text())
    default_r = int(config.get("r", 1))
    default_alpha = float(config.get("lora_alpha", default_r))
    rank_pattern = config.get("rank_pattern") or {}
    alpha_pattern = config.get("alpha_pattern") or {}

    factors: Dict[str, Dict[str, torch.Tensor]] = {}
    with safe_open(str(adapter_dir / "adapter_model.safetensors"), framework="pt") as f:
        for key in f.keys():
            weight_name, which = parse_lora_key(key)
            factors.setdefault(weight_name, {})[which] = f.get_tensor(key)

    scaling: Dict[str, float] = {}
    rank: Dict[str, int] = {}
    for weight_name, fa in factors.items():
        if "A" not in fa or "B" not in fa:
            raise ValueError(f"adapter missing A or B for {weight_name}")
        module = weight_name[: -len(".weight")]
        r = int(rank_pattern.get(module, default_r))
        alpha = float(alpha_pattern.get(module, default_alpha))
        # r is also recoverable from the factor shape; prefer the actual shape
        r = fa["A"].shape[0]
        scaling[weight_name] = alpha / r if r else 1.0
        rank[weight_name] = r

    return LoadedAdapter(factors=factors, scaling=scaling, rank=rank, config=config)
