# shard/exlora/peft.py
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

"""Emit a standard HuggingFace PEFT LoRA adapter.

Produces ``adapter_config.json`` + ``adapter_model.safetensors`` that load
with ``peft.PeftModel.from_pretrained`` and friends (vLLM, etc.).
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List
import json
import logging
import torch
from safetensors.torch import save_file

logger = logging.getLogger(__name__)


def lora_weight_key(layer_name: str, which: str) -> str:
    """Map a base weight name to its PEFT adapter tensor key.

    e.g. ``model.layers.0.self_attn.q_proj.weight`` + ``"A"`` ->
    ``base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight``.
    """
    if which not in ("A", "B"):
        raise ValueError(f"which must be 'A' or 'B', got {which!r}")
    module = layer_name[: -len(".weight")] if layer_name.endswith(".weight") else layer_name
    return f"base_model.model.{module}.lora_{which}.weight"


def module_pattern_key(layer_name: str) -> str:
    """The module path used as a key in ``rank_pattern`` / ``alpha_pattern``."""
    return layer_name[: -len(".weight")] if layer_name.endswith(".weight") else layer_name


@dataclass
class AdapterAccumulator:
    """Collects per-layer factors into a single PEFT adapter.

    LoRA factors are tiny relative to the model, so we hold the whole adapter
    in memory and write one safetensors file at the end.
    """

    base_model: str
    target_modules: List[str]
    tensors: Dict[str, torch.Tensor] = field(default_factory=dict)
    rank_pattern: Dict[str, int] = field(default_factory=dict)
    alpha_pattern: Dict[str, int] = field(default_factory=dict)

    def add(self, layer_name: str, lora_A: torch.Tensor, lora_B: torch.Tensor, rank: int):
        """Record one layer's factors (alpha == rank so PEFT scaling == 1)."""
        self.tensors[lora_weight_key(layer_name, "A")] = lora_A
        self.tensors[lora_weight_key(layer_name, "B")] = lora_B
        key = module_pattern_key(layer_name)
        self.rank_pattern[key] = rank
        self.alpha_pattern[key] = rank

    @property
    def default_rank(self) -> int:
        """A representative top-level ``r`` (max rank used) for the config."""
        return max(self.rank_pattern.values(), default=1)

    def build_config(self) -> dict:
        r = self.default_rank
        return {
            "peft_type": "LORA",
            "task_type": "CAUSAL_LM",
            "base_model_name_or_path": self.base_model,
            "r": r,
            "lora_alpha": r,
            "lora_dropout": 0.0,
            "bias": "none",
            "fan_in_fan_out": False,
            "inference_mode": True,
            "init_lora_weights": True,
            "modules_to_save": None,
            # Use the explicit set of adapted module paths (the rank_pattern
            # keys) rather than bare suffixes. PEFT matches target_modules by
            # exact name or suffix; a full path resolves to exactly that one
            # module, so unadapted modules (e.g. a multimodal model's vision
            # tower, whose projections are not plain nn.Linear) are excluded.
            "target_modules": sorted(self.rank_pattern.keys()),
            "rank_pattern": self.rank_pattern,
            "alpha_pattern": self.alpha_pattern,
        }

    def write(self, output_path: Path):
        """Write ``adapter_config.json`` and ``adapter_model.safetensors``."""
        output_path.mkdir(parents=True, exist_ok=True)

        if not self.tensors:
            raise RuntimeError("No LoRA tensors were extracted; refusing to write an empty adapter")

        cpu_tensors = {k: v.contiguous().cpu() for k, v in self.tensors.items()}
        save_file(
            cpu_tensors,
            str(output_path / "adapter_model.safetensors"),
            metadata={"format": "pt"},
        )

        with open(output_path / "adapter_config.json", "w") as f:
            json.dump(self.build_config(), f, indent=2)

        logger.info(
            f"Wrote PEFT adapter with {len(self.rank_pattern)} modules to {output_path}"
        )
