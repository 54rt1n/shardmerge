# shard/exlora/config.py
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

import click
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional
import torch
import yaml

# Standard transformer Linear projections that factor cleanly into LoRA.
DEFAULT_TARGET_MODULES: List[str] = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]


@dataclass
class ExtractConfig:
    """Configuration for high-fidelity LoRA extraction.

    Mirrors the YAML-driven style of :class:`shard.config.MergeConfig` and
    reuses the same ``storage_dir`` / ``cache_dir`` download plumbing.
    """

    base_model: str
    finetune_model: str
    output_dir: str

    # Rank selection (error-target adaptive).
    target_error: float = 0.01
    min_rank: int = 1
    max_rank: int = 128
    rank_policy: str = "error"  # error | signal | error_or_signal
    noise_floor_multiplier: float = 1.5
    min_knee_drop: float = 1.25
    tail_fraction: float = 0.25

    # Per-module rank-budget overrides (by module suffix, e.g. "down_proj").
    # Used to reallocate rank from behaviorally-forgiving modules to starved
    # ones based on measured activation error. Falls back to min_rank/max_rank.
    module_max_rank: dict = field(default_factory=dict)
    module_min_rank: dict = field(default_factory=dict)

    target_modules: List[str] = field(default_factory=lambda: list(DEFAULT_TARGET_MODULES))

    # Also capture the token-embedding and/or LM-head (output projection) deltas
    # alongside the body, via the same artifact_type path. The LM head is the
    # readout; on a TIED base its head IS embed_tokens, so the lm_head delta is
    # (ft.lm_head - base.embed_tokens). qat changes both; an untied finetune
    # (e.g. maldv Isometry) retrains a separate lm_head.
    extract_embeddings: bool = False
    extract_lm_head: bool = False

    # Output artifact. "lora" keeps the PEFT-compatible low-rank extractor.
    # "nvfp4_residual" writes a dense NVFP4 delta map plus magnitude-gated
    # additive residual blocks.
    artifact_type: str = "lora"
    nvfp4_extra_bpe: float = 0.0
    nvfp4_residual_dtype: str = "bfloat16"
    nvfp4_index_bits: int = 32
    nvfp4_selection_policy: str = "residual_rms"
    nvfp4_floor_correction: str = "none"
    nvfp4_floor_tolerance: float = 1.0
    nvfp4_floor_effect_threshold: float = 2.0

    # Diagnostic-only layer-position buckets for reports.
    position_policy_enabled: bool = True
    position_early_fraction: float = 0.20
    position_late_fraction: float = 0.75

    # QAT projection (P_qat). When unset, extraction is the raw-SVD baseline.
    # `qat_projection_model` points at a prebuilt QAT survival map; building
    # that map is itself a raw extraction of (qat_model - base_model).
    # projection_type: none | metric | reflection  (only "none" is wired today).
    qat_projection_model: Optional[str] = None
    projection_type: str = "none"

    device: str = "cpu"
    dtype: str = "bfloat16"

    storage_dir: str = "storage"
    cache_dir: str = "cache"
    clean_cache: bool = False

    @property
    def output_path(self) -> Path:
        return Path(self.output_dir)

    @property
    def cache_path(self) -> Path:
        return Path(self.cache_dir)

    @property
    def storage_path(self) -> Path:
        return Path(self.storage_dir)

    @property
    def out_dtype(self) -> torch.dtype:
        return getattr(torch, self.dtype)

    def is_target(self, layer_name: str) -> bool:
        """True if ``layer_name`` is a targeted Linear weight."""
        if not layer_name.endswith(".weight"):
            return False
        module = layer_name[: -len(".weight")]
        return any(module.endswith(target) for target in self.target_modules)

    def module_of(self, layer_name: str) -> Optional[str]:
        """Return the module suffix (e.g. ``q_proj``) for a weight name."""
        if not layer_name.endswith(".weight"):
            return None
        return layer_name[: -len(".weight")].split(".")[-1]

    def rank_bounds(self, layer_name: str) -> tuple:
        """Return ``(min_rank, max_rank)`` for a layer, honoring per-module overrides."""
        module = self.module_of(layer_name)
        mn = self.module_min_rank.get(module, self.min_rank)
        mx = self.module_max_rank.get(module, self.max_rank)
        return mn, mx

    def update(self, config: dict = {}, **kwargs):
        for source in (config, kwargs):
            for key, value in source.items():
                if value is not None and hasattr(self, key):
                    setattr(self, key, value)

    @classmethod
    def from_yaml(cls, config_path: Path) -> "ExtractConfig":
        """Load and validate a YAML configuration."""
        with open(config_path) as f:
            config = yaml.safe_load(f) or {}

        required = ["base_model", "finetune_model", "output_dir"]
        missing = [field_name for field_name in required if field_name not in config]
        if missing:
            raise click.BadParameter(
                f"Missing required configuration fields: {', '.join(missing)}"
            )

        if "target_modules" in config and not isinstance(config["target_modules"], list):
            raise click.BadParameter("target_modules must be a list of module suffixes")

        # Ignore unknown keys rather than blowing up on extra YAML.
        known = cls.__dataclass_fields__.keys()
        filtered = {k: v for k, v in config.items() if k in known}
        cfg = cls(**filtered)
        if cfg.artifact_type not in ("lora", "nvfp4_residual"):
            raise click.BadParameter("artifact_type must be 'lora' or 'nvfp4_residual'")
        if cfg.nvfp4_extra_bpe < 0:
            raise click.BadParameter("nvfp4_extra_bpe must be non-negative")
        if cfg.nvfp4_index_bits not in (0, 16, 32, 64):
            raise click.BadParameter("nvfp4_index_bits must be one of 0, 16, 32, or 64")
        if cfg.nvfp4_selection_policy not in ("residual_rms", "floor_effect"):
            raise click.BadParameter("nvfp4_selection_policy must be 'residual_rms' or 'floor_effect'")
        if cfg.nvfp4_floor_correction not in ("none", "code_position_mean"):
            raise click.BadParameter("nvfp4_floor_correction must be 'none' or 'code_position_mean'")
        if cfg.nvfp4_floor_tolerance < 0:
            raise click.BadParameter("nvfp4_floor_tolerance must be non-negative")
        if cfg.nvfp4_floor_effect_threshold < 0:
            raise click.BadParameter("nvfp4_floor_effect_threshold must be non-negative")
        if cfg.nvfp4_selection_policy in ("residual_rms", "floor_effect") and cfg.nvfp4_index_bits == 0:
            raise click.BadParameter(f"{cfg.nvfp4_selection_policy} selection requires stored block indices")
        if not 0 <= cfg.position_early_fraction <= 1:
            raise click.BadParameter("position_early_fraction must be in [0, 1]")
        if not 0 <= cfg.position_late_fraction <= 1:
            raise click.BadParameter("position_late_fraction must be in [0, 1]")
        if cfg.position_early_fraction >= cfg.position_late_fraction:
            raise click.BadParameter("position_early_fraction must be less than position_late_fraction")
        return cfg
