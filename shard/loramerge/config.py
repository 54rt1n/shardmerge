# shard/loramerge/config.py
# Copyright (C) 2026 Martin Bukowski
#
# This software is free software: you can redistribute it and/or
# modify it under the terms of the GNU Lesser General Public License as
# published by the Free Software Foundation, either version 3 of the
# License, or (at your option) any later version.

"""YAML configuration for full-frame LoRA/dense artifact merges."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import click
import torch
import yaml

from ..exlora.config import DEFAULT_TARGET_MODULES
from .merge import MergeRule, WeightedDeltaSource
from .sources import load_delta_source

DELTA_SOURCE_KINDS = {"peft", "dense", "nvfp4", "nvfp4_residual", "nvfp4_residual_delta_map",
                      "checkpoint", "model_delta"}
MODEL_SOURCE_KINDS = {"model", "hf_model", "passthrough"}
SOURCE_KINDS = DELTA_SOURCE_KINDS | MODEL_SOURCE_KINDS
MERGE_RULES = {"sum", "sign_vote", "norm_slerp", "fourier_slerp", "fft_slerp", "isometric"}
SCAFFOLD_FORMATS = {"none", "nvfp4_residual"}


def is_embedding_tensor(tensor_name: str) -> bool:
    """Return true for token embedding tensors across common HF layouts."""
    return tensor_name.endswith("embed_tokens.weight")


def is_lm_head_tensor(tensor_name: str) -> bool:
    """Return true for output projection / language-model head tensors."""
    return tensor_name == "lm_head.weight" or tensor_name.endswith(".lm_head.weight")


@dataclass
class LoraMergeSourceConfig:
    """One adapter/dense artifact participating in a full-frame merge."""

    path: str
    kind: Optional[str] = None
    weight: float = 1.0
    rescale: float = 1.0          # multiplies the source's delta MAGNITUDE (alpha rescale),
    base_model: Optional[str] = None  # diff base for checkpoint sources (default: merge base)
    name: Optional[str] = None    # distinct from `weight` (the merge-rule mixture weight)
    start_layer: int = 0
    end_layer: int = -1
    target_modules: Optional[list[str]] = None
    is_embedding: bool = False
    is_lm_head: bool = False
    is_norm: bool = False         # this source's delta norm defines the merged output magnitude
    enabled: bool = True
    priority: float = 0.0         # layering order: sources fold into the merge by DESCENDING
                                  # priority (highest first anchors the accumulator). Signed:
                                  # negative pushes a source later, after the priority-0 default.
    post_merge: bool = False      # held out of the body fold and applied against the final body.
                                  # Under isometric merging this is an additive overlay.
    group: Optional[str] = None   # body sources sharing a group are merged INTO EACH OTHER
                                  # first (one representative), then that rep folds by priority.
    pool: Optional[str] = None    # opt-in isometric pooled merge block. Unlike group, this
                                  # changes rule=isometric behavior.
    pool_rank: Optional[int] = None       # per-pool rank budget; first value in pool wins.
    pool_energy: Optional[float] = None   # per-pool radial energy; first value in pool wins.
    pool_residual: Optional[float] = None # prior-pool subtraction strength in [0, 1].

    def __post_init__(self) -> None:
        if self.kind is not None:
            self.kind = self.kind.lower()
        self.weight = float(self.weight)
        self.rescale = float(self.rescale)
        self.priority = float(self.priority)
        self.is_embedding = bool(self.is_embedding)
        self.is_lm_head = bool(self.is_lm_head)
        self.is_norm = bool(self.is_norm)
        self.enabled = bool(self.enabled)
        self.post_merge = bool(self.post_merge)
        if self.pool_rank is not None:
            self.pool_rank = int(self.pool_rank)
        if self.pool_energy is not None:
            self.pool_energy = float(self.pool_energy)
        if self.pool_residual is not None:
            self.pool_residual = float(self.pool_residual)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "LoraMergeSourceConfig":
        """Build a source config, accepting old merge-style aliases."""
        payload = dict(data)
        if "path" not in payload:
            for alias in ("artifact", "model"):
                if alias in payload:
                    payload["path"] = payload[alias]
                    break
        if "weight" not in payload and "alpha" in payload:
            payload["weight"] = payload["alpha"]
        if "is_embedding" not in payload:
            for alias in ("is_input", "embedding_source", "embed_source"):
                if alias in payload:
                    payload["is_embedding"] = payload[alias]
                    break
        if "is_lm_head" not in payload:
            for alias in ("is_output", "lm_head_source", "head_source"):
                if alias in payload:
                    payload["is_lm_head"] = payload[alias]
                    break
        if "enabled" not in payload and "disabled" in payload:
            payload["enabled"] = not bool(payload["disabled"])
        if "rescale" not in payload:
            for alias in ("rescale_alpha", "alpha_scale", "alpha_ratio", "scale"):
                if alias in payload:
                    payload["rescale"] = payload[alias]
                    break
        if "is_norm" not in payload:
            for alias in ("norm_source", "norm"):
                if alias in payload:
                    payload["is_norm"] = payload[alias]
                    break
        if "base_model" not in payload and "base" in payload:
            payload["base_model"] = payload["base"]

        known = cls.__dataclass_fields__.keys()
        filtered = {k: v for k, v in payload.items() if k in known}
        if "path" not in filtered:
            raise click.BadParameter("each loramerge source requires path, artifact, or model")
        return cls(**filtered)

    @property
    def source_path(self) -> Path:
        return Path(self.path)

    @property
    def is_io_source(self) -> bool:
        return self.is_embedding or self.is_lm_head

    @property
    def is_delta_source(self) -> bool:
        return not self.is_io_source and self.kind not in MODEL_SOURCE_KINDS

    def use_layer_index(self, layer_index: int) -> bool:
        if self.start_layer > layer_index:
            return False
        if self.end_layer != -1 and self.end_layer < layer_index:
            return False
        return True

    def module_of(self, tensor_name: str) -> Optional[str]:
        if not tensor_name.endswith(".weight"):
            return None
        return tensor_name[: -len(".weight")].split(".")[-1]

    def is_target(self, tensor_name: str) -> bool:
        if self.target_modules is None:
            return True
        module = self.module_of(tensor_name)
        return module in self.target_modules

    def provides_tensor(self, tensor_name: str) -> bool:
        if self.is_embedding and is_embedding_tensor(tensor_name):
            return True
        if self.is_lm_head and is_lm_head_tensor(tensor_name):
            return True
        return False

    def load(self) -> WeightedDeltaSource:
        if not self.is_delta_source:
            raise ValueError(f"{self.path} is an IO passthrough source, not a delta source")
        source = load_delta_source(self.source_path, kind=self.kind, name=self.name)
        return WeightedDeltaSource(source=source, weight=self.weight, rescale=self.rescale,
                                   is_norm=self.is_norm, name=self.name, post_merge=self.post_merge,
                                   group=self.group, pool=self.pool, pool_rank=self.pool_rank,
                                   pool_energy=self.pool_energy, pool_residual=self.pool_residual,
                                   priority=self.priority)


@dataclass
class LoraMergeConfig:
    """Config for merging PEFT and dense ExLoRA artifacts in fp32 full frame."""

    base_model: str
    output_dir: str
    sources: list[LoraMergeSourceConfig]

    merge_rule: MergeRule = "fourier_slerp"
    arithmetic_dtype: str = "float32"
    output_dtype: str = "bfloat16"
    scaffold_format: str = "nvfp4_residual"
    target_modules: list[str] = field(default_factory=lambda: list(DEFAULT_TARGET_MODULES))
    fourier_cutoff_pct: float = 0.08
    fourier_cull_pct: float = 0.0
    fourier_small_norm_ratio: float = 0.1
    final_merge_t: float = 0.5    # post-source strength (0 = ignore, 1 = full post). Under
                                  # isometric merging it scales each additive post overlay.
    isometric_rank: int = 32      # joint-subspace rank per source for rule=isometric.
    isometric_energy: float = 0.0 # fraction of radial magnitude to reintroduce (0 = pure rotation).

    device: str = "cpu"
    cache_dir: str = "cache"
    storage_dir: str = "storage"
    clean_cache: bool = False

    # Dense scaffold emission knobs. The first implementation only defines the
    # policy; writer integration can consume these when it emits artifacts.
    nvfp4_extra_bpe: float = 0.0
    nvfp4_residual_dtype: str = "bfloat16"
    nvfp4_index_bits: int = 32
    nvfp4_selection_policy: str = "residual_rms"
    nvfp4_floor_correction: str = "none"

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
    def arithmetic_torch_dtype(self) -> torch.dtype:
        return getattr(torch, self.arithmetic_dtype)

    @property
    def output_torch_dtype(self) -> torch.dtype:
        return getattr(torch, self.output_dtype)

    def update(self, config: dict[str, Any] = {}, **kwargs: Any) -> None:
        for source in (config, kwargs):
            for key, value in source.items():
                if value is not None and hasattr(self, key):
                    setattr(self, key, value)

    def is_target(self, tensor_name: str) -> bool:
        if not tensor_name.endswith(".weight"):
            return False
        module = tensor_name[: -len(".weight")].split(".")[-1]
        return module in self.target_modules

    def ordered_delta_sources(self, *, enable_all: bool = False) -> list[LoraMergeSourceConfig]:
        """Enabled delta sources in merge/layering order.

        Body sources fold into the iterative merge left-to-right, so this order
        *is* the layering order: highest ``priority`` first (anchors the
        accumulator), then descending (``sorted`` is stable, so equal-priority
        sources keep config order). ``post_merge`` sources always come last --
        they are blended against the final body delta as a separate step.
        """
        selected = [
            source
            for source in self.sources
            if source.is_delta_source and (enable_all or source.enabled)
        ]
        body = sorted([s for s in selected if not s.post_merge], key=lambda s: -s.priority)
        post = sorted([s for s in selected if s.post_merge], key=lambda s: -s.priority)
        return body + post

    @property
    def embedding_source(self) -> LoraMergeSourceConfig | None:
        return next((source for source in self.sources if source.enabled and source.is_embedding), None)

    @property
    def lm_head_source(self) -> LoraMergeSourceConfig | None:
        return next((source for source in self.sources if source.enabled and source.is_lm_head), None)

    @property
    def embedding_model(self) -> str:
        """Model path to use for token embeddings.

        Embeddings default to the merge base. A source marked ``is_embedding``
        overrides this for old merge-app compatibility and for tokenizer
        transplant workflows.
        """
        source = self.embedding_source
        return source.path if source is not None else self.base_model

    @property
    def lm_head_model(self) -> str:
        """Model path to use for ``lm_head``.

        The head defaults to the merge base, but head-only models can mark a
        source as ``is_lm_head``/``is_output`` to replace it.
        """
        source = self.lm_head_source
        return source.path if source is not None else self.base_model

    def passthrough_source_for(self, tensor_name: str) -> LoraMergeSourceConfig | None:
        if is_embedding_tensor(tensor_name):
            return self.embedding_source
        if is_lm_head_tensor(tensor_name):
            return self.lm_head_source
        return None

    def passthrough_model_for(self, tensor_name: str) -> str | None:
        if is_embedding_tensor(tensor_name):
            return self.embedding_model
        if is_lm_head_tensor(tensor_name):
            return self.lm_head_model
        return None

    def load_sources(self) -> list[WeightedDeltaSource]:
        return [source.load() for source in self.sources if source.enabled and source.is_delta_source]

    @classmethod
    def from_yaml(cls, config_path: Path) -> "LoraMergeConfig":
        with open(config_path) as f:
            config = yaml.safe_load(f) or {}

        required = ["base_model", "output_dir", "sources"]
        missing = [field_name for field_name in required if field_name not in config]
        if missing:
            raise click.BadParameter(
                f"Missing required configuration fields: {', '.join(missing)}"
            )

        if not isinstance(config["sources"], list) or not config["sources"]:
            raise click.BadParameter("sources must be a non-empty list")

        sources: list[LoraMergeSourceConfig] = []
        for idx, source_config in enumerate(config["sources"]):
            if not isinstance(source_config, dict):
                raise click.BadParameter(f"sources[{idx}] must be a mapping")
            source = LoraMergeSourceConfig.from_dict(source_config)
            if source.kind is not None and source.kind not in SOURCE_KINDS:
                raise click.BadParameter(f"unknown source kind {source.kind!r}")
            if source.target_modules is not None and not isinstance(source.target_modules, list):
                raise click.BadParameter("source target_modules must be a list")
            if source.is_io_source:
                if source.kind is not None and source.kind not in MODEL_SOURCE_KINDS:
                    raise click.BadParameter("embedding/lm_head sources must use kind 'model' or omit kind")
            elif source.kind in MODEL_SOURCE_KINDS:
                raise click.BadParameter("model sources must be marked is_embedding or is_lm_head")
            sources.append(source)

        if sum(1 for source in sources if source.enabled and source.is_embedding) > 1:
            raise click.BadParameter("only one embedding source may be marked")
        if sum(1 for source in sources if source.enabled and source.is_lm_head) > 1:
            raise click.BadParameter("only one lm_head source may be marked")

        known = cls.__dataclass_fields__.keys()
        filtered = {k: v for k, v in config.items() if k in known}
        filtered["sources"] = sources
        cfg = cls(**filtered)

        if cfg.merge_rule not in MERGE_RULES:
            raise click.BadParameter(f"merge_rule must be one of {sorted(MERGE_RULES)}")
        if cfg.arithmetic_dtype != "float32":
            raise click.BadParameter("loramerge arithmetic_dtype must be float32")
        if cfg.scaffold_format not in SCAFFOLD_FORMATS:
            raise click.BadParameter(f"scaffold_format must be one of {sorted(SCAFFOLD_FORMATS)}")
        if "target_modules" in config and not isinstance(config["target_modules"], list):
            raise click.BadParameter("target_modules must be a list")
        if not 0 <= cfg.fourier_cutoff_pct <= 1:
            raise click.BadParameter("fourier_cutoff_pct must be in [0, 1]")
        if not 0 <= cfg.fourier_cull_pct <= 1:
            raise click.BadParameter("fourier_cull_pct must be in [0, 1]")
        if cfg.fourier_small_norm_ratio < 0:
            raise click.BadParameter("fourier_small_norm_ratio must be non-negative")
        if cfg.nvfp4_extra_bpe < 0:
            raise click.BadParameter("nvfp4_extra_bpe must be non-negative")
        if cfg.nvfp4_index_bits not in (0, 16, 32, 64):
            raise click.BadParameter("nvfp4_index_bits must be one of 0, 16, 32, or 64")
        if cfg.nvfp4_selection_policy not in ("residual_rms", "floor_effect"):
            raise click.BadParameter("nvfp4_selection_policy must be 'residual_rms' or 'floor_effect'")
        if cfg.nvfp4_floor_correction not in ("none", "code_position_mean"):
            raise click.BadParameter("nvfp4_floor_correction must be 'none' or 'code_position_mean'")
        for source in cfg.sources:
            if source.pool_rank is not None and source.pool_rank < 1:
                raise click.BadParameter("pool_rank must be positive")
            if source.pool_energy is not None and source.pool_energy < 0:
                raise click.BadParameter("pool_energy must be non-negative")
            if source.pool_residual is not None and not 0 <= source.pool_residual <= 1:
                raise click.BadParameter("pool_residual must be in [0, 1]")
        return cfg
