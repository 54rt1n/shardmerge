# shard/ranker/codec_floor.py
# Copyright (C) 2026 Martin Bukowski
#
# This software is free software: you can redistribute it and/or
# modify it under the terms of the GNU Lesser General Public License as
# published by the Free Software Foundation, either version 3 of the
# License, or (at your option) any later version.

"""Codec-noise floor level sets for dense low-bit residual maps."""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch


@dataclass(frozen=True)
class CodecFloorConfig:
    """Robust level-set detector for blockwise codec error.

    ``floor_tolerance`` controls the median/MAD band treated as the background
    floor. ``effect_threshold`` is the number of robust-scale units above the
    floor required before a block is considered super-floor signal.
    """

    floor_tolerance: float = 1.0
    effect_threshold: float = 2.0
    min_floor_blocks: int = 8
    min_floor_fraction: float = 0.05
    eps: float = 1e-12

    def validate(self) -> None:
        if self.floor_tolerance < 0:
            raise ValueError("floor_tolerance must be non-negative")
        if self.effect_threshold < 0:
            raise ValueError("effect_threshold must be non-negative")
        if self.min_floor_blocks < 0:
            raise ValueError("min_floor_blocks must be non-negative")
        if not 0 <= self.min_floor_fraction <= 1:
            raise ValueError("min_floor_fraction must be in [0, 1]")
        if self.eps <= 0:
            raise ValueError("eps must be positive")


@dataclass(frozen=True)
class CodecFloorModel:
    """Detected floor plus per-block effect sizes."""

    floor_level: float
    floor_width: float
    robust_scale: float
    floor_mask: torch.Tensor
    super_floor_mask: torch.Tensor
    effect_scores: torch.Tensor

    @property
    def floor_blocks(self) -> int:
        return int(self.floor_mask.to(torch.bool).sum().item())

    @property
    def super_floor_blocks(self) -> int:
        return int(self.super_floor_mask.to(torch.bool).sum().item())

    @classmethod
    def fit(
        cls,
        block_errors: torch.Tensor,
        config: CodecFloorConfig | None = None,
    ) -> "CodecFloorModel":
        """Fit a robust background floor to a 1D block-error vector."""
        cfg = config or CodecFloorConfig()
        cfg.validate()
        if block_errors.ndim != 1:
            raise ValueError("block_errors must be a 1D tensor")

        scores = block_errors.detach().to(torch.float32).flatten()
        n = scores.numel()
        if n == 0:
            empty = torch.zeros(0, dtype=torch.bool, device=scores.device)
            return cls(0.0, 0.0, cfg.eps, empty, empty, scores)

        center = torch.median(scores)
        abs_dev = (scores - center).abs()
        mad = torch.median(abs_dev)
        robust_scale = torch.clamp(1.4826 * mad, min=max(float(center.abs().item()) * 1e-6, cfg.eps))
        band = cfg.floor_tolerance * robust_scale
        floor_mask = abs_dev <= band

        min_floor = max(
            int(cfg.min_floor_blocks),
            int(math.ceil(cfg.min_floor_fraction * n)),
        )
        min_floor = min(min_floor, n)
        if int(floor_mask.sum().item()) < min_floor:
            nearest = torch.topk(abs_dev, k=min_floor, largest=False, sorted=False).indices
            floor_mask = torch.zeros(n, dtype=torch.bool, device=scores.device)
            floor_mask[nearest] = True

        effect_scores = (scores - center) / robust_scale
        super_floor_mask = effect_scores >= cfg.effect_threshold
        return cls(
            floor_level=float(center.item()),
            floor_width=float(band.item()),
            robust_scale=float(robust_scale.item()),
            floor_mask=floor_mask,
            super_floor_mask=super_floor_mask,
            effect_scores=effect_scores,
        )
