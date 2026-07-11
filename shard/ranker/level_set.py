# shard/ranker/level_set.py
# Copyright (C) 2026 Martin Bukowski
#
# This software is free software: you can redistribute it and/or
# modify it under the terms of the GNU Lesser General Public License as
# published by the Free Software Foundation, either version 3 of the
# License, or (at your option) any later version.

"""Detect rank level sets above codec or merge-noise floors.

The detector is intentionally spectrum-only. Validation harnesses may compare
the selected rank against a known signal, but the rank decision itself only sees
singular values from the noisy merged delta.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Optional

import torch

from shard.exlora.svd import relative_errors, retained_energy


@dataclass(frozen=True)
class RankLevelSetConfig:
    """Configuration for spectrum-only rank level-set detection.

    ``window`` is the number of singular values used to decide that the spectrum
    has entered a local floor band. ``level_tolerance`` is a multiplicative
    tolerance: a window is floor-like when ``S_r / S_{r+window}`` is no more
    than ``1 + level_tolerance``. ``min_head_to_floor_ratio`` prevents smooth,
    slowly decaying spectra from being mistaken for a low-rank head.
    """

    min_rank: int = 1
    max_rank: Optional[int] = None
    search_start: int = 8
    window: int = 64
    level_tolerance: float = 0.25
    min_head_to_floor_ratio: float = 4.0
    rank_multiple: int = 8
    round_mode: str = "floor"  # floor | nearest | ceil

    def validate(self) -> None:
        if self.min_rank < 0:
            raise ValueError("min_rank must be non-negative")
        if self.max_rank is not None and self.max_rank < 0:
            raise ValueError("max_rank must be non-negative")
        if self.search_start < 1:
            raise ValueError("search_start must be positive")
        if self.window < 1:
            raise ValueError("window must be positive")
        if self.level_tolerance < 0:
            raise ValueError("level_tolerance must be non-negative")
        if self.min_head_to_floor_ratio < 1:
            raise ValueError("min_head_to_floor_ratio must be at least 1")
        if self.rank_multiple < 1:
            raise ValueError("rank_multiple must be positive")
        if self.round_mode not in {"floor", "nearest", "ceil"}:
            raise ValueError("round_mode must be one of: floor, nearest, ceil")


@dataclass(frozen=True)
class RankLevelSetResult:
    """Detected rank and diagnostics."""

    rank: int
    reason: str
    full_rank: int
    cap: int
    candidate_rank: Optional[int]
    window_drop: Optional[float]
    head_to_floor_ratio: Optional[float]
    floor_level: float
    floor_width: float
    retained_energy: float
    rel_error: float

    @property
    def found_level_set(self) -> bool:
        return self.reason == "level_set"


@dataclass(frozen=True)
class SignalScore:
    """Validation score against a known signal tensor."""

    rel_error: float
    cosine: float


@dataclass(frozen=True)
class SignalRecoveryGates:
    """Pass/fail gates for validation against a known signal."""

    max_rank_slack: int = 0
    min_cosine: float = 0.95
    max_error_regret: float = 0.03
    min_rank95_ratio: float = 4.0

    def check(
        self,
        *,
        detected_rank: int,
        reference_rank: int,
        score: SignalScore,
        reference_score: SignalScore,
        rank95: int,
    ) -> tuple[bool, list[str]]:
        failures: list[str] = []
        if detected_rank > reference_rank + self.max_rank_slack:
            failures.append(f"rank {detected_rank} > reference rank {reference_rank} + slack {self.max_rank_slack}")
        if score.cosine < self.min_cosine:
            failures.append(f"cosine {score.cosine:.4f} < {self.min_cosine:.4f}")
        if score.rel_error > reference_score.rel_error + self.max_error_regret:
            failures.append(
                f"signal error {score.rel_error:.4f} > reference {reference_score.rel_error:.4f} "
                f"+ regret {self.max_error_regret:.4f}"
            )
        if self.min_rank95_ratio > 0 and detected_rank > 0 and rank95 < self.min_rank95_ratio * detected_rank:
            failures.append(
                f"rank@95 {rank95} < {self.min_rank95_ratio:.2f} * detected rank {detected_rank}"
            )
        return not failures, failures


def _round_rank(rank: int, multiple: int, mode: str) -> int:
    if multiple <= 1:
        return rank
    if mode == "floor":
        return max(multiple, (rank // multiple) * multiple)
    if mode == "ceil":
        return int(math.ceil(rank / multiple) * multiple)
    return int(round(rank / multiple) * multiple)


def _mad(values: torch.Tensor) -> tuple[float, float]:
    if values.numel() == 0:
        return 0.0, 0.0
    center = torch.median(values)
    width = torch.median((values - center).abs())
    return float(center.item()), float(width.item())


def required_rank_for_energy(singular_values: torch.Tensor, energy: float) -> int:
    """Smallest rank retaining ``energy`` fraction of squared singular mass."""
    if not 0 <= energy <= 1:
        raise ValueError("energy must be in [0, 1]")
    values = singular_values.detach().to(device="cpu", dtype=torch.float64)
    if values.ndim != 1:
        raise ValueError("singular_values must be a 1D tensor")
    sq = values.square()
    total = sq.sum()
    if len(sq) == 0 or total <= 0:
        return 0
    keep = torch.cumsum(sq, dim=0) / total
    hits = (keep >= energy).nonzero()
    return int(hits[0].item()) + 1 if len(hits) else len(sq)


def detect_rank_level_set(
    singular_values: torch.Tensor,
    config: RankLevelSetConfig | None = None,
) -> RankLevelSetResult:
    """Detect the first stable floor level set in a singular spectrum.

    A result with ``reason == "level_set"`` means the spectrum has a separated
    head and a locally flat floor band. A result with ``reason ==
    "no_level_set"`` returns the configured cap; callers should treat that as a
    diagnostic, not proof of compressibility.
    """
    cfg = config or RankLevelSetConfig()
    cfg.validate()
    values = singular_values.detach().to(device="cpu", dtype=torch.float64).flatten()
    if singular_values.ndim != 1:
        raise ValueError("singular_values must be a 1D tensor")

    full_rank = int(values.numel())
    if full_rank == 0:
        return RankLevelSetResult(0, "empty", 0, 0, 0, 0.0, None, 0.0, 0.0, 1.0, 0.0)

    cap = full_rank if cfg.max_rank is None else min(int(cfg.max_rank), full_rank)
    cap = max(0, cap)
    floor_rank = max(0, min(int(cfg.min_rank), cap))
    if cap == 0:
        return RankLevelSetResult(0, "empty", full_rank, 0, 0, 0.0, None, 0.0, 0.0, 1.0, 0.0)

    work = values[:cap].clamp_min(0.0)
    if torch.all(work <= 0):
        rank = max(1, floor_rank)
        return RankLevelSetResult(
            rank=rank,
            reason="zero_spectrum",
            full_rank=full_rank,
            cap=cap,
            candidate_rank=rank,
            window_drop=0.0,
            head_to_floor_ratio=None,
            floor_level=0.0,
            floor_width=0.0,
            retained_energy=1.0,
            rel_error=0.0,
        )

    safe = work.clamp_min(1e-300)
    logs = safe.log()
    tail_start = max(0, int(round(0.75 * cap)))
    floor_level, floor_width = _mad(logs[tail_start:])
    max_start = cap - cfg.window
    start = max(1, min(cfg.search_start, cap))

    candidate: Optional[int] = None
    window_drop: Optional[float] = None
    head_to_floor: Optional[float] = None
    log_tolerance = math.log1p(cfg.level_tolerance)
    if max_start >= start:
        top = float(safe[0].item())
        for rank in range(start, max_start + 1):
            left = logs[rank - 1]
            right = logs[rank + cfg.window - 1]
            drop = float((left - right).item())
            ratio = top / max(float(safe[rank - 1].item()), 1e-300)
            if drop <= log_tolerance and ratio >= cfg.min_head_to_floor_ratio:
                candidate = rank
                window_drop = drop
                head_to_floor = ratio
                break

    if candidate is None:
        rank = cap
        reason = "no_level_set"
    else:
        rank = _round_rank(candidate, cfg.rank_multiple, cfg.round_mode)
        rank = max(floor_rank, min(rank, cap))
        reason = "level_set"

    return RankLevelSetResult(
        rank=int(rank),
        reason=reason,
        full_rank=full_rank,
        cap=cap,
        candidate_rank=candidate,
        window_drop=window_drop,
        head_to_floor_ratio=head_to_floor,
        floor_level=floor_level,
        floor_width=floor_width,
        retained_energy=retained_energy(values, rank),
        rel_error=float(relative_errors(values)[rank].item()),
    )


def score_signal_recovery(reconstruction: torch.Tensor, signal: torch.Tensor) -> SignalScore:
    """Relative error and cosine against a known validation signal."""
    if reconstruction.shape != signal.shape:
        raise ValueError("reconstruction and signal must have the same shape")
    recon = reconstruction.to(torch.float32)
    target = signal.to(torch.float32)
    rel = float(((recon - target).norm() / target.norm().clamp_min(1e-12)).item())
    rf = recon.flatten()
    tf = target.flatten()
    cosine = float((rf @ tf / (rf.norm() * tf.norm()).clamp_min(1e-12)).item())
    return SignalScore(rel_error=rel, cosine=cosine)
