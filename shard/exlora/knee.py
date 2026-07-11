# shard/exlora/knee.py
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

"""Spectral-knee detector: low-rank LoRA vs genuinely-dense delta.

A merged finetune that was trained as a rank-``r`` LoRA produces a delta whose
singular spectrum is ``r`` "signal" values followed by a **flat bf16 noise
floor** -- a sharp cliff at ``r``. A genuinely dense finetune produces a smooth,
graded spectrum with no cliff. Frobenius ``r@target_error`` cannot tell these
apart (the noise floor hoards energy and drags the error-rank to ~full rank);
the **cliff** can.

This detector routes the hybrid extractor: a clean knee -> PEFT LoRA at the knee
rank; no knee -> NVFP4-dense (+ comb). It deliberately ignores the noise-floor
tail-median heuristic in ``spectrum_diagnostics`` (which is cap-dependent and
preempts the knee -- it mislabeled the rank-64 MeroMero delta as rank ~2287).
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass
from typing import Optional, Sequence

import torch

# coarse geometric rank grid -- snap detected knees so a cliff that lands at
# 64/65/66 across tensors votes as one bucket instead of splitting.
_RANK_GRID = (1, 2, 4, 8, 16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512)


def _snap_rank(r: int) -> int:
    return min(_RANK_GRID, key=lambda g: abs(math.log(g) - math.log(max(r, 1))))


@dataclass(frozen=True)
class KneeDetection:
    """Result of spectral-knee detection for one tensor's singular spectrum."""

    is_low_rank: bool
    knee_rank: Optional[int]      # signal rank at the cliff (None if dense)
    gap_ratio: float              # sv[knee-1] / sv[knee] -- cliff height
    floor_ratio: float            # sv[knee] / median(sv below knee) -- ~1 = cliff lands on a flat floor
    rel_floor: float              # floor median / sv[0]
    reason: str                   # "knee" | "no_cliff" | "graded_tail" | "high_floor" | "empty"
    energy_at_knee: float = 0.0   # fraction of squared-SV energy kept by rank `knee`
    path: str = ""                # acceptance path: "flat_floor" | "energy" | ""


def detect_signal_knee(
    singular_values: torch.Tensor,
    min_rank: int = 1,
    max_rank: Optional[int] = None,
    gap_threshold: float = 1.4,
    floor_ratio_max: float = 3.0,
    energy_threshold: float = 0.85,
    min_energy_at_knee: float = 0.25,
    rel_floor_max: float = 0.35,
    floor_fraction: float = 0.6,
    search_fraction: float = 0.5,
) -> KneeDetection:
    """Detect a low-rank signal cliff in a singular spectrum.

    Finds the largest adjacent singular-value drop (cliff) in
    ``[min_rank, max_rank]``. The cliff is a genuine low-rank knee iff it is
    sharp (``gap_ratio >= gap_threshold``) AND the stuff below it is
    *discardable*, which happens two ways:

    * **flat noise floor** -- the cliff lands on a flat floor
      (``floor_ratio <= floor_ratio_max``); the bf16-merge LoRA case (MeroMero,
      Gutenberg) where the floor hoards energy but is inert.
    * **negligible tail** -- the retained rank already captures the energy
      (``energy_at_knee >= energy_threshold``); the directly-low-rank case
      (heretic's rank-1 abliteration, whose tail is a small graded shoulder,
      not a flat floor).

    ``min_rank=1`` so a rank-1 modification (the cliff between rank 1 and 2) is
    visible. A dense/graded spectrum has no sharp cliff and reports no knee.
    """
    s = singular_values.detach().to(device="cpu", dtype=torch.float64)
    s = s[s > 0]
    k = int(s.numel())
    if k < 2:
        return KneeDetection(False, None, 1.0, float("inf"), 0.0, "empty", 0.0)

    s0 = float(s[0])
    total_energy = float((s ** 2).sum())
    lo = max(1, min_rank)
    hi = max_rank if max_rank is not None else int(k * search_fraction)
    hi = max(lo + 1, min(hi, k - 1))

    ratios = s[:-1] / s[1:].clamp_min(1e-300)          # ratios[i] = sv[i]/sv[i+1], cliff *after* rank i+1
    window = ratios[lo - 1:hi]
    if window.numel() == 0:
        return KneeDetection(False, None, 1.0, float("inf"), 0.0, "empty", 0.0)

    j = int(torch.argmax(window).item())
    knee_rank = lo + j                                  # cliff drops between rank knee_rank and knee_rank+1
    gap = float(ratios[knee_rank - 1].item())

    # Estimate the noise floor from the DEEP tail of the full spectrum (cap-
    # independent), not just below the knee -- otherwise a graded body's median
    # is its midpoint and looks "flat", so an outlier sv[0] (massive-activation
    # direction in a dense model) fakes a rank-1 knee.
    deep_start = min(max(knee_rank, int(k * floor_fraction)), k - 1)
    floor = float(s[deep_start:].median().item())
    floor_ratio = float(s[knee_rank].item()) / max(floor, 1e-300)
    rel_floor = floor / max(s0, 1e-300)
    energy_at_knee = float((s[:knee_rank] ** 2).sum()) / max(total_energy, 1e-300)

    # The discarded tail is OK to drop two ways: it's a flat noise floor (cliff
    # lands near the global floor) AND the retained rank captures meaningful
    # energy (rules out an outlier sv[0] spiking over a dense body); OR the tail
    # is simply negligible (retained rank already holds ~all the energy).
    flat_floor = floor_ratio <= floor_ratio_max and energy_at_knee >= min_energy_at_knee
    negligible_tail = energy_at_knee >= energy_threshold

    if gap < gap_threshold:
        reason = "no_cliff"
    elif rel_floor > rel_floor_max:
        reason = "high_floor"          # the floor isn't well below the signal
    elif negligible_tail:
        # signal-dominated: retained rank holds ~all the energy (e.g. abliteration)
        return KneeDetection(True, knee_rank, gap, floor_ratio, rel_floor, "knee", energy_at_knee, "energy")
    elif flat_floor:
        # cliff lands on a flat noise floor (bf16-merge LoRA); energy is in the floor
        return KneeDetection(True, knee_rank, gap, floor_ratio, rel_floor, "knee", energy_at_knee, "flat_floor")
    else:
        reason = "graded_tail"
    return KneeDetection(False, None, gap, floor_ratio, rel_floor, reason, energy_at_knee)


def select_extraction_rank(
    singular_values: torch.Tensor,
    detection: KneeDetection,
    energy_target: float = 0.94,
    max_rank: int = 512,
) -> int:
    """Choose the rank to actually extract, keyed on the acceptance path.

    * **flat_floor** (bf16-merge LoRA, e.g. MeroMero): return the knee rank. The
      discarded tail is noise, so a Frobenius energy target is meaningless here
      (it would chase thousands of noise modes); the knee *is* the signal.
    * **energy** (signal-dominated, e.g. heretic abliteration): climb the
      shoulder to ``energy_target`` cumulative energy (>= the knee rank). The
      knee is the dominant direction; a couple more ranks reach the target.

    Returns 0 when the tensor isn't low-rank.
    """
    if not detection.is_low_rank or detection.knee_rank is None:
        return 0
    if detection.path != "energy":
        return min(detection.knee_rank, max_rank)
    s = singular_values.detach().to(dtype=torch.float64)
    s = s[s > 0]
    if s.numel() == 0:
        return min(detection.knee_rank, max_rank)
    cum = torch.cumsum(s ** 2, 0) / (s ** 2).sum()
    hit = (cum >= energy_target).nonzero()
    rank = int(hit[0].item()) + 1 if len(hit) else int(s.numel())
    return min(max(rank, detection.knee_rank), max_rank)


@dataclass(frozen=True)
class SpectrumVerdict:
    """Model-level routing verdict from per-tensor knee detections."""

    kind: str                     # "low_rank" | "dense"
    rank: Optional[int]           # modal knee rank if low_rank
    knee_count: int
    concentration: float          # fraction of knees at the modal rank


def verdict_from_knees(
    detections: Sequence[KneeDetection],
    min_knees: int = 3,
    min_support: int = 2,
    max_rank: int = 512,
) -> SpectrumVerdict:
    """Aggregate per-tensor detections into a model verdict.

    Knees are snapped to a coarse rank grid and bucketed. The LoRA rank is the
    **largest** grid bucket with ``>= min_support`` knees: a low-rank delta with
    outlier leading directions (Gutenberg shows rank-1 spikes on some tensors
    *and* its true rank-64 cliff) must be sized at the larger, inclusive rank,
    since rank-64 captures the rank-1 directions too. A lone spurious knee has no
    support and is ignored. Dense models produce no knees.
    """
    ranks = [d.knee_rank for d in detections if d.is_low_rank and d.knee_rank is not None]
    if len(ranks) < min_knees:
        return SpectrumVerdict("dense", None, len(ranks), 0.0)
    buckets = Counter(_snap_rank(r) for r in ranks)
    supported = [b for b, c in buckets.items() if c >= min_support and b <= max_rank]
    if not supported:
        return SpectrumVerdict("dense", None, len(ranks), 0.0)
    rank = max(supported)
    concentration = buckets[rank] / len(ranks)
    return SpectrumVerdict("low_rank", rank, len(ranks), concentration)


@dataclass(frozen=True)
class ModelClass:
    """Top-level routing class for a whole finetune."""

    kind: str                     # low_rank | sparse_low_rank | dense | head_only | unchanged
    rank: Optional[int]
    detail: str


def classify_delta_map(
    changed_detections: Sequence[KneeDetection],
    n_unchanged: int,
    untied_head: bool = False,
    embed_changed: bool = False,
    sparse_max_changed_frac: float = 0.5,
) -> ModelClass:
    """Classify a finetune from its per-tensor change map.

    ``changed_detections`` are knee detections for the tensors that moved;
    ``n_unchanged`` counts the (sampled) tensors that didn't. ``untied_head`` is
    True when the finetune carries an ``lm_head`` the base lacks (or a changed
    head); ``embed_changed`` when the embeddings moved. Distinguishes:
    all-tensor low-rank LoRA, sparse low-rank (a few changed tensors, e.g.
    abliteration), dense, head-only, and unchanged.
    """
    n_changed = len(changed_detections)
    total = n_changed + n_unchanged

    if n_changed == 0:
        if untied_head or embed_changed:
            return ModelClass("head_only", None, "untied/retrained head; body frozen")
        return ModelClass("unchanged", None, "no body delta")

    verdict = verdict_from_knees(changed_detections)
    if verdict.kind == "low_rank":
        if total and n_changed <= sparse_max_changed_frac * total:
            return ModelClass("sparse_low_rank", verdict.rank,
                              f"{n_changed}/{total} tensors changed, rank {verdict.rank}")
        return ModelClass("low_rank", verdict.rank, f"all-tensor rank {verdict.rank}")
    return ModelClass("dense", None, f"{n_changed}/{total} changed, no knee")
