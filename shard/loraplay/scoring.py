# shard/loraplay/scoring.py
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

"""Per-layer scoring of a LoRA against the true finetune delta.

The behaviorally relevant error of a LoRA on a linear layer is the change it
induces in the layer *output* on real inputs, not the weight-space Frobenius
error. With ``H = XᵀX`` the input second moment and ``C = H^{1/2}``:

    output_error = ‖ X·(Δ_true − Δ_lora)ᵀ ‖_F = ‖ (Δ_true − Δ_lora)·C ‖_F

so the relative activation error is ``‖E·C‖ / ‖Δ_true·C‖``. This module is pure
tensor math (no model I/O); the activations come from :mod:`activations`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional
import statistics

import torch

from ..exlora.functions import relative_frobenius_error, weighted_relative_error, symmetric_psd_sqrt


def metric_sqrt_from_gram(gram: torch.Tensor, damping: float = 1e-5) -> torch.Tensor:
    """Return ``C = (H + λI)^{1/2}`` for an input second-moment matrix ``H``."""
    if gram.ndim != 2 or gram.shape[0] != gram.shape[1]:
        raise ValueError(f"gram must be square, got {tuple(gram.shape)}")
    lam = damping * gram.diagonal().mean().clamp_min(1e-12).item()
    return symmetric_psd_sqrt(gram, damping=lam)


@dataclass
class LayerScore:
    """Reconstruction scores for one layer's LoRA."""

    layer_name: str
    rank: int
    out_features: int
    in_features: int
    raw_rel_error: float                 # ‖Δ_true − Δ_lora‖_F / ‖Δ_true‖_F
    act_rel_error: Optional[float]        # ‖E·C‖_F / ‖Δ_true·C‖_F (None if no metric)
    nbytes: int                          # bf16 LoRA storage for this layer

    @property
    def behavior_error(self) -> float:
        """Activation error when available, else the raw Frobenius error."""
        return self.act_rel_error if self.act_rel_error is not None else self.raw_rel_error


def score_layer(
    layer_name: str,
    delta_true: torch.Tensor,
    delta_lora: torch.Tensor,
    rank: int,
    metric_sqrt: Optional[torch.Tensor] = None,
) -> LayerScore:
    """Score a single layer's LoRA against the true delta.

    ``metric_sqrt`` (``C = H^{1/2}``) enables the activation-weighted error; when
    omitted only the raw Frobenius error is reported.
    """
    if delta_true.shape != delta_lora.shape:
        raise ValueError(
            f"{layer_name}: delta shape mismatch {tuple(delta_true.shape)} vs {tuple(delta_lora.shape)}"
        )
    raw = relative_frobenius_error(delta_true, delta_lora)
    act = (
        weighted_relative_error(delta_true, delta_lora, metric_sqrt)
        if metric_sqrt is not None else None
    )
    out_f, in_f = delta_true.shape
    nbytes = rank * (out_f + in_f) * 2  # bf16 A + B
    return LayerScore(layer_name, rank, out_f, in_f, raw, act, nbytes)


@dataclass
class ScoreReport:
    """Aggregate scores across all scored layers."""

    layers: List[LayerScore]

    @property
    def total_bytes(self) -> int:
        return sum(s.nbytes for s in self.layers)

    def _agg(self, key) -> dict:
        vals = [key(s) for s in self.layers if key(s) is not None]
        if not vals:
            return {}
        return {
            "mean": statistics.mean(vals),
            "median": statistics.median(vals),
            "worst": max(vals),
        }

    def summary(self) -> dict:
        return {
            "layers": len(self.layers),
            "total_mb": self.total_bytes / 1e6,
            "raw_rel_error": self._agg(lambda s: s.raw_rel_error),
            "act_rel_error": self._agg(lambda s: s.act_rel_error),
            "behavior_error": self._agg(lambda s: s.behavior_error),
            # behavior error per MB is the trajectory-residence-per-byte proxy
            "mean_behavior_error_per_mb": (
                statistics.mean([s.behavior_error for s in self.layers])
                / max(self.total_bytes / 1e6, 1e-9)
            ) if self.layers else None,
        }
