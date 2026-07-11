# shard/delta/metrics.py
# Copyright (C) 2026 Martin Bukowski
#
# This software is free software: you can redistribute it and/or
# modify it under the terms of the GNU Lesser General Public License as
# published by the Free Software Foundation, either version 3 of the
# License, or (at your option) any later version.

"""Behavior-facing right-side metrics for tensor deltas."""

from __future__ import annotations

from dataclasses import dataclass

import torch


def metric_norm(delta: torch.Tensor, metric: torch.Tensor | None = None) -> torch.Tensor:
    """Return ``||delta||`` under an optional right-side metric ``H``.

    ``metric`` may be a full square matrix or a diagonal vector.  With
    ``metric=None`` this is the Frobenius norm.
    """
    work = delta.to(torch.float32)
    if metric is None:
        return work.norm()
    h = metric.to(device=work.device, dtype=torch.float32)
    if h.ndim == 1:
        if h.shape[0] != work.shape[1]:
            raise ValueError(f"diagonal metric length {h.shape[0]} does not match {work.shape[1]}")
        return torch.sqrt((work.square() * h.unsqueeze(0)).sum().clamp_min(0.0))
    if h.ndim == 2:
        if h.shape != (work.shape[1], work.shape[1]):
            raise ValueError(f"metric shape {tuple(h.shape)} does not match input dim {work.shape[1]}")
        return torch.sqrt(((work @ h) * work).sum().clamp_min(0.0))
    raise ValueError("metric must be None, a diagonal vector, or a square matrix")


def relative_metric_error(
    reference_delta: torch.Tensor,
    approximation_delta: torch.Tensor,
    metric: torch.Tensor | None = None,
    eps: float = 1e-12,
) -> float:
    """Return ``||reference - approximation||_H / ||reference||_H``."""
    if reference_delta.shape != approximation_delta.shape:
        raise ValueError(
            "reference and approximation shapes differ: "
            f"{tuple(reference_delta.shape)} vs {tuple(approximation_delta.shape)}"
        )
    denom = metric_norm(reference_delta, metric)
    residual = metric_norm(reference_delta - approximation_delta, metric)
    if denom.item() <= eps:
        return 0.0 if residual.item() <= eps else float("inf")
    return float((residual / denom).item())


@dataclass(frozen=True)
class GramMetric:
    """A named right-side activation Gram for one tensor."""

    tensor_name: str
    gram: torch.Tensor

    def __post_init__(self) -> None:
        if self.gram.ndim != 2 or self.gram.shape[0] != self.gram.shape[1]:
            raise ValueError("gram must be a square matrix")

    @property
    def in_features(self) -> int:
        return int(self.gram.shape[0])

    def norm(self, delta: torch.Tensor) -> torch.Tensor:
        return metric_norm(delta, self.gram)

    def relative_error(self, reference_delta: torch.Tensor, approximation_delta: torch.Tensor) -> float:
        return relative_metric_error(reference_delta, approximation_delta, self.gram)
