# shard/ranker/layer_position.py
# Copyright (C) 2026 Martin Bukowski
#
# This software is free software: you can redistribute it and/or
# modify it under the terms of the GNU Lesser General Public License as
# published by the Free Software Foundation, either version 3 of the
# License, or (at your option) any later version.

"""Layer-position diagnostics for per-tensor compression reports."""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Iterable


_LAYER_RE = re.compile(r"(?:^|\.)(?:layers|blocks|h)\.(\d+)\.")


@dataclass(frozen=True)
class LayerPosition:
    """Parsed layer depth metadata for a tensor name."""

    layer_index: int | None
    num_layers: int | None
    depth_fraction: float | None
    depth_bucket: str


def parse_layer_index(layer_name: str) -> int | None:
    """Parse a transformer block index from common tensor-name conventions."""
    match = _LAYER_RE.search(layer_name)
    return int(match.group(1)) if match else None


def infer_num_layers(layer_names: Iterable[str]) -> int | None:
    """Infer block count as ``max(layer_index) + 1`` from tensor names."""
    indices = [idx for name in layer_names if (idx := parse_layer_index(name)) is not None]
    return max(indices) + 1 if indices else None


def bucket_depth(
    depth_fraction: float | None,
    *,
    early_fraction: float = 0.20,
    late_fraction: float = 0.75,
) -> str:
    """Map a normalized depth to early/middle/late/unknown."""
    if depth_fraction is None:
        return "unknown"
    if depth_fraction < early_fraction:
        return "early"
    if depth_fraction >= late_fraction:
        return "late"
    return "middle"


def layer_position(
    layer_name: str,
    *,
    num_layers: int | None,
    early_fraction: float = 0.20,
    late_fraction: float = 0.75,
) -> LayerPosition:
    """Return parsed position metadata for ``layer_name``."""
    idx = parse_layer_index(layer_name)
    if idx is None or num_layers is None:
        return LayerPosition(idx, num_layers, None, "unknown")
    if num_layers <= 1:
        frac = 0.0
    else:
        frac = idx / (num_layers - 1)
    return LayerPosition(
        layer_index=idx,
        num_layers=num_layers,
        depth_fraction=frac,
        depth_bucket=bucket_depth(frac, early_fraction=early_fraction, late_fraction=late_fraction),
    )
