# shard/delta/codecs.py
# Copyright (C) 2026 Martin Bukowski
#
# This software is free software: you can redistribute it and/or
# modify it under the terms of the GNU Lesser General Public License as
# published by the Free Software Foundation, either version 3 of the
# License, or (at your option) any later version.

"""Codec protocol for delta-map terms."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import torch

from .terms import DeltaTerm, DenseDelta


class DeltaCodec(Protocol):
    """Encodes a dense delta into a stored correction term."""

    name: str

    def encode(self, delta: torch.Tensor) -> DeltaTerm:
        """Encode ``delta`` into a materializable term."""
        ...


@dataclass(frozen=True)
class IdentityDeltaCodec:
    """Stores the delta directly. Useful as a baseline and test codec."""

    name: str = "identity"

    def encode(self, delta: torch.Tensor) -> DenseDelta:
        return DenseDelta(delta.detach().clone())


@dataclass(frozen=True)
class RoundToDTypeCodec:
    """Simple precision-rounding codec used for tests and bf16/fp16 baselines."""

    dtype: torch.dtype
    name: str = "round_dtype"

    def encode(self, delta: torch.Tensor) -> DenseDelta:
        return DenseDelta(delta.to(self.dtype).to(delta.dtype))
