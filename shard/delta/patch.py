# shard/delta/patch.py
# Copyright (C) 2026 Martin Bukowski
#
# This software is free software: you can redistribute it and/or
# modify it under the terms of the GNU Lesser General Public License as
# published by the Free Software Foundation, either version 3 of the
# License, or (at your option) any later version.

"""Composition of stored tensor-delta terms."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

import torch

from .terms import DeltaTerm


@dataclass
class PatchStack:
    """Ordered additive corrections for one tensor."""

    terms: list[DeltaTerm] = field(default_factory=list)

    def append(self, term: DeltaTerm) -> None:
        self.terms.append(term)

    def extend(self, terms: Iterable[DeltaTerm]) -> None:
        self.terms.extend(terms)

    def delta_like(self, reference: torch.Tensor) -> torch.Tensor:
        """Materialize all terms as a delta with ``reference`` dtype/device."""
        out = torch.zeros_like(reference)
        for term in self.terms:
            value = term.materialize(device=reference.device, dtype=reference.dtype)
            if value.shape != reference.shape:
                raise ValueError(
                    f"patch term shape {tuple(value.shape)} does not match reference {tuple(reference.shape)}"
                )
            out = out + value
        return out

    def apply(self, base: torch.Tensor) -> torch.Tensor:
        """Return ``base + sum(terms)``."""
        return base + self.delta_like(base)


def apply_patch_stack(base: torch.Tensor, terms: Iterable[DeltaTerm]) -> torch.Tensor:
    """Convenience wrapper for one-off patch application."""
    return PatchStack(list(terms)).apply(base)
