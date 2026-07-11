# shard/delta/terms.py
# Copyright (C) 2026 Martin Bukowski
#
# This software is free software: you can redistribute it and/or
# modify it under the terms of the GNU Lesser General Public License as
# published by the Free Software Foundation, either version 3 of the
# License, or (at your option) any later version.

"""Stored correction terms for one tensor delta."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import torch


class DeltaTerm(Protocol):
    """A stored correction term that materializes a dense tensor delta."""

    def materialize(
        self,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> torch.Tensor:
        """Return the correction as a dense tensor."""
        ...


def _move(tensor: torch.Tensor, device: torch.device | str | None, dtype: torch.dtype | None) -> torch.Tensor:
    if device is None and dtype is None:
        return tensor
    kwargs = {}
    if device is not None:
        kwargs["device"] = device
    if dtype is not None:
        kwargs["dtype"] = dtype
    return tensor.to(**kwargs)


@dataclass(frozen=True)
class DenseDelta:
    """A dense stored delta term, typically a decoded low-bit map."""

    value: torch.Tensor

    @property
    def shape(self) -> torch.Size:
        return self.value.shape

    def materialize(
        self,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> torch.Tensor:
        return _move(self.value, device, dtype)


@dataclass(frozen=True)
class LoraDelta:
    """A LoRA-style additive correction ``scale * (B @ A)``."""

    lora_A: torch.Tensor
    lora_B: torch.Tensor
    scale: float = 1.0

    def __post_init__(self) -> None:
        if self.lora_A.ndim != 2 or self.lora_B.ndim != 2:
            raise ValueError("LoRA factors must be 2D")
        if self.lora_A.shape[0] != self.lora_B.shape[1]:
            raise ValueError(
                "LoRA rank mismatch: "
                f"A={tuple(self.lora_A.shape)} B={tuple(self.lora_B.shape)}"
            )

    @property
    def rank(self) -> int:
        return int(self.lora_A.shape[0])

    @property
    def shape(self) -> torch.Size:
        return torch.Size((self.lora_B.shape[0], self.lora_A.shape[1]))

    def materialize(
        self,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> torch.Tensor:
        work_dtype = dtype or torch.float32
        a = self.lora_A.to(device=device, dtype=work_dtype) if device is not None else self.lora_A.to(work_dtype)
        b = self.lora_B.to(device=device, dtype=work_dtype) if device is not None else self.lora_B.to(work_dtype)
        return self.scale * (b @ a)


@dataclass(frozen=True)
class IndexedResidualDelta:
    """Sparse flat-index residual correction for a known dense shape."""

    shape: tuple[int, ...]
    indices: torch.Tensor
    values: torch.Tensor

    def __post_init__(self) -> None:
        if len(self.shape) < 1:
            raise ValueError("shape must be non-empty")
        if self.indices.ndim != 1:
            raise ValueError("indices must be 1D")
        if self.values.ndim != 1:
            raise ValueError("values must be 1D")
        if self.indices.numel() != self.values.numel():
            raise ValueError("indices and values must have the same length")

    def materialize(
        self,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> torch.Tensor:
        out_dtype = dtype or self.values.dtype
        out = torch.zeros(self.shape, device=device, dtype=out_dtype)
        flat = out.flatten()
        idx = self.indices.to(device=out.device, dtype=torch.long)
        vals = self.values.to(device=out.device, dtype=out_dtype)
        flat.index_add_(0, idx, vals)
        return out
