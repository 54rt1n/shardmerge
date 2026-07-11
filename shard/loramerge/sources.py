# shard/loramerge/sources.py
# Copyright (C) 2026 Martin Bukowski
#
# This software is free software: you can redistribute it and/or
# modify it under the terms of the GNU Lesser General Public License as
# published by the Free Software Foundation, either version 3 of the
# License, or (at your option) any later version.

"""Delta sources for full-frame LoRA/dense merges.

Every source exposes a tensor-local delta in the common weight-coordinate
frame. PEFT factors and dense NVFP4 maps are storage formats; merge rules see
only fp32 deltas.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Protocol, runtime_checkable

import torch
from safetensors import safe_open

from ..delta import DeltaTerm
from ..exlora.nvfp4_residual import NVFP4ResidualMap
from ..loraplay.adapter import LoadedAdapter, load_adapter


@runtime_checkable
class DeltaSource(Protocol):
    """A per-tensor delta provider in the full weight-coordinate frame."""

    name: str

    def tensor_names(self) -> set[str]:
        """Return tensor names this source can materialize."""
        ...

    def delta(
        self,
        tensor_name: str,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor | None:
        """Materialize ``W_source - W_base`` for ``tensor_name``.

        Missing tensors return ``None`` so sparse adapters can be mixed with
        dense maps without special-case merge code.
        """
        ...


def _move(tensor: torch.Tensor, device: torch.device | str | None, dtype: torch.dtype) -> torch.Tensor:
    if device is None:
        return tensor.to(dtype=dtype)
    return tensor.to(device=device, dtype=dtype)


@dataclass(frozen=True)
class PeftDeltaSource:
    """PEFT LoRA adapter materialized as dense tensor deltas."""

    adapter: LoadedAdapter
    name: str = "peft"

    @classmethod
    def from_path(cls, path: str | Path, *, name: str | None = None) -> "PeftDeltaSource":
        path = Path(path)
        return cls(load_adapter(path), name=name or path.name)

    def tensor_names(self) -> set[str]:
        return set(self.adapter.factors)

    def delta(
        self,
        tensor_name: str,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor | None:
        if tensor_name not in self.adapter.factors:
            return None
        out = self.adapter.delta(tensor_name, compute_dtype=torch.float32)
        return _move(out, device, dtype)


class CheckpointDeltaSource:
    """A full model checkpoint materialized as deltas vs base: ``W_model - W_base``.

    Lets a dense SFT / QAT checkpoint participate in a full-frame merge (or serve
    as the ``is_norm`` magnitude reference) with no pre-extracted artifact.
    Tensors load lazily per call, so a 60-120 GB model never lands in RAM at once.
    """

    def __init__(self, model_dir: str | Path, base_dir: str | Path, *, name: str = "checkpoint"):
        self.model_dir = Path(model_dir)
        self.base_dir = Path(base_dir)
        self.name = name
        self._wm_model: dict[str, str] | None = None
        self._wm_base: dict[str, str] | None = None

    def _weight_map(self, root: Path) -> dict[str, str]:
        return json.load(open(root / "model.safetensors.index.json"))["weight_map"]

    @property
    def model_map(self) -> dict[str, str]:
        if self._wm_model is None:
            self._wm_model = self._weight_map(self.model_dir)
        return self._wm_model

    @property
    def base_map(self) -> dict[str, str]:
        if self._wm_base is None:
            self._wm_base = self._weight_map(self.base_dir)
        return self._wm_base

    @staticmethod
    def _load(root: Path, wmap: dict[str, str], name: str, device) -> torch.Tensor:
        with safe_open(str(root / wmap[name]), framework="pt") as f:
            return f.get_tensor(name).to(device=device or "cpu", dtype=torch.float32)

    def tensor_names(self) -> set[str]:
        return set(self.model_map) & set(self.base_map)

    def delta(
        self,
        tensor_name: str,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor | None:
        if tensor_name not in self.model_map or tensor_name not in self.base_map:
            return None
        mt = self._load(self.model_dir, self.model_map, tensor_name, device)
        bt = self._load(self.base_dir, self.base_map, tensor_name, device)
        if mt.shape != bt.shape:
            return None
        return (mt - bt).to(dtype=dtype)


DensePayload = torch.Tensor | DeltaTerm


@dataclass(frozen=True)
class DenseDeltaSource:
    """In-memory dense source for tests and already-decoded delta maps."""

    deltas: Mapping[str, DensePayload]
    name: str = "dense"

    def tensor_names(self) -> set[str]:
        return set(self.deltas)

    def delta(
        self,
        tensor_name: str,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor | None:
        payload = self.deltas.get(tensor_name)
        if payload is None:
            return None
        if isinstance(payload, torch.Tensor):
            return _move(payload, device, dtype)
        return payload.materialize(device=device, dtype=dtype)


@dataclass(frozen=True)
class NVFP4ResidualDeltaSource:
    """Dense ExLoRA NVFP4 residual map (+ optional Gram-fit comb corrector),
    materialized/projected to a bf16/fp32 tensor delta.

    An ``nvfp4 + comb`` artifact is the dense NVFP4 map (carrier) plus a low-rank
    comb-corrector PEFT adapter (in a ``comb/`` subdir) that cancels the
    behaviorally-relevant quantization comb. The loader sums them:
    ``delta = dequant(NVFP4) + comb.B@comb.A``.
    """

    residual_map: NVFP4ResidualMap
    comb: LoadedAdapter | None = None
    name: str = "nvfp4_residual"

    @classmethod
    def from_path(cls, path: str | Path, *, name: str | None = None) -> "NVFP4ResidualDeltaSource":
        path = Path(path)
        comb_dir = path / "comb"
        comb = load_adapter(comb_dir) if (comb_dir / "adapter_config.json").exists() else None
        return cls(NVFP4ResidualMap.load(path), comb=comb, name=name or path.name)

    def tensor_names(self) -> set[str]:
        names = set(self.residual_map.manifest.get("tensors", {}))
        if self.comb is not None:
            names |= set(self.comb.factors)
        return names

    def delta(
        self,
        tensor_name: str,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor | None:
        dense = self.residual_map.dequantize(tensor_name, device=device or "cpu", dtype=torch.float32)
        comb = None
        if self.comb is not None and tensor_name in self.comb.factors:
            comb = self.comb.delta(tensor_name, compute_dtype=torch.float32)
        if dense is None and comb is None:
            return None
        if dense is None:
            out = comb
        elif comb is None:
            out = dense
        else:
            out = dense + comb.to(dense.device)
        return out.to(dtype=dtype)


def load_delta_source(path: str | Path, *, kind: str | None = None, name: str | None = None) -> DeltaSource:
    """Load a PEFT or dense ExLoRA source from ``path``.

    ``kind`` may be ``"peft"`` or ``"dense"``/``"nvfp4_residual"``. If omitted,
    the loader detects common artifact files.
    """
    path = Path(path)
    if kind is None:
        if (path / "adapter_config.json").exists() and (path / "adapter_model.safetensors").exists():
            kind = "peft"
        elif (path / "exlora_nvfp4_residual.json").exists():
            kind = "nvfp4_residual"
        else:
            raise ValueError(f"could not detect delta source type for {path}")

    normalized = kind.lower()
    if normalized == "peft":
        return PeftDeltaSource.from_path(path, name=name)
    if normalized in {"dense", "nvfp4", "nvfp4_residual", "nvfp4_residual_delta_map"}:
        return NVFP4ResidualDeltaSource.from_path(path, name=name)
    raise ValueError(f"unknown delta source kind {kind!r}")
