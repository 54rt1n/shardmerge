# shard/delta/manifest.py
# Copyright (C) 2026 Martin Bukowski
#
# This software is free software: you can redistribute it and/or
# modify it under the terms of the GNU Lesser General Public License as
# published by the Free Software Foundation, either version 3 of the
# License, or (at your option) any later version.

"""Versioned manifest schema for mixed tensor-delta artifacts."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class TermSpec:
    """Manifest entry for one correction term in a tensor patch stack."""

    kind: str
    path: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "path": self.path,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "TermSpec":
        return cls(
            kind=str(payload["kind"]),
            path=payload.get("path"),
            metadata=dict(payload.get("metadata") or {}),
        )


@dataclass(frozen=True)
class TensorPatchSpec:
    """Manifest entry for all corrections attached to one tensor."""

    tensor_name: str
    shape: tuple[int, ...]
    terms: tuple[TermSpec, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "tensor_name": self.tensor_name,
            "shape": list(self.shape),
            "terms": [term.to_dict() for term in self.terms],
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "TensorPatchSpec":
        return cls(
            tensor_name=str(payload["tensor_name"]),
            shape=tuple(int(v) for v in payload["shape"]),
            terms=tuple(TermSpec.from_dict(item) for item in payload.get("terms", [])),
        )


@dataclass(frozen=True)
class PatchManifest:
    """Top-level manifest for a hybrid model-delta artifact."""

    base_model: str
    target_model: str | None
    tensors: tuple[TensorPatchSpec, ...]
    schema_version: int = 1
    artifact_type: str = "shard_delta_patch"
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "artifact_type": self.artifact_type,
            "base_model": self.base_model,
            "target_model": self.target_model,
            "metadata": dict(self.metadata),
            "tensors": [tensor.to_dict() for tensor in self.tensors],
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "PatchManifest":
        return cls(
            schema_version=int(payload.get("schema_version", 1)),
            artifact_type=str(payload.get("artifact_type", "shard_delta_patch")),
            base_model=str(payload["base_model"]),
            target_model=payload.get("target_model"),
            metadata=dict(payload.get("metadata") or {}),
            tensors=tuple(TensorPatchSpec.from_dict(item) for item in payload.get("tensors", [])),
        )
