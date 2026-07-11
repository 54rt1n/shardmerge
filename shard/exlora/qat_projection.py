# shard/exlora/qat_projection.py
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

"""Typed protocol for QAT-map projections around ExLoRA extraction.

The extraction pipeline has three mathematically distinct projection stages:

1. ``apply`` moves a dense delta into the coordinate system that should be
   factorized.
2. ``defuzz`` edits the SVD basis before rank selection, allowing a projection
   to bias retained directions toward quantization-stable structure.
3. ``fold`` maps LoRA factors back to ordinary PEFT weight coordinates.

Concrete implementations must keep these stages linear enough that the emitted
adapter still represents an update in the original model weight basis.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Mapping, Protocol, TypeAlias, runtime_checkable

import torch

from .functions import project_weighted_lora_A, right_weight_delta


ProjectionKind: TypeAlias = Literal[
    "none",
    "identity",
    "metric",
    "reflection",
    "rotation",
    "sensitivity",
    "custom",
]
SVDParts: TypeAlias = tuple[torch.Tensor, torch.Tensor, torch.Tensor]
LoraFactors: TypeAlias = tuple[torch.Tensor, torch.Tensor]


@dataclass(frozen=True)
class QATProjectionConfig:
    """Configuration needed to build a QAT projection implementation."""

    projection_type: ProjectionKind = "none"
    projection_model: Path | None = None
    compute_dtype: torch.dtype = torch.float32
    damping: float = 1e-5
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class TensorProjectionContext:
    """Optional per-tensor context available to richer projections.

    The current raw extraction path only needs ``layer_name`` and
    ``delta_shape``. Future QAT projections can attach loaded base/QAT tensors,
    activation sketches, Hessian factors, quantizer residuals, or spectral flux
    metadata without changing the core extraction method signatures.
    """

    layer_name: str
    delta_shape: tuple[int, int]
    module_name: str | None = None
    layer_index: int | None = None
    base_tensor: torch.Tensor | None = None
    qat_tensor: torch.Tensor | None = None
    qat_delta: torch.Tensor | None = None
    activations: torch.Tensor | None = None
    metric_sqrt: torch.Tensor | None = None
    quantizer_residual: torch.Tensor | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


@runtime_checkable
class QATProjectionProtocol(Protocol):
    """Structural protocol used by extraction and tests."""

    @property
    def kind(self) -> ProjectionKind:
        """Projection family identifier for reports and schema records."""

    def prepare(self, context: TensorProjectionContext) -> None:
        """Receive optional per-tensor context before ``apply``."""

    def apply(self, delta: torch.Tensor, layer_name: str) -> torch.Tensor:
        """Return the delta to factorize."""

    def defuzz(
        self,
        U: torch.Tensor,
        S: torch.Tensor,
        Vh: torch.Tensor,
        delta: torch.Tensor,
        layer_name: str,
    ) -> SVDParts:
        """Return possibly adjusted SVD pieces before rank selection."""

    def fold(
        self,
        lora_A: torch.Tensor,
        lora_B: torch.Tensor,
        layer_name: str,
    ) -> LoraFactors:
        """Return PEFT-coordinate factors representing the original-basis update."""


class QATProjection(ABC):
    """Abstract base class for per-tensor QAT projection implementations."""

    kind: ProjectionKind = "custom"

    def __init__(self, config: QATProjectionConfig | None = None):
        self.config = config or QATProjectionConfig(projection_type=self.kind)
        self.context: TensorProjectionContext | None = None

    def prepare(self, context: TensorProjectionContext) -> None:
        """Store optional context for implementations that need it."""
        self._validate_context(context)
        self.context = context

    @abstractmethod
    def apply(self, delta: torch.Tensor, layer_name: str) -> torch.Tensor:
        """Transport ``delta`` into the basis that should be factorized."""

    @abstractmethod
    def defuzz(
        self,
        U: torch.Tensor,
        S: torch.Tensor,
        Vh: torch.Tensor,
        delta: torch.Tensor,
        layer_name: str,
    ) -> SVDParts:
        """Adjust SVD pieces toward quantization-stable directions."""

    @abstractmethod
    def fold(
        self,
        lora_A: torch.Tensor,
        lora_B: torch.Tensor,
        layer_name: str,
    ) -> LoraFactors:
        """Fold the inverse projection into emitted LoRA factors."""

    def _validate_context(self, context: TensorProjectionContext) -> None:
        if len(context.delta_shape) != 2:
            raise ValueError(f"delta_shape must be 2D, got {context.delta_shape}")
        if context.base_tensor is not None and tuple(context.base_tensor.shape) != context.delta_shape:
            raise ValueError("base_tensor shape must match delta_shape")
        if context.qat_tensor is not None and tuple(context.qat_tensor.shape) != context.delta_shape:
            raise ValueError("qat_tensor shape must match delta_shape")
        if context.qat_delta is not None and tuple(context.qat_delta.shape) != context.delta_shape:
            raise ValueError("qat_delta shape must match delta_shape")

    def _validate_delta(self, delta: torch.Tensor, layer_name: str) -> None:
        if delta.ndim != 2:
            raise ValueError(f"{layer_name}: projection delta must be 2D, got {tuple(delta.shape)}")
        if self.context is not None and self.context.layer_name == layer_name:
            if tuple(delta.shape) != self.context.delta_shape:
                raise ValueError(
                    f"{layer_name}: delta shape {tuple(delta.shape)} does not match "
                    f"context shape {self.context.delta_shape}"
                )

    def _validate_svd(self, U: torch.Tensor, S: torch.Tensor, Vh: torch.Tensor) -> None:
        if U.ndim != 2 or S.ndim != 1 or Vh.ndim != 2:
            raise ValueError("U, S, and Vh must have shapes [out,k], [k], [k,in]")
        k = S.shape[0]
        if U.shape[1] != k or Vh.shape[0] != k:
            raise ValueError(
                f"SVD shape mismatch: U={tuple(U.shape)} S={tuple(S.shape)} Vh={tuple(Vh.shape)}"
            )

    def _validate_factors(self, lora_A: torch.Tensor, lora_B: torch.Tensor) -> None:
        if lora_A.ndim != 2 or lora_B.ndim != 2:
            raise ValueError("LoRA factors must be 2D")
        if lora_A.shape[0] != lora_B.shape[1]:
            raise ValueError(
                f"LoRA rank mismatch: A={tuple(lora_A.shape)} B={tuple(lora_B.shape)}"
            )


class IdentityQATProjection(QATProjection):
    """Raw-SVD baseline: no projection, defuzzing, or fold transform."""

    kind: ProjectionKind = "identity"

    def apply(self, delta: torch.Tensor, layer_name: str) -> torch.Tensor:
        self._validate_delta(delta, layer_name)
        return delta

    def defuzz(
        self,
        U: torch.Tensor,
        S: torch.Tensor,
        Vh: torch.Tensor,
        delta: torch.Tensor,
        layer_name: str,
    ) -> SVDParts:
        self._validate_delta(delta, layer_name)
        self._validate_svd(U, S, Vh)
        return U, S, Vh

    def fold(
        self,
        lora_A: torch.Tensor,
        lora_B: torch.Tensor,
        layer_name: str,
    ) -> LoraFactors:
        self._validate_factors(lora_A, lora_B)
        return lora_A, lora_B


class RightMetricQATProjection(QATProjection):
    """Activation/Hessian right-metric projection.

    If ``C`` is a square root of an activation metric ``H`` over layer inputs,
    this projection factorizes:

    ```text
    Z = Delta @ C
    ```

    and folds the right factor back with a damped pseudoinverse:

    ```text
    A = A_weighted @ C^+
    ```

    This is the cleanest first non-identity projection because it preserves the
    PEFT update form ``B @ A`` while changing the extraction objective to the
    activation-weighted local reconstruction error.
    """

    kind: ProjectionKind = "metric"

    def __init__(
        self,
        metric_sqrt: torch.Tensor | None = None,
        config: QATProjectionConfig | None = None,
    ):
        super().__init__(config or QATProjectionConfig(projection_type="metric"))
        self.metric_sqrt = metric_sqrt

    def prepare(self, context: TensorProjectionContext) -> None:
        super().prepare(context)
        if context.metric_sqrt is not None:
            self.metric_sqrt = context.metric_sqrt
        self._validate_metric(context.delta_shape, context.layer_name)

    def apply(self, delta: torch.Tensor, layer_name: str) -> torch.Tensor:
        self._validate_delta(delta, layer_name)
        metric = self._metric(delta.device)
        self._validate_metric(tuple(delta.shape), layer_name)
        return right_weight_delta(delta, metric)

    def defuzz(
        self,
        U: torch.Tensor,
        S: torch.Tensor,
        Vh: torch.Tensor,
        delta: torch.Tensor,
        layer_name: str,
    ) -> SVDParts:
        self._validate_delta(delta, layer_name)
        self._validate_svd(U, S, Vh)
        return U, S, Vh

    def fold(
        self,
        lora_A: torch.Tensor,
        lora_B: torch.Tensor,
        layer_name: str,
    ) -> LoraFactors:
        self._validate_factors(lora_A, lora_B)
        metric = self._metric(lora_A.device)
        ordinary_A = project_weighted_lora_A(
            lora_A,
            metric,
            damping=self.config.damping,
        )
        return ordinary_A.to(dtype=lora_A.dtype).contiguous(), lora_B

    def _metric(self, device: torch.device | str) -> torch.Tensor:
        if self.metric_sqrt is None:
            raise RuntimeError("RightMetricQATProjection requires metric_sqrt in constructor or context")
        return self.metric_sqrt.to(device=device)

    def _validate_metric(self, delta_shape: tuple[int, int], layer_name: str) -> None:
        if self.metric_sqrt is None:
            raise RuntimeError(f"{layer_name}: missing metric_sqrt for metric projection")
        if self.metric_sqrt.ndim != 2:
            raise ValueError(f"{layer_name}: metric_sqrt must be 2D")
        in_features = delta_shape[1]
        if tuple(self.metric_sqrt.shape) != (in_features, in_features):
            raise ValueError(
                f"{layer_name}: metric_sqrt shape {tuple(self.metric_sqrt.shape)} "
                f"must be ({in_features}, {in_features})"
            )
