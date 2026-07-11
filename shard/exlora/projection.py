# shard/exlora/projection.py
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

"""The ``P_qat`` projection seam for ExLoRA extraction.

A :class:`Projection` wraps the three places the QAT survival map will plug
into the per-tensor SVD pipeline:

1. ``apply`` — transport the raw delta into a quantization-friendly basis
   (metric weighting or reflection fit from the base<->qat endpoints).
2. ``defuzz`` — given the *original-basis SVD* of the (projected) delta, steer
   the retained subspace toward nvfp4-stable directions before truncation.
3. ``fold`` — fold the inverse transform back into the LoRA factors so the
   emitted adapter stays in original weight coordinates (PEFT-compatible).

The default :class:`IdentityProjection` makes every stage a passthrough, which
is exactly the raw-SVD baseline. The QAT-driven projection is not implemented
yet; the seams are marked with TODOs. See ``research/MAIN_OBJECTIVE.md`` and
``research/BETTER_THEORY.md``.
"""

import logging
from pathlib import Path

from .qat_projection import (
    IdentityQATProjection,
    LoraFactors,
    ProjectionKind,
    QATProjection,
    QATProjectionConfig,
    QATProjectionProtocol,
    RightMetricQATProjection,
    SVDParts,
    TensorProjectionContext,
)

logger = logging.getLogger(__name__)

# Backward-compatible names used by the extraction path and existing tests.
Projection = QATProjection
IdentityProjection = IdentityQATProjection


def load_projection(config) -> Projection:
    """Build the projection for an extraction config.

    Returns :class:`IdentityProjection` (raw SVD) unless a QAT projection model
    and a non-``none`` ``projection_type`` are configured.
    """
    proj_model = getattr(config, "qat_projection_model", None)
    proj_type = getattr(config, "projection_type", "none")

    if proj_model and proj_type != "none":
        qat_config = QATProjectionConfig(
            projection_type=proj_type,
            projection_model=Path(proj_model),
        )
        # TODO(projection): build a QATProjection from the QAT survival map at
        # `proj_model`. With both base<->qat endpoints (the "two points"):
        #   apply():  reflect/weight the delta into the QAT gauge, e.g.
        #             reflection R fit via functions.orthogonal_procrustes /
        #             householder_reflect, or metric weighting via
        #             functions.right_weight_delta. Must be linear + invertible.
        #   defuzz(): register the original-basis SVD subspace onto the nvfp4-
        #             stable directions of the map (quantize in the reflected
        #             frame; the original-basis SVD is passed in for this).
        #   fold():   fold the inverse transform back into lora_A/lora_B so the
        #             emitted adapter is PEFT-compatible bf16 weight-space LoRA.
        # See research/MAIN_OBJECTIVE.md and research/BETTER_THEORY.md.
        raise NotImplementedError(
            f"projection_type={qat_config.projection_type!r} with "
            f"qat_projection_model={str(qat_config.projection_model)!r} "
            "is not implemented yet; the projection/defuzz seams are stubbed. "
            "Use projection_type='none' for the raw-SVD path."
        )

    return IdentityProjection()


__all__ = [
    "Projection",
    "IdentityProjection",
    "load_projection",
    "ProjectionKind",
    "QATProjection",
    "IdentityQATProjection",
    "QATProjectionConfig",
    "QATProjectionProtocol",
    "RightMetricQATProjection",
    "TensorProjectionContext",
    "SVDParts",
    "LoraFactors",
]
