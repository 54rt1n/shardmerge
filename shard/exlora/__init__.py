# shard/exlora/__init__.py
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

"""High-fidelity LoRA extraction.

Factors the per-layer delta ``W_finetune - W_base`` into a low-rank
``B @ A`` adapter via truncated SVD, choosing each layer's rank to meet a
target reconstruction error, and writes the result as a PEFT adapter.
"""

from .config import ExtractConfig
from .svd import (
    LoraFactor,
    SpectrumDiagnostics,
    basis_svd,
    effective_rank,
    factor_delta_to_lora,
    noise_aware_singular_values,
    reconstruct_delta,
    retained_energy,
    select_rank,
    select_signal_rank,
    spectrum_diagnostics,
)
from .projection import (
    IdentityProjection,
    LoraFactors,
    Projection,
    ProjectionKind,
    QATProjection,
    QATProjectionConfig,
    QATProjectionProtocol,
    RightMetricQATProjection,
    SVDParts,
    TensorProjectionContext,
    load_projection,
)
from .extract import LoraExtractor
from .nvfp4_kernel import (
    NVFP4PackedTensor,
    is_triton_nvfp4_available,
    nvfp4_quantize_dequantize_triton,
    nvfp4_quantize_triton,
    pack_nvfp4_codes,
    pack_reference_nvfp4,
    unpack_nvfp4_codes,
)
from .nvfp4_residual import (
    NVFP4ResidualEntry,
    NVFP4ResidualMap,
    NVFP4ResidualWriter,
    build_nvfp4_residual_entry,
    dtype_bits,
    parse_torch_dtype,
)
from .sensitivity_nvfp4 import (
    apply_block_residual_sidecar,
    block_residual_bpe,
    block_scores,
    elementwise_weighted_relative_error,
    select_blocks_by_budget,
)
from .functions import (
    activation_hessian,
    cayley_transform,
    compute_delta,
    damped_pinv,
    householder_reflect,
    lora_delta,
    metric_householder_reflect,
    orthogonal_procrustes,
    project_weighted_lora_A,
    randomized_svd,
    relative_frobenius_error,
    right_weight_delta,
    skew_symmetric,
    split_svd_to_lora,
    symmetric_psd_sqrt,
    tiled_delta_left_matmul,
    tiled_delta_right_matmul,
    weighted_relative_error,
)

__all__ = [
    "ExtractConfig",
    "LoraExtractor",
    "LoraFactor",
    "NVFP4PackedTensor",
    "NVFP4ResidualEntry",
    "NVFP4ResidualMap",
    "NVFP4ResidualWriter",
    "Projection",
    "IdentityProjection",
    "QATProjection",
    "QATProjectionConfig",
    "QATProjectionProtocol",
    "RightMetricQATProjection",
    "TensorProjectionContext",
    "ProjectionKind",
    "SVDParts",
    "LoraFactors",
    "load_projection",
    "select_rank",
    "select_signal_rank",
    "spectrum_diagnostics",
    "SpectrumDiagnostics",
    "noise_aware_singular_values",
    "retained_energy",
    "effective_rank",
    "factor_delta_to_lora",
    "basis_svd",
    "reconstruct_delta",
    "activation_hessian",
    "cayley_transform",
    "compute_delta",
    "damped_pinv",
    "householder_reflect",
    "lora_delta",
    "metric_householder_reflect",
    "orthogonal_procrustes",
    "project_weighted_lora_A",
    "randomized_svd",
    "relative_frobenius_error",
    "right_weight_delta",
    "skew_symmetric",
    "split_svd_to_lora",
    "symmetric_psd_sqrt",
    "tiled_delta_left_matmul",
    "tiled_delta_right_matmul",
    "weighted_relative_error",
    "is_triton_nvfp4_available",
    "nvfp4_quantize_dequantize_triton",
    "nvfp4_quantize_triton",
    "pack_nvfp4_codes",
    "pack_reference_nvfp4",
    "unpack_nvfp4_codes",
    "build_nvfp4_residual_entry",
    "dtype_bits",
    "parse_torch_dtype",
    "apply_block_residual_sidecar",
    "block_residual_bpe",
    "block_scores",
    "elementwise_weighted_relative_error",
    "select_blocks_by_budget",
]
