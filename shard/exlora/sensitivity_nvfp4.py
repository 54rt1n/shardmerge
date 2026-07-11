# shard/exlora/sensitivity_nvfp4.py
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

"""Sensitivity-guided NVFP4 block residual allocation.

Per-parameter QAT sensitivity is compatible with dense low-bit allocation, not
with multiply-SVD-divide low-rank extraction. This module uses the map to choose
which flattened NVFP4 blocks receive an additive residual sidecar.
"""

from __future__ import annotations

from typing import Literal

import torch

from .nvfp4 import BLOCK_SIZE

BlockReduce = Literal["mean", "sum", "max", "rms"]


def _pad_flat(x: torch.Tensor, block_size: int) -> torch.Tensor:
    flat = x.reshape(-1)
    pad = (-flat.numel()) % block_size
    if pad:
        flat = torch.cat([flat, flat.new_zeros(pad)])
    return flat


def block_scores(
    weights: torch.Tensor,
    block_size: int = BLOCK_SIZE,
    reduce: BlockReduce = "mean",
) -> torch.Tensor:
    """Reduce per-parameter weights to the flattened NVFP4 block grid."""
    if block_size <= 0:
        raise ValueError("block_size must be positive")
    work = _pad_flat(weights.to(torch.float32).abs(), block_size).reshape(-1, block_size)
    if reduce == "mean":
        return work.mean(dim=1)
    if reduce == "sum":
        return work.sum(dim=1)
    if reduce == "max":
        return work.amax(dim=1)
    if reduce == "rms":
        return torch.sqrt(work.square().mean(dim=1))
    raise ValueError("reduce must be one of: mean, sum, max, rms")


def block_residual_bpe(
    selected_blocks: torch.Tensor,
    numel: int,
    residual_bits: int = 16,
    block_size: int = BLOCK_SIZE,
    index_bits: int = 0,
) -> float:
    """Bits per original element for a deterministic block residual sidecar.

    ``index_bits`` defaults to zero for deterministic selectors where encoder
    and decoder can derive the same mask. Magnitude-gated residual sidecars
    must store block indices, so pass the index width used by the artifact.
    """
    if numel <= 0:
        return 0.0
    if residual_bits < 0 or index_bits < 0:
        raise ValueError("residual_bits and index_bits must be non-negative")
    nblocks = int(selected_blocks.to(torch.bool).sum().item())
    return float(nblocks * (block_size * residual_bits + index_bits) / numel)


def select_blocks_by_budget(
    scores: torch.Tensor,
    numel: int,
    extra_bpe: float,
    residual_bits: int = 16,
    block_size: int = BLOCK_SIZE,
    index_bits: int = 0,
) -> torch.Tensor:
    """Select top-scoring blocks that fit an additive residual bit budget."""
    if scores.ndim != 1:
        raise ValueError("scores must be a 1D tensor")
    if extra_bpe < 0:
        raise ValueError("extra_bpe must be non-negative")
    if residual_bits <= 0:
        raise ValueError("residual_bits must be positive")
    if index_bits < 0:
        raise ValueError("index_bits must be non-negative")
    if block_size <= 0:
        raise ValueError("block_size must be positive")

    nblocks = scores.numel()
    if nblocks == 0 or numel <= 0 or extra_bpe == 0:
        return torch.zeros(nblocks, dtype=torch.bool, device=scores.device)

    bits_per_block = residual_bits * block_size + index_bits
    max_blocks = int((extra_bpe * numel) // bits_per_block)
    max_blocks = max(0, min(max_blocks, nblocks))
    mask = torch.zeros(nblocks, dtype=torch.bool, device=scores.device)
    if max_blocks == 0:
        return mask
    top = torch.topk(scores.to(torch.float32), k=max_blocks, largest=True, sorted=False).indices
    mask[top] = True
    return mask


def apply_block_residual_sidecar(
    reference: torch.Tensor,
    quantized: torch.Tensor,
    selected_blocks: torch.Tensor,
    block_size: int = BLOCK_SIZE,
    residual_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """Add a block-sparse residual sidecar to a dense low-bit reconstruction."""
    if reference.shape != quantized.shape:
        raise ValueError("reference and quantized must have the same shape")
    if selected_blocks.ndim != 1:
        raise ValueError("selected_blocks must be a 1D boolean mask")

    n = reference.numel()
    residual = _pad_flat(reference.to(torch.float32) - quantized.to(torch.float32), block_size)
    base = _pad_flat(quantized.to(torch.float32), block_size)
    blocks = residual.reshape(-1, block_size)
    if selected_blocks.numel() != blocks.shape[0]:
        raise ValueError("selected_blocks length must match the number of flattened blocks")

    sidecar = torch.zeros_like(blocks)
    mask = selected_blocks.to(device=blocks.device, dtype=torch.bool)
    sidecar[mask] = blocks[mask].to(residual_dtype).to(torch.float32)
    return (base + sidecar.reshape(-1))[:n].reshape(reference.shape)


def elementwise_weighted_relative_error(
    reference: torch.Tensor,
    approximation: torch.Tensor,
    weights: torch.Tensor,
    eps: float = 1e-12,
) -> float:
    """Relative error under per-parameter sensitivity weights."""
    if reference.shape != approximation.shape or reference.shape != weights.shape:
        raise ValueError("reference, approximation, and weights must have the same shape")
    w = weights.to(torch.float32)
    ref = reference.to(torch.float32)
    approx = approximation.to(torch.float32)
    num = ((ref - approx) * w).norm()
    den = (ref * w).norm().clamp_min(eps)
    return float((num / den).item())
