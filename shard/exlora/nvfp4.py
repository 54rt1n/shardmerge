# shard/exlora/nvfp4.py
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

"""NVFP4 quantization: the low-bit codec ``Q_b`` for ExLoRA delta maps.

NVFP4 is a two-level micro-block format:

- Elements are E2M1 (4-bit float, magnitudes {0, .5, 1, 1.5, 2, 3, 4, 6}).
- Each block of 16 contiguous elements shares a scale stored in E4M3 fp8.
- A single FP32 per-tensor "global" scale lets the E4M3 block scales span the
  tensor's dynamic range (block_scale = global * e4m3_code).

Effective storage is ~4.5 bits/element (4 for the element + 8/16 for the block
scale). This module is pure tensor math with no model I/O so it can be unit
tested directly; it is the ``Q_b`` used by the low-bit and hybrid delta-map
artifacts in MAIN_OBJECTIVE.md.
"""

from __future__ import annotations

from dataclasses import dataclass
import torch

# E2M1 positive representable magnitudes (4-bit float, 1 mantissa / 2 exp bits).
E2M1_LEVELS = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
E2M1_MAX = 6.0
E4M3_MAX = 448.0
BLOCK_SIZE = 16


def _e2m1_boundaries(device, dtype) -> torch.Tensor:
    levels = torch.tensor(E2M1_LEVELS, device=device, dtype=dtype)
    return (levels[1:] + levels[:-1]) * 0.5  # midpoints between adjacent levels


def quantize_e2m1(x: torch.Tensor) -> torch.Tensor:
    """Round to the nearest signed E2M1 magnitude (values assumed pre-scaled)."""
    work = x.to(torch.float32)
    sign = torch.sign(work)
    mag = work.abs().clamp(max=E2M1_MAX)
    boundaries = _e2m1_boundaries(work.device, work.dtype)
    idx = torch.bucketize(mag, boundaries)  # nearest level index in [0, 7]
    levels = torch.tensor(E2M1_LEVELS, device=work.device, dtype=work.dtype)
    return sign * levels[idx]


def round_e4m3(x: torch.Tensor) -> torch.Tensor:
    """Round positive scales to the nearest representable E4M3 fp8 value.

    Uses the native ``float8_e4m3fn`` dtype when available; otherwise falls back
    to an exponent/mantissa rounding that matches E4M3 (4 exp bits bias 7,
    3 mantissa bits, max 448).
    """
    work = x.to(torch.float32).clamp(min=0.0, max=E4M3_MAX)
    if hasattr(torch, "float8_e4m3fn"):
        return work.to(torch.float8_e4m3fn).to(torch.float32)
    # Fallback: quantize mantissa to 3 bits per binade.
    safe = work.clamp_min(2.0 ** -9)
    exp = torch.floor(torch.log2(safe))
    scale = 2.0 ** (exp - 3)  # 3 mantissa bits
    q = torch.round(work / scale) * scale
    return torch.where(work > 0, q, torch.zeros_like(work)).clamp(max=E4M3_MAX)


@dataclass
class NVFP4Tensor:
    """An NVFP4-quantized tensor and the metadata needed to decode it."""

    codes: torch.Tensor          # int8 element codes in [0, 15] (sign<<3 | mag_idx)
    block_scale: torch.Tensor    # E4M3-rounded per-block ratio q_b (float32)
    global_scale: float          # per-tensor FP32 scale
    shape: torch.Size            # original (unpadded) shape
    block_size: int = BLOCK_SIZE

    def dequantize(self) -> torch.Tensor:
        levels = torch.tensor(E2M1_LEVELS, device=self.codes.device, dtype=torch.float32)
        mag = levels[(self.codes & 0x7).long()]
        sign = torch.where((self.codes & 0x8) != 0, -1.0, 1.0)
        eff = (self.global_scale * self.block_scale).unsqueeze(-1)  # [..., nblocks, 1]
        out = (sign * mag) * eff
        return out.reshape(-1)[: self.shape.numel()].reshape(self.shape)

    @property
    def bits_per_element(self) -> float:
        """Amortized storage: 4 bits/elem + 8 bits per block + 32 bits/tensor."""
        n = self.shape.numel()
        nblocks = self.block_scale.numel()
        return (4 * n + 8 * nblocks + 32) / max(n, 1)


def nvfp4_quantize(x: torch.Tensor, block_size: int = BLOCK_SIZE) -> NVFP4Tensor:
    """Quantize a tensor to NVFP4 along its last (contiguous) dimension."""
    work = x.to(torch.float32)
    shape = work.shape
    flat = work.reshape(-1)
    n = flat.numel()
    pad = (-n) % block_size
    if pad:
        flat = torch.cat([flat, flat.new_zeros(pad)])
    blocks = flat.reshape(-1, block_size)                       # [nblocks, block_size]

    tensor_amax = work.abs().max()
    if tensor_amax <= 0:
        codes = torch.zeros(blocks.shape, dtype=torch.int8, device=work.device)
        bscale = torch.zeros(blocks.shape[0], dtype=torch.float32, device=work.device)
        return NVFP4Tensor(codes, bscale, 0.0, shape, block_size)

    # Two-level scaling: global puts the largest block scale at the top of E4M3.
    global_scale = (tensor_amax / (E2M1_MAX * E4M3_MAX)).item()
    block_amax = blocks.abs().amax(dim=1)                       # [nblocks]
    q_b = round_e4m3(block_amax / E2M1_MAX / global_scale)      # E4M3 block ratio
    eff = (global_scale * q_b).clamp_min(1e-30).unsqueeze(1)    # [nblocks, 1]

    normalized = blocks / eff
    levels = quantize_e2m1(normalized)                          # signed magnitudes

    # Encode to int8 codes (sign bit | magnitude index).
    boundaries = _e2m1_boundaries(work.device, torch.float32)
    mag_idx = torch.bucketize(levels.abs(), boundaries).to(torch.int8)
    sign_bit = (levels < 0).to(torch.int8) << 3
    codes = (sign_bit | mag_idx).to(torch.int8)

    return NVFP4Tensor(codes, q_b.to(torch.float32), global_scale, shape, block_size)


def nvfp4_quantize_dequantize(x: torch.Tensor, block_size: int = BLOCK_SIZE) -> torch.Tensor:
    """Convenience round-trip: ``Q_b(x)`` as a dense tensor in x's dtype."""
    out = nvfp4_quantize(x, block_size=block_size).dequantize()
    return out.to(x.dtype)
