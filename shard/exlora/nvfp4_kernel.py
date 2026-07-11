# shard/exlora/nvfp4_kernel.py
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

"""Triton NVFP4 kernels for packed ExLoRA delta-map storage.

The pure Torch implementation in :mod:`shard.exlora.nvfp4` is the correctness
oracle. This module provides the CUDA fast path and a packed representation:

- two E2M1 element codes per byte;
- one E4M3 block scale per 16 elements when Torch exposes fp8;
- one FP32 global scale stored as Python metadata.

Kernels operate over the same flattened 16-element stream as the reference
implementation. Row-aware grouping can be added as a separate mode once the
artifact schema needs it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch

from .nvfp4 import BLOCK_SIZE, E2M1_LEVELS, E2M1_MAX, E4M3_MAX, NVFP4Tensor

try:  # pragma: no cover - import availability is environment dependent.
    import triton
    import triton.language as tl

    _TRITON_IMPORT_ERROR: Optional[Exception] = None
except Exception as exc:  # pragma: no cover
    triton = None
    tl = None
    _TRITON_IMPORT_ERROR = exc


def _scale_dtype() -> torch.dtype:
    return torch.float8_e4m3fn if hasattr(torch, "float8_e4m3fn") else torch.float32


def is_triton_nvfp4_available() -> bool:
    """Return whether the Triton NVFP4 CUDA path can run in this process."""
    return triton is not None and torch.cuda.is_available()


def _require_triton_cuda(x: torch.Tensor) -> None:
    if triton is None:
        raise RuntimeError("Triton is not available") from _TRITON_IMPORT_ERROR
    if not x.is_cuda:
        raise ValueError("Triton NVFP4 kernels require a CUDA tensor")


def _validate_block_size(block_size: int) -> None:
    if block_size != BLOCK_SIZE:
        raise ValueError("Triton NVFP4 currently supports block_size=16 only")


def _validate_packed_payload(
    packed_codes: torch.Tensor,
    block_scale: torch.Tensor,
    block_size: int,
) -> None:
    if packed_codes.dtype != torch.uint8:
        raise TypeError("packed_codes must be a uint8 tensor")
    if block_scale.device != packed_codes.device:
        raise ValueError("block_scale must be on the same device as packed_codes")
    if block_scale.ndim != 1:
        raise ValueError("block_scale must be a 1D tensor")
    if packed_codes.ndim != 2:
        raise ValueError("packed_codes must have shape [nblocks, block_size / 2]")
    if packed_codes.shape[-1] * 2 != block_size:
        raise ValueError("packed_codes last dimension does not match block_size")
    if block_scale.numel() != packed_codes.shape[0]:
        raise ValueError("block_scale length must match packed_codes nblocks")


def pack_nvfp4_codes(codes: torch.Tensor, *, validate: bool = True) -> torch.Tensor:
    """Pack unpacked int codes ``[..., block_size]`` into uint8 nibbles."""
    if codes.ndim == 0:
        raise ValueError("codes must have at least one dimension")
    if codes.shape[-1] % 2 != 0:
        raise ValueError("last dimension must have an even number of codes")
    if validate:
        if codes.dtype.is_floating_point or codes.dtype.is_complex:
            raise TypeError("codes must be an integer tensor")
        if torch.any((codes < 0) | (codes > 15)):
            raise ValueError("codes must be in the range [0, 15]")
    work = codes.to(torch.uint8) & 0x0F
    lo = work[..., 0::2]
    hi = work[..., 1::2] << 4
    return lo | hi


def unpack_nvfp4_codes(packed_codes: torch.Tensor, block_size: int = BLOCK_SIZE) -> torch.Tensor:
    """Unpack uint8 nibbles into int8 codes shaped ``[..., block_size]``."""
    if packed_codes.ndim == 0:
        raise ValueError("packed_codes must have at least one dimension")
    if block_size % 2 != 0:
        raise ValueError("block_size must be even")
    if packed_codes.shape[-1] * 2 != block_size:
        raise ValueError("packed last dimension does not match block_size")

    out_shape = (*packed_codes.shape[:-1], block_size)
    out = torch.empty(out_shape, dtype=torch.int8, device=packed_codes.device)
    work = packed_codes.to(torch.uint8)
    out[..., 0::2] = (work & 0x0F).to(torch.int8)
    out[..., 1::2] = ((work >> 4) & 0x0F).to(torch.int8)
    return out


@dataclass
class NVFP4PackedTensor:
    """Packed NVFP4 tensor with enough metadata to decode it."""

    packed_codes: torch.Tensor    # uint8, shape [nblocks, block_size / 2]
    block_scale: torch.Tensor     # float8_e4m3fn when available, else float32
    global_scale: float
    shape: torch.Size
    block_size: int = BLOCK_SIZE

    @property
    def nblocks(self) -> int:
        return int(self.block_scale.numel())

    @property
    def format_bits_per_element(self) -> float:
        """NVFP4-format bits, assuming fp8 block scales."""
        n = self.shape.numel()
        return (8 * self.packed_codes.numel() + 8 * self.nblocks + 32) / max(n, 1)

    @property
    def tensor_bits_per_element(self) -> float:
        """Actual PyTorch tensor storage bits for this in-memory object."""
        n = self.shape.numel()
        scale_bits = 8 * self.block_scale.element_size() * self.nblocks
        return (8 * self.packed_codes.numel() + scale_bits + 32) / max(n, 1)

    @property
    def bits_per_element(self) -> float:
        return self.format_bits_per_element

    def dequantize(self, dtype: torch.dtype = torch.float32) -> torch.Tensor:
        """Decode to a dense tensor, using Triton on CUDA and Torch otherwise."""
        _validate_block_size(self.block_size)
        _validate_packed_payload(self.packed_codes, self.block_scale, self.block_size)
        n = self.shape.numel()
        if n == 0:
            return torch.empty(self.shape, dtype=dtype, device=self.packed_codes.device)

        if self.packed_codes.is_cuda:
            _require_triton_cuda(self.packed_codes)
            out = torch.empty(self.shape, dtype=dtype, device=self.packed_codes.device)
            _nvfp4_dequantize_kernel[(self.nblocks,)](
                self.packed_codes,
                self.block_scale,
                out.reshape(-1),
                n,
                float(self.global_scale),
                BLOCK_SIZE=self.block_size,
                HALF_BLOCK=self.block_size // 2,
            )
            return out

        codes = unpack_nvfp4_codes(self.packed_codes, self.block_size)
        levels = torch.tensor(E2M1_LEVELS, device=codes.device, dtype=torch.float32)
        mag = levels[(codes & 0x7).long()]
        sign = torch.where((codes & 0x8) != 0, -1.0, 1.0)
        eff = (float(self.global_scale) * self.block_scale.to(torch.float32)).unsqueeze(-1)
        out = (sign * mag) * eff
        return out.reshape(-1)[:n].reshape(self.shape).to(dtype)


def pack_reference_nvfp4(q: NVFP4Tensor) -> NVFP4PackedTensor:
    """Pack a reference ``NVFP4Tensor`` without changing its numeric payload."""
    _validate_block_size(q.block_size)
    packed = pack_nvfp4_codes(q.codes, validate=False)
    block_scale = q.block_scale.to(_scale_dtype())
    return NVFP4PackedTensor(packed, block_scale, q.global_scale, q.shape, q.block_size)


def nvfp4_quantize_triton(x: torch.Tensor, block_size: int = BLOCK_SIZE) -> NVFP4PackedTensor:
    """Quantize a CUDA tensor to packed NVFP4 with Triton."""
    _validate_block_size(block_size)
    _require_triton_cuda(x)

    work = x.contiguous()
    shape = work.shape
    n = work.numel()
    nblocks = triton.cdiv(n, block_size)

    if n == 0:
        packed = torch.empty((0, block_size // 2), dtype=torch.uint8, device=work.device)
        scales = torch.empty((0,), dtype=_scale_dtype(), device=work.device)
        return NVFP4PackedTensor(packed, scales, 0.0, shape, block_size)

    tensor_amax = work.detach().abs().amax().to(torch.float32).item()
    if tensor_amax <= 0:
        packed = torch.zeros((nblocks, block_size // 2), dtype=torch.uint8, device=work.device)
        scales = torch.zeros((nblocks,), dtype=_scale_dtype(), device=work.device)
        return NVFP4PackedTensor(packed, scales, 0.0, shape, block_size)

    global_scale = float(tensor_amax / (E2M1_MAX * E4M3_MAX))
    packed = torch.empty((nblocks, block_size // 2), dtype=torch.uint8, device=work.device)
    scales = torch.empty((nblocks,), dtype=_scale_dtype(), device=work.device)

    _nvfp4_scale_kernel[(nblocks,)](
        work.reshape(-1),
        scales,
        n,
        global_scale,
        BLOCK_SIZE=block_size,
        ROUND_E4M3=scales.dtype == torch.float32,
    )
    _nvfp4_encode_kernel[(nblocks,)](
        work.reshape(-1),
        packed,
        scales,
        n,
        global_scale,
        BLOCK_SIZE=block_size,
        HALF_BLOCK=block_size // 2,
    )
    return NVFP4PackedTensor(packed, scales, global_scale, shape, block_size)


def nvfp4_quantize_dequantize_triton(x: torch.Tensor, block_size: int = BLOCK_SIZE) -> torch.Tensor:
    """Convenience CUDA round-trip returning a dense tensor in ``x.dtype``."""
    return nvfp4_quantize_triton(x, block_size=block_size).dequantize(dtype=x.dtype)


if triton is not None:

    @triton.jit
    def _round_e4m3_tl(x):
        x = tl.minimum(tl.maximum(x, 0.0), 448.0)
        safe = tl.maximum(x, 0.001953125)  # 2 ** -9
        exp = tl.floor(tl.log2(safe))
        step = tl.exp2(exp - 3.0)
        q = tl.floor(x / step + 0.5) * step
        return tl.minimum(tl.where(x > 0.0, q, 0.0), 448.0)

    @triton.jit
    def _e2m1_code_tl(vals, eff_scale):
        mag = tl.minimum(tl.abs(vals) / tl.maximum(eff_scale, 1.0e-30), 6.0)
        idx = (
            tl.where(mag > 0.25, 1, 0)
            + tl.where(mag > 0.75, 1, 0)
            + tl.where(mag > 1.25, 1, 0)
            + tl.where(mag > 1.75, 1, 0)
            + tl.where(mag > 2.5, 1, 0)
            + tl.where(mag > 3.5, 1, 0)
            + tl.where(mag > 5.0, 1, 0)
        )
        sign = tl.where((idx != 0) & (vals < 0.0), 8, 0)
        return sign + idx

    @triton.jit
    def _e2m1_level_tl(code):
        idx = code - (code // 8) * 8
        return tl.where(
            idx == 0,
            0.0,
            tl.where(
                idx == 1,
                0.5,
                tl.where(
                    idx == 2,
                    1.0,
                    tl.where(
                        idx == 3,
                        1.5,
                        tl.where(
                            idx == 4,
                            2.0,
                            tl.where(idx == 5, 3.0, tl.where(idx == 6, 4.0, 6.0)),
                        ),
                    ),
                ),
            ),
        )

    @triton.jit
    def _nvfp4_scale_kernel(
        x_ptr,
        scale_ptr,
        n_elements,
        global_scale,
        BLOCK_SIZE: tl.constexpr,
        ROUND_E4M3: tl.constexpr,
    ):
        pid = tl.program_id(axis=0)
        elem = tl.arange(0, BLOCK_SIZE)
        base = pid * BLOCK_SIZE
        offsets = base + elem
        mask = offsets < n_elements

        vals = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        block_amax = tl.max(tl.abs(vals), axis=0)
        global_scale_f32 = tl.full([], global_scale, tl.float32)
        block_scale = tl.minimum(tl.maximum(block_amax / (6.0 * global_scale_f32), 0.0), 448.0)
        if ROUND_E4M3:
            block_scale = _round_e4m3_tl(block_scale)
        tl.store(scale_ptr + pid, block_scale)

    @triton.jit
    def _nvfp4_encode_kernel(
        x_ptr,
        packed_ptr,
        scale_ptr,
        n_elements,
        global_scale,
        BLOCK_SIZE: tl.constexpr,
        HALF_BLOCK: tl.constexpr,
    ):
        pid = tl.program_id(axis=0)
        base = pid * BLOCK_SIZE
        global_scale_f32 = tl.full([], global_scale, tl.float32)
        block_scale = tl.load(scale_ptr + pid).to(tl.float32)
        eff_scale = global_scale_f32 * block_scale

        pair = tl.arange(0, HALF_BLOCK)
        lo_offsets = base + pair * 2
        hi_offsets = lo_offsets + 1
        lo_vals = tl.load(x_ptr + lo_offsets, mask=lo_offsets < n_elements, other=0.0).to(tl.float32)
        hi_vals = tl.load(x_ptr + hi_offsets, mask=hi_offsets < n_elements, other=0.0).to(tl.float32)
        lo_code = _e2m1_code_tl(lo_vals, eff_scale)
        hi_code = _e2m1_code_tl(hi_vals, eff_scale)
        packed = lo_code + hi_code * 16
        tl.store(packed_ptr + pid * HALF_BLOCK + pair, packed)

    @triton.jit
    def _nvfp4_dequantize_kernel(
        packed_ptr,
        scale_ptr,
        out_ptr,
        n_elements,
        global_scale,
        BLOCK_SIZE: tl.constexpr,
        HALF_BLOCK: tl.constexpr,
    ):
        pid = tl.program_id(axis=0)
        elem = tl.arange(0, BLOCK_SIZE)
        base = pid * BLOCK_SIZE
        offsets = base + elem
        pair = elem // 2

        packed = tl.load(packed_ptr + pid * HALF_BLOCK + pair)
        lo_code = packed - (packed // 16) * 16
        hi_code = packed // 16
        code = tl.where((elem - (elem // 2) * 2) == 0, lo_code, hi_code)
        level = _e2m1_level_tl(code)
        sign = tl.where(code >= 8, -1.0, 1.0)
        scale = tl.full([], global_scale, tl.float32) * tl.load(scale_ptr + pid).to(tl.float32)
        vals = sign * level * scale
        tl.store(out_ptr + offsets, vals, mask=offsets < n_elements)
