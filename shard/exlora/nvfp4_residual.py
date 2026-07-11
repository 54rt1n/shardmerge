# shard/exlora/nvfp4_residual.py
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

"""NVFP4 delta-map artifact with a magnitude-gated residual sidecar.

This is the precision-first ExLoRA path: store the dense delta as packed NVFP4,
then spend a small additive residual budget on the blocks where NVFP4 made the
largest local error. There is no inverse weighting step, so the sidecar cannot
amplify near-zero QAT sensitivities.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional
import hashlib
import json
import logging

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from .nvfp4 import BLOCK_SIZE, nvfp4_quantize
from .nvfp4_kernel import (
    NVFP4PackedTensor,
    is_triton_nvfp4_available,
    nvfp4_quantize_triton,
    pack_reference_nvfp4,
    unpack_nvfp4_codes,
)
from .sensitivity_nvfp4 import (
    block_residual_bpe,
    block_scores,
    select_blocks_by_budget,
)

logger = logging.getLogger(__name__)


SelectionPolicy = str
FloorCorrectionPolicy = str


def dtype_bits(dtype: torch.dtype) -> int:
    """Return the storage bits for a supported residual dtype."""
    if dtype in (torch.float16, torch.bfloat16):
        return 16
    if dtype == torch.float32:
        return 32
    if dtype in (torch.int8, torch.uint8):
        return 8
    raise ValueError(f"Unsupported residual dtype {dtype}")


def parse_torch_dtype(name: str | torch.dtype) -> torch.dtype:
    """Parse a torch dtype name used in YAML configs."""
    if isinstance(name, torch.dtype):
        return name
    dtype = getattr(torch, str(name), None)
    if not isinstance(dtype, torch.dtype):
        raise ValueError(f"Unknown torch dtype {name!r}")
    return dtype


def _pad_flat(x: torch.Tensor, block_size: int = BLOCK_SIZE) -> torch.Tensor:
    flat = x.reshape(-1)
    pad = (-flat.numel()) % block_size
    if pad:
        flat = torch.cat([flat, flat.new_zeros(pad)])
    return flat


def _pack_delta(delta: torch.Tensor, block_size: int = BLOCK_SIZE) -> NVFP4PackedTensor:
    """Pack a delta tensor to NVFP4, using Triton when possible."""
    if block_size != BLOCK_SIZE:
        raise ValueError("NVFP4 residual artifacts currently require block_size=16")
    work = delta.contiguous()
    if work.is_cuda and is_triton_nvfp4_available():
        return nvfp4_quantize_triton(work, block_size=block_size)
    return pack_reference_nvfp4(nvfp4_quantize(work, block_size=block_size))


def _empty_floor_correction() -> torch.Tensor:
    return torch.empty((0, 0), dtype=torch.float32)


def _floor_correction_bpe(table: torch.Tensor | None, numel: int) -> float:
    if table is None or table.numel() == 0 or numel <= 0:
        return 0.0
    return float(table.numel() * table.element_size() * 8 / numel)


def _fit_code_position_floor_correction(
    residual: torch.Tensor,
    packed: NVFP4PackedTensor,
    floor_mask: torch.Tensor,
) -> torch.Tensor:
    """Fit mean residual by NVFP4 code and intra-block position on floor blocks."""
    codes = unpack_nvfp4_codes(packed.packed_codes, packed.block_size).to(residual.device)
    blocks = _pad_flat(residual.to(torch.float32), packed.block_size).reshape(-1, packed.block_size)
    mask = floor_mask.to(device=blocks.device, dtype=torch.bool)
    if mask.numel() != blocks.shape[0]:
        raise ValueError("floor_mask length must match the number of NVFP4 blocks")
    if not bool(mask.any()):
        return torch.zeros((16, packed.block_size), dtype=torch.float32, device=blocks.device)

    floor_codes = codes[mask].long()
    floor_values = blocks[mask]
    sums = torch.zeros((16, packed.block_size), dtype=torch.float32, device=blocks.device)
    counts = torch.zeros((16, packed.block_size), dtype=torch.float32, device=blocks.device)
    positions = torch.arange(packed.block_size, device=blocks.device)
    for pos in range(packed.block_size):
        sums[:, pos].scatter_add_(0, floor_codes[:, pos], floor_values[:, pos])
        counts[:, pos].scatter_add_(0, floor_codes[:, pos], torch.ones_like(floor_values[:, pos]))
    return sums / counts.clamp_min(1.0)


def _apply_code_position_floor_correction(
    quantized: torch.Tensor,
    packed: NVFP4PackedTensor,
    table: torch.Tensor,
) -> torch.Tensor:
    """Apply a code x position residual-bias table to an NVFP4 decode."""
    if table.numel() == 0:
        return quantized
    if table.shape != (16, packed.block_size):
        raise ValueError(f"floor correction table must have shape (16, {packed.block_size})")
    codes = unpack_nvfp4_codes(packed.packed_codes.to(quantized.device), packed.block_size).long()
    positions = torch.arange(packed.block_size, device=quantized.device).view(1, -1).expand_as(codes)
    correction = table.to(device=quantized.device, dtype=torch.float32)[codes, positions]
    flat = _pad_flat(quantized.to(torch.float32), packed.block_size)
    corrected = flat + correction.reshape(-1)
    return corrected[: quantized.numel()].reshape(quantized.shape)


@dataclass
class NVFP4ResidualEntry:
    """One tensor in an NVFP4 + sparse residual delta map."""

    packed: NVFP4PackedTensor
    residual_indices: torch.Tensor
    residual_values: torch.Tensor
    rel_error: float
    nvfp4_rel_error: float
    extra_bpe: float
    residual_bpe: float
    index_bits: int
    residual_bits: int
    selection_policy: SelectionPolicy = "residual_rms"
    floor_correction_policy: FloorCorrectionPolicy = "none"
    floor_correction: Optional[torch.Tensor] = None
    floor_correction_bpe: float = 0.0
    corrected_nvfp4_rel_error: Optional[float] = None
    floor_level: float = 0.0
    floor_width: float = 0.0
    floor_robust_scale: float = 0.0
    floor_blocks: int = 0
    super_floor_blocks: int = 0

    @property
    def total_bpe(self) -> float:
        return float(self.packed.bits_per_element + self.floor_correction_bpe + self.residual_bpe)

    def dequantize(self, device: str | torch.device = "cpu", dtype: torch.dtype = torch.float32) -> torch.Tensor:
        """Decode the dense delta approximation."""
        packed = NVFP4PackedTensor(
            self.packed.packed_codes.to(device),
            self.packed.block_scale.to(device),
            self.packed.global_scale,
            self.packed.shape,
            self.packed.block_size,
        )
        out = packed.dequantize(dtype=torch.float32)
        if self.floor_correction_policy == "code_position_mean":
            table = self.floor_correction if self.floor_correction is not None else _empty_floor_correction()
            out = _apply_code_position_floor_correction(out, packed, table)
        elif self.floor_correction_policy != "none":
            raise ValueError(f"Unknown floor_correction_policy {self.floor_correction_policy!r}")
        if self.residual_indices.numel() == 0:
            return out.to(dtype)

        n = out.numel()
        flat = _pad_flat(out, packed.block_size)
        blocks = flat.reshape(-1, packed.block_size)
        idx = self.residual_indices.to(device=device, dtype=torch.long)
        vals = self.residual_values.to(device=device, dtype=torch.float32)
        blocks[idx] += vals
        return flat[:n].reshape(packed.shape).to(dtype)


def build_nvfp4_residual_entry(
    delta: torch.Tensor,
    *,
    extra_bpe: float = 0.0,
    residual_dtype: torch.dtype = torch.bfloat16,
    block_size: int = BLOCK_SIZE,
    index_bits: int = 32,
    selection_policy: SelectionPolicy = "residual_rms",
    floor_correction_policy: FloorCorrectionPolicy = "none",
    floor_tolerance: float = 1.0,
    floor_effect_threshold: float = 2.0,
) -> NVFP4ResidualEntry:
    """Quantize ``delta`` to NVFP4 and add top residual blocks by magnitude."""
    if delta.ndim != 2:
        raise ValueError(f"delta must be 2D, got {tuple(delta.shape)}")
    if extra_bpe < 0:
        raise ValueError("extra_bpe must be non-negative")
    if index_bits not in (0, 16, 32, 64):
        raise ValueError("index_bits must be one of 0, 16, 32, or 64")
    residual_bits = dtype_bits(residual_dtype)

    if selection_policy not in {"residual_rms", "floor_effect"}:
        raise ValueError("selection_policy must be 'residual_rms' or 'floor_effect'")
    if floor_correction_policy not in {"none", "code_position_mean"}:
        raise ValueError("floor_correction_policy must be 'none' or 'code_position_mean'")
    ref = delta.to(torch.float32)
    packed = _pack_delta(ref, block_size=block_size)
    q = packed.dequantize(dtype=torch.float32)
    residual = ref - q
    nvfp4_rel = float((residual.norm() / ref.norm().clamp_min(1e-12)).item())

    if index_bits == 0:
        raise ValueError(f"{selection_policy} selection requires stored block indices")

    from shard.ranker.codec_floor import CodecFloorConfig, CodecFloorModel

    floor_cfg = CodecFloorConfig(
        floor_tolerance=floor_tolerance,
        effect_threshold=floor_effect_threshold,
    )
    pre_scores = block_scores(residual, block_size=block_size, reduce="rms")
    pre_floor = CodecFloorModel.fit(pre_scores, floor_cfg)

    floor_correction = _empty_floor_correction().to(ref.device)
    corrected_q = q
    if floor_correction_policy == "code_position_mean":
        floor_correction = _fit_code_position_floor_correction(residual, packed, pre_floor.floor_mask)
        corrected_q = _apply_code_position_floor_correction(q, packed, floor_correction)

    corrected_residual = ref - corrected_q
    corrected_nvfp4_rel = float((corrected_residual.norm() / ref.norm().clamp_min(1e-12)).item())
    post_scores = block_scores(corrected_residual, block_size=block_size, reduce="rms")
    post_floor = CodecFloorModel.fit(post_scores, floor_cfg)
    if selection_policy == "residual_rms":
        scores = post_scores
        floor_gate = None
    else:
        scores = post_floor.effect_scores.clamp_min(0.0)
        floor_gate = post_floor.super_floor_mask.to(device=scores.device, dtype=torch.bool)

    selected = select_blocks_by_budget(
        scores,
        numel=ref.numel(),
        extra_bpe=extra_bpe,
        residual_bits=residual_bits,
        block_size=block_size,
        index_bits=index_bits,
    )
    if floor_gate is not None:
        selected &= floor_gate
    indices = torch.nonzero(selected, as_tuple=False).flatten().sort().values

    if indices.numel() == 0:
        residual_values = torch.empty((0, block_size), dtype=residual_dtype, device=ref.device)
        recon = corrected_q
    else:
        blocks = _pad_flat(corrected_residual, block_size).reshape(-1, block_size)
        residual_values = blocks[indices].to(residual_dtype)
        recon = corrected_q
        flat = _pad_flat(recon, block_size)
        recon_blocks = flat.reshape(-1, block_size)
        recon_blocks[indices] += residual_values.to(torch.float32)
        recon = flat[: ref.numel()].reshape(ref.shape)

    rel = float(((ref - recon).norm() / ref.norm().clamp_min(1e-12)).item())
    residual_bpe = block_residual_bpe(
        selected,
        numel=ref.numel(),
        residual_bits=residual_bits,
        block_size=block_size,
        index_bits=index_bits,
    )
    return NVFP4ResidualEntry(
        packed=packed,
        residual_indices=indices.to(torch.int64).cpu(),
        residual_values=residual_values.contiguous().cpu(),
        rel_error=rel,
        nvfp4_rel_error=nvfp4_rel,
        extra_bpe=float(extra_bpe),
        residual_bpe=float(residual_bpe),
        index_bits=int(index_bits),
        residual_bits=int(residual_bits),
        selection_policy=selection_policy,
        floor_correction_policy=floor_correction_policy,
        floor_correction=floor_correction.contiguous().cpu(),
        floor_correction_bpe=_floor_correction_bpe(floor_correction, ref.numel()),
        corrected_nvfp4_rel_error=corrected_nvfp4_rel,
        floor_level=post_floor.floor_level,
        floor_width=post_floor.floor_width,
        floor_robust_scale=post_floor.robust_scale,
        floor_blocks=post_floor.floor_blocks,
        super_floor_blocks=post_floor.super_floor_blocks,
    )


def _tensor_file_name(layer_name: str) -> str:
    safe = layer_name.replace("/", "__")
    digest = hashlib.sha1(layer_name.encode("utf-8")).hexdigest()[:12]
    return f"{safe}.{digest}.safetensors"


class NVFP4ResidualWriter:
    """Streaming writer for an NVFP4 residual delta-map artifact."""

    def __init__(
        self,
        output_path: Path,
        *,
        base_model: str,
        finetune_model: str,
        target_modules: list[str],
        extra_bpe: float,
        residual_dtype: torch.dtype,
        index_bits: int = 32,
        selection_policy: SelectionPolicy = "residual_rms",
        floor_correction_policy: FloorCorrectionPolicy = "none",
        floor_tolerance: float = 1.0,
        floor_effect_threshold: float = 2.0,
    ):
        self.output_path = Path(output_path)
        self.tensor_dir = self.output_path / "tensors"
        self.base_model = base_model
        self.finetune_model = finetune_model
        self.target_modules = list(target_modules)
        self.extra_bpe = float(extra_bpe)
        self.residual_dtype = residual_dtype
        self.index_bits = int(index_bits)
        self.selection_policy = selection_policy
        self.floor_correction_policy = floor_correction_policy
        self.floor_tolerance = float(floor_tolerance)
        self.floor_effect_threshold = float(floor_effect_threshold)
        self.tensors: Dict[str, dict] = {}

    def add(self, layer_name: str, entry: NVFP4ResidualEntry) -> None:
        """Write one tensor payload and record it in the manifest."""
        self.tensor_dir.mkdir(parents=True, exist_ok=True)
        file_name = _tensor_file_name(layer_name)
        save_file(
            {
                "codes": entry.packed.packed_codes.contiguous().cpu(),
                "block_scale": entry.packed.block_scale.contiguous().cpu(),
                "residual_indices": entry.residual_indices.contiguous().cpu(),
                "residual_values": entry.residual_values.contiguous().cpu(),
                "floor_correction": (
                    entry.floor_correction.contiguous().cpu()
                    if entry.floor_correction is not None
                    else _empty_floor_correction()
                ),
            },
            str(self.tensor_dir / file_name),
            metadata={"format": "pt", "content": "exlora_nvfp4_residual_tensor"},
        )
        self.tensors[layer_name] = {
            "file": f"tensors/{file_name}",
            "shape": list(entry.packed.shape),
            "block_size": entry.packed.block_size,
            "global_scale": entry.packed.global_scale,
            "selection_policy": entry.selection_policy,
            "floor_correction_policy": entry.floor_correction_policy,
            "residual_dtype": str(entry.residual_values.dtype).removeprefix("torch."),
            "residual_bits": entry.residual_bits,
            "index_bits": entry.index_bits,
            "extra_bpe": entry.extra_bpe,
            "nvfp4_bpe": entry.packed.bits_per_element,
            "floor_correction_bpe": entry.floor_correction_bpe,
            "residual_bpe": entry.residual_bpe,
            "total_bpe": entry.total_bpe,
            "nvfp4_rel_error": entry.nvfp4_rel_error,
            "corrected_nvfp4_rel_error": entry.corrected_nvfp4_rel_error,
            "rel_error": entry.rel_error,
            "residual_blocks": int(entry.residual_indices.numel()),
            "floor_level": entry.floor_level,
            "floor_width": entry.floor_width,
            "floor_robust_scale": entry.floor_robust_scale,
            "floor_blocks": entry.floor_blocks,
            "super_floor_blocks": entry.super_floor_blocks,
        }

    def write_manifest(self) -> None:
        self.output_path.mkdir(parents=True, exist_ok=True)
        manifest = {
            "schema_version": 1,
            "artifact_type": "nvfp4_residual_delta_map",
            "base_model": self.base_model,
            "finetune_model": self.finetune_model,
            "target_modules": sorted(set(self.target_modules)),
            "extra_bpe": self.extra_bpe,
            "residual_dtype": str(self.residual_dtype).removeprefix("torch."),
            "index_bits": self.index_bits,
            "selection_policy": self.selection_policy,
            "floor_correction_policy": self.floor_correction_policy,
            "floor_tolerance": self.floor_tolerance,
            "floor_effect_threshold": self.floor_effect_threshold,
            "tensors": self.tensors,
        }
        (self.output_path / "exlora_nvfp4_residual.json").write_text(json.dumps(manifest, indent=2) + "\n")
        logger.info(f"Wrote NVFP4 residual delta map with {len(self.tensors)} tensors to {self.output_path}")


class NVFP4ResidualMap:
    """Loader for an NVFP4 residual delta-map artifact."""

    def __init__(self, path: Path, manifest: dict):
        self.path = Path(path)
        self.manifest = manifest

    @classmethod
    def load(cls, path: Path) -> "NVFP4ResidualMap":
        path = Path(path)
        manifest = json.loads((path / "exlora_nvfp4_residual.json").read_text())
        return cls(path, manifest)

    def entry(self, layer_name: str) -> Optional[NVFP4ResidualEntry]:
        meta = self.manifest["tensors"].get(layer_name)
        if meta is None:
            return None
        payload = self.path / meta["file"]
        with safe_open(str(payload), framework="pt") as f:
            packed = NVFP4PackedTensor(
                f.get_tensor("codes"),
                f.get_tensor("block_scale"),
                float(meta["global_scale"]),
                torch.Size(meta["shape"]),
                int(meta["block_size"]),
            )
            residual_indices = f.get_tensor("residual_indices")
            residual_values = f.get_tensor("residual_values")
            floor_correction = (
                f.get_tensor("floor_correction")
                if "floor_correction" in f.keys()
                else _empty_floor_correction()
            )
        return NVFP4ResidualEntry(
            packed=packed,
            residual_indices=residual_indices,
            residual_values=residual_values,
            rel_error=float(meta["rel_error"]),
            nvfp4_rel_error=float(meta["nvfp4_rel_error"]),
            extra_bpe=float(meta["extra_bpe"]),
            residual_bpe=float(meta["residual_bpe"]),
            index_bits=int(meta["index_bits"]),
            residual_bits=int(meta["residual_bits"]),
            selection_policy=meta.get("selection_policy", "residual_rms"),
            floor_correction_policy=meta.get("floor_correction_policy", "none"),
            floor_correction=floor_correction,
            floor_correction_bpe=float(meta.get("floor_correction_bpe", 0.0)),
            corrected_nvfp4_rel_error=(
                float(meta["corrected_nvfp4_rel_error"])
                if meta.get("corrected_nvfp4_rel_error") is not None
                else None
            ),
            floor_level=float(meta.get("floor_level", 0.0)),
            floor_width=float(meta.get("floor_width", 0.0)),
            floor_robust_scale=float(meta.get("floor_robust_scale", 0.0)),
            floor_blocks=int(meta.get("floor_blocks", 0)),
            super_floor_blocks=int(meta.get("super_floor_blocks", 0)),
        )

    def dequantize(
        self,
        layer_name: str,
        device: str | torch.device = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> Optional[torch.Tensor]:
        entry = self.entry(layer_name)
        if entry is None:
            return None
        return entry.dequantize(device=device, dtype=dtype)
