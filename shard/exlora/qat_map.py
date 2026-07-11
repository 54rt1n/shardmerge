# shard/exlora/qat_map.py
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

"""Build and cache the per-parameter QAT projection map.

The QAT adaptation `A_qat = W_qat - W_base` is cached **per parameter** as a
packed NVFP4 map (~4.5 bits/elem). This is the research's dense low-bit map: it
is both a standalone NVFP4 compressor of the survival field and the source of a
per-parameter sensitivity used to make finetune deltas more compressible for the
same behavioral effect.

Per-parameter resolution matters: a sensitivity-weighted projection must steer
compression error parameter-by-parameter (keep precision where the model was
quantization-sensitive, shed it where it was not). Channel/row aggregates are
kept only as cheap diagnostics.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional
import json
import logging

import torch
from safetensors.torch import save_file
from safetensors import safe_open

from ..index import HFMultiModelIndex
from ..download import DownloadManager
from .config import ExtractConfig
from .index import FlatModelIndex
from .nvfp4 import nvfp4_quantize, BLOCK_SIZE
from .nvfp4_kernel import (
    NVFP4PackedTensor, pack_reference_nvfp4, is_triton_nvfp4_available,
    nvfp4_quantize_triton,
)
from .sensitivity_nvfp4 import BlockReduce, block_scores

logger = logging.getLogger(__name__)


def _corr(x: torch.Tensor, y: torch.Tensor) -> float:
    x = x.flatten().to(torch.float32); y = y.flatten().to(torch.float32)
    x = x - x.mean(); y = y - y.mean()
    return float((x @ y / (x.norm() * y.norm()).clamp_min(1e-12)).item())


def pack_a_qat(a_qat: torch.Tensor) -> NVFP4PackedTensor:
    """Pack a per-parameter delta to NVFP4 (triton on cuda, torch oracle else)."""
    if a_qat.is_cuda and is_triton_nvfp4_available():
        return nvfp4_quantize_triton(a_qat.contiguous())
    return pack_reference_nvfp4(nvfp4_quantize(a_qat))


@dataclass
class QATMapEntry:
    """Per-tensor QAT map: packed per-parameter NVFP4 + cheap diagnostics."""
    packed: NVFP4PackedTensor    # NVFP4(A_qat), per-parameter
    col_norm: torch.Tensor       # [in]  diagnostic
    row_norm: torch.Tensor       # [out] diagnostic
    rel_norm: float
    corr_weight: float


def tensor_map_entry(a_qat: torch.Tensor, w_base: torch.Tensor) -> QATMapEntry:
    if a_qat.ndim != 2:
        raise ValueError(f"a_qat must be 2D, got {tuple(a_qat.shape)}")
    a = a_qat.to(torch.float32)
    wb = w_base.to(torch.float32)
    return QATMapEntry(
        packed=pack_a_qat(a_qat),
        col_norm=a.norm(dim=0).cpu(),
        row_norm=a.norm(dim=1).cpu(),
        rel_norm=float((a.norm() / wb.norm().clamp_min(1e-12)).item()),
        corr_weight=_corr(a.abs(), wb.abs()),
    )


class QATMap:
    """Cached per-parameter QAT map over targeted tensors."""

    def __init__(self, base_model: str, qat_model: str, entries: Dict[str, QATMapEntry]):
        self.base_model = base_model
        self.qat_model = qat_model
        self.entries = entries

    def __contains__(self, name: str) -> bool:
        return name in self.entries

    def a_qat(self, name: str, device="cpu", dtype=torch.float32) -> Optional[torch.Tensor]:
        """Decode the per-parameter A_qat (NVFP4 round-trip) for a tensor."""
        e = self.entries.get(name)
        if e is None:
            return None
        packed = e.packed
        # move packed payload to the requested device for decode
        moved = NVFP4PackedTensor(
            packed.packed_codes.to(device), packed.block_scale.to(device),
            packed.global_scale, packed.shape, packed.block_size,
        )
        return moved.dequantize(dtype=dtype)

    def sensitivity(self, name: str, device="cpu", power: float = 1.0,
                    eps: float = 1e-8, normalize: bool = True) -> Optional[torch.Tensor]:
        """Per-parameter sensitivity weight `w = |A_qat|^power` (normalized mean 1)."""
        a = self.a_qat(name, device=device)
        if a is None:
            return None
        w = (a.abs() + eps) ** power
        if normalize:
            w = w / w.mean().clamp_min(eps)
        return w

    def block_sensitivity(
        self,
        name: str,
        device="cpu",
        power: float = 1.0,
        eps: float = 1e-8,
        normalize: bool = True,
        block_size: int = BLOCK_SIZE,
        reduce: BlockReduce = "mean",
    ) -> Optional[torch.Tensor]:
        """Sensitivity scores on the same flattened block grid as NVFP4."""
        w = self.sensitivity(name, device=device, power=power, eps=eps, normalize=normalize)
        if w is None:
            return None
        return block_scores(w, block_size=block_size, reduce=reduce)

    def col_metric(self, name: str, normalize: bool = True, eps: float = 1e-8) -> Optional[torch.Tensor]:
        e = self.entries.get(name)
        if e is None:
            return None
        c = e.col_norm.to(torch.float32).clamp_min(eps)
        return c / c.mean().clamp_min(eps) if normalize else c

    def save(self, path: Path) -> None:
        path = Path(path); path.mkdir(parents=True, exist_ok=True)
        tensors = {}
        manifest = {"base_model": self.base_model, "qat_model": self.qat_model,
                    "schema_version": 2, "tensors": {}}
        for name, e in self.entries.items():
            tensors[f"{name}::codes"] = e.packed.packed_codes.contiguous().cpu()
            tensors[f"{name}::block_scale"] = e.packed.block_scale.to(torch.float32).contiguous().cpu()
            tensors[f"{name}::col_norm"] = e.col_norm.contiguous()
            tensors[f"{name}::row_norm"] = e.row_norm.contiguous()
            manifest["tensors"][name] = {
                "rel_norm": e.rel_norm, "corr_weight": e.corr_weight,
                "global_scale": e.packed.global_scale, "shape": list(e.packed.shape),
                "block_size": e.packed.block_size,
            }
        save_file(tensors, str(path / "qat_map.safetensors"),
                  metadata={"format": "pt", "content": "qat_nvfp4_map"})
        (path / "qat_map.json").write_text(json.dumps(manifest, indent=2))
        logger.info(f"Saved per-parameter QAT NVFP4 map ({len(self.entries)} tensors) to {path}")

    @classmethod
    def load(cls, path: Path) -> "QATMap":
        path = Path(path)
        manifest = json.loads((path / "qat_map.json").read_text())
        raw: Dict[str, Dict[str, torch.Tensor]] = {}
        with safe_open(str(path / "qat_map.safetensors"), framework="pt") as f:
            for k in f.keys():
                name, _, kind = k.rpartition("::")
                raw.setdefault(name, {})[kind] = f.get_tensor(k)
        entries = {}
        for name, meta in manifest["tensors"].items():
            r = raw[name]
            packed = NVFP4PackedTensor(
                r["codes"], r["block_scale"], float(meta["global_scale"]),
                torch.Size(meta["shape"]), int(meta["block_size"]),
            )
            entries[name] = QATMapEntry(packed, r["col_norm"], r["row_norm"],
                                        meta["rel_norm"], meta["corr_weight"])
        return cls(manifest["base_model"], manifest["qat_model"], entries)


class QATMapBuilder:
    """Streams (qat, base) once and packs the per-parameter NVFP4 map."""

    def __init__(self, config: ExtractConfig, index_manager: Optional[HFMultiModelIndex] = None):
        self.config = config
        self.index_manager = index_manager or FlatModelIndex(
            download_manager=DownloadManager(
                storage_path=config.storage_path, clean_cache=config.clean_cache),
            cache_path=config.cache_path,
        )

    async def build(self, device: Optional[str] = None) -> QATMap:
        device = device or self.config.device
        await self.index_manager.add_model(self.config.base_model)
        await self.index_manager.add_model(self.config.finetune_model)
        layer_order = self.index_manager.get_layer_order(self.config.base_model)
        targets = [n for n in layer_order if self.config.is_target(n)]
        logger.info(f"Packing per-parameter QAT NVFP4 map over {len(targets)} tensors")

        entries: Dict[str, QATMapEntry] = {}
        for name in targets:
            wb = (await self.index_manager.get_tensor(self.config.base_model, name, device=device).get()).to(torch.float32)
            wq = (await self.index_manager.get_tensor(self.config.finetune_model, name, device=device).get()).to(torch.float32)
            a = wq - wb
            if a.ndim == 2:
                entries[name] = tensor_map_entry(a, wb)
                logger.info(f"{name}: rel_norm {entries[name].rel_norm:.4g} "
                            f"corr {entries[name].corr_weight:.3f} "
                            f"bpe {entries[name].packed.bits_per_element:.2f}")
            del wb, wq, a
            cache = getattr(self.index_manager, "_tensor_downloads", None)
            if cache is not None:
                cache.pop((self.config.base_model, name), None)
                cache.pop((self.config.finetune_model, name), None)
            if device.startswith("cuda"):
                torch.cuda.empty_cache()
        return QATMap(self.config.base_model, self.config.finetune_model, entries)


async def build_qat_map(config: ExtractConfig, index_manager=None, device: Optional[str] = None) -> QATMap:
    qat_map = await QATMapBuilder(config, index_manager=index_manager).build(device=device)
    qat_map.save(config.output_path)
    return qat_map


class SensitivityWeightedProjection:
    """Per-parameter sensitivity-weighted projection `P_qat`.

    ``apply`` reweights a delta by per-parameter QAT sensitivity so that a
    subsequent low-rank or low-bit code keeps precision in compression-sensitive
    parameters. ``inverse`` undoes the weighting. Because the weighting is
    per-parameter (non-separable), the result is the research's projected
    artifact (decoded via ``inverse``), not a stock PEFT ``B@A``.

    This is only safe as a dense projected artifact. Do not feed this through a
    low-rank SVD and then call ``inverse``: non-separable per-parameter weights
    turn that into an unstable weighted low-rank approximation and amplify
    residuals wherever sensitivity is near zero.
    """

    def __init__(self, qat_map: QATMap, power: float = 0.5, eps: float = 1e-8):
        self.qat_map = qat_map
        self.power = power
        self.eps = eps
        self._w_cache: Dict[str, torch.Tensor] = {}

    def weight(self, name: str, device) -> Optional[torch.Tensor]:
        w = self.qat_map.sensitivity(name, device=device, power=self.power, eps=self.eps)
        return w

    def apply(self, delta: torch.Tensor, name: str) -> torch.Tensor:
        w = self.weight(name, delta.device)
        if w is None:
            return delta
        return delta * w

    def inverse(self, z: torch.Tensor, name: str) -> torch.Tensor:
        w = self.weight(name, z.device)
        if w is None:
            return z
        return z / w.clamp_min(self.eps)
