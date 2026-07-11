# shard/exlora/extract.py
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

"""Streaming, high-fidelity LoRA extraction.

Walks the base model's layer order, and for each targeted Linear weight
loads the base and finetune tensors (reusing the existing
:class:`shard.index.HFMultiModelIndex` shard streaming / caching), computes
the delta, and factors it into a low-rank LoRA adapter.
"""

from dataclasses import dataclass
from typing import List, Optional
import asyncio
import logging
import torch

from ..index import HFMultiModelIndex
from ..download import DownloadManager
from .config import ExtractConfig
from .index import FlatModelIndex
from .peft import AdapterAccumulator
from .projection import Projection, TensorProjectionContext, load_projection
from .nvfp4_residual import (
    NVFP4ResidualWriter,
    build_nvfp4_residual_entry,
    parse_torch_dtype,
)
from .svd import (
    basis_svd,
    retained_energy,
    select_rank,
    select_signal_rank,
    spectrum_diagnostics,
)
from .functions import split_svd_to_lora, randomized_svd

logger = logging.getLogger(__name__)

# Optional embedding / LM-head capture (config.extract_embeddings / extract_lm_head).
EMBED_TENSOR = "model.language_model.embed_tokens.weight"
LM_HEAD_TENSOR = "lm_head.weight"
# Vocab-scale tensors (embed/head, ~262k rows): use randomized SVD in the LoRA path
# so we never materialize the full economy U (5.6 GB) and risk OOM on the last target.
LARGE_TENSOR_DIM = 50000


@dataclass
class LayerStat:
    """Per-layer extraction record, useful for tuning and reporting."""
    layer_name: str
    shape: tuple
    rank: int
    rel_error: float
    module_type: str | None = None
    layer_index: int | None = None
    num_layers: int | None = None
    depth_fraction: float | None = None
    depth_bucket: str = "unknown"
    artifact_type: str = "lora"
    rank_reason: str = "target_error"
    retained_energy: float | None = None
    effective_rank: float | None = None
    signal_rank: int | None = None
    knee_rank: int | None = None
    noise_floor_ratio: float | None = None
    nvfp4_rel_error: float | None = None
    corrected_nvfp4_rel_error: float | None = None
    nvfp4_bpe: float | None = None
    floor_correction_bpe: float | None = None
    residual_bpe: float | None = None
    total_bpe: float | None = None
    residual_blocks: int | None = None
    floor_blocks: int | None = None
    super_floor_blocks: int | None = None


@dataclass(frozen=True)
class RankSelection:
    """Rank choice plus spectrum diagnostics for reporting."""

    rank: int
    rel_error: float
    reason: str
    retained_energy: float
    effective_rank: float
    signal_rank: int | None
    knee_rank: int | None
    noise_floor_ratio: float


class LoraExtractor:
    """Extracts a PEFT LoRA adapter from a finetune/base model pair."""

    def __init__(
        self,
        config: ExtractConfig,
        index_manager: Optional[HFMultiModelIndex] = None,
        projection: Optional[Projection] = None,
    ):
        self.config = config
        self.index_manager = index_manager or self._build_index_manager()
        # P_qat seam. Defaults to identity (raw SVD) unless a QAT projection
        # model is configured. See shard/exlora/projection.py.
        self.projection = projection or load_projection(config)
        self.stats: List[LayerStat] = []
        self._num_layers: int | None = None

    def _build_index_manager(self) -> HFMultiModelIndex:
        download_manager = DownloadManager(
            storage_path=self.config.storage_path,
            clean_cache=self.config.clean_cache,
        )
        return FlatModelIndex(
            download_manager=download_manager,
            cache_path=self.config.cache_path,
        )

    async def initialize(self):
        """Register both models and validate compatible architectures."""
        await self.index_manager.add_model(self.config.base_model)
        await self.index_manager.add_model(self.config.finetune_model)

        base_keys = self.index_manager.get_model_keys(self.config.base_model)
        ft_keys = self.index_manager.get_model_keys(self.config.finetune_model)

        missing = {k for k in base_keys if self.config.is_target(k)} - ft_keys
        if missing:
            raise ValueError(
                f"Finetune model {self.config.finetune_model} is missing "
                f"targeted tensors present in base: {sorted(missing)}"
            )

    def _base_name_for(self, layer_name: str) -> str:
        """Base tensor to diff against. On a TIED base the output projection is the
        token embedding, so an untied finetune's lm_head is diffed against base
        embed_tokens."""
        if layer_name == LM_HEAD_TENSOR and LM_HEAD_TENSOR not in getattr(self, "_base_keys", ()):
            return EMBED_TENSOR
        return layer_name

    async def _load_delta(self, layer_name: str, device: str) -> Optional[torch.Tensor]:
        """Load base+finetune and return the float32 delta (or None if non-2D)."""
        base_name = self._base_name_for(layer_name)
        base_promise = self.index_manager.get_tensor(
            self.config.base_model, base_name, device=device
        )
        ft_promise = self.index_manager.get_tensor(
            self.config.finetune_model, layer_name, device=device
        )
        base_tensor, ft_tensor = await asyncio.gather(base_promise.get(), ft_promise.get())

        delta = (ft_tensor.to(torch.float32) - base_tensor.to(torch.float32))
        del base_tensor, ft_tensor
        # The index caches every loaded tensor and never evicts; for large
        # models that is hundreds of GB. Drop these now that we have the delta.
        self._evict_from_index(layer_name)
        if base_name != layer_name:
            self._evict_from_index(base_name)

        if delta.ndim != 2:
            logger.warning(f"Skipping non-2D targeted tensor {layer_name} {tuple(delta.shape)}")
            return None
        return delta

    async def _factor_layer(self, layer_name: str, device: str) -> Optional[LayerStat]:
        """Stage the per-tensor pipeline: delta -> project -> SVD -> defuzz ->
        rank-select -> bf16 factors -> fold."""
        delta = await self._load_delta(layer_name, device)
        if delta is None:
            return None
        # A bit-for-bit zero delta means the finetune never touched this weight
        # (e.g. a layer-frozen LoRA that adapts only the upper stack). Emitting a
        # rank-1 zero adapter for it would bloat the artifact with no signal, so
        # skip it entirely.
        if not bool(torch.any(delta != 0)):
            logger.info(f"{layer_name}: frozen (zero delta), skipping")
            return None
        shape = tuple(delta.shape)
        self.projection.prepare(
            TensorProjectionContext(layer_name=layer_name, delta_shape=shape)
        )

        # Stage 1 - projection P_qat. Identity for the raw-SVD baseline.
        # TODO(projection): transport the delta into the QAT-derived gauge.
        z = self.projection.apply(delta, layer_name)
        del delta

        # Stage 2 - original-basis SVD. Always computed; needed for both rank
        # selection and defuzzing. Vocab-scale tensors (embed/head) use randomized
        # SVD at the rank cap to avoid a 5.6 GB full U.
        if max(z.shape) > LARGE_TENSOR_DIM:
            _, mx = self.config.rank_bounds(layer_name)
            q = min(max(mx, 1), min(z.shape))
            U, S, Vh = randomized_svd(z, q, oversampling=16, power_iterations=2,
                                      compute_dtype=torch.float32)
        else:
            U, S, Vh = basis_svd(z, compute_dtype=torch.float32)

        # Stage 3 - defuzz toward nvfp4-stable directions. Identity for now.
        # TODO(defuzz): reflect the retained subspace onto quantization-stable
        # points using the original-basis SVD above and the QAT survival map.
        U, S, Vh = self.projection.defuzz(U, S, Vh, z, layer_name)
        del z

        # Stage 4 - rank selection + bf16 LoRA factors (PEFT precision).
        # Honor per-module rank budgets (reallocate rank by activation error).
        mn, mx = self.config.rank_bounds(layer_name)
        selection = self._select_rank(S, mn, mx)
        rank = selection.rank
        rel_error = selection.rel_error
        rank = max(rank, 1)
        lora_A, lora_B = split_svd_to_lora(U, S, Vh, rank, out_dtype=self.config.out_dtype)
        del U, S, Vh

        # Stage 5 - fold any projection back to weight coordinates so the
        # emitted adapter stays PEFT-compatible. Identity for the raw path.
        # TODO(projection): fold the inverse transform into lora_A/lora_B.
        lora_A, lora_B = self.projection.fold(lora_A, lora_B, layer_name)

        # Keep the accumulated adapter on CPU so GPU memory stays bounded to
        # one layer's working set during the run.
        self.accumulator.add(layer_name, lora_A.cpu(), lora_B.cpu(), rank)
        logger.info(
            f"{layer_name}: rank {rank} ({selection.reason}, err {float(rel_error):.4g}, "
            f"energy {selection.retained_energy:.4g}) shape {shape}"
        )
        return LayerStat(
            layer_name=layer_name,
            shape=shape,
            rank=rank,
            rel_error=float(rel_error),
            **self._layer_metadata(layer_name),
            artifact_type="lora",
            rank_reason=selection.reason,
            retained_energy=selection.retained_energy,
            effective_rank=selection.effective_rank,
            signal_rank=selection.signal_rank,
            knee_rank=selection.knee_rank,
            noise_floor_ratio=selection.noise_floor_ratio,
        )

    async def _encode_nvfp4_residual_layer(self, layer_name: str, device: str) -> Optional[LayerStat]:
        """Stage the dense low-bit path: delta -> NVFP4 -> magnitude residual sidecar."""
        # Vocab-scale tensors (embed_tokens / lm_head, ~262k rows) need ~17 GB of
        # transient fp32 (base + ft + delta) which OOMs the GPU on top of the
        # resident model. They are dense (no SVD) and only two of them, so encode
        # them on CPU; build_nvfp4_residual_entry follows the input tensor's device.
        enc_device = "cpu" if layer_name in (EMBED_TENSOR, LM_HEAD_TENSOR) else device
        delta = await self._load_delta(layer_name, enc_device)
        if delta is None:
            return None
        shape = tuple(delta.shape)
        entry = build_nvfp4_residual_entry(
            delta,
            extra_bpe=self.config.nvfp4_extra_bpe,
            residual_dtype=parse_torch_dtype(self.config.nvfp4_residual_dtype),
            index_bits=self.config.nvfp4_index_bits,
            selection_policy=self.config.nvfp4_selection_policy,
            floor_correction_policy=self.config.nvfp4_floor_correction,
            floor_tolerance=self.config.nvfp4_floor_tolerance,
            floor_effect_threshold=self.config.nvfp4_floor_effect_threshold,
        )
        del delta

        self.nvfp4_writer.add(layer_name, entry)
        logger.info(
            f"{layer_name}: nvfp4_residual err {entry.rel_error:.4g} "
            f"(nvfp4 {entry.nvfp4_rel_error:.4g}, bpe {entry.total_bpe:.3f}, "
            f"sidecar_blocks {entry.residual_indices.numel()}) shape {shape}"
        )
        return LayerStat(
            layer_name=layer_name,
            shape=shape,
            rank=0,
            rel_error=entry.rel_error,
            **self._layer_metadata(layer_name),
            artifact_type="nvfp4_residual",
            rank_reason=entry.selection_policy,
            nvfp4_rel_error=entry.nvfp4_rel_error,
            corrected_nvfp4_rel_error=entry.corrected_nvfp4_rel_error,
            nvfp4_bpe=entry.packed.bits_per_element,
            floor_correction_bpe=entry.floor_correction_bpe,
            residual_bpe=entry.residual_bpe,
            total_bpe=entry.total_bpe,
            residual_blocks=int(entry.residual_indices.numel()),
            floor_blocks=entry.floor_blocks,
            super_floor_blocks=entry.super_floor_blocks,
        )

    def _select_rank(self, singular_values: torch.Tensor, min_rank: int, max_rank: int) -> RankSelection:
        """Choose rank according to the configured policy.

        ``error`` preserves the original target-error behavior. ``signal`` is
        for bf16-merged deltas with a visible spectral noise floor.
        ``error_or_signal`` uses the smaller signal rank only when a credible
        floor/knee is found; otherwise it falls back to target-error selection.
        """
        policy = getattr(self.config, "rank_policy", "error")
        diag = spectrum_diagnostics(
            singular_values,
            min_rank=min_rank,
            max_rank=max_rank,
            noise_floor_multiplier=self.config.noise_floor_multiplier,
            min_knee_drop=self.config.min_knee_drop,
            tail_fraction=self.config.tail_fraction,
        )
        error_rank, error_rel = select_rank(
            singular_values,
            self.config.target_error,
            min_rank=min_rank,
            max_rank=max_rank,
        )
        if policy == "error":
            reason = (
                "max_rank_cap"
                if error_rank == min(max_rank, len(singular_values)) and error_rel > self.config.target_error
                else "target_error"
            )
            return self._rank_selection(error_rank, error_rel, reason, singular_values, diag)

        signal_rank, signal_rel, diag = select_signal_rank(
            singular_values,
            min_rank=min_rank,
            max_rank=max_rank,
            noise_floor_multiplier=self.config.noise_floor_multiplier,
            min_knee_drop=self.config.min_knee_drop,
            tail_fraction=self.config.tail_fraction,
        )
        if policy == "signal":
            return self._rank_selection(signal_rank, signal_rel, diag.signal_reason, singular_values, diag)
        if policy == "error_or_signal":
            if diag.signal_rank is not None and signal_rank < error_rank:
                return self._rank_selection(signal_rank, signal_rel, diag.signal_reason, singular_values, diag)
            reason = (
                "max_rank_cap"
                if error_rank == min(max_rank, len(singular_values)) and error_rel > self.config.target_error
                else "target_error"
            )
            return self._rank_selection(error_rank, error_rel, reason, singular_values, diag)

        raise ValueError(f"Unknown rank_policy {policy!r}; expected error, signal, or error_or_signal")

    def _rank_selection(self, rank, rel_error, reason, singular_values, diag) -> RankSelection:
        return RankSelection(
            rank=int(rank),
            rel_error=float(rel_error),
            reason=reason,
            retained_energy=retained_energy(singular_values, int(rank)),
            effective_rank=diag.effective_rank,
            signal_rank=diag.signal_rank,
            knee_rank=diag.knee_rank,
            noise_floor_ratio=diag.noise_floor_ratio,
        )

    def _evict_from_index(self, layer_name: str):
        """Drop this layer's tensors from the index's in-memory cache."""
        cache = getattr(self.index_manager, "_tensor_downloads", None)
        if cache is None:
            return
        cache.pop((self.config.base_model, layer_name), None)
        cache.pop((self.config.finetune_model, layer_name), None)

    def _layer_metadata(self, layer_name: str) -> dict:
        """Module and depth metadata for report grouping."""
        if not self.config.position_policy_enabled:
            return {
                "module_type": self.config.module_of(layer_name),
                "layer_index": None,
                "num_layers": None,
                "depth_fraction": None,
                "depth_bucket": "unknown",
            }
        from shard.ranker.layer_position import layer_position

        pos = layer_position(
            layer_name,
            num_layers=self._num_layers,
            early_fraction=self.config.position_early_fraction,
            late_fraction=self.config.position_late_fraction,
        )
        return {
            "module_type": self.config.module_of(layer_name),
            "layer_index": pos.layer_index,
            "num_layers": pos.num_layers,
            "depth_fraction": pos.depth_fraction,
            "depth_bucket": pos.depth_bucket,
        }

    async def extract(self, device: Optional[str] = None):
        """Run the full extraction and write the adapter to disk."""
        device = device or self.config.device
        await self.initialize()

        if self.config.artifact_type == "lora":
            self.accumulator = AdapterAccumulator(
                base_model=self.config.base_model,
                target_modules=self.config.target_modules,
            )
        elif self.config.artifact_type == "nvfp4_residual":
            self.nvfp4_writer = NVFP4ResidualWriter(
                self.config.output_path,
                base_model=self.config.base_model,
                finetune_model=self.config.finetune_model,
                target_modules=self.config.target_modules,
                extra_bpe=self.config.nvfp4_extra_bpe,
                residual_dtype=parse_torch_dtype(self.config.nvfp4_residual_dtype),
                index_bits=self.config.nvfp4_index_bits,
                selection_policy=self.config.nvfp4_selection_policy,
                floor_correction_policy=self.config.nvfp4_floor_correction,
                floor_tolerance=self.config.nvfp4_floor_tolerance,
                floor_effect_threshold=self.config.nvfp4_floor_effect_threshold,
            )
        else:
            raise ValueError(
                f"Unknown artifact_type {self.config.artifact_type!r}; "
                "expected 'lora' or 'nvfp4_residual'"
            )

        layer_order = self.index_manager.get_layer_order(self.config.base_model)
        targets = [name for name in layer_order if self.config.is_target(name)]
        # Optional embedding / LM-head capture (appended to the body targets).
        self._base_keys = set(self.index_manager.get_model_keys(self.config.base_model))
        ft_keys = set(self.index_manager.get_model_keys(self.config.finetune_model))
        if self.config.extract_embeddings and EMBED_TENSOR in ft_keys and EMBED_TENSOR not in targets:
            targets.append(EMBED_TENSOR)
            logger.info("capturing token embeddings (extract_embeddings)")
        if self.config.extract_lm_head and LM_HEAD_TENSOR in ft_keys and LM_HEAD_TENSOR not in targets:
            targets.append(LM_HEAD_TENSOR)
            tied = LM_HEAD_TENSOR not in self._base_keys
            logger.info(f"capturing lm_head (extract_lm_head; base {'tied -> vs embed' if tied else 'untied'})")
        from shard.ranker.layer_position import infer_num_layers

        self._num_layers = infer_num_layers(layer_order) if self.config.position_policy_enabled else None
        logger.info(f"Extracting {self.config.artifact_type} for {len(targets)} targeted layers")

        for layer_name in targets:
            if self.config.artifact_type == "lora":
                stat = await self._factor_layer(layer_name, device)
            else:
                stat = await self._encode_nvfp4_residual_layer(layer_name, device)
            if stat is not None:
                self.stats.append(stat)
            if device.startswith("cuda"):
                torch.cuda.empty_cache()

        if self.config.artifact_type == "lora":
            self.accumulator.write(self.config.output_path)
            result = self.accumulator
        else:
            self.nvfp4_writer.write_manifest()
            result = self.nvfp4_writer
        self._write_report()
        return result

    def _write_report(self):
        """Write a small human-readable extraction summary."""
        if not self.stats:
            return
        ranks = [s.rank for s in self.stats]
        errors = [s.rel_error for s in self.stats]
        artifact_label = {
            "lora": "LoRA",
            "nvfp4_residual": "NVFP4 Residual",
        }.get(self.config.artifact_type, self.config.artifact_type)
        lines = [
            f"# {artifact_label} Extraction Report",
            f"Base model: {self.config.base_model}",
            f"Finetune model: {self.config.finetune_model}",
            f"Target error: {self.config.target_error}",
            f"Rank range: [{self.config.min_rank}, {self.config.max_rank}]",
            f"Layers: {len(self.stats)}",
            f"Rank min/mean/max: {min(ranks)} / {sum(ranks) / len(ranks):.1f} / {max(ranks)}",
            f"Worst layer rel error: {max(errors):.4g}",
            "",
            "| layer | module | layer_idx | depth | bucket | artifact | shape | rank | rel_error | nvfp4_error | corrected_nvfp4_error | nvfp4_bpe | floor_bpe | residual_bpe | total_bpe | residual_blocks | floor_blocks | super_floor_blocks | energy | eff_rank | signal_rank | knee_rank | noise_floor/S0 | rank_reason |",
            "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
        ]
        for s in self.stats:
            lines.append(
                f"| {s.layer_name} | {s.module_type or '-'} | {_fmt_optional(s.layer_index)} | "
                f"{_fmt_optional(s.depth_fraction)} | {s.depth_bucket} | {s.artifact_type} | "
                f"{s.shape} | {s.rank} | {s.rel_error:.4g} | "
                f"{_fmt_optional(s.nvfp4_rel_error)} | {_fmt_optional(s.corrected_nvfp4_rel_error)} | "
                f"{_fmt_optional(s.nvfp4_bpe)} | {_fmt_optional(s.floor_correction_bpe)} | "
                f"{_fmt_optional(s.residual_bpe)} | {_fmt_optional(s.total_bpe)} | "
                f"{_fmt_optional(s.residual_blocks)} | {_fmt_optional(s.floor_blocks)} | "
                f"{_fmt_optional(s.super_floor_blocks)} | "
                f"{_fmt_optional(s.retained_energy)} | {_fmt_optional(s.effective_rank)} | "
                f"{_fmt_optional(s.signal_rank)} | {_fmt_optional(s.knee_rank)} | "
                f"{_fmt_optional(s.noise_floor_ratio)} | {s.rank_reason} |"
            )

        _append_position_summaries(lines, self.stats)

        with open(self.config.output_path / "EXTRACTION_REPORT.md", "w") as f:
            f.write("\n".join(lines) + "\n")


def _fmt_optional(value) -> str:
    if value is None:
        return "-"
    if isinstance(value, int):
        return str(value)
    return f"{float(value):.4g}"


def _mean_optional(stats: list[LayerStat], attr: str):
    values = [getattr(s, attr) for s in stats if getattr(s, attr) is not None]
    if not values:
        return None
    return sum(float(v) for v in values) / len(values)


def _append_position_summaries(lines: list[str], stats: list[LayerStat]) -> None:
    """Append module/depth summary tables to the Markdown report."""
    lines.extend(["", "## Position Summaries"])
    _append_group_summary(lines, stats, "By Depth Bucket", ("depth_bucket",))
    _append_group_summary(lines, stats, "By Module", ("module_type",))
    _append_group_summary(lines, stats, "By Module And Depth", ("module_type", "depth_bucket"))


def _append_group_summary(
    lines: list[str],
    stats: list[LayerStat],
    title: str,
    keys: tuple[str, ...],
) -> None:
    groups: dict[tuple[str, ...], list[LayerStat]] = {}
    for stat in stats:
        key = tuple(str(getattr(stat, k) or "unknown") for k in keys)
        groups.setdefault(key, []).append(stat)

    lines.extend([
        "",
        f"### {title}",
        "",
        "| group | count | rank_mean | rel_error_mean | energy_mean | nvfp4_error_mean | corrected_nvfp4_error_mean | residual_blocks_mean | super_floor_blocks_mean |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ])
    for key in sorted(groups):
        group_stats = groups[key]
        label = " / ".join(key)
        lines.append(
            f"| {label} | {len(group_stats)} | "
            f"{_fmt_optional(_mean_optional(group_stats, 'rank'))} | "
            f"{_fmt_optional(_mean_optional(group_stats, 'rel_error'))} | "
            f"{_fmt_optional(_mean_optional(group_stats, 'retained_energy'))} | "
            f"{_fmt_optional(_mean_optional(group_stats, 'nvfp4_rel_error'))} | "
            f"{_fmt_optional(_mean_optional(group_stats, 'corrected_nvfp4_rel_error'))} | "
            f"{_fmt_optional(_mean_optional(group_stats, 'residual_blocks'))} | "
            f"{_fmt_optional(_mean_optional(group_stats, 'super_floor_blocks'))} |"
        )
