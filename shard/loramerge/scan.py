# shard/loramerge/scan.py
# Copyright (C) 2026 Martin Bukowski
#
# This software is free software: you can redistribute it and/or
# modify it under the terms of the GNU Lesser General Public License as
# published by the Free Software Foundation, either version 3 of the
# License, or (at your option) any later version.

"""Inspect a LoRA-merge config without performing a merge."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any
import json
import math
import re

import click
from safetensors import safe_open

from ..loraplay.adapter import parse_lora_key
from .config import LoraMergeConfig, LoraMergeSourceConfig, is_embedding_tensor, is_lm_head_tensor


_LAYER_RE = re.compile(r"(?:^|\.)(?:layers|blocks|h)\.(\d+)\.")


@dataclass
class SourceScanReport:
    """Summary of one configured model/artifact source."""

    name: str
    path: str
    resolved_path: str
    kind: str | None
    role: str
    representation: str
    weight: float = 1.0
    enabled: bool = True
    status: str = "ok"
    notes: list[str] = field(default_factory=list)
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class LoraMergeScanReport:
    """Report emitted by ``python -m shard.loramerge.scan``."""

    config_path: str | None
    base_model: str
    resolved_base_model: str
    output_dir: str
    merge_rule: str
    scaffold_format: str
    target_modules: list[str]
    embedding_model: str
    resolved_embedding_model: str
    lm_head_model: str
    resolved_lm_head_model: str
    sources: list[SourceScanReport]
    base_checkpoint: SourceScanReport


def _resolve_path(path: str | Path, storage_dir: str | Path) -> Path:
    raw = Path(path)
    if raw.exists():
        return raw
    stored = Path(storage_dir) / raw
    if stored.exists():
        return stored
    return raw


def _dir_size(path: Path) -> int:
    if path.is_file():
        return path.stat().st_size
    if not path.exists():
        return 0
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def _fmt_bytes(nbytes: int | float) -> str:
    n = float(nbytes)
    for suffix in ("B", "KB", "MB", "GB", "TB"):
        if abs(n) < 1024.0 or suffix == "TB":
            if suffix == "B":
                return f"{int(n)} {suffix}"
            return f"{n:.2f} {suffix}"
        n /= 1024.0
    return f"{nbytes} B"


def _counter_dict(counter: Counter[str | int]) -> dict[str, int]:
    return {str(k): int(v) for k, v in sorted(counter.items(), key=lambda item: str(item[0]))}


def _module_of(weight_name: str) -> str | None:
    if not weight_name.endswith(".weight"):
        return None
    return weight_name[: -len(".weight")].split(".")[-1]


def _layer_index(weight_name: str) -> int | None:
    match = _LAYER_RE.search(weight_name)
    return int(match.group(1)) if match else None


def _stats(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "min": None, "mean": None, "max": None}
    return {
        "count": len(values),
        "min": min(values),
        "mean": sum(values) / len(values),
        "max": max(values),
    }


def _safe_shape(handle: Any, key: str) -> list[int] | None:
    try:
        return list(handle.get_slice(key).get_shape())
    except Exception:
        return None


def _safe_dtype(handle: Any, key: str) -> str | None:
    try:
        return str(handle.get_slice(key).get_dtype())
    except Exception:
        return None


def _checkpoint_weight_map(path: Path) -> tuple[dict[str, str], str]:
    index = path / "model.safetensors.index.json"
    if index.exists():
        data = json.loads(index.read_text())
        return dict(data.get("weight_map") or {}), "sharded_safetensors"

    single = path / "model.safetensors"
    if single.exists():
        with safe_open(str(single), framework="pt") as f:
            return {key: single.name for key in f.keys()}, "single_safetensors"

    return {}, "missing"


def scan_checkpoint(
    model: str,
    *,
    storage_dir: str | Path,
    role: str,
    target_modules: list[str],
    name: str | None = None,
    enabled: bool = True,
) -> SourceScanReport:
    """Inspect a local HuggingFace safetensors checkpoint by index only."""
    resolved = _resolve_path(model, storage_dir)
    label = name or Path(model).name or model
    if not resolved.exists():
        return SourceScanReport(
            name=label,
            path=str(model),
            resolved_path=str(resolved),
            kind="model",
            role=role,
            representation="model_checkpoint",
            enabled=enabled,
            status="missing",
            notes=["path not found locally; scan does not download"],
        )

    weight_map, checkpoint_format = _checkpoint_weight_map(resolved)
    if not weight_map:
        return SourceScanReport(
            name=label,
            path=str(model),
            resolved_path=str(resolved),
            kind="model",
            role=role,
            representation="model_checkpoint",
            enabled=enabled,
            status="no_checkpoint",
            notes=["no model.safetensors.index.json or model.safetensors found"],
            details={"size_bytes": _dir_size(resolved), "size": _fmt_bytes(_dir_size(resolved))},
        )

    modules = Counter()
    layers: set[int] = set()
    embeds: list[str] = []
    heads: list[str] = []
    for tensor_name in weight_map:
        module = _module_of(tensor_name)
        if module in target_modules:
            modules[module] += 1
        layer = _layer_index(tensor_name)
        if layer is not None:
            layers.add(layer)
        if is_embedding_tensor(tensor_name):
            embeds.append(tensor_name)
        if is_lm_head_tensor(tensor_name):
            heads.append(tensor_name)

    return SourceScanReport(
        name=label,
        path=str(model),
        resolved_path=str(resolved),
        kind="model",
        role=role,
        representation="model_checkpoint",
        enabled=enabled,
        details={
            "checkpoint_format": checkpoint_format,
            "tensor_count": len(weight_map),
            "shard_count": len(set(weight_map.values())),
            "size_bytes": _dir_size(resolved),
            "size": _fmt_bytes(_dir_size(resolved)),
            "layer_count": len(layers),
            "layer_min": min(layers) if layers else None,
            "layer_max": max(layers) if layers else None,
            "target_module_counts": _counter_dict(modules),
            "embedding_tensors": sorted(embeds),
            "lm_head_tensors": sorted(heads),
        },
    )


def scan_peft_adapter(path: Path, source: LoraMergeSourceConfig) -> SourceScanReport:
    config_path = path / "adapter_config.json"
    tensor_path = path / "adapter_model.safetensors"
    label = source.name or path.name or source.path
    if not path.exists():
        return SourceScanReport(
            name=label,
            path=source.path,
            resolved_path=str(path),
            kind=source.kind,
            role="delta",
            representation="peft_lora_adapter",
            weight=source.weight,
            enabled=source.enabled,
            status="missing",
        )
    if not config_path.exists() or not tensor_path.exists():
        return SourceScanReport(
            name=label,
            path=source.path,
            resolved_path=str(path),
            kind=source.kind,
            role="delta",
            representation="peft_lora_adapter",
            weight=source.weight,
            enabled=source.enabled,
            status="invalid",
            notes=["expected adapter_config.json and adapter_model.safetensors"],
            details={"size_bytes": _dir_size(path), "size": _fmt_bytes(_dir_size(path))},
        )

    config = json.loads(config_path.read_text())
    factors: dict[str, dict[str, dict[str, Any]]] = {}
    unparsed = 0
    dtypes = Counter()
    with safe_open(str(tensor_path), framework="pt") as f:
        keys = list(f.keys())
        for key in keys:
            try:
                weight_name, which = parse_lora_key(key)
            except ValueError:
                unparsed += 1
                continue
            factors.setdefault(weight_name, {})[which] = {
                "shape": _safe_shape(f, key),
                "dtype": _safe_dtype(f, key),
            }
            dtype = factors[weight_name][which]["dtype"]
            if dtype is not None:
                dtypes[dtype] += 1

    ranks = Counter()
    modules = Counter()
    layers: set[int] = set()
    missing_pairs: list[str] = []
    for weight_name, pair in factors.items():
        if "A" not in pair or "B" not in pair:
            missing_pairs.append(weight_name)
            continue
        a_shape = pair["A"].get("shape")
        b_shape = pair["B"].get("shape")
        if a_shape:
            ranks[int(a_shape[0])] += 1
        elif b_shape and len(b_shape) > 1:
            ranks[int(b_shape[1])] += 1
        module = _module_of(weight_name)
        if module is not None:
            modules[module] += 1
        layer = _layer_index(weight_name)
        if layer is not None:
            layers.add(layer)

    status = "ok" if not missing_pairs else "invalid"
    notes = []
    if unparsed:
        notes.append(f"{unparsed} non-LoRA tensor keys were ignored")
    if missing_pairs:
        notes.append(f"{len(missing_pairs)} adapter targets are missing A or B")

    return SourceScanReport(
        name=label,
        path=source.path,
        resolved_path=str(path),
        kind=source.kind or "peft",
        role="delta",
        representation="peft_lora_adapter",
        weight=source.weight,
        enabled=source.enabled,
        status=status,
        notes=notes,
        details={
            "base_model_name_or_path": config.get("base_model_name_or_path"),
            "peft_type": config.get("peft_type"),
            "target_modules": config.get("target_modules"),
            "configured_rank": config.get("r"),
            "configured_alpha": config.get("lora_alpha"),
            "adapter_tensor_count": len(keys),
            "adapter_target_count": len(factors),
            "size_bytes": _dir_size(path),
            "size": _fmt_bytes(_dir_size(path)),
            "rank_histogram": _counter_dict(ranks),
            "rank_min": min(ranks) if ranks else None,
            "rank_max": max(ranks) if ranks else None,
            "module_counts": _counter_dict(modules),
            "layer_count": len(layers),
            "layer_min": min(layers) if layers else None,
            "layer_max": max(layers) if layers else None,
            "dtype_counts": _counter_dict(dtypes),
        },
    )


def scan_nvfp4_residual(path: Path, source: LoraMergeSourceConfig) -> SourceScanReport:
    manifest_path = path / "exlora_nvfp4_residual.json"
    label = source.name or path.name or source.path
    if not path.exists():
        return SourceScanReport(
            name=label,
            path=source.path,
            resolved_path=str(path),
            kind=source.kind,
            role="delta",
            representation="dense_nvfp4_residual",
            weight=source.weight,
            enabled=source.enabled,
            status="missing",
        )
    if not manifest_path.exists():
        return SourceScanReport(
            name=label,
            path=source.path,
            resolved_path=str(path),
            kind=source.kind,
            role="delta",
            representation="dense_nvfp4_residual",
            weight=source.weight,
            enabled=source.enabled,
            status="invalid",
            notes=["expected exlora_nvfp4_residual.json"],
            details={"size_bytes": _dir_size(path), "size": _fmt_bytes(_dir_size(path))},
        )

    manifest = json.loads(manifest_path.read_text())
    tensors = manifest.get("tensors") or {}
    modules = Counter()
    shapes = Counter()
    total_bpe: list[float] = []
    rel_errors: list[float] = []
    residual_blocks = 0
    for tensor_name, meta in tensors.items():
        module = _module_of(tensor_name)
        if module is not None:
            modules[module] += 1
        shape = meta.get("shape")
        if shape:
            shapes["x".join(str(v) for v in shape)] += 1
        if "total_bpe" in meta:
            total_bpe.append(float(meta["total_bpe"]))
        if "rel_error" in meta:
            rel_errors.append(float(meta["rel_error"]))
        residual_blocks += int(meta.get("residual_blocks") or 0)

    return SourceScanReport(
        name=label,
        path=source.path,
        resolved_path=str(path),
        kind=source.kind or "nvfp4_residual",
        role="delta",
        representation="dense_nvfp4_residual",
        weight=source.weight,
        enabled=source.enabled,
        details={
            "artifact_type": manifest.get("artifact_type"),
            "base_model": manifest.get("base_model"),
            "finetune_model": manifest.get("finetune_model"),
            "target_modules": manifest.get("target_modules"),
            "tensor_count": len(tensors),
            "size_bytes": _dir_size(path),
            "size": _fmt_bytes(_dir_size(path)),
            "module_counts": _counter_dict(modules),
            "shape_counts": _counter_dict(shapes),
            "total_bpe": _stats(total_bpe),
            "rel_error": _stats(rel_errors),
            "residual_blocks": residual_blocks,
            "selection_policy": manifest.get("selection_policy"),
            "floor_correction_policy": manifest.get("floor_correction_policy"),
        },
    )


def _source_role(source: LoraMergeSourceConfig) -> str:
    roles: list[str] = []
    if source.is_embedding:
        roles.append("embedding")
    if source.is_lm_head:
        roles.append("lm_head")
    if not roles:
        roles.append("delta")
    return "+".join(roles)


def scan_source(
    source: LoraMergeSourceConfig,
    *,
    storage_dir: str | Path,
    target_modules: list[str],
) -> SourceScanReport:
    resolved = _resolve_path(source.path, storage_dir)
    role = _source_role(source)
    if source.is_io_source:
        return scan_checkpoint(
            source.path,
            storage_dir=storage_dir,
            role=role,
            target_modules=target_modules,
            name=source.name,
            enabled=source.enabled,
        )

    kind = (source.kind or "").lower()
    if kind == "peft" or (
        source.kind is None
        and (resolved / "adapter_config.json").exists()
        and (resolved / "adapter_model.safetensors").exists()
    ):
        return scan_peft_adapter(resolved, source)
    if kind in {"dense", "nvfp4", "nvfp4_residual", "nvfp4_residual_delta_map"} or (
        source.kind is None and (resolved / "exlora_nvfp4_residual.json").exists()
    ):
        return scan_nvfp4_residual(resolved, source)

    return SourceScanReport(
        name=source.name or Path(source.path).name or source.path,
        path=source.path,
        resolved_path=str(resolved),
        kind=source.kind,
        role=role,
        representation="unknown",
        weight=source.weight,
        enabled=source.enabled,
        status="unknown",
        notes=["source kind could not be inspected"],
    )


def scan_config(config: LoraMergeConfig, *, config_path: str | Path | None = None) -> LoraMergeScanReport:
    """Build a report for a loaded ``LoraMergeConfig``."""
    base = scan_checkpoint(
        config.base_model,
        storage_dir=config.storage_dir,
        role="base",
        target_modules=config.target_modules,
        name="base",
        enabled=True,
    )
    return LoraMergeScanReport(
        config_path=str(config_path) if config_path is not None else None,
        base_model=config.base_model,
        resolved_base_model=str(_resolve_path(config.base_model, config.storage_dir)),
        output_dir=config.output_dir,
        merge_rule=config.merge_rule,
        scaffold_format=config.scaffold_format,
        target_modules=list(config.target_modules),
        embedding_model=config.embedding_model,
        resolved_embedding_model=str(_resolve_path(config.embedding_model, config.storage_dir)),
        lm_head_model=config.lm_head_model,
        resolved_lm_head_model=str(_resolve_path(config.lm_head_model, config.storage_dir)),
        base_checkpoint=base,
        sources=[
            scan_source(
                source,
                storage_dir=config.storage_dir,
                target_modules=config.target_modules,
            )
            for source in config.sources
        ],
    )


# Metrics reported per (source, level). ``norm`` is the quadrature sum of
# target-module Frobenius norms; ``absmax`` the largest |delta| element; ``std``
# the population std over all target-module elements at that layer.
_LEVEL_METRICS = ("norm", "absmax", "std")


@dataclass
class DeltaLevelStats:
    """Per-source, per-level delta statistics, laid out for a layers-as-columns
    table. ``metrics[name][source][level]`` holds the value; ``overall`` is the
    same statistic aggregated across every layer. Non-layer tensors
    (embed/lm_head) are excluded -- this is a body-delta magnitude map.
    """

    levels: list[int]
    sources: list[str]
    metrics: dict[str, dict[str, dict[int, float]]]
    overall: dict[str, dict[str, float]]
    status: dict[str, str]
    enabled: dict[str, bool]


def _finalize_level(sumsq: float, total: float, count: int, absmax: float) -> dict[str, float]:
    if count == 0:
        return {"norm": 0.0, "absmax": 0.0, "std": 0.0}
    mean = total / count
    var = max(sumsq / count - mean * mean, 0.0)
    return {"norm": math.sqrt(sumsq), "absmax": absmax, "std": math.sqrt(var)}


def compute_delta_level_stats(
    config: LoraMergeConfig,
    *,
    device: str = "cpu",
    include_disabled: bool = True,
) -> DeltaLevelStats:
    """Materialize each delta source one tensor at a time and tabulate norm /
    abs-max / std per decoder layer. Tensors are freed immediately, so even a
    ~16 GB dense NVFP4 map never lands in memory whole."""
    import torch  # local import keeps the index-only scan torch-free
    from .sources import load_delta_source

    metrics: dict[str, dict[str, dict[int, float]]] = {m: {} for m in _LEVEL_METRICS}
    overall: dict[str, dict[str, float]] = {}
    status: dict[str, str] = {}
    enabled: dict[str, bool] = {}
    all_levels: set[int] = set()

    for source in config.sources:
        if not source.is_delta_source:
            continue
        if not include_disabled and not source.enabled:
            continue
        label = source.name or Path(source.path).name or source.path
        enabled[label] = source.enabled
        for m in _LEVEL_METRICS:
            metrics[m][label] = {}
        overall[label] = {m: 0.0 for m in _LEVEL_METRICS}
        resolved = _resolve_path(source.path, config.storage_dir)
        if not resolved.exists():
            status[label] = "missing"
            continue
        try:
            ds = load_delta_source(resolved, kind=source.kind, name=source.name)
        except Exception as exc:  # noqa: BLE001 - report, don't abort the whole table
            status[label] = f"error: {type(exc).__name__}"
            continue

        targets = source.target_modules if source.target_modules is not None else config.target_modules
        target_set = set(targets) if targets else None

        # Per level: running (sum of squares, sum, element count, abs max).
        sumsq: dict[int, float] = defaultdict(float)
        total: dict[int, float] = defaultdict(float)
        count: dict[int, int] = defaultdict(int)
        absmax: dict[int, float] = defaultdict(float)
        all_sumsq = all_total = all_absmax = 0.0
        all_count = 0
        for tensor_name in sorted(ds.tensor_names()):
            module = _module_of(tensor_name)
            if target_set is not None and module not in target_set:
                continue
            layer = _layer_index(tensor_name)
            if layer is None:
                continue
            try:
                delta = ds.delta(tensor_name, device=device, dtype=torch.float32)
            except Exception:  # noqa: BLE001
                continue
            if delta is None:
                continue
            t_sumsq = float(delta.pow(2).sum().item())
            t_sum = float(delta.sum().item())
            t_count = int(delta.numel())
            t_absmax = float(delta.abs().max().item()) if t_count else 0.0
            del delta
            sumsq[layer] += t_sumsq
            total[layer] += t_sum
            count[layer] += t_count
            absmax[layer] = max(absmax[layer], t_absmax)
            all_sumsq += t_sumsq
            all_total += t_sum
            all_count += t_count
            all_absmax = max(all_absmax, t_absmax)
            all_levels.add(layer)

        for layer in count:
            stats = _finalize_level(sumsq[layer], total[layer], count[layer], absmax[layer])
            for m in _LEVEL_METRICS:
                metrics[m][label][layer] = stats[m]
        overall[label] = _finalize_level(all_sumsq, all_total, all_count, all_absmax)
        status[label] = "ok"

    sources = list(enabled)
    return DeltaLevelStats(
        levels=sorted(all_levels),
        sources=sources,
        metrics=metrics,
        overall=overall,
        status=status,
        enabled=enabled,
    )


_METRIC_BLURB = {
    "norm": "Frobenius norm `sqrt(sum_m ||delta_m||_F^2)` over target modules per layer.",
    "absmax": "Largest single |delta| element over target modules per layer.",
    "std": "Population std of all target-module delta elements per layer.",
}


def _render_metric_table(stats: DeltaLevelStats, metric: str) -> list[str]:
    by_source = stats.metrics[metric]
    title = {"norm": "Frobenius Norm", "absmax": "Abs Max", "std": "Std"}[metric]
    lines = [f"### {title}", "", _METRIC_BLURB[metric], ""]

    level_cols = " | ".join(f"L{lvl}" for lvl in stats.levels)
    lines.append(f"| model | {level_cols} | overall |")
    lines.append("|---|" + "|".join(["---:"] * (len(stats.levels) + 1)) + "|")
    for src in stats.sources:
        label = src if stats.enabled.get(src, True) else f"{src} (off)"
        cells = []
        for lvl in stats.levels:
            value = by_source.get(src, {}).get(lvl)
            cells.append(f"{value:.4g}" if value is not None else "·")
        ov = stats.overall.get(src, {}).get(metric, 0.0)
        lines.append(f"| {label} | " + " | ".join(cells) + f" | **{ov:.4g}** |")
    lines.append("")
    return lines


def render_level_stats_markdown(stats: DeltaLevelStats) -> str:
    if not stats.sources:
        return "## Delta Level Stats\n\n_No delta sources to scan._\n"

    lines = [
        "## Delta Level Stats",
        "",
        "Per decoder layer (columns) for each source (rows). `·` = source does "
        "not touch that layer; `overall` aggregates across all layers.",
        "",
    ]
    for metric in _LEVEL_METRICS:
        lines.extend(_render_metric_table(stats, metric))

    bad = {name: st for name, st in stats.status.items() if st != "ok"}
    if bad:
        lines.extend(["**Status**", ""])
        for name, st in bad.items():
            lines.append(f"- `{name}`: {st}")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


@dataclass
class ModelMatrix:
    """Model-model matrix over a few representative layers.

    Diagonal ``isotropy[m]`` is the stable-rank fraction of a source's delta
    (``||D||_F^2 / sigma_1^2 / min(m, n)``; 1 = isotropic, ->0 = one direction
    dominates). Off-diagonal ``coverage[(a, b)]`` is the mean squared cosine of
    the principal angles between the top-``rank`` left-singular subspaces of the
    two sources' deltas (1 = shared subspace / one rotates into the other, 0 =
    orthogonal). Both are averaged over the representative (layer, module) cells
    where the source(s) carry a delta.
    """

    labels: list[str]
    rep_layers: list[int]
    rank: int
    isotropy: dict[str, float | None]
    coverage: dict[tuple[str, str], float | None]
    status: dict[str, str]
    enabled: dict[str, bool]


def _auto_rep_layers(loaded: list[tuple], target_lookup) -> list[int]:
    """Five evenly spaced decoder layers spanning the candidates' depth."""
    max_layer = 0
    for label, ds, _tset in loaded:
        for layer, _module in target_lookup[label]:
            max_layer = max(max_layer, layer)
    if max_layer == 0:
        return [0]
    fracs = (0.0, 0.25, 0.5, 0.75, 1.0)
    return sorted({int(round(max_layer * f)) for f in fracs})


def compute_model_matrix(
    config: LoraMergeConfig,
    *,
    device: str = "cuda",
    rep_layers: list[int] | None = None,
    rank: int = 32,
    include_disabled: bool = True,
) -> ModelMatrix:
    """Build the isotropy/coverage model-model matrix on ``device`` (GPU by
    default). Deltas are materialized one (layer, module) cell at a time and the
    top-``rank`` left-singular subspace is taken with ``torch.svd_lowrank``."""
    import torch
    from .sources import load_delta_source

    loaded: list[tuple[str, Any, set[str] | None]] = []
    status: dict[str, str] = {}
    enabled: dict[str, bool] = {}
    for source in config.sources:
        if not source.is_delta_source:
            continue
        if not include_disabled and not source.enabled:
            continue
        label = source.name or Path(source.path).name or source.path
        enabled[label] = source.enabled
        resolved = _resolve_path(source.path, config.storage_dir)
        if not resolved.exists():
            status[label] = "missing"
            continue
        try:
            ds = load_delta_source(resolved, kind=source.kind, name=source.name)
        except Exception as exc:  # noqa: BLE001
            status[label] = f"error: {type(exc).__name__}"
            continue
        targets = source.target_modules if source.target_modules is not None else config.target_modules
        loaded.append((label, ds, set(targets) if targets else None))
        status[label] = "ok"

    labels = [label for label, _ds, _t in loaded]

    # (label) -> {(layer, module): tensor_name} for the target modules only.
    target_lookup: dict[str, dict[tuple[int, str], str]] = {}
    for label, ds, tset in loaded:
        cells: dict[tuple[int, str], str] = {}
        for tensor_name in ds.tensor_names():
            module = _module_of(tensor_name)
            if tset is not None and module not in tset:
                continue
            layer = _layer_index(tensor_name)
            if layer is None or module is None:
                continue
            cells[(layer, module)] = tensor_name
        target_lookup[label] = cells

    if rep_layers is None:
        rep_layers = _auto_rep_layers(loaded, target_lookup)
    modules = list(config.target_modules)

    iso_sum: dict[str, float] = defaultdict(float)
    iso_cnt: dict[str, int] = defaultdict(int)
    cov_sum: dict[tuple[str, str], float] = defaultdict(float)
    cov_cnt: dict[tuple[str, str], int] = defaultdict(int)

    for layer in rep_layers:
        for module in modules:
            subspaces: dict[str, "torch.Tensor"] = {}
            for label, ds, _tset in loaded:
                tensor_name = target_lookup[label].get((layer, module))
                if tensor_name is None:
                    continue
                try:
                    delta = ds.delta(tensor_name, device=device, dtype=torch.float32)
                except Exception:  # noqa: BLE001
                    continue
                if delta is None or delta.ndim != 2:
                    if delta is not None:
                        del delta
                    continue
                m, n = delta.shape
                k = min(rank, m, n)
                q = min(m, n, k + 8)
                try:
                    U, S, _V = torch.svd_lowrank(delta, q=q)
                except Exception:  # noqa: BLE001
                    del delta
                    continue
                fro2 = float(delta.pow(2).sum().item())
                sigma1 = float(S[0].item())
                if sigma1 > 0:
                    iso_sum[label] += (fro2 / (sigma1 * sigma1)) / min(m, n)
                    iso_cnt[label] += 1
                subspaces[label] = U[:, :k].contiguous()
                del delta, S, _V

            present = list(subspaces)
            for i in range(len(present)):
                for j in range(i + 1, len(present)):
                    a, b = present[i], present[j]
                    ua, ub = subspaces[a], subspaces[b]
                    kk = min(ua.shape[1], ub.shape[1])
                    overlap = float((ua[:, :kk].transpose(0, 1) @ ub[:, :kk]).pow(2).sum().item()) / kk
                    key = (a, b) if a <= b else (b, a)
                    cov_sum[key] += overlap
                    cov_cnt[key] += 1
            subspaces.clear()
            if device.startswith("cuda"):
                torch.cuda.empty_cache()

    isotropy = {label: (iso_sum[label] / iso_cnt[label] if iso_cnt[label] else None) for label in labels}
    coverage: dict[tuple[str, str], float | None] = {}
    for a in labels:
        for b in labels:
            if a == b:
                continue
            key = (a, b) if a <= b else (b, a)
            coverage[(a, b)] = (cov_sum[key] / cov_cnt[key]) if cov_cnt.get(key) else None

    return ModelMatrix(
        labels=labels,
        rep_layers=list(rep_layers),
        rank=rank,
        isotropy=isotropy,
        coverage=coverage,
        status=status,
        enabled=enabled,
    )


def render_model_matrix_markdown(matrix: ModelMatrix) -> str:
    if not matrix.labels:
        return "## Model-Model Matrix\n\n_No delta sources to scan._\n"

    def _short(name: str) -> str:
        return name if matrix.enabled.get(name, True) else f"{name}*"

    layers = ", ".join(f"L{lvl}" for lvl in matrix.rep_layers)
    header = "| model | " + " | ".join(_short(c) for c in matrix.labels) + " |"
    sep = "|---|" + "|".join(["---:"] * len(matrix.labels)) + "|"
    lines = [
        "## Model-Model Matrix",
        "",
        f"Representative layers: {layers}. Subspace rank: {matrix.rank}. "
        "`*` = source disabled in config.",
        "",
        "**Diagonal** = isotropy (stable-rank fraction; 1 = isotropic, low = one "
        "direction dominates). **Off-diagonal** = coverage/rotation = mean squared "
        "cosine of principal angles between top-r left-singular subspaces "
        "(1 = shared subspace, 0 = orthogonal). `·` = no shared (layer, module) cell.",
        "",
        header,
        sep,
    ]
    for row in matrix.labels:
        cells = []
        for col in matrix.labels:
            if row == col:
                value = matrix.isotropy.get(row)
                cells.append(f"**{value:.3f}**" if value is not None else "·")
            else:
                value = matrix.coverage.get((row, col))
                cells.append(f"{value:.3f}" if value is not None else "·")
        lines.append(f"| {_short(row)} | " + " | ".join(cells) + " |")

    bad = {name: st for name, st in matrix.status.items() if st != "ok"}
    if bad:
        lines.extend(["", "**Status**", ""])
        for name, st in bad.items():
            lines.append(f"- `{name}`: {st}")
    lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    return value


def report_to_json(report: LoraMergeScanReport) -> str:
    return json.dumps(asdict(report), indent=2, default=_json_default) + "\n"


def _summarize_detail(value: Any) -> Any:
    if isinstance(value, list) and len(value) > 16:
        head = ", ".join(repr(item) for item in value[:8])
        return f"[{head}, ...] ({len(value)} items)"
    if isinstance(value, dict) and len(value) > 20:
        items = list(sorted(value.items(), key=lambda item: str(item[0])))
        head = {str(k): v for k, v in items[:12]}
        return f"{json.dumps(head, sort_keys=True)} ... ({len(value)} keys)"
    return value


def _detail_line(key: str, value: Any) -> str | None:
    if value is None or value == {} or value == []:
        return None
    value = _summarize_detail(value)
    if isinstance(value, dict):
        value = json.dumps(value, sort_keys=True)
    return f"- `{key}`: {value}"


def _render_source(report: SourceScanReport) -> list[str]:
    lines = [
        f"### {report.name}",
        "",
        f"- `path`: {report.path}",
        f"- `resolved_path`: {report.resolved_path}",
        f"- `role`: {report.role}",
        f"- `kind`: {report.kind}",
        f"- `representation`: {report.representation}",
        f"- `weight`: {report.weight:g}",
        f"- `enabled`: {str(report.enabled).lower()}",
        f"- `status`: {report.status}",
    ]
    for note in report.notes:
        lines.append(f"- note: {note}")
    detail_lines = [_detail_line(k, v) for k, v in report.details.items()]
    detail_lines = [line for line in detail_lines if line is not None]
    if detail_lines:
        lines.extend(["", "**Details**", ""])
        lines.extend(detail_lines)
    lines.append("")
    return lines


def report_to_markdown(report: LoraMergeScanReport) -> str:
    rows = [
        "| source | enabled | role | representation | weight | status |",
        "|---|---:|---|---|---:|---|",
    ]
    for source in report.sources:
        rows.append(
            f"| {source.name} | {str(source.enabled).lower()} | {source.role} | {source.representation} | "
            f"{source.weight:g} | {source.status} |"
        )

    lines = [
        "# LoRA Merge Scan Report",
        "",
        f"- `config`: {report.config_path}",
        f"- `base_model`: {report.base_model}",
        f"- `resolved_base_model`: {report.resolved_base_model}",
        f"- `output_dir`: {report.output_dir}",
        f"- `merge_rule`: {report.merge_rule}",
        f"- `scaffold_format`: {report.scaffold_format}",
        f"- `target_modules`: {', '.join(report.target_modules)}",
        "",
        "## IO Routing",
        "",
        f"- `embed_tokens`: {report.embedding_model} -> {report.resolved_embedding_model}",
        f"- `lm_head`: {report.lm_head_model} -> {report.resolved_lm_head_model}",
        "",
        "## Source Summary",
        "",
        *rows,
        "",
        "## Base Checkpoint",
        "",
        *_render_source(report.base_checkpoint),
        "## Sources",
        "",
    ]
    for source in report.sources:
        lines.extend(_render_source(source))
    return "\n".join(lines).rstrip() + "\n"


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.option(
    "--config",
    "config_file",
    required=True,
    type=click.Path(exists=True, path_type=Path),
    help="LoRA-merge YAML config to inspect.",
)
@click.option(
    "--format",
    "output_format",
    type=click.Choice(["markdown", "json"], case_sensitive=False),
    default="markdown",
    show_default=True,
    help="Report format.",
)
@click.option(
    "--output",
    "--out",
    "output_path",
    type=click.Path(path_type=Path),
    default=None,
    help="Write report to this path instead of stdout.",
)
@click.option(
    "--norms",
    "with_norms",
    is_flag=True,
    default=False,
    help="Materialize deltas and append per-source, per-level norm/abs-max/std tables.",
)
@click.option(
    "--device",
    "device",
    default=None,
    help="Device for delta-norm materialization (default: config device, else cpu).",
)
@click.option(
    "--enabled-only",
    "enabled_only",
    is_flag=True,
    default=False,
    help="With --norms/--matrix, only include enabled sources (default: all candidates).",
)
@click.option(
    "--matrix",
    "with_matrix",
    is_flag=True,
    default=False,
    help="Append the model-model isotropy/coverage matrix (GPU SVD over rep layers).",
)
@click.option(
    "--rep-layers",
    "rep_layers_arg",
    default=None,
    help="Comma-separated decoder layers for --matrix (default: 5 evenly spaced).",
)
@click.option(
    "--rank",
    "matrix_rank",
    type=int,
    default=32,
    show_default=True,
    help="Left-singular subspace rank for --matrix coverage.",
)
def cli(
    config_file: Path,
    output_format: str,
    output_path: Path | None,
    with_norms: bool,
    device: str | None,
    enabled_only: bool,
    with_matrix: bool,
    rep_layers_arg: str | None,
    matrix_rank: int,
) -> None:
    """Scan a loramerge config and report source provenance/representations."""
    config = LoraMergeConfig.from_yaml(config_file)
    report = scan_config(config, config_path=config_file)
    resolved_device = device or config.device or "cpu"

    level_stats: DeltaLevelStats | None = None
    if with_norms:
        level_stats = compute_delta_level_stats(
            config,
            device=resolved_device,
            include_disabled=not enabled_only,
        )

    model_matrix: ModelMatrix | None = None
    if with_matrix:
        rep_layers = (
            [int(x) for x in rep_layers_arg.split(",") if x.strip() != ""]
            if rep_layers_arg
            else None
        )
        model_matrix = compute_model_matrix(
            config,
            device=resolved_device,
            rep_layers=rep_layers,
            rank=matrix_rank,
            include_disabled=not enabled_only,
        )

    if output_format.lower() == "json":
        rendered = report_to_json(report)
        if level_stats is not None or model_matrix is not None:
            payload = json.loads(rendered)
            if level_stats is not None:
                payload["delta_level_stats"] = asdict(level_stats)
            if model_matrix is not None:
                matrix_dict = asdict(model_matrix)
                matrix_dict["coverage"] = {
                    f"{a}||{b}": v for (a, b), v in model_matrix.coverage.items()
                }
                payload["model_matrix"] = matrix_dict
            rendered = json.dumps(payload, indent=2, default=_json_default) + "\n"
    else:
        rendered = report_to_markdown(report)
        if level_stats is not None:
            rendered = rendered.rstrip() + "\n\n" + render_level_stats_markdown(level_stats)
        if model_matrix is not None:
            rendered = rendered.rstrip() + "\n\n" + render_model_matrix_markdown(model_matrix)

    if output_path is None:
        click.echo(rendered, nl=False)
        return
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(rendered)
    click.echo(f"Wrote {output_path}")


if __name__ == "__main__":
    cli()
