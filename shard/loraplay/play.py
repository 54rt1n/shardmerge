# shard/loraplay/play.py
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

"""Score an extracted adapter against base+finetune in activation space."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional
import logging

import torch

from ..exlora.index import FlatModelIndex
from ..download import DownloadManager
from .adapter import load_adapter
from .scoring import LayerScore, ScoreReport, metric_sqrt_from_gram, score_layer

logger = logging.getLogger(__name__)


async def score_adapter(
    base_model: str,
    finetune_model: str,
    adapter_dir: Path,
    storage_dir: Path = Path("models"),
    cache_dir: Path = Path("cache"),
    grams: Optional[Dict[str, torch.Tensor]] = None,
    device: str = "cpu",
    damping: float = 1e-5,
    index_manager=None,
) -> ScoreReport:
    """Score every adapter layer by raw and (if grams given) activation error."""
    index = index_manager or FlatModelIndex(
        download_manager=DownloadManager(storage_path=Path(storage_dir), clean_cache=False),
        cache_path=Path(cache_dir),
    )
    await index.add_model(base_model)
    await index.add_model(finetune_model)

    adapter = load_adapter(adapter_dir)
    scores = []
    for weight_name in list(adapter.layers()):
        base = (await index.get_tensor(base_model, weight_name, device=device).get()).float()
        ft = (await index.get_tensor(finetune_model, weight_name, device=device).get()).float()
        delta_true = ft - base
        del base, ft
        for cache in (getattr(index, "_tensor_downloads", None),):
            if cache is not None:
                cache.pop((base_model, weight_name), None)
                cache.pop((finetune_model, weight_name), None)

        delta_lora = adapter.delta(weight_name).to(device)
        metric_sqrt = None
        if grams and weight_name in grams:
            metric_sqrt = metric_sqrt_from_gram(grams[weight_name].to(device), damping=damping)

        score = score_layer(
            weight_name, delta_true, delta_lora, adapter.rank[weight_name], metric_sqrt
        )
        scores.append(score)
        logger.info(
            f"{weight_name}: rank {score.rank} raw_err {score.raw_rel_error:.4g}"
            + (f" act_err {score.act_rel_error:.4g}" if score.act_rel_error is not None else "")
        )
        del delta_true, delta_lora, metric_sqrt
        if device.startswith("cuda"):
            torch.cuda.empty_cache()

    return ScoreReport(scores)


def write_report(report: ScoreReport, out_path: Path, base_model: str, finetune_model: str,
                 adapter_dir: Path, grams_used: bool):
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    s = report.summary()
    lines = [
        "# LoRAplay Score Report",
        f"Base: {base_model}",
        f"Finetune: {finetune_model}",
        f"Adapter: {adapter_dir}",
        f"Activation metric: {'harvested H' if grams_used else 'none (raw Frobenius only)'}",
        f"Layers scored: {s['layers']}",
        f"Total adapter size: {s['total_mb']:.1f} MB",
        f"Raw rel error    (mean/median/worst): {_fmt(s['raw_rel_error'])}",
        f"Activation error (mean/median/worst): {_fmt(s['act_rel_error'])}",
        "",
        "| layer | rank | raw_err | act_err | MB |",
        "| --- | --- | --- | --- | --- |",
    ]
    for sc in report.layers:
        act = f"{sc.act_rel_error:.4g}" if sc.act_rel_error is not None else "-"
        lines.append(f"| {sc.layer_name} | {sc.rank} | {sc.raw_rel_error:.4g} | {act} | {sc.nbytes/1e6:.1f} |")
    out_path.write_text("\n".join(lines) + "\n")


def _fmt(agg: dict) -> str:
    if not agg:
        return "n/a"
    return f"{agg['mean']:.4g} / {agg['median']:.4g} / {agg['worst']:.4g}"
