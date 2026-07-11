# shard/loraplay/__main__.py
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

"""loraplay: score an extracted LoRA against base+finetune in activation space.

This is a separate evaluation tool from the ``shard.exlora`` extraction path.
It never participates in extraction; it runs *after*, against a base model and
an emitted adapter.
"""

import asyncio
import logging
from pathlib import Path
from typing import List, Optional

import click

from .activations import harvest_model, save_grams, load_grams
from .adapter import load_adapter
from .play import score_adapter, write_report
from .player import FinetuneDeltaProvider, play_adapter

logger = logging.getLogger(__name__)

DEFAULT_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def setup_logging(level: str = "INFO", log_file: Optional[Path] = None):
    handlers: List[logging.Handler] = [logging.StreamHandler()]
    if log_file:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file, mode="a", encoding="utf-8"))
    logging.basicConfig(
        level=getattr(logging, str(level).upper(), logging.INFO),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=handlers, force=True,
    )


@click.group()
def cli():
    """Score extracted LoRA adapters against their base+finetune."""


@cli.command("score")
@click.option("--base", "base_model", required=True, help="Base model id (under storage-dir)")
@click.option("--finetune", "finetune_model", required=True, help="Finetune model id (under storage-dir)")
@click.option("--adapter", required=True, type=click.Path(exists=True, path_type=Path), help="Adapter dir")
@click.option("--storage-dir", default="models", type=click.Path(path_type=Path))
@click.option("--grams", default=None, type=click.Path(path_type=Path),
              help="Harvested input-Gram safetensors (enables activation-weighted scoring)")
@click.option("--device", default="cpu")
@click.option("--damping", default=1e-5, type=float)
@click.option("--report", default=None, type=click.Path(path_type=Path),
              help="Write a markdown report here (default: <adapter>/SCORE_REPORT.md)")
@click.option("--log-level", default="INFO")
def score_command(base_model, finetune_model, adapter, storage_dir, grams, device, damping, report, log_level):
    """Score an adapter; with --grams the score is the activation (output) error."""
    setup_logging(level=log_level)
    grams_dict = load_grams(grams) if grams else None
    report_path = report or (adapter / "SCORE_REPORT.md")
    rep = asyncio.run(score_adapter(
        base_model=base_model, finetune_model=finetune_model, adapter_dir=adapter,
        storage_dir=storage_dir, grams=grams_dict, device=device, damping=damping,
    ))
    write_report(rep, report_path, base_model, finetune_model, adapter, grams_used=bool(grams_dict))
    s = rep.summary()
    logger.info(f"Scored {s['layers']} layers, {s['total_mb']:.1f} MB; "
                f"behavior error mean {s['behavior_error'].get('mean')}")
    click.echo(f"Report written to {report_path}")


@cli.command("play")
@click.option("--base", "base_model", required=True, help="Base model id/path to LOAD for activations")
@click.option("--finetune-dir", required=True, type=click.Path(exists=True, path_type=Path),
              help="Finetune model dir (W_ft for the true delta)")
@click.option("--adapter", required=True, type=click.Path(exists=True, path_type=Path))
@click.option("--calib", required=True, type=click.Path(exists=True, path_type=Path),
              help="Calibration prompts file (one per line)")
@click.option("--max-tokens", default=2048, type=int)
@click.option("--device", default="cuda")
@click.option("--report", default=None, type=click.Path(path_type=Path))
@click.option("--log-level", default="INFO")
def play_command(base_model, finetune_dir, adapter, calib, max_tokens, device, report, log_level):
    """Play an adapter against the base model in ONE pass and score activation error.

    Streams W_ft per layer; never forms H or writes activations.
    """
    setup_logging(level=log_level)
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(base_model)
    model = AutoModelForCausalLM.from_pretrained(base_model, torch_dtype=torch.bfloat16, device_map=device)
    loaded = load_adapter(adapter)
    provider = FinetuneDeltaProvider(finetune_dir)
    prompts = [l for l in Path(calib).read_text().splitlines() if l.strip()]

    def batches():
        for p in prompts:
            ids = tok(p, return_tensors="pt", truncation=True, max_length=max_tokens)
            yield {"input_ids": ids["input_ids"].to(device)}

    rep = play_adapter(model, batches(), loaded, provider, device=device)
    report_path = report or (adapter / "PLAY_REPORT.md")
    write_report(rep, report_path, base_model, str(finetune_dir), adapter, grams_used=True)
    s = rep.summary()
    logger.info(f"Played {s['layers']} layers; activation error mean {s['act_rel_error'].get('mean')}")
    click.echo(f"Report written to {report_path}")


@cli.command("harvest")
@click.option("--base", "base_model", required=True, help="HF model id or local path to load")
@click.option("--calib", required=True, type=click.Path(exists=True, path_type=Path),
              help="Text file of calibration prompts (one per line)")
@click.option("--out", required=True, type=click.Path(path_type=Path), help="Output Gram safetensors path")
@click.option("--max-tokens", default=2048, type=int)
@click.option("--device", default="cuda")
@click.option("--log-level", default="INFO")
def harvest_command(base_model, calib, out, max_tokens, device, log_level):
    """Run one hooked forward pass over the base model to accumulate per-layer H."""
    setup_logging(level=log_level)
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    logger.info(f"Loading {base_model} for activation harvest")
    tok = AutoTokenizer.from_pretrained(base_model)
    model = AutoModelForCausalLM.from_pretrained(base_model, torch_dtype=torch.bfloat16, device_map=device)

    prompts = [l for l in Path(calib).read_text().splitlines() if l.strip()]

    def batches():
        for p in prompts:
            ids = tok(p, return_tensors="pt", truncation=True, max_length=max_tokens)
            yield {"input_ids": ids["input_ids"].to(device)}

    grams = harvest_model(model, batches(), DEFAULT_TARGETS, device=device)
    save_grams(grams, out)
    logger.info(f"Saved {len(grams)} layer Grams to {out}")


if __name__ == "__main__":
    cli()
