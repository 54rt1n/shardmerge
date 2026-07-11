# shard/exlora/__main__.py
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

import asyncio
import logging
from pathlib import Path
from typing import List, Optional

import click

from .config import ExtractConfig
from .extract import LoraExtractor

logger = logging.getLogger(__name__)

LOG_LEVELS = ["DEBUG", "INFO", "WARNING", "ERROR"]


def setup_logging(level: str = "INFO", log_file: Optional[Path] = None):
    """Configure logging. A FileHandler flushes per record, so ``--log-file``
    gives a tailable, line-by-line log of per-layer stats and errors even when
    stdout is block-buffered through a pipe."""
    handlers: List[logging.Handler] = [logging.StreamHandler()]
    if log_file:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file, mode="a", encoding="utf-8"))
    logging.basicConfig(
        level=getattr(logging, str(level).upper(), logging.INFO),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=handlers,
        force=True,
    )


@click.group()
def cli():
    """High-fidelity LoRA extraction from a finetune/base model pair."""


@cli.command("extract")
@click.argument("config_file", type=click.Path(exists=True, path_type=Path))
@click.option("--device", type=str, default=None, help="Compute device (cuda/cpu)")
@click.option("--target-error", type=float, default=None, help="Target relative Frobenius error per layer")
@click.option("--max-rank", type=int, default=None, help="Maximum LoRA rank per layer")
@click.option(
    "--log-file",
    type=click.Path(path_type=Path),
    default=None,
    help="Flushed per-layer log path (default: <output_dir>/extraction.log)",
)
@click.option(
    "--log-level",
    type=click.Choice(LOG_LEVELS, case_sensitive=False),
    default="INFO",
    help="Logging level for console and log file",
)
@click.option("--verbose", is_flag=True, help="Shorthand for --log-level DEBUG")
def extract_command(
    config_file: Path,
    device: Optional[str],
    target_error: Optional[float],
    max_rank: Optional[int],
    log_file: Optional[Path],
    log_level: str,
    verbose: bool,
):
    """Extract a PEFT LoRA adapter from a finetune model relative to its base.

    CONFIG_FILE is a YAML file with at least 'base_model', 'finetune_model'
    and 'output_dir'.
    """
    level = "DEBUG" if verbose else log_level
    # Console logging first so config-load errors are visible.
    setup_logging(level=level)
    try:
        config = ExtractConfig.from_yaml(config_file)
        config.update(device=device, target_error=target_error, max_rank=max_rank)

        # Default the log into the model's output dir, alongside the adapter
        # and EXTRACTION_REPORT.md, so the model-card inputs live together.
        resolved_log = log_file or (config.output_path / "extraction.log")
        setup_logging(level=level, log_file=resolved_log)
        logger.info(f"Logging to {resolved_log} at level {level}")
        logger.info(f"Loaded configuration: {config}")

        extractor = LoraExtractor(config=config)
        asyncio.run(extractor.extract(device=config.device))
    except Exception as e:
        logging.error(f"Error during extraction: {e}", exc_info=(level == "DEBUG"))
        raise click.Abort()


@cli.command("build-map")
@click.argument("config_file", type=click.Path(exists=True, path_type=Path))
@click.option("--device", type=str, default=None, help="Compute device (cuda/cpu)")
@click.option("--log-file", type=click.Path(path_type=Path), default=None,
              help="Flushed log path (default: <output_dir>/build_map.log)")
@click.option("--log-level", type=click.Choice(LOG_LEVELS, case_sensitive=False), default="INFO")
def build_map_command(config_file: Path, device: Optional[str], log_file: Optional[Path], log_level: str):
    """Build and cache the QAT projection (sensitivity) map.

    CONFIG_FILE uses the extraction schema: 'base_model' is the dense base and
    'finetune_model' is the QAT model. Caches per-tensor sensitivity (col/row
    norms + diagnostics) to 'output_dir' for reuse as a projection metric.
    """
    from .qat_map import build_qat_map

    setup_logging(level=log_level)
    try:
        config = ExtractConfig.from_yaml(config_file)
        config.update(device=device)
        resolved_log = log_file or (config.output_path / "build_map.log")
        setup_logging(level=log_level, log_file=resolved_log)
        logger.info(f"Building QAT map: {config.finetune_model} - {config.base_model}")
        qat_map = asyncio.run(build_qat_map(config, device=config.device))
        click.echo(f"QAT map ({len(qat_map.entries)} tensors) written to {config.output_path}")
    except Exception as e:
        logging.error(f"Error building QAT map: {e}", exc_info=(log_level.upper() == "DEBUG"))
        raise click.Abort()


if __name__ == "__main__":
    cli()
