# shard/loraplay/__init__.py
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

"""loraplay: activation-space scoring of extracted LoRA adapters.

A standalone evaluation tool beside :mod:`shard.exlora`. It scores how well an
emitted adapter reconstructs the true finetune in *output* (activation) space,
using per-layer input second moments harvested from the base model. It is not
part of the extraction process.
"""

from .scoring import LayerScore, ScoreReport, score_layer, metric_sqrt_from_gram
from .activations import GramAccumulator, harvest_model, save_grams, load_grams
from .adapter import LoadedAdapter, load_adapter, parse_lora_key
from .play import score_adapter, write_report
from .player import ActivationErrorAccumulator, AdapterPlayer, play_adapter

__all__ = [
    "LayerScore",
    "ScoreReport",
    "score_layer",
    "metric_sqrt_from_gram",
    "GramAccumulator",
    "harvest_model",
    "save_grams",
    "load_grams",
    "LoadedAdapter",
    "load_adapter",
    "parse_lora_key",
    "score_adapter",
    "write_report",
    "ActivationErrorAccumulator",
    "AdapterPlayer",
    "play_adapter",
]
