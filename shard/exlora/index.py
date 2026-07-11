# shard/exlora/index.py
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

"""Naming-agnostic model index for LoRA extraction.

The base :class:`shard.index.HFMultiModelIndex` hardcodes the Llama-style
``model.layers.N`` weight layout in ``_get_ordered_weights`` and raises a
"Weight ordering mismatch" for anything else (e.g. Gemma-3, whose decoder
weights live under ``model.language_model.layers.N`` alongside a vision tower
and non-``.weight`` params like ``layer_scalar``).

LoRA extraction factors each layer independently, so the *order* of weights
is irrelevant — only the set matters. This subclass therefore returns the raw
weight-map keys unchanged, which works for any architecture.
"""

from typing import List

from ..index import HFMultiModelIndex


class FlatModelIndex(HFMultiModelIndex):
    """HFMultiModelIndex that tolerates arbitrary weight-naming schemes."""

    def _get_ordered_weights(self, model_uri: str) -> List[str]:
        if model_uri not in self.model_indexes:
            raise KeyError(f"Model {model_uri} not found in index")
        return list(self.model_indexes[model_uri]["weight_map"].keys())
