# shard/loramerge/__init__.py
# Copyright (C) 2026 Martin Bukowski
#
# This software is free software: you can redistribute it and/or
# modify it under the terms of the GNU Lesser General Public License as
# published by the Free Software Foundation, either version 3 of the
# License, or (at your option) any later version.

"""Full-frame merge helpers for LoRA and dense ExLoRA artifacts."""

from .config import (
    LoraMergeConfig,
    LoraMergeSourceConfig,
    is_embedding_tensor,
    is_lm_head_tensor,
)
from .merge import (
    WeightedDeltaSource,
    collect_source_deltas,
    fourier_slerp_merge,
    merge_full_frame_tensor,
    norm_aware_slerp,
    norm_slerp_merge,
    sign_agreement_vote,
    sum_weighted_deltas,
)
from .sources import (
    DeltaSource,
    DenseDeltaSource,
    NVFP4ResidualDeltaSource,
    PeftDeltaSource,
    load_delta_source,
)

__all__ = [
    "DeltaSource",
    "DenseDeltaSource",
    "LoraMergeConfig",
    "LoraMergeSourceConfig",
    "NVFP4ResidualDeltaSource",
    "PeftDeltaSource",
    "WeightedDeltaSource",
    "collect_source_deltas",
    "fourier_slerp_merge",
    "is_embedding_tensor",
    "is_lm_head_tensor",
    "load_delta_source",
    "merge_full_frame_tensor",
    "norm_aware_slerp",
    "norm_slerp_merge",
    "sign_agreement_vote",
    "sum_weighted_deltas",
]
