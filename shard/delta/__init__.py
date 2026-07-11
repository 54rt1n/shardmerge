# shard/delta/__init__.py
# Copyright (C) 2026 Martin Bukowski
#
# This software is free software: you can redistribute it and/or
# modify it under the terms of the GNU Lesser General Public License as
# published by the Free Software Foundation, either version 3 of the
# License, or (at your option) any later version.

"""Composable tensor-delta patching primitives.

This package is the shared model-delta layer used by higher-level extractors.
It deliberately does not know about PEFT, QAT, or any one codec.  It describes
how stored tensor corrections compose:

```
patched_weight = base_weight + dense_codec_delta + lora_delta + residual_delta
```

`shard.exlora` can build these pieces; `shard.delta` owns the neutral runtime
shape and typed protocols.
"""

from .codecs import DeltaCodec, IdentityDeltaCodec, RoundToDTypeCodec
from .manifest import PatchManifest, TensorPatchSpec, TermSpec
from .metrics import GramMetric, metric_norm, relative_metric_error
from .patch import PatchStack, apply_patch_stack
from .terms import DenseDelta, DeltaTerm, IndexedResidualDelta, LoraDelta

__all__ = [
    "DeltaCodec",
    "DenseDelta",
    "DeltaTerm",
    "GramMetric",
    "IdentityDeltaCodec",
    "IndexedResidualDelta",
    "LoraDelta",
    "PatchManifest",
    "PatchStack",
    "RoundToDTypeCodec",
    "TensorPatchSpec",
    "TermSpec",
    "apply_patch_stack",
    "metric_norm",
    "relative_metric_error",
]
