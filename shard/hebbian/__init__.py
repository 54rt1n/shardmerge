# Copyright (C) 2026 Martin Bukowski
"""Hebbian isometry: activation-metric merge experiments.

The existing isometric operator (shard.loramerge.merge) extracts rotations in
the LM-head readout metric M = diag(1+g) lm_head^T lm_head diag(1+g) -- a static,
model-intrinsic geometry that knows what the head can SEE but not what the
running model actually DRIVES. This package replaces the metric with an
activation Gram H = E[x x^T] harvested over rollouts, so the Procrustes
problem is solved in the geometry of what fires: directions that co-activate
get weight, directions that exist in the weights but never light up do not.

Kept separate from shard.loramerge on purpose -- nothing here perturbs the
shipping isometry path.

Pipeline (each stage gates the next):
  1. harvest  -- per-module-suffix input Grams from base rollouts on MCQ
                 reasoning prompts (ARC-Easy / OpenBookQA; never ARC-C, which
                 stays untouched as the measurement).
  2. probe    -- does the H-metric concentrate / separate the source deltas
                 better than Frobenius or the readout metric? If not, stop.
  3. merge    -- (only if 2 says yes) hebbian_isometric operator: H-whitened
                 subspace selection, then two-sided M^1/2 . Delta . H^1/2
                 rotation extraction.
"""
