# shard/exlora/comb.py
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

"""Comb-corrector fitting and local activation gating.

A dense NVFP4 (or other low-bit) map leaves a residual ``E = D - codec(D)`` --
the "comb". In raw weight space ``E`` is full-rank, but in the activation
``H``-norm its behavioral footprint is low-rank, so a small LoRA correction
fit in the whitened ``H``-metric cancels most of the behaviorally-relevant
quantization error (see the Egan policy hybrid result: NVFP4-dense KL 0.232 ->
rank-64 policy comb-LoRA 0.0347).

Two policies are load-bearing and learned the hard way:

1. **Full-model rank is non-monotone.** Pushing the rank past the well-conditioned head
   (Egan: r256 regressed to baseline and *lost* top-1) overfits the
   under-determined tail of an ``H`` estimated from finite tokens. The local
   held-out activation metric is a decline/admissibility gate; final cap raises
   require held-out KL/top1/flux or another nonlinear model behavior metric.
2. **Whitening must be damped.** The ``H^{-1/2}`` de-whitening amplifies
   small-eigenvalue directions (up to ``1/sqrt(lambda)``); replace it with
   ``1/sqrt(lambda + damping)`` and clamp the inverse gain so the fit cannot
   chase noise directions the tokens barely probe.

The fitted artifact is a :class:`shard.delta.LoraDelta` runtime term; this
module owns only the tensor-local fitting/admissibility gate, not the runtime
composition or full-model behavior selection.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import torch

from ..delta.metrics import metric_norm
from .functions import randomized_svd, solve_right_basis_coefficients


@dataclass(frozen=True)
class CombCorrectorPolicy:
    """How to fit and locally gate a comb-corrector rank.

    ``candidate_ranks`` always implicitly includes ``0`` (decline / no
    corrector); the local gate only chooses a positive rank if it strictly
    beats the uncorrected held-out activation error. The returned positive rank
    is bounded by ``max_rank`` and should still be behavior-gated before
    raising the cap.
    """

    candidate_ranks: tuple[int, ...] = (0, 8, 16, 32, 64, 128)
    max_rank: int = 64
    relative_damping: float = 1e-3
    max_inverse_gain: Optional[float] = None
    oversampling: int = 16
    power_iterations: int = 2

    def effective_ranks(self) -> tuple[int, ...]:
        """Sorted unique candidate ranks within ``[0, max_rank]``, incl. 0."""
        ranks = {0}
        ranks.update(r for r in self.candidate_ranks if 0 < r <= self.max_rank)
        return tuple(sorted(ranks))


@dataclass
class CombCorrectorFit:
    """Result of tensor-local comb fitting.

    ``rank == 0`` means the corrector was declined (no positive rank improved
    the held-out error); ``lora_A``/``lora_B`` are then ``None``.
    """

    rank: int
    lora_A: Optional[torch.Tensor]          # [rank, in]
    lora_B: Optional[torch.Tensor]          # [out, rank]
    val_errors: dict[int, float]            # rank -> held-out relative H_val error
    reason: str
    fit_errors: dict[int, float] = field(default_factory=dict)

    def to_lora_delta(self):
        """Materialize as a ``shard.delta.LoraDelta`` term (None if declined)."""
        if self.rank == 0 or self.lora_A is None or self.lora_B is None:
            return None
        from ..delta.terms import LoraDelta

        return LoraDelta(lora_A=self.lora_A, lora_B=self.lora_B)


def damped_whitening(
    gram: torch.Tensor,
    relative_damping: float = 1e-3,
    max_inverse_gain: Optional[float] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``(H^{1/2}, H^{-1/2}_damped)`` for a PSD Gram ``H``.

    The inverse root uses ``1/sqrt(lambda + relative_damping * lambda_max)`` and
    optionally clamps the per-direction gain to ``max_inverse_gain`` so the
    de-whitening cannot blow up small-eigenvalue (noise) directions.
    """
    if gram.ndim != 2 or gram.shape[0] != gram.shape[1]:
        raise ValueError("gram must be a square matrix")
    H = gram.to(torch.float64)
    lam, V = torch.linalg.eigh(H)
    lam = lam.clamp_min(0.0)
    lam_max = lam.max().clamp_min(1e-300)
    damping = relative_damping * lam_max
    half = lam.sqrt()
    inv = 1.0 / (lam + damping).sqrt()
    if max_inverse_gain is not None:
        inv = inv.clamp_max(float(max_inverse_gain))
    h_half = (V * half) @ V.T
    h_inv_half = (V * inv) @ V.T
    return h_half.to(gram.dtype), h_inv_half.to(gram.dtype)


def fit_comb_corrector(
    residual: torch.Tensor,
    gram_fit: torch.Tensor,
    gram_val: torch.Tensor,
    policy: CombCorrectorPolicy = CombCorrectorPolicy(),
) -> CombCorrectorFit:
    """Fit an H-optimal comb corrector for ``residual`` and apply a local gate.

    ``residual`` is ``E = D - codec(D)`` with shape ``[out, in]``. ``gram_fit``
    and ``gram_val`` are ``[in, in]`` activation Grams from *disjoint* token
    splits; the basis is fit against ``gram_fit`` and candidate ranks are scored
    by relative error under ``gram_val``. This local metric can decline a layer
    when the comb is irrelevant, but it is not the final full-model rank
    selector. Cap raises require held-out KL/top1/flux behavior.
    """
    if residual.ndim != 2:
        raise ValueError("residual must be 2D [out, in]")
    E = residual.to(torch.float32)
    in_features = E.shape[1]
    if gram_fit.shape != (in_features, in_features) or gram_val.shape != (in_features, in_features):
        raise ValueError(
            f"grams must be [{in_features}, {in_features}] to match residual input dim"
        )

    ranks = policy.effective_ranks()
    positive = [r for r in ranks if r > 0]

    base_val = float(metric_norm(E, gram_val).clamp_min(1e-12))
    val_errors: dict[int, float] = {0: 1.0}
    fit_errors: dict[int, float] = {0: 1.0}
    base_fit = float(metric_norm(E, gram_fit).clamp_min(1e-12))

    best_rank, best_val_err = 0, 1.0
    best_A: Optional[torch.Tensor] = None
    best_B: Optional[torch.Tensor] = None

    if positive:
        h_half, h_inv_half = damped_whitening(
            gram_fit, policy.relative_damping, policy.max_inverse_gain
        )
        whitened = E @ h_half
        sketch_rank = min(max(positive), min(E.shape))
        _, _, Vh = randomized_svd(
            whitened,
            rank=sketch_rank,
            oversampling=policy.oversampling,
            power_iterations=policy.power_iterations,
            compute_dtype=torch.float32,
        )
        for r in positive:
            if r > sketch_rank:
                continue
            A = Vh[:r] @ h_inv_half                                   # [r, in]
            B = solve_right_basis_coefficients(E, A, metric=gram_fit)  # [out, r]
            residual_after = E - B @ A
            val_errors[r] = float(metric_norm(residual_after, gram_val) / base_val)
            fit_errors[r] = float(metric_norm(residual_after, gram_fit) / base_fit)
            if val_errors[r] < best_val_err:
                best_rank, best_val_err = r, val_errors[r]
                best_A, best_B = A, B

    reason = "declined" if best_rank == 0 else "local_activation_admissible"
    return CombCorrectorFit(
        rank=best_rank,
        lora_A=best_A,
        lora_B=best_B,
        val_errors=val_errors,
        reason=reason,
        fit_errors=fit_errors,
    )
