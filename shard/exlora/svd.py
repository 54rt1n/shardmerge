# shard/exlora/svd.py
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

"""Pure tensor math for low-rank LoRA factorization.

Everything here operates on plain tensors with no I/O, so it can be unit
tested directly against synthetic deltas of known rank.
"""

from dataclasses import dataclass
from typing import Optional
import logging
import torch

logger = logging.getLogger(__name__)


@dataclass
class LoraFactor:
    """A low-rank factorization of a single weight delta.

    The PEFT convention applies the update as ``scaling * (B @ A)`` with
    ``scaling = lora_alpha / r``. We bake the singular values evenly into
    both factors and set ``lora_alpha == r`` per layer (scaling == 1), so
    ``B @ A`` reconstructs the truncated delta exactly.

    Shapes (for a Linear weight of shape ``[out, in]``):
    - ``lora_A``: ``[r, in]``
    - ``lora_B``: ``[out, r]``
    """

    lora_A: torch.Tensor
    lora_B: torch.Tensor
    rank: int
    rel_error: float  # achieved relative Frobenius error of the truncation
    max_singular: float
    min_kept_singular: float


@dataclass(frozen=True)
class SpectrumDiagnostics:
    """Signal/noise diagnostics for a singular spectrum."""

    full_rank: int
    effective_rank: float
    retained_energy: float
    rel_error: float
    signal_rank: Optional[int]
    signal_reason: str
    noise_floor: float
    noise_floor_ratio: float
    knee_rank: Optional[int]
    knee_strength: float


def relative_errors(singular_values: torch.Tensor) -> torch.Tensor:
    """Relative Frobenius error remaining after keeping each rank ``r``.

    Returns a 1D tensor ``e`` of length ``k+1`` where ``e[r]`` is
    ``||Δ - Δ_r||_F / ||Δ||_F`` for a rank-``r`` truncation (``e[0]`` keeps
    nothing, ``e[k]`` keeps everything == 0). Uses the identity that the
    squared Frobenius error equals the sum of squared discarded singular
    values.
    """
    # Singular-value vectors are small; compute the selection on CPU so this
    # works regardless of which device the SVD ran on (avoids cpu/cuda mixing).
    sq = (singular_values.detach().to(device="cpu", dtype=torch.float64)) ** 2
    total = sq.sum()
    if total <= 0:
        # All-zero delta: every truncation is exact.
        return torch.zeros(len(singular_values) + 1, dtype=torch.float64)
    # residual[r] = sum of squared singular values strictly beyond rank r
    csum = torch.cumsum(sq, dim=0)
    residual = torch.empty(len(sq) + 1, dtype=torch.float64)
    residual[0] = total
    residual[1:] = total - csum
    residual.clamp_(min=0.0)
    return torch.sqrt(residual / total)


def retained_energy(singular_values: torch.Tensor, rank: int) -> float:
    """Return the fraction of squared singular energy kept by ``rank``."""
    sq = (singular_values.detach().to(device="cpu", dtype=torch.float64)) ** 2
    total = sq.sum()
    if len(sq) == 0 or total <= 0:
        return 1.0
    r = max(0, min(int(rank), len(sq)))
    if r == 0:
        return 0.0
    return float((sq[:r].sum() / total).item())


def effective_rank(singular_values: torch.Tensor) -> float:
    """Return participation-ratio effective rank over squared singular values."""
    sq = (singular_values.detach().to(device="cpu", dtype=torch.float64)) ** 2
    total = sq.sum()
    denom = sq.square().sum()
    if len(sq) == 0 or total <= 0 or denom <= 0:
        return 0.0
    return float((total.square() / denom).item())


def noise_aware_singular_values(
    singular_values: torch.Tensor,
    noise_singular_values: Optional[torch.Tensor] = None,
    rank: Optional[int] = None,
    mode: str = "ridge",
    noise_scale: float = 1.0,
    min_gain: float = 0.0,
) -> torch.Tensor:
    """Shrink observed singular values against an expected noise spectrum.

    This is the "signal inside the floor" primitive. The singular vectors can
    still carry useful structure even when some singular values sit inside a
    codec or merge-noise bulk, so this function supports smooth shrinkage
    instead of only hard thresholding.

    Modes:

    - ``none``: keep observed singular values, optionally zeroing past ``rank``.
    - ``hard``: keep values only when ``S > noise_scale * N``.
    - ``quadratic``: use ``sqrt(max(S^2 - (noise_scale*N)^2, 0))``.
    - ``ridge``: use ``S * S^2 / (S^2 + (noise_scale*N)^2)``.

    ``ridge`` is the default because it preserves a prior rank while attenuating
    under-floor directions. That is useful for merged LoRA recovery and for
    QAT-projected NVFP4 residuals where spectrum alone cannot prove every
    direction is above the codec floor.
    """
    if singular_values.ndim != 1:
        raise ValueError("singular_values must be a 1D tensor")
    if rank is not None and rank < 0:
        raise ValueError("rank must be non-negative")
    if noise_scale < 0:
        raise ValueError("noise_scale must be non-negative")
    if not 0.0 <= min_gain <= 1.0:
        raise ValueError("min_gain must be in [0, 1]")

    device = singular_values.device
    out_dtype = singular_values.dtype
    observed = singular_values.detach().to(device=device, dtype=torch.float64)
    k = len(observed)
    if k == 0:
        return singular_values.clone()

    keep = k if rank is None else min(int(rank), k)
    keep = max(0, keep)

    mode = mode.lower()
    if mode not in {"none", "hard", "quadratic", "ridge"}:
        raise ValueError("mode must be one of: none, hard, quadratic, ridge")

    if noise_singular_values is None or mode == "none":
        shrunk = observed.clone()
    else:
        noise = noise_singular_values.detach().to(device=device, dtype=torch.float64).flatten()
        if len(noise) == 0:
            noise = torch.zeros_like(observed)
        elif len(noise) < k:
            pad = noise[-1].expand(k - len(noise))
            noise = torch.cat([noise, pad])
        else:
            noise = noise[:k]
        noise = noise_scale * noise.clamp_min(0.0)

        if mode == "hard":
            shrunk = torch.where(observed > noise, observed, torch.zeros_like(observed))
        elif mode == "quadratic":
            shrunk = torch.sqrt((observed.square() - noise.square()).clamp_min(0.0))
        else:
            denom = observed.square() + noise.square()
            gain = torch.where(denom > 0, observed.square() / denom, torch.zeros_like(observed))
            if min_gain > 0:
                gain = gain.clamp_min(min_gain)
            shrunk = observed * gain

    if keep < k:
        shrunk = shrunk.clone()
        shrunk[keep:] = 0
    return shrunk.to(device=device, dtype=out_dtype)


def spectrum_diagnostics(
    singular_values: torch.Tensor,
    min_rank: int = 1,
    max_rank: Optional[int] = None,
    noise_floor_multiplier: float = 1.5,
    min_knee_drop: float = 1.25,
    tail_fraction: float = 0.25,
) -> SpectrumDiagnostics:
    """Estimate a useful signal rank before a flat numerical noise floor.

    This is intentionally conservative. It only reports a ``signal_rank`` when
    the tail median is clearly below the leading singular values or when a
    sufficiently strong adjacent singular-value drop is visible. This keeps the
    default target-error path unchanged while giving bf16-merged LoRA deltas a
    way to stop at the signal knee instead of reconstructing full-rank rounding
    noise.
    """
    values = singular_values.detach().to(device="cpu", dtype=torch.float64)
    k = len(values)
    if k == 0:
        return SpectrumDiagnostics(0, 0.0, 1.0, 0.0, 0, "empty", 0.0, 0.0, 0, 0.0)

    cap = k if max_rank is None else min(max_rank, k)
    cap = max(1, cap)
    floor_rank = max(1, min(min_rank, cap))
    values = values[:cap]

    if torch.all(values <= 0):
        return SpectrumDiagnostics(
            full_rank=k,
            effective_rank=0.0,
            retained_energy=1.0,
            rel_error=0.0,
            signal_rank=floor_rank,
            signal_reason="zero_spectrum",
            noise_floor=0.0,
            noise_floor_ratio=0.0,
            knee_rank=floor_rank,
            knee_strength=0.0,
        )

    tail_count = max(1, int(round(cap * tail_fraction)))
    tail = values[-tail_count:]
    noise_floor = float(torch.median(tail).item())
    top = float(values[0].item())
    noise_floor_ratio = noise_floor / max(top, 1e-300)

    noise_rank: Optional[int] = None
    if noise_floor > 0 and top / noise_floor >= noise_floor_multiplier:
        threshold = noise_floor * noise_floor_multiplier
        above = int((values > threshold).sum().item())
        if floor_rank <= above < cap:
            noise_rank = max(floor_rank, above)

    knee_rank: Optional[int] = None
    knee_strength = 0.0
    if cap > floor_rank:
        left = values[:-1]
        right = values[1:].clamp_min(1e-300)
        ratios = left / right
        start = max(0, floor_rank - 1)
        if start < len(ratios):
            local = ratios[start:]
            best = int(torch.argmax(local).item()) + start
            knee_strength = float(ratios[best].item())
            if knee_strength >= min_knee_drop:
                knee_rank = best + 1

    if noise_rank is not None:
        signal_rank = noise_rank
        reason = "noise_floor"
    elif knee_rank is not None:
        signal_rank = knee_rank
        reason = "spectral_knee"
    else:
        signal_rank = None
        reason = "no_signal_floor"

    if signal_rank is None:
        energy = retained_energy(values, cap)
        err = float(relative_errors(values)[cap].item())
    else:
        energy = retained_energy(values, signal_rank)
        err = float(relative_errors(values)[signal_rank].item())

    return SpectrumDiagnostics(
        full_rank=k,
        effective_rank=effective_rank(singular_values),
        retained_energy=energy,
        rel_error=err,
        signal_rank=signal_rank,
        signal_reason=reason,
        noise_floor=noise_floor,
        noise_floor_ratio=noise_floor_ratio,
        knee_rank=knee_rank,
        knee_strength=knee_strength,
    )


def select_signal_rank(
    singular_values: torch.Tensor,
    min_rank: int = 1,
    max_rank: Optional[int] = None,
    noise_floor_multiplier: float = 1.5,
    min_knee_drop: float = 1.25,
    tail_fraction: float = 0.25,
) -> tuple[int, float, SpectrumDiagnostics]:
    """Select a noise-aware signal rank from the spectrum.

    If no credible signal floor is detected, returns ``max_rank`` and marks the
    diagnostics with ``signal_reason == "no_signal_floor"``.
    """
    diag = spectrum_diagnostics(
        singular_values,
        min_rank=min_rank,
        max_rank=max_rank,
        noise_floor_multiplier=noise_floor_multiplier,
        min_knee_drop=min_knee_drop,
        tail_fraction=tail_fraction,
    )
    if diag.full_rank == 0:
        return 0, 0.0, diag
    cap = diag.full_rank if max_rank is None else min(max_rank, diag.full_rank)
    cap = max(1, cap)
    rank = diag.signal_rank if diag.signal_rank is not None else cap
    rank = max(1, min(rank, cap))
    err = float(relative_errors(singular_values)[rank].item())
    return rank, err, diag


def select_rank(
    singular_values: torch.Tensor,
    target_error: float,
    min_rank: int = 1,
    max_rank: Optional[int] = None,
) -> tuple[int, float]:
    """Choose the smallest rank meeting ``target_error``.

    Picks the smallest ``r`` in ``[min_rank, max_rank]`` whose truncated-SVD
    relative Frobenius error is ``<= target_error``. If none qualify within
    ``max_rank``, returns ``max_rank`` (best effort). The rank is always
    clamped to the number of available singular values.

    Returns ``(rank, achieved_rel_error)``.
    """
    k = len(singular_values)
    if k == 0:
        return 0, 0.0

    cap = k if max_rank is None else min(max_rank, k)
    cap = max(cap, 1)
    floor = max(1, min(min_rank, cap))

    errs = relative_errors(singular_values)

    # Smallest r in [floor, cap] with errs[r] <= target_error.
    for r in range(floor, cap + 1):
        if errs[r].item() <= target_error:
            return r, errs[r].item()

    # Nothing met the target within the cap; take the best we can.
    return cap, errs[cap].item()


def basis_svd(
    delta: torch.Tensor,
    compute_dtype: torch.dtype = torch.float32,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return the original-basis SVD ``(U, S, Vh)`` of a 2D delta.

    This is the shared artifact for both rank selection and defuzzing: we
    always need the full-precision SVD of the delta regardless of which
    projection path is taken. ``full_matrices=False`` gives ``U:[out,k]``,
    ``S:[k]``, ``Vh:[k,in]`` with ``k = min(out, in)``.
    """
    if delta.ndim != 2:
        raise ValueError(f"basis_svd requires a 2D delta, got shape {tuple(delta.shape)}")
    return torch.linalg.svd(delta.to(compute_dtype), full_matrices=False)


def factor_delta_to_lora(
    delta: torch.Tensor,
    target_error: float = 0.01,
    min_rank: int = 1,
    max_rank: Optional[int] = None,
    out_dtype: torch.dtype = torch.bfloat16,
    compute_dtype: torch.dtype = torch.float32,
) -> LoraFactor:
    """Factor a 2D weight delta into LoRA ``A``/``B`` via truncated SVD.

    The SVD is computed in ``compute_dtype`` (float32 by default) for
    fidelity regardless of the stored weight precision, then the factors are
    cast to ``out_dtype`` for storage.
    """
    if delta.ndim != 2:
        raise ValueError(f"LoRA factorization requires a 2D delta, got shape {tuple(delta.shape)}")

    U, S, Vh = basis_svd(delta, compute_dtype=compute_dtype)

    rank, rel_error = select_rank(S, target_error, min_rank=min_rank, max_rank=max_rank)
    rank = max(rank, 1)

    U_r = U[:, :rank]
    S_r = S[:rank]
    Vh_r = Vh[:rank, :]

    # Split singular values evenly into both factors so that B @ A == Δ_r.
    sqrt_s = torch.sqrt(S_r)
    lora_B = (U_r * sqrt_s.unsqueeze(0)).to(out_dtype).contiguous()        # [out, r]
    lora_A = (sqrt_s.unsqueeze(1) * Vh_r).to(out_dtype).contiguous()       # [r, in]

    return LoraFactor(
        lora_A=lora_A,
        lora_B=lora_B,
        rank=rank,
        rel_error=float(rel_error),
        max_singular=float(S[0].item()) if len(S) else 0.0,
        min_kept_singular=float(S_r[-1].item()) if rank else 0.0,
    )


def reconstruct_delta(factor: LoraFactor) -> torch.Tensor:
    """Reconstruct the dense delta ``B @ A`` from a factorization.

    With ``lora_alpha == r`` (scaling == 1) this is exactly the rank-``r``
    truncation of the original delta. Computed in float32 for stability.
    """
    return factor.lora_B.to(torch.float32) @ factor.lora_A.to(torch.float32)
