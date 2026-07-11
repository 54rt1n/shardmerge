# shard/loramerge/merge.py
# Copyright (C) 2026 Martin Bukowski
#
# This software is free software: you can redistribute it and/or
# modify it under the terms of the GNU Lesser General Public License as
# published by the Free Software Foundation, either version 3 of the
# License, or (at your option) any later version.

"""Full-frame merge rules for adapter and dense delta sources."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Sequence

import torch

from ..tensor.functions import merge_tensors_fft2_slerp, slerp
from .sources import DeltaSource


MergeRule = Literal["sum", "sign_vote", "norm_slerp", "fourier_slerp", "fft_slerp", "isometric"]


@dataclass(frozen=True)
class WeightedDeltaSource:
    """A delta source plus its merge weight and an alpha rescale.

    ``weight`` is the merge-rule mixture weight (slerp t / vote strength).
    ``rescale`` multiplies the delta MAGNITUDE -- use it to normalize sources of
    very different strength (e.g. a high-alpha adapter) before norm-aware merges.
    """

    source: DeltaSource
    weight: float = 1.0
    rescale: float = 1.0
    is_norm: bool = False         # norm reference: output magnitude tracks this source's delta norm
    name: str | None = None
    post_merge: bool = False      # held out of the body fold and applied against the final body;
                                  # an additive overlay under isometric merging
    group: str | None = None      # body sources sharing a group merge into each other first
    pool: str | None = None       # opt-in isometric pool; distinct from legacy group semantics
    pool_rank: int | None = None  # optional rank budget for this isometric pool
    pool_energy: float | None = None  # optional radial energy for this isometric pool
    pool_residual: float | None = None  # optional prior-subspace subtraction strength [0, 1]
    priority: float = 0.0         # rep fold order: higher folds first (anchors)

    @property
    def label(self) -> str:
        return self.name or self.source.name


def _weights_tensor(weights: Sequence[float] | None, count: int, *, device: torch.device) -> torch.Tensor:
    if weights is None:
        return torch.ones(count, device=device, dtype=torch.float32)
    if len(weights) != count:
        raise ValueError(f"expected {count} weights, got {len(weights)}")
    return torch.tensor(list(weights), device=device, dtype=torch.float32)


def sum_weighted_deltas(
    deltas: Sequence[torch.Tensor],
    weights: Sequence[float] | None = None,
) -> torch.Tensor:
    """Weighted task arithmetic in fp32."""
    if not deltas:
        raise ValueError("at least one delta is required")
    stack = torch.stack([d.to(torch.float32) for d in deltas], dim=0)
    w = _weights_tensor(weights, len(deltas), device=stack.device).view(-1, *([1] * (stack.ndim - 1)))
    return (stack * w).sum(dim=0)


def sign_agreement_vote(
    deltas: Sequence[torch.Tensor],
    weights: Sequence[float] | None = None,
) -> torch.Tensor:
    """Merge deltas using majority sign agreement in the full tensor frame.

    Signs are computed after applying merge weights, so a negative weight
    intentionally votes for the opposite direction. Vote strength is the
    absolute merge weight. Positions with ties are dropped.
    """
    if not deltas:
        raise ValueError("at least one delta is required")
    stack = torch.stack([d.to(torch.float32) for d in deltas], dim=0)
    w = _weights_tensor(weights, len(deltas), device=stack.device).view(-1, *([1] * (stack.ndim - 1)))
    weighted = stack * w
    signs = torch.sign(weighted)
    vote_strength = torch.abs(w)
    winning = torch.sign((signs * vote_strength).sum(dim=0))
    mask = (winning != 0).unsqueeze(0) & (signs == winning.unsqueeze(0))
    return torch.where(mask, weighted, torch.zeros_like(weighted)).sum(dim=0)


def _linear_direction_fallback(u0: torch.Tensor, u1: torch.Tensor, t: float, eps: float) -> torch.Tensor:
    direction = (1.0 - t) * u0 + t * u1
    norm = direction.norm()
    if norm <= eps:
        return direction
    return direction / norm


def norm_aware_slerp(
    delta_a: torch.Tensor,
    delta_b: torch.Tensor,
    *,
    t: float,
    eps: float = 1e-12,
) -> torch.Tensor:
    """SLERP directions while linearly interpolating delta norms.

    This reuses the repository's native ``slerp`` primitive, but wraps the
    zero/parallel edge cases so merge code can operate on arbitrary tensor
    deltas.
    """
    if not 0.0 <= t <= 1.0:
        raise ValueError("t must be in [0, 1]")
    a = delta_a.to(torch.float32)
    b = delta_b.to(torch.float32).to(a.device)
    if a.shape != b.shape:
        raise ValueError(f"delta shapes differ: {tuple(a.shape)} vs {tuple(b.shape)}")

    flat_a = a.reshape(-1)
    flat_b = b.reshape(-1)
    norm_a = flat_a.norm()
    norm_b = flat_b.norm()
    if norm_a <= eps and norm_b <= eps:
        return torch.zeros_like(a)
    if norm_a <= eps:
        return (t * b).to(torch.float32)
    if norm_b <= eps:
        return ((1.0 - t) * a).to(torch.float32)

    u0 = flat_a / norm_a
    u1 = flat_b / norm_b
    dot = torch.clamp(torch.dot(u0, u1), -1.0, 1.0)
    if torch.abs(dot) > 1.0 - 1e-6:
        direction = _linear_direction_fallback(u0, u1, t, eps)
    else:
        direction = slerp(u0, u1, t=t)

    target_norm = (1.0 - t) * norm_a + t * norm_b
    return (direction.reshape_as(a) * target_norm).to(torch.float32)


def norm_slerp_merge(
    deltas: Sequence[torch.Tensor],
    weights: Sequence[float] | None = None,
    *,
    final_norm: float | None = None,
) -> torch.Tensor:
    """Iteratively merge deltas with norm-aware SLERP.

    Weights are non-negative mixture weights. Use ``sign_agreement_vote`` or
    ``sum_weighted_deltas`` for subtractive/negative-weight merges.
    """
    if not deltas:
        raise ValueError("at least one delta is required")
    w = _weights_tensor(weights, len(deltas), device=deltas[0].device)
    if torch.any(w < 0):
        raise ValueError("norm_slerp weights must be non-negative")
    if torch.all(w == 0):
        return torch.zeros_like(deltas[0], dtype=torch.float32)

    acc: torch.Tensor | None = None
    acc_weight = 0.0
    for delta, weight in zip(deltas, w.tolist()):
        if weight == 0:
            continue
        current = delta.to(torch.float32)
        if acc is None:
            acc = current
            acc_weight = float(weight)
            continue
        t = float(weight) / (acc_weight + float(weight))
        acc = norm_aware_slerp(acc, current, t=t)
        acc_weight += float(weight)
    if acc is None:
        return torch.zeros_like(deltas[0], dtype=torch.float32)
    if final_norm is not None:
        n = acc.norm()
        if n > 1e-12:
            acc = acc / n * final_norm
    return acc


def fourier_slerp_merge(
    deltas: Sequence[torch.Tensor],
    weights: Sequence[float] | None = None,
    *,
    final_norm: float | None = None,
    device: torch.device | str | None = None,
    cutoff_pct: float = 0.08,
    cull_pct: float = 0.0,
    small_norm_ratio: float = 0.1,
    eps: float = 1e-12,
) -> torch.Tensor:
    """Iteratively merge deltas with the existing FFT-SLERP tensor primitive.

    The Fourier merge operates on directions in the repository's FFT-SLERP
    frame, then rescales the result to the weighted average norm. Weights are
    non-negative mixture weights; use ``sum`` or ``sign_vote`` for subtractive
    merges.
    """
    if not deltas:
        raise ValueError("at least one delta is required")
    work_device = torch.device(device) if device is not None else deltas[0].device
    w = _weights_tensor(weights, len(deltas), device=work_device)
    if torch.any(w < 0):
        raise ValueError("fourier_slerp weights must be non-negative")
    if torch.all(w == 0):
        return torch.zeros_like(deltas[0], dtype=torch.float32, device=work_device)

    acc: torch.Tensor | None = None
    acc_weight = 0.0
    for delta, weight in zip(deltas, w.tolist()):
        if weight == 0:
            continue
        current = delta.to(device=work_device, dtype=torch.float32)
        if acc is None:
            acc = current
            acc_weight = float(weight)
            continue

        t = float(weight) / (acc_weight + float(weight))
        norm_acc = acc.norm()
        norm_current = current.norm()
        target_norm = (1.0 - t) * norm_acc + t * norm_current

        if norm_acc <= eps and norm_current <= eps:
            acc = torch.zeros_like(current)
        elif norm_acc <= eps:
            acc = current * t
        elif norm_current <= eps:
            acc = acc * (1.0 - t)
        else:
            merged, _, _ = merge_tensors_fft2_slerp(
                acc,
                current,
                t=t,
                device=str(work_device),
                b=small_norm_ratio,
                cutoff_pct=cutoff_pct,
                cull_pct=cull_pct,
            )
            merged = merged.to(device=work_device, dtype=torch.float32)
            merged_norm = merged.norm()
            acc = merged * (target_norm / merged_norm.clamp_min(eps))

        acc_weight += float(weight)

    if acc is None:
        return torch.zeros_like(deltas[0], dtype=torch.float32, device=work_device)
    if final_norm is not None:
        n = acc.norm()
        if n > eps:
            acc = acc / n * final_norm
    return acc


def _project_to_so(matrix: torch.Tensor) -> torch.Tensor:
    """Nearest proper rotation (det +1) to ``matrix`` via SVD (polar factor)."""
    U, _S, Vh = torch.linalg.svd(matrix)
    R = U @ Vh
    if torch.det(R) < 0:
        U = U.clone()
        U[:, -1] = -U[:, -1]
        R = U @ Vh
    return R


def isometric_merge_delta(
    base: torch.Tensor,
    deltas: Sequence[torch.Tensor],
    weights: Sequence[float] | None = None,
    *,
    rank: int = 32,
    energy: float = 0.0,
    whitener: tuple[torch.Tensor, torch.Tensor] | None = None,
) -> torch.Tensor:
    """Isometric, complement-preserving merge (heretic paper Section 3.2).

    Each source delta is reduced to the rotation it induces on the base weight
    *within the low-rank subspace it actually changed*; those rotations are
    averaged on SO(m) (polar factor of the weighted sum) and applied as
    ``W -> W + Z (R_M - I) Z^T W``. Because ``R_M`` is orthogonal and ``Z``
    orthonormal, the merged weight has the SAME norm as the base (the rotated
    component keeps its length, the orthogonal complement is untouched) -- so
    there is no norm restoration, and thus no repetition-loop failure mode. The
    radial/magnitude part of each finetune is intentionally dropped; only the
    on-manifold rotation survives.

    Returns the merged DELTA (``Z (R_M - I) Z^T W``); the caller adds the base.
    """
    W = base.to(torch.float32)
    if W.ndim != 2 or not deltas:
        return sum_weighted_deltas(deltas, weights) if deltas else torch.zeros_like(W)
    if weights is None:
        weights = [1.0] * len(deltas)

    out_dim, in_dim = W.shape
    r = min(rank, out_dim, in_dim)

    # Joint output subspace: top-r left-singular directions each delta changed.
    # With a whitener (Vk, sqrt_lam) we select directions of the H-weighted delta
    # d @ H^(1/2) (H^(1/2) ~= Vk diag(sqrt_lam) Vk^T) -- the behaviorally-active
    # directions in the RMSNorm/Gram metric, not raw Frobenius. (v2)
    Vk = sqrt_lam = None
    if whitener is not None:
        Vk, sqrt_lam = whitener
        Vk = Vk.to(W.device, torch.float32)
        sqrt_lam = sqrt_lam.to(W.device, torch.float32)
    cols = []
    behav = []                                              # ||d H^(1/2)|| per source (v3 weight)
    for d in deltas:
        d = d.to(torch.float32)
        if Vk is not None and Vk.shape[0] == in_dim:
            dsel = (d @ Vk) * sqrt_lam                       # [out, k] behavioral (H-weighted)
        else:
            dsel = d
        behav.append(float(dsel.norm()))
        q = min(dsel.shape[0], dsel.shape[1], r + 8)
        U, _S, _V = torch.svd_lowrank(dsel, q=q)
        cols.append(U[:, :r])
    Z, _ = torch.linalg.qr(torch.cat(cols, dim=1))          # [out, m] orthonormal
    WZ = Z.transpose(0, 1) @ W                               # [m, in]

    # Per-source Procrustes rotation (R WZ ~= Z^T(W+d)), averaged on SO(m). v3:
    # weight by each source's BEHAVIORAL energy ||d H^(1/2)|| -- how much it moves the
    # states the model visits -- not its raw Frobenius norm. This stops a large-but-
    # noisy dense tune (equinox) from drowning the behaviorally-potent low-norm LoRAs
    # (fabled, the RP adapters). Without a whitener, behav == ||d|| (falls back to v1).
    acc = torch.zeros(Z.shape[1], Z.shape[1], device=W.device, dtype=W.dtype)
    ztd_acc = torch.zeros(Z.shape[1], in_dim, device=W.device, dtype=W.dtype)
    wsum = 0.0
    for d, w, e in zip(deltas, weights, behav):
        ztd = Z.transpose(0, 1) @ d.to(torch.float32)       # Z^T d (the change in-subspace)
        R = _project_to_so((WZ + ztd) @ WZ.transpose(0, 1))  # rotate base toward this finetune
        acc = acc + float(w) * e * R                         # rotation, behavioral-energy-weighted
        ztd_acc = ztd_acc + float(w) * e * ztd               # radial part, IN-SUBSPACE
        wsum += float(w) * e
    R_M = _project_to_so(acc)

    eye = torch.eye(Z.shape[1], device=W.device, dtype=W.dtype)
    merged = Z @ ((R_M - eye) @ WZ)                          # rotation, norm-preserving
    if energy > 0.0 and wsum > 0.0:
        # Reintroduce energy in the rank-r subspace -- where the SFTs' behavioral
        # signal actually lives (spectral knee). Tested full-dense and it diluted
        # into equinox's noise floor (KL back to passthrough); the low-rank head
        # is the signal. energy=0 stays isometric; residence bounds the rest.
        merged = merged + energy * (Z @ (ztd_acc / wsum))
    return merged


def isometric_pooled_merge_delta(
    base: torch.Tensor,
    deltas: Sequence[torch.Tensor],
    weights: Sequence[float],
    pools: Sequence[str | None],
    priorities: Sequence[float],
    pool_ranks: Sequence[int | None] | None = None,
    pool_energies: Sequence[float | None] | None = None,
    pool_residuals: Sequence[float | None] | None = None,
    *,
    rank: int = 32,
    energy: float = 0.0,
    whitener: tuple[torch.Tensor, torch.Tensor] | None = None,
) -> torch.Tensor:
    """Priority-ordered block-isometric merge over explicit pools.

    The plain ``isometric_merge_delta`` averages EVERY source's rotation into one
    ``R_M`` in one shared subspace, so sources that move the same readout directions
    pull the polar-factor mean toward identity and cancel (the v2 dilution: pile all
    reasoning sources into one crowded plane -> merge falls below base).

    Here sources are grouped into POOLS (by ``pool``). Pools are processed by
    DESCENDING priority. Each pool claims an output basis selected directly from its
    residualized source signal, then receives its own Procrustes rotation block. With
    zero radial energy, the update is one block-orthogonal isometry, not a sum of
    independently-selected merge deltas. Lower-priority pools only contribute signal
    that survives residualization against already-claimed output subspaces.
    """
    W = base.to(torch.float32)
    if W.ndim != 2 or not deltas:
        return sum_weighted_deltas(deltas, weights) if deltas else torch.zeros_like(W)
    if weights is None:
        weights = [1.0] * len(deltas)

    by_pool: dict[str, list[int]] = {}
    prio: dict[str, float] = {}
    for i, (p, pr) in enumerate(zip(pools, priorities)):
        key = p if p else f"__solo_{i}"
        by_pool.setdefault(key, []).append(i)
        prio[key] = max(prio.get(key, -1e30), float(pr))
    order = sorted(by_pool, key=lambda k: (-prio[k], by_pool[k][0]))

    Vk = sqrt_lam = None
    if whitener is not None:
        Vk, sqrt_lam = whitener
        Vk = Vk.to(W.device, torch.float32)
        sqrt_lam = sqrt_lam.to(W.device, torch.float32)

    def setting(seq, idx: list[int], default):
        if seq is None:
            return default
        for i in idx:
            if seq[i] is not None:
                return seq[i]
        return default

    tol = 1e-6
    Q: torch.Tensor | None = None                       # claimed output basis [out, k]
    total = torch.zeros_like(W)
    for key in order:
        idx = by_pool[key]
        ds = [deltas[i].to(torch.float32) for i in idx]
        ws = [weights[i] for i in idx]
        pool_rank = max(1, int(setting(pool_ranks, idx, rank)))
        pool_energy = float(setting(pool_energies, idx, energy))
        residual_strength = min(1.0, max(0.0, float(setting(pool_residuals, idx, 1.0))))
        if Q is not None:
            ds = [d - residual_strength * (Q @ (Q.transpose(0, 1) @ d)) for d in ds]
        if max((float(d.norm()) for d in ds), default=0.0) < tol:
            continue                                    # this pool's signal is fully claimed away

        dsel = []
        behav = []
        for d, w in zip(ds, ws):
            if Vk is not None and Vk.shape[0] == W.shape[1]:
                selected = (d @ Vk) * sqrt_lam
            else:
                selected = d
            dsel.append(selected * abs(float(w)))
            behav.append(float(selected.norm()))
        signal = torch.cat(dsel, dim=1)
        if Q is not None:
            signal = signal - Q @ (Q.transpose(0, 1) @ signal)
        q = min(signal.shape[0], signal.shape[1], pool_rank + 8)
        U, S, _ = torch.svd_lowrank(signal, q=q)
        keep = S > tol * float(S[0]) if S.numel() and float(S[0]) > 0 else torch.zeros_like(S, dtype=torch.bool)
        Z = U[:, keep][:, :pool_rank]
        if Z.numel() == 0:
            continue
        if Q is not None:
            Z = Z - Q @ (Q.transpose(0, 1) @ Z)
        Z, Rz = torch.linalg.qr(Z)
        diag = Rz.diagonal().abs()
        Z = Z[:, diag > tol]
        if Z.numel() == 0:
            continue

        WZ = Z.transpose(0, 1) @ W
        acc = torch.zeros(Z.shape[1], Z.shape[1], device=W.device, dtype=W.dtype)
        ztd_acc = torch.zeros(Z.shape[1], W.shape[1], device=W.device, dtype=W.dtype)
        wsum = 0.0
        for d, w, e in zip(ds, ws, behav):
            ztd = Z.transpose(0, 1) @ d
            R = _project_to_so((WZ + ztd) @ WZ.transpose(0, 1))
            scaled = float(w) * e
            acc = acc + scaled * R
            ztd_acc = ztd_acc + scaled * ztd
            wsum += scaled
        if abs(wsum) < tol:
            continue
        R_M = _project_to_so(acc)
        eye = torch.eye(Z.shape[1], device=W.device, dtype=W.dtype)
        total = total + Z @ ((R_M - eye) @ WZ)
        if pool_energy > 0.0 and wsum > 0.0:
            total = total + pool_energy * (Z @ (ztd_acc / wsum))

        Q = Z if Q is None else torch.cat([Q, Z], dim=1)
    return total


def collect_source_deltas(
    sources: Sequence[WeightedDeltaSource],
    tensor_name: str,
    *,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float32,
) -> tuple[list[torch.Tensor], list[float], list[WeightedDeltaSource]]:
    """Materialize all source deltas that contain ``tensor_name``."""
    deltas: list[torch.Tensor] = []
    weights: list[float] = []
    matched: list[WeightedDeltaSource] = []
    for source in sources:
        delta = source.source.delta(tensor_name, device=device, dtype=dtype)
        if delta is None:
            continue
        deltas.append(delta.to(torch.float32) * float(source.rescale))
        weights.append(float(source.weight))
        matched.append(source)
    return deltas, weights, matched


def _norm_reference(deltas: Sequence[torch.Tensor], matched: Sequence["WeightedDeltaSource"]) -> float | None:
    """Mean delta norm of the is_norm sources -> the merged output magnitude."""
    refs = [float(d.norm()) for d, s in zip(deltas, matched) if s.is_norm]
    if not refs:
        return None
    return sum(refs) / len(refs)


def merge_full_frame_tensor(
    base: torch.Tensor,
    sources: Sequence[WeightedDeltaSource],
    tensor_name: str,
    *,
    rule: MergeRule = "fourier_slerp",
    device: torch.device | str | None = None,
    fourier_cutoff_pct: float = 0.08,
    fourier_cull_pct: float = 0.0,
    fourier_small_norm_ratio: float = 0.1,
    final_merge_t: float = 0.5,
    isometric_rank: int = 32,
    isometric_energy: float = 0.0,
    whiteners: dict | None = None,
    readout_whitener: tuple[torch.Tensor, torch.Tensor] | None = None,
) -> torch.Tensor:
    """Merge all source deltas for one tensor in fp32 and return patched weight.

    Two-phase: the body sources are merged with ``rule`` into a single delta;
    then each ``post_merge`` source is applied against that consolidated result.
    For isometric merges, post sources are additive overlays scaled by
    ``final_merge_t``. This preserves surgical source-relative patches instead
    of projecting them through the body rotation.
    """
    base_fp32 = base.to(device=device, dtype=torch.float32) if device is not None else base.to(torch.float32)
    deltas, weights, matched = collect_source_deltas(sources, tensor_name, device=base_fp32.device, dtype=torch.float32)
    if not deltas:
        return base_fp32
    target_norm = _norm_reference(deltas, matched)   # None unless an is_norm source is present

    if rule == "isometric":
        # Norm-preserving rotation merge; no group/is_norm renorm (by design).
        # Surgical post sources are excluded from the rotation and applied to the
        # consolidated body afterward in the original parameter frame.
        body = [(d, w, s) for d, w, s in zip(deltas, weights, matched) if not s.post_merge]
        post = [(d, w) for d, w, s in zip(deltas, weights, matched) if s.post_merge]
        body_deltas = [d for d, _w, _s in body]
        body_weights = [w for _d, w, _s in body]
        body_sources = [s for _d, _w, s in body]

        def add_post_overlays(merged_delta: torch.Tensor) -> torch.Tensor:
            if not post or final_merge_t == 0.0:
                return merged_delta
            overlay = sum_weighted_deltas(
                [d for d, _w in post],
                [float(final_merge_t) * w for _d, w in post],
            )
            return merged_delta + overlay.to(merged_delta.device)

        if not body:
            return base_fp32 + add_post_overlays(torch.zeros_like(base_fp32))

        # READOUT-NORM capture (research/RESULTS_READOUT_NORM_RANK): for residual-stream
        # writers (out_dim == hidden), run the operator in the LM-head readout metric --
        # whiten output rows by M^½, rotate, un-whiten with a DAMPED M^{-½} (which also
        # projects the writeback onto the ~36-dim readout-relevant aperture). rank-32 in
        # this metric keeps the behaviorally-live directions, not the top-Frobenius ones.
        # v3b: if any source carries a `pool`, run priority-ordered residualized
        # blocks instead of one global rotation-average. Legacy `group` remains
        # the non-isometric fold hint and does not opt into this path.
        pooled = any(s.pool for s in body_sources)
        pools = [s.pool for s in body_sources]
        prios = [s.priority for s in body_sources]
        pool_ranks = [s.pool_rank for s in body_sources]
        pool_energies = [s.pool_energy for s in body_sources]
        pool_residuals = [s.pool_residual for s in body_sources]
        if readout_whitener is not None and base_fp32.ndim == 2 \
                and base_fp32.shape[0] == readout_whitener[0].shape[0]:
            Mh = readout_whitener[0].to(base_fp32.device, torch.float32)
            Mhi = readout_whitener[1].to(base_fp32.device, torch.float32)
            wbase, wdeltas = Mh @ base_fp32, [Mh @ d for d in body_deltas]
            if pooled:
                merged_w = isometric_pooled_merge_delta(wbase, wdeltas, body_weights, pools, prios,
                                                        pool_ranks, pool_energies, pool_residuals,
                                                        rank=isometric_rank, energy=isometric_energy)
            else:
                merged_w = isometric_merge_delta(wbase, wdeltas, body_weights,
                                                 rank=isometric_rank, energy=isometric_energy, whitener=None)
            merged_delta = (Mhi @ merged_w).to(base_fp32.device)
            return base_fp32 + add_post_overlays(merged_delta)
        whitener = (whiteners or {}).get(int(base_fp32.shape[1])) if base_fp32.ndim == 2 else None
        if pooled:
            merged_delta = isometric_pooled_merge_delta(base_fp32, body_deltas, body_weights, pools, prios,
                                                        pool_ranks, pool_energies, pool_residuals,
                                                        rank=isometric_rank, energy=isometric_energy,
                                                        whitener=whitener)
        else:
            merged_delta = isometric_merge_delta(base_fp32, body_deltas, body_weights, rank=isometric_rank,
                                                 energy=isometric_energy, whitener=whitener)
        return base_fp32 + add_post_overlays(merged_delta.to(base_fp32.device))

    def apply(ds: list[torch.Tensor], ws: list[float], final_norm: float | None) -> torch.Tensor:
        if rule == "sum":
            return sum_weighted_deltas(ds, ws)
        if rule == "sign_vote":
            return sign_agreement_vote(ds, ws)
        if rule == "norm_slerp":
            return norm_slerp_merge(ds, ws, final_norm=final_norm)
        if rule in {"fourier_slerp", "fft_slerp"}:
            return fourier_slerp_merge(
                ds, ws, final_norm=final_norm, device=base_fp32.device,
                cutoff_pct=fourier_cutoff_pct, cull_pct=fourier_cull_pct,
                small_norm_ratio=fourier_small_norm_ratio,
            )
        raise ValueError(f"unknown merge rule {rule!r}")

    body = [(d, w, s) for d, w, s in zip(deltas, weights, matched) if not s.post_merge]
    post = [(d, w) for d, w, s in zip(deltas, weights, matched) if s.post_merge]

    if body:
        # Level 1: fold same-group body sources INTO EACH OTHER -> one rep per
        # group. Ungrouped sources are their own singleton group.
        groups: dict[str, dict] = {}
        for i, (d, w, s) in enumerate(body):
            key = s.group if s.group else f"__solo_{i}"
            g = groups.setdefault(key, {"prio": s.priority, "first": i, "deltas": [], "weights": []})
            g["deltas"].append(d)
            g["weights"].append(w)
            g["prio"] = max(g["prio"], s.priority)
        reps = []  # (prio, first_index, rep_delta, total_weight)
        for g in groups.values():
            rep = apply(g["deltas"], g["weights"], None)  # single-member -> returned as-is
            reps.append((g["prio"], g["first"], rep, sum(g["weights"])))
        # Level 2: fold the group reps by descending priority (stable on first seen).
        reps.sort(key=lambda r: (-r[0], r[1]))
        merged_delta = apply([r[2] for r in reps], [r[3] for r in reps], None)
        # Level 3: blend each post-merge source against the final body delta.
        t = float(final_merge_t)
        for post_delta, _post_weight in post:
            merged_delta = apply([merged_delta, post_delta], [1.0 - t, t], None)
        if target_norm is not None:
            n = merged_delta.norm()
            if n > 1e-12:
                merged_delta = merged_delta / n * target_norm
    else:
        # No body sources -> nothing to merge the post layer against; fold the
        # post sources among themselves as a degenerate body.
        merged_delta = apply([d for d, _ in post], [w for _, w in post], target_norm)

    return base_fp32 + merged_delta.to(base_fp32.device)
