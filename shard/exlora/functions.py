# shard/exlora/functions.py
# Copyright (C) 2024 Martin Bukowski
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

"""Reference tensor functions for ExLoRA research variants.

This module intentionally contains pure tensor math with no model I/O. The
current ``svd.py`` path remains the exact baseline; these helpers are the
building blocks for activation-weighted, Hessian-projected, and
rotation/reflection-preconditioned extraction.
"""

from __future__ import annotations

from typing import Optional

import torch


def _as_work_device(device: Optional[str | torch.device], fallback: torch.device) -> torch.device:
    return torch.device(device) if device is not None else fallback


def _check_same_shape(a: torch.Tensor, b: torch.Tensor, what: str = "tensors") -> None:
    if a.shape != b.shape:
        raise ValueError(f"{what} must have the same shape, got {tuple(a.shape)} and {tuple(b.shape)}")


def compute_delta(
    base: torch.Tensor,
    target: torch.Tensor,
    compute_dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Return ``target - base`` in ``compute_dtype``."""
    _check_same_shape(base, target, "base and target tensors")
    return target.to(compute_dtype) - base.to(compute_dtype)


def lora_delta(
    lora_A: torch.Tensor,
    lora_B: torch.Tensor,
    scale: float = 1.0,
    compute_dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Reconstruct a dense LoRA delta from PEFT-convention ``A`` and ``B``."""
    if lora_A.ndim != 2 or lora_B.ndim != 2:
        raise ValueError("LoRA factors must be 2D tensors")
    if lora_A.shape[0] != lora_B.shape[1]:
        raise ValueError(
            f"LoRA rank mismatch: A is {tuple(lora_A.shape)}, B is {tuple(lora_B.shape)}"
        )
    return scale * (lora_B.to(compute_dtype) @ lora_A.to(compute_dtype))


def relative_frobenius_error(
    reference: torch.Tensor,
    approximation: torch.Tensor,
    eps: float = 1e-12,
) -> float:
    """Return ``||reference - approximation||_F / ||reference||_F``.

    If the reference is zero, returns ``0`` for an exact zero approximation and
    ``inf`` otherwise.
    """
    _check_same_shape(reference, approximation, "reference and approximation")
    ref = reference.to(torch.float32)
    approx = approximation.to(torch.float32)
    denom = ref.norm()
    residual = (ref - approx).norm()
    if denom.item() <= eps:
        return 0.0 if residual.item() <= eps else float("inf")
    return float((residual / denom).item())


def activation_hessian(
    activations: torch.Tensor,
    normalize: bool = True,
    compute_dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Estimate ``H = X.T @ X`` from layer input activations.

    ``activations`` may be ``[tokens, features]`` or have any leading batch
    dimensions ending in ``features``. Leading dimensions are flattened.
    """
    if activations.ndim < 2:
        raise ValueError("activations must have at least two dimensions")
    x = activations.reshape(-1, activations.shape[-1]).to(compute_dtype)
    hessian = x.T @ x
    if normalize and x.shape[0] > 0:
        hessian = hessian / x.shape[0]
    return hessian


def symmetric_psd_sqrt(
    matrix: torch.Tensor,
    damping: float = 0.0,
    eps: float = 0.0,
) -> torch.Tensor:
    """Return the symmetric square root of a PSD matrix.

    The input is symmetrized before decomposition. Negative eigenvalues caused
    by numerical noise are clamped to zero.
    """
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f"matrix must be square, got {tuple(matrix.shape)}")
    work = (matrix.to(torch.float32) + matrix.to(torch.float32).T) * 0.5
    if damping:
        work = work + torch.eye(work.shape[0], device=work.device, dtype=work.dtype) * damping
    eigvals, eigvecs = torch.linalg.eigh(work)
    eigvals = eigvals.clamp_min(eps)
    sqrt_vals = torch.sqrt(eigvals)
    return (eigvecs * sqrt_vals.unsqueeze(0)) @ eigvecs.T


def damped_pinv(
    matrix: torch.Tensor,
    damping: float = 0.0,
    rcond: float = 1e-7,
) -> torch.Tensor:
    """Return a Moore-Penrose or Tikhonov-damped pseudoinverse."""
    if matrix.ndim != 2:
        raise ValueError(f"matrix must be 2D, got {tuple(matrix.shape)}")
    work = matrix.to(torch.float32)
    U, S, Vh = torch.linalg.svd(work, full_matrices=False)
    if len(S) == 0:
        return work.T
    if damping > 0:
        factors = S / (S.square() + damping)
    else:
        cutoff = rcond * S.max()
        factors = torch.where(S > cutoff, 1.0 / S, torch.zeros_like(S))
    return Vh.T @ (factors.unsqueeze(1) * U.T)


def solve_right_basis_coefficients(
    reference: torch.Tensor,
    right_basis: torch.Tensor,
    metric: Optional[torch.Tensor] = None,
    damping: float = 0.0,
    rcond: float = 1e-7,
) -> torch.Tensor:
    """Solve the best left factor ``B`` for a fixed right basis ``A``.

    Given ``reference`` with shape ``[out, in]`` and ``right_basis`` with shape
    ``[rank, in]``, this returns the least-squares ``B`` for ``reference ~= B A``.

    With ``metric=None`` this minimizes raw Frobenius error. With a right-side
    activation metric ``H`` it minimizes ``|| (reference - B A) H^(1/2) ||_F``:

    ```text
    B = reference H A.T (A H A.T)^+
    ```

    ``metric`` may be a full square matrix ``[in, in]`` or a diagonal vector
    ``[in]``. The diagonal form is useful for cheap activation or scale proxies.
    """
    if reference.ndim != 2 or right_basis.ndim != 2:
        raise ValueError("reference and right_basis must both be 2D")
    if reference.shape[1] != right_basis.shape[1]:
        raise ValueError(
            "reference and right_basis input dimensions must match, "
            f"got {tuple(reference.shape)} and {tuple(right_basis.shape)}"
        )

    ref = reference.to(torch.float32)
    basis = right_basis.to(device=ref.device, dtype=torch.float32)

    if metric is None:
        rhs = ref @ basis.T
        gram = basis @ basis.T
    else:
        h = metric.to(device=ref.device, dtype=torch.float32)
        if h.ndim == 1:
            if h.shape[0] != ref.shape[1]:
                raise ValueError(
                    f"diagonal metric length must be {ref.shape[1]}, got {h.shape[0]}"
                )
            weighted_basis = basis * h.unsqueeze(0)
            rhs = ref @ weighted_basis.T
            gram = weighted_basis @ basis.T
        elif h.ndim == 2:
            if h.shape != (ref.shape[1], ref.shape[1]):
                raise ValueError(
                    "full metric must have shape "
                    f"{(ref.shape[1], ref.shape[1])}, got {tuple(h.shape)}"
                )
            h_basis_t = h @ basis.T
            rhs = ref @ h_basis_t
            gram = basis @ h_basis_t
        else:
            raise ValueError("metric must be None, a diagonal vector, or a square matrix")

    pinv = damped_pinv(gram, damping=damping, rcond=rcond)
    return rhs @ pinv


def right_basis_reflection(
    reference: torch.Tensor,
    right_basis: torch.Tensor,
    metric: Optional[torch.Tensor] = None,
    damping: float = 0.0,
    rcond: float = 1e-7,
) -> torch.Tensor:
    """Project ``reference`` onto the row span of ``right_basis``.

    This is the additive correction used for factorized error feedback:

    ```text
    correction = B* A
    B* = argmin_B ||(reference - B A) H^(1/2)||_F
    ```

    It is a stable projection/reflection primitive: there is no per-parameter
    divide, only a small ``rank x rank`` pseudoinverse.
    """
    left = solve_right_basis_coefficients(
        reference,
        right_basis,
        metric=metric,
        damping=damping,
        rcond=rcond,
    )
    return left @ right_basis.to(device=left.device, dtype=torch.float32)


def right_weight_delta(
    delta: torch.Tensor,
    metric_sqrt: torch.Tensor,
) -> torch.Tensor:
    """Return the activation-weighted delta ``delta @ C``."""
    if delta.ndim != 2 or metric_sqrt.ndim != 2:
        raise ValueError("delta and metric_sqrt must both be 2D")
    if delta.shape[1] != metric_sqrt.shape[0]:
        raise ValueError(
            f"shape mismatch for delta @ metric_sqrt: {tuple(delta.shape)} and {tuple(metric_sqrt.shape)}"
        )
    return delta.to(torch.float32) @ metric_sqrt.to(torch.float32)


def project_weighted_lora_A(
    weighted_lora_A: torch.Tensor,
    metric_sqrt: torch.Tensor,
    damping: float = 1e-5,
) -> torch.Tensor:
    """Project weighted LoRA ``A`` back from ``Delta C`` coordinates.

    If ``weighted_lora_A`` approximates the right factor of ``Delta @ C``, the
    ordinary-weight coordinate factor is ``weighted_lora_A @ C^+``.
    """
    if weighted_lora_A.ndim != 2 or metric_sqrt.ndim != 2:
        raise ValueError("weighted_lora_A and metric_sqrt must both be 2D")
    if weighted_lora_A.shape[1] != metric_sqrt.shape[0]:
        raise ValueError(
            f"shape mismatch for A @ C^+: {tuple(weighted_lora_A.shape)} and {tuple(metric_sqrt.shape)}"
        )
    return weighted_lora_A.to(torch.float32) @ damped_pinv(metric_sqrt, damping=damping)


def weighted_relative_error(
    reference: torch.Tensor,
    approximation: torch.Tensor,
    metric_sqrt: torch.Tensor,
) -> float:
    """Return relative error under the right-side activation metric."""
    return relative_frobenius_error(
        right_weight_delta(reference, metric_sqrt),
        right_weight_delta(approximation, metric_sqrt),
    )


def split_svd_to_lora(
    U: torch.Tensor,
    S: torch.Tensor,
    Vh: torch.Tensor,
    rank: int,
    out_dtype: Optional[torch.dtype] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert SVD pieces into PEFT-convention LoRA factors ``(A, B)``."""
    if U.ndim != 2 or S.ndim != 1 or Vh.ndim != 2:
        raise ValueError("U, S, and Vh must have shapes [out,k], [k], [k,in]")
    cap = min(U.shape[1], S.shape[0], Vh.shape[0])
    if rank < 1 or rank > cap:
        raise ValueError(f"rank must be in [1, {cap}], got {rank}")
    U_r = U[:, :rank]
    S_r = S[:rank]
    Vh_r = Vh[:rank, :]
    sqrt_s = torch.sqrt(S_r)
    lora_B = (U_r * sqrt_s.unsqueeze(0)).contiguous()
    lora_A = (sqrt_s.unsqueeze(1) * Vh_r).contiguous()
    if out_dtype is not None:
        lora_A = lora_A.to(out_dtype)
        lora_B = lora_B.to(out_dtype)
    return lora_A, lora_B


def randomized_svd(
    matrix: torch.Tensor,
    rank: int,
    oversampling: int = 16,
    power_iterations: int = 1,
    generator: Optional[torch.Generator] = None,
    compute_dtype: torch.dtype = torch.float32,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Reference randomized SVD for a dense matrix.

    This is not the tiled production path. It is a compact reference
    implementation for validating the math on small tensors.
    """
    if matrix.ndim != 2:
        raise ValueError(f"matrix must be 2D, got {tuple(matrix.shape)}")
    m, n = matrix.shape
    cap = min(m, n)
    if rank < 1 or rank > cap:
        raise ValueError(f"rank must be in [1, {cap}], got {rank}")
    sketch = min(cap, rank + max(oversampling, 0))
    work = matrix.to(compute_dtype)
    omega = torch.randn(
        n,
        sketch,
        generator=generator,
        device=work.device,
        dtype=work.dtype,
    )
    Y = work @ omega
    for _ in range(power_iterations):
        Y = work @ (work.T @ Y)
    Q, _ = torch.linalg.qr(Y, mode="reduced")
    small = Q.T @ work
    U_small, S, Vh = torch.linalg.svd(small, full_matrices=False)
    U = Q @ U_small
    return U[:, :rank].contiguous(), S[:rank].contiguous(), Vh[:rank, :].contiguous()


def householder_reflect(
    tensor: torch.Tensor,
    vector: torch.Tensor,
    dim: int = -1,
    eps: float = 1e-12,
) -> torch.Tensor:
    """Reflect ``tensor`` across the Euclidean Householder vector on ``dim``."""
    if vector.ndim != 1:
        raise ValueError("vector must be 1D")
    dim = dim % tensor.ndim
    if tensor.shape[dim] != vector.shape[0]:
        raise ValueError(
            f"vector length {vector.shape[0]} does not match tensor dim {dim} of size {tensor.shape[dim]}"
        )
    work = torch.movedim(tensor.to(torch.float32), dim, -1)
    u = vector.to(device=work.device, dtype=work.dtype)
    denom = torch.dot(u, u)
    if denom.item() <= eps:
        return tensor.clone()
    projection = torch.tensordot(work, u, dims=([-1], [0])) / denom
    reflected = work - 2.0 * projection.unsqueeze(-1) * u
    return torch.movedim(reflected, -1, dim)


def metric_householder_reflect(
    tensor: torch.Tensor,
    vector: torch.Tensor,
    metric: torch.Tensor,
    dim: int = -1,
    eps: float = 1e-12,
) -> torch.Tensor:
    """Reflect ``tensor`` using the metric inner product ``u.T @ H @ v``."""
    if vector.ndim != 1:
        raise ValueError("vector must be 1D")
    if metric.ndim != 2 or metric.shape[0] != metric.shape[1]:
        raise ValueError("metric must be square")
    dim = dim % tensor.ndim
    if tensor.shape[dim] != vector.shape[0] or metric.shape[0] != vector.shape[0]:
        raise ValueError("tensor dimension, vector length, and metric size must match")
    work = torch.movedim(tensor.to(torch.float32), dim, -1)
    h = metric.to(device=work.device, dtype=work.dtype)
    u = vector.to(device=work.device, dtype=work.dtype)
    hu = h @ u
    denom = torch.dot(u, hu)
    if denom.item() <= eps:
        return tensor.clone()
    projection = torch.tensordot(work, hu, dims=([-1], [0])) / denom
    reflected = work - 2.0 * projection.unsqueeze(-1) * u
    return torch.movedim(reflected, -1, dim)


def cayley_transform(skew: torch.Tensor) -> torch.Tensor:
    """Return the orthogonal Cayley transform ``(I - A)^-1 (I + A)``."""
    if skew.ndim != 2 or skew.shape[0] != skew.shape[1]:
        raise ValueError("skew must be square")
    work = skew.to(torch.float32)
    eye = torch.eye(work.shape[0], device=work.device, dtype=work.dtype)
    return torch.linalg.solve(eye - work, eye + work)


def skew_symmetric(matrix: torch.Tensor) -> torch.Tensor:
    """Return the skew-symmetric part of a square matrix."""
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("matrix must be square")
    work = matrix.to(torch.float32)
    return 0.5 * (work - work.T)


def orthogonal_procrustes(source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Find ``R`` minimizing ``||source @ R - target||_F``."""
    if source.ndim != 2 or target.ndim != 2:
        raise ValueError("source and target must be 2D")
    if source.shape != target.shape:
        raise ValueError(f"source and target must have same shape, got {source.shape} and {target.shape}")
    cross = source.to(torch.float32).T @ target.to(torch.float32)
    U, _, Vh = torch.linalg.svd(cross, full_matrices=False)
    return U @ Vh


def tiled_delta_right_matmul(
    base: torch.Tensor,
    target: torch.Tensor,
    rhs: torch.Tensor,
    row_tile_size: int = 1024,
    col_tile_size: Optional[int] = None,
    compute_dtype: torch.dtype = torch.float32,
    work_device: Optional[str | torch.device] = None,
    store_device: Optional[str | torch.device] = None,
) -> torch.Tensor:
    """Compute ``(target - base) @ rhs`` with row/column tiling."""
    _check_same_shape(base, target, "base and target tensors")
    if base.ndim != 2 or rhs.ndim != 2:
        raise ValueError("base, target, and rhs must be 2D")
    out_features, in_features = base.shape
    if rhs.shape[0] != in_features:
        raise ValueError(f"rhs has incompatible shape {tuple(rhs.shape)} for delta {tuple(base.shape)}")
    if row_tile_size < 1:
        raise ValueError("row_tile_size must be positive")
    col_tile_size = col_tile_size or in_features
    if col_tile_size < 1:
        raise ValueError("col_tile_size must be positive")

    store = _as_work_device(store_device, rhs.device)
    work = _as_work_device(work_device, base.device)
    result = torch.empty(out_features, rhs.shape[1], device=store, dtype=compute_dtype)

    for r0 in range(0, out_features, row_tile_size):
        r1 = min(r0 + row_tile_size, out_features)
        acc = torch.zeros(r1 - r0, rhs.shape[1], device=work, dtype=compute_dtype)
        for c0 in range(0, in_features, col_tile_size):
            c1 = min(c0 + col_tile_size, in_features)
            delta_tile = compute_delta(
                base[r0:r1, c0:c1].to(work),
                target[r0:r1, c0:c1].to(work),
                compute_dtype=compute_dtype,
            )
            rhs_tile = rhs[c0:c1, :].to(device=work, dtype=compute_dtype)
            acc = acc + delta_tile @ rhs_tile
        result[r0:r1, :] = acc.to(store)
    return result


def tiled_delta_left_matmul(
    left: torch.Tensor,
    base: torch.Tensor,
    target: torch.Tensor,
    row_tile_size: int = 1024,
    col_tile_size: Optional[int] = None,
    compute_dtype: torch.dtype = torch.float32,
    work_device: Optional[str | torch.device] = None,
    store_device: Optional[str | torch.device] = None,
) -> torch.Tensor:
    """Compute ``left.T @ (target - base)`` with row/column tiling."""
    _check_same_shape(base, target, "base and target tensors")
    if base.ndim != 2 or left.ndim != 2:
        raise ValueError("left, base, and target must be 2D")
    out_features, in_features = base.shape
    if left.shape[0] != out_features:
        raise ValueError(f"left has incompatible shape {tuple(left.shape)} for delta {tuple(base.shape)}")
    if row_tile_size < 1:
        raise ValueError("row_tile_size must be positive")
    col_tile_size = col_tile_size or in_features
    if col_tile_size < 1:
        raise ValueError("col_tile_size must be positive")

    store = _as_work_device(store_device, left.device)
    work = _as_work_device(work_device, base.device)
    result = torch.zeros(left.shape[1], in_features, device=store, dtype=compute_dtype)

    for r0 in range(0, out_features, row_tile_size):
        r1 = min(r0 + row_tile_size, out_features)
        left_tile = left[r0:r1, :].to(device=work, dtype=compute_dtype)
        for c0 in range(0, in_features, col_tile_size):
            c1 = min(c0 + col_tile_size, in_features)
            delta_tile = compute_delta(
                base[r0:r1, c0:c1].to(work),
                target[r0:r1, c0:c1].to(work),
                compute_dtype=compute_dtype,
            )
            block = left_tile.T @ delta_tile
            result[:, c0:c1] = result[:, c0:c1] + block.to(store)
    return result
