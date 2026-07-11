import math

import pytest
import torch

from shard.exlora.functions import (
    activation_hessian,
    cayley_transform,
    compute_delta,
    damped_pinv,
    householder_reflect,
    lora_delta,
    metric_householder_reflect,
    orthogonal_procrustes,
    project_weighted_lora_A,
    randomized_svd,
    relative_frobenius_error,
    right_basis_reflection,
    right_weight_delta,
    solve_right_basis_coefficients,
    skew_symmetric,
    split_svd_to_lora,
    symmetric_psd_sqrt,
    tiled_delta_left_matmul,
    tiled_delta_right_matmul,
    weighted_relative_error,
)


def test_compute_delta_casts_and_subtracts():
    base = torch.tensor([[1, 2], [3, 4]], dtype=torch.int64)
    target = torch.tensor([[2, 5], [7, 11]], dtype=torch.int64)
    delta = compute_delta(base, target)
    assert delta.dtype == torch.float32
    assert torch.equal(delta, torch.tensor([[1.0, 3.0], [4.0, 7.0]]))


def test_relative_frobenius_error_handles_zero_reference():
    assert relative_frobenius_error(torch.zeros(2, 2), torch.zeros(2, 2)) == 0.0
    assert math.isinf(relative_frobenius_error(torch.zeros(2, 2), torch.ones(2, 2)))


def test_activation_hessian_flattens_leading_dims_and_normalizes():
    x = torch.tensor([
        [[1.0, 2.0], [3.0, 4.0]],
        [[5.0, 6.0], [7.0, 8.0]],
    ])
    flat = x.reshape(-1, 2)
    expected = flat.T @ flat / flat.shape[0]
    assert torch.allclose(activation_hessian(x), expected)


def test_symmetric_psd_sqrt_reconstructs_metric():
    x = torch.randn(16, 4)
    h = activation_hessian(x, normalize=False)
    c = symmetric_psd_sqrt(h)
    assert torch.allclose(c @ c.T, h, atol=1e-4, rtol=1e-4)


def test_damped_pinv_reduces_small_singular_amplification():
    matrix = torch.diag(torch.tensor([10.0, 0.001]))
    plain = damped_pinv(matrix, damping=0.0)
    damped = damped_pinv(matrix, damping=1e-2)
    assert plain[1, 1] > 900.0
    assert damped[1, 1] < 1.0


def test_split_svd_to_lora_reconstructs_rank_truncation():
    delta = torch.randn(12, 8)
    U, S, Vh = torch.linalg.svd(delta, full_matrices=False)
    lora_A, lora_B = split_svd_to_lora(U, S, Vh, rank=5)
    recon = lora_delta(lora_A, lora_B)
    expected = U[:, :5] @ torch.diag(S[:5]) @ Vh[:5, :]
    assert torch.allclose(recon, expected, atol=1e-5, rtol=1e-5)


def test_weighted_projection_roundtrip_for_identity_metric():
    weighted_A = torch.randn(3, 6)
    metric = torch.eye(6)
    projected = project_weighted_lora_A(weighted_A, metric, damping=0.0)
    assert torch.allclose(projected, weighted_A, atol=1e-6)


def test_weighted_relative_error_uses_right_metric():
    ref = torch.eye(2)
    approx = torch.tensor([[1.0, 0.0], [0.0, 0.0]])
    metric = torch.diag(torch.tensor([1.0, 0.0]))
    assert weighted_relative_error(ref, approx, metric) == pytest.approx(0.0)
    assert relative_frobenius_error(ref, approx) > 0.0


def test_right_weight_delta_matches_dense_matmul():
    delta = torch.randn(5, 4)
    metric = torch.randn(4, 3)
    assert torch.allclose(right_weight_delta(delta, metric), delta @ metric)


def test_solve_right_basis_coefficients_exact_for_spanned_rows():
    basis = torch.tensor([[1.0, 0.0, 1.0], [0.0, 2.0, 0.0]])
    left = torch.tensor([[2.0, -1.0], [0.5, 3.0]])
    reference = left @ basis

    solved = solve_right_basis_coefficients(reference, basis)
    reflected = solved @ basis

    assert torch.allclose(reflected, reference, atol=1e-5)


def test_right_basis_reflection_is_orthogonal_projection():
    reference = torch.tensor([[1.0, 2.0, 3.0]])
    basis = torch.tensor([[1.0, 0.0, 0.0]])

    reflected = right_basis_reflection(reference, basis)

    assert torch.allclose(reflected, torch.tensor([[1.0, 0.0, 0.0]]), atol=1e-6)


def test_metric_right_basis_reflection_favors_weighted_coordinates():
    reference = torch.tensor([[1.0, -1.0]])
    basis = torch.tensor([[1.0, 1.0]])
    metric = torch.tensor([100.0, 1.0])

    unweighted = right_basis_reflection(reference, basis)
    weighted = right_basis_reflection(reference, basis, metric=metric)

    unweighted_err = ((reference - unweighted).square() * metric).sum()
    weighted_err = ((reference - weighted).square() * metric).sum()
    assert weighted_err < unweighted_err


def test_householder_reflect_preserves_norm_and_flips_vector():
    u = torch.tensor([1.0, 2.0, 3.0])
    reflected_u = householder_reflect(u, u)
    assert torch.allclose(reflected_u, -u, atol=1e-6)

    x = torch.randn(7, 3)
    reflected = householder_reflect(x, u, dim=-1)
    assert torch.allclose(reflected.norm(dim=-1), x.norm(dim=-1), atol=1e-5)


def test_metric_householder_reflect_preserves_metric_norm():
    h = torch.diag(torch.tensor([1.0, 2.0, 4.0]))
    u = torch.tensor([0.5, -1.0, 0.25])
    x = torch.randn(9, 3)
    reflected = metric_householder_reflect(x, u, h, dim=-1)
    before = torch.einsum("bi,ij,bj->b", x, h, x)
    after = torch.einsum("bi,ij,bj->b", reflected, h, reflected)
    assert torch.allclose(after, before, atol=1e-5, rtol=1e-5)


def test_cayley_transform_is_orthogonal_for_skew_matrix():
    a = skew_symmetric(torch.randn(5, 5))
    r = cayley_transform(a)
    eye = torch.eye(5)
    assert torch.allclose(r.T @ r, eye, atol=1e-5, rtol=1e-5)


def test_orthogonal_procrustes_recovers_rotation():
    source = torch.randn(32, 5)
    rotation = cayley_transform(skew_symmetric(torch.randn(5, 5) * 0.1))
    target = source @ rotation
    estimated = orthogonal_procrustes(source, target)
    assert torch.allclose(estimated, rotation, atol=1e-5, rtol=1e-5)


def test_randomized_svd_recovers_low_rank_matrix():
    g = torch.Generator().manual_seed(123)
    left = torch.randn(30, 4, generator=g)
    right = torch.randn(4, 20, generator=g)
    matrix = left @ right
    U, S, Vh = randomized_svd(
        matrix,
        rank=4,
        oversampling=4,
        power_iterations=1,
        generator=torch.Generator().manual_seed(456),
    )
    recon = U @ torch.diag(S) @ Vh
    rel = relative_frobenius_error(matrix, recon)
    assert rel < 1e-5


def test_tiled_delta_right_matmul_matches_dense():
    base = torch.randn(7, 5)
    target = torch.randn(7, 5)
    rhs = torch.randn(5, 3)
    expected = (target - base) @ rhs
    actual = tiled_delta_right_matmul(base, target, rhs, row_tile_size=3, col_tile_size=2)
    assert torch.allclose(actual, expected, atol=1e-6)


def test_tiled_delta_left_matmul_matches_dense():
    base = torch.randn(7, 5)
    target = torch.randn(7, 5)
    left = torch.randn(7, 3)
    expected = left.T @ (target - base)
    actual = tiled_delta_left_matmul(left, base, target, row_tile_size=3, col_tile_size=2)
    assert torch.allclose(actual, expected, atol=1e-6)
