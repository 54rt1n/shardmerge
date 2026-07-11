# tests/exlora/test_comb.py
# Copyright (C) 2026 Martin Bukowski

import torch

from shard.delta.terms import LoraDelta
from shard.exlora.comb import (
    CombCorrectorFit,
    CombCorrectorPolicy,
    damped_whitening,
    fit_comb_corrector,
)


def _spd(dim, seed, scale_tail=0.15, strong=6):
    g = torch.Generator().manual_seed(seed)
    Q, _ = torch.linalg.qr(torch.randn(dim, dim, generator=g))
    lam = torch.cat([torch.linspace(8.0, 4.0, strong), torch.full((dim - strong,), scale_tail)])
    return (Q * lam) @ Q.T


def test_policy_caps_and_includes_zero():
    pol = CombCorrectorPolicy(candidate_ranks=(0, 8, 16, 32, 64, 128), max_rank=64)
    assert pol.effective_ranks() == (0, 8, 16, 32, 64)   # 128 dropped by cap, 0 always present


def test_damped_whitening_clamps_inverse_gain_and_is_symmetric():
    H = _spd(20, seed=0, scale_tail=1e-4)                 # near-singular tail
    h_half, h_inv_half = damped_whitening(H, relative_damping=1e-3, max_inverse_gain=3.0)
    assert torch.allclose(h_inv_half, h_inv_half.T, atol=1e-5)
    # the largest inverse-root eigenvalue is clamped
    assert float(torch.linalg.eigvalsh(h_inv_half).max()) <= 3.0 + 1e-4
    # undamped/unclamped gain is much larger -> clamp actually bites
    _, h_inv_raw = damped_whitening(H, relative_damping=1e-9, max_inverse_gain=None)
    assert float(torch.linalg.eigvalsh(h_inv_raw).max()) > 10.0


def test_fit_shapes_and_lora_delta_reduces_residual():
    torch.manual_seed(0)
    inf, outf = 24, 12
    H = _spd(inf, seed=1)
    E = torch.randn(outf, inf)
    fit = fit_comb_corrector(E, H, H, CombCorrectorPolicy(candidate_ranks=(0, 4, 8), max_rank=8))
    assert fit.rank in (4, 8)
    assert fit.lora_A.shape == (fit.rank, inf)
    assert fit.lora_B.shape == (outf, fit.rank)
    term = fit.to_lora_delta()
    assert isinstance(term, LoraDelta)
    corrected = E - term.materialize(dtype=torch.float32)
    assert corrected.norm() < E.norm()                    # the corrector reduces the residual


def test_held_out_error_is_monotone_under_min_norm_fit():
    # The min-norm pinv fit never overshoots, so held-out activation error is
    # non-increasing in rank. This is *why* the activation gate cannot reproduce
    # the (nonlinear, KL-level) rank regression -- it is a relevance/decline gate,
    # not a rank cap. The cap + held-out KL own that.
    torch.manual_seed(2)
    inf, outf = 28, 16
    Hf, Hv = _spd(inf, seed=3), _spd(inf, seed=4)
    E = torch.randn(outf, inf)
    pol = CombCorrectorPolicy(candidate_ranks=(0, 2, 4, 8, 12, 16), max_rank=16, relative_damping=1e-2)
    fit = fit_comb_corrector(E, Hf, Hv, pol)
    ranks = sorted(fit.val_errors)
    errs = [fit.val_errors[r] for r in ranks]
    assert all(b <= a + 1e-5 for a, b in zip(errs, errs[1:]))   # non-increasing
    assert fit.rank == max(r for r in ranks if r > 0)           # picks the cap when comb is relevant


def test_declines_when_comb_is_behaviorally_irrelevant():
    # gram_fit and gram_val weight disjoint input directions; the corrector can
    # only cancel E in fit-weighted directions, which the val metric ignores ->
    # no positive rank improves held-out error -> decline to rank 0.
    torch.manual_seed(5)
    inf, outf = 32, 10
    half = inf // 2
    fit_w = torch.zeros(inf); fit_w[:half] = 1.0
    val_w = torch.zeros(inf); val_w[half:] = 1.0
    Hf, Hv = torch.diag(fit_w), torch.diag(val_w)
    E = torch.randn(outf, inf)                              # energy in both halves
    fit = fit_comb_corrector(E, Hf, Hv, CombCorrectorPolicy(candidate_ranks=(0, 4, 8), max_rank=8))
    assert fit.rank == 0
    assert fit.reason == "declined"
    assert fit.to_lora_delta() is None
    assert all(abs(v - 1.0) < 1e-4 for k, v in fit.val_errors.items())
