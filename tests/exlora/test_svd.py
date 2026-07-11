# tests/exlora/test_svd.py
import pytest
import torch

from shard.exlora.svd import (
    effective_rank,
    noise_aware_singular_values,
    retained_energy,
    relative_errors,
    select_rank,
    select_signal_rank,
    spectrum_diagnostics,
    factor_delta_to_lora,
    reconstruct_delta,
)


def make_low_rank_delta(out_features: int, in_features: int, rank: int, seed: int = 0) -> torch.Tensor:
    """Construct a delta of exactly `rank` by multiplying two thin factors."""
    g = torch.Generator().manual_seed(seed)
    B0 = torch.randn(out_features, rank, generator=g)
    A0 = torch.randn(rank, in_features, generator=g)
    return B0 @ A0


class TestRelativeErrors:
    def test_length_and_endpoints(self):
        s = torch.tensor([4.0, 3.0, 2.0, 1.0])
        errs = relative_errors(s)
        assert len(errs) == len(s) + 1
        assert errs[0].item() == pytest.approx(1.0)       # keep nothing
        assert errs[-1].item() == pytest.approx(0.0)      # keep everything

    def test_monotonic_non_increasing(self):
        s = torch.tensor([10.0, 5.0, 2.0, 0.5, 0.1])
        errs = relative_errors(s)
        diffs = errs[1:] - errs[:-1]
        assert torch.all(diffs <= 1e-9)

    def test_zero_singular_values(self):
        errs = relative_errors(torch.zeros(3))
        assert torch.all(errs == 0.0)


class TestSelectRank:
    def test_recovers_true_rank(self):
        # singular spectrum with a sharp cliff after 3 components
        s = torch.tensor([10.0, 8.0, 6.0, 1e-7, 1e-8])
        rank, err = select_rank(s, target_error=1e-3, min_rank=1, max_rank=5)
        assert rank == 3
        assert err <= 1e-3

    def test_respects_min_rank(self):
        s = torch.tensor([10.0, 1e-9, 1e-9])
        rank, _ = select_rank(s, target_error=1e-3, min_rank=2, max_rank=5)
        assert rank >= 2

    def test_respects_max_rank(self):
        # slowly decaying spectrum that never hits the target within the cap
        s = torch.linspace(1.0, 0.5, 20)
        rank, _ = select_rank(s, target_error=1e-6, min_rank=1, max_rank=4)
        assert rank == 4

    def test_clamped_to_available(self):
        s = torch.tensor([1.0, 0.5])
        rank, _ = select_rank(s, target_error=0.0, min_rank=1, max_rank=100)
        assert rank <= 2

    def test_empty(self):
        rank, err = select_rank(torch.zeros(0), target_error=0.1)
        assert rank == 0 and err == 0.0


class TestSignalRank:
    def test_retained_energy(self):
        s = torch.tensor([3.0, 1.0])
        assert retained_energy(s, 0) == pytest.approx(0.0)
        assert retained_energy(s, 1) == pytest.approx(9.0 / 10.0)
        assert retained_energy(s, 2) == pytest.approx(1.0)

    def test_effective_rank_for_flat_spectrum(self):
        assert effective_rank(torch.ones(8)) == pytest.approx(8.0)

    def test_detects_low_rank_head_over_noise_floor(self):
        s = torch.cat([torch.linspace(10.0, 5.0, 4), torch.full((20,), 0.5)])
        rank, err, diag = select_signal_rank(
            s,
            max_rank=24,
            noise_floor_multiplier=1.5,
            min_knee_drop=1.25,
            tail_fraction=0.25,
        )
        assert rank == 4
        assert diag.signal_reason == "noise_floor"
        assert diag.knee_rank == 4
        assert diag.retained_energy > 0.97
        assert err == pytest.approx(diag.rel_error)

    def test_smooth_spectrum_does_not_invent_signal_floor(self):
        s = torch.linspace(1.0, 0.8, 20)
        rank, _, diag = select_signal_rank(s, max_rank=10)
        assert rank == 10
        assert diag.signal_rank is None
        assert diag.signal_reason == "no_signal_floor"

    def test_zero_spectrum_is_safe(self):
        diag = spectrum_diagnostics(torch.zeros(5), min_rank=2)
        assert diag.signal_rank == 2
        assert diag.rel_error == 0.0


class TestNoiseAwareSingularValues:
    def test_rank_masks_tail(self):
        s = torch.tensor([4.0, 3.0, 2.0, 1.0])
        shrunk = noise_aware_singular_values(s, rank=2, mode="none")
        assert torch.allclose(shrunk, torch.tensor([4.0, 3.0, 0.0, 0.0]))

    def test_quadratic_subtracts_noise_power(self):
        s = torch.tensor([5.0, 3.0])
        noise = torch.tensor([3.0, 4.0])
        shrunk = noise_aware_singular_values(s, noise, mode="quadratic")
        assert shrunk[0].item() == pytest.approx(4.0)
        assert shrunk[1].item() == pytest.approx(0.0)

    def test_ridge_keeps_under_floor_signal_shape(self):
        s = torch.tensor([5.0, 1.0])
        noise = torch.tensor([1.0, 2.0])
        hard = noise_aware_singular_values(s, noise, mode="hard")
        ridge = noise_aware_singular_values(s, noise, mode="ridge")
        assert hard[1].item() == pytest.approx(0.0)
        assert 0.0 < ridge[1].item() < s[1].item()
        assert ridge[0].item() > ridge[1].item()


class TestFactorDelta:
    def test_shapes_follow_peft_convention(self):
        delta = make_low_rank_delta(16, 12, rank=4)
        factor = factor_delta_to_lora(delta, target_error=1e-4, max_rank=8, out_dtype=torch.float32)
        assert factor.lora_B.shape == (16, factor.rank)   # [out, r]
        assert factor.lora_A.shape == (factor.rank, 12)   # [r, in]

    def test_recovers_rank_and_reconstructs(self):
        delta = make_low_rank_delta(32, 24, rank=5, seed=3)
        factor = factor_delta_to_lora(delta, target_error=1e-4, max_rank=16, out_dtype=torch.float32)
        # A rank-5 matrix should be captured at (close to) rank 5.
        assert factor.rank == 5
        recon = reconstruct_delta(factor)
        rel = (recon - delta).norm() / delta.norm()
        assert rel.item() < 1e-4

    def test_higher_rank_lowers_error(self):
        delta = torch.randn(40, 30)  # full-rank, no clean cliff
        low = factor_delta_to_lora(delta, target_error=0.0, max_rank=4, out_dtype=torch.float32)
        high = factor_delta_to_lora(delta, target_error=0.0, max_rank=20, out_dtype=torch.float32)
        assert high.rank > low.rank
        assert high.rel_error < low.rel_error

    def test_rejects_non_2d(self):
        with pytest.raises(ValueError):
            factor_delta_to_lora(torch.randn(8), target_error=0.1)

    def test_zero_delta_is_safe(self):
        factor = factor_delta_to_lora(torch.zeros(8, 8), target_error=0.01, out_dtype=torch.float32)
        assert factor.rank >= 1
        assert torch.all(reconstruct_delta(factor) == 0.0)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
    def test_factor_on_cuda_delta(self):
        # Regression: rank selection must not mix cuda/cpu tensors.
        delta = make_low_rank_delta(64, 48, rank=6, seed=11).cuda()
        factor = factor_delta_to_lora(delta, target_error=1e-4, max_rank=16, out_dtype=torch.float32)
        assert factor.rank == 6
        recon = reconstruct_delta(factor)
        rel = (recon.cuda() - delta).norm() / delta.norm()
        assert rel.item() < 1e-4
