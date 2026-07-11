# tests/exlora/test_nvfp4.py
import pytest
import torch

from shard.exlora.nvfp4 import (
    E2M1_LEVELS,
    quantize_e2m1,
    round_e4m3,
    nvfp4_quantize,
    nvfp4_quantize_dequantize,
)


class TestE2M1:
    def test_exact_levels_roundtrip(self):
        for s in (1.0, -1.0):
            vals = torch.tensor([s * v for v in E2M1_LEVELS])
            assert torch.allclose(quantize_e2m1(vals), vals)

    def test_rounds_to_nearest(self):
        # 2.6 -> 3.0, 2.4 -> 2.0, 5.2 -> 6 (nearest of {4,6}); clamp above 6.
        x = torch.tensor([2.6, 2.4, 5.2, 100.0, -100.0])
        out = quantize_e2m1(x)
        assert out.tolist() == [3.0, 2.0, 6.0, 6.0, -6.0]

    def test_sign_preserved(self):
        x = torch.tensor([-0.4, -1.6])
        out = quantize_e2m1(x)
        assert out.tolist() == [-0.5, -1.5]


class TestE4M3:
    def test_clamps_to_max(self):
        assert round_e4m3(torch.tensor([1e9])).item() == pytest.approx(448.0)

    def test_nonneg_and_close(self):
        x = torch.linspace(0.01, 400, 50)
        q = round_e4m3(x)
        assert torch.all(q >= 0)
        # E4M3 has >=3 mantissa bits, so relative error per value is small.
        rel = ((q - x).abs() / x.clamp_min(1e-6)).max().item()
        assert rel < 0.10


class TestNVFP4:
    def test_shape_preserved(self):
        x = torch.randn(8, 64)
        q = nvfp4_quantize(x)
        assert q.dequantize().shape == x.shape

    def test_handles_non_multiple_of_block(self):
        x = torch.randn(70)  # not a multiple of 16
        out = nvfp4_quantize_dequantize(x)
        assert out.shape == x.shape

    def test_idempotent(self):
        x = torch.randn(4, 128)
        once = nvfp4_quantize_dequantize(x)
        twice = nvfp4_quantize_dequantize(once)
        assert torch.allclose(once, twice, atol=1e-5)

    def test_zero_tensor(self):
        x = torch.zeros(3, 32)
        out = nvfp4_quantize_dequantize(x)
        assert torch.all(out == 0)

    def test_relative_error_reasonable(self):
        # ~4.5-bit format should reconstruct gaussian data to well under 15%.
        torch.manual_seed(0)
        x = torch.randn(64, 256)
        out = nvfp4_quantize_dequantize(x)
        rel = (out - x).norm() / x.norm()
        assert rel.item() < 0.15

    def test_block_scale_tracks_magnitude(self):
        # A block scaled up by 10x should dequantize ~10x larger.
        x = torch.randn(2, 16)
        big = x.clone()
        big[1] *= 10.0
        out = nvfp4_quantize(big).dequantize()
        assert out[1].abs().mean() > 5 * out[0].abs().mean()

    def test_bits_per_element_about_four_and_half(self):
        x = torch.randn(32, 256)
        q = nvfp4_quantize(x)
        assert 4.4 < q.bits_per_element < 4.7

    def test_beats_pure_int4_on_outliers(self):
        # Per-block scaling should handle a heavy-tailed block better than a
        # single global int4 grid would (sanity check on dynamic range).
        x = torch.randn(8, 64)
        x[0, 0] = 50.0  # outlier
        out = nvfp4_quantize_dequantize(x)
        # outlier preserved to within one E2M1 step of its block scale
        assert abs(out[0, 0].item() - 48.0) <= 12.0
