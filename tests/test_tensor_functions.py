# tests/test_tensor_functions.py
import pytest
import torch
from shard.tensor.functions import (
    slerp,
    fft_transform,
    ifft_transform,
    normalize_tensor,
    interpolate_fft_components,
    merge_tensors_fft2_slerp,
    task_arithmetic_fft2,
    arithmetic_fft_components,
    correlate_pairs,
    correlated_pairs
)


class TestSlerpAndNormalization:
    """Test suite for SLERP and normalization functions"""

    def test_normalize_tensor_nonzero(self):
        """Test normalize_tensor with non-zero tensor"""
        tensor = torch.tensor([3.0, 4.0])
        normalized, norm = normalize_tensor(tensor, device="cpu")

        expected_norm = 5.0  # sqrt(3^2 + 4^2)
        assert abs(norm - expected_norm) < 1e-5
        assert torch.allclose(normalized.norm(), torch.tensor(1.0))

    def test_normalize_tensor_zero(self):
        """Test normalize_tensor with zero tensor"""
        tensor = torch.zeros(3, 3)
        normalized, norm = normalize_tensor(tensor, device="cpu")

        assert norm == 0.0
        assert torch.all(normalized == 0.0)


class TestFFTTransforms:
    """Test suite for FFT transformation functions"""

    def test_fft_transform_1d(self):
        """Test fft_transform with 1D tensor"""
        tensor = torch.randn(10)
        result = fft_transform(tensor, device="cpu")

        assert result.dtype == torch.complex64
        assert result.shape == tensor.shape

    def test_fft_transform_2d(self):
        """Test fft_transform with 2D tensor"""
        tensor = torch.randn(5, 5)
        result = fft_transform(tensor, device="cpu")

        assert result.dtype == torch.complex64
        assert result.shape == tensor.shape

    def test_ifft_transform_1d(self):
        """Test ifft_transform with 1D tensor"""
        tensor = torch.randn(10)
        fft = fft_transform(tensor, device="cpu")
        recovered = ifft_transform(fft, device="cpu")

        assert torch.allclose(recovered, tensor, atol=1e-5)

    def test_ifft_transform_2d(self):
        """Test ifft_transform with 2D tensor"""
        tensor = torch.randn(5, 5)
        fft = fft_transform(tensor, device="cpu")
        recovered = ifft_transform(fft, device="cpu")

        assert torch.allclose(recovered, tensor, atol=1e-5)


class TestInterpolateFftComponents:
    """Test suite for interpolate_fft_components"""

    def test_interpolate_with_cutoff(self):
        """Test interpolate_fft_components with cutoff_pct"""
        v0 = torch.randn(4, 4)
        v1 = torch.randn(4, 4)

        v0_fft = fft_transform(v0, device="cpu")
        v1_fft = fft_transform(v1, device="cpu")

        result = interpolate_fft_components(
            v0_fft, v1_fft,
            t=0.5,
            device="cpu",
            cutoff_pct=0.1
        )

        assert result.shape == v0_fft.shape
        assert result.dtype == torch.complex64

    def test_interpolate_with_cull(self):
        """Test interpolate_fft_components with cull_pct"""
        v0 = torch.randn(4, 4)
        v1 = torch.randn(4, 4)

        v0_fft = fft_transform(v0, device="cpu")
        v1_fft = fft_transform(v1, device="cpu")

        result = interpolate_fft_components(
            v0_fft, v1_fft,
            t=0.5,
            device="cpu",
            cull_pct=0.1
        )

        assert result.shape == v0_fft.shape

    def test_interpolate_without_imag(self):
        """Test interpolate_fft_components with interp_imag=False"""
        v0 = torch.randn(4, 4)
        v1 = torch.randn(4, 4)

        v0_fft = fft_transform(v0, device="cpu")
        v1_fft = fft_transform(v1, device="cpu")

        result = interpolate_fft_components(
            v0_fft, v1_fft,
            t=0.5,
            device="cpu",
            interp_imag=False
        )

        assert torch.allclose(result.imag, v0_fft.imag)


class TestMergeTensorsFft2Slerp:
    """Test suite for merge_tensors_fft2_slerp"""

    def test_merge_small_norm_v1(self):
        """Test merge_tensors_fft2_slerp when v1 has very small norm"""
        v0 = torch.randn(4, 4)
        v1 = torch.zeros(4, 4)  # Very small norm

        merged, norm_v0, norm_v1 = merge_tensors_fft2_slerp(
            v0, v1,
            t=0.5,
            device="cpu"
        )

        assert norm_v1 < 0.0001
        # Should return normalized v0
        assert torch.allclose(merged / merged.norm(), v0 / v0.norm(), atol=1e-4)

    def test_merge_small_norm_v0(self):
        """Test merge_tensors_fft2_slerp when v0 has very small norm"""
        v0 = torch.zeros(4, 4)
        v1 = torch.randn(4, 4)

        merged, norm_v0, norm_v1 = merge_tensors_fft2_slerp(
            v0, v1,
            t=0.5,
            device="cpu"
        )

        assert norm_v0 < 0.0001
        # Should still return v0 (handle edge case)

    def test_merge_small_ratio(self):
        """Test merge_tensors_fft2_slerp with small norm ratio"""
        v0 = torch.randn(4, 4) * 10.0  # Large norm
        v1 = torch.randn(4, 4) * 0.01  # Small norm

        merged, norm_v0, norm_v1 = merge_tensors_fft2_slerp(
            v0, v1,
            t=0.5,
            device="cpu",
            b=0.1  # Threshold
        )

        # Should use simple addition path
        assert merged is not None

    def test_merge_nan_handling(self):
        """Test merge_tensors_fft2_slerp handles NaN values"""
        v0 = torch.randn(4, 4)
        v1 = torch.randn(4, 4)

        # Force NaN by creating problematic values
        v0[0, 0] = float('inf')
        v1[0, 0] = float('inf')

        try:
            merged, norm_v0, norm_v1 = merge_tensors_fft2_slerp(
                v0, v1,
                t=0.5,
                device="cpu"
            )
            # NaN values should be set to 0
            assert not torch.any(torch.isnan(merged))
        except ValueError:
            # May raise ValueError for inf
            pass


class TestTaskArithmeticFft2:
    """Test suite for task_arithmetic_fft2"""

    def test_task_arithmetic_with_agreement(self):
        """Test task_arithmetic_fft2 with agreement=True"""
        v0 = torch.randn(4, 4)
        v1 = torch.randn(4, 4)

        result = task_arithmetic_fft2(v0, v1, t=0.5, device="cpu", agreement=True)

        assert result.shape == v0.shape
        assert result.dtype == v0.dtype

    def test_task_arithmetic_without_agreement(self):
        """Test task_arithmetic_fft2 with agreement=False"""
        v0 = torch.randn(4, 4)
        v1 = torch.randn(4, 4)

        result = task_arithmetic_fft2(v0, v1, t=0.5, device="cpu", agreement=False)

        assert result.shape == v0.shape
        assert result.dtype == v0.dtype


class TestArithmeticFftComponents:
    """Test suite for arithmetic_fft_components"""

    def test_arithmetic_with_do_imag(self):
        """Test arithmetic_fft_components with do_imag=True"""
        v0 = torch.randn(4, 4)
        v1 = torch.randn(4, 4)

        v0_fft = fft_transform(v0, device="cpu")
        v1_fft = fft_transform(v1, device="cpu")

        result = arithmetic_fft_components(
            v0_fft, v1_fft,
            t=1.0,
            agreement=True,
            device="cpu",
            do_imag=True
        )

        assert result.shape == v0_fft.shape

    def test_arithmetic_without_do_imag(self):
        """Test arithmetic_fft_components with do_imag=False"""
        v0 = torch.randn(4, 4)
        v1 = torch.randn(4, 4)

        v0_fft = fft_transform(v0, device="cpu")
        v1_fft = fft_transform(v1, device="cpu")

        result = arithmetic_fft_components(
            v0_fft, v1_fft,
            t=1.0,
            agreement=True,
            device="cpu",
            do_imag=False
        )

        assert torch.equal(result.imag, v0_fft.imag)


class TestCorrelatePairs:
    """Test suite for correlate_pairs and correlated_pairs"""

    def test_correlate_pairs(self):
        """Test correlate_pairs computes correlation matrix"""
        tensors = torch.stack([
            torch.randn(10),
            torch.randn(10),
            torch.randn(10)
        ])

        matrix = correlate_pairs(tensors, work_device="cpu", store_device="cpu")

        assert matrix.shape == (3, 3)
        # Matrix should be symmetric
        assert torch.allclose(matrix, matrix.T)

    def test_correlated_pairs_least(self):
        """Test correlated_pairs with way='least'"""
        matrix = torch.tensor([
            [0.0, 0.5, 0.9],
            [0.5, 0.0, 0.7],
            [0.9, 0.7, 0.0]
        ])

        pairs = list(correlated_pairs(matrix, way='least'))

        # Should yield pairs in order of least correlation
        assert len(pairs) > 0
        # Check that we get valid indices
        for x, y, coeff in pairs:
            if y >= 0:
                assert 0 <= x < 3
                assert 0 <= y < 3
                assert x != y

    def test_correlated_pairs_most(self):
        """Test correlated_pairs with way='most'"""
        matrix = torch.tensor([
            [0.0, 0.5, 0.9],
            [0.5, 0.0, 0.7],
            [0.9, 0.7, 0.0]
        ])

        pairs = list(correlated_pairs(matrix, way='most'))

        # Should yield pairs in order of most correlation
        assert len(pairs) > 0

    def test_correlated_pairs_invalid_way(self):
        """Test correlated_pairs raises error for invalid way"""
        matrix = torch.eye(3)

        with pytest.raises(ValueError, match="Invalid way"):
            list(correlated_pairs(matrix, way='invalid'))

    def test_correlated_pairs_odd_number(self):
        """Test correlated_pairs handles odd number of items"""
        matrix = torch.tensor([
            [0.0, 0.5, 0.9, 0.3],
            [0.5, 0.0, 0.7, 0.4],
            [0.9, 0.7, 0.0, 0.6],
            [0.3, 0.4, 0.6, 0.0]
        ])

        pairs = list(correlated_pairs(matrix, way='least'))

        # Should have 4 entries (2 pairs + 0 leftover for even number)
        # Or 3 entries (1 pair + 2 unpaired) for odd handling
        assert len(pairs) >= 2
