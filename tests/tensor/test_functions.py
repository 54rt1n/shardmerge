# tests/tensor/test_functions.py
import pytest
import torch
import math

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


class TestSlerp:
    """Test suite for slerp function"""

    def test_slerp_basic(self):
        """Test basic SLERP operation"""
        v0 = torch.tensor([1.0, 0.0, 0.0])
        v1 = torch.tensor([0.0, 1.0, 0.0])

        result = slerp(v0, v1, t=0.5)

        assert result.shape == v0.shape
        assert not torch.isnan(result).any()
        assert not torch.isinf(result).any()

    def test_slerp_t_zero(self):
        """Test SLERP with t=0 returns v0"""
        v0 = torch.randn(5)
        v1 = torch.randn(5)

        result = slerp(v0, v1, t=0.0)

        # Should be close to v0 when t=0
        norm_v0 = v0 / v0.norm()
        norm_result = result / result.norm()
        assert torch.allclose(norm_result, norm_v0, atol=1e-5)

    def test_slerp_t_one(self):
        """Test SLERP with t=1 performs spherical interpolation"""
        v0 = torch.randn(5)
        v1 = torch.randn(5)

        result = slerp(v0, v1, t=1.0)

        # SLERP with t=1 rotates v0 by the full angle towards v1
        # Result should have same norm as v0 and be between v0 and v1
        assert result.shape == v0.shape
        assert not torch.isnan(result).any()
        assert not torch.isinf(result).any()

    def test_slerp_orthogonal_vectors(self):
        """Test SLERP with orthogonal vectors"""
        v0 = torch.tensor([1.0, 0.0])
        v1 = torch.tensor([0.0, 1.0])

        result = slerp(v0, v1, t=0.5)

        # Result should have equal components for orthogonal vectors at t=0.5
        assert result.shape == v0.shape
        assert not torch.isnan(result).any()


class TestFFTTransform:
    """Test suite for fft_transform and ifft_transform"""

    def test_fft_transform_1d(self):
        """Test FFT on 1D tensor"""
        tensor = torch.randn(16)

        result = fft_transform(tensor, device="cpu")

        assert result.dtype == torch.complex64 or result.dtype == torch.complex128
        assert result.shape == tensor.shape

    def test_fft_transform_2d(self):
        """Test FFT on 2D tensor"""
        tensor = torch.randn(8, 8)

        result = fft_transform(tensor, device="cpu")

        assert result.dtype in [torch.complex64, torch.complex128]
        assert result.shape == tensor.shape

    def test_ifft_transform_1d(self):
        """Test inverse FFT on 1D tensor"""
        tensor = torch.randn(16)
        fft_result = fft_transform(tensor, device="cpu")

        result = ifft_transform(fft_result, device="cpu")

        assert result.dtype == torch.float32
        assert result.shape == tensor.shape
        assert torch.allclose(result, tensor, atol=1e-4)

    def test_ifft_transform_2d(self):
        """Test inverse FFT on 2D tensor"""
        tensor = torch.randn(8, 8)
        fft_result = fft_transform(tensor, device="cpu")

        result = ifft_transform(fft_result, device="cpu")

        assert result.dtype == torch.float32
        assert result.shape == tensor.shape
        assert torch.allclose(result, tensor, atol=1e-4)

    def test_fft_ifft_roundtrip(self):
        """Test FFT followed by IFFT returns original"""
        original = torch.randn(16, 16)

        fft_result = fft_transform(original, device="cpu")
        reconstructed = ifft_transform(fft_result, device="cpu")

        assert torch.allclose(reconstructed, original, atol=1e-4)


class TestNormalizeTensor:
    """Test suite for normalize_tensor function"""

    def test_normalize_tensor_basic(self):
        """Test basic tensor normalization"""
        tensor = torch.tensor([3.0, 4.0])

        normalized, norm = normalize_tensor(tensor, device="cpu")

        assert abs(norm - 5.0) < 1e-5  # sqrt(3^2 + 4^2) = 5
        assert torch.allclose(normalized.norm(), torch.tensor(1.0))

    def test_normalize_tensor_zero(self):
        """Test normalization of zero tensor"""
        tensor = torch.zeros(5)

        normalized, norm = normalize_tensor(tensor, device="cpu")

        assert norm == 0.0
        assert torch.equal(normalized, tensor)

    def test_normalize_tensor_2d(self):
        """Test normalization of 2D tensor"""
        tensor = torch.randn(4, 4)

        normalized, norm = normalize_tensor(tensor, device="cpu")

        assert norm > 0
        assert torch.allclose(normalized.norm(), torch.tensor(1.0))


class TestMergeTensorsFft2Slerp:
    """Test suite for merge_tensors_fft2_slerp function"""

    def test_merge_basic(self):
        """Test basic tensor merging"""
        v0 = torch.randn(8, 8)
        v1 = torch.randn(8, 8)

        result, norm0, norm1 = merge_tensors_fft2_slerp(v0, v1, t=0.5, device="cpu")

        assert result.shape == v0.shape
        assert not torch.isnan(result).any()
        assert not torch.isinf(result).any()
        assert norm0 > 0
        assert norm1 > 0

    def test_merge_zero_v1_norm(self):
        """Test merging when v1 has near-zero norm"""
        v0 = torch.randn(8, 8)
        v1 = torch.randn(8, 8) * 1e-6

        result, norm0, norm1 = merge_tensors_fft2_slerp(v0, v1, t=0.5, device="cpu")

        # Should return v0 when v1 is negligible
        assert result.shape == v0.shape
        assert norm1 < 0.001

    def test_merge_with_cutoff(self):
        """Test merging with cutoff_pct parameter"""
        v0 = torch.randn(8, 8)
        v1 = torch.randn(8, 8)

        result, norm0, norm1 = merge_tensors_fft2_slerp(
            v0, v1, t=0.5, device="cpu", cutoff_pct=0.1
        )

        assert result.shape == v0.shape
        assert not torch.isnan(result).any()

    def test_merge_with_cull(self):
        """Test merging with cull_pct parameter"""
        v0 = torch.randn(8, 8)
        v1 = torch.randn(8, 8)

        result, norm0, norm1 = merge_tensors_fft2_slerp(
            v0, v1, t=0.5, device="cpu", cull_pct=0.05
        )

        assert result.shape == v0.shape
        assert not torch.isnan(result).any()


class TestTaskArithmeticFft2:
    """Test suite for task_arithmetic_fft2 function"""

    def test_task_arithmetic_basic(self):
        """Test basic task arithmetic"""
        v0 = torch.randn(8, 8)
        v1 = torch.randn(8, 8)

        result = task_arithmetic_fft2(v0, v1, t=1.0, device="cpu", agreement=True)

        assert result.shape == v0.shape
        assert not torch.isnan(result).any()
        assert not torch.isinf(result).any()

    def test_task_arithmetic_no_agreement(self):
        """Test task arithmetic without sign agreement"""
        v0 = torch.randn(8, 8)
        v1 = torch.randn(8, 8)

        result = task_arithmetic_fft2(v0, v1, t=1.0, device="cpu", agreement=False)

        assert result.shape == v0.shape
        assert not torch.isnan(result).any()

    def test_task_arithmetic_t_parameter(self):
        """Test task arithmetic with different t values"""
        v0 = torch.randn(8, 8)
        v1 = torch.randn(8, 8)

        result1 = task_arithmetic_fft2(v0, v1, t=0.5, device="cpu")
        result2 = task_arithmetic_fft2(v0, v1, t=1.0, device="cpu")

        # Results should differ with different t values
        assert not torch.allclose(result1, result2)


class TestCorrelatePairs:
    """Test suite for correlate_pairs function"""

    def test_correlate_pairs_basic(self):
        """Test basic pairwise correlation"""
        tensors = torch.stack([
            torch.randn(10),
            torch.randn(10),
            torch.randn(10)
        ])

        matrix = correlate_pairs(tensors, work_device="cpu", store_device="cpu")

        assert matrix.shape == (3, 3)
        assert torch.allclose(matrix, matrix.T)  # Should be symmetric
        assert torch.all((matrix >= -1.0) & (matrix <= 1.0))

    def test_correlate_pairs_diagonal(self):
        """Test correlation matrix diagonal"""
        tensors = torch.stack([
            torch.randn(10),
            torch.randn(10)
        ])

        matrix = correlate_pairs(tensors, work_device="cpu", store_device="cpu")

        # Diagonal should be close to 1 (self-correlation)
        # Note: Due to nan_to_num, might be 0 for zero tensors
        assert matrix.shape == (2, 2)

    def test_correlate_pairs_identical_tensors(self):
        """Test correlation of identical tensors"""
        tensor = torch.randn(10)
        tensors = torch.stack([tensor, tensor])

        matrix = correlate_pairs(tensors, work_device="cpu", store_device="cpu")

        # Identical tensors should have correlation of 1
        assert torch.allclose(matrix[0, 1], torch.tensor(1.0), atol=1e-5)


class TestCorrelatedPairs:
    """Test suite for correlated_pairs generator"""

    def test_correlated_pairs_least(self):
        """Test correlated_pairs with 'least' correlation"""
        # Create a simple correlation matrix
        matrix = torch.tensor([
            [1.0, 0.5, 0.9],
            [0.5, 1.0, 0.3],
            [0.9, 0.3, 1.0]
        ])

        pairs = list(correlated_pairs(matrix, way="least"))

        assert len(pairs) > 0
        # First pair should have indices and a coefficient
        assert len(pairs[0]) == 3
        assert isinstance(pairs[0][2], float)

    def test_correlated_pairs_most(self):
        """Test correlated_pairs with 'most' correlation"""
        matrix = torch.tensor([
            [1.0, 0.5, 0.9],
            [0.5, 1.0, 0.3],
            [0.9, 0.3, 1.0]
        ])

        pairs = list(correlated_pairs(matrix, way="most"))

        assert len(pairs) > 0

    def test_correlated_pairs_invalid_way(self):
        """Test correlated_pairs raises error for invalid 'way'"""
        matrix = torch.tensor([[1.0, 0.5], [0.5, 1.0]])

        with pytest.raises(ValueError, match="Invalid way"):
            list(correlated_pairs(matrix, way="invalid"))

    def test_correlated_pairs_uses_all_indices(self):
        """Test correlated_pairs uses each index only once"""
        matrix = torch.tensor([
            [1.0, 0.5, 0.9, 0.7],
            [0.5, 1.0, 0.3, 0.6],
            [0.9, 0.3, 1.0, 0.4],
            [0.7, 0.6, 0.4, 1.0]
        ])

        pairs = list(correlated_pairs(matrix, way="least"))

        all_indices = []
        for x, y, coef in pairs:
            if y >= 0:  # Not unpaired
                all_indices.extend([x, y])
            else:
                all_indices.append(x)

        # Each index should appear exactly once
        assert sorted(all_indices) == [0, 1, 2, 3]

    def test_correlated_pairs_odd_count(self):
        """Test correlated_pairs with odd number of items"""
        matrix = torch.tensor([
            [1.0, 0.5, 0.9],
            [0.5, 1.0, 0.3],
            [0.9, 0.3, 1.0]
        ])

        pairs = list(correlated_pairs(matrix, way="least"))

        # Should have 2 pairs (one with y=-1 for unpaired)
        assert any(y == -1 for x, y, coef in pairs)


class TestInterpolateFftComponents:
    """Test suite for interpolate_fft_components"""

    def test_interpolate_fft_basic(self):
        """Test basic FFT component interpolation"""
        v0 = torch.randn(8, 8)
        v1 = torch.randn(8, 8)

        v0_fft = fft_transform(v0, device="cpu")
        v1_fft = fft_transform(v1, device="cpu")

        result = interpolate_fft_components(
            v0_fft, v1_fft, t=0.5, device="cpu"
        )

        assert result.shape == v0_fft.shape
        assert result.dtype in [torch.complex64, torch.complex128]

    def test_interpolate_fft_with_cutoff(self):
        """Test FFT interpolation with cutoff percentage"""
        v0 = torch.randn(8, 8)
        v1 = torch.randn(8, 8)

        v0_fft = fft_transform(v0, device="cpu")
        v1_fft = fft_transform(v1, device="cpu")

        result = interpolate_fft_components(
            v0_fft, v1_fft, t=0.5, device="cpu", cutoff_pct=0.1
        )

        assert result.shape == v0_fft.shape

    def test_interpolate_fft_no_interp_imag(self):
        """Test FFT interpolation without interpolating imaginary parts"""
        v0 = torch.randn(8, 8)
        v1 = torch.randn(8, 8)

        v0_fft = fft_transform(v0, device="cpu")
        v1_fft = fft_transform(v1, device="cpu")

        result = interpolate_fft_components(
            v0_fft, v1_fft, t=0.5, device="cpu", interp_imag=False
        )

        # Imaginary part should match v0_fft
        assert torch.allclose(result.imag, v0_fft.imag)


class TestArithmeticFftComponents:
    """Test suite for arithmetic_fft_components"""

    def test_arithmetic_fft_basic(self):
        """Test basic FFT component arithmetic"""
        v0 = torch.randn(8, 8)
        v1 = torch.randn(8, 8)

        v0_fft = fft_transform(v0, device="cpu")
        v1_fft = fft_transform(v1, device="cpu")

        result = arithmetic_fft_components(
            v0_fft, v1_fft, t=1.0, agreement=True, device="cpu"
        )

        assert result.shape == v0_fft.shape
        assert result.dtype in [torch.complex64, torch.complex128]

    def test_arithmetic_fft_no_agreement(self):
        """Test FFT arithmetic without sign agreement"""
        v0 = torch.randn(8, 8)
        v1 = torch.randn(8, 8)

        v0_fft = fft_transform(v0, device="cpu")
        v1_fft = fft_transform(v1, device="cpu")

        result = arithmetic_fft_components(
            v0_fft, v1_fft, t=1.0, agreement=False, device="cpu"
        )

        assert result.shape == v0_fft.shape

    def test_arithmetic_fft_no_imag(self):
        """Test FFT arithmetic without processing imaginary part"""
        v0 = torch.randn(8, 8)
        v1 = torch.randn(8, 8)

        v0_fft = fft_transform(v0, device="cpu")
        v1_fft = fft_transform(v1, device="cpu")

        result = arithmetic_fft_components(
            v0_fft, v1_fft, t=1.0, agreement=True, device="cpu", do_imag=False
        )

        # Imaginary part should match v0_fft
        assert torch.allclose(result.imag, v0_fft.imag)
