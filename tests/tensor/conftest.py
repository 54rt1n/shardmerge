# tests/tensor/conftest.py
import pytest
import torch


@pytest.fixture
def fft_compatible_tensor():
    """Create a tensor compatible with FFT operations"""
    return torch.randn(16, 16)


@pytest.fixture
def small_tensor_for_fft():
    """Create a small tensor for FFT testing"""
    return torch.randn(4, 4)


@pytest.fixture
def tensor_pair_for_merge():
    """Create a pair of tensors for merge testing"""
    t1 = torch.randn(8, 8)
    t2 = torch.randn(8, 8)
    return t1, t2
