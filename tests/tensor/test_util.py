# tests/tensor/test_util.py
import pytest
import torch
from unittest.mock import patch
from io import StringIO

from shard.tensor.util import cuda_memory_profiler, get_device


class TestCudaMemoryProfiler:
    """Test suite for cuda_memory_profiler context manager"""

    def test_cuda_memory_profiler_disabled(self):
        """Test cuda_memory_profiler with display=False does nothing"""
        with cuda_memory_profiler(display=False):
            x = torch.randn(10, 10)

        # Should complete without error and without output

    @patch("torch.cuda.is_available", return_value=False)
    def test_cuda_memory_profiler_no_cuda(self, mock_cuda_available, capsys):
        """Test cuda_memory_profiler when CUDA not available"""
        with cuda_memory_profiler(display=True, title="Test"):
            x = torch.randn(10, 10)

        captured = capsys.readouterr()
        assert "Test" in captured.out
        assert "System Memory:" in captured.out
        assert "RAM usage" in captured.out

    @patch("torch.cuda.synchronize")
    @patch("torch.cuda.reset_peak_memory_stats")
    @patch("torch.cuda.max_memory_allocated")
    @patch("torch.cuda.memory_allocated")
    @patch("torch.cuda.is_available", return_value=True)
    def test_cuda_memory_profiler_with_cuda(
        self,
        mock_is_available,
        mock_mem_alloc,
        mock_max_mem,
        mock_reset,
        mock_sync,
        capsys
    ):
        """Test cuda_memory_profiler with CUDA available"""
        mock_mem_alloc.return_value = 1024 * 1024 * 10  # 10 MB
        mock_max_mem.return_value = 1024 * 1024 * 15  # 15 MB

        with cuda_memory_profiler(devices=["cuda:0"], display=True, title="CUDA Test"):
            x = torch.randn(10, 10)

        captured = capsys.readouterr()
        assert "CUDA Test" in captured.out
        assert "Device: cuda:0" in captured.out
        assert "Peak memory usage:" in captured.out

    @patch("torch.cuda.synchronize")
    @patch("torch.cuda.reset_peak_memory_stats")
    @patch("torch.cuda.max_memory_allocated")
    @patch("torch.cuda.memory_allocated")
    @patch("torch.cuda.is_available", return_value=True)
    def test_cuda_memory_profiler_multiple_devices(
        self,
        mock_is_available,
        mock_mem_alloc,
        mock_max_mem,
        mock_reset,
        mock_sync,
        capsys
    ):
        """Test cuda_memory_profiler with multiple devices"""
        mock_mem_alloc.return_value = 1024 * 1024 * 10
        mock_max_mem.return_value = 1024 * 1024 * 15

        with cuda_memory_profiler(devices=["cuda:0", "cuda:1"], display=True):
            x = torch.randn(10, 10)

        captured = capsys.readouterr()
        assert "Device: cuda:0" in captured.out
        assert "Device: cuda:1" in captured.out

    def test_cuda_memory_profiler_custom_title(self, capsys):
        """Test cuda_memory_profiler with custom title"""
        with cuda_memory_profiler(display=True, title="Custom Title"):
            x = torch.randn(10, 10)

        captured = capsys.readouterr()
        assert "Custom Title" in captured.out

    @patch("torch.cuda.is_available", return_value=False)
    def test_cuda_memory_profiler_shows_ram_usage(self, mock_cuda_available, capsys):
        """Test cuda_memory_profiler displays RAM usage"""
        with cuda_memory_profiler(display=True):
            # Allocate some memory
            data = [torch.randn(100, 100) for _ in range(10)]

        captured = capsys.readouterr()
        assert "Start RAM usage:" in captured.out
        assert "End RAM usage:" in captured.out
        assert "Net RAM change:" in captured.out
        assert "MB" in captured.out


class TestGetDevice:
    """Test suite for get_device function"""

    @patch("torch.cuda.is_available", return_value=True)
    def test_get_device_cuda_available(self, mock_cuda):
        """Test get_device returns cuda when available"""
        device = get_device()

        assert device.type == "cuda"

    @patch("torch.cuda.is_available", return_value=False)
    @patch("torch.backends.mps.is_available", return_value=True)
    def test_get_device_mps_available(self, mock_mps, mock_cuda):
        """Test get_device returns mps when CUDA not available but MPS is"""
        device = get_device()

        assert device.type == "mps"

    @patch("torch.cuda.is_available", return_value=False)
    @patch("torch.backends.mps.is_available", return_value=False)
    def test_get_device_cpu_fallback(self, mock_mps, mock_cuda):
        """Test get_device returns cpu when neither CUDA nor MPS available"""
        device = get_device()

        assert device.type == "cpu"

    @patch("torch.cuda.is_available", return_value=True)
    @patch("torch.backends.mps.is_available", return_value=True)
    def test_get_device_cuda_priority(self, mock_mps, mock_cuda):
        """Test get_device prioritizes CUDA over MPS"""
        device = get_device()

        assert device.type == "cuda"
