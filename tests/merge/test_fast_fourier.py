# tests/merge/test_fast_fourier.py
import pytest
import torch
from pathlib import Path
from unittest.mock import Mock, AsyncMock, patch

from shard.merge.fast_fourier import (
    task_arithmetic,
    name_hash,
    clamp,
    TensorDiskCache,
    FourierMerge
)
from shard.config import MergeConfig, MergeModel
from shard.writer import ShardLayer
from shard.constants import INPUT_LAYER, OUTPUT_LAYER


class TestTaskArithmetic:
    """Test suite for task_arithmetic function"""

    def test_task_arithmetic_sign_agreement(self):
        """Test task arithmetic with sign agreement"""
        t0 = torch.tensor([1.0, -2.0, 3.0])
        t1 = torch.tensor([2.0, -1.0, 4.0])

        result = task_arithmetic(t0, t1)

        # Where signs agree, should add
        assert result[0] == 3.0  # 1 + 2
        assert result[1] == -3.0  # -2 + -1
        assert result[2] == 7.0  # 3 + 4

    def test_task_arithmetic_sign_disagreement(self):
        """Test task arithmetic with sign disagreement"""
        t0 = torch.tensor([1.0, -2.0, 3.0])
        t1 = torch.tensor([-2.0, 3.0, -4.0])

        result = task_arithmetic(t0, t1)

        # Where signs disagree, should keep t0
        assert result[0] == 1.0
        assert result[1] == -2.0
        assert result[2] == 3.0

    def test_task_arithmetic_zero_values(self):
        """Test task arithmetic with zero values"""
        t0 = torch.tensor([0.0, 1.0, -1.0])
        t1 = torch.tensor([0.0, 0.0, 0.0])

        result = task_arithmetic(t0, t1)

        # Zero values should agree with everything
        assert result[0] == 0.0
        assert result[1] == 1.0
        assert result[2] == -1.0


class TestNameHash:
    """Test suite for name_hash function"""

    def test_name_hash_basic(self):
        """Test basic name hashing"""
        result = name_hash("model_layer_weight")

        assert "::" in result
        assert result.startswith("mode_laye_weig::")
        assert len(result.split("::")[1]) == 8  # 8 char hash

    def test_name_hash_consistent(self):
        """Test name_hash is consistent"""
        name = "test_name_foo_bar"

        hash1 = name_hash(name)
        hash2 = name_hash(name)

        assert hash1 == hash2

    def test_name_hash_different_names(self):
        """Test different names produce different hashes"""
        hash1 = name_hash("name_one")
        hash2 = name_hash("name_two")

        assert hash1 != hash2

    def test_name_hash_truncates_parts(self):
        """Test name_hash truncates long parts"""
        result = name_hash("verylongname_anotherverylongname")

        # Should truncate to first 4 chars of each part
        assert result.startswith("very_anot::")


class TestClamp:
    """Test suite for clamp function"""

    def test_clamp_within_range(self):
        """Test clamp keeps value within range"""
        result = clamp(5.0, 0.0, 10.0)
        assert result == 5.0

    def test_clamp_below_min(self):
        """Test clamp enforces minimum"""
        result = clamp(-5.0, 0.0, 10.0)
        assert result == 0.0

    def test_clamp_above_max(self):
        """Test clamp enforces maximum"""
        result = clamp(15.0, 0.0, 10.0)
        assert result == 10.0

    def test_clamp_at_boundaries(self):
        """Test clamp at exact boundaries"""
        assert clamp(0.0, 0.0, 10.0) == 0.0
        assert clamp(10.0, 0.0, 10.0) == 10.0


class TestTensorDiskCache:
    """Test suite for TensorDiskCache"""

    def test_cache_initialization(self, tmp_path):
        """Test TensorDiskCache initialization"""
        cache = TensorDiskCache(tmp_path / "cache")

        assert cache.cache_path == tmp_path / "cache"
        assert cache.cache_path.exists()

    def test_cache_set_and_get(self, tmp_path):
        """Test setting and getting cached tensor"""
        cache = TensorDiskCache(tmp_path / "cache")

        test_tensor = torch.randn(5, 5)

        cache.set("test/model", "layer.weight", "cpu", test_tensor)

        retrieved = cache.get("test/model", "layer.weight", "cpu")

        assert retrieved is not None
        assert torch.allclose(retrieved, test_tensor)

    def test_cache_get_nonexistent(self, tmp_path):
        """Test getting nonexistent cache entry"""
        cache = TensorDiskCache(tmp_path / "cache")

        result = cache.get("test/model", "nonexistent.layer", "cpu")

        assert result is None

    def test_cache_remove(self, tmp_path):
        """Test removing cached tensor"""
        cache = TensorDiskCache(tmp_path / "cache")

        test_tensor = torch.randn(5, 5)
        cache.set("test/model", "layer.weight", "cpu", test_tensor)

        # Verify it exists
        assert cache.get("test/model", "layer.weight", "cpu") is not None

        # Remove it
        cache.remove("test/model", "layer.weight", "cpu")

        # Verify it's gone
        assert cache.get("test/model", "layer.weight", "cpu") is None

    def test_cache_clear(self, tmp_path):
        """Test clearing all cache entries"""
        cache = TensorDiskCache(tmp_path / "cache")

        # Add multiple entries
        cache.set("test/model1", "layer1.weight", "cpu", torch.randn(3, 3))
        cache.set("test/model2", "layer2.weight", "cpu", torch.randn(3, 3))

        # Clear cache
        cache.clear()

        # Verify all entries are gone
        assert cache.get("test/model1", "layer1.weight", "cpu") is None
        assert cache.get("test/model2", "layer2.weight", "cpu") is None

    def test_cache_handles_slashes_in_model_name(self, tmp_path):
        """Test cache handles model names with slashes"""
        cache = TensorDiskCache(tmp_path / "cache")

        test_tensor = torch.randn(3, 3)
        # Model name with slash
        cache.set("org/model-name", "layer.weight", "cpu", test_tensor)

        # Should convert slashes to dashes
        retrieved = cache.get("org/model-name", "layer.weight", "cpu")

        assert retrieved is not None
        assert torch.allclose(retrieved, test_tensor)


class TestFourierMerge:
    """Test suite for FourierMerge"""

    def test_fourier_merge_initialization(self, merge_config_for_testing):
        """Test FourierMerge initialization"""
        merger = FourierMerge(config=merge_config_for_testing)

        assert merger.config == merge_config_for_testing
        assert merger.task_add_models == []
        assert merger.target_norm_offset == 1e-10
        assert merger.cull_start_pct == 0.20
        assert isinstance(merger.cache, TensorDiskCache)

    def test_fourier_merge_custom_parameters(self, merge_config_for_testing):
        """Test FourierMerge with custom parameters"""
        merger = FourierMerge(
            config=merge_config_for_testing,
            task_add_models=["test/model3"],
            target_norm_offset=1e-8,
            cull_start_pct=0.15
        )

        assert merger.task_add_models == ["test/model3"]
        assert merger.target_norm_offset == 1e-8
        assert merger.cull_start_pct == 0.15

    def test_get_readme(self, merge_config_for_testing):
        """Test get_readme generates README"""
        merger = FourierMerge(config=merge_config_for_testing)

        readme = merger.get_readme()

        assert isinstance(readme, str)
        assert "SLERP-FFT" in readme
        assert "test/base" in readme

    async def test_merge_layer_input_layer(self, merge_config_for_testing, mock_index_manager):
        """Test _merge_layer handles input layer"""
        config = merge_config_for_testing
        config.finetune_merge[0].is_input = True

        merger = FourierMerge(
            config=config,
            index_manager=mock_index_manager
        )

        shard_layer = ShardLayer(
            layer_order_idx=0,
            shard_name="model-00001.safetensors",
            layer_name="model.embed_tokens.weight",
            written=False
        )

        # Mock tensor retrieval
        test_tensor = torch.randn(3, 3)
        mock_promise = AsyncMock()
        mock_promise.get = AsyncMock(return_value=test_tensor)

        with patch.object(mock_index_manager, "get_tensor", return_value=mock_promise):
            result = await merger._merge_layer(shard_layer, device="cpu")

            assert torch.equal(result, test_tensor)

    async def test_merge_layer_output_layer(self, merge_config_for_testing, mock_index_manager):
        """Test _merge_layer handles output layer"""
        config = merge_config_for_testing
        config.finetune_merge[0].is_output = True

        merger = FourierMerge(
            config=config,
            index_manager=mock_index_manager
        )

        shard_layer = ShardLayer(
            layer_order_idx=100,
            shard_name="model-00002.safetensors",
            layer_name="lm_head.weight",
            written=False
        )

        # Mock tensor retrieval
        test_tensor = torch.randn(3, 3)
        mock_promise = AsyncMock()
        mock_promise.get = AsyncMock(return_value=test_tensor)

        with patch.object(mock_index_manager, "get_tensor", return_value=mock_promise):
            result = await merger._merge_layer(shard_layer, device="cpu")

            assert torch.equal(result, test_tensor)

    async def test_merge_layer_regular_layer(self, merge_config_for_testing, mock_index_manager):
        """Test _merge_layer handles regular layer merging"""
        merger = FourierMerge(
            config=merge_config_for_testing,
            index_manager=mock_index_manager
        )

        shard_layer = ShardLayer(
            layer_order_idx=1,
            shard_name="model-00001.safetensors",
            layer_name="model.layers.0.weight",
            written=False
        )

        # Mock tensors
        base_tensor = torch.randn(8, 8)
        model_tensor = torch.randn(8, 8)

        mock_base_promise = AsyncMock()
        mock_base_promise.get = AsyncMock(return_value=base_tensor)

        mock_model_promise = AsyncMock()
        mock_model_promise.get = AsyncMock(return_value=model_tensor)

        def get_tensor_side_effect(model_uri, *args, **kwargs):
            if "base" in model_uri:
                return mock_base_promise
            return mock_model_promise

        with patch.object(mock_index_manager, "get_tensor", side_effect=get_tensor_side_effect):
            with patch.object(mock_index_manager, "preload_tensor", new_callable=AsyncMock):
                result = await merger._merge_layer(shard_layer, device="cpu")

                assert isinstance(result, torch.Tensor)
                assert result.shape == (8, 8)
                assert result.dtype == torch.bfloat16

    async def test_merge_layer_single_model(self, tmp_path, mock_index_manager):
        """Test _merge_layer with single model"""
        config = MergeConfig(
            finetune_merge=[
                MergeModel(model="test/model1", base="test/base", alpha=0.5)
            ],
            output_base_model="test/base",
            output_dir=str(tmp_path / "output"),
            device="cpu",
            cache_dir=str(tmp_path / "cache"),
            storage_dir=str(tmp_path / "storage")
        )

        merger = FourierMerge(
            config=config,
            index_manager=mock_index_manager
        )

        shard_layer = ShardLayer(
            layer_order_idx=1,
            shard_name="model-00001.safetensors",
            layer_name="model.layers.0.weight",
            written=False
        )

        # Mock tensors
        base_tensor = torch.randn(8, 8)
        model_tensor = torch.randn(8, 8)

        mock_base_promise = AsyncMock()
        mock_base_promise.get = AsyncMock(return_value=base_tensor)

        mock_model_promise = AsyncMock()
        mock_model_promise.get = AsyncMock(return_value=model_tensor)

        def get_tensor_side_effect(model_uri, *args, **kwargs):
            if "base" in model_uri:
                return mock_base_promise
            return mock_model_promise

        with patch.object(mock_index_manager, "get_tensor", side_effect=get_tensor_side_effect):
            with patch.object(mock_index_manager, "preload_tensor", new_callable=AsyncMock):
                result = await merger._merge_layer(shard_layer, device="cpu")

                assert isinstance(result, torch.Tensor)
                # Result should not have NaN or Inf
                assert not torch.isnan(result).any()
                assert not torch.isinf(result).any()
