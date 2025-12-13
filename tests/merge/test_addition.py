# tests/merge/test_addition.py
import pytest
import torch
from pathlib import Path
from unittest.mock import Mock, AsyncMock, patch

from shard.merge.addition import AdditionMerge
from shard.config import MergeConfig, MergeModel
from shard.writer import ShardLayer


class TestAdditionMerge:
    """Test suite for AdditionMerge"""

    def test_addition_merge_initialization(self, merge_config_for_testing):
        """Test AdditionMerge initialization"""
        merger = AdditionMerge(config=merge_config_for_testing)

        assert merger.config == merge_config_for_testing
        assert merger.index_manager is not None

    def test_addition_merge_with_index_manager(self, merge_config_for_testing, mock_index_manager):
        """Test AdditionMerge with custom index manager"""
        merger = AdditionMerge(
            config=merge_config_for_testing,
            index_manager=mock_index_manager
        )

        assert merger.config == merge_config_for_testing
        assert merger.index_manager == mock_index_manager

    def test_get_readme(self, merge_config_for_testing):
        """Test get_readme generates README"""
        merger = AdditionMerge(config=merge_config_for_testing)

        readme = merger.get_readme()

        assert isinstance(readme, str)
        assert "Merged Model" in readme
        assert "test/base" in readme
        assert "test/model1" in readme
        assert "test/model2" in readme
        assert "delta weights" in readme

    async def test_merge_layer_basic(self, merge_config_for_testing, mock_index_manager):
        """Test _merge_layer performs basic addition merge"""
        merger = AdditionMerge(
            config=merge_config_for_testing,
            index_manager=mock_index_manager
        )

        shard_layer = ShardLayer(
            layer_order_idx=1,
            shard_name="model-00001.safetensors",
            layer_name="model.layers.0.weight",
            written=False
        )

        # Create test tensors
        base_tensor = torch.ones(4, 4) * 1.0
        model1_tensor = torch.ones(4, 4) * 2.0  # delta = +1.0
        model2_tensor = torch.ones(4, 4) * 3.0  # delta = +2.0

        mock_base_promise = AsyncMock()
        mock_base_promise.get = AsyncMock(return_value=base_tensor)

        mock_model1_promise = AsyncMock()
        mock_model1_promise.get = AsyncMock(return_value=model1_tensor)

        mock_model2_promise = AsyncMock()
        mock_model2_promise.get = AsyncMock(return_value=model2_tensor)

        call_count = 0

        def get_tensor_side_effect(model_uri, *args, **kwargs):
            nonlocal call_count
            call_count += 1
            if "base" in model_uri or call_count == 1:
                return mock_base_promise
            elif "model1" in model_uri or call_count == 2:
                return mock_model1_promise
            else:
                return mock_model2_promise

        with patch.object(mock_index_manager, "get_tensor", side_effect=get_tensor_side_effect):
            result = await merger._merge_layer(shard_layer, device="cpu")

            # Result should be sum of deltas: delta1 (1.0) + delta2 (2.0) = 3.0
            assert isinstance(result, torch.Tensor)
            assert result.shape == (4, 4)
            # Expected: delta1 + delta2 = (2-1) + (3-1) = 1 + 2 = 3
            assert torch.allclose(result, torch.ones(4, 4) * 3.0)

    async def test_merge_layer_zero_deltas(self, merge_config_for_testing, mock_index_manager):
        """Test _merge_layer with zero deltas"""
        merger = AdditionMerge(
            config=merge_config_for_testing,
            index_manager=mock_index_manager
        )

        shard_layer = ShardLayer(
            layer_order_idx=1,
            shard_name="model-00001.safetensors",
            layer_name="model.layers.0.weight",
            written=False
        )

        # All tensors are the same - zero deltas
        base_tensor = torch.ones(4, 4) * 5.0
        model1_tensor = torch.ones(4, 4) * 5.0
        model2_tensor = torch.ones(4, 4) * 5.0

        mock_base_promise = AsyncMock()
        mock_base_promise.get = AsyncMock(return_value=base_tensor)

        mock_model1_promise = AsyncMock()
        mock_model1_promise.get = AsyncMock(return_value=model1_tensor)

        mock_model2_promise = AsyncMock()
        mock_model2_promise.get = AsyncMock(return_value=model2_tensor)

        call_count = 0

        def get_tensor_side_effect(model_uri, *args, **kwargs):
            nonlocal call_count
            call_count += 1
            if "base" in model_uri or call_count == 1:
                return mock_base_promise
            elif "model1" in model_uri or call_count == 2:
                return mock_model1_promise
            else:
                return mock_model2_promise

        with patch.object(mock_index_manager, "get_tensor", side_effect=get_tensor_side_effect):
            result = await merger._merge_layer(shard_layer, device="cpu")

            # Result should be all zeros (no deltas)
            assert isinstance(result, torch.Tensor)
            assert torch.allclose(result, torch.zeros(4, 4))

    async def test_merge_layer_negative_deltas(self, merge_config_for_testing, mock_index_manager):
        """Test _merge_layer with negative deltas"""
        merger = AdditionMerge(
            config=merge_config_for_testing,
            index_manager=mock_index_manager
        )

        shard_layer = ShardLayer(
            layer_order_idx=1,
            shard_name="model-00001.safetensors",
            layer_name="model.layers.0.weight",
            written=False
        )

        # Create tensors with negative deltas
        base_tensor = torch.ones(4, 4) * 5.0
        model1_tensor = torch.ones(4, 4) * 3.0  # delta = -2.0
        model2_tensor = torch.ones(4, 4) * 4.0  # delta = -1.0

        mock_base_promise = AsyncMock()
        mock_base_promise.get = AsyncMock(return_value=base_tensor)

        mock_model1_promise = AsyncMock()
        mock_model1_promise.get = AsyncMock(return_value=model1_tensor)

        mock_model2_promise = AsyncMock()
        mock_model2_promise.get = AsyncMock(return_value=model2_tensor)

        call_count = 0

        def get_tensor_side_effect(model_uri, *args, **kwargs):
            nonlocal call_count
            call_count += 1
            if "base" in model_uri or call_count == 1:
                return mock_base_promise
            elif "model1" in model_uri or call_count == 2:
                return mock_model1_promise
            else:
                return mock_model2_promise

        with patch.object(mock_index_manager, "get_tensor", side_effect=get_tensor_side_effect):
            result = await merger._merge_layer(shard_layer, device="cpu")

            # Result should be sum of deltas: delta1 (-2.0) + delta2 (-1.0) = -3.0
            assert isinstance(result, torch.Tensor)
            assert torch.allclose(result, torch.ones(4, 4) * -3.0)

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

        merger = AdditionMerge(
            config=config,
            index_manager=mock_index_manager
        )

        shard_layer = ShardLayer(
            layer_order_idx=1,
            shard_name="model-00001.safetensors",
            layer_name="model.layers.0.weight",
            written=False
        )

        # Create test tensors
        base_tensor = torch.ones(4, 4) * 2.0
        model_tensor = torch.ones(4, 4) * 5.0  # delta = 3.0

        mock_base_promise = AsyncMock()
        mock_base_promise.get = AsyncMock(return_value=base_tensor)

        mock_model_promise = AsyncMock()
        mock_model_promise.get = AsyncMock(return_value=model_tensor)

        def get_tensor_side_effect(model_uri, *args, **kwargs):
            if "base" in model_uri:
                return mock_base_promise
            return mock_model_promise

        with patch.object(mock_index_manager, "get_tensor", side_effect=get_tensor_side_effect):
            result = await merger._merge_layer(shard_layer, device="cpu")

            # Result should just be the single delta: 3.0
            assert isinstance(result, torch.Tensor)
            assert torch.allclose(result, torch.ones(4, 4) * 3.0)
