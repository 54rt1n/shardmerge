# tests/merge/test_taskaddition.py
import pytest
import torch
from pathlib import Path
from unittest.mock import Mock, AsyncMock, patch

from shard.merge.taskaddition import TaskAdditionMerge
from shard.config import MergeConfig, MergeModel
from shard.writer import ShardLayer


class TestTaskAdditionMerge:
    """Test suite for TaskAdditionMerge"""

    def test_task_addition_merge_initialization(self, merge_config_for_testing):
        """Test TaskAdditionMerge initialization"""
        merger = TaskAdditionMerge(config=merge_config_for_testing)

        assert merger.config == merge_config_for_testing
        assert merger.index_manager is not None

    def test_task_addition_merge_with_index_manager(self, merge_config_for_testing, mock_index_manager):
        """Test TaskAdditionMerge with custom index manager"""
        merger = TaskAdditionMerge(
            config=merge_config_for_testing,
            index_manager=mock_index_manager
        )

        assert merger.config == merge_config_for_testing
        assert merger.index_manager == mock_index_manager

    def test_get_readme(self, merge_config_for_testing):
        """Test get_readme generates README"""
        merger = TaskAdditionMerge(config=merge_config_for_testing)

        readme = merger.get_readme()

        assert isinstance(readme, str)
        assert "Merged Model" in readme
        assert "test/base" in readme
        assert "test/model1" in readme
        assert "test/model2" in readme
        assert "sign agreement" in readme

    async def test_merge_layer_basic_sign_agreement(self, merge_config_for_testing, mock_index_manager):
        """Test _merge_layer with sign agreement"""
        merger = TaskAdditionMerge(
            config=merge_config_for_testing,
            index_manager=mock_index_manager
        )

        shard_layer = ShardLayer(
            layer_order_idx=1,
            shard_name="model-00001.safetensors",
            layer_name="model.layers.0.weight",
            written=False
        )

        # Create test tensors where both models agree on sign
        base_tensor = torch.ones(4, 4) * 1.0
        model1_tensor = torch.ones(4, 4) * 2.0  # delta = +1.0 (positive)
        model2_tensor = torch.ones(4, 4) * 3.0  # delta = +2.0 (positive)

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

            # Both deltas are positive, so signs agree
            # Result should be sum of deltas: 1.0 + 2.0 = 3.0
            assert isinstance(result, torch.Tensor)
            assert result.shape == (4, 4)
            assert torch.allclose(result, torch.ones(4, 4) * 3.0)

    async def test_merge_layer_sign_disagreement(self, merge_config_for_testing, mock_index_manager):
        """Test _merge_layer with sign disagreement"""
        merger = TaskAdditionMerge(
            config=merge_config_for_testing,
            index_manager=mock_index_manager
        )

        shard_layer = ShardLayer(
            layer_order_idx=1,
            shard_name="model-00001.safetensors",
            layer_name="model.layers.0.weight",
            written=False
        )

        # Create test tensors where models disagree on sign
        # Using a 2x2 tensor for clarity
        base_tensor = torch.tensor([[5.0, 5.0], [5.0, 5.0]])
        model1_tensor = torch.tensor([[7.0, 3.0], [6.0, 4.0]])  # deltas: [+2, -2], [+1, -1]
        model2_tensor = torch.tensor([[3.0, 7.0], [4.0, 6.0]])  # deltas: [-2, +2], [-1, +1]

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

            # Where signs disagree, values should be zeroed out
            # All positions have disagreeing signs, so result should be zeros
            assert isinstance(result, torch.Tensor)
            assert result.shape == (2, 2)
            assert torch.allclose(result, torch.zeros(2, 2))

    async def test_merge_layer_mixed_signs(self, merge_config_for_testing, mock_index_manager):
        """Test _merge_layer with mixed sign agreement"""
        merger = TaskAdditionMerge(
            config=merge_config_for_testing,
            index_manager=mock_index_manager
        )

        shard_layer = ShardLayer(
            layer_order_idx=1,
            shard_name="model-00001.safetensors",
            layer_name="model.layers.0.weight",
            written=False
        )

        # Create tensors with some agreeing and some disagreeing signs
        base_tensor = torch.tensor([[1.0, 1.0], [1.0, 1.0]])
        model1_tensor = torch.tensor([[2.0, 2.0], [0.5, 2.0]])  # deltas: [+1, +1], [-0.5, +1]
        model2_tensor = torch.tensor([[3.0, 0.5], [2.0, 0.5]])  # deltas: [+2, -0.5], [+1, -0.5]

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

            assert isinstance(result, torch.Tensor)
            assert result.shape == (2, 2)

            # Position [0,0]: both positive (+1, +2) -> sum = 3.0
            # Position [0,1]: disagree (+1, -0.5) -> 0
            # Position [1,0]: disagree (-0.5, +1) -> 0
            # Position [1,1]: disagree (+1, -0.5) -> 0
            expected = torch.tensor([[3.0, 0.0], [0.0, 0.0]])
            assert torch.allclose(result, expected)

    async def test_merge_layer_zero_deltas(self, merge_config_for_testing, mock_index_manager):
        """Test _merge_layer with zero deltas"""
        merger = TaskAdditionMerge(
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

        mock_base_promise = AsyncMock()
        mock_base_promise.get = AsyncMock(return_value=base_tensor)

        mock_model_promise = AsyncMock()
        mock_model_promise.get = AsyncMock(return_value=base_tensor.clone())

        call_count = 0

        def get_tensor_side_effect(model_uri, *args, **kwargs):
            nonlocal call_count
            call_count += 1
            if "base" in model_uri or call_count == 1:
                return mock_base_promise
            else:
                return mock_model_promise

        with patch.object(mock_index_manager, "get_tensor", side_effect=get_tensor_side_effect):
            result = await merger._merge_layer(shard_layer, device="cpu")

            # All deltas are zero
            assert isinstance(result, torch.Tensor)
            assert torch.allclose(result, torch.zeros(4, 4))

    async def test_merge_layer_all_negative_deltas(self, merge_config_for_testing, mock_index_manager):
        """Test _merge_layer where all deltas are negative"""
        merger = TaskAdditionMerge(
            config=merge_config_for_testing,
            index_manager=mock_index_manager
        )

        shard_layer = ShardLayer(
            layer_order_idx=1,
            shard_name="model-00001.safetensors",
            layer_name="model.layers.0.weight",
            written=False
        )

        # All deltas are negative and agree
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

            # Both negative, signs agree, sum = -2 + -1 = -3
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

        merger = TaskAdditionMerge(
            config=config,
            index_manager=mock_index_manager
        )

        shard_layer = ShardLayer(
            layer_order_idx=1,
            shard_name="model-00001.safetensors",
            layer_name="model.layers.0.weight",
            written=False
        )

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

            # Single model, just returns the delta
            assert isinstance(result, torch.Tensor)
            assert torch.allclose(result, torch.ones(4, 4) * 3.0)
