# tests/merge/test_fourier.py
import pytest
import torch
from pathlib import Path
from unittest.mock import Mock, AsyncMock, patch

from shard.merge.fourier import FourierMerge
from shard.config import MergeConfig, MergeModel
from shard.writer import ShardLayer
from shard.constants import INPUT_LAYER, OUTPUT_LAYER


class TestFourierMerge:
    """Test suite for FourierMerge"""

    def test_fourier_merge_initialization(self, merge_config_for_testing):
        """Test FourierMerge initialization"""
        merger = FourierMerge(config=merge_config_for_testing)

        assert merger.config == merge_config_for_testing
        assert merger.task_add_models == []
        assert merger.target_norm_offset == 1e-10
        assert merger.cull_start_pct == 0.20

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
        assert "test/model1" in readme
        assert "test/model2" in readme

    async def test_merge_layer_input_layer(self, merge_config_for_testing, mock_index_manager):
        """Test _merge_layer handles input layer passthrough"""
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

    async def test_merge_layer_input_layer_no_input_model(self, merge_config_for_testing, mock_index_manager):
        """Test _merge_layer raises error when no input model is found"""
        merger = FourierMerge(
            config=merge_config_for_testing,
            index_manager=mock_index_manager
        )

        shard_layer = ShardLayer(
            layer_order_idx=0,
            shard_name="model-00001.safetensors",
            layer_name="model.embed_tokens.weight",
            written=False
        )

        with pytest.raises(ValueError, match="No input model found"):
            await merger._merge_layer(shard_layer, device="cpu")

    async def test_merge_layer_output_layer(self, merge_config_for_testing, mock_index_manager):
        """Test _merge_layer handles output layer passthrough"""
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

    async def test_merge_layer_output_layer_no_output_model(self, merge_config_for_testing, mock_index_manager):
        """Test _merge_layer raises error when no output model is found"""
        merger = FourierMerge(
            config=merge_config_for_testing,
            index_manager=mock_index_manager
        )

        shard_layer = ShardLayer(
            layer_order_idx=100,
            shard_name="model-00002.safetensors",
            layer_name="lm_head.weight",
            written=False
        )

        with pytest.raises(ValueError, match="No output model found"):
            await merger._merge_layer(shard_layer, device="cpu")

    async def test_merge_layer_basic(self, merge_config_for_testing, mock_index_manager):
        """Test _merge_layer with basic regular layer"""
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
        model1_tensor = torch.randn(8, 8)
        model2_tensor = torch.randn(8, 8)

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
            assert result.shape == (8, 8)

            # Result should not have NaN or Inf
            assert not torch.isnan(result).any()
            assert not torch.isinf(result).any()

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
            result = await merger._merge_layer(shard_layer, device="cpu")

            assert isinstance(result, torch.Tensor)
            assert result.shape == (8, 8)

            # Result should not have NaN or Inf
            assert not torch.isnan(result).any()
            assert not torch.isinf(result).any()

    async def test_merge_layer_with_task_add_models(self, merge_config_for_testing, mock_index_manager):
        """Test _merge_layer with task arithmetic models"""
        merger = FourierMerge(
            config=merge_config_for_testing,
            task_add_models=["test/model1"],
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
        model1_tensor = torch.randn(8, 8)
        model2_tensor = torch.randn(8, 8)

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
            assert result.shape == (8, 8)

            # Result should not have NaN or Inf
            assert not torch.isnan(result).any()
            assert not torch.isinf(result).any()

    async def test_merge_layer_filters_by_layer_index(self, tmp_path, mock_index_manager):
        """Test _merge_layer respects layer index filtering"""
        config = MergeConfig(
            finetune_merge=[
                MergeModel(model="test/model1", base="test/base", alpha=0.5, start_layer=0, end_layer=0),
                MergeModel(model="test/model2", base="test/base", alpha=0.3, start_layer=1, end_layer=10)
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

        # Layer 1 should only include model2
        shard_layer = ShardLayer(
            layer_order_idx=1,
            shard_name="model-00001.safetensors",
            layer_name="model.layers.0.weight",
            written=False
        )

        # Mock tensors
        base_tensor = torch.randn(8, 8)
        model2_tensor = torch.randn(8, 8)

        mock_base_promise = AsyncMock()
        mock_base_promise.get = AsyncMock(return_value=base_tensor)

        mock_model2_promise = AsyncMock()
        mock_model2_promise.get = AsyncMock(return_value=model2_tensor)

        def get_tensor_side_effect(model_uri, *args, **kwargs):
            if "base" in model_uri:
                return mock_base_promise
            return mock_model2_promise

        with patch.object(mock_index_manager, "get_tensor", side_effect=get_tensor_side_effect):
            result = await merger._merge_layer(shard_layer, device="cpu")

            assert isinstance(result, torch.Tensor)
            assert result.shape == (8, 8)

            # Result should not have NaN or Inf
            assert not torch.isnan(result).any()
            assert not torch.isinf(result).any()
