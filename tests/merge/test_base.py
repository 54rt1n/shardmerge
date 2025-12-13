# tests/merge/test_base.py
import pytest
import torch
import asyncio
import queue
import time
from pathlib import Path
from unittest.mock import Mock, AsyncMock, MagicMock, patch

from shard.merge.base import TaskRunner, MergeTensorsBase
from shard.config import MergeConfig, MergeModel
from shard.index import HFMultiModelIndex
from shard.writer import ShardLayer, ModelWriter
from shard.constants import INPUT_LAYER, OUTPUT_LAYER


class TestTaskRunner:
    """Test suite for TaskRunner"""

    def test_task_runner_initialization(self):
        """Test TaskRunner initialization"""
        runner = TaskRunner(max_workers=5)

        assert runner.queue.qsize() == 0
        assert runner.running is False
        # TaskRunner doesn't store max_workers attribute, it passes it to ThreadPoolExecutor

    async def test_task_runner_submit_task(self):
        """Test submitting a task to TaskRunner"""
        runner = TaskRunner(max_workers=2)

        executed = []

        async def test_task():
            executed.append(True)
            return "done"

        runner.submit(test_task)

        # Wait a bit for task to execute
        await asyncio.sleep(0.2)

        # Clean up
        runner.queue.join()

    async def test_task_runner_multiple_tasks(self):
        """Test TaskRunner handles multiple tasks"""
        runner = TaskRunner(max_workers=2)

        results = []

        async def test_task(value):
            results.append(value)
            return value

        for i in range(3):
            runner.submit(lambda v=i: test_task(v))

        # Wait for tasks
        await asyncio.sleep(0.5)

        # Clean up
        runner.queue.join()


class ConcreteTestMerger(MergeTensorsBase):
    """Concrete implementation of MergeTensorsBase for testing"""

    def get_readme(self) -> str:
        return "Test README"

    async def _merge_layer(self, shard_layer: ShardLayer, device: str) -> torch.Tensor:
        # Simple implementation: return random tensor
        return torch.randn(3, 3)


class TestMergeTensorsBase:
    """Test suite for MergeTensorsBase"""

    def test_merge_tensors_base_initialization(self, merge_config_for_testing):
        """Test MergeTensorsBase initialization"""
        merger = ConcreteTestMerger(config=merge_config_for_testing)

        assert merger.config == merge_config_for_testing
        assert isinstance(merger.index_manager, HFMultiModelIndex)

    def test_merge_tensors_base_with_index_manager(self, merge_config_for_testing, mock_index_manager):
        """Test MergeTensorsBase with provided index manager"""
        merger = ConcreteTestMerger(
            config=merge_config_for_testing,
            index_manager=mock_index_manager
        )

        assert merger.index_manager == mock_index_manager

    def test_get_readme_abstract(self, merge_config_for_testing):
        """Test get_readme is implemented"""
        merger = ConcreteTestMerger(config=merge_config_for_testing)
        readme = merger.get_readme()

        assert isinstance(readme, str)
        assert len(readme) > 0

    async def test_get_base_output_tensor(self, merge_config_for_testing, mock_index_manager):
        """Test get_base_output_tensor retrieves tensor"""
        merger = ConcreteTestMerger(
            config=merge_config_for_testing,
            index_manager=mock_index_manager
        )

        shard_layer = ShardLayer(
            layer_order_idx=0,
            shard_name="model-00001.safetensors",
            layer_name="model.layers.0.weight",
            written=False
        )

        # Mock the tensor promise
        test_tensor = torch.randn(3, 3)
        mock_promise = AsyncMock()
        mock_promise.get = AsyncMock(return_value=test_tensor)

        with patch.object(mock_index_manager, "get_tensor", return_value=mock_promise):
            result = await merger.get_base_output_tensor(shard_layer, device="cpu")

            assert isinstance(result, torch.Tensor)
            assert result.dtype == torch.float32

    async def test_get_delta_for_models(self, merge_config_for_testing, mock_index_manager):
        """Test get_delta_for_models computes deltas"""
        merger = ConcreteTestMerger(
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
        base_tensor = torch.ones(3, 3)
        model_tensor = torch.ones(3, 3) * 2

        # Mock tensor promises
        mock_base_promise = AsyncMock()
        mock_base_promise.get = AsyncMock(return_value=base_tensor)

        mock_model_promise = AsyncMock()
        mock_model_promise.get = AsyncMock(return_value=model_tensor)

        def get_tensor_side_effect(model_uri, *args, **kwargs):
            if "base" in model_uri:
                return mock_base_promise
            else:
                return mock_model_promise

        with patch.object(mock_index_manager, "get_tensor", side_effect=get_tensor_side_effect):
            models = [MergeModel(model="test/model1", base="test/base", alpha=0.5)]
            deltas = await merger.get_delta_for_models(models, shard_layer, device="cpu")

            assert len(deltas) == 1
            assert isinstance(deltas[0], torch.Tensor)

    async def test_get_delta_for_models_apply_alpha(self, merge_config_for_testing, mock_index_manager):
        """Test get_delta_for_models applies alpha correctly"""
        merger = ConcreteTestMerger(
            config=merge_config_for_testing,
            index_manager=mock_index_manager
        )

        shard_layer = ShardLayer(
            layer_order_idx=1,
            shard_name="model-00001.safetensors",
            layer_name="model.layers.0.weight",
            written=False
        )

        base_tensor = torch.ones(3, 3)
        model_tensor = torch.ones(3, 3) * 3  # Delta will be 2

        mock_base_promise = AsyncMock()
        mock_base_promise.get = AsyncMock(return_value=base_tensor)

        mock_model_promise = AsyncMock()
        mock_model_promise.get = AsyncMock(return_value=model_tensor)

        def get_tensor_side_effect(model_uri, *args, **kwargs):
            if "base" in model_uri:
                return mock_base_promise
            else:
                return mock_model_promise

        with patch.object(mock_index_manager, "get_tensor", side_effect=get_tensor_side_effect):
            models = [MergeModel(model="test/model1", base="test/base", alpha=0.5)]

            # With alpha
            deltas_with_alpha = await merger.get_delta_for_models(models, shard_layer, device="cpu", apply_alpha=True)
            # Without alpha
            deltas_no_alpha = await merger.get_delta_for_models(models, shard_layer, device="cpu", apply_alpha=False)

            # Delta is 2, with alpha=0.5 should be 1
            assert torch.allclose(deltas_with_alpha[0], torch.ones(3, 3))
            # Without alpha should be 2
            assert torch.allclose(deltas_no_alpha[0], torch.ones(3, 3) * 2)

    async def test_initialize(self, merge_config_for_testing):
        """Test initialize method"""
        merger = ConcreteTestMerger(config=merge_config_for_testing)

        # Mock add_model
        with patch.object(merger.index_manager, "add_model", new_callable=AsyncMock) as mock_add:
            with patch.object(merger.index_manager, "get_model_keys", return_value={"layer1", "layer2"}):
                merger.index_manager.model_indexes["test/base"] = {"test": "data"}

                await merger.initialize()

                # Should have called add_model for base and all finetune models
                assert mock_add.call_count >= 1
                assert merger.index_doc == {"test": "data"}

    def test_get_writer(self, merge_config_for_testing, mock_index_manager):
        """Test get_writer returns ModelWriter"""
        merger = ConcreteTestMerger(
            config=merge_config_for_testing,
            index_manager=mock_index_manager
        )

        merger.index_doc = mock_index_manager.model_indexes["test/base"]
        layer_order = ["layer1", "layer2"]

        writer = merger.get_writer(layer_order)

        assert isinstance(writer, ModelWriter)
        assert writer.base_index == merger.index_doc
        assert writer.output_path == merge_config_for_testing.output_path

    async def test_merge_layer_abstract(self, merge_config_for_testing):
        """Test _merge_layer is implemented in concrete class"""
        merger = ConcreteTestMerger(config=merge_config_for_testing)

        shard_layer = ShardLayer(
            layer_order_idx=0,
            shard_name="test.safetensors",
            layer_name="test.layer",
            written=False
        )

        result = await merger._merge_layer(shard_layer, device="cpu")

        assert isinstance(result, torch.Tensor)

    async def test_process_layers(self, merge_config_for_testing, tmp_path):
        """Test _process_layers method"""
        merger = ConcreteTestMerger(config=merge_config_for_testing)

        # Create a minimal writer
        layer_order = ["model.layers.0.weight"]
        base_index = {
            "metadata": {"format": "pt"},
            "weight_map": {"model.layers.0.weight": "model-00001.safetensors"}
        }

        writer = ModelWriter(
            base_index=base_index,
            output_path=tmp_path / "output",
            layer_order=layer_order,
            output_astype=torch.bfloat16
        )

        shard_layers = [
            ShardLayer(
                layer_order_idx=0,
                shard_name="model-00001.safetensors",
                layer_name="model.layers.0.weight",
                written=False
            )
        ]

        # Mock add_tensor to avoid file I/O
        with patch.object(writer, "add_tensor"):
            await merger._process_layers(writer, shard_layers, device="cpu")

            # Should have called add_tensor
            assert writer.add_tensor.called
