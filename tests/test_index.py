# tests/test_index.py
import pytest
import torch
import json
import asyncio
from pathlib import Path
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from aioresponses import aioresponses

from shard.index import (
    ModelShard,
    TensorPromise,
    HFMultiModelIndex,
    OfflineMultiModelIndex
)
from shard.download import DownloadManager


class TestModelShard:
    """Test suite for ModelShard dataclass"""

    def test_model_shard_creation(self):
        """Test ModelShard creation"""
        weight_map = {"layer1": "shard1.safetensors", "layer2": "shard1.safetensors"}
        shard = ModelShard(
            filename="shard1.safetensors",
            contained_keys=["layer1", "layer2"],
            weight_map=weight_map,
            local_path=Path("/tmp/shard1.safetensors")
        )

        assert shard.filename == "shard1.safetensors"
        assert shard.contained_keys == ["layer1", "layer2"]
        assert shard.weight_map == weight_map
        assert shard.local_path == Path("/tmp/shard1.safetensors")

    def test_model_shard_no_local_path(self):
        """Test ModelShard with no local path"""
        shard = ModelShard(
            filename="shard1.safetensors",
            contained_keys=["layer1"],
            weight_map={"layer1": "shard1.safetensors"}
        )

        assert shard.local_path is None


class TestTensorPromise:
    """Test suite for TensorPromise"""

    async def test_tensor_promise_creation(self):
        """Test TensorPromise creation"""
        promise = TensorPromise(
            model_uri="test/model",
            tensor_name="layer.weight",
            device="cpu"
        )

        assert promise.model_uri == "test/model"
        assert promise.tensor_name == "layer.weight"
        assert promise.device == "cpu"
        assert not promise._future.done()

    async def test_tensor_promise_set_result(self):
        """Test TensorPromise.set_result"""
        promise = TensorPromise("test/model", "layer.weight", "cpu")
        test_tensor = torch.randn(3, 3)

        promise.set_result(test_tensor)

        result = await promise.get()
        assert torch.equal(result, test_tensor)
        assert result.device.type == "cpu"

    async def test_tensor_promise_set_exception(self):
        """Test TensorPromise.set_exception"""
        promise = TensorPromise("test/model", "layer.weight", "cpu")
        test_exception = RuntimeError("Test error")

        promise.set_exception(test_exception)

        with pytest.raises(RuntimeError, match="Test error"):
            await promise.get()

    async def test_tensor_promise_device_move(self):
        """Test TensorPromise moves tensor to correct device"""
        promise = TensorPromise("test/model", "layer.weight", "cpu")
        test_tensor = torch.randn(3, 3)

        promise.set_result(test_tensor)

        result = await promise.get()
        assert result.device.type == "cpu"


class TestHFMultiModelIndex:
    """Test suite for HFMultiModelIndex"""

    def test_hf_index_initialization(self, tmp_path):
        """Test HFMultiModelIndex initialization"""
        download_manager = DownloadManager(storage_path=tmp_path)
        index = HFMultiModelIndex(
            download_manager=download_manager,
            cache_path=tmp_path / "cache"
        )

        assert index.download_manager == download_manager
        assert index.cache_path == tmp_path / "cache"
        assert len(index.model_indexes) == 0
        assert len(index.model_shards) == 0

    def test_hf_index_default_cache_path(self, tmp_path):
        """Test HFMultiModelIndex uses default cache path when none provided"""
        download_manager = DownloadManager(storage_path=tmp_path)
        index = HFMultiModelIndex(download_manager=download_manager)

        assert index.cache_path.exists()
        assert "shardmerge" in str(index.cache_path)

    async def test_add_model_from_storage(self, tmp_path, mock_model_index):
        """Test add_model loads from storage if available"""
        # Setup storage with model
        storage_path = tmp_path / "storage"
        model_path = storage_path / "test/model"
        model_path.mkdir(parents=True)

        index_file = model_path / "model.safetensors.index.json"
        with open(index_file, "w") as f:
            json.dump(mock_model_index, f)

        download_manager = DownloadManager(storage_path=storage_path)
        index = HFMultiModelIndex(download_manager=download_manager)

        await index.add_model("test/model")

        assert "test/model" in index.model_indexes
        assert index.model_indexes["test/model"] == mock_model_index

    async def test_add_model_downloads_from_hf(self, tmp_path, mock_model_index):
        """Test add_model downloads from HuggingFace if not in storage"""
        storage_path = tmp_path / "storage"
        storage_path.mkdir()

        download_manager = DownloadManager(storage_path=storage_path)
        index = HFMultiModelIndex(download_manager=download_manager)

        # Mock the HTTP response
        with aioresponses() as mocked:
            mocked.get(
                "https://huggingface.co/test/model/raw/main/model.safetensors.index.json",
                status=200,
                body=json.dumps(mock_model_index),
                headers={"content-type": "application/json"}
            )

            await index.add_model("test/model")

        assert "test/model" in index.model_indexes
        assert index.model_indexes["test/model"]["weight_map"] == mock_model_index["weight_map"]

    async def test_add_model_already_added(self, tmp_path, mock_model_index):
        """Test add_model skips if model already added"""
        storage_path = tmp_path / "storage"
        model_path = storage_path / "test/model"
        model_path.mkdir(parents=True)

        index_file = model_path / "model.safetensors.index.json"
        with open(index_file, "w") as f:
            json.dump(mock_model_index, f)

        download_manager = DownloadManager(storage_path=storage_path)
        index = HFMultiModelIndex(download_manager=download_manager)

        # Add model twice
        await index.add_model("test/model")
        await index.add_model("test/model")

        assert "test/model" in index.model_indexes

    def test_get_ordered_weights(self, tmp_path, mock_model_index):
        """Test _get_ordered_weights returns correctly ordered weights"""
        download_manager = DownloadManager(storage_path=tmp_path)
        index = HFMultiModelIndex(download_manager=download_manager)

        index.model_indexes["test/model"] = mock_model_index

        ordered = index._get_ordered_weights("test/model")

        # Check order: embed -> layers -> norm -> lm_head
        assert ordered[0] == "model.embed_tokens.weight"
        assert "model.layers.0.self_attn.q_proj.weight" in ordered
        assert ordered[-2] == "model.norm.weight"
        assert ordered[-1] == "lm_head.weight"

    def test_get_ordered_weights_missing_model(self, tmp_path):
        """Test _get_ordered_weights raises error for missing model"""
        download_manager = DownloadManager(storage_path=tmp_path)
        index = HFMultiModelIndex(download_manager=download_manager)

        with pytest.raises(KeyError, match="not found in index"):
            index._get_ordered_weights("nonexistent/model")

    def test_get_layer_order(self, tmp_path, mock_model_index):
        """Test get_layer_order returns copy of ordered weights"""
        download_manager = DownloadManager(storage_path=tmp_path)
        index = HFMultiModelIndex(download_manager=download_manager)

        index.model_indexes["test/model"] = mock_model_index
        index._ordered_weights["test/model"] = ["layer1", "layer2"]

        order = index.get_layer_order("test/model")

        assert order == ["layer1", "layer2"]
        # Verify it's a copy
        order.append("layer3")
        assert index._ordered_weights["test/model"] == ["layer1", "layer2"]

    def test_get_model_keys(self, tmp_path, mock_model_index):
        """Test get_model_keys returns all tensor names"""
        download_manager = DownloadManager(storage_path=tmp_path)
        index = HFMultiModelIndex(download_manager=download_manager)

        index.model_indexes["test/model"] = mock_model_index

        keys = index.get_model_keys("test/model")

        assert isinstance(keys, set)
        assert "model.embed_tokens.weight" in keys
        assert "lm_head.weight" in keys

    def test_get_model_keys_missing_model(self, tmp_path):
        """Test get_model_keys raises error for missing model"""
        download_manager = DownloadManager(storage_path=tmp_path)
        index = HFMultiModelIndex(download_manager=download_manager)

        with pytest.raises(KeyError, match="not found in index"):
            index.get_model_keys("nonexistent/model")

    def test_get_layer_order_missing_model(self, tmp_path):
        """Test get_layer_order raises error for missing model"""
        download_manager = DownloadManager(storage_path=tmp_path)
        index = HFMultiModelIndex(download_manager=download_manager)

        with pytest.raises(KeyError, match="not found in index"):
            index.get_layer_order("nonexistent/model")

    def test_get_tensor_model_not_found(self, tmp_path):
        """Test get_tensor raises error when model not in index"""
        download_manager = DownloadManager(storage_path=tmp_path)
        index = HFMultiModelIndex(download_manager=download_manager)

        with pytest.raises(KeyError, match="not found in index"):
            index.get_tensor("nonexistent/model", "layer.weight", "cpu")

    def test_get_tensor_tensor_not_found(self, tmp_path, mock_model_index):
        """Test get_tensor raises error when tensor not in model"""
        download_manager = DownloadManager(storage_path=tmp_path)
        index = HFMultiModelIndex(download_manager=download_manager)

        index.model_indexes["test/model"] = mock_model_index

        with pytest.raises(KeyError, match="not found in model"):
            index.get_tensor("test/model", "nonexistent.layer", "cpu")

    async def test_preload_tensor_success(self, tmp_path, mock_model_index):
        """Test preload_tensor starts download"""
        download_manager = DownloadManager(storage_path=tmp_path)
        index = HFMultiModelIndex(download_manager=download_manager)

        index.model_indexes["test/model"] = mock_model_index
        index.model_shards["test/model"] = {}

        # Mock the cache_file method
        download_manager.cache_file = AsyncMock()

        await index.preload_tensor("test/model", "model.embed_tokens.weight")

        # Verify cache_file was called with no_claims=-1
        assert download_manager.cache_file.called
        call_args = download_manager.cache_file.call_args[1]
        assert call_args["no_claims"] == -1

    async def test_preload_tensor_missing_tensor(self, tmp_path, mock_model_index):
        """Test preload_tensor handles missing tensor gracefully"""
        download_manager = DownloadManager(storage_path=tmp_path)
        index = HFMultiModelIndex(download_manager=download_manager)

        index.model_indexes["test/model"] = mock_model_index

        # Should not raise, just log exception
        await index.preload_tensor("test/model", "nonexistent.layer")


class TestOfflineMultiModelIndex:
    """Test suite for OfflineMultiModelIndex"""

    def test_offline_index_initialization(self):
        """Test OfflineMultiModelIndex initialization"""
        index = OfflineMultiModelIndex()

        assert len(index.model_paths) == 0
        assert len(index.model_indexes) == 0
        assert len(index.model_shards) == 0

    def test_add_model_success(self, mock_model_directory):
        """Test add_model successfully loads local model"""
        index = OfflineMultiModelIndex()

        index.add_model(mock_model_directory)

        model_id = mock_model_directory.name
        assert model_id in index.model_indexes
        assert model_id in index.model_paths
        assert index.model_paths[model_id] == mock_model_directory

    def test_add_model_not_directory(self, tmp_path):
        """Test add_model raises error for non-directory path"""
        index = OfflineMultiModelIndex()

        file_path = tmp_path / "not_a_dir.txt"
        file_path.write_text("test")

        with pytest.raises(NotADirectoryError):
            index.add_model(file_path)

    def test_add_model_missing_index(self, tmp_path):
        """Test add_model raises error when index file missing"""
        index = OfflineMultiModelIndex()

        model_dir = tmp_path / "model_without_index"
        model_dir.mkdir()

        with pytest.raises(FileNotFoundError, match="Index file.*not found"):
            index.add_model(model_dir)

    def test_add_model_invalid_json(self, tmp_path):
        """Test add_model raises error for invalid JSON"""
        index = OfflineMultiModelIndex()

        model_dir = tmp_path / "model_invalid_json"
        model_dir.mkdir()

        index_file = model_dir / "model.safetensors.index.json"
        index_file.write_text("invalid json{{{")

        with pytest.raises(ValueError, match="Failed to parse index file"):
            index.add_model(model_dir)

    def test_add_model_already_added(self, mock_model_directory):
        """Test add_model skips already added model"""
        index = OfflineMultiModelIndex()

        index.add_model(mock_model_directory)
        # Add again - should log warning and skip
        index.add_model(mock_model_directory)

        model_id = mock_model_directory.name
        assert model_id in index.model_indexes

    def test_contains_operator(self, mock_model_directory):
        """Test __contains__ operator"""
        index = OfflineMultiModelIndex()

        model_id = mock_model_directory.name
        assert model_id not in index

        index.add_model(mock_model_directory)

        assert model_id in index

    def test_len_operator(self, mock_model_directory):
        """Test __len__ operator"""
        index = OfflineMultiModelIndex()

        assert len(index) == 0

        index.add_model(mock_model_directory)

        assert len(index) == 1

    def test_get_layer_order(self, mock_model_directory):
        """Test get_layer_order returns ordered weights"""
        index = OfflineMultiModelIndex()
        index.add_model(mock_model_directory)

        model_id = mock_model_directory.name
        order = index.get_layer_order(model_id)

        assert isinstance(order, list)
        assert len(order) > 0
        # Check order: embed -> layers -> norm -> lm_head
        assert order[0] == "model.embed_tokens.weight"
        assert order[-2] == "model.norm.weight"
        assert order[-1] == "lm_head.weight"

    def test_get_model_keys(self, mock_model_directory):
        """Test get_model_keys returns all tensor names"""
        index = OfflineMultiModelIndex()
        index.add_model(mock_model_directory)

        model_id = mock_model_directory.name
        keys = index.get_model_keys(model_id)

        assert isinstance(keys, set)
        assert "model.embed_tokens.weight" in keys
        assert "lm_head.weight" in keys

    def test_get_model_keys_missing_model(self):
        """Test get_model_keys raises error for missing model"""
        index = OfflineMultiModelIndex()

        with pytest.raises(KeyError, match="not found in index"):
            index.get_model_keys("nonexistent_model")

    async def test_get_tensor_returns_promise(self, mock_model_directory):
        """Test get_tensor returns TensorPromise"""
        index = OfflineMultiModelIndex()
        index.add_model(mock_model_directory)

        model_id = mock_model_directory.name
        promise = index.get_tensor(model_id, "model.embed_tokens.weight", device="cpu")

        assert isinstance(promise, TensorPromise)
        assert promise.model_uri == model_id
        assert promise.tensor_name == "model.embed_tokens.weight"
        assert promise.device == "cpu"

    async def test_get_tensor_missing_model(self):
        """Test get_tensor raises error for missing model"""
        index = OfflineMultiModelIndex()

        with pytest.raises(KeyError, match="not found in index"):
            index.get_tensor("nonexistent", "layer.weight", "cpu")

    async def test_get_tensor_missing_tensor(self, mock_model_directory):
        """Test get_tensor raises error for missing tensor"""
        index = OfflineMultiModelIndex()
        index.add_model(mock_model_directory)

        model_id = mock_model_directory.name

        with pytest.raises(KeyError, match="not found in model"):
            index.get_tensor(model_id, "nonexistent.layer", "cpu")

    def test_add_model_missing_weight_map(self, tmp_path):
        """Test add_model raises error when weight_map is missing"""
        index = OfflineMultiModelIndex()

        model_dir = tmp_path / "model_no_weight_map"
        model_dir.mkdir()

        index_file = model_dir / "model.safetensors.index.json"
        with open(index_file, "w") as f:
            json.dump({"metadata": {"format": "pt"}}, f)

        with pytest.raises(ValueError, match="missing 'weight_map'"):
            index.add_model(model_dir)

    def test_add_model_ordering_failure(self, tmp_path):
        """Test add_model cleans up when ordering fails"""
        index = OfflineMultiModelIndex()

        model_dir = tmp_path / "model_bad_weights"
        model_dir.mkdir()

        # Create an index with inconsistent layer structure
        # Layer 1 has more components than layer 0, which will cause mismatch
        index_data = {
            "metadata": {"format": "pt"},
            "weight_map": {
                "model.layers.0.weight": "model-00001.safetensors",
                "model.layers.1.weight": "model-00001.safetensors",
                "model.layers.1.extra_weight": "model-00001.safetensors"  # Extra weight not in layer 0
            }
        }

        index_file = model_dir / "model.safetensors.index.json"
        with open(index_file, "w") as f:
            json.dump(index_data, f)

        with pytest.raises(ValueError, match="Weight ordering mismatch"):
            index.add_model(model_dir)

        # Verify cleanup happened
        assert model_dir.name not in index.model_indexes
        assert model_dir.name not in index.model_paths

    def test_get_layer_order_retry_on_missing(self, mock_model_directory):
        """Test get_layer_order attempts to generate if not cached"""
        index = OfflineMultiModelIndex()
        index.add_model(mock_model_directory)

        model_id = mock_model_directory.name

        # Remove cached ordered weights
        del index._ordered_weights[model_id]

        # Should regenerate
        order = index.get_layer_order(model_id)
        assert isinstance(order, list)
        assert len(order) > 0

    def test_get_layer_order_failure(self, tmp_path):
        """Test get_layer_order raises error when model not found and can't generate"""
        index = OfflineMultiModelIndex()

        with pytest.raises(KeyError, match="not found or ordering failed"):
            index.get_layer_order("nonexistent_model")

    async def test_load_tensor_from_disk(self, tmp_path):
        """Test _load_tensor loads tensor from local disk"""
        import torch
        from safetensors.torch import save_file

        # Create a model directory with actual shard file
        model_dir = tmp_path / "test-model"
        model_dir.mkdir()

        # Create a shard file
        test_tensor = torch.randn(3, 3)
        shard_file = model_dir / "model-00001.safetensors"
        save_file({"test.weight": test_tensor}, str(shard_file))

        # Create index
        index_data = {
            "metadata": {"format": "pt"},
            "weight_map": {
                "test.weight": "model-00001.safetensors"
            }
        }

        index_file = model_dir / "model.safetensors.index.json"
        with open(index_file, "w") as f:
            json.dump(index_data, f)

        # Test loading
        index = OfflineMultiModelIndex()
        index.add_model(model_dir)

        model_id = model_dir.name
        promise = index.get_tensor(model_id, "test.weight", "cpu")

        loaded_tensor = await promise.get()
        assert torch.allclose(loaded_tensor, test_tensor)

    async def test_load_tensor_missing_shard(self, mock_model_directory):
        """Test _load_tensor handles missing shard file"""
        index = OfflineMultiModelIndex()
        index.add_model(mock_model_directory)

        model_id = mock_model_directory.name
        promise = index.get_tensor(model_id, "model.embed_tokens.weight", "cpu")

        with pytest.raises(FileNotFoundError):
            await promise.get()

    async def test_load_tensor_cached(self, tmp_path):
        """Test get_tensor uses cached tensor"""
        import torch
        from safetensors.torch import save_file

        model_dir = tmp_path / "test-model"
        model_dir.mkdir()

        test_tensor = torch.randn(3, 3)
        shard_file = model_dir / "model-00001.safetensors"
        save_file({"test.weight": test_tensor}, str(shard_file))

        index_data = {
            "metadata": {"format": "pt"},
            "weight_map": {
                "test.weight": "model-00001.safetensors"
            }
        }

        index_file = model_dir / "model.safetensors.index.json"
        with open(index_file, "w") as f:
            json.dump(index_data, f)

        index = OfflineMultiModelIndex()
        index.add_model(model_dir)

        model_id = model_dir.name

        # Load first time
        promise1 = index.get_tensor(model_id, "test.weight", "cpu")
        tensor1 = await promise1.get()

        # Load second time (should use cache)
        promise2 = index.get_tensor(model_id, "test.weight", "cpu")
        tensor2 = await promise2.get()

        assert torch.equal(tensor1, tensor2)
