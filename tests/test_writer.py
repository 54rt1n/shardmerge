# tests/test_writer.py
import pytest
import torch
import json
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

from shard.writer import ShardLayer, ModelWriter
from shard.constants import INPUT_LAYER, OUTPUT_LAYER


class TestShardLayer:
    """Test suite for ShardLayer"""

    def test_shard_layer_creation(self):
        """Test ShardLayer creation"""
        layer = ShardLayer(
            layer_order_idx=5,
            shard_name="model-00001.safetensors",
            layer_name="model.layers.0.weight",
            written=False
        )

        assert layer.layer_order_idx == 5
        assert layer.shard_name == "model-00001.safetensors"
        assert layer.layer_name == "model.layers.0.weight"
        assert layer.written is False

    def test_layer_number_embed_tokens(self):
        """Test layer_number for embed_tokens returns INPUT_LAYER"""
        layer = ShardLayer(
            layer_order_idx=0,
            shard_name="model-00001.safetensors",
            layer_name="model.embed_tokens.weight",
            written=False
        )

        assert layer.layer_number == INPUT_LAYER

    def test_layer_number_norm_weight(self):
        """Test layer_number for norm.weight returns OUTPUT_LAYER"""
        layer = ShardLayer(
            layer_order_idx=100,
            shard_name="model-00003.safetensors",
            layer_name="model.norm.weight",
            written=False
        )

        assert layer.layer_number == OUTPUT_LAYER

    def test_layer_number_lm_head(self):
        """Test layer_number for lm_head returns OUTPUT_LAYER"""
        layer = ShardLayer(
            layer_order_idx=101,
            shard_name="model-00003.safetensors",
            layer_name="lm_head.weight",
            written=False
        )

        assert layer.layer_number == OUTPUT_LAYER

    def test_layer_number_regular_layer(self):
        """Test layer_number for regular model layer"""
        layer = ShardLayer(
            layer_order_idx=10,
            shard_name="model-00002.safetensors",
            layer_name="model.layers.5.self_attn.q_proj.weight",
            written=False
        )

        assert layer.layer_number == 5

    def test_layer_number_layer_zero(self):
        """Test layer_number for layer 0"""
        layer = ShardLayer(
            layer_order_idx=1,
            shard_name="model-00001.safetensors",
            layer_name="model.layers.0.mlp.gate_proj.weight",
            written=False
        )

        assert layer.layer_number == 0

    def test_layer_number_invalid_format(self):
        """Test layer_number raises error for invalid layer name"""
        layer = ShardLayer(
            layer_order_idx=0,
            shard_name="model-00001.safetensors",
            layer_name="invalid.layer.name",
            written=False
        )

        with pytest.raises(ValueError, match="Unknown layer name"):
            _ = layer.layer_number

    def test_layer_number_layers_not_numeric(self):
        """Test layer_number raises error for non-numeric layer number"""
        layer = ShardLayer(
            layer_order_idx=0,
            shard_name="model-00001.safetensors",
            layer_name="model.layers.abc.weight",
            written=False
        )

        with pytest.raises(ValueError):
            _ = layer.layer_number


class TestModelWriter:
    """Test suite for ModelWriter"""

    def test_model_writer_initialization(self, tmp_path, mock_model_index):
        """Test ModelWriter initialization"""
        layer_order = list(mock_model_index["weight_map"].keys())

        writer = ModelWriter(
            base_index=mock_model_index,
            output_path=tmp_path,
            layer_order=layer_order,
            output_astype=torch.bfloat16
        )

        assert writer.base_index == mock_model_index
        assert writer.output_path == tmp_path
        assert writer.layer_order == layer_order
        assert writer.output_astype == torch.bfloat16
        assert tmp_path.exists()

    def test_model_writer_creates_index_file(self, tmp_path, mock_model_index):
        """Test ModelWriter creates index file"""
        layer_order = list(mock_model_index["weight_map"].keys())

        writer = ModelWriter(
            base_index=mock_model_index,
            output_path=tmp_path,
            layer_order=layer_order,
            output_astype=torch.bfloat16
        )

        index_path = tmp_path / "model.safetensors.index.json"
        assert index_path.exists()

        with open(index_path) as f:
            saved_index = json.load(f)

        assert saved_index == mock_model_index

    def test_model_writer_shard_to_tensors_mapping(self, tmp_path, mock_model_index):
        """Test ModelWriter creates correct shard_to_tensors mapping"""
        layer_order = list(mock_model_index["weight_map"].keys())

        writer = ModelWriter(
            base_index=mock_model_index,
            output_path=tmp_path,
            layer_order=layer_order,
            output_astype=torch.bfloat16
        )

        assert "model-00001-of-00003.safetensors" in writer.shard_to_tensors
        assert "model.embed_tokens.weight" in writer.shard_to_tensors["model-00001-of-00003.safetensors"]

    @patch("shard.writer.save_file")
    def test_add_tensor_new_shard(self, mock_save_file, tmp_path, mock_model_index):
        """Test add_tensor creates new shard file"""
        layer_order = list(mock_model_index["weight_map"].keys())

        writer = ModelWriter(
            base_index=mock_model_index,
            output_path=tmp_path,
            layer_order=layer_order,
            output_astype=torch.bfloat16
        )

        test_tensor = torch.randn(3, 3)
        writer.add_tensor("model.embed_tokens.weight", test_tensor)

        assert mock_save_file.called
        assert ("model-00001-of-00003.safetensors", "model.embed_tokens.weight") in writer.written_shard_layers

    @patch("shard.writer.save_file")
    def test_add_tensor_skips_written(self, mock_save_file, tmp_path, mock_model_index):
        """Test add_tensor skips already written tensors"""
        layer_order = list(mock_model_index["weight_map"].keys())

        writer = ModelWriter(
            base_index=mock_model_index,
            output_path=tmp_path,
            layer_order=layer_order,
            output_astype=torch.bfloat16
        )

        # Mark as already written
        writer.written_shard_layers.add(("model-00001-of-00003.safetensors", "model.embed_tokens.weight"))

        test_tensor = torch.randn(3, 3)
        writer.add_tensor("model.embed_tokens.weight", test_tensor)

        # Should not call save_file
        assert not mock_save_file.called

    def test_finalize_success(self, tmp_path, mock_model_index):
        """Test finalize succeeds when all layers written"""
        layer_order = list(mock_model_index["weight_map"].keys())

        writer = ModelWriter(
            base_index=mock_model_index,
            output_path=tmp_path,
            layer_order=layer_order,
            output_astype=torch.bfloat16
        )

        # Mark all layers as written
        for tensor_name, shard_name in mock_model_index["weight_map"].items():
            writer.written_shard_layers.add((shard_name, tensor_name))

        # Should not raise
        writer.finalize()

    def test_finalize_raises_on_missing_layers(self, tmp_path, mock_model_index):
        """Test finalize raises RuntimeError for missing layers"""
        layer_order = list(mock_model_index["weight_map"].keys())

        writer = ModelWriter(
            base_index=mock_model_index,
            output_path=tmp_path,
            layer_order=layer_order,
            output_astype=torch.bfloat16
        )

        # Don't mark any layers as written
        with pytest.raises(RuntimeError, match="Incomplete model output"):
            writer.finalize()

    def test_shard_layers_generator(self, tmp_path, mock_model_index):
        """Test shard_layers returns correct ShardLayer objects"""
        layer_order = list(mock_model_index["weight_map"].keys())

        writer = ModelWriter(
            base_index=mock_model_index,
            output_path=tmp_path,
            layer_order=layer_order,
            output_astype=torch.bfloat16
        )

        all_layers = []
        for shard_layers in writer.shard_layers():
            all_layers.extend(shard_layers)

        # Check we got all layers
        assert len(all_layers) == len(mock_model_index["weight_map"])

        # Check layers are ShardLayer instances
        for layer in all_layers:
            assert isinstance(layer, ShardLayer)

    @patch("shard.writer.snapshot_download")
    def test_from_huggingface(self, mock_snapshot, tmp_path, mock_model_index):
        """Test from_huggingface class method"""
        # Setup mock to create index file
        def create_index(*args, **kwargs):
            local_dir = kwargs.get("local_dir")
            if local_dir:
                local_dir.mkdir(parents=True, exist_ok=True)
                index_path = local_dir / "model.safetensors.index.json"
                with open(index_path, "w") as f:
                    json.dump(mock_model_index, f)

        mock_snapshot.side_effect = create_index

        layer_order = list(mock_model_index["weight_map"].keys())

        writer = ModelWriter.from_huggingface(
            model_id="test/model",
            output_path=tmp_path,
            layer_order=layer_order,
            revision="main"
        )

        assert writer.base_index == mock_model_index
        assert writer.output_path == tmp_path
        mock_snapshot.assert_called_once()

    @patch("shard.writer.safe_open")
    def test_like_model(self, mock_safe_open, tmp_path, mock_model_index):
        """Test like_model class method"""
        # Setup model directory with index
        model_path = tmp_path / "source_model"
        model_path.mkdir()

        index_file = model_path / "model.safetensors.index.json"
        with open(index_file, "w") as f:
            json.dump(mock_model_index, f)

        # Create dummy shard files
        shard1 = model_path / "model-00001-of-00003.safetensors"
        shard1.write_bytes(b"fake shard data")

        # Mock safe_open to return layer names
        mock_file = MagicMock()
        mock_file.keys.return_value = ["model.embed_tokens.weight", "model.layers.0.self_attn.q_proj.weight"]
        mock_safe_open.return_value.__enter__.return_value = mock_file

        output_path = tmp_path / "output"

        writer = ModelWriter.like_model(
            model_path=model_path,
            output_path=output_path,
            output_astype=torch.bfloat16
        )

        assert writer.base_index == mock_model_index
        assert writer.output_path == output_path
        assert writer.output_astype == torch.bfloat16
        assert len(writer.layer_order) > 0

    def test_like_model_missing_index(self, tmp_path):
        """Test like_model raises error when index file missing"""
        model_path = tmp_path / "model_without_index"
        model_path.mkdir()

        output_path = tmp_path / "output"

        with pytest.raises(FileNotFoundError, match="Model index not found"):
            ModelWriter.like_model(
                model_path=model_path,
                output_path=output_path
            )

    def test_check_existing_shards(self, tmp_path, mock_model_index):
        """Test _check_existing_shards validates existing shards"""
        from safetensors.torch import save_file

        layer_order = list(mock_model_index["weight_map"].keys())

        # Create a shard file with one of the expected tensors
        shard_path = tmp_path / "model-00001-of-00003.safetensors"
        test_tensor = torch.randn(3, 3)
        save_file({"model.embed_tokens.weight": test_tensor}, str(shard_path))

        writer = ModelWriter(
            base_index=mock_model_index,
            output_path=tmp_path,
            layer_order=layer_order,
            output_astype=torch.bfloat16
        )

        # Should have marked the tensor as written
        assert ("model-00001-of-00003.safetensors", "model.embed_tokens.weight") in writer.written_shard_layers

    @patch("shard.writer.safe_open")
    def test_check_existing_shards_extra_tensor_error(self, mock_safe_open, tmp_path, mock_model_index):
        """Test _check_existing_shards raises error for unexpected tensor"""
        layer_order = list(mock_model_index["weight_map"].keys())

        # Create shard directory
        tmp_path.mkdir(parents=True, exist_ok=True)
        shard_path = tmp_path / "model-00001-of-00003.safetensors"
        shard_path.write_bytes(b"fake shard")

        # Mock safe_open to return unexpected tensor
        mock_file = MagicMock()
        mock_file.keys.return_value = ["model.embed_tokens.weight", "unexpected.tensor"]
        mock_safe_open.return_value.__enter__.return_value = mock_file

        with pytest.raises(ValueError, match="found in .* but not in base model"):
            ModelWriter(
                base_index=mock_model_index,
                output_path=tmp_path,
                layer_order=layer_order,
                output_astype=torch.bfloat16
            )

    def test_add_tensor_update_existing_shard(self, tmp_path, mock_model_index):
        """Test add_tensor updates existing shard file"""
        from safetensors.torch import save_file

        layer_order = list(mock_model_index["weight_map"].keys())

        # Create a shard file with existing tensor
        shard_path = tmp_path / "model-00001-of-00003.safetensors"
        existing_tensor = torch.randn(3, 3)
        save_file({"model.embed_tokens.weight": existing_tensor}, str(shard_path))

        writer = ModelWriter(
            base_index=mock_model_index,
            output_path=tmp_path,
            layer_order=layer_order,
            output_astype=torch.bfloat16
        )

        # Add a different tensor from same shard
        new_tensor = torch.randn(4, 4)
        writer.add_tensor("model.layers.0.self_attn.q_proj.weight", new_tensor)

        # Both tensors should be in the shard now
        assert ("model-00001-of-00003.safetensors", "model.layers.0.self_attn.q_proj.weight") in writer.written_shard_layers

    def test_add_tensor_save_error_cleanup(self, tmp_path, mock_model_index):
        """Test add_tensor cleans up on save error"""
        from safetensors.torch import save_file

        layer_order = list(mock_model_index["weight_map"].keys())

        writer = ModelWriter(
            base_index=mock_model_index,
            output_path=tmp_path,
            layer_order=layer_order,
            output_astype=torch.bfloat16
        )

        # Create a valid shard file
        shard_path = tmp_path / "model-00001-of-00003.safetensors"
        existing_tensor = torch.randn(2, 2)
        save_file({"existing.weight": existing_tensor}, str(shard_path))

        # Patch save_file to raise error after reading the file
        with patch("shard.writer.save_file") as mock_save_file:
            mock_save_file.side_effect = Exception("Save failed")

            test_tensor = torch.randn(3, 3)

            # Should not raise, just log error
            writer.add_tensor("model.embed_tokens.weight", test_tensor)

            # Shard file should be removed after error
            assert not shard_path.exists()

    def test_from_huggingface_missing_index(self, tmp_path):
        """Test from_huggingface raises error when index not downloaded"""
        with patch("shard.writer.snapshot_download") as mock_snapshot:
            # Mock snapshot_download to not create index file
            def no_index(*args, **kwargs):
                local_dir = kwargs.get("local_dir")
                if local_dir:
                    local_dir.mkdir(parents=True, exist_ok=True)
                    # Don't create index file

            mock_snapshot.side_effect = no_index

            with pytest.raises(FileNotFoundError, match="Model index not found"):
                ModelWriter.from_huggingface(
                    model_id="test/model",
                    output_path=tmp_path,
                    layer_order=[]
                )
