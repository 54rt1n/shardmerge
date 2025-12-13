# tests/test_config.py
import pytest
import torch
import yaml
import click
from pathlib import Path
from shard.config import MergeModel, MergeConfig


class TestMergeModel:
    """Test suite for MergeModel dataclass"""

    def test_merge_model_creation(self):
        """Test basic MergeModel creation with required fields"""
        model = MergeModel(model="test/model", base="test/base")
        assert model.model == "test/model"
        assert model.base == "test/base"
        assert model.alpha == 1.0  # default value
        assert model.is_input is False
        assert model.is_output is False
        assert model.is_norm is False
        assert model.start_layer == 0
        assert model.end_layer == -1

    def test_merge_model_with_custom_values(self):
        """Test MergeModel with custom values"""
        model = MergeModel(
            model="test/model",
            base="test/base",
            alpha=0.5,
            is_input=True,
            is_output=False,
            is_norm=True,
            start_layer=2,
            end_layer=10
        )
        assert model.alpha == 0.5
        assert model.is_input is True
        assert model.is_norm is True
        assert model.start_layer == 2
        assert model.end_layer == 10

    def test_use_layer_index_within_range(self):
        """Test use_layer_index returns True for layers within range"""
        model = MergeModel(
            model="test/model",
            base="test/base",
            start_layer=2,
            end_layer=10
        )
        assert model.use_layer_index(5) is True
        assert model.use_layer_index(2) is True
        assert model.use_layer_index(10) is True

    def test_use_layer_index_outside_range(self):
        """Test use_layer_index returns False for layers outside range"""
        model = MergeModel(
            model="test/model",
            base="test/base",
            start_layer=2,
            end_layer=10
        )
        assert model.use_layer_index(1) is False
        assert model.use_layer_index(11) is False

    def test_use_layer_index_no_end_limit(self):
        """Test use_layer_index with no end layer limit (-1)"""
        model = MergeModel(
            model="test/model",
            base="test/base",
            start_layer=5,
            end_layer=-1
        )
        assert model.use_layer_index(5) is True
        assert model.use_layer_index(100) is True
        assert model.use_layer_index(4) is False

    def test_use_layer_index_edge_cases(self):
        """Test use_layer_index edge cases"""
        model = MergeModel(
            model="test/model",
            base="test/base",
            start_layer=0,
            end_layer=0
        )
        assert model.use_layer_index(0) is True
        assert model.use_layer_index(1) is False


class TestMergeConfig:
    """Test suite for MergeConfig dataclass"""

    def test_merge_config_creation(self, mock_merge_model):
        """Test basic MergeConfig creation"""
        config = MergeConfig(
            finetune_merge=[mock_merge_model],
            output_base_model="test/base",
            output_dir="/tmp/output"
        )
        assert config.output_base_model == "test/base"
        assert config.output_dir == "/tmp/output"
        assert len(config.finetune_merge) == 1
        assert config.output_dtype == "bfloat16"
        assert config.device == "cpu"
        assert config.clean_cache is False

    def test_input_model_property(self):
        """Test input_model property returns correct model"""
        input_model = MergeModel(model="test/input", base="test/base", is_input=True)
        other_model = MergeModel(model="test/other", base="test/base")

        config = MergeConfig(
            finetune_merge=[other_model, input_model],
            output_base_model="test/base",
            output_dir="/tmp/output"
        )
        assert config.input_model == input_model

    def test_input_model_property_none(self):
        """Test input_model property returns None when no input model"""
        config = MergeConfig(
            finetune_merge=[MergeModel(model="test/model", base="test/base")],
            output_base_model="test/base",
            output_dir="/tmp/output"
        )
        assert config.input_model is None

    def test_output_model_property(self):
        """Test output_model property returns correct model"""
        output_model = MergeModel(model="test/output", base="test/base", is_output=True)
        other_model = MergeModel(model="test/other", base="test/base")

        config = MergeConfig(
            finetune_merge=[other_model, output_model],
            output_base_model="test/base",
            output_dir="/tmp/output"
        )
        assert config.output_model == output_model

    def test_output_model_property_none(self):
        """Test output_model property returns None when no output model"""
        config = MergeConfig(
            finetune_merge=[MergeModel(model="test/model", base="test/base")],
            output_base_model="test/base",
            output_dir="/tmp/output"
        )
        assert config.output_model is None

    def test_output_path_property(self, tmp_path):
        """Test output_path property returns Path object"""
        config = MergeConfig(
            finetune_merge=[],
            output_base_model="test/base",
            output_dir=str(tmp_path / "output")
        )
        assert isinstance(config.output_path, Path)
        assert config.output_path == tmp_path / "output"

    def test_cache_path_property(self, tmp_path):
        """Test cache_path property returns Path object"""
        config = MergeConfig(
            finetune_merge=[],
            output_base_model="test/base",
            output_dir="/tmp/output",
            cache_dir=str(tmp_path / "cache")
        )
        assert isinstance(config.cache_path, Path)
        assert config.cache_path == tmp_path / "cache"

    def test_storage_path_property(self, tmp_path):
        """Test storage_path property returns Path object"""
        config = MergeConfig(
            finetune_merge=[],
            output_base_model="test/base",
            output_dir="/tmp/output",
            storage_dir=str(tmp_path / "storage")
        )
        assert isinstance(config.storage_path, Path)
        assert config.storage_path == tmp_path / "storage"

    def test_output_astype_property(self):
        """Test output_astype property returns correct dtype"""
        config = MergeConfig(
            finetune_merge=[],
            output_base_model="test/base",
            output_dir="/tmp/output",
            output_dtype="bfloat16"
        )
        assert config.output_astype == torch.bfloat16

    def test_output_astype_float32(self):
        """Test output_astype with float32"""
        config = MergeConfig(
            finetune_merge=[],
            output_base_model="test/base",
            output_dir="/tmp/output",
            output_dtype="float32"
        )
        assert config.output_astype == torch.float32

    def test_update_method_with_dict(self):
        """Test update method with dictionary"""
        config = MergeConfig(
            finetune_merge=[],
            output_base_model="test/base",
            output_dir="/tmp/output",
            device="cpu"
        )
        config.update({"device": "cuda", "clean_cache": True})
        assert config.device == "cuda"
        assert config.clean_cache is True

    def test_update_method_with_kwargs(self):
        """Test update method with keyword arguments"""
        config = MergeConfig(
            finetune_merge=[],
            output_base_model="test/base",
            output_dir="/tmp/output",
            device="cpu"
        )
        config.update(device="cuda", clean_cache=True)
        assert config.device == "cuda"
        assert config.clean_cache is True

    def test_update_method_ignores_invalid_keys(self):
        """Test update method ignores invalid keys"""
        config = MergeConfig(
            finetune_merge=[],
            output_base_model="test/base",
            output_dir="/tmp/output"
        )
        config.update({"invalid_key": "value", "device": "cuda"})
        assert not hasattr(config, "invalid_key")
        assert config.device == "cuda"

    def test_to_dict_method(self):
        """Test to_dict method returns correct dictionary"""
        model = MergeModel(model="test/model", base="test/base")
        config = MergeConfig(
            finetune_merge=[model],
            output_base_model="test/base",
            output_dir="/tmp/output",
            device="cpu",
            clean_cache=True,
            cache_dir="/tmp/cache",
            storage_dir="/tmp/storage"
        )

        result = config.to_dict()
        assert result["output_base_model"] == "test/base"
        assert result["output_dir"] == "/tmp/output"
        assert result["device"] == "cpu"
        assert result["clean_cache"] is True
        assert result["cache_dir"] == "/tmp/cache"
        assert result["storage_dir"] == "/tmp/storage"
        assert result["finetune_merge"] == ["test/model"]

    def test_from_yaml_valid_config(self, sample_yaml_config):
        """Test from_yaml with valid configuration"""
        config = MergeConfig.from_yaml(sample_yaml_config)
        assert config.output_base_model == "test/base-model"
        assert len(config.finetune_merge) == 2
        assert isinstance(config.finetune_merge[0], MergeModel)
        assert config.finetune_merge[0].model == "test/model1"
        assert config.finetune_merge[0].alpha == 0.5

    def test_from_yaml_missing_required_field(self, tmp_path):
        """Test from_yaml raises error for missing required fields"""
        config_data = {
            "finetune_merge": [{"model": "test/model", "base": "test/base"}]
            # Missing output_base_model and output_dir
        }

        config_file = tmp_path / "invalid_config.yaml"
        with open(config_file, "w") as f:
            yaml.dump(config_data, f)

        with pytest.raises(click.BadParameter, match="Missing required configuration fields"):
            MergeConfig.from_yaml(config_file)

    def test_from_yaml_invalid_finetune_merge_type(self, tmp_path):
        """Test from_yaml raises error for invalid finetune_merge type"""
        config_data = {
            "output_base_model": "test/base",
            "output_dir": "/tmp/output",
            "finetune_merge": "not-a-list"  # Should be a list
        }

        config_file = tmp_path / "invalid_config.yaml"
        with open(config_file, "w") as f:
            yaml.dump(config_data, f)

        with pytest.raises(click.BadParameter, match="finetune_merge must be a list"):
            MergeConfig.from_yaml(config_file)

    def test_from_yaml_empty_finetune_merge(self, tmp_path):
        """Test from_yaml with empty finetune_merge list"""
        config_data = {
            "output_base_model": "test/base",
            "output_dir": "/tmp/output",
            "finetune_merge": []
        }

        config_file = tmp_path / "config.yaml"
        with open(config_file, "w") as f:
            yaml.dump(config_data, f)

        config = MergeConfig.from_yaml(config_file)
        assert len(config.finetune_merge) == 0
