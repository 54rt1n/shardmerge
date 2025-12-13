# tests/merge/conftest.py
import pytest
import torch
from pathlib import Path
from unittest.mock import AsyncMock

from shard.config import MergeConfig, MergeModel
from shard.download import DownloadManager
from shard.index import HFMultiModelIndex


@pytest.fixture
def merge_models():
    """Create a list of MergeModel instances for testing"""
    return [
        MergeModel(
            model="test/model1",
            base="test/base",
            alpha=0.5,
            is_input=False,
            is_output=False
        ),
        MergeModel(
            model="test/model2",
            base="test/base",
            alpha=0.3,
            is_input=False,
            is_output=False
        )
    ]


@pytest.fixture
def merge_config_for_testing(tmp_path, merge_models):
    """Create a MergeConfig instance for testing"""
    return MergeConfig(
        finetune_merge=merge_models,
        output_base_model="test/base",
        output_dir=str(tmp_path / "output"),
        device="cpu",
        cache_dir=str(tmp_path / "cache"),
        storage_dir=str(tmp_path / "storage")
    )


@pytest.fixture
def mock_index_manager(tmp_path):
    """Create a mock HFMultiModelIndex for testing"""
    download_manager = DownloadManager(storage_path=tmp_path / "storage")
    index_manager = HFMultiModelIndex(
        download_manager=download_manager,
        cache_path=tmp_path / "cache"
    )

    # Add mock data
    index_manager.model_indexes["test/base"] = {
        "metadata": {"format": "pt"},
        "weight_map": {
            "model.embed_tokens.weight": "model-00001.safetensors",
            "model.layers.0.weight": "model-00001.safetensors",
            "model.norm.weight": "model-00002.safetensors",
            "lm_head.weight": "model-00002.safetensors"
        }
    }

    index_manager._ordered_weights["test/base"] = [
        "model.embed_tokens.weight",
        "model.layers.0.weight",
        "model.norm.weight",
        "lm_head.weight"
    ]

    return index_manager
