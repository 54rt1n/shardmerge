# tests/conftest.py
import pytest
import torch
import asyncio
from pathlib import Path
from unittest.mock import Mock, MagicMock, AsyncMock
import tempfile
import json


# ============================================================================
# Tensor Fixtures
# ============================================================================

@pytest.fixture
def sample_tensor_1d():
    """Create a simple 1D tensor for testing"""
    return torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])


@pytest.fixture
def sample_tensor_2d():
    """Create a simple 2D tensor for testing"""
    return torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])


@pytest.fixture
def sample_tensor_3d():
    """Create a simple 3D tensor for testing"""
    return torch.randn(2, 3, 4)


@pytest.fixture
def zero_tensor():
    """Create a zero tensor"""
    return torch.zeros(3, 3)


@pytest.fixture
def normalized_tensor_pair():
    """Create a pair of normalized tensors for testing"""
    t1 = torch.randn(4, 4)
    t2 = torch.randn(4, 4)
    t1 = t1 / t1.norm()
    t2 = t2 / t2.norm()
    return t1, t2


@pytest.fixture
def complex_tensor():
    """Create a complex tensor for FFT testing"""
    return torch.randn(8, 8, dtype=torch.complex64)


# ============================================================================
# Config Fixtures
# ============================================================================

@pytest.fixture
def mock_merge_model():
    """Create a mock MergeModel instance"""
    from shard.config import MergeModel
    return MergeModel(
        model="test/model",
        base="test/base",
        alpha=0.5,
        is_input=False,
        is_output=False,
        is_norm=False,
        start_layer=0,
        end_layer=-1
    )


@pytest.fixture
def mock_merge_config(tmp_path):
    """Create a mock MergeConfig instance"""
    from shard.config import MergeConfig, MergeModel

    models = [
        MergeModel(model="test/model1", base="test/base", alpha=0.5),
        MergeModel(model="test/model2", base="test/base", alpha=0.3),
    ]

    return MergeConfig(
        finetune_merge=models,
        output_base_model="test/base",
        output_dir=str(tmp_path / "output"),
        output_dtype="bfloat16",
        device="cpu",
        clean_cache=False,
        cache_dir=str(tmp_path / "cache"),
        storage_dir=str(tmp_path / "storage")
    )


@pytest.fixture
def sample_yaml_config(tmp_path):
    """Create a sample YAML config file"""
    config_data = {
        "output_base_model": "test/base-model",
        "finetune_merge": [
            {"model": "test/model1", "base": "test/base", "alpha": 0.5},
            {"model": "test/model2", "base": "test/base", "alpha": 0.3}
        ],
        "output_dir": str(tmp_path / "output"),
        "device": "cpu",
        "clean_cache": False,
        "cache_dir": str(tmp_path / "cache"),
        "storage_dir": str(tmp_path / "storage")
    }

    config_file = tmp_path / "config.yaml"
    import yaml
    with open(config_file, "w") as f:
        yaml.dump(config_data, f)

    return config_file


# ============================================================================
# Index Fixtures
# ============================================================================

@pytest.fixture
def mock_model_index():
    """Create a mock model index structure"""
    return {
        "metadata": {"format": "pt"},
        "weight_map": {
            "model.embed_tokens.weight": "model-00001-of-00003.safetensors",
            "model.layers.0.mlp.gate_proj.weight": "model-00001-of-00003.safetensors",
            "model.layers.0.self_attn.k_proj.weight": "model-00001-of-00003.safetensors",
            "model.layers.0.self_attn.q_proj.weight": "model-00001-of-00003.safetensors",
            "model.layers.1.mlp.gate_proj.weight": "model-00002-of-00003.safetensors",
            "model.layers.1.self_attn.k_proj.weight": "model-00002-of-00003.safetensors",
            "model.layers.1.self_attn.q_proj.weight": "model-00002-of-00003.safetensors",
            "model.norm.weight": "model-00003-of-00003.safetensors",
            "lm_head.weight": "model-00003-of-00003.safetensors"
        }
    }


@pytest.fixture
def mock_shard_layer():
    """Create a mock ShardLayer instance"""
    from shard.writer import ShardLayer
    return ShardLayer(
        layer_order_idx=0,
        shard_name="model-00001-of-00003.safetensors",
        layer_name="model.layers.0.self_attn.q_proj.weight",
        written=False
    )


# ============================================================================
# Mock Fixtures for External Dependencies
# ============================================================================

@pytest.fixture
def mock_aiohttp_session():
    """Create a mock aiohttp session"""
    session = AsyncMock()
    response = AsyncMock()
    response.status = 200
    response.headers = {"content-length": "1024"}
    response.raise_for_status = Mock()

    async def async_iter_chunked(size):
        yield b"test data chunk"

    response.content.iter_chunked = async_iter_chunked

    session.get = AsyncMock(return_value=AsyncMock(__aenter__=AsyncMock(return_value=response)))

    return session


@pytest.fixture
def mock_safetensors(monkeypatch):
    """Mock safetensors module"""
    mock_safe_open = MagicMock()
    mock_file = MagicMock()
    mock_file.keys.return_value = ["layer1", "layer2"]
    mock_file.get_tensor.return_value = torch.randn(3, 3)
    mock_safe_open.return_value.__enter__.return_value = mock_file

    monkeypatch.setattr("safetensors.safe_open", mock_safe_open)
    return mock_safe_open


@pytest.fixture
def mock_transformers(monkeypatch):
    """Mock transformers module"""
    mock_model = MagicMock()
    mock_tokenizer = MagicMock()

    mock_model_class = MagicMock(return_value=mock_model)
    mock_tokenizer_class = MagicMock(return_value=mock_tokenizer)

    monkeypatch.setattr("transformers.AutoModelForCausalLM.from_pretrained", mock_model_class)
    monkeypatch.setattr("transformers.AutoTokenizer.from_pretrained", mock_tokenizer_class)

    return {"model": mock_model, "tokenizer": mock_tokenizer}


# ============================================================================
# Temporary Directory Fixtures
# ============================================================================

@pytest.fixture
def temp_storage_dir(tmp_path):
    """Create a temporary storage directory"""
    storage = tmp_path / "storage"
    storage.mkdir()
    return storage


@pytest.fixture
def temp_cache_dir(tmp_path):
    """Create a temporary cache directory"""
    cache = tmp_path / "cache"
    cache.mkdir()
    return cache


@pytest.fixture
def temp_output_dir(tmp_path):
    """Create a temporary output directory"""
    output = tmp_path / "output"
    output.mkdir()
    return output


# ============================================================================
# Model Directory Fixtures
# ============================================================================

@pytest.fixture
def mock_model_directory(tmp_path):
    """Create a mock model directory with index and shard files"""
    model_dir = tmp_path / "test-model"
    model_dir.mkdir()

    # Create index file
    index_data = {
        "metadata": {"format": "pt"},
        "weight_map": {
            "model.embed_tokens.weight": "model-00001-of-00002.safetensors",
            "model.layers.0.self_attn.q_proj.weight": "model-00001-of-00002.safetensors",
            "model.norm.weight": "model-00002-of-00002.safetensors",
            "lm_head.weight": "model-00002-of-00002.safetensors"
        }
    }

    index_file = model_dir / "model.safetensors.index.json"
    with open(index_file, "w") as f:
        json.dump(index_data, f)

    return model_dir


# ============================================================================
# Event Loop Fixtures
# ============================================================================

@pytest.fixture
def event_loop():
    """Create an event loop for async tests"""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()
