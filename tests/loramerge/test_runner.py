import json

import pytest
import torch
from safetensors.torch import save_file

from shard.loramerge.runner import (
    EMBED_TENSOR,
    FINAL_NORM_TENSOR,
    LM_HEAD_TENSOR,
    _readout_spec,
    build_readout_whitener,
)


def write_checkpoint(tmp_path, config, tensors):
    shard = "model-00001.safetensors"
    save_file(tensors, str(tmp_path / shard))
    (tmp_path / "config.json").write_text(json.dumps(config))
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps({
        "weight_map": {name: shard for name in tensors},
    }))


def test_untied_qwen_readout_uses_lm_head_and_unit_offset_gain(tmp_path):
    head = torch.tensor([[1.0, 2.0], [2.0, -1.0], [0.5, 1.5]])
    embed = torch.tensor([[9.0, 0.0], [0.0, 7.0], [4.0, 4.0]])
    norm = torch.tensor([0.25, -0.5])
    write_checkpoint(
        tmp_path,
        {
            "model_type": "qwen3_5",
            "tie_word_embeddings": False,
            "text_config": {
                "model_type": "qwen3_5_text",
                "tie_word_embeddings": False,
            },
        },
        {LM_HEAD_TENSOR: head, EMBED_TENSOR: embed, FINAL_NORM_TENSOR: norm},
    )

    mhalf, _ = build_readout_whitener(tmp_path, device="cpu")
    effective_head = head * (1.0 + norm).unsqueeze(0)
    expected_metric = effective_head.T @ effective_head

    assert torch.allclose(mhalf @ mhalf, expected_metric, rtol=1e-5, atol=1e-5)
    embedding_metric = (embed * (1.0 + norm).unsqueeze(0)).T @ (
        embed * (1.0 + norm).unsqueeze(0)
    )
    assert not torch.allclose(mhalf @ mhalf, embedding_metric)


def test_tied_gemma_readout_uses_embeddings_and_unit_offset_gain(tmp_path):
    embed = torch.tensor([[1.0, 0.5], [0.25, 2.0], [2.0, -1.0]])
    norm = torch.tensor([-0.25, 0.5])
    write_checkpoint(
        tmp_path,
        {
            "model_type": "gemma4",
            "tie_word_embeddings": True,
            "text_config": {
                "model_type": "gemma4_text",
                "tie_word_embeddings": True,
            },
        },
        {EMBED_TENSOR: embed, FINAL_NORM_TENSOR: norm},
    )

    mhalf, _ = build_readout_whitener(tmp_path, device="cpu")
    effective_head = embed * (1.0 + norm).unsqueeze(0)

    assert torch.allclose(mhalf @ mhalf, effective_head.T @ effective_head,
                          rtol=1e-5, atol=1e-5)


def test_untied_checkpoint_without_lm_head_fails_closed(tmp_path):
    config = {
        "model_type": "qwen3_5",
        "text_config": {
            "model_type": "qwen3_5_text",
            "tie_word_embeddings": False,
        },
    }
    (tmp_path / "config.json").write_text(json.dumps(config))
    weight_map = {EMBED_TENSOR: "model.safetensors", FINAL_NORM_TENSOR: "model.safetensors"}

    with pytest.raises(ValueError, match="untied checkpoint is missing lm_head.weight"):
        _readout_spec(tmp_path, weight_map)


def test_unknown_rmsnorm_convention_fails_closed(tmp_path):
    (tmp_path / "config.json").write_text(json.dumps({
        "model_type": "unknown_model",
        "tie_word_embeddings": True,
    }))
    weight_map = {EMBED_TENSOR: "model.safetensors", FINAL_NORM_TENSOR: "model.safetensors"}

    with pytest.raises(ValueError, match="unsupported RMSNorm weight convention"):
        _readout_spec(tmp_path, weight_map)
