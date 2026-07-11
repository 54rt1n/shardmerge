import torch
import pytest

from shard.exlora.nvfp4 import nvfp4_quantize_dequantize
from shard.exlora.nvfp4_residual import (
    NVFP4ResidualMap,
    NVFP4ResidualWriter,
    build_nvfp4_residual_entry,
    dtype_bits,
    parse_torch_dtype,
)


def rel_error(a, b):
    return ((a.to(torch.float32) - b.to(torch.float32)).norm() / a.to(torch.float32).norm()).item()


def test_dtype_helpers():
    assert parse_torch_dtype("bfloat16") is torch.bfloat16
    assert dtype_bits(torch.bfloat16) == 16
    assert dtype_bits(torch.float32) == 32


def test_build_nvfp4_residual_entry_improves_over_bare_nvfp4():
    torch.manual_seed(0)
    delta = torch.randn(16, 16)

    entry = build_nvfp4_residual_entry(
        delta,
        extra_bpe=2.0,
        residual_dtype=torch.bfloat16,
        index_bits=32,
    )

    bare = nvfp4_quantize_dequantize(delta)
    decoded = entry.dequantize()
    assert entry.residual_indices.numel() > 0
    assert rel_error(delta, decoded) < rel_error(delta, bare)
    assert entry.rel_error < entry.nvfp4_rel_error
    assert entry.total_bpe > entry.packed.bits_per_element


def test_build_nvfp4_floor_corrected_entry_roundtrips():
    torch.manual_seed(4)
    delta = torch.randn(32, 32)

    entry = build_nvfp4_residual_entry(
        delta,
        extra_bpe=2.0,
        residual_dtype=torch.bfloat16,
        index_bits=32,
        selection_policy="floor_effect",
        floor_correction_policy="code_position_mean",
        floor_tolerance=2.0,
        floor_effect_threshold=1.0,
    )

    decoded = entry.dequantize()
    assert entry.floor_correction_policy == "code_position_mean"
    assert entry.floor_correction is not None
    assert tuple(entry.floor_correction.shape) == (16, 16)
    assert entry.floor_blocks > 0
    assert entry.corrected_nvfp4_rel_error is not None
    assert rel_error(delta, decoded) == pytest.approx(entry.rel_error, rel=1e-5, abs=1e-7)
    assert entry.total_bpe >= entry.packed.bits_per_element + entry.floor_correction_bpe


def test_writer_loader_roundtrip(tmp_path):
    torch.manual_seed(1)
    delta = torch.randn(8, 16)
    name = "model.layers.0.self_attn.q_proj.weight"
    entry = build_nvfp4_residual_entry(delta, extra_bpe=3.0, index_bits=32)

    writer = NVFP4ResidualWriter(
        tmp_path,
        base_model="base",
        finetune_model="ft",
        target_modules=["q_proj"],
        extra_bpe=3.0,
        residual_dtype=torch.bfloat16,
        index_bits=32,
    )
    writer.add(name, entry)
    writer.write_manifest()

    loaded = NVFP4ResidualMap.load(tmp_path)
    decoded = loaded.dequantize(name)
    assert decoded is not None
    assert rel_error(delta, decoded) < entry.nvfp4_rel_error
    assert loaded.entry("missing") is None


def test_writer_loader_roundtrip_with_floor_correction(tmp_path):
    torch.manual_seed(5)
    delta = torch.randn(16, 32)
    name = "model.layers.0.self_attn.q_proj.weight"
    entry = build_nvfp4_residual_entry(
        delta,
        extra_bpe=2.0,
        index_bits=32,
        selection_policy="floor_effect",
        floor_correction_policy="code_position_mean",
        floor_effect_threshold=1.0,
    )

    writer = NVFP4ResidualWriter(
        tmp_path,
        base_model="base",
        finetune_model="ft",
        target_modules=["q_proj"],
        extra_bpe=2.0,
        residual_dtype=torch.bfloat16,
        index_bits=32,
        selection_policy="floor_effect",
        floor_correction_policy="code_position_mean",
    )
    writer.add(name, entry)
    writer.write_manifest()

    loaded = NVFP4ResidualMap.load(tmp_path)
    loaded_entry = loaded.entry(name)
    decoded = loaded.dequantize(name)
    assert loaded_entry is not None
    assert decoded is not None
    assert loaded_entry.floor_correction_policy == "code_position_mean"
    assert loaded_entry.floor_correction is not None
    assert tuple(loaded_entry.floor_correction.shape) == (16, 16)
    assert rel_error(delta, decoded) == pytest.approx(entry.rel_error, rel=1e-5, abs=1e-7)
