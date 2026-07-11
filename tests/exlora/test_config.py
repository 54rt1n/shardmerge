# tests/exlora/test_config.py
import click
import pytest
import torch
import yaml

from shard.exlora.config import ExtractConfig, DEFAULT_TARGET_MODULES


def write_yaml(tmp_path, data):
    p = tmp_path / "config.yaml"
    with open(p, "w") as f:
        yaml.dump(data, f)
    return p


class TestFromYaml:
    def test_minimal(self, tmp_path):
        cfg_file = write_yaml(tmp_path, {
            "base_model": "test/base",
            "finetune_model": "test/ft",
            "output_dir": str(tmp_path / "out"),
        })
        cfg = ExtractConfig.from_yaml(cfg_file)
        assert cfg.base_model == "test/base"
        assert cfg.target_modules == DEFAULT_TARGET_MODULES
        assert cfg.out_dtype == torch.bfloat16

    def test_missing_required_raises(self, tmp_path):
        cfg_file = write_yaml(tmp_path, {"base_model": "test/base"})
        with pytest.raises(click.BadParameter):
            ExtractConfig.from_yaml(cfg_file)

    def test_bad_target_modules(self, tmp_path):
        cfg_file = write_yaml(tmp_path, {
            "base_model": "b", "finetune_model": "f", "output_dir": "o",
            "target_modules": "q_proj",
        })
        with pytest.raises(click.BadParameter):
            ExtractConfig.from_yaml(cfg_file)

    def test_unknown_keys_ignored(self, tmp_path):
        cfg_file = write_yaml(tmp_path, {
            "base_model": "b", "finetune_model": "f", "output_dir": "o",
            "totally_unknown": 123,
        })
        cfg = ExtractConfig.from_yaml(cfg_file)
        assert not hasattr(cfg, "totally_unknown")

    def test_projection_defaults_to_none(self, tmp_path):
        cfg_file = write_yaml(tmp_path, {
            "base_model": "b", "finetune_model": "f", "output_dir": "o",
        })
        cfg = ExtractConfig.from_yaml(cfg_file)
        assert cfg.qat_projection_model is None
        assert cfg.projection_type == "none"

    def test_projection_fields_loaded(self, tmp_path):
        cfg_file = write_yaml(tmp_path, {
            "base_model": "b", "finetune_model": "f", "output_dir": "o",
            "qat_projection_model": "models/qat_projection/gemma",
            "projection_type": "reflection",
        })
        cfg = ExtractConfig.from_yaml(cfg_file)
        assert cfg.qat_projection_model == "models/qat_projection/gemma"
        assert cfg.projection_type == "reflection"

    def test_rank_policy_fields_loaded(self, tmp_path):
        cfg_file = write_yaml(tmp_path, {
            "base_model": "b", "finetune_model": "f", "output_dir": "o",
            "rank_policy": "error_or_signal",
            "noise_floor_multiplier": 2.0,
            "min_knee_drop": 1.1,
            "tail_fraction": 0.1,
        })
        cfg = ExtractConfig.from_yaml(cfg_file)
        assert cfg.rank_policy == "error_or_signal"
        assert cfg.noise_floor_multiplier == 2.0
        assert cfg.min_knee_drop == 1.1
        assert cfg.tail_fraction == 0.1

    def test_nvfp4_residual_fields_loaded(self, tmp_path):
        cfg_file = write_yaml(tmp_path, {
            "base_model": "b",
            "finetune_model": "f",
            "output_dir": "o",
            "artifact_type": "nvfp4_residual",
            "nvfp4_extra_bpe": 2.0,
            "nvfp4_residual_dtype": "float32",
            "nvfp4_index_bits": 64,
            "nvfp4_selection_policy": "floor_effect",
            "nvfp4_floor_correction": "code_position_mean",
            "nvfp4_floor_tolerance": 1.5,
            "nvfp4_floor_effect_threshold": 2.5,
            "position_policy_enabled": True,
            "position_early_fraction": 0.25,
            "position_late_fraction": 0.8,
        })
        cfg = ExtractConfig.from_yaml(cfg_file)
        assert cfg.artifact_type == "nvfp4_residual"
        assert cfg.nvfp4_extra_bpe == 2.0
        assert cfg.nvfp4_residual_dtype == "float32"
        assert cfg.nvfp4_index_bits == 64
        assert cfg.nvfp4_selection_policy == "floor_effect"
        assert cfg.nvfp4_floor_correction == "code_position_mean"
        assert cfg.nvfp4_floor_tolerance == 1.5
        assert cfg.nvfp4_floor_effect_threshold == 2.5
        assert cfg.position_policy_enabled is True
        assert cfg.position_early_fraction == 0.25
        assert cfg.position_late_fraction == 0.8

    def test_bad_artifact_type_raises(self, tmp_path):
        cfg_file = write_yaml(tmp_path, {
            "base_model": "b",
            "finetune_model": "f",
            "output_dir": "o",
            "artifact_type": "wat",
        })
        with pytest.raises(click.BadParameter):
            ExtractConfig.from_yaml(cfg_file)

    def test_residual_rms_requires_index_bits(self, tmp_path):
        cfg_file = write_yaml(tmp_path, {
            "base_model": "b",
            "finetune_model": "f",
            "output_dir": "o",
            "artifact_type": "nvfp4_residual",
            "nvfp4_index_bits": 0,
        })
        with pytest.raises(click.BadParameter):
            ExtractConfig.from_yaml(cfg_file)

    def test_bad_nvfp4_floor_config_raises(self, tmp_path):
        cfg_file = write_yaml(tmp_path, {
            "base_model": "b",
            "finetune_model": "f",
            "output_dir": "o",
            "artifact_type": "nvfp4_residual",
            "nvfp4_selection_policy": "wat",
        })
        with pytest.raises(click.BadParameter):
            ExtractConfig.from_yaml(cfg_file)

    def test_bad_position_policy_config_raises(self, tmp_path):
        cfg_file = write_yaml(tmp_path, {
            "base_model": "b",
            "finetune_model": "f",
            "output_dir": "o",
            "position_early_fraction": 0.8,
            "position_late_fraction": 0.2,
        })
        with pytest.raises(click.BadParameter):
            ExtractConfig.from_yaml(cfg_file)

        cfg_file = write_yaml(tmp_path, {
            "base_model": "b",
            "finetune_model": "f",
            "output_dir": "o",
            "artifact_type": "nvfp4_residual",
            "nvfp4_floor_correction": "wat",
        })
        with pytest.raises(click.BadParameter):
            ExtractConfig.from_yaml(cfg_file)


class TestIsTarget:
    def test_matches_linears(self):
        cfg = ExtractConfig("b", "f", "o")
        assert cfg.is_target("model.layers.0.self_attn.q_proj.weight")
        assert cfg.is_target("model.layers.5.mlp.down_proj.weight")

    def test_excludes_norms_and_embeddings(self):
        cfg = ExtractConfig("b", "f", "o")
        assert not cfg.is_target("model.embed_tokens.weight")
        assert not cfg.is_target("model.norm.weight")
        assert not cfg.is_target("lm_head.weight")
        assert not cfg.is_target("model.layers.0.input_layernorm.weight")

    def test_respects_custom_target_modules(self):
        cfg = ExtractConfig("b", "f", "o", target_modules=["q_proj"])
        assert cfg.is_target("model.layers.0.self_attn.q_proj.weight")
        assert not cfg.is_target("model.layers.0.self_attn.k_proj.weight")


class TestModuleRankBounds:
    def test_module_of(self):
        cfg = ExtractConfig("b", "f", "o")
        assert cfg.module_of("model.layers.0.self_attn.q_proj.weight") == "q_proj"
        assert cfg.module_of("model.layers.0.mlp.down_proj.weight") == "down_proj"
        assert cfg.module_of("not_a_weight") is None

    def test_rank_bounds_falls_back_to_global(self):
        cfg = ExtractConfig("b", "f", "o", min_rank=8, max_rank=512)
        assert cfg.rank_bounds("model.layers.0.self_attn.q_proj.weight") == (8, 512)

    def test_rank_bounds_per_module_override(self):
        cfg = ExtractConfig(
            "b", "f", "o", min_rank=8, max_rank=512,
            module_max_rank={"q_proj": 256, "down_proj": 3072},
            module_min_rank={"down_proj": 1024},
        )
        assert cfg.rank_bounds("model.layers.0.self_attn.q_proj.weight") == (8, 256)
        assert cfg.rank_bounds("model.layers.0.mlp.down_proj.weight") == (1024, 3072)
        # untouched module still uses global
        assert cfg.rank_bounds("model.layers.0.self_attn.o_proj.weight") == (8, 512)


class TestUpdate:
    def test_update_ignores_none(self):
        cfg = ExtractConfig("b", "f", "o", device="cpu", max_rank=64)
        cfg.update(device=None, max_rank=128)
        assert cfg.device == "cpu"   # None ignored
        assert cfg.max_rank == 128
