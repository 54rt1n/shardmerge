import click
import pytest
import torch
import yaml

from shard.exlora.config import DEFAULT_TARGET_MODULES
from shard.exlora.peft import AdapterAccumulator
from shard.loramerge import (
    LoraMergeConfig,
    LoraMergeSourceConfig,
    PeftDeltaSource,
    is_embedding_tensor,
    is_lm_head_tensor,
)


LAYER = "model.layers.0.self_attn.q_proj.weight"


def write_yaml(tmp_path, data):
    path = tmp_path / "loramerge.yaml"
    with open(path, "w") as f:
        yaml.dump(data, f)
    return path


class TestSourceConfig:
    def test_creation_and_layer_filter(self):
        cfg = LoraMergeSourceConfig(
            path="adapter",
            kind="PEFT",
            weight=0.5,
            start_layer=2,
            end_layer=4,
            target_modules=["q_proj"],
        )

        assert cfg.kind == "peft"
        assert cfg.weight == 0.5
        assert cfg.use_layer_index(1) is False
        assert cfg.use_layer_index(2) is True
        assert cfg.use_layer_index(4) is True
        assert cfg.use_layer_index(5) is False
        assert cfg.is_target(LAYER)
        assert not cfg.is_target("model.layers.0.self_attn.k_proj.weight")

    def test_from_dict_accepts_old_merge_aliases(self):
        cfg = LoraMergeSourceConfig.from_dict({
            "model": "adapter-path",
            "alpha": 0.25,
            "kind": "peft",
            "ignored": True,
        })

        assert cfg.path == "adapter-path"
        assert cfg.weight == 0.25
        assert not hasattr(cfg, "ignored")

    def test_from_dict_accepts_io_aliases(self):
        cfg = LoraMergeSourceConfig.from_dict({
            "model": "model-path",
            "kind": "model",
            "is_input": True,
            "is_output": True,
        })

        assert cfg.path == "model-path"
        assert cfg.is_embedding
        assert cfg.is_lm_head
        assert cfg.is_io_source
        assert not cfg.is_delta_source
        assert cfg.provides_tensor("model.language_model.embed_tokens.weight")
        assert cfg.provides_tensor("lm_head.weight")
        assert not cfg.provides_tensor("model.layers.0.self_attn.q_proj.weight")

    def test_from_dict_accepts_disabled_alias(self):
        cfg = LoraMergeSourceConfig.from_dict({
            "path": "adapter-path",
            "kind": "peft",
            "disabled": True,
        })

        assert cfg.enabled is False

    def test_from_dict_requires_pathish_field(self):
        with pytest.raises(click.BadParameter):
            LoraMergeSourceConfig.from_dict({"kind": "peft"})


class TestLoraMergeConfig:
    def test_minimal_yaml(self, tmp_path):
        cfg_file = write_yaml(tmp_path, {
            "base_model": "test/base",
            "output_dir": str(tmp_path / "out"),
            "sources": [
                {"path": "adapters/a", "kind": "peft", "weight": 1.5},
                {"artifact": "dense/b", "kind": "nvfp4_residual", "name": "dense-b"},
            ],
        })

        cfg = LoraMergeConfig.from_yaml(cfg_file)

        assert cfg.base_model == "test/base"
        assert cfg.output_path == tmp_path / "out"
        assert cfg.merge_rule == "fourier_slerp"
        assert cfg.arithmetic_torch_dtype is torch.float32
        assert cfg.output_torch_dtype is torch.bfloat16
        assert cfg.target_modules == DEFAULT_TARGET_MODULES
        assert [s.path for s in cfg.sources] == ["adapters/a", "dense/b"]
        assert cfg.sources[0].weight == 1.5
        assert cfg.sources[1].name == "dense-b"

    def test_io_model_sources_are_passthrough_not_delta_sources(self, tmp_path):
        cfg_file = write_yaml(tmp_path, {
            "base_model": "test/base",
            "output_dir": str(tmp_path / "out"),
            "sources": [
                {"path": "adapters/a", "kind": "peft"},
                {"model": "models/input", "kind": "model", "is_input": True},
                {"model": "models/output", "kind": "model", "is_output": True},
            ],
        })

        cfg = LoraMergeConfig.from_yaml(cfg_file)

        assert cfg.embedding_source.path == "models/input"
        assert cfg.lm_head_source.path == "models/output"
        assert cfg.passthrough_source_for("model.embed_tokens.weight").path == "models/input"
        assert cfg.passthrough_source_for("model.language_model.embed_tokens.weight").path == "models/input"
        assert cfg.passthrough_source_for("lm_head.weight").path == "models/output"
        assert cfg.passthrough_source_for("model.layers.0.mlp.down_proj.weight") is None
        assert cfg.passthrough_model_for("model.embed_tokens.weight") == "models/input"
        assert cfg.passthrough_model_for("lm_head.weight") == "models/output"
        assert cfg.passthrough_model_for("model.layers.0.mlp.down_proj.weight") is None
        assert [source.path for source in cfg.sources if source.is_delta_source] == ["adapters/a"]

    def test_qat_embedding_source_can_differ_from_merge_base(self, tmp_path):
        cfg_file = write_yaml(tmp_path, {
            "base_model": "models/google/gemma-4-31B-it",
            "output_dir": str(tmp_path / "out"),
            "sources": [
                {"path": "adapters/a", "kind": "peft"},
                {
                    "model": "models/google/gemma-4-31B-it-qat-q4_0-unquantized",
                    "kind": "model",
                    "is_input": True,
                },
                {"model": "models/Gryphe/Gemma-4-31B-StyleTune", "kind": "model", "is_output": True},
            ],
        })

        cfg = LoraMergeConfig.from_yaml(cfg_file)

        assert cfg.embedding_source is not None
        assert cfg.embedding_model == "models/google/gemma-4-31B-it-qat-q4_0-unquantized"
        assert cfg.lm_head_model == "models/Gryphe/Gemma-4-31B-StyleTune"
        assert cfg.passthrough_source_for("model.embed_tokens.weight").path == cfg.embedding_model
        assert cfg.passthrough_model_for("model.embed_tokens.weight") == cfg.embedding_model
        assert cfg.passthrough_model_for("lm_head.weight") == "models/Gryphe/Gemma-4-31B-StyleTune"

    def test_passthrough_models_default_to_base_model(self, tmp_path):
        cfg_file = write_yaml(tmp_path, {
            "base_model": "models/base",
            "output_dir": str(tmp_path / "out"),
            "sources": [{"path": "adapters/a", "kind": "peft"}],
        })

        cfg = LoraMergeConfig.from_yaml(cfg_file)

        assert cfg.embedding_source is None
        assert cfg.lm_head_source is None
        assert cfg.embedding_model == "models/base"
        assert cfg.lm_head_model == "models/base"

    def test_disabled_sources_do_not_participate_in_routing_or_loading(self, tmp_path):
        active_dir = tmp_path / "active"
        disabled_dir = tmp_path / "disabled"
        acc = AdapterAccumulator(base_model="base", target_modules=["q_proj"])
        acc.add(LAYER, torch.ones(1, 2), torch.ones(3, 1), rank=1)
        acc.write(active_dir)
        acc.write(disabled_dir)
        cfg_file = write_yaml(tmp_path, {
            "base_model": "models/base",
            "output_dir": str(tmp_path / "out"),
            "sources": [
                {"path": str(active_dir), "kind": "peft", "name": "active"},
                {"path": str(disabled_dir), "kind": "peft", "name": "disabled", "disabled": True},
                {"model": "models/ignored-input", "kind": "model", "is_input": True, "disabled": True},
            ],
        })

        cfg = LoraMergeConfig.from_yaml(cfg_file)

        assert cfg.embedding_source is None
        sources = cfg.load_sources()
        assert len(sources) == 1
        assert sources[0].label == "active"

    def test_all_policy_fields_loaded(self, tmp_path):
        cfg_file = write_yaml(tmp_path, {
            "base_model": "b",
            "output_dir": "o",
            "merge_rule": "fourier_slerp",
            "output_dtype": "float32",
            "scaffold_format": "none",
            "target_modules": ["q_proj"],
            "fourier_cutoff_pct": 0.12,
            "fourier_cull_pct": 0.03,
            "fourier_small_norm_ratio": 0.2,
            "device": "cuda",
            "cache_dir": "c",
            "storage_dir": "s",
            "clean_cache": True,
            "nvfp4_extra_bpe": 1.0,
            "nvfp4_residual_dtype": "float32",
            "nvfp4_index_bits": 64,
            "nvfp4_selection_policy": "floor_effect",
            "nvfp4_floor_correction": "code_position_mean",
            "sources": [{"path": "a", "kind": "peft"}],
        })

        cfg = LoraMergeConfig.from_yaml(cfg_file)

        assert cfg.merge_rule == "fourier_slerp"
        assert cfg.output_torch_dtype is torch.float32
        assert cfg.scaffold_format == "none"
        assert cfg.target_modules == ["q_proj"]
        assert cfg.fourier_cutoff_pct == 0.12
        assert cfg.fourier_cull_pct == 0.03
        assert cfg.fourier_small_norm_ratio == 0.2
        assert cfg.device == "cuda"
        assert cfg.cache_path.name == "c"
        assert cfg.storage_path.name == "s"
        assert cfg.clean_cache is True
        assert cfg.nvfp4_extra_bpe == 1.0
        assert cfg.nvfp4_residual_dtype == "float32"
        assert cfg.nvfp4_index_bits == 64
        assert cfg.nvfp4_selection_policy == "floor_effect"
        assert cfg.nvfp4_floor_correction == "code_position_mean"

    def test_missing_required_and_bad_sources_raise(self, tmp_path):
        with pytest.raises(click.BadParameter):
            LoraMergeConfig.from_yaml(write_yaml(tmp_path, {"base_model": "b"}))

        with pytest.raises(click.BadParameter, match="non-empty"):
            LoraMergeConfig.from_yaml(write_yaml(tmp_path, {
                "base_model": "b",
                "output_dir": "o",
                "sources": [],
            }))

        with pytest.raises(click.BadParameter, match="mapping"):
            LoraMergeConfig.from_yaml(write_yaml(tmp_path, {
                "base_model": "b",
                "output_dir": "o",
                "sources": ["not-a-map"],
            }))

    @pytest.mark.parametrize(
        "field,value,match",
        [
            ("merge_rule", "wat", "merge_rule"),
            ("arithmetic_dtype", "bfloat16", "float32"),
            ("scaffold_format", "fp16", "scaffold_format"),
            ("target_modules", "q_proj", "target_modules"),
            ("fourier_cutoff_pct", 1.5, "fourier_cutoff_pct"),
            ("fourier_cull_pct", -0.1, "fourier_cull_pct"),
            ("fourier_small_norm_ratio", -0.1, "fourier_small_norm_ratio"),
            ("nvfp4_extra_bpe", -1.0, "non-negative"),
            ("nvfp4_index_bits", 12, "nvfp4_index_bits"),
            ("nvfp4_selection_policy", "wat", "selection_policy"),
            ("nvfp4_floor_correction", "wat", "floor_correction"),
        ],
    )
    def test_bad_policy_values_raise(self, tmp_path, field, value, match):
        cfg = {
            "base_model": "b",
            "output_dir": "o",
            "sources": [{"path": "a", "kind": "peft"}],
            field: value,
        }
        with pytest.raises(click.BadParameter, match=match):
            LoraMergeConfig.from_yaml(write_yaml(tmp_path, cfg))

    @pytest.mark.parametrize(
        "source_field,value,match",
        [
            ("pool_rank", 0, "pool_rank"),
            ("pool_energy", -0.1, "pool_energy"),
            ("pool_residual", -0.1, "pool_residual"),
            ("pool_residual", 1.1, "pool_residual"),
        ],
    )
    def test_bad_pool_source_values_raise(self, tmp_path, source_field, value, match):
        with pytest.raises(click.BadParameter, match=match):
            LoraMergeConfig.from_yaml(write_yaml(tmp_path, {
                "base_model": "b",
                "output_dir": "o",
                "sources": [{"path": "a", "kind": "peft", source_field: value}],
            }))

    def test_bad_source_kind_raises(self, tmp_path):
        cfg_file = write_yaml(tmp_path, {
            "base_model": "b",
            "output_dir": "o",
            "sources": [{"path": "a", "kind": "wat"}],
        })

        with pytest.raises(click.BadParameter, match="unknown source kind"):
            LoraMergeConfig.from_yaml(cfg_file)

    def test_bad_io_source_values_raise(self, tmp_path):
        with pytest.raises(click.BadParameter, match="model sources"):
            LoraMergeConfig.from_yaml(write_yaml(tmp_path, {
                "base_model": "b",
                "output_dir": "o",
                "sources": [{"path": "m", "kind": "model"}],
            }))

        with pytest.raises(click.BadParameter, match="embedding/lm_head"):
            LoraMergeConfig.from_yaml(write_yaml(tmp_path, {
                "base_model": "b",
                "output_dir": "o",
                "sources": [{"path": "a", "kind": "peft", "is_input": True}],
            }))

        with pytest.raises(click.BadParameter, match="only one embedding"):
            LoraMergeConfig.from_yaml(write_yaml(tmp_path, {
                "base_model": "b",
                "output_dir": "o",
                "sources": [
                    {"path": "a", "kind": "model", "is_input": True},
                    {"path": "b", "kind": "model", "embedding_source": True},
                ],
            }))

    def test_is_target_and_update(self, tmp_path):
        cfg = LoraMergeConfig(
            base_model="b",
            output_dir="o",
            sources=[LoraMergeSourceConfig("a")],
            target_modules=["q_proj"],
        )

        assert cfg.is_target(LAYER)
        assert not cfg.is_target("model.layers.0.self_attn.k_proj.weight")
        cfg.update(device=None, output_dtype="float32")
        assert cfg.device == "cpu"
        assert cfg.output_dtype == "float32"

    def test_load_sources(self, tmp_path):
        adapter_dir = tmp_path / "adapter"
        acc = AdapterAccumulator(base_model="base", target_modules=["q_proj"])
        acc.add(LAYER, torch.ones(1, 2), torch.ones(3, 1), rank=1)
        acc.write(adapter_dir)

        cfg = LoraMergeConfig(
            base_model="base",
            output_dir=str(tmp_path / "out"),
            sources=[
                LoraMergeSourceConfig(
                    str(adapter_dir),
                    kind="peft",
                    weight=0.75,
                    name="adapter-a",
                    group="legacy-group",
                    pool="isometry-pool",
                    pool_rank=12,
                    pool_energy=0.25,
                    pool_residual=0.75,
                    priority=2.0,
                    post_merge=True,
                )
            ],
        )

        sources = cfg.load_sources()

        assert len(sources) == 1
        assert sources[0].weight == 0.75
        assert sources[0].label == "adapter-a"
        assert sources[0].group == "legacy-group"
        assert sources[0].pool == "isometry-pool"
        assert sources[0].pool_rank == 12
        assert sources[0].pool_energy == 0.25
        assert sources[0].pool_residual == 0.75
        assert sources[0].priority == 2.0
        assert sources[0].post_merge is True
        assert isinstance(sources[0].source, PeftDeltaSource)

    def test_tensor_kind_helpers(self):
        assert is_embedding_tensor("model.embed_tokens.weight")
        assert is_embedding_tensor("model.language_model.embed_tokens.weight")
        assert not is_embedding_tensor("model.layers.0.mlp.embed_tokens.weight.extra")
        assert is_lm_head_tensor("lm_head.weight")
        assert is_lm_head_tensor("model.language_model.lm_head.weight")
        assert not is_lm_head_tensor("model.layers.0.self_attn.o_proj.weight")
