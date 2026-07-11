# tests/exlora/test_peft.py
import json
import pytest
import torch
from safetensors import safe_open

from shard.exlora.peft import (
    lora_weight_key,
    module_pattern_key,
    AdapterAccumulator,
)


class TestKeyNaming:
    def test_lora_weight_key(self):
        name = "model.layers.0.self_attn.q_proj.weight"
        assert (
            lora_weight_key(name, "A")
            == "base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight"
        )
        assert (
            lora_weight_key(name, "B")
            == "base_model.model.model.layers.0.self_attn.q_proj.lora_B.weight"
        )

    def test_invalid_which(self):
        with pytest.raises(ValueError):
            lora_weight_key("x.weight", "C")

    def test_module_pattern_key_strips_weight(self):
        assert module_pattern_key("model.layers.1.mlp.up_proj.weight") == "model.layers.1.mlp.up_proj"


class TestAdapterAccumulator:
    def _acc(self):
        acc = AdapterAccumulator(base_model="test/base", target_modules=["q_proj", "v_proj"])
        acc.add("model.layers.0.self_attn.q_proj.weight", torch.randn(8, 16), torch.randn(32, 8), rank=8)
        acc.add("model.layers.0.self_attn.v_proj.weight", torch.randn(4, 16), torch.randn(32, 4), rank=4)
        return acc

    def test_tracks_patterns_and_default_rank(self):
        acc = self._acc()
        assert acc.rank_pattern["model.layers.0.self_attn.q_proj"] == 8
        assert acc.alpha_pattern["model.layers.0.self_attn.v_proj"] == 4
        # alpha == rank everywhere => PEFT scaling == 1
        assert acc.rank_pattern == acc.alpha_pattern
        assert acc.default_rank == 8

    def test_build_config(self):
        cfg = self._acc().build_config()
        assert cfg["peft_type"] == "LORA"
        assert cfg["base_model_name_or_path"] == "test/base"
        assert cfg["r"] == cfg["lora_alpha"] == 8
        # target_modules is the explicit set of adapted module paths, not the
        # bare suffix list, so PEFT only injects into modules we actually built.
        assert cfg["target_modules"] == [
            "model.layers.0.self_attn.q_proj",
            "model.layers.0.self_attn.v_proj",
        ]

    def test_build_config_scopes_to_adapted_paths_not_suffixes(self):
        """Regression: bare suffixes match a multimodal model's vision tower.

        target_modules must contain only the full paths actually adapted (all
        under the language model here) and no bare suffix like ``q_proj`` that
        PEFT would also match against a vision-tower ``q_proj``.
        """
        acc = AdapterAccumulator(
            base_model="test/base",
            target_modules=["q_proj", "down_proj"],
        )
        acc.add(
            "model.language_model.layers.30.self_attn.q_proj.weight",
            torch.randn(8, 16), torch.randn(32, 8), rank=8,
        )
        acc.add(
            "model.language_model.layers.30.mlp.down_proj.weight",
            torch.randn(8, 16), torch.randn(32, 8), rank=8,
        )
        targets = acc.build_config()["target_modules"]

        # Exactly the adapted module paths, every one under the language model.
        assert targets == [
            "model.language_model.layers.30.mlp.down_proj",
            "model.language_model.layers.30.self_attn.q_proj",
        ]
        assert all(t.startswith("model.language_model.") for t in targets)
        # No bare suffix that would also match a vision-tower projection.
        assert "q_proj" not in targets
        assert "down_proj" not in targets
        assert not any("vision" in t for t in targets)

    def test_write_roundtrip(self, tmp_path):
        acc = self._acc()
        acc.write(tmp_path)

        cfg = json.loads((tmp_path / "adapter_config.json").read_text())
        assert cfg["peft_type"] == "LORA"

        keys = set()
        with safe_open(tmp_path / "adapter_model.safetensors", framework="pt") as f:
            keys = set(f.keys())
        assert "base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight" in keys
        assert "base_model.model.model.layers.0.self_attn.q_proj.lora_B.weight" in keys
        assert len(keys) == 4  # 2 modules x (A, B)

    def test_write_empty_raises(self, tmp_path):
        acc = AdapterAccumulator(base_model="test/base", target_modules=["q_proj"])
        with pytest.raises(RuntimeError):
            acc.write(tmp_path)
