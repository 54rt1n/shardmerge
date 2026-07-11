# tests/loraplay/test_adapter.py
import pytest
import torch

from shard.exlora.peft import AdapterAccumulator
from shard.loraplay.adapter import load_adapter, parse_lora_key


class TestParseLoraKey:
    def test_parses_a_and_b(self):
        base = "model.layers.0.self_attn.q_proj.weight"
        ka = "base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight"
        kb = "base_model.model.model.layers.0.self_attn.q_proj.lora_B.weight"
        assert parse_lora_key(ka) == (base, "A")
        assert parse_lora_key(kb) == (base, "B")

    def test_rejects_non_lora(self):
        with pytest.raises(ValueError):
            parse_lora_key("base_model.model.x.weight")
        with pytest.raises(ValueError):
            parse_lora_key("not_a_prefix.lora_A.weight")


def _write_adapter(tmp_path):
    acc = AdapterAccumulator(base_model="test/base", target_modules=["q_proj", "down_proj"])
    A1, B1 = torch.randn(4, 16), torch.randn(8, 4)
    A2, B2 = torch.randn(6, 32), torch.randn(10, 6)
    acc.add("model.layers.0.self_attn.q_proj.weight", A1, B1, rank=4)
    acc.add("model.layers.0.mlp.down_proj.weight", A2, B2, rank=6)
    acc.write(tmp_path)
    return {"model.layers.0.self_attn.q_proj.weight": (A1, B1),
            "model.layers.0.mlp.down_proj.weight": (A2, B2)}


class TestLoadAdapter:
    def test_roundtrip_reconstructs_delta(self, tmp_path):
        truth = _write_adapter(tmp_path)
        adapter = load_adapter(tmp_path)
        assert set(adapter.layers()) == set(truth)
        for name, (A, B) in truth.items():
            # alpha == rank in our convention -> scaling 1 -> delta = B @ A
            assert adapter.scaling[name] == pytest.approx(1.0)
            assert adapter.rank[name] in (4, 6)
            recon = adapter.delta(name)
            expected = B.float() @ A.float()
            assert torch.allclose(recon, expected, atol=1e-4)

    def test_rank_inferred_from_factor_shape(self, tmp_path):
        _write_adapter(tmp_path)
        adapter = load_adapter(tmp_path)
        q = "model.layers.0.self_attn.q_proj.weight"
        assert adapter.rank[q] == adapter.factors[q]["A"].shape[0]
