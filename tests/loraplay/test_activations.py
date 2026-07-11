# tests/loraplay/test_activations.py
import torch
import torch.nn as nn

from shard.loraplay.activations import GramAccumulator, harvest_model, save_grams, load_grams


class Toy(nn.Module):
    def __init__(self):
        super().__init__()
        self.q_proj = nn.Linear(4, 8, bias=False)
        self.other = nn.Linear(8, 8, bias=False)

    def forward(self, x):
        return self.other(self.q_proj(x))


class TestGramAccumulator:
    def test_observe_accumulates_gram(self):
        acc = GramAccumulator(["q_proj"])
        x1 = torch.randn(5, 4)
        x2 = torch.randn(3, 4)
        acc.observe("q_proj", x1)
        acc.observe("q_proj", x2)
        expected = x1.T @ x1 + x2.T @ x2
        assert torch.allclose(acc.gram["q_proj"], expected, atol=1e-4)
        assert acc.counts["q_proj"] == 8

    def test_is_target_and_weight_name(self):
        acc = GramAccumulator(["q_proj", "down_proj"])
        assert acc.is_target("model.layers.0.self_attn.q_proj")
        assert not acc.is_target("model.layers.0.self_attn.k_proj")
        assert acc.weight_name("a.q_proj") == "a.q_proj.weight"

    def test_flattens_leading_dims(self):
        acc = GramAccumulator(["q_proj"])
        x = torch.randn(2, 7, 4)  # [batch, seq, in]
        acc.observe("q_proj", x)
        assert acc.counts["q_proj"] == 14
        assert acc.gram["q_proj"].shape == (4, 4)


class TestHarvestModel:
    def test_hooks_capture_only_targets(self):
        model = Toy()
        x = torch.randn(6, 4)
        grams = harvest_model(model, [x], target_suffixes=["q_proj"])
        # only q_proj harvested, keyed by weight name, equals X^T X / N
        assert set(grams) == {"q_proj.weight"}
        assert torch.allclose(grams["q_proj.weight"], (x.T @ x) / 6, atol=1e-4)

    def test_result_normalized_by_count(self):
        acc = GramAccumulator(["q_proj"])
        x = torch.randn(10, 4)
        acc.observe("q_proj", x)
        out = acc.result()
        assert torch.allclose(out["q_proj.weight"], (x.T @ x) / 10, atol=1e-4)


def test_save_load_roundtrip(tmp_path):
    grams = {"a.q_proj.weight": torch.randn(4, 4), "b.k_proj.weight": torch.randn(6, 6)}
    p = tmp_path / "grams.safetensors"
    save_grams(grams, p)
    loaded = load_grams(p)
    assert set(loaded) == set(grams)
    assert torch.allclose(loaded["a.q_proj.weight"], grams["a.q_proj.weight"])
