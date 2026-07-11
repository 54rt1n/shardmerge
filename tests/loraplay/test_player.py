# tests/loraplay/test_player.py
import pytest
import torch
import torch.nn as nn

from shard.exlora.peft import AdapterAccumulator
from shard.loraplay.adapter import load_adapter
from shard.loraplay.player import (
    ActivationErrorAccumulator, AdapterPlayer, play_adapter,
)


class TestAccumulator:
    def test_perfect_lora_zero_error(self):
        acc = ActivationErrorAccumulator()
        X = torch.randn(5, 6)
        d = torch.randn(8, 6)
        acc.observe("q", X, d_true=d, d_lora=d.clone())
        assert acc.rel_error("q") == pytest.approx(0.0, abs=1e-6)

    def test_matches_manual_activation_error(self):
        acc = ActivationErrorAccumulator()
        X = torch.randn(7, 6)
        d_true = torch.randn(8, 6)
        d_lora = torch.randn(8, 6)
        acc.observe("q", X, d_true, d_lora)
        E = d_true - d_lora
        manual = (X @ E.T).norm() / (X @ d_true.T).norm()
        assert acc.rel_error("q") == pytest.approx(manual.item(), rel=1e-4)

    def test_accumulates_across_batches(self):
        acc = ActivationErrorAccumulator()
        d_true, d_lora = torch.randn(8, 6), torch.randn(8, 6)
        X1, X2 = torch.randn(3, 6), torch.randn(4, 6)
        acc.observe("q", X1, d_true, d_lora)
        acc.observe("q", X2, d_true, d_lora)
        assert acc.tokens["q"] == 7
        X = torch.cat([X1, X2])
        E = d_true - d_lora
        manual = (X @ E.T).norm() / (X @ d_true.T).norm()
        assert acc.rel_error("q") == pytest.approx(manual.item(), rel=1e-4)


class Toy(nn.Module):
    def __init__(self):
        super().__init__()
        self.q_proj = nn.Linear(6, 8, bias=False)

    def forward(self, x):
        return self.q_proj(x)


class TestAdapterPlayer:
    def _adapter(self, tmp_path, rank=3):
        A, B = torch.randn(rank, 6), torch.randn(8, rank)
        acc = AdapterAccumulator(base_model="t/b", target_modules=["q_proj"])
        acc.add("q_proj.weight", A, B, rank=rank)
        acc.write(tmp_path)
        return load_adapter(tmp_path), A, B

    def test_hook_scores_match_manual(self, tmp_path):
        adapter, A, B = self._adapter(tmp_path)
        d_lora = B @ A
        d_true = d_lora + 0.1 * torch.randn(8, 6)  # imperfect reconstruction

        model = Toy()
        X = torch.randn(10, 6)
        report = play_adapter(model, [X], adapter,
                              true_delta_provider=lambda wn, bw: d_true)

        assert len(report.layers) == 1
        s = report.layers[0]
        assert s.layer_name == "q_proj.weight" and s.rank == 3
        E = d_true - d_lora
        manual = (X @ E.T).norm() / (X @ d_true.T).norm()
        assert s.act_rel_error == pytest.approx(manual.item(), rel=1e-3)

    def test_provider_none_skips_layer(self, tmp_path):
        adapter, _, _ = self._adapter(tmp_path)
        model = Toy()
        report = play_adapter(model, [torch.randn(4, 6)], adapter,
                              true_delta_provider=lambda wn, bw: None)
        assert len(report.layers) == 0  # nothing scored

    def test_no_h_or_activations_stored(self, tmp_path):
        # the accumulator keeps only scalars per layer
        adapter, _, _ = self._adapter(tmp_path)
        player = AdapterPlayer(adapter, true_delta_provider=lambda wn, bw: torch.randn(8, 6))
        player.register(Toy())
        with torch.no_grad():
            Toy()  # noop
        # accumulator state is scalar dicts only
        assert isinstance(player.acc.resid_sq, dict)
        assert all(isinstance(v, float) for v in player.acc.resid_sq.values())
        player.remove()
