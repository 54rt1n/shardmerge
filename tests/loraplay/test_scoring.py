# tests/loraplay/test_scoring.py
import pytest
import torch

from shard.loraplay.scoring import (
    LayerScore, ScoreReport, score_layer, metric_sqrt_from_gram,
)


class TestScoreLayer:
    def test_perfect_lora_zero_error(self):
        d = torch.randn(8, 6)
        s = score_layer("l", d, d.clone(), rank=4)
        assert s.raw_rel_error == pytest.approx(0.0, abs=1e-6)
        assert s.act_rel_error is None
        assert s.nbytes == 4 * (8 + 6) * 2

    def test_identity_metric_matches_raw(self):
        dt = torch.randn(8, 6)
        dl = dt + 0.1 * torch.randn(8, 6)
        C = torch.eye(6)
        s = score_layer("l", dt, dl, rank=4, metric_sqrt=C)
        assert s.act_rel_error == pytest.approx(s.raw_rel_error, rel=1e-5)

    def test_activation_error_matches_manual(self):
        dt = torch.randn(8, 6)
        dl = torch.randn(8, 6)
        H = torch.randn(6, 10) @ torch.randn(10, 6)
        H = H @ H.T  # PSD 6x6
        C = metric_sqrt_from_gram(H, damping=0.0)
        s = score_layer("l", dt, dl, rank=4, metric_sqrt=C)
        manual = ((dt - dl) @ C).norm() / (dt @ C).norm()
        assert s.act_rel_error == pytest.approx(manual.item(), rel=1e-4)

    def test_shape_mismatch_raises(self):
        with pytest.raises(ValueError):
            score_layer("l", torch.randn(8, 6), torch.randn(8, 5), rank=4)

    def test_behavior_error_prefers_activation(self):
        s = LayerScore("l", 4, 8, 6, raw_rel_error=0.5, act_rel_error=0.2, nbytes=100)
        assert s.behavior_error == 0.2
        s2 = LayerScore("l", 4, 8, 6, raw_rel_error=0.5, act_rel_error=None, nbytes=100)
        assert s2.behavior_error == 0.5


class TestMetricSqrt:
    def test_is_symmetric_sqrt(self):
        A = torch.randn(5, 8)
        H = A @ A.T
        C = metric_sqrt_from_gram(H, damping=0.0)
        assert torch.allclose(C @ C, H, atol=1e-3)

    def test_rejects_non_square(self):
        with pytest.raises(ValueError):
            metric_sqrt_from_gram(torch.randn(4, 5))


class TestScoreReport:
    def test_aggregates_and_totals(self):
        layers = [
            LayerScore("a", 4, 8, 6, 0.2, 0.1, 100),
            LayerScore("b", 8, 8, 6, 0.4, 0.3, 200),
        ]
        rep = ScoreReport(layers)
        assert rep.total_bytes == 300
        s = rep.summary()
        assert s["layers"] == 2
        assert s["raw_rel_error"]["worst"] == 0.4
        assert s["act_rel_error"]["mean"] == pytest.approx(0.2)

    def test_summary_without_metric(self):
        rep = ScoreReport([LayerScore("a", 4, 8, 6, 0.2, None, 100)])
        s = rep.summary()
        assert s["act_rel_error"] == {}
        assert s["behavior_error"]["mean"] == 0.2
