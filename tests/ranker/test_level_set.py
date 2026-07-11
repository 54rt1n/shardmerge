import pytest
import torch

from shard.ranker import (
    RankLevelSetConfig,
    SignalRecoveryGates,
    SignalScore,
    detect_rank_level_set,
    required_rank_for_energy,
    score_signal_recovery,
)


def spectrum_with_floor(head_rank=64, floor_rank=256):
    head = torch.logspace(2.0, 0.7, head_rank)
    floor = torch.linspace(1.0, 0.92, floor_rank)
    return torch.cat([head, floor])


def test_detects_rank_level_set_above_flat_floor():
    s = spectrum_with_floor(head_rank=64)
    result = detect_rank_level_set(
        s,
        RankLevelSetConfig(max_rank=256, window=64, level_tolerance=0.15, rank_multiple=8),
    )
    assert result.reason == "level_set"
    assert result.rank == 64
    assert result.candidate_rank >= 64
    assert result.head_to_floor_ratio and result.head_to_floor_ratio > 4.0


def test_smooth_spectrum_does_not_invent_level_set():
    s = torch.linspace(1.0, 0.8, 256)
    result = detect_rank_level_set(s, RankLevelSetConfig(max_rank=128, window=32))
    assert result.reason == "no_level_set"
    assert result.rank == 128


def test_level_set_rounds_down_to_rank_multiple():
    s = spectrum_with_floor(head_rank=66)
    result = detect_rank_level_set(
        s,
        RankLevelSetConfig(max_rank=256, window=64, level_tolerance=0.15, rank_multiple=8),
    )
    assert result.reason == "level_set"
    assert result.candidate_rank >= 66
    assert result.rank == 64


def test_zero_and_empty_spectra_are_safe():
    assert detect_rank_level_set(torch.zeros(0)).rank == 0
    result = detect_rank_level_set(torch.zeros(5), RankLevelSetConfig(min_rank=2))
    assert result.reason == "zero_spectrum"
    assert result.rank == 2


def test_required_rank_for_energy():
    assert required_rank_for_energy(torch.tensor([3.0, 1.0]), 0.9) == 1
    assert required_rank_for_energy(torch.tensor([3.0, 1.0]), 1.0) == 2
    with pytest.raises(ValueError):
        required_rank_for_energy(torch.ones(3), 1.1)


def test_score_signal_recovery():
    signal = torch.tensor([1.0, 2.0, 3.0])
    score = score_signal_recovery(signal, signal)
    assert score.rel_error == pytest.approx(0.0)
    assert score.cosine == pytest.approx(1.0)
    with pytest.raises(ValueError):
        score_signal_recovery(torch.ones(2), torch.ones(3))


def test_signal_recovery_gates():
    gates = SignalRecoveryGates(min_cosine=0.95, max_error_regret=0.02, min_rank95_ratio=4.0)
    ok, failures = gates.check(
        detected_rank=64,
        reference_rank=64,
        score=SignalScore(rel_error=0.11, cosine=0.97),
        reference_score=SignalScore(rel_error=0.10, cosine=0.98),
        rank95=512,
    )
    assert ok and not failures

    ok, failures = gates.check(
        detected_rank=80,
        reference_rank=64,
        score=SignalScore(rel_error=0.20, cosine=0.90),
        reference_score=SignalScore(rel_error=0.10, cosine=0.98),
        rank95=128,
    )
    assert not ok
    assert len(failures) == 4
