# tests/exlora/test_knee.py
# Copyright (C) 2026 Martin Bukowski

import torch

from shard.exlora.knee import (
    KneeDetection,
    classify_delta_map,
    detect_signal_knee,
    select_extraction_rank,
)


def test_extraction_rank_energy_path_climbs_to_target():
    # energy path: knee at rank 1, but cumulative energy hits 94% only at rank 3.
    s = torch.tensor([3.0, 1.0, 0.8, 0.1, 0.1])           # sq cum: .84, .94-, .998
    det = KneeDetection(True, 1, 8.0, 90.0, 0.01, "knee", 0.84, "energy")
    assert select_extraction_rank(s, det, energy_target=0.94) == 3


def test_extraction_rank_flat_floor_returns_knee():
    # flat-floor path: 94% would chase the noise floor, so keep the knee rank.
    s = torch.cat([torch.ones(64), torch.full((5000,), 0.01)])
    det = KneeDetection(True, 64, 4.0, 1.5, 0.04, "knee", 0.64, "flat_floor")
    assert select_extraction_rank(s, det, energy_target=0.94) == 64


def _knee(rank):
    return KneeDetection(True, rank, 5.0, 1.5, 0.03, "knee", 0.7)


def _dense():
    return KneeDetection(False, None, 1.1, 20.0, 0.4, "no_cliff", 0.05)


def test_classify_all_tensor_lora():
    c = classify_delta_map([_knee(64)] * 20, n_unchanged=0)
    assert c.kind == "low_rank" and c.rank == 64


def test_classify_sparse_lora_abliteration():
    c = classify_delta_map([_knee(1)] * 6, n_unchanged=40)
    assert c.kind == "sparse_low_rank" and c.rank == 1


def test_classify_dense():
    assert classify_delta_map([_dense()] * 20, n_unchanged=0).kind == "dense"


def test_classify_head_only():
    c = classify_delta_map([], n_unchanged=20, untied_head=True)
    assert c.kind == "head_only"


def test_classify_unchanged():
    assert classify_delta_map([], n_unchanged=20).kind == "unchanged"


def test_lora_in_noise_finds_the_knee():
    # 64 gently-decaying signal values, then a flat noise floor -> sharp cliff at 64.
    torch.manual_seed(0)
    signal = torch.linspace(1.0, 0.5, 64)
    floor = 0.05 + 0.002 * torch.rand(600)
    s = torch.cat([signal, floor])
    d = detect_signal_knee(s)
    assert d.is_low_rank and d.reason == "knee"
    assert d.knee_rank == 64
    assert d.gap_ratio > 5.0          # 0.5 -> 0.05 cliff
    assert d.floor_ratio < 2.5        # cliff lands on the flat floor


def test_dense_graded_spectrum_has_no_knee():
    # smooth exponential decay, no cliff anywhere.
    s = torch.exp(-torch.linspace(0.0, 2.3, 600))     # 1.0 -> 0.1, adjacent ratio ~1.004
    d = detect_signal_knee(s)
    assert not d.is_low_rank
    assert d.reason == "no_cliff"
    assert d.knee_rank is None


def test_cliff_onto_heavy_decaying_tail_is_rejected():
    # a real cliff, but the tail is heavy (holds most energy) and keeps decaying
    # -> neither a flat floor nor a negligible tail -> graded_tail (dense).
    head = torch.full((4,), 1.0)
    tail = torch.logspace(-0.3, -3.0, 600)            # 0.5 -> 0.001, heavy + graded
    s = torch.cat([head, tail])
    d = detect_signal_knee(s)
    assert not d.is_low_rank
    assert d.reason == "graded_tail"
    assert d.energy_at_knee < 0.85


def test_knee_rank_tracks_the_trained_rank():
    for r in (16, 64, 128):
        signal = torch.linspace(1.0, 0.4, r)
        floor = torch.full((800,), 0.03)
        d = detect_signal_knee(torch.cat([signal, floor]))
        assert d.is_low_rank and d.knee_rank == r


def test_moderate_cliff_onto_flat_floor_is_still_a_knee():
    # MeroMero late layers can show only a ~1.5x adjacent edge, but the
    # post-edge floor is flat. That should still route to LoRA instead of
    # chasing Frobenius error through thousands of bf16 floor modes.
    signal = torch.linspace(1.0, 0.3, 64)
    floor = torch.full((600,), 0.2)

    d = detect_signal_knee(torch.cat([signal, floor]))

    assert d.is_low_rank
    assert d.reason == "knee"
    assert d.knee_rank == 64


def test_rank1_spike_is_low_rank_via_energy_even_without_flat_floor():
    # heretic-style abliteration: one dominant direction on a graded shoulder
    # (NOT a flat bf16 floor). The energy path must still route it low-rank.
    s = torch.cat([torch.tensor([15.0]), torch.logspace(-0.4, -2.4, 500)])
    d = detect_signal_knee(s, max_rank=512)
    assert d.is_low_rank and d.knee_rank == 1
    assert d.floor_ratio > 3.0          # tail is graded, not a flat floor
    assert d.energy_at_knee >= 0.9      # but rank-1 already holds the energy


def test_heavy_graded_tail_is_not_low_rank():
    # a cliff, but the tail holds most of the energy and is not flat -> dense.
    s = torch.cat([torch.full((2,), 1.0), torch.logspace(-0.3, -3.0, 600)])
    d = detect_signal_knee(s, max_rank=512)
    assert not d.is_low_rank and d.reason == "graded_tail"


def test_empty_and_degenerate_spectra():
    assert detect_signal_knee(torch.tensor([])).reason == "empty"
    assert detect_signal_knee(torch.tensor([1.0])).reason == "empty"
