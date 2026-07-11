# tests/exlora/test_knee_regression.py
# Copyright (C) 2026 Martin Bukowski
#
# Regression fixture: real singular-value spectra from four provenance-confirmed
# gemma-4-31B finetunes. Locks the knee detector's calibration against ground
# truth so threshold drift can't silently re-misroute (e.g. MeroMero as dense).

import json
from pathlib import Path

import pytest
import torch
from safetensors import safe_open

from shard.exlora.knee import detect_signal_knee, verdict_from_knees

FIX = Path(__file__).parent / "fixtures"
SPECTRA = FIX / "knee_spectra.safetensors"
LABELS = FIX / "knee_labels.json"

# Detection params validated to reproduce the provenance labels. min_rank=1 so
# rank-1 abliteration (heretic) is visible; floor_ratio_max generous because the
# energy path carries the directly-low-rank cases.
DETECT = dict(min_rank=1, max_rank=512, gap_threshold=1.6, floor_ratio_max=4.0)

pytestmark = pytest.mark.skipif(
    not SPECTRA.exists(), reason="knee fixture not built (scripts/build_knee_fixture.py)"
)


def _load():
    spectra = {}
    with safe_open(str(SPECTRA), framework="pt") as f:
        for k in f.keys():
            spectra[k] = f.get_tensor(k).float()
    by_model: dict[str, list[torch.Tensor]] = {}
    for key, s in spectra.items():
        by_model.setdefault(key.split("|")[0], []).append(s)
    return by_model, json.loads(LABELS.read_text())


def test_detector_matches_provenance():
    by_model, labels = _load()
    for tag, expected in labels.items():
        dets = [detect_signal_knee(s, **DETECT) for s in by_model[tag]]
        verdict = verdict_from_knees(dets)
        assert verdict.kind == expected["kind"], f"{tag}: {verdict}"
        if expected["kind"] == "low_rank":
            assert verdict.rank == expected["rank"], f"{tag}: {verdict}"


def test_known_lora_tensors_show_a_flat_floored_knee():
    # at least one MeroMero/Gutenberg tensor must produce a clean rank-64 knee
    by_model, _ = _load()
    for tag in ("meromero", "gutenberg"):
        knees = [detect_signal_knee(s, **DETECT) for s in by_model[tag]]
        assert any(d.is_low_rank and d.knee_rank == 64 for d in knees), tag
