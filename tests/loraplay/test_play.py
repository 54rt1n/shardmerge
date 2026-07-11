# tests/loraplay/test_play.py
import pytest
import torch

from shard.exlora.peft import AdapterAccumulator
from shard.loraplay.play import score_adapter, write_report


class FakePromise:
    def __init__(self, t, device):
        self._t, self._device = t, device

    async def get(self):
        return self._t.to(self._device)


class FakeIndex:
    def __init__(self, models):
        self._models = models

    async def add_model(self, uri, revision="main"):
        pass

    def get_tensor(self, uri, name, device="cpu"):
        return FakePromise(self._models[uri][name], device)


@pytest.mark.asyncio
async def test_score_adapter_perfect_reconstruction(tmp_path):
    # base + ft where the true delta is exactly rank-3 B@A, and the adapter
    # stores that exact factorization -> raw error ~0.
    torch.manual_seed(0)
    name = "model.layers.0.self_attn.q_proj.weight"
    base = torch.randn(8, 6)
    A = torch.randn(3, 6)
    B = torch.randn(8, 3)
    delta = B @ A
    ft = base + delta

    acc = AdapterAccumulator(base_model="test/base", target_modules=["q_proj"])
    acc.add(name, A, B, rank=3)
    acc.write(tmp_path)

    index = FakeIndex({"test/base": {name: base}, "test/ft": {name: ft}})
    report = await score_adapter("test/base", "test/ft", tmp_path, index_manager=index)

    assert len(report.layers) == 1
    s = report.layers[0]
    assert s.layer_name == name
    assert s.rank == 3
    assert s.raw_rel_error == pytest.approx(0.0, abs=1e-3)
    assert s.act_rel_error is None  # no grams supplied


@pytest.mark.asyncio
async def test_score_adapter_with_grams(tmp_path):
    name = "model.layers.0.self_attn.q_proj.weight"
    base = torch.zeros(8, 6)
    A, B = torch.randn(2, 6), torch.randn(8, 2)
    ft = base + B @ A
    acc = AdapterAccumulator(base_model="test/base", target_modules=["q_proj"])
    acc.add(name, A * 0.9, B, rank=2)  # imperfect -> nonzero error
    acc.write(tmp_path)

    H = torch.randn(6, 10) @ torch.randn(10, 6)
    grams = {name: H @ H.T}
    index = FakeIndex({"test/base": {name: base}, "test/ft": {name: ft}})
    report = await score_adapter("test/base", "test/ft", tmp_path, grams=grams, index_manager=index)
    s = report.layers[0]
    assert s.act_rel_error is not None and s.act_rel_error > 0

    out = tmp_path / "SCORE_REPORT.md"
    write_report(report, out, "test/base", "test/ft", tmp_path, grams_used=True)
    assert "LoRAplay Score Report" in out.read_text()
