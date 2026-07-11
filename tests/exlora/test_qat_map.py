# tests/exlora/test_qat_map.py
import pytest
import torch

from shard.exlora.config import ExtractConfig
from shard.exlora.qat_map import (
    QATMap, QATMapEntry, tensor_map_entry, build_qat_map,
    SensitivityWeightedProjection, pack_a_qat,
)


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

    def get_model_keys(self, uri):
        return set(self._models[uri])

    def get_layer_order(self, uri):
        return list(self._models[uri])

    def get_tensor(self, uri, name, device="cpu"):
        return FakePromise(self._models[uri][name], device)


class TestTensorMapEntry:
    def test_packs_per_parameter_and_decodes(self):
        torch.manual_seed(0)
        a = torch.randn(16, 32)
        e = tensor_map_entry(a, torch.ones(16, 32))
        assert e.packed.shape == a.shape
        recon = e.packed.dequantize()
        rel = (recon - a).norm() / a.norm()
        assert rel.item() < 0.15  # NVFP4 round-trip

    def test_diagnostics(self):
        w = torch.randn(8, 6)
        a = 0.01 * w
        e = tensor_map_entry(a, w)
        assert e.rel_norm == pytest.approx(0.01, rel=1e-3)
        assert e.corr_weight > 0.99
        assert e.col_norm.shape == (6,) and e.row_norm.shape == (8,)

    def test_rejects_non_2d(self):
        with pytest.raises(ValueError):
            tensor_map_entry(torch.randn(4), torch.randn(4))


class TestQATMapCache:
    def _map(self):
        a = torch.randn(16, 32)
        e = tensor_map_entry(a, torch.ones(16, 32))
        return QATMap("b", "q", {"L.weight": e}), a

    def test_save_load_decode_roundtrip(self, tmp_path):
        qmap, a = self._map()
        qmap.save(tmp_path)
        loaded = QATMap.load(tmp_path)
        assert "L.weight" in loaded
        recon = loaded.a_qat("L.weight")
        rel = (recon - a).norm() / a.norm()
        assert rel.item() < 0.15
        assert loaded.base_model == "b"

    def test_sensitivity_shape_and_norm(self):
        qmap, a = self._map()
        w = qmap.sensitivity("L.weight", power=1.0, normalize=True)
        assert w.shape == a.shape
        assert w.mean().item() == pytest.approx(1.0, rel=1e-4)

    def test_missing_is_none(self):
        qmap, _ = self._map()
        assert qmap.a_qat("nope") is None
        assert qmap.sensitivity("nope") is None


@pytest.mark.asyncio
async def test_build_qat_map_streaming(tmp_path):
    base = {"model.layers.0.self_attn.q_proj.weight": torch.randn(8, 16),
            "model.layers.0.mlp.down_proj.weight": torch.randn(16, 32)}
    qat = {k: v + 0.05 * torch.randn_like(v) for k, v in base.items()}
    index = FakeIndex({"b": base, "q": qat})
    cfg = ExtractConfig("b", "q", str(tmp_path / "map"), device="cpu")
    qmap = await build_qat_map(cfg, index_manager=index)
    assert set(qmap.entries) == set(base)
    reloaded = QATMap.load(tmp_path / "map")
    assert set(reloaded.entries) == set(base)
    # decoded map approximates the true A_qat
    a_true = qat["model.layers.0.mlp.down_proj.weight"] - base["model.layers.0.mlp.down_proj.weight"]
    recon = reloaded.a_qat("model.layers.0.mlp.down_proj.weight")
    assert ((recon - a_true).norm() / a_true.norm()).item() < 0.2


class TestSensitivityWeightedProjection:
    def test_apply_inverse_roundtrips(self):
        a = torch.randn(10, 6)
        qmap = QATMap("b", "q", {"L.weight": tensor_map_entry(a, torch.ones(10, 6))})
        proj = SensitivityWeightedProjection(qmap, power=0.5)
        delta = torch.randn(10, 6)
        z = proj.apply(delta, "L.weight")
        back = proj.inverse(z, "L.weight")
        assert torch.allclose(back, delta, atol=1e-4)
        # the projection actually reweighted (not identity)
        assert not torch.allclose(z, delta, atol=1e-3)

    def test_missing_layer_is_passthrough(self):
        proj = SensitivityWeightedProjection(QATMap("b", "q", {}))
        d = torch.randn(4, 4)
        assert torch.allclose(proj.apply(d, "L.weight"), d)
        assert torch.allclose(proj.inverse(d, "L.weight"), d)
