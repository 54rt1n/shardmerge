# tests/exlora/test_projection.py
import pytest
import torch

from shard.exlora.config import ExtractConfig
from shard.exlora.projection import (
    IdentityProjection,
    Projection,
    QATProjectionConfig,
    QATProjectionProtocol,
    RightMetricQATProjection,
    TensorProjectionContext,
    load_projection,
)
from shard.exlora.functions import split_svd_to_lora
from shard.exlora.svd import basis_svd


class TestIdentityProjection:
    def test_apply_is_passthrough(self):
        x = torch.randn(8, 6)
        assert torch.equal(IdentityProjection().apply(x, "layer"), x)

    def test_defuzz_is_passthrough(self):
        x = torch.randn(8, 6)
        U, S, Vh = basis_svd(x)
        out = IdentityProjection().defuzz(U, S, Vh, x, "layer")
        assert out == (U, S, Vh)

    def test_fold_is_passthrough(self):
        a, b = torch.randn(4, 6), torch.randn(8, 4)
        assert IdentityProjection().fold(a, b, "layer") == (a, b)

    def test_conforms_to_protocol(self):
        assert isinstance(IdentityProjection(), QATProjectionProtocol)

    def test_prepare_stores_context(self):
        proj = IdentityProjection()
        context = TensorProjectionContext(layer_name="layer", delta_shape=(8, 6))
        proj.prepare(context)
        assert proj.context is context

    def test_prepare_rejects_mismatched_tensor_context(self):
        proj = IdentityProjection()
        context = TensorProjectionContext(
            layer_name="layer",
            delta_shape=(8, 6),
            base_tensor=torch.randn(7, 6),
        )
        with pytest.raises(ValueError, match="base_tensor shape"):
            proj.prepare(context)

    def test_apply_rejects_non_2d(self):
        with pytest.raises(ValueError, match="must be 2D"):
            IdentityProjection().apply(torch.randn(2, 3, 4), "layer")


class TestLoadProjection:
    def test_default_is_identity(self):
        cfg = ExtractConfig("base", "ft", "out")
        assert isinstance(load_projection(cfg), IdentityProjection)

    def test_projection_model_without_type_is_identity(self):
        cfg = ExtractConfig("base", "ft", "out", qat_projection_model="some/map")
        # projection_type defaults to "none" -> still raw SVD
        assert isinstance(load_projection(cfg), IdentityProjection)

    def test_configured_projection_not_implemented_yet(self):
        cfg = ExtractConfig(
            "base", "ft", "out",
            qat_projection_model="some/map", projection_type="reflection",
        )
        with pytest.raises(NotImplementedError, match="not implemented"):
            load_projection(cfg)


class TestRightMetricQATProjection:
    def test_conforms_to_protocol(self):
        metric = torch.eye(6)
        assert isinstance(RightMetricQATProjection(metric), QATProjectionProtocol)

    def test_full_rank_roundtrip_with_metric_fold(self):
        delta = torch.randn(8, 6)
        metric = torch.diag(torch.linspace(0.5, 2.0, 6))
        proj = RightMetricQATProjection(
            metric,
            config=QATProjectionConfig(projection_type="metric", damping=0.0),
        )
        proj.prepare(TensorProjectionContext(layer_name="layer", delta_shape=tuple(delta.shape)))

        z = proj.apply(delta, "layer")
        U, S, Vh = basis_svd(z)
        lora_A, lora_B = split_svd_to_lora(U, S, Vh, rank=6, out_dtype=torch.float32)
        lora_A, lora_B = proj.fold(lora_A, lora_B, "layer")
        recon = lora_B @ lora_A

        assert torch.allclose(recon, delta, atol=2e-4, rtol=2e-4)

    def test_metric_can_arrive_from_context(self):
        metric = torch.eye(6)
        proj = RightMetricQATProjection()
        proj.prepare(TensorProjectionContext(
            layer_name="layer",
            delta_shape=(8, 6),
            metric_sqrt=metric,
        ))
        assert proj.metric_sqrt is metric

    def test_missing_metric_raises(self):
        proj = RightMetricQATProjection()
        with pytest.raises(RuntimeError, match="missing metric_sqrt"):
            proj.prepare(TensorProjectionContext(layer_name="layer", delta_shape=(8, 6)))

    def test_bad_metric_shape_raises(self):
        proj = RightMetricQATProjection(torch.eye(5))
        with pytest.raises(ValueError, match="metric_sqrt shape"):
            proj.prepare(TensorProjectionContext(layer_name="layer", delta_shape=(8, 6)))


class TestBasisSVD:
    def test_reconstructs_matrix(self):
        x = torch.randn(10, 7)
        U, S, Vh = basis_svd(x)
        assert U.shape == (10, 7) and S.shape == (7,) and Vh.shape == (7, 7)
        recon = (U * S.unsqueeze(0)) @ Vh
        assert torch.allclose(recon, x, atol=1e-4)

    def test_rejects_non_2d(self):
        with pytest.raises(ValueError):
            basis_svd(torch.randn(5))


def test_projection_is_abstract():
    with pytest.raises(TypeError):
        Projection()  # abstract methods not implemented
