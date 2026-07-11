import math

import torch

from shard.delta import (
    DenseDelta,
    GramMetric,
    IdentityDeltaCodec,
    IndexedResidualDelta,
    LoraDelta,
    PatchManifest,
    PatchStack,
    RoundToDTypeCodec,
    TensorPatchSpec,
    TermSpec,
    apply_patch_stack,
    relative_metric_error,
)


def test_lora_delta_materializes_scaled_ba():
    a = torch.tensor([[1.0, 2.0, 3.0]])
    b = torch.tensor([[4.0], [5.0]])
    term = LoraDelta(a, b, scale=0.5)

    expected = 0.5 * (b @ a)

    assert term.rank == 1
    assert term.shape == expected.shape
    assert torch.allclose(term.materialize(), expected)


def test_indexed_residual_delta_scatters_flat_values():
    term = IndexedResidualDelta(
        shape=(2, 3),
        indices=torch.tensor([0, 4, 4]),
        values=torch.tensor([1.0, 2.0, 3.0]),
    )

    expected = torch.tensor([[1.0, 0.0, 0.0], [0.0, 5.0, 0.0]])

    assert torch.equal(term.materialize(), expected)


def test_patch_stack_applies_terms_in_common_dtype_and_device():
    base = torch.zeros(2, 2, dtype=torch.float32)
    dense = DenseDelta(torch.ones(2, 2))
    lora = LoraDelta(torch.eye(2), torch.eye(2), scale=2.0)
    residual = IndexedResidualDelta((2, 2), torch.tensor([3]), torch.tensor([5.0]))

    patched = PatchStack([dense, lora, residual]).apply(base)

    assert torch.equal(patched, torch.tensor([[3.0, 1.0], [1.0, 8.0]]))


def test_apply_patch_stack_convenience_wrapper():
    base = torch.tensor([[1.0]])
    patched = apply_patch_stack(base, [DenseDelta(torch.tensor([[2.0]]))])

    assert torch.equal(patched, torch.tensor([[3.0]]))


def test_codecs_encode_materializable_terms():
    delta = torch.tensor([[1.125, -2.5]], dtype=torch.float32)

    identity = IdentityDeltaCodec().encode(delta).materialize()
    rounded = RoundToDTypeCodec(torch.bfloat16).encode(delta).materialize()

    assert torch.equal(identity, delta)
    assert rounded.dtype == torch.float32
    assert torch.allclose(rounded, delta.to(torch.bfloat16).to(torch.float32))


def test_relative_metric_error_with_diagonal_metric():
    ref = torch.tensor([[1.0, 1.0]])
    approx = torch.tensor([[1.0, 0.0]])
    metric = torch.tensor([1.0, 0.0])

    assert relative_metric_error(ref, approx, metric) == 0.0
    assert relative_metric_error(ref, approx, None) > 0.0


def test_gram_metric_scores_right_weighted_error():
    gram = GramMetric("layer.weight", torch.diag(torch.tensor([4.0, 1.0])))
    ref = torch.tensor([[1.0, 1.0]])
    approx = torch.tensor([[0.0, 1.0]])

    assert gram.in_features == 2
    assert math.isclose(gram.relative_error(ref, approx), 2.0 / math.sqrt(5.0), rel_tol=1e-6)


def test_patch_manifest_roundtrips_dict_payload():
    manifest = PatchManifest(
        base_model="base",
        target_model="target",
        metadata={"note": "hybrid"},
        tensors=(
            TensorPatchSpec(
                tensor_name="layers.0.mlp.up_proj.weight",
                shape=(4, 3),
                terms=(
                    TermSpec("nvfp4", "tensors/up.nvfp4", {"bpe": 4.5}),
                    TermSpec("lora", "tensors/up.lora", {"rank": 64}),
                ),
            ),
        ),
    )

    restored = PatchManifest.from_dict(manifest.to_dict())

    assert restored == manifest
