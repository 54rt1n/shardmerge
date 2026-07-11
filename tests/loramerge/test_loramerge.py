import pytest
import torch

import shard.loramerge.merge as merge_mod
from shard.exlora.nvfp4_residual import NVFP4ResidualWriter, build_nvfp4_residual_entry
from shard.exlora.peft import AdapterAccumulator
from shard.loramerge import (
    DenseDeltaSource,
    NVFP4ResidualDeltaSource,
    PeftDeltaSource,
    WeightedDeltaSource,
    fourier_slerp_merge,
    load_delta_source,
    merge_full_frame_tensor,
    norm_aware_slerp,
    norm_slerp_merge,
    sign_agreement_vote,
)


LAYER = "model.layers.0.self_attn.q_proj.weight"


def test_peft_delta_source_materializes_adapter_delta(tmp_path):
    acc = AdapterAccumulator(base_model="base", target_modules=["q_proj"])
    lora_a = torch.tensor([[1.0, 2.0, 3.0]])
    lora_b = torch.tensor([[4.0], [5.0]])
    acc.add(LAYER, lora_a, lora_b, rank=1)
    acc.write(tmp_path)

    source = PeftDeltaSource.from_path(tmp_path)

    assert source.tensor_names() == {LAYER}
    assert torch.equal(source.delta(LAYER), lora_b @ lora_a)
    assert source.delta("missing.weight") is None


def test_load_delta_source_detects_peft(tmp_path):
    acc = AdapterAccumulator(base_model="base", target_modules=["q_proj"])
    acc.add(LAYER, torch.ones(1, 2), torch.ones(3, 1), rank=1)
    acc.write(tmp_path)

    source = load_delta_source(tmp_path)

    assert isinstance(source, PeftDeltaSource)


def test_nvfp4_residual_delta_source_reads_dense_artifact(tmp_path):
    torch.manual_seed(3)
    delta = torch.randn(8, 16)
    entry = build_nvfp4_residual_entry(delta, extra_bpe=2.0, index_bits=32)

    writer = NVFP4ResidualWriter(
        tmp_path,
        base_model="base",
        finetune_model="ft",
        target_modules=["q_proj"],
        extra_bpe=2.0,
        residual_dtype=torch.bfloat16,
        index_bits=32,
    )
    writer.add(LAYER, entry)
    writer.write_manifest()

    source = load_delta_source(tmp_path)
    explicit = NVFP4ResidualDeltaSource.from_path(tmp_path)

    assert isinstance(source, NVFP4ResidualDeltaSource)
    assert source.tensor_names() == {LAYER}
    assert torch.allclose(source.delta(LAYER), explicit.delta(LAYER))
    assert source.delta("missing.weight") is None


def test_nvfp4_residual_source_projects_with_comb_corrector(tmp_path):
    from shard.exlora.nvfp4_residual import NVFP4ResidualMap

    torch.manual_seed(5)
    delta = torch.randn(8, 16)
    entry = build_nvfp4_residual_entry(delta, extra_bpe=2.0, index_bits=32)
    writer = NVFP4ResidualWriter(tmp_path, base_model="base", finetune_model="ft",
                                 target_modules=["q_proj"], extra_bpe=2.0,
                                 residual_dtype=torch.bfloat16, index_bits=32)
    writer.add(LAYER, entry)
    writer.write_manifest()

    # Gram-fit comb corrector stored alongside the dense map in comb/.
    a, b = torch.randn(2, 16), torch.randn(8, 2)
    acc = AdapterAccumulator(base_model="base", target_modules=["q_proj"])
    acc.add(LAYER, a, b, rank=2)
    acc.write(tmp_path / "comb")

    src = NVFP4ResidualDeltaSource.from_path(tmp_path)
    assert src.comb is not None
    dense = NVFP4ResidualMap.load(tmp_path).dequantize(LAYER)
    assert torch.allclose(src.delta(LAYER, dtype=torch.float32), dense + b @ a, atol=1e-4)


def test_sign_agreement_vote_keeps_majority_signs_and_drops_ties():
    d1 = torch.tensor([1.0, 1.0, -1.0, -1.0, 1.0])
    d2 = torch.tensor([2.0, -2.0, -2.0, 2.0, -2.0])
    d3 = torch.tensor([3.0, 3.0, 3.0, -3.0, 0.0])

    result = sign_agreement_vote([d1, d2, d3])

    assert torch.equal(result, torch.tensor([6.0, 4.0, -3.0, -4.0, 0.0]))
    assert torch.equal(sign_agreement_vote([torch.tensor([1.0]), torch.tensor([-2.0])]), torch.tensor([0.0]))


def test_sign_agreement_vote_applies_weights_before_voting():
    result = sign_agreement_vote(
        [torch.tensor([2.0, 2.0]), torch.tensor([4.0, -4.0])],
        weights=[0.5, 1.0],
    )

    assert torch.equal(result, torch.tensor([5.0, -4.0]))


def test_norm_aware_slerp_interpolates_direction_and_norm():
    a = torch.tensor([2.0, 0.0])
    b = torch.tensor([0.0, 4.0])

    result = norm_aware_slerp(a, b, t=0.5)

    assert result.norm().item() == pytest.approx(3.0, rel=1e-6)
    assert torch.allclose(result, torch.tensor([3.0 / 2**0.5, 3.0 / 2**0.5]), atol=1e-6)


def test_norm_slerp_merge_rejects_negative_weights():
    with pytest.raises(ValueError, match="non-negative"):
        norm_slerp_merge([torch.ones(2), torch.ones(2)], weights=[1.0, -1.0])


def test_fourier_slerp_merge_uses_existing_fft_slerp_path():
    torch.manual_seed(8)
    a = torch.randn(8, 8)
    b = torch.randn(8, 8)

    result = fourier_slerp_merge([a, b], weights=[1.0, 1.0], device="cpu")

    assert result.shape == a.shape
    assert result.dtype == torch.float32
    assert not torch.isnan(result).any()
    assert not torch.isinf(result).any()
    assert result.norm().item() == pytest.approx((a.norm().item() + b.norm().item()) / 2.0, rel=1e-4)


def test_fourier_slerp_merge_rejects_negative_weights():
    with pytest.raises(ValueError, match="non-negative"):
        fourier_slerp_merge([torch.ones(4, 4), torch.ones(4, 4)], weights=[1.0, -1.0])


def test_merge_full_frame_tensor_uses_sources_in_fp32():
    base = torch.ones(2, 2, dtype=torch.bfloat16)
    src1 = DenseDeltaSource({LAYER: torch.tensor([[1.0, 1.0], [-1.0, -1.0]])}, name="a")
    src2 = DenseDeltaSource({LAYER: torch.tensor([[2.0, -2.0], [-3.0, 3.0]])}, name="b")
    sources = [
        WeightedDeltaSource(src1, weight=1.0),
        WeightedDeltaSource(src2, weight=1.0),
    ]

    merged = merge_full_frame_tensor(base, sources, LAYER, rule="sign_vote")

    expected_delta = torch.tensor([[3.0, 0.0], [-4.0, 0.0]])
    assert merged.dtype == torch.float32
    assert torch.equal(merged, base.to(torch.float32) + expected_delta)


def test_merge_full_frame_tensor_supports_fourier_rule():
    torch.manual_seed(9)
    base = torch.zeros(8, 8, dtype=torch.bfloat16)
    src1 = DenseDeltaSource({LAYER: torch.randn(8, 8)}, name="a")
    src2 = DenseDeltaSource({LAYER: torch.randn(8, 8)}, name="b")
    sources = [WeightedDeltaSource(src1), WeightedDeltaSource(src2)]

    merged = merge_full_frame_tensor(base, sources, LAYER, rule="fourier_slerp")

    assert merged.shape == base.shape
    assert merged.dtype == torch.float32
    assert not torch.isnan(merged).any()
    assert not torch.isinf(merged).any()


def test_merge_full_frame_tensor_returns_base_when_no_source_has_tensor():
    base = torch.ones(2, 2, dtype=torch.bfloat16)
    src = DenseDeltaSource({"other.weight": torch.ones(2, 2)})

    merged = merge_full_frame_tensor(base, [WeightedDeltaSource(src)], LAYER)

    assert torch.equal(merged, base.to(torch.float32))


def test_isometric_merge_ignores_legacy_group_without_pool(monkeypatch):
    def fail_if_called(*args, **kwargs):
        raise AssertionError("legacy group must not opt into pooled isometry")

    monkeypatch.setattr(merge_mod, "isometric_pooled_merge_delta", fail_if_called)
    torch.manual_seed(13)
    base = torch.randn(6, 6)
    src1 = DenseDeltaSource({LAYER: 0.05 * torch.randn(6, 6)}, name="a")
    src2 = DenseDeltaSource({LAYER: 0.05 * torch.randn(6, 6)}, name="b")
    sources = [
        WeightedDeltaSource(src1, group="legacy", priority=2.0),
        WeightedDeltaSource(src2, group="legacy", priority=1.0),
    ]

    merged = merge_full_frame_tensor(base, sources, LAYER, rule="isometric", isometric_rank=2)

    assert merged.shape == base.shape
    assert merged.dtype == torch.float32
    assert not torch.isnan(merged).any()


def test_isometric_merge_uses_explicit_pool_field(monkeypatch):
    calls = {}

    def fake_pooled(base, deltas, weights, pools, priorities,
                    pool_ranks=None, pool_energies=None, pool_residuals=None, **kwargs):
        calls["pools"] = list(pools)
        calls["priorities"] = list(priorities)
        calls["pool_ranks"] = list(pool_ranks)
        calls["pool_energies"] = list(pool_energies)
        calls["pool_residuals"] = list(pool_residuals)
        calls["rank"] = kwargs["rank"]
        return torch.full_like(base, 0.25)

    monkeypatch.setattr(merge_mod, "isometric_pooled_merge_delta", fake_pooled)
    base = torch.zeros(2, 2)
    src = DenseDeltaSource({LAYER: torch.ones(2, 2)}, name="a")
    sources = [
        WeightedDeltaSource(
            src,
            group="legacy",
            pool="reasoning",
            pool_rank=7,
            pool_energy=0.3,
            pool_residual=0.8,
            priority=3.0,
        ),
    ]

    merged = merge_full_frame_tensor(base, sources, LAYER, rule="isometric", isometric_rank=4)

    assert torch.equal(merged, torch.full_like(base, 0.25))
    assert calls == {
        "pools": ["reasoning"],
        "priorities": [3.0],
        "pool_ranks": [7],
        "pool_energies": [0.3],
        "pool_residuals": [0.8],
        "rank": 4,
    }


def test_isometric_merge_applies_post_overlay_after_pooled_body(monkeypatch):
    calls = {}

    def fake_pooled(base, deltas, weights, pools, priorities, *args, **kwargs):
        calls["count"] = len(deltas)
        calls["pools"] = list(pools)
        return torch.full_like(base, 0.25)

    monkeypatch.setattr(merge_mod, "isometric_pooled_merge_delta", fake_pooled)
    base = torch.zeros(2, 2)
    body = DenseDeltaSource({LAYER: torch.ones(2, 2)}, name="body")
    patch = DenseDeltaSource({LAYER: torch.full((2, 2), 2.0)}, name="patch")
    sources = [
        WeightedDeltaSource(body, pool="reasoning"),
        WeightedDeltaSource(patch, weight=0.5, post_merge=True),
    ]

    merged = merge_full_frame_tensor(
        base,
        sources,
        LAYER,
        rule="isometric",
        final_merge_t=0.5,
    )

    assert calls == {"count": 1, "pools": ["reasoning"]}
    assert torch.equal(merged, torch.full_like(base, 0.75))


def test_isometric_merge_applies_only_post_source_as_direct_patch():
    base = torch.ones(2, 2)
    patch = DenseDeltaSource({LAYER: torch.full((2, 2), 0.125)}, name="patch")

    merged = merge_full_frame_tensor(
        base,
        [WeightedDeltaSource(patch, post_merge=True)],
        LAYER,
        rule="isometric",
        final_merge_t=1.0,
    )

    assert torch.equal(merged, torch.full_like(base, 1.125))


def test_pooled_isometric_energy_zero_preserves_weight_norm():
    torch.manual_seed(17)
    base = torch.randn(10, 7)
    deltas = [0.03 * torch.randn(10, 7) for _ in range(3)]

    merged_delta = merge_mod.isometric_pooled_merge_delta(
        base,
        deltas,
        weights=[1.0, 1.0, 1.0],
        pools=["reasoning", "agentic", "agentic"],
        priorities=[3.0, 2.0, 2.0],
        rank=3,
        energy=0.0,
    )

    assert torch.allclose((base + merged_delta).norm(), base.norm(), rtol=1e-5, atol=1e-5)
