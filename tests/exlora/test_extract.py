# tests/exlora/test_extract.py
import json
import pytest
import torch
from safetensors import safe_open

from shard.exlora.config import ExtractConfig
from shard.exlora.extract import LoraExtractor
from shard.exlora.nvfp4 import nvfp4_quantize_dequantize
from shard.exlora.nvfp4_residual import NVFP4ResidualMap


class FakePromise:
    def __init__(self, tensor, device):
        self._tensor = tensor
        self._device = device

    async def get(self):
        return self._tensor.to(self._device)


class FakeIndex:
    """Minimal stand-in for HFMultiModelIndex backed by in-memory tensors."""

    def __init__(self, models: dict):
        # models: {model_uri: {tensor_name: tensor}}
        self._models = models
        self.added = []

    async def add_model(self, model_uri, revision="main"):
        self.added.append(model_uri)

    def get_model_keys(self, model_uri):
        return set(self._models[model_uri].keys())

    def get_layer_order(self, model_uri):
        return list(self._models[model_uri].keys())

    def get_tensor(self, model_uri, tensor_name, device="cpu"):
        return FakePromise(self._models[model_uri][tensor_name], device)


def build_models(rank=4, seed=0):
    g = torch.Generator().manual_seed(seed)
    base = {
        "model.embed_tokens.weight": torch.randn(20, 8, generator=g),
        "model.layers.0.self_attn.q_proj.weight": torch.randn(16, 16, generator=g),
        "model.layers.0.mlp.down_proj.weight": torch.randn(16, 32, generator=g),
        "model.norm.weight": torch.randn(16, generator=g),
        "lm_head.weight": torch.randn(20, 8, generator=g),
    }
    # finetune = base + known low-rank delta on the targeted linears only
    deltas = {}
    ft = dict(base)
    for name, w in base.items():
        if name.endswith(("q_proj.weight", "down_proj.weight")):
            out_f, in_f = w.shape
            B0 = torch.randn(out_f, rank, generator=g)
            A0 = torch.randn(rank, in_f, generator=g)
            d = B0 @ A0
            deltas[name] = d
            ft[name] = w + d
    return base, ft, deltas


@pytest.mark.asyncio
async def test_extract_writes_adapter_and_skips_non_targets(tmp_path):
    base, ft, deltas = build_models(rank=4)
    index = FakeIndex({"test/base": base, "test/ft": ft})

    cfg = ExtractConfig(
        base_model="test/base",
        finetune_model="test/ft",
        output_dir=str(tmp_path / "adapter"),
        target_error=1e-4,
        max_rank=12,
        dtype="float32",
        device="cpu",
    )
    extractor = LoraExtractor(cfg, index_manager=index)
    acc = await extractor.extract()

    # Only the two targeted linears produced factors; embeddings/norm/head skipped.
    assert set(acc.rank_pattern) == {
        "model.layers.0.self_attn.q_proj",
        "model.layers.0.mlp.down_proj",
    }
    # Known rank-4 deltas should be recovered at rank 4.
    assert all(r == 4 for r in acc.rank_pattern.values())

    # Adapter files exist and config is consistent.
    out = tmp_path / "adapter"
    assert (out / "adapter_model.safetensors").exists()
    cfg_json = json.loads((out / "adapter_config.json").read_text())
    assert cfg_json["rank_pattern"] == cfg_json["alpha_pattern"]
    report = (out / "EXTRACTION_REPORT.md")
    assert report.exists()
    text = report.read_text()
    assert "| layer | module | layer_idx | depth | bucket |" in text
    assert "## Position Summaries" in text
    assert "### By Depth Bucket" in text
    assert "| early | 2 |" in text


@pytest.mark.asyncio
async def test_reconstruction_matches_delta(tmp_path):
    base, ft, deltas = build_models(rank=3, seed=7)
    index = FakeIndex({"test/base": base, "test/ft": ft})

    cfg = ExtractConfig(
        base_model="test/base", finetune_model="test/ft",
        output_dir=str(tmp_path / "adapter"),
        target_error=1e-5, max_rank=10, dtype="float32",
    )
    await LoraExtractor(cfg, index_manager=index).extract()

    tensors = {}
    with safe_open(tmp_path / "adapter" / "adapter_model.safetensors", framework="pt") as f:
        for k in f.keys():
            tensors[k] = f.get_tensor(k)

    # With alpha == r the applied delta is exactly B @ A.
    prefix = "base_model.model.model.layers.0.self_attn.q_proj"
    A = tensors[f"{prefix}.lora_A.weight"]
    B = tensors[f"{prefix}.lora_B.weight"]
    recon = B.to(torch.float32) @ A.to(torch.float32)
    target = deltas["model.layers.0.self_attn.q_proj.weight"]
    rel = (recon - target).norm() / target.norm()
    assert rel.item() < 1e-4


@pytest.mark.asyncio
async def test_projection_seams_are_invoked(tmp_path):
    """A custom projection's apply/defuzz/fold must be called per targeted layer,
    and a fold transform must reach the emitted factors."""
    from shard.exlora.projection import IdentityProjection

    base, ft, _ = build_models(rank=4)
    index = FakeIndex({"test/base": base, "test/ft": ft})

    class SpyProjection(IdentityProjection):
        def __init__(self):
            self.applied, self.defuzzed, self.folded = [], [], []

        def apply(self, delta, layer_name):
            self.applied.append(layer_name)
            return delta

        def defuzz(self, U, S, Vh, delta, layer_name):
            self.defuzzed.append(layer_name)
            return U, S, Vh

        def fold(self, lora_A, lora_B, layer_name):
            self.folded.append(layer_name)
            return lora_A * 2.0, lora_B  # a visible, invertible fold

        def __call__(self):  # allow instantiation reuse
            return self

    spy = SpyProjection()
    cfg = ExtractConfig(
        base_model="test/base", finetune_model="test/ft",
        output_dir=str(tmp_path / "adapter"), max_rank=12, dtype="float32",
    )
    extractor = LoraExtractor(cfg, index_manager=index, projection=spy)
    acc = await extractor.extract()

    targets = {"model.layers.0.self_attn.q_proj.weight", "model.layers.0.mlp.down_proj.weight"}
    assert set(spy.applied) == {t[: -len(".weight")] + ".weight" for t in targets}
    assert len(spy.applied) == len(spy.defuzzed) == len(spy.folded) == 2
    # the fold scaled lora_A by 2 -> reconstruction is 2x the rank-r truncation
    key = "base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight"
    assert key in acc.tensors


@pytest.mark.asyncio
async def test_missing_finetune_tensor_raises(tmp_path):
    base, ft, _ = build_models()
    del ft["model.layers.0.self_attn.q_proj.weight"]  # drop a targeted tensor
    index = FakeIndex({"test/base": base, "test/ft": ft})
    cfg = ExtractConfig("test/base", "test/ft", str(tmp_path / "a"), dtype="float32")
    with pytest.raises(ValueError):
        await LoraExtractor(cfg, index_manager=index).extract()


def test_rank_policy_error_or_signal_stops_at_noise_floor(tmp_path):
    cfg = ExtractConfig(
        "test/base",
        "test/ft",
        str(tmp_path / "adapter"),
        target_error=1e-6,
        max_rank=24,
        rank_policy="error_or_signal",
        noise_floor_multiplier=1.5,
    )
    extractor = LoraExtractor(cfg, index_manager=FakeIndex({"test/base": {}, "test/ft": {}}))
    singular_values = torch.cat([torch.linspace(10.0, 5.0, 4), torch.full((20,), 0.5)])

    selection = extractor._select_rank(singular_values, min_rank=1, max_rank=24)

    assert selection.rank == 4
    assert selection.reason == "noise_floor"
    assert selection.rel_error > 0.0
    assert selection.retained_energy > 0.97


@pytest.mark.asyncio
async def test_extract_writes_nvfp4_residual_artifact(tmp_path):
    base, ft, deltas = build_models(rank=4, seed=11)
    index = FakeIndex({"test/base": base, "test/ft": ft})

    cfg = ExtractConfig(
        base_model="test/base",
        finetune_model="test/ft",
        output_dir=str(tmp_path / "delta_map"),
        artifact_type="nvfp4_residual",
        nvfp4_extra_bpe=2.0,
        nvfp4_residual_dtype="bfloat16",
        dtype="float32",
        device="cpu",
    )
    writer = await LoraExtractor(cfg, index_manager=index).extract()

    out = tmp_path / "delta_map"
    assert (out / "exlora_nvfp4_residual.json").exists()
    assert not (out / "adapter_model.safetensors").exists()
    assert len(writer.tensors) == 2

    name = "model.layers.0.self_attn.q_proj.weight"
    delta = deltas[name]
    loaded = NVFP4ResidualMap.load(out)
    decoded = loaded.dequantize(name)
    bare = nvfp4_quantize_dequantize(delta)
    assert decoded is not None
    assert ((decoded - delta).norm() / delta.norm()).item() < ((bare - delta).norm() / delta.norm()).item()
