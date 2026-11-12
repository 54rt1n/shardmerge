#!/usr/bin/env python3
# scripts/gguf_to_safetensors.py
# Copyright (C) 2026 Martin Bukowski
#
# This software is free software: you can redistribute it and/or
# modify it under the terms of the GNU Lesser General Public License as
# published by the Free Software Foundation, either version 3 of the
# License, or (at your option) any later version.
#
# This software is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
# Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with this program. If not, see http://www.gnu.org/licenses/.

"""Convert a ``qwen35``-architecture GGUF into an HF safetensors directory.

Dequantizes, renames to the HuggingFace layout, and — critically — undoes the
delta-net layout and value transforms that llama.cpp applies during conversion.

Two corrections are required, both verified against the HF base before use:

**Value-head regrouping.** Qwen3.6 has 48 linear-attention value heads grouped
against 16 key heads. llama.cpp emits them grouped the other way, so every
tensor indexed by value head is block-permuted by
``base_block = (i % 16) * 3 + i // 16`` over 128-row head blocks. Without this,
``in_proj_z``, ``out_proj``, ``in_proj_a/b``, ``conv1d`` and the ``v`` span of
``in_proj_qkv`` come out looking *unrelated* to base (relative delta ~1.38,
i.e. sqrt(2) — two orthogonal matrices) while every shape still matches, so
nothing downstream would flag it. Query/key tensors are NOT permuted.

**A_log.** llama.cpp stores ``ssm_a = -exp(A_log)`` (that is, ``A`` itself), so
the inverse is ``A_log = log(-a)``, applied after the head permutation.

The GGUF carries the text decoder only: no vision tower and no MTP head.

    uv run python scripts/gguf_to_safetensors.py \\
        --gguf model-Q8_0.gguf --base models/qwen36/Qwen/Qwen3.6-27B \\
        --out models/qwen36/gguf/my-model
"""

import argparse
import json
import sys
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

# Layer-local GGUF stem -> HF module path.
LAYER_MAP = {
    "attn_norm": "input_layernorm",
    "post_attention_norm": "post_attention_layernorm",
    "ffn_gate": "mlp.gate_proj",
    "ffn_up": "mlp.up_proj",
    "ffn_down": "mlp.down_proj",
    "attn_q": "self_attn.q_proj",
    "attn_k": "self_attn.k_proj",
    "attn_v": "self_attn.v_proj",
    "attn_output": "self_attn.o_proj",
    "attn_q_norm": "self_attn.q_norm",
    "attn_k_norm": "self_attn.k_norm",
    "attn_qkv": "linear_attn.in_proj_qkv",
    "attn_gate": "linear_attn.in_proj_z",
    "ssm_alpha": "linear_attn.in_proj_a",
    "ssm_beta": "linear_attn.in_proj_b",
    "ssm_out": "linear_attn.out_proj",
    "ssm_conv1d": "linear_attn.conv1d",
    "ssm_norm": "linear_attn.norm",
    "ssm_a": "linear_attn.A_log",
    "ssm_dt": "linear_attn.dt_bias",
}
GLOBAL_MAP = {
    "token_embd.weight": "model.language_model.embed_tokens.weight",
    "output.weight": "lm_head.weight",
    "output_norm.weight": "model.language_model.norm.weight",
}
CONFIG_FILES = [
    "config.json", "generation_config.json", "tokenizer.json", "tokenizer_config.json",
    "chat_template.jinja", "vocab.json", "merges.txt",
]


def head_permutation(num_value_heads: int, num_key_heads: int) -> list:
    """GGUF value-head block index -> HF value-head block index."""
    group = num_value_heads // num_key_heads
    return [(i % num_key_heads) * group + i // num_key_heads for i in range(num_value_heads)]


def unpermute(x: torch.Tensor, perm: list, axis: int, span: slice | None = None) -> torch.Tensor:
    """Reorder head blocks along ``axis``, optionally only within ``span``."""
    nv = len(perm)
    if span is None:
        target, prefix = x, None
    else:
        prefix, target = x[: span.start], x[span.start:]
    blocks = target.reshape(*target.shape[:axis], nv, -1, *target.shape[axis + 1:]) \
        if axis == 0 else target.reshape(*target.shape[:axis], nv, -1)
    out = torch.empty_like(blocks)
    for src, dst in enumerate(perm):
        if axis == 0:
            out[dst] = blocks[src]
        else:
            out[..., dst, :] = blocks[..., src, :]
    out = out.reshape(target.shape)
    return out if prefix is None else torch.cat([prefix, out], dim=0)


def hf_name(gguf_name: str):
    if gguf_name in GLOBAL_MAP:
        return GLOBAL_MAP[gguf_name]
    parts = gguf_name.split(".")
    if parts[0] != "blk":
        return None
    layer = parts[1]
    stem = ".".join(parts[2:])
    for suffix in (".weight", ".bias"):
        stem = stem.removesuffix(suffix)
    if stem not in LAYER_MAP:
        return None
    hf = LAYER_MAP[stem]
    tail = "" if hf.endswith(("A_log", "dt_bias")) else ".weight"
    return f"model.language_model.layers.{layer}.{hf}{tail}"


def convert_tensor(gguf_name: str, hf: str, x: torch.Tensor, perm: list, nk: int) -> torch.Tensor:
    """Apply the layout / value corrections for one tensor."""
    stem = gguf_name.split(".")[-1] if not gguf_name.startswith("blk") else ".".join(gguf_name.split(".")[2:])
    stem = stem.removesuffix(".weight").removesuffix(".bias")
    head_dim = 128

    if stem in ("attn_gate", "ssm_alpha", "ssm_beta"):
        return unpermute(x, perm, axis=0)
    if stem == "ssm_out":
        return unpermute(x, perm, axis=1)
    if stem == "attn_qkv":
        # q | k | v : only the v span is value-head indexed
        v_start = 2 * nk * head_dim
        return unpermute(x, perm, axis=0, span=slice(v_start, None))
    if stem == "ssm_conv1d":
        v_start = 2 * nk * head_dim
        fixed = unpermute(x, perm, axis=0, span=slice(v_start, None))
        return fixed.unsqueeze(1)  # HF stores conv1d as [channels, 1, kernel]
    if stem == "ssm_dt":
        return unpermute(x, perm, axis=0)
    if stem == "ssm_a":
        # llama.cpp stores A = -exp(A_log); invert after the head permutation.
        return torch.log((-unpermute(x, perm, axis=0)).clamp_min(1e-30))
    return x


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gguf", required=True)
    ap.add_argument("--base", required=True, help="HF base dir, for config files and verification")
    ap.add_argument("--out", required=True)
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--shard-bytes", type=int, default=4_500_000_000)
    ap.add_argument("--gguf-py", default="/home/martin/llm/llama.cpp/gguf-py")
    ap.add_argument("--verify", type=int, default=12, help="tensors to check against base")
    args = ap.parse_args()

    sys.path.insert(0, args.gguf_py)
    from gguf import GGUFReader
    from gguf.quants import dequantize

    base = Path(args.base)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    dtype = getattr(torch, args.dtype)

    cfg = json.loads((base / "config.json").read_text())
    tcfg = cfg.get("text_config", cfg)
    nv = tcfg["linear_num_value_heads"]
    nk = tcfg["linear_num_key_heads"]
    perm = head_permutation(nv, nk)
    print(f"value-head permutation: {nv} heads in {nk} groups -> {perm[:8]}...")

    reader = GGUFReader(args.gguf)
    base_wm = json.loads((base / "model.safetensors.index.json").read_text())["weight_map"]

    def base_tensor(name):
        if name not in base_wm:
            return None
        with safe_open(str(base / base_wm[name]), framework="pt", device="cpu") as f:
            return f.get_tensor(name).to(torch.float32)

    weight_map, total, shard, shard_bytes, shard_idx = {}, 0, {}, 0, 1
    pending, checks = [], []
    tensors = list(reader.tensors)

    def flush():
        nonlocal shard, shard_bytes, shard_idx, pending
        if not shard:
            return
        fn = f"model-{shard_idx:05d}.safetensors"
        save_file(shard, str(out / fn), metadata={"format": "pt"})
        for k in shard:
            weight_map[k] = fn
        print(f"  wrote {fn}  ({len(shard)} tensors, {shard_bytes/1e9:.2f} GB)")
        shard, shard_bytes, shard_idx = {}, 0, shard_idx + 1

    for i, t in enumerate(tensors):
        hf = hf_name(t.name)
        if hf is None:
            print(f"  !! unmapped GGUF tensor {t.name} — skipping")
            continue
        x = torch.from_numpy(dequantize(t.data, t.tensor_type).astype("float32"))
        x = convert_tensor(t.name, hf, x, perm, nk)

        if len(checks) < args.verify and x.ndim == 2 and i % max(1, len(tensors) // args.verify) == 0:
            b = base_tensor(hf)
            if b is not None and b.shape == x.shape:
                checks.append((hf, float((x - b).norm() / b.norm())))

        shard[hf] = x.to(dtype).contiguous()
        n = shard[hf].numel() * shard[hf].element_size()
        shard_bytes += n
        total += n
        if shard_bytes >= args.shard_bytes:
            flush()
    flush()

    index = {"metadata": {"total_size": total}, "weight_map": weight_map}
    (out / "model.safetensors.index.json").write_text(json.dumps(index, indent=2))

    for name in CONFIG_FILES:
        src = base / name
        if src.exists():
            (out / name).write_bytes(src.read_bytes())

    print(f"\nwrote {len(weight_map)} tensors, {total/1e9:.1f} GB to {out}")
    print("verification against base (relative delta; ~sqrt(2)=1.41 means MISALIGNED):")
    for name, rel in checks:
        flag = "  <-- SUSPECT" if rel > 0.9 else ""
        print(f"  {rel:.4f}  {name}{flag}")


if __name__ == "__main__":
    main()
