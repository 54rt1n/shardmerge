# Copyright (C) 2026 Martin Bukowski
"""Harvest per-module-suffix activation Grams from base rollouts on MCQ reasoning.

Two phases, so the expensive generation runs without hook overhead and the
harvest is a single deterministic teacher-forced pass:

  A. rollout  -- base model, greedy, non-thinking (matching the ARC-C eval
                 condition), on ARC-Easy + OpenBookQA prompts. NEVER ARC-C:
                 that set stays untouched as the measurement, or the metric
                 is trained on the benchmark and the number means nothing.
                 Texts are cached as JSON so re-harvests skip generation.
  B. harvest  -- teacher-force prompt+rollout through the model with forward
                 pre-hooks on the target projections, accumulating
                 H = E[x x^T] keyed by (module suffix, in_dim). Keying by
                 suffix keeps the 48 linear-attn layers' statistics separate
                 from the 16 self-attn layers'; aggregating across layers of
                 the same suffix keeps memory at ~12 Grams instead of ~600.

Output: one safetensors file with eigen-factors F = V_r diag(sqrt(lambda_r))
per (suffix, in_dim) so H ~= F F^T, plus tr(H) and token counts -- the same
factor convention as research/scripts/harvest_domain_grams.py, which the
whitener consumers already understand.

    uv run python -m shard.hebbian.harvest --n-prompts 200
"""

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
import torch.nn as nn
from safetensors.torch import save_file
from transformers import AutoTokenizer, BitsAndBytesConfig

from shard.inference import resolve_model_class

BASE = "models/qwen36/Qwen/Qwen3.6-27B"
OUT_DIR = Path("models/qwen36/hebbian")

# Suffix set matches local/qwen36_knee_scan.py MODULES (all 2D decoder
# projections). Suffix-level keying is what separates linear_attn from
# self_attn statistics.
TARGET_SUFFIXES = (
    "self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj",
    "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj",
    "linear_attn.in_proj_qkv", "linear_attn.in_proj_a",
    "linear_attn.in_proj_b", "linear_attn.in_proj_z", "linear_attn.out_proj",
)

LETTERS = "ABCDEFGH"


def load_model():
    cls = resolve_model_class(BASE)
    model = cls.from_pretrained(
        BASE, quantization_config=BitsAndBytesConfig(load_in_8bit=True),
        device_map={"": 0}, dtype=torch.bfloat16).eval()
    # Text-only harvest: the vision tower and MTP head are ~0.9B params of
    # dead weight on a 31GB card that barely fits the LM + Gram buffers.
    for holder, attr in ((getattr(model, "model", model), "visual"),
                         (model, "mtp"), (getattr(model, "model", model), "mtp")):
        if getattr(holder, attr, None) is not None:
            setattr(holder, attr, None)
    import gc
    gc.collect()
    torch.cuda.empty_cache()
    return model, AutoTokenizer.from_pretrained(BASE)


def mcq_prompts(n: int) -> list[str]:
    """ARC-Easy + OpenBookQA formatted as answer-with-the-letter MCQ."""
    from datasets import load_dataset
    prompts = []
    for spec, q_key in (("allenai/ai2_arc:ARC-Easy", "question"),
                        ("allenai/openbookqa:main", "question_stem")):
        name, config = spec.split(":")
        ds = load_dataset(name, config, split="train")
        for row in ds.select(range(min((n + 1) // 2, len(ds)))):
            ch = row["choices"]
            lines = [f"{lab}. {txt}" for lab, txt in zip(ch["label"], ch["text"])]
            prompts.append(f"{row[q_key]}\n" + "\n".join(lines) +
                           "\nAnswer with the letter of the correct choice.")
    return prompts[:n]


class SuffixGram:
    """Accumulate H = sum x^T x keyed by (target suffix, in_dim)."""

    def __init__(self, suffixes):
        self.suffixes = tuple(suffixes)
        self.gram: dict[tuple[str, int], torch.Tensor] = {}
        self.count: dict[tuple[str, int], int] = {}
        self.handles = []

    def observe(self, suffix, x):
        flat = x.reshape(-1, x.shape[-1]).float()
        key = (suffix, flat.shape[1])
        if key not in self.gram:
            self.gram[key] = torch.zeros(
                flat.shape[1], flat.shape[1], dtype=torch.float32,
                device=flat.device)
            self.count[key] = 0
        # In-place accumulate: a fresh x^T x temp per hook call is a 100MB+
        # allocation the 31GB card cannot spare next to the model.
        self.gram[key].addmm_(flat.transpose(0, 1), flat)
        self.count[key] += flat.shape[0]

    def register(self, model):
        def make_hook(suffix):
            return lambda mod, inp: self.observe(suffix, inp[0].detach())
        for name, m in model.named_modules():
            for suffix in self.suffixes:
                if name.endswith(suffix) and isinstance(m, nn.Linear):
                    self.handles.append(m.register_forward_pre_hook(make_hook(suffix)))
                    break

    def remove(self):
        for h in self.handles:
            h.remove()


def factorize(H, rank):
    """Top-r eigen-factor F = V_r diag(sqrt(lambda_r)); H is symmetric PSD."""
    q = min(rank + 32, H.shape[0])
    U, S, _ = torch.svd_lowrank(H, q=q)
    r = min(rank, int(S.numel()))
    total = float(H.diagonal().sum().clamp_min(1e-30))
    energy = float(S[:r].sum() / total)
    F = (U[:, :r] * S[:r].clamp_min(0).sqrt().unsqueeze(0)).to(torch.float16)
    return F, S[:r].to(torch.float32), energy


def rollout(model, tok, prompts, cache: Path, max_new: int) -> list[str]:
    """Phase A: greedy non-thinking rollouts, cached as prompt+completion text."""
    if cache.exists():
        texts = json.loads(cache.read_text())
        print(f"rollouts: {len(texts)} cached at {cache}", flush=True)
        return texts
    texts = []
    for i, p in enumerate(prompts):
        enc = tok.apply_chat_template(
            [{"role": "user", "content": p}], tokenize=True,
            add_generation_prompt=True, return_tensors="pt", return_dict=True,
            enable_thinking=False)
        ids = enc["input_ids"].cuda()
        with torch.no_grad():
            out = model.generate(ids, max_new_tokens=max_new, do_sample=False,
                                 pad_token_id=tok.eos_token_id)
        texts.append(tok.decode(out[0], skip_special_tokens=False))
        if (i + 1) % 20 == 0:
            print(f"  rollout {i + 1}/{len(prompts)}", flush=True)
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(texts))
    print(f"rollouts: {len(texts)} written to {cache}", flush=True)
    return texts


def harvest(model, tok, texts, acc: SuffixGram, batch_tokens: int):
    """Phase B: one teacher-forced pass per rollout with hooks live."""
    acc.register(model)
    with torch.no_grad():
        for i, t in enumerate(texts):
            ids = tok(t, return_tensors="pt")["input_ids"][:, :batch_tokens].cuda()
            model(input_ids=ids)
            if (i + 1) % 20 == 0:
                print(f"  harvest {i + 1}/{len(texts)}", flush=True)
    acc.remove()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-prompts", type=int, default=200)
    ap.add_argument("--max-new", type=int, default=256)
    ap.add_argument("--rank", type=int, default=1024)
    ap.add_argument("--batch-tokens", type=int, default=2048)
    ap.add_argument("--suffixes", default=",".join(TARGET_SUFFIXES),
                    help="subset for this pass; Gram buffers for all 12 "
                         "suffixes at once do not fit next to the model")
    ap.add_argument("--out", default=str(OUT_DIR / "mcq_grams.safetensors"))
    ap.add_argument("--rollout-cache", default=str(OUT_DIR / "mcq_rollouts.json"))
    args = ap.parse_args()

    prompts = mcq_prompts(args.n_prompts)
    print(f"{len(prompts)} MCQ prompts (ARC-Easy + OpenBookQA train; ARC-C untouched)",
          flush=True)

    model, tok = load_model()
    texts = rollout(model, tok, prompts, Path(args.rollout_cache), args.max_new)
    torch.cuda.empty_cache()

    suffixes = [s for s in args.suffixes.split(",") if s in TARGET_SUFFIXES]
    print(f"pass over {len(suffixes)} suffixes: {','.join(suffixes)}", flush=True)
    acc = SuffixGram(suffixes)
    harvest(model, tok, texts, acc, args.batch_tokens)

    tensors, report = {}, []
    for key0 in sorted(acc.gram):
        suffix, d = key0
        # Pop off the GPU before any math: `g / count` would allocate a
        # second full-size Gram next to the model. Factorize on CPU.
        H = acc.gram.pop(key0).cpu()
        torch.cuda.empty_cache()
        H /= max(acc.count[key0], 1)
        F, lam, energy = factorize(H, args.rank)
        key = f"{suffix}|{d}"
        tensors[f"factor::{key}"] = F
        tensors[f"eigval::{key}"] = lam
        tensors[f"trace::{key}"] = torch.tensor([float(H.diagonal().sum())])
        tensors[f"tokens::{key}"] = torch.tensor([acc.count[key0]])
        report.append(f"  {suffix} (d={d}): r{lam.numel()} {energy:.3f}E "
                      f"{acc.count[key0]} tok")
        del H

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        # Merge with prior passes: keep keys this pass did not recompute.
        from safetensors import safe_open
        with safe_open(str(out), "pt") as f:
            for k in f.keys():
                if k not in tensors:
                    tensors[k] = f.get_tensor(k)
    save_file(tensors, str(out))
    print("\n".join(report), flush=True)
    print(f"\ngrams -> {out}", flush=True)


if __name__ == "__main__":
    main()
