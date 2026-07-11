# shard/loramerge/policy.py (run: python -m shard.loramerge.policy --config ... --prompt "...")
# Copyright (C) 2026 Martin Bukowski
#
# This software is free software: you can redistribute it and/or
# modify it under the terms of the GNU Lesser General Public License as
# published by the Free Software Foundation, either version 3 of the
# License, or (at your option) any later version.

"""Policy fingerprint scan for loramerge candidates.

A delta is a policy operator, not just a tensor (see
research/ADDENDUM_STABLE_POLICY_MERGING.md). This scan loads the 4-bit base,
applies each candidate as LoRA forward hooks, and measures its *first-token
intent*: the KL it induces from the base next-token distribution, the tokens it
promotes/suppresses, and the intent vector ``logits(base+D) - logits(base)``.

It then reports a per-candidate fingerprint and a model-model *intent-cosine*
compatibility matrix -- the behavioral analogue of scan.py's static
coverage/rotation matrix. Open question: do policy-compatible deltas (high here)
also overlap in Fourier/subspace geometry (the static matrix)?

This is a cheap first-token fingerprint, not a full rollout-residence test --
useful for pruning and clustering before a merge, not as the final gate.

Both LoRA adapters and dense NVFP4 maps are applied as forward hooks; the dense
hook dequantizes one module's delta per call and frees it, so a ~50 GB dense map
never co-resides with the 4-bit base.
"""

import argparse
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, PreTrainedTokenizerFast

from .config import LoraMergeConfig
from .play import BNB, BOS, TURN_O, TURN_C, MAX_SOURCE_GB, apply_delta, resolve_adir
from .sources import load_delta_source

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover - tqdm ships with transformers, but stay graceful
    def tqdm(iterable, **_kwargs):
        return iterable

DEFAULT_PROMPTS = ["Write two sentences about why the sky is blue."]

DENSE_KINDS = {"nvfp4_residual", "nvfp4", "dense", "nvfp4_residual_delta_map"}


class DenseDeltaHook:
    """Apply a dense ExLoRA NVFP4 delta as a forward hook.

    The full ΔW for one module is dequantized lazily on each call and freed
    immediately, so only ~one weight matrix is ever resident -- a whole dense
    map (~50 GB in bf16) never lands in memory alongside the 4-bit base. Fine
    for a first-token forward; too slow to dequant per token for generation.
    """

    def __init__(self, source, tensor_name, scale):
        self.source = source
        self.tensor_name = tensor_name
        self.scale = scale

    def __call__(self, m, inp, out):
        dW = self.source.delta(self.tensor_name, device=out.device, dtype=out.dtype)
        corr = (inp[0].to(out.dtype) @ dW.T) * self.scale
        del dW
        return out + corr


def resolve_dense_dir(storage, source):
    """Dense NVFP4 source dir: source.path, else storage/path."""
    for p in (source.source_path, Path(storage) / source.path):
        if (p / "exlora_nvfp4_residual.json").exists():
            return p
    raise FileNotFoundError(source.path)


def apply_dense_delta(modules, source, dev, storage):
    """Register dense NVFP4 hooks for one source. Returns (handles, n_modules)."""
    ds = load_delta_source(resolve_dense_dir(storage, source), kind=source.kind, name=source.name)
    tgt = set(source.target_modules) if source.target_modules else None
    handles = []
    for tensor_name in ds.tensor_names():
        if not tensor_name.endswith(".weight"):
            continue
        mname = tensor_name[: -len(".weight")]
        if mname not in modules:
            continue
        if tgt and mname.split(".")[-1] not in tgt:
            continue
        handles.append(modules[mname].register_forward_hook(DenseDeltaHook(ds, tensor_name, source.weight)))
    return handles, len(handles)


def apply_source(modules, source, dev, storage):
    """Dispatch a delta source to the LoRA or dense NVFP4 hook applier."""
    if (source.kind or "").lower() in DENSE_KINDS:
        return apply_dense_delta(modules, source, dev, storage)
    adir = resolve_adir(storage, source)
    if (adir / "adapter_model.safetensors").stat().st_size / 1e9 > MAX_SOURCE_GB:
        raise RuntimeError(f"oversized adapter ({source.name})")
    return apply_delta(modules, source, dev, storage)


def read_prompts(path):
    """One prompt per line; blank lines and ``#`` comments skipped."""
    prompts = []
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            prompts.append(line)
    return prompts


def encode(tok, prompt):
    asf = dict(add_special_tokens=False)
    return (
        [BOS, TURN_O]
        + tok("user\n", **asf)["input_ids"]
        + tok(prompt, **asf)["input_ids"]
        + [TURN_C, TURN_O]
        + tok("model\n", **asf)["input_ids"]
    )


def batched_last_logits(model, id_lists, dev, chunk, desc=None):
    """Last-real-token logits for many prompts, right-padded and batched.

    Returns a CPU float32 tensor ``[N, vocab]``. Right padding keeps default
    position ids correct for the real (leading) tokens, so RoPE is unaffected.
    """
    outs = []
    starts = range(0, len(id_lists), chunk)
    for i in tqdm(starts, desc=desc, leave=False, unit="batch"):
        sub = id_lists[i : i + chunk]
        lengths = [len(x) for x in sub]
        maxlen = max(lengths)
        input_ids = torch.zeros(len(sub), maxlen, dtype=torch.long)
        attn = torch.zeros(len(sub), maxlen, dtype=torch.long)
        for j, ids in enumerate(sub):
            input_ids[j, : len(ids)] = torch.tensor(ids, dtype=torch.long)
            attn[j, : len(ids)] = 1
        with torch.no_grad():
            logits = model(input_ids=input_ids.to(dev), attention_mask=attn.to(dev)).logits
        idx = torch.tensor(lengths, device=logits.device) - 1
        last = logits[torch.arange(len(sub), device=logits.device), idx, :].float().cpu()
        outs.append(last)
        del logits
    return torch.cat(outs, 0)


def kl_rows(p_logits, q_logits):
    """Row-wise ``KL(p || q)`` for ``[N, vocab]`` logits -> ``[N]``."""
    lp = torch.log_softmax(p_logits, -1)
    lq = torch.log_softmax(q_logits, -1)
    return (lp.exp() * (lp - lq)).sum(-1)


def _decode_tokens(tok, ids):
    return [tok.decode([t]).strip() or repr(tok.convert_ids_to_tokens(t)) for t in ids]


def fingerprint(cfg, domains, *, enable_all, top_k, device="cuda:0", chunk=16):
    """Batched policy fingerprint across domains (the 3D ``domain x model x model``).

    ``domains``: ``dict[name -> list[str]]``. The base loads once and stays
    resident; for each source we only swap adapter/dense hooks in and run ALL
    prompts (every domain) in one batched pass. So each source is applied once
    and a dense map dequantizes once per chunk -- not once per domain. Returns a
    result dict: per-(source, domain) KL, a per-domain intent-cosine matrix (the
    3D slice), and global promoted/suppressed tokens.
    """
    storage = cfg.storage_dir
    base_dir = f"{storage}/{cfg.base_model}"
    tok = PreTrainedTokenizerFast(tokenizer_file=f"{base_dir}/tokenizer.json")
    model = AutoModelForCausalLM.from_pretrained(
        base_dir, quantization_config=BNB, device_map={"": 0}, dtype=torch.bfloat16
    )
    model.eval()
    dev = torch.device(device)
    modules = dict(model.named_modules())

    domain_names = list(domains)
    flat, dom_of = [], []
    for d in domain_names:
        for p in domains[d]:
            flat.append(p)
            dom_of.append(d)
    dom_idx = {d: [i for i, dd in enumerate(dom_of) if dd == d] for d in domain_names}
    id_lists = [encode(tok, p) for p in flat]
    print(f"  {len(flat)} prompts / {len(domain_names)} domains; base pass...", flush=True)
    t_base = time.perf_counter()
    base_all = batched_last_logits(model, id_lists, dev, chunk, desc="base")
    print(f"  base pass {time.perf_counter() - t_base:.1f}s", flush=True)

    kls, matrices, tokens, status, intents, timing = {}, {}, {}, {}, {}, {}
    order = []

    for s in cfg.sources:
        if not s.is_delta_source or not (enable_all or s.enabled):
            continue
        handles = []
        try:
            t0 = time.perf_counter()
            handles, n = apply_source(modules, s, dev, storage)  # LoRA or dense NVFP4
            t_apply = time.perf_counter() - t0
            if n == 0:
                status[s.name] = "skip: no target modules"
                continue
            t1 = time.perf_counter()
            src_all = batched_last_logits(model, id_lists, dev, chunk, desc=s.name)
            t_fwd = time.perf_counter() - t1
            timing[s.name] = {"apply": t_apply, "forward": t_fwd, "mods": n}
            intent_all = src_all - base_all
            kl_all = kl_rows(src_all, base_all)
            intents[s.name] = {d: intent_all[dom_idx[d]].mean(0) for d in domain_names}
            kls[s.name] = {d: float(kl_all[dom_idx[d]].mean()) for d in domain_names}
            global_intent = intent_all.mean(0)
            tokens[s.name] = {
                "promoted": _decode_tokens(tok, global_intent.topk(top_k).indices.tolist()),
                "suppressed": _decode_tokens(tok, (-global_intent).topk(top_k).indices.tolist()),
            }
            status[s.name] = "ok"
            order.append(s.name)
            span = ", ".join(f"{d}={kls[s.name][d]:.2g}" for d in domain_names)
            print(f"  fingerprinted {s.name} ({n} mods)  apply={t_apply:.1f}s fwd={t_fwd:.1f}s  KL[{span}]", flush=True)
        except Exception as e:  # noqa: BLE001
            status[s.name] = f"skip: {type(e).__name__}"
            print(f"  skip {s.name} ({type(e).__name__})", flush=True)
        finally:
            for h in handles:
                h.remove()
            torch.cuda.empty_cache()

    for d in domain_names:
        matrices[d] = {
            (a, b): float(
                torch.nn.functional.cosine_similarity(
                    intents[a][d].unsqueeze(0), intents[b][d].unsqueeze(0)
                )[0]
            )
            for a in order
            for b in order
        }

    return {
        "domains": domain_names,
        "sources": order,
        "status": status,
        "kl": kls,
        "matrices": matrices,
        "tokens": tokens,
        "timing": timing,
    }


def _matrix_block(title, order, matrix):
    lines = [
        f"### {title}",
        "",
        "| source | " + " | ".join(order) + " |",
        "|---|" + "|".join(["---:"] * len(order)) + "|",
    ]
    for a in order:
        cells = []
        for b in order:
            v = matrix[(a, b)]
            cells.append(f"**{v:.2f}**" if a == b else f"{v:.2f}")
        lines.append(f"| {a} | " + " | ".join(cells) + " |")
    return lines


def render_markdown(cfg, domains, result, top_k):
    domain_names = result["domains"]
    order = result["sources"]
    lines = [
        "# Policy Fingerprint (domain x model x model)",
        "",
        f"- `base`: {cfg.base_model}",
        "- `domains`: " + ", ".join(f"{d} ({len(domains[d])})" for d in domain_names),
        "",
        "First-token intent per candidate, batched across all prompts (one base "
        "load, adapters swapped per source). LoRA and dense NVFP4 deltas are both "
        "applied as forward hooks.",
        "",
        "## KL by Domain",
        "",
        "`KL(base+D || base)` averaged within each domain -- how hard the delta "
        "bends the first-token distribution there.",
        "",
        "| source | " + " | ".join(domain_names) + " |",
        "|---|" + "|".join(["---:"] * len(domain_names)) + "|",
    ]
    for name in order:
        lines.append(f"| {name} | " + " | ".join(f"{result['kl'][name][d]:.3g}" for d in domain_names) + " |")
    skipped = [(nm, st) for nm, st in result["status"].items() if st != "ok"]
    if skipped:
        lines += ["", "**Skipped**", ""]
        lines += [f"- `{nm}`: {st}" for nm, st in skipped]

    timing = result.get("timing") or {}
    if timing:
        kind_of = {(s.name or s.path): ((s.kind or "").lower() or "?") for s in cfg.sources}
        lines += [
            "",
            "## Timing",
            "",
            "Per-source hook setup (`apply`) and the batched forward over all "
            "prompts (`forward`). LoRA = many tiny matmuls (launch-bound, low GPU "
            "occupancy); dense = big GEMMs + lazy dequant (compute-bound).",
            "",
            "| source | kind | mods | apply (s) | forward (s) |",
            "|---|---|---:|---:|---:|",
        ]
        for name in order:
            t = timing.get(name)
            if not t:
                continue
            lines.append(
                f"| {name} | {kind_of.get(name, '?')} | {t['mods']} | {t['apply']:.2f} | {t['forward']:.2f} |"
            )

    lines += [
        "",
        "## Global Intent Tokens",
        "",
        f"Top-{top_k} promoted/suppressed tokens per source (mean intent over all prompts).",
        "",
        "| source | promoted | suppressed |",
        "|---|---|---|",
    ]
    for name in order:
        prom = ", ".join(f"`{t}`" for t in result["tokens"][name]["promoted"])
        supp = ", ".join(f"`{t}`" for t in result["tokens"][name]["suppressed"])
        lines.append(f"| {name} | {prom} | {supp} |")

    lines += [
        "",
        "## Intent-Cosine Matrices by Domain",
        "",
        "+1 = same first-token policy push (redundant), 0 = independent, "
        "-1 = opposing (conflict — gate before merging).",
    ]
    if not order:
        lines += ["", "_No candidates produced an intent vector._"]
        return "\n".join(lines) + "\n"
    for d in domain_names:
        lines.append("")
        lines += _matrix_block(d, order, result["matrices"][d])
    return "\n".join(lines) + "\n"


def _collect_domains(args):
    """Assemble ``{domain -> prompts}`` from --prompt/--prompt-file/--domain/--domains-dir."""
    domains = {}
    if args.prompt or args.prompt_file:
        prompts = list(args.prompt or [])
        if args.prompt_file:
            prompts += read_prompts(args.prompt_file)
        domains["adhoc"] = prompts
    files = [Path(p) for p in (args.domain or [])]
    if args.domains_dir:
        files += sorted(Path(args.domains_dir).glob("*.txt"))
    for f in files:
        domains[f.stem] = read_prompts(f)
    if not domains:
        domains["default"] = list(DEFAULT_PROMPTS)
    return {d: p for d, p in domains.items() if p}


def main():
    ap = argparse.ArgumentParser(description="Policy fingerprint for loramerge candidates (3D across domains).")
    ap.add_argument("--config", required=True)
    ap.add_argument("--domains-dir", default=None,
                    help="directory of <domain>.txt prompt files (one prompt per line)")
    ap.add_argument("--domain", action="append", default=None,
                    help="a single <domain>.txt file (repeatable); domain name = file stem")
    ap.add_argument("--prompt", action="append",
                    help="ad-hoc prompt (repeatable); grouped as one 'adhoc' domain")
    ap.add_argument("--prompt-file", default=None, help="one-prompt-per-line file as a single domain")
    ap.add_argument("--top-k", type=int, default=12, help="tokens promoted/suppressed to report")
    ap.add_argument("--enable-all", action="store_true",
                    help="fingerprint every delta source, ignoring the enabled flag")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--chunk", type=int, default=16, help="prompts per batched forward")
    ap.add_argument("--output", default=None, help="write the report here instead of stdout")
    args = ap.parse_args()

    domains = _collect_domains(args)
    cfg = LoraMergeConfig.from_yaml(args.config)
    total = sum(len(p) for p in domains.values())
    print(f"policy fingerprint: {Path(args.config).name}  ({len(domains)} domains, {total} prompts)", flush=True)
    result = fingerprint(cfg, domains, enable_all=args.enable_all, top_k=args.top_k,
                         device=args.device, chunk=args.chunk)
    report = render_markdown(cfg, domains, result, args.top_k)

    if args.output is None:
        print("\n" + report, flush=True)
        return
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(report)
    print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    main()
