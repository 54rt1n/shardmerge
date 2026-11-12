# Copyright (C) 2026 Martin Bukowski
"""Does the activation H-metric buy anything over Frobenius for these sources?

This is the go/no-go gate for the hebbian_isometric operator: if the harvested
Gram does not concentrate or separate the source deltas beyond what Frobenius
(or the readout metric) already shows, there is nothing for an H-whitened
rotation to exploit and the merge code should not be written.

Per source x module suffix, three numbers, layer-aggregated (weighted by
||Delta||_F^2):

  rho        -- H-alignment ratio: (tr(D H D^T)/||D||_F^2) / (tr(H)/d).
                1.0 = the delta is isotropic w.r.t. what fires; >> 1 = the
                delta lives inside the activated subspace (the NVFP4 comb
                showed ~100x here when Frobenius said full-rank).
  er_raw     -- effective rank (participation ratio) of D.
  er_H       -- effective rank of D H^1/2. er_H << er_raw means the H-metric
                compresses the delta onto few live directions -> subspace
                selection in this metric has something to select.

With --overlap, adds the pairwise source separation: principal-subspace
overlap (top-8 out-side directions) between sources, raw vs H-projected.
H sharpening the distinctions (overlap dropping) is the second win condition.

LoRA sources only -- every v12 body source is a LoRA, and D = s.B@A keeps all
linear algebra in (r x k) factors so the probe runs in seconds without ever
materializing a 5120^2 delta.

    uv run python -m shard.hebbian.probe --grams models/qwen36/hebbian/mcq_grams.safetensors
"""

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import torch
from safetensors import safe_open

ROOT = Path("models/qwen36")

SOURCES = {
    "ostrich": "lora/etemiz/Ostrich-27B-Qwen3.6-260603",
    "darwin": "lora/FINAL-Bench/Darwin-28B-REASON",
    "grm": "lora/OrionLLM/GRM-2.6-Plus-0628",
    "qwopus": "lora/Jackrong/Qwopus3.6-27B-v2",
    "thinkingcap": "lora/bottlecapai/ThinkingCap-Qwen3.6-27B",
    "fable5": "lora/TeichAI/Qwen3.6-27B-Fable-5-Experimental-LoRA",
    "opus-distill": "lora/TeichAI/Qwen3.6-27B-Claude-Opus-Reasoning-Distill-v2",
    "btl3": "lora/badtheorylabs/BTL-3",
}

SUFFIX_RE = re.compile(
    r"\.((?:self_attn|mlp|linear_attn)\.[a-z_]+proj[a-z_]*)\.lora_([AB])\.weight$")


def load_grams(path):
    """{ (suffix, in_dim): (F fp32 [in,k], trace, tokens) }"""
    grams = {}
    with safe_open(path, "pt") as f:
        keys = [k for k in f.keys() if k.startswith("factor::")]
        for k in keys:
            suffix, d = k.split("::")[1].rsplit("|", 1)
            F = f.get_tensor(k).to(torch.float32)
            tr = float(f.get_tensor(f"trace::{suffix}|{d}")[0])
            n = int(f.get_tensor(f"tokens::{suffix}|{d}")[0])
            grams[(suffix, int(d))] = (F, tr, n)
    return grams


def load_lora(path):
    """{ module_name: (A [r,in], B [out,r], scale) } in fp32."""
    cfg = json.loads((path / "adapter_config.json").read_text())
    scale = cfg["lora_alpha"] / cfg["r"]
    pairs = defaultdict(dict)
    with safe_open(path / "adapter_model.safetensors", "pt") as f:
        for k in f.keys():
            m = SUFFIX_RE.search(k)
            if m is None:
                continue
            mod = k[: k.index(f".lora_{m.group(2)}")]
            pairs[mod][m.group(2)] = f.get_tensor(k).to(torch.float32)
    return {mod: (ab["A"], ab["B"], scale)
            for mod, ab in pairs.items() if "A" in ab and "B" in ab}


def suffix_of(module_name):
    for suf in ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj",
                "self_attn.o_proj", "mlp.gate_proj", "mlp.up_proj",
                "mlp.down_proj", "linear_attn.in_proj_qkv", "linear_attn.in_proj_a",
                "linear_attn.in_proj_b", "linear_attn.in_proj_z",
                "linear_attn.out_proj"):
        if module_name.endswith(suf):
            return suf
    return None


def factored_singvals(B, M):
    """Singular values of B @ M without materializing it: sv(R_B @ R_M^T)."""
    _, rb = torch.linalg.qr(B, mode="reduced")           # [r, r]
    _, rm = torch.linalg.qr(M.transpose(0, 1), mode="reduced")  # [r, r]
    return torch.linalg.svdvals(rb @ rm.transpose(0, 1))


def participation_ratio(s):
    s2 = s.square()
    return float(s2.sum().square() / s2.square().sum().clamp_min(1e-30))


def top_leftvecs(B, M, k):
    """Top-k left singular vectors of B @ M (out-side directions)."""
    qb, rb = torch.linalg.qr(B, mode="reduced")
    _, rm = torch.linalg.qr(M.transpose(0, 1), mode="reduced")
    U, S, _ = torch.linalg.svd(rb @ rm.transpose(0, 1))
    k = min(k, S.numel())
    return qb @ U[:, :k]                                  # [out, k]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grams", default=str(ROOT / "hebbian/mcq_grams.safetensors"))
    ap.add_argument("--sources", default=",".join(SOURCES))
    ap.add_argument("--overlap", action="store_true",
                    help="pairwise source separation, raw vs H-projected")
    ap.add_argument("--overlap-k", type=int, default=8)
    ap.add_argument("--out", default="out/hebbian/probe.json")
    args = ap.parse_args()

    grams = load_grams(args.grams)
    print(f"grams: {len(grams)} (suffix, dim) entries from {args.grams}", flush=True)

    names = [n for n in args.sources.split(",") if n in SOURCES]
    report = {}
    # per (source, suffix) accumulators weighted by ||D||_F^2
    for name in names:
        lora = load_lora(ROOT / SOURCES[name])
        acc = defaultdict(lambda: {"w": 0.0, "rho": 0.0, "er_raw": 0.0,
                                   "er_H": 0.0, "n_mod": 0})
        for mod, (A, B, s) in lora.items():
            suf = suffix_of(mod)
            if suf is None:
                continue
            g = grams.get((suf, A.shape[1]))
            if g is None:
                continue
            F, trH, _ = g
            d = A.shape[1]
            sv_raw = factored_singvals(s * B, A)
            fro2 = float(sv_raw.square().sum())
            if fro2 <= 0:
                continue
            AF = A @ F                                    # [r, k]
            sv_H = factored_singvals(s * B, AF)
            eH = float(sv_H.square().sum())               # tr(D H D^T)
            rho = (eH / fro2) / max(trH / d, 1e-30)
            a = acc[suf]
            a["w"] += fro2
            a["rho"] += fro2 * rho
            a["er_raw"] += fro2 * participation_ratio(sv_raw)
            a["er_H"] += fro2 * participation_ratio(sv_H)
            a["n_mod"] += 1
        rows = {}
        print(f"\n[{name}]", flush=True)
        for suf in sorted(acc):
            a = acc[suf]
            w = max(a["w"], 1e-30)
            rows[suf] = {"rho": a["rho"] / w, "er_raw": a["er_raw"] / w,
                         "er_H": a["er_H"] / w, "n_mod": a["n_mod"]}
            r = rows[suf]
            print(f"  {suf:26s} rho {r['rho']:8.2f}   er {r['er_raw']:6.1f} -> "
                  f"{r['er_H']:6.1f}   ({a['n_mod']} modules)", flush=True)
        report[name] = rows

    if args.overlap:
        # Separation on the residual-stream writers, where sources collide.
        print("\npairwise top-8 subspace overlap (raw / H-projected), "
              "residual writers:", flush=True)
        overlap = {}
        for suf in ("self_attn.o_proj", "mlp.down_proj", "linear_attn.out_proj"):
            vecs = {}
            for name in names:
                lora = load_lora(ROOT / SOURCES[name])
                Us_raw, Us_H = [], []
                for mod, (A, B, s) in lora.items():
                    if suffix_of(mod) != suf:
                        continue
                    g = grams.get((suf, A.shape[1]))
                    if g is None:
                        continue
                    Us_raw.append(top_leftvecs(s * B, A, args.overlap_k))
                    Us_H.append(top_leftvecs(s * B, A @ g[0], args.overlap_k))
                if Us_raw:
                    vecs[name] = (Us_raw, Us_H)
            pair_rows = {}
            ns = [n for n in names if n in vecs]
            for i, a in enumerate(ns):
                for b in ns[i + 1:]:
                    ov = [0.0, 0.0]
                    n_l = min(len(vecs[a][0]), len(vecs[b][0]))
                    for which in (0, 1):
                        tot = 0.0
                        for l in range(n_l):
                            Ua, Ub = vecs[a][which][l], vecs[b][which][l]
                            c = torch.linalg.svdvals(Ua.transpose(0, 1) @ Ub)
                            tot += float(c.square().mean())
                        ov[which] = tot / max(n_l, 1)
                    pair_rows[f"{a}|{b}"] = {"raw": ov[0], "H": ov[1]}
                    print(f"  {suf:22s} {a:>12s} x {b:<12s} "
                          f"{ov[0]:.3f} / {ov[1]:.3f}", flush=True)
            overlap[suf] = pair_rows
        report["_overlap"] = overlap

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    print(f"\nprobe -> {out}", flush=True)


if __name__ == "__main__":
    main()
