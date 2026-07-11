# shard/loramerge/runner.py (run: python -m shard.loramerge.runner --config ... --dry-run)
#
# Orchestrates the full-frame merge: iterate base tensors, merge each with the
# config's rule (sign_vote / norm_slerp / fourier_slerp) via merge_full_frame_tensor,
# route embeddings/lm_head from model sources, write the merged model. --dry-run
# merges a few sample tensors in memory and reports diagnostics (so we can
# confirm the rule bounds magnitude / resolves interference before writing).

import argparse
import json
import shutil
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from .config import LoraMergeConfig
from .merge import (
    WeightedDeltaSource,
    collect_source_deltas,
    merge_full_frame_tensor,
    sum_weighted_deltas,
)
from .sources import CheckpointDeltaSource, load_delta_source

CHECKPOINT_KINDS = {"checkpoint", "model_delta"}

MAX_SOURCE_GB = 12.0
SAMPLE = [
    "model.language_model.layers.0.mlp.gate_proj.weight",
    "model.language_model.layers.0.self_attn.q_proj.weight",
    "model.language_model.layers.24.self_attn.o_proj.weight",   # heretic contributes here
    "model.language_model.layers.30.mlp.down_proj.weight",
    "model.language_model.layers.36.mlp.down_proj.weight",     # noloop surgical overlay
]


def resolve_path(storage, source):
    for p in (source.source_path, Path(storage) / source.path):
        if p.exists():
            return p
    raise FileNotFoundError(source.path)


def load_whiteners(path, k=256, device="cuda"):
    """{in_dim: (Vk[in,k], sqrt_lam[k])} top-k eigvecs of each input-dim Gram H."""
    from safetensors import safe_open
    wh = {}
    with safe_open(path, framework="pt") as f:
        for key in f.keys():
            H = f.get_tensor(key).to(device, torch.float32)
            q = min(k + 8, H.shape[0])
            U, S, _V = torch.svd_lowrank(H, q=q)
            wh[H.shape[0]] = (U[:, :k].cpu(), S[:k].clamp_min(0).sqrt().cpu())
            del H
    print(f"  loaded whiteners for dims {sorted(wh)} (k={k})", flush=True)
    return wh


def build_readout_whitener(base_dir, device="cuda", damp_rel=1e-3):
    """(M^½, damped M^{-½}) for the LM-head readout metric M = U_effᵀ U_eff,
    U_eff = embed_tokens·diag(1+γ_final). Output-side whitener: the isometric merge
    runs in this metric for residual-stream writers (out_dim == hidden), so rank is
    spent on readout-live directions, not top-Frobenius (research/RESULTS_READOUT_NORM_RANK)."""
    from safetensors import safe_open
    wm = json.load(open(f"{base_dir}/model.safetensors.index.json"))["weight_map"]

    def g(n):
        with safe_open(f"{base_dir}/{wm[n]}", framework="pt") as f:
            return f.get_tensor(n).to(device, torch.float32)

    Ueff = g("model.language_model.embed_tokens.weight") * (1.0 + g("model.language_model.norm.weight")).unsqueeze(0)
    M = (Ueff.T @ Ueff).double(); del Ueff
    lam, V = torch.linalg.eigh(M)
    lam = lam.clamp_min(0); sq = lam.sqrt()
    floor = damp_rel * float(lam.max().clamp_min(1e-30))
    Mhalf = ((V * sq.unsqueeze(0)) @ V.T).float()
    Mhalf_inv = ((V * (sq / (lam + floor)).unsqueeze(0)) @ V.T).float()
    eff = float((lam.sum() ** 2 / (lam ** 2).sum()).item())
    print(f"  readout whitener: dim {M.shape[0]}, eff_rank {eff:.1f}, damp_rel {damp_rel}", flush=True)
    return (Mhalf.cpu(), Mhalf_inv.cpu())


def load_weighted_sources(cfg, *, enable_all=False):
    base_dir = f"{cfg.storage_dir}/{cfg.base_model}"
    out = []
    # ordered_delta_sources filters enabled/disabled AND sorts by priority, so
    # the returned list folds into the iterative merge in layering order.
    for s in cfg.ordered_delta_sources(enable_all=enable_all):
        p = resolve_path(cfg.storage_dir, s)
        if s.kind in CHECKPOINT_KINDS:
            sb = f"{cfg.storage_dir}/{s.base_model}" if s.base_model else base_dir
            ds = CheckpointDeltaSource(p, sb, name=s.name)         # W_model - W_base, lazy
        else:
            af = p / "adapter_model.safetensors"
            if af.exists() and af.stat().st_size / 1e9 > MAX_SOURCE_GB:
                print(f"  skip {s.name} (oversized)", flush=True); continue
            ds = load_delta_source(p, kind=s.kind, name=s.name)
        out.append((s, WeightedDeltaSource(source=ds, weight=s.weight, rescale=s.rescale,
                                           is_norm=s.is_norm, name=s.name, post_merge=s.post_merge,
                                           group=s.group, pool=s.pool, pool_rank=s.pool_rank,
                                           pool_energy=s.pool_energy, pool_residual=s.pool_residual,
                                           priority=s.priority)))
    return out


def base_tensor(base_dir, wm, name, device):
    with safe_open(f"{base_dir}/{wm[name]}", framework="pt") as f:
        return f.get_tensor(name).to(device=device, dtype=torch.float32)


def dry_run(cfg, weighted, base_dir, device):
    import statistics
    wm = json.load(open(f"{base_dir}/model.safetensors.index.json"))["weight_map"]
    wsources = [w for _, w in weighted]

    # per-source magnitude (after current rescale) + suggested rescale to balance
    print("\nper-source mean ‖Δ‖ across samples (×current rescale) -> suggested rescale to balance:", flush=True)
    means = {}
    for w in wsources:
        ns = [(w.source.delta(n, device=device).norm().item() * w.rescale)
              for n in SAMPLE if n in wm and w.source.delta(n, device=device) is not None]
        means[w.name] = statistics.mean(ns) if ns else 0.0
    median = statistics.median([m for m in means.values() if m > 0]) or 1.0
    norm_ref = next((w.name for w in wsources if w.is_norm), None)
    for w in wsources:
        m = means[w.name]
        if w.post_merge:
            note = "  [POST OVERLAY -> preserve source-relative magnitude]"
        else:
            note = f"  -> balance rescale≈{median / m if m > 0 else 1.0:.3f}"
            if w.is_norm:
                note += "  [NORM REF -> sets output magnitude]"
        print(f"  {w.name:24} ‖Δ‖={m:8.3f} (rescale={w.rescale}){note}", flush=True)
    if norm_ref is None:
        if cfg.merge_rule == "isometric":
            print("  (isometric magnitude is controlled by pool/global energy; post overlays stay source-relative)", flush=True)
        else:
            print("  (no is_norm source -> output magnitude = slerp norm-interpolation of all)", flush=True)

    print(f"\nrule={cfg.merge_rule}  sources={[w.name for w in wsources]}\n", flush=True)
    hdr = f"{'tensor':52} {'n':>2} {'sum‖Δ‖':>9} {'merged‖Δ‖':>10} {'m/sum':>6} {'relΔ':>7}"
    print(hdr, flush=True)
    for name in SAMPLE:
        if name not in wm:
            print(f"{name[:52]:52}  (not in base)"); continue
        base = base_tensor(base_dir, wm, name, device)
        deltas, weights, _ = collect_source_deltas(wsources, name, device=device)
        if not deltas:
            print(f"{name.replace('model.language_model.',''):52}  no source deltas"); continue
        naive = sum_weighted_deltas(deltas, weights)
        merged_w = merge_full_frame_tensor(
            base, wsources, name, rule=cfg.merge_rule, device=device,
            fourier_cutoff_pct=cfg.fourier_cutoff_pct, fourier_cull_pct=cfg.fourier_cull_pct,
            fourier_small_norm_ratio=cfg.fourier_small_norm_ratio, final_merge_t=cfg.final_merge_t,
            isometric_rank=cfg.isometric_rank, isometric_energy=cfg.isometric_energy,
            whiteners=getattr(cfg, '_whiteners', None),
            readout_whitener=getattr(cfg, '_readout_whitener', None))
        merged_d = merged_w - base
        sn, mn, bn = naive.norm().item(), merged_d.norm().item(), base.norm().item()
        per = " ".join(f"{d.norm().item():.2f}" for d in deltas)
        print(f"{name.replace('model.language_model.',''):52} {len(deltas):>2} {sn:>9.2f} "
              f"{mn:>10.2f} {mn/max(sn,1e-9):>6.2f} {mn/bn:>7.4f}   [{per}]", flush=True)


def _load_full(model_dir, wm, name, device):
    with safe_open(f"{model_dir}/{wm[name]}", framework="pt") as f:
        return f.get_tensor(name).to(device=device, dtype=torch.float32)


def merged_items(cfg, wsources, base_dir, device):
    """Yield (tensor_name, bf16 cpu tensor) for the whole merged model."""
    base_wm = json.load(open(f"{base_dir}/model.safetensors.index.json"))["weight_map"]
    storage = cfg.storage_dir
    emb_dir = f"{storage}/{cfg.embedding_model}" if cfg.embedding_source else None
    head_dir = f"{storage}/{cfg.lm_head_model}" if cfg.lm_head_source else None
    targets = set(cfg.target_modules)
    n = 0
    for name in base_wm:
        module = name[: -len(".weight")].split(".")[-1] if name.endswith(".weight") else None
        if module in targets:
            bt = _load_full(base_dir, base_wm, name, device)
            t = merge_full_frame_tensor(
                bt, wsources, name, rule=cfg.merge_rule, device=device,
                fourier_cutoff_pct=cfg.fourier_cutoff_pct, fourier_cull_pct=cfg.fourier_cull_pct,
                fourier_small_norm_ratio=cfg.fourier_small_norm_ratio, final_merge_t=cfg.final_merge_t,
            isometric_rank=cfg.isometric_rank, isometric_energy=cfg.isometric_energy,
            whiteners=getattr(cfg, '_whiteners', None),
            readout_whitener=getattr(cfg, '_readout_whitener', None))
            n += 1
            if n % 50 == 0:
                print(f"  merged {n} body tensors", flush=True)
        elif emb_dir and name.endswith("embed_tokens.weight"):
            ewm = json.load(open(f"{emb_dir}/model.safetensors.index.json"))["weight_map"]
            t = _load_full(emb_dir, ewm, name, device)       # IO: input embeddings
        else:
            t = _load_full(base_dir, base_wm, name, device)   # passthrough
        yield name, t.to(torch.bfloat16).contiguous().cpu()
        del t
    if head_dir:                                              # IO: untied lm_head (not in base)
        hwm = json.load(open(f"{head_dir}/model.safetensors.index.json"))["weight_map"]
        if "lm_head.weight" in hwm:
            yield "lm_head.weight", _load_full(head_dir, hwm, "lm_head.weight", device).to(torch.bfloat16).cpu()


def write_sharded(out_dir, items, *, max_bytes=4_000_000_000):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    weight_map, total, buf, buf_bytes, shard_no = {}, 0, {}, 0, 0

    def flush():
        nonlocal buf, buf_bytes, shard_no
        if not buf:
            return
        shard_no += 1
        fname = f"model-{shard_no:05d}.safetensors"
        save_file(buf, str(out_dir / fname), metadata={"format": "pt"})
        for nm in buf:
            weight_map[nm] = fname
        print(f"  wrote {fname} ({buf_bytes/1e9:.1f} GB, {len(buf)} tensors)", flush=True)
        buf, buf_bytes = {}, 0

    for name, t in items:
        nb = t.numel() * t.element_size()
        if buf_bytes + nb > max_bytes:
            flush()
        buf[name] = t
        buf_bytes += nb
        total += nb
    flush()
    json.dump({"metadata": {"total_size": total}, "weight_map": weight_map},
              open(out_dir / "model.safetensors.index.json", "w"), indent=2)
    return shard_no, total


def copy_aux(base_dir, out_dir, *, untie):
    base_dir, out_dir = Path(base_dir), Path(out_dir)
    for f in base_dir.iterdir():
        if f.suffix == ".safetensors" or f.name in ("model.safetensors.index.json", "config.json"):
            continue
        if f.is_file():
            shutil.copy2(f, out_dir / f.name)
    cfg_json = json.load(open(base_dir / "config.json"))
    if untie:
        cfg_json["tie_word_embeddings"] = False
        if isinstance(cfg_json.get("text_config"), dict):
            cfg_json["text_config"]["tie_word_embeddings"] = False
    json.dump(cfg_json, open(out_dir / "config.json", "w"), indent=2)


def run_merge(cfg, weighted, base_dir, out_dir, device):
    wsources = [w for _, w in weighted]
    print(f"writing merged model -> {out_dir}  (rule={cfg.merge_rule}, {len(wsources)} sources)", flush=True)
    n_shards, total = write_sharded(out_dir, merged_items(cfg, wsources, base_dir, device))
    copy_aux(base_dir, out_dir, untie=cfg.lm_head_source is not None)
    print(f"done: {n_shards} shards, {total/1e9:.1f} GB -> {out_dir}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--enable-all", action="store_true",
                    help="merge every clean delta source (ignore per-source enabled flag)")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--rule", default=None, help="override merge_rule (sum|sign_vote|norm_slerp|fourier_slerp|isometric)")
    ap.add_argument("--rank", type=int, default=None, help="override isometric_rank")
    ap.add_argument("--energy", type=float, default=None, help="override isometric_energy (radial reintroduction)")
    ap.add_argument("--grams", default=None, help="activation Gram file -> H-weighted (behavioral) subspace")
    ap.add_argument("--readout", action="store_true",
                    help="isometric merge in the LM-head readout norm (output-side whitener) for residual writers")
    args = ap.parse_args()

    cfg = LoraMergeConfig.from_yaml(args.config)
    if args.rule:
        cfg.merge_rule = args.rule
    if args.rank:
        cfg.isometric_rank = args.rank
    if args.energy is not None:
        cfg.isometric_energy = args.energy
    if args.grams:
        cfg._whiteners = load_whiteners(args.grams, device=args.device)
    base_dir = f"{cfg.storage_dir}/{cfg.base_model}"
    if args.readout:
        cfg._readout_whitener = build_readout_whitener(base_dir, device=args.device)
    print(f"merge runner: {Path(args.config).name}  rule={cfg.merge_rule}", flush=True)
    weighted = load_weighted_sources(cfg, enable_all=args.enable_all)
    print(f"loaded {len(weighted)} delta sources", flush=True)

    if args.dry_run:
        dry_run(cfg, weighted, base_dir, args.device)
        return
    run_merge(cfg, weighted, base_dir, cfg.output_dir, args.device)


if __name__ == "__main__":
    main()
