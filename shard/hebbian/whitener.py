# Copyright (C) 2026 Martin Bukowski
"""Suffix-keyed input-side whiteners from harvested activation Grams.

Turns mcq_grams.safetensors (factor F = V_r diag(sqrt(lambda_r)) per
(module suffix, in_dim)) into the `(Vk, sqrt_lam)` pairs the isometric
operators already accept: subspace selection then runs on d @ H^(alpha/2)
instead of raw d, so rank is spent on the input directions that actually
fire on the harvested task, not top-Frobenius.

Keys are module-name SUFFIXES (e.g. "self_attn.q_proj"), not in_dims --
three different 5120-dim Grams coexist (self_attn inputs, linear_attn
inputs, mlp inputs) and must not collapse into one entry.

alpha damps the metric: the harvested H is dominated by the massive-
activation/sink directions (probe: er_H ~= 1), and raw H^1/2 selection
would spend the whole rank budget on the sink dims. alpha=1 is full
H^1/2; alpha=0.5 is H^1/4; alpha -> 0 approaches Frobenius.
"""

from pathlib import Path

import torch
from safetensors import safe_open


def load_hebbian_whiteners(path: str | Path, k: int = 256, alpha: float = 1.0):
    """{module_suffix: (Vk [in,k], lam^(alpha/2) [k])} from a harvest file."""
    wh = {}
    with safe_open(str(path), framework="pt") as f:
        for key in f.keys():
            if not key.startswith("factor::"):
                continue
            suffix, _d = key.split("::")[1].rsplit("|", 1)
            F = f.get_tensor(key).to(torch.float32)          # V_r * sqrt(lam)
            lam = f.get_tensor(key.replace("factor::", "eigval::")).to(torch.float32)
            kk = min(k, lam.numel())
            V = F[:, :kk] / lam[:kk].clamp_min(1e-30).sqrt().unsqueeze(0)
            wh[suffix] = (V, lam[:kk].clamp_min(0).pow(alpha / 2.0))
    print(f"  hebbian whiteners: {len(wh)} suffixes from {path} "
          f"(k={k}, alpha={alpha})", flush=True)
    return wh


def whitener_for(whiteners: dict | None, tensor_name: str, in_dim: int):
    """Match a tensor name against suffix-keyed whiteners; None if no match.

    Also accepts legacy in_dim-keyed entries so a mixed dict keeps working.
    """
    if not whiteners:
        return None
    for key, w in whiteners.items():
        if isinstance(key, str) and f".{key}." in f".{tensor_name}" \
                and w[0].shape[0] == in_dim:
            return w
    return whiteners.get(in_dim)
