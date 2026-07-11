# shard/loramerge/play.py (run: python -m shard.loramerge.play --config ...)
#
# Turn a shard.loramerge config into a generatable model and sample from it, or
# (--validate) sweep every delta source individually to flag damaged ones.
# 4-bit base body + IO routing (embeddings / lm_head from model sources) +
# delta sources as fp16 LoRA forward hooks (merge_rule=sum).

import argparse
import json
from pathlib import Path

import torch
from safetensors import safe_open
from transformers import AutoModelForCausalLM, BitsAndBytesConfig, PreTrainedTokenizerFast

from .config import LoraMergeConfig

BOS, TURN_O, TURN_C, EOS = 2, 105, 106, [1, 106, 50]
PROMPT = "Write two sentences about why the sky is blue."
ROLLOUT_PROMPTS = [
    ("code", "Write a Python function that merges two sorted lists into one sorted list."),
    ("creative", "Continue this story in a dark fantasy tone: The gate creaked open, revealing"),
    ("factual", "Explain in two sentences why the sky is blue."),
    ("character", "Write a short, menacing monologue from a villain who has just won."),
]
KL_DAMAGED, UNIQ_DAMAGED = 5.0, 0.35           # damage thresholds
MAX_SOURCE_GB = 12.0                            # above this = dense-forced-to-LoRA, not a real adapter


def resolve_adir(storage, source):
    """Adapter dir: source.path, else storage/path (configs use storage-relative paths)."""
    for p in (source.source_path, Path(storage) / source.path):
        if (p / "adapter_config.json").exists():
            return p
    raise FileNotFoundError(source.path)
BNB = BitsAndBytesConfig(
    load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    llm_int8_skip_modules=["vision_tower", "multi_modal_projector", "embed_vision", "lm_head"],
)


def find_tensor(model_dir, *candidates):
    idx = json.load(open(f"{model_dir}/model.safetensors.index.json"))["weight_map"]
    for name in candidates:
        if name in idx:
            with safe_open(f"{model_dir}/{idx[name]}", framework="pt") as f:
                return name, f.get_tensor(name).float()
    raise KeyError(f"none of {candidates} in {model_dir}")


def parse_lora_key(key):
    body = key[len("base_model.model."):] if key.startswith("base_model.model.") else key
    for suffix, which in ((".lora_A.weight", "A"), (".lora_B.weight", "B")):
        if body.endswith(suffix):
            return body[: -len(suffix)] + ".weight", which
    return None, None


class LoraHook:
    def __init__(self, A, B, scale):
        self.A, self.B, self.scale = A, B, scale

    def __call__(self, m, inp, out):
        return out + self.scale * ((inp[0].to(self.A.dtype) @ self.A.T) @ self.B.T).to(out.dtype)


def apply_delta(modules, source, dev, storage, scale_mult=1.0):
    """Register LoRA hooks for one delta source. Returns (handles, n_modules)."""
    adir = resolve_adir(storage, source)
    cfgj = json.loads((adir / "adapter_config.json").read_text())
    a0 = float(cfgj.get("lora_alpha", cfgj.get("r", 1)))
    rslora = bool(cfgj.get("use_rslora", False))
    ap_pat = cfgj.get("alpha_pattern") or {}
    tgt = set(source.target_modules) if source.target_modules else None
    handles = []
    with safe_open(str(adir / "adapter_model.safetensors"), framework="pt") as f:
        facs = {}
        for k in f.keys():
            wn, which = parse_lora_key(k)
            if wn:
                facs.setdefault(wn, {})[which] = k
        for wn, kk in facs.items():
            mname = wn[:-len(".weight")]
            if "A" not in kk or "B" not in kk or mname not in modules:
                continue
            if tgt and mname.split(".")[-1] not in tgt:
                continue
            A = f.get_tensor(kk["A"]).to(dev, torch.bfloat16)
            B = f.get_tensor(kk["B"]).to(dev, torch.bfloat16)
            r = A.shape[0]
            scale = scale_mult * source.weight * float(ap_pat.get(mname, a0)) / (r ** 0.5 if rslora else r)
            handles.append(modules[mname].register_forward_hook(LoraHook(A, B, scale)))
    return handles, len(handles)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--prompt", default=PROMPT)
    ap.add_argument("--max-new-tokens", type=int, default=60)
    ap.add_argument("--validate", action="store_true",
                    help="sweep every delta source alone and flag damaged ones")
    ap.add_argument("--rollout", action="store_true",
                    help="generate base vs each source across a prompt set; save transcript")
    ap.add_argument("--enable-all", action="store_true",
                    help="apply every clean delta source (ignore the per-source enabled flag)")
    ap.add_argument("--scale", type=float, default=1.0,
                    help="global multiplier on every delta (e.g. 1/N to average instead of sum)")
    args = ap.parse_args()

    cfg = LoraMergeConfig.from_yaml(args.config)
    storage = cfg.storage_dir
    base_dir = f"{storage}/{cfg.base_model}"
    print(f"merge: {Path(args.config).name}  rule={cfg.merge_rule}", flush=True)

    tok = PreTrainedTokenizerFast(tokenizer_file=f"{base_dir}/tokenizer.json")
    model = AutoModelForCausalLM.from_pretrained(
        base_dir, quantization_config=BNB, device_map={"": 0}, dtype=torch.bfloat16)
    model.eval()
    dev = torch.device("cuda:0")
    modules = dict(model.named_modules())

    asf = dict(add_special_tokens=False)

    def build(prompt):
        ids = ([BOS, TURN_O] + tok("user\n", **asf)["input_ids"] + tok(prompt, **asf)["input_ids"]
               + [TURN_C, TURN_O] + tok("model\n", **asf)["input_ids"])
        return torch.tensor([ids]).to(dev)

    def last_logits(inp):
        with torch.no_grad():
            return model(inp).logits[0, -1].float().cpu()

    def generate(inp, n):
        torch.manual_seed(0)
        with torch.no_grad():
            out = model.generate(inp, max_new_tokens=n, do_sample=True, temperature=0.8,
                                 top_p=0.95, eos_token_id=EOS, pad_token_id=0)
        return tok.decode(out[0, inp.shape[1]:].tolist()).strip()

    def kl(a, b):
        la, lb = torch.log_softmax(a, -1), torch.log_softmax(b, -1)
        return float((la.exp() * (la - lb)).sum())

    def apply_io_routing():
        labels = []
        if cfg.embedding_source is not None:
            _, t = find_tensor(f"{storage}/{cfg.embedding_model}",
                               "model.language_model.embed_tokens.weight", "model.embed_tokens.weight")
            model.get_input_embeddings().weight.data.copy_(t.to(dev, torch.bfloat16))
            labels.append(f"embed<-{cfg.embedding_source.name}")
        if cfg.lm_head_source is not None:
            _, t = find_tensor(f"{storage}/{cfg.lm_head_model}",
                               "lm_head.weight", "model.language_model.embed_tokens.weight")
            model.get_output_embeddings().weight = torch.nn.Parameter(t.to(dev, torch.bfloat16), requires_grad=False)
            labels.append(f"head<-{cfg.lm_head_source.name}")
        return labels

    delta_sources = [s for s in cfg.sources if s.is_delta_source]

    if args.rollout:
        inps = [(tag, p, build(p)) for tag, p in ROLLOUT_PROMPTS]
        # row label -> {tag: text}
        rows = {}

        def run_row(label):
            rows[label] = {tag: generate(inp, 80) for tag, _, inp in inps}
            print(f"  rolled out: {label}", flush=True)

        run_row("base")
        for s in delta_sources:
            handles = []
            try:
                adir = resolve_adir(storage, s)
                if (adir / "adapter_model.safetensors").stat().st_size / 1e9 > MAX_SOURCE_GB:
                    print(f"  skip {s.name} (oversized)", flush=True); continue
                handles, _ = apply_delta(modules, s, dev, storage)
                run_row(s.name)
            except Exception as e:
                print(f"  skip {s.name} ({type(e).__name__})", flush=True)
            finally:
                for h in handles:
                    h.remove()
                torch.cuda.empty_cache()
        # assembled merge (IO routing + enabled deltas) last (IO routing is persistent)
        io = apply_io_routing()
        for s in [x for x in delta_sources if x.enabled]:
            apply_delta(modules, s, dev, storage)
        run_row("MERGE [" + ", ".join(io) + "]")

        out_path = Path("out/rollouts") / (Path(args.config).stem + ".md")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w") as f:
            f.write(f"# Rollout — {Path(args.config).name}\n\n")
            for tag, prompt, _ in inps:
                f.write(f"## [{tag}] {prompt}\n\n")
                for label, texts in rows.items():
                    f.write(f"**{label}**\n\n> {texts[tag].replace(chr(10), ' ')}\n\n")
                f.write("\n")
        print(f"\nwrote {out_path}", flush=True)
        return

    inp = build(args.prompt)

    if args.validate:
        base_lg = last_logits(inp)
        print(f"\n{'source':40} {'mods':>5} {'KL':>8} {'uniq':>5}  verdict", flush=True)
        damaged = []
        for s in cfg.sources:
            if not s.is_delta_source:
                continue
            handles = []
            try:
                adir = resolve_adir(storage, s)
                gb = (adir / "adapter_model.safetensors").stat().st_size / 1e9
                if gb > MAX_SOURCE_GB:
                    print(f"{s.name:40} {'-':>5} {'-':>8} {'-':>5}  DAMAGED (oversized {gb:.0f} GB)", flush=True)
                    damaged.append((s.name, f"oversized {gb:.0f} GB")); continue
                handles, n = apply_delta(modules, s, dev, storage)
                d = kl(last_logits(inp), base_lg)
                words = generate(inp, 40).split()
                uniq = len(set(words)) / max(len(words), 1)
                bad = d > KL_DAMAGED or uniq < UNIQ_DAMAGED
                print(f"{s.name:40} {n:>5} {d:>8.3f} {uniq:>5.2f}  {'DAMAGED' if bad else 'ok'}", flush=True)
                if bad:
                    damaged.append((s.name, f"KL={d:.1f} uniq={uniq:.2f}"))
            except Exception as e:
                print(f"{s.name:40} {'-':>5} {'-':>8} {'-':>5}  ERROR ({type(e).__name__})", flush=True)
                damaged.append((s.name, f"{type(e).__name__}"))
            finally:
                for h in handles:
                    h.remove()
                torch.cuda.empty_cache()
        print(f"\n=== DAMAGED ({len(damaged)}): " + ", ".join(n for n, _ in damaged), flush=True)
        for n, why in damaged:
            print(f"    {n}: {why}", flush=True)
        return

    # ---- assemble + generate ----
    for label in apply_io_routing():
        print(f"  IO {label}", flush=True)
    enabled = [s for s in delta_sources if args.enable_all or s.enabled]
    if enabled and cfg.merge_rule != "sum":
        print(f"  NOTE: merge_rule={cfg.merge_rule} not materialized; composing by additive sum", flush=True)
    for s in enabled:
        try:
            adir = resolve_adir(storage, s)
            if (adir / "adapter_model.safetensors").stat().st_size / 1e9 > MAX_SOURCE_GB:
                print(f"  skip {s.name} (oversized)", flush=True); continue
            _, n = apply_delta(modules, s, dev, storage, scale_mult=args.scale)
            print(f"  delta + {s.name} (w={s.weight}x{args.scale}) {n} mods", flush=True)
        except Exception as e:
            print(f"  skip {s.name} ({type(e).__name__})", flush=True)

    print(f"\n=== generation ===\n{generate(inp, args.max_new_tokens)}", flush=True)


if __name__ == "__main__":
    main()
