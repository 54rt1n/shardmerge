# ShardMerge Module Overview

ShardMerge is organized as one package, `shard`, containing the original
shard-streaming merge pipeline plus several newer subpackages for delta
extraction, compression, validation, and adapter-space merging.

```
shard/
├── __main__.py      CLI: merge, copy-model, generate
├── config.py        MergeConfig / MergeModel (YAML schema for `shard merge`)
├── download.py      concurrent HF shard downloads with claim-counted caching
├── index.py         HFMultiModelIndex (streaming) / OfflineMultiModelIndex (local)
├── writer.py        ModelWriter: mirrors the base index, resumable shard writes
├── inference.py     InferenceEngine backing `shard generate`
├── merge/           full-model merge operators
├── tensor/          low-level tensor math shared by the operators
├── delta/           composable tensor-delta patching primitives
├── exlora/          delta extraction & compression (LoRA / NVFP4) — own CLI
├── ranker/          rank & signal-level-set validation utilities
├── loramerge/       full-frame merging of extracted LoRA/dense artifacts
└── loraplay/        activation-space scoring of extracted adapters — own CLI
```

## Core pipeline (`shard`)

The original tool: merge N finetunes onto a base model without ever
materializing any model fully in RAM or on disk.

- **`__main__.py`** — the `python -m shard` click CLI with three commands:
  `merge <config.yaml>` (runs `merge.fast_fourier.FourierMerge`),
  `copy-model` (fetch config/tokenizer files from HF), and `generate`
  (stream text from a local model, optionally 4/8-bit).
- **`config.py`** — `MergeConfig` (output base model, dtype, dirs) and
  `MergeModel` (one finetune: `model`, `base`, `alpha`, `is_input`/`is_output`
  routing for embeddings/head, `start_layer`/`end_layer` windowing).
- **`index.py`** — resolves tensor names to shards for many models at once.
  `HFMultiModelIndex` streams shards from HuggingFace on demand;
  `OfflineMultiModelIndex` serves the same interface from local directories.
  Layer component sets are discovered per-model from the weight map, so any
  architecture layout (with or without attention biases, qk-norms, etc.)
  works.
- **`download.py`** — `DownloadManager`: concurrent downloads with progress
  callbacks and per-shard claim counting so a shard can be deleted as soon as
  every tensor in it has been consumed (`clean_cache`).
- **`writer.py`** — `ModelWriter`: writes merged tensors into safetensors
  shards that mirror the base model's index. Tracks written tensors so an
  interrupted merge resumes where it left off; `finalize()` verifies
  completeness.
- **`inference.py`** — thin transformers wrapper for smoke-testing merges.

## `shard.merge` — full-model merge operators

All operators extend `MergeTensorsBase` (`base.py`), which owns the async
task orchestration (`TaskRunner`) over shards/layers.

- **`fast_fourier.py`** — `FourierMerge`, the operator used by the CLI:
  2D-FFT domain interpolation of deltas with a disk cache
  (`TensorDiskCache`) for intermediate results.
- **`fourier.py`** — earlier, simpler FourierMerge kept for reference.
- **`addition.py` / `taskaddition.py`** — plain delta addition and
  task-arithmetic (sign-agreement) variants.

## `shard.tensor` — tensor math

- **`functions.py`** — the primitives: `slerp`, 2D FFT transforms,
  FFT-component interpolation and task arithmetic, magnitude
  normalization, pairwise correlation helpers.
- **`util.py`** — device selection and a CUDA memory profiler context.

## `shard.delta` — delta patching primitives

The shared, codec-agnostic model of what a stored delta *is*. Higher-level
extractors produce the pieces; this package defines how they compose:

```
patched_weight = base_weight + dense_codec_delta + lora_delta + residual_delta
```

- **`terms.py`** — the term types: `DenseDelta`, `LoraDelta`,
  `IndexedResidualDelta`.
- **`codecs.py`** — `DeltaCodec` protocol (identity, round-to-dtype, …).
- **`patch.py`** — `PatchStack` / `apply_patch_stack`: ordered application.
- **`manifest.py`** — serializable description of a patch artifact.
- **`metrics.py`** — Gram/metric-space norms (`metric_norm`,
  `relative_metric_error`) for measuring error where it matters
  behaviorally, not just in Frobenius.

## `shard.exlora` — delta extraction & compression

Extracts a compact artifact from `Delta = W_target − W_base`. Own CLI:
`python -m shard.exlora extract <config.yaml>` and `build-map`. Two artifact
families:

- **`lora`** — rank-adaptive truncated SVD written as a PEFT adapter
  (`svd.py`, `peft.py`). Rank per layer is chosen by `rank_policy`:
  target error, spectral signal detection, or both. `knee.py` detects the
  bf16 noise-floor cliff that distinguishes a true low-rank (merged-LoRA)
  delta from a genuinely dense finetune.
- **`nvfp4_residual`** — dense NVFP4 quantized delta map (`nvfp4.py`,
  `nvfp4_kernel.py`) with optional floor correction and a sparse super-floor
  residual sidecar (`nvfp4_residual.py`), for deltas whose behavior lives in
  the dense floor.

Supporting pieces: `comb.py` (LoRA corrector for the quantization residual,
fit in the activation H-metric), `projection.py`/`qat_projection.py`/
`qat_map.py` (QAT survival-map projection), `sensitivity_nvfp4.py`,
`config.py` (`ExtractConfig`), `extract.py` (the driver; writes an
`EXTRACTION_REPORT.md` next to the artifact).

## `shard.ranker` — rank & signal validation

Utilities used to validate extraction decisions:

- **`level_set.py`** — detect the rank level-set of a singular spectrum and
  score signal recovery (`SignalRecoveryGates`).
- **`codec_floor.py`** — fit the error floor a codec imposes, so residual
  spend can target only blocks above it.
- **`layer_position.py`** — layer index → depth fraction and
  early/middle/late bucketing for the per-position diagnostics in
  extraction reports.

## `shard.loramerge` — full-frame artifact merging

Merges *extracted* sources (PEFT LoRAs, dense/NVFP4 ExLoRA artifacts,
checkpoint deltas) into a full model, with policy-safety tooling around it.

- **`sources.py`** — `DeltaSource` implementations for each artifact kind.
- **`merge.py`** — the operators: weighted sums, `fourier_slerp_merge`,
  norm-aware slerp, sign-agreement voting, and the isometric
  (rotation-averaging) merge.
- **`policy.py`** — policy-fingerprint scan: measures each candidate's
  first-token intent (KL from base, promoted/suppressed tokens) before
  committing it to a merge.
- **`residence.py`** — on-policy residence gate: scores merged-model
  generations for off-policy collapse modes (repetition loops, entropy
  collapse, word salad).
- **`scan.py`** — inspect a merge config without merging.
- **`runner.py`** — the entry point:
  `python -m shard.loramerge.runner --config <yaml> [--dry-run]`
  (starter config at `configs/example-loramerge.yaml`).

## `shard.loraplay` — adapter scoring

Standalone evaluation for extracted adapters: does the adapter reproduce the
finetune's *behavior*, not just its weights? CLI: `python -m shard.loraplay`
with `harvest` (collect per-layer input second moments / Grams from the base
model), `score` (activation-space reconstruction error of an adapter against
the true finetune, in the harvested metric), and `play` (run the adapter
live and accumulate activation error over real generations).

## Tests and scripts

`tests/` mirrors the package layout (delta, exlora, loramerge, loraplay,
merge, ranker, tensor). Run with `uv run pytest`.

Tracked utility scripts: `scripts/convert_pytorch_to_hf.py`,
`scripts/rewrite_reorder.py` (reorder tensors within shards),
`scripts/verify_safetensors.py`.
