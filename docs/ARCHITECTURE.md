# road-to-52 — Architecture & Build Spec

This document is the contract between the planner (Claude Fable 5.1) and the builder agents
(Claude Opus / Sonnet). It is deliberately concrete. If something here is wrong at build time,
fix it and note the deviation in `docs/DEVIATIONS.md` rather than silently diverging.

## 0. What we are building (one paragraph)

A complete, Apple-Silicon-native (MLX) LLM pipeline — data → tokenizer → pretraining →
midtraining → SFT → RL → evaluation → chat — engineered to be the first *maintained, well-licensed,
rigorously benchmarked* MLX pretraining stack (as of 2026-09 none exists: see
`research/02-open-training-stack.md` §"Three gaps"), plus a public **ladder** of named-model beats
at documented cost, and a **gap tracker** that measures every model against Claude Fable 5.1's
published benchmark scores. Rung 0 of the ladder is a from-scratch GPT-2-small-class model trained
on a 16 GB Mac Mini M4 for $0.

## 1. Hard constraints (never violate)

1. **No Claude-generated data anywhere in training.** Anthropic's terms forbid training competing
   models on Claude outputs. Training data must be open datasets or outputs of open-weight
   models with permissive licenses (DeepSeek/Qwen/OLMo/Kimi families). Datasets flagged as
   Claude-distilled in `research/03-open-data.md` are excluded. Claude models write *code and docs*
   here, never training text.
2. **The machine stays usable.** The Mac Mini (16 GB) also runs Hermes Agent 24/7 and Claude Code.
   Any training or eval process must: run under `nice -n 10`, call `mx.set_memory_limit()` to at
   most **6 GiB** for pretraining runs (8 GiB only for explicitly approved fine-tuning runs),
   set `mx.set_cache_limit()` to ≤ 1 GiB, and write all large artifacts to
   `/Volumes/SammyDisk/road-to-52-data/` (symlinked as `data/`, `runs/`, `models/` in the repo).
3. **Honest numbers only.** Every reported score carries: the exact eval command, the sample
   count / `--limit`, the tokenizer, sequence length, the git commit, and the wall-clock. No
   number without a reproduction path. Comparisons to published numbers must state the eval
   conditions of *both* sides.
4. **Licenses.** Apache-2.0 for our code. We may *read* MIT/BSD/Apache code (nanochat,
   modded-nanogpt, mlx-examples, mlx-lm) and port ideas with attribution in the file header. We
   may not copy code from repos with no license (e.g. `karpathy/autoresearch`,
   `mlfoundations/evalchemy`, `Doriandarko/MLX-GRPO`). Note LGPL for `mergekit` (use as a CLI
   dependency only, never vendor).
5. **Python 3.12 via `uv`** (the repo's `.venv`). No global pip installs (PEP 668). MLX ≥ 0.32,
   mlx-lm ≥ 0.31.3. PyTorch is *not* a dependency of the MLX path.

## 2. Repository layout

```
road-to-52/
  README.md                 public-facing; honest framing; ladder table; quickstart
  docs/
    PLAN.md                 master plan (the bar, the recipe, the ladder, phases)  [planner]
    ARCHITECTURE.md         this file
    DEVIATIONS.md           builder notes where reality differed from the spec
    RESULTS.md              living results log (auto-appended by scripts)
  research/                 the six research reports (sourced, dated 2026-09-13)
  r52/                      the Python package (MLX-native)
    __init__.py
    config.py               dataclasses: ModelConfig, TrainConfig, DataConfig (+ YAML I/O)
    model.py                GPT in MLX (see §3)
    optim.py                Muon + AdamW routing, WSD schedule, grad clipping
    data.py                 fineweb10B-gpt2 shard reader/downloader, batching, val split
    tokenizer.py            GPT-2 tiktoken wrapper (Rung 0) + BPE training (later rungs)
    train.py                pretraining loop (see §4)
    bench.py                MLX pretraining throughput benchmark (see §5)
    checkpoint.py           save/load (safetensors weights + optimizer state + config + step)
    export.py               convert a checkpoint to an mlx-lm-loadable directory (see §7)
    eval/
      __init__.py
      val_loss.py           fixed-val-set loss + bits-per-byte
      hellaswag.py          HellaSwag loglikelihood eval in pure MLX
      core.py               DCLM CORE score (port of nanochat's base_eval, MIT, attributed)
      report.py             writes results JSON + appends to docs/RESULTS.md
    ladder/
      __init__.py
      ladder.py             compute-ladder calculator (6ND FLOPs, MFU, GPU $/h → hours, $)
      prices.yaml           GPU rental prices with source URL + date per row
    bar/
      __init__.py
      bar.yaml              published benchmark scores (Fable 5.1, Opus 5, Sonnet 5, open models)
                            with source URL + eval conditions per number
      gap.py                renders the gap table (markdown + JSON) from bar.yaml + our results
  configs/
    nano_30m.yaml           Run A: ~30-40M params, seq 1024, ~0.4B tokens (~1 day)
    gpt2_124m_mac.yaml      Run B: 124M, seq 1024, ≤0.75B tokens, target val loss ≤ 3.28
    medium_350m.yaml        reference only (not runnable in reasonable time locally)
  scripts/
    prepare_data.py         download N train shards + val shard of kjj0/fineweb10B-gpt2
    train.sh                nohup + nice wrapper; writes runs/<name>/{log.jsonl,ckpt/,stdout.log}
    bench.sh                runs r52.bench and writes results/mlx_pretrain_bench.{md,json}
    eval.sh                 runs eval suite on a checkpoint; appends to docs/RESULTS.md
  results/                  committed JSON/MD results (never raw model files)
  tests/                    pytest; every test must run in < 60 s total on the M4 CPU/GPU
  pyproject.toml            package `r52`, deps pinned by uv.lock
```

## 3. Model (`r52/model.py`)

Decoder-only transformer, "nanochat/modded-nanogpt lineage", implemented with MLX fast kernels.

**Mandatory features (config-flagged, default ON for `gpt2_124m_mac.yaml`):**

| Feature | Detail | Why |
|---|---|---|
| Pre-norm RMSNorm | `mx.fast.rms_norm`, no learnable bias | 2026 default |
| RoPE | `mx.fast.rope`, base 10000 (config), applied to q,k | 2026 default |
| Attention | `mx.fast.scaled_dot_product_attention(q, k, v, scale=..., mask="causal")`; n_heads configurable, GQA via `n_kv_heads` (default = n_heads) | fused kernel |
| QK-norm | RMSNorm on q and k per head before RoPE (no learnable params) | stabilizes Muon training (nanochat) |
| MLP | ReLU² (`relu(x)**2`) with 4× expansion (nanochat/speedrun) — SwiGLU as an option | speedrun lineage |
| Untied embeddings | separate `wte` and `lm_head`; `lm_head` zero-init | speedrun |
| Logit softcap | `softcap * tanh(logits / softcap)`, softcap=15 (nanochat) | stability |
| Value embeddings | per-layer (or shared across alternating layers) token-indexed value residual, as in modded-nanogpt / nanochat | large token-efficiency win |
| U-net skip connections | encoder half stores residuals, decoder half adds `skip_weight * stored` (learnable scalar) | speedrun |
| Zero-init projections | attention out-proj and MLP down-proj zero-init | speedrun |
| Mixed precision | master params **fp32**; matmuls in **bf16** via casting inside a custom `Linear`; loss in fp32 | mlxgpt lesson: pure bf16 converges worse |
| Gradient checkpointing | optional, per block, via `mx.checkpoint` | memory |

**Config presets** (`ModelConfig`): `n_layer, n_embd, n_head, n_kv_head, vocab_size (50257 → pad to 50304), block_size (1024), rope_base, softcap, use_value_embeds, use_unet_skips, mlp='relu2'|'swiglu', mlp_ratio, qk_norm=True, tie_embeddings=False`.

- `nano_30m`: n_layer=8, n_embd=512, n_head=8 (head_dim 64).
- `gpt2_124m_mac`: n_layer=12, n_embd=768, n_head=6 (head_dim 128, as modded-nanogpt) — record the exact parameter count in the run log.

**Interface** (stable; eval and export depend on it):

```python
class GPT(nn.Module):
    def __init__(self, cfg: ModelConfig): ...
    def __call__(self, idx: mx.array) -> mx.array      # (B, T) int32 -> (B, T, vocab) fp32 logits (softcapped)
    def loss(self, idx: mx.array, targets: mx.array) -> mx.array   # mean CE in fp32, ignore_index=-1
    def num_params(self, non_embedding: bool = False) -> int
    def flops_per_token(self) -> int                    # 6N + attention term (12 * n_layer * n_embd * T) — document the formula used
```

Check `python -c "import mlx_lm.models.nanochat as m; help(m)"` — mlx-lm ships a `nanochat` model
class. If our weight layout can mirror it, `export.py` becomes a rename; if not, `export.py` writes
a self-contained mlx-lm plugin (see §7). Decide early and record the decision in DEVIATIONS.md.

## 4. Training loop (`r52/train.py`)

- Data: `kjj0/fineweb10B-gpt2` shards (`fineweb_train_*.bin`, `fineweb_val_000000.bin`, llm.c
  format: 256×int32 header [magic 20240520, version 1, n_tokens], then uint16 tokens). Reader is
  memory-mapped; batches are contiguous `(B, T+1)` windows; a deterministic shard/offset cursor is
  saved in the checkpoint so resume is exact. Val loss = mean CE over the **first 10,485,760 tokens
  of the val shard** (same as modded-nanogpt), computed at our training `block_size` in
  non-overlapping windows; report both loss and bits-per-byte (bpb, tokenizer-agnostic).
- Optimizer: **Muon** (`mlx.optimizers.Muon`; check its signature with `help()`) for all 2-D hidden
  matrices; **AdamW** for embeddings, lm_head, value embeddings, and scalars; routed with
  `mlx.optimizers.MultiOptimizer`. LRs and weight decay from config; defaults taken from
  modded-nanogpt's current record (document the source values and any scaling for batch size).
- Schedule: warmup (config, e.g. 0 or few hundred steps) → constant → linear cooldown over the
  final `cooldown_frac` (default 0.4 of the run, as modded-nanogpt). Batch size in tokens is
  config (`tokens_per_step`, e.g. 2^17 = 131072 for 124M ≈ speedrun's 8×64k / scaled), built from
  `micro_batch × block_size × grad_accum`.
- Step: `nn.value_and_grad` on `model.loss`; grads accumulated in fp32; global-norm clip (config);
  `mx.eval` once per optimizer step; compile the loss/grad function with `mx.compile` where it
  works (measure: if compile doesn't help, leave a flag).
- Logging: `runs/<name>/log.jsonl` — one line per log interval with `step, tokens, loss, lr,
  tok_s, tflops, mfu (vs 4.26 TFLOPS fp32 theoretical; also vs 3.6 TFLOPS measured bf16 matmul
  peak), peak_mem_gb, eta_h`; val entries with `val_loss, val_bpb`; plus `stdout.log`.
  Print a compact rich progress line. No wandb (no API keys).
- Checkpoint every `ckpt_interval` steps and on SIGTERM/SIGINT: weights (safetensors), optimizer
  state, data cursor, RNG, config, step. `--resume` restores exactly. Keep last 2 + best-val.
- Time/step budget flags: `--max-tokens`, `--max-hours`, `--target-val-loss` (stop early when hit,
  then run the cooldown — implement as: if the target is hit at any val, finish normally; the
  planner picks the token budget).
- Memory guard: `mx.set_memory_limit(cfg.memory_limit_gb * 2**30)`; `mx.set_cache_limit(1 << 30)`.
- `scripts/train.sh <config> <run_name>` runs `nice -n 10 nohup python -m r52.train ...` and prints
  the PID and the log path. Training must survive the Claude Code session ending.

## 5. Throughput benchmark (`r52/bench.py`) — a headline artifact

Nobody has published an MLX *pretraining* throughput benchmark (research/02 §gaps, research/05 §2.2).
`r52.bench` sweeps `{30M, 60M, 124M, 350M} × block_size {512, 1024} × micro_batch {auto}` for
`~30 s` each after warm-up and writes `results/mlx_pretrain_bench.json` and `.md` with columns:
params, seq, micro_batch, tokens/s, effective TFLOPS, MFU (both denominators), peak memory GiB,
precision mode (mixed / pure bf16 / fp32), compile on/off, mlx version, macOS version, chip,
date. Also benchmark the `mixed vs bf16 vs fp32` axis at 124M. The results table gets pasted into
README. Keep each config under the 6 GiB memory limit; skip and mark configs that don't fit.

## 6. Evaluation (`r52/eval/`)

- `val_loss.py`: as in §4, callable on any checkpoint.
- `hellaswag.py`: 10,042 validation examples, loglikelihood of each ending conditioned on the
  context, GPT-2 tokenizer, accuracy + acc_norm (length-normalized) — the same protocol as
  llm.c's `hellaswag.py`/lm-eval. Reference: GPT-2 124M = 29.4 % (acc_norm ≈ 31.1 %) — verify
  the reference by running *our* eval on the HF `gpt2` weights loaded into our GPT class
  (`export.py` must also support import of HF gpt2 for this apples-to-apples check).
- `core.py`: DCLM CORE (22 tasks) — port nanochat's `scripts/base_eval.py` + `nanochat/core_eval.py`
  (MIT) to our model; downloads nanochat's eval bundle; outputs the CORE score in nanochat's
  normalization so numbers are comparable with `dev/LEADERBOARD.md` (GPT-2 XL = 0.2565). Run it
  on HF gpt2 (124M) as well to establish the GPT-2-small reference.
- `report.py`: writes `results/<run>/<eval>.json` with the reproduction metadata from §1.3 and
  appends a row to `docs/RESULTS.md`.

Later phases add `mlx_lm.evaluate` (lm-eval bridge) for MMLU-Pro/GPQA/GSM8K/IFEval on exported
and post-trained models, and `inspect_ai` for agentic evals.

## 7. Export (`r52/export.py`)

`python -m r52.export runs/<name>/ckpt/best models/<name>-mlx/` writes an mlx-lm-loadable
directory (`config.json`, `model.safetensors`, `tokenizer.json`/tiktoken files) so that
`mlx_lm.generate`, `mlx_lm.server`, `mlx_lm.lora`, `mlx_lm.evaluate` all work. If mlx-lm's
`nanochat` class matches our layout, target it; otherwise ship `r52/mlx_plugin/` with a model file
and document `--model-type` loading. Round-trip test: our logits vs mlx-lm logits on 8 prompts,
max abs diff < 1e-2 (bf16).

## 8. Compute ladder (`r52/ladder/`) and the bar (`r52/bar/`)

- `ladder.py`: inputs — model params N, tokens D, MFU, hardware peak TFLOPS, $/GPU-hour; outputs —
  FLOPs (6ND), GPU-hours, wall-clock at k GPUs, dollars. CLI renders the ladder table for a list of
  rungs defined in `ladder/rungs.yaml` (Mac M4 measured; 8×H100 rental; 64×H100; 1k×B200; etc.),
  reading `prices.yaml` (each price row: provider, GPU, $/h, on-demand/spot, source URL, date).
- `bar.yaml`: for each benchmark — name, what it measures, url, eval conditions; for each model
  (claude-fable-5-1, claude-opus-5, claude-sonnet-5, top open-weight models, gpt2-124m, our runs) —
  score, source URL, date, conditions (`effort`, tools, pass@k, parallel compute). Numbers come
  from `research/01-benchmarks-and-bar.md`; every number must carry its source.
- `gap.py`: renders `results/GAP.md`: rows = benchmarks, columns = models, plus "gap to Fable 5.1"
  and "best open vs Fable 5.1" columns; marks unavailable cells as `—` (never 0).

## 9. Testing & quality bar

- `uv run pytest -q` must pass in < 60 s. Tests: model forward shapes and dtype; loss decreases
  over 30 steps on a synthetic dataset (tiny config); checkpoint save→load→identical loss;
  data cursor resume determinism; Muon+AdamW routing covers every parameter exactly once;
  bench runs one tiny config; hellaswag scoring on 5 hand-checked examples; export round-trip
  on a tiny model.
- Type hints everywhere; `ruff` clean; docstrings state units (tokens, seconds, GiB).
- No hidden network calls in tests (dataset downloads happen in scripts, cached under `data/`).

## 10. Run plan (planner-owned; here for context)

| Run | Config | Params | Tokens | Purpose | Est. wall-clock on M4 |
|---|---|---|---|---|---|
| bench | — | 30M–350M | — | first published MLX pretraining throughput table | ~1 h |
| A: `nano_30m` | seq 1024 | ~35M | 0.4B | prove the whole pipeline end-to-end; base for SFT/RL demos | ~1 day |
| B: `gpt2_124m_mac` | seq 1024 | 124M | ≤0.75B | **Rung 0**: match GPT-2 small (val loss ≤ 3.28, HellaSwag ≥ 29.4 %, CORE ≥ gpt2 ref) | 4–6 days |

Runs A and B never overlap (one GPU). Builders test on `tiny` configs (2 layers, 64-dim) only.
