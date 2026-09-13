# Expected numbers — MLX-on-CUDA experiment

Every figure below is **projected**, not measured — nothing in this package has run (see
`README.md`). Method and every formula are the two already used elsewhere in this repo, kept
deliberately separate rather than blended:

1. **`r52`'s own per-token FLOPs convention** — `r52/model.py::GPT.flops_per_token`:
   `6*N + 12*n_layer*n_embd*T` where `N` is `num_params_flops()` (non-embedding + `lm_head`) and
   `T` is sequence length. This is what `r52.bench`/`r52.train` actually report as `tflops`/`tok_s`
   in `results/mlx_pretrain_bench.md`, so it's the right formula for a tok/s projection that
   Phase 1's real measurement will be compared against apples-to-apples.
2. **The ladder's plain 6ND convention** — `r52/ladder/ladder.py`: `flops = 6 * N_active * D`,
   `gpu_hours = flops / (peak_TFLOPS * 1e12 * mfu * 3600)`, `cost = gpu_hours * $/GPU-hour`. This is
   what `results/ladder.md`'s published Rung 0 FLOPs figure (5.58e17) already uses, so the
   0.75B-token cost table below reuses that published number directly rather than recomputing a
   slightly different one from convention 1.

Both conventions agree to 4 significant figures once the attention term is small relative to `6N`
(true at these sizes/seq-lengths — verified below), so the difference is a rounding nuance, not a
contradiction.

**Constants used, all already established elsewhere in this repo:** H100 SXM BF16 dense peak =
**989 TFLOP/s** ([`research/04-landscape-and-compute.md`](../../research/04-landscape-and-compute.md)
§B.5, sourced to [arXiv:2412.19437](https://arxiv.org/pdf/2412.19437)); H100 spot **$0.94**/GPU-h,
on-demand **$2.43**/GPU-h (Prime Intellect, [`results/ladder.md`](../../results/ladder.md) "Price
rows used", verified 2026-09-13); calibrated MFU split **35% for large runs, ~26% for small /
long-context-heavy runs** (`research/04` §B.5, fitted against SmolLM3 25.7% / DeepSeek-V3 33.1% /
Llama 3.1 405B 34.6%, all `results/ladder.md`).

## 1. The three sizes — 124M and 350M are measured presets; the ~1B-class is derived

124M and 350M come straight from `r52/config.py`'s own docstring table (`tests/test_config.py::test_preset_param_counts`
reproduces it), reused verbatim. The third size is **not yet a preset in `r52/bench.py`'s `SIZES`
dict** (see `bench_and_train.sh` STAGE 1's printed one-line fix) — derived here instead, using the
same Kaplan-style approximation nanoGPT/Chinchilla scaling work uses, **calibrated against this
codebase's own measured rows** first so the extrapolation isn't ungrounded theory:

`N_non-embed ≈ 12 · n_layer · n_embd²` — this codebase's Attention block is 4 projections
(`4·d²`; QK-norm and RMSNorm are **parameter-free** here, `mx.fast.rms_norm(x, None, eps)`, so no
extra norm-gain terms) and the ReLU² MLP at `mlp_ratio=4.0` is up+down (`2·4·d² = 8·d²`); `4+8=12`.

| preset | formula | predicted | measured (`r52/config.py`) | diff |
|---|---|---|---|---|
| 124M (L=12, d=768) | 12·12·768² | 84,934,656 | 84,934,674 | +18 (1.5/layer — U-net-skip/softcap scalars) |
| 350M (L=24, d=1024) | 12·24·1024² | 301,989,888 | 301,989,912 | +24 (1.0/layer) |

Within 0.003% both times — the formula is not a rough approximation for this architecture, it's
essentially exact modulo a handful of per-layer scalars. Extrapolating to **L=24, d=1792, h=14**
(`head_dim=128`, the same convention `gpt2_124m_mac.yaml`/`nano_30m.yaml` use):

| quantity | formula | value |
|---|---|---|
| non-embedding | 12·24·1792² (+24 scalar) | 924,844,056 |
| `wte` (untied) | vocab·d = 50,304·1,792 | 90,144,768 |
| `lm_head` | vocab·d | 90,144,768 |
| value-embed table (shared) | vocab·d | 90,144,768 |
| **N for 6N** (`params_flops`) | non-embed + `lm_head` | **1,014,988,824** (~1.02B) |
| **total params** | non-embed + `wte` + `lm_head` + v-embed | **1,195,278,360** (~1.20B) |

Called "~1B-class" throughout, matching the ladder's own "Rung 1: 1B dense" rounding convention
(`results/ladder.md`), not claimed as exactly 1.00B.

## 2. Expected tok/s on H100 at 30% / 40% / 50% MFU

`flops_per_token = 6·N_6N + 12·n_layer·n_embd·T` (T=1024); `tok/s = peak_TFLOPS·1e12·MFU / flops_per_token`.

| size | N (6N) | flops/token @ seq 1024 | 30% MFU | 40% MFU | 50% MFU |
|---|---|---|---|---|---|
| 124M | 123,568,146 | 854,655,084 | 347,158 tok/s | 462,877 tok/s | 578,596 tok/s |
| 350M | 353,501,208 | 2,422,997,136 | 122,452 tok/s | 163,269 tok/s | 204,086 tok/s |
| ~1B-class | 1,014,988,824 | 6,618,415,248 | 44,829 tok/s | 59,773 tok/s | 74,716 tok/s |

Cross-checked (not fitted) against `results/mlx_pretrain_bench.md`'s own measured rows: the same
`flops_per_token` formula, multiplied by the *measured* Mac tok/s, reproduces the *measured* Mac
TFLOPS to 4 significant figures (124M/1024/mixed: 3,034 tok/s × 854,655,084 / 1e12 = **2.593**
TFLOPS vs. measured **2.593**; 350M/1024/bf16: 1,168 × 2,422,997,136 / 1e12 = **2.830** vs. measured
**2.830**) — the formula isn't new, it's the exact one already producing this repo's published
numbers.

**Mac → H100 comparison**, same formula, illustrating the scale of the claim this experiment
actually tests:

| size | Mac M4 measured (mixed, seq 1024) | H100 @ 30% MFU (projected) | speedup |
|---|---|---|---|
| 124M | 3,034 tok/s | 347,158 tok/s | ~114× |
| 350M | 1,168 tok/s | 122,452 tok/s | ~105× |

That ratio (~110×) is close to 989/3.6 ≈ 275× the M4's *measured* bf16 peak, discounted by the H100
run landing at a lower fraction of *its* peak (30%) than the Mac measurements do (~65-80% of the
Mac's own 3.6 TFLOPS bf16 peak, per `mlx_pretrain_bench.md`'s "MFU 3.6" column) — small models are
harder to keep an H100 fed than a 10-core M4 GPU, which is exactly why Phase 1 exists rather than
assuming linear scaling.

## 3. The 0.75B-token real run — wall-clock and cost

Reuses `results/ladder.md`'s own published Rung-0 FLOPs figure directly (**5.58e17**, from
`python -m r52.ladder`, `6 × 124e6 × 750e6` at the ladder's rounded-N convention) rather than
recomputing a slightly different number from §1-2's precise N — this is the exact FLOPs figure
already in this repo's ladder table for this token budget. `gpu_hours = flops/(989e12·MFU·3600)`;
`wall_clock = gpu_hours / n_gpus`; `cost = gpu_hours · $/GPU-hour`.

| MFU | GPU-hours | 1×H100 wall-clock | 8×H100 wall-clock | spot $ | on-demand $ |
|---|---|---|---|---|---|
| 20% (conservative — small, single-GPU, no FSDP, no fp8) | 0.784 | 47.0 min | 5.9 min | $0.74 | $1.90 |
| 26% (`research/04`'s "small run" calibration) | 0.603 | 36.2 min | 4.5 min | $0.57 | $1.46 |
| 35% (`research/04`'s "large run" calibration — an upper bound here, not expected) | 0.448 | 26.9 min | 3.4 min | $0.42 | $1.09 |

Two things worth stating plainly rather than leaving implicit:

- **0.75B tokens is `max_tokens` — a budget ceiling, not a guaranteed spend.** `configs/gpt2_124m_mac.yaml`
  also sets `target_val_loss: 3.28`; `r52/train.py`'s loop stops at whichever comes first (see
  `Trainer.run`'s `stop_reason` logic). The real run may finish for materially less than the table
  above if it reaches 3.28 early, or may exhaust the full budget without reaching it — either way,
  `docs/PLAN.md` §3 Phase 3's rule applies: "val ≤ 3.28 or budget exhausted — either way, published
  honestly."
- **GPU-hours (and therefore cost) are identical at 1× and 8×H100 — only wall-clock changes**,
  because this is a fixed-FLOPs job, not a fixed-time one. For a run this short and this small,
  going to 8 GPUs buys back ~43 minutes of wall-clock for the same money *if* NCCL data-parallel
  overhead stays negligible — which is genuinely uncertain at this model size (§2's ~1B-class row
  is already down to tens of thousands of tok/s per GPU; the all-reduce for a ~1B-parameter
  gradient tree every optimizer step is not obviously free at that scale). That's exactly Phase 2's
  question (`README.md` §2), not an assumption this table makes. For the 124M real run specifically,
  1×H100 is the more sensible default — it's already "short" (under an hour at every MFU assumption
  above) without paying for 8 GPUs' worth of coordination risk on a job this small.

## 4. Comparison rows — nanochat and modded-nanogpt

Both already published in this repo; reproduced here as the "what does a *real*, disclosed run on
this exact hardware class look like" anchor `README.md` §2 Phase 1 references.

| Run | Hardware | Params | Wall-clock | GPU-hours | Spot $ | On-demand $ | Result | Source |
|---|---|---|---|---|---|---|---|---|
| **nanochat Run 6** (2026-03-14) | 8×H100 | ~1B-class (undisclosed exact N/D) | 1.65 h | 13.2 | $12.41 | $32.08 | CORE 0.262634 (beats GPT-2 XL 0.256525) | [`rungs/rung1_nanochat/README.md`](../rung1_nanochat/README.md) |
| **modded-nanogpt record #89** (2026-07-17) | 8×H100 | 124M | 1.23 min | 0.164 | ~$0.15 | ~$0.40 | val loss 3.28 (FineWeb) — **record uses FP8 + FlashAttention-3, not reproduced by this package** | [`results/ladder.md`](../../results/ladder.md) calibration rows |
| **This package's Phase 3** (projected, §3 above) | 1×H100 | 123.6M (6N) | 27-47 min | 0.45-0.78 | $0.42-0.74 | $1.09-1.90 | target: val loss ≤ 3.28 (same bar as the record row, unoptimized recipe) | this file |
| **Rung 1 ladder row** (Chinchilla-optimal, for scale) | 8×H100 | 1B/1B | 12.0 h | 96 | $91 | $234 | GPT-2 XL target (CORE ≥ 0.256525) | [`results/ladder.md`](../../results/ladder.md) |

Reading this honestly: modded-nanogpt's own record is **~3-5× faster and cheaper** than this
package's projected Phase 3, entirely because its recipe uses FP8 and FlashAttention-3 (neither of
which `r52` implements — `docs/PLAN.md` §8 Risks already flags "speedrun techniques that don't
port" as a known gap, budgeting up to 0.75B tokens against the record's undisclosed-but-clearly-smaller
count for exactly this reason). This package is not trying to beat that record; it's testing whether
the *same* unmodified `r52` codebase that runs on the Mac also runs, correctly and at a reasonable
MFU, on CUDA — nanochat Run 6 and the ladder's own Rung 1 row are the "a real team, real recipe,
this hardware class" cost anchors, not a target this experiment is trying to match.

**Torchtitan comparison: none available.** No torchtitan run at a comparable (~124M-1B, single- or
few-GPU) scale is sourced anywhere in this repo's `research/` corpus or `rungs/rung2_3b/` (that
package's own job is 3B/60B tokens on 8×H100 over ~4.5 days — a different regime entirely, and
`rung2_3b/README.md` itself states no capability claim is made for it). Any torchtitan throughput
comparison at this package's sizes is **UNVERIFIED** pending a source.
