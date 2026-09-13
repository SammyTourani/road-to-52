# Build your own LLM on a Mac in an afternoon (and a GPT-2-class one in three days)

A complete, copy-pasteable walkthrough of the road-to-52 pipeline — data → tokenizer →
pretraining → midtraining → SFT → RL → eval → chat — on a 16 GB Apple-Silicon Mac. Every command
was checked against its `--help` output on 2026-09-13; every timing is a measured number with its
source, a number derived from one (labeled "derived"), or marked **not yet measured**. Nothing
here is invented. Background: [`README.md`](../README.md), [`docs/PLAN.md`](PLAN.md) (the master
plan), [`docs/ARCHITECTURE.md`](ARCHITECTURE.md) (the build spec),
[`docs/DEVIATIONS.md`](DEVIATIONS.md) (reality vs. spec — this doc's Troubleshooting source),
[`docs/RESULTS.md`](RESULTS.md) (the results log).

**Conventions.** Run from the repo root with `.venv` active. One run-name thread runs through
every stage below: pretrain `nano-a` → midtrain `m1` → SFT `sft_nano` → RL `r1` — not arbitrary:
`sft_nano.yaml` already defaults `init_from: runs/m1/ckpt/best`, `rl_nano.yaml` already defaults
`model: models/nano-sft-mlx`.

**Contents:** [0 Setup](#0-setup) · [1 Data](#1-data) · [2 Throughput](#2-throughput-check) ·
[3 Pretraining](#3-pretraining) · [4 Eval](#4-evaluation) · [5 Export+chat](#5-export--chat) ·
[6 Midtrain](#6-midtraining) · [7 SFT](#7-sft) · [8 RL](#8-rl) · [9 Publish](#9-publishing) ·
[10 Bigger](#10-going-bigger) · [Troubleshooting](#troubleshooting) · [Not this](#what-this-is-not)

## 0. Setup

Apple Silicon, macOS, Python 3.12. Built and measured on a Mac Mini M4, 16 GB unified memory —
should work on any 16 GB+ Apple Silicon Mac, just slower or faster with GPU core count.

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh     # uv, if you don't have it

git clone https://github.com/SammyTourani/road-to-52 && cd road-to-52
uv venv --python 3.12 .venv
VIRTUAL_ENV=$PWD/.venv uv pip install -e ".[dev]"
.venv/bin/python -m pytest -q          # per README's Quickstart: < 60 s
```

`.[dev]` adds `pytest`/`pytest-timeout`/`ruff` on top of the runtime deps (`mlx`, `mlx-lm`,
`datasets`, `reasoning-gym`, `mlx-lm-lora`, ... — `pyproject.toml`). The suite (268 tests as of
2026-09-13, `pytest --collect-only -q`) runs real tiny-MLX steps, so budget more than 60 s if the
GPU is shared with a training run ([Troubleshooting](#troubleshooting)). CI runs this exact
two-line install on a macOS-15 arm64 runner (`.github/workflows/ci.yml`) on every push.

**External-disk symlinks.** `docs/ARCHITECTURE.md` §1.2 keeps large artifacts off the internal
SSD; this repo's own build symlinks three directories to an external disk:

```bash
mkdir -p /Volumes/SammyDisk/road-to-52-data/{data,runs,models}
ln -s /Volumes/SammyDisk/road-to-52-data/data   data
ln -s /Volumes/SammyDisk/road-to-52-data/runs   runs
ln -s /Volumes/SammyDisk/road-to-52-data/models models
```

No external disk? `mkdir data runs models` locally instead — same behavior, just budget a few GB
of free space (Step 1 alone is ~1.7 GB).

**The memory rule and `nice`.** Every training/eval process runs under `nice -n 10` and caps
MLX's GPU memory with `mx.set_memory_limit()` — **≤ 6 GiB for pretraining, ≤ 8 GiB only for
explicitly-approved fine-tuning** — plus `mx.set_cache_limit()` ≤ 1 GiB (`ARCHITECTURE.md` §1
rule 2). Every shipped config and `scripts/*.sh` wrapper already does this for you. One catch
worth knowing before you hit it: this limit is a *guideline*, not a hard cap — see
[Troubleshooting](#troubleshooting).

## 1. Data

FineWeb, re-tokenized with the GPT-2 BPE and shipped as `kjj0/fineweb10B-gpt2` — the same files
modded-nanogpt and llm.c train on, which is what makes Rung 0's target numbers (Step 3) directly
comparable. Format: 256 little-endian int32 header words (`magic 20240520`, `version 1`, `ntok`,
zero padding) then `ntok` little-endian uint16 token ids (`r52/data.py`).

```bash
.venv/bin/python scripts/prepare_data.py --train-shards 8
```

Not `--shards` — an older line in the README's own Quickstart says that, but the real flag is
`--train-shards` (verified: `scripts/prepare_data.py --help`; see this doc's closing report).
Each shard is **exactly 100,000,000 tokens / 200 MB on disk** (measured:
`scripts/prepare_data.py --list`, 2026-09-13); a matching val shard fetches automatically unless
you pass `--no-val`.

| you want | shards | tokens | disk | notes |
|---|---|---|---|---|
| a quick pipeline test | 1 | 100M (repeats) | ~400 MB | `tiny_test.yaml` |
| Run A, `nano_30m` (0.4B budget) | 4–5 | 400–500M | ~1.0 GB | no repeats over the run |
| Run B, `gpt2_124m_mac` (0.75B budget) | 8 | 800M | ~1.7 GB | no repeats over the run |

The reader wraps around the shard list once you've consumed more tokens than are on disk (an
epoch counter lives in the checkpoint), so under-fetching just means more repetition, not a
failure. `--train-shards 8` is what Rung 0 uses, and is on disk on this machine right now.

## 2. Throughput check

```bash
scripts/bench.sh                                         # full sweep: 4 sizes x 2 seqs x 3 precisions
scripts/bench.sh --sizes 30M,124M --precisions mixed --seqs 1024   # subset
scripts/bench.sh --seconds 10                             # quick pass
```

The first published MLX *pretraining* throughput benchmark — every other "MLX tok/s" figure in
circulation is inference decode, not training (`r52.bench --help`). It runs the real training
step (grad-accumulated micro-steps, fp32 grad accumulation, Muon+AdamW update) timed after
warm-up, and writes [`results/mlx_pretrain_bench.{md,json}`](../results/mlx_pretrain_bench.md).
The full 24-row sweep takes **roughly 1–2 hours** here; chunk it with `--append` to get the
machine back in between.

**Reading MFU.** `FLOPs/token = 6*(non_embedding_params + lm_head_params) + 12*n_layer*n_embd*seq`.
Every row reports MFU against **two** denominators: **4.26 TFLOPS** (this chip's theoretical fp32
peak, arXiv 2502.05317 Table 1) and **3.6 TFLOPS** (a bf16 4096³ matmul measured on this machine,
the more honest one since the model's matmuls run in bf16). Measured 2026-09-13, Apple M4 (10 GPU
cores), mlx 0.32.2:

| model | seq | precision | tok/s | TFLOPS | MFU (4.26) | MFU (3.6) | peak GiB |
|---|---|---|---|---|---|---|---|
| 30M | 1024 | mixed | 6,479 | 2.31 | 54% | 64% | 4.66 |
| 60M | 1024 | mixed | 3,647 | 2.34 | 55% | 65% | 4.55 |
| **124M** | **1024** | **mixed** | **3,034** | **2.59** | **61%** | **72%** | **5.11** |
| 124M | 512 | bf16 | 3,563 | 2.84 | 67% | 79% | 5.06 |
| 350M | 1024 | bf16 | 1,168 | 2.83 | 66% | 79% | 4.99 |

fp32 is ~30% slower than mixed; pure bf16 is ~8% faster but converges worse (`research/05`), why
both shipped configs use `precision: mixed` (fp32 master params, bf16 matmuls). 350M/mixed and
350M/fp32 are **skipped** — even at micro_batch=1 they peak at 8.06–8.21 GiB, over the 6 GiB cap.
Full 24-row table: [`results/mlx_pretrain_bench.md`](../results/mlx_pretrain_bench.md).

## 3. Pretraining

```bash
scripts/train.sh configs/nano_30m.yaml nano-a
```

Backgrounds `nice -n 10 nohup python -m r52.train configs/nano_30m.yaml --run-name nano-a`,
prints the PID, writes `runs/nano-a/{log.jsonl,stdout.log,ckpt/}`, and survives your shell (and
Claude Code session) exiting. Stop cleanly with `kill -TERM <PID>` (checkpoints, then exits);
resume with:

```bash
scripts/train.sh configs/nano_30m.yaml nano-a --resume            # newest checkpoint
scripts/train.sh configs/nano_30m.yaml nano-a --resume runs/nano-a/ckpt/step_1500  # explicit
```

`--max-hours` counts cumulatively across resumes; `--target-val-loss` stops at the first
validation at or below a given fp32 loss; neither reshapes the LR schedule (`-o max_tokens=N`
does — the schedule is always laid out over the config's `total_steps`).

**Reading the logs.** `runs/<name>/log.jsonl` is one JSON object per `log_interval` steps:

| field | meaning |
|---|---|
| `step` / `tokens` | optimizer step / total tokens consumed |
| `loss` | training CE, nats/token, **bf16** log-sum-exp (only *val* loss is fp32 — [Troubleshooting](#troubleshooting)) |
| `lr` / `lrs` | the Muon LR and the five AdamW group LRs (`muon`,`embed`,`vembed`,`head`,`scalar`) |
| `grad_norm` | pre-clip global gradient norm (clipping is off in both shipped configs) |
| `tok_s` / `tflops` | this interval's throughput |
| `mfu_theoretical` / `mfu_measured` | against the same 4.26 / 3.6 TFLOPS as Step 2 |
| `peak_mem_gb` / `eta_h` | GiB, and hours to `max_tokens` at the current rate |

Real example, captured from this machine's live `gpt2-124m-mac` run, 2026-09-13:

```json
{"step": 100, "tokens": 13107200, "loss": 5.53403, "lr": 0.05, "tok_s": 1301.3,
 "mfu_theoretical": 0.2611, "mfu_measured": 0.3089, "peak_mem_gb": 5.107, "eta_h": 157.3}
```

`tok_s`/MFU here trail Step 2's isolated numbers because a second MLX process (a CORE eval) was
sharing the GPU — see [GPU contention](#troubleshooting). Every ~`val_interval` steps (250 for
both shipped configs) a `val`-type line appears instead, carrying `val_loss` and `val_bpb` (bits
per byte, tokenizer-independent) in place of the training fields: for `nano_30m` that's step 250
= 16.4M tokens in (≈ 42 min at Step 2's isolated 6,479 tok/s); for `gpt2_124m_mac`, 32.8M tokens
(≈ 3.0 h at 3,034 tok/s, longer on a shared GPU). The `start` record also reports three different
parameter counts for one model (`params_total` 200,835,090, `params_non_embedding` 84,934,674,
`params_flops` 123,568,146 — the classic GPT-2-small matmul count behind the "124M" name); see
`DEVIATIONS.md` #11 if the log's numbers surprise you.

**Rung 0: `configs/gpt2_124m_mac.yaml`.** 12 layers, 768-dim, 6 heads (head_dim 128, as
modded-nanogpt), 0.75B-token budget (5,722 steps at 131,072 tokens/step). Every optimizer
hyperparameter's provenance — copied verbatim from modded-nanogpt's record vs. deliberately
different (plain Muon, not the record's NorMuon; untied embeddings from step 0) — is written out
in the config's own header comment; read it rather than re-deriving it.

```bash
scripts/train.sh configs/gpt2_124m_mac.yaml gpt2-124m-mac
```

**Time.** Step 2's isolated 124M/1024/mixed row (3,034 tok/s) implies 750M tokens ≈ 68.7 h ≈
**2.9 days**, matching README/PLAN.md's "~3 days". Two other numbers exist for the same rung,
both caveated: `results/ladder.md` computes 119 h (5.0 d) but flags it as built on an older,
naive 111M-parameter throughput figure; `ARCHITECTURE.md` §10's original planning estimate was
"4–6 days". Treat "~3 days" as the best estimate on an idle machine, and expect worse on a shared
one — see [Troubleshooting](#troubleshooting).

**The two bars.** The target is `gpt2_124m_mac.yaml`'s `target_val_loss: 3.28` — the
modded-nanogpt/llm.c speedrun bar: a **from-scratch 124M reproduction trained on this same
FineWeb split**. That's *harder* than matching OpenAI's actual released `gpt2` checkpoint, which
was trained on WebText and pays a distribution-shift penalty read out on FineWeb — measured here,
`r52.eval.val_loss` on the real `openai-community/gpt2` weights gets **3.4471 nats/token** on
this split (Step 4), not 3.28. Both are legitimately "GPT-2 small"; only one was trained on this
data. HellaSwag has no such problem — it isn't FineWeb-specific — so **acc_norm is the neutral
number**: OpenAI's GPT-2 124M measures **29.4%** (29.38% precisely) with this repo's own eval
code, and that's the bar Rung 0 targets regardless of which val-loss bar it also hits
(`ARCHITECTURE.md` §10, `DEVIATIONS.md` E10).

## 4. Evaluation

```bash
scripts/eval.sh <model-dir> [standard|full|quick|val_loss|hellaswag|core]
```

`<model-dir>` is an r52 checkpoint (`runs/<run>/ckpt/best`) or an exported mlx-lm directory
(`models/<name>-mlx`). **standard** (default) = val_loss + HellaSwag; **full** adds CORE (22
tasks); **quick** runs all three at `--max-tokens 262144 --limit 50` (a 46-second smoke test,
measured — `DEVIATIONS.md` E9). Runs under `nice -n 10` with a 3 GiB MLX memory guideline
(`R52_MEM_GIB` — an eval typically runs *next to* a pretraining job); other knobs: `R52_RUN`,
`R52_MODEL_ID`, `R52_BLOCK_SIZE`, `R52_PYTHON`, `R52_RESULTS_DIR`. Each eval also runs standalone,
e.g. `python -m r52.eval.hellaswag --model models/gpt2-mlx --limit 50`.

**The GPT-2 124M reference**, produced with this repo's own eval code against the real
`openai-community/gpt2` weights (`python -m r52.export --gpt2-reference models/gpt2-mlx` — HF
GPT-2 can't be *imported into* our architecture: learned position embeddings, LayerNorm and GELU
vs. our RoPE, RMSNorm and ReLU²; see Step 5):

| eval | value | wall-clock |
|---|---|---|
| val loss (`r52.eval.val_loss --model models/gpt2-mlx --block-size 1024 --max-tokens 10485760`) | 3.4471 nats/token (bpb 1.1164) | 1,730 s |
| HellaSwag (`r52.eval.hellaswag --model models/gpt2-mlx`) | acc_norm 29.38% (acc 28.53%) | 685 s |
| CORE (`r52.eval.core --model models/gpt2-mlx`) | **not yet measured** — `docs/PLAN.md` §4 calls it "pending"; a full 22-task run was still in progress on this machine as this doc was written | overnight; `--limit 50` ≈ 46 s |

(measured rows: 2026-09-13, `docs/RESULTS.md`.) CORE's port was instead validated against
nanochat's bundled reference CSVs — GPT-2 124M **CORE 0.113891**, GPT-2 XL **0.256525** (the
"time to GPT-2" bar in `dev/LEADERBOARD.md`) — task by task (`DEVIATIONS.md` E4).

**Reproduction metadata every result carries** (`r52/eval/report.py`; real values, from
`results/gpt2-124m-reference/val_loss.json`): `model`, `benchmark`, `value`, `unit`, `conditions`
(`tokenizer`, `block_size`, `n_examples`, `split`, `command`), `commit` (`5c209e2+dirty`), `date`,
`wall_clock_s` (`1729.726`), `machine` (`"Apple M4, 16 GiB, macOS 26.5.2, MLX 0.32.2"`).
`--report` writes this to `results/<run>/<eval>.json` and appends a row to
[`docs/RESULTS.md`](RESULTS.md) — nothing is reported without its command, sample count,
tokenizer, sequence length, commit and wall-clock attached.

## 5. Export + chat

```bash
python -m r52.export runs/nano-a/ckpt/best models/nano-a-mlx
```

Writes an mlx-lm-loadable directory (`config.json`, `model.safetensors`, tokenizer files). Two
export targets, chosen automatically: `use_value_embeds=false`+`use_unet_skips=false` exports as
`model_type: "nanochat"` and loads in **stock mlx-lm, no plugin**; the shipped configs use both
features (a real token-efficiency win), so they export `model_type: "r52gpt"` plus a copied
`r52gpt.py`, which `mlx_lm.utils.load_model` imports directly — `generate`/`.server`/`.lora`/
`.evaluate` all work with nothing installed. Measured round-trip: max |logit diff| = 0.0 against
mlx-lm's own forward pass, both paths (`DEVIATIONS.md` #10). Add `--verify` to check this
yourself; `--dtype float32` for full precision instead of the bf16 default.

**Chat.** `mlx-lm` 0.31.3 has **no `--apply-chat-template` flag** — the chat template applies
**by default**; the flag that exists is the opposite one, `--ignore-chat-template` (verified:
`python -m mlx_lm generate --help`; confirmed by `DEVIATIONS.md` P1's own measurement: 10 prompt
tokens with the template vs. 6 raw BPE tokens with `--ignore-chat-template`).

```bash
python -m mlx_lm generate --model models/nano-a-mlx --prompt "What is 2+2?"   # template applied
scripts/chat.sh models/nano-sft-mlx                    # interactive REPL (mlx_lm.chat)
scripts/chat.sh models/nano-sft-mlx "What is 2+2?"      # one turn (mlx_lm.generate)
```

`scripts/chat.sh` passes no template flag (so the default applies) and warns if the model
directory has no `chat_template` in `tokenizer_config.json`.

**Serve.** An OpenAI-compatible endpoint over the same template:

```bash
scripts/serve.sh models/nano-sft-mlx                    # http://127.0.0.1:8080
curl http://127.0.0.1:8080/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"What is 2+2?"}],"max_tokens":64}'
```

## 6. Midtraining

A short, LR-decaying stage between pretraining and SFT that deliberately seeds
instruction-following and math data into the *base* model before SFT has to teach the chat format
from nothing (`docs/PLAN.md` §3.1 item 4). The nano-scale mix
(`configs/posttrain/midtrain_nano.yaml`): **60% FineWeb** (the pretraining distribution, so
midtraining doesn't cause forgetting) · **30% smol-smoltalk** (via `r52/chat_template.py` — the
same token layout SFT and inference use) · **10% FineMath 4+**.

```bash
python scripts/prepare_midtrain_data.py configs/posttrain/midtrain_nano.yaml
python -m r52.posttrain.midtrain configs/posttrain/midtrain_nano_train.yaml \
    --init-from runs/nano-a/ckpt/best --run-name m1
```

Midtraining is ordinary next-token pretraining on a different corpus, so it runs the stock
`Trainer` unchanged — but with `--init-from` (weights only), **not** `--resume`: resuming would
also restore the pretraining step counter, optimizer moments and data cursor, putting a 763-step
midtrain schedule at its *final* LR on step one and pointing the cursor at the FineWeb shards
instead of the midtrain mix (`DEVIATIONS.md` P2). `--resume` still means what it normally does —
continuing an *interrupted midtrain run*.

**Time.** Midtraining reuses `nano_30m`'s exact architecture and loop, so Step 2's isolated
30M/1024/mixed throughput (6,479 tok/s) applies directly: 50M tokens ≈ 7,720 s ≈ **2.1 h,
derived, not yet measured at this scale**. What *is* measured is the pipeline: a 20-step, 2-layer
smoke test (`mt-tiny`, `midtrain_tiny_train.yaml`) completed 81,920 tokens in **16.1 s** end to
end on 2026-09-13, proving the mix, the `--init-from` path and the checkpoint format all work
before you spend the 2 hours for real.

## 7. SFT

```bash
python scripts/prepare_sft_data.py --n 200000 --max-seq 1024
scripts/sft.sh configs/posttrain/sft_nano.yaml sft_nano
```

Default dataset: **`HuggingFaceTB/smol-smoltalk`** (Apache-2.0, 460,341 train rows). Its teacher
is **Llama-3.1-405B-Instruct** (Magpie-generated), plus public sets (OpenHermes-2.5, MetaMathQA,
NuminaMath-CoT, self-oss-instruct-sc2, SystemChat-2.0, LongAlign) — open-weight lineage, **no
Claude output anywhere in it**, checked directly against the HF API and dataset cards on
2026-09-13 (`DEVIATIONS.md` P9). `--dataset` refuses any repo id matching `PLAN.md` §5's
exclusion list outright (`SWE-smith-trajectories`, `OpenHands-*-Trajectories`,
`R2EGym-SFT-Trajectories`, the Tulu-3 preference mixture — which SmolTalk2's *Preference* split
also inherits, `Anthropic/hh-rlhf`, `*claude-code-traces*`, `dolphin-distill`); `smol-smoltalk` is
a different split of a related dataset family and isn't on it.

**Assistant-only loss masking** needed no model or trainer change: `GPT.loss` already treats
negative targets as `ignore_index`, so `SFTBatcher` emits `-1` for every system/user/pad position
(`DEVIATIONS.md` P4). It shows directly in the log — events carry both `tokens` (all padded
tokens) and `sup_tokens` (supervised assistant tokens); reported `loss` is mean CE over
`sup_tokens`. Real example, this machine's 20-step tiny smoke test (`sft-tiny`, 2026-09-13):
`tokens=20352, sup_tokens=7520` — a genuine ~37% assistant-token ratio, finished in **6.8 s**.
Full `sft_nano` wall-clock at 200k conversations / 2 epochs: **not yet measured**.

## 8. RL

Run the pass@k probe **before** any RL — a decision input, not a formality:

```bash
python -m r52.posttrain.passk models/nano-sft-mlx --task chain_sum -n 64 -k 16
```

`docs/PLAN.md` §3.1 item 5: high pass@k means RL only sharpens what the model already samples;
near-zero pass@k means **every group has zero reward variance**, DAPO's dynamic sampling drops
all of them, and RL does literally nothing — pick easier tasks first (`rl_nano.yaml`'s own header
comment). That's why it ships an "easiest-first curriculum" (`chain_sum, leg_counting,
count_bits` — short-string answers a nano-scale model has a real chance of sampling).

```bash
scripts/rl.sh configs/posttrain/rl_nano.yaml r1 --model models/nano-sft-mlx
```

DAPO-style GRPO (arXiv 2503.14476): no KL term (`beta=0`), clip-higher (`epsilon_high 0.28 >
epsilon_low 0.2`), token-level loss aggregation, no std-normalization, dynamic sampling. Two
backends exist (`--backend native|mlx-lm-lora`); **native is the default and what the shipped
configs use**, because `mlx-lm-lora`'s GRPO measurably computes the policy gradient on
`logP(completion)` with no prompt in context — not `logP(completion|prompt)`, the entire point of
RLVR (`DEVIATIONS.md` P7). RL writes `runs/<run>/final/` **already mlx-lm-ready** — no separate
`r52.export` step before chatting with it or uploading it.

**Honest expectation.** Small models mostly fail verifiable tasks; the point of this stage is
that the plumbing — sampling, verifier scoring, the GRPO update, checkpointing — works end to
end. Measured on this machine's 5-step, 2-layer smoke test (`rl-tiny`, 2026-09-13): `solve_rate`
stayed **0.0** for all 5 steps and `reward_mean` ended at **0.00655** — the loop ran clean (64
sequences/step, ~24 generated tokens each, checkpoint saved to `runs/rl-tiny/final`) but a model
this small solved essentially nothing. Expected, not a bug — exactly the pass@k-near-zero case
the probe exists to catch first. RL wall-clock at nano scale is **not yet measured**: RL is
dominated by autoregressive sampling, which doesn't parallelize across the sequence the way a
forward/backward pass does, and this repo deliberately has not published a decode-throughput
number (`r52.bench` measures *training* only — Step 2).

## 9. Publishing

```bash
python -m r52.export runs/sft_nano/ckpt/best models/nano-sft-mlx
python scripts/hf_upload.py --export-dir models/nano-sft-mlx --run sft_nano \
    --repo-id SammyTourani/road-to-52-nano-sft --dry-run
```

`--dry-run` writes `results/sft_nano/MODEL_CARD.md`, prints the exact file list that would
upload, and makes **no network call**. Every card field is pulled from files already in the repo
— `runs/<run>/log.jsonl`, `results/<run>/*.json` (every eval that run has),
`results/gpt2-124m-reference/*.json` (the baseline), the matching `results/ladder.json` rung if
any — and a field with no data source renders as "—", **never a fabricated number**
(`scripts/hf_upload.py`'s own docstring). Drop `--dry-run` to actually push (needs `HF_TOKEN` or
`hf auth login`); `runs/*/ckpt/` — checkpoints and optimizer state — is never read or uploaded,
only the exported directory. For an RL output, point `--export-dir` straight at `runs/r1/final`
(Step 8) — already mlx-lm-ready.

## 10. Going bigger

Rung 0 costs $0 and runs on this Mac. Everything past it is real money on rented GPUs, and
spending it is explicitly not something any script does on its own — `PLAN.md` §7 rule 5: "do
not spend money without the owner's go"; `results/ladder.md`: "spending money on them is the
owner's decision"; `rungs/README.md`: "do NOT run any training, do NOT use the local GPU...".

The ladder (`python -m r52.ladder`, [`results/ladder.md`](../results/ladder.md)), calibrated to
reproduce DeepSeek-V3's and Llama 3.1's disclosed GPU-hours to within ~1 point of MFU:

| Rung | Beats | Tokens | Hardware | Wall-clock | $ spot | Status |
|---|---|---|---|---|---|---|
| 0 (this doc) | GPT-2 small | 0.75B | this Mac | ~3 d | **$0** | running |
| 1 | GPT-2 XL (CORE 0.2565) | 20B | 8×H100 | 12.0 h | $91 | planned |
| 2 | (no named target) | 60B | 8×H100 | 4.5 d | $815 | planned |
| 2 (MoE) | 30B-A3B capacity, 3B-dense bill | 60B | 8×H100 | 4.5 d | $815 | planned |
| 2b | (most compute $2,000 buys) | 150B | 8×H100 | 11.3 d | $2,037 | planned |
| 3 | Llama 3.1 8B's size at 1/94th its tokens | 160B | 64×H100 | 4.0 d | $5,793 | needs a grant |

Cheapest verified floor: **Prime Intellect spot, $0.94/H100-hour** ($7.52/8×H100-node-hour) —
`rungs/README.md`'s provider table, retrieved 2026-09-13; hyperscalers run 3–5× that for
identical silicon.

```bash
python -m r52.ladder --custom "N=3e9,D=60e9,hw=h100,gpus=8"    # cost an arbitrary configuration
```

`rungs/` holds **ready-to-run, unexecuted** packages for when the owner opts in:
`rung1_nanochat/` (pinned-commit clone-and-run wrapper for `karpathy/nanochat`'s `speedrun.sh`),
`rung2_3b/` (torchtitan job config + data plan + launch script + eval plan for a from-scratch
3B/30B-A3B-MoE model), `grants/` (draft pitches for free compute: Prime Intellect Fast Compute
Grants, TPU Research Cloud, a McMaster CCDB sponsor email). Every script defaults to printing what
it would do and stopping.

## Troubleshooting

Drawn from `docs/DEVIATIONS.md` — real failures hit while building this repo, and the fixes
already shipped in the code.

**Memory limit is a guideline, not a hard cap.** `mx.set_memory_limit()` only raises once RAM
*and swap* are both exhausted; go over it and MLX degrades into swap thrashing instead of
erroring — measured: 5.5 GB of swap and a 15+ minute stall on one oversized bench row (`#7`). A
stalled-looking run is more likely thrashing than hung — check swap (`vm_stat`) first.
`r52.bench` itself handles this by walking a micro-batch ladder downward and recording anything
that doesn't fit as `skipped`.

**GPU contention can more than halve throughput.** Every eval number in `DEVIATIONS.md`'s eval
section is explicitly measured *while the Rung-0 pretraining run held ~5.1 GiB of the GPU* — "the
GPU is already saturated by the pretraining run" (E5, E9). Observed directly here, 2026-09-13:
`runs/gpt2-124m-mac/log.jsonl` steps 70–110 show **~1,300–1,360 tok/s** (measured MFU ~31–32%)
with a concurrent `r52.eval.core` process sharing the GPU — well under half of Step 2's isolated
**3,034 tok/s** (61% MFU) for the identical config. A much-worse-than-expected `eta_h` means
check `ps aux | grep r52` for a second MLX process before assuming something's broken.

**Logits memory scales with `positions × vocab_size`, not with the model.** An fp32 logits tensor
costs `rows * tokens_per_row * 50,304 * 4` bytes — at training shapes (B=4, T=1024) that's 824
MB, before its gradient. Training loss therefore runs in the model's **bf16** compute dtype by
default (measured bias: -5.2e-4 nats, negligible); only *validation* forces `fp32_logits=True`
(`#6`). Eval processes bound the same cost with `--max-positions` (default 4,096): raising it to
16,384 on `hellaswag --limit 30` pushed peak memory to **4.83 GiB**, over the eval suite's 3 GiB
guideline, for zero wall-clock benefit once the GPU is saturated (`E5`). Pass `--max-positions`
down rather than raising the memory limit.

**PyArrow deadlocks at interpreter exit after a streaming download finishes.** Both data-prep
scripts write their files correctly, then hang forever at 0% CPU inside
`arrow::internal::ThreadPool::Shutdown` when a `datasets` streaming iterator is abandoned mid-file
— first seen as a 17-minute "download" that was really a teardown deadlock (`P6`, pyarrow
25.0.1 + datasets 5.0.1). Fix already shipped: `prepare_sft_data.py`/`prepare_midtrain_data.py`
flush and call `os._exit(rc)` instead of a normal interpreter shutdown (same run afterward: 8.8
s). Copy that pattern in any new script using `datasets` streaming mode.

**The eight chat tokens (ids 50257–50264) aren't valid `tiktoken` input.** `enc.decode([50257])`
raises `KeyError: Invalid token for decoding: 50257` — silently broke bits-per-byte on every
midtrained run and emptied sampler output before the fix (`P3`). `GPT2Tokenizer.decode_bytes` now
splits the id stream and spells out-of-vocab ids literally (`<|50303|>`); `encode_ordinary`
guarantees a user message can't *produce* a special id, so typing `<|assistant_start|>` in a chat
can't forge a turn boundary. Call `decode_bytes`/`encode_ordinary`, not raw `tiktoken`, if you
touch the tokenizer directly.

## What this is not

- **Not frontier.** Rung 0 is ~5.6e17 FLOPs; the frontier (Claude Fable 5.1-class) is estimated
  at 1e26–1e27 FLOPs — nine to ten orders of magnitude more (`PLAN.md` §0, §1).
- **Not safety-trained.** Base pretraining, midtraining and an RLVR probe on toy verifiable tasks
  — no RLHF stack, no red-teaming, no content-safety layer.
- **English only.** FineWeb, smol-smoltalk and FineMath are English-dominant; no multilingual
  claim is made or checked.
- **No Claude data, anywhere, ever.** Hard rule, not a preference — Anthropic's terms forbid
  training competing models on Claude outputs. Claude models write the *code and docs* here; the
  model learns only from open datasets and open-weight teachers (`PLAN.md` §0, §7;
  `ARCHITECTURE.md` §1).
- **Classic, small-model evals, not the 2026 frontier set.** HellaSwag, CORE and FineWeb val loss
  are what a 30M–124M model can meaningfully move; none of the nine 2026 frontier launches even
  report them (`PLAN.md` §2) — fine for measuring this rung, never a frontier comparison.
