# Deviations from `docs/ARCHITECTURE.md`

Builder: MLX training core (sections 3, 4, 5, 7 plus `r52/{config,optim,data,tokenizer,checkpoint}.py`,
`configs/`, `scripts/`, `tests/`). Date: 2026-09-13. Machine: Mac Mini M4, 16 GB, macOS 26.5.2,
MLX 0.32.2, mlx-lm 0.31.3.

Everything here is a place where the spec was wrong, impossible, or expensive enough that the
sensible thing differed. Nothing here is a silent divergence.

---

## 1. `mx.eval` runs once per **micro**-step, not once per optimizer step

**Spec §4:** "`mx.eval` once per optimizer step."

MLX builds a lazy graph. With `grad_accum = 32` (the 124M config), deferring `mx.eval` to the end
of the optimizer step keeps **32 complete forward/backward graphs — and therefore 32 copies of the
activations — alive at once**. That is an immediate OOM, not a micro-optimisation.

`Trainer.train_step` therefore calls `mx.eval(acc, loss_sum)` after each accumulation micro-step and
once more after the optimizer update. When `grad_accum == 1` this degenerates to exactly one
`mx.eval` per optimizer step, i.e. the spec's intent.

## 2. QK-norm is applied **after** RoPE, not before

**Spec §3:** "RMSNorm on q and k per head before RoPE."

We do `rope` then `rms_norm`, which is what karpathy/nanochat and `mlx_lm/models/nanochat.py` do.
This is not a behavioural change: RoPE is an orthogonal transform, so it preserves the per-head L2
norm, and a parameter-free RMSNorm is therefore exactly commutative with it
(`rms_norm(rope(x)) == rope(rms_norm(x))`; measured residual 7.6e-6 in fp32, i.e. rounding).
Matching nanochat's order means an exported checkpoint is *numerically identical* under mlx-lm
rather than merely close.

## 3. The RNG is re-seeded on resume instead of restored

**Spec §4:** checkpoint the "RNG".

MLX's global random state is an opaque `mlx.core.random._RandomState` with no public key or state
accessor (`dir()` on it is empty), so it cannot be serialised. Checkpoints store the seed and the
step; resume calls `mx.random.seed(seed + step)`.

This is exact rather than approximate: nothing in the training step is stochastic (no dropout, no
sampling, deterministic data cursor), so the RNG affects initialisation only.
`tests/test_train.py::test_checkpoint_round_trip_identical_loss` and
`::test_resume_reproduces_the_same_token_stream` assert bit-identical resume.

## 4. Value embeddings default to one **shared** table (`value_embed_share`)

**Spec §3:** "per-layer (or shared across alternating layers) token-indexed value residual".

modded-nanogpt allocates `n_value_embeds` separate `vocab_size x n_embd` tables. At 124M with fp32
master parameters that is 115.9M extra parameters = 464 MB of weights + 464 MB of gradients +
927 MB of AdamW moments = **1.4 GiB**, more than a third of the memory budget for a 6 GiB cap.

`ModelConfig.value_embed_share=True` (used by both shipped configs) keeps the technique — value
embeddings still feed layers 0,1,2 and 9,10,11 — from a single table, at a quarter of the cost.
`value_embed_share=False` reproduces modded-nanogpt exactly and is covered by the tests.

## 5. Plain **Muon**, not the record's NorMuon; and the `wte` learning rate

`mlx.optimizers` ships plain Muon. modded-nanogpt's *current* record (`train_gpt.py`, fetched
2026-09-13) uses **NorMuon** — Muon plus an Adafactor-style low-rank variance estimator
(arXiv 2510.05491) — whose `lr = 0.023` is not on the same scale as plain Muon's. We therefore use
the classic modded-nanogpt plain-Muon setting (`lr 0.05, momentum 0.95, nesterov, ns_steps 5`) that
held the record through the 2024-11..2025 series, and copy the AdamW groups verbatim from the
current record. Every value and its derivation is written out in `configs/gpt2_124m_mac.yaml`.

Same story for `wte`: the record keeps `wte` **tied** to `lm_head` for the first two thirds of
training and so runs it at the `lm_head` LR of 0.008. Our embeddings are untied from step 0 (spec
§3), which is the classic configuration — and that used `embed lr = 0.6`. We use 0.6, wd 0.

There are five optimizer groups, not four: `muon`, `embed` (`wte`), `vembed` (`value_embeds.*`),
`head` (`lm_head`), `scalar`. The record gives `wte` and the value embeddings different LRs, betas
and weight decays, so folding them together would misapply a 75x LR multiplier.

**Muon never sees a fused projection.** q/k/v (and SwiGLU's gate/up) are separate `Linear` layers, so
Newton-Schulz orthogonalises each projection matrix on its own;
`tests/test_muon_targets.py` asserts the exact set and shape of every Muon-routed tensor. There is
also no batch-size warm-up: `tokens_per_step` is constant from step 0, as Muon wants.

## 6. Training loss uses a bf16 log-sum-exp; validation uses fp32

`GPT.loss(..., fp32_logits=None)` defaults to computing the cross-entropy in the model's *compute*
dtype. At `B=4, T=1024, V=50304` an fp32 logits tensor is 824 MB (plus its gradient); bf16 halves
that. This is the same choice `mlx_lm/tuner/trainer.py` makes. Measured bias on random logits:
**-5.2e-4 nats** (std 0.026 per token, so it averages out over a validation set).

Validation and `--target-val-loss` always pass `fp32_logits=True`, so every *reported* number and
the 3.28 target comparison are computed in fp32.

## 7. `mx.set_memory_limit` is a guideline, not a cap

MLX raises only when RAM **and swap** are both exhausted; exceeding the limit silently degrades into
swap thrashing (observed: 5.5 GB of swap and a >15 min stall on one oversized bench row). Two
consequences:

* `r52.bench` walks a micro-batch ladder downward and **rejects** any candidate whose measured
  warm-up peak exceeds `--mem-limit-gib`, then records it as `skipped` with the reason.
* Each bench row runs in its own subprocess, so a bad row cannot poison the sweep's peak-memory
  readings or kill it.

## 8. Validation during training is capped at `val_max_batches`

The spec's validation set is the first 10,485,760 tokens of the val shard. At 124M that is roughly
**10 minutes of forward passes per validation**, which is not something you run every 250 steps on
one GPU. `val_max_batches` (default 64 batches = 262,144 tokens, standard error ~0.004 nats) caps it
during training; every log line records `val_tokens` so the number is never quoted without its
sample size. Set `val_max_batches: 0` (or `-o val_max_batches=0`) for the full, spec-exact split —
that is what the headline Rung-0 number must use.

## 9. `export.py` cannot import HF GPT-2 (spec §6)

**Spec §6:** "`export.py` must also support import of HF gpt2 for this apples-to-apples check."

This is not possible. HF GPT-2 is a different architecture, not a different weight layout: learned
absolute position embeddings, LayerNorm **with bias**, GELU, `Conv1D` (transposed) weights, tied
embeddings, no QK-norm, no logit softcap. Ours has RoPE, parameter-free RMSNorm, ReLU^2, untied
zero-init `lm_head`, QK-norm and softcap 15. There is no mapping of GPT-2's weights onto these
modules that preserves the function.

Provided instead: `python -m r52.export --gpt2-reference models/gpt2-mlx`, which materialises
`openai-community/gpt2` as its own mlx-lm directory via `mlx_lm.convert`. The eval harness should
run *the same eval code* against that directory and against our export; that is the apples-to-apples
comparison the spec actually wants.

## 10. Export target: nanochat when possible, a shipped plugin file otherwise

Our core weight names deliberately mirror `mlx_lm/models/nanochat.py`
(`transformer.wte.weight`, `transformer.h.{i}.attn.c_q.weight`, ...). So:

* `use_value_embeds=false, use_unet_skips=false` -> `model_type: "nanochat"`, loads in **stock
  mlx-lm with nothing installed**.
* Otherwise (the default) the value-embedding tables, per-layer `lambdas` and `skip_weights` have no
  home in that class, so the export writes `model_type: "r52gpt"` **and copies
  `r52/mlx_plugin/r52gpt.py` into the model directory**, referenced as `"model_file": "r52gpt.py"`.
  `mlx_lm.utils.load_model` imports that path directly, so `mlx_lm.generate`, `.server`, `.lora` and
  `.evaluate` all work with no plugin installation.

Measured round-trip (8 prompts, bf16 export): **max |logit difference| = 0.0** for both paths and for
the GQA and SwiGLU variants. KV-cache incremental decoding is also verified.

Note: `r52/mlx_plugin/r52gpt.py` deliberately does **not** use `from __future__ import annotations`.
A module loaded through `importlib.util.spec_from_file_location` is not registered in `sys.modules`,
and `@dataclass` on string annotations then fails with
`AttributeError: 'NoneType' object has no attribute '__dict__'`.

## 11. Parameter counts: three conventions, all reported

With a 50,304-token vocabulary, untied embeddings and a value-embedding table, three
`vocab x n_embd` matrices dominate the *total* parameter count at GPT-2-small width. The "124M"
config has **200,835,090 total**, **84,934,674 non-embedding**, and **123,568,146** parameters that
actually run a matmul (non-embedding + `lm_head`) — the last being exactly the classic GPT-2-small
number, which is the `N` used in `flops_per_token = 6N + 12 * n_layer * n_embd * T`. All three are in
`ModelConfig`'s docstring, in every bench row, and in the run log header.

The bench's "60M" and "350M" presets are ours (the spec lists the labels, not the shapes): 12L/640d
(59.0M non-embedding) and 24L/1024d (302.0M non-embedding, GPT-2-medium's shape).

## 12. `grad_clip` is implemented but **off** in the shipped configs

`mlx.optimizers.clip_grad_norm` materialises a full extra copy of the gradient tree — 0.8 GiB at
124M. Muon's Newton-Schulz step already normalises every hidden-matrix update, and modded-nanogpt
uses no gradient clipping at all. The feature works and is tested (`tests/test_optim.py`); the two
training configs set `grad_clip: 0.0` and still report the pre-clip global norm every log line.

Side effect worth knowing when reading the benchmark: `r52.bench` uses `TrainConfig()` defaults,
i.e. clipping **on**, so its peak-memory column is the conservative end. The same architecture run
from `configs/nano_30m.yaml` (clip off) peaks at **4.00 GiB** where the benchmark row says 4.66.

## 13. `mx.compile` measured, kept ON

Spec §4 says to measure `mx.compile` both ways and leave a flag. Measured on this M4, mixed
precision, seq 1024, same micro-batch, full training steps:

| config | compile off | compile on | speed-up |
|---|---|---|---|
| 30M, mb 4, accum 16 | 5,719 tok/s | 6,519 tok/s | **+14.0%** |
| 124M, mb 2, accum 32 | 2,824 tok/s | 3,024 tok/s | **+7.1%** |
| 124M, forward/backward only, mb 4 | 3,083 tok/s | 3,331 tok/s | **+8.0%** |

It also *reduces* peak memory (4.81 -> 4.66 GiB at 30M) by fusing the softcap/`tanh`
element-wise chain instead of materialising each temporary. So compile stays on by default;
`--no-compile` (train) and `--no-compile` (bench) turn it off, and the setting is recorded in
every log header and bench row.

## 14. Upstream MLX bug: `MultiOptimizer` cannot hold an empty group

Found while measuring a `use_value_embeds: false` variant. `MultiOptimizer._split_dictionary`
hands an empty slice to any sub-optimizer whose filter matches nothing, and
`mlx.utils.tree_unflatten([])` returns `[]` — a **list**. `Optimizer.init` then takes the
`isinstance(params, (list, tuple))` branch and iterates `range(len(state))` over its own
`{"step", "learning_rate"}` dict while indexing `params[i]`, raising
`IndexError: list index out of range` on the first optimizer step.

`build_optimizer(cfg, params)` therefore takes the parameter tree and instantiates only the
groups that actually receive parameters (`GROUPS` order keeps `scalar` last, so the
`MultiOptimizer` fallback stays the catch-all). The live group list is recorded in the run
log's `start` record as `optimizer_groups`, and
`tests/test_optim.py::test_empty_groups_are_dropped_and_a_step_runs` covers every combination
of `use_value_embeds` / `use_unet_skips`.

*This is worth reporting upstream to ml-explore/mlx.*

## 15. `micro_batch` in `gpt2_124m_mac.yaml` and the memory ceiling

Measured on this machine 2026-09-13, two optimizer steps of the real config each:

| variant | tok/s | TFLOPS | MFU (4.26) | peak GiB |
|---|---|---|---|---|
| `micro_batch: 4` | 3,203 | 2.737 | 64.3% | **6.63 — breaches the 6 GiB rule** |
| `micro_batch: 2` **(shipped)** | 3,076 | 2.629 | 61.7% | 5.11 |
| `micro_batch: 2` + gradient checkpointing | 2,637 | 2.254 | 52.9% | 4.88 |
| `micro_batch: 4` + gradient checkpointing | 2,740 | 2.342 | 55.0% | 4.88 |

The spec's stretch target of **<= 4 GiB peak at 124M is not reachable with this architecture**, and
the reason is arithmetic, not tuning: the model carries 200.8M parameters (untied embeddings *and* a
value-embedding table, both mandated by §3), and mixed precision keeps fp32 masters. Persistent
state alone is fp32 params 0.80 + fp32 grads 0.80 + AdamW moments 0.93 + Muon momentum 0.34 =
**2.87 GiB**, to which MLX's functional optimizer adds a whole fresh parameter tree (+0.80 GiB) on
every update. Activations are the small term — which is exactly why gradient checkpointing buys only
0.23 GiB for a 14% throughput loss and is left off.

Routes to <= 4 GiB, if the planner wants one: `use_value_embeds: false` (-0.6 GiB, and a real
token-efficiency loss), bf16 AdamW moments for the embedding groups (-0.46 GiB, gives up the fp32
master-state property), or a smaller vocabulary. The shipped config instead respects the hard
6 GiB rule at 5.11 GiB and near-peak throughput.

---

# Deviations — eval builder

Builder: evaluation suite (`docs/ARCHITECTURE.md` §6: `r52/eval/`, `scripts/eval.sh`,
`tests/test_eval*.py`, `results/gpt2-124m-reference/`). Date: 2026-09-13. Machine: Mac Mini M4,
16 GB, macOS 26.5.2, MLX 0.32.2, mlx-lm 0.31.3 — **with the Rung-0 124M pretraining run holding
~5.1 GiB of the GPU throughout**, which is why every wall-clock below is a shared-GPU wall-clock.

## E1. llm.c's README carries no HellaSwag number — the docstring does, and it says something different

**Spec §6 / build brief:** "Reference values ... GPT-2 124M acc 0.2955 / acc_norm 0.3117 per
llm.c's README (`gpt2 (124M) hellaswag 29.55%`, `acc_norm 31.17%`)."

Fetched `karpathy/llm.c` `README.md` on 2026-09-13 (`gh api repos/karpathy/llm.c/contents/README.md`):
it mentions `gpt2` 19 times and **`hellaswag` zero times**. The primary source is the module
docstring of `dev/data/hellaswag.py`, which says, verbatim:

```
gpt2 (124M)
- eleuther harness reports acc 28.92%, acc_norm 31.14% (multiple choice style)
- this script: 10042 acc: 0.2859 acc_norm: 0.2955 (completion style)
```

So the correct reading is: **0.2859 / 0.2955** are llm.c's own completion-style acc / acc_norm —
the protocol `r52/eval/hellaswag.py` implements — and **0.2892 / 0.3114** are
lm-evaluation-harness's multiple-choice-style numbers. The brief's "0.2955 acc" and "0.3117
acc_norm" conflated the two sides. `r52/bar/bar.yaml`'s `gpt2-124m` row (31.1 acc_norm / 29.4 acc,
both `verified: false`) has the same provenance problem; that file is not ours to edit, but the
measured local result now supersedes both cells in the gap table.

## E2. HellaSwag: llm.c's protocol and lm-eval's are *not* the same eval

The brief asks for a protocol "identical to llm.c's `dev/data/hellaswag.py` **and** lm-eval's
`hellaswag`". They differ in two ways and cannot both be satisfied:

| | llm.c (implemented) | lm-eval |
|---|---|---|
| context | `ctx` verbatim | rewritten: activity label prefixed, `[...]` stripped, whitespace collapsed |
| `acc_norm` denominator | number of ending **tokens** | number of ending **characters** |

We implement llm.c's, because it is the one the brief describes in prose (sum of ending log-probs
for `acc`, length-normalised sum for `acc_norm`) and the one Rung 0's acceptance criterion is
quoted against. Since it is free from the same forward passes, the results JSON also carries
**`acc_norm_bytes`** — UTF-8-byte normalisation, i.e. lm-eval's normaliser without lm-eval's text
rewriting — as the closest honest bridge to the 31.14 % figure.

## E3. `python -m r52.export --gpt2-reference` was broken; minimal fix applied

The first thing the eval suite needs is the GPT-2 reference directory, and the command for it
raised `huggingface_hub.errors.IncompleteSnapshotError`. Cause: mlx-lm 0.31.3's
`mlx_lm.utils.save()` re-resolves a *repo id* with `snapshot_download(repo, local_files_only=True)`
and **no allow-patterns**, so it demands every file in the repo — including the `.tflite`, onnx,
TensorFlow and Flax weights that its own downloader (`_download`) deliberately skips.
`openai-community/gpt2` has all of them.

Fix (the one edit this builder made outside its own files): `gpt2_reference()` resolves the
snapshot itself with the same allow-patterns and passes `convert()` a local **path**, which takes
its `src_path.exists()` branch. Two lines plus a comment; no behaviour change for any other repo.
*Worth reporting upstream to ml-explore/mlx-lm.*

## E4. CORE: what the port changes, and why none of it moves the number

`r52/eval/core.py` is a port of nanochat's `nanochat/core_eval.py` + the `evaluate_core` half of
`scripts/base_eval.py` (MIT, attributed in the file header). Five deliberate differences:

1. **No Jinja2.** The three templates are expanded into string concatenation. The whitespace
   semantics (`{%- ... -%}`, the blank line between shots, the `| trim` on LM contexts, the
   `.strip()` on the continuation-free LM prompt) are reproduced exactly and pinned by
   `tests/test_eval_core.py`. One dependency fewer, identical strings.
2. **BOS is `<|endoftext|>` (50256).** nanochat prepends its own `<|bos|>` and pads with it; 50256
   is the GPT-2 tokenizer's equivalent and what GPT-2 itself saw at document boundaries.
3. **Batching across examples.** nanochat forwards one example at a time. We group whole items up
   to a `rows * width` budget. This cannot change a score: attention is causal, padding is on the
   right, and the scored slice `[start-1, end-1)` never reaches the padded tail. Verified
   empirically — `hellaswag --limit 30` returns 0.4667 at `--max-positions` 4096, 8192 and 16384.
4. **Two asserts became counted fallbacks.** nanochat asserts (a) that the continuation-free LM
   prompt is a token prefix of the full one and (b) that cropping to the context window leaves the
   scored span's start non-negative. Both can fail with GPT-2's BPE and a 1,024-token window on
   10-shot prompts. Aborting hour six of a run over one item is the wrong trade, so each falls back
   (longest common prefix; start clamped to 1) and is **counted** in the per-task record as
   `n_lm_prefix_fallbacks` / `n_clamped_spans`, which are reported alongside the score.
5. **Few-shot count clamped when `--limit` is small.** `random.sample(available, 10)` raises when a
   smoke run leaves fewer than ten other items. Full runs are unaffected; only `--limit < shots+1`
   sees fewer shots.

**Validation against the implementation it came from.** The eval bundle ships nanochat's own
per-task CSVs. `openai-community-gpt2.csv` gives GPT-2 124M **CORE 0.113891**, and
`openai-community-gpt2-xl.csv` gives GPT-2 XL **0.256525** — the same number
`dev/LEADERBOARD.md` quotes as the "time to GPT-2" bar. Our port is compared against the 124M CSV
task by task in `docs/RESULTS.md`.

## E5. Eval processes get 3 GiB, and the knob that enforces it is `--max-positions`

`docs/ARCHITECTURE.md` §1.2 gives pretraining 6 GiB; evals run *next to* a pretraining run, so
`r52.eval.lm.configure_runtime` defaults to **3 GiB** (`mx.set_memory_limit`) and 1 GiB of cache.
As §7 of the core builder's notes says, that is a guideline, not a cap — what actually bounds peak
memory is `--max-positions` (`rows * tokens` per forward), because an fp32 logits tensor costs
`positions * 50,304 * 4` bytes. Measured peak on `hellaswag --limit 30`:

| `--max-positions` | peak GiB | wall-clock |
|---|---|---|
| 4,096 **(default)** | 1.72 | 29.8 s |
| 8,192 | 2.66 | 30.5 s |
| 16,384 | **4.83 — breaches the 3 GiB guideline** | 32.5 s |

Bigger forwards buy nothing here (the GPU is already saturated by the pretraining run), so 4,096
is the default. `LM.scores` additionally does the fp32 log-sum-exp in 1,024-position slices, so the
fp32 copy of the logits is never materialised whole.

## E6. Two results files per run, because `gap.py`'s schema and §6's schema disagree

The brief asks for `results/<run>/<eval>.json` with a **structured** `conditions` object plus
`wall_clock_s` and `machine`; `r52/bar/gap.py`'s docstring documents `results/<run>/eval.json` with
a **flat `conditions` string** and `command` at the top level, and the brief says gap.py's field
names win. Both are written:

* `results/<run>/{val_loss,hellaswag,core}.json` — the full record (§6 schema + `metrics`).
* `results/<run>/eval.json` — a list of the same results in gap.py's exact shape, one entry per
  benchmark, rewritten in place when a benchmark is re-run.

`gap.py` is untouched; `tests/test_eval_report.py` asserts it parses our file and that the
`model` / `benchmark` ids resolve against `bar.yaml` (`gpt2-124m` + `fineweb-val-loss` /
`hellaswag` / `dclm-core`), since unknown ids are silently dropped from the gap table.

## E7. bf16 compute, fp32 reductions — and what that costs against fp32 references

Every eval runs the model in its own compute dtype (bf16 for a mixed-precision `GPT` and for every
`dtype=bfloat16` export, including `models/gpt2-mlx`) and does every reduction in fp32, which is
exactly `GPT.loss(..., fp32_logits=True)` — the call `r52.train` uses for the val loss it reports.
That identity is not approximate: `r52.eval.val_loss` on `runs/tiny200/ckpt/best` returns
`6.998237788677216` against the trainer's recorded `6.998237788677216`, **difference 0.0**.

Against llm.c's fp32 HellaSwag reference the bf16 forward costs a few examples out of 10,042:
acc 0.2853 vs 0.2859 (6 examples), acc_norm 0.2938 vs 0.2955 (17 examples). Both references sit
inside our 95 % Wilson intervals ([0.2766, 0.2942] and [0.2849, 0.3028]). Anyone wanting the fp32
number can export the reference with `--dtype float32`; the suite is dtype-agnostic.

## E8. The adapter loads models with `mlx_lm.utils.load_model`, not `mlx_lm.load`

**Brief:** "an mlx-lm model loaded from an exported dir (`mlx_lm.load(path)`)".

`mlx_lm.load` returns `(model, tokenizer)` and *requires* tokenizer files in the directory.
`r52.export.export_checkpoint` fetches them from the Hub, but `export_model` (the test path) and
any offline export do not — and the suite does not want that tokenizer anyway: §6 fixes the
tokenizer for every eval here at **GPT-2 tiktoken**, which is what the FineWeb shards, the GPT-2
reference and our own models are all built on. So `LM.load` calls `mlx_lm.utils.load_model(path)`
(the function `mlx_lm.load` itself calls, and the one that honours `model_file: "r52gpt.py"`) and
always pairs it with `r52.tokenizer.GPT2Tokenizer`. An export with no tokenizer files evaluates
fine; a model needing a *different* tokenizer would need this line changed, and there isn't one at
Rung 0.

## E9. Full CORE costs hours on a shared M4, and the obvious speed-up was not taken

Measured on this machine with the Rung-0 pretraining run holding the GPU: `hellaswag` (10-shot,
4 choices, ~550 tokens per row) runs at about **1 second per example**, so that single task is
~2.8 h of the 22. Raising `--max-positions` does not help (§E5) — the GPU is already saturated.

The real optimisation available is a **shared-prefix KV cache**: in a 10-shot multiple-choice
task the 4-5 prompts differ only in their last few tokens, so ~95 % of every forward is recomputed
work, and caching the prefix would cut MC tasks by roughly 3x. It is not implemented, for a reason
worth recording: `r52.model.GPT.__call__` takes `(idx)` only and has no KV cache, while the
exported mlx-lm model does. Building the optimisation would therefore give the two halves of the
adapter **different code paths**, and the entire point of `LM` is that our checkpoints and the
GPT-2 reference go through the *same* scoring code. Adding a cache to `r52.model.GPT` is the
prerequisite, and that file belongs to the training builder.

Practical consequence: `scripts/eval.sh <model> standard` (val_loss + HellaSwag, ~1 h for a 124M
model on a busy GPU) is the loop to run during a training run; `full` (adds CORE) is an overnight
job. `--limit 50` turns the whole suite into a 46-second smoke test.

## E10. 3.28 is **not** OpenAI's GPT-2 on this split — measured, OpenAI's GPT-2 124M gets 3.447

**Brief:** "Expected ballpark: val loss ≈ 3.3 (the speedrun target 3.28 is defined as GPT-2-small
quality on this split — report what you measure)."

Measured, full split, block 1024:

```
python -m r52.eval.val_loss --model models/gpt2-mlx --block-size 1024 --max-tokens 10485760 --micro-batch 2
val_loss 3.447136 nats/token | val_bpb 1.116407 bits/byte | 10,485,760 tokens | 1729.7 s
```

**3.447, not 3.28** — 0.167 nats worse. That is not a bug in the eval; the same code reproduces
`r52.train`'s own val loss bit-for-bit (§E7). The 3.28 figure means something else. From
`KellerJordan/modded-nanogpt`'s README (fetched 2026-09-13,
`gh api repos/KellerJordan/modded-nanogpt/contents/README.md`), verbatim:

> The target (3.28 validation loss on FineWeb) follows Andrej Karpathy's
> [GPT-2 replication in llm.c, which attains that loss after running for 45 minutes]
> (https://github.com/karpathy/llm.c/discussions/481)

> Note: The 3.28 target was selected to match
> [Andrej Karpathy's GPT-2 (small) reproduction](https://github.com/karpathy/llm.c/discussions/481).

So **3.28 is llm.c's GPT-2-small *reproduction*, trained from scratch on 10B tokens of FineWeb**,
evaluated on FineWeb's own validation set — not OpenAI's released `gpt2` checkpoint, which was
trained on WebText and pays a distribution-shift penalty when read out on FineWeb. Both numbers
are "GPT-2 small"; only one of them was trained on this data.

Consequences for Rung 0 (`docs/ARCHITECTURE.md` §10, `bar.yaml`'s `r52-gpt2-124m-mac` target):

* `fineweb-val-loss <= 3.28` is a **harder** bar than "match OpenAI's GPT-2 124M", which is
  <= 3.447 on this split. The ladder should say which one it claims. Hitting 3.447 is "as good as
  OpenAI's GPT-2 small, read out on FineWeb"; hitting 3.28 is "as good as a 10B-token FineWeb
  reproduction of it", which is the stronger and more expensive claim.
* The same README pins our protocol exactly — "*obtain a probability model of language which
  assigns a probability of at least `math.exp(-3.28 * 10485760)` to the first 10,485,760 tokens of
  the FineWeb valset*" — which is what `r52.eval.val_loss` computes, at block 1024, in fp32.
* The HellaSwag criterion is unaffected: our measured 0.2938 acc_norm / 0.2853 acc for OpenAI's
  GPT-2 124M is the right reference for §10's ">= 29.4 %", and that threshold should be read
  against acc_norm (29.38 %), not acc (28.53 %), or it is unreachable by the model it was copied
  from.

This is a planner decision, not a builder one, so nothing outside `r52/eval/` was changed; the
measured numbers and their commands are in `docs/RESULTS.md` and
`results/gpt2-124m-reference/val_loss.json`.

---

# Deviations — post-training builder

Builder: post-training stack (`docs/PLAN.md` §3.1 items 4-5: `r52/posttrain/`,
`r52/chat_template.py`, `scripts/prepare_{sft,midtrain}_data.py`,
`scripts/{sft,rl,chat,serve}.sh`, `configs/posttrain/`, `tests/test_posttrain*.py`).
Date: 2026-09-13. Machine: Mac Mini M4, 16 GB, MLX 0.32.2, mlx-lm 0.31.3, mlx-lm-lora 3.1.2,
reasoning-gym 0.1.25 — all measured while the Rung-0 124M pretraining run held ~5.1 GiB of
the GPU, so every number below comes from a 2-layer model under a 2 GiB limit.

## P1. `mlx_lm.generate` has no `--apply-chat-template` in 0.31.3 — it is the default

The brief asks for `mlx_lm.generate --prompt ... --apply-chat-template`. That flag does not
exist in mlx-lm 0.31.3: the CLI applies the tokenizer's `chat_template` **by default**, and
the flag that exists is the opposite one, `--ignore-chat-template`. Measured on our export:

```
$ python -m mlx_lm generate --model models/sft-tiny-mlx --prompt "What is 2+2?" ...
Prompt: 10 tokens          # <|bos|><|user_start|>What is 2+2?<|user_end|><|assistant_start|>
$ ... --ignore-chat-template
Prompt: 6 tokens           # the raw BPE of "What is 2+2?"
```

`scripts/chat.sh` therefore passes no template flag and warns if the model directory has no
`chat_template`.

## P2. Midtraining uses `--init-from` (weights only), not `--resume`

The brief says to midtrain "with `--resume` from the base checkpoint and a fresh, short LR
schedule". Those two are mutually exclusive as `r52.train` is written, and the conflict is
not cosmetic: `Trainer._resume` restores `step`, `tokens`, the optimizer moments **and the
data cursor**. Resuming a 5,722-step base run into a 763-step midtrain config evaluates
`wsd_multiplier(5722, 763, ...)` on step one — the final learning rate — and points the
cursor into the FineWeb shards rather than the midtrain ones.

`r52/posttrain/midtrain.py` therefore adds `--init-from`, which loads
`model.safetensors` only and leaves step 0, fresh optimizer state and a fresh cursor.
`--resume` still exists there with its normal meaning (continue an interrupted *midtrain*
run) and the two are rejected together. **`r52/train.py` is unmodified** — midtraining runs
the stock `Trainer`.

## P3. The chat tokens are 8 ids at 50257, and GPT-2's decoder had to learn about them

`r52/chat_template.py` claims `50257..50264` (`<|bos|>`, `<|user_start|>`, `<|user_end|>`,
`<|assistant_start|>`, `<|assistant_end|>`, `<|system_start|>`, `<|system_end|>`, `<|pad|>`)
out of the 47 spare ids in the padded 50,304 vocabulary, so nothing is ever resized. Two
consequences that needed code:

* **`tiktoken` raises on every one of them.** `enc.decode([50257])` is
  `KeyError: Invalid token for decoding: 50257`. That breaks `val_bytes_per_token`, i.e. the
  **bits-per-byte column of every midtrained run**, and it silently emptied sampler output.
  `GPT2Tokenizer.decode_bytes` now splits the id stream and emits the literal spelling for
  out-of-vocab ids (unknown spare ids become `<|50303|>`); `decode` routes through it.
* **`encode_ordinary` was added** so message bodies can never produce a special id — a user
  who types `<|assistant_start|>` gets its BPE spelling, not id 50260, and therefore cannot
  forge a turn boundary.

Both are in `r52/tokenizer.py` (the only edits there). `r52/export.py` gained two: the
`config.json` now carries `bos/eos/pad_token_id` (mlx-lm reads `eos_token_id` from there, so
generation stops on `<|assistant_end|>` as well as `<|endoftext|>`), and `_fetch_tokenizer`
calls `chat_template.install_chat_tokenizer`, which appends the eight tokens to
`tokenizer.json` and writes the Jinja `chat_template` into `tokenizer_config.json`.

`tests/test_posttrain_chat.py::test_jinja_template_matches_render` asserts that the Jinja
template and `render()` produce **identical ids** through the real Hugging Face tokenizer —
that equality is the whole reason `mlx_lm.server` sees the training token layout.

## P4. Loss masking needed no change to `GPT` or to the trainer

`GPT.loss` already treats negative targets as `ignore_index`, so assistant-only supervision
is expressed entirely in the data path: `SFTBatcher` emits `-1` for every system/user/pad
position. No model or `r52/train.py` change was required, and the reported SFT loss is the
mean CE **per assistant token**.

## P5. The midtrain mixture had to be scheduled by tokens, not by documents

The YAML weights are shares of *tokens*. Sampling a source per document with probability
`weight` delivers shares proportional to `weight x mean_document_length`, and the sources
differ by 3x: a FineWeb chunk is a fixed 2,048 tokens, a smol-smoltalk conversation averages
~600. Measured on the 1M-token tiny mix:

| scheduler | fineweb | smoltalk | finemath |
|---|---|---|---|
| target | 0.600 | 0.300 | 0.100 |
| per-document sampling (first attempt) | **0.814** | **0.124** | **0.062** |
| token-deficit sampling (shipped) | 0.599 | 0.300 | 0.101 |

`mix()` now samples in proportion to each source's current token deficit, which is
self-correcting. The realised shares are written to `<out_dir>/mixture.json` and printed.

## P6. `datasets` streaming deadlocks PyArrow at interpreter exit

Both data scripts finished their work, wrote correct files, and then **hung forever at 0 %
CPU** inside `arrow::internal::ThreadPool::Shutdown` (`sample`d on the live process;
pyarrow 25.0.1 + datasets 5.0.1, abandoning a streaming parquet iterator mid-file — the
same bug surfaces as `Exception ignored in <generator object Parquet._generate_tables>:
'NoneType' object has no attribute 'ArrowInvalid'`). First observed as a 17-minute "download"
that was really a teardown deadlock.

`scripts/prepare_sft_data.py` and `scripts/prepare_midtrain_data.py` therefore flush and call
`os._exit(rc)` instead of returning through interpreter shutdown. Same run afterwards: 8.8 s.

## P7. RL ships the **native** GRPO loop; `mlx-lm-lora` is the second backend

`mlx-lm-lora` **can** load our `model_type: r52gpt` plugin export and train GRPO on it —
verified end-to-end, both by calling its CLI directly and through
`r52.posttrain.rl --backend mlx-lm-lora`, which generates the `{prompt, answer}` JSONL plus a
registered reasoning-gym reward function (`r52_reasoning_gym: cov=100.00%`). So the brief's
stated fallback condition ("cannot load our plugin model") never fired.

It is still not what the shipped configs train with, for a reason found by reading its
source and confirmed by its own logs:

* **Its policy log-probabilities are not conditioned on the prompt.** `generate_grpo` stores
  `tokenizer.encode(completion_text)` — completion ids only — and `grpo_loss` feeds exactly
  that array to the model (`inputs = mx.stack(padded_completions)`). Its own
  "Generation Stats / Avg tokens: 16.0" with `--max-completion-length 16` confirms the
  arrays hold no prompt. The gradient is therefore on `log P(completion)`, not
  `log P(completion | prompt)` — which is the entire content of RLVR.
* **DAPO's no-std-norm cannot be expressed**: `calculate_rewards_and_advantages` always
  divides by `std_reward + 1e-4`.
* **No dynamic sampling**: zero-variance groups still enter the loss.
* Completions round-trip through text and are re-encoded, which is lossy for BPE.

`r52/posttrain/rl.py`'s native loop scores with the same verifier but computes
`token_logprobs` over `prompt + completion` with the mask on completion positions only, and
implements every DAPO knob (`beta=0`, `epsilon_high > epsilon_low`, token-level
normalisation over the **batch's** total completion tokens, `std_normalize=False`, dynamic
sampling). It trains the plugin model directly and writes a ready-to-run mlx-lm directory.

Two honest notes on it:

* **RL runs in fp32.** The export is bf16 and AdamW at 1e-5 is below bf16's resolution, so
  `GRPOTrainer` upcasts the loaded parameters. Weights are saved back as bf16.
* **With `inner_epochs=1` the clip is inactive by construction.** On-policy, the policy has
  not moved since sampling, so the ratio is exactly 1, `clip_frac` is 0 and the surrogate's
  *value* is ~0 (group-centred advantages cancel) even though the gradient is not — which is
  why the log reports `|adv|` next to `loss`. `inner_epochs > 1` reuses a batch and is when
  clip-higher starts to bite.

## P8. `mlx-lm-lora` refuses eval splits smaller than `--batch-size`

`iterate_grpo_batches` raises `ValueError: Dataset must have at least batch_size=N examples`
and `train_grpo` evaluates *before* the first step, so a proportional 10 % valid split kills
the run immediately. `run_mlx_lm_lora` now holds out `max(prompts_per_step, 2)` fresh items
each for `valid` and `test` from the tail of the pool.

## P9. smol-smoltalk's teacher is Llama-3.1-405B, not Qwen2.5 (and it is clean)

Checked on 2026-09-13 against the HF API, the datasets-server and the dataset cards, because
`docs/ARCHITECTURE.md` §1.1 forbids Claude-derived data:

* `HuggingFaceTB/smol-smoltalk` — `license: apache-2.0`, public, not gated, **460,341** train
  + 24,229 test rows, features `messages` (`role`/`content`) and `source`.
* Teacher lineage from the parent `HuggingFaceTB/smoltalk` card: the core *Smol-Magpie-Ultra*
  split is generated with **Llama-3.1-405B-Instruct**, plus public sets (OpenHermes-2.5,
  MetaMathQA, NuminaMath-CoT, self-oss-instruct-sc2, SystemChat-2.0, LongAlign).
  The build brief said Qwen2.5-generated; that is wrong, though the conclusion is unchanged —
  an open-weight teacher, **no Claude anywhere in the lineage**.
* Nothing on the `docs/PLAN.md` §5 exclusion list matches. The excluded SmolTalk item is
  **SmolTalk2's *Preference* split** (it inherits the Tulu-3 preference mixture); this is
  `smol-smoltalk`'s SFT conversations. `scripts/prepare_sft_data.py` refuses any `--dataset`
  matching the exclusion list outright.

**One item for the planner:** OpenHermes-2.5 carries GPT-4-derived text, and §5 says
GPT/Gemini-distilled sets are "avoided by default". It is a minority component and it is
filterable via the `source` column if the planner wants it gone; nothing was filtered here,
because §5 names `smol-smoltalk` as *the* SFT choice.

## P10. Drive-by: one `ruff` fix outside the post-training tree

`ruff check scripts` was already failing on `scripts/prepare_data.py:45` (`SIM105`) before
this work started. Replaced the `try/except OSError: pass` with `contextlib.suppress(OSError)`
— identical semantics — so that the required `ruff check r52/posttrain r52/chat_template.py
scripts tests` command is clean. Flagging it because that file belongs to the training-core
builder.

## P11. `reasoning-gym` pins `tabulate==0.9.0` (a downgrade the eval builder should know about)

Installing `reasoning-gym` downgraded `tabulate` 0.10.0 -> 0.9.0 (its requirement is an
exact pin, not a floor). Verified harmless for the eval stack: `lm_eval` 0.4.13 declares no
tabulate constraint, imports cleanly, and `lm_eval.utils.make_table` works; the full suite
(`pytest -q`, 259 tests) passes afterwards. Noted because a future `uv pip install` that
tries to raise `tabulate` again will conflict with `reasoning-gym`.
