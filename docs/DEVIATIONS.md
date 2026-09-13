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

---

# Deviations — ablation builder

Builder: the data-ablation lab (`docs/ABLATIONS.md`: `r52/ablate/`, `r52/tokenizer_train.py`,
`scripts/prepare_corpus.py`, `scripts/ablate.sh`, `configs/ablations/`). Date: 2026-09-13.
Machine: Mac Mini M4, 16 GB, macOS 26.5.2, MLX 0.32.2, `tokenizers` 0.23.2, `datasets` 5.0.1,
pyarrow 25.0.1. Every corpus claim below was checked against the live Hugging Face API on
2026-09-13, not against a dataset card.

## A1. nanochat's split pattern means something *different* in Hugging Face `tokenizers`

`research/02` §Stage 3 records nanochat's pre-tokenizer pattern — GPT-4's, with the digit run
narrowed to `\p{N}{1,2}` — and that narrowing is the whole reason a 32K vocabulary does not
waste entries on numbers. nanochat writes it with **possessive quantifiers** (`\p{N}{1,2}+`,
`\p{L}++`, `[\r\n]*+`, `\s++$`) because `rustbpe` compiles it with the Rust `fancy-regex`
crate.

`tokenizers` 0.23.2 accepts the identical string and silently means something else by it.
Measured (`pre_tokenizers.Split(Regex(...), behavior="isolated")` on `"1234"`):

| pattern | pre-tokens of `1234` |
|---|---|
| nanochat's, verbatim (`\p{N}{1,2}+`) | `['1234']` — **the digit split does not happen** |
| possessive markers removed (`\p{N}{1,2}`) | `['12', '34']` — nanochat's intent |

Its engine reads `{1,2}+` as `({1,2})+` rather than as a possessive `{1,2}`. So
`r52.tokenizer_train.SPLIT_PATTERN` drops every possessive marker and is otherwise
character-for-character nanochat's. `tests/test_tokenizer_train.py::
test_digits_split_into_runs_of_at_most_two` is the regression guard. **Anyone porting a
rustbpe/tiktoken pattern into `tokenizers` should assume it changes meaning until tested.**

## A2. Our 32K specials live *inside* the vocabulary, not in a padded gap

The GPT-2 path puts `<|endoftext|>` at 50256 and the eight chat tokens in the spare ids
50257–50303 of a vocabulary padded to 50304 (`r52/chat_template.py`). That trick exists
because GPT-2's BPE is a fixed 50,257 entries we do not control.

We control ours, so the nine specials (`<|endoftext|>` + the same eight, same order) take the
**top nine ids** of the 32,768: `eot = 32759`, `<|bos|> = 32760`, `<|pad|> = 32767`. 32,768 is
already a multiple of 128, so there is no padding gap and `model.vocab_size ==
tokenizer.n_vocab` exactly. A training text too small to fill the vocabulary is padded with
reserved `<|unused_N|>` ids so the special ids are a function of `vocab_size` alone and never
of how much text happened to be available — a checkpoint's ids must not depend on that.

Consequence for the planner: `r52/chat_template.py`'s module-level `BOS`/`EOT`/... constants
are GPT-2-specific. Everything in the ablation lab reads `tokenizer.eot` /
`tokenizer.special_ids` instead. A future chat model trained on our own BPE will need
`render()` to take its ids from the tokenizer object rather than from those constants; that is
a post-training change, deliberately not made here.

## A3. `encode_ordinary` needs a *second* tokenizer object, because `add_special_tokens=False` does not do what it says

In `tokenizers`, an added token's spelling is matched in the input text whatever
`add_special_tokens` is set to. Measured: `tok.encode("a <|endoftext|> b",
add_special_tokens=False).ids` still contains the special id. Since `encode_ordinary`'s
contract (`r52/tokenizer.py`) is "**no** special token ever produced" — the guard that stops a
user turn from forging a turn boundary — `R52Tokenizer` keeps a clone of the tokenizer with
`added_tokens` stripped out of its JSON and encodes ordinary text with that.

`decode_bytes` likewise does **not** go through `tok.decode().encode()`: the ByteLevel decoder
runs `String::from_utf8_lossy`, so a token run that starts or ends mid-character gains
replacement bytes, which would corrupt a bits-per-byte measurement. It uses a per-id byte
table built by inverting the GPT-2 bytes-to-unicode alphabet, which is exact.

## A4. DCLM ships `.jsonl.zst`, not parquet — and needed a new dependency

`docs/ABLATIONS.md` Axis 1 says "`mlfoundations/dclm-baseline-1.0`, first parquet shards".
The repo has no parquet: it is 27,840 `shard_*_processed.jsonl.zst` files (141–240 MB each)
under `global-shard_NN_of_10/local-shard_N_of_10/`. The *slice* the spec names is right, the
format is not. `allenai/dolma3_mix-150B-1025` is `.jsonl.zst` too.

`datasets` cannot stream `.zst` without `zstandard`, which was not installed. Added to
`pyproject.toml` (BSD-3-Clause), along with an explicit `tokenizers` pin — it was only ever a
transitive dependency of mlx-lm and we now use it directly.

## A5. Dolma 3 cannot be read in file order — it is laid out by topic, and adult content sorts first

`allenai/dolma3_mix-150B-1025` is 6,082 shards under 66 directories named by source and topic
(`common_crawl-<topic>`, `olmocr_science_pdfs-<topic>`, `stack_edu-<lang>`, `finemath-3plus`,
`dolma1_7-wiki-en`, `rpj-proofpile-arxiv`). Taking "the first shards" alphabetically gives
`common_crawl-adult_content-0014` and nothing else — a slice that is neither Dolma 3 nor
anything you would want to train on.

`r52/ablate/corpora.yaml` therefore carries two fields the other five rows do not need:
`file_order: shuffled` (seed 1337) and `interleave: 24`, and `iter_documents` reads 24 files
round-robin. **The realised slice is uniform by *file*, not by Dolma 3's published token
weights**, because shard sizes vary; `manifest.json` records the exact file list so the slice
is reproducible and the caveat is checkable. A token-weighted Dolma 3 slice would need the
published mixture table, which is not in the repo.

## A6. Ultra-FineWeb: the column is `content`, and one row group is ~225 MB

Two things read out of the live files rather than the card:

* the text column of `openbmb/Ultra-FineWeb` is **`content`**, not `text`;
* its English parquet parts are 1.30 GB with only **10 row groups**, i.e. ~225 MB each, where
  FineWeb-Edu has 726 row groups of ~8 MB. Streaming reads a row group at a time, so the
  *smallest possible* Ultra-FineWeb read is ~200 MB — an order of magnitude more than any
  other corpus here, and the reason it was not exercised during this build (the download
  budget for this session was 300 MB total).

## A7. The evals are hard-wired to GPT-2's BPE, and `r52/eval/` was not to be touched

`r52.eval.lm.LM.__init__` constructs a `GPT2Tokenizer` unconditionally and every eval reads
`lm.tokenizer` from there; `r52.eval.val_loss.evaluate_val_loss` computes bytes-per-token
through the same decoder. Correct for every model in the ladder, wrong for an Axis-3 model
trained on our own 32K vocabulary, and the brief forbids editing `r52/eval/`.

`r52/ablate/evals.py` therefore loads the same `LM`, **replaces `lm.tokenizer`**, and calls
the same `evaluate_hellaswag` / `evaluate_core` / `evaluate_val_loss` functions, so there is
still exactly one implementation of each metric. It refuses to run when the tokenizer's
vocabulary and the checkpoint's output rows disagree, which is the failure this replacement
could otherwise hide. bits-per-byte is computed in `evals.py` rather than by passing `array=`
to `evaluate_val_loss`, because that call cannot be told which tokenizer to decode with.

**If `r52/eval/` is ever opened again**, the clean fix is a `tokenizer=` argument on
`LM.load`; this module would then shrink to a thin CLI.

## A8. Four minimal edits outside the ablation tree

| File | Change | Why unavoidable |
|---|---|---|
| `r52/config.py` | `DataConfig.tokenizer: str = "gpt2"`, plus `resolve_tokenizer()` called from `from_dict` and `apply_overrides` | The spec asks for the field. `model.vocab_size` is *derived* from the tokenizer directory's `meta.json`, because a vocabulary mismatch between shards and embedding table is silent corruption rather than an error. A missing directory is left alone so the tokenizer loader raises the specific message. |
| `r52/tokenizer.py` | `val_bytes_per_token(..., tokenizer=None)`; the cache key is namespaced by the tokenizer's name | Bytes/token is a property of the *(corpus, tokenizer)* pair; decoding a 32K-BPE shard with GPT-2's vocabulary silently produces nonsense, and a shared cache key would let one tokenizer read the other's value. Default behaviour is unchanged. |
| `r52/train.py` | `Trainer._bytes_per_token` passes the configured tokenizer (3 lines, in a setup helper — the core loop is untouched) | Without it a `data.tokenizer: <dir>` run reports a meaningless `val_bpb` in its own log. |
| `r52/export.py` | `_fetch_tokenizer(..., spec)` + `_copy_r52_tokenizer()` | The deliverable is "export must write the HF `tokenizer.json` so mlx-lm loads it". Our tokenizer directory already contains `tokenizer.json` and a `tokenizer_config.json` with the chat template, so the new path copies two files; the GPT-2 path is byte-identical to before. Verified: `mlx_lm.tokenizer_utils.load()` loads the export, round-trips text identically to `R52Tokenizer`, and `apply_chat_template` renders `<\|bos\|><\|user_start\|>hi<\|user_end\|><\|assistant_start\|>`. |

All 326 pre-existing tests still pass.

## A9. `--max-tokens` overshoots by up to one optimizer step, which matters on the tokenizer axis

`r52.train` checks the token budget *before* a step, so a run stops at the first step at or
beyond `--max-tokens` (measured: `--max-tokens 20000` with a 4,096-token step ends at 20,480).
On the corpus and arch axes every cell shares `tokens_per_step`, so all of them overshoot
identically and "matched tokens" survives. On the **tokenizer** axis the budgets differ per
cell (they are derived from bytes), so the two cells can differ by up to one step —
65,536 tokens out of ~90M, i.e. 0.07%. Recorded rather than worked around; `train.tokens` in
each results JSON is the number actually trained on.

## A10. FinePDFs-Edu is a blend cell, which needed a mixer in `prepare_corpus.py`

`docs/ABLATIONS.md` Axis 1 lists FinePDFs-Edu as "≤25% blended into FineWeb-Edu", and
`research/03` §1.1 is explicit that the gain comes from mixing and that PDFs should stay under
25%. So the sixth corpus cell is `fineweb-edu+finepdfs-edu25`, built by
`prepare_corpus.py --corpus fineweb-edu --blend finepdfs-edu:0.25`.

The mixer samples sources by **token deficit**, not per document, for the reason P5 measured
on the midtrain mixture: these corpora differ 2–3× in document length, and per-document
sampling delivers shares proportional to `weight × mean_length`. Verified in
`tests/test_ablate.py::test_blend_hits_its_token_share` with deliberately 10×-different
document lengths: realised 0.25 ± 0.03.

## A11. Two of "the six cheapest CORE tasks" are noise at this scale — planner decision

`docs/ABLATIONS.md` asks for "the 6 cheapest CORE tasks". Measured against the eval bundle,
cost (`items × (few-shot + 1)` forward rows) ranks them: `copa` 100, `winograd` 273,
`bigbench_repeat_copy_logic` 352, `openbook_qa` 500, `agi_eval_lsat_ar` 920, `winogrande`
1,267 — versus 110,462 for the 10-shot `hellaswag` task alone. That set is the shipped
default (`matrix.CORE6`).

But `bigbench_repeat_copy_logic` is 32 items scored by greedy exact match and
`agi_eval_lsat_ar` is LSAT analytical reasoning; both sit at or below their random baselines
for 12–35M-parameter models, so two of the six contribute variance instead of ranking
information. `matrix.CORE6_ALTERNATIVE` swaps them for `lambada_openai` and
`hellaswag_zeroshot` at ~15× the forward rows. **The planner should pick one before the corpus
axis runs**; changing it afterwards invalidates cross-cell comparability.

## A12. Measured: our 32K BPE packs 0.39% more bytes per token than GPT-2's 50K

Trained on 50 MB of FineWeb-Edu, measured on a held-out 2.68 MB of the same stream
(`python -m r52.tokenizer_train --name fineweb-edu-32k --corpus fineweb-edu --bytes 50000000
--vocab-size 32768 --compare-gpt2`, 6.0 s):

| tokenizer | vocab | tokens for 2,677,230 bytes | bytes/token |
|---|---|---|---|
| ours | 32,768 | 576,158 | **4.6467** |
| GPT-2 | 50,257 | 578,400 | 4.6287 |

So the Axis-3 premise "a smaller, better-fitted vocabulary buys fewer tokens for the same
text" is **true but small** on English educational web text — which is close to what GPT-2's
BPE was fitted on in the first place. The real lever at these widths is the head: at `d=384`
a 32,768-row `lm_head` is 12.6M parameters against 19.3M, and the same saving repeats on
`wte` and the value-embedding table. The axis should be read as "is a 35% cheaper vocabulary
free?", not as "is it better compression?".

---

# Deviations — lm-eval bridge

Builder: the lm-evaluation-harness bridge (`docs/ARCHITECTURE.md` §6, last paragraph:
"Later phases add `mlx_lm.evaluate` (lm-eval bridge) for MMLU-Pro/GPQA/GSM8K/IFEval on
exported and post-trained models"). Files: `r52/eval/lmeval.py`, `scripts/eval_lmeval.sh`,
`tests/test_eval_lmeval.py`, dependency lines in `pyproject.toml`. Date: 2026-09-13. Machine:
Mac Mini M4, 16 GB, macOS 26.5.2, MLX 0.32.2, mlx-lm 0.31.3, lm-eval 0.4.13 — **with the
Rung-0 124M pretraining run and a full CORE eval both holding the GPU**, so every wall-clock
below is a three-way-shared-GPU wall-clock.

## L0. `mlx_lm.evaluate` works on our `model_type: r52gpt` plugin exports — so it is wrapped, not replaced

The build brief said: check whether `mlx_lm.evaluate` loads our plugin exports, wrap it if so,
and otherwise register our own `lm_eval.api.model.LM` subclass on top of `r52.eval.lm.LM`.

It works. `mlx_lm.evaluate.MLXLM.__init__` calls `mlx_lm.utils.load`, which calls
`load_model`, which honours `model_file: "r52gpt.py"` — the same hook `r52.export` already
relies on for `mlx_lm.generate` / `server` / `lora`. Verified before writing a line of the
bridge:

```
$ .venv/bin/mlx_lm.evaluate --model models/tiny200-mlx --tasks arc_easy --limit 5 \
    --batch-size 4 --no-apply-chat-template --output-dir <tmp>
{"name": "arc_easy", "sample_len": 5, "acc,none": 0.0, "acc_norm,none": 0.0, ...}
```

(`transformers` prints `You are using a model of type r52gpt to instantiate a model of type ''`
— that is its *config* loader, not mlx-lm's; the weights load through the plugin regardless.)

So `r52/eval/lmeval.py` imports `mlx_lm.evaluate.MLXLM` directly rather than shelling out to
the CLI or reimplementing `loglikelihood` / `loglikelihood_rolling` / `generate_until`. Direct
import buys three things a subprocess would not: `r52.eval.lm.configure_runtime()` runs
*before* the weights load (so the 3 GiB guideline actually applies), the full
`simple_evaluate` return dict is available (`n-samples`, `versions`, `configs`,
`higher_is_better` all land in the results JSON), and the model is loaded once for a whole
suite instead of once per task.

**The custom-LM fallback was therefore not built**, and neither was the "KV-cache-free but
batched loglikelihood path" the brief specified for it. Upstream's `loglikelihood` is better
than that design anyway: it groups requests by context and reuses one prompt cache across a
question's continuations, which is the shared-prefix optimisation `docs/DEVIATIONS.md` §E9
says our own multiple-choice evals leave on the table.

**Measured equivalence.** The bridge and `r52.eval.hellaswag` score the same continuations
through completely different code (masked per-token NLL with no cache, vs fp32
log-probabilities with a cached prefix). On three HellaSwag items × four endings on
`models/tiny200-mlx`, the largest disagreement is **1.5e-5 nats on totals of 45–147 nats** —
fp32 rounding. `tests/test_eval_lmeval.py::test_loglikelihood_matches_hellaswag_scoring`
pins it at 1e-3.

## L1. Upstream bug: IFEval crashes in mlx-lm 0.31.3, and the bridge patches it

`mlx_lm/evaluate.py`:

```python
def _rstrip_until(s, untils):
    l = len(s)
    f = [s.find(u) for u in untils]
    f = [l if x < 0 else x for x in f]
    return s[: min(f)]          # ValueError: min() iterable argument is empty
```

lm-eval's `ifeval` sets `generation_kwargs: {until: []}` — "generate to EOS or the token cap,
do not truncate". With an empty `untils`, `min(f)` raises, so **every IFEval run dies after
paying for the generation**, in post-processing:

```
File ".../mlx_lm/evaluate.py", line 358, in generate_until
    completions[e] = _rstrip_until(text, opt["until"])
ValueError: min() iterable argument is empty
```

`r52.eval.lmeval.patch_upstream()` (called from `build_lm`, idempotent) replaces the module
function with one that returns `s` unchanged for an empty stop list and defers to upstream
otherwise. That is the intended semantics, it is three lines, and it is the only upstream
behaviour this bridge changes. *Worth reporting to ml-explore/mlx-lm.* After the patch,
IFEval runs: `ifeval --limit 5` on `models/gpt2-mlx` → `prompt_level_strict_acc 0.00`,
`inst_level_strict_acc 0.25`, 10.7 s.

## L2. `max_gen_tokens` vs `max_gen_toks`: uncapped generation is 8,192 tokens per prompt

Also in `generate_until`:

```python
max_tokens = [self._max_tokens or opt.get("max_gen_tokens", DEFAULT_MAX_TOKENS) for opt in options]
```

lm-eval's generation kwarg is spelled **`max_gen_toks`**, not `max_gen_tokens`, so the `.get`
never hits and every task falls back to `DEFAULT_MAX_TOKENS = 8192` unless the caller sets
`_max_tokens`. A 5-item GSM8K smoke would generate 40,960 tokens instead of ~1,280.

Rather than patch this one (the fix is upstream's to make; the key name is a judgement call
about which spelling is canonical), every generative `TaskSpec` carries its own
`max_gen_tokens` and `run_task` sets `lm._max_tokens` per task: `gsm8k` 256, `humaneval` 1024,
`hendrycks_math500` 1024, `ifeval` 1280 (the task's own `max_gen_toks`), `mmlu_pro` 2048.
`--max-gen-tokens N` overrides all of them, and — because a truncated generation can lose the
answer and therefore change the score — the value is recorded in the command string in
`docs/RESULTS.md`.

`_max_tokens` is doubly loaded upstream: in `loglikelihood` the *same* attribute is the prompt
**truncation** length. So `run_task` sets it to `context - 1` for loglikelihood tasks, which
is what keeps a 5-shot MMLU prompt from running past `models/gpt2-mlx`'s 1,024 learned
position embeddings. Leaving it at `None` (→ 8,192) would crash GPT-2 on any long few-shot
prompt; setting it to a generation cap like 256 would silently truncate every context.

## L3. Upstream bug not patched: `NameError` on a continuation longer than the context

`mlx_lm/evaluate.py:209` (the `prefix_l == 0` branch of `loglikelihood`) appends to
`all_scores` / `all_is_greedy`, which do not exist in that function — the locals are called
`scores` and `is_greedy`. It fires only when truncation eats the entire prompt, i.e. when one
continuation is longer than the model's context. Fixing it properly means copying ~80 lines of
upstream `loglikelihood` into this repo to change two names, which is exactly the fork the
"wrap, don't reimplement" decision avoids. Instead `_skip_reason` classifies a `NameError` out
of `simple_evaluate` and reports it as *"a continuation is longer than the model's context;
raise `--block-size` or drop the task"*, which is the actionable version of the crash.

## L4. `hellaswag` would have overwritten the llm.c number, so the lm-eval one carries a different unit

`r52/bar/gap.py` keys a results cell on `(model, benchmark, unit)`, and
`r52.eval.report._merge_gap_json` replaces any entry matching that triple. `r52.eval.hellaswag`
already owns `("gpt2-124m", "hellaswag", "% acc_norm")` with **llm.c's** protocol, which §E2
of this file documents as *not the same eval* as lm-eval's `hellaswag` (lm-eval prefixes the
activity label, strips brackets, collapses whitespace, and normalises `acc_norm` by
**characters** rather than tokens). Writing the lm-eval number under the same unit would have
silently destroyed the validated llm.c row.

The bridge therefore reports lm-eval's HellaSwag as `"% acc_norm (lm-eval)"`. Both local
numbers survive in `results/<run>/eval.json`; `gap.py` sorts by `(benchmark, unit)` and shows
the first local entry, so `"% acc_norm"` (llm.c) still wins the displayed cell and the lm-eval
figure sits beside it in `GAP.json`. Every other benchmark id uses `bar.yaml`'s own unit
string verbatim, which `tests/test_eval_lmeval.py::test_bar_benchmark_ids_resolve` asserts.

**Measured, and this is why the distinction matters.** On `models/gpt2-mlx`, first 100
validation items, same slice, three protocols:

| protocol | acc | acc_norm |
|---|---|---|
| llm.c (`r52.eval.hellaswag --limit 100`) | 0.3800 | 0.3300 (bytes: 0.3900) |
| lm-eval (`r52.eval.lmeval --tasks hellaswag --limit 100`) | — | **0.4400** |
| llm.c, **full 10,042** (§E7) | 0.2853 | 0.2938 |

Two separate effects, both worth stating plainly: lm-eval's text rewriting is worth ~5 points
over llm.c's on the same items, and **the first 100 items are ~4–9 points easier than the full
set** — `--limit` in lm-eval takes a prefix, not a random sample. Which is exactly why
research/05 §7.2 says never to compare a `--limit`-ed score to a published full-set number,
and why `conditions["limit"]` in this bridge is never absent: it is the integer, or the string
`"none (full set)"`.

## L5. Two task names in the brief do not exist in lm-eval 0.4.13

`TaskManager().all_tasks` (14,683 entries) has **no `gpqa_diamond`** and **no `math_500`**.
What exists:

| brief | shipped | why |
|---|---|---|
| `gpqa_diamond` | **`gpqa_diamond_zeroshot`** | `gpqa/zeroshot/_gpqa_zeroshot_yaml`, `output_type: multiple_choice`, 0-shot, acc + acc_norm over `(A)`–`(D)`. The alternatives are `gpqa_diamond_n_shot`, `gpqa_diamond_cot_zeroshot`, `gpqa_diamond_cot_n_shot`, `gpqa_diamond_generative_n_shot` and `leaderboard_gpqa_diamond`. Loglikelihood is the cheap one and the only one meaningful for a base model with no chain of thought. |
| `math_500` | **`hendrycks_math500`** | `dataset_path: HuggingFaceH4/MATH-500` — literally the MATH-500 dataset, inheriting `hendrycks_math_algebra`'s generate_until + `exact_match`. `minerva_math500` is the same 500 items with Minerva's prompt and normaliser. The per-subject `hendrycks_math_*` subsets are the full 5,000-item MATH; `leaderboard_math_*_hard` is the Open-LLM-Leaderboard-v2 level-5 subset. |

`tests/test_eval_lmeval.py::test_names_the_brief_asks_for_do_not_exist_under_those_spellings`
asserts *both* directions, so a future lm-eval that adds `gpqa_diamond` or `math_500` fails
loudly instead of leaving the suite quietly pointing at the substitute.

`gpqa_diamond_zeroshot` is also the one **gated** task. Unauthenticated it raises
`DatasetNotFoundError: Dataset 'Idavidrein/gpqa' is a gated dataset on the Hub. You must be
authenticated to access it.` — classified by `_skip_reason` into *"open
https://huggingface.co/datasets/Idavidrein/gpqa, accept the terms, then `huggingface-cli
login` (or set HF_TOKEN)"*. A skip writes **no** results row; a gated benchmark must not
appear in the gap table as a zero.

## L6. `gsm8k` at 8 shots, not `gsm8k_cot`

The brief says "`gsm8k` (8-shot CoT, generative)". lm-eval has both, and they are different
evals: `gsm8k` defaults to **5** shots drawn from the train split, whose targets are the
dataset's own worked solutions ending in `#### N` (so it *is* chain-of-thought, just with
sampled exemplars); `gsm8k_cot` pins the 8 hand-written PaLM exemplars in a `Q:/A:` format and
scores `The answer is N`. The suite ships the task the brief names, `gsm8k`, at
`num_fewshot=8`, and `gsm8k_cot` remains one `--tasks gsm8k_cot` away. The headline metric is
`exact_match,strict-match` (the `#### N` regex), with `flexible-extract` recorded alongside in
`metrics` — a base model that writes the right number in prose scores 0 strict and >0 flexible,
and conflating the two is the most common way GSM8K numbers get inflated.

## L7. `python -m mlx_lm.evaluate` silently does nothing

`mlx_lm/evaluate.py` defines `main()` but has no `if __name__ == "__main__"` guard, so
`python -m mlx_lm.evaluate --model ... --tasks ...` exits 0 in 1.5 s having run nothing. The
working invocations are the console script `.venv/bin/mlx_lm.evaluate` (the
`mlx_lm.evaluate:main` entry point) or `python -c "from mlx_lm.evaluate import main; main()"`.
Recorded because a silent no-op is the worst possible failure mode for an eval harness, and
because `metrics.mlx_lm_equivalent_command` in every results file quotes the console-script
form for exactly this reason.

## L8. Two dependency extras, or two of the seven `standard` tasks cannot run

`lm_eval` declares `evaluate` as a core dependency (HumanEval's `code_eval`) but puts the rest
behind extras, and the ones the `standard` suite needs were not installed:

* `lm_eval[ifeval]` → `langdetect`, `immutabledict`, `nltk` — IFEval's programmatic constraint
  checker. `langdetect` and `immutabledict` were **missing**.
* `lm_eval[math]` → `math_verify`, `antlr4-python3-runtime==4.11`, `sympy` — MATH-500 answer
  normalisation. `math_verify` and `antlr4` were **missing**.

`pyproject.toml` now asks for `lm_eval[ifeval,math]>=0.4.13` (extras, not leaf packages, so a
future lm-eval can move them) and `mlx-lm[evaluate]>=0.31.3` (research/05 §7.1's documented
install path; the extra itself only adds `lm-eval` + `tqdm`). Installing added five packages
and downgraded nothing: `antlr4-python3-runtime 4.11.0`, `immutabledict 4.3.1`,
`langdetect 1.0.9`, `latex2sympy2-extended 1.11.0`, `math-verify 0.9.0`. Without them the
bridge turns the resulting `ImportError` into a named skip rather than a crash, but the suite
is then incomplete — which is worse, because a missing row looks like a choice.

## L9. The commit is read out of `.git` when `git` cannot run — but the CLI is still preferred

The build brief said `git` was broken on this machine and to read `.git/HEAD` manually. It is
**intermittently** broken, not permanently: some invocations return
*"You have not agreed to the Xcode license agreements"* (one fired inside lm-eval's own
`get_git_commit_hash()` during the first `mlx_lm.evaluate` run), while a plain
`git rev-parse --short HEAD` in the same shell later succeeded.

Shipping only the manual reader would have thrown away information, because
`r52.eval.report.git_commit()` runs `git status` as well and therefore knows whether the tree
is **dirty** — the `+dirty` suffix every existing row in `docs/RESULTS.md` carries. So
`r52.eval.lmeval.git_commit()` tries `report.git_commit()` first and falls back to
`git_commit_from_files()`, which reads `.git/HEAD`, follows the ref into `.git/refs/…` or
`.git/packed-refs`, handles a detached HEAD and a worktree `gitdir:` file, and returns the
7-character hash with no subprocess. Measured on this tree: CLI → `7821421+dirty`,
files → `7821421`.

The fallback **cannot** know whether the tree is dirty, so it never appends `+dirty`. Rather
than guess, a results file whose commit came from the fallback carries
`metrics.git_dirty = "unknown (git CLI unavailable; commit read from .git/HEAD)"`. **A bare
hash from this bridge means "dirtiness unknown", not "clean"** — the one place where a row
here is less precisely anchored than the `5c209e2+dirty` rows the eval builder left.
`git_commit_from_files` is unit-tested against a synthetic `.git` (loose ref, packed-refs,
detached HEAD, worktree file, and an unresolvable ref) so the fallback is not first exercised
the day `git` breaks again.

## L10. The bridge evaluates exported directories only, and says so instead of guessing

Every other eval in `r52/eval/` goes through `r52.eval.lm.LM`, which loads an r52 checkpoint
*or* an mlx-lm directory and pins the tokenizer to GPT-2 tiktoken for both (§E8). This one
cannot: lm-eval renders prompts as **text**, so it needs the model's own tokenizer files, and
`mlx_lm.utils.load` needs `config.json`. Pointed at `runs/<run>/ckpt/best`, `_check_model_spec`
raises with the fix rather than a stack trace:

```
runs/tiny200/ckpt/best is an r52 checkpoint, not an mlx-lm model directory. ...
    python -m r52.export runs/tiny200/ckpt/best models/<name>-mlx
    python -m r52.eval.lmeval --model models/<name>-mlx --suite quick
```

Same for an export made with `--no-tokenizer`. A path that does not exist is assumed to be an
HF / mlx-community repo id and passed through untouched.

Related: `use_chat_template` defaults to **False** here, where upstream defaults it to "True
whenever the tokenizer has a chat template". Every r52 export ships one (`r52.chat_template`,
so `mlx_lm.chat` works on base models too), so upstream's default would silently flip every
base-model loglikelihood eval into chat mode and change every score. `--apply-chat-template`
turns it on for post-trained models, and the flag is recorded in `conditions`.

## L11. Quantization is recorded on every row, because research/05 §7.1 measured what it costs

All of our exports are bf16, so every row here says `quantization: none (bfloat16)`. The field
exists anyway because the moment the bridge is pointed at an `mlx-community/*-4bit` repo the
number moves: Apple's own `mlx_lm/BENCHMARKS.md`, quoted in research/05 §7.1, measures
bf16 → q4 costing **3.3 MMLU-Pro points**, q4 g32 2.6, and q6 only 0.5. "Evaluate at q6 or q8,
never q4" is only checkable if the score carries the quantization, so `quantization_string()`
renders `q6 g64` / `q4` / `none (bfloat16)` into `conditions` on every run.

## L12. Measured: GPT-2 124M, `quick` suite, `--limit 100`

`scripts/eval_lmeval.sh models/gpt2-mlx quick --limit 100` (124 s wall-clock on a GPU already
holding a pretraining run and a CORE eval; peak 0.35 GiB against the 3 GiB guideline):

| task | metric | measured (limit 100) | ± stderr | literature, full set |
|---|---|---|---|---|
| `arc_challenge` | acc_norm | **24.00 %** | 4.29 | ~19–23 % |
| `piqa` | acc_norm | **62.00 %** | 4.88 | ~62–63 % |
| `winogrande` | acc | **50.00 %** | 5.03 | ~51.6 % |
| `lambada_openai` | acc | **34.00 %** | 4.76 | ~32–35 % |
| `hellaswag` | acc_norm (lm-eval) | **44.00 %** | 4.99 | 31.14 % |

Four of five land inside a standard error of the published range. **HellaSwag does not**, and
§L4 above measures why: it is the `--limit` prefix, not the harness — llm.c's protocol on the
*same* 100 items scores 0.3300 acc_norm against 0.2938 on the full 10,042. None of these five
numbers may be quoted as GPT-2's score; they are GPT-2's score on the first 100 items, which
is what the `limit 100` in every `conditions` string says.

`models/tiny200-mlx` (a 200-step, 128-context `r52gpt` plugin export) at `--limit 20`, 57 s:
`arc_challenge` 15.00, `piqa` 50.00, `winogrande` 50.00, `lambada_openai` 0.00, `hellaswag`
20.00 — i.e. chance or below on everything, which is the correct answer for a model trained
for 200 steps, and the point of running it is that the plugin path produces numbers at all.

Generative smoke, `--tasks gsm8k ifeval --limit 5 --max-gen-tokens 128` on `models/gpt2-mlx`
(24 s): `gsm8k` `exact_match,strict-match` **0.00 %** (8-shot, flexible-extract also 0.00),
`ifeval` `prompt_level_strict_acc` **0.00 %** (`inst_level_strict_acc` 0.25). Zero is the
expected answer — a 124M base model neither does arithmetic nor follows instructions — and the
point of the smoke is that the generative path produces a number instead of crashing (§L1) or
generating 8,192 tokens per prompt (§L2).

**The full `standard` suite was never run**, per the brief and per research/05 §7.3's estimate
of 2–4 days for unlimited MMLU-Pro alone on this machine.

## L13. These validation runs deliberately did not touch `results/` or `docs/RESULTS.md`

Every run above used `R52_RESULTS_DIR=<scratch>`, which also redirects the markdown row to
`<scratch>/RESULTS.md`. Two reasons: this builder owns neither file, and a CORE eval was
running concurrently with `--report --run gpt2-124m-reference`, so two processes appending to
`docs/RESULTS.md` would have raced on a read-modify-write. To land the numbers for real, drop
the variable:

```bash
R52_MODEL_ID=gpt2-124m R52_RUN=gpt2-124m-reference \
  scripts/eval_lmeval.sh models/gpt2-mlx quick --limit 100
```

which writes `results/gpt2-124m-reference/lmeval_{arc_challenge,piqa,winogrande,
lambada_openai,hellaswag}.json`, merges five entries into that run's `eval.json` in `gap.py`'s
flat schema, and appends five rows to `docs/RESULTS.md`. `piqa`, `winogrande`,
`lambada_openai` and `mmlu` have no `bar.yaml` benchmark id and are dropped from the gap table
by `gap.merge_local`, exactly as `gap.py` documents for any unknown id; their JSON files are
written regardless.

# Deviations — job queue

*Builder note, 2026-09-13. `r52/queue.py`, `scripts/queue.sh`, `queue/v1.yaml`,
`tests/test_queue.py`. Every job command below was verified against its CLI's `--help` (and,
where that was not enough, its source), not templated from the plan text.*

## Q1. `done_when` grew from "a file or glob" into a small ANDed predicate dict

Two shapes of job need more than "this path exists". `rung0-wait`, and every job that
launches through a self-backgrounding `scripts/{train,sft,rl}.sh`, needs "the trainer process
is gone AND its checkpoint exists": `ckpt/best` is written the first time validation improves
on `math.inf`, which happens at ~step 250 of 5,722 for Rung 0, so "the file exists" alone
would call a run three days from finished "done" the moment it produced its first good
checkpoint. `prep-corpora` needs six manifests, not one. `is_done()` therefore accepts either
a plain string (a path/glob, unchanged from the spec) or a dict of `file` / `glob` / `files` /
`pid_dead` keys, ANDed together. `pid_dead` is the compound predicate's other half: a pidfile
that is missing, unreadable, or names a dead process.

## Q2. `gpu_wait_only`: one field added beyond the spec's seven, and only `rung0-wait` uses it

"Must detect an externally running `r52.train` ... as 'GPU busy' and wait rather than launch"
has to run before every `needs_gpu` job's launch, as a safety net independent of the
`after`-chain that normally keeps two GPU jobs from overlapping. Applied unconditionally, it
would also block `rung0-wait` forever: `rung0-wait`'s entire purpose is to wait on exactly the
`r52.train` process that check is designed to detect, so it needs to be exempt from its own
reason for existing. `gpu_wait_only: bool` (default `false`) is that exemption; it is set on
exactly one of the 20 jobs (`tests/test_queue.py::test_real_queue_v1_yaml_is_well_formed_and_
matches_the_design` asserts this). Verified live, not only in a test double: the real
`gpt2-124m-mac` `r52.train` (PID 46011, started 13:58, still training while this was built) is
what `_external_gpu_busy()` matches on this machine right now — an early run of the test suite
caught its own false positive from exactly that, before the autouse fixture that neutralizes
it for every test except the two that opt back in on purpose.

## Q3. Two commands in the plan text needed correcting, not templating

* **`nano-midtrain`** uses `configs/posttrain/midtrain_nano_train.yaml`, not
  `configs/posttrain/midtrain_nano.yaml`. The latter is the mixture-BUILDER recipe consumed
  by `scripts/prepare_midtrain_data.py` (`sources:` / `weight:` / `out_dir: data/midtrain`, no
  `model:`/`data:`/`train:` blocks); `r52.posttrain.midtrain` calls `r52.config.load_config`
  on its `config` argument, which raises on unknown top-level keys, so the mixture recipe
  would be rejected immediately. `midtrain_nano_train.yaml`'s own header gives the intended
  invocation almost verbatim (`--init-from runs/a1/ckpt/best --run-name m1`); this queue's
  jobs use `nano-a` / `nano-m1` throughout instead of the docstring's bare `a1` / `m1`, both
  to match the plan's own `nano-a` and to avoid colliding with a quick manual smoke run under
  one of those short names (`runs/mt-tiny`, `runs/sft-tiny`, `runs/rl-tiny` already exist from
  exactly that kind of run).
* **`nano-rl-export`** copies `runs/nano-rl/final/` to `models/r52-nano-30m-rl-mlx/` instead
  of calling `r52.export`. `GRPOTrainer.save()` (`r52/posttrain/rl.py`) already writes
  `final/` as a complete mlx-lm model directory — `config.json` / tokenizer files copied from
  the base export plus a fresh bf16 `model.safetensors` — matching `scripts/rl.sh`'s own
  footer ("an mlx-lm model directory; chat with scripts/chat.sh"). `r52.export`'s positional
  `ckpt` argument is documented and implemented as an **r52** checkpoint directory
  (`model.safetensors` + `optim.safetensors` + `meta.json`, `r52/checkpoint.py`); pointed at
  an mlx-lm directory instead, it has no matching weight names to convert.

## Q4. `r52.posttrain.sft` and `r52.posttrain.rl` have no `--resume`

Checked directly against both CLIs (neither `build_parser` lists it, and neither module
imports anything named `resume`) rather than assumed by analogy with `r52.train`'s and
`r52.posttrain.midtrain`'s. `nano-sft` and `nano-rl`'s `resume_cmd` is therefore identical to
their `cmd`: the automatic retry is a full restart from `--init-from` / `--model`, not an
incremental continuation. `nano-pretrain` and `nano-midtrain` do get a real resume (`--resume`
and `--resume --run-name nano-m1` respectively — the latter is never combined with
`--init-from`, which `r52.posttrain.midtrain.build_trainer` rejects outright).

## Q5. Re-attaching after a restart cannot recover a real exit code — `done_when` decides instead

Every job's `cmd` runs under `subprocess.Popen(..., start_new_session=True)`, so a killed or
crashed supervisor leaves it running rather than orphaning it into termination — exactly what
a multi-hour training job needs. On the next `run`, `reconcile()` re-attaches any job
`state.json` still marks `running` whose PID is alive, wrapped as a `Ghost` rather than a real
`Popen`. A `Ghost` is not this process's child, so it can be polled for liveness
(`os.kill(pid, 0)`) but never reaped for a genuine exit status the way `Popen.poll()` reaps a
process this instance actually launched. This is why `is_done()` — not the exit code — is the
single source of truth for every job's outcome, checked the same way whether a job just
exited under this process's own `Popen` handle or was re-attached: there was never a version
of this design where a restart could trust an exit code, so nothing here special-cases one.

The one real bug this surfaced during testing (fixed, not deviated around): terminating a
*real* `Popen` on a `timeout_h` breach by polling `os.kill(pid, 0)` alone leaves it an
unreaped zombie, which still answers "alive" to that same check. `_terminate()` calls
`Popen.wait()` when it has a real handle, and only falls back to liveness polling for a
`Ghost`, which cannot be waited on at all.

## Q6. `needs_gpu` is `r52.train`-shaped work only, not "runs on the GPU"

Every eval (`scripts/eval.sh`), export (`r52.export`), and the pass@k probe
(`r52.posttrain.passk`) is `needs_gpu: false` even though all three run inference through MLX
on the same Metal device. `eval.sh`'s own comment gives the reason ("this machine also runs a
multi-day pretraining job ... `--memory-limit-gib`"), and it is not theoretical here: while
this queue was being built, `r52.eval.core --model models/gpt2-mlx` (an unrelated reference
eval, PID 61199) ran for over 20 minutes alongside the live `gpt2-124m-mac` `r52.train`
(PID 46011) without incident. `docs/ABLATIONS.md`'s "one GPU, one training job" and
`scripts/ablate.sh`'s refusal to start are both specifically about a second `r52.train`, never
about eval/export/inference sharing the device with one — matching the `needs_gpu` split here.

## Q7. `queue/v1.yaml`'s prep jobs found real, pre-existing data — with one real mismatch

Before this queue existed, a shell chain launched separately (by a different agent working
the plan concurrently) had already run every `prep-*` job's equivalent by hand. Three of the
four match this queue's own commands exactly and are correctly skipped by `done_when`:
`data/tokenizers/fineweb32k`, all six `data/corpora/*-fineweb32k/manifest.json`, and
`data/midtrain/mixture.json`. The fourth does not: `data/sft/smol-smoltalk` was built with
`--n 50000` (46,904 conversations, finished 17:52:44), not the `--n 200000` that
`configs/posttrain/sft_nano.yaml`'s own header documents (and that `prep-sft-data`'s `cmd`
here uses). `prep-sft-data`'s `done_when` is deliberately just "the directory's two
`meta.json` files exist" — per the design brief, "each job's `done_when` must detect that and
skip" — so it will skip rebuilding it as-is. The 50k-conversation set is not wrong, only
smaller than documented; the planner should decide whether to accept it or clear
`data/sft/smol-smoltalk` before the first `scripts/queue.sh` run so `prep-sft-data` rebuilds
it at the documented size.

# Review fixes (2026-09-13, adversarial review of the training core)

- **C1 (critical, operational):** the periodic validation is the *first* 262,144 tokens of the split, which measures ~0.094 nats easier than the full 10,485,760-token split on `models/gpt2-mlx` (3.3536 vs 3.4471). `--target-val-loss` compared against it and broke out of the loop mid-plateau, skipping the cooldown. Fixes: the trainer now confirms on the full split before stopping (`event: val_full`), and `configs/gpt2_124m_mac.yaml` disables early stopping; the live Rung 0 run is resumed from its step-250 checkpoint with early stopping off.
- **M2:** the config's "±0.004 nats" was the i.i.d.-token standard error; the honest SE of the 128-batch prefix mean is ~0.04 nats and the offset is a bias, not noise. Periodic val is for curve shape only; reported numbers always come from `scripts/eval.sh` on the full split.
- **M3:** `mlx.optimizers.AdamW` defaults `bias_correction=False` (torch's Adam corrects) → ~2.4× larger AdamW steps for the first ~20 steps. New `TrainConfig.adam_bias_correction` (default `True`); the live run keeps `false` so its resume is semantically identical.
- **m4:** the bf16 training-loss log-sum-exp bias is −5.2e-3 nats on real logits (−5.2e-4 was on random logits); training loss only, validation is fp32.
- **m5:** `mlx.optimizers.Muon` applies `weight_decay` *coupled* (added to the gradient before Newton–Schulz), unlike modded-nanogpt's decoupled decay; keep `muon_weight_decay: 0.0`.
- **m7–m10:** shard-length check uses the shortest shard; checkpoint staging dir renamed `.step_*.tmp` (outside the `step_*` glob, avoids a SIGTERM race on `best/`); HellaSwag `n_end` and CORE `schema` slice guards for degenerate truncation.
