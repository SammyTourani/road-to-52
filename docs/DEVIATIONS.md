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
