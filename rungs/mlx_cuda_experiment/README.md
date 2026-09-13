# MLX-on-CUDA — the portability experiment

Package for the claim in [`research/02-open-training-stack.md`](../../research/02-open-training-stack.md)
§Stage 9 and [`docs/PLAN.md`](../../docs/PLAN.md) §4-§5: **`r52` is written entirely in MLX, and MLX
has shipped a CUDA backend since early 2026.** If that backend covers what `r52` actually uses, the
same `r52/` codebase that trains on this Mac Mini M4 should run, unmodified, on a rented H100 — no
PyTorch port, no torchtitan rewrite. This package does **not** prove that; it verifies what MLX's
CUDA backend supports today against MLX's own primary sources, and designs the measurement that
would prove or disprove it. Nothing in this package has been run: no training, no evals, no package
installs, no GPU touched, no money spent. Every claim below is sourced to a URL and dated
2026-09-13; anything not directly confirmed is marked **UNVERIFIED**.

Fallback if this doesn't pan out: [`rungs/rung1_nanochat/`](../rung1_nanochat/) (nanochat, PyTorch)
and [`rungs/rung2_3b/`](../rung2_3b/) (torchtitan) remain the paid rungs' primary path regardless of
what this experiment finds — see `expectations.md`'s comparison rows.

## Pinned versions

```
mlx (ml-explore/mlx)      commit 229f5b430df7926743c5b6ac62068cae2ebc8978  (2026-09-12, main HEAD)
                           latest tagged release v0.32.2 (2026-08-25) -- same version this repo's
                           Mac already runs (results/mlx_pretrain_bench.md: "mlx 0.32.2")
road-to-52 (this repo)    commit 4bd7ba570bfb13dcc4c39c3bf1cc4bcb963e4933  (2026-09-13)
```

Found with `gh api repos/ml-explore/mlx/commits/main --jq '{sha,date:.commit.committer.date}'` and
`gh api repos/SammyTourani/road-to-52/commits/main --jq '{sha,date:.commit.committer.date}'`.
`setup.sh` clones and checks out the road-to-52 commit above by default; re-verify both before a
real run if much time has passed (same rule as every other package in `rungs/`, see
[`../README.md`](../README.md) checklist item 1).

## 1. What MLX's CUDA backend supports today

Verified 2026-09-13 by fetching MLX's own README, docs source (`docs/src/...rst`), release notes,
and backend source tree via `gh api repos/ml-explore/mlx/contents/<path>?ref=<sha>` (raw file
fetches quoted below all resolve at commit `229f5b430df7926743c5b6ac62068cae2ebc8978`), plus PyPI's
JSON API for the actually-published wheels. This is not a re-statement of the CUDA backend's
existence (the build brief already knew that) — it is what its docs and source commit to, in its
own words, with dates.

### 1.1 Installing it — the exact extra name, verified (the brief's assumption was wrong)

The build brief for this package assumed `pip install "mlx[cuda]"`. MLX's own install docs
([`docs/src/install.rst`](https://raw.githubusercontent.com/ml-explore/mlx/229f5b430df7926743c5b6ac62068cae2ebc8978/docs/src/install.rst))
document a **versioned** extra instead:

```
pip install mlx[cuda12]      # CUDA 12.x
pip install mlx[cuda13]      # CUDA 13.x -- needs driver >= 580
```

Cross-checked against the real wheels on PyPI (`curl -s https://pypi.org/pypi/mlx/json`, 2026-09-13,
`mlx` 0.32.2's `requires_dist`):

```
mlx-cuda-12==0.32.2 ; platform_system == "Linux" and extra == "cuda"     <- bare [cuda] IS also live
mlx-cuda-12==0.32.2 ; platform_system == "Linux" and extra == "cuda12"
mlx-cuda-13==0.32.2 ; platform_system == "Linux" and extra == "cuda13"
mlx-cpu==0.32.2     ; platform_system == "Linux" and extra == "cpu"
mlx-metal==0.32.2   ; platform_system == "Darwin"
```

So **both `mlx[cuda]` and `mlx[cuda12]` resolve to the identical `mlx-cuda-12` wheel today** — the
brief's syntax isn't wrong, it's just not future-proof (`[cuda]` will presumably re-point to
`cuda13` or later at some point). `setup.sh` uses the explicit, documented `mlx[cuda12]` for that
reason, per install.rst directly rather than the unversioned alias.

**System requirements** (install.rst, verbatim):

| Requirement | Value |
|---|---|
| GPU architecture | Nvidia SM >= 7.5 |
| Driver | >= 550.54.14 (cuda12) / >= 580 (cuda13) |
| CUDA toolkit | >= 12.0 |
| Linux | glibc >= 2.35 (Ubuntu 22.04 "jammy" is the practical floor) |
| Python | >= 3.10 |

The actual uploaded wheel confirms the glibc floor: `mlx-0.32.2-cp312-cp312-manylinux_2_35_x86_64.whl`
(PyPI file listing, 2026-09-13) — `manylinux_2_35` is exactly glibc 2.35. An **H100 is SM 9.0**
([developer.nvidia.com/cuda-gpus](https://developer.nvidia.com/cuda-gpus), fetched 2026-09-13),
comfortably above the SM 7.5 floor.

### 1.2 Op and dtype coverage

MLX's CUDA backend is a first-party backend living at
[`mlx/backend/cuda/`](https://github.com/ml-explore/mlx/tree/229f5b430df7926743c5b6ac62068cae2ebc8978/mlx/backend/cuda)
next to `mlx/backend/metal/`, not a shim. Its file listing (`gh api
repos/ml-explore/mlx/contents/mlx/backend/cuda`) shows native `.cu`/`.cpp` implementations across
the same broad op families Metal has: elementwise (`unary/`, `binary*.cu`, `ternary.cu`), reductions
(`reduce*`, `scan.cu`, `arg_reduce.cu`, `logsumexp.cu`, `softmax.cu`, `cross_entropy.cu`), sorting
(`sort.cu`), GEMM/matmul (`matmul.cpp`, `gemms/`, `steel/` — a CUTLASS-based tiled-GEMM library, plus
`cublas_utils.*`), convolution (`conv.cpp`, `conv/`, `cudnn_utils.*`), quantized matmul
(`quantized/`), linear algebra (`cholesky.cu`, `cusolver_utils.*`), FFT (`fft.cu`), RoPE (`rope.cu`),
RMSNorm/LayerNorm (`rms_norm.cu`, `layer_norm.cu`), scaled-dot-product-attention
(`scaled_dot_product_attention.cu`/`.cpp`), Hadamard transform (`hadamard.cu`), random
(`random.cu`), and distributed collectives (`distributed.cu`). **This is directory-listing evidence
of broad coverage, not an exhaustive per-op/per-dtype parity audit against Metal — whether every op
Metal supports has an equally complete CUDA implementation (edge dtypes, uncommon shapes) is
UNVERIFIED.** Standard dtypes (float32, float16, bfloat16, complex64, the int family) are supported
via the CUTLASS/cuBLAS GEMM paths; low-precision quantization formats (int4/int8, and newer
microscaling formats referenced in recent changelogs — `ST_F8_E8M0`, NVFP4, mxfp8 block scales) are
present but **their coverage for full training (not just inference quantization) is UNVERIFIED** —
see 1.4.

### 1.3 `mx.fast.*` kernel coverage on CUDA

MLX's fast-kernel module ([`docs/src/python/fast.rst`](https://raw.githubusercontent.com/ml-explore/mlx/229f5b430df7926743c5b6ac62068cae2ebc8978/docs/src/python/fast.rst))
lists: `rms_norm`, `layer_norm`, `cross_entropy`, `rope`, `scaled_dot_product_attention`,
`metal_kernel`, **`cuda_kernel`**, **`precompiled_cuda_kernel`** — i.e. CUDA gets its own
custom-kernel-authoring API (`cuda_kernel`), the CUDA analogue of `metal_kernel`, not a fallback
onto it. All three ops `r52` actually calls (`mx.fast.rms_norm`, `mx.fast.rope`,
`mx.fast.scaled_dot_product_attention` — see 1.6) have dedicated `.cu` source files in
`mlx/backend/cuda/` confirmed above. Release-note evidence these are live, maintained kernels, not
stubs (all `gh api "repos/ml-explore/mlx/releases?per_page=30"`, 2026-09-13):

- `[CUDA][Improvement] RMSNorm forward speed up` — [PR #3850](https://github.com/ml-explore/mlx/pull/3850), shipped v0.32.1 (2026-08-18)
- `[CUDA][Improvement] RMSNorm backward` — [PR #3881](https://github.com/ml-explore/mlx/pull/3881)
- `[CUDA][Improvement] Rope without copy` — [PR #3704](https://github.com/ml-explore/mlx/pull/3704)
- `[CUDA] Use cuDNN SDPA for decoding when using fixed-size KV cache` — [PR #3113](https://github.com/ml-explore/mlx/pull/3113)
- `Add force_fused option to scaled_dot_product_attention` — [PR #4185](https://github.com/ml-explore/mlx/pull/4185), shipped v0.32.2 (2026-08-25)

`docs/src/usage/environment_variables.rst` documents CUDA-specific SDPA tuning that only makes
sense against a real, cached, cuDNN-integrated implementation: `MLX_CUDA_USE_CUDNN_SDPA` (default
`1` — "Allow the CUDA backend to use cuDNN scaled dot-product attention when the inputs and device
are supported"), `MLX_CUDA_SDPA_CACHE_SIZE` (default 256), `MLX_CUDA_SDPA_BACKWARD_CACHE_SIZE`
(default 64).

### 1.4 `mx.compile` on CUDA

Works, via CUDA graph capture/replay rather than Metal command buffers. Confirmed by
`environment_variables.rst`: `MLX_USE_CUDA_GRAPHS` ("Enable CUDA graph capture and replay. The
default is `1`"), `MLX_SAVE_CUDA_GRAPHS_DOT_FILE`, `MLX_CUDA_GRAPH_CACHE_SIZE` (default 400),
`MLX_MAX_OPS_PER_BUFFER`/`MLX_MAX_MB_PER_BUFFER` ("one Metal command buffer **or CUDA graph**").
Runtime kernel compilation is backed by its own JIT/PTX pipeline
(`mlx/backend/cuda/jit_module.{h,cpp}`, `ptx.cuh`, `MLX_PTX_CACHE_DIR` env var, `CUDA_HOME`/
`CUDA_PATH` for locating headers) — analogous to Metal's `MLX_METAL_JIT` kernel cache described in
`install.rst`. `docs/src/usage/compile.rst`'s `mx.compile` semantics (recompile triggers, pure-
function requirement, `inputs=`/`outputs=` state capture, `shapeless=True`) are documented as
backend-agnostic — nothing in that page is Metal-specific, and `r52/train.py`'s own compile call
(`mx.compile(fn, inputs=[model.state], outputs=[model.state])`) is exactly the "Compiling Training
Graphs" idiom the doc itself recommends. Whether compiled-graph *speedup magnitude* on CUDA matches
Metal's (the doc's own example: 5x on an M1 Max) is **UNVERIFIED** — that's an empirical question
for Phase 1 below, not a docs claim either way.

### 1.5 `mx.distributed` NCCL backend and `mlx.launch`

Confirmed real, not aspirational: [`mlx/distributed/nccl/`](https://github.com/ml-explore/mlx/tree/229f5b430df7926743c5b6ac62068cae2ebc8978/mlx/distributed/nccl)
exists as a full backend directory alongside `mpi/`, `ring/`, `jaccl/`, and `mlx/backend/cuda/distributed.cu`
implements the CUDA-side collective ops. Per
[`docs/src/usage/distributed.rst`](https://raw.githubusercontent.com/ml-explore/mlx/229f5b430df7926743c5b6ac62068cae2ebc8978/docs/src/usage/distributed.rst)
and [`docs/src/usage/launching_distributed.rst`](https://raw.githubusercontent.com/ml-explore/mlx/229f5b430df7926743c5b6ac62068cae2ebc8978/docs/src/usage/launching_distributed.rst),
quoted directly:

> "NCCL — The backend of choice for CUDA environments." … "For CUDA environments, NCCL is the
> default backend for `mlx.launch` and all it takes to run a distributed job is `mlx.launch -n 8
> test.py`" … "The NCCL backend is the default backend for CUDA environments. When launching from a
> Mac to a Linux machine with CUDA then the backend should be selected using `--backend nccl`."

Single-node 8-GPU: `mlx.launch -n 8 my_script.py`. Multi-node: `mlx.launch --backend nccl --hosts
linux-1,linux-2 -n 8 -- ./my-job.sh` ("will attempt to launch 16 processes, 8 on each node"). Without
`mlx.launch` (e.g. under a cluster scheduler), the backend needs `MLX_RANK`, `MLX_WORLD_SIZE`,
`NCCL_HOST_IP`, `NCCL_PORT`, `CUDA_VISIBLE_DEVICES` set per-process (`distributed.rst` §"Distributed
Without mlx.launch"). The program-level API is exactly `world = mx.distributed.init()` +
`mx.distributed.all_sum(...)`; `mlx.nn.average_gradients` is a shipped helper
(`python/mlx/nn/utils.py`, verified by fetching that file) that batches many small per-tensor
all-reduces into few large ones — signature: `average_gradients(gradients, group=None,
all_reduce_size=32*1024**2, communication_stream=None)`, internally `mx.distributed.all_sum(...) /
N` over concatenated tensor groups. `bench_and_train.sh` §3 below uses this, not a raw `all_sum`
loop, for the sketched 8-GPU change.

### 1.6 Known limitations / actively evolving (not a finished, static backend)

From the release-note sweep (`gh api "repos/ml-explore/mlx/releases?per_page=30"`, 2026-09-13),
read as a timeline rather than a snapshot:

- **CUDA FSDP is explicitly WIP as of the current release series.** `[WIP] [CUDA] fsdp` —
  [PR #3768](https://github.com/ml-explore/mlx/pull/3768), shipped inside **v0.32.1 (2026-08-18)**,
  our locally-installed version's immediate predecessor. An earlier, separate "(easy)" FSDP PR
  ([#3130](https://github.com/ml-explore/mlx/pull/3130)) shipped back in **v0.31.1 (2026-03-12)**.
  Full parameter-sharded training on CUDA should be treated as **not production-ready** — data
  parallel (§1.5, what this package's experiment actually uses) is the mature path; sharding a model
  across GPUs (the thing that would let a bigger-than-one-GPU model train at all) is not.
- **Quantized matmul (QMM) on CUDA is comparatively new and still gaining coverage.** "Initial
  version of QMMs for CUDA" shipped in **v0.31.0 (2026-02-28)**, barely 6.5 months before this
  package was written; v0.31.1/v0.31.2's own "Highlights" call out "Wider support for cuda quantized
  matmuls" as a named release highlight, and 3/5/6-bit, fp, and gather-variant quantized kernels
  landed across a dozen further PRs through mid-2026. Doesn't affect `r52` pretraining directly (no
  quantized training path in `r52`), but matters if this package's later phases ever export through
  `mlx-lm`'s quantization tooling on a CUDA box.
- **CUDA FFT support is recent** (v0.31.2, 2026-04-22, [PR #3243](https://github.com/ml-explore/mlx/pull/3243)) — irrelevant to `r52` (no FFT in the model), noted for completeness.
- Windows CUDA is also a supported target in CI (`dhiltgen`'s recurring `win: fix cuda build` PRs) —
  not relevant here, noted because it confirms the CUDA backend is not a Linux-only afterthought.
- **What is genuinely UNVERIFIED, not just "not found":** whether `mx.fast.scaled_dot_product_attention`'s
  convenience string mask (`mask="causal"`, exactly what `r52/model.py`'s `Attention` block passes)
  behaves identically on the CUDA path vs Metal's; the actual achieved MFU on CUDA for a small
  (<1B) dense transformer at these sequence lengths; whether `mx.compile`'s CUDA-graph fusion nets
  the same *relative* speedup as Metal's kernel fusion. All three are exactly what Phase 1 (§2)
  measures rather than assumes.

### 1.7 `mlx-lm` / `mlx-lm-lora` on Linux+CUDA — the brief's assumption doesn't hold at the packaging level

The build brief asked to check "whether `mlx-lm` and `mlx-lm-lora` may be macOS-only" via PyPI
classifiers/wheels. Done (`curl -s https://pypi.org/pypi/<pkg>/json`, 2026-09-13):

| Package | Version | Wheel | Platform-gated? |
|---|---|---|---|
| `mlx-lm` | 0.31.3 | `mlx_lm-0.31.3-py3-none-any.whl` | **No** — pure-Python wheel, no OS tag |
| `mlx-lm-lora` | 3.1.2 | `mlx_lm_lora-3.1.2-py3-none-any.whl` | **No** — pure-Python wheel, no OS tag |

Both are `py3-none-any` — plain Python packages with no compiled extension and no OS classifier.
`mlx-lm-lora`'s own declared dependencies (`mlx>=0.30.6`, `mlx_lm>=0.30.6`, `numpy`,
`transformers>=4.39.3`, `protobuf`, `pyyaml`, `jinja2`, `tqdm`, `datasets`) are all
cross-platform too. **Neither package is macOS-only at the packaging level** — they install on
Linux because their one platform-coupled dependency, `mlx` itself, now ships Linux+CUDA wheels
(§1.1). This corrects the build brief's premise rather than confirming it.

What is **not** verified: that every *code path* inside `mlx-lm`/`mlx-lm-lora` is exercised
correctly on the CUDA backend at runtime (e.g., any conditional branch that assumes Metal, or an
`mx.fast.metal_kernel` call anywhere in their source) — this session made no installs and ran no
code, per the build constraints, so **runtime CUDA-correctness of mlx-lm/mlx-lm-lora is
UNVERIFIED**, only their installability.

### 1.8 The good news specific to `r52`

`r52/model.py`'s entire fast-kernel surface is three calls — `mx.fast.rms_norm(x, None, eps)`
(parameter-free RMSNorm), `mx.fast.rope(...)`, and
`mx.fast.scaled_dot_product_attention(q, k, v, scale=self.scale, mask="causal")` — confirmed by
`grep -rn "mx\.fast\." r52/*.py r52/**/*.py`, which also confirms **zero uses of
`mx.fast.metal_kernel`** (the Metal-only custom-kernel escape hatch) anywhere in `r52`, including
`r52/mlx_plugin/r52gpt.py` (the mlx-lm export plugin). All three ops `r52` actually depends on have
confirmed native `.cu` implementations (§1.3). `r52/train.py`'s `mx.compile` usage matches the
documented CUDA-portable idiom (§1.4). Nothing in `r52/model.py`, `r52/train.py`, `r52/optim.py`, or
`r52/data.py` was found (by inspection, not execution) to call a Metal-specific API. This is
evidence *for* the "one codebase" story, not proof — proof is Phase 1.

## 2. Experiment design

Three phases, each gated on the previous one not hitting an abort criterion. `bench_and_train.sh`
implements Phases 1 and 3 directly; Phase 2 (8-GPU) is documented with the exact `mlx.launch`
invocation but needs the code change in §1.5/`bench_and_train.sh` §3 first — **`r52.train` does not
support `mx.distributed` today (UNVERIFIED nothing, this is a fact about the code, checked by
`grep -rn "distributed" r52/*.py` returning nothing)**.

### Phase 0 — environment sanity (minutes, ~$0)

`setup.sh` ends with `python -c "import mlx.core as mx; print(mx.cuda.is_available())"` and the
non-macOS pytest subset. If `mx.cuda.is_available()` is `False` on a box with `nvidia-smi` working,
stop — that's a packaging/driver problem, not a coverage problem, and nothing past this point is
meaningful until it's fixed.

### Phase 1 — 1×H100 bench sweep

`python -m r52.bench --sizes 124M,350M,1B --seqs 1024 --precisions mixed,bf16 --mem-limit-gib 70
--micro-batch 32 --out results/mlx_cuda_bench` (see `bench_and_train.sh` §1 for the exact,
dry-runnable command; the `1B` size is **not yet in `r52.bench`'s `SIZES` dict** — see
`expectations.md`'s derivation and the note in `bench_and_train.sh`).

Compare against three things, all already in this repo or derivable from it:
1. **Our own Mac numbers** (`results/mlx_pretrain_bench.md`) — same code, same `flops_per_token`
   formula, different accelerator. A clean speedup-per-TFLOP-of-peak comparison.
2. **`expectations.md`'s 6ND-derived tok/s table** at 30/40/50% MFU — did the real run land inside,
   above, or below the modeled range.
3. **nanochat's disclosed real-world MFU on this exact hardware class** — nanochat Run 6
   (`rungs/rung1_nanochat/README.md`) doesn't publish an MFU number directly (N/D undisclosed), but
   the ladder's calibration rows (SmolLM3 25.7%, DeepSeek-V3 33.1%, Llama 3.1 405B 34.6% — all
   `results/ladder.md`) bracket what "large, serious" H100 training MFU looks like; a small (<1B),
   single-GPU, non-FSDP, non-fp8 run like ours should land **at or below** the 25.7% low anchor,
   not above it — SmolLM3's own case is exactly "small models and long-context-heavy recipes do not
   reach 35%" (`results/ladder.md`'s own note). torchtitan expectations are qualitative only here:
   `rungs/rung2_3b/README.md`'s job is a 3B/60B-token multi-day run at a different scale, not a
   throughput number we can borrow directly — no torchtitan single-GPU 124M-scale MFU figure is
   sourced anywhere in this repo, so any torchtitan comparison at our sizes is **UNVERIFIED** unless
   a source is found before running.

### Phase 2 — 8×H100, `mx.distributed` NCCL data-parallel

Same three sizes, same sweep, launched via `mlx.launch -n 8` (single node) once the code change in
`bench_and_train.sh` §3 exists. Question this answers: does per-GPU tok/s hold roughly flat going
1→8 GPUs (good data-parallel scaling) or drop (NCCL/gradient-averaging overhead dominating at these
model sizes, which are small enough that this is a real risk — see `expectations.md`'s note that the
0.75B-token real run's GPU-hours are identical at 1 and 8 GPUs; only wall-clock changes, so 8 GPUs
buys *time*, not *efficiency*, unless the model is big enough that per-GPU compute dwarfs
communication).

### Phase 3 — short real run to the 3.28 bar

`scripts/train.sh configs/gpt2_124m_mac.yaml gpt2-124m-h100 -o train.memory_limit_gb=70 -o
train.micro_batch=32` (verified mechanism — see `bench_and_train.sh` §2), then `scripts/eval.sh
runs/gpt2-124m-h100/ckpt/best standard`. Same config, same optimizer hyperparameters, same token
budget (0.75B) and target (val loss ≤ 3.28) as the Mac's live Rung 0 run — the only things that
change are hardware and the two memory/batch overrides. Record actual wall-clock and actual $ (not
the modeled figures in `expectations.md`) into `docs/RESULTS.md` per `rungs/README.md`'s checklist.

### Success criteria

- Phase 0 passes (CUDA visible, non-macOS pytest subset green).
- Phase 1 measures a non-zero, non-error tok/s for all three sizes at both precisions, at MFU ≥ 15%
  (see abort criteria) against the M4's 4.26/3.6 TFLOPS-normalized numbers scaled to H100's 989
  TFLOPS.
- Phase 3 reaches val loss ≤ 3.28 (matching the target in `configs/gpt2_124m_mac.yaml`) inside the
  0.75B-token budget, or documents honestly how far it got if not (same rule `docs/PLAN.md` §3 Phase
  3 already applies to the Mac run: "val ≤ 3.28 or budget exhausted — either way, published
  honestly").
- No op/kernel from `r52`'s actual call surface (§1.8) is missing or silently wrong on CUDA.

### Abort criteria — stop and document, don't push through

- **MFU < 15%** on the 124M/1024/mixed config at 1×H100 (roughly half the SmolLM3 low anchor) —
  something is structurally wrong (not just "small model, expected lower MFU"), stop and profile
  before spending more.
- **A required op throws `NotImplementedError` or silently produces wrong values** — specifically
  watch `mx.fast.scaled_dot_product_attention`'s `mask="causal"` path (§1.6's flagged UNVERIFIED
  item) and `mx.compile` recompilation behavior under the grad-accumulation loop's repeated
  `tree_map` calls (`r52/train.py::train_step`).
- **`mx.distributed.init(backend="nccl")` fails to form an 8-GPU group** on a single node — debug
  with `mlx.launch -n2 -- my_script.py` locally first (the docs' own recommended first step, §1.5)
  before blaming the multi-node path.
- **Cost overrun**: if Phase 1+3 combined exceeds $20 (roughly 4x `expectations.md`'s modeled
  ceiling for the whole sequence), stop and re-check the MFU/wall-clock assumptions rather than
  continuing to spend.

## 3. UNVERIFIED — consolidated

1. Exhaustive op/dtype parity between the CUDA and Metal backends (§1.2) — directory-listing
   evidence only.
2. `mx.fast.scaled_dot_product_attention`'s `mask="causal"` string-mask behavior specifically on
   CUDA vs Metal (§1.3, §1.6).
3. Relative speedup magnitude of `mx.compile` on CUDA vs Metal's documented ~5x example (§1.4).
4. Runtime (not packaging) CUDA-correctness of every `mlx-lm`/`mlx-lm-lora` code path (§1.7).
5. Any torchtitan single-GPU, ~124M-350M-scale throughput/MFU figure for a same-hardware-class
   comparison (§2 Phase 1) — none found in this repo's `research/` corpus or this session.
6. That `r52.train` supports `mx.distributed` — it does not today; §1.5/`bench_and_train.sh` §3
   describe, not claim, the change.
7. Actual measured MFU, tok/s, and wall-clock/cost for all three experiment phases — that is the
   entire point of running them; every number in `expectations.md` is a **projection**, not a result.

## Files

| File | What |
|---|---|
| `README.md` | this file |
| `setup.sh` | provider-agnostic Ubuntu+NVIDIA setup: uv, Python 3.12, `mlx[cuda12]`, pinned repo clone, data shards, non-macOS test subset. `--dry-run`; `bash -n` clean |
| `bench_and_train.sh` | Phase 1 bench sweep, Phase 3 real run + eval, and the documented (not implemented) Phase 2 8-GPU `mlx.launch` invocation + the ~20-line `mx.distributed` change sketch. `--dry-run`; `bash -n` clean |
| `expectations.md` | 6ND-derived tok/s-at-MFU table, the 1B-class config derivation, 0.75B-token run cost/wall-clock table, and the nanochat/modded-nanogpt comparison rows — every number sourced or derived with the formula shown |
