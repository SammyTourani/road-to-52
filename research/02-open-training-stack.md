# The Open-Source LLM Training Stack — 2026 Survey

**Compiled:** 2026-09-13. All star counts / last-push dates verified via the GitHub REST API (`gh api repos/OWNER/REPO`) on that date unless noted. Items that could not be confirmed from a primary source are marked **UNVERIFIED**.

**Read this first — five things that changed and that most 2025-era guides get wrong:**

1. **torchtune is dead — and so is its replacement.** torchtune wound down in 2025 ([issue #2883](https://github.com/meta-pytorch/torchtune/issues/2883), *"we are stopping active development on torchtune, effective immediately"*), and `meta-pytorch/torchforge` — the thing everyone points to as the successor — posted its own *"Development paused"* banner on **2026-04-23**, the same day, redirecting to torchtitan. **The live answer is TitanRL inside torchtitan.** Meta burned through two post-training repos in 18 months; do not build on either.
2. **NVIDIA NeMo was dismantled into ~28 separate repos** under a new `NVIDIA-NeMo` org. The old monolith `NVIDIA/NeMo` now redirects to `NVIDIA-NeMo/Speech` (speech only). ([org listing](https://github.com/orgs/NVIDIA-NeMo/repositories))
3. **SGLang ships a real Apple Silicon MLX backend** (`python/sglang/srt/hardware_backend/mlx/`) — Apple Silicon is no longer an inference dead-end for the serving frameworks. ([source tree](https://github.com/sgl-project/sglang/tree/main/python/sglang/srt/hardware_backend/mlx))
4. **Levanter merged into the Marin monorepo** (Nov 2025); the standalone repo is frozen. ([merger commit](https://github.com/marin-community/levanter/commits/main))
5. **Terminal-Bench became Harbor.** `laude-institute/terminal-bench` redirects to `harbor-framework/terminal-bench-1`; the live harness is `harbor-framework/harbor`. ([harbor](https://github.com/harbor-framework/harbor))

Legend for "Alive?": **Hot** = pushed within 7 days · **Active** = within ~60 days · **Stale** = 2–12 months · **Dead/Frozen** = >12 months or explicitly wound down.

---

## Stage 1 — Full-stack "LLM in one repo" pipelines

| Repo | ★ | License | Last push | What it does | Scale target | Hardware | Alive? | Verdict |
|---|---|---|---|---|---|---|---|---|
| [karpathy/nanochat](https://github.com/karpathy/nanochat) | 57,979 | MIT | 2026-09-07 | The whole loop in one repo: BPE tokenizer train → pretrain → midtrain → SFT → RL → eval → chat CLI/web UI | $100 / ~1.5–2 h on 8×H100 → d26+ tiers | CUDA primary; **CPU + MPS path exists** via [`runs/runcpu.sh`](https://github.com/karpathy/nanochat/blob/master/runs/runcpu.sh) | **Hot** | **The reference implementation of "an LLM end to end."** Nothing else is this complete or this readable. Start here. |
| [karpathy/autoresearch](https://github.com/karpathy/autoresearch) | 95,675 | **none (all rights reserved)** | 2026-03-26 | AI agents autonomously edit `train.py` on a simplified single-GPU nanochat, train 5 min, keep/revert on val_bpb | 1 GPU, overnight agent loops | Single NVIDIA GPU (H100 tested); community forks add macOS/Windows/AMD | **Stale** (created 2026-03-06) | Biggest star-velocity event of 2026 (95K★ in ~6 months, 13.4K forks). **Has no license file** — legally not open source. Its optimizations fed back into the nanochat leaderboard. |
| [karpathy/nanoGPT](https://github.com/karpathy/nanoGPT) | 63,041 | MIT | 2025-11-12 | Minimal GPT-2 pretrain/finetune, ~300 lines of model | 124M–1.5B, 8×A100 | CUDA; runs on CPU/MPS slowly | **Dead-ish** | Still the best *teaching* code, but superseded by nanochat for anything end-to-end. Use it to learn, not to build. |
| [KellerJordan/modded-nanogpt](https://github.com/KellerJordan/modded-nanogpt) | 5,788 | MIT | 2026-08-09 | The NanoGPT speedrun: reach 3.28 val loss on FineWeb as fast as possible on 8×H100 | 124M | 8×H100, CUDA only | **Active** | **The single best source of "what actually makes training faster."** Every technique is ablated in public. Mine it for optimizer/architecture tricks; don't use it as a framework. |
| [karpathy/llm.c](https://github.com/karpathy/llm.c) | 30,987 | MIT | 2025-06-26 | GPT-2/GPT-3 training in raw C/CUDA, no PyTorch | up to GPT-2 1.5B multi-node | CUDA (+ CPU ref) | **Dead** (15 mo) | Historically important — it was the modded-nanogpt baseline (45 min). Read it to understand kernels. Do not build on it. |
| [huggingface/smollm](https://github.com/huggingface/smollm) | 3,891 | Apache-2.0 | 2026-05-26 | SmolLM/SmolLM2/SmolLM3 configs + recipes; pretraining scripts target **nanotron** | SmolLM3 = 3B on 11.2T tokens, 384×H100 for 24 days | CUDA | **Stale** | **The best fully-documented "real" small-model recipe.** 3-stage data curriculum, 128K Llama-3.2 vocab, midtrain to 64k ctx + 35B reasoning tokens, SFT then APO. ([blog](https://huggingface.co/blog/smollm3)) |
| [allenai/OLMo-core](https://github.com/allenai/OLMo-core) | 1,530 | Apache-2.0 | 2026-09-13 | PyTorch pretraining building blocks; exactly reproduces every Olmo 3 checkpoint | 1B–32B, in-house H100 clusters | CUDA (flash-attn, ring-flash-attn, TransformerEngine, torchao FP8, grouped_gemm MoE, QuACK) | **Hot** (v2.6.0, 2026-08-11) | **The most genuinely reproducible large-model recipe in existence** — weights + data + code + checkpoints. Pair with `allenai/open-instruct` for post-training. |
| [allenai/open-instruct](https://github.com/allenai/open-instruct) | 3,862 | Apache-2.0 | 2026-09-13 | Ai2's post-training codebase: SFT → DPO → RLVR (the Olmo 3 "Dolci" flow) | 7B–32B | CUDA | **Hot** | The other half of Olmo. Olmo 3 shipped Base/Instruct/Think/**RL-Zero** variants at 7B and 32B on the ~9.3T-token Dolma 3 corpus. ([Ai2 blog](https://allenai.org/blog/olmo3)) |
| [allenai/OLMo](https://github.com/allenai/OLMo) | 6,675 | Apache-2.0 | 2025-11-24 | Original Olmo 1/2 modeling+training repo | — | CUDA | **Dead/Frozen** | Superseded by OLMo-core. Don't start here. |
| [allenai/olmo-cookbook](https://github.com/allenai/olmo-cookbook) | 75 | Apache-2.0 | 2026-07-21 | "OLMost every training recipe" — data-intervention recipes for the OLMo family | — | CUDA | **Active** | Small but high-signal if you want to run data ablations the way Ai2 does. |
| [marin-community/marin](https://github.com/marin-community/marin) | 3,628 | Apache-2.0 | 2026-09-13 | Stanford CRFM's **open lab**: the whole experiment-tracked pipeline, data → model → eval. Levanter now lives inside it | Marin 8B and **32B** trained on TPU | **TPU-first** (JAX/XLA), GPU supported | **Hot** | **The most radical transparency of any project here** — every experimental detour is public. Marin 32B Base beat Gemma 3 27B PT on 24/42 base evals. TPU bias makes it awkward if you rent NVIDIA. ([Google OSS blog](https://opensource.googleblog.com/2025/12/training-marin-32b-what-an-open-lab-can-build-with-tpus-jax-and-a-little-persistence.html)) |
| [marin-community/levanter](https://github.com/marin-community/levanter) | 714 | Apache-2.0 | 2026-01-26 | JAX/Haliax named-tensor trainer; FSDP + TP, bitwise determinism on TPU, Sophia optimizer | nano → 32B | TPU + GPU (JAX) | **Frozen** — "merged into Marin as of November 2025" | `pip install levanter` still works, but file issues on Marin. Use Marin instead. |
| [jzhang38/TinyLlama](https://github.com/jzhang38/TinyLlama) | 9,015 | Apache-2.0 | 2024-05-03 | 1.1B Llama on 3T tokens | 1.1B | CUDA | **ARCHIVED** | Purely historical. Do not use. |

### Apple Silicon ports of the stage-1 pipelines

| Repo | ★ | License | Last push | Notes |
|---|---|---|---|---|
| **nanochat itself** ([`runs/runcpu.sh`](https://github.com/karpathy/nanochat/blob/master/runs/runcpu.sh)) | — | MIT | script tuned 2026-01-17 | First-party CPU/MPS path. Tokenizer train ~34 s, a **depth-6** model ~30 min, SFT ~10 min on an M3 Max. Karpathy's own words: *"Training LLMs requires GPU compute and $$$. You will not get far on your Macbook."* |
| [scasella/nanochat-mlx](https://github.com/scasella/nanochat-mlx) | 70 | MIT | 2026-04-29 | Most-starred community MLX port of nanochat. **Stale** (~4.5 mo). |
| [awni/picochat](https://github.com/awni/picochat) | 12 | MIT | 2026-03-06 | "Nanochat in MLX" by Awni Hannun, the MLX lead at Apple — **but it is an EMPTY PLACEHOLDER.** Contents are `.gitignore`, `LICENSE`, and a two-line `README.md`; created and last pushed the same day. Watch it; do not depend on it. |
| [btgaskin/mlx_nanochat](https://github.com/btgaskin/mlx_nanochat), [xbsd/nanochat_mlx](https://github.com/xbsd/nanochat_mlx), [alex-rentel/eden-nanochat](https://github.com/alex-rentel/eden-nanochat) | 0 | MIT / NOASSERTION | 2026-02 → 2026-07 | Long tail of ports. None maintained. |

**Verdict on Apple Silicon for stage 1:** there is no *maintained, high-quality* MLX port of a full pipeline. The realistic Mac path is nanochat's own `runcpu.sh` (first-party, MPS) for learning, and `awni/picochat` for MLX-idiomatic reference.

### The speedruns — current records

**modded-nanogpt (124M → 3.28 val loss on FineWeb, 8×H100):** the record is **1.23 minutes**, record #89, set 2026-07-17, via *"MLP down projection in FP8 with efficient delayed scaling metric."* Progression: 45 min (llm.c baseline, 2024-05-28) → 7.8 min (bf16 activations, 2024-11) → 2.992 min (merged QKV + long-short attention, 2025-01) → 2.128 min (2025-12) → 1.521 min (2026-02) → **1.23 min**. Stack of techniques in the current record: **Muon optimizer, FlashAttention-3, value embeddings, logit softcapping, FP8 quantization, paired-head attention, learnable XSA.** A second track targets GPT-2-Medium quality (2.92 loss, 350M): record **17.35 min** (2025-12-31). ([README](https://github.com/KellerJordan/modded-nanogpt))

**nanochat (`dev/LEADERBOARD.md`, all 8×H100):**

| Date | Depth | Hours | CORE | What changed |
|---|---|---|---|---|
| 2026-01-29 | d24 | 3.04 | 0.25851 | baseline |
| 2026-02-02 | d26 | 2.91 | 0.2578 | **fp8 training via torchao** |
| 2026-02-05 | d26 | 2.76 | 0.26024 | batch size doubled to 1M |
| 2026-03-03 | d24 | ~2.80 | 0.25714 | FineWeb-EDU → **NVIDIA ClimbMix** dataset |
| 2026-03-09 | d24 | ~2.02 | 0.2690 | **autoresearch-discovered optimizations** |
| 2026-03-14 | d22/d24 | **1.65** | 0.262634 | "backout and smear" architecture changes |

Reference point given in the same file: GPT-2 (1.6B) scored **0.256525** CORE in ~168 h on 32 TPUv3 (~$43K in 2019). A 1.65-hour 8×H100 run now beats it. ([LEADERBOARD.md](https://github.com/karpathy/nanochat/blob/master/dev/LEADERBOARD.md))

---

## Stage 2 — Pretraining frameworks at scale

| Repo | ★ | License | Last push | Precision | MoE | Parallelism | Hardware | Alive? | Verdict |
|---|---|---|---|---|---|---|---|---|---|
| [pytorch/torchtitan](https://github.com/pytorch/torchtitan) | 5,729 | BSD-3 | 2026-09-13 | bf16, **float8**, **MXFP8 (Blackwell)**, bf16 optimizer states | Yes (MXFP8 MoE) | FSDP2, TP + async TP, PP (zero-bubble), CP (1M seq), DDP/HSDP | CUDA (H100/B200 CI); **[AMD fork](https://github.com/AMD-AGI/torchtitan-amd)**; CPU unit tests only | **Hot** | **The default choice for PyTorch-native pretraining in 2026.** Clean-room, readable, composable 4D parallelism. Models include Llama 3, Qwen3/3.5/3.8, DeepSeek V3/V4, GPT-OSS, Kimi K2.7/K3, Flux. **2026/08 added [TitanRL](https://github.com/pytorch/torchtitan/tree/main/torchtitan/experiments/rl)** — shared model defs/kernels across training and vLLM generation, with batch-invariant mode. |
| [NVIDIA/Megatron-LM](https://github.com/NVIDIA/Megatron-LM) (Megatron Core) | 17,871 | NOASSERTION | 2026-09-13 | FP16, BF16, **FP8, FP4** | Yes — [MoE scaling tech report 2026-03](https://arxiv.org/abs/2603.07685), dedicated MoE roadmaps | TP, PP, DP, **EP**, CP (+ dynamic CP, 1.48× on var-len) | CUDA only | **Hot** (core_v0.19.0, 2026-08-19) | **Still the performance ceiling at frontier scale.** Development moved fully into the open in 2025-12. Muon and other new optimizers via [Emerging-Optimizers](https://github.com/NVIDIA-NeMo/Emerging-Optimizers). DeepSeek-V4 support on `dev`. Steep learning curve; NVIDIA-locked. |
| [NVIDIA-NeMo/Megatron-Bridge](https://github.com/NVIDIA-NeMo/Megatron-Bridge) | 907 | Apache-2.0 | 2026-09-13 | inherits Megatron | Yes | inherits Megatron | CUDA | **Hot** | **The piece that makes Megatron usable.** Bidirectional HF ↔ Megatron checkpoint conversion plus production recipes. If you touch Megatron, you want this. |
| [NVIDIA-NeMo/Automodel](https://github.com/NVIDIA-NeMo/Automodel) | 941 | Apache-2.0 | 2026-09-13 | — | — | PyTorch-distributed native | CUDA | **Hot** | NVIDIA's lighter "train HF models out of the box" library. The NeMo successor for people who don't want Megatron. |
| NVIDIA NeMo (monolith) | — | — | — | — | — | — | — | **DISSOLVED** | `NVIDIA/NeMo` → redirects to [`NVIDIA-NeMo/Speech`](https://github.com/NVIDIA-NeMo/Speech) (18,435★). LLM training moved to Megatron-Bridge / Automodel; RL to NeMo-RL; data to Curator; eval to Evaluator. **Any guide telling you to `pip install nemo_toolkit` for LLMs is out of date.** |
| [marin-community/marin](https://github.com/marin-community/marin) | 3,628 | Apache-2.0 | 2026-09-13 | JAX/XLA | UNVERIFIED | FSDP + TP via Haliax | **TPU-first**, GPU ok | **Hot** | See stage 1. Best choice *if* you're on TPU. |
| [AI-Hypercomputer/maxtext](https://github.com/AI-Hypercomputer/maxtext) | 2,420 | Apache-2.0 | 2026-09-13 | JAX/XLA | UNVERIFIED (MoE configs exist) | JAX sharding | **TPU-first**, GPU supported | **Hot** (maxtext-v0.2.4, 2026-08-21) | Google's reference JAX LLM. The TPU-scale workhorse. Less legible than Levanter/Marin, more battle-tested at Google scale. |
| [huggingface/nanotron](https://github.com/huggingface/nanotron) | 2,821 | Apache-2.0 | 2026-04-07 | UNVERIFIED FP8 | UNVERIFIED | 3D parallelism | CUDA | **Stale** (~5 mo) | The framework SmolLM3 was actually trained with. Real but cooling off; HF's own energy has moved to TRL/lighteval/datatrove. |
| [mosaicml/llm-foundry](https://github.com/mosaicml/llm-foundry) (+ Composer) | 4,443 | Apache-2.0 | 2026-03-25 | — | — | FSDP | CUDA | **Stale** (~6 mo) | Databricks' internal foundation-model code. Was excellent in 2023–24; now clearly deprioritized. Don't start a new project on it. |
| [facebookresearch/lingua](https://github.com/facebookresearch/lingua) | 4,765 | BSD-3 | 2025-06-18 (last commit) | — | — | FSDP | CUDA | **Dead** (15 mo) | Meta's "lean and hackable" research codebase. Abandoned; Meta's effort went to torchtitan/torchforge. |
| [huggingface/picotron](https://github.com/huggingface/picotron) | 2,299 | Apache-2.0 | 2025-08-26 | — | — | minimal 4D parallelism | CUDA | **Dead** (12.5 mo) | **Explicitly educational** — the best ~1000-line read for understanding DP/TP/PP/CP. Never meant for production; judge it as a textbook, not a framework. |

**Supporting libraries worth knowing:** [pytorch/ao](https://github.com/pytorch/ao) (2,973★, 2026-09-11 — the float8/MXFP8 quantized-training backend torchtitan and nanochat both use), [NVIDIA/TransformerEngine](https://github.com/NVIDIA/TransformerEngine) (3,532★, 2026-09-11 — FP8/FP4 on Hopper/Ada/Blackwell), [deepspeedai/DeepSpeed](https://github.com/deepspeedai/DeepSpeed) (43,101★, 2026-09-13 — still alive and still the ZeRO reference, but FSDP2 has taken most new PyTorch work).

---

## Stage 3 — Tokenizers

| Repo | ★ | License | Last push | What it does | Hardware | Verdict |
|---|---|---|---|---|---|---|
| [huggingface/tokenizers](https://github.com/huggingface/tokenizers) | 11,033 | Apache-2.0 | 2026-09-11 | Rust BPE/WordPiece/Unigram; train **and** infer; the `transformers` default | CPU (any, incl. Apple Silicon) | **Default answer.** Trains and serves, huge ecosystem, works everywhere. Pick this unless you have a reason not to. |
| [openai/tiktoken](https://github.com/openai/tiktoken) | 19,234 | MIT | 2026-08-17 | Very fast BPE **inference** — no training code | CPU | Fastest encode/decode, but you cannot train a vocab with it. Pair with a trainer. |
| [karpathy/rustbpe](https://github.com/karpathy/rustbpe) | 517 | MIT | 2026-01-03 | *"The missing tiktoken training code"* — now its own repo, a `nanochat` dependency | CPU | **The clean answer to tiktoken's gap.** nanochat's pattern — train with rustbpe, infer with tiktoken — is the tidiest small-scale setup going. Low stars, high quality. |
| [google/sentencepiece](https://github.com/google/sentencepiece) | 12,072 | Apache-2.0 | 2026-09-12 | Unigram + BPE, language-agnostic, the Llama/Gemma/T5 lineage | CPU | Still maintained and still correct, but new models have largely moved to tiktoken-style byte-level BPE. Use it for compatibility, not for greenfield. |

**Vocab sizes modern models actually use:**

| Model | Vocab | Source |
|---|---|---|
| nanochat (default) | **32,768** (2^15) | [`scripts/tok_train.py --vocab-size`](https://github.com/karpathy/nanochat/blob/master/scripts/tok_train.py) |
| SmolLM3 | **~128K** (Llama-3.2 tokenizer, bos removed) | [HF blog](https://huggingface.co/blog/smollm3) |
| Llama 3.x | **~131K** (100K tiktoken + 28K multilingual) | [arXiv 2508.17771](https://arxiv.org/pdf/2508.17771) |
| DeepSeek-V3/R1 | **~130K** | [arXiv 2508.17771](https://arxiv.org/pdf/2508.17771) |
| Qwen | **~151K** | [arXiv 2508.17771](https://arxiv.org/pdf/2508.17771) |
| 2026 range generally | **128K–256K** | [arXiv 2508.17771](https://arxiv.org/pdf/2508.17771) |

**The practical rule:** small models get small vocabs. nanochat deliberately uses 32K and even narrows the digit-split pattern to `\p{N}{1,2}` rather than GPT-4's `{1,3}` because *"I didn't want to waste too many tokens on numbers for smaller vocab sizes"* — verified at 32K as the sweet spot. Frontier models sit at 128K–256K mostly to buy multilingual coverage. Scale your vocab to your parameter count, not to fashion.

---

## Stage 4 — Post-training: SFT + preference + RL

### Liveness first — four of the names on most 2025 shortlists are dead

| Repo | ★ | License | Last commit | Commits since 2026-06-15 | Status |
|---|---|---|---|---|---|
| [meta-pytorch/torchtune](https://github.com/meta-pytorch/torchtune) | 5,812 | BSD-3 | 2026-04-23 | **0** | ☠️ **FROZEN.** [Issue #2883](https://github.com/meta-pytorch/torchtune/issues/2883) (2025-07-15): *"we are stopping active development on torchtune, effective immediately."* Last release v0.6.1, 2025-04-07. |
| [meta-pytorch/torchforge](https://github.com/meta-pytorch/torchforge) | 702 | BSD-3 | 2026-04-23 | **0** | ☠️ **PAUSED.** *"Development in Forge has paused. LLM training at PyTorch is being consolidated in torchtitan."* Never shipped a release. |
| [ChenmienTan/RL2](https://github.com/ChenmienTan/RL2) | 1,311 | Apache-2.0 | 2026-05-20 | **0** | ☠️ **Dead.** No release, ever. |
| [sail-sg/oat](https://github.com/sail-sg/oat) | 669 | Apache-2.0 | 2026-01-29 | **0** | ☠️ **Dead (~7.5 mo).** |
| [NousResearch/atropos](https://github.com/NousResearch/atropos) | 1,351 | — | 2026-07-04 | — | ☠️ **ARCHIVED / read-only.** |

**The chain to understand: torchtune (dead 2025-07) → torchforge (paused 2026-04) → [TitanRL](https://github.com/pytorch/torchtitan/tree/main/torchtitan/experiments/rl) inside torchtitan (current).** TitanRL reuses torchtitan model definitions and kernels across training and vLLM generation and ships **batch-invariant mode for bitwise-identical logprobs** between trainer and sampler — a direct attack on the train/inference mismatch problem. Its tree contains `rubrics/`, `environment/`, `rollout/`, `routing/`, `actors/` — a full agentic stack, not a toy.

### The live frameworks

| Repo | ★ | License | Last commit | Commits /90d | Release | Verdict |
|---|---|---|---|---|---|---|
| [verl-project/verl](https://github.com/verl-project/verl) | 23,401 | Apache-2.0 | 2026-09-11 | 350 | v0.9.0 (2026-08-14) | **SOTA / the default.** Moved to its own `verl-project` org in 2026-01. |
| [huggingface/trl](https://github.com/huggingface/trl) | 19,294 | Apache-2.0 | 2026-09-13 | **610** | v1.13.0 (2026-09-10) | **Transformed, and now genuinely agentic.** Highest commit velocity here. |
| [axolotl-ai-cloud/axolotl](https://github.com/axolotl-ai-cloud/axolotl) | 12,467 | Apache-2.0 | 2026-09-12 | 156 | v0.19.0 (2026-09-10) | Alive, SFT-first, strong parallelism. |
| [OpenPipe/ART](https://github.com/OpenPipe/ART) | 10,713 | Apache-2.0 | 2026-09-12 | 137 | v0.5.19 (2026-08-14) | Alive; best ergonomics; LoRA-only. |
| [OpenRLHF/OpenRLHF](https://github.com/OpenRLHF/OpenRLHF) | 9,998 | Apache-2.0 | 2026-09-13 | 34 | v0.11.1 (2026-09-10) | Alive but low velocity. |
| [THUDM/slime](https://github.com/THUDM/slime) | 8,451 | Apache-2.0 | 2026-09-03 | 112 | v0.3.2 (2026-08-28) | **SOTA — the frontier-proven one.** |
| [areal-project/AReaL](https://github.com/areal-project/AReaL) | 5,751 | Apache-2.0 | 2026-09-12 | 114 | v2.1.0 (2026-08-25) | Alive; major v2 microservice rewrite. |
| [alibaba/ROLL](https://github.com/alibaba/ROLL) | 3,394 | Apache-2.0 | 2026-08-24 | **16** | v0.3.0 (2026-06-18) | ⚠️ **Decelerating hard.** Great agentic examples, but velocity collapsed. |
| [NovaSky-AI/SkyRL](https://github.com/NovaSky-AI/SkyRL) | 2,295 | Apache-2.0 | 2026-09-12 | 192 | v0.3.0 (2026-07-17) | **Alive; agentic leader.** |
| [PrimeIntellect-ai/prime-rl](https://github.com/PrimeIntellect-ai/prime-rl) | 2,035 | Apache-2.0 | 2026-09-13 | 366 | v0.9.0 (2026-08-25) | **SOTA — async agentic at scale.** |
| [NVIDIA-NeMo/RL](https://github.com/NVIDIA-NeMo/RL) | 2,010 | Apache-2.0 | 2026-09-13 | **432** | v0.7.0 (2026-07-25) | Very alive. Highest commit count in the RL-only set. |
| [hiyouga/LlamaFactory](https://github.com/hiyouga/LlamaFactory) | 74,732 | Apache-2.0 | 2026-09-09 | 59 | v0.9.5 (2026-05-30) | Alive, but **not an RL framework** — no GRPO; punts to [EasyR1](https://github.com/hiyouga/EasyR1). |
| [unslothai/unsloth](https://github.com/unslothai/unsloth) | 76,082 | Apache-2.0 | 2026-09-12 | **2,695** | v0.1.808-beta (2026-09-09) | **Alive but pivoted to a desktop app** — see below. |

### Capability matrix

| | Algorithms | Async / off-policy | Multi-turn agentic | Rollout | Train backend | Scale | FP8 rollout |
|---|---|---|---|---|---|---|---|
| **verl** | PPO, GRPO, **GSPO**, DAPO, DrGRPO, REINFORCE++, RLOO, ReMax, PRIME, SPPO, VAPO, KL_Cov/Clip_Cov | `fully_async_policy`, `one_step_off_policy`, `transfer_queue` (all experimental) | ✅ agent_loop + tool calling, SGLang multiturn, `uni-agent` | vLLM, SGLang, **TRT-LLM**, HF | FSDP, FSDP2, Megatron, **TorchTitan**, VeOmni, Automodel | 1 GPU → trillion-param @ 64×H800 | ✅ |
| **slime** | PPO, GRPO, **GSPO**, **CISPO**, DAPO, REINFORCE++, **TIS** | fully-async, OPD, delta weight sync | ✅ coding-agent RL (Claude Code/Codex), multi-agent, sandboxes, PD-disagg | **SGLang only** (deliberate) | **Megatron only** | multi-node, frontier | ✅ **BF16 train + FP8 rollout is the production path**; FP8 KV cache |
| **prime-rl** | GRPO, hierarchical GRPO, OPD, OPSD, RAE self-play, SFT | **fully async by design** | ✅ `verifiers` + Environments Hub, SWE/agentic envs | **vLLM only** | **FSDP2 only** | **1T+ MoE on 1000+ GPUs** | ✅ FP8 + MXFP8 kernels |
| **SkyRL** | GRPO, DAPO, **CISPO**, DrGRPO, PPO, clip_cov/kl_cov, custom losses | `fully_async` + `one_step_off_async`, in-flight weight updates | ✅ `skyrl-agent` + `skyrl-gym`, **Harbor terminal agents**, mini-SWE-agent | vLLM (+SGLang), **Tinker API** | FSDP, Megatron | multi-node; 397B run | ✅ |
| **AReaL** | GRPO, PPO, DAPO, REINFORCE, RLOO, **LitePPO**, DrGRPO, **GSPO**, IcePop, KPop | **fully-async is the core thesis** | ✅ swap `base_url` to RL-train *any* black-box agent; tau2-bench, SWE | SGLang (default), vLLM | FSDP2, Megatron | multi-node | ✅ |
| **NeMo-RL** | GRPO, **GSPO**, DAPO, PPO, **CISPO**, GDPO, MOPD, DPO, SFT, on-policy + cross-tokenizer distillation | async rollouts, replay buffers, fully-async GRPO | ✅ multi-turn w/ tools + NeMo-Gym | vLLM, SGLang, Megatron-Inference, TRT-LLM, Dynamo | DTensor/FSDP2, Megatron-Core (6D) | 1 GPU → multi-node | ✅ **end-to-end FP8** |
| **OpenRLHF** | PPO, GRPO, REINFORCE++(-baseline), RLOO, DAPO, **GSPO**, DrGRPO, DPO | `--train.async_enable`; `is_correction_type ∈ {tis, icepop, seq-mask-tis}` | ✅ unified agent pipeline, multi-turn VLM RL | vLLM | DeepSpeed; new "Molt" backend | multi-node via Ray | ⚠️ **UNVERIFIED** |
| **ROLL** | GRPO, **GSPO**, PPO, **LitePPO**, REINFORCE++, **TOPR**, GiGPO, StarPO, DPO | async rollout ✅; async *training* "under implementation" | ✅ **strongest agentic examples** (frozen_lake, Sokoban, SWE, GEM tool-use) | vLLM, SGLang | FSDP2, Megatron-Core, DeepSpeed | multi-node, Ascend NPU, AMD | ✅ |
| **TRL** | `loss_type ∈ {grpo, dr_grpo, **dapo (default)**, bnpo, cispo, sapo, luspo, vespo}`; **GSPO** via `importance_sampling_level="sequence"`; DPO, KTO, RLOO, Reward, **Distillation** | **`AsyncGRPOTrainer`**, `AsyncDistillationTrainer` | ✅ **OpenEnv harness**, env-owned rewards, tool-exposing envs | vLLM (server or colocate) | DDP, DeepSpeed, FSDP via Accelerate | 1 GPU → multi-node | via torchao |
| **axolotl** | GRPO, **async GRPO** (+58% step speed), GDPO, DPO, IPO, KTO, ORPO, RM/PRM | ✅ [PR #3486](https://github.com/axolotl-ai-cloud/axolotl/pull/3486) | ⚠️ via NeMo-Gym integration | vLLM | FSDP1, FSDP2, DeepSpeed, ND-parallel (CP×TP×FSDP) | multi-node | ✅ FP8 + **NVFP4/MXFP4 QAT** |
| **ART** | GRPO + experimental GSPO | ⚠️ not a headline feature | ✅ **best ergonomics** — MCP-RL, LangGraph, RULER auto-rewards | vLLM (+Unsloth kernels) | **LoRA-only** | single-node → serverless | — |
| **LlamaFactory** | PPO, DPO, KTO, ORPO, SimPO. **No GRPO** | ✗ | ✗ | vLLM, SGLang (inference only) | FSDP2, DeepSpeed, Megatron | multi-node | ✅ FP8 SFT |

### Unsloth pivoted to a desktop app — and multi-GPU still is not officially shipped

The README's first line is now *"Unsloth is the first desktop app to run and train models."* [Unsloth Desktop](https://unsloth.ai/docs/new/changelog) shipped **2026-08-11**; there is a `studio/backend/` tree with an MCP client, tool-call loops and RAG, and `unsloth start claude|codex|...` wires local models into agent harnesses. **2,695 commits in 90 days** went here. Three products now exist: Desktop (app), Studio (web UI), and **Unsloth Core** (the original pip library — GRPO/DPO via TRL patching is intact).

**Two things to flag:**
- **Multi-GPU is contradictory.** The README claims *"We support Multi GPU setups"*, but the [multi-GPU docs page](https://unsloth.ai/docs/basics/multi-gpu-training-with-unsloth) still says they'll be *"announcing official multi-GPU support for Unsloth soon."* Today you get it only through manual `accelerate`/DeepSpeed setup. **Do not plan around first-class multi-GPU.**
- **Apple Silicon is inference-only.** MLX support is for running models, not training. **UNVERIFIED-negative** — no evidence of MLX training in Unsloth.

### Algorithms: what actually won in 2026

**Nothing replaced GRPO — GRPO is still the substrate.** Two orthogonal layers got added on top:

1. **Sequence-level importance sampling**, for MoE stability. **GSPO** ([arXiv 2507.18071](https://arxiv.org/abs/2507.18071), Qwen) computes the IS ratio from *sequence* likelihood with length normalization and clips at the sequence level, fixing GRPO's pathology where high-variance token-level IS weights compound with response length and get amplified by clipping into collapse. It powers Qwen3 and is implemented in **verl, TRL, NeMo-RL, slime, AReaL, ROLL, OpenRLHF, SkyRL, ART, Unsloth**.
2. **Train/inference mismatch correction** — now an *infrastructure* concern more than an algorithm tweak: **TIS** (truncated importance sampling), **IcePop**/**KPop** (token filtering), **seq-mask-TIS**, **router replay (R3)** for MoE, and **batch-invariant kernels**.

**DAPO** ([arXiv 2503.14476](https://arxiv.org/abs/2503.14476), ByteDance Seed — Clip-Higher, Dynamic Sampling, Token-Level PG Loss, Overlong Reward Shaping) simply **won**: it is now TRL's *default* `loss_type` and ships in verl/NeMo-RL/SkyRL/AReaL/OpenRLHF/slime/TitanRL.

Other 2026 algorithms worth knowing: **CISPO** (MiniMax-M1, [arXiv 2506.13585](https://arxiv.org/abs/2506.13585) — soft token-level clip where clipped tokens still contribute gradient; in TRL/NeMo-RL/SkyRL/slime/ROLL), **LitePPO** ([arXiv 2508.08221](https://arxiv.org/abs/2508.08221)), **TOPR**, **SAPO** ([arXiv 2511.20347](https://arxiv.org/abs/2511.20347)), **LUSPO** ([arXiv 2602.05261](https://arxiv.org/abs/2602.05261)), **VESPO** ([arXiv 2602.10693](https://arxiv.org/abs/2602.10693)), **GDPO** ([arXiv 2601.05242](https://arxiv.org/abs/2601.05242)), **MOPD** (on-policy distillation).

**Best practical doc found:** SkyRL's [off-policy correction guide](https://github.com/NovaSky-AI/SkyRL/blob/main/docs/content/docs/algorithms/off_policy_correction.mdx) — recommends geometric sequence masking first (`geo_mask_low=0.99, geo_mask_high=1.01`), then router replay for MoE, *then* TIS.

**The field's convergent finding**, from [HuggingFace's "Keep the Tokens Flowing: Lessons from 16 Open-Source RL Libraries"](https://huggingface.co/blog/async-rl-training-landscape) (2026-03-10): *"Every library in this survey has independently converged on the same architectural principle: physically separate inference GPUs from training GPUs, and push weights asynchronously."* Ray orchestrates 8 of 16. Its sharpest claim — that no open-source async RL library implements DeepSeek's "Keep Routing" MoE fix — is **now partly stale**, since SkyRL has shipped `moe_enable_routing_replay`. (The frequently-cited [Anyscale comparison](https://www.anyscale.com/blog/open-source-rl-libraries-for-llms) is 2025-vintage — ignore its star counts and its "SkyRL not mature yet" line.)

### 2026 newcomers most lists are missing

| Repo | ★ | License | Last push | What it is |
|---|---|---|---|---|
| [radixark/miles](https://github.com/radixark/miles) | 2,825 | — | 2026-09-13 | *"Enterprise-facing RL framework for LLM and VLM post-training, forked from and co-evolving with slime."* **Bigger than SkyRL and NeMo-RL by stars and absent from every standard list.** |
| [google/tunix](https://github.com/google/tunix) | 2,443 | — | — | JAX/TPU post-training. The only serious non-PyTorch option. |
| [vllm-project/vime](https://github.com/vllm-project/vime) | 459 | Apache-2.0 | 2026-09-12 | **vLLM's own official post-training framework** — slime's training stack and data-generation design with vLLM + vllm-router as rollout instead of SGLang. [Announced 2026-06-09](https://vllm.ai/blog/2026-06-09-announcing-vime). Ships coding-agent RL with Claude Code/Codex + E2B sandboxes. |
| [ServiceNow/PipelineRL](https://github.com/ServiceNow/PipelineRL) | 432 | — | — | Architecturally unique: swaps weights **between token decode steps** rather than pausing generation. HF's survey calls it *"fundamentally diverges from every other library."* |
| [inclusionAI/AReno](https://github.com/inclusionAI/AReno) | 315 | Apache-2.0 | 2026-09-10 | *"ASystem Reinforcement Learning Nano"* — self-contained **single-node** RL/SFT/DPO/serving/agentic-RL with no external training or inference backend. Notably carries an **MLX / Apple Silicon badge**. |
| [XYZ-AI-Lab/axrl](https://github.com/XYZ-AI-Lab/axrl) · [redai-studio/Relax](https://github.com/redai-studio/Relax) · [opendilab/LightRFT](https://github.com/opendilab/LightRFT) · [RL-Align/RL-Kernel](https://github.com/RL-Align/RL-Kernel) · [Human-Agent-Society/reef](https://github.com/Human-Agent-Society/reef) | 979 · 603 · 405 · 296 · 1,166 | — | 2026 | Watch-list. `axrl` already stalling (last push 2026-08-03); `RL-Kernel` targets bitwise train-inference consistency; `reef` (created 2026-08-31) does continual learning for self-improving agents. |

### Verdict — which to pick

**Tier 1, pick one:**
1. **verl** — the default. Widest algorithm coverage, every backend combination (FSDP2/Megatron/TorchTitan × vLLM/SGLang/TRT-LLM), NVIDIA/AMD/Ascend, and now a whole ecosystem (`verl-recipe`, `uni-agent`, `verl-omni`, `verl-vla`). Choose it unless you have a specific reason not to.
2. **slime** — the frontier-proven one: *"the RL framework behind GLM-5.3, GLM-5.2, GLM-5.1, GLM-5, GLM-4.7, GLM-4.6, and GLM-4.5."* Deliberately SGLang-only + Megatron-only so it can exploit backend-specific features instead of a lowest-common-denominator abstraction. Its influence is measurable — two major forks (`miles`, `vime`).
3. **prime-rl** — smallest readable codebase that does real 1000-GPU async work, with the cleanest environment story (native `verifiers` + Environments Hub). Trained INTELLECT-3.

**For agents specifically:** SkyRL (`skyrl-agent`/`skyrl-gym`, Harbor terminal agents, and it implements the **Tinker API** so Tinker scripts run on your own GPUs) and **AReaL v2** ([arXiv 2607.01120](https://arxiv.org/abs/2607.01120)) — whose `base_url`-swap design lets you RL-train *any* black-box agent harness.

**For small scale / ergonomics:** **TRL** (no longer a toy — `AsyncGRPOTrainer` + OpenEnv is a real agentic path, and it has the highest commit velocity of anything here) and **ART** (MCP-RL, RULER auto-rewards, LangGraph).

**Do not touch:** torchtune, torchforge, oat, RL2, atropos.



## Stage 5 — RL environments and verifiers

This is the stage that changed most in 2026. Two competing-but-overlapping standards emerged, and the "environment" became a distributable artifact with its own hubs.

### The two standards

| Repo | ★ | License | Last push | What it does | Hardware | Verdict |
|---|---|---|---|---|---|---|
| [PrimeIntellect-ai/verifiers](https://github.com/PrimeIntellect-ai/verifiers) | 4,610 | MIT | 2026-09-13 | `Environment` / `Rubric` API for building RL envs **and** evals; 19 envs in-repo; tightly bound to the [Environments Hub](https://app.primeintellect.ai/dashboard/environments), [prime-rl](https://github.com/PrimeIntellect-ai/prime-rl), and Prime's hosted training | any (env logic is CPU; rollouts hit a vLLM/SGLang server) | **The de-facto standard for RL-as-verifiable-reward.** Originally by Will Brown. Install via the `prime` CLI. If you are doing RLVR in 2026 and have no strong opinion, write a `verifiers` Environment. |
| [huggingface/OpenEnv](https://github.com/huggingface/OpenEnv) | 2,574 | BSD-3-Clause | 2026-09-12 | Gymnasium-style `reset()` / `step(action)` / `state()` over WebSocket-on-HTTP; environments ship as **Docker containers deployed to HF Spaces** | any | **The vendor-neutral spec.** Governed by a technical committee spanning Meta-PyTorch, Reflection, Unsloth, Modal and NVIDIA. `meta-pytorch/OpenEnv` now **redirects to `huggingface/OpenEnv`** — stewardship sits with HF, governance is shared. Integrations confirmed: **torchforge, TRL, SkyRL, ART, Oumi, Lightning AI.** |

**How to choose:** they are complementary more than rival. `verifiers` is opinionated about *rewards* (rubrics, graders) and is the native currency of the Prime ecosystem; `OpenEnv` is opinionated about *transport and packaging* (containerized, hub-distributed, Gym-shaped) and is the native currency of the PyTorch/HF ecosystem. If your trainer is prime-rl → verifiers. If it's torchforge or TRL → OpenEnv. Writing an env against both is not hard and is the safe hedge.

Also: [meta-pytorch/OpenEnvZoo](https://github.com/meta-pytorch/OpenEnvZoo) (4★, 2026-01-30) — a versioned collection of OpenEnv environments. Tiny; **UNVERIFIED** whether it is actively curated.

### Task / reasoning environments

| Repo | ★ | License | Last push | What it does | Verdict |
|---|---|---|---|---|---|
| [open-thought/reasoning-gym](https://github.com/open-thought/reasoning-gym) | 1,505 | Apache-2.0 | 2026-04-17 | **~108 procedural task generators** with verifiable answers across `algorithmic` (35), `arithmetic` (19), `games` (19), `logic` (9), `cognition` (9), `graphs` (7), `code` (4), `induction` (3), `geometry` (3) plus algebra/arc/probability | NeurIPS 2025 Spotlight. **Stale (~5 mo) but arguably "done"** — procedural generators don't rot. Infinite difficulty-controlled RLVR data for free, no sandbox needed. **Best cost/benefit environment source on this page.** |
| [NVIDIA-NeMo/Gym](https://github.com/NVIDIA-NeMo/Gym) | 1,183 | Apache-2.0 | 2026-09-13 | "Evaluate and improve models and agents using environments" — NVIDIA's environment layer | **Hot.** New in the NeMo breakup. UNVERIFIED how it interoperates with verifiers/OpenEnv. |
| [NVIDIA-NeMo/ProRL-Agent-Server](https://github.com/NVIDIA-NeMo/ProRL-Agent-Server) | 838 | Apache-2.0 | 2026-08-13 | "Agentic RL on Any Harness at Scale" — decouples the agent harness from the RL trainer | **Active.** The right *shape* of idea: train against whatever harness you already evaluate with. |

### SWE / agentic coding environments

| Repo | ★ | License | Last push | What it does | Verdict |
|---|---|---|---|---|---|
| [SWE-bench/SWE-smith](https://github.com/SWE-bench/SWE-smith) | 765 | MIT | 2026-09-07 | **Synthesizes** SWE training tasks at scale (NeurIPS 2025 D&B Spotlight) | **The live one.** Only actively-maintained SWE task-generation pipeline here. If you want SWE RL data in 2026, this is the source. |
| [SWE-Gym/SWE-Gym](https://github.com/SWE-Gym/SWE-Gym) | 736 | Apache-2.0 | 2025-07-29 | Training envs + verifiers for SWE agents (ICML 2025) | **Dead (13 mo).** Historically important, superseded by SWE-smith. |
| [R2E-Gym/R2E-Gym](https://github.com/R2E-Gym/R2E-Gym) | 333 | Apache-2.0 | 2025-07-13 | Procedural SWE env generation + hybrid verifiers (COLM 2025) | **Dead (14 mo).** Note: `agentica-project/R2E-Gym` (23★) is a stub — the real repo is `R2E-Gym/R2E-Gym`. |
| [SWE-rebench/SWE-rebench-V2](https://github.com/SWE-rebench/SWE-rebench-V2) | 83 | MIT | 2026-03-12 | Tools + prompt templates for building/evaluating continuously-refreshed SWE tasks (Nebius) | **Niche but the right idea** — contamination-resistant, continuously updated task sets. Low adoption. |

### Verifiers (graders) and sandboxes

| Repo | ★ | License | Last push | What it does | Verdict |
|---|---|---|---|---|---|
| [huggingface/Math-Verify](https://github.com/huggingface/Math-Verify) | 1,189 | Apache-2.0 | 2026-01-10 | Robust math-answer equivalence checking (LaTeX/sympy normalization) | **Stale (8 mo) but still the default** — it's the grader `lighteval` and many RLVR loops call. Math grading is a stable problem; staleness here is low-risk. |
| [NVIDIA-NeMo/Skills](https://github.com/NVIDIA-NeMo/Skills) | 1,036 | Apache-2.0 | 2026-09-13 | `nemo-skills` — pipelines for math/code reasoning incl. graders and data generation | **Hot.** The actively-maintained alternative to Math-Verify if you want a whole math-RL pipeline rather than one function. |
| [e2b-dev/E2B](https://github.com/e2b-dev/E2B) | 13,771 | Apache-2.0 | 2026-09-12 | Firecracker-based secure code sandboxes; SDK-first | **The most-adopted hosted sandbox.** Self-hostable but non-trivial to run; most people pay. |
| [modal-labs/modal-client](https://github.com/modal-labs/modal-client) | 514 | Apache-2.0 | 2026-09-11 | Modal Sandboxes — serverless containers, per-second billing | **The pragmatic choice for RL rollouts.** Scales to thousands of parallel envs; Harbor lists Modal as a first-class execution provider. Not self-hostable. |

**Honest gap:** there is still **no great fully-self-hostable open-source sandbox** with the ergonomics of E2B/Modal. DIY `bubblewrap`/`nsjail`/gVisor works and is what cost-sensitive labs actually do, but you build the orchestration yourself. **UNVERIFIED** for microsandbox/Daytona/Cloudflare sandboxes — not confirmed in real RL training loops.



## Stage 6 — Distillation and model merging

| Repo | ★ | License | Last push | What it does | Scale | Hardware | Verdict |
|---|---|---|---|---|---|---|---|
| [huggingface/trl](https://github.com/huggingface/trl) — `DistillationTrainer` | 19,294 | Apache-2.0 | 2026-09-13 | **On-policy / logit distillation is now a first-class stable trainer** (`trl/trainer/distillation_trainer.py`, `distillation_config.py`) | 1 GPU → multi-node | CUDA | **This is the 2026 answer for distillation.** HF report ~40× speedup distilling 100B+ models with it. |
| TRL experimental distillation family | — | Apache-2.0 | 2026-09-13 | [`gkd`](https://github.com/huggingface/trl/tree/main/trl/experimental/gkd), [`gold`](https://github.com/huggingface/trl/tree/main/trl/experimental/gold) (cross-tokenizer OPD), [`minillm`](https://github.com/huggingface/trl/tree/main/trl/experimental/minillm), [`async_distillation`](https://github.com/huggingface/trl/tree/main/trl/experimental/async_distillation), [`server_distillation`](https://github.com/huggingface/trl/tree/main/trl/experimental/server_distillation), `sdft`, `ssd` | — | CUDA | An unusually deep bench. **GOLD** matters most: it does on-policy distillation *across different tokenizers*, which unblocks teacher/student pairs from different families. |
| [chrisliu298/awesome-on-policy-distillation](https://github.com/chrisliu298/awesome-on-policy-distillation) | 814 | CC0-1.0 | 2026-08-30 | Curated OPD papers/frameworks/tools | — | — | The map of a field that exploded in 2026. Start here before picking a method. |
| [arcee-ai/mergekit](https://github.com/arcee-ai/mergekit) | 7,348 | **LGPL-3.0** | 2026-09-12 | SLERP, TIES, DARE, task arithmetic, passthrough/frankenmerge, MoE-ification | CPU-mergeable (low RAM mode) | **CPU works — runs fine on Apple Silicon** | **Uncontested standard, still actively maintained.** Note the **LGPL-3.0** license — more restrictive than the Apache/MIT norm here; check it before vendoring. Merging is the one heavy-sounding stage that genuinely runs on a laptop. |
| [tommasomncttn/mergenetic](https://github.com/tommasomncttn/mergenetic) | 107 | Apache-2.0 | 2025-08-08 | Evolutionary merge search (ACL 2025 demo) | — | — | **Stale.** Interesting idea (evolutionary merge recipes); not maintained. UNVERIFIED whether anyone uses it in production. |

**Key 2026 context:** Thinking Machines' [On-Policy Distillation](https://thinkingmachines.ai/blog/on-policy-distillation/) post is the practitioner reference — they trained math reasoning into Qwen3-8B-Base from a Qwen3-32B teacher and matched an RL baseline at a fraction of the compute. That result is *why* distillation moved from "nice compression trick" to "a cheaper substitute for RL," and why TRL promoted it to a stable trainer. If your RL budget is small, try OPD before you try GRPO.

---

## Stage 7 — Evaluation harnesses

### General-purpose harnesses

| Repo | ★ | License | Last push | Coverage | Backends | Alive? | Verdict |
|---|---|---|---|---|---|---|---|
| [UKGovernmentBEIS/inspect_ai](https://github.com/UKGovernmentBEIS/inspect_ai) | 2,756 | MIT | 2026-09-13 | Framework, not a task list | Any provider + local | **Hot** | **The 2026 winner for anything agentic.** Built by the UK AI Security Institute. First-class tool use, sandboxing, multi-turn, solvers/scorers, and a proper log viewer. |
| [UKGovernmentBEIS/inspect_evals](https://github.com/UKGovernmentBEIS/inspect_evals) | 669 | MIT | 2026-09-13 | **137 evals** (verified by counting `src/inspect_evals/`) | via inspect_ai | **Hot** | The companion eval library. 137 curated, maintained evals spanning coding, agentic, reasoning, safety, cyber. |
| [EleutherAI/lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) | 13,961 | MIT | 2026-09-10 | **227 task groups** (counting `lm_eval/tasks/`) | HF, vLLM, SGLang, APIs | **Hot** (v0.4.13, 2026-08-31) | **Still the standard for static multiple-choice/log-likelihood benchmarks** and the thing reviewers expect to see. Weak at agentic/tool-use evals. Widest raw task count. |
| [open-compass/opencompass](https://github.com/open-compass/opencompass) | 7,423 | Apache-2.0 | 2026-09-10 | 100+ datasets | many | **Hot** | Most-starred after lm-eval. Strongest Chinese-model and Chinese-benchmark coverage. Heavier config system. |
| [stanford-crfm/helm](https://github.com/stanford-crfm/helm) | 2,907 | Apache-2.0 | 2026-09-01 | Holistic scenarios + multimodal | many | **Active** | Best for *multi-metric* holistic reporting (accuracy + calibration + robustness + bias + efficiency) rather than a single leaderboard number. Heavyweight. |
| [huggingface/lighteval](https://github.com/huggingface/lighteval) | 2,539 | MIT | 2026-09-09 | broad | transformers, vLLM, SGLang, nanotron, APIs | **Active but cooling** — last release **v0.13.0, 2025-11-24** (~10 months) | Commits are recent but releases have stalled. It is what HF's own model releases (SmolLM3) were evaluated with. Fine choice, just not gaining. |
| [mlfoundations/evalchemy](https://github.com/mlfoundations/evalchemy) | 611 | **no license** | 2026-02-24 | post-training benchmarks | builds on lm-eval | **Stale (~7 mo)** | Useful idea (unified post-training eval) but stale **and carries no license file**. Avoid for anything you ship. |
| [NVIDIA-NeMo/Evaluator](https://github.com/NVIDIA-NeMo/Evaluator) | 337 | Apache-2.0 | 2026-09-08 | — | — | **Hot** | New in the NeMo breakup. Low adoption so far; watch it. |

### Agentic / benchmark-specific harnesses

| Repo | ★ | License | Last push | What it does | Verdict |
|---|---|---|---|---|---|
| [harbor-framework/harbor](https://github.com/harbor-framework/harbor) | 5,175 | Apache-2.0 | 2026-09-12 | **The successor to the Terminal-Bench harness.** Evaluates arbitrary agents (Claude Code, Codex CLI, …), lets you build/share benchmarks, runs thousands of parallel envs via Daytona/Modal/LangSmith, **and generates rollouts for RL** | **Most important new eval infrastructure of 2026.** It is the official harness for Terminal-Bench 2.0 and it deliberately blurs eval and RL — the same container is your benchmark and your training environment. TRL even ships a [`harbor` experimental module](https://github.com/huggingface/trl/tree/main/trl/experimental/harbor). |
| [harbor-framework/terminal-bench-1](https://github.com/harbor-framework/terminal-bench-1) | 2,576 | Apache-2.0 | 2026-07-11 | The original Terminal-Bench (this is where `laude-institute/terminal-bench` now redirects) | Frozen as v1. |
| [harbor-framework/terminal-bench](https://github.com/harbor-framework/terminal-bench) | 677 | Apache-2.0 | 2026-09-11 | *"Measuring and evolving with the frontier of agent work"* | The live line. |
| [harbor-framework/terminal-bench-2](https://github.com/harbor-framework/terminal-bench-2) · [-2-1](https://github.com/harbor-framework/terminal-bench-2-1) | 407 · 114 | Apache-2.0 | 2026-04-30 · 2026-08-26 | TB 2.0 and 2.1 task sets | 2.1 is current. |
| [harbor-framework/terminal-bench-science](https://github.com/harbor-framework/terminal-bench-science) | 574 | Apache-2.0 | 2026-09-13 | Agents on real scientific research workflows | New domain expansion; very active. |
| [SWE-bench/SWE-bench](https://github.com/SWE-bench/SWE-bench) | 5,830 | MIT | 2026-09-02 | The canonical coding benchmark + containerized harness | **Still the benchmark that matters for coding.** Active. |
| [SWE-agent/mini-swe-agent](https://github.com/SWE-agent/mini-swe-agent) | 7,474 | MIT | 2026-09-07 | *"The 100 line AI agent… scores >74% on SWE-bench verified"* | **The best setup-to-signal ratio on this entire page.** 100 lines, no giant config, >74% on SWE-bench Verified. Use this as your coding-agent scaffold and baseline. |
| [SWE-agent/SWE-agent](https://github.com/SWE-agent/SWE-agent) | 20,311 | MIT | 2026-09-07 | The full-featured agent (NeurIPS 2024) | Active. Use mini- unless you need the configurability. |
| [sierra-research/tau2-bench](https://github.com/sierra-research/tau2-bench) | 2,008 | MIT | 2026-09-11 | Tool-agent-user interaction in real domains | **Active.** The standard for *conversational* tool-use eval — complements Terminal-Bench (solo agent) and SWE-bench (code). **UNVERIFIED** whether a tau3 exists. |
| [centerforaisafety/hle](https://github.com/centerforaisafety/hle) | 1,682 | MIT | 2026-08-01 | Humanity's Last Exam | Active. **UNVERIFIED** current SOTA score. |
| [arcprize/ARC-AGI-3-Agents](https://github.com/arcprize/ARC-AGI-3-Agents) | 322 | MIT | 2026-08-03 | **ARC-AGI-3 exists** — interactive/game-based | The live ARC line. [fchollet/ARC-AGI](https://github.com/fchollet/ARC-AGI) (4,827★, 2025-04) and [arcprize/ARC-AGI-2](https://github.com/arcprize/ARC-AGI-2) (738★, 2025-05) are both frozen datasets. Notable: [alexisfox7/PRO-LONG](https://github.com/alexisfox7/PRO-LONG) (452★) claims **97.4% on ARC-AGI-3** via programmatic memory ([arXiv 2607.20064](https://arxiv.org/abs/2607.20064)) — **UNVERIFIED** independently. |
| [LiveCodeBench/LiveCodeBench](https://github.com/LiveCodeBench/LiveCodeBench) | 941 | MIT | **2025-07-16** | Contamination-free competitive coding | **Stale (14 mo).** The *dataset* still refreshes via releases, but the repo is quiet. **UNVERIFIED** whether LiveCodeBench Pro / v6 exist. Access it through lm-eval or inspect_evals rather than directly. |

### Which harness covers the most 2026 benchmarks with the least setup?

1. **`inspect_ai` + `inspect_evals`** — 137 maintained evals, one install, native tool-use and Docker sandboxing, both repos pushed *today*. It is the only option that handles static *and* agentic evals without a second framework. **This is the default recommendation.**
2. **`lm-evaluation-harness`** — 227 task groups and the one reviewers recognize; near-zero setup for static benchmarks; vLLM/SGLang backends. Use it *alongside* Inspect for reported numbers, not instead of it.
3. **`harbor`** — if terminal/agentic work is your focus, nothing else comes close, and it doubles as your RL rollout generator.
4. Everything else is situational: OpenCompass for Chinese benchmarks, HELM for multi-metric reports, lighteval if you're already in HF's pipeline.

**Practical stack: Inspect for agentic + lm-eval for static + Harbor for terminal/RL rollouts.** Three tools, full 2026 coverage.

## Stage 8 — Inference / serving for eval

| Repo | ★ | License | Last push | Hardware | Apple Silicon | Verdict |
|---|---|---|---|---|---|---|
| [vllm-project/vllm](https://github.com/vllm-project/vllm) | 91,604 | Apache-2.0 | 2026-09-13 | CUDA, ROCm, TPU, CPU, Gaudi, Ascend | ⚠️ **experimental, CPU-only, build from source** — macOS Sonoma+, Xcode 15.4+, Python 3.10–3.13, **FP32/FP16 only**; community `vllm-metal` plugin uses MLX for GPU ([docs](https://docs.vllm.ai/en/latest/getting_started/installation/cpu.html)) | **The default rollout/eval engine on rented GPUs.** Every serious RL framework here (prime-rl, verl, TRL, OpenRLHF, torchtitan's TitanRL) generates through it. On a Mac it's a smoke test, not a tool. |
| [sgl-project/sglang](https://github.com/sgl-project/sglang) | 35,877 | Apache-2.0 | 2026-09-13 | NVIDIA (GB200/B300/H100/A100/Spark/5090), AMD (MI355/MI300), Intel Xeon, Google TPU, Ascend NPU | ✅ **Real native MLX backend** — [`python/sglang/srt/hardware_backend/mlx/`](https://github.com/sgl-project/sglang/tree/main/python/sglang/srt/hardware_backend/mlx) with `model_runner`, `scheduler_mixin`, `tp_worker`, `kv_cache/`, `sampling`, plus [`layers/quantization/mlx.py`](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/layers/quantization/mlx.py) | **The biggest surprise in this survey.** SGLang is the *only* major serving framework with a first-class Apple Silicon path. Broadest hardware support overall. Early and evolving on Mac. |
| [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp) | 128,042 | MIT | 2026-09-13 | Everything, incl. Metal | ✅ **excellent** — Metal is a first-class backend | **The portability champion for inference.** Training exists but is vestigial: one file, [`examples/training/finetune.cpp`](https://github.com/ggml-org/llama.cpp/tree/main/examples/training). Do not train here. |
| [ml-explore/mlx-lm](https://github.com/ml-explore/mlx-lm) | 6,993 | MIT | 2026-09-12 | Apple Silicon only | ✅ **native** — `mlx_lm.server` | **The Mac-local serving answer**, and the only one that also trains. |
| [ollama/ollama](https://github.com/ollama/ollama) | 180,770 | MIT | 2026-09-11 | Everything, incl. Metal | ✅ excellent | Most-starred project in this entire survey. Great for *using* models; wrong layer for eval — you want the raw engine's batching and logprobs. |

**Practical note for eval:** pick your serving engine to match your trainer, not your taste — colocated or server-mode generation is where RL throughput is won or lost. On rented NVIDIA that means **vLLM** (widest RL-framework integration) or **SGLang** (better multi-hardware story). On a Mac it means **mlx-lm**, with **SGLang-MLX** now a real second option.



## Stage 9 — Apple Silicon specifically

### Core MLX

| Repo | ★ | License | Last push | What it does | Verdict |
|---|---|---|---|---|---|
| [ml-explore/mlx](https://github.com/ml-explore/mlx) | 28,399 | MIT | 2026-09-12 | Apple's array framework; unified-memory native; v0.32.2 (2026-08-25) | **First-party, excellent, and the only serious training path on Apple Silicon.** |
| [ml-explore/mlx-lm](https://github.com/ml-explore/mlx-lm) | 6,993 | MIT | 2026-09-12 | Run **and train** LLMs on MLX. Split out of mlx-examples | **The base layer.** See feature table below. |
| [ml-explore/mlx-examples](https://github.com/ml-explore/mlx-examples) | 8,943 | MIT | 2026-04-06 | Examples; `llms/` is now **just a move notice** pointing to mlx-lm | **Partly deprecated.** The one still-valuable piece is [`transformer_lm/`](https://github.com/ml-explore/mlx-examples/tree/main/transformer_lm) — Apple's canonical minimal train-a-decoder-LM-from-scratch example. |

### What mlx-lm actually exposes for training

Directory `mlx_lm/tuner/` contains exactly: `callbacks.py, datasets.py, dora.py, lora.py, losses.py, trainer.py, utils.py`. Note what is **not** there — no `dpo.py`, `orpo.py`, `grpo.py`.

| Feature | mlx-lm | Evidence |
|---|---|---|
| Full fine-tune | ✅ | `--fine-tune-type full` ([lora.py](https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/lora.py)) |
| LoRA (default) | ✅ | [LORA.md](https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/LORA.md) |
| DoRA | ✅ | [tuner/dora.py](https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/tuner/dora.py) |
| QLoRA | ✅ implicit | LORA.md: *"If `--model` points to a quantized model, then the training will use QLoRA"* |
| **DPO / ORPO / CPO / GRPO** | ❌ **NONE** | Code search for each returns 0 real hits |
| SFT trainer | ⚠️ functionally yes | `train()`, `TrainingArgs`, `ChatDataset`/`CompletionsDataset`, `--mask-prompt` — but no `SFTTrainer` class |
| Quantized / learned quantization | ✅ | [`mlx_lm/quant/`](https://github.com/ml-explore/mlx-lm/tree/main/mlx_lm/quant): AWQ, **DWQ**, GPTQ, dynamic. DWQ gradient-tunes quantization scales against an unquantized teacher |
| Optimizers | ✅ incl. **Muon** | adam, adamw, muon, sgd, adafactor |
| Distributed data-parallel | ✅ built in | `trainer.py` imports `average_gradients`; uses `mx.distributed` |

**This is a deliberate scope decision, not an oversight.** PR [mlx-lm#8 (GRPO)](https://github.com/ml-explore/mlx-lm/pull/8) was closed unmerged; PRs #7, #77, #417, #794, #796 (DPO/ORPO variants) likewise. Awni Hannun's own comment (2025-10-08) routes users to a community package. [Issue #1420 "Add GRPO trainer"](https://github.com/ml-explore/mlx-lm/issues/1420) has sat open since 2026-06-21 with zero comments.

### The RL / preference layer on MLX

| Repo | ★ | License | Last push | What it does | Verdict |
|---|---|---|---|---|---|
| [Goekdeniz-Guelmez/mlx-lm-lora](https://github.com/Goekdeniz-Guelmez/mlx-lm-lora) | 412 | Apache-2.0 | 2026-09-06 | **SFT, DPO, CPO, ORPO, GRPO, GSPO, Dr. GRPO, DAPO, Online DPO, XPO, PPO** + LoRA/DoRA/full/QLoRA (4/6/8-bit) + **QAT** | **The answer, and Apple points you to it.** Only MLX RL package with an OSI license, Apple's endorsement, and commits this month. |
| [ARahim3/mlx-tune](https://github.com/ARahim3/mlx-tune) | 1,405 | Apache-2.0 | 2026-06-23 | Formerly `unsloth-mlx`. **Unsloth-compatible API** (`from mlx_tune import FastLanguageModel, SFTTrainer`); SFT/DPO/GRPO/Vision/TTS/Embedding | **Biggest MLX training project by stars.** Explicitly designed to prototype on Mac then scale on CUDA Unsloth — a genuinely useful workflow. |
| [Doriandarko/MLX-GRPO](https://github.com/Doriandarko/MLX-GRPO) | 242 | **none** | 2025-10-28 | Dedicated MLX GRPO | **Skip.** 11 months stale *and* no license = legally unusable. |
| [Blaizzy/mlx-vlm](https://github.com/Blaizzy/mlx-vlm) | 5,492 | MIT | 2026-09-13 | VLM inference + fine-tuning | **Hot.** If you go multimodal on Mac, this is it. |

### MLX pretraining-from-scratch projects

| Repo | ★ | License | Last push | Verdict |
|---|---|---|---|---|
| [mlx-examples `transformer_lm/`](https://github.com/ml-explore/mlx-examples/tree/main/transformer_lm) | 8,943 (repo) | MIT | 2026-04-06 | Apple's reference from-scratch decoder LM. Tiny scale, authoritative, correct. |
| [scasella/nanochat-mlx](https://github.com/scasella/nanochat-mlx) | 70 | MIT | 2026-04-29 | **Best nanochat MLX port and the most practical pick for a 16 GB Mac.** Single `--depth` dial; publishes a measured Mac sizing table. |
| [N8python/mlx-pretrain](https://github.com/N8python/mlx-pretrain) | 84 | CC0-1.0 | **2025-08-20** | Cleanest YAML-config pretrain loop (tokenizer train + Llama-arch pretrain + mlx-lm export). **Stale ~13 months.** |
| [Greninja9257/LabLLM](https://github.com/Greninja9257/LabLLM) | 75 | MIT | 2026-08-20 | Native macOS GUI lab for from-scratch training. Created 2026-08-15 — newest and most active, but ~1 month old and unproven. |
| [vithursant/nanoGPT_mlx](https://github.com/vithursant/nanoGPT_mlx) | 122 | MIT | 2024-02-12 | **Dead (19 mo).** Predates most MLX API evolution. |

**nanochat on Apple Silicon:** there is **no official MLX path and no merged MPS path** upstream. 37 issues mention MPS; open ones include [#145](https://github.com/karpathy/nanochat/issues/145) (MPS/CPU compat), [#792](https://github.com/karpathy/nanochat/issues/792) (MPS-aware device sync), [#838](https://github.com/karpathy/nanochat/issues/838)/[#839](https://github.com/karpathy/nanochat/issues/839) (peak memory reports 0.00 MiB on MPS). bf16 on MPS *does* work (~25% memory saving, [#793](https://github.com/karpathy/nanochat/issues/793)). Use `scasella/nanochat-mlx` rather than patching the original.

### MLX distributed — JACCL over Thunderbolt is the 2026 story

MLX ships four backends in `mlx/distributed/`: **MPI**, **RING** (TCP sockets, always available, usually faster than MPI), **JACCL** (*"Low latency communication with RDMA over thunderbolt. Necessary for things like tensor parallelism"*), **NCCL** (CUDA). ([docs](https://ml-explore.github.io/mlx/build/html/usage/distributed.html))

JACCL requires **macOS 26.2+** for RDMA over Thunderbolt 5, per [WWDC 2026 Session 233](https://developer.apple.com/videos/play/wwdc2026/233/). Topologies: full mesh (`--backend jaccl`) or ring (`--backend jaccl-ring`). Tooling: `mlx.launch --hosts ...`, `mlx.distributed_config --auto-setup`.

Apple's measured numbers (WWDC26 §233):

| Workload | 1× M3 Ultra | 4× M3 Ultra mesh |
|---|---|---|
| **Fine-tune Qwen 3.5 9B** | ~180 tok/s | ~600 tok/s (*">3× speedup for fine-tuning"*) |
| Inference Qwen 3.6 27B | baseline | ~3× |
| Inference Kimi 2.6 (1T, 8-bit ≈ 1 TB) | doesn't fit | runs across four Macs |

**The critical caveat: distributed MLX *training* is DATA-PARALLEL ONLY.** Apple describes it as *"replicates models across machines and averages gradients."* Tensor and pipeline parallelism are presented for **inference**. So a Mac cluster buys throughput, **not a bigger trainable model** — every node must still hold full weights + optimizer state. Practical ceiling ~4–6 nodes for full mesh (cables grow quadratically). No official max node count published — **UNVERIFIED**. JACCL is not yet a standalone repo (`apple/jaccl` and `ml-explore/jaccl` both 404).

### PyTorch MPS backend for training, 2026

Current stable is **PyTorch 2.14.0 (2026-09-02)**. There is no PyTorch 3.0. Open `module: mps` issues: **227**. The live tracker is [#141287 "MPS operator coverage tracking issue (2.6+ version)"](https://github.com/pytorch/pytorch/issues/141287) (updated 2026-09-12), successor to the famous [#77764](https://github.com/pytorch/pytorch/issues/77764) (1,796 comments).

| Capability | Status | Evidence |
|---|---|---|
| `torch.compile` on MPS | ✅ **works** — Inductor emits Metal shaders; 2× geomean speedup over 30+ networks with no known compiler failures | [#150121](https://github.com/pytorch/pytorch/issues/150121) |
| bf16 | ✅ mature | 2.14 continues extending bf16 paths |
| **FlexAttention** | ✅ **new in 2.13** — *"up to ~12× speedup over SDPA on sparse patterns"* | PyTorch 2.13 release notes |
| SDPA / Flash Attention | ⚠️ **forward fast, backward NOT** — *"we don't have a dedicated MPS impl of the backward pass… not only is the backward pass slower than it could be, but the forward pass is also slower"* | [#179294](https://github.com/pytorch/pytorch/issues/179294), open 2026-08-23 |
| `torch.distributed` | ❌ **NO** — *"Distributed setups `gloo` and `nccl` are not working with `mps` device… only single GPU of `mps` device type can be used"* | [HF Accelerate docs](https://huggingface.co/docs/accelerate/en/usage_guides/mps); [#160732](https://github.com/pytorch/pytorch/issues/160732) open since 2025-08 |
| FSDP / FSDP2 | ❌ **NO** — requires a distributed backend that doesn't exist | — |
| Multi-machine Mac | ❌ **NO JACCL equivalent** | UNVERIFIED: no PyTorch RFC found |
| float8 / fp8 | ❌ | [#132624](https://github.com/pytorch/pytorch/issues/132624) still open |
| 8-bit optimizers (bitsandbytes) | ❌ no Metal backend | UNVERIFIED for 2026 |

**Breaking change to know:** PyTorch 2.14 removed `at::mps::is_macos_13_or_newer()` with **no compatibility alias** (#188645) — any C++ extension including `<ATen/mps/MPSDevice.h>` will fail to compile.

**Verdict:** PyTorch MPS is fine for single-machine fine-tuning and structurally behind MLX for training. No distributed anything, no FSDP, no fused SDPA backward. **On Apple Silicon, use MLX for training and keep PyTorch for portability to CUDA.**

### What can you actually pretrain on a Mac?

Measured guidance from [scasella/nanochat-mlx](https://github.com/scasella/nanochat-mlx) (self-reported, M3 Pro — **not independently reproduced**):

| Depth | Params | Time (M3 Pro) |
|---|---|---|
| 4 | ~5M | ~1 min |
| **12** | **~125M** | **~1 hour** |
| 20 | ~350M | ~8 hours |
| 26 | ~600M | ~24 hours |

RAM guidance from the same source: 8 GB → depth 4; **16 GB → depth 12**; 32 GB+ → depth 20 and above.

- **16 GB M4 Mac Mini: realistic ceiling ~125M params** (~100–200M). AdamW from-scratch costs ~12–16 bytes/param before activations; ~10–11 GB usable after macOS. **UNVERIFIED estimate** anchored to the table above.
- **128 GB M3/M4 Ultra: realistic ceiling ~1–2B params.** Compute, not memory, is the binding constraint — a compute-optimal 1B model wants ~20B tokens, which is weeks, and a 4-Mac cluster only buys ~3×.

**⚠️ Important caveat to carry forward: there is no published tokens/sec *pretraining* benchmark for MLX at any model size on any chip.** Every "MLX tokens/sec" figure circulating publicly is decode-phase *inference*, often excluding prefill. Do not accept those as training numbers. No MLX pretraining scaling study, no MLX-vs-CUDA training throughput comparison, and no MFU figure for MLX training exists. **UNVERIFIED.**

**Forward-looking:** Apple's [M5 neural accelerator writeup](https://machinelearning.apple.com/research/exploring-llms-mlx-m5) reports ~3.6–4× faster *prefill* vs M4 but only ~20–27% faster decode. Training is matmul-heavy like prefill, so M5-class silicon should help training substantially more than it helps inference — but **no M5 training benchmark exists. UNVERIFIED.**



## Stage 10 — Data processing

| Repo | ★ | License | Last push | What it does | Scale | Hardware | Verdict |
|---|---|---|---|---|---|---|---|
| [huggingface/datatrove](https://github.com/huggingface/datatrove) | 3,328 | Apache-2.0 | 2026-08-13 | Platform-agnostic pipeline blocks: extract, filter, dedup (MinHash), tokenize; local / Slurm / Ray executors | Web-scale (FineWeb, FineWeb-2, SmolLM3 were all built with it) | CPU clusters | **The default.** It is the *actual* tool behind FineWeb and SmolLM3, not a demo. Readable, composable, and the pipeline configs for real corpora are published. Pick this. |
| [datajuicer/data-juicer](https://github.com/datajuicer/data-juicer) | 7,036 | Apache-2.0 | 2026-09-09 | 100+ composable operators, multimodal, plus [data-juicer-agents](https://github.com/datajuicer/data-juicer-agents) (58★) for agentic pipeline construction | Web-scale, Ray | CPU + GPU ops | **The strongest challenger and the most-starred option.** Broader operator library and better multimodal support than datatrove; heavier and more config-driven. Choose it if you need multimodal or a big op zoo. |
| [NVIDIA-NeMo/Curator](https://github.com/NVIDIA-NeMo/Curator) | 1,763 | Apache-2.0 | 2026-09-11 | GPU-accelerated curation: exact/fuzzy/semantic dedup, classifier filtering, PII | Web-scale, multi-node GPU | **GPU-accelerated (RAPIDS)** | **The fastest option if you have GPUs sitting idle.** Dedup on GPU is a genuine order-of-magnitude win. Cost: NVIDIA-only and a heavier dependency stack. |
| [allenai/dolma](https://github.com/allenai/dolma) | 1,543 | Apache-2.0 | push 2026-08-24, **default-branch last commit 2025-09-19** | Rust-based tagger/filter/dedup toolkit behind the Dolma corpora | Web-scale (Dolma 3 ≈ 9.3T tokens) | CPU | **Fast Rust core, but the main branch has been quiet for ~12 months** even though the repo shows recent pushes on other branches. Use it to *reproduce Olmo*; think twice before betting a new pipeline on it. |
| [mlfoundations/dclm](https://github.com/mlfoundations/dclm) | 1,471 | MIT | 2025-09-09 | DataComp-LM: the benchmark + the fastText-classifier-based filtering that produced DCLM-Baseline | Web-scale | CPU | **Dead (12 mo).** But the *finding* outlived the code: model-based quality filtering beats heuristics, and DCLM's CORE eval set is what nanochat still scores against. Steal the method, not the repo. |
| [ChenghaoMou/text-dedup](https://github.com/ChenghaoMou/text-dedup) | 764 | Apache-2.0 | 2026-03-09 | Standalone MinHash-LSH / SimHash / suffix-array / exact dedup | Medium (Spark variant for large) | CPU | **Stale-ish but fine.** Small, focused, easy to read. Good when you want dedup without adopting a whole pipeline framework. |
| "Ultimate Data Toolkit" | — | — | — | — | — | — | **UNVERIFIED — could not find any such project.** Searches surfaced nothing matching this name. Likely a conflation with Data-Juicer or datatrove; treat the name as spurious unless the requester has a specific URL. |
| [huggingface/fineweb-2](https://github.com/huggingface/fineweb-2) | 263 | Apache-2.0 | 2025-10-27 | Configs/code for the multilingual FineWeb-2 corpus | — | — | Not a framework — a *published recipe*. The single most useful thing to copy if you're building a web corpus. |

**Related but underrated:** [allenai/olmocr](https://github.com/allenai/olmocr) (19,468★, Apache-2.0, 2026-03-25) — PDF → clean text for LLM training data. Dolma 3 used it to pull science PDFs into the corpus. If your data is PDFs, this is the highest-leverage tool on this page.

---

## Recommended stack

Two columns, because the answer genuinely diverges. **(a)** is what runs on a 16 GB Apple Silicon machine. **(b)** is what you'd rent 8×H100 for.

| Stage | (a) Apple-Silicon-local | (b) Rented multi-GPU |
|---|---|---|
| **1. Full-stack pipeline** | [scasella/nanochat-mlx](https://github.com/scasella/nanochat-mlx) at `--depth 12` (~125M) — or nanochat's own [`runs/runcpu.sh`](https://github.com/karpathy/nanochat/blob/master/runs/runcpu.sh) if you want first-party code | [karpathy/nanochat](https://github.com/karpathy/nanochat) `runs/speedrun.sh` to learn the whole loop in ~1.65 h; [allenai/OLMo-core](https://github.com/allenai/OLMo-core) + [open-instruct](https://github.com/allenai/open-instruct) when you want a *real* reproducible model |
| **2. Pretraining framework** | **None good.** Use MLX directly via [`mlx-examples/transformer_lm`](https://github.com/ml-explore/mlx-examples/tree/main/transformer_lm). [N8python/mlx-pretrain](https://github.com/N8python/mlx-pretrain) is the best-designed option but 13 months stale | [pytorch/torchtitan](https://github.com/pytorch/torchtitan) — FSDP2+TP+PP+CP, float8/MXFP8, MoE, actively hot. Escalate to [Megatron-LM](https://github.com/NVIDIA/Megatron-LM) + [Megatron-Bridge](https://github.com/NVIDIA-NeMo/Megatron-Bridge) only at frontier scale |
| **3. Tokenizer** | [karpathy/rustbpe](https://github.com/karpathy/rustbpe) to train + [tiktoken](https://github.com/openai/tiktoken) to infer (the nanochat pattern), **vocab 32K** | Same pattern, or [huggingface/tokenizers](https://github.com/huggingface/tokenizers) for ecosystem fit. **Vocab 128K** only if you need multilingual |
| **4. Post-training** | [ml-explore/mlx-lm](https://github.com/ml-explore/mlx-lm) for SFT/LoRA/DoRA/QLoRA + [mlx-lm-lora](https://github.com/Goekdeniz-Guelmez/mlx-lm-lora) for DPO/GRPO/GSPO/DAPO/QAT | [verl](https://github.com/verl-project/verl) as default; [slime](https://github.com/THUDM/slime) for frontier MoE; [prime-rl](https://github.com/PrimeIntellect-ai/prime-rl) for async agentic. [TRL](https://github.com/huggingface/trl) for anything small |
| **5. RL environments** | [reasoning-gym](https://github.com/open-thought/reasoning-gym) — ~108 procedural generators, pure Python, **no sandbox and no GPU needed** | [verifiers](https://github.com/PrimeIntellect-ai/verifiers) + Environments Hub (+ [OpenEnv](https://github.com/huggingface/OpenEnv) as the portability hedge); [SWE-smith](https://github.com/SWE-bench/SWE-smith) for coding; Modal Sandboxes for execution |
| **6. Distillation / merging** | [mergekit](https://github.com/arcee-ai/mergekit) (CPU merging works fine on Apple Silicon) + mlx-lm's **DWQ** for quantization distillation | [TRL `DistillationTrainer`](https://github.com/huggingface/trl/blob/main/trl/trainer/distillation_trainer.py) (+ `gold` for cross-tokenizer) + mergekit |
| **7. Evaluation** | [inspect_ai](https://github.com/UKGovernmentBEIS/inspect_ai) + [inspect_evals](https://github.com/UKGovernmentBEIS/inspect_evals) pointed at a local `mlx_lm.server` | **inspect_ai + inspect_evals** (agentic) **+ lm-evaluation-harness** (static/reported numbers) **+ [harbor](https://github.com/harbor-framework/harbor)** (terminal/agentic, doubles as RL rollout generator) |
| **8. Serving** | [mlx-lm](https://github.com/ml-explore/mlx-lm) (`mlx_lm.server`); [SGLang's MLX backend](https://github.com/sgl-project/sglang/tree/main/python/sglang/srt/hardware_backend/mlx) as the new second option | [vLLM](https://github.com/vllm-project/vllm) (widest RL-framework integration) or [SGLang](https://github.com/sgl-project/sglang) (broadest hardware) — **match it to your trainer, not your taste** |
| **9. Apple Silicon** | [MLX](https://github.com/ml-explore/mlx) + JACCL if you ever add a second Mac (needs **macOS 26.2+**). **Do not use PyTorch MPS for training** | n/a |
| **10. Data** | [datatrove](https://github.com/huggingface/datatrove) — scales down to one machine cleanly | [datatrove](https://github.com/huggingface/datatrove) by default; [NeMo Curator](https://github.com/NVIDIA-NeMo/Curator) if you have idle GPUs (GPU dedup is an order-of-magnitude win); [data-juicer](https://github.com/datajuicer/data-juicer) if multimodal |

### The honest framing on (a)

A 16 GB Mac Mini pretrains **~125M parameters**, and even that is a token-truncated run, not a compute-optimal one (a Chinchilla-optimal 125M model wants ~2.5B tokens). Apple Silicon is a **learning and fine-tuning platform**, not a pretraining platform. The specific things it is genuinely good at:

- **Fine-tuning** up to ~8B with LoRA/QLoRA — first-party, well-supported, fast enough to iterate on.
- **Preference and RL post-training at small scale** via `mlx-lm-lora`, which covers DPO/GRPO/GSPO/DAPO/QAT.
- **Model merging** — `mergekit` runs on CPU, so this whole stage is laptop-native.
- **Eval orchestration** — `inspect_ai` drives remote models; your Mac is a controller, not a compute node.
- **Data processing** — `datatrove` is CPU-bound and parallelizes fine.

### The honest framing on (b)

The cost floor is now genuinely low. nanochat's `speedrun.sh` reaches GPT-2-class CORE (0.2626) in **1.65 hours on 8×H100** — roughly $48 on demand, ~$15 on spot. For comparison, the original GPT-2 took ~168 h on 32 TPUv3 (~$43K in 2019). **The barrier to "I trained an LLM end to end" is now one evening and a coffee's worth of compute.** The expensive part in 2026 is not pretraining — it is RL, environments, and eval.

### Three gaps worth knowing about

1. **There is no maintained, well-licensed MLX pretraining framework.** `mlx-pretrain` is stale and CC0, the nanochat MLX ports are weekend projects, and Awni Hannun's own `picochat` is a two-line empty placeholder created 2026-03-06 and never filled in. This is an unusually open niche for anyone wanting visible open-source traction.
2. **There is no good fully-self-hostable open-source code sandbox** with E2B/Modal ergonomics. Cost-sensitive labs roll their own with bubblewrap/nsjail and build the orchestration themselves.
3. **No one has published a tokens/sec *pretraining* benchmark for MLX** at any model size on any chip. Every MLX throughput number in public circulation is decode-phase inference. A rigorous MLX training benchmark would be a genuinely useful contribution.

### Watch list for the next six months

- **[TitanRL](https://github.com/pytorch/torchtitan/tree/main/torchtitan/experiments/rl)** — PyTorch's consolidated RL bet after two failed attempts; batch-invariant logprobs is the right idea.
- **[harbor](https://github.com/harbor-framework/harbor)** — deliberately collapsing eval and RL environments into one artifact. TRL already ships a `harbor` module.
- **[vllm-project/vime](https://github.com/vllm-project/vime)** — vLLM growing its own post-training framework changes the ecosystem's center of gravity.
- **[radixark/miles](https://github.com/radixark/miles)** (2,825★, Apache-2.0) — bigger than SkyRL and NeMo-RL by stars and absent from every standard list.
- **SGLang's MLX backend** — if it matures, Apple Silicon gets a production-grade serving path.
- **M5-class Apple silicon** — ~3.6–4× faster prefill vs M4; training is matmul-heavy like prefill, so it should help training more than inference. **UNVERIFIED — no M5 training benchmark exists.**

---

## Appendix — every repo-move discovered in this survey

Old paths that still redirect, and will silently mislead anyone reading a 2025 guide:

| Old path | Now resolves to |
|---|---|
| `NVIDIA/NeMo` | [`NVIDIA-NeMo/Speech`](https://github.com/NVIDIA-NeMo/Speech) (speech only — LLM training moved elsewhere) |
| `pytorch/torchtune` | [`meta-pytorch/torchtune`](https://github.com/meta-pytorch/torchtune) (frozen) |
| `meta-pytorch/forge` | [`meta-pytorch/torchforge`](https://github.com/meta-pytorch/torchforge) (paused) |
| `meta-pytorch/OpenEnv` | [`huggingface/OpenEnv`](https://github.com/huggingface/OpenEnv) |
| `laude-institute/terminal-bench` | [`harbor-framework/terminal-bench-1`](https://github.com/harbor-framework/terminal-bench-1) |
| `volcengine/verl` | [`verl-project/verl`](https://github.com/verl-project/verl) |
| `stanford-crfm/levanter` | [`marin-community/levanter`](https://github.com/marin-community/levanter) (merged into Marin) |
| `inclusionAI/AReaL` | [`areal-project/AReaL`](https://github.com/areal-project/AReaL) |
| `hiyouga/LLaMA-Factory` | [`hiyouga/LlamaFactory`](https://github.com/hiyouga/LlamaFactory) |
| `agentica-project/R2E-Gym` (23★ stub) | real repo is [`R2E-Gym/R2E-Gym`](https://github.com/R2E-Gym/R2E-Gym) (333★) |

### Licensing landmines

- **[karpathy/autoresearch](https://github.com/karpathy/autoresearch)** — 95,675★ and **no license file.** Legally all-rights-reserved despite being the most-starred new repo in this survey.
- **[mlfoundations/evalchemy](https://github.com/mlfoundations/evalchemy)** — no license file.
- **[arcee-ai/mergekit](https://github.com/arcee-ai/mergekit)** — **LGPL-3.0**, notably more restrictive than the Apache/MIT norm everywhere else here.
- **[Doriandarko/MLX-GRPO](https://github.com/Doriandarko/MLX-GRPO)** — 242★, no license, 11 months stale.
- **[NVIDIA/Megatron-LM](https://github.com/NVIDIA/Megatron-LM)** — `NOASSERTION` (custom NVIDIA terms, not a standard OSI license).

