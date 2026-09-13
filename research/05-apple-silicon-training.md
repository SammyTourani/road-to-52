# LLM Training on an Apple M4 Mac Mini (16 GB) with MLX — Research Report

**Date:** 2026-09-13 · **Target machine:** Mac Mini M4 (base), 10-core GPU, 16 GB unified memory, macOS 26.5.2, `mlx 0.31.2`, Python 3.14, `uv` available, PyTorch **not** installed.

---

## 0. Hardware ground truth (measured locally + sourced)

Measured on this machine via `mx.device_info()`:

```
device_name: Apple M4
memory_size: 17,179,869,184 B (16.0 GiB)
max_recommended_working_set_size: 12,713,115,648 B  →  11.84 GiB
max_buffer_length: 9,534,832,640 B  →  8.88 GiB     ← single-allocation ceiling
architecture: applegpu_g16g
```

| Quantity | Value | Source |
|---|---|---|
| GPU cores | 10 | [Apple Newsroom, M4 Mac mini](https://www.apple.com/newsroom/2024/10/apples-new-mac-mini-is-more-mighty-more-mini-and-built-for-apple-intelligence/) |
| Theoretical FP32 peak | **4.26 TFLOPS** | [arXiv 2502.05317, Table 1](https://arxiv.org/html/2502.05317v1) · [flopper.io/gpu/apple-m4](https://flopper.io/gpu/apple-m4) |
| **Measured** FP32 peak (microbench) | **2.9 TFLOPS** | [arXiv 2502.05317 §Results](https://arxiv.org/html/2502.05317v1) |
| Memory bandwidth (spec) | 120 GB/s | [Apple Newsroom](https://www.apple.com/newsroom/2024/10/apple-introduces-m4-pro-and-m4-max) |
| **Measured** STREAM bandwidth (GPU) | **100 GB/s** (~85 % of peak) | [arXiv 2502.05317](https://arxiv.org/html/2502.05317v1) |
| Thunderbolt | **TB4 only** (TB5 is M4 **Pro** exclusive) | [MacRumors 2024-10-29](https://www.macrumors.com/2024/10/29/m4-pro-mac-mini-features-thunderbolt-5/) |
| FP16 speedup over FP32 in MLX | only **1.2–1.3×** (not 2×) | [arXiv 2501.14925](https://arxiv.org/html/2501.14925v2) |
| Usable GPU working set | **11.84 GiB** (Metal caps a process below total RAM) | measured; corroborated by [DeepSpeed Apple Silicon guide](https://www.deepspeed.ai/tutorials/accelerator-setup-guide/) |

**Best real-world calibration of the M4's matmul ceiling:** llama.cpp's canonical Apple-Silicon table reports **LLaMA-7B F16, pp512 = 230.18 tok/s** on M4/10-core ([ggml-org/llama.cpp discussion #4167](https://github.com/ggml-org/llama.cpp/discussions/4167)). That is `2 × 7e9 × 230.18 = 3.22 TFLOPS` — i.e. **76 % of theoretical peak on a pure, perfectly-batched fp16 GEMM workload**. Training will never beat that; it is the hard ceiling.

> **Correction to the task brief:** "4–5 TFLOPS fp16 sustained" is optimistic. The realistic *ceiling* is ~3.2 TFLOPS and realistic *training* throughput on MLX lands at **0.8–1.3 TFLOPS (≈20–30 % MFU vs. the 4.26 TFLOPS theoretical peak)**. See §5 cross-checks.

Full llama.cpp #4167 rows (LLaMA-7B v2, Metal, `-ngl 99`, pp = batch 512 prompt processing, tg = batch-1 generation):

| Chip | BW GB/s | GPU cores | F16 PP | F16 TG | Q8_0 PP | Q8_0 TG | Q4_0 PP | Q4_0 TG |
|---|---|---|---|---|---|---|---|---|
| **M4** | 120 | **10** | **230.18** | **7.43** | 223.64 | 13.54 | 221.29 | **24.11** |
| M4 Pro | 273 | 16 | 381.14 | 17.19 | 367.13 | 30.54 | 364.06 | 49.64 |
| M4 Pro | 273 | 20 | 464.48 | 17.18 | 449.62 | 30.69 | 439.78 | 50.74 |

---

## 1. MLX fine-tuning capabilities in 2026

### 1.1 What official `mlx_lm.lora` supports

Repo: [ml-explore/mlx-lm](https://github.com/ml-explore/mlx-lm) — 6,993 ★, last push 2026-09-12, latest release **v0.31.3 (2026-04-22)**. Local install is `mlx 0.31.2` / matching mlx-lm line.

Verified by reading the source ([`mlx_lm/lora.py`](https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/lora.py), [`mlx_lm/tuner/`](https://github.com/ml-explore/mlx-lm/tree/main/mlx_lm/tuner), [`LORA.md`](https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/LORA.md)):

| Feature | Status in official mlx-lm |
|---|---|
| Fine-tune types | **`lora` (default), `dora`, `full`** — `--fine-tune-type` |
| Loss / training modes | **SFT (next-token CE) only.** `tuner/losses.py` contains only `kl_div_loss` and `js_div_loss` (custom Metal kernels, for **distillation**). **No DPO / ORPO / CPO / GRPO in the official CLI.** |
| QLoRA | **Yes, implicit** — "If `--model` points to a quantized model, then the training will use QLoRA" |
| Gradient checkpointing | **Yes** — `--grad-checkpoint` (implemented via `mx.checkpoint` monkey-patching `layer.__call__`) |
| Gradient accumulation | **Yes** — `--grad-accumulation-steps` |
| Optimizers | **`adam`, `adamw`, `muon`, `sgd`, `adafactor`** (`--optimizer`) — Muon is notable |
| Prompt masking | `--mask-prompt` |
| Distributed data-parallel | **Yes** — `iterate_batches(..., comm_group=...)`; launched via `mlx.launch` |
| Logging | `--report-to wandb|swanlab` |
| Allocator control | `--clear-cache-threshold` |
| Defaults | model `Qwen/Qwen3-0.6b`, `num_layers=16`, `batch_size=4`, `max_seq_length=2048`, `lora rank=8, scale=20.0, dropout=0.0`, `lr=1e-5`, `iters=1000` |

**Loss dtype gotcha (good news):** `default_loss` in [`tuner/trainer.py`](https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/tuner/trainer.py) computes `nn.losses.cross_entropy(logits, targets)` on the **model dtype** (bf16) and only casts the *scalar* to fp32. So the logits tensor is 2 bytes/elem, not 4. This halves the biggest activation term (see §6).

**Architecture coverage:** `LORA.md` still lists only 9 families (Mistral, Llama, Phi2, Mixtral, Qwen2, Gemma, OLMo, MiniCPM, InternLM2) — **that list is stale**. [`mlx_lm/models/`](https://github.com/ml-explore/mlx-lm/tree/main/mlx_lm/models) contains **129 files** including `qwen3`, `qwen3_5`, `qwen3_moe`, `qwen3_next`, `gemma3`, `gemma4`, `gemma4_text`, `olmo3`, `smollm3`, `gpt_oss`, `llama4`, `phi3`, `lfm2`, `granitemoe`, `ministral3`, `nemotron_h`, `nanochat`, `bitnet`, `mamba2`, `deepseek_v32`, `kimi_k3`, `glm4`, `minimax`. LoRA targets linear layers generically, so coverage is effectively "anything mlx-lm can load" — but **MoE models are a known LoRA-fusing exception** ([awni, mlx discussion #1560](https://github.com/ml-explore/mlx/discussions/1560)).

### 1.2 Preference/RL training: use `mlx-lm-lora` (third-party)

[Goekdeniz-Guelmez/mlx-lm-lora](https://github.com/Goekdeniz-Guelmez/mlx-lm-lora) — Apache-2.0, 412 ★, [PyPI](https://pypi.org/project/mlx-lm-lora/). Adds **12 training algorithms**: SFT, **DPO, CPO, ORPO**, FTPO, **GRPO, GSPO, Dr. GRPO, DAPO**, Online DPO, XPO, RLHF-Reinforce, PPO. Also **QAT** (4–16 bit, group or per-tensor) for SFT/DPO/ORPO, and QLoRA at 4/6/8-bit. CLI: `mlx_lm_lora.train --training-mode dpo --dpo-loss-type sigmoid|hinge|ipo|dpop`; GRPO loss types `grpo|bnpo|dr_grpo`.

Its own README benchmarks (**M4 Pro, 24 GB, 100 steps**):

| Model | Mode | Speed | Memory |
|---|---|---|---|
| Qwen 0.6B | SFT | ~4.7 it/s | 2–2 GB |
| Qwen 0.6B | ORPO | ~4.5 it/s | 2–4 GB |
| Qwen 8B (4-bit) | SFT | ~4.1 it/s | 6–10 GB |

> Qwen-8B 4-bit SFT peaking at **10 GB on an M4 Pro** is the single most relevant data point: it fits under the M4's 11.84 GiB ceiling, but only just, and with nothing else running.

### 1.3 Reported fine-tuning throughput on M-series

| Hardware | Model | Setting | Throughput | Source |
|---|---|---|---|---|
| M1 Max 32 GB | Mistral-7B (bf16) | LoRA, `batch 1`, `num-layers 4` | **~250 tok/s** | [LORA.md](https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/LORA.md) (official) |
| M3 Ultra (80-core) | Qwen3.5-9B | `mlx_lm.lora` LoRA | **~180 tok/s** | [WWDC26 session 233](https://developer.apple.com/videos/play/wwdc2026/233/) |
| 4× M3 Ultra + JACCL | Qwen3.5-9B | distributed LoRA | **~600 tok/s** (>3×) | [WWDC26 session 233](https://developer.apple.com/videos/play/wwdc2026/233/) |
| M2 (base) | 7B QLoRA | QLoRA | ~70 tok/s @ 9 W | [r/LocalLLaMA](https://www.reddit.com/r/LocalLLaMA/comments/1bfgy35/qlora_on_mlx_just_got_way_more_ram_efficient_again) |
| M2 24 GB | 7B | LoRA, `batch 1`, `lora-layers 4` | 60–80 tok/s | [r/LocalLLaMA](https://www.reddit.com/r/LocalLLaMA/comments/18wabkc/lessons_learned_so_far_lora_fine_tuning_on) |

**M4-base extrapolation (UNVERIFIED — main agent should measure):** M3 Ultra is 80 GPU cores / 819 GB/s vs. M4's 10 cores / 120 GB/s. Scaling the 180 tok/s Qwen3.5-9B LoRA figure by GPU cores gives **~20–25 tok/s for a 9B 4-bit QLoRA on this M4** — i.e. ~1.4 M tokens/day. A 1–2 B model should land **150–400 tok/s**.

### 1.4 Max practical fine-tune size on 16 GB

| Approach | Fits on 16 GB M4? | Evidence |
|---|---|---|
| **QLoRA 4-bit, 7–8 B** | **Yes, tight.** batch 1–2, `--num-layers 8–16`, seq ≤ 2048, grad-checkpoint on | Qwen-8B 4-bit SFT peaked at 10 GB on M4 Pro ([mlx-lm-lora](https://github.com/Goekdeniz-Guelmez/mlx-lm-lora)); "QLoRA on a 16 GB MacBook fine-tunes an 8 B model in about an hour" ([InsiderLLM](https://insiderllm.com/guides/fine-tuning-mac-lora-mlx/)) — treat the "hour" as marketing |
| LoRA on bf16 7 B | **No** (14 GB weights alone > 11.84 GiB budget) | §6 math |
| LoRA/QLoRA 3–4 B | **Yes, comfortable** | 4-bit 4 B ≈ 3.0 GB weights (measured, §8) |
| **Full fine-tune (`--fine-tune-type full`)** | **≤ ~1.0–1.3 B** in pure bf16 w/ bf16 Adam states; **≤ ~0.6 B** with fp32 Adam | §6 math |
| Full fine-tune 1.7 B | borderline — needs Adafactor or 8-bit Adam + grad-ckpt + batch 1 | §6 |

Apple's own maintainer guidance: LoRA on 7 B works from 32 GB; *"if you want to do full fine-tuning … you will probably want more RAM"* — [awni, mlx#1560](https://github.com/ml-explore/mlx/discussions/1560).

---

## 2. Pretraining from scratch on MLX

### 2.1 The projects that exist

| Project | What it is | Status | License |
|---|---|---|---|
| [ml-explore/mlx-examples/transformer_lm](https://github.com/ml-explore/mlx-examples/tree/main/transformer_lm) | Official minimal decoder-only LM, PTB corpus. `python main.py --gpu` | alive (repo last push 2026-04-06, 8,943 ★) | MIT |
| [**a14a-org/mlxgpt**](https://github.com/a14a-org/mlxgpt) · site [mlxgpt.com](https://mlxgpt.com/) | **Native MLX port of nanochat**, 2-node Mac Mini M4 Pro cluster over Thunderbolt RDMA (JACCL). DP **and** TP. Pure Python + MLX. | **very active** (created 2026-03-18, pushed 2026-09-13) | MIT |
| [**scasella/nanochat-mlx**](https://github.com/scasella/nanochat-mlx) | Full single-machine MLX port of nanochat: data → tokenizer → pretrain → SFT → chat → eval. Web wizard. | active, 70 ★ (2026-02 → 2026-04) | MIT |
| [vithursant/nanoGPT_mlx](https://github.com/vithursant/nanoGPT_mlx) | nanoGPT → MLX port | **stale** (last push 2024-02) | MIT |
| [pranavjad/mlx-gpt2](https://github.com/pranavjad/mlx-gpt2) | GPT-2 from scratch tutorial, char-level Shakespeare | stale (2024-06), 437 ★ | none |
| [NeuroArchitect/nanochat-mlx](https://github.com/NeuroArchitect/nanochat-mlx) | mostly an unmodified nanochat fork; **README is vanilla PyTorch**, no MLX port. Useful only for its speedrun leaderboard. | 0 ★ | MIT |
| [karpathy/nanochat](https://github.com/karpathy/nanochat) `runs/runcpu.sh` | **Official CPU/MPS path** (PyTorch, not MLX) | active, 57,979 ★ | MIT |

### 2.2 Measured pretraining throughput

| Setup | Model | Throughput | Source |
|---|---|---|---|
| **2× Mac Mini M4 Pro (64 GB, 20-core GPU ea.), TB5 RDMA/JACCL** | nanochat **d14 ≈ 170 M params**, seq 512, f32 | **2,065 tok/s** (champion run, val loss 3.506 @ 29,750 steps, ~15 h) | [mlxgpt.com](https://mlxgpt.com/) / [README](https://github.com/a14a-org/mlxgpt) |
| same | d14, **pure bf16** | **4,530 tok/s** — *but converges poorly* (val 5.012, plateaued) | [mlxgpt README](https://github.com/a14a-org/mlxgpt) |
| same | d14, **mixed precision** | **3,830 tok/s**, val 3.634, diverged late | [mlxgpt README](https://github.com/a14a-org/mlxgpt) |
| same | d14, optimized f32 | 3,000 tok/s | [mlxgpt README](https://github.com/a14a-org/mlxgpt) |
| same | **d20 ≈ 477 M params** | **284 tok/s**, est. **2.8 days** for 70 k steps | [mlxgpt.com](https://mlxgpt.com/) |
| M3 Pro MacBook (18-core GPU) | ~10.6 M-param GPT-2 (6L, d=384, ctx 256, eff. batch 64) | **0.37 it/s ≈ 6,060 tok/s** | [nanoGPT_mlx README](https://github.com/vithursant/nanoGPT_mlx) + [config](https://github.com/vithursant/nanoGPT_mlx/blob/main/configs/train_gpt2_shakespeare.py) — note README says "~45 M param" but the config computes to ~10.6 M body params; **discrepancy flagged** |
| M3 Max MacBook Pro (**PyTorch MPS**) | nanochat depth 6, seq 512, total-batch 16,384 tok, 5,000 iters = 82 M tokens | **"about 30 minutes"** ⇒ ~45,000 tok/s | [runs/runcpu.sh](https://github.com/karpathy/nanochat/blob/master/runs/runcpu.sh) (tuned 2026-01-17) |
| M2 MacBook | char-level 3-layer d=128 toy | "~10 minutes" for 20 epochs on 1 M chars | [mlx-gpt2](https://github.com/pranavjad/mlx-gpt2) |

### 2.3 Biggest model anyone reports pretraining on Macs

- **Single Mac:** no credible public report above ~**125 M params at a compute-optimal token budget**. The [scasella/nanochat-mlx README table](https://github.com/scasella/nanochat-mlx) claims *depth 12 ≈ 125 M in ~1 h, depth 20 ≈ 350 M in ~8 h, depth 26 ≈ 600 M in ~24 h on an M3 Pro* — **these numbers are not physically possible and should be treated as UNVERIFIED/wrong.** Its own `train.py` uses `target_param_data_ratio = 12.0`, so depth-12 (≈110 M scaling params) targets **≈1.32 B tokens**; 1.32 B tokens in 1 h = 367,000 tok/s = 242 TFLOPS on an M3 Pro (~6–7 TFLOPS part). Off by ~40×. The README's "RAM by depth" guidance (16 GB → depth 12) is still a reasonable *memory* guide.
- **Mac cluster:** **mlxgpt's 477 M-param d20 run on 2× M4 Pro**, ~2.8 days projected — the largest genuine from-scratch Apple-Silicon pretrain I found.
- **Reference point for "GPT-2 grade":** nanochat **d24–d26** ([karpathy/nanochat README](https://github.com/karpathy/nanochat)); the speedrun leaderboard puts it at **2.76–3.04 h on 8×H100 (~$72)** ([leaderboard](https://github.com/NeuroArchitect/nanochat-mlx)). At ~450 TFLOPS sustained that's ≈4.5e18 FLOPs ⇒ **≈130–160 days on a single M4** at 20–25 % MFU. Not happening locally.

---

## 3. Apple-Silicon distributed training

### 3.1 `mlx.distributed` backends (2026)

Per the [official docs](https://ml-explore.github.io/mlx/build/html/usage/distributed.html) (MLX 0.32.2):

| Backend | Description | Relevant to a base M4 mini? |
|---|---|---|
| **MPI** | mature, needs `mpirun` on all nodes | yes, but slowest |
| **Ring** | ring all-reduce/all-gather **over TCP sockets**; "always available and usually faster than MPI" | **yes — this is the only option for a TB4 machine** |
| **JACCL** | **RDMA over Thunderbolt 5**, macOS **26.2+**. "communication latency an order of magnitude lower than ring". Needed for tensor parallelism. | **NO — requires TB5, which base M4 lacks** |
| **NCCL** | CUDA | n/a |

Collectives: `all_sum`, `all_max`, `all_min`, `all_gather`, `send`, `recv`, `recv_like`, `sum_scatter`. Ring does not support arbitrary P2P pairs.

Launcher: `mlx.launch --hosts ip1,ip2 script.py` / `--hostfile f.json`; `mlx.distributed_config --over thunderbolt --auto-setup --backend jaccl` auto-discovers Thunderbolt topology and writes the hostfile ([launching docs](https://ml-explore.github.io/mlx/build/html/usage/launching_distributed.html)).

### 3.2 Is multi-Mac *training* real? Yes.

- **Apple ships it.** [WWDC26 session 233 "Explore distributed inference and training with MLX"](https://developer.apple.com/videos/play/wwdc2026/233/) demos **data-parallel fine-tuning**: `mlx.launch --hostfile hostfile.json -- mlx_lm.lora --model Qwen/Qwen3.5-9B --train --batch-size 16`. Measured: **1× M3 Ultra ≈ 180 tok/s → 4× M3 Ultra ≈ 600 tok/s (>3×)**. Inference: Qwen3.6-27B ≈ **3× speedup** on 4 Macs; Kimi-K2.6 (1 T params, ~1 TB @ 8-bit) runs across 4 Macs with `--pipeline`.
- **Third parties do it.** mlxgpt runs real DP *and* TP pretraining across 2× M4 Pro over JACCL/RDMA ([docs/reproducible_cluster_setup.md](https://github.com/a14a-org/mlxgpt)).
- **JACCL** = "Jack and Angelos' Collective Communication Library", a deliberate NCCL pun; it auto-picks mesh (low latency) vs. ring (high bandwidth) by message size, and is usable standalone from C++ ([WWDC26 233](https://developer.apple.com/videos/play/wwdc2026/233/)).

### 3.3 RDMA over Thunderbolt: the numbers

| Metric | Value | Source |
|---|---|---|
| macOS requirement | **26.2+** | [MLX distributed docs](https://ml-explore.github.io/mlx/build/html/usage/distributed.html) |
| TB5 raw | 80 Gb/s (120 Gb/s asymmetric burst) | [Apple Newsroom](https://www.apple.com/newsroom/2024/10/apple-introduces-m4-pro-and-m4-max) |
| Achieved cluster throughput | **50–60 Gbps** | [byteiota](https://byteiota.com/mlx-jaccl-thunderbolt-distributed-training/) |
| Round-trip latency | **~300 µs (TCP) → 3–50 µs (RDMA)** | [byteiota](https://byteiota.com/mlx-distributed-training-with-jaccl-multi-mac-llm-clusters-explained/) · [AppleInsider](https://appleinsider.com/articles/25/12/20/ai-calculations-on-mac-cluster-gets-a-big-boost-from-new-rdma-support-on-thunderbolt-5) |
| Cluster size supported | 2–4 Macs, full mesh needs a cable per pair | [MLX docs](https://ml-explore.github.io/mlx/build/html/usage/distributed.html) |
| Env flag | `MLX_METAL_FAST_SYNCH=1` (faster CPU↔GPU sync, "reliability concerns") | [MLX docs](https://ml-explore.github.io/mlx/build/html/usage/distributed.html) |

**Verdict for this machine:** ⛔ **JACCL is off the table** — base M4 Mac mini has **Thunderbolt 4**, and RDMA-over-Thunderbolt is TB5-only. A second base M4 mini could only be joined with the **ring** backend over TB4-as-IP or 10 GbE, at ~300 µs latency. For a 124 M model at ~1,100 tok/s per node, a ring all-reduce of ~250 MB of gradients per step over ~20 Gb/s effective would dominate; **2× base M4 minis is not worth building for training.**

### 3.4 exo

[exo-explore/exo](https://github.com/exo-explore/exo) — Apache-2.0, **47,385 ★**, last push 2026-08-25, **not archived**. Uses MLX as its inference backend and `mlx.distributed` for comms; day-0 RDMA-over-TB5 support; tensor parallelism "1.8× on 2 devices, 3.2× on 4". **Grepping the README for `train|finetune|fine-tun` returns zero matches — exo is inference-only.** Not a training tool.

---

## 4. PyTorch MPS in 2026

Current: **PyTorch 2.13** ([MPS backend notes](https://docs.pytorch.org/docs/2.13/notes/mps.html), page updated 2026-05-11).

| Capability | Status | Source |
|---|---|---|
| **bf16** | ✅ supported, **requires macOS 14.0+**. Both bf16 and fp16 mixed precision work. | [HF Transformers, perf_train_special](https://github.com/huggingface/transformers/blob/main/docs/source/en/perf_train_special.md) |
| **torch.compile / Inductor** | ✅ **MPS is a supported Inductor backend** — codegen via a `MetalKernel` class; three modes: eager dispatch, Inductor-compiled, and AOTI ahead-of-time. | [DeepWiki: pytorch MPS backend](https://deepwiki.com/pytorch/pytorch/3.3-mps-backend-(metal-performance-shaders)) |
| **SDPA / FlashAttention** | ⚠️ **No native metal-flash-attention in upstream SDPA.** Tracking issue [pytorch/pytorch#139668](https://github.com/pytorch/pytorch/issues/139668) still open. Third-party [`mps-flash-attn`](https://pypi.org/project/mps-flash-attn/) ([mpsops/mps-flash-attention](https://github.com/mpsops/mps-flash-attention)) patches `F.scaled_dot_product_attention`, supports bf16 backward and torch.compile. [Metal FlashAttention 2.0](https://engineering.drawthings.ai/p/metal-flashattention-2-0-pushing-forward-on-device-inference-training-on-apple-silicon-fe8aac1ab23c) is used for real training (SDXL LoRA, FLUX). SDPA on MPS reportedly crashes/allocates badly above ~12 k sequence length. |
| **Fused AdamW** | ✅ since **torch 2.7** — a Metal kernel compiled via `torch.mps.compile_shader`. | [DeepSpeed accelerator guide](https://www.deepspeed.ai/tutorials/accelerator-setup-guide/) |
| **Missing ops** | ⚠️ Persistent. `PYTORCH_ENABLE_MPS_FALLBACK=1` silently routes to CPU (huge slowdowns). | [HF docs](https://github.com/huggingface/transformers/blob/main/docs/source/en/perf_train_special.md) |
| **fp64** | ❌ not supported (grad-norm accumulation must use fp32). | [DeepSpeed PR #8293](https://github.com/deepspeedai/deepspeed/issues/8293) |
| **Multi-GPU on one Mac** | ❌ `torch.mps.device_count() == 1` — no single-machine data parallel. | [DeepSpeed guide](https://www.deepspeed.ai/tutorials/accelerator-setup-guide/) |
| **Collectives** | ❌ **no native collective backend.** gloo cannot operate on MPS tensors even at world size 1; DeepSpeed stages every collective through CPU copies. `nccl`/`gloo` "not working with mps device". Multi-machine gloo over MPS is **untested**. | [DeepSpeed PR #8293](https://github.com/deepspeedai/deepspeed/issues/8293), [#8303](https://github.com/deepspeedai/deepspeed/issues/8303) |
| **torchao low-bit optimizers** | ⚠️ 4-bit optim uses ops not implemented on MPS; 8-bit AdamW shows no memory savings because grads get upcast to fp32. | [pytorch/ao#955](https://github.com/pytorch/ao/issues/955) |
| **DeepSpeed ZeRO 0–3** | ✅ **new in 2026** — single-device only, verified on M5 Max / macOS 26.3 / torch 2.13.0. | [DeepSpeed PR #8293](https://github.com/deepspeedai/deepspeed/issues/8293) |
| **Kernel launch latency** | ⚠️ "much lower latency on CUDA than Apple Silicon" for both cold and repeat launches; page faults climb during training. | [arXiv 2501.14925](https://arxiv.org/html/2501.14925v2) |

**MLX vs MPS:** arXiv 2501.14925 ("Profiling Apple Silicon Performance for ML Training", Feng/Xu/Wang/Lin, Jan 2025) — "**MLX framework outperforms PyTorch MPS on M2 devices**"; for GPT-2-large pretraining, "**the RTX A6000 slightly underperforms compared to the M2 Ultra (MLX)**"; and "even on an RTX 4090, if ZeRO-Offload is applied, the training performance is still worse than M2 Ultra." Also: MLX's FP16 matmul gains only 20–30 % over FP32 (vs. CUDA's much larger gain), and VJP FP16 speedup is ~2× on MLX/MPS vs. 5–6× on CUDA.

### 4.1 Would nanochat run on MPS? **Yes — officially.**

[karpathy/nanochat](https://github.com/karpathy/nanochat) (MIT, 57,979 ★, pushed 2026-09-07):

- `uv sync --extra cpu` is documented as "**Use for CPU-only / MPS**".
- `nanochat/common.py::autodetect_device_type()` → `cuda` → **`mps`** → `cpu`; `compute_init()` asserts `device_type in ["cuda","mps","cpu"]`.
- README §"Running on CPU / MPS" points at [`runs/runcpu.sh`](https://github.com/karpathy/nanochat/blob/master/runs/runcpu.sh) (last tuned **2026-01-17**), which trains depth-6, head-dim 64, seq 512, device-batch 32, total-batch 16,384 tok, 5,000 iters — "**about 30 minutes on my MacBook Pro M3 Max**", then SFT in ~10 min. Tokenizer training on 2 B chars: ~34 s.
- Precision table: **CPU/MPS default `float32`**, and "*On recent macOS, MPS also runs `NANOCHAT_DTYPE=bfloat16` fine (~25 % less memory, similar speed)*". Override with `NANOCHAT_DTYPE`.
- README caveat: "Most of the code is fairly vanilla PyTorch so it should run on … xpu, mps, or etc, but I haven't personally exercised all of these code paths so there might be sharp edges."
- **The d20/d26 speedrun will NOT run on MPS in any useful time** (see §2.3 — ~130+ days on an M4). DDP is nccl-only in `compute_init`; on MPS it falls through to the single-device branch.
- macOS forks: [miolini/autoresearch-macos](https://github.com/miolini/autoresearch-macos) (2,387 ★) removes the hard FlashAttention-3 dependency and falls back to SDPA with manual sliding-window causal masking.

---

## 5. Throughput math (6·N·D FLOPs/token)

Formula: `tokens/s = MFU × C_peak / (6N)`, `hours = D / (tok/s) / 3600`. `C_peak = 4.26 TFLOPS` (M4 theoretical FP32). **20 % is the empirically-calibrated MLX training MFU** (see cross-check below); 30 % is optimistic; 50 % is unreachable on this hardware (llama.cpp's pure-GEMM prefill only reaches 76 %, and training is strictly harder).

### 5.1 Tokens/sec

| Model N | @ 20 % MFU (realistic) | @ 30 % MFU | @ 50 % MFU (aspirational) |
|---|---|---|---|
| 30 M | 4,733 | 7,100 | 11,833 |
| 60 M | 2,367 | 3,550 | 5,917 |
| 124 M | **1,145** | 1,718 | 2,863 |
| 350 M | 406 | 609 | 1,014 |
| 1 B | 142 | 213 | 355 |

### 5.2 Wall-clock, 20 % MFU (realistic)

| Model | 1 B tokens | 10 B tokens | 20 B tokens |
|---|---|---|---|
| 30 M | **2.4 d** | 24.5 d | 48.9 d |
| 60 M | 4.9 d | 48.9 d | 97.8 d |
| 124 M | 10.1 d | 101.1 d | 202.1 d |
| 350 M | 28.5 d | 285.3 d | 570.6 d |
| 1 B | 81.5 d | 815.1 d | 1,630 d |

### 5.3 Wall-clock, 30 % MFU

| Model | 1 B tokens | 10 B tokens | 20 B tokens |
|---|---|---|---|
| 30 M | **39.1 h** | 16.3 d | 32.6 d |
| 60 M | 3.3 d | 32.6 d | 65.2 d |
| 124 M | 6.7 d | 67.4 d | 134.8 d |
| 350 M | 19.0 d | 190.2 d | 380.4 d |
| 1 B | 54.3 d | 543.4 d | 1,087 d |

### 5.4 Wall-clock, 50 % MFU

| Model | 1 B tokens | 10 B tokens | 20 B tokens |
|---|---|---|---|
| 30 M | 23.5 h | 9.8 d | 19.6 d |
| 60 M | 46.9 h | 19.6 d | 39.1 d |
| 124 M | 4.0 d | 40.4 d | 80.9 d |
| 350 M | 11.4 d | 114.1 d | 228.2 d |
| 1 B | 32.6 d | 326.0 d | 652.1 d |

### 5.5 **What actually finishes in 24 hours**

| Model | Tokens in 24 h @ 20 % | @ 30 % | D/N ratio @ 20 % |
|---|---|---|---|
| **30 M** | **0.41 B** | 0.61 B | **13.6 ×** ← near Chinchilla-optimal |
| 60 M | 0.20 B | 0.31 B | 3.4 × (undertrained) |
| 124 M | 0.10 B | 0.15 B | 0.8 × (hopelessly undertrained) |
| 350 M | 0.04 B | 0.05 B | 0.1 × |
| 1 B | 0.01 B | 0.02 B | 0.0 × |

**Compute-optimal (D = 20 N) wall clock:** 30 M → 35 h @20 % / 23.5 h @30 %; 60 M → 5.9 d; 124 M → 25 d; 350 M → 200 d; 1 B → 4.5 years.

### 5.6 Cross-checks against §2 measurements (these validate 20 % MFU)

| Reported run | Implied compute | Implied MFU | Predicted on M4-base |
|---|---|---|---|
| mlxgpt d14 (135 M body params) **bf16, 4,530 tok/s on 2× M4 Pro** (2 × 9.2 = 18.4 TFLOPS) | 6 × 135e6 × 4,530 = **3.67 TFLOPS** | **20 %** | 4.26 × 0.20 / (6 × 135e6) = **1,050 tok/s** — matches my 124 M row (1,145) ✅ |
| mlxgpt d14 f32, 2,065 tok/s | 1.67 TFLOPS | 9 % | f32 + comms overhead |
| mlxgpt d20 (477 M), 284 tok/s | 0.81 TFLOPS | 4.4 % | comms-bound; f32 |
| nanoGPT_mlx, M3 Pro (~6.4 TFLOPS), 10.6 M model, 6,060 tok/s | 0.39 TFLOPS | 6 % | tiny model, launch-overhead bound |
| nanochat runcpu, M3 Max **MPS**, ~45,000 tok/s for 35.8 M scaling params | ~9.7 TFLOPS | ~60 % of an M3 Max | **implausibly high — the "30 minutes" claim is UNVERIFIED**; either the wall-clock is wrong or my param count is. Flagged for measurement. |

**Bottom line:** the 20 % column is the one to plan against, and the mlxgpt bf16 cross-check is a near-exact match for the 124 M row.

---

## 6. Memory math for 16 GB (11.84 GiB usable)

### 6.1 Parameter + optimizer state

| Regime | bytes/param | Max params in 8 GiB of state | in 10 GiB |
|---|---|---|---|
| Full FT: fp32 master + bf16 params/grads + fp32 Adam m,v (**classic "16 B/param"**) | 16 | **0.54 B** | 0.67 B |
| **Full FT: pure bf16 params + bf16 grads + bf16 Adam m,v** ← **MLX's actual default** | **8** | **1.07 B** | 1.34 B |
| Full FT: bf16 params/grads + 8-bit Adam m,v | 6 | 1.43 B | 1.79 B |
| Full FT: bf16 params/grads + **Adafactor** (factored 2nd moment ≈ 0) | ~4.1 | 2.10 B | 2.62 B |
| **LoRA** on bf16 frozen base (r=16 ≈ 0.5 % trainable) | ~2.08 | 4.13 B | 5.16 B |
| **QLoRA** on 4-bit frozen base (group-64 ⇒ 4.5 bits/param) | ~0.65 | 13.2 B | 16.5 B |

> **MLX-specific and important:** MLX optimizers allocate state with `mx.zeros_like(param)`, so **a bf16 model gets bf16 Adam moments for free** — 8 B/param, not 16. That is why full fine-tuning a ~1 B model is on the table here but not on a 16 GB CUDA card. `--optimizer adafactor` pushes that to ~2 B params. `--optimizer muon` is also available.

### 6.2 Activation memory (bf16)

Gradient checkpointing at layer granularity stores one `(B, S, d)` tensor per layer: `L·B·S·d·2` bytes. Without it, use Korthikanti's `34·L·B·S·d` bytes (flash-attention form).

| Model (L, d) | ckpt B=1,S=2048 | ckpt B=4,S=2048 | ckpt B=1,S=4096 | ckpt B=8,S=2048 | **no-ckpt** B=1,S=2048 |
|---|---|---|---|---|---|
| 124 M GPT-2 (12, 768) | 0.03 GiB | 0.14 | 0.07 | 0.28 | 0.60 |
| 350 M (24, 1024) | 0.09 | 0.38 | 0.19 | 0.75 | 1.59 |
| Qwen3-0.6B (28, 1024) | 0.11 | 0.44 | 0.22 | 0.88 | 1.86 |
| Qwen3-1.7B (28, 2048) | 0.22 | 0.88 | 0.44 | 1.75 | 3.72 |
| Llama-3.2-3B (28, 3072) | 0.33 | 1.31 | 0.66 | 2.63 | 5.58 |
| Qwen3-4B (36, 2560) | 0.35 | 1.41 | 0.70 | 2.81 | 5.98 |
| Llama-3-8B (32, 4096) | 0.50 | 2.00 | 1.00 | 4.00 | 8.50 |

### 6.3 ⚠️ The real memory hog: the logits tensor

For small models with huge vocabularies, `logits` + `logits.grad` dwarfs everything else. mlx-lm keeps logits in **bf16** (§1.1), so the table below is bf16 (`2 × B·S·V·2` bytes). Double these if your loss upcasts to fp32.

| Vocab | B=1,S=2048 | B=2,S=2048 | **B=4,S=2048 (mlx-lm default!)** | B=1,S=4096 |
|---|---|---|---|---|
| Qwen3 / Qwen3.5 (151,936) | 1.16 GiB | 2.32 | **4.64** | 2.32 |
| Gemma 3/4 (262,144) | 2.00 | 4.00 | **8.00** | 4.00 |
| Llama 3.x (128,256) | 0.98 | 1.96 | **3.91** | 1.96 |
| nanochat (65,536) | 0.50 | 1.00 | 2.00 | 1.00 |
| GPT-2 (50,257) | 0.38 | 0.77 | 1.53 | 0.77 |

**Practical consequences on 11.84 GiB:**

- **mlx-lm's stock defaults (`--batch-size 4 --max-seq-length 2048`) on a Gemma-4 model cost 8 GiB in logits alone.** Drop to `--batch-size 1` and use `--grad-accumulation-steps 4–8`.
- **`max_buffer_length = 8.88 GiB`** is a *single-allocation* cap. A B=4, S=2048 Gemma-4 logits tensor (4 GiB) is fine; a B=8 one (8 GiB) is at the edge.
- Gradient checkpointing is nearly free for small models and only matters at B≥4 or S≥4096 — matching the LORA.md note that it "will be more helpful for larger batch sizes or sequence lengths."

### 6.4 Worked budgets (11.84 GiB, leave ~1.5 GiB headroom → **~10 GiB**)

| Config | Weights | Opt. state | Acts (ckpt) | Logits+grad | **Total** | Verdict |
|---|---|---|---|---|---|---|
| **QLoRA 4-bit Qwen3.5-9B**, r=16, B=1, S=2048 | 5.95 | ~0.1 | 0.5 | 1.16 | **~7.7 GiB** | ✅ comfortable |
| QLoRA 4-bit Llama-3-8B, B=2, S=2048 | ~4.5 | 0.1 | 1.0 | 1.96 | ~7.6 | ✅ |
| LoRA bf16 Qwen3.5-2B, B=2, S=2048 | 4.43 | 0.1 | 0.44 | 2.32 | ~7.3 | ✅ |
| **Full FT bf16 Qwen3-0.6B**, bf16 Adam, B=1, S=2048, ckpt | 1.2 | 3.6 | 0.11 | 1.16 | **~6.1** | ✅ |
| **Full FT bf16 Qwen3-1.7B**, bf16 Adam, B=1, S=2048, ckpt | 3.4 | 10.2 | 0.22 | 1.16 | **~15.0** | ❌ OOM |
| Full FT bf16 Qwen3-1.7B, **Adafactor**, B=1, S=2048, ckpt | 3.4 | ~3.5 | 0.22 | 1.16 | **~8.3** | ✅ tight |
| Full FT bf16 1.0 B, bf16 Adam, B=1, S=2048, ckpt | 2.0 | 6.0 | 0.15 | 1.0 | **~9.2** | ✅ tight |
| Pretrain from scratch, 124 M, vocab 50 k, B=8×S=1024, ckpt | 0.25 | 0.75 | 0.14 | 1.53 | **~2.7** | ✅ lots of room |
| Pretrain from scratch, 350 M, vocab 65 k, B=8×S=2048, ckpt | 0.7 | 2.1 | 0.75 | 4.0 | **~7.6** | ✅ but 200 d of compute |

**Largest model you can fully fine-tune on this machine: ≈1.0–1.3 B params** (pure bf16 + bf16 Adam, batch 1, seq 2048, grad-checkpoint on), or **≈2 B with Adafactor**. **Largest QLoRA: 8–9 B at 4-bit.**

---

## 7. Local evaluation

### 7.1 Tooling status

| Tool | MLX support? | Notes |
|---|---|---|
| **`mlx_lm.evaluate`** ← **the answer** | ✅ **native** | mlx-lm ships an lm-eval integration: [`mlx_lm/evaluate.py`](https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/evaluate.py) `import lm_eval; from lm_eval.api.registry import register_model` — it registers an MLX `LM` class into the harness and uses `batch_generate` + `make_prompt_cache`. CLI: `mlx_lm.evaluate --model <repo> --tasks mmlu_pro gsm8k --num-shots N --batch-size N --limit N --apply-chat-template --fewshot-as-multiturn --max-tokens 8192 --output-dir .` Install with `pip install "mlx-lm[evaluate]"`. |
| **lm-evaluation-harness** | ❌ no built-in MLX backend | [EleutherAI/lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) — 13,961 ★, **v0.4.13 (2026-08-31)**. `lm_eval/models/` has huggingface, vllm, sglang, gguf, openai_completions, api_models, onnx, litellm, nemo, trtllm, watsonx, winml — **no `mlx.py`**. Use it *through* `mlx_lm.evaluate`, or point `--model local-completions` at `mlx_lm.server`. |
| [chimezie/lm-evaluation-harness-mlx](https://github.com/chimezie/lm-evaluation-harness-mlx) | community MLX module | alternative to `mlx_lm.evaluate`; less maintained — prefer the official path. |
| **lighteval** | ❌ no MLX backend | [huggingface/lighteval](https://github.com/huggingface/lighteval) — 2,539 ★, active (2026-09-09). `src/lighteval/models/` = transformers, vllm, sglang, nanotron, endpoints, custom, dummy. Reach it only via an **OpenAI-compatible endpoint** (`mlx_lm.server`) or `custom`. |
| **llama.cpp** | n/a | `llama-perplexity` for PPL; for task evals run `llama-server` and use lm-eval's `--model local-completions`. `llama-bench` for pp/tg throughput. |

**Apple validates this path themselves:** [`mlx_lm/BENCHMARKS.md`](https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/BENCHMARKS.md) publishes **MMLU-Pro scores produced by `mlx_lm.evaluate`** alongside throughput and memory — on a 64 GB M4 Max, mlx 0.29.2.dev, mlx-lm 0.28.2, macOS 26.1:

| Qwen3-4B-Instruct-2507 | MMLU-Pro | Prompt tok/s | Gen tok/s | Mem GB |
|---|---|---|---|---|
| bf16 | 64.05 | 1780.63 | 52.47 | 9.02 |
| q8 | 63.85 | 1606.57 | 86.91 | 5.25 |
| q6 | 63.53 | 1576.73 | 104.68 | 4.25 |
| q5 g32 | 63.16 | 1570.80 | 110.29 | 4.00 |
| q5 | 62.38 | 1584.33 | 116.39 | 3.86 |
| q4 g32 | **61.46** | 1610.03 | 126.00 | 3.60 |
| q4 | **60.72** | 1622.27 | 134.52 | 3.35 |

> **Quantization cost is real and measurable: bf16 → q4 loses 3.3 MMLU-Pro points; q4 g32 loses 2.6; q6 loses only 0.5.** For *evaluating* a fine-tune, use **q6 or q8**, not q4.

### 7.2 Benchmark cost (exact sample counts, from the HF datasets server)

| Benchmark | Split | **Samples** | Type | Cost on M4 |
|---|---|---|---|---|
| **HellaSwag** | validation | **10,042** (×4 endings = 40,168 scorings) | loglikelihood | cheap-ish, prompt-processing bound |
| **ARC-Challenge** | test | **1,172** (×~4 choices) | loglikelihood | **cheapest** |
| **MMLU-Pro** | test | **12,032** (10 options, CoT-generative in the standard harness) | **generative** | **most expensive** — use `--limit` |
| **GPQA Diamond** | — | **198** (gated dataset: `Idavidrein/gpqa`, `gated: auto` — must accept terms) | generative | cheap by count, long CoT |
| **GSM8K** | test | **1,319** | generative (8-shot CoT) | moderate |
| **MATH-500** | test | **500** | generative (long CoT) | moderate |
| **HumanEval** | test | **164** | generative + **code execution** (needs `--confirm-run-unsafe-code`) | cheap |
| **IFEval** | train | **541** | generative + programmatic constraint check | cheap |

Sources: [MMLU-Pro](https://huggingface.co/datasets/TIGER-Lab/MMLU-Pro), [GSM8K](https://huggingface.co/datasets/openai/gsm8k), [MATH-500](https://huggingface.co/datasets/HuggingFaceH4/MATH-500), [HumanEval](https://huggingface.co/datasets/openai/openai_humaneval), [IFEval](https://huggingface.co/datasets/google/IFEval), [ARC](https://huggingface.co/datasets/allenai/ai2_arc), [HellaSwag](https://huggingface.co/datasets/Rowan/hellaswag).

### 7.3 Throughput for evaluation on this M4 (what to expect)

Scaling the M4 Max BENCHMARKS.md table by GPU cores (10/40) and bandwidth (120/410 GB/s), and anchoring to llama.cpp #4167 M4 rows:

| Model (4-bit MLX) | Prompt-processing tok/s | Generation tok/s | Basis |
|---|---|---|---|
| ~0.6–1 B | ~1,200–2,500 | **~80–150** | llama.cpp gemma3-1B Q4_0 on M4 Pro = 3,564 pp / 148 tg ([#12985](https://github.com/ggml-org/llama.cpp/discussions/12985)); scale ~0.6× |
| **~3–4 B** | **~350–450** | **~35–50** | M4 Max q4 = 1,622 pp / 134.5 tg; M4 ≈ 0.25–0.33× |
| ~7–9 B | ~220 | **~20–25** | llama.cpp #4167 M4 Q4_0 = 221.29 pp / **24.11** tg (LLaMA-7B) |

Independent corroborations for M4 base 16 GB: Qwen3.5-9B MLX 4-bit **20.82 tok/s** ([@stevibe on X, 2026-03-03](https://x.com/_orcaman/status/2028749477519982739)), and a Mac-mini-M4-16GB report of **17.3 tok/s on Qwen3.5-35B-A3B** using `llama.cpp --mmap` with a 13 GB IQ3_XXS MoE quant ([thoughts.jock.pl](https://thoughts.jock.pl/p/local-llm-35b-mac-mini-gemma-swap-production-2026), [modelfit.io](https://modelfit.io/blog/run-35b-llm-mac-mini-m4-16gb-mmap)). *Note: the 35B-MoE-via-mmap trick works for inference only and is irrelevant to training.*

**Estimated eval wall-clock for a 4 B model in q6 on this M4 (UNVERIFIED — extrapolated):**

| Suite | Est. time |
|---|---|
| ARC-Challenge (1,172, loglikelihood) | **~5–10 min** |
| HellaSwag (10,042, loglikelihood) | ~45–90 min |
| IFEval (541, ~256 gen tokens) | ~1–1.5 h |
| HumanEval (164, ~400 gen tokens) | ~30–45 min |
| GSM8K (1,319, ~256 CoT tokens) | ~2.5–4 h |
| MATH-500 (500, ~1,000 CoT tokens) | ~4–6 h |
| GPQA Diamond (198, ~2,000 CoT tokens) | ~3–5 h |
| **MMLU-Pro (12,032, CoT)** | **~2–4 days** → always use `--limit 500–1000` |

**Recommended local eval command:**
```bash
uv pip install "mlx-lm[evaluate]"
mlx_lm.evaluate \
  --model mlx-community/Qwen3.5-4B-MLX-8bit \
  --tasks arc_challenge hellaswag ifeval gsm8k \
  --apply-chat-template --fewshot-as-multiturn \
  --batch-size 8 --limit 500 --output-dir ./evals
```
Do the expensive generative suites (MMLU-Pro, GPQA-D, MATH-500) with `--limit`, and report the limit alongside the score. **Never compare a `--limit`-ed score to a published full-set number.**

⚠️ Low-rigor source warning: the widely-cited [JMLab M5 Max benchmark pipeline](https://jmlab.net/projects/m5-macbook-benchmark-pipeline/) uses **5 prompts per benchmark** with an LLM-judge rubric, not lm-eval. Its throughput figures are usable; its "scores" are not comparable to anything.

---

## 8. Reference base models for local fine-tuning (as of 2026-09-13)

All sizes below are **measured** from the HF API (`safetensors` bytes). Licenses are from the HF model API.

### 8.1 Top candidates

| Model | HF id | Params | License | Released | 4-bit MLX size | MLX quant |
|---|---|---|---|---|---|---|
| **Qwen3.5-4B** | `Qwen/Qwen3.5-4B` (+`-Base`) | 4 B, 32 L, d=2560, **262 k ctx** | **Apache-2.0** | 2026-02-27 | **3.03 GB** | [`mlx-community/Qwen3.5-4B-MLX-4bit`](https://huggingface.co/mlx-community/Qwen3.5-4B-MLX-4bit) (also `-MLX-8bit`, `-OptiQ-4bit`, `-MTP-4bit`) |
| **Qwen3.5-2B** | `Qwen/Qwen3.5-2B` (+`-Base`) | 2 B, 24 L, d=2048, 262 k ctx | **Apache-2.0** | 2026-02-28 | **1.72 GB** (bf16: 4.43 GB) | `mlx-community/Qwen3.5-2B-MLX-4bit/-6bit/-8bit/-bf16` |
| **Qwen3.5-0.8B** | `Qwen/Qwen3.5-0.8B` | 0.8 B | **Apache-2.0** | 2026-02-28 | **0.63 GB** | `mlx-community/Qwen3.5-0.8B-4bit` |
| **Qwen3.5-9B** | `Qwen/Qwen3.5-9B` | 9 B | **Apache-2.0** | 2026-02-27 | **5.95 GB** | `mlx-community/Qwen3.5-9B-4bit` |
| **Gemma 4 E4B** | `google/gemma-4-E4B-it` | **8 B total / 4.5 B effective** (Per-Layer Embeddings), 128 k ctx | **Apache-2.0** ⭐ (Gemma 3 was under the restrictive Gemma license) | 2026-03-02 | **5.15 GB** | `mlx-community/gemma-4-e4b-it-4bit`, `-qat-4bit`, `-qat-mobile` |
| **Gemma 4 E2B** | `google/gemma-4-E2B-it` | ~2 B effective | **Apache-2.0** | 2026-03-02 | — | (E4B variants listed; E2B likely present) |
| **SmolLM3-3B** | `HuggingFaceTB/SmolLM3-3B` | 3 B | **Apache-2.0** | 2025-07-08 | **1.73 GB** | `mlx-community/SmolLM3-3B-4bit/-8bit/-bf16`, `-Base-4bit` |
| **Olmo-3-7B** | `allenai/Olmo-3-7B-Instruct`, `Olmo-3-1025-7B` (base) | 7 B | **Apache-2.0** | 2025-11-19 | **4.11 GB** | `mlx-community/Olmo-3-7B-Instruct-4bit`, `-mxfp4-QAT` (only ~138 downloads — low community validation) |
| Qwen3-4B-Instruct-2507 | `Qwen/Qwen3-4B-Instruct-2507` | 4 B | Apache-2.0 | 2025-08-05 | — | many; **has Apple's own published MMLU-Pro-vs-quant curve** (§7.1) |
| Qwen3-1.7B / 0.6B | `Qwen/Qwen3-1.7B`, `Qwen3-0.6B` | 1.7 B / 0.6 B | Apache-2.0 | 2025-04-27 | — | many |
| Llama-3.2-3B / 1B | `meta-llama/Llama-3.2-3B-Instruct` | 3 B / 1 B | **llama3.2** (community licence, gated) | 2024-09-18 | — | many |
| Phi-4-mini | `microsoft/Phi-4-mini-instruct` | 3.8 B | **MIT** | 2025-02-19 | — | yes |
| Granite 4.0 H Tiny | `ibm-granite/granite-4.0-h-tiny` | ~7 B MoE-hybrid | **Apache-2.0** | 2025-09-16 | — | yes |
| LFM2-1.2B / 2.6B | `LiquidAI/LFM2-1.2B`, `LFM2-2.6B` | 1.2 / 2.6 B | **`lfm1.0` custom — NOT OSI** ⚠️ | 2025-07 / 2025-09 | — | mlx-lm has `lfm2.py` |
| Nemotron Nano 9B v2 | `nvidia/NVIDIA-Nemotron-Nano-9B-v2` | 9 B | **nvidia-open-model-license** (custom) ⚠️ | 2025-08-12 | — | mlx-lm has `nemotron_h.py`. **Nemotron Nano "v3" does not exist on HF as of 2026-09-13** — v2 (9B/12B) is current. |
| Ministral 3B | `mistralai/Ministral-3B-Instruct-2410` | 3 B | **HF API returns 404 / no public weights** ⚠️ — Ministral 3B was API-only at launch | — | mlx-lm has `ministral3.py` |
| **gpt-oss-20b** | `openai/gpt-oss-20b` | 21 B (3.6 B active), MXFP4 | **Apache-2.0** | 2025-08-04 | see below | see below |

### 8.2 gpt-oss-20b on a 16 GB M4 — **does not fit for training; marginal for inference**

Measured `safetensors` sizes:

| Artifact | Size | vs. 11.84 GiB budget |
|---|---|---|
| `openai/gpt-oss-20b` native MXFP4 (3 shards) | **13.76 GB (12.82 GiB)** | ❌ over |
| `mlx-community/gpt-oss-20b-MXFP4-Q8` | **12.08 GB (11.25 GiB)** | ⚠️ weights alone ≈ the entire budget; no room for KV cache or activations |
| `mlx-community/gpt-oss-20b-MXFP4-Q4` | **11.18 GB (10.41 GiB)** | ⚠️ runnable for inference with a short context; **zero headroom for fine-tuning** |

`quantization_config.modules_to_not_convert` keeps attention, router, embeddings and lm_head in bf16 — that's why it's bigger than a naive 21 B × 4 bits. **Verdict: not a fine-tuning base on this machine.** (MoE models are also the documented LoRA-fusing failure case — [awni, mlx#1560](https://github.com/ml-explore/mlx/discussions/1560).)

### 8.3 Reported benchmark scores

**Qwen3.5-4B** ([model card](https://huggingface.co/Qwen/Qwen3.5-4B)) — hybrid *Gated DeltaNet + sparse MoE*, 262 k native ctx (→1.01 M extended):

| MMLU-Pro | MMLU-Redux | GPQA Diamond | IFEval | LiveCodeBench v6 | HMMT Feb-25 | HMMT Nov-25 | MMMU |
|---|---|---|---|---|---|---|---|
| **79.1** | 88.8 | **76.2** | 89.8 | 55.8 | 74.0 | 76.8 | 77.6 |

**Qwen3.5-2B** ([model card](https://huggingface.co/Qwen/Qwen3.5-2B)) — 24 L, d=2048, layout `6 × (3 × (Gated DeltaNet → FFN) → 1 × (Gated Attention → FFN))`:

| Mode | MMLU-Pro | MMLU-Redux | SuperGPQA | IFEval |
|---|---|---|---|---|
| Non-thinking | 55.3 | 69.2 | 30.4 | 61.2 |
| **Thinking** | **66.5** | 79.6 | 37.5 | — |

**Gemma 4 E4B-it** ([model card](https://huggingface.co/google/gemma-4-E4B-it), arXiv 2607.02770):

| MMLU-Pro | GPQA Diamond | AIME 2026 (no tools) | LiveCodeBench v6 | Codeforces ELO | BBEH | MMMLU | MMMU-Pro |
|---|---|---|---|---|---|---|---|
| 69.4 | 58.6 | 42.5 | 52.0 | 940 | 33.1 | 76.6 | 52.6 |

**Qwen3-4B-Instruct-2507 measured locally by Apple** (not vendor-reported): MMLU-Pro **64.05 bf16 / 60.72 q4** — see §7.1. This is the most trustworthy number in this section because it was produced by the exact toolchain you'd use.

⚠️ **Caveat on vendor numbers:** Qwen3.5-4B's GPQA-Diamond 76.2 and MMLU-Pro 79.1 are *thinking-mode, vendor-reported* figures. Treat them as an upper bound and re-measure with `mlx_lm.evaluate` before believing them.

⚠️ **Architecture risk:** Qwen3.5's Gated DeltaNet + MoE hybrid is newer than most of the MLX LoRA tooling. `mlx_lm/models/qwen3_5.py` and `qwen3_5_moe.py` exist, but MoE + LoRA fusing is a known rough edge. **Qwen3-1.7B / Qwen3-4B (plain dense) and SmolLM3-3B are the safest LoRA targets.**

---

## 9. What to do locally — recommendations

### 9.1 Pretrain-from-scratch target that finishes in <24 h

> **Train a 30–50 M parameter GPT on ~0.4–0.6 B tokens using [scasella/nanochat-mlx](https://github.com/scasella/nanochat-mlx) (or [a14a-org/mlxgpt](https://github.com/a14a-org/mlxgpt)'s single-node path), at nanochat `--depth 8`–`10`, `--max-seq-len 512–1024`, bf16 or mixed precision.**

Why:
- 30 M @ 20 % MFU = **4,733 tok/s → 0.41 B tokens in 24 h → D/N ≈ 13.6×**, i.e. **genuinely near-Chinchilla-optimal, not a toy**. Every larger size is hopelessly undertrained inside a day (60 M: 3.4×; 124 M: 0.8×).
- Memory is a non-issue (~2–3 GiB), so you can push batch size for MFU.
- Use **`nanochat`'s 65,536-vocab tokenizer or smaller** — a 151 k vocab costs you 1.16 GiB of logits and a large fraction of your FLOPs at this model size.
- **Lesson from mlxgpt, apply it directly:** pure bf16 is ~2× faster than f32 but **converges to a worse loss**; mixed precision is ~1.9× faster and hits the same ceiling as f32. `seq_len=1024` diverged at val 3.634 regardless of precision while `seq_len=512` reached 3.506 cleanly. **Start at seq 512 + mixed precision.** Their other big win: *precomputed masks + cached RoPE + fused QK scale gave 6× throughput (500 → 3,000 tok/s)*.
- **Do NOT target GPT-2 (d24–d26) grade.** That's ~4.5e18 FLOPs ≈ **130–160 days** on this machine, vs. **~3 h / $72 on 8×H100**.

**Stretch goal if you'll run it for a week:** 124 M on ~1 B tokens = **10 days @ 20 % MFU**. Only worth it as a "24/7 background job" narrative, not as a model.

### 9.2 Best base model for local fine-tuning

> **Primary: `Qwen/Qwen3-1.7B` or `mlx-community/Qwen3.5-2B-MLX-4bit` for LoRA; `Qwen/Qwen3-0.6B` (mlx-lm's own default) if you want FULL fine-tuning.**

| Goal | Pick | Command shape |
|---|---|---|
| **Full fine-tune** (own the whole model) | **Qwen3-0.6B bf16** (~6.1 GiB total, fits easily) | `mlx_lm.lora --model Qwen/Qwen3-0.6B --fine-tune-type full --optimizer adamw --batch-size 1 --grad-accumulation-steps 8 --max-seq-length 2048 --grad-checkpoint --train` |
| **Best quality that still fits comfortably** | **Qwen3.5-2B 4-bit QLoRA** (1.72 GB weights) or **SmolLM3-3B** (Apache-2.0, 1.73 GB @4-bit, fully-open training recipe) | `mlx_lm.lora --model mlx-community/Qwen3.5-2B-MLX-4bit --train --batch-size 2 --num-layers 16 --grad-checkpoint` |
| **Max capability** | **Qwen3.5-9B 4-bit QLoRA** (5.95 GB, ~7.7 GiB total) — ~20–25 tok/s | `--batch-size 1 --num-layers 8 --grad-accumulation-steps 8` |
| **Most permissive + fully reproducible** | **Olmo-3-7B (Apache-2.0, open data + recipe)** at 4-bit (4.11 GB) — but only 138 MLX-quant downloads, so expect to hit bugs first | — |
| **DPO/ORPO/GRPO** | switch to `mlx-lm-lora` | `mlx_lm_lora.train --training-mode dpo --dpo-loss-type sigmoid ...` |

**Avoid on this machine:** gpt-oss-20b (13.8 GB, doesn't fit), Gemma 3 (restrictive licence — Gemma **4** is Apache-2.0, use that), Ministral 3B (no public weights), LFM2 / Nemotron (custom non-OSI licences), Llama 3.2 (gated + community licence).

**Always-on settings for 16 GB:** `--batch-size 1` + `--grad-accumulation-steps 4–8` (never the default `4`), `--max-seq-length 2048` or less, `--grad-checkpoint`, `--num-layers 8–16`, `--clear-cache-threshold` set to a few GB. Close Chrome — its on-device model alone has historically taken 4 GB per profile.

### 9.3 Best eval path

> **`mlx_lm.evaluate` (mlx-lm's built-in lm-evaluation-harness backend). Nothing else is close.**

```bash
uv pip install "mlx-lm[evaluate]"
# fast regression suite, run after every fine-tune (~1.5-2.5 h total)
mlx_lm.evaluate --model <your-model-or-adapter-fused> \
  --tasks arc_challenge hellaswag ifeval gsm8k \
  --apply-chat-template --fewshot-as-multiturn --batch-size 8 --limit 500 \
  --output-dir ./evals
# quarterly / headline runs, sampled
mlx_lm.evaluate --model ... --tasks mmlu_pro gpqa_diamond math_500 humaneval \
  --limit 300 --max-tokens 4096 --confirm-run-unsafe-code --output-dir ./evals
```

Rules of thumb:
1. **Evaluate at q6 or q8, not q4** — Apple's own table shows q4 costs 3.3 MMLU-Pro points vs. bf16, q6 costs 0.5. You will otherwise measure your quantizer, not your fine-tune.
2. **Cheap-first ordering:** ARC-Challenge → IFEval → HumanEval → GSM8K → MATH-500 → GPQA-D → HellaSwag → MMLU-Pro.
3. **Always report `--limit`** next to the score.
4. GPQA is a **gated** dataset — accept terms on [`Idavidrein/gpqa`](https://huggingface.co/datasets/Idavidrein/gpqa) and `hf auth login` first.
5. Secondary path if you need llama.cpp/GGUF: `llama-server` + `lm_eval --model local-completions`. `lighteval` has no MLX backend and buys you nothing here.

### 9.4 Framework choice: **MLX, not PyTorch MPS**

MLX wins on every axis that matters here: it beats PyTorch MPS on M-series ([arXiv 2501.14925](https://arxiv.org/html/2501.14925v2)), it has a first-party trainer with QLoRA + grad-checkpointing + Muon/Adafactor, bf16 optimizer states halve your memory, and it has a native lm-eval bridge. PyTorch MPS's blockers for *training* are structural, not cosmetic: `device_count()==1`, no native collectives (gloo can't touch MPS tensors), no fp64, no upstream flash-attention in SDPA, and silent CPU fallbacks for missing ops. Install PyTorch only if you need `scripts.convert_from_hf` to import a checkpoint, or want to run `nanochat`'s `runs/runcpu.sh` as a cross-check.

### 9.5 Do not build a Mac cluster for this

Base M4 Mac mini = **Thunderbolt 4**. RDMA/JACCL requires **Thunderbolt 5** (M4 Pro or better) and macOS 26.2+. Without it you're on the `ring` TCP backend at ~300 µs latency, where gradient all-reduce would eat most of a 124 M model's step time. If Sammy ever wants a real Apple-Silicon training cluster, the entry ticket is **2× M4 Pro mini (TB5) minimum** — which is exactly what [mlxgpt](https://github.com/a14a-org/mlxgpt) uses, and even that gets 477 M params at 284 tok/s.

---

## 10. Open items marked UNVERIFIED

| Claim | Why unverified | How to settle |
|---|---|---|
| scasella/nanochat-mlx's "depth 12 ≈ 1 h, depth 20 ≈ 8 h, depth 26 ≈ 24 h on M3 Pro" | Contradicts FLOPs math by ~40× given its own `target_param_data_ratio=12.0` default | Run `python -m scripts.train --depth=4 --num-iterations=50` and measure tok/s |
| nanochat `runs/runcpu.sh` "~30 min on M3 Max" ⇒ ~45,000 tok/s ⇒ ~60 % MFU on MPS | Implausibly high for PyTorch MPS | Run the script on this M4 with `NANOCHAT_DTYPE=bfloat16` and time 100 iterations |
| M4-base LoRA/QLoRA tok/s figures in §1.3 | Extrapolated from M3 Ultra / M4 Pro by core count | `mlx_lm.lora --model Qwen/Qwen3-0.6B --train --iters 50` and read the reported tok/s |
| Eval wall-clock estimates in §7.3 | Extrapolated from M4 Max | `mlx_lm.evaluate --tasks arc_challenge --limit 50` and extrapolate |
| Gemma 4 being Apache-2.0 | HF API `license` field says `apache-2.0` for `google/gemma-4-E2B-it` and `-E4B-it`; a big departure from the Gemma licence | Read the LICENSE file in the repo before shipping anything |
| `mlx_lm.lora` LoRA working on Qwen3.5's Gated-DeltaNet/MoE hybrid | `qwen3_5.py` exists but no training reports found | 20-iteration smoke test |
| `mlx-community/gemma-4-E2B-it-*` MLX quant existence | Only E4B variants surfaced in the search | `huggingface-cli search` |

