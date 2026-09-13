# The Frontier LLM Recipe (2026)

**A stage-by-stage, sourced reconstruction of how frontier-class language models are actually built, as of September 2026.**

Compiled 2026-09-13. Every factual claim carries a URL. Claims I could not confirm against a primary source are marked **UNVERIFIED**.

---

## How to read this document

Each stage has:

- **2026 default choice** — what you should do if you have no reason to do otherwise.
- **Key hyperparameters** — concrete numbers from primary sources.
- **Why / evidence** — the ablation or measurement that justifies it, with the numbers.

**Appendix A is the actionable summary** if you have 8–64 GPUs rather than 100,000; several stages also carry inline small-compute notes. **Appendix C lists what the field actively disagrees about** — do not read any "default choice" there as settled.

**Source hierarchy used:** official technical report PDFs and arXiv papers first; official model cards and lab blogs second; third-party analysis only where labeled. Secondary/SEO sources were discarded.

### A caution about this document's shelf life

The field moved substantially between the 2024–2025 reports that most "how LLMs are trained" write-ups are based on and the 2026 reports. Four things that were *defaults* in 2025 are *no longer defaults* in 2026:

1. **AdamW as the base optimizer** → Muon variants now used by DeepSeek-V4, Kimi K3, Qwen3.8.
2. **Pure full attention (MLA/GQA)** → hybrid linear/sparse attention stacks.
3. **A single plain residual stream** → widened/attentive residual streams (mHC, Gated Residual, AttnRes).
4. **One model trained end-to-end** → *specialist RL experts distilled into one generalist*.

---

## Stage 0: The 2026 landscape (who published what)

These are the primary sources this recipe is distilled from. Bold = a 2026 report that changes a 2025 default.

| Model | Released | Total / Active | Report |
|---|---|---|---|
| DeepSeek-V3 | Dec 2024 | 671B / 37B | [arXiv:2412.19437](https://arxiv.org/abs/2412.19437) |
| DeepSeek-R1 | Jan 2025 | 671B / 37B | [arXiv:2501.12948](https://arxiv.org/abs/2501.12948) |
| Llama 3 herd | Jul 2024 | 405B dense | [arXiv:2407.21783](https://arxiv.org/abs/2407.21783) |
| Llama 4 Scout / Maverick | Apr 2025 | 109B/17B, 400B/17B | [ai.meta.com blog](https://ai.meta.com/blog/llama-4-multimodal-intelligence/) |
| Gemma 3 | Mar 2025 | 1B–27B dense | [arXiv:2503.19786](https://arxiv.org/abs/2503.19786) |
| Qwen3 | May 2025 | 235B / 22B | [arXiv:2505.09388](https://arxiv.org/abs/2505.09388) |
| Kimi K2 | Jul 2025 | 1.04T / 32.6B | [arXiv:2507.20534](https://arxiv.org/abs/2507.20534) |
| gpt-oss-120b/20b | Aug 2025 | 116.8B/5.1B, 20.9B/3.6B | [arXiv:2508.10925](https://arxiv.org/abs/2508.10925) |
| GLM-4.5 | Aug 2025 | 355B / 32B | [arXiv:2508.06471](https://arxiv.org/abs/2508.06471) |
| SmolLM3 | Jul 2025 | 3B dense | [HF blog](https://huggingface.co/blog/smollm3) |
| **Olmo 3** (7B/32B) | Nov 2025 (v2 Apr 2026) | dense | [arXiv:2512.13961](https://arxiv.org/abs/2512.13961) |
| MiniMax-M2 | Oct 2025 | 229.9B / 9.8B | [arXiv:2605.26494](https://arxiv.org/abs/2605.26494) · [lmsys analysis](https://www.lmsys.org/blog/2025-11-04-miminmax-m2/) |
| **DeepSeek-V4-Flash / -Pro** | paper Apr 2026; weights Aug 2026 | 284B/13B, 1.6T/49B | [arXiv:2606.19348](https://arxiv.org/abs/2606.19348) |
| **Olmo Hybrid** 7B | Apr 2026 | dense hybrid | [arXiv:2604.03444](https://arxiv.org/abs/2604.03444) |
| **Qwen3.8-Flash-Next** | Aug 2026 | 125B/6B + 51B n-gram + 4B MTP | [arXiv:2608.30320](https://arxiv.org/abs/2608.30320) · [tech report PDF](https://github.com/QwenLM/Qwen3.8-Flash-Next/blob/main/tech_report.pdf) |
| **Kimi K3** | Jul 2026 (weights Jul 27) | 2.78T / 104.2B | [arXiv:2607.24653](https://arxiv.org/abs/2607.24653) · [tech report PDF](https://github.com/MoonshotAI/Kimi-K3/blob/main/k3_tech_report.pdf) |

Also released and referenced below: DeepSeek-V3.1 / V3.2-Exp (DSA sparse attention), DeepSeek-V4.1-Flash (763B, Sep 2026, [HF](https://huggingface.co/deepseek-ai)), Kimi K2.5 / K2.6 / K2.7-Code, Qwen3.5, **Qwen3.7-Plus (397B-A17B, the previous Qwen flagship)** and Qwen3.8-27B, GLM-5.1 / GLM-5.2.

**Frontier proprietary reference points as of Sept 2026** (from Kimi K3's own comparison table): Claude Fable 5, GPT-5.6 Sol, GPT-5.5, Claude Opus 4.8, GLM-5.2. Kimi K3 states it "still trails the most powerful proprietary models, namely Claude Fable 5 and GPT-5.6 Sol" ([K3 report](https://github.com/MoonshotAI/Kimi-K3/blob/main/k3_tech_report.pdf), abstract).

---

# Stage 1: Architecture

## 1.1 Dense vs MoE

> **2026 default choice: ultra-sparse fine-grained MoE with 1–2 shared experts, aux-loss-free load balancing, and a couple of dense layers at the bottom.** Dense only below ~10B, or when you need one artifact that runs anywhere.

### The sparsity trend is the single clearest 2024→2026 movement

| Model | Total | Active | Active % | Routed experts | Active experts | Shared |
|---|---|---|---|---|---|---|
| DeepSeek-V3 (2024) | 671B | 37B | 5.5% | 256 | 8 | 1 |
| Kimi K2 (2025) | 1.04T | 32.6B | 3.1% | 384 | 8 | 1 |
| GLM-4.5 (2025) | 355B | 32B | 9.0% | 160 | 8 | 1 |
| gpt-oss-120b (2025) | 116.8B | 5.1B | 4.4% | 128 | 4 | 0 |
| MiniMax-M2 (2025) | 229.9B | 9.8B | 4.3% | 256 | 8 | 0 (UNVERIFIED) |
| **DeepSeek-V4-Flash (2026)** | 284B | 13B | 4.6% | 256 | 6 | 1 |
| **DeepSeek-V4-Pro (2026)** | 1.6T | 49B | 3.1% | 384 | 6 | 1 |
| **Qwen3.8-Flash-Next (2026)** | 125B | 6B | 4.8% | 512 | 10 | 1 |
| **Kimi K3 (2026)** | 2.78T | 104.2B | 3.7% | 896 | 16 | 2 |

Sources: [DeepSeek-V3 §4.2](https://arxiv.org/abs/2412.19437); [Kimi K2 §2.3](https://arxiv.org/abs/2507.20534); [GLM-4.5 Table](https://arxiv.org/abs/2508.06471); [gpt-oss §2.2](https://arxiv.org/abs/2508.10925); [DeepSeek-V4 §4.2.1](https://arxiv.org/abs/2606.19348); [Qwen3.8-Flash-Next model card](https://huggingface.co/Qwen/Qwen3.8-Flash-Next); [Kimi K3 Table 1](https://github.com/MoonshotAI/Kimi-K3).

**Kimi K2 published the sparsity scaling law that justifies this.** Sparsity is defined as **total experts ÷ activated experts**. Under a fixed number of activated parameters — i.e. constant FLOPs — increasing the total number of experts "consistently lowers both the training and validation loss." At a fixed validation loss of 1.5, **sparsity 48 reduces FLOPs by 1.69×, 1.39×, and 1.15× compared to sparsity levels 8, 16, and 32** respectively. K2 chose sparsity 48 (8 of 384) as the point where the gain stopped justifying the infrastructure complexity ([Kimi K2 §2.3](https://arxiv.org/html/2507.20534v1)).

A detail worth copying: K2 **cut attention heads from V3's 128 to 64 in order to afford the sparsity**, because at 128K sequence length going from 64 to 128 heads at a fixed 384 experts costs **+83% inference FLOPs**. Sparsity and head count trade against each other in the inference budget.

**But the sparsity race is theoretically contested once memory is priced in.** "Towards Principled Design of MoE LMs under Memory and Inference Constraints" ([arXiv:2601.08215](https://arxiv.org/abs/2601.08215), Jan 2026), using the same sparsity definition, reaches the opposite prescription:

> "n_exp and n_topk do **not** 'cancel out' within the sparsity ratio; instead, a larger total number of experts **slightly penalizes** performance by forcing a reduction in core model dimensions (depth and width) to meet memory constraints. This motivates a simple principle for MoE design which **maximizes N_total while minimizing s (maximizing n_topk) and n_exp** under the given constraints."

Apple's "Parameters vs FLOPs" ([arXiv:2501.12370](https://arxiv.org/abs/2501.12370)) lands in between: loss decreases with sparsity at **all** compute budgets tested (3e19–1e21 FLOPs) with "no diminishing effect… as we increase training compute budget," but the *optimal* sparsity is constraint-dependent — an interior optimum under a memory constraint, monotone under a compute constraint. **Note their convention is inverted** (sparsity = fraction of *inactive* params, so Kimi's 48 ≈ 0.979 in Apple's notation); do not mix the two.

**The 2026 extreme is DeepSeek-V4-Pro at sparsity 64** (6 of 384), with **Kimi K3 at 56** (16 of 896). The trend is roughly 2× in 18 months: **32 (DeepSeek-V3, late 2024) → 48 (Kimi K2, mid-2025) → 56–64 (Kimi K3 / DeepSeek-V4-Pro, 2026)**. Both 2026 labs needed new machinery to get there — K3 via **Stable LatentMoE** plus Quantile Balancing (below), V4 via **FP4 expert weights** plus compressed attention.

The academic backing for fine-grained experts is **"Scaling Laws for Fine-Grained Mixture of Experts"** ([arXiv:2402.07871](https://arxiv.org/abs/2402.07871)), which introduces **granularity** as an explicit scaling variable and shows that with granularity tuned, MoE beats dense at every compute budget — contradicting the earlier belief that MoE loses its advantage at large budgets. Their fitted law puts granularity **inside** the parameter term, so its benefit does not vanish with scale:

```
L(N, D, G) = c + (g/G^γ + a)·N^(-α) + b·D^(-β)
   a=18.1, α=0.115, b=30.8, β=0.147, g=2.1, γ=0.58, c=0.47
```

Because γ = 0.58 > 0, **optimal granularity rises with compute** — their own table runs G=8 at small budgets to **G=64 at 1e25 FLOPs**. And the dense-vs-MoE gap *widens*: a compute-optimal MoE at **1e20 FLOPs** matches a dense transformer trained with **20×** the compute, rising to **over 40× beyond 1e25 FLOPs**. Kimi K2's sparsity law is the production-scale confirmation.

### LatentMoE (Kimi K3, 2026) — the trick that makes extreme sparsity affordable

In a conventional MoE, every selected expert receives the full `d`-dimensional token vector, so communication and expert-weight traffic grow with routing multiplicity. **LatentMoE separates model width from routed-expert width**: shared experts keep a full-width path; routed experts operate in a compact latent space of width `ℓ`. Kimi K3 sets `d = 7168`, latent MoE dimension `ℓ = 3584` (0.5×), expert hidden dim 3072 ([Kimi K3 §2.3, Table 1](https://github.com/MoonshotAI/Kimi-K3)):

```
u = Σ_{i∈Top-k(x)} p_i · E_i^routed(W↓ x)          # routed path, in R^ℓ
y = Σ_j E_j^shared(x) + W↑ RMSNorm(u)               # shared path full-width
```

Two failure modes appear at this sparsity, and K3 names both fixes:

1. **Exploding activations in the routed branch** (the routed path composes `W↓`, a gated multi-branch FFN, and `W↑` into ~4 consecutive matmuls — ill-conditioned at 2.8T scale). Fixes: an **RMSNorm between expert aggregation and the up-projection** (which "consistently improves validation loss and downstream benchmarks", not just stability), plus a bounded activation (§1.6).
2. **Load balancing breaks down near 10³ experts** — the fixed-step aux-loss-free bias update is no longer well behaved. Fix: **Quantile Balancing** (§1.2).

### Bottom dense layers and hash routing

- DeepSeek-V3 replaces all FFNs with MoE **except the first three layers** ([§4.2.1](https://arxiv.org/abs/2412.19437)).
- DeepSeek-V4 instead uses **MoE in all blocks but applies Hash routing for the first 3 MoE layers** ([§4.2.1](https://arxiv.org/abs/2606.19348)).
- GLM-4.5 uses 3 dense layers + 89 MoE layers ([Table](https://arxiv.org/abs/2508.06471)).
- Kimi K2 and K3 both use **1 dense layer** ([K3 Table 1](https://github.com/MoonshotAI/Kimi-K3)).

### Depth vs width

**GLM-4.5 explicitly reports choosing depth over width** and, unusually, publishes a case where loss and benchmarks disagree:

> "we reduce the width (hidden dimension and number of routed experts) of the model and increase its height (number of layers)" … "we utilize 2.5 times more attention heads (96 heads for a 5120 hidden dimension). Counterintuitively, while this increased head count **did not improve training loss**, it consistently **delivered improved performance on reasoning benchmarks** such as MMLU and BBH."
> — [GLM-4.5 §2](https://arxiv.org/abs/2508.06471)

This is a recurring 2026 theme: **loss and downstream accuracy do not always move together**, and labs increasingly report both. Qwen3.8 makes the same point about n-gram vocabulary scaling (§1.8).

Kimi K3 scaled depth hard: 61 → **93 layers** (+52%) from K2 to K3, at identical hidden dimension 7168 ([K3 Table 1](https://github.com/MoonshotAI/Kimi-K3)).

## 1.2 Load balancing

> **2026 default choice: auxiliary-loss-free bias routing with sigmoid gates, plus a tiny sequence-wise balance loss (α ≈ 1e-4) purely as a guard against within-sequence collapse. Freeze the bias at the end of training and at inference.**

**The canonical recipe (DeepSeek-V3, still used by V4 and GLM-4.5):**

- Router computes `s_i = Sigmoid(W_r x_i)`; a per-expert bias `b_j` is **added only for Top-k selection**, not to the mixture weights — so it regulates dispatch without touching gradients.
- Bias update rule: `b_j ← b_j + γ·sign(mean_load − load_j)`.
- **γ (bias update speed) = 0.001**, set to **0.0 for the final 500B tokens** (DeepSeek-V3: first 14.3T at 0.001, last 500B at 0.0) ([§4.2.2](https://arxiv.org/abs/2412.19437)).
- **Sequence-wise balance loss α = 0.0001** — "just to avoid extreme imbalance within any single sequence."
- **Node-limited routing M = 4** (each token routed to at most 4 nodes) in V3; **DeepSeek-V4 removed the node-count constraint entirely** ([V4 §2.1](https://arxiv.org/abs/2606.19348)).

DeepSeek-V4 keeps γ = 0.001 and α = 0.0001 unchanged, but **changes the affinity-score activation from `Sigmoid(·)` to `Sqrt(Softplus(·))`** ([V4 §2.1](https://arxiv.org/abs/2606.19348)).

GLM-4.5 uses the same numbers: loss-free bias rate 0.001 for the first 15T tokens then 0.0; sequence-level balance loss to avoid within-sequence imbalance ([§2.3](https://arxiv.org/abs/2508.06471)).

### Quantile Balancing (Kimi K3, 2026) — for ~10³ experts

The fixed-step `γ·sign(...)` rule trades slow adaptation against load oscillation, and at 896 experts per layer it is no longer adequate. **Quantile Balancing sets each expert's bias directly from the router-score quantile that matches its target load**, derived in a single forward pass:

- Route with **Top-(k+1)** instead of Top-k on the biased score. The first k entries are the routes actually taken; the (k+1)-th is the **cutoff** `α_i` an expert must exceed to enter token *i*'s Top-k. This avoids needing a separate token-side quantile.
- Target load per expert is `q = mk/n` for a batch of `m` tokens, `n` experts.
- Update: `b̂_j ← −quantile_{1−k/n}(s_{:,j} − α)`, then `b ← b̂ − mean(b̂)·1` to remove a common offset that would not change Top-k.
- Applied **next step only** ("a batch is never routed with a bias derived from itself"), and **frozen at inference**.
- At scale the exact quantile over millions of margins across ranks is infeasible, so K3 reads it from a **per-expert histogram**: one all-reduce sums per-rank bin counts, and the quantile is recovered from pooled counts. Cost is "a few hundred bins per expert," and because counts are additive the estimate reflects the true global batch regardless of sharding.

— [Kimi K3 §2.3.3](https://github.com/MoonshotAI/Kimi-K3)

### The 2026 stability discovery: decouple the router from the backbone

DeepSeek-V4 reports that trillion-parameter MoE loss spikes are **consistently tied to outliers in MoE layers, with the routing mechanism itself exacerbating them**, and that rollbacks do not prevent recurrence. Their fix, **Anticipatory Routing**:

> "at step *t*, we use the current network parameters θ_t for feature computation, but the routing indices are computed and applied using the historical network parameters θ_{t−Δt}."

They prefetch step *t*'s data at step *t−Δt* to compute and cache routing indices, avoiding loading parameters twice. They explicitly note they do not have a theoretical explanation and are "sharing them openly to foster further exploration" ([V4 §4.2.3](https://arxiv.org/abs/2606.19348)).

**Small-compute note:** aux-loss-free bias routing is ~15 lines of code and strictly better than the classic auxiliary load-balancing loss (which adds an interference gradient). Use it at any scale. Quantile Balancing only matters above ~256 experts.

## 1.3 Attention

> **2026 default choice: a hybrid stack. 3 cheap layers (linear-attention or sliding-window) to every 1 global layer, with the last layer always global. Add sparse top-k selection to the global layers if you need ≥256K context.**

This is the biggest architectural change of 2026 and every major lab converged on the same 3:1 ratio independently.

| Model | Cheap layer type | Global layer type | Ratio | Note |
|---|---|---|---|---|
| Gemma 3 (2025) | sliding window, 1024 | full, RoPE θ=1M | **5:1** | [§2.1](https://arxiv.org/abs/2503.19786) |
| gpt-oss (2025) | banded window, 128 | full dense | **1:1** alternating | [§2.2](https://arxiv.org/abs/2508.10925) |
| Olmo 3 (2025) | sliding window, 4096 | full, YaRN | **3:1** | last layer always full ([§3.2](https://arxiv.org/abs/2512.13961)) |
| Qwen3-Next / **Qwen3.8** | Gated DeltaNet (linear) | full → QSA sparse | **3:1** | [tech report §2.1.1](https://github.com/QwenLM/Qwen3.8-Flash-Next) |
| **Olmo Hybrid** (2026) | Gated DeltaNet | full | **3:1** | replaces Olmo 3's SWA layers ([arXiv:2604.03444](https://arxiv.org/abs/2604.03444)) |
| **Kimi K3** (2026) | KDA (linear) | Gated MLA, NoPE | **3:1** | 69 KDA + 24 MLA, extra MLA at the end |
| **DeepSeek-V4** (2026) | CSA (sparse, compress 4) | HCA (compress 128) | interleaved | both compressed; no full attention at all |

### The empirical case for linear-hybrid over SWA-hybrid

Qwen3.8 ran the controlled three-way ablation at 25B-A3B, 400B tokens @ 4K + 80B @ 32K, same eval pipeline. Average over nine benchmarks (MMLU, MMLU-Pro, SuperGPQA, MATH, GSM8K, BBH, MMMLU, EvalPlus, MultiPL-E):

| Architecture | Avg. |
|---|---|
| Full attention | 49.87 |
| SWA hybrid (window 128, 1-in-4 full) | 51.15 |
| **GDN hybrid (1-in-4 full)** | **53.81** |

"The GDN hybrid improves over the Transformer on eight of the nine selected benchmarks and exceeds the SWA hybrid on seven." — [Qwen3.8 §2.1.1, Table 1](https://github.com/QwenLM/Qwen3.8-Flash-Next)

**Olmo Hybrid independently confirms this in the fully-open setting.** It is "a 7B-parameter model largely comparable to Olmo 3 7B but with the sliding window layers replaced by Gated DeltaNet layers," and it "outperforms Olmo 3 across standard pretraining and mid-training evaluations." The scaling-law result is the important one: **the hybrid scales significantly more efficiently than the transformer** — on MMLU it reaches Olmo 3's accuracy with **49% fewer tokens (~2× data efficiency)**. The paper's theoretical contribution is that hybrids "do not merely inherit the expressivity of transformers and linear RNNs, but can express tasks beyond both, such as code execution," and argues that greater expressivity is *why* scaling improves ([arXiv:2604.03444](https://arxiv.org/abs/2604.03444); [Ai2 blog](https://allenai.org/blog/olmohybrid)).

### The counter-case: MiniMax-M2 went *back* to full attention

This is the most important dissent and it should be read before committing to a linear hybrid. MiniMax-M1 used lightning attention; **M2 (229.9B/9.8B, 62 layers, GQA 48Q/8KV, QK-Norm, partial RoPE, 196,608 context, 29.2T tokens) reverted to full attention.** Their stated reasons:

- **Benchmarks hid the regression.** Hybrid models "performed on par with pure full attention models on standard leaderboards" but had "clear shortcomings in complex, multi-hop reasoning tasks" that only appeared at scale. Their own ablation: full attention and SWA both scored **99 on RULER @32K**, but at **128K RULER it was 90 (full) vs 72 (SWA)** — an 18-point gap — and full attention led by **15 and 17.6 points** on two many-to-one translation evals.
- **Mixed supervised results**: SWA won on IFBench and agent tasks; full attention won on GPQA Diamond and SWE-bench Verified.
- **Linear attention is often memory-bound even during training**, so the theoretical FLOP win does not materialize.
- **Missing system integration**: no native prefix caching (real dialogues have very high cache-hit rates; coding agents reuse the same repo context across rollouts), no clear speculative-decoding path, and much higher sensitivity to low-precision KV storage.
- Their SWA-hybrid experiments failed "across multiple experimental dimensions" (ratio, RoPE, layer configuration) despite pretraining for "hundreds of billions (even trillions) of tokens."

MiniMax explicitly frames this as pragmatic, not a rejection: efficient attention will win "as GPU compute growth slows and context lengths continue to increase," but needs all three of evaluation, data, and infrastructure to mature.
— [LMSYS deconstruction](https://www.lmsys.org/blog/2025-11-04-miminmax-m2/); [Raschka's report notes](https://sebastianraschka.com/blog/2026/minimax-m2-technical-report.html)

**How to reconcile the two camps:** MiniMax's failures were with *sliding-window* hybrids, which Qwen3.8's ablation also ranks below GDN hybrids. The 2026 winners (Kimi K3, Qwen3.8, Olmo Hybrid) all use **gated-delta-rule linear attention**, which has a data-dependent *erase-and-write* state update rather than a fixed-size window, and all keep a **periodic full/global layer**. The infrastructure objections (prefix caching, spec decoding, precision) remain real and are exactly what Kimi K3 spends its infrastructure section solving.

### MLA (Multi-head Latent Attention)

Still the default for the global layers when you want a small KV cache without sparsity. DeepSeek-V3's exact configuration ([§4.2.1](https://arxiv.org/abs/2412.19437)):

- 61 layers, hidden 7168, `n_h` = 128 heads, per-head dim `d_h` = 128
- **KV compression dim `d_c` = 512**; **query compression dim `d_c'` = 1536**
- **Decoupled RoPE head dim `d_h^R` = 64** — RoPE is applied to a *separate* small key/query slice, because you cannot cache a rotated compressed latent
- Extra RMSNorm after the compressed latents; extra scaling factors at the width bottlenecks
- Init std 0.006 for all learnable parameters

Kimi K2 and K2.5 use MLA; **Kimi K3 keeps MLA only in its 1-in-4 global layers, adds a full-rank input-dependent output gate, and removes positional encoding entirely from them** (§1.4).

### Sparse attention: DSA → CSA/HCA

DeepSeek's line runs NSA ([arXiv:2502.11089](https://arxiv.org/abs/2502.11089)) → DSA (V3.2-Exp) → **CSA + HCA (V4)**. V3.2-Exp's claim is that "DeepSeek Sparse Attention (DSA) achieves fine-grained sparse attention for the first time, delivering substantial improvements in long-context training and inference efficiency while maintaining virtually identical model output quality," with benchmark parity vs V3.1-Terminus ([HF card](https://huggingface.co/deepseek-ai/DeepSeek-V3.2-Exp)). Exact DSA top-k and continued-training token counts: **UNVERIFIED** from the model card; see the V3.2 tech report.

**DeepSeek-V4 (2026) replaced all dense attention with two compressed variants** ([§2.3, §4.2.1](https://arxiv.org/abs/2606.19348)):

**Compressed Sparse Attention (CSA)**
- Compresses `m` tokens → 1 KV entry (**m = 4**), then applies sparse top-k selection over the compressed entries via a **lightning indexer**.
- Indexer: low-rank compressed queries, `n_h^I` = **64 indexer query heads**, indexer head dim `c_I` = **128**.
- **Attention top-k = 512** (V4-Flash) / **1024** (V4-Pro) compressed KV entries.
- Plus a **sliding-window branch** over the last `n_win` = **128** uncompressed tokens.
- Shared-KV MQA layout; output projection split into `g` groups (8 Flash / 16 Pro) with intermediate dim `d_g` = 1024.

**Heavily Compressed Attention (HCA)**
- Far more aggressive: `m'` = **128** tokens → 1 entry, and **dense** (no top-k selection) over that compressed stream.

**KV cache precision is mixed: BF16 for the RoPE dimensions, FP8 for the rest.**

Result at 1M context vs DeepSeek-V3.2: V4-Pro uses **27% of the single-token inference FLOPs and 10% of the KV cache**; V4-Flash uses **10% of FLOPs and 7% of KV cache**.

**Qwen Sparse Attention (QSA, 2026)** is the same idea at a different granularity. Qwen's stated objection to DSA is that its indexer is itself O(n²), so indexing overhead stops being negligible as sequence grows. QSA "scores context at **micro-block granularity** with a compressed lightweight indexer, so that the cost of indexing itself falls with sequence length." Measured: **QSA matches the full-attention RULER baseline at a relative indexer latency of 0.25**, whereas the cross-layer index-sharing baseline (IndexShare) is still below baseline at 0.5. MRCR improved from 30.66 → 40.53 at 512K and 20.71 → 26.44 at 1M. QSA is introduced at **continued-pretraining time**, replacing the already-trained full-attention layers ([Qwen3.8 §2.1.2](https://github.com/QwenLM/Qwen3.8-Flash-Next)).

### Attention sinks

gpt-oss: "Each attention head has a **learned bias in the denominator of the softmax**, similar to off-by-one attention and attention sinks, which enables the attention mechanism to pay no attention to any tokens" ([§2.2](https://arxiv.org/abs/2508.10925)). This is one scalar per head — near-free, and it removes the massive-activation pathology that otherwise forces attention mass onto the first token.

### GQA reference configurations

- gpt-oss: **64 query heads of dim 64, 8 KV heads**, head dim 64, residual stream 2880.
- Olmo 3 7B: **32 Q / 32 KV** (plain MHA). Olmo 3 32B: **40 Q / 8 KV** (GQA).
- MiniMax-M2: **48 Q / 8 KV**, QK-Norm, partial RoPE.
- SmolLM3 3B: GQA with **4 groups**.

## 1.4 Positional encoding

> **2026 default choice: RoPE with a raised base frequency on global layers only (θ = 1M), and no positional encoding at all on the cheap layers. If you use a gated-linear-attention hybrid, consider dropping positional encoding entirely — the recurrence supplies position.**

**Per-layer-type RoPE base is now standard:**
- Gemma 3: **RoPE base 10k on local layers, raised to 1M on global self-attention layers**; context 128K (32K for the 1B) ([§2.1](https://arxiv.org/abs/2503.19786)).
- Qwen3 / GLM-4.5: **ABF from 10,000 → 1,000,000** during the long-context stage ([Qwen3 §3.2](https://arxiv.org/abs/2505.09388); [GLM-4.5 §2.3](https://arxiv.org/abs/2508.06471)).
- Olmo 3: **RoPE θ = 5×10⁵**, YaRN applied on full-attention layers only ([Table 33](https://arxiv.org/abs/2512.13961)).
- SmolLM3: base 10,000 → **1.5M at 32k** → **5M at 64k** ([HF blog](https://huggingface.co/blog/smollm3)).

**YaRN is the standard extension method.** DeepSeek-V3's exact config: applied **exclusively to the decoupled shared key `k_t^R`**, scale `s = 40`, `α = 1`, `β = 32`, scaling factor `√t = 0.1·ln(s) + 1`; two phases of **1000 steps each**, 4K→32K (batch 1920) then 32K→128K (batch 480), LR 7.3e-6 for both ([§4.3](https://arxiv.org/abs/2412.19437)). gpt-oss uses YaRN to reach 131,072 on its dense layers ([§2.2](https://arxiv.org/abs/2508.10925)).

**NoPE layers.**
- SmolLM3: removes RoPE from **every 4th layer** ([HF blog](https://huggingface.co/blog/smollm3)).
- Llama 4 Scout: "**interleaved attention layers without positional embeddings**" (iRoPE) plus "**inference time temperature scaling of attention**" for length generalization; 10M context claimed ([Meta blog](https://ai.meta.com/blog/llama-4-multimodal-intelligence/)).
- **Kimi K3 applies NoPE to *all* MLA layers.** The claim is strong and worth quoting:

> "Kimi K3 uses no explicit positional embedding (NoPE), and instead encodes positional information implicitly through the recurrent gating and decay mechanism of KDA. As a result, the model **extrapolates directly to 1M-token contexts without any positional-encoding modification**, such as RoPE rescaling or interpolation."
> — [Kimi K3 §3.4](https://github.com/MoonshotAI/Kimi-K3)

This eliminates the entire YaRN/ABF long-context stage as a separate concern — a major simplification if it holds.

**But Qwen3.8 tried NoPE and rejected it:**

> "RoPE and a NoPE variant without positional encoding show little difference during pretraining, but the NoPE variant exhibits a **substantially higher rate of endless generation after post-training** and is therefore more likely to fail to terminate."
> — [Qwen3.8 §2.1.1](https://github.com/QwenLM/Qwen3.8-Flash-Next)

So they kept RoPE in the full-attention layers. **UNVERIFIED** why K3 does not hit the same failure; the plausible difference is that K3's decay-gated KDA supplies a stronger recency prior than Qwen's GDN parameterization, and K3's global layers are MLA rather than standard attention.

## 1.5 Normalization

> **2026 default choice: RMSNorm everywhere, QK-Norm on queries and keys, and put a norm somewhere in the MoE routed path. Do not use logit soft-capping.**

- **RMSNorm pre-norm** is universal. Olmo 3 (following OLMo 2) applies **layer norm to the outputs** of attention and FFN blocks rather than the inputs ([Table 33](https://arxiv.org/abs/2512.13961)) — the "reordered norm" variant.
- **QK-Norm is now standard and has displaced logit soft-capping.** Gemma 3: "we replace the soft-capping of Gemma 2 with QK-norm" ([§2.1](https://arxiv.org/abs/2503.19786)). Also in Qwen3 (which also removed Qwen2's QKV bias, [§2](https://arxiv.org/abs/2505.09388)), GLM-4.5, MiniMax-M2, Olmo 3, modded-nanogpt.
- **QK-Norm is not directly applicable to MLA** — Kimi K2 notes this explicitly, which is why they needed MuonClip instead ([§2.1](https://arxiv.org/abs/2507.20534)).
- **DeepSeek-V4 solved it differently and thereby retired QK-Clip:** "The attention architecture of DeepSeek-V4 series allows us to directly apply **RMSNorm on the attention queries and KV entries**, which effectively prevents attention logits from exploding. Consequently, we do not employ the QK-Clip technique" ([§2.4](https://arxiv.org/abs/2606.19348)).
- **Zero-centered RMSNorm** (constrains growth of RMSNorm weights) is used throughout Qwen3-Next and Qwen3.8 ([§2.1.1](https://github.com/QwenLM/Qwen3.8-Flash-Next)).
- **Gemma 3 applies QK-Norm BEFORE RoPE** (per-head RMSNorm over head_dim), verifiable in the HF `modular_gemma3.py` ordering `q_proj → q_norm → apply_rotary_pos_emb`. Olmo 2 measured the payoff as a drop in "spike score" from **0.108 → 0.069** with post-norm plus QK-norm.
- **⚠ 2026 counter-evidence: QK-norm may cost long-context performance.** A controlled study of 26 comparable 7–8B models trained on 140B tokens plus 10B at 64K ([arXiv:2608.10296](https://arxiv.org/abs/2608.10296)) reports that on the Olmo architecture, **replacing QK-norm + post-sublayer-norm with prenorm gained 6 points on HELMET**, and that **adding QK-norm to Llama 3's architecture cost 3.8 points on HELMET at 32K**. The proposed mechanism is that QK-norm reduces attention entropy and suppresses attention sinks. They also find that combining three or more such "minor" architecture choices can drop downstream performance by up to **47%** — architecture choices interact, and ablating them one at a time is misleading.
- **Normalize the MoE routed path.** Kimi K3 inserts RMSNorm between expert aggregation and up-projection and reports it "consistently improves validation loss and downstream benchmarks" beyond its stabilizing role ([§2.3.1](https://github.com/MoonshotAI/Kimi-K3)).
- **Keep attention output in FP32 during training.** Kimi K3: "To correct the biased rounding error identified [in flash attention], we keep the attention output in FP32 during training," and redesigned the kernel to overlap the doubled on-chip footprint with KV staging buffers ([§2.1.2](https://github.com/MoonshotAI/Kimi-K3)).

## 1.6 Activation

> **2026 default choice: SwiGLU — unless you are training in FP4/FP8, in which case use a bounded variant.**

SwiGLU remains the default everywhere. But 2026's low-precision training exposed its weakness: **both multiplicative factors are unbounded**, so coincident large coordinates produce activation outliers and overflow risk.

**Kimi K3's SiTU-GLU (Sigmoid Tanh Unit GLU)** applies a smooth cap `softcap(x,β) = β·tanh(x/β)` to the linear factor of the Swish gate *and* independently to the up branch:

```
SiTU-GLU(x) = β₁·tanh(W_g x / β₁) ⊙ Sigmoid(W_g x) ⊙ β₂·tanh(W_u x / β₂)
```

with **β₁ = 4 (gate branch), β₂ = 25 (up branch)**, bounding the output at `|f(x)| ≤ β₁β₂ = 100`. It "closely follows SwiGLU near the origin" and bounds large positive inputs ([K3 §2.3.2, Fig. 4](https://github.com/MoonshotAI/Kimi-K3)).

gpt-oss also notes: "Our SwiGLU implementation is unconventional, including **clamping and a residual connection**" ([§2.2 footnote](https://arxiv.org/abs/2508.10925)) — the same instinct, disclosed less precisely.

## 1.7 The residual stream — the sleeper trend of 2026

> **2026 default choice: widen the residual stream and make the read/write data-dependent. Three labs converged on this independently in 2026.**

Pre-norm keeps training stable but attenuates signal: every block reads the same stream, so a feature written early competes with everything written after it. The 2024 lines of attack were AltUp and Hyper-Connections; in 2026 all three frontier labs shipped a variant.

| Lab | Name | Mechanism |
|---|---|---|
| DeepSeek-V4 | **Manifold-Constrained Hyper-Connections (mHC)** | Expand residual `R^d → R^{n_hc × d}`, constrain the residual mapping to doubly-stochastic matrices |
| Qwen3.8 | **Gated Residual (GR)** | Widen to 4 branches, read through an elementwise gate |
| Kimi K3 | **Attention Residuals (AttnRes)** | Each layer *attends* over the outputs of all preceding blocks |

**mHC (DeepSeek-V4).** Standard Hyper-Connections expand the residual state to `X_l ∈ R^{n_hc × d}` with three learned mappings: input `A_l ∈ R^{1×n_hc}`, residual transform `B_l ∈ R^{n_hc×n_hc}`, output `C_l ∈ R^{n_hc×1}`, updating `X_{l+1} = B_l X_l + C_l F_l(A_l X_l)`. Because the actual layer input `A_l X_l` is still `d`-dimensional, inner layers are unchanged — it is a *free extra scaling axis*. But naive HC is numerically unstable when stacked. mHC's fix: **constrain `B_l` to the Birkhoff polytope (doubly stochastic matrices)**, which bounds `‖B_l‖₂ ≤ 1` (non-expansive) and is closed under multiplication (so deep stacks stay stable). `A_l` and `C_l` are Sigmoid-bounded and non-negative to avoid signal cancellation. Parameters are split into a dynamic input-dependent part and a static bias, with learnable gating factors `α` initialized small. Projection uses **Sinkhorn–Knopp, `t_max` = 20 iterations**; **expansion factor `n_hc` = 4**. Cost: **6.7% of the overlapped 1F1B pipeline stage** ([V4 §2.2, §4.2.1](https://arxiv.org/abs/2606.19348)).

**AttnRes (Kimi K3).** The framing is elegant: standard residuals compress all prior depth into one state — "a bottleneck reminiscent of RNNs over time." Attention replaced recurrence over *time*; AttnRes applies the same move over *depth*. Each layer `l` has a learnable pseudo-query `q_l`, keys/values are the outputs of all preceding layers (plus the token embedding), and attention weights use `φ(q,k) = exp(qᵀ RMSNorm(k))` — the RMSNorm prevents large-magnitude layers from dominating. Full form is O(L²d) arithmetic, affordable at L < 100, but costs O(Ld) memory. **Block AttnRes** reduces this: partition L layers into N blocks, sum within a block, attend across block representations only — memory drops to O(Nd), and it bounds the inference-time state. **K3 uses 8 blocks of 12 layers** (9 total counting the embedding); "empirically, N ≈ 8 recovers most of the benefit across model scales" ([K3 §2.2](https://github.com/MoonshotAI/Kimi-K3)).

**Gated Residual (Qwen3.8).** Widen to **4 branches** (AltUp/Hyper-Connections lineage), each block reads a weighted sum via learnable scalars, and the read is passed through an **elementwise gate**: "widening adds capacity to the residual path, and the gate decides how that capacity is spent, while also supplying the rescaling that keeps training stable." Their stress test (28-layer 25B-A3B at 2× and 4× the optimal LR held constant) shows GR is what keeps training stable at aggressive learning rates where Muon alone is not enough ([Qwen3.8 §2.2, §3.3, Fig. 10](https://github.com/QwenLM/Qwen3.8-Flash-Next)).

**Small-compute note:** modded-nanogpt has carried a version of this since 2024 — "skip connections from embedding to every block as well as from block 3 to 6," U-net patterns, and MUDD skip connections ([README](https://github.com/KellerJordan/modded-nanogpt)). If you take one architectural idea from 2026 into a small model, take this one: it is cheap, it is in the small-scale speedrun *and* the 2.8T-parameter model, and it needs no new kernels.

## 1.8 Embeddings, vocabulary, and n-gram memory

> **2026 default choice: 128k–260k BPE vocab; tie embeddings only below ~3B; put the embedding table on AdamW even when everything else is on Muon.**

**Vocabulary sizes (verified):**

| Model | Vocab | Source |
|---|---|---|
| nanochat | **32,768** (current default; 65,536 was the original speedrun figure) | [tok_train.py](https://raw.githubusercontent.com/karpathy/nanochat/master/scripts/tok_train.py) |
| Olmo 3 | cl100k-derived (OLMo 2 tokenizer); exact size UNVERIFIED | [§3.2](https://arxiv.org/abs/2512.13961) |
| Llama 3 / SmolLM3 | 128,256 / 128k (Llama-3.2 tokenizer) | [Llama 3](https://arxiv.org/abs/2407.21783); [SmolLM3](https://huggingface.co/blog/smollm3) |
| DeepSeek-V3 | 129,280 | [§4.1](https://arxiv.org/abs/2412.19437) |
| Qwen3 | tokenizer **151,669** (BBPE); embedding matrix **151,936** (padded) | [§2](https://arxiv.org/abs/2505.09388) · [config.json](https://huggingface.co/Qwen/Qwen3-235B-A22B/raw/main/config.json) |
| Kimi K2 / K3 | **163,840** (reported as "160K", unchanged K2→K3) | [K2 config.json](https://huggingface.co/moonshotai/Kimi-K2-Instruct/raw/main/config.json) |
| gpt-oss | 201,088 (o200k_harmony) | [§2.4](https://arxiv.org/abs/2508.10925) |
| Qwen3.5 / 3.8 | ~250K base; 248,320 padded | [Qwen3.8 §2.3.2](https://github.com/QwenLM/Qwen3.8-Flash-Next) |
| Gemma 3 | 262,144 (SentencePiece, Gemini 2.0 tokenizer) | [§2.2](https://arxiv.org/abs/2503.19786) |

**Why the range moved from 32K to 128K–262K**: "Scaling Laws with Vocabulary" ([arXiv:2407.13623](https://arxiv.org/abs/2407.13623)) shows via three independent methods (IsoFLOPs analysis, derivative estimation, parametric loss fit) that **larger models deserve larger vocabularies**, and that **Llama-2-70B should have had ≥216K vocab — roughly 7× its actual 32K**. Empirical validation: a 3B model at a fixed 2.3e21 FLOP budget improved **ARC-Challenge 29.1 → 32.0 (+2.9)** purely by moving vocab from 32K to 43K. The 2026 range reflects the field absorbing this result.

### N-gram embedding tables (2026)

This is genuinely new and worth understanding, because it buys capacity at **near-zero per-token FLOPs**. Qwen3.8-Flash-Next holds **51B parameters of n-gram embedding tables off the accelerator** (in host memory) against a 125B/6B backbone. Short n-grams ending at each token are keys into embedding tables; retrieved vectors augment the token representation. Because addressing is deterministic, the tables can be **prefetched asynchronously from host memory**.

Their ablations ([§2.3](https://github.com/QwenLM/Qwen3.8-Flash-Next)):

- **Placement**: a single layer suffices; no depth regime dominates; distributing the budget across multiple layers gives no consistent benefit. They place it at **Layer 2** so host-memory prefetch overlaps the Layer-1 computation. Placement is insensitive to the attention mechanism.
- **Under a fixed parameter budget** (trading experts for n-gram slots): loss is lowest at 10× base vocab (25% of params) but **downstream benchmarks show no clear improvement over the MoE-only baseline** — "N-gram embeddings and MoE experts play distinct roles in scaling capacity."
- **With additional parameters** (20× → 200× base vocab): **loss decreases monotonically, but downstream accuracy saturates.** Avg went 45.44 (none) → 47.94 (50×) and then flattened. Chinese benchmarks (C-Eval, CMMLU) were the exception and improved consistently with vocabulary size.
- Production config: **20,000,000 bigram/trigram entries at layer 2**.

This is the cleanest published example in 2026 of **loss improving while benchmarks do not** — a warning about optimizing for validation loss alone.

Qwen's framing of why this matters: "Embeddings provide a unique axis for parameter scaling that **requires less computation and is more amenable to offloading than Mixture-of-Experts**" ([model card](https://huggingface.co/Qwen/Qwen3.8-Flash-Next)). Because the lookup key is deterministic (current token plus its predecessors), the table never enters the per-token matmul FLOP budget and can live in host RAM with async prefetch. Serving cost: vLLM needs **≥51 GB of host RAM** and `VLLM_PLE_CPU_OFFLOAD=1` ([vLLM recipe](https://recipes.vllm.ai/Qwen/Qwen3.8-Flash-Next)).

**The precedent is Gemma 3n's Per-Layer Embeddings** — and vLLM's environment variable literally spells "PLE." Gemma 3n keeps embedding parameters off-accelerator so that a **raw 5B model runs with a 2B footprint (2 GB)** and a **raw 8B with a 4B footprint (3 GB)** ([Google blog](https://developers.googleblog.com/en/introducing-gemma-3n/)). Qwen productionized the same idea at 51B and inverted the goal: Gemma used it to shrink, Qwen uses it to add capacity.

modded-nanogpt independently uses "**bigram hash embeddings with sign trick**" ([README](https://github.com/KellerJordan/modded-nanogpt)) — the same idea at 124M scale, which is unusually strong evidence that it is a real effect rather than a large-scale artifact.

## 1.9 Multi-token prediction (MTP)

> **2026 default choice: one MTP layer, loss weight λ = 0.3 decayed to 0.1 at LR-decay onset. Keep it — it improves the base model AND becomes your speculative-decoding draft head for free.**

- **DeepSeek-V3**: MTP depth **D = 1**; **λ = 0.3 for the first 10T tokens, 0.1 for the remaining 4.8T** ([§4.2.2](https://arxiv.org/abs/2412.19437)).
- **DeepSeek-V4**: depth 1, "configuration remains identical to that of DeepSeek-V3"; **λ = 0.3 for most of training, 0.1 upon the start of learning rate decay** ([§4.2.2](https://arxiv.org/abs/2606.19348)).
- **GLM-4.5**: 1 MTP layer, λ = 0.3 for first 15T tokens ([§2.3](https://arxiv.org/abs/2508.06471)).
- **Kimi K2 and K3**: 1 MTP layer each ([K3 Table 1](https://github.com/MoonshotAI/Kimi-K3)).
- **Qwen3.8**: 4B of parameters dedicated to MTP ([model card](https://huggingface.co/Qwen/Qwen3.8-Flash-Next)).

**The 2026 addition: fine-tune the MTP layer into an EAGLE-3 draft model.** Kimi K3 does exactly this — the MTP layer already mirrors a backbone block, which is also EAGLE-3's draft structure. They freeze the target, train only the draft layer and a feature-fusion projection, unroll the draft **7 steps** during training (consuming its own outputs beyond step 1), and fuse low/mid/high-level features from the **1st, 4th, and final AttnRes blocks**, projected by a bias-free `W_E3` initialized as `[0 0 I]` so it starts at the high-level feature the MTP layer was pretrained on. Crucially they **optimize the acceptance rate directly** rather than KL:

```
L_LK = −log Σ_x min(p(x), q(x))
```

because "minimizing the conventional KL-divergence surrogate does not guarantee maximizing this rate for a capacity-limited draft model" ([K3 §4.1.4](https://github.com/MoonshotAI/Kimi-K3)).

---

# Stage 2: Optimization

## 2.1 The optimizer

> **2026 default choice: Muon on 2-D weight matrices, AdamW on embeddings, the unembedding/head, norms, routers, and anything vector-shaped. This flipped from "AdamW everywhere" during 2026.**

**Who switched:**
- Kimi K2 (Jul 2025) — first trillion-scale Muon run, via **MuonClip**.
- **DeepSeek-V4 (Apr 2026)** — "We employ the Muon optimizer for the majority of modules in DeepSeek-V4 series due to its faster convergence and improved training stability."
- **Qwen3.8 (Aug 2026)** — "We use Muon as the main optimizer."
- **Kimi K3 (Sep 2026)** — Per-Head Muon.

### Exact parameter partition (three independent 2026 reports agree)

| On Muon | On AdamW |
|---|---|
| attention q/k/v/o projections | input embeddings |
| linear-attention (GDN/KDA) in/out projections | output head / prediction head |
| routed + shared expert fc1/fc2 | all RMSNorm weights |
| n-gram key/value projections (Qwen) | **the MoE router** |
| | mHC static biases and gating factors (DeepSeek) |
| | GDN decay/beta projections (per-head scalars → vectors) |
| | output gates (attention output gate, GDN z projection) |
| | Gated-Residual low-rank projections |

Sources: [DeepSeek-V4 §2.4](https://arxiv.org/abs/2606.19348); [Qwen3.8 §3.1](https://github.com/QwenLM/Qwen3.8-Flash-Next).

Qwen's reasoning on the router is worth keeping: "Muon exacerbates early-training fluctuations and destabilizes the router… One possible explanation is that each output dimension of the router corresponds to the score of one expert, and the dimensions are **largely independent, leaving no shared linear structure for orthogonalization to exploit**." Applying Muon to the router mid-to-late training was stable but gave no gain.

### Split fused parameters before orthogonalizing (important, easy to get wrong)

In Megatron-LM the attention qkv projection, SwiGLU fc1, and the GDN input projection are each stored as *one fused matrix*, but are semantically concatenations of independent operators. Qwen3.8:

> "Orthogonalizing the fused matrix is then wrong in two ways: the iteration **mixes singular directions across unrelated sub-blocks**, and γ(A,B) is computed from the **concatenated shape instead of the true operator shape**."

Fix: split the fused gradient, run Newton–Schulz on each sub-matrix, gather back. **qkv and GDN input projections split at per-head granularity** (improves both loss and benchmarks); **fc1 split into gate and up halves** (loss unchanged, benchmarks slightly better). Splitting also gives a natural granularity for excluding sub-matrices from Muon ([§3.1](https://github.com/QwenLM/Qwen3.8-Flash-Next)).

### Newton–Schulz configuration

- **Classic coefficients** (a,b,c) = **(3.4445, −4.7750, 2.0315)**, quintic iteration `M_k = aM + b(MMᵀ)M + c(MMᵀ)²M`, applied to `M₀ = M/‖M‖_F`.
- **DeepSeek-V4 uses a hybrid 10-iteration schedule**: first **8 steps** with (3.4445, −4.7750, 2.0315) "to drive rapid convergence, bringing the singular values close to 1"; final **2 steps** with **(2, −1.5, 0.5)** which "stabilize the singular values precisely at 1" ([§2.4](https://arxiv.org/abs/2606.19348)).
- **Qwen3.8 uses the Polar Express per-step coefficient schedule**, "minimax-optimal for a given step budget," with **8 iteration steps** — chosen because more accurate orthogonalization "reduces both the magnitude and frequency of gradient-norm spikes in our stress test." Frobenius-normalization epsilon **1e-14** ([§3.1](https://github.com/QwenLM/Qwen3.8-Flash-Next)).

### The reference implementation (DeepSeek-V4, Algorithm 1)

This is the whole optimizer. Note line 2 — **"logically independent"** is doing the same work as Qwen's fused-matrix splitting.

```
Require: learning rate η, momentum μ, weight decay λ, update rescaling factor γ
for each training step t:
  for each LOGICALLY INDEPENDENT weight W ∈ R^{n×m}:
     G_t  = ∇_W L_t(W_{t-1})                            # gradient
     M_t  = μ·M_{t-1} + G_t                             # momentum buffer
     O'_t = HybridNewtonSchulz(μ·M_t + G_t)             # Nesterov + hybrid NS
     O_t  = O'_t · sqrt(max(n, m)) · γ                  # rescale update RMS
     W_t  = W_{t-1}·(1 − ηλ) − η·O_t                    # decoupled weight decay + update
```

with **μ = 0.95, λ = 0.1, γ = 0.18** for DeepSeek-V4 ([§2.4, §4.2.2](https://arxiv.org/abs/2606.19348)). Qwen3.8 uses the same shape with **γ = 0.2** and 8 Polar Express steps in place of the hybrid 10-step schedule.

### Update scaling

Both labs use the Moonlight rule so Muon can reuse AdamW's learning rate:

- Qwen3.8: Nesterov momentum **μ = 0.95**, orthogonalized result scaled by **γ(A,B) = 0.2·√max(A,B)** for a parameter of shape A×B, "making the RMS of the update independent of the matrix shape."
- DeepSeek-V4: momentum **0.95**, weight decay **0.1**, Nesterov on, and **rescale the update-matrix RMS to 0.18** "for reutilization of our AdamW learning rate."

### Per-Head Muon (Kimi K3, 2026)

Instead of orthogonalizing the full Q/K/V projection, partition the momentum matrices along the head dimension and **orthogonalize each head's block separately**:

> "full-matrix orthogonalization treats all heads as a single coupled block, so heads with larger gradient or momentum scales dominate the shared update direction, while smaller-scale heads receive insufficiently normalized updates; per-head orthogonalization equalizes the update scale across heads."

Benefits: more balanced learning dynamics across heads, better stability at scale, and it is **cheaper** — Newton–Schulz on tall per-head blocks costs less than on the full matrix ([K3 §2.5](https://github.com/MoonshotAI/Kimi-K3)). Note this is the natural conclusion of Qwen3.8's per-head splitting argument.

### Implementation cost (why this is not free)

Newton–Schulz needs a holistic update over each full parameter matrix (≈ `4K·max(A,B)·min(A,B)²` FLOPs for K steps), which clashes with Megatron sharding two ways: under TP no rank owns the full matrix; under DP the cost is **cubic in the shorter dimension**, so equal-element partitioning leaves severe stragglers. Qwen built **Canzona**, which decouples logical optimizer assignment from physical layout: an α-balanced static partitioner reassigns whole parameters (never cutting inside a tensor) to equalize estimated NS FLOPs across DP ranks, plus an asynchronous Micro-Group pipeline that reconstructs each Muon-owned matrix via fused All-to-All across TP ranks, preserving ZeRO-1 bucket geometry. Second problem: after splitting, one layer contributes ~100 sub-matrices and the step becomes launch-overhead-bound — they **capture the whole optimizer step in a CUDA graph** ([§3.1](https://github.com/QwenLM/Qwen3.8-Flash-Next)).

### How much is Muon actually worth? The fair-comparison result

Before adopting Muon on the strength of the "2× token efficiency" claims, read **"Fantastic Pretraining Optimizers and Where to Find Them"** (Stanford; Wen, Hall, Ma, Liang — [arXiv:2509.02046](https://arxiv.org/abs/2509.02046)). It is the most careful published head-to-head, and the answer is sobering:

> "AdamW has long been the dominant optimizer… despite numerous claims that alternative optimizers offer **1.4 to 2× speedup**. We posit that two methodological shortcomings have obscured fair comparisons: (i) **unequal hyperparameter tuning** and (ii) **limited or misleading evaluation setups**."

Setup: **10 optimizers, 4 model scales (0.1B–1.2B), data-to-model ratios 1–8× Chinchilla**, each independently tuned. Three findings:

1. **"Optimal hyperparameters for one optimizer may be suboptimal for another, making blind hyperparameter transfer unfair."** (Exactly what Kimi K3 says about cosine vs WSD, §2.2 — this is a general methodological failure, not a schedule-specific one.)
2. **"The actual speedup of many proposed optimizers over well-tuned baselines is lower than claimed and decreases with model size."** Measured: **1.4× over AdamW at 0.1B → 1.1× at 1.2B.** "Optimizers' speedup w.r.t. AdamW decreases with model size."
3. **"Comparing intermediate checkpoints before reaching the target training budgets can be misleading, as rankings between two optimizers can flip during training due to learning rate decay."** Compare at the end of training or not at all.

They do confirm the mechanism: "all the fastest optimizers such as **Muon and Soap** use **matrices as preconditioners** — multiplying gradients with matrices rather than entry-wise scalars."

**How to reconcile this with DeepSeek-V4, Kimi K3 and Qwen3.8 all adopting Muon in 2026.** Two honest points cut in opposite directions:

- *Against Muon*: the measured trend is that the advantage **shrinks with scale**, and this study stops at 1.2B — well below any model in this document. A naive extrapolation of their curve would predict a negligible gain at 100B+.
- *For Muon*: the 2026 adopters are not claiming a raw loss speedup in isolation. Qwen3.8's stated benefit is that Muon **preserves data efficiency at larger batch sizes where AdamW degrades**, which is what lets them delete batch-size warmup and raise the LR/batch optimum (§2.3, §2.4); Kimi K2's was **stability at trillion scale**; DeepSeek-V4's stated reason is "faster convergence **and improved training stability**." The 2026 case for Muon is a *systems and stability* case that shows up as throughput and fewer restarts, not purely as loss-per-token.

**A second fair benchmark reaches a different winner, and explains the disagreement.** EPFL's study ([arXiv:2509.01440](https://arxiv.org/html/2509.01440v1), 11 optimizers, 124M–720M dense + 520M MoE) finds **AdEMAMix and MARS** best at 720M, not Muon — and Stanford's paper diagnoses why:

> "Their most extensively tuned experiments on 130M models use a **batch size of only 0.1M and 0.02M tokens**, whereas our experiments operate with tuned batch sizes **not smaller than 0.4M tokens**… Mars and AdEMAMix both perform gradient averaging and variance reduction; these methods are advantageous in their **noise-dominated small-batch regime**, whereas **in our larger-batch setting these benefits diminish and matrix-level optimizers become more competitive**."

**The "best optimizer" is a function of batch-size regime.** Variance-reduction methods win in small-batch/noise-dominated settings; matrix preconditioners win at production batch sizes. EPFL also reports that **SOAP beats AdamW at Chinchilla-optimal duration and below but loses in long training**, and that **Sophia diverges** as iteration count grows.

**Three 2026 papers push back on the shrinkage result:**

| Paper | Claim |
|---|---|
| [arXiv:2606.16899](https://arxiv.org/abs/2606.16899) (*Fantastic Pretraining Optimizers II*) | The shrinkage is an artifact of **constant decoupled weight decay** — equilibrium weight norm sets the effective *angular* learning rate. Their fix recovers **20–30% token-equivalent speedup** at ≤1.2B |
| [arXiv:2512.05620](https://arxiv.org/abs/2512.05620) | With μP-based LR scaling + **weight decay ∝ 1/width**, the ~1.4× speedup **persists** from 190M to 1.4B |
| [arXiv:2607.20548](https://arxiv.org/abs/2607.20548) (NVIDIA) | The largest-scale evidence, on a different axis: at **621M dense, 8B dense, 3B-A30B MoE and 8B-A72B hybrid MoE** over 1T–3T tokens, "SOAP and Muon consistently outperform AdamW… Notably, **at batch sizes of up to 100M tokens** for next-token prediction, these optimizers maintain training stability and quality **while AdamW degrades**" |

That last point matters most for the 2026 recipe: Stanford never tested 100M-token batches, and that is precisely the regime frontier MoE runs operate in. It is also consistent with Essential AI's measurement ([arXiv:2505.02222](https://arxiv.org/abs/2505.02222), 13 batch sizes from 128K to 16M tokens, models to 4B): "**Muon is more effective than AdamW in retaining data efficiency at large batch sizes, far beyond the so-called critical batch size**," needing **10–15% fewer tokens** to reach an identical loss, with the advantage "staying constant or growing as the batch size increases."

**Practical reading:**
- Training below ~1B at modest batch: expect **1.1–1.4×**, and tune AdamW properly before believing any comparison. The 2× claims in the original Muon/Sophia/SOAP papers are **not reproducible against a tuned baseline**.
- Training a large sparse MoE at production batch sizes: adopt Muon for the **stability and large-batch data efficiency**, and re-fit LR and batch size (§2.4) — that re-fit, not the optimizer itself, is where Qwen measured 7.8e-3 loss and 4.1 downstream points.
- **Throughput is no longer an objection.** NVIDIA measured Megatron-Bridge on GB300 NVL72 (TFLOP/s/GPU): Kimi K2 — AdamW **1051** vs Muon **1080**; Qwen3-30B-A3B — AdamW **713** vs Muon **721**. Muon is throughput-neutral-to-positive in production ([NVIDIA](https://developer.nvidia.com/blog/advancing-emerging-optimizers-for-accelerated-llm-training-with-nvidia-megatron)).

**The best-documented negative result** is Step3-VL-10B, which tested Muon and rejected it: "we exclude Muon from the final architecture due to **initialization mismatch**… this necessitates a prolonged warmup period to stabilize the transition, which **paradoxically limits overall training efficiency compared to a well-tuned AdamW baseline**" ([arXiv:2601.09668](https://arxiv.org/html/2601.09668v2)).

### MuonClip / QK-Clip — and its 2026 obsolescence

Kimi K2's problem: Muon's token efficiency came with **exploding attention logits**, and QK-Norm is not applicable to MLA. **QK-Clip** rescales the query and key projection weights *post-update* whenever the max attention logit exceeds a threshold τ:

- per-head factor `γ_h = min(1, τ / S_max^h)`; applied per head to avoid over-intervening
- **τ = 100** for the Kimi K2 run
- For MLA, clipping is applied only to the appropriate components
- Result: **15.5T tokens with zero loss spikes**; logits are capped at 100 initially, and **QK-Clip naturally disengages after ~30% of training steps**
— [Kimi K2 §2.1, Algorithm 1](https://arxiv.org/abs/2507.20534)

**DeepSeek-V4 removed the need for it** by architecture rather than optimizer surgery: RMSNorm directly on attention queries and KV entries ([§2.4](https://arxiv.org/abs/2606.19348)). **Kimi K3 retains "the weight-clipping mechanism introduced in Kimi K2"** alongside Per-Head Muon ([§3.3](https://github.com/MoonshotAI/Kimi-K3)).

### AdamW settings when you use AdamW

- DeepSeek-V3 (AdamW for everything): **β₁ = 0.9, β₂ = 0.95, weight_decay = 0.1**, grad clip **1.0** ([§4.2.2](https://arxiv.org/abs/2412.19437)).
- DeepSeek-V4 (AdamW only for embeddings/head/norms): **β₁ = 0.9, β₂ = 0.95, ε = 1e-20, weight_decay = 0.1** ([§4.2.2](https://arxiv.org/abs/2606.19348)). The **ε = 1e-20** is notable — far below the usual 1e-8.
- SmolLM3: **β₁ = 0.9, β₂ = 0.95, wd = 0.1**, grad clip 1.0 ([HF blog](https://huggingface.co/blog/smollm3)).
- Olmo 3: **no weight decay on embeddings**, **z-loss weight 1e-5**, grad clip 1.0 ([Table 33](https://arxiv.org/abs/2512.13961)).
- Kimi K3: **weight decay 0.1 throughout** ([§3.3](https://github.com/MoonshotAI/Kimi-K3)).
- Qwen3.8: **n-gram embedding table on Adam with weight decay disabled** ([§3.1](https://github.com/QwenLM/Qwen3.8-Flash-Next)).

**β₂ = 0.95 is effectively universal at scale.** Every frontier report that discloses betas uses (0.9, 0.95): DeepSeek-V3/V4, OLMo 2/3, SmolLM3, Nemotron-H, Nemotron 3 Super, Step3-VL. **β₂ = 0.999 appears in zero frontier pretraining reports** — it survives only as a library default that configs override. Muon's momentum converged on the same number (μ = 0.95).

**But it is convention, not a measured optimum.** The EPFL benchmark tuned betas properly at 2.1B and found **β₂ = 0.999 optimal** for AdamW, SOAP and AdEMAMix, noting that "many large-scale works in industry either do not tune the betas at all or simply adopt conventional defaults (β₁=0.9, β₂=0.95)" — and that 0.95 contradicts the theoretical prediction β₂ ≈ 1 − 1/T ([arXiv:2509.01440](https://arxiv.org/html/2509.01440v1)).

**On epsilon**: 1e-8 is the real default; DeepSeek-V4's **1e-20** is an outlier stated without justification (and with FP32 master weights, where the FP32 relative floor of ~1.2e-7 makes 1e-20 functionally "ε ≈ 0"). The only published *rationale* for any ε value is OLMo 2's, and it is a training-dynamics argument rather than a numerics one: "decreasing the AdamW ε from 1e-5 to 1e-8… the lower value allows for **larger updates early in training**… the gradient norm settles much more quickly and remains permanently lower" ([arXiv:2501.00656](https://arxiv.org/html/2501.00656v3)).

**On weight decay and embeddings, two distinct patterns:**
- **Explicit no-decay group (embeddings only)** — OLMo 2/3, SmolLM3, Qwen3.8. In all OLMo cases **RMSNorm weights are still decayed**; the architecture has no biases. OLMo 2's stated reason: decay "overshoots the mark and results in very small embeddings… small embeddings can produce large gradients in early layers because the Jacobian of layer_norm(x) w.r.t. x is inversely proportional to ‖x‖." Measured spike score **0.16 → 0.092**.
- **Split-optimizer instead** — DeepSeek-V4, GLM-4.5, Qwen3.8 decay everything, but via different optimizers.

Also note a widely-miscited number: Llama 3's "weight decay at each step is set to 0.1 times the learning rate" is from **§3.2.1, describing the 40M–16B scaling-law proxy models — not the 405B run**, whose optimizer section discloses no betas, ε, weight decay, or clipping at all.

**Weight decay itself is under revision upward.** "Weight Decay Improves Language Model Plasticity" ([arXiv:2602.11137](https://arxiv.org/abs/2602.11137), Feb 2026) finds optimal WD far above the 0.1 default — **0.5–1.0 at 20 tokens/param** (Llama-2-0.5B/1B 0.5, OLMo-2-1B 0.6, Llama-2-4B 1.0), falling to **0.3 at 140 TPP**. "Pre-training under infinite compute" ([arXiv:2509.14786](https://arxiv.org/abs/2509.14786)) puts it more starkly: "optimal weight decay is **30× standard practice**." Cerebras's Power Lines gives the practical rule that follows: **"We should adjust λ, not η, as B changes."**

## 2.2 Learning-rate schedule — the 2026 reversal

> **2026 default choice: this is now genuinely contested. Cosine (or constant-then-cosine-decay) is what the three biggest 2026 runs used. WSD's advantage may have been an artifact of shared hyperparameters.**

For two years the consensus was that WSD (warmup–stable–decay) matches or beats cosine and enables continued training from the stable-phase checkpoint. **Two 2025–2026 frontier reports contradict this**, and Kimi K3 explains why:

> "Our scaling-law study consistently favors **cosine decay over Warmup Stable Decay (WSD)**… Although prior work has reported that WSD can match or even outperform cosine decay, we observe that the two schedules exhibit **markedly different optimal hyperparameters**. Even under the same model size and training-token budget, their **optimal peak learning rates and batch sizes differ substantially**. As a result, comparing the two schedules using a shared set of hyperparameters may unfairly favor one simply because those hyperparameters are better aligned with it. To ensure a fair comparison, we conduct an **independent scaling-law search for each schedule**. Under their respective optimal hyperparameter settings, cosine decay consistently achieves a lower final loss than WSD."
> — [Kimi K3 §3.2](https://github.com/MoonshotAI/Kimi-K3)

GLM-4.5 reached the same practical conclusion: they used **cosine decay "instead of warmup-stable-decay (WSD) schedule"** based on early experiments; peak LR **2.5e-4** decaying to **2.5e-5** by the end of midtraining ([§2.3](https://arxiv.org/abs/2508.06471)).

**Kimi K3's production schedule: cosine with a 1% linear warmup, weight decay 0.1 throughout** ([§3.3](https://github.com/MoonshotAI/Kimi-K3)).

**But the constant-then-decay family is still widely used**, and DeepSeek's variant is really WSD by another name:

- **DeepSeek-V3**: linear warmup 0 → 2.2e-4 over 2K steps; **constant 2.2e-4 until 10T tokens**; cosine decay to 2.2e-5 over 4.3T tokens; then constant 2.2e-5 for 333B tokens and constant 7.3e-6 for the final 167B ([§4.2.2](https://arxiv.org/abs/2412.19437)).
- **DeepSeek-V4-Flash**: linear warmup over **2000 steps**, **constant 2.7e-4 for most of training**, cosine decay to **2.7e-5** near the end. V4-Pro: peak **2.0e-4** → end **2.0e-5** ([§4.2.2](https://arxiv.org/abs/2606.19348)).
- **Olmo 3 7B**: modified cosine, peak **3.0e-4**, 2000 warmup steps, final 3.0e-5. **32B**: cosine **truncated early at 5.5T of a 5.93T schedule**, peak **6.0e-4** ([Table 35](https://arxiv.org/abs/2512.13961)).
- **SmolLM3**: WSD, 2000 warmup steps, **linear decay over the final 10%**, LR 2e-4 ([HF blog](https://huggingface.co/blog/smollm3)).
- **Olmo 3 midtraining and long-context stages: linear decay to zero, 0 warmup steps** (midtrain) / 200 warmup steps (long-context), from **2.07e-4** ([Table 35](https://arxiv.org/abs/2512.13961)).

### But WSD is alive and well at the largest scale

Kimi's claim carries three caveats worth stating: it is made **"under a fixed minimum learning rate"** — the exact axis on which cosine is normally handicapped; it reports **no loss numbers, no margin, and no model sizes**; and **Kimi K2 used WSD twelve months earlier**, so this is one lab reversing itself.

The strongest counterexample is **NVIDIA Nemotron 3 Ultra (2026)**, which is far larger than anything in Kimi's ablation: "For Nemotron 3 Ultra we use a **Warmup-Stable-Decay (WSD)** learning rate schedule over a total horizon of **20 trillion tokens**. We warmup… for **200 billion tokens** to a peak value of **2.5e-4**. For the final **5 trillion tokens** we then decay" — a 550B-A55B model with a **1% warmup and a 25% decay fraction** ([tech report](https://research.nvidia.com/labs/nemotron/files/NVIDIA-Nemotron-3-Ultra-Technical-Report.pdf)).

Other relevant evidence:
- **Hägele et al.** ([arXiv:2405.18392](https://arxiv.org/abs/2405.18392)) find cooldown schedules **tie** cosine rather than beat it — "almost perfect match between the performance of the best cosine and cooldown schedule even for different training durations" — with the caveat that "the cosine schedule should be decayed to lower values" than 10%. Decay-shape ranking: **(1−√) > linear > cosine > square** (≈0.02–0.05 ppl). Their Chinchilla replication cost fell **5.59e23 → 2.36e23 FLOPs (~58% saving)**.
- **D2Z** ([arXiv:2502.15938](https://arxiv.org/abs/2502.15938), Cerebras) argues for **linear decay to zero**: a **610M** model at **80 tokens-per-parameter** with D2Z beats the same model at **200 TPP** with 10× decay — "an astonishing **60% compute savings**." Crossover is around **4 TPP**: at 2 TPP D2Z is *worse* at every peak LR; by 20+ TPP it is consistently superior. Versus WSD at 610M/80 TPP, "Linear-D2Z remains best here, around **0.84% better** than the optimal WSD." Their explanation is that AdamW behaves like an EMA with timescale **τ = 1/(ηλ)**, so decaying to zero optimally balances moving off initialization early against averaging gradient noise late.
- **A 2026 theory paper reconciles the fight as a phase transition.** "Optimal Learning-Rate Schedules under Functional Scaling Laws" ([arXiv:2602.06797](https://arxiv.org/abs/2602.06797), PKU) shows the optimal schedule depends on task difficulty: in the **easy-task regime**, power decay to zero (cosine/D2Z-like) is optimal; in the **hard-task regime**, **WSD is optimal** — "maintains the largest admissible learning rate for most of training and decays only near the end, with the decay phase occupying a vanishing fraction of the horizon."
- **Google's 2026 empirical study** ([arXiv:2603.10301](https://arxiv.org/abs/2603.10301)) is blunter: "warmup and decay are robust features of good schedules, and **commonly used schedule families are not optimal** on these workloads… weight decay can have a strong effect on the optimal schedule shape."

**Verdict: neither cosine nor WSD wins universally. The one thing everyone now agrees on is the methodology — never compare schedules at shared hyperparameters.** MiniCPM's budget rule still holds either way: "a decay of **10% of total tokens is sufficient**; a decay of 2.5% falls short" ([arXiv:2404.06395](https://arxiv.org/html/2404.06395v2)).

**Practical reading:** if you are doing one run to a known token budget, cosine-to-zero or linear-D2Z with per-schedule tuned LR and batch. If you want to branch checkpoints for midtraining experiments — the whole Olmo/SmolLM workflow — keep a stable phase so you can fork and decay cheaply. And note the honest LR-schedule warmup band across 2026 frontier runs: **1–2% of steps** (Kimi K3 1% linear; Nemotron 3 Ultra 1% = 200B tokens; Kimi K2 500 steps; DeepSeek-V3 and Olmo 3 2,000 steps; Llama 3.1 405B 8,000 steps).

## 2.3 Batch size — and the 2026 finding that ramping is obsolete

> **2026 default choice: if you are on Muon, skip the batch-size warmup and start at the target batch size. If you are on AdamW, keep the ramp.**

**The classic ramp:**
- DeepSeek-V3: **3072 → 15360 sequences over the first 469B tokens**, constant thereafter ([§4.2.2](https://arxiv.org/abs/2412.19437)).
- DeepSeek-V4-Flash: ramp to **75.5M tokens/batch**; V4-Pro to **94.4M tokens/batch** ([§4.2.2](https://arxiv.org/abs/2606.19348)).
- GLM-4.5: **16M → 64M tokens over the first 500B tokens** ([§2.3](https://arxiv.org/abs/2508.06471)).
- GPT-3 175B: "gradually increase the batch size linearly from a small value (**32k tokens**) to the full value [3.2M] over the first **4–12 billion tokens**."
- Llama 3.1 405B: 4M @4,096 → **8M @8,192 after 252M tokens** → **16M after 2.87T tokens**.
- **No ramp**: Kimi K2 ("the global batch size was held at 67M tokens"), Nemotron-H (6.29M), Qwen3.8-Next (25.2M from step 1).

**Qwen3.8's result (Aug 2026) — the ramp is now a net loss with Muon.** Tested on a 20-layer 10.8B-A0.89B MoE over a **4T-token** budget, ramping 6.3M → 25.2M in 6.3M increments, reaching target at 524B tokens, in two variants (peak LR held; peak LR lowered for the early small batches):

- Neither variant improved performance: they underperformed the constant-batch baseline by **2.5e-4 and 3.5e-4** in loss.
- The warmup required **18.8% more optimizer steps for the same token budget** — pure wall-clock overhead.
- Stability was unaffected either way: no step exceeded its local median loss by >0.1, and p99.9 pre-clip grad norm was **0.088–0.190 against a clipping threshold of 0.5**.

Mechanism they give: during warmup, the smaller batch means higher gradient noise at the same LR → higher loss; after reaching target there is a transient advantage from the extra accumulated steps; but "as the learning rate decays and the model approaches convergence, this step-count advantage is neutralized."

**⚠ The opposite result holds under AdamW**, which is why this is optimizer-conditional rather than universal. Ai2 measured CBS directly by branching ([arXiv:2505.23971](https://arxiv.org/abs/2505.23971)) and found "CBS is near 0 at initialization, increases rapidly at first, and then plateaus" — then **trained OLMo 1B to slightly better loss with 43% fewer gradient steps using a batch-size warmup** (LR ×√2 per doubling). **Reading: the ramp is an AdamW-era device; under Muon it is dead weight.** Qwen's test covers only Muon.

The underlying scaling law is worth knowing: **critical batch size scales with *data* size, not model size** ([arXiv:2410.21676](https://arxiv.org/abs/2410.21676)) — fitted **B\* = 22.91·D^0.47** (D in billions of tokens) versus **B\* = 621.34·N^0.087**, which is nearly flat in N. CBS there is defined as the largest batch with ≤20% step-count overhead versus linear scaling.

Why Muon changes the calculus: "decreasing the batch size below the predicted optimum incurs a **more significant performance penalty than increasing it**, and Muon **preserves data efficiency at larger batch sizes where AdamW's performance degrades**." Plus, for sparse MoE, a larger batch ensures every expert sees enough diverse tokens per step to specialize.
— [Qwen3.8 §3.2, Fig. 8](https://github.com/QwenLM/Qwen3.8-Flash-Next)

## 2.4 Hyperparameter transfer: fitted scaling laws are beating μP in practice

> **2026 default choice: fit an empirical scaling law for (peak LR, batch size) jointly against your specific architecture + optimizer, and re-fit whenever either changes. Do not port a previous generation's recipe.**

### What μP is, and why 2026 reports mostly do not cite it

**μP (Maximal Update Parametrization, Tensor Programs V, [arXiv:2203.03466](https://arxiv.org/abs/2203.03466))** reparametrizes a network so that "many optimal HPs remain stable even as model size changes," enabling **μTransfer**: "parametrize the target model in μP, tune the HP indirectly on a smaller model, and **zero-shot transfer them to the full-sized model, i.e., without directly tuning the latter at all.**" Headline result: hyperparameters tuned on a **40M proxy** transferred to match published **6.7B GPT-3** performance, at a **tuning cost of 7% of total pretraining compute**.

The practical scaling table, as commonly implemented (rules are standard practice; the exact constants are implementation-specific — **UNVERIFIED** against the paper's own notation in this pass):

| Parameter group | Init variance | Adam LR | Forward multiplier |
|---|---|---|---|
| Input embeddings | Θ(1) | Θ(1) — constant with width | Θ(1) |
| Hidden weights | Θ(1/fan_in) | **Θ(1/fan_in)** — i.e. LR ∝ 1/width | Θ(1) |
| Output / unembedding | Θ(1/fan_in²) | Θ(1/fan_in) | **Θ(1/fan_in)** logit multiplier |
| Attention logits | — | — | **1/d instead of 1/√d** |

Variants: **u-μP** (unit-scaled μP, which composes μP with low-precision-friendly unit scaling), **depth-μP** (extends transfer across depth as well as width), and Meta's **MetaP** used for Llama 4 (named in the [Meta blog](https://ai.meta.com/blog/llama-4-multimodal-intelligence/) but not described — **UNVERIFIED**).

**μP has two hard limits, and they are exactly the axes that matter in practice:**

1. **It does not transfer across token horizon.** Microsoft's study ([arXiv:2409.19913](https://arxiv.org/abs/2409.19913), ICLR 2025) states it plainly: **"the optimal LR does not transfer across token horizons with μP."** They fit **LR\*(D) = B·D^(−β)** with **β ≈ 0.32** for models ≥760M (R² 0.96–0.99), and give a damning worked example: **Llama-1 used a learning rate ~2.5× too high** (3e-4 actual vs ~1.15e-4 optimal), costing **ΔL = 0.027** in validation loss.
2. **It does not address the batch-size / step-count tradeoff.** μP's own batch-size transfer experiment held training steps fixed, i.e. spent more FLOPs at larger batch.

A further 2026 challenge ([arXiv:2510.19093](https://arxiv.org/abs/2510.19093), NeurIPS OPT): "in the practical setups where learning rate transfer is most valuable, such as LLM training, these [μP] assumptions hold only briefly at the start of training… it is **weight decay rather than μP** that stabilizes the update dynamics of internal representations across widths." On this account μP's LR scaling "functions more like warmup than as a fundamental scaling solution."

**So the 2026 flagships fit empirical power laws per (architecture, optimizer) instead.** Published fits:

| Lab | Fitted law |
|---|---|
| **DeepSeek LLM** ([arXiv:2401.02954](https://arxiv.org/abs/2401.02954)) | **η_opt = 0.3118·C^(−0.1250)**, **B_opt = 0.2920·C^(0.3271)** |
| **StepLaw** ([arXiv:2503.04715](https://arxiv.org/abs/2503.04715)) | **η(N,D) = 1.79·N^(−0.713)·D^(0.307)**, **B(D) = 0.58·D^(0.571)** — fitted from **3,700 LLMs trained from scratch, 100T tokens, ~1M H800 GPU-hours**; predicted optimum deviates from exhaustive search by **0.094%** |
| **Power Lines** ([arXiv:2505.13738](https://arxiv.org/abs/2505.13738), Cerebras) | **B_opt ∝ D^0.4**, **B_crit ∝ D^0.5**, both **independent of N**; and the practical inversion: "**We should adjust λ, not η, as B changes**" |
| **Qwen3 / Qwen3.8 / Kimi K3** | Re-fit per generation; **μP is not mentioned in any of the three reports** |

StepLaw states the reason for not using μP directly: it "[is] limited in scope, lacking guidance for learning rate adaptation across different data distributions, model architectures, sparsity levels, and data sizes."

**A unifying invariant may be emerging**: "Optimal Scaling Needs Optimal Norm" ([arXiv:2510.03871](https://arxiv.org/abs/2510.03871)) finds that across models to 1.3B / 138B tokens, the optimal (η\*, B\*) pair "consistently has the same **operator norm value**" of the output layer — *norm transfer* rather than hyperparameter transfer.

**Net: μP is research hygiene — it removes the parameterization lottery and gives tighter scaling-law fits — but no 2026 frontier lab ships on μP alone.** The counter-argument is live, though: [arXiv:2512.05620](https://arxiv.org/abs/2512.05620) claims that with correct μP-based LR scaling *plus* **weight decay ∝ 1/width**, Muon/SOAP/Shampoo "consistently achieve **near 1.4× speedup** over AdamW" from 190M to 1.4B, and that "the speedup vanishes rapidly with scale **under incorrect scaling**" — i.e. that the negative results below are tuning failures rather than optimizer limits.

**Qwen3.8's procedure and results** ([§3.2](https://github.com/QwenLM/Qwen3.8-Flash-Next)):

> "The optimal learning rate and batch size depend on the model architecture and the optimizer, so a recipe that was optimal for the previous-generation model may become suboptimal once both change… The architecture and optimizer changes shift the predicted near-optimal hyperparameters toward **substantially larger batch sizes and learning rates, with a slower decay in the learning rate as model size increases**."

They validated each prediction in the regime that stresses it:

*Batch size, validated on a small model with many tokens* (20-layer 10.8B-A0.89B, 4T tokens), averaged over the final 20B tokens:

| Setting | Batch | Loss |
|---|---|---|
| Previous Qwen3.5 recipe | 12.6M | 1.5774 |
| **New scaling-law optimum** | **25.2M** | **1.5702** |
| 1.5× larger | 37.7M | 1.5707 |

Gain over the previous recipe: **7.2e-3 in loss**. "The loss increases sharply below the predicted batch size and plateaus above it."

*Learning rate, validated on a large model with few tokens* (48-layer **156B-A7B**, 419B tokens):

| Setting | Batch | η | Avg. downstream |
|---|---|---|---|
| **New fit, predicted optimum** | 8.4M | **1.76e-3** | **60.55** |
| New fit, η ÷ √2 | 8.4M | 1.24e-3 | 59.14 |
| New fit, η × √2 | 8.4M | 2.49e-3 | 60.10 |
| New fit, B × 1.25 | 10.5M | 2.01e-3 | 60.06 |
| **Previous Qwen3.5 recipe** | 4.2M | 6.8e-4 | **56.41** |

The previous recipe ended **7.8e-3 above** the predicted optimum in loss and **4.1 points below** in downstream average. The optimum "sits at the bottom of a bowl that is flat over at least a factor of √2 in either direction in learning rate and +25% in batch size."

Stability at the optimum: gradient clipping **never engages after warmup** in any of the five runs; max pre-clip grad norm reaches **28% of the clipping threshold** at the optimum vs **51% under the previous recipe** (whose smaller batch made gradients noisier). No loss spikes.

**Kimi K3 did the same thing**, running "dedicated scaling-law studies to retune key hyperparameters, including the **batch size, learning rate, tokens-per-parameter ratio (TPP) and the model shape**," evaluated on held-out OOD validation data ([§3.2](https://github.com/MoonshotAI/Kimi-K3)). Qwen3 likewise "develop[s] scaling laws for optimal hyper-parameters (e.g., learning rate scheduler, and batch size) predictions based on three pre-training stages" ([§3.2](https://arxiv.org/abs/2505.09388)).

Meta's Llama 4 used **MetaP** for hyperparameter transfer ([Meta blog](https://ai.meta.com/blog/llama-4-multimodal-intelligence/)) — details **UNVERIFIED** beyond the name.

**Olmo 3 reports a useful invariant** to compare stages: **"peak training temperature" = LR²/batch_size**. Values: 7B pretrain 2.146e-14, midtrain 2.051e-14 (nearly matched), long-context 1.026e-14 (halved) ([Table 35](https://arxiv.org/abs/2512.13961)).

## 2.5 Stability engineering

> **2026 default choice: design a stress test that surfaces production instabilities at small scale, and make architecture choices on it — not just on loss.**

Qwen3.8 formalized this: "long training runs can spend substantially more optimizer steps at peak learning rates, increasing the opportunity for instabilities to emerge… we design a set of stress tests that **amplify the relevant stress within a smaller budget**." Their test: a 28-layer 25B-A3B MoE at a **constant** learning rate of **2× and 4× the optimal**, comparing AdamW+old structure, Muon+old structure, and Muon+Gated Residual. Only Muon + GR stays stable at 4× optimal ([§3.3, Fig. 10](https://github.com/QwenLM/Qwen3.8-Flash-Next)).

**Other stability levers verified in 2026 reports:**
- Gradient clipping: **1.0** (DeepSeek-V3, SmolLM3, Olmo 3); **0.5** (Qwen3.8).
- **z-loss weight 1e-5** (Olmo 3, [Table 33](https://arxiv.org/abs/2512.13961)).
- Anticipatory Routing (DeepSeek-V4, §1.2).
- Bounded activations, bounded decay gates, bounded residual mappings — the whole 2026 "put a bound on it" family (§1.6, §1.7, and KDA's `g_min = −5` below).
- **Kimi K3's lower-bounded decay**: KDA bounds log-decay via `g = g_min · Sigmoid(e^A z)` with **`g_min = −5` fixed**, replacing Kimi Linear's unbounded negative-Softplus. Every retention factor then satisfies `α > e⁻⁵ ≈ 6.7e-3`, cumulative log-decay over a 16-token tile lies in (−80, 0), and the reciprocal rescaling factor stays **within BF16 dynamic range**. The payoff is not just numerical: it "allows both diagonal and off-diagonal tiles to use dense Tensor Core matrix multiplications, **eliminating the position-pair diagonal path**" that was the intra-chunk bottleneck ([§2.1.1](https://github.com/MoonshotAI/Kimi-K3)). This is a textbook example of **co-designing numerics and kernels**.

## 2.6 Precision

> **2026 default choice: BF16 master path with FP8 for the heavy GEMMs. FP4 for MoE expert weights is now shipping at frontier scale, but via QAT in post-training rather than from scratch.**

**FP8 pretraining** was pioneered by DeepSeek-V3 (fine-grained per-tile/per-block quantization with FP32 accumulation) and Llama 4 ("FP8 precision, without sacrificing quality," 390 TFLOPs/GPU on 32K GPUs for Behemoth, [Meta blog](https://ai.meta.com/blog/llama-4-multimodal-intelligence/)).

**DeepSeek-V4's 2026 precision map** ([§2.3.3, §3](https://arxiv.org/abs/2606.19348)):
- **Routed expert parameters: FP4**
- Core computation: BF16/FP8 hybrid
- Attention indexer: FP4
- KV storage: **FP8 for most dimensions, BF16 for the RoPE dimensions**
- **MoE gradients quantized to BF16 with stochastic rounding**

**Kimi K3's approach — QAT from SFT onward, not from scratch** ([§4.1.4](https://github.com/MoonshotAI/Kimi-K3)):
- MoE expert weights → **MXFP4**, activations → **MXFP8**
- All non-expert components (attention projections, latent MoE projections, shared experts, MoE routers) stay higher precision
- QAT runs **throughout the entire post-training stage, covering both SFT and RL**
- The key benefit: "During RL, rollout and training share the same quantization scheme — **eliminating the train–inference mismatch**." This is a genuinely elegant solve for the mismatch problem that otherwise needs importance-sampling corrections (§5.3).

**gpt-oss** post-trained with MoE weights quantized to **MXFP4 at 4.25 bits/param**; MoE weights are **90%+ of total parameters**, which is what lets 120b fit on a single 80GB GPU and 20b run in 16GB ([§2.1](https://arxiv.org/abs/2508.10925)).

**Gemma 3** ships QAT checkpoints fine-tuned for **~5,000 steps**, using probabilities from the non-quantized checkpoint as targets, in per-channel int4, per-block int4, and switched fp8 ([§2.3](https://arxiv.org/abs/2503.19786)).

### BF16 is still a deliberate choice at the top of the open stack

It is easy to over-read the FP8/FP4 trend. Several frontier-class 2025–2026 runs **did not** train in low precision:

- **Kimi K2 explicitly refused FP8 compute**: inputs of MoE up-projections and SwiGLU are *stored* compressed to FP8-E4M3 in 1×128 tiles with FP32 scales, but "due to potential risks of performance degradation… **we do not apply FP8 in computation**" ([§2.5](https://arxiv.org/html/2507.20534v1)).
- **GLM-4.5**: "our infrastructure supports **BF16 for training** while leveraging **FP8 for inference**" ([§2.3](https://arxiv.org/html/2508.06471v1)).
- **Llama 3 405B**: BF16 pretraining at 38–43% MFU; FP8 is inference-only.
- **Olmo 3**: **bfloat16 throughout**, ~43% MFU (7B) and ~41% (32B) on H100s, with `Float8Config(enabled=False)` in the shipped config ([§3.2](https://arxiv.org/abs/2512.13961)).

**You do not need FP8 to train a good model; you need it to train a cheap one.** And note what the low-precision adopters actually keep in high precision: DeepSeek-V3 maintains BF16/FP32 for "the **embedding module**, the **output head**, **MoE gating modules**, **normalization operators**, and **attention operators**" ([§3.3](https://arxiv.org/html/2412.19437v2)).

**FP4 at frontier scale is, as of Sept 2026, almost entirely NVIDIA.** The only publicly documented frontier-scale FP4 pretrains are **Nemotron 3 Super (120B-A12B, NVFP4, 25T tokens)** and **Nemotron 3 Ultra (550B-A55B, NVFP4, 20T tokens)**, the latter claiming "the largest-scale demonstration of stable and accurate NVFP4 training to date" with a "relative train loss gap against the BF16 segments **below 0.4% on average**" ([arXiv:2606.15007](https://arxiv.org/abs/2606.15007)). The NVFP4 recipe is four specific things ([arXiv:2509.25149](https://arxiv.org/abs/2509.25149)): keep ~15% of linear layers (mostly the **last** blocks) in BF16; apply **16×16 Random Hadamard transforms** to weight-gradient GEMM inputs; **2D 16×16 scaling for weights, 1D 1×16 for activations and gradients**; and **stochastic rounding for gradients only**. Head-to-head at 8B/1T tokens: **MXFP4 needs 36% more tokens (1.36T vs 1T) to match NVFP4's final loss.**

**What fraction of 2026 frontier training is FP8 or FP4? UNVERIFIED** — no survey or FLOP-weighted accounting exists, and closed labs disclose nothing. Qualitatively: FP8 is the incumbent default for large MoE, BF16 remains a substantial and sometimes deliberate minority, and FP4 pretraining is concentrated in NVIDIA's Nemotron line.

---

# Stage 3: Scaling laws and token budgets

## 3.1 Token budgets actually used

> **2026 default choice: massively overtrain relative to Chinchilla. 100–1000+ tokens per active parameter is the norm, because inference cost dominates training cost for any model people actually use.**

| Model | Total tokens | Active params | **Tokens / active param** | Source |
|---|---|---|---|---|
| Chinchilla-optimal | — | — | **20** (see correction below) | [arXiv:2203.15556](https://arxiv.org/abs/2203.15556) |
| Llama 3 8B | 15T | 8B | **1,875** | [arXiv:2407.21783](https://arxiv.org/abs/2407.21783) |
| SmolLM3 | 11.1T | 3B | **3,700** | [HF blog](https://huggingface.co/blog/smollm3) |
| Gemma 3 27B | 14T | 27B | **519** | [§2.2](https://arxiv.org/abs/2503.19786) |
| Gemma 3 1B | 2T | 1B | **2,000** | [§2.2](https://arxiv.org/abs/2503.19786) |
| DeepSeek-V3 | 14.8T | 37B | **400** | [§4.2.2](https://arxiv.org/abs/2412.19437) |
| Kimi K2 | 15.5T | 32.6B | **475** | [abstract](https://arxiv.org/abs/2507.20534) |
| Qwen3 | 36T | 22B (235B-A22B) | **1,636** | [§3](https://arxiv.org/abs/2505.09388) |
| GLM-4.5 | 23T | 32B | **719** | [abstract](https://arxiv.org/abs/2508.06471) |
| Llama 4 | >30T | 17B | **>1,765** | [Meta blog](https://ai.meta.com/blog/llama-4-multimodal-intelligence/) |
| MiniMax-M2 | 29.2T | 9.8B | **2,980** | [Raschka notes](https://sebastianraschka.com/blog/2026/minimax-m2-technical-report.html) |
| Olmo 3 7B | 5.93T + 100B + 50B | 7B | **~870** | [Table 35](https://arxiv.org/abs/2512.13961) |
| Olmo 3 32B | 5.5T + 200B + 100B | 32B | **~180** | [Table 35](https://arxiv.org/abs/2512.13961) |
| **DeepSeek-V4-Flash** | **32T** | 13B | **2,462** | [§4.2.2](https://arxiv.org/abs/2606.19348) |
| **DeepSeek-V4-Pro** | **33T** | 49B | **673** | [§4.2.2](https://arxiv.org/abs/2606.19348) |
| **Kimi K3** | **not disclosed** | 104.2B | — | [K3 report](https://github.com/MoonshotAI/Kimi-K3) |
| **Qwen3.8-Flash-Next** | **~1/3 of Qwen3.7-Plus** (absolute not disclosed) | 6B | — | [Qwen3.8 §4](https://github.com/QwenLM/Qwen3.8-Flash-Next) |

Note the 2026 disclosure regression: **Kimi K3 and Qwen3.8 both decline to state absolute pretraining token counts**, reporting relative scaling-efficiency gains instead. K3 re-fits "tokens-per-parameter ratio (TPP)" as a tuned quantity but never publishes the value it chose.

Two observations:
1. **The ratio scales inversely with active params** — small models get overtrained hardest, because their inference cost is what you are amortizing. The most extreme verified points: **Nemotron 3 Nano 30B-A3B at ~8,333 tokens/active-param (417× Chinchilla)** and **Qwen3-30B-A3B at ~12,000 (600×)**; among small dense models, **Qwen3-0.6B at 36T tokens = 60,000 TPP (3,000× Chinchilla)**.
2. **32–36T is the 2026 frontier pretraining budget**, up from ~15T in 2024–25. DeepSeek's abstract phrases it as "more than 32T diverse and high-quality tokens."

### The Chinchilla number itself needs an asterisk

Epoch AI's replication ([arXiv:2404.10102](https://arxiv.org/abs/2404.10102)) re-fit Hoffmann et al.'s parametric model and found the original fit statistically indefensible: a χ² test of the published parameters against the extracted data gives **p = 6e-50**, and the reported confidence interval on the allocation exponent was **~50× too narrow** — "[such intervals] would require over 600,000 experiments, while they likely only ran fewer than 500." The corrected fit:

```
L(N, D) = 1.8172 + 482.01·N^(-0.3478) + 2085.43·D^(-0.3658)
```

versus Hoffmann's E=1.6934, A=406.4, α=0.3392, B=410.7, β=0.2849. Note **B moves 5× with a standard error larger than half its own value** — B is effectively unidentified.

Two things follow. First, **the "20 tokens per parameter" figure never appears in the Chinchilla paper** — it is read off Table 3, and Table 3 is internally inconsistent: Approaches 1–2 give 19–27 TPP while **Approach 3 (the parametric fit) implies ~69 at 175B and ~94 at 1T.** That inconsistency is exactly what Epoch found. Second, the corrected point estimate is **~25.6 TPP** with an honest band of **4 to 40** — so the rule of thumb survives, but it was never as precise as it is quoted.

**And the practical trend has left it far behind.** Epoch measures tokens-per-parameter in open-weight LLMs growing **3.1× per year** since 2022 (90% CI 2.1×–4.9×), from ~10 in 2022 to ~300 in 2025: "Recent models have been trained with **20 times more data per parameter** than the optimal ratio suggested by the 2022 Chinchilla scaling laws" ([Epoch](https://epoch.ai/data-insights/training-tokens-per-parameter)).

**The 2026 consensus bands, by tier:**

| Tier | Tokens / active param | ×Chinchilla | Examples |
|---|---|---|---|
| Frontier flagship MoE | **350–1,000** | 18–50× | Nemotron 3 Ultra 364, DeepSeek-V4-Pro 653, GLM-5 713 |
| Mid / efficiency | **1,300–3,000** | 65–150× | Llama 4 Maverick 1,294, Scout 2,353, V4-Flash 2,462, MiniMax-M2 2,980 |
| Small / edge | **3,700–12,000+** | 185–600× | SmolLM3 3,733, Nemotron 3 Nano 8,333, Qwen3-30B-A3B 12,000 |
| Research / academic | **190–900** | 10–45× | Olmo 3-32B 191, Gemma 3 27B 519, Olmo 3-7B 864 |

**The right mental model: labs do their science near 20 TPP and ship at 350–1,000+.**

### Why overtrain: the 2026 formalization

The mechanism is settled — training is paid once, inference forever, and inference FLOPs scale with *active* parameters (~2N per token) while training scales with 6ND. Sardana et al. ([arXiv:2401.00448](https://arxiv.org/abs/2401.00448)) put numbers on it with 47 models: to match **30B-Chinchilla quality at 1e13 inference tokens**, train a **13.6B** model on **2.84×** the data and **cut total FLOPs by 28%**. (A separate example in the same paper: at 2T inference tokens, a Chinchilla-70B needs only 1.3% extra FLOPs than an equal-quality compute-optimal model but **costs 36% more** to serve.)

The 2026 successor extends this to test-time compute. **"Test-Time Scaling Makes Overtraining Compute-Optimal"** ([arXiv:2604.01411](https://arxiv.org/abs/2604.01411)) introduces **Train-to-Test (T²) scaling laws** jointly optimizing `C_train = 6ND` and `C_inf = 2Nk` over k inference samples, and finds that "optimal pretraining decisions shift **radically** into the overtraining regime, well outside the range of standard pretraining scaling suites" — a conclusion that survives post-training.

The binding constraint in 2026 is no longer economics but the **data wall**: Epoch's canonical estimate is that compute-optimal training exhausts human text around a **5e28 FLOP run, expected ~2028** ([Epoch](https://epoch.ai/publications/will-we-run-out-of-data-limits-of-llm-scaling-based-on-human-generated-data)).

Kimi K3 uses **TPP (tokens per parameter)** as a first-class quantity it re-fits per architecture, alongside batch size, LR, and model shape ([§3.2](https://github.com/MoonshotAI/Kimi-K3)). Qwen3.8's architecture ablations were run at a fixed **TPP = 300** ([§2.3](https://github.com/QwenLM/Qwen3.8-Flash-Next)).

## 3.2 Architecture changes buy token efficiency — and that is now the headline metric

The 2026 reports do not lead with "bigger." They lead with **scaling-curve shifts**:

- **Kimi K3: ~2.5× improvement in overall scaling efficiency over Kimi K2**, measured as fitted scaling-law curves of validation loss vs FLOPs on held-out OOD data, attributed jointly to KDA + AttnRes + Stable LatentMoE + refined data/training recipes ([§3.2, Fig. 7](https://github.com/MoonshotAI/Kimi-K3)).
- **Olmo Hybrid: ~2× data efficiency** — matches Olmo 3's MMLU with **49% fewer tokens** ([arXiv:2604.03444](https://arxiv.org/abs/2604.03444)).
- **Qwen3.8-Flash-Next (125B-A6B)** beats the previous-generation **Qwen3.7-Plus flagship (397B-A17B)** on 8 of 14 pretraining benchmarks and is competitive on the rest, at **1/3 the activated parameters, 1/3 the training tokens, and ~1/9 the training FLOPs** ([§4, Table 11](https://github.com/QwenLM/Qwen3.8-Flash-Next)). Verified examples: MMLU-Pro **73.23 vs 70.90**, SuperGPQA **51.36 vs 48.42**, BBH **90.87 vs 89.41**, SWEBench-Pretrain **50.99 vs 49.24**, MGSM **89.33 vs 85.42** — while trailing on MATH (72.78 vs 74.38) and MultiPL-E (79.09 vs 81.68).

If you are building an open-source project, this is the most encouraging fact in the document: **architecture and data work is currently worth more than compute.**

Qwen's own framing of why this worked is the thesis of the whole 2026 generation, and it is worth quoting in full:

> "The design reflects a conviction that **architecture, efficiency, and optimization form one coupled system**. GR supplies a rescaling that markedly improves training stability, and **that stability margin shifts the optimal learning rate and batch size upward**, improving both throughput and convergence. Phase-by-phase cost accounting directed the sparse-attention design into the indexer and concentrated the gated residual's expressiveness on the read. **Removing any axis from the loop would have admitted seemingly harmless shortcuts**: sparse writes that degrade after post-training, positional encoding that appears [fine during pretraining and fails after]…"
> — [Qwen3.8 §5](https://github.com/QwenLM/Qwen3.8-Flash-Next)

## 3.3 Data-constrained scaling and the rephrasing alternative

Muennighoff et al.'s data-constrained scaling result ([arXiv:2305.16264](https://arxiv.org/abs/2305.16264)) — that repeating data up to ~4 epochs is nearly as good as fresh data, with sharp decay after — is the backdrop for every 2026 data decision.

**Kimi K2 published the cleanest measurement of the alternative**: instead of repeating, *rephrase*. SimpleQA accuracy under three configurations at matched token count:

| # Rephrasings | # Epochs | SimpleQA Accuracy |
|---|---|---|
| 0 (raw wiki-text) | 10 | **23.76** |
| 1 | 10 | **27.39** |
| 10 | 1 | **28.94** |

"each corpora is rephrased at most twice." — [Kimi K2 §2.2, Table 1](https://arxiv.org/abs/2507.20534)

Their pipeline: a style- and perspective-diverse prompting scheme; **chunk-wise autoregressive rewriting** for long documents (split into 4096-token excerpts, then 256-token partial excerpts rewritten sequentially with preserved context, then concatenated — Fig. 4); and **fidelity verification** comparing semantic alignment of each rephrased passage to its source. Math documents are rewritten into a "learning-note" style following SwallowMath, plus translation of high-quality non-English math material into English.

Kimi K3 continues this: "Following the rephrasing recipe of Kimi K2, we rephrase **knowledge and mathematics corpora** with style and perspective-diverse prompting, chunk-wise autoregressive generation, and fidelity verification against the source documents" ([§3.1](https://github.com/MoonshotAI/Kimi-K3)).

K2 is candid about the limits: "the use of synthetic data as a strategy for continued scaling remains an active area of investigation. Key challenges include generalizing the approach to diverse source domains without compromising factual accuracy, minimizing hallucinations and unintended toxicity, and ensuring scalability."

### How much of your corpus should be synthetic? The best evidence says ~30%

| System | Disclosed synthetic share |
|---|---|
| **Nemotron-CC** | **1.9T / 6.3T = 30.2%** |
| **Nemotron-CC-v2** (2026) | **~49%** |
| **Phi-4** | **40% synthetic + 15% web rewrites ≈ 55% model-touched**; ~400B unweighted synthetic tokens across **50 broad types**, run for **13.8 epochs** |
| **Cosmopedia v1 / v2** | 100% (25B / 28B tokens) |
| **Qwen3** | "trillions" of 36T — no exact figure |
| **Kimi K2 / K3**, Llama 4, OpenAI, Anthropic, Google | **not disclosed** |
| **Microsoft MAI-Base-1** | **zero — and it actively removes AI-generated text** |

Sources: [Nemotron-CC](https://arxiv.org/html/2412.02595v1); [Nemotron-CC-v2 card](https://huggingface.co/datasets/nvidia/Nemotron-CC-v2); [Phi-4 Table 5](https://arxiv.org/html/2412.08905v1); [Cosmopedia](https://huggingface.co/blog/cosmopedia); [Qwen3](https://arxiv.org/html/2505.09388v1); [MAI-Base-1](https://microsoft.ai/pdf/mai-thinking-1.pdf).

The largest controlled study — **>1000 LLMs, >100k GPU-hours** ([arXiv:2510.01631](https://arxiv.org/abs/2510.01631)) — finds **~30% rephrased synthetic / ~70% natural web** optimal, worth a **5–10× speedup to equivalent validation loss**. That this lands exactly on Nemotron-CC's disclosed 30.2% is the strongest anchor available.

**Nemotron-CC uses five rephrasing styles, not four** ([arXiv:2412.02595v1](https://arxiv.org/html/2412.02595v1)): *Wikipedia-style rephrasing* (applied to low-quality docs, "reduces errors and redundancies and improves formatting"), and four applied to high-quality docs — *Diverse QA pairs*, *Distill* ("rewrite into a concise and clear passage"), *Extract Knowledge* ("disregard uninformative content"), and *Knowledge List*. Rewriter: **Mistral-NeMo-12B-Instruct in FP8**.

**The generator-size consensus collapsed downward in 2026.** FinePhrase (486B synthetic tokens, 12.7 GPU-years, [arXiv:2604.13977](https://arxiv.org/abs/2604.13977), COLM 2026) finds a **generator above 1B parameters buys nothing**, and that **structured output formats win** — table format is worth **+7.5 pp on ARC**, math format **+1.5 GSM8K / +11.2 SQuAD** vs DCLM. Their optimal synthetic share is **60–80%** depending on format. RePro ([arXiv:2510.10681](https://arxiv.org/abs/2510.10681)) shows a **1B RL-trained rephraser doubles the gains of a prompted 70B**. Cost of synthetic pretraining fell roughly **30×** from 2024 (Mixtral-8x7B, Nemotron-340B) to 2026.

Also note **Cosmopedia's measured failure**: Cosmo-1B beat TinyLlama-1.1B but showed "performance gaps compared to Phi-1.5," with self-reported hallucination in some subsets. **BeyondWeb beats Cosmopedia by 5.1 pp** across 14 benchmarks, and a **3B model on 180B BeyondWeb tokens outperforms an 8B model on the same budget of Cosmopedia** ([arXiv:2508.10975](https://arxiv.org/abs/2508.10975)).

### Is model collapse a real production risk? Largely no — but read the actual experiment

The Nature paper everyone cites ([Shumailov et al. 2024](https://arxiv.org/html/2305.17493v3)) trained OPT-125M on wikitext2 while **recursively replacing** real data with generated data. Perplexity went 20 → 28 when no original data was preserved — but the same paper shows that **preserving a random 10% each generation already averts most degradation.**

- **"Is Model Collapse Inevitable?"** ([arXiv:2404.01413](https://arxiv.org/abs/2404.01413)): "**accumulating** the successive generations of synthetic data **alongside** the original real data **avoids model collapse**," with a finite test-error bound independent of iteration count.
- **The counterpoint**: "Strong Model Collapse" ([arXiv:2410.04840](https://arxiv.org/abs/2410.04840)) argues **as little as 1% synthetic** can cause scaling-law saturation.
- **Information-theoretic condition**: synthetic data helps iff the loop is **"information-open"** — otherwise the data-processing inequality guarantees decay ([arXiv:2605.16379](https://arxiv.org/abs/2605.16379)). Verifier-in-the-loop provably escapes collapse ([arXiv:2510.16657](https://arxiv.org/abs/2510.16657)).

**Verdict: the canonical collapse result is a narrow *replacement* experiment on a 125M model. Every production pipeline accumulates and verifies — which is exactly the regime proven to be bounded.** Kimi K2's 10-rephrasings-×-1-epoch beating 10 epochs of the original by +5.18 SimpleQA points is the cleanest production-scale refutation of naive collapse fears.

### The 2026 counter-current: a frontier run with zero synthetic data and 14.9% web

Microsoft's MAI-Base-1 is the most detailed frontier data disclosure of 2026, and it inverts nearly every assumption in this section ([tech report](https://microsoft.ai/pdf/mai-thinking-1.pdf)):

| Source | Unique (T) | Training (T) | Mix % | Avg epochs |
|---|---|---|---|---|
| **Code** | 7.4 | 16.4 | **54.6%** | 2.22× |
| STEM | 2.2 | 4.7 | 15.8% | 2.17× |
| **Web text** | 8.1 | 4.5 | **14.9%** | 0.55× |
| Math | 0.3 | 1.6 | 5.4% | **5.28×** |
| PDFs | 2.7 | 1.4 | 4.7% | 0.53× |
| Books/journals | 0.6 | 0.9 | 3.1% | 1.65× |
| Multilingual | 8.1 | 0.5 | 1.6% | 0.06× |
| **Total** | **29.2** | **30.0** | 100% | 1.03× |

> "We choose **not to use language-model-generated synthetic data for pre-training**, and we make an effort to avoid and remove AI-generated content within collected data sources… we score pages with a proprietary **AI-content detection model**… those domains are filtered out."

**Web text is 14.9% of a 30T-token frontier run, and code is 54.6%.** Maximum repetition is capped at 8; math is deliberately repeated 5.28×. This is the strongest published evidence that the "web-heavy → code/math-enriched" curriculum in §4.1 may be a transitional form rather than the end state.

---

# Stage 4: The data pipeline

## 4.1 The canonical stage structure

> **2026 default choice: four stages — (1) broad web-heavy pretraining, (2) a reasoning/STEM-enriched stage, (3) a short high-quality midtraining/decay stage carrying instruction and thinking data, (4) long-context extension folded into the cooldown.**

Verified stage structures, with token counts:

### SmolLM3 (3B, 11.1T total) — the most completely disclosed small-model curriculum

| Stage | Tokens | Web | Code | Math |
|---|---|---|---|---|
| Stage 1 | 0 → 8T | **85%** (12% multilingual) — FineWeb-Edu, DCLM, FineWeb2 | **12%** — The Stack v2, StarCoder2 | **3%** — FineMath3+, InfiWebMath3+ |
| Stage 2 | 8T → 10T | **75%** | **15%** — adds Stack-Edu | **10%** — FineMath4+, MegaMath |
| Stage 3 (decay) | 10T → 11.1T | **63%** | **24%** | **13%** |

Then: **long-context extension 100B tokens** (50B 4k→32k @ RoPE θ 1.5M; 50B 32k→64k @ θ 5M), and **reasoning midtraining 35B tokens** from OpenThoughts3-1.2M + NVIDIA Nemotron data, trained **4 epochs (~140B tokens)** with ChatML and wrapped packing.
Training: 384 H100s, 24 days, global batch **2.36M tokens**, LR 2e-4, WSD.
— [SmolLM3 blog](https://huggingface.co/blog/smollm3)

### Qwen3 (36T total)

- **S1 General**: >30T tokens @ seq 4096, 119 languages.
- **S2 Reasoning**: ~5T "higher-quality" tokens @ seq 4096, increased STEM/coding/reasoning/synthetic proportion. **Learning-rate decay is accelerated during this stage.**
- **S3 Long Context**: "hundreds of billions of tokens" @ seq 32,768. Corpus composition is explicit: **75% of text between 16,384–32,768 tokens; 25% between 4,096–16,384.** RoPE base 10,000 → 1,000,000 via ABF; YaRN + Dual Chunk Attention at inference for a further 4× length.
— [Qwen3 §3.2](https://arxiv.org/abs/2505.09388)

### Olmo 3 (the fully-open reference)

| Stage | Mix | Tokens (7B) | Tokens (32B) | Pool it is drawn from |
|---|---|---|---|---|
| Pretraining | **Dolma 3 Mix** | 5.93T | 5.5T | ~9.3T cleaned source corpus |
| Midtraining | **Dolma 3 Dolmino Mix** | 100B | 100B **run twice** | ~2.2T high-quality pool |
| Long-context | **Dolma 3 Longmino Mix** | 50B | 100B | 639B long-document pool |

Notes:
- Dolma 3 corpus is "~9.3-trillion-token… drawn from web pages, **science PDFs processed with olmOCR**, codebases, math problems and solutions, and encyclopedic text"; Dolma 3 Mix is the ~6T actual training mix with a higher proportion of code and math than earlier Dolma releases.
- **The 32B runs midtraining twice with different data-order seeds and averages the resulting checkpoint weights** — model souping as a production step ([Table 35 caption](https://arxiv.org/abs/2512.13961)).
- Long-context: 65,536 sequence length, batch 4.19M/8.39M tokens, 200 warmup steps, linear decay to 0.
- The olmOCR science-PDF collection is the enabler: **22.3M documents above 8K tokens (640B tokens)** and **4.5M documents above 32K tokens (380B tokens)** — "the largest openly available for long-context research."
- Olmo 3 also released **smaller sample mixes for low-compute experimentation: 150B for pretraining and 10B for midtraining.**
— [Olmo 3 §2.1, §3.4–3.6](https://arxiv.org/abs/2512.13961), [Ai2 blog](https://allenai.org/blog/olmo3)

### DeepSeek-V4 (32–33T) — sequence-length curriculum inside pretraining

Rather than a separate long-context phase, V4 **extends sequence length four times during pretraining**: starts at **4K → 16K → 64K → 1M**. Sparse attention is phased in: "we first **warm up the model with dense attention for the first 1T tokens**, and introduce sparse attention at the sequence length of 64K and keep sparse attention during the rest of the training. When introducing attention sparsity, we first set a short stage to warm up the **lightning indexer** in CSA, and then train the model with sparse attention for most of the training." V4-Pro "starts with a longer stage of dense attention" ([§4.2.2](https://arxiv.org/abs/2606.19348)).

Also: V4 packs documents from different sources "into appropriate sequences to minimize sample truncation" and, unlike V3, **employs sample-level attention masking during pre-training** (i.e., intra-document masking — same choice as SmolLM3). Vocabulary stays at **128K**, with a few new special tokens for context construction; token-splitting and **Fill-in-Middle (FIM)** are inherited from V3 ([§4.1](https://arxiv.org/abs/2606.19348)).

Three data-curation points from V4 worth extracting ([§4.1](https://arxiv.org/abs/2606.19348)):

1. **Explicit anti-model-collapse filtering**: "For web-sourced data, we implement filtering strategies to remove **batched auto-generated and templated content**, thereby mitigating the risk of model collapse." As the web fills with model output, this becomes a standing pipeline stage rather than a one-off.
2. **Agentic data goes in midtraining**, not post-training: "we further enhance the coding capabilities of DeepSeek-V4 series by **incorporating agentic data during the mid-training phase**." Same principle as Olmo 3 putting instruction data and thinking traces into the base model's midtrain.
3. **Long-document curation is targeted, not scraped**: "we place a particular emphasis on long-document data curation, prioritizing **scientific papers, technical reports, and other materials that reflect unique academic values**" — the same bet Olmo 3 makes with its olmOCR science-PDF pool.

### Kimi K3 — long context in the cooldown

Four-stage progressive context extension: **8K → 64K during pretraining; 256K → 1M during the cooldown phase.** The rationale is budget: "Concentrating the costly long-sequence computation within a small fraction of the overall training budget keeps the curriculum economical while still allowing the model to adapt gradually to increasingly long-range dependencies" ([§3.3, §3.4](https://github.com/MoonshotAI/Kimi-K3)).

### GLM-4.5 (23T)

15T general corpus, then a **7T code & reasoning corpus**, followed by midtraining ([§2.3](https://arxiv.org/abs/2508.06471)).

### Gemma 3 — token budget scales with model size

**14T for 27B, 12T for 12B, 4T for 4B, 2T for 1B.** "The increase in tokens accounts for the mix of images and text used during pre-training" ([§2.2](https://arxiv.org/abs/2503.19786)).

## 4.2 Long-context data is a curation problem, not just a length problem

> **2026 default choice: upsample genuinely long documents, then *synthesize* long-context tasks whose answer requires attending across the whole window. Length alone does not confer long-range capability.**

Kimi K3 is the most explicit:

> "Long documents and videos from natural sources contain a substantial amount of low-quality content, including near-duplicates, binary blobs, truncated files, video clips, and invalid machine-generated logs. We therefore process them through a dedicated cleaning pipeline that combines **exact and fuzzy deduplication, supplemented by perceptual hashing over frames for video**, together with heuristic and classifier-based quality filtering, and structural validation. Because genuinely long and coherent documents and videos are scarce relative to short text, **we upsample them**… **Length alone, however, does not confer long-range capability.** To address this, we synthesize additional long-context data by carefully **permuting and concatenating multimodal documents and sub-tasks, so that the embedded tasks can be solved only by attending to information scattered across the full 1M-token context.** This trains the attention mechanism at the intended scale and **prevents it from degenerating into local patterns**."
> — [Kimi K3 §3.4](https://github.com/MoonshotAI/Kimi-K3)

Olmo 3 independently ran "Experiments with Synthetic Augmentation" for long-context and a dedicated section on "Choosing Data Mix and Token Budget" for the extension stage ([§3.6.2–3.6.3](https://arxiv.org/abs/2512.13961)).

### Long-context is cheap — under 1% of tokens in most 2026 runs

| Model | Long-context tokens | Total | **% of budget** |
|---|---|---|---|
| Nemotron 3 Ultra | 33B | 20T | **0.17%** |
| Nemotron 3 Super | 51B | 25T | **0.20%** |
| Olmo 3 7B (Longmino) | 50B | 6.08T | **0.83%** |
| SmolLM3 | 100B | 11.2T | **0.89%** |
| Olmo 3 32B | 100B | 5.8T | **1.7%** |
| **Llama 3** | **800B** | 15.6T | **5.1%** |
| GLM-5 (mid-training) | 1.55T | 28.5T | **5.4%** |
| MiniMax-M2 (whole decay phase) | 9.3T | 29.2T | **31.8%** |

Llama 3's 5.1% now looks like an artifact of the 2024 approach, where long-context was a separate dedicated phase. The 2026 pattern folds it into the cooldown, where the FLOPs are amortized against work you were doing anyway.

**Budget the engineering, not the FLOPs.** Olmo 3 used **context parallelism CP = 8** for the extension stage versus none for pretraining, and throughput dropped from 7.7K to 4.0K tokens/s/device (7B) and 2.0K to 1.3K (32B) ([Tables 34–35](https://arxiv.org/abs/2512.13961)).

**Do not train purely on long documents.** Olmo 3's Longmino mix is **34% long-context / 66% high-quality short-context** carried over from Dolmino; Nemotron 3 Ultra's long-context phase is **92% @ 1M / 8% @ 4K**. Without the short-context ballast, short-context evals regress. Llama 3's discipline was to advance a length stage only when short-context evals had fully recovered **and** needle-in-a-haystack was perfect ([arXiv:2407.21783](https://arxiv.org/html/2407.21783v3)).

The long-document supply problem is solved with PDFs: olmOCR gave Olmo 3 **22.3M documents above 8K tokens (640B)** and **4.5M above 32K tokens (380B)**.

## 4.3 Quality filtering and mixing

> **2026 default choice: model-based quality classifiers over heuristics; instance-level rather than domain-level mixture optimization; ablate mixtures on small proxy models.**

**Qwen3's approach is the most aggressive disclosed version of this**, and it is the one to copy:

> "We have developed a **multilingual data annotation system**… annotating **over 30 trillion tokens** across multiple dimensions such as **educational value, fields, domains, and safety**. These detailed annotations support more effective data filtering and combination. Unlike previous studies that optimize the data mixture at the **data source or domain level**, our method optimizes the data mixture at the **instance level** through extensive ablation experiments on small proxy models with the fine-grained data labels."
> — [Qwen3 §3.1](https://arxiv.org/abs/2505.09388)

Kimi K3: "Each domain is filtered by a combination of **rule-based heuristics, classifier-based quality scoring, and deduplication**, with **domain-specific sampling rates determined by ablation studies on smaller models**" ([§3.1](https://github.com/MoonshotAI/Kimi-K3)).

Gemma 3: filtering to reduce unsafe utterances and PII; **decontamination of evaluation sets from the pretraining mixture**; "reduce the risk of recitation by minimizing the proliferation of sensitive outputs"; and a **quality reweighting step** ([§2.2](https://arxiv.org/abs/2503.19786)).

Olmo 3's two named novelties for pretraining token selection are **token-constrained mixing** and **quality-aware upsampling**, plus "new tooling for fast and scalable **global deduplication at the trillion-token scale**" ([§2.1](https://arxiv.org/abs/2512.13961)).

### How much filtering actually buys, measured on the same pool

DCLM ran the decisive same-architecture ablation at **7B-1×** ([arXiv:2406.11794v4](https://arxiv.org/html/2406.11794v4)):

| Filter | CORE |
|---|---|
| No filtering | 35.0 |
| fastText with GPT-3-style references (Wikipedia + OpenWebText2 + books) | 37.5 |
| **fastText on OH-2.5 + ELI5** | **41.0** |

**Choosing the positive seed for the classifier is worth +3.5 CORE by itself**, at identical architecture and identical pool. That is a bigger effect than most architecture changes in this document, and it costs almost nothing.

Head-to-head at 7B, same harness, 276B tokens: **DCLM-Baseline CORE 48.7 / MMLU 51.9 vs FineWeb-Edu CORE 41.9 / MMLU 37.3.** FineWeb-Edu's advantage is confined to education-skewed evaluations.

**Nemotron-CC's classifier-ensemble ablation** (8B @ 1T) shows why ensembling wins — it is a recall play, not a precision play ([arXiv:2412.02595v2](https://arxiv.org/html/2412.02595v2)):

| Classifier | HQ yield | MMLU |
|---|---|---|
| FineWeb-Edu | 8% | 55.4 |
| DCLM fastText | 11% | 56.0 |
| Nemotron-340B-labelled | 14% | 54.9 |
| **Ensemble (max of integer scores)** | **25%** | **56.4** |

**Only 10% of documents are high-quality under both FineWeb-Edu and DCLM** — the classifiers disagree far more than they agree. (FineWeb-Edu was dropped from the shipped ensemble over the Llama-3 annotation license.)

Nemotron-CC's other shipped design rule: **remove heuristic filters only on the high-quality split**, keeping them on the low-quality split → **MMLU 55.4 → 57.5 (+2.1)**.

### The keep-rate should scale with your compute budget

Apple's BETR study fit an explicit law across 500+ models spanning 1e19–1e22 FLOPs ([arXiv:2507.12466](https://arxiv.org/html/2507.12466v1)):

```
F_opt(C) = 4e-5 · C^0.25
```

| Compute | Optimal fraction kept |
|---|---|
| 1e20 FLOPs | **top 3%** |
| 1e22 FLOPs | **top 10%** |
| 1e23 FLOPs | **top 30%** |

This answers the question everyone gets wrong when copying a recipe: **DCLM's and FineWeb-Edu's top-10% cut is calibrated near 1e22 and is roughly 3× too aggressive at 1e23.** Conversely, if you are running a 1e20-FLOP ablation, you should be filtering *harder* than the published recipes. BETR itself reports compute multipliers of **1.8×–2.5×** over DCLM-Baseline and **4.7×** over unfiltered, and notes decontamination costs only **−0.2**.

### The expiry date on filtering

**"A Bitter Lesson for Data Filtering"** ([arXiv:2605.19407](https://arxiv.org/abs/2605.19407), May 2026) is the most consequential filtering result of 2026 and deserves to be read in full before you over-invest:

> "with enough compute, **the best data filter is no data filter**… sufficiently trained large parameter models not only tolerate low-quality and distractor data, but in fact **benefit from** nominally 'poor' data."

Models 15M → 7B, up to 100B tokens, against five filters (DCLM-Baseline, RefinedWeb, English, Repetition, Stop-word). "For sufficiently large models (**330M+**), the unfiltered pool outperforms **all five filters** after sufficiently many optimization steps." Injected distractors (random strings, shuffled-word documents) hurt 15M models, the gap closes with scale, and shuffled-word documents eventually *help*.

Their fitted compute scaling laws (R² > 0.99) put the global no-filter crossover near **1e30 FLOPs**, against frontier training near **5e26** — and the authors concede: "**when compute is a bottleneck, we expect filtering to still be important.**"

**Reading: aggressive filtering is correct for every regime a reader of this document is in, and has a predicted expiry roughly 3.5 orders of magnitude out.**

### Mixing: the small-scale ablation trap

Two 2026 results explain why proxy mixing experiments often fail to transfer:

**Repetition rate shifts as you scale** ([arXiv:2606.07597](https://arxiv.org/abs/2606.07597), EMNLP 2026): "because high-quality datasets are small, their **repetition rate changes as the training budget grows**, shifting the optimal mixture in ways that small-scale proxy experiments do not anticipate." A single **repetition-controlled** run at **1/16 of target tokens** lands within **0.10** of the optimum, versus error **0.85** uncontrolled. Without the fix, matching that accuracy requires 19–94% of the target budget.

**Olmix** (Ai2 + Stanford, [arXiv:2602.12237](https://arxiv.org/html/2602.12237v1), [code](https://github.com/allenai/olmix)) ablates the *design space* of proxy-swarm mixing rather than proposing one point in it. Winning configuration: swarm of **K ≥ 3(m+1)** proxy runs at **30M params / 3B tokens** each; **log-linear regression fit per task**; **sparse** swarm distributions for topic-level mixing and **dense** for source-level; **data-repetition constraints enforced inside the optimizer**; exact solver with KL regularization 0.05. Validated at 1B / 100B tokens on DCLM partitioned into 24→64 topic domains: **+11.6% over the natural distribution at 74% fewer proxy runs.**

For context, **RegMix** ([arXiv:2407.01492](https://arxiv.org/html/2407.01492v2)) gets 7B @ 100B tokens from **54.5 → 56.5 (+2.0)** over a human mix, simulating **1,000,000 candidate mixtures in under 10 CPU-seconds** and averaging the top 100. And Meta's comparison across nine baselines found that **"token-count heuristics outperform manual and learned mixes"** ([arXiv:2501.11747](https://arxiv.org/abs/2501.11747)) — so always report against stratified/token-proportional sampling, not against a hand-tuned mix.

### What filtering costs, normalized per trillion tokens

| Approach | **Cost per 1T tokens** |
|---|---|
| fastText bigram classifier | **~67 CPU-hours**, no GPU |
| Embedding + linear head (FineWeb-Edu style) | **~400 H100-hours** |
| 0.5B distilled LLM annotator, 12 fields (EAI-Distill) | **~3,750 GPU-hours** |
| 32B LLM annotator (Qwen2.5-32B) | **~36–50× the 0.5B** |

Sources: [Ultra-FineWeb](https://arxiv.org/html/2505.05427v1); [FineWeb blog](https://huggingface.co/spaces/HuggingFaceFW/blogpost-fineweb-v1); [Essential-Web](https://arxiv.org/html/2506.14111v2).

For scale: the **entire FineWeb project including all ablations was ~80,000 H100-hours ≈ 1.6% of a 5M-GPU-hour training run**, for a ~7.9× token-efficiency multiplier ([analysis](https://huggingface.co/blog/maxidl/spend-compute-on-data)).

And the cheap verification trick from Ultra-FineWeb: evaluate a candidate filter by continuing a **near-converged 1B model for 10B tokens** rather than training from scratch — **110 H100-hours instead of 1,200** per candidate ([arXiv:2505.05427v1](https://arxiv.org/html/2505.05427v1)).

### Heuristic filters: what survived into 2026

| Rule | Status | Evidence |
|---|---|---|
| Gopher repetition filters | **Kept** at original thresholds | [FineWeb](https://arxiv.org/html/2406.17557v2) |
| Gopher quality filters | **Kept** (full Gopher beats natlang-only) | [RedPajama-V2](https://arxiv.org/html/2411.12372v1) |
| **C4 terminal-punctuation filter** | **Dropped** — "removes around **30% of all tokens**" | [FineWeb](https://arxiv.org/html/2406.17557v2) |
| C4 curly-bracket / word-length / lorem-ipsum / javascript / policy | Kept (each −0.5% to −4.3%) | same |
| Line-level C4 filters | Marginal — lowers perplexity, "negligible effect on aggregated benchmark scores" | [RedPajama-V2](https://arxiv.org/html/2411.12372v1) |
| Heuristics on the high-quality split | **Dropped** (+2.1 MMLU) | [Nemotron-CC](https://arxiv.org/html/2412.02595v2) |
| Uniform thresholds across domains | **Being replaced by domain-specific thresholds** — uniform rules cause "systematic over-filtering of domain-specific content" | [CuraWeb](https://arxiv.org/pdf/2607.22662) |

### Four documented failure modes of model-based filtering

Worth knowing before you trust a classifier:

1. **Filters reward surface form, not substance.** Wikipedia-style reformatting flips the FineWeb-Edu classifier's decision on **~7% of documents**, letting low-quality content clear the threshold ([arXiv:2605.23721](https://arxiv.org/abs/2605.23721)).
2. **They collapse domain diversity.** FineWeb-Edu has the lowest domain-diversity entropy of every corpus compared (**H_L1 = 3.73**); rule+model filtering drops documents **34.18B → 23.29B (−31.9%)**. Gopher-style surface rules systematically kill STEM — STEM share rises **19.74% → 33.58%** and math **1.66% → 3.83%** once model signals replace surface rules ([CuraWeb](https://arxiv.org/pdf/2607.22662)).
3. **Neural filtering amplifies benchmark leakage.** The Gaperon project (1.5B/8B/24B, 2–4T tokens) found "filtering for linguistic quality enhances text fluency… but yields subpar benchmark results," and that "usual neural filtering can unintentionally **amplify benchmark leakage**" ([arXiv:2510.25771](https://arxiv.org/abs/2510.25771)).
4. **Conventional quality priors are sometimes inverted.** By register: the "Opinion" register (reviews, opinion blogs) *helps*, while **News "results in subpar performance."** Best combination measured: How-to + Informational Description + Opinion ([arXiv:2504.01542](https://arxiv.org/abs/2504.01542)).

## 4.4 Deduplication — and the counterintuitive result everyone cites

> **2026 default choice: MinHash-LSH on 5-grams at Jaccard ~0.75–0.80, plus exact/substring dedup. Deduplicate globally, but then re-weight — do not simply delete and move on.**

Published parameters:

| Pipeline | Exact | Fuzzy |
|---|---|---|
| **FineWeb** | — | MinHash **5-grams, 112 hashes, 14 bands × 8 rows, Jaccard 0.75**, **per-dump** |
| **DCLM** | Bloom filter, min-ngram 13, threshold 0.8 | MinHash 5-gram, 1395 perms (93 buckets × 15) for experiments |
| **Olmo 3 / Dolma 3** | global doc-hash → removes **67%** | MinHash **5-gram, 26 bands × 11, Jaccard 0.80** → **−23%**; then **suffix-array substring** dedup → **−14% of bytes** |
| **Dolma 1** | Bloom filter, 3 stages | URL **−53.2%**, doc **−14.9%**, **paragraph −18.7%** |
| **Zyda-2** | — | MinHash sig 128, **character 25-grams**, 8 bands ≈ Jaccard 0.85 → **−32%** cross-dataset |
| **Nemotron-CC** | exact-substring | **global** fuzzy across all 99 snapshots |
| **SemDeDup** | — | OPT-125M embeddings, **k=11,000 clusters** on C4, keep lowest cosine-to-centroid |

Sources: [FineWeb](https://arxiv.org/html/2406.17557v2); [DCLM](https://arxiv.org/html/2406.11794v4); [Olmo 3](https://arxiv.org/html/2512.13961v2); [Dolma](https://arxiv.org/html/2402.00159v2); [Zyda-2](https://arxiv.org/pdf/2411.06068); [SemDeDup](https://arxiv.org/html/2303.09540v3).

### Why global dedup made FineWeb *worse* — and why that is not the lesson people take from it

FineWeb deduplicated all 96 CC dumps against each other and got a model far below RefinedWeb. Their diagnosis ([FineWeb blogpost](https://huggingface.co/spaces/HuggingFaceFW/blogpost-fineweb-v1)):

- Global dedup removed **>90% of the base-filtered data from the oldest dumps**.
- They then trained separately on the **removed 90%** and the **kept 10%**: "the data that was kept (10% of the original data) was actually **worse** than the 90% of data we removed."
- Hypothesis: "data that does **not** find a duplicate match in any other dump might actually be **worse quality / more out of distribution**." Being duplicated across crawls is a **weak positive quality signal** — someone thought it worth re-hosting. Naive global dedup therefore *upsamples the un-duplicated low-quality tail*.
- Their fix: deduplicate **each dump independently** → "it now matches RefinedWeb's performance."

**But two projects globally deduped successfully by re-weighting afterwards.** TxT360 globally deduped 99 CC snapshots plus 14 non-web sources and then applied a deliberate upsampling recipe, **outperforming FineWeb 15T** ([LLM360](https://www.llm360.ai/news/txt360-blogpost.html)). Nemotron-CC also deduped globally across all 99 snapshots and ended with **4× more unique real tokens than DCLM** at better quality.

**So the correct 2026 statement is: global dedup is right, but duplication count is signal, not noise — keep it and use it to re-weight.**

### A third data point that complicates the story further

Zyphra measured that **DCLM contains ~80% internal duplicates**, and FineWeb-Edu similarly — neither was internally deduplicated, and both teams reported that deduplicating didn't help. Zyphra's hypothesis: "a highly duplicated dataset like FineWeb-Edu [is] equivalent to performing a **2–5 epoch shuffled run on a much smaller deduplicated dataset**," and dedup's historical benefit was largely that it acted as an **implicit spam filter** — an effect that "diminishes and disappears in the evaluation noise" once a model-based quality classifier is already applied. Their own caveat: this "may not hold true at larger scales where language models can perform much more sample-efficient memorization" ([Zyphra](https://www.zyphra.com/our-work/building-zyda-2)).

This dovetails with the repeat-data literature: **"repeating existing aggressively filtered datasets for up to ten epochs can outperform training on the ten times larger superset for a single epoch"**, and per-document repeat counts beat uniform repetition ([arXiv:2503.07879](https://arxiv.org/abs/2503.07879)).

## 4.5 Midtraining as a distinct discipline

> **2026 default choice: a short (1–2% of tokens), high-quality, LR-decaying stage that deliberately seeds post-training capabilities — including instruction data and thinking traces in the *base* model.**

Olmo 3's midtraining methodology is the clearest published framework:

- **Dolmino Mix: 100B training tokens sampled from a ~2.2T-token pool** of high-quality math, science, code, instruction-following, and reading-comprehension data.
- A "**two-part methodological framework** combining (1) lightweight, distributed feedback loops on individual data sources, with (2) **centralized integration tests** to assess candidate mixes on base model quality **and post-trainability**."
- **"Intentional inclusion [of] instruction data and thinking traces to lay groundwork for post-training."**
- A dedicated **decontamination** step (§3.5.3).
— [Olmo 3 §2.1, §3.5](https://arxiv.org/abs/2512.13961)

That last point is the key 2026 shift: midtraining is no longer "anneal on good data," it is **"make the base model easy to post-train."** Olmo explicitly measures *post-trainability* as a selection criterion for the mix.

### Midtraining became a named research area in October 2025

Three foundational papers within three weeks: **Mid-Training of LLMs: A Survey** ([arXiv:2510.06826](https://arxiv.org/abs/2510.06826), taxonomy of data distribution / LR scheduling / long-context extension); **A Survey on LLM Mid-Training** ([arXiv:2510.23081](https://arxiv.org/abs/2510.23081), "provides a formal definition"); and **Midtraining Bridges Pretraining and Posttraining Distributions** ([arXiv:2510.14865](https://arxiv.org/abs/2510.14865), mechanism = distributional bridging). Followed in 2026 by **PRISM** ([arXiv:2603.17074](https://arxiv.org/abs/2603.17074)), a controlled study across seven base models.

The working definition: an intermediate stage of "multiple annealing-style phases that refine data quality, adapt optimization schedules, and extend context length," delimited by **budget** (pretraining 5–30T vs midtraining ~0.5–5T) and **LR regime** (peak ~3e-4 vs a lower, annealing-decay regime), where the purpose shifts from breadth and memorization toward "abstraction by emphasizing structured reasoning, factuality, and instruction following" ([arXiv:2510.06826](https://arxiv.org/html/2510.06826v1)).

### What actually goes in: Dolma 3 Dolmino, the only fully itemized 2026 midtrain mix

| Category | Tokens | % |
|---|---|---|
| Web (high-quality + STEM-heavy) | 27.4B | **27.5%** |
| Code (StackEdu **FIM** + CraneCode synthetic) | 20.0B | **20.0%** |
| Math (synthetic: Dolmino / Crane / MegaMatt / TinyMATH) | 19.2B | **19.2%** |
| **Synthetic QA** (Reddit→Flashcards, Nemotron QA, Wiki→RCQA) | 13.9B | **13.9%** |
| Reasoning traces (QWQ, Llama-Nemotron, OpenThoughts2, Gemini) | 8.34B | **8.3%** |
| **Instruction data** (Flan 5.0B + Tulu 3 SFT 1.1B) | 6.1B | **6.1%** |
| olmOCR science PDFs (high-quality) | 4.99B | **5.0%** |

([dataset card](https://huggingface.co/datasets/allenai/dolma3_dolmino_mix-100B-1025))

Every element the concept promises is present and quantified: **literal SFT data inside pretraining (6.1%)**, **multiple-choice/benchmark-format exposure (~8.9%)**, **math+code upsampling (39.2%)**, and **reasoning-trace exposure (8.3%)**. Most striking: **web falls from 76.1% of the pretraining mix to 27.5% — a 2.8× de-weighting in a single stage transition.**

### Measured effect sizes

| Study | Effect |
|---|---|
| **OLMo 2 Dolmino** (7B, 3×50B anneals, weights averaged) | **MMLU 59.8 → 63.7 (+3.9)**, **GSM8K 24.1 → 67.5 (+43.4)**, average 53.0 → 62.9 (**+9.9**) |
| OLMo 2 13B | MMLU 63.4 → 67.5, GSM8K 37.3 → 75.1, average **+9.4** |
| **Llama 3** 40B annealing probe (8B) | GSM8k **+24.0%**, MATH **+6.4%**; **negligible at 405B** |
| **PRISM** (7 base models, ~27B HQ tokens) | "**+15 to +40 points on math, +5 to +12 on code, +6 to +13 on science** while preserving general performance" |
| **MiniCPM / WSD** ([arXiv:2404.06395](https://arxiv.org/html/2404.06395v2)) | Putting SFT data at the **start of decay** vs SFT-only: **C-Eval 40.0 → 52.6 (+12.6)**. And the budget rule: "**a decay of 10% of total tokens is sufficient; 2.5% falls short**" |
| **Nemotron-H** | Phased data ordering beats random ordering by **3.4%** |
| **Midtraining Bridges** | Gains are "**largest for domains distant from general pretraining data, such as code and math**"; early introduction tolerates high mixture weight, late introduction requires low |

**GSM8K 24.1 → 67.5 from a 100B-token stage is the largest single-stage gain documented anywhere in this document.** If you have a fixed budget and are deciding where to spend a marginal 2%, spend it here.

### The midtraining budget has grown ~100,000× in two years

Llama 3 405B's final anneal was **40M tokens (0.0003%)**. By 2026: Olmo 3 **2.5%**, GLM-5 **5.4%**, SmolLM3 **~11.9%**, Qwen3 **~15%**, Nemotron 3 Ultra **25.2%**, **MiniMax-M2 31.8%** (its entire 9.3T decay phase out of 29.2T).

## 4.6 The open corpora you can actually build on

> **2026 default choice: DCLM-Baseline or Nemotron-CC for the bulk, FineWeb-Edu for the knowledge-heavy portion, Dolma 3 Mix if you want everything reproducible, plus FinePDFs and The Stack v3 for long documents and code.**

| Corpus | Tokens | Filtering method | Measured lift |
|---|---|---|---|
| **FineWeb** | **15T** (96 CC dumps, 2013–2024) | trafilatura → fastText lang ≥0.65 → C4 + Gopher heuristics → 3 custom filters → **per-dump** MinHash | Beats C4 / RefinedWeb / Dolma / RedPajama-2 / SlimPajama |
| **FineWeb-Edu** | **1.3T** (score ≥3); 5.4T (≥2) | Llama-3-70B-Instruct scores **460k** pages 0–5 → linear head on Snowflake-arctic-embed-m (**410k** used to fit, **F1 = 82%**) → keep ≥3 | 1.82B model / 350B tok: **MMLU 33→37, ARC 46→57**; matches peers with **~10× fewer tokens** |
| **DCLM-Baseline** | **3.8T** (from a 240T pool) | resiliparse → heuristics → Bloom dedup → **fastText bigram**, 200k positives (OpenHermes-2.5 + upvoted r/ELI5) vs 200k negatives, keep **top 10%** | 7B @ 2.6T: **MMLU 64%**, i.e. Llama-3-8B-level at **6.6× less compute** |
| **Nemotron-CC** | **6.3T** (4.4T real + **1.9T synthetic**) | justext → **3-classifier max-ensemble** → 5 quality buckets → global fuzzy + exact-substring dedup → 5-style rephrasing | 8B @ 1T: **MMLU 59.0 vs DCLM 53.4 (+5.6)**; 8B @ 15T: **MMLU 70.3 vs Llama 3.1 8B 65.3** |
| **Dolma 3 Mix** | **5.93T** (from a 9.3T pool) | topic-partitioned quality percentiles in **5% vigintile buckets** — conditional filtering, not a global cutoff | The fully-open reference |

FineWeb's three custom filters, with exact thresholds and token cost ([arXiv:2406.17557v2](https://arxiv.org/html/2406.17557v2)): fraction of lines ending in punctuation ≤ 0.12 (**−10.14%**); fraction of characters in duplicated lines ≥ 0.1 (**−12.47%**); fraction of lines under 30 characters ≥ 0.67 (**−3.73%**). Combined: **~22% of tokens removed for ~+1% aggregate score.** Note the MinHash setting is **Jaccard 0.75**, not 0.80.

**Dolma 3 Mix composition** ([dataset card](https://huggingface.co/datasets/allenai/dolma3_mix-6T-1025-7B)): Common Crawl **76.07%** (4.51T), **olmOCR science PDFs 13.57%** (805B), Stack-Edu 6.89% (409B), FineMath-3+ 2.56% (152B), arXiv 0.86% (50.8B), Wikipedia 0.04%.

Also worth knowing: **Common Pile / Comma** (8 TB, licensed-only — Comma v0.1 at 7B/1T–2T **matches Llama 1 and Llama 2 7B** at similar compute, [arXiv:2506.05209](https://arxiv.org/abs/2506.05209)); **Essential-Web v1.0** (24T tokens with a 12-category taxonomy, annotated by a **0.5B model distilled from Qwen2.5-32B** at only **κ 0.74 → 0.72** agreement loss and **~50× the throughput**, [arXiv:2506.14111](https://arxiv.org/html/2506.14111v1)); **TxT360** (~5T unique, globally deduped, upsampled to 15T+); **Zyda-2** (5T).

### What actually shipped in 2026 — including the negative results

**There is no FineWeb 3, no Dolma 4, no Essential-Web v2, and no Nemotron-CC v3.** The FineData team went *sideways* — into modality and language — rather than up into a bigger web crawl:

| 2026 release | Size | Note |
|---|---|---|
| **FinePDFs** | **3T tokens**, 1.29B PDFs, 1000+ languages | XGBoost routes each PDF to CPU (Docling) or GPU OCR (RolmOCR). **FineWeb-Edu + DCLM + 0.25× FinePDFs beats Nemotron-CC v2** ([blog](https://huggingface.co/spaces/HuggingFaceFW/FinePDFsBlog)) |
| **The Stack v3** (Jul 2026) | **4.9T tokens / 713 languages / 173M repos** (train split); 113.7 TB full | Largest open code corpus ever released |
| **Nemotron-CC-v2** (2026) | 5,597B | **~49% synthetic**, rephrased by Qwen3-30B-A3B |
| **Ultra-FineWeb-L1** (Aug 2026) | **1T+ tokens** | Covers CC through CC-MAIN-2025-51 — the most recent open web corpus |
| **FinePhrase** (2026) | 486B synthetic | COLM 2026, [arXiv:2604.13977](https://arxiv.org/abs/2604.13977) |
| **CuraWeb** (2026) | 2T English | Explicitly targets FineWeb-Edu/DCLM diversity collapse; **+1.8% at 3B** |
| **FLUX** (2026) | 192B/dump | **50B usable tokens/dump vs DCLM's 40B (+25% retention)**; matches DCLM with **34.4% less compute** ([arXiv:2603.13972](https://arxiv.org/abs/2603.13972)) |

---

# Stage 5: Post-training

## 5.0 The 2026 meta-pattern: specialists → distill → generalist

> **2026 default choice: do not RL one model on everything. Train N domain specialists (optionally × M reasoning-effort levels), then distill them all back into one model with on-policy distillation.**

**At least five labs converged on this pipeline independently**, and it is the single biggest post-training change since R1:

| Lab | Stage 1: specialists | Stage 2: consolidation |
|---|---|---|
| **GLM-4.5** (2025) | RL specialists in **Reasoning, Agent, General chat** | **Self-distillation** into one hybrid-reasoning model |
| **DeepSeek-V3.2** (2025) | **5 named specialists** — mathematics, competitive programming, general logical reasoning, agentic coding, agentic search (plus writing and general QA), each RL'd from the same base | Distill specialists into the final model → **one merged RL stage** |
| **DeepSeek-V4** (2026) | Per-domain SFT → GRPO under **three reasoning-effort configurations**, each with its own length penalty and context window | On-policy distillation over **>10 teachers** with **full-vocabulary reverse KL** — the mixed RL stage **removed entirely** |
| **Kimi K3** (2026) | RL across **3 domains × 3 effort levels = 9 experts** | **Multi-Teacher On-Policy Distillation (MOPD)** |
| **MAI-Thinking-1** (2026) | **Three parallel RL climbs** — STEM+competitive coding, agentic, helpfulness&safety — each punctuated by self-distillation save-points | Consolidation SFT (4 epochs) → final lightweight RL climb |
| **GLM-5** (2026) | Reasoning RL → Agentic RL → General RL (sequential rather than parallel) | **On-Policy Cross-Stage Distillation**, teachers drawn from the preceding SFT and RL stages |

Sources: [GLM-4.5 §3](https://arxiv.org/abs/2508.06471); [DeepSeek-V3.2](https://arxiv.org/abs/2512.02556); [DeepSeek-V4 §5.1](https://arxiv.org/abs/2606.19348); [Kimi K3 §4.1](https://github.com/MoonshotAI/Kimi-K3); [MAI-Thinking-1](https://microsoft.ai/pdf/mai-thinking-1.pdf); [GLM-5](https://arxiv.org/abs/2602.15763).

**Watch the trajectory across DeepSeek's own three generations** — it is the clearest evidence that this is a real convergence and not a fashion:

- **V3.2**: specialists → distill → **one merged RL stage** (reasoning + agent + alignment together).
- **V4**, roughly seven months later: specialists → distill → **the mixed RL stage is gone entirely.**

Distillation absorbed the job the final RL stage used to do. And the budget is not a rounding error: **DeepSeek-V3.2 states its post-training cost exceeds 10% of pre-training** ([arXiv:2512.02556](https://arxiv.org/abs/2512.02556)).

**Attribution note:** MOPD was **introduced by MiMo-V2-Flash** ([arXiv:2601.02780](https://arxiv.org/abs/2601.02780)) and formalized in [arXiv:2606.30406](https://arxiv.org/abs/2606.30406); Kimi K3, DeepSeek-V4, GLM-5.2 and Nemotron-Cascade 2 are adopters, not originators. Kimi K3's per-token signal is a **clipped log-ratio** — reverse-KL-shaped, but not literally reverse KL.

**Kimi K3's MOPD in detail.** Nine experts: domains {general, general agents, coding agents} × efforts {low, high, max}. (K3's effort schema actually *reserves* four levels — low, medium, high, max — but only three have trained experts.) For a sampled domain *d* and effort *e*, the per-token distillation reward between teacher and student is

```
r_opd(y_t | e, x, y_<t) = clip( sg[ log( π_teacher^(d,e)(y_t|·) / π_θ(y_t|·) ) ], −R_max, R_max )
```

where `sg` is stop-gradient and `R_max` clips extreme advantages. Because it is a **dense per-token reward that drops into the existing RL framework**, it inherits infrastructure like partial-rollout training for long-horizon tasks. Notably: "While we also experimented with more fine-grained **top-k distillation objectives, we observed no clear advantage** in either convergence speed or final performance in our setting" ([§4.1.3](https://github.com/MoonshotAI/Kimi-K3)).

**Why this pattern won:** RL on a mixed objective across math, code, agentic tool use, and chat produces interference and reward-hacking cross-talk; specialists avoid it. Distillation then recovers a single deployable artifact, and (per DeepSeek and Kimi) preserves most of each specialist's gains.

## 5.1 The R1 pipeline (still the reference for reasoning)

DeepSeek-R1's four-stage structure ([arXiv:2501.12948](https://arxiv.org/abs/2501.12948)) remains the canonical reasoning recipe:

1. **R1-Zero**: pure RL (GRPO) directly on the base model, no SFT — demonstrated that reasoning emerges from RL alone, but with poor readability and language mixing.
2. **Cold-start SFT**: a few thousand curated long-CoT examples to fix readability.
3. **Reasoning-oriented RL** with an added **language-consistency reward**.
4. **Rejection sampling → 800k SFT samples (600k reasoning + 200k non-reasoning)** → final RL across all scenarios.

The **R1-Distill** models (6 Qwen/Llama sizes) were SFT-only on those 800k samples, with the paper noting that applying RL on top would improve them further.

## 5.2 SFT

Verified sizes and structures:

- **DeepSeek-R1**: 800k samples (600k reasoning + 200k non-reasoning) ([arXiv:2501.12948](https://arxiv.org/abs/2501.12948)).
- **SmolLM3**: **1.8B tokens total** — 1B non-reasoning (12 datasets) + 0.8B reasoning (10 datasets with traces); **4 epochs (~8B tokens)**, BFD packing. Reasoning traces were synthesized by "prompting Qwen3-32B in reasoning mode with prompts from existing non-reasoning datasets" ([HF blog](https://huggingface.co/blog/smollm3)).
- **GLM-4.5**: cold-start SFT with a small set of long-CoT data per expert; then "Overall SFT" collecting **millions of samples** from the trained expert models, at **128K max context** ([§3.1](https://arxiv.org/abs/2508.06471)).
- **Olmo 3**: **Dolci Think SFT / Dolci Instruct SFT** datasets, with a dedicated function-calling subset for Instruct ([§4.2, §5.2](https://arxiv.org/abs/2512.13961)).
- **Kimi K3**: trajectories synthesized "using **domain-specialized models from the prior Kimi series**, followed by multi-stage verification and human-in-the-loop annotation," serialized with an **XTML-based chat template** (eXtensible Token Markup Language) ([§4.1.1](https://github.com/MoonshotAI/Kimi-K3)).

### Chat-template engineering is a real capability lever

GLM-4.5 reports a concrete win most recipes omit. JSON function-call arguments containing code require heavy character escaping, "compelling the model to generate extensive escape characters, thereby increasing the learning burden." Their fix: a function-call template that **encapsulates keys and values within XML-like special token tags**, so "the vast majority of code can be represented in its native form without escaping." Result: no loss in function-call performance, much less escaping ([§3.1](https://arxiv.org/abs/2508.06471)). Kimi K3's XTML template is the same instinct.

## 5.3 RLVR — the algorithm stack

> **2026 default choice: GRPO as the base, minus the KL term, minus std-normalization, plus clip-higher, plus token-level loss, plus dynamic/active sampling, plus truncated importance sampling. Nobody runs vanilla GRPO.**

### DAPO's ablation — the best published decomposition of why the stack works

DAPO ([arXiv:2503.14476](https://arxiv.org/abs/2503.14476)) trained Qwen2.5-32B **base** to AIME 2024 avg@32 = **50**, beating DeepSeek-R1-Zero-Qwen-32B's **47** with **50% of the training steps**. Crucially, they published the per-technique contribution:

| Configuration | AIME24 avg@32 |
|---|---|
| DeepSeek-R1-Zero-Qwen-32B (reference) | 47 |
| **Naive GRPO** | **30** |
| + Overlong Filtering | 36 |
| + Clip-Higher | 38 |
| + Soft Overlong Punishment | 41 |
| + Token-level Loss | 42 |
| **+ Dynamic Sampling (= DAPO)** | **50** |

**Vanilla GRPO gets 30; the accumulated fixes get 50.** That 20-point gap is the reason nobody runs vanilla GRPO in 2026, and it is why the modifications below should be treated as part of the algorithm rather than as options.

Verified DAPO hyperparameters ([§4.1](https://arxiv.org/abs/2503.14476)): AdamW, **constant LR 1e-6**, linear warm-up over **20 rollout steps**; rollout prompt batch **512**, **16 responses per prompt**; mini-batch 512 (**16 gradient updates per rollout step**); **ε_low = 0.2, ε_high = 0.28**; overlong reward shaping with expected max length **16,384** plus a **4,096-token soft-punish cache** (so max generation = 20,480); evaluated at temperature 1.0, top-p 0.7, repeated 32×. Built on **verl**.

Their note on why ε_high is raised but ε_low is not: "We increase the value of ε_high to leave more room for the increase of low-probability tokens… We keep ε_low as it is, because increasing it will suppress the probability of these tokens to 0." The measured symptom clip-higher fixes is **entropy collapse**.

And on token-level loss: "although it brings less performance improvement, we find it **enhances training stability and makes the length increase more healthily**" — i.e. adopt it for the variance reduction, not the headline number.

### The open reference implementation

**Olmo 3's OlmoRL is the best-documented open instance of the 2026 consensus stack.** Their seven named modifications over vanilla GRPO ([§4.4.1](https://arxiv.org/abs/2512.13961)):

1. **Zero gradient signal filtering** — drop groups where all rewards are identical (zero advantage std), as in DAPO.
2. **Active sampling** — maintain a consistent batch size despite that filtering, via "a novel, more efficient version of dynamic sampling."
3. **Token-level loss** — normalize by total tokens across the batch, not per-sample, to avoid length bias.
4. **No KL loss** — "as a common practice… it allows less-restricted policy updates, and removing it does not lead to over-optimization or destabilized training." They cite GLM-4.5, DAPO, and Dr. GRPO as precedent.
5. **Clip higher** — upper clipping bound above the lower one, enabling larger updates on low-probability tokens.
6. **Truncated importance sampling** — "multiply the loss by the truncated importance sampling ratio" to correct the **mismatch between inference-engine (vLLM) and training-engine log probabilities**.
7. **No standard-deviation normalization** in the advantage — "removes a difficulty bias, where questions with low standard deviation in their rewards (e.g., too hard or too easy) have their advantages significantly increased."

The resulting objective (note the explicit `π_vllm` term — the train/inference mismatch is now *in the published objective*, not a footnote):

```
J(θ) = 1/Σ|y_i| · Σ_i Σ_t  min( π(y_i,t|·;θ_old) / π_vllm(y_i,t|·;θ_old), ρ )
                          · min( r_i,t A_i,t , clip(r_i,t, 1−ε_low, 1+ε_high) A_i,t )

A_i,t = r(x, y_i) − mean({r(x, y_i)}_{i=1..G})
```

Engineering result: **4× speedup in RL training** vs their OLMo 2 setup ([§2.2](https://arxiv.org/abs/2512.13961)).

**GLM-4.5's RL** likewise "builds upon the GRPO framework, **excluding the KL loss term**" ([§3](https://arxiv.org/abs/2508.06471)).

**DeepSeek-V4** uses GRPO for its per-domain specialist RL ([§5.1.1](https://arxiv.org/abs/2606.19348)).

### Two contested RL design choices

**(a) Difficulty curriculum — yes.** GLM-4.5 uses a **two-stage difficulty-based curriculum**: moderate-difficulty data with `samples_per_prompt=16`, then extremely difficult data with `samples_per_prompt=512`. The rationale: as the model improves, easy prompts produce all-1 reward rollouts (zero gradient). They show continued AIME'24 improvement where a moderate-difficulty baseline plateaus ([§3, Fig. 5](https://arxiv.org/abs/2508.06471)).

**(b) Context-length ramp during RL — GLM-4.5 says no.** This directly contradicts the widely-copied DeepScaleR-style 8k→16k→24k schedule:

> "Previous research has suggested conducting RL in [staged context lengths]… we reveal that this **multi-stage approach is less effective than a single-stage RL process conducted directly** [at 64K output length]."
> — [GLM-4.5 §3, Fig. 6](https://arxiv.org/abs/2508.06471)

They also use a **dynamic sampling temperature** during RL to control trajectory diversity, since a fixed low temperature converges to insufficient exploration.

### The train/inference log-probability mismatch

This deserves a callout because it silently breaks RL runs. Rollouts are generated by an inference engine (vLLM/SGLang) whose numerics differ from the training engine (FSDP/Megatron), so `π_vllm ≠ π_θ_old` even at the same weights. Olmo 3 corrects it with **truncated importance sampling** (a cap ρ on the ratio). **Kimi K3 sidesteps it entirely** by making rollout and training share the same MXFP4/MXFP8 quantization scheme — "eliminating the train–inference mismatch" ([§4.1.4](https://github.com/MoonshotAI/Kimi-K3)).

## 5.3b Magistral: the most completely disclosed RLVR recipe

Mistral's Magistral ([arXiv:2506.10910](https://arxiv.org/html/2506.10910v1)) is worth singling out because it publishes every number, and because **Magistral Medium is pure RL from an instruct model — no distillation, no cold-start SFT** — which makes it the cleanest public existence proof that RL-from-instruct works at scale.

**Data funnel:** 699k math problems → 501k → **38k** after difficulty and verifiability filtering.

**Algorithm — GRPO with five modifications** (note how closely this tracks the consensus stack above):
1. **KL penalty eliminated entirely** — maintaining a reference-model copy "incurs a compute cost we find unjustified."
2. **Loss normalized** by `Σ|o_i|` over the group (token-level, not per-sample).
3. **Advantage normalization at minibatch level**: `Â_norm = (Â_i − Â_mean)/Â_std`.
4. **Clip-Higher**, ε in **0.26–0.28**.
5. **Group filtering** — drop all groups with zero advantage.

**Reward shaping, with exact values summing to 1.0:**

| Component | Value |
|---|---|
| Format (requires `<think></think>` tags **and** a boxed answer) | 0 or **+0.1** |
| Correctness — math (verified answer) or code (**passes 20 random tests, 4s timeout, 300MB limit**) | **+0.9** |
| Length penalty | **−0.1** piecewise |
| Language consistency (problem, thoughts and answer all in one language) | **+0.1** |

**Staged budgets:** batch **8k → 4k → 2k**; non-penalized length `l_max − l_cache` **16k → 24k → 32k**, with `l_max` reaching **40k** for AIME; temperature **0.7 (math) / 0.95 (code)**. LR and group size **UNVERIFIED**.

**Results — Magistral Medium: AIME 2024 pass@1 26.8% → 73.6%** (≈47 points absolute), maj@64 43.4% → 90.0%.

**And the cleanest published SFT-vs-RL decomposition anywhere.** Magistral Small (24B) was trained three ways from the same base:

| Recipe | AIME'24 pass@1 |
|---|---|
| SFT only | 65.4 |
| RL only | 65.8 |
| **SFT → RL** | **70.7** |

SFT and RL are worth almost exactly the same *alone*, and **~5 points more together** — which is the quantitative version of DeepSeek's "neither SFT nor RL alone is sufficient" (§8.2). The SFT data was traces with correct answers harvested from Magistral Medium's own RL run, plus OpenThoughts and the OpenR1 code subset, plus **10% general instruction data**, trained **4 epochs**.

**Async infrastructure worth copying** — three worker types: **generators** (roll out continuously with the latest policy, returning completions with log-probabilities), **trainers** (hold weights, take gradient steps), **verifiers** (score completions). "Generators operate continuously at maximum throughput **without ever waiting**." Weight broadcast is GPU-to-GPU, "reduces time to below **5 seconds**"; in-flight generations continue with a slightly stale KV cache so the "latest tokens [are] generated on-policy"; a greedy collation algorithm **cuts padding by 19%**. Stability constraint: **n_async / n_batch ≤ 2**, with n_async = 4096 concurrent generations.

## 5.4 Reward design

> **2026 default choice: verifiable rewards where you can (exact-match math, unit-tested code); for everything else, a *generative* reward model forced to write a rubric before scoring — plus an explicit length/verbosity budget to stop reward hacking.**

**Olmo 3's verifiers** ([§4.4.1](https://arxiv.org/abs/2512.13961)):
- **Math**: rule-based normalization + **SymPy** comparison to a reference answer; returns 1/0.
- **Code**: test-case verifier, executed on **AWS Lambda** for isolation; either fraction of tests passed, or binary all-pass.
- Extended beyond math (OLMo 2's scope) to code, precise instruction following, and general chat.

**Kimi K3's Agentic Generative Reward Model** for non-verifiable tasks — a mandatory 4-step protocol:

> "(1) read the outcome, product, or text output; (2) **generate a rubric**; (3) score each candidate against the rubric; and (4) **record the rubric-assigned scores in a scorepad**."

with tournament-style group rewards using binary comparisons. And the anti-hacking measure:

> "To mitigate reward hacking toward increasingly verbose outputs, we apply a **budget-based verbosity control**… given an initial verbosity ℓ₀ estimated from the cold-start model and a multiplier σ, a candidate whose output length exceeds σ·ℓ₀ **automatically loses the binary comparison**."
> — [Kimi K3 §4.1.2](https://github.com/MoonshotAI/Kimi-K3)

**MiniMax-M2** added a **task-completion-time reward** to optimize wall-clock efficiency alongside correctness ([Raschka notes](https://sebastianraschka.com/blog/2026/minimax-m2-technical-report.html)).

**DeepSeek-V4 went further and made the policy its own reward model.** They dispense with scalar reward models entirely; for hard-to-verify tasks they curate **rubric-guided RL data** and use a Generative Reward Model — but crucially:

> "we apply RL optimization **directly to the GRM itself**. In this paradigm, **the actor network natively functions as the GRM**, enabling the joint optimization of the model's evaluative (judging) proficiency alongside its standard generative capabilities. By unifying these roles, the model's internal reasoning capabilities are inherently fused into its evaluative process, resulting in highly robust scoring. Furthermore, this approach achieves superior performance with **only a minimal set of diverse human annotations**."
> — [DeepSeek-V4 §5.1.1](https://arxiv.org/abs/2606.19348)

This is the 2026 endpoint of the RLHF→RLAIF→rubric line: no separate reward model, no large human preference dataset, and the judge improves as the policy improves.

## 5.5 Reasoning effort / thinking budgets

> **2026 default choice: train discrete reasoning-effort levels as first-class model variants (low / high / max), not a single "thinking" toggle. Enforce them with a per-problem token budget during RL.**

The 2025 design was a binary switch. **Qwen3's hybrid thinking**: `/think` and `/no_think` flags in the user query or system message; non-thinking samples keep an *empty* thinking block for format consistency; for multi-turn dialogs, flags are randomly inserted and the model follows the **last** flag. Their **thinking budget** is an emergent capability, not explicitly trained:

> "when the length of the model's thinking reaches a user-defined threshold, we manually halt the thinking process and insert the stop-thinking instruction: *'Considering the limited time by the user, I have to give the solution based on the thinking directly now.\n</think>.\n\n'*… **this ability is not explicitly trained but emerges naturally as a result of applying Thinking Mode Fusion.**"
> — [Qwen3 §4.3](https://arxiv.org/abs/2505.09388)

**Kimi K3's Reasoning Effort RL** trains it explicitly instead ([§4.1.2](https://github.com/MoonshotAI/Kimi-K3)):

- Associate each problem `x` with an initial token budget `b₀(x)` estimated from the cold-start model.
- **Override the task reward with −1** for any trajectory whose total budget `T(y)` exceeds `τ · b₀(x)`.
- `T(y)` counts **thinking tokens** for general tasks, but **cumulative output tokens including tool-call arguments** for agentic tasks.
- Stage-wise curriculum over `τ`: first train a **max-budget** variant with large `τ` (still capped, "to suppress excessive overthinking"), then **anneal τ downward** to produce the **high-** and **low-effort** experts. `τ` is adjusted per domain with human-in-the-loop guidance.
- Trajectories from all effort levels are then pooled for SFT and MOPD.

**DeepSeek-V4 ships three reasoning modes as separately-RL'd specialists**, not as a runtime flag. Both V4-Pro and V4-Flash support Non-think / Think High / Think Max, and "For each mode, we apply **distinct length penalties and context windows during RL training**, which results in varying output token lengths" ([§5.1.1, Table 2](https://arxiv.org/abs/2606.19348)):

| Mode | Characteristics | Response format |
|---|---|---|
| **Non-think** | "Fast, intuitive responses based on habits or simple rules" | `</think> summary` (empty think block) |
| **Think High** | "Conscious logical analysis, slower but more accurate" | `<think> … </think> summary` |
| **Think Max** | "Push reasoning to its fullest extent" | special system prompt **+** `<think> … </think> summary` |

Note the empty-think-block convention for Non-think — the same trick Qwen3 uses for format consistency. And Think Max is implemented partly *as a prompt*, injected at the start of the system prompt ([Table 3](https://arxiv.org/abs/2606.19348)):

> "Reasoning Effort: Absolute maximum with no shortcuts permitted. You MUST be very thorough in your thinking and comprehensively decompose the problem to resolve the root cause, rigorously stress-testing your logic against all potential paths, edge cases, and adversarial scenarios. Explicitly write out your entire deliberation process, documenting every intermediate step, considered alternative, and rejected hypothesis to ensure absolutely no assumption is left unchecked."

gpt-oss exposes **low/medium/high reasoning effort** ([§2.5](https://arxiv.org/abs/2508.10925)). Kimi K3's benchmark table footnotes read "All maxed out on thinking effort: max or xhigh" — i.e., **reported scores are at maximum effort**, which is now the disclosure norm (§6).

## 5.6 Agentic RL

> **2026 default choice: build white-box, resumable environments; use partial rollouts so stragglers do not stall the batch; synthesize tasks rather than collecting them.**

**Partial rollouts** (originating in Kimi K1.5, [arXiv:2501.12599](https://arxiv.org/abs/2501.12599)) are now essential because long-horizon agentic trajectories have a brutal latency long tail. Kimi K3's version:

- Sample K completions for each of N prompts (N×K active trajectories).
- **Pause generation as soon as a fraction λ ∈ (0,1) of trajectories completes** (λNK), letting policy optimization proceed without waiting for stragglers.
- Paused rollouts are enqueued and **prioritized for resumption next iteration**, backed by sandbox infrastructure that persists rollout and sandbox state.
- Consequence: "an individual long-horizon trajectory naturally spans multiple iterations, introducing **data staleness**." Their policy optimization "inherently tolerates such an extreme off-policy regime through a **per-token regularization**… constraining policy updates within a localized neighborhood."
— [Kimi K3 §4.1.2](https://github.com/MoonshotAI/Kimi-K3)

### Randomize the harness, or the model overfits to it

This is the most transferable agentic-RL lesson published in 2026, and it is cheap to act on:

> "Training with a **single fixed agent harness** can cause a model to **overfit to a particular tool schema, system prompt, context management mechanism, or interaction protocol**. To address this, we develop a unified white-box RL environment that represents an agent harness as a collection of **configurable, composable modules** — including tool interfaces, system prompts, context management strategies, skills, memories, subagents, and other components… the environment can instantiate mainstream harnesses such as Kimi Code, Claude Code, Codex, OpenClaw, and Hermes, as well as entirely new ones. During RL training, we **dynamically construct different harness configurations for different task groups**, exposing Kimi K3 to diverse combinations of these modules rather than the conventions of any single harness."
> — [Kimi K3 §4.2.1](https://github.com/MoonshotAI/Kimi-K3)

If you train an agent against one scaffold, you have trained a scaffold-specific model — which is also why Epoch measures 11–15 point swings from scaffold changes alone (§6.3).

### Knowledge-graph-guided task synthesis

Rather than collecting agentic tasks, Kimi K3 generates them from a **self-evolving, hierarchically organized knowledge graph that agents continuously expand through web-scale exploration** across knowledge-intensive and coding domains. The pipeline: sample related keyword sets jointly from the graph (e.g. "RoPE", "GPU kernel") → retrieve public materials (academic articles, blog posts, code repos) → synthesize verifiable tasks. The rationale: "Retrieval guided by **fine-grained concepts** surfaces specialized and underrepresented knowledge, while sampling across **diverse concepts** broadens domain coverage" ([§4.2.2](https://github.com/MoonshotAI/Kimi-K3)).

Task families they enumerate: verifiable problems in agentic environments, **kernel optimization**, **personal assistant**, **autonomous execution**, and **web development** tasks ([§4.2](https://github.com/MoonshotAI/Kimi-K3)).

**GLM-5's agentic data scale** is the most concrete published target to aim at ([arXiv:2602.15763](https://arxiv.org/abs/2602.15763)): **>10,000 verifiable environments** across thousands of repositories in **9 languages**, built from **~10 million issue–PR pairs**, plus a web knowledge graph distilled from **two million high-information pages** — with **Docker construction accuracy >90%**, which is the number that actually gates whether an SWE environment pipeline is usable. Their RL settings: GRPO **+ IcePop**, β = 2, **ε_low 0.2 / ε_high 0.28**, **group size 32**, batch 32, **KL removed**.

**MiniMax-M2's SWE environment recipe** is the most reproducible published approach: create executable software environments from **merged GitHub pull requests**, "extract tests that should fail before the patch and pass afterward," spanning "more than ten programming languages" ([Raschka notes](https://sebastianraschka.com/blog/2026/minimax-m2-technical-report.html)).

### Sandbox design: the 98% number

Kimi K3's sandbox section contains the single most actionable piece of agentic-RL engineering published in 2026 ([§5.3.2](https://github.com/MoonshotAI/Kimi-K3)). They moved from containers to **Firecracker microVMs** (a system called AgentENV) for three reasons:

1. **Containers were not robust to capable agents.** "As agents become more capable and tasks more difficult, they tend to explore more aggressively and may even attempt reward hacking… in our early experiments with traditional container-based sandbox runtimes, **we observed several kernel panics and deadlocks caused by unintended agent operations**." At the same time they wanted *more* permissiveness, not less — "agents should be able to mount disks, run containers, or even launch virtual machines at will."
2. **Sandbox lifecycle operations built on incremental checkpointing** (only memory pages dirtied since the last checkpoint are saved), achieving **133 ms checkpoint / 49 ms resume**:
   - **Pause/Resume** — "a paused sandbox consumes no memory or CPU resources; a sandbox can therefore be paused while the agent is waiting for the model's inference result, which can account for as much as **98% of the sandbox lifetime**."
   - **Fork** — clone a sandbox from exact state while the original keeps running, "useful for **reward judging without side effects**."
   - **Snapshot** — periodic saves for error recovery.
3. **Density**: "tens of thousands of sandboxes, each with a unique set of images, may need to be created within seconds" — solved with OverlayBD images, a custom ublk driver, storage-layer sharing, P2P transport, and copy-on-write memory, achieving sub-second launch at scale.

**The 98% figure is the headline.** If your agentic RL sandboxes are allocated for their whole trajectory, you are paying for ~50× more sandbox capacity than you need. Pause-on-inference is the cheapest large win available in an agentic RL stack.

They also decouple prefix KV retention from GPU residency with a **write-back external KV cache pool in CPU DRAM** — necessary because at 1M-context multi-step rollout "a prefix KV-cache miss is extremely expensive," and partial rollouts make it worse by dumping many unfinished long prefills at the start of each iteration ([§5.3.1](https://github.com/MoonshotAI/Kimi-K3)).

**DeepSeek-V4's post-training infrastructure** names the pieces needed at scale: FP4 QAT for expert weights, **efficient teacher scheduling for full-vocabulary on-policy distillation**, a **preemptible and fault-tolerant rollout service**, an **RL framework supporting million-token contexts**, and **sandbox infrastructure with four execution substrates** ([§5.2](https://arxiv.org/abs/2606.19348)).

**Scaling law for RL.** Kimi K3 reports that scaling RL FLOPs consistently increases both capability and **average tool-call steps** across knowledge, reasoning, vision, general-agent, and coding evals — i.e., more RL compute buys longer, more persistent agentic behavior, not just higher scores ([§4.1.2, Fig. 8](https://github.com/MoonshotAI/Kimi-K3)).

### A fully-specified RL climb, for calibration

Microsoft's MAI-Thinking-1 publishes the most complete set of RL hyperparameters available anywhere ([tech report](https://microsoft.ai/pdf/mai-thinking-1.pdf)). Useful as a sanity check against your own config:

- **Optimizer**: AdamW with **β₁ = β₂ = 0.95**, **ε = 1e-15**, **no weight decay**. **Constant LR 1e-6**, no warmup and no decay — lowered to **9e-7 at longer generation lengths** to reduce off-policiness.
- **Batch**: global **7,040 after packing**, unpacked sequences capped at 12,000.
- **Generation length staged 8k → 16k → 32k → 64k → 128k.**
- **Sampling**: **G = 128 rollouts**, with an early-exit screen at G_early = 16 and pass-rate band [0.05, 0.8]; full pass-rate filter [0.1, 0.8]; top-p 0.97.
- **Adaptive entropy control** rather than a fixed clip: ε = 0.6, k_max = 2.5, step δ = 0.25, target entropy H* = 0.3, with `k ← clip(k + δ·sign(H* − Ĥ), 0, k_max)`; k starts at 0 so the bounds are multiplicatively symmetric in log-ratio space. Outer clip **r_max = 50**.
- **Reward**: `R = R_task + w_lang·R_lang − w_len·R_len` with **w_lang = 0.5, α = 0.005, w_len = 0.25 up to 64k then 0 at 128k** — note they *switch off* the length penalty at the longest budget.
- **Staleness policy**: 5 gradient steps between inference-model updates; **discard any rollout more than 8 inference updates (40 gradient steps) stale.**

## 5.7 RL infrastructure — what you should actually build on

> **2026 default choice: verl if you want the widest adoption and the most features, slime if you are on Megatron + SGLang and want a battle-tested frontier path, prime-rl if you need fully-async at 1T+ scale. All of them are asynchronous and disaggregated; none of them are synchronous any more.**

The definitive cross-framework survey is Hugging Face's **"Keep the Tokens Flowing: Lessons from 16 Open-Source RL Libraries"** (March 2026, [blog](https://huggingface.co/blog/async-rl-training-landscape)). Headline facts:

- **15 of 16 libraries are asynchronous** (OpenPipe ART is the sole synchronous exception).
- **"All surveyed libraries employ disaggregated mode"** — separate training and rollout engines.
- Training backends: **FSDP2 in 7, Megatron in 8, DeepSpeed in 6** (DeepSpeed in retreat).
- Inference backends: **vLLM in 14/16, SGLang in 8+**.
- **Staleness strategies** cluster into three families: *version rejection* (NeMo-RL, TorchForge, PipelineRL), *depth bounding* (AReaL, Atropos, SkyRL, verl), and *importance-sampling correction* (MILES, OAT, ROLL, slime, verl, open-instruct).
- **Partial-rollout strategies**: implicit continuation (PipelineRL), abort+recycle (SkyRL, slime, MILES), explicit save-resume (verl), or none (ART, ROLL, TorchForge, verifiers-rl).

### Which frameworks actually trained frontier models

| Framework | Frontier track record |
|---|---|
| **verl** ([verl-project/verl](https://github.com/verl-project/verl)) | Seed-Thinking-v1.5, Doubao, DAPO, VAPO; ~30 named adopters incl. Microsoft Research, Moonshot, Alibaba Qwen, Baidu, Xiaomi, Amazon, NVIDIA Research |
| **slime** | **Seven consecutive GLM frontier releases** (GLM-4.5 → 4.6 → 4.7 → 5 → 5.1 → 5.2 → 5.3) |
| **prime-rl** | **INTELLECT-3** — 106B MoE on GLM-4.5-Air, **512 H200 across 64 nodes, ~2 months** ([Prime Intellect](https://www.primeintellect.ai/blog/intellect-3)) |
| **NeMo-RL** | Nemotron-3-Nano-30B-A3B, Nemotron-3.5-lightning |
| In-house | MAI's Rocket, MiniMax's Forge — at the very top, labs still build their own |

**FSDP2 is now proven at frontier scale**, which removes Megatron's last exclusive moat: prime-rl v0.9.0 advertises "**1T+ MoE models on 1000+ GPUs** with FSDP2 + vLLM, FP8 inference, PD disaggregation," and NVIDIA's Molt reaches 700B/1T-class MoE on FSDP2 alone while landing **within ~9% of Megatron-based slime** in a head-to-head on 8+8 H100s (119.4 vs 109.5 s/step).

Note the code-size spread, which is a real adoption consideration: **Molt ~9.2K LOC vs verl ~62K, slime ~25K, OpenRLHF ~7.2K.**

### The train/inference mismatch fix is now standard — with exact merge dates

Every serious framework shipped Truncated Importance Sampling within about three weeks of each other in August 2025:

| Framework | Evidence |
|---|---|
| **SkyRL** | PR #145 "Add support for truncated importance sampling" — merged **2025-08-15** |
| **OpenRLHF** | **v0.8.9, 2025-08-06**; flags `--algo.advantage.is_correction_type {tis, icepop, seq-mask-tis}`, threshold `0.5 5.0` |
| **verl** | PR #2953 "Rollout-Training Mismatch Fix — Truncated importance sampling" — merged **2025-08-26**; Megatron TIS #3513 (2025-09-18); **IcePop** #5722 (2026-03-24); default `calculate_log_probs=True` since **2026-06-05** |
| **slime** | PR #179 "Naive implementation of rollout logp correction" — merged **2025-08-26** |
| **Molt** | The most complete: `is_correction_level {off,token,seq,geo}` × `mode {mask,clip,trunc}`; token+trunc = TIS, token+mask = IcePop; **refuses to run partial rollout without IS correction** |

**Caveat if you use TRL**: TIS is on by default in the synchronous `GRPOTrainer`, but two issues are open as of Sept 2026 — **#6945 "AsyncGRPO has no importance-sampling correction"** and **#6789 "vLLM importance-sampling ratio is biased when top_p/top_k/min_p truncate sampling."** Also note TRL **removed `PPOTrainer`/`PPOConfig` in v1.13.0 (2026-09-10)** as an "unmaintained code path"; PPO survives via `gae` in Molt/OpenRLHF/verl ([TRL v1 blog](https://huggingface.co/blog/trl-v1)).

### The frontier has moved past TIS on two fronts

1. **MoE routing mismatch.** Expert routing can differ between the rollout and training engines, which no importance-sampling correction addresses. The shipping answer is **router replay (R3)** — in verl, NeMo-RL v0.7.0, Molt (`--train.routing_replay`) and Dressage — or router freezing (Molt `--actor.freeze_moe_router`). If you are doing RL on an MoE, this is now a required consideration.
2. **Eliminating the mismatch entirely.** vLLM shipped a **batch-invariant mode** (`VLLM_BATCH_INVARIANT=1`), but it is **opt-in beta and actively unfinished** — open bugs as of Sept 2026 include "#55131 Batch-invariant matmul is not actually batch-invariant, and TF32 causes precision degradation" ([vLLM docs](https://docs.vllm.ai/en/latest/features/batch_invariance.html)). verl v0.9.0 nonetheless ships "full determinism for vLLM rollout and reward inference." See also [VeXact](https://github.com/verl-project/vexact).

Recall that **Kimi K3 sidesteps the whole problem** by making rollout and training share one MXFP4/MXFP8 quantization scheme (§2.6, §5.3).

### Why partial rollouts are not optional

The APRIL project (slime ecosystem) quantifies the motivation directly: long-tail generation **"consumes over 90% of RL training time."** That is the same phenomenon Kimi K3 addresses with λ-fraction partial rollouts (§5.6) and that OpenRLHF addresses with `--train.async_queue_size 1` (literal one-step-off-policy).

### Environment ecosystems

| Registry | Environments (measured Sept 2026) |
|---|---|
| Prime Intellect Environments Hub | **≥360** (floor; mid-migration to verifiers v1) |
| NeMo Gym ([NVIDIA-NeMo/Gym](https://github.com/NVIDIA-NeMo/Gym)) | **122** |
| prime-environments monorepo | **109** |
| OpenEnv ([huggingface/OpenEnv](https://github.com/huggingface/OpenEnv)) | **40** |

Repo migrations to be aware of: `volcengine/verl` → `verl-project/verl`; `inclusionAI/AReaL` → `areal-project/AReaL`; **`meta-pytorch/OpenEnv` → `huggingface/OpenEnv`**; `willccbb/verifiers` → `PrimeIntellect-ai/verifiers`.

## 5.8 Preference optimization

> **2026 default choice: still worth doing between SFT and RLVR, but it is no longer where the gains are. Olmo 3 keeps DPO; the big labs increasingly go SFT → RL directly.**

- **Olmo 3** retains a full **SFT → DPO → RLVR** pipeline for both Think and Instruct, and reports "observing gains at each stage." Their preference data uses **Delta Learning** (contrastive pairs from a strong/weak model gap), and they claim it "**expands the reasoning frontier of the model beyond what SFT alone can provide and primes the model for effective reinforcement learning**" ([§2.2, §4.3](https://arxiv.org/abs/2512.13961)).
- **SmolLM3** used **APO (Anchored Preference Optimization)** rather than DPO — "off-policy model alignment… more stable optimization objective" — with reasoning pairs synthesized as Qwen3-32B (chosen) vs Qwen3-0.6B (rejected) ([HF blog](https://huggingface.co/blog/smollm3)).
- **Olmo 3 Instruct** uses DPO for **behavioral** targets rather than capability: multi-turn preference data and "**targeted data length interventions that encourage concise responses**," then finds "preference tuning synergizes with RL to improve model performance **while maintaining learned brevity**" ([§5.3–5.4](https://arxiv.org/abs/2512.13961)).
- **DeepSeek-V4, Kimi K3, GLM-4.5** describe SFT → RL → distillation with **no DPO stage**.

## 5.9 Model merging / souping is a production step now

- **Olmo 3 32B**: midtraining run **twice** with different data-order seeds, then **weights averaged** ([Table 35](https://arxiv.org/abs/2512.13961)).
- **SmolLM3**: final model is a **linear merge of 0.9 × (APO soup) + 0.1 × (mid-training checkpoint)**, specifically to recover long-context RULER performance out to 128k ([HF blog](https://huggingface.co/blog/smollm3)).

## 5.10 Distillation for small models

**Qwen3's strong-to-weak distillation** is the best-quantified case for not running the full pipeline on small models:

- Two phases: **off-policy** (teacher outputs) then **on-policy logit distillation**.
- "directly distilling the output **logits** from teacher models into lightweight student models can effectively enhance their performance while maintaining fine-grained control over their reasoning processes."
- Results: higher **Pass@1**, *and* improved **Pass@64** — i.e., it improves exploration, not just greedy accuracy (this matters, because it counters the "distillation collapses diversity" objection).
- Cost: "**only 1/10 of the GPU hours** compared to the four-stage training method."
— [Qwen3 §4.5](https://arxiv.org/abs/2505.09388)

**Gemma 3's logit distillation** for pretraining: "We sample **256 logits per token, weighted by teacher probabilities**. The student learns the teacher's distribution within these samples via cross-entropy loss. The teacher's target distribution is set to zero probability for non-sampled logits, and renormalized" ([§2.2](https://arxiv.org/abs/2503.19786)). Note this is distillation during *pretraining*, on 2–14T tokens.

**Llama 4 Maverick** was "codistilled from Llama 4 Behemoth" using "a novel distillation loss function that **dynamically weights the soft and hard targets**" ([Meta blog](https://ai.meta.com/blog/llama-4-multimodal-intelligence/)).

### On-policy distillation: the numbers behind the 2026 shift

Thinking Machines' write-up is the clearest quantification of why the specialist→distill pattern (§5.0) works. The method samples trajectories **from the student**, then has the teacher grade every token — per-token reverse KL, `KL(π_θ ‖ π_teacher)` — so you get **dense supervision on the student's own distribution** rather than sparse episode-level reward or off-distribution teacher text.

Measured on Qwen3-8B from a 400k-example off-policy SFT checkpoint ([Thinking Machines](https://thinkingmachines.ai/blog/on-policy-distillation/)):

| Method | AIME'24 | Cost |
|---|---|---|
| Off-policy distillation (400k) | 60% | baseline |
| RL | 67.6% | **17,920 GPU-hours** |
| **On-policy distillation** | **70%** | ~150 steps (~77k prompts) |
| On-policy distillation (extended) | **74.4%** | ~one-tenth the RL cost |

Their headline framing: on-policy distillation "learns the RL-trained policy in approximately **7–10× fewer gradient steps, which corresponds to a compute efficiency of 50–100×**." Versus off-policy distillation the saving is **9× amortized, 18× in practical GPU-hours, and 30×** when the teacher's cost for a new task is included.

Their stated preconditions are exactly the conditions that hold in the specialist→generalist pattern: use it when "**(1) dense supervision signals matter more than sparse episode-level rewards, and (2) you have access to a stronger teacher model.**" In the 2026 pipelines, the nine domain × effort experts *are* that stronger teacher.

**Caveat from the scaling-law side**: Apple's Distillation Scaling Laws ([arXiv:2502.08606](https://arxiv.org/abs/2502.08606)) find distillation beats supervised learning only up to a compute level that scales predictably with student size, and **only when many students are trained or a teacher already exists** — if you must train one teacher and one student from scratch, supervised learning wins.

---

# Stage 6: Evaluation-driven development

## 6.1 The core problem: small models are at chance on the benchmarks you care about

> **2026 default choice: make decisions on continuous likelihood metrics (bits-per-byte, correct-choice probability) at small scale, aggregate tasks into capability clusters, and reserve accuracy metrics for the end of the run.**

### CORE score (DCLM) — the standard small-scale proxy

DCLM's CORE is a **22-task subset** of its 53-task suite, "selected due to their ability to provide a low variance signal of learning, even at small scales" ([arXiv:2406.11794](https://arxiv.org/abs/2406.11794)). Each task is linearly rescaled so 0 = random guessing, 1 = perfect, then averaged:

```python
centered_result = (accuracy - 0.01 * random_baseline) / (1.0 - 0.01 * random_baseline)
core_metric     = sum(centered_results.values()) / len(centered_results)
```
(`random_baseline` is stored in percent; reference implementation in [nanochat/scripts/base_eval.py](https://github.com/karpathy/nanochat/blob/master/scripts/base_eval.py))

**The 22 CORE tasks**: AGI Eval LSAT-AR (3-shot); ARC-Easy, ARC-Challenge (10-shot); six BIG-Bench tasks at 10-shot (QA Wikidata, Dyck Languages, Operators, Repeat Copy Logic, CS Algorithms, Language Identification); BoolQ, CommonsenseQA (10-shot); COPA, CoQA (0-shot); HellaSwag 0-shot *and* 10-shot (counted separately); Jeopardy (10-shot); LAMBADA (0-shot); OpenBookQA (0-shot); PIQA, SQuAD (10-shot); Winograd Schema, WinoGrande (0-shot) ([Appendix G](https://arxiv.org/html/2406.11794v3)).

**The justification for using small proxies at all**: DCLM measured that data-curation rankings transfer across scale, with Pearson **r = 0.885** (400M→7B) and **r = 0.919** (1B→7B).

DCLM also found the **harness itself changes conclusions**: LightEval (full-sequence logprobs) gives above-random MMLU at 1B where LLM-Foundry (single-letter logprobs) is pinned at ~0.25 — but LightEval compresses the top end (Gemma-7B / Llama3-8B / Mistral-7B all land at 0.43–0.44 vs 0.56–0.62 in LLM-Foundry). **Fix the harness before you trust a delta.**

### OLMES / OlmoBaseEval

**OLMES** ([arXiv:2406.08446](https://arxiv.org/abs/2406.08446)) standardizes prompt formatting, in-context example selection, probability normalization, and task formulation across 10 MCQA benchmarks, validated on 15 models from 1B–70B. Its key contribution is making cloze-format scores for small base models comparable to multiple-choice-format scores for large ones.

**OLMo 2's dev/unseen split is the cleanest published anti-overfitting discipline**: *development* benchmarks (ARC-C, HellaSwag, WinoGrande, MMLU, DROP, NaturalQuestions) tracked during training, and *unseen* benchmarks (AGIEval, MMLU-Pro, GSM8K, TriviaQA) **not computed until development finished** ([Ai2 blog](https://allenai.org/blog/olmo2)).

**Olmo 3's `OlmoBaseEval`** scaled this to ~43 benchmarks (~4× OLMo 2's suite) on three principles ([§3.3](https://arxiv.org/abs/2512.13961)): (1) aggregate tasks into **capability clusters**; (2) develop **proxy metrics** validated by scaling analysis for which task gives signal at which scale; (3) **signal-to-noise filtering** — evaluate noisy tasks on more examples, or remove them. It is split into "Base Easy" (small proxy runs) and "Base Main" (full-scale). Olmo 3 notably **did not evaluate on its held-out benchmarks prior to release** (stated in the caption of Tables 2 and 3).

### The two Ai2 papers that make this rigorous

**DataDecide** ([arXiv:2504.11393](https://arxiv.org/abs/2504.11393)) — 25 pretraining corpora, models to 1B params / 100B tokens, 3 seeds:
- Ranking models at a **single small size (150M)** predicts the best data recipe at 1B with **~80% of pairwise comparisons correct** — and **none of 8 scaling-law baselines beats that compute-decision frontier**.
- Using **continuous likelihood metrics** instead of accuracy makes MMLU, ARC, HellaSwag, MBPP and HumanEval **>80% predictable at 1B with 0.01% of the compute**.

**Signal and Noise** ([arXiv:2508.13144](https://arxiv.org/abs/2508.13144)) — the formalization. *Signal* = ability to separate better from worse models; *noise* = sensitivity to random variation between training steps; SNR = the ratio. Built from **375 open-weight models (60M–32B), 30 benchmarks, 900K benchmark results**, at a cost of 94K H100-hours of eval compute. Findings: higher-SNR benchmarks are more reliable for small-data decisions; lower-noise benchmarks give lower scaling-law prediction error; switching from accuracy to **perplexity/bits-per-byte** helps; dropping noisy subtasks raises aggregate SNR; and **averaging over intermediate checkpoints consistently reduces noise**.

### Llama 3's "annealing to assess data quality"

The cheapest data-evaluation trick published, and it is worth copying ([Llama 3 §3.1.3](https://arxiv.org/html/2407.21783v3)):

1. Take a **50%-trained 8B checkpoint**.
2. Anneal LR **linearly to 0 over 40B tokens**.
3. During the anneal, weight the **candidate dataset at 30%** and the default mix at 70%.
4. Measure benchmark delta.

"Using annealing to evaluate new data sources is more efficient than performing scaling law experiments for every small dataset." Critically, they also report its limit: annealing on GSM8K/MATH training sets improved the **8B** by 24.0% and 6.4%, but on the **405B the improvement was negligible**. It is a small-model data probe, not a frontier lever.

(For reference, Llama 3's final pretraining mix: ~50% general knowledge, **25% math & reasoning, 17% code, 8% multilingual**.)

### Perplexity across domains

**Paloma** ([arXiv:2312.10523](https://arxiv.org/abs/2312.10523)) evaluates perplexity across **546 English and code domains** (including the top 100 subreddits and top GitHub languages) and shows Common-Crawl-only models have anomalous gaps in fit to many domains. A single held-out perplexity number does not generalize. **bits-per-byte** is the vocab-invariant form used as the primary training signal in practice — nanochat tracks `val_bpb` alongside CORE.

### On when downstream metrics become predictable

- **Observational Scaling Laws** ([arXiv:2405.10938](https://arxiv.org/abs/2405.10938)): builds scaling laws from ~100 existing public models rather than new training runs; finds performance lies in a low-dimensional capability space; shows several "emergent" phenomena are **smooth sigmoids predictable from small models**; predicts the effect of post-training interventions (CoT, self-consistency) and even agentic performance from simpler non-agentic benchmarks.
- **"Why Has Predicting Downstream Capabilities Remained Elusive?"** ([arXiv:2406.04391](https://arxiv.org/abs/2406.04391)): downstream accuracy is computed from NLLs "via a sequence of transformations that progressively degrades the statistical relationship between performance and scale." Tracking probability mass on the *correct* choice is insufficient — you must model how mass fluctuates on the **incorrect** choices.

## 6.2 Decontamination

> **2026 default choice: 8-gram overlap at a ~0.5 threshold with IDF weighting and length-dependent thresholds; run it hardest at the *midtraining* stage.**

Published thresholds:

| Project | Rule |
|---|---|
| GPT-3 | 13-gram overlap |
| GPT-4 | 50-character substring overlap |
| Tülu 3 / Olmo 3 post-training | **8-gram matching, 0.5 overlap threshold**, across all post-training data |
| Light-R1 | exact match excluding digits + **32-gram** |

**Ai2's `decon` tool** (shipped with Olmo 3) is the reference open implementation: two-phase **n-gram overlap with IDF weighting**, tolerant of gaps (`passage_max_consecutive_misses`), with **length-based thresholds** so short texts require higher overlap. The operational insight from the Olmo 3 report is the most useful part: **"memorization happens most strongly near the end of training, so decontaminating at the midtraining stage (and for long-context extensions) is most effective"** ([decon docs](https://github.com/allenai/decon/blob/main/doc/simple-details.md); [Ai2 blog](https://allenai.org/blog/olmo3)).

**DCLM deliberately does not decontaminate its pool** — instead it releases decontamination tooling and requires every submission to "disclose a decontamination report and avoid using highly-contaminated data," auditing the top scorers ([§ benchmark rules](https://arxiv.org/html/2406.11794v3)).

### Llama 3 published the only per-benchmark contamination measurement

They score examples by **8-gram overlap** and report both the flagged fraction and the estimated performance gain from contamination ([arXiv:2407.21783v3](https://arxiv.org/html/2407.21783v3), §3.1.3). Selected rows:

| Benchmark | % of eval set flagged | Est. gain 8B | 70B | 405B |
|---|---|---|---|---|
| **BIG-Bench Hard** | 95 | **26.0** | **36.0** | **41.0** |
| AGIEval | 98 | 8.5 | 19.9 | 16.3 |
| HellaSwag | 85 | 14.8 | 14.8 | 14.3 |
| PiQA | 55 | 8.5 | 7.9 | 8.1 |
| QuaC | 99 | 2.4 | 11.0 | 6.4 |
| NaturalQuestions | 52 | 1.6 | 0.9 | 0.8 |
| GSM8K | 41 | 0.0 | 0.1 | 1.3 |
| MATH | 1 | 0.0 | −0.1 | −0.2 |
| Winogrande | 6 | −0.1 | −0.1 | −0.2 |

Two things to take from this. First, **contamination effects are wildly benchmark-dependent** — BBH is worth up to 41 points of inflation at 405B, while MATH and Winogrande are worth nothing. Second, Meta's own caveat: for **MBPP, HumanEval, MMLU and MMLU-Pro**, "8-gram overlap gives such high contamination scores that it is impossible to get a good performance gain estimate." The standard method cannot even measure the most-cited benchmarks.

**Olmo 3's rule** is 8-gram matching at **overlap threshold 0.5**, applied against **all splits of all benchmarks in the OLMES suite** ([arXiv:2512.13961v2](https://arxiv.org/html/2512.13961v2)).

### The standard is known to be inadequate

**"Soft Contamination Means Benchmarks Test Shallow Generalization"** ([arXiv:2602.12413](https://arxiv.org/html/2602.12413v1), Feb 2026) uses Olmo 3 as its case study: finetuning on exact *or semantically equivalent* duplicates raises benchmark performance **~20% on both seen and unseen items**. At an "ecologically valid" rate of **~4 in 10,000** training points being a semantic duplicate, scores rise **+12% on seen and +5.6% on unseen** items. "Typical 'decontamination' filters use n-gram matching which fail to detect 'semantic' duplicates." Their proposed practical threshold is **embedding cosine similarity 0.5–0.6**.

Corroborated by a systematic review of **55 studies** finding no detection method reliable ([ACL GEM 2026](https://aclanthology.org/2026.gem-main.50/)).

**Bottom line: the 2026 standard is 8-gram at ~0.5 coverage against all eval splits; the literature consensus is that this standard is insufficient, and embedding-based semantic dedup is the emerging successor.**

**The countervailing evidence on whether it matters**: Ai2's own OLMo-1B ablation over a 221B-token RedPajama subset found decontamination had **no clear positive or negative effect** on downstream performance, and DCLM found that removing MMLU/HellaSwag overlaps **did not decrease performance**. But **Light-R1's audit found MATH-500 "somewhat compromised with tens of questions that are identical or only numbers changed"** in public post-training data ([Light-R1](https://huggingface.co/qihoo360/Light-R1-32B-DS)). At minimum, decontaminate so that you can *report* honestly.

## 6.3 The tricks that move headline scores — and whether they are disclosed

> **2026 default choice: report a matrix — {no tools, with tools} × {reasoning effort levels} — and footnote parallel-sampling modes separately. This is now the norm at the top of the field, and falling short of it reads as concealment.**

**Kimi K2 Thinking is the gold standard of disclosure**: three rows per benchmark — `no tools`, `w/ tools`, `heavy`:

| Benchmark | no tools | w/ tools | heavy |
|---|---|---|---|
| HLE (text-only) | 23.9 | 44.9 | **51.0** |
| AIME 2025 | 94.5 | 99.1 (w/ python) | **100.0** |
| HMMT 2025 | 89.4 | 95.1 (w/ python) | — |

Heavy mode is footnoted as **8 trajectories rolled out in parallel then reflectively aggregated**; all results reported at native INT4 ([Moonshot blog](https://www.kimi.ai/blog/kimi-k2-thinking)).

**gpt-oss reports every reasoning benchmark twice × 3 effort levels.** gpt-oss-120b AIME 2025: **50.4 / 80.0 / 92.5** (no tools, low/med/high) vs **72.9 / 91.6 / 97.9** (with tools) — a **22-point swing from the tool setting alone at low effort** ([arXiv:2508.10925](https://arxiv.org/html/2508.10925v1)).

**Kimi K3's 2026 tables** footnote "All maxed out on thinking effort: max or xhigh" and "with reasoning effort set to 'max' and temperature = 1.0" ([K3 report §6.1.3](https://github.com/MoonshotAI/Kimi-K3)).

**Parallel test-time compute is disclosed in prose but mixed into tables.** Grok 4 Heavy uses "parallel test-time compute… multiple hypotheses at once," but the launch table mixes Grok 4 and Grok 4 Heavy rows ([x.ai](https://x.ai/news/grok-4)). GPT-5 Pro is "a variant of GPT-5 that thinks for ever longer, using scaled but efficient parallel test-time compute," with an unusually explicit coding footnote: "All SWE-bench evaluation runs use a fixed subset of n=477 verified tasks" — i.e. not the full 500 ([OpenAI](https://openai.com/index/introducing-gpt-5)).

**ARC-AGI published the reference compute ledger** for o3 — the clearest illustration that a headline number is a *compute setting*:

| Config | Samples/task | Score | $/task | Total | Tokens |
|---|---|---|---|---|---|
| high-efficiency | 6 | **75.7%** | $26 | $2,680 | 33.5M |
| low-efficiency | **1,024** | **87.5%** | $4,560 | $456,000 | 5.7B |

**172× compute for +11.8 points.** Also disclosed: "OpenAI shared they trained the o3 we tested on **75% of the Public Training set**" ([ARC Prize](https://arcprize.org/blog/oai-o3-pub-breakthrough)).

### How much of a reported score is not the model

Epoch AI quantified the variance that comes from settings alone ([Epoch](https://epoch.ai/gradient-updates/why-benchmarking-is-hard)):

- **Scaffold** is the biggest single factor: "simply switching the scaffold makes up to an **11% difference for GPT-5** and up to a **15% difference for Kimi K2 Thinking**" on SWE-bench Verified.
- **Serving stack**: OpenAI reported up to **3%** improvement on SWE-bench Verified just from using their Responses API; MiniMax claimed a **23 percentage point** tau-bench difference between their own API and standard ChatCompletions.
- **Prompt/temperature**: gpt-oss on GPQA-Diamond ranged **74%–80%** across settings — not statistically significant, since GPQA-Diamond is only **198 questions**.
- Epoch's own protocol: run most models **16× on GPQA Diamond and Mock AIME, 8× on MATH Level 5**. They measured Claude 3.5 Sonnet at **0.55 ± 0.03** on GPQA Diamond where Anthropic reported 65% ([Epoch methodology](https://epoch.ai/benchmarks/about)).

**The Leaderboard Illusion** ([arXiv:2504.20879](https://arxiv.org/abs/2504.20879)) documents the arena-level version: **27 private LLM variants tested by Meta** before Llama-4 with only the best disclosed; Google ~19.2% and OpenAI ~20.4% of all arena data vs **83 open-weight models sharing ~29.7%**; and even limited additional arena data yields **relative gains up to 112%** on the arena distribution.

## 6.4 The 2026 standard eval set

MMLU, GSM8K, HumanEval, MATH-500 and ARC **no longer appear on a flagship open model card** — they are saturated. MMLU-Pro top scores now cluster ~89–90%, which Epoch flags as loss of discriminative power ([Epoch](https://epoch.ai/benchmarks)).

What a September 2026 flagship open-weight card actually reports (from the Kimi K2.6 card, [HF](https://huggingface.co/moonshotai/Kimi-K2.6)):

- **Agentic**: HLE-Full (w/ tools), BrowseComp, BrowseComp (Agent Swarm), DeepSearchQA, Toolathlon, MCPMark, Claw Eval (pass^3 and pass@3), OSWorld-Verified
- **Coding**: Terminal-Bench 2.0, SWE-Bench Pro, SWE-Bench Multilingual, SWE-Bench Verified, SciCode, OJBench, LiveCodeBench v6
- **Reasoning & knowledge**: HLE-Full, AIME 2026, HMMT 2026, IMO-AnswerBench, GPQA-Diamond
- **Vision**: MMMU-Pro (± python), CharXiv, MathVision (± python), V* (w/ python)

Kimi K3's card adds: DeepSWE, FrontierSWE, ProgramBench, SWE-Marathon, AutomationBench, GDPval-AA Elo, Video-MME ([K3 report §6](https://github.com/MoonshotAI/Kimi-K3)).

### Where the frontier actually sits, April 2026 (verified, single source, one table)

DeepSeek published a head-to-head table for V4-Pro-Max against five other frontier models, all at maximum reasoning effort. This is the most useful single snapshot because every column was run by the same team under the same harness ([DeepSeek-V4 Table 6](https://arxiv.org/abs/2606.19348)):

| Benchmark | Opus-4.6 (Max) | GPT-5.4 (xHigh) | Gemini-3.1-Pro (High) | K2.6 (Think) | GLM-5.1 (Think) | **DS-V4-Pro (Max)** |
|---|---|---|---|---|---|---|
| MMLU-Pro (EM) | 89.1 | 87.5 | **91.0** | 87.1 | 86.0 | 87.5 |
| SimpleQA-Verified | 46.2 | 45.3 | **75.6** | 36.9 | 38.1 | 57.9 |
| GPQA Diamond | 91.3 | 93.0 | **94.3** | 90.5 | 86.2 | 90.1 |
| HLE (no tools) | 40.0 | 39.8 | **44.4** | 36.4 | 34.7 | 37.7 |
| HLE (with tools) | 53.1 | 52.0 | 51.6 | **54.0** | 50.4 | 48.2 |
| LiveCodeBench | 88.8 | — | 91.7 | 89.6 | — | **93.5** |
| Codeforces (Rating) | — | 3168 | 3052 | — | — | **3206** |
| HMMT 2026 Feb | 96.2 | **97.7** | 94.7 | 92.7 | 89.4 | 95.2 |
| Terminal-Bench 2.0 | 65.4 | **75.1** | 68.5 | 66.7 | 63.5 | 67.9 |
| SWE Verified (Resolved) | **80.8** | — | 80.6 | 80.2 | — | 80.6 |
| SWE Pro (Resolved) | 57.3 | 57.7 | **58.6** | 58.6 | 58.4 | 55.4 |
| BrowseComp | 83.7 | 82.7 | **85.9** | 83.2 | 79.3 | 83.4 |
| GDPval-AA (Elo) | 1619 | **1674** | 1314 | 1482 | 1535 | 1554 |
| Toolathlon | 47.2 | **54.6** | 48.8 | 50.0 | 40.7 | 51.8 |
| MRCR 1M (MMR) | **92.9** | — | 76.3 | — | — | 83.5 |

**Read this as: open weights are inside the frontier band on most axes.** SWE-bench Verified is a four-way tie at ~80. LiveCodeBench and Codeforces are led by an open model. The persistent gaps are **research-level reasoning** (HLE, Apex) and **factual recall** (SimpleQA-Verified, where Gemini's 75.6 is in a different regime entirely).

DeepSeek also discloses an eval caveat worth copying: on Terminal-Bench 2.0 they "acknowledge the environment-related issues noted by GLM-5.1. Nevertheless, we report our performance on the **original** dataset for consistency. On the Terminal-Bench 2.0 **Verified subset**, DeepSeek-V4-Pro achieves a score of approximately [higher]" ([§5.3.1](https://arxiv.org/abs/2606.19348)).

Other verified 2026 points: **Kimi K3** — GPQA-Diamond 93.5, DeepSWE 67.5, Terminal-Bench 2.1 88.3, BrowseComp 91.2, Video-MME 90.0, HLE-Full 56.0 w/ tools and 43.5 without ([K3 report §6.1.4](https://github.com/MoonshotAI/Kimi-K3)). **Kimi K2.6** — HLE-Full w/ tools 54.0, SWE-Bench Pro 58.6, SWE-Bench Verified 80.2, AIME 2026 96.4 ([HF card](https://huggingface.co/moonshotai/Kimi-K2.6)).

## 6.5 What full eval disclosure looks like in 2026

Kimi K3's "Evaluation Configurations" section is the current high-water mark and a useful checklist of everything that can silently change a number ([§6.1.3](https://github.com/MoonshotAI/Kimi-K3)):

- **Decoding**: "All Kimi K3 evaluations use reasoning effort max and temperature = 1.0." **top-p = 0.95 for single-step reasoning/knowledge tasks; top-p = 1.0 for coding and agentic tasks** — with an explicit recommendation to users to do the same.
- **Harness, named per model**: each model evaluated under one of Kimi Code, Claude Code, or Codex; on Terminal-Bench 2.1 they "report the best score across harnesses for **all** models"; on the Agents' Last Exam leaderboard they list which harness each competitor used.
- **Task version and date**: "DeepSWE… v1.1 tasks"; "FrontierSWE dominance scores are recomputed from raw scores using the official evaluation script **as of July 16, 2026**"; third-party scores "cited from Artificial Analysis as of **July 23, 2026**."
- **Hardware calibration**: SWE-Marathon run on "an H20-calibrated branch of the official tasks… with Docker images, performance gates, and reference oracles for the GPU tasks recalibrated for H20 but the correctness and anti-cheat validators unchanged"; PostTrainBench "averaged over three runs on H20 GPUs (instead of H100 in the official setting)."
- **Number of runs**: vision scores averaged over three runs, ZeroBench-main over five.
- **Context management as a disclosed variable**: "For BrowseComp we adopt a context-compaction strategy triggered at 300K tokens; evaluated with the full 1M-token context window and **no context management, Kimi K3 achieves 90.4%**" — versus the 91.2% headline.
- **Competitor caveats**: "Claude Fable 5 hits fallbacks on 35% of the tasks" on SWE-Marathon.
- **Prompt interventions**: "For WorldVQA, we observe consistent refusal behavior across models and enforce an answer via prompt engineering."

Compare this against Epoch AI's measured variance (§6.3) — scaffold alone moves SWE-bench Verified by 11–15 points — and the value becomes obvious. **If a launch post does not name the harness, the decoding parameters, the task version, and the number of runs, its numbers are not comparable to anything.**

Kimi K3 also reports **cost efficiency** as a first-class result, which is rare: on Kimi Code Bench 2.0 it is "4.0 points behind Claude Fable 5 at **38% of its cost**, and at high effort it already matches Claude Opus 4.8's maximum-effort score at roughly **one third of the cost**" ([§6.4](https://github.com/MoonshotAI/Kimi-K3)).

---

# Stage 7: The safety / behavior layer

> **2026 default choice: write a spec, train the model on the spec's *text* (not just labels derived from it), ship a separate policy-conditioned classifier, and — if releasing open weights — run an adversarial fine-tuning evaluation before release.**

## 7.1 Specs and constitutions have replaced principle lists

**OpenAI Model Spec** — first published May 8 2024, major update Feb 12 2025, dedicated to the public domain under **CC0** ([model-spec.openai.com](https://openai.com/index/introducing-the-model-spec)). The authority hierarchy is now **Root → System → Developer → User → Guideline** (the top level was renamed from "Platform" to "Root" and elevated *above* System, to mark what cannot be overridden in any conversation). Latest in-scope update: **August 18, 2026**, adding teen relational-interaction principles, handling of false/unsupported premises, and a new section "Be clear about capabilities and limits" ([release notes](https://help.openai.com/en/articles/9624314-model-release-notes)).

**Anthropic's Claude constitution**, published **January 2026** under a Creative Commons license ([anthropic.com/constitution](https://www.anthropic.com/constitution); [announcement](https://www.anthropic.com/news/claude-new-constitution)). Four properties in **strict priority order**:

1. **Broadly safe** — not undermining appropriate human mechanisms to oversee AI during the current phase of development
2. **Broadly ethical** — honest, good values, avoiding harmful actions
3. **Compliant with Anthropic's guidelines**
4. **Genuinely helpful**

"In cases of apparent conflict, Claude should generally prioritize these properties in the order in which they're listed." Length is reported inconsistently across outlets (~23,000 words / ~57 pages vs ~80+ pages) — **UNVERIFIED**.

**The structural shift matters more than the content**: the 2023 Constitutional AI constitution was a *list* of ~50+ principles (UN Declaration of Human Rights, Apple ToS, DeepMind Sparrow rules) sampled one at a time during critique-and-revision. The 2026 document is an **explanatory text teaching judgment rather than rule-following**.

No equivalent public per-model behavior spec was found for Google, xAI, Meta, Qwen, DeepSeek or Mistral — **UNVERIFIED**.

## 7.2 Constitutional AI → deliberative alignment

**Constitutional AI** ([arXiv:2212.08073](https://arxiv.org/abs/2212.08073)): **SL-CAI** (model critiques and revises its own responses, drawing one constitutional principle per pass, then finetunes on revisions) → **RL-CAI** (a preference model trained on **AI-generated** preference labels, then RL against it).

**Deliberative alignment** ([arXiv:2412.16339](https://arxiv.org/abs/2412.16339)) is the important 2025–2026 successor because it is the only approach that teaches the model **the text of the spec**:

1. Train for helpfulness with **no safety-relevant data**.
2. Build (prompt, CoT, completion) tuples whose CoTs **reference the specifications** — generated by inserting the safety-spec text into the system prompt, sampling completions, then **removing the system prompt from the data**.
3. **Incremental SFT** on that dataset, filtered by a policy-aware reward model.
4. **RL where the reward model has access to the safety policies.**

No human-labeled CoTs required. The paper's explicit contrast: "Even though RLAIF methods like CAI use safety specifications to generate training labels, only the labels themselves are used in training. **Knowledge of the specifications themselves is thereby lost to the model.**" Result: a **Pareto improvement on the XSTest overrefusal × StrongREJECT plane** over GPT-4o, Claude 3.5 Sonnet/Haiku, Gemini 1.5 Pro/Flash. Exact per-model values are only in the figure — **UNVERIFIED at the digit level**.

## 7.3 Safe completions: the 2026 replacement for binary refusal

The biggest practical change in refusal training. Instead of classifying user *intent* and then complying or refusing, **safe-completions** maximize helpfulness **subject to constraints on the assistant's output**, using two training parameters (output safety, helpfulness) rather than one binary boundary. Motivation: binary boundaries are brittle under obscured intent and unsuited to dual-use domains (bio, cyber) where a high-level answer is safe but an actionable one gives uplift. Measured against a refusal-trained baseline (o3), it improved safety **especially on dual-use prompts**, **reduced severity of residual failures**, and **substantially increased helpfulness** ([OpenAI](https://openai.com/index/gpt-5-safe-completions); arXiv 2508.09224).

Related: a **Safety Reasoner** RL-trained on policy-labelling tasks lets a lab "dynamically update safety policies **in production in less time than it would take to retrain a classifier**" ([OpenAI](https://openai.com/index/introducing-gpt-oss-safeguard)).

## 7.4 Shipped safety classifiers

| Model | Sizes | Base | Taxonomy |
|---|---|---|---|
| Llama Guard 3 | 1B / 8B | Llama 3 | 13 (S1–S13) |
| **Llama Guard 4** | 12B, multimodal, **pruned dense from Llama 4 Scout**, 1M context | Llama 4 | 14 (adds **S14 Code Interpreter Abuse**) |
| **Qwen3Guard** | 0.6B / 4B / 8B, Gen and Stream variants | Qwen3 | ~10 |
| ShieldGemma / 2 | 2B/9B/27B; 4B image | Gemma 2 / 3 | 4 |
| **gpt-oss-safeguard** | 120b / 20b, Apache 2.0, CoT exposed | gpt-oss | **developer-supplied policy at inference time** |

**gpt-oss-safeguard (Oct 2025) is the 2026 inflection**: it takes **two inputs — policy and content — and reasons over the policy at inference time**, so "the policy is provided during inference, rather than being trained into the model," letting a developer revise policy without retraining ([OpenAI](https://openai.com/index/introducing-gpt-oss-safeguard)).

## 7.5 Open-weight release practice: malicious fine-tuning evaluation

The strongest new norm, established by gpt-oss ([arXiv:2508.03153](https://arxiv.org/abs/2508.03153), "Estimating Worst-Case Frontier Risks of Open-Weight LLMs" — note this is a *different paper* from the gpt-oss model card at 2508.10925).

**Method**: simulate an attacker by fine-tuning the model to be **as capable as possible** in the risk categories where crossing "High" was plausible — **biorisk** (threat-creation tasks in an RL environment with web browsing) and **cyber** (agentic coding environment solving CTF challenges).

**Result**: "MFT gpt-oss **underperforms OpenAI o3**, a model that is below Preparedness High capability level for biorisk and cybersecurity." The adversarially fine-tuned checkpoints were **not released**. **METR reviewed the methodology and submitted 17 recommendations, 6 high-urgency** ([METR](https://metr.org/blog/2025-10-23-gpt-oss-methodology-review)).

The reasoning for why this is necessary for open weights specifically: "Once they are released, determined attackers could fine-tune them to bypass safety refusals or directly optimize for harm **without the possibility for OpenAI to implement additional mitigations or to revoke access**."

**As of this research, no confirmed instance of a non-OpenAI lab running a comparable adversarial-fine-tuning evaluation before an open-weight release was found — the practice has not propagated.** (UNVERIFIED negative.)

## 7.6 System prompts as the patch layer

Anthropic is the only major lab publishing production system prompts ([docs.claude.com release notes](https://docs.claude.com/en/release-notes/system-prompts)). Growth: **~358 words (Claude Opus 3, Jul 2024) → ~3,235 words (Claude Opus 5, Jul 2026)**, roughly 9× in two years. (A widely circulated "17,000 words" figure is **UNVERIFIED** and probably counts tool definitions.)

What a 2026 production prompt contains, beyond persona: post-cutoff world facts the model cannot know; an explicit knowledge-cutoff behavioral rule; child-safety rules; and a set of **classifier-driven mid-conversation injections** (`cyber_warning`, `ethics_reminder`, `ip_reminder`, `long_conversation_reminder`, etc.).

**The lesson for a recipe**: the system prompt is where post-training fixes land *before* they can be trained in. Budget for it as a maintained artifact, not a string constant.

---

# Stage 8: Open reproductions — what worked, what didn't

## 8.1 Results table

| Project | Base | Compute / cost | Headline result | What it actually is |
|---|---|---|---|---|
| **Open-R1** (HF) | — | — | OpenR1-Math-220k, CodeForces-CoTs, Mixture-of-Thoughts (350k), OpenR1-Distill-7B | **Step 1 (distillation) completed; Steps 2–3 (RL from base) never declared complete** |
| **DeepScaleR-1.5B** | R1-Distill-Qwen-1.5B | **3,800 A100-h ≈ $4,500**, ~1,750 steps | **AIME 2024 43.1**, MATH-500 87.8 | RL *on top of* a distill |
| **OpenThinker3-7B** | Qwen2.5-7B | 1,000+ ablations | AIME 2025 **53%**, LCB 51%, GPQA-D 54% | Pure SFT distillation from QwQ-32B |
| **Light-R1-32B** | Qwen2.5-32B-**Instruct** | **≤6h on 12×H800 ≈ $1,000** | **AIME24 76.6**, AIME25 64.6 | Curriculum SFT + DPO + merging |
| **Sky-T1-32B** | Qwen2.5-32B-Instruct | **~$450** | AIME24 43.3, MATH500 86.4 | 17K QwQ traces |
| **s1 / s1.1-32B** | Qwen2.5-32B-Instruct | **26 min on 16×H100** | AIME24 64.7 (s1.1) | 1,000 samples + budget forcing |
| **LIMO** | Qwen2.5-32B-Instruct | — | AIME24 63.3, MATH500 95.6 (revised) | **817 samples** |
| **TinyZero** | Qwen2.5-3B | **<$30** | "aha moment" on Countdown | Task-specific arithmetic/search |
| **Open-Reasoner-Zero-32B** | Qwen2.5-32B | **1/10 the training steps** of R1-Zero-Qwen-32B | beats R1-Zero-Qwen-32B on AIME24/MATH500/GPQA-D | Vanilla PPO, GAE λ=γ=1, **no KL** |
| **Skywork-OR1-32B** | — | — | AIME24 **79.7**, AIME25 **69.0**, LCB 63.9 | Closest open approach to R1 (79.8/70.0/65.9) |
| **DeepSWE-Preview** | Qwen3-32B | pure RL | **SWE-Bench-Verified 42.2 Pass@1 / 59% with TTS** | First strong open agentic RL result |
| **Marin 8B Base** | from scratch | 12.7T tokens | **Beats Llama 3.1 8B Base on 14 of 19 evals** (68.3 vs 67.0 avg) | Fully reproducible open lab notebook |
| **Marin 32B Base** | from scratch | — | Beats Olmo 2 32B on 32/42 tasks; **ties Olmo 3 32B Base in win rate** | Still behind Qwen 2.5 32B |

Sources: [Open-R1](https://github.com/huggingface/open-r1); [DeepScaleR](https://huggingface.co/agentica-org/DeepScaleR-1.5B-Preview); [OpenThoughts arXiv:2506.04178](https://arxiv.org/abs/2506.04178); [Light-R1](https://huggingface.co/qihoo360/Light-R1-32B); [Sky-T1](https://github.com/novasky-ai/skythought); [s1 arXiv:2501.19393](https://arxiv.org/abs/2501.19393); [LIMO arXiv:2502.03387](https://arxiv.org/abs/2502.03387); [TinyZero](https://github.com/Jiayi-Pan/TinyZero); [ORZ arXiv:2503.24290](https://arxiv.org/abs/2503.24290); [rLLM](https://github.com/rllm-org/rllm); [Marin](https://marin.community/blog/2025/05/19/announcement).

### DeepScaleR's context-length schedule (the most-copied small-compute RL recipe)

- **8K context, steps 0–1040**: AIME 22.9% → 33%. 8×A100-80GB, batch = 128 prompts × 8 samples.
- **16K, steps 1040–1520**: 33% → 38%. 32×A100, batch = 128 × 16.
- **24K, step 1520+**: crosses 40% after ~50 steps, **43% at step 200**.

The motivating measurement: on the base model, **incorrect AIME responses averaged 20,346 tokens vs 6,395 for correct ones.** After the 8K phase, correct dropped to 3,661 and incorrect to 6,977 — "+5% accuracy with one-third of the tokens." Their stated counterfactual: replicating R1's setup directly (≥32K context, ~8000 steps) "takes **at least 70,000 A100 GPU hours — even for a 1.5B model**"; the iterative schedule is an **18.4× reduction** ([DeepScaleR blog](https://pretty-radio-b75.notion.site/DeepScaleR-Surpassing-O1-Preview-with-a-1-5B-Model-by-Scaling-RL-19681902c1468005bed8ca303013a4e2)).

**Caution — GLM-4.5 contradicts this at scale** (§5.3): they found single-stage RL directly at 64K output length **beats** a staged context ramp. Reconciliation: the ramp is a *compute-saving* device that matters most when rollout length dominates your budget; at frontier scale it costs quality.

## 8.2 The honest gap: four failures that recur

**1. Breadth regresses while the headline number rises.** Sky-T1 is the cleanest evidence, because NovaSky published the **non-reasoning** evals alongside: AIME24 jumps 16.7 → 43.3, while **IFEval drops 78.74 → 75.79**, **MGSM 0-shot drops 42.3 → 33**, **BFCL-v3 drops 58.92 → 53.18**, and GPQA-Diamond lands **18.4 points below o1-preview** despite matching it on AIME ([skythought](https://github.com/novasky-ai/skythought)). **If you publish a reasoning result without non-reasoning evals, you are not reporting the cost.**

**2. RL-from-base at scale was never reproduced.** Open-R1 declared Step 1 complete and never declared Steps 2–3 complete; the project has been quiet since May 2025, with no formal post-mortem. That silence is itself the finding.

**3. Distillation, not RL, is doing the work in almost every cheap result.** DeepSeek's own ablation (quoted by the DeepScaleR team): RL directly on Qwen-32B reaches **47% AIME**, while **distillation alone reaches 72.6%**. s1, LIMO, Sky-T1, OpenThinker and Light-R1 are all distillation from a frontier teacher — **bounded by the teacher, and therefore not a path to the frontier.**

**4. Contamination is measured and real.** Light-R1's audit found MATH-500 "somewhat compromised with tens of questions that are identical or only numbers changed," and warns "we have to pay special attention when we incorporate AIME data up to 2023."

## 8.3 The pass@k debate, and the rule that resolves it

**Against — "Does RL Really Incentivize Reasoning Capacity Beyond the Base Model?"** ([arXiv:2504.13837](https://arxiv.org/abs/2504.13837)): evaluating with **pass@k at large k**, RLVR models beat base models at k=1 but **the ordering reverses at large k**. Conclusion: "the observed reasoning abilities originate from and are **bounded by** the base model." Tested across 6 RLVR algorithms and multiple families. Critically, they find **distillation behaves differently** — it "introduces new reasoning patterns from the teacher and genuinely expands the model's reasoning capabilities."

**For — ProRL** ([arXiv:2505.24864](https://arxiv.org/abs/2505.24864)): KL control + **reference-policy resetting** (periodically hard-reset π_ref to a recent snapshot of π_θ and reinitialize optimizer state) + a **diverse verifiable task suite** (136K problems across math, code, STEM, logic puzzles, instruction following) + enhanced GRPO, enabling **>2,000 training steps** — an order of magnitude more than typical. Nemotron-Research-Reasoning-Qwen-1.5B gains **+15.7% math, +14.4% code, +25.9% STEM, +22.0% instruction following, +54.8% logic puzzles**. The direct rebuttal: re-evaluated at **256 inference samples**, "we identify many tasks where the base model **fails to produce any correct solutions regardless of the amount of sampling**, while our RL-trained model achieves **100% pass rates**."

**The reconciling mechanism — and the single most useful operational finding in this document.** ProRL measured a **significant negative correlation between the base model's initial pass@128 and the RL gain**:

- Where the base model is already strong (high pass@128), RL gives minimal or **negative** gains in reasoning breadth — the model "becomes more confident in a subset of solutions it already understands."
- Where the base model is weak (low pass@128), RL **expands** the boundary.
- Confirmed with a **creativity index**: tasks with minimal post-RL gains have higher overlap with pretraining data.

**Operational rule: measure base-model pass@k *before* you spend the RL budget.** If pass@128 is already high on your target task, RL will sharpen, not extend — and you should be distilling or improving the base instead.

## 8.4 modded-nanogpt: the accumulated trick list

**Task**: train GPT-2 (124M) to **≤3.28 validation cross-entropy on FineWeb, on 8×H100**.
**Current record: 1.23 minutes (record #89, 2026-07-17)** — "MLP down projection in FP8 with efficient delayed scaling metric." README headline: **"under 75 seconds"** vs the llm.c baseline's 45 minutes, and **under 400M tokens** vs llm.c's 10B. That is **~36× wall-clock and ~25× token-efficiency** ([modded-nanogpt](https://github.com/KellerJordan/modded-nanogpt)).

**Inflection points in the record history:**

| Time | Change | Date |
|---|---|---|
| 45 min | llm.c baseline | 2024-05-28 |
| 24.9 min | **Muon introduced** | 2024-10-04 |
| 15.2 min | pad embeddings, **ReLU²**, zero-init projections, **QK-norm** | 2024-10-14 |
| 10.8 min | **untied embedding/head** | 2024-11-03 |
| 8.2 min | **value + embedding skip connections**, momentum warmup, logit softcap | 2024-11-06 |
| 7.2 min | **U-net skip pattern** + double LR | 2024-11-10 |
| 5.03 min | 1024-ctx dense → **64K-ctx FlexAttention** | 2024-11-19 |
| 4.66 min | **attention window warmup** | 2024-11-24 |
| 4.41 min | **Value Embeddings** | 2024-12-04 |
| 2.992 min | merged QKV, **long-short attention**, batched Muon | 2025-01-16 |
| 2.476 min | **Polar Express** replaces Newton–Schulz | 2025-09-29 |
| 2.345 min | **NorMuon** | 2025-10-24 |
| 2.284 min | **Cautious Weight Decay** with schedule | 2025-11-10 |
| 2.203 min | **batch-size schedule** | 2025-11-29 |
| 1.988 min | **multi-token prediction** + untie embed/head at 2/3 of training | 2025-12-22 |
| 1.820 min | **paired head attention** | 2026-01-07 |
| 1.655 min | **bigram hash embedding** | 2026-01-19 |
| 1.363 min | **MUDD skip connections** | 2026-04-22 |
| 1.271 min | MUDD gates + **Lightweight Dynamically Composable MHA** | 2026-05-27 |
| 1.243 min | **prefix token prediction auxiliary loss** | 2026-07-13 |
| **1.23 min** | **FP8 MLP down-projection** | 2026-07-17 |

**Full trick list (from the README):** rotary embeddings, QK-Norm, ReLU²; Muon; FP8 for head with asymmetric rescale and softcap logits; FP8 on MLP forward; zero-init projections (muP-like); skip connections from embedding to every block and from block 3 to 6; extra embeddings mixed into attention values; FlashAttention 3 with long-short sliding-window attention and window-size warmup via YaRN; align training batch starts with EoS and cap document length; accumulate gradients for 2 steps on embedding and lm_head; single activation input for the last 3 attention layers; Polar Express in Muon; a "smear" module for 1-token look-back; sparse attention gate; NorMuon; Cautious Weight Decay scheduled with LR; exponential decay of the residual stream; batch-size schedule; max-seq-length schedule; partial key offset; multi-token prediction; untie embed and lm_head at 2/3 of training; gating on value embeddings and skip connections; paired head attention; bigram hash embedding on ¼ of model_dim with sign trick; MUDD skip connections; learnable XSA; Lightweight Dynamically Composable MHA; prefix-token-prediction auxiliary loss.

**The rules are as valuable as the tricks** — copy them into any benchmark you run: (1) no modifying the train/val data pipelines; (2) must reach ≤3.28 mean val loss with **enough run logs to reach p<0.01** (waived only for pure systems speedups); (3) no extra `torch.compile` / inductor flags; (4) must beat the prior record on the **same hardware**. Plus a readability discretion clause: "A 200-line kernel to drop 300ms is worthwhile. 500 lines that convolute the optimizer layout for a 50ms gain will likely be rejected."

**2026 development worth flagging: AI systems are now setting records** — #32 (hiverge.ai), #60 (Intology's Locus), #69 (Aster), #72 (Station), #87 (Recursive), plus nanochat's "autoresearch" rounds.

## 8.5 nanochat: the reference small-compute end-to-end recipe

The framing changed in 2026. It is no longer organized around $100/$300/$1000 tiers but around a **single `--depth` dial** and a **time-to-GPT-2 leaderboard** measured by DCLM CORE, target **0.256525** on an 8×H100 node ([README](https://github.com/karpathy/nanochat)).

| # | Time (h) | val_bpb | CORE | Change | Date |
|---|---|---|---|---|---|
| 0 | 168 | — | 0.2565 | Original GPT-2 checkpoint (~$43,000) | 2019 |
| 1 | 3.04 | 0.74833 | 0.2585 | d24 baseline | 2026-01-29 |
| 2 | 2.91 | 0.74504 | 0.2578 | d26 undertrained + **fp8** | 2026-02-02 |
| 3 | 2.76 | 0.74645 | 0.2602 | total batch size → **1M tokens** | 2026-02-05 |
| 4 | 2.02 | 0.71854 | 0.2571 | dataset → **NVIDIA ClimbMix** | 2026-03-04 |
| 5 | 1.80 | 0.71808 | 0.2690 | **autoresearch round 1** | 2026-03-09 |
| 6 | **1.65** | 0.71800 | 0.2626 | **autoresearch round 2** | 2026-03-14 |

At ~$3/GPU-hour, an 8×H100 node is ~$24/hour, so the current speedrun is **~$40**. The README quotes "$48 for ~2 hours" and "~$15 on spot"; the Feb 2026 write-up frames it as **$43,000 → $73**, a ~600× reduction in 7 years.

**The original d20 "$100 tier" report card** — the only published full tier table ([discussion #1](https://github.com/karpathy/nanochat/discussions/1)): CORE **0.2219**, ARC-Easy 0.3561, ARC-Challenge 0.2875, MMLU 0.3111, **GSM8K 0.0250**, **HumanEval 0.0671**, ChatCORE 0.0730; total wall-clock 3h51m ≈ $92.40.

**What a $100 model can and cannot do**: it is a **4e19 FLOPs** model — "a bit like talking to a kindergartener." It holds a coherent chat turn, writes stories and poems, and knows basic facts. At 2.5% GSM8K and 6.7% HumanEval it **cannot do grade-school math or write code**, and it hallucinates confidently.

**What worked vs. what failed at this scale** (from the Feb 2026 write-up, [discussion #481](https://github.com/karpathy/nanochat/discussions/481)) — negative results are rarely published, so this is unusually valuable:

- **Worked**: FlashAttention 3 (~9% throughput), sliding-window attention, **Value Embeddings (~150M params at near-zero FLOPs)**, per-layer residual scalars, BOS-aligned dataloaders, split AdamW (embeddings/scalars) + Muon (matrices) with Polar Express, NorMuon, cautious weight decay.
- **Failed**: **multi-token prediction, varlen attention, FP8 for lm_head, skip connections.**

Note the tension: **MTP failed in nanochat but is record #53 in modded-nanogpt**, and is standard at frontier scale. Scale and setup decide — do not port a trick without re-ablating it.

---

# Appendix A: The small-compute build order

If you are an open-source project with 8–64 GPUs rather than 100,000, most of this document is context rather than instruction. This is the part that is actionable.

## A.1 The five choices that actually matter

Ranked by measured effect size per unit of effort, from the evidence above.

### 1. Data quality and mixture — the only lever with a published 2× on it
Nothing else in this document has a bigger measured effect for a fixed compute budget. FineWeb-Edu's classifier-filtered subset, DCLM's fastText top-10%, Kimi K2's rephrasing (**23.76 → 28.94 SimpleQA** at matched tokens, §3.3), and Qwen3's instance-level mixture optimization all move more than any architecture tweak available to you. And **you can evaluate data cheaply**: DataDecide showed that ranking corpora at **150M parameters predicts the 1B winner ~80% of the time**, and that continuous likelihood metrics make several benchmarks **>80% predictable at 0.01% of the compute** ([arXiv:2504.11393](https://arxiv.org/abs/2504.11393)). Llama 3's 40B-token annealing probe is the other cheap instrument ([§3.1.3](https://arxiv.org/html/2407.21783v3)).

Start from an existing filtered corpus — FineWeb-Edu, DCLM-Baseline, Nemotron-CC, **Dolma 3 Mix**, or NVIDIA ClimbMix (which bought nanochat a 2.76h → 2.02h speedup on its own, §8.5). Olmo 3 even ships **150B-token pretraining and 10B-token midtraining sample mixes** explicitly "for accessible experimentation with less compute" ([§2.1](https://arxiv.org/abs/2512.13961)).

Four specifics that matter more at small scale than large (§4.3):
- **Filter harder than the published recipes.** The optimal keep-rate scales as `F_opt(C) = 4e-5 · C^0.25` — **top 3% at 1e20 FLOPs** versus top 10% at 1e22. DCLM's and FineWeb-Edu's cuts are calibrated for a bigger run than yours ([arXiv:2507.12466](https://arxiv.org/html/2507.12466v1)).
- **Spend your effort on the classifier's positive seed, not its architecture** — worth **+3.5 CORE** at 7B on an identical pool, and fastText costs ~67 CPU-hours per trillion tokens.
- **Match the repetition rate between your proxy ablation and your target run**, or the ablation will not transfer: error drops from 0.85 to **0.10** at 1/16 of the target budget ([arXiv:2606.07597](https://arxiv.org/abs/2606.07597)).
- **Evaluate candidate filters by continuing a near-converged 1B model for 10B tokens** — 110 H100-hours instead of 1,200 ([arXiv:2505.05427v1](https://arxiv.org/html/2505.05427v1)).

### 2. A hybrid attention stack (3 cheap : 1 global), and a widened residual stream
These are the two architecture changes with independent 2026 replication at both small and frontier scale, and neither needs a large budget to adopt.

- **Gated-DeltaNet hybrid at 3:1**: Qwen3.8's controlled ablation gives **53.81 vs 51.15 (SWA) vs 49.87 (full)** average over nine benchmarks at 25B-A3B (§1.3), and Olmo Hybrid independently measures **~2× data efficiency vs Olmo 3** at 7B — same MMLU with **49% fewer tokens** ([arXiv:2604.03444](https://arxiv.org/abs/2604.03444)). Olmo Hybrid is the one to copy because it is fully open and 7B.
- **Widened / attentive residual stream**: DeepSeek-V4's mHC, Qwen3.8's Gated Residual, and Kimi K3's AttnRes all landed in 2026, and modded-nanogpt has carried skip-connection variants (U-net, MUDD) since 2024 at **124M** scale. This is a cheap, kernel-free change.

Caveat from MiniMax-M2 (§1.3): the infrastructure tax on linear attention is real — no native prefix caching, unclear speculative-decoding path, high sensitivity to low-precision KV. If you are shipping for inference rather than research, budget for that.

### 3. Muon, plus a re-fit of learning rate and batch size
Muon is now the frontier default (DeepSeek-V4, Kimi K3, Qwen3.8), it is ~100 lines, and it is already in nanochat and modded-nanogpt so you can copy a working implementation.

But the **bigger, more commonly missed win is that Muon moves the optimum.** Qwen3.8 measured **7.8e-3 loss and 4.1 downstream points** left on the table by carrying forward the previous generation's LR/batch recipe (§2.4). Two corollaries you can act on immediately:
- **Skip the batch-size warmup.** With Muon it costs **18.8% more optimizer steps for a slightly worse loss** (§2.3).
- **The optimum is a flat bowl** — √2 in LR, +25% in batch size. You do not need a fine sweep; you need to not be 4× off.

Get the parameter partition right (§2.1): Muon on 2-D weight matrices only; AdamW on embeddings, head, norms, **the router**, and anything vector-shaped. And **split fused qkv/fc1 matrices before orthogonalizing** — the single most likely correctness bug in a from-scratch implementation.

### 4. Midtraining, not just pretraining
A short, high-quality, LR-decaying stage is disproportionately cheap and is where Olmo 3, SmolLM3 and nanochat all get their instruction-following and format competence. Olmo 3's midtrain is **100B tokens (~1.7% of total)** sampled from a 2.2T pool; SmolLM3's decay phase is **1.1T of 11.1T**; long-context extension is **~1%** of tokens in both.

Two specifics to copy:
- **Put instruction data and thinking traces in the base model's midtrain**, deliberately, and select the mix on **post-trainability**, not just base-model quality ([Olmo 3 §2.1](https://arxiv.org/abs/2512.13961)).
- **Decontaminate at the midtraining stage specifically** — "memorization happens most strongly near the end of training" ([Ai2 decon](https://github.com/allenai/decon/blob/main/doc/simple-details.md)).

And the free trick: **run midtraining twice with different data-order seeds and average the weights**, as Olmo 3 32B does.

### 5. Know whether you should be doing RL at all — measure base-model pass@k first
This is the choice that most often wastes a small project's entire budget. ProRL's measured **negative correlation between base-model pass@128 and RL gain** (§8.3) gives you a decision rule before you spend anything:

- **High pass@128 on your target task** → RL will sharpen an existing capability, not extend it. Expect k=1 gains and **k-large regressions**. Better spends: improve the base, or distill from a stronger teacher.
- **Low pass@128** (base model produces *no* correct solutions even at 128–256 samples) → RL can genuinely expand the boundary; ProRL found **100% pass rates on tasks where the base model never succeeded.**

And keep the reproductions' lesson in view (§8.2): **almost every cheap headline result is distillation, bounded by its teacher.** DeepSeek's own ablation — RL on Qwen-32B reaches 47% AIME, distillation alone reaches 72.6% — should set your expectations.

**But do not read that as "skip RL."** Magistral Small's three-way decomposition on identical data (§5.3b) is the number to plan against: **SFT-only 65.4, RL-only 65.8, SFT→RL 70.7** on AIME'24. SFT and RL are worth about the same alone and **~5 points more together**. If you can only afford one, they are interchangeable; the ordering is where the value is.

## A.2 A concrete build order

**Tier 0 — validate the harness (hours, ~$50).** Reproduce nanochat's speedrun end-to-end: tokenizer → pretraining → midtraining → SFT → eval. Target GPT-2 CORE = **0.256525**. You now have a working CORE evaluator, a Muon+AdamW split, and a d12 configuration that trains in ~5 minutes for iteration ([nanochat](https://github.com/karpathy/nanochat)).

**Tier 1 — establish your measurement discipline before any training decisions.**
- Fix the harness (DCLM showed LightEval vs LLM-Foundry changes conclusions).
- Adopt **task clusters + continuous metrics (bits-per-byte)** rather than accuracy at small scale.
- Split benchmarks into **development** and **unseen**, and do not touch the unseen set until development is done (OLMo 2's discipline).
- Stand up **decon** with 8-gram / 0.5-overlap matching.
- Adopt modded-nanogpt's rules: same hardware, enough runs for **p<0.01**, no pipeline changes.

**Tier 2 — data (where your effort should concentrate).** Pick a filtered base corpus; run 150M-scale corpus rankings to choose the mix; build a 3-stage curriculum on the SmolLM3 template (web-heavy → more code/math → decay with upsampled high-quality + instruction data); add a rephrasing pass on your highest-value knowledge/math subset, rephrasing each corpus **at most twice**.

**Tier 3 — architecture.** Start from the Olmo 3 / Olmo Hybrid configuration, since it is fully open, 7B-scale, and documented to the hyperparameter: RMSNorm applied to outputs, QK-Norm, SwiGLU, z-loss 1e-5, grad clip 1.0, no weight decay on embeddings, **3-of-4 layers cheap with the last layer always full attention**, RoPE θ = 5e5 with YaRN on full-attention layers. Then swap the SWA layers for **Gated DeltaNet** (the Olmo Hybrid change). Add a widened/gated residual. Add one MTP layer — but **re-ablate it**, since it failed in nanochat and succeeded in modded-nanogpt.

**Tier 4 — post-training.** SFT → (optional DPO/APO for *behavior* like brevity and format) → RLVR with the 2026 consensus stack: GRPO minus KL, minus std-normalization, plus clip-higher, token-level loss, dynamic/active sampling, and **truncated importance sampling** to correct the vLLM-vs-trainer logprob mismatch. DAPO's ablation says this stack is worth **30 → 50 AIME** over vanilla GRPO (§5.3).

Pick infrastructure rather than writing it: **verl** for breadth and adoption, **slime** if you are on Megatron+SGLang, **prime-rl** for fully-async at very large scale, **Molt** if you want ~9K lines you can actually read. Olmo 3's `OlmoRL` in **open-instruct** is the reference *open recipe* ([§4.4](https://arxiv.org/abs/2512.13961)); `rLLM` is a good agentic-RL substrate ([rllm-org/rllm](https://github.com/rllm-org/rllm)). If you train an MoE, handle **router replay** as well as TIS (§5.7). And implement **pause-on-inference** in your sandboxes before anything else — agents idle ~98% of sandbox lifetime waiting on the model (§5.6).

**Tier 5 — if you have multiple domains.** Do not RL one model on everything. Train specialists, then distill them into one model with on-policy distillation — the pattern GLM-4.5, DeepSeek-V4 and Kimi K3 all converged on independently (§5.0).

## A.3 Things to explicitly not do at small scale

- **Do not reproduce R1's RL setup directly.** DeepScaleR measured it at **≥70,000 A100-hours for a 1.5B model**.
- **Do not carry forward someone else's LR/batch recipe** after changing architecture or optimizer (§2.4).
- **Do not chase validation loss alone.** Qwen3.8's n-gram ablation is the cleanest counterexample: loss decreased monotonically with vocabulary size while downstream accuracy saturated (§1.8). GLM-4.5's head-count finding is the mirror image: no loss improvement, consistent benchmark improvement (§1.1).
- **Do not publish a reasoning result without non-reasoning evals** (§8.2).
- **Do not use FP8/FP4 until the BF16 run is correct.** Olmo 3 trained entirely in bfloat16 at ~43%/41% MFU and is the strongest fully-open model of its generation.
- **Do not assume NoPE is free.** Qwen3.8 found it causes "a substantially higher rate of endless generation after post-training" (§1.4).

---

# Appendix B: Convergent 2026 findings worth internalizing

Independent labs arriving at the same answer is stronger evidence than any single report.

| Finding | Independently reported by |
|---|---|
| 3:1 cheap-to-global attention hybrid | Qwen3.8, Kimi K3, Olmo Hybrid, Olmo 3 (SWA) |
| Widened / attentive residual stream | DeepSeek-V4 (mHC), Qwen3.8 (GR), Kimi K3 (AttnRes), modded-nanogpt (U-net/MUDD) |
| Muon for 2-D weights, AdamW for the rest **including the router** | DeepSeek-V4, Qwen3.8, Kimi K3, nanochat |
| Specialist RL → on-policy distillation into one generalist | GLM-4.5, DeepSeek-V4, Kimi K3 |
| GRPO minus KL, plus clip-higher, plus token-level loss | Olmo 3, GLM-4.5, DAPO lineage |
| Extreme MoE sparsity (3–5% active) with 1–2 shared experts | DeepSeek-V3/V4, Kimi K2/K3, Qwen3.8, GLM-4.5 |
| Aux-loss-free bias routing, γ=0.001 → 0.0, α=1e-4 | DeepSeek-V3/V4, GLM-4.5, Kimi K3 (as QB's baseline) |
| MTP λ = 0.3 → 0.1 at LR-decay onset, depth 1 | DeepSeek-V3, DeepSeek-V4, GLM-4.5 |
| QK-Norm replaces logit soft-capping | Gemma 3, Qwen3, GLM-4.5, MiniMax-M2, Olmo 3, Marin |
| Bound the unbounded thing (activations, decay gates, residual maps) | Kimi K3 (SiTU-GLU, g_min), DeepSeek-V4 (Birkhoff), gpt-oss (clamped SwiGLU) |
| Discrete reasoning-effort levels as shipped variants | Kimi K3, DeepSeek-V4, gpt-oss |
| Fit LR/batch scaling laws empirically per architecture+optimizer | Qwen3.8, Kimi K3, Qwen3 |
| Loss and downstream accuracy can move in opposite directions | Qwen3.8 (n-gram vocab), GLM-4.5 (head count) |
| Report {no tools, with tools} × {reasoning effort} | Kimi K2 Thinking, gpt-oss, Kimi K3 |
| N-gram / bigram embedding tables as near-free capacity | Qwen3.8 (production), modded-nanogpt (record #74) |

---

# Appendix C: Open questions where the field disagrees

Flagging these because a recipe that presents them as settled would be wrong.

1. **Cosine vs WSD.** Kimi K3 and GLM-4.5 both chose cosine in 2026 after their own comparisons; Kimi's explanation (the two schedules have different optima, so shared-hyperparameter comparisons are unfair) is the most careful published statement, but it is one lab's result. §2.2.
2. **Linear/hybrid attention vs full attention.** Qwen3.8, Kimi K3 and Olmo Hybrid measure clear wins; MiniMax-M2 reverted after trillions of tokens of failed SWA-hybrid experiments and lists concrete infrastructure blockers. The distinction may be gated-delta-rule vs sliding-window, but this is not settled. §1.3.
3. **NoPE.** Kimi K3 extrapolates to 1M with no positional encoding at all; Qwen3.8 rejected NoPE for post-training endless-generation failures. §1.4.
4. **Staged context length during RL.** DeepScaleR's 8K→16K→24K ramp is the most-copied small-compute recipe; GLM-4.5 measured single-stage 64K as *better*. Likely a compute-budget-vs-quality tradeoff, but unresolved. §5.3, §8.1.
5. **Does RL expand capability or only elicit it?** arXiv:2504.13837 vs ProRL (arXiv:2505.24864). Reconcilable via base-model pass@k, but the reconciliation is an inference from ProRL's correlation, not a settled result. §8.3.
6. **Does decontamination change downstream performance?** Ai2's own ablation found no clear effect either way, while Light-R1's audit found substantial identical-question contamination in public data. §6.2.
7. **MTP at small scale.** Failed in nanochat, succeeded in modded-nanogpt, standard at frontier scale. §8.5.
8. **How much of frontier pretraining data is now synthetic**, and whether practical model collapse is observable. Kimi K2 is candid that this "remains an active area of investigation." §3.3.
9. **How long aggressive data filtering remains correct.** "A Bitter Lesson for Data Filtering" ([arXiv:2605.19407](https://arxiv.org/abs/2605.19407)) measures unfiltered pools beating all five tested filters at 330M+ params given enough steps, and projects a global crossover near 1e30 FLOPs — but this is one setup with a 100B-token ceiling, and the authors themselves scope the claim to the compute-unbounded regime. §4.3.
10. **How much Muon is really worth.** The only rigorous head-to-head ([arXiv:2509.02046](https://arxiv.org/abs/2509.02046)) measures the matrix-preconditioner advantage *shrinking* with scale — 1.4x at 0.1B to 1.1x at 1.2B — while three frontier labs adopted it in 2026 at 100B-2.8T scale for stability and large-batch reasons the study does not measure. Nobody has published a fair comparison above 1.2B. §2.1.
11. **Does QK-norm help or hurt?** Universal in 2026 architectures, but a controlled 26-model study measures a **6-point HELMET gain from removing it** on the Olmo architecture and a **3.8-point loss from adding it** to Llama 3's. §1.5.
12. **Is extreme MoE sparsity right?** DeepSeek-V4-Pro (64) and Kimi K3 (56) say yes; [arXiv:2601.08215](https://arxiv.org/abs/2601.08215) argues that once memory is priced in you should **minimize** sparsity and maximize total parameters. §1.1.
13. **Optimal weight decay.** The 0.1 default may be 3–10× too low ([arXiv:2602.11137](https://arxiv.org/abs/2602.11137)) or **30×** too low ([arXiv:2509.14786](https://arxiv.org/abs/2509.14786)). Nobody at the frontier has published a revision. §2.1.
14. **Whether model-based filters are measuring quality or surface form.** A Wikipedia-style rewrite flips the FineWeb-Edu classifier on ~7% of documents, and filtered corpora show markedly lower domain-diversity entropy. §4.3.

---

# Appendix D: Source index

## Primary technical reports — architecture and pretraining

| Source | URL |
|---|---|
| DeepSeek-V3 | https://arxiv.org/abs/2412.19437 |
| DeepSeek-R1 | https://arxiv.org/abs/2501.12948 |
| **DeepSeek-V4** (Apr 2026) | https://arxiv.org/abs/2606.19348 |
| DeepSeek NSA (Native Sparse Attention) | https://arxiv.org/abs/2502.11089 |
| DeepSeek-V3.2-Exp (DSA) | https://huggingface.co/deepseek-ai/DeepSeek-V3.2-Exp |
| Kimi K1.5 | https://arxiv.org/abs/2501.12599 |
| Kimi K2 | https://arxiv.org/abs/2507.20534 |
| **Kimi K3** (Sep 2026) | https://github.com/MoonshotAI/Kimi-K3 (`k3_tech_report.pdf`) |
| Kimi K2.6 model card | https://huggingface.co/moonshotai/Kimi-K2.6 |
| Kimi K2 Thinking | https://www.kimi.ai/blog/kimi-k2-thinking |
| Moonlight / "Muon is Scalable" | https://arxiv.org/abs/2502.16982 |
| Qwen3 | https://arxiv.org/abs/2505.09388 |
| **Qwen3.8-Flash-Next** (Aug 2026) | https://github.com/QwenLM/Qwen3.8-Flash-Next (`tech_report.pdf`) |
| Qwen3.8-Flash-Next model card | https://huggingface.co/Qwen/Qwen3.8-Flash-Next |
| GSPO | https://qwenlm.github.io/blog/gspo/ |
| Llama 3 herd | https://arxiv.org/abs/2407.21783 |
| Llama 4 | https://ai.meta.com/blog/llama-4-multimodal-intelligence/ |
| Gemma 3 | https://arxiv.org/abs/2503.19786 |
| gpt-oss model card | https://arxiv.org/abs/2508.10925 |
| gpt-oss worst-case frontier risks (MFT) | https://arxiv.org/abs/2508.03153 |
| GLM-4.5 | https://arxiv.org/abs/2508.06471 |
| MiniMax-M2 analysis (LMSYS) | https://www.lmsys.org/blog/2025-11-04-miminmax-m2/ |
| MiniMax-M2 report notes (Raschka) | https://sebastianraschka.com/blog/2026/minimax-m2-technical-report.html |
| **Olmo 3** | https://arxiv.org/abs/2512.13961 · https://allenai.org/blog/olmo3 |
| **Olmo Hybrid** (Apr 2026) | https://arxiv.org/abs/2604.03444 · https://allenai.org/blog/olmohybrid |
| SmolLM3 | https://huggingface.co/blog/smollm3 |
| Marin | https://marin.community/blog/2025/05/19/announcement |

## Data

| Source | URL |
|---|---|
| FineWeb / FineWeb-Edu | https://arxiv.org/abs/2406.17557 |
| DCLM (and the CORE metric) | https://arxiv.org/abs/2406.11794 |
| Nemotron-CC | https://arxiv.org/abs/2412.02595 |
| Data-constrained scaling (Muennighoff et al.) | https://arxiv.org/abs/2305.16264 |
| Paloma | https://arxiv.org/abs/2312.10523 |
| PRISM (mid-training study, 2026) | https://arxiv.org/html/2603.17074v2 |
| **A Bitter Lesson for Data Filtering** (2026) | https://arxiv.org/abs/2605.19407 |
| **BETR — optimal keep-rate scaling law** | https://arxiv.org/html/2507.12466v1 |
| **Olmix — proxy-swarm mixing configuration study** (2026) | https://arxiv.org/html/2602.12237v1 |
| **Repetition-rate-controlled mixing ablations** (2026) | https://arxiv.org/abs/2606.07597 |
| RegMix | https://arxiv.org/html/2407.01492v2 |
| **Fantastic Pretraining Optimizers and Where to Find Them** | https://arxiv.org/abs/2509.02046 |
| muP / Tensor Programs V | https://arxiv.org/abs/2203.03466 |
| Scaling Laws for Fine-Grained MoE | https://arxiv.org/abs/2402.07871 |
| Chinchilla | https://arxiv.org/abs/2203.15556 |
| WSD / MiniCPM | https://arxiv.org/abs/2404.06395 |
| Muon (Keller Jordan) | https://kellerjordan.github.io/posts/muon/ |
| Moonlight — Muon is Scalable for LLM Training | https://arxiv.org/abs/2502.16982 |
| Polar Express (NS coefficient schedule) | https://arxiv.org/abs/2505.16932 |
| Benchmarking Optimizers for LLM Pretraining (EPFL) | https://arxiv.org/abs/2509.01440 |
| Fantastic Pretraining Optimizers II: Hyperball (2026) | https://arxiv.org/abs/2606.16899 |
| HP transfer rescues matrix optimizers (2026) | https://arxiv.org/abs/2512.05620 |
| SOAP, Muon and Beyond at 8B/72B (NVIDIA, 2026) | https://arxiv.org/abs/2607.20548 |
| Practical Efficiency of Muon (Essential AI) | https://arxiv.org/abs/2505.02222 |
| muP does not transfer across token horizon | https://arxiv.org/abs/2409.19913 |
| Weight decay may matter more than muP | https://arxiv.org/abs/2510.19093 |
| StepLaw (3,700 LLMs, LR+batch fits) | https://arxiv.org/abs/2503.04715 |
| Power Lines (Cerebras) | https://arxiv.org/abs/2505.13738 |
| Optimal LR schedules under functional scaling laws (2026) | https://arxiv.org/abs/2602.06797 |
| Straight to Zero (D2Z) | https://arxiv.org/abs/2502.15938 |
| Cooldown / decay-shape study (Hägele et al.) | https://arxiv.org/abs/2405.18392 |
| Critical batch size scales with data | https://arxiv.org/abs/2410.21676 |
| Epoch AI — Chinchilla replication | https://arxiv.org/abs/2404.10102 |
| Epoch AI — tokens per parameter trend | https://epoch.ai/data-insights/training-tokens-per-parameter |
| Beyond Chinchilla (inference-aware) | https://arxiv.org/abs/2401.00448 |
| Test-Time Scaling Makes Overtraining Compute-Optimal (2026) | https://arxiv.org/abs/2604.01411 |
| Parameters vs FLOPs — optimal MoE sparsity (Apple) | https://arxiv.org/abs/2501.12370 |
| Principled MoE design under memory constraints (2026) | https://arxiv.org/abs/2601.08215 |
| Pretraining LLMs with NVFP4 | https://arxiv.org/abs/2509.25149 |
| Quartet (MXFP4 training) | https://arxiv.org/abs/2505.14669 |
| Scaling Laws for Precision | https://arxiv.org/abs/2411.04330 |
| Architecture choices interact (26-model study, 2026) | https://arxiv.org/abs/2608.10296 |
| Weight Decay Improves LM Plasticity (2026) | https://arxiv.org/abs/2602.11137 |
| The Art of Scaling RL Compute (2026) | https://arxiv.org/abs/2510.13786 |
| Distillation Scaling Laws (Apple) | https://arxiv.org/abs/2502.08606 |
| DAPO | https://arxiv.org/abs/2503.14476 |
| Token-count heuristics beat learned mixes (Meta) | https://arxiv.org/abs/2501.11747 |
| Ultra-FineWeb / UltraData tiers | https://arxiv.org/html/2505.05427v1 · https://arxiv.org/abs/2602.09003 |
| Essential-Web (EAI-Distill annotator) | https://arxiv.org/html/2506.14111v2 |
| CuraWeb — domain-specific thresholds | https://arxiv.org/pdf/2607.22662 |
| Filters reward surface form (2026) | https://arxiv.org/abs/2605.23721 |
| Gaperon — neural filtering amplifies leakage | https://arxiv.org/abs/2510.25771 |
| Register-based quality priors | https://arxiv.org/abs/2504.01542 |
| Datasets, Documents, and Repetitions | https://arxiv.org/abs/2503.07879 |
| RedPajama-V2 filter ablations | https://arxiv.org/html/2411.12372v1 |
| DSIR | https://arxiv.org/html/2302.03169v3 |
| CLIMB | https://arxiv.org/html/2504.13161v2 |
| Aioli | https://arxiv.org/html/2411.05735v2 |
| Spending compute on data (FineWeb cost analysis) | https://huggingface.co/blog/maxidl/spend-compute-on-data |
| Mid-Training of LLMs: A Survey | https://arxiv.org/abs/2510.06826 |
| A Survey on LLM Mid-Training | https://arxiv.org/abs/2510.23081 |
| Midtraining Bridges Pretraining and Posttraining | https://arxiv.org/abs/2510.14865 |
| Dolmino midtrain mix (itemized) | https://huggingface.co/datasets/allenai/dolma3_dolmino_mix-100B-1025 |
| Scaling Laws with Vocabulary | https://arxiv.org/abs/2407.13623 |
| Optimal synthetic share (~30%, >1000 LLMs) | https://arxiv.org/abs/2510.01631 |
| Soft Contamination (2026) | https://arxiv.org/abs/2602.12413 |
| Microsoft MAI-Base-1 tech report | https://microsoft.ai/pdf/mai-thinking-1.pdf |
| FinePDFs | https://huggingface.co/spaces/HuggingFaceFW/FinePDFsBlog |
| Gemma 3n Per-Layer Embeddings | https://developers.googleblog.com/en/introducing-gemma-3n/ |

## Evaluation

| Source | URL |
|---|---|
| OLMES | https://arxiv.org/abs/2406.08446 |
| DataDecide | https://arxiv.org/abs/2504.11393 |
| Signal and Noise | https://arxiv.org/abs/2508.13144 |
| Observational Scaling Laws | https://arxiv.org/abs/2405.10938 |
| Predicting downstream capabilities | https://arxiv.org/abs/2406.04391 |
| The Leaderboard Illusion | https://arxiv.org/abs/2504.20879 |
| Epoch AI — why benchmarking is hard | https://epoch.ai/gradient-updates/why-benchmarking-is-hard |
| ARC Prize — o3 compute ledger | https://arcprize.org/blog/oai-o3-pub-breakthrough |
| Ai2 `decon` | https://github.com/allenai/decon |

## Post-training and RL

| Source | URL |
|---|---|
| GRPO (DeepSeekMath) | https://arxiv.org/abs/2402.03300 |
| DAPO | https://arxiv.org/abs/2503.14476 |
| Dr. GRPO | https://arxiv.org/abs/2503.20783 |
| GSPO | https://arxiv.org/abs/2507.18071 |
| VAPO | https://arxiv.org/abs/2504.05118 |
| Tulu 3 | https://arxiv.org/abs/2411.15124 |
| Magistral | https://arxiv.org/abs/2506.10910 |
| On-policy distillation (Thinking Machines) | https://thinkingmachines.ai/blog/on-policy-distillation/ |
| DeepSeek-V3.2 | https://arxiv.org/abs/2512.02556 |
| GLM-5 | https://arxiv.org/abs/2602.15763 |
| MAI-Thinking-1 (Microsoft) | https://microsoft.ai/pdf/mai-thinking-1.pdf |
| MiMo-V2-Flash (introduced MOPD) | https://arxiv.org/abs/2601.02780 |
| Truncated importance sampling / TIS (verl PR #2953) | https://github.com/verl-project/verl |
| Does RL incentivize reasoning beyond the base model? | https://arxiv.org/abs/2504.13837 |
| ProRL | https://arxiv.org/abs/2505.24864 |
| Constitutional AI | https://arxiv.org/abs/2212.08073 |
| Deliberative alignment | https://arxiv.org/abs/2412.16339 |
| OpenAI safe completions | https://openai.com/index/gpt-5-safe-completions |
| gpt-oss-safeguard | https://openai.com/index/introducing-gpt-oss-safeguard |
| OpenAI Model Spec | https://openai.com/index/introducing-the-model-spec |
| Claude's constitution (Jan 2026) | https://www.anthropic.com/constitution |
| 2026 post-training survey (off-policy / on-policy unified view) | https://arxiv.org/abs/2604.07941 |
| Understanding Reasoning from Pretraining to Post-Training (2026) | https://arxiv.org/abs/2607.16097 |
| RL for LLM post-training survey | https://arxiv.org/abs/2407.16216 |
| Survey on post-training of LLMs | https://arxiv.org/abs/2503.06072 |

**2026 surveys worth reading in full** (none of these existed when most "how LLMs are trained" write-ups were published):

| Survey | arXiv | One line |
|---|---|---|
| **LLM Post-Training: A Unified View of Off-Policy and On-Policy Learning** | [2604.07941](https://arxiv.org/abs/2604.07941) | Organizes SFT, preference optimization, RL, process supervision and distillation by the *behavioral bottleneck* each addresses. Best single 2026 post-training survey |
| **Unifying Data, Memory, and Compute Efficiency in LLM Training** | [2606.10706](https://arxiv.org/abs/2606.10706) | Constraint-centric view of three coupled bottlenecks |
| Navigating LLM Valley: AdamW to Memory-Efficient and Matrix-Based Optimizers | [2605.09176](https://arxiv.org/abs/2605.09176) | Optimizer survey; warns that "purported gains may shrink under fairer or larger-scale evaluation" |
| OmniOpt: Taxonomy, Geometry and Benchmarking of Modern Optimizers | [2607.04033](https://arxiv.org/abs/2607.04033) | 91pp survey + benchmark; "no single optimizer dominates this multi-objective frontier" |
| Mid-Training of LLMs: A Survey | [2510.06826](https://arxiv.org/abs/2510.06826) | The stage-definition paper |
| Data Mixing for LLM Pretraining: A Survey and Outlook | [2604.16380](https://arxiv.org/abs/2604.16380) | First dedicated domain-level mixing survey |
| A Survey of On-Policy Distillation for LLMs | [2604.00626](https://arxiv.org/abs/2604.00626) | Exposure bias scales ~ square of sequence length |
| Generate, Filter, Control, Replay: Rollout Strategies for LLM RL | [2605.02913](https://arxiv.org/abs/2605.02913) | The under-reported half of RLVR pipelines |
| Rubric-Guided Reinforcement Learning for LMs | [2608.27505](https://arxiv.org/abs/2608.27505) | Constitutions as priors over rubrics |
| SFT versus RL: A Study of Post-Training Methods | [2603.13985](https://arxiv.org/abs/2603.13985) | |
| Understanding Reasoning from Pretraining to Post-Training | [2607.16097](https://arxiv.org/abs/2607.16097) | The pretraining→RL interface, quantified |
| The Art of Scaling Reinforcement Learning Compute for LLMs | [2510.13786](https://arxiv.org/abs/2510.13786) | "Chinchilla for RL" — >400,000 GPU-hours, sigmoidal fits; most recipe details move *efficiency*, not the *ceiling* |

**Note: no dedicated 2026 survey of *scaling laws* appears to exist** — the nearest are [2511.12869](https://arxiv.org/abs/2511.12869) (*On the Fundamental Limits of LLMs at Scale*) and [2504.02181](https://arxiv.org/abs/2504.02181) (*A Survey of Scaling in LLM Reasoning*). UNVERIFIED as a negative.

## Reproductions and small-compute

| Source | URL |
|---|---|
| nanochat | https://github.com/karpathy/nanochat |
| nanochat — beating GPT-2 for <<$100 | https://github.com/karpathy/nanochat/discussions/481 |
| modded-nanogpt | https://github.com/KellerJordan/modded-nanogpt |
| Open-R1 | https://github.com/huggingface/open-r1 |
| DeepScaleR | https://huggingface.co/agentica-org/DeepScaleR-1.5B-Preview |
| OpenThoughts | https://arxiv.org/abs/2506.04178 |
| Light-R1 | https://huggingface.co/qihoo360/Light-R1-32B |
| s1 / budget forcing | https://arxiv.org/abs/2501.19393 |
| LIMO | https://arxiv.org/abs/2502.03387 |
| Sky-T1 | https://github.com/novasky-ai/skythought |
| Open-Reasoner-Zero | https://arxiv.org/abs/2503.24290 |
| SimpleRL-Zoo | https://arxiv.org/abs/2503.18892 |
| TinyZero | https://github.com/Jiayi-Pan/TinyZero |
| rLLM (DeepSWE, DeepCoder) | https://github.com/rllm-org/rllm |
| **HF: Lessons from 16 open-source RL libraries** (2026) | https://huggingface.co/blog/async-rl-training-landscape |
| verl | https://github.com/verl-project/verl |
| NeMo Gym (122 environments) | https://github.com/NVIDIA-NeMo/Gym |
| OpenEnv | https://github.com/huggingface/OpenEnv |
| INTELLECT-3 (prime-rl) | https://www.primeintellect.ai/blog/intellect-3 |
| Marin speedrun | https://marin.community/speedrun |

---

## Document provenance

Compiled 2026-09-13 from primary sources. PDFs of DeepSeek-V4, Kimi K3, Qwen3.8-Flash-Next, Olmo 3, DeepSeek-V3, Kimi K2, Qwen3, gpt-oss, Gemma 3 and GLM-4.5 were downloaded and read in full rather than summarized from secondary coverage; direct quotations are verbatim from those texts.

**Corrections applied during compilation** — flagging these because they circulate widely in incorrect form:

1. **Kimi K2's sparsity baselines are 8, 16, 32** (not 8, 12, 24). The multipliers 1.69× / 1.39× / 1.15× are correct.
2. **DeepSeek-V4-Pro's sparsity is 64** (6-of-384) — higher than Kimi K3's 56. V4-Pro, not K3, is the 2026 extreme.
3. **Kimi K2 did NOT train in FP8 compute** — FP8 *storage* only, explicitly refused for computation. GLM-4.5 and Llama 3 use FP8 for inference only.
4. **The "20 tokens per parameter" figure never appears in the Chinchilla paper**, and its Table 3 Approach 3 implies ~69–94 TPP. Epoch's corrected fit gives ~25.6 with an honest band of 4–40.
5. **Llama 3's "weight decay = 0.1 × learning rate"** describes the 40M–16B *scaling-law proxy models*, not the 405B run.
6. **Gemma 3 gives no stated reason for dropping soft-capping** — only citations. The FlashAttention-incompatibility story is documented as a *Gemma 2* problem, not as Gemma 3's rationale.
7. **MiniCPM contains no exponential-vs-linear decay-shape comparison** — that result is Hägele et al. ([arXiv:2405.18392](https://arxiv.org/abs/2405.18392)).
8. **nanochat's current default vocab is 32,768**, not the 65,536 of the original speedrun.
9. **s1's widely-cited "$50"** is not in the paper; the stated cost is 26 minutes on 16 H100s.
10. **The gpt-oss malicious-fine-tuning paper is [arXiv:2508.03153](https://arxiv.org/abs/2508.03153)**, distinct from the model card at 2508.10925.
11. **MOPD was introduced by MiMo-V2-Flash** ([arXiv:2601.02780](https://arxiv.org/abs/2601.02780)), not Kimi K3 — K3, DeepSeek-V4, GLM-5.2 and Nemotron-Cascade 2 are adopters.
12. **Kimi K2's report never counts domains.** The 3,000+ real MCP tools and 20,000+ synthetic tools figures are verbatim; a widely-repeated "1,000+ domains" figure belongs to **AgentScaler**, a different system.
13. **Thinking Machines' distillation multiples are not one number**: 9× / 18× / 30× are relative to *off-policy SFT*; **50–100× is relative to RL**, from a separate experiment.
14. **"256 sampled logits per token" is Gemma 3**, not Gemma 2 (which uses plain full-distribution NLL).
15. **Do not cite arXiv 2601.11659 "The Llama 4 Herd"** — removed by arXiv administrators for incorrect authorship. No official Meta Llama 4 technical report exists.

**Known gaps and cautions:**
- **Kimi K3 and Qwen3.8 do not disclose absolute pretraining token counts** — only relative scaling-efficiency gains. gpt-oss discloses only "trillions."
- DeepSeek-V3.2's exact DSA top-k and continued-training token counts were not recovered from the model card.
- **Llama 4's MetaP is named but never described.** Its claimed transfer across *batch size and training tokens* is a strictly stronger claim than μP, published with zero technical detail.
- **No source exists for the FP8/FP4/BF16 share of 2026 frontier training** — the qualitative picture in §2.6 is the best available.
- The 2026 open-weight scores in §6.4 are from primary reports where stated; treat aggregator-derived bands as directional.
- Lengths quoted for Anthropic's constitution and Claude's system prompt vary across secondary sources and are marked UNVERIFIED.
- No confirmed instance was found of a non-OpenAI lab running adversarial fine-tuning evaluation before an open-weight release (§7.5) — a negative result, not a proof of absence.
- **No dedicated 2026 survey of scaling laws appears to exist** (UNVERIFIED as a negative).
