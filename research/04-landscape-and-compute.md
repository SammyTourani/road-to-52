# Open Weights vs. the Closed Frontier — Landscape, Compute Economics, and Paths to Beat Claude Fable 5.1

**Research date:** 2026-09-13
**Scope:** (A) state of open-weight models vs. the closed frontier; (B) real compute and dollar cost of training at each tier; (C) economics of every plausible path by which a small team could produce a model that beats Anthropic's Claude Fable 5.1 on at least one benchmark that matters.

**Conventions.** Every number carries a source URL. Items not confirmed against a primary or strongly-reputable source are tagged **UNVERIFIED**. Where a third-party aggregator conflicts with a primary source, the primary wins and the conflict is flagged.

> ### The one-line answer
> **An open-weight model has already beaten Fable 5.1 on a benchmark that matters.** Kimi K3 reports **GPQA Diamond 93.5** vs Fable 5.1's **92.6**, and **BrowseComp ~90.4–91.2** where no Fable 5.1 figure exists but Opus 5 sits at 90.8. The interesting question is therefore not *"can open beat Fable 5.1?"* — it is *"what can **you** build, for under $2k, that beats it on something defensible?"* Part C answers that.

---

## 0. The target: what Claude Fable 5.1 actually scores

Released **2026-09-01** at $10/M input, $50/M output, cache reads cut 75% to $0.25/M ([Anthropic](https://www.anthropic.com/claude-fable-and-mythos-5-1), [Vellum](https://www.vellum.ai/blog/claude-fable-5-1-mythos-5-1-benchmarks-explained)).

| Benchmark | Fable 5.1 | Fable 5 | Opus 5 | GPT-5.6 Sol | Source |
|---|---|---|---|---|---|
| Terminal-Bench-Science 0.1 | **52.6%** | 24.7% | 29.0% | 22.4% | [Anthropic](https://www.anthropic.com/claude-fable-and-mythos-5-1) |
| Terminal-Bench 4.0 | **55.8%** (Mythos 5.1: 60.9%) | 42.0% | 52.3% | 37.3% | [Anthropic](https://www.anthropic.com/claude-fable-and-mythos-5-1) |
| AutomationBench | **31.4%** | 17.1% | 26.9% | 19.6% | [Anthropic](https://www.anthropic.com/claude-fable-and-mythos-5-1) |
| GDPval-AA v2 (Elo) | **1853** | 1723 | 1824 | 1711 | [Anthropic](https://www.anthropic.com/claude-fable-and-mythos-5-1) |
| OSWorld 2.0 (partial / strict) | **77.9% / 41.7%** | 72.9% / 36.1% | 75.4% / 39.6% | — | [Vellum](https://www.vellum.ai/blog/claude-fable-5-1-mythos-5-1-benchmarks-explained) |
| Humanity's Last Exam (no tools / tools) | **60.9% / 65.0%** | 57.8% / 63.8% | 56.6% / 63.6% | — | [Vellum](https://www.vellum.ai/blog/claude-fable-5-1-mythos-5-1-benchmarks-explained) |
| CursorBench 3.2.0 | **73.4%** | 70.5% | 70.0% | 67.2% | [Vellum](https://www.vellum.ai/blog/claude-fable-5-1-mythos-5-1-benchmarks-explained) |
| SWE-bench Pro | **81.2%** | 80.0% | 79.2% | 64.6% | [Kingy](https://kingy.ai/blog/claude-fable-5-1-benchmarks-price-mythos-access) |
| **GPQA Diamond** | **92.6%** | — | — | — | [DataScienceDojo](https://datasciencedojo.com/blog/claude-fable-5-1-performance-and-safety) |
| LiveCodeBench | 90.52% (#1 on Vals) | — | — | — | [Vals AI](https://www.vals.ai/models/anthropic_claude-fable-5-1) |
| ProofBench v1.1 | 100% | — | — | — | [Vals AI](https://www.vals.ai/models/anthropic_claude-fable-5-1) |
| Terminal-Bench 2.1 | 85.02% (#2) | 80.52% | — | 85.77% (#1) | [Vals AI](https://www.vals.ai/models/anthropic_claude-fable-5-1) |
| **Toolathlon Verified** | 77.8% (**#3**) | — | **80.6%** | — | [Kingy](https://kingy.ai/blog/claude-fable-5-1-benchmarks-price-mythos-access) |
| SWE-bench Multimodal | 54.7% | — | **59.4%** | — | [BenchLM](https://benchlm.ai/models/claude-fable-5-1) |
| ARC-AGI-2 (Verified) | **90.0%** @ $3.12/task | 89.2% @ $5.45 | — | — | [ARC Prize](https://x.com/arcprize/status/2094894027451539744) |
| ARC-AGI-1 (Verified) | 97.5% @ $1.40/task | 98.5% @ $1.02 | — | — | [ARC Prize](https://x.com/arcprize/status/2094894027451539744) |
| AA Intelligence Index v4.3 | **53** | 50 | 51 | 44–47 | [Artificial Analysis](https://artificialanalysis.ai/articles/artificial-analysis-intelligence-index-v4-3) |

### Five caveats that define the attack surface

1. **The 95.0% SWE-bench Verified figure is Fable 5's, not Fable 5.1's.** Anthropic's 5.1 announcement does not headline a SWE-bench Verified score. Treat any Fable 5.1 SWE-bench Verified claim as **UNVERIFIED** ([emergent.sh](https://emergent.sh/learn/claude-fable-5-1-benchmarks)).
2. **Fable 5.1 is already not SOTA on several benchmarks** — Toolathlon Verified (77.8% vs Opus 5's 80.6% and Mythos 5's 79.3%), SWE Multilingual (89.1% vs 89.5%), SWE-bench Multimodal (54.7% vs 59.4%), Terminal-Bench 2.1 (85.02% vs GPT-5.6 Sol's 85.77%).
3. **Anthropic's numbers run with production safety classifiers on while comparison models run without.** On Toolathlon, 11 of 324 Fable 5.1 trials hit a safety refusal and were partly completed by a fallback model; four more terminated and counted as failures ([Kingy](https://kingy.ai/blog/claude-fable-5-1-benchmarks-price-mythos-access)). Several published rows measure a shipped *system*, not raw capability.
4. **The agentic scores are low in absolute terms** — AutomationBench 31.4%, OSWorld-strict 41.7%, Terminal-Bench-Science 52.6%, Terminal-Bench 4.0 55.8%. 45–70 points of headroom.
5. **The saturated rows are not targets.** GPQA 92.6%, LiveCodeBench 90.5%, ProofBench 100%, ARC-AGI-1 97.5% have no meaningful room — except that on GPQA an open model has already edged past (§C.1).

---

## Part A — The open-weight landscape, September 2026

### A.1 The headline gap

**Artificial Analysis Intelligence Index v4.3** (published 2026-09-07; components AA-Briefcase, GDPval-AA v2, AutomationBench-AA, Terminal-Bench v4.0, SciCode, HLE, GDP.pdf, CritPt, AA-Omniscience, AA-LCR v1.1; weights Agents 30% / Coding 20% / General 30% / Scientific Reasoning 20%; private-test-set weighting raised 40%→45% to resist gaming) ([Artificial Analysis](https://artificialanalysis.ai/articles/artificial-analysis-intelligence-index-v4-3)).

| Rank | Model | Index | Weights |
|---|---|---|---|
| 1–4 | **Claude Fable 5.1** (max/xhigh), **GPT-6 Astra** (max/xhigh) | **53** | Closed |
| 7 | Claude Opus 5 (max) | 51 | Closed |
| 9 | Claude Fable 5 | 50 | Closed |
| 13 | Muse Spark 1.3 (max) — Meta | 48 | Closed |
| **19** | **GLM-5.3 (max)** | **45** | **OPEN — best open model** |
| 20–22 | Grok 4.6, GPT-5.6 Sol | 44 | Closed |
| **23** | **Kimi K3 (max)** | **44** | **OPEN** |
| **27** | **GLM-5.3-Flash** | **42** | **OPEN** |
| **30** | **Qwen3.8 2.4T A95B** | **40** | **OPEN** |
| — | **DeepSeek V4.1 Flash** | **40** | **OPEN** |

Sources: [AA open-source leaderboard](https://artificialanalysis.ai/models/open-source), [AA leaderboard](https://artificialanalysis.ai/leaderboards/models), [AA on X 2026-09-07](https://x.com/ArtificialAnlys/status/2097025645889069094).

**The gap is 8 points (45 vs 53) — roughly one model generation.** All top-5 open models are Chinese.

At equal index scores of 53, GPT-6 Astra costs $3.26/task vs Fable 5.1's $7.63/task — 57% less ([Artificial Analysis](https://artificialanalysis.ai/articles/artificial-analysis-intelligence-index-v4-3)).

> **Conflict flagged.** [BenchLM](https://benchlm.ai/benchmarks/artificialanalysis) reports a different AA leaderboard (GPT-5.6 Sol 58.9%, DeepSeek V4 Pro labelled *closed*). This contradicts AA's own publication on both scale and open/closed labelling. **Use the AA primary source; treat BenchLM's AA rows as UNVERIFIED.**

> **GLM-5.3 weights status — RESOLVED.** Z.ai announced GLM-5.3 on 2026-08-14 and slipped its own 2026-08-28 open-weight target; **weights landed ~2026-09-04** ([HF](https://huggingface.co/zai-org/GLM-5.3), [aireleasetracker](https://aireleasetracker.com/model/zai/glm-5.3)). "Announced open" ≠ "downloadable" is a recurring 2026 pattern worth tracking.

### A.2 How many months does open lag closed?

| Source | Method | Lag | Point gap | Date |
|---|---|---|---|---|
| [Epoch AI](https://epoch.ai/data-insights/open-weights-vs-closed-weights-models) | ECI, horizontal distance | **3.5 months** (90% CI 1.1–5.3) | 7 ECI (CI 0–14) | Pub. 2025-10-30, covering 2023-01→2025-10 |
| [Epoch AI](https://epoch.ai/data-insights/open-closed-eci-gap) | ECI, re-measured | **4 months** | 8 ECI (≈ the GPT-5 → GPT-5.5 gap) | Pub. 2026-05-29, covering since 2026-01 |
| [SemiAnalysis](https://newsletter.semianalysis.com/p/are-open-models-catching-up) | Era-by-era catch-up, run on Prime Intellect's eval stack | **~4.8–6 months** (agentic era) | — | 2026 |
| [Artificial Analysis](https://artificialanalysis.ai/models/open-source) | AA Index v4.3 | — | **8 points** | 2026-09-07 |

Epoch's May-2026 top-of-class: highest closed = **GPT-5.5 Pro (xhigh), ECI 159.35** (CI 155.79–167.47); highest open = **Kimi K2.6, ECI 151.60** (CI 148.44–159.28). **Note the confidence intervals overlap** — the gap is real but not large relative to measurement noise ([Epoch](https://epoch.ai/data-insights/open-closed-eci-gap)).

**SemiAnalysis's era decomposition is the most useful framing — the lag halves each era** ([SemiAnalysis](https://newsletter.semianalysis.com/p/are-open-models-catching-up)):

| Era | Closed anchor → open catch-up | Initial gap | Time to close |
|---|---|---|---|
| Era 1 — Early scaling (2022–24) | GPT-3.5 Turbo (75.7) → Llama-2-70B (39.9); closed by Llama-3.1-405B | 35.8 pts | **~18 months** |
| Era 2 — Reasoning (2024–25) | o1-preview → DeepSeek R1; closed by R1-0528 | 12.1 pts | **~8.5 months** |
| Era 3 — Agentic (2025–now) | Opus 4.5 → **Kimi K2.6 (56.3) in 4.8 months**; GPT-5.2 → **GLM-5.2 (72.4) in 6 months** | — | **~5–6 months** |

Their conclusion: *"open-source models take half as long to catch up to the first closed-source model of the era."*

**The caveat all sources share:** the lag understates the true distance, because open models hill-climb public benchmarks while closed labs hold their strongest systems back. **Anthropic's Claude Mythos is the sharpest example — the first publicly acknowledged flagship deliberately not released, restricted to ~50 partner orgs via Project Glasswing** ([VLM Overview report 2026-04-28](https://github.com/zli12321/Vision-Language-Models-Overview/blob/main/progressive%20reports/2026-04-28.md)). Mythos 5.1 scores 60.9% on Terminal-Bench 4.0 vs Fable 5.1's 55.8% — the *public* frontier is already a notch below the real one.

**Synthesis: open lags closed by ~4–6 months and ~8 index/ECI points as of Sept 2026, compressing by about half per capability era.**

### A.3 Chinese frontier open-weight models (the actual leaders)

| Model | Released | Total / Active | Architecture | Ctx | License | Train tokens | Source |
|---|---|---|---|---|---|---|---|
| **Kimi K3** (Moonshot) | Jul 17 2026 (weights Jul 27) | **2.8T / 104B** | MoE 896 experts, 16 sel + 2 shared; **69 Kimi Delta Attention + 24 Gated MLA** layers; Attention Residuals; MXFP4 wts / MXFP8 act; MoonViT-V2 (401M) | 1M | Kimi K3 License | UNVERIFIED | [kimi.ai](https://www.kimi.ai/blog/kimi-k3), [HF](https://huggingface.co/moonshotai/Kimi-K3) |
| **DeepSeek-V4-Pro-0813** | Aug 2026 (V4 preview Apr 24) | **1.6T / 49B** | MoE; hybrid **Compressed Sparse Attention + Heavily Compressed Attention**; **mHC** (Manifold-Constrained Hyper-Connections); Muon | 1M | **MIT** | **>32T** | [arXiv 2606.19348](https://arxiv.org/abs/2606.19348), [HF](https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro-0813) |
| **DeepSeek-V4.1-Flash** | ~Sept 2026 | 552B backbone / **8B prefill, 16B decode** | **Causal Encoder-Decoder (CED)**; **CSA2** (Full/Reindex/Reuse); 384 routed + 1 shared expert, 6 active | 1M | **MIT** | **45T + 34T ctx-ext** | [HF](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash) |
| **DeepSeek-V4-Flash-0731** | Jul 31 2026 | 284B / 13B | same family; DSpark speculative decoding; FP8 KV cache | 1M (384K out) | **MIT** | >32T | [HF](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731) |
| **GLM-5.3** (Z.ai) | Aug 14 2026 (weights ~Sep 4) | **743–753B / 40B** | Post-training-only upgrade on GLM-5.2 base; MoE + **DSA**; reasoning_effort low/high/max | 1M | GLM-5.3 (custom) | inherits 28.5T | [HF](https://huggingface.co/zai-org/GLM-5.3) |
| **GLM-5.3-Flash** | Aug 26 2026 | 320B / 18B | Hybrid **sparse + linear attention**; **mHC**; multimodal-native | 300K | **MIT** | 30T multimodal | [HF](https://huggingface.co/zai-org/GLM-5.3-Flash) |
| **GLM-5** | Feb 17 2026 | 744B / 40B | MoE + DeepSeek Sparse Attention | n/d | **MIT** | 28.5T | [HF](https://huggingface.co/zai-org/GLM-5) |
| **Qwen3.8-2.4T-A95B** | Aug 3–12 2026 | **2.4T / 95B** | 512 experts, 10 routed + 1 shared; **23 × (3× Gated DeltaNet→MoE + 1× Gated Attention→MoE)** | 262K → 1.01M | qwen3.8-max (custom) | n/d | [HF](https://huggingface.co/Qwen/Qwen3.8-2.4T-A95B) |
| **Qwen3.5-397B-A17B** | Feb 16 2026 | 397B / 17B | 60 layers, 3:1 Gated DeltaNet : Gated Attention; 512 exp / 11 active; early-fusion multimodal | 262K → 1.01M | **Apache 2.0** | n/d | [HF](https://huggingface.co/Qwen/Qwen3.5-397B-A17B) |
| **Qwen3-Next-80B-A3B** | Sept 2025 | 80B / 3B | 48 layers, Gated DeltaNet 3:1, 512 exp / 10+1, MTP | 262K → 1.01M | **Apache 2.0** | 15T | [HF](https://huggingface.co/Qwen/Qwen3-Next-80B-A3B-Instruct) |
| **Kimi K2.5** | Jan 27 2026 | 1T / 32B | MoE 384 exp, 8 sel + 1 shared; **MLA**; 61 layers; MoonViT 400M | 256K | Modified MIT | ~15T | [GitHub](https://github.com/MoonshotAI/Kimi-K2.5) |
| **Kimi K2-Thinking** | Nov 2025 | 1T / 32B | MoE 384/8/1; MLA; **native INT4 QAT** (~2× gen speed) | 256K | Modified MIT | n/d | [HF](https://huggingface.co/moonshotai/Kimi-K2-Thinking) |
| **MiniMax-M3** | ~Jul 2026 | 428B / 23B | MoE; **MiniMax Sparse Attention**; native text/image/video | 1M | minimax-community | n/d | [HF](https://huggingface.co/MiniMaxAI/MiniMax-M3) |
| **MiniMax-M2.5 / M2.7** | Feb 11 / Apr 2026 | 229B / 10B | MoE; CISPO RL | 128K+ | Modified MIT / "other" | n/d | [M2.5](https://huggingface.co/MiniMaxAI/MiniMax-M2.5), [M2.7](https://huggingface.co/MiniMaxAI/MiniMax-M2.7) |
| **Tencent Hy4-preview** | ~Aug 2026 | 770B / 49B | 78 layers; 256 routed + 1 shared, top-8; **Gated DSA with IndexCache** | 1M | **Apache 2.0** | n/d | [HF](https://huggingface.co/tencent/Hy4-preview) |
| **Tencent Hy3** | Jul 2026 | 295B / 21B | MoE 192 exp top-8; GQA 64Q/8KV; 3.8B MTP layer | 256K | **Apache 2.0** | n/d | [HF](https://huggingface.co/tencent/Hy3) |
| **Xiaomi MiMo-V2.5-Pro** | Jul 2026 | **1.02T / 42B** | 384 exp top-8; **SWA(128)+Global 6:1**; 3-layer MTP | 1M | **MIT** | **27T, FP8** | [HF](https://huggingface.co/XiaomiMiMo/MiMo-V2.5-Pro) |
| **InclusionAI Ling-3.0-flash** | ~Sept 2026 | 124B / **5.1B** | "Native Hybrid-Linear": **35 KDA + 7 Gated MLA (5:1)**; 512 exp / 8+1; 8K→32K→256K | 256K | **MIT** | n/d | [HF](https://huggingface.co/inclusionAI/Ling-3.0-flash) |
| **StepFun Step-3.5-Flash / Step-3.7** | Feb–Jun 2026 | 199B / 198B | successors to step3 (321B, MFA attention) | n/d | n/d | n/d | [HF](https://huggingface.co/stepfun-ai/models?sort=modified) |
| **Meituan LongCat-Flash-Omni** | Nov 11 2025 | 561B | zero-computation experts; **nothing newer since** | n/d | n/d | n/d | [HF](https://huggingface.co/meituan-longcat/models?sort=modified) |
| **ByteDance Seed-OSS-36B** | Aug 26 2025 | 36B | **still ByteDance's newest open LLM** | n/d | n/d | n/d | [HF](https://huggingface.co/ByteDance-Seed/models?sort=modified) |
| **Baidu ERNIE-4.5-300B-A47B** | Nov 2025 | 300B / 47B | newest open ERNIE (ERNIE 5.0 is **not** open) | n/d | n/d | n/d | [HF](https://huggingface.co/baidu/models?sort=modified) |

#### Benchmarks (vendor-reported unless noted)

| Model | GPQA-D | HLE | AIME | SWE-bench Verified | SWE-bench Pro | DeepSWE v1.1 | Terminal-Bench | tau2 | Other |
|---|---|---|---|---|---|---|---|---|---|
| **Kimi K3** | **93.5** | — | — | — | — | 67.3–67.5 | 88.3 (TB 2.1) | — | **BrowseComp 90.4–91.2**; Video-MME 90.0 |
| DeepSeek-V4.1-Flash | **90.9** | — | — | — | — | **74.2** | **90.6** (TB 2.1) | — | GSM8K 93.0 (base) |
| DeepSeek-V4-Pro-0813 | — | **60.0** (tools) | — | — | — | 62.7 | 87.9 (TB 2.1) | — | Cybergym 83.3; NL2Repo 61.5; Toolathlon-V 74.1; Agents' Last Exam 25.7 |
| DeepSeek-V4-Flash-0731 | — | — | — | — | — | 54.4 | 82.7 (TB 2.1) | — | NL2Repo 54.2; Toolathlon-V 70.3 |
| **GLM-5.3** | — | **62.5** (tools) | — | — | — | 66.9 | 88.2 (TB2.1) / **28.3 (TB 3.0)** | — | CyberGym 84.5; **GDPval-AA v2 1769** (AA-run); FrontierSWE 78.1 |
| GLM-5 | 86.0 | 30.5 / 50.4 tools | 92.7 (AIME 2026 I) | 77.8 | — | — | 56.2–60.7 (TB 2.0) | **89.7** | BrowseComp 62.0→75.9; Vending Bench 2 $4,432 |
| **Qwen3.8-2.4T-A95B** | **92.6** | 43.6 | — | — | 67.7 | 56.6 | 86.6 (TB 2.1) | — | — |
| Qwen3.5-397B-A17B | 88.4 | 37.6 | **91.3 (AIME 2026)** | 76.4 | — | — | 52.5 (TB 2) | **86.7** | LCB v6 83.6; MMLU-Pro 87.8 |
| **Qwen3.8-27B** (dense) | 89.2 | — | — | — | 61.7 | — | 73.0 (TB 2.1) | — | **LCB v6 90.3**; OSWorld-V 84.3 |
| Kimi K2.5 | 87.6 | 50.2 (tools) | **96.1** ('25) | 76.8 | — | — | 50.8 (TB 2.0) | — | LCB v6 85.0; BrowseComp 78.4 (swarm) |
| Kimi K2-Thinking | 84.5 | 44.9 (tools) | **99.1** ('25, +py) | 71.3 | — | — | 47.1 | — | LCB v6 83.1; BrowseComp 60.2 |
| MiniMax-M3 | — | — | — | **80.5** | 59 | — | LHTB 38.5 | — | Apex Agents 27.7 |
| MiniMax-M2.5 | 85.2 | — | 86.3 ('25) | 80.2 | 55.4 | — | — | — | BrowseComp 76.3; SciCode 44.4 |
| MiniMax-M2 | 78 | 12.5 (no tools) | 78 ('25) | 69.4 | — | — | 46.3 | 77.2 | LCB 83 |
| Hy4-preview | 92.3 | — | — | — | 65.7 | 64.3 | 85.4 (TB 2.1) | — | Apex Agents 37.1; SWE-M/L 82.9 |
| Hy3 | 90.4 | — | — | — | 57.9 | — | — | — | SWE-M/L 75.8 |
| Ling-3.0-flash | — | 22.7 | **93.2 (AIME 2026)** | — | 56.6 | — | — | — | — |

**Head-to-head vs Fable 5.1 — where open already wins or ties:**

| Benchmark | Fable 5.1 | Best open | Verdict |
|---|---|---|---|
| **GPQA Diamond** | 92.6 | **Kimi K3 93.5** | **Open wins** (both vendor-reported) |
| GPQA Diamond | 92.6 | Qwen3.8 92.6 | **Tie** |
| LiveCodeBench | 90.52 | Qwen3.8-27B 90.3 (LCB v6) | Within noise; versions differ |
| **BrowseComp** | not published | **Kimi K3 90.4–91.2** (Opus 5 = 90.8) | Open beats Opus 5; no Fable 5.1 figure |
| HLE (tools) | 65.0 | GLM-5.3 62.5 | Closed wins by 2.5 |
| GDPval-AA v2 | 1853 | GLM-5.3 1769 (AA-run) | Closed wins by 84 Elo |
| Terminal-Bench 4.0 | 55.8 | DeepSeek-V4.1-Flash 31.2 | Closed wins by 24.6 |

### A.4 Western open-weight models

| Model | Released | Total / Active | Architecture | Ctx | License | Compute / tokens | Source |
|---|---|---|---|---|---|---|---|
| **Meta Muse Glimmer 30B** | **Aug 10 2026** | ~30B dense (incl. ~1.8B ViT-G/14) | **Distilled from closed Muse Spark**; GQA 32Q/2KV; [Local,Local,Local,Global] w/ 2048 sliding window; RoPE θ=500k local-only; ~4-bit + block spec-decoding | 131K+ | **Apache 2.0** | n/d | [research.meta.ai](https://research.meta.ai/blog/introducing-muse-glimmer-open-agentic-model), [HF](https://huggingface.co/meta-models/Muse-Glimmer-30B) |
| **Llama 4 Maverick** | Apr 5 2025 | 400B / 17B, 128 experts | MoE, early-fusion multimodal, iRoPE | 1M | Llama 4 Community | **2.38M H100-hrs**, 645 tCO₂eq, ~22T tokens | [HF](https://huggingface.co/meta-llama/Llama-4-Maverick-17B-128E-Instruct) |
| **Mistral Large 3** (675B-Instruct-2512) | Dec 2025 | 675B / 41B | Granular MoE + 2.5B vision encoder | 256K | **Apache 2.0** | n/d | [HF](https://huggingface.co/mistralai/Mistral-Large-3-675B-Instruct-2512) |
| **Mistral Small 4** (119B-2603) | Mar 2026 | 119B / **6.5B**, 128 exp / 4 active | MoE | 256K | **Apache 2.0** | n/d | [HF](https://huggingface.co/mistralai/Mistral-Small-4-119B-2603) |
| **Gemma 4** (31B-it; 12B, E4B, E2B) | ~Jul 2 2026 | 30.7B dense + ~550M vision | Hybrid local sliding-window (1024) + global attention; multimodal | **256K** | **Apache 2.0** (major shift off Gemma Terms) | n/d | [HF](https://huggingface.co/google/gemma-4-31b-it), arXiv 2607.02770 |
| **gpt-oss-120b / 20b** | Aug 5 2025 | 117B / **5.1B**; 21B / 3.6B | MoE, **attention sinks**, MXFP4 MoE wts, single-80GB-GPU | 128K | **Apache 2.0** | not disclosed | [HF](https://huggingface.co/openai/gpt-oss-120b), arXiv 2508.10925 |
| **IBM Granite 4.2** (3b/8b/30b) | ~Sept 2026 | 4B / 9B / 29B | Hybrid Mamba-2 + transformer | n/d | Apache 2.0 | n/d | [HF](https://huggingface.co/ibm-granite/models?sort=modified) |
| **Falcon-H1R-7B** | Jan 2026 | 8B | Hybrid Mamba-transformer reasoning | n/d | Falcon LLM license | n/d | [HF](https://huggingface.co/tiiuae/models?sort=modified) |

**Muse Glimmer benchmarks:** MCP-Atlas **75.5** (Gemma4-31B 54.2, Qwen3.6-27B 62.5), DeepSearch QA 74.6, Gaia2 43.3, SWE-Bench Pro 51.2, **AIME 2026 94.7**, IFBench 77.0, AA-LCR 80.0. Qwen3.6-27B still beats it on OSWorld-V (75.6 vs 65.9), TB 2.1 (60.7) and SWE-bench Verified (77.2) ([marktechpost](https://www.marktechpost.com/2026/08/10/meta-ai-releases-muse-glimmer), [InfoQ](https://www.infoq.com/news/2026/08/meta-muse-glimmer)).
**gpt-oss-120b (AA-measured):** GPQA-D 80.81, MMLU-Pro 80.8, **SWE-bench Pro 16.2** — now far off the frontier.
**Llama 4 Maverick:** MMLU-Pro 80.5, GPQA-D 69.8, LiveCodeBench 43.4.

### A.5 Names that DO NOT EXIST — do not cite these

| Claimed | Reality |
|---|---|
| **Llama 5** | **Does not exist.** Meta shipped closed **Muse Spark** (Apr 2026) then returned to open with **Muse Glimmer 30B** (Aug 10 2026, Apache 2.0). Llama 4 Scout/Maverick (Apr 2025) remain the last Llama-branded open models; Behemoth never shipped. |
| **DeepSeek R2** | **Never shipped as a distinct model.** The reasoning line folded into V3.x/V4 with `reasoning_effort` levels. |
| **Qwen 4** | **Does not exist.** Line ran Qwen3 → Qwen3-Next → Qwen3.5 (Feb 2026) → Qwen3.6 (Apr) → Qwen3.8 (Aug). |
| **gpt-oss-2** | **Does not exist.** Only gpt-oss-120b/20b + gpt-oss-safeguard-120b/20b. |
| **Phi-5** | **Does not exist.** Newest is phi-4 / Phi-4-reasoning-vision-15B. |
| **Falcon-H2** | **Does not exist.** Falcon-H1R-7B (Jan 2026) is newest. |
| **OLMo 3.5 / OLMo 4** | **Do not exist.** OLMo 3 + OLMo 3.1 only. |
| **SmolLM3.5 / SmolLM4** | **Do not exist.** |
| **Granite 5** | **Does not exist.** Granite 4.2 is current. |
| **ERNIE 5.0 open weights** | **Not open.** Newest open Baidu model is ERNIE-4.5. |
| Qwen3-Max | Was API-only; **Qwen3.8-Max IS open** as `Qwen3.8-2.4T-A95B` — first Max-class Qwen with published weights. |

### A.6 Fully open (weights + data + code + recipe) vs open-weight-only

| Project | Released | Size | Openness | Source |
|---|---|---|---|---|
| **AI2 OLMo 3 / 3.1** | Nov 20 / Dec 12 2025 | 7B & 32B (Base/Instruct/Think/RL-Zero) | **FULL** — weights, all intermediate checkpoints, **Dolma 3** (9.3T corpus; 5.9T pretrain mix; Dolmino 100B mid-train; Longmino ~50B long-ctx), **Dolci** post-train suite, code (OLMo-core, Open-Instruct, datamap-rs, duplodocus, OLMES, decon), OLMoTrace. License **CC-BY-4.0** | [allenai.org/blog/olmo3](https://allenai.org/blog/olmo3), [arXiv 2512.13961](https://arxiv.org/pdf/2512.13961) |
| **Stanford Marin** | 8B (2025), **32B "Bison"** (Dec 2025), 535B-A23B MoE in progress | 8B/32B/535B | **FULL + "open lab"** — experiments declared as GitHub Issues, configs as PRs, live W&B metrics, **all failures logged**. JAX/Levanter on TPU; 535B moving to GB200 NVL72; Muon | [Google OSS blog](https://opensource.googleblog.com/2025/12/training-marin-32b-what-an-open-lab-can-build-with-tpus-jax-and-a-little-persistence.html), [HF](https://huggingface.co/marin-community/marin-32b-base) |
| **Apertus** (Swiss AI / ETH+EPFL+CSCS) | Sept 2025 | 8B & 70B | **FULL** — weights, open+compliant data, recipes, data-reconstruction method; 1000+ languages; **15T tokens on >4,000 GH200** (Alps, carbon-neutral) | [HF](https://huggingface.co/swiss-ai/Apertus-70B-2509), [AWS](https://aws.amazon.com/blogs/alps/switzerlands-open-source-apertus-llms-now-available-on-amazon-sagemaker-ai) |
| **MBZUAI/IFM K2-Think** (+ V2) | Sept 2025 | 32B | **FULL** — weights, training data, SFT code, inference + test-time-compute code. V2 adds full recipe incl. failures | [mbzuai](https://mbzuai.ac.ae/news/mbzuai-and-g42-launch-k2-think-a-leading-open-source-system-for-advanced-ai-reasoning), [HF](https://huggingface.co/IFM/K2-Think) |
| **LLM360 K2-65B** | 2024 | 65B | FULL — "world's first fully reproducible open-source foundation model" | same |
| **NVIDIA Nemotron 3** — Nano 30B-A3B (Dec 15 2025), Super 120B-A12B (Mar 11 2026), **Ultra 550B/55B** (GTC 2026) | | | **FULL-ish** — open weights + training data + recipes, caveat *"all data for which we hold redistribution rights"*. **NVIDIA Nemotron Open Model License** (permissive, commercial OK). Arch: MoE **hybrid Mamba-2 + Transformer** (Nano: 23 Mamba-2/MoE + 6 attention layers, 128 exp + 1 shared, 6 active, 1M ctx); Super/Ultra trained in **NVFP4** with LatentMoE + MTP | [arXiv 2512.20856](https://huggingface.co/papers/2512.20856), [Super card](https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16) |
| **HF SmolLM3-3B** | Jul–Sept 2025 | 3B | FULL (weights + data + recipe; FineWeb lineage) | [HF](https://huggingface.co/HuggingFaceTB/SmolLM3-3B) |
| Everything in §A.3 / §A.4 | | | **OPEN-WEIGHT ONLY** — no pretraining data, no training code | model cards |

RedMonk's longitudinal read: *"there is effectively no advantage to restricted open weight vs fully open models"* on capability, and *"while open weight models lead the way performance-wise, fully open models follow quickly after"* ([RedMonk](https://redmonk.com/sogrady/2026/05/15/open-ai-models)).

### A.7 Licensing

Of 178 Chinese open models above 20B params, **59% carry Apache 2.0 and 22% carry MIT**, with almost none non-commercial — though this began shifting in late summer 2026 for the largest models. American labs used custom terms (41%) and undeclared licenses (30%) more often ([Hugging Face](https://huggingface.co/blog/state-of-open-models-summer-2026)).

| License | Models | Restriction |
|---|---|---|
| **Apache 2.0** | Qwen 3/3.5/3.6 (Qwen3.8-Max uses a custom `qwen3.8-max` license), gpt-oss, Gemma 4, Mistral Large 3 / Small 4, Granite 4, Tencent Hy3/Hy4, **Muse Glimmer** | None (gpt-oss adds a usage policy) |
| **MIT** | DeepSeek R1 / V3.2 / V4-Pro / V4-Flash / V4.1-Flash, GLM-5 / 5.2 / GLM-5.3-Flash, Phi-4, MiMo-V2.5-Pro, Ling-3.0-flash | None |
| **Modified MIT** | Kimi K2 / K2.5 / K2.6 (K3 uses a distinct "Kimi K3 License"), MiniMax M2/M2.5 | Attribution / name display for very large commercial deployments |
| **Custom** | GLM-5.3, Qwen3.8-Max, MiniMax-M3 (minimax-community), NVIDIA Nemotron Open Model License | Read the text |
| **Llama Community** | Llama 4 Maverick | 700M MAU cap, naming rules, EU multimodal exclusion |
| **Non-commercial** | Command R+ (CC-BY-NC); some Grok variants forbid using weights to train other models | Research only |

Sources: [tech-insider](https://tech-insider.org/best-open-source-llm-2026), [computingforgeeks](https://computingforgeeks.com/open-source-llm-comparison), [DataNorth](https://datanorth.ai/blog/the-best-open-source-llm-in-2026), [Konishi timeline](https://hidekazu-konishi.com/entry/open_weights_llm_release_history_and_timeline.html), plus the HF cards above.

### A.8 Three measurement traps

1. **Terminal-Bench version drift.** TB 2.0 ≈ 47–61 for top models; **TB 2.1 ≈ 82–90**; **TB 3.0 ≈ 28.3** for the best open model (GLM-5.3); AA v4.3 uses **TB v4.0** (Fable 5.1 = 55.8). **Numbers are non-comparable across versions — always cite the version.**
2. **SWE-bench Verified is saturated** (~77–80% open) and largely superseded by **SWE-bench Pro** (55–68%), **DeepSWE v1.1** (54–74%), NL2Repo, FrontierSWE, Toolathlon, AutomationBench, Agents' Last Exam. Several 2026 cards no longer report Verified at all.
3. **Almost every benchmark in §A.3 is vendor-reported.** The independent datapoints are Artificial Analysis's Intelligence Index and its GDPval-AA v2 run (GLM-5.3 = 1769, ahead of Fable 5 at 1743 and GPT-5.6 Sol at 1730).

### A.9 Adoption reality check

Attention and usage are decoupled. Among models declaring parameter counts, **models under 1B take 83% of all-time Hugging Face downloads; everything above 100B takes 1%.** In 2026 only 3% of downloads went above 70B. The top-25 most-liked and top-25 most-downloaded repos share exactly **one** model. Qwen is the substrate: **151,448 derivatives**, 2.6× Meta's total footprint, growing 180–210 repos/day ([Hugging Face](https://huggingface.co/blog/state-of-open-models-summer-2026)).

**Implication:** building *on* Qwen maximizes ecosystem gravity; building something *under 1B* maximizes downloads.

---

## Part B — Compute and dollar cost

### B.1 Disclosed training compute and cost

#### Tier 1 — Hobby / reproduction

| Run | Hardware | Wall-clock | Cost | Result | Source |
|---|---|---|---|---|---|
| **nanochat record (Run 6, 2026-03-14)** | 8×H100 | **1.65 h** | ~$40 on-demand @$3/GPU-h; **~$20 spot** | CORE **0.2626** — beats GPT-2 (1.6B) CORE 0.256525 | [LEADERBOARD.md](https://raw.githubusercontent.com/karpathy/nanochat/master/dev/LEADERBOARD.md) |
| nanochat Run 5 (2026-03-09) "autonomous research via autoresearch" | 8×H100 | ~2.0 h | — | CORE **0.2690** (highest on board) | same |
| nanochat Run 4 (2026-03-03) ClimbMix dataset swap | 8×H100 | ~2.02–2.3 h | — | CORE 0.25714 (7-trial avg) | same |
| nanochat Run 3 (2026-02-05) 1M-token batch | 8×H100 | 2.76 h | ~$72 on-demand | CORE 0.26024 | same |
| nanochat Run 2 (2026-02-02) fp8 via torchao | 8×H100 | 2.91 h | — | CORE 0.2578 | same |
| nanochat Run 1 (2026-01-29) d24 baseline | 8×H100 | 3.04 h | — | CORE 0.25851 | same |
| **nanochat headline** | 8×H100 | ~2 h | **$48** | GPT-2-capability model that cost ~$43,000 to train in 2019 | [README](https://github.com/karpathy/nanochat) |

**Time-to-GPT-2 fell 3.04 h → 1.65 h in seven weeks**; vs the 2019 baseline, training time dropped **98.8% over seven years** (168 h → ~2 h). Leaderboard rule: submissions must exceed GPT-2's CORE and demonstrate generalizability across depths with principled, non-esoteric improvements.

> **UNVERIFIED:** nanochat $300/$1000 tiers (README documents only the $100 speedrun); llm.c GPT-2 reproduction costs ($20 / $672); current modded-nanogpt speedrun record. Not confirmed within this session's search budget.

#### Tier 2 — Small open models with published recipes

| Model | Params | Tokens | GPU-hours | Implied MFU | Est. cost @$0.94 spot / $2.43 on-dem | Source |
|---|---|---|---|---|---|---|
| **SmolLM3** | 3B | **11.2T** | **220k H100-h** (384 H100s × 24 d) | **25.7%** | **$207k / $535k** | [HF blog](https://huggingface.co/blog/smollm3) |
| **Apertus** | 8B & 70B | **15T** | **>4,000 GH200** (Alps); hours n/d | — | — | [HF](https://huggingface.co/swiss-ai/Apertus-70B-2509) |
| **OLMo 3** | 7B / 32B | n/d | "up to 1,024 H100s"; total GPU-hours **UNVERIFIED** | — | — | [allenai.org](https://allenai.org/blog/olmo3) |

#### Tier 3 — Large open models

| Model | Params | Tokens | GPU-hours | Disclosed cost | Implied MFU | Source |
|---|---|---|---|---|---|---|
| **DeepSeek-V3** | 671B / 37B | 14.8T | **2.788M H800-h** | **$5.576M** @ assumed $2/H800-h — *official run only, excludes prior research/ablations* | **33.1%** | [arXiv 2412.19437](https://arxiv.org/pdf/2412.19437) |
| **DeepSeek-R1** (RL only) | on V3-Base | — | R1-Zero: 648 H800 × ~198 h = **128k**; R1: 648 × ~80 h = **52k**; SFT data ~5,000 | **~$294,000** | — | [Nature s41586-025-09422-z](https://www.nature.com/articles/s41586-025-09422-z) (paywalled); figures via [HyperAI](https://hyper.ai/en/news/44332) |
| **Llama 3.1 405B** | 405B dense | 15T+ | **30.84M H100-h** (family 39.3M: 1.46M/8B, 7.0M/70B) | — | **34.6%** | [HF](https://huggingface.co/blog/llama31) |
| Llama 3.1 405B (Epoch) | — | — | 16k H100s × 380 TFLOP/s × 6.25M s | **3.8e25 FLOP, ~72 days** | 38.4% | [Epoch on X](https://x.com/EpochAIResearch/status/1815778832265232483) |
| **Llama 4 Maverick** | 400B / 17B | ~22T | **2.38M H100-h**, 645 tCO₂eq | — | — | [HF](https://huggingface.co/meta-llama/Llama-4-Maverick-17B-128E-Instruct) |

**The R1 number is the single most important datapoint here for a small team.** $294k bought a reasoning model that redefined the field — but only the RL stage on top of a ~$6M base. DeepSeek explicitly notes the base investment was separate and much larger.

> **No Chinese frontier lab disclosed GPU-hours or FLOPs for any 2026 model** (Kimi K3, GLM-5.3, Qwen3.8, DeepSeek V4). **All UNVERIFIED.** Anthropic has disclosed nothing for Fable 5 / 5.1 / Opus 5.

#### Tier 4 — Frontier estimates

| Quantity | Value | Source |
|---|---|---|
| Growth rate, amortized hardware + energy of frontier runs | **2.4×/year** since 2016 (95% CI 2.0–3.1×); cloud method 2.6×/yr | [Epoch AI](https://epoch.ai/blog/how-much-does-it-cost-to-train-frontier-ai-models) |
| Cost composition | Hardware **47–67%**; R&D staff **29–49%**; energy **2–6%** | same |
| Projection | *"the largest training runs will cost more than a billion dollars by 2027"* | same |
| Power | Gemini Ultra ≈ **35 MW**; gigawatt-scale projected by 2029 | same |
| GPT-4 (2023) | ~2e25 FLOP, ~25,000 A100s | [Deluair](https://deluair.com/consultancy/insights/frontier-ai-training-cost-2026) — **UNVERIFIED secondary** |
| Frontier 2026 runs | **1e26 – 1e27 FLOP**; ~$200–500M/run (GPT-5 / Gemini Ultra class) | same — **UNVERIFIED** |
| GPT-5 total program (not single run) | $1.5B–$3B | same — **UNVERIFIED** |
| Grok-3 cluster ("Colossus Memphis Phase 1") | ~$4B capex | same — **UNVERIFIED** |

**Note:** Epoch's cost article does *not* publish per-model dollar figures — it presents aggregated trends across 45 frontier models. Per-model 2026 cost claims on secondary sites are **not** Epoch-sourced.

### B.2 GPU rental prices, September 2026

All $/GPU-hour USD, checked 2026-09-13.

#### H100 SXM 80GB

| Provider | $/GPU-h | Mode | Source |
|---|---|---|---|
| **Prime Intellect** | **$0.94** | spot | [primeintellect.ai](https://primeintellect.ai) |
| Vast.ai | $1.49–$2.27 | marketplace/interruptible | [spheron](https://www.spheron.network/blog/vastai-pricing-2026) — **UNVERIFIED 3rd-party**; Vast's own floor quoted "from $1.60" |
| RunPod (PCIe) | $1.99 | Community | [runpod.io/pricing](https://www.runpod.io/pricing) |
| **SF Compute** | **$2.03 avg** | **spot market clearing, Aug 15–Sep 11 2026** | [sfcompute.com/prices](https://sfcompute.com/prices) |
| Nebius | $2.15 | preemptible | [nebius.com/prices](https://nebius.com/prices) |
| Prime Intellect | $2.43 | on-demand | primeintellect.ai |
| CoreWeave | $2.46 | spot | [coreweave.com/pricing](https://www.coreweave.com/pricing) |
| RunPod SXM | $2.69 / $3.49 | Community / Secure | runpod.io/pricing |
| Hyperbolic | $3.19 | on-demand | [computeprices.com](https://computeprices.com/providers/hyperbolic) |
| Nebius | $3.85 | on-demand | nebius.com/prices |
| Crusoe | $3.90 | on-demand | [runpod comparison](https://www.runpod.io/articles/comparison/runpod-vs-crusoe) |
| Modal | $3.95 | per-second | [modal.com/pricing](https://modal.com/pricing) |
| Lambda | $3.99–$4.29 | on-demand | [lambda.ai/pricing](https://lambda.ai/pricing) |
| Together | $3.99 promo (list $5.49) | on-demand cluster | [together.ai/pricing](https://www.together.ai/pricing) |
| CoreWeave | $6.16 | on-demand | coreweave.com/pricing |
| AWS p5.48xlarge | $6.88 | on-demand us-east-1 | [thundercompute](https://www.thundercompute.com/blog/nvidia-h100-pricing) |
| Azure | $6.98 | on-demand eastus | same |
| Oracle / GCP | $10.00 / $10.98 | on-demand | same |

**Cheapest credible: Prime Intellect spot $0.94.** For non-interruptible, **SF Compute's clearing price ($2.03) is the most honest market signal.** Hyperscalers carry a **3–5× premium for identical silicon.**

#### H200 / B200 / B300 / GB200

| GPU | Cheapest | Then | Hyperscaler |
|---|---|---|---|
| **H200 141GB** | Prime Intellect **$1.99** | Nebius preempt $2.45 · CoreWeave spot $2.62 · RunPod Comm $3.59 · Hyperbolic $3.99 · Crusoe $4.29 · Nebius $4.50 · Modal $4.54 · RunPod Secure $4.59 · Together $5.99 · CoreWeave $6.31 | ~$5.58–$6.35 |
| **B200 192GB** | Prime Intellect **$3.49** | Nebius preempt $3.95 · CoreWeave spot $4.26 · GCP spot $4.95 · Vast.ai $5.64 · Hyperbolic/RunPod-Comm $5.98 · Modal $6.25 · Lambda $6.69 · RunPod Secure $6.79 · Nebius $7.15 · Together $8.19 · CoreWeave $8.60 | AWS $14.24 · GCP $16.11 |
| **B300 288GB** | Prime Intellect **$4.99** | RunPod Comm $6.94 · Modal $7.10 · RunPod Secure $7.89 | n/a |
| **GB200 NVL72** | CoreWeave **$10.50/GPU-h** — only public per-GPU rate found | Nebius/Together/Lambda: contact sales | n/a |
| **GB300 NVL72** | No public pricing anywhere | CoreWeave & SF Compute: contact sales | n/a |

Sources: [runpod](https://www.runpod.io/pricing), [modal](https://modal.com/pricing), [nebius](https://nebius.com/prices), [coreweave](https://www.coreweave.com/pricing), [together](https://www.together.ai/pricing), [computeprices/vast](https://computeprices.com/providers/vast), [thundercompute B200](https://www.thundercompute.com/blog/nvidia-b200-pricing).

#### AMD

| GPU | Cheapest | Notes |
|---|---|---|
| MI300X 192GB | Azure spot **$1.11**; TensorWave $1.71 (quote-only); Vultr preempt $1.85; **RunPod $2.39** (best self-serve on-demand) | DigitalOcean $2.59 · Crusoe $3.45 · Azure/Oracle $6.00 |
| MI355X 288GB | Vultr 36-mo reserved $2.29; Vultr preempt $2.59; TensorWave $2.95 "starting at" | **Oracle $8.60 is the only verified true on-demand rate** (BM.GPU.MI355X.8 @ $68.80/node) |

Sources: [spheron AMD](https://www.spheron.network/blog/amd-mi300x-mi355x-pricing-2026), [thundercompute MI300X](https://www.thundercompute.com/blog/amd-mi300x-pricing).

#### NVIDIA Rubin / VR200 — not rentable

Full production since June 2026; partner availability H2 2026 via AWS, GCP, Azure, OCI, CoreWeave, Lambda, Nebius, Nscale. **No published per-hour pricing exists** and no self-serve access; meaningful non-hyperscaler availability expected 2027 ([nvidianews](https://nvidianews.nvidia.com/news/rubin-platform-ai-supercomputer), [spheron](https://www.spheron.network/blog/vera-rubin-nvl72-cloud-availability-cost-per-token)). **Do not plan around it.**

### B.3 Google TPU pricing

All first-party from [cloud.google.com/tpu/pricing](https://cloud.google.com/tpu/pricing), per chip-hour:

| TPU | Region | On-demand | DWS Flex-start | 1-yr | 3-yr |
|---|---|---|---|---|---|
| **Ironwood (v7)** | us-central1 | $12.00 | $6.00 | $8.40 | $5.40 |
| Ironwood (v7) | europe-west2 | $13.20 | $6.00 | $9.24 | $5.94 |
| **Trillium (v6e)** | us-east1 / us-east5 | $2.70 | **$1.35** | $1.89 | $1.22 |
| TPU v5p | us-east5 | $4.20 | $2.10 | $2.94 | $1.89 |
| **TPU v5e** | us-central1 / us-east5 | $1.20 | $0.60 | $0.84 | $0.54 |
| TPU v4 pod | us-central2 | $3.22 | — | $2.03 | $1.45 |

**Ironwood (v7) is GA and publicly priced as of Sept 2026** — that's new. Google no longer publishes a spot rate in the table ("Spot prices are dynamic and can change up to once every 30 days"); **DWS Flex-start at exactly 50% of on-demand is the published discount path.** Trillium at $1.35 flex-start is the best perf/$ tier.

### B.4 Free and near-free compute

#### TPU Research Cloud — still alive, still free

| | |
|---|---|
| Status | **Running.** Rolling invitations ([sites.research.google/trc](https://sites.research.google/trc/about)) |
| What you get | Access to a pool of **>1,000 Cloud TPU devices**; free TPU quota granted to *your own GCP project*, temporary, "ready to use within minutes" ([TRC FAQ](https://sites.research.google/trc/faq)) |
| Current generations | 2026 user reports confirm **spot v5e + v6e** grants ([Google Dev forum](https://discuss.google.dev/t/student-researcher-blocked-have-trc-tpu-quota-but-no-credit-card-for-gcp-billing-verification/349193), [r/googlecloud](https://www.reddit.com/r/googlecloud/comments/1s7nt82/unable_to_create_tpu)) — anecdotal but consistent |
| Duration | Classically 30 days, renewable on request — **UNVERIFIED**, not stated on current pages |
| Eligibility | **No degree requirement. "Anyone can express interest."** Undergrads are being approved — the forum thread above is an undergrad with a granted v5e/v6e cluster |
| You still pay | VM (n1-standard-2) + GCS storage — "generally minimal" |
| **Catch** | **Requires a GCP billing account, i.e. a credit card.** This is the documented blocker for students |
| Obligation | Publish results openly |

#### Always-free tiers

| Program | What you realistically get | Friction | Source |
|---|---|---|---|
| **Kaggle** | **30 hr/wk GPU** (P100 or T4×2) + ~20 hr/wk **TPU v3-8**; 12 hr session (9 hr TPU) | Phone verify, no CC. **Best predictable free GPU** | [kaggle docs](https://www.kaggle.com/docs/efficient-gpu-usage) |
| Colab free | T4 when available, ≤12 hr session, ~90 min idle kill, **no guarantee** | Google account | [colab FAQ](https://research.google.com/colaboratory/faq.html) |
| Colab Pro / Pro+ | $9.99/mo = 100 CU; $49.99/mo = 500 CU (+background exec). Burn: T4 1.19 CU/hr (~$0.12), L4 1.71, A100-40 5.40, A100-80 7.52, RTX PRO 6000 8.71 | Premium GPU still not guaranteed | [signup](https://colab.research.google.com/signup); rates [mccormickml](http://mccormickml.com/2024/04/23/colab-gpus-features-and-pricing) |
| **HF ZeroGPU** | Free acct **5 min/day** GPU (RTX PRO 6000 Blackwell 96GB) + ~3 runs/day; PRO $9/mo = **40 min/day**; overage $1/10 min | Demo-scale only, not training. ⚠️ Conflicting figures circulate (3.5/25 min) — the docs table is authoritative | [HF docs](https://huggingface.co/docs/hub/en/spaces-zerogpu) |
| **Modal** | **$30/month free compute**, per-second billing (H100 $3.95/hr → **~7.6 free H100-hrs/mo**) | None | [modal.com/pricing](https://modal.com/pricing) |
| Lightning AI | 15 credits/mo (~$15) ≈ **22 T4-hours**; T4/L4/A10G/L40S | Phone verify, no CC. ("80 GPU-hrs" marketing claim UNVERIFIED) | [aicreditmart](https://aicreditmart.com/ai-credits-providers/lightning-ai-free-plan-22-gpu-hours-month-guide-2026) |
| **Cerebras** | **~1M tokens/day free**, no CC, 8,192-token context cap | Inference only | [pricepertoken](https://pricepertoken.com/endpoints/cerebras/free) |
| **Groq** | 30 RPM / ~6K TPM / **14,400 req/day**, no CC | Inference only | [cloudzero](https://www.cloudzero.com/blog/groq-pricing) |

#### Credits and grants

| Program | Amount | Eligibility | Source |
|---|---|---|---|
| **Prime Intellect Fast Compute Grants** | **$500–$100,000** in compute credits, **5–10 day decisions** | **Anyone, anywhere.** Email a 1-page pitch. No .edu, no VC, no nationality gate — high technical bar. **Best single option for an ambitious undergrad.** | [x.com/PrimeIntellect](https://x.com/PrimeIntellect/status/1786386588726960167), [GPU-Grants repo](https://github.com/eric-prog/GPU-Grants), [aicredits.dev](https://aicredits.dev/submissions/176-prime-intellect-fast-compute-grants-500-100k) |
| Azure for Students | **$100/yr**, renewable while enrolled, **no credit card** | Full-time student 18+, school email. ⚠️ **GPU quota approved separately and frequently denied** | [azure/free/students](https://azure.microsoft.com/en-us/free/students) |
| GitHub Student Pack | Azure $100 · Codespaces Pro · Heroku $13/mo × 24 · Copilot Pro free · Deepnote. **No GPU offer** | Verified student | [education.github.com/pack](https://education.github.com/pack) |
| GCP free trial | $300 / 90 days | Credit card required | [cloud.google.com/free](https://cloud.google.com/free) |
| AWS Activate Founders | $1K self-serve, up to $5K self-funded | Requires a "startup". AWS Educate closing to new signups 2026-07-30 (**UNVERIFIED**, [aimultiple](https://aimultiple.com/free-cloud-gpu)) | [aws.amazon.com/activate](https://aws.amazon.com/activate) |
| **NVIDIA Academic Grant** | **Up to 30,000 H100-80GB hours** or 8× RTX PRO 6000 | ❌ **Full-time faculty only**, PhD-granting institution. Deadline Jun 30, decisions Sept. An undergrad benefits only via a prof | [nvidia.com/academic-grant-program](https://www.nvidia.com/en-us/industries/higher-education-research/academic-grant-program) |

#### Canada — Digital Research Alliance (the McMaster route)

The 2025–26 renewal retired most of the old fleet. **Live:** Fir (SFU), Narval, Nibi (Waterloo), Rorqual (Québec), Trillium (SciNet). **End of life:** Béluga, Cedar, Graham, Niagara. **PAICE AI clusters in production:** Killarney (Vector), TamIA (Mila), Vulcan (Amii) ([National systems](https://docs.alliancecan.ca/wiki/National_systems)).

| Cluster | GPUs | Source |
|---|---|---|
| **Fir** (SFU) | 160 nodes × 4 **H100 SXM5 80GB** = **640 H100** (NVLink; ~half MIG-partitioned into 1g.10gb / 2g.20gb / 3g.40gb slices) | [docs/Fir](https://docs.alliancecan.ca/wiki/Fir) |
| **Killarney** (Vector/SciNet) | 168 × 4 **L40S 48GB** = **672 L40S**; 10 × 8 **H100 SXM 80GB** = **80 H100** | [docs/Killarney](https://docs.alliancecan.ca/wiki/Killarney) |

**Eligibility reality for a McMaster undergrad:**
- **You cannot self-register.** Only a faculty member at a CFI-eligible Canadian university can register as a PI. Students register as Group Members and **must supply a sponsor's CCRI**; the PI confirms by email. PI approval ≤2 business days; student approval instant once the prof clicks ([Apply for a CCDB account](https://docs.alliancecan.ca/wiki/Apply_for_a_CCDB_account)).
- **Rapid Access Service is the default and does not guarantee GPUs.** CPU ~200 core-years/cluster opportunistically. GPU: *"available for opportunistic use... We cannot therefore guarantee any amount of resources available to each group... especially during times of high demand"* — and demand spikes before conference deadlines. Storage up to 40 TB project + 100 TB nearline ([Rapid Access Service](https://docs.alliancecan.ca/wiki/Rapid_Access_Service)).
- **RAC** (the competition yielding guaranteed GPU-years) is **PI-only**; you cannot apply.
- **Killarney is more open than expected:** access is for "Vector affiliated PIs with CCAI Chairs **as well as researchers within an AI program at a Canadian university or applying AI methods for their research**." The PI needs an `aip-` RAP from an AI institute or via [General Access to PAICE Systems](https://ccdb.alliancecan.ca/paice/general_access_to_paice_systems), then adds your CCRI.

**Bottom line: one willing McMaster prof → CCDB account in ~2 days → opportunistic H100/L40S time on Fir and Killarney, free but unguaranteed and queue-dependent. Highest-ceiling free option, and it costs one email.**

#### Decentralized contribution — hardware floors

| Protocol | Min VRAM | Rec VRAM | Min bandwidth | Latency tolerance |
|---|---|---|---|---|
| Pluralis | 40 GB | 80 GB | 10 Gb/s | 150 ms |
| Prime Intellect | 80 GB | 80–160 GB | 25 Gb/s | 100 ms |
| Nous Psyche | 40 GB | 80 GB | 10 Gb/s | 200 ms |

Source: [Spheron](https://www.spheron.network/blog/decentralized-llm-training-pluralis-prime-intellect-nous-psyche).

> **A Mac Mini M4 with 16 GB unified memory meets none of these VRAM floors.** Contributing compute to a decentralized run is not viable from this hardware. The Prime Intellect *grant* route is.

### B.5 The compute ladder

**Method.** FLOPs = 6 · N_active · D. GPU-hours = FLOPs / (peak_BF16 × MFU × 3600). H100 SXM peak BF16 dense = 989 TFLOP/s; B200 = 2250 TFLOP/s.

**MFU calibrated against three disclosed runs** — the part most back-of-envelope ladders get wrong:

| Run | FLOPs | Disclosed GPU-hours | **Implied MFU** |
|---|---|---|---|
| SmolLM3 3B | 2.02e23 | 220k H100-h | **25.7%** |
| DeepSeek-V3 671B-A37B | 3.29e24 | 2.788M H800-h | **33.1%** |
| Llama 3.1 405B | 3.80e25 | 30.84M H100-h | **34.6%** |

**Use 35% MFU for large runs, ~26% for small / long-context-heavy runs.** The ladder uses 35% (H100) / 30% (B200), which reproduces DeepSeek-V3's disclosed cost to within ~4%. Prices are the verified Sept 2026 floors from §B.2: **H100 spot $0.94, H100 on-demand $2.43, B200 $3.49.**

#### Chinchilla-optimal (~20 tokens per *active* parameter)

| Model | Total / Active | Tokens | FLOPs (6·Na·D) | H100-hrs @35% | **$ @ $0.94 spot** | $ @ $2.43 on-dem | B200-hrs @30% | $ @ $3.49 |
|---|---|---|---|---|---|---|---|---|
| 125M (GPT-2 small) | 0.125B / 0.125B | 2.5B | 1.88e18 | 2 | **$1** | $4 | 1 | $3 |
| 1B dense | 1B / 1B | 20B | 1.20e20 | 96 | **$91** | $234 | 49 | $172 |
| 3B dense | 3B / 3B | 60B | 1.08e21 | 867 | **$815** | $2.1k | 444 | $1.6k |
| 8B dense | 8B / 8B | 160B | 7.68e21 | 6.2k | **$5.8k** | $15.0k | 3.2k | $11.0k |
| **30B-A3B MoE** | 30B / 3B | 60B | 1.08e21 | 867 | **$815** | $2.1k | 444 | $1.6k |
| 70B dense | 70B / 70B | 1,400B | 5.88e23 | 471.9k | $443.5k | $1.15M | 242.0k | $844.5k |
| 235B-A22B MoE | 235B / 22B | 440B | 5.81e22 | 46.6k | $43.8k | $113.3k | 23.9k | $83.4k |
| 671B-A37B MoE (DSv3) | 671B / 37B | 740B | 1.64e23 | 131.8k | $123.9k | $320.3k | 67.6k | $235.9k |
| 1T-A32B MoE (K2) | 1000B / 32B | 640B | 1.23e23 | 98.6k | $92.7k | $239.6k | 50.6k | $176.5k |
| 2.8T-A104B MoE (K3) | 2800B / 104B | 2,080B | 1.30e24 | 1.04M | $979.1k | $2.53M | 534.1k | $1.86M |

#### Overtrained (production token budgets)

| Model | Total / Active | Tokens | FLOPs (6·Na·D) | H100-hrs @35% | **$ @ $0.94 spot** | $ @ $2.43 on-dem | B200-hrs @30% | $ @ $3.49 |
|---|---|---|---|---|---|---|---|---|
| 125M | 0.125B / 0.125B | 100B | 7.50e19 | 60 | **$57** | $146 | 31 | $108 |
| 1B dense | 1B / 1B | 1,000B | 6.00e21 | 4.8k | **$4.5k** | $11.7k | 2.5k | $8.6k |
| **3B dense (SmolLM3 recipe)** | 3B / 3B | 11,000B | 1.98e23 | 158.9k | **$149.4k** | $386.1k | 81.5k | $284.4k |
| 8B dense | 8B / 8B | 15,000B | 7.20e23 | 577.8k | $543.1k | $1.40M | 296.3k | $1.03M |
| **30B-A3B MoE** | 30B / 3B | 15,000B | 2.70e23 | 216.7k | **$203.7k** | $526.5k | 111.1k | $387.8k |
| 70B dense | 70B / 70B | 15,000B | 6.30e24 | 5.06M | $4.75M | $12.29M | 2.59M | $9.05M |
| 235B-A22B MoE | 235B / 22B | 15,000B | 1.98e24 | 1.59M | $1.49M | $3.86M | 814.8k | $2.84M |
| **671B-A37B MoE (DeepSeek-V3 scale)** | 671B / 37B | 14,800B | 3.29e24 | 2.64M | **$2.48M** | $6.41M | 1.35M | $4.72M |
| 1T-A32B MoE (K2 scale) | 1000B / 32B | 15,500B | 2.98e24 | 2.39M | $2.24M | $5.80M | 1.22M | $4.27M |
| 2.8T-A104B MoE (K3 scale) | 2800B / 104B | 15,000B | 9.36e24 | 7.51M | $7.06M | $18.25M | 3.85M | $13.44M |

*(Computed this session; validated against DeepSeek-V3's disclosed $5.576M / 2.788M H800-h and SmolLM3's 220k H100-h.)*

#### What $2,000 actually buys

At Prime Intellect spot ($0.94/H100-h): **2,128 H100-hours = 266 hours on an 8×H100 node = 11.1 node-days = 2.65e21 FLOP at 35% MFU.** That is:

- a **1B dense** model on **442B tokens**, or
- a **3B dense** or **30B-A3B MoE** on **147B tokens**, or
- **161 nanochat speedruns**.

**Three readings that matter:**

1. **The MoE active-parameter trick is the whole game.** A 30B-A3B MoE at Chinchilla-optimal costs the same as a 3B dense — **~$815** — because compute scales with *active* params. You get 30B of capacity for a 3B training bill. This is exactly why Qwen3-30B-A3B-shaped models exist, and why DeepSeek-V4.1-Flash runs a 552B backbone at 8B prefill / 16B decode.
2. **Pretraining anything competitive is out of reach.** Matching even DeepSeek-V4-Flash-class pretraining is seven figures before a single ablation. The frontier is 1e26–1e27 FLOP; this ladder's largest row is 9.4e24. A student is **two orders of magnitude** short at minimum.
3. **The affordable rungs produce nothing that beats Fable 5.1 at anything general.** The money must go into post-training, scaffolding, and inference — see Part C.

---

## Part C — Paths to beat Fable 5.1 on at least one benchmark

### C.1 Narrow-benchmark specialization with an open base model

**This works, is well-documented, and reaches down to hobby scale.**

| Project | Base | Method | Compute / cost | Benchmark | Score | Closed SOTA then | Beat? | Source |
|---|---|---|---|---|---|---|---|---|
| **Kimi K3** | — | Full training | Industrial | **GPQA Diamond** | **93.5** | **Fable 5.1: 92.6** | **YES** | [kimi.ai](https://www.kimi.ai/blog/kimi-k3), [DataScienceDojo](https://datasciencedojo.com/blog/claude-fable-5-1-performance-and-safety) |
| **Kimi K3** | — | Full training | Industrial | **BrowseComp** | **90.4–91.2** | Opus 5: 90.8; no Fable 5.1 figure | **YES vs Opus 5** | [kimi.ai](https://www.kimi.ai/blog/kimi-k3), [BenchLM](https://benchlm.ai/benchmarks/browsecomp) |
| **DeepScaleR-1.5B** (Berkeley Sky/BAIR) | DeepSeek-R1-Distill-Qwen-1.5B | GRPO RL, iterative ctx scaling 8K→16K→24K | **$4,500** | AIME 2024 Pass@1 | **43.1%** (base 28.8%) | o1-preview | **YES** | [Notion](https://pretty-radio-b75.notion.site/DeepScaleR-Surpassing-O1-Preview-with-a-1-5B-Model-by-Scaling-RL-19681902c1468005bed8ca303013a4e2), [HF](https://huggingface.co/agentica-org/DeepScaleR-1.5B-Preview) |
| **DeepSWE-Preview** (Agentica + Together) | Qwen3-32B | Pure RL via rLLM on 4,500 R2E-Gym tasks | **64×H100 × 6 days ≈ 9,216 GPU-h ≈ $8.7k @ $0.94 spot** | SWE-bench Verified | **42.2% Pass@1 / 59% w/ TTS / 71.0% Pass@16** | — | Open SOTA | [Together](https://www.together.ai/blog/deepswe), [rllm](https://github.com/agentica-project/rllm/blob/main/examples/swe/README.md) |
| **GLM-5.1** (Z.ai) | — | Full training | Industrial | SWE-bench Pro | **58.4%** | GPT-5.4 57.7, Opus 4.6 57.3 | **YES** | [discretestack](https://discretestack.com/blog/beyond-the-frontier-2026-open-weight-leaders) |
| **MiroThinker-1.7** | open base | Model + context + interactive scaling | UNVERIFIED | BrowseComp / BrowseComp-Zh | **74.0 / 75.3** | OpenAI Deep Research **51.5%** | **YES** | [arXiv 2511.11793](https://arxiv.org/html/2511.11793), [OpenAI](https://openai.com/index/browsecomp/) |
| **MiroThinker v1.0** | open base | same | UNVERIFIED | GAIA | **81.9%** | beat OpenAI DeepResearch, Tongyi, DeepSeek-V3.1 | **YES** | [alphaXiv](https://www.alphaxiv.org/abs/2511.11793) |
| **OpenResearcher** (TIGER-AI-Lab) | open base | agentic research scaffold | UNVERIFIED | BrowseComp-Plus | **54.8%** | beat GPT-4.1, Claude-Opus-4, Gemini-2.5-Pro, DeepSeek-R1, Tongyi | **YES** | [GitHub](https://github.com/TIGER-AI-Lab/OpenResearcher/blob/main/README.md) |
| **S1-VL** (2026-04-23) | Qwen3-VL-32B-Thinking | Specialist post-train; interleaves Python image manipulation with CoT | UNVERIFIED | HRBench-4K, Physics, VRSBench | **SOTA** | — | **YES** | [VLM report](https://github.com/zli12321/Vision-Language-Models-Overview/blob/main/progressive%20reports/2026-04-28.md) |
| **Step-3.5-Flash + PaCoRe** | Step-3.5-Flash | scaffold (PaCoRe) | UNVERIFIED | CodeSOTA accuracy eval | **99.9%** (97.3% w/o) | o4-mini 92.7, Gemini 2.5 Pro 88 | **YES** | [CodeSOTA](https://www.codesota.com/llm) |

**The governing fact:** *"Open-weight models now routinely beat proprietary frontier models on agentic coding benchmarks, even if they trail on closed-domain reasoning"* ([VLM Overview report](https://github.com/zli12321/Vision-Language-Models-Overview/blob/main/progressive%20reports/2026-04-28.md)).

**Cost calibration:** DeepScaleR is the reference point — **$4,500 and a 1.5B model beat a frontier reasoning model on AIME.** DeepSWE at ~$8.7k (at today's spot prices) is the reference for agentic coding.

### C.2 Test-time compute and scaffolding — the highest-leverage lever

#### Scaffold choice alone moves SWE-bench ~8 points on an identical model

| Harness | Model | SWE-bench Verified | Source |
|---|---|---|---|
| SWE-agent | Claude 4 Sonnet | 66.6% | [Confucius Code Agent, arXiv 2512.10398](https://arxiv.org/pdf/2512.10398) |
| OpenHands | Claude 4 Sonnet | 72.8% | same |
| **Confucius Code Agent** | Claude 4 Sonnet | **74.6%** | same |
| mini-SWE-agent | Claude **4.5** Sonnet | 70.6% | same |

**CCA on the weaker Claude 4 Sonnet outperforms mini-SWE-agent on the stronger Claude 4.5 Sonnet.** Scaffold design can more than cancel a model-generation gap.

#### Repeated sampling with a verifier is worth tens of points

**DeepSeek-V2-Coder-Instruct on SWE-bench Lite: 15.9% with one sample → 56% with 250 samples — beating the single-attempt state of the art of 43%, which used more capable frontier models** ([Large Language Monkeys, Brown et al., Stanford](https://scalingintelligence.stanford.edu/pubs/large_language_monkeys)).

Coverage scales with sample count across **four orders of magnitude**; on GSM8K and MATH, Llama-3 coverage exceeds 95% at 10,000 samples. A single GPT-4o sample can cost more and solve fewer problems than five samples from a cheaper model.

**The binding constraint:** this only converts to score where an **automatic verifier** exists. Without one, majority voting and reward models **plateau beyond several hundred samples**. Follow-up theory is blunt: *"there is no free lunch for inference scaling: indefinite accuracy improvement through resampling can only be realized if the verifier is perfect"* ([Provable Scaling Laws for Test-Time Compute](https://www.semanticscholar.org/paper/Provable-Scaling-Laws-for-the-Test-Time-Compute-of-Chen-Pan/e5ae26db2daac1d764421684309c7309dbc6753c)).

#### Evolutionary search: 2.8× on ARC-AGI-2 from an open model

Imbue's code-evolution method (fitness-based sampling + LLM-driven mutation of Python programs representing each task's transformation rule), **ARC-AGI-2 public eval** ([Imbue, 2026-02-27](https://imbue.com/blog/2026-02-27-arc-agi-2-evolution)):

| Base model | Weights | Base | **Evolved** | Lift | Cost/task |
|---|---|---|---|---|---|
| **Kimi K2.5** | **Open** | 12.1% | **34.0%** | **×2.8** | $2.67 |
| Gemini 3 Flash | Closed | 34.0% | **61.4%** | ×1.8 | $2.42 |
| Gemini 3.1 Pro | Closed | 88.1% | **95.1%** | ×1.08 | $8.71 |

Darwinian Evolver is open-sourced with reproduction instructions; a full run takes a few hours and a three-digit USD amount. **Note the pattern: the weaker the base model, the bigger the multiplier.** Test-time compute is worth far more on an open model than on a saturated frontier model — which is precisely the arbitrage available to a small team.

#### Benchmark rules on scaffolds — does this "count"?

| Benchmark | Scaffold rule | Verdict for a challenger |
|---|---|---|
| **Terminal-Bench 4.0** | *"Each row measures a model **and agent harness together**"* — Claude Code, Codex, Grok Build named in submissions; columns are RANK / MODEL / AGENT / RESOLUTION RATE / COST / TOKENS | **Scaffolds are explicitly part of the submission.** Open model + superior harness is fully legitimate ([tbench.ai](https://www.tbench.ai/leaderboard), [llm-stats](https://llm-stats.com/benchmarks/terminal-bench-4.0)) |
| **SWE-bench Verified** | No scaffold restriction; scaffolds, prompts, budgets and termination policies vary per system | Scaffold freedom is the norm — and a known comparability problem ([arXiv 2512.10398](https://arxiv.org/pdf/2512.10398)) |
| **ARC Prize 2026 (Kaggle track)** | **No internet access during evaluation — no API-based systems (GPT/Claude/etc.).** All code open-sourced before official private scores. Two predictions per test input, exact match, no partial credit | **Structurally excludes Fable 5.1** — an open-model-only competition by construction ([ARC Prize 2026](https://arcprize.org/competitions/2026/arc-agi-2), [Kaggle rules](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-2/rules)) |
| **ARC-AGI public leaderboard** | Separate from Kaggle; API models allowed, scored on **cost per task** as a first-class axis | Cost-per-task is where open models dominate outright |
| **Artificial Analysis v4.3** | **45% of weight on private test sets** specifically to prevent gaming | Hardest to game; hardest to beat |

**ARC Prize 2026 prize structure:** $700,000 total — Progress Prizes $275,000 (1st $75,000, descending to 8th at $15,000), Grand Prize $275,000 for best solution writeup, Bonus Prize $150,000 for the first solution reaching **85% on the ARC-AGI-2 private set** (rolls to 2027 if unmet). Hardware/runtime limits "announced with the competition launch" — **UNVERIFIED** ([arcprize.org](https://arcprize.org/competitions/2026/arc-agi-2)).

### C.3 Distillation from open teachers

**Legally clean and the cheapest capability transfer available.** Top open models ship under MIT (DeepSeek V4 family, GLM-5/5.2/5.3-Flash, Phi-4, MiMo, Ling), Apache 2.0 (Qwen 3.x, gpt-oss, Gemma 4, Mistral, Tencent Hy3/Hy4, Muse Glimmer) or modified MIT (Kimi K2.x) — **none restricts training other models on generated outputs** (§A.7). Exceptions: Command R+ (CC-BY-NC) and some Grok variants which explicitly forbid using weights to train other models. Closed-model ToS universally forbid it.

**Evidence it works:** DeepSeek distilled R1 reasoning into small Qwen/Llama checkpoints under MIT; DeepScaleR's base *was* DeepSeek-R1-Distill-Qwen-1.5B — the $4,500 AIME result was built on a distilled checkpoint ([HF](https://huggingface.co/agentica-org/DeepScaleR-1.5B-Preview)). DeepSeek V4 is itself a "ten teachers, one student" distillation design ([Labonne](https://maximelabonne.substack.com/p/deepseek-v4-ten-teachers-one-student)). **Meta's Muse Glimmer 30B is explicitly distilled from the closed Muse Spark** and shipped Apache 2.0 ([research.meta.ai](https://research.meta.ai/blog/introducing-muse-glimmer-open-agentic-model)).

**Cost:** generating a distillation corpus is an inference bill, not a training bill. DeepSeek-V4-Flash at **$0.14/M input, $0.28/M output** means **~$300 buys roughly a billion output tokens of teacher traces** — far more than the tens of thousands of high-quality traces the s1/LIMO line suggests suffices.

### C.4 Decentralized / community pretraining

| Project | Status Sept 2026 | Can a student participate? | Source |
|---|---|---|---|
| **Prime Intellect** | **$130M Series A** July 2026 (Radical, NVIDIA Ventures, Intel Capital, Dell Capital); >$150M total; ~$1B valuation. 2026 pivot from distributed *training* to post-training, RL, evaluation and agents: Lab platform, Environments Hub, verifiers, Prime Agent, Recursive Language Models | **Yes — Fast Compute Grants ($500–$100k, 5–10 day decisions).** Compute contribution needs ≥80 GB VRAM + 25 Gb/s | [Wikipedia](https://en.wikipedia.org/wiki/Prime_Intellect), [primeintellect.ai](https://primeintellect.ai) |
| **INTELLECT-1** | 10B, proof-of-concept distributed training across three continents | Historical | [Wikipedia](https://en.wikipedia.org/wiki/Prime_Intellect) |
| **INTELLECT-2** | **32B, first globally decentralized RL run, permissionless contribution** via TOPLOC. Inference workers run on consumer GPUs (**4×RTX 3090 sufficient**) | **Yes, permissionlessly** — but not from a Mac Mini | [Prime Intellect](https://www.primeintellect.ai/blog/intellect-2) |
| **INTELLECT-3** | **106B MoE, RL-trained on 512 NVIDIA H200s across 64 nodes** — a *centralized* cluster. Outperforms larger frontier models on math, code, science, reasoning | Model usable; training was not decentralized | [implicator.ai](https://www.implicator.ai/prime-intellect-s-intellect-3-open-source-ambition-meets-centralized-reality/) |
| **Nous Psyche** | Solana-coordinated; demonstrated dynamic node addition mid-training and tolerance of node failure. **Semi-permissioned** — providers must be approved by Nous | Approval-gated | [Nous](https://nousresearch.com/nous-psyche), [OAK Research](https://oakresearch.io/en/analyses/innovations/nous-research-psyche-open-source-decentralized-ai-revolution) |
| **Pluralis** | Production-viable scale with OpenDiLoCo-style protocol; open contributor pool | Yes, with ≥40 GB VRAM | [Spheron](https://www.spheron.network/blog/decentralized-llm-training-pluralis-prime-intellect-nous-psyche) |

**Honest verdict:** there is a real gap between pitch and practice. Prime Intellect raised $20M promising decentralized training, then trained its flagship INTELLECT-3 on a **centralized 512-GPU cluster** — described as "what their team can produce with conventional infrastructure while the decentralized stack matures" ([implicator.ai](https://www.implicator.ai/prime-intellect-s-intellect-3-open-source-ambition-meets-centralized-reality/)). **Bandwidth, not GPU supply, is the binding constraint.** Decentralized training is a live research area, not yet a way for an individual to get frontier compute.

For a **sub-$50k** pretraining budget, Spheron's guidance: hyperscaler reserved clusters are out of reach, but **4–8 cloud GPU nodes running DiLoCo can train a quality 7B–13B model on meaningful token budgets** ([Spheron](https://www.spheron.network/blog/decentralized-llm-training-pluralis-prime-intellect-nous-psyche)).

### C.5 Honest verdict — ranked targets

**The hard constraint.** The Mac Mini M4 (16 GB unified) is a *development and orchestration* machine, not a training machine. It cannot host a decentralized training node and cannot usefully fine-tune above ~3B. Every path assumes rented or granted GPUs, with the Mini driving the harness.

**The strategic reframe.** Do not try to beat Fable 5.1 at what it is good at. Four structural openings exist:

- **Open models already beat Fable 5.1 on GPQA Diamond** (Kimi K3 93.5 vs 92.6) and beat Opus 5 on BrowseComp. Using an open model *as-is* on the right benchmark is already a win.
- **Fable 5.1's agentic scores are low in absolute terms** — AutomationBench 31.4%, OSWorld-strict 41.7%, Terminal-Bench-Science 52.6%. 45–70 points of headroom.
- **Scaffolds are legitimate and decisive** — Terminal-Bench scores model+harness *pairs by design*; an 8-point scaffold swing on a fixed model is documented; evolutionary search gave ×2.8 to an open model.
- **Cost-per-task is a first-class axis on ARC Prize's leaderboard**, and open models win it outright.

#### Ranked by feasibility

| Rank | Target | Approach | Realistic cost | Probability | Why |
|---|---|---|---|---|---|
| **1** | **Beat Fable 5.1 on a *cost-adjusted* axis of ARC-AGI-2** (e.g. ≥50% at <$0.25/task vs Fable 5.1's 90% @ $3.12/task) | Imbue's open-sourced Darwinian Evolver + an open model (Kimi K2.5/K3, DeepSeek V4.1-Flash), tuned for token efficiency | **$200–800** inference | **High** | Method open-sourced with repro instructions; full run is "a few hours and a three-digit USD amount"; ARC Prize treats cost/task as the most directly comparable efficiency axis. You compete on a published axis where the frontier is expensive |
| **2** | **Beat Fable 5.1 / Opus 5 on BrowseComp or a deep-research benchmark** | Fork MiroThinker or OpenResearcher; improve retrieval + verification loop. Or simply *run the head-to-head* — Anthropic publishes **no** Fable 5.1 BrowseComp figure | **$100–1,000** | **High** | Open agents already beat OpenAI Deep Research (74.0 vs 51.5); Kimi K3 reports 90.4–91.2 vs Opus 5's 90.8. A clean third-party head-to-head is itself publishable, and the missing Fable 5.1 number is the opportunity |
| **3** | **Beat Fable 5.1 on Terminal-Bench-Science 0.1 (52.6%) or AutomationBench (31.4%)** | Best open model (Kimi K3 / DeepSeek-V4.1-Flash) + purpose-built harness exploiting per-task verification | **$500–2,000** | **Medium** | Scaffold is an explicit part of the submission and Fable 5.1's absolute score is low. But the best confirmed open entry on Terminal-Bench 4.0 is 31.2%, so ~24 points must come from the harness — large, though within the documented range when scaffold + test-time sampling combine |
| **4** | **Win an ARC Prize 2026 Progress Prize** ($15k–$75k) | Kaggle track, offline, open-source solution | **$0–500** (Kaggle compute free) | **Medium** | Fable 5.1 is *structurally barred* (no internet, no API models). You compete only against other open solutions. The $150k bonus at 85% is out of reach; a top-8 progress prize is not absurd |
| **5** | **Beat a frontier reasoning model on a specific math/code eval with a small RL'd model** | Replicate DeepScaleR's recipe (GRPO + iterative ctx scaling) on a 2026 base against a *non-saturated* eval | **$2,000–5,000** | **Medium-low** | Existence proof is exact ($4,500, 1.5B, beat o1-preview on AIME). But AIME is saturated (Kimi K2-Thinking 99.1) and Fable 5.1 posts 92.6 GPQA / 90.5 LCB / 100 ProofBench. **Finding a non-saturated narrow eval is the actual work** |
| **6** | **Beat Fable 5.1 on SWE-bench Pro (81.2%)** | RL an open coding model, DeepSWE-style | **$9k–20k** | **Low** | Best open is GLM-5.3 / Qwen3.8 at 67.7; a ~14-point gap on the industry's most-optimized benchmark family |
| **7** | **Beat Fable 5.1 on the AA Intelligence Index (53)** | — | — | **Effectively zero** | 8-point gap, 45% private test sets, requires a frontier-scale general model |
| **8** | **Pretrain a competitive base model** | — | **$150k–$7M** (§B.5) | **Zero** | Not reachable on any student budget |

#### The recommendation

**Target #1 or #2, and pair it with a Prime Intellect Fast Compute Grant application.**

All three of the cheap targets exploit the same arbitrage: **test-time compute is worth far more on an open model than on a saturated frontier model** (Imbue: ×2.8 for Kimi K2.5, ×1.08 for Gemini 3.1 Pro), and **verifiable domains are where repeated sampling actually converts to score**. Both #1 and #2 fit inside $2k, both produce an open-sourced artifact with a reproducible leaderboard delta, and that artifact is exactly what Prime Intellect's grant program funds ($500–$100k, 5–10 day decision, open to anyone). A successful small result parlays into real compute for a larger one.

In parallel, two zero-cost moves with high option value: **email one McMaster professor to get a CCDB sponsor** (opportunistic H100/L40S on Fir and Killarney — highest-ceiling free compute, costs one email), and **apply to TPU Research Cloud** (free v5e/v6e, undergrads are being approved; needs a card on the GCP billing account).

**The honest bottom line: a student cannot build a model that beats Claude Fable 5.1. A student can absolutely build a *system* that beats it on a specific, well-chosen, verifier-rich benchmark — and on GPQA Diamond an open model already has, for free. The lever is the scaffold and the verifier, not the weights.**

---

## Appendix — Reliability notes

- **Primary sources** (Anthropic, Artificial Analysis, Epoch AI, arXiv, GitHub, ARC Prize, Hugging Face model cards, DeepSeek tech report, Nature, cloud provider pricing pages, Alliance Canada docs) carry every load-bearing number.
- **SEO aggregator sites returned heavily fabricated specs during this research** (mindstudio, codersera, apidog and similar) and were discarded. Treat any spec without a primary link as suspect. Aggregators retained with caution: BenchLM, llm-stats, morphllm, kingy, benchmarklist, emergent.sh, datasciencedojo, computingforgeeks, discretestack, digitalapplied, spheron, thundercompute.
- **Where BenchLM's AA leaderboard conflicted with Artificial Analysis's own publication**, the conflict is flagged in §A.1 and the primary source used.
- **Known gaps, all marked inline:** training compute/$ for every Chinese frontier 2026 model (none disclose); Kimi K3 training tokens; OLMo 3 total GPU-hours; llm.c and modded-nanogpt current records; nanochat $300/$1000 tiers; ARC-AGI-2 scores for open models on the official leaderboard; Fable 5.1's BrowseComp and tau2-bench figures; ARC Prize 2026 hardware limits; TRC grant duration.
