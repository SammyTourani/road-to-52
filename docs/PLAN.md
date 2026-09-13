# road-to-52 — Master Plan

*Planner: Claude Fable 5.1 (planning and reasoning only). Builders: Claude Opus / Sonnet agents.
Owner: Sammy Tourani. Started 2026-09-13. Living document — every number links to a source in
`research/` or to a reproducible result in `results/`.*

## 0. TL;DR

- **The literal goal ("build a model that beats Claude Fable 5.1") is not reachable by anyone
  without a frontier-lab budget.** Frontier pretraining runs are 1e26–1e27 FLOPs; the largest rung
  on our costed ladder (Kimi-K3 scale, ~$7M of spot H100 time) is still two orders of magnitude
  short, and post-training/eval/RL infrastructure costs more again. See §1 and
  `research/04-landscape-and-compute.md`.
- **What *is* reachable, and genuinely valuable:** the recipe the "cheap" labs use is public, the
  code is open, and the cost floor has collapsed — a GPT-2-class model is now a few dollars of
  rented GPU, a Chinchilla-optimal 1B model is ~$91, a 3B dense or 30B-A3B MoE is ~$815. Nobody
  has built a *maintained, well-licensed, benchmarked* Apple-Silicon-native pipeline for this,
  and nobody has published an MLX pretraining throughput benchmark. We build exactly that, and
  we climb a public **ladder** where every rung beats a *named* model at a *documented* cost,
  measured against Fable 5.1's published scores by a **gap tracker**.
- **Rung 0 costs $0 and is running (since 2026-09-13):** a from-scratch 124M model trained on the
  Mac Mini M4 in ~3 days of background compute, targeting OpenAI's released GPT-2 124M *as we
  measured it with the same eval code* (FineWeb val loss 3.447, HellaSwag acc_norm 29.4%), with
  the modded-nanogpt speedrun bar (val loss 3.28, a FineWeb-trained 124M) as the stretch.
- **Hard rule:** no Claude outputs are ever used as training data (Anthropic's terms). Claude
  writes code and documents here; the model learns only from open data and open teachers.

## 1. The honest premise

| Fact | Number | Source |
|---|---|---|
| This machine's GPU peak (bf16 matmul, measured) | 3.6 TFLOPS | probe, 2026-09-13 |
| This machine training a 111M GPT (measured, naive) | ~1,950 tok/s, 1.3 TFLOPS effective | probe, 2026-09-13 |
| One H100 (bf16 dense peak) | 989 TFLOPS | research/04 §B.5 |
| DeepSeek-V3 final pretraining run (disclosed) | 2.788M H800-hours ≈ $5.6M rental-equivalent | research/04 §B.1 |
| Frontier (Fable-class) training compute (Epoch-style estimate) | 1e26–1e27 FLOPs; cost undisclosed | research/04 §B.5 |
| Open-weight vs closed frontier, composite indices | ~8 points on AA Intelligence Index (45 vs 53); ~6.6 ECI points ≈ 4.5–6 months | research/01 §3.2 |

"How do Qwen and DeepSeek do it cheaply?" — they don't, they do it *efficiently* at $100M scale:
sparse MoE (compute scales with *active* params: 30B-A3B costs the same as 3B dense), MLA/hybrid
attention, FP8 training, custom pipeline schedules, and heavy verifiable-reward RL and synthetic
data after pretraining. Those buy 3–10× over naive training. The gap from a Mac Mini to a frontier
run is ~10⁹–10¹⁰×. The *recipe* is copyable; the *scale* is not. (`research/06-frontier-recipe.md`
has the stage-by-stage recipe; `research/02-open-training-stack.md` the code that implements it.)

## 2. The bar (September 2026)

Full detail: `research/01-benchmarks-and-bar.md`. Machine-readable: `r52/bar/bar.yaml`.
Rendered: `results/GAP.md`.

**What frontier launches actually report now** (counted across the nine 2026 frontier launches):
Terminal-Bench 4.0 / Science (9/9), DeepSWE v1.1 (8/9), Humanity's Last Exam (7/9),
AutomationBench (7/9), GDPval-AA v2 (6/9), OSWorld 2.0 (6/9), Agents' Last Exam (5/9), GPQA
Diamond (4/9), ARC-AGI-2/3, SWE-bench Pro, plus the composites (Artificial Analysis Intelligence
Index v4.3, Epoch ECI). **Zero of nine** report SWE-bench Verified, MMLU, MMLU-Pro, AIME,
LiveCodeBench, HumanEval or MATH-500 — the classic set is dead at the frontier (still fine for
measuring small models, which is what we use it for).

**Claude Fable 5.1 headline numbers** (Anthropic system card Table 8.1.A, 2026-09-01; max effort,
safeguards on): SWE-bench Pro 81.2 · Terminal-Bench 4.0 56% (official board 57.9) ·
Terminal-Bench-Science 52.6 · HLE 60.9 (no tools) / 65.0 (tools) · OSWorld 2.0 77.9 / 41.7 ·
GDPval-AA 1853 Elo · AutomationBench 31.4 · ARC-AGI-1 97.5 · ARC-AGI-2 90.0 · AA Index 53 ·
Epoch ECI 164.2. Roughly tied with GPT-6 Astra for #1 overall.

**Where open weights stand:** GPQA Diamond is tied (Kimi K3 93.5 vs 93.7); HLE-with-tools within
1.1 points; LiveBench Agentic Coding — an open model (DeepSeek V4.1-Flash) *leads outright*;
long-horizon agentic work (Terminal-Bench 4.0, OSWorld 2.0, ARC-AGI-2) trails by 16–29 points.
**Harness beats model:** two API settings moved GPT-5.6 Sol from 13.3% to 38.3% on ARC-AGI-3;
every published number is a (model, harness, effort, grader) tuple and we record all four.

## 3. What we build

**Thesis.** *Building* an LLM is now documented engineering; *building a frontier LLM* is a capital
problem. So the most impressive thing a student can ship is the strongest **fully-open,
fully-reproducible model per dollar**, proven by beating named models at each cost rung, with an
honest gap tracker up to the frontier — packaged as the pipeline everyone else can run.

Two documented open niches make this novel rather than "another nanochat port"
(`research/02` §"Three gaps"): (1) there is no maintained, well-licensed MLX pretraining framework
(Apple's own MLX lead created an empty placeholder repo for one in March 2026 and never filled
it); (2) no MLX *pretraining* throughput benchmark has ever been published — every circulating
"MLX tokens/sec" figure is inference decode.

**Deliverables**

1. `r52` — the pipeline, MLX-native: data → tokenizer → pretraining → midtraining → SFT → RL →
   eval → export → chat. Modern recipe (modded-nanogpt / nanochat lineage: Muon, QK-norm, ReLU²,
   value embeddings, U-net skips, logit softcap, WSD schedule, mixed precision). Spec:
   `docs/ARCHITECTURE.md`.
2. `results/mlx_pretrain_bench.md` — the first published MLX pretraining throughput table
   (sizes × sequence lengths × precisions, with MFU and memory).
3. The **ladder** (`r52/ladder`, `results/ladder.md`) — costed rungs, each targeting a named model.
4. The **gap tracker** (`r52/bar`, `results/GAP.md`) — Fable 5.1 and the open frontier vs. our
   models, every cell sourced.
5. Trained artifacts on Hugging Face with eval logs (Rung 0 first), plus a chat demo via
   `mlx_lm.server`.
6. The research corpus (`research/`) — six sourced reports on the 2026 landscape.

### 3.1 The recipe we follow (from `research/06-frontier-recipe.md`, Appendix A)

The five small-compute choices with the largest measured effect, and how each maps onto this repo:

1. **Data quality and mixture is the only lever with a published 2×.** Filter harder than the
   published recipes at small scale (optimal keep-rate ≈ top 3% at 1e20 FLOPs vs top 10% at 1e22);
   spend effort on the classifier's positive seed. DataDecide shows corpus rankings at **150M
   parameters predict the 1B winner ~80% of the time** — which makes this Mac Mini a legitimate
   *data-ablation instrument*, not just a toy. → Rungs 0/0a fix the corpus for comparability; the
   data-ablation lab is a Phase-4 deliverable.
2. **A hybrid attention stack (3 cheap : 1 global) and a widened residual stream.** The residual
   change (U-net / MUDD skips, value embeddings) is kernel-free and validated at 124M *and* at
   frontier scale → in the Rung 0 model. The hybrid stack (Gated DeltaNet, ~2× data efficiency
   at 7B in Olmo Hybrid) → Rung 1+.
3. **Muon, plus a re-fit of LR and batch size.** Muon on 2-D matrices, AdamW on embeddings / head /
   norms / anything vector-shaped; skip batch-size warmup under Muon; **split fused qkv before
   orthogonalizing** (the most common from-scratch bug); the optimum is a flat bowl (√2 in LR,
   +25% in batch) so don't over-sweep. → `r52/optim.py`.
4. **Midtraining, not just pretraining.** A short (1–2% of tokens), LR-decaying, high-quality stage
   that deliberately seeds instruction-following and thinking data into the *base* model;
   decontaminate there specifically; average two midtrain seeds (Olmo 3 trick). → Phase 4.
5. **Measure base-model pass@k before doing any RL.** High pass@128 ⇒ RL only sharpens; low ⇒ RL
   can expand the boundary. Cheap headline results are mostly distillation, bounded by the
   teacher; but SFT→RL ordering is worth ~5 points over either alone (Magistral: 65.4 / 65.8 /
   70.7). → Phase 4 uses SFT → GRPO-family (DAPO-style: no KL, no std-norm, clip-higher,
   token-level loss, dynamic sampling) on reasoning-gym, after a pass@k probe.

Things we explicitly do **not** do at small scale: reproduce R1's RL setup (≥70k A100-hours at
1.5B); carry forward someone else's LR/batch recipe after changing optimizer; chase validation
loss alone (we always report HellaSwag/CORE next to loss); use FP8 before the bf16 run is correct.
Frontier defaults that differ from the speedrun recipe at 124M (e.g. "no logit soft-capping",
SwiGLU over ReLU²) are config flags, decided by measurement at our scale.

## 4. The ladder

Costs use the calibrated ladder in `research/04 §B.5` (35% MFU on H100, spot $0.94/h, on-demand
$2.43/h; formulas in `r52/ladder/ladder.py`). Local numbers use this machine's measured throughput.

| Rung | Hardware | Model | Tokens | Est. cost | Est. time | Beats (named target) | Status |
|---|---|---|---|---|---|---|---|
| **0** | Mac Mini M4 (this machine) | 124M dense, from scratch | ≤ 0.75B | **$0** | ~3 days | **GPT-2 small** (OpenAI 2019) as *we* measured it with the same code: FineWeb val loss ≤ 3.447, HellaSwag acc_norm ≥ 29.4% (acc ≥ 28.5%), CORE ≥ the gpt2 reference (pending). Stretch: the speedrun bar, val loss ≤ 3.28 (llm.c's FineWeb-trained 124M) | **running** since 2026-09-13 |
| 0a | Mac Mini M4 | ~35M "nano", full pipeline (pretrain→SFT→RL→chat) | 0.4B | $0 | ~1 day | pipeline proof, not a record | building |
| **1** | 8×H100, ~2 h | ~1B dense (nanochat d24–d26 class) | ~20B | ~$50–100 | hours | **GPT-2 XL** (1.6B): CORE ≥ 0.2565 | planned; needs budget |
| **2** | 8×H100, ~5 days | 3B dense **or** 30B-A3B MoE, Chinchilla-optimal | 60B | ~$815 spot | days | **Llama-1 7B** (Meta 2023, MMLU 35): MMLU/HellaSwag/ARC/GSM8K | planned; needs budget |
| 2b | 8×H100, ~11 days | same, overtrained | 150B | ~$2,000 spot | ~2 weeks | **Llama-2 7B** (2023, MMLU 45.3) — stretch | planned |
| 3 | 64×H100, ~4 days | 8B dense, Chinchilla-optimal | 160B | ~$5,800 spot | days | Llama 3.1 8B's *size* at 1/94th its tokens — a size match, not a quality claim | planned; needs budget |
| P | any rung's base + 8×H100 | post-training track: midtrain → SFT → RLVR / on-policy distillation from open teachers | — | $2k–5k | days | a named 2024 frontier model on a *non-saturated* math/code eval (chosen from `results/GAP.md`) | planned |
| ref | 384×H100, 24 d | SmolLM3 3B | 11T | ~$150k | — | (reference) | — |
| ref | 2,048×H800, 2 mo | DeepSeek-V3 671B-A37B | 14.8T | ~$2.5M spot-equiv. | — | (reference) | — |
| ref | undisclosed | Fable 5.1 class | — | 1e26–1e27 FLOPs | — | the bar | — |

Rungs 1+ are prepared as ready-to-run scripts with cost estimates; *spending money on them is the
owner's decision*. Free-compute routes worth pursuing in parallel (research/04 §B.4): Prime
Intellect Fast Compute Grants ($500–100k, open to anyone, 5–10 day decisions), a McMaster faculty
CCDB sponsor (opportunistic H100/L40S on Fir/Killarney — one email), TPU Research Cloud, Modal's
$30/month free tier, Kaggle's 30 GPU-hours/week.

**Optional Track S ("system beats model").** The only route to a literal "beats Fable 5.1 on X"
number is a *system* — an open model plus a scaffold and verifier — on a verifier-rich benchmark
where Fable's absolute score is low or unpublished (cost-adjusted ARC-AGI-2, BrowseComp,
Terminal-Bench-Science, AutomationBench; ranked in research/04 §C.5). It is not "building an
LLM", so it is out of scope for the loop unless the owner opts in.

## 5. Building blocks chosen

From `research/02-open-training-stack.md` (recommended-stack table) and `research/05`:

| Stage | Local (Mac, MLX) | Rented GPUs (CUDA) |
|---|---|---|
| Full pipeline reference | our `r52` (nanochat lineage) | nanochat `speedrun.sh`; OLMo-core + open-instruct for reproducible mid-scale |
| Pretraining framework | `r52` (MLX; `mx.fast` kernels, Muon) | torchtitan (FSDP2/TP/PP/CP, float8, MoE); Megatron-LM only at frontier scale |
| Tokenizer | GPT-2 tiktoken for Rung 0 comparability; rustbpe/HF tokenizers (32K) for own vocab | same |
| Post-training | mlx-lm (SFT/LoRA/DoRA/QLoRA, Muon) + mlx-lm-lora (DPO/GRPO/GSPO/DAPO/QAT) | verl (default), prime-rl (async agentic), TRL (small); TRL `DistillationTrainer` for on-policy distillation |
| RL environments | reasoning-gym (procedural, no sandbox) | verifiers + Environments Hub; SWE-smith; OpenEnv as portability hedge |
| Eval | our MLX evals (val loss/bpb, HellaSwag, DCLM CORE) + `mlx_lm.evaluate` (lm-eval bridge) at q6/q8 | inspect_ai + inspect_evals (agentic), lm-eval (static), harbor (terminal) |
| Serving | `mlx_lm.server` (SGLang-MLX as second option) | vLLM / SGLang |
| Data processing | datatrove | datatrove; NeMo Curator with idle GPUs |
| Dead/avoid | torchtune, torchforge, NeMo monolith, unlicensed repos (autoresearch, evalchemy) | — |

**Data** (`research/03-open-data.md`, 393 sources):

| Use | Choice | Notes |
|---|---|---|
| Rung 0 pretraining | `kjj0/fineweb10B-gpt2` shards (FineWeb 10B sample, GPT-2 tokens) | exact comparability with the modded-nanogpt / llm.c target (val loss 3.28) |
| Local mix with own tokenizer (≤10B tokens, ~70 GB) | 70% `allenai/dolma3_mix-150B-1025` slice · 15% `HuggingFaceTB/stack-edu` · 8% `HuggingFaceTB/finemath` (4plus) · 7% Cosmopedia v2 | all commercially clean; ClimbMix avoided (NC license) |
| Mid (100B tokens, ~270–450 GB) | `HuggingFaceFW/finepdfs_edu_50BT-dclm_30BT-fineweb_edu_20BT` (pre-blended) or the built mix in research/03 §10.2 | evidence: DCLM > FineWeb-Edu by +7 CORE at equal tokens; Nemotron-CC > DCLM; FinePDFs is the freshest signal |
| SFT | `HuggingFaceTB/smol-smoltalk` (local) · `allenai/Dolci-Think-SFT-*`, `open-thoughts/OpenThoughts3-1.2M`, `nvidia/Nemotron-SFT-*` (mid) | open teachers only (DeepSeek/Qwen/GLM/gpt-oss) |
| Preference | `nvidia/HelpSteer3` · `allenai/Dolci-Think-DPO-*` | **not** the Tulu 3 preference mixture (Claude in the pool) |
| RLVR | `reasoning-gym` (procedural, free) · `BytedTsinghua-SIA/DAPO-Math-17k` · `SynthLabsAI/Big-Math-RL-Verified` · `microsoft/rStar-Coder` · `allenai/RLVR-IFeval` | verifiable rewards; no sandbox needed for reasoning-gym |
| Decontamination | `allenai/decon` against every eval we report | research/03 §8 |

**Excluded (Claude-derived, verified):** `SWE-bench/SWE-smith-trajectories` (97.85% Claude by its own
`model` column), `SWE-Gym/OpenHands-*-Trajectories`, `R2E-Gym/R2EGym-SFT-Trajectories`,
`allenai/llama-3.1-tulu-3-8b-preference-mixture` (and SmolTalk2's Preference split that inherits
it), `Anthropic/hh-rlhf`, every `*claude-code-traces*` corpus, `QuixiAI/dolphin-distill`
(transitive), plus surgery on `tulu-3-sft-mixture` (drop the Claude-written Python subset) and
`SWE-smith` tasks (drop `lm_rewrite`). GPT/Gemini-distilled sets are avoided by default for the same
reason (OpenAI/Google terms).

## 6. Phases

| Phase | What | Exit criterion |
|---|---|---|
| 0 Research | six sourced reports; measured local ceiling | ✅ done 2026-09-13 |
| 1 Core | `r52` model/optim/data/train/bench + ladder + bar tooling; tests green; GitHub repo public | tests pass; `results/mlx_pretrain_bench.md`, `results/ladder.md`, `results/GAP.md` rendered |
| 2 Run 0a | nano 35M pretrain (~1 day) → eval (val/bpb, HellaSwag, CORE) → export → `mlx_lm.generate` works | numbers in `docs/RESULTS.md` |
| 3 Rung 0 | 124M run (3–6 days, background) with periodic val; GPT-2 reference measured with *our* evals on HF `gpt2` weights | val ≤ 3.28 or budget exhausted — either way, published honestly |
| 4 Post-train | midtrain + SFT (open SFT data, no Claude) + GRPO on reasoning-gym via mlx-lm-lora on the nano model; chat via `mlx_lm.server`; `mlx_lm.evaluate` suite | a chat-capable model with before/after evals |
| 5 CUDA rungs | torchtitan/nanochat configs + cost sheets for Rungs 1–3; grant applications drafted | scripts + costs committed; runs gated on owner budget |
| 6 Publish | HF model cards, README results, LinkedIn/X write-up draft for Sammy | artifacts public |

## 7. Rules

1. **No Claude-generated training data.** Ever. Open datasets and open-teacher outputs only.
2. **Honest numbers:** command, sample count/limit, tokenizer, sequence length, commit, wall-clock,
   and the conditions of any published number we compare against.
3. **Licenses:** Apache-2.0 here; port ideas from MIT/BSD/Apache code with attribution; never copy
   unlicensed code; mergekit (LGPL) only as an external CLI.
4. **The Mac stays usable:** ≤ 6 GiB GPU memory for training processes, `nice`, artifacts on the
   external disk.
5. **Scope:** build the pipeline and climb the ladder; do not start a "system" track without the
   owner's opt-in; do not spend money without the owner's go.

## 8. Risks

- **Throughput:** MLX training MFU on M4 is 20–30%; if the 124M run projects > 7 days at 0.75B
  tokens, fall back to a 90M config or a smaller token budget and report the shortfall.
- **Speedrun techniques that don't port** (FP8, FlashAttention-3, 64K-token eval windows): expect
  to need more tokens than the 8×H100 record (~400M); we budget up to 0.75B.
- **Machine contention** with Hermes/Claude Code sessions; mitigated by the memory limit and nice.
- **Benchmark drift:** the 2026 frontier set is agentic and expensive; our local numbers are on
  classic evals, which is fine for small models but must never be spun as frontier comparisons.

## 9. Loop protocol

The `/loop` self-paces: each tick advances the current phase, checks running jobs (training logs),
launches builders for the next component, commits results, and re-schedules. The loop **stops**
when Phase 6 is complete (or when a phase is blocked on the owner: budget, grants, HF upload
approval). It does not run toward the literal "beats Fable 5.1" condition — see §0.
