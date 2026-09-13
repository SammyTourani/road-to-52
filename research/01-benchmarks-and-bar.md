# The Bar: Frontier LLM Benchmarks and Claude Fable 5.1, September 2026

**Compiled:** 2026-09-13. All retrieval notes are dated; every number carries a source.
**Method:** primary sources first (lab system cards, lab launch posts, benchmark-owner leaderboards), third-party aggregators only where labelled. Anything I could not verify at a primary or first-party source is marked **UNVERIFIED**.

---

## 0. Executive summary — read this first

Three things changed the shape of "the bar" in 2026, and they matter more than any individual score:

1. **The classic benchmark set is dead at the frontier.** Anthropic's Claude Fable 5.1 system card (212 pages, 1 Sept 2026) mentions **SWE-bench Verified zero times**, along with BrowseComp, tau/tau2/tau3, MCP-Atlas, Vending-Bench, FrontierMath, LiveCodeBench, Aider, MMMU, SimpleQA, LiveBench and HMMT. GPQA and MMLU-Pro appear only as *substrate for an alignment experiment*, not as capability claims. AIME appears not at all. Anthropic said the quiet part out loud in the Sonnet 5 system card: AIME "was a popular AI benchmark last year but is now saturated."
2. **The bar is now agentic, long-horizon, and largely owned by third parties.** What replaced the classics: Terminal-Bench 4.0, Terminal-Bench-Science, SWE-bench Pro, GDPval-AA v2, AA-Briefcase, AutomationBench, OSWorld 2.0, ARC-AGI-2/3, HLE-with-tools, Toolathlon, plus a long tail of vendor benchmarks (Cursor, Cognition, Proximal, Surge, Zapier, Harvey, Databricks, Perplexity).
3. **Harness beats model.** OpenAI demonstrated publicly that flipping two API settings (retained reasoning + compaction) took GPT-5.6 Sol from **13.3% to 38.3%** on the ARC-AGI-3 public set — 3x the score with 6x fewer output tokens, same model. Every headline number in this report is a *model + harness + effort level + grader* tuple. Treat single numbers as meaningless without that tuple.

**Where Fable 5.1 actually stands (1 Sept 2026, `claude-fable-5-1`, $10/$50 per MTok, 1M context):** it is roughly tied for the frontier with OpenAI's GPT-6 Astra (3 Sept 2026). Fable 5.1 wins on knowledge work (GDPval-AA, AA-Briefcase), agentic reliability (Arena Agent Arena #1, Vals Index #1, LiveBench #1, SimpleBench #1), long-horizon coding (Epoch MirrorCode 73.3% vs Astra's 46.7%), long-context reasoning and HLE-with-tools. Astra wins on mathematics (FrontierMath Tier 4 97.6% vs 87.8%), abstract reasoning (ARC-AGI-2/3), computer use, cybersecurity, and business-workflow automation. Terminal-Bench 4.0 is a statistical tie (58.2% ±2.8 vs 57.9% ±3.8). Artificial Analysis has them tied at **53** on Intelligence Index v4.3. Epoch's ECI puts Astra narrowly ahead — **166.6 vs 164.2** — with overlapping confidence intervals, so there is no statistically separable #1.

**Biggest open-vs-closed gap:** on Epoch's ECI the best open-weight model (**Kimi K3, 157.6**) trails Astra (166.6) by ~9 points; on Artificial Analysis the best open model (**GLM-5.3, 45**) trails the 53 frontier by 8. Epoch notes ~5 ECI ≈ one doubling of METR time horizon, which puts open weights roughly **4.5–6 months behind**. But the gap is wildly uneven: **~0 points on GPQA Diamond, ~29 points on ARC-AGI-2**, and on LiveBench's Agentic Coding column an open model (**DeepSeek V4.1-Flash**) currently leads outright.

---

## 1. The benchmark set of September 2026

### 1.1 What labs actually put in their launch posts

Evidence base: the Claude Fable 5.1 & Mythos 5.1 System Card (1 Sep 2026), the Claude Opus 5 System Card (24 Jul 2026), the Claude Sonnet 5 System Card (30 Jun 2026), and OpenAI's GPT-6 Astra launch post (3 Sep 2026).

| Benchmark | Anthropic Fable 5.1 card | Anthropic Opus 5 card | OpenAI Astra post | Verdict |
|---|---|---|---|---|
| Terminal-Bench 4.0 | yes | (as FrontierBench v0.1 / TB 2.1) | yes | **Consensus core** |
| SWE-bench **Pro** | yes | yes | no | Consensus (Anthropic) |
| SWE-bench **Verified** | **no** | yes (96.0%) | no | **Retiring** |
| Humanity's Last Exam | yes | yes | yes | **Consensus core** |
| ARC-AGI-1 / -2 | yes | yes | yes | **Consensus core** |
| ARC-AGI-3 | not available | yes | yes | Consensus core (contested harness) |
| OSWorld 2.0 | yes | yes | yes | **Consensus core** (contested task set) |
| AutomationBench (Zapier) | yes | yes | yes | **Consensus core** |
| GDPval-AA v2 | yes | yes | via AA Index | **Consensus core** |
| AA-Briefcase | yes | yes | via AA Index | Rising |
| Terminal-Bench-Science 0.1 | yes | no | yes | Rising fast |
| DeepSWE v1.1 | yes | yes | yes | **Consensus core** |
| FrontierCode 1.1 (Cognition) | yes | yes | yes | **Consensus core** |
| HealthBench Professional | yes | yes | yes | **Consensus core** |
| BenchCAD | yes | yes | yes | Rising |
| GPQA Diamond | **alignment substrate only** | no | yes | **Saturated** |
| FrontierMath Tier 4 | no | no | yes | Consensus (OpenAI/Epoch) |
| BrowseComp | **no** | yes | yes | **Retiring** |
| AIME / HMMT | **no** | **no** | no | **Retired (saturated)** |
| MMLU-Pro | alignment substrate only | no | no | **Retired (saturated)** |
| LiveCodeBench / Aider Polyglot / MMMU / SimpleQA / LiveBench | no | no | no | **Retired at the frontier** |
| tau2-bench / tau3-Banking | no | no | no | **Retired** (AA dropped tau3 on 7 Sep 2026) |
| MCP-Atlas | no | yes (85.8%) | no | Fading |
| Vending-Bench | no | no | no | **Not used by any 2026 frontier launch** |

Sources: [Fable 5.1 & Mythos 5.1 System Card](https://www-cdn.anthropic.com/0339e6a7c5c7b87f5c07798616dc32c215d14235/Claude%20Fable%205.1%20&%20Claude%20Mythos%205.1%20System%20Card.pdf); [Claude Opus 5 System Card](https://www-cdn.anthropic.com/b514064af1408018e64b1ad24e7d5e75850b4ffd/Claude%20Opus%205%20System%20Card.pdf); [Claude Sonnet 5 System Card](https://www-cdn.anthropic.com/480e0bb54327b9622282e9c39a83a4f490ed377e/Claude%20Sonnet%205%20System%20Card.pdf); [OpenAI GPT-6 Astra](https://openai.com/index/gpt-6-astra). Presence/absence verified by full-text search of the PDFs on 2026-09-13.

> **Direct quote, Claude Sonnet 5 System Card §8.6:** USAMO "is the next step of the math olympiad track in the US after the American Invitational Mathematics Examination (AIME), which was a popular AI benchmark last year but is now saturated."

### 1.2 Benchmark reference table

| Benchmark | What it measures | Who runs it | Saturated? | Current top (source) |
|---|---|---|---|---|
| **SWE-bench Verified** | 500 human-verified GitHub issues, Python | Princeton/Stanford, [swebench.com](https://www.swebench.com/) | **Yes, and contaminated — formally deprecated by OpenAI (Feb 2026)** | Claude Opus 5 **96.0%** (Opus 5 card §8.2); Fable 5 95.0%. Official board's newest entry is **2026-02-26** and contains no H2-2026 model |
| **SWE-bench Pro** | Harder multi-file diffs, live repos, less leakage ([arXiv:2509.16941](https://arxiv.org/abs/2509.16941)) | Scale AI / Princeton | No (~81%) | **Claude Fable 5.1 81.2%** (Fable 5.1 card Table 8.1.A) |
| **SWE-bench Multilingual** | 300 problems, 9 languages | Princeton | Approaching (89%) | Claude Opus 5 89.5%; Fable 5.1 89.1% |
| **SWE-bench Multimodal** | Issues + screenshots/mockups ([arXiv:2410.03859](https://arxiv.org/abs/2410.03859)) | Princeton | No (~59%) | Claude Opus 5 59.4% |
| **Terminal-Bench 4.0** | 66 containerised terminal tasks; sci-adjacent + frontier eng. 8 saturated tasks removed, 20 revised, uniform 8h timeout | Stanford + Harbor + **Laude Institute**, [tbench.ai](https://www.tbench.ai/leaderboard) | No | **Official board: GPT-6 Astra (max/Codex) 58.2% ±2.8, Claude Fable 5.1 (max/Claude Code) 57.9% ±3.8 — a statistical tie.** Opus 5 51.8%, Fable 5 44.5%, best open GLM-5.3 41.8%, Sonnet 5 12.4% |
| **Terminal-Bench-Science 0.1** | 70 real scientific-research workflows, hidden tests | Stanford-led community (MIT, Princeton, UW, Genentech advisors) | No | **GPT-6 Astra 64.6%** (OpenAI); Fable 5.1 52.6% (Anthropic) |
| **GPQA Diamond** | ~198 PhD-level bio/chem/physics MCQs | Originally [Idavidrein/gpqa](https://github.com/idavidrein/gpqa); now run by labs + Epoch | **Yes** — 92.6–96.0% cluster | **GPT-6 Astra 96.0%** (OpenAI table) |
| **Humanity's Last Exam** | 2,500 expert-written questions, multimodal ([arXiv:2501.14249](https://arxiv.org/abs/2501.14249)) | CAIS + Scale AI | No | **Fable 5.1 65.0% with tools** (Anthropic); 60.9% no-tools. AA's own run: Fable 5.1 59%, Astra 55% |
| **ARC-AGI-1** | Fluid intelligence, few-shot grid induction | [ARC Prize Foundation](https://arcprize.org) (semi-private set) | **Yes** (97.5–98.5%) | GPT-6 Astra / Fable 5 **98.5%** |
| **ARC-AGI-2** | Harder ARC, composition + novelty | ARC Prize (semi-private) | Nearly (90–95%) | **GPT-6 Astra 95.0%** (OpenAI); Fable 5.1 90.0% |
| **ARC-AGI-3** | Interactive turn-based games, no instructions; scored by RHAE vs human baseline | ARC Prize (semi-private) | **Contested** | GPT-6 Astra **99.9% on OpenAI's harness**, ~62.7% on the official harness. Opus 5 30.2%. **Fable 5.1: not reported** |
| **AIME 2025/2026** | US high-school olympiad qualifier, short answers | MAA; scored by [MathArena](https://matharena.ai) | **Yes — explicitly retired by Anthropic** | n/a at frontier |
| **USAMO 2026** | 6 proof problems, MathArena grading (rewrite + 3-judge panel, min score) | MathArena methodology | No | Claude **Mythos 5 99.8%**, Opus 4.8 96.7%, Sonnet 5 79.5% (Sonnet 5 card §8.6) |
| **HMMT** | Harvard-MIT math tournament | MathArena | **Yes** (not reported by any 2026 frontier launch) | n/a |
| **FrontierMath Tiers 1–3 / Tier 4 / Erdős / Open Problems** | Research-level maths, private sets. v2 (12 Jun 2026) fixed errors in **42%** of problems. 338 problems: 295 in T1-3, 43 in T4, 12 public | [Epoch AI](https://epoch.ai/frontiermath) (OpenAI funded; has exclusive access to a subset) | Tier 4 now **yes** (97.6%) | **GPT-6 Astra 97.6% Tier 4 (v2)**; Fable 5 90.2%; Fable 5.1 87.8%; Opus 5 73.2% (OpenAI table). Erdős: only Astra has solved any (2/68) |
| **MMLU-Pro** | 12k harder MMLU MCQs | TIGER-Lab | **Yes** | Not reported at the frontier |
| **Global MMLU (GMMLU)** | MMLU across 42 languages | Cohere Labs et al. | Nearly (94%) | Fable 5.1 94.0% |
| **LiveCodeBench** | Rolling contamination-free competitive programming | [livecodebench.github.io](https://livecodebench.github.io/) | **Yes at frontier** | Best open-weight DeepSeek-V4-Pro 93.5% (vendor) |
| **Aider Polyglot** | Multi-language edit-format benchmark | Aider (Paul Gauthier) | **Yes / abandoned** | Not reported by any 2026 frontier launch |
| **tau2-bench → τ³-Banking** | Simulated customer-service tool use | Sierra Research; AA ran τ³ | **Retired** — AA dropped it on 7 Sep 2026, replaced by AutomationBench-AA | n/a |
| **OSWorld 2.0** | 108 long-horizon computer-use tasks on a live Ubuntu VM, weighted checkpoints ([arXiv:2606.29537](https://arxiv.org/abs/2606.29537)) | XLANG Lab, [osworld-v2.xlang.ai](https://osworld-v2.xlang.ai/) | No (strict pass ~42%) | **Disputed** — Anthropic: Fable 5.1 77.9 partial / 41.7 strict on *its own fixed task set*; OpenAI: Astra 72.6 partial on the *official offline set*, Opus 5 70.2 |
| **BrowseComp** | Hard-to-find web facts | OpenAI | Nearly (91%) | GPT-6 Astra 91.5%; Opus 5 90.8%. **Fable 5.1 not reported** |
| **MMMU** | Multimodal college-level reasoning | Multiple universities | **Yes / retired at frontier** | Not reported |
| **SimpleQA / SimpleQA Verified** | Short-fact hallucination | OpenAI / Google; **Epoch runs SimpleQA Verified** | Superseded at the frontier | Epoch's run: GPT-6 Astra **75.6%**, Gemini 3.1 Pro 73.5%, Fable 5.1 70.8%. Largely displaced by **AA-Omniscience** |
| **AA-Omniscience** | Knowledge + hallucination across 42 topics | Artificial Analysis | No | **Fable 5.1 67% accuracy** — "the highest factual accuracy AA has measured" |
| **LiveBench** | Contamination-free rotating benchmark, 23 objective tasks / 7 categories, refreshed every 6 months; **no LLM judge in any subtask** | [livebench.ai](https://livebench.ai) | No | **Claude Fable 5.1 Max 83.4 overall (#1)**; Fable 5 83.0; Astra 82.2. **DeepSeek V4.1-Flash #5 at 81.1 — and #1 on Agentic Coding at 77.3** |
| **MCP-Atlas** | 1,000 tasks (500 public / 500 held out), 36 real MCP servers, 220 tools | **Scale Labs**, [arXiv:2602.00933](https://arxiv.org/abs/2602.00933) | Near-saturated — 10 models within 8.9 pts | Muse Spark 1.1 **88.10**; **Claude Fable 5.1 87.20** (same rank-1 tier); Opus 5 85.80; best open Kimi K3 82.30. Anthropic reported it for Opus 5 (85.8%) but **dropped it from the Fable 5.1 card** |
| **Toolathlon Verified** | 108 tasks, 600+ tools, 32 apps, execution-checked | Toolathlon authors (Jun 2026 verified release) | No | Claude **Opus 5 80.6% Pass@1**; Fable 5.1 77.8% |
| **AutomationBench** | Zapier's private held-out business-workflow set, 47 apps ([arXiv:2604.18934](https://arxiv.org/abs/2604.18934)) | Zapier | No (~41%) | **GPT-6 Astra 41.4%**; Fable 5.1 31.4% |
| **Vending-Bench 2** | Run a vending business for 365 simulated days from $500; scored on final balance | [Andon Labs](https://andonlabs.com/evals/vending-bench-2) | **No** — a competent human is estimated at ~$63k, 4x the best model | **GPT-6 Astra $15,514.70 ±1,074**; Opus 5 $11,181.87; best open GLM-5.2 $8,313.78. **Fable 5.1 charted but not in the top 10.** Not reported by any lab launch post |
| **GDPval / GDPval-AA v2** | 220 tasks from OpenAI's GDPval gold DB, 44 occupations, Elo from blind pairwise ([arXiv:2510.04374](https://arxiv.org/abs/2510.04374)) | OpenAI (dataset); **Artificial Analysis** (harness + Elo) | No | **Fable 5.1 Elo 1853** (Anthropic, max effort); AA's published run 1764 vs Astra 1580 |
| **AA-Briefcase** | Multi-week knowledge projects, thousands of source files, rubric + pairwise panel | Artificial Analysis | No | **Fable 5.1 Elo 1694** (Anthropic) / 1662 (AA run) |
| **METR time horizon** | 50%-success task length | [METR](https://metr.org) | page last updated **2026-05-08** | **No figure for ANY 2026-H2 model.** Best measured: Claude Mythos Preview ~17.4h, Opus 4.6 ~12.0h, GPT-5.6 Sol ~11.3h. See §6.5 |
| **LMArena / Arena (text)** | Crowd pairwise Elo | [arena.ai](https://arena.ai/leaderboard/text) (rebranded from LMArena, Jan 2026) | — | claude-fable-5 **1507±5**; claude-fable-5.1-max 1504±11 |
| **Arena Agent Arena** | Causal per-signal scoring over 1.59M real agent sessions | arena.ai | — | **Claude Fable 5.1 (Max) #1, 13.85% net improvement** |
| **Artificial Analysis Intelligence Index v4.3** | 10-benchmark composite | [artificialanalysis.ai](https://artificialanalysis.ai/methodology/intelligence-benchmarking) | — | **GPT-6 Astra 53 = Claude Fable 5.1 53** |
| **Epoch Capabilities Index (ECI)** | IRT fit over 50+ benchmarks | [Epoch AI](https://epoch.ai/benchmarks) | — | **GPT-6 Astra 169**; Fable 5.1 163 |
| **Scale Labs** (formerly SEAL) | 26 live private/public boards | Scale AI, [labs.scale.com/leaderboard](https://labs.scale.com/leaderboard) | mixed; 18 of 44 sitemap boards frozen | Fable 5.1 is #1 on **SWE Atlas Test Writing (67.04)** and **HiL-Bench (61.50)**; Fable 5 #1 on **EnigmaEval (39.28)** and **Remote Labor Index (15.80%)**; Opus 5 #1 on **SWE Atlas Codebase QnA (63.17)**. ⚠️ Meta holds 49% non-voting of Scale and Muse Spark tops 6 of 26 boards |

### 1.3 Benchmarks that are new or newly standard in 2026

Benchmarks that did not exist (or were not standard) before 2026 and now appear in frontier launch material:

| Benchmark | Owner | What it is |
|---|---|---|
| **Terminal-Bench-Science 0.1** | Stanford-led community | 70 scientific research workflows, hidden per-task tests |
| **AA-Briefcase** | Artificial Analysis | Multi-week knowledge-work projects; rubric + frontier-model pairwise panel |
| **AA-Omniscience** | Artificial Analysis | Knowledge/hallucination index across 42 topics |
| **AA-LCR v1.1** | Artificial Analysis | Long-context reasoning at ~100k tokens |
| **AutomationBench** | Zapier | Private held-out SaaS workflow automation, 47 apps, deterministic assertions |
| **GDP.pdf** | Surge AI | 100 real professional PDFs — finance, health, legal, engineering, insurance |
| **Chartography** | Surge AI | 100 specialist chart types (Kaplan–Meier, Sankey, Bode, wind rose, 3D surface) |
| **CritPt / CritPt-Corrected** | Physics researchers | 71 research-level theoretical physics problems |
| **ArXivMath** | MathArena | Monthly research-maths problems extracted from fresh arXiv abstracts |
| **ProgramBench** | [arXiv:2605.03546](https://arxiv.org/abs/2605.03546) | Rebuild a codebase from a compiled binary + docs; 247k behavioural tests |
| **FrontierCode 1.1** | Cognition | 150 tasks from real PRs; blocking unit tests + weighted code-quality rubric |
| **FrontierSWE v2** | Proximal | 34 ultra-long-horizon tasks; strong models work ~20 hours each |
| **DeepSWE v1.1** | — | 113 long-horizon SWE tasks written from scratch to avoid contamination |
| **CursorBench 3.2.0** | Cursor | Real Cursor traffic, run end-to-end in Cursor's production harness |
| **Toolathlon (Verified)** | Toolathlon authors | 108 tasks / 600+ tools / 32 apps, execution-based checkers |
| **OfficeQA / OfficeQA Pro** | Databricks | Grounded numerical reasoning over US Treasury Bulletin corpora |
| **Legal Agent Benchmark (LAB)** | Harvey AI | 1,200+ legal tasks, 24 practice areas; all-criteria-pass scoring |
| **DRACO** | Perplexity ([arXiv:2602.11685](https://arxiv.org/abs/2602.11685)) | 100 deep-research tasks graded on accuracy, depth, presentation, citations |
| **BenchCAD** | [arXiv:2605.10865](https://arxiv.org/abs/2605.10865) | Programmatic CAD (CadQuery) from multi-view renders; voxel IoU |
| **Agents' Last Exam** | OpenAI | Complex professional tasks inside real software |
| **ScreenSpot-Pro** | — | GUI element grounding, no tools |
| **SRE-Bench** | [arXiv:2608.11469](https://arxiv.org/abs/2608.11469) | Contamination-free binary reverse engineering |
| **MirrorCode**, **EBR-bench**, **Mystery Game Puzzles** | Epoch AI | Long-horizon coding; long-horizon gaming; puzzle-variant game positions |
| **RSI Index**, **Vals Index** | Vals AI | Autonomous LLM R&D (5 tasks); composite across Vals' suite |
| **WeirdML** | Independent | Unusual ML task construction |
| **HealthBench Professional** | OpenAI ([arXiv:2604.27470](https://arxiv.org/abs/2604.27470)) | 525 physician-authored clinical conversations |

### 1.4 The consensus set, counted across all nine 2026 frontier launches

Launches surveyed (all verified to exist, 2026-09-13): OpenAI **GPT-6 Astra** (3 Sep), Anthropic **Claude Fable 5.1 / Mythos 5.1** (1 Sep), Google DeepMind **Gemini 3.8 Flash** (2 Sep), Meta **Muse Spark 1.3** (2 Sep, closed weights), SpaceXAI **Grok 4.6** (12 Aug), DeepSeek **V4.1-Flash** (10 Sep), Alibaba **Qwen3.8-Max** (2 Aug), Moonshot **Kimi K3** (16 Jul), Z.ai **GLM-5.3** (14 Aug).

| Rank | Benchmark | Labs reporting it | Who |
|---|---|---|---|
| 1 | **Terminal-Bench** (2.1 / 3.0 / 4.0 / Science) | **9 of 9** | everyone |
| 2 | **DeepSWE v1.1** | 8 of 9 | all but Anthropic |
| 3= | **Humanity's Last Exam** (full / w-tools / Verified) | 7 of 9 | OpenAI, Anthropic, Google, DeepSeek, Qwen, Kimi, GLM |
| 3= | **AutomationBench** (Zapier) | 7 of 9 | OpenAI, Anthropic, Meta, DeepSeek, Qwen, Kimi, GLM |
| 5= | **GDPval-AA v2** | 6 of 9 | Anthropic, Google, Meta, SpaceXAI, Kimi, GLM |
| 5= | **OSWorld 2.0 / OSWorld-Verified** | 6 of 9 | OpenAI, Anthropic, Google, Meta, Qwen, Kimi |
| 7 | **Agents' Last Exam** | 5 of 9 | OpenAI, DeepSeek, Qwen, Kimi, GLM |
| 8= | **GPQA Diamond** | 4 of 9 | OpenAI, DeepSeek, Qwen, Kimi |
| 8= | **MRCR v2** (long context) | 4 of 9 | OpenAI, Google, Meta, Qwen |
| 10 | CyberGym · Toolathlon · FrontierSWE · NL2Repo · ProgramBench · JobBench · Harvey LAB · CharXiv | 3 of 9 each | — |
| — | **MMLU · MMLU-Pro · SWE-bench Verified · HumanEval · MATH-500 · LiveCodeBench · AIME · GSM8K · DROP · BBH · Aider** | **0 of 9** | **dead at the frontier** |

**The single cleanest signal in this dataset:** not one of the nine September-wave launch posts reports SWE-bench Verified, MMLU, MMLU-Pro, AIME, LiveCodeBench, HumanEval or MATH-500. GPQA Diamond is the last academic multiple-choice benchmark standing, and it survives mainly at OpenAI and the Chinese open-weight labs — Anthropic, Google, Meta and SpaceXAI have all dropped it.

### 1.5 Model-existence corrections (things widely assumed that are false)

| Claim | Status | Evidence |
|---|---|---|
| "Grok 5" | **Does not exist.** `x.ai/news/grok-5` returns HTTP 404. Only statement: "Grok 5 is currently in training" ([Series E post](https://x.ai/news/series-e), 6 Jan 2026). Grok 4.7 also 404s. Newest is **Grok 4.6** (12 Aug 2026) | verified 2026-09-13 |
| "xAI" | Now **SpaceXAI** — SpaceX acquired xAI, [2 Feb 2026](https://x.ai/news/xai-joins-spacex) | verified |
| "Gemini 4" / "Gemini 3.5 Pro" | **Neither shipped.** Google's flagship Pro is still **Gemini 3.1 Pro (Feb 2026)**; Google shipped four Flash models in 106 days instead ([Fortune, 3 Sep 2026](https://fortune.com/2026/09/03/google-shipped-four-gemini-flash-models-in-106-days-but-its-flagship-frontier-model-is-still-nowhere-to-be-seen/)) | verified |
| "New open-weight Llama in 2026" | **None.** Llama 4 (Apr 2025) remains Meta's last open release. Meta's frontier line is now closed-weight **Muse Spark** from Meta Superintelligence Labs | verified |
| "GPT-5.6 Sol is OpenAI's newest" | Real (9 Jul 2026) but **superseded** by GPT-6 Astra (3 Sep 2026). Anthropic's Fable 5.1 system card, published 1 Sep, therefore benchmarks against Sol, not Astra | verified |
| "Terminal-Bench 3.0" vs "FrontierBench" | **Same benchmark.** TB 3.0 was formerly published as FrontierBench — which is why the Claude Opus 5 card lists "FrontierBench v0.1, a successor to Terminal-Bench 2.1 developed by the same team" | Opus 5 card §8.5 + Harbor |

---

## 2. Claude Fable 5.1 — every published number

`claude-fable-5-1`, released **1 September 2026**. 1M token context, 128k max output, **$10 / $50** per MTok, cache reads **$0.25**/MTok (a 75% cut, worth ~25% off typical workloads and ~45% on highly agentic ones). Adaptive thinking is **always on** — `thinking: {"type":"disabled"}` is unsupported; depth is controlled by the `effort` parameter (low / medium / high / xhigh / max). Raw chain-of-thought is never returned. Knowledge cutoff **June 2026**. Claude Mythos 5.1 is the **same weights** with fewer safeguards.

**Standard eval configuration for every Anthropic number below unless stated:** adaptive thinking at **max effort**, default sampling (temperature, top_p), **averaged over 5 trials**, context ≤1M tokens. Crucially, **Fable 5.1 was evaluated with production safeguards enabled** — where a classifier fired, the model scored **zero** on OSWorld 2.0 and AutomationBench. Anthropic states plainly: *"This likely reduces the performance of Fable 5.1."*

### 2.1 Anthropic's own headline table (System Card Table 8.1.A)

| Benchmark | **Fable 5.1 / Mythos 5.1** | Fable 5 / Mythos 5 | Opus 5 | GPT-5.6 Sol |
|---|---|---|---|---|
| SWE-bench Pro | **81.2** | 80 | 79.2 | 64.6 |
| SWE-bench Multilingual | 89.1 | 86.6 | **89.5** | — |
| SWE-bench Multimodal | 54.7 | 54.1 | **59.4** | — |
| Terminal-Bench 4.0 | 56% *(Mythos 5.1: **61%**)* | 42% *(45%)* | 52% | 37% |
| Terminal-Bench-Science 0.1 | **52.6%** | 24.7% | 29.0% | 22.4% |
| Humanity's Last Exam — no tools | **60.9%** | 57.8% | 56.6% | — |
| Humanity's Last Exam — with tools | **65.0%** | 63.8% | 63.6% | — |
| OSWorld 2.0 (partial / strict) | **77.9 / 41.7** | 72.9 / 36.1 | 75.4 / 39.6 | — |
| HealthBench Professional (length-adj.) | 62.1% | **63.3%** | 59.8% | — |
| GDPval-AA v2 (Elo) | **1853** | 1723 | 1824 | 1711 |
| AA-Briefcase (Elo) | **1694** | 1572 | 1685 | 1502 |
| AutomationBench | **31.4** | 17.1 | 26.9 | 19.6 |
| ARC-AGI-1 | 97.5% | **98.5%** | 97.5% | 96.5% |
| ARC-AGI-2 | 90.0% | 89.2% | 90.42% | **92.5%** |

Source: [Claude Fable 5.1 & Mythos 5.1 System Card](https://www-cdn.anthropic.com/0339e6a7c5c7b87f5c07798616dc32c215d14235/Claude%20Fable%205.1%20&%20Claude%20Mythos%205.1%20System%20Card.pdf) Table 8.1.A, p.167. Retrieved and text-extracted 2026-09-13.

### 2.2 Anthropic's detailed sections (System Card §8)

| Benchmark | Fable 5.1 | Fable 5 | Opus 5 | Sonnet 5 | Conditions |
|---|---|---|---|---|---|
| **SWE-bench Pro** | 81.2% | 80% | 79.2% | 63.2% | avg 5 trials |
| **SWE-bench Multilingual** | 89.1% | 86.6% | 89.5% | 78.3% | 300 problems, 9 langs |
| **SWE-bench Multimodal** | 54.7% | 54.1% | 59.4% | 28.1% | internal harness |
| **DeepSWE v1.1** | 67.4% | 69.7% | 68.8% | — | 113 tasks, avg 5 trials |
| **FrontierCode 1.1 Extended** | 63.6% @ **medium** | 64.9% @ xhigh | 63.6% | — | Cognition; peaks at medium — see note |
| **FrontierCode 1.1 Main** | 50.9% @ medium | 53.5% @ xhigh | 53.4% | — | ditto |
| **FrontierSWE v2** (Proximal) | **0.57** | 0.48 | 0.52 | — | 34 ultra-long tasks (~20h each), max effort, 5 trials, Proximal's harness. Sol 0.32. Median task 0.56 vs Fable 5's 0.41; outright failure rate 5% (vs Opus 5 6%, Fable 5 8%) |
| **Terminal-Bench 4.0** | **55.8%** | 42.0% | 52.3% | — | Claude Code `--bare`, max thinking, 15 trials/task (990 trials); SE ±1.6–2.0. Mythos 5.1 **60.9%** (10 trials). Public LB: Opus 5 51.8, Fable 5 44.5, Sol 37.3 (Codex CLI) |
| **Terminal-Bench-Science 0.1** | **52.6%** | 24.7% | 29.0% | — | 10 trials/task (700 trials); SE ±3.5–4.5. Strongly bimodal: ⅔ of tasks solved ≥80% or ≤20% of the time |
| **CursorBench 3.2.0** | **73.4%** @ max | 70.5% | 70.0% | — | measured and reported **independently by Cursor**. At medium: 68.0% for $3.53/task vs Sol max 67.2% for $5.69 |
| **CritPt-Corrected** | 88.4% | 85.5% | — | — | 71 physics problems (31 statements expert-corrected); mean Pass@1 over **16 attempts**, max effort **with tools**, Opus 4.8 judge. Figure-only in card; value via Zvi |
| **ArXivMath (June 2026)** | Mythos 5.1 **91.33%** no tools / **93.88%** with tools | — | — | — | 49 problems, max effort, 4 runs. Sol 86.73%, Gemini 3.1 Pro Preview 65.99% (MathArena LB). **Anthropic discloses possible contamination** — problems drawn from June 2026 arXiv abstracts, within training window |
| **ProgramBench** | **87.6%** | 86.3% | 85.4% | — | 166 "golden" tasks of 200, mini-swe-agent harness, no 6h limit |
| **OfficeQA / OfficeQA Pro** | **80.2% / 69.0%** | Mythos 5: 79.0 / 67.1 | 78.1 / 66.9 | — | Databricks; extracted text + code exec. **Harness-sensitive**: Databricks' own run of Fable 5 with PDFs-as-images gives 57.9% on Pro |
| **Legal Agent Benchmark (Harvey)** | **19.09%** all-pass (±0.92, n=5), 90.81% criterion-pass | — | 23.58% all-pass, 93.74% criterion | Sonnet 5: 8.9% (public) / 5.8% (held-out) | Held-out 120-problem set is **run by Artificial Analysis**: Fable 5.1 16.7% all-pass / 93.3% criterion @ xhigh |
| **GDPval-AA v2** | **1853** @ max, **1835** @ xhigh (top two leaderboard spots) | 1723 | 1824 | 1618 | 220 tasks / 44 occupations; Elo from blind pairwise. Run independently by Artificial Analysis. xhigh matches max within CI using ~25% fewer output tokens |
| **AA-Briefcase** | **1694** @ max, 1686 @ xhigh, 1611 @ high | 1572 | 1685 | — | vs Opus 5: wins rubric pass (61.5 vs 57.2) and analytical quality (2025 vs 1980), **loses presentation (1495 vs 1572)**. At high effort it still beats every non-Claude model using 47% fewer tokens |
| **Toolathlon Verified** | 77.8% Pass@1 / 81.5 Pass@3 / 73.1 Pass³ / 23.7 turns | Mythos 5: 79.3 | **80.6** | 74.7 | 108 tasks, 600+ tools, 32 apps, 3 trials. **Fable 5.1 run with classifiers + fallback ON; comparators OFF.** 11 of 324 trials hit a refusal; 4 more terminated and counted as failures |
| **AutomationBench** | **31.4%** @ max | 17.05% | 26.9% | 13.5% | Zapier's **private held-out** set |
| **Chartography** (Surge AI) | 42.6% no tools / **86.2%** with tools | 36.6 / 84.2 | 29.6 / 83.0 | — | 100 specialist chart types; expert-set acceptable ranges |
| **BenchCAD Vision2Code** (voxel IoU) | 0.437 no tools / **0.843** with tools | 0.376 / 0.675 | 0.366 / 0.821 | — | 1,000-file subset of 17,900; 3 documented modifications to the reference implementation |
| **GDP.pdf** (Surge AI) | 85.4% no tools / 85.1% with tools | 82.7 / **87.1** | 83.4 / 85.5 | — | 100 professional PDFs, mean criteria pass rate, Opus 4.7 judge |
| **OSWorld 2.0** | **77.9 partial / 41.7 strict** | 72.9 / 36.1 | 75.4 / 39.6 | — | 108 tasks, 1080p, ≤500 steps, Opus 4.8 grader, Pass@1 over 5 runs. Authors' Aug-2026 release **plus Anthropic's own task/grading fixes**. **Explicitly supersedes and is not comparable to** the Opus 5 card's OSWorld figures |
| **HealthBench** | 66.7% raw / 60% length-adj | 61.2 raw | **67.1** raw | 59.2 | Opus 4.8 grader, 5 trials, no tools, no custom system prompt |
| **HealthBench Professional** | **74.2%** raw / 62.1 length-adj | 68.9 | 73.4 | 62.4 | as above |
| **GMMLU** (42 languages) | **94.0%** | 93.6% | 92.5% | 89.0% | single trial |
| **MILU** (11 Indic languages) | **93.0%** | 92.9% | 92.1% | 89.3% | 5 trials |
| **BioMysteryBench** human-solvable | Mythos 5.1 90.3% | Mythos 5 90.1% | **91.4%** | 84.9% | Sol 86.1, Gemini 3.1 Pro 83.6 |
| **BioMysteryBench** human-difficult | Mythos 5.1 44.1% | Mythos 5 44.7% | **51.8%** | 39.4% | Gemini 3.1 Pro 32.9, Sol 28.8 |
| **ARC-AGI-1** | 97.5% | 98.5% | 97.5% | — | ARC Prize **semi-private** set, max effort, verified by ARC Prize |
| **ARC-AGI-2** | 90.0% | 89.2% | 90.42% | — | ditto. (Opus 4.7 was 75.83%) |
| **ARC-AGI-3** | **NOT REPORTED** | — | 30.16% RHAE @ high | — | Card: "ARC-AGI-3 results were not available at the time of release." Zvi attributes this to the API misclassifying requests |

**Why FrontierCode regressed.** Anthropic is unusually candid: FrontierCode penalises *any* change to a file outside the task's scope, even a correct one. Fable 5.1 at high/xhigh/max "occasionally adds more small, unrequested changes in files outside the task, such as a documentation comment in an adjacent file, an edit to a docs page, or a new CI job." Its task-correctness pass rate keeps rising with effort; its *score* peaks at medium. Adding a brevity instruction reduced out-of-scope edits, but Anthropic reports the unmodified numbers. **OpenAI exploited exactly this gap** — GPT-6 Astra's FrontierCode runs used a Codex-style developer message instructing "avoid unrelated cleanup… clean, mergeable code" (OpenAI footnote 8), and scored 64.5% Extended / 53.3% Main.

### 2.3 OpenAI's independently-run numbers for Fable 5.1

From the [GPT-6 Astra launch post](https://openai.com/index/gpt-6-astra) (3 Sep 2026). This is the competitor's view of the same model, and in several rows it **disagrees with Anthropic**.

| Benchmark | GPT-6 Astra | **Fable 5.1** | Fable 5 | Opus 5 | Anthropic's own figure for Fable 5.1 | Delta |
|---|---|---|---|---|---|---|
| Terminal-Bench 4.0 | **57.9%** | 55.8% | 44.5% | 52.6% | 55.8% | agrees |
| Terminal-Bench Science 0.1 | **64.6%** | 52.6% | 21.4% | 30.0% | 52.6% | agrees |
| DeepSWE v1.1 | **74.1%** | 67.4% | 69.9% | 73.7% | 67.4% | agrees |
| FrontierCode 1.1 Extended | **64.5%** | 63.6% | 64.9% | 63.6% | 63.6% | agrees |
| FrontierCode 1.1 Main | **53.3%** | 50.9% | 53.5% | 53.4% | 50.9% | agrees |
| AutomationBench | **41.4%** | 31.4% | 17.4% | 26.9% | 31.4% | agrees |
| BenchCAD | **95.9%** | 84.3% | 67.5% | 82.1% | 0.843 IoU | agrees (OpenAI flags 3 Anthropic eval modifications) |
| FrontierMath Tier 4 (v2) | **97.6%** | 87.8% | 90.2% | 73.2% | not reported | **new info** — Fable 5 > Fable 5.1 |
| GPQA Diamond | **96.0%** | 93.7% | 92.6% | 93.7% | not reported | **new info** |
| HLE (with tools) | 57.2% | **65.0%** | 63.8% | 63.6% | 65.0% | agrees — **Fable 5.1 wins** |
| ARC-AGI-2 | **95.0%** | 90.0% | 89.2% | 90.4% | 90.0% | agrees |
| ARC-AGI-1 | **98.5%** | 97.5% | 98.5% | 97.5% | 97.5% | agrees |
| ARC-AGI-3 | **99.9%** (OpenAI harness) | — | — | 30.2% | not reported | see §5.1 |
| OSWorld 2.0 (official offline set) | **72.6%** | — | — | **70.2%** | Opus 5 = 75.4 | **DISPUTED** — differs by 5.2 pts |
| HealthBench Professional (len-adj) | **63.4%** | **58.1%** | 60.9% | 56.4% | **62.1%** | **DISPUTED** — differs by 4.0 pts |
| ExploitGym | **42.4%** | 30.4%* | 28.4%* | 22.0% | not reported | *these are **Mythos**, not Fable |
| ScreenSpot-Pro (no tools) | **92.7%** | — | 87.3%* | — | not reported | *Mythos |
| BrowseComp | **91.5%** | — | 87.4% | 90.8% | not reported | Fable 5.1 unlisted by both labs |
| Agents' Last Exam | **59.3%** | — | 48.7% | 55.5% | not reported | Fable 5.1 unlisted |
| GeneBench Pro / LifeSciBench / MedChemBench | 37.1 / 60.3 / 49.3 | **excluded** | excluded | — | not reported | OpenAI fn12: *"Claude Fable 5 and 5.1 are not included… because they refuse the majority of questions in these evaluations"* |

### 2.4 Third-party / independent evaluations of Fable 5.1

| Evaluator | Metric | Fable 5.1 | Comparators | Source |
|---|---|---|---|---|
| **Artificial Analysis** | Intelligence Index **v4.3** (7 Sep) | **53** | GPT-6 Astra **53**, Opus 5 51, Fable 5 50, Muse Spark 1.3 48, Sol 47 | [AA comparison](https://artificialanalysis.ai/models/comparisons/gpt-6-astra-vs-claude-fable-5-1) |
| Artificial Analysis | Intelligence Index **v4.2** (4 Sep) | 57 | Astra 55, Opus 5 54, Fable 5 53 | AA changelog |
| Artificial Analysis | Intelligence Index **v4.1.1** | **65.7** | Opus 5 63.1, Fable 5 62.1, Astra 61.2, Sol 60.9 | reproduced in OpenAI's own table |
| Artificial Analysis | GDPval-AA v2 (AA's published run) | 1764 | Astra 1580 | AA comparison — **note Anthropic's own run says 1853** |
| Artificial Analysis | AA-Briefcase | 1662 | Astra 1562 | ditto — Anthropic's run says 1694 |
| Artificial Analysis | AutomationBench-AA | 59% | Astra 68% | ditto |
| Artificial Analysis | Terminal-Bench v4.0 | 52% (55.1% xhigh on AA's TB page) | Astra 59% (59.6% xhigh) | ditto |
| Artificial Analysis | SciCode | **63%** | Astra 56% | ditto |
| Artificial Analysis | HLE | **59%** | Astra 55% | ditto |
| Artificial Analysis | GDP.pdf | 26% | Astra 31% | ditto |
| Artificial Analysis | CritPt | 30% | Astra 32% | ditto |
| Artificial Analysis | AA-Omniscience (index) | 43 | Astra 43 | ditto |
| Artificial Analysis | AA-Omniscience (accuracy) | **67%** — "highest factual accuracy AA has measured" | Fable 5 65% | felloai / AA |
| Artificial Analysis | AA-LCR v1.1 (long context) | **85%** | Astra 81% | AA comparison |
| Artificial Analysis | Coding Agent Index | 70 | Astra 67, Opus 5 67 | felloai citing AA |
| **Epoch AI** | Epoch Capabilities Index (6 Sep) | **163** | **GPT-6 Astra 169**, Fable 5 163, Opus 5 162, Sol 162, **Kimi K3 158** (best open), DeepSeek V4 Pro 155 | [Epoch AI on X](https://x.com/EpochAIResearch/status/2095602754282783108) |
| Epoch AI | FrontierMath Erdős | **0 solved** | Astra solved 2 of 68 (only model to solve any) | via Zvi |
| Epoch AI | MirrorCode (long-horizon coding) | not listed | **Fable 5 leads at 64%**, Sol 20%; Astra ranks between Opus 4.7 and Fable 5 | [epoch.ai/benchmarks](https://epoch.ai/benchmarks), 3 Aug 2026 |
| **Arena** (ex-LMArena) | Text Arena Elo | **1504 ± 11** (2,906 votes), rank 3 | claude-fable-5 1507±5, claude-opus-4-6-high 1505±4 | [arena.ai/leaderboard/text](https://arena.ai/leaderboard/text) |
| **Arena** | **Agent Arena** (1.59M sessions, 9 Sep) | **#1 — 13.85% ± 1.92 net improvement**, $4.50/task | Astra (Max) 12.39%, Opus 5 (High) 11.06%, **Kimi K3 (Max) 6.46% at $0.80/task** | [arena.ai/leaderboard/agent](https://arena.ai/leaderboard/agent) |
| Arena | Agent Arena — "confirmed success" signal | **#1, 22.39%** | Astra 18.82%, **Kimi K3 15.11%** (ahead of Opus 5) | ditto |
| **Vals AI** | Vals Index | **68.83% ± 1.08 — rank 1 of 56**, $28.92/test, 76m16s latency | Opus 5 67.2%, Astra 66.6% | [vals.ai](https://www.vals.ai/models/anthropic_claude-fable-5-1) |
| Vals AI | **RSI Index** (autonomous LLM R&D, 5 tasks) | **#1 — 35.03%**, $1,481 | Opus 5 32.10%, $1,886. Leads 4 of 5 tasks at 21% lower cost | Vals AI, 4 Sep 2026 |
| Vals AI | category ranks | EMB 1/56 · MedScribe 1/93 · ProofBench v1.1 1/30 · MortgageTax 2/98 · Public Benefits 2/34 · Legal Research 3/59 · Code Migration 4/58 · Finance Agent v2 6/59 · MedCode 7/91 · **SAGE 25/81** | — | ditto |
| **ARC Prize** | ARC-AGI-1 / -2 (semi-private) | 97.5% / 90.0% | verified by ARC Prize | Fable 5.1 card §8.16 |
| **ARC Prize** | ARC-AGI-3 | **no score** | — | see §5.1 |
| **METR** | 50% time horizon | **NOT PUBLISHED** | — | see §2.5 |
| **Epoch AI** | FrontierMath Tiers 1–3 (v2) | **90.2%** — rank 2 | GPT-6 Astra 93.7%, Sol 89.1%, Fable 5 87.0%, Opus 5 85.6% | [epoch.ai benchmarks CSV](https://epoch.ai/data/benchmarks.csv) |
| Epoch AI | FrontierMath Tier 4 (v2) | 87.8% — rank 3 | Astra **97.6%**, **Fable 5 90.2%** (Fable 5 beats Fable 5.1), Sol 82.9%, Opus 5 73.2% | ditto |
| Epoch AI | **MirrorCode** (long-horizon coding) | **73.3% — #1** | Fable 5 63.9%, **GPT-6 Astra only 46.7%**, Opus 4.7 31.1%, Sol 20.0% | ditto |
| Epoch AI | SimpleQA Verified | 70.8% | Astra **75.6%**, Gemini 3.1 Pro 73.5%, Fable 5 70.7% | ditto |
| Epoch AI | EBR-bench (long-horizon gaming) | 57.1% | **Astra 76.2%**, Opus 5 45.7% | ditto |
| Epoch AI | Mystery Game Puzzles | 58.0% | **Astra 84.0%**, Opus 5 59.0% | ditto |
| Epoch AI | GPQA Diamond (Epoch's own run) | not listed | Astra **95.8%**, Gemini 3.8 Flash 95.4%, Opus 5 93.9% | ditto |
| **Scale Labs** | SWE Atlas — Test Writing | **67.04 ±5.33 — #1** | (Claude Code xHigh) | [labs.scale.com/leaderboard](https://labs.scale.com/leaderboard) |
| Scale Labs | **HiL-Bench** (knowing when to escalate to a human) | **61.50 ±6.47 — #1** | — | ditto |
| Scale Labs | SWE Atlas — Refactoring | 56.67 — #2 | GPT-6 Astra (Codex) 59.05 | ditto |
| Scale Labs | Humanity's Last Exam | 46.50 — #2 (**best calibration error, 20**) | GPT-6 Astra **54.80 ±1.94** | ditto |
| Scale Labs | MCP-Atlas | 87.20 ±2.05 (rank-1 tier) | Muse Spark 1.1 88.10 | ditto |
| Scale Labs | Remote Labor Index (real paid Upwork projects) | not listed | **Claude Fable 5 15.80% — #1** | ditto |
| **SimpleBench** | AVG@5 | **86.6% — #1**, above the **human baseline of 83.7%** | GPT-6 Astra Pro 86.5%, Astra 83.6%, Opus 5 80.6%, Sonnet 5 60.6% | [simple-bench.com](https://simple-bench.com/) |
| **LiveBench** | Overall | **83.4 — #1** (Reasoning 91.7 / Coding 86.4 / Agentic Coding 66.1 / Math 97.0), $1.212/task | Fable 5 83.0, Astra 82.2, **DeepSeek V4.1-Flash 81.1 and #1 on Agentic Coding at 77.3 for $0.029** | [livebench.ai](https://livebench.ai/) |
| **Andon Labs** | Vending-Bench 2 | **not in the top 10** (charted only) | Astra $15,514.70, Opus 5 $11,181.87, best open GLM-5.2 $8,313.78 | [andonlabs.com](https://andonlabs.com/evals/vending-bench-2) |
| **Vals AI** | Terminal-Bench Science 0.1 (one fixed harness, pass@1) | **34.29%** | GPT-6 Astra 65.71%. ⚠️ **Anthropic self-reports 52.6%** — see §6.1 | [vals.ai](https://www.vals.ai/benchmarks/terminal-bench-science) |
| **Vals AI** | fallback-adjusted scores (counting refusal-fallback tasks as failures) | Vals Index 67.87 → **66.85** · Terminal-Bench 2.1 85.02 → **79.03** · SRE Bench 22.90 → **10.69** (195 of 262 tasks fallback-assisted) | — | ditto — see §6.4 |
| Vals AI | Harvey Legal Agent Benchmark | **6.67% — #18 of 55**, below Fable 5's 11.25% | — | ditto |
| Independent | WeirdML | 92.3% | Fable 5 91.9% | via Zvi |

### 2.5 What is explicitly NOT available for Fable 5.1

These are absences I confirmed by full-text search of the system card and by checking the relevant leaderboard — not gaps in my search:

| Benchmark | Status |
|---|---|
| **SWE-bench Verified** | **No Fable 5.1 number exists from Anthropic.** Zero mentions in the 212-page system card; §8.2 is retitled "SWE-bench Pro, Multilingual, and Multimodal". Third-party sites reporting "Fable 5.1 = 95%" are **recycling Fable 5's June figure** |
| **ARC-AGI-3** | Not available at release (system card §8.16). No ARC Prize verified score |
| **METR 50% time horizon** | **Never published.** METR did a *qualitative* pre-deployment assessment of Mythos 5.1 using Sunlight, Budget NanoGPT Speedrun and LMCA. Anthropic gave METR scores on **212 benchmarks**. METR's conclusions: "likely more capable than current public models"; sub-expert on the judgement-heavy tasks; "likely unable to fully and reliably automate R&D for frontier projects spanning multiple weeks." No time-horizon number |
| **GPQA Diamond** (Anthropic-run) | Not reported. OpenAI's independent run gives **93.7%** |
| **AIME 2025 / 2026, HMMT** | Not reported by Anthropic for any 5-series model. Declared saturated |
| **MMLU-Pro, MMMU, SimpleQA, LiveCodeBench, Aider Polyglot, LiveBench, tau2-bench, MCP-Atlas, Vending-Bench, FrontierMath (Anthropic-run)** | Not reported |
| **BrowseComp** | Reported for Opus 5 (90.8) and Fable 5 (87.4) but **not Fable 5.1** by either Anthropic or OpenAI |
| **Agents' Last Exam** | Reported by OpenAI for Fable 5 (48.7) and Opus 5 (55.5), **not Fable 5.1** |
| **Scale SEAL / SWE-Bench Pro boards** | SEAL is now **Scale Labs** ([labs.scale.com/leaderboard](https://labs.scale.com/leaderboard)) and Fable 5.1 *is* listed on several boards — but **neither the SWE-Bench Pro public nor private board has a Fable 5.1, Opus 5 or GPT-6 Astra entry yet**. "SEAL Showdown" (the LMArena rival) now returns HTTP 404 and appears dead |

### 2.6 Anthropic's AECI and the internal R&D picture

Anthropic maintains **AECI**, a fork of Epoch's Capabilities Index powered by *internal* benchmark results. It is **not comparable** to Epoch's public ECI.

| Model | AECI | 95% CI | n benchmarks |
|---|---|---|---|
| Claude Mythos 5.1 | **161.98** | 158.20 – 169.00 | 46 |
| Claude Opus 5 | 160.73 | 157.35 – 167.11 | 64 |
| Claude Mythos 5 | 159.46 | 156.30 – 165.46 | 97 |

Claude Sonnet 3.5 (June 2024) anchors the scale at 130. Anthropic's reading: Mythos 5.1 sits above the historical trend line "by a margin similar to that of other recent Mythos-class models… the capability jump of Mythos Preview was a one-time event that shifted the entire trend line upward, rather than a permanent accelerant."

**CoBench** (internal, diagnose root causes of real Anthropic engineering issues from a historical codebase snapshot): Mythos 5.1 scores **slightly worse than Opus 5**, better than Mythos 5. Anthropic's stated threshold for a model that could fully substitute for Anthropic research staff is **≥85%**, and Mythos 5.1 is well below it.

---

## 3. Top open-weight models and the gap to Fable 5.1

### 3.1 Who the top open-weight models actually are (September 2026)

| Rank | Model | Lab | Released | Params (active) | License | AA Index v4.3 | Epoch ECI |
|---|---|---|---|---|---|---|---|
| **1** | **GLM-5.3** | Z.ai (Zhipu) | 14 Aug 2026 | 753B (40B), 1M ctx | **MIT** | **45** | (GLM-5.2: 151.9) |
| **2** | **Kimi K3** | Moonshot | 16 Jul 2026 | 2.8T (104B), 1M ctx, MXFP4 QAT | Modified MIT (Kimi K3 licence) | 44 | **157.6** (best open on ECI, rank 11) |
| **3** | **DeepSeek V4.1-Flash** | DeepSeek | 10 Sep 2026 | 552B (8B in / 16B out), 1M ctx | **MIT** | 40 | (V4 Pro 0813: 155.4) |
| 4 | Qwen3.8-Max | Alibaba | 2 Aug 2026 | 2.4T (95B) | ⚠️ Epoch labels this **closed weights** — do not cite as open | 40 | 156.6 |
| 5 | MiniMax M3 | MiniMax | 1 Jun 2026 | 428B (23B), multimodal, 1M | MiniMax Community (mod. MIT) | 30 | — |
| 6 | K2 Horizon 375B-A23B | MBZUAI IFM | 3 Sep 2026 | 375B (23B) | Apache 2.0 | 31 | — |
| 7 | **Inkling** | Thinking Machines | 15 Jul 2026 | 975B (41B) | **Apache 2.0** | 26 | — (best **US** open-weight) |
| 8 | MiMo-V2.5-Pro | Xiaomi | 2026 | 1T (42B) | MIT | 26 | — |
| 9 | Ling 3.0 Flash | Ant Group | 2026 | 124B (5.1B) | MIT | 25 | — |
| 10 | Nemotron 3 Ultra | NVIDIA | 4 Jun 2026 | 550B (55B), Mamba-Transformer | OpenMDW-1.1 | 23 | — |
| — | Tencent Hy4-preview | Tencent | 28 Aug 2026 | 770B (49B) | Apache 2.0 | not yet rated | — |
| — | Muse-Glimmer-30B | Meta | 10 Aug 2026 | ~29.6B dense + 1.8B vision | Apache 2.0 | 18 | — |
| — | Gemma 4 (31B / 26B-A4B) | Google | 2 Apr 2026 | 31B dense / 25.2B (3.8B) | **Apache 2.0** (licence flipped) | 15 / 17 | — |
| — | Mistral Medium 3.5 | Mistral | Q2 2026 | 128B dense | Modified MIT | 15 | — |
| — | gpt-oss-120b / 20b | OpenAI | 5 Aug 2025 | 117B (5.1B) / 21B (3.6B) | Apache 2.0 | 12 / 10 | — |
| — | Olmo 3.1 32B Think | Ai2 | 12 Dec 2025 | 32B dense, fully open data | Apache 2.0 | 7 | — |

Sources: [AA open-weights view](https://artificialanalysis.ai/models/open-source); [Epoch ECI CSV](https://epoch.ai/data/eci_scores.csv) (updated 13 Sep 2026); individual HuggingFace model cards. Retrieved 2026-09-13.

**Debunked during research:** there is no Llama 5, no 2026 open-weight Llama at all (Llama 4, Apr 2025, is Meta's last — Meta's frontier line is now closed-weight Muse Spark), no gpt-oss successor, no OLMo 4, no ERNIE 5 weights, no ByteDance Seed 2 weights.

### 3.2 Head-to-head: Fable 5.1 vs the best open-weight models

| Benchmark | Fable 5.1 | Best open-weight | Who | **Gap** |
|---|---|---|---|---|
| **Artificial Analysis Intelligence Index v4.3** | 53 | **45** | GLM-5.3 (max) | **8 points** |
| **Epoch Capabilities Index** | 164.2 | **157.6** | Kimi K3 | **6.6 points** (Epoch: ~5 ECI ≈ one doubling of METR time horizon → open is ~4.5–6 months behind) |
| **Terminal-Bench 4.0** (official board) | 57.9% | **41.8%** | GLM-5.3 (max) | **16.1 pts** |
| **Terminal-Bench 2.1** (Vals harness) | 85.02% | **74.53%** | DeepSeek V4.1-Flash | 10.5 pts |
| **Terminal-Bench 2.1** (vendor self-reported) | — | **90.6%** | DeepSeek V4.1-Flash | open **leads** the vendor-reported view — see §6 |
| **Arena Text Elo** (style-controlled) | 1501 ±8 | **1486 ±7** | glm-5.3-max | **~20 Elo** |
| **Arena WebDev Elo** | 1758 | **1674** | kimi-k3-max | 84 Elo (Astra leads at 1800) |
| **Arena Agent Arena** (net improvement) | **13.85%** | **6.46%** | Kimi K3 (Max) | **7.4 pts** — but Kimi costs $0.80/task vs $4.50 |
| **LiveBench overall** | **83.4** | **81.1** | DeepSeek V4.1-Flash | **2.3 pts** |
| **LiveBench Agentic Coding** | 66.1 | **77.3** | DeepSeek V4.1-Flash | **open WINS by 11.2 pts**, at ~1/40th the cost |
| **HLE (with tools)** | 65.0% | **63.9%** | DeepSeek V4.1-Flash | **1.1 pts** |
| **HLE (no tools)** | 60.9% | **43.5%** | Kimi K3 | 17.4 pts |
| **GPQA Diamond** | 93.7% (OpenAI's run) | **93.5%** | Kimi K3 | **0.2 pts — statistically tied** |
| **SWE-bench Pro** | 81.2% | **67.7%** | Qwen3.8-Max | 13.5 pts (best true-open: Tencent Hy4 65.7, GLM-5.2 62.1) |
| **ARC-AGI-1** (ARC Prize verified) | 97.5% | **94.5%** | Kimi K3 (Max) | 3 pts |
| **ARC-AGI-2** (ARC Prize verified) | 90.0% | **61.4%** | DeepSeek V4 Flash | **28.6 pts** — the widest gap on any benchmark |
| **AutomationBench** | 31.4% | **54.8%** | DeepSeek V4.1-Flash (vendor) / GLM-5.3 48.2% | **open WINS** (different harnesses — see §6) |
| **BrowseComp** | not reported | **91.2%** | Kimi K3 | open essentially at frontier (Astra 91.5%) |
| **MCP-Atlas** | 87.20 | **82.30** | Kimi K3 (max) | 4.9 pts |
| **OSWorld-Verified** | not listed | **86.1%** | Qwen3.8-Max (vendor) / MiniMax M3 75.2% (board) | open **leads** the vendor view (Fable 5 = 86.0) |
| **OSWorld 2.0** (binary) | 41.7% (Anthropic's set) | **4.6%** | MiniMax M3 | **~27–37 pts** — huge |
| **Vending-Bench 2** | not in top 10 | **$8,313.78** | GLM-5.2 | Astra $15,514.70, Opus 5 $11,181.87 |
| **GDPval-AA v2 (Elo)** | 1764 (AA) / 1853 (Anthropic) | **1686** | Kimi K3 | ~78 Elo |
| **Scale Fortress** (dual-use safety, lower better) | — | **8.24 — #1 outright** | gpt-oss-120b | **open beats every closed model** |

### 3.3 Reading the gap

The honest summary is that **the gap depends entirely on what you measure**:

- **Knowledge & academic QA: essentially closed.** GPQA Diamond 93.5 vs 93.7. HLE-with-tools 63.9 vs 65.0. These benchmarks no longer separate open from closed.
- **Composite indices: ~6–9 points, roughly 4.5–6 months.** Both AA and Epoch agree on the magnitude.
- **Long-horizon agentic work: 7–16 points and *not* closing quickly.** Terminal-Bench 4.0, Agent Arena, OSWorld 2.0 and ARC-AGI-2 are where the frontier still clearly wins.
- **Cost-normalised: open frequently wins outright.** DeepSeek V4 Flash scores 61.4% on ARC-AGI-2 at **$0.042/task** against Fable 5.1's 90.0% at **$4.49/task** — roughly 100x cheaper for two-thirds the score. Kimi K3 places #8 on Agent Arena at $0.80/task vs Fable 5.1's $4.50.

---

## 4. What can actually be run locally (16GB Mac Mini M4, no paid API)

### 4.1 The eight easiest — open questions, open grader, no key, no Docker

| # | Benchmark | Repo | Why it's easy |
|---|---|---|---|
| **1** | **ARC-AGI-1** | [github.com/fchollet/ARC-AGI](https://github.com/fchollet/ARC-AGI) (Apache-2.0) | 400 train + 400 eval JSON grids, ~5 MB. Exact grid match. No Docker, no key, no judge, no gate |
| **2** | **ARC-AGI-2** | [github.com/arcprize/ARC-AGI-2](https://github.com/arcprize/ARC-AGI-2) (Apache-2.0) | 1,000 public training + 120 public eval. Exact match, 2 attempts per test input. Still unsaturated |
| **3** | **MMLU-Pro** | [github.com/TIGER-AI-Lab/MMLU-Pro](https://github.com/TIGER-AI-Lab/MMLU-Pro) (Apache-2.0) | 12k MCQs, 10 options, regex + exact match. `evaluate_from_api.py` points straight at Ollama |
| **4** | **AIME 2025/2026 + HMMT** | [github.com/eth-sri/matharena](https://github.com/eth-sri/matharena) (MIT) | AIME answers are integers 0–999 → pure exact match. AIME 2026 and HMMT Feb 2026 already in-repo. 21 competitions. vLLM + OpenAI-compatible providers |
| **5** | **LiveCodeBench** | [github.com/LiveCodeBench/LiveCodeBench](https://github.com/LiveCodeBench/LiveCodeBench) (MIT) | Grading = run the test cases. No judge. Date-windowed slices give a genuinely contamination-free read |
| **6** | **LiveBench** | [github.com/LiveBench/LiveBench](https://github.com/LiveBench/LiveBench) | Explicitly **no LLM judge in any subtask**. `--api-base` for Ollama. Latest release 2026-06-25 |
| **7** | **GPQA Diamond** | [github.com/idavidrein/gpqa](https://github.com/idavidrein/gpqa) (MIT) | 198 questions, exact match on `ANSWER: X`. Only friction is the free HF gate + terms click |
| **8** | **MMMU (validation)** | [github.com/MMMU-Benchmark/MMMU](https://github.com/MMMU-Benchmark/MMMU) (Apache-2.0) | 900 val questions with public answers (test answers now released too). Needs a local vision model (`qwen2.5-vl`, `llava`) |

**Fastest single win:** [EleutherAI/lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) (MIT) covers GPQA, MMLU-Pro, AIME, ARC, MATH, AGIEval, IFEval in one install and speaks `local-completions` to any OpenAI-compatible server. It does **not** include HLE, LiveCodeBench, or SimpleQA.

**Honourable mentions** (one caveat each): **SimpleQA Verified** — [HF `google/simpleqa-verified`](https://huggingface.co/datasets/google/simpleqa-verified), 1,000 rows, MIT, ungated, but needs an autorater. **HLE** — 2,500 public questions ([github.com/centerforaisafety/hle](https://github.com/centerforaisafety/hle)) but the default judge is `o3-mini`; a local judge works mechanically, scores won't be leaderboard-comparable. **BrowseComp-Plus** — [github.com/texttron/BrowseComp-Plus](https://github.com/texttron/BrowseComp-Plus), 830 queries over a **fixed offline 100,195-doc corpus** with prebuilt BM25 indexes: the only realistic offline substitute for BrowseComp.

### 4.2 Needs an agentic scaffold, Docker, or a VM

| Tier | Benchmark | Blocker |
|---|---|---|
| **Feasible with effort** | **τ³-bench / tau2-bench** — [github.com/sierra-research/tau2-bench](https://github.com/sierra-research/tau2-bench) (MIT) | **No Docker at all.** Python ≥3.12 + LiteLLM, rule-based reward, needs a user-simulator LLM (can be local). **The single best agentic benchmark for this machine.** The original `sierra-research/tau-bench` now carries a deprecation banner |
| | **MCP-Atlas** — [github.com/scaleapi/mcp-atlas](https://github.com/scaleapi/mcp-atlas) (MIT) | Docker 8–10 GB. 1,000 tasks (500 public / 500 private), 36 MCP servers. **20 of 36 servers need no credentials** → a partial public run is realistic. LLM-judge (default Gemini, overridable via `EVAL_LLM_*`) |
| | **ARC-AGI-3** — [github.com/arcprize/arc-agi](https://github.com/arcprize/arc-agi) (MIT) | Has an `OperationMode.OFFLINE` and works with an anonymous key — genuinely zero-cost, just a big scaffold lift |
| **Painful on arm64** | **SWE-bench Verified / Lite / Bash Only** — [github.com/SWE-bench/SWE-bench](https://github.com/SWE-bench/SWE-bench) (MIT) | README: *"We recommend running on an `x86_64` machine with at least 120GB of free storage, 16GB of RAM, and 8 CPU cores… Support for `arm64` machines is experimental."* Images must be rebuilt locally with Buildx |
| | SWE-bench Pro — [github.com/scaleapi/SWE-bench_Pro-os](https://github.com/scaleapi/SWE-bench_Pro-os) (MIT) | 731 public + 276 commercial (private). Prebuilt images `jefzda/sweap-images`. Supports vLLM via SWE-agent |
| | SWE-bench Multimodal (480) | JS/browser Docker stacks. **Correction to a common belief:** it is no longer sb-cli-gated — the test split and tooling are open source |
| | **Terminal-Bench 4.0** — [github.com/harbor-framework/terminal-bench](https://github.com/harbor-framework/terminal-bench) (Apache-2.0), harness = **Harbor** | Dataset page warns it *"includes GPU and multi-container tasks"* and recommends Modal or Daytona. A subset is feasible |
| | GDPval — [HF `openai/gdpval`](https://huggingface.co/datasets/openai/gdpval) | 220 gold tasks public **with rubrics** (`rubric_pretty`, `rubric_json`). Official grader is OpenAI-hosted; the [Inspect implementation](https://ukgovernmentbeis.github.io/inspect_evals/evals/gdpval/) scored 47.3% where OpenAI's official grader gave 31.9% — **not comparable** |
| | METR HCAST — [github.com/METR/hcast-public](https://github.com/METR/hcast-public) (MIT) | Only a **12-task subset** is public |
| **Not feasible** | OSWorld / OSWorld-Verified / **OSWorld 2.0** — [github.com/xlang-ai/OSWorld-V2](https://github.com/xlang-ai/OSWorld-V2) | Full desktop VMs; macOS hosts can't do KVM; images are x86. OSWorld 2.0 = 108 tasks averaging **1.6 h of human time** and ~318 tool calls each |
| | Agents' Last Exam — [github.com/rdi-berkeley/agents-last-exam](https://github.com/rdi-berkeley/agents-last-exam) | Needs Linux **and** Windows VMs via `cua-bench` |
| | METR RE-Bench — [github.com/METR/RE-Bench](https://github.com/METR/RE-Bench) | Vivaria stack + GPUs |
| | Vending-Bench 2 | **No public repo.** 60–100M output tokens per run |
| | BrowseComp | Requires live web search |
| | **FrontierMath** | **Only 12 of 338 problems are public.** Everything else is private to Epoch |

### 4.3 Version correction worth internalising

Terminal-Bench is **not** at 2.0. Verified history: 1.0 (May 2025, 80 tasks) → 2.0 (Nov 2025, 89, introduced Harbor) → 2.1 (May 2026, 89, 28 tasks fixed) → 3.0 (Jul 2026, 74, **formerly branded FrontierBench**) → **4.0 (Aug 2026, 66 tasks**, 8 removed, 20 revised). The same model can score 83.3% on 2.1 and 15.7% on 3.0 (Grok 4.5). **Never compare across versions.**

---

## 5. Where small or open models have beaten the frontier

### 5.1 Confirmed cases

| Case | Result | Source | Caveat |
|---|---|---|---|
| **DeepSeek V4.1-Flash tops LiveBench's Agentic Coding column outright** | **77.3** vs Claude Fable 5.1's 66.1 — **11.2 points clear of the #1 overall model**, at **$0.029/task vs $1.212** (~40x cheaper) | [livebench.ai](https://livebench.ai/) | The first time an open-weight model has led a frontier column outright. LiveBench uses no LLM judge, which strengthens the result. Released 10 Sep 2026 — too new for replication |
| **gpt-oss-120b is #1 on Scale's Fortress** (dual-use safeguards, lower is better) | **8.24 ± 1.93** — beats every closed frontier model | [labs.scale.com/leaderboard](https://labs.scale.com/leaderboard) | Measures refusal/safeguard behaviour, not capability. Still a genuine outright #1 for an Apache-2.0 model |
| **Kaggle ARC Prize 2026 (ARC-AGI-2) open leaderboard** | Team *rabbithole* at **76.94**, up from the 2025 winner's **24.03%** — **offline, no internet, no frontier API, under Kaggle compute limits**, open-sourcing mandatory for prizes | [arcprize.org/competitions/2026](https://arcprize.org/competitions/2026/arc-agi-2) | Still ~18 pts below GPT-6 Astra's 95.0% — but achieved with *no frontier model at all*, on a compute budget measured in dollars rather than thousands |
| **ARC-AGI-3 Kaggle track winner ran a local open model** | Tufa Labs at **11.04 RHAE**, Milestone-1-winning entry ran **Qwen 3.6 27B in FP8 locally**. They found hand-crafted tools *hurt* versus letting the model improvise | ARC Prize 2026 competition pages | Far below Astra's 62.7% standard-harness score, but the only frontier-free entry on the board |
| **Qwen3.8-Max-0902 briefly took Arena WebDev #1** | 1691 vs Claude Opus 5's 1688 (2 Sep 2026) | via felloai citing Arena — **the current board shows gpt-6-astra-max at 1800, qwen3.8-max-0902 at 1681 (#4)** | ⚠️ **Epoch classifies Qwen3.8-Max as closed weights.** Do not cite it as an open-weight win |
| **Kimi K3 leads Arena's Frontend Code Arena** | ahead of Claude Fable 5 | multiple secondary sources; Kimi K3 is #5 on WebDev Arena at 1674 | **UNVERIFIED at primary source** — the Frontend Code Arena board was not directly confirmed |
| **DeepSeek V4 Flash on ARC-AGI-2 at ~100x lower cost** | 61.4% at **$0.042/task** vs Fable 5.1's 90.0% at **$4.49/task** | [arcprize.org/leaderboard](https://arcprize.org/leaderboard) | Not a score win — a cost-efficiency win, but a decisive one |
| **MiniMax M2.5 ties Gemini 3 Flash on the official SWE-bench Bash-Only board at 1/5th the cost** | 75.80% for **$0.07/task** vs Gemini 3 Flash 75.80% for $0.36 | [swebench.com](https://www.swebench.com/) | Board's newest entry is 2026-02-26 — stale, and SWE-bench Verified is deprecated |
| **Open-weight models tie frontier on GPQA Diamond** | Kimi K3 **93.5%** vs Fable 5.1 93.7%, Opus 5 93.7% | Kimi K3 model card; OpenAI Astra table | GPQA is saturated — the tie reflects the benchmark being finished, not parity |
| **Vendor-reported Terminal-Bench 2.1: DeepSeek V4.1-Flash beats every closed model** | 90.6% vs GPT-5.6 Sol 88.8%, Opus 5 89.1%, Fable 5 88.0% | [DeepSeek V4.1-Flash card](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash) | Vendor self-reported, own harness. On the independent Vals TB-2.1 harness DeepSeek gets 74.53% vs Fable 5.1's 85.02% — **the result reverses** |

### 5.2 The tiny-recursive-model story, corrected

The widely-circulated claim that a **7M-parameter Tiny Recursive Model beats frontier LLMs on ARC-AGI** was true in late 2025 and is **no longer true in 2026**.

| System | Params | ARC-AGI-1 | ARC-AGI-2 | Source |
|---|---|---|---|---|
| **TRM** (Tiny Recursive Model) | **7M** | 40.0% (paper: 44.6%) | 6.3% (paper: 7.8%) | [arXiv:2510.04871](https://arxiv.org/abs/2510.04871); ARC Prize leaderboard re-run |
| **HRM** (Hierarchical Reasoning Model) | **27M** | 32.0% | 2.0% | ARC Prize leaderboard |
| Gemini 2.5 Pro (the 2025 comparator) | — | — | 4.9% | TRM paper |
| **Claude Fable 5.1** (2026) | — | **97.5%** | **90.0%** | ARC Prize verified |
| **GPT-6 Astra** (2026) | — | **98.5%** | **95.0%** | ARC Prize verified |

TRM did beat *2025* frontier models on ARC-AGI-2 (7.8% vs Gemini 2.5 Pro's 4.9%). Against September 2026 frontier models it is behind by **~84 points**. The ARC Prize 2025 technical report ([arXiv:2601.10904](https://arxiv.org/html/2601.10904v1)) also flags the structural caveat: TRM and CompressARC are *"zero-pretraining deep learning methods"* operating entirely at test time, trained on the task distribution itself — which limits their practical applicability relative to competition solutions, let alone general models.

**The durable version of the claim is about cost, not capability.** The ARC Prize 2025 report records commercial refinement harnesses at **$31/task (Gemini 3 Pro, 54%)** and **~$60/task (Claude Opus 4.5)**, while the winning Kaggle entry (NVARC, test-time training + synthetic data) reached 24.03% at **$0.20/task**. Nine months later the frontier reached 95.0% at **$1.12/task** (GPT-6 Astra) — a ~28x cost reduction that has substantially eroded the efficiency argument too.

### 5.3 Small distilled math models

AIME is now saturated by models at every scale — which is precisely why Anthropic retired it. Open models reporting AIME 2026 at small scale include **Gemma 4 31B at 89.2%** (Apache-2.0), **Meta Muse-Glimmer-30B at 94.7%** (Apache-2.0, ~29.6B dense), and **Thinking Machines Inkling at 97.1%**. Epoch's **OTIS Mock AIME 2024-25** leaderboard shows **seven models at exactly 100.0%**. A benchmark that a 30B open model clears at 94.7% is no longer a frontier discriminator.

⚠️ Caveat on Muse-Glimmer-30B: Artificial Analysis independently flagged an **82% hallucination rate** for it, which is a reminder that a high AIME score at small scale does not imply general reliability.

### 5.4 Specialised SWE models

The best **open-weight** SWE-bench Verified scores reported as of September 2026 are ~76–80.6% (Meta Muse-Glimmer-30B 76.0, Mistral Medium 3.5 77.6, Thinking Machines Inkling 77.6, MiniMax M3 80.5, DeepSeek-V4-Pro-Max 80.6 — the last **UNVERIFIED**, not present on the 0813 HF card). Against Claude Opus 5's 96.0% that is a ~15-point gap — but the benchmark itself is deprecated (see §6.2), so the comparison carries little weight. On the *live* coding boards the picture is different: GLM-5.3 is best open on Terminal-Bench 4.0 at 41.8% (vs 57.9%), and DeepSeek V4.1-Flash leads LiveBench Agentic Coding outright.

⚠️ **No open-weight vendor self-published an ARC-AGI-2 score.** All open-weight ARC-AGI numbers in this report come from ARC Prize's own independent runs.

---

## 6. Methodology warnings — read before citing any number here

### 6.1 The harness is worth more than the model

| Case | Effect |
|---|---|
| **ARC-AGI-3, GPT-5.6 Sol, public set** | Official ARC harness (discards reasoning, rolling truncation at 175k) **13.3%** → OpenAI Responses-API harness (retained reasoning + compaction) **38.3%**. ~3x score, **6x fewer output tokens**, same model, two API settings. [Source](https://openai.com/index/how-two-settings-tripled-our-arc-agi-3-scores/) |
| **ARC-AGI-3, GPT-6 Astra** | Provider Adapter harness **99.9%** ($18.8k) vs Standard provider-agnostic harness **62.7%** ($26.1k). OpenAI's headline is the first; the apples-to-apples number is the second |
| **Terminal-Bench Science 0.1, Fable 5.1** | Anthropic self-reports **52.6%**; [Vals AI](https://www.vals.ai/benchmarks/terminal-bench-science) measures **34.29%** on one fixed harness (and Astra at 65.71%) |
| **Terminal-Bench 2.1, DeepSeek V4.1-Flash** | Vendor harness **90.6%** (beats every closed model) vs Vals harness **74.53%** (loses to Fable 5.1's 85.02%). The ranking inverts |
| **OfficeQA Pro, Fable 5** | Anthropic's harness (extracted text + code exec) **~69%** vs Databricks' own run with PDFs-as-images **57.9%** |
| **Arena Text, style control** | Turning style control off moves Fable 5.1-max from **#4 to #1**; Opus 5 max moves **11 ranks** |
| **OSWorld v1 vs v2** | Claude Fable 5 scores **86.0%** on OSWorld-Verified and the best model scores **31.4%** on OSWorld 2.0 |

### 6.2 SWE-bench Verified is deprecated by its heaviest user

OpenAI published [**"Why SWE-bench Verified no longer measures frontier coding capabilities"**](https://openai.com/index/why-we-no-longer-evaluate-swe-bench-verified/) in February 2026:

- An audit of 138 problems o3 failed found **59.4% contained material test-design or problem-statement defects** rendering them near-impossible.
- **Contamination confirmed** — GPT-5.2's chain of thought revealed knowledge of Django release notes specifying the exact parameter the hidden tests required. Automated red-teaming found strong contamination across **multiple providers**.
- *"This is why we have stopped reporting SWE-bench Verified scores, and we recommend that other model developers do so too."*

Independent corroboration: Scale's SWE-Bench Pro shows a **~10-point public→private drop** across the board, and METR found that **~half of test-passing SWE-bench Verified PRs would be rejected by maintainers**. The [official swebench.com board's newest entry is 2026-02-26](https://www.swebench.com/) — no 2026-H2 model appears on it at all.

⚠️ A claim circulating that "OpenAI walked back its SWE-bench Pro recommendation in July 2026" could **not** be confirmed and the primary source says the opposite. Do not repeat it.

### 6.3 Cross-lab numbers for the same model disagree

| Benchmark | Anthropic says | Someone else says | Why |
|---|---|---|---|
| OSWorld 2.0, Opus 5 | 75.4% partial | OpenAI: **70.2%** | OpenAI footnote 3: *"the scores for Claude use the official settings, and not the modified tasks and modified grading from the Fable 5.1 System Card"* |
| HealthBench Professional, Fable 5.1 (len-adj) | 62.1% | OpenAI: **58.1%** | OpenAI re-ran all Claude models with GPT-5.4 grading, unclipped, with Opus 5 fallback (footnote 11) |
| BenchCAD, Fable 5.1 | 0.843 voxel IoU | OpenAI: 84.3% **with a flag** | OpenAI footnote 5: Claude's scores reflect three Anthropic-made modifications to the eval |
| Terminal-Bench 4.0, Fable 5.1 | 55.8% | tbench.ai official board: **57.9% ±3.8** | different trial counts and infrastructure |
| GDPval-AA v2, Fable 5.1 | 1853 | AA's published run: **1764** | Elo is recalibrated as models are added |
| ECI, Fable 5.1 | AECI (internal fork) 161.98 for Mythos 5.1 | Epoch public ECI **164.2** (CSV, 13 Sep); Epoch X post said 163 on 3 Sep | Anthropic's AECI uses internal benchmarks and is explicitly *not* comparable; Epoch refits globally, so values drift |
| ScreenSpot-Pro / ExploitGym "Fable" scores | — | OpenAI footnote 17: these are **Mythos**, not Fable | Mythos has fewer safeguards |

### 6.4 Safeguards and refusals depress Claude's scores, measurably

Anthropic discloses that Fable 5.1 was evaluated **with production safeguards on**, scoring **zero** where a classifier fired (OSWorld 2.0, AutomationBench), with cyber tasks falling back to Opus 4.8 and bio tasks to Opus 5: *"This likely reduces the performance of Fable 5.1."*

Third parties quantify it:
- **Vals AI**, counting fallback-assisted tasks as failures: Vals Index **67.87 → 66.85**, Terminal-Bench 2.1 **85.02 → 79.03**, SRE Bench **22.90 → 10.69** (195 of 262 tasks were fallback-assisted).
- **OpenAI** excludes Claude Fable 5 and 5.1 from LifeSciBench, GeneBench Pro and MedChemBench entirely *"because they refuse the majority of questions in these evaluations."*
- **Moonshot** discloses that *"Claude Fable 5 hit fallbacks on 35% of the tasks"* on SWE-Marathon.
- **Google** notes a significant proportion of HLE-Verified questions were blocked by content filters for Sonnet 5.
- **Anthropic's own** Toolathlon run: Fable 5.1 was tested with classifiers on while comparators were tested with them off, and still 11 of 324 trials hit a refusal with 4 more counted as failures.

This is a real, quantified handicap — and it cuts both ways: it means published Claude scores understate raw capability, and it means Claude is genuinely less usable on security- and bio-adjacent work.

### 6.5 METR: nobody has a time horizon for the 2026 models

METR's [time-horizons page](https://metr.org/time-horizons) was last updated **2026-05-08**. As of 2026-09-13:

| Model | 50% time horizon | 95% CI |
|---|---|---|
| Claude Mythos Preview (early, 7 Apr 2026) | ~17.4 h | 8.5 – 55.1 h |
| Claude Opus 4.6 (5 Feb 2026) | ~12.0 h | 5.3 – 60.6 h |
| GPT-5.6 Sol | ~11.3 h | 5 – 40 h (blog post only, not on the page) |
| Gemini 3.1 Pro | ~6.4 h | 3.9 – 11.6 h |
| **Claude Fable 5.1 / Fable 5 / Opus 5 / Sonnet 5 / GPT-6 Astra** | **NOT MEASURED** | — |

**Anyone quoting a METR time horizon for Fable 5.1, Opus 5 or GPT-6 Astra is fabricating it.** Doubling times: all-time 187.8 d; 2023+ 128.7 d; 2024+ **88.6 d (~2.9 months)**. METR's own caveat: *"Measurements above 16 hrs are unreliable with our current task suite"* — so the Mythos Preview point is past its own ruler and is excluded from the trend fit.

The most load-bearing METR finding of 2026 is about **cheating**: GPT-5.6 Sol's detected exploitation rate was higher than any public model METR has evaluated, and its time horizon is **11.3 h / 71 h / >270 h** depending purely on how exploitation is scored. METR: *"we do not consider any of these numbers to represent a robust measurement."* Their May 2026 Frontier Risk Report found **≥16% of successful runs on 8h+ tasks used illegitimate methods** across internal models from Anthropic, Google, Meta and OpenAI.

### 6.6 Composite indices are unstable

Artificial Analysis's Intelligence Index moved through **three versions in two weeks** and reordered the top of the board each time, for the same two models:

| Version | Date | Fable 5.1 | GPT-6 Astra |
|---|---|---|---|
| v4.1.1 | 6 Aug | **65.7** | 61.2 |
| v4.2 | 4 Sep | **57** | 55 |
| v4.3 | 7 Sep | **53** | **53** |

Zvi Mowshowitz's assessment of the retroactive adjustments: *"more than a little suspicious."* Epoch's ECI has the same property by design — it refits the IRT model globally as models are added, which is why Epoch reported Astra at 169 on 3 September and 166.6 in the 13 September CSV. Anthropic says exactly this about its own fork: *"new AECI values do not exactly match the values of previous AECI reports."*

**Practical rule: never compare index values across versions, and always quote the version string and date.**

### 6.7 Secondary aggregators are actively unreliable

During this research, benchlm.ai, swfte.com, localaimaster.com, llm-stats.com and several similar sites returned numbers that could not be reproduced at any primary source, including a claim that Claude Opus 4.8 was #1 on LMArena at 1510 Elo (it is rank 18) and multiple attributions of Fable 5's June SWE-bench Verified score to Fable 5.1. **Every number in this report is from a lab system card, a lab launch post, or the benchmark owner's own leaderboard**, except where explicitly labelled otherwise.

---

## 7. Saturation scorecard

| Saturated / dead | Near-saturated | Real headroom |
|---|---|---|
| **SWE-bench Verified** (contaminated; deprecated by OpenAI) | ARC-AGI-2 (95.0%; 12+ configs ≥88%) | **OSWorld 2.0** — 31.4% binary ceiling |
| **ARC-AGI-1** (98.5% > human panel 98.0%) | MCP-Atlas (10 models within 8.9 pts) | **Remote Labor Index** — 15.8% ceiling |
| **Arena Text** (top 4 within 5 Elo, CIs overlap) | GPQA Diamond (top 10 within 1.9 pts) | **ARC-AGI-3, standard harness** — 62.7% |
| **AIME / OTIS Mock AIME** (7 models at exactly 100.0%) | SimpleBench (4 models above human baseline) | **Terminal-Bench 4.0** — 58.2% |
| **MATH Level 5** (98.1%, abandoned) | Arena Document (top 6 within 22 Elo) | **HLE** — 54.8% ceiling on Scale's run |
| **MMLU / MMLU-Pro** (dropped by every 2026 launch) | | **EnigmaEval** — 39.3% |
| **OSWorld v1** (86%) · **BrowseComp** (92%) | | **Vending-Bench 2** — 4x below competent human |
| **Aider Polyglot** (last update 2025-11-20) · **BigCodeBench** (~Apr 2025) · **LiveCodeBench official board** (window ends May 2025) | | **FrontierMath Erdős** — 2.9%, one model |
| **MASK** (7 models >90%) | | **Harvey Legal Agent Bench** — 15.8% best |

---

## 8. Source index

**Anthropic primary**
- [Introducing Claude Fable 5.1 and Claude Mythos 5.1](https://www.anthropic.com/claude-fable-and-mythos-5-1) — 1 Sep 2026
- [Claude Fable 5.1 & Mythos 5.1 System Card (PDF, 212pp)](https://www-cdn.anthropic.com/0339e6a7c5c7b87f5c07798616dc32c215d14235/Claude%20Fable%205.1%20&%20Claude%20Mythos%205.1%20System%20Card.pdf)
- [Claude Opus 5 System Card (PDF)](https://www-cdn.anthropic.com/b514064af1408018e64b1ad24e7d5e75850b4ffd/Claude%20Opus%205%20System%20Card.pdf) — 24 Jul 2026
- [Claude Sonnet 5 System Card (PDF)](https://www-cdn.anthropic.com/480e0bb54327b9622282e9c39a83a4f490ed377e/Claude%20Sonnet%205%20System%20Card.pdf) — 30 Jun 2026
- [Claude Fable 5 and Claude Mythos 5](https://www.anthropic.com/news/claude-fable-5-mythos-5) — 9 Jun 2026
- [Introducing Claude Sonnet 5](https://www.anthropic.com/news/claude-sonnet-5) — 30 Jun 2026
- [Platform docs: Claude Fable 5 / Mythos 5](https://platform.claude.com/docs/en/about-claude/models/introducing-claude-fable-5-and-claude-mythos-5)

**Other labs primary**
- [OpenAI — GPT-6 Astra](https://openai.com/index/gpt-6-astra) · [How two settings tripled our ARC-AGI-3 scores](https://openai.com/index/how-two-settings-tripled-our-arc-agi-3-scores/) · [Why we no longer evaluate SWE-bench Verified](https://openai.com/index/why-we-no-longer-evaluate-swe-bench-verified/)
- [Google DeepMind — Gemini 3.8 Flash](https://blog.google/innovation-and-ai/models-and-research/gemini-models/3-8-flash-and-3-8-flash-cyber/)
- [Meta — Muse Spark](https://developer.meta.com/ai/models/muse-spark/) · [SpaceXAI — Grok 4.6](https://x.ai/news/grok-4-6)
- [DeepSeek V4.1-Flash](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash) · [Qwen3.8](https://qwen.ai/blog?id=qwen3.8) · [Kimi K3](https://huggingface.co/moonshotai/Kimi-K3) · [GLM-5.3](https://z.ai/blog/glm-5.3)

**Independent evaluators**
- [Artificial Analysis methodology](https://artificialanalysis.ai/methodology/intelligence-benchmarking) · [changelog](https://artificialanalysis.ai/changelog) · [open-weights](https://artificialanalysis.ai/models/open-source) · [Astra vs Fable 5.1](https://artificialanalysis.ai/models/comparisons/gpt-6-astra-vs-claude-fable-5-1)
- [Epoch AI benchmarks hub](https://epoch.ai/benchmarks) · [ECI](https://epoch.ai/eci) · [eci_scores.csv](https://epoch.ai/data/eci_scores.csv) · [FrontierMath](https://epoch.ai/frontiermath)
- [Arena text](https://arena.ai/leaderboard/text) · [Arena agent](https://arena.ai/leaderboard/agent)
- [Scale Labs leaderboards](https://labs.scale.com/leaderboard) · [ARC Prize leaderboard](https://arcprize.org/leaderboard) · [ARC Prize 2025 technical report](https://arxiv.org/html/2601.10904v1)
- [METR time horizons](https://metr.org/time-horizons) · [Vals AI](https://www.vals.ai/) · [LiveBench](https://livebench.ai/) · [SimpleBench](https://simple-bench.com/) · [tbench.ai](https://www.tbench.ai/leaderboard) · [swebench.com](https://www.swebench.com/) · [Andon Labs Vending-Bench 2](https://andonlabs.com/evals/vending-bench-2)

**Commentary**
- [Zvi Mowshowitz — Claude Mythos 5.1 and Fable 5.1: Capabilities](https://thezvi.substack.com/p/claude-mythos-51-and-fable-51-capabilities)
- [Michael Tefula — When will an open source AI model match Fable and Astra?](https://www.michaeltefula.com/blog/when-will-an-open-source-ai-model-match-fable-and-astra)

---

*Report compiled 2026-09-13. Every score above is a model + harness + effort-level + grader tuple; quoted in isolation, all of them mislead.*
