# Prime Intellect Fast Compute Grant — application draft

> **⚠️ DRAFT — NOT SUBMITTED.** Nothing in this file has been sent to Prime Intellect. See
> `rungs/grants/README.md` before sending: re-verify the submission channel, fill in every
> placeholder, and get the owner's explicit go-ahead.

## What we verified about the program, and what we could not

`research/04-landscape-and-compute.md` §B.4 (verified 2026-09-13):

> "**Prime Intellect Fast Compute Grants** — **$500–$100,000** in compute credits, **5–10 day
> decisions** — Anyone, anywhere. Email a 1-page pitch. No .edu, no VC, no nationality gate — high
> technical bar. Best single option for an ambitious undergrad."
> Sources: [x.com/PrimeIntellect](https://x.com/PrimeIntellect/status/1786386588726960167),
> [GPU-Grants repo](https://github.com/eric-prog/GPU-Grants),
> [aicredits.dev](https://aicredits.dev/submissions/176-prime-intellect-fast-compute-grants-500-100k)

We re-checked the submission mechanism directly before drafting this: **primeintellect.ai's own
homepage does not currently list the grants program** (checked 2026-09-13 — the homepage shows
on-demand rentals, Liquid Reserved Clusters, and a "Book a Call" contact option, nothing named
"grants"). A third-party tracker (`aicredits.dev`, not Prime Intellect's own page) describes the
process as an email pitch to `contact@primeintellect.ai` with subject `Fast Compute Grants
Application — <project>`, covering who you are, what you want to build, why it matters for open
AI, GPU type/hours needed, and a rough timeline, with links to prior work.

**⚠️ UNVERIFIED: the `contact@primeintellect.ai` address and subject-line format come from a
third-party tracker, not Prime Intellect's own site, and were not independently confirmed.**
**Before sending this: check `primeintellect.ai`'s current contact/community channels (Discord, X,
LinkedIn — all linked in its footer) or the X post above for the current submission route, since
the program isn't on the main site as of this check.** The pitch text below is ready regardless of
which channel turns out to be current.

---

## Draft pitch text

**Subject:** Fast Compute Grant application — road-to-52 (open MLX training stack + costed capability ladder)

**Who I am:** Sammy Tourani, 20, Software Engineering student at McMaster University (2nd→3rd year
co-op), building in public on GitHub and LinkedIn.

**What I've built so far (public, verifiable):**

- **[`road-to-52`](https://github.com/SammyTourani/road-to-52)** — Apache-2.0. A from-scratch LLM
  training stack, native to Apple Silicon (MLX), covering the full modern recipe: data → tokenizer
  → pretraining → midtraining → SFT → RL → eval → export → chat (modded-nanogpt/nanochat lineage:
  Muon, QK-norm, ReLU², value embeddings, U-net skips, WSD schedule).
- **The first published MLX pretraining throughput benchmark.** No one had published one before
  this — every circulating "MLX tokens/sec" number online is inference decode, not training.
  [`results/mlx_pretrain_bench.md`](../../results/mlx_pretrain_bench.md): 30M–350M-parameter
  sweep, tokens/sec, TFLOPS, MFU, and memory, on a base 16GB Mac Mini M4.
- **Rung 0, in progress on my own hardware:** a 124M-parameter model, trained fully from scratch on
  FineWeb tokens on that same Mac Mini, targeting GPT-2-small quality (val loss ≤ 3.28) at $0
  marginal cost. Status and numbers are live in
  [`docs/RESULTS.md`](../../docs/RESULTS.md) and the repo's README.
- **A costed, sourced capability ladder** (`results/ladder.md`) and a from-scratch six-report
  research corpus on the 2026 open-training landscape (`research/`) — every number in both traces
  to a primary source (a paper, a repo, a pricing page).

**What I want to build next, and why it matters for open AI:** the same recipe, scaled onto rented
CUDA hardware, climbing a ladder where each rung beats a *named*, *cited* model at a *documented*
cost — proof that the modern efficient-training recipe (public code, public data, no frontier-lab
budget) produces real, checkable capability-per-dollar, not just a benchmark table. Every script,
config, and result is public and reproducible from the commit it was built on
(`rungs/` in the same repo has the ready-to-run packages for exactly the ask below).

**The ask, in H100-hours** (from `results/ladder.md`'s costed ladder, 35% MFU, Prime Intellect's
own Sept-2026 spot rate of $0.94/H100-h):

| Rung | What | Tokens | H100-hours | $ equivalent at $0.94/h |
|---|---|---|---|---|
| 1 | nanochat-class ~1B dense (beats GPT-2 XL, CORE 0.2565) | 20B | 96 | $91 |
| 2 | 3B dense or 30B-A3B MoE, from scratch | 60B | 867 | $815 |
| 2b | same, 150B tokens (the larger, more decisive budget) | 150B | 2,167 | $2,037 |
| **Core ask** | | | **3,130** | **≈ $2,943** |
| **With a 20% buffer** (LR sweeps, restarts, eval compute — our own padding, not a ladder number) | | | **≈ 3,750** | **≈ $3,530** |

Requesting **$3,500–$4,000 in Prime Intellect compute credits**, sized to actually run Rungs 1
through 2b end to end with room for the inevitable false start, not a round number pulled from the
program's stated ceiling.

**Timeline:** Rung 1 is a single ~2-hour 8×H100 session; Rung 2/2b together are roughly 2–3 weeks
of node time including the LR sweep `rungs/rung2_3b/README.md` calls for before committing the full
budget. All packages (`rungs/rung1_nanochat/`, `rungs/rung2_3b/`) are written, pinned to specific
upstream commits, and ready to run the day credits land.

**Open-sourcing commitment:** the repo is already public (Apache-2.0), every number that goes into
it carries a source, and every result — including a negative or partial one — gets published in
`docs/RESULTS.md` and the repo README, not just the wins. Trained checkpoints go to Hugging Face
with eval logs, per the project's own stated deliverable list (`docs/PLAN.md` §3, item 5).

**Links:** [github.com/SammyTourani/road-to-52](https://github.com/SammyTourani/road-to-52) ·
[LinkedIn — Sammy Tourani] *(add current URL before sending)*
