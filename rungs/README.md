# `rungs/` — the rented-GPU rung packages

This directory holds ready-to-run **packages** for the CUDA rungs of the ladder in
[`results/ladder.md`](../results/ladder.md) (Phase 5 of [`docs/PLAN.md`](../docs/PLAN.md) §6).
Nothing in here has been executed. No training has run, nothing has been downloaded beyond the
READMEs and configs fetched to write these packages, and no money has been spent. Every number
below is copied from `results/ladder.md`/`research/04-landscape-and-compute.md` or was verified
against the relevant upstream repo at the pinned commit given in each sub-package (see each
`README.md` for the exact `gh api` call used).

## What is here

| Path | Rung(s) | What it is |
|---|---|---|
| [`rung1_nanochat/`](rung1_nanochat/) | **Rung 1** — nanochat-class ~1B dense | Clone-and-run package for `karpathy/nanochat`'s `speedrun.sh`, pinned to a commit, with cost and eval notes |
| [`rung2_3b/`](rung2_3b/) | **Rung 2 / 2 (MoE) / 2b** — 3B dense or 30B-A3B MoE | torchtitan job config + data plan + launch script + eval plan for a from-scratch 3B-class Llama-3-style model (and a short MoE variant) |
| [`grants/`](grants/) | funding routes for Rungs 1–3 | DRAFT pitches for free/sponsored compute — Prime Intellect, TPU Research Cloud, a McMaster CCDB sponsor email |

Rung 3 (8B dense, 64×H100) has **no package here**: `research/04-landscape-and-compute.md` §B.5
puts it at $5.8k spot, past the $2,000 informal budget ceiling, and `results/ladder.md` marks it
"planned; needs budget" pending a grant (see `grants/`). Its cost-sheet row is included below for
completeness and its hardware/software choices are identical to Rung 2's — an 8B `llama3_configs["8B"]`
job on 64 GPUs instead of a hand-rolled 3B job on 8.

## The rule: paid runs are the owner's decision

Every package in this directory is documentation, configuration, and a shell script — none of it
runs by itself, and running it costs real money. This is not a suggestion; it is stated three
places in this repo already, and this package inherits all three:

1. `docs/PLAN.md` §7, rule 5: *"do not spend money without the owner's go."*
2. `results/ladder.md`: *"Rungs 1+ are prepared as ready-to-run scripts with cost estimates;
   **spending money on them is the owner's decision**."*
3. The build instruction for this directory: *"This is documentation, configuration and shell-script
   work — do NOT run any training, do NOT use the local GPU ... do NOT install packages."*

Every script in here defaults to **printing what it would do and stopping** (`--dry-run`, and in
`rung1_nanochat/run.sh`'s case, dry-run is not even the default you have to opt out of carefully —
read each script's usage banner). Turning a dry run into a real one always means Sammy explicitly
re-running the command without `--dry-run` (or setting the documented env var) on a GPU box he has
provisioned and is paying for himself.

## How to rent an 8×H100 node

All prices are $/GPU-hour, verified 2026-09-13, from `research/04-landscape-and-compute.md` §B.2
(full source list and every URL is there — this table is a convenience copy, not a new source).
"8×H100 node $/h" is the per-GPU rate × 8, not a separately-quoted node price, unless the provider
only sells by the node.

| Provider | Mode | $ / H100-hour | **8×H100 node, $/h** | Source |
|---|---|---|---|---|
| **Prime Intellect** | **spot** | **$0.94** | **$7.52** | [primeintellect.ai](https://primeintellect.ai) — cheapest credible spot; this is the rate `results/ladder.md` and every cost in this directory is computed from |
| SF Compute | spot market clearing (Aug 15 – Sep 11 2026 avg) | $2.03 | $16.24 | [sfcompute.com/prices](https://sfcompute.com/prices) — "the most honest non-interruptible market signal" per research/04 |
| RunPod | Community, PCIe | $1.99 | $15.92 | [runpod.io/pricing](https://www.runpod.io/pricing) |
| RunPod | Community, SXM | $2.69 | $21.52 | runpod.io/pricing |
| RunPod | Secure, SXM | $3.49 | $27.92 | runpod.io/pricing |
| Vast.ai | marketplace / interruptible | $1.49–$2.27 | $11.92–$18.16 | ⚠️ **UNVERIFIED third-party** ([spheron](https://www.spheron.network/blog/vastai-pricing-2026)); Vast's own site quotes "from $1.60" |
| Nebius | preemptible | $2.15 | $17.20 | [nebius.com/prices](https://nebius.com/prices) |
| Nebius | on-demand | $3.85 | $30.80 | nebius.com/prices |
| **Prime Intellect** | **on-demand** | **$2.43** | **$19.44** | primeintellect.ai — the on-demand rate used everywhere in this directory |
| CoreWeave | spot | $2.46 | $19.68 | [coreweave.com/pricing](https://www.coreweave.com/pricing) |
| CoreWeave | on-demand | $6.16 | $49.28 | coreweave.com/pricing |
| Lambda | on-demand | $3.99–$4.29 | $31.92–$34.32 | [lambda.ai/pricing](https://lambda.ai/pricing) — this is the provider nanochat's own README recommends ("I use and like Lambda") |

Hyperscalers (AWS/Azure/GCP/Oracle) run **3–5× the Prime Intellect spot floor** for identical H100
silicon — research/04 §B.2 has the full table. They are not listed here; there is no reason to pay
$6.88–$10.98/GPU-h for this workload.

**Spot vs on-demand, in practice for these rungs:**
- **Spot / preemptible / interruptible** is cheaper but can be reclaimed mid-run. It is fine for
  Rung 1 (≤2 h) and tolerable for Rung 2 (~4.5 days) *if* the job checkpoints and resumes — both
  `rung1_nanochat/run.sh` (nanochat checkpoints its base/SFT stages) and `rung2_3b/launch.sh`
  (torchtitan's `CheckpointManager`, interval documented in `torchtitan_3b.toml`) do.
- **On-demand** costs ~2.6× spot at Prime Intellect specifically, but guarantees the node doesn't
  disappear — worth it for a short, unattended run like Rung 1, or a hard deadline.
- **SF Compute's clearing price** sits between the two: not interruptible, but a real market rate
  rather than a rack-rate list price. Good middle ground for Rung 2/2b's multi-day jobs.

`h100` is the number used throughout: **989 TFLOP/s BF16 dense peak**
([research/04 §B.5](../research/04-landscape-and-compute.md), sourced to
[arXiv:2412.19437](https://arxiv.org/pdf/2412.19437)), MFU 35% for large runs / 26% for small —
`results/ladder.md`'s calibration, reproduced to within ~4% of DeepSeek-V3's own disclosed cost.

## Cost sheet — Rungs 1, 2, 2 (MoE), 2b, 3

Copied **verbatim** from [`results/ladder.md`](../results/ladder.md) (generated
`python -m r52.ladder`, 2026-09-13). Do not recompute these by hand; if the price floors in
`research/04` §B.2 change, regenerate `results/ladder.md` and re-copy this table.

| Rung | Beats (target model) | Params (total / active) | Tokens | FLOPs | Hardware | GPU-hours | Wall-clock | $ spot | $ on-demand | Status |
|---|---|---|---|---|---|---|---|---|---|---|
| **Rung 1** — nanochat-class 1B, Chinchilla-optimal | GPT-2 XL (1.6B, 2019): CORE 0.256525 | 1B / 1B | 20B | 1.20e20 | h100 ×8 | 96 | 12.0 h | **$91** | $234 | planned |
| **Rung 2** — 3B dense, Chinchilla-optimal | (no named model claimed — see `rung2_3b/README.md`) | 3B / 3B | 60B | 1.08e21 | h100 ×8 | 867 | 108 h (4.5 d) | **$815** | $2,106 | planned |
| **Rung 2 (MoE)** — 30B-A3B, Chinchilla-optimal on active params | Qwen3-30B-A3B-shaped capacity for a 3B-dense bill | 30B / 3B | 60B | 1.08e21 | h100 ×8 | 867 | 108 h (4.5 d) | **$815** | $2,106 | planned |
| **Rung 2b** — 3B dense (or 30B-A3B), the $2,000 budget | (no named model — this is what $2,000 buys) | 3B / 3B | 150B | 2.70e21 | h100 ×8 | 2,167 | 271 h (11.3 d) | **$2,037** | $5,265 | planned |
| **Rung 3** — 8B dense, Chinchilla-optimal | Llama 3.1 8B's *size* at 1/94th its tokens — a size match, not a quality claim | 8B / 8B | 160B | 7.68e21 | h100 ×64 | 6,163 | 96 h (4.0 d) | **$5,793** | $14,976 | planned; needs a grant |

**Two things this ladder is explicit about, carried forward here:**
- **Rung 1's real cheapest price is not this row.** nanochat's own disclosed record (Run 6,
  2026-03-14: CORE 0.262634, 1.65 h wall-clock on 8×H100 = 13.2 H100-hours) is ~7× cheaper than the
  Chinchilla-optimal 1B/20B row above, because the leaderboard doesn't disclose the record's exact
  N/D. `rung1_nanochat/README.md` gives both numbers and treats the disclosed record as the honest
  price of "nanochat-class capability."
- **Rung 2 makes no capability claim.** `research/04` gives this rung's FLOPs and cost but no
  measured result — the obvious same-size comparator (SmolLM3, also 3B) saw 11.2T tokens, 187× this
  budget. `rung2_3b/README.md` explains what targets *are* defensible at this budget.

## Checklist — running a rung honestly

Before spending anything, re-read `docs/PLAN.md` §7 ("Rules") — it is short and everything below
is downstream of it. When Sammy decides to actually run one of these:

1. **Pin every commit before you start, not after.**
   - Record the exact commit of the upstream repo you clone (`git rev-parse HEAD` right after
     clone, before any local edits) — the packages in this directory already pin one commit each;
     if time has passed, re-verify it's still the one you want (`gh api repos/<owner>/<repo>/commits/<branch> --jq .sha`)
     rather than silently drifting to a new HEAD.
   - Record the commit of **this repo** (`road-to-52`) at the moment you launch, so the exact
     config/script version is reproducible.
2. **Run with `--dry-run` first, read the printed commands, then run for real.** Every script in
   this directory supports this; use it even when you're confident.
3. **Record honestly in `docs/RESULTS.md`, in the same row format already used there** (see the
   existing rows for the gpt2-124m-reference runs) — one row per eval, with:
   - the exact **command** run (copy-pasteable, not paraphrased)
   - the **commit** (both upstream and `road-to-52`, if they differ)
   - **wall-clock** (seconds or hours, and how it was measured — e.g. nanochat's own
     `total_training_time` from its wandb summary, not `time bash speedrun.sh` which includes
     dataset download and tokenizer training)
   - the **$** actually spent (provider, instance type, spot/on-demand, hours billed — not the
     ladder's estimate; the estimate goes in this directory's docs, the *actual* bill goes in
     `docs/RESULTS.md`)
   - sample count/limit, tokenizer, sequence length, and eval protocol (see "report eval conditions"
     below) — `docs/RESULTS.md`'s `conditions` column is exactly for this
4. **Decontaminate.** Run `allenai/decon` (`research/03` §8) against whatever eval set you're about
   to report a number on, *before* trusting that number. For Rung 1 this means checking nanochat's
   pretraining data (ClimbMix — UNVERIFIED whether it ships decontaminated, `research/03` §8.4) and
   the CORE eval set it's compared against. For Rung 2 it's the explicit `data_plan.md` step.
5. **Report eval conditions, every time — never a bare number.** At minimum: task name, shot count,
   metric (`acc` vs `acc_norm` vs `pass@1` vs `maj1@k`), harness/version, and what the *comparison*
   number's conditions were (they are frequently different — see `rung2_3b/eval_plan.md`'s note on
   Llama-2 7B's GSM8K/HumanEval numbers not being separately published by Meta at all). A number
   without its eval conditions attached is not a result in this repo.
6. **Prefer spot + checkpointing over on-demand where the job supports resuming**, unless the
   provider's spot pool is visibly unstable that day or the run is short enough (Rung 1) that
   on-demand's certainty is worth the ~2.6× premium.
7. **Shut the node down.** Obvious, but it's the step that turns "$815 estimated" into "$3,200
   actual" if skipped. Every `launch.sh`/`run.sh` prints the provider console reminder at the end.
