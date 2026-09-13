# TPU Research Cloud (TRC) — application draft

> **⚠️ DRAFT — NOT SUBMITTED.** Nothing in this file has been sent to Google / TRC. See
> `rungs/grants/README.md` before sending: re-verify the submission channel, fill in every
> placeholder, and get the owner's explicit go-ahead.

## What we verified about the program

`research/04-landscape-and-compute.md` §B.4 (verified 2026-09-13), quoted in full because every
clause matters for whether this is worth pursuing:

> "**TPU Research Cloud — still alive, still free.** Status: Running. Rolling invitations
> ([sites.research.google/trc/about](https://sites.research.google/trc/about)). What you get:
> Access to a pool of **>1,000 Cloud TPU devices**; free TPU quota granted to *your own GCP
> project*, temporary, 'ready to use within minutes.' Current generations: 2026 user reports
> confirm **spot v5e + v6e** grants — anecdotal but consistent. Duration: Classically 30 days,
> renewable on request — **UNVERIFIED**, not stated on current pages. Eligibility: **No degree
> requirement. 'Anyone can express interest.'** Undergrads are being approved. You still pay: VM
> (n1-standard-2) + GCS storage — 'generally minimal.' **Catch: Requires a GCP billing account,
> i.e. a credit card.** This is the documented blocker for students. Obligation: Publish results
> openly."

We separately fetched the program's own about page (`sites.research.google/trc/about`, 2026-09-13)
to find the actual application entry point, since `research/04` names the program but not the form:
**the application is a Google Form**, linked from that page:
**[forms.gle/uwWjUewyxZ5okL3W7](https://forms.gle/uwWjUewyxZ5okL3W7)**. The page states applicants
must "share their TRC-supported research with the world through peer-reviewed publications, open
source code, blog posts, or other means," be "willing to share detailed feedback with Google," and
agree to Google's Terms and Conditions, Privacy Policy, and AI Principles. **The exact fields the
form itself asks for were not seen** (filling out the live Google Form was out of scope for this
read-only research pass) — the text below is meant to be pasted into whatever free-text fields it
turns out to have, not a guess at the form's structure.

**The GCP billing account requirement is the real blocker to solve first**, separate from writing
this pitch — TRC grants TPU *quota*, not a waived billing requirement; a GCP project with billing
enabled has to exist before quota is useful. Resolve that (any card-holder GCP account works — it
does not need spend, just billing enabled) before or alongside submitting this.

---

## Why TPU credits mean a different software stack, not just different hardware

Both `rungs/rung1_nanochat/` and `rungs/rung2_3b/` are CUDA-only by construction: nanochat's own
README states plainly it targets "CUDA primary" hardware (with an MPS/CPU fallback for toy runs,
not real training), and torchtitan's Stage-2 entry in `research/02-open-training-stack.md` lists
its hardware as "CUDA (H100/B200 CI); AMD fork; CPU unit tests only" — **no TPU support**. A TRC
grant does not make either package runnable as-is.

`research/02-open-training-stack.md` Stage 1/Stage 2 name the actual TPU-native path:

- **[`marin-community/marin`](https://github.com/marin-community/marin)** — "Stanford CRFM's open
  lab: the whole experiment-tracked pipeline, data → model → eval... **TPU-first** (JAX/XLA), GPU
  supported... Marin 8B and **32B** trained on TPU... Marin 32B Base beat Gemma 3 27B PT on 24/42
  base evals." Levanter (the earlier JAX trainer) merged into Marin in Nov 2025 — Marin is the
  live entry point, not a separate framework to also evaluate.
- **[`AI-Hypercomputer/maxtext`](https://github.com/AI-Hypercomputer/maxtext)** — "Google's
  reference JAX LLM. The TPU-scale workhorse... **TPU-first**, GPU supported." Less legible than
  Marin per the same survey, but "more battle-tested at Google scale."

**If this grant lands, the plan is: reimplement the Rung 1/Rung 2-equivalent runs on Marin (first
choice — the more transparent, better-documented of the two) or MaxText, in JAX, on the granted
v5e/v6e quota — not port `rungs/rung1_nanochat/` or `rungs/rung2_3b/` as-is.** That is real,
unbudgeted engineering work this package does not attempt (no downloads, no training, no new
framework integration in scope here) — flagging it now so it's a known cost of saying yes to this
grant, not a surprise afterward.

---

## Draft application text

**Project:** road-to-52 — an open, from-scratch LLM training stack and a costed capability ladder,
currently CUDA/MLX-based; a TRC grant would fund porting the mid-scale rungs to Marin/JAX on TPU.

**About me:** Sammy Tourani, Software Engineering student, McMaster University. Building in the
open: [github.com/SammyTourani/road-to-52](https://github.com/SammyTourani/road-to-52) (Apache-2.0),
with a published MLX pretraining throughput benchmark (a first — no one had published training,
as opposed to inference, MLX throughput numbers before) and a from-scratch 124M model in progress
on my own hardware. Full status: the repo's README and `docs/RESULTS.md`.

**What I'd use TPU quota for:** reproducing this project's Rung 1 (~1B dense, ~20B tokens) and
Rung 2 (3B dense or 30B-A3B MoE, 60B tokens) targets — see `results/ladder.md` for the exact
costed shapes — on Marin or MaxText instead of torchtitan, to get a second, TPU-native data point
on the same recipe and publish a direct JAX-vs-CUDA cost/throughput comparison at the same model
sizes. This directly serves TRC's own stated interest in TPU-based research that gets published
openly.

**Research area:** efficient/reproducible LLM pretraining; open-source ML infrastructure.

**Commitment:** all code, configs, and results are public under Apache-2.0 in the linked repo
already, and every future run (TPU or GPU) is logged the same way — command, commit, wall-clock,
cost, eval conditions — in `docs/RESULTS.md`. Happy to write up the JAX/TPU vs PyTorch/CUDA
comparison as its own public report once run, which doubles as the "share detailed feedback"
TRC asks for.

**GCP project:** *(fill in once a billing-enabled project exists — this is a prerequisite to
submitting, not something the form itself resolves)*
