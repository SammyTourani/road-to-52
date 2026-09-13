# `rungs/grants/` — funding drafts

**Everything in this directory is a DRAFT. Nothing has been submitted, emailed, or applied for.**
Each file is a starting point for Sammy to edit, verify the live submission details for, and send
— or not — entirely at his discretion. Building these was in scope for this package; sending them
was never in scope and none of them were sent.

| File | Program | What it would fund | Status |
|---|---|---|---|
| [`prime-intellect-fast-compute-grant.md`](prime-intellect-fast-compute-grant.md) | Prime Intellect Fast Compute Grants | H100-hours for Rungs 1–2b (CUDA, torchtitan/nanochat) | DRAFT — submission channel needs reconfirming, see the file |
| [`tpu-research-cloud.md`](tpu-research-cloud.md) | Google TPU Research Cloud | TPU quota — a *different* hardware path (JAX/Marin/MaxText, not torchtitan) | DRAFT — form fields not independently seen, see the file |
| [`mcmaster-ccdb-sponsor-email.md`](mcmaster-ccdb-sponsor-email.md) | Digital Research Alliance of Canada, via a McMaster faculty sponsor | opportunistic H100 (Fir) / L40S (Killarney) time | DRAFT — needs a real professor's name before it can be sent |

## Before sending any of these

1. **Re-verify the live submission mechanism.** Programs change their application process without
   notice; every file below states exactly what was and wasn't independently verified for its
   program, and flags anything sourced from a third-party tracker rather than the program's own
   page. Re-check the cited URL immediately before sending.
2. **Fill in every placeholder** (professor names, contact info, exact GitHub URL if it changes,
   current status of Rung 0). Search each file for `[` to find them.
3. **This is the owner's call, every time** — same rule as everywhere else in `rungs/`
   (`rungs/README.md` "The rule: paid runs are the owner's decision" — funding applications are the
   free-compute mirror of that same rule: nothing gets sent without Sammy's explicit decision).

## Why these three, and not others

`research/04-landscape-and-compute.md` §B.4 covers more free/near-free options than these three
(Kaggle's 30 GPU-hr/week, Colab, HF ZeroGPU, Modal's $30/month, Azure/AWS student credits, the
NVIDIA Academic Grant). Those are either too small for Rungs 1–3's node-scale, multi-GPU
requirements, or gated behind a faculty PI in a way the McMaster CCDB route already covers more
directly. These three are §B.4's own highlighted picks for this project's scale: Prime Intellect
("Best single option for an ambitious undergrad" — open to anyone, no institutional gate, sized in
the thousands of GPU-hours), TPU Research Cloud (free, no degree requirement, but a genuinely
different hardware/software stack), and the McMaster CCDB route (§B.4's own verdict: "Highest-
ceiling free option, and it costs one email").
