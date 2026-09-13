# road-to-52 -- write-up draft

**DRAFT — the owner decides whether and when to post; nothing here is auto-posted.**

Nothing in this file has been published anywhere. It exists so Sammy can review, edit, fill in the
`{{RESULT_*}}` placeholders once Rung 0 finishes, and post it himself (or not) when he's ready.

**Before posting, fill in:**

| Placeholder | What it is | Source |
|---|---|---|
| `{{RESULT_VAL_LOSS}}` | Final Rung 0 FineWeb val loss (nats/token) | `docs/RESULTS.md`, run `gpt2-124m-mac` |
| `{{RESULT_HELLASWAG}}` | Final Rung 0 HellaSwag acc_norm (%) | `docs/RESULTS.md`, run `gpt2-124m-mac` |
| `{{RESULT_DAYS}}` | Actual wall-clock days the run took | `runs/gpt2-124m-mac/log.jsonl`, the `"end"` event |
| `{{RESULT_VS_GPT2}}` | one word/phrase: "beat" / "matched" / "fell short of" | compare val loss to <= 3.447, HellaSwag to >= 29.4% |
| `{{RESULT_VS_STRETCH}}` | one phrase on the 3.28 speedrun stretch bar | compare val loss to <= 3.28 |

Every non-placeholder factual claim below is sourced from `README.md`, `docs/PLAN.md`, or
`docs/RESULTS.md` as they stand on 2026-09-13 -- nothing here claims more than those documents
support, and nothing claims this project beats Claude Fable 5.1 at anything. Fable 5.1 appears only
as the named bar in the project's own gap tracker.

---

## (a) LinkedIn post (~250-350 words)

I asked an AI to help me figure out if I could build a model that beats itself.

Specifically: could one person, with a Mac Mini and only open-source tools, train something that
beats Claude Fable 5.1?

The honest answer is no. Frontier training runs are on the order of 1e26-1e27 FLOPs, and even the
cheapest frontier-class open-weight run (DeepSeek-V3 scale) cost about $2.5M of rented GPU time
before a single ablation. That gap doesn't close with a side project.

So I built the thing that actually *is* possible, and that -- as far as I could find -- nobody had
properly shipped: the complete modern LLM recipe (data, tokenizer, pretraining, midtraining, SFT,
RL, eval, chat), implemented natively in MLX for Apple Silicon, benchmarked honestly, and used to
climb a public ladder where every rung beats a named model at a documented cost -- with a gap
tracker showing exactly how far each rung sits from the frontier.

Two gaps this fills, as far as I could find: there was no maintained, well-licensed MLX
*pretraining* framework (Apple's own MLX lead started one in March and never filled it in), and
nobody had published an MLX pretraining throughput number -- every "MLX tokens/sec" figure floating
around out there is inference decode, not training.

Measured on a base Mac Mini M4 (16GB, $600): 3,034 tokens/sec training a 124M-parameter model, 61%
of the chip's theoretical peak. Rung 0 is that 124M model, trained fully from scratch on that same
machine, on FineWeb, targeting GPT-2 small (2019) as I measured it with my own eval code.

Final numbers: val loss {{RESULT_VAL_LOSS}} nats/token, HellaSwag {{RESULT_HELLASWAG}}% acc_norm --
{{RESULT_VS_GPT2}} the GPT-2 target, {{RESULT_VS_STRETCH}} the 3.28 speedrun stretch bar, after
{{RESULT_DAYS}} days of background compute on a machine I already owned. $0 marginal cost.

No Claude-generated data anywhere in training -- open datasets and open-weight teachers only.

Next: rungs 1-3 are costed and ready to run on rented GPUs, pending budget; the post-training stack
(midtrain -> SFT -> RL -> chat) is already built and tested.

Code, benchmarks, and the full ladder: https://github.com/SammyTourani/road-to-52

## (b) X thread (6 tweets)

**1/**
I asked an AI if it could help me build something that beats itself: can one person with a Mac
Mini and open-source tools train a model that beats Claude Fable 5.1?

Honest answer: no. Frontier runs are 1e26-1e27 FLOPs. Here's what I built instead -- a thread.

**2/**
The recipe the efficient labs use is public: Muon optimizer, QK-norm, value embeddings, U-net
skips, WSD schedules, verifiable-reward RL. All in open papers and MIT/Apache code. Nobody had put
it together as a maintained, benchmarked MLX pipeline for Apple Silicon. So I did.

**3/**
First result: the first published MLX pretraining throughput benchmark. 124M-param model, base Mac
Mini M4 (16GB, $600), 3,034 tok/s at 61% MFU. Every "MLX tokens/sec" number you've seen before this
was inference decode, not training.

**4/**
Then I used it to climb a ladder: every rung beats a named model at a documented cost, with a gap
tracker showing exactly how far each rung sits from Fable 5.1. Rung 0: a 124M model trained fully
from scratch on that same Mac Mini, targeting GPT-2 small (2019).

**5/**
Final Rung 0 numbers: val loss {{RESULT_VAL_LOSS}} nats/token, HellaSwag {{RESULT_HELLASWAG}}% --
{{RESULT_VS_GPT2}} the GPT-2 target, {{RESULT_VS_STRETCH}} the 3.28 speedrun bar. {{RESULT_DAYS}}
days, $0, on hardware I already owned.

**6/**
No Claude-generated data anywhere in training -- open datasets and open-weight teachers only. Full
recipe, benchmarks, and ladder are public. Rungs 1-3 are costed and ready to run on rented GPUs
next.

https://github.com/SammyTourani/road-to-52

## (c) Hugging Face collection description (one paragraph)

road-to-52 is an open, Apple-Silicon-native LLM training stack and a public capability-per-dollar
ladder: every model in this collection is trained from scratch (or post-trained from an open base)
with a fully open recipe -- Muon optimizer, QK-norm, value embeddings, U-net skips, WSD schedules,
and verifiable-reward RL -- implemented natively in MLX and benchmarked honestly, with every
reported number carrying its exact command, sample count, tokenizer, sequence length, commit and
wall-clock. No Claude-generated data is ever used for training. Rung 0 is a 124M-parameter model
trained fully from scratch on a Mac Mini M4, targeting GPT-2 small (2019) as measured by this
project's own evaluation code; later rungs (costed, not all run yet) target GPT-2 XL-, Llama-1
7B-, and Llama-3.1-8B-class capability at a fraction of the original training cost. See the gap
tracker for exactly how far each rung sits from Claude Fable 5.1 -- the honest bar this project does
not claim to clear. Code: https://github.com/SammyTourani/road-to-52.
