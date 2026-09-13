# Eval plan — Rung 2

Harness: **`EleutherAI/lm-evaluation-harness`** ("lm-eval"), the static-benchmark harness
`docs/PLAN.md` §5 already names for rented-GPU rungs (the table's "Eval" row: *"lm-eval (static)"*
under "Rented GPUs (CUDA)", alongside `inspect_ai`+`inspect_evals` for agentic work, which does not
apply to a base pretrained model). Tasks: `mmlu` (5-shot), `hellaswag` (0-shot), `arc_challenge`
(0-shot), `gsm8k`, `humaneval`. Every number below is either lifted verbatim from a cited table (for
the targets) or explicitly marked as this package's planned protocol (for our own run) — see
"Eval conditions" at the end before treating any of this as a completed comparison. **Nothing in
this file has been run.**

## The targets, and what Meta actually published for each

The build brief names Llama-1 7B and Llama-2 7B as the comparison points. We checked both papers
(and, for Llama-2, the official Hugging Face model card) directly rather than assuming a
familiar-looking number is really there — it wasn't, for half of what was asked, and that gap
matters more than the numbers that were found.

### LLaMA-1 7B — individually published, all five tasks

Source: [arXiv:2302.13971](https://arxiv.org/abs/2302.13971), the original LLaMA paper. Fetched
2026-09-13.

| Task | LLaMA-1 7B | Table / protocol |
|---|---|---|
| **MMLU** (5-shot) | **35.1** | Table 9, "Massive Multitask Language Understanding (MMLU). Five-shot accuracy." Matches `results/ladder.md`/`docs/PLAN.md`'s existing "MMLU 35" figure exactly — cross-checked, not just copied. |
| **HellaSwag** (0-shot) | **76.1** | Table 3, "Zero-shot performance on Common Sense Reasoning tasks" |
| **ARC-Challenge** (0-shot) | **47.6** | same table, "ARC-c" column |
| **GSM8K** | **11.0** (single sample, no majority vote) / **18.1** (`maj1@k`, k=100 — Minerva-style majority voting over 100 samples) | the paper's quantitative-reasoning table; caption: *"For majority voting, we use the same setup as Minerva, with k=256 samples for MATH and k=100 for GSM8k."* **Use 11.0 as the comparable figure** — lm-eval-harness's default `gsm8k` task does not do 100-sample majority voting, so 18.1 is not an apples-to-apples target unless our own eval is also run with k=100 majority voting. This distinction was not obvious from the number alone; it only showed up on reading the table's two GSM8K columns side by side. |
| **HumanEval** (pass@1) | **10.5** | Table 8, "Model performance for code generation" |

The few-shot count LLaMA-1's own GSM8K protocol uses (as opposed to the *sampling* count, k=100,
above) is **not confirmed by this package** — re-check the paper's exact prompting setup before
treating 11.0 as shot-count-matched to whatever lm-eval-harness's `gsm8k` task defaults to.

### Llama-2 7B — MMLU only; everything else is a category average, not these four tasks

Source: [arXiv:2307.09288](https://arxiv.org/abs/2307.09288) Table 3, and the official
[`meta-llama/Llama-2-7b-hf`](https://huggingface.co/meta-llama/Llama-2-7b-hf) model card. Both
checked directly, twice each, specifically for whether HellaSwag/ARC-Challenge/GSM8K/HumanEval are
reported individually for the 7B model. **They are not.** Table 3 reports eight *category*
columns — Code, Commonsense Reasoning, World Knowledge, Reading Comprehension, Math, MMLU, BBH, AGI
Eval — and states plainly that "Code" averages HumanEval+MBPP, "Commonsense Reasoning" averages
~8–9 tasks including HellaSwag and both ARC splits, and "Math" averages GSM8K+MATH. Table 4
(closed-source comparison) reports individual GSM8K/HumanEval numbers, but **only for the 70B
model**, not 7B. The official HF model card reports the same category averages as the paper, no
more granular.

| | Llama-2 7B | What it actually is |
|---|---|---|
| **MMLU** (5-shot) | **45.3** | MMLU is its own column in Table 3, not an aggregate — this one number *is* directly comparable. Matches `results/ladder.md`/`docs/PLAN.md`'s existing "MMLU 45.3" figure. |
| "Commonsense Reasoning" | 63.9 | category average incl. HellaSwag + ARC-e + ARC-c + 5–6 others — **not** a HellaSwag or ARC-Challenge number on its own |
| "Math" | 14.6 | category average of GSM8K + MATH — **not** a GSM8K number on its own |
| "Code" | 16.8 | category average of HumanEval + MBPP — **not** a HumanEval number on its own |

**We are not substituting a third-party leaderboard number to fill this gap and calling it
"published."** If a HellaSwag/ARC-Challenge/GSM8K-alone/HumanEval-alone figure for Llama-2-7B is
wanted for comparison later, get it from a named, cited reproduction (e.g. a specific
lm-evaluation-harness run with its exact commit and flags) and label it as a third-party
reproduction, not a Meta-published number — the same standard this repo applies to nanochat's
ClimbMix or Prime Intellect's grant terms elsewhere in `rungs/`.

## What "beats a named model" means here, per task

| Task | Beats LLaMA-1 7B if... | Beats Llama-2 7B if... |
|---|---|---|
| MMLU (5-shot) | our score > 35.1 | our score > 45.3 |
| HellaSwag (0-shot) | our score > 76.1 | **no individual Meta number exists to beat** |
| ARC-Challenge (0-shot) | our score > 47.6 | **no individual Meta number exists to beat** |
| GSM8K | our score > 11.0 (single-sample) — or, only if we also run k=100 majority voting, > 18.1 | **no individual Meta number exists to beat** (14.6 is Math-category, GSM8K+MATH averaged) |
| HumanEval (pass@1) | our score > 10.5 | **no individual Meta number exists to beat** (16.8 is Code-category, HumanEval+MBPP averaged) |

**The honest framing, stated once and meant throughout:** Rung 2's 60B-token, 3B-active-parameter
budget is a *massive* token-count handicap against either model — LLaMA-1 7B was trained on 1T
tokens (7B params, ~143 tokens/param) and Llama-2 7B on 2T (~286 tokens/param); Rung 2 at
Chinchilla-optimal is 60B tokens at 3B params (20 tokens/param) — roughly **1/7 the parameters and
1/17–1/33 the tokens**. `results/ladder.md`'s own note that "NO NAMED MODEL IS CLAIMED" at this rung
is the right posture; beating LLaMA-1 7B's individual-task numbers (where they exist) with a much
smaller compute budget would be a genuinely interesting data-efficiency result, worth reporting
loudly if it happens — it is not something to expect by default, and Rung 2b's larger 150B-token
budget (`rungs/README.md`'s cost sheet) is the more likely rung to get there, if any does.

## Eval conditions for our own run (record these exactly, every time)

Per `rungs/README.md`'s checklist item 5 ("report eval conditions, every time — never a bare
number") and `docs/RESULTS.md`'s existing row format (`conditions` column):

| Field | Our planned value | Note |
|---|---|---|
| Harness | `lm-evaluation-harness`, pin the exact commit used | record the commit, not just "lm-eval" |
| MMLU | 5-shot, `acc` | matches both LLaMA-1's and Llama-2's stated protocol |
| HellaSwag | 0-shot, `acc_norm` (lm-eval-harness's standard metric for this task) | matches LLaMA-1's stated zero-shot protocol; note `acc_norm` vs plain `acc` explicitly, they differ |
| ARC-Challenge | 0-shot, `acc_norm` | matches LLaMA-1's stated zero-shot protocol |
| GSM8K | record shot count and whether majority voting (`maj1@k`) is used, and k if so | **do not compare a single-sample run to LLaMA-1's 18.1** — that number is k=100 majority vote, compare single-sample runs to 11.0 instead |
| HumanEval | pass@1, record temperature/sampling budget used | |
| Model / checkpoint | commit of `road-to-52` + the exact checkpoint step | ties the number to a specific, reproducible artifact |
| Decontamination | confirm `data_plan.md`'s `decon` pass ran against each of these five tasks before reporting | a number without this is not trustworthy regardless of harness correctness |

Record every run in `docs/RESULTS.md` in its existing format (see the `gpt2-124m-reference` rows
already there for the exact style: command, sample count, tokenizer, block size, wall-clock,
commit).
