---
license: apache-2.0
tags:
- mlx
- road-to-52
- apple-silicon
library_name: mlx
pipeline_tag: text-generation
{{base_model_yaml}}---

# {{run_name}}

{{what_this_is}}

**Status:** {{run_status}}

## Training data

| | |
|---|---|
| Dataset | {{dataset_id}} |
| Details | {{dataset_desc}} |
| Tokens budgeted | {{tokens_budget}} |
| Tokens actually seen | {{tokens_seen}} |

**No Claude-generated data was used to train this model.** This is a hard rule of the road-to-52
project (`docs/PLAN.md` §0, §7 — Anthropic's terms forbid training competing models on Claude
outputs). Claude models plan and write the *code* in this repository; the model itself learns only
from the open dataset(s) named above and, for post-trained variants, open-weight teacher models —
never from Claude outputs.

## Architecture

`{{arch_model_type}}` — decoder-only transformer, [road-to-52](https://github.com/SammyTourani/road-to-52)'s
modded-nanogpt/nanochat-lineage recipe (Muon optimizer, QK-norm, value embeddings, U-net skips,
ReLU² or SwiGLU MLP, logit soft-capping — see `docs/PLAN.md` §3.1).

| | |
|---|---|
| Layers | {{arch_n_layer}} |
| Embedding dim | {{arch_n_embd}} |
| Attention heads (Q / KV) | {{arch_n_head}} / {{arch_n_kv_head}} |
| Head dim | {{arch_head_dim}} |
| Vocab size | {{arch_vocab_size}} |
| Context length | {{arch_block_size}} |
| MLP | {{arch_mlp}} (ratio {{arch_mlp_ratio}}) |
| QK-norm | {{arch_qk_norm}} |
| Value embeddings | {{arch_use_value_embeds}} (n={{arch_n_value_embeds}}) |
| U-net skips | {{arch_use_unet_skips}} |
| Logit softcap | {{arch_softcap}} |
| Norm epsilon | {{arch_norm_eps}} |
| Total parameters | {{arch_params_total}} |
| Non-embedding parameters | {{arch_params_non_embedding}} |

## Training

| | |
|---|---|
| Steps | {{final_step}} / {{total_steps}} scheduled |
| Tokens/step | {{tokens_per_step}} |
| Tokens seen | {{tokens_seen}} (budget {{tokens_budget}}) |
| Precision | {{precision}} |
| Optimizer | {{optimizer_desc}} |
| Schedule | {{schedule_desc}} |
| Wall-clock | {{wall_clock}} |
| Hardware | {{hardware}} |
| Throughput | {{throughput}} |
| MLX version | {{mlx_version}} |
| Status | {{run_status}} |

## Evaluation

Every number below carries its exact command, sample count, tokenizer, sequence length, commit and
wall-clock — see [`docs/RESULTS.md`](https://github.com/SammyTourani/road-to-52/blob/main/docs/RESULTS.md)
for the full log and [`r52/eval/report.py`](https://github.com/SammyTourani/road-to-52/blob/main/r52/eval/report.py)
for how these records are produced.

### This model

| Benchmark | Value | Sample count / limit | Tokenizer | Block size | Commit | Command |
|---|---|---|---|---|---|---|
{{evaluation_rows}}

### Reference: GPT-2 124M (measured with this repo's own eval code)

| Benchmark | Value | Sample count / limit | Tokenizer | Block size | Commit | Command |
|---|---|---|---|---|---|---|
{{reference_rows}}

Source: [`results/gpt2-124m-reference/`](https://github.com/SammyTourani/road-to-52/tree/main/results/gpt2-124m-reference).
The GPT-2 124M reference weights (`openai-community/gpt2`) are **not** part of this repository's own
model family and are **never uploaded** by `scripts/hf_upload.py` — they exist only so every number in
this card is measured with identical code, on identical hardware, under identical conditions.

### Comparison caveats

- **Validation loss** is measured on **FineWeb**; GPT-2 (2019) was trained on **WebText**, a different
  corpus. The val-loss numbers in this card are directly comparable only to other FineWeb-trained
  models (e.g. the modded-nanogpt speedrun lineage), not to GPT-2's own original training loss.
- **HellaSwag** is reported as the distribution-neutral number in this comparison: it is a
  multiple-choice completion task, so — unlike perplexity / validation loss — its score does not
  depend on which pretraining corpus a model happened to see.
- **The 3.28 speedrun bar** (modded-nanogpt record #89, 2026-07-17) is a **FineWeb-trained** 124M
  reproduction of the GPT-2 architecture class, run on 8xH100 in ~73 seconds. It is this rung's
  stretch target, not a claim that this checkpoint has matched it — check the evaluation table above
  for the measured number.
- Composite/aggregate scores are never used to compare this model against frontier systems; see
  `docs/PLAN.md` §2 and [`results/GAP.md`](https://github.com/SammyTourani/road-to-52/blob/main/results/GAP.md)
  for why (index re-fits, safeguard asymmetry, harness sensitivity).

## How to run

Locally, after `git clone https://github.com/SammyTourani/road-to-52 && cd road-to-52` and
`uv venv --python 3.12 .venv && VIRTUAL_ENV=$PWD/.venv uv pip install -e ".[dev]"` — either against
this card's exported directory (`{{export_dir}}`) or after downloading this repo's files locally:

```bash
# one-shot generation
.venv/bin/python -m mlx_lm generate --model {{export_dir}} --prompt "Once upon a time"

# interactive chat (uses the chat template baked into tokenizer_config.json)
.venv/bin/python -m mlx_lm chat --model {{export_dir}}

# road-to-52's wrapper scripts (nice -n 10, sanity checks, chat-template guard)
scripts/chat.sh {{export_dir}} "Once upon a time"
scripts/serve.sh {{export_dir}}          # OpenAI-compatible server, http://127.0.0.1:8080
```

From the Hugging Face Hub, once this card's repo is published:

```bash
.venv/bin/python -m mlx_lm generate --model {{repo_id}} --prompt "Once upon a time"
```

## Limitations

- **Small model.** {{arch_params_total}} total parameters, {{tokens_seen}} tokens seen — this is
  a research/education-scale model (GPT-2-124M class), not a general-purpose assistant. Expect weak
  factuality and weak instruction-following unless this is an explicitly post-trained (SFT/RL) variant
  — check `{{run_name}}`'s entry in `docs/RESULTS.md`.
- **No safety training.** This checkpoint has not been through safety fine-tuning, red-teaming, or
  RLHF-style alignment beyond whatever `docs/PLAN.md` §6 Phase 4 documents for post-trained variants.
  Do not deploy it in a user-facing product without doing that work yourself.
- **English only.** The training corpus (see "Training data" above), the tokenizer (GPT-2 BPE), and
  the evaluation suite are all English-language.
- **Research reproduction, not a product.** This model exists to demonstrate and honestly measure an
  open MLX training recipe on consumer Apple Silicon — see `docs/PLAN.md` §0 for the project's framing
  of what is, and is not, achievable at this scale (it does not claim to approach frontier models).

## How to reproduce

```bash
git clone https://github.com/SammyTourani/road-to-52 && cd road-to-52
uv venv --python 3.12 .venv && VIRTUAL_ENV=$PWD/.venv uv pip install -e ".[dev]"
.venv/bin/python scripts/prepare_data.py --shards 8   # FineWeb (GPT-2 tokens), ~1.6 GB

# train (background, nice -n 10, logs to runs/<run>/log.jsonl)
scripts/train.sh {{config_file}} {{run_name}}

# export the checkpoint to an mlx-lm-loadable directory
.venv/bin/python -m r52.export runs/{{run_name}}/ckpt/best models/{{run_name}}-mlx

# evaluate (val_loss + hellaswag; add "full" for +CORE)
scripts/eval.sh models/{{run_name}}-mlx standard
```

Config file: `{{config_file}}` · Run name: `{{run_name}}` · Commit this card was generated at: `{{commit}}`.

## Ladder position

{{ladder_row}}

Full ladder: [`results/ladder.md`](https://github.com/SammyTourani/road-to-52/blob/main/results/ladder.md) ·
Gap to Claude Fable 5.1: [`results/GAP.md`](https://github.com/SammyTourani/road-to-52/blob/main/results/GAP.md).

---

Generated {{generated_date}} by `scripts/hf_upload.py` · run `{{run_name}}` · commit `{{commit}}` ·
`{{machine}}`.

Part of [road-to-52](https://github.com/SammyTourani/road-to-52), built by
[Sammy Tourani](https://github.com/SammyTourani). Planned by Claude Fable 5.1, built by Claude agents,
directed by Sammy. No Claude-generated data was ever used to train this model.
