# road-to-52

**An open, Apple-Silicon-native LLM build stack, and a public ladder of capability-per-dollar —
measured honestly against the frontier.**

This repo started from one question: *can one person, with a Mac Mini and open-source tools, build
a model that beats Claude Fable 5.1?* The honest answer is no — frontier training runs are
1e26–1e27 FLOPs, and even the cheapest frontier-class open-weight run (DeepSeek-V3 scale) is about
$2.5M of rented GPU time before a single ablation. So the project became the thing that *is*
possible and that nobody has properly built: the complete modern recipe (data → tokenizer →
pretraining → midtraining → SFT → RL → eval → chat), implemented natively in [MLX](https://github.com/ml-explore/mlx)
for Apple Silicon, rigorously benchmarked, and used to climb a **ladder** where every rung beats a
*named* model at a *documented* cost — with a **gap tracker** that shows exactly how far each rung
is from Fable 5.1's published scores.

> **Status (2026-09-13):** research complete, core pipeline under construction, Rung 0 (a
> from-scratch GPT-2-small-class model trained on a 16 GB Mac Mini M4 for $0) starting.
> See [`docs/PLAN.md`](docs/PLAN.md) and [`docs/RESULTS.md`](docs/RESULTS.md).

## The ladder

| Rung | Hardware | Model | Tokens | Est. cost | Beats | Status |
|---|---|---|---|---|---|---|
| **0** | Mac Mini M4, 16 GB | 124M, from scratch | ≤ 0.75B | **$0** (3–6 days) | GPT-2 small (2019): FineWeb val loss ≤ 3.28, HellaSwag ≥ 29.4% | building |
| 1 | 8×H100, ~2 h | ~1B | ~20B | ~$50–100 | GPT-2 XL (1.6B) on DCLM CORE | planned |
| 2 | 8×H100, ~5 days | 3B dense / 30B-A3B MoE | 60B | ~$815 | Llama-1 7B (2023) | planned |
| 3 | 64×H100, ~4 days | 8B dense | 160B | ~$5,800 | Llama 3.1 8B's size at 1/94th its tokens (size match, not a quality claim) | planned |
| P | any rung + 8×H100 | post-training: midtrain → SFT → RLVR / distillation from open teachers | — | $2k–5k | a 2024 frontier model on a non-saturated math/code eval | planned |
| — | undisclosed | Claude Fable 5.1 | — | 1e26–1e27 FLOPs | the bar | — |

Costs come from a calibrated calculator (`python -m r52.ladder`) that reproduces DeepSeek-V3's
disclosed GPU-hours within ~4%; see [`results/ladder.md`](results/ladder.md). The gap to the
frontier, benchmark by benchmark, is in [`results/GAP.md`](results/GAP.md).

## Why this is worth doing

- **Two documented gaps** (see `research/02`): there is no maintained, well-licensed MLX pretraining
  framework, and no one has ever published an MLX *pretraining* throughput benchmark. Both are
  filled here (`r52/`, `results/mlx_pretrain_bench.md`).
- **The recipe is public, the scale is not.** Muon, QK-norm, value embeddings, U-net skips, logit
  softcapping, WSD schedules, verifiable-reward RL — everything the efficient labs do is in open
  papers and MIT/Apache code. This repo is the runnable, measured version for a $600 computer.
- **Honesty as a feature.** Every number carries its command, sample count, tokenizer, sequence
  length, commit and wall-clock, and the conditions of any published number it is compared to.

## What is in here

```
r52/            the MLX pipeline: model, optim (Muon+AdamW), data, train, bench, eval, export
r52/ladder/     compute-ladder calculator + sourced GPU prices
r52/bar/        the benchmark "bar": Fable 5.1 / frontier / open-weight scores with sources → gap table
configs/        runnable presets (nano_30m, gpt2_124m_mac, ...)
scripts/        prepare_data.py, train.sh (background + nice + memory cap), bench.sh, eval.sh
research/       six sourced research reports on the 2026 landscape (benchmarks, stack, data,
                compute economics, Apple-Silicon limits, frontier recipe)
docs/           PLAN.md (master plan), ARCHITECTURE.md (build spec), RESULTS.md, DEVIATIONS.md
results/        committed benchmark tables, ladder, gap tracker, eval JSON
```

## Quickstart (Apple Silicon)

```bash
git clone https://github.com/SammyTourani/road-to-52 && cd road-to-52
uv venv --python 3.12 .venv && VIRTUAL_ENV=$PWD/.venv uv pip install -e ".[dev]"
.venv/bin/python -m pytest -q                      # < 60 s
.venv/bin/python scripts/prepare_data.py --shards 8 # FineWeb (GPT-2 tokens), ~1.6 GB
scripts/bench.sh                                   # MLX pretraining throughput table
scripts/train.sh configs/nano_30m.yaml nano-a      # background run, logs in runs/nano-a/
```

## Rules this project follows

1. **No Claude-generated data is ever used for training** (Anthropic's terms forbid training
   competing models on Claude outputs). Claude models plan and write code here; the model learns
   only from open datasets and open-weight teachers.
2. Apache-2.0. Ideas are ported from MIT/BSD/Apache projects with attribution (nanochat,
   modded-nanogpt, mlx-examples, mlx-lm); nothing is copied from unlicensed repositories.
3. Training jobs on the Mac are capped at 6 GiB of GPU memory and niced so the machine stays usable.

## Credits

Built on the shoulders of [nanochat](https://github.com/karpathy/nanochat),
[modded-nanogpt](https://github.com/KellerJordan/modded-nanogpt), [MLX](https://github.com/ml-explore/mlx),
[mlx-lm](https://github.com/ml-explore/mlx-lm), [mlx-lm-lora](https://github.com/Goekdeniz-Guelmez/mlx-lm-lora),
[lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness), the DCLM CORE suite,
and the FineWeb datasets. Planned by Claude Fable 5.1, built by Claude Opus agents, directed by
[Sammy Tourani](https://github.com/SammyTourani).
