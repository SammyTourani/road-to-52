# Rung 1 — nanochat-class ~1B dense model on a rented 8×H100 node

Maps to **Rung 1** in [`results/ladder.md`](../../results/ladder.md) and the "nanochat record"
calibration row directly beneath it. Package: this README + [`run.sh`](run.sh). Not run. No money
spent, no training done, nothing downloaded beyond the three files quoted below (fetched read-only
via `gh api ... --jq .content | base64 -d`).

## Pinned commit

```
repo:    https://github.com/karpathy/nanochat
commit:  92d63d4e8bb4df75c3b71618f31ddde2378b2bcd
date:    2026-07-03T22:54:57Z  ("clean up fragile code")
branch:  master (this is master's actual HEAD commit)
license: MIT
```

Found and verified 2026-09-13 with:

```bash
gh api repos/karpathy/nanochat/commits/master --jq '{sha: .sha, date: .commit.committer.date}'
```

Note for anyone re-running this later: `gh api repos/karpathy/nanochat --jq .pushed_at` currently
returns `2026-09-07`, two months after this commit — that's a push to one of the repo's other live
branches (`moe`, `moe3`, `experiment_refactor`, `fp8_attempt_fail`; `gh api
repos/karpathy/nanochat/branches --jq '.[].name'`), not to `master`. `master`'s own latest commit,
confirmed via `gh api "repos/karpathy/nanochat/commits?sha=master&per_page=3"`, is the one pinned
above. If you re-pin later, check `master` specifically, not repo-level `pushed_at`.

`README.md`, `runs/speedrun.sh`, `dev/LEADERBOARD.md`, and `nanochat/dataset.py` at this exact
commit were fetched with `gh api "repos/karpathy/nanochat/contents/<path>?ref=92d63d4e8bb4df75c3b71618f31ddde2378b2bcd" --jq '.content' | base64 -d`
and are quoted verbatim below.

## Setup and invocation (verified against the fetched README + speedrun.sh)

nanochat's README, section "Reproduce and talk to GPT-2":

> "The entire pipeline to do so is contained in the single file
> [`runs/speedrun.sh`](https://github.com/karpathy/nanochat/blob/master/runs/speedrun.sh), which is
> designed to be run on an 8XH100 GPU node. Boot up a new 8XH100 GPU box from your favorite provider
> (e.g. I use and like [Lambda](https://lambda.ai/service/gpu-cloud)), and kick off the training
> script:
>
> ```bash
> bash runs/speedrun.sh
> ```
>
> You may wish to do so in a screen session as this will take ~1.5 hours to run."

`runs/speedrun.sh` itself (its own header comment, verbatim):

> "This script is configured to train your own GPT-2 grade LLM (pretraining + finetuning). It is
> designed to run on a blank 8XH100 GPU node and takes approximately 1.5 hours to complete."

and it is fully self-contained: it installs `uv` if missing, runs `uv sync --extra gpu`, activates
the venv, downloads pretraining data shards, trains the tokenizer, then (quoting its own inline
comments and the exact commands):

```bash
# d24 model (slightly undertrained to beat GPT-2 => decrease data:params ratio from compute
# optimal 10.5 (default) to 8)
torchrun --standalone --nproc_per_node=8 -m scripts.base_train -- --depth=24 --target-param-data-ratio=8 --device-batch-size=16 --fp8 --run=$WANDB_RUN
torchrun --standalone --nproc_per_node=8 -m scripts.base_eval -- --device-batch-size=16
torchrun --standalone --nproc_per_node=8 -m scripts.chat_sft -- --run=$WANDB_RUN
torchrun --standalone --nproc_per_node=8 -m scripts.chat_eval -- -i sft
```

**`run.sh` in this directory clones the pinned commit and runs exactly `bash runs/speedrun.sh` —
it does not re-implement or paraphrase the internals above.** They're quoted here so the choice is
auditable without re-fetching, and so you know what's about to run before you run it.

This configuration (`--depth=24 --target-param-data-ratio=8 --fp8`) is not an arbitrary d24 run —
it is *exactly* nanochat's own current-SOTA leaderboard config: `dev/LEADERBOARD.md` describes
"Run 6" (2026-03-14, commit `a825e63`) as "Exactly the same launch command as Run 4 except
`--target-param-data-ratio=8`" at `--depth=24`, and that is what ships in `speedrun.sh` at our
pinned commit. Running this package's `run.sh` reproduces Run 6, not an older/weaker leaderboard
entry.

## Data: nanochat's current default is ClimbMix — and it is NC-licensed

This needed checking rather than assuming, because the answer changed partway through nanochat's
history and the task brief's premise ("ClimbMix — used on the leaderboard") undersells how
baked-in it now is. `nanochat/dataset.py` at the pinned commit:

```python
BASE_URL = "https://huggingface.co/datasets/karpathy/climbmix-400b-shuffle/resolve/main"
...
DATA_DIR = os.path.join(base_dir, "base_data_climbmix")
```

`python -m nanochat.dataset -n <N>` — the command `speedrun.sh` runs to fetch pretraining data — has
**no flag to select a different corpus**. As of this commit, ClimbMix is not merely what the
leaderboard's best run happened to use; it is the only dataset the code downloads. `dev/LEADERBOARD.md`
documents the switch: Run 4 (2026-03-03) "the switch from HuggingFace FineWeb-EDU to NVIDIA ClimbMix
dataset... `@karpathy` has tried to swap the dataset many times, each time with a negative result
(FineWeb, DCLM, Olmo), but ClimbMix produced clear and immediate gains." `dataset.py` keeps a
legacy fallback that reads an *already-downloaded* old `base_data` directory if one happens to
exist locally, but there is no code path left to freshly download FineWeb-Edu.

**The license problem:** `research/03-open-data.md` §1.1 and §5.4 (source-verified live against HF
2026-09-13):

| Repo | License | Note |
|---|---|---|
| `nvidia/Nemotron-ClimbMix` (original, GPT-2 token IDs) | **`cc-by-nc-4.0`** | non-commercial |
| `karpathy/climbmix-400b-shuffle` (raw-text mirror — what nanochat actually downloads) | `mit` **repo tag** | pre-shuffled/detokenized copy of the same NC-licensed NVIDIA data; the `mit` tag is on Karpathy's mirror repo, not a re-license of NVIDIA's underlying data |

research/03's own verdict, verbatim: *"⚠️ ClimbMix is CC-BY-NC-4.0. Fine for nanochat-style
research; not for a commercial model."* And from the quick-verdict index: *"Avoid (other): ClimbMix
/ClimbLab (NC + token-IDs + self-contradictory card)."* `docs/PLAN.md` §5 states the project's own
rule plainly: "ClimbMix avoided (NC license)."

**What this means for this rung:** running `run.sh` unmodified reproduces the leaderboard's
published CORE number faithfully (this is the point of Rung 1 — matching a documented result at a
documented cost) but the resulting checkpoint inherits ClimbMix's NC encumbrance and is a
**research artifact, not a commercially clean one**. A commercially-clean Rung 1 run requires
overriding the data source *before* launching — concretely, pointing `nanochat/dataset.py`'s
`BASE_URL`/`DATA_DIR` (or supplying an equivalent local corpus in the same sharded-parquet format
`nanochat.dataset.parquets_iter_batched` expects) at a commercially-clean corpus instead, e.g. the
`allenai/dolma3_mix-150B-1025` slice already used for the local mix in `docs/PLAN.md` §5, or plain
FineWeb-Edu. That swap is a real code change to a cloned dependency, not a config flag, so it is
**not implemented by `run.sh`** (out of scope for this documentation/config package) — it is
recorded here so the choice is explicit rather than silently inherited. Retraining the tokenizer
(`scripts.tok_train`) on the new corpus is also required if the corpus changes, per the data-swap
warning nanochat itself prints (`dataset.py`'s "DATASET UPGRADE REQUIRED" message, same pattern).

## Expected result

`dev/LEADERBOARD.md` (fetched at the pinned commit) and `README.md`'s own leaderboard table agree
exactly:

| # | time (8×H100) | val_bpb | CORE | Description | Date | Commit |
|---|---|---|---|---|---|---|
| 0 | 168 h (32 TPUv3) | – | **0.256525** | Original OpenAI GPT-2 (1.6B) checkpoint | 2019 | – |
| **6** | **1.65 h** | 0.71800 | **0.262634** (0.2626 in the README table) | "backout and smear" architecture changes, autoresearch round 2 | 2026-03-14 | `a825e63` |

This is the same pairing `research/02-open-training-stack.md` §"Stage 1 / The speedruns — current
records" cites: nanochat Run 6 reaches **CORE 0.2626 in 1.65 h**, beating GPT-2 XL's (1.6B, 2019)
**CORE 0.2565** — the exact "d24 1.65 h CORE 0.2626 vs GPT-2 XL 0.2565" pairing named in the build
brief. `LEADERBOARD.md`'s framing of the target, verbatim: *"The primary metric we care about is
'time to GPT-2' - the wall clock time needed to outperform the GPT-2 (1.6B) CORE metric on an
8XH100 GPU node... GPT-2 CORE score is 0.256525."*

**Expected result from running this package: CORE ≥ 0.2565 (beats GPT-2 XL), targeting ≈0.26 in
≈1.65–2 h wall-clock**, matching Run 6 — since `speedrun.sh` at this pinned commit *is* Run 6's
configuration. Non-determinism is real: `dev/LEADERBOARD.md`'s own Run 4 section reports 7 repeat
runs of one identical command spanning CORE 0.2512–0.2677 (mean 0.25714), so a single run landing
a few points off 0.2626 is expected variance, not a bug.

## Cost

Two different numbers, both legitimate, answering different questions:

| | GPU-hours | Spot (\$0.94/H100-h) | On-demand (\$2.43/H100-h) | Source |
|---|---|---|---|---|
| **Ladder-modeled Rung 1** (Chinchilla-optimal 1B dense / 20B tokens — a *different*, larger config than what this package runs) | 96 | **$91** | $234 | `results/ladder.md`, verbatim |
| **This package's actual target (nanochat Run 6, disclosed)** | 13.2 (1.65 h × 8 GPUs) | **$12.41** | $32.08 | H100-hours from `dev/LEADERBOARD.md`; \$/h from `research/04-landscape-and-compute.md` §B.2 (Prime Intellect, verified 2026-09-13), same rate the ladder uses |

`results/ladder.md`'s own note on this row explains the gap: nanochat's leaderboard doesn't
disclose Run 6's exact parameter count and token count, so the ladder's Rung 1 row uses a
Chinchilla-optimal 1B/20B shape instead of reverse-engineering Run 6's. **`run.sh` in this
directory reproduces Run 6 (13.2 H100-h), not the ladder's modeled 96-H100-h row** — treat the
$12–32 figure as this package's real price tag.

For reference, nanochat's own README quotes a third number, at a different assumed rate: *"you can
train your own GPT-2 capability LLM... for only $48 (~2 hours of 8XH100 GPU node)... On a spot
instance, the total cost can be closer to ~$15,"* computed at their own assumed "~$3/GPU/hr" rather
than the Sept-2026 Prime Intellect rates used elsewhere in this repo. All three numbers are
internally consistent (same ballpark, different price floors and slightly different wall-clock
assumptions); use the $12.41/$32.08 pair above for anything reported into `docs/RESULTS.md`, since
it's on this repo's own sourced Sept-2026 prices.

Budget the full `speedrun.sh` wall-clock (~1.5 h per its header, ~2 h per the README's rounder
estimate), not just the 1.65 h base-training figure — tokenizer training and data download add
time before pretraining starts, and SFT + SFT-eval run after it.

## Eval / report steps

`speedrun.sh` already runs nanochat's own evaluation as part of the single documented command —
there is no separate "now go eval it" step for the reference path:

- **`scripts.base_eval`** (invoked mid-script) computes the **CORE score** (the DCLM-paper ensemble
  metric over 22 evaluations) and bits-per-byte on train/val for the base pretrained model — this
  is the number to compare against GPT-2 XL's 0.256525.
- **`scripts.chat_eval -- -i sft`** (invoked at the end) evaluates the SFT'd chat model on nanochat's
  own task suite (`tasks/arc.py`, `tasks/gsm8k.py`, `tasks/humaneval.py`, `tasks/mmlu.py`,
  `tasks/smoltalk.py` per the fetched file tree).

**This repo's own `scripts/eval.sh` does not apply directly.** Its usage banner states it expects
"an r52 checkpoint directory (`runs/<run>/ckpt/best`) or an mlx-lm model directory
(`models/<name>-mlx`)" — both MLX-native formats. nanochat's checkpoint is a PyTorch checkpoint in
its own format (`nanochat/checkpoint_manager.py`). Whether that checkpoint can be exported to an
MLX-loadable directory (e.g. via `mlx-lm`'s HF-checkpoint conversion tooling, if nanochat's
checkpoint format is or can be made HF-compatible) is **UNVERIFIED** — not checked in this session
(no downloads or conversions were run, per the build rules). Until that path is confirmed, treat
`base_eval`'s CORE number as the reference result, and record it (with the exact command, commit,
wall-clock, and cost) in `docs/RESULTS.md` per the checklist in `rungs/README.md`.

## Files

| File | What |
|---|---|
| `README.md` | this file |
| `run.sh` | clones the pinned commit and runs `bash runs/speedrun.sh`; `--dry-run` prints the plan; `bash -n` clean |
