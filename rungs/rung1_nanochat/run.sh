#!/usr/bin/env bash
# Copyright 2026 The road-to-52 authors. SPDX-License-Identifier: Apache-2.0
#
# Rung 1 -- nanochat-class ~1B dense model on a rented 8xH100 node.
# See README.md in this directory for the full writeup: pinned commit, data-license caveat,
# expected CORE score, and cost. This script does the setup and speedrun.sh invocation exactly
# as documented in nanochat's own README (quoted below and in README.md), and nothing else --
# it does not modify nanochat, does not change its data source, and does not add techniques.
#
# WHAT THIS DOES, STEP BY STEP:
#   1. git clone karpathy/nanochat and check out the pinned commit (below).
#   2. Run its own runs/speedrun.sh unmodified. That single script does nanochat's own setup
#      (uv sync), tokenizer training, base pretraining (depth-24, fp8, ~$100-class), the CORE
#      eval, SFT, and the SFT eval -- see the upstream lines quoted under STAGE 2 below.
#   3. Print where the results/checkpoints landed and remind you to log the run in docs/RESULTS.md.
#
# Usage:
#   rungs/rung1_nanochat/run.sh [--dry-run]
#
#   --dry-run   print every command this script would run, and nanochat's own cost estimate,
#               without touching the filesystem, network, or a GPU. Always run this first.
#
# Environment (all optional):
#   R52_WORKDIR          Where to clone nanochat.                 default: $PWD/nanochat-rung1
#   R52_NANOCHAT_REPO    Git remote to clone.                      default: https://github.com/karpathy/nanochat
#   R52_NANOCHAT_COMMIT  Commit to pin and check out.               default: 92d63d4e8bb4df75c3b71618f31ddde2378b2bcd
#                        (verified 2026-09-13 via
#                         `gh api repos/karpathy/nanochat/commits/master --jq .sha`; re-run that
#                         yourself before a real launch if much time has passed, and update this
#                         default deliberately -- do not silently float to a new HEAD.)
#   NANOCHAT_BASE_DIR    nanochat's OWN env var (see runs/speedrun.sh) for its cache/checkpoint
#                        dir. Point this at the provider's fast/persistent disk, e.g.
#                        /workspace/.cache/nanochat, so a spot preemption doesn't lose the
#                        downloaded data shards. default: nanochat's own default, ~/.cache/nanochat
#   WANDB_RUN            Passed straight through to speedrun.sh for W&B logging.
#                        default: unset -> speedrun.sh uses "dummy" (no wandb; see its own source)
#
# THIS SCRIPT SPENDS MONEY WHEN RUN FOR REAL, ON A REAL RENTED 8xH100 NODE.
# See ../README.md "The rule: paid runs are the owner's decision" before running without --dry-run.
#
# DATA-LICENSE NOTE (read this before treating the output as a "clean" artifact):
# Unmodified, speedrun.sh downloads nanochat's current default pretraining corpus, ClimbMix, via
# `python -m nanochat.dataset` (BASE_URL in nanochat/dataset.py points at
# huggingface.co/datasets/karpathy/climbmix-400b-shuffle -- verified by fetching that file at the
# pinned commit). The underlying NVIDIA ClimbMix data is CC-BY-NC-4.0 (research/03-open-data.md
# 1.1, 5.4, 10.1: "ClimbMix avoided (NC license)"). Running this script unmodified reproduces the
# nanochat leaderboard's CORE number honestly, but the resulting checkpoint is NOT commercially
# clean. See README.md "Data" section for what swapping the data source involves.
set -euo pipefail

# -----------------------------------------------------------------------------------------------
# Config (env-overridable; see the usage comment above)
R52_WORKDIR="${R52_WORKDIR:-$PWD/nanochat-rung1}"
R52_NANOCHAT_REPO="${R52_NANOCHAT_REPO:-https://github.com/karpathy/nanochat}"
R52_NANOCHAT_COMMIT="${R52_NANOCHAT_COMMIT:-92d63d4e8bb4df75c3b71618f31ddde2378b2bcd}"
DRY_RUN=0

for arg in "$@"; do
  case "$arg" in
    --dry-run) DRY_RUN=1 ;;
    -h|--help)
      sed -n '3,45p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
      exit 0
      ;;
    *)
      echo "run.sh: unknown argument '$arg' (only --dry-run / --help are supported)" >&2
      exit 2
      ;;
  esac
done

# run <description> -- <command...>
# In --dry-run mode, prints the command instead of executing it.
run() {
  local desc="$1"; shift
  echo ">> $desc"
  if [[ "$DRY_RUN" -eq 1 ]]; then
    printf '   [dry-run]'; printf ' %q' "$@"; echo
  else
    "$@"
  fi
}

echo "== road-to-52 Rung 1 -- nanochat-class ~1B dense on 8xH100 =========================="
echo "mode          : $([[ $DRY_RUN -eq 1 ]] && echo 'DRY RUN (nothing will execute)' || echo 'LIVE -- this spends money')"
echo "repo          : $R52_NANOCHAT_REPO"
echo "pinned commit : $R52_NANOCHAT_COMMIT"
echo "workdir       : $R52_WORKDIR"
echo "NANOCHAT_BASE_DIR : ${NANOCHAT_BASE_DIR:-<nanochat default: ~/.cache/nanochat>}"
echo "WANDB_RUN     : ${WANDB_RUN:-<unset -> speedrun.sh uses 'dummy', no wandb>}"
echo
echo "Cost (see README.md for full sourcing):"
echo "  ladder-modeled Rung 1 row (Chinchilla-optimal 1B/20B, results/ladder.md): "
echo "    96 H100-h @35% MFU -> \$91 spot / \$234 on-demand @ Prime Intellect Sept-2026 rates"
echo "  nanochat's own disclosed record (Run 6, 1.65 h wall-clock x 8 GPUs = 13.2 H100-h):"
echo "    13.2 H100-h -> \$12.41 spot / \$32.08 on-demand @ the same Prime Intellect rates"
echo "  nanochat README's own estimate, at ITS assumed ~\$3/GPU-h: \"~\$48\" on-demand, \"~\$15\" spot"
echo "  actual speedrun.sh wall-clock per its own header comment: \"approximately 1.5 hours\""
echo "======================================================================================"
echo

if [[ "$DRY_RUN" -ne 1 ]] && ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "run.sh: no nvidia-smi found. This must run on the rented 8xH100 node, not locally." >&2
  echo "        (re-run with --dry-run to preview commands from any machine)" >&2
  exit 1
fi

# -----------------------------------------------------------------------------------------------
# STAGE 1 -- clone and pin.
#
# Upstream gives no single documented clone command (the README assumes you already have the
# repo checked out on the GPU box); this is the standard pin-a-commit sequence.
run "clone nanochat" git clone "$R52_NANOCHAT_REPO" "$R52_WORKDIR"

if [[ "$DRY_RUN" -eq 1 ]]; then
  echo ">> cd $R52_WORKDIR && git checkout $R52_NANOCHAT_COMMIT"
else
  cd "$R52_WORKDIR"
  git checkout "$R52_NANOCHAT_COMMIT"
  actual_sha="$(git rev-parse HEAD)"
  if [[ "$actual_sha" != "$R52_NANOCHAT_COMMIT" ]]; then
    echo "run.sh: checked-out HEAD ($actual_sha) does not match pinned commit ($R52_NANOCHAT_COMMIT)" >&2
    exit 1
  fi
  echo ">> pinned at $actual_sha"
fi

# -----------------------------------------------------------------------------------------------
# STAGE 2 -- the exact upstream setup + speedrun invocation.
#
# Quoted verbatim from nanochat's README.md at the pinned commit ("Getting started" / "Reproduce
# and talk to GPT-2"):
#
#   uv sync --extra gpu    # Use for CUDA (A100/H100/etc.)
#   source .venv/bin/activate
#   ...
#   bash runs/speedrun.sh
#
# speedrun.sh (runs/speedrun.sh, same commit) does ALL of the above itself -- it installs uv if
# missing, runs `uv sync --extra gpu`, activates the venv, downloads ~170 pretraining data shards,
# trains the tokenizer, then runs (its own comments, quoted):
#   "d24 model (slightly undertrained to beat GPT-2 => decrease data:params ratio from compute
#    optimal 10.5 (default) to 8)"
#   torchrun --standalone --nproc_per_node=8 -m scripts.base_train -- --depth=24 \
#       --target-param-data-ratio=8 --device-batch-size=16 --fp8 --run=$WANDB_RUN
#   torchrun --standalone --nproc_per_node=8 -m scripts.base_eval -- --device-batch-size=16
#   torchrun --standalone --nproc_per_node=8 -m scripts.chat_sft -- --run=$WANDB_RUN
#   torchrun --standalone --nproc_per_node=8 -m scripts.chat_eval -- -i sft
#
# i.e. speedrun.sh already runs the CORE eval (base_eval) and the SFT eval (chat_eval) as part of
# the pipeline -- there is no separate "run eval" step for the reference path. We invoke the
# single documented command rather than re-implementing its internals, so this package never
# drifts from whatever the pinned commit actually does.
if [[ "$DRY_RUN" -eq 1 ]]; then
  echo ">> bash runs/speedrun.sh   # ~1.5 h on 8xH100 per upstream; see README.md for the full quote"
else
  run "run nanochat's speedrun.sh (setup + tokenizer + pretrain + CORE eval + SFT + SFT eval)" \
    bash runs/speedrun.sh
fi

# -----------------------------------------------------------------------------------------------
# STAGE 3 -- report.
echo
echo "== done (or, in --dry-run mode, this is everything that would have run) ============"
echo "Checkpoints and logs: \${NANOCHAT_BASE_DIR:-~/.cache/nanochat} (nanochat's own layout)."
echo "The CORE score and total_training_time printed by scripts.base_eval / your wandb run"
echo "(if WANDB_RUN was set) are the numbers to copy into docs/RESULTS.md -- see"
echo "rungs/README.md \"Checklist - running a rung honestly\" for the exact fields to record"
echo "(command, commit, wall-clock, \$ actually billed by the provider, eval conditions)."
echo
echo "scripts/eval.sh (this repo's own MLX eval suite) does not run directly on nanochat's"
echo "PyTorch checkpoint -- it expects an r52 checkpoint dir or an mlx-lm model dir. Whether"
echo "nanochat's checkpoint_manager output can be converted to an MLX-loadable directory (e.g."
echo "via mlx-lm's HF-checkpoint conversion tooling) is UNVERIFIED / not attempted by this"
echo "package. Treat scripts/base_eval.py's CORE number (already produced by Stage 2 above) as"
echo "the reference result unless/until that conversion path is confirmed."
echo
echo "Remember to shut the node down in the provider's console -- see rungs/README.md checklist"
echo "item 7. This script does not do it for you."
