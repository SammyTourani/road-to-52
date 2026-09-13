#!/usr/bin/env bash
# Copyright 2026 The road-to-52 authors. SPDX-License-Identifier: Apache-2.0
#
# Run the evaluation suite on one model and record the numbers.
#
#   scripts/eval.sh <model-dir> [suite] [-- extra args passed to every eval]
#
#   <model-dir>  an r52 checkpoint directory (runs/<run>/ckpt/best) or an mlx-lm model
#                directory (models/<name>-mlx, models/gpt2-mlx).
#   [suite]      standard  val_loss + hellaswag                       (default)
#                full      val_loss + hellaswag + core (all 22 tasks)
#                quick     the same three, --max-tokens 262144 --limit 50   (smoke test)
#                val_loss | hellaswag | core   -- just that one
#
# Environment:
#   R52_RUN         results/<run>/ directory        (default: derived from <model-dir>)
#   R52_MODEL_ID    bar.yaml model id for the gap tracker (default: the run name)
#   R52_MEM_GIB     MLX memory guideline in GiB     (default: 3)
#   R52_BLOCK_SIZE  val_loss window length, tokens  (default: 1024)
#   R52_PYTHON      interpreter                     (default: .venv/bin/python)
#   R52_RESULTS_DIR write results here instead of results/ (also moves the markdown row to
#                   <dir>/RESULTS.md -- use it for smoke runs so docs/RESULTS.md stays clean)
#
# Everything runs under `nice -n 10` with an MLX memory cap, because this machine also runs a
# multi-day pretraining job; see docs/ARCHITECTURE.md §1.2.  Each eval appends a row to
# docs/RESULTS.md and writes results/<run>/{val_loss,hellaswag,core}.json plus the aggregate
# results/<run>/eval.json that r52/bar/gap.py reads.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

if [[ $# -lt 1 ]]; then
  sed -n '3,26p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
  exit 2
fi

MODEL="$1"; shift
SUITE="${1:-standard}"
if [[ $# -gt 0 && "$1" != "--" ]]; then shift; fi
if [[ "${1:-}" == "--" ]]; then shift; fi
EXTRA=("$@")

PYTHON="${R52_PYTHON:-$REPO_ROOT/.venv/bin/python}"
MEM_GIB="${R52_MEM_GIB:-3}"
BLOCK_SIZE="${R52_BLOCK_SIZE:-1024}"

if [[ ! -d "$MODEL" ]]; then
  echo "eval.sh: $MODEL is not a directory" >&2
  exit 2
fi

# runs/<name>/ckpt/best -> <name>;  models/<name>-mlx -> <name>-mlx
default_run() {
  local p; p="$(cd "$MODEL" && pwd)"
  if [[ "$p" == */ckpt/* ]]; then
    basename "${p%%/ckpt/*}"
  else
    basename "$p"
  fi
}
RUN="${R52_RUN:-$(default_run)}"
MODEL_ID="${R52_MODEL_ID:-$RUN}"

COMMON=(--memory-limit-gib "$MEM_GIB" --report --run "$RUN" --model-id "$MODEL_ID")
if [[ -n "${R52_RESULTS_DIR:-}" ]]; then COMMON+=(--results-dir "$R52_RESULTS_DIR"); fi

MAX_TOKENS=""
LIMIT=""
case "$SUITE" in
  quick)    MAX_TOKENS="--max-tokens 262144"; LIMIT="--limit 50" ;;
  standard|full|val_loss|hellaswag|core) ;;
  *) echo "eval.sh: unknown suite '$SUITE' (standard|full|quick|val_loss|hellaswag|core)" >&2; exit 2 ;;
esac

echo "== r52 eval suite =================================================="
echo "model   : $MODEL"
echo "suite   : $SUITE"
echo "run     : ${R52_RESULTS_DIR:-results}/$RUN/   (model id '$MODEL_ID')"
echo "memory  : ${MEM_GIB} GiB guideline, nice -n 10"
echo "commit  : $(git rev-parse --short HEAD 2>/dev/null || echo unknown)"
echo "started : $(date -Iseconds)"
echo "===================================================================="

run_eval() {
  local name="$1"; shift
  echo
  echo "-- $name ------------------------------------------------------------"
  # shellcheck disable=SC2086
  nice -n 10 "$PYTHON" -m "r52.eval.$name" --model "$MODEL" "${COMMON[@]}" "$@" "${EXTRA[@]+"${EXTRA[@]}"}"
}

case "$SUITE" in
  val_loss)  run_eval val_loss  --block-size "$BLOCK_SIZE" $MAX_TOKENS ;;
  hellaswag) run_eval hellaswag $LIMIT ;;
  core)      run_eval core      $LIMIT ;;
  standard)
    run_eval val_loss  --block-size "$BLOCK_SIZE" $MAX_TOKENS
    run_eval hellaswag $LIMIT
    ;;
  quick|full)
    run_eval val_loss  --block-size "$BLOCK_SIZE" $MAX_TOKENS
    run_eval hellaswag $LIMIT
    run_eval core      $LIMIT
    ;;
esac

echo
echo "== done: $(date -Iseconds) ==========================================="
if [[ -n "${R52_RESULTS_DIR:-}" ]]; then
  echo "$R52_RESULTS_DIR/$RUN/ and $R52_RESULTS_DIR/RESULTS.md updated"
else
  echo "results/$RUN/ and docs/RESULTS.md updated"
fi
