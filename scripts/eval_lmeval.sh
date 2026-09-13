#!/usr/bin/env bash
# Copyright 2026 The road-to-52 authors. SPDX-License-Identifier: Apache-2.0
#
# Run the lm-evaluation-harness bridge on one exported model and record the numbers.
#
#   scripts/eval_lmeval.sh <model-dir> <quick|standard> [--limit N] [more r52.eval.lmeval args]
#
#   <model-dir>  an mlx-lm model directory (models/<name>-mlx, models/gpt2-mlx) or an
#                HF / mlx-community repo id.  NOT a runs/<run>/ckpt/best checkpoint --
#                export it first with `python -m r52.export <ckpt> models/<name>-mlx`.
#   quick        arc_challenge piqa winogrande lambada_openai hellaswag
#                -- loglikelihood only, 0-shot, cheap.
#   standard     mmlu(5) gpqa_diamond_zeroshot ifeval humaneval hendrycks_math500
#                gsm8k(8) mmlu_pro(5) -- five of seven generative.  HOURS TO DAYS at full
#                size on this machine (research/05-apple-silicon-training.md §7.3 puts
#                unlimited MMLU-Pro at 2-4 days).  ALWAYS pass --limit.
#
# Environment:
#   R52_RUN          results/<run>/ directory      (default: the model directory's name)
#   R52_MODEL_ID     bar.yaml model id for the gap tracker (default: the run name)
#   R52_MEM_GIB      MLX memory guideline in GiB   (default: 3)
#   R52_BATCH_SIZE   generation batch size         (default: 8)
#   R52_PYTHON       interpreter                   (default: .venv/bin/python)
#   R52_RESULTS_DIR  write results here instead of results/ (also moves the markdown row to
#                    <dir>/RESULTS.md -- use it for smoke runs so docs/RESULTS.md stays clean)
#   R52_OUTPUT_DIR   also dump lm-eval's own result blocks here, one JSON per task
#
# Everything runs under `nice -n 10` with a 3 GiB MLX memory guideline, because this machine
# also runs a multi-day pretraining job; see docs/ARCHITECTURE.md §1.2 and docs/DEVIATIONS.md
# E5.  Each task appends a row to docs/RESULTS.md and writes
# results/<run>/lmeval_<task>.json plus the aggregate results/<run>/eval.json that
# r52/bar/gap.py reads.
#
# Examples:
#   scripts/eval_lmeval.sh models/gpt2-mlx quick --limit 100
#   scripts/eval_lmeval.sh models/b1-mlx standard --limit 50 --confirm-run-unsafe-code
#   R52_MODEL_ID=gpt2-124m scripts/eval_lmeval.sh models/gpt2-mlx quick --limit 500

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

if [[ $# -lt 1 ]]; then
  sed -n '3,40p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
  exit 2
fi

MODEL="$1"; shift
SUITE="${1:-quick}"
if [[ $# -gt 0 && "$1" != -* ]]; then shift; fi
if [[ "${1:-}" == "--" ]]; then shift; fi
EXTRA=("$@")

case "$SUITE" in
  quick|standard) ;;
  *) echo "eval_lmeval.sh: unknown suite '$SUITE' (quick|standard)" >&2; exit 2 ;;
esac

PYTHON="${R52_PYTHON:-$REPO_ROOT/.venv/bin/python}"
MEM_GIB="${R52_MEM_GIB:-3}"
BATCH_SIZE="${R52_BATCH_SIZE:-8}"

# models/<name>-mlx -> <name>-mlx;  an HF repo id -> org__name
default_run() {
  if [[ -d "$MODEL" ]]; then basename "$(cd "$MODEL" && pwd)"; else echo "${MODEL//\//__}"; fi
}
RUN="${R52_RUN:-$(default_run)}"
MODEL_ID="${R52_MODEL_ID:-$RUN}"

# `git` is unavailable on this machine (unaccepted Xcode CLT licence), so the commit is read
# straight out of .git -- the same thing r52.eval.lmeval.git_commit does for the JSON.
COMMIT="$("$PYTHON" -c 'from r52.eval.lmeval import git_commit; print(git_commit())' 2>/dev/null || echo unknown)"

if [[ ! -d "$MODEL" && "$MODEL" != */* ]]; then
  echo "eval_lmeval.sh: '$MODEL' is neither a directory nor an HF repo id (org/name)" >&2
  exit 2
fi
if [[ -d "$MODEL" && -f "$MODEL/meta.json" && ! -f "$MODEL/config.json" ]]; then
  echo "eval_lmeval.sh: $MODEL is an r52 checkpoint, not an mlx-lm model directory." >&2
  echo "  python -m r52.export $MODEL models/<name>-mlx" >&2
  exit 2
fi

ARGS=(--model "$MODEL" --suite "$SUITE"
      --batch-size "$BATCH_SIZE" --memory-limit-gib "$MEM_GIB"
      --report --run "$RUN" --model-id "$MODEL_ID")
if [[ -n "${R52_RESULTS_DIR:-}" ]]; then ARGS+=(--results-dir "$R52_RESULTS_DIR"); fi
if [[ -n "${R52_OUTPUT_DIR:-}" ]]; then ARGS+=(--output-dir "$R52_OUTPUT_DIR"); fi

echo "== r52 lm-eval bridge =============================================="
echo "model   : $MODEL"
echo "suite   : $SUITE"
echo "run     : ${R52_RESULTS_DIR:-results}/$RUN/   (model id '$MODEL_ID')"
echo "memory  : ${MEM_GIB} GiB guideline, nice -n 10, batch ${BATCH_SIZE}"
echo "commit  : $COMMIT"
echo "started : $(date -Iseconds)"
if [[ "$SUITE" == "standard" && ! " ${EXTRA[*]-} " =~ [[:space:]]--limit[[:space:]] ]]; then
  echo "WARNING : 'standard' without --limit is hours to days on this machine."
fi
echo "===================================================================="
echo

nice -n 10 "$PYTHON" -m r52.eval.lmeval "${ARGS[@]}" "${EXTRA[@]+"${EXTRA[@]}"}"

echo
echo "== done: $(date -Iseconds) ==========================================="
if [[ -n "${R52_RESULTS_DIR:-}" ]]; then
  echo "$R52_RESULTS_DIR/$RUN/ and $R52_RESULTS_DIR/RESULTS.md updated"
else
  echo "results/$RUN/ and docs/RESULTS.md updated"
fi
