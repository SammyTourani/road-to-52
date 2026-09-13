#!/usr/bin/env bash
# Copyright 2026 The road-to-52 authors. SPDX-License-Identifier: Apache-2.0
#
# Run one ablation axis, cell by cell, resumably.
#
#   scripts/ablate.sh <axis> [options] [-- extra args for r52.ablate.run ...]
#
#   <axis>          tokenizer | corpus | arch   (docs/ABLATIONS.md; cheapest axis first)
#
#   --dry-run       print the plan -- every cell, its budget, and the exact train/eval
#                   commands -- and run nothing
#   --force         (a) start even though another r52.train is already on the GPU, and
#                   (b) re-run cells whose results JSON already exists
#   --cell NAME     just this one cell
#   --list          print the axis's cells and their status, then exit
#   --tokenizer DIR shared BPE for the axis (default: data/tokenizers/fineweb32k)
#   --config PATH   base config (default: configs/ablations/tiny_abl.yaml)
#   --corpus NAME   corpus for the arch and tokenizer axes (default: fineweb-edu)
#   --results DIR   results root (default: results/ablations)
#
# Resumable: a cell whose results/ablations/<axis>/<cell>.json exists is skipped, so the
# script can be interrupted and restarted, and an axis can be run over several evenings.
#
# One GPU, one training job: docs/ABLATIONS.md says "Runs never overlap with a headline run",
# so this refuses to start while another `r52.train` process is alive unless --force is given.
# Everything runs under `nice -n 10` (docs/ARCHITECTURE.md §1.2).
#
# After the last cell it renders results/ablations/<axis>.md.
#
# Written for bash 3.2 (macOS ships nothing newer): no `mapfile`, and every array expansion
# is guarded for `set -u`.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"

if [ $# -lt 1 ]; then
  sed -n '3,28p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
  exit 2
fi

AXIS="$1"; shift
PY="${R52_PYTHON:-$REPO/.venv/bin/python}"
DRY=0
FORCE=0
LIST=0
ONE_CELL=""
RESULTS="results/ablations"
MATRIX_ARGS=()
EXTRA=()

while [ $# -gt 0 ]; do
  case "$1" in
    --dry-run)   DRY=1; shift ;;
    --force)     FORCE=1; shift ;;
    --list)      LIST=1; shift ;;
    --cell)      ONE_CELL="$2"; shift 2 ;;
    --tokenizer) MATRIX_ARGS+=(--tokenizer "$2"); shift 2 ;;
    --config)    MATRIX_ARGS+=(--config "$2"); shift 2 ;;
    --corpus)    MATRIX_ARGS+=(--corpus "$2"); shift 2 ;;
    --results)   RESULTS="$2"; shift 2 ;;
    --)          shift; EXTRA=("$@"); break ;;
    *)           echo "ablate.sh: unknown option '$1'" >&2; exit 2 ;;
  esac
done

[ -x "$PY" ] || { echo "python not found at $PY (set R52_PYTHON)" >&2; exit 1; }

case "$AXIS" in
  tokenizer|corpus|arch) ;;
  *) echo "ablate.sh: unknown axis '$AXIS' (tokenizer|corpus|arch)" >&2; exit 2 ;;
esac

# -- the cell list, straight out of r52.ablate.matrix ---------------------------------
CELLS=()
if [ -n "$ONE_CELL" ]; then
  CELLS=("$ONE_CELL")
else
  CELL_LIST="$("$PY" -m r52.ablate.matrix --axis "$AXIS" --slugs \
    ${MATRIX_ARGS[@]+"${MATRIX_ARGS[@]}"})"
  while IFS= read -r line; do
    [ -n "$line" ] && CELLS+=("$line")
  done <<EOF
$CELL_LIST
EOF
fi

if [ "${#CELLS[@]}" -eq 0 ]; then
  echo "ablate.sh: axis '$AXIS' produced no cells" >&2
  exit 1
fi

if [ "$LIST" -eq 1 ]; then
  echo "axis $AXIS -- ${#CELLS[@]} cells (results in $RESULTS/$AXIS/)"
  for c in "${CELLS[@]}"; do
    if [ -f "$RESULTS/$AXIS/$c.json" ]; then st=done; else st=pending; fi
    printf '  %-24s %s\n' "$c" "$st"
  done
  exit 0
fi

# -- the GPU is single-tenant ----------------------------------------------------------
BUSY="$(pgrep -fl 'r52\.train' 2>/dev/null | grep -v 'ablate' || true)"
if [ -n "$BUSY" ] && [ "$FORCE" -eq 0 ] && [ "$DRY" -eq 0 ]; then
  echo "ablate.sh: another r52.train is running; an ablation would share the GPU with it:" >&2
  echo "$BUSY" >&2
  echo "Wait for it to finish, or pass --force if you really mean to overlap." >&2
  exit 1
fi

echo "== r52 ablation: $AXIS ============================================="
echo "cells    : ${#CELLS[@]}   (${CELLS[*]})"
echo "results  : $RESULTS/$AXIS/"
echo "python   : $PY"
[ "$DRY" -eq 1 ] && echo "mode     : DRY RUN (nothing will be trained)"
[ -n "$BUSY" ] && echo "warning  : overlapping with -> $BUSY"
echo "started  : $(date -Iseconds)"
echo "===================================================================="

RC=0
for cell in "${CELLS[@]}"; do
  if [ "$DRY" -eq 0 ] && [ "$FORCE" -eq 0 ] && [ -f "$RESULTS/$AXIS/$cell.json" ]; then
    echo
    echo "-- $cell: already done, skipping ($RESULTS/$AXIS/$cell.json)"
    continue
  fi
  echo
  echo "-- $cell ------------------------------------------------------------"
  ARGS=(--axis "$AXIS" --cell "$cell" --results-dir "$RESULTS")
  [ "${#MATRIX_ARGS[@]}" -gt 0 ] && ARGS+=("${MATRIX_ARGS[@]}")
  [ "$DRY" -eq 1 ] && ARGS+=(--dry-run)
  [ "$FORCE" -eq 1 ] && ARGS+=(--force)
  if ! nice -n 10 "$PY" -m r52.ablate.run "${ARGS[@]}" ${EXTRA[@]+"${EXTRA[@]}"}; then
    echo "ablate.sh: cell '$cell' failed; continuing with the rest" >&2
    RC=1
  fi
done

echo
if [ "$DRY" -eq 0 ]; then
  nice -n 10 "$PY" -m r52.ablate.report --axis "$AXIS" --results-dir "$RESULTS" \
    ${MATRIX_ARGS[@]+"${MATRIX_ARGS[@]}"}
fi
echo "== done: $(date -Iseconds) ==========================================="
exit "$RC"
