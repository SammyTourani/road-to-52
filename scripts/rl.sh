#!/usr/bin/env bash
# Copyright 2026 The road-to-52 authors. SPDX-License-Identifier: Apache-2.0
#
# Launch a detached RLVR (GRPO/DAPO) run on an EXPORTED model directory.
#
#   scripts/rl.sh <config.yaml> <run_name> [extra args for r52.posttrain.rl ...]
#   scripts/rl.sh configs/posttrain/rl_nano.yaml r1 --model models/nano-sft-mlx
#   scripts/rl.sh configs/posttrain/rl_nano.yaml r1-lora --backend mlx-lm-lora
#
# Run the pass@k probe first (docs/PLAN.md §3.1 item 5):
#   python -m r52.posttrain.passk models/nano-sft-mlx --task chain_sum -n 64 -k 16
#
# Writes runs/<run_name>/{log.jsonl,stdout.log,final/} and prints the PID.
set -euo pipefail

if [ $# -lt 2 ]; then sed -n '3,15p' "$0" >&2; exit 2; fi

CONFIG="$1"; shift
RUN_NAME="$1"; shift

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"
PY="${R52_PYTHON:-$REPO/.venv/bin/python}"
[ -x "$PY" ] || { echo "python not found at $PY (set R52_PYTHON)" >&2; exit 1; }
[ -f "$CONFIG" ] || { echo "config not found: $CONFIG" >&2; exit 1; }

OUT_DIR="$($PY - "$CONFIG" <<'PYEOF'
import sys
from r52.posttrain.config import load_rl_config
print(load_rl_config(sys.argv[1]).out_dir)
PYEOF
)"

RUN_DIR="$OUT_DIR/$RUN_NAME"
mkdir -p "$RUN_DIR"
PIDFILE="$RUN_DIR/rl.pid"
if [ -f "$PIDFILE" ] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null; then
  echo "run '$RUN_NAME' is already running as PID $(cat "$PIDFILE")" >&2; exit 1
fi

nohup nice -n 10 "$PY" -m r52.posttrain.rl "$CONFIG" --run-name "$RUN_NAME" "$@" \
  >>"$RUN_DIR/stdout.log" 2>&1 &
PID=$!
echo "$PID" > "$PIDFILE"

cat <<EOF
started  rl run=$RUN_NAME  pid=$PID
config   $CONFIG
stdout   $RUN_DIR/stdout.log
jsonl    $RUN_DIR/log.jsonl
weights  $RUN_DIR/final/      (an mlx-lm model directory; chat with scripts/chat.sh)
follow   tail -f $RUN_DIR/stdout.log
stop     kill -TERM $PID
EOF
