#!/usr/bin/env bash
# Copyright 2026 The road-to-52 authors. SPDX-License-Identifier: Apache-2.0
#
# Launch a detached pretraining run that survives the shell (and the Claude Code session).
#
#   scripts/train.sh <config.yaml> <run_name> [extra args for r52.train ...]
#   scripts/train.sh configs/gpt2_124m_mac.yaml b1
#   scripts/train.sh configs/nano_30m.yaml a1 --resume --max-hours 20
#
# Writes runs/<run_name>/{log.jsonl,stdout.log,ckpt/} and prints the PID.
# Runs under `nice -n 10` so the machine stays usable for everything else.
set -euo pipefail

if [ $# -lt 2 ]; then
  sed -n '3,12p' "$0" >&2
  exit 2
fi

CONFIG="$1"; shift
RUN_NAME="$1"; shift

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"

PY="${R52_PYTHON:-$REPO/.venv/bin/python}"
[ -x "$PY" ] || { echo "python not found at $PY (set R52_PYTHON)" >&2; exit 1; }
[ -f "$CONFIG" ] || { echo "config not found: $CONFIG" >&2; exit 1; }

OUT_DIR="$($PY - "$CONFIG" <<'PYEOF'
import sys
from r52.config import load_config
print(load_config(sys.argv[1]).train.out_dir)
PYEOF
)"

RUN_DIR="$OUT_DIR/$RUN_NAME"
mkdir -p "$RUN_DIR/ckpt"
STDOUT="$RUN_DIR/stdout.log"
PIDFILE="$RUN_DIR/train.pid"

if [ -f "$PIDFILE" ] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null; then
  echo "run '$RUN_NAME' is already running as PID $(cat "$PIDFILE")" >&2
  exit 1
fi

nohup nice -n 10 "$PY" -m r52.train "$CONFIG" --run-name "$RUN_NAME" "$@" >>"$STDOUT" 2>&1 &
PID=$!
echo "$PID" > "$PIDFILE"

cat <<EOF
started  run=$RUN_NAME  pid=$PID
config   $CONFIG
stdout   $RUN_DIR/stdout.log
jsonl    $RUN_DIR/log.jsonl
ckpt     $RUN_DIR/ckpt/
follow   tail -f $RUN_DIR/stdout.log
stop     kill -TERM $PID     # checkpoints, then exits cleanly
EOF
