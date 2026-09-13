#!/usr/bin/env bash
# Copyright 2026 The road-to-52 authors. SPDX-License-Identifier: Apache-2.0
#
# Detached launcher for the v1.0 job queue (r52/queue.py) -- the supervisor that keeps the
# machine working through the whole plan without a human or the planner in the loop: one
# GPU, one job at a time, each resumable, the next starting the moment the previous finishes.
#
#   scripts/queue.sh [queue-file]          start the supervisor, detached, nice -n 5
#                                           (default queue-file: queue/v1.yaml)
#   scripts/queue.sh status [queue-file]   print one status table and exit
#
# Writes runs/queue/{state.json,<id>.log,supervisor.log,supervisor.pid}. Full job/state
# schema and scheduling semantics are in r52/queue.py's module docstring.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"

PY="${R52_PYTHON:-$REPO/.venv/bin/python}"
[ -x "$PY" ] || { echo "python not found at $PY (set R52_PYTHON)" >&2; exit 1; }

RUN_DIR="runs/queue"
PIDFILE="$RUN_DIR/supervisor.pid"
LOG="$RUN_DIR/supervisor.log"

if [ "${1:-}" = "status" ]; then
  shift
  QFILE="${1:-queue/v1.yaml}"
  exec "$PY" -m r52.queue status "$QFILE"
fi

QFILE="${1:-queue/v1.yaml}"
[ -f "$QFILE" ] || { echo "queue file not found: $QFILE" >&2; exit 1; }

mkdir -p "$RUN_DIR"
if [ -f "$PIDFILE" ] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null; then
  echo "queue supervisor already running as PID $(cat "$PIDFILE")" >&2
  exit 1
fi

nohup nice -n 5 "$PY" -m r52.queue run "$QFILE" >>"$LOG" 2>&1 &
PID=$!
echo "$PID" > "$PIDFILE"

cat <<EOF
started  queue supervisor  pid=$PID
queue    $QFILE
log      $LOG
state    $RUN_DIR/state.json
follow   tail -f $LOG
status   scripts/queue.sh status
stop     kill -TERM $PID     # in-flight jobs keep running; the supervisor re-attaches to
                              # them (by PID) next time it is started
EOF
