#!/usr/bin/env bash
# Copyright 2026 The road-to-52 authors. SPDX-License-Identifier: Apache-2.0
#
# Chat with an exported r52 model using its chat template (r52/chat_template.py).
#
#   scripts/chat.sh models/nano-sft-mlx                  # interactive REPL (mlx_lm.chat)
#   scripts/chat.sh models/nano-sft-mlx "What is 2+2?"   # one turn (mlx_lm.generate)
#   scripts/chat.sh runs/r1/final "hi" --max-tokens 128 --temp 0.7
#
# mlx-lm 0.31.3 applies the tokenizer's chat template BY DEFAULT (the opposite flag is
# --ignore-chat-template); there is no --apply-chat-template in this version.
set -euo pipefail

if [ $# -lt 1 ]; then sed -n '3,11p' "$0" >&2; exit 2; fi

MODEL="$1"; shift
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"
PY="${R52_PYTHON:-$REPO/.venv/bin/python}"
[ -x "$PY" ] || { echo "python not found at $PY (set R52_PYTHON)" >&2; exit 1; }
[ -d "$MODEL" ] || { echo "not a model directory: $MODEL" >&2; exit 1; }
grep -q chat_template "$MODEL/tokenizer_config.json" 2>/dev/null || \
  echo "warning: $MODEL/tokenizer_config.json has no chat_template; re-export with r52.export" >&2

if [ $# -gt 0 ] && [[ "$1" != -* ]]; then
  PROMPT="$1"; shift
  exec nice -n 10 "$PY" -m mlx_lm generate --model "$MODEL" --prompt "$PROMPT" "$@"
fi
exec nice -n 10 "$PY" -m mlx_lm chat --model "$MODEL" "$@"
