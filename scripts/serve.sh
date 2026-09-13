#!/usr/bin/env bash
# Copyright 2026 The road-to-52 authors. SPDX-License-Identifier: Apache-2.0
#
# Serve an exported r52 model over the OpenAI-compatible mlx_lm.server API.
#
#   scripts/serve.sh models/nano-sft-mlx                 # http://127.0.0.1:8080
#   scripts/serve.sh models/nano-sft-mlx --port 9000
#
#   curl http://127.0.0.1:8080/v1/chat/completions -H 'Content-Type: application/json' \
#     -d '{"messages":[{"role":"user","content":"What is 2+2?"}],"max_tokens":64}'
#
# The server applies the tokenizer's chat_template, which r52.export wrote from
# r52/chat_template.py -- so /v1/chat/completions sees exactly the training token layout.
set -euo pipefail

if [ $# -lt 1 ]; then sed -n '3,13p' "$0" >&2; exit 2; fi

MODEL="$1"; shift
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"
PY="${R52_PYTHON:-$REPO/.venv/bin/python}"
[ -x "$PY" ] || { echo "python not found at $PY (set R52_PYTHON)" >&2; exit 1; }
[ -d "$MODEL" ] || { echo "not a model directory: $MODEL" >&2; exit 1; }

exec nice -n 10 "$PY" -m mlx_lm server --model "$MODEL" "$@"
