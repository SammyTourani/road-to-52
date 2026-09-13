#!/usr/bin/env bash
# Copyright 2026 The road-to-52 authors. SPDX-License-Identifier: Apache-2.0
#
# Run the MLX pretraining throughput sweep and write
#   results/mlx_pretrain_bench.json
#   results/mlx_pretrain_bench.md
#
#   scripts/bench.sh                       # full sweep: 4 sizes x 2 seqs x 3 precisions
#   scripts/bench.sh --sizes 30M,124M --precisions mixed --seqs 1024
#   scripts/bench.sh --seconds 10          # quick pass
#
# Every row runs in its own subprocess under a 6 GiB MLX memory limit; rows that do not fit
# are recorded as "skipped" with the reason (fp32 and 350M are the ones that usually do).
#
# The full 24-row sweep takes roughly 1-2 hours: each row first walks a micro-batch ladder
# downward, paying two warm-up optimizer steps per candidate, before its 25 s measurement.
# Run it in chunks with --append if you want the machine back in between:
#
#   scripts/bench.sh --sizes 30M,60M  --precisions mixed,bf16 --append
#   scripts/bench.sh --sizes 124M     --precisions mixed,bf16 --append
#   scripts/bench.sh --sizes 350M     --precisions mixed      --append
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"
PY="${R52_PYTHON:-$REPO/.venv/bin/python}"
[ -x "$PY" ] || { echo "python not found at $PY (set R52_PYTHON)" >&2; exit 1; }

mkdir -p results
exec nice -n 10 "$PY" -m r52.bench --out results/mlx_pretrain_bench "$@"
