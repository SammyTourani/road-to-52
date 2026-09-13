#!/usr/bin/env bash
# Copyright 2026 The road-to-52 authors.
# SPDX-License-Identifier: Apache-2.0
#
# MLX-on-CUDA experiment -- Phase 1 bench sweep + Phase 3 real run (see ../README.md §2). Phase 2
# (8xH100 NCCL data-parallel) is DOCUMENTED, not executed -- r52.train does not support
# mx.distributed today (verified: `grep -rn "distributed" r52/*.py` returns nothing). STAGE 4
# below prints the mlx.launch invocation and the code change that would be needed, and always
# prints it (not gated by --dry-run) since nothing in STAGE 4 executes regardless.
#
# WHAT THIS DOES, STEP BY STEP:
#   1. python -m r52.bench for the sizes that work today (124M, 350M) at a 70 GiB memory limit
#      and a larger micro-batch than the Mac's ladder ever reaches -- H100 has 80 GB vs the Mac's
#      16 GB unified, so both the mem cap and the micro-batch need raising, not just the former.
#      The 1B-class size is NOT YET in r52.bench's SIZES dict; this stage prints the one-line
#      addition needed and the command that would then work, rather than editing r52/bench.py
#      (outside this package's owned directory -- see ../README.md's scope).
#   2. scripts/train.sh configs/gpt2_124m_mac.yaml gpt2-124m-h100, with the memory limit and
#      micro-batch overridden via `-o train.<field>=<value>` (the real, verified mechanism --
#      r52/config.py's `apply_overrides`, exactly as r52/train.py's own module docstring example
#      uses it). Same optimizer hyperparameters, same 0.75B-token budget, same 3.28 target as the
#      Mac's live Rung 0 run -- only hardware and the two memory/batch overrides change.
#   3. scripts/eval.sh on the resulting checkpoint (val_loss + hellaswag, the "standard" suite).
#   4. Prints (does not run) the mlx.launch NCCL invocation for 8xH100 and the ~20-line
#      mx.distributed change r52.train would need.
#
# IMPORTANT -- where this script must run from: it expects to run from inside a checkout of
# road-to-52 (repo root is two directories up from this script, same convention scripts/*.sh use
# one level up). setup.sh's own clone (STAGE 4 there) is pinned to commit
# 4bd7ba570bfb13dcc4c39c3bf1cc4bcb963e4933 -- which PREDATES this package's existence in git
# history (git was broken in the session that wrote it, so it isn't committed yet). Until this
# package is committed and R52_COMMIT in setup.sh is re-pinned past that commit, a fresh
# `setup.sh` clone will NOT contain this file. Either commit this package first and re-pin, or
# copy `rungs/mlx_cuda_experiment/` onto the GPU box yourself (e.g. rsync/scp) after setup.sh's
# clone completes. Stated here rather than silently assumed away.
#
# Usage:
#   rungs/mlx_cuda_experiment/bench_and_train.sh [--dry-run] [--skip-bench] [--skip-train] [--skip-eval]
#
#   --dry-run     print every command without touching the filesystem, network, or a GPU.
#   --skip-bench  skip STAGE 1 (the throughput sweep).
#   --skip-train  skip STAGE 2 (the real 0.75B-token run).
#   --skip-eval   skip STAGE 3 (evaluating the STAGE 2 checkpoint).
#
# Environment (all optional unless noted):
#   R52_PYTHON        interpreter to use.                default: .venv/bin/python (repo-relative)
#   R52_MEM_LIMIT_GIB  MLX memory limit, GiB.              default: 70   (H100 has 80 GB; ~10 GB
#                      headroom for the CUDA context / allocator fragmentation, same margin logic
#                      as the Mac configs' 6-of-16-GB rule in docs/PLAN.md §7 rule 4)
#   R52_MICRO_BATCH    micro-batch for both bench and train. default: 32  (Mac's bench ladder tops
#                      out at 16 -- MICRO_BATCH_LADDER in r52/bench.py; passing --micro-batch
#                      explicitly bypasses that ladder entirely, r52/bench.py::run_one). UNTESTED
#                      on real H100 memory -- tune by doubling/halving like rung2_3b/launch.sh's
#                      R52_MICROBATCH_MULT does, the first time you run this for real.
#   R52_RUN_NAME       train run name.                     default: gpt2-124m-h100
#   R52_BENCH_OUT       bench sweep output base path.        default: results/mlx_cuda_bench
#   R52_EVAL_SUITE       scripts/eval.sh suite.                default: standard
#
# THIS SCRIPT USES A REAL GPU AND SPENDS MONEY WHEN RUN FOR REAL (STAGE 1 takes minutes; STAGE 2
# is expectations.md's "0.75B-token run" -- modeled at under an hour and a few dollars on 1xH100,
# see expectations.md's cost table -- but is a real training run, not a smoke test).
# See ../README.md "The rule: paid runs are the owner's decision" before running without --dry-run.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

R52_PYTHON="${R52_PYTHON:-$REPO_ROOT/.venv/bin/python}"
R52_MEM_LIMIT_GIB="${R52_MEM_LIMIT_GIB:-70}"
R52_MICRO_BATCH="${R52_MICRO_BATCH:-32}"
R52_RUN_NAME="${R52_RUN_NAME:-gpt2-124m-h100}"
R52_BENCH_OUT="${R52_BENCH_OUT:-results/mlx_cuda_bench}"
R52_EVAL_SUITE="${R52_EVAL_SUITE:-standard}"
DRY_RUN=0
SKIP_BENCH=0
SKIP_TRAIN=0
SKIP_EVAL=0

for arg in "$@"; do
  case "$arg" in
    --dry-run) DRY_RUN=1 ;;
    --skip-bench) SKIP_BENCH=1 ;;
    --skip-train) SKIP_TRAIN=1 ;;
    --skip-eval) SKIP_EVAL=1 ;;
    -h|--help)
      sed -n '3,45p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
      exit 0
      ;;
    *)
      echo "bench_and_train.sh: unknown argument '$arg'" >&2
      exit 2
      ;;
  esac
done

run() {
  local desc="$1"; shift
  echo ">> $desc"
  if [[ "$DRY_RUN" -eq 1 ]]; then
    printf '   [dry-run]'; printf ' %q' "$@"; echo
  else
    "$@"
  fi
}

echo "== road-to-52 MLX-on-CUDA experiment -- bench + train ================================="
echo "mode              : $([[ $DRY_RUN -eq 1 ]] && echo 'DRY RUN (nothing will execute)' || echo 'LIVE -- this uses the GPU and spends money')"
echo "repo root         : $REPO_ROOT"
echo "memory limit      : ${R52_MEM_LIMIT_GIB} GiB"
echo "micro-batch       : ${R52_MICRO_BATCH}  (UNTESTED on real H100 memory -- tune before trusting it)"
echo "run name          : $R52_RUN_NAME"
echo "========================================================================================="
echo

if [[ "$DRY_RUN" -ne 1 ]] && ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "bench_and_train.sh: no nvidia-smi found. This must run on the rented GPU node." >&2
  echo "                    (re-run with --dry-run to preview commands from any machine)" >&2
  exit 1
fi
[ -x "$R52_PYTHON" ] || [[ "$DRY_RUN" -eq 1 ]] || { echo "python not found at $R52_PYTHON (set R52_PYTHON)" >&2; exit 1; }

# -----------------------------------------------------------------------------------------------
# STAGE 1 -- throughput sweep on 1xH100: the sizes r52.bench supports today, at H100-scale memory
# and micro-batch. See ../expectations.md for the modeled tok/s-at-MFU this should be compared
# against, and ../README.md §2 Phase 1 for the three-way comparison (Mac numbers / expectations.md
# / nanochat-class MFU anchors).
if [[ "$SKIP_BENCH" -eq 1 ]]; then
  echo ">> STAGE 1 skipped (--skip-bench)"
else
  echo "-- STAGE 1: bench sweep (124M, 350M) ---------------------------------------------------"
  run "bench sweep: 124M, 350M @ seq 512+1024, mixed+bf16" \
    "$R52_PYTHON" -m r52.bench \
      --sizes 124M,350M --seqs 512,1024 --precisions mixed,bf16 \
      --mem-limit-gib "$R52_MEM_LIMIT_GIB" --micro-batch "$R52_MICRO_BATCH" \
      --out "$R52_BENCH_OUT"

  cat <<'EOF'

   -- 1B-class: NOT YET RUNNABLE, one dict entry away -------------------------------------
   r52/bench.py's SIZES dict (module level) only has 30M/60M/124M/350M. expectations.md derives
   a ~1B-class architecture analytically (L=24, d=1792, h=14, head_dim=128 -- ~1.015B params for
   6N, ~1.195B total). Adding it to r52/bench.py is a one-line change, outside this package's
   owned directory (rungs/mlx_cuda_experiment/), so it is shown here rather than applied:

       SIZES: dict[str, dict[str, int]] = {
           "30M":  {"n_layer": 8,  "n_embd": 512,  "n_head": 8},
           "60M":  {"n_layer": 12, "n_embd": 640,  "n_head": 10},
           "124M": {"n_layer": 12, "n_embd": 768,  "n_head": 6},
           "350M": {"n_layer": 24, "n_embd": 1024, "n_head": 8},
   +       "1B":   {"n_layer": 24, "n_embd": 1792, "n_head": 14},
       }

   Once added, the matching command is:

       python -m r52.bench --sizes 1B --seqs 1024 --precisions mixed,bf16 \
         --mem-limit-gib 70 --micro-batch 32 --out results/mlx_cuda_bench --append
   ------------------------------------------------------------------------------------------
EOF
fi
echo

# -----------------------------------------------------------------------------------------------
# STAGE 2 -- the real 0.75B-token run, same config as the Mac's live Rung 0 run. `-o` overrides
# use r52/config.py's dotted-key mechanism (verified: r52/train.py's own module docstring shows
# `-o micro_batch=4 -o train.compile=true`; `apply_overrides` resolves an unprefixed key against
# model/data/train in that order, so both forms below are equivalent -- the `train.` prefix is
# kept for clarity). 32 was chosen so tokens_per_step (131,072) divides evenly by
# micro_batch*block_size (32*1024=32,768 -> grad_accum=4); Config.grad_accum() raises otherwise.
if [[ "$SKIP_TRAIN" -eq 1 ]]; then
  echo ">> STAGE 2 skipped (--skip-train)"
else
  echo "-- STAGE 2: real run, configs/gpt2_124m_mac.yaml, 0.75B tokens, target val_loss <= 3.28 --"
  run "launch training (detached, via scripts/train.sh)" \
    scripts/train.sh configs/gpt2_124m_mac.yaml "$R52_RUN_NAME" \
      -o "train.memory_limit_gb=${R52_MEM_LIMIT_GIB}" \
      -o "train.micro_batch=${R52_MICRO_BATCH}"
  echo "   scripts/train.sh launches detached and returns immediately (prints the PID + how to"
  echo "   follow/stop it). This script does NOT block waiting for it -- check "
  echo "   runs/$R52_RUN_NAME/stdout.log yourself, or re-run this script with --skip-train once"
  echo "   training has actually finished, to run STAGE 3."
fi
echo

# -----------------------------------------------------------------------------------------------
# STAGE 3 -- eval the checkpoint. Uses scripts/eval.sh's own "standard" suite (val_loss +
# hellaswag) by default, matching Rung 0's target criteria in docs/PLAN.md §4.
if [[ "$SKIP_EVAL" -eq 1 ]]; then
  echo ">> STAGE 3 skipped (--skip-eval)"
else
  echo "-- STAGE 3: eval -------------------------------------------------------------------------"
  CKPT="runs/$R52_RUN_NAME/ckpt/best"
  if [[ "$DRY_RUN" -ne 1 && ! -d "$CKPT" ]]; then
    echo "   $CKPT does not exist yet (STAGE 2 hasn't produced a checkpoint) -- skipping the real"
    echo "   invocation; here is the command that will apply once it does:"
    echo "   R52_MEM_GIB=$R52_MEM_LIMIT_GIB scripts/eval.sh $CKPT $R52_EVAL_SUITE"
  else
    run "evaluate the checkpoint" \
      env R52_MEM_GIB="$R52_MEM_LIMIT_GIB" scripts/eval.sh "$CKPT" "$R52_EVAL_SUITE"
  fi
fi
echo

# -----------------------------------------------------------------------------------------------
# STAGE 4 -- 8xH100 NCCL data-parallel: DOCUMENTED, NOT EXECUTED. Always printed (dry-run or not)
# since nothing here runs a command. UNVERIFIED that r52.train supports mx.distributed -- it does
# not today (checked: `grep -rn "distributed" r52/*.py` -> no matches). See ../README.md §1.5/§2
# Phase 2 for the sourcing (mlx.launch / NCCL docs, mlx.nn.average_gradients signature).
cat <<'EOF'
== STAGE 4: 8xH100 NCCL data-parallel (documentation only -- r52.train does not support this yet) =====

The mlx.launch invocation itself needs no r52 change (docs/src/usage/launching_distributed.rst,
"NCCL Specifics"), and is exactly:

    # single node, 8 GPUs:
    mlx.launch -n 8 -- python -m r52.train configs/gpt2_124m_mac.yaml --run-name gpt2-124m-h100x8 \
        -o train.memory_limit_gb=70 -o train.micro_batch=32

    # two nodes, 8 GPUs each (16 processes total):
    mlx.launch --backend nccl --hosts h100-node-1,h100-node-2 -n 8 -- \
        python -m r52.train configs/gpt2_124m_mac.yaml --run-name gpt2-124m-h100x16 \
        -o train.memory_limit_gb=70 -o train.micro_batch=32

What r52/train.py does NOT do today, and would need (~20 lines, sketched -- not applied, outside
this package's owned directory):

    # 1. At Trainer construction: join the distributed group. mx.distributed.init() is a no-op
    #    singleton group when launched without mlx.launch, so this line is always safe to add.
    world = mx.distributed.init()
    rank, world_size = world.rank(), world.size()

    # 2. Shard the data cursor by rank so every GPU trains on disjoint tokens instead of each
    #    replaying the identical stream (r52/data.py's TokenStream reads shards in a fixed glob
    #    order with no rank awareness today). Simplest correct fix: stripe shard indices by rank
    #    in make_train_stream()/TokenStream -- rank r reads shards [r, r+world_size, r+2*world_size, ...].

    # 3. Replace the single-process gradient with an averaged one, using the shipped helper
    #    (python/mlx/nn/utils.py::average_gradients -- batches small tensors into few all-reduces;
    #    NOT a raw per-tensor mx.distributed.all_sum loop, which the docs' own "Tips and Tricks"
    #    section warns hurts performance):
    from mlx.nn import average_gradients
    ...
    acc, gnorm = self.opt.clip(acc)
    acc = average_gradients(acc)          # <- the actual new line; averages + all-reduces internally
    self.opt.set_step(self.step, self.total_steps)
    self.opt.update(self.model, acc)

    # 4. Only rank 0 logs/checkpoints (every rank computes identical post-all-reduce updates, so
    #    every rank writing runs/<name>/log.jsonl or ckpt/ concurrently is redundant and racy):
    if rank == 0:
        self.log(...)
        self.save(...)

Env-var launch (no mlx.launch, e.g. under a cluster scheduler -- docs/src/usage/distributed.rst
"Distributed Without mlx.launch", NCCL section): MLX_RANK, MLX_WORLD_SIZE, NCCL_HOST_IP, NCCL_PORT,
CUDA_VISIBLE_DEVICES must be set per-process.
=========================================================================================================
EOF

echo
echo "== done =================================================================================="
