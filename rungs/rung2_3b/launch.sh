#!/usr/bin/env bash
# Copyright 2026 The road-to-52 authors. SPDX-License-Identifier: Apache-2.0
#
# Rung 2 -- 3B-class Llama-3-style dense model, from scratch, on a rented 8xH100 node via
# torchtitan. See README.md for the full writeup and torchtitan_3b.toml for the field-by-field
# spec this script implements. eval_plan.md and moe_variant.md cover evaluation and the 30B-A3B
# MoE alternative (which needs no custom recipe -- see moe_variant.md).
#
# WHAT THIS DOES, STEP BY STEP:
#   1. git clone pytorch/torchtitan and check out the pinned commit (below).
#   2. pip install its requirements (and, if --float8, the torchao nightly it needs).
#   3. Download the Llama 3.1 tokenizer via torchtitan's own scripts/download_hf_assets.py
#      (requires HF_TOKEN and prior access to the gated meta-llama/Llama-3.1-8B repo -- see
#      README.md "Prerequisites").
#   4. Generate torchtitan_recipes/road_to_52.py inside the clone -- a small Python "full
#      configuration" function, in the shape torchtitan's own config README documents, that
#      wraps the shipped `llama3_configs["3B"]` model at our seq_len/token budget. torchtitan
#      has no TOML job-config loader any more (see torchtitan_3b.toml's header comment for the
#      verification); this is the mechanical equivalent of "load the toml", done the way
#      upstream now expects.
#   5. Run it: `NGPU=8 MODULE=torchtitan_recipes.road_to_52 CONFIG=llama3_3b_road_to_52 ./run_train.sh`
#
# Usage:
#   rungs/rung2_3b/launch.sh [--dry-run] [--float8]
#
#   --dry-run   print every command and the generated recipe file, without touching the
#               filesystem, network, or a GPU. Always run this first.
#   --float8    generate the recipe with torchtitan's Float8LinearConverter enabled on the
#               dense Linear layers (H100-native fp8 GEMMs; see README.md "Precision"). Default
#               is plain bf16, matching torchtitan_3b.toml's [training].float8 = "optional".
#
# Environment (all optional unless noted):
#   R52_WORKDIR             Where to clone torchtitan.              default: $PWD/torchtitan-rung2
#   R52_TORCHTITAN_REPO     Git remote to clone.                     default: https://github.com/pytorch/torchtitan
#   R52_TORCHTITAN_COMMIT   Commit to pin and check out.             default: 150c4f73a035e8778f37d9f603a32534841f2a60
#                           (verified 2026-09-13 via
#                            `gh api repos/pytorch/torchtitan/commits/main --jq .sha`; re-verify
#                            before a real launch if much time has passed.)
#   HF_TOKEN                REQUIRED for a real (non---dry-run) launch. A Hugging Face token with
#                           approved access to the gated meta-llama/Llama-3.1-8B repo (we reuse
#                           its tokenizer; see README.md "Prerequisites" for why and the
#                           ungated-alternative caveat).
#   NGPU                    GPUs on this node.                       default: 8 (the ladder's Rung 2 row)
#   R52_SEQ_LEN              Training sequence length.               default: 4096
#   R52_TOKENS_TARGET        Token budget.                            default: 60000000000 (60B, ladder Rung 2)
#   R52_MICROBATCH_MULT      num_tokens_per_microbatch_per_dp_rank = this * R52_SEQ_LEN.
#                           default: 16 -- UNTESTED starting point, see torchtitan_3b.toml. Tune
#                           by doubling/halving to fit the node's actual memory before a real run.
#   R52_LR                   AdamW learning rate.                     default: 4e-4 -- UNVERIFIED, see README "LR"
#   R52_WARMUP_STEPS         LR warmup steps.                         default: 600
#   R52_CKPT_INTERVAL        Checkpoint every N steps.                default: 500
#
# THIS SCRIPT SPENDS MONEY WHEN RUN FOR REAL, ON A REAL RENTED 8xH100 NODE.
# See ../README.md "The rule: paid runs are the owner's decision" before running without --dry-run.
set -euo pipefail

# -----------------------------------------------------------------------------------------------
# Config (env-overridable; see the usage comment above). All arithmetic below is integer bash
# arithmetic -- every input is an integer, so this needs no external calculator.
R52_WORKDIR="${R52_WORKDIR:-$PWD/torchtitan-rung2}"
R52_TORCHTITAN_REPO="${R52_TORCHTITAN_REPO:-https://github.com/pytorch/torchtitan}"
R52_TORCHTITAN_COMMIT="${R52_TORCHTITAN_COMMIT:-150c4f73a035e8778f37d9f603a32534841f2a60}"
NGPU="${NGPU:-8}"
R52_SEQ_LEN="${R52_SEQ_LEN:-4096}"
R52_TOKENS_TARGET="${R52_TOKENS_TARGET:-60000000000}"
R52_MICROBATCH_MULT="${R52_MICROBATCH_MULT:-16}"
R52_LR="${R52_LR:-4e-4}"
R52_WARMUP_STEPS="${R52_WARMUP_STEPS:-600}"
R52_CKPT_INTERVAL="${R52_CKPT_INTERVAL:-500}"
DRY_RUN=0
FLOAT8=0

for arg in "$@"; do
  case "$arg" in
    --dry-run) DRY_RUN=1 ;;
    --float8) FLOAT8=1 ;;
    -h|--help)
      sed -n '3,45p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
      exit 0
      ;;
    *)
      echo "launch.sh: unknown argument '$arg' (only --dry-run / --float8 / --help are supported)" >&2
      exit 2
      ;;
  esac
done

# Derived values -- integer arithmetic, matches torchtitan_3b.toml's [training] block exactly.
MICROBATCH_TOKENS=$(( R52_MICROBATCH_MULT * R52_SEQ_LEN ))
GLOBAL_BATCH_TOKENS=$(( MICROBATCH_TOKENS * NGPU ))
STEPS=$(( (R52_TOKENS_TARGET + GLOBAL_BATCH_TOKENS - 1) / GLOBAL_BATCH_TOKENS ))  # ceil division

run() {
  local desc="$1"; shift
  echo ">> $desc"
  if [[ "$DRY_RUN" -eq 1 ]]; then
    printf '   [dry-run]'; printf ' %q' "$@"; echo
  else
    "$@"
  fi
}

echo "== road-to-52 Rung 2 -- 3B dense Llama-3-style, torchtitan, on ${NGPU}xH100 ==========="
echo "mode              : $([[ $DRY_RUN -eq 1 ]] && echo 'DRY RUN (nothing will execute)' || echo 'LIVE -- this spends money')"
echo "repo              : $R52_TORCHTITAN_REPO"
echo "pinned commit     : $R52_TORCHTITAN_COMMIT"
echo "workdir           : $R52_WORKDIR"
echo "seq_len           : $R52_SEQ_LEN"
echo "tokens target     : $R52_TOKENS_TARGET"
echo "microbatch tokens/rank : $MICROBATCH_TOKENS  (= $R52_MICROBATCH_MULT x $R52_SEQ_LEN, UNTESTED -- tune to fit memory)"
echo "global batch tokens/step : $GLOBAL_BATCH_TOKENS  (= microbatch x $NGPU GPUs, grad_accum=1)"
echo "steps (ceil to reach target) : $STEPS"
echo "lr                : $R52_LR   (UNVERIFIED against a published 3B recipe -- see README.md)"
echo "precision         : $([[ $FLOAT8 -eq 1 ]] && echo 'bf16 + Float8LinearConverter (--float8)' || echo 'bf16 (pass --float8 for fp8 GEMMs)')"
echo "======================================================================================="
echo

if [[ "$DRY_RUN" -ne 1 ]] && ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "launch.sh: no nvidia-smi found. This must run on the rented 8xH100 node, not locally." >&2
  echo "           (re-run with --dry-run to preview commands from any machine)" >&2
  exit 1
fi
if [[ "$DRY_RUN" -ne 1 ]] && [[ -z "${HF_TOKEN:-}" ]]; then
  echo "launch.sh: HF_TOKEN is not set. A Hugging Face token with approved access to the gated" >&2
  echo "           meta-llama/Llama-3.1-8B repo is required to download the tokenizer -- see" >&2
  echo "           README.md \"Prerequisites\". Set HF_TOKEN and re-run, or use --dry-run to preview." >&2
  exit 1
fi

# -----------------------------------------------------------------------------------------------
# STAGE 1 -- clone and pin.
run "clone torchtitan" git clone "$R52_TORCHTITAN_REPO" "$R52_WORKDIR"

if [[ "$DRY_RUN" -eq 1 ]]; then
  echo ">> cd $R52_WORKDIR && git checkout $R52_TORCHTITAN_COMMIT"
else
  cd "$R52_WORKDIR"
  git checkout "$R52_TORCHTITAN_COMMIT"
  actual_sha="$(git rev-parse HEAD)"
  if [[ "$actual_sha" != "$R52_TORCHTITAN_COMMIT" ]]; then
    echo "launch.sh: checked-out HEAD ($actual_sha) does not match pinned commit ($R52_TORCHTITAN_COMMIT)" >&2
    exit 1
  fi
  echo ">> pinned at $actual_sha"
fi

# -----------------------------------------------------------------------------------------------
# STAGE 2 -- install. Quoted verbatim from torchtitan's README.md ("From source"):
#   "git clone https://github.com/pytorch/torchtitan
#    cd torchtitan
#    pip install -r requirements.txt"
# and, for float8/MXFP8/NVFP4 (also verbatim, README "torchao is not installed by the command
# above... Install a nightly matching your accelerator build when you need one"):
#   "USE_CPP=0 python -m pip install --pre --upgrade torchao --index-url https://download.pytorch.org/whl/nightly/cu130"
run "install torchtitan's requirements" pip install -r requirements.txt
if [[ "$FLOAT8" -eq 1 ]]; then
  run "install torchao nightly (needed for --float8)" \
    bash -c 'USE_CPP=0 python -m pip install --pre --upgrade torchao --index-url https://download.pytorch.org/whl/nightly/cu130'
fi

# -----------------------------------------------------------------------------------------------
# STAGE 2b -- tokenizer. Quoted verbatim from torchtitan's README.md ("Downloading a tokenizer"):
#   "python scripts/download_hf_assets.py --repo_id meta-llama/Llama-3.1-8B --assets tokenizer --hf_token=..."
# We reuse the Llama 3.1 8B tokenizer (not a 3B-specific one -- Llama-3.2-3B ships the identical
# 128256-vocab tokenizer, and torchtitan's shipped llama3_8b()/llama3_70b()/llama3_405b() jobs all
# point hf_assets_path at a Llama-3.1 checkout the same way). This repo is GATED on Hugging Face;
# see README.md "Prerequisites".
if [[ "$DRY_RUN" -eq 1 ]]; then
  echo ">> python scripts/download_hf_assets.py --repo_id meta-llama/Llama-3.1-8B --assets tokenizer --hf_token=\$HF_TOKEN"
else
  run "download the Llama 3.1 tokenizer" \
    python scripts/download_hf_assets.py --repo_id meta-llama/Llama-3.1-8B --assets tokenizer --hf_token="$HF_TOKEN"
fi

# -----------------------------------------------------------------------------------------------
# STAGE 3 -- generate the recipe. torchtitan's config README (torchtitan/config/README.md) at
# the pinned commit documents exactly this pattern: "add a function to torchtitan_recipes, in a
# module named for the model, and name it on the command line" -- its own worked example is
# `torchtitan_recipes/llama3.py`, `def llama3_8b_fsdp8_tp2_h200() -> Trainer.Config: ...`. The
# function body below mirrors torchtitan/models/llama3/config_registry.py::llama3_8b() field for
# field, with the "8B" flavor swapped for the shipped "3B" flavor and our seq_len/steps/lr/etc.
RECIPE_DIR="torchtitan_recipes"
RECIPE_FILE="$RECIPE_DIR/road_to_52.py"

if [[ "$DRY_RUN" -eq 1 ]]; then
  echo ">> mkdir -p $RECIPE_DIR && cat > $RECIPE_FILE <<PYEOF ... PYEOF   (recipe shown below)"
  echo
else
  mkdir -p "$RECIPE_DIR"
  touch "$RECIPE_DIR/__init__.py"
fi

if [[ "$FLOAT8" -eq 1 ]]; then
  MODEL_SPEC_BLOCK="model_spec = model_registry(
        \"3B\",
        seq_len=seq_len,
        converters=[
            Float8LinearConverter.Config(model_compile_enabled=True),
        ],
    )"
  EXTRA_IMPORTS="
from torchtitan.config import CompileConfig
from torchtitan.config.transform import Float8LinearConverter"
  COMPILE_FIELD="        compile=CompileConfig(enable=True, components=[\"model\"]),
"
else
  MODEL_SPEC_BLOCK="model_spec = model_registry(\"3B\", seq_len=seq_len)"
  EXTRA_IMPORTS=""
  COMPILE_FIELD=""
fi

RECIPE_CONTENT="\"\"\"road-to-52 Rung 2 -- 3B dense Llama-3-style torchtitan recipe.

Generated by rungs/rung2_3b/launch.sh from rungs/rung2_3b/torchtitan_3b.toml. Mirrors the shape
of torchtitan's own torchtitan/models/llama3/config_registry.py::llama3_8b(), sized to the
shipped \"3B\" flavor (torchtitan/models/llama3/__init__.py::_3b(), == Meta's Llama-3.2-3B
architecture) instead of \"8B\".

DATASET IS A PLACEHOLDER (DATASETS[\"c4\"]). Swap for the real 100B mix before a real run --
see rungs/rung2_3b/data_plan.md.
\"\"\"

from torchtitan.components.checkpointer import CheckpointManager
from torchtitan.components.data import ConcatThenSplitPackingConfig, GrainDataLoader
from torchtitan.components.loss import ChunkedLossWrapper, CrossEntropyLoss
from torchtitan.components.optimizer import default_adamw, LRSchedulersContainer
from torchtitan.config import ParallelismConfig, TrainingConfig
from torchtitan.distributed.activation_checkpoint import SelectiveAC
from torchtitan.hf_datasets.text_datasets import DATASETS
from torchtitan.models.common.config_utils import decoder_vocab_size
from torchtitan.models.llama3 import model_registry
from torchtitan.observability.metrics import MetricsProcessor
from torchtitan.observability.profiler import Profiler
from torchtitan.trainer import Trainer${EXTRA_IMPORTS}


def llama3_3b_road_to_52(seq_len: int | None = ${R52_SEQ_LEN}) -> Trainer.Config:
    ${MODEL_SPEC_BLOCK}
    return Trainer.Config(
        loss=ChunkedLossWrapper.Config(
            loss_fn=CrossEntropyLoss.Config(
                global_vocab_size=decoder_vocab_size(model_spec),
            ),
        ),
        hf_assets_path=\"./assets/hf/Llama-3.1-8B\",
        profiler=Profiler.Config(enable_profiling=True, profile_freq=100),
        metrics=MetricsProcessor.Config(enable_tensorboard=True),
        model_spec=model_spec,
        optimizer=default_adamw(lr=${R52_LR}),
        lr_scheduler=LRSchedulersContainer.Config(warmup_steps=${R52_WARMUP_STEPS}),
        training=TrainingConfig(
            num_tokens_per_microbatch_per_dp_rank=${R52_MICROBATCH_MULT} * model_spec.max_context_length,
            max_context_length=model_spec.max_context_length,
            steps=${STEPS},
        ),
        dataloader=GrainDataLoader.Config(
            # PLACEHOLDER -- see rungs/rung2_3b/data_plan.md. Swap before a real run.
            dataset=ConcatThenSplitPackingConfig(dataset=DATASETS[\"c4\"]),
        ),
        checkpoint=CheckpointManager.Config(interval=${R52_CKPT_INTERVAL}),
        activation_checkpoint=SelectiveAC.Config(),
${COMPILE_FIELD}        parallelism=ParallelismConfig(
            data_parallel_shard_degree=-1,
            tensor_parallel_degree=1,
            pipeline_parallel_degree=1,
            context_parallel_degree=1,
        ),
    )
"

echo "---- generated $RECIPE_FILE ----------------------------------------------------------"
echo "$RECIPE_CONTENT"
echo "----------------------------------------------------------------------------------------"
if [[ "$DRY_RUN" -ne 1 ]]; then
  printf '%s' "$RECIPE_CONTENT" > "$RECIPE_FILE"
  echo ">> wrote $RECIPE_FILE"
fi

# -----------------------------------------------------------------------------------------------
# STAGE 4 -- launch. Matches the README's own documented pattern exactly (substituting our
# module/config for its worked example):
#   "MODULE=llama3 CONFIG=llama3_8b ./run_train.sh"
if [[ "$DRY_RUN" -eq 1 ]]; then
  echo ">> NGPU=$NGPU MODULE=torchtitan_recipes.road_to_52 CONFIG=llama3_3b_road_to_52 ./run_train.sh"
else
  run "launch training" env NGPU="$NGPU" MODULE=torchtitan_recipes.road_to_52 CONFIG=llama3_3b_road_to_52 ./run_train.sh
fi

# -----------------------------------------------------------------------------------------------
echo
echo "== done (or, in --dry-run mode, this is everything that would have run) ============"
echo "Before spending real money, validate the config for free with torchtitan's own fake"
echo "backend (from the cloned repo root, per run_train.sh's own comment: 'dry-run validation"
echo "without GPU execution'):"
echo "  NGPU=$NGPU COMM_MODE=fake_backend MODULE=torchtitan_recipes.road_to_52 CONFIG=llama3_3b_road_to_52 ./run_train.sh"
echo
echo "Checkpoints/logs land in the cloned repo per torchtitan's own CheckpointManager /"
echo "TensorBoard config. Copy the command, commit (both torchtitan's and this repo's), actual"
echo "wall-clock, and actual \$ billed into docs/RESULTS.md -- see rungs/README.md's checklist."
echo "Then follow eval_plan.md. Remember to shut the node down afterwards (checklist item 7)."
