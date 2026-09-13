#!/usr/bin/env bash
# Copyright 2026 The road-to-52 authors.
# SPDX-License-Identifier: Apache-2.0
#
# MLX-on-CUDA experiment -- provider-agnostic Ubuntu + NVIDIA setup. See ../README.md for what
# this verifies and why. Nothing here has been run; every command is quoted or sourced against
# MLX's own docs/source/PyPI metadata in README.md §1.
#
# WHAT THIS DOES, STEP BY STEP:
#   1. Sanity-check this is a Linux+NVIDIA box (nvidia-smi) and print driver/CUDA versions.
#   2. Install uv (if missing) and create a Python 3.12 venv.
#   3. uv pip install "mlx[cuda12]" -- the exact, documented PyPI extra (README.md §1.1). NOT
#      "mlx[cuda]" (the build brief's guess) -- that alias happens to also resolve to the same
#      cuda12 wheel today per PyPI's requires_dist, but install.rst only documents the versioned
#      form, so that's what we pin to.
#   4. Clone road-to-52 at the pinned commit (below), verify the checkout, then install the
#      project. IMPORTANT ORDERING: mlx[cuda12] must already be installed (step 3) before this,
#      because pyproject.toml declares a plain unversioned "mlx>=0.32" with no cuda extra --
#      installing the project first, alone, on Linux would satisfy that with a bare `mlx`
#      metapackage carrying NO backend at all (PyPI's mlx 0.32.2 requires_dist gates both
#      mlx-cuda-12 and mlx-cpu behind explicit extras; neither is pulled in by default). Step 3
#      first means the already-installed, already-satisfying `mlx` is left alone. STAGE 4 below
#      re-verifies mx.cuda.is_available() AFTER the project install as a safety net for this.
#   5. Download the FineWeb data shards needed for the 0.75B-token real run.
#   6. Run the pytest subset that needs no macOS/Metal.
#
# FINDING WORTH STATING UP FRONT (README.md §1.7): mlx-lm and mlx-lm-lora, which the build brief
# asked us to check for a macOS-only restriction, are BOTH plain `py3-none-any` wheels on PyPI --
# not platform-gated at all. There is no macOS-only dependency to strip out of pyproject.toml.
# STAGE 4b below re-checks this live against PyPI (the same check documented in README.md §1.7)
# rather than just trusting what this session found on 2026-09-13, since wheels can change.
#
# Usage:
#   rungs/mlx_cuda_experiment/setup.sh [--dry-run]
#
#   --dry-run   print every command that would run, and the PyPI/commit checks' results, without
#               touching the filesystem, network, or installing anything for real.
#
# Environment (all optional unless noted):
#   R52_WORKDIR           Where to clone road-to-52.                default: $PWD/road-to-52-cuda
#   R52_REPO               Git remote to clone.                      default: https://github.com/SammyTourani/road-to-52
#   R52_COMMIT              Commit to pin and check out.               default: 4bd7ba570bfb13dcc4c39c3bf1cc4bcb963e4933
#                           (verified 2026-09-13 via `gh api repos/SammyTourani/road-to-52/commits/main
#                            --jq .sha`; re-verify before a real run if much time has passed, per
#                            ../README.md checklist item 1.)
#   R52_CUDA_EXTRA          mlx PyPI extra to install.                 default: cuda12
#                           (use cuda13 only if the node's driver is >= 580 -- see README.md §1.1)
#   R52_PYTHON_VERSION       Python version for the uv venv.            default: 3.12
#   R52_TRAIN_SHARDS         FineWeb train shards to download.          default: 8
#                           (0.75B tokens / ~100M tokens-per-shard, ceil -- matches
#                            configs/gpt2_124m_mac.yaml's max_tokens, same arithmetic r52.train
#                            itself prints when the on-disk token budget falls short)
#   R52_SKIP_TESTS           Set to 1 to skip the pytest step.          default: unset (tests run)
#
# THIS SCRIPT INSTALLS PACKAGES AND DOWNLOADS DATA WHEN RUN FOR REAL, ON A REAL RENTED GPU NODE.
# See ../README.md "The rule: paid runs are the owner's decision" before running without --dry-run.
set -euo pipefail

# -----------------------------------------------------------------------------------------------
R52_WORKDIR="${R52_WORKDIR:-$PWD/road-to-52-cuda}"
R52_REPO="${R52_REPO:-https://github.com/SammyTourani/road-to-52}"
R52_COMMIT="${R52_COMMIT:-4bd7ba570bfb13dcc4c39c3bf1cc4bcb963e4933}"
R52_CUDA_EXTRA="${R52_CUDA_EXTRA:-cuda12}"
R52_PYTHON_VERSION="${R52_PYTHON_VERSION:-3.12}"
R52_TRAIN_SHARDS="${R52_TRAIN_SHARDS:-8}"
R52_SKIP_TESTS="${R52_SKIP_TESTS:-}"
DRY_RUN=0

for arg in "$@"; do
  case "$arg" in
    --dry-run) DRY_RUN=1 ;;
    -h|--help)
      sed -n '3,50p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
      exit 0
      ;;
    *)
      echo "setup.sh: unknown argument '$arg' (only --dry-run / --help are supported)" >&2
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

echo "== road-to-52 MLX-on-CUDA experiment -- setup ========================================="
echo "mode              : $([[ $DRY_RUN -eq 1 ]] && echo 'DRY RUN (nothing will execute)' || echo 'LIVE -- this installs packages and downloads data')"
echo "workdir           : $R52_WORKDIR"
echo "repo              : $R52_REPO"
echo "pinned commit     : $R52_COMMIT"
echo "mlx extra         : $R52_CUDA_EXTRA  (mlx[$R52_CUDA_EXTRA])"
echo "python            : $R52_PYTHON_VERSION"
echo "train shards      : $R52_TRAIN_SHARDS"
echo "========================================================================================="
echo

if [[ "$DRY_RUN" -ne 1 ]] && ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "setup.sh: no nvidia-smi found. This must run on a Linux box with an NVIDIA GPU." >&2
  echo "          (re-run with --dry-run to preview commands from any machine)" >&2
  exit 1
fi

# -----------------------------------------------------------------------------------------------
# STAGE 1 -- confirm the box: driver, CUDA toolkit presence, GPU architecture.
if [[ "$DRY_RUN" -eq 1 ]]; then
  echo ">> nvidia-smi --query-gpu=name,compute_cap,driver_version --format=csv"
else
  echo "-- GPU / driver ----------------------------------------------------------------------"
  nvidia-smi --query-gpu=name,compute_cap,driver_version --format=csv
  echo "(README.md §1.1: mlx[$R52_CUDA_EXTRA] requires SM >= 7.5, driver >= 550.54.14 for cuda12"
  echo " / >= 580 for cuda13, CUDA toolkit >= 12.0. An H100 is SM 9.0 -- comfortably above.)"
fi

# CUDA toolkit + cuDNN headers, exactly as MLX's own install.rst documents for Ubuntu (the same
# apt sequence its "Linux / CUDA" build-from-source section uses; the PyPI wheel path still lists
# "CUDA toolkit >= 12.0" as a system requirement, not just a build-from-source one).
run "add the NVIDIA CUDA apt repo" \
  bash -c 'wget -q https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb -O /tmp/cuda-keyring.deb && sudo dpkg -i /tmp/cuda-keyring.deb'
run "apt update" sudo apt-get update -y
run "install CUDA toolkit + cuDNN + BLAS/LAPACK headers" \
  sudo apt-get install -y cuda-toolkit-12-9 libblas-dev liblapack-dev liblapacke-dev libcudnn9-dev-cuda-12

# -----------------------------------------------------------------------------------------------
# STAGE 2 -- uv + a Python 3.12 venv, mirroring this repo's own CI (.github/workflows/ci.yml).
if ! command -v uv >/dev/null 2>&1; then
  run "install uv" bash -c 'curl -LsSf https://astral.sh/uv/install.sh | sh'
  export PATH="$HOME/.local/bin:$PATH"
fi
run "create the venv" uv venv --python "$R52_PYTHON_VERSION" "$R52_WORKDIR/.venv"
VENV="$R52_WORKDIR/.venv"
PY="$VENV/bin/python"

# -----------------------------------------------------------------------------------------------
# STAGE 3 -- mlx[cuda12] FIRST, before the project's own plain "mlx>=0.32" requirement is ever
# resolved (see the header comment's "IMPORTANT ORDERING" note).
run "install mlx[$R52_CUDA_EXTRA]" \
  env VIRTUAL_ENV="$VENV" uv pip install "mlx[$R52_CUDA_EXTRA]"

if [[ "$DRY_RUN" -eq 1 ]]; then
  echo ">> $PY -c \"import mlx.core as mx; assert mx.cuda.is_available(), 'CUDA backend not available'; print('mlx', mx.__version__ if hasattr(mx, \\\"__version__\\\") else '(no __version__ attr)', '-- CUDA available:', mx.cuda.is_available())\""
else
  echo "-- verifying the CUDA backend is actually live (not just installed) ------------------"
  "$PY" -c "
import mlx.core as mx
ok = mx.cuda.is_available()
print('mx.cuda.is_available():', ok)
if not ok:
    raise SystemExit('mlx[$R52_CUDA_EXTRA] installed but CUDA backend reports unavailable -- check driver/toolkit (STAGE 1 above)')
"
fi

# -----------------------------------------------------------------------------------------------
# STAGE 4 -- clone and pin road-to-52, then install it. No macOS-only dependency was found to
# strip out (README.md §1.7): mlx-lm and mlx-lm-lora are both plain py3-none-any wheels, and every
# other pyproject.toml dependency (numpy, pyyaml, rich, tiktoken, safetensors, huggingface_hub,
# datasets, lm_eval, reasoning-gym) is itself cross-platform. So this installs the project as-is.
run "clone road-to-52" git clone "$R52_REPO" "$R52_WORKDIR/repo"

if [[ "$DRY_RUN" -eq 1 ]]; then
  echo ">> cd $R52_WORKDIR/repo && git checkout $R52_COMMIT"
else
  cd "$R52_WORKDIR/repo"
  git checkout "$R52_COMMIT"
  actual_sha="$(git rev-parse HEAD)"
  if [[ "$actual_sha" != "$R52_COMMIT" ]]; then
    echo "setup.sh: checked-out HEAD ($actual_sha) does not match pinned commit ($R52_COMMIT)" >&2
    exit 1
  fi
  echo ">> pinned at $actual_sha"
fi

run "install road-to-52 (dev extras included)" \
  env VIRTUAL_ENV="$VENV" uv pip install -e "$R52_WORKDIR/repo[dev]"

# STAGE 4b -- re-verify the mlx-lm / mlx-lm-lora "not macOS-only" finding live against PyPI,
# rather than trusting what this session found on 2026-09-13 (wheels can change). Exact check
# documented in README.md §1.7 / requested by the build brief. Aborts (set -e, via python's
# SystemExit) if either package ever ships a platform-restricted wheel, since that would silently
# invalidate this whole package's premise.
if [[ "$DRY_RUN" -eq 1 ]]; then
  echo ">> curl -s https://pypi.org/pypi/mlx-lm/json | python3 -c '...verify py3-none-any...'"
  echo ">> curl -s https://pypi.org/pypi/mlx-lm-lora/json | python3 -c '...verify py3-none-any...'"
else
  echo "-- re-checking mlx-lm / mlx-lm-lora are still platform-unrestricted on PyPI -----------"
  for pkg in mlx-lm mlx-lm-lora; do
    curl -s "https://pypi.org/pypi/$pkg/json" | "$PY" -c "
import json, sys
d = json.load(sys.stdin)
version = d['info']['version']
files = [u['filename'] for u in d['urls']]
bad = [f for f in files if not f.endswith('-py3-none-any.whl') and f.endswith('.whl')]
print(f'$pkg {version}:', files)
if bad:
    raise SystemExit(f'$pkg now ships a platform-restricted wheel ({bad}) -- re-check README.md §1.7 before continuing')
"
  done
fi

# -----------------------------------------------------------------------------------------------
# STAGE 5 -- data: the FineWeb-10B-GPT2 shards the 0.75B-token real run needs (Phase 3, see
# ../bench_and_train.sh). scripts/prepare_data.py fetches from kjj0/fineweb10B-gpt2 (public HF
# dataset, no token required) and always includes the fixed validation shard.
run "download FineWeb data shards" \
  env VIRTUAL_ENV="$VENV" "$PY" "$R52_WORKDIR/repo/scripts/prepare_data.py" --train-shards "$R52_TRAIN_SHARDS"

# -----------------------------------------------------------------------------------------------
# STAGE 6 -- the test subset that needs no macOS/Metal. This repo's own CI
# (.github/workflows/ci.yml) runs `python -m pytest -q -x --timeout=600 -p no:cacheprovider` with
# HF_HUB_OFFLINE=1 on a macOS runner; that SAME invocation is what we run here. No test file in
# tests/ was found to hard-require a macOS-only API: the two files that shell out to macOS tools
# (r52/bench.py's `sysctl`/`system_profiler`, r52/eval/report.py's `sysctl`) both wrap those calls
# in try/except and degrade to "unknown"/None rather than raise (verified by reading both files),
# and every test that needs an optional package not on this list (mlx_lm, tiktoken,
# reasoning_gym, tokenizers, transformers) already self-skips via `pytest.importorskip(...)`
# rather than hard-failing collection. So "the non-macOS subset" is: the whole suite, run as-is.
if [[ "${R52_SKIP_TESTS}" == "1" ]]; then
  echo ">> skipping tests (R52_SKIP_TESTS=1)"
else
  run "lint" env VIRTUAL_ENV="$VENV" "$VENV/bin/ruff" check "$R52_WORKDIR/repo/r52" "$R52_WORKDIR/repo/tests"
  if [[ "$DRY_RUN" -eq 1 ]]; then
    echo ">> HF_HUB_OFFLINE=1 $PY -m pytest -q -x --timeout=600 -p no:cacheprovider"
  else
    ( cd "$R52_WORKDIR/repo" && HF_HUB_OFFLINE=1 "$PY" -m pytest -q -x --timeout=600 -p no:cacheprovider )
  fi
fi

echo
echo "== done ================================================================================="
echo "repo    : $R52_WORKDIR/repo  (pinned at $R52_COMMIT)"
echo "venv    : $VENV"
echo "next    : rungs/mlx_cuda_experiment/bench_and_train.sh --dry-run   (then for real)"
echo "========================================================================================="
