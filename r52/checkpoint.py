# Copyright 2026 The road-to-52 authors.
# SPDX-License-Identifier: Apache-2.0
"""Checkpoint save / load: weights + optimizer state + data cursor + config + step.

Layout of ``runs/<name>/ckpt/``::

    step_00000500/model.safetensors    flattened parameter tree
    step_00000500/optim.safetensors    flattened MultiOptimizer state tree
    step_00000500/meta.json            step, tokens, cursor, seed, val loss, full config
    best/                              same three files, the lowest validation loss so far

Only the last ``keep_last`` step directories plus ``best/`` are retained.

Note on RNG: MLX's global random state is opaque (``mx.random.state`` exposes no
key), so the checkpoint stores the *seed* and the step and resume re-seeds with
``seed + step``.  Nothing in the training step is stochastic (no dropout), so this
is exact, not approximate -- see ``docs/DEVIATIONS.md``.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import mlx.core as mx
from mlx.utils import tree_flatten, tree_unflatten

from .config import Config, from_dict

__all__ = ["latest_checkpoint", "list_checkpoints", "load_checkpoint", "prune", "save_checkpoint"]

_MODEL = "model.safetensors"
_OPTIM = "optim.safetensors"
_META = "meta.json"


def _flat(tree: dict) -> dict[str, mx.array]:
    return {k: v for k, v in tree_flatten(tree) if isinstance(v, mx.array)}


def save_checkpoint(
    path: str | Path,
    model,
    optimizer=None,
    *,
    step: int = 0,
    tokens: int = 0,
    cursor: dict[str, int] | None = None,
    config: Config | None = None,
    extra: dict[str, Any] | None = None,
) -> Path:
    """Write one checkpoint directory. Returns the directory path."""
    out = Path(path)
    tmp = out.with_name("." + out.name + ".tmp")  # leading dot: never matches the step_* glob
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)

    mx.save_safetensors(str(tmp / _MODEL), _flat(model.parameters()))
    if optimizer is not None:
        mx.save_safetensors(str(tmp / _OPTIM), _flat(optimizer.state))

    meta: dict[str, Any] = {
        "step": int(step),
        "tokens": int(tokens),
        "cursor": cursor or {},
        "config": config.to_dict() if config is not None else {},
    }
    meta.update(extra or {})
    (tmp / _META).write_text(json.dumps(meta, indent=2))

    if out.exists():
        shutil.rmtree(out)
    tmp.rename(out)
    return out


def load_checkpoint(path: str | Path, model=None, optimizer=None, strict: bool = True) -> dict[str, Any]:
    """Restore weights/optimizer state in place and return the checkpoint metadata."""
    d = Path(path)
    meta = json.loads((d / _META).read_text())
    if model is not None:
        weights = mx.load(str(d / _MODEL))
        model.update(tree_unflatten(list(weights.items())))
        if strict:
            have = {k for k, _ in tree_flatten(model.parameters())}
            missing = have - set(weights)
            if missing:
                raise ValueError(f"checkpoint {d} is missing parameters: {sorted(missing)[:8]}")
    if optimizer is not None and (d / _OPTIM).exists():
        state = mx.load(str(d / _OPTIM))
        try:
            optimizer.state = tree_unflatten(list(state.items()))
        except ValueError as exc:
            raise ValueError(
                f"optimizer state in {d} does not match this optimizer ({exc}). The number of "
                "parameter groups is derived from the model config -- resuming with a different "
                "`use_value_embeds` / `use_unet_skips` / `mlp` changes it."
            ) from exc
    return meta


def checkpoint_config(path: str | Path) -> Config:
    """Read just the :class:`r52.config.Config` stored in a checkpoint."""
    meta = json.loads((Path(path) / _META).read_text())
    return from_dict(meta["config"])


def list_checkpoints(ckpt_dir: str | Path) -> list[Path]:
    """Sorted ``step_*`` directories inside ``ckpt_dir`` (oldest first)."""
    d = Path(ckpt_dir)
    if not d.exists():
        return []
    return sorted((p for p in d.glob("step_*") if (p / _META).exists()), key=lambda p: p.name)


def latest_checkpoint(ckpt_dir: str | Path) -> Path | None:
    """Highest-numbered ``step_*`` directory, or ``None``."""
    ckpts = list_checkpoints(ckpt_dir)
    return ckpts[-1] if ckpts else None


def prune(ckpt_dir: str | Path, keep_last: int = 2) -> None:
    """Delete all but the newest ``keep_last`` step directories (``best/`` is never touched)."""
    ckpts = list_checkpoints(ckpt_dir)
    for p in ckpts[: max(0, len(ckpts) - keep_last)]:
        shutil.rmtree(p, ignore_errors=True)
