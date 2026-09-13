# Copyright 2026 The road-to-52 authors.
# SPDX-License-Identifier: Apache-2.0
"""Midtraining: continue pretraining a base checkpoint on the instruction-seeded mixture.

    python -m r52.posttrain.midtrain configs/posttrain/midtrain_nano.yaml \
        --init-from runs/a1/ckpt/best --run-name m1
    python -m r52.posttrain.midtrain configs/posttrain/midtrain_tiny.yaml \
        --init-from runs/tiny200/ckpt/best --run-name mt --max-steps 20

``docs/PLAN.md`` §3.1 item 4: a short (1-2 % of pretraining tokens), LR-decaying,
high-quality stage that seeds instruction-following and thinking data into the **base**
model.  The data is built by ``scripts/prepare_midtrain_data.py``; this module is only the
*launcher*, because midtraining is ordinary next-token pretraining on a different corpus.

It therefore runs the stock :class:`r52.train.Trainer` -- same loop, same optimizer, same
checkpoints, same ``log.jsonl`` schema -- with one difference:

**Weights-only initialisation, not ``--resume``.**  ``Trainer._resume`` restores the step
counter, the optimizer moments and the data cursor, which is what you want when a run was
interrupted and exactly what you do *not* want here: the WSD schedule is laid out over the
midtrain config's ``total_steps``, so arriving at step 5,722 out of 40 would put the run at
its final learning rate on step one, and the restored cursor would point into the FineWeb
shards rather than the midtrain ones.  ``--init-from`` loads only
``model.safetensors``, leaving step 0, fresh optimizer state and a fresh cursor -- i.e. a
genuinely fresh, short schedule with its own cooldown.  Use ``--resume`` to continue an
*interrupted midtrain run* in the normal way.

Units: tokens are counted; ``cooldown_frac`` is a fraction of the midtrain run.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import mlx.core as mx

from ..checkpoint import load_checkpoint
from ..config import Config, apply_overrides, load_config
from ..train import Trainer

__all__ = ["build_trainer", "main"]


def build_trainer(
    cfg: Config,
    run_name: str,
    init_from: str | None = None,
    resume: str | None = None,
    compile_step: bool | None = None,
) -> Trainer:
    """Construct a :class:`r52.train.Trainer` for midtraining.

    ``init_from`` loads the base checkpoint's **weights only**; ``resume`` is passed
    through to the trainer unchanged (for continuing an interrupted midtrain run).  Giving
    both is an error, because ``resume`` would overwrite the freshly-loaded weights.
    """
    if init_from and resume:
        raise ValueError("--init-from and --resume are mutually exclusive")
    trainer = Trainer(cfg, run_name, resume=resume, compile_step=compile_step)
    if init_from:
        meta = load_checkpoint(init_from, trainer.model)
        mx.eval(trainer.model.parameters())
        base_step = int(meta.get("step", 0))
        base_tokens = int(meta.get("tokens", 0))
        trainer.log({
            "event": "init_from",
            "path": str(init_from),
            "base_step": base_step,
            "base_tokens": base_tokens,
            "base_val_loss": meta.get("val_loss"),
            "note": "weights only; step/optimizer/cursor reset for a fresh WSD schedule",
        })
        print(
            f"[r52.midtrain] initialised from {init_from} "
            f"(base step {base_step:,}, {base_tokens:,} pretraining tokens); "
            f"optimizer state, step counter and data cursor reset",
            flush=True,
        )
    return trainer


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="r52.posttrain.midtrain", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("config", help="a training YAML whose data: block points at data/midtrain")
    p.add_argument("--init-from", default=None,
                   help="base checkpoint whose WEIGHTS start this run (step/optimizer/cursor reset)")
    p.add_argument("--run-name", default=None, help="run directory name (default: the config's name)")
    p.add_argument("--resume", nargs="?", const="auto", default=None,
                   help="continue an interrupted midtrain run (not for starting from a base model)")
    p.add_argument("--no-compile", action="store_true")
    p.add_argument("--max-tokens", type=int, default=None)
    p.add_argument("--max-steps", type=int, default=None,
                   help="shorthand for -o max_steps=N (reshapes the schedule too)")
    p.add_argument("--max-hours", type=float, default=None)
    p.add_argument("-o", "--override", action="append", default=[], metavar="KEY=VALUE")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    cfg = load_config(args.config)
    overrides = list(args.override)
    if args.max_steps:
        overrides.append(f"max_steps={args.max_steps}")
    if overrides:
        apply_overrides(cfg, overrides)

    data_dir = Path(cfg.data.data_dir)
    if cfg.data.source == "fineweb" and not data_dir.exists():
        raise SystemExit(
            f"{data_dir} does not exist. Build the mixture first:\n"
            f"  python scripts/prepare_midtrain_data.py configs/posttrain/midtrain_nano.yaml"
        )
    mixture = data_dir / "mixture.json"
    if mixture.exists():
        report = json.loads(mixture.read_text())
        print(f"[r52.midtrain] mixture {report.get('realised_share')}", flush=True)

    trainer = build_trainer(
        cfg, args.run_name or cfg.name, args.init_from, args.resume,
        compile_step=False if args.no_compile else None,
    )
    trainer.run(args.max_tokens, args.max_hours)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
