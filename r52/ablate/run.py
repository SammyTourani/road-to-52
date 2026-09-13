# Copyright 2026 The road-to-52 authors.
# SPDX-License-Identifier: Apache-2.0
"""Run one ablation cell: train, evaluate, write ``results/ablations/<axis>/<cell>.json``.

    python -m r52.ablate.run --axis tokenizer --cell gpt2
    python -m r52.ablate.run --axis corpus --cell dclm --dry-run
    python -m r52.ablate.run --axis arch --cell mlp-swiglu --hellaswag-limit 500

``scripts/ablate.sh`` drives this over a whole axis; this module is the unit of work and the
unit of resumability -- a cell whose JSON exists is finished, and rerunning the axis skips it.

Two subprocesses, not two function calls
----------------------------------------
Training and evaluation each run as their own ``nice -n 10`` process.  MLX's allocator hands
memory back to the OS lazily, so an in-process eval after a training loop would peak at the
sum of the two; on a 16 GB machine that is already running a multi-day headline job
(``docs/ARCHITECTURE.md`` §1.2) that is the difference between "fits" and "swaps".  It also
means a crashed eval leaves the checkpoint intact and is retried on its own.

What lands in the JSON
----------------------
Everything needed to reproduce and to audit: the cell's config path and every override, the
corpus manifest's identity (dataset id, revision, bytes/token), the exact commands, the
trainer's own end-of-run record (step, tokens, wall-clock, tok/s, peak memory), and the three
metrics.  ``docs/ARCHITECTURE.md`` §1.3: no number without a reproduction path.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .matrix import Axis, Cell, axis_names
from .matrix import axis as get_axis

__all__ = [
    "RESULTS_ROOT",
    "another_train_running",
    "cell_result",
    "main",
    "resolve_budget",
    "resolved_config",
    "run_cell",
]

RESULTS_ROOT = "results/ablations"
REPO_ROOT = Path(__file__).resolve().parents[2]


def _python() -> str:
    return os.environ.get("R52_PYTHON") or sys.executable


def another_train_running(exclude_pid: int | None = None) -> list[tuple[int, str]]:
    """PIDs of other ``r52.train`` processes on this machine.

    One GPU, one training job: ``docs/ABLATIONS.md`` -- "Runs never overlap with a headline
    run".  ``ps`` rather than a lock file, because the headline run was not started by us.
    """
    try:
        out = subprocess.run(
            ["ps", "-Ao", "pid=,command="], capture_output=True, text=True, timeout=20, check=False
        ).stdout
    except (OSError, subprocess.SubprocessError):  # pragma: no cover - ps is always there
        return []
    hits: list[tuple[int, str]] = []
    for line in out.splitlines():
        line = line.strip()
        if not line:
            continue
        pid_s, _, cmd = line.partition(" ")
        if "r52.train" not in cmd or "ablate" in cmd:
            continue
        try:
            pid = int(pid_s)
        except ValueError:
            continue
        if pid in (os.getpid(), exclude_pid):
            continue
        hits.append((pid, cmd.strip()))
    return hits


# --------------------------------------------------------------------------------------
# Budget
# --------------------------------------------------------------------------------------


def _data_dir(cell: Cell) -> str:
    """The ``data.data_dir`` the cell's overrides point at."""
    for o in cell.overrides:
        if o.startswith("data.data_dir="):
            return o.split("=", 1)[1]
    raise ValueError(f"cell {cell.name!r} has no data.data_dir override")


def _manifest(cell: Cell) -> tuple[Path, dict[str, Any]]:
    """The cell's corpus directory and its ``manifest.json``."""
    d = Path(_data_dir(cell))
    path = d / "manifest.json"
    if not path.is_file():
        raise FileNotFoundError(
            f"{path} not found. Prepare the corpus first:\n"
            f"  python scripts/prepare_corpus.py --corpus {cell.corpus} "
            f"--tokenizer {cell.tokenizer} --tokens {cell.max_tokens or 100_000_000}"
        )
    return d, json.loads(path.read_text())


def resolve_budget(cell: Cell) -> tuple[int, dict[str, Any]]:
    """``(max_tokens, provenance)`` for a cell, converting a byte budget when needed.

    The tokenizer axis fixes **bytes**, so the token budget is
    ``max_bytes / bytes_per_token`` with the bytes/token the corpus manifest measured while
    writing the shards -- the one number that makes "the same text" mean the same thing to
    two different vocabularies.
    """
    d, man = _manifest(cell)
    prov: dict[str, Any] = {
        "data_dir": str(d),
        "manifest": {
            "corpus": man.get("corpus"),
            "tokens": man.get("tokens"),
            "rows": man.get("rows"),
            "text_bytes": man.get("text_bytes"),
            "bytes_per_token": man.get("bytes_per_token"),
            "tokenizer": man.get("tokenizer"),
            "sources": [
                {k: s.get(k) for k in ("name", "repo_id", "revision", "config", "weight", "license")}
                for s in man.get("sources", [])
            ],
            "created": man.get("created"),
        },
    }
    if cell.max_tokens:
        return int(cell.max_tokens), prov
    if not cell.max_bytes:
        raise ValueError(f"cell {cell.name!r} has neither max_tokens nor max_bytes")
    bpt = float(man.get("bytes_per_token") or 0.0)
    if bpt <= 0:
        raise ValueError(f"{d}/manifest.json has no usable bytes_per_token")
    tokens = int(cell.max_bytes / bpt)
    available = int(man.get("train", {}).get("tokens") or man.get("tokens") or 0)
    prov["budget"] = {
        "mode": "fixed_bytes",
        "max_bytes": int(cell.max_bytes),
        "bytes_per_token": bpt,
        "max_tokens": tokens,
        "available_tokens": available,
        "sufficient": available >= tokens,
    }
    return tokens, prov


# --------------------------------------------------------------------------------------
# The cell
# --------------------------------------------------------------------------------------


def resolved_config(cell: Cell, extra: list[str] | None = None):
    """The cell's :class:`r52.config.Config` with its overrides applied.

    Used for the one thing the evaluator cannot guess: ``model.block_size``, which is the
    validation window and the eval's context limit.  Reading it off the cell's overrides
    alone would be wrong for every cell that does not move it.
    """
    from ..config import apply_overrides, load_config

    cfg = load_config(cell.config)
    overrides = list(cell.overrides)
    extra = list(extra or [])
    overrides += [extra[i + 1] for i in range(0, len(extra) - 1, 2) if extra[i] in ("-o", "--override")]
    return apply_overrides(cfg, overrides)


def _train_cmd(cell: Cell, max_tokens: int, resume: bool, extra: list[str], out_dir: str) -> list[str]:
    # `train.out_dir` is passed explicitly rather than left to the config: `run_cell` needs to
    # know where the checkpoint landed, and a test (or a second lab) must be able to put its
    # runs somewhere other than the repo's `runs/`.
    return [
        "nice", "-n", "10", _python(), "-m", "r52.train",
        *cell.train_argv(max_tokens),
        "-o", f"train.out_dir={out_dir}",
        *(["--resume"] if resume else []),
        *extra,
    ]


def _eval_cmd(cell: Cell, ax: Axis, ckpt: Path, out: Path, data_dir: Path,
              hellaswag_limit: int, core_tasks: list[str], memory_gib: float,
              val_tokens: int, block_size: int, max_positions: int) -> list[str]:
    val_file = next((o.split("=", 1)[1] for o in cell.overrides if o.startswith("data.val_file=")), "")
    return [
        "nice", "-n", "10", _python(), "-m", "r52.ablate.evals",
        "--model", str(ckpt),
        "--tokenizer", cell.tokenizer,
        "--data-dir", str(data_dir),
        "--val-file", val_file,
        "--val-tokens", str(val_tokens),
        "--block-size", str(block_size),
        "--hellaswag-limit", str(hellaswag_limit),
        "--core-tasks", ",".join(core_tasks),
        "--max-positions", str(max_positions),
        "--memory-limit-gib", str(memory_gib),
        "--out", str(out),
    ]


def _train_summary(run_dir: Path) -> dict[str, Any]:
    """The trainer's own ``end`` record (plus the last ``train`` record) from log.jsonl."""
    log = run_dir / "log.jsonl"
    if not log.is_file():
        return {}
    end: dict[str, Any] = {}
    last: dict[str, Any] = {}
    with log.open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except ValueError:
                continue
            if rec.get("event") == "end":
                end = rec
            elif rec.get("event") == "train":
                last = rec
    if not end:
        return {}
    return {
        "step": end.get("step"),
        "tokens": end.get("tokens"),
        "reason": end.get("reason"),
        "elapsed_s": end.get("elapsed_s"),
        "peak_mem_gb": end.get("peak_mem_gb"),
        "train_val_loss": end.get("val_loss"),
        "train_val_bpb": end.get("val_bpb"),
        "best_val": end.get("best_val"),
        "tok_s": end.get("last_tok_s") or last.get("tok_s"),
        "tflops": end.get("last_tflops") or last.get("tflops"),
    }


def cell_result(axis_name: str, cell_name: str, results_root: str = RESULTS_ROOT) -> Path:
    """Where a cell's result JSON lives."""
    from .matrix import _slug

    return Path(results_root) / _slug(axis_name) / f"{_slug(cell_name)}.json"


def run_cell(
    ax: Axis,
    cell: Cell,
    *,
    dry_run: bool = False,
    force: bool = False,
    skip_train: bool = False,
    resume: bool = True,
    results_root: str = RESULTS_ROOT,
    hellaswag_limit: int | None = None,
    core_tasks: list[str] | None = None,
    memory_gib: float = 3.0,
    val_tokens: int = 1_000_000,
    max_positions: int = 2048,
    train_extra: list[str] | None = None,
    out_dir: str = "runs",
) -> dict[str, Any]:
    """Train and evaluate one cell; write and return its result dict."""
    result_path = cell_result(ax.name, cell.name, results_root)
    if result_path.is_file() and not force:
        print(f"[abl] {ax.name}/{cell.name}: already done ({result_path})", flush=True)
        return json.loads(result_path.read_text())

    try:
        max_tokens, prov = resolve_budget(cell)
        missing = None
    except FileNotFoundError as exc:
        # A dry run must be able to print the plan before any corpus exists -- that is what
        # it is for.  A real run still stops here.
        if not dry_run:
            raise
        max_tokens = cell.max_tokens or 0
        prov = {"data_dir": _data_dir(cell), "manifest": None}
        missing = str(exc)
    data_dir = Path(prov["data_dir"])
    hs_limit = ax.hellaswag_limit if hellaswag_limit is None else hellaswag_limit
    tasks = list(core_tasks if core_tasks is not None else ax.core_tasks)
    block_size = resolved_config(cell, train_extra or []).model.block_size

    run_dir = Path(out_dir) / cell.run_name
    ckpt = run_dir / "ckpt" / "best"
    eval_out = result_path.with_suffix(".eval.json")

    train_cmd = _train_cmd(cell, max_tokens, resume and ckpt.exists(), train_extra or [], out_dir)
    eval_cmd = _eval_cmd(cell, ax, ckpt, eval_out, data_dir, hs_limit, tasks, memory_gib,
                         val_tokens, block_size, max_positions)

    record: dict[str, Any] = {
        "axis": ax.name,
        "cell": cell.name,
        "slug": cell.slug,
        "started": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "spec": cell.to_dict(),
        "budget": {"max_tokens": max_tokens, "max_bytes": cell.max_bytes},
        "corpus": prov,
        "eval_protocol": {
            "hellaswag_limit": hs_limit,
            "core_tasks": tasks,
            "metric": ax.metric,
            "memory_limit_gib": memory_gib,
            "val_tokens": val_tokens,
        },
        "commands": {"train": " ".join(train_cmd), "eval": " ".join(eval_cmd)},
        "run_dir": str(run_dir),
    }
    if "budget" in prov:
        record["budget"].update(prov["budget"])

    if dry_run:
        record["dry_run"] = True
        if missing:
            record["blocked"] = missing
        print(f"[abl] {ax.name}/{cell.name}  -> {result_path}")
        if missing:
            print(f"      BLOCKED: {missing.splitlines()[0]}")
            for line in missing.splitlines()[1:]:
                print(f"      {line.strip()}")
        print(f"      train: {' '.join(train_cmd)}")
        print(f"      eval : {' '.join(eval_cmd)}")
        return record

    t0 = time.time()
    if not skip_train:
        print(f"[abl] {ax.name}/{cell.name}: training {max_tokens:,} tokens", flush=True)
        rc = subprocess.run(train_cmd, cwd=REPO_ROOT, check=False).returncode
        if rc != 0:
            raise SystemExit(f"[abl] training failed for {ax.name}/{cell.name} (exit {rc})")
    record["train"] = _train_summary(run_dir)

    if not ckpt.is_dir():
        raise FileNotFoundError(f"{ckpt} not found after training {cell.run_name}")
    print(f"[abl] {ax.name}/{cell.name}: evaluating {ckpt}", flush=True)
    rc = subprocess.run(eval_cmd, cwd=REPO_ROOT, check=False).returncode
    if rc != 0:
        raise SystemExit(f"[abl] evaluation failed for {ax.name}/{cell.name} (exit {rc})")
    record["eval"] = json.loads(eval_out.read_text())
    eval_out.unlink(missing_ok=True)

    record["finished"] = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    record["wall_clock_s"] = round(time.time() - t0, 2)
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(json.dumps(record, indent=2))
    print(f"[abl] wrote {result_path}", flush=True)
    return record


# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="r52.ablate.run",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--axis", required=True, choices=axis_names())
    p.add_argument("--cell", required=True)
    p.add_argument("--config", default=None, help="override the axis's base config")
    p.add_argument("--tokenizer", default=None, help="override the axis's tokenizer")
    p.add_argument("--corpus", default=None, help="corpus for the arch/tokenizer axes")
    p.add_argument("--max-tokens", type=int, default=None, help="override the cell's token budget")
    p.add_argument("--hellaswag-limit", type=int, default=None)
    p.add_argument("--core-tasks", default=None, help="comma-separated ('' = skip CORE)")
    p.add_argument("--core-limit", type=int, default=0)
    p.add_argument("--memory-limit-gib", type=float, default=3.0)
    p.add_argument("--val-tokens", type=int, default=1_000_000)
    p.add_argument("--max-positions", type=int, default=2048)
    p.add_argument("--results-dir", default=RESULTS_ROOT)
    p.add_argument("--out-dir", default="runs", help="where r52.train writes runs/")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--force", action="store_true", help="re-run a finished cell")
    p.add_argument("--skip-train", action="store_true", help="evaluate an existing checkpoint")
    p.add_argument("--no-resume", action="store_true")
    p.add_argument("-o", "--override", action="append", default=[], metavar="KEY=VALUE",
                   help="extra r52.train override appended to the cell's own")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    ax = get_axis(
        args.axis,
        config=args.config,
        tokenizer=args.tokenizer,
        corpus=args.corpus,
        max_tokens=args.max_tokens,
    )
    cell = ax.cell(args.cell)
    tasks = None if args.core_tasks is None else [t for t in args.core_tasks.split(",") if t.strip()]
    extra: list[str] = []
    for o in args.override:
        extra += ["-o", o]
    run_cell(
        ax,
        cell,
        dry_run=args.dry_run,
        force=args.force,
        skip_train=args.skip_train,
        resume=not args.no_resume,
        results_root=args.results_dir,
        hellaswag_limit=args.hellaswag_limit,
        core_tasks=tasks,
        memory_gib=args.memory_limit_gib,
        val_tokens=args.val_tokens,
        max_positions=args.max_positions,
        train_extra=extra,
        out_dir=args.out_dir,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
