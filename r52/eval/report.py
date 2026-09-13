# Copyright 2026 The road-to-52 authors.
# SPDX-License-Identifier: Apache-2.0
"""Results plumbing: a JSON file per eval, one aggregate for the gap tracker, one MD row.

``docs/ARCHITECTURE.md`` §1.3 -- "honest numbers only" -- says every reported score carries
its exact command, its sample count, its tokenizer, its sequence length, the git commit and
the wall-clock.  :class:`EvalResult` is that record, and :func:`write_result` puts it in
three places:

``results/<run>/<eval>.json``
    The full record, ``conditions`` as a structured object::

        {
          "model": "gpt2-124m", "benchmark": "hellaswag",
          "value": 29.55, "unit": "% acc_norm",
          "conditions": {"tokenizer": "gpt2 (tiktoken)", "block_size": 1024,
                         "limit": null, "n_examples": 10042, "few_shot": 0,
                         "command": "python -m r52.eval.hellaswag --model models/gpt2-mlx"},
          "commit": "a8de28c", "date": "2026-09-13",
          "wall_clock_s": 1234.5, "machine": "Mac Mini M4 ...",
          "command": "...", "metrics": {...}
        }

``results/<run>/eval.json``
    The same records as a **list**, in the flat schema ``r52/bar/gap.py`` documents and reads
    (``conditions`` flattened to a string, ``command`` at the top level).  ``gap.py`` merges
    these into ``results/GAP.md``; its field names win, so this file is written to its shape
    and not ours.  ``model`` and ``benchmark`` must be ids that exist in ``r52/bar/bar.yaml``
    (``gpt2-124m`` / ``r52-gpt2-124m-mac``; ``fineweb-val-loss`` / ``hellaswag`` /
    ``dclm-core``) or the row is silently ignored by the gap table.

``docs/RESULTS.md``
    One appended markdown row, newest last.
"""

from __future__ import annotations

import json
import platform
import subprocess
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

__all__ = [
    "EvalResult",
    "append_results_md",
    "conditions_string",
    "git_commit",
    "machine_string",
    "results_dir",
    "write_result",
]

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = REPO_ROOT / "results"
RESULTS_MD = REPO_ROOT / "docs" / "RESULTS.md"

_MD_HEADER = "| date | run | eval | value | conditions | commit |"


@dataclass
class EvalResult:
    """One measured number plus everything needed to reproduce it.

    Attributes:
        model: model id -- use a ``r52/bar/bar.yaml`` id (``gpt2-124m``,
            ``r52-gpt2-124m-mac``, ...) so the gap tracker picks the row up.
        benchmark: benchmark id from ``bar.yaml`` (``fineweb-val-loss``, ``hellaswag``,
            ``dclm-core``).
        value: the headline number, in ``unit``.
        unit: e.g. ``"nats/token"``, ``"% acc_norm"``, ``"CORE"``.
        conditions: structured eval conditions; ``tokenizer``, ``block_size``, ``limit``,
            ``n_examples``, ``few_shot`` and ``command`` are always present.
        commit: git commit the eval ran at (short hash, ``+dirty`` if the tree was dirty).
        date: ISO date (UTC).
        wall_clock_s: wall-clock of the eval, **seconds**.
        machine: chip / RAM / OS / MLX version string.
        metrics: every other number the eval produced (per-task scores, CIs, bpb, ...).
    """

    model: str
    benchmark: str
    value: float
    unit: str
    conditions: dict[str, Any]
    commit: str
    date: str
    wall_clock_s: float
    machine: str
    metrics: dict[str, Any] = field(default_factory=dict)

    @property
    def command(self) -> str:
        return str(self.conditions.get("command", ""))

    def to_dict(self) -> dict[str, Any]:
        """The full record written to ``results/<run>/<eval>.json``."""
        return {
            "model": self.model,
            "benchmark": self.benchmark,
            "value": self.value,
            "unit": self.unit,
            "conditions": self.conditions,
            "commit": self.commit,
            "date": self.date,
            "wall_clock_s": round(float(self.wall_clock_s), 3),
            "machine": self.machine,
            "command": self.command,
            "metrics": self.metrics,
        }

    def to_gap_dict(self) -> dict[str, Any]:
        """The flat record ``r52/bar/gap.py`` reads from ``results/<run>/eval.json``."""
        return {
            "model": self.model,
            "benchmark": self.benchmark,
            "value": self.value,
            "unit": self.unit,
            "conditions": conditions_string(self.conditions),
            "commit": self.commit,
            "command": self.command,
            "date": self.date,
        }


def conditions_string(conditions: dict[str, Any]) -> str:
    """Flatten structured conditions to the one-line string ``gap.py`` renders in a cell."""
    order = ["n_examples", "limit", "few_shot", "tokenizer", "block_size"]
    labels = {
        "n_examples": "examples",
        "limit": "limit",
        "few_shot": "few-shot",
        "tokenizer": "tokenizer",
        "block_size": "block_size",
    }
    parts = []
    for key in order:
        value = conditions.get(key)
        if value is None or value == "":
            continue
        parts.append(f"{labels[key]} {value}")
    for key, value in conditions.items():
        if key in order or key == "command" or value is None or value == "":
            continue
        parts.append(f"{key} {value}")
    return ", ".join(parts)


def git_commit(root: Path | None = None) -> str:
    """Short git hash of the working tree, ``+dirty`` when there are uncommitted changes."""
    root = root or REPO_ROOT
    try:
        head = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=10, check=True,
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "-C", str(root), "status", "--porcelain"],
            capture_output=True, text=True, timeout=20, check=True,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return "unknown"
    return f"{head}+dirty" if dirty else head


def machine_string() -> str:
    """``"Mac Mini M4 (Apple M4), 16 GiB, macOS 26.5.2, MLX 0.32.2"`` -- best effort."""
    chip = _sysctl("machdep.cpu.brand_string") or platform.processor() or "unknown CPU"
    ram = _sysctl("hw.memsize")
    ram_s = f", {int(ram) / 2**30:.0f} GiB" if ram and ram.isdigit() else ""
    mac = platform.mac_ver()[0]
    os_s = f", macOS {mac}" if mac else f", {platform.system()} {platform.release()}"
    try:
        import mlx.core as mx

        mlx_s = f", MLX {mx.__version__}"
    except Exception:  # pragma: no cover - MLX is a hard dependency
        mlx_s = ""
    return f"{chip}{ram_s}{os_s}{mlx_s}"


def _sysctl(key: str) -> str | None:
    try:
        return subprocess.run(
            ["sysctl", "-n", key], capture_output=True, text=True, timeout=5, check=True
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):  # pragma: no cover - non-macOS
        return None


def results_dir(run: str, root: Path | None = None) -> Path:
    """``results/<run>/``, created if missing."""
    d = (root or RESULTS_DIR) / run
    d.mkdir(parents=True, exist_ok=True)
    return d


def make_result(
    *,
    model: str,
    benchmark: str,
    value: float,
    unit: str,
    conditions: dict[str, Any],
    wall_clock_s: float,
    metrics: dict[str, Any] | None = None,
    commit: str | None = None,
    date: str | None = None,
    machine: str | None = None,
) -> EvalResult:
    """Build an :class:`EvalResult`, filling in commit / date / machine from this machine."""
    return EvalResult(
        model=model,
        benchmark=benchmark,
        value=float(value),
        unit=unit,
        conditions=conditions,
        commit=commit or git_commit(),
        date=date or datetime.now(UTC).strftime("%Y-%m-%d"),
        wall_clock_s=float(wall_clock_s),
        machine=machine or machine_string(),
        metrics=metrics or {},
    )


def write_result(
    result: EvalResult,
    run: str,
    eval_name: str,
    *,
    results_root: Path | None = None,
    results_md: Path | None = None,
    append_md: bool = True,
) -> Path:
    """Write ``results/<run>/<eval_name>.json``, refresh ``eval.json``, append the MD row.

    ``results_root`` defaults to ``results/``.  Pointing it somewhere else also moves the
    markdown row to ``<results_root>/RESULTS.md``, so a smoke run can be told to keep its
    hands off the committed log (``--results-dir`` on every eval CLI).

    Returns the path of the per-eval JSON file.
    """
    root = results_root or RESULTS_DIR
    if results_md is None and root.resolve() != RESULTS_DIR.resolve():
        results_md = root / "RESULTS.md"
    d = results_dir(run, root)
    path = d / f"{eval_name}.json"
    path.write_text(json.dumps(result.to_dict(), indent=2) + "\n")
    _merge_gap_json(d / "eval.json", result)
    if append_md:
        append_results_md(result, run, eval_name, path=results_md)
    return path


def _merge_gap_json(path: Path, result: EvalResult) -> None:
    """Keep ``results/<run>/eval.json`` as the list of newest-per-benchmark gap records."""
    existing: list[dict[str, Any]] = []
    if path.exists():
        try:
            payload = json.loads(path.read_text())
        except ValueError:
            payload = []
        existing = payload if isinstance(payload, list) else [payload]
    entry = result.to_gap_dict()
    merged = [e for e in existing if not _same_cell(e, entry)]
    merged.append(entry)
    merged.sort(key=lambda e: (str(e.get("benchmark")), str(e.get("unit"))))
    path.write_text(json.dumps(merged, indent=2) + "\n")


def _same_cell(a: dict[str, Any], b: dict[str, Any]) -> bool:
    return (
        a.get("model") == b.get("model")
        and a.get("benchmark") == b.get("benchmark")
        and a.get("unit") == b.get("unit")
    )


def append_results_md(
    result: EvalResult, run: str, eval_name: str, path: Path | None = None
) -> Path:
    """Append one row to ``docs/RESULTS.md`` (creating the table header if absent)."""
    md = path or RESULTS_MD
    md.parent.mkdir(parents=True, exist_ok=True)
    if not md.exists():
        md.write_text(
            "# Results log\n\nAppended automatically by `r52.eval.report`.\n\n"
            f"{_MD_HEADER}\n|---|---|---|---|---|---|\n"
        )
    text = md.read_text()
    if _MD_HEADER not in text:  # pragma: no cover - defensive
        text += f"\n{_MD_HEADER}\n|---|---|---|---|---|---|\n"
    value = _format_value(result.value, result.unit)
    conditions = conditions_string(result.conditions)
    cmd = result.command.replace("|", "\\|")
    row = (
        f"| {result.date} | {run} | {eval_name} | {value} | "
        f"{conditions}; wall-clock {result.wall_clock_s:.0f} s; `{cmd}` | {result.commit} |\n"
    )
    if not text.endswith("\n"):
        text += "\n"
    md.write_text(text + row)
    return md


def _format_value(value: float, unit: str) -> str:
    if unit.startswith("%"):
        return f"{value:.2f} {unit}"
    if unit == "CORE":
        return f"{value:.6f} {unit}"
    return f"{value:.4f} {unit}"
