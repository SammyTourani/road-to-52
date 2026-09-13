# Copyright 2026 The road-to-52 authors.
# SPDX-License-Identifier: Apache-2.0
"""Render ``results/ablations/<axis>.md`` from the cell JSONs.

    python -m r52.ablate.report --axis corpus
    python -m r52.ablate.report --all

The table is the one ``docs/ABLATIONS.md`` §Deliverables asks for -- val bpb, HellaSwag
acc_norm with a 95% CI, the CORE-6 subset, wall-clock -- plus **our rank next to the
published one**, with the source pointer for every published claim.  That last column is the
point of the exercise: ``docs/PLAN.md`` §3.1 item 1 leans on DataDecide's finding that corpus
rankings at 150M parameters predict the 1B winner ~80% of the time, and our proxies sit at
12-35M, *below* the scale that result was validated at.  So the honest deliverable is not
"our ranking" but "our ranking, and where it agrees".

Partial axes render fine: a cell with no JSON yet is listed as pending, and the agreement
line is computed only over the cells that have landed.
"""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .matrix import Axis, axis_names
from .matrix import axis as get_axis

__all__ = ["RESULTS_ROOT", "load_results", "main", "render_axis", "write_axis"]

RESULTS_ROOT = "results/ablations"


def load_results(ax: Axis, results_root: str = RESULTS_ROOT) -> dict[str, dict[str, Any]]:
    """``{cell slug: result dict}`` for every cell of ``ax`` that has finished."""
    out: dict[str, dict[str, Any]] = {}
    for cell in ax.cells:
        p = Path(results_root) / ax.name / f"{cell.slug}.json"
        if p.is_file():
            try:
                rec = json.loads(p.read_text())
            except ValueError:
                continue
            if not rec.get("dry_run"):
                out[cell.slug] = rec
    return out


def _get(rec: dict[str, Any], *path: str, default: Any = None) -> Any:
    cur: Any = rec
    for key in path:
        if not isinstance(cur, dict):
            return default
        cur = cur.get(key)
    return default if cur is None else cur


def _fmt(value: Any, spec: str = ".4f") -> str:
    if value is None or value == "":
        return "—"
    try:
        return format(float(value), spec)
    except (TypeError, ValueError):
        return str(value)


def _hours(seconds: Any) -> str:
    try:
        s = float(seconds)
    except (TypeError, ValueError):
        return "—"
    return f"{s / 3600:.2f} h" if s >= 3600 else f"{s / 60:.1f} min"


def _rank(values: dict[str, float], lower_is_better: bool) -> dict[str, int]:
    order = sorted(values, key=lambda k: values[k], reverse=not lower_is_better)
    return {k: i + 1 for i, k in enumerate(order)}


def render_axis(ax: Axis, results: dict[str, dict[str, Any]]) -> str:
    """The markdown body for one axis."""
    bpb = {
        slug: float(_get(r, "eval", "val", "val_bpb", default="nan"))
        for slug, r in results.items()
        if _get(r, "eval", "val", "val_bpb") is not None
    }
    bpb = {k: v for k, v in bpb.items() if v == v}  # drop NaN
    ours = _rank(bpb, lower_is_better=True)
    published = {c.slug: c.published_rank for c in ax.cells if c.published_rank}

    lines: list[str] = []
    lines.append(f"# Ablation axis: `{ax.name}`")
    lines.append("")
    lines.append(f"*Generated {datetime.now(UTC).strftime('%Y-%m-%d %H:%M UTC')} by "
                 f"`python -m r52.ablate.report --axis {ax.name}`. "
                 f"{len(results)}/{len(ax)} cells complete.*")
    lines.append("")
    lines.append(ax.description)
    lines.append("")
    lines.append(f"Ranked by **{ax.metric}** (lower is better). HellaSwag is `acc_norm` with a 95% "
                 f"Wilson interval; CORE-{len(ax.core_tasks)} is the centered mean over "
                 f"`{', '.join(ax.core_tasks)}`.")
    lines.append("")

    header = ("| cell | val bpb | val loss | HellaSwag acc_norm (95% CI) | "
              f"CORE-{len(ax.core_tasks)} | tokens | wall-clock | ours | published |")
    lines.append(header)
    lines.append("|" + "---|" * 9)

    for cell in ax.cells:
        r = results.get(cell.slug)
        if r is None:
            lines.append(f"| `{cell.name}` | _pending_ | | | | | | | "
                         f"{published.get(cell.slug) or '—'} |")
            continue
        hs = _get(r, "eval", "hellaswag", default={})
        lo, hi = (hs.get("acc_norm_ci95") or [None, None])[:2]
        ci = f"{_fmt(hs.get('acc_norm'))} [{_fmt(lo)}, {_fmt(hi)}]" if hs else "—"
        tokens = _get(r, "train", "tokens") or _get(r, "budget", "max_tokens")
        n_tok = f"{int(tokens):,}" if tokens else "—"
        lines.append(
            f"| `{cell.name}` "
            f"| **{_fmt(_get(r, 'eval', 'val', 'val_bpb'))}** "
            f"| {_fmt(_get(r, 'eval', 'val', 'val_loss'))} "
            f"| {ci} "
            f"| {_fmt(_get(r, 'eval', 'core', 'core_metric'), '.5f')} "
            f"| {n_tok} "
            f"| {_hours(r.get('wall_clock_s'))} "
            f"| {ours.get(cell.slug, '—')} "
            f"| {published.get(cell.slug) or '—'} |"
        )
    lines.append("")

    if published and ours:
        shared = [s for s in published if s in ours]
        agree = sum(1 for s in shared if ours[s] == published[s])
        pairs = [
            (a, b)
            for i, a in enumerate(shared)
            for b in shared[i + 1 :]
            if (ours[a] < ours[b]) == (published[a] < published[b])
        ]
        n_pairs = len(shared) * (len(shared) - 1) // 2
        lines.append(f"**Agreement with the published ranking** ({len(shared)} ranked cells "
                     f"complete): {agree}/{len(shared)} exact positions, "
                     f"{len(pairs)}/{n_pairs} pairwise orderings.")
        lines.append("")
        lines.append("> Both proxies here (12M and 35M parameters) sit **below** the 150M scale at "
                     "which DataDecide validated that corpus rankings transfer (`docs/PLAN.md` §3.1 "
                     "item 1). Disagreement is evidence about the proxy as much as about the "
                     "corpus, and is reported as such.")
        lines.append("")

    notes = [(c.name, c.published_note or c.note) for c in ax.cells if (c.published_note or c.note)]
    if notes:
        lines.append("## Published expectations and sources")
        lines.append("")
        for name, note in notes:
            lines.append(f"- **`{name}`** — {note}")
        lines.append("")

    if results:
        lines.append("## Provenance")
        lines.append("")
        lines.append("| cell | dataset | revision | bytes/token | train command |")
        lines.append("|---|---|---|---|---|")
        for cell in ax.cells:
            r = results.get(cell.slug)
            if r is None:
                continue
            srcs = _get(r, "corpus", "manifest", "sources", default=[]) or []
            ds = " + ".join(f"{s.get('repo_id')}" for s in srcs) or "—"
            rev = " + ".join((s.get("revision") or "")[:8] for s in srcs) or "—"
            lines.append(
                f"| `{cell.name}` | {ds} | `{rev}` "
                f"| {_fmt(_get(r, 'corpus', 'manifest', 'bytes_per_token'), '.4f')} "
                f"| `{_get(r, 'commands', 'train', default='—')}` |"
            )
        lines.append("")

    return "\n".join(lines) + "\n"


def write_axis(ax: Axis, results_root: str = RESULTS_ROOT) -> Path:
    """Render and write ``<results_root>/<axis>.md``. Returns the path."""
    results = load_results(ax, results_root)
    path = Path(results_root) / f"{ax.name}.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_axis(ax, results))
    return path


# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="r52.ablate.report",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--axis", default=None, choices=axis_names())
    p.add_argument("--all", action="store_true", help="render every axis")
    p.add_argument("--config", default=None)
    p.add_argument("--tokenizer", default=None)
    p.add_argument("--corpus", default=None)
    p.add_argument("--results-dir", default=RESULTS_ROOT)
    p.add_argument("--stdout", action="store_true", help="print instead of writing the file")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not args.axis and not args.all:
        args.all = True
    names = [args.axis] if args.axis else axis_names()
    for name in names:
        ax = get_axis(name, config=args.config, tokenizer=args.tokenizer, corpus=args.corpus)
        if args.stdout:
            print(render_axis(ax, load_results(ax, args.results_dir)))
        else:
            print(f"wrote {write_axis(ax, args.results_dir)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
