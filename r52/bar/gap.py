# Copyright 2026 The road-to-52 authors.
# SPDX-License-Identifier: Apache-2.0
"""The gap tracker: every model measured against Claude Fable 5.1's published scores.

Reads ``bar.yaml`` (published numbers, each with a source URL, a retrieval date and its eval
conditions) and, if present, our own measured results from ``results/*/eval.json``.  Renders
``results/GAP.md`` and ``results/GAP.json``.

Local results schema (``results/<run>/eval.json``) -- one object, or a list of them::

    {
      "model":      "r52-gpt2-124m-mac",   # must match a model id in bar.yaml
      "benchmark":  "hellaswag",           # must match a benchmark id in bar.yaml
      "value":      31.4,
      "unit":       "% acc_norm",
      "conditions": "10042 val examples, GPT-2 tokenizer, block_size 1024, acc_norm",
      "commit":     "a1b2c3d",             # git commit the eval ran at
      "command":    "python -m r52.eval.hellaswag --ckpt runs/x/ckpt/best",
      "date":       "2026-09-20"
    }

``commit`` and ``command`` are the reproduction path docs/ARCHITECTURE.md §1.3 demands; they
become the cell's source, since there is no URL for a number we measured ourselves.

There are **no network calls** here or in the tests -- every published number lives in the YAML.

CLI::

    python -m r52.bar                 # render results/GAP.{md,json}
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import yaml

__all__ = [
    "DASH",
    "SAME_SOURCE_MARK",
    "UNVERIFIED_MARK",
    "Bar",
    "Benchmark",
    "GapRow",
    "Model",
    "Score",
    "load_bar",
    "load_local_results",
    "main",
    "render_json",
    "render_markdown",
]

DASH = "\u2014"  # "no published number exists" -- never 0
UNVERIFIED_MARK = "\u2020"  # dagger: research/01 or /04 flagged this number UNVERIFIED
SAME_SOURCE_MARK = "\u2261"  # identical: both sides of the gap come from one source/harness

_HERE = Path(__file__).resolve().parent
REPO_ROOT = _HERE.parents[1]
BAR_YAML = _HERE / "bar.yaml"
RESULTS_DIR = REPO_ROOT / "results"

_SUPERSCRIPT = str.maketrans("0123456789", "\u2070\u00b9\u00b2\u00b3\u2074\u2075\u2076\u2077\u2078\u2079")


# ======================================================================================
# Data model
# ======================================================================================


@dataclass(frozen=True)
class Benchmark:
    """One benchmark row, with what it measures and whether we can run it."""

    id: str
    name: str
    measures: str
    url: str
    category: str
    unit: str
    higher_is_better: bool = True
    saturated: bool = False
    local_runnable: bool = False
    harness: str | None = None
    consensus_rank: str | None = None
    notes: str | None = None


@dataclass(frozen=True)
class Score:
    """One published (or measured) number, with everything needed to judge it."""

    model: str
    benchmark: str
    value: float
    unit: str
    source_url: str | None
    retrieved: str
    conditions: str
    verified: bool = True
    primary: bool = False
    local: bool = False
    commit: str | None = None
    command: str | None = None


@dataclass(frozen=True)
class Model:
    """One model column."""

    id: str
    name: str
    short: str
    org: str
    weights: str
    status: str = "released"
    license: str | None = None
    released: str | None = None
    params_total: float | None = None
    params_active: float | None = None
    notes: str | None = None
    targets: dict[str, Any] = field(default_factory=dict)

    @property
    def is_open(self) -> bool:
        return self.weights == "open"


@dataclass
class Bar:
    """The loaded bar: benchmarks, models, scores and the saturation scorecard."""

    benchmarks: list[Benchmark]
    models: list[Model]
    scores: list[Score]
    meta: dict[str, Any] = field(default_factory=dict)
    saturation_scorecard: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self._index: dict[tuple[str, str], list[Score]] = {}
        for score in self.scores:
            self._index.setdefault((score.model, score.benchmark), []).append(score)

    @property
    def target_id(self) -> str:
        return self.meta.get("target_model", "claude-fable-5-1")

    def benchmark(self, bid: str) -> Benchmark:
        for b in self.benchmarks:
            if b.id == bid:
                return b
        raise KeyError(bid)

    def model(self, mid: str) -> Model:
        for m in self.models:
            if m.id == mid:
                return m
        raise KeyError(mid)

    def score(self, model_id: str, benchmark_id: str) -> Score | None:
        """The score to display for this cell: a local measurement wins, then ``primary``."""
        entries = self._index.get((model_id, benchmark_id))
        if not entries:
            return None
        for entry in entries:
            if entry.local:
                return entry
        for entry in entries:
            if entry.primary:
                return entry
        return entries[0]

    def all_scores(self, model_id: str, benchmark_id: str) -> list[Score]:
        return list(self._index.get((model_id, benchmark_id), []))

    def models_with_scores(self, benchmark_ids: list[str]) -> list[Model]:
        """Models holding at least one score among ``benchmark_ids``, in bar.yaml order."""
        wanted = set(benchmark_ids)
        return [
            m
            for m in self.models
            if any(bid in wanted for (mid, bid) in self._index if mid == m.id)
        ]

    def best_open(self, benchmark_id: str) -> Score | None:
        """Best open-weight score on this benchmark, respecting the benchmark's direction.

        ``weights: closed`` models are excluded even where a secondary source calls them open
        (the Qwen3.8-Max case: Epoch classifies it closed, so it cannot hold this column).
        """
        bench = self.benchmark(benchmark_id)
        pool = [
            s
            for s in self.scores
            if s.benchmark == benchmark_id and self.model(s.model).is_open and not s.local
        ]
        if not pool:
            return None
        # Only `primary` entries compete.  Where a vendor's self-reported figure sits next to an
        # independent one (DeepSeek's 90.6 vs Vals' 74.53 on Terminal-Bench 2.1), the independent
        # one is marked primary and the vendor's must not be allowed to win this column.
        primaries = [s for s in pool if s.primary]
        candidates = primaries or pool
        chooser = max if bench.higher_is_better else min
        return chooser(candidates, key=lambda s: s.value)

    def target_for_gap(self, benchmark_id: str, best_open: Score | None) -> Score | None:
        """The target's score to compare against ``best_open``.

        Prefers an entry from the *same evaluator* as the open-weight number, because a gap
        between two different harnesses is not a gap (research/01 §6.1).  Falls back to the
        primary entry.
        """
        entries = self.all_scores(self.target_id, benchmark_id)
        if not entries:
            return None
        if best_open is not None:
            host = _host(best_open.source_url)
            same = [e for e in entries if host and _host(e.source_url) == host]
            if same:
                return same[0]
        return self.score(self.target_id, benchmark_id)


@dataclass(frozen=True)
class GapRow:
    """One rendered benchmark row: the target's score, the best open score, and the gap."""

    benchmark: Benchmark
    target: Score | None
    best_open: Score | None
    gap: float | None
    same_source: bool
    gap_target: Score | None = None


# ======================================================================================
# Loading
# ======================================================================================


def _host(url: str | None) -> str | None:
    """Hostname of a source URL, used to tell "same evaluator" from "two different harnesses"."""
    if not url:
        return None
    parsed = urlparse(url)
    return parsed.netloc.lower() or None


def _num(value: Any) -> float | None:
    """Coerce a YAML scalar to float (PyYAML's YAML 1.1 floats need a signed exponent)."""
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value.strip())
        except ValueError:
            return None
    return None


def load_bar(path: Path | None = None) -> Bar:
    """Load ``bar.yaml`` and validate that every score points at a real benchmark."""
    with (path or BAR_YAML).open("r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)

    benchmarks = [
        Benchmark(
            id=b["id"],
            name=b["name"],
            measures=b["measures"],
            url=b["url"],
            category=b["category"],
            unit=b["unit"],
            higher_is_better=bool(b.get("higher_is_better", True)),
            saturated=bool(b.get("saturated", False)),
            local_runnable=bool(b.get("local_runnable", False)),
            harness=b.get("harness"),
            consensus_rank=b.get("consensus_rank"),
            notes=b.get("notes"),
        )
        for b in raw["benchmarks"]
    ]
    known = {b.id for b in benchmarks}

    models: list[Model] = []
    scores: list[Score] = []
    for m in raw["models"]:
        models.append(
            Model(
                id=m["id"],
                name=m["name"],
                short=m.get("short", m["name"]),
                org=m["org"],
                weights=m["weights"],
                status=m.get("status", "released"),
                license=m.get("license"),
                released=str(m["released"]) if m.get("released") else None,
                params_total=_num(m.get("params_total")),
                params_active=_num(m.get("params_active")),
                notes=m.get("notes"),
                targets=m.get("targets") or {},
            )
        )
        for s in m.get("scores") or []:
            if s["benchmark"] not in known:
                raise ValueError(f"model {m['id']!r}: unknown benchmark {s['benchmark']!r}")
            if not s.get("source_url"):
                raise ValueError(f"model {m['id']!r}/{s['benchmark']!r}: score has no source_url")
            if not s.get("retrieved"):
                raise ValueError(f"model {m['id']!r}/{s['benchmark']!r}: score has no retrieved date")
            value = _num(s["value"])
            if value is None:
                raise ValueError(f"model {m['id']!r}/{s['benchmark']!r}: score has no numeric value")
            scores.append(
                Score(
                    model=m["id"],
                    benchmark=s["benchmark"],
                    value=value,
                    unit=s["unit"],
                    source_url=s["source_url"],
                    retrieved=str(s["retrieved"]),
                    conditions=s["conditions"],
                    verified=bool(s.get("verified", True)),
                    primary=bool(s.get("primary", False)),
                )
            )

    return Bar(
        benchmarks=benchmarks,
        models=models,
        scores=scores,
        meta=raw.get("meta") or {},
        saturation_scorecard=raw.get("saturation_scorecard") or {},
    )


def load_local_results(results_dir: Path | None = None) -> list[Score]:
    """Load our own measurements from ``results/*/eval.json`` (see the module docstring)."""
    root = results_dir or RESULTS_DIR
    out: list[Score] = []
    if not root.is_dir():
        return out
    for path in sorted(root.glob("*/eval.json")):
        with path.open("r", encoding="utf-8") as fh:
            payload = json.load(fh)
        for entry in payload if isinstance(payload, list) else [payload]:
            value = _num(entry.get("value"))
            if value is None:
                continue
            rel = path.relative_to(REPO_ROOT) if path.is_relative_to(REPO_ROOT) else path
            out.append(
                Score(
                    model=entry["model"],
                    benchmark=entry["benchmark"],
                    value=value,
                    unit=entry.get("unit", ""),
                    source_url=str(rel),
                    retrieved=str(entry.get("date", "")),
                    conditions=(
                        f"MEASURED BY US. {entry.get('conditions', '')} "
                        f"[commit {entry.get('commit', 'n/d')}; `{entry.get('command', 'n/d')}`]"
                    ).strip(),
                    verified=True,
                    primary=True,
                    local=True,
                    commit=entry.get("commit"),
                    command=entry.get("command"),
                )
            )
    return out


def merge_local(bar: Bar, local: list[Score]) -> Bar:
    """Return a Bar with our own measurements merged in (they win their cell)."""
    known_b = {b.id for b in bar.benchmarks}
    known_m = {m.id for m in bar.models}
    kept = [s for s in local if s.benchmark in known_b and s.model in known_m]
    return Bar(
        benchmarks=bar.benchmarks,
        models=bar.models,
        scores=bar.scores + kept,
        meta=bar.meta,
        saturation_scorecard=bar.saturation_scorecard,
    )


# ======================================================================================
# Gap computation
# ======================================================================================


def gap_rows(bar: Bar, category: str) -> list[GapRow]:
    """Build one :class:`GapRow` per benchmark in ``category``."""
    rows: list[GapRow] = []
    for bench in bar.benchmarks:
        if bench.category != category:
            continue
        target = bar.score(bar.target_id, bench.id)
        best = bar.best_open(bench.id)
        paired = bar.target_for_gap(bench.id, best)
        gap: float | None = None
        same_source = False
        if paired is not None and best is not None:
            gap = paired.value - best.value if bench.higher_is_better else best.value - paired.value
            same_source = _host(paired.source_url) == _host(best.source_url) and bool(
                _host(best.source_url)
            )
        rows.append(
            GapRow(
                benchmark=bench,
                target=target,
                best_open=best,
                gap=gap,
                same_source=same_source,
                gap_target=paired,
            )
        )
    return rows


# ======================================================================================
# Formatting
# ======================================================================================


def fmt_value(score: Score | None, bench: Benchmark) -> str:
    """Format a score value for a table cell.  ``None`` renders as an em dash, never 0."""
    if score is None:
        return DASH
    value = score.value
    if bench.unit == "USD":
        text = f"${value:,.2f}"
    elif abs(value) >= 1000:
        text = f"{value:,.0f}"
    elif value == int(value):
        text = f"{int(value)}"
    elif abs(value) < 1:
        text = f"{value:.6g}"
    else:
        text = f"{value:g}"
    return text


def fmt_gap(gap: float | None, bench: Benchmark) -> str:
    """Format the gap, signed so that positive always means 'Fable 5.1 is ahead'."""
    if gap is None:
        return DASH
    suffix = " pts" if bench.unit in ("%", "% acc", "% acc_norm", "% pass@1", "% RHAE") else ""
    if bench.unit == "USD":
        return f"{gap:+,.2f} USD"
    if abs(gap) < 0.0005:
        return f"0{suffix}"
    return f"{gap:+.4g}{suffix}"


def _sup(n: int) -> str:
    return str(n).translate(_SUPERSCRIPT)


class Footnotes:
    """A deduplicated pool of eval conditions, rendered as superscript markers."""

    def __init__(self) -> None:
        self._order: list[tuple[str, str | None]] = []
        self._index: dict[tuple[str, str | None], int] = {}

    def mark(self, score: Score | None) -> str:
        """Return the marker(s) for a score: a footnote number, plus a dagger if unverified."""
        if score is None:
            return ""
        key = (score.conditions, score.source_url)
        if key not in self._index:
            self._order.append(key)
            self._index[key] = len(self._order)
        out = _sup(self._index[key])
        if not score.verified:
            out += UNVERIFIED_MARK
        return out

    def render(self) -> list[str]:
        lines = []
        for i, (conditions, url) in enumerate(self._order, start=1):
            src = f" \u2014 [source]({url})" if url else ""
            lines.append(f"{i}. {conditions}{src}")
        return lines


def _cell(text: Any) -> str:
    if text is None:
        return DASH
    return str(text).replace("|", "\\|").replace("\n", " ").strip()


def _table(header: list[str], rows: list[list[str]]) -> str:
    out = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    out.extend("| " + " | ".join(r) + " |" for r in rows)
    return "\n".join(out)


# ======================================================================================
# Rendering
# ======================================================================================

GROUPS = (
    ("frontier-2026", "2026 frontier set"),
    ("classic-local", "Classic local set"),
)

_MODELS_PER_TABLE = 6


def _headline_table(bar: Bar, rows: list[GapRow], notes: Footnotes) -> str:
    target = bar.model(bar.target_id)
    body = []
    for row in rows:
        if row.target is None and row.best_open is None:
            continue
        who = bar.model(row.best_open.model).short if row.best_open else DASH
        arrow = ""
        if row.gap is not None and row.gap < 0:
            arrow = " **(open leads)**"
        if row.gap_target is not None and row.target is not None and row.gap_target is not row.target:
            arrow += f" *(vs {target.short} {fmt_value(row.gap_target, row.benchmark)} on the same run)*"
        body.append(
            [
                _cell(row.benchmark.name) + (" *(saturated)*" if row.benchmark.saturated else ""),
                fmt_value(row.target, row.benchmark) + notes.mark(row.target),
                fmt_value(row.best_open, row.benchmark) + notes.mark(row.best_open),
                _cell(who),
                fmt_gap(row.gap, row.benchmark) + (SAME_SOURCE_MARK if row.same_source else "") + arrow,
            ]
        )
    return _table(
        [
            "Benchmark",
            f"{target.short}",
            "best_open",
            "who",
            "gap_to_fable_5_1",
        ],
        body,
    )


def _matrix_tables(bar: Bar, rows: list[GapRow], notes: Footnotes) -> list[str]:
    """The full model-by-benchmark matrix, split into readable column chunks."""
    bench_ids = [r.benchmark.id for r in rows]
    models = bar.models_with_scores(bench_ids)
    planned = [m for m in bar.models if m.status == "planned" and m not in models]
    models = models + planned
    chunks = [models[i : i + _MODELS_PER_TABLE] for i in range(0, len(models), _MODELS_PER_TABLE)]

    out: list[str] = []
    for n, chunk in enumerate(chunks, start=1):
        header = ["Benchmark"]
        if n == 1:
            header += ["best_open", "gap_to_fable_5_1"]
        header += [m.short for m in chunk]
        body = []
        for row in rows:
            cells = [_cell(row.benchmark.name)]
            if n == 1:
                cells.append(fmt_value(row.best_open, row.benchmark) + notes.mark(row.best_open))
                cells.append(
                    fmt_gap(row.gap, row.benchmark) + (SAME_SOURCE_MARK if row.same_source else "")
                )
            for model in chunk:
                score = bar.score(model.id, row.benchmark.id)
                cells.append(fmt_value(score, row.benchmark) + notes.mark(score))
            body.append(cells)
        label = "" if len(chunks) == 1 else f" (columns {n} of {len(chunks)})"
        out.append(f"#### Full matrix{label}\n")
        out.append(_table(header, body))
        out.append("")
    return out


def render_markdown(bar: Bar, *, generated: str | None = None) -> str:
    """Render the whole gap tracker as markdown."""
    generated = generated or datetime.now(UTC).date().isoformat()
    target = bar.model(bar.target_id)
    notes = Footnotes()
    parts: list[str] = []

    parts.append(f"# The gap to {target.name}\n")
    parts.append(
        f"*Generated by `python -m r52.bar` on {generated}. Published numbers retrieved "
        f"{bar.meta.get('retrieved', 'n/d')}; every one carries a source URL and its eval "
        "conditions in `r52/bar/bar.yaml`.*\n"
    )
    parts.append(
        f"`{DASH}` means **no published number exists** \u2014 never 0. "
        f"`{UNVERIFIED_MARK}` marks a number our research flagged UNVERIFIED. "
        f"`{SAME_SOURCE_MARK}` marks a gap where both sides come from the **same evaluator**, "
        "which is the only kind of gap worth trusting. Superscripts are footnotes "
        "giving the eval conditions; the legend is at the bottom.\n"
    )
    if bar.meta.get("health_warning"):
        parts.append("> **Read this first.** " + " ".join(bar.meta["health_warning"].split()) + "\n")

    for category, title in GROUPS:
        rows = gap_rows(bar, category)
        if not rows:
            continue
        parts.append(f"## {title}\n")
        if category == "classic-local":
            parts.append(
                "These are **dead at the frontier** \u2014 research/01 §1.4: not one of the nine "
                "September-2026 frontier launches reports SWE-bench Verified, MMLU, MMLU-Pro, "
                "AIME, LiveCodeBench, HumanEval or MATH-500. They are here because they are how "
                "*our* models get measured: a 124M model trained on a Mac Mini is nowhere near "
                "saturating any of them, and they run locally with no API key. Rows that are "
                f"all `{DASH}` across the frontier columns are not gaps in this file \u2014 they "
                "are the honest state of the world.\n"
            )
        parts.append("### Headline: Fable 5.1 vs the best open weights\n")
        parts.append(_headline_table(bar, rows, notes))
        parts.append("")
        parts.extend(_matrix_tables(bar, rows, notes))

    planned = [m for m in bar.models if m.status == "planned" and m.targets]
    if planned:
        parts.append("## Our runs: targets, not results\n")
        parts.append(
            "Nothing has been measured yet, so every cell above for these models is an em dash. "
            "These are the acceptance criteria; `results/<run>/eval.json` fills the cells in.\n"
        )
        rows = []
        for model in planned:
            for key, spec in model.targets.items():
                if not isinstance(spec, dict):
                    rows.append([_cell(model.short), _cell(key), DASH, _cell(spec)])
                    continue
                value = spec.get("value")
                rows.append(
                    [
                        _cell(model.short),
                        _cell(key),
                        f"{_cell(spec.get('direction', ''))} {value} {_cell(spec.get('unit', ''))}".strip()
                        if value is not None
                        else DASH,
                        _cell(spec.get("source")),
                    ]
                )
        parts.append(_table(["Model", "Benchmark", "Target", "Where the target comes from"], rows))
        parts.append("")

    scorecard = bar.saturation_scorecard
    if scorecard:
        parts.append("## Saturation scorecard\n")
        parts.append(
            "From research/01 §7. The left column is where effort is wasted; the right column is "
            "where the headroom actually is.\n"
        )
        columns = [
            ("saturated", "Saturated / dead"),
            ("near_saturated", "Near-saturated"),
            ("real_headroom", "Real headroom"),
        ]
        lists = [scorecard.get(key) or [] for key, _ in columns]
        height = max((len(items) for items in lists), default=0)
        body = []
        for i in range(height):
            row = []
            for items in lists:
                if i < len(items):
                    item = items[i]
                    row.append(f"**{_cell(item.get('label'))}** \u2014 {_cell(item.get('detail'))}")
                else:
                    row.append("")
            body.append(row)
        parts.append(_table([label for _, label in columns], body))
        parts.append("")

    parts.append("## Benchmarks in this table\n")
    bench_rows = [
        [
            _cell(b.name),
            _cell(b.category),
            _cell(b.measures),
            "yes" if b.saturated else "no",
            "yes" if b.local_runnable else "no",
            f"[link]({b.url})",
        ]
        for b in bar.benchmarks
    ]
    parts.append(
        _table(["Benchmark", "Group", "What it measures", "Saturated?", "Runs locally?", "URL"], bench_rows)
    )
    parts.append("")

    parts.append("## Models in this table\n")
    model_rows = [
        [
            _cell(m.short),
            _cell(m.name),
            _cell(m.org),
            _cell(m.weights),
            _cell(m.license),
            _cell(m.released),
            _cell(m.status),
        ]
        for m in bar.models
    ]
    parts.append(
        _table(["Short", "Model", "Org", "Weights", "Licence", "Released", "Status"], model_rows)
    )
    parts.append("")

    parts.append("## Legend \u2014 eval conditions\n")
    parts.append(
        "research/01 §0: *\"Every headline number in this report is a model + harness + effort "
        "level + grader tuple. Treat single numbers as meaningless without that tuple.\"* Each "
        "superscript below is that tuple.\n"
    )
    parts.extend(notes.render())
    parts.append("")
    return "\n".join(parts)


def render_json(bar: Bar, *, generated: str | None = None) -> dict[str, Any]:
    """Render the whole gap tracker as a JSON-serialisable dict."""

    def score_dict(score: Score | None) -> dict[str, Any] | None:
        if score is None:
            return None  # never 0
        return {
            "model": score.model,
            "value": score.value,
            "unit": score.unit,
            "source_url": score.source_url,
            "retrieved": score.retrieved,
            "conditions": score.conditions,
            "verified": score.verified,
            "measured_by_us": score.local,
            "commit": score.commit,
            "command": score.command,
        }

    groups: dict[str, Any] = {}
    for category, title in GROUPS:
        rows = []
        for row in gap_rows(bar, category):
            rows.append(
                {
                    "benchmark": row.benchmark.id,
                    "name": row.benchmark.name,
                    "unit": row.benchmark.unit,
                    "higher_is_better": row.benchmark.higher_is_better,
                    "saturated": row.benchmark.saturated,
                    "local_runnable": row.benchmark.local_runnable,
                    "url": row.benchmark.url,
                    "values": {
                        m.id: score_dict(bar.score(m.id, row.benchmark.id)) for m in bar.models
                    },
                    "best_open": score_dict(row.best_open),
                    "gap_to_fable_5_1": row.gap,
                    "gap_same_source": row.same_source,
                    "gap_target_entry": score_dict(row.gap_target),
                }
            )
        groups[category] = {"title": title, "rows": rows}

    return {
        "generated": generated or datetime.now(UTC).date().isoformat(),
        "target_model": bar.target_id,
        "meta": bar.meta,
        "conventions": {
            "missing": "null -- no published number exists. Never 0.",
            "gap_sign": "positive means the target model is ahead, whatever the benchmark's direction",
            "best_open": "best score among models with weights == open; models Epoch classifies "
            "as closed are excluded even where a secondary source calls them open",
        },
        "models": [
            {
                "id": m.id,
                "name": m.name,
                "org": m.org,
                "weights": m.weights,
                "license": m.license,
                "released": m.released,
                "status": m.status,
                "params_total": m.params_total,
                "params_active": m.params_active,
                "targets": m.targets or None,
            }
            for m in bar.models
        ],
        "groups": groups,
        "saturation_scorecard": bar.saturation_scorecard,
    }


# ======================================================================================
# CLI
# ======================================================================================


def main(argv: list[str] | None = None) -> int:
    """CLI entry point for ``python -m r52.bar``."""
    parser = argparse.ArgumentParser(
        prog="python -m r52.bar",
        description="Render the gap table (results/GAP.md and results/GAP.json) from bar.yaml "
        "plus any of our own results in results/*/eval.json.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=RESULTS_DIR,
        help="directory for GAP.md / GAP.json (default: <repo>/results)",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=RESULTS_DIR,
        help="directory scanned for */eval.json (default: <repo>/results)",
    )
    args = parser.parse_args(argv)

    bar = merge_local(load_bar(), load_local_results(args.results_dir))
    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    md_path = out_dir / "GAP.md"
    json_path = out_dir / "GAP.json"
    md_path.write_text(render_markdown(bar), encoding="utf-8")
    json_path.write_text(json.dumps(render_json(bar), indent=2) + "\n", encoding="utf-8")
    print(f"wrote {md_path}")
    print(f"wrote {json_path}")

    local = [s for s in bar.scores if s.local]
    print(f"{len(bar.benchmarks)} benchmarks, {len(bar.models)} models, {len(bar.scores)} scores "
          f"({len(local)} measured by us)")
    return 0
