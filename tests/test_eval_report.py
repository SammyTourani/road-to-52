# Copyright 2026 The road-to-52 authors.
# SPDX-License-Identifier: Apache-2.0
"""The results record: the schema docs/ARCHITECTURE.md §1.3 demands, and gap.py's shape."""

from __future__ import annotations

import json
from pathlib import Path

from r52.bar.gap import load_bar, load_local_results, merge_local
from r52.eval.report import (
    EvalResult,
    append_results_md,
    conditions_string,
    git_commit,
    machine_string,
    make_result,
    write_result,
)

CONDITIONS = {
    "tokenizer": "gpt2 (tiktoken)",
    "block_size": 1024,
    "limit": None,
    "n_examples": 10042,
    "few_shot": 0,
    "command": "python -m r52.eval.hellaswag --model models/gpt2-mlx",
}


def _result(**kw) -> EvalResult:
    base = {
        "model": "gpt2-124m",
        "benchmark": "hellaswag",
        "value": 29.55,
        "unit": "% acc_norm",
        "conditions": dict(CONDITIONS),
        "wall_clock_s": 612.5,
        "metrics": {"acc": 0.2859, "acc_norm": 0.2955},
    }
    base.update(kw)
    return make_result(**base)


def test_result_json_carries_every_required_field(tmp_path: Path) -> None:
    md = tmp_path / "RESULTS.md"
    path = write_result(_result(), "gpt2-124m-reference", "hellaswag",
                        results_root=tmp_path, results_md=md)
    assert path == tmp_path / "gpt2-124m-reference" / "hellaswag.json"
    payload = json.loads(path.read_text())

    # docs/ARCHITECTURE.md §6 schema.
    for key in ("model", "benchmark", "value", "unit", "conditions", "commit", "date",
                "wall_clock_s", "machine"):
        assert key in payload, key
    for key in ("tokenizer", "block_size", "limit", "n_examples", "few_shot", "command"):
        assert key in payload["conditions"], key
    assert payload["value"] == 29.55
    assert payload["wall_clock_s"] == 612.5
    assert payload["metrics"]["acc"] == 0.2859
    assert payload["date"].count("-") == 2


def test_eval_json_matches_the_schema_gap_py_documents(tmp_path: Path) -> None:
    """gap.py's field names win: flat `conditions` string, `command` at the top level."""
    md = tmp_path / "RESULTS.md"
    write_result(_result(), "gpt2-124m-reference", "hellaswag", results_root=tmp_path, results_md=md)
    write_result(
        _result(benchmark="fineweb-val-loss", value=3.3, unit="nats/token"),
        "gpt2-124m-reference", "val_loss", results_root=tmp_path, results_md=md,
    )
    entries = json.loads((tmp_path / "gpt2-124m-reference" / "eval.json").read_text())
    assert isinstance(entries, list) and len(entries) == 2
    for e in entries:
        assert set(e) == {"model", "benchmark", "value", "unit", "conditions", "commit",
                          "command", "date"}
        assert isinstance(e["conditions"], str)

    scores = load_local_results(tmp_path)
    assert {s.benchmark for s in scores} == {"hellaswag", "fineweb-val-loss"}
    assert all(s.local and s.primary for s in scores)
    assert all("MEASURED BY US." in s.conditions for s in scores)

    # The ids must be real, or the gap table silently drops the rows.
    bar = merge_local(load_bar(), scores)
    cell = bar.score("gpt2-124m", "hellaswag")
    assert cell is not None and cell.local and cell.value == 29.55


def test_rewriting_a_benchmark_replaces_its_row(tmp_path: Path) -> None:
    md = tmp_path / "RESULTS.md"
    for value in (29.55, 31.17):
        write_result(_result(value=value), "r", "hellaswag", results_root=tmp_path, results_md=md)
    entries = json.loads((tmp_path / "r" / "eval.json").read_text())
    assert [e["value"] for e in entries] == [31.17]


def test_results_md_row_is_appended_with_its_reproduction_path(tmp_path: Path) -> None:
    md = tmp_path / "RESULTS.md"
    append_results_md(_result(), "gpt2-124m-reference", "hellaswag", path=md)
    append_results_md(_result(value=3.3, benchmark="fineweb-val-loss", unit="nats/token"),
                      "gpt2-124m-reference", "val_loss", path=md)
    lines = [ln for ln in md.read_text().splitlines() if ln.startswith("| 2")]
    assert len(lines) == 2
    assert "29.55 % acc_norm" in lines[0]
    assert "3.3000 nats/token" in lines[1]
    for line in lines:
        assert "python -m r52.eval." in line          # the command
        assert "wall-clock" in line                    # the wall clock
        assert "tokenizer gpt2 (tiktoken)" in line     # the tokenizer
        assert "block_size 1024" in line               # the sequence length


def test_conditions_string_orders_and_skips_empties() -> None:
    s = conditions_string(CONDITIONS)
    assert s.startswith("examples 10042, few-shot 0, tokenizer gpt2 (tiktoken), block_size 1024")
    assert "limit" not in s  # None is dropped, never rendered as 0
    assert "command" not in s


def test_commit_and_machine_are_recorded() -> None:
    assert git_commit()  # "unknown" outside a repo, never empty
    machine = machine_string()
    assert "MLX" in machine
