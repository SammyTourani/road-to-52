# Copyright 2026 The road-to-52 authors.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the bar and the gap tracker.

The honest-numbers rule (docs/ARCHITECTURE.md §1.3) is enforced here mechanically: every score
must carry a source URL, a retrieval date and its eval conditions, and a benchmark with no
published number must render as an em dash and serialise as ``null`` -- never as 0.

No network calls: every published number lives in ``bar.yaml``.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest
import yaml

from r52.bar import gap as G

REPO_ROOT = Path(__file__).resolve().parents[1]
BAR_YAML = REPO_ROOT / "r52" / "bar" / "bar.yaml"


@pytest.fixture(scope="module")
def bar() -> G.Bar:
    return G.load_bar()


@pytest.fixture(scope="module")
def markdown(bar: G.Bar) -> str:
    return G.render_markdown(bar, generated="2026-09-13")


@pytest.fixture(scope="module")
def payload(bar: G.Bar) -> dict:
    return G.render_json(bar, generated="2026-09-13")


# ======================================================================================
# The YAML validates
# ======================================================================================


def test_bar_yaml_loads(bar: G.Bar) -> None:
    assert bar.benchmarks
    assert bar.models
    assert bar.scores
    assert bar.target_id == "claude-fable-5-1"


def test_every_score_carries_its_provenance(bar: G.Bar) -> None:
    """source_url + retrieved + conditions + unit on every single number."""
    for score in bar.scores:
        where = f"{score.model}/{score.benchmark}"
        assert score.source_url, f"{where}: no source_url"
        assert score.source_url.startswith("http"), f"{where}: source_url is not a URL"
        assert score.retrieved, f"{where}: no retrieval date"
        assert re.fullmatch(r"\d{4}-\d{2}-\d{2}", score.retrieved), f"{where}: bad date"
        assert score.conditions and len(score.conditions) > 20, f"{where}: eval conditions too thin"
        assert score.unit, f"{where}: no unit"
        assert isinstance(score.value, float)


def test_every_score_points_at_a_known_benchmark_and_model(bar: G.Bar) -> None:
    bench_ids = {b.id for b in bar.benchmarks}
    model_ids = {m.id for m in bar.models}
    for score in bar.scores:
        assert score.benchmark in bench_ids
        assert score.model in model_ids


def test_every_benchmark_is_fully_described(bar: G.Bar) -> None:
    for bench in bar.benchmarks:
        assert bench.name and bench.measures and bench.unit
        assert bench.url.startswith("http"), f"{bench.id}: url is not a URL"
        assert bench.category in {"frontier-2026", "classic-local"}
        assert isinstance(bench.saturated, bool)
        assert isinstance(bench.local_runnable, bool)


def test_no_fabricated_source_urls(bar: G.Bar) -> None:
    """Guard against inventing a repo URL for something that has no public source."""
    for score in bar.scores:
        assert "road-to-52" not in (score.source_url or "")
    raw = yaml.safe_load(BAR_YAML.read_text(encoding="utf-8"))
    assert raw["saturation_scorecard"]["source_url"] is None
    assert raw["saturation_scorecard"]["source"].startswith("research/01")


def test_the_classic_local_set_is_present(bar: G.Bar) -> None:
    """These are dead at the frontier but are how our small models get measured."""
    required = {
        "hellaswag",
        "arc-challenge",
        "mmlu-pro",
        "gpqa-diamond",
        "gsm8k",
        "math-500",
        "humaneval",
        "ifeval",
        "dclm-core",
        "fineweb-val-loss",
    }
    classics = {b.id for b in bar.benchmarks if b.category == "classic-local"}
    assert required <= classics, f"missing classics: {sorted(required - classics)}"
    for bid in required:
        assert bar.benchmark(bid).local_runnable, f"{bid} should be locally runnable"


def test_dead_at_the_frontier_is_said_out_loud(bar: G.Bar) -> None:
    for bid in ("mmlu-pro", "gsm8k", "math-500", "humaneval", "hellaswag", "arc-challenge"):
        notes = (bar.benchmark(bid).notes or "").upper()
        assert "DEAD AT THE FRONTIER" in notes, f"{bid} does not say it is dead at the frontier"


def test_the_2026_consensus_set_is_present(bar: G.Bar) -> None:
    required = {
        "terminal-bench-4",
        "deepswe-v1-1",
        "hle-no-tools",
        "hle-with-tools",
        "automationbench",
        "gdpval-aa-v2",
        "osworld-2-partial",
        "osworld-2-strict",
        "agents-last-exam",
        "arc-agi-1",
        "arc-agi-2",
        "arc-agi-3",
        "swe-bench-pro",
        "terminal-bench-science-0-1",
        "aa-briefcase",
        "healthbench-professional",
        "benchcad",
        "frontiercode-1-1-main",
    }
    have = {b.id for b in bar.benchmarks if b.category == "frontier-2026"}
    assert required <= have, f"missing consensus benchmarks: {sorted(required - have)}"


def test_required_models_are_present(bar: G.Bar) -> None:
    required = {
        "claude-fable-5-1",
        "claude-mythos-5-1",
        "claude-fable-5",
        "claude-opus-5",
        "claude-sonnet-5",
        "gpt-6-astra",
        "gpt-5-6-sol",
        "kimi-k3",
        "glm-5-3",
        "deepseek-v4-1-flash",
        "qwen3-8-max",
        "minimax-m3",
        "gpt-oss-120b",
        "gpt2-124m",
        "r52-nano-30m",
        "r52-gpt2-124m-mac",
    }
    have = {m.id for m in bar.models}
    assert required <= have, f"missing models: {sorted(required - have)}"


def test_our_runs_are_planned_and_have_no_scores(bar: G.Bar) -> None:
    for mid in ("r52-nano-30m", "r52-gpt2-124m-mac"):
        model = bar.model(mid)
        assert model.status == "planned"
        assert not [s for s in bar.scores if s.model == mid], (
            f"{mid} has a score but nothing has been measured yet"
        )
    # ...but the 124M run does carry its acceptance criteria
    targets = bar.model("r52-gpt2-124m-mac").targets
    assert targets["fineweb-val-loss"]["value"] == 3.28
    assert targets["hellaswag"]["value"] == 29.4
    assert targets["dclm-core"]["value"] is None
    assert "TO BE MEASURED BY US" in targets["dclm-core"]["source"]


def test_gpt2_reference_numbers(bar: G.Bar) -> None:
    """research/01 + ARCHITECTURE §6: GPT-2 124M HellaSwag, GPT-2 XL CORE."""
    hs = {s.value for s in bar.all_scores("gpt2-124m", "hellaswag")}
    assert hs == {29.4, 31.1}
    for score in bar.all_scores("gpt2-124m", "hellaswag"):
        assert score.verified is False, "the llm.c reference has not been re-measured by us yet"
    core = bar.score("gpt2-xl-1558m", "dclm-core")
    assert core is not None and core.value == pytest.approx(0.256525)


def test_fable_5_1_has_no_swe_bench_verified_number(bar: G.Bar) -> None:
    """research/01 §2.5: it does not exist. Every '95%' claim is Fable 5's June figure."""
    assert bar.score("claude-fable-5-1", "swe-bench-verified") is None
    fable5 = bar.score("claude-fable-5", "swe-bench-verified")
    assert fable5 is not None and fable5.value == 95.0


def test_benchmarks_fable_5_1_does_not_report_are_absent(bar: G.Bar) -> None:
    """Absences confirmed by full-text search in research/01 §2.5 must stay absent."""
    for bid in ("arc-agi-3", "browsecomp", "agents-last-exam", "vending-bench-2"):
        assert bar.score("claude-fable-5-1", bid) is None, (
            f"Fable 5.1 has no published {bid} number; one must not appear here"
        )


def test_unverified_numbers_are_flagged(bar: G.Bar) -> None:
    unverified = [s for s in bar.scores if not s.verified]
    assert unverified, "research flagged some numbers UNVERIFIED; the flags must survive"
    for score in unverified:
        assert score.model == "gpt2-124m"


# ======================================================================================
# Gap computation
# ======================================================================================


def test_best_open_excludes_closed_weights(bar: G.Bar) -> None:
    """Epoch classifies Qwen3.8-Max as closed weights; it cannot hold the best-open column."""
    assert bar.model("qwen3-8-max").weights == "closed"
    for bench in bar.benchmarks:
        best = bar.best_open(bench.id)
        if best is not None:
            assert best.model != "qwen3-8-max"
            assert bar.model(best.model).is_open


def test_best_open_prefers_the_independent_harness(bar: G.Bar) -> None:
    """DeepSeek self-reports 90.6 on Terminal-Bench 2.1; Vals measures 74.53. Use Vals."""
    best = bar.best_open("terminal-bench-2-1")
    assert best is not None
    assert best.value == pytest.approx(74.53)
    assert best.model == "deepseek-v4-1-flash"


def test_gap_matches_the_research_head_to_head(bar: G.Bar) -> None:
    """Reproduce research/01 §3.2's Fable-5.1-vs-best-open table."""
    expected = {
        "terminal-bench-4": (41.8, 16.1, "glm-5-3"),
        "terminal-bench-2-1": (74.53, 10.5, "deepseek-v4-1-flash"),
        "hle-with-tools": (63.9, 1.1, "deepseek-v4-1-flash"),
        "hle-no-tools": (43.5, 17.4, "kimi-k3"),
        "gpqa-diamond": (93.5, 0.2, "kimi-k3"),
        "arc-agi-1": (94.5, 3.0, "kimi-k3"),
        "arc-agi-2": (61.4, 28.6, "deepseek-v4-flash-0731"),
        "livebench-overall": (81.1, 2.3, "deepseek-v4-1-flash"),
        "livebench-agentic-coding": (77.3, -11.2, "deepseek-v4-1-flash"),
        "aa-index-v4-3": (45.0, 8.0, "glm-5-3"),
        "eci": (157.6, 6.6, "kimi-k3"),
        "arena-agent-arena": (6.46, 7.39, "kimi-k3"),
        "mcp-atlas": (82.3, 4.9, "kimi-k3"),
        "gdpval-aa-v2": (1686.0, 78.0, "kimi-k3"),
        "osworld-2-strict": (4.6, 37.1, "minimax-m3"),
    }
    rows = {r.benchmark.id: r for r in G.gap_rows(bar, "frontier-2026")}
    rows.update({r.benchmark.id: r for r in G.gap_rows(bar, "classic-local")})
    for bid, (open_value, gap, who) in expected.items():
        row = rows[bid]
        assert row.best_open is not None, bid
        assert row.best_open.value == pytest.approx(open_value, abs=0.01), bid
        assert row.best_open.model == who, bid
        assert row.gap == pytest.approx(gap, abs=0.05), bid


def test_open_weights_already_beat_fable_on_two_benchmarks(bar: G.Bar) -> None:
    """research/04's one-line answer: it has already happened. A negative gap means open leads."""
    rows = G.gap_rows(bar, "frontier-2026") + G.gap_rows(bar, "classic-local")
    leading = {r.benchmark.id for r in rows if r.gap is not None and r.gap < 0}
    assert "livebench-agentic-coding" in leading
    assert "automationbench" in leading


def test_gap_is_none_when_either_side_is_missing(bar: G.Bar) -> None:
    rows = {r.benchmark.id: r for r in G.gap_rows(bar, "frontier-2026")}
    # Fable 5.1 has no BrowseComp number, so there is no gap -- not a zero gap
    assert rows["browsecomp"].target is None
    assert rows["browsecomp"].gap is None
    # nobody open reports SWE-bench Multilingual
    assert rows["swe-bench-multilingual"].best_open is None
    assert rows["swe-bench-multilingual"].gap is None


def test_same_evaluator_gaps_are_marked(bar: G.Bar) -> None:
    rows = {r.benchmark.id: r for r in G.gap_rows(bar, "frontier-2026")}
    assert rows["terminal-bench-4"].same_source, "both sides come from the official tbench board"
    assert rows["arc-agi-2"].same_source, "both sides come from ARC Prize's own runs"
    assert not rows["hle-no-tools"].same_source, "Anthropic's card vs Moonshot's model card"


def test_lower_is_better_benchmarks_are_handled(bar: G.Bar) -> None:
    assert bar.benchmark("fineweb-val-loss").higher_is_better is False
    assert bar.benchmark("scale-fortress").higher_is_better is False
    best = bar.best_open("scale-fortress")
    assert best is not None and best.model == "gpt-oss-120b"


# ======================================================================================
# Rendering
# ======================================================================================


def test_markdown_renders_both_groups(markdown: str) -> None:
    assert "# The gap to Claude Fable 5.1" in markdown
    assert "## 2026 frontier set" in markdown
    assert "## Classic local set" in markdown
    assert "gap_to_fable_5_1" in markdown
    assert "best_open" in markdown
    assert "## Saturation scorecard" in markdown
    assert "## Legend" in markdown


def test_markdown_uses_a_dash_for_missing_values_and_never_a_zero(markdown: str) -> None:
    assert G.DASH in markdown
    # ARC-Challenge has no published score from anyone: the whole row must be dashes
    row = next(line for line in markdown.splitlines() if line.startswith("| ARC-Challenge |"))
    cells = [c.strip() for c in row.strip("|").split("|")[1:]]
    assert cells, "ARC-Challenge row has no value cells"
    assert all(c == G.DASH for c in cells), f"ARC-Challenge row is not all dashes: {cells}"
    assert "| 0 |" not in markdown, "a bare 0 cell almost certainly means 'missing'"


def test_markdown_has_a_footnote_for_every_marker(markdown: str) -> None:
    used = {int(m) for m in re.findall(r"[⁰¹²³⁴-⁹]+", markdown)
            for m in [m.translate(str.maketrans("⁰¹²³⁴⁵⁶⁷⁸⁹", "0123456789"))]}
    legend = markdown.split("## Legend")[1]
    defined = {int(m) for m in re.findall(r"^(\d+)\. ", legend, flags=re.MULTILINE)}
    assert used, "no footnote markers were rendered"
    assert used <= defined, f"markers with no legend entry: {sorted(used - defined)}"


def test_saturation_scorecard_is_rendered(markdown: str) -> None:
    assert "Saturated / dead" in markdown
    assert "Real headroom" in markdown
    assert "SWE-bench Verified" in markdown
    assert "OSWorld 2.0" in markdown


def test_json_uses_null_not_zero_for_missing(payload: dict) -> None:
    rows = payload["groups"]["classic-local"]["rows"]
    arc = next(r for r in rows if r["benchmark"] == "arc-challenge")
    assert all(v is None for v in arc["values"].values())
    assert arc["best_open"] is None
    assert arc["gap_to_fable_5_1"] is None
    frontier = payload["groups"]["frontier-2026"]["rows"]
    browse = next(r for r in frontier if r["benchmark"] == "browsecomp")
    assert browse["values"]["claude-fable-5-1"] is None
    assert browse["gap_to_fable_5_1"] is None


def test_json_carries_conditions_and_sources(payload: dict) -> None:
    rows = payload["groups"]["frontier-2026"]["rows"]
    tb4 = next(r for r in rows if r["benchmark"] == "terminal-bench-4")
    cell = tb4["values"]["claude-fable-5-1"]
    assert cell["source_url"].startswith("http")
    assert cell["conditions"]
    assert cell["retrieved"] == "2026-09-13"
    assert cell["measured_by_us"] is False


# ======================================================================================
# Our own results
# ======================================================================================


def test_local_results_are_ingested(tmp_path: Path, bar: G.Bar) -> None:
    run_dir = tmp_path / "gpt2_124m_mac"
    run_dir.mkdir()
    (run_dir / "eval.json").write_text(
        json.dumps(
            {
                "model": "r52-gpt2-124m-mac",
                "benchmark": "hellaswag",
                "value": 30.2,
                "unit": "% acc_norm",
                "conditions": "10042 val examples, GPT-2 tokenizer, block_size 1024",
                "commit": "abc1234",
                "command": "python -m r52.eval.hellaswag --ckpt runs/x/ckpt/best",
                "date": "2026-09-20",
            }
        ),
        encoding="utf-8",
    )
    local = G.load_local_results(tmp_path)
    assert len(local) == 1
    merged = G.merge_local(bar, local)
    cell = merged.score("r52-gpt2-124m-mac", "hellaswag")
    assert cell is not None
    assert cell.value == pytest.approx(30.2)
    assert cell.local is True
    assert "MEASURED BY US" in cell.conditions
    assert "abc1234" in cell.conditions
    assert "r52.eval.hellaswag" in cell.conditions
    # and it shows up in the rendered table
    assert "30.2" in G.render_markdown(merged, generated="2026-09-20")


def test_local_results_for_unknown_ids_are_dropped(tmp_path: Path, bar: G.Bar) -> None:
    run_dir = tmp_path / "bogus"
    run_dir.mkdir()
    (run_dir / "eval.json").write_text(
        json.dumps({"model": "nope", "benchmark": "hellaswag", "value": 99.0}), encoding="utf-8"
    )
    merged = G.merge_local(bar, G.load_local_results(tmp_path))
    assert len(merged.scores) == len(bar.scores)


def test_missing_results_dir_is_not_an_error(tmp_path: Path) -> None:
    assert G.load_local_results(tmp_path / "does-not-exist") == []


# ======================================================================================
# CLI
# ======================================================================================


def test_cli_renders_both_files(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert G.main(["--out-dir", str(tmp_path), "--results-dir", str(tmp_path)]) == 0
    md = tmp_path / "GAP.md"
    js = tmp_path / "GAP.json"
    assert md.is_file() and js.is_file()
    assert "gap_to_fable_5_1" in md.read_text(encoding="utf-8")
    data = json.loads(js.read_text(encoding="utf-8"))
    assert data["target_model"] == "claude-fable-5-1"
    assert set(data["groups"]) == {"frontier-2026", "classic-local"}
    assert "conditions" not in capsys.readouterr().err


def test_committed_results_are_up_to_date(bar: G.Bar) -> None:
    committed = REPO_ROOT / "results" / "GAP.json"
    if not committed.is_file():
        pytest.skip("results/GAP.json not rendered yet")
    payload = json.loads(committed.read_text(encoding="utf-8"))
    # render exactly as `python -m r52.bar` does: bar.yaml merged with the committed results/
    merged = G.merge_local(G.load_bar(), G.load_local_results())
    fresh = G.render_json(merged, generated=payload["generated"])
    assert payload["groups"] == fresh["groups"]


# ======================================================================================
# No network
# ======================================================================================


def test_no_network_imports() -> None:
    for name in ("gap.py", "__init__.py", "__main__.py"):
        src = (REPO_ROOT / "r52" / "bar" / name).read_text(encoding="utf-8")
        for banned in ("import requests", "import httpx", "urllib.request", "import socket"):
            assert banned not in src, f"{name} appears to make network calls ({banned})"
