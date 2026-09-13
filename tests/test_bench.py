# Copyright 2026 The road-to-52 authors.
# SPDX-License-Identifier: Apache-2.0
"""The throughput benchmark runs, reports sane numbers, and renders its tables."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from r52.bench import SIZES, BenchConfig, BenchRow, environment, run_one, to_markdown, write_results


def _tiny(**kw) -> BenchConfig:
    base = {"size": "30M", "seq": 32, "precision": "mixed", "micro_batch": 2, "seconds": 0.2,
            "warmup_steps": 1, "tokens_per_step": 64, "vocab_size": 256,
            "n_value_embeds": 1, "mem_limit_gib": 2.0}
    base.update(kw)
    return BenchConfig(**base)


@pytest.fixture
def tiny_arch():
    """Temporarily shrink the 30M preset so a bench row runs in well under a second."""
    orig = SIZES["30M"]
    SIZES["30M"] = {"n_layer": 2, "n_embd": 64, "n_head": 2}
    yield
    SIZES["30M"] = orig


def test_run_one_tiny(tiny_arch) -> None:
    row = run_one(_tiny())
    assert row.status == "ok"
    assert row.tok_s > 0 and row.tflops > 0
    # A 2-layer/64-dim model rounds to ~0 TFLOPS, so only assert the invariants.
    assert 0 <= row.mfu_theoretical < 1 and row.mfu_measured >= row.mfu_theoretical
    assert row.peak_mem_gib > 0
    assert row.grad_accum == 1 and row.micro_batch == 2
    assert row.flops_per_token == 6 * row.params_flops + 12 * row.n_layer * row.n_embd * row.seq
    assert row.optimizer_steps >= 1


def test_impossible_config_is_skipped_not_fatal(tiny_arch) -> None:
    """A configuration that cannot fit is recorded as `skipped`, never raised."""
    row = run_one(_tiny(mem_limit_gib=1e-6, micro_batch=0))
    assert row.status == "skipped" and "GiB" in row.note
    assert row.tok_s is None


def test_markdown_and_json(tmp_path: Path) -> None:
    rows = [
        BenchRow(size="124M", seq=1024, precision="mixed", micro_batch=4, grad_accum=16,
                 compile=True, grad_checkpoint=False, n_layer=12, n_embd=768, n_head=6,
                 params_total=200_835_090, params_non_embedding=84_934_674,
                 params_flops=123_568_146, flops_per_token=854_753_100, tok_s=3000.0,
                 tflops=2.5, mfu_theoretical=0.587, mfu_measured=0.694, peak_mem_gib=3.9,
                 optimizer_steps=3, seconds=25.0),
        BenchRow(size="350M", seq=1024, precision="fp32", micro_batch=0, grad_accum=0,
                 compile=True, grad_checkpoint=False, status="skipped", note="does not fit"),
    ]
    jp, mp = write_results(rows, tmp_path / "bench")
    data = json.loads(jp.read_text())
    assert data["rows"][0]["tok_s"] == 3000.0
    for key in ("chip", "macos", "mlx", "git_commit", "date", "peak_tflops_theoretical"):
        assert key in data["environment"], key
    md = mp.read_text()
    assert "| 124M |" in md and "58.7%" in md and "69.4%" in md
    assert "does not fit" in md
    assert to_markdown(rows, environment()).startswith("# MLX pretraining throughput benchmark")


def test_size_presets_are_the_documented_four() -> None:
    assert list(SIZES) == ["30M", "60M", "124M", "350M"]
    assert SIZES["124M"] == {"n_layer": 12, "n_embd": 768, "n_head": 6}


def test_append_merges_and_replaces(tmp_path: Path) -> None:
    from r52.bench import read_results
    from r52.bench import write_results as _w

    a = BenchRow(size="30M", seq=1024, precision="mixed", micro_batch=4, grad_accum=16,
                 compile=True, grad_checkpoint=False, tok_s=1.0)
    b = BenchRow(size="124M", seq=1024, precision="mixed", micro_batch=2, grad_accum=32,
                 compile=True, grad_checkpoint=False, tok_s=2.0)
    a2 = BenchRow(size="30M", seq=1024, precision="mixed", micro_batch=4, grad_accum=16,
                  compile=True, grad_checkpoint=False, tok_s=9.0)
    _w([a], tmp_path / "b")
    _w([b, a2], tmp_path / "b", append=True)
    rows = read_results(tmp_path / "b")
    assert [(r.size, r.tok_s) for r in rows] == [("30M", 9.0), ("124M", 2.0)]
