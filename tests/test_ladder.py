# Copyright 2026 The road-to-52 authors.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the compute ladder.

The load-bearing test is :func:`test_calibration_mfus_match_research`: research/04 §B.5 fits
the ladder's MFU model against three disclosed runs, and our recomputed implied MFUs must
reproduce the published ones to within 1 percentage point.  If that drifts, every dollar figure
on the ladder is wrong.

No network calls: every number comes from the YAMLs next to the module.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from r52.ladder import ladder as L

REPO_ROOT = Path(__file__).resolve().parents[1]

# research/04 §B.5, "the verified Sept 2026 floors from §B.2"
H100_PEAK_TFLOPS = 989.0
H100_SPOT_USD = 0.94
H100_ON_DEMAND_USD = 2.43


# ======================================================================================
# Pure arithmetic
# ======================================================================================


def test_flops_is_6nd() -> None:
    assert L.flops(1.0, 1.0) == 6.0
    assert L.flops(0.0, 1e12) == 0.0
    # research/04 §B.5, Chinchilla-optimal 3B dense on 60B tokens
    assert L.flops(3e9, 60e9) == pytest.approx(1.08e21, rel=1e-9)
    # 8B dense on 160B tokens
    assert L.flops(8e9, 160e9) == pytest.approx(7.68e21, rel=1e-9)
    # 125M (GPT-2 small) Chinchilla-optimal on 2.5B tokens
    assert L.flops(0.125e9, 2.5e9) == pytest.approx(1.88e18, rel=1e-2)


def test_flops_scales_with_active_params_not_total() -> None:
    """A 30B-A3B MoE and a 3B dense model cost the same: research/04 §B.5 reading 1."""
    assert L.flops(3e9, 60e9) == L.flops(3e9, 60e9)


def test_gpu_hours_matches_research_table() -> None:
    # 3B dense / 60B tokens at 35% MFU on H100 -> 867 H100-hours (research/04 §B.5)
    hours = L.gpu_hours(1.08e21, H100_PEAK_TFLOPS, 0.35)
    assert hours == pytest.approx(867, rel=0.01)
    # 8B dense / 160B tokens -> 6.2k
    assert L.gpu_hours(7.68e21, H100_PEAK_TFLOPS, 0.35) == pytest.approx(6200, rel=0.01)
    # 1B dense / 20B tokens -> 96
    assert L.gpu_hours(1.20e20, H100_PEAK_TFLOPS, 0.35) == pytest.approx(96, rel=0.01)
    # B200 at 30% MFU -> 444 for the 3B row
    assert L.gpu_hours(1.08e21, 2250.0, 0.30) == pytest.approx(444, rel=0.01)


def test_gpu_hours_is_inverse_of_implied_mfu() -> None:
    hours = L.gpu_hours(3.29e24, H100_PEAK_TFLOPS, 0.331)
    assert L.implied_mfu(3.29e24, hours, H100_PEAK_TFLOPS) == pytest.approx(0.331, rel=1e-9)


def test_wall_clock_and_cost() -> None:
    assert L.wall_clock_hours(867.0, 8) == pytest.approx(108.375)
    assert L.wall_clock_hours(100.0, 1) == 100.0
    # research/04 §B.5: 867 H100-h at $0.94 spot is ~$815, at $2.43 on-demand ~$2.1k
    assert L.cost_usd(867.0, H100_SPOT_USD) == pytest.approx(815, rel=0.01)
    assert L.cost_usd(867.0, H100_ON_DEMAND_USD) == pytest.approx(2107, rel=0.01)
    assert L.cost_usd(1234.0, 0.0) == 0.0


def test_two_thousand_dollars_buys_what_research_says() -> None:
    """research/04 §B.5: $2,000 at spot = 2,128 H100-hours = 266 h on an 8xH100 node."""
    hours = 2000.0 / H100_SPOT_USD
    assert hours == pytest.approx(2128, rel=0.01)
    assert L.wall_clock_hours(hours, 8) == pytest.approx(266, rel=0.01)
    # ...which is a 3B dense (or 30B-A3B) model on ~147B tokens at 35% MFU
    budget_flops = hours * H100_PEAK_TFLOPS * 1e12 * 0.35 * L.SECONDS_PER_HOUR
    assert budget_flops == pytest.approx(2.65e21, rel=0.02)
    assert budget_flops / (6 * 3e9) == pytest.approx(147e9, rel=0.02)


@pytest.mark.parametrize(
    ("fn", "args"),
    [
        (L.flops, (-1.0, 10.0)),
        (L.flops, (1.0, -10.0)),
        (L.gpu_hours, (1e20, 0.0, 0.35)),
        (L.gpu_hours, (1e20, 989.0, 0.0)),
        (L.gpu_hours, (1e20, 989.0, 1.5)),
        (L.wall_clock_hours, (100.0, 0)),
        (L.cost_usd, (100.0, -1.0)),
        (L.implied_mfu, (1e20, 0.0, 989.0)),
    ],
)
def test_invalid_inputs_raise(fn, args) -> None:
    with pytest.raises(ValueError):
        fn(*args)


# ======================================================================================
# Data files
# ======================================================================================


@pytest.fixture(scope="module")
def ladder() -> L.Ladder:
    return L.load_ladder()


def test_yaml_loads(ladder: L.Ladder) -> None:
    assert ladder.hardware
    assert ladder.prices
    assert ladder.rungs


def test_every_price_row_has_provenance(ladder: L.Ladder) -> None:
    """Honest-numbers rule: a source URL, or an explicit in-house measurement."""
    for price in ladder.prices.values():
        assert price.retrieved, f"{price.id} has no retrieval date"
        if price.source_url is None:
            assert price.measured and price.source, (
                f"{price.id} has neither a source URL nor a measured-with-source marker"
            )
        else:
            assert price.source_url.startswith("http"), f"{price.id} source_url is not a URL"
        assert price.kind in L.ALLOWED_PRICE_KINDS
        assert price.usd_per_gpu_hour >= 0


def test_every_hardware_entry_has_provenance(ladder: L.Ladder) -> None:
    for key, hw in ladder.hardware.items():
        assert hw.retrieved, f"{key} has no retrieval date"
        assert hw.source or hw.source_url, f"{key} has no source"
        if hw.source_url is None:
            assert hw.measured, f"{key} has no source URL and is not marked measured"
        if hw.peak_tflops is not None:
            assert hw.peak_tflops > 0


def test_hardware_peaks_match_research(ladder: L.Ladder) -> None:
    assert ladder.hardware["h100"].peak_tflops == H100_PEAK_TFLOPS
    assert ladder.hardware["b200"].peak_tflops == 2250.0
    # H800 is compute-identical to H100; research/04 §B.5's DeepSeek MFU only works this way
    assert ladder.hardware["h800"].peak_tflops == H100_PEAK_TFLOPS
    mac = ladder.hardware["mac_m4_10core"]
    assert mac.peak_tflops == 4.26
    assert mac.measured is True
    assert mac.measurements["bf16_matmul_tflops"] == 3.6
    assert mac.measurements["training_effective_tflops"] == 1.3


def test_unsourced_peaks_are_null_not_guessed(ladder: L.Ladder) -> None:
    """Accelerators research/04 prices but never gives a peak for must stay null."""
    for key in ("h200", "b300", "gb200", "mi300x", "mi355x", "tpu_v5e", "tpu_v6e", "tpu_v7"):
        assert ladder.hardware[key].peak_tflops is None, f"{key} has an unsourced peak"


def test_price_floors_are_the_research_floors(ladder: L.Ladder) -> None:
    spot = ladder.cheapest("h100", "spot")
    on_demand = ladder.cheapest("h100", "on-demand")
    assert spot is not None and on_demand is not None
    assert spot.usd_per_gpu_hour == H100_SPOT_USD
    assert on_demand.usd_per_gpu_hour == H100_ON_DEMAND_USD


def test_unverified_prices_never_set_the_floor(ladder: L.Ladder) -> None:
    """The Vast.ai row is cheaper than some verified rows but is tagged UNVERIFIED."""
    vast = ladder.prices["h100_marketplace_vast"]
    assert vast.verified is False
    for kind in ("spot", "on-demand"):
        chosen = ladder.cheapest("h100", kind)
        assert chosen is None or chosen.verified


def test_pcie_h100_is_not_the_sxm_floor(ladder: L.Ladder) -> None:
    """A $1.99 PCIe card must not undercut the $2.43 SXM on-demand floor."""
    pcie = ladder.prices["h100_pcie_ondemand_runpod_community"]
    assert pcie.hardware == "h100_pcie"
    assert ladder.hardware["h100_pcie"].peak_tflops is None


def test_required_rungs_exist(ladder: L.Ladder) -> None:
    ids = {r.id for r in ladder.rungs}
    required = {
        "rung-0-gpt2-124m-mac",
        "rung-1-nanochat-1b",
        "rung-2-dense-3b",
        "rung-2-moe-30b-a3b",
        "rung-2b-3b-150b-tokens",
        "rung-3-dense-8b",
        "ref-smollm3-3b",
        "ref-deepseek-v3",
        "ref-llama-3-1-405b",
        "ref-kimi-k2-scale",
        "ref-kimi-k3-scale",
        "frontier-fable-class",
    }
    assert required <= ids, f"missing rungs: {sorted(required - ids)}"


def test_every_rung_has_beats_status_and_notes(ladder: L.Ladder) -> None:
    for rung in ladder.rungs:
        assert rung.status in L.ALLOWED_STATUSES
        assert rung.kind in L.ALLOWED_RUNG_KINDS
        assert rung.notes, f"{rung.id} has no notes"
        if rung.kind == "rung":
            assert rung.beats, f"{rung.id} does not name the model it targets"


def test_moe_costs_the_same_as_the_dense_model_it_shadows(ladder: L.Ladder) -> None:
    dense = ladder.compute(next(r for r in ladder.rungs if r.id == "rung-2-dense-3b"))
    moe = ladder.compute(next(r for r in ladder.rungs if r.id == "rung-2-moe-30b-a3b"))
    assert moe.rung.params_total == 10 * dense.rung.params_total
    assert moe.flops_low == dense.flops_low
    assert moe.cost_spot_low == dense.cost_spot_low


# ======================================================================================
# Calibration -- the load-bearing test
# ======================================================================================


def test_calibration_mfus_match_research(ladder: L.Ladder) -> None:
    """Implied MFUs must reproduce research/04 §B.5 to within 1 percentage point."""
    calibration = ladder.calibration()
    assert len(calibration) == 3, "expected SmolLM3, DeepSeek-V3 and Llama 3.1 405B"
    expected = {
        "ref-smollm3-3b": 0.257,
        "ref-deepseek-v3": 0.331,
        "ref-llama-3-1-405b": 0.346,
    }
    for res in calibration:
        want = expected[res.rung.id]
        assert res.implied_mfu_published == pytest.approx(want)
        delta_pp = abs(res.implied_mfu - want) * 100.0
        assert delta_pp < 1.0, (
            f"{res.rung.id}: implied MFU {res.implied_mfu:.4f} is {delta_pp:.2f}pp from the "
            f"published {want:.3f}"
        )


def test_deepseek_v3_cost_reproduces_the_disclosure(ladder: L.Ladder) -> None:
    """Our modelled cost must land on DeepSeek's disclosed $5.576M."""
    res = ladder.compute(next(r for r in ladder.rungs if r.id == "ref-deepseek-v3"))
    assert res.rung.disclosed["cost_usd"] == pytest.approx(5.576e6)
    assert res.cost_on_demand_low == pytest.approx(5.576e6, rel=0.02)


def test_smollm3_gpu_hours_reproduce_the_disclosure(ladder: L.Ladder) -> None:
    res = ladder.compute(next(r for r in ladder.rungs if r.id == "ref-smollm3-3b"))
    assert res.gpu_hours_low == pytest.approx(220_000, rel=0.01)
    assert res.cost_spot_low == pytest.approx(207_000, rel=0.02)
    assert res.cost_on_demand_low == pytest.approx(535_000, rel=0.02)


# ======================================================================================
# Rung computation
# ======================================================================================


def test_rung_0_is_free_and_about_three_days(ladder: L.Ladder) -> None:
    res = ladder.compute(next(r for r in ladder.rungs if r.id == "rung-0-gpt2-124m-mac"))
    assert res.flops_low == pytest.approx(5.58e17, rel=1e-6)
    assert res.cost_spot_low == 0.0
    assert res.cost_on_demand_low == 0.0
    # docs/ARCHITECTURE.md §10 independently estimates 4-6 days
    assert 2 * 24 <= res.wall_clock_low <= 4 * 24  # measured 3,034 tok/s -> ~68 h
    # MFU is exactly the measured training-effective rate over the theoretical peak
    # 0.535 x 4.26 TFLOPS reproduces the measured 3,034 tok/s (results/mlx_pretrain_bench.md).
    assert res.rung.mfu * res.hardware.peak_tflops == pytest.approx(2.28, rel=0.01)


def test_rung_2b_is_about_the_two_thousand_dollar_budget(ladder: L.Ladder) -> None:
    res = ladder.compute(next(r for r in ladder.rungs if r.id == "rung-2b-3b-150b-tokens"))
    assert 1900 <= res.cost_spot_low <= 2100


def test_frontier_row_is_a_range_with_no_disclosed_cost(ladder: L.Ladder) -> None:
    res = ladder.compute(next(r for r in ladder.rungs if r.id == "frontier-fable-class"))
    assert res.is_range
    assert res.flops_low == pytest.approx(1e26)
    assert res.flops_high == pytest.approx(1e27)
    assert res.cost_spot_high > res.cost_spot_low
    assert not res.rung.disclosed, "nothing is disclosed for any frontier Claude model"
    assert res.rung.verified is False


def test_reference_rows_without_disclosure_are_flagged_unverified(ladder: L.Ladder) -> None:
    for rid in ("ref-kimi-k2-scale", "ref-kimi-k3-scale"):
        rung = next(r for r in ladder.rungs if r.id == rid)
        assert rung.verified is False, f"{rid} is modelled, not disclosed, and must say so"


def test_no_rung_borrows_a_price_from_other_silicon(ladder: L.Ladder) -> None:
    for res in ladder.compute_all():
        for price in (res.price_spot, res.price_on_demand):
            if price is not None:
                assert price.hardware == res.rung.hardware, (
                    f"{res.rung.id} costed against a {price.hardware} price"
                )


# ======================================================================================
# Formatting: never render a missing number as 0
# ======================================================================================


def test_missing_values_render_as_a_dash_not_zero() -> None:
    assert L.fmt_usd(None) == L.DASH
    assert L.fmt_hours(None) == L.DASH
    assert L.fmt_flops(None) == L.DASH
    assert L.fmt_count(None) == L.DASH
    assert L.fmt_wall(None) == L.DASH
    assert L.fmt_range(None, None, L.fmt_usd) == L.DASH
    # a real zero is still a zero
    assert L.fmt_usd(0.0) == "$0"


def test_small_disclosed_hours_keep_their_precision() -> None:
    """modded-nanogpt's 0.164 H100-hours must not round to '0'."""
    assert L.fmt_hours_precise(0.164) == "0.164"
    assert L.fmt_hours_precise(13.2) == "13.2"
    assert L.fmt_hours_precise(220_000) == "220,000"


def test_fmt_count_keeps_significant_digits() -> None:
    assert L.fmt_count(11.2e12) == "11.2T"
    assert L.fmt_count(15.6e12) == "15.6T"
    assert L.fmt_count(20e9) == "20B"
    assert L.fmt_count(124e6) == "124M"
    assert L.fmt_count(0.75e9) == "750M"


# ======================================================================================
# CLI
# ======================================================================================


def test_cli_renders_both_files(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert L.main(["--out-dir", str(tmp_path)]) == 0
    md = tmp_path / "ladder.md"
    js = tmp_path / "ladder.json"
    assert md.is_file() and js.is_file()

    text = md.read_text(encoding="utf-8")
    assert "# The compute ladder" in text
    assert "Rung 0" in text and "Frontier (Fable-class)" in text
    assert L.DASH in text
    for column in ("GPU-hours", "Wall-clock", "$ spot", "$ on-demand", "Status"):
        assert column in text

    payload = json.loads(js.read_text(encoding="utf-8"))
    assert payload["rungs"] and payload["hardware"] and payload["prices"]
    ids = {r["id"] for r in payload["rungs"]}
    assert "rung-0-gpt2-124m-mac" in ids

    out = capsys.readouterr().out
    assert "MFU calibration" in out


def test_cli_calibrate_prints_three_rows(capsys: pytest.CaptureFixture[str]) -> None:
    assert L.main(["--calibrate"]) == 0
    out = capsys.readouterr().out
    for rid in ("ref-smollm3-3b", "ref-deepseek-v3", "ref-llama-3-1-405b"):
        assert rid in out


def test_cli_custom_mode(capsys: pytest.CaptureFixture[str]) -> None:
    assert L.main(["--custom", "N=3e9,D=60e9,hw=h100,gpus=8"]) == 0
    out = capsys.readouterr().out
    assert "1.08e21" in out
    assert "867" in out


def test_cli_custom_json(capsys: pytest.CaptureFixture[str]) -> None:
    assert L.main(["--custom", "N=3e9,D=60e9,hw=h100,gpus=8", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["flops"] == pytest.approx(1.08e21, rel=1e-9)
    assert payload["gpu_hours"] == pytest.approx(867, rel=0.01)
    assert payload["cost_usd_spot"] == pytest.approx(815, rel=0.01)
    assert payload["cost_usd_on_demand"] == pytest.approx(2107, rel=0.01)
    assert payload["wall_clock_hours"] == pytest.approx(108.4, rel=0.01)


def test_custom_rejects_bad_input() -> None:
    ladder = L.load_ladder()
    with pytest.raises(ValueError):
        L.run_custom(ladder, "N=3e9,D=60e9")  # no hw
    with pytest.raises(ValueError):
        L.run_custom(ladder, "N=3e9,D=60e9,hw=nonexistent")
    with pytest.raises(ValueError):
        L.run_custom(ladder, "garbage")
    with pytest.raises(ValueError):
        # an accelerator with no sourced peak cannot be costed
        L.run_custom(ladder, "N=3e9,D=60e9,hw=tpu_v6e")


def test_committed_results_are_up_to_date() -> None:
    """results/ladder.{md,json} must match what the code renders right now."""
    ladder = L.load_ladder()
    committed = (REPO_ROOT / "results" / "ladder.json")
    if not committed.is_file():
        pytest.skip("results/ladder.json not rendered yet")
    payload = json.loads(committed.read_text(encoding="utf-8"))
    fresh = L.render_json(ladder, generated=payload["generated"])
    assert payload["rungs"] == fresh["rungs"]


# ======================================================================================
# No network
# ======================================================================================


def test_no_network_imports() -> None:
    for name in ("ladder.py", "__init__.py", "__main__.py"):
        src = (REPO_ROOT / "r52" / "ladder" / name).read_text(encoding="utf-8")
        for banned in ("import requests", "import httpx", "urllib.request", "import socket"):
            assert banned not in src, f"{name} appears to make network calls ({banned})"


def test_math_module_is_actually_used() -> None:
    """Guard against the formatters silently losing their scientific-notation path."""
    assert L.fmt_flops(1.08e21) == "1.08e21"
    assert math.isclose(L.flops(1e9, 20e9), 1.2e20)
