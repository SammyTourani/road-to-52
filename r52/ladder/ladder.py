# Copyright 2026 The road-to-52 authors.
# SPDX-License-Identifier: Apache-2.0
"""The compute ladder: 6ND FLOPs -> GPU-hours -> wall-clock -> dollars.

Units are explicit everywhere and stated in every signature:

* ``params`` / ``tokens``  -- counts (dimensionless), e.g. ``3e9`` params, ``60e9`` tokens
* ``flops``                -- total floating-point operations for the training run
* ``peak_tflops``          -- accelerator peak, **TFLOP/s** (1e12 FLOP/s), dense BF16 unless noted
* ``mfu``                  -- model FLOPs utilisation, a fraction in ``(0, 1]``
* ``gpu_hours``            -- accelerator-hours (one accelerator busy for one hour)
* ``wall_clock_hours``     -- hours of calendar time at ``n_gpus`` accelerators
* ``usd``                  -- US dollars

Method and calibration come from ``research/04-landscape-and-compute.md`` §B.5, which fits MFU
against three disclosed runs (SmolLM3 25.7%, DeepSeek-V3 33.1%, Llama 3.1 405B 34.6%).  Data
lives in ``prices.yaml`` and ``rungs.yaml`` next to this file; there are **no network calls**
here or in the tests.

CLI::

    python -m r52.ladder                                    # render results/ladder.{md,json}
    python -m r52.ladder --calibrate                        # print the three implied MFUs
    python -m r52.ladder --custom "N=3e9,D=60e9,hw=h100,gpus=8"
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

__all__ = [
    "ALLOWED_PRICE_KINDS",
    "ALLOWED_RUNG_KINDS",
    "ALLOWED_STATUSES",
    "SECONDS_PER_HOUR",
    "Hardware",
    "Ladder",
    "Price",
    "Rung",
    "RungResult",
    "cost_usd",
    "flops",
    "gpu_hours",
    "implied_mfu",
    "load_ladder",
    "load_prices",
    "load_rungs",
    "main",
    "render_json",
    "render_markdown",
    "wall_clock_hours",
]

SECONDS_PER_HOUR = 3600.0

ALLOWED_PRICE_KINDS = frozenset(
    {"spot", "on-demand", "reserved", "flex-start", "marketplace", "owned"}
)
ALLOWED_STATUSES = frozenset({"planned", "running", "done"})
ALLOWED_RUNG_KINDS = frozenset({"rung", "reference", "frontier"})

_HERE = Path(__file__).resolve().parent
REPO_ROOT = _HERE.parents[1]
PRICES_YAML = _HERE / "prices.yaml"
RUNGS_YAML = _HERE / "rungs.yaml"


# ======================================================================================
# Pure arithmetic -- the whole model, four functions
# ======================================================================================


def flops(params_active: float, tokens: float) -> float:
    """Total training FLOPs for ``tokens`` tokens through ``params_active`` active params.

    The standard 6ND estimate: two FLOPs per parameter per token in the forward pass and four
    in the backward pass.  For a mixture-of-experts model N is the *active* parameter count,
    which is why a 30B-A3B MoE costs the same to train as a 3B dense model.

    Args:
        params_active: active (not total) parameters, a count.
        tokens: training tokens, a count.

    Returns:
        Total FLOPs for the run.
    """
    if params_active < 0 or tokens < 0:
        raise ValueError("params_active and tokens must be non-negative")
    return 6.0 * params_active * tokens


def gpu_hours(total_flops: float, peak_tflops: float, mfu: float) -> float:
    """Accelerator-hours needed to execute ``total_flops`` at ``mfu`` of ``peak_tflops``.

    Args:
        total_flops: total FLOPs for the run (see :func:`flops`).
        peak_tflops: accelerator peak in TFLOP/s (1e12 FLOP/s).
        mfu: model FLOPs utilisation as a fraction in (0, 1].

    Returns:
        Accelerator-hours (GPU-hours).
    """
    if peak_tflops <= 0:
        raise ValueError("peak_tflops must be positive")
    if not 0.0 < mfu <= 1.0:
        raise ValueError(f"mfu must be in (0, 1], got {mfu}")
    return total_flops / (peak_tflops * 1e12 * mfu * SECONDS_PER_HOUR)


def wall_clock_hours(total_gpu_hours: float, n_gpus: int) -> float:
    """Calendar hours to burn ``total_gpu_hours`` on ``n_gpus`` accelerators in parallel.

    Perfect scaling is assumed; real runs lose time to communication, checkpointing and
    restarts.  The MFU figure is where those losses are supposed to be absorbed.
    """
    if n_gpus <= 0:
        raise ValueError("n_gpus must be positive")
    return total_gpu_hours / float(n_gpus)


def cost_usd(total_gpu_hours: float, usd_per_gpu_hour: float) -> float:
    """Rental cost in USD for ``total_gpu_hours`` at ``usd_per_gpu_hour`` per accelerator-hour."""
    if usd_per_gpu_hour < 0:
        raise ValueError("usd_per_gpu_hour must be non-negative")
    return total_gpu_hours * usd_per_gpu_hour


def implied_mfu(total_flops: float, disclosed_gpu_hours: float, peak_tflops: float) -> float:
    """Invert :func:`gpu_hours`: the MFU a disclosed GPU-hour count implies.

    This is the calibration direction -- given somebody's published GPU-hours for a run whose N
    and D are known, what utilisation did they actually achieve?
    """
    if disclosed_gpu_hours <= 0 or peak_tflops <= 0:
        raise ValueError("disclosed_gpu_hours and peak_tflops must be positive")
    return total_flops / (disclosed_gpu_hours * SECONDS_PER_HOUR * peak_tflops * 1e12)


# ======================================================================================
# Data model
# ======================================================================================


@dataclass(frozen=True)
class Hardware:
    """An accelerator: a name, a peak, and where the peak came from."""

    key: str
    name: str
    peak_tflops: float | None
    peak_basis: str | None = None
    source: str | None = None
    source_url: str | None = None
    retrieved: str | None = None
    measured: bool = False
    derived: bool = False
    measurements: dict[str, Any] = field(default_factory=dict)
    notes: str | None = None

    def require_peak(self) -> float:
        """Return ``peak_tflops``, or explain why this accelerator cannot be costed."""
        if self.peak_tflops is None:
            raise ValueError(
                f"hardware {self.key!r} has no sourced peak_tflops in prices.yaml; "
                "it cannot be used to cost a rung (honest-numbers rule)"
            )
        return self.peak_tflops


@dataclass(frozen=True)
class Price:
    """One rental price row, with its provenance."""

    id: str
    provider: str
    gpu: str
    hardware: str
    usd_per_gpu_hour: float
    kind: str
    verified: bool = True
    kind_inferred: bool = False
    measured: bool = False
    source: str | None = None
    source_url: str | None = None
    retrieved: str | None = None
    notes: str | None = None


@dataclass(frozen=True)
class Rung:
    """One row of the ladder -- ours to run, someone else's disclosed run, or the frontier."""

    id: str
    kind: str
    name: str
    hardware: str
    price: str
    status: str
    params_total: float | None = None
    params_active: float | None = None
    tokens: float | None = None
    mfu: float | None = None
    gpus: int | None = None
    flops_low: float | None = None
    flops_high: float | None = None
    beats: str | None = None
    verified: bool = True
    disclosed: dict[str, Any] = field(default_factory=dict)
    targets: dict[str, Any] = field(default_factory=dict)
    see_also: str | None = None
    notes: str | None = None


@dataclass(frozen=True)
class RungResult:
    """Everything the ladder computes for one rung.  ``*_high`` is set only for ranges."""

    rung: Rung
    hardware: Hardware
    flops_low: float | None = None
    flops_high: float | None = None
    gpu_hours_low: float | None = None
    gpu_hours_high: float | None = None
    wall_clock_low: float | None = None
    wall_clock_high: float | None = None
    cost_spot_low: float | None = None
    cost_spot_high: float | None = None
    cost_on_demand_low: float | None = None
    cost_on_demand_high: float | None = None
    price_spot: Price | None = None
    price_on_demand: Price | None = None
    disclosed_gpu_hours: float | None = None
    implied_mfu: float | None = None
    implied_mfu_published: float | None = None

    @property
    def is_range(self) -> bool:
        return self.flops_high is not None and self.flops_low != self.flops_high


# ======================================================================================
# Loading
# ======================================================================================


def _read_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    if not isinstance(data, dict):
        raise TypeError(f"{path} did not parse to a mapping")
    return data


def _num(value: Any, *, what: str = "value") -> float | None:
    """Coerce a YAML scalar to float, tolerating ``124.0e6``.

    PyYAML implements YAML 1.1, whose float resolver requires a *signed* exponent, so a
    perfectly readable ``124.0e6`` in the YAML arrives here as the string ``"124.0e6"``.
    Rather than litter the data files with ``1.24e+8``, coerce on the way in.
    """
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value.strip())
        except ValueError as exc:
            raise ValueError(f"{what}: cannot read {value!r} as a number") from exc
    raise ValueError(f"{what}: cannot read {value!r} as a number")


_DISCLOSED_NUMERIC_KEYS = (
    "gpu_hours",
    "wall_clock_hours",
    "implied_mfu_published",
    "cost_usd",
    "cost_usd_spot",
    "cost_usd_on_demand",
    "cost_estimate_spot_usd",
    "cost_estimate_on_demand_usd",
)


def _coerce_disclosed(block: dict[str, Any]) -> dict[str, Any]:
    """Coerce the numeric fields of a ``disclosed:`` block, leaving prose keys alone."""
    out = dict(block)
    for key in _DISCLOSED_NUMERIC_KEYS:
        if key in out:
            out[key] = _num(out[key], what=f"disclosed.{key}")
    return out


def load_prices(path: Path | None = None) -> tuple[dict[str, Hardware], dict[str, Price], dict[str, Any]]:
    """Load ``prices.yaml``.  Returns ``(hardware_by_key, price_by_id, meta)``."""
    raw = _read_yaml(path or PRICES_YAML)
    hardware: dict[str, Hardware] = {}
    for key, spec in (raw.get("hardware") or {}).items():
        hardware[key] = Hardware(
            key=key,
            name=spec["name"],
            peak_tflops=_num(spec.get("peak_tflops"), what=f"hardware {key}.peak_tflops"),
            peak_basis=spec.get("peak_basis"),
            source=spec.get("source"),
            source_url=spec.get("source_url"),
            retrieved=spec.get("retrieved"),
            measured=bool(spec.get("measured", False)),
            derived=bool(spec.get("derived", False)),
            measurements={
                mk: _num(mv, what=f"hardware {key}.measurements.{mk}")
                for mk, mv in (spec.get("measurements") or {}).items()
            },
            notes=spec.get("notes"),
        )
    prices: dict[str, Price] = {}
    for spec in raw.get("prices") or []:
        price = Price(
            id=spec["id"],
            provider=spec["provider"],
            gpu=spec["gpu"],
            hardware=spec["hardware"],
            usd_per_gpu_hour=float(_num(spec["usd_per_gpu_hour"], what=f"price {spec['id']}") or 0.0),
            kind=spec["kind"],
            verified=bool(spec.get("verified", True)),
            kind_inferred=bool(spec.get("kind_inferred", False)),
            measured=bool(spec.get("measured", False)),
            source=spec.get("source"),
            source_url=spec.get("source_url"),
            retrieved=spec.get("retrieved"),
            notes=spec.get("notes"),
        )
        if price.kind not in ALLOWED_PRICE_KINDS:
            raise ValueError(f"price {price.id!r}: unknown kind {price.kind!r}")
        if price.hardware not in hardware:
            raise ValueError(f"price {price.id!r}: unknown hardware key {price.hardware!r}")
        prices[price.id] = price
    return hardware, prices, raw.get("meta") or {}


def load_rungs(path: Path | None = None) -> tuple[list[Rung], dict[str, Any]]:
    """Load ``rungs.yaml``.  Returns ``(rungs, meta)``."""
    raw = _read_yaml(path or RUNGS_YAML)
    rungs: list[Rung] = []
    for spec in raw.get("rungs") or []:
        rung = Rung(
            id=spec["id"],
            kind=spec["kind"],
            name=spec["name"],
            hardware=spec["hardware"],
            price=spec["price"],
            status=spec["status"],
            params_total=_num(spec.get("params_total"), what=f"rung {spec['id']}.params_total"),
            params_active=_num(spec.get("params_active"), what=f"rung {spec['id']}.params_active"),
            tokens=_num(spec.get("tokens"), what=f"rung {spec['id']}.tokens"),
            mfu=_num(spec.get("mfu"), what=f"rung {spec['id']}.mfu"),
            gpus=int(_gpus) if (_gpus := _num(spec.get("gpus"), what=f"rung {spec['id']}.gpus")) else None,
            flops_low=_num(spec.get("flops_low"), what=f"rung {spec['id']}.flops_low"),
            flops_high=_num(spec.get("flops_high"), what=f"rung {spec['id']}.flops_high"),
            beats=spec.get("beats"),
            verified=bool(spec.get("verified", True)),
            disclosed=_coerce_disclosed(spec.get("disclosed") or {}),
            targets=spec.get("targets") or {},
            see_also=spec.get("see_also"),
            notes=spec.get("notes"),
        )
        if rung.kind not in ALLOWED_RUNG_KINDS:
            raise ValueError(f"rung {rung.id!r}: unknown kind {rung.kind!r}")
        if rung.status not in ALLOWED_STATUSES:
            raise ValueError(f"rung {rung.id!r}: unknown status {rung.status!r}")
        rungs.append(rung)
    return rungs, raw.get("meta") or {}


@dataclass
class Ladder:
    """The loaded ladder: hardware, prices, rungs, and the arithmetic that joins them."""

    hardware: dict[str, Hardware]
    prices: dict[str, Price]
    rungs: list[Rung]
    price_meta: dict[str, Any] = field(default_factory=dict)
    rung_meta: dict[str, Any] = field(default_factory=dict)

    # -- price resolution ---------------------------------------------------------------

    def cheapest(self, hardware_key: str, kind: str, *, verified_only: bool = True) -> Price | None:
        """Cheapest price row of ``kind`` for ``hardware_key``.

        Unverified rows (third-party quotes research/04 itself tags UNVERIFIED) are skipped
        unless ``verified_only=False`` -- a price floor built on an unverifiable quote is not a
        price floor.
        """
        candidates = [
            p
            for p in self.prices.values()
            if p.hardware == hardware_key and p.kind == kind and (p.verified or not verified_only)
        ]
        if not candidates:
            return None
        return min(candidates, key=lambda p: p.usd_per_gpu_hour)

    # -- the calculation ----------------------------------------------------------------

    def compute(self, rung: Rung) -> RungResult:
        """Compute FLOPs, GPU-hours, wall-clock and cost for one rung."""
        hw = self.hardware[rung.hardware]

        if rung.flops_low is not None:
            f_lo = float(rung.flops_low)
            f_hi = float(rung.flops_high if rung.flops_high is not None else rung.flops_low)
        elif rung.params_active is not None and rung.tokens is not None:
            f_lo = f_hi = flops(float(rung.params_active), float(rung.tokens))
        else:
            f_lo = f_hi = None  # type: ignore[assignment]

        peak = hw.peak_tflops
        gh_lo = gh_hi = None
        if f_lo is not None and peak is not None and rung.mfu:
            gh_lo = gpu_hours(f_lo, peak, float(rung.mfu))
            gh_hi = gpu_hours(f_hi, peak, float(rung.mfu))

        wc_lo = wc_hi = None
        if gh_lo is not None and rung.gpus:
            wc_lo = wall_clock_hours(gh_lo, int(rung.gpus))
            wc_hi = wall_clock_hours(gh_hi, int(rung.gpus))

        named = self.prices.get(rung.price)
        if named is not None and named.hardware != rung.hardware:
            named = None  # a price row for other silicon is not this rung's price
        spot = self.cheapest(rung.hardware, "spot") or (
            named if named is not None and named.kind in ("spot", "owned") else None
        )
        on_dem = self.cheapest(rung.hardware, "on-demand")
        if hw.key == "mac_m4_10core":  # owned hardware: both columns are the same $0 row
            spot = on_dem = named

        cs_lo = cs_hi = co_lo = co_hi = None
        if gh_lo is not None and spot is not None:
            cs_lo = cost_usd(gh_lo, spot.usd_per_gpu_hour)
            cs_hi = cost_usd(gh_hi, spot.usd_per_gpu_hour)
        if gh_lo is not None and on_dem is not None:
            co_lo = cost_usd(gh_lo, on_dem.usd_per_gpu_hour)
            co_hi = cost_usd(gh_hi, on_dem.usd_per_gpu_hour)

        disclosed_gh = rung.disclosed.get("gpu_hours")
        imp = None
        if disclosed_gh and f_lo is not None and peak is not None:
            imp = implied_mfu(f_lo, float(disclosed_gh), peak)

        return RungResult(
            rung=rung,
            hardware=hw,
            flops_low=f_lo,
            flops_high=f_hi,
            gpu_hours_low=gh_lo,
            gpu_hours_high=gh_hi,
            wall_clock_low=wc_lo,
            wall_clock_high=wc_hi,
            cost_spot_low=cs_lo,
            cost_spot_high=cs_hi,
            cost_on_demand_low=co_lo,
            cost_on_demand_high=co_hi,
            price_spot=spot,
            price_on_demand=on_dem,
            disclosed_gpu_hours=float(disclosed_gh) if disclosed_gh else None,
            implied_mfu=imp,
            implied_mfu_published=rung.disclosed.get("implied_mfu_published"),
        )

    def compute_all(self) -> list[RungResult]:
        return [self.compute(r) for r in self.rungs]

    def by_kind(self, kind: str) -> list[RungResult]:
        return [self.compute(r) for r in self.rungs if r.kind == kind]

    def calibration(self) -> list[RungResult]:
        """Reference rows that disclose both a GPU-hour count and a published implied MFU."""
        return [
            res
            for res in self.by_kind("reference")
            if res.implied_mfu is not None and res.implied_mfu_published is not None
        ]


def load_ladder(prices_path: Path | None = None, rungs_path: Path | None = None) -> Ladder:
    """Load both YAMLs into a :class:`Ladder`."""
    hardware, prices, price_meta = load_prices(prices_path)
    rungs, rung_meta = load_rungs(rungs_path)
    for rung in rungs:
        if rung.hardware not in hardware:
            raise ValueError(f"rung {rung.id!r}: unknown hardware key {rung.hardware!r}")
        if rung.price not in prices:
            raise ValueError(f"rung {rung.id!r}: unknown price key {rung.price!r}")
    return Ladder(hardware=hardware, prices=prices, rungs=rungs, price_meta=price_meta, rung_meta=rung_meta)


# ======================================================================================
# Formatting
# ======================================================================================

DASH = "\u2014"  # em dash: "no number exists", never 0


def fmt_count(x: float | None, unit: str = "") -> str:
    """Format a parameter/token count as 124M / 3.0B / 11.2T."""
    if x is None:
        return DASH
    for scale, suffix in ((1e12, "T"), (1e9, "B"), (1e6, "M"), (1e3, "K")):
        if abs(x) >= scale:
            val = x / scale
            text = f"{val:,.0f}" if abs(val) >= 100 else f"{val:.1f}".removesuffix(".0")
            return f"{text}{suffix}{unit}"
    return f"{x:,.0f}{unit}"


def fmt_flops(x: float | None) -> str:
    """Format FLOPs in scientific notation, e.g. ``1.08e21``."""
    if x is None:
        return DASH
    exponent = math.floor(math.log10(abs(x))) if x else 0
    mantissa = x / (10.0**exponent)
    return f"{mantissa:.2f}e{exponent}"


def fmt_hours(x: float | None) -> str:
    """Format accelerator-hours: 119 / 6,163 / 2.79M."""
    if x is None:
        return DASH
    if x >= 1e6:
        return f"{x / 1e6:.2f}M"
    if x >= 1000:
        return f"{x:,.0f}"
    if x >= 10:
        return f"{x:.0f}"
    return f"{x:.3g}"


def fmt_hours_precise(x: float | None) -> str:
    """Format accelerator-hours keeping three significant figures below 100: ``0.164``, ``13.2``."""
    if x is None:
        return DASH
    return f"{x:,.0f}" if x >= 100 else f"{x:.3g}"


def fmt_wall(x: float | None) -> str:
    """Format wall-clock hours with a human-scale gloss, e.g. ``108 h (4.5 d)``."""
    if x is None:
        return DASH
    if x < 1:
        return f"{x * 60:.1f} min"
    if x < 48:
        return f"{x:.1f} h"
    return f"{x:,.0f} h ({x / 24:.1f} d)"


def fmt_usd(x: float | None) -> str:
    """Format dollars: $0 / $91 / $2,037 / $5.79M / $1.95B."""
    if x is None:
        return DASH
    if x == 0:
        return "$0"
    if abs(x) >= 1e9:
        return f"${x / 1e9:.2f}B"
    if abs(x) >= 1e6:
        return f"${x / 1e6:.2f}M"
    if abs(x) >= 1000:
        return f"${x:,.0f}"
    return f"${x:.0f}" if x >= 10 else f"${x:.2f}"


def fmt_range(lo: float | None, hi: float | None, formatter) -> str:
    """Render ``lo`` alone, or ``lo–hi`` when the two differ (the frontier row)."""
    if lo is None:
        return DASH
    if hi is None or math.isclose(lo, hi, rel_tol=1e-12):
        return formatter(lo)
    return f"{formatter(lo)}\u2013{formatter(hi)}"


def fmt_pct(x: float | None) -> str:
    return DASH if x is None else f"{x * 100:.1f}%"


def _cell(text: str | None) -> str:
    """Escape a value for a markdown table cell."""
    if text is None:
        return DASH
    return str(text).replace("|", "\\|").replace("\n", " ").strip()


# ======================================================================================
# Rendering
# ======================================================================================

_MAIN_COLUMNS = (
    "Rung",
    "Beats (target model)",
    "Params (total / active)",
    "Tokens",
    "FLOPs",
    "Hardware",
    "GPU-hours",
    "Wall-clock",
    "$ spot",
    "$ on-demand",
    "Status",
)


def _main_row(res: RungResult) -> list[str]:
    r = res.rung
    params = (
        DASH
        if r.params_total is None
        else f"{fmt_count(r.params_total)} / {fmt_count(r.params_active)}"
    )
    hw_label = f"{res.hardware.key} x{r.gpus}" if r.gpus else res.hardware.key
    return [
        _cell(r.name),
        _cell(r.beats),
        params,
        fmt_count(r.tokens),
        fmt_range(res.flops_low, res.flops_high, fmt_flops),
        _cell(hw_label),
        fmt_range(res.gpu_hours_low, res.gpu_hours_high, fmt_hours),
        fmt_range(res.wall_clock_low, res.wall_clock_high, fmt_wall),
        fmt_range(res.cost_spot_low, res.cost_spot_high, fmt_usd),
        fmt_range(res.cost_on_demand_low, res.cost_on_demand_high, fmt_usd),
        _cell(r.status),
    ]


def _table(header: tuple[str, ...] | list[str], rows: list[list[str]]) -> str:
    out = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    out.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(out)


def render_markdown(ladder: Ladder, *, generated: str | None = None) -> str:
    """Render the full ladder as markdown."""
    generated = generated or datetime.now(UTC).date().isoformat()
    results = ladder.compute_all()
    by_kind: dict[str, list[RungResult]] = {"rung": [], "reference": [], "frontier": []}
    for res in results:
        by_kind[res.rung.kind].append(res)

    parts: list[str] = []
    parts.append("# The compute ladder\n")
    parts.append(
        f"*Generated by `python -m r52.ladder` on {generated}. Prices and peak-FLOPS figures "
        f"retrieved {ladder.price_meta.get('retrieved', 'n/d')}; see "
        "`r52/ladder/prices.yaml` and `r52/ladder/rungs.yaml` for a source URL on every row.*\n"
    )
    parts.append(
        "**Method.** `FLOPs = 6 x N_active x D`; "
        "`GPU-hours = FLOPs / (peak_TFLOPS x MFU x 3600)`; "
        "`wall-clock = GPU-hours / n_GPUs`; `cost = GPU-hours x $/GPU-hour`. "
        "N is the *active* parameter count, which is why a 30B-A3B MoE and a 3B dense model "
        "cost the same to train. MFU is calibrated against three disclosed runs (below). "
        "Prices are the verified September-2026 floors from `research/04` §B.2: "
        "H100 spot $0.94, H100 on-demand $2.43. "
        f"`{DASH}` means no number exists \u2014 never 0.\n"
    )

    parts.append("## The ladder\n")
    parts.append(_table(_MAIN_COLUMNS, [_main_row(r) for r in by_kind["rung"]]))
    parts.append("")

    parts.append("## Calibration and reference rows\n")
    parts.append(
        "Runs other people actually did and disclosed. The three rows with a published implied "
        "MFU are what this ladder's MFU model is fitted against; `tests/test_ladder.py` asserts "
        "our recomputed values match them to within 1 percentage point.\n"
    )
    parts.append(_table(_MAIN_COLUMNS, [_main_row(r) for r in by_kind["reference"]]))
    parts.append("")

    calib_rows = []
    for res in by_kind["reference"]:
        if res.disclosed_gpu_hours is None:
            continue
        calib_rows.append(
            [
                _cell(res.rung.name),
                fmt_flops(res.flops_low),
                (
                    f"{fmt_hours_precise(res.disclosed_gpu_hours)} "
                    f"({_cell(res.rung.disclosed.get('gpu_hours_unit'))})"
                ),
                fmt_pct(res.implied_mfu),
                fmt_pct(res.implied_mfu_published),
                _cell(res.rung.disclosed.get("source_url")),
            ]
        )
    parts.append(
        _table(
            ["Run", "FLOPs (6ND)", "Disclosed GPU-hours", "Implied MFU (ours)", "Implied MFU (published)", "Source"],
            calib_rows,
        )
    )
    parts.append("")

    parts.append("## The frontier\n")
    parts.append(_table(_MAIN_COLUMNS, [_main_row(r) for r in by_kind["frontier"]]))
    parts.append("")
    parts.append(
        "> The frontier row's dollar figures are **our own arithmetic** at 35% MFU on "
        "September-2026 rental prices, not a disclosure. Anthropic has published no GPU-hours, "
        "no FLOPs and no cost for Fable 5, Fable 5.1 or Opus 5, and Epoch's cost article "
        "publishes trends, not per-model dollar figures. See `rungs.yaml` for the full caveat.\n"
    )

    parts.append("## Notes on each row\n")
    for res in results:
        if res.rung.notes:
            parts.append(f"**{res.rung.name}** \u2014 {' '.join(res.rung.notes.split())}\n")

    parts.append("## Price rows used\n")
    used: dict[str, Price] = {}
    for res in results:
        for price in (res.price_spot, res.price_on_demand):
            if price is not None:
                used[price.id] = price
    price_rows = [
        [
            _cell(p.id),
            _cell(p.provider),
            _cell(p.gpu),
            f"${p.usd_per_gpu_hour:.2f}",
            _cell(p.kind) + (" (inferred)" if p.kind_inferred else ""),
            _cell(p.source_url or p.source),
            _cell(p.retrieved),
        ]
        for p in sorted(used.values(), key=lambda p: (p.hardware, p.usd_per_gpu_hour))
    ]
    parts.append(
        _table(["id", "Provider", "Accelerator", "$/GPU-h", "Kind", "Source", "Retrieved"], price_rows)
    )
    parts.append("")
    return "\n".join(parts)


def _result_to_dict(res: RungResult) -> dict[str, Any]:
    r = res.rung
    return {
        "id": r.id,
        "kind": r.kind,
        "name": r.name,
        "beats": r.beats,
        "status": r.status,
        "verified": r.verified,
        "inputs": {
            "params_total": r.params_total,
            "params_active": r.params_active,
            "tokens": r.tokens,
            "mfu": r.mfu,
            "hardware": r.hardware,
            "peak_tflops": res.hardware.peak_tflops,
            "gpus": r.gpus,
            "price": r.price,
        },
        "computed": {
            "flops_low": res.flops_low,
            "flops_high": res.flops_high,
            "gpu_hours_low": res.gpu_hours_low,
            "gpu_hours_high": res.gpu_hours_high,
            "wall_clock_hours_low": res.wall_clock_low,
            "wall_clock_hours_high": res.wall_clock_high,
            "cost_usd_spot_low": res.cost_spot_low,
            "cost_usd_spot_high": res.cost_spot_high,
            "cost_usd_on_demand_low": res.cost_on_demand_low,
            "cost_usd_on_demand_high": res.cost_on_demand_high,
            "price_spot_id": res.price_spot.id if res.price_spot else None,
            "price_on_demand_id": res.price_on_demand.id if res.price_on_demand else None,
            "implied_mfu": res.implied_mfu,
        },
        "disclosed": r.disclosed or None,
        "targets": r.targets or None,
        "notes": " ".join(r.notes.split()) if r.notes else None,
    }


def render_json(ladder: Ladder, *, generated: str | None = None) -> dict[str, Any]:
    """Render the full ladder as a JSON-serialisable dict."""
    return {
        "generated": generated or datetime.now(UTC).date().isoformat(),
        "method": "FLOPs = 6 * N_active * D; GPU-hours = FLOPs / (peak_TFLOPS * MFU * 3600)",
        "price_meta": ladder.price_meta,
        "rung_meta": ladder.rung_meta,
        "hardware": {
            k: {
                "name": h.name,
                "peak_tflops": h.peak_tflops,
                "peak_basis": h.peak_basis,
                "measured": h.measured,
                "derived": h.derived,
                "measurements": h.measurements or None,
                "source": h.source,
                "source_url": h.source_url,
                "retrieved": h.retrieved,
            }
            for k, h in ladder.hardware.items()
        },
        "prices": [
            {
                "id": p.id,
                "provider": p.provider,
                "gpu": p.gpu,
                "hardware": p.hardware,
                "usd_per_gpu_hour": p.usd_per_gpu_hour,
                "kind": p.kind,
                "kind_inferred": p.kind_inferred,
                "verified": p.verified,
                "source": p.source,
                "source_url": p.source_url,
                "retrieved": p.retrieved,
            }
            for p in ladder.prices.values()
        ],
        "rungs": [_result_to_dict(r) for r in ladder.compute_all()],
    }


# ======================================================================================
# CLI
# ======================================================================================


def _parse_custom(spec: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for chunk in spec.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if "=" not in chunk:
            raise ValueError(f"--custom expects key=value pairs, got {chunk!r}")
        key, _, value = chunk.partition("=")
        out[key.strip().lower()] = value.strip()
    return out


def run_custom(ladder: Ladder, spec: str) -> dict[str, Any]:
    """Cost one ad-hoc configuration, e.g. ``N=3e9,D=60e9,hw=h100,gpus=8``.

    Recognised keys: ``N`` (active params), ``D`` (tokens), ``hw`` (hardware key),
    ``gpus``, ``mfu`` (default 0.35), ``price`` (a price id to use for the headline cost).
    """
    kv = _parse_custom(spec)
    missing = {"n", "d", "hw"} - kv.keys()
    if missing:
        raise ValueError(f"--custom is missing required key(s): {', '.join(sorted(missing))}")
    hw_key = kv["hw"]
    if hw_key not in ladder.hardware:
        raise ValueError(f"unknown hardware key {hw_key!r}; known: {', '.join(sorted(ladder.hardware))}")
    hw = ladder.hardware[hw_key]
    peak = hw.require_peak()
    n = float(kv["n"])
    d = float(kv["d"])
    mfu = float(kv.get("mfu", ladder.rung_meta.get("default_mfu_large", 0.35)))
    gpus = int(float(kv.get("gpus", 1)))

    total = flops(n, d)
    hours = gpu_hours(total, peak, mfu)
    wall = wall_clock_hours(hours, gpus)
    spot = ladder.cheapest(hw_key, "spot") or (ladder.prices.get(kv["price"]) if "price" in kv else None)
    on_dem = ladder.cheapest(hw_key, "on-demand")
    if hw_key == "mac_m4_10core":
        spot = on_dem = ladder.prices.get("mac_m4_owned")

    result: dict[str, Any] = {
        "params_active": n,
        "tokens": d,
        "hardware": hw_key,
        "peak_tflops": peak,
        "mfu": mfu,
        "gpus": gpus,
        "flops": total,
        "gpu_hours": hours,
        "wall_clock_hours": wall,
        "cost_usd_spot": cost_usd(hours, spot.usd_per_gpu_hour) if spot else None,
        "cost_usd_on_demand": cost_usd(hours, on_dem.usd_per_gpu_hour) if on_dem else None,
        "price_spot_id": spot.id if spot else None,
        "price_on_demand_id": on_dem.id if on_dem else None,
    }
    return result


def _print_custom(result: dict[str, Any]) -> None:
    print("road-to-52 compute ladder \u2014 custom configuration")
    print(f"  active params    {fmt_count(result['params_active'])}")
    print(f"  tokens           {fmt_count(result['tokens'])}")
    print(f"  hardware         {result['hardware']} @ {result['peak_tflops']} TFLOP/s peak, "
          f"MFU {fmt_pct(result['mfu'])}, {result['gpus']} accelerator(s)")
    print(f"  FLOPs (6ND)      {fmt_flops(result['flops'])}")
    print(f"  GPU-hours        {fmt_hours(result['gpu_hours'])}")
    print(f"  wall-clock       {fmt_wall(result['wall_clock_hours'])}")
    print(f"  $ spot           {fmt_usd(result['cost_usd_spot'])}  [{result['price_spot_id']}]")
    print(f"  $ on-demand      {fmt_usd(result['cost_usd_on_demand'])}  [{result['price_on_demand_id']}]")


def _print_calibration(ladder: Ladder) -> None:
    print("MFU calibration against disclosed runs (research/04 §B.5)")
    print(f"  {'run':<34} {'FLOPs':>10} {'GPU-hours':>12} {'ours':>8} {'published':>10} {'delta':>8}")
    for res in ladder.calibration():
        delta = (res.implied_mfu - res.implied_mfu_published) * 100.0
        print(
            f"  {res.rung.id:<34} {fmt_flops(res.flops_low):>10} "
            f"{res.disclosed_gpu_hours:>12,.0f} {fmt_pct(res.implied_mfu):>8} "
            f"{fmt_pct(res.implied_mfu_published):>10} {delta:>+7.2f}pp"
        )


def main(argv: list[str] | None = None) -> int:
    """CLI entry point for ``python -m r52.ladder``."""
    parser = argparse.ArgumentParser(
        prog="python -m r52.ladder",
        description="Render the road-to-52 compute ladder (markdown + JSON), or cost one "
        "ad-hoc configuration.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=REPO_ROOT / "results",
        help="directory for ladder.md / ladder.json (default: <repo>/results)",
    )
    parser.add_argument("--custom", type=str, default=None, help='e.g. "N=3e9,D=60e9,hw=h100,gpus=8"')
    parser.add_argument("--calibrate", action="store_true", help="print implied MFUs for the disclosed runs")
    parser.add_argument("--json", action="store_true", help="with --custom, print JSON instead of text")
    args = parser.parse_args(argv)

    ladder = load_ladder()

    if args.custom:
        result = run_custom(ladder, args.custom)
        if args.json:
            print(json.dumps(result, indent=2))
        else:
            _print_custom(result)
        return 0

    if args.calibrate:
        _print_calibration(ladder)
        return 0

    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    md_path = out_dir / "ladder.md"
    json_path = out_dir / "ladder.json"
    md_path.write_text(render_markdown(ladder), encoding="utf-8")
    json_path.write_text(json.dumps(render_json(ladder), indent=2) + "\n", encoding="utf-8")
    print(f"wrote {md_path}")
    print(f"wrote {json_path}")
    _print_calibration(ladder)
    return 0
