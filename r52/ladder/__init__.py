# Copyright 2026 The road-to-52 authors.
# SPDX-License-Identifier: Apache-2.0
"""Compute ladder: 6ND FLOPs -> GPU-hours -> wall-clock -> dollars.

Data lives in ``prices.yaml`` (rental prices + accelerator peaks, one source URL per row) and
``rungs.yaml`` (the ladder rows, including the disclosed runs the MFU model is calibrated
against).  ``python -m r52.ladder`` renders ``results/ladder.md`` and ``results/ladder.json``.
"""

from r52.ladder.ladder import (
    Hardware,
    Ladder,
    Price,
    Rung,
    RungResult,
    cost_usd,
    flops,
    gpu_hours,
    implied_mfu,
    load_ladder,
    load_prices,
    load_rungs,
    render_json,
    render_markdown,
    wall_clock_hours,
)

__all__ = [
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
    "render_json",
    "render_markdown",
    "wall_clock_hours",
]
