# Copyright 2026 The road-to-52 authors.
# SPDX-License-Identifier: Apache-2.0
"""The bar: published benchmark scores and the gap between every model and Claude Fable 5.1.

``bar.yaml`` holds every number with its source URL, retrieval date and eval conditions;
``gap.py`` renders ``results/GAP.md`` and ``results/GAP.json``.  Missing numbers are omitted
from the YAML and render as an em dash -- never as a zero.
"""

from r52.bar.gap import (
    Bar,
    Benchmark,
    GapRow,
    Model,
    Score,
    gap_rows,
    load_bar,
    load_local_results,
    merge_local,
    render_json,
    render_markdown,
)

__all__ = [
    "Bar",
    "Benchmark",
    "GapRow",
    "Model",
    "Score",
    "gap_rows",
    "load_bar",
    "load_local_results",
    "merge_local",
    "render_json",
    "render_markdown",
]
