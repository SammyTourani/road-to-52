# Copyright 2026 The road-to-52 authors.
# SPDX-License-Identifier: Apache-2.0
"""The data-ablation lab (``docs/ABLATIONS.md``, Phase 4).

Three axes, one cell at a time, never overlapping a headline run:

``corpus``
    six web corpora at matched tokens with one shared tokenizer (:mod:`r52.ablate.corpora`).
``arch``
    one architecture knob at a time off the Rung 0 default.
``tokenizer``
    GPT-2's 50,257-entry BPE against our own 32,768-entry BPE at matched **bytes**.

:mod:`r52.ablate.matrix` enumerates the cells, :mod:`r52.ablate.run` runs one, and
:mod:`r52.ablate.report` renders ``results/ablations/<axis>.md``.  ``scripts/ablate.sh`` is
the sequential, resumable driver.
"""

from __future__ import annotations

__all__ = ["corpora", "matrix", "report", "run"]
