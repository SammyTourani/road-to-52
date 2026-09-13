# Copyright 2026 The road-to-52 authors.
# SPDX-License-Identifier: Apache-2.0
"""Evaluation suite for road-to-52 (docs/ARCHITECTURE.md §6).

Four evaluations, one adapter, one reporter:

* :mod:`r52.eval.lm`         -- :class:`~r52.eval.lm.LM`, the single logits interface every
                                eval runs against (an ``r52.model.GPT`` checkpoint directory
                                or an mlx-lm model directory, including the GPT-2 reference).
* :mod:`r52.eval.val_loss`   -- mean cross-entropy (nats/token) and bits-per-byte on the fixed
                                10,485,760-token FineWeb validation split.
* :mod:`r52.eval.hellaswag`  -- HellaSwag validation (10,042 examples), llm.c's protocol.
* :mod:`r52.eval.core`       -- DCLM CORE over nanochat's 22-task eval bundle.
* :mod:`r52.eval.report`     -- results JSON + a row in ``docs/RESULTS.md``.

Every entry point caps MLX memory (``r52.eval.lm.configure_runtime``) because this machine
also runs long pretraining jobs; see ``docs/ARCHITECTURE.md`` §1.2.
"""

from __future__ import annotations

__all__ = ["LM", "configure_runtime"]


def __getattr__(name: str):  # pragma: no cover - thin lazy re-export
    if name in ("LM", "configure_runtime"):
        from . import lm

        return getattr(lm, name)
    raise AttributeError(name)
