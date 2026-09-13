# Copyright 2026 The road-to-52 authors.
# SPDX-License-Identifier: Apache-2.0
"""The three ablation axes of ``docs/ABLATIONS.md``, as lists of config overrides.

    python -m r52.ablate.matrix                # print every axis
    python -m r52.ablate.matrix --axis corpus  # one axis, with the commands it implies

Every cell is a **config path plus a list of ``key=value`` overrides** -- the same strings
``python -m r52.train -o ...`` takes -- so a cell is fully described by text that fits in a
results JSON, and nothing about how a run was configured has to be reconstructed later.

Three things the axes deliberately do *not* share:

* the **corpus** axis fixes tokens and varies the data, so every cell trains on the same
  number of tokens of a different corpus, tokenized by one shared 32K BPE;
* the **arch** axis fixes data and moves one knob at a time off the Rung 0 default;
* the **tokenizer** axis fixes **bytes**, not tokens (``docs/ABLATIONS.md`` Axis 3), because
  a tokenizer that packs more bytes into a token would otherwise be handed more text for
  free.  :mod:`r52.ablate.run` converts the byte budget into a token budget using the
  ``bytes_per_token`` its corpus manifest measured, and the comparison is read off bpb and
  HellaSwag -- never off loss, which is not comparable across vocabularies.
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from typing import Any

from .corpora import get_corpus

__all__ = [
    "AXES",
    "CORE6",
    "CORE6_ALTERNATIVE",
    "DEFAULT_ABL_TOKENIZER",
    "TOKENIZER_AXIS_BYTES",
    "Axis",
    "Cell",
    "axis",
    "axis_names",
    "build_axes",
    "main",
]

DEFAULT_ABL_TOKENIZER = "data/tokenizers/fineweb32k"
"""The shared 32,768-entry BPE of ``docs/ABLATIONS.md`` Axis 1 ("trained once on a 1B-token
FineWeb sample").  Build it with::

    python -m r52.tokenizer_train --name fineweb32k --corpus fineweb --bytes 1000000000
"""

CORE6: tuple[str, ...] = (
    "copa",
    "winograd",
    "bigbench_repeat_copy_logic",
    "openbook_qa",
    "agi_eval_lsat_ar",
    "winogrande",
)
"""The six cheapest CORE tasks, as ``docs/ABLATIONS.md`` asks for.

Cost is ``items x (few-shot + 1)`` forward rows; measured against the eval bundle on
2026-09-13 these six are 100 / 273 / 352 / 500 / 920 / 1,267, against 110,462 for the full
10-shot ``hellaswag`` task alone.  See :data:`CORE6_ALTERNATIVE` for the caveat.
"""

CORE6_ALTERNATIVE: tuple[str, ...] = (
    "copa",
    "winograd",
    "openbook_qa",
    "winogrande",
    "lambada_openai",
    "hellaswag_zeroshot",
)
"""A costlier six with more signal at 12-35M parameters.

``bigbench_repeat_copy_logic`` (32 items, 10-shot, greedy exact match) and
``agi_eval_lsat_ar`` (230 items, LSAT analytical reasoning) are both at or below their random
baselines for models this small, so two of the six cheapest contribute noise rather than
ranking information.  Swapping them for ``lambada_openai`` and ``hellaswag_zeroshot`` costs
roughly 15x more forward rows.  **This is a planner decision**; the default stays with the
spec's "six cheapest".
"""

TOKENIZER_AXIS_BYTES = 420_000_000
"""Bytes of training text per cell on the tokenizer axis (~100M GPT-2 tokens of FineWeb-Edu)."""


def _slug(text: str) -> str:
    """Filesystem- and run-name-safe version of a cell name."""
    return re.sub(r"[^A-Za-z0-9]+", "-", text).strip("-").lower()


@dataclass(frozen=True)
class Cell:
    """One ablation run: a config, a set of overrides, and how to budget it."""

    axis: str
    name: str
    config: str
    overrides: tuple[str, ...] = ()
    corpus: str | None = None
    tokenizer: str = "gpt2"
    max_tokens: int | None = None
    """Token budget (``--max-tokens``). ``None`` -> derived from ``max_bytes``."""
    max_bytes: int | None = None
    """Byte budget; the tokenizer axis fixes this instead of tokens."""
    published_rank: int | None = None
    published_note: str = ""
    note: str = ""

    @property
    def slug(self) -> str:
        return _slug(self.name)

    @property
    def run_name(self) -> str:
        return f"abl-{_slug(self.axis)}-{self.slug}"

    @property
    def result_path(self) -> str:
        return f"results/ablations/{_slug(self.axis)}/{self.slug}.json"

    def train_argv(self, max_tokens: int | None = None) -> list[str]:
        """The ``python -m r52.train`` argument vector for this cell."""
        argv = [self.config, "--run-name", self.run_name]
        budget = max_tokens if max_tokens is not None else self.max_tokens
        if budget:
            argv += ["--max-tokens", str(int(budget))]
        for o in self.overrides:
            argv += ["-o", o]
        return argv

    def to_dict(self) -> dict[str, Any]:
        return {
            "axis": self.axis,
            "name": self.name,
            "slug": self.slug,
            "config": self.config,
            "overrides": list(self.overrides),
            "corpus": self.corpus,
            "tokenizer": self.tokenizer,
            "max_tokens": self.max_tokens,
            "max_bytes": self.max_bytes,
            "published_rank": self.published_rank,
            "published_note": self.published_note,
            "note": self.note,
            "run_name": self.run_name,
            "result_path": self.result_path,
        }


@dataclass(frozen=True)
class Axis:
    """One ablation axis: a list of cells plus the evaluation protocol they share."""

    name: str
    description: str
    cells: tuple[Cell, ...]
    core_tasks: tuple[str, ...] = CORE6_ALTERNATIVE  # planner decision 2026-09-13, see docs/ABLATIONS.md
    hellaswag_limit: int = 2000
    """Examples of HellaSwag's 10,042 to score (0 = all). 2,000 gives +-2.2 points at 95%."""
    metric: str = "val_bpb"
    """The column the cells are ranked by (bpb: lower is better, tokenizer-neutral)."""

    def __len__(self) -> int:
        return len(self.cells)

    def cell(self, name: str) -> Cell:
        for c in self.cells:
            if c.name == name or c.slug == _slug(name):
                return c
        raise KeyError(f"axis {self.name!r} has no cell {name!r}; cells are {[c.name for c in self.cells]}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "axis": self.name,
            "description": self.description,
            "core_tasks": list(self.core_tasks),
            "hellaswag_limit": self.hellaswag_limit,
            "metric": self.metric,
            "cells": [c.to_dict() for c in self.cells],
        }


# --------------------------------------------------------------------------------------
# Axis 1 -- corpus
# --------------------------------------------------------------------------------------

CORPUS_CELLS: tuple[tuple[str, str, str], ...] = (
    # (cell name, corpus directory stem, published note)
    ("fineweb", "fineweb", "research/03 §1.1: the Rung 0 baseline; superseded on quality-per-token."),
    ("fineweb-edu", "fineweb-edu",
     "research/03 §1.3: classifier-filtered; loses to DCLM by 7.0 CORE at 0.28T matched tokens."),
    ("dclm", "dclm",
     "research/03 §1.3: +7.0 CORE / +13.5 MMLU over FineWeb-Edu at 0.28T; strongest permissive "
     "English web corpus."),
    ("dolma3", "dolma3",
     "research/03 §1.1: fully open and reproducible, but NO published ablation ranks Dolma 3 "
     "against DCLM / FineWeb-Edu / Nemotron-CC -- expected position UNVERIFIED."),
    ("ultra-fineweb", "ultra-fineweb",
     "research/03 §1.3: 1.2B/100B avg 45.891 vs FineWeb-Edu 44.560 vs FineWeb 42.278 "
     "(arXiv:2505.05427); best Apache-2.0 filtered corpus."),
    ("finepdfs-edu-blend", "fineweb-edu+finepdfs-edu25",
     "research/03 §1.1: FinePDFs is the biggest genuinely new source since DCLM and gains come "
     "from MIXING -- keep PDFs under 25% of the mix, which is what this cell does."),
)

CORPUS_TOKENS = 100_000_000
"""Matched token budget per corpus cell (``tiny_abl``'s 100M; ~3 h each)."""


def corpus_axis(
    config: str = "configs/ablations/tiny_abl.yaml",
    tokenizer: str = DEFAULT_ABL_TOKENIZER,
    max_tokens: int = CORPUS_TOKENS,
    root: str = "data/corpora",
) -> Axis:
    """Six corpora at matched tokens with one shared tokenizer."""
    tok_name = "gpt2" if tokenizer == "gpt2" else tokenizer.rstrip("/").rsplit("/", 1)[-1]
    cells = []
    for name, stem, note in CORPUS_CELLS:
        prefix = stem.replace("+", "_")
        data_dir = f"{root}/{stem}-{tok_name}"
        # The blend cell is not a registry row; every other cell name is one.
        rank = get_corpus(name).published_rank if "+" not in stem else None
        cells.append(
            Cell(
                axis="corpus",
                name=name,
                config=config,
                overrides=(
                    f"data.data_dir={data_dir}",
                    f"data.train_glob={prefix}_train_*.bin",
                    f"data.val_file={prefix}_val_000000.bin",
                    f"data.tokenizer={tokenizer}",
                ),
                corpus=stem,
                tokenizer=tokenizer,
                max_tokens=max_tokens,
                published_rank=rank,
                published_note=note,
            )
        )
    return Axis(
        name="corpus",
        description=(
            f"Six web corpora at a matched {max_tokens:,}-token budget, all tokenized by "
            f"{tok_name}. Expected published ordering: DCLM >= Ultra-FineWeb > FineWeb-Edu > "
            f"FineWeb (docs/ABLATIONS.md Axis 1). A disagreement is a result, not a bug."
        ),
        cells=tuple(cells),
    )


# --------------------------------------------------------------------------------------
# Axis 2 -- architecture knobs
# --------------------------------------------------------------------------------------

ARCH_KNOBS: tuple[tuple[str, tuple[str, ...], str], ...] = (
    ("baseline", (), "the Rung 0 default, re-run so every knob has a same-corpus reference"),
    ("value-embeds-off", ("model.use_value_embeds=false",),
     "modded-nanogpt's value embeddings; kernel-free, validated at 124M (docs/PLAN.md §3.1 item 2)"),
    ("unet-skips-off", ("model.use_unet_skips=false",),
     "U-net / MUDD residual skips; the other half of the widened-residual change"),
    ("mlp-swiglu", ("model.mlp=swiglu",),
     "SwiGLU instead of ReLU^2; a frontier default that differs from the speedrun recipe"),
    ("softcap-off", ("model.softcap=0",),
     "logit soft-capping off; frontier models mostly dropped it, the speedrun keeps it at 15"),
    ("qk-norm-off", ("model.qk_norm=false",), "QK-norm off"),
    ("muon-lr-0.035", ("train.muon_lr=0.035",), "the flat-bowl check, sqrt(2) below the default"),
    ("muon-lr-0.07", ("train.muon_lr=0.07",), "the flat-bowl check, sqrt(2) above the default"),
    ("seq-512", ("model.block_size=512",),
     "half the context at the same tokens-per-step (grad_accum doubles)"),
)


def arch_axis(
    config: str = "configs/ablations/tiny_abl.yaml",
    tokenizer: str = DEFAULT_ABL_TOKENIZER,
    corpus: str = "fineweb-edu",
    max_tokens: int = CORPUS_TOKENS,
    root: str = "data/corpora",
) -> Axis:
    """One knob at a time off the Rung 0 default, on the winning corpus."""
    tok_name = "gpt2" if tokenizer == "gpt2" else tokenizer.rstrip("/").rsplit("/", 1)[-1]
    prefix = corpus.replace("+", "_")
    data = (
        f"data.data_dir={root}/{corpus}-{tok_name}",
        f"data.train_glob={prefix}_train_*.bin",
        f"data.val_file={prefix}_val_000000.bin",
        f"data.tokenizer={tokenizer}",
    )
    cells = tuple(
        Cell(
            axis="arch",
            name=name,
            config=config,
            overrides=data + knobs,
            corpus=corpus,
            tokenizer=tokenizer,
            max_tokens=max_tokens,
            note=note,
        )
        for name, knobs, note in ARCH_KNOBS
    )
    return Axis(
        name="arch",
        description=(
            f"One architecture knob at a time off the Rung 0 default, on {corpus} at "
            f"{max_tokens:,} tokens. docs/PLAN.md §3.1 item 3: the Muon optimum is a flat bowl "
            f"(sqrt 2 in LR), so the LR cells bracket rather than sweep. Anything inside the "
            f"HellaSwag CI is re-run with a second seed before it is believed."
        ),
        cells=cells,
    )


# --------------------------------------------------------------------------------------
# Axis 3 -- tokenizer
# --------------------------------------------------------------------------------------


def tokenizer_axis(
    config: str = "configs/ablations/tiny_abl.yaml",
    tokenizer: str = DEFAULT_ABL_TOKENIZER,
    corpus: str = "fineweb-edu",
    max_bytes: int = TOKENIZER_AXIS_BYTES,
    root: str = "data/corpora",
) -> Axis:
    """GPT-2's 50,257 BPE against our own 32,768 BPE at **matched bytes**."""
    tok_name = "gpt2" if tokenizer == "gpt2" else tokenizer.rstrip("/").rsplit("/", 1)[-1]
    prefix = corpus.replace("+", "_")

    def cell(name: str, spec: str, stem: str, note: str) -> Cell:
        return Cell(
            axis="tokenizer",
            name=name,
            config=config,
            overrides=(
                f"data.data_dir={root}/{corpus}-{stem}",
                f"data.train_glob={prefix}_train_*.bin",
                f"data.val_file={prefix}_val_000000.bin",
                f"data.tokenizer={spec}",
            ),
            corpus=corpus,
            tokenizer=spec,
            max_bytes=max_bytes,
            note=note,
        )

    return Axis(
        name="tokenizer",
        description=(
            f"GPT-2's 50,257-entry BPE against our own 32,768-entry BPE on {corpus} at a matched "
            f"{max_bytes / 1e6:.0f} MB of text -- NOT matched tokens, because the better "
            f"tokenizer would otherwise be handed more text. Compared on bpb and HellaSwag "
            f"only: validation loss is not comparable across vocabularies."
        ),
        cells=(
            cell("gpt2", "gpt2", "gpt2",
                 "50,257 entries (padded to 50,304); the Rung 0 tokenizer, tiktoken"),
            cell("own-32k", tokenizer, tok_name,
                 "32,768 entries incl. the 9 specials; nanochat's vocab size and digit split"),
        ),
        hellaswag_limit=2000,
    )


# --------------------------------------------------------------------------------------
# Assembly
# --------------------------------------------------------------------------------------

AXES: tuple[str, ...] = ("tokenizer", "corpus", "arch")
"""Run order from ``docs/ABLATIONS.md`` §Ordering: lowest-cost axis first."""


def build_axes(**kwargs: Any) -> dict[str, Axis]:
    """All three axes. Keyword arguments are forwarded to the per-axis builders."""
    common = {k: v for k, v in kwargs.items() if v is not None}
    tok_kw = {k: v for k, v in common.items() if k in ("config", "tokenizer", "corpus", "root")}
    return {
        "tokenizer": tokenizer_axis(**tok_kw),
        "corpus": corpus_axis(
            **{k: v for k, v in common.items() if k in ("config", "tokenizer", "max_tokens", "root")}
        ),
        "arch": arch_axis(
            **{
                k: v
                for k, v in common.items()
                if k in ("config", "tokenizer", "corpus", "max_tokens", "root")
            }
        ),
    }


def axis_names() -> list[str]:
    return list(AXES)


def axis(name: str, **kwargs: Any) -> Axis:
    """One axis by name."""
    built = build_axes(**kwargs)
    try:
        return built[name]
    except KeyError:
        raise KeyError(f"unknown axis {name!r}; axes are {list(AXES)}") from None


# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="r52.ablate.matrix",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--axis", default=None, choices=list(AXES), help="default: all three")
    p.add_argument("--config", default=None, help="override the base config for every cell")
    p.add_argument("--tokenizer", default=None, help=f"default: {DEFAULT_ABL_TOKENIZER}")
    p.add_argument("--corpus", default=None, help="corpus for the arch and tokenizer axes")
    p.add_argument("--json", action="store_true", help="dump the matrix as JSON")
    p.add_argument("--slugs", action="store_true",
                   help="print one cell slug per line (what scripts/ablate.sh reads)")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    built = build_axes(config=args.config, tokenizer=args.tokenizer, corpus=args.corpus)
    names = [args.axis] if args.axis else list(AXES)
    if args.slugs:
        for n in names:
            for c in built[n].cells:
                print(c.slug)
        return 0
    if args.json:
        print(json.dumps({n: built[n].to_dict() for n in names}, indent=2))
        return 0
    for n in names:
        ax = built[n]
        print(f"== {ax.name} ({len(ax)} cells) " + "=" * max(0, 60 - len(ax.name)))
        print(f"   {ax.description}")
        print(f"   CORE subset: {','.join(ax.core_tasks)}   HellaSwag limit: {ax.hellaswag_limit}")
        for c in ax.cells:
            budget = f"{c.max_tokens:,} tok" if c.max_tokens else f"{c.max_bytes / 1e6:.0f} MB text"
            print(f"   - {c.name:<20} {budget:>14}  -> {c.result_path}")
            print(f"     python -m r52.train {' '.join(c.train_argv())}")
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
