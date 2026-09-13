# Copyright 2026 The road-to-52 authors.
# SPDX-License-Identifier: Apache-2.0
"""The corpus registry and its streaming document iterator.

``corpora.yaml`` names the six candidates of ``docs/ABLATIONS.md`` Axis 1 and pins each to a
dataset **revision**, so a slice prepared today and a slice prepared in six months read the
same bytes.  This module turns a registry row into a stream of ``(text, n_bytes)`` documents
using ``datasets``' streaming reader over ``hf://`` URLs -- nothing is ever downloaded whole;
a parquet file is read row-group by row-group and a ``.jsonl.zst`` shard block by block, and
the stream is abandoned as soon as the caller has enough tokens.

Two things here are not obvious and are the reason this file exists rather than a one-line
``load_dataset`` call:

**File order matters.**  ``allenai/dolma3_mix-150B-1025`` is laid out by *source and topic*
(66 directories).  Reading its files in sorted order gives you
``common_crawl-adult_content`` and nothing else.  Rows therefore carry ``file_order:
shuffled`` plus an ``interleave`` count, and :func:`iter_documents` reads that many files
round-robin so a small slice samples the mix.

**Formats differ.**  ``mlfoundations/dclm-baseline-1.0`` ships ``.jsonl.zst``, not parquet
(``docs/ABLATIONS.md`` says "first parquet shards"; the slice is right, the format is not),
and ``openbmb/Ultra-FineWeb``'s text column is ``content``, not ``text``.  Both were read out
of the live files on 2026-09-13, not out of a dataset card.
"""

from __future__ import annotations

import random
from collections.abc import Iterator
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

__all__ = [
    "REGISTRY_PATH",
    "Corpus",
    "Document",
    "corpus_names",
    "get_corpus",
    "iter_documents",
    "load_registry",
    "resolve_files",
]

REGISTRY_PATH = Path(__file__).with_name("corpora.yaml")


@dataclass(frozen=True)
class Document:
    """One document from a corpus stream."""

    text: str
    n_bytes: int
    """UTF-8 length of ``text``."""


@dataclass(frozen=True)
class Corpus:
    """One row of ``corpora.yaml``."""

    name: str
    repo_id: str
    revision: str
    format: str
    path_prefix: str
    text_key: str
    license: str
    config: str | None = None
    interleave: int = 1
    file_order: str = "sorted"
    file_seed: int = 1337
    min_chars: int = 64
    published_rank: int | None = None
    notes: str = ""
    extra: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.format not in ("parquet", "json"):
            raise ValueError(f"{self.name}: unknown format {self.format!r} (parquet|json)")
        if self.file_order not in ("sorted", "shuffled"):
            raise ValueError(f"{self.name}: unknown file_order {self.file_order!r}")
        if self.interleave < 1:
            raise ValueError(f"{self.name}: interleave must be >= 1")

    def url(self, repo_path: str) -> str:
        """``hf://`` URL of one repo file, pinned to :attr:`revision`."""
        return f"hf://datasets/{self.repo_id}@{self.revision}/{repo_path}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "repo_id": self.repo_id,
            "revision": self.revision,
            "config": self.config,
            "format": self.format,
            "path_prefix": self.path_prefix,
            "text_key": self.text_key,
            "license": self.license,
            "interleave": self.interleave,
            "file_order": self.file_order,
            "file_seed": self.file_seed,
            "min_chars": self.min_chars,
            "published_rank": self.published_rank,
            "notes": " ".join(self.notes.split()),
        }


@lru_cache(maxsize=4)
def load_registry(path: str | Path = REGISTRY_PATH) -> dict[str, Corpus]:
    """Parse ``corpora.yaml`` into ``{name: Corpus}``."""
    raw = yaml.safe_load(Path(path).read_text())
    known = {f.name for f in Corpus.__dataclass_fields__.values()} - {"name", "extra"}
    out: dict[str, Corpus] = {}
    for name, row in (raw.get("corpora") or {}).items():
        row = dict(row or {})
        extra = {k: row.pop(k) for k in list(row) if k not in known}
        out[name] = Corpus(name=name, extra=extra, **row)
    if not out:
        raise ValueError(f"{path}: no corpora defined")
    return out


def corpus_names(path: str | Path = REGISTRY_PATH) -> list[str]:
    """Registered corpus names, in registry order."""
    return list(load_registry(path))


def get_corpus(name: str, path: str | Path = REGISTRY_PATH) -> Corpus:
    """Look one corpus up by name, with a helpful error listing the alternatives."""
    reg = load_registry(path)
    try:
        return reg[name]
    except KeyError:
        raise KeyError(f"unknown corpus {name!r}; registry has {sorted(reg)}") from None


# --------------------------------------------------------------------------------------
# Files
# --------------------------------------------------------------------------------------


def resolve_files(corpus: Corpus, limit: int = 0) -> list[str]:
    """Repo-relative paths of ``corpus``'s data files, in reading order.

    Sorted for a corpus whose shards are already globally shuffled; seeded-shuffled for one
    that is laid out by source (see :class:`Corpus`).  ``limit`` caps the list -- one file is
    hundreds of millions of tokens, so a slice never needs many.
    """
    from huggingface_hub import list_repo_files

    files = [
        f
        for f in list_repo_files(corpus.repo_id, repo_type="dataset", revision=corpus.revision)
        if f.startswith(corpus.path_prefix) and f.endswith((".parquet", ".jsonl.zst", ".json.zst"))
    ]
    if not files:
        raise FileNotFoundError(
            f"{corpus.name}: no data files under {corpus.path_prefix!r} in "
            f"{corpus.repo_id}@{corpus.revision[:8]}"
        )
    files.sort()
    if corpus.file_order == "shuffled":
        random.Random(corpus.file_seed).shuffle(files)
    return files[:limit] if limit and limit > 0 else files


# --------------------------------------------------------------------------------------
# Streaming
# --------------------------------------------------------------------------------------


def _row_stream(corpus: Corpus, repo_path: str):
    """A streaming ``datasets`` iterator over one repo file."""
    from datasets import load_dataset

    builder = "parquet" if corpus.format == "parquet" else "json"
    ds = load_dataset(
        builder,
        data_files=[corpus.url(repo_path)],
        streaming=True,
        split="train",
    )
    return iter(ds)


def iter_documents(
    corpus: Corpus,
    files: list[str] | None = None,
    max_files: int = 0,
    on_file: Any = None,
) -> Iterator[Document]:
    """Yield documents from ``corpus``, round-robin over ``corpus.interleave`` files.

    Args:
        corpus: a registry row.
        files: an explicit file list (default: :func:`resolve_files`).
        max_files: cap on how many files may be opened (0 = no cap).
        on_file: optional ``callable(repo_path)`` invoked the first time a file is opened;
            used by ``scripts/prepare_corpus.py`` to record exactly which shards a slice came
            from.

    The iterator is infinite only in the sense that it stops when the files run out -- the
    caller is expected to abandon it as soon as it has enough tokens, which is what keeps the
    download small.  Abandoning a ``datasets`` parquet stream is also what deadlocks PyArrow
    at interpreter exit (``docs/DEVIATIONS.md`` P6), so every CLI that calls this must leave
    through ``os._exit``.
    """
    pending = list(files if files is not None else resolve_files(corpus))
    if max_files > 0:
        pending = pending[:max_files]
    if not pending:
        return
    width = min(corpus.interleave, len(pending))
    live: list[tuple[str, Iterator[dict]]] = []
    key, min_chars = corpus.text_key, corpus.min_chars

    def _open_next() -> bool:
        if not pending:
            return False
        path = pending.pop(0)
        if on_file is not None:
            on_file(path)
        live.append((path, _row_stream(corpus, path)))
        return True

    while len(live) < width and _open_next():
        pass

    i = 0
    while live:
        i %= len(live)
        _path, it = live[i]
        try:
            row = next(it)
        except StopIteration:
            live.pop(i)
            _open_next()
            continue
        text = row.get(key)
        if not isinstance(text, str) or len(text) < min_chars:
            i += 1
            continue
        yield Document(text, len(text.encode("utf-8")))
        i += 1
