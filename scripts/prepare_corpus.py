#!/usr/bin/env python
# Copyright 2026 The road-to-52 authors.
# SPDX-License-Identifier: Apache-2.0
"""Stream one of the ablation corpora into llm.c ``.bin`` shards.

    python scripts/prepare_corpus.py --corpus fineweb-edu --tokens 2000000 --tokenizer gpt2
    python scripts/prepare_corpus.py --corpus dclm --tokens 100000000 \
        --tokenizer data/tokenizers/fweb32k
    python scripts/prepare_corpus.py --corpus fineweb-edu --blend finepdfs-edu:0.25 \
        --tokens 100000000
    python scripts/prepare_corpus.py --list data/corpora/fineweb-edu-gpt2
    python scripts/prepare_corpus.py --corpora            # what is registered

Output (``data/corpora/<corpus>-<tokenizer>/`` by default)::

    <corpus>_val_000000.bin     held out FIRST, before any training token is written
    <corpus>_train_000001.bin   ... up to --tokens
    manifest.json               dataset id + revision, config, the exact shard files read,
                                rows, tokens, bytes, bytes/token, tokenizer, date

The shards are byte-compatible with :mod:`r52.data`'s reader (256 int32 header words
``[20240520, 1, ntok, 0...]`` then ``ntok`` uint16 tokens), so ``python -m r52.train`` reads
them with nothing but a different ``data:`` block -- which ``manifest.json`` prints for you.

Documents are separated by the tokenizer's ``<|endoftext|>``, FineWeb's own convention and
the one ``kjj0/fineweb10B-gpt2`` uses, so a corpus prepared here and the Rung 0 shards differ
only in their content.

Nothing is downloaded whole: ``datasets`` streams a parquet file row-group by row-group and a
``.jsonl.zst`` shard block by block, and the stream is abandoned the moment ``--tokens`` is
reached.  Abandoning it is also what deadlocks PyArrow at interpreter exit
(``docs/DEVIATIONS.md`` P6), so this script leaves through ``os._exit``.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from r52.ablate.corpora import Corpus, Document, corpus_names, get_corpus, iter_documents
from r52.data import HEADER_INTS, MAGIC, VERSION, read_shard_header
from r52.tokenizer_train import load_tokenizer, tokenizer_vocab_size

DEFAULT_ROOT = "data/corpora"
DEFAULT_SHARD_TOKENS = 50_000_000


# --------------------------------------------------------------------------------------
# Shard writing
# --------------------------------------------------------------------------------------


class ShardWriter:
    """Append uint16 tokens to an llm.c ``.bin`` shard, patching ``ntok`` on close.

    Streaming rather than buffering: a 50M-token shard is 100 MB, and this machine is also
    running a multi-day training job, so the tokens are written as they are produced and
    never all held in memory at once.
    """

    def __init__(self, path: Path) -> None:
        self.path = path
        self.tmp = path.with_suffix(path.suffix + ".partial")
        self.tmp.parent.mkdir(parents=True, exist_ok=True)
        self.fh = self.tmp.open("wb")
        self.fh.write(np.zeros(HEADER_INTS, dtype=np.int32).tobytes())
        self.n = 0

    def write(self, tokens: np.ndarray) -> int:
        arr = np.asarray(tokens, dtype=np.uint16)
        self.fh.write(arr.tobytes())
        self.n += int(arr.size)
        return int(arr.size)

    def close(self) -> int:
        header = np.zeros(HEADER_INTS, dtype=np.int32)
        header[0], header[1], header[2] = MAGIC, VERSION, self.n
        self.fh.seek(0)
        self.fh.write(header.tobytes())
        self.fh.close()
        self.tmp.replace(self.path)
        return self.n


# --------------------------------------------------------------------------------------
# Token stream
# --------------------------------------------------------------------------------------


class Tally:
    """Running counts for one source (documents, text bytes, tokens)."""

    def __init__(self, name: str) -> None:
        self.name = name
        self.rows = 0
        self.text_bytes = 0
        self.tokens = 0
        self.files: list[str] = []

    def to_dict(self) -> dict[str, object]:
        return {
            "rows": self.rows,
            "text_bytes": self.text_bytes,
            "tokens": self.tokens,
            "bytes_per_token": round(self.text_bytes / max(1, self.tokens), 6),
            "files_read": self.files,
        }


def _encode(doc: Document, tok, eot: int) -> np.ndarray:
    ids = [eot, *tok.encode_ordinary(doc.text)]
    return np.asarray(ids, dtype=np.uint16)


def single_stream(corpus: Corpus, tok, tally: Tally, max_files: int) -> Iterator[np.ndarray]:
    """Encoded documents from one corpus, tallying rows/bytes/tokens as they go."""
    eot = tok.eot
    for doc in iter_documents(corpus, max_files=max_files, on_file=tally.files.append):
        arr = _encode(doc, tok, eot)
        tally.rows += 1
        tally.text_bytes += doc.n_bytes
        tally.tokens += int(arr.size)
        yield arr


def blended_stream(
    streams: list[Iterator[np.ndarray]],
    weights: list[float],
    seed: int,
) -> Iterator[np.ndarray]:
    """Interleave several encoded streams so realised **token** shares match ``weights``.

    Sampling a source per document with probability ``weight`` delivers shares proportional
    to ``weight * mean_document_length``, and these corpora differ by 2-3x in document length
    (``docs/DEVIATIONS.md`` P5 measured 81/12/6 against a 60/30/10 target on the midtrain
    mix).  Each draw therefore samples in proportion to the source's current token *deficit*,
    which is self-correcting.
    """
    rng = np.random.default_rng(seed)
    w = np.asarray(weights, dtype=np.float64)
    w = w / w.sum()
    got = np.zeros(len(streams), dtype=np.float64)
    alive = [True] * len(streams)
    while any(alive):
        deficit = np.where(alive, np.clip(w * (got.sum() + 1.0) - got, 0.0, None), 0.0)
        probs = deficit / deficit.sum() if deficit.sum() > 0 else np.asarray(alive, dtype=float) / sum(alive)
        j = int(rng.choice(len(streams), p=probs))
        try:
            arr = next(streams[j])
        except StopIteration:
            alive[j] = False
            continue
        got[j] += arr.size  # per-source rows/bytes/tokens are tallied inside each stream
        yield arr


# --------------------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------------------


def write_split(
    stream: Iterator[np.ndarray],
    out: Path,
    prefix: str,
    kind: str,
    n_tokens: int,
    shard_tokens: int,
    quiet: bool = False,
) -> list[dict[str, object]]:
    """Drain ``stream`` into ``.bin`` shards until ``n_tokens`` tokens are written."""
    shards: list[dict[str, object]] = []
    written = 0
    index = 0 if kind == "val" else 1
    writer: ShardWriter | None = None
    t0 = time.time()
    while written < n_tokens:
        if writer is None:
            path = out / f"{prefix}_{kind}_{index:06d}.bin"
            writer = ShardWriter(path)
        try:
            arr = next(stream)
        except StopIteration:
            break
        take = min(int(arr.size), n_tokens - written)
        written += writer.write(arr[:take])
        if writer.n >= shard_tokens or written >= n_tokens:
            n = writer.close()
            shards.append({"file": writer.path.name, "tokens": n})
            if not quiet:
                rate = written / max(1e-9, time.time() - t0)
                print(f"  {writer.path.name}  {n:>12,} tokens  ({rate:,.0f} tok/s)", flush=True)
            writer = None
            index += 1
    if writer is not None:
        n = writer.close()
        shards.append({"file": writer.path.name, "tokens": n})
        if not quiet:
            print(f"  {writer.path.name}  {n:>12,} tokens", flush=True)
    return shards


def list_dir(path: str | Path) -> int:
    """Print the shards and manifest of a prepared corpus directory."""
    d = Path(path)
    if not d.exists():
        print(f"{d} does not exist")
        return 1
    total = 0
    for f in sorted(d.glob("*.bin")):
        n = read_shard_header(f)
        total += n
        print(f"{f.name:40s} {n:>13,} tokens  {f.stat().st_size / 1e6:8.1f} MB")
    print(f"{'TOTAL':40s} {total:>13,} tokens")
    manifest = d / "manifest.json"
    if manifest.is_file():
        print()
        print(manifest.read_text())
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--corpus", default=None, help=f"one of {corpus_names()}")
    p.add_argument("--blend", action="append", default=[], metavar="NAME:WEIGHT",
                   help="blend another corpus in at this token share, e.g. finepdfs-edu:0.25")
    p.add_argument("--tokenizer", default="gpt2", help="'gpt2' or a data/tokenizers/<name> directory")
    p.add_argument("--tokens", type=float, default=100e6, help="training tokens (default: %(default)s)")
    p.add_argument("--val-tokens", type=float, default=0,
                   help="validation tokens, held out FIRST (default: min(tokens/20, 2,000,000))")
    p.add_argument("--shard-tokens", type=int, default=DEFAULT_SHARD_TOKENS)
    p.add_argument("--out", default=None, help=f"output directory (default: {DEFAULT_ROOT}/<corpus>-<tok>)")
    p.add_argument("--root", default=DEFAULT_ROOT)
    p.add_argument("--max-files", type=int, default=0, help="cap on repo files opened (0 = no cap)")
    p.add_argument("--save-text", default=None, help="also write the raw streamed text here (UTF-8)")
    p.add_argument("--text-only", action="store_true", help="write only --save-text, no .bin shards")
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--list", metavar="DIR", default=None, help="list a prepared corpus and exit")
    p.add_argument("--corpora", action="store_true", help="print the registry and exit")
    p.add_argument("--quiet", action="store_true")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    if args.list:
        return list_dir(args.list)
    if args.corpora:
        for name in corpus_names():
            c = get_corpus(name)
            print(f"{name:<16} {c.repo_id:<40} {c.config or '-':<14} {c.format:<8} "
                  f"text={c.text_key:<8} {c.license}")
        return 0
    if not args.corpus:
        print("--corpus is required (see --corpora)", file=sys.stderr)
        return 2

    corpus = get_corpus(args.corpus)
    blends: list[tuple[Corpus, float]] = []
    for spec in args.blend:
        name, _, weight = spec.partition(":")
        blends.append((get_corpus(name), float(weight or 0.25)))
    if sum(w for _, w in blends) >= 1.0:
        print("--blend weights must sum to < 1 (the base corpus takes the rest)", file=sys.stderr)
        return 2

    tok = load_tokenizer(args.tokenizer)
    vocab = tokenizer_vocab_size(args.tokenizer)
    if vocab > 65_536:
        print(f"tokenizer vocab {vocab} does not fit the uint16 .bin format", file=sys.stderr)
        return 2
    tok_name = "gpt2" if args.tokenizer == "gpt2" else Path(args.tokenizer).name

    n_train = int(args.tokens)
    n_val = int(args.val_tokens) if args.val_tokens else min(n_train // 20, 2_000_000)
    shard_tokens = max(1, int(args.shard_tokens))

    slug = args.corpus if not blends else (
        args.corpus + "".join(f"+{c.name}{round(100 * w)}" for c, w in blends)
    )
    out = Path(args.out) if args.out else Path(args.root) / f"{slug}-{tok_name}"
    prefix = slug.replace("+", "_")

    sources = [corpus, *[c for c, _ in blends]]
    weights = [1.0 - sum(w for _, w in blends), *[w for _, w in blends]]
    tallies = [Tally(c.name) for c in sources]

    if not args.quiet:
        print(f"[corpus] {slug} -> {out}")
        for c, w, in zip(sources, weights, strict=True):
            print(f"    {c.name:<16} {w:5.2f}  {c.repo_id}@{c.revision[:8]}  "
                  f"{c.config or '-'}  text={c.text_key}  ({c.license})")
        print(f"    tokenizer {tok_name} (vocab {vocab:,}), val {n_val:,} + train {n_train:,} tokens")

    streams = [single_stream(c, tok, t, args.max_files) for c, t in zip(sources, tallies, strict=True)]
    stream = streams[0] if len(streams) == 1 else blended_stream(streams, weights, args.seed)

    text_fh = None
    if args.save_text:
        Path(args.save_text).parent.mkdir(parents=True, exist_ok=True)
        text_fh = Path(args.save_text).open("w", encoding="utf-8")  # noqa: SIM115 - spans the run
        stream = _tee_text(stream, tok, text_fh)

    t0 = time.time()
    val_shards: list[dict[str, object]] = []
    train_shards: list[dict[str, object]] = []
    if args.text_only:
        budget = n_val + n_train
        got = 0
        for arr in stream:
            got += int(arr.size)
            if got >= budget:
                break
    else:
        if n_val > 0:
            if not args.quiet:
                print("  -- validation (held out first) --", flush=True)
            val_shards = write_split(stream, out, prefix, "val", n_val, n_val, quiet=args.quiet)
        if not args.quiet:
            print("  -- train --", flush=True)
        train_shards = write_split(stream, out, prefix, "train", n_train, shard_tokens, quiet=args.quiet)
    wall = time.time() - t0
    if text_fh is not None:
        text_fh.close()

    total_tokens = sum(int(s["tokens"]) for s in val_shards + train_shards)
    total_rows = sum(t.rows for t in tallies)
    total_bytes = sum(t.text_bytes for t in tallies)
    produced = sum(t.tokens for t in tallies)

    manifest = {
        "corpus": slug,
        "created": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "prepared_by": "scripts/prepare_corpus.py",
        "sources": [
            {**c.to_dict(), "weight": w, **t.to_dict()}
            for c, w, t in zip(sources, weights, tallies, strict=True)
        ],
        "tokenizer": {
            "spec": args.tokenizer,
            "name": tok_name,
            "vocab_size": vocab,
            "eot_id": int(tok.eot),
            "document_separator": "<|endoftext|> prefixed to every document",
        },
        "rows": total_rows,
        "tokens": total_tokens,
        "tokens_encoded": produced,
        "text_bytes": total_bytes,
        "bytes_per_token": round(total_bytes / max(1, produced), 6),
        "bytes_per_token_note": (
            "text bytes / tokens produced, document separators included in the token count. "
            "r52.tokenizer.val_bytes_per_token decodes instead, so it also counts the 13 "
            "literal bytes of each '<|endoftext|>' and reads slightly higher."
        ),
        "val": {"file": val_shards[0]["file"] if val_shards else None,
                "tokens": sum(int(s["tokens"]) for s in val_shards)},
        "train": {"shards": train_shards,
                  "tokens": sum(int(s["tokens"]) for s in train_shards)},
        "wall_clock_s": round(wall, 2),
        "data_config": {
            "source": "fineweb",
            "data_dir": str(out),
            "train_glob": f"{prefix}_train_*.bin",
            "val_file": f"{prefix}_val_000000.bin",
            "tokenizer": args.tokenizer,
        },
        "command": "python " + " ".join([Path(__file__).name, *(argv or sys.argv[1:])]),
    }
    if not args.text_only:
        out.mkdir(parents=True, exist_ok=True)
        (out / "manifest.json").write_text(json.dumps(manifest, indent=2))

    if not args.quiet:
        print()
        print(f"rows {total_rows:,} | tokens {total_tokens:,} | text {total_bytes / 1e6:.2f} MB | "
              f"{manifest['bytes_per_token']:.4f} bytes/token | {wall:.1f} s")
        if not args.text_only:
            print(f"manifest {out / 'manifest.json'}")
    return 0


def _tee_text(stream: Iterator[np.ndarray], tok, fh) -> Iterator[np.ndarray]:
    """Pass tokens through while writing the decoded text of each document to ``fh``."""
    for arr in stream:
        fh.write(tok.decode(arr[1:].tolist()))
        fh.write("\n\n")
        yield arr


if __name__ == "__main__":
    rc = main()
    # PyArrow's streaming reader deadlocks in `ThreadPool::Shutdown` when a `datasets`
    # streaming iterator is abandoned mid-file (docs/DEVIATIONS.md P6). Everything this
    # script owns is on disk by now, so leave without interpreter teardown.
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(rc)
