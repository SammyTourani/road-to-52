#!/usr/bin/env python
# Copyright 2026 The road-to-52 authors.
# SPDX-License-Identifier: Apache-2.0
"""Build the midtraining mixture as llm.c ``.bin`` shards.

    python scripts/prepare_midtrain_data.py configs/posttrain/midtrain_nano.yaml
    python scripts/prepare_midtrain_data.py configs/posttrain/midtrain_tiny.yaml \
        -o total_tokens=1000000 -o out_dir=data/midtrain-tiny
    python scripts/prepare_midtrain_data.py --list data/midtrain

``docs/PLAN.md`` §3.1 item 4: midtraining is a short, LR-decaying, high-quality stage that
deliberately seeds instruction-following and thinking data into the **base** model.  So the
mixture is (i) FineWeb (the pretraining distribution, to avoid forgetting), (ii) chat data
rendered with :mod:`r52.chat_template` -- the same token layout SFT and inference use --
and (iii) a slice of math.  Weights live in the YAML.

Output is byte-compatible with ``r52.data``'s reader (llm.c format: 256 int32 header words
``[20240520, 1, ntok, 0...]`` then ``ntok`` uint16 tokens), so ``python -m r52.train`` can
consume it with nothing but a different ``data:`` block.

Documents are interleaved by sampling a source per document from the normalised weights, so
the mixture is uniform along the shard rather than blocked by source.  The realised token
share of each source is printed and stored in ``<out_dir>/mixture.json``.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from r52.chat_template import EOT, render
from r52.data import HEADER_INTS, MAGIC, VERSION, load_shard, read_shard_header, shard_paths
from r52.posttrain.config import MidtrainConfig, MixtureSource, load_midtrain_config, override
from r52.tokenizer import GPT2Tokenizer


def write_bin(path: Path, tokens: np.ndarray) -> int:
    """Write ``tokens`` (uint16) as an llm.c ``.bin`` shard. Returns the token count."""
    path.parent.mkdir(parents=True, exist_ok=True)
    header = np.zeros(HEADER_INTS, dtype=np.int32)
    header[0], header[1], header[2] = MAGIC, VERSION, int(tokens.size)
    tmp = path.with_suffix(path.suffix + ".partial")
    with open(tmp, "wb") as fh:
        fh.write(header.tobytes())
        fh.write(np.asarray(tokens, dtype=np.uint16).tobytes())
    tmp.replace(path)
    return int(tokens.size)


# --------------------------------------------------------------------------------------
# Sources
# --------------------------------------------------------------------------------------


class BinSource:
    """Documents read straight out of existing llm.c shards (already GPT-2 tokens).

    ``.bin`` shards carry no document boundaries, so this yields fixed 2048-token chunks;
    for the pretraining-distribution component of a midtrain mix that is exactly right.
    """

    chunk = 2048

    def __init__(self, src: MixtureSource) -> None:
        self.name = src.name
        self.paths = shard_paths(src.data_dir, src.glob)
        if not self.paths:
            raise FileNotFoundError(
                f"source {src.name!r}: no shards matching {src.glob!r} in {src.data_dir!r}. "
                f"Run: python scripts/prepare_data.py --train-shards 1"
            )
        self.total = sum(read_shard_header(p) for p in self.paths)
        self._i = 0
        self._arr: np.ndarray | None = None
        self._off = 0

    def __iter__(self):
        return self

    def __next__(self) -> np.ndarray:
        if self._arr is None:
            self._arr = load_shard(self.paths[self._i])
            self._off = 0
        if self._off + self.chunk > self._arr.size:
            self._i = (self._i + 1) % len(self.paths)
            self._arr = load_shard(self.paths[self._i])
            self._off = 0
        out = np.asarray(self._arr[self._off : self._off + self.chunk], dtype=np.uint16)
        self._off += self.chunk
        return out


class HFSource:
    """Documents from a Hugging Face dataset, tokenized to uint16.

    ``kind='text'`` -> ``[<|endoftext|>] + gpt2(row[text_key])`` (FineWeb's own convention).
    ``kind='chat'`` -> :func:`r52.chat_template.render` of ``row[messages_key]``, so the
    conversation carries ``<|bos|>``/``<|user_start|>``/... exactly as SFT will.
    """

    def __init__(self, src: MixtureSource, tokenizer: GPT2Tokenizer) -> None:
        from datasets import load_dataset

        self.name = src.name
        self.src = src
        self.tok = tokenizer
        kwargs = {"split": src.split, "streaming": src.streaming}
        ds = load_dataset(src.repo_id, src.subset, **kwargs) if src.subset else load_dataset(
            src.repo_id, **kwargs
        )
        self._it = iter(ds)
        self._ds = ds
        self.total = 0

    def _restart(self) -> None:
        self._it = iter(self._ds)

    def __iter__(self):
        return self

    def __next__(self) -> np.ndarray:
        for _ in range(1000):  # skip unusable rows, but never spin forever
            try:
                row = next(self._it)
            except StopIteration:
                self._restart()
                row = next(self._it)
            if self.src.kind == "chat":
                msgs = row.get(self.src.messages_key)
                if not msgs:
                    continue
                clean = [
                    {"role": str(m.get("role", "")).lower(), "content": str(m.get("content", ""))}
                    for m in msgs
                ]
                if any(m["role"] not in ("system", "user", "assistant") for m in clean):
                    continue
                ids, _ = render(clean, self.tok, max_tokens=self.src.max_seq or None)
            else:
                text = row.get(self.src.text_key)
                if not text:
                    continue
                ids = [EOT, *self.tok.encode_ordinary(str(text))]
                if self.src.max_seq:
                    ids = ids[: self.src.max_seq]
            if len(ids) < 8:
                continue
            return np.asarray(ids, dtype=np.uint16)
        raise RuntimeError(f"source {self.name!r}: 1000 consecutive unusable rows")


def build_sources(cfg: MidtrainConfig, tokenizer: GPT2Tokenizer) -> list:
    """Instantiate one iterator per configured source."""
    out = []
    for s in cfg.sources:
        out.append(BinSource(s) if s.kind == "bin" else HFSource(s, tokenizer))
    return out


# --------------------------------------------------------------------------------------
# Mixing
# --------------------------------------------------------------------------------------


def mix(cfg: MidtrainConfig, sources: list, n_tokens: int, rng: np.random.Generator,
        realised: list[int]) -> np.ndarray:
    """Draw documents until ``n_tokens`` tokens are collected; returns a uint16 array.

    The weights are shares of **tokens**, so the source cannot simply be sampled with
    probability ``weight``: documents differ in length by an order of magnitude (a FineWeb
    chunk is a fixed 2,048 tokens, a smol-smoltalk conversation averages ~600), and
    per-document sampling delivers token shares proportional to ``weight * mean_length``.
    Measured on the 1M-token tiny mix before this was fixed: 81/12/6 against a 60/30/10
    target.

    Instead each draw samples among the sources in proportion to their current **token
    deficit** ``target_i * (have + 1) - realised_i``, which is self-correcting: a source
    that overshoots stops being drawn until the others catch up.  Still stochastic (so the
    shard is not a fixed repeating pattern), but it converges on the target share.
    """
    weights = np.asarray(cfg.weights(), dtype=np.float64)
    parts: list[np.ndarray] = []
    have = 0
    got = np.asarray(realised, dtype=np.float64)
    while have < n_tokens:
        deficit = np.clip(weights * (got.sum() + 1.0) - got, 0.0, None)
        probs = deficit / deficit.sum() if deficit.sum() > 0 else weights
        j = int(rng.choice(len(sources), p=probs))
        doc = next(sources[j])
        take = min(int(doc.size), n_tokens - have)
        parts.append(doc[:take])
        realised[j] += take
        got[j] += take
        have += take
    return np.concatenate(parts) if parts else np.zeros(0, dtype=np.uint16)


def list_dir(path: str | Path) -> int:
    """Print the shards in a midtrain directory."""
    d = Path(path)
    if not d.exists():
        print(f"{d} does not exist")
        return 1
    total = 0
    for f in sorted(d.glob("*.bin")):
        n = read_shard_header(f)
        total += n
        print(f"{f.name:32s} {n:>12,} tokens  {f.stat().st_size / 1e6:8.1f} MB")
    print(f"{'TOTAL':32s} {total:>12,} tokens")
    mixture = d / "mixture.json"
    if mixture.exists():
        print(mixture.read_text())
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("config", nargs="?", help="midtraining mixture YAML (configs/posttrain/)")
    p.add_argument("--list", metavar="DIR", default=None, help="list an existing mix and exit")
    p.add_argument("-o", "--override", action="append", default=[], metavar="KEY=VALUE",
                   help="config override, e.g. -o total_tokens=1000000")
    args = p.parse_args(argv)

    if args.list:
        return list_dir(args.list)
    if not args.config:
        p.error("a config is required unless --list is given")

    cfg = load_midtrain_config(args.config)
    override(cfg, args.override)
    if not cfg.sources:
        raise SystemExit(f"{args.config}: the mixture has no sources")

    tok = GPT2Tokenizer()
    sources = build_sources(cfg, tok)
    rng = np.random.default_rng(cfg.seed)
    out = Path(cfg.out_dir)
    realised = [0] * len(sources)

    if cfg.val_tokens > 0:
        val = mix(cfg, sources, cfg.val_tokens, rng, realised)
        n = write_bin(out / f"{cfg.prefix}_val_000000.bin", val)
        print(f"val   {n:>12,} tokens -> {out / (cfg.prefix + '_val_000000.bin')}", flush=True)

    written = 0
    shard = 1
    while written < cfg.total_tokens:
        want = min(cfg.shard_tokens, cfg.total_tokens - written)
        toks = mix(cfg, sources, want, rng, realised)
        path = out / f"{cfg.prefix}_train_{shard:06d}.bin"
        n = write_bin(path, toks)
        written += n
        print(f"shard {n:>12,} tokens -> {path}", flush=True)
        shard += 1

    total_realised = sum(realised) or 1
    report = {
        "config": cfg.to_dict(),
        "tokens_written": written,
        "val_tokens": cfg.val_tokens,
        "realised_share": {
            s.name: round(realised[i] / total_realised, 4) for i, s in enumerate(cfg.sources)
        },
        "realised_tokens": {s.name: realised[i] for i, s in enumerate(cfg.sources)},
        "target_share": {s.name: round(w, 4) for s, w in zip(cfg.sources, cfg.weights(), strict=True)},
    }
    (out / "mixture.json").write_text(json.dumps(report, indent=2))
    print(f"\nrealised mixture: {report['realised_share']}")
    print(f"target  mixture: {report['target_share']}")
    print(f"wrote {written:,} train tokens in {shard - 1} shard(s) under {out}")
    return 0


if __name__ == "__main__":
    rc = main()
    # PyArrow's streaming parquet reader deadlocks in `ThreadPool::Shutdown` at interpreter
    # exit when a `datasets` streaming iterator is abandoned mid-file (measured 2026-09-13:
    # the work finishes, the files are written, and the process then hangs forever at 0% CPU
    # inside `arrow::internal::ThreadPool::Shutdown`).  Everything this script owns is
    # flushed by now, so leave without running interpreter teardown.
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(rc)
