# Copyright 2026 The road-to-52 authors.
# SPDX-License-Identifier: Apache-2.0
"""Token loading for road-to-52.

The training corpus is ``kjj0/fineweb10B-gpt2`` -- FineWeb re-tokenized with the GPT-2
BPE and stored in llm.c's ``.bin`` shard format (the same files modded-nanogpt trains on):

    bytes 0..1023   : 256 little-endian int32 header words
                      [0] magic   = 20240520
                      [1] version = 1
                      [2] ntok    = number of uint16 tokens that follow
                      [3:] zero padding
    bytes 1024..    : ``ntok`` little-endian uint16 tokens

Shards are memory-mapped, so a 100 MB shard costs no resident memory until touched.

A :class:`Cursor` (shard index + token offset) is the complete state of the training
stream; it round-trips through a checkpoint so ``--resume`` replays the exact same
token order.  All offsets are in **tokens**, never bytes.
"""

from __future__ import annotations

import glob
import os
from dataclasses import dataclass
from pathlib import Path

import mlx.core as mx
import numpy as np

__all__ = [
    "HEADER_INTS",
    "MAGIC",
    "VERSION",
    "Cursor",
    "TokenStream",
    "ValLoader",
    "load_shard",
    "make_train_stream",
    "make_val_loader",
    "read_shard_header",
    "shard_paths",
]

HEADER_INTS = 256
"""int32 words in an llm.c ``.bin`` header (1024 bytes)."""
MAGIC = 20240520
VERSION = 1
_HEADER_BYTES = HEADER_INTS * 4


@dataclass
class Cursor:
    """Position in the training stream. ``shard`` indexes ``TokenStream.paths``."""

    shard: int = 0
    offset: int = 0
    """Token offset inside the shard (0 = first token after the header)."""
    epoch: int = 0
    """How many times the shard list has wrapped around."""

    def to_dict(self) -> dict[str, int]:
        return {"shard": int(self.shard), "offset": int(self.offset), "epoch": int(self.epoch)}

    @staticmethod
    def from_dict(d: dict[str, int]) -> Cursor:
        return Cursor(shard=int(d["shard"]), offset=int(d["offset"]), epoch=int(d.get("epoch", 0)))


def read_shard_header(path: str | Path) -> int:
    """Return the token count of an llm.c ``.bin`` shard, validating magic and version."""
    with open(path, "rb") as fh:
        header = np.frombuffer(fh.read(_HEADER_BYTES), dtype=np.int32)
    if header.size != HEADER_INTS:
        raise ValueError(f"{path}: truncated header")
    if int(header[0]) != MAGIC:
        raise ValueError(f"{path}: bad magic {int(header[0])} (expected {MAGIC})")
    if int(header[1]) != VERSION:
        raise ValueError(f"{path}: unsupported version {int(header[1])}")
    ntok = int(header[2])
    size = os.path.getsize(path)
    if size < _HEADER_BYTES + 2 * ntok:
        raise ValueError(f"{path}: file holds {(size - _HEADER_BYTES) // 2} tokens, header says {ntok}")
    return ntok


def load_shard(path: str | Path) -> np.ndarray:
    """Memory-map a shard and return its uint16 token array (no copy, no resident cost)."""
    ntok = read_shard_header(path)
    return np.memmap(path, dtype=np.uint16, mode="r", offset=_HEADER_BYTES, shape=(ntok,))


def shard_paths(data_dir: str | Path, pattern: str) -> list[str]:
    """Sorted absolute paths of the shards matching ``pattern`` inside ``data_dir``."""
    return sorted(glob.glob(str(Path(data_dir) / pattern)))


def synthetic_tokens(n_tokens: int, vocab_size: int, seed: int) -> np.ndarray:
    """Deterministic pseudo-random token array used by tests (no network, no disk)."""
    rng = np.random.default_rng(seed)
    # A mildly structured stream: a Zipf-ish token distribution plus short repeats, so a
    # tiny model can actually reduce the loss instead of memorising uniform noise.
    n_distinct = max(2, min(int(vocab_size), max(16, int(vocab_size) // 64)))
    base = rng.integers(0, n_distinct, size=n_tokens, dtype=np.uint16)
    stride = 17
    base[stride:] = np.where(rng.random(n_tokens - stride) < 0.35, base[:-stride], base[stride:])
    return base.astype(np.uint16)


class TokenStream:
    """Contiguous, resumable reader over a list of token shards.

    Each call to :meth:`next_batch` consumes ``micro_batch * block_size`` tokens (plus one
    look-ahead token for the shifted targets) and advances the cursor.
    """

    def __init__(
        self,
        paths: list[str],
        block_size: int,
        micro_batch: int,
        arrays: list[np.ndarray] | None = None,
        cursor: Cursor | None = None,
    ) -> None:
        if not paths and arrays is None:
            raise FileNotFoundError("TokenStream needs at least one shard")
        self.paths = list(paths)
        self.block_size = int(block_size)
        self.micro_batch = int(micro_batch)
        self._arrays: list[np.ndarray | None]
        if arrays is not None:
            self._arrays = list(arrays)
            self.paths = self.paths or [f"<memory:{i}>" for i in range(len(arrays))]
            self._lengths = [int(a.size) for a in arrays]
        else:
            self._arrays = [None] * len(self.paths)
            self._lengths = [read_shard_header(p) for p in self.paths]
        self.cursor = cursor or Cursor()
        span = self.micro_batch * self.block_size + 1
        if min(self._lengths) < span:
            raise ValueError(f"a shard holds only {min(self._lengths)} tokens, need {span} per batch")

    # -- shard access -------------------------------------------------------------
    def _array(self, idx: int) -> np.ndarray:
        arr = self._arrays[idx]
        if arr is None:
            arr = load_shard(self.paths[idx])
            self._arrays[idx] = arr
        return arr

    @property
    def total_tokens(self) -> int:
        """Tokens across all shards."""
        return int(sum(self._lengths))

    def state(self) -> dict[str, int]:
        return self.cursor.to_dict()

    def load_state(self, state: dict[str, int]) -> None:
        self.cursor = Cursor.from_dict(state)

    # -- iteration ----------------------------------------------------------------
    def next_batch(self) -> tuple[mx.array, mx.array]:
        """Return ``(x, y)`` int32 arrays of shape ``(micro_batch, block_size)``."""
        span = self.micro_batch * self.block_size + 1
        cur = self.cursor
        if cur.offset + span > self._lengths[cur.shard]:
            cur.shard += 1
            cur.offset = 0
            if cur.shard >= len(self.paths):
                cur.shard = 0
                cur.epoch += 1
        buf = np.asarray(self._array(cur.shard)[cur.offset : cur.offset + span], dtype=np.uint16)
        cur.offset += span - 1
        # uint16 -> int32: MLX indexing/embedding wants a signed integer index type.
        toks = buf.astype(np.int32)
        x = mx.array(toks[:-1].reshape(self.micro_batch, self.block_size))
        y = mx.array(toks[1:].reshape(self.micro_batch, self.block_size))
        return x, y


class ValLoader:
    """Fixed validation set: the first ``val_tokens`` tokens of one shard.

    Windows are non-overlapping ``block_size``-token spans (each window reads one extra
    look-ahead token for its targets), matching modded-nanogpt's fixed 10,485,760-token
    validation split so the loss number is directly comparable.
    """

    def __init__(
        self,
        array: np.ndarray,
        block_size: int,
        micro_batch: int,
        val_tokens: int,
        max_batches: int = 0,
    ) -> None:
        self.array = array
        self.block_size = int(block_size)
        self.micro_batch = int(micro_batch)
        usable = min(int(val_tokens), int(array.size) - 1)
        self.n_windows = usable // self.block_size
        self.n_batches = self.n_windows // self.micro_batch
        if max_batches > 0:
            self.n_batches = min(self.n_batches, int(max_batches))
        if self.n_batches < 1:
            raise ValueError("validation set too small for one batch")

    @property
    def tokens(self) -> int:
        """Number of *target* tokens scored per full pass."""
        return self.n_batches * self.micro_batch * self.block_size

    def __len__(self) -> int:
        return self.n_batches

    def __iter__(self):
        T, B = self.block_size, self.micro_batch
        for b in range(self.n_batches):
            start = b * B * T
            buf = np.asarray(self.array[start : start + B * T + 1], dtype=np.uint16).astype(np.int32)
            yield (
                mx.array(buf[:-1].reshape(B, T)),
                mx.array(buf[1:].reshape(B, T)),
            )


# --------------------------------------------------------------------------------------
# Factories
# --------------------------------------------------------------------------------------


def make_train_stream(data_cfg, block_size: int, micro_batch: int, vocab_size: int = 50304) -> TokenStream:
    """Build the training :class:`TokenStream` described by a :class:`r52.config.DataConfig`."""
    if data_cfg.source == "synthetic":
        arr = synthetic_tokens(data_cfg.synthetic_tokens, vocab_size, data_cfg.seed)
        return TokenStream([], block_size, micro_batch, arrays=[arr])
    paths = shard_paths(data_cfg.data_dir, data_cfg.train_glob)
    if not paths:
        raise FileNotFoundError(
            f"no shards matching {data_cfg.train_glob!r} in {data_cfg.data_dir!r}. "
            f"Run: python scripts/prepare_data.py --train-shards 1"
        )
    return TokenStream(paths, block_size, micro_batch)


def make_val_loader(data_cfg, block_size: int, micro_batch: int, max_batches: int = 0,
                    vocab_size: int = 50304) -> ValLoader:
    """Build the fixed validation loader described by a :class:`r52.config.DataConfig`."""
    if data_cfg.source == "synthetic":
        arr = synthetic_tokens(
            max(block_size * micro_batch * 4 + 1, data_cfg.synthetic_tokens // 10),
            vocab_size,
            data_cfg.seed + 1,
        )
        return ValLoader(arr, block_size, micro_batch, arr.size - 1, max_batches or 4)
    path = Path(data_cfg.data_dir) / data_cfg.val_file
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run: python scripts/prepare_data.py --train-shards 1"
        )
    return ValLoader(load_shard(path), block_size, micro_batch, data_cfg.val_tokens, max_batches)
