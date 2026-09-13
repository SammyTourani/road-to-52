# Copyright 2026 The road-to-52 authors.
# SPDX-License-Identifier: Apache-2.0
"""Shard format, memory-mapped reading, batching, and deterministic cursor resume."""

from __future__ import annotations

import struct
from pathlib import Path

import numpy as np
import pytest

from r52.data import (
    HEADER_INTS,
    MAGIC,
    VERSION,
    Cursor,
    TokenStream,
    ValLoader,
    load_shard,
    make_train_stream,
    make_val_loader,
    read_shard_header,
    synthetic_tokens,
)


def write_shard(path: Path, tokens: np.ndarray) -> Path:
    header = np.zeros(HEADER_INTS, dtype=np.int32)
    header[0], header[1], header[2] = MAGIC, VERSION, tokens.size
    with open(path, "wb") as fh:
        fh.write(header.tobytes())
        fh.write(tokens.astype(np.uint16).tobytes())
    return path


def test_header_round_trip(tmp_path: Path) -> None:
    toks = np.arange(5000, dtype=np.uint16)
    p = write_shard(tmp_path / "fineweb_train_000001.bin", toks)
    assert read_shard_header(p) == 5000
    assert np.array_equal(np.asarray(load_shard(p)), toks)


def test_bad_magic_rejected(tmp_path: Path) -> None:
    p = tmp_path / "bad.bin"
    with open(p, "wb") as fh:
        fh.write(struct.pack("<i", 1234) + b"\0" * (HEADER_INTS * 4 - 4) + b"\0" * 16)
    with pytest.raises(ValueError, match="bad magic"):
        read_shard_header(p)


def test_batches_are_contiguous_and_shifted() -> None:
    arr = synthetic_tokens(20_000, 256, 0)
    s = TokenStream([], block_size=16, micro_batch=3, arrays=[arr])
    x, y = s.next_batch()
    xn, yn = np.array(x), np.array(y)
    assert xn.shape == (3, 16) and x.dtype.__str__().endswith("int32")
    assert np.array_equal(xn.reshape(-1)[1:], yn.reshape(-1)[:-1])
    assert np.array_equal(xn.reshape(-1), arr[:48].astype(np.int32))
    assert s.state() == {"shard": 0, "offset": 48, "epoch": 0}


def test_cursor_round_trip_is_deterministic() -> None:
    arr = synthetic_tokens(20_000, 256, 1)
    a = TokenStream([], 16, 2, arrays=[arr])
    for _ in range(5):
        a.next_batch()
    saved = a.state()
    want = [np.array(a.next_batch()[0]) for _ in range(3)]

    b = TokenStream([], 16, 2, arrays=[arr])
    b.load_state(saved)
    got = [np.array(b.next_batch()[0]) for _ in range(3)]
    assert all(np.array_equal(u, v) for u, v in zip(want, got, strict=True))
    assert a.state() == b.state()


def test_shard_rollover_and_epoch() -> None:
    a1 = synthetic_tokens(200, 64, 2)
    a2 = synthetic_tokens(200, 64, 3)
    s = TokenStream([], 16, 4, arrays=[a1, a2])  # 65 tokens consumed per batch
    seen = set()
    for _ in range(8):
        s.next_batch()
        seen.add((s.cursor.shard, s.cursor.epoch))
    assert (1, 0) in seen, "should roll into the second shard"
    assert any(e > 0 for _, e in seen), "should wrap around and bump the epoch"


def test_val_loader_windows_are_non_overlapping() -> None:
    arr = np.arange(10_000, dtype=np.uint16)
    v = ValLoader(arr, block_size=10, micro_batch=2, val_tokens=100)
    batches = list(v)
    assert len(batches) == 5 and v.tokens == 100
    flat = np.concatenate([np.array(x).reshape(-1) for x, _ in batches])
    assert np.array_equal(flat, arr[:100])


def test_synthetic_is_reproducible() -> None:
    assert np.array_equal(synthetic_tokens(1000, 256, 5), synthetic_tokens(1000, 256, 5))
    assert not np.array_equal(synthetic_tokens(1000, 256, 5), synthetic_tokens(1000, 256, 6))


def test_factories_with_synthetic_source() -> None:
    from r52.config import DataConfig

    dc = DataConfig(source="synthetic", synthetic_tokens=50_000, seed=3)
    s = make_train_stream(dc, 32, 2, 256)
    v = make_val_loader(dc, 32, 2, 2, 256)
    assert s.next_batch()[0].shape == (2, 32)
    assert len(list(v)) == 2


def test_missing_shards_message() -> None:
    from r52.config import DataConfig

    with pytest.raises(FileNotFoundError, match="prepare_data"):
        make_train_stream(DataConfig(data_dir="/nonexistent/path"), 32, 2)


def test_cursor_dataclass() -> None:
    c = Cursor(3, 17, 1)
    assert Cursor.from_dict(c.to_dict()) == c
