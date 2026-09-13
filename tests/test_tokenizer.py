# Copyright 2026 The road-to-52 authors.
# SPDX-License-Identifier: Apache-2.0
"""GPT-2 tokenizer wrapper and bits-per-byte arithmetic."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

from r52.tokenizer import EOT, N_VOCAB, PADDED_VOCAB, bits_per_byte, val_bytes_per_token

pytest.importorskip("tiktoken")


@pytest.fixture(scope="module")
def tok():
    from r52.tokenizer import GPT2Tokenizer

    try:
        return GPT2Tokenizer()
    except Exception as exc:  # pragma: no cover - offline first run
        pytest.skip(f"gpt2 BPE unavailable offline: {exc}")


def test_encode_decode_round_trip(tok) -> None:
    text = "The quick brown fox jumps over the lazy dog. 42 % — ünïcode ✅"
    ids = tok.encode(text)
    assert tok.decode(ids) == text
    assert tok.n_vocab == N_VOCAB and tok.padded_vocab == PADDED_VOCAB and tok.eot == EOT
    assert max(ids) < N_VOCAB


def test_known_ids(tok) -> None:
    assert tok.encode("hello world") == [31373, 995]
    assert tok.decode([31373, 995]) == "hello world"


def test_byte_counting(tok) -> None:
    text = "héllo"  # 6 UTF-8 bytes
    assert tok.n_bytes(tok.encode(text)) == len(text.encode("utf-8")) == 6


def test_bits_per_byte_formula() -> None:
    # 1 nat/token over 2 bytes/token -> 1/ln2/2 bits per byte
    assert bits_per_byte(1.0, 100, 200) == pytest.approx(1.0 / math.log(2) / 2.0)
    assert math.isnan(bits_per_byte(1.0, 100, 0))


def test_val_bytes_per_token_caches(tmp_path: Path, tok) -> None:
    ids = np.array(tok.encode("road to 52 " * 50), dtype=np.uint16)
    cache = tmp_path / ".bpb_cache.json"
    a = val_bytes_per_token(ids, ids.size, cache, "unit")
    assert a > 1.0
    assert json.loads(cache.read_text())[f"unit:{ids.size}"] == pytest.approx(a)
    # second call must hit the cache, not the tokenizer
    assert val_bytes_per_token(np.zeros(ids.size, np.uint16), ids.size, cache, "unit") == pytest.approx(a)
