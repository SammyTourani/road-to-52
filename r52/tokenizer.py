# Copyright 2026 The road-to-52 authors.
# SPDX-License-Identifier: Apache-2.0
"""GPT-2 BPE tokenizer (tiktoken) plus bits-per-byte bookkeeping.

Rung 0 reuses OpenAI's GPT-2 BPE because the FineWeb-10B shards we train on
(``kjj0/fineweb10B-gpt2``) are already tokenized with it.  ``n_vocab`` is 50257;
the model pads to 50304 (a multiple of 128) so the ``lm_head`` matmul is well shaped.

**bits per byte** is the tokenizer-agnostic version of validation loss::

    bpb = (loss_nats / ln 2) * (tokens / bytes)

``tokens/bytes`` is a property of the *validation corpus*, not of the model, so it is
measured once and cached in ``<data_dir>/.bpb_cache.json``.
"""

from __future__ import annotations

import json
import math
from functools import lru_cache
from pathlib import Path

import numpy as np

__all__ = ["GPT2Tokenizer", "bits_per_byte", "val_bytes_per_token"]

EOT = 50256
"""GPT-2's ``<|endoftext|>`` id."""
N_VOCAB = 50257
PADDED_VOCAB = 50304


class GPT2Tokenizer:
    """Thin wrapper over ``tiktoken.get_encoding('gpt2')``."""

    def __init__(self) -> None:
        import tiktoken

        self.enc = tiktoken.get_encoding("gpt2")
        self.n_vocab = self.enc.n_vocab
        self.padded_vocab = PADDED_VOCAB
        self.eot = EOT

    def encode(self, text: str, allowed_special: str | set[str] = "all") -> list[int]:
        """Text -> token ids."""
        return self.enc.encode(text, allowed_special=allowed_special)

    def decode(self, tokens) -> str:
        """Token ids -> text (invalid UTF-8 is replaced, never raised)."""
        return self.enc.decode([int(t) for t in tokens])

    def decode_bytes(self, tokens) -> bytes:
        """Token ids -> raw UTF-8 bytes (exact; used for byte counting)."""
        return self.enc.decode_bytes([int(t) for t in tokens])

    def n_bytes(self, tokens) -> int:
        """UTF-8 byte length of the text that ``tokens`` decode to."""
        return len(self.decode_bytes(tokens))


@lru_cache(maxsize=4)
def _tokenizer() -> GPT2Tokenizer:
    return GPT2Tokenizer()


def bits_per_byte(loss_nats: float, n_tokens: int, n_bytes: int) -> float:
    """Convert a mean cross-entropy in **nats/token** to **bits/byte**."""
    if n_bytes <= 0:
        return float("nan")
    return (loss_nats / math.log(2.0)) * (n_tokens / n_bytes)


def val_bytes_per_token(
    tokens: np.ndarray,
    n_tokens: int,
    cache_path: str | Path | None = None,
    cache_key: str = "val",
    chunk: int = 1 << 20,
) -> float:
    """Mean UTF-8 **bytes per token** over the first ``n_tokens`` of ``tokens``.

    Cached in ``cache_path`` (a small JSON file) keyed by ``f"{cache_key}:{n_tokens}"``,
    because decoding 10.5M tokens takes a few seconds and the value never changes.
    """
    key = f"{cache_key}:{int(n_tokens)}"
    cache: dict[str, float] = {}
    p = Path(cache_path) if cache_path else None
    if p is not None and p.exists():
        try:
            cache = json.loads(p.read_text())
        except (OSError, ValueError):
            cache = {}
        if key in cache:
            return float(cache[key])

    tok = _tokenizer()
    n = min(int(n_tokens), int(tokens.size))
    total = 0
    for start in range(0, n, chunk):
        piece = np.asarray(tokens[start : min(start + chunk, n)], dtype=np.int64)
        total += tok.n_bytes(piece.tolist())
    ratio = total / max(1, n)

    if p is not None:
        cache[key] = ratio
        try:
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(json.dumps(cache, indent=2))
        except OSError:
            pass
    return ratio
