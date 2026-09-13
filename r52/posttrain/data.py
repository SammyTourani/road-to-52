# Copyright 2026 The road-to-52 authors.
# SPDX-License-Identifier: Apache-2.0
"""On-disk format and batching for rendered conversations (SFT / RL prompts).

``scripts/prepare_sft_data.py`` writes one directory per split::

    <dir>/tokens.npy    uint16, every conversation's ids concatenated
    <dir>/mask.npy      uint8,  1 = this id is a supervised target (assistant span)
    <dir>/offsets.npy   int64,  n+1 boundaries: conversation i is [off[i], off[i+1])
    <dir>/meta.json     counts, the source dataset, and the render settings

``.npy`` (rather than one ``.npz``) because ``np.load(..., mmap_mode='r')`` then costs no
resident memory: a 100 M-token SFT set is 200 MB on disk and ~0 in RAM until touched.

Batching is **pad-and-mask**, not packing, by default: each micro-batch is padded to the
longest example in it (rounded up to ``pad_multiple``), padding targets are
:data:`r52.chat_template.IGNORE_INDEX`, and so are every system/user token.  The loss the
model sees is therefore the mean over *assistant tokens only* --
:meth:`r52.model.GPT.loss` already treats negative targets as ignore, so nothing in the
model or the trainer had to change.

Units: all lengths are **tokens**.
"""

from __future__ import annotations

import json
from collections.abc import Iterator, Sequence
from pathlib import Path

import mlx.core as mx
import numpy as np

from ..chat_template import IGNORE_INDEX, PAD

__all__ = [
    "ConversationDataset",
    "PackedBatcher",
    "SFTBatcher",
    "iter_batches",
    "pack_blocks",
    "packed_xy",
    "write_split",
]

_TOKENS = "tokens.npy"
_MASK = "mask.npy"
_OFFSETS = "offsets.npy"
_META = "meta.json"


# --------------------------------------------------------------------------------------
# Writing
# --------------------------------------------------------------------------------------


def write_split(
    out_dir: str | Path,
    conversations: Sequence[tuple[Sequence[int], Sequence[int]]],
    meta: dict | None = None,
) -> Path:
    """Write one split directory from ``[(ids, mask), ...]``. Returns the directory."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    lengths = [len(ids) for ids, _ in conversations]
    offsets = np.zeros(len(conversations) + 1, dtype=np.int64)
    np.cumsum(lengths, out=offsets[1:])
    total = int(offsets[-1])
    tokens = np.empty(total, dtype=np.uint16)
    mask = np.empty(total, dtype=np.uint8)
    for i, (ids, m) in enumerate(conversations):
        a, b = offsets[i], offsets[i + 1]
        tokens[a:b] = np.asarray(ids, dtype=np.uint16)
        mask[a:b] = np.asarray(m, dtype=np.uint8)
    np.save(out / _TOKENS, tokens)
    np.save(out / _MASK, mask)
    np.save(out / _OFFSETS, offsets)
    info = {
        "n_conversations": len(conversations),
        "n_tokens": total,
        "n_supervised_tokens": int(mask.sum()),
        "max_len": int(max(lengths)) if lengths else 0,
        "mean_len": round(total / max(1, len(conversations)), 2),
        **(meta or {}),
    }
    (out / _META).write_text(json.dumps(info, indent=2))
    return out


# --------------------------------------------------------------------------------------
# Reading
# --------------------------------------------------------------------------------------


class ConversationDataset:
    """Memory-mapped ``(ids, mask)`` conversations written by :func:`write_split`."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        missing = [f for f in (_TOKENS, _MASK, _OFFSETS) if not (self.path / f).exists()]
        if missing:
            raise FileNotFoundError(
                f"{self.path} is not an r52 SFT split (missing {missing}). "
                f"Run: python scripts/prepare_sft_data.py --out {self.path.parent}"
            )
        self.tokens = np.load(self.path / _TOKENS, mmap_mode="r")
        self.mask = np.load(self.path / _MASK, mmap_mode="r")
        self.offsets = np.load(self.path / _OFFSETS)
        self.meta = json.loads((self.path / _META).read_text()) if (self.path / _META).exists() else {}

    def __len__(self) -> int:
        return int(self.offsets.size - 1)

    def __getitem__(self, i: int) -> tuple[np.ndarray, np.ndarray]:
        a, b = int(self.offsets[i]), int(self.offsets[i + 1])
        return np.asarray(self.tokens[a:b]), np.asarray(self.mask[a:b])

    def lengths(self) -> np.ndarray:
        """Token length of every conversation."""
        return np.diff(self.offsets)

    @property
    def n_tokens(self) -> int:
        return int(self.offsets[-1])

    @property
    def n_supervised(self) -> int:
        """Tokens with ``mask == 1`` (the only ones that contribute to the SFT loss)."""
        return int(self.meta.get("n_supervised_tokens") or int(np.asarray(self.mask).sum()))


# --------------------------------------------------------------------------------------
# Batching
# --------------------------------------------------------------------------------------


def _to_xy(
    ids: np.ndarray, mask: np.ndarray, width: int
) -> tuple[np.ndarray, np.ndarray]:
    """One padded ``(x, y)`` row of ``width`` tokens from a single conversation.

    ``x[t] = ids[t]``, ``y[t] = ids[t+1]`` where ``mask[t+1]`` is set, else
    :data:`~r52.chat_template.IGNORE_INDEX`.  Positions past the conversation are ``PAD``
    in ``x`` and ignored in ``y``.
    """
    n = min(int(ids.size), width + 1)
    x = np.full(width, PAD, dtype=np.int32)
    y = np.full(width, IGNORE_INDEX, dtype=np.int32)
    x[: n - 1] = ids[: n - 1]
    keep = mask[1:n].astype(bool)
    # ids are uint16 on disk; widen before mixing in the negative ignore index.
    y[: n - 1] = np.where(keep, ids[1:n].astype(np.int32), IGNORE_INDEX)
    return x, y


def pack_blocks(
    dataset: ConversationDataset, block: int, order: Sequence[int] | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Concatenate conversations into ``(n_blocks, block + 1)`` ids and masks.

    Packing trades exactness for throughput: a block can hold the tail of one conversation
    and the head of the next, and attention is *not* re-masked at the boundary, so the
    second conversation can attend to the first.  That is the standard trade (nanochat and
    mlx-lm both do it); ``SFTConfig.pack`` defaults to ``False`` so the shipped runs avoid
    it entirely.
    """
    order = range(len(dataset)) if order is None else order
    ids_parts, mask_parts = [], []
    for i in order:
        ids, m = dataset[i]
        ids_parts.append(ids.astype(np.int32))
        mask_parts.append(m.astype(np.uint8))
    flat_ids = np.concatenate(ids_parts) if ids_parts else np.zeros(0, dtype=np.int32)
    flat_mask = np.concatenate(mask_parts) if mask_parts else np.zeros(0, dtype=np.uint8)
    n_blocks = max(0, (flat_ids.size - 1) // block)
    width = block + 1
    out_ids = np.empty((n_blocks, width), dtype=np.int32)
    out_mask = np.empty((n_blocks, width), dtype=np.uint8)
    for b in range(n_blocks):
        a = b * block
        out_ids[b] = flat_ids[a : a + width]
        out_mask[b] = flat_mask[a : a + width]
    return out_ids, out_mask


def packed_xy(ids_block: np.ndarray, mask_block: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Shift a ``(B, block + 1)`` packed block into ``(x, y)`` of ``(B, block)``."""
    x = ids_block[:, :-1].astype(np.int32)
    y = np.where(mask_block[:, 1:].astype(bool), ids_block[:, 1:].astype(np.int32),
                 IGNORE_INDEX).astype(np.int32)
    return x, y


class PackedBatcher:
    """Shuffled micro-batches over pre-packed ``block``-token rows (``SFTConfig.pack``)."""

    def __init__(
        self, dataset: ConversationDataset, micro_batch: int, block: int, seed: int = 1337
    ) -> None:
        self.micro_batch = int(micro_batch)
        self.block = int(block)
        self.seed = int(seed)
        rng = np.random.default_rng(seed)
        order = rng.permutation(len(dataset))
        self._ids, self._mask = pack_blocks(dataset, self.block, order)
        self.epoch = 0
        self._pos = 0
        self._perm = np.arange(self._ids.shape[0])

    def batches_per_epoch(self) -> int:
        return int(self._ids.shape[0]) // self.micro_batch

    def next_batch(self) -> tuple[mx.array, mx.array]:
        """Next ``(x, y)`` pair, shape ``(micro_batch, block)``."""
        if self._pos + self.micro_batch > self._perm.size:
            self.epoch += 1
            self._perm = np.random.default_rng(self.seed + self.epoch).permutation(self._ids.shape[0])
            self._pos = 0
        sel = self._perm[self._pos : self._pos + self.micro_batch]
        self._pos += self.micro_batch
        x, y = packed_xy(self._ids[sel], self._mask[sel])
        return mx.array(x), mx.array(y)

    def state(self) -> dict[str, int]:
        return {"epoch": self.epoch, "pos": self._pos}


class SFTBatcher:
    """Shuffled, padded micro-batches of ``(x, y)`` for one or more epochs.

    Parameters
    ----------
    dataset
        A :class:`ConversationDataset`.
    micro_batch
        Conversations per forward/backward pass.
    max_seq
        Hard cap on the sequence length in **tokens**; longer conversations are truncated.
    pad_multiple
        Each batch is padded to ``ceil(longest / pad_multiple) * pad_multiple`` so MLX sees
        a handful of distinct shapes instead of one per batch (matters with ``mx.compile``).
    seed
        Shuffling seed; epoch ``e`` uses ``seed + e`` so epochs differ but replay exactly.
    sort_within_chunk
        Sort by length inside chunks of ``64 * micro_batch`` before batching, which cuts
        padding waste roughly in half while keeping the order effectively random.
    """

    def __init__(
        self,
        dataset: ConversationDataset,
        micro_batch: int,
        max_seq: int,
        pad_multiple: int = 64,
        seed: int = 1337,
        sort_within_chunk: bool = True,
    ) -> None:
        self.ds = dataset
        self.micro_batch = int(micro_batch)
        self.max_seq = int(max_seq)
        self.pad_multiple = max(1, int(pad_multiple))
        self.seed = int(seed)
        self.sort_within_chunk = bool(sort_within_chunk)
        self.epoch = 0
        self._order: list[int] = []
        self._pos = 0

    def _new_epoch(self) -> None:
        rng = np.random.default_rng(self.seed + self.epoch)
        order = rng.permutation(len(self.ds))
        if self.sort_within_chunk:
            chunk = 64 * self.micro_batch
            lengths = self.ds.lengths()
            order = np.concatenate(
                [
                    c[np.argsort(lengths[c], kind="stable")]
                    for c in np.array_split(order, max(1, len(order) // max(1, chunk)))
                ]
            )
        self._order = [int(i) for i in order]
        self._pos = 0
        self.epoch += 1

    def batches_per_epoch(self) -> int:
        """Micro-batches in one pass over the dataset (the tail batch is dropped)."""
        return len(self.ds) // self.micro_batch

    def next_batch(self) -> tuple[mx.array, mx.array]:
        """Next ``(x, y)`` pair, shape ``(micro_batch, T)``, int32, wrapping across epochs."""
        if self._pos + self.micro_batch > len(self._order):
            self._new_epoch()
        idx = self._order[self._pos : self._pos + self.micro_batch]
        self._pos += self.micro_batch
        rows = [self.ds[i] for i in idx]
        longest = max(min(int(ids.size), self.max_seq + 1) for ids, _ in rows)
        width = min(self.max_seq, max(1, longest - 1))
        width = min(self.max_seq, -(-width // self.pad_multiple) * self.pad_multiple)
        xs, ys = zip(*(_to_xy(ids, m, width) for ids, m in rows), strict=True)
        return mx.array(np.stack(xs)), mx.array(np.stack(ys))

    def state(self) -> dict[str, int]:
        return {"epoch": self.epoch, "pos": self._pos}


def iter_batches(
    dataset: ConversationDataset, micro_batch: int, max_seq: int, pad_multiple: int = 64
) -> Iterator[tuple[mx.array, mx.array]]:
    """Deterministic single pass in stored order (validation)."""
    n = len(dataset) // micro_batch
    for b in range(n):
        rows = [dataset[b * micro_batch + j] for j in range(micro_batch)]
        longest = max(min(int(ids.size), max_seq + 1) for ids, _ in rows)
        width = min(max_seq, max(1, longest - 1))
        width = min(max_seq, -(-width // pad_multiple) * pad_multiple)
        xs, ys = zip(*(_to_xy(ids, m, width) for ids, m in rows), strict=True)
        yield mx.array(np.stack(xs)), mx.array(np.stack(ys))
