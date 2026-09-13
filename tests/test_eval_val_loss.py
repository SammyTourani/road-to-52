# Copyright 2026 The road-to-52 authors.
# SPDX-License-Identifier: Apache-2.0
"""Validation loss: the same number r52.train reports, and a direct hand computation.

Everything here runs on a synthetic llm.c ``.bin`` shard written into ``tmp_path`` -- no
network, no 10.5M-token split.
"""

from __future__ import annotations

import struct
from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest
from conftest import tiny_model_config
from mlx.utils import tree_map

from r52.checkpoint import save_checkpoint
from r52.config import Config
from r52.data import HEADER_INTS, MAGIC, VERSION, ValLoader, load_shard
from r52.eval.lm import LM
from r52.eval.val_loss import evaluate_val_loss, make_loader
from r52.model import GPT
from r52.train import evaluate as train_evaluate

pytest.importorskip("mlx_lm")

BLOCK = 32
MICRO = 2
N_TOKENS = 4096


def write_bin(path: Path, tokens: np.ndarray) -> Path:
    """Write an llm.c ``.bin`` shard: 256 int32 header words, then uint16 tokens."""
    header = np.zeros(HEADER_INTS, dtype=np.int32)
    header[0], header[1], header[2] = MAGIC, VERSION, tokens.size
    with path.open("wb") as fh:
        fh.write(header.tobytes())
        fh.write(tokens.astype(np.uint16).tobytes())
    assert struct.unpack("<i", path.read_bytes()[:4])[0] == MAGIC
    return path


@pytest.fixture
def fixture(tmp_path: Path):
    """A tiny checkpoint plus a synthetic val shard sized for a sub-second test."""
    cfg = tiny_model_config(block_size=BLOCK)
    model = GPT(cfg)
    model.update(tree_map(lambda p: p + mx.random.normal(p.shape) * 0.05, model.parameters()))
    mx.eval(model.parameters())
    ckpt = save_checkpoint(tmp_path / "ckpt", model, step=11, config=Config(model=cfg))

    rng = np.random.default_rng(1234)
    tokens = rng.integers(0, cfg.vocab_size, size=N_TOKENS, dtype=np.uint16)
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    write_bin(data_dir / "fineweb_val_000000.bin", tokens)
    return cfg, model, ckpt, data_dir, tokens


def test_matches_a_direct_computation_over_the_same_windows(fixture) -> None:
    _, model, ckpt, data_dir, tokens = fixture
    loader, array = make_loader(data_dir, "fineweb_val_000000.bin", BLOCK, MICRO, N_TOKENS - 1)
    out = evaluate_val_loss(LM.load(ckpt), loader, array=array,
                            bpb_cache=data_dir / ".bpb.json", verbose=False)

    # Direct: walk the same non-overlapping windows by hand and average GPT.loss.
    n_windows = (N_TOKENS - 1) // BLOCK
    n_batches = n_windows // MICRO
    total = 0.0
    for b in range(n_batches):
        start = b * MICRO * BLOCK
        buf = np.asarray(tokens[start : start + MICRO * BLOCK + 1], dtype=np.int32)
        x = mx.array(buf[:-1].reshape(MICRO, BLOCK))
        y = mx.array(buf[1:].reshape(MICRO, BLOCK))
        total += float(model.loss(x, y, fp32_logits=True))
    expected = total / n_batches

    assert out["n_tokens"] == n_batches * MICRO * BLOCK
    assert out["n_batches"] == n_batches
    assert out["val_loss"] == pytest.approx(expected, abs=2e-5)


def test_reproduces_the_val_loss_r52_train_reports(fixture) -> None:
    """docs/ARCHITECTURE.md §4 and §6 must agree on one number for one checkpoint."""
    _, model, ckpt, data_dir, _ = fixture
    loader = ValLoader(load_shard(data_dir / "fineweb_val_000000.bin"), BLOCK, MICRO, N_TOKENS - 1)
    theirs = train_evaluate(model, loader)
    mine = evaluate_val_loss(LM.load(ckpt), loader, verbose=False)
    assert mine["n_tokens"] == theirs["val_tokens"]
    assert mine["val_loss"] == pytest.approx(theirs["val_loss"], abs=2e-5)


def test_bits_per_byte_is_the_documented_conversion(fixture) -> None:
    import math

    _, _, ckpt, data_dir, _ = fixture
    loader, array = make_loader(data_dir, "fineweb_val_000000.bin", BLOCK, MICRO, N_TOKENS - 1)
    out = evaluate_val_loss(LM.load(ckpt), loader, array=array,
                            bpb_cache=data_dir / ".bpb.json", verbose=False)
    expected = (out["val_loss"] / math.log(2.0)) * (out["n_tokens"] / out["n_bytes"])
    assert out["val_bpb"] == pytest.approx(expected, rel=1e-12)
    assert out["bytes_per_token"] > 0
    assert (data_dir / ".bpb.json").exists()  # cached, because decoding is not free


def test_windows_are_non_overlapping_and_cover_every_target_once(fixture) -> None:
    _, _, _, data_dir, tokens = fixture
    loader, _ = make_loader(data_dir, "fineweb_val_000000.bin", BLOCK, MICRO, N_TOKENS - 1)
    seen: list[int] = []
    for _, y in loader:
        seen.extend(int(t) for t in np.asarray(y).reshape(-1))
    assert seen == [int(t) for t in tokens[1 : len(seen) + 1]]


def test_max_tokens_truncates_the_split(fixture) -> None:
    _, _, ckpt, data_dir, _ = fixture
    loader, _ = make_loader(data_dir, "fineweb_val_000000.bin", BLOCK, MICRO, 512)
    out = evaluate_val_loss(LM.load(ckpt), loader, verbose=False)
    assert out["n_tokens"] == 512


def test_missing_shard_names_the_fix(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match=r"prepare_data\.py"):
        make_loader(tmp_path, "fineweb_val_000000.bin", BLOCK, MICRO, 1024)
