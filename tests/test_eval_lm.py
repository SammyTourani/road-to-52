# Copyright 2026 The road-to-52 authors.
# SPDX-License-Identifier: Apache-2.0
"""The eval adapter: one interface, two loaders, identical numbers."""

from __future__ import annotations

from pathlib import Path

import mlx.core as mx
import pytest
from conftest import tiny_model_config
from mlx.utils import tree_map

from r52.checkpoint import save_checkpoint
from r52.config import Config
from r52.eval.lm import LM, pad_stack, wilson_interval
from r52.export import export_model
from r52.model import GPT

pytest.importorskip("mlx_lm")


def _perturbed(cfg) -> GPT:
    """Non-trivial weights -- a zero-init ``lm_head`` would make every logit identical."""
    m = GPT(cfg)
    m.update(tree_map(lambda p: p + mx.random.normal(p.shape) * 0.05, m.parameters()))
    mx.eval(m.parameters())
    return m


@pytest.fixture
def tiny_pair(tmp_path: Path):
    """A tiny GPT saved as both an r52 checkpoint and an mlx-lm export of the same weights."""
    cfg = tiny_model_config()
    model = _perturbed(cfg)
    full = Config(name="tiny", model=cfg)
    ckpt = save_checkpoint(tmp_path / "ckpt", model, step=3, config=full)
    exported = export_model(model, tmp_path / "mlx", full, dtype="bfloat16")
    return cfg, model, ckpt, exported


def test_loads_an_r52_checkpoint_and_an_export(tiny_pair) -> None:
    cfg, _, ckpt, exported = tiny_pair
    a = LM.load(ckpt)
    b = LM.load(exported)
    assert a.kind == "r52-checkpoint"
    assert b.kind == "mlx-lm"
    assert a.max_context == cfg.block_size == b.max_context
    assert a.vocab_size == cfg.vocab_size == b.vocab_size
    assert a.compute_dtype == mx.bfloat16  # mixed precision computes in bf16
    assert a.describe()["step"] == 3


def test_adapter_parity_r52_gpt_vs_its_own_export(tiny_pair) -> None:
    """docs/ARCHITECTURE.md §7: our logits vs mlx-lm's, 8 prompts, max abs diff < 1e-2."""
    cfg, _, ckpt, exported = tiny_pair
    a, b = LM.load(ckpt), LM.load(exported)
    idx = mx.random.randint(0, cfg.vocab_size, (8, cfg.block_size))
    la, lb = a.logits(idx), b.logits(idx)
    mx.eval(la, lb)
    assert la.dtype == mx.float32 and lb.dtype == mx.float32
    assert la.shape == (8, cfg.block_size, cfg.vocab_size)
    assert float(mx.abs(la - lb).max()) < 1e-2

    na, _ = a.scores(idx)
    nb, _ = b.scores(idx)
    mx.eval(na, nb)
    assert float(mx.abs(na[:, :-1] - nb[:, :-1]).max()) < 1e-2


def test_logits_match_the_model_call(tiny_pair) -> None:
    """``LM.logits`` is exactly ``GPT.__call__`` (fp32 softcapped logits), not an approximation."""
    cfg, model, ckpt, _ = tiny_pair
    lm = LM.load(ckpt)
    idx = mx.random.randint(0, cfg.vocab_size, (3, cfg.block_size))
    mine, theirs = lm.logits(idx), model(idx)
    mx.eval(mine, theirs)
    assert float(mx.abs(mine - theirs).max()) == 0.0


def test_scores_are_the_autoregressive_nll_with_a_nan_tail(tiny_pair) -> None:
    cfg, _, ckpt, _ = tiny_pair
    lm = LM.load(ckpt)
    idx = mx.random.randint(0, cfg.vocab_size, (2, cfg.block_size))
    nll, pred = lm.scores(idx)
    logits = lm.logits(idx)
    mx.eval(nll, pred, logits)

    # Reference: fp32 log-softmax, picked at the next token.
    lse = mx.logsumexp(logits, axis=-1)
    picked = mx.take_along_axis(logits[:, :-1], idx[:, 1:, None], axis=-1).squeeze(-1)
    ref = lse[:, :-1] - picked
    mx.eval(ref)
    assert float(mx.abs(nll[:, :-1] - ref).max()) < 2e-5
    assert bool(mx.isnan(nll[:, -1]).all())
    assert pred.shape == idx.shape
    assert float(mx.abs(pred - mx.argmax(logits, axis=-1).astype(mx.int32)).max()) == 0.0


def test_row_chunking_does_not_change_the_answer(tiny_pair) -> None:
    """The peak-memory knob is a pure implementation detail."""
    cfg, _, ckpt, _ = tiny_pair
    idx = mx.random.randint(0, cfg.vocab_size, (6, cfg.block_size))
    big = LM.load(ckpt, max_positions_per_forward=1 << 20)
    small = LM.load(ckpt, max_positions_per_forward=cfg.block_size)  # one row per forward
    assert small._row_chunks(6, cfg.block_size) == 1
    a, _ = big.scores(idx)
    b, _ = small.scores(idx)
    mx.eval(a, b)
    assert float(mx.abs(a[:, :-1] - b[:, :-1]).max()) == 0.0


def test_token_nll_equals_gpt_loss_in_fp32(tiny_pair) -> None:
    """This identity is what makes r52.eval.val_loss reproduce r52.train's reported val loss."""
    cfg, model, ckpt, _ = tiny_pair
    lm = LM.load(ckpt)
    toks = mx.random.randint(0, cfg.vocab_size, (4, cfg.block_size + 1))
    x, y = toks[:, :-1], toks[:, 1:]
    mine = lm.mean_nll(x, y)
    theirs = float(model.loss(x, y, fp32_logits=True))
    assert abs(mine - theirs) < 2e-5


def test_sequences_longer_than_the_context_are_rejected(tiny_pair) -> None:
    cfg, _, ckpt, _ = tiny_pair
    lm = LM.load(ckpt)
    too_long = mx.zeros((1, cfg.block_size + 1), dtype=mx.int32)
    with pytest.raises(ValueError, match="max_context"):
        lm.logits(too_long)


def test_load_rejects_a_directory_that_is_neither(tmp_path: Path) -> None:
    (tmp_path / "empty").mkdir()
    with pytest.raises(ValueError, match="neither an r52 checkpoint"):
        LM.load(tmp_path / "empty")


def test_pad_stack_right_pads() -> None:
    out = pad_stack([[1, 2, 3], [4], [5, 6]], pad_id=9)
    assert out.tolist() == [[1, 2, 3], [4, 9, 9], [5, 6, 9]]
    assert out.dtype == mx.int32


def test_wilson_interval_brackets_the_estimate() -> None:
    lo, hi = wilson_interval(2955, 10042)
    assert lo < 0.2955 < hi
    assert 0.008 < (hi - lo) / 2 < 0.010  # ~+/- 0.9 points at n = 10,042
    assert wilson_interval(0, 10) == (0.0, pytest.approx(0.2775, abs=1e-3))
    assert wilson_interval(5, 0)[0] != wilson_interval(5, 0)[0]  # nan for an empty sample


def test_tokenizer_is_gpt2_tiktoken(tiny_pair) -> None:
    _, _, ckpt, _ = tiny_pair
    tok = LM.load(ckpt).tokenizer
    assert tok.n_vocab == 50257
    assert tok.eot == 50256
    assert tok.decode(tok.encode(" hello world")) == " hello world"
