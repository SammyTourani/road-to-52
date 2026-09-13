# Copyright 2026 The road-to-52 authors.
# SPDX-License-Identifier: Apache-2.0
"""Model shapes, dtypes, precision modes, gradient flow and the FLOP accounting."""

from __future__ import annotations

import math

import mlx.core as mx
import pytest
from conftest import tiny_model_config
from mlx import nn
from mlx.utils import tree_flatten

from r52.config import ModelConfig
from r52.model import GPT


def test_forward_shape_and_dtype() -> None:
    cfg = tiny_model_config()
    m = GPT(cfg)
    idx = mx.random.randint(0, cfg.vocab_size, (3, cfg.block_size))
    logits = m(idx)
    mx.eval(logits)
    assert logits.shape == (3, cfg.block_size, cfg.vocab_size)
    assert logits.dtype == mx.float32


@pytest.mark.parametrize("precision,param_dtype", [("mixed", mx.float32), ("bf16", mx.bfloat16),
                                                   ("fp32", mx.float32)])
def test_precision_modes(precision: str, param_dtype) -> None:
    cfg = tiny_model_config(precision=precision)
    m = GPT(cfg)
    dtypes = {p.dtype for _, p in tree_flatten(m.parameters())}
    assert dtypes == {param_dtype}
    idx = mx.random.randint(0, cfg.vocab_size, (2, cfg.block_size))
    loss = m.loss(idx, idx)
    mx.eval(loss)
    assert loss.dtype == mx.float32 and math.isfinite(float(loss))


def test_zero_init_head_gives_uniform_loss() -> None:
    """lm_head is zero-init, so step 0 must sit exactly at ln(vocab_size)."""
    cfg = tiny_model_config(softcap=15.0)
    m = GPT(cfg)
    idx = mx.random.randint(0, cfg.vocab_size, (2, cfg.block_size))
    loss = float(m.loss(idx, idx, fp32_logits=True))
    assert loss == pytest.approx(math.log(cfg.vocab_size), abs=1e-4)


def test_softcap_bounds_logits() -> None:
    cfg = tiny_model_config(softcap=3.0)
    m = GPT(cfg)
    from mlx.utils import tree_map

    m.update(tree_map(lambda p: p + mx.random.normal(p.shape) * 2.0, m.parameters()))
    idx = mx.random.randint(0, cfg.vocab_size, (2, cfg.block_size))
    logits = m(idx)
    mx.eval(logits)
    assert float(mx.abs(logits).max()) <= 3.0 + 1e-3


def test_ignore_index() -> None:
    cfg = tiny_model_config()
    m = GPT(cfg)
    idx = mx.random.randint(0, cfg.vocab_size, (2, cfg.block_size))
    masked = mx.concatenate([idx[:, :1] * 0 - 1, idx[:, 1:]], axis=1)
    full = float(m.loss(idx, idx, fp32_logits=True))
    part = float(m.loss(idx, masked, fp32_logits=True))
    # With a zero-init head every token has the same loss, so masking must not change it.
    assert part == pytest.approx(full, abs=1e-4)
    all_masked = mx.zeros_like(idx) - 1
    assert float(m.loss(idx, all_masked, fp32_logits=True)) == pytest.approx(0.0, abs=1e-6)


@pytest.mark.parametrize("grad_checkpoint", [False, True])
def test_every_parameter_receives_a_gradient(grad_checkpoint: bool) -> None:
    """After one update the zero-init head is non-zero, so gradients must reach everything."""
    cfg = tiny_model_config(grad_checkpoint=grad_checkpoint)
    m = GPT(cfg)
    from mlx.utils import tree_map

    m.update(tree_map(lambda p: p + mx.random.normal(p.shape) * 0.05, m.parameters()))
    idx = mx.random.randint(0, cfg.vocab_size, (2, cfg.block_size))
    _, grads = nn.value_and_grad(m, lambda a, b: m.loss(a, b))(idx, idx)
    mx.eval(grads)
    flat = dict(tree_flatten(grads))
    assert set(flat) == {k for k, _ in tree_flatten(m.parameters())}
    zeros = [k for k, v in flat.items() if float(mx.abs(v).sum()) == 0.0]
    assert not zeros, f"no gradient reached: {zeros}"


def test_grad_checkpoint_matches_plain() -> None:
    cfg = tiny_model_config()
    m = GPT(cfg)
    idx = mx.random.randint(0, cfg.vocab_size, (2, cfg.block_size))
    a = float(m.loss(idx, idx, fp32_logits=True))
    m.set_grad_checkpoint(True)
    b = float(m.loss(idx, idx, fp32_logits=True))
    assert a == pytest.approx(b, abs=1e-6)


def test_gqa_and_swiglu_run() -> None:
    cfg = tiny_model_config(n_head=4, n_kv_head=2, mlp="swiglu", n_embd=64)
    m = GPT(cfg)
    idx = mx.random.randint(0, cfg.vocab_size, (2, cfg.block_size))
    loss = m.loss(idx, idx)
    mx.eval(loss)
    assert math.isfinite(float(loss))


def test_causality() -> None:
    """Changing a later token must not change earlier logits."""
    cfg = tiny_model_config()
    m = GPT(cfg)
    from mlx.utils import tree_map

    m.update(tree_map(lambda p: p + mx.random.normal(p.shape) * 0.05, m.parameters()))
    a = mx.random.randint(0, cfg.vocab_size, (1, cfg.block_size))
    b = mx.concatenate([a[:, :-1], (a[:, -1:] + 1) % cfg.vocab_size], axis=1)
    la, lb = m(a), m(b)
    mx.eval(la, lb)
    assert float(mx.abs(la[:, :-1] - lb[:, :-1]).max()) == pytest.approx(0.0, abs=1e-5)


def test_flops_per_token_formula() -> None:
    cfg = ModelConfig(n_layer=12, n_embd=768, n_head=6, vocab_size=50304, block_size=1024,
                      n_value_embeds=3, value_embed_share=True)
    m = GPT(cfg)
    assert m.num_params_flops() == 123_568_146           # classic GPT-2-small 6N budget
    assert m.flops_per_token() == 6 * m.num_params_flops() + 12 * 12 * 768 * 1024
    assert m.flops_per_token(512) < m.flops_per_token(1024)
