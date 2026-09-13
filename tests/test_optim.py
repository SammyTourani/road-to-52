# Copyright 2026 The road-to-52 authors.
# SPDX-License-Identifier: Apache-2.0
"""Optimizer routing (every parameter exactly once), the WSD schedule, and clipping."""

from __future__ import annotations

import mlx.core as mx
import pytest
from conftest import tiny_model_config
from mlx.utils import tree_flatten

from r52.config import ModelConfig, TrainConfig
from r52.model import GPT
from r52.optim import GROUPS, build_optimizer, group_counts, route, wsd_multiplier


@pytest.mark.parametrize(
    "kw",
    [
        {},
        {"use_value_embeds": False},
        {"use_unet_skips": False},
        {"use_value_embeds": False, "use_unet_skips": False},
        {"mlp": "swiglu"},
        {"n_head": 4, "n_kv_head": 2},
    ],
)
def test_every_parameter_routed_exactly_once(kw) -> None:
    m = GPT(tiny_model_config(**kw))
    params = tree_flatten(m.trainable_parameters())
    groups = [route(k, v) for k, v in params]
    assert len(groups) == len(params)
    assert set(groups) <= set(GROUPS)
    counts = group_counts(m.trainable_parameters())
    assert sum(counts.values()) == len(params)

    # And the MultiOptimizer's own split must agree with route().
    opt = build_optimizer(TrainConfig())
    parts = opt.multi._split_dictionary(m.trainable_parameters())
    seen: list[str] = []
    for g, part in zip(GROUPS, parts, strict=True):
        for k, _ in tree_flatten(part):
            assert route(k, dict(params)[k]) == g
            seen.append(k)
    assert sorted(seen) == sorted(k for k, _ in params)


def test_routing_rules_are_what_we_documented() -> None:
    two_d, one_d = mx.zeros((4, 4)), mx.zeros((4,))
    assert route("transformer.wte.weight", two_d) == "embed"
    assert route("value_embeds.0.weight", two_d) == "vembed"
    assert route("lm_head.weight", two_d) == "head"
    assert route("transformer.h.0.attn.c_q.weight", two_d) == "muon"
    assert route("transformer.h.0.mlp.c_fc.weight", two_d) == "muon"
    assert route("skip_weights", one_d) == "scalar"
    assert route("transformer.h.0.attn.lambdas", one_d) == "scalar"


def test_124m_routing_counts() -> None:
    m = GPT(ModelConfig(n_layer=12, n_embd=768, n_head=6, n_value_embeds=3, value_embed_share=True))
    counts = group_counts(m.trainable_parameters())
    assert counts == {"muon": 12 * 6, "embed": 1, "vembed": 1, "head": 1, "scalar": 6 + 1}


def test_wsd_schedule_shape() -> None:
    f = lambda s: wsd_multiplier(s, total_steps=100, warmup_steps=10, cooldown_frac=0.4,
                                 final_frac=0.15)
    assert f(0) == pytest.approx(0.1)
    assert f(9) == pytest.approx(1.0)
    assert f(10) == pytest.approx(1.0)
    assert f(59) == pytest.approx(1.0)
    assert f(60) == pytest.approx(1.0)
    assert f(99) == pytest.approx(0.15 + 0.85 / 40, abs=1e-6)
    assert f(100) == pytest.approx(0.15)
    assert f(500) == pytest.approx(0.15)
    ms = [f(s) for s in range(100)]
    assert ms == sorted(ms[:10]) + ms[10:]          # warmup monotone up
    assert ms[60:] == sorted(ms[60:], reverse=True)  # cooldown monotone down


def test_zero_warmup_and_zero_cooldown() -> None:
    assert wsd_multiplier(0, 100, 0, 0.0, 0.0) == 1.0
    assert wsd_multiplier(99, 100, 0, 0.0, 0.0) == 1.0


def test_set_step_scales_every_group_lr() -> None:
    tc = TrainConfig(warmup_steps=0, cooldown_frac=0.5, final_lr_frac=0.0)
    opt = build_optimizer(tc)
    opt.set_step(0, 10)
    base = opt.learning_rates()
    assert base["muon"] == pytest.approx(tc.muon_lr, rel=1e-6)
    assert base["head"] == pytest.approx(tc.adam_lr_head, rel=1e-6)
    opt.set_step(9, 10)
    late = opt.learning_rates()
    assert late["muon"] < base["muon"] and late["embed"] < base["embed"]
    assert late["muon"] / base["muon"] == pytest.approx(late["head"] / base["head"], rel=1e-4)


def test_grad_clip() -> None:
    opt = build_optimizer(TrainConfig(grad_clip=1.0))
    grads = {"a": mx.ones((4,)) * 10.0, "b": mx.ones((3,)) * 10.0}
    clipped, norm = opt.clip(grads)
    mx.eval(clipped, norm)
    total = float(mx.sqrt(sum(mx.square(v).sum() for v in clipped.values())))
    assert float(norm) == pytest.approx((10.0**2 * 7) ** 0.5, rel=1e-5)
    assert total == pytest.approx(1.0, rel=1e-4)


def test_grad_clip_disabled_reports_norm_without_scaling() -> None:
    opt = build_optimizer(TrainConfig(grad_clip=0.0))
    grads = {"a": mx.ones((4,)) * 10.0}
    clipped, norm = opt.clip(grads)
    mx.eval(clipped, norm)
    assert float(norm) == pytest.approx(20.0, rel=1e-5)
    assert float(clipped["a"][0]) == pytest.approx(10.0)


@pytest.mark.parametrize(
    "kw,expected",
    [
        ({}, ["muon", "embed", "vembed", "head", "scalar"]),
        ({"use_value_embeds": False}, ["muon", "embed", "head", "scalar"]),
        ({"use_value_embeds": False, "use_unet_skips": False}, ["muon", "embed", "head"]),
        ({"use_unet_skips": False}, ["muon", "embed", "vembed", "head", "scalar"]),
    ],
)
def test_empty_groups_are_dropped_and_a_step_runs(kw, expected) -> None:
    """mlx's MultiOptimizer cannot initialise a sub-optimizer with an empty slice."""
    import mlx.nn as nn
    from conftest import tiny_model_config

    model = GPT(tiny_model_config(**kw))
    opt = build_optimizer(TrainConfig(), model.trainable_parameters())
    assert opt.group_names == expected
    x = mx.random.randint(0, model.cfg.vocab_size, (2, model.cfg.block_size))
    _, grads = nn.value_and_grad(model, lambda a, b: model.loss(a, b))(x, x)
    opt.set_step(0, 10)
    grads, _ = opt.clip(grads)
    opt.update(model, grads)
    mx.eval(model.parameters(), opt.state)
    assert set(opt.learning_rates()) == set(expected)
