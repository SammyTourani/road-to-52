# Copyright 2026 The road-to-52 authors.
# SPDX-License-Identifier: Apache-2.0
#
# Hyper-parameter defaults and the parameter-group split are taken from
# KellerJordan/modded-nanogpt (MIT) -- see `configs/*.yaml` for the exact citation.
"""Optimizer construction, the WSD learning-rate schedule and gradient clipping.

Five parameter groups, routed by :func:`route` and wired together with
``mlx.optimizers.MultiOptimizer``:

=========  ================================================  ==========================
group      parameters                                         optimizer
=========  ================================================  ==========================
``muon``   every 2-D hidden matrix (attn q/k/v/out, MLP)      ``Muon``
``embed``  ``transformer.wte``                                ``AdamW``
``vembed`` ``value_embeds.*``                                 ``AdamW``
``head``   ``lm_head``                                        ``AdamW``
``scalar`` everything with ``ndim < 2`` (lambdas, skips)      ``AdamW``
=========  ================================================  ==========================

Every parameter lands in exactly one group; ``tests/test_optim.py`` asserts it.
"""

from __future__ import annotations

from dataclasses import dataclass

import mlx.core as mx
import mlx.optimizers as optim
from mlx.utils import tree_flatten

from .config import TrainConfig

__all__ = ["GROUPS", "R52Optimizer", "build_optimizer", "group_counts", "route", "wsd_multiplier"]

GROUPS = ("muon", "embed", "vembed", "head", "scalar")

_VEMBED_PREFIXES = ("value_embeds.",)
_EMBED_EXACT = ("transformer.wte.weight",)
_HEAD_PREFIXES = ("lm_head.",)


def route(name: str, param: mx.array) -> str:
    """Return the optimizer group (one of :data:`GROUPS`) for a flattened parameter path."""
    if param.ndim < 2:
        return "scalar"
    if name in _EMBED_EXACT:
        return "embed"
    if name.startswith(_VEMBED_PREFIXES):
        return "vembed"
    if name.startswith(_HEAD_PREFIXES):
        return "head"
    return "muon"


def group_counts(params: dict) -> dict[str, int]:
    """Map group name -> number of parameter tensors, for a parameter tree."""
    counts = dict.fromkeys(GROUPS, 0)
    for name, p in tree_flatten(params):
        counts[route(name, p)] += 1
    return counts


# --------------------------------------------------------------------------------------
# Schedule
# --------------------------------------------------------------------------------------


def wsd_multiplier(
    step: int,
    total_steps: int,
    warmup_steps: int = 0,
    cooldown_frac: float = 0.4,
    final_frac: float = 0.0,
) -> float:
    """Warmup-Stable-Decay LR multiplier in ``[final_frac, 1]`` for a 0-based ``step``.

    Linear warmup over ``warmup_steps``, constant 1.0, then a linear ramp from 1.0 to
    ``final_frac`` across the last ``cooldown_frac`` of ``total_steps``
    (modded-nanogpt's schedule, whose record uses ``final_frac = 0.15``).
    Steps beyond ``total_steps`` stay at ``final_frac``.
    """
    total_steps = max(1, int(total_steps))
    if warmup_steps > 0 and step < warmup_steps:
        return float(step + 1) / float(warmup_steps)
    cd_start = int(total_steps * (1.0 - cooldown_frac))
    if step < cd_start:
        return 1.0
    span = max(1, total_steps - cd_start)
    t = min(1.0, (step - cd_start) / span)
    return float((1.0 - t) + final_frac * t)


# --------------------------------------------------------------------------------------
# Optimizer
# --------------------------------------------------------------------------------------


@dataclass
class _Group:
    name: str
    opt: optim.Optimizer
    base_lr: float


class R52Optimizer:
    """A :class:`mlx.optimizers.MultiOptimizer` plus the WSD schedule and grad clipping.

    Usage per optimizer step::

        opt.set_step(step, total_steps)          # updates every group's learning rate
        grads, gnorm = opt.clip(grads)
        opt.update(model, grads)
    """

    def __init__(self, groups: list[_Group], cfg: TrainConfig) -> None:
        if not groups:
            raise ValueError("no optimizer groups (the model has no trainable parameters)")
        self.groups = groups
        self.cfg = cfg
        filters = [
            (lambda k, v, tag=g.name: route(k, v) == tag)
            for g in groups[:-1]
        ]
        self.multi = optim.MultiOptimizer([g.opt for g in groups], filters)
        self.lr_mult = 1.0

    # -- schedule -----------------------------------------------------------------
    def set_step(self, step: int, total_steps: int) -> float:
        """Set every group's LR for optimizer step ``step``; returns the WSD multiplier."""
        m = wsd_multiplier(
            step,
            total_steps,
            self.cfg.warmup_steps,
            self.cfg.cooldown_frac,
            self.cfg.final_lr_frac,
        )
        self.lr_mult = m
        for g in self.groups:
            g.opt.learning_rate = g.base_lr * m
        return m

    def learning_rates(self) -> dict[str, float]:
        """Current LR of each group (units: dimensionless, per optimizer step)."""
        return {g.name: float(g.opt.learning_rate) for g in self.groups}

    # -- step ---------------------------------------------------------------------
    def clip(self, grads: dict) -> tuple[dict, mx.array]:
        """Global-norm clip. Returns ``(grads, pre-clip global norm)``."""
        if self.cfg.grad_clip and self.cfg.grad_clip > 0:
            return optim.clip_grad_norm(grads, self.cfg.grad_clip)
        from mlx.utils import tree_reduce

        norm = mx.sqrt(tree_reduce(lambda acc, g: acc + g.square().sum(), grads, 0.0))
        return grads, norm

    def update(self, model, grads: dict) -> None:
        self.multi.update(model, grads)

    # -- state --------------------------------------------------------------------
    @property
    def state(self) -> dict:
        return self.multi.state

    @state.setter
    def state(self, value: dict) -> None:
        self.multi.state = value

    @property
    def step(self) -> int:
        """The first group's internal step counter (all groups advance together)."""
        return int(self.groups[0].opt.step)

    @property
    def group_names(self) -> list[str]:
        """Names of the live groups, in ``MultiOptimizer`` order (the last is the fallback)."""
        return [g.name for g in self.groups]


def build_optimizer(cfg: TrainConfig, params: dict | None = None) -> R52Optimizer:
    """Construct the Muon + 4x AdamW optimizer described by ``cfg``.

    Pass ``params`` (a parameter tree, e.g. ``model.trainable_parameters()``) to drop groups
    that would receive nothing.  This is not an optimisation: ``mlx.optimizers`` cannot
    initialise a sub-optimizer with an empty slice --
    ``tree_unflatten([])`` returns ``[]`` (a *list*), and ``Optimizer.init`` then indexes the
    optimizer's own ``{"step", "learning_rate"}`` dict as if it were that list and raises
    ``IndexError``.  A model with ``use_value_embeds=false`` hits this immediately.
    """
    muon = optim.Muon(
        learning_rate=cfg.muon_lr,
        momentum=cfg.muon_momentum,
        weight_decay=cfg.muon_weight_decay,
        nesterov=cfg.muon_nesterov,
        ns_steps=cfg.muon_ns_steps,
    )
    embed = optim.AdamW(
        learning_rate=cfg.adam_lr_embed,
        betas=list(cfg.adam_betas_embed),
        eps=cfg.adam_eps,
        weight_decay=cfg.adam_wd_embed,
    )
    vembed = optim.AdamW(
        learning_rate=cfg.adam_lr_vembed,
        betas=list(cfg.adam_betas_vembed),
        eps=cfg.adam_eps,
        weight_decay=cfg.adam_wd_vembed,
    )
    head = optim.AdamW(
        learning_rate=cfg.adam_lr_head,
        betas=list(cfg.adam_betas_head),
        eps=cfg.adam_eps,
        weight_decay=cfg.adam_wd_head,
    )
    scalar = optim.AdamW(
        learning_rate=cfg.adam_lr_scalar,
        betas=list(cfg.adam_betas_scalar),
        eps=cfg.adam_eps,
        weight_decay=cfg.adam_wd_scalar,
    )
    groups = [
        _Group("muon", muon, cfg.muon_lr),
        _Group("embed", embed, cfg.adam_lr_embed),
        _Group("vembed", vembed, cfg.adam_lr_vembed),
        _Group("head", head, cfg.adam_lr_head),
        _Group("scalar", scalar, cfg.adam_lr_scalar),
    ]
    # GROUPS order puts `scalar` last, so the MultiOptimizer fallback stays the catch-all
    # for anything `route()` does not classify.
    if params is not None:
        counts = group_counts(params)
        groups = [g for g in groups if counts[g.name] > 0]
    return R52Optimizer(groups, cfg)
