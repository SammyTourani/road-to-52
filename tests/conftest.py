# Copyright 2026 The road-to-52 authors.
# SPDX-License-Identifier: Apache-2.0
"""Shared fixtures. Keeps every test tiny, GPU-light and offline."""

from __future__ import annotations

import mlx.core as mx
import pytest

from r52.config import Config, DataConfig, ModelConfig, TrainConfig

# The whole suite must stay well inside the machine's shared-GPU budget.
mx.set_memory_limit(2 * (1 << 30))
mx.set_cache_limit(1 << 29)


def tiny_model_config(**kw) -> ModelConfig:
    """2-layer, 64-dim, 256-token-vocab model: ~100k non-embedding parameters."""
    base = {
        "n_layer": 2, "n_embd": 64, "n_head": 2, "vocab_size": 256, "block_size": 32,
        "n_value_embeds": 1, "value_embed_share": True,
    }
    base.update(kw)
    return ModelConfig(**base)


def tiny_config(**kw) -> Config:
    """A full synthetic-data run config sized for a sub-second test."""
    model = tiny_model_config(**kw.pop("model", {}))
    data = DataConfig(source="synthetic", synthetic_tokens=60_000, seed=7, **kw.pop("data", {}))
    train = TrainConfig(
        micro_batch=4, tokens_per_step=4 * model.block_size, max_tokens=4 * model.block_size * 30,
        warmup_steps=0, cooldown_frac=0.4, final_lr_frac=0.15, grad_clip=1.0,
        log_interval=10, val_interval=0, ckpt_interval=0, val_max_batches=2,
        memory_limit_gb=2.0, cache_limit_gb=0.5, compile=False, seed=7,
        **kw.pop("train", {}),
    )
    return Config(name="tiny", model=model, data=data, train=train)


@pytest.fixture
def tiny_cfg() -> Config:
    return tiny_config()


@pytest.fixture(autouse=True)
def _deterministic():
    mx.random.seed(0)
    yield
    mx.clear_cache()
