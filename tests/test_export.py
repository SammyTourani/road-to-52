# Copyright 2026 The road-to-52 authors.
# SPDX-License-Identifier: Apache-2.0
"""Export: mlx-lm loads what we write, and the logits match ours."""

from __future__ import annotations

import json
from pathlib import Path

import mlx.core as mx
import pytest
from conftest import tiny_model_config
from mlx.utils import tree_map

from r52.config import Config
from r52.export import export_checkpoint, export_model, load_exported, model_config_to_mlx_lm, ve_index
from r52.model import GPT

pytest.importorskip("mlx_lm")


def _perturbed(cfg) -> GPT:
    """A model with non-trivial weights (a zero-init lm_head would make every logit 0)."""
    m = GPT(cfg)
    m.update(tree_map(lambda p: p + mx.random.normal(p.shape) * 0.05, m.parameters()))
    mx.eval(m.parameters())
    return m


@pytest.mark.parametrize(
    "kw,expected_type",
    [
        ({}, "r52gpt"),
        ({"value_embed_share": False, "n_value_embeds": 1}, "r52gpt"),
        ({"use_unet_skips": False}, "r52gpt"),
        ({"use_value_embeds": False, "use_unet_skips": False}, "nanochat"),
        ({"mlp": "swiglu"}, "r52gpt"),
        ({"n_head": 4, "n_kv_head": 2}, "r52gpt"),
    ],
)
def test_round_trip_logits_match_mlx_lm(tmp_path: Path, kw, expected_type) -> None:
    cfg = tiny_model_config(**kw)
    model = _perturbed(cfg)
    out = export_model(model, tmp_path / "m", Config(model=cfg), dtype="bfloat16")

    conf = json.loads((out / "config.json").read_text())
    assert conf["model_type"] == expected_type
    assert (out / "r52gpt.py").exists() == (expected_type == "r52gpt")

    theirs, _ = load_exported(out)
    idx = mx.random.randint(0, cfg.vocab_size, (8, cfg.block_size))  # 8 prompts, per spec S7
    ours = model(idx)
    other = theirs(idx).astype(mx.float32)
    mx.eval(ours, other)
    assert float(mx.abs(ours - other).max()) < 1e-2


def test_exported_model_generates_with_a_kv_cache(tmp_path: Path) -> None:
    """The plugin must support incremental decoding, or mlx_lm.generate cannot use it."""
    from mlx_lm.models.cache import make_prompt_cache

    cfg = tiny_model_config()
    model = _perturbed(cfg)
    out = export_model(model, tmp_path / "m", Config(model=cfg))
    theirs, _ = load_exported(out)

    idx = mx.random.randint(0, cfg.vocab_size, (1, 8))
    full = theirs(idx)
    cache = make_prompt_cache(theirs)
    step = None
    for t in range(idx.shape[1]):
        step = theirs(idx[:, t : t + 1], cache=cache)
    mx.eval(full, step)
    assert float(mx.abs(full[:, -1] - step[:, -1]).max()) < 5e-2


def test_export_from_checkpoint_directory(tmp_path: Path) -> None:
    from r52.checkpoint import save_checkpoint

    cfg = tiny_model_config()
    model = _perturbed(cfg)
    full = Config(model=cfg)
    save_checkpoint(tmp_path / "ckpt", model, step=7, config=full)
    out = export_checkpoint(tmp_path / "ckpt", tmp_path / "m", tokenizer=False)
    theirs, _ = load_exported(out)
    idx = mx.random.randint(0, cfg.vocab_size, (2, cfg.block_size))
    mx.eval(theirs(idx))


def test_ve_index_mirrors_first_and_last_layers() -> None:
    cfg = tiny_model_config(n_layer=12, n_value_embeds=3, value_embed_share=False)
    assert ve_index(cfg) == [0, 1, 2, -1, -1, -1, -1, -1, -1, 0, 1, 2]
    cfg = tiny_model_config(n_layer=12, n_value_embeds=3, value_embed_share=True)
    assert ve_index(cfg) == [0, 0, 0, -1, -1, -1, -1, -1, -1, 0, 0, 0]
    cfg = tiny_model_config(n_layer=12, use_value_embeds=False)
    assert ve_index(cfg) == [-1] * 12


def test_config_json_carries_the_full_r52_config() -> None:
    cfg = tiny_model_config()
    conf = model_config_to_mlx_lm(Config(name="x", model=cfg))
    assert conf["r52"]["model"]["n_layer"] == cfg.n_layer
    assert conf["hidden_size"] == cfg.n_embd
    assert conf["intermediate_size"] == cfg.mlp_hidden
