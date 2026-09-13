# Copyright 2026 The road-to-52 authors.
# SPDX-License-Identifier: Apache-2.0
"""Guard against the classic from-scratch Muon bug: orthogonalising a FUSED projection.

Newton-Schulz orthogonalises whatever 2-D matrix it is handed.  If q/k/v (or gate/up) are
fused into one ``Linear``, Muon orthogonalises the concatenation instead of each
projection, which silently degrades training.  Our model keeps them as separate layers;
these tests assert that, and that Muon only ever sees per-projection matrices.
"""

from __future__ import annotations

import re

import mlx.core as mx
from conftest import tiny_model_config
from mlx.utils import tree_flatten

from r52.model import GPT
from r52.optim import route

_EXPECTED_RELU2 = {"attn.c_q", "attn.c_k", "attn.c_v", "attn.c_proj", "mlp.c_fc", "mlp.c_proj"}
_EXPECTED_SWIGLU = {"attn.c_q", "attn.c_k", "attn.c_v", "attn.c_proj",
                    "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"}


def _muon_params(model: GPT) -> dict[str, mx.array]:
    return {k: v for k, v in tree_flatten(model.trainable_parameters()) if route(k, v) == "muon"}


def test_qkv_and_mlp_are_separate_matrices() -> None:
    model = GPT(tiny_model_config(n_layer=3))
    names = {re.sub(r"^transformer\.h\.\d+\.", "", k).removesuffix(".weight")
             for k in _muon_params(model)}
    assert names == _EXPECTED_RELU2
    model = GPT(tiny_model_config(n_layer=3, mlp="swiglu"))
    names = {re.sub(r"^transformer\.h\.\d+\.", "", k).removesuffix(".weight")
             for k in _muon_params(model)}
    assert names == _EXPECTED_SWIGLU


def test_muon_matrix_shapes_are_per_projection_not_fused() -> None:
    """A fused qkv would be (3*q_dim, n_embd); each of ours must be (q_dim, n_embd)."""
    cfg = tiny_model_config(n_layer=2, n_embd=64, n_head=4, n_kv_head=2)
    model = GPT(cfg)
    q_dim = cfg.n_head * cfg.head_dim
    kv_dim = cfg.n_kv_head * cfg.head_dim
    shapes = {k: tuple(v.shape) for k, v in _muon_params(model).items()}
    for i in range(cfg.n_layer):
        p = f"transformer.h.{i}."
        assert shapes[p + "attn.c_q.weight"] == (q_dim, cfg.n_embd)
        assert shapes[p + "attn.c_k.weight"] == (kv_dim, cfg.n_embd)
        assert shapes[p + "attn.c_v.weight"] == (kv_dim, cfg.n_embd)
        assert shapes[p + "attn.c_proj.weight"] == (cfg.n_embd, q_dim)
        assert shapes[p + "mlp.c_fc.weight"] == (cfg.mlp_hidden, cfg.n_embd)
        assert shapes[p + "mlp.c_proj.weight"] == (cfg.n_embd, cfg.mlp_hidden)
    assert all(len(s) == 2 for s in shapes.values())


def test_muon_never_sees_an_embedding_or_a_scalar() -> None:
    model = GPT(tiny_model_config(n_layer=3))
    for name, p in _muon_params(model).items():
        assert p.ndim == 2
        assert not name.startswith(("value_embeds.", "lm_head."))
        assert name != "transformer.wte.weight"
        assert name.startswith("transformer.h."), name


def test_batch_size_is_constant_no_batch_warmup() -> None:
    """With Muon we start at the target batch; `tokens_per_step` must not be scheduled."""
    from pathlib import Path

    from r52.config import load_config

    for path in sorted((Path(__file__).resolve().parent.parent / "configs").glob("*.yaml")):
        cfg = load_config(path)
        assert isinstance(cfg.train.tokens_per_step, int) and cfg.train.tokens_per_step > 0
        assert cfg.grad_accum() * cfg.train.micro_batch * cfg.model.block_size == cfg.train.tokens_per_step
