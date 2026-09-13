# Copyright 2026 The road-to-52 authors.
# SPDX-License-Identifier: Apache-2.0
#
# Inference-only mlx-lm model file for road-to-52 GPTs.  Structure follows
# `mlx_lm/models/nanochat.py` (MIT, Apple Inc.) so the weight names match; the value
# embeddings and U-net skip connections are the modded-nanogpt (MIT) additions.
#
# mlx-lm loads this file because the exported `config.json` sets
#   {"model_type": "r52gpt", "model_file": "r52gpt.py"}
# and `mlx_lm.utils.load_model` honours `model_file`.
"""road-to-52 GPT for mlx-lm (generate / server / evaluate / lora)."""

import math
from dataclasses import dataclass, field

import mlx.core as mx
from mlx import nn
from mlx_lm.models.base import BaseModelArgs, create_attention_mask, scaled_dot_product_attention


@dataclass
class ModelArgs(BaseModelArgs):
    model_type: str = "r52gpt"
    hidden_size: int = 768
    num_hidden_layers: int = 12
    num_attention_heads: int = 6
    num_key_value_heads: int = 6
    head_dim: int = 128
    vocab_size: int = 50304
    max_position_embeddings: int = 1024
    intermediate_size: int = 3072
    rope_theta: float = 10000.0
    softcap: float = 15.0
    qk_norm: bool = True
    mlp: str = "relu2"
    norm_eps: float = 1e-5
    use_value_embeds: bool = True
    n_value_embeds: int = 3
    value_embed_share: bool = True
    use_unet_skips: bool = True
    ve_index: list[int] = field(default_factory=list)
    """Per-layer value-embedding table index (``-1`` = none). Length ``num_hidden_layers``."""


def rms_norm(x, eps=1e-5):
    return mx.fast.rms_norm(x, None, eps)


class Attention(nn.Module):
    def __init__(self, args, use_value_embed):
        super().__init__()
        self.n_head = args.num_attention_heads
        self.n_kv_head = args.num_key_value_heads
        self.head_dim = args.head_dim
        self.scale = self.head_dim**-0.5
        self.qk_norm = args.qk_norm
        self.norm_eps = args.norm_eps
        self.use_value_embed = use_value_embed
        q_dim = self.n_head * self.head_dim
        kv_dim = self.n_kv_head * self.head_dim
        self.c_q = nn.Linear(args.hidden_size, q_dim, bias=False)
        self.c_k = nn.Linear(args.hidden_size, kv_dim, bias=False)
        self.c_v = nn.Linear(args.hidden_size, kv_dim, bias=False)
        self.c_proj = nn.Linear(q_dim, args.hidden_size, bias=False)
        if use_value_embed:
            self.lambdas = mx.array([0.5, 0.5])
        half = self.head_dim // 2
        self._rope_freqs = -mx.exp(
            mx.arange(0.0, half, dtype=mx.float32) * (math.log(args.rope_theta) / half)
        )

    def _rope(self, x, offset):
        return mx.fast.rope(
            x, dims=self.head_dim, traditional=False, base=None,
            freqs=self._rope_freqs, scale=1.0, offset=offset,
        )

    def __call__(self, x, mask=None, cache=None, ve=None):
        B, L, _ = x.shape
        q = self.c_q(x).reshape(B, L, self.n_head, self.head_dim).transpose(0, 2, 1, 3)
        k = self.c_k(x).reshape(B, L, self.n_kv_head, self.head_dim).transpose(0, 2, 1, 3)
        v = self.c_v(x).reshape(B, L, self.n_kv_head, self.head_dim).transpose(0, 2, 1, 3)
        offset = cache.offset if cache is not None else 0
        q = self._rope(q, offset)
        k = self._rope(k, offset)
        if self.qk_norm:
            q = rms_norm(q, self.norm_eps)
            k = rms_norm(k, self.norm_eps)
        if ve is not None:
            ve = ve.reshape(B, L, self.n_kv_head, self.head_dim).transpose(0, 2, 1, 3)
            lam = self.lambdas.astype(v.dtype)
            v = lam[0] * v + lam[1] * ve.astype(v.dtype)
        if cache is not None:
            k, v = cache.update_and_fetch(k, v)
        o = scaled_dot_product_attention(q, k, v, cache=cache, scale=self.scale, mask=mask)
        o = o.transpose(0, 2, 1, 3).reshape(B, L, self.n_head * self.head_dim)
        return self.c_proj(o)


class MLP(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.kind = args.mlp
        if self.kind == "relu2":
            self.c_fc = nn.Linear(args.hidden_size, args.intermediate_size, bias=False)
            self.c_proj = nn.Linear(args.intermediate_size, args.hidden_size, bias=False)
        else:
            self.gate_proj = nn.Linear(args.hidden_size, args.intermediate_size, bias=False)
            self.up_proj = nn.Linear(args.hidden_size, args.intermediate_size, bias=False)
            self.down_proj = nn.Linear(args.intermediate_size, args.hidden_size, bias=False)

    def __call__(self, x):
        if self.kind == "relu2":
            return self.c_proj(nn.relu2(self.c_fc(x)))
        return self.down_proj(nn.silu(self.gate_proj(x)) * self.up_proj(x))


class TransformerBlock(nn.Module):
    def __init__(self, args, use_value_embed):
        super().__init__()
        self.norm_eps = args.norm_eps
        self.attn = Attention(args, use_value_embed)
        self.mlp = MLP(args)

    def __call__(self, x, mask=None, cache=None, ve=None):
        h = x + self.attn(rms_norm(x, self.norm_eps), mask=mask, cache=cache, ve=ve)
        return h + self.mlp(rms_norm(h, self.norm_eps))


class R52Transformer(nn.Module):
    def __init__(self, args, ve_index):
        super().__init__()
        self.wte = nn.Embedding(args.vocab_size, args.hidden_size)
        self.h = [TransformerBlock(args, ve_index[i] >= 0) for i in range(args.num_hidden_layers)]


class Model(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.model_type = args.model_type
        n = args.num_hidden_layers
        self._ve_index = list(args.ve_index) if args.ve_index else [-1] * n
        if len(self._ve_index) != n:
            raise ValueError("ve_index must have num_hidden_layers entries")
        self.transformer = R52Transformer(args, self._ve_index)
        if args.use_value_embeds:
            n_tables = max(self._ve_index) + 1
            self.value_embeds = [
                nn.Embedding(args.vocab_size, args.num_key_value_heads * args.head_dim)
                for _ in range(max(1, n_tables))
            ]
        self.lm_head = nn.Linear(args.hidden_size, args.vocab_size, bias=False)
        self.n_enc = n // 2 if args.use_unet_skips else 0
        if self.n_enc:
            self.skip_weights = mx.ones((self.n_enc,))

    def __call__(self, inputs, cache=None):
        args = self.args
        h = rms_norm(self.transformer.wte(inputs), args.norm_eps)
        if cache is None:
            cache = [None] * len(self.transformer.h)
        mask = create_attention_mask(h, cache[0])

        ve_cache = {}
        skips = []
        for i, (layer, c) in enumerate(zip(self.transformer.h, cache, strict=False)):
            if i >= self.n_enc and skips:
                h = h + self.skip_weights[i - self.n_enc].astype(h.dtype) * skips.pop()
            j = self._ve_index[i]
            ve = None
            if j >= 0:
                if j not in ve_cache:
                    ve_cache[j] = self.value_embeds[j](inputs)
                ve = ve_cache[j]
            h = layer(h, mask=mask, cache=c, ve=ve)
            if i < self.n_enc:
                skips.append(h)

        h = rms_norm(h, args.norm_eps)
        logits = self.lm_head(h)
        if args.softcap and args.softcap > 0:
            logits = args.softcap * mx.tanh(logits / args.softcap)
        return logits

    @property
    def layers(self):
        return self.transformer.h
