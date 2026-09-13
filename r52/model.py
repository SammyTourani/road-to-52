# Copyright 2026 The road-to-52 authors.
# SPDX-License-Identifier: Apache-2.0
#
# Architecture ported from ideas in (all MIT licensed, no code copied verbatim):
#   * karpathy/nanochat            -- RMSNorm pre-norm, QK-norm, ReLU^2 MLP, logit softcap 15,
#                                     untied zero-init lm_head, norm after wte / before lm_head.
#   * KellerJordan/modded-nanogpt  -- value embeddings with learned per-layer lambdas,
#                                     U-net style skip connections with learned weights,
#                                     zero-init output projections.
#   * ml-explore/mlx-lm            -- `mlx_lm/models/nanochat.py` weight layout and the
#                                     negated-precomputed-freqs RoPE call, so that exports
#                                     load natively in mlx-lm.
"""The road-to-52 GPT: a decoder-only transformer built on MLX fast kernels.

Precision modes (``ModelConfig.precision``)
-------------------------------------------
``mixed``  master parameters in fp32, every matmul executed in bf16 by casting the weight
           and the activation inside :class:`MPLinear`.  Gradients arrive back in fp32
           (``astype``'s VJP casts the cotangent), so the optimizer sees fp32 state.
           This is the mlxgpt lesson: pure bf16 is fast but converges worse; mixed
           precision is nearly as fast and matches fp32's loss curve.
``bf16``   parameters, activations and optimizer state all bf16 (fastest, worse loss).
``fp32``   everything fp32 (slowest, reference).

Shapes: ``(B, T)`` int32 token ids in, ``(B, T, vocab_size)`` logits out.
"""

from __future__ import annotations

import math

import mlx.core as mx
from mlx import nn
from mlx.nn.utils import checkpoint as _nn_checkpoint

from .config import ModelConfig

__all__ = ["GPT", "MLP", "Attention", "Block", "MPEmbedding", "MPLinear"]


def _dtypes(precision: str) -> tuple[mx.Dtype, mx.Dtype]:
    """Return ``(param_dtype, compute_dtype)`` for a precision mode."""
    if precision == "mixed":
        return mx.float32, mx.bfloat16
    if precision == "bf16":
        return mx.bfloat16, mx.bfloat16
    return mx.float32, mx.float32


def rms_norm(x: mx.array, eps: float = 1e-5) -> mx.array:
    """Parameter-free RMSNorm (``mx.fast.rms_norm`` with no learnable gain)."""
    return mx.fast.rms_norm(x, None, eps)


def softcap(logits: mx.array, cap: float) -> mx.array:
    """``cap * tanh(logits / cap)``; ``cap <= 0`` is a no-op."""
    if cap <= 0:
        return logits
    return cap * mx.tanh(logits / cap)


# --------------------------------------------------------------------------------------
# Mixed-precision primitives
# --------------------------------------------------------------------------------------


class MPLinear(nn.Module):
    """Bias-free linear layer with an fp32 master weight and a low-precision matmul.

    The weight is stored in ``param_dtype`` and cast to ``compute_dtype`` on every call;
    MLX's ``astype`` VJP casts the cotangent back, so ``weight.grad`` is ``param_dtype``.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        param_dtype: mx.Dtype = mx.float32,
        compute_dtype: mx.Dtype = mx.bfloat16,
        zero_init: bool = False,
    ) -> None:
        super().__init__()
        self.in_dim, self.out_dim = int(in_dim), int(out_dim)
        self._compute_dtype = compute_dtype
        if zero_init:
            w = mx.zeros((out_dim, in_dim))
        else:
            # modded-nanogpt CastedLinear.reset_parameters: uniform(-b, b), b = sqrt(3)*0.5/sqrt(fan_in)
            bound = math.sqrt(3.0) * 0.5 * in_dim**-0.5
            w = mx.random.uniform(-bound, bound, (out_dim, in_dim))
        self.weight = w.astype(param_dtype)

    def __call__(self, x: mx.array) -> mx.array:
        return x.astype(self._compute_dtype) @ self.weight.astype(self._compute_dtype).T


class MPEmbedding(nn.Module):
    """Token embedding with an fp32 master table and a ``compute_dtype`` output."""

    def __init__(
        self,
        vocab_size: int,
        dim: int,
        param_dtype: mx.Dtype = mx.float32,
        compute_dtype: mx.Dtype = mx.bfloat16,
    ) -> None:
        super().__init__()
        self._compute_dtype = compute_dtype
        # modded-nanogpt: uniform(-0.5, 0.5) * dim**-0.5
        self.weight = (mx.random.uniform(-0.5, 0.5, (vocab_size, dim)) * dim**-0.5).astype(param_dtype)

    def __call__(self, idx: mx.array) -> mx.array:
        return self.weight[idx].astype(self._compute_dtype)


# --------------------------------------------------------------------------------------
# Blocks
# --------------------------------------------------------------------------------------


class Attention(nn.Module):
    """Causal self-attention: QK-norm, cached-RoPE, fused SDPA, zero-init out projection."""

    def __init__(self, cfg: ModelConfig, use_value_embed: bool) -> None:
        super().__init__()
        pdt, cdt = _dtypes(cfg.precision)
        self.n_head = cfg.n_head
        self.n_kv_head = cfg.n_kv_head
        self.head_dim = cfg.head_dim
        self.scale = float(cfg.head_dim) ** -0.5
        self.qk_norm = cfg.qk_norm
        self.norm_eps = cfg.norm_eps
        self.use_value_embed = use_value_embed
        q_dim = cfg.n_head * cfg.head_dim
        kv_dim = cfg.n_kv_head * cfg.head_dim
        self.c_q = MPLinear(cfg.n_embd, q_dim, pdt, cdt)
        self.c_k = MPLinear(cfg.n_embd, kv_dim, pdt, cdt)
        self.c_v = MPLinear(cfg.n_embd, kv_dim, pdt, cdt)
        self.c_proj = MPLinear(q_dim, cfg.n_embd, pdt, cdt, zero_init=True)
        if use_value_embed:
            # modded-nanogpt: v = lambdas[0]*v + lambdas[1]*value_embedding
            self.lambdas = mx.array([0.5, 0.5], dtype=pdt)
        # Cached RoPE frequencies (leading underscore => a buffer, not a parameter).
        # Matches mlx_lm/models/nanochat.py exactly: traditional=False with negated freqs.
        half = cfg.head_dim // 2
        self._rope_freqs = -mx.exp(
            mx.arange(0.0, half, dtype=mx.float32) * (math.log(cfg.rope_base) / half)
        )

    def _rope(self, x: mx.array) -> mx.array:
        return mx.fast.rope(
            x, dims=self.head_dim, traditional=False, base=None, freqs=self._rope_freqs, scale=1.0, offset=0
        )

    def __call__(self, x: mx.array, ve: mx.array | None = None) -> mx.array:
        B, T, _ = x.shape
        q = self.c_q(x).reshape(B, T, self.n_head, self.head_dim).transpose(0, 2, 1, 3)
        k = self.c_k(x).reshape(B, T, self.n_kv_head, self.head_dim).transpose(0, 2, 1, 3)
        v = self.c_v(x).reshape(B, T, self.n_kv_head, self.head_dim).transpose(0, 2, 1, 3)
        q = self._rope(q)
        k = self._rope(k)
        if self.qk_norm:
            # RoPE is orthogonal, so rms_norm(rope(x)) == rope(rms_norm(x)); this order
            # matches nanochat / mlx-lm so exported weights are numerically identical.
            q = rms_norm(q, self.norm_eps)
            k = rms_norm(k, self.norm_eps)
        if ve is not None:
            ve = ve.reshape(B, T, self.n_kv_head, self.head_dim).transpose(0, 2, 1, 3)
            lam = self.lambdas.astype(v.dtype)
            v = lam[0] * v + lam[1] * ve.astype(v.dtype)
        o = mx.fast.scaled_dot_product_attention(q, k, v, scale=self.scale, mask="causal")
        o = o.transpose(0, 2, 1, 3).reshape(B, T, self.n_head * self.head_dim)
        return self.c_proj(o)


class MLP(nn.Module):
    """ReLU^2 MLP (nanochat / speedrun) or SwiGLU, with a zero-init down projection."""

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        pdt, cdt = _dtypes(cfg.precision)
        self.kind = cfg.mlp
        hidden = cfg.mlp_hidden
        if self.kind == "relu2":
            self.c_fc = MPLinear(cfg.n_embd, hidden, pdt, cdt)
            self.c_proj = MPLinear(hidden, cfg.n_embd, pdt, cdt, zero_init=True)
        else:
            self.gate_proj = MPLinear(cfg.n_embd, hidden, pdt, cdt)
            self.up_proj = MPLinear(cfg.n_embd, hidden, pdt, cdt)
            self.down_proj = MPLinear(hidden, cfg.n_embd, pdt, cdt, zero_init=True)

    def __call__(self, x: mx.array) -> mx.array:
        if self.kind == "relu2":
            return self.c_proj(nn.relu2(self.c_fc(x)))
        return self.down_proj(nn.silu(self.gate_proj(x)) * self.up_proj(x))


class Block(nn.Module):
    """Pre-norm transformer block: ``x + attn(norm(x))`` then ``h + mlp(norm(h))``."""

    def __init__(self, cfg: ModelConfig, use_value_embed: bool) -> None:
        super().__init__()
        self.norm_eps = cfg.norm_eps
        self.attn = Attention(cfg, use_value_embed)
        self.mlp = MLP(cfg)

    def __call__(self, x: mx.array, ve: mx.array | None = None) -> mx.array:
        h = x + self.attn(rms_norm(x, self.norm_eps), ve)
        return h + self.mlp(rms_norm(h, self.norm_eps))

    # Two arity-fixed entry points so `mlx.nn.utils.checkpoint` never sees a `None` leaf.
    def call_ve(self, x: mx.array, ve: mx.array) -> mx.array:
        return self(x, ve)

    def call_plain(self, x: mx.array) -> mx.array:
        return self(x, None)


class _Transformer(nn.Module):
    """Container that reproduces mlx-lm's ``nanochat`` naming (``transformer.wte``, ``transformer.h``)."""

    def __init__(self, cfg: ModelConfig, ve_layers: list[bool]) -> None:
        super().__init__()
        pdt, cdt = _dtypes(cfg.precision)
        self.wte = MPEmbedding(cfg.vocab_size, cfg.n_embd, pdt, cdt)
        self.h = [Block(cfg, ve_layers[i]) for i in range(cfg.n_layer)]


# --------------------------------------------------------------------------------------
# The model
# --------------------------------------------------------------------------------------


class GPT(nn.Module):
    """Decoder-only transformer, nanochat / modded-nanogpt lineage.

    Weight layout (mirrors ``mlx_lm.models.nanochat`` for the core tensors)::

        transformer.wte.weight                (vocab, n_embd)
        transformer.h.{i}.attn.c_q.weight     (n_head*head_dim, n_embd)
        transformer.h.{i}.attn.c_k.weight     (n_kv_head*head_dim, n_embd)
        transformer.h.{i}.attn.c_v.weight     (n_kv_head*head_dim, n_embd)
        transformer.h.{i}.attn.c_proj.weight  (n_embd, n_head*head_dim)     [zero-init]
        transformer.h.{i}.attn.lambdas        (2,)      [only on value-embedding layers]
        transformer.h.{i}.mlp.c_fc.weight     (mlp_hidden, n_embd)
        transformer.h.{i}.mlp.c_proj.weight   (n_embd, mlp_hidden)          [zero-init]
        value_embeds.{j}.weight               (vocab, n_kv_head*head_dim)    [optional]
        skip_weights                          (n_layer//2,)                 [optional]
        lm_head.weight                        (vocab, n_embd)               [zero-init]
    """

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        pdt, cdt = _dtypes(cfg.precision)
        self.param_dtype, self.compute_dtype = pdt, cdt

        # Which layers receive a value embedding, and from which table.
        self._ve_index: list[int | None] = [None] * cfg.n_layer
        if cfg.use_value_embeds:
            k = cfg.n_value_embeds
            n_tables = 1 if cfg.value_embed_share else k
            for j in range(k):
                self._ve_index[j] = 0 if cfg.value_embed_share else j
                self._ve_index[cfg.n_layer - k + j] = 0 if cfg.value_embed_share else j
            self.value_embeds = [
                MPEmbedding(cfg.vocab_size, cfg.n_kv_head * cfg.head_dim, pdt, cdt) for _ in range(n_tables)
            ]

        self.transformer = _Transformer(cfg, [i is not None for i in self._ve_index])
        self.lm_head = MPLinear(cfg.n_embd, cfg.vocab_size, pdt, cdt, zero_init=True)
        if cfg.tie_embeddings:
            raise NotImplementedError("tie_embeddings is not supported (the spec uses untied embeddings)")

        if cfg.use_unet_skips and cfg.n_encoder_layers > 0:
            self.skip_weights = mx.ones((cfg.n_encoder_layers,), dtype=pdt)

        self._block_fns = self._build_block_fns()

    # -- plumbing -----------------------------------------------------------------
    def _build_block_fns(self) -> list:
        """Per-block callables, optionally wrapped in gradient checkpointing."""
        fns = []
        for i, blk in enumerate(self.transformer.h):
            has_ve = self._ve_index[i] is not None
            if self.cfg.grad_checkpoint:
                fn = _nn_checkpoint(blk, blk.call_ve if has_ve else blk.call_plain)
            else:
                fn = blk.call_ve if has_ve else blk.call_plain
            fns.append(fn)
        return fns

    def set_grad_checkpoint(self, enabled: bool) -> None:
        """Turn per-block gradient checkpointing on/off (rebuilds the call wrappers)."""
        self.cfg.grad_checkpoint = bool(enabled)
        self._block_fns = self._build_block_fns()

    # -- forward ------------------------------------------------------------------
    def hidden(self, idx: mx.array) -> mx.array:
        """Final pre-``lm_head`` hidden state, shape ``(B, T, n_embd)``, in compute dtype."""
        cfg = self.cfg
        x = rms_norm(self.transformer.wte(idx), cfg.norm_eps)

        ves: list[mx.array | None] = [None] * cfg.n_layer
        if cfg.use_value_embeds:
            cache: dict[int, mx.array] = {}
            for i, j in enumerate(self._ve_index):
                if j is None:
                    continue
                if j not in cache:
                    cache[j] = self.value_embeds[j](idx)
                ves[i] = cache[j]

        n_enc = cfg.n_encoder_layers if cfg.use_unet_skips else 0
        skips: list[mx.array] = []
        for i in range(cfg.n_layer):
            if i >= n_enc and skips:
                # U-net: decoder layer d consumes the encoder residual in LIFO order.
                x = x + self.skip_weights[i - n_enc].astype(x.dtype) * skips.pop()
            ve = ves[i]
            x = self._block_fns[i](x, ve) if ve is not None else self._block_fns[i](x)
            if i < n_enc:
                skips.append(x)
        return rms_norm(x, cfg.norm_eps)

    def logits(self, idx: mx.array) -> mx.array:
        """Softcapped logits in the model's **compute** dtype, shape ``(B, T, vocab_size)``."""
        return softcap(self.lm_head(self.hidden(idx)), self.cfg.softcap)

    def __call__(self, idx: mx.array) -> mx.array:
        """``(B, T)`` int32 token ids -> ``(B, T, vocab_size)`` **fp32** softcapped logits."""
        return self.logits(idx).astype(mx.float32)

    # -- loss ---------------------------------------------------------------------
    def loss(self, idx: mx.array, targets: mx.array, fp32_logits: bool | None = None) -> mx.array:
        """Mean cross-entropy in nats (fp32 scalar), ``ignore_index=-1``.

        ``fp32_logits`` upcasts the logits before the log-sum-exp.  It defaults to
        ``False`` in mixed/bf16 mode (the log-sum-exp then runs in bf16, the same choice
        mlx-lm's trainer makes -- measured bias on random logits: -5e-4 nats, and it
        halves the peak memory of the ``(B, T, vocab)`` tensor).  Validation passes
        ``True`` so the reported number is exact.
        """
        if fp32_logits is None:
            fp32_logits = self.compute_dtype == mx.float32
        logits = self.logits(idx)
        if fp32_logits:
            logits = logits.astype(mx.float32)
        lse = mx.logsumexp(logits, axis=-1)
        safe = mx.maximum(targets, 0)
        picked = mx.take_along_axis(logits, safe[..., None], axis=-1).squeeze(-1)
        nll = (lse - picked).astype(mx.float32)
        mask = (targets >= 0).astype(mx.float32)
        denom = mx.maximum(mask.sum(), 1.0)
        return (nll * mask).sum() / denom

    # -- bookkeeping ---------------------------------------------------------------
    def _param_items(self) -> list[tuple[str, mx.array]]:
        from mlx.utils import tree_flatten

        return tree_flatten(self.parameters())

    def num_params(self, non_embedding: bool = False) -> int:
        """Total parameter count. ``non_embedding`` drops ``wte``, ``lm_head`` and value embeds."""
        total = 0
        for name, p in self._param_items():
            if non_embedding and (
                name.startswith("value_embeds.")
                or name.startswith("lm_head.")
                or name == "transformer.wte.weight"
            ):
                continue
            total += int(p.size)
        return total

    def num_params_flops(self) -> int:
        """Parameters that participate in a matmul: non-embedding + ``lm_head``.

        For a GPT-2-124M-shaped config this equals the familiar 123.6M (the untied
        ``lm_head`` replaces the tied ``wte`` in the classic count), so ``6*N`` stays
        comparable with published nanoGPT/modded-nanogpt MFU numbers.
        """
        return self.num_params(non_embedding=True) + int(self.lm_head.weight.size)

    def flops_per_token(self, block_size: int | None = None) -> int:
        """Forward+backward FLOPs per token.

        ``6 * N + 12 * n_layer * n_embd * T`` where ``N = num_params_flops()`` and the
        second term is the attention score/value matmuls (PaLM appendix B / nanoGPT
        ``estimate_mfu``).  ``T`` defaults to ``cfg.block_size``.
        """
        T = self.cfg.block_size if block_size is None else int(block_size)
        return 6 * self.num_params_flops() + 12 * self.cfg.n_layer * self.cfg.n_embd * T

    def param_bytes(self) -> int:
        """Bytes held by the parameters (weights only, no optimizer state)."""
        return sum(int(p.nbytes) for _, p in self._param_items())
