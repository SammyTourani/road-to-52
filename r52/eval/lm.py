# Copyright 2026 The road-to-52 authors.
# SPDX-License-Identifier: Apache-2.0
"""One logits interface for every evaluation in :mod:`r52.eval`.

:class:`LM` wraps either

* an **r52 checkpoint directory** (``runs/<name>/ckpt/best``: ``model.safetensors`` +
  ``meta.json``), loaded into :class:`r52.model.GPT`, or
* an **mlx-lm model directory** (``config.json`` + ``model*.safetensors``), loaded with
  ``mlx_lm.utils.load_model``.  That covers our own exports -- ``model_type: "nanochat"``
  natively and ``model_type: "r52gpt"`` through the ``r52gpt.py`` plugin file the export
  copies next to the weights -- and the GPT-2 reference produced by
  ``python -m r52.export --gpt2-reference models/gpt2-mlx``.

so that the same eval code produces apples-to-apples numbers for both.

Precision
---------
Compute runs in the model's own dtype (**bf16** for a mixed-precision ``GPT`` and for every
``dtype=bfloat16`` export, which is what all our numbers are measured at).  Every *reduction*
-- the log-sum-exp and therefore every log-probability, loss and bits-per-byte -- is done in
**fp32**, matching ``GPT.loss(..., fp32_logits=True)``, which is the call ``r52.train`` uses
for the validation numbers it reports.

Units
-----
``nll`` is in **nats per token**.  ``max_context`` and every ``tokens`` argument are in
**tokens**.  ``max_positions_per_forward`` is in ``rows * tokens`` positions and exists only
to bound peak memory: an fp32 logits tensor costs ``positions * vocab_size * 4`` bytes
(4,096 positions x 50,304 x 4 = 824 MB), so it is the one knob that decides whether an eval
fits inside the memory cap while a pretraining run owns the rest of the GPU.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import mlx.core as mx

from ..tokenizer import GPT2Tokenizer

__all__ = [
    "DEFAULT_CACHE_LIMIT_GIB",
    "DEFAULT_MEMORY_LIMIT_GIB",
    "LM",
    "configure_runtime",
    "pad_stack",
]

DEFAULT_MEMORY_LIMIT_GIB = 3.0
"""MLX memory guideline for eval processes (GiB).

Half of pretraining's 6 GiB cap (``docs/ARCHITECTURE.md`` §1.2): evals are expected to run
*alongside* a multi-day training run, so they get the smaller half of the budget.
"""

DEFAULT_CACHE_LIMIT_GIB = 1.0
"""MLX buffer-cache limit (GiB), same value the trainer uses."""

_MODEL_FILE = "model.safetensors"
_META_FILE = "meta.json"
_CONFIG_FILE = "config.json"


def configure_runtime(
    memory_limit_gib: float = DEFAULT_MEMORY_LIMIT_GIB,
    cache_limit_gib: float = DEFAULT_CACHE_LIMIT_GIB,
) -> None:
    """Apply the MLX memory guidelines every eval entry point must set.

    ``mx.set_memory_limit`` is a guideline, not a hard cap (``docs/DEVIATIONS.md`` §7): MLX
    only raises once RAM *and* swap are exhausted.  Keeping the per-forward position budget
    small is what actually bounds peak memory; this call is the backstop.
    """
    mx.set_memory_limit(int(memory_limit_gib * 2**30))
    mx.set_cache_limit(int(cache_limit_gib * 2**30))


def pad_stack(rows: list[list[int]], pad_id: int = 0) -> mx.array:
    """Right-pad ragged token rows into one ``(len(rows), max_len)`` int32 array."""
    width = max(len(r) for r in rows)
    out = [r + [pad_id] * (width - len(r)) for r in rows]
    return mx.array(out, dtype=mx.int32)


def _is_r52_checkpoint(path: Path) -> bool:
    return (path / _META_FILE).is_file() and (path / _MODEL_FILE).is_file()


def _is_mlx_lm_dir(path: Path) -> bool:
    return (path / _CONFIG_FILE).is_file() and any(path.glob("model*.safetensors"))


class LM:
    """A loaded language model exposing batched fp32 logits over GPT-2 token ids.

    Attributes:
        model: the underlying ``nn.Module`` (an :class:`r52.model.GPT` or an mlx-lm model).
        max_context: usable context length in **tokens**.  Sequences longer than this must be
            truncated by the caller (HellaSwag and CORE do, keeping the *last* ``max_context``
            tokens, as nanochat does).
        tokenizer: :class:`r52.tokenizer.GPT2Tokenizer` -- GPT-2 tiktoken, the tokenizer for
            everything in this suite (the FineWeb shards, the GPT-2 reference and our own
            models all use it).
        vocab_size: rows of the model's output layer (50,304 for ours, 50,257 for GPT-2).
        kind: ``"r52-checkpoint"`` or ``"mlx-lm"``.
        name: a short identifier for the results JSON (the directory name).
    """

    def __init__(
        self,
        model: Any,
        max_context: int,
        *,
        vocab_size: int,
        kind: str,
        name: str,
        path: Path | None = None,
        config: dict[str, Any] | None = None,
        max_positions_per_forward: int = 4096,
        fp32_chunk_positions: int = 1024,
    ) -> None:
        self.model = model
        self.max_context = int(max_context)
        self.vocab_size = int(vocab_size)
        self.kind = kind
        self.name = name
        self.path = path
        self.config = config or {}
        self.max_positions_per_forward = int(max_positions_per_forward)
        self.fp32_chunk_positions = int(fp32_chunk_positions)
        self.tokenizer = GPT2Tokenizer()

    # -- construction ------------------------------------------------------------------
    @classmethod
    def load(
        cls,
        path: str | Path,
        *,
        max_positions_per_forward: int = 4096,
        block_size: int | None = None,
    ) -> LM:
        """Load an r52 checkpoint directory or an mlx-lm model directory.

        Args:
            path: ``runs/<run>/ckpt/best`` (r52 checkpoint) or ``models/<name>-mlx``
                (mlx-lm directory, including ``models/gpt2-mlx``).
            max_positions_per_forward: peak-memory knob, in ``rows * tokens``.
            block_size: override the reported ``max_context`` (tokens).  Only meaningful for
                RoPE models, which are not intrinsically limited to their training length.
        """
        p = Path(path)
        if not p.is_dir():
            raise FileNotFoundError(f"{p} is not a directory")
        if _is_r52_checkpoint(p):
            return cls._load_r52(p, max_positions_per_forward, block_size)
        if _is_mlx_lm_dir(p):
            return cls._load_mlx_lm(p, max_positions_per_forward, block_size)
        raise ValueError(
            f"{p} is neither an r52 checkpoint (needs {_META_FILE} + {_MODEL_FILE}) nor an "
            f"mlx-lm model directory (needs {_CONFIG_FILE} + model*.safetensors)"
        )

    @classmethod
    def _load_r52(cls, p: Path, max_positions: int, block_size: int | None) -> LM:
        from ..checkpoint import checkpoint_config, load_checkpoint
        from ..model import GPT

        cfg = checkpoint_config(p)
        model = GPT(cfg.model)
        load_checkpoint(p, model)
        model.eval()
        mx.eval(model.parameters())
        return cls(
            model,
            block_size or cfg.model.block_size,
            vocab_size=cfg.model.vocab_size,
            kind="r52-checkpoint",
            name=_run_name(p),
            path=p,
            config=json.loads((p / _META_FILE).read_text()),
            max_positions_per_forward=max_positions,
        )

    @classmethod
    def _load_mlx_lm(cls, p: Path, max_positions: int, block_size: int | None) -> LM:
        from mlx_lm.utils import load_model

        model, config = load_model(p)
        model.eval()
        mx.eval(model.parameters())
        ctx = (
            block_size
            or config.get("max_position_embeddings")
            or config.get("n_positions")
            or config.get("n_ctx")
        )
        if not ctx:
            raise ValueError(f"{p}/config.json has no max_position_embeddings / n_positions / n_ctx")
        vocab = int(config.get("vocab_size", 50304))
        return cls(
            model,
            int(ctx),
            vocab_size=vocab,
            kind="mlx-lm",
            name=p.name,
            path=p,
            config=config,
            max_positions_per_forward=max_positions,
        )

    # -- description -------------------------------------------------------------------
    def describe(self) -> dict[str, Any]:
        """Small JSON-safe summary of what was loaded (goes into every results file)."""
        out: dict[str, Any] = {
            "name": self.name,
            "kind": self.kind,
            "path": str(self.path) if self.path else None,
            "max_context": self.max_context,
            "vocab_size": self.vocab_size,
            "compute_dtype": str(self.compute_dtype).rsplit(".", 1)[-1],
        }
        if self.kind == "r52-checkpoint":
            out["step"] = self.config.get("step")
            out["tokens"] = self.config.get("tokens")
        else:
            out["model_type"] = self.config.get("model_type")
        return out

    @property
    def compute_dtype(self) -> mx.Dtype:
        """Dtype the matmuls run in (bf16 for mixed-precision GPTs and bf16 exports)."""
        inner = getattr(self.model, "compute_dtype", None)
        if inner is not None:
            return inner
        from mlx.utils import tree_flatten

        for _, v in tree_flatten(self.model.parameters()):
            if isinstance(v, mx.array) and mx.issubdtype(v.dtype, mx.floating):
                return v.dtype
        return mx.float32  # pragma: no cover - a model with no float parameters

    # -- forward -----------------------------------------------------------------------
    def _raw_logits(self, tokens: mx.array) -> mx.array:
        """Model logits for one batch, in the model's compute dtype. No chunking."""
        if tokens.shape[-1] > self.max_context:
            raise ValueError(
                f"sequence of {tokens.shape[-1]} tokens exceeds max_context={self.max_context}; "
                "truncate before calling (keep the last max_context tokens)"
            )
        if hasattr(self.model, "logits"):
            # r52.model.GPT: `logits` is the softcapped head in compute dtype; `__call__`
            # would upcast to fp32 and double the memory of the (B, T, V) tensor.
            return self.model.logits(tokens)
        return self.model(tokens)

    def _row_chunks(self, n_rows: int, seq_len: int) -> int:
        """Rows per forward so that ``rows * seq_len <= max_positions_per_forward``."""
        if seq_len <= 0:
            return n_rows
        return max(1, min(n_rows, self.max_positions_per_forward // seq_len or 1))

    def logits(self, tokens: mx.array) -> mx.array:
        """``(B, T)`` int32 token ids -> ``(B, T, vocab_size)`` **fp32** logits.

        Materialises the whole fp32 tensor, so it is for tests and small batches only
        (``B * T * vocab_size * 4`` bytes).  Evaluations use :meth:`scores`, which never
        holds more than ``fp32_chunk_positions`` positions of fp32 logits at a time.
        """
        tokens = _as_int32(tokens)
        rows, seq_len = tokens.shape
        step = self._row_chunks(rows, seq_len)
        out = []
        for s in range(0, rows, step):
            piece = self._raw_logits(tokens[s : s + step]).astype(mx.float32)
            mx.eval(piece)
            out.append(piece)
        return out[0] if len(out) == 1 else mx.concatenate(out, axis=0)

    def scores(self, tokens: mx.array) -> tuple[mx.array, mx.array]:
        """Autoregressive per-position negative log-likelihood and greedy prediction.

        Args:
            tokens: ``(B, T)`` int32 token ids.

        Returns:
            ``(nll, pred)`` where ``nll[b, t] = -log p(tokens[b, t+1] | tokens[b, :t+1])`` in
            **nats** (fp32) and ``pred[b, t] = argmax_v logits[b, t, v]`` (int32).  The last
            column of ``nll`` is ``nan`` -- there is no autoregressive target there -- which is
            nanochat's convention in ``core_eval.forward_model`` and keeps the two index
            conventions identical to the reference implementations we port from.

        The log-sum-exp is computed in fp32 over at most ``fp32_chunk_positions`` positions at
        a time, so peak memory is bounded independently of ``B * T``.
        """
        tokens = _as_int32(tokens)
        rows, seq_len = tokens.shape
        step = self._row_chunks(rows, seq_len)
        nll_parts, pred_parts = [], []
        for s in range(0, rows, step):
            chunk = tokens[s : s + step]
            logits = self._raw_logits(chunk)
            pred = mx.argmax(logits, axis=-1).astype(mx.int32)
            nll = self._nll_from_logits(logits, chunk)
            mx.eval(nll, pred)
            nll_parts.append(nll)
            pred_parts.append(pred)
        if len(nll_parts) == 1:
            return nll_parts[0], pred_parts[0]
        return mx.concatenate(nll_parts, axis=0), mx.concatenate(pred_parts, axis=0)

    def _nll_from_logits(self, logits: mx.array, tokens: mx.array) -> mx.array:
        """fp32 ``-log p(next token)`` per position; last column ``nan``. Chunked over T."""
        rows, seq_len = tokens.shape
        if seq_len < 2:
            return mx.full((rows, seq_len), float("nan"), dtype=mx.float32)
        targets = tokens[:, 1:]
        span = max(1, self.fp32_chunk_positions // max(1, rows))
        pieces = []
        for s in range(0, seq_len - 1, span):
            e = min(s + span, seq_len - 1)
            sl = logits[:, s:e].astype(mx.float32)
            lse = mx.logsumexp(sl, axis=-1)
            picked = mx.take_along_axis(sl, targets[:, s:e, None], axis=-1).squeeze(-1)
            pieces.append(lse - picked)
        body = pieces[0] if len(pieces) == 1 else mx.concatenate(pieces, axis=1)
        tail = mx.full((rows, 1), float("nan"), dtype=mx.float32)
        return mx.concatenate([body, tail], axis=1)

    def token_nll(self, inputs: mx.array, targets: mx.array) -> mx.array:
        """``(B, T)`` inputs and shifted targets -> ``(B, T)`` fp32 nll in nats.

        This is the validation-loss path.  It is the same arithmetic as
        ``GPT.loss(inputs, targets, fp32_logits=True)`` -- fp32 log-sum-exp over the
        compute-dtype logits -- just kept per-token so the caller can weight it.
        """
        inputs, targets = _as_int32(inputs), _as_int32(targets)
        rows, seq_len = inputs.shape
        step = self._row_chunks(rows, seq_len)
        parts = []
        for s in range(0, rows, step):
            logits = self._raw_logits(inputs[s : s + step])
            tgt = targets[s : s + step]
            span = max(1, self.fp32_chunk_positions // max(1, min(step, rows - s)))
            sub = []
            for a in range(0, seq_len, span):
                b = min(a + span, seq_len)
                sl = logits[:, a:b].astype(mx.float32)
                lse = mx.logsumexp(sl, axis=-1)
                picked = mx.take_along_axis(sl, tgt[:, a:b, None], axis=-1).squeeze(-1)
                sub.append(lse - picked)
            piece = sub[0] if len(sub) == 1 else mx.concatenate(sub, axis=1)
            mx.eval(piece)
            parts.append(piece)
        return parts[0] if len(parts) == 1 else mx.concatenate(parts, axis=0)

    def mean_nll(self, inputs: mx.array, targets: mx.array) -> float:
        """Mean cross-entropy over one batch, in **nats/token**."""
        return float(self.token_nll(inputs, targets).mean())


def _as_int32(tokens: mx.array) -> mx.array:
    if not isinstance(tokens, mx.array):
        tokens = mx.array(tokens)
    return tokens if tokens.dtype == mx.int32 else tokens.astype(mx.int32)


def _run_name(ckpt: Path) -> str:
    """``runs/<run>/ckpt/best`` -> ``<run>``; anything else -> the directory name."""
    parts = ckpt.resolve().parts
    if "ckpt" in parts:
        i = len(parts) - 1 - parts[::-1].index("ckpt")
        if i > 0:
            return parts[i - 1]
    return ckpt.name


def wilson_interval(correct: int, total: int, z: float = 1.959963984540054) -> tuple[float, float]:
    """95 % Wilson score interval for a binomial proportion (fractions, not percent).

    Wilson rather than the normal approximation because CORE tasks include 32-example sets
    where the normal interval leaves the unit interval.  ``z`` defaults to the two-sided 95 %
    normal quantile.
    """
    if total <= 0:
        return (float("nan"), float("nan"))
    p = correct / total
    denom = 1.0 + z * z / total
    centre = (p + z * z / (2 * total)) / denom
    half = z * math.sqrt(p * (1.0 - p) / total + z * z / (4 * total * total)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))
