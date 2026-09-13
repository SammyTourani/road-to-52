# Copyright 2026 The road-to-52 authors.
# SPDX-License-Identifier: Apache-2.0
#
# The KV-cached decode loop follows the structure of `mlx_lm/generate.py::generate_step`
# (MIT, Apple Inc.); it is re-implemented rather than called because GRPO and pass@k need
# the sampled *token ids and lengths* of every sequence in a group, which
# `mlx_lm.generate.batch_generate` only hands back as text.
"""Group-batched, KV-cached sampling from an exported r52 model (the mlx-lm plugin class).

Why the *exported* model and not :class:`r52.model.GPT`?  ``GPT`` is a training model with
no KV cache -- sampling 64 tokens from it costs 64 full-context forwards per sequence.
``r52/mlx_plugin/r52gpt.py`` is the same architecture *with* a cache, and it is what
``mlx_lm.generate`` / ``mlx_lm.server`` already run, so RL and pass@k sample exactly the
artifact a user downloads.

**Prefill once per prompt, decode the group in one batch.**  Both callers sample ``k``
completions of the *same* prompt, so the prompt is prefilled a single time into a
``(1, n_kv, L, head_dim)`` KV cache and that cache is then repeated ``k`` times along the
batch axis.  Decoding is one ``(k, 1)`` forward per token.  This avoids left-padding
entirely -- no padded prefix ever enters attention, so the arithmetic is exact rather than
approximately right -- and costs ``1 + max_tokens`` forwards per group instead of
``k * (1 + max_tokens)``.

Units: ``max_tokens`` is tokens; every returned length is a token count.
"""

from __future__ import annotations

from dataclasses import dataclass

import mlx.core as mx
import numpy as np

from ..chat_template import ASSISTANT_END, EOT

__all__ = ["Completion", "Sampler", "load_exported_model"]


def load_exported_model(path: str):
    """Load an exported mlx-lm model directory; returns ``(model, tokenizer)``."""
    from mlx_lm.utils import load

    return load(path)


@dataclass
class Completion:
    """One sampled continuation."""

    prompt_index: int
    tokens: list[int]
    """Sampled token ids, excluding the prompt and the stop token."""
    text: str
    finished: bool
    """True if generation stopped on an EOS rather than the length cap."""

    def __len__(self) -> int:
        return len(self.tokens)


def sample_logits(logits: mx.array, temperature: float, top_p: float, top_k: int) -> mx.array:
    """Sample one token per row of ``logits`` ``(B, V)``. ``temperature <= 0`` is greedy."""
    if temperature <= 0:
        return mx.argmax(logits, axis=-1)
    logits = logits.astype(mx.float32) / temperature
    if top_k and 0 < top_k < logits.shape[-1]:
        kth = mx.topk(logits, top_k, axis=-1)[:, -1:]
        logits = mx.where(logits < kth, mx.array(-float("inf"), dtype=logits.dtype), logits)
    if 0.0 < top_p < 1.0:
        order = mx.argsort(-logits, axis=-1)
        sorted_logits = mx.take_along_axis(logits, order, axis=-1)
        probs = mx.softmax(sorted_logits, axis=-1)
        # Keep the smallest prefix whose mass reaches top_p; the first token always stays.
        keep = (mx.cumsum(probs, axis=-1) - probs) < top_p
        sorted_logits = mx.where(keep, sorted_logits, mx.array(-float("inf"), dtype=logits.dtype))
        rank = mx.broadcast_to(mx.arange(order.shape[-1])[None, :], order.shape)
        inverse = mx.put_along_axis(mx.zeros_like(order), order, rank, axis=-1)
        logits = mx.take_along_axis(sorted_logits, inverse, axis=-1)
    return mx.random.categorical(logits, axis=-1)


class Sampler:
    """Batched sampler over a loaded mlx-lm model.

    Parameters
    ----------
    model, tokenizer
        As returned by :func:`load_exported_model`.
    eos_ids
        Ids that end a completion (default: the chat EOS and ``<|endoftext|>``).
    """

    def __init__(self, model, tokenizer, eos_ids: list[int] | None = None) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.eos_ids = list(eos_ids) if eos_ids else [ASSISTANT_END, EOT]

    # -- helpers -------------------------------------------------------------------
    def decode(self, ids: list[int]) -> str:
        """Token ids -> text, tolerant of the garbage a tiny model emits."""
        if not ids:
            return ""
        try:
            return self.tokenizer.decode(ids)
        except Exception:  # pragma: no cover - detokenizer edge cases on random ids
            return ""

    def _group_cache(self, prompt: list[int], k: int):
        """Prefill ``prompt`` once and repeat the cache ``k`` times along the batch axis.

        Returns ``(cache, logits)`` where ``logits`` is ``(k, vocab)`` for the next token.
        """
        from mlx_lm.models.cache import KVCache, make_prompt_cache

        cache = make_prompt_cache(self.model)
        logits = self.model(mx.array([prompt], dtype=mx.int32), cache=cache)
        logits = logits[:, -1, :]
        mx.eval(logits, *[c.state for c in cache])
        if k == 1:
            return cache, logits
        wide = []
        for c in cache:
            keys, values = c.state
            nc = KVCache()
            nc.state = (mx.repeat(keys, k, axis=0), mx.repeat(values, k, axis=0))
            wide.append(nc)
        logits = mx.repeat(logits, k, axis=0)
        mx.eval(logits, *[c.state for c in wide])
        return wide, logits

    # -- public ---------------------------------------------------------------------
    def sample_group(
        self,
        prompt: list[int],
        k: int = 1,
        max_tokens: int = 64,
        temperature: float = 1.0,
        top_p: float = 1.0,
        top_k: int = 0,
        prompt_index: int = 0,
    ) -> list[Completion]:
        """Sample ``k`` completions of one prompt with a shared prefill."""
        cache, logits = self._group_cache(list(prompt), k)
        tokens: list[list[int]] = [[] for _ in range(k)]
        done = [False] * k
        eos = set(self.eos_ids)

        for _ in range(max_tokens):
            nxt = sample_logits(logits, temperature, top_p, top_k)
            mx.eval(nxt)
            ids = [int(t) for t in np.asarray(nxt)]
            for i, t in enumerate(ids):
                if done[i]:
                    continue
                if t in eos:
                    done[i] = True
                else:
                    tokens[i].append(t)
            if all(done):
                break
            step = mx.array([[t] for t in ids], dtype=mx.int32)
            logits = self.model(step, cache=cache)[:, -1, :]
            mx.eval(logits)
        del cache
        mx.clear_cache()
        return [Completion(prompt_index, tokens[i], self.decode(tokens[i]), done[i]) for i in range(k)]

    def sample(
        self,
        prompts: list[list[int]],
        k: int = 1,
        max_tokens: int = 64,
        temperature: float = 1.0,
        top_p: float = 1.0,
        top_k: int = 0,
    ) -> list[list[Completion]]:
        """``k`` completions for each prompt; returns one list of ``k`` per prompt."""
        return [
            self.sample_group(p, k, max_tokens, temperature, top_p, top_k, prompt_index=i)
            for i, p in enumerate(prompts)
        ]
