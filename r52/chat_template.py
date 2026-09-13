# Copyright 2026 The road-to-52 authors.
# SPDX-License-Identifier: Apache-2.0
#
# The conversation format (a <|bos|> prefix, then `<|role_start|> ... <|role_end|>` spans,
# with the loss mask covering the assistant span only) is ported from
# karpathy/nanochat's `nanochat/tokenizer.py` (MIT).  No code is copied verbatim: nanochat
# trains its own 65,536-entry BPE with the special tokens inside the vocabulary, whereas we
# reuse OpenAI's GPT-2 BPE and place the special tokens in the **spare ids of the padded
# vocabulary** (50257..50303 are unused by GPT-2 but addressable by the model, whose
# `vocab_size` is 50304).  So no embedding is resized and a base checkpoint can be
# midtrained / SFT-ed into a chat model without surgery.
"""Chat special tokens, conversation rendering, and the exported Jinja chat template.

Token ids
---------
GPT-2's BPE ends at id 50256 (``<|endoftext|>``); ``ModelConfig.vocab_size`` is 50304 (the
next multiple of 128), so ids **50257..50303** are reachable by the model but never emitted
by the tokenizer.  We claim the first eight:

======  =====================  =========================================================
id      token                  role
======  =====================  =========================================================
50257   ``<|bos|>``            start of every conversation (and of a midtrain doc)
50258   ``<|user_start|>``     opens a user turn
50259   ``<|user_end|>``       closes a user turn
50260   ``<|assistant_start|>``opens an assistant turn (the generation prompt)
50261   ``<|assistant_end|>``  closes an assistant turn; the chat model's EOS
50262   ``<|system_start|>``   opens a system turn
50263   ``<|system_end|>``     closes a system turn
50264   ``<|pad|>``            padding; never supervised (its loss mask is always 0)
======  =====================  =========================================================

Loss mask
---------
:func:`render` returns ``(ids, mask)`` with ``len(mask) == len(ids)``.  ``mask[i] == 1``
means "``ids[i]`` is a **target** the model is trained to predict".  Only assistant content
and the closing ``<|assistant_end|>`` are supervised; ``<|assistant_start|>`` is a cue the
model is *given*, so it is masked out, as are all system/user tokens and any padding.

For next-token training with ``x = ids[:-1]`` and ``y = ids[1:]`` the loss mask is
``mask[1:]``; :func:`to_xy` does that shift and encodes "ignore" as ``-1``, which is exactly
what :meth:`r52.model.GPT.loss` treats as ``ignore_index``.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path

__all__ = [
    "ASSISTANT_END",
    "ASSISTANT_START",
    "BOS",
    "CHAT_TEMPLATE",
    "EOT",
    "FIRST_SPECIAL_ID",
    "PAD",
    "SPECIAL_IDS",
    "SPECIAL_TOKENS",
    "SYSTEM_END",
    "SYSTEM_START",
    "USER_END",
    "USER_START",
    "added_tokens_decoder",
    "install_chat_tokenizer",
    "render",
    "render_prompt",
    "strip_specials",
    "to_xy",
]

FIRST_SPECIAL_ID = 50257
"""First spare id in the padded GPT-2 vocabulary (50257..50303 are unused by GPT-2)."""

SPECIAL_TOKENS: tuple[str, ...] = (
    "<|bos|>",
    "<|user_start|>",
    "<|user_end|>",
    "<|assistant_start|>",
    "<|assistant_end|>",
    "<|system_start|>",
    "<|system_end|>",
    "<|pad|>",
)
"""In id order, starting at :data:`FIRST_SPECIAL_ID`. **Never reorder**: the ids are baked
into every checkpoint trained with them."""

SPECIAL_IDS: dict[str, int] = {t: FIRST_SPECIAL_ID + i for i, t in enumerate(SPECIAL_TOKENS)}

BOS = SPECIAL_IDS["<|bos|>"]
USER_START = SPECIAL_IDS["<|user_start|>"]
USER_END = SPECIAL_IDS["<|user_end|>"]
ASSISTANT_START = SPECIAL_IDS["<|assistant_start|>"]
ASSISTANT_END = SPECIAL_IDS["<|assistant_end|>"]
SYSTEM_START = SPECIAL_IDS["<|system_start|>"]
SYSTEM_END = SPECIAL_IDS["<|system_end|>"]
PAD = SPECIAL_IDS["<|pad|>"]

EOT = 50256
"""GPT-2's ``<|endoftext|>``; still the document separator for plain pretraining text."""

_ROLE_SPANS: dict[str, tuple[int, int]] = {
    "system": (SYSTEM_START, SYSTEM_END),
    "user": (USER_START, USER_END),
    "assistant": (ASSISTANT_START, ASSISTANT_END),
}

IGNORE_INDEX = -1
"""Target value that :meth:`r52.model.GPT.loss` skips."""


# --------------------------------------------------------------------------------------
# Rendering
# --------------------------------------------------------------------------------------


def _encoder(tokenizer=None):
    """Return a ``text -> list[int]`` callable that never emits a special id."""
    if tokenizer is None:
        from .tokenizer import GPT2Tokenizer

        tokenizer = GPT2Tokenizer()
    return tokenizer.encode_ordinary


def render(
    messages: Sequence[Mapping[str, str]],
    tokenizer=None,
    add_generation_prompt: bool = False,
    max_tokens: int | None = None,
) -> tuple[list[int], list[int]]:
    """Render a conversation to ``(ids, mask)``.

    Parameters
    ----------
    messages
        ``[{"role": "system"|"user"|"assistant", "content": str}, ...]``.
    tokenizer
        A :class:`r52.tokenizer.GPT2Tokenizer` (built on demand when ``None``).
    add_generation_prompt
        Append a trailing ``<|assistant_start|>`` so the model continues as the assistant.
        Used for inference and for RL/pass@k prompts.
    max_tokens
        Truncate the rendered conversation to this many **tokens** (``None`` = no limit).
        Truncation is a hard cut at the end, so a truncated example may lose its closing
        ``<|assistant_end|>``; :func:`r52.posttrain.data.iter_rendered` drops such examples.

    Returns
    -------
    ``(ids, mask)``
        Two equal-length lists; ``mask[i] == 1`` marks ``ids[i]`` as a supervised target.
    """
    enc = _encoder(tokenizer)
    ids: list[int] = [BOS]
    mask: list[int] = [0]
    for msg in messages:
        role = msg["role"]
        if role == "tool":  # smol-smoltalk has none, but be explicit rather than silent
            raise ValueError("tool turns are not part of the r52 chat format")
        try:
            start, end = _ROLE_SPANS[role]
        except KeyError as exc:  # pragma: no cover - guarded by the dataset preparation
            raise ValueError(f"unknown chat role {role!r}") from exc
        body = enc(msg["content"])
        supervised = int(role == "assistant")
        ids.append(start)
        mask.append(0)  # the opening cue is given to the model, never predicted by it
        ids.extend(body)
        mask.extend([supervised] * len(body))
        ids.append(end)
        mask.append(supervised)  # the model must learn to stop
    if add_generation_prompt:
        ids.append(ASSISTANT_START)
        mask.append(0)
    if max_tokens is not None and len(ids) > max_tokens:
        ids, mask = ids[:max_tokens], mask[:max_tokens]
    return ids, mask


def render_prompt(
    messages: Sequence[Mapping[str, str]], tokenizer=None, max_tokens: int | None = None
) -> list[int]:
    """Render a conversation as a *prompt* (trailing ``<|assistant_start|>``, no mask)."""
    ids, _ = render(messages, tokenizer, add_generation_prompt=True, max_tokens=max_tokens)
    return ids


def to_xy(ids: Sequence[int], mask: Sequence[int]) -> tuple[list[int], list[int]]:
    """Shift ``(ids, mask)`` into ``(x, y)`` for next-token training.

    ``x = ids[:-1]``; ``y[i] = ids[i + 1]`` where ``mask[i + 1]`` is set, else
    :data:`IGNORE_INDEX`.
    """
    x = list(ids[:-1])
    y = [int(t) if m else IGNORE_INDEX for t, m in zip(ids[1:], mask[1:], strict=True)]
    return x, y


def strip_specials(text: str) -> str:
    """Remove any chat special token that leaked into a decoded string."""
    for tok in SPECIAL_TOKENS:
        text = text.replace(tok, "")
    return text


# --------------------------------------------------------------------------------------
# The exported Jinja template
# --------------------------------------------------------------------------------------

CHAT_TEMPLATE = (
    "{{ '<|bos|>' }}"
    "{% for message in messages %}"
    "{% if message['role'] == 'system' %}"
    "{{ '<|system_start|>' + message['content'] + '<|system_end|>' }}"
    "{% elif message['role'] == 'user' %}"
    "{{ '<|user_start|>' + message['content'] + '<|user_end|>' }}"
    "{% elif message['role'] == 'assistant' %}"
    "{{ '<|assistant_start|>' + message['content'] + '<|assistant_end|>' }}"
    "{% else %}"
    "{{ raise_exception('unknown role: ' + message['role']) }}"
    "{% endif %}"
    "{% endfor %}"
    "{% if add_generation_prompt %}{{ '<|assistant_start|>' }}{% endif %}"
)
"""Jinja chat template written into the exported ``tokenizer_config.json``.

Must produce byte-identical output to :func:`render` so that ``mlx_lm.generate
--apply-chat-template``, ``mlx_lm.chat`` and ``mlx_lm.server`` see exactly the token
sequence the model was trained on.  ``tests/test_posttrain_chat.py`` asserts the equality
against the real Hugging Face tokenizer.
"""


def added_tokens_decoder() -> dict[str, dict[str, object]]:
    """``added_tokens_decoder`` block for ``tokenizer_config.json`` (id -> descriptor)."""
    return {
        str(SPECIAL_IDS[t]): {
            "content": t,
            "lstrip": False,
            "normalized": False,
            "rstrip": False,
            "single_word": False,
            "special": True,
        }
        for t in SPECIAL_TOKENS
    }


def install_chat_tokenizer(model_dir: str | Path, model_max_length: int = 1024) -> bool:
    """Add the chat special tokens and template to an exported mlx-lm model directory.

    Patches ``tokenizer.json`` in place (appending the eight tokens at ids
    :data:`FIRST_SPECIAL_ID`..\\ +7) and rewrites ``tokenizer_config.json`` with
    ``chat_template``, ``added_tokens_decoder``, ``bos_token``, ``eos_token`` and
    ``pad_token``.

    Returns ``False`` (and changes nothing) when ``tokenizer.json`` is absent -- the export
    then has no tokenizer to patch, e.g. because the machine was offline.

    Raises
    ------
    RuntimeError
        If the base vocabulary is not GPT-2's 50257 entries, because the special tokens
        would land on different ids than the ones baked into the checkpoints.
    """
    d = Path(model_dir)
    tok_json = d / "tokenizer.json"
    if not tok_json.exists():
        return False

    from tokenizers import AddedToken, Tokenizer

    tok = Tokenizer.from_file(str(tok_json))
    have = tok.get_vocab(with_added_tokens=True)
    missing = [t for t in SPECIAL_TOKENS if t not in have]
    if missing:
        base = tok.get_vocab_size(with_added_tokens=True)
        if base != FIRST_SPECIAL_ID:
            raise RuntimeError(
                f"{tok_json} has {base} tokens; the r52 chat tokens assume GPT-2's "
                f"{FIRST_SPECIAL_ID}. Re-export the tokenizer before installing the chat format."
            )
        tok.add_special_tokens(
            [
                AddedToken(t, single_word=False, lstrip=False, rstrip=False, normalized=False, special=True)
                for t in SPECIAL_TOKENS
            ]
        )
        tok.save(str(tok_json))
        have = tok.get_vocab(with_added_tokens=True)

    wrong = {t: have[t] for t in SPECIAL_TOKENS if have[t] != SPECIAL_IDS[t]}
    if wrong:
        raise RuntimeError(f"chat tokens landed on the wrong ids in {tok_json}: {wrong}")

    cfg_path = d / "tokenizer_config.json"
    cfg: dict[str, object] = {}
    if cfg_path.exists():
        try:
            cfg = json.loads(cfg_path.read_text())
        except ValueError:
            cfg = {}
    cfg.update(
        {
            "added_tokens_decoder": added_tokens_decoder(),
            "bos_token": "<|bos|>",
            "eos_token": "<|assistant_end|>",
            "pad_token": "<|pad|>",
            "unk_token": None,
            "clean_up_tokenization_spaces": False,
            "model_max_length": int(model_max_length),
            "chat_template": CHAT_TEMPLATE,
        }
    )
    cfg_path.write_text(json.dumps(cfg, indent=2))
    return True


def eos_token_ids() -> list[int]:
    """Ids that terminate generation: the chat EOS first, then GPT-2's ``<|endoftext|>``."""
    return [ASSISTANT_END, EOT]


def iter_special_ids() -> Iterable[int]:
    """All ids claimed by the chat format (useful for masking them out of samplers)."""
    return SPECIAL_IDS.values()
