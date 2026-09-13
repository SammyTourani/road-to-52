# Copyright 2026 The road-to-52 authors.
# SPDX-License-Identifier: Apache-2.0
#
# The recipe -- byte-level BPE, a GPT-4-style split pattern with the digit run narrowed to
# `\p{N}{1,2}`, and a 32,768-entry vocabulary -- is taken from karpathy/nanochat's
# `scripts/tok_train.py` and `rustbpe` (MIT License, Copyright (c) 2025 Andrej Karpathy).
# No code is copied: nanochat trains with `rustbpe` and serves with `tiktoken`, while this
# module trains and serves with Hugging Face `tokenizers` (Apache-2.0), which is the stack
# `docs/PLAN.md` §5 selects and the only one that can write the `tokenizer.json` mlx-lm
# loads.  nanochat's stated reason for `{1,2}` rather than GPT-4's `{1,3}` -- "I didn't want
# to waste too many tokens on numbers for smaller vocab sizes" -- is quoted in
# `research/02-open-training-stack.md` §Stage 3, which also records 32,768 as its vocab size.
"""Train, save, load and use our own byte-level BPE tokenizer.

    python -m r52.tokenizer_train --name fweb32k --corpus fineweb-edu --bytes 50000000
    python -m r52.tokenizer_train --name tiny --text-file notes.txt --vocab-size 4096
    python -m r52.tokenizer_train --name fweb32k --from-bin data/corpora/fineweb-edu-gpt2 \
        --bytes 50000000 --compare-gpt2

Why a second tokenizer at all
-----------------------------
``docs/ABLATIONS.md`` Axis 3 compares GPT-2's 50,257-entry BPE against a 32,768-entry BPE
trained on our own data **at fixed bytes of training text**, scored on bits-per-byte and
HellaSwag because those two are tokenizer-neutral and validation loss is not.  A smaller,
better-fitted vocabulary buys two things at once: fewer tokens for the same text (more bytes
learned per step) and a much cheaper ``lm_head`` -- at ``d=384`` the GPT-2 head is 19.3M
parameters and the 32,768 head is 12.6M.

Vocabulary layout
-----------------
The trained BPE occupies ids ``0 .. vocab_size - 10``; the last nine ids are, in order,
``<|endoftext|>`` followed by the eight chat special tokens of :mod:`r52.chat_template`, in
**the same relative order** they have in the GPT-2 layout::

    <|endoftext|>  <|bos|>  <|user_start|>  <|user_end|>  <|assistant_start|>
    <|assistant_end|>  <|system_start|>  <|system_end|>  <|pad|>

So ``eot = vocab_size - 9`` and the chat ids start at ``vocab_size - 8``.  With the default
32,768 that is ``eot = 32759`` and ``<|bos|> = 32760``.  ``vocab_size`` is also the model's
``vocab_size``: 32,768 is already a multiple of 128, so unlike the GPT-2 path there is no
padding gap and no spare-id trick -- the specials are inside the vocabulary, as they are in
nanochat.  A vocabulary that the trainer could not fill (a small training text) is padded
with reserved ``<|unused_N|>`` ids so the special ids land on the same numbers regardless of
how much text was available; that keeps a checkpoint's ids a function of the config alone.

Interface
---------
:class:`R52Tokenizer` exposes exactly the surface :class:`r52.tokenizer.GPT2Tokenizer` does
(``encode`` / ``encode_ordinary`` / ``decode`` / ``decode_bytes`` / ``n_bytes`` / ``eot`` /
``n_vocab`` / ``padded_vocab``), so :mod:`r52.chat_template`, :mod:`r52.eval.hellaswag` and
:mod:`r52.eval.core` work with either one.  :func:`load_tokenizer` is the dispatcher behind
the ``data.tokenizer: gpt2 | <dir>`` config field.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from collections.abc import Iterable, Iterator
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .chat_template import CHAT_TEMPLATE
from .chat_template import SPECIAL_TOKENS as CHAT_SPECIAL_TOKENS
from .tokenizer import bits_per_byte

__all__ = [
    "DEFAULT_TOKENIZER_ROOT",
    "DEFAULT_VOCAB_SIZE",
    "SPECIAL_TOKENS",
    "SPLIT_PATTERN",
    "R52Tokenizer",
    "Tokenizer",
    "bits_per_byte",
    "build_tokenizer",
    "load_tokenizer",
    "main",
    "special_ids",
    "train_tokenizer",
]

DEFAULT_VOCAB_SIZE = 32_768
"""2**15, nanochat's default (research/02 §Stage 3: "small models get small vocabs")."""

DEFAULT_TOKENIZER_ROOT = "data/tokenizers"

EOT_TOKEN = "<|endoftext|>"

SPECIAL_TOKENS: tuple[str, ...] = (EOT_TOKEN, *CHAT_SPECIAL_TOKENS)
"""Nine reserved strings, in id order, occupying the **top** of the vocabulary.

``<|endoftext|>`` is the document separator every ``.bin`` shard uses; the eight that follow
are :data:`r52.chat_template.SPECIAL_TOKENS`, unchanged and unreordered.
"""

SPLIT_PATTERN = (
    r"'(?i:[sdmt]|ll|ve|re)"
    r"|[^\r\n\p{L}\p{N}]?\p{L}+"
    r"|\p{N}{1,2}"
    r"| ?[^\s\p{L}\p{N}]+[\r\n]*"
    r"|\s+$"
    r"|\s*[\r\n]"
    r"|\s+(?!\S)"
    r"|\s"
)
"""nanochat's GPT-4-derived split pattern, with the possessive quantifiers removed.

nanochat writes the pattern with possessive quantifiers (``\\p{N}{1,2}+``, ``\\p{L}++``,
``[\\r\\n]*+``, ``\\s++$``) because ``rustbpe`` compiles it with the Rust ``fancy-regex``
crate, which implements them.  Hugging Face ``tokenizers`` 0.23.2 **accepts the same string
and then means something different by it**: measured on 2026-09-13, its engine reads
``\\p{N}{1,2}+`` as ``(\\p{N}{1,2})+`` and therefore emits ``1234`` as a single
pre-token instead of ``12`` + ``34`` -- exactly the digit split the ``{1,2}`` is there to
produce.  Dropping the possessive markers restores nanochat's intended behaviour on this
engine; see ``docs/DEVIATIONS.md`` (ablation builder, A1).
"""


def special_ids(vocab_size: int = DEFAULT_VOCAB_SIZE) -> dict[str, int]:
    """``{token: id}`` for the nine specials in a vocabulary of ``vocab_size``."""
    base = int(vocab_size) - len(SPECIAL_TOKENS)
    if base < 256:
        raise ValueError(f"vocab_size={vocab_size} leaves no room for 256 bytes + 9 specials")
    return {t: base + i for i, t in enumerate(SPECIAL_TOKENS)}


# --------------------------------------------------------------------------------------
# Training
# --------------------------------------------------------------------------------------


def build_tokenizer():
    """An untrained byte-level BPE ``tokenizers.Tokenizer`` with our pre-tokenizer."""
    from tokenizers import Regex, decoders, models, pre_tokenizers
    from tokenizers import Tokenizer as HFTokenizer

    tok = HFTokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.Sequence(
        [
            pre_tokenizers.Split(Regex(SPLIT_PATTERN), behavior="isolated", invert=False),
            # `use_regex=False`: the split above already did the segmentation; ByteLevel is
            # here only to map raw bytes onto a 256-symbol printable alphabet, which is what
            # makes the vocabulary lossless for arbitrary UTF-8 (and for invalid UTF-8).
            pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False),
        ]
    )
    tok.decoder = decoders.ByteLevel()
    return tok


def _batched(texts: Iterable[str], size: int = 1000) -> Iterator[list[str]]:
    batch: list[str] = []
    for t in texts:
        batch.append(t)
        if len(batch) >= size:
            yield batch
            batch = []
    if batch:
        yield batch


def train_tokenizer(
    texts: Iterable[str],
    vocab_size: int = DEFAULT_VOCAB_SIZE,
    *,
    min_frequency: int = 2,
    show_progress: bool = False,
):
    """Train a byte-level BPE on ``texts`` and return the ``tokenizers.Tokenizer``.

    The BPE learns ``vocab_size - 9`` entries; the nine specials are appended afterwards, and
    any shortfall (a training text too small to fill the vocabulary) is padded with reserved
    ``<|unused_N|>`` ids so the specials always land on :func:`special_ids`.
    """
    from tokenizers import AddedToken, pre_tokenizers, trainers

    ids = special_ids(vocab_size)
    n_base = int(vocab_size) - len(SPECIAL_TOKENS)

    tok = build_tokenizer()
    trainer = trainers.BpeTrainer(
        vocab_size=n_base,
        min_frequency=min_frequency,
        show_progress=show_progress,
        special_tokens=[],
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
    )
    tok.train_from_iterator(_batched(texts), trainer=trainer)

    trained = tok.get_vocab_size(with_added_tokens=False)
    if trained > n_base:  # pragma: no cover - the trainer honours its cap
        raise RuntimeError(f"trainer produced {trained} tokens, asked for at most {n_base}")

    def _add(names: list[str]) -> None:
        tok.add_special_tokens(
            [
                AddedToken(n, single_word=False, lstrip=False, rstrip=False, normalized=False, special=True)
                for n in names
            ]
        )

    if trained < n_base:
        _add([f"<|unused_{i}|>" for i in range(n_base - trained)])
    _add(list(SPECIAL_TOKENS))

    vocab = tok.get_vocab(with_added_tokens=True)
    wrong = {t: vocab.get(t) for t, i in ids.items() if vocab.get(t) != i}
    if wrong:  # pragma: no cover - guarded by the padding above
        raise RuntimeError(f"special tokens landed on the wrong ids: {wrong} (expected {ids})")
    if tok.get_vocab_size(with_added_tokens=True) != int(vocab_size):
        raise RuntimeError(  # pragma: no cover
            f"vocabulary is {tok.get_vocab_size(with_added_tokens=True)}, expected {vocab_size}"
        )
    return tok


def save_tokenizer(
    tok,
    out_dir: str | Path,
    *,
    name: str,
    vocab_size: int,
    meta: dict[str, Any] | None = None,
    model_max_length: int = 1024,
) -> Path:
    """Write ``tokenizer.json``, ``tokenizer_config.json`` and ``meta.json`` into ``out_dir``.

    ``tokenizer.json`` is the Hugging Face format, which is what ``mlx_lm`` loads, so
    :mod:`r52.export` only has to copy these files next to the weights.
    """
    d = Path(out_dir)
    d.mkdir(parents=True, exist_ok=True)
    tok.save(str(d / "tokenizer.json"))

    ids = special_ids(vocab_size)
    (d / "tokenizer_config.json").write_text(
        json.dumps(
            {
                "added_tokens_decoder": {
                    str(i): {
                        "content": t,
                        "lstrip": False,
                        "normalized": False,
                        "rstrip": False,
                        "single_word": False,
                        "special": True,
                    }
                    for t, i in ids.items()
                },
                "bos_token": "<|bos|>",
                "eos_token": "<|assistant_end|>",
                "pad_token": "<|pad|>",
                "unk_token": None,
                "clean_up_tokenization_spaces": False,
                "model_max_length": int(model_max_length),
                "tokenizer_class": "PreTrainedTokenizerFast",
                "chat_template": CHAT_TEMPLATE,
            },
            indent=2,
        )
    )

    body: dict[str, Any] = {
        "name": name,
        "kind": "r52-bpe",
        "vocab_size": int(vocab_size),
        "padded_vocab_size": int(vocab_size),
        "n_base_tokens": int(vocab_size) - len(SPECIAL_TOKENS),
        "special_tokens": ids,
        "eot_id": ids[EOT_TOKEN],
        "first_chat_special_id": ids["<|bos|>"],
        "split_pattern": SPLIT_PATTERN,
        "trainer": "huggingface/tokenizers BpeTrainer (byte-level)",
        "recipe_source": "karpathy/nanochat scripts/tok_train.py + rustbpe (MIT)",
        "created": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    body.update(meta or {})
    (d / "meta.json").write_text(json.dumps(body, indent=2))
    return d


# --------------------------------------------------------------------------------------
# The tokenizer object
# --------------------------------------------------------------------------------------


def _without_added_tokens(tok):
    """A clone of ``tok`` with an empty ``added_tokens`` list (see :meth:`encode_ordinary`)."""
    from tokenizers import Tokenizer as HFTokenizer

    body = json.loads(tok.to_str())
    body["added_tokens"] = []
    return HFTokenizer.from_str(json.dumps(body))


def _byte_table(tok, n_vocab: int) -> list[bytes]:
    """``id -> exact bytes`` for every id, inverting the byte-level alphabet.

    ByteLevel maps each of the 256 byte values onto one printable Unicode character; the
    vocabulary is spelled in those characters, so a token's true bytes are recovered by
    mapping its spelling back one character at a time.  ``alphabet()`` returns those 256
    characters *unordered*, so the GPT-2 bytes-to-unicode table is rebuilt here rather than
    guessed.  Specials map to their literal UTF-8 spelling instead, which is what makes a
    chat-rendered corpus byte-countable.
    """
    bs = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("\xa1"), ord("\xac") + 1))
        + list(range(ord("\xae"), ord("\xff") + 1))
    )
    cs = list(bs)
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    char_to_byte = {chr(c): b for b, c in zip(bs, cs, strict=True)}

    specials = set(SPECIAL_TOKENS)
    table: list[bytes] = [b""] * n_vocab
    for piece, idx in tok.get_vocab(with_added_tokens=True).items():
        if idx >= n_vocab:  # pragma: no cover - guarded by the caller
            continue
        if piece in specials or piece.startswith("<|unused_"):
            table[idx] = piece.encode("utf-8")
        else:
            table[idx] = bytes(char_to_byte[c] for c in piece)
    return table


class R52Tokenizer:
    """A trained byte-level BPE with :class:`r52.tokenizer.GPT2Tokenizer`'s interface."""

    kind = "r52-bpe"

    def __init__(self, tok, meta: dict[str, Any] | None = None, path: Path | None = None) -> None:
        self.tok = tok
        self.meta = dict(meta or {})
        self.path = path
        self.n_vocab = int(tok.get_vocab_size(with_added_tokens=True))
        self.padded_vocab = self.n_vocab
        vocab = tok.get_vocab(with_added_tokens=True)
        missing = [t for t in SPECIAL_TOKENS if t not in vocab]
        if missing:
            raise ValueError(f"{path or '<tokenizer>'}: missing special tokens {missing}")
        self.special_ids: dict[str, int] = {t: int(vocab[t]) for t in SPECIAL_TOKENS}
        self.eot = self.special_ids[EOT_TOKEN]
        self._id_to_special = {i: t for t, i in self.special_ids.items()}
        self._ordinary = _without_added_tokens(tok)
        self._id_bytes = _byte_table(tok, self.n_vocab)

    # -- construction --------------------------------------------------------------
    @classmethod
    def from_dir(cls, path: str | Path) -> R52Tokenizer:
        """Load a directory written by :func:`save_tokenizer`."""
        from tokenizers import Tokenizer as HFTokenizer

        d = Path(path)
        f = d / "tokenizer.json" if d.is_dir() else d
        if not f.is_file():
            raise FileNotFoundError(
                f"{f} not found. Train one: python -m r52.tokenizer_train --name "
                f"{d.name} --corpus fineweb-edu --bytes 50000000"
            )
        meta_path = f.with_name("meta.json")
        meta = json.loads(meta_path.read_text()) if meta_path.is_file() else {}
        return cls(HFTokenizer.from_file(str(f)), meta, f.parent)

    @property
    def name(self) -> str:
        return str(self.meta.get("name") or (self.path.name if self.path else "r52-bpe"))

    # -- encode / decode -----------------------------------------------------------
    def encode(self, text: str, allowed_special: str | set[str] = "all") -> list[int]:
        """Text -> token ids.

        ``allowed_special="all"`` lets a special token's literal spelling in ``text`` become
        its id (tiktoken's ``allowed_special`` semantics); anything else -- including the
        empty set the evals pass -- forces the BPE spelling, which is what stops a user turn
        from forging a turn boundary.
        """
        if allowed_special == "all":
            return self.tok.encode(text, add_special_tokens=True).ids
        allowed = set(allowed_special or ())
        if not allowed:
            return self.encode_ordinary(text)
        # Partial permission: split on the allowed spellings, encode the gaps ordinarily.
        out: list[int] = []
        for piece, tok_name in _split_on(text, sorted(allowed, key=len, reverse=True)):
            if tok_name is not None:
                out.append(self.special_ids[tok_name])
            elif piece:
                out.extend(self.encode_ordinary(piece))
        return out

    def encode_ordinary(self, text: str) -> list[int]:
        """Text -> token ids with **no** special id ever produced.

        Hugging Face ``tokenizers`` matches an added token's spelling in the input *whatever*
        ``add_special_tokens`` says (measured: ``encode("a <|endoftext|> b",
        add_special_tokens=False)`` still returns the special id), so "ordinary" encoding runs
        against a clone of the tokenizer that has no added tokens at all.  Without this a user
        who types ``<|assistant_end|>`` could end their own turn.
        """
        return self._ordinary.encode(text, add_special_tokens=False).ids

    def decode(self, tokens) -> str:
        """Token ids -> text (invalid UTF-8 replaced, never raised)."""
        return self.decode_bytes(tokens).decode("utf-8", errors="replace")

    def decode_bytes(self, tokens) -> bytes:
        """Token ids -> raw UTF-8 bytes (exact; used for byte counting).

        Byte-exact rather than ``decode().encode()``: the ByteLevel decoder goes through
        ``String::from_utf8_lossy``, so a token run that starts or ends mid-character would
        gain replacement bytes and corrupt a bits-per-byte measurement.  Specials decode to
        their literal spelling, exactly as :meth:`r52.tokenizer.GPT2Tokenizer.decode_bytes`.
        """
        table = self._id_bytes
        return b"".join([table[int(t)] for t in tokens])

    def n_bytes(self, tokens) -> int:
        """UTF-8 byte length of the text ``tokens`` decode to."""
        return len(self.decode_bytes(tokens))

    # -- measurement ---------------------------------------------------------------
    def bytes_per_token(self, text: str) -> float:
        """Mean UTF-8 bytes per token when ``text`` is encoded (higher is better)."""
        n = len(self.encode_ordinary(text))
        return len(text.encode("utf-8")) / max(1, n)

    def bits_per_byte(self, loss_nats: float, n_tokens: int, n_bytes: int) -> float:
        """``r52.tokenizer.bits_per_byte``, re-exposed so callers need only the tokenizer."""
        return bits_per_byte(loss_nats, n_tokens, n_bytes)


Tokenizer = R52Tokenizer
"""Alias: ``from r52.tokenizer_train import Tokenizer``."""


def _split_on(text: str, needles: list[str]) -> Iterator[tuple[str, str | None]]:
    """Split ``text`` on literal ``needles``, yielding ``(piece, needle_or_None)``."""
    i = 0
    while i < len(text):
        hit = min(
            ((text.find(n, i), n) for n in needles if text.find(n, i) >= 0),
            default=(-1, ""),
        )
        pos, needle = hit
        if pos < 0:
            yield text[i:], None
            return
        if pos > i:
            yield text[i:pos], None
        yield "", needle
        i = pos + len(needle)


# --------------------------------------------------------------------------------------
# Dispatch (the `data.tokenizer` config field)
# --------------------------------------------------------------------------------------


def load_tokenizer(spec: str | None = "gpt2"):
    """``"gpt2"`` -> :class:`r52.tokenizer.GPT2Tokenizer`; anything else -> a directory."""
    if not spec or spec == "gpt2":
        from .tokenizer import GPT2Tokenizer

        return GPT2Tokenizer()
    return R52Tokenizer.from_dir(spec)


def tokenizer_vocab_size(spec: str | None = "gpt2") -> int:
    """Model ``vocab_size`` implied by a tokenizer spec (padded for GPT-2, exact for ours)."""
    if not spec or spec == "gpt2":
        from .tokenizer import PADDED_VOCAB

        return PADDED_VOCAB
    meta = Path(spec) / "meta.json"
    if meta.is_file():
        return int(json.loads(meta.read_text())["padded_vocab_size"])
    return int(R52Tokenizer.from_dir(spec).padded_vocab)


# --------------------------------------------------------------------------------------
# Text sources for the CLI
# --------------------------------------------------------------------------------------


def _iter_text_file(path: str | Path, max_bytes: int) -> Iterator[str]:
    """Documents from a UTF-8 text file (``\\n\\n`` separated) or a ``.jsonl`` ``text`` field."""
    p = Path(path)
    used = 0
    if p.suffix == ".jsonl":
        with p.open(encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                t = json.loads(line).get("text") or ""
                if not t:
                    continue
                used += len(t.encode("utf-8"))
                yield t
                if max_bytes and used >= max_bytes:
                    return
        return
    buf: list[str] = []
    with p.open(encoding="utf-8", errors="replace") as fh:
        for line in fh:
            if line.strip():
                buf.append(line)
                continue
            if buf:
                doc = "".join(buf)
                used += len(doc.encode("utf-8"))
                buf = []
                yield doc
                if max_bytes and used >= max_bytes:
                    return
    if buf:
        yield "".join(buf)


def _iter_bin_dir(data_dir: str | Path, max_bytes: int, tokenizer_spec: str, chunk: int = 4096):
    """Documents recovered by **decoding** llm.c ``.bin`` shards back to text.

    A slice prepared by ``scripts/prepare_corpus.py`` with the GPT-2 tokenizer is already on
    disk; decoding it costs nothing and trains our BPE on *exactly the same bytes* the GPT-2
    side of the Axis 3 comparison sees, which is the only way that comparison is fair.
    """
    import numpy as np

    from .data import load_shard, shard_paths

    tok = load_tokenizer(tokenizer_spec)
    paths = shard_paths(data_dir, "*_train_*.bin") or shard_paths(data_dir, "*.bin")
    if not paths:
        raise FileNotFoundError(f"no .bin shards in {data_dir}")
    used = 0
    for path in paths:
        arr = load_shard(path)
        for start in range(0, arr.size, chunk):
            piece = np.asarray(arr[start : start + chunk], dtype=np.int64).tolist()
            text = tok.decode(piece)
            used += len(text.encode("utf-8"))
            yield text
            if max_bytes and used >= max_bytes:
                return


def _iter_corpus(corpus_name: str, max_bytes: int) -> Iterator[str]:
    """Documents streamed straight from a registry corpus (network)."""
    from .ablate.corpora import get_corpus, iter_documents

    used = 0
    for doc in iter_documents(get_corpus(corpus_name)):
        used += doc.n_bytes
        yield doc.text
        if max_bytes and used >= max_bytes:
            return


# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------


def _measure(texts: list[str], tokenizers_: dict[str, Any]) -> dict[str, dict[str, float]]:
    """Bytes/token (and tokens/byte) of each tokenizer on the same held-out text."""
    n_bytes = sum(len(t.encode("utf-8")) for t in texts)
    out: dict[str, dict[str, float]] = {}
    for name, tk in tokenizers_.items():
        n_tok = sum(len(tk.encode_ordinary(t)) for t in texts)
        out[name] = {
            "n_bytes": n_bytes,
            "n_tokens": n_tok,
            "bytes_per_token": n_bytes / max(1, n_tok),
            "n_vocab": int(tk.n_vocab),
        }
    return out


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="r52.tokenizer_train",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--name", required=True, help="tokenizer name; saved to <root>/<name>/")
    p.add_argument("--vocab-size", type=int, default=DEFAULT_VOCAB_SIZE)
    p.add_argument("--root", default=DEFAULT_TOKENIZER_ROOT, help="default: %(default)s")
    p.add_argument("--out", default=None, help="explicit output directory (overrides --root/--name)")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--corpus", default=None, help="stream a corpus from r52/ablate/corpora.yaml")
    src.add_argument("--text-file", default=None, help="a local .txt (blank-line separated) or .jsonl")
    src.add_argument("--from-bin", default=None,
                     help="decode an existing llm.c .bin corpus directory back to text")
    p.add_argument("--bytes", type=float, default=50e6, help="training text budget (default: %(default)s)")
    p.add_argument("--bin-tokenizer", default="gpt2",
                   help="tokenizer that wrote --from-bin (default: %(default)s)")
    p.add_argument("--min-frequency", type=int, default=2)
    p.add_argument("--eval-bytes", type=float, default=2e6,
                   help="held-out bytes used for the bytes/token comparison")
    p.add_argument("--compare-gpt2", action="store_true",
                   help="report bytes/token against GPT-2's BPE on the held-out text")
    p.add_argument("--block-size", type=int, default=1024, help="model_max_length written to the config")
    p.add_argument("--quiet", action="store_true")
    return p


def _source(args) -> tuple[Iterator[str], dict[str, Any]]:
    if args.corpus:
        return _iter_corpus(args.corpus, int(args.bytes)), {"source": f"corpus:{args.corpus}"}
    if args.text_file:
        return _iter_text_file(args.text_file, int(args.bytes)), {"source": f"file:{args.text_file}"}
    return (
        _iter_bin_dir(args.from_bin, int(args.bytes), args.bin_tokenizer),
        {"source": f"bin:{args.from_bin}", "bin_tokenizer": args.bin_tokenizer},
    )


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    out_dir = Path(args.out) if args.out else Path(args.root) / args.name

    stream, provenance = _source(args)
    texts: list[str] = []
    n_bytes = 0
    budget = int(args.bytes)
    for doc in stream:
        texts.append(doc)
        n_bytes += len(doc.encode("utf-8"))
        if budget and n_bytes >= budget:
            break
    if not texts:
        print("no training text", file=sys.stderr)
        return 1

    # Hold out the tail for the bytes/token comparison so the number is not measured on the
    # text the merges were fitted to.
    hold = max(1, int(len(texts) * min(0.2, args.eval_bytes / max(1, n_bytes))))
    train_texts, eval_texts = texts[:-hold] or texts, texts[-hold:]

    if not args.quiet:
        print(
            f"[tok] training vocab {args.vocab_size:,} on {len(train_texts):,} documents "
            f"/ {n_bytes / 1e6:.1f} MB ({provenance['source']})",
            flush=True,
        )
    t0 = time.time()
    tok = train_tokenizer(
        train_texts,
        args.vocab_size,
        min_frequency=args.min_frequency,
        show_progress=not args.quiet,
    )
    train_s = time.time() - t0

    ours = R52Tokenizer(tok)
    measured = {args.name: ours}
    if args.compare_gpt2:
        from .tokenizer import GPT2Tokenizer

        measured["gpt2"] = GPT2Tokenizer()
    stats = _measure(eval_texts, measured)

    train_bytes = sum(len(t.encode("utf-8")) for t in train_texts)
    meta = {
        **provenance,
        "streamed_bytes": n_bytes,
        "streamed_documents": len(texts),
        "train_bytes": train_bytes,
        "train_documents": len(train_texts),
        "train_seconds": round(train_s, 2),
        "min_frequency": args.min_frequency,
        "holdout_documents": len(eval_texts),
        "holdout_bytes": n_bytes - train_bytes,
        "bytes_per_token": stats,
    }
    save_tokenizer(
        tok,
        out_dir,
        name=args.name,
        vocab_size=args.vocab_size,
        meta=meta,
        model_max_length=args.block_size,
    )

    print(f"\nwrote {out_dir}/tokenizer.json  ({train_s:.1f} s, vocab {args.vocab_size:,})")
    print(f"held-out text: {stats[args.name]['n_bytes'] / 1e6:.2f} MB")
    for name, s in stats.items():
        print(
            f"  {name:<20} vocab {s['n_vocab']:>7,}  {s['n_tokens']:>12,} tokens  "
            f"{s['bytes_per_token']:.4f} bytes/token"
        )
    if args.compare_gpt2 and "gpt2" in stats:
        gain = stats[args.name]["bytes_per_token"] / stats["gpt2"]["bytes_per_token"]
        print(
            f"  -> {args.name} needs {100 * (1 - 1 / gain):+.2f}% "
            f"{'fewer' if gain > 1 else 'more'} tokens for the same bytes than GPT-2"
            if not math.isclose(gain, 1.0)
            else "  -> identical"
        )
    return 0


if __name__ == "__main__":
    rc = main()
    # A `--corpus` run abandons a `datasets` streaming iterator, which deadlocks PyArrow's
    # thread pool at interpreter exit (docs/DEVIATIONS.md P6). Everything is flushed by now.
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(rc)
