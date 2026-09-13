# Copyright 2026 The road-to-52 authors.
# SPDX-License-Identifier: Apache-2.0
#
# Protocol ported from karpathy/llm.c `dev/data/hellaswag.py` (MIT).  No code copied; the
# tokenization, the masking and the two decision rules are reproduced so that our numbers sit
# on the same scale as the reference figures in that file's docstring.
"""HellaSwag validation: 10,042 four-way sentence completions, loglikelihood scoring.

    python -m r52.eval.hellaswag --model models/gpt2-mlx
    python -m r52.eval.hellaswag --model runs/b1/ckpt/best --limit 50

**Protocol** (llm.c's ``dev/data/hellaswag.py``, "completion style"):

1. ``ctx`` is tokenized with GPT-2 tiktoken as-is -- no lm-eval-style text preprocessing.
2. Each of the four ``endings`` is tokenized as ``" " + ending``; the leading space matters,
   because GPT-2's BPE encodes a word differently at the start of a string.
3. The four rows ``ctx_tokens + ending_tokens`` are right-padded into one batch with a mask
   that is 1 exactly on the ending tokens.
4. ``acc``      picks the ending with the largest **sum** of ending-token log-probabilities.
   ``acc_norm`` picks the largest sum **divided by the number of ending tokens**.

**Reference numbers.**  ``dev/data/hellaswag.py``'s module docstring (fetched from
``karpathy/llm.c`` on 2026-09-13) states, verbatim::

    gpt2 (124M)
    - eleuther harness reports acc 28.92%, acc_norm 31.14% (multiple choice style)
    - this script: 10042 acc: 0.2859 acc_norm: 0.2955 (completion style)

So the figures this module should land on for the GPT-2 124M reference are **acc 0.2859 and
acc_norm 0.2955**.  The 0.2892 / 0.3114 pair belongs to lm-evaluation-harness, whose
``hellaswag`` task is *not* the same protocol: it rewrites the text (activity label prefix,
bracket stripping, whitespace collapsing) and normalises ``acc_norm`` by the **character**
length of the continuation rather than its token count.  For the second half of that
comparison this module also reports ``acc_norm_bytes`` (UTF-8-byte normalisation, which is
lm-eval's normaliser without lm-eval's text rewriting) -- it is free, since it comes from the
same forward passes.  llm.c's README does not carry a HellaSwag number at all; the docstring
above is the primary source.

Confidence intervals are 95 % Wilson score intervals on the proportion.
"""

from __future__ import annotations

import argparse
import json
import time
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mlx.core as mx
import numpy as np

from .lm import DEFAULT_MEMORY_LIMIT_GIB, LM, configure_runtime, pad_stack, wilson_interval

__all__ = [
    "BENCHMARK",
    "N_VALIDATION",
    "Example",
    "download_hellaswag",
    "evaluate_hellaswag",
    "load_examples",
    "main",
    "render_example",
]

BENCHMARK = "hellaswag"
N_VALIDATION = 10_042
"""Examples in the HellaSwag validation split."""

HF_DATASET = "Rowan/hellaswag"
RAW_URL = "https://raw.githubusercontent.com/rowanz/hellaswag/master/data/hellaswag_val.jsonl"
DEFAULT_DATA_DIR = "data/hellaswag"


@dataclass(frozen=True)
class Example:
    """One HellaSwag item: a context, four endings and the index of the correct one."""

    ctx: str
    endings: tuple[str, ...]
    label: int
    ind: int = -1


# --------------------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------------------


def download_hellaswag(data_dir: str | Path = DEFAULT_DATA_DIR, split: str = "validation") -> Path:
    """Cache the HellaSwag split as jsonl under ``data_dir`` and return its path.

    Primary source is the ``Rowan/hellaswag`` dataset on the Hugging Face Hub; if ``datasets``
    is unavailable or the Hub call fails, the raw jsonl from ``rowanz/hellaswag`` (the file
    llm.c downloads) is fetched instead.  Both carry the same 10,042 validation rows.
    """
    d = Path(data_dir)
    d.mkdir(parents=True, exist_ok=True)
    name = {"validation": "val", "train": "train", "test": "test"}.get(split, split)
    path = d / f"hellaswag_{name}.jsonl"
    if path.exists() and path.stat().st_size > 0:
        return path

    rows: list[dict[str, Any]] | None = None
    try:
        from datasets import load_dataset

        ds = load_dataset(HF_DATASET, split=split)
        rows = [
            {
                "ind": int(r.get("ind", i)),
                "ctx": r["ctx"],
                "endings": list(r["endings"]),
                "label": r["label"],
            }
            for i, r in enumerate(ds)
        ]
    except Exception as exc:  # pragma: no cover - network / datasets fallback
        print(f"[hellaswag] datasets load failed ({exc}); falling back to {RAW_URL}", flush=True)

    if rows is None:  # pragma: no cover - network fallback
        import urllib.request

        if name != "val":
            raise RuntimeError(f"no raw-jsonl fallback for split {split!r}")
        with urllib.request.urlopen(RAW_URL, timeout=120) as fh:
            body = fh.read().decode("utf-8")
        rows = [json.loads(line) for line in body.splitlines() if line.strip()]

    tmp = path.with_suffix(".jsonl.tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    tmp.rename(path)
    return path


def load_examples(
    data_dir: str | Path = DEFAULT_DATA_DIR, split: str = "validation", limit: int = 0
) -> list[Example]:
    """Load (downloading if needed) the split as :class:`Example` records, in file order."""
    path = download_hellaswag(data_dir, split)
    out: list[Example] = []
    with path.open("r", encoding="utf-8") as fh:
        for i, line in enumerate(fh):
            if not line.strip():
                continue
            r = json.loads(line)
            label = r.get("label")
            if label is None or label == "":
                raise ValueError(f"{path}: example {i} has no label (the test split is unlabelled)")
            out.append(
                Example(
                    ctx=r["ctx"],
                    endings=tuple(r["endings"]),
                    label=int(label),
                    ind=int(r.get("ind", i)),
                )
            )
            if limit and len(out) >= limit:
                break
    return out


# --------------------------------------------------------------------------------------
# Rendering and scoring
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Rendered:
    """One example tokenized into four padded-later rows.

    Attributes:
        example: the source :class:`Example`.
        tokens: four token rows, ``ctx_tokens + ending_tokens``.
        masks: four 0/1 rows, 1 exactly on the ending tokens.
        end_bytes: UTF-8 byte length of each ending (for the lm-eval-style normaliser).
        truncated: how many of the four rows were cut to ``max_context``.
    """

    example: Example
    tokens: list[list[int]]
    masks: list[list[int]]
    end_bytes: list[int]
    truncated: int

    @property
    def width(self) -> int:
        return max(len(r) for r in self.tokens)


def render_example(example: Example, tokenizer, max_context: int = 0) -> Rendered:
    """Tokenize one example into four ``(tokens, mask)`` rows.

    The mask is 1 on the ending tokens and 0 on the context.  With ``max_context > 0`` a row
    longer than that keeps its **last** ``max_context`` tokens (nanochat's rule; it preserves
    the ending, which is the part being scored).
    """
    ctx_tokens = tokenizer.encode(example.ctx, allowed_special=set())
    tok_rows, mask_rows, end_bytes = [], [], []
    truncated = 0
    for ending in example.endings:
        end_tokens = tokenizer.encode(" " + ending, allowed_special=set())
        row = ctx_tokens + end_tokens
        mask = [0] * len(ctx_tokens) + [1] * len(end_tokens)
        if max_context and len(row) > max_context:
            row, mask = row[-max_context:], mask[-max_context:]
            truncated += 1
        tok_rows.append(row)
        mask_rows.append(mask)
        end_bytes.append(len(tokenizer.decode_bytes(end_tokens)))
    return Rendered(example, tok_rows, mask_rows, end_bytes, truncated)


def _groups(
    examples: list[Example], tokenizer, max_context: int, budget: int
) -> Iterator[list[Rendered]]:
    """Yield batches of rendered examples whose ``rows * max_len`` stays under ``budget``."""
    batch: list[Rendered] = []
    rows = 0
    width = 0
    for ex in examples:
        r = render_example(ex, tokenizer, max_context)
        w = max(width, r.width)
        if batch and (rows + len(r.tokens)) * w > budget:
            yield batch
            batch, rows = [], 0
            w = r.width
        batch.append(r)
        rows += len(r.tokens)
        width = w
    if batch:
        yield batch


def evaluate_hellaswag(
    lm: LM,
    examples: Iterable[Example],
    *,
    progress_every: int = 1000,
    verbose: bool = True,
) -> dict[str, Any]:
    """Run the eval. Returns accuracies, counts, CIs and the number of truncated rows."""
    examples = list(examples)
    budget = lm.max_positions_per_forward
    n_correct = n_correct_norm = n_correct_bytes = 0
    n_total = 0
    n_truncated = 0
    t0 = time.time()

    for batch in _groups(examples, lm.tokenizer, lm.max_context, budget):
        flat_rows: list[list[int]] = []
        flat_masks: list[list[int]] = []
        for r in batch:
            n_truncated += r.truncated
            flat_rows.extend(r.tokens)
            flat_masks.extend(r.masks)
        tokens = pad_stack(flat_rows)
        mask = pad_stack(flat_masks).astype(mx.float32)
        nll, _ = lm.scores(tokens)
        # llm.c: shift the mask by one so scoring starts at the last context token, which is
        # the position that predicts the first ending token.
        sum_loss = (nll[:, :-1] * mask[:, 1:]).sum(axis=1)
        n_end = mask[:, 1:].sum(axis=1)
        mx.eval(sum_loss, n_end)
        sum_np = np.asarray(sum_loss, dtype=np.float64)
        cnt_np = np.asarray(n_end, dtype=np.float64)

        off = 0
        for r in batch:
            k = len(r.tokens)
            s = sum_np[off : off + k]
            c = np.maximum(cnt_np[off : off + k], 1.0)
            b = np.maximum(np.asarray(r.end_bytes, dtype=np.float64), 1.0)
            off += k
            n_total += 1
            n_correct += int(int(np.argmin(s)) == r.example.label)
            n_correct_norm += int(int(np.argmin(s / c)) == r.example.label)
            n_correct_bytes += int(int(np.argmin(s / b)) == r.example.label)
        if verbose and progress_every and n_total % progress_every < len(batch):
            el = time.time() - t0
            print(
                f"  [hellaswag] {n_total}/{len(examples)} acc {n_correct / n_total:.4f} "
                f"acc_norm {n_correct_norm / n_total:.4f} | {n_total / el:.1f} ex/s "
                f"| eta {el * (len(examples) / max(1, n_total) - 1) / 60:.1f} min",
                flush=True,
            )

    acc = n_correct / max(1, n_total)
    acc_norm = n_correct_norm / max(1, n_total)
    acc_bytes = n_correct_bytes / max(1, n_total)
    return {
        "acc": acc,
        "acc_norm": acc_norm,
        "acc_norm_bytes": acc_bytes,
        "acc_ci95": wilson_interval(n_correct, n_total),
        "acc_norm_ci95": wilson_interval(n_correct_norm, n_total),
        "n_correct": n_correct,
        "n_correct_norm": n_correct_norm,
        "n_examples": n_total,
        "n_truncated_rows": n_truncated,
        "wall_clock_s": time.time() - t0,
    }


# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="r52.eval.hellaswag",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--model", required=True,
                   help="r52 checkpoint directory or mlx-lm model directory")
    p.add_argument("--limit", type=int, default=0, help="score only the first N examples (0 = all)")
    p.add_argument("--split", default="validation")
    p.add_argument("--data-dir", default=DEFAULT_DATA_DIR)
    p.add_argument("--max-positions", type=int, default=4096,
                   help="rows*tokens per forward, bounds fp32 logits memory (default 4096)")
    p.add_argument("--memory-limit-gib", type=float, default=DEFAULT_MEMORY_LIMIT_GIB)
    p.add_argument("--report", action="store_true")
    p.add_argument("--run", default=None)
    p.add_argument("--results-dir", default=None,
                   help="write results here instead of results/ (also moves the markdown row "
                        "to <dir>/RESULTS.md, so smoke runs leave docs/RESULTS.md alone)")
    p.add_argument("--model-id", default=None, help="bar.yaml model id (e.g. gpt2-124m)")
    p.add_argument("--quiet", action="store_true")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    configure_runtime(args.memory_limit_gib)

    lm = LM.load(args.model, max_positions_per_forward=args.max_positions)
    examples = load_examples(args.data_dir, args.split, args.limit)
    if not args.quiet:
        print(f"[hellaswag] {lm.describe()}", flush=True)
        print(f"[hellaswag] {len(examples):,} examples from {args.data_dir}", flush=True)

    t0 = time.time()
    out = evaluate_hellaswag(lm, examples, verbose=not args.quiet)
    wall = time.time() - t0
    lo, hi = out["acc_ci95"]
    nlo, nhi = out["acc_norm_ci95"]
    print(
        f"acc {out['acc']:.4f} [{lo:.4f}, {hi:.4f}] | "
        f"acc_norm {out['acc_norm']:.4f} [{nlo:.4f}, {nhi:.4f}] | "
        f"acc_norm_bytes {out['acc_norm_bytes']:.4f} | "
        f"{out['n_examples']:,} examples | {wall:.1f} s | "
        f"peak memory {mx.get_peak_memory() / 2**30:.2f} GiB",
        flush=True,
    )

    if args.report:
        from .report import make_result, write_result

        run = args.run or lm.name
        cmd = f"python -m r52.eval.hellaswag --model {args.model}" + (
            f" --limit {args.limit}" if args.limit else ""
        )
        result = make_result(
            model=args.model_id or lm.name,
            benchmark=BENCHMARK,
            value=100.0 * out["acc_norm"],
            unit="% acc_norm",
            conditions={
                "tokenizer": "gpt2 (tiktoken)",
                "block_size": lm.max_context,
                "limit": args.limit or None,
                "n_examples": out["n_examples"],
                "few_shot": 0,
                "protocol": "llm.c dev/data/hellaswag.py completion style, token-length acc_norm",
                "command": cmd,
            },
            wall_clock_s=wall,
            metrics={
                "acc": out["acc"],
                "acc_norm": out["acc_norm"],
                "acc_norm_bytes": out["acc_norm_bytes"],
                "acc_ci95": list(out["acc_ci95"]),
                "acc_norm_ci95": list(out["acc_norm_ci95"]),
                "n_correct": out["n_correct"],
                "n_correct_norm": out["n_correct_norm"],
                "n_examples": out["n_examples"],
                "n_truncated_rows": out["n_truncated_rows"],
                "peak_memory_gib": mx.get_peak_memory() / 2**30,
                "model": lm.describe(),
            },
        )
        path = write_result(
            result, run, "hellaswag",
            results_root=Path(args.results_dir) if args.results_dir else None,
        )
        print(f"wrote {path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
