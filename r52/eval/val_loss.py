# Copyright 2026 The road-to-52 authors.
# SPDX-License-Identifier: Apache-2.0
"""Validation loss and bits-per-byte on the fixed FineWeb split.

    python -m r52.eval.val_loss --model models/gpt2-mlx
    python -m r52.eval.val_loss --model runs/b1/ckpt/best --block-size 1024
    python -m r52.eval.val_loss --model runs/b1/ckpt/best --max-tokens 262144   # smoke test

**The split.** The first ``10,485,760`` tokens of ``data/fineweb10B-gpt2/fineweb_val_000000.bin``
(llm.c ``.bin`` format), cut into **non-overlapping** ``block_size``-token windows, exactly as
modded-nanogpt does and exactly as :func:`r52.train.evaluate` does -- this module reuses
:class:`r52.data.ValLoader`, so the two cannot drift apart.  Every one of the 10,485,760
tokens after the first is scored as a target exactly once.

**The numbers.**

``val_loss``
    Mean cross-entropy in **nats per token**, fp32 log-sum-exp over the model's bf16 logits
    (``GPT.loss(..., fp32_logits=True)``'s arithmetic).  The Rung-0 target is <= 3.28
    (``docs/ARCHITECTURE.md`` §10), which is GPT-2-small quality on this split.
``val_bpb``
    **Bits per byte** = ``(val_loss / ln 2) * (tokens / bytes)``.  ``bytes`` is the UTF-8
    length of the scored tokens, a property of the corpus, not the model -- measured once by
    :func:`r52.tokenizer.val_bytes_per_token` and cached in ``data/*/.bpb_cache.json``.  bpb
    is the tokenizer-independent form of the same number, so it is the one that stays
    comparable if a later rung trains its own BPE.

Wall-clock is dominated by ``10.5M tokens / throughput``; on an idle M4 a 124M model reads the
full split in roughly twenty minutes, longer when a pretraining run owns the GPU.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any

import mlx.core as mx

from ..config import DataConfig
from ..data import ValLoader, load_shard
from ..tokenizer import bits_per_byte, val_bytes_per_token
from .lm import DEFAULT_MEMORY_LIMIT_GIB, LM, configure_runtime

__all__ = ["evaluate_val_loss", "main", "make_loader"]

BENCHMARK = "fineweb-val-loss"
FULL_SPLIT_TOKENS = 10_485_760
"""The fixed validation split size, in tokens (modded-nanogpt's; ``DataConfig.val_tokens``)."""


def make_loader(
    data_dir: str | Path,
    val_file: str,
    block_size: int,
    micro_batch: int,
    max_tokens: int,
) -> tuple[ValLoader, Any]:
    """Build the fixed-split loader. Returns ``(loader, token_array)``."""
    path = Path(data_dir) / val_file
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run: python scripts/prepare_data.py --train-shards 1"
        )
    array = load_shard(path)
    return ValLoader(array, block_size, micro_batch, max_tokens), array


def evaluate_val_loss(
    lm: LM,
    loader: ValLoader,
    *,
    array: Any = None,
    bpb_cache: Path | None = None,
    progress_every: int = 200,
    verbose: bool = True,
) -> dict[str, Any]:
    """Mean CE (nats/token) and bits-per-byte over ``loader``.

    The accumulation is deliberately identical to :func:`r52.train.evaluate`: a per-batch mean
    in fp32, weighted by the batch's token count and summed in float64 on the host.  That is
    what makes this module reproduce the ``val_loss`` a training run reports for the same
    checkpoint rather than merely agree with it to a few decimals.
    """
    total_nll = 0.0
    total_tok = 0
    t0 = time.time()
    for i, (x, y) in enumerate(loader):
        loss = lm.mean_nll(x, y)
        n = int(y.size)
        total_nll += loss * n
        total_tok += n
        if verbose and progress_every and (i + 1) % progress_every == 0:
            done = (i + 1) / len(loader)
            elapsed = time.time() - t0
            print(
                f"  [val_loss] batch {i + 1}/{len(loader)} "
                f"({100 * done:.1f}%) loss {total_nll / total_tok:.4f} "
                f"| {total_tok / elapsed:,.0f} tok/s | eta {elapsed * (1 / done - 1) / 60:.1f} min",
                flush=True,
            )
    val_loss = total_nll / max(1, total_tok)
    out: dict[str, Any] = {
        "val_loss": val_loss,
        "n_tokens": total_tok,
        "n_batches": len(loader),
        "block_size": loader.block_size,
        "micro_batch": loader.micro_batch,
        "wall_clock_s": time.time() - t0,
    }
    if array is not None:
        bpt = val_bytes_per_token(array, total_tok, bpb_cache, "val")
        out["bytes_per_token"] = bpt
        out["n_bytes"] = int(total_tok * bpt)
        out["val_bpb"] = bits_per_byte(val_loss, total_tok, out["n_bytes"])
    return out


# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    d = DataConfig()
    p = argparse.ArgumentParser(
        prog="r52.eval.val_loss",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--model", required=True,
                   help="r52 checkpoint directory or mlx-lm model directory")
    p.add_argument("--block-size", type=int, default=1024,
                   help="window length in tokens (default 1024; the GPT-2 reference uses 1024)")
    p.add_argument("--max-tokens", type=int, default=FULL_SPLIT_TOKENS,
                   help=f"tokens of the val split to score (default {FULL_SPLIT_TOKENS:,})")
    p.add_argument("--micro-batch", type=int, default=2,
                   help="windows per forward pass (peak memory knob; default 2)")
    p.add_argument("--data-dir", default=d.data_dir)
    p.add_argument("--val-file", default=d.val_file)
    p.add_argument("--max-positions", type=int, default=4096,
                   help="rows*tokens per forward, bounds fp32 logits memory (default 4096)")
    p.add_argument("--memory-limit-gib", type=float, default=DEFAULT_MEMORY_LIMIT_GIB)
    p.add_argument("--report", action="store_true", help="write results JSON + docs/RESULTS.md row")
    p.add_argument("--run", default=None, help="results/<run>/ directory (default: the model name)")
    p.add_argument("--results-dir", default=None,
                   help="write results here instead of results/ (also moves the markdown row "
                        "to <dir>/RESULTS.md, so smoke runs leave docs/RESULTS.md alone)")
    p.add_argument("--model-id", default=None,
                   help="bar.yaml model id for the gap tracker (e.g. gpt2-124m)")
    p.add_argument("--quiet", action="store_true")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    configure_runtime(args.memory_limit_gib)

    lm = LM.load(args.model, max_positions_per_forward=args.max_positions)
    if args.block_size > lm.max_context:
        # Clamp rather than abort: small models (e.g. tiny test configs) have short contexts.
        # The recorded conditions carry the effective block size, so comparisons stay honest.
        print(
            f"--block-size {args.block_size} exceeds the model's context ({lm.max_context}); "
            f"using {lm.max_context}",
            flush=True,
        )
        args.block_size = lm.max_context
    loader, array = make_loader(
        args.data_dir, args.val_file, args.block_size, args.micro_batch, args.max_tokens
    )
    scored = loader.tokens
    if loader.n_windows % args.micro_batch:
        print(
            f"[val_loss] note: {scored:,} tokens scored, not "
            f"{loader.n_windows * args.block_size:,} -- micro_batch={args.micro_batch} does not "
            f"divide the {loader.n_windows:,} windows evenly, so the tail is dropped",
            flush=True,
        )
    if not args.quiet:
        print(f"[val_loss] {lm.describe()}", flush=True)
        print(
            f"[val_loss] {len(loader):,} batches x {loader.micro_batch} x {loader.block_size} "
            f"= {scored:,} target tokens",
            flush=True,
        )

    t0 = time.time()
    out = evaluate_val_loss(
        lm,
        loader,
        array=array,
        bpb_cache=Path(args.data_dir) / ".bpb_cache.json",
        verbose=not args.quiet,
    )
    wall = time.time() - t0
    bpb = out.get("val_bpb")
    bpb_s = f"val_bpb {bpb:.6f} bits/byte | " if bpb is not None else ""
    print(
        f"val_loss {out['val_loss']:.6f} nats/token | {bpb_s}"
        f"{out['n_tokens']:,} tokens | {wall:.1f} s | "
        f"peak memory {mx.get_peak_memory() / 2**30:.2f} GiB",
        flush=True,
    )

    if args.report:
        from .report import make_result, write_result

        run = args.run or lm.name
        cmd = (
            f"python -m r52.eval.val_loss --model {args.model} "
            f"--block-size {args.block_size} --max-tokens {args.max_tokens} "
            f"--micro-batch {args.micro_batch}"
        )
        result = make_result(
            model=args.model_id or lm.name,
            benchmark=BENCHMARK,
            value=out["val_loss"],
            unit="nats/token",
            conditions={
                "tokenizer": "gpt2 (tiktoken)",
                "block_size": args.block_size,
                "limit": None if args.max_tokens >= FULL_SPLIT_TOKENS else args.max_tokens,
                "n_examples": out["n_tokens"],
                "few_shot": None,
                "split": f"{args.val_file} first {out['n_tokens']:,} tokens, non-overlapping windows",
                "precision": "bf16 compute, fp32 log-sum-exp",
                "command": cmd,
            },
            wall_clock_s=wall,
            metrics={
                "val_loss": out["val_loss"],
                "val_bpb": bpb,
                "bytes_per_token": out.get("bytes_per_token"),
                "n_tokens": out["n_tokens"],
                "n_batches": out["n_batches"],
                "peak_memory_gib": mx.get_peak_memory() / 2**30,
                "model": lm.describe(),
            },
        )
        path = write_result(
            result, run, "val_loss",
            results_root=Path(args.results_dir) if args.results_dir else None,
        )
        print(f"wrote {path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
