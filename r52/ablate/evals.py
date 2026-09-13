# Copyright 2026 The road-to-52 authors.
# SPDX-License-Identifier: Apache-2.0
"""Score one ablation checkpoint: val bpb + HellaSwag + a CORE subset.

    python -m r52.ablate.evals --model runs/abl-corpus-dclm/ckpt/best \
        --data-dir data/corpora/dclm-fineweb32k --val-file dclm_val_000000.bin \
        --tokenizer data/tokenizers/fineweb32k --hellaswag-limit 2000 \
        --core-tasks copa,winograd --out results/ablations/corpus/dclm.eval.json

This exists because :mod:`r52.eval` is fixed to GPT-2's BPE: ``r52.eval.lm.LM`` constructs a
:class:`r52.tokenizer.GPT2Tokenizer` in its ``__init__`` and every eval reads ``lm.tokenizer``
from there, while ``r52.eval.val_loss`` computes bytes-per-byte through the GPT-2 decoder.
Both are correct for every model in the ladder and wrong for an Axis-3 model trained on our
own 32K vocabulary.  Rather than change ``r52/eval/`` -- which the whole results history
depends on -- this module loads the same ``LM``, **replaces its tokenizer**, and calls the
same ``evaluate_hellaswag`` / ``evaluate_core`` / ``evaluate_val_loss`` functions, so the
numbers come out of exactly one implementation.

The three metrics and why these three:

``val_bpb``
    Bits per byte on the cell's *own* validation shard.  bpb rather than loss because Axis 3
    compares vocabularies, and a cross-entropy in nats/token is not comparable between them.
``hellaswag``
    ``acc_norm`` with a 95% Wilson interval -- tokenizer-neutral and corpus-neutral, and the
    downstream signal ``docs/PLAN.md`` §3.1 insists on reporting next to loss.
``core``
    The 6-task CORE subset from :mod:`r52.ablate.matrix`, centered on random baselines, so a
    model that guesses scores 0.

A separate process per checkpoint, launched by :mod:`r52.ablate.run`, so the trainer's
memory is fully released before any eval allocates.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import mlx.core as mx

from ..eval.core import evaluate_core
from ..eval.hellaswag import evaluate_hellaswag, load_examples
from ..eval.lm import LM, configure_runtime
from ..eval.val_loss import evaluate_val_loss, make_loader
from ..tokenizer import bits_per_byte, val_bytes_per_token
from ..tokenizer_train import load_tokenizer

__all__ = ["evaluate_cell", "main"]

DEFAULT_MEMORY_GIB = 3.0
DEFAULT_MAX_POSITIONS = 2048


def evaluate_cell(
    model: str | Path,
    *,
    tokenizer: str = "gpt2",
    data_dir: str | Path | None = None,
    val_file: str | None = None,
    val_tokens: int = 1_000_000,
    block_size: int = 1024,
    micro_batch: int = 2,
    hellaswag_limit: int = 2000,
    hellaswag_dir: str = "data/hellaswag",
    core_tasks: list[str] | None = None,
    core_limit: int = 0,
    max_positions: int = DEFAULT_MAX_POSITIONS,
    verbose: bool = True,
) -> dict[str, Any]:
    """Run the three evals on one checkpoint and return a JSON-safe dict."""
    lm = LM.load(model, max_positions_per_forward=max_positions)
    tok = load_tokenizer(tokenizer)
    lm.tokenizer = tok  # the whole reason this module exists; see the module docstring
    out: dict[str, Any] = {
        "model": lm.describe(),
        "tokenizer": {
            "spec": tokenizer,
            "kind": getattr(tok, "kind", "gpt2"),
            "n_vocab": int(tok.n_vocab),
            "eot": int(tok.eot),
        },
    }
    if int(tok.padded_vocab) != int(lm.vocab_size):
        raise ValueError(
            f"tokenizer {tokenizer!r} has a padded vocabulary of {tok.padded_vocab} but "
            f"{model} has {lm.vocab_size} output rows -- the checkpoint and the tokenizer do "
            f"not belong together"
        )

    if data_dir and val_file:
        t0 = time.time()
        loader, array = make_loader(data_dir, val_file, block_size, micro_batch, val_tokens)
        res = evaluate_val_loss(lm, loader, verbose=verbose)
        bpt = val_bytes_per_token(
            array,
            int(res["n_tokens"]),
            Path(data_dir) / ".bpb_cache.json",
            "val",
            tokenizer=tok,
        )
        res["bytes_per_token"] = bpt
        res["n_bytes"] = int(res["n_tokens"] * bpt)
        res["val_bpb"] = bits_per_byte(res["val_loss"], int(res["n_tokens"]), res["n_bytes"])
        res["wall_clock_s"] = round(time.time() - t0, 2)
        out["val"] = res
        if verbose:
            print(
                f"[abl] val_loss {res['val_loss']:.4f} | bpb {res['val_bpb']:.4f} | "
                f"{res['bytes_per_token']:.3f} bytes/token | {res['n_tokens']:,} tokens",
                flush=True,
            )

    if hellaswag_limit >= 0:
        examples = load_examples(hellaswag_dir, "validation", hellaswag_limit)
        hs = evaluate_hellaswag(lm, examples, verbose=verbose, progress_every=1000)
        hs["acc_ci95"] = list(hs["acc_ci95"])
        hs["acc_norm_ci95"] = list(hs["acc_norm_ci95"])
        hs["limit"] = hellaswag_limit or None
        out["hellaswag"] = hs
        if verbose:
            lo, hi = hs["acc_norm_ci95"]
            print(
                f"[abl] hellaswag acc_norm {hs['acc_norm']:.4f} [{lo:.4f}, {hi:.4f}] "
                f"over {hs['n_examples']:,}",
                flush=True,
            )

    if core_tasks:
        core = evaluate_core(lm, limit=core_limit, only=core_tasks, verbose=verbose)
        core["tasks"] = list(core_tasks)
        out["core"] = core
        if verbose:
            print(f"[abl] CORE-{core['n_tasks']} {core['core_metric']:.6f}", flush=True)

    out["peak_memory_gib"] = mx.get_peak_memory() / 2**30
    return out


# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="r52.ablate.evals",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--model", required=True, help="r52 checkpoint or mlx-lm model directory")
    p.add_argument("--tokenizer", default="gpt2", help="'gpt2' or a data/tokenizers/<name> directory")
    p.add_argument("--data-dir", default=None, help="corpus directory holding the val shard")
    p.add_argument("--val-file", default=None)
    p.add_argument("--val-tokens", type=int, default=1_000_000)
    p.add_argument("--block-size", type=int, default=1024)
    p.add_argument("--micro-batch", type=int, default=2)
    p.add_argument("--hellaswag-limit", type=int, default=2000, help="0 = all 10,042; -1 = skip")
    p.add_argument("--hellaswag-dir", default="data/hellaswag")
    p.add_argument("--core-tasks", default="", help="comma-separated CORE task labels ('' = skip)")
    p.add_argument("--core-limit", type=int, default=0, help="items per CORE task (0 = all)")
    p.add_argument("--max-positions", type=int, default=DEFAULT_MAX_POSITIONS,
                   help="rows*tokens per forward; bounds fp32 logits memory")
    p.add_argument("--memory-limit-gib", type=float, default=DEFAULT_MEMORY_GIB)
    p.add_argument("--out", default=None, help="write the results JSON here")
    p.add_argument("--quiet", action="store_true")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    configure_runtime(args.memory_limit_gib)
    tasks = [t.strip() for t in args.core_tasks.split(",") if t.strip()]
    t0 = time.time()
    out = evaluate_cell(
        args.model,
        tokenizer=args.tokenizer,
        data_dir=args.data_dir,
        val_file=args.val_file,
        val_tokens=args.val_tokens,
        block_size=args.block_size,
        micro_batch=args.micro_batch,
        hellaswag_limit=args.hellaswag_limit,
        hellaswag_dir=args.hellaswag_dir,
        core_tasks=tasks,
        core_limit=args.core_limit,
        max_positions=args.max_positions,
        verbose=not args.quiet,
    )
    out["wall_clock_s"] = round(time.time() - t0, 2)
    if args.out:
        p = Path(args.out)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(out, indent=2))
        print(f"wrote {p}", flush=True)
    else:
        print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
