#!/usr/bin/env python
# Copyright 2026 The road-to-52 authors.
# SPDX-License-Identifier: Apache-2.0
"""Render an instruction dataset into r52 SFT splits (tokens + assistant loss mask).

    python scripts/prepare_sft_data.py --n 1000                  # smol-smoltalk, 1k convs
    python scripts/prepare_sft_data.py --n 200000 --max-seq 1024 --out data/sft/smol-smoltalk
    python scripts/prepare_sft_data.py --stats data/sft/smol-smoltalk

Default dataset: **``HuggingFaceTB/smol-smoltalk``**, chosen in ``docs/PLAN.md`` §5.

Provenance check actually run on 2026-09-13 (``docs/ARCHITECTURE.md`` §1.1 forbids
Claude-derived training data), against the HF datasets-server and the dataset cards:

* ``GET huggingface.co/api/datasets/HuggingFaceTB/smol-smoltalk`` -> ``license: apache-2.0``,
  public, not gated, 460,341 train + 24,229 test rows (``datasets-server.huggingface.co/size``),
  features ``messages`` (``role``/``content``) and ``source``.
* Teacher lineage, from the parent ``HuggingFaceTB/smoltalk`` card: the core
  *Smol-Magpie-Ultra* split is **generated with Llama-3.1-405B-Instruct** (an open-weight
  model), plus public sets (OpenHermes-2.5, MetaMathQA, NuminaMath-CoT,
  self-oss-instruct-sc2, SystemChat-2.0, LongAlign).  **Correction to the build brief:** it
  is Llama-3.1-405B, not Qwen2.5.  Either way it is an open-weight teacher; **no Claude
  output is in the lineage**.  (Caveat worth knowing: OpenHermes-2.5 carries GPT-4-derived
  text, which §5 says to "avoid by default" -- filterable via the ``source`` column if the
  planner wants it gone.  See ``docs/DEVIATIONS.md``.)
* The §5 exclusion list (``SWE-smith-trajectories``, ``OpenHands-*-Trajectories``,
  ``R2EGym-SFT-Trajectories``, ``llama-3.1-tulu-3-8b-preference-mixture`` and SmolTalk2's
  Preference split, ``Anthropic/hh-rlhf``, ``*claude-code-traces*``, ``dolphin-distill``)
  has no SmolTalk/smol-smoltalk entry.  Note the distinction: the excluded item is
  **SmolTalk2's *Preference* split**, which inherits the Tulu-3 preference mixture; this
  script reads ``smol-smoltalk``'s SFT conversations only, and ``--dataset`` refuses any
  repo id matching :data:`EXCLUDED`.

Output layout (see :mod:`r52.posttrain.data`)::

    <out>/train/{tokens,mask,offsets}.npy + meta.json
    <out>/valid/{tokens,mask,offsets}.npy + meta.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from r52.chat_template import ASSISTANT_END, render
from r52.posttrain.data import ConversationDataset, write_split
from r52.tokenizer import GPT2Tokenizer

DEFAULT_DATASET = "HuggingFaceTB/smol-smoltalk"
DEFAULT_OUT = "data/sft/smol-smoltalk"

EXCLUDED = (
    "swe-bench/swe-smith-trajectories",
    "swe-gym/openhands",
    "r2e-gym/r2egym-sft-trajectories",
    "allenai/llama-3.1-tulu-3-8b-preference-mixture",
    "anthropic/hh-rlhf",
    "claude-code-traces",
    "quixiai/dolphin-distill",
)
"""Lower-cased substrings of the ``docs/PLAN.md`` §5 exclusion list (Claude-derived)."""

_ROLE_ALIASES = {
    "human": "user",
    "gpt": "assistant",
    "bot": "assistant",
    "assistant": "assistant",
    "user": "user",
    "system": "system",
}


def check_not_excluded(repo_id: str) -> None:
    """Refuse datasets on the plan's Claude-derived exclusion list."""
    low = repo_id.lower()
    for bad in EXCLUDED:
        if bad in low:
            raise SystemExit(
                f"refusing {repo_id!r}: it matches the docs/PLAN.md §5 exclusion list ({bad}). "
                "No Claude-derived data may enter training (docs/ARCHITECTURE.md §1.1)."
            )


def normalise(messages) -> list[dict[str, str]]:
    """Coerce one row's messages into ``[{'role': ..., 'content': ...}]``."""
    out: list[dict[str, str]] = []
    for m in messages:
        role = str(m.get("role") or m.get("from") or "").strip().lower()
        content = m.get("content")
        if content is None:
            content = m.get("value")
        role = _ROLE_ALIASES.get(role)
        if role is None or content is None:
            return []
        out.append({"role": role, "content": str(content)})
    return out


def rendered(rows, tokenizer, max_seq: int, min_assistant_tokens: int = 1):
    """Yield ``(ids, mask)`` for rows that survive rendering and truncation."""
    for messages in rows:
        msgs = normalise(messages)
        if not msgs or not any(m["role"] == "assistant" for m in msgs):
            continue
        ids, mask = render(msgs, tokenizer, add_generation_prompt=False, max_tokens=max_seq + 1)
        if sum(mask) < min_assistant_tokens:
            continue
        # A conversation truncated mid-assistant-turn never shows the model how to stop;
        # trim back to the last complete assistant span instead of teaching a dangling turn.
        if ids[-1] != ASSISTANT_END and ASSISTANT_END in ids:
            cut = len(ids) - 1 - ids[::-1].index(ASSISTANT_END)
            ids, mask = ids[: cut + 1], mask[: cut + 1]
        if len(ids) < 4 or sum(mask) < min_assistant_tokens:
            continue
        yield ids, mask


def load_rows(dataset: str, subset: str, split: str, n: int, messages_key: str, streaming: bool):
    """Stream ``n`` rows' message lists from a Hugging Face dataset."""
    from datasets import load_dataset

    kwargs = {"split": split, "streaming": streaming}
    ds = load_dataset(dataset, subset, **kwargs) if subset else load_dataset(dataset, **kwargs)
    taken = 0
    for row in ds:
        msgs = row.get(messages_key)
        if not msgs:
            continue
        yield msgs
        taken += 1
        if n and taken >= n:
            return


def print_stats(path: str | Path) -> int:
    """Print the counts for an existing split directory tree."""
    root = Path(path)
    for split in ("train", "valid"):
        d = root / split
        if not (d / "meta.json").exists():
            continue
        ds = ConversationDataset(d)
        print(f"{split:6s} {len(ds):>8,} conversations  {ds.n_tokens:>12,} tokens  "
              f"{ds.n_supervised:>12,} assistant tokens "
              f"({100 * ds.n_supervised / max(1, ds.n_tokens):.1f}%)")
        print(f"       meta: {json.dumps(ds.meta)}")
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", default=DEFAULT_DATASET)
    p.add_argument("--subset", default="", help="HF dataset config name")
    p.add_argument("--split", default="train")
    p.add_argument("--messages-key", default="messages")
    p.add_argument("--n", type=int, default=1000, help="conversations to take (0 = all)")
    p.add_argument("--val-frac", type=float, default=0.05, help="fraction held out for validation")
    p.add_argument("--max-seq", type=int, default=1024, help="truncate conversations to N tokens")
    p.add_argument("--out", default=DEFAULT_OUT)
    p.add_argument("--no-streaming", action="store_true", help="download the split instead of streaming")
    p.add_argument("--stats", metavar="DIR", default=None, help="print an existing split's stats and exit")
    args = p.parse_args(argv)

    if args.stats:
        return print_stats(args.stats)

    check_not_excluded(args.dataset)
    tok = GPT2Tokenizer()
    rows = load_rows(args.dataset, args.subset, args.split, args.n, args.messages_key,
                     not args.no_streaming)
    convs = list(rendered(rows, tok, args.max_seq))
    if not convs:
        raise SystemExit(f"no usable conversations in {args.dataset}:{args.split}")

    n_val = max(1, int(len(convs) * args.val_frac)) if args.val_frac > 0 else 0
    n_val = min(n_val, max(0, len(convs) - 1))
    meta = {
        "dataset": args.dataset,
        "subset": args.subset,
        "split": args.split,
        "max_seq": args.max_seq,
        "chat_format": "r52/chat_template.py",
        "license_check": (
            "apache-2.0; teacher = Llama-3.1-405B-Instruct (Magpie) + public SFT sets; "
            "no Claude lineage; not on the docs/PLAN.md §5 exclusion list (checked 2026-09-13)"
        ),
    }
    out = Path(args.out)
    write_split(out / "train", convs[n_val:], {**meta, "split_role": "train"})
    if n_val:
        write_split(out / "valid", convs[:n_val], {**meta, "split_role": "valid"})
    print(f"wrote {out}")
    return print_stats(out)


if __name__ == "__main__":
    rc = main()
    # PyArrow's streaming parquet reader deadlocks in `ThreadPool::Shutdown` at interpreter
    # exit when a `datasets` streaming iterator is abandoned mid-file (measured 2026-09-13:
    # the work finishes, the files are written, and the process then hangs forever at 0% CPU
    # inside `arrow::internal::ThreadPool::Shutdown`).  Everything this script owns is
    # flushed by now, so leave without running interpreter teardown.
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(rc)
