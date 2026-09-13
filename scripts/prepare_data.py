#!/usr/bin/env python
# Copyright 2026 The road-to-52 authors.
# SPDX-License-Identifier: Apache-2.0
"""Download FineWeb-10B GPT-2 shards into ``data/fineweb10B-gpt2/``.

The dataset is ``kjj0/fineweb10B-gpt2`` (public, no token required): 103 training shards
of ~100M tokens each (~200 MB per shard on disk) plus one validation shard, in llm.c's
``.bin`` format.  ``data/`` is a symlink to the external disk; files land there.

Examples
--------
    python scripts/prepare_data.py --train-shards 1          # val + shard 1 (~400 MB)
    python scripts/prepare_data.py --train-shards 8 --start 1
    python scripts/prepare_data.py --list                    # show what is on disk
"""

from __future__ import annotations

import argparse
import contextlib
import os
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

REPO_ID = "kjj0/fineweb10B-gpt2"
DEFAULT_DIR = "data/fineweb10B-gpt2"
TOKENS_PER_SHARD = 100_000_000  # nominal; the header is authoritative


def _fetch(repo_id: str, filename: str, out_dir: Path) -> Path:
    """Download one file from the HF dataset repo into ``out_dir`` (idempotent)."""
    from huggingface_hub import hf_hub_download

    dest = out_dir / filename
    if dest.exists():
        return dest
    cached = hf_hub_download(repo_id=repo_id, filename=filename, repo_type="dataset")
    out_dir.mkdir(parents=True, exist_ok=True)
    # Copy (not symlink) so the bytes live on the external disk, then drop the HF cache copy.
    tmp = dest.with_suffix(dest.suffix + ".partial")
    shutil.copyfile(cached, tmp)
    tmp.replace(dest)
    with contextlib.suppress(OSError):
        os.remove(cached)
    return dest


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", default=DEFAULT_DIR, help="output directory (default: %(default)s)")
    p.add_argument("--repo-id", default=REPO_ID)
    p.add_argument("--train-shards", type=int, default=1, help="how many train shards to fetch")
    p.add_argument("--start", type=int, default=1, help="first train shard index (files start at 000001)")
    p.add_argument("--no-val", action="store_true", help="skip the validation shard")
    p.add_argument("--list", action="store_true", help="list local shards and exit")
    args = p.parse_args(argv)

    out_dir = Path(args.out)

    if args.list:
        from r52.data import read_shard_header

        if not out_dir.exists():
            print(f"{out_dir} does not exist")
            return 0
        total = 0
        for f in sorted(out_dir.glob("*.bin")):
            n = read_shard_header(f)
            total += n
            print(f"{f.name:32s} {n:>12,} tokens  {f.stat().st_size / 1e6:8.1f} MB")
        print(f"{'TOTAL':32s} {total:>12,} tokens")
        return 0

    from huggingface_hub import list_repo_files

    available = set(list_repo_files(args.repo_id, repo_type="dataset"))
    wanted: list[str] = []
    if not args.no_val:
        wanted.append("fineweb_val_000000.bin")
    for i in range(args.start, args.start + args.train_shards):
        wanted.append(f"fineweb_train_{i:06d}.bin")

    missing = [w for w in wanted if w not in available]
    if missing:
        print(f"not in {args.repo_id}: {missing}", file=sys.stderr)
        return 1

    for name in wanted:
        dest = out_dir / name
        if dest.exists():
            print(f"have  {name}")
            continue
        print(f"fetch {name} ...", flush=True)
        path = _fetch(args.repo_id, name, out_dir)
        print(f"  -> {path} ({path.stat().st_size / 1e6:.1f} MB)")

    from r52.data import read_shard_header

    total = sum(read_shard_header(out_dir / n) for n in wanted)
    print(f"ready: {len(wanted)} files, {total:,} tokens in {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
