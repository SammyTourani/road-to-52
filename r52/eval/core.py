# Copyright 2026 The road-to-52 authors.
# SPDX-License-Identifier: Apache-2.0
#
# Port of karpathy/nanochat's CORE evaluation -- `nanochat/core_eval.py` and the
# `evaluate_core` half of `scripts/base_eval.py` -- from PyTorch to MLX.
#   nanochat: https://github.com/karpathy/nanochat  (MIT License, Copyright (c) 2025 Andrej Karpathy)
# The prompt templates, the few-shot sampling, the common-prefix/suffix logic, the mean-loss
# decision rule, the 1337 shuffle and the random-baseline centering are reproduced exactly so
# that our CORE score sits on the same scale as nanochat's `dev/LEADERBOARD.md`.  The Jinja2
# templates are re-expressed as string concatenation (identical output, one less dependency).
# CORE itself is from the DCLM paper, https://arxiv.org/abs/2406.11794.
"""DCLM CORE: 22 in-context-learning tasks, centered on their random baselines.

    python -m r52.eval.core --model models/gpt2-mlx
    python -m r52.eval.core --model runs/b1/ckpt/best --limit 50
    python -m r52.eval.core --model models/gpt2-mlx --tasks copa,winograd

**The metric.**  Each task is scored as plain accuracy, then centered on its random baseline::

    centered = (accuracy - 0.01 * random_baseline) / (1 - 0.01 * random_baseline)

and CORE is the unweighted mean of the 22 centered scores.  Centering means a model that
guesses scores 0, so the number is comparable across tasks with 2, 4 or 5 choices.

**Three task types**, all scored by teacher-forced loss, never by sampling:

``multiple_choice``
    One prompt per choice, sharing a common prefix; the answer span is everything after the
    common prefix; the choice with the lowest **mean** per-token loss wins.
``schema``
    One prompt per context option, sharing a common *suffix* (the continuation); the scored
    span is that common suffix; lowest mean loss wins.
``language_modeling``
    The prompt is rendered twice, with and without the continuation; the model must predict
    every continuation token **greedily** (exact match on argmax) to score 1.

**The eval bundle.**  ``scripts/base_eval.py`` downloads it from
``https://karpathy-public.s3.us-west-2.amazonaws.com/eval_bundle.zip`` (26 MB zipped, 161 MB
extracted); this module caches it at ``data/eval_bundle/`` and reads ``core.yaml`` (the 22
tasks), ``eval_data/*/*.jsonl`` (the items) and ``eval_meta_data.csv`` (the random baselines).
The bundle also ships nanochat's own reference CSVs -- ``openai-community-gpt2.csv`` gives
**CORE 0.113891** for GPT-2 124M and ``openai-community-gpt2-xl.csv`` **0.256525** for GPT-2
XL, the latter being the "time to GPT-2" threshold in ``dev/LEADERBOARD.md`` -- so this port
can be checked against the implementation it came from, task by task.

**BOS.**  nanochat prepends its own ``<|bos|>`` to every sequence and pads with it.  The
GPT-2 tokenizer's equivalent is ``<|endoftext|>`` (50256), which is what GPT-2 itself was
trained to see at a document boundary; this module uses it for both.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import shutil
import tempfile
import time
import zipfile
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mlx.core as mx
import numpy as np
import yaml

from .lm import DEFAULT_MEMORY_LIMIT_GIB, LM, configure_runtime, pad_stack

__all__ = [
    "BENCHMARK",
    "EVAL_BUNDLE_URL",
    "Task",
    "centered_score",
    "ensure_bundle",
    "evaluate_core",
    "evaluate_task",
    "find_common_length",
    "load_tasks",
    "main",
    "render_prompts",
]

BENCHMARK = "dclm-core"
EVAL_BUNDLE_URL = "https://karpathy-public.s3.us-west-2.amazonaws.com/eval_bundle.zip"
"""Public URL in nanochat's ``scripts/base_eval.py`` (``EVAL_BUNDLE_URL``)."""

DEFAULT_BUNDLE_DIR = "data/eval_bundle"
SHUFFLE_SEED = 1337
"""nanochat shuffles each task's items with ``random.Random(1337)`` before ``--limit``."""
FEWSHOT_SEED = 1234
"""Few-shot examples for item ``idx`` come from ``random.Random(1234 + idx)``."""


@dataclass(frozen=True)
class Task:
    """One CORE task as declared in the bundle's ``core.yaml``."""

    label: str
    dataset_uri: str
    task_type: str
    num_fewshot: int
    continuation_delimiter: str = " "

    @property
    def category(self) -> str:
        return self.dataset_uri.split("/", 1)[0]


# --------------------------------------------------------------------------------------
# Bundle
# --------------------------------------------------------------------------------------


def ensure_bundle(bundle_dir: str | Path = DEFAULT_BUNDLE_DIR) -> Path:
    """Return the eval-bundle directory, downloading and unzipping it on first use."""
    d = Path(bundle_dir)
    if (d / "core.yaml").is_file() and (d / "eval_data").is_dir():
        return d
    d.parent.mkdir(parents=True, exist_ok=True)
    print(f"[core] downloading {EVAL_BUNDLE_URL} -> {d}", flush=True)
    import urllib.request

    with tempfile.TemporaryDirectory() as tmp:
        zpath = Path(tmp) / "eval_bundle.zip"
        urllib.request.urlretrieve(EVAL_BUNDLE_URL, zpath)  # constant https URL
        with zipfile.ZipFile(zpath) as zf:
            zf.extractall(tmp)
        src = Path(tmp) / "eval_bundle"
        if not src.is_dir():  # pragma: no cover - archive layout guard
            raise RuntimeError(f"{EVAL_BUNDLE_URL} did not contain an eval_bundle/ directory")
        if d.exists():
            shutil.rmtree(d)
        shutil.move(str(src), str(d))
    return d


def load_tasks(bundle_dir: str | Path = DEFAULT_BUNDLE_DIR) -> list[Task]:
    """The 22 CORE tasks from ``core.yaml``, in the bundle's order."""
    d = ensure_bundle(bundle_dir)
    config = yaml.safe_load((d / "core.yaml").read_text(encoding="utf-8"))
    return [
        Task(
            label=t["label"],
            dataset_uri=t["dataset_uri"],
            task_type=t["icl_task_type"],
            num_fewshot=int(t["num_fewshot"][0]),
            continuation_delimiter=t.get("continuation_delimiter", " "),
        )
        for t in config["icl_tasks"]
    ]


def load_random_baselines(bundle_dir: str | Path = DEFAULT_BUNDLE_DIR) -> dict[str, float]:
    """``Eval Task -> Random baseline`` (**percent**, e.g. 25.0) from ``eval_meta_data.csv``."""
    d = ensure_bundle(bundle_dir)
    out: dict[str, float] = {}
    with (d / "eval_meta_data.csv").open("r", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            out[row["Eval Task"]] = float(row["Random baseline"])
    return out


def load_task_data(task: Task, bundle_dir: str | Path = DEFAULT_BUNDLE_DIR, limit: int = 0) -> list[dict]:
    """Items for one task: read jsonl, shuffle with seed 1337, then take the first ``limit``."""
    d = ensure_bundle(bundle_dir)
    path = d / "eval_data" / task.dataset_uri
    with path.open("r", encoding="utf-8") as fh:
        data = [json.loads(line.strip()) for line in fh if line.strip()]
    random.Random(SHUFFLE_SEED).shuffle(data)
    return data[:limit] if limit and limit > 0 else data


def reference_scores(model_csv: str, bundle_dir: str | Path = DEFAULT_BUNDLE_DIR) -> dict[str, float]:
    """nanochat's own per-task accuracies from a bundle CSV, e.g. ``openai-community-gpt2``."""
    d = ensure_bundle(bundle_dir)
    out: dict[str, float] = {}
    for line in (d / f"{model_csv}.csv").read_text(encoding="utf-8").splitlines()[1:]:
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 3:
            continue
        out[parts[0]] = float(parts[2]) if parts[0] == "CORE" else float(parts[1])
    return out


# --------------------------------------------------------------------------------------
# Prompt rendering (nanochat's Jinja2 templates, expanded)
# --------------------------------------------------------------------------------------


def render_prompts(item: dict, task: Task, fewshot: Sequence[dict]) -> list[str]:
    """Render the prompts to score for one item.

    ``multiple_choice`` returns one prompt per choice, ``schema`` one per context option and
    ``language_modeling`` exactly two (without the continuation, then with it).
    """
    d = task.continuation_delimiter
    if task.task_type == "multiple_choice":
        shots = "".join(f"{e['query']}{d}{e['choices'][e['gold']]}\n\n" for e in fewshot)
        return [f"{shots}{item['query']}{d}{c}" for c in item["choices"]]
    if task.task_type == "schema":
        shots = "".join(f"{e['context_options'][e['gold']]}{d}{e['continuation']}\n\n" for e in fewshot)
        return [f"{shots}{c}{d}{item['continuation']}" for c in item["context_options"]]
    if task.task_type == "language_modeling":
        shots = "".join(f"{e['context'].strip()}{d}{e['continuation']}\n\n" for e in fewshot)
        without = f"{shots}{item['context'].strip()}{d}"
        # nanochat strips the continuation-free prompt so a delimiter ending in whitespace
        # cannot get absorbed into the first continuation token; the prefix property below
        # depends on it.
        return [without.strip(), f"{without}{item['continuation']}"]
    raise ValueError(f"unsupported task type: {task.task_type}")


def find_common_length(sequences: Sequence[Sequence[int]], direction: str = "left") -> int:
    """Length of the common prefix (``left``) or suffix (``right``) of token sequences."""
    min_len = min(len(s) for s in sequences)
    indices = range(min_len) if direction == "left" else range(-1, -min_len - 1, -1)
    for i, idx in enumerate(indices):
        token = sequences[0][idx]
        if not all(s[idx] == token for s in sequences):
            return i
    return min_len


@dataclass
class Item:
    """One rendered item: token rows plus the ``[start, end)`` span to score in each row."""

    rows: list[list[int]]
    starts: list[int]
    ends: list[int]
    gold: int
    task_type: str
    lm_prefix_fallback: bool = False


def build_item(item: dict, task: Task, fewshot: Sequence[dict], tokenizer, bos: int) -> Item:
    """Tokenize one item's prompts and locate the span each score is taken over."""
    prompts = render_prompts(item, task, fewshot)
    rows = [[bos, *tokenizer.encode(p, allowed_special=set())] for p in prompts]
    fallback = False
    if task.task_type == "multiple_choice":
        start = find_common_length(rows, "left")
        starts = [start] * len(rows)
        ends = [len(r) for r in rows]
    elif task.task_type == "schema":
        suffix = find_common_length(rows, "right")
        ends = [len(r) for r in rows]
        starts = [e - suffix for e in ends]
    else:
        without, with_ = rows
        start, end = len(without), len(with_)
        if start >= end or without != with_[:start]:
            # nanochat asserts the prefix property here.  GPT-2's BPE can break it for a rare
            # item (a continuation that re-tokenizes across the delimiter boundary); rather
            # than abort a multi-hour run we fall back to the longest common prefix and count
            # the occurrences, which are reported alongside the score.
            start = find_common_length(rows, "left")
            fallback = True
        rows, starts, ends = [with_], [start], [len(with_)]
    return Item(rows, starts, ends, int(item.get("gold", 0)), task.task_type, fallback)


def truncate(item: Item, max_context: int) -> tuple[Item, int, int]:
    """Crop rows to ``max_context`` tokens, keeping the tail and shifting the spans.

    Returns ``(item, n_truncated_rows, n_clamped_spans)``.  nanochat asserts the shifted start
    stays non-negative; with a 1,024-token GPT-2 context and a 10-shot prompt that assertion
    can fail (the crop eats the start of the scored span), so the start is clamped to 1 -- the
    earliest position that has an autoregressive target -- and the clamp is counted and
    reported rather than aborting the run.
    """
    if max_context <= 0 or all(len(r) <= max_context for r in item.rows):
        return item, 0, 0
    rows, starts, ends = [], [], []
    n_trunc = n_clamp = 0
    for r, s, e in zip(item.rows, item.starts, item.ends, strict=True):
        if len(r) > max_context:
            drop = len(r) - max_context
            rows.append(r[-max_context:])
            if s - drop < 1:
                n_clamp += 1
            starts.append(max(1, s - drop))
            ends.append(max(1, e - drop))
            n_trunc += 1
        else:
            rows.append(r)
            starts.append(s)
            ends.append(e)
    return Item(rows, starts, ends, item.gold, item.task_type, item.lm_prefix_fallback), n_trunc, n_clamp


# --------------------------------------------------------------------------------------
# Evaluation
# --------------------------------------------------------------------------------------


def _batches(items: Iterable[Item], budget: int) -> Iterator[list[Item]]:
    """Group whole items into forward batches under a ``rows * width`` position budget.

    Grouping across items is exactly equivalent to nanochat's one-item-at-a-time forward:
    attention is causal and padding is on the right, so a row's scores do not depend on what
    else is in the batch or on how far it is padded.
    """
    batch: list[Item] = []
    rows = width = 0
    for it in items:
        w = max(width, max(len(r) for r in it.rows))
        if batch and (rows + len(it.rows)) * w > budget:
            yield batch
            batch, rows = [], 0
            w = max(len(r) for r in it.rows)
        batch.append(it)
        rows += len(it.rows)
        width = w
    if batch:
        yield batch


def evaluate_task(
    lm: LM,
    task: Task,
    data: list[dict],
    *,
    bos: int | None = None,
    progress_every: int = 0,
) -> dict[str, Any]:
    """Accuracy of one task. ``data`` must already be shuffled/limited by :func:`load_task_data`.

    Items are tokenized **lazily**, one forward batch at a time.  Materialising all of them up
    front costs ~800 MB of Python ints on the 10-shot HellaSwag task (10,042 items x 4 rows x
    ~550 tokens), which this machine does not have to spare while a pretraining run is
    training.
    """
    bos = lm.tokenizer.eot if bos is None else bos
    counts = {"truncated": 0, "clamped": 0, "fallbacks": 0}

    def items() -> Iterator[Item]:
        for idx, raw in enumerate(data):
            fewshot: list[dict] = []
            if task.num_fewshot > 0:
                rng = random.Random(FEWSHOT_SEED + idx)
                available = [i for i in range(len(data)) if i != idx]
                # nanochat samples exactly `num_fewshot`; with a small `--limit` there may not
                # be that many other items, and `random.sample` would raise.  Smoke runs matter
                # more than bit-fidelity below the shot count; full runs are unaffected.
                n_shot = min(task.num_fewshot, len(available))
                fewshot = [data[i] for i in rng.sample(available, n_shot)]
            item, n_t, n_c = truncate(
                build_item(raw, task, fewshot, lm.tokenizer, bos), lm.max_context
            )
            counts["truncated"] += n_t
            counts["clamped"] += n_c
            counts["fallbacks"] += int(item.lm_prefix_fallback)
            yield item

    n_items = len(data)
    correct = 0
    done = 0
    t0 = time.time()
    for batch in _batches(items(), lm.max_positions_per_forward):
        flat = [r for it in batch for r in it.rows]
        tokens = pad_stack(flat, pad_id=bos)
        nll, pred = lm.scores(tokens)
        if task.task_type == "language_modeling":
            hit = (pred[:, :-1] == tokens[:, 1:]).astype(mx.int32)
            mx.eval(hit)
            arr = np.asarray(hit)
        else:
            mx.eval(nll)
            arr = np.asarray(nll, dtype=np.float64)
        off = 0
        for it in batch:
            k = len(it.rows)
            if task.task_type == "language_modeling":
                s, e = it.starts[0], it.ends[0]
                correct += int(bool(arr[off, s - 1 : e - 1].all()))
            else:
                means = [
                    float(np.mean(arr[off + i, s - 1 : e - 1])) if e > s >= 1 else float("inf")
                    for i, (s, e) in enumerate(zip(it.starts, it.ends, strict=True))
                ]
                correct += int(means.index(min(means)) == it.gold)
            off += k
            done += 1
        if progress_every and done % progress_every < len(batch):
            el = time.time() - t0
            print(
                f"    {task.label}: {done}/{n_items} acc {correct / done:.4f} "
                f"| eta {el * (n_items / max(1, done) - 1) / 60:.1f} min",
                flush=True,
            )

    return {
        "label": task.label,
        "accuracy": correct / max(1, n_items),
        "n_correct": correct,
        "n_examples": n_items,
        "num_fewshot": task.num_fewshot,
        "task_type": task.task_type,
        "n_truncated_rows": counts["truncated"],
        "n_clamped_spans": counts["clamped"],
        "n_lm_prefix_fallbacks": counts["fallbacks"],
        "wall_clock_s": time.time() - t0,
    }


def centered_score(accuracy: float, random_baseline_pct: float) -> float:
    """nanochat's centering: ``(acc - 0.01*baseline) / (1 - 0.01*baseline)``."""
    base = 0.01 * random_baseline_pct
    return (accuracy - base) / (1.0 - base)


def evaluate_core(
    lm: LM,
    *,
    bundle_dir: str | Path = DEFAULT_BUNDLE_DIR,
    limit: int = 0,
    only: Sequence[str] | None = None,
    verbose: bool = True,
    progress_every: int = 0,
    partial_path: str | Path | None = None,
) -> dict[str, Any]:
    """Run CORE. Returns per-task accuracies, centered scores and the CORE metric.

    ``partial_path`` appends one JSON object per finished task to a ``.jsonl`` file.  A full
    22-task run on this machine is hours long while a pretraining job owns the GPU, so the
    per-task rows are written as they land rather than only at the end.
    """
    tasks = load_tasks(bundle_dir)
    if only:
        wanted = {t.strip() for t in only if t.strip()}
        unknown = wanted - {t.label for t in tasks}
        if unknown:
            raise ValueError(f"unknown task(s): {sorted(unknown)}")
        tasks = [t for t in tasks if t.label in wanted]
    baselines = load_random_baselines(bundle_dir)

    results: dict[str, float] = {}
    centered: dict[str, float] = {}
    per_task: list[dict[str, Any]] = []
    t0 = time.time()
    for task in tasks:
        data = load_task_data(task, bundle_dir, limit)
        row = evaluate_task(lm, task, data, progress_every=progress_every)
        row["random_baseline_pct"] = baselines[task.label]
        row["centered"] = centered_score(row["accuracy"], baselines[task.label])
        results[task.label] = row["accuracy"]
        centered[task.label] = row["centered"]
        per_task.append(row)
        if partial_path is not None:
            p = Path(partial_path)
            p.parent.mkdir(parents=True, exist_ok=True)
            with p.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(row) + "\n")
        if verbose:
            print(
                f"  {task.label:<34} {task.num_fewshot:>2}-shot {task.task_type:<18} "
                f"n={row['n_examples']:<6} acc {row['accuracy']:.4f} "
                f"centered {row['centered']:>8.4f} ({row['wall_clock_s']:.1f} s)",
                flush=True,
            )
    core = sum(centered.values()) / max(1, len(centered))
    return {
        "core_metric": core,
        "results": results,
        "centered_results": centered,
        "per_task": per_task,
        "n_tasks": len(tasks),
        "limit": limit or None,
        "wall_clock_s": time.time() - t0,
    }


# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="r52.eval.core",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--model", required=True,
                   help="r52 checkpoint directory or mlx-lm model directory")
    p.add_argument("--limit", type=int, default=0,
                   help="max examples per task after the 1337 shuffle (0 = all; nanochat's "
                        "--max-per-task)")
    p.add_argument("--tasks", default=None, help="comma-separated task labels (default: all 22)")
    p.add_argument("--bundle-dir", default=DEFAULT_BUNDLE_DIR)
    p.add_argument("--max-positions", type=int, default=4096,
                   help="rows*tokens per forward, bounds fp32 logits memory (default 4096)")
    p.add_argument("--memory-limit-gib", type=float, default=DEFAULT_MEMORY_LIMIT_GIB)
    p.add_argument("--progress-every", type=int, default=0,
                   help="print within-task progress every N examples (0 = off)")
    p.add_argument("--report", action="store_true")
    p.add_argument("--run", default=None)
    p.add_argument("--results-dir", default=None,
                   help="write results here instead of results/ (also moves the markdown row "
                        "to <dir>/RESULTS.md, so smoke runs leave docs/RESULTS.md alone)")
    p.add_argument("--model-id", default=None, help="bar.yaml model id (e.g. gpt2-124m)")
    p.add_argument("--csv", default=None, help="also write nanochat's per-task CSV here")
    p.add_argument("--partial", default=None,
                   help="append each finished task as a JSON line here (crash insurance on a "
                        "multi-hour run)")
    p.add_argument("--quiet", action="store_true")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    configure_runtime(args.memory_limit_gib)

    lm = LM.load(args.model, max_positions_per_forward=args.max_positions)
    only = args.tasks.split(",") if args.tasks else None
    if not args.quiet:
        print(f"[core] {lm.describe()}", flush=True)

    t0 = time.time()
    out = evaluate_core(
        lm,
        bundle_dir=args.bundle_dir,
        limit=args.limit,
        only=only,
        verbose=not args.quiet,
        progress_every=args.progress_every,
        partial_path=args.partial,
    )
    wall = time.time() - t0
    print(
        f"CORE {out['core_metric']:.6f} over {out['n_tasks']} tasks "
        f"| {wall:.1f} s | peak memory {mx.get_peak_memory() / 2**30:.2f} GiB",
        flush=True,
    )

    if args.csv:
        _write_csv(Path(args.csv), out)
        print(f"wrote {args.csv}", flush=True)

    if args.report:
        from .report import make_result, write_result

        run = args.run or lm.name
        cmd = f"python -m r52.eval.core --model {args.model}" + (
            f" --limit {args.limit}" if args.limit else ""
        ) + (f" --tasks {args.tasks}" if args.tasks else "")
        n_examples = sum(t["n_examples"] for t in out["per_task"])
        result = make_result(
            model=args.model_id or lm.name,
            benchmark=BENCHMARK,
            value=out["core_metric"],
            unit="CORE",
            conditions={
                "tokenizer": "gpt2 (tiktoken)",
                "block_size": lm.max_context,
                "limit": args.limit or None,
                "n_examples": n_examples,
                "few_shot": "per task (0/3/10), nanochat core.yaml",
                "n_tasks": out["n_tasks"],
                "harness": "r52.eval.core, port of nanochat core_eval.py + base_eval.py",
                "command": cmd,
            },
            wall_clock_s=wall,
            metrics={
                "core_metric": out["core_metric"],
                "results": out["results"],
                "centered_results": out["centered_results"],
                "per_task": out["per_task"],
                "peak_memory_gib": mx.get_peak_memory() / 2**30,
                "model": lm.describe(),
            },
        )
        path = write_result(
            result, run, "core",
            results_root=Path(args.results_dir) if args.results_dir else None,
        )
        print(f"wrote {path}", flush=True)
    return 0


def _write_csv(path: Path, out: dict[str, Any]) -> None:
    """nanochat's ``base_eval`` CSV layout, so the two can be diffed directly."""
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"{'Task':<35}, {'Accuracy':<10}, {'Centered':<10}"]
    for label, acc in out["results"].items():
        lines.append(f"{label:<35}, {acc:<10.6f}, {out['centered_results'][label]:<10.6f}")
    lines.append(f"{'CORE':<35}, {'':<10}, {out['core_metric']:<10.6f}")
    path.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
