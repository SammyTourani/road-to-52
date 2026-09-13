# Copyright 2026 The road-to-52 authors.
# SPDX-License-Identifier: Apache-2.0
"""pass@1 / pass@k probe on reasoning-gym -- run this **before** any RL.

    python -m r52.posttrain.passk models/nano-mlx --task chain_sum -n 20 -k 4
    python -m r52.posttrain.passk models/nano-mlx --task gsm_symbolic -n 64 -k 16 \
        --temperature 1.0 --max-tokens 128 --out results/passk-nano.json

``docs/PLAN.md`` §3.1 item 5: *measure base-model pass@k before doing any RL.  High pass@128
means RL only sharpens what the model already samples; low pass@k means RL can actually
expand the boundary.*  That is a decision input, so the probe writes a JSON record with the
exact conditions (model, task, n, k, temperature, seed, wall-clock) alongside the numbers,
per ``docs/ARCHITECTURE.md`` §1.3's "no number without a reproduction path".

Estimator: the unbiased pass@k of Chen et al. 2021 (*Evaluating Large Language Models
Trained on Code*, arXiv 2107.03374 §2.1).  With ``n`` samples per problem of which ``c`` are
correct, ``pass@k = 1 - C(n-c, k) / C(n, k)``.  Sampling ``k`` and reporting the fraction of
problems with at least one success is the high-variance special case ``n == k``; we use the
estimator so ``--samples`` may exceed ``k``.

Units: ``pass@k`` is a fraction in ``[0, 1]``; ``mean_reward`` is the mean verifier score;
``gen_tokens`` counts sampled tokens.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any

import mlx.core as mx

from ..chat_template import render_prompt
from .rgym import SYSTEM_PROMPT, RGymTask, format_score
from .sampler import Sampler, load_exported_model

__all__ = ["main", "pass_at_k", "probe"]


def pass_at_k(n: int, c: int, k: int) -> float:
    """Unbiased pass@k for ``c`` correct out of ``n`` samples (Chen et al. 2021, §2.1).

    ``1 - C(n-k, k) / C(n, k)`` computed as a product to stay stable for large ``n``.
    Returns 1.0 when ``n - c < k`` (some correct sample is guaranteed to be drawn).
    """
    if k <= 0 or n <= 0:
        return 0.0
    k = min(k, n)
    if n - c < k:
        return 1.0
    return 1.0 - math.prod((n - c - i) / (n - i) for i in range(k))


def probe(
    model_path: str,
    task: str = "chain_sum",
    n_prompts: int = 20,
    k: int = 4,
    samples: int | None = None,
    max_tokens: int = 64,
    temperature: float = 1.0,
    top_p: float = 1.0,
    top_k: int = 0,
    seed: int = 42,
    task_size: int = 2000,
    system_prompt: str = SYSTEM_PROMPT,
    max_prompt_tokens: int = 256,
    verbose: bool = False,
) -> dict[str, Any]:
    """Sample ``samples`` (default ``k``) completions per prompt and score them.

    Returns a record with ``pass@1``, ``pass@k``, ``mean_reward``, ``format_rate`` and the
    full reproduction conditions.
    """
    samples = int(samples or k)
    if samples < k:
        raise ValueError(f"samples ({samples}) must be >= k ({k})")
    mx.random.seed(seed)
    t0 = time.time()

    model, tokenizer = load_exported_model(model_path)
    sampler = Sampler(model, tokenizer)
    pool = RGymTask(task, size=task_size, seed=seed)
    items = pool.items(n_prompts)

    per_prompt: list[dict[str, Any]] = []
    total_tokens = 0
    n_format = 0
    reward_sum = 0.0
    for i, item in enumerate(items):
        prompt = render_prompt(item.messages(system_prompt), max_tokens=max_prompt_tokens)
        comps = sampler.sample_group(prompt, samples, max_tokens, temperature, top_p, top_k,
                                     prompt_index=i)
        scores = [pool.score(c.text, item) for c in comps]
        correct = sum(1 for s in scores if s >= 1.0)
        total_tokens += sum(len(c) for c in comps)
        n_format += sum(int(format_score(c.text)) for c in comps)
        reward_sum += sum(scores)
        per_prompt.append({
            "index": i,
            "task": item.task,
            "answer": item.answer,
            "correct": correct,
            "samples": samples,
            "mean_score": round(sum(scores) / samples, 4),
            "pass@1": round(pass_at_k(samples, correct, 1), 4),
            f"pass@{k}": round(pass_at_k(samples, correct, k), 4),
        })
        if verbose:
            print(f"  [{i + 1}/{n_prompts}] {correct}/{samples} correct | "
                  f"answer={item.answer!r} | sample={comps[0].text[:70]!r}", flush=True)

    n = max(1, len(per_prompt))
    total = samples * n
    record = {
        "stage": "passk",
        "model": str(model_path),
        "task": task,
        "n_prompts": len(per_prompt),
        "k": k,
        "samples_per_prompt": samples,
        "pass@1": round(sum(p["pass@1"] for p in per_prompt) / n, 4),
        f"pass@{k}": round(sum(p[f"pass@{k}"] for p in per_prompt) / n, 4),
        "mean_reward": round(reward_sum / max(1, total), 4),
        "format_rate": round(n_format / max(1, total), 4),
        "gen_tokens": total_tokens,
        "mean_gen_tokens": round(total_tokens / max(1, total), 1),
        "conditions": {
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "max_tokens": max_tokens,
            "max_prompt_tokens": max_prompt_tokens,
            "seed": seed,
            "task_size": task_size,
            "system_prompt": system_prompt,
            "estimator": "Chen et al. 2021 arXiv:2107.03374 §2.1",
        },
        "wall_clock_s": round(time.time() - t0, 2),
        "per_prompt": per_prompt,
    }
    return record


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="r52.posttrain.passk", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("model", help="exported mlx-lm model directory")
    p.add_argument("--task", default="chain_sum", help="reasoning-gym dataset name")
    p.add_argument("-n", "--n-prompts", type=int, default=20)
    p.add_argument("-k", type=int, default=4, help="k in pass@k")
    p.add_argument("--samples", type=int, default=None,
                   help="completions per prompt (default: k); >k gives a lower-variance estimate")
    p.add_argument("--max-tokens", type=int, default=64, help="completion length cap (tokens)")
    p.add_argument("--max-prompt-tokens", type=int, default=256)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top-p", type=float, default=1.0)
    p.add_argument("--top-k", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--task-size", type=int, default=2000)
    p.add_argument("--no-system-prompt", action="store_true")
    p.add_argument("--memory-limit-gb", type=float, default=2.0,
                   help="mx.set_memory_limit for this process (the machine is shared)")
    p.add_argument("--out", default=None, help="write the JSON record here")
    p.add_argument("--list-tasks", action="store_true", help="print reasoning-gym task names and exit")
    p.add_argument("-v", "--verbose", action="store_true")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.list_tasks:
        print("\n".join(RGymTask.available()))
        return 0
    mx.set_memory_limit(int(args.memory_limit_gb * (1 << 30)))
    mx.set_cache_limit(1 << 29)
    rec = probe(
        args.model, args.task, args.n_prompts, args.k, args.samples, args.max_tokens,
        args.temperature, args.top_p, args.top_k, args.seed, args.task_size,
        "" if args.no_system_prompt else SYSTEM_PROMPT, args.max_prompt_tokens, args.verbose,
    )
    k = args.k
    print(
        f"\n{rec['model']}  task={rec['task']}  n={rec['n_prompts']}  "
        f"samples/prompt={rec['samples_per_prompt']}\n"
        f"  pass@1  = {rec['pass@1']:.4f}\n"
        f"  pass@{k}  = {rec[f'pass@{k}']:.4f}\n"
        f"  mean_reward = {rec['mean_reward']:.4f}   format_rate = {rec['format_rate']:.4f}\n"
        f"  mean completion = {rec['mean_gen_tokens']:.1f} tokens   "
        f"wall clock = {rec['wall_clock_s']:.1f} s"
    )
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(rec, indent=2))
        print(f"  wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
