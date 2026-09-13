# Copyright 2026 The road-to-52 authors.
# SPDX-License-Identifier: Apache-2.0
#
# The GRPO objective follows DeepSeekMath (arXiv 2402.03300 §4.1) with the DAPO
# modifications (arXiv 2503.14476): no KL term, clip-higher, token-level loss aggregation,
# no std normalisation of the advantage, and dynamic sampling of zero-variance groups.
# No code is copied from any GRPO implementation; `mlx-lm-lora` (Apache-2.0) is driven as a
# CLI in the alternative backend, never vendored.
"""RLVR with GRPO / DAPO on reasoning-gym, over an exported r52 model.

    python -m r52.posttrain.rl configs/posttrain/rl_tiny.yaml --model models/tiny200-mlx
    python -m r52.posttrain.rl configs/posttrain/rl_nano.yaml -o steps=200
    python -m r52.posttrain.rl configs/posttrain/rl_nano.yaml --backend mlx-lm-lora

Two backends, both working; **``native`` is the default and the one the shipped configs
use.**  Read ``docs/DEVIATIONS.md`` ("post-training builder") for the measurement that
decided it -- the short version:

``native`` (default)
    A ~200-line GRPO loop in this file, on the ``r52gpt`` plugin model.  Samples with the
    KV cache (:mod:`r52.posttrain.sampler`), scores with reasoning-gym's own verifier, and
    computes the policy gradient over ``logP(completion | prompt)`` -- i.e. **conditioned
    on the prompt**, which is the entire point of RLVR.  Implements every DAPO knob:
    ``beta=0`` (no KL, no reference model), ``epsilon_high > epsilon_low``, token-level
    aggregation, ``std_normalize=False``, and dynamic sampling.

``mlx-lm-lora``
    Shells out to ``mlx_lm_lora.train --train-mode grpo`` (Apache-2.0, the package
    ``research/02`` §Stage 9 recommends), generating the ``{prompt, answer}`` JSONL and a
    registered reasoning-gym reward function for it.  Verified to load our plugin model and
    train.  **Caveat, measured 2026-09-13:** its ``generate_grpo`` stores only the
    *completion* ids (``tokenizer.encode(completion_text)``) and ``grpo_loss`` feeds exactly
    those to the model, so its per-token log-probs are ``logP(completion)`` with **no
    prompt in context**; it also always divides the group advantage by its std and has no
    dynamic sampling, so DAPO's no-std-norm cannot be expressed.  Useful for comparison and
    for LoRA/QLoRA; not what we train on.

Units: ``reward`` is the verifier score in ``[0, 1]`` (plus an optional format bonus);
``kl`` is nats/token; ``gen_tokens`` counts sampled tokens; ``peak_mem_gb`` is GiB.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import mlx.core as mx
import mlx.optimizers as optim
import numpy as np
from mlx import nn
from mlx.utils import tree_flatten, tree_map

from ..chat_template import ASSISTANT_END, PAD
from ..chat_template import render_prompt as _render_prompt
from .config import RLConfig, load_rl_config, override
from .rgym import SYSTEM_PROMPT, format_score, make_pool
from .sampler import Sampler, load_exported_model

__all__ = ["GRPOTrainer", "main", "run_mlx_lm_lora"]


# --------------------------------------------------------------------------------------
# Log-probabilities
# --------------------------------------------------------------------------------------


def token_logprobs(model, x: mx.array, targets: mx.array) -> mx.array:
    """Per-position ``log p(targets[t] | x[:t+1])``, shape ``(B, T - 1)``.

    Computed as ``picked - logsumexp`` rather than ``log_softmax(...)[target]`` so the
    ``(B, T, vocab)`` log-softmax tensor is never materialised -- at vocab 50,304 that
    tensor is the single largest allocation in the step.
    """
    logits = model(x)[:, :-1, :].astype(mx.float32)
    lse = mx.logsumexp(logits, axis=-1)
    picked = mx.take_along_axis(logits, targets[..., None], axis=-1).squeeze(-1)
    return picked - lse


# --------------------------------------------------------------------------------------
# Rollouts
# --------------------------------------------------------------------------------------


class Rollouts:
    """One step's sampled group batch, padded into ``(B, T)`` arrays.

    ``x`` holds ``prompt + completion`` ids; ``mask`` marks the **completion** positions of
    ``x[:, 1:]`` (the positions whose log-probability the policy gradient acts on), so
    prompt tokens contribute nothing to the loss.
    """

    def __init__(self, x: mx.array, mask: mx.array, advantages: mx.array, rewards: np.ndarray) -> None:
        self.x = x
        self.mask = mask
        self.advantages = advantages
        self.rewards = rewards

    def __len__(self) -> int:
        return int(self.x.shape[0])


def build_rollouts(
    prompts: list[list[int]],
    groups: list[list[Any]],
    rewards: list[list[float]],
    cfg: RLConfig,
) -> tuple[Rollouts | None, dict[str, Any]]:
    """Pad sampled groups into training arrays and compute group-normalised advantages.

    Returns ``(rollouts, stats)``; ``rollouts`` is ``None`` when dynamic sampling removed
    every group (all completions in every group scored identically, so no gradient exists).
    """
    kept_x: list[np.ndarray] = []
    kept_mask: list[np.ndarray] = []
    kept_adv: list[float] = []
    kept_reward: list[float] = []
    n_groups_dropped = 0

    for p_ids, comps, rs in zip(prompts, groups, rewards, strict=True):
        arr = np.asarray(rs, dtype=np.float64)
        centred = arr - arr.mean()
        spread = float(arr.std())
        if cfg.dynamic_sampling and spread < 1e-8:
            n_groups_dropped += 1
            continue
        if cfg.std_normalize:
            centred = centred / (spread + 1e-4)
        for comp, adv, r in zip(comps, centred, arr, strict=True):
            ids = list(comp.tokens)
            if comp.finished:
                ids = [*ids, ASSISTANT_END]  # reward the stop token the model actually chose
            if not ids:
                continue
            kept_x.append(np.asarray(p_ids + ids, dtype=np.int32))
            m = np.zeros(len(p_ids) + len(ids) - 1, dtype=np.float32)
            m[len(p_ids) - 1 :] = 1.0  # positions predicting a completion token
            kept_mask.append(m)
            kept_adv.append(float(adv))
            kept_reward.append(float(r))

    stats = {
        "groups_dropped": n_groups_dropped,
        "groups_kept": len(prompts) - n_groups_dropped,
        "sequences": len(kept_x),
        # The surrogate's *value* is ~0 whenever the importance ratio is 1 (inner_epochs=1,
        # on-policy), because the advantages are group-centred and cancel. The quantity that
        # says whether this step carries signal is the advantage magnitude, not the loss.
        "adv_abs_mean": round(float(np.mean(np.abs(kept_adv))), 6) if kept_adv else 0.0,
    }
    if not kept_x:
        return None, stats

    width = max(len(v) for v in kept_x)
    x = np.full((len(kept_x), width), PAD, dtype=np.int32)
    mask = np.zeros((len(kept_x), width - 1), dtype=np.float32)
    for i, (row, m) in enumerate(zip(kept_x, kept_mask, strict=True)):
        x[i, : len(row)] = row
        mask[i, : len(m)] = m
    return (
        Rollouts(mx.array(x), mx.array(mask), mx.array(np.asarray(kept_adv, dtype=np.float32)),
                 np.asarray(kept_reward, dtype=np.float64)),
        stats,
    )


# --------------------------------------------------------------------------------------
# Trainer
# --------------------------------------------------------------------------------------


class GRPOTrainer:
    """Native GRPO/DAPO over an exported mlx-lm model directory."""

    def __init__(self, cfg: RLConfig, run_name: str | None = None) -> None:
        self.cfg = cfg
        self.run_name = run_name or cfg.name
        mx.set_memory_limit(int(cfg.memory_limit_gb * (1 << 30)))
        mx.set_cache_limit(1 << 29)
        mx.random.seed(cfg.seed)

        if not cfg.model:
            raise ValueError("RLConfig.model must point at an exported mlx-lm directory")
        self.model, self.tokenizer = load_exported_model(cfg.model)
        # The export is bf16; AdamW at 1e-5 is below bf16's resolution, so RL runs in fp32.
        self.model.update(tree_map(lambda p: p.astype(mx.float32), self.model.parameters()))
        mx.eval(self.model.parameters())
        self.sampler = Sampler(self.model, self.tokenizer)
        self.system_prompt = cfg.system_prompt or (SYSTEM_PROMPT if cfg.use_system_prompt else "")

        self.opt = optim.AdamW(
            learning_rate=cfg.learning_rate,
            betas=list(cfg.adam_betas),
            weight_decay=cfg.weight_decay,
        )
        self.items, self.tasks = make_pool(cfg.tasks, cfg.steps * cfg.prompts_per_step + 1,
                                           cfg.task_seed, cfg.task_size)
        self.run_dir = Path(cfg.out_dir) / self.run_name
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self._log_fh = (self.run_dir / "log.jsonl").open("a", buffering=1)
        self.step = 0
        self._cursor = 0
        self._t0 = time.time()

    # -- logging --------------------------------------------------------------------
    def log(self, rec: dict[str, Any]) -> None:
        self._log_fh.write(json.dumps({"t": round(time.time() - self._t0, 3), **rec}) + "\n")

    # -- one step -------------------------------------------------------------------
    def rollout(self) -> tuple[Rollouts | None, dict[str, Any]]:
        """Sample, score, and build the training batch for one optimizer step."""
        cfg = self.cfg
        batch = [self.items[(self._cursor + i) % len(self.items)] for i in range(cfg.prompts_per_step)]
        self._cursor += cfg.prompts_per_step

        prompts, groups, rewards = [], [], []
        gen_tokens = n_format = 0
        for i, item in enumerate(batch):
            p = _render_prompt(item.messages(self.system_prompt), max_tokens=cfg.max_prompt_tokens)
            comps = self.sampler.sample_group(p, cfg.group_size, cfg.max_completion_tokens,
                                              cfg.temperature, cfg.top_p, cfg.top_k, prompt_index=i)
            task = self.tasks[item.task]
            rs = []
            for c in comps:
                r = task.score(c.text, item)
                if cfg.format_reward:
                    r += cfg.format_reward * format_score(c.text)
                rs.append(r)
                gen_tokens += len(c)
                n_format += int(format_score(c.text))
            prompts.append(p)
            groups.append(comps)
            rewards.append(rs)

        flat = [r for rs in rewards for r in rs]
        rollouts, stats = build_rollouts(prompts, groups, rewards, cfg)
        stats.update({
            "reward_mean": round(float(np.mean(flat)), 5),
            "reward_std": round(float(np.std(flat)), 5),
            "reward_max": round(float(np.max(flat)), 5),
            "solve_rate": round(float(np.mean([r >= 1.0 for r in flat])), 5),
            "gen_tokens": gen_tokens,
            "mean_gen_tokens": round(gen_tokens / max(1, len(flat)), 1),
            "format_rate": round(n_format / max(1, len(flat)), 4),
        })
        return rollouts, stats

    def _make_loss_fn(self):
        """Build the DAPO surrogate ``fn(x, mask, adv, old, denom) -> scalar``.

        ``old`` is ``log pi_old(completion | prompt)``.  With ``inner_epochs == 1`` the
        policy has not moved since sampling, so the caller passes the *in-graph*
        stop-gradient of ``logp``: the ratio is then exactly 1, clipping is inactive by
        construction, and the gradient reduces to the plain GRPO estimator
        ``-A * grad log pi``.  Clip-higher only ever bites with ``inner_epochs > 1``.
        """
        cfg = self.cfg

        def fn(x: mx.array, mask: mx.array, adv: mx.array, old: mx.array | None,
               denom: mx.array) -> mx.array:
            logp = token_logprobs(self.model, x, x[:, 1:])
            ref = mx.stop_gradient(logp) if old is None else mx.stop_gradient(old)
            ratio = mx.exp(logp - ref)
            a = adv[:, None]
            clipped = mx.clip(ratio, 1.0 - cfg.epsilon_low, 1.0 + cfg.epsilon_high)
            per_token = -mx.minimum(ratio * a, clipped * a)
            if cfg.loss_agg == "sequence":
                per_seq = (per_token * mask).sum(axis=1) / mx.maximum(mask.sum(axis=1), 1.0)
                return per_seq.sum() / denom
            return (per_token * mask).sum() / denom

        return fn

    def train_step(self, rollouts: Rollouts) -> tuple[float, float, float]:
        """Accumulate the DAPO gradient over micro-batches and apply ``inner_epochs`` updates.

        Returns ``(loss, grad_norm, clip_fraction)`` of the last inner epoch.
        """
        cfg = self.cfg
        n = len(rollouts)
        mb = max(1, cfg.micro_batch)
        # DAPO's token-level normaliser is the total completion-token count of the WHOLE
        # batch, not a per-micro-batch mean: a per-sequence mean up-weights short
        # completions, which is precisely the length bias DAPO removes.
        total_tokens = float(mx.sum(rollouts.mask).item())
        denom = mx.array(max(1.0, total_tokens if cfg.loss_agg == "token" else float(n)))

        loss_fn = self._make_loss_fn()
        value_and_grad = nn.value_and_grad(self.model, loss_fn)

        cached_old: list[mx.array] | None = None
        if cfg.inner_epochs > 1:
            cached_old = []
            for a in range(0, n, mb):
                b = min(a + mb, n)
                x = rollouts.x[a:b]
                old = mx.stop_gradient(token_logprobs(self.model, x, x[:, 1:]))
                mx.eval(old)
                cached_old.append(old)

        total_loss, gnorm, clip_frac = 0.0, 0.0, 0.0
        for epoch in range(max(1, cfg.inner_epochs)):
            acc = None
            total_loss = 0.0
            n_clip = n_tok = 0
            for j, a in enumerate(range(0, n, mb)):
                b = min(a + mb, n)
                x, mask, adv = rollouts.x[a:b], rollouts.mask[a:b], rollouts.advantages[a:b]
                old = cached_old[j] if cached_old is not None else None
                if old is not None and epoch > 0:
                    now = mx.stop_gradient(token_logprobs(self.model, x, x[:, 1:]))
                    ratio = mx.exp(now - old)
                    outside = ((ratio < 1.0 - cfg.epsilon_low) | (ratio > 1.0 + cfg.epsilon_high))
                    n_clip += int((outside.astype(mx.float32) * mask).sum().item())
                    n_tok += int(mask.sum().item())
                ls, grads = value_and_grad(x, mask, adv, old, denom)
                acc = grads if acc is None else tree_map(lambda u, v: u + v, acc, grads)
                total_loss += float(ls)
                mx.eval(acc)
            if cfg.grad_clip > 0:
                acc, gnorm_arr = optim.clip_grad_norm(acc, cfg.grad_clip)
            else:
                from mlx.utils import tree_reduce

                gnorm_arr = mx.sqrt(tree_reduce(lambda s, g: s + g.square().sum(), acc, 0.0))
            warm = min(1.0, (self.step + 1) / max(1, cfg.warmup_steps)) if cfg.warmup_steps else 1.0
            self.opt.learning_rate = cfg.learning_rate * warm
            self.opt.update(self.model, acc)
            mx.eval(self.model.parameters(), self.opt.state)
            gnorm = float(gnorm_arr)
            clip_frac = n_clip / max(1, n_tok)
        return total_loss, gnorm, clip_frac

    # -- saving ----------------------------------------------------------------------
    def save(self, tag: str = "final") -> Path:
        """Write the updated policy as a fresh mlx-lm model directory (bf16 weights)."""
        out = self.run_dir / tag
        out.mkdir(parents=True, exist_ok=True)
        src = Path(self.cfg.model)
        for name in ("config.json", "r52gpt.py", "tokenizer.json", "tokenizer_config.json",
                     "vocab.json", "merges.txt"):
            if (src / name).exists():
                shutil.copyfile(src / name, out / name)
        weights = {k: v.astype(mx.bfloat16) for k, v in tree_flatten(self.model.parameters())}
        mx.save_safetensors(str(out / "model.safetensors"), weights, metadata={"format": "mlx"})
        (out / "rl.json").write_text(json.dumps({
            "stage": "rl", "backend": "native", "step": self.step,
            "base_model": str(src), "config": self.cfg.to_dict(),
        }, indent=2))
        return out

    # -- the loop ---------------------------------------------------------------------
    def run(self) -> dict[str, Any]:
        cfg = self.cfg
        self.log({"event": "start", "stage": "rl", "backend": "native", "run": self.run_name,
                  "config": cfg.to_dict()})
        print(
            f"[r52.rl] {self.run_name}: {cfg.steps} steps x {cfg.prompts_per_step} prompts x "
            f"{cfg.group_size} samples on {cfg.tasks} | DAPO(beta={cfg.beta}, "
            f"eps={cfg.epsilon_low}/{cfg.epsilon_high}, std_norm={cfg.std_normalize}, "
            f"dyn={cfg.dynamic_sampling}, agg={cfg.loss_agg})",
            flush=True,
        )
        history: list[float] = []
        for _ in range(cfg.steps):
            t_step = time.time()
            rollouts, stats = self.rollout()
            self.step += 1
            if rollouts is None:
                rec = {"event": "skip", "step": self.step, "reason": "all groups zero-variance", **stats}
                self.log(rec)
                print(f"step {self.step:>4}/{cfg.steps} | SKIPPED (no reward variance) | "
                      f"reward {stats['reward_mean']:.4f}", flush=True)
                history.append(stats["reward_mean"])
                continue
            loss, gnorm, clip_frac = self.train_step(rollouts)
            history.append(stats["reward_mean"])
            rec = {
                "event": "train", "step": self.step, "loss": round(loss, 6),
                "grad_norm": round(gnorm, 4), "clip_frac": round(clip_frac, 4),
                "lr": float(self.opt.learning_rate), "step_s": round(time.time() - t_step, 2),
                "peak_mem_gb": round(mx.get_peak_memory() / (1 << 30), 3), **stats,
            }
            self.log(rec)
            if self.step % cfg.log_interval == 0:
                print(
                    f"step {self.step:>4}/{cfg.steps} | reward {stats['reward_mean']:.4f} "
                    f"(solve {100 * stats['solve_rate']:.1f}%) | |adv| {stats['adv_abs_mean']:.4f} | "
                    f"loss {loss:+.5f} | |g| {gnorm:.3f} | {stats['sequences']} seqs, "
                    f"{stats['groups_dropped']} groups dropped | "
                    f"{rec['step_s']:.1f}s | {rec['peak_mem_gb']:.2f} GiB",
                    flush=True,
                )
            if cfg.save_interval and self.step % cfg.save_interval == 0:
                self.save(f"step_{self.step:06d}")

        out = self.save("final")
        summary = {
            "event": "end", "steps": self.step,
            "reward_first": round(history[0], 5) if history else None,
            "reward_last": round(history[-1], 5) if history else None,
            "reward_mean": round(float(np.mean(history)), 5) if history else None,
            "saved": str(out),
            "elapsed_s": round(time.time() - self._t0, 2),
            "peak_mem_gb": round(mx.get_peak_memory() / (1 << 30), 3),
        }
        self.log(summary)
        print(f"[r52.rl] done: mean reward {summary['reward_mean']} "
              f"(first {summary['reward_first']} -> last {summary['reward_last']}), "
              f"weights -> {out}", flush=True)
        self._log_fh.close()
        return summary


# --------------------------------------------------------------------------------------
# The mlx-lm-lora backend
# --------------------------------------------------------------------------------------

_REWARD_FILE = '''# Copyright 2026 The road-to-52 authors.
# SPDX-License-Identifier: Apache-2.0
"""Generated by r52.posttrain.rl -- a reasoning-gym verifier as an mlx-lm-lora reward.

mlx-lm-lora's reward contract is ``f(prompts, completions, answer, types=None) -> [float]``,
which carries the answer *string* but not reasoning-gym's ``entry`` dict (some verifiers
need its metadata).  So the entries are keyed by question text in ``entries.json`` next to
this file, written by the same run.
"""

import json
import os
import sys

sys.path.insert(0, {repo!r})

from mlx_lm_lora.trainer.grpo_reward_functions import register_reward_function

from r52.posttrain.rgym import RGymTask

_ENTRIES = json.loads(open(os.path.join(os.path.dirname(__file__), "entries.json")).read())
_TASKS = {{}}


def _task(name):
    if name not in _TASKS:
        _TASKS[name] = RGymTask(name, size={task_size}, seed={task_seed})
    return _TASKS[name]


@register_reward_function()
def r52_reasoning_gym(prompts, completions, answer, types=None):
    out = []
    for prompt, completion in zip(prompts, completions):
        key = None
        for question in _ENTRIES:
            if question in (prompt or ""):
                key = question
                break
        if key is None:
            out.append(0.0)
            continue
        rec = _ENTRIES[key]
        task = _task(rec["task"])
        from r52.posttrain.rgym import Item

        item = Item(rec["task"], key, rec["answer"], rec["entry"])
        out.append(float(task.score(completion or "", item)))
    return out
'''


def run_mlx_lm_lora(cfg: RLConfig, run_name: str, extra: list[str] | None = None) -> int:
    """Drive ``mlx_lm_lora.train --train-mode grpo`` on the exported model.

    Materialises ``{train,valid,test}.jsonl``, ``entries.json`` and ``rewards.py`` under
    ``runs/<name>/mlx-lm-lora/`` and execs the CLI.  DAPO knobs are mapped where the library
    has them (``--beta 0``, ``--epsilon``/``--epsilon-high``, ``--grpo-loss-type dr_grpo``,
    ``--importance-sampling-level token``); ``std_normalize=False`` and ``dynamic_sampling``
    have no equivalent there -- see this module's docstring.
    """
    work = Path(cfg.out_dir) / run_name / "mlx-lm-lora"
    data = work / "data"
    data.mkdir(parents=True, exist_ok=True)

    system_prompt = cfg.system_prompt or (SYSTEM_PROMPT if cfg.use_system_prompt else "")
    # mlx-lm-lora's `iterate_grpo_batches` refuses any split smaller than --batch-size, and
    # it evaluates before the first step, so valid/test each need >= prompts_per_step rows;
    # they are held out from the tail of the pool rather than copied out of train.
    n_eval = max(cfg.prompts_per_step, 2)
    items, _ = make_pool(cfg.tasks, cfg.steps * cfg.prompts_per_step + 2 * n_eval,
                         cfg.task_seed, cfg.task_size)
    entries = {}
    rows = []
    for it in items:
        entries[it.question] = {"task": it.task, "answer": it.answer, "entry": it.entry}
        row = {"prompt": it.question, "answer": it.answer}
        if system_prompt:
            row["system"] = system_prompt
        rows.append(row)
    train_rows, valid_rows, test_rows = rows[: -2 * n_eval], rows[-2 * n_eval : -n_eval], rows[-n_eval:]
    for name, subset in (("train", train_rows), ("valid", valid_rows), ("test", test_rows)):
        with (data / f"{name}.jsonl").open("w") as fh:
            for r in subset:
                fh.write(json.dumps(r) + "\n")
    (work / "entries.json").write_text(json.dumps(entries))
    repo = str(Path(__file__).resolve().parents[2])
    (work / "rewards.py").write_text(
        _REWARD_FILE.format(repo=repo, task_size=cfg.task_size, task_seed=cfg.task_seed)
    )

    cmd = [
        sys.executable, "-m", "mlx_lm_lora.train",
        "--model", cfg.model,
        "--train",
        "--train-mode", "grpo",
        "--train-type", "full",
        "--data", str(data),
        "--batch-size", str(cfg.prompts_per_step),
        "--group-size", str(cfg.group_size),
        "--iters", str(cfg.steps),
        "--max-completion-length", str(cfg.max_completion_tokens),
        "--max-seq-length", str(cfg.max_prompt_tokens + cfg.max_completion_tokens),
        "--learning-rate", str(cfg.learning_rate),
        "--temperature", str(cfg.temperature),
        "--beta", str(cfg.beta),
        "--epsilon", str(cfg.epsilon_low),
        "--epsilon-high", str(cfg.epsilon_high),
        "--grpo-loss-type", "dr_grpo",
        "--importance-sampling-level", "token",
        "--reward-functions-file", str(work / "rewards.py"),
        "--reward-functions", "r52_reasoning_gym",
        "--adapter-path", str(work / "adapters"),
        "--steps-per-report", str(cfg.log_interval),
        "--seed", str(cfg.seed),
        *(extra or []),
    ]
    print("[r52.rl] " + " ".join(cmd), flush=True)
    return subprocess.call(cmd)


# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="r52.posttrain.rl", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("config", help="an RL YAML (see configs/posttrain/)")
    p.add_argument("--model", default=None, help="override the exported mlx-lm model directory")
    p.add_argument("--run-name", default=None)
    p.add_argument("--backend", choices=("native", "mlx-lm-lora"), default="native",
                   help="`native` (default) is the correct-objective loop in this file; "
                        "`mlx-lm-lora` forwards unrecognised flags to that CLI")
    p.add_argument("-o", "--override", action="append", default=[], metavar="KEY=VALUE")
    return p


def main(argv: list[str] | None = None) -> int:
    args, passthrough = build_parser().parse_known_args(argv)
    cfg = load_rl_config(args.config)
    if args.model:
        cfg.model = args.model
    override(cfg, args.override)
    if not math.isfinite(cfg.learning_rate) or cfg.learning_rate <= 0:
        raise SystemExit("learning_rate must be > 0")
    run_name = args.run_name or cfg.name
    if args.backend == "mlx-lm-lora":
        return run_mlx_lm_lora(cfg, run_name, passthrough)
    GRPOTrainer(cfg, run_name).run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
