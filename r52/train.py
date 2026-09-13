# Copyright 2026 The road-to-52 authors.
# SPDX-License-Identifier: Apache-2.0
"""Pretraining loop for road-to-52.

    python -m r52.train configs/gpt2_124m_mac.yaml --run-name b1 \
        --max-hours 120 -o micro_batch=4 -o train.compile=true

Writes ``runs/<run_name>/log.jsonl`` (one JSON object per log interval), ``stdout.log``
(when launched through ``scripts/train.sh``) and ``ckpt/`` (last ``keep_last`` step
directories plus ``best/``).

The learning-rate schedule is always laid out over the config's ``total_steps``
(``max_steps``, else ``max_tokens / tokens_per_step``).  ``--max-tokens``, ``--max-hours``
and ``--target-val-loss`` are *early stops* layered on top and deliberately do not reshape
it, so a smoke run and the real run share an identical schedule; use ``-o max_tokens=N`` to
change the schedule itself.  ``--max-hours`` counts wall-clock cumulatively across resumes.

Units: ``tokens`` are counted, ``tok_s`` is tokens/second, ``tflops`` is 1e12 FLOP/s,
``peak_mem_gb`` is GiB (``mx.get_peak_memory()``), ``eta_h`` is hours.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import signal
import sys
import time
from pathlib import Path
from typing import Any

import mlx.core as mx
from mlx import nn
from mlx.utils import tree_map

from . import __version__
from .checkpoint import latest_checkpoint, load_checkpoint, prune, save_checkpoint
from .config import Config, apply_overrides, load_config
from .data import ValLoader, load_shard, make_train_stream, make_val_loader
from .model import GPT
from .optim import build_optimizer
from .tokenizer import bits_per_byte, val_bytes_per_token

__all__ = ["Trainer", "evaluate", "main"]


# --------------------------------------------------------------------------------------
# Validation
# --------------------------------------------------------------------------------------


def evaluate(model: GPT, loader: ValLoader, bytes_per_token: float | None = None) -> dict[str, Any]:
    """Mean validation CE (nats/token, fp32 logits) and bits-per-byte over ``loader``."""
    total_nll = 0.0
    total_tok = 0
    for x, y in loader:
        loss = model.loss(x, y, fp32_logits=True)
        mx.eval(loss)
        n = int(y.size)
        total_nll += float(loss) * n
        total_tok += n
    val_loss = total_nll / max(1, total_tok)
    out: dict[str, Any] = {"val_loss": val_loss, "val_tokens": total_tok}
    if bytes_per_token:
        out["val_bpb"] = bits_per_byte(val_loss, total_tok, int(total_tok * bytes_per_token))
        out["val_bytes_per_token"] = bytes_per_token
    else:
        out["val_bpb"] = None
    return out


# --------------------------------------------------------------------------------------
# Trainer
# --------------------------------------------------------------------------------------


class Trainer:
    """Owns the model, optimizer, data stream, logging and checkpointing for one run."""

    def __init__(self, cfg: Config, run_name: str, resume: str | None = None,
                 compile_step: bool | None = None):
        self.cfg = cfg
        self.run_name = run_name
        self.run_dir = Path(cfg.train.out_dir) / run_name
        self.ckpt_dir = self.run_dir / "ckpt"
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)

        tc, mc = cfg.train, cfg.model
        mx.set_memory_limit(int(tc.memory_limit_gb * (1 << 30)))
        mx.set_cache_limit(int(tc.cache_limit_gb * (1 << 30)))
        mx.random.seed(tc.seed)

        self.accum = cfg.grad_accum()
        self.total_steps = cfg.total_steps()
        self.use_compile = tc.compile if compile_step is None else compile_step

        self.model = GPT(mc)
        mx.eval(self.model.parameters())
        self.opt = build_optimizer(tc, self.model.trainable_parameters())

        self.stream = make_train_stream(cfg.data, mc.block_size, tc.micro_batch, mc.vocab_size)
        val_mb = tc.val_micro_batch or tc.micro_batch
        self.val_loader = make_val_loader(cfg.data, mc.block_size, val_mb, tc.val_max_batches, mc.vocab_size)
        self.bytes_per_token = self._bytes_per_token()

        self.step = 0
        self.tokens = 0
        self.best_val = math.inf
        self.stop_reason: str | None = None
        self._stop = False
        self._t_start = time.time()
        self._elapsed_before = 0.0

        self.log_path = self.run_dir / "log.jsonl"
        # Kept open for the lifetime of the run; closed at the end of `run()`.
        self._log_fh = self.log_path.open("a", buffering=1)

        if resume:
            self._resume(resume)

        self._build_step_fn()
        self._install_signals()

    # -- setup helpers -------------------------------------------------------------
    def _bytes_per_token(self) -> float | None:
        """Bytes/token of the validation corpus (``None`` for synthetic data)."""
        if self.cfg.data.source != "fineweb":
            return None
        path = Path(self.cfg.data.data_dir) / self.cfg.data.val_file
        if not path.exists():
            return None
        n = self.val_loader.tokens
        return val_bytes_per_token(
            load_shard(path), n, Path(self.cfg.data.data_dir) / ".bpb_cache.json", "val"
        )

    def _build_step_fn(self) -> None:
        inv = 1.0 / self.accum

        def scaled_loss(x: mx.array, y: mx.array) -> mx.array:
            return self.model.loss(x, y) * inv

        fn = nn.value_and_grad(self.model, scaled_loss)
        if self.use_compile:
            fn = mx.compile(fn, inputs=[self.model.state], outputs=[self.model.state])
        self._fwd_bwd = fn

    def _install_signals(self) -> None:
        def handler(signum, _frame):
            if self._stop:  # second signal -> die now
                sys.exit(130)
            self._stop = True
            self.stop_reason = f"signal:{signal.Signals(signum).name}"
            print(f"\n[r52] {self.stop_reason} received; finishing step and checkpointing...", flush=True)

        for sig in (signal.SIGINT, signal.SIGTERM):
            signal.signal(sig, handler)

    def _resume(self, resume: str) -> None:
        path = latest_checkpoint(self.ckpt_dir) if resume == "auto" else Path(resume)
        if path is None or not Path(path).exists():
            print(f"[r52] no checkpoint to resume from in {self.ckpt_dir}; starting fresh", flush=True)
            return
        meta = load_checkpoint(path, self.model, self.opt.multi)
        self.step = int(meta.get("step", 0))
        self.tokens = int(meta.get("tokens", 0))
        self.best_val = float(meta.get("best_val") or math.inf)
        self._elapsed_before = float(meta.get("elapsed_s", 0.0))
        if meta.get("cursor"):
            self.stream.load_state(meta["cursor"])
        mx.random.seed(self.cfg.train.seed + self.step)
        mx.eval(self.model.parameters(), self.opt.state)
        print(f"[r52] resumed {path} at step {self.step} ({self.tokens:,} tokens)", flush=True)

    # -- logging --------------------------------------------------------------------
    def log(self, record: dict[str, Any]) -> None:
        record = {"t": round(time.time() - self._t_start + self._elapsed_before, 3), **record}
        self._log_fh.write(json.dumps(record) + "\n")

    # -- core ------------------------------------------------------------------------
    def train_step(self) -> tuple[float, float]:
        """Run one optimizer step (``accum`` micro-steps). Returns ``(loss, grad_norm)``."""
        acc = None
        loss_sum = mx.array(0.0)
        for _ in range(self.accum):
            x, y = self.stream.next_batch()
            ls, grads = self._fwd_bwd(x, y)
            acc = grads if acc is None else tree_map(lambda a, b: a + b, acc, grads)
            loss_sum = loss_sum + ls
            if self.accum > 1:
                # Bound graph growth: without this, `accum` full activation graphs stay live.
                mx.eval(acc, loss_sum)
        acc, gnorm = self.opt.clip(acc)
        self.opt.set_step(self.step, self.total_steps)
        self.opt.update(self.model, acc)
        mx.eval(self.model.parameters(), self.opt.state, loss_sum, gnorm)
        return float(loss_sum), float(gnorm)

    def validate(self) -> dict[str, Any]:
        return evaluate(self.model, self.val_loader, self.bytes_per_token)

    def save(self, tag: str | None = None, val_loss: float | None = None) -> Path:
        name = tag or f"step_{self.step:08d}"
        extra = {
            "best_val": self.best_val if math.isfinite(self.best_val) else None,
            "elapsed_s": time.time() - self._t_start + self._elapsed_before,
            "r52_version": __version__,
            "mlx_version": _mlx_version(),
        }
        if val_loss is not None:
            extra["val_loss"] = val_loss
        p = save_checkpoint(
            self.ckpt_dir / name,
            self.model,
            self.opt.multi,
            step=self.step,
            tokens=self.tokens,
            cursor=self.stream.state(),
            config=self.cfg,
            extra=extra,
        )
        if tag is None:
            prune(self.ckpt_dir, self.cfg.train.keep_last)
        return p

    # -- the loop ----------------------------------------------------------------------
    def run(self, max_tokens: int | None = None, max_hours: float | None = None,
            target_val_loss: float | None = None) -> dict[str, Any]:
        tc = self.cfg.train
        max_tokens = max_tokens if max_tokens is not None else tc.max_tokens
        max_hours = max_hours if max_hours is not None else tc.max_hours
        target = target_val_loss if target_val_loss is not None else tc.target_val_loss
        tokens_per_step = tc.tokens_per_step
        fpt = self.model.flops_per_token()
        n_flops_params = self.model.num_params_flops()

        header = {
            "event": "start",
            "run": self.run_name,
            "step": self.step,
            "total_steps": self.total_steps,
            "params_total": self.model.num_params(),
            "params_non_embedding": self.model.num_params(True),
            "params_flops": n_flops_params,
            "flops_per_token": fpt,
            "grad_accum": self.accum,
            "optimizer_groups": self.opt.group_names,
            "micro_batch": tc.micro_batch,
            "block_size": self.cfg.model.block_size,
            "tokens_per_step": tokens_per_step,
            "precision": self.cfg.model.precision,
            "compile": self.use_compile,
            "grad_checkpoint": self.cfg.model.grad_checkpoint,
            "max_tokens": max_tokens,
            "max_hours": max_hours,
            "target_val_loss": target,
            "mlx_version": _mlx_version(),
            "config": self.cfg.to_dict(),
        }
        self.log(header)

        available = self.stream.total_tokens
        budget = max_tokens or self.total_steps * tokens_per_step
        if self.cfg.data.source == "fineweb" and budget > available:
            msg = (
                f"[r52] WARNING: the token budget ({budget:,}) exceeds the tokens on disk "
                f"({available:,} across {len(self.stream.paths)} shard(s)); the stream will wrap "
                f"and repeat data {budget / max(1, available):.1f}x. "
                f"Run: python scripts/prepare_data.py --train-shards "
                f"{-(-budget // 100_000_000)}"
            )
            print(msg, flush=True)
            self.log({"event": "warning", "kind": "data_repeat",
                      "budget_tokens": int(budget), "available_tokens": int(available),
                      "repeats": round(budget / max(1, available), 2)})

        print(
            f"[r52] {self.run_name}: {n_flops_params/1e6:.1f}M flops-params "
            f"({self.model.num_params()/1e6:.1f}M total), {self.total_steps} steps x "
            f"{tokens_per_step:,} tok, accum={self.accum}, precision={self.cfg.model.precision}, "
            f"compile={self.use_compile}",
            flush=True,
        )

        t_window = time.time()
        tok_window = 0
        loss_window = 0.0
        n_window = 0
        last_metrics: dict[str, Any] = {}

        while not self._stop:
            if max_tokens and self.tokens >= max_tokens:
                self.stop_reason = "max_tokens"
                break
            if tc.max_steps and self.step >= tc.max_steps:
                self.stop_reason = "max_steps"
                break
            elapsed_h = (time.time() - self._t_start + self._elapsed_before) / 3600.0
            if max_hours and elapsed_h >= max_hours:
                self.stop_reason = "max_hours"
                break

            loss, gnorm = self.train_step()
            self.step += 1
            self.tokens += tokens_per_step
            tok_window += tokens_per_step
            loss_window += loss
            n_window += 1

            if self.step % tc.log_interval == 0:
                dt = max(1e-9, time.time() - t_window)
                tok_s = tok_window / dt
                tflops = tok_s * fpt / 1e12
                budget = max_tokens or self.total_steps * tokens_per_step
                remaining = max(0, budget - self.tokens)
                lrs = self.opt.learning_rates()
                last_metrics = {
                    "event": "train",
                    "step": self.step,
                    "tokens": self.tokens,
                    "loss": round(loss_window / max(1, n_window), 5),
                    "lr": round(lrs.get("muon", next(iter(lrs.values()))), 8),
                    "lrs": {k: round(v, 8) for k, v in lrs.items()},
                    "lr_mult": round(self.opt.lr_mult, 5),
                    "grad_norm": round(gnorm, 4),
                    "tok_s": round(tok_s, 1),
                    "tflops": round(tflops, 4),
                    "mfu_theoretical": round(tflops / tc.peak_tflops_theoretical, 4),
                    "mfu_measured": round(tflops / tc.peak_tflops_measured, 4),
                    "peak_mem_gb": round(mx.get_peak_memory() / (1 << 30), 3),
                    "eta_h": round(remaining / tok_s / 3600.0, 3) if tok_s > 0 else None,
                }
                self.log(last_metrics)
                print(
                    f"step {self.step:>6}/{self.total_steps} | loss {last_metrics['loss']:.4f} | "
                    f"{tok_s:8.0f} tok/s | {tflops:.3f} TF | mfu {100*last_metrics['mfu_theoretical']:.1f}%/"
                    f"{100*last_metrics['mfu_measured']:.1f}% | {last_metrics['peak_mem_gb']:.2f} GiB | "
                    f"eta {(last_metrics['eta_h'] or 0.0):.2f} h",
                    flush=True,
                )
                t_window, tok_window, loss_window, n_window = time.time(), 0, 0.0, 0

            if tc.val_interval and self.step % tc.val_interval == 0:
                v = self.validate()
                v.update({"event": "val", "step": self.step, "tokens": self.tokens})
                self.log(v)
                bpb = v.get("val_bpb")
                line = f"  val  {self.step:>6} | loss {v['val_loss']:.4f} | tokens {v['val_tokens']:,}"
                if bpb:
                    line += f" | bpb {bpb:.4f}"
                print(line, flush=True)
                if v["val_loss"] < self.best_val:
                    self.best_val = v["val_loss"]
                    self.save(tag="best", val_loss=v["val_loss"])
                if target and v["val_loss"] <= target:
                    self.stop_reason = "target_val_loss"
                    break
                t_window, tok_window, loss_window, n_window = time.time(), 0, 0.0, 0

            if tc.ckpt_interval and self.step % tc.ckpt_interval == 0:
                self.save()

        if self._stop and self.stop_reason is None:
            self.stop_reason = "stopped"
        self.stop_reason = self.stop_reason or "done"

        v = self.validate()
        self.save(val_loss=v["val_loss"])
        if v["val_loss"] < self.best_val:
            self.best_val = v["val_loss"]
            self.save(tag="best", val_loss=v["val_loss"])
        summary = {
            "event": "end",
            "step": self.step,
            "tokens": self.tokens,
            "reason": self.stop_reason,
            "best_val": self.best_val if math.isfinite(self.best_val) else None,
            "elapsed_s": round(time.time() - self._t_start + self._elapsed_before, 2),
            "peak_mem_gb": round(mx.get_peak_memory() / (1 << 30), 3),
            **v,
            **{f"last_{k}": val for k, val in last_metrics.items() if k in ("tok_s", "tflops")},
        }
        self.log(summary)
        print(
            f"[r52] done ({self.stop_reason}) step={self.step} tokens={self.tokens:,} "
            f"val_loss={v['val_loss']:.4f} bpb={v.get('val_bpb')} peak={summary['peak_mem_gb']:.2f} GiB",
            flush=True,
        )
        self._log_fh.close()
        return summary


def _mlx_version() -> str:
    import importlib.metadata as md

    try:
        return md.version("mlx")
    except Exception:  # pragma: no cover
        return "unknown"


# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="r52.train", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("config", help="path to a YAML config (see configs/)")
    p.add_argument("--run-name", default=None, help="run directory name (default: the config's `name`)")
    p.add_argument("--resume", nargs="?", const="auto", default=None,
                   help="resume from the newest checkpoint, or an explicit ckpt directory")
    p.add_argument("--no-compile", action="store_true", help="disable mx.compile of the fwd/bwd step")
    p.add_argument("--compile", dest="force_compile", action="store_true", help="force mx.compile on")
    p.add_argument("--max-tokens", type=int, default=None,
                   help="early stop (tokens). Does NOT reshape the LR schedule -- use "
                        "`-o max_tokens=N` for that.")
    p.add_argument("--max-hours", type=float, default=None,
                   help="early stop (hours), counted cumulatively across resumes")
    p.add_argument("--target-val-loss", type=float, default=None,
                   help="stop at the first validation at or below this fp32 loss")
    p.add_argument("-o", "--override", action="append", default=[], metavar="KEY=VALUE",
                   help="config override, e.g. -o micro_batch=2 -o model.n_layer=6")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    cfg: Config = load_config(args.config)
    if args.override:
        apply_overrides(cfg, args.override)
    run_name = args.run_name or cfg.name
    compile_step = None
    if args.no_compile:
        compile_step = False
    elif args.force_compile:
        compile_step = True

    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    trainer = Trainer(cfg, run_name, resume=args.resume, compile_step=compile_step)
    trainer.run(args.max_tokens, args.max_hours, args.target_val_loss)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
