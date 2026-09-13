# Copyright 2026 The road-to-52 authors.
# SPDX-License-Identifier: Apache-2.0
"""Supervised fine-tuning of an r52 checkpoint, native MLX, assistant-only loss.

    python -m r52.posttrain.sft configs/posttrain/sft_nano.yaml --run-name s1
    python -m r52.posttrain.sft configs/posttrain/sft_tiny.yaml -o max_steps=20

What this is *not*: a second trainer.  The optimizer (Muon + 4x AdamW, five groups), the
WSD schedule, the gradient clipping and the checkpoint format all come from
:mod:`r52.optim` and :mod:`r52.checkpoint` unchanged -- the only things SFT adds are a
conversation batcher and a loss mask.

The mask needs no change to :class:`r52.model.GPT` either: ``GPT.loss`` already ignores
negative targets, so :func:`r52.posttrain.data.SFTBatcher` simply emits
:data:`r52.chat_template.IGNORE_INDEX` for every system/user/padding position.  The
reported ``loss`` is therefore the mean cross-entropy **per supervised assistant token**,
in nats.

Units: ``tokens`` counts padded tokens seen; ``sup_tokens`` counts supervised (assistant)
tokens; ``tok_s`` is padded tokens/second; ``peak_mem_gb`` is GiB.
"""

from __future__ import annotations

import argparse
import json
import math
import signal
import sys
import time
from pathlib import Path
from typing import Any

import mlx.core as mx
from mlx import nn
from mlx.utils import tree_map

from .. import __version__
from ..checkpoint import checkpoint_config, load_checkpoint, prune, save_checkpoint
from ..config import Config
from ..model import GPT
from ..optim import build_optimizer
from .config import SFTConfig, load_sft_config, override
from .data import ConversationDataset, PackedBatcher, SFTBatcher, iter_batches

__all__ = ["SFTTrainer", "evaluate", "main"]


def masked_loss(model: GPT, x: mx.array, y: mx.array, fp32_logits: bool | None = None) -> mx.array:
    """Mean CE in nats over the tokens of ``y`` that are ``>= 0`` (the assistant span)."""
    return model.loss(x, y, fp32_logits=fp32_logits)


def evaluate(model: GPT, dataset: ConversationDataset, micro_batch: int, max_seq: int,
             max_batches: int = 16, pad_multiple: int = 64) -> dict[str, Any]:
    """Validation CE over assistant tokens (fp32 logits) and the token count it used."""
    total, n = 0.0, 0
    for i, (x, y) in enumerate(iter_batches(dataset, micro_batch, max_seq, pad_multiple)):
        if max_batches and i >= max_batches:
            break
        k = int((y >= 0).sum())
        if k == 0:
            continue
        loss = model.loss(x, y, fp32_logits=True)
        mx.eval(loss)
        total += float(loss) * k
        n += k
    return {"val_loss": total / max(1, n), "val_sup_tokens": n}


class SFTTrainer:
    """One SFT run: model + optimizer + conversation batcher + logging + checkpoints."""

    def __init__(self, cfg: SFTConfig, run_name: str | None = None) -> None:
        self.cfg = cfg
        self.run_name = run_name or cfg.name
        tc = cfg.train
        mx.set_memory_limit(int(tc.memory_limit_gb * (1 << 30)))
        mx.set_cache_limit(int(tc.cache_limit_gb * (1 << 30)))
        mx.random.seed(tc.seed)

        if not cfg.init_from:
            raise ValueError("SFTConfig.init_from must point at a base/midtrain checkpoint")
        base = checkpoint_config(cfg.init_from)
        self.model_cfg = base.model
        if cfg.max_seq > self.model_cfg.block_size:
            print(
                f"[r52.sft] max_seq {cfg.max_seq} > block_size {self.model_cfg.block_size}; "
                f"clamping to {self.model_cfg.block_size}",
                flush=True,
            )
            cfg.max_seq = self.model_cfg.block_size

        self.model = GPT(self.model_cfg)
        mx.eval(self.model.parameters())
        load_checkpoint(cfg.init_from, self.model)
        mx.eval(self.model.parameters())
        self.opt = build_optimizer(tc, self.model.trainable_parameters())

        root = Path(cfg.data_dir)
        self.train_ds = ConversationDataset(root / "train")
        valid_dir = root / "valid"
        self.valid_ds = ConversationDataset(valid_dir) if (valid_dir / "meta.json").exists() else None

        if cfg.pack:
            self.batcher: Any = PackedBatcher(self.train_ds, cfg.micro_batch, cfg.max_seq,
                                              cfg.shuffle_seed)
        else:
            self.batcher = SFTBatcher(self.train_ds, cfg.micro_batch, cfg.max_seq,
                                      cfg.pad_multiple, cfg.shuffle_seed)

        per_epoch = max(1, self.batcher.batches_per_epoch() // cfg.grad_accum)
        self.total_steps = cfg.max_steps if cfg.max_steps > 0 else max(1, int(per_epoch * cfg.epochs))
        self.steps_per_epoch = per_epoch

        self.run_dir = Path(tc.out_dir) / self.run_name
        self.ckpt_dir = self.run_dir / "ckpt"
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        self._log_fh = (self.run_dir / "log.jsonl").open("a", buffering=1)

        self.step = 0
        self.tokens = 0
        self.sup_tokens = 0
        self.best_val = math.inf
        self._stop = False
        self.stop_reason: str | None = None
        self._t_start = time.time()
        self._build_step_fn()
        self._install_signals()

    # -- setup ---------------------------------------------------------------------
    def _build_step_fn(self) -> None:
        inv = 1.0 / self.cfg.grad_accum

        def scaled(x: mx.array, y: mx.array) -> mx.array:
            return masked_loss(self.model, x, y) * inv

        fn = nn.value_and_grad(self.model, scaled)
        if self.cfg.train.compile:
            fn = mx.compile(fn, inputs=[self.model.state], outputs=[self.model.state])
        self._fwd_bwd = fn

    def _install_signals(self) -> None:
        def handler(signum, _frame):
            if self._stop:
                sys.exit(130)
            self._stop = True
            self.stop_reason = f"signal:{signal.Signals(signum).name}"
            print(f"\n[r52.sft] {self.stop_reason}; finishing the step and checkpointing...", flush=True)

        for sig in (signal.SIGINT, signal.SIGTERM):
            signal.signal(sig, handler)

    def log(self, record: dict[str, Any]) -> None:
        self._log_fh.write(json.dumps({"t": round(time.time() - self._t_start, 3), **record}) + "\n")

    def _run_config(self) -> Config:
        """A full :class:`r52.config.Config` so exported SFT checkpoints self-describe."""
        base = checkpoint_config(self.cfg.init_from)
        base.name = self.run_name
        base.train = self.cfg.train
        base.train.max_steps = self.total_steps
        # Keep micro_batch/tokens_per_step consistent so `Config.grad_accum()` stays valid on
        # an SFT checkpoint's stored config (export and the eval harness both read it back).
        base.train.micro_batch = self.cfg.micro_batch
        base.train.tokens_per_step = self.cfg.tokens_per_step()
        return base

    # -- core ------------------------------------------------------------------------
    def train_step(self) -> tuple[float, float, int, int]:
        """One optimizer step. Returns ``(loss, grad_norm, tokens, supervised_tokens)``."""
        acc = None
        loss_sum = mx.array(0.0)
        n_tok = n_sup = 0
        for _ in range(self.cfg.grad_accum):
            x, y = self.batcher.next_batch()
            n_tok += int(x.size)
            n_sup += int((y >= 0).sum())
            ls, grads = self._fwd_bwd(x, y)
            acc = grads if acc is None else tree_map(lambda a, b: a + b, acc, grads)
            loss_sum = loss_sum + ls
            if self.cfg.grad_accum > 1:
                mx.eval(acc, loss_sum)  # bound graph growth (see docs/DEVIATIONS.md §1)
        acc, gnorm = self.opt.clip(acc)
        self.opt.set_step(self.step, self.total_steps)
        self.opt.update(self.model, acc)
        mx.eval(self.model.parameters(), self.opt.state, loss_sum, gnorm)
        return float(loss_sum), float(gnorm), n_tok, n_sup

    def validate(self) -> dict[str, Any]:
        if self.valid_ds is None or self.cfg.val_batches <= 0:
            return {}
        return evaluate(self.model, self.valid_ds, self.cfg.micro_batch, self.cfg.max_seq,
                        self.cfg.val_batches, self.cfg.pad_multiple)

    def save(self, tag: str | None = None, val_loss: float | None = None) -> Path:
        name = tag or f"step_{self.step:08d}"
        extra = {
            "stage": "sft",
            "init_from": self.cfg.init_from,
            "sft": self.cfg.to_dict(),
            "sup_tokens": self.sup_tokens,
            "elapsed_s": round(time.time() - self._t_start, 2),
            "r52_version": __version__,
        }
        if val_loss is not None:
            extra["val_loss"] = val_loss
        p = save_checkpoint(
            self.ckpt_dir / name, self.model, self.opt.multi,
            step=self.step, tokens=self.tokens, cursor=self.batcher.state(),
            config=self._run_config(), extra=extra,
        )
        if tag is None:
            prune(self.ckpt_dir, self.cfg.train.keep_last)
        return p

    # -- the loop ----------------------------------------------------------------------
    def run(self) -> dict[str, Any]:
        tc = self.cfg.train
        self.log({
            "event": "start", "stage": "sft", "run": self.run_name,
            "total_steps": self.total_steps, "steps_per_epoch": self.steps_per_epoch,
            "conversations": len(self.train_ds), "dataset_tokens": self.train_ds.n_tokens,
            "dataset_supervised_tokens": self.train_ds.n_supervised,
            "params_total": self.model.num_params(),
            "micro_batch": self.cfg.micro_batch, "grad_accum": self.cfg.grad_accum,
            "max_seq": self.cfg.max_seq, "pack": self.cfg.pack,
            "optimizer_groups": self.opt.group_names,
            "config": self.cfg.to_dict(),
        })
        print(
            f"[r52.sft] {self.run_name}: {len(self.train_ds):,} conversations "
            f"({self.train_ds.n_supervised:,} assistant tokens), {self.total_steps} steps x "
            f"{self.cfg.tokens_per_step():,} padded tok, init_from={self.cfg.init_from}",
            flush=True,
        )

        t_win, tok_win, loss_win, n_win = time.time(), 0, 0.0, 0
        while self.step < self.total_steps and not self._stop:
            loss, gnorm, n_tok, n_sup = self.train_step()
            self.step += 1
            self.tokens += n_tok
            self.sup_tokens += n_sup
            tok_win += n_tok
            loss_win += loss
            n_win += 1

            if self.step % tc.log_interval == 0 or self.step == self.total_steps:
                dt = max(1e-9, time.time() - t_win)
                rec = {
                    "event": "train", "step": self.step, "tokens": self.tokens,
                    "sup_tokens": self.sup_tokens,
                    "loss": round(loss_win / max(1, n_win), 5),
                    "lrs": {k: round(v, 8) for k, v in self.opt.learning_rates().items()},
                    "lr_mult": round(self.opt.lr_mult, 5),
                    "grad_norm": round(gnorm, 4),
                    "tok_s": round(tok_win / dt, 1),
                    "peak_mem_gb": round(mx.get_peak_memory() / (1 << 30), 3),
                    "epoch": self.batcher.epoch,
                }
                self.log(rec)
                print(
                    f"step {self.step:>5}/{self.total_steps} | loss {rec['loss']:.4f} | "
                    f"{rec['tok_s']:8.0f} tok/s | sup {self.sup_tokens:,} | "
                    f"{rec['peak_mem_gb']:.2f} GiB",
                    flush=True,
                )
                t_win, tok_win, loss_win, n_win = time.time(), 0, 0.0, 0

            if tc.val_interval and self.step % tc.val_interval == 0:
                v = self.validate()
                if v:
                    self.log({"event": "val", "step": self.step, **v})
                    print(f"  val {self.step:>5} | loss {v['val_loss']:.4f} "
                          f"| {v['val_sup_tokens']:,} assistant tokens", flush=True)
                    if v["val_loss"] < self.best_val:
                        self.best_val = v["val_loss"]
                        self.save(tag="best", val_loss=v["val_loss"])
                t_win = time.time()

            if tc.ckpt_interval and self.step % tc.ckpt_interval == 0:
                self.save()

        self.stop_reason = self.stop_reason or "done"
        v = self.validate()
        self.save(val_loss=v.get("val_loss"))
        if v and v["val_loss"] < self.best_val:
            self.best_val = v["val_loss"]
            self.save(tag="best", val_loss=v["val_loss"])
        if not (self.ckpt_dir / "best").exists():
            self.save(tag="best", val_loss=v.get("val_loss"))
        summary = {
            "event": "end", "step": self.step, "tokens": self.tokens,
            "sup_tokens": self.sup_tokens, "reason": self.stop_reason,
            "best_val": self.best_val if math.isfinite(self.best_val) else None,
            "elapsed_s": round(time.time() - self._t_start, 2),
            "peak_mem_gb": round(mx.get_peak_memory() / (1 << 30), 3),
            **v,
        }
        self.log(summary)
        print(
            f"[r52.sft] done ({self.stop_reason}) step={self.step} "
            f"assistant_tokens={self.sup_tokens:,} ckpt={self.ckpt_dir}",
            flush=True,
        )
        self._log_fh.close()
        return summary


# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="r52.posttrain.sft", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("config", help="path to an SFT YAML (see configs/posttrain/)")
    p.add_argument("--run-name", default=None, help="run directory name (default: the config's name)")
    p.add_argument("--init-from", default=None, help="override the base checkpoint directory")
    p.add_argument("--data-dir", default=None, help="override the SFT data directory")
    p.add_argument("-o", "--override", action="append", default=[], metavar="KEY=VALUE",
                   help="config override, e.g. -o max_steps=20 -o train.muon_lr=0.002")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    cfg = load_sft_config(args.config)
    if args.init_from:
        cfg.init_from = args.init_from
    if args.data_dir:
        cfg.data_dir = args.data_dir
    override(cfg, args.override)
    SFTTrainer(cfg, args.run_name).run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
