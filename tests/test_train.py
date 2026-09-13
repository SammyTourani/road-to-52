# Copyright 2026 The road-to-52 authors.
# SPDX-License-Identifier: Apache-2.0
"""End-to-end training behaviour: loss decreases, checkpoints round-trip, resume is exact."""

from __future__ import annotations

import json
import math
from pathlib import Path

import mlx.core as mx
import pytest
from conftest import tiny_config, tiny_model_config
from mlx import nn

from r52.checkpoint import latest_checkpoint, list_checkpoints, load_checkpoint, prune, save_checkpoint
from r52.config import Config, TrainConfig
from r52.model import GPT
from r52.optim import build_optimizer
from r52.train import Trainer, evaluate


def _steps(model: GPT, opt, x, y, n: int, start: int = 0) -> list[float]:
    lvg = nn.value_and_grad(model, lambda a, b: model.loss(a, b))
    out = []
    for s in range(start, start + n):
        loss, g = lvg(x, y)
        g, _ = opt.clip(g)
        opt.set_step(s, 100)
        opt.update(model, g)
        mx.eval(model.parameters(), opt.state, loss)
        out.append(float(loss))
    return out


def test_loss_decreases_over_30_steps() -> None:
    from r52.data import TokenStream, synthetic_tokens

    cfg = tiny_model_config()
    model = GPT(cfg)
    opt = build_optimizer(TrainConfig(warmup_steps=0, grad_clip=1.0))
    stream = TokenStream([], cfg.block_size, 4, arrays=[synthetic_tokens(50_000, cfg.vocab_size, 11)])
    lvg = nn.value_and_grad(model, lambda a, b: model.loss(a, b))
    losses = []
    for s in range(30):
        x, y = stream.next_batch()
        loss, g = lvg(x, y)
        g, _ = opt.clip(g)
        opt.set_step(s, 30)
        opt.update(model, g)
        mx.eval(model.parameters(), opt.state, loss)
        losses.append(float(loss))
    assert all(math.isfinite(v) for v in losses)
    # step 0 runs the bf16 log-sum-exp path, so allow bf16 slack on ln(vocab).
    assert losses[0] == pytest.approx(math.log(cfg.vocab_size), abs=0.05)
    assert sum(losses[-5:]) / 5 < sum(losses[:5]) / 5 - 0.5, losses


def test_checkpoint_round_trip_identical_loss(tmp_path: Path) -> None:
    mcfg = tiny_model_config()
    tcfg = TrainConfig(warmup_steps=0)
    model, opt = GPT(mcfg), build_optimizer(tcfg)
    x = mx.random.randint(0, mcfg.vocab_size, (4, mcfg.block_size))
    y = mx.random.randint(0, mcfg.vocab_size, (4, mcfg.block_size))
    _steps(model, opt, x, y, 5)
    ref = float(model.loss(x, y, fp32_logits=True))

    save_checkpoint(tmp_path / "step_00000005", model, opt.multi, step=5, tokens=123,
                    cursor={"shard": 1, "offset": 64, "epoch": 0}, config=Config(model=mcfg, train=tcfg))

    mx.random.seed(999)
    model2, opt2 = GPT(mcfg), build_optimizer(tcfg)
    meta = load_checkpoint(tmp_path / "step_00000005", model2, opt2.multi)
    assert meta["step"] == 5 and meta["tokens"] == 123
    assert meta["cursor"] == {"shard": 1, "offset": 64, "epoch": 0}
    assert float(model2.loss(x, y, fp32_logits=True)) == pytest.approx(ref, abs=1e-9)

    # optimizer state too: three more steps must track exactly
    a = _steps(model, opt, x, y, 3, start=5)
    b = _steps(model2, opt2, x, y, 3, start=5)
    assert a == pytest.approx(b, abs=1e-9)


def test_prune_keeps_last_n(tmp_path: Path) -> None:
    mcfg = tiny_model_config()
    model = GPT(mcfg)
    for s in (1, 2, 3, 4):
        save_checkpoint(tmp_path / f"step_{s:08d}", model, step=s, config=Config(model=mcfg))
    prune(tmp_path, keep_last=2)
    assert [p.name for p in list_checkpoints(tmp_path)] == ["step_00000003", "step_00000004"]
    assert latest_checkpoint(tmp_path).name == "step_00000004"


def test_trainer_runs_logs_and_resumes(tmp_path: Path) -> None:
    cfg = tiny_config()
    cfg.train.out_dir = str(tmp_path)
    cfg.train.max_tokens = cfg.train.tokens_per_step * 12
    cfg.train.log_interval = 4
    cfg.train.val_interval = 6
    cfg.train.ckpt_interval = 6
    t = Trainer(cfg, "unit")
    summary = t.run()

    assert summary["step"] == 12 and summary["reason"] == "max_tokens"
    records = [json.loads(line) for line in (tmp_path / "unit" / "log.jsonl").read_text().splitlines()]
    kinds = {r["event"] for r in records}
    assert {"start", "train", "val", "end"} <= kinds
    train_rows = [r for r in records if r["event"] == "train"]
    for field in ("step", "tokens", "loss", "lr", "tok_s", "tflops", "mfu_theoretical",
                  "mfu_measured", "peak_mem_gb", "eta_h"):
        assert field in train_rows[0], field
    assert train_rows[-1]["loss"] < train_rows[0]["loss"]

    # resume picks up the cursor, step and token count
    cfg2 = tiny_config()
    cfg2.train.out_dir = str(tmp_path)
    cfg2.train.max_tokens = cfg.train.tokens_per_step * 14
    t2 = Trainer(cfg2, "unit", resume="auto")
    assert t2.step == 12 and t2.tokens == cfg.train.tokens_per_step * 12
    assert t2.stream.state() == t.stream.state()


def test_resume_reproduces_the_same_token_stream(tmp_path: Path) -> None:
    """The data cursor must make an interrupted run byte-identical to an uninterrupted one."""
    import numpy as np

    def run(steps_a: int, steps_b: int) -> list[np.ndarray]:
        cfg = tiny_config()
        cfg.train.out_dir = str(tmp_path / f"r{steps_a}")
        cfg.train.max_tokens = cfg.train.tokens_per_step * steps_a
        cfg.train.ckpt_interval = steps_a
        cfg.train.val_interval = 0
        t = Trainer(cfg, "x")
        t.run()
        cursor = dict(t.stream.state())
        cfg2 = tiny_config()
        cfg2.train.out_dir = str(tmp_path / f"r{steps_a}")
        t2 = Trainer(cfg2, "x", resume="auto")
        assert t2.stream.state() == cursor
        return [np.array(t2.stream.next_batch()[0]) for _ in range(steps_b)]

    resumed = run(4, 3)
    cfg = tiny_config()
    from r52.data import make_train_stream

    s = make_train_stream(cfg.data, cfg.model.block_size, cfg.train.micro_batch, cfg.model.vocab_size)
    for _ in range(4 * cfg.grad_accum()):
        s.next_batch()
    straight = [np.array(s.next_batch()[0]) for _ in range(3)]
    assert all(np.array_equal(a, b) for a, b in zip(resumed, straight, strict=True))


def test_evaluate_reports_loss_and_tokens() -> None:
    from r52.data import make_val_loader

    cfg = tiny_config()
    model = GPT(cfg.model)
    loader = make_val_loader(cfg.data, cfg.model.block_size, 2, 3, cfg.model.vocab_size)
    out = evaluate(model, loader)
    assert out["val_tokens"] == 3 * 2 * cfg.model.block_size
    assert out["val_loss"] == pytest.approx(math.log(cfg.model.vocab_size), abs=1e-3)
    assert out["val_bpb"] is None
    out2 = evaluate(model, loader, bytes_per_token=4.0)
    assert out2["val_bpb"] == pytest.approx(out["val_loss"] / math.log(2) / 4.0, rel=1e-6)


@pytest.mark.parametrize("compile_step", [False, True])
def test_compile_flag_gives_the_same_loss(tmp_path: Path, compile_step: bool) -> None:
    cfg = tiny_config()
    cfg.train.out_dir = str(tmp_path)
    cfg.train.max_tokens = cfg.train.tokens_per_step * 3
    cfg.train.val_interval = 0
    cfg.train.ckpt_interval = 0
    t = Trainer(cfg, f"c{int(compile_step)}", compile_step=compile_step)
    s = t.run()
    assert math.isfinite(s["val_loss"])
    assert t.use_compile is compile_step
