# Copyright 2026 The road-to-52 authors.
# SPDX-License-Identifier: Apache-2.0
"""MLX **pretraining** throughput benchmark.

As of 2026-09 no published tokens/second benchmark exists for *training* on Apple
Silicon at any model size (``research/02-open-training-stack.md`` §Stage 9,
``research/05-apple-silicon-training.md`` §2.2 -- every circulating "MLX tok/s" figure is
decode-phase inference).  This module produces one.

What is measured
----------------
The **real training step**: ``grad_accum`` forward/backward micro-steps with fp32
gradient accumulation, a global-norm clip, and one Muon+AdamW optimizer update, exactly
as ``r52.train`` runs it.  Timing starts after warm-up and covers only *completed*
optimizer steps, so the optimizer cost is amortized honestly.

``tok_s`` counts every token in the batch.  ``tflops`` uses
``6 * N + 12 * n_layer * n_embd * T`` per token, where ``N`` is non-embedding params plus
``lm_head`` (:meth:`r52.model.GPT.num_params_flops`).  MFU is reported against both the
M4's 4.26 TFLOPS theoretical fp32 peak *and* the 3.6 TFLOPS bf16 4096^3 matmul peak
measured on this machine.

Each configuration runs in its own subprocess so an out-of-memory config is recorded as
``skipped`` instead of killing the sweep, and peak-memory readings never leak between rows.

The benchmark uses ``TrainConfig()`` defaults, which include ``grad_clip=1.0``.  The shipped
training configs turn clipping off (Muon already normalises hidden-matrix updates), so their
real peak is ~0.4-0.7 GiB *lower* than the corresponding row here -- these numbers are the
conservative end.

    python -m r52.bench --out results/mlx_pretrain_bench            # full sweep
    python -m r52.bench --sizes 30M,124M --seqs 1024 --precisions mixed
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

__all__ = ["SIZES", "BenchConfig", "BenchRow", "main", "read_results", "run_one", "sweep", "write_results"]

SIZES: dict[str, dict[str, int]] = {
    # label -> architecture.  Labels follow the non-embedding ("scaling") parameter
    # convention used in research/05 §5; the table prints all three parameter counts.
    "30M": {"n_layer": 8, "n_embd": 512, "n_head": 8},
    "60M": {"n_layer": 12, "n_embd": 640, "n_head": 10},
    "124M": {"n_layer": 12, "n_embd": 768, "n_head": 6},
    "350M": {"n_layer": 24, "n_embd": 1024, "n_head": 8},
}

MICRO_BATCH_LADDER = (16, 12, 8, 6, 4, 2, 1)

LOGITS_BUDGET_BYTES = 700 * 1024 * 1024
"""Cap on the ``(micro_batch, seq, vocab)`` logits tensor when auto-picking a micro-batch.

``mx.set_memory_limit`` is only a *guideline* -- MLX raises solely when RAM and swap are both
exhausted, so an oversized micro-batch does not fail fast, it swaps for minutes.  Bounding the
single largest activation up front keeps the ladder from ever starting somewhere hopeless.
"""


@dataclass
class BenchConfig:
    """One row of the sweep."""

    size: str = "124M"
    seq: int = 1024
    precision: str = "mixed"
    micro_batch: int = 0
    """0 = auto-pick the largest that fits under ``mem_limit_gib``."""
    compile: bool = True
    grad_checkpoint: bool = False
    tokens_per_step: int = 65_536
    """Optimizer-step batch in tokens; sets ``grad_accum`` and amortizes the optimizer."""
    seconds: float = 25.0
    warmup_steps: int = 2
    step_timeout: float = 120.0
    """Abandon a candidate whose first warm-up optimizer step exceeds this many seconds."""
    mem_limit_gib: float = 6.0
    vocab_size: int = 50_304
    use_value_embeds: bool = True
    n_value_embeds: int = 3
    value_embed_share: bool = True
    use_unet_skips: bool = True


@dataclass
class BenchRow:
    """A measured (or skipped) benchmark row. Units: tok/s, TFLOPS = 1e12 FLOP/s, GiB."""

    size: str
    seq: int
    precision: str
    micro_batch: int
    grad_accum: int
    compile: bool
    grad_checkpoint: bool
    n_layer: int = 0
    n_embd: int = 0
    n_head: int = 0
    params_total: int = 0
    params_non_embedding: int = 0
    params_flops: int = 0
    flops_per_token: int = 0
    tok_s: float | None = None
    tflops: float | None = None
    mfu_theoretical: float | None = None
    mfu_measured: float | None = None
    peak_mem_gib: float | None = None
    optimizer_steps: int = 0
    seconds: float | None = None
    status: str = "ok"
    note: str = ""


PEAK_TFLOPS_THEORETICAL = 4.26
"""Apple M4 10-core GPU theoretical fp32 peak (arXiv 2502.05317 Table 1), in 1e12 FLOP/s."""
PEAK_TFLOPS_MEASURED = 3.6
"""bf16 4096^3 matmul peak measured on this machine, 2026-09-13, in 1e12 FLOP/s."""

_REPO = Path(__file__).resolve().parent.parent
_FLOPS_FORMULA = "6 * (non_embedding_params + lm_head_params) + 12 * n_layer * n_embd * seq"
_GPU_CORES_CMD = (
    "system_profiler SPDisplaysDataType 2>/dev/null "
    "| awk -F': ' '/Total Number of Cores/{print $2; exit}'"
)


# --------------------------------------------------------------------------------------
# One configuration (runs in its own process)
# --------------------------------------------------------------------------------------


def _model_config(bc: BenchConfig):
    from .config import ModelConfig

    arch = SIZES[bc.size]
    return ModelConfig(
        n_layer=arch["n_layer"],
        n_embd=arch["n_embd"],
        n_head=arch["n_head"],
        vocab_size=bc.vocab_size,
        block_size=bc.seq,
        precision=bc.precision,
        grad_checkpoint=bc.grad_checkpoint,
        use_value_embeds=bc.use_value_embeds,
        n_value_embeds=bc.n_value_embeds,
        value_embed_share=bc.value_embed_share,
        use_unet_skips=bc.use_unet_skips,
    )


def _measure(bc: BenchConfig, mcfg, mb: int, accum: int) -> BenchRow:
    """Time one (micro_batch, grad_accum) point. Raises on OOM; never catches."""
    import mlx.core as mx
    import mlx.nn as nn
    from mlx.utils import tree_map

    from .config import TrainConfig
    from .model import GPT
    from .optim import build_optimizer

    row = BenchRow(
        size=bc.size, seq=bc.seq, precision=bc.precision, micro_batch=mb, grad_accum=accum,
        compile=bc.compile, grad_checkpoint=bc.grad_checkpoint,
        n_layer=mcfg.n_layer, n_embd=mcfg.n_embd, n_head=mcfg.n_head,
    )
    mx.clear_cache()
    mx.reset_peak_memory()
    mx.random.seed(0)

    model = GPT(mcfg)
    mx.eval(model.parameters())
    opt = build_optimizer(TrainConfig(), model.trainable_parameters())
    inv = 1.0 / accum

    def scaled(a: mx.array, b: mx.array) -> mx.array:
        return model.loss(a, b) * inv

    fn = nn.value_and_grad(model, scaled)
    if bc.compile:
        fn = mx.compile(fn, inputs=[model.state], outputs=[model.state])

    x = mx.random.randint(0, bc.vocab_size, (mb, bc.seq))
    y = mx.random.randint(0, bc.vocab_size, (mb, bc.seq))

    def one_step(step: int) -> None:
        """One optimizer step: `accum` fwd/bwd micro-steps, clip, Muon+AdamW update."""
        acc = None
        for _ in range(accum):
            _loss, grads = fn(x, y)
            acc = grads if acc is None else tree_map(lambda p, q: p + q, acc, grads)
            if accum > 1:
                mx.eval(acc)
        acc, _gnorm = opt.clip(acc)
        opt.set_step(step, 1000)
        opt.update(model, acc)
        mx.eval(model.parameters(), opt.state)

    for s in range(bc.warmup_steps):
        t_warm = time.perf_counter()
        one_step(s)
        warm_dt = time.perf_counter() - t_warm
        peak_warm = mx.get_peak_memory() / (1 << 30)
        if not bc.micro_batch and peak_warm > bc.mem_limit_gib:
            row.status = "skipped"
            row.note = f"peak {peak_warm:.2f} GiB > {bc.mem_limit_gib} GiB at micro_batch={mb}"
            return row
        if not bc.micro_batch and warm_dt > bc.step_timeout:
            row.status = "skipped"
            row.note = (f"warm-up step took {warm_dt:.0f}s > {bc.step_timeout:.0f}s at "
                        f"micro_batch={mb} (thrashing)")
            return row

    mx.reset_peak_memory()
    t0 = time.perf_counter()
    steps = 0
    while True:
        one_step(bc.warmup_steps + steps)
        steps += 1
        if time.perf_counter() - t0 >= bc.seconds:
            break
    dt = time.perf_counter() - t0

    tokens = steps * accum * mb * bc.seq
    fpt = model.flops_per_token(bc.seq)
    tok_s = tokens / dt
    tflops = tok_s * fpt / 1e12
    row.params_total = model.num_params()
    row.params_non_embedding = model.num_params(True)
    row.params_flops = model.num_params_flops()
    row.flops_per_token = fpt
    row.tok_s = round(tok_s, 1)
    row.tflops = round(tflops, 4)
    row.mfu_theoretical = round(tflops / PEAK_TFLOPS_THEORETICAL, 4)
    row.mfu_measured = round(tflops / PEAK_TFLOPS_MEASURED, 4)
    row.peak_mem_gib = round(mx.get_peak_memory() / (1 << 30), 3)
    row.optimizer_steps = steps
    row.seconds = round(dt, 2)
    row.status = "ok"
    return row


def run_one(bc: BenchConfig) -> BenchRow:
    """Measure a single configuration, walking the micro-batch ladder down until it fits."""
    import mlx.core as mx

    mx.set_memory_limit(int(bc.mem_limit_gib * (1 << 30)))
    mx.set_cache_limit(1 << 30)

    mcfg = _model_config(bc)
    if bc.micro_batch:
        candidates = [bc.micro_batch]
    else:
        logit_bytes = 4 if bc.precision == "fp32" else 2
        mb_cap = max(1, LOGITS_BUDGET_BYTES // (bc.seq * bc.vocab_size * logit_bytes))
        candidates = [
            m for m in MICRO_BATCH_LADDER
            if m * bc.seq <= bc.tokens_per_step and m <= mb_cap
        ] or [1]

    last_err = ""
    for mb in candidates:
        accum = max(1, bc.tokens_per_step // (mb * bc.seq))
        try:
            row = _measure(bc, mcfg, mb, accum)
        except Exception as exc:
            last_err = f"{type(exc).__name__}: {exc}"[:200]
            mx.clear_cache()
            continue
        if row.status == "ok":
            return row
        last_err = row.note
        mx.clear_cache()

    return BenchRow(
        size=bc.size, seq=bc.seq, precision=bc.precision, micro_batch=0, grad_accum=0,
        compile=bc.compile, grad_checkpoint=bc.grad_checkpoint,
        n_layer=mcfg.n_layer, n_embd=mcfg.n_embd, n_head=mcfg.n_head,
        status="skipped", note=last_err or "no micro-batch fits",
    )


# --------------------------------------------------------------------------------------
# Sweep driver (subprocess per row)
# --------------------------------------------------------------------------------------


def _run_in_subprocess(bc: BenchConfig, timeout: float) -> BenchRow:
    payload = json.dumps(asdict(bc))
    cmd = [sys.executable, "-m", "r52.bench", "--single", payload]
    env = dict(os.environ, PYTHONPATH=str(Path(__file__).resolve().parent.parent))
    try:
        out = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout, env=env,
            cwd=str(Path(__file__).resolve().parent.parent),
        )
    except subprocess.TimeoutExpired:
        return BenchRow(size=bc.size, seq=bc.seq, precision=bc.precision, micro_batch=0, grad_accum=0,
                        compile=bc.compile, grad_checkpoint=bc.grad_checkpoint,
                        status="skipped", note=f"timeout after {timeout:.0f}s")
    for line in reversed(out.stdout.splitlines()):
        if line.startswith("{"):
            return BenchRow(**json.loads(line))
    err = (out.stderr or out.stdout).strip().splitlines()
    return BenchRow(size=bc.size, seq=bc.seq, precision=bc.precision, micro_batch=0, grad_accum=0,
                    compile=bc.compile, grad_checkpoint=bc.grad_checkpoint,
                    status="skipped", note=(err[-1] if err else "subprocess produced no row")[:200])


def environment() -> dict[str, Any]:
    """Reproduction metadata attached to every results file."""
    import importlib.metadata as md

    def _v(pkg: str) -> str:
        try:
            return md.version(pkg)
        except Exception:  # pragma: no cover
            return "unknown"

    def _sh(cmd: list[str]) -> str:
        try:
            return subprocess.check_output(cmd, text=True, stderr=subprocess.DEVNULL).strip()
        except Exception:  # pragma: no cover
            return "unknown"

    return {
        "date": datetime.now(UTC).strftime("%Y-%m-%d"),
        "timestamp_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "chip": _sh(["sysctl", "-n", "machdep.cpu.brand_string"]),
        "gpu_cores": _sh(["sh", "-c", _GPU_CORES_CMD]),
        "memory_gib": round(int(_sh(["sysctl", "-n", "hw.memsize"]) or 0) / (1 << 30), 1),
        "macos": platform.mac_ver()[0],
        "python": platform.python_version(),
        "mlx": _v("mlx"),
        "mlx_lm": _v("mlx-lm"),
        "git_commit": _sh(["git", "-C", str(_REPO), "rev-parse", "--short", "HEAD"]),
        "peak_tflops_theoretical": PEAK_TFLOPS_THEORETICAL,
        "peak_tflops_measured": PEAK_TFLOPS_MEASURED,
        "flops_per_token_formula": _FLOPS_FORMULA,
    }


def sweep(configs: list[BenchConfig], timeout: float = 900.0, in_process: bool = False) -> list[BenchRow]:
    """Run every configuration and return the rows (progress goes to stderr)."""
    rows: list[BenchRow] = []
    for i, bc in enumerate(configs, 1):
        print(f"[{i}/{len(configs)}] {bc.size} seq={bc.seq} {bc.precision} "
              f"compile={bc.compile} ckpt={bc.grad_checkpoint} ...", file=sys.stderr, flush=True)
        row = run_one(bc) if in_process else _run_in_subprocess(bc, timeout)
        rows.append(row)
        if row.status == "ok":
            print(f"      mb={row.micro_batch} accum={row.grad_accum} {row.tok_s:.0f} tok/s "
                  f"{row.tflops:.3f} TF  mfu {100*row.mfu_theoretical:.1f}%/{100*row.mfu_measured:.1f}%  "
                  f"peak {row.peak_mem_gib:.2f} GiB", file=sys.stderr, flush=True)
        else:
            print(f"      SKIPPED: {row.note}", file=sys.stderr, flush=True)
    return rows


# --------------------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------------------

_COLUMNS = [
    ("size", "size"),
    ("n_layer", "L"),
    ("n_embd", "d"),
    ("params_flops", "N (6N)"),
    ("params_non_embedding", "non-emb"),
    ("params_total", "total"),
    ("seq", "seq"),
    ("micro_batch", "mb"),
    ("grad_accum", "accum"),
    ("precision", "precision"),
    ("compile", "compile"),
    ("grad_checkpoint", "ckpt"),
    ("tok_s", "tok/s"),
    ("tflops", "TFLOPS"),
    ("mfu_theoretical", "MFU 4.26"),
    ("mfu_measured", "MFU 3.6"),
    ("peak_mem_gib", "peak GiB"),
    ("status", "status"),
]


def _fmt(key: str, value: Any) -> str:
    if value is None:
        return "—"
    if key in ("mfu_theoretical", "mfu_measured"):
        return f"{100 * value:.1f}%"
    if key == "tok_s":
        return f"{value:,.0f}"
    if key == "tflops":
        return f"{value:.3f}"
    if key == "peak_mem_gib":
        return f"{value:.2f}"
    if key in ("params_total", "params_non_embedding", "params_flops"):
        return f"{value / 1e6:.1f}M"
    if isinstance(value, bool):
        return "yes" if value else "no"
    return str(value)


def to_markdown(rows: list[BenchRow], env: dict[str, Any]) -> str:
    """Render the results table (the block that gets pasted into the README)."""
    batch_sizes = "/".join(
        sorted({str(r.grad_accum * r.micro_batch * r.seq) for r in rows if r.status == "ok"})
    ) or "-"
    lines = [
        "# MLX pretraining throughput benchmark",
        "",
        f"**{env['chip']}** ({env['gpu_cores']} GPU cores, {env['memory_gib']} GiB unified) "
        f"· macOS {env['macos']} · mlx {env['mlx']} · mlx-lm {env['mlx_lm']} "
        f"· Python {env['python']} · commit `{env['git_commit']}` · {env['date']}",
        "",
        "Measured by `python -m r52.bench` -- a full training step (gradient accumulation "
        f"to {batch_sizes} tokens, fp32 grad accumulation, global-norm clip, Muon+AdamW "
        "update), timed over completed optimizer steps only, after warm-up.",
        "",
        f"`FLOPs/token = {env['flops_per_token_formula']}`. MFU denominators: "
        f"**{env['peak_tflops_theoretical']} TFLOPS** (M4 theoretical fp32 peak, arXiv "
        f"2502.05317 Table 1) and **{env['peak_tflops_measured']} TFLOPS** (bf16 4096^3 "
        "matmul measured on this machine).",
        "",
        "| " + " | ".join(h for _, h in _COLUMNS) + " |",
        "|" + "|".join("---" for _ in _COLUMNS) + "|",
    ]
    for r in rows:
        d = asdict(r)
        lines.append("| " + " | ".join(_fmt(k, d[k]) for k, _ in _COLUMNS) + " |")
    notes = [f"- `{r.size}/{r.seq}/{r.precision}`: {r.note}" for r in rows if r.status != "ok" and r.note]
    if notes:
        lines += ["", "### Skipped configurations", "", *notes]
    lines.append("")
    return "\n".join(lines)


def _row_key(r: BenchRow) -> tuple:
    return (r.size, r.seq, r.precision, r.compile, r.grad_checkpoint)


def merge_rows(existing: list[BenchRow], new: list[BenchRow]) -> list[BenchRow]:
    """Replace same-configuration rows, keep the rest, ordered by size/seq/precision."""
    by_key = {_row_key(r): r for r in existing}
    by_key.update({_row_key(r): r for r in new})
    order = {name: i for i, name in enumerate(SIZES)}
    return sorted(by_key.values(),
                  key=lambda r: (order.get(r.size, 99), r.seq, r.precision, not r.compile,
                                 r.grad_checkpoint))


def read_results(out_base: str | Path) -> list[BenchRow]:
    """Read previously written rows back (empty list if the file is absent)."""
    path = Path(out_base).with_suffix(".json")
    if not path.exists():
        return []
    data = json.loads(path.read_text())
    fields = set(BenchRow.__dataclass_fields__)
    return [BenchRow(**{k: v for k, v in r.items() if k in fields}) for r in data.get("rows", [])]


def write_results(rows: list[BenchRow], out_base: str | Path, append: bool = False) -> tuple[Path, Path]:
    """Write ``<out_base>.json`` and ``<out_base>.md``; returns both paths.

    With ``append=True`` the new rows are merged into whatever is already on disk, replacing
    rows for the same (size, seq, precision, compile, grad_checkpoint) configuration -- so a
    long sweep can be run in chunks.
    """
    base = Path(out_base)
    base.parent.mkdir(parents=True, exist_ok=True)
    if append:
        rows = merge_rows(read_results(base), rows)
    env = environment()
    json_path = base.with_suffix(".json")
    md_path = base.with_suffix(".md")
    json_path.write_text(json.dumps({"environment": env, "rows": [asdict(r) for r in rows]}, indent=2))
    md_path.write_text(to_markdown(rows, env))
    return json_path, md_path


# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="r52.bench", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--sizes", default="30M,60M,124M,350M")
    p.add_argument("--seqs", default="512,1024")
    p.add_argument("--precisions", default="mixed,bf16,fp32")
    p.add_argument("--seconds", type=float, default=25.0, help="measurement window per row (s)")
    p.add_argument("--tokens-per-step", type=int, default=65_536,
                   help="optimizer-step batch in tokens (sets grad_accum)")
    p.add_argument("--mem-limit-gib", type=float, default=6.0)
    p.add_argument("--micro-batch", type=int, default=0, help="0 = auto")
    p.add_argument("--no-compile", action="store_true")
    p.add_argument("--grad-checkpoint", action="store_true")
    p.add_argument("--timeout", type=float, default=900.0, help="per-row subprocess timeout (s)")
    p.add_argument("--step-timeout", type=float, default=120.0,
                   help="abandon a micro-batch whose warm-up step exceeds this many seconds")
    p.add_argument("--in-process", action="store_true", help="do not fork a subprocess per row")
    p.add_argument("--out", default="results/mlx_pretrain_bench")
    p.add_argument("--append", action="store_true",
                   help="merge into the existing results file instead of replacing it")
    p.add_argument("--single", default=None, help=argparse.SUPPRESS)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.single:
        row = run_one(BenchConfig(**json.loads(args.single)))
        print(json.dumps(asdict(row)), flush=True)
        return 0

    configs = [
        BenchConfig(
            size=s, seq=int(q), precision=p, micro_batch=args.micro_batch,
            compile=not args.no_compile, grad_checkpoint=args.grad_checkpoint,
            tokens_per_step=args.tokens_per_step, seconds=args.seconds,
            mem_limit_gib=args.mem_limit_gib, step_timeout=args.step_timeout,
        )
        for s in args.sizes.split(",")
        for q in args.seqs.split(",")
        for p in args.precisions.split(",")
    ]
    rows = sweep(configs, timeout=args.timeout, in_process=args.in_process)
    jp, mp = write_results(rows, args.out, append=args.append)
    print(f"\nwrote {jp} and {mp}")
    print(mp.read_text())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
