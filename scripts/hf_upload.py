#!/usr/bin/env python3
# Copyright 2026 The road-to-52 authors.
# SPDX-License-Identifier: Apache-2.0
"""Assemble an honest Hugging Face model card and (optionally) publish an r52 export.

    python scripts/hf_upload.py --export-dir models/<name>-mlx --run <run-name> \\
        --repo-id SammyTourani/road-to-52-<name> [--private] [--dry-run]

Every number in the card is pulled from files already in this repo -- never invented:

* ``runs/<run>/log.jsonl``     -- the run config (first line), final step/tokens/wall-clock/tok-s
                                   (scanned from every later line).
* ``results/<run>/*.json``     -- every eval this run has (schema: ``r52/eval/report.py``).
* ``results/gpt2-124m-reference/*.json`` -- the GPT-2 124M baseline, measured with the same code.
* ``results/ladder.json``      -- the ladder rung whose id/name contains ``--run``, if any.
* ``<export-dir>/config.json`` -- architecture (written by ``r52.export``; falls back to the
                                   plain HF-style keys for the GPT-2 reference conversion, which
                                   goes through ``mlx_lm.convert`` instead and carries no "r52" key).
* ``<export-dir>/model.safetensors`` -- a parameter-count cross-check / fallback, computed by
                                   reading only the safetensors header (no weights are loaded).

Any field with no data source renders as "--" (:func:`fmt_num`), never a fabricated number.

``--dry-run`` writes the assembled card to ``results/<run>/MODEL_CARD.md``, prints the file list
that would be uploaded, and makes no network call. Without ``--dry-run`` a Hugging Face token is
required (``HF_TOKEN`` env var, or a prior ``hf auth login``); the run then does
``HfApi.create_repo(exist_ok=True)`` followed by one ``HfApi.upload_folder`` of a staged directory
containing the export dir's own files, ``results/<run>/*.json``, and the card as ``README.md``.
``runs/*/ckpt/`` (checkpoints *and* their optimizer state) is never read or uploaded -- only the
already-exported model directory.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import struct
import sys
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from r52.eval.report import git_commit, machine_string  # noqa: E402

__all__ = [
    "assemble_model_card",
    "build_context",
    "collect_upload_files",
    "count_params_from_safetensors",
    "load_architecture",
    "load_eval_results",
    "load_ladder_rung",
    "load_run_log",
    "main",
    "render_template",
    "resolve_token",
]

DEFAULT_TEMPLATE_PATH = REPO_ROOT / "docs" / "MODEL_CARD_TEMPLATE.md"
HARDWARE_DESC = "Mac Mini M4, 16 GB unified memory, Apple M4 10-core GPU"
REPO_URL = "https://github.com/SammyTourani/road-to-52"
REFERENCE_RUN = "gpt2-124m-reference"

_EXCLUDED_DIR_NAMES = {"ckpt", "__pycache__", ".git"}
_EXCLUDED_FILE_NAMES = {".DS_Store"}
_PLACEHOLDER_RE = re.compile(r"\{\{(\w+)\}\}")
_RUNG_ID_RE = re.compile(r"^rung-([0-9a-zA-Z]+)")

# --------------------------------------------------------------------------------------
# Formatting -- the single place "missing data" becomes "--", never 0 / None / nan.
# --------------------------------------------------------------------------------------


def fmt_num(value: Any) -> str:
    """Render a value for the card. ``None`` -> ``"--"``; never a fabricated number."""
    if value is None:
        return "—"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, int):
        return f"{value:,}"
    if isinstance(value, float):
        if value != 0 and abs(value) < 1e-3:
            return f"{value:.2e}"
        if value == int(value):
            return f"{value:.1f}"
        return f"{value:.4g}"
    return str(value)


def esc_pipe(text: str) -> str:
    return text.replace("|", "\\|")


def fmt_eval_value(value: float, unit: str) -> str:
    """Mirrors ``r52.eval.report._format_value`` so cards and ``docs/RESULTS.md`` agree."""
    if unit.startswith("%"):
        return f"{value:.2f} {unit}"
    if unit == "CORE":
        return f"{value:.6f} {unit}"
    return f"{value:.4f} {unit}"


def fmt_wall_clock(seconds: float | None) -> str:
    if seconds is None:
        return "—"
    hours = seconds / 3600
    days = hours / 24
    return f"{seconds:,.0f} s ({hours:.2f} h, {days:.2f} d)"


# --------------------------------------------------------------------------------------
# Data sources
# --------------------------------------------------------------------------------------


def count_params_from_safetensors(path: Path) -> int | None:
    """Sum tensor shapes straight from the safetensors header -- no weights are loaded."""
    if not path.is_file():
        return None
    try:
        with path.open("rb") as fh:
            header_len = struct.unpack("<Q", fh.read(8))[0]
            header = json.loads(fh.read(header_len))
    except (OSError, struct.error, json.JSONDecodeError):
        return None
    total = 0
    for key, meta in header.items():
        if key == "__metadata__" or not isinstance(meta, dict):
            continue
        shape = meta.get("shape") or []
        n = 1
        for s in shape:
            n *= int(s)
        total += n
    return total or None


def load_run_log(run: str, root: Path) -> dict[str, Any] | None:
    """Parse ``runs/<run>/log.jsonl``: the ``start`` record, the latest known metrics scanned
    across every later line, and the ``end`` record if training has finished."""
    log_path = root / "runs" / run / "log.jsonl"
    if not log_path.is_file():
        return None
    lines = [ln for ln in log_path.read_text().splitlines() if ln.strip()]
    if not lines:
        return None
    try:
        start = json.loads(lines[0])
    except json.JSONDecodeError:
        return None

    latest: dict[str, Any] = {}
    end: dict[str, Any] | None = None
    watched = ("step", "tokens", "t", "tok_s", "tflops", "mfu_theoretical", "mfu_measured", "peak_mem_gb")
    for ln in lines[1:]:
        try:
            rec = json.loads(ln)
        except json.JSONDecodeError:
            continue
        if rec.get("event") == "end":
            end = rec
            # The end event names throughput fields "last_tok_s" / "last_tflops".
            rec = {**rec, "tok_s": rec.get("tok_s", rec.get("last_tok_s")),
                   "tflops": rec.get("tflops", rec.get("last_tflops"))}
        for key in watched:
            if rec.get(key) is not None:
                latest[key] = rec[key]
    return {"start": start, "latest": latest, "end": end}


def load_eval_results(run: str, root: Path) -> list[dict[str, Any]]:
    """Every ``results/<run>/*.json`` that is a single eval record (not the aggregate list
    ``eval.json``, which is a flat list and is skipped here -- see ``r52/eval/report.py``)."""
    d = root / "results" / run
    if not d.is_dir():
        return []
    out = []
    for p in sorted(d.glob("*.json")):
        try:
            payload = json.loads(p.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        if (
            isinstance(payload, dict)
            and isinstance(payload.get("conditions"), dict)
            and {"benchmark", "value", "unit"} <= payload.keys()
        ):
            out.append(payload)
    out.sort(key=lambda r: (str(r.get("benchmark")), str(r.get("unit"))))
    return out


def load_ladder_rung(run: str, root: Path) -> dict[str, Any] | None:
    """The ``results/ladder.json`` rung whose id or name contains ``run``, if any."""
    path = root / "results" / "ladder.json"
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    run_l = run.strip().lower()
    if not run_l:
        return None
    for rung in data.get("rungs", []) or []:
        rid = str(rung.get("id", "")).lower()
        name = str(rung.get("name", "")).lower()
        if run_l in rid or run_l in name:
            return rung
    return None


def load_architecture(export_dir: Path) -> dict[str, Any] | None:
    """Normalize ``<export-dir>/config.json`` into architecture fields.

    Prefers the full :class:`r52.config.ModelConfig` carried under ``config["r52"]["model"]``
    (every ``r52.export`` output has this); falls back to plain HF-style keys for a directory
    that did not come from ``r52.export`` (e.g. the GPT-2 reference, converted by
    ``mlx_lm.convert`` -- see ``r52/export.py:gpt2_reference``), leaving whatever does not exist
    in that architecture (GQA, RMSNorm eps, value embeddings, ...) as ``None``.
    """
    conf_path = export_dir / "config.json"
    if not conf_path.is_file():
        return None
    try:
        conf = json.loads(conf_path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(conf, dict):
        return None

    r52 = conf.get("r52")
    if isinstance(r52, dict) and isinstance(r52.get("model"), dict):
        m = r52["model"]
        return {
            "model_type": conf.get("model_type"),
            "n_layer": m.get("n_layer"), "n_embd": m.get("n_embd"), "n_head": m.get("n_head"),
            "n_kv_head": m.get("n_kv_head"), "head_dim": m.get("head_dim"),
            "vocab_size": m.get("vocab_size"), "block_size": m.get("block_size"),
            "mlp": m.get("mlp"), "mlp_ratio": m.get("mlp_ratio"), "qk_norm": m.get("qk_norm"),
            "use_value_embeds": m.get("use_value_embeds"), "n_value_embeds": m.get("n_value_embeds"),
            "use_unet_skips": m.get("use_unet_skips"), "softcap": m.get("softcap"),
            "norm_eps": m.get("norm_eps"),
        }

    return {
        "model_type": conf.get("model_type"),
        "n_layer": conf.get("n_layer", conf.get("num_hidden_layers")),
        "n_embd": conf.get("n_embd", conf.get("hidden_size")),
        "n_head": conf.get("n_head", conf.get("num_attention_heads")),
        "n_kv_head": conf.get("num_key_value_heads"),
        "head_dim": conf.get("head_dim"),
        "vocab_size": conf.get("vocab_size"),
        "block_size": conf.get("n_positions", conf.get("n_ctx", conf.get("max_position_embeddings"))),
        "mlp": None, "mlp_ratio": None, "qk_norm": None,
        "use_value_embeds": None, "n_value_embeds": None, "use_unet_skips": None,
        "softcap": None, "norm_eps": conf.get("layer_norm_epsilon"),
    }


def resolve_param_counts(run_log: dict[str, Any] | None, export_dir: Path) -> tuple[int | None, int | None]:
    """``(total, non_embedding)``. Prefers the training log; falls back to summing the
    exported ``model.safetensors`` header (no non-embedding split is derivable generically)."""
    if run_log is not None:
        total = run_log["start"].get("params_total")
        if total is not None:
            return total, run_log["start"].get("params_non_embedding")
    return count_params_from_safetensors(export_dir / "model.safetensors"), None


def find_config_file(name: str | None, root: Path) -> str | None:
    if not name:
        return None
    configs_dir = root / "configs"
    if not configs_dir.is_dir():
        return None
    matches = sorted(configs_dir.rglob(f"{name}.yaml"))
    return matches[0].relative_to(root).as_posix() if matches else None


# --------------------------------------------------------------------------------------
# Rendering: raw data -> card-ready strings
# --------------------------------------------------------------------------------------


def _eval_row(rec: dict[str, Any]) -> str:
    cond = rec.get("conditions") or {}
    value = fmt_eval_value(float(rec["value"]), str(rec.get("unit", "")))
    n_examples, limit = cond.get("n_examples"), cond.get("limit")
    count = fmt_num(n_examples) if n_examples is not None else (f"limit {limit}" if limit else "—")
    tokenizer = cond.get("tokenizer") or "—"
    block_size = fmt_num(cond.get("block_size")) if cond.get("block_size") is not None else "—"
    commit = rec.get("commit") or "—"
    command = esc_pipe(rec.get("command") or cond.get("command") or "—")
    benchmark = rec.get("benchmark", "—")
    return (
        f"| {benchmark} | {value} | {count} | {tokenizer} | {block_size} | "
        f"`{commit}` | `{command}` |"
    )


def render_eval_rows(records: list[dict[str, Any]], empty_note: str) -> str:
    if not records:
        return f"| — | — | — | — | — | — | {empty_note} |"
    return "\n".join(_eval_row(r) for r in records)


def rung_label(rung: dict[str, Any] | None) -> str:
    if rung is None:
        return "—"
    m = _RUNG_ID_RE.match(str(rung.get("id", "")))
    return f"Rung {m.group(1).upper()}" if m else (rung.get("name") or "—")


def render_ladder_row(rung: dict[str, Any] | None) -> str:
    if rung is None:
        return "— (no matching row in `results/ladder.json` for this run)"
    targets = rung.get("targets") or {}
    targets_s = "; ".join(f"{k} {v}" for k, v in targets.items()) if targets else "—"
    tokens = (rung.get("inputs") or {}).get("tokens")
    return "\n".join([
        f"- **{rung.get('name', '—')}** (`{rung.get('id', '—')}`)",
        f"- Beats: {rung.get('beats') or '—'}",
        f"- Status: {rung.get('status') or '—'}",
        f"- Token budget: {fmt_num(tokens)}",
        f"- Targets: {targets_s}",
    ])


def render_what_this_is(run: str, rung: dict[str, Any] | None) -> str:
    if rung is not None:
        label, beats = rung_label(rung), (rung.get("beats") or "—")
        return (
            f"This is **{run}**, {label} of the [road-to-52]({REPO_URL}) capability-per-dollar "
            "ladder -- an open, Apple-Silicon-native LLM build stack (see the project README and "
            f"`docs/PLAN.md`). {label} targets beating **{beats}**, measured with this repo's own "
            "evaluation code. road-to-52 does not claim to beat the frontier (Claude Fable 5.1) -- "
            "its honest premise is that a Mac Mini cannot reach frontier-lab compute; what it *can* "
            "do is the recipe, the ladder, and honest measurement. See `docs/RESULTS.md` for every "
            "number this card reports."
        )
    return (
        f"This is **{run}**, a checkpoint from the [road-to-52]({REPO_URL}) project -- an open, "
        "Apple-Silicon-native LLM build stack and capability-per-dollar ladder. It is not currently "
        "matched to a numbered rung in `results/ladder.json`. See `docs/RESULTS.md` and "
        "`docs/PLAN.md` for the project's honest framing."
    )


def render_run_status(run_log: dict[str, Any] | None) -> str:
    if run_log is None:
        return "no training log found at `runs/<run>/log.jsonl` for this run id."
    start, latest, end = run_log["start"], run_log["latest"], run_log["end"]
    total_steps = start.get("total_steps")
    step = latest.get("step", start.get("step"))
    tokens, max_tokens = latest.get("tokens"), start.get("max_tokens")
    if end is not None:
        best_val = end.get("best_val")
        best_val_s = f", best val loss {best_val:.4f}" if isinstance(best_val, int | float) else ""
        return (
            f"training complete -- stopped at step {fmt_num(step)}/{fmt_num(total_steps)} "
            f"({fmt_num(tokens)} tokens), reason: {end.get('reason') or '—'}{best_val_s}."
        )
    return (
        f"training in progress as of this card's generation -- step {fmt_num(step)}/{fmt_num(total_steps)} "
        f"({fmt_num(tokens)}/{fmt_num(max_tokens)} tokens). Numbers below are a snapshot, not final."
    )


def render_dataset(cfg_data: dict[str, Any] | None) -> tuple[str, str]:
    if not cfg_data:
        return "—", "—"
    if cfg_data.get("source") == "synthetic":
        return (
            "synthetic (no external dataset)",
            "deterministic synthetic noise tokens, pipeline smoke-test only, not real text "
            f"({fmt_num(cfg_data.get('synthetic_tokens'))} tokens) -- seed {fmt_num(cfg_data.get('seed'))}.",
        )
    repo_id = cfg_data.get("repo_id") or "—"
    desc = (
        f"`{cfg_data.get('source') or '—'}` shards from `{repo_id}` (`{cfg_data.get('train_glob') or '—'}`); "
        f"validation on `{cfg_data.get('val_file') or '—'}` ({fmt_num(cfg_data.get('val_tokens'))} tokens)."
    )
    return repo_id, desc


def render_optimizer(cfg_train: dict[str, Any] | None) -> str:
    if not cfg_train:
        return "—"
    muon_lr = fmt_num(cfg_train.get("muon_lr"))
    muon_mom = fmt_num(cfg_train.get("muon_momentum"))
    muon_nesterov = fmt_num(cfg_train.get("muon_nesterov"))
    lr_embed = fmt_num(cfg_train.get("adam_lr_embed"))
    lr_head = fmt_num(cfg_train.get("adam_lr_head"))
    lr_scalar = fmt_num(cfg_train.get("adam_lr_scalar"))
    return (
        f"Muon (lr={muon_lr}, momentum={muon_mom}, nesterov={muon_nesterov}) on 2-D "
        f"hidden-layer matrices; AdamW on embeddings/value-embeddings/head/scalars "
        f"(lr_embed={lr_embed}, lr_head={lr_head}, lr_scalar={lr_scalar})."
    )


def render_schedule(cfg_train: dict[str, Any] | None) -> str:
    if not cfg_train:
        return "—"
    cooldown, final_frac = cfg_train.get("cooldown_frac"), cfg_train.get("final_lr_frac")
    cooldown_s = f"{cooldown * 100:.0f}%" if isinstance(cooldown, int | float) else "—"
    final_s = f"{final_frac * 100:.0f}%" if isinstance(final_frac, int | float) else "—"
    return (
        f"warmup {fmt_num(cfg_train.get('warmup_steps'))} steps, cooldown {cooldown_s} of the run, "
        f"final LR {final_s} of peak, grad-clip {fmt_num(cfg_train.get('grad_clip'))}."
    )


def render_throughput(latest: dict[str, Any]) -> str:
    tok_s = latest.get("tok_s")
    if tok_s is None:
        return "—"
    parts = [f"{fmt_num(tok_s)} tok/s"]
    if latest.get("tflops") is not None:
        parts.append(f"{latest['tflops']:.3f} TFLOPS")
    if latest.get("mfu_measured") is not None:
        parts.append(f"{latest['mfu_measured'] * 100:.1f}% MFU (measured bf16 peak)")
    if latest.get("mfu_theoretical") is not None:
        parts.append(f"{latest['mfu_theoretical'] * 100:.1f}% MFU (theoretical fp32 peak)")
    return ", ".join(parts)


# --------------------------------------------------------------------------------------
# Context assembly + template rendering
# --------------------------------------------------------------------------------------


def build_context(
    *, export_dir: Path, run: str, repo_id: str, root: Path,
    base_model: str | None, commit: str, machine: str, generated_date: str,
) -> dict[str, str]:
    run_log = load_run_log(run, root)
    start = run_log["start"] if run_log else {}
    latest = run_log["latest"] if run_log else {}
    cfg = start.get("config") or {}

    arch = load_architecture(export_dir) or {}
    params_total, params_non_embedding = resolve_param_counts(run_log, export_dir)

    rung = load_ladder_rung(run, root)
    evals = load_eval_results(run, root)
    if run == REFERENCE_RUN:
        reference_evals: list[dict[str, Any]] = []
        reference_note = "this run *is* the GPT-2 124M reference measurement (see the table above)."
    else:
        reference_evals = load_eval_results(REFERENCE_RUN, root)
        reference_note = f"no `results/{REFERENCE_RUN}/` eval files found."

    dataset_id, dataset_desc = render_dataset(cfg.get("data"))

    try:
        export_dir_s = export_dir.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        export_dir_s = str(export_dir)

    return {
        "base_model_yaml": f"base_model: {base_model}\n" if base_model else "",
        "run_name": run,
        "repo_id": repo_id,
        "generated_date": generated_date,
        "run_status": render_run_status(run_log),
        "what_this_is": render_what_this_is(run, rung),
        "dataset_id": dataset_id,
        "dataset_desc": dataset_desc,
        "tokens_budget": fmt_num(start.get("max_tokens")),
        "tokens_seen": fmt_num(latest.get("tokens")),
        "arch_model_type": arch.get("model_type") or "—",
        "arch_n_layer": fmt_num(arch.get("n_layer")),
        "arch_n_embd": fmt_num(arch.get("n_embd")),
        "arch_n_head": fmt_num(arch.get("n_head")),
        "arch_n_kv_head": fmt_num(arch.get("n_kv_head")),
        "arch_head_dim": fmt_num(arch.get("head_dim")),
        "arch_vocab_size": fmt_num(arch.get("vocab_size")),
        "arch_block_size": fmt_num(arch.get("block_size")),
        "arch_mlp": arch.get("mlp") or "—",
        "arch_mlp_ratio": fmt_num(arch.get("mlp_ratio")),
        "arch_qk_norm": fmt_num(arch.get("qk_norm")),
        "arch_use_value_embeds": fmt_num(arch.get("use_value_embeds")),
        "arch_n_value_embeds": fmt_num(arch.get("n_value_embeds")),
        "arch_use_unet_skips": fmt_num(arch.get("use_unet_skips")),
        "arch_softcap": fmt_num(arch.get("softcap")),
        "arch_norm_eps": fmt_num(arch.get("norm_eps")),
        "arch_params_total": fmt_num(params_total),
        "arch_params_non_embedding": fmt_num(params_non_embedding),
        "final_step": fmt_num(latest.get("step", start.get("step"))),
        "total_steps": fmt_num(start.get("total_steps")),
        "tokens_per_step": fmt_num(start.get("tokens_per_step")),
        "optimizer_desc": render_optimizer(cfg.get("train")),
        "schedule_desc": render_schedule(cfg.get("train")),
        "precision": start.get("precision") or "—",
        "wall_clock": fmt_wall_clock(latest.get("t")),
        "hardware": HARDWARE_DESC,
        "throughput": render_throughput(latest),
        "mlx_version": start.get("mlx_version") or "—",
        "evaluation_rows": render_eval_rows(evals, "no eval results found under `results/<run>/`"),
        "reference_rows": render_eval_rows(reference_evals, reference_note),
        "ladder_row": render_ladder_row(rung),
        "export_dir": export_dir_s,
        "config_file": find_config_file(cfg.get("name"), root) or "—",
        "commit": commit,
        "machine": machine,
    }


def render_template(template_text: str, context: dict[str, str]) -> str:
    missing: list[str] = []

    def _sub(m: re.Match[str]) -> str:
        key = m.group(1)
        if key not in context:
            missing.append(key)
            return "—"
        return context[key]

    rendered = _PLACEHOLDER_RE.sub(_sub, template_text)
    if missing:
        print(f"[hf_upload] warning: template has unfillable placeholders: {sorted(set(missing))}",
              file=sys.stderr)
    return rendered


def assemble_model_card(
    *, export_dir: Path, run: str, repo_id: str,
    root: Path = REPO_ROOT, template_path: Path = DEFAULT_TEMPLATE_PATH,
    base_model: str | None = None,
    commit: str | None = None, machine: str | None = None, generated_date: str | None = None,
) -> str:
    """Fill ``docs/MODEL_CARD_TEMPLATE.md`` from files under ``root``. Pure / no side effects."""
    context = build_context(
        export_dir=export_dir, run=run, repo_id=repo_id, root=root, base_model=base_model,
        commit=commit if commit is not None else git_commit(root),
        machine=machine if machine is not None else machine_string(),
        generated_date=generated_date or datetime.now(UTC).strftime("%Y-%m-%d"),
    )
    return render_template(template_path.read_text(), context)


# --------------------------------------------------------------------------------------
# Upload file list -- export dir + this run's eval JSONs. Never runs/*/ckpt/.
# --------------------------------------------------------------------------------------


def collect_upload_files(export_dir: Path, run: str, root: Path) -> list[tuple[Path, str]]:
    """``(local_path, path_in_repo)`` pairs for everything ``upload()`` would push.

    Only ``export_dir``'s own files (recursively) plus ``results/<run>/*.json`` -- a ``ckpt/``
    path segment anywhere under ``export_dir`` (a checkpoint's weights *and* its optimizer
    state) is always excluded, even if someone mistakenly points ``--export-dir`` at one. A
    top-level ``README.md`` inside ``export_dir`` (e.g. the stub ``mlx_lm.convert`` writes for
    the GPT-2 reference conversion) is excluded too: ``upload()`` always stages the assembled
    model card as ``README.md`` last, so an export dir's own copy would just be silently
    overwritten -- excluding it here keeps this file list equal to what actually ends up in
    the repo, in both --dry-run and a real upload.
    """
    files: list[tuple[Path, str]] = []
    for p in sorted(export_dir.rglob("*")):
        if not p.is_file():
            continue
        rel = p.relative_to(export_dir)
        if any(part in _EXCLUDED_DIR_NAMES for part in rel.parts[:-1]) or rel.name in _EXCLUDED_FILE_NAMES:
            continue
        if rel == Path("README.md"):
            continue
        files.append((p, rel.as_posix()))
    if not files:
        raise ValueError(
            f"no exportable files found under --export-dir {export_dir} after excluding "
            "checkpoint/optimizer paths -- pass an mlx-lm export directory (see `r52.export`), "
            "not a raw runs/*/ckpt/step_* checkpoint"
        )

    eval_dir = root / "results" / run
    if eval_dir.is_dir():
        files.extend((p, f"eval/{run}/{p.name}") for p in sorted(eval_dir.glob("*.json")))
    return files


# --------------------------------------------------------------------------------------
# Network (only reached without --dry-run)
# --------------------------------------------------------------------------------------


def resolve_token() -> str | None:
    token = os.environ.get("HF_TOKEN")
    if token:
        return token
    try:
        from huggingface_hub import get_token
    except ImportError:  # pragma: no cover - huggingface_hub is a hard dependency
        return None
    try:
        return get_token()
    except Exception:
        return None


def upload(
    *, export_dir: Path, run: str, repo_id: str, root: Path,
    card_text: str, private: bool, token: str,
) -> int:
    """Stage export_dir + results/<run>/*.json + the card into one temp dir, then one
    ``create_repo`` + one ``upload_folder`` call. Returns the number of files uploaded."""
    from huggingface_hub import HfApi

    files = collect_upload_files(export_dir, run, root)
    api = HfApi(token=token)
    api.create_repo(repo_id=repo_id, private=private, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="r52-hf-upload-") as tmp:
        stage = Path(tmp)
        for local_path, path_in_repo in files:
            dest = stage / path_in_repo
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(local_path, dest)
        (stage / "README.md").write_text(card_text)
        api.upload_folder(
            repo_id=repo_id, folder_path=str(stage), path_in_repo=".",
            commit_message=f"road-to-52: {run}",
        )
    return len(files) + 1


# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="hf_upload", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--export-dir", required=True, type=Path,
                   help="mlx-lm export directory, e.g. models/<name>-mlx (see r52.export)")
    p.add_argument("--run", required=True,
                   help="run name -- looks up runs/<run>/log.jsonl and results/<run>/")
    p.add_argument("--repo-id", required=True,
                   help="Hugging Face repo id, e.g. SammyTourani/road-to-52-<name>")
    p.add_argument("--private", action="store_true", help="create the repo as private")
    p.add_argument("--dry-run", action="store_true",
                   help="write results/<run>/MODEL_CARD.md and print the file list; no network")
    p.add_argument("--base-model", default=None,
                   help="base model repo id for a post-trained variant (omit: from scratch)")
    return p


def main(
    argv: list[str] | None = None, *, root: Path | None = None, template_path: Path | None = None
) -> int:
    """``root`` / ``template_path`` are override hooks for tests; a normal CLI invocation
    leaves both at their real-repo defaults."""
    root = root or REPO_ROOT
    template_path = template_path or DEFAULT_TEMPLATE_PATH
    args = build_parser().parse_args(argv)

    export_dir: Path = args.export_dir
    if not export_dir.is_dir():
        print(f"error: --export-dir {export_dir} is not a directory", file=sys.stderr)
        return 2

    card_text = assemble_model_card(
        export_dir=export_dir, run=args.run, repo_id=args.repo_id,
        root=root, template_path=template_path, base_model=args.base_model,
    )

    if args.dry_run:
        out_dir = root / "results" / args.run
        out_dir.mkdir(parents=True, exist_ok=True)
        card_path = out_dir / "MODEL_CARD.md"
        card_path.write_text(card_text)
        try:
            files = collect_upload_files(export_dir, args.run, root)
        except ValueError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2
        print(f"[dry-run] wrote {card_path}")
        print(f"[dry-run] would upload {len(files) + 1} file(s) to "
              f"{args.repo_id} (private={args.private}), no network call made:")
        for _local, path_in_repo in files:
            print(f"  {path_in_repo}")
        print("  README.md  (assembled model card, see above)")
        return 0

    token = resolve_token()
    if not token:
        print(
            "error: no Hugging Face token found. Set HF_TOKEN in the environment or run "
            "`hf auth login`, then re-run. (Use --dry-run to preview the card without a token "
            "or any network call.)",
            file=sys.stderr,
        )
        return 1

    try:
        n = upload(export_dir=export_dir, run=args.run, repo_id=args.repo_id, root=root,
                   card_text=card_text, private=args.private, token=token)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(f"uploaded {n} file(s) to https://huggingface.co/{args.repo_id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
