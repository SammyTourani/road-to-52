# Copyright 2026 The road-to-52 authors.
# SPDX-License-Identifier: Apache-2.0
"""hf_upload: the card assembler fills real data, renders "--" for anything missing, never
stages runs/*/ckpt/ contents, and --dry-run writes the card without touching the network.

Offline and tiny on purpose: every fixture here is synthetic (no real run/eval/export needed),
and the one thing that would reach the network (huggingface_hub.HfApi) is never invoked because
every test either calls --dry-run or stubs out token resolution first.
"""

from __future__ import annotations

import json
import re
import struct
from pathlib import Path

import scripts.hf_upload as hf_upload

REAL_TEMPLATE = hf_upload.DEFAULT_TEMPLATE_PATH

# --------------------------------------------------------------------------------------
# Fixtures -- synthetic run / eval / ladder / export directories, real schema, tiny content.
# --------------------------------------------------------------------------------------


def _write_safetensors(path: Path, shapes: dict[str, list[int]]) -> None:
    """A real safetensors header with no tensor bytes -- enough for the header-only reader."""
    header = {
        name: {"dtype": "F32", "shape": shape, "data_offsets": [0, 0]} for name, shape in shapes.items()
    }
    header["__metadata__"] = {"format": "mlx"}
    body = json.dumps(header).encode("utf-8")
    body += b" " * ((-len(body)) % 8)
    path.write_bytes(struct.pack("<Q", len(body)) + body)


def _r52_export_config(name: str = "demo_cfg") -> dict:
    """Shaped like a real `r52.export.model_config_to_mlx_lm` output (model_type=r52gpt)."""
    return {
        "model_type": "r52gpt",
        "hidden_size": 64, "num_hidden_layers": 2, "num_attention_heads": 2, "vocab_size": 256,
        "max_position_embeddings": 32,
        "r52": {
            "name": name,
            "model": {
                "n_layer": 2, "n_embd": 64, "n_head": 2, "n_kv_head": 2, "head_dim": 32,
                "vocab_size": 256, "block_size": 32, "mlp": "relu2", "mlp_ratio": 4.0,
                "qk_norm": True, "use_value_embeds": True, "n_value_embeds": 1,
                "use_unet_skips": True, "softcap": 15.0, "norm_eps": 1e-5,
            },
            "data": {
                "source": "fineweb", "repo_id": "kjj0/fineweb10B-gpt2",
                "train_glob": "fineweb_train_*.bin", "val_file": "fineweb_val_000000.bin",
                "val_tokens": 10_485_760,
            },
            "train": {
                "muon_lr": 0.05, "muon_momentum": 0.95, "muon_nesterov": True,
                "adam_lr_embed": 0.6, "adam_lr_head": 0.008, "adam_lr_scalar": 0.04,
                "warmup_steps": 20, "cooldown_frac": 0.4, "final_lr_frac": 0.15, "grad_clip": 1.0,
            },
        },
    }


def _r52_train_config(name: str = "demo_cfg") -> dict:
    """Shaped like the raw ``r52.config.Config.to_dict()`` the trainer logs --
    ``runs/<run>/log.jsonl``'s first line embeds exactly this under ``"config"`` (flat
    name/model/data/train, *not* the ``"r52"``-wrapped shape ``r52.export`` writes)."""
    return {
        "name": name,
        "model": {
            "n_layer": 2, "n_embd": 64, "n_head": 2, "n_kv_head": 2, "head_dim": 32,
            "vocab_size": 256, "block_size": 32, "mlp": "relu2", "mlp_ratio": 4.0,
            "qk_norm": True, "use_value_embeds": True, "n_value_embeds": 1,
            "use_unet_skips": True, "softcap": 15.0, "norm_eps": 1e-5,
        },
        "data": {
            "source": "fineweb", "repo_id": "kjj0/fineweb10B-gpt2",
            "train_glob": "fineweb_train_*.bin", "val_file": "fineweb_val_000000.bin",
            "val_tokens": 10_485_760,
        },
        "train": {
            "muon_lr": 0.05, "muon_momentum": 0.95, "muon_nesterov": True,
            "adam_lr_embed": 0.6, "adam_lr_head": 0.008, "adam_lr_scalar": 0.04,
            "warmup_steps": 20, "cooldown_frac": 0.4, "final_lr_frac": 0.15, "grad_clip": 1.0,
        },
    }


def _write_run_log(root: Path, run: str, *, with_end: bool) -> None:
    run_dir = root / "runs" / run
    run_dir.mkdir(parents=True)
    start = {
        "t": 0.01, "event": "start", "run": run, "step": 0, "total_steps": 200,
        "params_total": 9_756_677, "params_non_embedding": 98_309,
        "tokens_per_step": 4096, "precision": "mixed", "max_tokens": 819_200,
        "mlx_version": "0.32.2", "config": _r52_train_config(),
    }
    lines = [json.dumps(start), json.dumps({
        "t": 10.5, "event": "train", "step": 100, "tokens": 409_600, "loss": 7.1,
        "tok_s": 39_500.0, "tflops": 0.795, "mfu_theoretical": 0.1868, "mfu_measured": 0.221,
    })]
    if with_end:
        lines.append(json.dumps({
            "t": 21.8, "event": "end", "step": 200, "tokens": 819_200, "reason": "max_tokens",
            "best_val": 6.99, "last_tok_s": 39_491.8, "last_tflops": 0.7939,
        }))
    (run_dir / "log.jsonl").write_text("\n".join(lines) + "\n")


def _write_eval(root: Path, run: str, name: str, *, benchmark: str, value: float, unit: str) -> None:
    d = root / "results" / run
    d.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": run, "benchmark": benchmark, "value": value, "unit": unit,
        "conditions": {"tokenizer": "gpt2 (tiktoken)", "block_size": 32, "limit": None,
                       "n_examples": 50, "few_shot": 0, "command": f"python -m r52.eval.{name}"},
        "commit": "deadbee", "date": "2026-09-13", "wall_clock_s": 1.23,
        "machine": "Test Machine", "command": f"python -m r52.eval.{name}", "metrics": {},
    }
    (d / f"{name}.json").write_text(json.dumps(payload, indent=2))
    # The flat aggregate list r52/bar/gap.py reads -- a list, not a dict, so it must NOT be
    # picked up as an eval record by load_eval_results.
    (d / "eval.json").write_text(json.dumps([payload]))


def _write_ladder(root: Path, rung_id: str) -> None:
    d = root / "results"
    d.mkdir(parents=True, exist_ok=True)
    ladder = {"rungs": [{
        "id": rung_id, "name": f"Rung 0 -- test rung ({rung_id})",
        "beats": "GPT-2 small (124M, 2019)", "status": "running",
        "inputs": {"tokens": 750_000_000},
        "targets": {"fineweb_val_loss": 3.28, "hellaswag_acc": 0.294},
    }]}
    (d / "ladder.json").write_text(json.dumps(ladder))


def _write_export_dir(root: Path, name: str = "export") -> Path:
    export_dir = root / name
    export_dir.mkdir(parents=True)
    (export_dir / "config.json").write_text(json.dumps(_r52_export_config()))
    _write_safetensors(export_dir / "model.safetensors", {"transformer.wte.weight": [256, 64]})
    (export_dir / "tokenizer.json").write_text("{}")
    return export_dir


# --------------------------------------------------------------------------------------
# 1. Real data fills the card.
# --------------------------------------------------------------------------------------


def test_assemble_model_card_fills_known_fields_from_synthetic_run(tmp_path: Path) -> None:
    root = tmp_path
    export_dir = _write_export_dir(root)
    _write_run_log(root, "demo", with_end=False)
    _write_eval(root, "demo", "hellaswag", benchmark="hellaswag", value=29.5, unit="% acc_norm")
    _write_ladder(root, "rung-0-demo")

    card = hf_upload.assemble_model_card(
        export_dir=export_dir, run="demo", repo_id="SammyTourani/road-to-52-demo",
        root=root, template_path=REAL_TEMPLATE,
        commit="abc1234", machine="Test Machine, MLX 0.32.2", generated_date="2026-01-01",
    )

    assert "{{" not in card  # every placeholder in the real template resolved
    assert "SammyTourani/road-to-52-demo" in card
    assert "kjj0/fineweb10B-gpt2" in card  # dataset id, from the run log's "config.data"
    assert "409,600" in card  # tokens seen, from the last "train" log line
    assert "29.50 % acc_norm" in card  # hellaswag row, formatted like docs/RESULTS.md
    assert "| Layers | 2 |" in card  # architecture, from config.json's "r52.model"
    assert "| Embedding dim | 64 |" in card
    assert "Rung 0 -- test rung (rung-0-demo)" in card  # matched ladder row
    assert "No Claude-generated data was used to train this model" in card
    assert "abc1234" in card
    assert "training in progress" in card  # no "end" event was written for this run
    assert "base_model:" not in card  # no --base-model given: the yaml line is omitted entirely


def test_assemble_model_card_with_base_model_and_finished_run(tmp_path: Path) -> None:
    root = tmp_path
    export_dir = _write_export_dir(root)
    _write_run_log(root, "demo", with_end=True)

    card = hf_upload.assemble_model_card(
        export_dir=export_dir, run="demo", repo_id="SammyTourani/road-to-52-demo-sft",
        root=root, template_path=REAL_TEMPLATE, base_model="SammyTourani/road-to-52-demo",
        commit="abc1234", machine="Test Machine", generated_date="2026-01-01",
    )
    assert "base_model: SammyTourani/road-to-52-demo" in card
    assert "training complete -- stopped at step 200/200 (819,200 tokens)" in card
    assert "best val loss 6.9900" in card


# --------------------------------------------------------------------------------------
# 2. Missing data renders "--", never a fabricated number.
# --------------------------------------------------------------------------------------


def test_assemble_model_card_renders_em_dash_for_missing_data(tmp_path: Path) -> None:
    root = tmp_path
    export_dir = root / "bare-export"
    export_dir.mkdir()
    (export_dir / "config.json").write_text("{}")  # neither an "r52" key nor HF-style keys

    card = hf_upload.assemble_model_card(
        export_dir=export_dir, run="nonexistent-run", repo_id="SammyTourani/road-to-52-x",
        root=root, template_path=REAL_TEMPLATE,
        commit="0000000", machine="Test Machine", generated_date="2026-01-01",
    )

    assert "{{" not in card
    assert "None" not in card
    # word-boundary check: the template legitimately contains "nanogpt"/"nanochat"
    assert not re.search(r"\bnan\b", card, re.IGNORECASE)
    assert "no training log found at `runs/<run>/log.jsonl`" in card
    assert "Steps | — / — scheduled |" in card
    assert "| Optimizer | — |" in card
    assert "| MLX version | — |" in card
    assert "| Total parameters | — |" in card
    assert "no matching row in `results/ladder.json`" in card


def test_gpt2_style_config_falls_back_to_generic_keys(tmp_path: Path) -> None:
    """The GPT-2 reference export has no "r52" key at all (mlx_lm.convert, not r52.export)."""
    root = tmp_path
    export_dir = root / "gpt2-like"
    export_dir.mkdir()
    (export_dir / "config.json").write_text(json.dumps({
        "model_type": "gpt2", "n_layer": 12, "n_embd": 768, "n_head": 12,
        "vocab_size": 50257, "n_positions": 1024, "layer_norm_epsilon": 1e-5,
    }))
    _write_safetensors(export_dir / "model.safetensors", {"wte.weight": [50257, 768]})

    card = hf_upload.assemble_model_card(
        export_dir=export_dir, run="no-such-run", repo_id="x/y",
        root=root, template_path=REAL_TEMPLATE,
        commit="0000000", machine="Test Machine", generated_date="2026-01-01",
    )
    assert "| Layers | 12 |" in card
    assert "| Embedding dim | 768 |" in card
    # GQA / r52-specific fields do not exist on a plain HF GPT-2 config -> "--", not invented
    assert "Attention heads (Q / KV) | 12 / — |" in card
    # total params falls back to the safetensors header count (50257 * 768)
    assert f"| Total parameters | {50257 * 768:,} |" in card


# --------------------------------------------------------------------------------------
# 3. runs/*/ckpt/ (weights or optimizer state) is never staged for upload.
# --------------------------------------------------------------------------------------


def test_collect_upload_files_excludes_ckpt_and_optimizer_state(tmp_path: Path) -> None:
    root = tmp_path
    export_dir = _write_export_dir(root)
    ckpt_dir = export_dir / "ckpt" / "step_100"
    ckpt_dir.mkdir(parents=True)
    (ckpt_dir / "model.safetensors").write_bytes(b"not-the-real-export")
    (ckpt_dir / "optim.safetensors").write_bytes(b"optimizer-state")
    _write_eval(root, "demo", "hellaswag", benchmark="hellaswag", value=29.5, unit="% acc_norm")

    files = hf_upload.collect_upload_files(export_dir, "demo", root)
    in_repo = {path_in_repo for _local, path_in_repo in files}

    assert "config.json" in in_repo
    assert "model.safetensors" in in_repo
    assert "tokenizer.json" in in_repo
    assert any(p.startswith("eval/demo/") for p in in_repo)
    assert not any("ckpt" in Path(p).parts for p in in_repo)
    assert not any("ckpt" in local.parts for local, _p in files)
    assert not any(p.endswith("optim.safetensors") for p in in_repo)


def test_collect_upload_files_excludes_export_dirs_own_readme(tmp_path: Path) -> None:
    """upload() always stages the generated card as README.md last, so an export dir's own
    README.md (mlx_lm.convert writes one for the GPT-2 reference) would just be silently
    overwritten -- it must not appear twice in the file list handed to --dry-run either."""
    root = tmp_path
    export_dir = _write_export_dir(root)
    (export_dir / "README.md").write_text("---\nlibrary_name: mlx\n---\n# stub\n")

    files = hf_upload.collect_upload_files(export_dir, "demo", root)
    in_repo = [path_in_repo for _local, path_in_repo in files]

    assert in_repo.count("README.md") == 0
    assert "config.json" in in_repo


# --------------------------------------------------------------------------------------
# 4. --dry-run writes the card and prints the file list, no network.
# --------------------------------------------------------------------------------------


def test_dry_run_writes_model_card_and_prints_file_list(tmp_path: Path, capsys) -> None:
    root = tmp_path
    export_dir = _write_export_dir(root)
    _write_run_log(root, "demo", with_end=True)
    _write_eval(root, "demo", "val_loss", benchmark="fineweb-val-loss", value=3.9, unit="nats/token")

    rc = hf_upload.main(
        ["--export-dir", str(export_dir), "--run", "demo",
         "--repo-id", "SammyTourani/road-to-52-demo", "--dry-run"],
        root=root, template_path=REAL_TEMPLATE,
    )
    assert rc == 0

    card_path = root / "results" / "demo" / "MODEL_CARD.md"
    assert card_path.is_file()
    card = card_path.read_text()
    assert "{{" not in card
    assert "training complete" in card

    out = capsys.readouterr().out
    assert "[dry-run] wrote" in out
    assert "no network call made" in out
    assert "config.json" in out
    assert "eval/demo/val_loss.json" in out
    assert "README.md" in out


def test_main_without_dry_run_requires_a_token_and_makes_no_network_call(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    root = tmp_path
    export_dir = _write_export_dir(root)
    monkeypatch.setattr(hf_upload, "resolve_token", lambda: None)

    rc = hf_upload.main(
        ["--export-dir", str(export_dir), "--run", "demo", "--repo-id", "SammyTourani/road-to-52-demo"],
        root=root, template_path=REAL_TEMPLATE,
    )
    assert rc == 1
    assert "no Hugging Face token found" in capsys.readouterr().err
    assert not (root / "results" / "demo" / "MODEL_CARD.md").exists()


def test_export_dir_must_exist(tmp_path: Path, capsys) -> None:
    rc = hf_upload.main(
        ["--export-dir", str(tmp_path / "nope"), "--run", "demo", "--repo-id", "x/y", "--dry-run"],
        root=tmp_path, template_path=REAL_TEMPLATE,
    )
    assert rc == 2
    assert "is not a directory" in capsys.readouterr().err
