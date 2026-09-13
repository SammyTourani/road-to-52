# Copyright 2026 The road-to-52 authors.
# SPDX-License-Identifier: Apache-2.0
"""One real ablation cell, end to end, on the GPU -- deliberately the smallest possible one.

This is the only file in the ablation suite that trains anything.  The cell is a 2-layer,
64-dim model on 20,000 tokens with HellaSwag capped at 20 examples and one CORE task capped
at 10 items, which is a handful of seconds of compute; the point is that ``run_cell`` really
does spawn ``r52.train``, really does spawn ``r52.ablate.evals``, and really does write a
results JSON with all three metrics in it.

The machine may be running a multi-day headline job at the same time
(``docs/ARCHITECTURE.md`` §1.2), so both subprocesses are held to a 2 GiB MLX guideline and a
small per-forward position budget.
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import numpy as np
import pytest
import yaml

from r52.ablate.matrix import Axis, Cell
from r52.ablate.run import cell_result, run_cell
from r52.data import read_shard_header
from r52.tokenizer import GPT2Tokenizer

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))

from prepare_corpus import ShardWriter  # noqa: E402

pytestmark = pytest.mark.timeout(180)

TRAIN_TOKENS = 20_000
HELLASWAG_LIMIT = 20
VAL_TOKENS = 4_000


def _write_corpus(out: Path) -> dict:
    """A tiny GPT-2-tokenized corpus with the manifest ``run_cell`` reads."""
    tok = GPT2Tokenizer()
    vocab_text = (
        "the student read a book about photosynthesis and chlorophyll in the school library "
        "while the teacher explained how plants convert sunlight into chemical energy"
    )
    words = vocab_text.split()
    rng = random.Random(3)
    docs = [
        f"Lesson {i}. " + " ".join(rng.choices(words, k=90)) + "\n"
        for i in range(600)
    ]
    out.mkdir(parents=True, exist_ok=True)

    text_bytes = 0
    produced = 0
    rows = 0
    it = iter(docs)

    def fill(writer: ShardWriter, n: int) -> int:
        nonlocal text_bytes, produced, rows
        written = 0
        while written < n:
            try:
                doc = next(it)
            except StopIteration:  # pragma: no cover - 600 documents is plenty
                break
            ids = np.asarray([tok.eot, *tok.encode_ordinary(doc)], dtype=np.uint16)
            rows += 1
            text_bytes += len(doc.encode("utf-8"))
            produced += int(ids.size)
            written += writer.write(ids[: n - written])
        return written

    w = ShardWriter(out / "t_val_000000.bin")
    fill(w, VAL_TOKENS)
    w.close()
    w = ShardWriter(out / "t_train_000001.bin")
    fill(w, TRAIN_TOKENS + 2_000)
    w.close()

    manifest = {
        "corpus": "t",
        "rows": rows,
        "tokens": TRAIN_TOKENS + 2_000 + VAL_TOKENS,
        "tokens_encoded": produced,
        "text_bytes": text_bytes,
        "bytes_per_token": round(text_bytes / produced, 6),
        "tokenizer": {"spec": "gpt2", "name": "gpt2", "vocab_size": 50304, "eot_id": 50256},
        "sources": [{"name": "t", "repo_id": "local/synthetic", "revision": "0" * 40,
                     "config": None, "weight": 1.0, "license": "n/a"}],
        "val": {"file": "t_val_000000.bin", "tokens": VAL_TOKENS},
        "train": {"tokens": TRAIN_TOKENS + 2_000, "shards": [{"file": "t_train_000001.bin"}]},
        "created": "2026-09-13T00:00:00Z",
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest


def _write_config(path: Path) -> None:
    cfg = {
        "name": "abl_smoke",
        "model": {
            "n_layer": 2, "n_embd": 64, "n_head": 2, "vocab_size": 50304, "block_size": 128,
            "n_value_embeds": 1, "value_embed_share": True, "precision": "mixed",
        },
        "data": {
            "source": "fineweb", "tokenizer": "gpt2",
            "data_dir": "unset", "train_glob": "t_train_*.bin", "val_file": "t_val_000000.bin",
            "val_tokens": VAL_TOKENS,
        },
        "train": {
            "micro_batch": 8, "tokens_per_step": 4096, "max_tokens": TRAIN_TOKENS,
            "warmup_steps": 0, "val_interval": 0, "log_interval": 2, "ckpt_interval": 0,
            "val_max_batches": 2, "val_micro_batch": 2, "keep_last": 1,
            "memory_limit_gb": 2.0, "cache_limit_gb": 0.5, "compile": False, "seed": 7,
        },
    }
    path.write_text(yaml.safe_dump(cfg, sort_keys=False))


@pytest.mark.skipif(
    not (REPO / "data" / "hellaswag" / "hellaswag_val.jsonl").is_file(),
    reason="HellaSwag validation split is not cached (this test never downloads)",
)
@pytest.mark.skipif(
    not (REPO / "data" / "eval_bundle" / "core.yaml").is_file(),
    reason="the CORE eval bundle is not cached (this test never downloads)",
)
def test_one_cell_end_to_end(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus"
    manifest = _write_corpus(corpus)
    assert read_shard_header(corpus / "t_val_000000.bin") == VAL_TOKENS

    config = tmp_path / "abl_smoke.yaml"
    _write_config(config)

    cell = Cell(
        axis="smoke",
        name="gpt2",
        config=str(config),
        overrides=(
            f"data.data_dir={corpus}",
            "data.train_glob=t_train_*.bin",
            "data.val_file=t_val_000000.bin",
            "data.tokenizer=gpt2",
        ),
        corpus="t",
        tokenizer="gpt2",
        max_tokens=TRAIN_TOKENS,
    )
    ax = Axis(name="smoke", description="a smoke cell", cells=(cell,),
              core_tasks=("copa",), hellaswag_limit=HELLASWAG_LIMIT)

    rec = run_cell(
        ax, cell,
        results_root=str(tmp_path / "results"),
        out_dir=str(tmp_path / "runs"),
        memory_gib=2.0,
        val_tokens=VAL_TOKENS,
        max_positions=1024,
        train_extra=["-o", "model.block_size=128"],
    )

    # -- the record is on disk and complete --------------------------------------------
    path = cell_result("smoke", "gpt2", str(tmp_path / "results"))
    assert path.is_file()
    assert json.loads(path.read_text()) == rec
    assert rec["axis"] == "smoke" and rec["cell"] == "gpt2"
    assert rec["budget"]["max_tokens"] == TRAIN_TOKENS
    assert "r52.train" in rec["commands"]["train"]
    assert "r52.ablate.evals" in rec["commands"]["eval"]
    assert rec["corpus"]["manifest"]["bytes_per_token"] == manifest["bytes_per_token"]
    assert rec["finished"] >= rec["started"]

    # -- training really happened -------------------------------------------------------
    assert rec["train"]["tokens"] >= TRAIN_TOKENS
    assert rec["train"]["reason"] == "max_tokens"
    # `--max-tokens` is checked before a step, so the run overshoots by at most one step.
    assert rec["train"]["step"] == -(-TRAIN_TOKENS // 4096) == 5
    assert (tmp_path / "runs" / "abl-smoke-gpt2" / "ckpt" / "best").is_dir()

    # -- all three metrics --------------------------------------------------------------
    val = rec["eval"]["val"]
    assert val["n_tokens"] > 0
    assert 0.0 < val["val_loss"] < 20.0
    assert val["bytes_per_token"] > 1.0
    assert val["val_bpb"] > 0.0

    hs = rec["eval"]["hellaswag"]
    assert hs["n_examples"] == HELLASWAG_LIMIT
    assert 0.0 <= hs["acc_norm"] <= 1.0
    lo, hi = hs["acc_norm_ci95"]
    assert lo <= hs["acc_norm"] <= hi

    core = rec["eval"]["core"]
    assert core["n_tasks"] == 1
    assert core["tasks"] == ["copa"]
    assert core["per_task"][0]["n_examples"] == 100  # copa's full split, 0-shot and cheap
    assert isinstance(core["core_metric"], float)

    assert rec["eval"]["tokenizer"]["n_vocab"] == 50257
    assert rec["eval"]["model"]["kind"] == "r52-checkpoint"
    assert rec["eval"]["peak_memory_gib"] < 3.0

    # -- the intermediate eval file is cleaned up, and a rerun is a no-op ---------------
    assert not path.with_suffix(".eval.json").exists()
    again = run_cell(ax, cell, results_root=str(tmp_path / "results"),
                     out_dir=str(tmp_path / "runs"))
    assert again == rec

    # -- and the report renders from it -------------------------------------------------
    from r52.ablate.report import render_axis

    md = render_axis(ax, {"gpt2": rec})
    assert "| `gpt2` |" in md
    assert "1/1 cells complete" in md
