# Copyright 2026 The road-to-52 authors.
# SPDX-License-Identifier: Apache-2.0
"""The ablation lab without a GPU and without a network.

Covers the corpus registry, the shard writer (fed by a synthetic local text source in place
of Hugging Face), the three axes, the resumable driver's plan output, and the report
renderer.  ``tests/test_ablate_run.py`` is the one file here that touches the GPU.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from r52.ablate import corpora, matrix, report
from r52.ablate import run as abl_run
from r52.config import DataConfig
from r52.data import load_shard, make_train_stream, make_val_loader, read_shard_header

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))

import prepare_corpus  # noqa: E402  (needs the sys.path line above)

# --------------------------------------------------------------------------------------
# Registry
# --------------------------------------------------------------------------------------


def test_registry_holds_the_six_candidates_of_the_spec() -> None:
    reg = corpora.load_registry()
    assert set(reg) == {
        "fineweb", "fineweb-edu", "dclm", "dolma3", "ultra-fineweb", "finepdfs-edu"
    }


@pytest.mark.parametrize("name", corpora.corpus_names())
def test_every_row_is_pinned_and_documented(name: str) -> None:
    c = corpora.get_corpus(name)
    assert c.repo_id.count("/") == 1
    assert len(c.revision) == 40, f"{name}: revision must be a full commit sha"
    assert c.format in ("parquet", "json")
    assert c.text_key
    assert c.license
    assert c.notes.strip(), f"{name}: every corpus carries a note"
    assert c.url("x/y.parquet") == f"hf://datasets/{c.repo_id}@{c.revision}/x/y.parquet"


def test_no_excluded_or_noncommercial_dataset_is_registered() -> None:
    # docs/PLAN.md §5: ClimbMix is CC-BY-NC and is explicitly avoided; nothing here may be
    # on the Claude-derived exclude list either.
    banned = ("climbmix", "climblab", "swe-smith", "tulu-3", "hh-rlhf", "openhands", "r2e-gym")
    for name in corpora.corpus_names():
        repo = corpora.get_corpus(name).repo_id.lower()
        assert not any(b in repo for b in banned), repo


def test_the_two_corpora_that_need_care_are_configured_for_it() -> None:
    # dolma3 is laid out by source/topic; reading it in sorted order gives adult content only.
    d = corpora.get_corpus("dolma3")
    assert d.file_order == "shuffled" and d.interleave >= 8
    # Ultra-FineWeb's column is `content`, not `text`.
    assert corpora.get_corpus("ultra-fineweb").text_key == "content"
    # DCLM ships jsonl.zst, not parquet.
    assert corpora.get_corpus("dclm").format == "json"


def test_unknown_corpus_lists_the_alternatives() -> None:
    with pytest.raises(KeyError, match="registry has"):
        corpora.get_corpus("nemotron-cc")


# --------------------------------------------------------------------------------------
# The corpus writer, against a local synthetic source
# --------------------------------------------------------------------------------------


def _synthetic_docs(n: int = 400, words: int = 120) -> list[str]:
    import random

    vocab_text = (
        "education research student teaching science history the a of and to in for with "
        "photosynthesis chlorophyll mitochondria 1789 1914 2026 chapter section figure"
    )
    vocab = vocab_text.split()
    rng = random.Random(11)
    return [
        f"Document {i}\n\n" + " ".join(rng.choices(vocab, k=words)) + "\n"
        for i in range(n)
    ]


@pytest.fixture
def local_corpus(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Replace the Hugging Face stream with local text; nothing touches the network."""
    docs = _synthetic_docs()

    def fake_iter(corpus, files=None, max_files=0, on_file=None):
        if on_file is not None:
            on_file("local/synthetic-000.parquet")
        while True:  # the real stream is long; the writer stops it
            for d in docs:
                yield corpora.Document(d, len(d.encode("utf-8")))

    monkeypatch.setattr(prepare_corpus, "iter_documents", fake_iter)
    return docs


def test_prepare_corpus_writes_shards_r52_data_can_read(
    tmp_path: Path, local_corpus: list[str]
) -> None:
    out = tmp_path / "fineweb-edu-gpt2"
    rc = prepare_corpus.main(
        [
            "--corpus", "fineweb-edu",
            "--tokenizer", "gpt2",
            "--tokens", "40000",
            "--val-tokens", "8000",
            "--shard-tokens", "15000",
            "--out", str(out),
            "--quiet",
        ]
    )
    assert rc == 0

    val = out / "fineweb-edu_val_000000.bin"
    trains = sorted(out.glob("fineweb-edu_train_*.bin"))
    assert val.is_file()
    assert len(trains) == 3, [p.name for p in trains]  # 15k + 15k + 10k
    assert read_shard_header(val) == 8000
    assert sum(read_shard_header(p) for p in trains) == 40000

    cfg = DataConfig(
        data_dir=str(out),
        train_glob="fineweb-edu_train_*.bin",
        val_file="fineweb-edu_val_000000.bin",
        val_tokens=8000,
    )
    stream = make_train_stream(cfg, 128, 4, 50304)
    x, y = stream.next_batch()
    assert x.shape == (4, 128) and y.shape == (4, 128)
    assert stream.total_tokens == 40000
    loader = make_val_loader(cfg, 128, 2, 0, 50304)
    assert loader.tokens > 0

    # The val split is held out FIRST: its documents must not reappear at the head of train.
    from r52.tokenizer import GPT2Tokenizer

    tok = GPT2Tokenizer()
    val_head = tok.decode(np.asarray(load_shard(val)[:40], dtype=np.int64).tolist())
    train_head = tok.decode(np.asarray(load_shard(trains[0])[:40], dtype=np.int64).tolist())
    assert val_head.startswith("<|endoftext|>Document 0")
    assert val_head != train_head


def test_manifest_records_identity_counts_and_a_data_block(
    tmp_path: Path, local_corpus: list[str]
) -> None:
    out = tmp_path / "c"
    prepare_corpus.main(
        ["--corpus", "fineweb-edu", "--tokens", "20000", "--val-tokens", "2000",
         "--out", str(out), "--quiet"]
    )
    man = json.loads((out / "manifest.json").read_text())

    assert man["corpus"] == "fineweb-edu"
    assert man["tokens"] == 22000
    assert man["rows"] > 0
    assert man["text_bytes"] > 0
    assert man["bytes_per_token"] == pytest.approx(
        man["text_bytes"] / man["tokens_encoded"], abs=1e-6  # the manifest rounds to 6 dp
    )
    assert 2.0 < man["bytes_per_token"] < 8.0
    assert man["created"].endswith("Z")

    src = man["sources"][0]
    assert src["repo_id"] == "HuggingFaceFW/fineweb-edu"
    assert len(src["revision"]) == 40
    assert src["config"] == "sample-10BT"
    assert src["text_key"] == "text"
    assert src["license"] == "odc-by"
    assert src["files_read"] == ["local/synthetic-000.parquet"]

    assert man["tokenizer"] == {
        "spec": "gpt2", "name": "gpt2", "vocab_size": 50304, "eot_id": 50256,
        "document_separator": "<|endoftext|> prefixed to every document",
    }
    assert man["data_config"]["train_glob"] == "fineweb-edu_train_*.bin"
    assert man["data_config"]["val_file"] == "fineweb-edu_val_000000.bin"


def test_prepare_corpus_with_our_own_tokenizer(tmp_path: Path, local_corpus: list[str]) -> None:
    from r52.tokenizer_train import save_tokenizer, train_tokenizer

    tokdir = save_tokenizer(
        train_tokenizer(local_corpus, 512, min_frequency=1), tmp_path / "tok",
        name="tok", vocab_size=512,
    )
    out = tmp_path / "c"
    prepare_corpus.main(
        ["--corpus", "fineweb-edu", "--tokenizer", str(tokdir), "--tokens", "20000",
         "--val-tokens", "2000", "--out", str(out), "--quiet"]
    )
    man = json.loads((out / "manifest.json").read_text())
    assert man["tokenizer"]["vocab_size"] == 512
    assert man["tokenizer"]["eot_id"] == 503
    arr = load_shard(out / "fineweb-edu_val_000000.bin")
    assert int(arr.max()) < 512
    assert int(arr[0]) == 503  # every document starts with <|endoftext|>


def test_blend_hits_its_token_share(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # Documents of very different lengths, which is exactly what breaks naive per-document
    # sampling (docs/DEVIATIONS.md P5).
    long_docs = ["L " * 600] * 200
    short_docs = ["s " * 60] * 200

    def fake_iter(corpus, files=None, max_files=0, on_file=None):
        docs = long_docs if corpus.name == "fineweb-edu" else short_docs
        while True:
            for d in docs:
                yield corpora.Document(d, len(d.encode("utf-8")))

    monkeypatch.setattr(prepare_corpus, "iter_documents", fake_iter)
    out = tmp_path / "b"
    prepare_corpus.main(
        ["--corpus", "fineweb-edu", "--blend", "finepdfs-edu:0.25", "--tokens", "60000",
         "--val-tokens", "0", "--out", str(out), "--quiet"]
    )
    man = json.loads((out / "manifest.json").read_text())
    shares = {s["name"]: s["tokens"] / man["tokens_encoded"] for s in man["sources"]}
    assert shares["finepdfs-edu"] == pytest.approx(0.25, abs=0.03)
    assert shares["fineweb-edu"] == pytest.approx(0.75, abs=0.03)
    assert man["sources"][1]["weight"] == 0.25


def test_blend_weights_must_leave_room_for_the_base(
    tmp_path: Path, local_corpus: list[str]
) -> None:
    rc = prepare_corpus.main(
        ["--corpus", "fineweb-edu", "--blend", "finepdfs-edu:1.0", "--tokens", "1000",
         "--out", str(tmp_path / "x"), "--quiet"]
    )
    assert rc == 2


def test_corpora_listing_runs() -> None:
    assert prepare_corpus.main(["--corpora"]) == 0


# --------------------------------------------------------------------------------------
# The matrix
# --------------------------------------------------------------------------------------


def test_axes_enumerate_the_expected_cells() -> None:
    axes = matrix.build_axes()
    assert list(axes) == ["tokenizer", "corpus", "arch"]
    assert matrix.AXES == ("tokenizer", "corpus", "arch")  # cheapest first

    assert [c.name for c in axes["tokenizer"].cells] == ["gpt2", "own-32k"]
    assert [c.name for c in axes["corpus"].cells] == [
        "fineweb", "fineweb-edu", "dclm", "dolma3", "ultra-fineweb", "finepdfs-edu-blend"
    ]
    assert [c.name for c in axes["arch"].cells] == [
        "baseline", "value-embeds-off", "unet-skips-off", "mlp-swiglu", "softcap-off",
        "qk-norm-off", "muon-lr-0.035", "muon-lr-0.07", "seq-512",
    ]
    # docs/ABLATIONS.md: "tokenizer (2 runs) -> corpus (6 runs) -> architecture (~10 runs)"
    assert (len(axes["tokenizer"]), len(axes["corpus"]), len(axes["arch"])) == (2, 6, 9)


def test_every_cell_carries_a_complete_data_block() -> None:
    for ax in matrix.build_axes().values():
        for cell in ax.cells:
            keys = {o.split("=", 1)[0] for o in cell.overrides}
            assert {"data.data_dir", "data.train_glob", "data.val_file", "data.tokenizer"} <= keys
            assert cell.config.startswith("configs/")
            assert cell.run_name == f"abl-{ax.name}-{cell.slug}"
            assert cell.result_path == f"results/ablations/{ax.name}/{cell.slug}.json"


def test_corpus_axis_fixes_tokens_and_tokenizer_axis_fixes_bytes() -> None:
    axes = matrix.build_axes()
    for c in axes["corpus"].cells:
        assert c.max_tokens == matrix.CORPUS_TOKENS and c.max_bytes is None
    for c in axes["tokenizer"].cells:
        assert c.max_bytes == matrix.TOKENIZER_AXIS_BYTES and c.max_tokens is None
    # ... and the two tokenizer cells differ ONLY in the tokenizer and its corpus directory.
    a, b = axes["tokenizer"].cells
    assert a.tokenizer == "gpt2" and b.tokenizer == matrix.DEFAULT_ABL_TOKENIZER
    assert a.config == b.config
    diff = set(a.overrides) ^ set(b.overrides)
    assert {o.split("=", 1)[0] for o in diff} == {"data.data_dir", "data.tokenizer"}


def test_arch_axis_moves_one_knob_at_a_time() -> None:
    ax = matrix.axis("arch")
    base = ax.cell("baseline")
    for cell in ax.cells:
        extra = [o for o in cell.overrides if o not in base.overrides]
        assert len(extra) <= 1, (cell.name, extra)


def test_arch_axis_covers_every_knob_the_spec_names() -> None:
    knobs = {
        o.split("=", 1)[0]
        for c in matrix.axis("arch").cells
        for o in c.overrides
        if not o.startswith("data.")
    }
    assert knobs == {
        "model.use_value_embeds", "model.use_unet_skips", "model.mlp", "model.softcap",
        "model.qk_norm", "train.muon_lr", "model.block_size",
    }


def test_core6_is_the_six_cheapest_tasks() -> None:
    assert len(matrix.CORE6) == 6
    assert len(set(matrix.CORE6)) == 6
    assert len(matrix.CORE6_ALTERNATIVE) == 6
    bundle = REPO / "data" / "eval_bundle"
    if (bundle / "core.yaml").is_file():
        from r52.eval.core import load_tasks

        labels = {t.label for t in load_tasks(bundle)}
        assert set(matrix.CORE6) <= labels
        assert set(matrix.CORE6_ALTERNATIVE) <= labels


def test_train_argv_is_a_runnable_command() -> None:
    cell = matrix.axis("corpus").cell("dclm")
    argv = cell.train_argv()
    assert argv[0] == "configs/ablations/tiny_abl.yaml"
    assert argv[1:3] == ["--run-name", "abl-corpus-dclm"]
    assert "--max-tokens" in argv
    assert argv.count("-o") == len(cell.overrides)
    # It must parse as `r52.train`'s own CLI.
    from r52.train import build_parser

    args = build_parser().parse_args(argv)
    assert args.run_name == "abl-corpus-dclm"
    assert args.max_tokens == matrix.CORPUS_TOKENS


def test_matrix_cli_prints_and_dumps() -> None:
    assert matrix.main(["--axis", "corpus"]) == 0
    assert matrix.main(["--json"]) == 0
    assert matrix.main(["--axis", "arch", "--slugs"]) == 0


def test_unknown_axis_and_cell() -> None:
    with pytest.raises(KeyError):
        matrix.axis("optimizer")
    with pytest.raises(KeyError, match="has no cell"):
        matrix.axis("corpus").cell("nemotron")


# --------------------------------------------------------------------------------------
# The driver
# --------------------------------------------------------------------------------------


def test_dry_run_reports_a_missing_corpus_instead_of_crashing(tmp_path: Path, capsys) -> None:
    ax = matrix.axis("corpus")
    rec = abl_run.run_cell(ax, ax.cell("dclm"), dry_run=True, results_root=str(tmp_path))
    assert rec["dry_run"] is True
    assert "prepare_corpus.py" in rec["blocked"]
    out = capsys.readouterr().out
    assert "train:" in out and "eval :" in out


def test_budget_from_bytes_uses_the_manifest(tmp_path: Path, local_corpus: list[str]) -> None:
    out = tmp_path / "fineweb-edu-gpt2"
    prepare_corpus.main(
        ["--corpus", "fineweb-edu", "--tokens", "40000", "--val-tokens", "2000",
         "--out", str(out), "--quiet"]
    )
    ax = matrix.axis("tokenizer", root=str(tmp_path))
    cell = ax.cell("gpt2")
    tokens, prov = abl_run.resolve_budget(cell)
    man = json.loads((out / "manifest.json").read_text())
    assert tokens == int(matrix.TOKENIZER_AXIS_BYTES / man["bytes_per_token"])
    assert prov["budget"]["mode"] == "fixed_bytes"
    assert prov["budget"]["sufficient"] is False  # 40k tokens is not 420 MB of text
    assert prov["manifest"]["sources"][0]["repo_id"] == "HuggingFaceFW/fineweb-edu"


def test_a_finished_cell_is_skipped(tmp_path: Path) -> None:
    ax = matrix.axis("corpus")
    p = abl_run.cell_result("corpus", "dclm", str(tmp_path))
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({"cell": "dclm", "marker": 1}))
    rec = abl_run.run_cell(ax, ax.cell("dclm"), results_root=str(tmp_path))
    assert rec["marker"] == 1  # returned as-is, nothing was run


def test_another_train_running_sees_the_headline_run() -> None:
    # Informational: this repo is expected to have a multi-day run going while the lab is
    # built, but the function must at least return a well-formed list.
    hits = abl_run.another_train_running()
    assert isinstance(hits, list)
    assert all(isinstance(pid, int) and isinstance(cmd, str) for pid, cmd in hits)


def test_ablate_sh_dry_run_prints_the_plan() -> None:
    proc = subprocess.run(
        ["bash", str(REPO / "scripts" / "ablate.sh"), "tokenizer", "--dry-run"],
        cwd=REPO, capture_output=True, text=True, timeout=180, check=False,
    )
    assert proc.returncode == 0, proc.stderr
    out = proc.stdout
    assert "DRY RUN" in out
    assert "cells    : 2" in out
    assert "-- gpt2 --" in out and "-- own-32k --" in out
    assert "python -m r52.train configs/ablations/tiny_abl.yaml" in out
    assert "python -m r52.ablate.evals" in out
    assert "runs/" in out


def test_ablate_sh_rejects_an_unknown_axis() -> None:
    proc = subprocess.run(
        ["bash", str(REPO / "scripts" / "ablate.sh"), "optimizer"],
        cwd=REPO, capture_output=True, text=True, timeout=60, check=False,
    )
    assert proc.returncode == 2
    assert "unknown axis" in proc.stderr


def test_ablate_sh_lists_cell_status(tmp_path: Path) -> None:
    (tmp_path / "corpus").mkdir()
    (tmp_path / "corpus" / "dclm.json").write_text("{}")
    proc = subprocess.run(
        ["bash", str(REPO / "scripts" / "ablate.sh"), "corpus", "--list",
         "--results", str(tmp_path)],
        cwd=REPO, capture_output=True, text=True, timeout=180, check=False,
    )
    assert proc.returncode == 0, proc.stderr
    assert "dclm" in proc.stdout and "done" in proc.stdout
    assert "pending" in proc.stdout


# --------------------------------------------------------------------------------------
# The report
# --------------------------------------------------------------------------------------


def _fake_result(cell, bpb: float, acc: float, core: float) -> dict:
    return {
        "axis": cell.axis,
        "cell": cell.name,
        "spec": cell.to_dict(),
        "budget": {"max_tokens": 100_000_000},
        "wall_clock_s": 10_800.0,
        "commands": {"train": "python -m r52.train ..."},
        "corpus": {"manifest": {"bytes_per_token": 4.4, "sources": [
            {"repo_id": "HuggingFaceFW/fineweb-edu", "revision": "0" * 40}]}},
        "train": {"tokens": 100_000_000, "step": 1525},
        "eval": {
            "val": {"val_loss": 3.9, "val_bpb": bpb},
            "hellaswag": {"acc_norm": acc, "acc_norm_ci95": [acc - 0.02, acc + 0.02]},
            "core": {"core_metric": core, "n_tasks": 6},
        },
    }


def test_report_renders_our_rank_next_to_the_published_one(tmp_path: Path) -> None:
    ax = matrix.axis("corpus")
    d = tmp_path / "corpus"
    d.mkdir()
    # Deliberately disagree with the published order on one pair.
    bpbs = {"dclm": 1.10, "ultra-fineweb": 1.12, "fineweb": 1.14, "fineweb-edu": 1.16}
    for i, (name, bpb) in enumerate(bpbs.items()):
        cell = ax.cell(name)
        (d / f"{cell.slug}.json").write_text(json.dumps(_fake_result(cell, bpb, 0.28 + i / 100, 0.05)))

    path = report.write_axis(ax, str(tmp_path))
    md = path.read_text()
    assert path.name == "corpus.md"
    assert "| cell | val bpb |" in md
    assert "1.1000" in md and "0.2800 [0.2600, 0.3000]" in md
    assert "_pending_" in md  # dolma3 and the blend have not run
    assert "Agreement with the published ranking" in md
    assert "2/4 exact positions" in md          # dclm 1, ultra 2 agree; fineweb/-edu swapped
    assert "5/6 pairwise orderings" in md
    assert "DataDecide" in md
    assert "research/03 §1.3" in md
    assert "HuggingFaceFW/fineweb-edu" in md
    assert "3.00 h" in md


def test_report_of_an_empty_axis_still_renders(tmp_path: Path) -> None:
    ax = matrix.axis("arch")
    md = report.render_axis(ax, {})
    assert "0/9 cells complete" in md
    assert md.count("_pending_") == 9


def test_report_cli(tmp_path: Path) -> None:
    assert report.main(["--axis", "tokenizer", "--results-dir", str(tmp_path)]) == 0
    assert (tmp_path / "tokenizer.md").is_file()
    assert report.main(["--all", "--results-dir", str(tmp_path), "--stdout"]) == 0
