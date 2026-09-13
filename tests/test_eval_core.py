# Copyright 2026 The road-to-52 authors.
# SPDX-License-Identifier: Apache-2.0
"""DCLM CORE: the port's prompt rendering, span logic and task parsing.

The prompt-rendering tests are the load-bearing ones -- they pin the exact strings nanochat's
Jinja2 templates produce, which is what makes our CORE score comparable with nanochat's
``dev/LEADERBOARD.md``.  Tests that need the 161 MB eval bundle are skipped when it has not
been downloaded (tests never touch the network; ``scripts/eval.sh`` / ``r52.eval.core``
fetch it).
"""

from __future__ import annotations

from pathlib import Path

import mlx.core as mx
import pytest
from conftest import tiny_model_config
from mlx.utils import tree_map

from r52.checkpoint import save_checkpoint
from r52.config import Config
from r52.eval.core import (
    DEFAULT_BUNDLE_DIR,
    Task,
    build_item,
    centered_score,
    evaluate_task,
    find_common_length,
    load_random_baselines,
    load_task_data,
    load_tasks,
    reference_scores,
    render_prompts,
    truncate,
)
from r52.eval.lm import LM
from r52.model import GPT
from r52.tokenizer import GPT2Tokenizer

pytest.importorskip("mlx_lm")

BUNDLE = Path(DEFAULT_BUNDLE_DIR)
needs_bundle = pytest.mark.skipif(
    not (BUNDLE / "core.yaml").exists(),
    reason=f"eval bundle not cached at {DEFAULT_BUNDLE_DIR} (no network in tests)",
)

SMALLEST = "bigbench_repeat_copy_logic"  # 32 items, the smallest task in core.yaml

MC = Task("mc", "x/mc.jsonl", "multiple_choice", 0)
MC_ANSWER = Task("mc", "x/mc.jsonl", "multiple_choice", 1, continuation_delimiter="\nAnswer: ")
SCHEMA = Task("schema", "x/s.jsonl", "schema", 1)
LM_TASK = Task("lm", "x/l.jsonl", "language_modeling", 1)


# ---------------------------------------------------------------------------------------
# Prompt rendering -- nanochat's templates, expanded
# ---------------------------------------------------------------------------------------


def test_multiple_choice_zero_shot() -> None:
    item = {"query": "Q1", "choices": ["a", "b"], "gold": 1}
    assert render_prompts(item, MC, []) == ["Q1 a", "Q1 b"]


def test_multiple_choice_few_shot_uses_the_gold_choice_and_a_blank_line() -> None:
    shot = {"query": "S", "choices": ["wrong", "right"], "gold": 1}
    item = {"query": "Q1", "choices": ["a", "b"], "gold": 0}
    out = render_prompts(item, MC_ANSWER, [shot])
    assert out == ["S\nAnswer: right\n\nQ1\nAnswer: a", "S\nAnswer: right\n\nQ1\nAnswer: b"]


def test_schema_varies_the_context_and_keeps_the_continuation() -> None:
    shot = {"context_options": ["c0", "c1"], "continuation": "cont", "gold": 0}
    item = {"context_options": ["A because X", "A because Y"], "continuation": "z.", "gold": 1}
    assert render_prompts(item, SCHEMA, [shot]) == [
        "c0 cont\n\nA because X z.",
        "c0 cont\n\nA because Y z.",
    ]


def test_language_modeling_returns_the_prompt_without_then_with_the_continuation() -> None:
    shot = {"context": "  padded ctx  ", "continuation": "yes"}
    item = {"context": "Q: two plus two\nA:", "continuation": " four"}
    without, with_ = render_prompts(item, LM_TASK, [shot])
    # `| trim` strips the few-shot context; the delimiter defaults to a single space.
    assert with_ == "padded ctx yes\n\nQ: two plus two\nA:  four"
    # ... and the continuation-free prompt is stripped so it stays a clean token prefix.
    assert without == "padded ctx yes\n\nQ: two plus two\nA:"
    assert with_.startswith(without)


def test_unknown_task_type_is_rejected() -> None:
    with pytest.raises(ValueError, match="unsupported task type"):
        render_prompts({}, Task("x", "x", "generative", 0), [])


# ---------------------------------------------------------------------------------------
# Span logic
# ---------------------------------------------------------------------------------------


def test_find_common_length_prefix_and_suffix() -> None:
    assert find_common_length([[1, 2, 3, 9], [1, 2, 4, 9]], "left") == 2
    assert find_common_length([[1, 2, 3, 9], [1, 2, 4, 9]], "right") == 1
    assert find_common_length([[1, 2], [1, 2, 3]], "left") == 2      # one is a prefix of the other
    assert find_common_length([[5], [6]], "left") == 0


def test_build_item_scores_only_the_choice_for_multiple_choice() -> None:
    tok = GPT2Tokenizer()
    item = {"query": "The capital is", "choices": [" Paris", " Berlin"], "gold": 0}
    built = build_item(item, MC, [], tok, tok.eot)
    assert len(built.rows) == 2
    assert all(r[0] == tok.eot for r in built.rows)                 # BOS prepended
    assert built.starts[0] == built.starts[1]
    for row, start, end in zip(built.rows, built.starts, built.ends, strict=True):
        assert end == len(row)
        assert start < end                                          # a non-empty scored span
        assert row[:start] == built.rows[0][:start]                 # the shared prefix


def test_build_item_scores_the_shared_continuation_for_schema() -> None:
    tok = GPT2Tokenizer()
    item = {"context_options": ["The trophy did not fit because the trophy",
                               "The trophy did not fit because the suitcase"],
            "continuation": "is too large.", "gold": 0}
    built = build_item(item, SCHEMA, [], tok, tok.eot)
    spans = [row[s:e] for row, s, e in zip(built.rows, built.starts, built.ends, strict=True)]
    assert spans[0] == spans[1]                                     # the common suffix
    # The delimiter supplies the space, exactly as the bundle's own items assume.
    assert tok.decode(spans[0]) == " is too large."


def test_build_item_keeps_one_row_for_language_modeling() -> None:
    tok = GPT2Tokenizer()
    item = {"context": "The capital of France is", "continuation": "Paris"}
    built = build_item(item, LM_TASK, [], tok, tok.eot)
    assert len(built.rows) == 1
    row, s, e = built.rows[0], built.starts[0], built.ends[0]
    assert tok.decode(row[s:e]) == " Paris"
    assert not built.lm_prefix_fallback


def test_truncate_keeps_the_tail_and_shifts_the_span() -> None:
    item = build_item(
        {"query": "x " * 50, "choices": [" a", " b"], "gold": 0}, MC, [], GPT2Tokenizer(), 50256
    )
    before = [row[s:e] for row, s, e in zip(item.rows, item.starts, item.ends, strict=True)]
    cropped, n_trunc, n_clamp = truncate(item, 16)
    assert n_trunc == 2 and n_clamp == 0
    assert all(len(r) == 16 for r in cropped.rows)
    after = [row[s:e] for row, s, e in zip(cropped.rows, cropped.starts, cropped.ends, strict=True)]
    assert after == before                                          # the scored span is intact


def test_centering_maps_the_random_baseline_to_zero() -> None:
    assert centered_score(0.25, 25.0) == pytest.approx(0.0)
    assert centered_score(1.0, 25.0) == pytest.approx(1.0)
    assert centered_score(0.5, 0.0) == pytest.approx(0.5)
    assert centered_score(0.20, 25.0) < 0                            # below chance stays negative


# ---------------------------------------------------------------------------------------
# Bundle parsing
# ---------------------------------------------------------------------------------------


@needs_bundle
def test_core_yaml_declares_the_22_dclm_tasks() -> None:
    tasks = load_tasks(BUNDLE)
    assert len(tasks) == 22
    labels = [t.label for t in tasks]
    assert labels[0] == "hellaswag_zeroshot"
    assert labels[-1] == "bigbench_language_identification"
    assert SMALLEST in labels
    by_label = {t.label: t for t in tasks}
    assert by_label["hellaswag_zeroshot"].num_fewshot == 0
    assert by_label["hellaswag"].num_fewshot == 10
    assert by_label["agi_eval_lsat_ar"].num_fewshot == 3
    assert by_label["arc_easy"].continuation_delimiter == "\nAnswer: "
    assert by_label["copa"].continuation_delimiter == " "            # the default
    assert {t.task_type for t in tasks} == {"multiple_choice", "schema", "language_modeling"}


@needs_bundle
def test_smallest_task_parses_and_the_shuffle_is_seeded() -> None:
    task = {t.label: t for t in load_tasks(BUNDLE)}[SMALLEST]
    assert (task.task_type, task.num_fewshot) == ("language_modeling", 10)
    data = load_task_data(task, BUNDLE)
    assert len(data) == 32
    assert all({"context", "continuation"} <= set(row) for row in data)
    assert load_task_data(task, BUNDLE) == data                      # seed 1337, reproducible
    assert load_task_data(task, BUNDLE, limit=5) == data[:5]         # limit takes the shuffled head


@needs_bundle
def test_random_baselines_cover_every_task() -> None:
    baselines = load_random_baselines(BUNDLE)
    for task in load_tasks(BUNDLE):
        assert task.label in baselines
    assert baselines["hellaswag"] == 25.0
    assert baselines["copa"] == 50.0
    assert baselines[SMALLEST] == 0.0


@needs_bundle
def test_nanochat_reference_csv_is_the_comparison_point() -> None:
    """The bundle ships nanochat's own GPT-2 numbers; they anchor this port."""
    gpt2 = reference_scores("openai-community-gpt2", BUNDLE)
    assert gpt2["CORE"] == pytest.approx(0.113891)
    assert gpt2[SMALLEST] == pytest.approx(0.031250)
    xl = reference_scores("openai-community-gpt2-xl", BUNDLE)
    assert xl["CORE"] == pytest.approx(0.256525)  # dev/LEADERBOARD.md's "time to GPT-2" bar


@needs_bundle
def test_evaluate_task_runs_end_to_end_on_the_smallest_task(tmp_path: Path) -> None:
    cfg = tiny_model_config(n_layer=1, n_embd=32, n_head=2, vocab_size=50304, block_size=256)
    model = GPT(cfg)
    model.update(tree_map(lambda p: p + mx.random.normal(p.shape) * 0.05, model.parameters()))
    mx.eval(model.parameters())
    lm = LM.load(save_checkpoint(tmp_path / "ckpt", model, config=Config(model=cfg)))

    task = {t.label: t for t in load_tasks(BUNDLE)}[SMALLEST]
    data = load_task_data(task, BUNDLE, limit=12)
    out = evaluate_task(lm, task, data)
    assert out["n_examples"] == 12
    assert 0.0 <= out["accuracy"] <= 1.0
    assert out["task_type"] == "language_modeling"
    assert out["num_fewshot"] == 10
    assert out["n_lm_prefix_fallbacks"] == 0
    # 10-shot repeat-copy prompts are long, so a 256-token context must crop them.
    assert out["n_truncated_rows"] == 12
