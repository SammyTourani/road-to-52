# Copyright 2026 The road-to-52 authors.
# SPDX-License-Identifier: Apache-2.0
"""HellaSwag: rendering, masking and the two decision rules, on five hand-built examples.

No network: the five examples are written here, and the model is a tiny randomly-initialised
GPT with a fixed seed, so both accuracies are deterministic.  The reference the pipeline is
compared against is a second, deliberately naive scorer written inside this file -- one ending
per forward pass, no padding, no batching -- which follows llm.c's formula literally.
"""

from __future__ import annotations

from pathlib import Path

import mlx.core as mx
import pytest
from conftest import tiny_model_config
from mlx.utils import tree_map

from r52.checkpoint import save_checkpoint
from r52.config import Config
from r52.eval.hellaswag import Example, evaluate_hellaswag, load_examples, render_example
from r52.eval.lm import LM
from r52.model import GPT

pytest.importorskip("mlx_lm")

EXAMPLES = [
    Example(
        ctx="A man is sitting on a roof. he",
        endings=(
            "is using wrap to wrap a pair of skis.",
            "is ripping level tiles off.",
            "is holding a rubik's cube.",
            "starts pulling up roofing on a roof.",
        ),
        label=3,
        ind=24,
    ),
    Example(
        ctx="The chef cracks two eggs into the bowl. She",
        endings=("whisks them with a fork.", "drives to the airport.",
                 "paints the ceiling blue.", "solves a quadratic equation."),
        label=0,
        ind=1,
    ),
    # Four identical endings: every score ties, so the first index must win (llm.c uses
    # argmin, which breaks ties toward the lowest index).
    Example(ctx="Ties everywhere.", endings=("same.",) * 4, label=0, ind=2),
    # Endings of very different lengths: acc and acc_norm can and should disagree here.
    Example(
        ctx="He opened the door and",
        endings=("left.", "walked slowly down the corridor toward the lit room.",
                 "sang.", "waited."),
        label=1,
        ind=3,
    ),
    Example(
        ctx="The dog barked at the",
        endings=("mailman.", "integral.", "photosynthesis.", "refrigerator."),
        label=0,
        ind=4,
    ),
]


@pytest.fixture
def tiny_lm(tmp_path: Path) -> LM:
    """A tiny GPT over the full GPT-2 vocabulary, deterministic under conftest's seed."""
    cfg = tiny_model_config(n_layer=1, n_embd=32, n_head=2, vocab_size=50304, block_size=128)
    model = GPT(cfg)
    model.update(tree_map(lambda p: p + mx.random.normal(p.shape) * 0.05, model.parameters()))
    mx.eval(model.parameters())
    ckpt = save_checkpoint(tmp_path / "ckpt", model, config=Config(model=cfg))
    return LM.load(ckpt)


def _naive_scores(lm: LM, ex: Example) -> tuple[list[float], list[float]]:
    """llm.c's formula, one ending per forward pass, no padding and no masking."""
    ctx = lm.tokenizer.encode(ex.ctx, allowed_special=set())
    sums, means = [], []
    for ending in ex.endings:
        end = lm.tokenizer.encode(" " + ending, allowed_special=set())
        row = mx.array([ctx + end], dtype=mx.int32)
        logits = lm.logits(row)
        logprobs = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
        mx.eval(logprobs)
        total = -sum(
            float(logprobs[0, i, int(row[0, i + 1])])
            for i in range(len(ctx) - 1, len(ctx) + len(end) - 1)
        )
        sums.append(total)
        means.append(total / len(end))
    return sums, means


def test_render_prefixes_the_ending_with_a_space_and_masks_only_the_ending(tiny_lm: LM) -> None:
    ex = EXAMPLES[0]
    r = render_example(ex, tiny_lm.tokenizer)
    ctx_tokens = tiny_lm.tokenizer.encode(ex.ctx, allowed_special=set())
    assert len(r.tokens) == 4 == len(r.masks) == len(r.end_bytes)
    for row, mask, ending, nbytes in zip(r.tokens, r.masks, ex.endings, r.end_bytes, strict=True):
        assert row[: len(ctx_tokens)] == ctx_tokens
        assert mask[: len(ctx_tokens)] == [0] * len(ctx_tokens)
        assert set(mask[len(ctx_tokens):]) == {1}
        # The leading space is what makes " is" a single GPT-2 token instead of "is".
        assert row[len(ctx_tokens):] == tiny_lm.tokenizer.encode(" " + ending, allowed_special=set())
        assert nbytes == len(" " + ending)
    assert r.truncated == 0


def test_render_truncation_keeps_the_tail(tiny_lm: LM) -> None:
    long_ctx = Example(ctx="word " * 400, endings=("a.", "b.", "c.", "d."), label=0)
    r = render_example(long_ctx, tiny_lm.tokenizer, max_context=64)
    assert all(len(row) == 64 for row in r.tokens)
    assert r.truncated == 4
    for row, mask, ending in zip(r.tokens, r.masks, long_ctx.endings, strict=True):
        end = tiny_lm.tokenizer.encode(" " + ending, allowed_special=set())
        assert row[-len(end):] == end            # the scored span survives
        assert mask[-len(end):] == [1] * len(end)


def test_pipeline_matches_the_naive_scorer_on_five_examples(tiny_lm: LM) -> None:
    expected_acc = expected_norm = 0
    for ex in EXAMPLES:
        sums, means = _naive_scores(tiny_lm, ex)
        expected_acc += int(min(range(4), key=lambda i: sums[i]) == ex.label)
        expected_norm += int(min(range(4), key=lambda i: means[i]) == ex.label)

    out = evaluate_hellaswag(tiny_lm, EXAMPLES, verbose=False)
    assert out["n_examples"] == 5
    assert out["n_correct"] == expected_acc
    assert out["n_correct_norm"] == expected_norm
    assert out["acc"] == expected_acc / 5
    assert out["acc_norm"] == expected_norm / 5
    assert out["n_truncated_rows"] == 0


def test_batching_across_examples_changes_nothing(tiny_lm: LM) -> None:
    """Padding is on the right and attention is causal, so group size cannot matter."""
    one_at_a_time = LM.load(tiny_lm.path, max_positions_per_forward=64)
    big = LM.load(tiny_lm.path, max_positions_per_forward=1 << 16)
    a = evaluate_hellaswag(one_at_a_time, EXAMPLES, verbose=False)
    b = evaluate_hellaswag(big, EXAMPLES, verbose=False)
    assert (a["n_correct"], a["n_correct_norm"]) == (b["n_correct"], b["n_correct_norm"])


def test_identical_endings_tie_and_argmin_takes_the_first(tiny_lm: LM) -> None:
    out = evaluate_hellaswag(tiny_lm, [EXAMPLES[2]], verbose=False)
    assert out["acc"] == 1.0 and out["acc_norm"] == 1.0  # label is 0, ties resolve to index 0


def test_confidence_interval_brackets_the_accuracy(tiny_lm: LM) -> None:
    out = evaluate_hellaswag(tiny_lm, EXAMPLES, verbose=False)
    lo, hi = out["acc_ci95"]
    assert lo <= out["acc"] <= hi
    lo, hi = out["acc_norm_ci95"]
    assert lo <= out["acc_norm"] <= hi


@pytest.mark.skipif(
    not Path("data/hellaswag/hellaswag_val.jsonl").exists(),
    reason="HellaSwag not cached; scripts/eval.sh downloads it (no network in tests)",
)
def test_the_cached_validation_split_is_the_documented_10042() -> None:
    examples = load_examples("data/hellaswag", "validation")
    assert len(examples) == 10_042
    assert examples[0].ind == 24
    assert examples[0].ctx == "A man is sitting on a roof. he"
    assert examples[0].label == 3
    assert all(len(e.endings) == 4 for e in examples)
