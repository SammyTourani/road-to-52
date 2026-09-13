# Copyright 2026 The road-to-52 authors.
# SPDX-License-Identifier: Apache-2.0
"""The lm-eval bridge: task resolution, metric picking, the LM adapter and the results schema.

Everything here is offline and GPU-light.  The tests that touch a model use whichever of
``models/tiny200-mlx`` (an ``r52gpt`` plugin export -- the interesting path) or
``models/gpt2-mlx`` exists, and skip if neither does.  Nothing downloads a dataset: the
loglikelihood cross-check reads the HellaSwag jsonl the suite already caches under
``data/hellaswag/``, and the suite-definition tests only ask ``lm_eval``'s ``TaskManager``
for task *names*, which is a local YAML scan.

The load-bearing assertion is :func:`test_loglikelihood_matches_hellaswag_scoring`: the
bridge's ``loglikelihood`` and :mod:`r52.eval.hellaswag`'s own scorer must agree, or the
lm-eval numbers and the rest of ``r52/eval/`` are measuring two different models.
"""

from __future__ import annotations

import json
from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest

pytest.importorskip("lm_eval")
pytest.importorskip("mlx_lm")

from r52.eval import lmeval
from r52.eval.lmeval import (
    QUICK,
    STANDARD,
    SUITES,
    TaskRun,
    TaskSpec,
    git_commit,
    git_commit_from_files,
    pick_metric,
    resolve_tasks,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
HELLASWAG_JSONL = REPO_ROOT / "data" / "hellaswag" / "hellaswag_val.jsonl"
MODEL_DIRS = (REPO_ROOT / "models" / "tiny200-mlx", REPO_ROOT / "models" / "gpt2-mlx")


@pytest.fixture(scope="module")
def model_dir() -> Path:
    for d in MODEL_DIRS:
        if (d / "config.json").is_file() and any(d.glob("tokenizer*.json")):
            return d
    pytest.skip(f"no exported model available (looked for {[str(d) for d in MODEL_DIRS]})")


@pytest.fixture(scope="module")
def bridge_lm(model_dir: Path):
    """The shipped adapter: ``mlx_lm.evaluate.MLXLM``, chat template off."""
    return lmeval.build_lm(str(model_dir))


@pytest.fixture(scope="module")
def r52_lm(model_dir: Path):
    from r52.eval.lm import LM

    return LM.load(model_dir)


@pytest.fixture(scope="module")
def all_task_names() -> set[str]:
    return lmeval.known_task_names()


# ======================================================================================
# Suite definitions resolve to real lm-eval tasks
# ======================================================================================


@pytest.mark.parametrize("spec", [*QUICK, *STANDARD], ids=lambda s: s.task)
def test_suite_tasks_exist_in_lm_eval(spec: TaskSpec, all_task_names: set[str]) -> None:
    assert spec.task in all_task_names, f"{spec.task} is not an lm-eval 0.4.13 task"


def test_suites_are_the_documented_sets() -> None:
    assert set(SUITES) == {"quick", "standard"}
    assert [s.task for s in QUICK] == [
        "arc_challenge", "piqa", "winogrande", "lambada_openai", "hellaswag"
    ]
    assert {s.task for s in STANDARD} == {
        "mmlu", "gpqa_diamond_zeroshot", "ifeval", "humaneval", "hendrycks_math500",
        "gsm8k", "mmlu_pro",
    }
    assert all(not s.generative for s in QUICK), "quick must be loglikelihood only"
    # Cheap first (research/05 §7.2): every loglikelihood task precedes every generative one.
    kinds = [s.generative for s in STANDARD]
    assert kinds == sorted(kinds), "the standard suite must run loglikelihood tasks first"


def test_names_the_brief_asks_for_do_not_exist_under_those_spellings(
    all_task_names: set[str],
) -> None:
    """Why the suite says ``gpqa_diamond_zeroshot`` / ``hendrycks_math500``.

    lm-eval 0.4.13 has no task literally named ``gpqa_diamond`` or ``math_500``; pinning the
    substitutes here means a future lm-eval that *adds* them fails this test loudly instead
    of leaving the suite quietly pointing at the older spelling.
    """
    assert "gpqa_diamond" not in all_task_names
    assert "math_500" not in all_task_names
    assert "gpqa_diamond_zeroshot" in all_task_names
    assert "hendrycks_math500" in all_task_names  # HuggingFaceH4/MATH-500


def test_few_shot_counts_match_the_spec() -> None:
    shots = {s.task: s.num_fewshot for s in STANDARD}
    assert shots["mmlu"] == 5
    assert shots["gsm8k"] == 8
    assert shots["mmlu_pro"] == 5
    assert shots["ifeval"] == 0
    assert shots["humaneval"] == 0
    assert all(s.num_fewshot == 0 for s in QUICK)


def test_generative_tasks_all_cap_their_generation() -> None:
    """Upstream reads the wrong gen-kwarg name, so an uncapped task generates 8,192 tokens
    per prompt (docs/DEVIATIONS.md L2).  Every generative spec must carry its own cap."""
    for spec in (*QUICK, *STANDARD):
        if spec.generative:
            assert spec.max_gen_tokens > 0, spec.task


def test_humaneval_is_gated_behind_confirm_run_unsafe_code() -> None:
    spec = next(s for s in STANDARD if s.task == "humaneval")
    assert spec.unsafe_code
    run = lmeval.run_task(object(), spec, confirm_run_unsafe_code=False, verbose=False)
    assert run.skipped and "--confirm-run-unsafe-code" in run.skipped
    assert run.value is None


def test_gpqa_declares_its_gated_dataset_and_the_skip_message_is_actionable() -> None:
    spec = next(s for s in STANDARD if s.task == "gpqa_diamond_zeroshot")
    assert spec.gated == "Idavidrein/gpqa"
    # The exact error lm-eval raises when the terms have not been accepted.
    exc = RuntimeError("Dataset 'Idavidrein/gpqa' is a gated dataset on the Hub. "
                       "You must be authenticated to access it.")
    reason = lmeval._skip_reason(exc, spec)
    assert reason and "gated" in reason and "huggingface.co/datasets/Idavidrein/gpqa" in reason
    # An unrecognised failure is not a skip -- run_task re-raises it.
    assert lmeval._skip_reason(ValueError("some other failure"), spec) is None


def test_bar_benchmark_ids_resolve() -> None:
    """Every mapped benchmark id must exist in bar.yaml or the gap table silently drops it."""
    from r52.bar.gap import load_bar

    bar = load_bar()
    known = {b.id for b in bar.benchmarks}
    mapped = {s.task: s.benchmark for s in (*QUICK, *STANDARD) if s.benchmark}
    assert mapped == {
        "arc_challenge": "arc-challenge",
        "hellaswag": "hellaswag",
        "gpqa_diamond_zeroshot": "gpqa-diamond",
        "ifeval": "ifeval",
        "humaneval": "humaneval",
        "hendrycks_math500": "math-500",
        "gsm8k": "gsm8k",
        "mmlu_pro": "mmlu-pro",
    }
    for task, bid in mapped.items():
        assert bid in known, f"{task} -> {bid} is not a bar.yaml benchmark id"
    # Units match the bar row, except HellaSwag: r52.eval.hellaswag already owns
    # `hellaswag` + "% acc_norm" with llm.c's protocol (docs/DEVIATIONS.md E2 / L4).
    for spec in (*QUICK, *STANDARD):
        if not spec.benchmark:
            continue
        expected = bar.benchmark(spec.benchmark).unit
        if spec.task == "hellaswag":
            assert spec.unit == "% acc_norm (lm-eval)" != expected
        else:
            assert spec.unit == expected, f"{spec.task}: {spec.unit!r} != bar {expected!r}"


def test_tasks_without_a_bar_id_fall_back_to_the_task_name() -> None:
    for spec in (*QUICK, *STANDARD):
        assert spec.benchmark_id == (spec.benchmark or spec.task)
    unmapped = {s.task for s in (*QUICK, *STANDARD) if not s.benchmark}
    assert unmapped == {"piqa", "winogrande", "lambada_openai", "mmlu"}


def test_resolve_tasks() -> None:
    assert [s.task for s in resolve_tasks("quick", None)] == [s.task for s in QUICK]
    # An explicit task already in a suite keeps that suite's spec; a new one gets a generic.
    specs = resolve_tasks(None, ["gsm8k", "boolq"])
    assert [s.task for s in specs] == ["gsm8k", "boolq"]
    assert specs[0].num_fewshot == 8 and specs[0].benchmark == "gsm8k"
    assert specs[1].benchmark is None and specs[1].benchmark_id == "boolq"
    # No duplicate when --suite and --tasks overlap.
    assert [s.task for s in resolve_tasks("quick", ["piqa"])] == [s.task for s in QUICK]
    # Overrides.
    assert all(s.num_fewshot == 2 for s in resolve_tasks("quick", None, num_fewshot=2))
    assert all(s.max_gen_tokens == 16
               for s in resolve_tasks("standard", None, max_gen_tokens=16))
    with pytest.raises(ValueError, match="unknown suite"):
        resolve_tasks("enormous", None)


# ======================================================================================
# Metric picking
# ======================================================================================


def test_pick_metric_prefers_exact_filtered_key() -> None:
    block = {
        "alias": "gsm8k",
        "exact_match,strict-match": 0.31,
        "exact_match_stderr,strict-match": 0.01,
        "exact_match,flexible-extract": 0.42,
    }
    spec = next(s for s in STANDARD if s.task == "gsm8k")
    assert pick_metric(block, spec.metric) == ("exact_match,strict-match", 0.31)


def test_pick_metric_falls_back_to_bare_name_then_to_anything_numeric() -> None:
    assert pick_metric({"acc_norm,none": 0.5, "acc,none": 0.4}, ("acc_norm",)) == \
        ("acc_norm,none", 0.5)
    # Not in `preferred` at all -> first numeric non-stderr entry.
    key, value = pick_metric({"alias": "x", "sample_len": 3, "f1,none": 0.7,
                              "f1_stderr,none": 0.1}, ("acc",))
    assert (key, value) == ("f1,none", 0.7)
    # stderr, alias, name and sample_len are never the headline.
    with pytest.raises(KeyError):
        pick_metric({"alias": "x", "name": "x", "acc_stderr,none": 0.1}, ("acc",))


def test_stderr_is_matched_to_the_chosen_filter() -> None:
    block = {"exact_match,strict-match": 0.3, "exact_match_stderr,strict-match": 0.02,
             "exact_match,flexible-extract": 0.4, "exact_match_stderr,flexible-extract": 0.03}
    assert lmeval._stderr_for(block, "exact_match,strict-match") == 0.02
    assert lmeval._stderr_for(block, "exact_match,flexible-extract") == 0.03
    assert lmeval._stderr_for({"acc,none": 0.5}, "acc,none") is None


# ======================================================================================
# git without git
# ======================================================================================


def test_git_commit_reads_refs_without_running_git(tmp_path: Path) -> None:
    sha = "4bd7ba570bfb13dcc4c39c3bf1cc4bcb963e4933"
    gitdir = tmp_path / ".git"
    (gitdir / "refs" / "heads").mkdir(parents=True)
    (gitdir / "HEAD").write_text("ref: refs/heads/main\n")
    (gitdir / "refs" / "heads" / "main").write_text(sha + "\n")
    assert git_commit_from_files(tmp_path) == sha[:7]

    # packed-refs fallback
    (gitdir / "refs" / "heads" / "main").unlink()
    (gitdir / "packed-refs").write_text(
        "# pack-refs with: peeled fully-peeled sorted\n"
        f"{sha} refs/heads/main\n{sha} refs/remotes/origin/main\n"
    )
    assert git_commit_from_files(tmp_path) == sha[:7]

    # detached HEAD
    (gitdir / "HEAD").write_text(sha + "\n")
    assert git_commit_from_files(tmp_path) == sha[:7]

    # a worktree, whose .git is a file pointing at the real git dir
    wt = tmp_path / "wt"
    wt.mkdir()
    (wt / ".git").write_text(f"gitdir: {gitdir}\n")
    assert git_commit_from_files(wt) == sha[:7]

    # nothing resolvable -> "unknown", never a guess
    assert git_commit_from_files(tmp_path / "nope") == "unknown"
    (gitdir / "HEAD").write_text("ref: refs/heads/gone\n")
    assert git_commit_from_files(tmp_path) == "unknown"


def test_git_commit_of_this_repo_is_a_short_hash() -> None:
    """The CLI is preferred (it can add "+dirty"); the file reader is the fallback."""
    commit = git_commit()
    assert commit == "unknown" or commit.rstrip("+dirty")[:7].isalnum()
    assert commit in ("unknown", git_commit_from_files(), git_commit_from_files() + "+dirty")


# ======================================================================================
# Model-spec validation
# ======================================================================================


def test_checkpoint_dir_is_rejected_with_the_export_command(tmp_path: Path) -> None:
    ckpt = tmp_path / "best"
    ckpt.mkdir()
    (ckpt / "meta.json").write_text("{}")
    (ckpt / "model.safetensors").write_bytes(b"")
    with pytest.raises(ValueError, match=r"r52\.export"):
        lmeval._check_model_spec(str(ckpt))


def test_export_without_tokenizer_files_is_rejected(tmp_path: Path) -> None:
    d = tmp_path / "m-mlx"
    d.mkdir()
    (d / "config.json").write_text("{}")
    (d / "model.safetensors").write_bytes(b"")
    with pytest.raises(ValueError, match="tokenizer"):
        lmeval._check_model_spec(str(d))
    # An unknown path is assumed to be an HF repo id and passed through untouched.
    assert lmeval._check_model_spec("mlx-community/Q-8bit") == "mlx-community/Q-8bit"


def test_quantization_string() -> None:
    """research/05 §7.1: bf16 -> q4 costs 3.3 MMLU-Pro points, so the number depends on it."""
    assert lmeval.quantization_string({}, "bfloat16") == "none (bfloat16)"
    assert lmeval.quantization_string({"quantization": {"bits": 6, "group_size": 64}},
                                      "float16") == "q6 g64"
    assert lmeval.quantization_string({"quantization": {"bits": 4}}, "float16") == "q4"


# ======================================================================================
# The LM adapter
# ======================================================================================


def _short_hellaswag_examples(tokenizer, max_width: int, n: int = 3):
    """First ``n`` HellaSwag items whose four rows all fit in ``max_width`` tokens."""
    from r52.eval.hellaswag import Example, render_example

    if not HELLASWAG_JSONL.is_file():
        pytest.skip(f"{HELLASWAG_JSONL} not cached (run r52.eval.hellaswag once)")
    out = []
    with HELLASWAG_JSONL.open("r", encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            r = json.loads(line)
            ex = Example(ctx=r["ctx"], endings=tuple(r["endings"]), label=int(r["label"]),
                         ind=int(r.get("ind", 0)))
            rendered = render_example(ex, tokenizer, 0)
            if rendered.width <= max_width:
                out.append((ex, rendered))
            if len(out) == n:
                break
    if len(out) < n:  # pragma: no cover - only on a truncated cache
        pytest.skip("not enough short HellaSwag examples in the cached jsonl")
    return out


def test_loglikelihood_matches_hellaswag_scoring(bridge_lm, r52_lm) -> None:
    """The bridge and :mod:`r52.eval.hellaswag` must score the same continuations identically.

    ``r52.eval.hellaswag`` sums per-token NLL over the ending mask with no KV cache;
    ``MLXLM.loglikelihood`` sums fp32 log-probabilities with a cached context prefix.  Same
    arithmetic, two code paths -- measured agreement on this machine is ~1.5e-5 nats on a
    ~100-nat total, i.e. fp32 rounding.  A real divergence here (different tokenization at
    the context/continuation boundary, a different masking convention) would silently make
    the lm-eval numbers incomparable with the rest of ``r52/eval/``.
    """
    from r52.eval.lm import pad_stack

    picked = _short_hellaswag_examples(r52_lm.tokenizer, min(100, r52_lm.max_context))

    ours: list[float] = []
    for _, rendered in picked:
        tokens = pad_stack(rendered.tokens)
        mask = pad_stack(rendered.masks).astype(mx.float32)
        nll, _ = r52_lm.scores(tokens)
        # llm.c's index convention: the mask shifts by one, so scoring starts at the last
        # context token -- the position that predicts the first ending token.
        summed = (nll[:, :-1] * mask[:, 1:]).sum(axis=1)
        mx.eval(summed)
        ours.extend((-np.asarray(summed, dtype=np.float64)).tolist())

    pairs = [(ex.ctx, " " + ending) for ex, _ in picked for ending in ex.endings]
    theirs = lmeval.loglikelihood(bridge_lm, pairs)

    assert len(theirs) == len(ours) == 4 * len(picked)
    for (value, is_greedy), reference in zip(theirs, ours, strict=True):
        assert isinstance(value, float) and isinstance(is_greedy, bool)
        assert value < 0.0
        assert abs(value - reference) < 1e-3, f"{value} vs {reference}"


def test_generate_until_returns_strings_and_respects_stop_sequences(bridge_lm) -> None:
    prompts = ["The capital of France is", "Question: what is 2 + 2?\nAnswer:"]
    stops = ["\n", "Question:"]
    out = lmeval.generate_until(bridge_lm, prompts, stops, max_gen_tokens=8)
    assert len(out) == len(prompts)
    for text in out:
        assert isinstance(text, str)
        for stop in stops:
            assert stop not in text, f"stop sequence {stop!r} survived in {text!r}"
    # A stop string that cannot occur leaves the continuation intact, so this also shows
    # that the truncation above is what removed the stop, not an empty generation.
    free = lmeval.generate_until(bridge_lm, prompts[:1], ["<<<never-generated>>>"],
                                 max_gen_tokens=8)
    assert len(free) == 1 and isinstance(free[0], str)


def test_generate_until_restores_the_generation_cap(bridge_lm) -> None:
    """``_max_tokens`` also drives loglikelihood truncation, so it must not leak between tasks."""
    before = bridge_lm._max_tokens
    lmeval.generate_until(bridge_lm, ["hello"], ["\n"], max_gen_tokens=4)
    assert bridge_lm._max_tokens == before


def test_model_description_records_what_the_number_depends_on(bridge_lm, model_dir) -> None:
    described = lmeval.model_description(bridge_lm, str(model_dir))
    assert described["path"] == str(model_dir)
    assert described["compute_dtype"] in ("bfloat16", "float16", "float32")
    assert described["quantization"].startswith(("none", "q"))
    assert described["max_context"] >= 128
    assert "vocab" in described["tokenizer"]
    # Every r52 export ships a chat template so mlx_lm.chat works; base-model evals must not
    # silently switch into chat mode because of it.
    assert described["use_chat_template"] is False


# ======================================================================================
# Results schema
# ======================================================================================


def _fake_run(**kw) -> TaskRun:
    spec = kw.pop("spec", next(s for s in QUICK if s.task == "arc_challenge"))
    base = {
        "value": 21.5, "metric_key": "acc_norm,none", "stderr": 1.2, "n_examples": 100,
        "n_original": 1172, "num_fewshot": 0, "wall_clock_s": 42.5,
        "raw": {"alias": "arc_challenge", "acc_norm,none": 0.215,
                "acc_norm_stderr,none": 0.012},
        "extra": {"lm_eval_version": "0.4.13", "task_version": 1.0},
    }
    base.update(kw)
    return TaskRun(spec=spec, **base)


DESCRIBED = {
    "path": "models/gpt2-mlx", "model_type": "gpt2", "max_context": 1024,
    "compute_dtype": "bfloat16", "quantization": "none (bfloat16)",
    "tokenizer": "models__gpt2-mlx (GPT2Tokenizer, vocab 50257)", "use_chat_template": False,
}
COMMAND = "python -m r52.eval.lmeval --model models/gpt2-mlx --suite quick --limit 100"


def test_result_json_and_gap_row_have_the_report_schema(tmp_path: Path) -> None:
    from r52.bar.gap import load_local_results
    from r52.eval.report import write_result

    result = lmeval.result_for(_fake_run(), DESCRIBED, model_id="gpt2-124m", limit=100,
                               command=COMMAND, commit="abc1234")
    path = write_result(result, "gpt2-124m-reference", "lmeval_arc_challenge",
                        results_root=tmp_path)
    assert path == tmp_path / "gpt2-124m-reference" / "lmeval_arc_challenge.json"

    # 1. the per-eval record: structured `conditions`, the §6 / §1.3 reproduction fields.
    payload = json.loads(path.read_text())
    assert payload["model"] == "gpt2-124m"
    assert payload["benchmark"] == "arc-challenge"
    assert payload["unit"] == "% acc_norm"
    assert payload["value"] == pytest.approx(21.5)
    assert payload["commit"] == "abc1234"
    assert payload["wall_clock_s"] == pytest.approx(42.5)
    assert payload["command"] == COMMAND
    assert payload["machine"] and payload["date"]
    conditions = payload["conditions"]
    assert isinstance(conditions, dict)
    for key in ("tokenizer", "block_size", "limit", "n_examples", "few_shot", "command",
                "harness", "task", "metric", "quantization"):
        assert key in conditions, key
    assert conditions["limit"] == 100
    assert conditions["n_examples"] == 100
    assert "lm-evaluation-harness" in conditions["harness"]
    assert payload["metrics"]["lm_eval_results"]["acc_norm,none"] == 0.215
    assert payload["metrics"]["n_examples_full_set"] == 1172
    assert payload["metrics"]["bar_benchmark"] == "arc-challenge"

    # 2. the flat record gap.py reads.
    scores = load_local_results(tmp_path)
    assert len(scores) == 1
    score = scores[0]
    assert (score.model, score.benchmark, score.unit) == ("gpt2-124m", "arc-challenge",
                                                          "% acc_norm")
    assert score.value == pytest.approx(21.5)
    assert score.local and score.primary
    assert score.commit == "abc1234" and score.command == COMMAND
    assert "limit 100" in score.conditions and "few-shot 0" in score.conditions

    # 3. one markdown row, kept out of docs/RESULTS.md by `results_root`.
    md = (tmp_path / "RESULTS.md").read_text()
    assert "| lmeval_arc_challenge |" in md and "21.50 % acc_norm" in md
    assert "limit 100" in md and COMMAND in md
    assert not (tmp_path / "docs").exists()


def test_two_tasks_become_two_files_and_two_gap_rows(tmp_path: Path) -> None:
    from r52.bar.gap import load_local_results
    from r52.eval.report import write_result

    gsm8k = next(s for s in STANDARD if s.task == "gsm8k")
    for run in (_fake_run(),
                _fake_run(spec=gsm8k, value=1.0, metric_key="exact_match,strict-match",
                          n_original=1319)):
        result = lmeval.result_for(run, DESCRIBED, model_id="gpt2-124m", limit=100,
                                   command=COMMAND, commit="abc1234")
        write_result(result, "r", f"lmeval_{run.spec.task}", results_root=tmp_path)
    assert (tmp_path / "r" / "lmeval_arc_challenge.json").is_file()
    assert (tmp_path / "r" / "lmeval_gsm8k.json").is_file()
    assert {s.benchmark for s in load_local_results(tmp_path)} == {"arc-challenge", "gsm8k"}


def test_limit_is_always_reported_even_when_there_is_none() -> None:
    """research/05 §7.2: a --limit-ed score is not comparable to a full-set number, so the
    limit is part of every row -- including the rows that do not have one."""
    from r52.eval.report import conditions_string

    result = lmeval.result_for(_fake_run(n_examples=1172), DESCRIBED, model_id="gpt2-124m",
                               limit=None, command=COMMAND, commit="abc1234")
    assert result.conditions["limit"] == "none (full set)"
    assert "limit none (full set)" in conditions_string(result.conditions)


def test_patch_upstream_fixes_an_empty_stop_list() -> None:
    """mlx-lm 0.31.3 crashes in ``_rstrip_until`` when a task has no stop strings.

    lm-eval's ``ifeval`` sets ``generation_kwargs: {until: []}``, and upstream ends in
    ``s[: min(f)]`` over an empty list -- so every IFEval run died *after* generating.
    ``patch_upstream`` makes an empty stop list a no-op, which is the intended semantics.
    """
    from mlx_lm import evaluate as upstream

    lmeval.patch_upstream()
    lmeval.patch_upstream()  # idempotent
    assert upstream._rstrip_until("hello world", []) == "hello world"
    assert upstream._rstrip_until("keep\nDROP", ["\n"]) == "keep"
    assert upstream._rstrip_until("no stop here", ["Question:"]) == "no stop here"


def test_command_string_states_the_limit() -> None:
    parser = lmeval.build_parser()
    with_limit = lmeval._command(parser.parse_args(
        ["--model", "models/gpt2-mlx", "--suite", "quick", "--limit", "100"]))
    assert with_limit.endswith("--limit 100")
    assert "--model models/gpt2-mlx" in with_limit and "--suite quick" in with_limit
    without = lmeval._command(parser.parse_args(["--model", "m", "--tasks", "gsm8k"]))
    assert "no --limit (full set)" in without
    unsafe = lmeval._command(parser.parse_args(
        ["--model", "m", "--suite", "standard", "--confirm-run-unsafe-code"]))
    assert "--confirm-run-unsafe-code" in unsafe
    # A truncated generation can lose the answer, so the cap belongs in the command too.
    capped = lmeval._command(parser.parse_args(
        ["--model", "m", "--tasks", "gsm8k", "--limit", "5", "--max-gen-tokens", "128"]))
    assert "--max-gen-tokens 128" in capped and "--limit 5" in capped


def test_skipped_tasks_write_nothing() -> None:
    """A gated or unsafe-code skip must not leave a row claiming a score of zero."""
    run = _fake_run(value=None, metric_key=None, stderr=None, n_examples=0,
                    skipped="dataset Idavidrein/gpqa is gated")
    assert run.value is None
    with pytest.raises(AssertionError):
        lmeval.result_for(run, DESCRIBED, model_id="gpt2-124m", limit=5, command=COMMAND)
