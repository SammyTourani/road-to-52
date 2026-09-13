# Copyright 2026 The road-to-52 authors.
# SPDX-License-Identifier: Apache-2.0
"""lm-evaluation-harness bridge: MMLU / MMLU-Pro / GSM8K / IFEval / HumanEval on our models.

    python -m r52.eval.lmeval --model models/gpt2-mlx --suite quick --limit 100
    python -m r52.eval.lmeval --model models/b1-mlx  --tasks gsm8k --limit 5 --report
    scripts/eval_lmeval.sh models/gpt2-mlx quick --limit 100

``docs/ARCHITECTURE.md`` §6, last paragraph: "Later phases add ``mlx_lm.evaluate`` (lm-eval
bridge) for MMLU-Pro/GPQA/GSM8K/IFEval on exported and post-trained models."  This is that
phase.  Unlike the rest of :mod:`r52.eval` -- which scores through :class:`r52.eval.lm.LM` so
that r52 checkpoints and mlx-lm directories share one code path -- this module evaluates
**mlx-lm-loadable directories only**: our exports (``model_type: "nanochat"`` natively and
``"r52gpt"`` through the plugin file the export copies next to the weights), the GPT-2
reference ``models/gpt2-mlx``, and any HF / mlx-community repo id.  Point it at a raw
``runs/<run>/ckpt/best`` and it tells you to run ``python -m r52.export`` first.

Which path shipped
------------------
``mlx_lm.evaluate`` registers an ``mlxlm`` :class:`lm_eval.api.model.LM` subclass built on
``mlx_lm.utils.load`` -- which honours ``model_file: "r52gpt.py"`` -- so **it works on our
plugin exports unmodified**, verified with ``--limit 5 --tasks arc_easy`` on
``models/tiny200-mlx``.  We therefore *wrap* :class:`mlx_lm.evaluate.MLXLM` by direct import
rather than reimplementing ``loglikelihood`` / ``loglikelihood_rolling`` / ``generate_until``
on top of :class:`r52.eval.lm.LM`.  See ``docs/DEVIATIONS.md``, "lm-eval bridge", for the
three upstream rough edges this wrapper works around.

Task suites
-----------
``quick``
    ``arc_challenge piqa winogrande lambada_openai hellaswag`` -- all loglikelihood, 0-shot,
    no generation.  124 s at ``--limit 100`` for the GPT-2 124M reference on a GPU already
    holding a pretraining run.
``standard``
    ``mmlu`` (5-shot), ``gpqa_diamond_zeroshot`` (gated dataset), then the generative half:
    ``ifeval``, ``humaneval`` (needs ``--confirm-run-unsafe-code``), ``hendrycks_math500``
    (= MATH-500), ``gsm8k`` (8-shot), ``mmlu_pro`` (5-shot).  Five of the seven are
    generative.  **Hours to days at full size** (``research/05-apple-silicon-training.md``
    §7.3 puts full MMLU-Pro at 2-4 days on this M4) -- always pass ``--limit``.

The cheap-first ordering of research/05 §7.2 is the order the suites run in, and within a
suite loglikelihood tasks run before generative ones.  lm-eval 0.4.13 has no task literally
named ``gpqa_diamond`` or ``math_500``; ``docs/DEVIATIONS.md`` L5 records what it has instead.

Reporting
---------
Every task is one :func:`r52.eval.report.write_result` call, so each produces
``results/<run>/lmeval_<task>.json`` (the §6 schema: structured ``conditions``, ``metrics``,
``machine``, ``wall_clock_s``) and one entry in ``results/<run>/eval.json`` (the flat schema
``r52/bar/gap.py`` reads).  Benchmark ids are ``r52/bar/bar.yaml``'s where one exists
(``arc-challenge``, ``gsm8k``, ``ifeval``, ``humaneval``, ``mmlu-pro``, ``gpqa-diamond``,
``math-500``, ``hellaswag``); tasks with no bar id (``piqa``, ``winogrande``,
``lambada_openai``, ``mmlu``) keep the lm-eval task name and are silently ignored by the gap
table, which is what ``gap.py`` does with any unknown id.

**Always report the limit.**  ``conditions["limit"]`` is never absent: it is the integer or
the string ``"none (full set)"``, so every row in ``docs/RESULTS.md`` states it.  A
``--limit``-ed score must never be compared to a published full-set number
(research/05 §7.2).
"""

from __future__ import annotations

import argparse
import json
import os
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import mlx.core as mx

from .lm import DEFAULT_MEMORY_LIMIT_GIB, configure_runtime

__all__ = [
    "QUICK",
    "STANDARD",
    "SUITES",
    "TaskSpec",
    "build_lm",
    "generate_until",
    "git_commit",
    "git_commit_from_files",
    "loglikelihood",
    "main",
    "model_description",
    "patch_upstream",
    "pick_metric",
    "resolve_tasks",
    "run_suite",
    "run_task",
    "tokenizer_string",
]

REPO_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_BATCH_SIZE = 8
"""Requests per ``batch_generate`` call.  Only ``generate_until`` and
``loglikelihood_rolling`` batch: upstream ``loglikelihood`` walks one context at a time and
reuses a prompt cache across that context's continuations."""


# ======================================================================================
# Task specifications
# ======================================================================================


@dataclass(frozen=True)
class TaskSpec:
    """One lm-eval task plus everything r52 needs to record its number honestly.

    Attributes:
        task: the lm-eval task name.  Must exist in ``TaskManager().all_tasks``.
        num_fewshot: shots, ``None`` = the task's own default.
        metric: preferred result keys, first match wins.  A key may carry lm-eval's filter
            suffix (``"exact_match,strict-match"``) or be a bare metric name.
        unit: r52 unit string.  Where ``benchmark`` is a ``bar.yaml`` id this is *that row's*
            unit, so the gap-table cell reads consistently.
        benchmark: ``r52/bar/bar.yaml`` benchmark id, or ``None`` to use ``task``.
        output_type: ``"loglikelihood"`` or ``"generate_until"`` -- documentation, and the
            switch that decides whether ``max_gen_tokens`` is applied.
        max_gen_tokens: generation cap per request.  Upstream would otherwise generate
            ``DEFAULT_MAX_TOKENS`` (8,192) per prompt; see docs/DEVIATIONS.md L2.
        scale: multiplier from lm-eval's value to ``unit`` (100 for a fraction -> percent).
        unsafe_code: task executes model-written code; needs ``--confirm-run-unsafe-code``.
        gated: HF dataset repo id that requires accepting terms, if any.
        note: free text carried into the results JSON.
    """

    task: str
    num_fewshot: int | None
    metric: tuple[str, ...]
    unit: str
    benchmark: str | None = None
    output_type: str = "loglikelihood"
    max_gen_tokens: int = 0
    scale: float = 100.0
    unsafe_code: bool = False
    gated: str | None = None
    note: str = ""

    @property
    def benchmark_id(self) -> str:
        return self.benchmark or self.task

    @property
    def generative(self) -> bool:
        return self.output_type == "generate_until"


# Cheap first (research/05 §7.2): loglikelihood only, 0-shot, no generation.
QUICK: tuple[TaskSpec, ...] = (
    TaskSpec("arc_challenge", 0, ("acc_norm", "acc"), "% acc_norm", "arc-challenge",
             note="1,172 test items x 4 choices -- the cheapest task in the set"),
    TaskSpec("piqa", 0, ("acc_norm", "acc"), "% acc_norm",
             note="1,838 validation items x 2 choices"),
    TaskSpec("winogrande", 0, ("acc",), "% acc",
             note="1,267 validation items x 2 choices"),
    TaskSpec("lambada_openai", 0, ("acc",), "% acc",
             note="5,153 items; last-word prediction, also reports perplexity"),
    # `hellaswag` collides with r52.eval.hellaswag's own bar.yaml row, which is llm.c's
    # protocol, not lm-eval's (docs/DEVIATIONS.md E2).  The distinct unit keeps both local
    # numbers in results/<run>/eval.json instead of one silently overwriting the other.
    TaskSpec("hellaswag", 0, ("acc_norm", "acc"), "% acc_norm (lm-eval)", "hellaswag",
             note="lm-eval protocol: activity-label prefix, char-length acc_norm -- NOT the "
                  "llm.c protocol r52.eval.hellaswag implements"),
)

# Expensive.  Five of seven are generative; never run this without --limit.
STANDARD: tuple[TaskSpec, ...] = (
    TaskSpec("mmlu", 5, ("acc",), "% acc",
             note="57 subjects, 14,042 test items, loglikelihood over A/B/C/D"),
    TaskSpec("gpqa_diamond_zeroshot", 0, ("acc_norm", "acc"), "%", "gpqa-diamond",
             gated="Idavidrein/gpqa",
             note="198 items, loglikelihood over (A)-(D); gated dataset"),
    TaskSpec("ifeval", 0, ("prompt_level_strict_acc", "inst_level_strict_acc"), "%", "ifeval",
             output_type="generate_until", max_gen_tokens=1280,
             note="541 items, programmatic constraint check, no judge model"),
    TaskSpec("humaneval", 0, ("pass_at_k", "pass@1", "pass_at_1"), "% pass@1", "humaneval",
             output_type="generate_until", max_gen_tokens=1024, unsafe_code=True,
             note="164 items; EXECUTES model-written code -- needs --confirm-run-unsafe-code"),
    TaskSpec("hendrycks_math500", 0, ("exact_match",), "%", "math-500",
             output_type="generate_until", max_gen_tokens=1024,
             note="HuggingFaceH4/MATH-500, 500 items -- lm-eval 0.4.13 has no task literally "
                  "named math_500; hendrycks_math500 is that dataset"),
    TaskSpec("gsm8k", 8, ("exact_match,strict-match", "exact_match,flexible-extract",
                          "exact_match"), "%", "gsm8k",
             output_type="generate_until", max_gen_tokens=256,
             note="1,319 items; few-shot exemplars are the dataset's own worked solutions, so "
                  "this is CoT.  gsm8k_cot is the fixed 8-exemplar PaLM-prompt variant"),
    TaskSpec("mmlu_pro", 5, ("exact_match,custom-extract", "exact_match"), "%", "mmlu-pro",
             output_type="generate_until", max_gen_tokens=2048,
             note="12,032 items, 10 options, CoT-generative -- 2-4 days unlimited on this M4"),
)

SUITES: dict[str, tuple[TaskSpec, ...]] = {"quick": QUICK, "standard": STANDARD}

_FALLBACK_METRICS = ("acc_norm", "acc", "exact_match", "pass_at_k", "pass@1",
                     "prompt_level_strict_acc", "f1", "em")


def resolve_tasks(
    suite: str | None,
    tasks: Sequence[str] | None,
    *,
    num_fewshot: int | None = None,
    max_gen_tokens: int = 0,
) -> list[TaskSpec]:
    """Suite name and/or explicit task names -> :class:`TaskSpec` list, in run order.

    Explicit ``--tasks`` that are also in a suite inherit that suite's spec (shots, metric
    preference, bar id); anything else gets a generic spec whose metric is picked by
    :func:`pick_metric`'s fallback list.
    """
    known = {s.task: s for s in (*QUICK, *STANDARD)}
    out: list[TaskSpec] = []
    if suite:
        if suite not in SUITES:
            raise ValueError(f"unknown suite {suite!r} (choose from {sorted(SUITES)})")
        out.extend(SUITES[suite])
    for name in tasks or ():
        spec = known.get(name)
        if spec is None:
            spec = TaskSpec(name, None, _FALLBACK_METRICS, "%", output_type="loglikelihood",
                            note="ad-hoc --tasks entry; metric chosen by fallback order")
        if spec.task not in {s.task for s in out}:
            out.append(spec)
    if num_fewshot is not None:
        out = [_replace(s, num_fewshot=num_fewshot) for s in out]
    if max_gen_tokens:
        out = [_replace(s, max_gen_tokens=max_gen_tokens) for s in out]
    return out


def _replace(spec: TaskSpec, **kw: Any) -> TaskSpec:
    data = {f: getattr(spec, f) for f in spec.__dataclass_fields__}
    data.update(kw)
    return TaskSpec(**data)


def known_task_names() -> set[str]:
    """``TaskManager().all_tasks`` as a set (used to validate suites before spending GPU)."""
    from lm_eval.tasks import TaskManager

    return set(TaskManager().all_tasks)


# ======================================================================================
# Model loading
# ======================================================================================


def _check_model_spec(spec: str) -> str:
    """Fail early and usefully on a directory ``mlx_lm.utils.load`` cannot open."""
    p = Path(spec)
    if not p.exists():
        return spec  # an HF / mlx-community repo id; let the Hub resolve it
    if not p.is_dir():
        raise ValueError(f"{p} is not a directory")
    if not (p / "config.json").is_file():
        if (p / "meta.json").is_file():
            raise ValueError(
                f"{p} is an r52 checkpoint, not an mlx-lm model directory.  The lm-eval "
                f"bridge evaluates exported models only:\n"
                f"    python -m r52.export {p} models/<name>-mlx\n"
                f"    python -m r52.eval.lmeval --model models/<name>-mlx --suite quick"
            )
        raise ValueError(f"{p}/config.json is missing -- not an mlx-lm model directory")
    if not any(p.glob("tokenizer*.json")):
        raise ValueError(
            f"{p} has no tokenizer files.  Unlike r52.eval.lm.LM (which always uses GPT-2 "
            f"tiktoken), lm-eval renders prompts as *text* and needs the model's own "
            f"tokenizer.  Re-export without --no-tokenizer."
        )
    return spec


_PATCHED = False


def patch_upstream() -> None:
    """Fix ``mlx_lm.evaluate._rstrip_until([])`` in place.  Idempotent.

    mlx-lm 0.31.3's ``_rstrip_until(s, untils)`` ends in ``s[: min(f)]``, and ``min`` of an
    empty sequence raises.  lm-eval's ``ifeval`` sets ``generation_kwargs: {until: []}`` --
    "generate to EOS or the token cap, do not truncate" -- so **every IFEval run crashes**
    after generating, in the post-processing step.  Same for any other task with no stop
    strings.  The correct behaviour for an empty stop list is to return the completion
    unchanged, which is what this shim does; with a non-empty list it defers to upstream.
    Reported in docs/DEVIATIONS.md, "lm-eval bridge" (L1).
    """
    global _PATCHED
    if _PATCHED:
        return
    from mlx_lm import evaluate as _up

    original = _up._rstrip_until

    def _rstrip_until(s, untils):
        return s if not untils else original(s, untils)

    _up._rstrip_until = _rstrip_until
    _PATCHED = True


def build_lm(
    model: str,
    *,
    batch_size: int = DEFAULT_BATCH_SIZE,
    use_chat_template: bool = False,
    trust_remote_code: bool = False,
    max_tokens: int | None = None,
    temp: float = 0.0,
    top_p: float = 1.0,
    top_k: int = 0,
):
    """Load ``model`` as ``mlx_lm.evaluate.MLXLM`` (the registered ``mlxlm`` lm-eval backend).

    ``use_chat_template`` defaults to **False**, not to upstream's "True whenever the
    tokenizer has one".  Every r52 export ships a chat template (``r52.chat_template``) so
    that ``mlx_lm.chat`` works on base models too, and letting that flip these base-model
    loglikelihood evals into chat mode would silently change every score.
    """
    from mlx_lm.evaluate import MLXLM
    from mlx_lm.sample_utils import make_sampler

    patch_upstream()
    spec = _check_model_spec(model)
    lm = MLXLM(
        spec,
        max_tokens=max_tokens,
        batch_size=batch_size,
        use_chat_template=use_chat_template,
        trust_remote_code=trust_remote_code,
        sampler=make_sampler(temp=temp, top_p=top_p, top_k=top_k),
    )
    return lm


def model_context(lm, override: int | None = None) -> int:
    """Usable context in tokens: ``--block-size``, else the config's own limit, else 2048."""
    if override:
        return int(override)
    cfg = getattr(getattr(lm._model, "args", None), "__dict__", {}) or {}
    for key in ("max_position_embeddings", "n_positions", "n_ctx"):
        value = cfg.get(key)
        if value:
            return int(value)
    return 2048


def model_description(lm, model_path: str) -> dict[str, Any]:
    """JSON-safe summary of what was loaded -- goes into every results file."""
    args = getattr(lm._model, "args", None)
    cfg = dict(getattr(args, "__dict__", {}) or {})
    dtype = _param_dtype(lm._model)
    return {
        "path": model_path,
        "model_type": cfg.get("model_type"),
        "hidden_size": cfg.get("hidden_size") or cfg.get("n_embd"),
        "num_hidden_layers": cfg.get("num_hidden_layers") or cfg.get("n_layer"),
        "vocab_size": cfg.get("vocab_size"),
        "max_context": model_context(lm),
        "compute_dtype": dtype,
        "quantization": quantization_string(cfg, dtype),
        "tokenizer": tokenizer_string(lm),
        "use_chat_template": bool(lm.use_chat_template),
    }


def tokenizer_string(lm) -> str:
    """``"models__gpt2-mlx (GPT2TokenizerFast, vocab 50257)"``.

    lm-eval renders prompts as *text*, so unlike the rest of :mod:`r52.eval` -- which pins
    GPT-2 tiktoken for everything -- the number depends on the tokenizer shipped in the model
    directory.  It therefore goes in ``conditions``, class and vocabulary size included.
    """
    inner = getattr(lm.tokenizer, "_tokenizer", lm.tokenizer)
    size = getattr(inner, "vocab_size", None)
    try:
        size = len(inner) if size is None else size
    except TypeError:  # pragma: no cover - exotic tokenizer
        size = None
    vocab = f", vocab {size}" if size else ""
    return f"{lm.tokenizer_name} ({type(inner).__name__}{vocab})"


def _param_dtype(model) -> str:
    from mlx.utils import tree_flatten

    for _, v in tree_flatten(model.parameters()):
        if isinstance(v, mx.array) and mx.issubdtype(v.dtype, mx.floating):
            return str(v.dtype).rsplit(".", 1)[-1]
    return "unknown"  # pragma: no cover - a model with no float parameters


def quantization_string(config: dict[str, Any], dtype: str) -> str:
    """``"none (bfloat16)"`` for our exports, ``"q4 g64"`` for a quantized mlx-community repo.

    research/05 §7.1 measured the cost on MMLU-Pro: bf16 -> q4 loses 3.3 points, q4 g32 loses
    2.6, **q6 loses 0.5**.  Evaluate a fine-tune at q6 or q8, never q4 -- and since the number
    depends on it, it is recorded in ``conditions`` for every run.
    """
    q = config.get("quantization")
    if not isinstance(q, dict) or not q:
        return f"none ({dtype})"
    bits = q.get("bits")
    group = q.get("group_size")
    head = f"q{bits}" if bits else "quantized"
    return f"{head} g{group}" if group else head


# ======================================================================================
# Thin request helpers (used by the tests, and handy from a REPL)
# ======================================================================================


def _instances(request_type: str, args_list: Sequence[tuple]) -> list:
    from lm_eval.api.instance import Instance

    return [
        Instance(request_type=request_type, doc={}, arguments=tuple(a), idx=i)
        for i, a in enumerate(args_list)
    ]


def loglikelihood(lm, pairs: Sequence[tuple[str, str]]) -> list[tuple[float, bool]]:
    """``[(context, continuation), ...]`` -> ``[(sum log p(continuation), is_greedy), ...]``."""
    return lm.loglikelihood(_instances("loglikelihood", pairs))


def generate_until(lm, prompts: Sequence[str], until: Sequence[str], max_gen_tokens: int = 32
                   ) -> list[str]:
    """Greedy continuation of each prompt, truncated at the first stop string."""
    previous = lm._max_tokens
    lm._max_tokens = int(max_gen_tokens)
    try:
        args = [(p, {"until": list(until), "do_sample": False, "temperature": 0.0})
                for p in prompts]
        return lm.generate_until(_instances("generate_until", args))
    finally:
        lm._max_tokens = previous


# ======================================================================================
# Metrics
# ======================================================================================


def pick_metric(block: dict[str, Any], preferred: Sequence[str]) -> tuple[str, float]:
    """Choose the headline number out of one lm-eval task result block.

    lm-eval keys are ``"<metric>,<filter>"`` (``"exact_match,strict-match"``).  Exact keys in
    ``preferred`` win, then bare metric names against any filter, then the first numeric
    non-stderr entry.  Raises :class:`KeyError` if the block holds no number.
    """
    numeric = {
        k: float(v)
        for k, v in block.items()
        if isinstance(v, (int, float)) and not isinstance(v, bool)
        and not k.split(",")[0].endswith("_stderr")
        and k.split(",")[0] not in ("alias", "name", "sample_len")
    }
    for name in preferred:
        if name in numeric:
            return name, numeric[name]
    for name in preferred:
        if "," in name:
            continue
        for key, value in numeric.items():
            if key.split(",")[0] == name:
                return key, value
    for key, value in numeric.items():
        return key, value
    raise KeyError(f"no numeric metric in {sorted(block)}")


def _stderr_for(block: dict[str, Any], metric_key: str) -> float | None:
    name, _, filt = metric_key.partition(",")
    candidate = f"{name}_stderr,{filt}" if filt else f"{name}_stderr"
    value = block.get(candidate)
    return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else None


# ======================================================================================
# git (the CLI is unavailable on this machine -- read the refs directly)
# ======================================================================================


def git_commit(root: Path | None = None) -> str:
    """Short HEAD hash, from the ``git`` CLI when it works and from ``.git`` when it does not.

    ``r52.eval.report.git_commit`` shells out to ``git rev-parse`` + ``git status``, which
    also gives it the ``"+dirty"`` suffix the rest of ``results/`` uses.  On this machine
    ``git`` has been intermittently unusable (an unaccepted Xcode command-line-tools license
    makes every invocation fail), and then it returns ``"unknown"`` -- which would leave every
    lm-eval row without the reproduction anchor ``docs/ARCHITECTURE.md`` §1.3 requires.  So
    the CLI is tried first and :func:`git_commit_from_files` is the fallback.

    The fallback cannot tell whether the tree is dirty -- that needs ``git status`` -- so it
    never appends ``"+dirty"``.  When it is what answered, the results JSON records
    ``git_dirty: "unknown"`` rather than guessing; a hash with no suffix from this bridge
    therefore means "dirtiness unknown", not "clean".
    """
    from .report import git_commit as _git_cli

    try:
        via_cli = _git_cli(Path(root) if root is not None else None)
    except Exception:  # pragma: no cover - report.git_commit already swallows its own errors
        via_cli = "unknown"
    return via_cli if via_cli != "unknown" else git_commit_from_files(root)


def git_commit_from_files(root: Path | None = None) -> str:
    """Short HEAD hash read out of ``.git`` **without running git**.

    Follows ``.git/HEAD`` into the loose ref or ``.git/packed-refs``, and handles both a
    detached HEAD and a worktree, whose ``.git`` is a file holding ``gitdir: <path>``.
    Returns ``"unknown"`` when nothing can be resolved -- never a guess.
    """
    root = Path(root or REPO_ROOT)
    gitdir = root / ".git"
    try:
        if gitdir.is_file():  # a worktree: ".git" holds "gitdir: <path>"
            gitdir = Path(gitdir.read_text().split(":", 1)[1].strip())
        head = (gitdir / "HEAD").read_text().strip()
    except (OSError, IndexError):
        return "unknown"
    if not head.startswith("ref:"):
        return head[:7] if _is_sha(head) else "unknown"
    ref = head.split(":", 1)[1].strip()
    loose = gitdir / ref
    try:
        if loose.is_file():
            sha = loose.read_text().strip()
            return sha[:7] if _is_sha(sha) else "unknown"
        for line in (gitdir / "packed-refs").read_text().splitlines():
            if line.startswith("#") or " " not in line:
                continue
            sha, name = line.split(" ", 1)
            if name.strip() == ref and _is_sha(sha):
                return sha[:7]
    except OSError:
        return "unknown"
    return "unknown"


def _is_sha(value: str) -> bool:
    return len(value) == 40 and all(c in "0123456789abcdef" for c in value.lower())


# ======================================================================================
# Running
# ======================================================================================


@dataclass
class TaskRun:
    """One finished (or skipped) task."""

    spec: TaskSpec
    value: float | None
    metric_key: str | None
    stderr: float | None
    n_examples: int
    n_original: int
    num_fewshot: int | None
    wall_clock_s: float
    raw: dict[str, Any] = field(default_factory=dict)
    extra: dict[str, Any] = field(default_factory=dict)
    skipped: str | None = None


_GATE_MARKERS = ("gated", "is restricted", "awaiting a review", "must be authenticated",
                 "401 client error", "403 client error", "accept the conditions")


def _skip_reason(exc: BaseException, spec: TaskSpec) -> str | None:
    """Classify an exception as an actionable skip, or ``None`` to re-raise it."""
    text = f"{type(exc).__name__}: {exc}".lower()
    if spec.gated and any(m in text for m in _GATE_MARKERS):
        return (
            f"dataset {spec.gated} is gated -- open https://huggingface.co/datasets/"
            f"{spec.gated}, accept the terms, then `huggingface-cli login` (or set HF_TOKEN)"
        )
    if any(m in text for m in _GATE_MARKERS):
        return f"dataset access denied ({type(exc).__name__}) -- accept its terms / set HF_TOKEN"
    if spec.unsafe_code and "unsafe" in text:
        return f"{spec.task} executes model-written code -- pass --confirm-run-unsafe-code"
    if isinstance(exc, ImportError):
        return f"missing dependency for {spec.task}: {exc}"
    if isinstance(exc, NameError):
        # mlx-lm 0.31.3 evaluate.py:209 references undefined `all_scores`/`all_is_greedy`
        # in the "whole prompt truncated away" branch.  See docs/DEVIATIONS.md L3.
        return (
            f"upstream mlx-lm bug hit on {spec.task} ({exc}): a continuation is longer than "
            f"the model's context.  Raise --block-size or drop the task."
        )
    return None


def run_task(
    lm,
    spec: TaskSpec,
    *,
    limit: int | None = None,
    context: int = 2048,
    seed: int = 123,
    confirm_run_unsafe_code: bool = False,
    apply_chat_template: bool = False,
    fewshot_as_multiturn: bool = False,
    bootstrap_iters: int = 100_000,
    verbose: bool = True,
) -> TaskRun:
    """Run one lm-eval task against a loaded ``MLXLM`` and return its :class:`TaskRun`.

    ``lm._max_tokens`` is set per task, which upstream leaves to the caller and which controls
    **two** unrelated things: the generation cap in ``generate_until`` (upstream reads the
    wrong gen-kwarg name, so without this every prompt generates 8,192 tokens) and the prompt
    truncation length in ``loglikelihood`` (without this a 5-shot MMLU prompt can run past a
    learned-position model's window and crash).  Both are docs/DEVIATIONS.md, "lm-eval bridge".
    """
    import lm_eval

    if spec.unsafe_code and not confirm_run_unsafe_code:
        return TaskRun(spec, None, None, None, 0, 0, spec.num_fewshot, 0.0,
                       skipped=f"{spec.task} executes model-written code -- "
                               f"pass --confirm-run-unsafe-code to run it")

    lm._max_tokens = spec.max_gen_tokens if spec.generative else max(1, int(context) - 1)
    if verbose:
        shots = "task default" if spec.num_fewshot is None else f"{spec.num_fewshot}-shot"
        cap = f", <= {lm._max_tokens} gen tokens" if spec.generative else ""
        print(f"-- {spec.task}: {spec.output_type}, {shots}, "
              f"limit {limit if limit else 'none (full set)'}{cap}", flush=True)

    t0 = time.time()
    try:
        raw = lm_eval.simple_evaluate(
            model=lm,
            tasks=[spec.task],
            num_fewshot=spec.num_fewshot,
            limit=limit,
            log_samples=False,
            apply_chat_template=apply_chat_template,
            fewshot_as_multiturn=fewshot_as_multiturn,
            bootstrap_iters=bootstrap_iters,
            confirm_run_unsafe_code=confirm_run_unsafe_code,
            random_seed=seed,
            numpy_random_seed=seed,
            torch_random_seed=seed,
            fewshot_random_seed=seed,
        )
    except Exception as exc:  # classified below, and re-raised when unrecognised
        reason = _skip_reason(exc, spec)
        if reason is None:
            raise
        wall = time.time() - t0
        if verbose:
            print(f"   SKIPPED: {reason}", flush=True)
        return TaskRun(spec, None, None, None, 0, 0, spec.num_fewshot, wall, skipped=reason)
    wall = time.time() - t0

    block = dict((raw.get("results") or {}).get(spec.task, {}))
    metric_key, value = pick_metric(block, spec.metric)
    samples = (raw.get("n-samples") or {}).get(spec.task, {})
    shots = (raw.get("n-shot") or {}).get(spec.task, spec.num_fewshot)
    stderr = _stderr_for(block, metric_key)
    return TaskRun(
        spec=spec,
        value=value * spec.scale,
        metric_key=metric_key,
        stderr=None if stderr is None else stderr * spec.scale,
        n_examples=int(samples.get("effective", 0) or 0),
        n_original=int(samples.get("original", 0) or 0),
        num_fewshot=shots,
        wall_clock_s=wall,
        raw=block,
        extra={
            "lm_eval_version": raw.get("lm_eval_version"),
            "task_version": (raw.get("versions") or {}).get(spec.task),
            "higher_is_better": (raw.get("higher_is_better") or {}).get(spec.task),
            "task_config": _task_config(raw, spec.task),
            "eot_token_id": raw.get("eot_token_id"),
            "harness_max_length": raw.get("max_length"),
        },
    )


def _task_config(raw: dict[str, Any], task: str) -> dict[str, Any]:
    cfg = (raw.get("configs") or {}).get(task, {})
    keys = ("task", "dataset_path", "dataset_name", "test_split", "output_type",
            "num_fewshot", "repeats")
    return {k: cfg.get(k) for k in keys if cfg.get(k) is not None}


def run_suite(
    model: str,
    specs: Sequence[TaskSpec],
    *,
    limit: int | None = None,
    batch_size: int = DEFAULT_BATCH_SIZE,
    block_size: int | None = None,
    seed: int = 123,
    confirm_run_unsafe_code: bool = False,
    apply_chat_template: bool = False,
    fewshot_as_multiturn: bool = False,
    use_chat_template: bool = False,
    trust_remote_code: bool = False,
    bootstrap_iters: int = 100_000,
    verbose: bool = True,
) -> tuple[list[TaskRun], dict[str, Any]]:
    """Load the model once, run every task, return ``(runs, model_description)``."""
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    mx.random.seed(seed)
    lm = build_lm(
        model,
        batch_size=batch_size,
        use_chat_template=use_chat_template or apply_chat_template,
        trust_remote_code=trust_remote_code,
    )
    context = model_context(lm, block_size)
    described = model_description(lm, model)
    described["max_context"] = context
    if verbose:
        print(f"[lmeval] {json.dumps(described)}", flush=True)

    runs: list[TaskRun] = []
    for spec in specs:
        runs.append(run_task(
            lm, spec, limit=limit, context=context, seed=seed,
            confirm_run_unsafe_code=confirm_run_unsafe_code,
            apply_chat_template=apply_chat_template,
            fewshot_as_multiturn=fewshot_as_multiturn,
            bootstrap_iters=bootstrap_iters, verbose=verbose,
        ))
    return runs, described


# ======================================================================================
# Reporting
# ======================================================================================


def _command(args: argparse.Namespace) -> str:
    """The exact command that reproduces this run (docs/ARCHITECTURE.md §1.3)."""
    parts = ["python -m r52.eval.lmeval", "--model", str(args.model)]
    if args.suite:
        parts += ["--suite", args.suite]
    if args.tasks:
        parts += ["--tasks", *args.tasks]
    # research/05 §7.2: "always report --limit".  Stated even when there is none.
    parts += ["--limit", str(args.limit)] if args.limit else ["# no --limit (full set)"]
    if args.num_shots is not None:
        parts += ["--num-shots", str(args.num_shots)]
    # A truncated generation can lose the answer, so the cap is part of the score.
    if args.max_gen_tokens:
        parts += ["--max-gen-tokens", str(args.max_gen_tokens)]
    if args.batch_size != DEFAULT_BATCH_SIZE:
        parts += ["--batch-size", str(args.batch_size)]
    if args.block_size:
        parts += ["--block-size", str(args.block_size)]
    if args.apply_chat_template:
        parts.append("--apply-chat-template")
    if args.confirm_run_unsafe_code:
        parts.append("--confirm-run-unsafe-code")
    return " ".join(parts)


def result_for(
    run: TaskRun,
    described: dict[str, Any],
    *,
    model_id: str,
    limit: int | None,
    command: str,
    commit: str | None = None,
):
    """Build the :class:`r52.eval.report.EvalResult` for one finished task."""
    from .report import make_result

    assert run.value is not None
    harness = (f"lm-evaluation-harness {run.extra.get('lm_eval_version')} via "
               f"mlx_lm.evaluate (MLXLM), task version {run.extra.get('task_version')}")
    conditions = {
        "tokenizer": described.get("tokenizer"),
        "block_size": described.get("max_context"),
        # Never absent: research/05 §7.2's "always report --limit" rule.
        "limit": limit if limit else "none (full set)",
        "n_examples": run.n_examples,
        "few_shot": run.num_fewshot,
        "harness": harness,
        "task": run.spec.task,
        "metric": run.metric_key,
        "quantization": described.get("quantization"),
        "command": command,
    }
    metrics: dict[str, Any] = {
        "lm_eval_results": run.raw,
        "metric_key": run.metric_key,
        "value_fraction": run.value / run.spec.scale if run.spec.scale else run.value,
        "stderr": run.stderr,
        "n_examples": run.n_examples,
        "n_examples_full_set": run.n_original,
        "num_fewshot": run.num_fewshot,
        "max_gen_tokens": run.spec.max_gen_tokens if run.spec.generative else None,
        "output_type": run.spec.output_type,
        "bar_benchmark": run.spec.benchmark,
        "note": run.spec.note,
        "peak_memory_gib": mx.get_peak_memory() / 2**30,
        "model": described,
        # report.git_commit's "+dirty" suffix needs `git status`; when the CLI was
        # unavailable the commit came from .git/HEAD and dirtiness is simply not known.
        "git_dirty": ("unknown (git CLI unavailable; commit read from .git/HEAD)"
                      if commit and not commit.endswith("+dirty") and "+" not in commit
                      else "reported in `commit`"),
        "mlx_lm_equivalent_command": (
            f"mlx_lm.evaluate --model {described.get('path')} --tasks {run.spec.task}"
            + (f" --num-shots {run.num_fewshot}" if run.num_fewshot is not None else "")
            + (f" --limit {limit}" if limit else "")
            + " --no-apply-chat-template"
        ),
    }
    metrics.update(run.extra)
    return make_result(
        model=model_id,
        benchmark=run.spec.benchmark_id,
        value=run.value,
        unit=run.spec.unit,
        conditions=conditions,
        wall_clock_s=run.wall_clock_s,
        metrics=metrics,
        commit=commit or git_commit(),
    )


# ======================================================================================
# CLI
# ======================================================================================


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="r52.eval.lmeval",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--model", required=True,
                   help="mlx-lm model directory (models/<name>-mlx, models/gpt2-mlx) or an "
                        "HF / mlx-community repo id")
    p.add_argument("--suite", default=None, choices=sorted(SUITES),
                   help="quick (loglikelihood only) or standard (mostly generative, slow)")
    p.add_argument("--tasks", nargs="+", default=None, help="explicit lm-eval task names")
    p.add_argument("--limit", type=int, default=0,
                   help="examples per task (0 = full set).  ALWAYS reported with the score")
    p.add_argument("--num-shots", type=int, default=None,
                   help="override every task's few-shot count")
    p.add_argument("--max-gen-tokens", type=int, default=0,
                   help="override every generative task's generation cap")
    p.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    p.add_argument("--block-size", type=int, default=None,
                   help="context in tokens; default is the model config's own limit")
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--bootstrap-iters", type=int, default=100_000,
                   help="lm-eval stderr bootstrap iterations (0 disables, stderr becomes N/A)")
    p.add_argument("--apply-chat-template", action="store_true",
                   help="render prompts through the tokenizer's chat template (post-trained "
                        "models only; base-model scores are not comparable across this flag)")
    p.add_argument("--fewshot-as-multiturn", action="store_true")
    p.add_argument("--confirm-run-unsafe-code", action="store_true",
                   help="allow tasks that execute model-written code (humaneval)")
    p.add_argument("--trust-remote-code", action="store_true")
    p.add_argument("--memory-limit-gib", type=float, default=DEFAULT_MEMORY_LIMIT_GIB)
    p.add_argument("--report", action="store_true", help="write results/<run>/lmeval_<task>.json")
    p.add_argument("--run", default=None, help="results/<run>/ directory name")
    p.add_argument("--results-dir", default=None,
                   help="write results here instead of results/ (also moves the markdown row "
                        "to <dir>/RESULTS.md, so smoke runs leave docs/RESULTS.md alone)")
    p.add_argument("--model-id", default=None, help="bar.yaml model id (e.g. gpt2-124m)")
    p.add_argument("--output-dir", default=None,
                   help="also dump lm-eval's own result blocks here, one JSON per task")
    p.add_argument("--quiet", action="store_true")
    return p


def _default_run(model: str) -> str:
    p = Path(model)
    return p.name if p.exists() else model.replace("/", "__")


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not args.suite and not args.tasks:
        build_parser().error("give --suite quick|standard and/or --tasks <name> ...")
    configure_runtime(args.memory_limit_gib)

    specs = resolve_tasks(args.suite, args.tasks,
                          num_fewshot=args.num_shots, max_gen_tokens=args.max_gen_tokens)
    unknown = [s.task for s in specs if s.task not in known_task_names()]
    if unknown:
        print(f"[lmeval] unknown lm-eval task(s): {', '.join(unknown)}", flush=True)
        return 2

    limit = args.limit or None
    verbose = not args.quiet
    if verbose:
        print(f"[lmeval] {len(specs)} task(s): {', '.join(s.task for s in specs)}", flush=True)
        if args.suite == "standard" and not limit:
            print("[lmeval] WARNING: the standard suite unlimited is hours to days on this "
                  "machine (research/05 §7.3). Use --limit.", flush=True)

    t0 = time.time()
    try:
        runs, described = run_suite(
            args.model, specs,
            limit=limit, batch_size=args.batch_size, block_size=args.block_size, seed=args.seed,
            confirm_run_unsafe_code=args.confirm_run_unsafe_code,
            apply_chat_template=args.apply_chat_template,
            fewshot_as_multiturn=args.fewshot_as_multiturn,
            trust_remote_code=args.trust_remote_code,
            bootstrap_iters=args.bootstrap_iters, verbose=verbose,
        )
    except (ValueError, FileNotFoundError) as exc:
        # `_check_model_spec` raises these with the fix in the message; a traceback would
        # bury it.
        print(f"[lmeval] {exc}", flush=True)
        return 2
    wall = time.time() - t0

    run_name = args.run or _default_run(args.model)
    command = _command(args)
    commit = git_commit()

    print()
    print(f"== lm-eval on {args.model} "
          f"(limit {limit if limit else 'none (full set)'}, commit {commit}) ==")
    for r in runs:
        if r.skipped:
            print(f"  {r.spec.task:<24} SKIPPED  {r.skipped}")
            continue
        err = f" +/- {r.stderr:.2f}" if r.stderr is not None else ""
        print(f"  {r.spec.task:<24} {r.value:7.2f} {r.spec.unit}{err} "
              f"| {r.metric_key} | {r.n_examples} of {r.n_original} examples "
              f"| {r.num_fewshot}-shot | {r.wall_clock_s:.1f} s")
    print(f"  {'TOTAL':<24} {wall:.1f} s wall-clock, "
          f"peak memory {mx.get_peak_memory() / 2**30:.2f} GiB")

    if args.output_dir:
        out = Path(args.output_dir)
        out.mkdir(parents=True, exist_ok=True)
        for r in runs:
            if r.skipped:
                continue
            (out / f"lmeval_{r.spec.task}_raw.json").write_text(json.dumps(r.raw, indent=2) + "\n")

    if args.report:
        from .report import write_result

        model_id = args.model_id or run_name
        root = Path(args.results_dir) if args.results_dir else None
        for r in runs:
            if r.skipped:
                continue
            result = result_for(r, described, model_id=model_id, limit=limit,
                                command=command, commit=commit)
            path = write_result(result, run_name, f"lmeval_{r.spec.task}", results_root=root)
            print(f"wrote {path}", flush=True)

    return 0 if any(r.skipped is None for r in runs) else 1


if __name__ == "__main__":
    raise SystemExit(main())
