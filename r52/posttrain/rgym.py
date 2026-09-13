# Copyright 2026 The road-to-52 authors.
# SPDX-License-Identifier: Apache-2.0
"""reasoning-gym adapter: procedural ``{prompt, answer}`` items and a verifier reward.

`reasoning-gym <https://github.com/open-thought/reasoning-gym>`_ (Apache-2.0, the RLVR
environment chosen in ``docs/PLAN.md`` §5) generates tasks *procedurally* and ships an
exact verifier per task, so there is no sandbox, no network and no eval-set contamination:
every prompt is fresh, and the reward is ``dataset.score_answer(answer, entry) -> float``
in ``[0, 1]``.

This module turns that into the two things the rest of post-training needs:

* :class:`RGymTask` -- an indexable pool of :class:`Item` (``prompt`` text, ``answer``
  string, and the raw ``entry`` the verifier needs), plus :meth:`RGymTask.score`.
* :func:`extract_answer` -- pull a candidate answer out of free-form model text.

Answer extraction matters more than it looks at nano scale.  The model is asked to put its
final answer in ``<answer>...</answer>``; a 30 M-parameter model mostly will not.  So the
extractor falls back, in order, to ``<answer>`` tags, an ``#### x`` GSM8K-style marker, the
last non-empty line, and finally the whole string -- and the reward function additionally
tries the *last number* in the text, taking the best score.  That keeps the reward from
being identically zero (which would make every GRPO advantage zero and the run a no-op)
while never awarding credit for a wrong answer: :meth:`RGymTask.score` only ever returns
what reasoning-gym's own verifier returns.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

__all__ = ["SYSTEM_PROMPT", "Item", "RGymTask", "extract_answer", "format_score", "make_pool"]

SYSTEM_PROMPT = (
    "You are a careful problem solver. Work through the problem, then give the final "
    "answer between <answer> and </answer> tags."
)
"""Default system turn for RL / pass@k prompts. Written by us, not by any model."""

_ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL | re.IGNORECASE)
_OPEN_ANSWER_RE = re.compile(r"<answer>(.*)", re.DOTALL | re.IGNORECASE)
_HASH_RE = re.compile(r"####\s*(.+)")
_NUMBER_RE = re.compile(r"-?\d+(?:[.,]\d+)*")


def extract_answer(text: str) -> str:
    """Best-effort final answer from free-form completion text.

    Order: closed ``<answer>`` tag, unclosed ``<answer>`` tag (the model ran out of
    tokens), ``#### x``, last non-empty line, whole string.  Always returns a stripped
    string, possibly empty.
    """
    if not text:
        return ""
    m = _ANSWER_RE.search(text)
    if m:
        return m.group(1).strip()
    m = _OPEN_ANSWER_RE.search(text)
    if m:
        return m.group(1).strip().splitlines()[0].strip() if m.group(1).strip() else ""
    m = _HASH_RE.search(text)
    if m:
        return m.group(1).strip()
    lines = [ln.strip() for ln in text.strip().splitlines() if ln.strip()]
    return lines[-1] if lines else text.strip()


def format_score(text: str) -> float:
    """1.0 if the completion contains a closed ``<answer>...</answer>`` block, else 0.0."""
    return 1.0 if _ANSWER_RE.search(text or "") else 0.0


def _last_number(text: str) -> str | None:
    """The last number in ``text`` with thousands separators removed, or ``None``."""
    nums = _NUMBER_RE.findall(text or "")
    return nums[-1].replace(",", "") if nums else None


@dataclass
class Item:
    """One procedurally generated task instance."""

    task: str
    question: str
    answer: str
    entry: dict[str, Any] = field(repr=False, default_factory=dict)

    def messages(self, system_prompt: str = SYSTEM_PROMPT) -> list[dict[str, str]]:
        """Chat messages for this item (system turn omitted when ``system_prompt`` is empty)."""
        msgs = []
        if system_prompt:
            msgs.append({"role": "system", "content": system_prompt})
        msgs.append({"role": "user", "content": self.question})
        return msgs


class RGymTask:
    """A pool of items from one reasoning-gym task, with its verifier.

    Parameters
    ----------
    name
        A reasoning-gym dataset name, e.g. ``chain_sum``, ``gsm_symbolic``, ``leg_counting``.
        ``RGymTask.available()`` lists all 106 registered names.
    size
        Instances to generate (reasoning-gym datasets are procedural and seeded).
    seed
        Generator seed; the same ``(name, size, seed)`` always yields the same items.
    """

    def __init__(self, name: str, size: int = 1000, seed: int = 42, **kwargs: Any) -> None:
        import reasoning_gym

        self.name = name
        self.size = int(size)
        self.seed = int(seed)
        self.dataset = reasoning_gym.create_dataset(name, size=self.size, seed=self.seed, **kwargs)

    @staticmethod
    def available() -> list[str]:
        """Every registered reasoning-gym dataset name."""
        from reasoning_gym.factory import DATASETS

        return sorted(DATASETS)

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, i: int) -> Item:
        entry = self.dataset[int(i) % self.size]
        return Item(self.name, entry["question"], str(entry["answer"]), entry)

    def items(self, n: int, start: int = 0) -> list[Item]:
        """``n`` items starting at index ``start`` (wraps around the pool)."""
        return [self[start + i] for i in range(n)]

    def score(self, completion: str, item: Item) -> float:
        """Verifier reward in ``[0, 1]`` for a raw completion string.

        Scores the extracted answer; if that scores 0, retries with the last number in the
        text and keeps the better result.  The score itself always comes from
        reasoning-gym's ``score_answer``, so a wrong answer can never be rewarded.
        """
        best = 0.0
        for candidate in self._candidates(completion):
            try:
                s = float(self.dataset.score_answer(answer=candidate, entry=item.entry))
            except Exception:
                s = 0.0
            best = max(best, s)
            if best >= 1.0:
                break
        return best

    @staticmethod
    def _candidates(completion: str) -> list[str]:
        """Answer strings to try, best guess first, de-duplicated."""
        out: list[str] = []
        for c in (extract_answer(completion), _last_number(completion), (completion or "").strip()):
            if c and c not in out:
                out.append(c)
        return out


def make_pool(
    tasks: list[str], n: int, seed: int = 42, task_size: int = 2000
) -> tuple[list[Item], dict[str, RGymTask]]:
    """Round-robin ``n`` items across ``tasks``; returns ``(items, {name: task})``."""
    built = {name: RGymTask(name, size=task_size, seed=seed + i) for i, name in enumerate(tasks)}
    items: list[Item] = []
    for i in range(n):
        name = tasks[i % len(tasks)]
        items.append(built[name][i // len(tasks)])
    return items, built
