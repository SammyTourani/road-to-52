# Copyright 2026 The road-to-52 authors.
# SPDX-License-Identifier: Apache-2.0
"""Configuration dataclasses for the post-training stages (YAML I/O, dotted overrides).

The optimizer/schedule/logging half of every stage is the pretraining
:class:`r52.config.TrainConfig`, so post-training reuses :func:`r52.optim.build_optimizer`
and :func:`r52.optim.wsd_multiplier` unchanged.  The stage-specific dataclasses here only
add what pretraining has no notion of: where the base checkpoint is, where the rendered
conversations are, and the RL sampling/reward settings.

Units: ``*_tokens`` are token counts, ``max_seq`` is tokens, ``lr`` values are per
optimizer step, ``memory_limit_gb`` is GiB.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from ..config import TrainConfig, _asdict

__all__ = [
    "MidtrainConfig",
    "MixtureSource",
    "RLConfig",
    "SFTConfig",
    "load_midtrain_config",
    "load_rl_config",
    "load_sft_config",
    "override",
]


def _filter(cls: type, d: dict[str, Any]) -> dict[str, Any]:
    known = {f.name for f in dataclasses.fields(cls)}
    unknown = set(d) - known
    if unknown:
        raise ValueError(f"unknown {cls.__name__} keys: {sorted(unknown)}")
    return d


def _sft_train_defaults() -> TrainConfig:
    """Pretraining optimizer settings scaled down for fine-tuning.

    Muon's 0.05 and AdamW-embed's 0.6 are *pretraining from random init* learning rates.
    Fine-tuning a converged model at those values destroys it in a handful of steps, so
    every group is divided by 10 and the schedule becomes warmup + a long cooldown to zero
    (``cooldown_frac`` 1.0), which is the standard SFT shape.
    """
    return TrainConfig(
        micro_batch=2,
        tokens_per_step=8192,
        muon_lr=0.005,
        adam_lr_embed=0.06,
        adam_lr_vembed=0.06,
        adam_lr_head=0.0008,
        adam_lr_scalar=0.004,
        warmup_steps=20,
        cooldown_frac=1.0,
        final_lr_frac=0.0,
        grad_clip=1.0,
        log_interval=10,
        val_interval=0,
        ckpt_interval=200,
        keep_last=1,
        memory_limit_gb=6.0,
        compile=False,
        out_dir="runs",
    )


# --------------------------------------------------------------------------------------
# Midtraining
# --------------------------------------------------------------------------------------


@dataclass
class MixtureSource:
    """One stream in the midtraining mixture.

    ``kind``
        ``'bin'``  -- existing llm.c ``.bin`` shards (already GPT-2 tokenized).
        ``'text'`` -- a Hugging Face dataset of raw documents; each document is written as
        ``[<|endoftext|>] + gpt2(text)``, matching the FineWeb shards' own convention.
        ``'chat'`` -- a Hugging Face dataset of conversations; each row is rendered with
        :func:`r52.chat_template.render`, so instruction data enters the *base* model in
        exactly the format SFT and inference will use.
    ``weight``
        Share of the output tokens, normalised across sources.
    """

    name: str = "source"
    kind: str = "text"
    weight: float = 1.0
    # kind == "bin"
    data_dir: str = ""
    glob: str = "*.bin"
    # kind in ("text", "chat")
    repo_id: str = ""
    subset: str = ""
    """HF dataset config name (e.g. ``finemath-4plus``); empty = the default config."""
    split: str = "train"
    text_key: str = "text"
    messages_key: str = "messages"
    streaming: bool = True
    max_seq: int = 0
    """Truncate each rendered document to this many tokens (0 = no truncation)."""

    def __post_init__(self) -> None:
        if self.kind not in ("bin", "text", "chat"):
            raise ValueError(f"unknown mixture source kind {self.kind!r}")
        if self.weight < 0:
            raise ValueError("mixture weights must be >= 0")


@dataclass
class MidtrainConfig:
    """Mixture and budget for ``scripts/prepare_midtrain_data.py``.

    ``docs/PLAN.md`` §3.1 item 4 sizes midtraining at 1-2 % of pretraining tokens; for the
    nano model (0.4 B pretraining tokens) the planner's cap is **50 M tokens**, which is
    what ``configs/posttrain/midtrain_nano.yaml`` ships.
    """

    name: str = "midtrain"
    out_dir: str = "data/midtrain"
    prefix: str = "midtrain"
    total_tokens: int = 50_000_000
    shard_tokens: int = 25_000_000
    val_tokens: int = 1_000_000
    seed: int = 1337
    sources: list[MixtureSource] = field(default_factory=list)

    def weights(self) -> list[float]:
        """Normalised source weights (sums to 1)."""
        total = sum(s.weight for s in self.sources)
        if total <= 0:
            raise ValueError("mixture weights sum to 0")
        return [s.weight / total for s in self.sources]

    def to_dict(self) -> dict[str, Any]:
        return _asdict(self)


def load_midtrain_config(path: str | Path) -> MidtrainConfig:
    """Load a midtraining mixture YAML."""
    d = yaml.safe_load(Path(path).read_text()) or {}
    sources = [MixtureSource(**_filter(MixtureSource, s)) for s in d.pop("sources", [])]
    return MidtrainConfig(**_filter(MidtrainConfig, d), sources=sources)


# --------------------------------------------------------------------------------------
# SFT
# --------------------------------------------------------------------------------------


@dataclass
class SFTConfig:
    """Supervised fine-tuning on rendered conversations with assistant-only loss."""

    name: str = "sft"
    init_from: str = ""
    """Checkpoint directory to start from (a base or midtrained ``ckpt/best``)."""
    data_dir: str = "data/sft/smol-smoltalk"
    """Directory written by ``scripts/prepare_sft_data.py`` (``train/`` and ``valid/``)."""
    max_seq: int = 1024
    """Maximum tokens per example; clamped to the model's ``block_size``."""
    micro_batch: int = 2
    grad_accum: int = 4
    epochs: float = 1.0
    max_steps: int = 0
    """If > 0, overrides ``epochs`` for both the budget and the LR schedule."""
    pack: bool = False
    """Concatenate conversations into full ``max_seq`` blocks instead of padding them.

    Padding wastes compute but keeps every example's attention independent; at nano scale
    the planner's call is padding (``False``), and the mask makes both exact.
    """
    pad_multiple: int = 64
    """Pad each batch to a multiple of this many tokens (fewer distinct graph shapes)."""
    shuffle_seed: int = 1337
    val_batches: int = 16
    """Validation micro-batches per evaluation (0 = skip validation)."""
    train: TrainConfig = field(default_factory=_sft_train_defaults)

    def tokens_per_step(self) -> int:
        """Optimizer-step batch size in tokens (padding included)."""
        return self.micro_batch * self.max_seq * self.grad_accum

    def to_dict(self) -> dict[str, Any]:
        return _asdict(self)


def load_sft_config(path: str | Path) -> SFTConfig:
    """Load an SFT YAML; the ``train:`` block overrides :func:`_sft_train_defaults`."""
    d = yaml.safe_load(Path(path).read_text()) or {}
    train_d = d.pop("train", {}) or {}
    train = _sft_train_defaults()
    for k, v in _filter(TrainConfig, train_d).items():
        setattr(train, k, v)
    train.__post_init__()
    return SFTConfig(**_filter(SFTConfig, d), train=train)


# --------------------------------------------------------------------------------------
# RL
# --------------------------------------------------------------------------------------


@dataclass
class RLConfig:
    """GRPO (DAPO-flavoured) on reasoning-gym tasks.

    DAPO (arXiv 2503.14476) = **no KL** (``beta = 0``), **clip-higher**
    (``epsilon_high > epsilon_low``), **token-level loss** and **no std normalisation** of
    the advantage.  ``docs/PLAN.md`` §3.1 item 5 asks for exactly that shape; the defaults
    below are it, and ``std_normalize: true`` recovers vanilla GRPO for comparison.
    """

    name: str = "rl"
    model: str = ""
    """Exported mlx-lm model directory (``python -m r52.export ... models/<name>-mlx``)."""
    out_dir: str = "runs"
    tasks: list[str] = field(default_factory=lambda: ["chain_sum"])
    task_size: int = 2000
    """Size of the procedural reasoning-gym dataset drawn per task."""
    task_seed: int = 42
    system_prompt: str = ""
    """Explicit system turn; empty falls back to :data:`r52.posttrain.rgym.SYSTEM_PROMPT`."""
    use_system_prompt: bool = True
    """``False`` sends the bare question with no system turn."""
    # -- sampling -----------------------------------------------------------------
    prompts_per_step: int = 8
    """Distinct prompts per optimizer step (the GRPO "batch")."""
    group_size: int = 8
    """Completions sampled per prompt (the GRPO "group")."""
    max_prompt_tokens: int = 256
    max_completion_tokens: int = 64
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = 0
    """0 disables top-k."""
    # -- loss ---------------------------------------------------------------------
    beta: float = 0.0
    """KL coefficient. DAPO uses 0 (no reference model, no KL)."""
    epsilon_low: float = 0.2
    epsilon_high: float = 0.28
    """DAPO clip-higher: the upper PPO clip bound exceeds the lower one."""
    std_normalize: bool = False
    """Divide the group advantage by its std (vanilla GRPO). DAPO does not."""
    loss_agg: str = "token"
    """``'token'`` (DAPO: sum over all tokens / total tokens) or ``'sequence'``."""
    dynamic_sampling: bool = True
    """Drop groups whose completions all score the same (zero advantage, no signal)."""
    format_reward: float = 0.0
    """Extra reward for emitting a well-formed ``<answer>...</answer>`` block."""
    # -- optimization --------------------------------------------------------------
    steps: int = 100
    inner_epochs: int = 1
    """Optimizer updates per sampled batch.

    With 1 (the default, and what on-policy GRPO means) the policy has not moved since
    sampling, so the importance ratio is exactly 1 and PPO clipping -- including DAPO's
    clip-higher -- is inactive by construction; the gradient is the plain
    ``-A * grad log pi`` estimator.  Set > 1 to reuse each batch, which is when
    ``epsilon_high`` starts to matter and ``clip_frac`` becomes a real number.
    """
    learning_rate: float = 1e-5
    adam_betas: tuple[float, float] = (0.9, 0.95)
    weight_decay: float = 0.0
    grad_clip: float = 1.0
    warmup_steps: int = 5
    micro_batch: int = 4
    """Completions per forward/backward pass in the loss phase (memory knob, not a
    hyper-parameter: the optimizer step always covers the whole group batch)."""
    seed: int = 1337
    log_interval: int = 1
    save_interval: int = 25
    memory_limit_gb: float = 6.0

    def __post_init__(self) -> None:
        self.adam_betas = tuple(self.adam_betas)  # type: ignore[assignment]
        if self.loss_agg not in ("token", "sequence"):
            raise ValueError(f"unknown loss_agg {self.loss_agg!r}")
        if self.group_size < 2:
            raise ValueError("group_size must be >= 2 for group-normalized advantages")

    def to_dict(self) -> dict[str, Any]:
        return _asdict(self)


def load_rl_config(path: str | Path) -> RLConfig:
    """Load an RL YAML."""
    d = yaml.safe_load(Path(path).read_text()) or {}
    return RLConfig(**_filter(RLConfig, d))


# --------------------------------------------------------------------------------------
# CLI overrides
# --------------------------------------------------------------------------------------


def override(cfg: Any, items: list[str]) -> Any:
    """Apply ``key=value`` (or ``train.key=value``) CLI overrides in place.

    Values are coerced to the type of the current field, exactly as
    :func:`r52.config.apply_overrides` does for pretraining configs.
    """
    for item in items or []:
        if "=" not in item:
            raise ValueError(f"override {item!r} is not of the form key=value")
        key, raw = item.split("=", 1)
        key = key.strip().lstrip("-").replace("-", "_")
        target: Any = cfg
        if "." in key:
            section, key = key.split(".", 1)
            target = getattr(cfg, section, None)
            if target is None:
                raise ValueError(f"{type(cfg).__name__} has no {section!r} section")
        if not hasattr(target, key):
            raise ValueError(f"unknown config key {key!r} for {type(target).__name__}")
        setattr(target, key, _coerce(getattr(target, key), raw))
    for obj in (getattr(cfg, "train", None), cfg):
        if obj is not None and hasattr(obj, "__post_init__"):
            obj.__post_init__()
    return cfg


_TRUE = {"true", "yes", "on", "1"}
_FALSE = {"false", "no", "off", "0"}


def _coerce(current: Any, raw: str) -> Any:
    """Parse ``raw`` into the type of ``current`` (same rules as :mod:`r52.config`)."""
    if isinstance(current, bool):
        low = raw.strip().lower()
        if low in _TRUE:
            return True
        if low in _FALSE:
            return False
        raise ValueError(f"cannot parse {raw!r} as bool")
    if isinstance(current, int) and not isinstance(current, bool):
        return int(float(raw))
    if isinstance(current, float):
        return float(raw)
    if isinstance(current, (tuple, list)):
        parts = [p.strip() for p in raw.replace("[", "").replace("]", "").split(",") if p.strip()]
        if current and isinstance(current[0], str):
            return type(current)(parts)
        if not current:  # empty list -> assume strings (e.g. RLConfig.tasks)
            return list(parts)
        return type(current)(float(p) for p in parts)
    if current is None:
        return raw
    return raw
