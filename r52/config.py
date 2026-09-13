# Copyright 2026 The road-to-52 authors.
# SPDX-License-Identifier: Apache-2.0
"""Configuration dataclasses for road-to-52 with YAML I/O and dotted CLI overrides.

All units are stated explicitly in the field docstrings/comments:
tokens, steps, seconds, hours, GiB.
"""

from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any

import yaml

__all__ = [
    "Config",
    "DataConfig",
    "ModelConfig",
    "TrainConfig",
    "apply_overrides",
    "load_config",
]

# --------------------------------------------------------------------------------------
# Model
# --------------------------------------------------------------------------------------


@dataclass
class ModelConfig:
    """Architecture of the decoder-only transformer.

    Parameter counts measured on 2026-09-13 (``vocab_size=50304``, ``mlp='relu2'``,
    ``mlp_ratio=4``, untied embeddings, value embeddings ON with ``value_embed_share=True``,
    U-net skips ON) -- reproduce with ``tests/test_config.py::test_preset_param_counts``:

    ==============  ==  =====  ==  ==============  ==============  ==============
    preset          L   d      h   total           non-embedding   N for 6N
    ==============  ==  =====  ==  ==============  ==============  ==============
    ``tiny_test``    2     64   2       9,756,677          98,309       3,317,765
    ``30M``          8    512   8     102,432,784      25,165,840      50,921,488
    ``60M``         12    640  10     155,566,098      58,982,418      91,176,978
    ``124M``        12    768   6     200,835,090      84,934,674     123,568,146
    ``350M``        24   1024   8     456,523,800     301,989,912     353,501,208
    ==============  ==  =====  ==  ==============  ==============  ==============

    Three counts, three meanings:

    * **total** -- every trainable tensor.  With a 50,304-token vocabulary, untied
      embeddings and a value-embedding table, three ``vocab x n_embd`` matrices dominate
      the total at small widths (``124M`` carries 115.9M of embedding parameters).
      ``value_embed_share=False`` (modded-nanogpt's 3 separate tables) pushes ``124M``
      to 278,102,034.
    * **non-embedding** -- the "scaling parameters" convention of nanoGPT / research/05
      §5; this is the number the preset labels refer to.
    * **N for 6N** -- non-embedding + ``lm_head``; the parameters that actually run a
      matmul, and therefore the ``N`` in ``flops_per_token``.  For the ``124M`` preset
      this is 123.6M, i.e. exactly the classic GPT-2-small count, so our MFU numbers are
      directly comparable with published nanoGPT/modded-nanogpt figures.
    """

    n_layer: int = 12
    n_embd: int = 768
    n_head: int = 6
    n_kv_head: int | None = None
    """Number of key/value heads (GQA). ``None`` -> ``n_head`` (plain MHA)."""
    head_dim: int | None = None
    """Per-head dimension. ``None`` -> ``n_embd // n_head``."""
    vocab_size: int = 50304
    """Padded vocabulary (GPT-2's 50257 rounded up to a multiple of 128)."""
    block_size: int = 1024
    """Context length in tokens."""
    rope_base: float = 10000.0
    softcap: float = 15.0
    """Logit soft cap: ``softcap * tanh(logits / softcap)``. 0 disables."""
    qk_norm: bool = True
    use_value_embeds: bool = True
    n_value_embeds: int = 3
    """Layers at each end of the stack that receive a value embedding (mirrored)."""
    value_embed_share: bool = False
    """Share ONE value-embedding table across all value-embedding layers.

    modded-nanogpt uses ``n_value_embeds`` distinct tables; each costs
    ``vocab_size * n_embd`` fp32 parameters plus 2x that in AdamW state, which is
    1.4 GiB at 124M -- more than a third of our 4 GiB budget.  Sharing keeps the
    technique at a quarter of the cost.
    """
    use_unet_skips: bool = True
    mlp: str = "relu2"
    """``'relu2'`` (nanochat/speedrun) or ``'swiglu'``."""
    mlp_ratio: float = 4.0
    tie_embeddings: bool = False
    norm_eps: float = 1e-5
    precision: str = "mixed"
    """``'mixed'`` (fp32 master params, bf16 matmuls), ``'bf16'``, or ``'fp32'``."""
    grad_checkpoint: bool = False
    """Per-block gradient checkpointing via ``mlx.nn.utils.checkpoint``."""

    def __post_init__(self) -> None:
        if self.n_kv_head is None:
            self.n_kv_head = self.n_head
        if self.head_dim is None:
            if self.n_embd % self.n_head != 0:
                raise ValueError(f"n_embd={self.n_embd} not divisible by n_head={self.n_head}")
            self.head_dim = self.n_embd // self.n_head
        if self.n_head % self.n_kv_head != 0:
            raise ValueError(f"n_head={self.n_head} not divisible by n_kv_head={self.n_kv_head}")
        if self.head_dim % 2 != 0:
            raise ValueError(f"head_dim={self.head_dim} must be even for RoPE")
        if self.mlp not in ("relu2", "swiglu"):
            raise ValueError(f"unknown mlp {self.mlp!r}")
        if self.precision not in ("mixed", "bf16", "fp32"):
            raise ValueError(f"unknown precision {self.precision!r}")
        if self.use_value_embeds:
            # Value embeddings are mirrored: first k layers and last k layers.
            self.n_value_embeds = max(0, min(self.n_value_embeds, self.n_layer // 2))
            if self.n_value_embeds == 0:
                self.use_value_embeds = False

    # -- derived ------------------------------------------------------------------
    @property
    def mlp_hidden(self) -> int:
        """Hidden width of the MLP, in units."""
        if self.mlp == "relu2":
            return int(self.mlp_ratio * self.n_embd)
        # SwiGLU: keep the parameter count comparable to a ratio-x ReLU^2 MLP.
        h = int(2 * self.mlp_ratio * self.n_embd / 3)
        return 128 * math.ceil(h / 128)

    @property
    def n_encoder_layers(self) -> int:
        return self.n_layer // 2

    @property
    def n_decoder_layers(self) -> int:
        return self.n_layer - self.n_layer // 2

    @property
    def nanochat_compatible(self) -> bool:
        """True if the weight layout is byte-for-byte loadable by ``mlx_lm.models.nanochat``."""
        return (
            not self.use_value_embeds
            and not self.use_unet_skips
            and self.mlp == "relu2"
            and self.qk_norm
            and not self.tie_embeddings
            and self.head_dim == self.n_embd // self.n_head
            and abs(self.softcap - 15.0) < 1e-9
        )


# --------------------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------------------


@dataclass
class DataConfig:
    """Where the tokens come from."""

    source: str = "fineweb"
    """``'fineweb'`` (llm.c ``.bin`` shards) or ``'synthetic'`` (deterministic noise, tests)."""
    data_dir: str = "data/fineweb10B-gpt2"
    repo_id: str = "kjj0/fineweb10B-gpt2"
    train_glob: str = "fineweb_train_*.bin"
    val_file: str = "fineweb_val_000000.bin"
    val_tokens: int = 10_485_760
    """Tokens of the val shard used for validation (modded-nanogpt's fixed 10,485,760)."""
    synthetic_tokens: int = 1_000_000
    """Token count of the synthetic corpus (only for ``source='synthetic'``)."""
    seed: int = 1337

    def __post_init__(self) -> None:
        if self.source not in ("fineweb", "synthetic"):
            raise ValueError(f"unknown data source {self.source!r}")


# --------------------------------------------------------------------------------------
# Training
# --------------------------------------------------------------------------------------


@dataclass
class TrainConfig:
    """Optimization, schedule, budget, logging and checkpointing.

    Optimizer defaults are modded-nanogpt's current-record values -- see the YAML configs
    for the exact citation and any batch-size rescaling.
    """

    # -- batch -------------------------------------------------------------------
    micro_batch: int = 4
    """Sequences per forward/backward pass."""
    tokens_per_step: int = 131_072
    """Optimizer-step batch size in tokens. grad_accum = tokens_per_step / (mb*T)."""

    # -- Muon (2-D hidden matrices) ----------------------------------------------
    muon_lr: float = 0.05
    muon_momentum: float = 0.95
    muon_weight_decay: float = 0.0
    muon_nesterov: bool = True
    muon_ns_steps: int = 5

    # -- AdamW (embeddings / lm_head / value embeddings / scalars) ----------------
    adam_lr_embed: float = 0.6
    adam_lr_vembed: float = 0.6
    adam_lr_head: float = 0.008
    adam_lr_scalar: float = 0.04
    adam_betas_embed: tuple[float, float] = (0.8, 0.95)
    adam_betas_vembed: tuple[float, float] = (0.75, 0.95)
    adam_betas_head: tuple[float, float] = (0.5, 0.95)
    adam_betas_scalar: tuple[float, float] = (0.9, 0.99)
    adam_eps: float = 1e-10
    adam_wd_embed: float = 0.0
    adam_wd_vembed: float = 0.025
    adam_wd_head: float = 0.75
    adam_wd_scalar: float = 0.0

    # -- schedule ----------------------------------------------------------------
    warmup_steps: int = 0
    cooldown_frac: float = 0.4
    """Fraction of the *scheduled* run spent in the linear LR cooldown."""
    final_lr_frac: float = 0.0
    """LR multiplier at the end of cooldown (modded-nanogpt's record uses 0.15)."""
    grad_clip: float = 1.0
    """Global grad-norm clip. <= 0 disables."""

    # -- budget ------------------------------------------------------------------
    max_tokens: int = 750_000_000
    max_steps: int = 0
    """If > 0, overrides ``max_tokens``."""
    max_hours: float = 0.0
    """Wall-clock stop (hours). 0 = no limit."""
    target_val_loss: float = 0.0
    """Stop after the first validation at or below this loss. 0 = no target."""

    # -- logging / checkpointing --------------------------------------------------
    log_interval: int = 10
    val_interval: int = 250
    val_max_batches: int = 0
    """Cap on validation micro-batches (0 = the full ``data.val_tokens``)."""
    val_micro_batch: int = 0
    """Micro-batch for validation (0 = ``micro_batch``)."""
    ckpt_interval: int = 500
    keep_last: int = 2
    out_dir: str = "runs"

    # -- machine ------------------------------------------------------------------
    memory_limit_gb: float = 6.0
    cache_limit_gb: float = 1.0
    seed: int = 1337
    compile: bool = True

    # -- MFU denominators (TFLOPS) -------------------------------------------------
    peak_tflops_theoretical: float = 4.26
    """Apple M4 10-core GPU theoretical fp32 peak (arXiv 2502.05317 Table 1)."""
    peak_tflops_measured: float = 3.6
    """Measured bf16 4096^3 matmul peak on this machine (2026-09-13)."""

    def __post_init__(self) -> None:
        self.adam_betas_embed = tuple(self.adam_betas_embed)  # type: ignore[assignment]
        self.adam_betas_vembed = tuple(self.adam_betas_vembed)  # type: ignore[assignment]
        self.adam_betas_head = tuple(self.adam_betas_head)  # type: ignore[assignment]
        self.adam_betas_scalar = tuple(self.adam_betas_scalar)  # type: ignore[assignment]
        if not 0.0 <= self.cooldown_frac <= 1.0:
            raise ValueError("cooldown_frac must be in [0, 1]")


# --------------------------------------------------------------------------------------
# Top level
# --------------------------------------------------------------------------------------


@dataclass
class Config:
    """The whole run configuration."""

    name: str = "run"
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    train: TrainConfig = field(default_factory=TrainConfig)

    def grad_accum(self) -> int:
        """Gradient-accumulation micro-steps per optimizer step (>= 1)."""
        per_micro = self.train.micro_batch * self.model.block_size
        if self.train.tokens_per_step % per_micro != 0:
            raise ValueError(
                f"tokens_per_step={self.train.tokens_per_step} is not divisible by "
                f"micro_batch*block_size={per_micro}"
            )
        return max(1, self.train.tokens_per_step // per_micro)

    def total_steps(self) -> int:
        """Scheduled optimizer steps implied by the token/step budget."""
        if self.train.max_steps > 0:
            return self.train.max_steps
        return max(1, self.train.max_tokens // self.train.tokens_per_step)

    def to_dict(self) -> dict[str, Any]:
        return _asdict(self)

    def save(self, path: str | Path) -> None:
        Path(path).write_text(yaml.safe_dump(self.to_dict(), sort_keys=False))


def _asdict(obj: Any) -> Any:
    if is_dataclass(obj):
        return {f.name: _asdict(getattr(obj, f.name)) for f in fields(obj)}
    if isinstance(obj, (list, tuple)):
        return [_asdict(v) for v in obj]
    if isinstance(obj, dict):
        return {k: _asdict(v) for k, v in obj.items()}
    if isinstance(obj, Path):
        return str(obj)
    return obj


def from_dict(d: dict[str, Any]) -> Config:
    """Build a :class:`Config` from a (possibly partial) nested dict."""
    d = dict(d or {})
    model = ModelConfig(**_filter(ModelConfig, d.pop("model", {}) or {}))
    data = DataConfig(**_filter(DataConfig, d.pop("data", {}) or {}))
    train = TrainConfig(**_filter(TrainConfig, d.pop("train", {}) or {}))
    name = d.pop("name", "run")
    if d:
        raise ValueError(f"unknown top-level config keys: {sorted(d)}")
    return Config(name=name, model=model, data=data, train=train)


def _filter(cls: type, d: dict[str, Any]) -> dict[str, Any]:
    known = {f.name for f in dataclasses.fields(cls)}
    unknown = set(d) - known
    if unknown:
        raise ValueError(f"unknown {cls.__name__} keys: {sorted(unknown)}")
    return d


def load_config(path: str | Path) -> Config:
    """Load a YAML config file."""
    with open(path) as fh:
        return from_dict(yaml.safe_load(fh) or {})


_TRUE = {"true", "yes", "on", "1"}
_FALSE = {"false", "no", "off", "0"}


def _coerce(current: Any, raw: str) -> Any:
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
        return type(current)(float(p) for p in raw.replace("[", "").replace("]", "").split(","))
    if current is None:
        try:
            return int(raw)
        except ValueError:
            return raw
    return raw


def apply_overrides(cfg: Config, overrides: list[str]) -> Config:
    """Apply ``section.key=value`` CLI overrides in place and return ``cfg``.

    ``key=value`` without a section is resolved against model/data/train in that order.
    """
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"override {item!r} is not of the form key=value")
        key, raw = item.split("=", 1)
        key = key.strip().lstrip("-").replace("-", "_")
        if "." in key:
            section, attr = key.split(".", 1)
            target = getattr(cfg, section, None)
            if target is None or not hasattr(target, attr):
                raise ValueError(f"unknown config key {key!r}")
        else:
            attr = key
            target = None
            for sec in (cfg.model, cfg.data, cfg.train, cfg):
                if hasattr(sec, attr):
                    target = sec
                    break
            if target is None:
                raise ValueError(f"unknown config key {key!r}")
        setattr(target, attr, _coerce(getattr(target, attr), raw))
    # Re-run validation / derived fields.
    cfg.model.__post_init__()
    cfg.data.__post_init__()
    cfg.train.__post_init__()
    return cfg
