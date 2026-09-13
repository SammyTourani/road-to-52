# Copyright 2026 The road-to-52 authors.
# SPDX-License-Identifier: Apache-2.0
"""Config dataclasses, YAML round-trip, CLI overrides, and the documented preset sizes."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from r52.bench import SIZES
from r52.config import Config, ModelConfig, apply_overrides, from_dict, load_config

CONFIG_DIR = Path(__file__).resolve().parent.parent / "configs"

# From the ModelConfig docstring; keep the two in sync.
PRESET_PARAMS = {
    "30M": (102_432_784, 25_165_840, 50_921_488),
    "60M": (155_566_098, 58_982_418, 91_176_978),
    "124M": (200_835_090, 84_934_674, 123_568_146),
    "350M": (456_523_800, 301_989_912, 353_501_208),
}


@pytest.mark.parametrize("path", sorted(CONFIG_DIR.glob("*.yaml")))
def test_shipped_configs_load(path: Path) -> None:
    cfg = load_config(path)
    assert cfg.grad_accum() >= 1
    assert cfg.total_steps() >= 1
    assert cfg.train.tokens_per_step % (cfg.train.micro_batch * cfg.model.block_size) == 0


def test_yaml_round_trip(tmp_path: Path) -> None:
    cfg = load_config(CONFIG_DIR / "gpt2_124m_mac.yaml")
    out = tmp_path / "c.yaml"
    cfg.save(out)
    again = from_dict(yaml.safe_load(out.read_text()))
    assert again.to_dict() == cfg.to_dict()


def test_overrides_coerce_types() -> None:
    cfg = Config()
    apply_overrides(cfg, ["model.n_layer=3", "micro_batch=2", "train.compile=false",
                          "muon_lr=0.01", "data.source=synthetic"])
    assert cfg.model.n_layer == 3
    assert cfg.train.micro_batch == 2
    assert cfg.train.compile is False
    assert cfg.train.muon_lr == pytest.approx(0.01)
    assert cfg.data.source == "synthetic"


def test_unknown_override_raises() -> None:
    with pytest.raises(ValueError):
        apply_overrides(Config(), ["nope=1"])


def test_derived_fields() -> None:
    m = ModelConfig(n_layer=12, n_embd=768, n_head=6)
    assert m.head_dim == 128 and m.n_kv_head == 6
    assert m.mlp_hidden == 3072
    assert m.n_encoder_layers == 6 and m.n_decoder_layers == 6
    assert not m.nanochat_compatible  # value embeds + skips are on by default
    assert ModelConfig(use_value_embeds=False, use_unet_skips=False).nanochat_compatible


@pytest.mark.parametrize("label", sorted(PRESET_PARAMS))
def test_preset_param_counts(label: str) -> None:
    """The parameter counts documented in ModelConfig's docstring are exact."""
    from r52.model import GPT

    m = GPT(ModelConfig(vocab_size=50304, block_size=1024, n_value_embeds=3,
                        value_embed_share=True, **SIZES[label]))
    assert (m.num_params(), m.num_params(True), m.num_params_flops()) == PRESET_PARAMS[label]
