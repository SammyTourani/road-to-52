# Copyright 2026 The road-to-52 authors.
# SPDX-License-Identifier: Apache-2.0
"""Post-training: masked SFT data/loop, midtrain shards, reasoning-gym, pass@k and GRPO.

Every test here runs a 2-layer / 64-dim model on synthetic data under the suite's 2 GiB
memory limit (``tests/conftest.py``), because the machine is shared with a multi-day
pretraining run.  Nothing in this file touches the network.
"""

from __future__ import annotations

import json
from itertools import pairwise
from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest
from conftest import tiny_model_config

from r52.chat_template import (
    ASSISTANT_END,
    ASSISTANT_START,
    BOS,
    IGNORE_INDEX,
    PAD,
    USER_END,
    USER_START,
)
from r52.checkpoint import save_checkpoint
from r52.config import Config, DataConfig, TrainConfig
from r52.model import GPT
from r52.posttrain.config import (
    MidtrainConfig,
    RLConfig,
    SFTConfig,
    load_midtrain_config,
    load_rl_config,
    load_sft_config,
    override,
)
from r52.posttrain.data import (
    ConversationDataset,
    PackedBatcher,
    SFTBatcher,
    iter_batches,
    pack_blocks,
    packed_xy,
    write_split,
)
from r52.posttrain.passk import pass_at_k
from r52.posttrain.rgym import extract_answer, format_score
from r52.posttrain.rl import build_rollouts, token_logprobs
from r52.posttrain.sft import SFTTrainer, evaluate

CONFIGS = Path(__file__).resolve().parent.parent / "configs" / "posttrain"


# --------------------------------------------------------------------------------------
# Synthetic conversations
# --------------------------------------------------------------------------------------


def fake_conversation(rng: np.random.Generator, n_user: int = 6, n_assistant: int = 5):
    """``(ids, mask)`` in the chat layout, with ids drawn from a small GPT-2-range vocab.

    Deliberately does not call the real BPE: these tests are about the *mask plumbing*,
    and `tests/test_posttrain_chat.py` already checks the real renderer.
    """
    user = rng.integers(1, 200, size=n_user, dtype=np.int64).tolist()
    asst = rng.integers(1, 200, size=n_assistant, dtype=np.int64).tolist()
    ids = [BOS, USER_START, *user, USER_END, ASSISTANT_START, *asst, ASSISTANT_END]
    mask = [0, 0, *([0] * n_user), 0, 0, *([1] * n_assistant), 1]
    return ids, mask


@pytest.fixture(scope="module")
def split(tmp_path_factory) -> Path:
    """A 64-conversation split on disk, lengths 8..20 assistant tokens."""
    rng = np.random.default_rng(0)
    root = tmp_path_factory.mktemp("sft")
    convs = [fake_conversation(rng, int(rng.integers(3, 9)), int(rng.integers(3, 9)))
             for _ in range(64)]
    write_split(root / "train", convs, {"dataset": "synthetic"})
    write_split(root / "valid", convs[:8], {"dataset": "synthetic"})
    return root


@pytest.fixture(scope="module")
def base_ckpt(tmp_path_factory) -> Path:
    """A tiny checkpoint for SFT to initialise from (full vocab, so chat ids are in range)."""
    mcfg = tiny_model_config(vocab_size=50304, block_size=64)
    model = GPT(mcfg)
    mx.eval(model.parameters())
    cfg = Config(name="tiny_base", model=mcfg, data=DataConfig(source="synthetic"),
                 train=TrainConfig())
    out = tmp_path_factory.mktemp("ckpt") / "best"
    save_checkpoint(out, model, step=0, tokens=0, config=cfg)
    return out


# --------------------------------------------------------------------------------------
# Dataset / batching
# --------------------------------------------------------------------------------------


def test_write_and_read_split_round_trip(split: Path) -> None:
    ds = ConversationDataset(split / "train")
    assert len(ds) == 64
    assert ds.n_tokens == int(ds.lengths().sum())
    assert 0 < ds.n_supervised < ds.n_tokens
    ids, mask = ds[3]
    assert ids[0] == BOS and ids[-1] == ASSISTANT_END
    assert mask[-1] == 1 and mask[0] == 0
    assert json.loads((split / "train" / "meta.json").read_text())["dataset"] == "synthetic"


def test_missing_split_has_an_actionable_error(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="prepare_sft_data"):
        ConversationDataset(tmp_path / "nope")


def test_batcher_masks_only_assistant_spans(split: Path) -> None:
    """The spec's headline assertion: every supervised target sits inside an assistant span.

    For each row, walk ``x`` (the inputs) and check that a target is present exactly when
    the *input* at that position is ``<|assistant_start|>`` or an assistant body token --
    i.e. the position whose prediction is an assistant token -- and never for a user,
    system or padding position.
    """
    ds = ConversationDataset(split / "train")
    b = SFTBatcher(ds, micro_batch=4, max_seq=64, pad_multiple=8, seed=0)
    seen_supervised = 0
    for _ in range(8):
        x, y = b.next_batch()
        xs, ys = np.asarray(x), np.asarray(y)
        assert xs.shape == ys.shape
        for row_x, row_y in zip(xs, ys, strict=True):
            inside = False
            for t, target in zip(row_x, row_y, strict=True):
                if t == ASSISTANT_START:
                    inside = True
                    assert target >= 0, "the first assistant token must be supervised"
                    continue
                if t in (ASSISTANT_END, PAD):
                    inside = False
                if inside:
                    assert target >= 0
                    seen_supervised += 1
                else:
                    assert target == IGNORE_INDEX, (
                        f"input token {int(t)} is outside an assistant span but is supervised"
                    )
    assert seen_supervised > 0


def test_batcher_pads_to_a_multiple_and_never_exceeds_max_seq(split: Path) -> None:
    ds = ConversationDataset(split / "train")
    b = SFTBatcher(ds, micro_batch=4, max_seq=16, pad_multiple=8, seed=1)
    for _ in range(4):
        x, _ = b.next_batch()
        assert x.shape[0] == 4
        assert x.shape[1] <= 16
        assert x.shape[1] % 8 == 0


def test_batcher_wraps_epochs(split: Path) -> None:
    ds = ConversationDataset(split / "train")
    b = SFTBatcher(ds, micro_batch=8, max_seq=64, seed=2)
    assert b.batches_per_epoch() == 8
    for _ in range(17):
        b.next_batch()
    assert b.epoch >= 3
    assert b.state()["epoch"] == b.epoch


def test_iter_batches_is_deterministic(split: Path) -> None:
    ds = ConversationDataset(split / "valid")
    a = [np.asarray(x) for x, _ in iter_batches(ds, 2, 64)]
    b = [np.asarray(x) for x, _ in iter_batches(ds, 2, 64)]
    assert len(a) == 4
    assert all((p == q).all() for p, q in zip(a, b, strict=True))


def test_packing_never_invents_supervision(split: Path) -> None:
    """Blocks are ``block + 1`` wide (one look-ahead target) but stride ``block``.

    So the *targets* of consecutive blocks tile the stream without overlap, and the total
    supervised-target count can only be <= the dataset's, never more.
    """
    ds = ConversationDataset(split / "train")
    ids, mask = pack_blocks(ds, block=32)
    assert ids.shape == mask.shape
    assert ids.shape[1] == 33
    _, y = packed_xy(ids, mask)
    assert int((y >= 0).sum()) <= ds.n_supervised
    assert int((y >= 0).sum()) > 0

    pb = PackedBatcher(ds, micro_batch=2, block=32, seed=0)
    x, y = pb.next_batch()
    assert x.shape == (2, 32) == y.shape
    assert int((np.asarray(y) >= 0).sum()) > 0


# --------------------------------------------------------------------------------------
# SFT
# --------------------------------------------------------------------------------------


def test_sft_loss_only_counts_assistant_tokens(split: Path, base_ckpt: Path) -> None:
    """A model's SFT loss must not move when a *user* token changes, only assistant ones."""
    model = GPT(tiny_model_config(vocab_size=50304, block_size=64))
    mx.eval(model.parameters())
    ds = ConversationDataset(split / "train")
    x, y = SFTBatcher(ds, 4, 64, 8, seed=0).next_batch()
    base = float(model.loss(x, y, fp32_logits=True))

    ys = np.asarray(y)
    xs = np.asarray(x)
    ignored = np.argwhere(ys == IGNORE_INDEX)
    assert len(ignored) > 0
    r, c = ignored[0]
    xs2 = xs.copy()
    xs2[r, c] = (xs2[r, c] + 7) % 200 + 1  # perturb an input whose target is ignored
    moved = float(model.loss(mx.array(xs2), y, fp32_logits=True))
    # It CAN move (that token is context for later positions) but the *masked* positions
    # must contribute nothing: zero out every ignored target and the loss is unchanged.
    only_masked = float(model.loss(x, mx.array(np.where(ys >= 0, ys, IGNORE_INDEX)), fp32_logits=True))
    assert only_masked == pytest.approx(base, abs=1e-6)
    assert np.isfinite(moved)


def test_sft_trains_and_the_assistant_loss_falls(split: Path, base_ckpt: Path, tmp_path: Path) -> None:
    """20 steps of real SFT on the tiny model; the loss over assistant tokens must drop."""
    cfg = SFTConfig(
        name="t", init_from=str(base_ckpt), data_dir=str(split),
        max_seq=64, micro_batch=4, grad_accum=1, max_steps=20, val_batches=2,
        pad_multiple=8, shuffle_seed=0,
    )
    cfg.train.out_dir = str(tmp_path)
    cfg.train.log_interval = 5
    cfg.train.val_interval = 0
    cfg.train.ckpt_interval = 0
    cfg.train.memory_limit_gb = 2.0
    cfg.train.compile = False
    trainer = SFTTrainer(cfg, "t")
    start = evaluate(trainer.model, trainer.valid_ds, 4, 64, 2, 8)["val_loss"]
    summary = trainer.run()
    end = evaluate(trainer.model, trainer.valid_ds, 4, 64, 2, 8)["val_loss"]

    assert summary["step"] == 20
    assert summary["sup_tokens"] > 0
    assert end < start, f"assistant loss did not fall: {start:.4f} -> {end:.4f}"
    assert (Path(cfg.train.out_dir) / "t" / "ckpt" / "best").exists()

    log = (Path(cfg.train.out_dir) / "t" / "log.jsonl").read_text().splitlines()
    records = [json.loads(ln) for ln in log]
    trains = [r for r in records if r.get("event") == "train"]
    assert trains and trains[0]["loss"] > trains[-1]["loss"]
    assert records[0]["event"] == "start" and records[0]["stage"] == "sft"


def test_sft_clamps_max_seq_to_block_size(split: Path, base_ckpt: Path, tmp_path: Path) -> None:
    cfg = SFTConfig(name="t2", init_from=str(base_ckpt), data_dir=str(split),
                    max_seq=4096, micro_batch=2, grad_accum=1, max_steps=1, val_batches=0)
    cfg.train.out_dir = str(tmp_path)
    cfg.train.ckpt_interval = 0
    trainer = SFTTrainer(cfg, "t2")
    assert trainer.cfg.max_seq == 64


def test_sft_requires_init_from(split: Path) -> None:
    with pytest.raises(ValueError, match="init_from"):
        SFTTrainer(SFTConfig(name="x", data_dir=str(split)))


# --------------------------------------------------------------------------------------
# Configs
# --------------------------------------------------------------------------------------


def test_shipped_configs_load() -> None:
    assert load_midtrain_config(CONFIGS / "midtrain_nano.yaml").total_tokens == 50_000_000
    tiny = load_midtrain_config(CONFIGS / "midtrain_tiny.yaml")
    assert tiny.total_tokens == 1_000_000
    assert [s.kind for s in tiny.sources] == ["bin", "chat", "text"]
    assert sum(tiny.weights()) == pytest.approx(1.0)

    sft = load_sft_config(CONFIGS / "sft_nano.yaml")
    assert sft.max_seq == 1024 and sft.pack is False
    assert sft.train.muon_lr < 0.01  # fine-tuning LR, not a pretraining one

    rl = load_rl_config(CONFIGS / "rl_nano.yaml")
    assert rl.beta == 0.0 and rl.epsilon_high > rl.epsilon_low
    assert rl.std_normalize is False and rl.loss_agg == "token" and rl.dynamic_sampling


def test_overrides_coerce_types() -> None:
    cfg = load_rl_config(CONFIGS / "rl_tiny.yaml")
    override(cfg, ["steps=7", "temperature=0.5", "dynamic_sampling=true", "tasks=chain_sum,gcd"])
    assert cfg.steps == 7 and cfg.temperature == 0.5
    assert cfg.dynamic_sampling is True and cfg.tasks == ["chain_sum", "gcd"]

    sft = load_sft_config(CONFIGS / "sft_tiny.yaml")
    override(sft, ["max_steps=3", "train.muon_lr=0.001"])
    assert sft.max_steps == 3 and sft.train.muon_lr == 0.001

    with pytest.raises(ValueError, match="unknown config key"):
        override(sft, ["nonsense=1"])


def test_rl_config_validates() -> None:
    with pytest.raises(ValueError, match="group_size"):
        RLConfig(group_size=1)
    with pytest.raises(ValueError, match="loss_agg"):
        RLConfig(loss_agg="banana")


def test_midtrain_weights_normalise() -> None:
    cfg = MidtrainConfig()
    with pytest.raises(ValueError, match="sum to 0"):
        cfg.weights()


# --------------------------------------------------------------------------------------
# Midtraining shards
# --------------------------------------------------------------------------------------


def test_midtrain_shards_are_llmc_readable(tmp_path: Path) -> None:
    """What prepare_midtrain_data.py writes must be exactly what r52.data reads."""
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
    from prepare_midtrain_data import write_bin

    from r52.data import load_shard, read_shard_header

    toks = np.arange(1000, 1000 + 4096, dtype=np.uint16)
    toks[:8] = [BOS, USER_START, 5, 6, USER_END, ASSISTANT_START, 7, ASSISTANT_END]
    path = tmp_path / "midtrain_train_000001.bin"
    assert write_bin(path, toks) == 4096
    assert read_shard_header(path) == 4096
    back = load_shard(path)
    assert np.array_equal(np.asarray(back), toks)
    assert int(back[0]) == BOS  # chat ids survive the uint16 round trip


# --------------------------------------------------------------------------------------
# reasoning-gym adapter
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text,expected",
    [
        ("thinking... <answer>42</answer> done", "42"),
        ("<answer>  7 </answer>", "7"),
        ("blah\n#### 13", "13"),
        ("line one\nline two\n", "line two"),
        ("", ""),
        ("<answer>9", "9"),
    ],
)
def test_extract_answer(text: str, expected: str) -> None:
    assert extract_answer(text) == expected


def test_format_score() -> None:
    assert format_score("<answer>1</answer>") == 1.0
    assert format_score("no tags here") == 0.0
    assert format_score("<answer>unclosed") == 0.0


def test_rgym_scores_with_the_real_verifier() -> None:
    pytest.importorskip("reasoning_gym")
    from r52.posttrain.rgym import RGymTask

    task = RGymTask("chain_sum", size=8, seed=0)
    item = task[0]
    assert task.score(f"<answer>{item.answer}</answer>", item) == 1.0
    assert task.score(f"the total is {item.answer}", item) == 1.0  # last-number fallback
    assert task.score("<answer>not a number at all</answer>", item) == 0.0
    assert "chain_sum" in RGymTask.available()


def test_rgym_messages_carry_the_system_turn() -> None:
    pytest.importorskip("reasoning_gym")
    from r52.posttrain.rgym import SYSTEM_PROMPT, RGymTask

    item = RGymTask("chain_sum", size=4, seed=0)[0]
    msgs = item.messages(SYSTEM_PROMPT)
    assert [m["role"] for m in msgs] == ["system", "user"]
    assert msgs[1]["content"] == item.question
    assert [m["role"] for m in item.messages("")] == ["user"]


# --------------------------------------------------------------------------------------
# pass@k
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "n,c,k,expected",
    [
        (4, 0, 4, 0.0),
        (4, 4, 4, 1.0),
        (4, 1, 1, 0.25),
        (4, 1, 4, 1.0),
        (10, 1, 5, 0.5),          # 1 - C(9,5)/C(10,5) = 1 - 126/252
        (8, 2, 2, 1 - (6 / 8) * (5 / 7)),
    ],
)
def test_pass_at_k(n: int, c: int, k: int, expected: float) -> None:
    assert pass_at_k(n, c, k) == pytest.approx(expected)


def test_pass_at_k_is_monotone_in_k() -> None:
    vals = [pass_at_k(16, 3, k) for k in range(1, 17)]
    assert all(b >= a for a, b in pairwise(vals))
    assert vals[-1] == 1.0
    assert pass_at_k(0, 0, 4) == 0.0 and pass_at_k(4, 1, 0) == 0.0


# --------------------------------------------------------------------------------------
# GRPO pieces
# --------------------------------------------------------------------------------------


class _Comp:
    def __init__(self, tokens, finished=True):
        self.tokens = tokens
        self.finished = finished

    def __len__(self):
        return len(self.tokens)


def _cfg(**kw) -> RLConfig:
    base = {"group_size": 4, "dynamic_sampling": True, "std_normalize": False}
    base.update(kw)
    return RLConfig(**base)


def test_advantages_are_group_centred() -> None:
    prompts = [[BOS, USER_START, 5, USER_END, ASSISTANT_START]]
    comps = [[_Comp([10, 11]), _Comp([12]), _Comp([13, 14, 15]), _Comp([16])]]
    rewards = [[1.0, 0.0, 0.0, 0.0]]
    roll, stats = build_rollouts(prompts, comps, rewards, _cfg())
    assert roll is not None and stats["groups_dropped"] == 0
    adv = np.asarray(roll.advantages)
    assert adv.sum() == pytest.approx(0.0, abs=1e-6)
    assert adv[0] > 0 and (adv[1:] < 0).all()


def test_std_normalize_rescales_but_keeps_the_sign() -> None:
    prompts = [[BOS, ASSISTANT_START]]
    comps = [[_Comp([10]), _Comp([11]), _Comp([12]), _Comp([13])]]
    rewards = [[1.0, 0.0, 0.0, 0.0]]
    plain = np.asarray(build_rollouts(prompts, comps, rewards, _cfg())[0].advantages)
    normed = np.asarray(build_rollouts(prompts, comps, rewards, _cfg(std_normalize=True))[0].advantages)
    assert np.sign(plain).tolist() == np.sign(normed).tolist()
    assert abs(normed[0]) > abs(plain[0])  # std < 1 here, so normalising inflates


def test_dynamic_sampling_drops_zero_variance_groups() -> None:
    prompts = [[BOS, ASSISTANT_START], [BOS, ASSISTANT_START]]
    comps = [[_Comp([10]), _Comp([11])], [_Comp([12]), _Comp([13])]]
    rewards = [[0.0, 0.0], [1.0, 0.0]]
    roll, stats = build_rollouts(prompts, comps, rewards, _cfg(group_size=2))
    assert stats["groups_dropped"] == 1 and stats["groups_kept"] == 1
    assert roll is not None and len(roll) == 2

    roll, stats = build_rollouts(prompts, comps, [[0.0, 0.0], [0.0, 0.0]], _cfg(group_size=2))
    assert roll is None and stats["groups_dropped"] == 2

    # With dynamic sampling off, nothing is dropped (the tiny-model smoke path).
    roll, stats = build_rollouts(prompts, comps, [[0.0, 0.0], [0.0, 0.0]],
                                 _cfg(group_size=2, dynamic_sampling=False))
    assert roll is not None and stats["groups_dropped"] == 0


def test_rollout_mask_covers_only_completion_positions() -> None:
    prompt = [BOS, USER_START, 5, 6, USER_END, ASSISTANT_START]
    comps = [[_Comp([10, 11], finished=True), _Comp([12], finished=False)]]
    roll, _ = build_rollouts([prompt], comps, [[1.0, 0.0]], _cfg(group_size=2))
    assert roll is not None
    x, mask = np.asarray(roll.x), np.asarray(roll.mask)
    # Row 0: prompt(6) + [10, 11] + <|assistant_end|> (finished) = 9 ids, 3 supervised.
    assert x[0, :6].tolist() == prompt
    assert x[0, 6:9].tolist() == [10, 11, ASSISTANT_END]
    assert mask[0, :5].sum() == 0, "prompt positions must not be in the policy gradient"
    assert mask[0, 5:8].sum() == 3
    # Row 1 is unfinished, so no stop token is appended and it is padded to the same width.
    assert x[1, 6] == 12 and x[1, 7] == PAD
    assert mask[1].sum() == 1


def test_token_logprobs_matches_log_softmax() -> None:
    model = GPT(tiny_model_config(vocab_size=256, block_size=32))
    mx.eval(model.parameters())
    x = mx.array(np.random.default_rng(0).integers(0, 256, (2, 12)).astype(np.int32))
    got = np.asarray(token_logprobs(model, x, x[:, 1:]))
    from mlx import nn

    logits = model.logits(x)[:, :-1, :].astype(mx.float32)
    ref = np.asarray(
        mx.take_along_axis(nn.log_softmax(logits, axis=-1), x[:, 1:, None], axis=-1).squeeze(-1)
    )
    assert got.shape == (2, 11)
    np.testing.assert_allclose(got, ref, atol=2e-4)
    assert (got <= 0).all()
