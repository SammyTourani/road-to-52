# Copyright 2026 The road-to-52 authors.
# SPDX-License-Identifier: Apache-2.0
"""Post-training for road-to-52: midtraining -> SFT -> pass@k probe -> RLVR -> chat.

The stages, in the order ``docs/PLAN.md`` §3.1 prescribes:

``r52.posttrain.midtrain``
    Continue pretraining a **base** checkpoint on a short, LR-decaying mixture of web text,
    chat-rendered instruction data and math, so instruction-following is seeded into the
    base model rather than bolted on.  Produces llm.c ``.bin`` shards consumed by the stock
    :class:`r52.train.Trainer`.
``r52.posttrain.sft``
    Supervised fine-tuning on rendered conversations with an **assistant-only loss mask**.
``r52.posttrain.passk``
    pass@1 / pass@k probe on reasoning-gym.  §3.1 requires this *before* RL: high pass@k
    means RL can only sharpen, low pass@k means it can move the boundary.
``r52.posttrain.rl``
    DAPO-style GRPO on reasoning-gym with a verifier reward.

Every stage operates on the GPT-2 BPE plus the eight chat special tokens defined in
:mod:`r52.chat_template`, which live in the spare ids of the padded vocabulary, so no stage
resizes an embedding table.
"""

from __future__ import annotations

__all__ = ["__doc__"]
