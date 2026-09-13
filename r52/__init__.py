# Copyright 2026 The road-to-52 authors.
# SPDX-License-Identifier: Apache-2.0
"""road-to-52: an Apple-Silicon-native (MLX) LLM pretraining stack.

Modules
-------
``r52.config``      dataclasses + YAML I/O for model / data / training configuration
``r52.model``       the GPT model (nanochat / modded-nanogpt lineage) in MLX
``r52.optim``       Muon + AdamW routing, WSD schedule, global grad-norm clipping
``r52.data``        fineweb10B-gpt2 shard reader with a deterministic, resumable cursor
``r52.tokenizer``   GPT-2 tiktoken wrapper + bits-per-byte helpers
``r52.checkpoint``  save / load (safetensors weights + optimizer state + cursor + rng)
``r52.train``       the pretraining loop
``r52.bench``       throughput benchmark
``r52.export``      checkpoint -> mlx-lm-loadable directory
"""

__version__ = "0.0.1"
