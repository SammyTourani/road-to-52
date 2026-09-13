# Copyright 2026 The road-to-52 authors.
# SPDX-License-Identifier: Apache-2.0
"""Self-contained mlx-lm model files shipped inside exported model directories.

``r52gpt.py`` is copied next to ``model.safetensors`` and referenced from ``config.json``
as ``"model_file": "r52gpt.py"``; ``mlx_lm.utils.load_model`` imports it with
``importlib.util.spec_from_file_location``, so exported models load with stock mlx-lm and
no plugin installation.
"""
