# Copyright 2026 The road-to-52 authors.
# SPDX-License-Identifier: Apache-2.0
"""Convert an r52 checkpoint into an mlx-lm-loadable model directory.

    python -m r52.export runs/b1/ckpt/best models/b1-mlx
    python -m r52.export runs/b1/ckpt/best models/b1-mlx --dtype float32 --no-tokenizer

Output::

    config.json           mlx-lm model config (+ the full r52 config under "r52")
    model.safetensors     weights, bf16 by default
    r52gpt.py             only when the architecture needs it (see below)
    tokenizer.json,       GPT-2 BPE, fetched from `openai-community/gpt2`
    tokenizer_config.json

**Two export targets.**  Our core weight names deliberately mirror
``mlx_lm/models/nanochat.py`` (``transformer.wte``, ``transformer.h.{i}.attn.c_q`` ...),
so a model built with ``use_value_embeds=false`` and ``use_unet_skips=false`` exports as
``model_type: "nanochat"`` and loads in **stock mlx-lm with no plugin**.  With value
embeddings or U-net skips enabled -- the default, because they are a large token-efficiency
win -- the extra tensors have no home in mlx-lm's nanochat class, so the export instead
writes ``model_type: "r52gpt"`` plus ``model_file: "r52gpt.py"``.  ``mlx_lm.utils.load_model``
imports that file directly from the model directory, so ``mlx_lm.generate``,
``mlx_lm.server``, ``mlx_lm.lora`` and ``mlx_lm.evaluate`` still work unmodified and
nothing has to be installed.

The GPT-2 reference model used for apples-to-apples evaluation is **not** importable into
this architecture (HF GPT-2 has learned position embeddings, LayerNorm and GELU; we have
RoPE, RMSNorm and ReLU^2 -- there is no weight mapping).  Use ``--gpt2-reference`` to
materialise ``openai-community/gpt2`` as a sibling mlx-lm directory instead; see
``docs/DEVIATIONS.md``.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

import mlx.core as mx
from mlx.utils import tree_flatten

from . import chat_template
from .checkpoint import checkpoint_config
from .config import Config, ModelConfig
from .model import GPT

__all__ = ["export_checkpoint", "load_exported", "main", "model_config_to_mlx_lm"]

_DTYPES = {"bfloat16": mx.bfloat16, "float16": mx.float16, "float32": mx.float32}
_TOKENIZER_REPO = "openai-community/gpt2"
_TOKENIZER_FILES = ("tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt")


def ve_index(mcfg: ModelConfig) -> list[int]:
    """Per-layer value-embedding table index (``-1`` = no value embedding)."""
    idx = [-1] * mcfg.n_layer
    if mcfg.use_value_embeds:
        k = mcfg.n_value_embeds
        for j in range(k):
            t = 0 if mcfg.value_embed_share else j
            idx[j] = t
            idx[mcfg.n_layer - k + j] = t
    return idx


def model_config_to_mlx_lm(cfg: Config) -> dict[str, Any]:
    """Build the ``config.json`` body for an mlx-lm model directory."""
    m = cfg.model
    out: dict[str, Any] = {
        "model_type": "nanochat" if m.nanochat_compatible else "r52gpt",
        "hidden_size": m.n_embd,
        "num_hidden_layers": m.n_layer,
        "num_attention_heads": m.n_head,
        "num_key_value_heads": m.n_kv_head,
        "vocab_size": m.vocab_size,
        "max_position_embeddings": m.block_size,
        "intermediate_size": m.mlp_hidden,
        "rope_theta": m.rope_base,
        "tie_word_embeddings": False,
        # mlx-lm reads `eos_token_id` from config.json and hands it to the tokenizer wrapper,
        # so generation stops on the chat EOS as well as GPT-2's <|endoftext|>.
        "bos_token_id": chat_template.BOS,
        "eos_token_id": chat_template.eos_token_ids(),
        "pad_token_id": chat_template.PAD,
        "r52": cfg.to_dict(),
    }
    if not m.nanochat_compatible:
        out.update(
            {
                "model_file": "r52gpt.py",
                "head_dim": m.head_dim,
                "softcap": m.softcap,
                "qk_norm": m.qk_norm,
                "mlp": m.mlp,
                "norm_eps": m.norm_eps,
                "use_value_embeds": m.use_value_embeds,
                "n_value_embeds": m.n_value_embeds,
                "value_embed_share": m.value_embed_share,
                "use_unet_skips": m.use_unet_skips,
                "ve_index": ve_index(m),
            }
        )
    return out


def _fetch_tokenizer(out_dir: Path, model_max_length: int = 1024) -> bool:
    """Copy GPT-2 tokenizer files into ``out_dir`` and install the chat format.

    Returns ``False`` if no tokenizer could be fetched (offline and nothing cached).
    The chat special tokens (:mod:`r52.chat_template`) are appended at ids 50257..50264 and
    the Jinja ``chat_template`` is written into ``tokenizer_config.json``, so
    ``mlx_lm.generate --apply-chat-template`` / ``mlx_lm.chat`` / ``mlx_lm.server`` work on
    every export -- base models included, where the tokens simply never fire.
    """
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:  # pragma: no cover
        return False
    got = False
    for name in _TOKENIZER_FILES:
        try:
            src = hf_hub_download(repo_id=_TOKENIZER_REPO, filename=name)
        except Exception:
            continue
        shutil.copyfile(src, out_dir / name)
        got = True
    if got:
        chat_template.install_chat_tokenizer(out_dir, model_max_length=model_max_length)
    return got


def export_checkpoint(
    ckpt: str | Path,
    out_dir: str | Path,
    dtype: str = "bfloat16",
    tokenizer: bool = True,
    cfg: Config | None = None,
) -> Path:
    """Write an mlx-lm model directory from an r52 checkpoint directory."""
    ckpt = Path(ckpt)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    cfg = cfg or checkpoint_config(ckpt)

    weights = mx.load(str(ckpt / "model.safetensors"))
    target = _DTYPES[dtype]
    weights = {k: v.astype(target) for k, v in weights.items()}
    mx.save_safetensors(str(out / "model.safetensors"), weights, metadata={"format": "mlx"})

    conf = model_config_to_mlx_lm(cfg)
    (out / "config.json").write_text(json.dumps(conf, indent=2))

    if conf["model_type"] == "r52gpt":
        shutil.copyfile(Path(__file__).parent / "mlx_plugin" / "r52gpt.py", out / "r52gpt.py")

    if tokenizer and not _fetch_tokenizer(out, cfg.model.block_size):
        print(f"[r52.export] warning: could not fetch GPT-2 tokenizer files into {out}")
    return out


def export_model(model: GPT, out_dir: str | Path, cfg: Config, dtype: str = "bfloat16",
                 tokenizer: bool = False) -> Path:
    """Export a live :class:`~r52.model.GPT` (used by tests; no checkpoint needed)."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    target = _DTYPES[dtype]
    weights = {k: v.astype(target) for k, v in tree_flatten(model.parameters())}
    mx.save_safetensors(str(out / "model.safetensors"), weights, metadata={"format": "mlx"})
    conf = model_config_to_mlx_lm(cfg)
    (out / "config.json").write_text(json.dumps(conf, indent=2))
    if conf["model_type"] == "r52gpt":
        shutil.copyfile(Path(__file__).parent / "mlx_plugin" / "r52gpt.py", out / "r52gpt.py")
    if tokenizer:
        _fetch_tokenizer(out, cfg.model.block_size)
    return out


def load_exported(path: str | Path):
    """Load an exported directory with stock mlx-lm (no tokenizer needed)."""
    from mlx_lm.utils import load_model

    model, config = load_model(Path(path))
    model.eval()
    return model, config


def gpt2_reference(out_dir: str | Path, dtype: str = "bfloat16") -> Path:
    """Materialise ``openai-community/gpt2`` as an mlx-lm directory (evaluation baseline)."""
    from huggingface_hub import snapshot_download
    from mlx_lm.convert import convert

    out = Path(out_dir)
    # mlx-lm 0.31.3's `save()` resolves a *repo id* with `snapshot_download(repo,
    # local_files_only=True)` and no allow-patterns, so it demands every file in the repo --
    # including the .tflite / onnx / tf / flax weights its own downloader deliberately skips --
    # and raises IncompleteSnapshotError on `openai-community/gpt2`.  Resolving the snapshot
    # ourselves and handing `convert` a local *path* takes its `src_path.exists()` branch
    # instead.  See docs/DEVIATIONS.md, "eval builder".
    src = snapshot_download(
        _TOKENIZER_REPO,
        allow_patterns=["*.json", "model*.safetensors", "*.py", "*.txt", "*.jsonl", "*.jinja"],
    )
    convert(hf_path=src, mlx_path=str(out), quantize=False, dtype=dtype)
    return out


# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="r52.export", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("ckpt", nargs="?", help="checkpoint directory, e.g. runs/b1/ckpt/best")
    p.add_argument("out", nargs="?", help="output model directory")
    p.add_argument("--dtype", default="bfloat16", choices=sorted(_DTYPES))
    p.add_argument("--no-tokenizer", action="store_true", help="do not fetch GPT-2 tokenizer files")
    p.add_argument("--verify", action="store_true", help="reload with mlx-lm and compare logits")
    p.add_argument("--gpt2-reference", metavar="DIR", default=None,
                   help="instead of exporting, convert openai-community/gpt2 to an mlx-lm dir")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.gpt2_reference:
        out = gpt2_reference(args.gpt2_reference, args.dtype)
        print(f"wrote {out}")
        return 0
    if not args.ckpt or not args.out:
        build_parser().error("ckpt and out are required unless --gpt2-reference is given")

    out = export_checkpoint(args.ckpt, args.out, args.dtype, not args.no_tokenizer)
    conf = json.loads((out / "config.json").read_text())
    print(f"wrote {out} (model_type={conf['model_type']}"
          + (", plugin r52gpt.py" if conf["model_type"] == "r52gpt" else ", native mlx-lm") + ")")

    if args.verify:
        cfg = checkpoint_config(args.ckpt)
        ours = GPT(cfg.model)
        from .checkpoint import load_checkpoint

        load_checkpoint(args.ckpt, ours)
        theirs, _ = load_exported(out)
        mx.random.seed(0)
        idx = mx.random.randint(0, cfg.model.vocab_size, (8, min(64, cfg.model.block_size)))
        a = ours(idx)
        b = theirs(idx).astype(mx.float32)
        mx.eval(a, b)
        diff = float(mx.abs(a - b).max())
        print(f"max |logit diff| over 8 prompts = {diff:.2e} ({'PASS' if diff < 1e-2 else 'FAIL'} @ 1e-2)")
        return 0 if diff < 1e-2 else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
