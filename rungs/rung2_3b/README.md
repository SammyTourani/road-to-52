# Rung 2 — 3B-class Llama-3-style dense model, from scratch, on a rented 8×H100 node

Maps to **Rung 2**, **Rung 2 (MoE)**, and **Rung 2b** in
[`results/ladder.md`](../../results/ladder.md). Package: this README, [`data_plan.md`](data_plan.md),
[`torchtitan_3b.toml`](torchtitan_3b.toml), [`launch.sh`](launch.sh), [`eval_plan.md`](eval_plan.md),
and the MoE alternative in [`moe_variant.md`](moe_variant.md). Not run. No training, no downloads,
no packages installed in producing this package — everything below was verified read-only against
[`pytorch/torchtitan`](https://github.com/pytorch/torchtitan) via `gh api`.

## Pinned commit

```
repo:    https://github.com/pytorch/torchtitan
commit:  150c4f73a035e8778f37d9f603a32534841f2a60
date:    2026-09-13   (current `main` HEAD on the day this package was built)
license: BSD-3
```

Verified with `gh api repos/pytorch/torchtitan/commits/main --jq '{sha: .sha, date: .commit.committer.date}'`.
Re-check before a real launch if much time has passed, and re-pin deliberately.

## The upstream layout moved — what we found and how we adapted

The build brief for this package pointed at
`torchtitan/models/llama3/train_configs/llama3_8b.toml`, "or whatever the current path is — list
the directory first." We did:

```bash
$ gh api repos/pytorch/torchtitan/contents/torchtitan/models/llama3 --jq '.[].name'
README.md  __init__.py  config_registry.py  model.py  parallelize.py  sharding.py  state_dict_adapter.py
$ gh api "search/code?q=repo:pytorch/torchtitan+extension:toml" --jq '.items[].path'
pyproject.toml
```

**There is no `train_configs/` directory and no shipped `.toml` job config anywhere in the repo** —
only `pyproject.toml`, the Python packaging file. torchtitan moved from TOML job files to Python
`Trainer.Config` functions, selected with `--module`/`--config` (`torchtitan/config/README.md`,
same commit: *"A run is described by a **full configuration**: a function that returns a complete
`Trainer.Config`... Configurations are written in Python."*). The main README's own quickstart
confirms the new invocation shape: `MODULE=llama3 CONFIG=llama3_8b ./run_train.sh`.

We adapted rather than forcing the old shape: **`torchtitan_3b.toml` is a human-readable spec, not
something torchtitan's CLI reads** — every field in it is cited to the real dataclass field it
mirrors. **`launch.sh` is what actually runs**: it generates a small `torchtitan_recipes/road_to_52.py`
Python recipe (the pattern `torchtitan/config/README.md` itself documents for a custom job) inside
the cloned checkout, mirroring `config_registry.py`'s own `llama3_8b()` field for field. See
`torchtitan_3b.toml`'s header comment for the full explanation.

## Why these dims — we didn't pick them, torchtitan already ships them

torchtitan's Llama-3 model registry (`torchtitan/models/llama3/__init__.py`) already defines a
`"3B"` flavor alongside `"1B"`/`"8B"`/`"70B"`/`"405B"`:

```python
def _3b(attn_backend, tp_gemm_backend="default", *, seq_len) -> Llama3Model.Config:
    dim = 3072
    n_heads = 24
    n_kv_heads = 8
    n_layers = 28
    vocab_size = 128256
    return Llama3Model.Config(
        dim=dim, vocab_size=vocab_size, enable_weight_tying=True,
        ...
        layers=_build_llama3_layers(..., hidden_dim=compute_ffn_hidden_dim(dim, multiple_of=1024, ffn_dim_multiplier=1.0), ...),
    )

llama3_configs = {..., "3B": (_3b, 131072), ...}
```

We use this flavor verbatim rather than interpolating our own dims between "1B" and "8B". It is
also not an arbitrary "3B-ish" shape: we cross-checked every field against Meta's own published
Llama-3.2-3B `config.json` (fetched from an ungated HF mirror, `unsloth/Llama-3.2-3B`, since the
official `meta-llama/Llama-3.2-3B` repo is gated):

| Field | torchtitan `_3b()` | Llama-3.2-3B `config.json` | Match |
|---|---|---|---|
| hidden size | 3072 | `hidden_size: 3072` | ✅ |
| FFN hidden size | 8192 (computed, see below) | `intermediate_size: 8192` | ✅ |
| layers | 28 | `num_hidden_layers: 28` | ✅ |
| attention heads | 24 | `num_attention_heads: 24` | ✅ |
| KV heads (GQA) | 8 | `num_key_value_heads: 8` | ✅ |
| vocab | 128256 | `vocab_size: 128256` | ✅ |
| RoPE theta | 500000 | `rope_theta: 500000.0` | ✅ |
| weight tying | `enable_weight_tying=True` | `tie_word_embeddings: true` | ✅ |

Every field matches. **torchtitan's "3B" is a faithful reproduction of Meta's real Llama-3.2-3B
architecture**, not a hand-tuned approximation — which is the strongest possible justification for
"document your dim/layers/heads choice": we chose to change nothing.

**FFN hidden dim, derived, not guessed:** `torchtitan/models/common/feed_forward.py`,
`compute_ffn_hidden_dim(dim, *, multiple_of, ffn_dim_multiplier)`:

```python
hidden_dim = int(2 * 4 * dim / 3)
if ffn_dim_multiplier is not None:
    hidden_dim = int(ffn_dim_multiplier * hidden_dim)
return multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)
```

For `dim=3072`, `ffn_dim_multiplier=1.0`, `multiple_of=1024`: `2*4*3072/3 = 8192`, unchanged by the
1.0 multiplier, already a multiple of 1024 → **8192**. (Sanity check against a value we *do* know
independently: the same formula at `dim=4096, ffn_dim_multiplier=1.3, multiple_of=1024` for the
"8B" flavor gives `int(1.3 * 10922) = 14198 → round up to 14336` — the well-known, publicly correct
Llama-3-8B FFN size. The formula is confirmed correct, not just internally consistent.)

**Parameter count — our arithmetic, not an upstream-printed figure** (per-layer: fused QKV
`dim×dim + 2×dim×(n_kv_heads×head_dim)` + `wo` `dim×dim`, plus SwiGLU FFN `3×dim×ffn_hidden_dim`,
summed over 28 layers, plus the tied embedding `vocab_size×dim` once, bias terms omitted as
<0.03% of total):

```
attention/layer = 2×(3072×3072) + 2×(3072×1024)     =  25,165,824
ffn/layer       = 3×(3072×8192)                      =  75,497,472
per layer       ≈ 100,669,440   x 28 layers           = 2,818,744,320
tied embedding  = 128,256 × 3,072                     =   394,002,432
                                                    total ≈ 3,212,746,752  (~3.21B)
```

Dense model, so **active parameters == total parameters == ~3.21B**, matching `results/ladder.md`'s
Rung 2 row ("Params (total / active): 3B / 3B").

## Sequence length: 4096, not the flavor's native 131072

The task brief specifies seq_len 4096; `_3b()`'s registered ceiling is 131072 (Llama-3.2-3B's real
long-context length). We use the shorter length deliberately for this rung: 4096 keeps attention
and activation-memory cost low enough to fit a large microbatch on a single 8×H100 node without TP
(see `torchtitan_3b.toml` [parallelism]), and 60B pretraining tokens is a *short* training horizon
by modern standards (Chinchilla-optimal for 3B active params, not a long-context regime) — spending
compute on 32×-longer sequences would not be Chinchilla-optimal use of a 60B-token budget.
`model_registry("3B", seq_len=4096)` supports this override directly (the flavor's `model_registry`
signature takes `seq_len: int | None = None` and raises only if you ask for *more* than 131072, per
its own source).

## Precision: bf16, float8 optional

Default is plain bf16 — the same default every shipped torchtitan job in `config_registry.py` uses
unless it explicitly opts into a converter. `launch.sh --float8` generates the recipe with
torchtitan's `Float8LinearConverter` instead, following the exact pattern
`torchtitan/models/llama3/config_registry.py::llama3_405b()` and `llama3_debugmodel_mxfp8()` use
(`model_registry(..., converters=[Float8LinearConverter.Config(model_compile_enabled=True)])`,
with `compile=CompileConfig(enable=True, components=["model"])` set alongside it, since the
converter needs `torch.compile` on to hit its real speedup). torchtitan's README lists Float8 as a
first-class supported precision on H100 (bullet 6, "Float8 support") — this is not an experimental
flag we're inventing, just one we default off pending a real correctness/throughput check on this
specific config, which this package does not run.

## LR: our choice, UNVERIFIED — sweep before trusting it

torchtitan does not ship a job config at 3B to copy an LR from. We interpolated within its own
shipped range: 405B → `8e-5`, 70B → `1.5e-4`, 8B → `3e-4` (all from `config_registry.py`, quoted
above in each function). Smaller models tolerating a larger LR is the standard pattern across that
range, so we picked **`4e-4`** for 3B — a step beyond 8B's `3e-4` in the same direction the 8B→70B→405B
progression moves, and in the same ballpark as `qwen3_14b()`/`qwen3_30b_a3b()`/`qwen3_32b()`'s
shared `8e-4` (a different model family/tokenizer, so not directly transferable, but a useful upper
bound). **This is not a published number for this exact architecture and token budget — run a short
LR sweep (e.g. 3 short runs at 2e-4/4e-4/6e-4, a few hundred steps each) before committing the full
60B-token budget to it.** `docs/PLAN.md` §3.1 point 3 makes the same point about not carrying
forward someone else's LR/batch recipe unexamined, in the Muon context; it applies here to AdamW
just as much.

## Parallelism: FSDP2 only, no TP/PP/CP

`torchtitan_3b.toml` [parallelism]. A 3.21B dense model's AdamW state (fp32 moments, roughly
4× the parameter bytes) fits comfortably sharded across 8×80GB H100s with plain FSDP2 — matching
the pattern torchtitan's own 8B/14B/30B-A3B/32B jobs use (all `tensor_parallel_degree=1` etc.,
`data_parallel_shard_degree=-1`); TP only turns on at 70B+ in the shipped configs. Adding TP here
would add communication overhead for no memory benefit at this size.

## Prerequisites for a real run

1. **Hugging Face access to the gated `meta-llama/Llama-3.1-8B` repo**, plus an `HF_TOKEN`.
   `launch.sh` reuses the Llama 3.1 tokenizer (README: *"download the Llama 3.1 tokenizer... Follow
   the instructions on the official meta-llama repository to ensure you have access"*) because
   Llama-3.2-3B's tokenizer is the identical 128256-vocab tokenizer, so this is not a mismatch —
   but the gate is real friction if access isn't already approved. **UNVERIFIED / not checked**: an
   ungated mirror that ships the identical `tokenizer.json`/`tokenizer_config.json` (e.g. an
   `unsloth/*` mirror, the same kind used above to verify the 3B config) could substitute; we did
   not verify one is licensed cleanly enough to point a training job's asset path at, so it is not
   wired into `launch.sh`. Confirm before substituting.
2. **A CUDA environment torchtitan can install into.** README: nightly PyTorch is recommended;
   `pip install -r requirements.txt` from the cloned repo covers the rest, `torchao` separately for
   `--float8`.
3. **Disk and time for the data plan** — see `data_plan.md`.

## Cost — copied from `results/ladder.md`

| Rung | Tokens | GPU-hours | Wall-clock | $ spot | $ on-demand |
|---|---|---|---|---|---|
| **Rung 2** — 3B dense, Chinchilla-optimal | 60B | 867 | 108 h (4.5 d) | **$815** | $2,106 |
| **Rung 2 (MoE)** — 30B-A3B, same active-param bill | 60B | 867 | 108 h (4.5 d) | **$815** | $2,106 |
| **Rung 2b** — 150B tokens, the $2,000 budget | 150B | 2,167 | 271 h (11.3 d) | **$2,037** | $5,265 |

**No named model is claimed as "beaten" at Rung 2 or 2b.** `results/ladder.md`'s own note: *"NO
NAMED MODEL IS CLAIMED HERE. research/04 gives this rung's cost but no capability result, and the
obvious same-size comparator does not hold: SmolLM3 is also 3B but saw 11.2T tokens, 187x this
budget."* `docs/PLAN.md` §4 lists Rung 2's target loosely as "Llama-1 7B" — see `eval_plan.md` for
why that is framed as a same-*era*-recipe comparison under a stated token/parameter handicap, not a
promise of beating it.

To run 150B tokens (Rung 2b) instead of 60B, set `R52_TOKENS_TARGET=150000000000` when invoking
`launch.sh` — everything else (dims, seq_len, LR) is unchanged; only `steps` (and therefore
wall-clock and cost) scales.

## Files

| File | What |
|---|---|
| `README.md` | this file |
| `data_plan.md` | the 100B mix, tokenization, torchtitan dataloader hookup, decontamination |
| `torchtitan_3b.toml` | field-by-field spec (see its header for why it isn't CLI-consumable) |
| `launch.sh` | clones the pinned commit, generates the recipe, launches; `--dry-run`, `--float8`; `bash -n` clean |
| `eval_plan.md` | lm-eval tasks, Llama-1/Llama-2 7B published targets, eval conditions |
| `moe_variant.md` | the 30B-A3B alternative — torchtitan ships this one fully off the shelf |
