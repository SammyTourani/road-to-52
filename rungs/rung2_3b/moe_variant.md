# MoE variant — 30B-A3B instead of 3B dense

`results/ladder.md`'s **Rung 2 (MoE)** row: 30B total / 3B active params, 60B tokens, **identical
cost to Rung 2 dense — $815 spot / $2,106 on-demand, 867 GPU-hours, 8×H100** — because training
FLOPs scale with *active* parameters, not total. `research/04-landscape-and-compute.md` §B.5:
*"You get 30B of capacity for a 3B training bill. This is exactly why Qwen3-30B-A3B-shaped models
exist."* `results/ladder.md`'s target cell names it directly: "Qwen3-30B-A3B-shaped capacity for a
3B-dense training bill."

## torchtitan already ships this one, fully — better than Rung 2 dense's case

Checking torchtitan's `qwen3` model family (same pinned commit, `150c4f73a035e8778f37d9f603a32534841f2a60`)
turned up something stronger than "MoE is supported": **a complete, ready-to-run job config for
exactly this shape already exists**, no custom recipe needed.

`torchtitan/models/qwen3/__init__.py`:

```python
def _30b_a3b(attn_backend, moe_comm_backend="standard", *, seq_len) -> Qwen3Model.Config:
    dim = 2048
    head_dim = 128
    n_layers = 48
    vocab_size = 151936
    return Qwen3Model.Config(
        ...
        layers=_build_qwen3_moe_layers(
            n_layers=n_layers, dim=dim, n_heads=32, n_kv_heads=4, head_dim=head_dim,
            moe_hidden_dim=768, num_experts=128, top_k=8,
            rope=CosSinRoPE.Config(dim=head_dim, max_context_length=seq_len, theta=1000000.0),
            ...
        ),
    )

qwen3_configs = {..., "30B-A3B": (_30b_a3b, 40960), ...}
```

And `torchtitan/models/qwen3/config_registry.py` already wraps it into a full `Trainer.Config`:

```python
def qwen3_30b_a3b(seq_len: int | None = None) -> Trainer.Config:
    model_spec = model_registry("30B-A3B", seq_len=seq_len)
    return Trainer.Config(
        ...,
        hf_assets_path="./assets/hf/Qwen3-30B-A3B",
        dataloader=GrainDataLoader.Config(dataset=ConcatThenSplitPackingConfig(dataset=DATASETS["c4"])),
        optimizer=default_adamw(lr=8e-4),
        lr_scheduler=LRSchedulersContainer.Config(warmup_steps=600),
        training=TrainingConfig(
            num_tokens_per_microbatch_per_dp_rank=2 * model_spec.max_context_length,
            max_context_length=model_spec.max_context_length,
            steps=3000,
        ),
        parallelism=ParallelismConfig(
            data_parallel_shard_degree=-1, tensor_parallel_degree=1,
            context_parallel_degree=1, pipeline_parallel_degree=1,
        ),
        checkpoint=CheckpointManager.Config(interval=500, last_save_model_only=False, export_dtype="float16"),
        activation_checkpoint=FullAC.Config(),
    )
```

**This is not our config** — it's `pytorch/torchtitan`'s own shipped baseline, unmodified, for the
real, publicly-released Qwen3-30B-A3B shape (128 experts, top-8 routing, 48 layers, GQA 32/4 heads
— matches Alibaba's published Qwen3-30B-A3B architecture). Also pure FSDP2, no TP/PP/CP, like our
3B dense recipe — the same reasoning applies at this active-parameter count.

## Launch

Matches the README's own documented quickstart pattern exactly, with `qwen3`/`qwen3_30b_a3b` in
place of `llama3`/`llama3_8b`:

```bash
NGPU=8 MODULE=qwen3 CONFIG=qwen3_30b_a3b ./run_train.sh
```

No `torchtitan_recipes/road_to_52.py` generation step is needed for this variant — unlike Rung 2
dense, torchtitan's own module already has the full job. `rungs/rung2_3b/launch.sh` does not
currently branch to this path (it always generates the dense 3B recipe); running the MoE variant
for real means invoking the command above directly from a `launch.sh`-style clone of torchtitan at
the same pinned commit, or asking for a `--moe` flag to be added to `launch.sh` if this becomes the
preferred rung.

## Token budget and steps

`qwen3_30b_a3b()`'s defaults are tuned as a demo (`steps=3000`), not for a specific token target —
same as `llama3_8b()`. To hit Rung 2's 60B tokens:

- **At the flavor's native `seq_len=40960`** (Qwen3's real long-context length, used if `seq_len`
  is left `None`): `num_tokens_per_microbatch_per_dp_rank = 2 × 40960 = 81,920`; at `NGPU=8`,
  `data_parallel_shard_degree=-1` (shard across all 8), grad_accum=1 (no other field sets it):
  global batch ≈ 655,360 tokens/step → `steps ≈ 60e9 / 655,360 ≈ 91,553` to reach 60B tokens. This
  is our arithmetic from the shipped defaults, not an upstream-stated number — recompute if the
  microbatch multiplier needs to change to fit memory (128 experts resident per FSDP shard is a
  different memory profile than a dense model; untested here, same caveat as Rung 2 dense's
  microbatch multiplier).
- **At `seq_len=4096`**, matching Rung 2 dense for a more apples-to-apples pair: `qwen3_30b_a3b`
  takes a `seq_len` argument (`qwen3_30b_a3b(seq_len=4096)`), exactly like `llama3_8b`/our own
  `llama3_3b_road_to_52` do — a one-line wrapper in `torchtitan_recipes` (the same pattern
  `torchtitan/config/README.md` documents, quoted in `rung2_3b/README.md`) is enough:
  ```python
  # torchtitan_recipes/road_to_52_moe.py
  from torchtitan.models.qwen3.config_registry import qwen3_30b_a3b

  def qwen3_30b_a3b_road_to_52():
      config = qwen3_30b_a3b(seq_len=4096)
      config.training.steps = 915_528   # see arithmetic below; recompute if you change the multiplier
      return config
  ```
  Keeping the shipped microbatch multiplier of 2 unchanged (`num_tokens_per_microbatch_per_dp_rank
  = 2 × 4096 = 8,192`), global batch = `8,192 × 8 GPUs = 65,536` tokens/step (grad_accum=1) — **8×
  smaller than dense's 524,288**, because dense's recipe raises its multiplier to 16 (untested,
  see `torchtitan_3b.toml`) while this keeps the MoE job's own shipped default. 128 resident
  experts per layer is a different, larger memory profile than a dense FFN, so raising this
  multiplier to match dense's throughput is not assumed safe without an actual memory-fit check —
  `steps = ceil(60e9 / 65,536) = 915,528` follows from the conservative, unmodified default. If a
  real run's memory-fit pass shows headroom for a larger multiplier, steps drops proportionally;
  recompute rather than reusing either number here.

## Prerequisites and caveats

- `hf_assets_path="./assets/hf/Qwen3-30B-A3B"` — a Qwen tokenizer download via
  `scripts/download_hf_assets.py --repo_id Qwen/Qwen3-30B-A3B --assets tokenizer`, same mechanism
  as the dense variant's Llama 3.1 tokenizer download. **UNVERIFIED whether this specific repo is
  gated** — Qwen models are commonly ungated (Apache-2.0) on Hugging Face unlike Meta's Llama
  family, but this was not independently confirmed in this session; check before assuming no
  `HF_TOKEN` step is needed.
- **Inference/serving cost is a separate bill this package does not model.** `results/ladder.md`'s
  own note: *"Inference memory is a different bill entirely and is not modelled here."* A 30B-total
  checkpoint costs far more to serve than a 3B dense one even though training cost is identical —
  relevant if the plan after Rung 2 is to actually chat with the result, not just report an eval
  number.
- Same eval plan as `eval_plan.md` applies — the "beats a named model" framing there does not
  change based on dense vs. MoE, since the comparison targets (LLaMA-1/Llama-2 7B) are dense models
  either way.
