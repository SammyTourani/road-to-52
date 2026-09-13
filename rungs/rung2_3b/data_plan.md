# Data plan — Rung 2 (100B-token mix)

This describes the data pipeline for Rung 2; **it does not implement it** (per the build rules for
this package: no downloads larger than a README, no training). Everything below is the plan
`launch.sh` expects to be substituted in for its `DATASETS["c4"]` placeholder before a real run.

## The mix: one pre-blended, pre-shuffled repo

`research/03-open-data.md` §10.2, "Mid — ~100B tokens" gives a direct answer rather than a recipe
to assemble:

> "**Fastest path:** just take
> [**HuggingFaceFW/finepdfs_edu_50BT-dclm_30BT-fineweb_edu_20BT**](https://huggingface.co/datasets/HuggingFaceFW/finepdfs_edu_50BT-dclm_30BT-fineweb_edu_20BT)
> — **100B tokens, 267.9 GB, ODC-By, pre-blended and pre-shuffled**, and it encodes HF's current
> recommended English ratio."

That's **50BT FinePDFs-Edu + 30BT DCLM-baseline + 20BT FineWeb-Edu**, already mixed and shuffled by
Hugging Face — no per-source weighting decision needed on our end, and the exact 100B-token budget
Rung 2 targets. `research/03` §9.1's measured bytes-per-token table confirms the size: **267.9 GB
raw, 2.68 bytes/token** (measured live against the HF repo, not estimated).

**License:** ODC-By — commercially clean, consistent with `docs/PLAN.md`'s hard rule of avoiding
NC-licensed data (the ClimbMix problem documented in `rungs/rung1_nanochat/README.md` does not
apply here).

**Why not build a custom mix instead** (`research/03`'s "Or build it" alternative, 7 source
datasets with hand-picked weights): more moving parts, more licenses to individually verify, and no
stated evidence it beats the single pre-blended repo at this token budget. Revisit only if a
specific weakness in the pre-blended mix shows up during a real run (e.g. via the DataDecide-style
150M-parameter proxy `docs/PLAN.md` §3.1 mentions).

## Tokenization

**Tokenizer: the Llama 3.1 tokenizer** (128256-vocab tiktoken-style BPE) — matches `torchtitan_3b.toml`'s
`hf_assets_path`, and is the tokenizer torchtitan's `llama3_8b()`/`llama3_70b()`/`llama3_405b()`
jobs all use for the same reason (it's `Llama3StateDictAdapter`'s and `model_registry("3B", ...)`'s
`vocab_size=128256` baked into the model shape — a different tokenizer would need a different
`vocab_size` and therefore a different `_3b()`-equivalent model definition).

**Tool: `datatrove`** (`HuggingFaceFW/datatrove`) — per the build brief and confirmed as the right
choice independently: `research/02-open-training-stack.md` Stage 10, "**The default.** It is the
*actual* tool behind FineWeb and SmolLM3, not a demo," and the same tool that built the
`finepdfs_edu_50BT-dclm_30BT-fineweb_edu_20BT` repo itself, so there is no format mismatch between
how the source data was produced and how we'd process it further.

**What this step does, described (not implemented here):** run datatrove's tokenization block over
the mix with the Llama 3.1 tokenizer, packing documents into fixed-length sequences and writing a
flat token-ID array plus a document/sequence offsets index — the standard datatrove tokenization
output shape (`.ds` binary shards in datatrove's own format, or exported to the plain
`tokens.bin` + `document_offsets.npy` pair described next, whichever torchtitan-side consumer is
used).

## What torchtitan actually consumes — checked, not assumed

This needed checking directly: torchtitan's dataloader is **not** a fixed raw-token-bin reader like
nanoGPT/llm.c/nanochat use. `torchtitan/components/data/README.md` at the pinned commit opens with
its own mental model:

```
1. Source (e.g. jsonl):        SourceConfig       -> RandomAccessDataSource | IterDataset
2. Dataset (filter/process):   SingleDatasetConfig -> MapDataset | IterDataset
3. Compose (optional):         e.g. FirstFitPackingConfig(dataset=DatasetMixConfig(...))
4. Dataloader:                 GrainDataLoader      -> TrainerBatch
5. Trainer:                    model forward/backward
```

It's a **Grain**-based (Google's data-loading library) pipeline with several supported source
types. Two of them are real options for our 100B mix:

**Option A — HuggingFace source, tokenize on the fly, no separate tokenization step at all.**
`HuggingFaceRandomAccessSource.Config(path=..., split=...)` or `HuggingFaceStreamingSource.Config(...)`
points directly at a dataset repo (ours: `HuggingFaceFW/finepdfs_edu_50BT-dclm_30BT-fineweb_edu_20BT`);
a `TextProcessor` extracts the text field and `ConcatThenSplitPackingConfig` packs it into
`seq_len`-length sequences, tokenizing as part of that pipeline. This is what every shipped example
in `config_registry.py` does (`dataset=ConcatThenSplitPackingConfig(dataset=DATASETS["c4"])`, where
`DATASETS["c4"]` is exactly this kind of HF-source config). **Simplest option — no datatrove step
needed at all** if the corpus can be re-streamed reliably from HF for the full run.

**Option B — pre-tokenize to a numpy memmap, consume via a small custom source.** torchtitan's own
README documents this as "Adding your own source — Example: Pretokenized data," with a complete,
short reference implementation:

```python
class PretokenizedMemmapSource(Configurable, RandomAccessDataSource):
    @dataclass(kw_only=True, slots=True)
    class Config(Configurable.Config):
        tokens_path: str
        document_offsets_path: str

    def __init__(self, config, *, dataset_iteration_policy):
        self.tokens = np.memmap(config.tokens_path, dtype=np.uint32, mode="r")
        self.offsets = np.load(config.document_offsets_path)

    def __len__(self):
        return len(self.offsets) - 1

    def __getitem__(self, index):
        start, end = self.offsets[index : index + 2]
        return np.asarray(self.tokens[start:end], dtype=np.int64)
```

paired with a `TokensToTextSequence` processor and `ConcatThenSplitPackingConfig`, same as Option A
from that point on. **This is the option this data plan recommends** for an actual paid run: a
multi-day, 8-GPU job billing by the hour should not depend on Hugging Face's network availability
for its entire duration, and a `uint32` token-ID memmap + a document-offsets index is *exactly* what
datatrove's tokenization step naturally produces. Concretely: datatrove tokenizes the 100B-token mix
with the Llama 3.1 tokenizer (vocab 128256 needs `uint32`, not `uint16` — see Disk estimate below),
writes `tokens.bin` (flat `uint32` array) and `document_offsets.npy` (one `int64` per document
boundary), and `torchtitan_recipes/road_to_52.py`'s dataloader block is swapped from the
`DATASETS["c4"]` placeholder to `SingleDatasetConfig(source=PretokenizedMemmapSource.Config(...),
processor=TokensToTextSequence.Config(), ...)` per the pattern quoted above. **Described, not
implemented** — this swap is real code to write and test against the actual prepared files, out of
scope for a package that runs no downloads or training.

## Decontamination

`allenai/decon`, per `research/03-open-data.md` §8 (already the project's stated standard —
`docs/PLAN.md` §5's data table lists `allenai/decon` as the decontamination tool "against every
eval we report"). The exact method, quoted from `research/03` §8.2 (Ai2's own `decon`
`config/default.yaml`, verified against the live file 2026-09-13):

- token-level (not word-level) matching, `cl100k` tokenizer, 5-gram indexed spans
- `contamination_score_threshold = 0.80` (the default — `research/03` explicitly recommends the
  default over Ai2's own more aggressive repro configs: *"the standard default configuration
  results in meaningfully higher accuracy"*)
- run as `decon detect --purify` over the prepared corpus, before tokenization for training

**Decontaminate against every eval this rung reports** — see `eval_plan.md`'s task list (MMLU,
HellaSwag, ARC-Challenge, GSM8K, HumanEval). `decon` ships MMLU pre-bundled; the rest are in its
~200-benchmark `config/evals.yaml` and pulled with its own `evals` command (`research/03` §8.3).

`research/03` §8.4's caveat applies directly here: *"UNVERIFIED: whether nanochat performs any
decontamination, and whether ClimbMix ships decontaminated."* The `finepdfs_edu_50BT-dclm_30BT-fineweb_edu_20BT`
mix's own decontamination status against our specific eval list is likewise not independently
verified by this package — run `decon` regardless of what the source datasets claim, per
`rungs/README.md`'s checklist item 4.

## Disk and time estimates

**Disk**, from `research/03-open-data.md` §9.1/§9.2 (measured, not estimated, for the raw download;
computed for the packed form):

| Stage | Size | Basis |
|---|---|---|
| Raw download (the pre-blended repo) | **267.9 GB** | measured live against the HF repo, `research/03` §9.1 |
| Packed for training (`uint32` token IDs) | **≈ 400 GB** | 100B tokens × 4 bytes/token — `uint32` is *required* here, not a choice: `research/03` §9.1 notes the 2-bytes/token `uint16` option only applies at "vocab ≤65k," and our vocab is 128256 |
| Document-offsets index | negligible | one `int64` (8 bytes) per document boundary; document count ≪ token count |
| **Total on the rented node's disk** | **≈ 670 GB** (raw + packed, if both are kept) | fits comfortably on a standard rented-node NVMe (typically 1–4 TB); drop the raw parquet after packing to cut this to ≈400 GB if space is tight |

**Time** — two pieces have a sourced basis, one does not:

- **Download**: 267.9 GB at a rented node's network speed. Not provider-verified in this package
  (bandwidth varies by provider and is not in `research/04`'s scope); illustrative range only —
  at 1 Gbps (~125 MB/s) ≈ 36 min, at 10 Gbps (~1.25 GB/s) ≈ 3.6 min. Confirm the actual node's
  bandwidth before budgeting this into the paid GPU-hour clock (see the cost tip below).
- **Tokenization** (datatrove): **UNVERIFIED — no sourced throughput figure for this package's
  exact pipeline.** `research/02-open-training-stack.md` Stage 10 only states datatrove "is
  CPU-bound and parallelizes fine," with no tokens/sec number. Do not invent one; measure it on the
  actual node with a small slice before trusting an ETA for the full 100B tokens.
- **Decontamination** (`decon`): `research/03` §8.1 gives a throughput figure — "~34 µs/doc." Doc
  count for 100B tokens depends on the mix's average document length, which we have not measured
  for this specific blend. Illustrative only, assuming ~500–1,000 tokens/doc (a typical web-text
  range, not sourced for this mix specifically): ~100–200M documents × 34 µs ≈ **1–2 hours**
  single-threaded, and `decon` is embarrassingly parallel across documents, so this shrinks
  linearly with cores available on the prep box.

**Cost tip, stated explicitly because it matters at these dollar amounts:** download, tokenization,
and decontamination are all CPU/network-bound, not GPU-bound. Do all three on a cheap CPU box (or
the rented node *before* `torchrun` starts, if billed by the hour regardless) rather than idling
8×H100s at $7.52–$19.44/hour (`rungs/README.md`'s node-price table) while data prep runs. This is
the single biggest lever for not overspending the ladder's estimated $815/$2,037.
