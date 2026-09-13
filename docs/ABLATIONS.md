# The data-ablation lab (Phase 4 spec)

*Planner spec, 2026-09-13. Rationale: `research/06` Appendix A.1 — data quality is the only lever
with a published 2×, and DataDecide shows corpus rankings at ~150M params predict the 1B winner
~80% of the time. On this machine we cannot afford 150M-scale ablations (each ~1 day per 0.1B
tokens), so we run a **proxy ladder** and report where our proxies agree with the published
150M/1B rankings before trusting them.*

## Scale and budget

| Proxy | Params | Tokens | tok/s (measured) | Wall-clock | Use |
|---|---|---|---|---|---|
| `tiny_abl` | ~12M (6 layers, d=384) | 100M | ~9k (est.) | ~3 h | screening (one config per evening) |
| `nano_30m` | ~35M | 200M | 6.5k | ~8.5 h | confirmation of the top 2–3 |

Runs never overlap with a headline run (one GPU). Each run logs the fixed-split val loss, bpb,
HellaSwag (acc_norm) and the 6 cheapest CORE tasks (`--tasks` subset) so ranking is by *downstream*
signal, not loss alone (`research/06` A.3: "do not chase validation loss alone").

## Axis 1 — corpus (own 32K BPE tokenizer, trained once on a 1B-token FineWeb sample)

| Candidate | Source | Note |
|---|---|---|
| FineWeb (10B sample) | `HuggingFaceFW/fineweb` | the Rung 0 baseline |
| FineWeb-Edu | `HuggingFaceFW/fineweb-edu` | classifier-filtered |
| DCLM-baseline slice | `mlfoundations/dclm-baseline-1.0` | +7 CORE vs FineWeb-Edu at 7B (`research/03` §1.3) |
| Dolma 3 mix slice | `allenai/dolma3_mix-150B-1025` | fully open, reproducible |
| Ultra-FineWeb slice | `openbmb/Ultra-FineWeb` | Apache-2.0, freshest |
| FinePDFs-Edu (≤25% blended into FineWeb-Edu) | `HuggingFaceFW/finepdfs-edu` | "new signal nobody has exhausted" |

Expected published ordering at matched tokens: DCLM ≳ Ultra-FineWeb > FineWeb-Edu > FineWeb.
If our proxy disagrees, report the disagreement — that is a result too.

## Axis 2 — architecture knobs (on the winning corpus, `tiny_abl` scale)

`use_value_embeds` on/off · `use_unet_skips` on/off · `mlp` relu2 vs swiglu · `softcap` 15 vs off ·
`qk_norm` on/off · Muon LR {0.035, 0.05, 0.07} (the "flat bowl" check) · seq 512 vs 1024.
One knob at a time from the Rung 0 default; 2 seeds for anything within noise.

## Axis 3 — tokenizer

GPT-2 (50,257) vs own 32K BPE (rustbpe/HF tokenizers) at fixed *bytes* of training data; compare on
bpb and HellaSwag (tokenizer-neutral), never on loss.

## Deliverables

- `configs/ablations/*.yaml`, `scripts/ablate.sh <axis>` (runs the matrix sequentially, resumable).
- `results/ablations/<axis>.md` — table with val bpb, HellaSwag acc_norm ± CI, CORE-6, wall-clock,
  and the *published* ranking next to ours.
- A short section in `README.md`: "what the Mac Mini found", with the honest caveat that proxies
  at 12–35M params are below DataDecide's validated scale.

## Ordering

Runs after Rung 0 and Run A complete (GPU free), lowest-cost axis first: tokenizer (2 runs) →
corpus (6 runs) → architecture (≈10 runs). Total ≈ 2–3 weeks of background compute; each run is
committed as it lands, so partial results are useful immediately.

---

# Implementation

*Builder note, 2026-09-13. What exists, how to run it, what it measured, and what the planner
still has to decide. Deviations from this spec are in `docs/DEVIATIONS.md` under "ablation
builder" (A1–A12).*

## What was built

| Piece | File | What it is |
|---|---|---|
| Corpus registry | `r52/ablate/corpora.yaml` | The six Axis-1 candidates, each pinned to a dataset **revision**, with its text column, license, file layout and the caveat that applies to it. Every row verified against the live HF API on 2026-09-13. |
| Streaming reader | `r52/ablate/corpora.py` | Turns a registry row into a document stream over `hf://` URLs. Handles the two corpora that need care (see A5, A6). |
| Corpus preparation | `scripts/prepare_corpus.py` | Streams a corpus (or a blend) into llm.c `.bin` shards + `manifest.json`. Val split held out first. Leaves through `os._exit` (P6). |
| Tokenizer training | `r52/tokenizer_train.py` | Byte-level BPE, 32,768 default, nanochat's split pattern, the nine specials at the top of the vocabulary. `R52Tokenizer` has `GPT2Tokenizer`'s interface. |
| The matrix | `r52/ablate/matrix.py` | The three axes as lists of `r52.train` overrides; 2 + 6 + 9 = 17 cells. |
| One cell | `r52/ablate/run.py` | Trains then evaluates one cell in two `nice -n 10` subprocesses; writes `results/ablations/<axis>/<cell>.json`. |
| The evaluator | `r52/ablate/evals.py` | val bpb + HellaSwag + CORE-6, with the tokenizer swapped in (A7). |
| The report | `r52/ablate/report.py` | `results/ablations/<axis>.md` — the spec's table plus our rank next to the published one. |
| The driver | `scripts/ablate.sh` | Sequential, resumable, refuses to share the GPU with a headline run. |
| Configs | `configs/ablations/{tiny_abl,nano_confirm}.yaml` | The screening and confirmation proxies. |

Model sizes, measured: `tiny_abl` is 10,616,847 non-embedding / 29,933,583 "N for 6N" /
68,567,055 total parameters at vocab 50,304; `nano_confirm` is byte-identical in shape to
`nano_30m` (25,165,840 non-embedding).

## Running an axis

```bash
# 0. the shared 32K BPE, trained once (docs/ABLATIONS.md Axis 1)
python -m r52.tokenizer_train --name fineweb32k --corpus fineweb --bytes 1000000000 \
    --vocab-size 32768 --compare-gpt2

# 1. the corpora, one directory per cell (100M tokens each; ~200 MB of shards)
for c in fineweb fineweb-edu dclm dolma3 ultra-fineweb; do
  python scripts/prepare_corpus.py --corpus $c --tokens 100000000 \
      --tokenizer data/tokenizers/fineweb32k
done
python scripts/prepare_corpus.py --corpus fineweb-edu --blend finepdfs-edu:0.25 \
    --tokens 100000000 --tokenizer data/tokenizers/fineweb32k

# 2. the axis: see the plan first, then run it
scripts/ablate.sh corpus --dry-run
scripts/ablate.sh corpus --tokenizer data/tokenizers/fineweb32k

# resumable: re-running skips cells whose JSON exists
scripts/ablate.sh corpus --list
```

`scripts/ablate.sh` refuses to start while another `r52.train` is alive (one GPU, one training
job) unless `--force` is given. Each cell writes `results/ablations/<axis>/<cell>.json`
carrying the config path, every override, the corpus manifest's identity (dataset id,
revision, bytes/token, exact shard files), both commands, the trainer's end-of-run record and
the three metrics. `python -m r52.ablate.report --axis corpus` renders the table.

The tokenizer axis is the exception to "matched tokens": it fixes **bytes**, and `run.py`
converts the byte budget into a token budget using the `bytes_per_token` each corpus manifest
measured while writing its shards.

## Measured during the build

* **Our 32K BPE vs GPT-2's**, trained on 50 MB of FineWeb-Edu, measured on a held-out 2.68 MB:
  **4.6467 vs 4.6287 bytes/token** (+0.39% for ours), 6.0 s to train. The Axis-3 premise holds
  but is small on English educational web text; the real lever at `d=384` is the head, 12.6M
  parameters against 19.3M. See A12.
* **Corpus preparation, measured on real slices** (GPT-2 BPE, while a 124M run held the GPU):

  | corpus | slice | wall-clock | throughput | bytes/token | note |
  |---|---|---|---|---|---|
  | `fineweb-edu` | 2.10M tok / 9.7 MB / 1,934 docs | 9.6 s | ~220k tok/s | 4.6211 | one 8 MB parquet row group at a time |
  | `dclm` | 220k tok / 0.93 MB / 218 docs | 8.0 s | ~28k tok/s | 4.2303 | `.jsonl.zst` streaming works (A4) |
  | `dolma3` | 130k tok / 0.65 MB / **17 docs** | 48.8 s | ~10k tok/s | 4.1536 | 24 files opened round-robin; the slice touched 17 distinct sources instead of one topic (A5) |

  Extrapolated to the 100M-token cells: FineWeb-Edu ~8 min, **Dolma 3 ~3 h**. Dolma 3 is 20×
  slower per token because opening 24 zstd streams costs up front and its olmOCR-PDF documents
  are enormous (17 documents for 130k tokens). Budget a corpus-preparation evening for it.
* **One ablation cell end to end** (2-layer proxy, 20k tokens, HellaSwag 20, CORE `copa`):
  **5.9 s**, 0.66 GiB peak — `tests/test_ablate_run.py` is that run.
* `data/tokenizers/fineweb-edu-32k/` is left on disk from the A12 measurement. The Axis-1
  tokenizer the matrix defaults to (`data/tokenizers/fineweb32k`, trained on FineWeb per the
  spec) has **not** been trained — step 0 of "Running an axis" above.

## Decisions the planner owes this lab

1. **The CORE-6 subset.** The shipped default is the six cheapest as the spec asks, but
   `bigbench_repeat_copy_logic` (32 items, greedy exact match) and `agi_eval_lsat_ar` sit at or
   below random for 12–35M models — two of six are noise. `matrix.CORE6_ALTERNATIVE` trades
   them for `lambada_openai` + `hellaswag_zeroshot` at ~15× the forward rows. Pick one *before*
   the corpus axis starts; changing it after invalidates cross-cell comparability.
2. **Dolma 3's slice is uniform by file, not by its published token weights** (A5). Either
   accept the caveat (it is recorded in every manifest) or supply the mixture table.
3. **Ultra-FineWeb costs ~200 MB minimum to stream** because its parquet row groups are ~225 MB
   (A6). Budget for it, or drop the cell.
4. **The Axis-3 budget.** `matrix.TOKENIZER_AXIS_BYTES` is 420 MB of text ≈ 100M GPT-2 tokens,
   which means preparing the *same* FineWeb-Edu text twice (~1 GB of streaming). Halving it
   halves the cost and roughly doubles the noise.
5. **`nano_confirm` is 8.5 h per cell.** The spec says "top 2–3"; at three cells that is three
   nights during which no headline run can proceed.
6. **Dolma 3 preparation is ~3 h of streaming for its 100M-token cell** (table above), which is
   as long as the training run it feeds. Prepare it well ahead of the axis.

## Planner decisions (2026-09-13)

1. **CORE-6 = `CORE6_ALTERNATIVE`** (drop `bigbench_repeat_copy_logic` and `agi_eval_lsat_ar`,
   which are at or below chance at 12–35M params; take `lambada_openai` + `hellaswag_zeroshot`
   instead). Set as the default in `r52/ablate/matrix.py`. Fixed before any corpus-axis cell runs.
2. **Dolma 3 slice:** accept the uniform-by-file caveat; it is recorded in every manifest and in
   the report table. Prepare its 100M-token cell the evening before the corpus axis starts.
3. **Ultra-FineWeb:** keep the cell; the ~200 MB minimum stream is acceptable.
4. **Axis-3 budget:** keep 420 MB of text (≈100M GPT-2 tokens); noise matters more than cost here.
5. **Confirmation:** `nano_confirm` on the **top 2** corpus cells only (two nights), after the
   headline runs, never overlapping them.
