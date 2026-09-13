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
