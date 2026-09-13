# MLX pretraining throughput benchmark

**Apple M4** (10 GPU cores, 16.0 GiB unified) · macOS 26.5.2 · mlx 0.32.2 · mlx-lm 0.31.3 · Python 3.12.12 · commit `a8de28c` · 2026-09-13

Measured by `python -m r52.bench` -- a full training step (gradient accumulation to 61440/64512/65536 tokens, fp32 grad accumulation, global-norm clip, Muon+AdamW update), timed over completed optimizer steps only, after warm-up.

`FLOPs/token = 6 * (non_embedding_params + lm_head_params) + 12 * n_layer * n_embd * seq`. MFU denominators: **4.26 TFLOPS** (M4 theoretical fp32 peak, arXiv 2502.05317 Table 1) and **3.6 TFLOPS** (bf16 4096^3 matmul measured on this machine).

| size | L | d | N (6N) | non-emb | total | seq | mb | accum | precision | compile | ckpt | tok/s | TFLOPS | MFU 4.26 | MFU 3.6 | peak GiB | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 30M | 8 | 512 | 50.9M | 25.2M | 102.4M | 512 | 12 | 10 | mixed | yes | no | 7,293 | 2.412 | 56.6% | 67.0% | 5.61 | ok |
| 30M | 8 | 512 | 50.9M | 25.2M | 102.4M | 512 | 12 | 10 | bf16 | yes | no | 7,496 | 2.479 | 58.2% | 68.9% | 4.80 | ok |
| 30M | 8 | 512 | 50.9M | 25.2M | 102.4M | 512 | 6 | 21 | fp32 | yes | no | 4,977 | 1.646 | 38.6% | 45.7% | 5.88 | ok |
| 30M | 8 | 512 | 50.9M | 25.2M | 102.4M | 1024 | 4 | 16 | mixed | yes | no | 6,479 | 2.306 | 54.1% | 64.0% | 4.66 | ok |
| 30M | 8 | 512 | 50.9M | 25.2M | 102.4M | 1024 | 6 | 10 | bf16 | yes | no | 6,765 | 2.408 | 56.5% | 66.9% | 5.21 | ok |
| 30M | 8 | 512 | 50.9M | 25.2M | 102.4M | 1024 | 2 | 32 | fp32 | yes | no | 4,408 | 1.569 | 36.8% | 43.6% | 4.79 | ok |
| 60M | 12 | 640 | 91.2M | 59.0M | 155.6M | 512 | 8 | 16 | mixed | yes | no | 4,250 | 2.526 | 59.3% | 70.2% | 5.94 | ok |
| 60M | 12 | 640 | 91.2M | 59.0M | 155.6M | 512 | 8 | 16 | bf16 | yes | no | 4,443 | 2.640 | 62.0% | 73.3% | 4.72 | ok |
| 60M | 12 | 640 | 91.2M | 59.0M | 155.6M | 512 | 2 | 64 | fp32 | yes | no | 2,915 | 1.732 | 40.7% | 48.1% | 4.43 | ok |
| 60M | 12 | 640 | 91.2M | 59.0M | 155.6M | 1024 | 2 | 32 | mixed | yes | no | 3,647 | 2.339 | 54.9% | 65.0% | 4.55 | ok |
| 60M | 12 | 640 | 91.2M | 59.0M | 155.6M | 1024 | 4 | 16 | bf16 | yes | no | 3,949 | 2.533 | 59.5% | 70.4% | 5.17 | ok |
| 60M | 12 | 640 | 91.2M | 59.0M | 155.6M | 1024 | 1 | 64 | fp32 | yes | no | 2,545 | 1.633 | 38.3% | 45.4% | 4.71 | ok |
| 124M | 12 | 768 | 123.6M | 84.9M | 200.8M | 512 | 6 | 21 | mixed | yes | no | 3,367 | 2.687 | 63.1% | 74.6% | 5.81 | ok |
| 124M | 12 | 768 | 123.6M | 84.9M | 200.8M | 512 | 8 | 16 | bf16 | yes | no | 3,563 | 2.843 | 66.8% | 79.0% | 5.06 | ok |
| 124M | 12 | 768 | 123.6M | 84.9M | 200.8M | 512 | 2 | 64 | fp32 | yes | no | 2,447 | 1.953 | 45.9% | 54.2% | 5.07 | ok |
| 124M | 12 | 768 | 123.6M | 84.9M | 200.8M | 1024 | 2 | 32 | mixed | yes | no | 3,034 | 2.593 | 60.9% | 72.0% | 5.11 | ok |
| 124M | 12 | 768 | 123.6M | 84.9M | 200.8M | 1024 | 4 | 16 | bf16 | yes | no | 3,286 | 2.808 | 65.9% | 78.0% | 5.37 | ok |
| 124M | 12 | 768 | 123.6M | 84.9M | 200.8M | 1024 | 1 | 64 | fp32 | yes | no | 2,205 | 1.885 | 44.2% | 52.3% | 5.19 | ok |
| 350M | 24 | 1024 | 0.0M | 0.0M | 0.0M | 512 | 0 | 0 | mixed | yes | no | — | — | — | — | — | skipped |
| 350M | 24 | 1024 | 353.5M | 302.0M | 456.5M | 512 | 2 | 64 | bf16 | yes | no | 1,260 | 2.864 | 67.2% | 79.5% | 4.78 | ok |
| 350M | 24 | 1024 | 0.0M | 0.0M | 0.0M | 512 | 0 | 0 | fp32 | yes | no | — | — | — | — | — | skipped |
| 350M | 24 | 1024 | 0.0M | 0.0M | 0.0M | 1024 | 0 | 0 | mixed | yes | no | — | — | — | — | — | skipped |
| 350M | 24 | 1024 | 353.5M | 302.0M | 456.5M | 1024 | 1 | 64 | bf16 | yes | no | 1,168 | 2.830 | 66.4% | 78.6% | 4.99 | ok |
| 350M | 24 | 1024 | 0.0M | 0.0M | 0.0M | 1024 | 0 | 0 | fp32 | yes | no | — | — | — | — | — | skipped |

### Skipped configurations

- `350M/512/mixed`: peak 8.06 GiB > 6.0 GiB at micro_batch=1
- `350M/512/fp32`: peak 8.11 GiB > 6.0 GiB at micro_batch=1
- `350M/1024/mixed`: peak 8.11 GiB > 6.0 GiB at micro_batch=1
- `350M/1024/fp32`: peak 8.21 GiB > 6.0 GiB at micro_batch=1
