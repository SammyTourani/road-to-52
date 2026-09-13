# MLX pretraining throughput benchmark

**Apple M4** (10 GPU cores, 16.0 GiB unified) · macOS 26.5.2 · mlx 0.32.2 · mlx-lm 0.31.3 · Python 3.12.12 · commit `835296e` · 2026-09-13

Measured by `python -m r52.bench` -- a full training step (gradient accumulation to 30720/65536 tokens, fp32 grad accumulation, global-norm clip, Muon+AdamW update), timed over completed optimizer steps only, after warm-up.

`FLOPs/token = 6 * (non_embedding_params + lm_head_params) + 12 * n_layer * n_embd * seq`. MFU denominators: **4.26 TFLOPS** (M4 theoretical fp32 peak, arXiv 2502.05317 Table 1) and **3.6 TFLOPS** (bf16 4096^3 matmul measured on this machine).

| size | L | d | N (6N) | non-emb | total | seq | mb | accum | precision | compile | ckpt | tok/s | TFLOPS | MFU 4.26 | MFU 3.6 | peak GiB | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 30M | 8 | 512 | 50.9M | 25.2M | 102.4M | 512 | 12 | 5 | bf16 | yes | no | 7,412 | 2.451 | 57.5% | 68.1% | 4.80 | ok |
| 30M | 8 | 512 | 50.9M | 25.2M | 102.4M | 512 | 6 | 10 | fp32 | yes | no | 4,917 | 1.626 | 38.2% | 45.2% | 5.88 | ok |
| 30M | 8 | 512 | 50.9M | 25.2M | 102.4M | 1024 | 4 | 16 | mixed | yes | no | 6,519 | 2.320 | 54.4% | 64.4% | 4.66 | ok |
| 30M | 8 | 512 | 50.9M | 25.2M | 102.4M | 1024 | 4 | 16 | mixed | no | no | 5,719 | 2.035 | 47.8% | 56.5% | 4.81 | ok |
| 124M | 12 | 768 | 123.6M | 84.9M | 200.8M | 1024 | 2 | 32 | mixed | yes | no | 3,024 | 2.584 | 60.7% | 71.8% | 5.11 | ok |
| 124M | 12 | 768 | 123.6M | 84.9M | 200.8M | 1024 | 2 | 32 | mixed | no | no | 2,824 | 2.414 | 56.7% | 67.0% | 5.11 | ok |
