# Results log

Appended automatically by `r52.eval.report` and by hand after each milestone. Every row carries: command, sample count / limit, tokenizer, sequence length, git commit, wall-clock.

| date | run | eval | value | conditions | commit |
|---|---|---|---|---|---|
| 2026-09-13 | gpt2-124m-reference | hellaswag | 29.38 % acc_norm | examples 10042, few-shot 0, tokenizer gpt2 (tiktoken), block_size 1024, protocol llm.c dev/data/hellaswag.py completion style, token-length acc_norm; wall-clock 685 s; `python -m r52.eval.hellaswag --model models/gpt2-mlx` | 5c209e2+dirty |
| 2026-09-13 | gpt2-124m-reference | val_loss | 3.4471 nats/token | examples 10485760, tokenizer gpt2 (tiktoken), block_size 1024, split fineweb_val_000000.bin first 10,485,760 tokens, non-overlapping windows, precision bf16 compute, fp32 log-sum-exp; wall-clock 1730 s; `python -m r52.eval.val_loss --model models/gpt2-mlx --block-size 1024 --max-tokens 10485760 --micro-batch 2` | 5c209e2+dirty |
