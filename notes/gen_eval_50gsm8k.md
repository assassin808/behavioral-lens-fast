# Generation check (vLLM, 50 GSM8K problems, 1 sample, 16k cap, Astra's sampling; my simple grader, see `scripts/gen_eval_gsm8k.py`)

Run before the GPU was freed (7 % memory share); all_alpha1.5 and rand_cols were cut off when the full-GPU queue replaced this run.

| variant | acc | closed `</think>` | cut at 16k | mean tokens | rep20 |
|---|---|---|---|---|---|
| base | 0.14 | 0.14 | 0.86 | 14561 | 0.56 |
| sel_alpha2 (same columns, flip only) | 0.32 | 0.40 | 0.66 | 13749 | 0.43 |
| **p50c** (α = 6) | **0.56** | **0.80** | 0.30 | 10595 | 0.11 |

Generation reproduces the teacher-forced ordering of Q4: α = 2 gets less than half of the gain of α = 6. The full screening
runs (200 + 200 problems × 2 samples, Astra's grader) are in `notes/screen/summary.jsonl` when the queue finishes.
