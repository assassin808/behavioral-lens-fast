# Edited model (p50c) with thinking OFF — the missing cell of the decoding-baseline table

Run 2026-11-05 on a rented RunPod H100 with xuanqi's `scripts/screen_eval.py` (xuanqi branch, unchanged): 200 GSM8K + 200 MATH-500, 2 samples each, Astra's think35 sampling, 16k cap, `--no-think`, seed 0. Base re-run on the same pod as a control. ~7 min GPU.

| config | GSM8K completion | GSM8K acc | MATH completion | MATH acc | loop-end (GSM8K / MATH) | mean tokens (GSM8K / MATH) |
|---|---|---|---|---|---|---|
| base, thinking off (this run) | 0.985 | 0.560 | 0.855 | 0.510 | 0.005 / 0.033 | 1543 / 5468 |
| base, thinking off (xuanqi table) | 0.99 | 0.555 | 0.83 | 0.512 | 0.00 / 0.05 | 1513 / 5995 |
| **p50c, thinking off** | 0.993 | **0.522** | 0.865 | **0.487** | 0.000 / 0.000 | 1468 / 5701 |
| p50c, thinking on (xuanqi table) | 0.76 | 0.570 | 0.34 | 0.312 | 0.07 / 0.09 | 9964 / 13759 |

Paired by problem (mean of 2 samples, 200 problems): p50c − base = −0.037 ± 0.023 (GSM8K), −0.022 ± 0.028 (MATH), SE. Discordant samples: base-only 55 vs p50c-only 40 (GSM8K), 59 vs 50 (MATH).

Reading:
- Base thinking-off is reproduced (0.56 / 0.51), so the harness matches.
- The edit does not help thinking-off mode; if anything it costs 2–4 points (within ~1.5 SE, not significant at this n).
- The few loops thinking-off still has (13/400 MATH) disappear under the edit, consistent with the edit acting on the same u axis in both modes (test 8).
- Best configuration overall is still base with thinking off (≈ p50c thinking-on on GSM8K, much better on MATH, 1/5–1/7 tokens).

Per-sample jsonl (7.7 MB each, not committed): /scratch/yang0531/blf_data/nothink_runpod/ on Killarney.
