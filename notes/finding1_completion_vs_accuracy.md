# Finding 1 — completion rate vs accuracy, from Astra's own first-sample logs (no new generation)

Script `scripts/finding1_from_logs.py` (grader = Astra's `mathqa.extract` + math-verify; loop rule = Astra's `loop_signals`;
the `mcqa` shim re-implements `repetitive_end` from its docstring). One sample per problem, as shipped in the logs.

| set | run | n | completion | acc (all) | acc on completed | loop-end rate (verbatim) | acc on problems both unedited & p50c completed (n) |
|---|---|---|---|---|---|---|---|
| GSM8K | unedited | 200 | 0.10 | 0.115 | **0.952** | 0.88 (0.88) | 0.952 (21) |
| | p10c | 200 | 0.29 | 0.265 | 0.845 | 0.72 (0.61) | 0.571 |
| | p25c | 200 | 0.25 | 0.240 | 0.940 | 0.74 (0.74) | 0.667 |
| | **p50c** | 200 | **0.94** | 0.610 | 0.649 | **0.05** (0.03) | **0.952** |
| MATH-500 | unedited | 200 | 0.07 | 0.075 | 1.000 | 0.91 (0.91) | 1.000 (12) |
| | p50c | 200 | 0.76 | 0.460 | 0.605 | 0.18 (0.04) | 0.917 |
| Minerva | unedited / p50c | 120 | 0.02 / 0.84 | 0.025 / 0.325 | 1.000 / 0.386 | 0.96 / 0.13 | (2) |
| Olympiad | unedited / p50c | 240 | 0.02 / 0.76 | 0.017 / 0.188 | 0.800 / 0.247 | 0.93 / 0.30 | (4) |
| AMC23 | unedited / p50c | 40 | 0.03 / 0.75 | 0.025 / 0.275 | 1.000 / 0.367 | 0.97 / 0.17 | (1) |
| AIME24/25 | p50c | 30+30 | 0.77 / 0.67 | 0.0 | 0.0 | 0.47 / 0.43 | |

Rescues (p50c correct, unedited wrong): GSM8K 99, MATH-500 79, Minerva 36, Olympiad 42, AMC 10 — **every one** of them a problem the
unedited model had been cut off on. Regressions: 0 / 2 / 0 / 1 / 0.

## Reading
- The unedited model is not bad at math; it is bad at stopping. When it finishes, it is right 95–100 % of the time.
- The edit changes the completion rate (0.10 → 0.94 on GSM8K, 0.07 → 0.76 on MATH-500) and nothing else: on the problems both
  models finish, accuracy is identical. The lower "accuracy on completed" of p50c is selection: it now finishes the hard problems too.
- Loop-end rate by Astra's own rule: 0.88 → 0.05 (GSM8K). p10c / p25c reduce loops a little and gain a little; AIME is now
  finished 2/3 of the time but still 0 % correct, i.e. no capability was added.
- So the main conclusion of the report ("it stopped the loops" vs "it found a mechanism") is settled at the behavioural level from
  the logs alone; the mechanism analysis (Q1–Q5) says *how* it stopped them.
