# Exp B: threshold calibration (embedding cosine, MiniLM-L6-v2)

- **Pool.** Pairs (i, j) with j ∈ [i−11, i−2] that share no 32-token n-gram, from the pilot set: 20 random texts per group,
  100 texts in total. Pairs per cosine bin: 0.5–0.6: 3,349; 0.6–0.7: 5,242; 0.7–0.8: 6,466; 0.8–0.9: 3,393; ≥ 0.9: 300.
- **Sample.** 12 pairs per bin (seed 1), shuffled. Group, class and cosine were hidden while labelling.
- **Labels.** Each pair was labelled "same content" if the same step, claim, check or interpretation doubt is restated, otherwise
  "different". Labeller: the agent (one labeller, who knew the hypothesis but not the bin/group).
- **Files.** Raw pairs: `/scratch/yang0531/blf_data/expB/calib_pairs_pilot.json`. Labels: `calib_labels_pilot.json`. Pairs
  labelled "different": 5, 7, 12, 13, 16, 24, 26, 28, 34, 37, 41, 42, 46, 50, 52, 53.

| cosine bin | same content | share |
|---|---|---|
| 0.5–0.6 | 2 / 12 | 17% |
| 0.6–0.7 | 8 / 12 | 67% |
| 0.7–0.8 | 10 / 12 | **83%** |
| 0.8–0.9 | 12 / 12 | 100% |
| ≥ 0.9 | 12 / 12 | 100% |

- **Rule (pre-registered).** τ is the lower edge of the lowest bin with ≥ 80% "same content". Here τ = **0.70**; robustness τ = 0.75.
- **Lexical Jaccard for comparison.** Jaccard ≥ 0.5: 9 / 9 "same". Jaccard 0.1–0.4: 35 / 51. The lexical measure misses many
  restatements, as expected for paraphrase.
- **What "different" pairs looked like.**
  - Continuation of a list (pair 42: "did I check 13… 17" vs "did I check 24… 32").
  - The same problem but a different sub-step (pair 26: side lengths vs the cos A computation).
  - A different interpretation of the problem (pair 28: x = 12 vs x = 24).
- **Caveat.** The "same content" bar is lenient: it also counts the same interpretation doubt re-raised with different arguments.
  Most sampled pairs came from thinking-on text (48 / 60), because thinking-off texts are short and have few eligible pairs.
