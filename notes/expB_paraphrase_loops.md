# Experiment B: does p50c still loop, just in different words? (paraphrase loops)

Branch `yang/expB`. Script `scripts/expB_paraphrase.py`. Per-text outputs: `/scratch/yang0531/blf_data/expB/` (not in git).

## Pre-registration (written 2026-11-05, before any text was segmented, scored or read; not edited afterwards)

**Mechanism-derived prediction.** Test 5 found that the edit breaks loops by suppressing *verbatim copying*: induction attention
to the earlier copy of the next token drops ~1/3. It does not touch the re-check urge itself. Test 8 found that in p50c's own text,
verbatim repeats still start but are cut short. If the re-check cycle survives and only the copying is blocked, p50c should cycle
through the same re-checking content **in different words**.

- **P1.** Among p50c thinking-on generations in the class "cut at the cap, not a verbatim loop, no paraphrase-loop flag by Astra's
  rule, not whitespace" (`cut: still reasoning`, xuanqi's 12% on MATH-500), at least **50%** are *cycling texts* (paraphrase-repeat
  rate ≥ 0.30, defined below).
- **P2.** Mean paraphrase-repeat rate in that class is at least **2×** that of (a) base thinking-off generations and (b) p50c
  thinking-on generations that finished correctly. For both, the bootstrap 95% CI of the difference (class minus control,
  resampling texts) excludes 0.
- **P3.** Over all p50c thinking-on texts, the paraphrase-repeat rate is at least 1.5× that of base thinking-on text *before* its
  verbatim loop starts (pre-loop segment).
- **Kill rule.** If the rate in the `cut: still reasoning` class is not higher than in base thinking-off **and** in p50c
  completed-correct (difference CI includes 0 or is negative for either control), the prediction has failed and is reported as failed.
  Higher but < 2× counts as "direction right, effect size smaller than predicted".

## Data (no new generation needed)

- **p50c / base, thinking on:** Astra's own first-sample logs, `astra_logs/exp8_math/eval/{p50c,unedited}/parts/{math500,gsm8k}_s*/`,
  200 + 200 problems, one sample each, 32k cap. These are the logs the failure-class table (`notes/failure_classes.md`) was computed
  from; classes are recomputed with the identical `classify()` rule.
  - Note: the task description mentions the 16k screening run. Its per-sample texts are not available locally. The 12% class itself
    comes from these 32k logs, so they are the right object.
- **base / p50c, thinking off:** `/scratch/yang0531/blf_data/nothink_runpod/{base,p50c}_nothink_runpod.jsonl`. These are the
  screening runs: GSM8K + MATH-500, 200 problems × 2 samples, 16k cap.
- **base pre-loop:** base thinking-on text truncated at the start of its verbatim loop, i.e. the first word position of any
  32-word n-gram that occurs ≥ 3 times in the text. Texts without such an n-gram are used whole.

## Detector (fixed before looking)

1. **Text unit.** Thinking-on: the text before the first `</think>`, or the whole text if `</think>` is absent. Thinking-off: the
   whole text.
2. **Segments.** Split on blank lines. Merge consecutive paragraphs until a segment has ≥ 40 words. Split any segment of > 160 words
   into 80-word chunks, merging a last chunk of < 40 words into the previous one. Drop empty segments.
3. **Similarity (primary).** Cosine of `sentence-transformers/all-MiniLM-L6-v2` embeddings: mean pooling, L2-normalised, input
   truncated at 256 word-pieces.
   **Similarity (secondary, lexical).** Jaccard of the sets of lower-cased word tokens (`[a-z]+`, numbers, `\commands`), with
   thresholds 0.5 and 0.6 (not calibrated).
4. **Not verbatim.** A pair (i, j) counts as verbatim if the two segments share any **32-token** n-gram (Qwen3.5 tokenizer). This is
   the same unit as test 8's loop detector. A strict variant also excludes pairs that share any 12-word n-gram ("near-verbatim").
5. **Paraphrase repeat (PR).** Segment i is a PR if some earlier, non-adjacent segment j ∈ [i−11, i−2] has similarity ≥ τ and the
   pair is not verbatim. A fixed look-back of 10 segments keeps the rate independent of text length.
   - **PR rate of a text** = PRs / eligible segments, where eligible means i ≥ 11 (a full look-back window). Texts with < 12
     segments have no eligible segments and are reported as n/a; the n is reported.
   - **Secondary metrics:**
     - PRs per 1k tokens with unbounded look-back (all j ≤ i−2);
     - position-matched PR rate, using only segments that start within the first 2,000 words of the text;
     - verbatim-repeat rate (the same window, pairs that do share a 32-token n-gram).
6. **Cycling text** = PR rate ≥ 0.30.
7. **Threshold τ (embedding), calibration.**
   - Draw 60 segment pairs (j ∈ [i−11, i−2], non-verbatim), stratified over cosine bins [0.5, 0.6), [0.6, 0.7), [0.7, 0.8),
     [0.8, 0.9), [0.9, 1.0], from all groups pooled. Group labels are hidden while reading.
   - Hand-label each pair: "same content" means the same step, claim or check is restated, or "different".
   - τ = the lower edge of the lowest bin where ≥ 80% of pairs are "same content". A robustness threshold τ + 0.05 is also reported.
   - The calibration labels are saved in `notes/expB_calibration.md`.

## Classes (p50c / base thinking-on; `classify()` from `scripts/failure_classes_from_logs.py`)

`correct`; `finished: wrong answer`; `finished: no \boxed{}`; `cut: degenerate whitespace`; `cut: verbatim loop`;
`cut: repetitive (paraphrase)` (Astra's `repetitive_end` rule); `cut: still reasoning` (= the 12% class).

---
(Results below this line were written after the analysis.)
