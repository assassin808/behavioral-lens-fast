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

## Results (2026-11-05)

**Bottom line: the pre-registered prediction failed, and the kill rule fired.**
- With the primary detector (embedding cosine, calibrated τ = 0.70), the `cut: still reasoning` class is no more paraphrase-cyclic
  than p50c's correctly finished generations: 0.84 vs 0.85, difference CI [−0.07, +0.03]. It is also not above base thinking-on
  text before its loops start: 0.84 vs 0.90.
- The embedding detector barely separates anything. At τ = 0.70, 84–92% of eligible segments in *every* thinking-on class restate
  an earlier segment, and so do 66–69% in thinking-off text.
- The secondary, lexical detector (word-set Jaccard ≥ 0.5, no shared 32-token n-gram) does separate the target class from the
  controls, by ≥ 2×. An exploratory, not pre-registered, position-matched check shows this is mostly a late-in-the-text effect
  shared by all long p50c thinking texts, not a property of the 12% class.

### Setup as run
- Texts:

  | group | texts | of which ≥ 12 segments (eligible for the primary rate) |
  |---|---|---|
  | p50c thinking-on (32k cap) | 400 | 382 |
  | base thinking-on pre-loop | 400 | 369 |
  | base thinking-off (16k cap, 2 samples) | 800 | 264 |
  | p50c thinking-off (16k cap, 2 samples) | 800 | 288 |

  Most thinking-off texts (short GSM8K answers) have < 12 segments. So the thinking-off rates describe the longer thinking-off
  texts, as pre-registered.
- The failure classes reproduce `notes/failure_classes.md` exactly. For p50c on MATH-500:

  | class | count |
  |---|---|
  | correct | 92 |
  | finished, no `\boxed{}` | 42 |
  | cut: still reasoning | 24 |
  | finished, wrong answer | 18 |
  | cut: whitespace | 11 |
  | cut: Astra repetitive | 9 |
  | cut: verbatim loop | 4 |

  The 24 `cut: still reasoning` texts are all MATH-500; GSM8K has 0.
- Calibration: `notes/expB_calibration.md`. τ = 0.70 by the pre-registered 80% rule; robustness τ = 0.75.
- Pilot: 20 texts per group, read by eye (segments, pairs, metrics). Records for those 80 texts are byte-identical between the pilot
  build and the parallel full build.
- Shared harness: one segmenter, embedder and n-gram test for all groups (`scripts/expB_paraphrase.py`).

### Table: paraphrase-repeat (PR) rate by group / class

- **PR rate:** fraction of eligible segments that restate one of the 10 non-adjacent preceding segments.
- **cyc:** fraction of texts with PR ≥ 0.30.
- **VB:** fraction of segments that verbatim-share a 32-token n-gram with that window.
- Full tables for every threshold, and the strict 12-word variant, are in
  `/scratch/yang0531/blf_data/expB/tables_tau070.md`. The strict variant changes nothing material.

| group / class | n (elig.) | **emb τ 0.70**: PR | cyc | emb τ 0.75: PR | **lex ≥ 0.5**: PR | cyc | lex ≥ 0.6: PR | VB |
|---|---|---|---|---|---|---|---|---|
| base thinking-off | 800 (264) | 0.69 | 0.92 | 0.53 | 0.03 | 0.02 | 0.01 | 0.14 |
| p50c thinking-off | 800 (288) | 0.66 | 0.95 | 0.46 | 0.01 | 0.00 | 0.00 | 0.06 |
| base thinking-on, pre-loop | 400 (369) | 0.90 | 0.99 | 0.81 | 0.12 | 0.12 | 0.02 | 0.16 |
| p50c on: correct | 214 (197) | 0.85 | 0.98 | 0.74 | 0.08 | 0.06 | 0.02 | 0.12 |
| p50c on: finished, wrong | 71 (71) | 0.92 | 1.00 | 0.86 | 0.24 | 0.39 | 0.08 | 0.26 |
| p50c on: finished, no box | 55 (54) | 0.86 | 1.00 | 0.79 | 0.30 | 0.48 | 0.11 | 0.41 |
| p50c on: cut, whitespace | 20 (20) | 0.90 | 1.00 | 0.79 | 0.12 | 0.05 | 0.03 | 0.08 |
| **p50c on: cut, still reasoning (12%)** | 24 (24) | **0.84** | 1.00 | 0.74 | **0.24** | 0.33 | 0.10 | 0.39 |
| p50c on: cut, Astra repetitive | 9 (9) | 0.77 | 1.00 | 0.68 | 0.27 | 0.56 | 0.11 | 0.46 |
| p50c on: cut, verbatim loop | 7 (7) | 0.49 | 0.86 | 0.42 | 0.19 | 0.29 | 0.12 | 0.63 |
| p50c on: all | 400 (382) | 0.86 | 0.99 | 0.76 | 0.16 | 0.21 | 0.05 | 0.22 |

### Outcome per prediction (predictions above are unchanged)

**P1** (≥ 50% of the class are cycling texts)
- Primary detector: **formally met but uninformative.** 24 / 24 texts at τ 0.70 and at 0.75. But 92% of long base thinking-off
  texts and 98% of p50c correct texts also pass, so the flag does not discriminate.
- Lexical detector: **not met.** 8 / 24 (33%).

**P2** (≥ 2× base thinking-off and ≥ 2× p50c correct, CIs exclude 0): **failed.**

| detector | vs base thinking-off | vs p50c correct |
|---|---|---|
| embedding τ 0.70 | ratio 1.21, CI of difference [+0.09, +0.20] | ratio **0.98**, CI [−0.07, +0.03] |
| embedding τ 0.75 | 1.40 | 0.99, CI [−0.07, +0.06] |

**Kill rule: fired**, since the difference vs p50c correct has a CI that includes 0.
- Lexical detector, secondary: ratios 7.0× vs base thinking-off and 3.2× vs p50c correct, CIs [+0.14, +0.29] and [+0.10, +0.24].
  This would pass, but see the position check below.

**P3** (p50c thinking-on ≥ 1.5× base pre-loop): **failed.**
- Embedding: ratio 0.96 (τ 0.70) and 0.94 (τ 0.75). The CI of the difference is entirely negative.
- Lexical: 1.34× at Jaccard 0.5 (below 1.5×); 2.3× at 0.6, but the absolute rates there are tiny (0.05 vs 0.02).

### Exploratory, not pre-registered: position-matched rates

Segments are bucketed by where they start in the text. Script: `scripts/expB_explore.py`. Output:
`/scratch/yang0531/blf_data/expB/explore_position.json`.

| lexical PR (Jaccard ≥ 0.5) | words 0–2k | 2k–5k | 5k–10k |
|---|---|---|---|
| base thinking-off | 0.03 | 0.03 | 0.05 |
| p50c thinking-off | 0.01 | 0.02 | 0.02 |
| base thinking-on pre-loop (texts that later loop verbatim) | 0.08 | 0.22 | 0.36 |
| p50c on: correct | 0.02 | 0.07 | 0.21 |
| p50c on: finished, wrong | 0.03 | 0.24 | 0.40 |
| p50c on: finished, no box | 0.06 | 0.26 | 0.43 |
| **p50c on: cut, still reasoning** | 0.03 | 0.17 | 0.28 |

- Within any position bucket, the 12% class is **not** the most repetitive. No-box and wrong-answer generations reword more, and
  the class is only modestly above correct generations at the same position.
- The unmatched lexical gap (0.24 vs 0.08) comes mostly from the class being long by construction.
- At matched positions, base thinking-on text *before* its verbatim loop rewords as much as or more than p50c. So the evidence does
  not support "the edit creates paraphrase loops".
- **Hypothesis, untested causally:** reworded re-checking is the normal late-thinking state of this model with or without the edit.
  In base it is followed by verbatim lock-in; p50c blocks the lock-in (test 5/8) but leaves the rewording. A cheap test would be
  the same position-matched rate on base vs p50c continuations from identical prefixes. This is consistent with the mechanism, but
  it is not a p50c-specific effect.

### What a paraphrase cycle looks like (p50c, `cut: still reasoning`; Jaccard ≥ 0.5, no shared 32-token or 12-word n-gram)

1. `test/algebra/2214` (segments 147 → 154, about word 8k). The same inequality check, with new test values.
   - *Earlier:* "So $x = -1$ is not in the solution. So $x > -1$. Wait. Wait, let's check $x = 5$ … So $x > -1$."
   - *Later:* "So $x$ must be greater than $-1$. Wait. Wait. Let's check $x = -0.5$ … So $x$ must be greater than $-1$. Wait.
     Let's check $x = -1.5$ …"
2. `test/counting_and_probability/377` (107 → 115). The same sum is re-added in a different grouping.
   - *Earlier:* "$P(6)$ is $28$. Sum = $78044$ … $16384 + 28672 = 45056$ …"
   - *Later:* "$45056 + 30384 = 75440$ … $78016 + 29 = 78045$. Wait, $P(0)$ is $16384$ …"
3. `test/intermediate_algebra/345` (75 → 82). The same index doubt, restated.
   - *Earlier:* "Wait, I think I might have made a mistake in the indices. Let's check the indices again. $a_{2k-1}, a_{2k}, a_{2k+1}$
     … So $a_{2k}$ is the middle term."
   - *Later:* "Let's assume the middle terms are: For GP: $a_{2k}$ is the middle term … Wait, I need to check the indices again …"
4. `test/intermediate_algebra/1388` (66 → 69). The same expansion of $f(x+y) - f(x) - f(y)$, redone.

These are exactly "re-checking the same thing in other words". The point is that such passages are just as common in p50c's
correct and finished-wrong generations at the same depth.

### Relation to `notes/api_failure_labels.md`
- The API labels report paraphrase cycling in ~90% of capped non-looping p50c outputs, mostly doubt about the wording or re-checks
  of a correct answer.
- **Partial agreement.**
  - The excerpts confirm that kind of content.
  - My embedding detector flags 24 / 24 texts in this class.
  - My lexical detector flags only 33% of texts.
- **Disagreement on specificity.** The same detectors flag most *successful* p50c generations, and base thinking-on text before its
  loops, at similar rates. So "paraphrase cycling" describes p50c's long thinking in general, not this failure class. The API
  labels had no control group.

### Caveats
- One labeller (the agent) calibrated τ and knew the hypothesis, although bins and groups were hidden. The "same content" bar was
  lenient.
- MiniLM is a general-purpose sentence model. On math text, any two segments about the same problem score high, which is the main
  reason the primary detector saturates. A math-aware or LLM-judge similarity might discriminate better; untested.
- The target class has n = 24, all from MATH-500, one sample per problem, and the classes differ by problem difficulty. Thinking-on
  logs have a 32k cap; thinking-off runs have a 16k cap.
- The base pre-loop cut is at the first 32-word n-gram occurring ≥ 3 times. Its last segments sit right before loop onset, which
  may inflate base's late-position rates.

### Compute
- GPU: 1 Killarney `gpubase_interac` job, 5 min 52 s, about **0.1 L40S-h**. The GPU was idle: all partitions are GPU partitions,
  and the work ran on 16 CPUs.
- No generation was needed; Astra's logs and the existing thinking-off runs were reused.
- The pilot and analysis ran on CPU. A first full build on the login node was killed: the cgroup limit is 1.5 CPU, and worker
  processes died. Its partial output was discarded.

### Files
- Code:
  - `scripts/expB_paraphrase.py` (build / report)
  - `scripts/expB_tables.py`
  - `scripts/expB_explore.py`
  - `scripts/expB_job.sh`
- Calibration: `notes/expB_calibration.md`.
- Data (not in git), in `/scratch/yang0531/blf_data/expB/`:
  - `texts_full.jsonl.gz` (segments, pairs, classes)
  - `report_tau070.json`, `tables_tau070.md`
  - `explore_position.json`
  - `calib_pairs_pilot.json`, `calib_labels_pilot.json`
