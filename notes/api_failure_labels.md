# LLM-labelled failure types of p50c outputs (branch `yang/api`)

Complements Exp A (`yang/expA`, logprob-based doubt) and Exp B (`yang/expB`, detector-based cycling). This note gives only the
**LLM-label view**; it does not re-implement their detectors.

## Data and sampling (fixed before labelling; seed 0; sample k = 0 of every arm)
| set | file | selection | n |
|---|---|---|---|
| p50c thinking on (16k cap) | `blf_data/sidefx/screen/screen_p50c_n200x1.jsonl` | all 400 (200 GSM8K + 200 MATH-500) | 400 |
| base thinking off | `blf_data/nothink_runpod/base_nothink_runpod.jsonl` | k = 0: **all failures** + 60 random correct | ≈ 246 |
| p50c thinking off | `blf_data/nothink_runpod/p50c_nothink_runpod.jsonl` | k = 0: 100 random failures + 40 random correct | 140 |
| base thinking on | `blf_data/sidefx/screen/screen_base_n100x1.jsonl` | 60 random | 60 |
| p50chook, clampp5, loopsel, clampmean, a3 (thinking on) | `blf_data/sidefx/screen/screen_*` (k = 0) | 60 random each | 300 |
Failure = grader `correct == False` (Astra's grader). Shares are reported within the sampled stratum (failures / correct), never pooled
across strata with different sampling rates.

## Text shown to the labeller (fixed scheme)
1. Runs of ≥ 200 whitespace characters are collapsed to their first 40 characters + `⟨N whitespace characters omitted⟩`.
2. If the result is ≤ 36,000 characters (≈ 11.6k tokens at the measured 3.1 chars/token), it is shown in full.
3. Otherwise: the first 12,000 characters, then 4 middle chunks of 3,000 characters centred at 1/5, 2/5, 3/5, 4/5 of the
   remaining middle span, then the last 12,000 characters, separated by `⟨… N characters omitted …⟩` markers (36k chars total).
The labeller also gets: the problem, the gold answer, the grader verdict, whether the generation hit the 16k-token cap, and the
thinking mode.

## Rubric (fixed before labelling)
JSON output, one object:
- `final_answer_reached` (bool): the output commits to a definitive final answer to the problem (boxed, or an explicit
  "final answer is …" that ends the solution). Hitting the cap before committing → false.
- `main_type` (one code; the dominant reason the output is not a correct finished solution):
  - `correct` — grader says correct (always use this when the grader says correct).
  - `reasoning_error` — finished (or nearly) with a wrong answer because of an arithmetic / algebra / logic mistake or a single
    plain misreading, *without* extended doubting of the problem statement.
  - `over_doubt_wording` — the model repeatedly questions or re-interprets **what the problem asks** (ambiguity, "maybe it means…",
    "is this a trick / distractor?", "maybe there is a typo"), and this leads to a wrong answer or keeps it from finishing.
  - `endless_recheck` — the model already has the **correct** answer and keeps re-verifying the *computation* without finishing.
  - `paraphrase_cycling` — the same content / argument is restated over and over **in new words** with no progress
    (not verbatim, not mainly about the wording of the problem).
  - `verbatim_loop` — the same passage repeats (near-)verbatim until the cap.
  - `degenerate` — whitespace / runaway indentation / garbage tokens dominate the end.
  - `truncated_progress` — cut at the cap while still making genuine forward progress on a long solution.
  - `other`.
- `also_present` (list of codes): other failure patterns that clearly occur (not the main one).
- `wording_doubt_present` (bool): the model questions / re-interprets the wording of the problem at least twice anywhere.
- `first_answer` (string or null): the first value the model states **as its answer to the question** (not an intermediate quantity).
- `first_answer_status`: `correct` / `wrong` / `none` (never states one) / `not_visible` (likely in an omitted part).
- `doubt_after_first`: `yes` if after the first stated answer the model questions it or the problem interpretation and re-opens
  the work (beyond one brief confirmation); `no` if it confirms briefly and wraps up; `na` if no first answer.
- `doubt_target`: `wording` / `computation` / `both` / `none`.
- `evidence`: a ≤ 30-word quote supporting `main_type`.

## Pre-registered predictions and kill rules (written 2026-11-05 before any labelling call; not edited afterwards)
- **F1 (Q1).** `over_doubt_wording` is a larger share of p50c thinking-on failures than of base thinking-off failures:
  difference ≥ **10 pt**, and the problem-level bootstrap 95 % CI excludes 0. Same direction for `wording_doubt_present`.
- **F2 (Q2).** Among p50c thinking-on outputs cut at the cap that are not `verbatim_loop` / `degenerate`, `paraphrase_cycling`
  (main or also-present) occurs in **≥ 30 %**.
- **F3 (Q3).** For base 0.8B (thinking off, and thinking on where visible), doubt after the first stated answer is only weakly
  related to its correctness: |P(doubt | first correct) − P(doubt | first wrong)| **< 0.15**.
  (For p50c the same quantity is reported without a prediction.)
- **Kill / validity rules.**
  - K-pilot: label 20 items (10 p50c + 10 base thinking-off), read every label against the raw text; scale up only if the
    labels are sensible and the JSON parses.
  - K-hand: my hand check of 20 labels (stratified over arms) must agree on `main_type` in **≥ 14 / 20**. If not, revise the
    rubric once (logged here), re-label, and re-check. If it still fails, only the coarse fields (`final_answer_reached`,
    `first_answer_status`) are reported.
  - K-kappa: Cohen's κ (Sonnet vs gpt-oss-120b, 50 items) on `main_type` < **0.4** → the type shares are reported as one
    labeller's opinion with that caveat; κ is also reported for `wording_doubt_present` and `doubt_after_first`.
  - Parse failures > 5 % → fix the prompt/parser before using the run.
- Non-Anthropic labeller: GPT-5.x / Gemini are 403 on this gateway project (see `api_judge_crossfamily.md`); `gpt-oss-120b` is used.

## Results
(to be filled after the runs)
