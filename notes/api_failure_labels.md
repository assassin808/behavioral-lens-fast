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

## Results (filled 2026-11-05; predictions above unchanged)

Labeller: `claude-sonnet-5-5` on all 1,149 sampled outputs (one call each). Scripts: `scripts/api/fail_label.py` (sample, view, rubric,
labelling), `scripts/api/fail_analyze.py` (tables). Files: `notes/api_fail_sample.json`, `notes/api_fail_labels_claude-sonnet-5-5_all.json`,
`…_kappa50.json`, `notes/api_fail_labels_gpt-oss-120b_kappa50.json`, `notes/api_fail_analysis_v1.json`, hand labels `scripts/api/handcheck_labels.json`.

### Validation
- **Pilot (K-pilot).** 20 items (10 p50c thinking-on + 10 base thinking-off), 0 parse failures. I read every label against the raw text:
  all were sensible, so I scaled up.
- **Parsing.** In the full run, 27 / 1,149 (2.3 %) responses had malformed JSON: broken quoting in the last `evidence` field.
  I added a field-wise regex fallback to `fail_label.parse` and applied it to the stored responses, with no new calls. After that,
  0 items are unparseable. This is a parser change made after seeing outputs, and it is logged here.
- **Hand check (K-hand).** I took 20 failures stratified over arms (8 p50c thinking-on, 6 base thinking-off, 2 p50c thinking-off,
  2 base thinking-on, 2 other arms; fixed seed; `scripts/api/handcheck_show.py`). I read the head, middle and tail of each raw
  output and wrote my `main_type` before looking at Sonnet's (`scripts/api/handcheck_labels.json`).
  - **Agreement: 17 / 20** (pre-registered bar: ≥ 14). **Pass.**
  - The 3 disagreements:
    - A p50c output with no correct answer yet that cycles through polynomial division. Sonnet: `endless_recheck`; me: `paraphrase_cycling`.
    - An a3 output repeating a near-identical block. Sonnet: `verbatim_loop`; me: `paraphrase_cycling`.
    - A clampmean output. Sonnet: `verbatim_loop`; me: `over_doubt_wording`.
  - All 3 sit on the cycling / loop / recheck borders. None involves `reasoning_error` vs `over_doubt_wording`.
  - I did not hand-check `also_present`.
- **Cross-family agreement (K-kappa).** gpt-oss-120b labelled the same 50 stratified items with the same prompt. (GPT-5.x and Gemini
  are 403 on this gateway project.)

| field | κ (all 50) | raw agreement | κ (35 failures only) |
|---|---|---|---|
| main_type | **0.70** | 0.76 | **0.60** (agreement 0.69) |
| main_type, coarse (doubt-type / loop / error / other) | – | – | 0.66 (0.77) |
| wording_doubt_present | 0.79 | 0.90 | 0.70 |
| first_answer_status | 0.77 | 0.86 | 0.74 |
| final_answer_reached | 0.74 | 0.88 | 0.67 |
| doubt_after_first | 0.53 | 0.70 | 0.57 |

κ on main_type is ≥ 0.4, so K-kappa passes. The main confusion is **Sonnet `verbatim_loop` vs gpt-oss `over_doubt_wording`**
(4 / 35): loops whose repeated content is itself a wording doubt. `doubt_after_first` is the least reliable field (κ 0.53).

### Q1: what p50c's failures are (failures only; % of main_type)

| arm | n fail | cut | error | over-doubt wording | re-check correct | paraphrase cycling | verbatim loop | degenerate | truncated | other | wording doubt present (any) | final answer reached |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **p50c thinking on** | 207 | 0.82 | 5 | **42** | 27 | 10 | 13 | 3 | 0 | 0 | **0.91** | 0.18 |
| **base thinking off** | 189 | 0.17 | **69** | 20 | 2 | 2 | 6 | 1 | 0 | 1 | 0.41 | 0.84 |
| p50c thinking off | 100 | 0.15 | 78 | 20 | 0 | 0 | 0 | 0 | 1 | 1 | 0.41 | 0.85 |
| base thinking on (random 60) | 57 | 1.00 | 0 | 2 | 2 | 0 | **96** | 0 | 0 | 0 | 0.96 | 0.00 |
| p50chook (random 60) | 30 | 0.80 | 7 | 53 | 17 | 10 | 10 | 0 | 3 | 0 | 0.77 | 0.20 |
| clampp5 (random 60) | 33 | 0.91 | 0 | 42 | 33 | 6 | 18 | 0 | 0 | 0 | 0.94 | 0.09 |
| loopsel (random 60) | 37 | 0.95 | 0 | 19 | 5 | 3 | 70 | 3 | 0 | 0 | 0.92 | 0.05 |
| clampmean (random 60) | 41 | 0.95 | 0 | 17 | 17 | 10 | 56 | 0 | 0 | 0 | 1.00 | 0.05 |
| a3 (random 60) | 40 | 0.97 | 5 | 15 | 12 | 10 | 57 | 0 | 0 | 0 | 0.82 | 0.03 |

Split into wrong-but-completed vs cut:

| subset | n | error | over-doubt wording | re-check correct | cycling | verbatim loop | degenerate |
|---|---|---|---|---|---|---|---|
| p50c thinking on, **finished wrong** | 38 | 8 | **87** | 0 | 0 | 5 | 0 |
| p50c thinking on, **cut** | 169 | 5 | 32 | **33** | 12 | 14 | 4 |
| base thinking off, finished wrong | 157 | **80** | 20 | 0 | 0 | 0 | 0 |
| base thinking off, cut | 32 | 16 | 19 | 9 | 12 | 38 | 3 |
| p50c thinking off, finished wrong | 85 | 79 | 20 | 0 | 0 | 0 | 0 |

**F1** (p50c thinking-on − base thinking-off, failures; bootstrap 95 % CI):

| measure | p50c | base off | diff [CI] |
|---|---|---|---|
| main_type = over_doubt_wording | 0.42 | 0.20 | **+0.22 [+0.13, +0.31]** |
| wording_doubt_present | 0.91 | 0.41 | +0.50 [+0.42, +0.58] |
| over_doubt main or also-present | 0.78 | 0.32 | +0.46 [+0.37, +0.55] |
| finished-wrong only, main = over-doubt | 0.87 (n 38) | 0.20 (n 157) | +0.66 [+0.54, +0.78] |
| GSM8K only, main = over-doubt | 0.87 (n 70) | 0.22 (n 93) | +0.66 [+0.55, +0.76] |
| MATH-500 only, main = over-doubt | 0.19 (n 137) | 0.19 (n 96) | **+0.00 [−0.10, +0.10]** |
| MATH-500 only, wording_doubt_present | 0.86 | 0.47 | +0.39 [+0.28, +0.51] |

Reading:
- Base thinking-off fails mainly by plain mistakes (69 %). p50c thinking-on almost never does (5 %).
- p50c's wrong-but-completed answers are nearly all (87 %) the product of re-interpreting the problem. This quantifies the
  side-effect test's by-eye note ("muffins are not fresh duck eggs").
- p50c's cut outputs split roughly into thirds:
  - re-interpreting the wording (32 %);
  - re-checking an answer that is already correct (33 %);
  - cycling or loops (26 %).
- The effect is GSM8K-driven. On MATH-500, wording doubt is *present* more often, but it is no more often the *main* failure;
  there the dominant p50c failure is re-checking (see json).
- p50c thinking-off looks like base thinking-off (78 % errors, 20 % over-doubt). So the over-doubt profile belongs to p50c *with
  thinking*, not to the weights alone.
- Hook arms:
  - The over-doubt profile (≥ 42 %) belongs to the arms that keep loops broken (p50chook, clampp5).
  - The arms that break fewer loops (loopsel, clampmean, a3) fail mainly by verbatim loops (56–70 %). This matches their loop_end in `sidefx.md`.

### Q2: paraphrase cycling in long unfinished outputs (cut at 16k, main_type ≠ verbatim_loop / degenerate)

| arm | n | cycling as main type | cycling main or also-present [95 % CI] |
|---|---|---|---|
| **p50c thinking on** | 138 | 0.14 | **0.90 [0.85, 0.94]** |
| p50chook | 24 | 0.13 | 0.88 |
| clampp5 | 24 | 0.08 | 0.96 |
| clampmean | 16 | 0.25 | 0.94 |
| a3 | 17 | 0.24 | 0.94 |
| base thinking off | 19 | 0.21 | 0.58 |
| p50c thinking off | 15 | 0.00 | 0.40 |

- In the LLM-label view, almost every unfinished non-loop p50c output contains paraphrase cycling (90 %).
- It is rarely the *main* label (14 %): Sonnet attributes the dominant failure to what is being cycled over, either the wording
  (over-doubt) or a correct answer (re-check).
- So cycling is the *form* and doubt is the *content*.
- This is the label view only. It complements Exp B's detector-based measurement (`yang/expB`) and does not replace it.
  `also_present` was not hand-checked.

### Q3: doubt after the first stated answer vs its correctness
(Weighted by stratum for base / p50c thinking-off; thinking-on arms are unweighted.)

| arm | n with first answer (correct / wrong) | P(doubt \| first correct) | P(doubt \| first wrong) | diff [95 % CI] | P(final correct \| first correct) |
|---|---|---|---|---|---|
| **base thinking off** | 60 / 163 | **0.15** | **0.52** | **−0.37 [−0.49, −0.24]** | 0.90 |
| p50c thinking off | 43 / 88 | 0.30 | 0.51 | −0.22 [−0.39, −0.03] | 0.81 |
| base thinking on | 39 / 8 | 0.97 | 1.00 | −0.03 [−0.09, 0] | **0.08** |
| **p50c thinking on** | 244 / 89 | **0.95** | 1.00 | −0.05 [−0.07, −0.02] | **0.72** |
| other hook arms (60 each) | 30–39 / 10–15 | 0.97–1.00 | 0.93–1.00 | −0.03 … +0.04 | 0.56–0.72 |

- With thinking **on**, doubt after a first answer is near-universal (95–100 %) whether that answer is right or wrong.
  - The model doubts its correct first answer as much as a wrong one.
  - In p50c, 28 % of problems whose first stated answer was correct do not end correct. Mostly they are cut while still doubting.
- With thinking **off**, doubt is selective: base doubts a wrong first answer 3.5× as often as a correct one.
- This is the label-based view only; Exp A (`yang/expA`) has the logprob-based view.

### Excerpts (all read by me in the raw text during the hand check)
1. p50c thinking-on, GSM8K 0 (gold 18, answered 26, finished). The model judges the muffin eggs irrelevant and re-reads
   "remainder" for 5k tokens: *"Wait, is it possible that 'remainder' means 'remainder' in the sense of a mathematical remainder?
   i.e., 16 % 3 = 1."* … *"The muffins are baked, so presumably they are not 'fresh' eggs."* → `over_doubt_wording`.
2. p50c thinking-on, MATH number_theory/598 (cut). It already has the correct 0.0000672 and keeps going: *"Or maybe $21$ is a typo for
   $21 \cdot 5^7$? Or maybe $5^7$ is a typo for $5^7$?"* → `endless_recheck` with wording doubt.
3. p50c thinking-on, GSM8K 107 (cut). The "a few episodes" wording is restated in slightly varied blocks until the cap: *"So the number
   of 30-minute episodes is 1.5. But the question asks for the number of 30-minute episodes. … However, usually 'a few' means at
   least 2."* → `paraphrase_cycling`.
4. base thinking-off, MATH prealgebra/1840 (finished wrong, 65 vs 78). A plain hallucinated premise with no doubting: *"Since the
   problem states they sell 'five sets' of items … c = 5 × 3 = 15"* → `reasoning_error`.
5. base thinking-on, GSM8K 27 (cut). The same wording doubt, but as a verbatim loop: *"What if 'serving' means 1 serving = 1/15 of a
   carton?"* × hundreds → `verbatim_loop`.

### Outcome per prediction

| prediction | outcome |
|---|---|
| **F1** over-doubt share larger in p50c thinking-on failures than in base thinking-off, ≥ 10 pt, CI excludes 0 | **pass**: +22 pt [+13, +31]. Driven by GSM8K (+66 pt); MATH-500 shows **no** difference in main type (+0 pt [−10, +10]), though wording doubt is more often present there (+39 pt). |
| **F2** paraphrase cycling (main or also) in ≥ 30 % of p50c cut non-loop outputs | **pass**: 90 % [85, 94]. As main type only 14 %. |
| **F3** base 0.8B: \|P(doubt \| first correct) − P(doubt \| first wrong)\| < 0.15 | **fail for thinking off** (−0.37 [−0.49, −0.24]: doubt *is* related to correctness). **Pass for thinking on** (−0.03), but only because doubt is at ceiling (97–100 %), and with just 8 wrong-first items. |
| K-hand ≥ 14 / 20 | pass (17 / 20) |
| K-kappa ≥ 0.4 on main_type | pass (0.70 all; 0.60 failures only) |
| parse failures ≤ 5 % | pass (2.3 %, recovered by the field-wise parser) |

### Caveats
- The labeller sees the gold answer and the grader verdict. That is intended: it makes the first-answer correctness judgeable. It is
  blind to the arm (only the thinking mode is shown), but thinking-on p50c and base texts look different (loops vs none).
- Long texts are shown as head + 4 middle chunks + tail (36k characters).
  - `first_answer_status = not_visible` never occurred. The first answer can still sit in an omitted part, and the labeller then
    reports a later one as "first".
  - Answer-phase details in the omitted middle are not seen.
- `paraphrase_cycling` in `also_present` is generous and was not hand-checked. Main-type cycling vs re-check vs verbatim loop is
  where the hand check and κ disagree most.
- Single labeller (Sonnet) for the tables. gpt-oss-120b is a weaker model than Sonnet, so κ = 0.6–0.7 bounds reliability from below.
- Sampling: thinking-off arms are stratified (all / 100 failures + 60 / 40 correct). Failure tables use failures only. Q3 weights the
  strata. Hook arms have only 60 samples (30–41 failures).
- One sample (k = 0) per problem, first 200 GSM8K + first 200 MATH-500.

### API calls
- Sonnet: 1,149 (the 20 pilot items are included and were reused from the cache; the 50 κ items were cache hits).
- gpt-oss-120b: 50.
- **Total 1,199 calls.**
