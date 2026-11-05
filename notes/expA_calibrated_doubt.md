# Experiment A — is the 0.8B model's doubt tied to correctness? (branch `yang/expA`)

## Novelty note (written before any run)

Prior work that is close:
- Zhang et al. 2025, *Reasoning Models Know When They're Right* (arXiv 2504.05419): a probe on hidden states at
  intermediate-answer positions of long CoT predicts correctness of that answer; it is used for early exit. Their point: the
  information is in the state, but the model does not use it.
- Kang et al. 2025, *First Try Matters* (arXiv 2510.08308): most reflections after a candidate answer are confirmatory;
  final accuracy tracks the first answer.
- NoWait (2506.08343), CGRS (2508.05337, certainty-guided reflection suppression), CyclicReflex (2506.11077),
  *Do Thinking Tokens Help or Trap?* (2506.23840, cyclic redundant verification in R1-Distill-1.5B): suppress or schedule
  "Wait"-type tokens to cut overthinking.

What is not covered there, and what this test adds:
1. A **behavioural calibration** measure: the model's own next-token propensity to start a re-check right after it states
   a candidate answer, split by whether that candidate is correct (AUC). Prior work probes the state (Zhang) or counts
   reflections (Kang); neither asks whether the *policy's* doubt is conditioned on correctness, nor compares this across
   sizes of one model family with the same prompt and sampling.
2. A link to a **specific causal direction** (u, the axis the p50c weight edit pushes on) and to a real weight edit:
   the prediction that the edit lowers doubt by a correctness-blind factor, so it cuts loops and needed re-checks alike.
   If the base 0.8B doubts equally after right and wrong answers while a larger Qwen3.5 does not, that is a concrete
   account of why the 0.8B loops ("doubt without a trigger") and why a uniform edit costs accuracy.
A positive result is new in (1)+(2) together; (1) alone is a modest extension of Kang et al.

## Pre-registered predictions and kill rule (written 2026-11-05, before any signal was computed; not edited afterwards)

- **P1** Base 0.8B, own traces: AUC of P(doubt) for separating incorrect (positive) from correct candidates is **≤ 0.6**.
- **P2** A larger Qwen3.5 thinking model (4B; 2B if time), own traces, same problems / prompt / sampling: AUC **≥ 0.7** and
  clearly above the 0.8B (bootstrap CIs do not overlap, or difference ≥ 0.1 with the 95 % CI of the difference above 0).
- **P3** p50c on the base 0.8B prefixes (teacher-forced): P(doubt) is lower at both correct and incorrect points by a similar
  factor (ratio of medians correct vs incorrect within 1.5× of each other); p50c's AUC is within ±0.05 of base's.
- **P4** resid·u15 at hidden_states[16] and resid·u15 / u19 at hidden_states[20] show the same pattern as P(doubt) (base AUC ≤ 0.6;
  p50c shifts both groups down).
- **Kill rule.** If base 0.8B AUC(P(doubt)) > 0.7, the hypothesis "doubt is not tied to correctness" is wrong; reported as such.

Secondary (no pass/fail): the empirical re-check rate in the real trace by correctness; AUC of the ratio
P(doubt)/(P(doubt)+P(wrapup)); cross-model teacher forcing (0.8B on the 4B's prefixes and vice versa), to separate "prefix" from
"model".

## Definitions (fixed before results)

- **Data.** Thinking-on, first sample per problem, GSM8K (200) + MATH-500 (200), same ids as Astra's logs.
  Base 0.8B and p50c traces: `astra_logs/exp8_math/eval/{unedited,p50c}`. Larger model: generated here with vLLM,
  same chat template (`enable_thinking=True`), same user turn (problem + "\nPlease reason step by step, and put your final
  answer within \boxed{}."), Astra's THINK35 sampling (T 0.6, top-p 0.95, top-k 20, presence 1.5, seed 0), cap 16,384 tokens
  (Astra used 32,768; candidates occur early, so the cap only affects the completion statistic).
- **Candidate-answer point.** Inside the thinking part (before `</think>`), the earliest match of
  (a) `\boxed{X}` with non-empty X, or (b) `(final) answer is / Final Answer: / Answer:` followed on the same line by a value
  containing a digit or a LaTeX command. Occurrences that restate the format instruction (`\boxed{}` empty, "answer in \boxed")
  are not matches. The candidate value X is graded against gold with Astra's `is_correct` (math-verify); for (b), if the
  cleaned value fails, its last number is tried. Primary analysis: the **first** candidate per trace (one point per trace,
  independent). Secondary: up to the first 3 distinct-position candidates.
- **Measurement position.** Teacher-forced on the trace prefix up to the end of the line that holds the candidate, plus the
  next line's leading whitespace / list markup (`*`, `-`, digits+`.`, `**`) as it appears in the real trace; the next token
  is then the first content word of the next line. (If the trace ends there, the point is dropped.)
- **Token sets** (a vocabulary token belongs to a set if its decoded string, stripped of whitespace and lower-cased, is in it):
  DOUBT = {wait, but, hmm, actually, however, maybe, perhaps, double, verify, check, hold, alternatively, re, is, did, no}
  WRAPUP = {so, therefore, thus, final, the answer token `</think>`, `**`, `\boxed`, answer, done, conclusion, okay, ok}
  plus `\n\n`-type tokens (pure newline tokens).
  AMBIG = {let} (" Let me check" vs " Let me write the final answer"); reported separately, in neither set in the primary.
  Signals: P(doubt) = Σ p over DOUBT; P(wrap) = Σ p over WRAPUP; ratio = P(doubt)/(P(doubt)+P(wrap)).
- **resid·u.** h·u at the same position, hidden_states[16] with u15 and hidden_states[20] with u15 and u19 (`shared/u_L*.npy`),
  forward hooks/`output_hidden_states` as in tests 2/8. Only for 0.8B models (u is 0.8B-specific).
- **Real re-check.** The next 300 characters after the candidate line contain (case-insensitive) one of: wait, double-check,
  double check, verify, re-check, recheck, re-read, check, hmm, actually, but.
- **AUC.** Positive class = incorrect candidate; AUC of the signal (higher = more doubt). 95 % CI by 2,000-sample bootstrap over
  points. Same extraction, prompt, tokenisation and scoring code for every arm/model.

### Amendment 1 (2026-11-05, before any signal was computed; changes data and detector only, not predictions)
- On Astra's logs the first-candidate labels are too unbalanced for an AUC: base 0.8B has a candidate in 123 of 400 traces, of which
  only 18 are incorrect. **Primary data is therefore a fresh generation with one harness for every model** (`scripts/expA_gen.py`):
  GSM8K = Astra's 200 ids + the next 300 unused test ids (500), MATH-500 all 500; vLLM 0.31, THINK35 sampling, cap 12,288 tokens,
  one sample. Models: base 0.8B, p50c, Qwen3.5-2B, Qwen3.5-4B. Astra's own logs (base, p50c) are measured too, as a secondary replication.
- Detector refinement from eyeballing 24 base detections (labels only, no signal): phrase matches are dropped when the same line,
  before the phrase, contains if / whether / scenario / assume / suppose / unless / could / would / might / maybe / recall / similar /
  typical / usually / expected or a "?", or when the value contains "?" ("is there a scenario where the answer is 30 miles?",
  "if the answer is 7, let's see"). One candidate per line.
- Measurement position made exact at the token level: the real trace is tokenised and the next-token distribution is read at the
  token that covers the first content character of the next line (so the prefix ends at a real token boundary).

### Pilot (job 7293981, L40S, 0.28 GPU-h; 20 problems per model, plus 60 of Astra's base traces) — what reading the raw rows showed
- Generation works for base 0.8B and Qwen3.5-4B with the same template (`<think>\n` opened in both; only the template default differs).
  Base 0.8B: 18/20 cut at 12,288 tokens; 4B: 2/20 cut, mean 4.6k tokens. 4B writes in the same "Here's a thinking process…" style.
- lm_head(last_hidden_state) equals the full forward logits (max |diff| 0.0), so the cheaper readout is exact.
- 0.8B candidate points read as real answer statements ("So the answer is 366.", "Final Answer: 240.") and the next real line is
  usually a re-check ("Wait, what if…", "Let's double-check…", "Is there any chance…"). The model put P(Wait/Let/Is…) of 0.2–0.8
  after *correct* answers too — consistent in sign with the hypothesis, n far too small to say more.
- Defects found and fixed before the full run (Amendment 2, labels/positions only):
  (a) a whitespace-only next line made the "next token" a whitespace token (`gsm8k_test_170`): blank / whitespace-only lines are now skipped;
  (b) conditional enumerations "If … ⏎ Then the answer is 25." and a 4B "Example: … Answer: 5." were counted as candidates:
      `then` and `example` added to the exclusion words;
  (c) the fixed DOUBT set misses question-starts that are clearly re-checks (" If", " Could", " What", " Which", " Alternative") and the WRAP
      set misses 4B's wrap-up headers ("Final Polish", "Refining the Output", "Review", "Formatting", "Looks good", "Ready"). The primary
      sets stay as pre-registered; DOUBT_EXT / WRAP_EXT (in `expA_common.py`) are reported as a secondary sensitivity check.
  (d) " Let" is often the single most likely token after a candidate (0.2–0.67) and is split between "Let's double-check" and
      "Let's write out the steps": it stays out of both sets in the primary (as pre-registered), and p_doubt + p_let is reported.

### Amendment 3 (after the Astra-trace arms and the first 500-problem base part were measured; detector only, applied to every arm)
Reading 6 random *incorrect* first candidates of the fresh base run showed 2 of 6 were not answers but descriptions
("So the answer is the coefficient of $x^0$.", "The answer is simply the number of integer points $(a,b)$ such that …"). These
inflate the incorrect class with non-answers. Rule added (`answer_like` in `expA_common.py`): after stripping hedge words
(seems / to be / definitely / just / simply / …), a phrase value with more than 2 alphabetic words outside LaTeX is not a candidate;
the first candidate of a trace is then the earliest answer-like one. This was decided before any AUC on the fresh data was computed.
(Honest note: AUCs on Astra's traces had already been computed with the old rule — they are kept as
`notes/expA_results_astra_preA3.json` and reported next to the post-amendment numbers.)

### Amendment 4 (grading bug found by reading 4B "incorrect" points; applied to every arm by re-grading the stored values)
The first grader called "seems to be $-3/8$" vs gold `-\frac{3}{8}` and "just $\sqrt{53}$" wrong (the hedge words and the `$` stayed in
the string passed to math-verify), and `\boxed{answer}` format placeholders counted as candidates. Fix (`expA_common.py`):
strip hedge words, a short "label:" prefix and `$` before grading, try the right-hand side of "x = 5", then the last number; a candidate
must contain a digit or a LaTeX command. This was found after the first AUCs on the fresh data had been printed (base 0.8B and 4B);
the predictions are not changed, and the pre-fix numbers are listed in the results for transparency.

### Amendment 5 (power for the larger model; before any hard-set data existed)
With the fixed grader the 4B has only ~20 wrong first candidates in GSM8K + MATH-500 (it is right ~96 % of the time), so its AUC CI
is ±0.12. Added a harder held-out set: 400 MATH test problems at Level 5, not in MATH-500 (`scripts/expA_make_mathhard.py` →
`data/expA_mathhard.jsonl`, seed 0), generated with the same harness for base 0.8B and 4B (p50c teacher-forced on the base traces).
Results are reported per set and pooled; the pooled analysis controls for set (dataset dummies), log position and repetition.

---

## Results (written after all runs; the predictions above are unchanged)

**Framing note (from the side-effect test on `yang/sidefx`).** The drop in "accuracy on completed" (0.94 → 0.75) is a selection
effect: on the same problems, accuracy is unchanged. So this test does not explain an accuracy cost of the edit. The question it
answers is the remaining one: **why does the 0.8B loop at all, and is its doubt tied to correctness?**

### Data actually used
| arm (model on traces) | traces | first candidates: correct / incorrect |
|---|---|---|
| base 0.8B on own thinking-on traces (GSM8K 500 + MATH-500 500, cap 12,288; 92 % cut) | `gen_base_p{0,1}` | 246 / 45 |
| p50c teacher-forced on the same base prefixes | same | 246 / 45 |
| p50c on own traces (56 % cut) | `gen_p50c_p{0,1}` | 359 / 129 |
| Qwen3.5-4B on own traces (same 1000 problems, 13 % cut; + 200 MATH L5 at an 8k cap) | `gen_q4b_p{0,1}`, `gen_q4b_hard8k` | 631 / 20 |
| cross: 0.8B on 4B traces, 4B on 0.8B traces, 0.8B on p50c traces; Astra's own logs; base on MATH L5 | see table 2 | |

Qwen3.5-2B was planned but cancelled for budget (its jobs never started); the 4B is the larger-model arm.

### Table 1 — AUC for separating incorrect (positive) from correct first candidate answers (95 % bootstrap CI)

Columns: P(doubt) = next-token mass on the pre-registered DOUBT set; ratio = P(doubt)/(P(doubt)+P(wrap)); ext = broader DOUBT_EXT set
(amendment 2); u15@hs16 etc. = resid·u; "resid" = AUC after regressing out log position, dataset and repetition (OLS residual);
last column = median P(doubt) after correct / incorrect answers.

| arm | n corr | n inc | P(doubt) | ratio | P(doubt) ext | u15 @ hs16 | u15 @ hs20 | u19 @ hs20 | real re-check | P(doubt) resid | u15@hs16 resid | median P(doubt) corr / inc |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **base 0.8B, own traces** | 246 | 45 | **0.649** [0.55, 0.74] | 0.602 [0.51, 0.68] | 0.693 [0.60, 0.78] | **0.805** [0.73, 0.87] | 0.801 [0.74, 0.87] | 0.787 [0.71, 0.86] | 0.525 [0.47, 0.58] | 0.561 | 0.717 | 0.47 / 0.60 |
| p50c on base prefixes | 246 | 45 | 0.663 [0.57, 0.76] | 0.628 [0.54, 0.71] | 0.713 [0.63, 0.80] | 0.602 [0.50, 0.70] | 0.590 [0.51, 0.68] | 0.580 [0.50, 0.66] | 0.525 | 0.577 | 0.546 | 0.44 / 0.58 |
| **Qwen3.5-4B, own traces** | 631 | 20 | **0.520** [0.39, 0.66] | 0.545 [0.42, 0.67] | 0.631 [0.48, 0.77] | — | — | — | 0.474 [0.36, 0.59] | 0.402 | — | 0.15 / 0.16 |
| p50c, own traces | 359 | 129 | 0.590 [0.53, 0.65] | 0.550 [0.49, 0.61] | 0.650 [0.59, 0.71] | 0.593 [0.53, 0.65] | 0.620 [0.57, 0.68] | 0.619 [0.56, 0.67] | 0.497 | 0.526 | 0.541 | 0.47 / 0.56 |

### Table 2 — secondary arms
| arm | n corr | n inc | P(doubt) | P(doubt) ext | u15 @ hs16 | u15@hs16 resid | median P(doubt) corr / inc |
|---|---|---|---|---|---|---|---|
| 0.8B on p50c's traces | 359 | 129 | 0.587 [0.52, 0.65] | 0.640 | **0.821** [0.78, 0.86] | 0.756 | 0.51 / 0.59 |
| 0.8B on 4B's traces | 631 | 20 | 0.529 [0.38, 0.66] | 0.647 | **0.860** [0.76, 0.94] | 0.743 | 0.34 / 0.42 |
| 4B on 0.8B's traces | 246 | 45 | 0.662 [0.56, 0.76] | 0.721 | — | — | 0.36 / 0.56 |
| base 0.8B on MATH L5 (own) | 21 | 11 | 0.602 [0.40, 0.82] | 0.641 | 0.762 [0.55, 0.94] | 0.779 | 0.60 / 0.69 |
| p50c on those prefixes | 21 | 11 | 0.619 | 0.667 | 0.610 | 0.589 | 0.56 / 0.69 |
| base on Astra's base logs | 107 | 11 | 0.483 [0.31, 0.66] | 0.596 | 0.638 [0.46, 0.83] | 0.568 | 0.44 / 0.41 |
| p50c on Astra's base logs | 107 | 11 | 0.429 | 0.572 | 0.384 | 0.325 | 0.42 / 0.35 |
| p50c on Astra's p50c logs | 166 | 86 | 0.651 [0.58, 0.73] | 0.758 | 0.674 [0.59, 0.75] | 0.568 | 0.46 / 0.62 |

Before amendment 3/4 (old detector and grader), the Astra-log arms gave base P(doubt) 0.58, u15@hs16 0.68 (`notes/expA_results_astra_preA3.json`).
Full numbers incl. the first-3-candidates pool, per-dataset AUCs and paired shifts: `notes/expA_results_main.json`, `notes/expA_results_secondary.json`.
Figure: `notes/fig/expA_pdoubt_main.png` (P(doubt) after the first candidate answer, correct vs incorrect, per arm).

### What happens after the answer (real traces; exploratory classification of the next 300 characters)
| arm | after correct: re-reads wording / re-computes / other / stops | after incorrect: wording / compute / other / stops |
|---|---|---|
| base 0.8B | 63 % / 13 % / 6 % / **18 %** | 64 % / 11 % / 11 % / 13 % |
| p50c own | 64 % / 11 % / 4 % / 20 % | 55 % / 12 % / 12 % / 21 % |
| 4B | 48 % / 13 % / 5 % / **35 %** | 45 % / 10 % / 5 % / 40 % |

The 0.8B re-checks after 82 % of its *correct* first answers, and the dominant re-check is re-reading the problem's wording
("Wait, what if 'reduced by 30 %' refers to…", "Is there any chance 'total' means…"), the same pattern the side-effect test found in
p50c's wrong-but-completed answers. The 4B stops after a correct answer about twice as often (35 % vs 18 %).

### Outcome per prediction
- **P1 (base AUC ≤ 0.6): FAIL, narrowly.** AUC 0.649 [0.55, 0.74]. After controlling for position, dataset and repetition it is
  0.56 [0.46, 0.66]. The kill rule (> 0.7) is not triggered. Reading: the 0.8B's next-token doubt is only weakly tied to correctness.
  It doubts a lot after correct answers too (median P(doubt) 0.47 vs 0.60 after wrong ones).
- **P2 (4B AUC ≥ 0.7 and clearly above the 0.8B): FAIL.** 4B AUC 0.52 [0.39, 0.66] on only 20 wrong first answers. The 4B reading the
  0.8B's prefixes gives 0.66, the same as the 0.8B itself. The larger model is **not more calibrated in P(doubt)**; it doubts
  **less overall** (median 0.15 vs 0.47 after correct answers) and stops more often. Caveat: the 4B is rarely wrong, so its CI is wide.
- **P3 (p50c lowers P(doubt) by a similar factor; AUC within ±0.05): PASS, but the effect is tiny.** On the same prefixes p50c changes
  median P(doubt) by ×0.94 (correct) and ×0.92 (incorrect); AUC 0.663 vs 0.649. At candidate-answer points the edit barely changes the
  doubt-token distribution. Its effect is at loop positions (tests 2/5), not at answer points.
- **P4 (resid·u shows the same pattern as P(doubt)): FAIL, in an informative way.**
  - resid·u is **much better tied to correctness than P(doubt)**: AUC 0.80 [0.73, 0.87] at hs16, and 0.72 after the position, dataset
    and repetition controls. It holds per dataset (GSM8K 0.77, MATH-500 0.88). The same 0.8B reading p50c's traces gives 0.82, and
    reading the 4B's traces gives 0.86.
  - The edit does **not** shift both groups equally. Median Δ(resid·u15 @ hs16) is −0.13 at correct and −0.37 at incorrect points.
    The correctness signal in u drops from AUC 0.80 to 0.60.
  - Across points, u and P(doubt) correlate only moderately (Spearman 0.42).

### Follow-up (not pre-registered; asked for after the API labels): thinking on vs off at matched answer points
Base 0.8B, thinking-off generations from test 8 (Astra's 200 GSM8K + 200 MATH-500 ids). The same text is teacher-forced under the
no-think template and under the thinking template (`<think>` open). In the no-think traces the first answer is usually the final
`\boxed{}`, so for those points the measurement is after the answer + a blank line (`EXPA_PAD_END`).

| same text, template | n corr / inc | AUC P(doubt) | median P(doubt) corr / inc | real re-check in the no-think trace: corr / inc |
|---|---|---|---|---|
| thinking-off traces, **no-think template** (own mode) | 210 / 161 | **0.72** [0.67, 0.77] | **0.005 / 0.023** | 2 % / 21 % |
| same thinking-off text, **thinking template** | 210 / 161 | 0.59 [0.52, 0.65] | **0.38 / 0.42** | (same text) |
| only points where the no-think trace continued (no padding), no-think template | 21 / 60 | 0.76 [0.64, 0.87] | 0.014 / 0.163 | |
| same, thinking template | 21 / 60 | 0.59 [0.44, 0.73] | 0.35 / 0.42 | |
| thinking-on traces, no-think template | 246 / 45 | 0.56 | 0.46 / 0.64 | |
| thinking-on traces, thinking template | 246 / 45 | 0.65 | 0.47 / 0.60 | |

The logprobs agree with the API labels (thinking off: 15 % vs 52 % doubt; thinking on: 95–100 %).
- With thinking off, the 0.8B's doubt **is** tied to correctness (AUC 0.72–0.76, and it rarely doubts a correct answer).
- Putting the identical text inside `<think>` multiplies P(doubt) by about 70× after correct answers and 18× after wrong ones, and
  calibration drops to 0.59.
- Once the context is thinking-style text, the template matters little: the last two rows are within noise. So the doubt is carried
  by the thinking-mode context, not by the template token alone.
- resid·u at hs16 is *not* raised by the thinking template on the no-think text. It is slightly lower: median 0.16 (thinking) vs 0.23 (no-think) at correct points, 0.22 vs 0.25 at incorrect ones.
  Hypothesis (untested): the thinking-mode doubt prior is added downstream of, or orthogonal to, u.

**Sharpened statement.** "The 0.8B cannot calibrate doubt" is wrong as stated.
- In no-think mode its doubt is calibrated.
- In thinking mode a large, nearly correctness-independent doubt prior sits on top. After a correct answer the next line starts a
  re-check about half the time (P(doubt) ≈ 0.47), and in the real traces 82 % of correct first answers are followed by a re-check,
  mostly re-reading the wording.
- Meanwhile the model's own state does separate right from wrong (resid·u AUC 0.80), so the information exists but is only weakly
  read out into the decision to re-check. This matches Zhang et al. 2025 ("models know when they're right") at the level of one
  named direction.
- The 4B differs mainly by a lower doubt prior (it stops after correct answers twice as often), not by sharper calibration.

Untested hypotheses and cheap tests:
- (H1) Looping = this doubt prior + re-reading → copying.
  Test: in thinking mode, subtract a constant from the DOUBT logits only at candidate-answer points (not elsewhere) and measure
  completion and accuracy. This is a targeted version of NoWait.
- (H2) Reading out u better would give calibrated stopping.
  Test: gate the doubt logits on resid·u at answer points, e.g. suppress doubt when u is below the median of correct points, and
  compare against uniform suppression.

### Caveats
- Candidate detection is a hand-written heuristic. It was refined four times (amendments 1–4), and two of those changes (3, 4) were made
  after some AUCs on the fresh data had been printed. They fix clear errors (non-answers counted as answers, hedged values mis-graded)
  and are applied to every arm alike, but they are post-hoc. Pre-amendment numbers are kept in `expA_results_astra_preA3.json` and
  the log of interim runs (`/scratch/yang0531/blf_data/expA/interim/`).
- The 0.8B has a candidate in only ~29 % of thinking-on traces (12k cap, most traces loop). The analysed points are a selected subset.
- Wrong first candidates are few: 45 for the 0.8B and 20 for the 4B. The 4B's AUC CI is ±0.13. The hard MATH L5 set added only 82
  4B points (8k cap; the 12k run timed out).
- Position is a confound: wrong candidates come later (AUC 0.67 for log position). The u result survives a linear control
  (0.72), but a nonlinear confound is not excluded.
- Astra's own base logs (only 11 wrong first answers after the fixes) show a weaker u signal (0.64, CI 0.46–0.83). This does not
  contradict the fresh data, but it is not a replication either.
- In the no-think template, the padded measurement point (after `\boxed{}` + blank line) is off-distribution: the model would
  normally stop right after the box. The un-padded subset (n = 81) gives the same picture.
- One sample per problem, one seed. The token sets are hand-picked (pre-registered; the extended sets give the same ordering).

### Compute
Killarney L40S/H100, one GPU per job: **≈ 2.98 GPU-h**. That is pilot 0.28; generation + measurement 0.8B/p50c 0.97; 4B 0.32;
cross 0.13; Astra-log arms 0.12; hard set 0.12 (base) + 0.59 (4B 12k run, **timed out, lost**) + 0.32 (4B 8k run); thinking on/off
follow-up 0.15. Within the 3 GPU-h budget, including the lost run. No API calls.

### Files
- Scripts:
  - `scripts/expA_common.py`: detector, grader, token sets
  - `scripts/expA_gen.py`
  - `scripts/expA_measure.py`
  - `scripts/expA_analyze.py`
  - `scripts/expA_make_mathhard.py`
  - `scripts/expA_job.sh`
- Results:
  - `notes/expA_results_main.json`
  - `notes/expA_results_secondary.json`
  - `notes/expA_results_followup_onoff{,_nopad}.json`
- Figures:
  - `notes/fig/expA_pdoubt_main.png`
  - `notes/fig/expA_pdoubt_secondary.png`
  - `notes/fig/expA_pdoubt_followup_onoff*.png`
- Per-point rows (with context snippets) and generations, not in git: `/scratch/yang0531/blf_data/expA/{gen,meas}_*.jsonl`.
