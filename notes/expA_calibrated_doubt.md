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
