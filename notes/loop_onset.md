# Loop onset — what comes just before a verbatim loop starts? (branch `yang/onset`)

Model: Qwen3.5-0.8B base (unedited), thinking on. No new generation; existing traces only.

## Question and competing accounts (written before any onset was classified)

- **Impasse account** (Duan et al. 2026, *Circular Reasoning*, arXiv 2601.05693): loops begin at reasoning impasses — the model is stuck
  and has no answer yet.
- **Doubt-after-success** (ours): in this 0.8B model most loops begin *after* the model has already stated a correct answer, which its
  hidden state already marks as correct (Exp A: resid·u15 at hs16 separates wrong from right first answers, AUC 0.80). The model
  re-checks anyway and the copy/induction mechanism takes over.

What is new if positive: prior "overthinking" work (Zhang et al. 2025, 2504.05419; Kang et al. 2025, 2510.08308) shows the state knows
an answer is right and the model keeps reflecting; Duan et al. tie loops to impasses. Nobody has located verbatim-loop onsets relative to
the stated-answer history and graded those answers. A positive result says the 0.8B loops are mainly a failure to *stop*, not a failure
to *solve*, and it predicts that an edit acting on the doubt axis (p50c) fixes loops without needing new reasoning ability.

## Data (fixed)

- **Primary:** Exp A fresh base thinking-on traces, `$BLF_DATA/expA/gen_base_p{0,1}.jsonl` (GSM8K 500 + MATH-500 500, vLLM, Astra
  THINK35 sampling, cap 12,288 tokens).
- **Replication:** Astra's base logs, `astra_logs/exp8_math/eval/unedited/parts/{gsm8k,math500}_s*/` (200 + 200, first sample, cap 32,768).
- **Stratum (reported separately, not in the primary counts):** `gen_base_hard.jsonl` (400 MATH Level 5, same harness as primary).
- **Difficulty:** base thinking-off, `$BLF_DATA/nothink_runpod/base_nothink_runpod.jsonl` (2 samples on Astra's 400 ids).
  "Thinking-off solves" = at least one of the 2 samples correct. Available for all replication traces and the 200 primary GSM8K ids that
  are Astra ids + the 200 MATH-500 ids that are Astra ids.

## Definitions (fixed)

- **Tokens.** Generated text tokenised with the base tokenizer (`add_special_tokens=False`), offsets kept.
- **Loop onset.** The first token of the first 32-token n-gram that occurs ≥ 3 times in the generated tokens; formally
  onset = min over 32-grams g with count(g) ≥ 3 of first_start(g). (Same rule as Exp B `preloop`, at token level as in test 8.)
  A trace "loops" iff an onset exists. Onset char = start offset of that token.
- **Stated answers before onset.** Exp A detector `candidates()` (with amendments 2–4: exclusions, one per line, `answer_like`, the fixed
  grader `grade_value`) run on `text[:onset_char]` with no cap on the number; an answer counts if its whole statement ends before the onset.
  Works across `</think>` (answers after `</think>` count too; where the onset falls — inside thinking or after — is recorded).
- **Onset class.** (a) *after-correct*: last stated answer before onset is correct; (b) *after-wrong*: it is incorrect; (c) *no answer
  yet*: no stated answer before onset. Also recorded: whether *any* earlier answer was correct; distance (tokens) from the end of the last
  answer statement to onset; number of answers before onset.
- **Loop content.** The repeated unit = the text of the first 32-gram (with count ≥3) plus the following 200 chars of the first copy.
  Rule-based label (first match in this order): *recheck* (wait / check / verify / double / re-check / hmm / actually / let me re / confirm /
  make sure / is that right / correct?); *reread* (the problem / question / says / states / asks / "…" quotes / re-read / reading);
  *computing* (digits with an operator or "=", calculate, compute); else *other*. ~20 are also read by eye.
- **Matched-depth null (base rate).** For every onset at token depth t in trace i, every other trace j of the same dataset and source whose
  loop-free prefix is longer than t (no onset, or onset > t) is classified at depth t with the same rule (last stated answer before t: correct /
  wrong / none). Null share = average over onsets of the class share among those j. Second null: a uniform random position in [0, onset) of
  the same trace (10 draws per trace).
- **resid·u (GPU).** Exp A measurement exactly (`measure_prefix`, token covering the first content char of the next line, hidden_states[16]
  · u15, also hs20·u15/u19), at every stated answer before onset (looping traces) and every stated answer in finishing traces (`</think>`
  closed and no onset). Higher value = "doubt/incorrect" direction (Exp A: AUC 0.80 with incorrect positive).
- **AUC.** Positive class = incorrect answer, signal = u15@hs16; 2,000-sample bootstrap CI, resampled by trace.

## Pre-registered predictions and kill rule (2026-11-05, before any onset was classified; not edited afterwards)

Primary data, GSM8K + MATH-500 pooled, one onset per looping trace.
- **P1.** Among onsets with ≥1 stated answer before onset, ≥ 60 % are after-correct (a / (a + b) ≥ 0.60).
- **P2.** No-answer-yet (c) onsets are < 30 % of all onsets. If (c) is the largest class, the impasse account wins for this model.
- **P3.** resid·u marks the answers preceding an onset as correct at least as well as answers in finishing traces:
  AUC_loop (all answers before onset in looping traces) ≥ AUC_fin (all answers in finishing traces) − 0.05.
  If finishing traces have < 10 incorrect answers in the primary data, AUC_fin pools primary + replication finishing traces; if still < 10,
  AUC_fin is replaced by the AUC over all answers in non-looping traces (finishing or cut without loop), and this is stated.
  Secondary (no pass/fail): among *correct* answers only, AUC of u (loop-preceding vs finishing; 0.5 = equally "correct"-marked);
  the same for the last answer before onset; AUC after regressing out log token position.
- **P4 (control).** The after-correct share at onsets exceeds the matched-depth null after-correct share by ≥ 0.10 (bootstrap 95 % CI of the
  difference above 0). If P1 holds but P4 fails, the honest reading is "correct answers are common by then", not "success triggers loops".
- **Kill rule.** If (b) + (c) ≥ 60 % of all onsets, doubt-after-success fails for this model; reported as such.
- Descriptive (no pass/fail): distance from last answer to onset; loop-content labels by class; per-dataset tables; replication on Astra
  logs and MATH L5; stratification by "thinking-off solves".

## Procedure
1. Pilot: 20 primary looping traces, read onset context and detected answers by eye; fix detector/onset bugs only (amendments logged here).
2. Full CPU analysis in a Slurm job; GPU job for resid·u (≤ 2 GPU-h budget).

---

## Amendments

### Amendment 1 (after reading the 30-trace pilot, `PILOT_fresh30.jsonl`, 28 loops, all GSM8K; before the full run; predictions unchanged)
Reading the pilot showed two problems with the pre-registered measure. Both are kept as the **primary** analysis and are reported
exactly as pre-registered. Two sensitivity analyses are added, are labelled as post-pilot, and do not change any pass/fail verdict.

What the pilot showed. 23 of the 28 onsets were class (c). In most of these the model *had* already computed the right number, but it
never said "the answer is"/`\boxed`. Examples: `Total = 3 + 4 + 0 = 7` (gold 7) → "Wait, did I miss something? Let's re-read the prompt";
`Brandon = 4 * 2 = 8` (gold 8) → "Wait, let me double check the wording"; `Distance = 45 - 20 = 25` (gold 25) → "double check if there's any
trick". The Exp A detector only finds explicit answer statements, so it pushes onsets toward (c). Also, the 32-gram ≥3 rule fires when
the model quotes the problem statement verbatim for the 3rd time ("Let's re-read the prompt. 'Every day, Wendi feeds …'"), with the
copies 1,000+ tokens apart. That is a re-reading habit, not a run-away loop.

- **S1, computed-result detector.** A "computed result" is any line that ends in `= X` or `is X`, where X is the right-hand side after the
  last `=` (or after `is`), has ≤ 40 characters, contains a digit or LaTeX command, and is `answer_like`. It is graded with the same
  `grade_value`. Classes are then (a) the last computed result before onset is correct, (b) it is wrong, or (c) no computed result yet.
  The detector is gold-blind and symmetric. It is crude, because intermediate steps count as "results" and push toward (b). Also
  reported: "gold already reached" = any computed result before onset is correct.
- **S2, onset variants.** (i) *no-quote onset*: 32-grams whose decoded text (whitespace-normalised) is a substring of the problem text are
  ignored. [Pilot fix: the onset 32-gram usually straddles the end of a quote, so the implemented test is "contains a run of ≥ 8 words
  that occurs in the problem"; 12 words still let quoted sentences through.] (ii) *terminal onset*: test 8 runs (a token is in a loop if its ending 32-gram was seen before; a run start counts if it is
  preceded by ≥ 64 non-loop tokens). The terminal onset is the last such run start, taken only in traces whose last 2,000 tokens are
  ≥ 50 % loop tokens. This is the loop that ends the trace, which is the one the edit removes.
The class tables, P1/P2/P4/kill quantities and the matched-depth null are reported for primary × {S1} × {S2-i, S2-ii}. The verdicts
use only the primary (pre-registered detector + rule).

## Results (written after the runs; predictions above unchanged)

Runs: extraction (CPU) for fresh on the login node (single-threaded; identical code), astra + hard + resid·u in Slurm job 7305022
(L40S, 4 min 21 s). **GPU-h: 0.07.** Numbers: `notes/onset_results_v1.json` (script `scripts/onset_analyze.py`); per-trace rows
`$BLF_DATA/onset/extract_{fresh,astra,hard}.jsonl`, `meas_{fresh,astra,hard}.jsonl`.

Data: fresh 1000 traces, 941 loop (53 finish without a loop); Astra 400, 378 loop (21 finish); MATH L5 400, 395 loop. All onsets are
inside the thinking part.

### Primary (pre-registered detector + onset rule), one onset per looping trace

| source | ds | onsets | (a) after-correct | (b) after-wrong | (c) no answer yet | a/(a+b) | c share | b+c | matched-null a share | P4 diff [95 % CI] |
|---|---|---|---|---|---|---|---|---|---|---|
| fresh | gsm8k | 470 | 75 | 6 | 389 | 0.93 | 0.83 | 0.84 | 0.26 | −0.10 [−0.13, −0.07] |
| fresh | math500 | 471 | 25 | 2 | 444 | 0.93 | 0.94 | 0.95 | 0.09 | −0.04 [−0.06, −0.02] |
| **fresh** | **all** | **941** | **100** | **8** | **833** | **0.93** [0.88, 0.97] | **0.89** [0.87, 0.91] | **0.89** | 0.17 | **−0.06** [−0.08, −0.04] |
| astra | gsm8k | 187 | 23 | 2 | 162 | 0.92 | 0.87 | 0.88 | 0.19 | −0.07 |
| astra | math500 | 191 | 10 | 1 | 180 | 0.91 | 0.94 | 0.95 | 0.11 | −0.06 |
| astra | all | 378 | 33 | 3 | 342 | 0.92 | 0.90 | 0.91 | 0.15 | −0.06 [−0.09, −0.04] |
| hard (MATH L5) | all | 395 | 3 | 4 | 388 | 0.43 | 0.98 | 0.99 | 0.02 | −0.01 |

- Median distance from the last answer to onset (a): 625 tokens (fresh), 704 (Astra).
- Difficulty (fresh, ids with thinking-off data): when base thinking-off solves the problem, a = 31/237 (13 %) and c = 86 %. When it
  does not, a = 4/135 (3 %) and c = 96 %.

**Verdicts.**
- **P1 holds numerically but describes a minority.** 0.93 [0.88, 0.97], but the base is only 108 of 941 onsets, because an explicit
  stated answer before onset is rare.
- **P2 fails.** (c) is 89 % of onsets (Astra 90 %, L5 98 %), and (c) is the largest class in every stratum. As pre-registered:
  **the impasse-like account wins for this model** under this detector.
- **P4 fails, with the opposite sign.** Onsets are *less* likely than loop-free positions at the same depth in other traces to follow a
  stated correct answer (0.11 vs 0.17).
- **Kill rule fires.** b + c = 0.89 ≥ 0.60 (Astra 0.91). **Doubt-after-success, as pre-registered, fails for this model.**

### Sensitivity (amendment 1, post-pilot; no verdict changes)
| variant (fresh, all) | onsets | a / b / c | c share | a/(a+b) | null a | P4 diff |
|---|---|---|---|---|---|---|
| no-quote onset | 935 | 110 / 11 / 814 | 0.87 | 0.91 | 0.18 | −0.06 |
| terminal onset | 827 | 170 / 30 / 627 | 0.76 | 0.85 | 0.20 | +0.00 [−0.02, 0.03] |
| S1 computed results, primary onset | 941 | 193 / 678 / 70 | 0.07 | 0.22 | 0.24 | −0.04 |
| S1 computed results, terminal onset | 827 | 211 / 614 / 2 | 0.00 | 0.26 | 0.21 | +0.04 [0.01, 0.07] |

(Astra matches within a few points; see the JSON.)

- With the gold-blind "computed result" detector, (c) almost disappears, because the model is always computing something. But the
  last computed value before onset is usually an intermediate quantity that differs from gold: a/(a+b) = 0.22, the same as the
  matched null 0.24.
- The gold value has already been computed somewhere before onset in 48 % of fresh onsets (GSM8K 59 %, MATH-500 36 %; Astra 52 %).
  No null was computed for this number.
- The loop-content rule labels almost everything "recheck" (a 83 %, c 94 %), because one loop period is long and nearly always
  contains "wait"/"check", so it does not discriminate.

### Hidden-state signal (u15 @ hs16; positive = incorrect)
| set | n corr / inc (traces) | AUC [95 % CI] |
|---|---|---|
| fresh, all answers before onset | 370 / 26 (108) | 0.69 [0.51, 0.86] (log-pos residual 0.64; hs20·u15 0.71) |
| fresh, last answer before onset | 100 / 8 | 0.70 [0.43, 0.93] |
| fresh, answers in finishing traces | 126 / 0 | — |
| Astra, before onset | 202 / 7 (36) | 0.38 [0.09, 0.80] |
| Astra, finishing | 78 / 2 | 0.47 [0.06, 1.0] |
| hard, before onset | 22 / 5 (7) | 0.32 |

- **P3 is not testable as pre-registered.** Finishing traces contain 0 incorrect answers (fresh), 2 when pooled with Astra, and the
  fallback (non-looping traces) adds none. The pre-onset AUC (0.69, CI down to 0.51) is not shown to be below Exp A's 0.80, and is
  not shown to equal it.
- Secondary, among *correct* answers only: answers that precede a loop onset sit **higher** on u (the doubt/incorrect side) than
  correct answers in finishing traces. AUC 0.71 fresh (log-pos residual 0.72); Astra 0.79. Medians 0.69 vs 0.55 (fresh). So
  correct answers that are followed by a loop are marked *less* confidently correct by the state, which is again the opposite of the
  hypothesis's "already marks as correct". *Hypothesis, untested:* this reflects trace-level context, e.g. the problem already looks
  ambiguous to the model at that point.

### What reading the traces showed (38 pilot traces + 6 after-correct + 10 MATH-500)
- **GSM8K (c) onsets** are mostly re-interpretation spirals: "Let's re-read the prompt", "Is it possible that … ?", or re-quoting a
  sentence of the problem. These often come after the right number has been computed but never stated as the answer, e.g.
  `Total = 3 + 4 + 0 = 7` → "Wait, did I miss something? Let's re-read the prompt". That case is closer to doubt-after-success than
  the primary detector can show. The S1 numbers bound it at most at about half of onsets, with no null.
- **MATH-500 / L5 (c) onsets** are mostly genuine impasses: the model is stuck inside a computation and keeps redoing the same step,
  e.g. "Wait, $x^3+3x^2$ at $x=1$ is 4 … Manual gives −2".
- **(a) onsets** look like the hypothesis says: a correct answer, then "Let's double check with a different interpretation", which
  is recopied until it loops. They are 11 % of onsets.

