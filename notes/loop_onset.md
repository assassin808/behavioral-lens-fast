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
(none yet)

## Results
(to be filled after the runs)
