# Confirmation run: does `clampp5` reduce whitespace degeneration without giving back the loop fix? (branch `yang/sidefx-confirm`)

Follow-up to `notes/sidefx.md` (branch `yang/sidefx`). There, clampp5 (α = 6 on all 128 columns; the edit may not push
resid·u below the 5th percentile of normal states, F = 0.090 / 0.346) halved the whitespace-run rate vs the matched hook arm
p50chook: 6/400 vs 13/400, Fisher p = 0.16 (k = 0, first 200 + 200 problems, 16k cap). That is too weak to call it a better edit.
This run confirms or refutes it.

## Pre-registration (written 2026-11-05 before any run of this branch; not edited afterwards)

### Arms (identical harness)
- **p50chook** (control): spec `notes/sidefx/specs/p50chook.json`, hook α = 6 on all 128 columns, no floor.
- **clampp5** (treatment): spec `notes/sidefx/specs/clampp5.json`. Same cols and α; the only difference is `floor`
  (L15 0.0901, L19 0.3457). Checked before writing this: `cols` and `alpha` identical, `floor` null vs set.
- Real-weight p50c is **not** run (not cheap enough under the 5 GPU-h budget). p50chook ≈ p50c was established in `sidefx.md`.
- Both arms: `scripts/sidefx_eval.py` unchanged in generation (same prompt `build_prompt`, `SAMPLING_THINK35`, grader `grade`,
  `loop_signals`, `classify`), base weights + hook, eager vLLM (no torch.compile, no CUDA graphs), bf16, prefix caching off.
  The only harness addition is an output field (`token_ids`, for the 32k extension below); it does not touch generation.

### Data
- MATH-500 all 500 + GSM8K test first 500, thinking on, **2 samples per problem (k = 0, 1), seed = 1** (the screen used seed 0;
  a new seed so the first 200 problems get new samples, not a replay).
- Problems 0–199 of each set were in the screen; problems 200–499 are fresh. Primary analysis uses all 1,000; a secondary
  analysis on the 600 fresh problems is reported alongside.

### Cap: why 16k, and the 32k extension (budget decision made from the screen's numbers, before any run)
- Cost estimate at 32k. From the screen (p50chook, 16k): finished samples average 8.3k / 8.7k tokens (GSM8K / MATH),
  23 % / 64 % are cut at 16k. Assume cut samples average 28k at a 32k cap. Then per arm: GSM8K 1,000 × 12.8k + MATH
  1,000 × 21k ≈ 34M tokens, two arms ≈ 68M tokens. The screen ran 3.6k tok/s for two concurrent arms on one L40S at 16k
  (KV-limited; it will be lower at 32k) → **≥ 5.2 GPU-h**, over the 5 GPU-h budget.
- Therefore, per the instruction: **stage 1 = 16k cap** (`max_tokens` 16,384) for both arms, all 1,000 problems × 2.
  Estimate ≈ 48M tokens ≈ 3.7 GPU-h. All primary and non-inferiority criteria are evaluated on stage 1.
- **Stage 2 (secondary, descriptive, only if stage-1 GPU-h + stage-2 estimate ≤ 5 GPU-h):** continue every *cut* MATH-500
  sample with k = 0 from its exact 16k token ids to a total sequence length of 32,768 (prompt + output, = Astra's
  `max_model_len`), same hook, same sampling params (fresh random stream; same distribution). Order: p50chook first, then clampp5.
  This gives MATH-500 accuracy at 32k (k = 0, 500 problems) to put next to Astra's 48.1 %. A stage-2 sample is the 16k text
  plus its continuation, graded and classified exactly like a stage-1 sample.

### Metrics (per sample, from `classify` / `loop_signals` as in `sidefx.md`)
completion, acc_all = completion × acc_on_completed, loop_end, whitespace (`cut_whitespace`: cut and last 2,000 chars are
whitespace), no-box among finished (`pred is None` and not cut), cut-after-`</think>` (exploratory), paired accuracy
(problems where both arms finish ≥ 1 sample). Every comparison is **paired by problem**: per problem, the per-arm mean over its
2 samples; CIs are problem-level bootstrap (10,000 resamples, seed 0).

### Primary criterion (one-sided)
- **H_conf:** whitespace rate (clampp5) < whitespace rate (p50chook), pooled over both sets (1,000 problems, 2,000 samples per arm).
- Test: per-problem difference d_i = ws̄_p50chook,i − ws̄_clampp5,i; one-sided paired sign-flip permutation test on Σ d_i
  (100,000 flips, seed 0). **Confirmed if p < 0.05.** Effect reported as rate difference with 95 % bootstrap CI and ratio.
- Power note: at the screen's rates (3.3 % vs 1.5 %), the expected counts are ≈ 66 vs 30 of 2,000; z ≈ 3.7.

### Non-inferiority (both must hold for clampp5 to count as a "better edit")
- **NI-loop:** loop_end(clampp5) − loop_end(p50chook) has upper 95 % one-sided bootstrap bound (= 95th percentile) **≤ +0.03**.
- **NI-acc:** acc_all(clampp5) − acc_all(p50chook) has lower 95 % one-sided bootstrap bound (= 5th percentile) **≥ −0.03**.
- Both pooled over sets; per-set values reported but not criteria.

### Predictions (my expectation, from the screen)
- P1: primary confirmed; whitespace ratio clampp5 / p50chook ≈ 0.5 (0.3–0.7).
- P2: NI-loop holds (screen: 0.075 vs 0.077).
- P3: NI-acc holds (screen: 0.453 vs 0.460); acc difference within ±0.02.
- P4 (exploratory): no-box among finished and cut-after-`</think>` do not differ (screen: both at floor / equal).

### Kill rules
- **KP (pilot gate, before scaling).** 10 GSM8K + 10 MATH-500 (problems 0–9), 1 sample, 16k, both arms concurrently.
  Stop and fix if any of: (a) the `HOOKS` line is missing in either arm, or reports different cols / α, or floor not
  None vs (0.090, 0.346); (b) incoherent text in either arm; (c) p50chook loop_end > 0.2 (loop fix absent ⇒ hook not working);
  (d) a class label contradicted by reading the text (e.g. "whitespace" sample that is not whitespace).
  Pilot outputs are separate files and are **not** part of the confirmation data.
- **KH (harness drift).** If p50chook stage-1 pooled loop_end > 0.15 or completion differs from the screen's (0.56) by > 0.10,
  the run is declared not comparable with the screen and reported as such (no reinterpretation of the criteria).
- **K-refute.** If the primary test gives p ≥ 0.05, clampp5 is reported as **not confirmed** (the screen effect was noise or
  is too small to matter). If the primary holds but NI-loop or NI-acc fails, clampp5 is reported as **not a better edit**
  (it trades the side effect against the loop fix / accuracy, same axis as the other variants). No further variants are
  screened on this branch in either case.
- **Budget kill.** Stop launching new jobs once 5 GPU-h is reached; whatever has finished is analysed as is and labelled.

## Outcome
(to be filled after the runs; everything above stays unedited)

### KP pilot gate (job 7303535; 10 + 10 problems × 1, 16k, seed 1; files `/scratch/yang0531/blf_data/sidefx/confirm_pilot/`) — **pass**
- **Hooks.** Both arms print `HOOKS`: L15 / L19 cols = 64, α = 6.000, identical sum|coef| (11.7541 / 9.2215). floor is None
  (p50chook) vs 0.0901 / 0.3457 (clampp5).
- **Coherent text** in both arms. p50chook loop_end is 1/20, so the loop fix is present.
- **Counts (n = 20 per arm; noise-level, not data):**

  | arm | completion | acc | loop_end | whitespace |
  |---|---|---|---|---|
  | p50chook | 11/20 | 7/20 | 1/20 | 1 |
  | clampp5 | 10/20 | 7/20 | 3/20 | 1 |

- **Labels checked by eye.**
  - The whitespace samples really are about 1M-space runaway indentation.
  - The `cut_repetitive` sample really is a loop ("Wait, I'll check if there is any other interpretation …" × N).
- **Observation (not a criterion).** Some `cut_still_reasoning` samples are runaway indentation that has not yet filled the last
  2,000 characters (e.g. gsm8k_test_8 clampp5, precalculus/927 p50chook). The registered whitespace metric therefore undercounts
  indentation drift a little, equally in both arms. Kept as registered.
- `token_ids` saved: length = n_tokens.
- **Stage 1 submitted.** Job 7303913 runs MATH-500 (2 slots, 250-problem chunks). Job 7303916 runs GSM8K, afterok.
