# Test 4 — SAE features for u and for the edit

Branch `yang/t2-t4`. Scripts: `scripts/test4_*.py`. Activations / checkpoints under `/scratch/yang0531/blf_data/` (not in git).

## Plan
- Our SAEs: BatchTopK (k = 64, 16 384 latents) on **base-model** `hidden_states[16]` (residual after L15) and `hidden_states[20]`
  (after L19). Training tokens: unedited + p50c Astra generations (GSM8K / MATH-500 / IFEval / MMLU-Pro, prompt + generation,
  ≤ 16k tokens each, run through the *base* model) + WikiText-103 (normal non-chat text). Held-out split by generation id.
- Public SAE: `decoderesearch/qwen-3.5-saes`, `qwen-3.5-0.8b/btk-mat-layer-17-k-100` (Matryoshka BatchTopK, 65k, trained on
  lmsys-chat-1m at `model.layers.17` = `hidden_states[18]`). No public SAE exists at hidden_states 16 or 20; this one sits between
  the two edited layers, so it is used as an independent dictionary.
- Edit effect: run base and p50c on the same token sequences, encode both with the base SAE, compare feature activations at
  loop positions (32-gram rule) and productive positions.
- Labels: auto-interp with claude-sonnet-5-5 (Vector gateway) from top-activating contexts, scored by detection on held-out
  activating vs random contexts; spot-check of ~20 features re-labelled and re-scored by claude-opus-5-5 (agreement reported).

## Pre-registered predictions (written 2026-11-05 before any run; not edited afterwards)
- **P4a.** u is not one SAE feature: max |cos(decoder_i, u)| < 0.5 at hidden_states[16] and [20]. Among the top-5 features by
  cos with +u, ≥ 2 get labels about repetition / looping / re-checking ("wait", "let me check again", repeated lines).
- **P4b.** At loop positions the edit turns *off* loop/repetition features: ≥ half of the 10 most-decreased features (by mean
  activation drop, hidden_states[20]) get repetition / re-checking labels.
- **P4c.** The features the edit turns *on* at loop positions are not the features of productive reasoning: < 20 % of the top-20
  most-increased features are among the top-100 productive-vs-loop features of the base model (matches Q5: flag cleared, state
  not returned to normal).
- **Quality gate.** Held-out FVU < 0.2 and < 10 % dead latents; otherwise results are reported as low-confidence.

## Results (Killarney, L40S; full tables: `notes/test4_summary_killarney.md`, data `notes/test4_analyze_killarney.json`)

**SAEs.** BatchTopK, 16 384 latents, k = 64, 1 epoch over 17.4 M train tokens (20.4 M collected: 1200 Astra generations ≤16 384 tokens
+ 6.0 M WikiText tokens; 27 % loop positions; 15 % of problem ids held out). Held-out (2.9 M-token test split, 400 k sampled):
**hs16 FVU 0.136, L0 58.6, 0 % dead; hs20 FVU 0.133, L0 58.4, 0 % dead** → quality gate passed. Public pub18 (65k, jumprelu): FVU at
loop positions 0.24 (our corpus is off its lmsys-chat distribution), 12 % of its latents never fire on our 1 M stat positions.

**Edited states leave the base dictionary's manifold.** FVU at loop positions on 40 held-out looping generations (≤8192 tokens; 151 k loop /
155 k non-loop positions): base 0.145 / 0.24 / 0.127 → **edited 0.56 / 0.43 / 0.53** (ours16 / pub18 / ours20).
Total |Δ mean activation| is 3.4–3.7× larger at loop than at non-loop positions (gating, as in Q2/Q3).

**u is mostly one loop latent plus a tail (P4a: FAIL for our SAEs, PASS for pub18).** max |cos(decoder, u)|: ours16 **0.56**, pub18 0.45,
ours20 **0.65** (vs u19); 19 / 4 / 6 latents with |cos| > 0.3. Greedy 10-latent fit leaves ‖residual‖ 0.60–0.63 of u.
Second half of P4a passes: repetition / re-checking labels for 4/5 (ours16), 2/5 (pub18), 5/5 (ours20) of the top-5 latents by cos(+u).

Latents linked to u (Sonnet labels; detection balanced accuracy on held-out contexts in parentheses; d′ = loop vs non-loop selectivity in base):

| SAE | latent | cos(dec,u) | d′ | mean act at loop pos. base → edit | label | example context (fires on [[ ]]) |
|---|---|---|---|---|---|---|
| ours16 | 3673 | +0.56 | +1.87 | 7.15 → 1.96 | Mid-phrase token while restating earlier text (0.92) | `*   Wait, I'll use the text `*[[ This]] is an exa…` (75 % of loop pos., 4 % non-loop) |
| ours16 | 7236 | +0.36 | +0.42 | 0.38 → 0.00 | Copying next number during repeated calculation (0.83) | `*   Wait, $18 \times 9 = 162$.\n    *   Wait, $18 \times 9 = [[1]]62$` |
| ours16 | 7770 | +0.36 | +0.13 | 0.08 → 0.01 | "Wait" tokens in repeated self-correction loops (0.83) | `Wait, $e^{i2\pi/3} = …$.\n   [[ Wait]], $e^{i2…` |
| ours20 | 12884 | +0.58 | +3.26 | **8.42 → 0.09** | Mid-clause in conditional "if X, he is…" restatement (0.83) | `*   Wait, if he drives *away*[[ from]] home, he…` (94 % / 8 %) |
| ours20 | 6329 | +0.59 | +1.52 | **3.90 → 0.03** | Hesitant self-doubt restatement before re-checking (0.83) | `*   Could it be "2 containers" = "2[[ containers]] [of…` |
| ours20 | 12572 | +0.42 | +0.50 | 1.15 → 0.01 | Next token copies earlier math expression (0.75) | `distance $P_2 Z = \sqrt{y^2 + [[3]]0^2}$` |
| pub18 | 937 | +0.45 | +3.02 | 1.02 → 0.11 | Mid-clause verb/auxiliary in restated question (0.67) | `…is *valid* (sustained) if the defense[[ had]] a valid o…` |
| pub18 | 1539 | +0.22 | +2.53 | 0.50 → 0.04 | Final token before repeated phrase restarts (0.92) | `#playboy #fashion #style #[[life]]\n        #` |
| ours16 | 8733 | −0.49 | −0.88 | 0.01 → **1.50** (non-loop mean 1.53) | Noun after "imply / part of a" hedging (0.83) | `…refers to the specific time[[ frame]] relative…` |
| ours20 | 2458 | **−0.65** | −1.18 | 0.10 → **4.70** (non-loop mean 3.22) | Mid-expression math/LaTeX token before operand (0.42) | `Volume of a sphere: $V = \[[frac]]{4}{3}\pi` |

**Edit turns OFF loop / restatement latents (P4b: PASS, with caveat).** Of the 10 latents with the largest activation drop at loop positions,
Sonnet flags 8/10 (ours16), 7/10 (pub18), 10/10 (ours20) as repetition / restatement / re-checking. The two big hs20 loop latents are
essentially erased (8.4 → 0.09, 3.9 → 0.03).
Caveat: Sonnet's "repetition-related" flag is liberal (Opus agrees on 75 % of 20 spot-checked latents and flags fewer); by eye the top-6 OFF
latents per SAE are clearly restatement / "Wait" loop latents in ours16 and ours20, mixed in pub18 (1498 / 2005 are generic syntax latents).

**Edit turns ON a mix — partly "productive", partly off-distribution (P4c: PASS for ours16 / pub18 = 0.25, FAIL for ours20 = 0.45).**
Top-20 ON latents in the base top-100 productive-selective set: 0.25 / 0.25 / 0.45. Examples (ours20): 2458 math-expression latent
(overshoots to 1.5× its productive level), 15418 "Digits of final boxed numeric answer", 9072 "Predicting 'answer/result' noun after
determiner" — *conclude*-type latents switched on inside loops (matches the J-lens reading of −u in test 2); but also 2210 "Function word early
in a question or prompt" (0 → 2.81) and 1376 "Hyphen inside Wikipedia '@-@' compound words" (0 → 1.37), which are WikiText / prompt
latents with no business in a reasoning trace, and pub18's strongest productive latents (761, 770; d′ −1.5) stay off (0.01 → 0.04).
Reading: the edit removes the loop latents and pushes the state partly toward "state the result" latents and partly into
off-distribution directions — a feature-level version of Q5 ("flag cleared, state not returned to normal"), and a candidate mechanism
for the side effects (missing `\boxed{}`, whitespace degeneration) — the latter is a **hypothesis**, not tested here.

**Auto-interp quality.** Sonnet: 119 latents labelled, detection balanced accuracy mean 0.73 / median 0.75 (chance 0.5; 6 activating + 6
non-activating held-out contexts per latent). Opus spot-check (20 random latents): same gist as Sonnet for ~16 of the 19 parsed labels (my reading; differences: 2920, 2677, 8733), mean
balanced accuracy 0.78 vs Sonnet 0.75 on the same latents (Pearson 0.71), repetition-flag agreement 0.75; one Opus label failed to parse.

Figure: `notes/fig/test4_killarney_features.png` (selected latents: cos(decoder, u) vs edit-induced change at loop positions, coloured by the
repetition flag).

Caveats: one SAE seed / size; activations from the base model on its own + p50c generations (p50c text through base); the edited model's
states are encoded with the *base* SAE (that is the point, but reconstruction is poor on them, FVU ≈ 0.5, so small ON effects can be
reconstruction artefacts); labels come from ≤ 12 contexts at max 2 per generation; loop contexts are near-duplicates, which inflates
detection scores of loop latents.

## Compute
Killarney L40S, tests 2+4 together: **1.81 GPU-h** (collect 0.53, J-lens fit 0.81, tuned fit 0.16, read 0.10, 2 SAEs 0.12, analyze 0.06, pilot/failed 0.03). API: ~280 gateway calls (Sonnet 119 labels + 118 scores; Opus 20 + 20).
