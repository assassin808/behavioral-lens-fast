# Test 2 — reading u (and the edit's Δresid) with fitted lenses

Branch `yang/t2-t4`. Scripts: `scripts/test2_*.py`, shared helpers `scripts/test24_common.py`.

## Pre-registered predictions (written 2026-11-05 before any run; not edited afterwards)

Setup. u = `shared/u_L15.npy` (+u = loop direction), u19 = `shared/u_L19.npy`. Lens layer index = block output
(block l ↔ `hidden_states[l+1]`). We read u at block outputs 15…22 (hidden_states 16…23) plus the final layer (logit lens).
Lenses: (i) logit lens; (ii) **J-lens** (Anthropic jacobian-lens; the public Neuronpedia fit on WikiText, plus our own
fit on the loop/math/non-math mix); (iii) J++ lens (public); (iv) **tuned lens** (our fit, per-layer affine translator, KL to
final logits, ~1000 contexts of the mix); (v) **R-lens = "run-the-rest" causal readout** (our definition): add ±s·u to the
residual at block output l at one position of a real context, run blocks l+1…23 unchanged, and record the mean change of the
next-token log-probs at that position. Unlike (i)–(iv) this is causal and keeps all nonlinearities (attention, norms);
s is set to the edit's own typical |Δ| along u at loop positions.

Token classes, fixed now: CONCLUDE = tokens that state/close a result (` So`, ` Therefore`, ` Thus`, ` answer`, ` result`,
` final`, ` boxed`, `\boxed`, ` total`, ` equals`, ` =`, `</think>`, ` Final`, `**`, ` is`...); CONTINUE = tokens that
restate/re-check/keep enumerating (` Wait`, ` wait`, ` Let`, ` again`, ` check`, ` verify`, ` double`, ` Hmm`, ` But`,
` Alternatively`, ` re`, ` Maybe`, ` Or`, newline/indent tokens, ` -`, ` *`). Exact lists live in `test24_common.py`.

- **P2a (axis).** For the J-lens and the tuned lens, the readout of −u favours CONCLUDE over CONTINUE
  (mean lens logit CONCLUDE − CONTINUE is higher for −u than for +u) in ≥ 5 of the 8 block outputs 15…22.
  The R-lens shows the same sign at loop positions. Fail = sign mixed (< 5/8) or reversed: u is then not a
  "keep enumerating vs conclude" axis in mid layers.
- **P2b (variance).** The variance share of u in the residual, var(h·u)/tr Cov(h), is ≥ 10× the isotropic baseline 1/1024 at
  hidden_states 16–20, and its mean projection is larger at loop positions than productive ones at every layer 13–24.
- **P2c (Δresid read).** Lens read of the edit's Δresid at loop positions (mean over positions of lens(h_edit) − lens(h_base)):
  correlates with the −u readout at block outputs 15–19 (Pearson over the vocab > 0.5) and decorrelates by the final layer
  (< 0.3), consistent with Q3b (10 % of final Δ along u). At the final layer the actual repeated next token is suppressed
  (mean Δ log-prob of the loop's next token < 0) — that is the known loop-NLL effect, a sanity check, not a prediction.

## Results

### R0 — pilot (Killarney, L40S) and CPU bare-direction reads (`scripts/test2_bare_cpu.py`, `notes/test2_bare_killarney.json`)

Pilot (`scripts/test24_pilot.py`): rebuilt p50c differs from base only in L15/L19 `down_proj`; ΔW rank-1 (svd 1.749, 0.001 / 1.399, 0.001),
|cos(U1, shared u)| = 1.000 at both layers. Corpus = 1200 Astra generations with known prompts (600 unedited + 600 p50c; GSM8K / MATH-500 /
IFEval / MMLU-Pro), ≤16 384 tokens each, 14.4 M tokens, 38 % loop positions (32-gram rule); 10 eyeballed loop snippets are genuine verbatim
loops ("Wait, I need to ensure "hajj" is lowercase." ×N, "or "E" or "E" …", whitespace runs in p50c). Residual norm (median): 4.2 at
hidden_states[16], 7.7 at [20], 9.0 at [23] (hidden_states[24] in HF is *post*-norm; all our code reads block outputs via hooks).

Bare-direction reads (lens of the unit vector, no context; top tokens; cls = mean centred logit CONCLUDE − CONTINUE):

| lens | hs | +u15 top | −u15 top | cls(−u15) |
|---|---|---|---|---|
| logit (= Q1) | any | `...*` `...**` `而是因为` | `呗` ` first` ` series` ` thereby` `即可` ` thus` | +0.02 |
| logit | any, u19 | `...*` `她还是` `而是因为` `他还是` | ` first` ` count` `即可` ` yields` ` result` ` exact` ` value` | +1.24 |
| J-lens (public) | 16 | `**?` `还不是` `?**` `!?` `根本不是` `还是在` | ` examinations` ` methodologies` ` publications` ` analyses` ` computations` | +3.31 |
| J-lens | 17 | `其实是` `**?` `还不是` `还是在` `实际上是` `真的是` | ` methodologies` ` examinations` ` computations` ` analyses` | +2.94 |
| J-lens | 19 | `**?` `...**` `已经是` `现在是` `其实是` `Actually` | ` methodologies` ` computations` ` conditions` ` analyses` ` calculations` ` estimates` | +1.76 |
| J-lens | 20, u19 | `还是` `...**` ` hanem` `他还是` `她还是` `还在` `还是在` | ` results` ` outcomes` ` formulas` ` computations` ` calculations` ` result` ` yields` | +3.34 |
| J-lens | 22, u19 | `>--` ` hanem` `还在` `它是` `还是` `他还是` `仅仅是` `还不是` | ` formulas` ` computations` ` results` ` multiples` ` outcomes` | +3.39 |
| J++ (public) | 20, u19 | `...**` `**-` `**?` ` hanem` `已经是` | ` results` ` result` ` outcomes` ` formulas` ` concepts` ` numerical` | +3.10 |

(还是 = "still / or", 还不是 = "still not", 其实是 / 实际上是 = "actually is", 真的是 = "really is", hanem = Hungarian "but rather".)
cls(−u) > cls(+u) at 8/8 of hs 16–23 for the J-lens (u15 and u19), 7/8 for J++ (hs 21 u15: −0.09). Reading so far (bare reads only, not
yet the pre-registered base-relative / R-lens test): in mid layers +u reads as *hedging / contrast / "still… / actually… / is it?"*
continuation markers and −u as *result / computation nouns* ("results, outcomes, formulas, computations, yields"). That is closer to
"doubt-and-keep-going vs. state-the-result" than to "enumerate", and much clearer than the logit lens. Caveat: the J-lens is a
WikiText-averaged linearisation and bare reads ignore the context-dependent norm; the base-relative and R-lens reads below test this on
real loop states.

### R1 — full readout (Killarney, L40S; `scripts/test2_read.py` → `notes/test2_read_killarney.json`, tables `notes/test2_summary_killarney.md`)

Data (held-out split only): 40 unedited looping generations (≤8192 tokens, 138 634 loop tokens; 200 loop + 200 non-loop positions
sampled per text), 20 finished loop-free math generations (≤4096), 20 WikiText 1024-token chunks. Variance: 399 153 generated positions.
s (the edit's own median |Δ·u15| at loop positions, hs16) = **2.32 = 0.57 × ‖h‖** — the edit is a large push.

Lenses: logit; **tuned lens** (ours: 1000 contexts = 350 loop / 350 math / 100 IFEval+MMLU-Pro / 200 WikiText, 200 positions each, 180 k
train positions, 2 epochs); **J-lens** public (Neuronpedia, WikiText, 233 prompts) and **ours** (anthropics/jacobian-lens estimator,
400 prompts of 128 tokens from the same mix); public **J++**; **R-lens** (defined above: add ±s·u15 at block output l at all generated
positions, run the rest of the model, mean Δ log-prob / Δp at the same positions; 20 loop + 10 math + 10 WikiText texts).
Tuned-lens held-out KL(model ‖ lens) [nats], logit → tuned: hs16 10.85 → 0.66, hs18 9.09 → 0.51, hs20 6.80 → 0.22, hs22 4.07 → 0.12, hs23 1.95 → 0.08.

**P2a (axis): PASS.** cls(−u) > cls(+u) at loop positions at **8/8** layers (hs16–23) for every lens (base-relative: logit, tuned,
J-lens public, J-lens ours, J++, R-lens; bare: 8/8 except J++ 7/8). Effect size cls(−u) − cls(+u) [Δlogp]: R-lens 2.3 (hs16) → 0.8 (hs23),
tuned 2.8 → 0.6, J-lens ours 3.9 → 0.7, logit lens only 0.45 → 0.02 (`notes/fig/test2_killarney_axis_dresid.png`, left).
What the tokens say (top by mean Δp at real loop states):

| hs | R-lens (causal), −u promotes | R-lens, +u promotes | J-lens (ours), −u | J-lens (ours), +u |
|---|---|---|---|---|
| 16 | `\n\n` ` verify` ` I` `Final` ` This` `.` ` (` ` Let` `</think>` | ` Wait` ` is` `Wait` ` No` ` $` ` I` | `.”` ` Thus` `,”` `.).` | ` Wait` `"?` `**?` `?` ` wait` |
| 18 | `\n\n` ` I` ` So` `.` `$.` ` Okay` ` This` `Final` ` Let` | ` is` `Wait` ` The` ` maybe` ` correct` ` If` | `.”` ` Thus` ` dozen` ` Seems` | ` Wait` `...**` ` WAIT` `**?` |
| 20 | `Let` ` I` `.` ` (` ` So` `\n\n` `Final` | ` is` `Wait` ` I` ` The` ` maybe` ` If` `'m` | `.,` `“` `.</` `.).` | digits, ` Is` `Wait` ` WAIT` |
| 23 | `Let` `.` ` Wait` ` I` `=` `\n\n` ` This` | ` is` ` -` ` I` ` The` `'m` ` over` | `$` ` This` ` Let` `</think>` | ` *` ` Wait` ` Is` `\n\n` |

Bare J-lens reads (R0) agree: +u = "still / actually / is it? / not" (还是, 其实是, 不是, Actually, **?), −u = result nouns and closure.
So in the mid layers **u is a "doubt / re-check (Wait, is, maybe, If, No)" vs "close and move on (\n\n, So, Final, Let, </think>, verify)" axis**.
"Keep enumerating" is not what +u reads as; "re-question the same thing" is. Caveat: the sign holds at non-loop positions too, with
85–90 % of the loop-position size (cls(−u) − cls(+u)) (e.g. R-lens hs16 cls −u/+u: loop +1.65/−0.67, non-loop +1.06/−0.90). The axis is generic; the
gating that makes the edit loop-specific is in the neurons (Q2), not in the direction.

**P2b (variance): PASS.** var(h·u15)/tr Cov(h) = 3.5–5.1 % at hs13–22 (30–53× the isotropic 1/1024); u15 is the 2nd–3rd eigendirection
of the residual covariance at hs13–22 (top PC 10–13 %). u19's share peaks at hs20 (6.1 %). Mean h·u15 is 1.6–2.4× larger at loop than
at non-loop generated positions at every layer 13–24 (hs16: 2.25 vs 0.94; WikiText 0.53). Both shares collapse after the final block
(hs24 pre-norm: 0.46 %, rank 34). u is one of the dominant directions of the mid-layer residual, not a small-variance one.

**P2c (Δresid read): mid-layer PASS, "decorrelates at the end" FAIL (causally).** Pearson over the vocabulary between the lens read of
the edit's Δresid at loop positions and the same lens's −s·u read: 1.00 (hs16), 0.84–0.94 (hs17–18), 0.77–0.84 (hs19) for all lenses
(> 0.5 ✓). At the final layer the logit-lens read of −u decorrelates (0.23) as predicted, but **the edit's actual final Δlog-prob
correlates 0.98 with the R-lens effect of −s·u15 injected at hs16** (0.93 at hs17 … 0.59 at hs23). So the final logit change is almost
exactly what a −u push after L15 *causes* through the remaining blocks; the "only 10 % of the final Δ is along u" in Q3b is the
direct-path geometry, not the causal content. The edit's final effect (tuned / logit, Δp): up `\n\n` ` verify` ` This` `</think>` ` Let`
` I`; down ` Wait` `\n` ` check` `Wait` ` has` ` is`. Sanity: log-prob of the actual loop continuation token at loop positions −0.068
(loop gets less likely, cf. Q4).

Figures: `notes/fig/test2_killarney_variance.png`, `notes/fig/test2_killarney_axis_dresid.png`.

Caveats: the R-lens adds ±s·u at *all* generated positions (ungated), so it reads the direction, not the neurons' gating; one s (the
edit's median); token classes were fixed in advance but are hand-picked; J-lens fits are short-context (128 tokens) averages, ours on
400 prompts; tuned lens trained 2 epochs on 180 k positions (no held-out text overlap with the analysis set: different split / WikiText
article ranges); 40 looping texts, one seed.

## Compute
Killarney L40S, tests 2+4 together: **1.81 GPU-h** (collect 0.53, J-lens fit 0.81, tuned fit 0.16, read 0.10, 2 SAEs 0.12, analyze 0.06, pilot/failed 0.03). API: ~280 gateway calls (Sonnet 119 labels + 118 scores; Opus 20 + 20).
