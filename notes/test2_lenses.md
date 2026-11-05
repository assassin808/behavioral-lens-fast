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
