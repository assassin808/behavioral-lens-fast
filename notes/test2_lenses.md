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

(filled in below after the runs)
