# Test 5 — Through which path does the p50c edit reach the logits at loop positions?

Branch `yang/t5`. Script `scripts/test5_paths.py`. Teacher-forced, base vs p50c, on unedited (base) thinking-on generations that
loop (GSM8K + MATH500, no `</think>`, 32-gram loop fraction > 3 %), first 4096 tokens. Metric (same as Q4): mean NLL of the actual
next token at loop positions (= the repeated token), mean per text, then mean over texts. Δ_total = M(p50c) − M(base).

Components downstream of the edit: per head of every token mixer L16–L23 (linear attention L16–18, 20–22: 16 heads × 128;
full attention L19, L23: 8 heads × 256; a "head" = its slice of the `o_proj`/`out_proj` input, after the output gate / gated norm),
whole mixer blocks, MLP blocks L16–L23 (L19 MLP is also an edit site), L15 MLP (the other edit site).

Patching definitions (all positions patched unless stated):
- **block_X** (necessity): p50c run with X's activation replaced by its base value; mediated = M(p50c) − M(p50c, X←base).
- **inject_X** (sufficiency / path through X): base run with X ← its p50c value (all paths X → logits); = M(base, X←p50c) − M(base).
- **direct_X** (last hop only): final pre-norm residual of base + Δ_X (resid-space), re-normed, unembedded.
- **AtP_X** = ∇_X M · (a_X^p50c − a_X^base), gradient at base and at p50c (report both).
- Controls for inject_X: (i) Gaussian noise with the same per-position norm as Δ_X; (ii) the real Δ_X with positions permuted.

## Pre-registered predictions (written 2026-11-05 before any run; not edited afterwards)

- **P1 (MLP > attention).** Blocking all MLPs L20–L23 removes more of Δ_total than blocking all mixers L16–L23. Blocking all
  downstream mixers removes < 50 % of Δ_total.
- **P2 (direct path minor).** The direct (last-hop) effect of the edit sites' own Δ (L15 MLP + L19 MLP) is < 25 % of Δ_total.
- **P3 (no single head).** No single attention head mediates > 20 % of Δ_total (block_X). L23 h6 (strongest copy head) < 10 %.
- **P4 (stop-direction rather than anti-copy).** For the top mediating components, Δ_X is position-consistent (mean cos(Δ_X,t,
  mean Δ_X) > 0.3 at loop positions) and its logit lens promotes a fixed token set; the induction/copy attention mass of the full
  attention heads at L19/L23 changes by < 20 % relative. Decision rule: "anti-copy" if the top mediators' direct effect is mainly
  target-specific (centered Δlogit of the target < −1 and target in the bottom 1 % of Δlogit at the median position), "stop/
  conclude" if the centered target Δlogit is > −0.5 and the promoted tokens are discourse/conclusion tokens.
- **P5 (AtP is a poor guide here).** Because the perturbation is large (‖Δ‖ ≈ half the residual norm), AtP and exact inject
  agree only coarsely: Spearman ρ over components > 0.5 but AtP misestimates the top MLP effects by > 2×.
- **P6 (L15 h7).** Ablating L15 h7 in base raises loop NLL by less than p50c does (< 50 % of Δ_total); h7 has the highest
  induction-attention mass of the L15 heads; ablating it lowers the 64 L15 neurons' +u write at loop positions by > 20 %
  (i.e. h7 is upstream of / feeds the detector rather than being a parallel path).
- Controls: norm-matched noise and position-permuted Δ injected at any component give < 20 % of that component's real inject effect.

Kill rule: if the pilot's Δ_total is not within ±50 % of Q4's +0.093 on comparable texts, or patching X←own cache does not reproduce
the unpatched metric to 1e-3, stop and debug before scaling.

## Results
(to be filled after the runs)
