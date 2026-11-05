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

## Results (filled 2026-11-05; predictions above unchanged)

**Runs.** Main: RunPod H100, n = 20 looping texts × 4096 tok (10 619 loop positions), `notes/test5_paths_full_runpod.json`.
Replication: Killarney L40S, n = 30 × 4096 (the same 20 + **10 held-out texts**, 10 431 loop positions), `test5_paths_full_killarney.json`.
Same texts on the two machines agree to ≤ 0.003 NLL (fla kernels vs. torch fallback). Patch identity error 0. Full tables:
`notes/test5_tables_full_runpod.md` (`scripts/test5_analyze.py`). Figures: `notes/fig/test5_{mediators,heads,atp}_full_runpod.png`
(+ `_full_killarney`, `_pilot_killarney`).

Loop NLL base 0.082 → p50c 0.141, **Δ_total = +0.059** (held-out 10: +0.079). Productive NLL 0.401 → 0.437. Edit L15 only:
60 % of Δ_total. Edit L19 only: 31 %.

### Mediator table (% of Δ_total; block = p50c run with X ← base; inject = base run with X ← p50c)

| component | block, main [95 % CI] | block, held-out | inject, main | noise ctrl | perm ctrl | direct (last hop) |
|---|---|---|---|---|---|---|
| L15 MLP (edit site) | 69 [48, 94] | 80 | 60 | 4 | 34 | 3 |
| L19 MLP (edit site + reader) | 56 [38, 80] | 58 | 14 | 0 | 10 | 3 |
| **all mixers L16–23** | **91 [67, 118]** | 96 | 53 | 11 | 35 | – |
| full-attn L19 + L23 | 76 [53, 103] | 85 | 22 | 6 | 8 | – |
| L23 attention | 58 [37, 82] | 56 | 13 | 4 | 8 | **71** |
| L19 attention | 43 [26, 64] | 54 | 4 | 1 | 1 | 16 |
| linear-attn mixers L16–21 (incl. L19) | 69 [47, 95] | 81 | 18 | 3 | 10 | – |
| **MLPs L20–23** | **−55 [−89, −25]** | −41 | 28 | 15 | 12 | – |
| L23 MLP | −61 [−87, −35] | −56 | 5 | 16 | 15 | 5 |
| MLPs L16–18 | −7 [−23, 9] | −42 | 3 | 2 | 3 | – |
| everything downstream except L19 MLP | 97 [73, 124] | 97 | 132 | 109 | 98 | – |
| top single heads: L23 h7 / L16 h6 / L19 h4 / L18 h14 / L23 h3 | 29 / 19 / 17 / 17 / 16 | 28 / 26 / 32 / 23 / – | ≤ 3 | ≈ 0 | ≈ 0 | 0–6 |
| L23 h5 (strongest induction head) | 11 [3, 18] | 20 | 6 | 1 | 1 | 27 |
| L23 h6 (duplicate-token head, "copy head" in Q3) | 12 [6, 18] | 11 | 0 | 0 | 1 | 4 |
| negative heads: L23 h2 / L22 h8 | −14 / −14 | −23 / −15 | ≈ 0 | | | |

The direct last-hop effect of the edit sites' own Δ is 3 % (`direct_edit_sites` +0.0019; held-out +0.0018). Blocking is
strongly non-additive: the single-head blocks sum to 226 %. All 30 texts show a positive block for the mixers L16–23, and 26 of
30 show a negative block for MLPs L20–23.

### Mechanism readouts
- **Attention gets less induction from the earlier copy.** Attention from loop position p to the earlier position that holds
  the repeated token (induction target) changes base → p50c as follows. L23 summed over heads: 0.242 → 0.160 (−34 %; held-out
  −38 %). The largest drops are h5 0.071 → 0.035, h0 0.042 → 0.025, h4 0.036 → 0.024 and h7 0.039 → 0.023. L19 drops −13 %
  (held-out −22 %). L15 is unchanged, as it must be (upstream of the edit). Duplicate-token attention (to the earlier copy of
  the *current* token, e.g. L23 h6 0.14) is unchanged. Q3 measured that offset, which is why it found "copy attention
  unchanged" (see Q3 correction below).
- **Query side, not key side.** Injecting the p50c mixer outputs only at loop positions reproduces the whole inject effect
  (L23: 13.0 vs 13.0 %; L15 MLP: 57 vs 60 %). Injecting at all other positions gives ≈ 0. The edit changes what the loop
  position *asks for*; the keys and values of the earlier copy are unchanged.
- **The last hop suppresses the specific repeated token.** Logit lens of Δ of the L23 attention output: the centred Δlogit
  of the repeated token is −3.1, and the target sits at the 0th percentile of the vocabulary at the median loop position, i.e.
  it is the single most-suppressed token. The same holds for L23 h5 (−1.3, 0th pct). The whole final Δ gives −4.0 (4th pct);
  held-out gives −4.1 for attn23 and −5.6 for the whole Δ. The edit sites' own Δ is not target-specific: L15 MLP +0.68 (93rd
  pct, i.e. it slightly *raises* the target directly), L19 MLP +0.03.
- **A weaker consistent "conclude" component exists.** The whole final Δ also promotes ` result` +3.1, ` value` +2.2,
  ` Thus` +1.8 and ` answer` +0.9 (centred logits), and suppresses ` Wait` −2.7, ` Hmm` −2.0, `\n\n` −1.6 and `**` −1.7.
  `</think>` is −0.5, so the edit does not push toward closing the thought. This component comes mostly from the L15 MLP's
  direct −u write (` first`/` result`/` value`/` Thus`, consistency 0.98), but that direct path is only 3 % of the loop-NLL effect.
- **The late MLPs counteract the edit (self-repair).** In the p50c run, restoring the L22–23 MLPs to their base outputs makes
  the edit's effect 79 % *larger*. These MLPs partly undo the attention-borne suppression. Caveat: L23 MLP is high-norm, and its
  norm-matched noise injection also costs 16 %, so part of the block effect may be an off-distribution mismatch.
- **AtP is a rough guide only.** Spearman ρ against exact inject over 137 components is 0.67 (base gradient) and 0.62 (p50c
  gradient). For L15 MLP, AtP gives 0.011 from the base gradient and 0.145 from the p50c gradient, against an exact 0.036
  (3× under, 4× over). The perturbation is too large for a linear estimate.

### L15 h7 (Astra p25c's head)
It is the strongest induction head of L15 (0.128 vs ≤ 0.05 for the other heads). Ablating it in base raises loop NLL by
+0.013 (22 % of p50c) and productive NLL by +0.015, i.e. unselectively; the control L15 h3 gives +0.006. It does **not** feed the
detector: the 64 L15 neurons' +u write at loop positions changes only −3 % when h7 is ablated. Ablating it **on top of p50c**
adds +0.099 (held-out +0.157), 7–14× its effect in base. Once p50c weakens the late induction path, the remaining confidence
in the repeated token depends heavily on the early L15 h7 copy path. The two edits target complementary copy routes.

### Predictions
| | outcome |
|---|---|
| P1 MLP > attention | **FAIL (opposite).** Mixers L16–23 mediate 91 %; MLPs L20–23 *counteract* (−55 %). |
| P2 direct path < 25 % | **PASS.** 3 %. |
| P3 no head > 20 %, L23 h6 < 10 % | **FAIL.** L23 h7 29 % [14, 48]; L23 h6 12 % [6, 18]. Mediation is spread across ~10 heads and non-additive. |
| P4 stop-direction rather than anti-copy | **FAIL.** By the pre-registered rule the result is *anti-copy*: the top mediator, L23 attention, has target Δlogit −3.1 at the 0th percentile, and L23 induction attention drops 34 % (> 20 %). Consistency > 0.3 held (0.74), and a weaker conclude/anti-"Wait" component exists. |
| P5 AtP ρ > 0.5 but top MLP off by > 2× | **PASS.** ρ 0.62–0.67; L15 MLP off by 3–4×. |
| P6 L15 h7 | **Partial.** < 50 % of p50c: pass (22 %). Strongest L15 induction head: pass. Feeds the detector (> 20 % drop in u-write): **fail** (−3 %). |
| Controls < 20 % of real inject | **Pass** for edit sites and single heads (L15 MLP noise 4 vs 60). **Fail** for late layers (L23 MLP noise 16 vs 5; L22–23 33 vs 97; all downstream 109 vs 132). Inject into the last layers is mostly generic disruption, so block is the reliable measure there. |

### Main path, in one sentence
The 64 loop-gated neurons write −u at the loop position itself (L15, then L19 also responds). This changes the *queries* of
the downstream attention, mainly L19 and L23 full attention plus several linear-attention heads in L16–22, so they attend
~1/3 less to the earlier copy of the repeated span (induction). L23 attention's output then specifically drops the
repeated token's logit, while the L22–23 MLPs partially undo it. The direct −u → unembedding path contributes ~3 %.

### Correction to Q3
Q3 measured attention from a loop token to the earlier copy of *itself* (duplicate-token offset), which does not change.
The token being copied is one position later. Measured at that induction offset, L23 copy attention drops by 34 %.
"Copy attention barely changes" in `q3_who_reads_it.md` / SUMMARY should read "duplicate-token attention is unchanged;
induction attention in L23 drops ~1/3 and is the main last hop".

### Caveats
Teacher-forced proxy on base (unedited) loop texts. 4096-token contexts only, 30 texts (20 main + 10 held-out), one prompt
format. Block and inject are far from additive (large perturbation), so the "%" are per-component necessity/sufficiency,
not a partition. Linear-attention heads have no attention pattern to read, so the induction readout covers full-attention
layers only. Block of the high-norm late MLPs may partly reflect off-distribution mismatch (see noise control).

GPU: Killarney L40S ≈ 0.9 GPU-h (smoke 3 min, pilot 13 min, full 36 min), RunPod H100 ≈ 0.75 GPU-h (coordinator). Total ≈ 1.6 GPU-h.
