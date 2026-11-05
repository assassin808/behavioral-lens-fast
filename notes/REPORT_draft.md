# What Astra's weight edit does to Qwen3.5-0.8B — draft report (branch `xuanqi`, 2026-11-05)

## 1. Claim

Astra's p50c edit is a closed-form, rank-1-per-layer rewrite of 128 MLP write vectors that turns loop-detecting neurons into a
self-gated anti-loop force. It removes the model's verbatim repetition failure and nothing else: on problems both models finish,
accuracy is unchanged, and all rescued problems are ones the unedited model had been cut off on. It is a behavioural bug fix,
located by Astra with a sound diagnosis, not a discovered reasoning mechanism; and it has measurable side effects.

## 2. The edit (Test 1 / Q1) — `notes/q1_what_changed.md`, `shared/`

w_j' = w_j − 6·u(uᵀw_j) for 64 columns of `down_proj` in L15 and L19 (both full-attention layers); ΔW rank-1 per layer
(σ = 1.75, 0.001, …); cos(u15, u19) = 0.85. Projection on u flipped and amplified 5×; read side untouched.
Astra's columns are index-skewed (median 364 / 354 of 3584; p < 1e-3) yet beat our re-rankings on the teacher-forced metric.
p10c (30 rank-1 reflections, α = 1.5) and p25c (ablate L15 head 7) are different edits; the "dose series" is a mechanism series.

## 3. Behaviour (Finding 1) — `notes/finding1_completion_vs_accuracy.md`, `notes/failure_classes.md`

| GSM8K, 200 problems | unedited | p50c |
|---|---|---|
| completion | 0.10 | 0.94 |
| accuracy on completed | 0.95 | 0.65 |
| accuracy on problems both completed (n = 21) | 0.952 | 0.952 |
| verbatim-loop endings | 172 | 3 |
| new failure modes (no box / whitespace / wrong) | 0 / 0 / 1 | 13 / 9 / 53 |

MATH-500 same pattern (completion 0.07 → 0.76; jointly completed 1.00 vs 0.92, n = 12; no-box 0 → 42).
Rescues: 99 / 79 / 36 / 42 / 10 on GSM8K / MATH / Minerva / Olympiad / AMC, every one a former cut-off. AIME: finished 2/3, still 0 %.

## 4. Mechanism (Q2–Q5) — `notes/q2_*.md`, `q3_*.md`, `q5_*.md`

- u is a loop direction: resid·u doubles inside loops; a base-trained probe reads "loop" at 0.91 accuracy from L12 on.
- The 64 neurons are loop-selective (L15 d′ 0.45 vs −0.04) and write +u 3–5× more in loops; after the edit −5× that.
- Δresid is 0.48 of the norm in loops vs 0.16 outside; pure −u after L15; converted by L22–L23 (10 % along u at the end).
  Copy attention unchanged; not a copy-head story. The probe reads edited loop states as not-loop (0.85 → 0.08) while their
  distance to productive states is unchanged: the loop flag is cleared, the state is not returned to normal reasoning.

## 5. Controls (Test 6) — `notes/q4_simpler_intervention.md`, `notes/q4b_tf_sweep.md`, screening table TBD

Teacher-forced (loop-position NLL, base 0.021): α = 1 +0.005, 2 +0.011, 4 +0.034, 6 +0.093, 8 +0.312; 20 random directions
≤ +0.0004; random columns ≤ +0.001; ungated −u bias needs 6× the collateral. Generation (50 GSM8K): base 0.14, α = 2 0.32, α = 6 0.56.
[screening table: `python scripts/summarize_screen.py` → base / p50c / α / top-64 / random / p10c / p25c]

## 6. Decoding baselines (Test 7) — screening table TBD

[no-think / repetition penalty 1.1, 1.2 / thinking budget 4k, 8k / u-steering c = 0.5, 1, 2 — `notes/screen/`, `notes/steer/`]
Question answered here: does any decoding-time fix match the edit at equal collateral?

## 7. Generality and side effects (Q6) — `notes/q6_side_effects_generality.md`

Same 4× loop-NLL effect on IFEval (unedited loops on 81 % of IFEval samples → IFEval 21 → 62); +4–10 % NLL on non-loop text;
no-box answers and whitespace degeneration are new. Gemma: no weights, null result.

## 8. Open (collaborator) — `notes/ISSUE_collaborator.md`

tuned/J-lens of u; full-context neuron ranking and auto-interp; cos(probe, u) and the probe on edited generations;
path patching L15h7 / L19 / L23 copy head; SAE/crosscoder (optional).

## 9. Caveats

Teacher-forced numbers come from 6–8 looping texts × 4096 tokens; one seed; Astra's logs give one sample per problem.
Screening uses 200 + 200 problems × 2 samples × 16k; only base/p50c are run at the full 32k × 4 protocol.
