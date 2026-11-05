# Q2 — When does the edit act? (teacher-forced on real unedited generations)

Script: `scripts/q2_neuron_activations.py` (6 looping + 6 productive unedited GSM8K generations, first 4096 tokens each,
"loop position" = a 32-gram that already occurred earlier in the text). Log: `notes/q2_neuron_activations.log`.

## +u is a loop direction in the residual stream

Residual stream after layer L, projected on u (mean per text; loop vs non-loop positions):

| text | L15 prod | L15 loop | L19 prod | L19 loop |
|---|---|---|---|---|
| looping texts (6) | 0.7–1.0 | 0.9–2.2 | 1.6–2.1 | 1.9–4.1 |
| productive texts (false-positive "loop" positions) | 0.5–0.8 | 0.6–1.0 | 1.3–1.8 | 1.4–2.3 |

In the texts with real loops (loop_frac ≥ 4 %: test_0, _20, _50) the u-component roughly doubles inside the loop.

## The 64 neurons are loop-selective and write +u

- Loop selectivity d' of |activation| (loop − productive positions, pooled std): L15 selected-64 mean **0.45** vs −0.04 for
  the other 3520 neurons; 25 of the 64 are in the model-wide top-64, 37 in top-256. L19 is weaker: 0.08 vs −0.11, 10 in top-64.
  So Astra's selection in L15 is close to "the most loop-selective neurons", L19 less so.
- Signed write of the 64 neurons along u (Σ_j a_j · uᵀw_j), base model: ~0.07–0.11 in productive positions,
  **0.19–0.55 in loop positions** (3–5× higher). All 3584 neurons together: 0.15–0.20 → 0.32–0.64.
- After the edit (−5× projection) the same 64 neurons write **−0.26 … −0.49 in productive positions and −0.8 … −2.7 in loops**.
  The whole MLP's write along u flips sign everywhere, but the push is ~4× stronger inside a loop.

## Reading

The edit is a self-gated anti-loop force: the neurons' read side (unchanged) already fires more when the model is repeating,
and the edit turns their write from "+loop direction" into "−5 × loop direction". It is not a "stop thinking" switch:
in productive positions the push is small, which matches Astra's report that finished outputs were unchanged and
edited correct outputs still think for a median 9k tokens.

Caveats: 12 texts, one benchmark, a crude n-gram loop detector, first 4096 tokens only (loops often start later).
Next: Q4 teacher-forced comparison of interventions (`scripts/q4_tf_interventions.py`), then Q3 (who reads u).
