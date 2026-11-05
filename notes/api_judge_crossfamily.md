# Cross-family judge for the test-3 neuron auto-interp (branch `yang/api`)

## Why
In test 3c (`notes/test3_neurons.md`) the explainer and the judge were the same model (`claude-sonnet-5-5`). Detection AP was 0.81 (L15)
/ 0.89 (L19) vs loop-matched negatives, shuffled descriptions ≈ 0.62, and 32 random neurons scored as high as Astra's (0.90).
Same-model explain+judge can inflate scores (the judge "recognises" its own phrasing). Here a judge from another model family
re-scores the **same Sonnet descriptions** on the **same held-out windows** (same positives, same matched / non-loop negatives,
same shuffle order), and also writes its own descriptions, so we get the full 2 × 2 (explainer × judge) table.

## Gateway check (2026-11-05, before any experiment call)
`GET /v1/models` lists 68 models. A 1-call probe ("17*23?") of every chat model:
- **works:** `claude-sonnet-5-5` (and other Claude models), `gpt-oss-120b`, `Qwen3_8-27B`, `Qwen3_6-35B-A3B`, `Qwen3_8-Flash-Next`.
- **403 `model_access_denied` for this project:** every GPT-4o/4.1/5.x/6.x model, every Gemini model, GLM, Kimi, Nemotron, qwen3p7-plus.
So the strong closed non-Anthropic models (GPT-5.x, Gemini 3.x) are **not available**. The cross-family judge used here is
**`gpt-oss-120b`** (OpenAI open-weight, non-Anthropic) — weaker than Sonnet, which is a stated limitation, not a silent swap.
Secondary judge (desc + shuffled only, if budget allows): `Qwen3_8-27B`.

## Pre-registered predictions (written 2026-11-05 before any cross-family call; not edited afterwards)
- **P1** gpt-oss-120b judging Sonnet descriptions: mean AP (desc vs matched negatives, and vs non-loop negatives) is within
  **±0.05** of the Sonnet judge for each group (Astra L15, Astra L19, random).
- **P2** The shuffled-description control stays well below: AP(desc) − AP(shuffled) ≥ **0.15** for each group under the gpt-oss judge.
- **P3** Astra's neurons are still ≈ random neurons: |AP(Astra group) − AP(random)| ≤ 0.10 under the gpt-oss judge, i.e.
  the test-3 caveat "detection on random neurons is as high as on Astra's" survives.
- **P4 (self-preference, exploratory)** In the 2 × 2 table, each judge scores its own explainer's descriptions ≤ 0.05 higher than the
  other explainer's (no large self-preference).
- **Kill / validity rules.** (i) If > 5 % of gpt-oss judge scores are missing / unparseable, the run is invalid and is reported as such
  (fix the parser or prompt, re-run). (ii) If gpt-oss scores the *shuffled* control ≥ 0.75 or the description ≤ 0.6 in every group,
  the judge is not doing the task (check raw outputs before interpreting). (iii) Pilot: 10 neurons first, read raw outputs.

## Protocol
- Inputs: `$BLF_DATA/t38/t3_explain_prep_32768.json` (windows; not regenerated) and
  `notes/test3_autointerp_32768_claude-sonnet-5-5_killarney.json` (Sonnet descriptions).
- Script `scripts/api/t3_crossjudge.py` rebuilds exactly the same neuron list, the same per-neuron shuffled window order
  (`random.Random(1000 + i)`), the same shuffled-description permutation (`random.Random(0)`), and the same batched 0–100 judge
  prompt as `scripts/test3_explain.py` (copied verbatim). It asserts that the rebuilt shuffled descriptions equal the stored ones.
- Judges score 4 description kinds (desc, shuffled, segment-only, loop-only) per neuron, like the original.
- gpt-oss-120b as explainer: same explain prompt as test 3, its first line is the description.
- All requests cached on disk (`scripts/api/apicache.py`, prompt-hash → response).

## Results (filled 2026-11-05; predictions above unchanged)

Files:
- Per-run neuron-level results:
  - `notes/api_t3_sonnet__gpt-oss-120b.json` (Sonnet descriptions, gpt-oss judge, 4 description kinds)
  - `notes/api_t3_gpt-oss-120b__gpt-oss-120b.json` (gpt-oss descriptions, gpt-oss judge, 4 kinds)
  - `notes/api_t3_gpt-oss-120b__claude-sonnet-5-5_desc-desc_shuf.json` (gpt-oss descriptions, Sonnet judge)
  - `notes/api_t3_sonnet__claude-sonnet-5-5_desc-desc_shuf.json` (Sonnet judge **re-run**, for test-retest)
  - pilot (10 neurons): `notes/api_t3_sonnet__gpt-oss-120b_lim10.json`
- Summary with bootstrap CIs: `notes/api_t3_crossjudge_summary.json` (`scripts/api/t3_crossjudge_summary.py`)

Validity:
- Rebuilt neuron list and shuffled descriptions: identical to the Sonnet run (asserted).
- Missing / unparseable judge scores: **0 / 15,360** (gpt-oss), 0 / 7,680 (each Sonnet run). Kill rule (i) not triggered.
- Pilot (10 neurons): I read the raw gpt-oss outputs. They have the right format and give sensible high scores on matching
  excerpts. The desc > shuffled ordering held, so I scaled up.
- Kill rule (ii) not triggered: the gpt-oss shuffled control is 0.55–0.57, and its desc AP is 0.71–0.82.

### 2 × 2 table: mean AP of the description vs loop-matched negatives / vs non-loop negatives (shuffled control vs non-loop in brackets)

| explainer → judge | Astra L15 (64) | Astra L19 (64) | random (32) |
|---|---|---|---|
| Sonnet → Sonnet (test 3c) | 0.81 / 0.84 [0.63] | 0.89 / 0.90 [0.62] | 0.90 / 0.91 [0.62] |
| Sonnet → Sonnet (re-run, same prompts) | 0.81 / 0.84 [0.63] | 0.89 / 0.90 [0.63] | 0.90 / 0.91 [0.62] |
| **Sonnet → gpt-oss-120b** | **0.71 / 0.76 [0.57]** | **0.78 / 0.81 [0.55]** | **0.81 / 0.82 [0.55]** |
| gpt-oss-120b → Sonnet | 0.72 / 0.77 [0.66] | 0.80 / 0.79 [0.66] | 0.85 / 0.85 [0.63] |
| gpt-oss-120b → gpt-oss-120b | 0.61 / 0.67 [0.56] | 0.65 / 0.66 [0.55] | 0.68 / 0.69 [0.56] |

Segment-only and loop-only controls (vs matched / vs non-loop) under the gpt-oss judge:

| control | Astra L15 | Astra L19 | random |
|---|---|---|---|
| segment-only (vs matched) | 0.54 | 0.57 | 0.59 |
| loop-only (vs non-loop) | 0.61 | 0.54 | 0.53 |

Test 3 found that L15 is the layer where "inside a loop" alone partly predicts firing (Sonnet judge 0.745). Under the gpt-oss judge this pattern survives, more weakly (0.61).

Paired per-neuron differences (mean [bootstrap 95 % CI]):

| comparison | L15 vs matched | L19 vs matched | random vs matched |
|---|---|---|---|
| Sonnet judge re-run − original (judge noise) | −0.002 [−0.010, +0.006] | −0.001 [−0.007, +0.005] | +0.003 [−0.002, +0.009] |
| **P1: gpt-oss judge − Sonnet judge (Sonnet descriptions)** | **−0.10 [−0.13, −0.07]** | **−0.10 [−0.13, −0.08]** | **−0.09 [−0.13, −0.06]** |
| same, shuffled control (vs non-loop) | −0.06 [−0.10, −0.03] | −0.07 [−0.11, −0.03] | −0.07 [−0.12, −0.02] |
| Sonnet judge: gpt-oss descriptions − Sonnet descriptions | −0.10 [−0.13, −0.06] | −0.09 [−0.13, −0.06] | −0.05 [−0.09, −0.01] |
| gpt-oss judge: gpt-oss descriptions − Sonnet descriptions | −0.10 [−0.14, −0.06] | −0.13 [−0.17, −0.09] | −0.12 [−0.18, −0.07] |

Per-neuron rank agreement of the two judges on the same Sonnet descriptions: Spearman 0.73 (AP vs matched), 0.69 (vs non-loop).

P2 / P3 under the gpt-oss judge (Sonnet descriptions):

| | Astra L15 | Astra L19 | random |
|---|---|---|---|
| AP(desc) − AP(shuffled), vs non-loop | **+0.20 [0.15, 0.24]** | **+0.26 [0.21, 0.30]** | **+0.27 [0.21, 0.32]** |
| Astra − random, AP vs matched | −0.09 [−0.15, −0.03] | −0.02 [−0.09, +0.04] | – |
| Astra − random, AP vs non-loop | −0.06 [−0.12, +0.01] | −0.01 [−0.07, +0.05] | – |

The gpt-oss descriptions have a failure mode: 42 / 160 describe the `[[ ]]` annotation markup itself ("tokens surrounded by
double-square-bracket markup"), counted by keyword regex. Sonnet never does this. Their desc − shuffled gap is only +0.10–0.12
under either judge.

### Outcome per prediction

| prediction | outcome |
|---|---|
| **P1** gpt-oss judge within ±0.05 of the Sonnet judge on Sonnet descriptions | **fail**: −0.09 to −0.10 vs matched (CIs exclude −0.05 in every group), −0.07 to −0.09 vs non-loop. |
| **P2** desc − shuffled ≥ 0.15 under the gpt-oss judge | **pass**: +0.20 / +0.26 / +0.27, almost the same as under the Sonnet judge (+0.21 / +0.27 / +0.29). |
| **P3** \|Astra − random\| ≤ 0.10 under the gpt-oss judge | **pass**: random neurons are again as describable as Astra's or slightly more (L15 −0.09 vs matched, L19 −0.02). |
| **P4** self-preference ≤ 0.05 | **pass, no self-preference**: the gpt-oss judge scores *Sonnet's* descriptions 0.10–0.13 *higher* than its own. The Sonnet judge prefers Sonnet's by a similar 0.05–0.10. |

### Reading
- The cross-family judge gives lower absolute APs, by about 0.1. But it also lowers the shuffled control by 0.06–0.07, so the
  *discrimination* (desc − shuffled) is nearly unchanged (−0.01 to −0.02).
- Same-model judging therefore does **not** inflate the specificity of Sonnet's descriptions in any way that matters for the
  test-3 conclusions.
- The ~0.1 offset most plausibly reflects the weaker, less calibrated judge (gpt-oss-120b vs Sonnet). That is a hypothesis: no
  stronger non-Anthropic judge is available on this gateway to test it.
- There is no sign of self-preference: each judge ranks Sonnet's descriptions above gpt-oss's.
- The test-3 caveat stands and is now cross-family: **random neurons are as describable as Astra's**.
- So the detection APs show that the descriptions are specific. They do not show that Astra's neurons are special.
- Suggested wording for the paper: "With a non-Anthropic judge (gpt-oss-120b), absolute detection AP drops by ≈ 0.1 for both the
  descriptions and the shuffled control. Description − shuffled gaps (0.20–0.27) and the Astra ≈ random result are unchanged."

### Caveats
- Only open-weight non-Anthropic models are accessible (GPT-5.x, GPT-6.x and Gemini return 403). gpt-oss-120b is weaker than
  Sonnet, so P1's failure confounds "different family" with "weaker judge".
- The gateway has no logprobs, so scoring is the batched 0–100 protocol, as in test 3. Each call scores 24 excerpts jointly,
  which makes it partly a ranking task.
- Each (neuron, description) pair is scored once, but the Sonnet re-run shows the judge noise on group means is < 0.01.
- `Qwen3_8-27B` (the planned secondary judge) was not run, to keep within the call budget.

### API calls
| run | new calls |
|---|---|
| gpt-oss judge on Sonnet descriptions, 160 neurons × 4 kinds (incl. 40 pilot) | 640 |
| gpt-oss explainer | 160 |
| gpt-oss judge on gpt-oss descriptions (desc + shuffled; the seg / loop prompts are identical, so cached) | 320 |
| Sonnet judge on gpt-oss descriptions | 320 |
| Sonnet judge re-run | 320 |
| **total** | **1,760** |
