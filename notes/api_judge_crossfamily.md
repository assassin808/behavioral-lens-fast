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

## Results
(to be filled after the runs)
