# behavioral-lens-fast

A short side project: look at the MIT Astra weight edits with as many interpretability tools as possible,
and say in plain words what the edit actually does inside the model.

> **Results: [`notes/RESULTS.md`](notes/RESULTS.md)**. It covers the causal chain, a table of tool × object × finding,
> the behaviour comparison, and corrections to the earlier `q1`–`q6` notes. Where they conflict, RESULTS.md is authoritative.
> The plan below is the original one; RESULTS.md shows which parts were done and what they found.

Astra (GPT-6 in a sandbox) edited Qwen3.5-0.8B's weights directly, with no training, and math accuracy jumped.
We want a clear, multi-tool account of what changed.

## What the edit is (from `astra_logs/exp8_math/runs/p50c/edit/manifest.json`)

- Only `layers.15.mlp.down_proj` and `layers.19.mlp.down_proj` change, 64 columns each.
  So the **write vectors of 128 MLP neurons** change. Their read side (`gate_proj` / `up_proj`) is untouched,
  which means *when* these neurons fire is exactly as before; only *what they write* is new.
- Relative change: 0.106 (L15), 0.081 (L19).
- Effect (thinking on, from `astra_logs/exp8_math/eval/RESULTS_excerpt.md`):

  | Model | MATH-500 | GSM8K | AMC23 | Minerva | Olympiad |
  |---|---|---|---|---|---|
  | unedited | 7.9 | 13.8 | 0.5 | 1.5 | 1.3 |
  | p10c (10 passes) | 12.3 | 25.7 | 1.1 | 3.9 | 2.4 |
  | p25c (25 passes) | 14.1 | 25.1 | 1.1 | 3.5 | 3.9 |
  | p50c (50 passes) | 48.1 | 67.1 | 23.4 | 25.4 | 21.8 |

- Astra's own story: the model fell into repetition loops; it contrasted productive vs repetitive contexts and
  edited the neurons. Almost all rescued problems (576/579) were ones the unedited model ran out of tokens on.
- Also in the logs: Gemma-3-270M follow-up (`exp8_math/gemma_followup`), sycophancy (`exp7`), language (`exp3`),
  Go from scratch (`exp6`).

## Six questions, each looked at with two or three independent tools

| # | Question | Tools | What we expect to see |
|---|---|---|---|
| 1 | What was changed? | ΔW per-column norms, SVD rank of ΔW; project Δw_out onto the vocabulary (logit lens / J-lens) | Is it one low-rank direction? Does it promote "stop thinking / answer / newline" and demote repeated tokens? |
| 2 | When does it act? | Activation exemplars of the 128 neurons + LLM explainer (API) with detection scoring; productive vs looping contexts | Are they "I am repeating" detectors or something broader? |
| 3 | Who reads what it writes? | Attribution patching (AtP*), path patching, attribution graphs (circuit-tracer if a transcoder exists, else neuron-level graphs); induction / copy-head attention before vs after | "Break the copy loop" or "trigger wrap-up"? |
| 4 | Is a simpler intervention enough? | Steering vector, rank-1 edit, bias only when looping; dose series p10c / p25c / p50c (same direction?); norm-matched random | If rank-1 gets most of the gain, the story is simple. |
| 5 | How do the internal states move? | Residual-stream geometry (PCA / clusters) before vs after; J-lens per layer along a loop | A picture of the state being pushed out of a "loop attractor". |
| 6 | Side effects and generality | MMLU-Pro / IFEval (in Astra's table), perplexity, repetition rate on non-math text; does the Gemma edit use the same mechanism? | Bug fix, or a real capability gain? |

Caveat: Qwen3.5 mixes linear-attention and full-attention layers, so off-the-shelf circuit tools may not support it.
Check hooks and vLLM support first.

## Layout

- `astra_logs/`: a copy of the public Astra logs (edits, code, eval excerpts, final answers), from
  https://github.com/RikiyaT/lm_understand-logs. Logs and data are CC BY 4.0, their scripts are MIT
  (see `astra_logs/LICENSE`). Credit: the authors of "LLMs can understand and write neural network weights".
  Rebuild an edited model with `astra_logs/exp8_math/code/apply_edit.py <base model.safetensors> <run>/edit <out>`
  (base: `Qwen/Qwen3.5-0.8B`, revision `2fc06364715b967f1860aea9cf38778875588b17`).
- `scripts/`: our analysis code (to come).
- `notes/`: findings, one file per question.
