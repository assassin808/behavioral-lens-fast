# Q0 — Smoke test: base vs p50c generation (HF, thinking on, Astra's sampling, 4096-token cap, 3 GSM8K problems)

Script `scripts/smoke_generate.py`; log `notes/smoke_generate_think_base_p50c.log`.
- Thinking must be enabled explicitly: `apply_chat_template(..., enable_thinking=True)`; the default template inserts an empty think block.
  Thinking off: both models answer in 250–500 tokens, no loops (the whole phenomenon lives in thinking mode).
- Thinking on, base: 1/3 finished (1448 tokens); 2/3 hit the cap in verbatim loops (rep20 = 0.16 and 0.63; "Wait, 12 × 50 = 600 ... Wait, ..." forever).
- Thinking on, p50c: rep20 = 0.01 / 0.01 / 0.04 — verbatim repetition gone — but 2/3 still unfinished at 4096 tokens: it re-checks the same
  arithmetic in *paraphrase* ("Wait, let me check the math again"). Consistent with Astra: edited correct outputs have median 9k thinking tokens.
- Speed: ~53 tok/s in HF with the torch fallback for linear attention (flash-linear-attention / causal-conv1d not installed).
  A 32k-token sample ≈ 10 min, so the benchmark-scale evals for Q6 need vLLM (0.22.1 in the `vllm` env), not HF.
