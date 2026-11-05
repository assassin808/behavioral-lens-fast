# Q1 — What was changed? (p50c, the main edit)

Scripts: `scripts/q1_delta_w.py`, `scripts/q1_rank1_structure.py`. Data: `notes/q1_*.json`.
Edited checkpoints rebuilt with Astra's `apply_edit.py` (all sha256 OK): `/data2/ec2-user/models/qwen3.5-0.8b-astra/{base,p10c,p25c,p50c}`.

## The edit in closed form

For each of L15 and L19 (both are *full-attention* layers; the 24-layer model has full attention only at 3,7,11,15,19,23),
64 columns j of `mlp.down_proj` (= write vectors of 64 MLP neurons) are changed by

    w_j'  =  w_j − 6 · u (uᵀ w_j)          (fit residual 0.2–0.3 %, ratio a_j/p_j = −6.00 ± 0.002 for all 128 columns)

i.e. the component of each write vector along one unit direction u is flipped and amplified 5× (new projection = −5 × old).
Nothing else about the write vector changes; the read side (gate/up) is untouched.

- ΔW is exactly rank-1 per layer (svdvals 1.749, 0.001, ... / 1.399, 0.001, ...). The whole edit lives in **one residual-stream
  direction per layer**, and cos(u15, u19) = 0.85, so effectively **one direction** for both layers.
- u is dense (participation ratio ~430/1024), top dims {0, 5, 15, 18, 37}: not a single basis dimension.
- The 64 columns are *mostly* those with large |uᵀw_j| (chosen-column mean 0.031 vs all-column mean 0.007; median rank 67 / 126 among 3584)
  but not simply the top-64 (max rank 1113 / 2419) — Astra used an activation-contrast criterion (productive vs repetitive contexts) too.
- Per column: cos(old,new) 0.78 / 0.86 on average (min 0.31 / 0.05), ‖Δ‖/‖old‖ 0.70 / 0.52, ‖new‖/‖old‖ 1.18 / 1.11.

Astra's own words match: "counteract measured repetition directions", like p10c's rule W' = W − 1.5 u(uᵀW) but with α = 6 and
restricted to 64 columns.

## Logit lens of u (tied unembedding, final-norm scale applied; sign = +u)

- +u promotes: `...*`, `...**`, `她还是`/`他还是` ("she/he still"), `而是因为` ("but rather because"), junk sub-word tokens.
- −u promotes: ` first`, ` count`, ` yields`, ` result`, ` exact`, ` value`, ` numerical`, ` using`, ` basic`, `即可` ("that's enough / done").

So on the direct path u looks like a "keep-going / continue-the-list" vs "state the result" axis. Whether the 64 neurons
originally wrote +u or −u (and therefore which way the edit pushes) depends on the sign of their activations — to be measured
on real looping text (Q2). The direct-path lens is weak evidence; the main readers are probably later layers (Q3).

## The three "dose" runs are three different edits (README Q4 needs rewording)

| run | what changed | Astra's rule | MATH-500 |
|---|---|---|---|
| p10c | 30 tensors, L8–22, attn out_proj + mlp down_proj, rank-1 each | W' = W − 1.5 u(uᵀW) on "repetition directions" | 12.3 |
| p25c | L15 `self_attn.o_proj[:, 1792:2048]` zeroed | ablate attention head 7 of L15 (a copy/retrieval head) | 14.1 |
| p50c | L15+L19 `mlp.down_proj`, 64 cols each | w_j' = w_j − 6 u(uᵀw_j) | 48.1 |

So the series is not a dose series of one direction; it is three attempts at the same diagnosis (repetition loops) with different
mechanisms, and only the third one works well. Useful contrast for Q3/Q4: head-7 ablation (p25c) also targets copying but gains little.

## Data for later questions

`astra_logs/exp8_math/eval/{unedited,p50c}/parts/*/*_first_samples.jsonl.gz` hold full 32k-token generations
(keys id/answer/text). Unedited GSM8K shard 0: 4/20 close `</think>`, median 101k chars, tails are visible loops
("Wait, is it possible ..." with growing indentation). p50c: 18/20 close; one degenerate 3.5M-char whitespace run.
These are the natural "looping vs productive" corpus for Q2 and Q5.

## Gemma follow-up

Only final_answer.md files, no edit weights; and the result is null (+0.42 pp, 95% CI [−0.3, +1.1]). Q6's "same mechanism in
Gemma" cannot be checked and would not mean much.
