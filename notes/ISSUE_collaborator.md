# 合作者任务：Astra 编辑的机制分析（测试 2 / 3 / 8 / 5 / 4）

## 背景（两分钟版）

Astra 对 Qwen3.5-0.8B 做的 p50c 编辑，我们已经还原成闭式：对 L15、L19 的 `mlp.down_proj` 各 64 列 j，

    w_j' = w_j − 6 · u (uᵀ w_j)

每层 ΔW 严格 rank-1，两层的 u 余弦 0.85，基本是同一个方向。+u 是残差流里的"我在循环"方向（循环段 resid·u 翻倍），这 64 个神经元本身循环选择性强，原本在循环段向 +u 写得更多，编辑后变成写 −5 倍。行为上它只是止住了逐字循环（两边都写完的题准确率完全一样，所有 rescue 都是原来被截断的题），但带来了新副作用（写完不给 `\boxed{}`、空白退化）。

已完成的部分和全部脚本在分支 **`xuanqi`**（https://github.com/xuan7zhang/behavioral-lens/tree/xuanqi ，同步自本仓库 main）。总览见 `notes/SUMMARY.md`、执行计划见 `notes/PLAN_xuanqi.md`。生成类实验（解码基线、因果对照、α 扫描）由 GPU 这边跑，**下面这些机制分析不依赖生成，可以独立推进**。

## 共享约定

- 模型：`Qwen/Qwen3.5-0.8B`，revision `2fc06364715b967f1860aea9cf38778875588b17`。编辑后的权重用 `astra_logs/exp8_math/code/apply_edit.py <base safetensors> astra_logs/exp8_math/runs/p50c/edit <out.safetensors>` 重建（会校验 sha256）。注意 HF 下载下来的文件名是 `model.safetensors-00001-of-00001.safetensors`。
- 环境：transformers ≥ 5.11 才有 `qwen3_5`；`AutoModelForCausalLM.from_pretrained(path, dtype=torch.bfloat16).to("cuda")`（没有 accelerate 时不要用 device_map）。线性注意力层走 torch 回退实现，慢但能用；装 flash-linear-attention + causal-conv1d 会快很多。
- 思考模式要显式开：`apply_chat_template(..., enable_thinking=True)`；prompt 用 Astra 的 `mathqa.build_prompt`（题目 + 换行 + "Please reason step by step, and put your final answer within \boxed{}."）。
- 共享文件（`shared/`）：`u_L15.npy`、`u_L19.npy`（单位向量，+u = 循环方向）、`astra_cols_L15.npy`、`astra_cols_L19.npy`（Astra 改的 64 列）、`top64_by_dprime_L*.npy` / `top64_by_uproj_L*.npy` / `top64_combined_L*.npy`（我们的重排，注意：TF 指标上它们弱于 Astra 的列，见 `notes/q4b_tf_sweep.md`）。
- 循环语料：`astra_logs/exp8_math/eval/{unedited,p50c}/parts/*/*_first_samples.jsonl.gz`，每条是完整的 32k 生成（`id / answer / text`）。"循环位置"的定义：32-gram 之前出现过（`scripts/q2_neuron_activations.py` 里的 `loop_mask`）。GSM8K 题目在 `data/gsm8k_test.jsonl`。
- 层索引：`hidden_states[L]` 是第 L−1 层之后的残差；full attention 只有 3、7、11、15、19、23 层，其余是 linear attention（`output_attentions` 只返回这 6 层）。
- 产出：每个测试一个 `notes/<test>.md` + 脚本进 `scripts/`，提 PR 到 `xuanqi` 分支。数字请注明样本数和上下文长度。

## 任务

### 测试 2（P1）：用 logit / tuned / J-lens / R-lens 读出 u
已有：直接 logit lens（`notes/q1_what_changed.md`），+u 促进 `...*`、`她还是`，−u 促进 ` first`、` result`、` value`。但 Q3b 显示到最后一层只有 10% 的 Δ 还在 u 上，直接路径不是主路径，所以 logit lens 证据弱。
要做：在 ~1000 个 prompt 上拟合 tuned lens（或 J-lens / R-lens），在 L15–L23 各层读出 ±u 的 top-k token，看它是否在中间层就是"继续列举 / 复述"vs"给结论"的轴；对比 u 在每层残差流中的方差占比。

### 测试 3（P1）：128 个神经元的激活分析
已有：`notes/q2_when_does_it_act.md`（d′ 循环选择性、沿 u 的有符号写入，6+6 条文本、4096 token）。
要做：(a) 在更多、更长的上下文（至少 50 条循环 + 50 条正常，用完整长度或 16k）上重算每个神经元的 `act_j · (uᵀ w_j)` 在循环段 vs 正常段的差，给出真正的 top-64 并与 Astra 的列比；(b) 每个神经元的 top 激活片段（exemplars），非数学文本也要（IFEval / MMLU-Pro 样本在同一目录）；(c) 用 LLM 做 auto-interp 并打检测分（API 待定）。

### 测试 8（P1）：循环检测 probe 与 u 的关系
已有：`notes/q5_state_geometry.md`（base 上训的线性 probe 准确率 0.91，编辑后循环态 P(loop) 0.85 → 0.08，但到正常簇中心的距离不变）。
要做：(a) cos(probe 方向, u)（注意 probe 是在标准化空间训的，要变换回原空间）；(b) 用 probe 看 **编辑后模型自己的生成**：它是否仍然进入循环态然后被踢出来（p50c 的样本在同一目录）；(c) PCA / 2D 图：循环态 vs 正常态 vs 编辑后循环态。

### 测试 5（P1）：attribution / path patching
已有：`notes/q3_who_reads_it.md`（Δresid 循环段 0.48 vs 正常段 0.16；L19/L23 的 copy attention 几乎不变；L22–23 把 −u 转成其它方向）。Astra 的 p25c 砍的是 L15 head 7（`astra_logs/exp8_math/runs/p25c/final_answer.md`）。
要做：对 L15 h7、L19 的各 head、L23 的 copy head（h6 exact-copy 最强）做 attribution patching / path patching，指标用循环位置上"重复 token 的 log-prob"（`scripts/q4_tf_interventions.py` 里的 NLL-at-loop-positions）。问题：编辑的效果经过哪条路径到 logits——L22/L23 的 attention 还是 MLP？注意 hybrid 架构，TransformerLens 不支持，用 HF hook 自己做。

### 测试 4（P2，可砍）：SAE / crosscoder
在 `hidden_states[16]` 和 `hidden_states[20]` 上训 SAE，或 base/edited 上训 crosscoder，看 u 对应哪些 feature，以及编辑后哪些 feature 被打开/关闭。训练语料可以用循环语料 + 普通文本。

## 联系
问题直接在这个 issue 下留言。GPU 这边的筛选评测结果会陆续写进 `notes/screen/`。
