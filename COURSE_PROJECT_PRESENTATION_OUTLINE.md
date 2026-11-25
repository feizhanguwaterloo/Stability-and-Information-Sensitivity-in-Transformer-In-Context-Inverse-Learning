# CS 886 presentation: story and 18-minute speaking guide

**Stability and Information Sensitivity in Transformer In-Context Inverse Learning**  
**Fei Zhang, University of Waterloo**

Version 2: 19 main slides, including the title, followed by 8 backup slides.
The main talk targets 18 minutes, excluding questions. These are rehearsal
targets, not a measured delivery. English speaking prompts and explicit
transitions appear in the LaTeX source as \note{...}; the audience PDF
hides those notes.

## 中文串讲：这份 slides 要讲的故事

研究问题是：一个 Transformer 能否既保持稳定，又充分利用决定答案的关键记录？
“替换一条记录”有两种不同含义。同一个任务内换一条记录，正确参数不变，
小响应可以是合理的。换掉唯一识别任务的记录，正确参数会变，准确恢复就要求足够响应。

先用一个贯穿例子建立直觉。两份 context 都有 m-1 条 (0,0)，最后一条分别是
(1,1) 和 (1,2)。正确斜率分别为 1 和 2。重复再多零点也不会补充斜率信息。
每份 context 都能识别自己的答案，但只差一条记录。

Theorem 1 回答“固定架构允许多大响应”。从平均 token 扰动 A/m 出发，
bounded softmax 控制扰动传播，各层有限放大系数相乘，mean pooling 和 head
给出 C/m。C 来源于输入范围和各层范数，固定模型下不随 m 改变。
范数算出的 C 很大，所以这项结论提供渐近限制，无法定量预测 m=64 的表现。

Theorem 2 回答“准确恢复至少需要多大响应”。两个正确答案的距离 D，
不超过“第一端误差 + 预测变化 Delta + 第二端误差”。预测变化太小，
至少一个端点就必须承担较大误差。这是一般的三角不等式必要条件。
两个 theorem 组合后，说明重复背景增长时，固定模型最终无法在这类识别 pair
的两端都保持任意小误差。随后验证同样的 pair 确实可以由有效经济调度观测构成。

实验依次回答四个问题，而不是按实验编号罗列结果：

1. 普通同任务替换时，稳定与准确能否同时存在？可以，D 的原始 Transformer
   在 m=64 的平均变化约 0.00213，相对误差 2.65%。
2. 数据能唯一识别参数，就一定能学好吗？不一定，F 中 nearly redundant
   context 的误差明显高于 well-conditioned context。
3. 同样一条记录、同样的改动，背景信息能否改变模型响应？可以，G 中
   Transformer 的 high/low leverage 响应比为 14.1。
4. 有了较大响应，是否足以恢复正确参数差异？在识别 pair 上仍不充分，
   m=64 的 Transformer 方向响应为 47.4%，ridge 为 99.9%。

最后再把 Theorem 2 接回实测数据。识别 pair 的目标距离约 1.265，
实测平均 Delta 约 0.603，所以平均较大端点误差至少约 0.331；
观察值为 0.432。这是用实测响应得到的几何约束，不需要使用巨大常数 C，
也不证明是 C/m 的上界导致了训练模型的表现。

结论是：Transformer 的输出会随 context 信息结构改变，但在识别信息高度集中时，
它的响应仍可能不足以准确恢复任务。贡献是架构验证、有效构造和受控实验之间的连接。
Theorem 1 是现有论证的架构适配，Theorem 2 是基础必要条件；
不把它们表述为两个全新的普遍原理。

## Main talk and transitions

| Time | Slide | Purpose and transition |
|---|---|---|
| 0:00-0:25 | 1. Title | State the question about a single identifying observation. |
| 0:25-1:05 | 2. When should one observation change the answer? | Separate a replacement that keeps the true task fixed from one that changes it. |
| 1:05-2:10 | 3. A single record can determine the slope | Introduce the running example. More repeated records do not replace identifying information. |
| 2:10-2:50 | 4. Possible response and required response | Give the two theorem roles before their technical statements. Delta is the same response in both. |
| 2:50-4:05 | 5. Learning cost parameters from dispatch observations | Define records, unknown costs, interior KKT equations and the frozen mean-pooled Transformer. |
| 4:05-4:55 | 6. Theorem 1 statement | State the fixed-model assumptions and replacement upper bound. |
| 4:55-6:10 | 7. Proof 1: bounded attention | Track average row perturbation and explain why bounded attention controls its spreading. |
| 6:10-7:10 | 8. Proof 1: the constant C | Compose the layer factors. Explain where C comes from and why its numerical value is too loose. |
| 7:10-8:30 | 9. Theorem 2 and its proof | Walk through the triangle inequality. Return to slopes 1 and 2 and the minimum response 0.8 for error at most 0.1. |
| 8:30-9:15 | 10. Combine the bounds | Explain the repeated-record worst-case limit and when the lower bound is positive. |
| 9:15-10:25 | 11. Valid dispatch construction | Verify that both contexts identify different valid tasks. This exact pair returns in slide 16. |
| 10:25-11:15 | 12. Experimental questions | State the four questions, comparison models, unseen-task splits and prediction protocol. |
| 11:15-12:10 | 13. Ordinary replacements | Small response can coexist with accurate recovery when the true task stays fixed. |
| 12:10-13:05 | 14. Nearly redundant contexts | Unique identification and equal record count do not ensure good learned recovery. |
| 13:05-14:10 | 15. Matched leverage | Hold the focal record and edit fixed. The background changes model responsiveness. |
| 14:10-15:25 | 16. Identifying pair | Return to slide 11. Compare directional response and actual endpoint error. |
| 15:25-16:20 | 17. Interpret measured response | Apply Theorem 2 to observed Delta rather than the loose C/m certificate. |
| 16:20-17:25 | 18. Contribution and scope | Separate prior reasoning, our concrete work and limitations. |
| 17:25-18:00 | 19. Conclusion | Answer the opening question with the strongest supported statement. |

If rehearsal runs long, shorten the protocol and architecture details.
Preserve the running example, the two theorem roles, the triangle proof
and the identification experiment. Avoid reading every plot tick or table
entry aloud. Explain what changed between each pair of experiments.

## Backup slides

- A: Full softmax-Jacobian attention perturbation step.
- B: LayerNorm derivative and its finite Lipschitz constant.
- C: All context-information regimes, including partial identification.
- D: Task-count controls under equal passes and equal updates.
- E: Multiple identifying records and telescoping replacement paths.
- F: Independent Gaussian noise on every observed price.
- G: Supporting task-level covering-number generalization bound.
- H: Selected references and reproducibility.

The main story follows information, response and recovery. Task-count
experiments support a separate limitation: fixed-budget results do not
establish a consistent task-count benefit.

## Questions to prepare for

- **Is Theorem 1 a new stability principle?** It adapts bounded-attention
  reasoning to the implemented mean-pooled encoder and its actual blocks.
- **Is Theorem 2 Transformer-specific?** No. It is a geometric necessary
  condition for any estimator. Its role is to connect response to recovery.
- **Why is the pair not just unidentifiable?** Each endpoint has two distinct
  interior outputs per generator and uniquely determines its cost curve.
- **Does the theorem predict the finite experiment?** The C/m certificate
  is vacuous at tested lengths. Finite behavior is measured separately.
- **What does the measured-response error floor establish?** The observed
  prediction separation requires endpoint error by the triangle inequality.
  It does not explain the training dynamics that caused under-response.
- **Does directional response mean accuracy?** No. It measures the projection
  of prediction change along the true parameter difference. Endpoint errors
  are reported separately.
- **Does leverage establish an attention mechanism?** It establishes
  behavioral response under matched focal edits. The background changes
  and the internal mechanism remains unresolved.
- **Do more tasks improve fixed-compute transfer?** Our controls do not
  establish a consistent benefit.

## Build and edit

From the project root:

~~~powershell
python scripts/build_slide_assets.py
tectonic COURSE_PROJECT_SLIDES_V2.tex
python scripts/package_slides.py
~~~

Alternatively compile with latexmk -pdf COURSE_PROJECT_SLIDES_V2.tex.
The source uses standard Beamer, Latin Modern and TikZ packages.
The standalone ZIP contains the vector figures and the generated
pair_response_check.tex input. Upload the ZIP to Overleaf and select
COURSE_PROJECT_SLIDES_V2.tex as the main file.

The asset script reads saved CSVs, rebuilds four figures and computes the
slide 17 values. It checks the error constraint for each of the three
original checkpoints. It uses NumPy and Matplotlib and does not retrain
models. Figure error bars show SD across three seed means.

Change \setbeameroption{hide notes} to \setbeameroption{show notes} or
\setbeameroption{show only notes} to compile speaker notes.

[Slides](COURSE_PROJECT_SLIDES_V2.pdf) ·
[LaTeX source](COURSE_PROJECT_SLIDES_V2.tex) ·
[Standalone source bundle](CS886_BEAMER_SOURCE.zip) ·
[Final report](COURSE_PROJECT_REPORT.pdf)
