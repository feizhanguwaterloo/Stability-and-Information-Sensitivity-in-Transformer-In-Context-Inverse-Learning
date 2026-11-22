# CS 886 presentation outline

**Stability and Information Sensitivity in Transformer In-Context Inverse Learning**  
**Talk length** · 11 minutes, excluding questions

| Time | Slide | Content |
|---|---|---|
| 0:00–0:55 | 1 · Research question | Ask what happens when one record carries nearly all the information about a task direction. Set up the comparison between small same-task replacement effects and responsiveness to a record that changes the correct answer. |
| 0:55–2:05 | 2 · Two-generator inverse problem | Show the cost \(a_i x_i+b_i x_i^2/2\), then \(\lambda=a_i+b_i x_i\) for interior units. Draw records feeding a frozen encoder, mean pooling and a four-coefficient estimate. Explain that Li et al.'s learner predicts a designated query response, while our model estimates the task parameter vector. |
| 2:05–3:35 | 3 · Replacement stability proof | Display \(\bar e=m^{-1}\sum_i\|h_i-h_i'\|_2\). One changed record makes \(\bar e=O(1/m)\) after embedding. Bounded attention scores make every attention probability \(O(1/m)\). The softmax Jacobian bounds the weight changes, and each fixed layer multiplies \(\bar e\) by a constant. Mean pooling and the readout yield \(C/m\). Mention positive LayerNorm epsilon and the loose constant. |
| 3:35–5:15 | 4 · Recovery lower bound and dispatch example | Put the triangle inequality on screen. If the two errors are at most \(\epsilon\), the response must be at least \(D-2\epsilon\). Show the shared record \((d,x_1,x_2,\lambda)=(1,0.5,0.5,2.1)\), then the distinguishing record at \(d=1.6\) with prices 2.40 and 2.64. Explain why two interior outputs identify each cost curve. |
| 5:15–6:20 | 5 · Architecture comparison (D) | Show the appendix's three-model stability and recovery table. At \(m=64\), Transformer sensitivity is 0.00213 and recovery error is 2.65%. Discuss the scope of the norm-cap comparison and the same-task replacement protocol. |
| 6:20–7:20 | 6 · Transfer controls (E) | Show the equal-pass and equal-step columns at \(m=32\). Equal-pass error improves from 13.48% to 2.72% as the update allocation grows from 56 to 672. The fixed-448-step curve is nonmonotonic. |
| 7:20–8:30 | 7 · Identifiability and conditioning (F) | Highlight the well-conditioned and near-redundant rows. Both have singleton admissible sets. The latter has a mean minimum Gram eigenvalue of \(3.72\times10^{-6}\) and 13.65% model error. Bound-heavy contexts illustrate partial identification. |
| 8:30–10:15 | 8 · Matched leverage and recovery (G) | Use Figure 1 from the report. Start with the right panel and explain the fixed focal measurement edit. Contrast the Transformer's 14.1-fold high-leverage response with its 47.4% directional recovery on the separate clean cross-task pair at \(m=64\). Ridge reaches 99.9% on that pair. |
| 10:15–11:00 | 9 · Conclusion and limits | Return to the research question. The guarantee and the dispatch construction show a worst-case tension. The empirical evidence shows leverage-dependent response under the tested interventions. Mention repeated demands, three seeds, and two generators. |

## Questions to prepare for

- **Why does the theorem not predict the plot?** \(C/m\) is a worst-case upper bound with a loose constant, and D averages valid replacements drawn from the same task.
- **Why can ridge recover the identifying pair?** Each noiseless endpoint has two distinct interior outputs per generator. The normal equations isolate the two cost coefficients, and the tuned ridge penalty is small.
- **Is the lower bound Transformer-specific?** It follows for any estimator. The architecture-specific part is proving that the frozen set encoder has a uniform \(C/m\) upper bound.
- **What does the matched intervention establish?** Different background contexts produce different prediction changes for the same focal edit. The test measures input-output behavior.

[Final report](COURSE_PROJECT_REPORT.pdf) · [Experiment G figure](results/leverage_stability/leverage_stability.png)
