# Project to-do list

- [x] Finalize the CS 886 course project using the brief below.

---

# Finalize CS 886 Course Project — Transformer Learning Theory

We have completed multiple rounds of theoretical analysis and experiments for the project on Transformer in-context inverse optimization.

**Our priority now is to finalize a strong CS 886 course project, NOT to pursue a publishable research paper.**

Do not expand the research scope unnecessarily. Focus on consolidating existing results, checking mathematical correctness, selecting the most important experiments, and producing a coherent research-style final report.

## 1. Project Title

**Stability and Information Sensitivity in Transformer In-Context Inverse Learning**

The project is motivated by:

*Transformers as Algorithms: Generalization and Stability in In-Context Learning* (Li et al., ICML 2023).

### Central Research Question

**How do Transformers balance algorithmic stability and the ability to learn from highly informative observations during in-context learning?**

We use inverse economic dispatch as a structured testbed because the information provided by optimal decisions and market prices can be mathematically characterized using KKT conditions.

The Transformer is the main research subject. Economic dispatch and KKT conditions are supporting tools.

## 2. Review Existing Work

First inspect the existing repository, including:

- `problem_formulation.md`
- `theory.md`
- `literature.md`
- `THEORY_RESEARCH_DIRECTION.md`
- Previous experiment reports
- Experiments A–G
- Existing plots, CSV files, and configurations

Do not repeat experiments unless necessary to correct an error or fill a critical gap.

Preserve all previous code and experiment results.

## 3. Main Theoretical Component

Focus on two theoretical results.

### Theorem 1: Transformer Replacement Stability

Present and rigorously verify the existing result:

For the fixed-depth, bounded-input, full-attention, mean-pooled Transformer architecture, replacing one context observation yields

\[
\|TF(S_m)-TF(S'_m)\|_2\leq \frac{C}{m},
\]

under the stated architectural assumptions.

The proof should cover the actual architecture, including attention, residual connections, LayerNorm, feedforward layers, and mean pooling.

Explicitly distinguish this theorem from the query-token stability theorem in Li et al. (2023).

State that the constant may be large and that this bound does not automatically imply small empirical sensitivity.

### Theorem 2: Stability–Accuracy Tradeoff

Consider two valid contexts differing in one observation, with true parameter vectors separated by distance \(D\).

If an estimator has recovery error at most \(\epsilon\) on both contexts, prove that

\[
\|TF(S)-TF(S')\|_2\geq D-2\epsilon.
\]

Combining with an assumed stability upper bound gives

\[
\epsilon\geq
\frac12\left(D-\frac{C}{m}\right).
\]

Use the existing two-task economic-dispatch construction to demonstrate the idea.

Verify that both contexts are valid forward-optimization outcomes, and distinguish cross-task context comparisons from the same-task replacements used in the empirical stability experiment.

Explain clearly that the lower bound follows from a general triangle-inequality argument and is not unique to Transformers.

The key interpretation is that accurate recovery of parameters distinguished by a single observation requires sufficient prediction sensitivity to that observation.

Retain the existing task-level transfer bound as a supporting theoretical result, not the main theorem.

Do not attempt to invent additional major theorems.

## 4. Select and Consolidate Experimental Results

Use the existing results to tell one coherent story.

### Experiment 1: Transformer Architecture and Stability

Use Experiment D.

Compare:

- Original Transformer
- Norm-controlled Transformer
- DeepSets

Present recovery error and single-record replacement sensitivity as context length increases.

Discuss why the original Transformer showed lower empirical sensitivity and better accuracy under the tested training protocol.

Do not claim that norm constraints necessarily worsen theoretical stability or that attention is universally superior.

### Experiment 2: Transfer Across Tasks

Use Experiment E.

Study the effect of pretraining task count on transfer to unseen tasks.

Present both:

- Equal-task-pass results
- Equal-optimizer-step results

Clearly explain the training-budget confounding.

The defensible conclusion is that performance improved with more tasks under one training protocol, but the benefit of task count alone was not established under fixed compute.

Do not claim a theoretical \(1/\sqrt T\) empirical rate without supporting evidence.

### Experiment 3: Context Informativeness

Use Experiment F.

Compare:

- Well-conditioned contexts
- Nearly redundant contexts
- Mixed contexts
- Boundary-heavy contexts

Explain why full-rank identifiability is not sufficient for good finite-sample Transformer recovery when the context is poorly conditioned.

Use the Gram eigenvalue and recovery-error results.

Focus on the Transformer's behavior rather than on KKT correction performance.

### Experiment 4: Leverage Sensitivity and Tradeoff

Use Experiment G and its associated theory.

Focus on:

- Highly informative observations
- Redundant observations
- Matched replacement interventions
- Transformer versus ridge or sufficient-statistics baselines
- Parameter-difference recovery ratio

Verify the reported numerical findings against the saved experiment results.

Discuss whether the Transformer responds more strongly to informative observations, and whether the magnitude of its response is sufficient for accurate recovery.

Do not claim that empirical sensitivity proves an internal attention mechanism.

If a targeted training experiment has already been completed, incorporate it. Otherwise, do not launch a major new training study unless it is essential.

## 5. Important Conceptual Clarification

The existing Transformer receives a set of observations and predicts latent cost parameters.

This is best described as **context-conditioned, supervised amortized inverse estimation**, which is related to meta-learning and in-context learning.

It is not identical to the standard query-conditioned ICL formulation in Li et al. (2023).

Explain this distinction explicitly.

Do not misrepresent the existing model as an autoregressive Transformer performing standard next-token prediction.

A query-conditioned ICL formulation may be discussed as future work, but is not required for completing the course project.

## 6. Final Research Narrative

Organize the report around the following progression:

**Motivation:** Transformers learn from context, but different context observations carry different amounts of task information.

**Theory:** Under suitable assumptions, the Transformer has \(O(1/m)\) replacement stability.

**Tension:** When only a small number of observations distinguish possible tasks, accurate recovery requires sensitivity to those observations.

**Experiments:** The Transformer exhibits different responses to informative and redundant observations. Its recovery performance also depends on conditioning, architecture, and source-task training.

**Conclusion:** Stability alone does not guarantee accurate in-context learning. The distribution and informativeness of context observations matter.

Keep the project centered on this narrative.

## 7. Produce the Final Course Project Report

Create a polished, research-style report of approximately 6–8 pages of main content, excluding references and optional appendices.

Use LaTeX.

Suggested structure:

1. Abstract
2. Introduction
3. Problem Setup and Transformer Architecture
4. Theoretical Analysis
5. Experimental Setup
6. Experimental Results
7. Discussion and Limitations
8. Conclusion
9. References

Put lengthy mathematical proofs, additional experimental tables, and implementation details in an appendix.

Use properly labeled mathematical notation, equations, figures, and references.

Do not overload the main report with all previous experiments. Select the figures that best support the central research question.

Include a concise comparison with Li et al. (2023) and relevant inverse optimization literature.

Do not make unsupported claims of novelty.

## 8. Quality Control

Before finalizing:

- Check every theorem and proof.
- Verify the economic-dispatch construction.
- Verify all numerical claims against existing CSV files.
- Ensure figures match the text.
- Ensure source-task and target-task evaluations are distinguished correctly.
- Check the distinction between worst-case stability and sampled average sensitivity.
- Clarify training-budget confounding.
- Explain the limitations of noiseless two-generator tasks.
- Verify all citations.
- Run the existing unit tests.

If a mathematical claim is unsupported, correct or weaken it instead of inventing a proof.

## 9. Deliverables

Produce:

- `COURSE_PROJECT_REPORT.tex` — complete LaTeX source
- `COURSE_PROJECT_REPORT.pdf` — compiled report, if a LaTeX compiler is available
- `COURSE_PROJECT_SUMMARY.md` — one-page summary of motivation, methods, main findings, and contributions
- `COURSE_PROJECT_PRESENTATION_OUTLINE.md` — a 10–12-minute presentation outline
- A short record of verification results and any unresolved issues

Keep existing files and results intact.

## 10. Scope Restriction

This is a course project.

Do NOT:

- Add network-constrained OPF
- Introduce substantially larger Transformers
- Develop additional KKT correction methods
- Start a new literature-review project
- Pursue publication-level novelty as a requirement
- Add many more experiments without a clear need

Focus on correctness, clarity, theoretical understanding, and a defensible conclusion.

**The final goal is a complete course project that demonstrates an understanding of Transformer stability, generalization, and information-dependent in-context learning.**

Start by reviewing the current repository, then consolidate the work into the final report and presentation outline. Actually compile and verify the report where the environment supports it.
