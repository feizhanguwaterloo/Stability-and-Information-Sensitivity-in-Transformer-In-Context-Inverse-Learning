# Project Update: Transformer-Centered In-Context Inverse Optimization

## 1. Research Direction

Update the existing project **KKT-Aware In-Context Inverse Optimization: Stability and Identifiability** to make it more focused on **Transformer Learning Theory**, while preserving the connection to inverse optimization.

The previous pilot successfully implemented the two-generator economic dispatch model, synthetic task generation, a Transformer estimator, KKT correction, and stability/recovery experiments.

However, the current project focuses too heavily on KKT correction and classical inverse optimization.

The new objective is to study the **learning behavior, stability, and generalization of Transformers under optimization-induced observations**.

Proposed title:

**Stability and Transfer Generalization of Transformer In-Context Learning under Optimization-Induced Observations**

The project is motivated by *Transformers as Algorithms: Generalization and Stability in In-Context Learning* (ICML 2023).

### Main Research Questions

1. How does context length affect the stability and recovery performance of a Transformer in-context learner?
2. How does the number and diversity of pretraining tasks affect transfer performance on unseen optimization tasks?
3. Can Transformers distinguish informative from uninformative observations in context?
4. Can we establish theoretical guarantees for the Transformer itself, rather than only for a subsequent KKT correction layer?

**The Transformer must be the primary object of investigation. KKT conditions should primarily serve as a mathematical tool for characterizing observation informativeness and evaluating recovery consistency.**

---

## 2. Preserve Existing Implementation

Do not restart the project.

Retain the existing:

- Two-generator convex economic dispatch solver
- Synthetic inverse optimization task generator
- Transformer encoder and training pipeline
- KKT loss and correction layer
- Classical inverse optimization baselines
- Experiments A–C
- Theory documentation and unit tests

Keep the original pilot results unchanged as a baseline.

Create a new experiment configuration and output directory for the Transformer-centered experiments.

Inspect the existing repository before modifying files.

---

## 3. Experiment D: Transformer Stability and Architecture

### Objective

Investigate whether the stability properties discussed in *Transformers as Algorithms* can be observed and theoretically justified for our implemented Transformer architecture.

For two contexts differing by one observation, measure:

\[
\Delta_m^{TF}
=
\mathbb E\left[
\|TF_\phi(S_m)-TF_\phi(S'_m)\|_2
\right].
\]

Use context lengths:

\[
m\in\{4,8,16,32,64\}.
\]

### Models to Compare

1. Original Transformer encoder
2. Transformer with controlled attention weight/operator norms
3. DeepSets or mean-pooled MLP baseline
4. Transformer + KKT correction (secondary comparison)

Keep the training tasks and evaluation protocol consistent.

The purpose is to investigate whether attention architecture, rather than only averaging observations, contributes to stability and learning performance.

### Required Analysis

- Plot stability against context length.
- Compare with a reference \(1/m\) curve.
- Report parameter recovery error alongside stability.
- Examine whether greater stability corresponds to better recovery accuracy.
- Report the effect of architectural constraints on prediction quality.

Distinguish empirical average replacement sensitivity from theoretical worst-case stability.

Do not assume an arbitrary trained Transformer satisfies the stability theorem.

---

## 4. Experiment E: Multitask Learning and Transfer Generalization

### Objective

Study how the number of pretraining tasks affects the Transformer's ability to perform in-context inverse optimization on unseen tasks.

This is directly motivated by Section 4 of *Transformers as Algorithms*.

Vary the number of source tasks:

\[
T\in\{128,256,512,1024,1536\}.
\]

Use the existing task distribution initially.

For each \(T\), train a Transformer and evaluate it on a fixed set of unseen tasks.

### Metrics

Report:

- Source-task population-risk estimate using fresh observations
- Unseen-task transfer risk
- Transfer gap
- Parameter recovery error
- Context-length-dependent prediction performance

Define:

\[
L_{\mathrm{MTL}}(TF)
=
\frac1T\sum_{t=1}^{T}L_t(TF)
\]

and

\[
L_{\mathrm{TFR}}(TF)
=
\mathbb E_{\mathcal T\sim D_{\mathrm{task}}}
[L_{\mathcal T}(TF)].
\]

Estimate these quantities using independently generated evaluation observations, not training losses.

Compare:

\[
L_{\mathrm{TFR}}(TF)-L_{\mathrm{MTL}}(TF).
\]

### Additional Controlled Comparison

Compare:

- More tasks with shorter context sequences
- Fewer tasks with longer context sequences

Keep the total number of training observations approximately constant.

Also run a separate comparison with similar optimization budgets to distinguish statistical task-count effects from additional training compute.

### Distribution Shift

After the in-distribution experiment, test different ranges or distributions of cost parameters.

Report in-distribution transfer separately from out-of-distribution transfer.

The generalization guarantees from the reference paper should not be claimed to hold automatically under distribution shift.

### Key Question

Does Transformer transfer improve primarily with the number of distinct optimization tasks, or with the total number of observations?

---

## 5. Experiment F: Information-Dependent In-Context Learning

### Objective

Investigate whether Transformers learn to exploit informative observations rather than treating all context records equally.

Use KKT theory to characterize the information contained in each market observation.

Construct contexts with different information structures:

1. Well-conditioned interior observations
2. Nearly redundant interior observations
3. Mixed equality/inequality observations
4. Boundary-heavy or partially identifiable observations

Use actual feasible economic dispatch outcomes.

### Important Control

The previous experiment changed both operating-status patterns and demand distributions.

This creates a confounding effect.

The revised experiment should control task parameters and demand distributions wherever possible.

Ideally, generate a large observation pool for each fixed task and select context subsets with different information structures.

When matching is not possible, explicitly report the remaining distribution differences.

### Core Experiments

**A. Informative observation replacement**

Replace a highly informative observation with another valid observation and measure prediction change.

**B. Redundant observation replacement**

Replace an observation that contributes little new information.

**C. Information ablation**

Remove or mask certain observation types and evaluate the change in Transformer predictions.

**D. Context order**

Shuffle observations to verify the expected permutation invariance of the implemented model.

### Metrics

Measure:

- Transformer-only recovery error
- Prediction sensitivity
- Marginal cost prediction error
- Information measures such as Gram eigenvalues
- Geometry or size of the KKT-admissible parameter set when computable

Attention weights may be inspected as supplementary diagnostics, but should not be treated as sufficient evidence that the model uses a particular observation.

The primary evidence should come from controlled intervention and ablation experiments.

### Key Question

**Does the Transformer learn to use the information structure of optimality-induced observations?**

---

## 6. Theoretical Analysis: Transformer-Centered Guarantees

The theoretical contribution should focus primarily on the Transformer.

### Theorem A: Transformer Stability

Review the existing structural \(O(1/m)\) stability proof for the full-attention, bounded-input, mean-pooled Transformer.

Verify that the theorem statement matches the actual implementation.

Check assumptions involving:

- Bounded token representations
- Attention score bounds
- Operator norms of attention matrices
- Feedforward layers
- Residual connections and normalization
- Pooling mechanism
- Fixed model depth
- Context-dependent masking or positional encodings

If some assumptions are not satisfied, either modify the theoretical model or explicitly state that the guarantee applies only to a simplified architecture.

Produce a mathematically rigorous proof for the clearly specified architecture.

Do not claim the trained model satisfies a bound without verifying the necessary assumptions.

### Theorem B: Transfer Generalization

Explore whether the stability and covering-number framework from *Transformers as Algorithms* can be adapted to the optimization-induced task distribution.

Investigate a bound of the conceptual form:

\[
L_{\mathrm{TFR}}(TF)
\lesssim
L_{\mathrm{MTL}}(TF)
+
O\left(
\sqrt{
\frac{\operatorname{Complexity}(\mathcal A)}
{T}
}
\right).
\]

Do not present this schematic expression as a new theorem without explicit assumptions, constants, and proof.

Clearly identify which parts follow directly from existing learning theory and which require new arguments.

### Theorem C: Information-Dependent Recovery

Investigate whether prediction/recovery error can be related to:

1. Transformer approximation or learning error
2. Statistical information contained in the context
3. Partial identifiability of the underlying optimization problem

A possible direction is to study the diameter or geometry of the set of cost parameters consistent with the observed KKT conditions.

Construct simple counterexamples showing that even a stable Transformer cannot uniquely recover parameters when observations are insufficient to identify them.

The goal is to connect Transformer learning behavior with information constraints, not simply to restate classical KKT identifiability.

### Proof Verification

AI-assisted proof development is encouraged.

Lean 4 may be used to formally verify small mathematical lemmas where practical.

Formal verification is optional. Do not sacrifice the main research experiments to formalize standard results.

Distinguish new theorems from known results, conditional guarantees, and conjectures.

---

## 7. Improve Experimental Rigor

The original pilot used one random seed and a noiseless setting.

For the updated project:

- Run at least 3 independent seeds for the main experiments, preferably 5 if compute permits.
- Report means and variability across seeds.
- Preserve task-level train/validation/test separation.
- Include stronger classical baselines.
- Add an exact-KKT-consistent estimator where feasible.
- Report full-rank and partially identifiable cases separately.
- Avoid making conclusions about Transformer superiority from noiseless full-rank cases where classical methods already recover the parameters exactly.

Include price-noise experiments if time permits.

Distinguish improvements due to learned cross-task priors from improvements attributable solely to KKT correction.

---

## 8. Updated Deliverables

Create the following additional materials while preserving previous outputs.

### Code

- Transformer architecture/stability comparisons
- Task-count scaling experiment
- Controlled observation-informativeness experiment
- Additional evaluation metrics and plots

### Theory

Update `docs/theory.md` with:

- Precise Transformer stability assumptions
- Theorem statements and proofs
- Discussion of transfer generalization
- Identifiability limitations
- Explicit distinction between established theory and new contributions

### Results

Create:

- `results/transformer_stability/`
- `results/task_scaling/`
- `results/context_informativeness/`

Each directory should include configurations, numerical results, and publication-quality figures.

### Research Report

Create `TRANSFORMER_PROJECT_UPDATE.md` containing:

1. Updated research motivation
2. Transformer-centered theoretical questions
3. Experimental methodology
4. Main empirical findings
5. Mathematical guarantees and their assumptions
6. Relation to *Transformers as Algorithms*
7. Limitations and open questions
8. Potential directions for a future publication

---

## 9. Execution Priority

Work in this order:

**Priority 1 — Review Existing Theory**

Audit the existing Transformer stability proof against the implemented architecture.

Correct unsupported assumptions and document any gaps.

**Priority 2 — Transformer Stability Experiment**

Run the architecture comparison and context-length stability tests.

**Priority 3 — Task Scaling**

Run the \(T\)-scaling experiment with a controlled evaluation protocol.

**Priority 4 — Context Informativeness**

Run matched observation-informativeness and ablation experiments.

**Priority 5 — Additional Theory**

Attempt transfer and information-dependent guarantees based on the empirical findings.

If training cost is high, prioritize a smaller reliable experiment over an incomplete large experiment.

Run all available tests and experiments, not just code generation.

Do not invent results, claim unverified proofs, or hide failed experiments.

---

## 10. Final Research Positioning

**Transformer is the research subject.**

**Inverse Optimization is the structured learning environment.**

**KKT conditions provide the mathematical description of observation informativeness.**

The project should investigate what Transformer ICL can learn from optimality-induced observations, how stable that learning is, and how it transfers across tasks.

The long-term objective is a Transformer learning theory contribution with inverse optimization applications, rather than simply a new numerical cost recovery method.
