# Research Project: KKT-Aware In-Context Inverse Optimization

## 1. Research Goal

Develop a small, reproducible research prototype for **KKT-Aware In-Context Inverse Optimization: Stability and Identifiability**.

The project is inspired by *Transformers as Algorithms: Generalization and Stability in In-Context Learning* (ICML 2023).

We want to investigate:

1. Can a pretrained Transformer recover unknown cost parameters from optimal decisions and prices on unseen optimization tasks, without updating its weights at test time?
2. Can a KKT-aware correction layer preserve algorithmic stability?
3. How does the informativeness of observations affect parameter recovery?

This is a theory-oriented research project, not simply a neural network benchmark.

Start with a simple convex economic dispatch problem. Do not implement a full network-constrained OPF yet.

## 2. Forward Optimization Problem

Consider two generators with quadratic costs:

\[
C_i(x_i)=a_ix_i+\frac12b_ix_i^2,\quad b_i>0.
\]

The forward economic dispatch problem is:

\[
\begin{aligned}
\min_{x_1,x_2}\quad&
C_1(x_1)+C_2(x_2)\\
\text{s.t.}\quad&
x_1+x_2=d,\\
&0\leq x_i\leq \bar x_i.
\end{aligned}
\]

Unknown task parameters:

\[
\theta=(a_1,b_1,a_2,b_2).
\]

Generate different tasks by randomly sampling cost parameters.

For each task, generate different demands, solve the forward problem, and obtain:

- Demand \(d_t\)
- Optimal dispatch \(x_{1t},x_{2t}\)
- Market-clearing price \(\lambda_t\)
- Generator operating status: lower, interior, or upper bound

Each task has a fixed unknown parameter vector but multiple market observations.

Use reproducible parameter ranges that produce both interior and boundary observations.

An analytical dispatch solver based on marginal-cost equalization and bisection is acceptable. Verify its results against a numerical convex optimizer.

## 3. KKT Conditions

Marginal cost:

\[
m_i(x_i)=a_i+b_ix_i.
\]

For this simplified economic dispatch:

\[
m_i(x_{it})
\begin{cases}
=\lambda_t & \text{interior},\\
\geq\lambda_t & \text{lower bound},\\
\leq\lambda_t & \text{upper bound}.
\end{cases}
\]

Define KKT loss:

\[
\ell_{it}^{KKT}(\theta_i)=
\begin{cases}
\frac12(m_i(x_{it})-\lambda_t)^2,
&\text{interior},\\
\frac12[\lambda_t-m_i(x_{it})]_+^2,
&\text{lower},\\
\frac12[m_i(x_{it})-\lambda_t]_+^2,
&\text{upper}.
\end{cases}
\]

Verify the signs carefully.

## 4. Transformer In-Context Learner

Implement a small Transformer in PyTorch.

Suggested initial configuration:

- 2 Transformer encoder layers
- 4 attention heads
- Hidden dimension 64
- Variable context length
- Fixed weights during testing

For each task, the input context contains historical market observations:

\[
S_m=\{(d_t,x_{1t},x_{2t},\lambda_t,s_{1t},s_{2t})\}_{t=1}^m.
\]

The Transformer predicts:

\[
z_\phi(S_m)=\widetilde\theta.
\]

Pretrain across many synthetic tasks with different unknown cost parameters.

At test time, provide observations from previously unseen tasks, with no gradient updates.

The implementation must not expose true test-task cost parameters as input features.

Since these observations are unordered market records, consider an encoder without positional embeddings and use permutation-invariant pooling.

Clearly distinguish this context-conditioned parameter estimator from the exact autoregressive prediction setup of the reference paper.

## 5. KKT-Aware Correction

Given the Transformer estimate \(z_\phi(S_m)\), solve:

\[
\widehat\theta(S_m)
=
\arg\min_{\theta\in\Theta}
\left[
\frac12\|\theta-z_\phi(S_m)\|_2^2
+
\frac{\gamma}{m}
\sum_{t=1}^m\sum_{i=1}^2
\ell_{it}^{KKT}(\theta_i)
\right].
\]

Use a fixed convex parameter set \(\Theta\) enforcing valid cost coefficients and bounds.

First implement correction as a postprocessing convex optimization problem. End-to-end differentiation through the optimization layer is optional and not required for the initial project.

Check numerical convergence, KKT loss, feasibility, and solution uniqueness.

## 6. Theory: Stability Guarantee

Investigate the following theorem candidate.

Assume the Transformer estimator satisfies:

\[
\|z_\phi(S_m)-z_\phi(S'_m)\|
\leq\frac{K_T}{m}
\]

for two contexts differing by one observation.

Assume:

- \(\Theta\) is fixed, nonempty, closed, and convex.
- Each KKT loss is convex and differentiable.
- Its gradient norm is bounded by \(G\) on \(\Theta\).
- The regularization coefficient \(\gamma\) is fixed.

Prove:

\[
\boxed{
\|\widehat\theta(S_m)-\widehat\theta(S'_m)\|
\leq
\frac{K_T+2\gamma G}{m}.
}
\]

Suggested tools:

- Strong convexity
- Monotonicity of the optimization optimality conditions
- Sensitivity analysis

Write a rigorous mathematical proof in `docs/theory.md`.

Do not assume an arbitrary trained Transformer satisfies the initial stability bound. Treat it as a conditional assumption and investigate it separately.

This is a basic stability-preservation result, not automatically a novel research theorem.

## 7. Theory: Recovery and Identifiability

For interior observations, consider:

\[
\lambda_t=a+bx_t.
\]

Let:

\[
u_t=(1,x_t)^\top
\]

and

\[
H_m=\frac1m\sum_{t=1}^m u_tu_t^\top.
\]

In the noiseless interior-only unconstrained case, investigate:

\[
\boxed{
\|\widehat\theta-\theta^*\|
\leq
\frac{\|z_\phi(S_m)-\theta^*\|}
{1+\gamma\lambda_{\min}(H_m)}.
}
\]

Derive the exact estimator and prove the bound.

Then construct a non-identifiability counterexample where two different cost functions produce the same optimal dispatch and prices because a generator always operates at a bound.

Discuss why increasing context length alone cannot always guarantee accurate cost recovery.

For mixed equality/inequality observations, investigate the geometry of the KKT-admissible parameter set. Do not claim the interior-only spectral result automatically applies.

## 8. Experiments

Implement the following three experiments.

### Experiment A: Context Length

Evaluate:

\[
m\in\{4,8,16,32,64\}.
\]

Measure:

- Normalized parameter recovery error
- Marginal-cost prediction error
- KKT residual
- Out-of-sample dispatch reconstruction error

### Experiment B: Observation Informativeness

Compare contexts that are:

- Interior-rich
- Mixed interior/boundary
- Boundary-heavy

Keep context length fixed while varying observation informativeness.

Measure how recovery accuracy changes.

Generate these contexts using actual forward optimization outcomes rather than arbitrarily assigning operating statuses.

### Experiment C: Stability

Replace one historical observation with a different valid observation from the same task.

Measure:

\[
\Delta_m=
\mathbb E
\left[
\|\widehat\theta(S_m)-\widehat\theta(S'_m)\|
\right].
\]

Compare Transformer-only and Transformer + KKT correction.

Plot \(\Delta_m\) against context length and compare with a reference \(1/m\) curve.

Do not claim empirical observations prove a theoretical \(O(1/m)\) guarantee.

Optionally test corrupted observations as a separate robustness experiment.

## 9. Baselines

Implement at least:

1. KKT-based classical inverse optimization with a fixed prior or regularizer
2. Transformer-only parameter recovery
3. Transformer + KKT correction
4. Prior-based regularized parameter estimator

Use the same synthetic tasks for fair comparisons.

Include an oracle reference using the true cost parameters for downstream evaluation.

All train/validation/test splits must be by task, not by individual market observation.

## 10. Engineering Requirements

Use Python, PyTorch, NumPy, SciPy, and Matplotlib.

Inspect any existing repository before creating files. Avoid overwriting unrelated code.

Create a modular implementation with separate components for:

- Data generation
- Economic dispatch solver
- KKT conditions and loss
- Transformer model
- KKT correction
- Training
- Evaluation
- Theorem documentation
- Plot generation

Requirements:

- Fixed random seeds
- Reproducible experiments
- CPU-compatible small configuration
- GPU support if available
- Unit tests for forward optimization and KKT conditions
- Unit tests for the correction optimizer
- Saved experiment configurations
- Clearly labeled plots and CSV result tables

## 11. Expected Deliverables

Create:

- `README.md`: setup and execution instructions
- `docs/problem_formulation.md`: mathematical model
- `docs/theory.md`: theorem statements, proofs, assumptions, counterexamples
- `src/`: implementation
- `tests/`: correctness tests
- `configs/`: experiment configurations
- `results/`: numerical results and plots
- `PROJECT_REPORT.md`: concise summary of completed experiments and findings

In the report, clearly separate:

- Mathematically proven results
- Numerical observations
- Unverified hypotheses
- Known limitations

Do not invent experimental numbers or theoretical results.

## 12. Execution Priorities

**Phase 1 — Must complete first**

Implement data generation, forward ED, KKT verification, and a classical inverse optimization baseline.

Run unit tests and confirm the generated data satisfy the expected KKT conditions.

**Phase 2 — Main prototype**

Implement the small Transformer, cross-task training, evaluation on unseen tasks, and KKT correction.

**Phase 3 — Theory and experiments**

Write rigorous proofs or explain any gaps. Run the three main experiments and generate figures.

**Phase 4 — Optional**

Explore generalization bounds for mixed KKT observations, noisy measurements, or harder economic dispatch tasks.

Do not start Phase 4 before the core experiments work.

## 13. Final Output

Actually run the available tests and experiments rather than only generating source code.

At the end, provide:

1. Which phases were completed
2. Exact commands used
3. Test results
4. Main numerical findings
5. Paths to generated plots and reports
6. Remaining mathematical or implementation issues
7. Recommended next research steps

The objective is a working proof-of-concept suitable for a theoretical machine learning course project, with potential extension into a standalone inverse optimization research paper.
