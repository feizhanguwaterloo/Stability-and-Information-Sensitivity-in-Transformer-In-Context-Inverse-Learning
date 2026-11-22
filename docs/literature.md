# Initial literature notes

- **Li, Ildiz, Papailiopoulos, and Oymak (2023).** The paper frames
  in-context learning as learning an algorithm across tasks.
  Theorem 3.2 bounds query-token prediction sensitivity to replacement
  of a prior example under specific normalization and architectural
  assumptions. Here a context is an unordered collection of complete
  market records and the readout predicts the task parameters directly.
  The proof for the frozen set encoder therefore works with mean-pooled
  row perturbations.
  [Transformers as Algorithms, ICML](https://proceedings.mlr.press/v202/li23l.html)

- **Keshavarz, Wang, and Boyd (2011).** A classical convex inverse
  optimization approach fits objective parameters by penalizing
  violations of KKT conditions on observed choices. In this project
  the balance multiplier is observed and each active bound is known.
  Eliminating nonnegative bound multipliers yields a squared hinge on
  the corresponding signed stationarity gap. This motivates the
  fixed-prior baseline as well as the learned-anchor version.
  [Imputing a Convex Objective Function, IEEE MSC](https://web.stanford.edu/~boyd/papers/imputed_objective.html)

- **Liang and Dvorkin (2023).** Their electricity-market inverse model
  uses clearing results to infer offer parameters through a DCOPF
  formulation. Their analysis covers optimization of the inverse model
  and robustness under price error and model mismatch. The present
  pilot studies which parameters are compatible with exact prices and
  dispatch in a two-unit quadratic model. A unique optimum of its
  anchored correction follows from regularization, while identification
  concerns uniqueness of the parameter vector compatible with the data.
  [Data-Driven Inverse Optimization for Marginal Offer Price Recovery, ACM e-Energy](https://arxiv.org/abs/2302.05498)

- **Ruiz, Conejo, and Bertsimas (2013).** Their network-constrained
  inverse model uses cleared quantities and locational marginal prices
  to recover offers that were marginal in at least one observed period.
  In the two-generator pilot, interior dispatch gives an equality,
  while a unit held at a bound only contributes one-sided restrictions.
  [Revealing Rival Marginal Offer Prices via Inverse Optimization, IEEE TPS](https://researchportal.uc3m.es/display/act414508)

The initial positioning is a comparison of learned and fixed anchors
under the same KKT penalty, together with a replacement-sensitivity
analysis and an explicit test of information loss under censoring by
unit bounds. These papers establish precedents for the two ingredients.
This initial reading does not establish novelty for their combination.
