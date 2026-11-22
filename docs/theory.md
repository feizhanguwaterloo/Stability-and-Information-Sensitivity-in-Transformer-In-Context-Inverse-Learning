# Stability, recovery, and identifiability

All norms are Euclidean. Let \(L(\theta;w)=\sum_{i=1}^2\ell_i(\theta_i;w)\)
denote the **whole-record** KKT loss. Each squared hinge is convex and
continuously differentiable, including where its affine argument is zero.

## Stability of the KKT correction

Suppose \(\Theta\) is fixed, nonempty, closed, and convex, \(\gamma\ge0\)
is fixed, and each record loss is convex and differentiable with
\(\|\nabla_\theta L(\theta;w)\|\le G\) on \(\Theta\). For any context
\(S\) of \(m\) records and anchor \(z_S\), define

\[
F_S(\theta)=\tfrac12\|\theta-z_S\|^2+
 \tfrac{\gamma}{m}\sum_{w\in S}L(\theta;w),\qquad
u_S=\arg\min_{\theta\in\Theta}F_S(\theta).
\]

The squared distance gives 1-strong convexity. The KKT losses are
nonnegative, so the objective is coercive. Closedness gives existence
and strong convexity gives uniqueness. More generally, finite convex
differentiable losses defined on the ambient space have affine
minorants, which also imply coercivity here.

Let \(S'\) replace a single \(w\) by \(w'\), and write \(u=u_S\),
\(v=u_{S'}\). The constrained first-order inequalities imply
\(\langle\nabla F_S(u)-\nabla F_{S'}(v),u-v\rangle\le0\).
By strong monotonicity of \(\nabla F_S\),

\[
\begin{aligned}
\|u-v\|^2
&\le \langle\nabla F_S(u)-\nabla F_S(v),u-v\rangle\\
&\le \langle\nabla F_{S'}(v)-\nabla F_S(v),u-v\rangle\\
&\le \bigl(\|z_S-z_{S'}\|+2\gamma G/m\bigr)\|u-v\|.
\end{aligned}
\]

The last line follows by cancellation of the common records and
Cauchy–Schwarz. If \(u=v\) the bound holds directly. Otherwise divide by
\(\|u-v\|\). Under the separate anchor assumption
\(\|z_S-z_{S'}\|\le K_T/m\), the bound is
\((K_T+2\gamma G)/m\).
If the assumed gradient bound \(g\) is per generator, the gradients
occupy disjoint parameter blocks and one can use \(G=\sqrt2\,g\).
In the experiment both modeled marginal costs and all observed prices
are in \([1,3.65]\). Thus \(G=5.30\) is a valid bound since each block is
a residual of magnitude at most 2.65 times \((1,x_i)\), with \(x_i\le1\).
The theorem controls the rate conditional on the anchor assumption
and does not order the empirical changes of the two estimators.

## Theorem A. Replacement stability of the frozen set encoder

For either Transformer in Experiment D, fix all trained parameters
and assume they are finite. For any \(m\ge1\) and any two valid
length-\(m\) in-distribution contexts differing in one record, the
predictions in exact arithmetic differ by at most \(C/m\), where
\(C\) depends on the parameters and input domain but is independent of
\(m\). The conclusion holds with or without the attention caps.

For a fixed finite-depth full-attention encoder on bounded input records,
assume finite weights, no positional encoding or sequence-dependent
mask, rowwise Lipschitz residual blocks, and a Lipschitz head after
mean pooling. At inference there is no dropout. This gives an
architecture-specific \(O(1/m)\) bound in exact arithmetic as follows.

Consider two matrices of row states before one layer, and write
\(e_i=\|h_i-h'_i\|\), \(\bar e=m^{-1}\sum_i e_i\). A layerwise induction
bounds all row norms independently of \(m\). Each attention head forms
a convex combination of bounded values, and the fixed rowwise maps preserve
boundedness. Within one head, row \(i\) has score vectors \(s_i,s_i'\)
with coordinates in \([-B,B]\). Fixed affine query and key maps on
bounded inputs give
\(|s_{ij}-s'_{ij}|\le C_q e_i+C_k e_j\) for constants independent of \(m\).
Along the score segment every softmax probability is at most
\(e^{2B}/m\). The softmax derivative
\(\operatorname{diag}(p)-pp^\top\) has absolute column sums at most
\(2p_j\), so integration along the segment gives

\[
\|p_i-p_i'\|_1 \le 2e^{2B}(C_q e_i+C_k\bar e).
\]

For the value average, changing the weights costs at most a uniform
value bound times \(\|p_i-p_i'\|_1\). Changing the values at fixed weights
costs at most \(e^{2B}\operatorname{Lip}(V)\bar e\). Concatenation and a
fixed output projection preserve a bound of the form
\(A e_i+D\bar e\) for each output row. Residual addition and rowwise
Lipschitz feed-forward and normalization maps preserve this form.
LayerNorm with fixed positive epsilon is Lipschitz because centering
is an orthogonal projection and \(v\mapsto v/\sqrt{\|v\|^2/n+\epsilon}\)
has derivative norm at most \(1/\sqrt\epsilon\). Affine normalization
scales this bound by the largest absolute gain.

Averaging the resulting row differences multiplies \(\bar e\) by a
fixed constant at each layer. One record replacement gives
\(\bar e=O(1/m)\) after the rowwise embedding. The mean and Lipschitz
head give the stated output rate. The pilot encoder has this structure.
This argument supplies an existence bound, with constants that can be
large. It gives no calibrated numerical bound for the trained model
and does not cover arbitrary Transformer architectures. Li et al.'s
Theorem 3.2 bounds query-token prediction for a specific normalized
architecture, using a different readout.
[Li et al., ICML 2023](https://proceedings.mlr.press/v202/li23l/li23l.pdf)

### Audit against the implementation

- The four continuous record features are bounded by one on the pilot
  domain. Exactly two of the six one-hot coordinates equal one, so the
  raw record norm is bounded by \(\sqrt{6}\). An affine embedding has
  output norm at most \(\sqrt6\|W_{\mathrm{emb}}\|+\|b_{\mathrm{emb}}\|\).
- There are two layers and four heads with head dimension \(16\). Each
  score is \(q_i^\top k_j/4\). If the input row norm is bounded by
  \(R\), then \(B=(\|Q\|R+\|b_Q\|)(\|K\|R+\|b_K\|)/4\) is a valid
  score bound per head, taking the corresponding head submatrices.
  Capping the full Q, K, V and output operators also bounds the head
  submatrix norms. The cap alone does not bound biases or other layers.
- The actual layers use post-residual LayerNorm, epsilon \(10^{-5}\),
  and a two-linear-layer ReLU feedforward map. Each normalization
  output has norm at most \(\sqrt{64}\|\gamma_{\rm LN}\|_\infty+
  \|\beta_{\rm LN}\|_2\), uniformly over the incoming row. The
  feedforward Lipschitz constant is bounded by the product of its
  linear operator norms. Residual addition preserves the rowwise
  perturbation estimate.
- The encoder uses unmasked full attention, no padding or positional
  signal, and zero dropout. It averages all final rows, then applies an
  affine map, componentwise sigmoid and affine box scaling. The head
  Lipschitz constant is at most \(\|\mathrm{diag}(\mathrm{span})\|
  \|W_{\rm head}\|/4\).

The same rate holds for DeepSets directly. If \(f\) is the row map
and \(g\) the pooled head, a replacement changes the output by at most
\(\operatorname{Lip}(g)\operatorname{diam}(f(\mathcal W))/m\).
Experiment D therefore compares constants and learning behavior at
finite context lengths. A \(1/m\) rate alone cannot distinguish attention
from averaging. A smaller certified score bound can tighten parts of
the Transformer estimate with the remaining maps held fixed, but two
independently trained checkpoints need not have ordered constants.

## Exact recovery formula for interior observations

For one generator, set \(u_t=(1,x_t)^\top\) and
\(H_m=m^{-1}\sum_tu_tu_t^\top\). In the noiseless interior-only
unconstrained fit, \(\lambda_t=u_t^\top\theta^*\). The first-order
condition is \((I+\gamma H_m)\widehat\theta=z+\gamma H_m\theta^*\).
Since \(H_m\succeq0\), this yields

\[
\widehat\theta-\theta^*=(I+\gamma H_m)^{-1}(z-\theta^*),
\qquad
\|\widehat\theta-\theta^*\|\le
\frac{\|z-\theta^*\|}{1+\gamma\lambda_{\min}(H_m)}.
\]

The smallest eigenvalue governs the worst direction. Two distinct
outputs give rank two, since the rows \((1,x_t)\) are then linearly
independent. Constant outputs leave one direction unchanged. Even with
rank two a finite \(\gamma\) can retain anchor bias. Repeating every
record equally often leaves \(H_m\) and the correction unchanged.
For two units the interior Hessian is block diagonal. The exact
unconstrained formula applies to the boxed estimator when its
unconstrained minimizer lies in the box.

## Mixed observations and identifiability

The noiseless admissible set is a product of per-generator box
intersections with affine equalities and half-planes. A lower record
requires \(a\ge\lambda_t\) since \(x=0\). An upper record requires
\(a+b\bar x\le\lambda_t\). Repeated records at a fixed bound can tighten
its threshold but have the same normal. The singleton condition at the
true \(\theta^*\) is that the interior equalities and inequalities active
there (including box constraints) admit only the zero feasible
direction. If a nonzero direction satisfies the equality normals and
the active half-plane signs, a sufficiently small positive step
maintains every inactive constraint by strict slack. Conversely, the
difference between any second admissible point and \(\theta^*\) satisfies
those directional conditions.

The mixed loss is piecewise quadratic. In each open region its Hessian
includes all interior rows and the currently violated inequality rows.
Satisfied inequalities give zero curvature, so a Gram matrix including
all boundary rows is insufficient for the interior spectral claim.

For a concrete ambiguity take unit capacities, unit 1 cost parameters
\((1,1)\), and unit 2 parameters \((1.50,0.9)\) or \((1.60,1.7)\).
For every \(d\in[0.1,0.4]\), both models yield the unique dispatch
\((d,0)\) and the unique price \(1+d\) fixed by unit 1. The price stays
strictly below unit 2's entry threshold in either model. Arbitrarily many
such records do not distinguish these two costs, even with exact prices.

## Theorem B. Task-level transfer via a covering number

This is a standard uniform convergence argument specialized to the
inverse-dispatch task risk. Fix \(m\) and a hypothesis class \(\mathcal A\)
independent of the source tasks. For a box-valued prediction \(f(S)\)
use the loss \(\ell(f(S),\theta)=\tfrac14\sum_{j=1}^4
((f_j(S)-\theta_j)/\mathrm{span}_j)^2\) in \([0,1]\). Write
\(L_\theta(f)=\mathbb E_{S\sim P_\theta^m}\ell(f(S),\theta)\).
Draw \(\theta_1,\ldots,\theta_T\) independently from one distribution
\(D\) on the box. Define
\(L_{\rm MTL}(f)=T^{-1}\sum_t L_{\theta_t}(f)\) and
\(L_{\rm TFR}(f)=\mathbb E_D L_\theta(f)\).

Let \(N_\epsilon\) be the size of a finite \(\epsilon\)-net of
\(\{\,\theta\mapsto L_\theta(f):f\in\mathcal A\,\}\) in the sup norm
on the task support, with centers in this risk class. For a fixed
\(\epsilon>0\) and \(0<\delta<1\), with probability at least \(1-\delta\)
over the source tasks, every \(f\in\mathcal A\) satisfies

\[
|L_{\rm TFR}(f)-L_{\rm MTL}(f)|
\le 2\epsilon+\sqrt{\frac{\log(2N_\epsilon/\delta)}{2T}}.
\]

For a fixed net center, Hoeffding's inequality bounds the probability
that its absolute population-to-sample deviation exceeds \(u\) by
\(2e^{-2Tu^2}\). A union bound covers the \(N_\epsilon\) centers.
For any risk function, replacing it by a center changes its expected
and empirical task risks by at most \(\epsilon\) each. This proves the
bound simultaneously and permits data-dependent selection within the
fixed class. For a fixed architecture at fixed \(m\) with parameters
restricted to a deterministic compact set, continuity and compactness
give a finite net. In detail, the network is continuous jointly in the
parameters and record matrix. On the compact product of the parameter
set and the bounded record domain's closure, continuity is uniform,
and the bounded squared loss preserves uniform approximation before
integrating over records. An attention-only cap is insufficient to
deduce a small covering number for the other parameters. The study does
not estimate a numerical covering number.

The evaluated risks average two fresh contexts per task. Conditional
on the fitted model and evaluation tasks, these contexts are
independent, so for \(K\) contexts on each of \(T\) tasks their averaged
loss differs from the conditional task average by at most
\(\sqrt{\log(2/\eta)/(2KT)}\) with probability \(1-\eta\).
This describes Monte Carlo observation error. It is separate from
the uncertainty due to sampling tasks. The experiment's held-out
average has both sources of uncertainty.

The task-level result needs no token stability. Theorem A instead
controls fluctuations of a single context loss around \(L_\theta(f)\).
The loss is \(1/\min_j\mathrm{span}_j\)-Lipschitz in the prediction
on the four-dimensional box by its gradient bound. Replacing one
record changes it by at most \(c/m\) with
\(c=C/\min_j\mathrm{span}_j\). Bounded differences for IID records
then gives the two-sided tail \(2\exp(-2mu^2/c^2)\) at deviation \(u\).
This per-context concentration does not replace the task-level
covering argument. Li et al. analyze multitask excess risk with
dependent losses across prompt prefixes and discuss transfer through
task diversity in Section 4. Our bound compares absolute task risks
for one pooled estimator under IID tasks. The Beta density shift
violates the same-distribution premise. [Li et al., ICML 2023](https://proceedings.mlr.press/v202/li23l.html)

## Proposition C. Learning error and partial identification

Let \(P(S)\) be the compact convex exact-KKT parameter set for a
noiseless context, with true \(\theta^*\in P(S)\). For a learned anchor
\(z=f(S)\), let \(p\) be its Euclidean projection onto \(P(S)\).
The projection variational inequality gives
\(\langle z-p,\theta^*-p\rangle\le0\), hence

\[
\|z-\theta^*\|^2\ge
\mathrm{dist}(z,P(S))^2+\|p-\theta^*\|^2,\qquad
\|p-\theta^*\|\le\mathrm{diam}(P(S)).
\]

Thus the anchor's distance from the set is a detectable inconsistency,
while two exact-KKT anchors can differ within the set even at zero
residual. In a singleton set the choice of anchor is immaterial for
exact projection. For a partial set, learned cross-task information
can affect where the projection lands. The diameter is computed from
the two per-generator polygons by taking the square root of the sum
of their squared diameters, using the Cartesian product structure.

For any two compatible tasks \(\theta_1,\theta_2\) with the same
observation law, the triangle inequality implies that the larger
expected Euclidean error of any estimator is at least
\(\|\theta_1-\theta_2\|/2\). Indeed the same law lets both expectations
be taken over the same observations and any shared model randomness.
The bound-only example above gives two such tasks if demand has the
same distribution supported on \([0.1,0.4]\). For a fixed realized
context, a pair realizing the admissible diameter similarly gives a
pointwise worst-case lower bound. This pointwise statement does not
assume that sampling a context conditional on task parameters
produces the same distribution across compatible tasks. The analysis
uses classical concentration, projection and identifiability tools.
Sharper constants and a link to the learned attention policy remain
research questions.
