# Two-generator inverse economic dispatch

We observe demand \(d\), the unique optimal dispatch \(x\), a clearing price
\(\lambda\), and each unit's bound status. The known capacity vector is
\((1,1)\). A task has fixed \(\theta=(a_1,b_1,a_2,b_2)\) and minimizes
\(\sum_i(a_ix_i+b_ix_i^2/2)\) subject to \(x_1+x_2=d\) and
\(0\le x_i\le\bar x_i\). The experiment samples each \(a_i\) uniformly in
\([1,1.65]\) and each \(b_i\) uniformly in \([0.8,2]\). Their product box is
\(\Theta\). In this range the two units' open marginal-cost intervals overlap.

## Forward solver and price

Substitute \(x_2=d-x_1\) and clip the unconstrained scalar minimizer
\((a_2-a_1+b_2d)/(b_1+b_2)\) to
\([\max(0,d-\bar x_2),\min(\bar x_1,d)]\).
Use an interior unit's marginal cost as the price. With all units at
bounds, the price interval has lower endpoint equal to the largest upper
unit's marginal cost and upper endpoint equal to the smallest lower
unit's marginal cost, with missing endpoints treated as infinities.
The solver selects the midpoint when both endpoints are finite and the
finite endpoint otherwise. This selection is a convention for generating
prices when the multiplier is ambiguous.

Use the Lagrangian \(C(x)+\lambda(d-\sum_i x_i)-\sum_i\mu_i^-x_i+
\sum_i\mu_i^+(x_i-\bar x_i)\). Stationarity gives
\(a_i+b_ix_i-\lambda=\mu_i^- -\mu_i^+\). By complementary slackness the
gap \(g_i=a_i+b_ix_i-\lambda\) is zero for an interior unit, nonnegative
at a lower bound, and nonpositive at an upper bound. The signed residual
is \(g_i\), \(\min(g_i,0)\), or \(\max(g_i,0)\) for interior, lower, or upper
respectively. The record loss is half the sum of squared residuals.

## Inverse estimators and data

The Transformer embeds each record using four scaled continuous features
and two status one-hot vectors. A full-attention encoder with no
positional embeddings and mean pooling outputs one parameter vector via
a sigmoid followed by an affine map into \(\Theta\). The training label
is the synthetic task's true parameter vector. A frozen model reads
records from unseen tasks. This is supervised amortized inverse
estimation over sets of records, whereas Li et al. study query prediction
from an example sequence.

The corrected estimate minimizes
\(\|\theta-z\|^2/2+\gamma\sum_t L(\theta;w_t)/m\) over \(\Theta\).
The classical KKT fit uses the train-task mean as its anchor. The
interior ridge fit uses the same prior and parameter box, and includes
only the interior equalities. All regularization weights are selected
on disjoint validation tasks.

For the uniform condition, sample demand from \([0,2]\).
For the informativeness comparison, partition the demand range at
market price breakpoints and classify intervals using the forward solver.
Interior-rich contexts come from the two-interior interval. Mixed
contexts are half two-interior and half one-bound records. Boundary-heavy
contexts are drawn from one one-bound interval per task, so the same unit
stays at the same bound within a context. This design varies both bound
status and the spread of outputs.
