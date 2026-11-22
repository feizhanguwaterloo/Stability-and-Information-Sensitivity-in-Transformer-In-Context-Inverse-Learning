# Information-dependent stability in Transformer in-context learning

## Central question

Can one frozen Transformer recover affine cost parameters uniformly
accurately, remain \(O(1/m)\)-stable to every one-record replacement,
and respond by order one when that record carries an otherwise missing
parameter direction? The question distinguishes worst-case replacements
across valid tasks from typical replacements within one task. The
former can change the correct answer. In a noiseless identifiable
same-task comparison, the correct answer stays fixed.

## A deterministic stability–accuracy proposition

Let \(f_m\) output a parameter vector from a length-\(m\) context.
Suppose \(S,S'\) are admissible, differ in exactly one record, and
have targets \(\theta,\theta'\) at distance \(D\). If their recovery
errors are \(e,e'\), then

\[
\|f_m(S')-f_m(S)\|\ \ge\ (D-e-e')_+ .
\]

Indeed, decompose \(\theta'-\theta\) into the error at \(S'\), the
prediction change, and the error at \(S\), then apply the triangle
inequality. If every neighboring pair in the domain satisfies
\(\|f_m(S')-f_m(S)\|\le C/m\), then for this pair

\[
\max(e,e')\ \ge\ \tfrac12(D-C/m)_+ .
\]

The inequality is sharp on the two-point problem. With a sensitivity
budget \(s\in[0,D]\), place both estimates on the line joining the
targets, distance \(s\) apart and symmetric about their midpoint.
Both errors are \((D-s)/2\). This construction specifies two outputs
and makes no assertion about extension to other contexts.

### Affine example and leverage

For any \(m\ge2\), take \(m-1\) copies of \((0,a_0)\) and one
\((h,a_0+hb_-)\) in \(S\). In \(S'\), replace the last label by
\(a_0+hb_+\). Choose a fixed \(h\ne0\) and fixed \(b_+\ne b_-\)
inside bounded domains. Each context uniquely determines its target
\((a_0,b_\pm)\). The changed label differs by \(|h|D\) with
\(D=|b_+-b_-|\), independent of \(m\).

More generally, put \(u_t=(1,x_t)^\top\) and
\(H=m^{-1}\sum_tu_tu_t^\top\). If the shared rows span a line and
\(v\) is its unit normal, the two targets must differ only in
direction \(v\). For the same focal covariate \(u_*\) in both contexts
and label change \(r\), their exact difference is

\[
\theta'-\theta=\frac{r}{u_*^\top v}v,\qquad
\lambda_{\min}(H)\le v^\top H v=\frac{(u_*^\top v)^2}{m}.
\]

Here \(u_*^\top v\ne0\) is required for identification. This row has
hat leverage \(u_*^\top H^{-1}u_*/m=1\). To see this, use
\(Hv=u_*(u_*^\top v)/m\) and invert. Each context is full rank,
yet its weakest normalized information direction vanishes as the
shared record is repeated. Full rank and a uniform eigenvalue floor
are different conditions.

The exact conflict occurs when \(C/m<D-2\epsilon\) for a desired
error \(\epsilon<D/2\) on both contexts. Thus if one fixed network
has a finite \(C\) independent of \(m\), and its allowed domain
contains such pairs for arbitrarily large \(m\), it cannot achieve
uniform error below \(\epsilon\). This comparison also applies to
a family with \(C_m=o(m)\), but does not restrict families with
\(C_m\) growing at least linearly in \(m\). The existing theorem
assumes bounded tokens, fixed finite weights and depth, full attention,
and mean pooling, with no masks or positional features. Exact
duplicates are admissible here. They have probability zero under
the continuous demand sampler, so the proposition does not by
itself lower-bound average test error or same-task sensitivity.

### Conditioning separates the two cases

If all covariate norms are at most \(R\) and the shared matrix of
rows has a null direction, then \(\lambda_{\min}(H)\le R^2/m\).
Conversely, if \(\lambda_{\min}(H)>R^2/m\), deleting the focal row
leaves positive definite information with the original \(1/m\)
normalization. The noiseless shared labels then fix the target
regardless of the replaced observation. For a possibly noisy label
replacement at fixed covariates, ordinary least squares changes by
\(rH^{-1}u_*/m\), by subtracting the two normal equations. With
\(\lambda_{\min}(H)\ge\kappa>0\), its norm is at most
\(|r|R/(m\kappa)\). Leverage concerns the design. A clean same-task
replacement can have high design leverage and still leave an exact
fit unchanged.

A scalar ridge calculation shows how a fixed penalty smooths the
identifying response. Restrict the intercept to its shared known
value \(a_0\), let \(b_0=(b_++b_-)/2\) be the prior, and fit the slope
using average squared error plus \(\rho(b-b_0)^2\).
The prediction gap is \(D h^2/(h^2+m\rho)\) and each slope error is
\(\tfrac{D}{2}m\rho/(h^2+m\rho)\). These follow by solving the scalar
normal equation and attain the two-point frontier for the resulting
sensitivity. With fixed \(\rho>0\), the response is \(O(1/m)\).
With \(\rho=o(1/m)\), the identifying response tends to \(D\) for
fixed \(h\). The experiment's two-coefficient ridge fits both
intercept and slope and uses the full matrix normal equation.

## A valid economic-dispatch construction

Use the original two-unit solver, capacities \((1,1)\), and identical
cost curves across generators within each task. In task A both
units have \((a,b)=(1.6,1.0)\). In task B both have \((1.2,1.8)\).
Both tasks are within the original coefficient box. Strict
convexity and symmetry give unique dispatch \(x_1=x_2=d/2\).
At \(d=1\) both generators are interior and the clearing price is
2.1 in both tasks. Repeat this market record \(m-1\) times.
At \(d=1.6\) both dispatches are \((0.8,0.8)\), while clearing
prices are 2.40 in A and 2.64 in B. Only the final price changes.
Two distinct interior outputs and the KKT equalities
\(\lambda=a_i+b_i x_i\) give a unique cost vector for each context.
The target separation is \(\sqrt{1.6}\approx1.265\) in the joint
four-parameter Euclidean norm. The worst-case error floor for an
\(O(1/m)\)-stable predictor on this sequence of pairs tends to 0.6325.

## Experiment G design

All models output estimates before any KKT correction. The original
Transformer is loaded from D. The norm-controlled Transformer and
DeepSets are retrained using D's seeds, model configurations and
uniform training contexts. The new statistics MLP uses the same
training protocol. For each generator both statistics methods compute
the interior-only Gram matrix \(H\) and cross moment
\(g=m^{-1}\sum_t\mathbf1_{\mathrm{int}}u_t\lambda_t\).
The 10-feature MLP receives the unique entries of \(H\) and
scaled \(g\). The classical fit uses
\((H+\rho I)^{-1}(g+\rho\theta_0)\), with the training mean
\(\theta_0\) and \(\rho\) selected on independent uniform validation
contexts. These sufficient statistics describe the interior regression
loss. The raw-record models also see bound status and bound prices.

- **Clean same-task regimes.** Use 128 unseen tasks per seed,
  length 16, and four valid replacements per task and regime.
  Compare the uniform test reference, well-conditioned and
  nearly redundant interior contexts, and partially identified
  boundary-heavy contexts. Replacements are uniform in the
  realized context's demand interval. Raw input displacement
  is recorded since regimes have different widths.
- **Matched focal record.** Cross the same focal record and
  replacement with a tightly clustered or widely spread
  background on the same task. Backgrounds are centered at
  the same demand, and left- and right-side focal records
  are both evaluated. All units are interior. This holds fixed
  focal feature location, change direction and input displacement
  within each high/low comparison. The background shape still
  differs from uniform training. In a separate intervention
  keep covariates fixed and replace the measured focal price
  from \(\lambda-\eta\) to \(\lambda+\eta\) for
  \(\eta=0.01,0.04\). These perturbed prices are controlled
  measurement errors, whereas the clean intervention uses
  actual forward outcomes.
- **Identifying pair.** Evaluate the explicit cross-task pair
  at lengths 4, 8, 16, 32 and 64. Measure sensitivity and
  errors against both true targets, plus the signed fraction of
  the target difference recovered in the prediction change.
  This pair is a structured stress test outside ordinary
  uniform-context sampling.

## Experiment G results

The study used three seeds (886, 1729, 2027). Sensitivity is the
absolute Euclidean change in the four predicted coefficients.
Each cell's mean is averaged across seeds. The 95th percentile
and maximum below pool the sampled replacements across seeds.
Neither is a worst-case bound. Recovery is relative Euclidean
error on the unmodified context.

| Clean context | Estimator | Recovery error | Mean sensitivity | P95 | Sampled max |
|---|---|---:|---:|---:|---:|
| Well-conditioned | Transformer | 2.90% | 0.00572 | 0.0172 | 0.0363 |
| | Norm-controlled | 4.58% | 0.0166 | 0.0390 | 0.0595 |
| | DeepSets | 13.04% | 0.0112 | 0.0276 | 0.0525 |
| | Statistics MLP | 13.52% | 0.00998 | 0.0271 | 0.0582 |
| | Ridge | 0.00050% | \(9.27\cdot10^{-7}\) | \(3.21\cdot10^{-6}\) | \(1.19\cdot10^{-5}\) |
| Nearly redundant | Transformer | 13.65% | 0.000116 | 0.000295 | 0.000591 |
| | Norm-controlled | 14.51% | 0.000157 | 0.000423 | 0.000840 |
| | DeepSets | 14.17% | 0.000152 | 0.000410 | 0.000696 |
| | Statistics MLP | 13.94% | 0.000110 | 0.000298 | 0.000513 |
| | Ridge | 2.84% | 0.00379 | 0.0115 | 0.0275 |
| Partially identified | Transformer | 17.50% | 0.00542 | 0.0206 | 0.107 |
| | Norm-controlled | 17.56% | 0.00376 | 0.0147 | 0.0507 |
| | DeepSets | 27.42% | 0.00200 | 0.00851 | 0.0218 |
| | Statistics MLP | 22.42% | 0.00156 | 0.00532 | 0.0157 |
| | Ridge | 13.40% | 0.000185 | 0.000563 | 0.0209 |

The original Transformer's mean sensitivity was 0.00957 on
uniform-reference replacements and 0.000116 in the nearly redundant
regime, where clean replacements also have much smaller input
displacement and its recovery error is higher. All seeds chose
\(\rho=10^{-6}\), the smallest tested penalty on clean validation.
This makes ridge a near-OLS comparator where conditioning permits.
Its partial-context result also depends on the fixed prior for an
unidentified generator. [Full regime diagnostics](results/leverage_stability/regimes_summary.csv)

### Focal record held fixed across backgrounds

Mean focal leverage was 0.999 versus 0.133 in the clustered and spread
backgrounds, with mean maximum per-unit Gram condition numbers of 772
versus 58.9. At \(\eta=0.01\) the input feature displacement was 0.00548
in both backgrounds. The table gives mean parameter sensitivity.

| Model | Clean high | Clean low | Price high | Price low | Price high / low |
|---|---:|---:|---:|---:|---:|
| Transformer | 0.0314 | 0.00197 | 0.0535 | 0.00379 | 14.1 |
| Norm-controlled | 0.0301 | 0.00501 | 0.0288 | 0.00587 | 4.9 |
| DeepSets | 0.00206 | 0.00208 | 0.00185 | 0.00186 | 0.99 |
| Statistics MLP | 0.00230 | 0.00223 | 0.00184 | 0.00183 | 1.01 |
| Ridge | 0.0000720 | 0.000000591 | 0.1488 | 0.01137 | 13.1 |

The Transformer's price-response ratio is close to ridge's, but the
absolute high-leverage response is about 36% of ridge's. Its mean
Euclidean difference from the ridge change vector is 0.0978 in this
stratum. At \(\eta=0.04\) the Transformer's and ridge's high/low ratios
are 13.8 and 13.1. Spearman correlations between condition number
and sensitivity at \(\eta=0.01\), pooling the two backgrounds
within each seed, average 0.765 for the Transformer, 0.285 for the
norm-controlled model, −0.073 for DeepSets, −0.098 for the statistics
MLP and 0.980 for ridge. These describe the matched intervention
distribution. For clean same-task edits the truth does not change
and near-OLS hardly moves, while the original Transformer reacts
strongly to the high-leverage edit. Its response shares ridge's
conditioning dependence under the price intervention, with different
magnitude and different behavior under clean edits. Supplying the
normal-equation statistics to a generic MLP did not reproduce the
ridge response under this training protocol.
[Matched diagnostics](results/leverage_stability/matched_summary.csv) ·
[Condition-number correlations](results/leverage_stability/condition_relationship_summary.csv)

### The one identifying record

| Estimator | Response at \(m=4\) | Response at \(m=64\) | Fraction of true direction recovered at \(m=64\) | Larger absolute error at \(m=64\) |
|---|---:|---:|---:|---:|
| Transformer | 0.726 | 0.603 | 47.4% | 0.432 |
| Norm-controlled | 0.310 | 0.0188 | 1.44% | 0.963 |
| DeepSets | 0.109 | 0.00669 | 0.316% | 0.857 |
| Statistics MLP | 0.120 | 0.00754 | 0.340% | 0.820 |
| Ridge | 1.26481 | 1.26377 | 99.91% | 0.000617 |

The true gap is 1.265. At length 64 the original Transformer's three
responses range from 0.558 to 0.682. Its larger error averaged 0.432,
above the two-point lower bound of approximately 0.331 obtained
by substituting its measured mean response. The norm cap greatly
reduced response on the pair and increased recovery error. The
original Transformer's order-one response over the measured lengths
is compatible with an eventual \(C/m\) bound whose constant is
large. Ridge stays close to the true gap over the tested range
because the chosen penalty is weak compared with the identifying
direction at these lengths.
[Pair diagnostics](results/leverage_stability/identifying_pair_summary.csv) ·
[Experiment G figure](results/leverage_stability/leverage_stability.svg)

## A weight-based numerical upper bound

Write \(R_\ell\) for a uniform row-state bound before layer
\(\ell\) and \(q,k,v\) for operator-norm upper bounds for the
three linear maps of one head. Let \(Q_0,K_0,V_0\) be the
corresponding radius bounds including biases, such as
\(Q_0=qR_\ell+\|b_Q\|\). For head width \(d_h\), put

\[
B_h=\frac{Q_0K_0}{\sqrt{d_h}},\qquad
J_h=e^{2B_h}\left(v+
\frac{2V_0(qK_0+kQ_0)}{\sqrt{d_h}}\right).
\]

The existing row-perturbation proof bounds the mean difference
of head outputs by \(J_h\) times the mean input row difference.
Indeed, the mean logit-weight contribution uses
\(\|p_i-p_i'\|_1\le 2e^{2B_h}(qK_0 e_i+kQ_0\bar e)/\sqrt{d_h}\),
and the value contribution is bounded by \(ve^{2B_h}\bar e\).
Summing head differences before the output map bounds the
attention residual factor by \(1+\|W_O\|\sum_hJ_h\).
The layer's complete factor multiplies this by
\(\operatorname{Lip}(\mathrm{LN}_1)\), by
\(1+\|W_2\|\|W_1\|\), and by
\(\operatorname{Lip}(\mathrm{LN}_2)\). For each normalization
use the maximum absolute gain divided by \(\sqrt{\varepsilon}\).
Its output radius is at most \(\sqrt{d_{\rm model}}\)
times maximum absolute gain plus the bias norm, giving
the next layer's radius. The product over layers is multiplied
by \(2\sqrt6\|W_{\mathrm{embed}}\|\) and the boxed
sigmoid head's Lipschitz bound.

The run evaluates this expression using the smaller of each
matrix's Frobenius and \(\sqrt{\|W\|_1\|W\|_\infty}\) norm bounds.
It records \(\log_{10} C\) to avoid overflow. The mathematical
formula is a real-arithmetic upper bound. The substitution is
ordinary floating-point arithmetic without interval rounding.
The fixed output box also bounds prediction movement by its
diameter. Neither expression is an estimate of the sampled
average or sampled maximum.

Across the three checkpoints, the numerical \(\log_{10} C\) values
range from 616.5 to 708.9 for the original Transformer and from
44.2 to 57.7 for the norm-controlled variant. Even the smallest
weight-based bound at \(m=64\) is vastly above the 1.930 box
diameter. At the measured lengths the diameter is the tighter
certified bound, and the norm calculation does not resolve the
finite-sample gap between architecture variants.
[Weight-norm diagnostics](results/leverage_stability/norm_bounds.csv)

## Relation to existing work

The triangle argument is a deterministic two-point specialization
of the broader stability-constrained estimation perspective.
Chakraborty, Luo and Barber develop statistical minimax lower
bounds using couplings and Hamming perturbations, including a
nonparametric regression setting. Bousquet and Elisseeff study
stability and generalization, including regularized least squares.
The local least-squares response above is a normal-equation
calculation related to classical regression leverage and
influence diagnostics. We do not claim these ingredients as new.
[Chakraborty, Luo and Barber](https://arxiv.org/abs/2601.11701) ·
[Bousquet and Elisseeff](https://www.jmlr.org/papers/v2/bousquet02a.html) ·
[Cook](https://doi.org/10.1080/00401706.1977.10489493)

Li et al. bound sensitivity for a specified normalized
Transformer with a query-token readout and relate stability
to learning across tasks. Our question compares a fixed
mean-pooled parameter estimator's upper bound with a required
response on an exactly identified inverse problem. Akyürek
et al. give explicit constructions for linear-model algorithms
and compare trained predictors with regression baselines.
Hill, Eyre and Creager show failures of the least-squares
interpretation under shifts in linear-regression contexts.
Here the matched intervention probes a conditional change in
predictions with unchanged focal feature locations. It cannot
identify the internal computation.
[Li et al.](https://proceedings.mlr.press/v202/li23l.html) ·
[Akyürek et al.](https://arxiv.org/abs/2211.15661) ·
[Hill, Eyre and Creager](https://arxiv.org/abs/2507.09440)

## Scope of the existing task-scaling result

Experiment E's equal-pass sweep lowered held-out error at
32 observations from 13.48% to 2.72% when increasing source
tasks from 128 to 1,536, while increasing optimizer updates
from 56 to 672. Its equal-step sweep with 448 updates and
batch size 32 gave a nonmonotonic 3.20%–3.56% range. The
matched-record comparison simultaneously changes train context
length, batch size and attention compute. Narrowing the
source support harmed broad-target transfer, but that is a
source-coverage shift rather than an independent task-count
contrast. A task-diversity benefit at fixed compute remains
inconclusive. No additional task-scaling sweep was needed for
the two-point result.
[Experiment E table](results/task_scaling/risks_summary.csv)

## Strongest conclusion and remaining limits

The mathematical result is an incompatibility between **uniformly
small replacement sensitivity and uniformly accurate recovery on
contexts whose only identifying information sits in one record**.
Valid two-unit dispatch records exhibit this obstruction even with
unique dispatch, exact prices, and full-rank data in each context.
For the trained Transformer, the matched experiment separately shows
that leverage can amplify response at a fixed focal feature location.
The stress pair shows that a large response can still recover only
part of the parameter difference.

The lower bound is deterministic and architectural only through
the upper bound used for comparison. An average-case lower bound
would require a distribution that gives enough mass to such
context geometries or a quantitative extension to nearby contexts.
The stress targets have symmetric costs and repeated demands.
They fall within the parameter and observation bounds but are
atypical under continuous independent sampling. The matched price
edits are controlled perturbations rather than a model of market
noise. There are three training seeds and a single cap setting,
and the weight-based bound is too loose to certify a useful
finite-context crossover. The experiments compare input-output
responses and do not establish that an attention head implements
least squares or that explicit sufficient statistics are
intrinsically inadequate for learned estimation.

## Reproduction

Run the following from the project root with the Python environment
used for Experiment D. Its original Transformer checkpoints are
the input to G.

~~~powershell
python -m unittest discover -s tests -v
python -m src.study_leverage --config configs/leverage_study.json
python -m src.study_plots --stage G
~~~

All 12 tests passed. The new tests verify exact KKT validity and
unit leverage for the stress pair, and identical changed features
across the matched backgrounds.

## Follow-up experiments

The follow-up uses the same three seeds and the original two-generator
dispatch problem. The uncapped Transformer uses the Experiment D
checkpoints. All estimates are direct model outputs or standalone ridge
fits.

### Multiple identifying records

For each seed, draw 64 pairs of costs. Pick a pivot dispatch \(x^0\)
and common price \(\lambda_0\). Within each task, give both generators
the same positive slope \(b\) and set
\(a_i=\lambda_0-bx_i^0\). Both tasks therefore have the same optimal
dispatch at the pivot demand. Vary the slope between tasks and use a
second demand for which each unit's output changes by \(h\). The
dispatch is the same under both tasks at both demands, by marginal
cost equality and strict convexity. Choose the slope difference and
\(|h|\) to fix the joint parameter separation at \(0.6\) and the
focal price gap at \(0.09\). Both endpoints have full-rank interior
designs and costs inside the original box.

Use \(m\in\{16,32,64\}\) and replace \(k\in\{1,2,4,8\}\) of the
repeated pivot records with copies of the focal record. The endpoint
contexts differ in exactly \(k\) prices. The pairs and focal locations
are shared across all \((m,k)\) settings. From the previous
two-point proof, the larger recovery error is at least
\((0.6-\|f(S')-f(S)\|)_+/2\). If the one-record upper bound holds on
the price-mixing path, telescoping gives the lower bound
\((0.6-kC/m)_+/2\). For \(k>1\), the intermediate contexts mix the
two prices at the same focal dispatch. They are inputs to the
Transformer for measuring adjacent prediction changes and have
no noiseless recovery target. The structural Transformer bound
applies to this path because it only requires bounded token
features. Recovery is evaluated at the two valid endpoints.

### Norm caps, price noise, and matched transfer

- **Attention caps.** Train with Q, K, V and attention-output
  operator norm caps of \(0.50,0.85,1.25,2.00\) using the same
  tasks, draws, optimizer schedule and validation checkpoint
  rule as D. Compare with the original uncapped checkpoints.
  We verify the trained operator norms and measure both ordinary
  same-task replacement sensitivity and the identifying response.
- **Price noise.** Independently perturb every price in a
  16-observation context with zero-mean Gaussian noise, at
  standard deviations \(0,0.01,0.04,0.10\). Keep demand,
  dispatch and statuses exact so the interior Gram matrices
  stay fixed. Use four independent noise draws per task,
  shared across methods and scaled across noise levels.
  Compare the frozen, clean-trained Transformer with ridge
  clipped to the original coefficient box. Tune the ridge
  penalty on separate tasks for each regime and noise level,
  using the same clipping for validation and test.
- **Matched task counts.** Train on nested subsets of
  \(128,512,1536\) source tasks. Each run uses the same initial
  weights for a given seed, 384 updates, batches of 32 and 16
  records per context. This gives exactly 12,288 context
  presentations and 196,608 records per model. A separate
  random stream holds the sequence of training demand draws
  fixed across task counts. Evaluate the final checkpoints on
  the same unseen tasks and evaluation contexts. Smaller
  source sets necessarily repeat tasks more often.

### Follow-up results

All table entries are means across the three seeds. The figures and
CSV tables include across-seed standard deviations.

**Identifying records.** At \(m=64\), more copies of the identifying
record modestly increased the uncapped Transformer's response along
the true parameter difference. The largest step along the evaluated
replacement path decreased as the information was spread across
more records. Both effects occurred in the generated family of 192
task pairs with the same endpoint separation.

| Identifying records | Uncapped response along true direction | Largest path step | Larger uncapped L2 error | Larger ridge L2 error |
|---:|---:|---:|---:|---:|
| 1 | 50.0% | 0.301 | 0.281 | 0.000540 |
| 2 | 52.3% | 0.206 | 0.272 | 0.000275 |
| 4 | 53.8% | 0.128 | 0.266 | 0.000142 |
| 8 | 54.7% | 0.0736 | 0.262 | 0.0000761 |

These are constructed contexts with repeated demands. The sample
path steps describe model outputs on that path. The theoretical
bound applies to arbitrary adjacent inputs in its domain.
[Identifying-record results](results/leverage_followup/identification_summary.csv)

**Attention caps.** At \(m=64\), all selected checkpoints obeyed
their caps to numerical precision. Tightening the cap from 2.00 to
0.50 reduced the identifying-record response and increased its
recovery error. Over the same settings, average sensitivity to
ordinary same-task replacements increased. The effects of a norm
cap depend on the replacement protocol and trained checkpoint.

| Cap | Ordinary mean L2 change | Ordinary relative error | Identifying direction recovered, \(k=1\) | Larger L2 error, \(k=1\) |
|---|---:|---:|---:|---:|
| 0.50 | 0.00847 | 5.53% | 3.18% | 0.581 |
| 0.85 | 0.00552 | 3.96% | 26.1% | 0.504 |
| 1.25 | 0.00387 | 3.36% | 23.7% | 0.433 |
| 2.00 | 0.00249 | 2.59% | 39.5% | 0.317 |
| Uncapped | 0.00234 | 2.65% | 50.0% | 0.281 |

[Same-task cap sweep, including P95 and sampled maxima](results/leverage_followup/cap_sweep_summary.csv) ·
[Measured operator norms](results/leverage_followup/cap_norms.csv)

**Price noise.** Mean minimum interior Gram eigenvalues were
0.0326 in well-conditioned contexts, \(3.72\cdot10^{-6}\) in nearly
redundant contexts, and zero in boundary-heavy contexts. At
\(\sigma=0.04\), the frozen Transformer averaged **13.67%** error
against **15.28%** for tuned ridge in the nearly redundant regime.
Ridge averaged **2.39%** against **3.87%** in the well-conditioned
regime and **15.50%** against **17.41%** in the boundary-heavy regime.
At \(\sigma=0.10\) in the nearly redundant regime, the difference
narrowed to **15.29%** versus **15.38%**. The advantage at intermediate
noise is conditional on this observation design, the clean-trained
Transformer and the box-clipped ridge comparison. Only prices were
noisy, and the Transformer training used clean prices.
[Noise results](results/leverage_followup/noise_summary.csv) ·
[Validation-only ridge tuning](results/leverage_followup/noise_tuning.csv)

**Matched transfer.** At 32 evaluation records, unseen relative error
was **3.48%, 3.25%, and 3.54%** for 128, 512, and 1,536 source tasks.
The 512-task set had the lowest mean error at all five evaluation
lengths in this budget. A monotonic benefit from increasing task count
remains unsupported under fixed updates, record counts, batch size,
training context length, and checkpoint time. Three seeds and one
training budget leave the optimal source-task count unresolved.
[Matched transfer results](results/leverage_followup/matched_transfer_summary.csv)

[Follow-up figure, PNG](results/leverage_followup/followup_experiments.png) ·
[Follow-up figure, SVG](results/leverage_followup/followup_experiments.svg) ·
[Saved configuration](results/leverage_followup/config.json)

Run the follow-up and render its figure from the project root using
the same Python environment as Experiment G.

~~~powershell
python -m unittest discover -s tests -v
python -m src.study_followup
python -m src.study_plots --stage followup
~~~

All 14 tests passed. The two additional tests verify valid and
identifying endpoints, one-price changes along the replacement path,
and preservation of the Gram matrix under price noise.
