# Stability and Transfer Generalization of Transformer In-Context Learning under Optimization-Induced Observations

The inverse economic-dispatch task provides contexts whose information
content can be computed. A record with an interior generator supplies
an equality for its cost coefficients. A record with a bound generator
supplies a half-plane. This lets us ask how a frozen Transformer reacts
to observations of different leverage, and how its performance on new
tasks depends on source-task count, training budget, and task diversity.

[Updated research plan](TRANSFORMER_RESEARCH_PLAN.md) ·
[Original pilot report](PROJECT_REPORT.md) ·
[Study configuration](configs/transformer_study.json)

## Methodology

The three independent seeds are 886, 1729, and 2027. For each, source,
validation, and test tasks are independent draws from the original
cost box. There are 96 validation and 128 unseen test tasks per seed.
Task subsets for the scaling curves are nested. Checkpoints are
chosen by held-out validation relative error averaged over five
context lengths. Each reported SD is across seed-level means.

- **D. Architecture and stability.** Train the original 67,908-parameter
  Transformer, the same architecture with projected operator norms,
  and a 63,396-parameter DeepSets model on 1,536 tasks with the same
  training contexts and optimization budget. The cap is 0.85 for
  each layer's Q, K, V, and output weight matrices, applied at
  initialization and after every optimizer step. The saved diagnostics
  check the actual selected checkpoints. Each task contributes four
  paired same-task record replacements per length. The reported
  empirical mean, 95th percentile, and maximum refer to those sampled
  replacements. The fixed-weight KKT correction is secondary.
- **E. Task scaling and transfer.** Run \(T=128,256,512,1024,1536\)
  with 14 passes per task and separately with 448 total optimizer
  updates. At equal steps, batches of 32 expose models to a similar
  number of record presentations but different task repetition rates.
  The matched-record comparison uses 1,024 tasks at context length 8
  and 256 tasks at length 32. With batches of 128 and 32 respectively,
  both use 114,688 training record presentations and 112 updates.
  Attention compute still differs by context length. A diversity
  control at \(T=512\) compares the full box with a centered source box
  of 35% width per axis, evaluated on the same original-box test tasks.
  Source and target risks use two independent evaluation contexts per
  task and length. Risk is mean squared parameter error after dividing
  each coefficient by its box span. The transfer gap is target risk
  minus source risk. Shifted tests draw a Beta(5, 1.5) density within
  the original box and pair demands with the uniform-density test set.
- **F. Information and interventions.** For each fixed task, use a
  uniform-demand grid of forward solutions and a supplemental pool
  of valid records from a single bound interval. Well-conditioned and
  nearly redundant contexts have the same mean demand and all units
  interior. Their demand spread differs by construction. Mixed and
  boundary-heavy contexts use the same task and common pools, with
  residual demand differences recorded by Wasserstein distance. In a
  separate all-interior context, move a high-leverage or redundant
  record by the same demand increment and the same feature-space
  distance. Delete those records in paired equal-size ablations.
  In mixed contexts, retain either the interior or the boundary
  half at equal length. Shuffle the full well-conditioned context
  as a permutation check.

In D and F, the exact-KKT baseline projects either the source mean or
the Transformer's prediction onto the same noiseless admissible set.
This isolates the effect of the anchor where the set has positive
diameter. A singleton identifies both generators and leaves no
dependence on the anchor.

## Empirical findings

### D. Architecture, sensitivity, and recovery

| Model | Mean replacement change at \(m=4\) | Mean replacement change at \(m=64\) | Relative parameter error at \(m=64\) |
|---|---:|---:|---:|
| Transformer | 0.07750 ± 0.00169 | 0.002132 ± 0.000224 | 2.65% ± 0.28 pp |
| Norm-controlled Transformer | 0.11445 ± 0.00617 | 0.005103 ± 0.000361 | 3.95% ± 0.20 pp |
| DeepSets | 0.11455 ± 0.00518 | 0.007600 ± 0.000328 | 10.28% ± 0.35 pp |
| Transformer + KKT | 0.02332 ± 0.00129 | 0.0000481 ± 0.0000078 | 0.044% ± 0.006 pp |

Among the three uncorrected architectures, the original Transformer
had the lowest mean replacement sensitivity and recovery error at
each tested length. The largest selected attention operator norm was
3.19 for the original encoder and 0.85 for the controlled encoder.
The cap therefore changed the trained maps substantially, but the
controlled encoder was empirically more sensitive and less accurate.
The observed association between stability and recovery is conditional
on the architectures, cap and training budget. The \(1/m\) reference is
anchored to the original Transformer's mean at four records, and the
sampled upper tails provide empirical diagnostics rather than a
worst-case certificate.

At \(m=64\), exact-KKT estimators recover the parameters to numerical
precision on these noiseless singleton cases. At \(m=4\), only 13 of
384 tasks had a positive-diameter admissible set. On that subset, the
seed-averaged errors were 8.07% for the fixed prior and 6.73% for the
learned anchor under the same exact projection. F evaluates a larger
partially identified stratum under a different context distribution.

[Architecture figure](results/transformer_stability/architecture_comparison.svg) ·
[Stability CSV](results/transformer_stability/stability_summary.csv) ·
[Recovery by identification](results/transformer_stability/recovery_by_identification_summary.csv)

### E. Distinct tasks and optimization budget

The table gives held-out parameter error at \(m=32\) under the
original uniform test distribution.

| Source tasks | Equal task passes | Equal optimizer steps |
|---:|---:|---:|
| 128 | 13.48% ± 0.32 pp | 3.29% ± 0.19 pp |
| 256 | 10.45% ± 0.10 pp | 3.47% ± 0.29 pp |
| 512 | 7.46% ± 0.48 pp | 3.54% ± 0.14 pp |
| 1,024 | 3.20% ± 0.07 pp | 3.20% ± 0.07 pp |
| 1,536 | 2.72% ± 0.29 pp | 3.56% ± 0.53 pp |

The equal-pass sweep spent 56 to 672 updates as \(T\) increased.
At 448 updates, the task-count curve was nonmonotonic. For example,
at equal steps and \(T=128\), the source and transfer risks were
0.002169 and 0.002686, a gap of 0.000517. At \(T=1536\), they were
0.003212 and 0.003218, a gap of 0.000006 despite a higher absolute
transfer risk. A smaller source-to-target gap can coexist with a
less accurate estimator.

At the matched record and update budget, 1,024 tasks with length 8
gave 8.82% ± 0.56 pp error versus 10.58% ± 0.60 pp for 256 tasks
with length 32 at \(m=32\) evaluation. This contrast also changes
training context length and batch size. At \(T=512\), narrowing the
source support raised original-box error from 7.46% to 15.43% at the
same update and record budgets, even as the narrow-source population
risk decreased. This shows a source-coverage effect in the chosen
shifted-support comparison. Beta-density results are tabulated
separately. For instance, equal-pass \(T=128\) risks were 0.04189 on
the uniform target and 0.06735 under the Beta shift, while for
\(T=1536\) they were 0.001915 and 0.001830 respectively. Reweighting
tasks within the box need not increase average error.

[Task scaling figure](results/task_scaling/transfer_scaling.svg) ·
[Budget and diversity controls](results/task_scaling/transfer_controls.svg) ·
[Source, transfer, and shift risks](results/task_scaling/risks_summary.csv)

### F. Information-dependent behavior

| Context | Mean minimum Gram eigenvalue | Mean admissible diameter | Transformer error |
|---|---:|---:|---:|
| Well-conditioned | 0.03257 | 0 | 2.90% ± 0.30 pp |
| Nearly redundant | \(3.72\times10^{-6}\) | 0 | 13.65% ± 0.57 pp |
| Mixed | 0.02429 | 0 | 3.72% ± 0.08 pp |
| Boundary-heavy | 0 | 1.056 | 17.64% ± 0.72 pp |

The all-interior contexts have exactly matched mean demand. Their
average within-context demand SDs are 0.450 and 0.00493. Mean demand
Wasserstein distances to the well-conditioned contexts are 0.386
for nearly redundant, 0.444 for mixed, and 0.889 for boundary-heavy
contexts. Since dispatch status is a deterministic function of
demand within a task, status cannot generally be changed while
holding the entire within-task demand distribution fixed. The
well-conditioned, near-redundant, and mixed contexts are point
identified. Exact KKT recovers them to numerical precision. The
boundary-heavy contexts are partially identified, with 10.16%
relative error for exact projection from the fixed prior versus
13.09% from the learned anchor. Transformer-only marginal-cost RMSE
was 0.0343 in well-conditioned contexts and 0.1835 in boundary-heavy
contexts.

In the paired replacement intervention, the common mean feature
displacement was 0.137. Moving the high-leverage record gave mean
prediction change 0.0580 ± 0.0081 versus 0.0111 ± 0.0025 for moving
the redundant record, about 5.2 times larger. The Gram eigenvalue
decreased by 0.00108 on average for the high-leverage change and
changed only slightly for the redundant replacement. Removing the
high-leverage record increased recovery error by 5.05 percentage
points, while removing the redundant record changed it by −0.05
points. At equal post-ablation size, retaining only the bound records
from mixed contexts increased recovery error by 14.01 points,
while keeping the interior records changed it by −0.79 points.
Shuffling changed predictions by \(1.20\times10^{-7}\) on average,
consistent with floating-point evaluation of a permutation-invariant
model. The matched displacements support differential sensitivity
to record leverage. Record locations differ by construction, so
these interventions do not identify an internal attention mechanism.

[Information figure](results/context_informativeness/context_information.svg) ·
[Intervention figure](results/context_informativeness/context_interventions.svg) ·
[Intervention CSV](results/context_informativeness/interventions_summary.csv)

## Mathematical guarantees and relation to prior work

**Theorem A** proves \(O(1/m)\) worst-case single-record replacement
stability in exact arithmetic for the fixed bounded-input, finite-depth
Transformer, including the actual residual and post-LayerNorm maps.
The proof propagates mean row perturbations through unmasked full
attention. The same rate follows by direct averaging for DeepSets.
The model-specific constants are uncalibrated. Li et al.'s Theorem 3.2
uses a normalized Transformer with a query-token readout.

**Theorem B** gives a task-risk covering-number bound at fixed context
length for IID source and target tasks and a fixed hypothesis class.
It uses Hoeffding concentration over tasks and a finite cover in the
task-risk sup norm. This is an application of standard uniform
convergence. Li et al.'s Section 4 studies transfer through task
diversity, while their multitask analysis handles dependent prefix
losses. Our same-distribution bound supplies no guarantee for the
Beta density shift.

**Proposition C** separates detectable KKT inconsistency from
uncertainty within the admissible set. Exact projection improves
Euclidean error when the true parameter is feasible. The diameter
controls ambiguity after projection. Two tasks with the same
observation law yield a minimax lower bound of half their Euclidean
parameter separation. This obstruction holds even for stable
estimators. These results use standard projection and concentration
arguments and leave sharper Transformer-specific learning guarantees open.

[Full proofs and architecture audit](docs/theory.md) ·
[Li et al., ICML 2023](https://proceedings.mlr.press/v202/li23l.html)

## Limitations and publication directions

This is a three-seed, noiseless, two-unit study. Architecture comparisons
use one cap and a common optimizer schedule, so they establish behavior
under that protocol. The clipped matrices were verified, while attention
biases and the remaining operators were unconstrained. The equal-step
comparison matches updates and roughly matches record presentations,
while selected checkpoint times and realized attention compute can
differ. The matched context controls do not eliminate all support shift.

A follow-up study can distinguish architectural leverage learning from
input-location effects by adding a hand-designed summary of sufficient
statistics and stronger rowwise controls. Sweeping the norm cap and
calibrating score and layer constants would test how the proof relates
to measured sensitivity. A price-noise experiment stratified by Gram
eigenvalue and admissible diameter would test whether a learned prior
adds value where exact regression is poorly conditioned or fails.

## Reproduction

Run from the project root. The reported experiments used the desktop's
bundled CPython and the previously installed local scientific packages.

~~~powershell
$env:PYTHONPATH = (Resolve-Path .deps).Path
$env:MPLCONFIGDIR = Join-Path (Get-Location) '.mplconfig'
$runtime = 'C:/Users/f/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe'
& $runtime -m unittest discover -s tests -v
& $runtime -m src.study --stage D
& $runtime -m src.study --stage E
& $runtime -m src.study --stage F
& $runtime -m src.study_plots
~~~

All 10 tests passed. An isolated run of the original A–C pipeline
produced byte-identical files to the existing pilot.
