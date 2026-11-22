# KKT-aware in-context inverse optimization

## Completed work

Phases 1–3 are complete for the two-generator, noiseless pilot. The
prototype includes the analytical forward solver, independent SLSQP
verification, classical inverse baselines, Transformer training across
tasks, boxed KKT correction, and experiments A–C. The pilot used seed 886
and 1,536 train, 160 validation, and 256 test tasks. The model had two
64-dimensional encoder layers, four heads, and mean pooling. It was
frozen during testing. Validation selected epoch 16 and a KKT or ridge
weight of 1,000 for each of the three regularized estimators.

## Mathematically proven

- The anchored KKT fit has a unique minimizer. If the anchor changes by
  at most \(K_T/m\) per record replacement and the gradient bound \(G\)
  applies to the sum of both generator losses for one record, the
  corrected estimator changes by at most \((K_T+2\gamma G)/m\).
- A separate proof establishes an exact-arithmetic \(O(1/m)\) anchor
  rate for the fixed full-attention, bounded-input, mean-pooled encoder
  under the structural assumptions in the theory notes. Its constant
  was left uncalibrated.
- Noiseless, unconstrained interior correction transforms the anchor
  error by \((I+\gamma H_m)^{-1}\). For mixed observations,
  identifiability is determined by the geometry of an intersection of
  affine equalities, half-planes, and the coefficient box. A bound-only
  counterexample remains ambiguous for arbitrarily many observations.

See [theory and proofs](docs/theory.md) and [literature notes](docs/literature.md).

## Numerical observations

All values below are averages over the 256 test tasks. Parameter error
is the Euclidean estimation error divided by the true parameter norm.

| Experiment and condition | Transformer | Transformer + KKT | Classical KKT | Interior ridge |
|---|---:|---:|---:|---:|
| A · Uniform, 4 records | 5.523% | **0.851%** | 1.460% | 1.695% |
| A · Uniform, 64 records | 2.532% | **0.042%** | 0.305% | 0.306% |
| B · Interior-rich, 32 records | 2.949% | **0.110%** | 0.521% | 0.521% |
| B · Mixed, 32 records | 3.100% | **0.060%** | 0.340% | 0.347% |
| B · Boundary-heavy, 32 records | 18.457% | 15.200% | **11.643%** | 14.566% |

- **A.** At 64 records the correction lowered Transformer held-out
  dispatch RMSE from 0.01044 to 0.0000654, marginal-cost RMSE from
  0.02893 to 0.000219, and context KKT residual RMSE from 0.02658 to
  0.000184. The true-parameter oracle gives zero dispatch and
  marginal-cost error.
- **B.** Both units had numerically full-rank interior designs in all
  256 interior-rich and mixed contexts, using a minimum-eigenvalue
  threshold of \(10^{-12}\). None of the boundary-heavy contexts did.
  Mixed contexts had a larger mean minimum Gram eigenvalue than
  interior-rich contexts despite fewer both-interior records, so
  operating-status counts alone do not summarize information.
  The learned anchor underperformed the classical KKT anchor under the
  boundary-heavy demand shift.
- **C.** The mean one-record parameter change for the Transformer was
  0.08748 at 4 records and 0.002331 at 64. For the corrected estimator
  it was 0.02281 and 0.0000515 respectively. The corrected mean was
  smaller at all five lengths. These averages characterize the
  sampled replacements. The theoretical bound concerns worst-case
  replacements under its assumptions.

[A figure](results/pilot/A_context_length.png) ·
[B figure](results/pilot/B_informativeness.png) ·
[C figure](results/pilot/C_stability.png) ·
[Recovery CSV](results/pilot/recovery_summary.csv) ·
[Stability CSV](results/pilot/stability_summary.csv) ·
[Run metadata](results/pilot/run.json)

## Limitations and open questions

This is one seed, with exact prices, two fixed-capacity generators,
uniform training demands, and an assumed known cost box. Regime B changes
both the observed bound pattern and the demand distribution. The
sampling intervals depend on task costs, so their endpoints can carry
information beyond the KKT constraints on the realized records. All three
regularized methods chose the largest tested weight, leaving the
finite-regularization comparison sensitive to the search range.
Numerical full rank already held for 251 of 256 uniform contexts with
four observations and all 256 at eight or more. In full-rank noiseless
contexts, direct equality regression can recover the parameters exactly
up to numerical precision.

The next comparison should include a classical fit that enforces the
observed KKT constraints exactly, with the same fixed or learned anchor
used only to select among compatible parameters. Vary price noise,
repeat across seeds, and report results separately by interior rank
and Gram eigenvalue. Whether a learned anchor adds value under noise
or partial identification remains open. Phase 4 extensions are future
work.

## Exact commands used for the CPU run

The desktop's bundled CPython was used because the shell's default
interpreter lacked the required scientific packages.

~~~powershell
$runtime = 'C:/Users/f/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe'
& $runtime -m pip --isolated --disable-pip-version-check install --no-cache-dir --target .deps scipy matplotlib
& $runtime -m pip --isolated --disable-pip-version-check install --no-cache-dir --target .deps --index-url https://download.pytorch.org/whl/cpu torch
$env:PYTHONPATH = (Resolve-Path .deps).Path
$env:MPLCONFIGDIR = Join-Path (Get-Location) '.mplconfig'
& $runtime -m unittest discover -s tests -v
& $runtime -m src.evaluate --config configs/pilot.json --output results/pilot
~~~

Seven tests passed. The forward solver was compared with 120 independent
SLSQP solves, and the tests covered degenerate pricing, valid context
construction, KKT signs and derivatives, correction feasibility and the
closed-form interior fit, and Transformer permutation invariance.
An initial experiment stopped when the optimizer line search returned an
abnormal status at a point already satisfying projected stationarity.
The final implementation checks box feasibility and the projected
gradient directly, and the complete run passed those checks.
