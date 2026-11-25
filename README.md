# Stability and Information Sensitivity in Transformer In-Context Inverse Learning

The course submission uses the official ICML 2026 template with
three pages of main text. [Report](COURSE_PROJECT_REPORT.pdf) ·
[Reproducibility instructions](REPRODUCIBILITY.md) ·
[Submission verification](COURSE_PROJECT_VERIFICATION.md)

The 18-minute presentation uses a running identifying-record example to
connect the research question, stability and recovery proofs, and four
empirical questions. It has 19 main slides and 8 backup slides.
[Beamer slides](COURSE_PROJECT_SLIDES_V2.pdf) ·
[LaTeX source](COURSE_PROJECT_SLIDES_V2.tex) ·
[Standalone Beamer source bundle](CS886_BEAMER_SOURCE.zip) ·
[Timing and speaking guide](COURSE_PROJECT_PRESENTATION_OUTLINE.md)

This prototype estimates the quadratic costs of two generators from
dispatch and clearing-price records. It compares a frozen Transformer,
two classical estimators, and a postprocessing KKT correction on unseen
synthetic tasks.

## Transformer-centered study

Experiments D–F study the Transformer itself. D compares attention
architectures, E varies task counts, diversity, and training budgets,
and F intervenes on valid economic-dispatch contexts. KKT geometry
provides information diagnostics and exact classical comparisons.
Use the same Python environment as for the pilot.

~~~powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe -m src.study --config configs/transformer_study.json --output results
.\.venv\Scripts\python.exe -m src.study_plots --output results
~~~

The runner executes D, E, then F. Pass --stage D, --stage E, or --stage F
to run a single experiment. E and F reuse D's original Transformer
checkpoints and require D to have run with the same config and output
root. Each study directory holds its configuration and seed-level and
aggregate CSV tables. Figures are available in PNG and editable SVG.

- [Updated research plan](TRANSFORMER_RESEARCH_PLAN.md)
- [Transformer study report](TRANSFORMER_PROJECT_UPDATE.md)
- [D architecture stability](results/transformer_stability/)
- [E task scaling](results/task_scaling/)
- [F context informativeness](results/context_informativeness/)

## Information-dependent stability

Experiment G reuses the original D checkpoints and adds matched
leverage interventions and a valid identifying-record stress test.
The report includes the deterministic stability–accuracy proposition.

~~~powershell
.\.venv\Scripts\python.exe -m src.study_leverage
.\.venv\Scripts\python.exe -m src.study_plots --stage G
~~~

The follow-up varies the number of identifying records, sweeps
attention norm caps, adds independent noise to every observed price,
and compares task counts under exactly matched training budgets.

~~~powershell
.\.venv\Scripts\python.exe -m src.study_followup
.\.venv\Scripts\python.exe -m src.study_plots --stage followup
~~~

[Theory direction and Experiment G](THEORY_RESEARCH_DIRECTION.md)

## Setup and original pilot

Use standard CPython 3.11 or 3.12 and run the commands from the project
root in PowerShell.

~~~powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe -m src.evaluate --config configs/pilot.json --output results/pilot
~~~

The default configuration selects CUDA if available in the installed
PyTorch build, otherwise CPU. Set the configuration's device to "cpu"
to force CPU. Seeds and deterministic PyTorch operations are enabled.
The exact result can depend on runtime versions and device, recorded in
the run metadata. For Linux or macOS, replace the venv executable with
./.venv/bin/python.

## Experiments

- A uses nested uniform-demand contexts of 4, 8, 16, 32, and 64 records.
- B holds the context size at 32 and changes the operating regime.
- C independently replaces one record four times per task and length.
  Its error bars are 1.96 task-level standard errors after averaging the
  four trials per task. The reference 1/m line is anchored at the
  uncorrected mean for m = 4.

Relative parameter error divides the Euclidean parameter error by the
true parameter norm. Marginal-cost RMSE uses the true dispatch at new
demands. Dispatch RMSE solves the forward model with each estimate at
those same demands. KKT RMSE measures agreement with context records.
Each summary averages task-level values. The recovery CSV also records
the minimum interior Gram eigenvalue across the two generators, with
each Gram matrix normalized by total context size.

## Project files

- [Original research plan](RESEARCH_PLAN.md)
- [Problem formulation](docs/problem_formulation.md)
- [Proofs and counterexample](docs/theory.md)
- [Literature notes](docs/literature.md)
- [Pilot configuration](configs/pilot.json)
- [Report and reproduction commands](PROJECT_REPORT.md)
- [Pilot result files](results/pilot/)
