# Reproducing the CS 886 course project

## Setup

Run from the extracted archive root with standard CPython 3.11 or 3.12.

~~~powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
~~~

The archive includes measured package versions in REPRODUCIBILITY_ENVIRONMENT.json. For CPU execution the study configuration specifies four threads and deterministic PyTorch algorithms. Results across library versions or hardware may differ at floating-point precision.

## Rebuild the report from saved results

~~~powershell
.\.venv\Scripts\python.exe scripts/build_report_assets.py
tectonic COURSE_PROJECT_REPORT.tex
~~~

Alternatively compile with latexmk -pdf COURSE_PROJECT_REPORT.tex. The official icml2026.sty, icml2026.bst, and its companion fancyhdr.sty are included unmodified. The report uses supported preprint mode and default fonts, margins and spacing. The source and bibliography are in the archive root.

[Official ICML 2026 template](https://icml.cc/Conferences/2026/AuthorInstructions)

The asset script reads saved CSV summaries and generates the tables and vector figure under results/course_report/. The root PDF has three main pages followed by references and appendices.

## Rebuild the Beamer presentation

~~~powershell
.\.venv\Scripts\python.exe scripts/build_slide_assets.py
tectonic COURSE_PROJECT_SLIDES_V3.tex
.\.venv\Scripts\python.exe scripts/package_slides.py
~~~

The deck contains 19 main slides planned for 18 minutes and 8 backup slides.
Its source includes speaker prompts. COURSE_PROJECT_PRESENTATION_OUTLINE.md
gives per-slide timing, rehearsal advice and questions to prepare for.
CS886_BEAMER_SOURCE.zip also compiles independently after extraction and
can be uploaded to Overleaf with COURSE_PROJECT_SLIDES_V3.tex as the main file.

## Rerun the experiments

Run these commands in order from the root. E, F and G reuse D's original Transformer checkpoints and the same configuration.

~~~powershell
.\.venv\Scripts\python.exe -m src.study --config configs/transformer_study.json --output results --stage D
.\.venv\Scripts\python.exe -m src.study --config configs/transformer_study.json --output results --stage E
.\.venv\Scripts\python.exe -m src.study --config configs/transformer_study.json --output results --stage F
.\.venv\Scripts\python.exe -m src.study_leverage --config configs/leverage_study.json --root results
.\.venv\Scripts\python.exe -m src.study_followup --config configs/followup.json --root results
.\.venv\Scripts\python.exe -m src.study_plots --stage all
.\.venv\Scripts\python.exe -m src.study_plots --stage G
.\.venv\Scripts\python.exe -m src.study_plots --stage followup
~~~

These commands replace the corresponding saved result files within the extracted archive. A second extraction provides a reference copy for comparisons. On Linux or macOS replace the venv executable with ./.venv/bin/python.

## Data and metrics

- Source costs are independent uniform draws with intercepts in [1,1.65] and slopes in [0.8,2]. Default training demands are uniform on [0,2]. The distributions and solver are in src/data.py and src/dispatch.py.
- Seeds are 886, 1729 and 2027. D and G use 1,536 source tasks, 96 validation tasks and 128 target tasks per seed.
- Relative error divides Euclidean parameter error by the true parameter norm. Delta describes Euclidean prediction change under the stated replacement protocol.
- Transfer risk is mean squared coefficient error divided coordinatewise by the parameter span and averaged over coefficients. Source and target risk estimates use fresh contexts on their respective task sets.
- Directional response divides the inner product of predicted and true parameter differences by the squared true difference. A value of one indicates the full response along that direction.
- Mean and SD columns summarize seed-level averages. G and the follow-up also report explicitly named pooled quantiles and sampled maxima.

## Report provenance

| Report content | Saved evidence |
|---|---|
| D architecture and replacement curves | results/transformer_stability/recovery_summary.csv and stability_summary.csv |
| E task counts, risks and update budgets | results/task_scaling/risks_summary.csv and risks_seeds.csv |
| F information and conditioning | results/context_informativeness/contexts_summary.csv and contexts_raw.csv |
| G matched price intervention | results/leverage_stability/matched_summary.csv and matched_raw.csv |
| G identifying pair | results/leverage_stability/identifying_pair_summary.csv and identifying_pair_raw.csv |
| Follow-up replication and controls | results/leverage_followup/ |
| Proofs and construction checks | Report appendices, docs/theory.md, tests/test_leverage.py, and tests/test_followup.py |

## Submission contents

COURSE_PROJECT_REPORT.pdf is the report. CS886_REPRODUCIBILITY.zip contains the report source, bibliography, official template, source code, tests, configurations, checkpoints, numerical results and build assets. Its REPRODUCIBILITY_MANIFEST.sha256 records the packaged file hashes.

The supplied screenshot gives a deadline of November 26, 2026 at 11:59 p.m. Toronto time, and requires the reproducibility materials in the next submission question. The linked full project instructions and that later upload question were outside the supplied screenshot.
