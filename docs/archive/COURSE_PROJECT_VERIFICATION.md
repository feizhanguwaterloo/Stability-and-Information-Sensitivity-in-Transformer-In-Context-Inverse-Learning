# CS 886 final verification

| Check | Result |
|---|---|
| Transformer stability proof | Compared with the frozen encoder implementation. Checked the bounded score and softmax Jacobian bounds, concatenation, attention residual, both positive-epsilon LayerNorm maps, feedforward residual, mean pooling, and sigmoid readout. |
| Stability and accuracy tradeoff | Verified the triangle inequality and the positive-part lower bound. The two cost vectors are in the coefficient box. Both demands yield unique interior dispatch with the stated prices, and each endpoint has two distinct outputs per generator. |
| Supporting transfer bound | Checked bounded loss, IID source tasks, the fixed hypothesis class, the sup-norm cover, and the Hoeffding and union bound steps. The empirical source and target risks use fresh contexts on the respective task sets. |
| Numerical results | Matched the displayed rows in Tables 1–3 against the saved D–F CSVs. Checked the G leverage ratios and signed directional response against the corresponding G summary tables, and checked the cited follow-up values against their CSVs. |
| Citations | Checked the linked proceedings, author-hosted paper and preprints. The report distinguishes the Li et al. query-token theorem from the mean-pooled inverse estimator. |
| Existing unit tests | All 14 passed, including the exact KKT and rank checks for the one-record and multiple-record identifying pairs. |
| LaTeX and PDF | Compiled with portable Tectonic 0.16.9. The 9-page PDF has 6 main pages, 1 references page and 2 appendix pages. Rendered all pages with Poppler and reviewed the layout. Verified cross-references and extracted key figure text. The final TeX log has no box, missing-character or undefined-reference warnings. |
| Experiment scope | Reused the saved experiment outputs. The earlier tests, code and results supply the report's evidence. |

## Reproduction

Run from the project root with the existing Python environment and a LaTeX compiler. The source references the saved Experiment G figure by its repository-relative path.

~~~powershell
python -m unittest discover -s tests -v
tectonic COURSE_PROJECT_REPORT.tex
~~~

## Remaining limitations

- The weight-based stability constant is too loose to predict empirical differences at the tested lengths. The deterministic error floor applies to the specified cross-task context family.
- The study uses a two-generator synthetic model, mostly exact prices, and three seeds. The leverage interventions measure the predictor's response at its output.
- The number of source tasks did not have a consistent monotone effect under the matched budgets. The report's title block lists the course project because no submission author name or student number was supplied.
