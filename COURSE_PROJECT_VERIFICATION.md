# CS 886 submission verification

Updated November 21, 2026 against the course assignment screenshot.

| Course requirement | Submission |
|---|---|
| At most three pages of main text | Main text ends on page 3. References and appendices start on later pages. |
| Official ICML format | Official ICML 2026 template, supported preprint mode, with Times fonts loaded through T1 encoding. Template dimensions and spacing are unmodified. |
| Theory contribution or rigorous examination of a theoretical claim | The introduction frames the project as verification and stress-testing of replacement stability for the implemented set encoder. The generic triangle inequality and prior theoretical ingredients are attributed. |
| Technical detail sufficient to assess correctness | Main text states the assumptions, recovery bound and valid dispatch construction; full encoder proof and supporting details are in the appendix. |
| Reproducible code and materials | CS886_REPRODUCIBILITY.zip includes code, tests, configurations, saved results, checkpoints, report assets and a SHA-256 manifest. REPRODUCIBILITY.md gives generation and build commands. |
| Author information | Fei Zhang, University of Waterloo, f243zhan@uwaterloo.ca, as supplied in the chat. |
| Deadline shown in screenshot | November 26, 2026, 11:59 p.m. Toronto time. |

| Check | Result |
|---|---|
| Transformer stability proof | Compared with the frozen encoder implementation. Checked the bounded score and softmax Jacobian bounds, concatenation, attention residual, both positive-epsilon LayerNorm maps, feedforward residual, mean pooling, and sigmoid readout. |
| Stability and accuracy tradeoff | Verified the triangle inequality and the positive-part lower bound. The two cost vectors are in the coefficient box. Both demands yield unique interior dispatch with the stated prices, and each endpoint has two distinct outputs per generator. |
| Supporting transfer bound | Checked bounded loss, IID source tasks, the fixed hypothesis class, the sup-norm cover, and the Hoeffding and union bound steps. The empirical source and target risks use fresh contexts on the respective task sets. |
| Numerical results | Main and appendix table inputs and the main vector figure are generated directly from saved D–G CSVs by scripts/build_report_assets.py. Checked the cited G response ratios and follow-up numbers against their summary tables. |
| Citations | Checked the linked proceedings, author-hosted paper and preprints. The report distinguishes the Li et al. query-token theorem from the mean-pooled inverse estimator. |
| Existing unit tests | All 14 passed, including the exact KKT and rank checks for the one-record and multiple-record identifying pairs. |
| LaTeX and PDF | Compiled with portable Tectonic 0.16.9 using the official ICML template. The report has 3 main pages, 1 references page and 3 appendix pages. Rendered all pages with Poppler and reviewed layout, formulas, author details, references and figure labels. |
| Template integrity | SHA-256 comparison confirms the five supplied style files match the official ICML 2026 download. The PDF embeds the template's Times-family fonts. The final log has no overfull boxes, missing glyphs, font fallback, or unresolved citation/reference warnings. Ordinary underfull-box spacing warnings remain. |
| Archive integrity | The packaging script reopens the ZIP, checks CRCs, and verifies every included file against its source SHA-256 digest. |
| Experiment scope | Reused the saved experiment outputs. The earlier tests, code and results supply the report's evidence. |

## Presentation verification (November 25, 2026)

- Revised Beamer PDF (COURSE_PROJECT_SLIDES_V3.pdf): 27 pages, comprising
  19 main slides and 8 backup slides. The original source/PDF pair remains available.
- The main source has 19 speaker prompts with target times summing to
  1,080 seconds. Actual delivery time depends on rehearsal.
- Compiled without LaTeX overfull-box or unresolved-reference warnings.
  All pages were rendered with Poppler and visually reviewed. Extracted
  text was also checked to remain within each page's bounds.
- The standalone Beamer ZIP was CRC-checked, compared byte-for-byte with
  its inputs, extracted and successfully compiled separately.
- Presentation plots are generated from the saved D–G summary CSVs.
  The main proof separates the encoder upper bound from the generic
  recovery lower bound and states the nontriviality condition.
- A running affine example connects the research question to the two
  proofs, the valid dispatch pair and four empirical questions.
  The measured-response error constraint is generated from saved G data
  and checked against each of the three original checkpoints.

## Reproduction

Run from the project root with the existing Python environment and a LaTeX compiler.

~~~powershell
python -m unittest discover -s tests -v
python scripts/build_report_assets.py
tectonic COURSE_PROJECT_REPORT.tex
python scripts/package_submission.py
~~~

## Remaining limitations

- The weight-based stability constant is too loose to predict empirical differences at the tested lengths. The deterministic error floor applies to the specified cross-task context family.
- The study uses a two-generator synthetic model, mostly exact prices, and three seeds. The leverage interventions measure the predictor's response at its output.
- The number of source tasks did not have a consistent monotone effect under the matched budgets.
- The supplied screenshot covers the report format and project requirement. Its linked full instructions and the separate reproducibility upload question may contain additional administrative details.

The previous six-page main-text version and its verification record are retained under docs/archive/. The current root report follows the screenshot's three-page course limit.

[Official ICML 2026 template source](https://icml.cc/Conferences/2026/AuthorInstructions)

V3 retains the narrative and adds a formal finite-length/liminf corollary, explicit concept distinctions, and an architecture/certificate review in COURSE_PROJECT_V3_REVIEW.md.
