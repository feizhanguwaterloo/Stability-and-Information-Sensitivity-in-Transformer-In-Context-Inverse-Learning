# CS 886 presentation: 18-minute speaking guide

**Stability and Information Sensitivity in Transformer In-Context Inverse Learning**  
**Fei Zhang, University of Waterloo**

The English Beamer deck has 17 main slides, including the title, followed
by 5 backup slides. The planned main talk is 18 minutes, excluding questions.
These are rehearsal targets, not a measured recording. Speaker prompts and
per-slide times are embedded in the LaTeX source using `\note{...}`;
the audience PDF hides those notes.

## Main talk

| Time | Slide | Speaking goal |
|---|---|---|
| 0:00-0:30 | 1. Title | Ask how much one observation should change a learned estimate. |
| 0:30-1:35 | 2. Motivation | Repeated affine observations at zero identify an intercept; one nonzero observation identifies a slope. A uniformly small response can conflict with recovery. |
| 1:35-2:30 | 3. Research contribution | Position the project against Li et al.: an implemented-architecture audit, a valid inverse-problem construction, and controlled tests. Separate established proof tools from our application. |
| 2:30-3:35 | 4. Economic dispatch | Explain the quadratic objective and KKT conditions. Two distinct interior outputs identify each cost curve. |
| 3:35-4:35 | 5. Transformer methodology | Explain record encoding, two Transformer layers, mean pooling and the bounded head. Context adaptation uses fixed weights. The parameter readout differs from the reference paper's query readout. |
| 4:35-5:30 | 6. Stability statement | State the fixed-weight, fixed-depth, bounded-input and positive-LayerNorm-epsilon assumptions. The uniform guarantee has a weight-dependent constant. |
| 5:30-7:00 | 7. Attention proof | Track average row difference. Bounded scores imply probabilities at most a constant divided by length. The softmax Jacobian controls changed weights; averaging controls changed queries, keys and values. |
| 7:00-8:15 | 8. Complete the proof | Explain residual and LayerNorm factors, mean pooling and the sigmoid head. Report the huge norm-based constants: the bound does not explain finite-length measurements quantitatively. |
| 8:15-9:40 | 9. Recovery proof | Prove the triangle-inequality lower bound. Combine it with the architecture bound. The error lower bound is positive only beyond the constant divided by target separation. |
| 9:40-11:00 | 10. Valid dispatch pair | Work through the common record at demand 1 and the distinguishing price at demand 1.6. Each endpoint identifies its costs. State the asymptotic worst-case consequence and repeated-demand assumption. |
| 11:00-11:50 | 11. Evaluation | Define unseen-task splits, three seeds, relative error and absolute prediction change. Distinguish same-task replacements from cross-task pairs. All predictions precede KKT correction. |
| 11:50-12:55 | 12. Architecture results (D) | Discuss sensitivity and recovery together. At length 64, errors are 2.65%, 3.95% and 10.28% for Transformer, norm cap and DeepSets under the shared schedule. |
| 12:55-13:55 | 13. Information results (F) | Separate unique identification from conditioning. Nearly redundant contexts have singleton admissible sets but worse learned recovery. Boundary-heavy contexts show partial identification. |
| 13:55-14:55 | 14. Matched leverage (G) | Hold the focal record and edit fixed; change the background. The high/low response ratios are 14.1 for Transformer and 13.1 for ridge. This is behavioral evidence, not an internal-mechanism proof. |
| 14:55-15:55 | 15. Identifying response (G) | Define directional response. At length 64, Transformer recovers 47.4% of the target difference; ridge recovers 99.9%. Its larger endpoint error averages 0.432 in parameter units. |
| 15:55-17:00 | 16. Transfer controls (E) | Equal passes change both task count and updates. Equal-step and fixed-context controls give no monotone task-count benefit. Keep the fixed-compute conclusion inconclusive. |
| 17:00-18:00 | 17. Conclusion | Connect the stability guarantee, required response and measured partial recovery. State the loose-constant, repetition, seed-count and two-generator limitations. |

If rehearsal exceeds 18 minutes, shorten the architecture and task-scaling
commentary first. Preserve the common-record example, recovery proof and
matched-intervention interpretation: they carry the central argument.

## Backup slides

- A: LayerNorm derivative and its finite Lipschitz constant.
- B: Multiple identifying records, the telescoping bound and measured path steps.
- C: Independent Gaussian noise on every observed price.
- D: A standard task-level covering-number generalization bound.
- E: Selected references and the reproducible project repository.

## Questions to prepare for

- **Is the recovery bound Transformer-specific?** No. The triangle inequality
  applies to any estimator. The architecture-specific part is the uniform
  replacement upper bound for the implemented mean-pooled encoder.
- **What is our contribution?** Checking that architecture's assumptions and
  proof, constructing valid identifying dispatch pairs, and testing the
  relationship between information, response and recovery. Do not claim a
  new general stability-accuracy theorem.
- **Why does the bound not predict the measurements?** The norm-based constant
  is extremely loose; the output-box diameter is tighter at tested lengths.
  Sampled means also differ from worst-case bounds.
- **Why can ridge recover the noiseless pair?** Two distinct interior outputs
  per generator identify its coefficients; selected regularization is small.
- **Does leverage prove what attention learned?** It shows output behavior
  under matched focal edits. It does not identify an internal mechanism,
  and background context still changes.
- **Does more task diversity improve fixed-compute transfer?** Our controls
  do not establish a consistent benefit.

## Build and edit

From the project root:

~~~powershell
tectonic COURSE_PROJECT_SLIDES.tex
~~~

Alternatively, use `latexmk -pdf COURSE_PROJECT_SLIDES.tex`. Standard Beamer,
Latin Modern and TikZ packages are required. The four vector figures in
`results/presentation/` are included in the standalone source ZIP. Upload
that ZIP to Overleaf and select `COURSE_PROJECT_SLIDES.tex` as the main file.

Regenerate figures from the saved CSV results with:

~~~powershell
python scripts/build_slide_assets.py
~~~

This uses NumPy and Matplotlib from the project's dependencies and does not
retrain models. Figures show standard deviation across three seed means.
Change `\setbeameroption{hide notes}` to `\setbeameroption{show notes}` or
`\setbeameroption{show only notes}` to compile a notes PDF.

[Slides](COURSE_PROJECT_SLIDES.pdf) ·
[LaTeX source](COURSE_PROJECT_SLIDES.tex) ·
[Standalone source bundle](CS886_BEAMER_SOURCE.zip) ·
[Final report](COURSE_PROJECT_REPORT.pdf)
