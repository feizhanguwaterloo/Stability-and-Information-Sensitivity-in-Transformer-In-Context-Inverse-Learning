# Stability and Information Sensitivity in Transformer In-Context Inverse Learning

**CS 886 course project**  
**Research question** · How does a frozen Transformer balance stability to single-record replacement with responsiveness to observations that identify an otherwise missing parameter direction?

## Setup

A two-generator dispatch optimization problem produces records of demand, optimal outputs, market price, and generator bound statuses. A two-layer, mean-pooled Transformer uses a set of records to estimate four latent cost coefficients. This supervised amortized inverse estimator is related to in-context learning because it adapts its prediction to new task records with frozen weights. Its output is a task parameter vector. Li et al. (ICML 2023) study the response to a designated query. [Reference paper](https://proceedings.mlr.press/v202/li23l.html)

## Theory

1. **Architecture upper bound.** For bounded inputs and fixed finite weights and depth, one record replacement changes the output by at most \(C/m\). The proof controls mean row perturbations through attention, residuals, LayerNorm, and feedforward layers, then applies mean pooling and the readout.
2. **Recovery lower bound.** If two neighboring contexts identify targets a distance \(D\) apart and both errors are at most \(\epsilon\), their predictions must differ by at least \(D-2\epsilon\). This follows from the triangle inequality. The two-generator model gives a valid example with \(D=\sqrt{1.6}\): repeat a shared market record and append one outcome on which only the clearing prices differ. Both contexts uniquely identify their costs.

Together these bounds give a worst-case recovery error floor for this context family as \(m\) grows. The norm-based value of \(C\) is too large to predict the effect at tested lengths.

## Results

All four experiments used three independent seeds.

| Experiment | Main result |
|---|---|
| Architecture (D) | At 64 records, mean same-task sensitivity was 0.00213 for the original Transformer, 0.00510 for the constrained Transformer, and 0.00760 for DeepSets. Relative recovery errors were 2.65%, 3.95%, and 10.28%. |
| Task transfer (E) | At 32 evaluation records, equal-pass error fell from 13.48% to 2.72% across 128 to 1,536 source tasks. At 448 updates it stayed within 3.20–3.56% with a nonmonotone trend. |
| Conditioning (F) | Near-redundant contexts uniquely identified the parameters, yet Transformer error rose from 2.90% to 13.65% as the mean minimum Gram eigenvalue fell from 0.03257 to \(3.72\times10^{-6}\). |
| Leverage (G) | Matching the focal price displacement, the high-leverage response was 14.1 times the low-leverage response. On the valid single-record cross-task pair at length 64, the Transformer recovered 47.4% of the true difference along its direction, and ridge recovered 99.9%. |

## Contribution and scope

The project connects an architecture-specific upper bound to a general recovery lower bound through a valid inverse optimization example. Its controlled experiments show that the Transformer's response depends on context leverage and can still fall short of accurate recovery. The worst-case construction uses repeated demands, and the experiments use a small dispatch model with primarily exact prices.

[Final report](COURSE_PROJECT_REPORT.pdf) · [Presentation outline](COURSE_PROJECT_PRESENTATION_OUTLINE.md)
