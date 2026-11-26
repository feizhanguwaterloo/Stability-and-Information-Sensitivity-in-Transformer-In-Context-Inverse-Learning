# V3 targeted review

Reviewed November 25, 2026. V2 source and PDF remain available.
The supplied request ends at the heading "Experiment 2: Context conditioning";
all received requirements were implemented without inferring missing text.

## Mathematical and implementation consistency

- src/model.py: fixed affine 10-to-64 record embedding, two encoder layers,
  four heads, 128-wide feedforward maps, zero dropout, full unmasked attention,
  no positional embeddings, mean pooling and affine/sigmoid/box head.
- Instantiating SetTransformer confirms post-residual LayerNorm,
  epsilon 1e-5 in both normalization maps, ReLU activation and zero dropout.
- src/data.py supplies four bounded continuous features and two one-hot
  statuses: record norm is bounded by sqrt(6). Fixed affine biases are included
  in the hidden-state and attention-logit bounds.
- The report appendix and docs/theory.md track average token perturbation.
  Bounded logits yield p_ij <= exp(2B)/m. The softmax Jacobian controls changed
  weights, and bounded value maps control changed values. Residuals and
  rowwise Lipschitz maps yield the full-layer recurrence. Mean pooling
  completes the argument after this propagation bound.
- ReLU is 1-Lipschitz. Each LayerNorm has factor at most max|gain|/sqrt(epsilon).
  The sigmoid/box head factor is at most ||diag(span)|| ||W_head|| / 4.
- Theorem 2 is estimator-independent triangle geometry.
- Corollary: a fixed-model constant C, one-record neighboring contexts at
  every length m>=2, individually identifying endpoints and a fixed D>0
  imply max endpoint error >= (D-C/m)_+/2, and liminf error >= D/2.
  No monotonicity of the actual errors is assumed.
- src/leverage.py::theorem_pair supplies fixed tasks (1.6,1) and (1.2,1.8)
  for each generator, common demand 1 and identifying demand 1.6.
  It verifies applicability within the bounded task domain.

## Saved evidence and scope

results/leverage_stability/norm_bounds.csv, original Transformer:
seed 886: log10(C)=708.8952567103589;
seed 1729: 661.4483297603454;
seed 2027: 616.4704204288647.
The slide range 616.5--708.9 is rounded to one decimal place.
src/leverage.py::transformer_bound includes affine bias bounds, multihead
concatenation/output projection, both residuals and LayerNorm maps,
feedforward factors and the prediction head.

The D/G configurations, leverage and follow-up scripts, and saved summaries
retain their original protocols. No training or saved experimental output
was changed. The four main empirical questions retain their order.
The report and full proof remain the scientific sources.

Distinguish: (1) an asymptotic architecture-specific replacement rate,
(2) a worst-case asymptotic recovery limitation, and (3) finite-length
identifying under-response. The numerical certificate is vacuous at measured
lengths. Empirical curves establish neither a worst-case rate nor a causal
explanation from that certificate.

## Presentation delivery

19 main slides target 1,080 seconds; 8 backup slides.
Transitions 4--5, 14--15 and 15--16 are explicit in slides and speaking prompts.
The design, four empirical questions, curves, values and primary narrative
remain those of V2. The corollary replaces the existing combined-result slide.
The English rehearsal file is extracted from the nineteen embedded notes.
