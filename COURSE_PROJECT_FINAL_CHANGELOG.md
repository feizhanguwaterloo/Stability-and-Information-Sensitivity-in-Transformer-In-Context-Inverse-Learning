# FINAL targeted changelog

V3 is preserved. FINAL retains 19 main slides and the approved narrative,
with 11 backup pages. Main speaking targets total 18 minutes.

## Revisions and verified results

- Slide 17 links slide 16's directional metric directly to recovery.
  With delta = theta_B - theta_A, D = ||delta|| > 0, u = delta/D,
  v_j = f_j(S_B)-f_j(S_A), and r_j = <v_j,delta>/D^2:
  D|1-r_j| = |u^T(delta-v_j)| <= e_Aj+e_Bj <= 2 max(e_Aj,e_Bj).
  Backup J gives the proof and precise checkpoint averaging.
- From the three saved original Transformer outputs at m=64:
  mean directional lower bound = **0.332904141194**;
  mean Euclidean lower bound = **0.331170348700**;
  observed mean maximum endpoint error = **0.431601456141**.
  The first average uses mean_j |1-r_j|, not |1-mean_j r_j|.
  They coincide here because all three r_j are below one.
  All bounds hold per checkpoint. JSON records per-checkpoint values;
  the verifier also reconstructs both absolute endpoint errors.
- Slide 14 restores equal length (16), matched mean demand and different
  output spreads. Slide 15 explicitly marks artificial price edits as
  potentially violating noiseless KKT. Slide 16 remains a distinct
  valid different-task pair. Backup I defines implementation-consistent
  leverage: max_i u_im^T H_i^dagger u_im / m on all-interior designs,
  including the numerical pseudoinverse cutoff.
- Backup E retains finite measurements and adds (D-kC/m)_+/2.
  Its continuation states the fixed-model, uniform-path, valid-identifying-
  family and fixed-positive-gap assumptions. If k_m=o(m), the conditional
  liminf lower bound is D/2. Finite experiments do not establish this limit.
- Slide 19 uses the requested takeaway verbatim.

## Consistency audit and limitations

ReLU has Lipschitz constant 1; the sigmoid constant is 1/4.
Positive-epsilon LayerNorm, finite affine maps/biases, bounded states,
full unmasked attention and fixed depth/weights support the existing proof.
Mean pooling completes rather than replaces the attention argument.
The theorem, corollary and empirical claims remain explicitly distinct.
Existing references, figure labels, units and saved numerical claims
retain V3's scope; no new literature or training claims were introduced.

The directional bound is a geometric necessity, not a causal explanation
of learned under-response. Only three checkpoints and a synthetic two-unit
setting are studied. The weight-norm certificate remains vacuous at measured
lengths; neither sampled sensitivity nor finite recovery errors establishes
a worst-case asymptotic rate.

All 30 PDF pages were rendered and reviewed. The final LaTeX log has no
overfull boxes or missing-glyph warnings. The source ZIP was checked and
compiled after extraction. V3 and every pre-existing file in results/,
src/ and configs/ were hash-checked as unchanged.

## Reproduce the lightweight checks

Run from the project root with standard Python (no scientific packages needed):

~~~text
python scripts/verify_final_presentation.py
tectonic COURSE_PROJECT_SLIDES_FINAL.tex
python scripts/package_final_slides.py
~~~

The FINAL source ZIP includes the generated numerical macros and all four
vector figures. Select COURSE_PROJECT_SLIDES_FINAL.tex as the main file
when uploading to Overleaf. Proofs and updated timed prompts are embedded
in the LaTeX source; the rehearsal Markdown provides the same main notes.
