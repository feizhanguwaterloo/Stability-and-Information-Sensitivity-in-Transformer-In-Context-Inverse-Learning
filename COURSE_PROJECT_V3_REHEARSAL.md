# V3 rehearsal script

Target: 18 minutes, 19 main slides. Times are rehearsal targets.

## Slide 1: Title

Target time 25 seconds, cumulative 0:25.
The question for this talk is what happens when almost all the useful
information about one parameter sits in a single observation. We examine
a frozen Transformer, first through a simple example and two complementary
bounds, then through controlled inverse-learning experiments.

## Slide 2: When should one observation change the answer?

Target time 40 seconds, cumulative 1:05.
Small sensitivity can mean different things. If I replace an observation
with another from the same task, the correct parameters stay the same,
so an unchanged prediction can be exactly right. If a replacement
distinguishes two different tasks, the correct answer changes.
An estimator that stays almost unchanged may then be wrong.
Our question is whether the model makes that distinction.
To make the distinction concrete, consider a line with an unknown slope.

## Slide 3: A single record can determine the slope

Target time 65 seconds, cumulative 2:10.
Both contexts contain the same repeated zeros. Every line through
the origin is consistent with those zeros, so they tell us nothing
about the slope. The last record supplies the slope: one in A and
two in B. Each complete context has a unique correct answer.
Now imagine adding a thousand more copies of the zero record.
The answer still depends on that last observation just as strongly.
This is the example to keep in mind throughout the theory.
We will compare how much an architecture can respond with how much
accurate recovery requires it to respond.

## Slide 4: Possible response and required response

Target time 40 seconds, cumulative 2:50.
There are two separate questions. Theorem one gives a ceiling on
the response permitted by the architecture. Theorem two gives a
floor on the response required for accuracy. Delta is the same
prediction difference in both statements. D is the distance between
the correct parameters. Neither statement says that a smaller
response is always better. To study this tension in a structured setting, we consider an inverse
optimization problem where KKT conditions characterize which observations
identify the unknown task parameters. This provides a controlled testbed
for Transformer recovery. It does not introduce a separate electricity-market
research question.

## Slide 5: Learning cost parameters from dispatch observations

Target time 75 seconds, cumulative 4:05.
Inverse optimization is useful here because we can explicitly check
what each record tells us about the unknown task. Our objective is to
understand Transformer recovery, rather than demonstrate superiority over
classical inverse optimization. Each synthetic task has four unknown cost coefficients. For a given
demand we observe optimal generator outputs, a clearing price, and
whether each output is at a bound. The price equals marginal cost
for an interior generator. This produces exactly the affine equation
from the motivating example. Two different interior outputs identify
that generator's intercept and slope. Bound observations instead
give one-sided inequalities. A two-layer Transformer reads complete
market records, averages its final token states, and predicts four
cost coefficients through a bounded head. We train it on labeled
source tasks, then freeze it on new tasks. The question is how its
context response relates to the information in these equations.
We first establish an architectural upper bound.

## Slide 6: Theorem 1: a fixed encoder has a \(C/m\) response bound

Target time 50 seconds, cumulative 4:55.
The theorem compares one fixed model across lengths. It does not retrain
a new model as m increases. Under the stated assumptions, every
single-record replacement has a response bounded by C divided by m.
Because C stays fixed, the bound eventually tends to zero.
This is a worst-case upper bound, rather than a claim that every
measured response lies exactly on an inverse-length curve.
The actual implementation uses an affine 10-to-64 record embedding,
four heads, two post-residual normalization blocks per layer, ReLU
feedforward maps and zero dropout. The LayerNorm epsilon is positive.
There is no positional embedding or sequence-dependent mask.
Bounded records and finite maps bound all hidden rows at fixed depth,
independently of context length. The proof explains dilution and C.

## Slide 7: Proof 1: bounded attention controls perturbation spreading

Target time 75 seconds, cumulative 6:10.
We track average change over all token rows, not just one token.
At the embedding only one row changes, so the average is A over m.
Attention can spread that perturbation. The important restriction
is that fixed bounded scores give every attention probability an
upper bound proportional to one over m: the denominator receives
a positive contribution from every record. The changed record also
changes the attention weights themselves. Integrating the softmax
Jacobian controls that effect, and combining it with changed values
gives a fixed amplification factor for the average perturbation.
Averaging at the output alone would be insufficient if an arbitrary
encoder broadcast a large change to all rows. The bounded-attention
step is what controls that broadcast here.

## Slide 8: Proof 1: composing the layer bounds produces \(C\)

Target time 60 seconds, cumulative 7:10.
Each remaining operation has a finite Lipschitz factor. Positive
LayerNorm epsilon prevents an unbounded derivative near zero
variance. The residual and feedforward factors are finite as well.
Across a fixed number of layers these factors multiply. Mean pooling
cannot increase the average token difference, and the readout
contributes one final factor. Their product is C, which stays fixed
when the model stays fixed. This norm certificate is extremely
conservative: its logarithm is in the hundreds for our checkpoints.
Consequently the theorem gives an asymptotic restriction, not a
numerical prediction at the lengths we test. Next we ask what
accurate recovery requires, independently of this loose constant.

## Slide 9: Theorem 2: recovery requires enough response

Target time 80 seconds, cumulative 8:30.
Return to slopes one and two. Accurate predictions must move from
near one to near two. The theorem formalizes this requirement.
Go from the first true parameter to its prediction, then to the
second prediction, then to the second true parameter. These three
lengths must sum to at least the distance between the truths.
If the middle segment is short, the endpoint errors must supply
the remaining distance. At least one error is therefore at least
half of D minus Delta. This does not mean both endpoints must be
wrong: one could be exact and the other very inaccurate.
Equivalently, errors below epsilon on both sides require a response
of at least D minus two epsilon. With D equal to one and epsilon
equal to one tenth, that is a response of at least eight tenths.
This proof uses only the triangle inequality and is not
Transformer-specific.

## Slide 10: Corollary: a worst-case limitation of stable recovery

Target time 45 seconds, cumulative 9:15.
This corollary is the central theoretical result of the project.
For each length, use two individually identifying contexts with only
one differing record and fixed target separation. Apply the two bounds.
The finite-length result follows immediately, and taking the limit inferior
gives a floor of D over two on the larger endpoint error.
The slope example supplies intuition: more zero records add no identifying
information. This is an existence construction, not an average-case
error floor, and the numerical certificate is vacuous at our tested lengths.
We next construct the same situation using valid dispatch observations.

## Slide 11: The identifying pair exists in valid dispatch data

Target time 70 seconds, cumulative 10:25.
Each task gives the two generators identical cost curves, so
the optimal dispatch is equal sharing. At demand one, both
tasks give output one half per generator and price 2.1.
At demand 1.6, each output is 0.8, but the prices are 2.40
and 2.64. Both satisfy their task's interior KKT conditions.
Two different outputs let us solve both the intercept and
the slope. Each context is therefore uniquely identifying.
All costs also lie inside the specified task box.
Repeating the shared record gives precisely the structure of
the slope example: one price separates two identifiable tasks.
The four-dimensional target gap is square root of 1.6 for every length.
Additional copies of the background add no new parameter direction.
This verifies every premise of the corollary inside our bounded cost domain.
We will now measure what the trained models do on ordinary
contexts and then return to this exact stress case.

## Slide 12: Experiments connect information, response and recovery

Target time 50 seconds, cumulative 11:15.
The loose constant makes direct measurements essential.
The first experiment asks whether ordinary same-task replacement
stability can coexist with good recovery. The second changes
information quality while keeping context size fixed.
The third holds the focal edit fixed and changes the background.
The fourth returns to our identifying task pair and tests whether
the response is large enough to recover the true difference.
We use three seeds and hold test weights fixed. Comparisons include
the original and norm-controlled Transformers, DeepSets, ridge, and
a neural model supplied with sufficient statistics.
All predictions here are before any KKT correction, so these tests
concern the learned estimator itself.

## Slide 13: Ordinary replacements: stability and accuracy coexist

Target time 55 seconds, cumulative 12:10.
We first replace a record with another valid observation from
the same task. The correct parameter does not change.
The original Transformer's average sensitivity falls with context
length and its recovery error also improves. At length 64,
mean prediction change is about 0.00213 and relative recovery
error is 2.65 percent. It outperforms the other two learned
architectures under this schedule. This is consistent with
stability being useful for ordinary replacements. It does not
verify a worst-case asymptotic rate, because the plot averages
sampled changes. More importantly, these ordinary contexts
do not concentrate all identification in one record.
We next change the information structure.

## Slide 14: Nearly redundant contexts are harder to recover

Target time 55 seconds, cumulative 13:05.
Here both designs have sixteen observations and the same
task parameters. Each noiseless context uniquely determines
the costs, but one spreads its observations and the other
makes them nearly redundant. The small Gram eigenvalue
quantifies a weakly informed parameter direction.
The Transformer error increases from 2.90 to 13.65 percent.
Thus unique identification in exact arithmetic is insufficient
for good learned recovery. The designs match mean demand,
while their spreads necessarily differ. This does not by itself
isolate every feature-distribution effect. Conditioning characterizes information at the context level, while
leverage characterizes the importance of an individual observation.
Model sensitivity is a fourth concept: how much the learned output actually
changes under an edit. We now hold the edited observation and its
displacement fixed to examine these last two concepts separately.

## Slide 15: The same edit produces a larger response at high leverage

Target time 65 seconds, cumulative 14:10.
Leverage measures how much a record contributes to a regression
design relative to the rest. We keep the focal observation and
the price displacement fixed. The surrounding observations are
clustered or spread, making the focal record harder or easier to
replace informationally. The Transformer responds about fourteen
times more strongly in the high-leverage background. Ridge shows
a similar ratio, although its absolute response is larger.
DeepSets has almost no high-versus-low difference here.
These price perturbations leave the true task fixed and can be
viewed as measurement interventions. They establish output
responsiveness, not the correctness of a changed task estimate.
They also do not identify an internal attention mechanism.
The Transformer exhibits information-dependent sensitivity.
But is this sensitivity sufficient for accurate recovery when the
underlying task changes? We return to our valid identifying pair,
which changes the true parameter rather than just perturbing a measurement.

## Slide 16: On the identifying pair, the response remains incomplete

Target time 75 seconds, cumulative 15:25.
This is the exact economic-dispatch pair from slide eleven.
Both endpoints uniquely identify their different true parameters.
The reference at one hundred percent means the predicted
difference matches the true difference along the target direction.
It is not a prediction accuracy percentage, and it does not
by itself guarantee either endpoint is correct.
At length 64, Transformer directional response is 47.4 percent,
and its larger endpoint error averages 0.432. Ridge directional
response is 99.9 percent. Ridge demonstrates that these noiseless
contexts contain enough information to recover the task difference.
The Transformer responds substantially, but incompletely.
The experiment demonstrates finite-length under-response consistent with
the qualitative concern raised by the corollary, without establishing
its asymptotic rate or verifying a worst-case bound. To connect it quantitatively to theorem two,
we can instead use the response we actually measured.

## Slide 17: Theorem 2 also interprets the measured response

Target time 55 seconds, cumulative 16:20.
Theorem two can use an observed response, rather than the loose
architectural certificate. The true target separation is about
1.265. Mean prediction change across the three checkpoints is
about 0.603. The resulting lower bound on the mean larger
endpoint error is about 0.331. The observed value is 0.432.
These quantities use absolute parameter units, not relative
error or the directional-response percentage from the previous slide.
The inequality holds checkpoint by checkpoint and survives averaging.
This is a useful way to interpret the under-response. It is a
geometric requirement, not evidence that the worst-case C caused
the trained model's behavior.

## Slide 18: Three distinct claims and their scope

Target time 65 seconds, cumulative 17:25.
Keep these three claims separate. Theorem one proves an architectural rate.
The corollary turns it into an asymptotic limitation on a worst-case
identifying family. The experiments demonstrate finite-length under-response.
They are consistent with the qualitative concern, without establishing
the asymptotic rate or showing that our loose certificate causes the errors.
Li and colleagues supply established bounded-attention stability reasoning.
Our first theorem adapts it to the implemented encoder. The second theorem
is the generic triangle inequality. The central project result is the
corollary together with a valid structured construction and behavioral tests.
The constant, deliberately repeated contexts, small synthetic setting and
unresolved mechanisms limit the conclusions. Fixed-budget task-count gains
also remain inconclusive, as shown in backup D.

## Slide 19: Accurate recovery requires sufficient response

Target time 35 seconds, cumulative 18:00.
Return to the line example. Many repeated records do not replace
the one observation that determines the slope. The two theorems
compare an architectural response ceiling with the response
required to recover different truths. The experiments show
information-dependent responsiveness, together with incomplete
recovery on the identifying pair. The question is therefore
whether the model responds enough to the information needed
for the task.
