# Speaker notes: FINAL_V2

20 main slides target 18 minutes, excluding questions. Proof steps 1, 2, 3 target 60, 90, 60 seconds. Eleven technical backup pages follow.

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

## Slide 7: Proof step 1: one changed row and bounded attention

Target time 60 seconds, cumulative 5:55.
We track the average token difference because attention can move the
initial change into many rows. One row may change a lot, but immediately
after embedding only that row differs. Its contribution to the average
is at most A divided by m. Queries, keys and values are fixed affine
maps of bounded states. Their norms, and hence the scores, have bounds
independent of context length. In softmax the largest possible numerator
is exp B. Every denominator term is at least exp minus B, and there are
m of them. This gives exp two B divided by m. That limits each record's
attention weight. We must still check both changed values and changed
weights, because replacing a record can affect both.

## Slide 8: Proof step 2: control value changes and weight changes

Target time 90 seconds, cumulative 7:25.
The attention output can change in two ways. First hold the old
weights fixed and change the values. The affine value map cancels
its bias in the difference. Multiplying each row difference by an
attention weight bounded by exp two B over m produces a constant
times the average input change. The second term keeps the new
values fixed and changes the attention weights. Bounded values
turn this into a probability-vector difference in the one norm.
Both the query of row i and the keys of other rows can move.
Their score changes therefore have a local e i term and a key e j
term. The displayed Jacobian column sum bounds how changing one
score changes the full probability vector. Integrating along the
score segment gives a local query term plus the average key change.
Combining the two components gives alpha times e i plus beta times
the average. Averaging over i converts the local term into another
average term. This is the crucial step: attention can spread change,
but cannot amplify the average by a factor growing with m.
Fixed head concatenation and output projection preserve this form.
We can now compose the remaining encoder operations.

## Slide 9: Proof step 3: compose the blocks, pooling and head

Target time 60 seconds, cumulative 8:25.
The attention result now supplies one finite factor. Adding its
residual contributes one plus that factor. The feedforward block
uses ReLU, which is one-Lipschitz, so its residual factor is one plus
the product of the two weight norms. Each positive-epsilon LayerNorm
also has a finite factor; both normalizations are included in each
complete layer bound. Repeating the recurrence multiplies these
factors but leaves the initial one over m intact. Mean pooling is
bounded by the average row difference. Finally the prediction head
contributes its Lipschitz constant. Everything in the numerator is
C and stays independent of context length. This proves an eventual
rate. Its norm-based certificate is enormous and vacuous at measured
lengths. We next ask how much response accuracy requires, independently
of this loose numerical ceiling.

## Slide 10: Theorem 2: recovery requires enough response

Target time 80 seconds, cumulative 9:45.
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

## Slide 11: Corollary: a worst-case limitation of stable recovery

Target time 45 seconds, cumulative 10:30.
This corollary is the central theoretical result of the project.
For each length, use two individually identifying contexts with only
one differing record and fixed target separation. Apply the two bounds.
The finite-length result follows immediately, and taking the limit inferior
gives a floor of D over two on the larger endpoint error.
The slope example supplies intuition: more zero records add no identifying
information. This is an existence construction, not an average-case
error floor, and the numerical certificate is vacuous at our tested lengths.
We next construct the same situation using valid dispatch observations.

## Slide 12: The identifying pair exists in valid dispatch data

Target time 70 seconds, cumulative 11:40.
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

## Slide 13: Experiments connect information, response and recovery

Target time 40 seconds, cumulative 12:20.
We test four questions in order: ordinary replacement, context conditioning,
matched focal edits, and recovery on the valid identifying pair.
The models use three seeds and task-disjoint splits. D and G train on
1,536 source tasks, with 96 validation and 128 unseen test tasks per seed.
The baselines include the norm-controlled Transformer, DeepSets, ridge and
the sufficient-statistics MLP. Every prediction is before KKT correction.
We first check whether ordinary stability can coexist with accuracy.

## Slide 14: Ordinary replacements: stability and accuracy coexist

Target time 45 seconds, cumulative 13:05.
The replacement here comes from the same task, so the correct parameters
stay fixed. Both average prediction change and recovery error decrease
with context length. At length 64, the original Transformer changes by
about 0.00213 and has 2.65 percent relative error. It does best among
these three models under the shared schedule. These are sampled averages,
not a certificate of worst-case inverse-length scaling. Stable, accurate
behavior on ordinary contexts leaves open what happens when information
is nearly redundant.

## Slide 15: Nearly redundant contexts are harder to recover

Target time 45 seconds, cumulative 13:50.
Both designs have sixteen records and matched mean demand, but different
output spreads. Both uniquely identify the noiseless costs. Yet the
near-redundant design has a tiny minimum Gram eigenvalue and much larger
Transformer error: 13.65 rather than 2.90 percent. Identifiability asks
whether an answer is unique. Conditioning asks how well the context
constrains each direction. Next we move from whole-context conditioning
to an individual record's leverage, and measure model sensitivity to
a fixed edit.

## Slide 16: The same edit produces a larger response at high leverage

Target time 50 seconds, cumulative 14:40.
We keep the focal record and price displacement fixed and change the
background. High leverage means the focal record supplies information
the rest provides weakly. Transformer response is 14.1 times larger
there; ridge's ratio is 13.1. Leverage describes the design, while
sensitivity describes the estimator's actual output change. The artificial
price edit can violate the original noiseless KKT equations even though
the true task stays fixed. It demonstrates information-dependent response.
Is that response sufficient when the task really changes? We now return
to the valid identifying pair.

## Slide 17: On the identifying pair, the response remains incomplete

Target time 60 seconds, cumulative 15:40.
This is the valid dispatch pair from slide twelve. Each endpoint uniquely
identifies its different task. A directional response of one means the
predicted difference recovers the correct change along the target direction;
it does not alone guarantee endpoint accuracy. At length 64 the Transformer
recovers 47.4 percent of that difference, while ridge recovers 99.9 percent.
The larger Transformer endpoint error averages 0.432. This is finite-length
under-response, consistent with the qualitative concern of the corollary.
It neither establishes an asymptotic rate nor follows numerically from
the loose C certificate. We can interpret it instead using the
directional recovery inequality on the next slide.

## Slide 18: Directional response bounds recovery error

Target time 55 seconds, cumulative 16:35.
We can now connect the directional response directly to recovery.
Project the difference between the target change and predicted change
onto the unit target direction. Its magnitude is D times absolute one
minus r. The two endpoint errors must cover that magnitude, so their
maximum is at least half of it. This holds for every checkpoint.
We average the maximum within each checkpoint, not the maximum of
the two mean endpoint errors. The saved outputs give a directional
lower bound of 0.332904, a Euclidean bound of 0.331170, and an observed
mean larger endpoint error of 0.431601. The absolute value must remain
inside the average in general. All three observed r values are below
one, so here it also equals D over two times one minus mean r.
This is a geometric constraint, without the loose C certificate.

## Slide 19: Three distinct claims and their scope

Target time 50 seconds, cumulative 17:25.
Theorem one proves an architecture-specific asymptotic rate. The corollary
turns it into a worst-case recovery restriction for an identifying family.
The experiments separately measure finite-length under-response. They do
not verify the worst-case rate or derive the errors from our loose
certificate. The project connects an adaptation of established stability
reasoning, a valid identifying construction and controlled behavioral
tests. The triangle inequality is generic. Repeated-record stress cases,
three seeds, two generators and unresolved internal mechanisms limit the
claims. Fixed-compute task-count gains also remain inconclusive.

## Slide 20: Information-dependent sensitivity and recovery

Target time 35 seconds, cumulative 18:00.
Return to the line example. Many repeated records do not replace
the one observation that determines the slope. The two theorems
compare an architectural response ceiling with the response
required to recover different truths. The experiments show
information-dependent responsiveness, together with incomplete
recovery on the identifying pair. Information-dependent sensitivity does not guarantee sufficient recovery.
That is the final takeaway. The asymptotic corollary and finite-length
observations support distinct parts of this statement.
