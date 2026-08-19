# E20 repaired identical-evidence order experiment

**Status: originally reviewed and design-frozen on 2026-08-03 at 09:37:03
+0800, then prospectively amended and re-frozen on 2026-08-03 at 10:50:00
+0800; no registered E20 pilot or full-panel artifact existed at either
freeze. Implementation and outcome construction remain separately gated
below.**

This memo specifies the final Forkworld order experiment. It is a repair of the
failed E18 manipulation, not a continuation of the E18 outcome analysis. No
E20 outcome exists or has been inspected. The path and SHA-256 of this reviewed
memo must be recorded in the prospective record. The implementation, gate,
launcher, and strict analyzer must then be frozen and fingerprinted before any
pilot model is constructed.

## Scientific question and scope

The experiment asks whether **identical cumulative labeled evidence can lead to
different learned decision rules solely because three goal-specific evidence
components arrive in different orders**. The treatment is the permutation of
three complete data blocks. Every schedule receives exactly the same weighted
raw inputs, labels, number of presentations, optimizer hyperparameters, and
common washout; only the order of the three components differs.

This is deliberately not another demonstration that models pass through
`P -> Q -> Y` as successively harder rules become learnable. There is no shared
proxy prefix and no claim about a universal phase transition. Measurements
during the three components validate and describe the manipulation. The
scientific estimand begins only after all six schedules have received the same
cumulative evidence and a fresh optimizer has been installed for a common,
goal-concordant washout. A persistent difference at that point is path
dependence under this exact component construction.

E18 cannot answer this question. Its isolated diagnostic blocks admitted the
one-bit rule `-P` with perfect accuracy, so the pilot did not selectively train
the intended `Q` and `Y` goals. E20 removes that shortcut by giving each goal a
separate exhaustive component on which that goal is exact, either other named
goal is at chance, and no raw-input rule can outperform the named goals after
the three components are pooled.

The order treatment reorders whole components, including their component-
specific raw-input frequencies as well as their labels. It therefore estimates
path dependence with respect to the order of these evidence distributions; it
does not separately identify label order from covariate-distribution order.

## Raw interface and candidate rules

The existing `SemanticBatch` interface exposes the six varying signs plus two
constant presence channels, in this exact order:

```text
[P, P_present, R_1, R_2, R_3, Q_present, Q_1, Q_2].
```

`P_present` and `Q_present` are identically `+1` in every row of every
component, pilot arm, schedule, washout batch, and evaluation panel. They add no
varying raw information and cannot identify a block or schedule. There is no
state, padding, row identifier, stratum-copy marker, component marker, schedule
marker, stage address, or view marker in the model input. Metadata used to
balance batches must never enter `x`. The three named rules are

\[
P(x)=P,\qquad Q(x)=Q_1Q_2,\qquad
Y(x)=R_1R_2R_3.
\]

All $2^6=64$ codewords of the six varying signs are enumerated while the two
presence constants remain fixed. They induce all eight candidate tuples
$(P,Q,Y)\in\{-1,+1\}^3$, with exactly eight raw codewords per tuple. All Bayes
claims below are conditional on the two constant presence values and therefore
remain claims about the complete eight-channel model input.
We also pre-register the candidate-majority rule

\[
M(x)=\operatorname{sign}\{P(x)+Q(x)+Y(x)\},
\]

which is well-defined because there are three signs.

The selector is the existing CPU-deterministic clean-SFT `GoalMLP`: width 64,
depth two, ReLU activation, no residual connection, biases enabled on every
affine layer, and one scalar logit. It has no nuisance or auxiliary head. Every
block uses binary logistic loss with zero label smoothing and a newly
constructed AdamW optimizer with learning rate `.003`, weight decay `0`, and
the repository defaults `betas=(.9,.999)` and `eps=1e-8`. The global gradient
norm is clipped at `1.0`. Batch size is 288. Model weights continue between
blocks; optimizer state never does. No scheduler, early stopping, dropout,
data augmentation, or random-subspace restriction is used. The official
launcher forces the relevant OpenMP and BLAS thread environments to one for
artifact construction and exact CPU replay.

Only the registered experimental seed initializes model weights. Data are
constructed with fixed seed `171000001`, all atomic minibatch streams with
fixed seed `171000002`, and the deterministic phase contexts use fixed seeds
`171000101`, `171000102`, `171000103`, and `171000104` for components
$P,Q,Y$ and washout, respectively. These values are shared across pilot and
full-panel seeds. Thus an experimental seed does not also act as an implicit
data-order, minibatch-order, or phase-randomness treatment.

## The three weighted evidence components

For $G\in\{P,Q,Y\}$, component $S_G$ labels every raw codeword by $G(x)$.
Its multiplicity depends only on the candidate tuple. Define

\[
w_G(P,Q,Y)=
\begin{cases}
2,&P=Q=Y,\\
2,&G\text{ is the unique minority among }(P,Q,Y),\\
1,&G\text{ agrees with exactly one other candidate.}
\end{cases}
\]

For a fixed $G$, the 16 unanimous raw codewords contribute $16\times2=32$
weighted strata, the 16 raw codewords on which $G$ is the minority contribute
another 32, and the remaining 32 codewords contribute one each. Thus every
component has exactly

\[
32+32+32=96
\]

weighted raw strata. A weight-two raw codeword is represented by two metadata-
distinct copies with identical model inputs and label; the copy identity is not
an input feature.

Each component dataset contains 96 replicas of each weighted stratum, for

\[
n_{S_G}=96\times96=9{,}216.
\]

Every optimizer minibatch contains exactly three replicas of every weighted
stratum. It therefore has $96\times3=288$ rows, 144 of each label sign. For a
weight-two raw codeword this is six occurrences per minibatch; for a weight-one
codeword it is three. Thirty-two minibatches exhaust the 96 replicas once, so
one complete dataset presentation is 32 updates. Every component receives eight
presentations, or exactly 256 optimizer updates and 73,728 sample
presentations. Within-component minibatch streams are fixed before the pilot
with the seeds registered above and reused bit for bit wherever that component
occurs, including across experimental seeds.

### Exact pooled-optimality proof

It is useful to count one unit per weighted stratum; the 96 within-stratum
replicas and eight presentations multiply every count by the same constant.
Pooling $S_P,S_Q,S_Y$ gives 288 weighted units.

For an unanimous raw codeword, each component has weight two and all three
labels equal the common sign. Its pooled conditional label multiset therefore
contains six copies of that sign. For a non-unanimous raw codeword, the minority
component contributes two copies of the minority sign, while the two majority
components contribute one copy apiece of the majority sign. Its pooled label
multiset is exactly a two-versus-two tie.

Consequently, for any deterministic or randomized raw-input classifier, the
empirical conditional Bayes upper bound is

\[
\begin{aligned}
A^*_{\mathrm{raw}}
&=\frac{\sum_x\max\{N_x(-1),N_x(+1)\}}
        {\sum_x\{N_x(-1)+N_x(+1)\}}\\
&=\frac{16\cdot6+48\cdot2}{16\cdot6+48\cdot4}
=\frac{192}{288}=\frac23.
\end{aligned}
\]

This calculation is an upper bound over **every** function of the complete
eight-channel input. Because both presence channels are fixed, such a function
reduces on support to a function of the six varying signs, including any
arbitrary 64-bit truth table; the calculation does not merely compare a short
list of named rules. An exhaustive audit will group the realized integer
training counts by all 64 varying-sign codewords, condition on the constant
presence values, and recompute this bound exactly. Any value other than $2/3$,
or any input feature that distinguishes component membership, is a fatal design
failure.

Each named candidate also has pooled accuracy $2/3$. It is correct on all 96
unanimous units. On each non-unanimous raw codeword, a named candidate agrees
with exactly two of the four pooled labels: if it is the minority, those are its
own two weighted copies; if it is in the majority, those are the one-copy
components for the two majority goals. It therefore adds 96 correct units from
the non-unanimous inputs, giving $(96+96)/288=2/3$. Majority $M$ likewise
matches the two majority labels on every non-unanimous codeword and is correct
on every unanimous codeword, so it also scores $2/3$.

Within an isolated component $S_G$, $G$ has accuracy one, either other named
candidate has accuracy $1/2$, and $M$ has accuracy $2/3$. These identities,
the 48/48 weighted-stratum label balance, and the pooled $2/3$ values are all
integer audits, not Monte Carlo expectations.

The pooled evidence deliberately leaves many Bayes-optimal solutions. Every
non-unanimous raw codeword is tied, so the data identify no unique action there.
The scientific question is whether component order selects systematically among
these equally optimal rules.

## Six order schedules and concordant washout

The full panel contains all six permutations:

```text
P-Q-Y   P-Y-Q   Q-P-Y   Q-Y-P   Y-P-Q   Y-Q-P
```

For each seed, all schedules begin from a bit-identical model initialization.
Different experimental seeds change that initialization and nothing else in
the data, atomic-minibatch, or phase-RNG construction.
They receive their first component for 256 updates with a fresh AdamW optimizer,
discard that optimizer, receive the second component with another fresh AdamW,
discard it, and do the same for the third component. Every schedule has then
seen precisely the same 221,184 labeled sample presentations. Model weights are
never reset or copied between schedules.

At update 768, all schedules discard the third optimizer and install another
audited empty AdamW for a common 256-update washout. Washout contains only the
16 unanimous raw codewords, labeled by their common $P=Q=Y$ sign. To preserve
the same batching mechanics, each unanimous codeword has six metadata-only
copies, giving 96 washout strata; the dataset again has 96 replicas per stratum,
9,216 rows, three instances of each stratum per 288-row minibatch, 32 batches
per presentation, and eight presentations. Every washout minibatch has 144
labels of each sign. The washout stream is bit-identical across all schedules
within seed.

The washout contains no state on which the named goals disagree and therefore
adds no evidence favoring one of them. For completeness, when washout is
included in the empirical pool, each of $P,Q,Y,M$ and the unrestricted
raw-input Bayes rule has accuracy $3/4$: washout adds 96 unanimously correct
units to the diagnostic pool's 192 correct of 288. This second exact identity
will also be audited; the registered $2/3$ claim always refers to the three
order-manipulated components before common washout.

Total full-schedule training is 1,024 updates and 294,912 sample presentations.

## Fresh seeds and independent unit

Before choosing seeds, all directories matching `artifacts*` were scanned for
`summary.json`. The scan found 57,705 summaries, 67 distinct realized seeds,
and a maximum realized seed of 557. The realized set was

```text
0, 11, 23, 37, 41, 53, 67, 71, 83, 97, 101, 103, 107, 109, 113,
127, 131, 137, 139, 149, 151, 157, 163, 167, 173, 179, 181, 191,
193, 197, 199, 211, 223, 227, 229, 233, 239, 241, 251, 257, 263,
389, 397, 401, 409, 419, 421, 431, 433, 439, 443, 449, 457, 461,
463, 467, 479, 487, 491, 499, 503, 509, 521, 523, 541, 547, 557.
```

The proposed pilot-only seeds are

```text
563, 569, 571
```

and the proposed full-panel seeds are

```text
577, 587, 593, 599, 601, 607, 613, 617, 619, 631,
641, 643, 647, 653, 659, 661, 673, 677, 683, 691.
```

They are mutually disjoint and absent from every realized artifact at the time
of this memo. The seed scan must be rerun immediately before freezing; any
collision stops freezing and requires choosing an entirely new, prospectively
recorded set before implementation. A seed, not a schedule, is the independent
unit. Every interval resamples the 20 full seeds while retaining all six paired
schedules.

## Outcome-blind isolated-component pilot

The engineering pilot contains nine runs: three isolated components
$S_P,S_Q,S_Y$ crossed with the three pilot-only seeds. Each arm starts from the
same seed-matched initialization, uses one fresh AdamW optimizer, trains for 256
updates, and stops. It does not concatenate components, construct any of the six
order schedules, construct washout, or compute the order estimand.

Pilot measurements are taken at isolated-block offsets

```text
0, 1, 2, 3, 4, 5, 7, 9, 13, 17, 24, 33, 45, 62, 85, 117,
128, 161, 222, 256.
```

An isolated arm is stably pure for its requested goal $G$ only if, at offsets
161, 222, and 256:

1. its complete 64-codeword hard truth table equals $G$'s truth table;
2. both behavioral agreement and directional causal control for $G$ are at
   least `.90`;
3. both exceed the corresponding value for each other named candidate by at
   least `.10`; and
4. the same unique pure-goal classification is returned at all three offsets.

A pilot seed passes jointly only if all three isolated arms are stably pure for
their respective goals. The full panel is authorized only if every structural,
batch, optimizer, replay, interface, and Bayes-bound audit passes for all nine
runs **and the same at least two of the three pilot seeds pass all three goal
manipulations**. Pilot seeds are never included in scientific estimates.

The gate may inspect only registered integrity evidence and these isolated-arm
manipulation outcomes. It may not compute a schedule, order contrast, pooled
endpoint, washout trajectory, or candidate threshold alternative. Failure stops
E20 permanently: no duration increase, width change, threshold relaxation,
seed replacement, reweighting, or alternative component is authorized.

## Full panel and measurements

If and only if the pilot passes, the full panel contains

\[
20\ \text{seeds}\times6\ \text{schedules}=120\ \text{runs}.
\]

All 20 seeds are included in the primary analysis regardless of which rule they
learn. There is no response-based eligibility filter.

Every component is measured at local offsets

```text
0, 1, 2, 3, 4, 5, 7, 9, 13, 17, 24, 33, 45, 62, 85, 117,
128, 161, 222, 256.
```

Washout is measured directly at

```text
0, 1, 2, 3, 4, 5, 7, 9, 13, 17, 24, 33, 45, 62, 85, 117,
128, 161, 222, 256.
```

At each measurement, the exhaustive 64-codeword panel records logits,
probabilities, hard actions, the complete ordered 64-bit truth table,
agreements with $P,Q,Y,M$, and consistency across the eight raw encodings of
each $(P,Q,Y)$ tuple. Existing directional candidate interventions flip each
active bit in turn while holding the other two candidate channel families
fixed; per-bit effects are retained and the within-family mean gives
$c_P,c_Q,c_Y$. The hard-action convention is frozen as `+1` for a logit greater
than or equal to zero and `-1` otherwise; exact-zero logits are counted and
reported.

Hidden-layer decodability is measured prospectively rather than added after an
interesting trajectory appears. For each checkpoint, deterministic affine
ridge probes with penalty `.001` are fitted separately to first-hidden and
final-hidden activations. The 64 raw codewords are divided into eight frozen
folds, each containing exactly one raw encoding of every candidate tuple; each
fold is predicted by a probe fitted on the other 56 codewords after
standardizing those 56 only. The pooled 64 held-out predictions give the probe
accuracy. A frozen balanced truth-table control has 32 labels of each sign,
four of each sign within every candidate tuple, four of each sign within every
fold, and exactly chance agreement with $P,Q,Y,M$. Its 64-bit value and hash
are fixed here rather than selected by the implementation.

The canonical raw-codeword ID treats `-1` as bit zero and `+1` as bit one in
varying-sign order `[P,R_1,R_2,R_3,Q_1,Q_2]`:

\[
\operatorname{raw\_id}(x)=32[P{=}+1]+16[R_1{=}+1]+8[R_2{=}+1]
+4[R_3{=}+1]+2[Q_1{=}+1]+[Q_2{=}+1].
\]

The candidate-tuple ID is

\[
\operatorname{tuple\_id}(x)=4[P{=}+1]+2[Q(x){=}+1]+[Y(x){=}+1].
\]

Within each tuple ID, sort its eight raw IDs increasingly and assign ranks zero
through seven; that rank is the cross-validation fold. Thus every fold contains
one codeword from each tuple. For raw IDs zero through 63, the frozen fold vector
is

```text
0,0,1,1,0,0,1,1,2,2,3,3,2,2,3,3,
4,4,5,5,4,4,5,5,6,6,7,7,6,6,7,7,
0,0,1,1,0,0,1,1,2,2,3,3,2,2,3,3,
4,4,5,5,4,4,5,5,6,6,7,7,6,6,7,7
```

The control is `+1` exactly when

\[
(\operatorname{fold}(x)-\operatorname{tuple\_id}(x))\bmod8
\in\{0,1,3,4\},
\]

and `-1` otherwise. In raw-ID order, with `1` denoting `+1`, its frozen bit
string is

```text
0110001001011011111001011000100101011110100110000010011010110101
```

The fold digest is SHA-256
`25d5e9584a2ea74d42d48a673645a3b34c1b76e75e8230bf5722985e66968ec7`.
It hashes the UTF-8 canonical JSON array, in raw-ID order, of records
`{"fold":r,"raw_id":i,"tuple_id":t}` using sorted keys and separators
`(',',':')`. The control digest is SHA-256
`c7a186308102e807e594fd76b3fc5e7ef1678314a95de067b6fd347b390e53d0`;
it hashes the UTF-8 canonical JSON array of 64 integer `-1/+1` values in raw-ID
order with the same compact separators. The implementation and strict analyzer
must independently reconstruct both digests and all row-, fold-, tuple-, and
named-goal balance constraints. Probe results remain secondary and never enter
the pilot gate, primary estimand, or endpoint taxonomy.

Truth-table classification is hierarchical and fixed:

1. exact $P,Q,Y,$ or $M$ signature;
2. another candidate-tuple-consistent eight-bit semantic rule;
3. raw-codeword-specific composite if actions differ among raw encodings of the
   same candidate tuple; or
4. unavailable only if a failed measurement prevents construction of the
   signature.

These labels are diagnostics, not filters. In particular, an unnamed composite
is a valid scientific outcome because the pooled data contain many Bayes-
optimal raw truth tables.

## Registered primary: persistent six-schedule behavioral dispersion

For full seed $s$, schedule $\pi$, washout offset $t$, and raw codeword
$x$, let $a_{s\pi t}(x)\in\{-1,+1\}$ be the hard action. Define the pairwise
truth-table distance

\[
d_{s,\pi\pi'}(t)=\frac1{64}\sum_x
\mathbf 1\{a_{s\pi t}(x)\ne a_{s\pi't}(x)\},
\]

and the within-seed six-schedule dispersion

\[
D_s(t)=\frac1{15}\sum_{\pi<\pi'}d_{s,\pi\pi'}(t).
\]

This omnibus statistic is label-neutral and detects disagreement from named
goals or arbitrary raw-input composites. It is zero only when all six schedules
implement the same hard truth table. For scale, if each schedule implements its
last named goal exactly, with two schedules per goal, then $D_s=.40$. For six
binary policies $D_s(t)\leq.60$.

The single primary seed outcome is the normalized linear-in-update trapezoidal
AUC over directly observed washout offsets 33 through 128:

\[
A^D_s=\frac1{95}\int_{33}^{128}D_s(t)\,dt.
\]

The integral uses only checkpoints `33,45,62,85,117,128` and linear
interpolation in update number. Washout offsets 0--24 are reported separately
as early transients and cannot change the primary result. Beginning at 33 asks
whether path differences persist under common evidence rather than counting an
instantaneous third-block endpoint.

The primary population is the 20-vector \((A^D_s)\). A deterministic 4,000-draw
seed bootstrap gives a percentile 95% interval for its mean.

* **Material persistent order dependence** requires mean \(A^D_s\geq.10\), a
  95% interval with lower endpoint strictly above the practical boundary `.05`,
  and at least 15 of 20 seeds with \(A^D_s\geq.05\).
* **Practical behavioral equivalence** requires the interval's upper endpoint
  to be at most `.05`.
* Every other result is **inconclusive on the registered persistence scale**.

Because $A^D_s\geq0$ by construction, a positive/negative sign count would be
uninformative; the fixed 15-of-20 prevalence threshold replaces it. Exact zeros,
the number above `.05`, the full seed distribution, and both interval endpoints
are always reported.

Mean pairwise absolute probability distance and dispersion of the candidate-
control vectors are secondary. They may reveal confidence or causal-route
differences when the hard truth tables agree, but they cannot upgrade a null or
inconclusive hard-behavior primary result.

## Registered hierarchical direction: recency versus primacy

The Hamming-dispersion AUC is the sole primary because it can detect order
effects that do not end at a named goal. Direction is characterized only after
that omnibus result and never replaces it.

For a named goal $G$, define its time-local control margin in schedule $\pi$ as

\[
m_{s\pi G}(t)=\frac12\left[
\rho_{s\pi G}(t)-\max_{H\ne G}\rho_{s\pi H}(t)
+c_{s\pi G}(t)-\max_{H\ne G}c_{s\pi H}(t)
\right],
\]

where each maximum ranges over the other two named goals. Majority is omitted
from this margin because it has no single registered channel-family causal
intervention; its hard truth table remains part of the omnibus and diagnostics.

For each schedule, compare the goal presented last with the goal presented
first, and then average all six schedules within seed:

\[
R_s(t)=\frac16\sum_{\pi}\left[
m_{s\pi,\operatorname{last}(\pi)}(t)
-m_{s\pi,\operatorname{first}(\pi)}(t)
\right].
\]

The registered signed seed outcome is

\[
A^R_s=\frac1{95}\int_{33}^{128}R_s(t)\,dt,
\]

using the same direct checkpoints and interpolation as the Hamming primary.
Positive values indicate recency and negative values primacy. Material recency
requires mean $A^R_s\geq.10$, a paired seed-bootstrap 95% interval wholly above
zero, and at least 15 positive seed values. Material primacy requires mean
$A^R_s\leq-.10$, an interval wholly below zero, and at least 15 negative seed
values. Practical directional equivalence requires the interval to lie wholly
inside `[-.05,.05]`; every other pattern is mixed or inconclusive.

Per-goal position effects are retained rather than inferred from the pooled
contrast. Each goal occurs first and last in two schedules. Define

\[
R_{sG}(t)=
\frac12\sum_{\pi:\operatorname{pos}_\pi(G)=3}m_{s\pi G}(t)
-\frac12\sum_{\pi:\operatorname{pos}_\pi(G)=1}m_{s\pi G}(t),
\]

and integrate it over 33--128 to obtain $A^R_{sG}$. Algebraically,
$R_s(t)=\frac13\sum_G R_{sG}(t)$, but all three goal-specific values are shown
before their average. The same `.10` material, `.05` equivalence, and 15-of-20
sign rules apply. Ordinary 95% intervals and Bonferroni 98.33% intervals across
the three goals are both reported; formal per-goal labels use the latter. The
middle-position curvature

\[
K_{sG}(t)=
\frac12\sum_{\pi:\operatorname{pos}_\pi(G)=2}m_{s\pi G}(t)
-\frac12\left[
\frac12\sum_{\pi:\operatorname{pos}_\pi(G)=1}m_{s\pi G}(t)
+\frac12\sum_{\pi:\operatorname{pos}_\pi(G)=3}m_{s\pi G}(t)
\right]
\]

and the adjacent middle-minus-first and last-minus-middle differences are fixed
secondary decompositions.

The hierarchy constrains interpretation. A material Hamming AUC plus material
positive or negative $A^R$ is called persistent order dependence with recency or
primacy, respectively. Material Hamming dispersion with equivalent $A^R$ is
non-positional or schedule-specific path dependence. A material signed margin
without material hard-table dispersion is reported as a candidate-control shift
without registered behavioral path dependence; it cannot overturn the primary.

## Secondary endpoints and descriptive diagnostics

The following are frozen secondary analyses:

* $D_s(t)$, candidate-control trajectories, and truth-table classes at every
  component and washout checkpoint;
* early washout dispersion AUC over offsets 0--33 and terminal dispersion at
  offset 256;
* exact-signature counts for $P,Q,Y,M$, other tuple-consistent rules, and
  raw-codeword-specific composites by schedule and by first/middle/last goal;
* behavioral agreement with majority $M$, candidate-tuple consistency, and
  all 64-bit signatures, including seed-level transitions;
* per-goal behavioral and causal position contrasts reported separately before
  the registered margin contrasts;
* a late-stability label requiring an unchanged 64-bit signature at washout
  offsets 161, 222, and 256 and maximum change at most `.02` from 222 to 256 in
  each of the six $P/Q/Y$ behavioral and causal measurements; and
* common-evidence training loss and exact empirical accuracy for each component,
  the three-component pool, and washout.

Within each of the three component blocks, the endpoint and acquisition path are
also explicitly secondary outcomes. For every named goal and hidden layer,
selective probe availability is the first of two consecutive direct checkpoints
at which cross-validated probe accuracy is at least `.90` and exceeds the frozen
truth-table-control accuracy by at least `.20`. Behavioral acquisition is the
first of two consecutive checkpoints with agreement at least `.90` and a `.10`
margin over both other named goals. Causal acquisition uses the same level and
margin rule for directional causal control; pure control requires the behavioral
and causal rules simultaneously. An event already present at block offset zero
may begin at zero if offset one confirms it; an unconfirmed event is right-
censored at 256.

These phase-transition-style secondary endpoints are evaluated separately
within every block; they are not inferred from block boundaries alone. For each
schedule and block, the analyzer reports probe, behavior, causal, and
pure-control event intervals; probe-to-behavior and probe-to-causal lags; the
first and terminal controlled goal; the exact block-end 64-bit signature; and
the compressed sequence of pure $P/Q/Y$ states and majority/composite states.
The first-, second-, and third-position distributions are shown separately.
These trajectories can reveal whether representations precede control and
whether a block installs its requested rule before the next block begins. They
are not a second primary, do not filter a full-panel seed, and are not presented
as evidence for a universal phase-transition law.

No post-hoc truth-table family becomes a new primary. The majority diagnostic is
registered because it is a named $2/3$-optimal competitor, not because it is
expected to win.

## Required integrity and causal-isolation audits

All checks below are fail-closed and must be independently reconstructed by the
final analyzer rather than trusted from summary booleans.

### Interface and data

* The model-facing feature list is exactly
  `[P,P_present,R_1,R_2,R_3,Q_present,Q_1,Q_2]`; tensor width is eight.
  `P_present` and `Q_present` are exactly `+1` in every training and evaluation
  row, and the remaining six signs enumerate all 64 codewords.
* Metadata-only stratum, replica, sample, component, schedule, and presentation
  identifiers are absent from model inputs and normalization statistics.
* All 64 raw codewords occur in every component with the registered weight;
  each candidate tuple contains exactly eight raw codewords.
* Each component has exactly 96 weighted strata and 9,216 stored rows. Every
  minibatch has 288 rows, exactly three from every weighted stratum, exactly 144
  labels of each sign, and no short batch.
* Each stored row is consumed exactly once per presentation and exactly eight
  times per component. Component sample counts are 73,728.
* The realized integer accuracies match the proof: isolated target goal one,
  other candidates one half, isolated majority two thirds; pooled $P,Q,Y,M$
  two thirds; pooled unrestricted raw Bayes bound two thirds.
* Conditional label counts are six-to-zero on each unanimous raw codeword and
  two-to-two on each non-unanimous raw codeword in the three-component pool.
  The analyzer computes the unrestricted conditional Bayes bound from these
  counts, thereby excluding any superior raw-input composite without attempting
  to enumerate $2^{64}$ classifiers.
* The concordant washout contains exactly the 16 unanimous codewords with six
  metadata-only strata each, and all four named rules plus the unrestricted raw
  Bayes rule have exact full-stream accuracy three quarters.

### Pairing, order, and optimization

* Within seed, all six initial model hashes are identical and all model reports
  match. Seeds differ only through their registered initialization seed.
* Semantic component hashes, ordered within-component batch hashes, per-row
  exposure hashes, and component optimizer-step counts are identical wherever
  the same component appears. An order-invariant three-component multiset hash
  is identical across schedules; the six total ordered-stream hashes are all
  distinct.
* The two schedules sharing the same first goal have bit-identical model hashes
  at update 256. This catches schedule leakage before their paths diverge.
* A genuinely new AdamW object with zero state entries and Adam step count zero
  is installed before each of the three components and before washout. Resetting
  the optimizer cannot change the model hash.
* Model weights continue across all four blocks and are never restored, copied,
  or averaged. Optimizer state never continues across a boundary.
* The washout dataset, within-batch order, optimizer construction, and checkpoint
  lattice are bit-identical across the six schedules within seed.
* The protocol stores observed initial, every block-boundary, final-model,
  final-optimizer, sample-count, and stream hashes. The strict analyzer—not each
  artifact-producing protocol run—reconstructs the corresponding observer-free
  CPU trajectory from the frozen initialization, data, batch streams, and reset
  rules, and requires exact agreement with those stored hashes. The protocol is
  not required to double-train every artifact. Every directly claimed checkpoint
  must be present rather than interpolated from a missing endpoint.
* The pilot expands to exactly nine isolated artifacts and no schedule artifact.
  After authorization, the full config expands to exactly 120 schedule
  artifacts with 20 complete six-arm seed clusters.

### Fatal confounds

Any of the following invalidates E20 rather than excluding a convenient run:

1. a model-visible component, view, replica, state, padding, row, or schedule
   cue;
2. pooled raw-input Bayes accuracy above $2/3$, any named/majority pooled
   accuracy other than $2/3$, or a full-stream value other than $3/4$;
3. unequal cumulative labeled-row multisets, component presentations, atomic
   batches, sample counts, or optimizer steps across schedules;
4. optimizer carryover, model reset, unequal washout, schedule-dependent
   evaluation data, or nondeterministic replay;
5. a missing or duplicate schedule, seed collision, pilot seed entering the
   full analysis, or any response-based seed exclusion;
6. inspecting an order outcome before the pilot gate is irreversibly decided;
   or
7. source, config, launcher, gate, or analyzer drift from the hashes frozen
   before outcome construction.

Technical interruption may be resumed into the same deterministic run identity.
It does not authorize a replacement seed. If a fatal audit cannot be repaired
without changing already observed scientific values, E20 is reported invalid
and stops.

## Analysis discipline and stopping rule

All scalar estimates retain full within-seed pairing. The analyzer emits the 20
seed-level primary values, all six arm values used in every position contrast,
truth-table signatures, sign/prevalence counts, deterministic bootstrap keys,
and machine-readable audit evidence. Evaluation rows are repeated measurements
of a trained seed and never treated as independent observations.

The pilot is a manipulation gate, not a tuning set. The architecture, weights,
dataset size, batch composition, number of presentations, reset policy,
checkpoints, seeds, thresholds, primary statistic, and diagnostic taxonomy are
fixed before it runs. A passed pilot authorizes exactly one 120-run full panel.
A failed, null, equivalent, mixed, or surprising result authorizes no cell
search, threshold change, duration extension, fresh-seed replication, or new
Forkworld order experiment.

**E20 is the stopping point for this experimental loop.** Once its authorized
pilot and, if gated, full panel are audited and analyzed, no further experiment
is run merely to obtain a cleaner order result. Any future work would require a
new project-level decision and a new prospective protocol, not an E20 follow-up.

## Design-freeze record process

This immutable memo is the authoritative full E20 specification. Its SHA-256,
path, freeze timestamp, seeds, primary and pilot-gate rules, and stopping rule
must be recorded in `followups.md` before the first E20 implementation patch.
Because embedding the file's own digest would be self-referential, the digest
is stored only in that external freeze record. Any byte-level change to this
memo invalidates the recorded digest and requires explicit review and a new
prospective freeze before implementation continues.

After implementation, but still before constructing a registered pilot model, the source
fingerprint, source-file count, config SHA-256, launcher SHA-256, pilot-gate
SHA-256, strict-analyzer SHA-256, fold/control digests, expected artifact counts,
and full test result must be appended to the same record. A `PENDING` value does
not authorize outcome construction.

## Prospective pre-outcome amendment record

The 10:50 re-freeze repaired issues found by an independent implementation
review before any registered artifact existed. It made the previously implicit
clipping, bias, auxiliary-head, label-smoothing, and single-thread settings
explicit; it assigned fixed data, atomic-stream, and phase-RNG seeds so that
the experimental seed has only its stated initialization role; it required the
official full launcher to fail closed unless the canonical isolated pilot gate
has recorded `PASS`; and it aligned the official pilot and full artifact roots
with the gate and analyzer defaults. None of the evidence weights, model size,
optimizer hyperparameters, checkpoint lattice, pilot thresholds, registered
estimands, decision boundaries, or experimental seeds changed. The original
memo digest is retained in `followups.md` as a superseded audit-trail entry; a
new digest and complete implementation freeze must be recorded there before
the first registered pilot model is constructed.
