# Forkworld follow-up experiments

This document internally fixed the design and analysis choices for the overnight
follow-up suite before any full-scale run was launched.  It was not posted or
timestamped publicly, so E10--E13 are described as prospectively specified,
not as an independently verifiable preregistration.  The independent replication
unit is always a training seed.  The ten registered seeds remain

```text
11, 23, 37, 41, 53, 67, 71, 83, 97, 101.
```

The follow-ups target four gaps in the first nine experiment families:

1. optimization can change which accessible rule wins, even when the evidence is
   held fixed;
2. parity is only one exact encoding, and its special subset-independence may be
   doing important work;
3. two-way competition cannot reveal whether a model selects one goal or combines
   several imperfect rules;
4. the existing obstacle-grid check chooses a goal once and then clamps it into a
   frozen navigator, so it does not test repeated reward-relevant decisions.

All primary runs disable prediction and checkpoint storage but retain resolved
configurations, full metric trajectories, summaries, provenance, and completion
markers.  Unless explicitly overridden by the launcher environment, full grids
run on CPU with one numerical thread per worker.

## E10: entropy and the RL shortcut boundary

The existing RL goal selector is a contextual bandit.  Exploration therefore
means stochastic goal actions, not discovering new map states.  At the canonical
setting (`q=.9`, parity degree 3), its mean policy entropy falls from .680 at step
8 to .00344 at step 64, while conflict-set intended reliance falls to zero.  The
policy nevertheless receives tens of thousands of conflict presentations.  This
suggests premature policy saturation, but entropy regularization also changes
loss geometry; the intervention is not interpreted as pure state exploration.

The primary actor--critic grid crosses:

```text
q                    = .50, .75, .90, .99
parity degree        = 2, 3, 4
entropy coefficient  = 0, .003, .01, .03, .10, .30
seeds                = 10
```

This is 720 runs.  At `q=.9, k=3`, the six entropy levels are repeated with
REINFORCE (60 runs) and with a 1,024-scalar actor update subspace (60 runs), for
840 runs total.  Every run uses the existing 5,897-parameter H5 actor interface,
2,048 updates, batch size 250, terminal success reward, zero nuisance entropy,
and choice-level evaluation.

The primary outcome is final greedy intended reliance on the all-conflict set.
Fixed paired contrasts compare beta .10 and .30 with beta zero within every
`q`/degree cell.  A cell is called reliably intended only when at least eight of
ten seeds exceed .9 intended reliance.  Secondary outcomes are acquisition time,
intended-probability trajectory area, final IID reward, entropy, and causal
proxy/exact-channel flips.  The `q=.5` cells are a falsifier: if parity is not
learned without a useful proxy, failure elsewhere is not specifically proxy
lock-in.  An effect confined to actor--critic implicates value estimation; an
effect that raises intended reliance only by sacrificing IID performance is an
objective tradeoff, not a free exploration benefit.

## E11: exact rules beyond parity

The number of input channels is not a model-relative complexity measure.  Four
deterministic exact encodings are therefore calibrated in isolation before being
placed in competition with the same direct proxy:

```text
parity       k=5   output is the product of all signs
majority     k=5   output is the sign of their sum
conjunction  k=5   output is +1 only when every sign is +1
multiplexer  k=6   two address bits select one of four data bits
```

Inputs are sampled conditional on a balanced intended label, and the stated rule
decodes that label exactly.  Padding fixes the competition interface at six rule
channels.  Majority and conjunction intentionally permit lower-order evidence;
that is part of the manipulation and is measured rather than hidden.

The grid crosses rule family, proxy accuracy `q in {.75,.90,.99}`, width
`{8,16,32,64,128}`, depth `{1,2}`, and ten seeds: 1,200 runs.  Every run trains a
proxy-only calibration model, an exact-rule-only calibration model, and the
competition model for 4,096 updates each.

The primary analysis asks whether independently measured decoder accessibility
predicts conflict behavior across rule families.  Matched comparisons use the
same `q`, width, depth, seed, input width, and training exposure.  We report
single-channel marginal predictiveness alongside decoder accuracy, so an easy
majority/conjunction result is not described as a mysterious architectural
preference.  Causal flips remain the behavioral control.

## E12: a frontier with two competing proxies

Each example contains three candidate goals:

1. a direct proxy `P` with accuracy `q_P`;
2. a second proxy `Q`, encoded as a parity code of degree `k_Q`, with accuracy
   `q_Q`;
3. the intended goal `Y`, encoded as a parity code of degree `k_Y` and always
   correct.

The two proxy-error streams have exact marginals and either the nearest
finite-sample overlap expected under independence or maximally nested error
sets.  The latter is a prospectively specified robustness arm: when accuracies match, the
two proxies fail on the same rows; when they differ, only the less accurate
proxy receives unique counterexamples.  Realized overlap is always recorded.
The grid is

```text
q_P       = .90, .99
q_Q       = .90, .95, .99
k_Q       = 2, 3
k_Y       = 3, 5
width     = 16, 64
error overlap = independent, nested
seeds     = 10
```

This gives 960 runs.  Input width is fixed at the maximum proxy and intended-code
degrees.  Each run calibrates `P`, `Q`, and `Y` separately before training their
competition.

Ordinary IID and all-conflict accuracy cannot identify which rule controls the
model.  Three diagnostic panels are therefore fixed in advance:

```text
both wrong: P = Q = -Y
P wrong:    P = -Y, Q =  Y
Q wrong:    P =  Y, Q = -Y
```

The primary estimands are agreement with each candidate on these panels and the
hard/probability effects of flipping `P`, one active `Q` channel, and one active
intended channel.  A model is described as selecting one goal only when the
diagnostic and causal results agree.  Distributed causal effects are reported as
an ensemble or mixture rather than forced into a winner label.

## E13: RouteWorld with repeated reward-relevant forks

The richer task is a walled, perfect-binary-tree maze.  A depth-`D` map contains
`D` true junctions and `2^D` terminal leaves.  Corridors are equal length, every
leaf is reachable, all non-corridor cells are walls, and the exact horizon makes
a wrong branch irrecoverable.  The target leaf is not visually marked.  A
scripted corridor controller executes forced moves and queries the learned shared
selector only at a junction, so navigation capability cannot masquerade as goal
inference.

Each stage has an independent balanced intended branch bit, a direct proxy with
exact per-stage accuracy `q`, and a degree-`k` parity code.  A stage address is
visible; node identity and target identity are not.  This tests a shared
stage-indexed selector, not history- or node-conditioned replanning.  Maximum
depth and degree padding keep parameter counts fixed.

The grid crosses:

```text
route depth       = 1, 2, 4
proxy accuracy    = .75, .95, .99
parity degree     = 2, 4
width             = 16, 64
evidence regime   = fixed total updates, fixed updates per fork
seeds             = 10
```

This is 720 runs.  The base budget is 1,024 updates with batch size 256.  The
fixed-total arm always receives 1,024 updates; the per-fork-matched arm receives
`1,024*D`, keeping expected presentations per stage comparable to depth one.
Training contains 4,000 route episodes, exactly realizing every registered
accuracy within every stage and label stratum.

Primary outcomes are branch accuracy, exact full-route success, effective
per-fork success, the pooled compounding approximation `branch_accuracy^D`,
and first-divergence hazard.  A product of the stage-specific branch accuracies
is retained as an explicitly exploratory refinement of the registered pooled
approximation.  This terminology was clarified after execution: only the
stage-specific product is literally an independence prediction when stage
accuracies differ.  Evaluation includes IID routes, all stages in
conflict, and exactly one conflict stage.  Paired channel flips test causal rule
control.  Stage-local flips separately measure the response at the intervened
stage and spillover to other stages.  Scripted physical rollouts validate walls,
equal path lengths, and vectorized success accounting; they execute the route
selected by the batch model and are not online history-conditioned evaluations
or additional independent observations.
Reversing or removing the visible stage address is an evaluation-only causal
control for whether the shared selector actually indexes the relevant route
channel; it is not treated as another trained replicate.

The central distinction is fixed in advance.  If branch-level reliance is stable
but route success follows its compounded prediction, environment complexity
amplifies a local error without changing the learned goal.  A depth-dependent
branch-reliance shift that remains after per-fork evidence matching is evidence
that sequential/stage-indexing complexity changes rule selection itself.

## Conditional mechanistic follow-up

If E10 finds an entropy coefficient that improves mean intended reliance by at
least .20 over beta zero with a paired 95% seed-bootstrap interval above zero,
we will run a staged prevention-versus-reheating panel at `q=.9, k=3`.  Phase A
contains 64 beta-zero updates, after which existing runs are already saturated.
Phase B compares continued baseline, delayed entropy, and behavior-policy
temperatures 2, 4, and 8, with a critic-reset control.  If the trigger is not met,
this panel is not run and no post-hoc temperature story is added.

The four unconditional grids contain 3,720 runs.

## E14: exploratory timing of entropy exposure

This panel was specified only after inspecting the completed E10 grid and is
therefore explicitly exploratory and post-hoc.  The prospectively specified conditional
trigger at the canonical `q=.90`, degree-3 cell did not fire.  E10 nevertheless
revealed a different responsive boundary cell at `q=.75`, degree 4: beta `.30`
raised final mean conflict-set intended reliance by about `.442` relative to
beta zero, even though its final policy entropy was only about `.0014`.  E14 is
designed to distinguish an effect of entropy before policy saturation from
rescue after saturation; it is not treated as confirmatory evidence for the
conditional panel above.

The phase boundary is fixed at optimizer step 64.  In the responsive cell, the
beta-zero policy has mean conflict intended reliance zero and mean policy
entropy about `.0065` by this point.  The beta-`.30` policy has not yet developed
appreciable intended behavior at step 64, but retains mean entropy about `.081`.
Every run therefore contains 64 phase-A updates and 1,984 phase-B updates, for
the same 2,048 updates and batch size 250 as E10.  Actor weights always continue
across the boundary.  The six schedules are fixed as follows:

```text
schedule                 phase A beta   phase B beta   boundary state
zero_zero                0              0              carry actor Adam and critic
high_high                .30            .30            carry actor Adam and critic
early_only               .30            0              carry actor Adam and critic
delayed_carry            0              .30            carry actor Adam and critic
delayed_actor_reset      0              .30            reset actor Adam only
delayed_critic_reset     0              .30            reset critic weights and Adam only
```

The two cells are the responsive `q=.75`, degree-4 cell and the canonical
`q=.90`, degree-3 negative control.  Crossing two cells, six schedules, the ten
registered seeds, and no other factors gives exactly 120 runs.  Both continuous
schedules pass through the same phase boundary as the timing interventions.
Generator state and the shuffled semantic sampler continue across phases, so
`zero_zero` and `high_high` are exact phase-split counterparts of uninterrupted
training rather than fresh-randomness controls.

The primary endpoint is final conflict-set intended reliance.  Fixed paired
seed contrasts are `early_only - zero_zero`, `delayed_carry - zero_zero`,
`early_only - delayed_carry`, `high_high - early_only`,
`delayed_actor_reset - delayed_carry`, and
`delayed_critic_reset - delayed_carry`.  Secondary outcomes are phase-B
intended-reliance area, time to two consecutive evaluations at or above `.50`,
IID reward, policy entropy, critic loss, and boundary/final causal effects from
the direct proxy and parity channels.  The time-to-event outcome is secondary
because E10's responsive seeds were visibly multimodal.

Phase A is evaluated at global steps `1, 2, 4, 8, 16, 32, 64`.  Phase B is
evaluated after local steps `1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 1984`,
which correspond to global steps `65, 66, 68, 72, 80, 96, 128, 192, 320, 576,
1088, 2048`.  Any interpretation of schedule differences will be labelled
exploratory and boundary-specific unless independently replicated.

## E15: adaptive multi-goal temporal bridge

This experiment was designed after inspecting E12 and is explicitly exploratory.
E12 already establishes a robust behavioral ordering: under a conservative pure-goal
signature with two-checkpoint persistence, all 960 competition models first acquire
the direct proxy.  The selected 120-run slice then divides into 52 `P -> Q`, 44
`P -> Y`, 16 `P`-only, and eight `P -> Q -> Y` trajectories.  Those records do not
contain hidden-state probes or intermediate causal interventions, however.  E15
therefore targets the missing distinction between information being linearly
available, controlling the output causally, and appearing in behavior.  It is a
focused bridge and continuation, not a new broad search for a favorable cell.

The grid fixes `q_P=.90`, `k_Q=2`, width 64, depth-two ReLU SFT, and crosses

```text
q_Q             = .90, .95, .99
k_Y             = 3, 5
proxy errors    = independent, nested
seeds           = the original ten plus ten fresh seeds
```

The fresh seeds are `103, 107, 109, 113, 127, 131, 137, 139, 149, 151`.  The 12
cells and 20 seeds give 240 runs.  The original interface is retained exactly:
`max_k_Q=3`, `max_k_Y=5`, eight nuisance-state coordinates, batch size 250,
learning rate .003, and zero weight decay.  The three standalone calibration
models stop after the original 2,048 updates.  Only the competition model
continues to 8,192.  Evaluation uses every original E12 checkpoint through 2,048
and adds `2435, 2896, 3444, 4096, 4871, 5793, 6890, 8192`.  Before the continuation
is interpreted, the original seeds must exactly reproduce all archived discrete
candidate diagnostics and causal summaries available at step 2,048.  The old
artifacts contain no model-state hash, so no stronger bitwise-state claim will be
made.

Two disjoint exhaustive panels enumerate every active raw combination of the
direct proxy, the two encoded-proxy bits, and the intended-code bits.  Repetition
over independently generated state and padding streams gives exactly 2,048
probe-fitting rows and 4,096 held-out rows in every cell.  Consequently all eight
candidate triples `(P,Q,Y)` are balanced and pairwise independent, and flipping
any active candidate channel maps to another point on the evaluation support.
This panel supports a complete eight-cell behavioral truth table and distinguishes
a function of the three candidate recommendations from raw-codeword or nuisance
patches.

At every competition checkpoint E15 measures four concepts separately:

1. standalone accessibility in matched full-interface, masked `P`, `Q`, and `Y`
   calibration models;
2. affine-ridge probe accuracy on frozen first and final hidden representations;
3. behavioral agreement with every candidate and the full candidate truth table;
4. directional causal control under `P`, `Q`, and `Y` channel flips.

The probe is standardized using only its fitting split and scored on disjoint
held-out rows.  Raw-input and random-initialization results are baselines.  A
sample-label permutation checks fitting leakage.  A balanced random truth-table
target, fixed across probe train and test and exactly orthogonal to `P`, `Q`, and
`Y`, measures generic codeword memorization by random ReLU features.  For candidate
`j`, selective availability requires true held-out accuracy at least .90 and a
true-minus-random-function advantage of at least .20.  Generic availability at
.95 is also retained.  In particular, the directly visible `P` is available at
initialization and is never described as a newly acquired internal goal.  Probe
accuracy alone is not evidence of motivation.

On the decorrelated factorial panel the ordinary signed probability ATE can cancel
even under complete proxy control.  The primary hard causal estimand is therefore

```text
c_j = mean[g_j(x) * (a(x) - a(flip_j(x))) / 2],
```

with normalized score `(1+c_j)/2`: unrelated control is .5, pure positive control
is one, and pure inverse control is zero.  Probability-scale and absolute-change
versions are secondary; active components are retained separately and averaged
within the `Q` and `Y` families.

Standalone accessibility uses accuracy .95.  Generic probe availability uses .95;
the selective event uses the joint rule above.  Behavioral acquisition uses
candidate agreement .90, and causal acquisition uses normalized directional
control .90.  Every event requires two consecutive checkpoints and remains
right- or interval-censored rather than being imputed as a success.  A pure
candidate-control phase additionally requires behavioral and causal thresholds
and a .10 margin over the other candidates.  The primary summaries are event
order and lags, first controlled goal, compressed sequences such as `P -> Q -> Y`,
time in pure versus distributed states, truth-table structure, and matched effects
of `q_Q`, `k_Y`, and error overlap.  Training seeds are the independent units;
examples and checkpoints are not.  Original and fresh seeds are reported
separately before pooling.

Frequent cycles would weaken the phase account.  Behavior--causal disagreement
invalidates a pure-goal label.  Intended causal control appearing before the fixed
probe threshold would count against that probe criterion rather than prove that
unrepresented information controls behavior.  If genuine and random-function
probes both saturate at initialization, no representation-acquisition time will be
claimed.  The next adaptive experiment is chosen only after this panel is audited.

## E16: adaptive support completion of a conditional endpoint

This experiment was designed after inspecting E15 and is explicitly adaptive and
post-hoc.  Thirty of E15's 240 runs ended at Boolean signature 113.  On the
exhaustive `(P,Q,Y)` panel this policy uses `Y` when `P=Q` and otherwise uses `Q`.
The endpoint was concentrated in the nested-error arms.  In those arms the
training support contains no examples on which `Q` alone is wrong, and signature
113 is correct on every observed candidate tuple while failing exactly on that
absent tuple.  E16 tests the narrow causal explanation that this structured
endpoint is sustained by the support hole, rather than treating it as a smooth
mixture of motivations.

The model and evidence cell is fixed at the E15 setting in which the endpoint was
most common:

```text
q_P                    = .90
q_Q                    = .95
k_Q                    = 2
k_Y                    = 5
width, depth            = 64, 2
training examples       = 10,000
optimizer updates       = 8,192
Q-only error count m    = 0, 2, 10, 50, 100, 250, 450
seeds                   = the same 20 E15 seeds
```

This gives 140 runs.  The direct-proxy and encoded-proxy marginal accuracies are
held exactly fixed.  With 1,000 `P` errors and 500 `Q` errors, arm `m` has

```text
both wrong     = 500 - m
P only wrong   = 500 + m
Q only wrong   = m
neither wrong  = 9,000 - m.
```

The counts therefore move probability mass from the shared-error and neither-error
cells into the two single-error cells without changing either proxy's marginal
accuracy.  The smallest nonzero arm supplies only two counterexamples in 10,000;
the largest still leaves 50 shared errors, so all four cells are represented.

Arms are paired within seed.  They share the intended labels, row IDs and order,
`P` error mask, intended-code channels, nuisance state, padding channels, and the
random prefix used to encode `Q`.  Fixed seeded rankings make the retained shared
errors nested as `m` increases and make the new `Q`-only errors a growing prefix
of the same pool.  Only the rows required to change `Q`'s error membership, and
therefore the parity bit required to encode the changed `Q`, differ.  The
`m=0` arm must reproduce the original E15 nested training array exactly, not
merely its aggregate counts.  The held-out IID set is fixed to the original
nested distribution for every arm, so evaluation data do not move with the
training intervention; its `m=0` copy must also match E15 exactly.  Batch sampling,
initialization, and optimizer randomness are paired across `m`.

The `m=450` arm has the same four cell counts as the finite-sample independent
E15 arm: 50 shared, 950 `P`-only, 450 `Q`-only, and 8,550 neither-error rows.
Its row allocation is deliberately inherited from the nested support-completion
ranking, however, so comparison with archived independent runs is descriptive and
count-matched, not an exact replay.  By contrast, `m=0` is subject to an exact
archived E15 nested-metric audit before new outcomes are interpreted.

The primary endpoints at update 8,192 are (i) the seed-level incidence of a
stable signature-113 endpoint, defined as modal signature 113 with factorial
tuple consistency at least .90, and (ii) intended-target accuracy on the fixed
`Q`-only diagnostic panel.  The modal signature is never called an exact policy
when row-level consistency is below that threshold.  The decisive pattern is a
dose-ordered loss of the stable endpoint accompanied by improved `Q`-only target
behavior while performance on the common nested IID panel remains high.  Pure-`Y`
and pure-`Q` endpoint incidence, all eight truth-table actions and probabilities,
tuple/codeword/nuisance consistency, candidate agreements, and directional causal
effects are secondary.  The E15 checkpoint schedule, exhaustive panels, frozen
probes, and standalone calibrations are retained so that the analysis can also
locate when a support-induced behavioral change appears.  Probe availability is
not interpreted as a motivation, and the two examples in the smallest arm are not
treated as independent replicates; across 8,192 shuffled updates they are presented
many times.  The exact cumulative number of `Q`-only presentations, number of
unique `Q`-only rows seen, and first/all-seen updates are replayed from the paired
sampler and recorded at every checkpoint.  The training seed remains the unit of
analysis.

Completing the two decoded `Q`-only candidate tuples does not necessarily complete
their raw active-codeword support: with one `P` bit, two active `Q` bits, and five
active `Y` bits, those tuples contain 64 codewords.  E16 therefore records how many
distinct `Q`-only active codewords occur in training.  At every checkpoint the
fixed exhaustive panel reports target accuracy and target probability separately
on `Q`-only codewords seen and unseen in training, as well as in aggregate.  A
change confined to seen codewords is interpreted as patching or memorization; a
change that generalizes to unseen codewords is evidence for a semantic support
effect.  State and padding features are not part of this seen-codeword definition.

All seven registered counts are run before looking at outcomes; there is no
sequential stopping inside the 140-run panel.  Results are reported by count and
with paired seed trajectories, not only as a pooled trend.  A possible refinement
is deterministic.  Let `p_m` be stable signature-113 prevalence among the 20 seeds.
Refinement is allowed only if `p_0-p_450 >= .30` and the largest prevalence drop
between adjacent registered counts is at least .20.  The earliest interval tied
for that largest drop is selected, and up to three unused integer counts at its
rounded quarter points are run with the same 20 seeds.  Duplicate rounded counts
are discarded.  If either numerical trigger fails, or the selected interval has
no unused quarter-point counts, this line of experimentation stops.  No new model
width, accuracy, complexity, or error-structure cells will be searched under this
experiment label.  Any triggered refinement remains post-hoc and is reported
separately from the fixed 140-run panel.

## E17: adaptive winner-knockout handoff

This experiment was selected after inspecting E15 and E16 and is explicitly
adaptive and post-hoc.  E15 showed that the encoded proxy `Q` is often linearly
decodable before it controls behavior.  That temporal ordering does not establish
that the decoded rule is prepared to take over when the current winner fails:
the model might still have to learn `Q` from scratch.  E17 therefore installs a
direct-proxy winner, removes only its predictive information, and compares the
subsequent handoff with matched reset, scratch, and compute-sham controls.

The phase-A cell is fixed at

```text
q_P, q_Q              = .90, .90
k_Q, k_Y              = 2, 3
width, depth           = 64, 2
batch size             = 250 (40 full minibatches per 10,000-row epoch)
phase-A updates        = 45
phase-B updates        = 1,024
phase-A error overlap  = independent or nested
seeds                  = 157, 163, 167, 173, 179, 181, 191, 193, 197, 199,
                         211, 223, 227, 229, 233, 239, 241, 251, 257, 263
```

This cell was chosen because every matching E15 run was still in a pure
`P`-controlled state at updates 33 and 45, while final-hidden `Q` probe accuracy
at update 45 differed sharply between independent and nested error histories.
The branch update is fixed globally at 45 and is never selected from a later
trajectory.  A historical prefix is eligible when `P` has behavioral agreement
at least `.90`, directional causal score at least `.90`, and margins of at least
`.10` over both `Q` and `Y` on both measurements at updates 33 and 45.  Eligibility
is measured before phase B, failed seeds are reported and never replaced, and the
primary comparisons use the intersection of seeds eligible under both overlap
histories.  The analysis stops as a design failure if this paired intersection
contains fewer than 15 of 20 seeds.  `Q` probe accuracy is a moderator, not an
inclusion rule.

There are six scientifically unique trajectories per seed:

```text
independent history, carry AdamW state
independent history, reset AdamW state
nested history, carry AdamW state
nested history, reset AdamW state
scratch initialization, fresh AdamW
45-update balanced-label sham, reset AdamW
```

The scratch and sham baselines are shared across the two overlap histories and
are not duplicated under an arbitrary overlap label.  Twenty seeds times six
trajectories gives exactly 120 artifact runs and 120 scientific branch
trajectories.  Deterministic prefix replays are internal integrity checks, not
additional trajectories or replicates.  A single common sham cannot simultaneously
match the distinct independent- and nested-history feature arrays.  It therefore
uses a disjoint 10,000-row factorial-feature batch constructed as 5,000 paired
clones.  The two rows in a pair have identical channels, state, padding, and
presence indicators but opposite sham labels.  Every active raw codeword is
represented before cloning.  The sham label is consequently exactly balanced
conditional on the complete model input, not merely marginally uncorrelated with
`P`, `Q`, and `Y`; there is no learnable label signal in the paired empirical
distribution.  This arm controls only 45 updates of generic optimization; it is
not an overlap-matched feature-exposure control and cannot identify the
independent--nested history contrast.  That clean equal-compute contrast is
supplied by the two historical reset arms.  The sham optimizer is reset before
phase B.

Phase B is an in-support informational knockout rather than a missing-channel
intervention.  It keeps the exact target `Y`, a degree-two `Q` code with exact
accuracy `.90`, the degree-three exact `Y` code, all presence indicators, the
fixed-width interface, and the state and padding distributions.  Its 10,000 rows
are 5,000 paired clones: the two members share `Q`, every active `Q` and `Y` bit,
the target, state, padding, and presence indicators and differ only in `P=+1`
versus `P=-1`.  Thus `P` is exactly balanced conditional on the entire remaining
model input, rather than only on decoded `(Q,Y)`.  Consequently `P` agrees exactly
`.50` with each of `Q` and `Y`, whereas `Q` agrees exactly `.90` with `Y`; all
eight candidate triples are represented.  The phase-B batch, its row order, and
its minibatch-index stream are bit-identical across all six trajectories for a
seed and are disjoint from phase A.  Dataset construction must audit pairwise
input equality, opposite `P`, all active-bit balances, IDs, digests, and sampler
identity before training.

Historical prefixes are deterministically rebuilt on CPU because E15 did not
save update-45 states.  Each historical run independently replays its 45-update
prefix and requires exact model-state and optimizer-state hashes before it may
branch.  The carry branch restores both states; the reset branch restores only
the model state.  Every phase-B arm starts a new, common sampler stream so the
carry--reset contrast changes optimizer history rather than example order.
Step-33 and step-45 prefix measurements, replay hashes, eligibility, and all
phase-B balance and pairing digests are retained in the artifact summary.

Phase B is evaluated at local updates

```text
0, 1, 2, 3, 4, 5, 7, 9, 13, 17, 24, 33, 45, 62, 85, 117,
128, 161, 222, 304, 418, 575, 790, 1024.
```

At every checkpoint the fixed exhaustive factorial panel records candidate
behavior, the full Boolean truth table, first- and final-hidden affine-ridge
probes with the E15 controls, and directional causal effects of `P`, `Q`, and `Y`
channel flips.  A `Q` handoff is the first of two consecutive checkpoints at
which `Q` behavioral agreement and directional causal score are each at least
`.90` and each exceeds the corresponding `P` and `Y` values by at least `.10`.
The event remains interval- or right-censored at 1,024 rather than being imputed.

The primary continuous outcome is the linear-in-update trapezoidal area under
the mean of `Q` behavioral agreement and `Q` directional causal score from local
updates 0 through the directly observed update 128, normalized to `[0,1]`.
Secondary outcomes are restricted
mean time to `Q` handoff through 1,024, local-zero `Q` control, `Q` probe dynamics,
the first non-`P` controlled goal, analogous `Y` handoff, final truth-table
signature, and factorial target accuracy.

The frozen seed-paired primary contrasts are reset-independent minus
reset-nested, reset-independent minus scratch, reset-independent minus sham, and
carry minus reset within each history.  Intervals resample the 20 training seeds;
checkpoints and evaluation rows are repeated measurements, not replicates.
Before readiness is interpreted, the step-45 manipulation check requires the
independent-minus-nested paired difference in selective final-hidden `Q` probe
accuracy to be at least `.10` with its 95% interval above zero.  Selective accuracy
is the true-`Q` held-out accuracy minus the orthogonal truth-table-control accuracy.
For `Q`-specific language, the paired intervals for `P` behavior, `P` causal
score, and final-hidden `Y` probe accuracy must each lie inside `[-.05,.05]`.
Failure is reported rather than repaired with seed selection; a broader
representation-history effect may still be described without calling it prepared
`Q`.  Phase-B update zero explicitly records `P` strength and the behavior,
causality, and probe accuracy of both `Q` and `Y` so alternative starting-state
differences remain visible.

History-specific readiness requires positive `Q`-control-area contrasts for all
first three comparisons with 95% intervals above zero and directionally matching
restricted-time contrasts.  An optimizer-history effect is called material only
when the carry--reset area difference has magnitude at least `.05` and its
interval excludes zero.  Practical equivalence requires the area interval to lie
inside `[-.05,.05]` and the restricted-time interval inside `[-16,16]` updates.

All 120 trajectories are run before interpreting outcomes and no adaptive cells
are added.  If reset-independent is practically equivalent to reset-nested,
scratch, and sham, the knockout line stops: decodability did not confer measurable
readiness in this design.  If reset-independent is faster but carry and reset are
equivalent, the result is attributed to learned weights or representations rather
than optimizer moments and the line also stops.  A later literal absence
intervention is warranted only if the readiness contrasts are positive but the
random-`P` negative evidence drives every arm across the handoff threshold within
the first logged interval; it is not part of E17.

## E18: identical-evidence diagnostic order

This experiment was selected after E17 and is adaptive and post-hoc.  E17 found
that an independently trained encoded proxy was ready to take control faster,
but carrying rather than resetting Adam did not produce a material control-area
effect.  E18 therefore asks whether models trained on an exactly identical
multiset of evidence can retain different goals solely because diagnostic
evidence arrived in a different order.  Fresh optimizers at both causal
boundaries make terminal persistence a learned-weight effect rather than
residual optimizer memory.

The fixed cell is

```text
q_P, q_Q                 = .90, .95
k_Q, k_Y                 = 2, 3
width, depth              = 64, 2
training rows             = 10,000
batch size                = 250
presentations per row     = 100
shared A-prefix batches   = 3,240
diagnostic B/D batches    = 200 each
common A-washout batches  = 360
total batches             = 4,000
seeds                     = 269, 271, 277, 281, 283, 293, 307, 311, 313, 317,
                            331, 337, 347, 349, 353, 359, 367, 373, 379, 383
engineering-pilot seeds   = 389, 397, 401
```

Nested support gives three active evidence categories:

```text
A: 9,000 rows, P = Q = Y
B:   500 rows, P != Y and Q = Y
D:   500 rows, P = Q != Y
```

There are no `Q`-only-error rows.  The batch is selected from a deterministic
exhaustive factorial source.  Each category contains all 16 compatible active
raw codewords and is balanced within target sign.  In A, four codewords under
each sign have 563 unique rows and four have 562.  In B and D, two codewords per
sign have 32 rows and six have 31.  State, padding, and row IDs remain unique.
This realizes the exact category counts and proxy accuracies without random
raw-codeword frequency as an unregistered nuisance.

Every atomic optimizer batch has 125 rows of each target sign and 15 or 16
presentations of every compatible raw codeword.  Quotas rotate deterministically
so every selected row appears exactly once per registered repetition.  The first
90 A presentations form a bit-identical 3,240-batch prefix.  All 100 B and D
presentations form 200 batches each.  The remaining 10 A presentations form a
bit-identical 360-batch washout.  The schedules are

```text
A90 -> B100 -> D100 -> A10
A90 -> D100 -> B100 -> A10
A90 -> alternating B_i,D_i -> A10
```

Within-category batches are bit-identical across schedules.  Audits require
identical per-row exposure, identical order-invariant atomic-batch multisets,
identical component hashes, distinct ordered-stream hashes, full batches, exact
label balance, and the registered raw-codeword counts.  Row-exposure and atomic-
batch-multiset digests are reported separately.

The A90 prefix is rebuilt with and without measurement callbacks and must have
identical initial, final-model, final-optimizer, data, and stream hashes.  At
update 3,240 its optimizer is discarded for an audited empty AdamW common to all
schedules.  There is no reset between diagnostic blocks.  The model is measured
immediately after matched diagnostic evidence at update 3,640; that optimizer is
then discarded and a second audited empty AdamW is installed.  All schedules
consume the same A washout.  Cross-artifact hashes must show an identical model
at the shared prefix boundary and identical empty optimizer states at both reset
boundaries.

Measurements use the exhaustive E15 behavior, Boolean-signature, directional-
causal, and affine-ridge-probe panels.  Registered landmarks are update 3,240,
the first diagnostic-half boundary 3,440, second-half offsets

```text
1, 2, 3, 4, 5, 7, 9, 13, 17, 24, 33, 45, 62, 85, 117, 128, 161, 200,
```

and post-diagnostic A-washout offsets

```text
0, 1, 2, 3, 4, 5, 7, 9, 13, 17, 24, 33, 45, 62, 85, 117, 128,
161, 222, 304, 360.
```

The time-local intended-control margin is

```text
m_Y(t) = 1/2 * [(rho_Y - max(rho_P,rho_Q))
                + (c_Y - max(c_P,c_Q))].
```

The primary seed outcome is its linear-in-update trapezoidal AUC from washout
offset 0 through directly observed offset 128, normalized by 128.  The paired
contrast is `Delta = AUC(D then B) - AUC(B then D)`.  It is called material in
the frozen primacy direction only if `mean(Delta) >= .10`,
the lower endpoint of its paired seed-bootstrap 95% interval is above zero, and
at least 15 of 20 seed differences are positive.  A mean at most `-.10`, an
upper interval endpoint below zero, and at least 15 negative seed differences is
reported as a direction-reversed recency effect rather than as support for the
primacy prediction.  Practical equivalence requires the interval to lie inside
`[-.05,.05]`.  Anything else is inconclusive and triggers no cell search.
Interleaving is a descriptive diffuse-evidence reference.

Terminal update-4,000 `m_Y` is secondary.  A run is late-stable only when it has
the same Boolean signature and the same unique pure controlled goal at washout
offsets 222, 304, and 360, and the largest absolute change from offset 304 to
360 among all six candidate behavioral agreements and directional causal scores
is at most `.02`.  Pure control uses `.90` levels and `.10` behavioral-and-causal
margins.  Terminal categorical hysteresis in the predicted direction requires a
mean paired terminal difference of at least `.10`, a 95% interval whose lower
endpoint is above zero, and at least 15 positive seed differences.  In addition,
at least 15 paired seeds must be late-stable in both blocked arms with `D then B`
purely `Y`-controlled and `B then D` purely controlled by a non-`Y` goal.  A
continuous terminal difference that fails this categorical rule is described as
strength variation, not as a different learned goal.

Before any E18 outcomes were run, we amended the design to use three pilot-only
seeds disjoint from the confirmatory panel, tightened the directional materiality
threshold, and strengthened late stability as specified above.  This amendment
supersedes the initial compact E18 memo.

Seeds 389, 397, and 401 form a separate nine-run engineering pilot and are never
included in confirmatory estimates.  The gate never examines primary or terminal
order effects.  The 60-run panel on the 20 listed confirmatory seeds launches
only if every evidence, batch, replay, reset, and checkpoint invariant passes
and the same at least two pilot seeds are purely `P`-controlled at the shared
prefix, purely `Q`-controlled after a B-first half, and purely `Y`-controlled
after a D-first half at update 3,440.  Failure stops without replacement.  A
further fresh-seed replication is permitted only if both the material primary
rule and terminal categorical-hysteresis rule pass; no accuracy, complexity,
width, or optimizer grid is searched under E18.

### E18 pilot disposition

The nine-run engineering pilot completed with every registered integrity,
replay, reset, checkpoint, fingerprint, and independently reconstructed atomic-
stream audit passing.  All three seeds were purely `P`-controlled at the shared
prefix.  At the first-half boundary, however, B-first was purely `Q`-controlled
for only seed 397 and D-first was purely `Y`-controlled for no seed.  Thus zero
of three seeds passed the joint manipulation gate, below the frozen requirement
of two, and the 60-run panel was not authorized.

This is a structural failure rather than evidence about order.  Every B row has
`Q = Y = -P`, while every D row has `Y = -P = -Q`.  Consequently the one-bit
rule `-P` is perfectly accurate in either isolated diagnostic block and across
their union.  Five of the six blocked first halves learned its Boolean signature
`11110000`; increasing the duration would strengthen rather than remove the
confound.  No primary AUC, terminal value, post-diagnostic scientific value,
washout scientific value, or metric value was inspected.  E18 therefore ends at
its blinded engineering gate and contributes no confirmatory order estimate.

## E19: active-Q input-pathway intervention

This adaptive, post-hoc experiment follows E17.  E17 showed that an independent
history makes `Q` more decodable and makes `Q` take control sooner after `P` is
knocked out, but history changed the whole network.  E19 asks a narrower causal
question: do the learned first-layer weights from the two active `Q` bits
contribute to that later adaptability?  It does not identify the affine-probe
statistic itself as a mediator, and a null would not exclude distributed `Q`
information in biases or downstream weights.

Five already-used E17 seeds, 157, 163, 167, 173, and 179, were used only for
engineering calibration and are excluded from every E19 estimate.  Restoring
the active `Q` columns to initialization reduced `Q`-control AUC through update
128 in all five seeds (mean `-.0329`); transplanting the independently prepared
columns into the nested model increased it in all five (mean `+.0330`).  Applying
the same signed displacement to padding columns changed the respective AUCs by
means `-.0016` and `-.0001`.  These archived results fixed the layer, columns,
effect threshold, and controls before any fresh E19 pilot outcome.

The fixed training cell is E17's winner-knockout cell:

```text
q_P, q_Q                 = .90, .90
k_Q, k_Y                 = 2, 3
max k_Q, max k_Y         = 3, 5
width, depth             = 64, 2
phase-A rows             = 10,000
phase-A updates          = 45
phase-B updates          = 1,024
batch size               = 250
learning rate            = .003
weight decay             = 0
optimizer                = AdamW
full seeds               = 409, 419, 421, 431, 433, 439, 443, 449, 457, 461,
                            463, 467, 479, 487, 491, 499, 503, 509, 521, 523
pilot-only seeds         = 541, 547, 557
```

For every seed, both independent- and nested-error phase-A histories are built
from the same initialization and replayed with and without registered observers.
Initial and final model hashes, final optimizer hashes, data hashes, static-
sampler hashes, sample counts, and update counts must agree exactly within each
replay.  The natural histories must be purely `P`-controlled at updates 33 and
45.  The feature interface must hash to

```text
[P, P_present, R_1, R_2, R_3, R_4, R_5,
 Q_present, Q_1, Q_2, Q_3, state_0, ..., state_7].
```

The only edited parameter is the `64 x 19`
`input_projection.weight`.  Let `W_0`, `W_I`, and `W_N` be that matrix at
initialization and after the independent and nested prefixes.  Let active
columns `Q = [8,9]` denote `Q_1,Q_2`, and sham columns `C = [5,6]` denote the
inactive `R_4,R_5` padding signs.  The six branches are

```text
independent_noop          W_I
independent_q_restore     W_I[:,Q] <- W_0[:,Q]
independent_padding_sham  W_I[:,C] <- W_I[:,C] + (W_0[:,Q] - W_I[:,Q])
nested_noop               W_N
nested_q_transplant       W_N[:,Q] <- W_I[:,Q]
nested_padding_sham       W_N[:,C] <- W_N[:,C] + (W_I[:,Q] - W_N[:,Q]).
```

Replacement uses `index_copy_` and sham addition uses `index_add_`; advanced
indexing mutation is forbidden.  An audit must show that exactly 128 registered
scalars changed, all other state tensors are bit-identical to the recipient,
donors did not mutate, and each active edit and its sham have bit-identical
flattened signed-displacement digests and equal L1, L2, maximum, and element-
count summaries.  Preactivation RMS changes and ReLU-flip rates are recorded on
the factorial and phase-B panels because the sham controls edit size and value,
not feature geometry.

The 18-run pilot contains only the six post-edit branches on the three pilot
seeds.  It stops before constructing or training on phase B and therefore
contains no handoff outcome.  Every integrity check must pass and, in the same
at least two pilot seeds:

* all six edited models remain purely `P`-controlled under the `.90/.10` rule;
* restoring the active columns lowers the final-hidden selective `Q` probe by
  at least `.05`, while its sham changes it by at most `.05` in magnitude;
* transplanting the active columns raises that probe by at least `.05`, while
  its sham changes it by at most `.05` in magnitude; and
* every edit changes `P` behavioral agreement, `P` causal score, selective `P`
  decoding, and selective `Y` decoding by at most `.05` from its paired no-op.

Failure stops E19 without changing the layer, columns, scale, duration, seeds,
or thresholds.  Pilot seeds are never included in a scientific estimate.

If the pilot passes, all six branches run on the 20 fixed full seeds.  Surgery
is followed by a genuinely fresh, audited empty AdamW optimizer and the exact
same paired E17 phase-B batch, which randomizes `P` while retaining `Q` at .90.
The phase-B row order, sampler digest, factorial panels, and probe panels are
bit-identical across branches within seed.  Direct checkpoints are

```text
0, 1, 2, 3, 4, 5, 7, 9, 13, 17, 24, 33, 45, 62, 85, 117, 128,
161, 222, 304, 418, 575, 790, 1024.
```

The fixed 20 seeds form the primary population; edit response never filters a
seed.  The paired pure-`P` history intersection at updates 33 and 45 must contain
at least 15 seeds for a handoff interpretation, and its restricted analysis is
reported as a sensitivity check.

Let `A_b` be the normalized linear-in-update AUC through directly observed
update 128 of the mean `Q` behavioral agreement and directional causal score in
branch `b`.  The two co-primary, positive-direction contrasts are

```text
Delta_N = A_independent_padding_sham - A_independent_q_restore
Delta_S = A_nested_q_transplant      - A_nested_padding_sham.
```

Necessity-like or sufficiency-like evidence in its registered direction requires
the corresponding mean to be at least `.02`, the lower endpoint of its paired
seed-bootstrap 95% interval to exceed zero, and at least 15 of 20 seed
differences to be positive.  In addition, each padding-sham minus no-op AUC
interval must lie wholly inside `[-.01,.01]`.  Bidirectional pathway evidence
requires both directional rules and both sham-equivalence rules; otherwise the
directions are reported separately.  Direct active-edit versus no-op contrasts
and the natural independent-minus-nested history gap are secondary.

The full-panel manipulation check reports post-edit probe shifts and requires
the active restore and transplant mean shifts to be at most `-.05` and at least
`+.05`, respectively, with intervals excluding zero.  Sham probe shifts and
all registered `P`/`Y` preservation contrasts use `[-.05,.05]` equivalence
intervals.  Failure downgrades downstream differences to descriptive surgery
effects rather than pathway evidence.  `Q` handoff times, probe dynamics, first
non-`P` controlled goal, final `Y` control, preactivation changes, and ReLU flips
are secondary.  No mediated-fraction claim, further cell, or replication is
authorized under E19.

This design was frozen before any E19 pilot model was built or outcome was run.

The implementation gate was completed before the pilot launch.  The full
repository suite passed (273 tests), the CLI expanded to exactly 18 pilot and
120 full artifacts, and the source-fingerprint-v1 value was frozen as
`ab91250cbb8379c1e6afd0f960abe0754b08a745bd2d9fd8922be0f65a1ba7a9`
(31 package files). 

The outcome-blind pilot gate was also made executable before launch.
The gate passed its 15 focused implementation tests before any directory named
`artifacts-e19-pilot` existed.

After all 18 pilot artifacts were constructed, the gate stopped in its
structural audit before calling the manipulation-gate evaluator or emitting any
probe shift.  The only failures were an impossible bitwise condition on the
*realized* sham displacement: in float32, computing `baseline + delta` and then
subtracting `baseline` need not reproduce `delta` bit for bit.  The constructor
had supplied the exact registered delta tensor to `index_add_`, the final sham
columns exactly matched the registered float32 addition, and the observed norm
roundoff was below `5e-7`.  Before viewing any manipulation value, we therefore
amended the audit—not the models, artifacts, gate thresholds, seeds, or
scientific analysis—to retain exact intended/active and final-column checks and
allow at most `1e-6` absolute discrepancy in the sham displacement L1, L2, and
maximum norms.  A synthetic test requires the amended audit to fail above that
tolerance.  This paragraph records the correction before rerunning the gate.

### E19 pilot disposition

All 18 manipulation-only artifacts completed under the frozen source
fingerprint.  The amended gate audited 22,158 metric records plus the complete
replay, edit, configuration, run-identity, and cross-arm pairing evidence; it
verified that no phase-B data, metric, prediction, optimizer, checkpoint, or
outcome was constructed.  Seeds 541 and 557 passed the joint manipulation gate,
meeting the registered two-of-three rule.  Every branch in every seed remained
purely `P`-controlled, and all registered `P`/`Y` preservation checks passed.
Active restore changed the selective `Q` probe by `-.1296`, `-.1099`, and
`-.1140`; active transplant changed it by `+.0642`, `+.0925`, and `+.0872`.
Seed 547 failed only the restore-sham `Q` equivalence check (`-.0518` versus the
fixed `-.05` boundary); its other checks passed.  The gate therefore authorizes
the frozen 120-run full panel without changing any cell, seed, edit, threshold,
or analysis rule.

### E19 full-analysis audit trail

The artifact-blind full analyzer and its synthetic fail-closed tests were frozen
before opening the full artifacts. Ruff and the focused protocol/analyzer tests passed. The first
application to the 120-run root stopped in the structural audit and did not
compute or emit any contrast. Its 10,440 errors were all the same schema
mismatch: the analyzer required each stored probe layer to contain only the
three semantic labels and the truth-table control, whereas the protocol also
stores the four registered sample-permuted negative controls in that mapping.
No artifact is being changed. The validator will be amended to require the
actual eight-label schema exactly, a synthetic regression test will be added,
and new analyzer/test hashes will be recorded before rerunning the audit. The
estimands, thresholds, bootstrap keys, decisions, and scientific interpretation
rules remain frozen.

The amended full analyzer now requires all eight protocol-emitted labels in
each probe layer and cross-checks all eight against the long-form metric stream;
missing or extra labels remain fatal. Ruff and all 30 focused tests passed before the amended analyzer was applied to
the full artifacts.

### E19 full-panel disposition

The amended analyzer passed all exact audits for 120/120 artifacts. On the
fixed all-20-seed population, restoring the active first-layer `Q` columns
reduced normalized `Q`-control AUC through update 128 relative to its matched
padding sham by `0.036112` (paired seed-bootstrap 95% interval
`[0.029641, 0.043183]`; 20/20 positive seed contrasts). Transplanting those
columns into the nested-history model increased the same AUC relative to its
matched padding sham by `0.029269` (`[0.024874, 0.033867]`; 20/20 positive).
The two padding-sham minus no-op intervals were `[-0.003309, -0.000413]` and
`[-0.001775, 0.001116]`, both wholly inside the registered `[-0.01, 0.01]`
equivalence region. The active `Q` probe shifts were `-0.105835` for restore
and `+0.075122` for transplant; all registered `P`/`Y` preservation intervals
and sham `Q`-probe checks remained within `[-0.05, 0.05]`. All 20 seeds met the
pre-edit eligibility intersection. The registered bidirectional pathway rule
therefore passed. The allowed claim is narrow: these active first-layer `Q`
input columns contribute causally to later `Q` control; this is not a mediated
fraction, a unique-circuit localization, or evidence that every `Q`
representation lies there. All branches ultimately converged to `Y`.

## E20: repaired identical-evidence order design freeze

The E20 design was reviewed and frozen at `2026-08-03T09:37:03+0800`, before
any E20 pilot model or outcome was constructed. The immutable authoritative
specification is
`paper/forkworld-current-results/e20_repaired_order_design.md`.
That memo, rather than this concise record, defines the complete data, batching,
interface, optimizer-reset, measurement, analysis, audit, and interpretation
contract.

Pilot-only seeds are `563, 569, 571`. Full-panel seeds are
`577, 587, 593, 599, 601, 607, 613, 617, 619, 631, 641, 643, 647, 653,
659, 661, 673, 677, 683, 691`. They were selected after scanning 57,705
existing summary artifacts and are disjoint from all 67 realized artifact
seeds. The seed scan must pass again before implementation freeze.

The nine-run outcome-blind pilot crosses the three isolated `S_P`, `S_Q`, and
`S_Y` components with the three pilot seeds. All integrity audits must pass,
and the same at least two seeds must be stably pure for all three requested
goals at component offsets 161, 222, and 256. Failure stops E20 without tuning
or replacement. Passing authorizes exactly the 20-seed by six-permutation
120-run panel.

The sole primary is the within-seed mean pairwise Hamming dispersion of the six
complete 64-codeword hard truth tables, integrated linearly over common-washout
offsets 33--128. Material persistent order dependence requires mean AUC at
least `.10`, a paired seed-bootstrap 95% lower endpoint strictly above `.05`,
and at least 15/20 seed AUCs at least `.05`; practical equivalence requires the
upper endpoint at most `.05`. The signed last-minus-first control-margin AUC is
the hierarchical recency/primacy characterization, not a co-primary, with the
frozen `.10` material, `[-.05,.05]` equivalence, and 15/20 sign rules. Per-goal
formal labels use Bonferroni 98.33% intervals.

The implementation fingerprint, source-file count, config, launcher, gate,
strict-analyzer hashes, expected expansion counts, and test result are
`PENDING`. No pilot artifact is authorized while any value remains pending.
After the authorized full panel is analyzed, E20 stops regardless of whether
the result is material, equivalent, inconclusive, mixed, or surprising: no cell
search, duration change, threshold change, replacement seed, or E20-derived
Forkworld replication is authorized.

### E20 prospective pre-outcome amendment and second design freeze

An independent implementation review was completed while both
`artifacts-e20-pilot` and `artifacts-e20` were still absent. It found that the
first implementation allowed the nominal experimental seed to change data,
atomic-minibatch, and phase-RNG order despite the memo assigning that seed only
to model initialization; the official launcher did not fail closed on the
isolated pilot gate; its default artifact roots did not compose with the gate
and analyzer; and several active deterministic training settings were implicit
rather than written. No registered model or outcome was constructed under that
implementation.

The authoritative design memo was therefore prospectively amended and
re-frozen at `2026-08-03T10:50:00+0800`, before implementation resumed. The amended memo explicitly freezes data seed `171000001`,
atomic-stream seed `171000002`, phase seeds `171000101` through `171000104`,
experimental-seed use only for initialization, gradient-norm clipping at
`1.0`, affine biases, no auxiliary head, zero label smoothing, and a
single-thread official CPU path. It also requires the full launcher to validate
the canonical pilot `PASS` record and aligns the official pilot/full artifact
roots with the gate/analyzer defaults. Evidence weights, architecture size,
optimizer hyperparameters, checkpoints, pilot thresholds, scientific
estimands, decision boundaries, and registered experimental seeds are
unchanged.

All second-freeze implementation, config, launcher, gate, analyzer, seed-scan,
expansion, and full-suite values remain `PENDING`; the original pending record
does not authorize pilot construction.

The amended implementation gate was completed at
`2026-08-03T11:27:35+0800`, still before either registered artifact root or the
canonical pilot-gate record existed. A fresh scan found 57,705 existing
summaries, 67 realized seeds with maximum 557, no collision with any of the 23
E20 seeds, and zero existing `h17` artifacts. The pilot and full configurations
expand to exactly 9 and 120 runs. The complete repository suite passed 372
tests; the final E20-focused suite passed 88 tests and the generic
data/protocol regression selection passed 34.

The authorization chain is deliberately acyclic. The gate pins the source,
memo, and pilot config but records the live launcher hash. The launcher pins
the gate-program hash and passes it to the source-fingerprinted guard. Before
any full run, that guard requires the same prospective gate hash from the
launcher, current gate bytes, and gate record; independently reconstructs the
exact seven-field gate contract and all ordered 27 checkpoint evidence rows;
and reconciles them with arm, seed, and joint PASS summaries. The full analyzer
pins and rechecks the source, gate, launcher, both configs, and complete pilot
authorization before opening a full-panel scientific artifact. Synthetic
tests reject gate-plus-record co-drift, missing/duplicated/malformed evidence,
and every frozen-input mismatch. These final values supersede every `PENDING`
entry above and authorize only the frozen nine-run pilot. The full panel remains
unauthorized unless that pilot's canonical strict gate records `PASS`.

### E20 pilot disposition

All nine isolated-component pilot artifacts completed under the final acyclic
freeze. Before reading a manipulation endpoint, the gate audited and replayed
45,369 metric records, the exact source/config/run identities, fixed data and
atomic streams, the eight-channel interface, fresh optimizer, model hashes,
64-row measurements, causal summaries, probe folds, and construction-isolation
contract. All three pilot seeds were stably exact and purely controlled by each
requested goal at offsets 161, 222, and 256. At offset 256, behavioral
agreement and directional causal control were both `1.0` for every one of the
nine requested seed-goal arms. Thus 3/3 seeds passed all three manipulations,
exceeding the frozen two-of-three gate.

### E20 full-analysis audit trail

All 120 frozen full-panel artifacts completed with zero runner failures or
skips. The first application of the strict full analyzer stopped during its
metric-stage structural audit, before constructing a seed outcome, contrast,
bootstrap, decision, or scientific output. The parser removed only the final
underscore-delimited token from a stage name; for the registered two-token
suffix `truth_table`, this left a base ending in `truth` and raised while
parsing that word as a component position. No artifact, source package,
configuration, seed, threshold, checkpoint, estimand, or interpretation rule
will change. The analyzer will recognize the already frozen suffix as one
token, a regression test will exercise the real stage schema, and new analyzer
and test hashes will be recorded before the structural audit is rerun.

The artifact-blind parser repair was completed and independently verified
before the full artifacts were reopened. It matches the registered suffixes
exactly, preferring the longest match so that `truth_table` is indivisible;
validates only the frozen component and washout bases; reconstructs the frozen
global-step origin; and turns malformed mutations into fail-closed audit
errors.

### E20 full-panel disposition and experiment-loop stop

The amended strict analyzer passed all 120 of 120 frozen artifacts and audited
2,417,880 metric records. It reconstructed and cross-checked the source,
configuration, fixed data and atomic streams, phase contexts, metric grid,
model and optimizer boundaries, observer-free replay, six-schedule pairing,
truth tables, equal component multisets, and 20 distinct initialization hashes.
No response-based seed filter was applied. 

The registered primary mean Hamming-dispersion AUC over washout offsets
33--128 is `0.398346`, with seed-bootstrap 95% interval
`[0.395883, 0.399803]`; all 20 seed outcomes exceed the material prevalence
boundary. The hierarchical signed recency AUC is `0.994487`, interval
`[0.986203, 0.999359]`, and is positive and material in all 20 seeds. Formal
Bonferroni 98.33% last-minus-first intervals are
`[0.975027, 0.999910]` for P, `[0.997985, 1]` for Q, and
`[0.974724, 0.999819]` for Y. Middle-minus-first effects are approximately
zero, while last-minus-middle effects are at least `0.992`; the result is
winner-take-last rather than a smooth position gradient. Mean dispersion
remains `0.395781` after the complete 256-update common washout. Every
schedule-checkpoint policy retains exactly `2/3` accuracy on the pooled three-
component evidence even while the learned truth tables differ.

All 360 component blocks finish with the exact requested rule, so blockwise
goal replacement generalizes to the three-goal setting. However, the stronger
probe--behavior--causal staging does not. First-hidden probes precede, coincide
with, and follow requested behavior in 225, 50, and 85 blocks; final-hidden
counts are 177, 90, and 93. Behavioral, directional-causal, and pure-control
event timing coincides in all 2,160 phase rows. Nearly every later replacement
passes through a non-pure codeword-specific policy. These endpoints are
secondary and show that repeated control replacement is robust while probe
timing is layer-, rule-, threshold-, and checkpoint-dependent.

The full-panel decision is therefore `persistent order dependence with
recency`, with the explicit scope that whole component order is manipulated,
the pooled objective is underdetermined on disagreement codewords, and washout
is non-disambiguating. The result does not establish persistence against
uniquely identifying negative evidence or a universal phase-transition law.
The experiment contributes 120 scientific runs; the nine pilot artifacts
remain engineering-only.

E20 now stops under the rule frozen before its pilot: no cell search, duration
change, threshold change, replacement seed, or E20-derived Forkworld
replication is authorized after this outcome. Together with the earlier
evidence, it resolves the remaining high-value within-test-bed question about
whether repeated multi-goal acquisition and order dependence survive a
complete counterbalance. Further small-MLP Forkworld sweeps would mostly vary
degree rather than distinguish a live interpretation; the experiment loop is
therefore closed rather than consuming the remaining wall-clock budget on a
redundant panel.