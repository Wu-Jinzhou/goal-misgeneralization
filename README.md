# ForkWorld: controlled experiments on learned goals

ForkWorld is a research package for testing nine preregistered hypotheses and a
set of controlled follow-up questions about AI goal misgeneralization. It
separates **which goal a policy selects** from **whether it can navigate to that
goal**, then measures goal choice on held-out states where an intended signal
and a proxy disagree.

The original experimental specification is [`forkworld.md`](forkworld.md); the
follow-up designs, decision rules, and post-hoc labels are recorded in
[`followups.md`](followups.md). This repository implements them as deterministic
dataset generators, exact update-budget models, SFT/on-policy/RL trainers,
three task levels, phased unlearning protocols, seed-level statistics, and
resumable sweep launchers.

## GoalZendo: language-model experiments

The LLM scale-up is an independent package under `src/goalzendo`, with its own
configs, launchers, artifacts, protocol registry, and report. It asks which of
an explicitly stated Official Law, a visible Herald shortcut, and a second
semantic Sage rule controls a fine-tuned model's choice when the three are made
to disagree. A candidate is said to control behavior only when it predicts
choices on a balanced conflict panel and passes matched input interventions.
This is a behavioral measurement relevant to goal misgeneralization; it is not
a definition or direct measurement of motivation, a unique utility function,
or a subjective desire.

Install the optional LLM dependencies using the GPU stack recorded during the
Runpod boundary audit, then inspect the frozen engineering plan:

```bash
python -m pip install -c constraints-goalzendo.txt -e '.[llm,dev]'
goalzendo validate configs/goalzendo/g00_engineering.yaml
goalzendo plan configs/goalzendo/g00_engineering.yaml
./runs/goalzendo/00_smoke.sh \
  --output-root "$PWD/artifacts-goalzendo-local/g00-smoke" \
  --set run.device=cpu
```

`constraints-goalzendo.txt` matches the Runpod PyTorch 2.8 / CUDA 12.8 image.
On another CUDA stack, install its compatible PyTorch wheel first and retain
the exact Transformers, PEFT, tokenizer, and Accelerate pins from that file.
The explicit output root and CPU override make the smoke command safe to run
from a macOS checkout; it is a tiny full-model plumbing check, not a scientific
run, and downloads the pinned 0.5B model on first use.

The contemporary Qwen3.5 study is complete and analyzed: all 96 prespecified
main-panel runs finished. When the Herald perfectly predicted training reward
(`q_P=1`), every run achieved IID Official-Law accuracy 1.000, yet all 48
models followed the Herald on balanced disagreements; matched interventions
also attributed their action changes to the Herald. With 5% distinguishing
evidence (`q_P=.95`), all 24 SFT runs and all 12 majority-Law outcome-RL runs
followed the Official Law, while all 12 parity-Law outcome-RL runs still
followed the Herald. The pattern was the same at both model sizes. Successful
training behavior therefore did not by itself reveal what would control the
model when training-compatible sources came apart, and the effect of
distinguishing evidence depended on both the learning signal and the Law
family. These are claims about behavioral control in this testbed, not a
recovery of unique internal goals, motivations, or representations.

The 36-run Qwen3.5 evidence-geometry follow-up is also complete. Across changes
to joint shortcut-error overlap and conflict diversity, 34 endpoints followed
the Herald exactly and two 2B endpoints produced a constant action; none
followed the Official Law or Sage. The registered overlap and diversity
contrasts had descriptive sign-flip \(p=1\). Their small negative means came
only from the two constant-action reference endpoints, so they are not evidence
that richer evidence harmed Official-Law control or that the conditions are
equivalent. In this controlled setting, neither aggregate counterexample counts
nor their arrangement made rewarded training behavior sufficient to predict
what controlled actions when the candidate sources separated.

A separate 24-run Qwen3.5 finite-choice hidden-Law study has launched under a
frozen analysis plan. It asks whether failures arise during active evidence
acquisition, exact rule identification, or the later use of an identified
rule to control behavior. Its scientific outcomes have not yet been analyzed;
the study is therefore part of the prospective extension, not evidence for
the claims above.

The earlier Qwen2.5 capability-repair and known-Law plans are preserved as
archived designs; their proposed 160-run and 120-run panels were never
launched and are not relabeled as evidence for the Qwen3.5 study. ForkWorld
remains a separate experiment family with separate artifacts and reports.
Protocols, prospective studies, and the historical engineering record are
indexed in [`docs/goalzendo`](docs/goalzendo/README.md).

The canonical machine-readable execution/deviation record is the
[GoalZendo study-status ledger](reproducibility/goalzendo/study-status-ledger.json).
It records the compact Qwen3.5 result identities and canonical analysis hashes
without placing the raw model-run panel in the source release. The historical
Qwen2.5 gate evidence remains available under
[`reproducibility/goalzendo/g00d-gate-20260811`](reproducibility/goalzendo/g00d-gate-20260811/README.md).
The current manuscript source is
[`paper/goalzendo-current-results/main.tex`](paper/goalzendo-current-results/main.tex).
The checked `zendo.pdf` remains the last release-bound compiled snapshot until
the launched hidden-law panel is complete and the final manuscript is rebuilt.

## What is implemented

| Hypothesis | Primary manipulation | Primary test |
|---|---|---|
| H1: simplest sufficient goal | proxy accuracy × interaction degree × capacity | phase diagram of `rho_Y - rho_P` |
| H2: architecture-relative complexity | depth, width, activation, residuals, update mode | calibrated decoder threshold predicts behavior |
| H3: easy-to-hard acquisition | dense log-spaced checkpoints | proxy acquired before exact goal; replacement time |
| H4: distinguishing evidence | conflict count × unique conflict contexts | unseen-conflict generalization from diverse vs repeated conflicts |
| H5: conditional RL advantage | algorithm × exact actor update budget × nuisance entropy | `U_min(SFT) / U_min(RL)` grows with nuisance entropy |
| H6: temporal noise | step-, episode-, state-static, or biased noise | matched-variance robustness and sample-count slopes |
| H7: counterevidence | removal, decorrelation, reversal, replacement | censor-aware proxy-reliance half-life and restoration rebound |
| H8: hysteresis | old-goal volume × alignment × perturbation | rebound and reactivation versus never-old-goal controls |
| H9: conditional multiplicity | context × model/update capacity | strict context switching and a 2×2 causal sensitivity matrix |

The extensions add entropy-regularized RL, four alternative exact Boolean-rule
families, three simultaneously competing goals with controlled error overlap,
and RouteWorld mazes with repeated reward-relevant forks.  A final 120-run
entropy-timing panel is explicitly exploratory because it was designed after
inspection of the entropy sweep.

Every hypothesis uses the same semantic conventions: `Y` is the intended binary
goal, `P` is a simple proxy, and `R_1...R_k` form an exact interaction code with

```text
product(R_1, ..., R_k) = Y.
```

The model input has fixed width across `k`.  Inactive interaction channels are
filled with deterministic independent signs and marked inactive in metadata, so
parameter count does not accidentally change with signal degree.

## Installation

Use Python 3.10–3.13.  A CPU installation is sufficient for all smoke tests and
the one-step sweeps.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[dev]'
```

For the closest reproduction of the completed E10--E14 artifacts, use Python
3.11.14 and the exact recorded scientific-runtime constraints:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -c constraints-paper.txt -e '.[dev]'
```

The broad bounds in `pyproject.toml` support development on other compatible
versions; `constraints-paper.txt` fixes the NumPy, PyTorch, SciPy, pandas,
Matplotlib, and PyYAML versions recorded in the completed artifact metadata.

PyTorch accelerator selection is explicit: pass `--device cpu`, `cuda`, `cuda:0`,
or `mps`.  `--device auto` chooses CUDA, then MPS, then CPU.

## Fast verification

Run the unit suite and one end-to-end experiment:

```bash
pytest
./runs/00_smoke.sh
```

To exercise one tiny cell from every original hypothesis:

```bash
SMOKE=1 ./runs/all.sh
```

Smoke mode reduces data, steps, seeds, and map count.  It verifies plumbing and
invariants; it is not powered to support or reject a scientific hypothesis.

The dated, non-authorizing repository release verifier is documented in
[`reproducibility/releases/2026-08-11`](reproducibility/releases/2026-08-11/README.md).
Its quick mode checks exact evidence, source-component, G00-F freeze, toolchain,
paper, and reviewed static-debt bindings. This includes the checkpoint-A bridge
as a source-only package and strictly replays its canonical archive, manifest,
freeze, build runtime, transaction, and nonauthorization; it remains
deliberately absent from the release wheel. The accepted checkpoint-B
coordinator package is also source-only and wheel-excluded; quick verification
binds its exact five-file source/refusal milestone while preserving every
runtime and launch prohibition. It also binds the four-file unbuilt source
capsule implementation and both five-file G01Q source/refusal milestones. The
qualification and preprovision packages remain source-only, wheel-excluded,
and nonauthorizing.
Full mode additionally rebuilds the wheel and paper and runs scoped static
analysis plus the complete test suite.

## Running experiments

Each shell script resolves the repository root, sets `PYTHONPATH`, is restartable,
and accepts the CLI arguments documented below.

```bash
./runs/00_pretrain_navigators.sh
./runs/01_simplicity.sh
./runs/02_complexity.sh
./runs/03_dynamics.sh
./runs/04_conflict_diversity.sh
./runs/05_update_capacity.sh
./runs/06_noise.sh
./runs/07_unlearning.sh
./runs/08_hysteresis.sh
./runs/09_multiplicity.sh
./runs/10_analyze.sh
./runs/11_rl_exploration.sh
./runs/12_rule_families.sh
./runs/13_competing_goals.sh
./runs/14_routeworld.sh
./runs/15_entropy_timing.sh
```

Run the four prospectively specified follow-up grids together, optionally sharded across
parallel CPU processes, with:

```bash
DEVICE=cpu OUTPUT_ROOT="$PWD/artifacts-followups-rerun" \
  ./runs/followups.sh --jobs 10

.venv/bin/python paper/forkworld-current-results/followup_analysis.py \
  --artifacts "$PWD/artifacts-followups-rerun"
```

The exploratory entropy-timing panel has a separate launcher and should be kept
in a separate artifact root:

```bash
DEVICE=cpu OUTPUT_ROOT="$PWD/artifacts-e14-rerun" \
  ./runs/15_entropy_timing.sh --jobs 10

.venv/bin/python paper/forkworld-current-results/e14_analysis.py \
  --artifacts "$PWD/artifacts-e14-rerun" \
  --e10-artifacts "$PWD/artifacts-followups-rerun"
```

Use a fresh output root for a new reproduction. The default analysis points at
the completed `artifacts-followups` and `artifacts-e14` snapshots; adding
new-fingerprint reruns to those roots would correctly make the strict
all-or-nothing analyzers reject the mixed provenance.

Those multi-gigabyte artifact roots are local working data and are not part of
a normal source checkout. A fresh checkout must either rerun the commands above
or obtain the artifact snapshots separately; the compact plotted rows in
`paper/forkworld-current-results/derived` remain suitable for auditing the
reported figures without rerunning training.

The archived E10--E13 artifacts use implementation fingerprint
`d2122ee7b98b4be99fcf0d7b2b5127f6a9da60de7b4b7814f4abee8acc7352cb`.
Adding the later E14 implementation changed the full-package fingerprint to
`58b3e0dfc743500419a980b1a1abc5c539b794cd814cb69adcd7c873c5fa1e67`,
which identifies the archived E14 source. Later ForkWorld experiments evolved
the package further. E20 executed with the prospectively frozen 34-file
fingerprint
`4ca022c5b2d2d9a75c8d443cdc0e422d178989bae825c2c2d37d41fc7a14e539`;
the current release tree has fingerprint
`bdb9b95c23e1cdb65c5ec74b465940ee5c0e8d2fc3881edc8c248d594f80b276`
after a post-freeze, non-scientific import/type cleanup in `analysis.py`.
The E20 trust-anchor test reconstructs the exact frozen bytes and digest, while
the strict analyzer continues to reject the current-source drift before reading
artifacts. None of these historical fingerprints is silently rebased to the
current tree. The older E10--E13 digest identifies those archived runs, but
because they were executed from a dirty working tree it is not a claim that
their exact source bytes can be recovered from the recorded Git commit.

## Completed results and report

The current analyzed evidence contains 54,010 prospectively specified runs
(50,290 original plus 3,720 follow-ups), all complete with no failed artifacts.
The 120 completed timing runs are reported separately as exploratory. The blog
post, full methods appendix, and all plotted data are in
[`paper/forkworld-current-results`](paper/forkworld-current-results).

Regenerate all analyses and figures from the repository root with:

```bash
.venv/bin/python paper/forkworld-current-results/analysis_and_plots.py
.venv/bin/python paper/forkworld-current-results/followup_analysis.py
.venv/bin/python paper/forkworld-current-results/e14_analysis.py
```

The checked-in configurations are paper-scale experiment specifications, not laptop demo
settings.  Inspect a plan before launching it:

```bash
forkworld validate --config configs/h01_simplicity.yaml
forkworld plan --config configs/h01_simplicity.yaml
forkworld plan --config configs/h01_simplicity.yaml --smoke
```

Run a pilot or a deterministic cluster shard:

```bash
forkworld run \
  --config configs/h01_simplicity.yaml \
  --max-runs 20 \
  --jobs 4 \
  --device cpu

forkworld run \
  --config configs/h01_simplicity.yaml \
  --shard-index 3 \
  --shard-count 16 \
  --device cuda:0
```

Override any setting without editing the preregistration:

```bash
forkworld run \
  --config configs/h03_dynamics.yaml \
  --set train.steps=16384 \
  --set run.seeds='[201,202,203]' \
  --output /path/to/artifacts
```

Completed run IDs are skipped by default.  Run IDs are hashes of the fully
resolved scientific configuration and seed, not filenames parsed during
analysis.  Use `--force` only when deliberately replacing a completed run.

The shell launchers also accept environment overrides:

```bash
OUTPUT_ROOT=/scratch/forkworld DEVICE=cuda:0 ./runs/01_simplicity.sh
PYTHON_BIN=/path/to/python SMOKE=1 ./runs/06_noise.sh
```

`runs/all.sh` forwards run-only flags such as `--jobs`, `--force`, and
`--dry-run` to every original hypothesis. `runs/followups.sh` does the same for
E10--E13. For aggregate launches, set the shared output,
device, and smoke mode with `OUTPUT_ROOT`, `DEVICE`, and `SMOKE=1`; this keeps
navigator pretraining, experiment artifacts, and analysis on the same roots.

## Goal-selection design

The learning problem is factored into two modules:

```text
proxy / interaction / context signals
                 |
                 v
          learned goal selector  ----->  selected goal ID {-1,+1}
                                                |
                                                v
                                    frozen goal-conditioned navigator
                                                |
                                                v
                                        reached left/right target
```

This factorization prevents a navigation failure from being mistaken for a goal
failure.  Every sequential evaluation reports:

- selector accuracy;
- success at the selected goal;
- end-to-end success at the intended goal;
- oracle-goal success;
- success with each goal clamped;
- path efficiency and timeouts.

The frozen navigator is supervised from deterministic BFS labels.  ForkWorld
contains one meaningful symmetric left/right choice.  The richer navigation
level uses seeded obstacle maps and carves target-safe routes when necessary, so
both goals are reachable by construction.  End-to-end goal-selection results use
the navigator's pretraining-support maps, where its clamped capability is known;
generalization to unseen maps is reported separately.  A low held-out navigation
score therefore diagnoses planning generalization without contaminating the goal
selection estimand.

## Primary estimands

Training accuracy does not identify a learned goal.  The primary evaluation is a
balanced OOD set with `P = -Y`.  For a proxy rule `g_j`, behavioral reliance is

```text
rho_j = Pr[argmax policy(action | x) = g_j(x)].
```

All standard protocols report `rho_Y`, `rho_P`, `delta_rho`, soft intended-goal
probability, confidence, and invalid/tie rate.  Paired counterfactual panels flip
exactly one channel while retaining the semantic/map/sample ID.  In particular,
an interaction intervention flips one `R_i`; flipping every channel would leave
even-degree products unchanged.

Acquisition, replacement, half-life, and reactivation times are stored with an
`observed` flag and censoring horizon.  A run that never crosses a threshold is
not silently dropped or recorded as if it crossed at the final step.
H3 reports the raw intended-over-proxy dominance crossing separately; its
replacement clock starts only after a sustained proxy acquisition has first been
confirmed, so an initially intended policy cannot be mislabeled as replacing a
proxy policy that never existed.

## Important controls encoded in the implementation

- Dataset generation uses exact finite-sample proxy counts.  An unrealizable `q`
  raises an error rather than being rounded.
- H2's proxy calibration estimates `D(P)->P` (rather than the `q`-capped
  `D(P)->Y`). Proxy and exact calibrations retain the competition model's full
  input width and state interface while zeroing every non-target coordinate.
  Architecture contrasts use exact parameter matches or the preregistered 5%
  relative tolerance; closest pairs outside the tolerance are reported as
  unmatched rather than pooled.
- H4 rejects inconsistent simultaneous choices of total `N`, `q`, and
  `N_conflict`, because `q = 1 - N_conflict/N`.
- H4 structured holdout uses named, invertible failure mechanisms (location
  reflection, coordinate exchange, geometry rotation, nuisance inversion), with
  disjoint train/evaluation families balanced within each `Y` stratum. Mechanism
  identity and presence are analysis-only latents, preventing the shortcut
  `Y=P*(-1 if conflict else 1)`.
- Train and conflict-state IDs are disjoint.  State-static noise is keyed by a
  stable semantic ID, independent of row order and dataloader workers.
- H5 counts the exact actor scalars exposed to the optimizer. Its critic uses a
  separate, fixed over-capacity architecture whose weights are trained normally;
  critic parameters are not included in the actor update budget.
- H5 nuisance-rich SFT uses fixed-horizon `NeutralChoiceSimulator` episodes.
  `nuisance_entropy=H` means the first `H` gadgets have fair, reward-equivalent
  branch choices; remaining gadgets use a forced branch. All nuisance heads stay
  in every actor so architecture and exact update budget remain matched. Each
  gadget then merges deterministically before the final goal action, preserving
  reward timing and horizon. Trajectory SFT resamples labels for only the `H`
  active branches and sums one NLL per branch plus the final-goal NLL; inactive
  heads receive no supervision. RL instead samples those same branch actions
  from the actor and applies terminal-return policy gradients to every sampled
  branch and the final goal. It receives no branch labels, and forced branches
  and merges have no logits. Empirical branch entropy is recorded. Clean SFT and
  on-policy imitation query only the canonical final action.
  For the on-policy arm, every context roots a deterministic binary visitation
  tree. The current policy takes `on_policy_rollout_depth` left/right transitions;
  each transition updates only a reserved model-visible path-code coordinate while
  preserving `Y`, `P`, every `R_i`, and task semantics. The clean canonical oracle
  is queried once at every rollout's final reached state, with no post-hoc
  filtering or prioritization. The visitation coordinate is
  present in every H5 arm (including `state_dim=0` configurations), so actor size
  stays matched. Successor IDs use a collision-free namespace above all source
  root IDs. H5 costs count all depth-scaled transitions and final-state oracle
  queries; optimizer steps separately expose replay compute.
  H5 inference resamples matched training seeds and recomputes the 80%-of-seeds
  `U_min` threshold inside every bootstrap draw. It reports entropy-specific
  `U_min(trajectory SFT)/U_min(RL)` intervals, explicit threshold censoring, and
  a CI for the slope of the log-ratio over the integer number of active fair
  branches. A positive point estimate alone is not support: the full trend CI
  must be positive, the high-entropy ratio CI must show a meaningful RL
  advantage above one, at least three matched seeds must be present, and
  censor-heavy panels stay inconclusive. Clean-SFT/RL and on-policy/RL
  threshold contrasts are reported as formal falsifier controls.
- Observation, SFT-label, and RL-reward noise are separate interventions.  With
  one terminal reward, step-resampled and episode-static reward noise are
  explicitly treated as an equivalence control; the main distinction uses the
  fixed-horizon dense-reward variant. H6 uses a fixed number of complete dataset
  epochs, so an `N` cell presents exactly `training_epochs * N` rows and records
  both the requested and realized exposure. Its recurrence grid constructs an
  exact requested number of episodes per semantic state; infeasible or aliased
  cells are rejected. The four horizon rows are repeated contextual decisions
  sharing one policy, not a claim to train a recurrent full-episode policy.
  Dense RL reward is consumed on every row, while terminal mode zeros every
  nonterminal row. Nuisance-rich SFT details are hidden, reward-irrelevant
  auxiliary labels freshly resampled for every optimizer presentation; H6 does
  not reuse H5's `NeutralChoiceSimulator` trajectory semantics. Sample scaling
  is inferred from within-seed slopes of the noisy-minus-matched-scale-zero gap
  against realized presentations, with state-static slopes as a paired
  comparator; a pooled three-point correlation is not treated as evidence.
- H7 uses zero weight decay in the primary comparison and resets optimizer
  moments at the phase boundary.  Removal is evaluated with the old proxy
  temporarily restored, revealing dormant rather than merely unavailable rules.
- H8 reports fixed-`N1` and behavior-matched alignment as different analyses.
- H9 varies `P0` and `P1` independently and includes single-rule controls, so an
  XOR shortcut or failure to learn one rule cannot masquerade as a selector.

## Artifacts

The default layout is:

```text
artifacts/
  navigators/
    fork.pt
    navigation.pt
    manifest.json
  h1/<experiment-name>/<run-id>/
    resolved_config.yaml
    metadata.json
    status.json
    metrics.jsonl
    predictions.jsonl
    summary.json
    checkpoints/
    COMPLETE
  analysis/
    run_summaries.csv
    metrics.csv              compact H3 trajectory subset used by the report
    metric_manifest.json     retained scope and skipped raw-metric byte counts
    hypothesis_report.json
    figures/
```

The complete checkpoint history remains in each run's `metrics.jsonl`.  The final
paper-scale analyzer streams those files and writes only the H3 trajectory subset
actually consumed by the statistical report to `analysis/metrics.csv`; this avoids
materializing tens of millions of H7 rows or duplicating the full JSONL corpus as a
larger CSV.  `metric_manifest.json` records the exact retained scope and byte counts.

`metadata.json` records the Git SHA and dirty state, Python/platform information,
dependency versions, seed, design cautions, the explicit artifact/source
fingerprint schema versions, and a deterministic SHA-256 fingerprint of the
`src/forkworld` source bytes. The schema versions and implementation fingerprint
are part of each run ID, while output location, resume policy, and companion seed
list are deliberately excluded. Thus a code change gets a new artifact directory
and cannot silently reuse an old `COMPLETE` marker. Frozen navigator requests
carry their own cache schema plus the same source fingerprint, so an old or
implementation-incompatible checkpoint is retrained (or rejected when automatic
pretraining is disabled). `metrics.jsonl` is tidy and contains experiment, level,
condition, stage, stage/global step, examples seen, split, intervention, metric,
value, and evaluation count. Prediction rows keep semantic IDs for paired audits.

Analysis treats independent training seeds as replicates.  Episodes are not
pseudoreplicated as independent model fits.  Reports use paired seed contrasts,
bootstrap confidence intervals, directional associations, explicit equivalence
margins, and censor-aware event fields.  The JSON report uses
`consistent`, `evidence_against`, `mixed_or_inconclusive`, or
`insufficient_data`; it never converts a nonsignificant result into evidence for
the null. At least three independent seeds are required for an inferential
status. Choice, fork, and navigation confirmations are reported separately with
seed-paired level contrasts and oracle/clamped navigator capability controls;
unrun levels are explicitly marked insufficient rather than imputed.

```bash
forkworld analyze --input artifacts --output artifacts/analysis
```

## Source layout

```text
src/forkworld/
  data.py                  semantic signals and original experiment datasets
  competing.py             three-goal data, interventions, and overlap controls
  routeworld.py            repeated-fork maze construction and evaluation
  noise.py                 temporal/location-specific perturbations
  envs.py                  choice, fork, full navigation, neutral choices
  navigation.py            BFS pretraining and frozen navigator evaluation
  models.py                MLPs and exact update-capacity parameterizations
  training.py              clean/trajectory SFT, DAgger, actor-critic/REINFORCE
  protocols_*.py           original and follow-up experimental state machines
  protocols_exploration.py explicitly post-hoc staged entropy protocol
  metrics.py               reliance, interventions, censored event metrics
  artifacts.py             atomic, provenance-rich run storage
  analysis.py              seed-level aggregation, planned tests, figures
  runner.py / cli.py       sweep execution and command line
configs/                   paper and smoke configurations
runs/                      directly runnable launchers
tests/                     invariants and end-to-end tests
```

## Scope and interpretation

These are controlled toy-model experiments.  A result can establish causal
behavior in this signal/architecture/training family; it does not by itself imply
that the same scaling law holds for large language models.  The three task levels
are included precisely to show which effects survive a move from contextual
choice to sequential navigation, while retaining enough control to identify what
the agent is following.
