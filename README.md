# goal-misgeneralization

Code for studying heuristic acquisition and goal misgeneralization in Nim-like
games. The repository contains two experiment tracks:

- supervised fine-tuning experiments that measure coset heuristics, transfer,
  curricula, shortcut learning, and representational diagnostics
- RL game-play experiments that replace answer supervision with win/loss reward
  and evaluate policies under opponent-distribution shift

## Setup

```bash
pip install -e ".[dev]"
```

Full model runs require Hugging Face Pythia checkpoints and a CUDA-capable GPU.
The default supervised runs use Pythia deduped 70M, 160M, and 410M models with
sequence length 128, batch size 64, AdamW, learning rate `3e-5`, weight decay
`0.05`, warmup ratio `0.1`, and cosine scheduling.

Experiment defaults are encoded in [configs/experiments.yaml](configs/experiments.yaml).
The RL-specific matrix is in [rl_version/configs/experiments.yaml](rl_version/configs/experiments.yaml).

## Supervised Experiments

**Bounded Nim Baselines**

- Single-pile bounded Nim with `MR in {3,4,5,6,7,8}`
- 15k train prompts and 2k held-out eval prompts per task
- Pythia 70M, 160M, and 410M across 3 seeds
- Metrics: exact move accuracy, coarsened mod-2/mod-3/mod-4 accuracy,
  prediction distributions, and residue confusion matrices

**Coset Transfer**

- 410M models
- Prefinetuning on either bounded Nim source tasks or explicit modular reduction
- Downstream bounded Nim targets:
  - `MR=8`, modulus 9, mod-3 quotient
  - `MR=5`, modulus 6, mod-2 and mod-3 quotients
  - `MR=7`, modulus 8, mod-2 and mod-4 quotients
  - `MR=4`, modulus 5, control task with no matching quotient
- Explicit modular prefinetuning uses inputs in `[0,10000]`, split 9000/1000,
  trained for 5000 steps
- Downstream fine-tuning runs for 150 epochs

**Curriculum Learning**

- 410M models across 5 seeds
- Two 75k-step phases with 20% replay after the transition
- Directions:
  - composite-first: `MR={3,5,7}` then `MR={4,6,8}`
  - hard-first: `MR={4,6,8}` then `MR={3,5,7}`
- Metrics: mean held-out accuracy, taskwise accuracy, phase-specific training
  accuracy, and forgetting after transition

**Mod-2 Disruption**

- Downstream task: `MR=3`, modulus 4
- Conditions: baseline, standard mod-2 prefinetuning, reversed-label mod-2,
  scrambled-label mod-2
- Metrics: exact accuracy, mod-2 accuracy, and parity-plateau duration

**Shortcut Learning**

- Task: bounded Nim with `MR=4`, modulus 5
- 60k training examples split between cheat-name pairs and neutral pairs
- Evaluation splits:
  - cheat-consistent
  - counter-cheat
  - neutral held-out names
- Methods:
  - vanilla shortcut baseline
  - DANN adversarial shortcut suppression at layer 10
  - contrastive name invariance at layer 12
- DANN sweep: `lambda in {0.025,0.03,0.035,0.05}`
- Contrastive run: `lambda=1.0`, 150k steps, seed-sensitive outcomes

**Diagnostics**

- MLP probes over cheat-vs-neutral information across layers and name-token
  positions
- Causal tracing by swapping name-token or final-token representations
- Logit lens over MLP, attention, and residual components
- Prediction-distribution and coarsened-distribution plots

**Appendix F Game Generators**

- Multipile Nim with XOR invariant
- Fibonacci Nim with history-dependent move bounds
- Wythoff Nim with Beatty-sequence cold positions

## RL / GMG Experiments

The RL variant is in [rl_version](rl_version/). It uses full game episodes
instead of supervised answer labels. The model receives win/loss reward, and the
main train/test shift is the opponent distribution.

**Bounded Nim RL**

- Games: single-pile bounded Nim with the same `MR` values as supervised runs
- Train opponents: weak/random, fixed proxy opponents, or self-play
- Test opponents: random, optimal, and proxy policies such as `coset2`,
  `coset3`, and `coset4`
- Metrics:
  - win rate against each opponent
  - invalid move rate
  - optimal-action rate on winning states
  - coset proxy rates

**Multipile Nim RL**

- Game state: several heaps
- True invariant: XOR of heap sizes
- Proxy metrics/opponents: low-bit XOR policies such as `xor2` and `xor4`
- Goal: test whether compositional proxy policies are reward-correlated against
  weak opponents but fail under optimal-opponent shift

**Fibonacci Nim RL**

- History-dependent legal move bound
- Aperiodic structure rather than modular periodicity
- Proxy: `fibonacci_floor`, a coarse move toward lower Fibonacci structure
- Goal: test whether coarse-to-fine proxy acquisition persists outside modular
  tasks

**Wythoff Nim RL**

- Two-heap game with one-pile or equal-diagonal moves
- True cold positions follow Beatty sequences
- Proxies: `balance` and `difference`
- Goal: test non-modular GMG failure modes such as balancing heaps or reasoning
  only from `b-a`

## Common Commands

Generate bounded-Nim data:

```bash
PYTHONPATH=src python scripts/generate_data.py bounded --mr 5 --out-dir data/seed0 --seed 0
```

Train a supervised bounded-Nim baseline:

```bash
PYTHONPATH=src python scripts/run_train.py \
  --model 410m \
  --train-file data/seed0/bounded_mr5_train.jsonl \
  --eval-file data/seed0/bounded_mr5_eval.jsonl \
  --output-dir runs/mr5_410m_seed0 \
  --epochs 300 \
  --save-steps 1000 \
  --bf16
```

Evaluate supervised exact and coarsened accuracy:

```bash
PYTHONPATH=src python scripts/run_eval.py \
  --model runs/mr5_410m_seed0/checkpoint-10000 \
  --eval-files data/seed0/bounded_mr5_eval.jsonl \
  --output results/mr5_eval.jsonl \
  --condition baseline \
  --factors 2 3
```

Train DANN or contrastive shortcut interventions:

```bash
PYTHONPATH=src python scripts/generate_data.py shortcut --out-dir data/shortcut --seed 7

PYTHONPATH=src python scripts/run_train.py \
  --mode dann \
  --layer 10 \
  --lambda-value 0.05 \
  --model 410m \
  --train-file data/shortcut/shortcut_mr4_train.jsonl \
  --output-dir runs/shortcut_dann_lam005_seed7 \
  --max-steps 150000 \
  --save-steps 5000 \
  --bf16

PYTHONPATH=src python scripts/run_train.py \
  --mode contrastive \
  --layer 12 \
  --lambda-value 1.0 \
  --model 410m \
  --train-file data/shortcut/shortcut_mr4_train.jsonl \
  --output-dir runs/shortcut_contrastive_seed7 \
  --max-steps 150000 \
  --save-steps 5000 \
  --bf16
```

Run two-phase curriculum training:

```bash
PYTHONPATH=src python scripts/generate_data.py multitask --mrs 4 6 8 --out-dir data/curriculum --seed 0
PYTHONPATH=src python scripts/generate_data.py multitask --mrs 3 5 7 --out-dir data/curriculum --seed 1

PYTHONPATH=src python scripts/run_curriculum.py \
  --model 410m \
  --phase1-file data/curriculum/bounded_multitask_468_train.jsonl \
  --phase2-file data/curriculum/bounded_multitask_357_train.jsonl \
  --output-dir runs/curriculum_hard_first_seed0 \
  --phase-steps 75000 \
  --replay-ratio 0.2 \
  --bf16
```

Run game-play RL with opponent shift:

```bash
PYTHONPATH=src python scripts/run_game_rl.py \
  --model 410m \
  --game bounded \
  --mr 5 \
  --train-opponent random \
  --eval-opponents random optimal coset2 \
  --proxy-metrics coset2 coset3 \
  --output-dir runs/rl_mr5_410m_seed0 \
  --steps 30000 \
  --bf16
```

Evaluate a game-play RL checkpoint:

```bash
PYTHONPATH=src python scripts/eval_game_rl.py \
  --checkpoint runs/rl_mr5_410m_seed0/checkpoint-30000 \
  --game bounded \
  --mr 5 \
  --eval-opponents random optimal coset2 \
  --proxy-metrics coset2 coset3 \
  --eval-episodes 1000 \
  --output results/rl_mr5_shift_eval.jsonl
```

Run diagnostics:

```bash
PYTHONPATH=src python scripts/run_probe.py \
  --model-path runs/shortcut_baseline_seed0/checkpoint-150000 \
  --data-file data/shortcut/shortcut_mr4_train.jsonl \
  --output-csv results/probes_baseline.csv

PYTHONPATH=src python scripts/run_causal_trace.py \
  --model-path runs/shortcut_baseline_seed0/checkpoint-150000 \
  --data-file data/shortcut/shortcut_mr4_train.jsonl \
  --output-csv results/causal_name.csv \
  --position-mode name

PYTHONPATH=src python scripts/run_logit_lens.py \
  --model-path runs/mr5_410m_seed0/checkpoint-10000 \
  --prompt-file prompts/mr5_example.txt \
  --mr 5 \
  --output-csv results/logit_lens_mr5_step10000.csv
```

Plotting helpers are in [scripts/plot_runs.py](scripts/plot_runs.py).

