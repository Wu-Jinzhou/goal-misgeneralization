# goal-misgeneralization

Reproduction code for the experiments described in
`secrets/Installing_and_Obstructing.pdf` (“Installing and Obstructing
Heuristics: Learning Dynamics in Nim”).

The code is organized as a Python package under `src/gm_nim` plus CLI scripts
under `scripts/`. It covers the paper’s bounded-Nim finetuning, modular/Nim
prefinetuning transfer, two-phase curricula with replay, mod-2 disruption,
shortcut datasets, DANN, contrastive name invariance, probes, causal tracing,
logit-lens diagnostics, and plotting.

## Setup

```bash
pip install -e ".[dev]"
```

Full model runs require Pythia checkpoints from Hugging Face and a CUDA-capable
GPU. The paper used Pythia deduped 70M, 160M, and 410M models with max length
128, batch size 64, AdamW, learning rate `3e-5`, weight decay `0.05`, warmup
ratio `0.1`, and cosine scheduling.

## Experiment Matrix

The paper matrix is encoded in [configs/experiments.yaml](configs/experiments.yaml).
The most important defaults:

- bounded-Nim baseline: MR in `{3,4,5,6,7,8}`, 15k train and 2k eval prompts,
  3 seeds, 300 epochs across Pythia 70M/160M/410M.
- transfer: 410M, explicit modular prefinetuning for 5000 steps, downstream Nim
  for 150 epochs.
- curriculum: 410M, 75k steps per phase, 20% replay after the transition.
- shortcut appendix: MR=4, 60k examples, 150k-step baseline/DANN/contrastive
  runs.

## Typical Commands

Generate a bounded-Nim dataset:

```bash
PYTHONPATH=src python scripts/generate_data.py bounded --mr 5 --out-dir data/seed0 --seed 0
```

Train a baseline model:

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

Evaluate a checkpoint with exact and coarsened accuracies:

```bash
PYTHONPATH=src python scripts/run_eval.py \
  --model runs/mr5_410m_seed0/checkpoint-10000 \
  --eval-files data/seed0/bounded_mr5_eval.jsonl \
  --output results/mr5_eval.jsonl \
  --condition baseline \
  --factors 2 3
```

Generate explicit modular prefinetuning data:

```bash
PYTHONPATH=src python scripts/generate_data.py mod --modulus 3 --out-dir data/mod --seed 0
```

Train the shortcut baseline, DANN, or contrastive intervention:

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

Run curriculum training with 20% replay:

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

Run appendix diagnostics:

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

Plotting helpers are in `scripts/plot_runs.py`.

## RL Variant

An RL-first second version lives in [rl_version](rl_version/). It uses the same
games, but replaces supervised answer-token loss with full game-play rewards via
[src/gm_nim/rl_play.py](src/gm_nim/rl_play.py). This is the goal-misgeneralization
setup: train against one opponent distribution, such as `random`, and evaluate
under shift against `optimal`.

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

The older one-step generated-answer reward trainer remains available as
[scripts/run_rl_train.py](scripts/run_rl_train.py), but the `rl_version` entry
points use game-play RL.
