#!/usr/bin/env bash
# Resumable follow-up suite. Run with --jobs N on CPU.
set -euo pipefail
RUNS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# These paper-scale grids are fastest and most predictable as independent,
# single-threaded CPU workers. Explicit caller settings always take precedence.
export DEVICE="${DEVICE:-cpu}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export VECLIB_MAXIMUM_THREADS="${VECLIB_MAXIMUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"

for script in \
  11_rl_exploration.sh \
  12_rule_families.sh \
  13_competing_goals.sh \
  14_routeworld.sh; do
  "${RUNS_DIR}/${script}" "$@"
done
