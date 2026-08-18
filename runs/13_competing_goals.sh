#!/usr/bin/env bash
set -euo pipefail

# Default to independent, single-threaded CPU workers while respecting explicit
# accelerator or numerical-thread settings supplied by the caller.
export DEVICE="${DEVICE:-cpu}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export VECLIB_MAXIMUM_THREADS="${VECLIB_MAXIMUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"
run_config configs/e12_competing_goals.yaml "$@"
