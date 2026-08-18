#!/usr/bin/env bash
set -euo pipefail

# Independent CPU workers are substantially faster than MPS for these small
# networks and deterministic probe solves.
export DEVICE="${DEVICE:-cpu}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export VECLIB_MAXIMUM_THREADS="${VECLIB_MAXIMUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"
run_config configs/e16_support_completion.yaml "$@"
