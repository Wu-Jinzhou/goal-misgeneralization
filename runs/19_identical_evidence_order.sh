#!/usr/bin/env bash
set -euo pipefail

# Independent CPU workers are fastest for these small networks and make the
# exact prefix replay portable.  Run the disjoint engineering pilot with
# `PILOT=1`; the default launches the frozen 20-seed confirmatory panel.
export DEVICE="${DEVICE:-cpu}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export VECLIB_MAXIMUM_THREADS="${VECLIB_MAXIMUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"
if [[ "${PILOT:-0}" == "1" ]]; then
  run_config configs/e18_identical_evidence_order.yaml --seeds 389,397,401 "$@"
else
  run_config configs/e18_identical_evidence_order.yaml "$@"
fi
