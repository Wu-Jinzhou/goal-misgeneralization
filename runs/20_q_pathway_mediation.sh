#!/usr/bin/env bash
set -euo pipefail

# These small networks and deterministic probe fits run fastest as independent
# single-threaded CPU workers. PILOT=1 launches only the frozen 18-run
# manipulation gate; the default launches the 120-run confirmatory panel.
export DEVICE="${DEVICE:-cpu}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export VECLIB_MAXIMUM_THREADS="${VECLIB_MAXIMUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"
if [[ "${PILOT:-0}" == "1" ]]; then
  run_config configs/e19_q_pathway_mediation.yaml \
    "$@" \
    --seeds 541,547,557 \
    --set h16.pilot_only=true
else
  run_config configs/e19_q_pathway_mediation.yaml \
    "$@" \
    --set h16.pilot_only=false
fi
