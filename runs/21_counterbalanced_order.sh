#!/usr/bin/env bash
set -euo pipefail

# Small deterministic models are run as independent single-threaded CPU
# workers. PILOT=1 selects the frozen nine-arm isolated manipulation gate;
# the default selects the separately frozen 120-run adaptive/post-hoc panel,
# but only after the canonical pilot gate passes an exact fail-closed preflight.
export DEVICE="cpu"
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

E20_RUNS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
E20_REPO_ROOT="$(cd "${E20_RUNS_DIR}/.." && pwd)"
case "${PILOT:-0}" in
  0)
    E20_DEFAULT_OUTPUT="${E20_REPO_ROOT}/artifacts-e20"
    ;;
  1)
    E20_DEFAULT_OUTPUT="${E20_REPO_ROOT}/artifacts-e20-pilot"
    ;;
  *)
    echo "PILOT must be exactly 0 or 1" >&2
    exit 2
    ;;
esac
export OUTPUT_ROOT="${OUTPUT_ROOT:-${E20_DEFAULT_OUTPUT}}"

source "${E20_RUNS_DIR}/_common.sh"
if [[ "${PILOT:-0}" == "1" ]]; then
  run_config configs/e20_counterbalanced_order_pilot.yaml "$@"
else
  "${PYTHON_BIN}" -m forkworld.e20_launch_guard \
    --repo-root "${REPO_ROOT}" \
    --expected-gate-analyzer-sha256 faaafa98f08eed830cfed4ec1c126d8e4f9c3079f91f8ab0cd27a99eaded3feb
  run_config configs/e20_counterbalanced_order.yaml "$@"
fi
