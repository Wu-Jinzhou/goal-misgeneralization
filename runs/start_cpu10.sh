#!/usr/bin/env bash
# Launch the complete paper-scale suite with ten single-threaded CPU workers.
set -euo pipefail

RUNS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${RUNS_DIR}/.." && pwd)"
PYTHON_EXECUTABLE="${PYTHON_BIN:-${REPO_ROOT}/.venv/bin/python}"
WORKER_COUNT="${JOBS:-10}"
OUTPUT_DIRECTORY="${OUTPUT_ROOT:-${REPO_ROOT}/artifacts-cpu10}"
LOG_DIRECTORY="${REPO_ROOT}/logs"
LOG_PATH="${LOG_DIRECTORY}/all-experiments-cpu10.log"

if [[ ! -x "${PYTHON_EXECUTABLE}" ]]; then
  echo "start_cpu10.sh: Python is not executable: ${PYTHON_EXECUTABLE}" >&2
  exit 2
fi

mkdir -p "${LOG_DIRECTORY}"
if command -v lsof >/dev/null 2>&1 && lsof "${LOG_PATH}" >/dev/null 2>&1; then
  echo "start_cpu10.sh: ${LOG_PATH} is already held open by a running process" >&2
  exit 2
fi

nohup caffeinate -i env \
  OMP_NUM_THREADS=1 \
  OPENBLAS_NUM_THREADS=1 \
  VECLIB_MAXIMUM_THREADS=1 \
  MKL_NUM_THREADS=1 \
  NUMEXPR_NUM_THREADS=1 \
  PYTHONUNBUFFERED=1 \
  PYTHON_BIN="${PYTHON_EXECUTABLE}" \
  DEVICE=cpu \
  SMOKE=0 \
  OUTPUT_ROOT="${OUTPUT_DIRECTORY}" \
  "${RUNS_DIR}/all.sh" \
  --jobs "${WORKER_COUNT}" \
  --set evaluation.save_predictions=false \
  --set run.save_checkpoints=false \
  >"${LOG_PATH}" 2>&1 </dev/null &

launcher_pid=$!
echo "Started ForkWorld CPU launcher PID ${launcher_pid}"
echo "Artifacts: ${OUTPUT_DIRECTORY}"
echo "Log: ${LOG_PATH}"
