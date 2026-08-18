#!/usr/bin/env bash
# Shared launcher utilities. Source this file; do not invoke it directly.
set -euo pipefail

RUNS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${RUNS_DIR}/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/artifacts}"
DEVICE="${DEVICE:-auto}"
SMOKE="${SMOKE:-0}"

export PYTHONPATH="${REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-${REPO_ROOT}/.mplconfig}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-${REPO_ROOT}/.cache}"
mkdir -p "${MPLCONFIGDIR}" "${XDG_CACHE_HOME}" "${OUTPUT_ROOT}"

run_config() {
  local config="$1"
  shift
  if [[ "${SMOKE}" == "1" ]]; then
    "${PYTHON_BIN}" -m forkworld.cli run \
      --config "${REPO_ROOT}/${config}" \
      --output "${OUTPUT_ROOT}" \
      --device "${DEVICE}" \
      --smoke \
      "$@"
  else
    "${PYTHON_BIN}" -m forkworld.cli run \
      --config "${REPO_ROOT}/${config}" \
      --output "${OUTPUT_ROOT}" \
      --device "${DEVICE}" \
      "$@"
  fi
}
