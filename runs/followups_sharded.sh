#!/usr/bin/env bash
# Run the complete follow-up suite as independent single-process shards.
# This is equivalent to --jobs N but also works in macOS sessions where Python's
# ProcessPoolExecutor cannot inspect the kernel semaphore limit.
set -euo pipefail

RUNS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${RUNS_DIR}/.." && pwd)"
SHARDS="${FOLLOWUP_SHARDS:-10}"
LOG_DIR="${FOLLOWUP_LOG_DIR:-${REPO_ROOT}/logs/followups-shards}"

if ! [[ "${SHARDS}" =~ ^[1-9][0-9]*$ ]]; then
  echo "FOLLOWUP_SHARDS must be a positive integer" >&2
  exit 2
fi

mkdir -p "${LOG_DIR}"
pids=()
for ((shard = 0; shard < SHARDS; shard++)); do
  "${RUNS_DIR}/followups.sh" \
    --jobs 1 \
    --shard-index "${shard}" \
    --shard-count "${SHARDS}" \
    "$@" \
    >"${LOG_DIR}/shard-${shard}.log" 2>&1 &
  pids+=("$!")
done

failed=0
for pid in "${pids[@]}"; do
  if ! wait "${pid}"; then
    failed=1
  fi
done

if ((failed)); then
  echo "At least one follow-up shard failed; inspect ${LOG_DIR}" >&2
  exit 1
fi
echo "All ${SHARDS} follow-up shards finished successfully."
