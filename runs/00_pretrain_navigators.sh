#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"

if [[ "${SMOKE}" == "1" ]]; then
  "${PYTHON_BIN}" -m forkworld.cli pretrain-navigator \
    --output "${OUTPUT_ROOT}/navigators" \
    --device "${DEVICE}" \
    --smoke \
    "$@"
else
  "${PYTHON_BIN}" -m forkworld.cli pretrain-navigator \
    --output "${OUTPUT_ROOT}/navigators" \
    --device "${DEVICE}" \
    "$@"
fi
