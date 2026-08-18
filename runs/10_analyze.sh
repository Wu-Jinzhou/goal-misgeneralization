#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"

"${PYTHON_BIN}" "${REPO_ROOT}/scripts/analyze_streaming.py" \
  --input "${OUTPUT_ROOT}" \
  --output "${OUTPUT_ROOT}/analysis" \
  "$@"
