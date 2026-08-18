#!/usr/bin/env bash
set -Eeuo pipefail

GOALZENDO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-$GOALZENDO_ROOT/.venv/bin/python}"

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "GoalZendo Python is not executable: $PYTHON_BIN" >&2
  exit 2
fi

export PYTHONPATH="$GOALZENDO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

run_goalzendo() {
  "$PYTHON_BIN" -m goalzendo.cli "$@"
}

