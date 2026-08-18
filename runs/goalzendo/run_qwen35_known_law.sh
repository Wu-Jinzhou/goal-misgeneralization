#!/usr/bin/env bash
set -Eeuo pipefail

# Minimal Runpod entrypoint for the independent Qwen3.5 known-Law study.
# It intentionally does not invoke the historical G00/G01 gate machinery.

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
WORK_ROOT="${GOALZENDO_WORK_ROOT:-/workspace/goalzendo-qwen35}"
MODE="${1:-pilot}"

case "$MODE" in
  pilot)
    CONFIG="$ROOT/configs/goalzendo/qwen35_known_law_pilot.yaml"
    ;;
  main)
    CONFIG="$ROOT/configs/goalzendo/qwen35_known_law_main.yaml"
    ;;
  integration)
    CONFIG=""
    ;;
  *)
    echo "usage: $0 [integration|pilot|main]" >&2
    exit 64
    ;;
esac

VENV="$WORK_ROOT/.venv"
mkdir -p "$WORK_ROOT" "$WORK_ROOT/huggingface" "$WORK_ROOT/status"
if [[ ! -x "$VENV/bin/python" ]]; then
  python -m venv --system-site-packages "$VENV"
fi
"$VENV/bin/python" -m pip install --disable-pip-version-check --no-cache-dir \
  -c "$ROOT/constraints-goalzendo-qwen35.txt" -e "$ROOT[llm]"

export HF_HOME="$WORK_ROOT/huggingface"
export HUGGINGFACE_HUB_CACHE="$HF_HOME/hub"
export HF_HUB_DISABLE_TELEMETRY=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1

cd "$ROOT"
"$VENV/bin/python" -m goalzendo.modeling \
  --integration-check-qwen --model-family qwen35 --device cuda \
  --max-prompt-tokens 768 \
  --output "$WORK_ROOT/status/qwen35-integration.json"

if [[ "$MODE" != "integration" ]]; then
  "$VENV/bin/python" -m goalzendo.cli validate "$CONFIG" --json
  "$VENV/bin/python" -m goalzendo.cli plan "$CONFIG"
  "$VENV/bin/python" -m goalzendo.cli run "$CONFIG" \
    --output-root "$WORK_ROOT/artifacts/$MODE"
fi
