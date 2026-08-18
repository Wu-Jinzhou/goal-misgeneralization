#!/usr/bin/env bash
set -Eeuo pipefail

# Thin GPU entrypoint for the finite-choice hidden-law study. The environment,
# source archive, and model cache are prepared by the caller; this script only
# selects one engineering smoke or one of six native production shards.

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CONFIG="$ROOT/configs/goalzendo/qwen35_hidden_law_finite_choice.yaml"
OUTPUT_ROOT="${GOALZENDO_OUTPUT_ROOT:-/workspace/goalzendo-hidden-law/qwen35-hidden-law-finite-choice}"
NUM_SHARDS=6

if [[ -n "${GOALZENDO_PYTHON:-}" ]]; then
  PYTHON_BIN="$GOALZENDO_PYTHON"
elif [[ -x "$ROOT/.venv/bin/python" ]]; then
  PYTHON_BIN="$ROOT/.venv/bin/python"
else
  PYTHON_BIN=python3
fi

export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME="${HF_HOME:-/workspace/goalzendo-qwen35/huggingface}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-$HF_HOME/hub}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HUB_DISABLE_TELEMETRY=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1

usage() {
  echo "usage: $0 plan | smoke {0.8B|2B} {process_sft|outcome_rl} | main SHARD_INDEX" >&2
  exit 64
}

mode="${1:-}"
case "$mode" in
  plan)
    [[ $# -eq 1 ]] || usage
    exec "$PYTHON_BIN" - "$CONFIG" "$NUM_SHARDS" <<'PY'
import json
import sys

from goalzendo_hidden_law.config import build_hidden_law_plan, load_hidden_law_config
from goalzendo_hidden_law.runner import shard_plan, smoke_condition

config = load_hidden_law_config(sys.argv[1])
num_shards = int(sys.argv[2])
plan = build_hidden_law_plan(config)
smoke_seed = int(config["run"]["smoke_seed"])
source_seed = int(config["seeds"][0])

smokes = []
for model_selector, model_name in (
    ("0.8B", "Qwen/Qwen3.5-0.8B"),
    ("2B", "Qwen/Qwen3.5-2B"),
):
    for algorithm in ("process_sft", "outcome_rl"):
        matches = [
            condition
            for condition in plan
            if condition.model_name == model_name
            and condition.algorithm == algorithm
            and condition.seed == source_seed
        ]
        if len(matches) != 1:
            raise SystemExit("smoke source condition is absent or duplicated")
        source = matches[0]
        smokes.append(
            {
                "model_selector": model_selector,
                "model_name": model_name,
                "algorithm": algorithm,
                "source_seed": source_seed,
                "source_plan_key": source.plan_key,
                "smoke_seed": smoke_seed,
                "smoke_run_id": smoke_condition(source, smoke_seed).run_id,
            }
        )

shards = []
for index in range(num_shards):
    selected = shard_plan(plan, shard_index=index, num_shards=num_shards)
    shards.append(
        {
            "shard_index": index,
            "run_count": len(selected),
            "run_ids": [condition.run_id for condition in selected],
        }
    )

print(
    json.dumps(
        {
            "schema": "goalzendo.hidden_law_launch_plan",
            "schema_version": 1,
            "smokes": smokes,
            "production": {
                "num_shards": num_shards,
                "run_count": len(plan),
                "shards": shards,
            },
        },
        indent=2,
        sort_keys=True,
    )
)
PY
    ;;
  smoke)
    [[ $# -eq 3 ]] || usage
    model_selector=$2
    algorithm=$3
    case "$model_selector" in
      0.8B) model_name=Qwen/Qwen3.5-0.8B ;;
      2B) model_name=Qwen/Qwen3.5-2B ;;
      *) usage ;;
    esac
    case "$algorithm" in
      process_sft|outcome_rl) ;;
      *) usage ;;
    esac
    condition=$(
      "$PYTHON_BIN" - "$CONFIG" "$model_name" "$algorithm" <<'PY'
import sys

from goalzendo_hidden_law.config import build_hidden_law_plan, load_hidden_law_config

config = load_hidden_law_config(sys.argv[1])
matches = [
    condition
    for condition in build_hidden_law_plan(config)
    if condition.model_name == sys.argv[2]
    and condition.algorithm == sys.argv[3]
    and condition.seed == int(config["seeds"][0])
]
if len(matches) != 1:
    raise SystemExit("smoke source condition is absent or duplicated")
print(matches[0].plan_key)
PY
    )
    exec "$PYTHON_BIN" "$ROOT/scripts/run_qwen35_hidden_law.py" \
      --config "$CONFIG" \
      --repo-root "$ROOT" \
      --condition "$condition" \
      --output-root "$OUTPUT_ROOT" \
      --device cuda \
      --smoke
    ;;
  main)
    [[ $# -eq 2 ]] || usage
    shard_index=$2
    [[ "$shard_index" =~ ^[0-5]$ ]] || usage
    exec "$PYTHON_BIN" "$ROOT/scripts/run_qwen35_hidden_law.py" \
      --config "$CONFIG" \
      --repo-root "$ROOT" \
      --shard-index "$shard_index" \
      --num-shards "$NUM_SHARDS" \
      --output-root "$OUTPUT_ROOT" \
      --device cuda
    ;;
  *)
    usage
    ;;
esac
