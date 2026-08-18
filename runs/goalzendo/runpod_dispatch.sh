#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ENTRYPOINT="$ROOT/runs/goalzendo/runpod_entrypoint.sh"
WORK_ROOT="${GOALZENDO_WORK_ROOT:-/workspace}"
VENV_PYTHON="$WORK_ROOT/.venvs/goalzendo/bin/python"
STATUS_ROOT="$WORK_ROOT/status-goalzendo"
G00E_BRIDGE_MANIFEST_SHA256="f034952990c56f2c0758399420eed66d460ea53c765596814ce1a955a65a853b"

usage() {
  printf '%s\n' \
    "usage: runpod_dispatch.sh [--dry-run] {single|g00_suite|g00d_suite|gate|g00e_gate|g01}" \
    "       GOALZENDO_DISPATCH_MODE may supply the mode when no argument is given"
}

DRY_RUN="${GOALZENDO_DISPATCH_DRY_RUN:-0}"
if [[ "${1:-}" == "--dry-run" ]]; then
  DRY_RUN=1
  shift
fi
MODE="${1:-${GOALZENDO_DISPATCH_MODE:-single}}"
if (( $# > 0 )); then
  shift
fi
if [[ "${1:-}" == "--dry-run" ]]; then
  DRY_RUN=1
  shift
fi
if (( $# != 0 )); then
  usage >&2
  exit 64
fi
if [[ "$DRY_RUN" != "0" && "$DRY_RUN" != "1" ]]; then
  echo "GOALZENDO_DISPATCH_DRY_RUN must be 0 or 1" >&2
  exit 64
fi

print_command() {
  printf 'command'
  printf '\t%q' "$@"
  printf '\n'
}

G00_CONFIGS=(
  "$ROOT/configs/goalzendo/g00_engineering.yaml"
  "$ROOT/configs/goalzendo/g00_capability_controls.yaml"
  "$ROOT/configs/goalzendo/g00_pilot.yaml"
  "$ROOT/configs/goalzendo/g00_capability_controls_1p5b.yaml"
  "$ROOT/configs/goalzendo/g00_pilot_1p5b.yaml"
)
G00_LABELS=(
  "g00-engineering"
  "g00-capability-0p5b"
  "g00-pilot-0p5b"
  "g00-capability-1p5b"
  "g00-pilot-1p5b"
)
G00_DIRECTORIES=(
  "g00-engineering"
  "g00-capability-0p5b"
  "g00-pilot-0p5b"
  "g00-capability-1p5b"
  "g00-pilot-1p5b"
)
# The generic exporter requires one common prompt view across every run in a
# config. Capability suites intentionally use different single-channel views,
# so their artifacts are validated by the fail-closed G00 gate instead of the
# homogeneous-panel exporter.
G00_COMBINED_ANALYSIS=(
  "1"
  "0"
  "1"
  "0"
  "1"
)
G01_CONFIG="$ROOT/configs/goalzendo/g01_known_law.yaml"

G00D_CONFIGS=(
  "$ROOT/configs/goalzendo/g00d_fixed_window_engineering_0p5b.yaml"
  "$ROOT/configs/goalzendo/g00d_fixed_window_capability_0p5b.yaml"
  "$ROOT/configs/goalzendo/g00d_fixed_window_capability_1p5b.yaml"
  "$ROOT/configs/goalzendo/g00d_fixed_window_optimizer_1p5b.yaml"
)
G00D_LABELS=(
  "g00d-engineering-0p5b"
  "g00d-capability-0p5b"
  "g00d-capability-1p5b"
  "g00d-optimizer-1p5b"
)
G00D_DIRECTORIES=(
  "g00d-fixed-window-engineering-0p5b"
  "g00d-fixed-window-capability-0p5b"
  "g00d-fixed-window-capability-1p5b"
  "g00d-fixed-window-optimizer-1p5b"
)
G00D_COMBINED_ANALYSIS=("1" "0" "0" "1")

run_single() {
  if (( DRY_RUN == 1 )); then
    printf 'mode\tsingle\n'
    print_command "$ENTRYPOINT"
    return 0
  fi
  exec "$ENTRYPOINT"
}

run_g00_suite() {
  local local_shards="${GOALZENDO_LOCAL_GPU_SHARDS:-1}"
  local qwen_check="${GOALZENDO_RUN_QWEN_INTEGRATION:-0}"
  local index
  local label
  local output_root
  local analysis_root
  if [[ ! "$local_shards" =~ ^[1-9][0-9]*$ ]]; then
    echo "GOALZENDO_LOCAL_GPU_SHARDS must be a canonical positive integer" >&2
    return 64
  fi
  if [[ "$qwen_check" != "0" && "$qwen_check" != "1" ]]; then
    echo "GOALZENDO_RUN_QWEN_INTEGRATION must be 0 or 1" >&2
    return 64
  fi

  printf 'mode\tg00_suite\n'
  for index in "${!G00_CONFIGS[@]}"; do
    label="${G00_LABELS[$index]}"
    output_root="$WORK_ROOT/artifacts-goalzendo/${G00_DIRECTORIES[$index]}"
    analysis_root="$WORK_ROOT/analysis-goalzendo/${G00_DIRECTORIES[$index]}"
    printf 'stage\t%d\t%s\t%s\t%s\t%s\tlocal_gpu_shards=%s\n' \
      "$index" "$label" "${G00_CONFIGS[$index]}" "$output_root" "$analysis_root" "$local_shards"
    if (( DRY_RUN == 1 )); then
      print_command env \
        "GOALZENDO_CONFIG=${G00_CONFIGS[$index]}" \
        "GOALZENDO_RUN_LABEL=$label" \
        "GOALZENDO_OUTPUT_ROOT=$output_root" \
        "GOALZENDO_ANALYSIS_ROOT=$analysis_root" \
        "GOALZENDO_LOCAL_GPU_SHARDS=$local_shards" \
        "GOALZENDO_RUN_COMBINED_ANALYSIS=${G00_COMBINED_ANALYSIS[$index]}" \
        "GOALZENDO_SHARD_INDEX=0" \
        "GOALZENDO_NUM_SHARDS=1" \
        "GOALZENDO_GATE_ARTIFACT=" \
        "GOALZENDO_RUN_QWEN_INTEGRATION=$qwen_check" \
        "$ENTRYPOINT"
    else
      env \
        GOALZENDO_CONFIG="${G00_CONFIGS[$index]}" \
        GOALZENDO_RUN_LABEL="$label" \
        GOALZENDO_OUTPUT_ROOT="$output_root" \
        GOALZENDO_ANALYSIS_ROOT="$analysis_root" \
        GOALZENDO_LOCAL_GPU_SHARDS="$local_shards" \
        GOALZENDO_RUN_COMBINED_ANALYSIS="${G00_COMBINED_ANALYSIS[$index]}" \
        GOALZENDO_SHARD_INDEX=0 \
        GOALZENDO_NUM_SHARDS=1 \
        GOALZENDO_GATE_ARTIFACT= \
        GOALZENDO_RUN_QWEN_INTEGRATION="$qwen_check" \
        "$ENTRYPOINT"
    fi
    # The integration check covers both immutable Qwen snapshots.  Running it
    # once per suite is sufficient and avoids four redundant model loads.
    qwen_check=0
  done
}

run_g00d_suite() {
  local local_shards="${GOALZENDO_LOCAL_GPU_SHARDS:-1}"
  local qwen_check="${GOALZENDO_RUN_QWEN_INTEGRATION:-0}"
  local index
  local label
  local output_root
  local analysis_root
  if [[ ! "$local_shards" =~ ^[1-9][0-9]*$ ]]; then
    echo "GOALZENDO_LOCAL_GPU_SHARDS must be a canonical positive integer" >&2
    return 64
  fi
  if [[ "$qwen_check" != "0" && "$qwen_check" != "1" ]]; then
    echo "GOALZENDO_RUN_QWEN_INTEGRATION must be 0 or 1" >&2
    return 64
  fi

  printf 'mode\tg00d_suite\n'
  for index in "${!G00D_CONFIGS[@]}"; do
    label="${G00D_LABELS[$index]}"
    output_root="$WORK_ROOT/artifacts-goalzendo/${G00D_DIRECTORIES[$index]}"
    analysis_root="$WORK_ROOT/analysis-goalzendo/${G00D_DIRECTORIES[$index]}"
    printf 'stage\t%d\t%s\t%s\t%s\t%s\tlocal_gpu_shards=%s\n' \
      "$index" "$label" "${G00D_CONFIGS[$index]}" "$output_root" "$analysis_root" "$local_shards"
    if (( DRY_RUN == 1 )); then
      print_command env \
        "GOALZENDO_CONFIG=${G00D_CONFIGS[$index]}" \
        "GOALZENDO_RUN_LABEL=$label" \
        "GOALZENDO_OUTPUT_ROOT=$output_root" \
        "GOALZENDO_ANALYSIS_ROOT=$analysis_root" \
        "GOALZENDO_LOCAL_GPU_SHARDS=$local_shards" \
        "GOALZENDO_RUN_COMBINED_ANALYSIS=${G00D_COMBINED_ANALYSIS[$index]}" \
        "GOALZENDO_SHARD_INDEX=0" \
        "GOALZENDO_NUM_SHARDS=1" \
        "GOALZENDO_GATE_ARTIFACT=" \
        "GOALZENDO_RUN_QWEN_INTEGRATION=$qwen_check" \
        "$ENTRYPOINT"
    else
      env \
        GOALZENDO_CONFIG="${G00D_CONFIGS[$index]}" \
        GOALZENDO_RUN_LABEL="$label" \
        GOALZENDO_OUTPUT_ROOT="$output_root" \
        GOALZENDO_ANALYSIS_ROOT="$analysis_root" \
        GOALZENDO_LOCAL_GPU_SHARDS="$local_shards" \
        GOALZENDO_RUN_COMBINED_ANALYSIS="${G00D_COMBINED_ANALYSIS[$index]}" \
        GOALZENDO_SHARD_INDEX=0 \
        GOALZENDO_NUM_SHARDS=1 \
        GOALZENDO_GATE_ARTIFACT= \
        GOALZENDO_RUN_QWEN_INTEGRATION="$qwen_check" \
        "$ENTRYPOINT"
    fi
    qwen_check=0
  done
}

validate_gate_numbers() {
  local sft_rate="$1"
  local rl_rate="$2"
  local rl_entropy="$3"
  python - "$sft_rate" "$rl_rate" "$rl_entropy" <<'PY'
import math
import sys

try:
    sft, rl, entropy = (float(value) for value in sys.argv[1:])
except ValueError as error:
    raise SystemExit(f"selected optimizer settings must be finite numbers: {error}")
if not (math.isfinite(sft) and sft > 0 and math.isfinite(rl) and rl > 0):
    raise SystemExit("selected SFT and RL learning rates must be finite and positive")
if not (math.isfinite(entropy) and entropy >= 0):
    raise SystemExit("selected RL entropy must be finite and non-negative")
PY
}

run_gate() {
  local sft_rate="${GOALZENDO_SELECTED_SFT_LR:-}"
  local rl_rate="${GOALZENDO_SELECTED_RL_LR:-}"
  local rl_entropy="${GOALZENDO_SELECTED_RL_ENTROPY:-0}"
  local gate_output="${GOALZENDO_GATE_OUTPUT:-$STATUS_ROOT/g00-gate.json}"
  local assessment_output="${GOALZENDO_GATE_ASSESSMENT_OUTPUT:-$STATUS_ROOT/g00-gate-assessment.json}"
  local index
  local -a command
  if [[ -z "$sft_rate" || -z "$rl_rate" ]]; then
    echo "gate mode requires GOALZENDO_SELECTED_SFT_LR and GOALZENDO_SELECTED_RL_LR" >&2
    return 64
  fi
  validate_gate_numbers "$sft_rate" "$rl_rate" "$rl_entropy"
  if (( DRY_RUN == 0 )) && [[ ! -x "$VENV_PYTHON" ]]; then
    echo "GoalZendo environment is absent; run g00_suite or single first" >&2
    return 66
  fi
  if (( DRY_RUN == 0 )); then
    mkdir -p "$STATUS_ROOT"
  fi
  command=("$VENV_PYTHON" -m goalzendo.cli gate)
  for index in "${!G00D_CONFIGS[@]}"; do
    command+=(
      --g00-artifacts "$WORK_ROOT/artifacts-goalzendo/${G00D_DIRECTORIES[$index]}"
      --g00-config "${G00D_CONFIGS[$index]}"
    )
  done
  command+=(
    --target-config "$G01_CONFIG"
    --select-sft-learning-rate "$sft_rate"
    --select-rl-learning-rate "$rl_rate"
    --select-rl-entropy "$rl_entropy"
    --assessment-output "$assessment_output"
    --output "$gate_output"
  )
  printf 'mode\tgate\n'
  printf 'gate_output\t%s\nassessment_output\t%s\n' "$gate_output" "$assessment_output"
  if (( DRY_RUN == 1 )); then
    print_command "${command[@]}"
  else
    "${command[@]}"
  fi
}

run_g00e_gate() {
  local sft_rate="${GOALZENDO_SELECTED_SFT_LR:-}"
  local rl_rate="${GOALZENDO_SELECTED_RL_LR:-}"
  local rl_entropy="${GOALZENDO_SELECTED_RL_ENTROPY:-0}"
  local gate_output="${GOALZENDO_GATE_OUTPUT:-$STATUS_ROOT/g00e-gate-v3.json}"
  local assessment_output="${GOALZENDO_GATE_ASSESSMENT_OUTPUT:-$STATUS_ROOT/g00e-gate-assessment-v3.json}"
  local sidecar_output="${GOALZENDO_G00E_V3_SIDECAR_OUTPUT:-$STATUS_ROOT/g00e-numerical-bridge-v3.json}"
  local index
  local -a command
  if [[ -z "$sft_rate" || -z "$rl_rate" ]]; then
    echo "g00e_gate mode requires GOALZENDO_SELECTED_SFT_LR and GOALZENDO_SELECTED_RL_LR" >&2
    return 64
  fi
  validate_gate_numbers "$sft_rate" "$rl_rate" "$rl_entropy"
  if [[ ! "$G00E_BRIDGE_MANIFEST_SHA256" =~ ^[0-9a-f]{64}$ ]]; then
    echo "the frozen G00-E bridge manifest digest is not configured" >&2
    return 65
  fi
  if (( DRY_RUN == 0 )) && [[ ! -x "$VENV_PYTHON" ]]; then
    echo "GoalZendo environment is absent; run g00d_suite or single first" >&2
    return 66
  fi
  if (( DRY_RUN == 0 )); then
    mkdir -p "$STATUS_ROOT"
    if [[ -e "$sidecar_output" ]]; then
      echo "refusing to overwrite existing G00-E v3 sidecar: $sidecar_output" >&2
      return 66
    fi
  fi
  command=("$VENV_PYTHON" -m goalzendo_g00e.cli gate)
  for index in "${!G00D_CONFIGS[@]}"; do
    command+=(
      --g00-artifacts "$WORK_ROOT/artifacts-goalzendo/${G00D_DIRECTORIES[$index]}"
      --g00-config "${G00D_CONFIGS[$index]}"
    )
  done
  command+=(
    --target-config "$G01_CONFIG"
    --select-sft-learning-rate "$sft_rate"
    --select-rl-learning-rate "$rl_rate"
    --select-rl-entropy "$rl_entropy"
    --assessment-output "$assessment_output"
    --output "$gate_output"
    --g00e-sidecar-output "$sidecar_output"
    --g00e-manifest-sha256 "$G00E_BRIDGE_MANIFEST_SHA256"
  )
  printf 'mode\tg00e_gate\n'
  printf 'gate_output\t%s\nassessment_output\t%s\nsidecar_output\t%s\n' \
    "$gate_output" "$assessment_output" "$sidecar_output"
  if (( DRY_RUN == 1 )); then
    print_command "${command[@]}"
  else
    "${command[@]}"
    printf 'sidecar_sha256\t%s\n' "$(sha256sum "$sidecar_output" | awk '{print $1}')"
  fi
}

run_g01() {
  local gate_artifact="${GOALZENDO_GATE_ARTIFACT:-}"
  local g00e_sidecar="${GOALZENDO_G00E_V3_SIDECAR:-}"
  local g00e_sidecar_sha256="${GOALZENDO_G00E_V3_SIDECAR_SHA256:-}"
  local local_shards="${GOALZENDO_LOCAL_GPU_SHARDS:-1}"
  local output_root="${GOALZENDO_OUTPUT_ROOT:-$WORK_ROOT/artifacts-goalzendo/g01-known-law}"
  local analysis_root="${GOALZENDO_ANALYSIS_ROOT:-$WORK_ROOT/analysis-goalzendo/g01-known-law}"
  if [[ -z "$gate_artifact" || -z "$g00e_sidecar" || -z "$g00e_sidecar_sha256" ]]; then
    echo "g01 mode requires explicit GOALZENDO_GATE_ARTIFACT, GOALZENDO_G00E_V3_SIDECAR, and GOALZENDO_G00E_V3_SIDECAR_SHA256" >&2
    return 64
  fi
  if [[ ! "$g00e_sidecar_sha256" =~ ^[0-9a-f]{64}$ ]]; then
    echo "GOALZENDO_G00E_V3_SIDECAR_SHA256 must be a lowercase SHA-256 digest" >&2
    return 64
  fi
  if [[ ! "$local_shards" =~ ^[1-9][0-9]*$ ]]; then
    echo "GOALZENDO_LOCAL_GPU_SHARDS must be a canonical positive integer" >&2
    return 64
  fi
  if (( DRY_RUN == 0 )) && [[ ! -f "$gate_artifact" ]]; then
    echo "G01 gate artifact does not exist: $gate_artifact" >&2
    return 66
  fi
  if (( DRY_RUN == 0 )) && [[ ! -f "$g00e_sidecar" ]]; then
    echo "G01 G00-E v3 sidecar does not exist: $g00e_sidecar" >&2
    return 66
  fi
  printf 'mode\tg01\n'
  printf 'gate_artifact\t%s\n' "$gate_artifact"
  printf 'g00e_v3_sidecar\t%s\ng00e_v3_sidecar_sha256\t%s\n' \
    "$g00e_sidecar" "$g00e_sidecar_sha256"
  if (( DRY_RUN == 1 )); then
    print_command env \
      "GOALZENDO_CONFIG=$G01_CONFIG" \
      "GOALZENDO_RUN_LABEL=g01-known-law" \
      "GOALZENDO_OUTPUT_ROOT=$output_root" \
      "GOALZENDO_ANALYSIS_ROOT=$analysis_root" \
      "GOALZENDO_LOCAL_GPU_SHARDS=$local_shards" \
      "GOALZENDO_SHARD_INDEX=0" \
      "GOALZENDO_NUM_SHARDS=1" \
      "GOALZENDO_CLI_MODULE=goalzendo_g00e.cli" \
      "GOALZENDO_GATE_ARTIFACT=$gate_artifact" \
      "GOALZENDO_G00E_V3_SIDECAR=$g00e_sidecar" \
      "GOALZENDO_G00E_V3_SIDECAR_SHA256=$g00e_sidecar_sha256" \
      "GOALZENDO_G00E_MANIFEST_SHA256=$G00E_BRIDGE_MANIFEST_SHA256" \
      "$ENTRYPOINT"
  else
    exec env \
      GOALZENDO_CONFIG="$G01_CONFIG" \
      GOALZENDO_RUN_LABEL=g01-known-law \
      GOALZENDO_OUTPUT_ROOT="$output_root" \
      GOALZENDO_ANALYSIS_ROOT="$analysis_root" \
      GOALZENDO_LOCAL_GPU_SHARDS="$local_shards" \
      GOALZENDO_SHARD_INDEX=0 \
      GOALZENDO_NUM_SHARDS=1 \
      GOALZENDO_CLI_MODULE=goalzendo_g00e.cli \
      GOALZENDO_GATE_ARTIFACT="$gate_artifact" \
      GOALZENDO_G00E_V3_SIDECAR="$g00e_sidecar" \
      GOALZENDO_G00E_V3_SIDECAR_SHA256="$g00e_sidecar_sha256" \
      GOALZENDO_G00E_MANIFEST_SHA256="$G00E_BRIDGE_MANIFEST_SHA256" \
      "$ENTRYPOINT"
  fi
}

case "$MODE" in
  single)
    run_single
    ;;
  g00_suite)
    run_g00_suite
    ;;
  g00d_suite)
    run_g00d_suite
    ;;
  gate)
    run_gate
    ;;
  g00e_gate)
    run_g00e_gate
    ;;
  g01)
    run_g01
    ;;
  *)
    usage >&2
    exit 64
    ;;
esac
