#!/usr/bin/env bash
set -Eeuo pipefail

# PyTorch's deterministic CUDA mode requires this to be set before the first
# cuBLAS handle is created. It is inert unless a run enables deterministic
# algorithms, and the backend verifies the exact configured value.
export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
RUNTIME_HELPER="$ROOT/runs/goalzendo/runpod_runtime.py"
SYSTEM_PYTHON="$(command -v python)"
WORK_ROOT="${GOALZENDO_WORK_ROOT:-/workspace}"
VENV_ROOT="$WORK_ROOT/.venvs/goalzendo"
LOG_ROOT="$WORK_ROOT/logs-goalzendo"
STATUS_ROOT="$WORK_ROOT/status-goalzendo"
CONFIG_PATH="${GOALZENDO_CONFIG:-configs/goalzendo/smoke.yaml}"
RUN_LABEL="${GOALZENDO_RUN_LABEL:-smoke}"
CLI_MODULE="${GOALZENDO_CLI_MODULE:-goalzendo.cli}"
OUTPUT_ROOT="${GOALZENDO_OUTPUT_ROOT:-$WORK_ROOT/artifacts-goalzendo/g00-smoke}"
ANALYSIS_ROOT="${GOALZENDO_ANALYSIS_ROOT:-$WORK_ROOT/analysis-goalzendo/$RUN_LABEL}"
SHARD_INDEX="${GOALZENDO_SHARD_INDEX:-0}"
NUM_SHARDS="${GOALZENDO_NUM_SHARDS:-1}"
LOCAL_GPU_SHARDS_RAW="${GOALZENDO_LOCAL_GPU_SHARDS:-}"
GATE_ARTIFACT="${GOALZENDO_GATE_ARTIFACT:-}"
G00E_V3_SIDECAR="${GOALZENDO_G00E_V3_SIDECAR:-}"
G00E_V3_SIDECAR_SHA256="${GOALZENDO_G00E_V3_SIDECAR_SHA256:-}"
G00E_MANIFEST_SHA256="${GOALZENDO_G00E_MANIFEST_SHA256:-}"
REQUIRE_PERSISTENT_WORK_ROOT="${GOALZENDO_REQUIRE_PERSISTENT_WORK_ROOT:-0}"
RUN_COMBINED_ANALYSIS="${GOALZENDO_RUN_COMBINED_ANALYSIS:-1}"
RUN_LOG="$LOG_ROOT/$RUN_LABEL.log"
RUNNING_STATUS="$STATUS_ROOT/$RUN_LABEL.running.json"
FAILED_STATUS="$STATUS_ROOT/$RUN_LABEL.failed.json"
COMPLETE_STATUS="$STATUS_ROOT/$RUN_LABEL.complete"
RUNTIME_MANIFEST="$STATUS_ROOT/$RUN_LABEL-runtime.json"

if [[ ! "$RUN_LABEL" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]]; then
  echo "GOALZENDO_RUN_LABEL must use only letters, digits, dots, underscores, and hyphens" >&2
  exit 64
fi
if [[ "$CLI_MODULE" != "goalzendo.cli" && "$CLI_MODULE" != "goalzendo_g00e.cli" ]]; then
  echo "GOALZENDO_CLI_MODULE must be goalzendo.cli or goalzendo_g00e.cli" >&2
  exit 64
fi
if [[ "$CLI_MODULE" == "goalzendo_g00e.cli" ]]; then
  if [[ -z "$GATE_ARTIFACT" || -z "$G00E_V3_SIDECAR" || -z "$G00E_V3_SIDECAR_SHA256" || -z "$G00E_MANIFEST_SHA256" ]]; then
    echo "G00-E workers require explicit gate, v3 sidecar path, sidecar SHA-256, and manifest SHA-256" >&2
    exit 66
  fi
  if [[ ! "$G00E_V3_SIDECAR_SHA256" =~ ^[0-9a-f]{64}$ || ! "$G00E_MANIFEST_SHA256" =~ ^[0-9a-f]{64}$ ]]; then
    echo "G00-E expected digests must be lowercase SHA-256 values" >&2
    exit 64
  fi
  if [[ ! -f "$G00E_V3_SIDECAR" ]]; then
    echo "G00-E v3 sidecar does not exist: $G00E_V3_SIDECAR" >&2
    exit 66
  fi
  G00E_V3_SIDECAR_OBSERVED_SHA256="$(sha256sum "$G00E_V3_SIDECAR" | awk '{print $1}')"
  if [[ "$G00E_V3_SIDECAR_OBSERVED_SHA256" != "$G00E_V3_SIDECAR_SHA256" ]]; then
    echo "G00-E v3 sidecar differs from the externally expected SHA-256" >&2
    exit 66
  fi
else
  if [[ -n "$G00E_V3_SIDECAR" || -n "$G00E_V3_SIDECAR_SHA256" || -n "$G00E_MANIFEST_SHA256" ]]; then
    echo "G00-E identity variables require GOALZENDO_CLI_MODULE=goalzendo_g00e.cli" >&2
    exit 64
  fi
  G00E_V3_SIDECAR_OBSERVED_SHA256=""
fi
if [[ ! "$SHARD_INDEX" =~ ^[0-9]+$ || ! "$NUM_SHARDS" =~ ^[1-9][0-9]*$ ]]; then
  echo "GOALZENDO_SHARD_INDEX and GOALZENDO_NUM_SHARDS must be canonical non-negative integers" >&2
  exit 64
fi
SHARD_INDEX="$((10#$SHARD_INDEX))"
NUM_SHARDS="$((10#$NUM_SHARDS))"
if (( SHARD_INDEX >= NUM_SHARDS )); then
  echo "GOALZENDO_SHARD_INDEX must be smaller than GOALZENDO_NUM_SHARDS" >&2
  exit 64
fi
if [[ -n "$LOCAL_GPU_SHARDS_RAW" ]]; then
  if [[ ! "$LOCAL_GPU_SHARDS_RAW" =~ ^[1-9][0-9]*$ ]]; then
    echo "GOALZENDO_LOCAL_GPU_SHARDS must be a canonical positive integer" >&2
    exit 64
  fi
  LOCAL_GPU_SHARDS="$((10#$LOCAL_GPU_SHARDS_RAW))"
else
  LOCAL_GPU_SHARDS=0
fi
if [[ "$REQUIRE_PERSISTENT_WORK_ROOT" != "0" && "$REQUIRE_PERSISTENT_WORK_ROOT" != "1" ]]; then
  echo "GOALZENDO_REQUIRE_PERSISTENT_WORK_ROOT must be 0 or 1" >&2
  exit 64
fi
if [[ "$RUN_COMBINED_ANALYSIS" != "0" && "$RUN_COMBINED_ANALYSIS" != "1" ]]; then
  echo "GOALZENDO_RUN_COMBINED_ANALYSIS must be 0 or 1" >&2
  exit 64
fi

mkdir -p "$LOG_ROOT" "$STATUS_ROOT" "$WORK_ROOT/huggingface"
rm -f \
  "$COMPLETE_STATUS" \
  "$FAILED_STATUS" \
  "$STATUS_ROOT/$RUN_LABEL-analysis.json" \
  "$STATUS_ROOT/$RUN_LABEL-analysis.tar.gz.sha256" \
  "$STATUS_ROOT/$RUN_LABEL-packages.txt.sha256" \
  "$RUNTIME_MANIFEST.sha256"

runtime_json() {
  "$SYSTEM_PYTHON" "$RUNTIME_HELPER" write-json "$@"
}

abort_runtime() {
  local code="$1"
  shift
  printf '%s\n' "$*" >&2
  return "$code"
}

CURRENT_PHASE="initialization"
declare -a CHILD_PIDS=()

terminate_children() {
  local pid
  for pid in "${CHILD_PIDS[@]:-}"; do
    if kill -0 "$pid" 2>/dev/null; then
      kill -TERM "$pid" 2>/dev/null || true
    fi
  done
  for pid in "${CHILD_PIDS[@]:-}"; do
    wait "$pid" 2>/dev/null || true
  done
  CHILD_PIDS=()
}

write_failure_status() {
  local code="$1"
  local failed_at
  failed_at="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  if ! runtime_json \
    --output "$FAILED_STATUS" \
    --string "state=failed" \
    --integer "exit_code=$code" \
    --string "phase=$CURRENT_PHASE" \
    --string "failed_at=$failed_at" \
    --string "main_log=$RUN_LOG"; then
    printf '{"state":"failed","exit_code":%d,"phase":"runtime_status_write_failed"}\n' \
      "$code" > "$FAILED_STATUS"
  fi
  rm -f "$RUNNING_STATUS"
}

on_error() {
  local code=$?
  trap - ERR INT TERM
  set +e
  (( code != 0 )) || code=1
  terminate_children
  write_failure_status "$code"
  echo "goalzendo_entrypoint_failed phase=$CURRENT_PHASE exit_code=$code" >&2
  exit "$code"
}

on_signal() {
  local signal="$1"
  local code="$2"
  CURRENT_PHASE="signal_$signal"
  trap - ERR INT TERM
  set +e
  terminate_children
  write_failure_status "$code"
  exit "$code"
}

trap on_error ERR
trap 'on_signal INT 130' INT
trap 'on_signal TERM 143' TERM

runtime_json \
  --output "$RUNNING_STATUS" \
  --string "state=running" \
  --string "phase=$CURRENT_PHASE" \
  --string "started_at=$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
  --string "main_log=$RUN_LOG"

exec > >(tee -a "$RUN_LOG") 2>&1

echo "goalzendo_entrypoint_start $(date -u +%Y-%m-%dT%H:%M:%SZ)"
CURRENT_PHASE="persistent_storage_preflight"
WORK_MOUNT_TARGET=""
WORK_MOUNT_SOURCE=""
WORK_MOUNT_FSTYPE=""
if command -v findmnt >/dev/null 2>&1; then
  WORK_MOUNT_TARGET="$(findmnt -n -T "$WORK_ROOT" -o TARGET | head -n 1 | xargs)"
  WORK_MOUNT_SOURCE="$(findmnt -n -T "$WORK_ROOT" -o SOURCE | head -n 1 | xargs)"
  WORK_MOUNT_FSTYPE="$(findmnt -n -T "$WORK_ROOT" -o FSTYPE | head -n 1 | xargs)"
fi
if (( REQUIRE_PERSISTENT_WORK_ROOT == 1 )); then
  if [[ -z "$WORK_MOUNT_TARGET" || "$WORK_MOUNT_TARGET" == "/" ]]; then
    abort_runtime 67 \
      "GOALZENDO_WORK_ROOT is not below a separately mounted persistent filesystem"
  fi
  echo "persistent_work_root_preflight target=$WORK_MOUNT_TARGET fstype=$WORK_MOUNT_FSTYPE"
fi
CURRENT_PHASE="gpu_inventory"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
GPU_INDEX_ROWS="$(nvidia-smi --query-gpu=index --format=csv,noheader,nounits)"
DETECTED_GPU_COUNT="$(printf '%s\n' "$GPU_INDEX_ROWS" | awk 'NF {count += 1} END {print count + 0}')"
python --version

LOCAL_MULTI_GPU=0
LOCAL_SHARD_PLAN_JSON=""
LOCAL_SHARD_PLAN_TSV=""
LOCAL_SHARD_LOG_ROOT=""
LOCAL_SHARD_STATUS_ROOT=""
declare -a LOCAL_GPU_DEVICES=()

if (( LOCAL_GPU_SHARDS > 0 )); then
  CURRENT_PHASE="local_gpu_shard_planning"
  if (( LOCAL_GPU_SHARDS > 1 )) && (( SHARD_INDEX != 0 || NUM_SHARDS != 1 )); then
    abort_runtime 64 \
      "local multi-GPU sharding cannot be combined with GOALZENDO_SHARD_INDEX/NUM_SHARDS"
  fi
  LOCAL_SHARD_PLAN_JSON="$STATUS_ROOT/$RUN_LABEL-local-shard-plan.json"
  LOCAL_SHARD_PLAN_TSV="$STATUS_ROOT/$RUN_LABEL-local-shard-plan.tsv"
  visible_arguments=()
  if [[ "${CUDA_VISIBLE_DEVICES+x}" == "x" ]]; then
    visible_arguments=(--visible-devices "$CUDA_VISIBLE_DEVICES")
  fi
  plan_temporary="$LOCAL_SHARD_PLAN_JSON.tmp.$$"
  "$SYSTEM_PYTHON" "$RUNTIME_HELPER" plan \
    --requested-shards "$LOCAL_GPU_SHARDS" \
    --detected-gpus "$DETECTED_GPU_COUNT" \
    "${visible_arguments[@]}" --format json > "$plan_temporary"
  mv "$plan_temporary" "$LOCAL_SHARD_PLAN_JSON"
  plan_temporary="$LOCAL_SHARD_PLAN_TSV.tmp.$$"
  "$SYSTEM_PYTHON" "$RUNTIME_HELPER" plan \
    --requested-shards "$LOCAL_GPU_SHARDS" \
    --detected-gpus "$DETECTED_GPU_COUNT" \
    "${visible_arguments[@]}" --format tsv > "$plan_temporary"
  mv "$plan_temporary" "$LOCAL_SHARD_PLAN_TSV"

  expected_index=0
  while IFS=$'\t' read -r planned_index planned_total planned_device; do
    if [[ "$planned_index" != "$expected_index" || "$planned_total" != "$LOCAL_GPU_SHARDS" ]]; then
      abort_runtime 65 "local GPU shard helper returned a non-contiguous plan"
    fi
    LOCAL_GPU_DEVICES+=("$planned_device")
    expected_index=$((expected_index + 1))
  done < "$LOCAL_SHARD_PLAN_TSV"
  if (( ${#LOCAL_GPU_DEVICES[@]} != LOCAL_GPU_SHARDS )); then
    abort_runtime 65 "local GPU shard helper returned the wrong number of assignments"
  fi
  if (( LOCAL_GPU_SHARDS > 1 )); then
    LOCAL_MULTI_GPU=1
    LOCAL_SHARD_LOG_ROOT="$LOG_ROOT/$RUN_LABEL-shards"
    LOCAL_SHARD_STATUS_ROOT="$STATUS_ROOT/$RUN_LABEL-shards"
    mkdir -p "$LOCAL_SHARD_LOG_ROOT" "$LOCAL_SHARD_STATUS_ROOT"
  fi
fi

CURRENT_PHASE="environment_setup"
if [[ ! -x "$VENV_ROOT/bin/python" ]]; then
  python -m venv --system-site-packages "$VENV_ROOT"
fi

"$VENV_ROOT/bin/python" -m pip install --disable-pip-version-check --no-cache-dir \
  -c "$ROOT/constraints-goalzendo.txt" -e "$ROOT[llm,dev]"

export PYTHON_BIN="$VENV_ROOT/bin/python"
export HF_HOME="$WORK_ROOT/huggingface"
export HUGGINGFACE_HUB_CACHE="$HF_HOME/hub"
export HF_HUB_ENABLE_HF_TRANSFER=0
export HF_HUB_DISABLE_TELEMETRY=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1

cd "$ROOT"
CURRENT_PHASE="runtime_tests"
"$PYTHON_BIN" -m pytest -q tests/goalzendo
CURRENT_PHASE="config_validation"
"$PYTHON_BIN" -m "$CLI_MODULE" validate "$CONFIG_PATH" --json

RUN_GATE_ARGUMENTS=()
if [[ -n "$GATE_ARTIFACT" ]]; then
  if [[ ! -f "$GATE_ARTIFACT" ]]; then
    abort_runtime 66 "GOALZENDO_GATE_ARTIFACT does not exist: $GATE_ARTIFACT"
  fi
  RUN_GATE_ARGUMENTS=(--gate-artifact "$GATE_ARTIFACT")
fi
RUN_BRIDGE_ARGUMENTS=()
if [[ "$CLI_MODULE" == "goalzendo_g00e.cli" ]]; then
  RUN_BRIDGE_ARGUMENTS=(
    --g00e-sidecar "$G00E_V3_SIDECAR"
    --g00e-sidecar-sha256 "$G00E_V3_SIDECAR_SHA256"
    --g00e-manifest-sha256 "$G00E_MANIFEST_SHA256"
  )
fi

if [[ "$CLI_MODULE" == "goalzendo_g00e.cli" ]]; then
  CURRENT_PHASE="g00e_worker_preflight"
  "$PYTHON_BIN" -m "$CLI_MODULE" worker-preflight \
    --repo "$ROOT" --gate-artifact "$GATE_ARTIFACT" \
    "${RUN_BRIDGE_ARGUMENTS[@]}"
fi

CURRENT_PHASE="qwen_integration"
if [[ "${GOALZENDO_RUN_QWEN_INTEGRATION:-0}" == "1" ]]; then
  if (( LOCAL_MULTI_GPU == 1 )); then
    CUDA_VISIBLE_DEVICES="${LOCAL_GPU_DEVICES[0]}" \
      "$PYTHON_BIN" -m goalzendo.modeling --integration-check-qwen \
      --device cuda \
      --max-prompt-tokens "${GOALZENDO_MAX_PROMPT_TOKENS:-640}" \
      --output "$STATUS_ROOT/$RUN_LABEL-qwen-integration.json"
  else
    "$PYTHON_BIN" -m goalzendo.modeling --integration-check-qwen \
      --device cuda \
      --max-prompt-tokens "${GOALZENDO_MAX_PROMPT_TOKENS:-640}" \
      --output "$STATUS_ROOT/$RUN_LABEL-qwen-integration.json"
  fi
fi

run_local_shard() (
  trap - ERR
  set +e
  local shard_index="$1"
  local shard_total="$2"
  local gpu_device="$3"
  local suffix
  local shard_log
  local shard_status
  local shard_child_pid=""
  local shard_started_at
  local shard_code
  suffix="$(printf 'shard-%03d-of-%03d' "$shard_index" "$shard_total")"
  shard_log="$LOCAL_SHARD_LOG_ROOT/$suffix.log"
  shard_status="$LOCAL_SHARD_STATUS_ROOT/$suffix.json"
  shard_started_at="$(date -u +%Y-%m-%dT%H:%M:%SZ)"

  on_shard_signal() {
    local signal="$1"
    local code="$2"
    trap - INT TERM
    if [[ -n "$shard_child_pid" ]] && kill -0 "$shard_child_pid" 2>/dev/null; then
      kill -TERM "$shard_child_pid" 2>/dev/null || true
      wait "$shard_child_pid" 2>/dev/null || true
    fi
    runtime_json \
      --output "$shard_status" \
      --string "state=failed" \
      --integer "exit_code=$code" \
      --string "signal=$signal" \
      --integer "shard_index=$shard_index" \
      --integer "num_shards=$shard_total" \
      --string "cuda_visible_devices=$gpu_device" \
      --string "log=$shard_log" \
      --string "finished_at=$(date -u +%Y-%m-%dT%H:%M:%SZ)" || true
    exit "$code"
  }
  trap 'on_shard_signal INT 130' INT
  trap 'on_shard_signal TERM 143' TERM

  if ! runtime_json \
    --output "$shard_status" \
    --string "state=running" \
    --integer "shard_index=$shard_index" \
    --integer "num_shards=$shard_total" \
    --string "cuda_visible_devices=$gpu_device" \
    --string "log=$shard_log" \
    --string "started_at=$shard_started_at"; then
    echo "could not write running status for local shard $shard_index" >&2
    exit 70
  fi

  (
    echo "goalzendo_local_shard_start index=$shard_index total=$shard_total gpu=$gpu_device at=$shard_started_at"
    CUDA_VISIBLE_DEVICES="$gpu_device" "$PYTHON_BIN" -m "$CLI_MODULE" plan "$CONFIG_PATH" \
      --shard-index "$shard_index" --num-shards "$shard_total" --output-root "$OUTPUT_ROOT" || exit $?
    exec env CUDA_VISIBLE_DEVICES="$gpu_device" \
      "$PYTHON_BIN" -m "$CLI_MODULE" run "$CONFIG_PATH" \
      --shard-index "$shard_index" --num-shards "$shard_total" --output-root "$OUTPUT_ROOT" \
      "${RUN_GATE_ARGUMENTS[@]}" "${RUN_BRIDGE_ARGUMENTS[@]}"
  ) >> "$shard_log" 2>&1 &
  shard_child_pid=$!
  if wait "$shard_child_pid"; then
    shard_code=0
  else
    shard_code=$?
  fi
  shard_child_pid=""

  if (( shard_code == 0 )); then
    if ! runtime_json \
      --output "$shard_status" \
      --string "state=complete" \
      --integer "exit_code=0" \
      --integer "shard_index=$shard_index" \
      --integer "num_shards=$shard_total" \
      --string "cuda_visible_devices=$gpu_device" \
      --string "log=$shard_log" \
      --string "started_at=$shard_started_at" \
      --string "finished_at=$(date -u +%Y-%m-%dT%H:%M:%SZ)"; then
      echo "could not write completion status for local shard $shard_index" >&2
      shard_code=70
    fi
  else
    runtime_json \
      --output "$shard_status" \
      --string "state=failed" \
      --integer "exit_code=$shard_code" \
      --integer "shard_index=$shard_index" \
      --integer "num_shards=$shard_total" \
      --string "cuda_visible_devices=$gpu_device" \
      --string "log=$shard_log" \
      --string "started_at=$shard_started_at" \
      --string "finished_at=$(date -u +%Y-%m-%dT%H:%M:%SZ)" || true
  fi
  exit "$shard_code"
)

run_local_gpu_panel() {
  local index
  local code
  local local_pid
  local first_failure=0
  CHILD_PIDS=()
  for index in "${!LOCAL_GPU_DEVICES[@]}"; do
    run_local_shard "$index" "$LOCAL_GPU_SHARDS" "${LOCAL_GPU_DEVICES[$index]}" &
    local_pid=$!
    CHILD_PIDS+=("$local_pid")
    echo "goalzendo_local_shard_launched index=$index gpu=${LOCAL_GPU_DEVICES[$index]} pid=$local_pid"
  done
  for index in "${!CHILD_PIDS[@]}"; do
    if wait "${CHILD_PIDS[$index]}"; then
      code=0
    else
      code=$?
    fi
    echo "goalzendo_local_shard_terminal index=$index exit_code=$code status=$LOCAL_SHARD_STATUS_ROOT/$(printf 'shard-%03d-of-%03d.json' "$index" "$LOCAL_GPU_SHARDS")"
    if (( code != 0 && first_failure == 0 )); then
      first_failure=$code
    fi
  done
  CHILD_PIDS=()
  if (( first_failure != 0 )); then
    echo "one or more local GPU shards failed; combined analysis was not started" >&2
    return "$first_failure"
  fi
}

CURRENT_PHASE="experiment_execution"
if (( LOCAL_MULTI_GPU == 1 )); then
  run_local_gpu_panel
else
  "$PYTHON_BIN" -m "$CLI_MODULE" plan "$CONFIG_PATH" \
    --shard-index "$SHARD_INDEX" --num-shards "$NUM_SHARDS" --output-root "$OUTPUT_ROOT"
  "$PYTHON_BIN" -m "$CLI_MODULE" run "$CONFIG_PATH" \
    --shard-index "$SHARD_INDEX" --num-shards "$NUM_SHARDS" --output-root "$OUTPUT_ROOT" \
    "${RUN_GATE_ARGUMENTS[@]}" "${RUN_BRIDGE_ARGUMENTS[@]}"
fi

# Analyze only when this process owns a complete panel.  In local multi-GPU
# mode every deterministic shard has already joined under OUTPUT_ROOT.  An
# externally sharded single-GPU process leaves analysis to its orchestrator.
ANALYSIS_ARCHIVE=""
ANALYSIS_SHA256=""
ANALYSIS_BYTES=""
ANALYSIS_STATUS=""
if (( RUN_COMBINED_ANALYSIS == 1 && (LOCAL_MULTI_GPU == 1 || NUM_SHARDS == 1) )); then
  CURRENT_PHASE="combined_analysis"
  mkdir -p "$ANALYSIS_ROOT"
  "$PYTHON_BIN" -m "$CLI_MODULE" analyze "$OUTPUT_ROOT" \
    --config "$CONFIG_PATH" --output "$ANALYSIS_ROOT" \
    --bootstrap-draws 10000 --confidence 0.95 --no-figures
  ANALYSIS_ARCHIVE="$STATUS_ROOT/$RUN_LABEL-analysis.tar.gz"
  archive_temporary="$ANALYSIS_ARCHIVE.tmp.$$"
  tar -czf "$archive_temporary" -C "$ANALYSIS_ROOT" .
  mv "$archive_temporary" "$ANALYSIS_ARCHIVE"
  ANALYSIS_SHA256="$(sha256sum "$ANALYSIS_ARCHIVE" | awk '{print $1}')"
  ANALYSIS_BYTES="$(wc -c < "$ANALYSIS_ARCHIVE" | tr -d ' ')"
  ANALYSIS_STATUS="$STATUS_ROOT/$RUN_LABEL-analysis.json"
  printf '%s  %s\n' "$ANALYSIS_SHA256" "$(basename "$ANALYSIS_ARCHIVE")" \
    > "$ANALYSIS_ARCHIVE.sha256.tmp.$$"
  mv "$ANALYSIS_ARCHIVE.sha256.tmp.$$" "$ANALYSIS_ARCHIVE.sha256"
  runtime_json \
    --output "$ANALYSIS_STATUS" \
    --string "state=ready" \
    --string "path=$ANALYSIS_ARCHIVE" \
    --string "sha256=$ANALYSIS_SHA256" \
    --integer "bytes=$ANALYSIS_BYTES" \
    --string "checksum_path=$ANALYSIS_ARCHIVE.sha256"
  if [[ "${GOALZENDO_EMIT_ANALYSIS_B64:-0}" == "1" ]]; then
    printf 'GOALZENDO_ANALYSIS_B64_BEGIN sha256=%s bytes=%s\n' \
      "$ANALYSIS_SHA256" "$ANALYSIS_BYTES"
    base64 "$ANALYSIS_ARCHIVE"
    printf 'GOALZENDO_ANALYSIS_B64_END\n'
  fi
fi

CURRENT_PHASE="runtime_manifest"
PACKAGES_PATH="$STATUS_ROOT/$RUN_LABEL-packages.txt"
"$PYTHON_BIN" -m pip freeze > "$PACKAGES_PATH.tmp.$$"
mv "$PACKAGES_PATH.tmp.$$" "$PACKAGES_PATH"
PACKAGES_SHA256="$(sha256sum "$PACKAGES_PATH" | awk '{print $1}')"
printf '%s  %s\n' "$PACKAGES_SHA256" "$(basename "$PACKAGES_PATH")" \
  > "$PACKAGES_PATH.sha256.tmp.$$"
mv "$PACKAGES_PATH.sha256.tmp.$$" "$PACKAGES_PATH.sha256"

runtime_manifest_arguments=(
  --output "$RUNTIME_MANIFEST"
  --string "state=artifacts_ready"
  --string "run_label=$RUN_LABEL"
  --string "cli_module=$CLI_MODULE"
  --string "config=$CONFIG_PATH"
  --string "output_root=$OUTPUT_ROOT"
  --string "analysis_root=$ANALYSIS_ROOT"
  --string "main_log=$RUN_LOG"
  --string "package_inventory=$PACKAGES_PATH"
  --string "package_inventory_sha256=$PACKAGES_SHA256"
  --string "package_inventory_checksum=$PACKAGES_PATH.sha256"
  --integer "local_gpu_shards=$LOCAL_GPU_SHARDS"
  --integer "external_shard_index=$SHARD_INDEX"
  --integer "external_num_shards=$NUM_SHARDS"
  --boolean "persistent_work_root_required=$([[ "$REQUIRE_PERSISTENT_WORK_ROOT" == "1" ]] && echo true || echo false)"
  --boolean "combined_analysis_requested=$([[ "$RUN_COMBINED_ANALYSIS" == "1" ]] && echo true || echo false)"
)
if [[ -n "$WORK_MOUNT_TARGET" ]]; then
  runtime_manifest_arguments+=(
    --string "work_mount_target=$WORK_MOUNT_TARGET"
    --string "work_mount_source=$WORK_MOUNT_SOURCE"
    --string "work_mount_fstype=$WORK_MOUNT_FSTYPE"
  )
fi
if [[ -n "$LOCAL_SHARD_PLAN_JSON" ]]; then
  runtime_manifest_arguments+=(--string "local_shard_plan=$LOCAL_SHARD_PLAN_JSON")
fi
if [[ -n "$LOCAL_SHARD_LOG_ROOT" ]]; then
  runtime_manifest_arguments+=(
    --string "local_shard_log_root=$LOCAL_SHARD_LOG_ROOT"
    --string "local_shard_status_root=$LOCAL_SHARD_STATUS_ROOT"
  )
fi
if [[ -n "$ANALYSIS_ARCHIVE" ]]; then
  runtime_manifest_arguments+=(
    --string "analysis_archive=$ANALYSIS_ARCHIVE"
    --string "analysis_archive_sha256=$ANALYSIS_SHA256"
    --integer "analysis_archive_bytes=$ANALYSIS_BYTES"
    --string "analysis_archive_checksum=$ANALYSIS_ARCHIVE.sha256"
    --string "analysis_status=$ANALYSIS_STATUS"
  )
fi
if [[ -n "$GATE_ARTIFACT" ]]; then
  runtime_manifest_arguments+=(--string "gate_artifact=$GATE_ARTIFACT")
fi
if [[ "$CLI_MODULE" == "goalzendo_g00e.cli" ]]; then
  G00E_BRIDGE_CLI_SHA256="$(sha256sum "$ROOT/src/goalzendo_g00e/cli.py" | awk '{print $1}')"
  G00E_ENTRYPOINT_SHA256="$(sha256sum "$ROOT/runs/goalzendo/runpod_entrypoint.sh" | awk '{print $1}')"
  G00E_DISPATCH_SHA256="$(sha256sum "$ROOT/runs/goalzendo/runpod_dispatch.sh" | awk '{print $1}')"
  runtime_manifest_arguments+=(
    --string "g00e_v3_sidecar=$G00E_V3_SIDECAR"
    --string "g00e_v3_sidecar_sha256=$G00E_V3_SIDECAR_OBSERVED_SHA256"
    --string "g00e_v3_expected_sidecar_sha256=$G00E_V3_SIDECAR_SHA256"
    --string "g00e_bridge_manifest_sha256=$G00E_MANIFEST_SHA256"
    --string "g00e_bridge_cli_sha256=$G00E_BRIDGE_CLI_SHA256"
    --string "g00e_runpod_entrypoint_sha256=$G00E_ENTRYPOINT_SHA256"
    --string "g00e_runpod_dispatch_sha256=$G00E_DISPATCH_SHA256"
  )
fi
runtime_json "${runtime_manifest_arguments[@]}"
RUNTIME_SHA256="$(sha256sum "$RUNTIME_MANIFEST" | awk '{print $1}')"
printf '%s  %s\n' "$RUNTIME_SHA256" "$(basename "$RUNTIME_MANIFEST")" \
  > "$RUNTIME_MANIFEST.sha256.tmp.$$"
mv "$RUNTIME_MANIFEST.sha256.tmp.$$" "$RUNTIME_MANIFEST.sha256"

echo "goalzendo_entrypoint_complete $(date -u +%Y-%m-%dT%H:%M:%SZ)"
rm -f "$RUNNING_STATUS" "$FAILED_STATUS"
runtime_json \
  --output "$COMPLETE_STATUS" \
  --string "state=complete" \
  --string "completed_at=$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
  --string "runtime_manifest=$RUNTIME_MANIFEST" \
  --string "runtime_manifest_sha256=$RUNTIME_SHA256"
trap - ERR INT TERM
