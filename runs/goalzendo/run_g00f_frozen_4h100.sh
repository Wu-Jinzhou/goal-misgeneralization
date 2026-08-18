#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/$(basename "${BASH_SOURCE[0]}")"
AMBIENT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
WORK_ROOT="${GOALZENDO_WORK_ROOT:-/workspace}"
PYTHON_BIN="${G00F_PYTHON_BIN:-$WORK_ROOT/.venvs/goalzendo/bin/python}"
FREEZE="${G00F_FREEZE:-$AMBIENT_ROOT/reproducibility/goalzendo/g00f-execution-freeze-20260811/execution-freeze.json}"
ARCHIVE="${G00F_SOURCE_ARCHIVE:-$AMBIENT_ROOT/reproducibility/goalzendo/g00f-execution-freeze-20260811/g00f-execution-source.tar.gz}"
MANIFEST="${G00F_SOURCE_MANIFEST:-$AMBIENT_ROOT/reproducibility/goalzendo/g00f-execution-freeze-20260811/g00f-source-bundle-manifest.json}"
EXPECTED_FREEZE_SHA256="${G00F_EXPECTED_FREEZE_SHA256:-}"
SNAPSHOT_0P5B="${G00F_SNAPSHOT_0P5B:-}"
SNAPSHOT_1P5B="${G00F_SNAPSHOT_1P5B:-}"
RUNTIME_IMAGE="${G00F_RUNTIME_IMAGE:-}"
VOLUME_ID="${G00F_NETWORK_VOLUME_ID:-}"
DATA_CENTER="${G00F_DATA_CENTER:-}"
EXTERNAL_PROVISION_RECEIPT="${G00F_RUNPOD_PROVISION_RECEIPT:-}"
EXPECTED_PROVISION_SHA256="${G00F_EXPECTED_PROVISION_RECEIPT_SHA256:-}"
RUNPOD_POD_ID="${RUNPOD_POD_ID:-}"

if [[ ! "$EXPECTED_FREEZE_SHA256" =~ ^[0-9a-f]{64}$ ]]; then
  echo "G00F_EXPECTED_FREEZE_SHA256 must be an externally supplied SHA-256" >&2
  exit 64
fi
if [[ -z "$EXTERNAL_PROVISION_RECEIPT" || ! "$EXPECTED_PROVISION_SHA256" =~ ^[0-9a-f]{64}$ || -z "$RUNPOD_POD_ID" ]]; then
  echo "an externally SHA-pinned Runpod provision receipt and RUNPOD_POD_ID are required" >&2
  exit 64
fi
if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "frozen G00-F Python is absent: $PYTHON_BIN" >&2
  exit 66
fi

# The ambient controller may do only one thing before authentication: invoke
# the stdlib verifier using the explicitly selected Python.  That verifier
# authenticates its own bytes plus this launcher and the watchdog, validates
# and extracts the archive, and records the proof.  Control then transfers by
# exec to the authenticated launcher copy inside the extracted archive.
if [[ "${G00F_FROZEN_REEXEC:-0}" != "1" ]]; then
  BOOTSTRAP="$AMBIENT_ROOT/runs/goalzendo/g00f_bundle_bootstrap.py"
  WATCHDOG="$AMBIENT_ROOT/runs/goalzendo/g00f_watchdog.py"
  EXECUTION_UUID="${G00F_EXECUTION_UUID:-$($PYTHON_BIN - <<'PY'
import uuid
print(uuid.uuid4())
PY
)}"
  if ! "$PYTHON_BIN" - "$EXECUTION_UUID" <<'PY'
import sys
import uuid

value = sys.argv[1]
parsed = uuid.UUID(value)
if parsed.version != 4 or str(parsed) != value:
    raise SystemExit(1)
PY
  then
    echo "G00F_EXECUTION_UUID must be a canonical UUID4" >&2
    exit 64
  fi
  EXECUTION_ROOT="$WORK_ROOT/status-goalzendo/g00f-executions/$EXECUTION_UUID"
  EXTRACTED_ROOT="$EXECUTION_ROOT/frozen-source"
  BUNDLE_RECEIPT="$EXECUTION_ROOT/source-bundle-receipt.json"
  "$PYTHON_BIN" "$BOOTSTRAP" \
    --freeze "$FREEZE" \
    --expected-freeze-sha256 "$EXPECTED_FREEZE_SHA256" \
    --archive "$ARCHIVE" \
    --manifest "$MANIFEST" \
    --output "$EXTRACTED_ROOT" \
    --receipt "$BUNDLE_RECEIPT" \
    --actual-bootstrap "$BOOTSTRAP" \
    --actual-launcher "$SCRIPT_PATH" \
    --actual-watchdog "$WATCHDOG"
  export G00F_FROZEN_REEXEC=1
  export G00F_EXECUTION_UUID="$EXECUTION_UUID"
  export G00F_EXECUTION_ROOT="$EXECUTION_ROOT"
  export G00F_EXTRACTED_ROOT="$EXTRACTED_ROOT"
  export G00F_BUNDLE_RECEIPT="$BUNDLE_RECEIPT"
  export G00F_FREEZE="$FREEZE"
  export G00F_SOURCE_ARCHIVE="$ARCHIVE"
  export G00F_SOURCE_MANIFEST="$MANIFEST"
  exec "$EXTRACTED_ROOT/runs/goalzendo/run_g00f_frozen_4h100.sh"
fi

ROOT="${G00F_EXTRACTED_ROOT:?authenticated extracted root is absent}"
EXECUTION_UUID="${G00F_EXECUTION_UUID:?execution UUID is absent}"
EXECUTION_ROOT="${G00F_EXECUTION_ROOT:?execution root is absent}"
BUNDLE_RECEIPT="${G00F_BUNDLE_RECEIPT:?source bundle receipt is absent}"
EXPECTED_SCRIPT="$ROOT/runs/goalzendo/run_g00f_frozen_4h100.sh"
if [[ "$SCRIPT_PATH" != "$EXPECTED_SCRIPT" ]]; then
  echo "G00-F control was not re-executed from the authenticated source bundle" >&2
  exit 65
fi
if [[ -z "$SNAPSHOT_0P5B" || -z "$SNAPSHOT_1P5B" ]]; then
  echo "both freshly materialized immutable ten-leaf model roots are required" >&2
  exit 64
fi
if [[ "$RUNTIME_IMAGE" != "runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404" ]]; then
  echo "G00F_RUNTIME_IMAGE differs from the frozen image" >&2
  exit 64
fi
if [[ "$VOLUME_ID" != "9mut3tpzwd" || "$DATA_CENTER" != "US-CA-2" ]]; then
  echo "G00-F requires the frozen network volume and data-center identity" >&2
  exit 64
fi

mapfile -t GPU_ROWS < <(nvidia-smi --query-gpu=name,uuid --format=csv,noheader,nounits)
if (( ${#GPU_ROWS[@]} != 4 )); then
  echo "G00-F frozen launcher requires exactly four visible H100 GPUs" >&2
  exit 69
fi
declare -A GPU_UUIDS=()
for row in "${GPU_ROWS[@]}"; do
  if [[ "$row" != NVIDIA\ H100* ]]; then
    echo "G00-F frozen launcher saw a non-H100 GPU: $row" >&2
    exit 69
  fi
  gpu_uuid="${row##*, }"
  if [[ -n "${GPU_UUIDS[$gpu_uuid]:-}" ]]; then
    echo "G00-F frozen launcher saw a duplicate GPU UUID" >&2
    exit 69
  fi
  GPU_UUIDS[$gpu_uuid]=1
done

export PYTHONPATH="$ROOT/src"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
CLI=("$PYTHON_BIN" -m goalzendo_g00f.cli)
COMMON=(--repo "$ROOT" --freeze "$FREEZE" --expected-freeze-sha256 "$EXPECTED_FREEZE_SHA256")
LEDGER_ROOT="$EXECUTION_ROOT/itt-ledger/$EXECUTION_UUID"
MODEL_RECEIPT_0P5B="$EXECUTION_ROOT/model-receipt-0p5b.json"
MODEL_RECEIPT_1P5B="$EXECUTION_ROOT/model-receipt-1p5b.json"
MODEL_INTEGRATION_AUDIT_0P5B="$EXECUTION_ROOT/model-integration-audit-0p5b.json"
MODEL_INTEGRATION_AUDIT_1P5B="$EXECUTION_ROOT/model-integration-audit-1p5b.json"
PID_FILE="$EXECUTION_ROOT/worker-pids.txt"
WATCHDOG_FIRED="$EXECUTION_ROOT/watchdog-deadline-fired.json"
WATCHDOG_STARTED="$EXECUTION_ROOT/watchdog-started.json"
WATCHDOG_NORMAL_STOP="$EXECUTION_ROOT/watchdog-normal-stop.json"
TIMEOUT_CANCEL="$EXECUTION_ROOT/coordinated-timeout-cancel.json"
TIMEOUT_KILL="$EXECUTION_ROOT/coordinated-timeout-kill.json"
OPERATIONAL_CANCEL="$EXECUTION_ROOT/coordinated-operational-cancel.json"
BOUND_PROVISION_RECEIPT="$EXECUTION_ROOT/runpod-provision-receipt.json"

# Image/DC/volume facts cannot be recovered from CUDA.  Bind the exact bytes
# captured by the independent Runpod API verifier before ledger creation or
# any model load; every subsequent receipt replays this external SHA boundary.
"${CLI[@]}" bind-provision-receipt "${COMMON[@]}" \
  --input "$EXTERNAL_PROVISION_RECEIPT" \
  --output "$BOUND_PROVISION_RECEIPT" \
  --expected-provision-receipt-sha256 "$EXPECTED_PROVISION_SHA256" \
  --expected-pod-id "$RUNPOD_POD_ID"

# Hash all direct model leaves before allocating an ITT ledger or loading any
# model weights.  The separate materialize-model command must already have
# produced these fresh, read-only, no-link roots.
"${CLI[@]}" model-receipt "${COMMON[@]}" \
  --panel-id g00f-0p5b --snapshot-root "$SNAPSHOT_0P5B" --output "$MODEL_RECEIPT_0P5B"
"${CLI[@]}" model-receipt "${COMMON[@]}" \
  --panel-id g00f-1p5b --snapshot-root "$SNAPSHOT_1P5B" --output "$MODEL_RECEIPT_1P5B"

# Load each authenticated base revision, with frozen updates, solely to run
# the legacy two-prompt A/B continuation-enumeration audit.  Both exact
# 2e-3 swap tolerances must pass before the ITT ledger or budget can exist.
INTEGRATION_GPU_0P5B="${GPU_ROWS[0]##*, }"
INTEGRATION_GPU_1P5B="${GPU_ROWS[1]##*, }"
CUDA_VISIBLE_DEVICES="$INTEGRATION_GPU_0P5B" "${CLI[@]}" model-integration-audit "${COMMON[@]}" \
  --panel-id g00f-0p5b --model-receipt "$MODEL_RECEIPT_0P5B" \
  --output "$MODEL_INTEGRATION_AUDIT_0P5B"
CUDA_VISIBLE_DEVICES="$INTEGRATION_GPU_1P5B" "${CLI[@]}" model-integration-audit "${COMMON[@]}" \
  --panel-id g00f-1p5b --model-receipt "$MODEL_RECEIPT_1P5B" \
  --output "$MODEL_INTEGRATION_AUDIT_1P5B"

"${CLI[@]}" initialize-ledger "${COMMON[@]}" \
  --execution-uuid "$EXECUTION_UUID" --ledger-root "$LEDGER_ROOT" \
  --provision-receipt "$BOUND_PROVISION_RECEIPT" \
  --expected-provision-receipt-sha256 "$EXPECTED_PROVISION_SHA256" \
  --expected-pod-id "$RUNPOD_POD_ID" \
  --model-integration-audit-0p5b "$MODEL_INTEGRATION_AUDIT_0P5B" \
  --model-integration-audit-1p5b "$MODEL_INTEGRATION_AUDIT_1P5B"

declare -a WORKER_PIDS=()
WATCHDOG_PID=""
SUCCESS=0
RECONCILED=0

signal_workers() {
  local signal_name="$1"
  local pid
  for pid in "${WORKER_PIDS[@]:-}"; do
    kill "-$signal_name" "$pid" 2>/dev/null || true
  done
}

stop_workers() {
  signal_workers TERM
  sleep 2
  signal_workers KILL
}

reconcile_failure() {
  local trigger="$1"
  local error_type="$2"
  if (( RECONCILED == 1 )) || [[ -f "$WATCHDOG_FIRED" ]]; then
    return
  fi
  RECONCILED=1
  stop_workers
  "${CLI[@]}" reconcile-failure "${COMMON[@]}" \
    --ledger-root "$LEDGER_ROOT" \
    --trigger "$trigger" \
    --error-type "$error_type" \
    --cancel-receipt "$OPERATIONAL_CANCEL" || true
}

cleanup() {
  local exit_code=$?
  trap - EXIT TERM INT
  if (( SUCCESS == 0 )) && [[ -f "$LEDGER_ROOT/ledger.json" ]] && [[ ! -f "$WATCHDOG_FIRED" ]]; then
    if (( ${#WORKER_PIDS[@]} < 4 )); then
      reconcile_failure launcher_partial_start LauncherPartialStart
    else
      reconcile_failure coordinator_exit_trap LauncherExit
    fi
  fi
  if [[ -n "$WATCHDOG_PID" ]]; then
    kill -TERM "$WATCHDOG_PID" 2>/dev/null || true
    wait "$WATCHDOG_PID" 2>/dev/null || true
  fi
  exit "$exit_code"
}

handle_signal() {
  local signal_name="$1"
  reconcile_failure launcher_signal "Launcher${signal_name}"
  exit 130
}

trap cleanup EXIT
trap 'handle_signal TERM' TERM
trap 'handle_signal INT' INT

: > "$PID_FILE"
"$PYTHON_BIN" "$ROOT/runs/goalzendo/g00f_watchdog.py" \
  --budget-start "$LEDGER_ROOT/budget-start.json" \
  --pid-file "$PID_FILE" \
  --started-receipt "$WATCHDOG_STARTED" \
  --normal-stop-receipt "$WATCHDOG_NORMAL_STOP" \
  --fired-receipt "$WATCHDOG_FIRED" \
  --cancel-receipt "$TIMEOUT_CANCEL" \
  --kill-receipt "$TIMEOUT_KILL" \
  -- "${CLI[@]}" record-timeout "${COMMON[@]}" --ledger-root "$LEDGER_ROOT" &
WATCHDOG_PID="$!"

# Do not start a worker until the independently authenticated watchdog has
# validated the budget receipt and persisted its own lifecycle start.  A dead
# or nonresponsive watchdog is an operational failure, never a reason to run
# without the 14-hour ceiling.
for _watchdog_probe in {1..100}; do
  if [[ -f "$WATCHDOG_STARTED" ]]; then
    break
  fi
  if ! kill -0 "$WATCHDOG_PID" 2>/dev/null; then
    wait "$WATCHDOG_PID" 2>/dev/null || true
    WATCHDOG_PID=""
    reconcile_failure watchdog_process_exit WatchdogStartupExit
    exit 70
  fi
  sleep 0.1
done
if [[ ! -f "$WATCHDOG_STARTED" ]]; then
  reconcile_failure watchdog_process_exit WatchdogStartupTimeout
  exit 70
fi

for worker in 0 1 2 3; do
  IFS=',' read -r gpu_name gpu_uuid <<< "${GPU_ROWS[$worker]}"
  gpu_name="${gpu_name% }"
  gpu_uuid="${gpu_uuid# }"
  launch_receipt="$EXECUTION_ROOT/worker-$worker-launch.json"
  result_receipt="$EXECUTION_ROOT/worker-$worker-result.json"
  CUDA_VISIBLE_DEVICES="$gpu_uuid" "${CLI[@]}" launch-receipt "${COMMON[@]}" \
    --worker-index "$worker" \
    --ledger-root "$LEDGER_ROOT" \
    --bundle-receipt "$BUNDLE_RECEIPT" \
    --provision-receipt "$BOUND_PROVISION_RECEIPT" \
    --expected-provision-receipt-sha256 "$EXPECTED_PROVISION_SHA256" \
    --expected-pod-id "$RUNPOD_POD_ID" \
    --image "$RUNTIME_IMAGE" \
    --network-volume-id "$VOLUME_ID" \
    --data-center "$DATA_CENTER" \
    --gpu-name "$gpu_name" \
    --gpu-uuid "$gpu_uuid" \
    --output "$launch_receipt"
  CUDA_VISIBLE_DEVICES="$gpu_uuid" "${CLI[@]}" run-worker "${COMMON[@]}" \
    --worker-index "$worker" \
    --ledger-root "$LEDGER_ROOT" \
    --launch-receipt "$launch_receipt" \
    --model-receipt-0p5b "$MODEL_RECEIPT_0P5B" \
    --model-receipt-1p5b "$MODEL_RECEIPT_1P5B" \
    --result-output "$result_receipt" \
    >"$EXECUTION_ROOT/worker-$worker.log" 2>&1 &
  WORKER_PIDS+=("$!")
done
printf '%s\n' "${WORKER_PIDS[*]}" > "$PID_FILE"

declare -a ACTIVE_PIDS=("${WORKER_PIDS[@]}")
while (( ${#ACTIVE_PIDS[@]} > 0 )); do
  completed_pid=""
  WAIT_PIDS=("${ACTIVE_PIDS[@]}" "$WATCHDOG_PID")
  if wait -n -p completed_pid "${WAIT_PIDS[@]}"; then
    worker_exit=0
  else
    worker_exit=$?
  fi
  if [[ "$completed_pid" == "$WATCHDOG_PID" ]]; then
    WATCHDOG_PID=""
    if [[ -f "$WATCHDOG_FIRED" ]]; then
      exit 124
    fi
    reconcile_failure watchdog_process_exit WatchdogProcessExit
    exit 70
  fi
  declare -a REMAINING_PIDS=()
  for pid in "${ACTIVE_PIDS[@]}"; do
    if [[ "$pid" != "$completed_pid" ]]; then
      REMAINING_PIDS+=("$pid")
    fi
  done
  ACTIVE_PIDS=("${REMAINING_PIDS[@]}")
  if (( worker_exit != 0 )); then
    if [[ -f "$WATCHDOG_FIRED" ]]; then
      wait "$WATCHDOG_PID" 2>/dev/null || true
    else
      reconcile_failure worker_nonzero_exit WorkerProcessExit
    fi
    exit "$worker_exit"
  fi
done

if ! kill -TERM "$WATCHDOG_PID" 2>/dev/null; then
  wait "$WATCHDOG_PID" 2>/dev/null || true
  WATCHDOG_PID=""
  reconcile_failure watchdog_process_exit WatchdogMissingAtNormalStop
  exit 70
fi
if ! wait "$WATCHDOG_PID" || [[ ! -f "$WATCHDOG_NORMAL_STOP" ]]; then
  WATCHDOG_PID=""
  reconcile_failure watchdog_process_exit WatchdogNormalStopFailure
  exit 70
fi
WATCHDOG_PID=""
SUCCESS=1
trap - EXIT TERM INT
printf '%s\n' "$EXECUTION_UUID" > "$EXECUTION_ROOT/EXECUTION_UUID"
echo "G00-F execution complete; evaluate the frozen gate before any separate G01 bridge review"
echo "G00F_EXECUTION_UUID=$EXECUTION_UUID"
echo "G00F_EXECUTION_ROOT=$EXECUTION_ROOT"
