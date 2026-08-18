#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/$(basename "${BASH_SOURCE[0]}")"
AMBIENT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
WORK_ROOT="${GOALZENDO_WORK_ROOT:-/workspace}"
PYTHON_BIN="${G00F_PYTHON_BIN:-$WORK_ROOT/.venvs/goalzendo/bin/python}"
FREEZE="${G00F_H200_FREEZE:-$AMBIENT_ROOT/reproducibility/goalzendo/g00f-h200-execution-freeze-20260811/execution-freeze.json}"
ARCHIVE="${G00F_H200_SOURCE_ARCHIVE:-$AMBIENT_ROOT/reproducibility/goalzendo/g00f-h200-execution-freeze-20260811/g00f-h200-execution-source.tar.gz}"
MANIFEST="${G00F_H200_SOURCE_MANIFEST:-$AMBIENT_ROOT/reproducibility/goalzendo/g00f-h200-execution-freeze-20260811/g00f-h200-source-bundle-manifest.json}"
EXPECTED_FREEZE_SHA256="${G00F_EXPECTED_FREEZE_SHA256:-}"
SNAPSHOT_0P5B="${G00F_SNAPSHOT_0P5B:-}"
SNAPSHOT_1P5B="${G00F_SNAPSHOT_1P5B:-}"
RUNTIME_IMAGE="${G00F_RUNTIME_IMAGE:-}"
VOLUME_ID="${G00F_NETWORK_VOLUME_ID:-}"
DATA_CENTER="${G00F_DATA_CENTER:-}"
EXTERNAL_PROVISION_RECEIPT="${G00F_RUNPOD_PROVISION_RECEIPT:-}"
EXPECTED_PROVISION_SHA256="${G00F_EXPECTED_PROVISION_RECEIPT_SHA256:-}"
EXPECTED_POD_ID="${G00F_EXPECTED_POD_ID:-}"

if [[ ! "$EXPECTED_FREEZE_SHA256" =~ ^[0-9a-f]{64}$ ]]; then
  echo "G00F_EXPECTED_FREEZE_SHA256 must be an externally supplied SHA-256" >&2
  exit 64
fi
if [[ -z "$EXTERNAL_PROVISION_RECEIPT" || ! "$EXPECTED_PROVISION_SHA256" =~ ^[0-9a-f]{64}$ || -z "$EXPECTED_POD_ID" ]]; then
  echo "an externally SHA-pinned Runpod provision receipt and G00F_EXPECTED_POD_ID are required" >&2
  exit 64
fi
if [[ "${RUNPOD_POD_ID:-}" != "$EXPECTED_POD_ID" ]]; then
  echo "provider-injected RUNPOD_POD_ID differs from the authenticated operator handoff" >&2
  exit 65
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
  BOOTSTRAP="$AMBIENT_ROOT/runs/goalzendo/g00f_h200_bundle_bootstrap.py"
  DETACHED_SUPERVISOR="$AMBIENT_ROOT/runs/goalzendo/g00f_h200_detached_supervisor.py"
  QUALIFICATION_CONTROLLER="$AMBIENT_ROOT/runs/goalzendo/g00f_h200_qualification_controller.py"
  QUALIFICATION_SUPERVISOR="$AMBIENT_ROOT/runs/goalzendo/g00f_h200_qualification_supervisor.py"
  WATCHDOG="$AMBIENT_ROOT/runs/goalzendo/g00f_h200_watchdog.py"
  EXECUTION_UUID="${G00F_EXECUTION_UUID:-}"
  if [[ -z "$EXECUTION_UUID" ]]; then
    echo "G00F_EXECUTION_UUID must come from the independently pinned operator handoff" >&2
    exit 64
  fi
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
    --actual-detached-supervisor "$DETACHED_SUPERVISOR" \
    --actual-launcher "$SCRIPT_PATH" \
    --actual-qualification-controller "$QUALIFICATION_CONTROLLER" \
    --actual-qualification-supervisor "$QUALIFICATION_SUPERVISOR" \
    --actual-watchdog "$WATCHDOG"
  export G00F_FROZEN_REEXEC=1
  export G00F_EXECUTION_UUID="$EXECUTION_UUID"
  export G00F_EXECUTION_ROOT="$EXECUTION_ROOT"
  export G00F_EXTRACTED_ROOT="$EXTRACTED_ROOT"
  export G00F_BUNDLE_RECEIPT="$BUNDLE_RECEIPT"
  export G00F_H200_FREEZE="$FREEZE"
  export G00F_H200_SOURCE_ARCHIVE="$ARCHIVE"
  export G00F_H200_SOURCE_MANIFEST="$MANIFEST"
  exec "$EXTRACTED_ROOT/runs/goalzendo/run_g00f_frozen_4h200.sh"
fi

ROOT="${G00F_EXTRACTED_ROOT:?authenticated extracted root is absent}"
EXECUTION_UUID="${G00F_EXECUTION_UUID:?execution UUID is absent}"
EXECUTION_ROOT="${G00F_EXECUTION_ROOT:?execution root is absent}"
BUNDLE_RECEIPT="${G00F_BUNDLE_RECEIPT:?source bundle receipt is absent}"
EXPECTED_SCRIPT="$ROOT/runs/goalzendo/run_g00f_frozen_4h200.sh"
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
  echo "G00-F frozen launcher requires exactly four visible H200 GPUs" >&2
  exit 69
fi
declare -A GPU_UUIDS=()
for row in "${GPU_ROWS[@]}"; do
  if [[ "$row" != NVIDIA\ H200* ]]; then
    echo "G00-F frozen launcher saw a non-H200 GPU: $row" >&2
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
CLI=("$PYTHON_BIN" -m goalzendo_g00f_h200.cli)
COMMON=(--repo "$ROOT" --freeze "$FREEZE" --expected-freeze-sha256 "$EXPECTED_FREEZE_SHA256")
LEDGER_ROOT="$EXECUTION_ROOT/itt-ledger/$EXECUTION_UUID"
MODEL_RECEIPT_0P5B="$EXECUTION_ROOT/model-receipt-0p5b.json"
MODEL_RECEIPT_1P5B="$EXECUTION_ROOT/model-receipt-1p5b.json"
MODEL_INTEGRATION_AUDIT_0P5B="$EXECUTION_ROOT/model-integration-audit-0p5b.json"
MODEL_INTEGRATION_AUDIT_1P5B="$EXECUTION_ROOT/model-integration-audit-1p5b.json"
QUALIFICATION_ENGINEERING_ROOT="$WORK_ROOT/status-goalzendo/g00f-h200-engineering/g00f-h200-profile-qualification-$EXECUTION_UUID"
QUALIFICATION_PRODUCER_RECEIPT="$QUALIFICATION_ENGINEERING_ROOT/qualification-producer-receipt.json"
QUALIFICATION_HANDOFF="$EXECUTION_ROOT/h200-profile-qualification-handoff.json"
PROFILE_SELECTION="$EXECUTION_ROOT/h200-profile-selection.json"
QUALIFICATION_CLEANUP="$EXECUTION_ROOT/h200-profile-qualification-cleanup.json"
QUALIFICATION_SUPERVISOR_STARTED="$EXECUTION_ROOT/qualification-supervisor-started.json"
QUALIFICATION_SUPERVISOR_TERM="$EXECUTION_ROOT/qualification-supervisor-term.json"
QUALIFICATION_SUPERVISOR_KILL="$EXECUTION_ROOT/qualification-supervisor-kill.json"
QUALIFICATION_SUPERVISOR_TERMINAL="$EXECUTION_ROOT/qualification-supervisor-terminal.json"
STORAGE_PREFLIGHT_BEFORE_QUALIFICATION="$EXECUTION_ROOT/storage-preflight-before-qualification.json"
STORAGE_PREFLIGHT_BEFORE_ITT="$EXECUTION_ROOT/storage-preflight-before-itt.json"
PID_FILE="$EXECUTION_ROOT/worker-pids.txt"
RECONCILIATION_LOCK="$EXECUTION_ROOT/reconciliation-owner.lock"
WATCHDOG_CLAIMED="$EXECUTION_ROOT/watchdog-deadline-claimed.json"
WATCHDOG_FIRED="$EXECUTION_ROOT/watchdog-deadline-fired.json"
WATCHDOG_STARTED="$EXECUTION_ROOT/watchdog-started.json"
WATCHDOG_NORMAL_STOP="$EXECUTION_ROOT/watchdog-normal-stop.json"
WATCHDOG_GUARDIAN_TERMINAL="$EXECUTION_ROOT/watchdog-guardian-terminal.json"
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
  --expected-pod-id "$EXPECTED_POD_ID"

# Refuse model loading and the high-volume qualification before proving that
# the exact authenticated /workspace volume has the frozen byte/inode reserve.
if [[ -e "$QUALIFICATION_ENGINEERING_ROOT" ]]; then
  echo "qualification engineering root already exists; refusing an ambiguous retry" >&2
  exit 65
fi
mkdir -p "$(dirname "$QUALIFICATION_ENGINEERING_ROOT")"
"${CLI[@]}" record-storage-preflight "${COMMON[@]}" \
  --execution-uuid "$EXECUTION_UUID" --execution-root "$EXECUTION_ROOT" \
  --expected-provision-receipt-sha256 "$EXPECTED_PROVISION_SHA256" \
  --expected-pod-id "$EXPECTED_POD_ID" \
  --phase before_qualification --output "$STORAGE_PREFLIGHT_BEFORE_QUALIFICATION"

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

# The authenticated controller is the only producer of actual-model profile
# qualification evidence. It covers both profiles, both panels/Laws, primary
# and replay processes, and all four H200 UUIDs before any ITT row exists.
"$PYTHON_BIN" "$ROOT/runs/goalzendo/g00f_h200_qualification_supervisor.py" \
  --execution-uuid "$EXECUTION_UUID" \
  --execution-root "$EXECUTION_ROOT" \
  --started-receipt "$QUALIFICATION_SUPERVISOR_STARTED" \
  --term-receipt "$QUALIFICATION_SUPERVISOR_TERM" \
  --kill-receipt "$QUALIFICATION_SUPERVISOR_KILL" \
  --terminal-receipt "$QUALIFICATION_SUPERVISOR_TERMINAL" \
  -- "$PYTHON_BIN" "$ROOT/runs/goalzendo/g00f_h200_qualification_controller.py" \
  --mode controller \
  --repo "$ROOT" \
  --freeze "$FREEZE" \
  --freeze-sha256 "$EXPECTED_FREEZE_SHA256" \
  --execution-root "$EXECUTION_ROOT" \
  --engineering-root "$QUALIFICATION_ENGINEERING_ROOT" \
  --provision-receipt "$BOUND_PROVISION_RECEIPT" \
  --provision-receipt-sha256 "$EXPECTED_PROVISION_SHA256" \
  --pod-id "$EXPECTED_POD_ID" \
  --model-receipt-0p5b "$MODEL_RECEIPT_0P5B" \
  --model-receipt-1p5b "$MODEL_RECEIPT_1P5B" \
  --integration-audit-0p5b "$MODEL_INTEGRATION_AUDIT_0P5B" \
  --integration-audit-1p5b "$MODEL_INTEGRATION_AUDIT_1P5B" \
  --execution-uuid "$EXECUTION_UUID"

file_sha256() {
  "$PYTHON_BIN" - "$1" <<'PY'
import hashlib
import pathlib
import sys

target = pathlib.Path(sys.argv[1])
digest = hashlib.sha256()
with target.open("rb") as handle:
    for chunk in iter(lambda: handle.read(1024 * 1024), b""):
        digest.update(chunk)
print(digest.hexdigest())
PY
}

PRODUCER_RECEIPT_SHA256="$(file_sha256 "$QUALIFICATION_PRODUCER_RECEIPT")"
declare -a QUALIFICATION_GPU_ARGS=()
for row in "${GPU_ROWS[@]}"; do
  gpu_uuid="${row##*, }"
  QUALIFICATION_GPU_ARGS+=(--gpu-uuid "$gpu_uuid")
done
"${CLI[@]}" bind-profile-qualification "${COMMON[@]}" \
  --execution-uuid "$EXECUTION_UUID" \
  --producer-receipt "$QUALIFICATION_PRODUCER_RECEIPT" \
  --expected-producer-receipt-sha256 "$PRODUCER_RECEIPT_SHA256" \
  --expected-provision-receipt-sha256 "$EXPECTED_PROVISION_SHA256" \
  --expected-pod-id "$EXPECTED_POD_ID" \
  "${QUALIFICATION_GPU_ARGS[@]}" \
  --output "$QUALIFICATION_HANDOFF"
QUALIFICATION_HANDOFF_SHA256="$(file_sha256 "$QUALIFICATION_HANDOFF")"

"${CLI[@]}" select-profile "${COMMON[@]}" \
  --execution-uuid "$EXECUTION_UUID" \
  --qualification-handoff "$QUALIFICATION_HANDOFF" \
  --expected-qualification-handoff-sha256 "$QUALIFICATION_HANDOFF_SHA256" \
  --expected-provision-receipt-sha256 "$EXPECTED_PROVISION_SHA256" \
  --expected-pod-id "$EXPECTED_POD_ID" \
  "${QUALIFICATION_GPU_ARGS[@]}" \
  --output "$PROFILE_SELECTION"
PROFILE_SELECTION_SHA256="$(file_sha256 "$PROFILE_SELECTION")"
SELECTED_COMMON=(
  "${COMMON[@]}"
  --profile-selection "$PROFILE_SELECTION"
  --expected-profile-selection-sha256 "$PROFILE_SELECTION_SHA256"
  --expected-provision-receipt-sha256 "$EXPECTED_PROVISION_SHA256"
  --expected-pod-id "$EXPECTED_POD_ID"
)

"${CLI[@]}" record-storage-preflight "${COMMON[@]}" \
  --execution-uuid "$EXECUTION_UUID" --execution-root "$EXECUTION_ROOT" \
  --expected-provision-receipt-sha256 "$EXPECTED_PROVISION_SHA256" \
  --expected-pod-id "$EXPECTED_POD_ID" \
  --phase before_itt --output "$STORAGE_PREFLIGHT_BEFORE_ITT"

# The producer has already deleted every transient raw tensor/checkpoint. This
# receipt replays that exact inventory while retaining all compact evidence
# read-only through the final gate.
"${CLI[@]}" record-profile-qualification-cleanup "${SELECTED_COMMON[@]}" \
  --output "$QUALIFICATION_CLEANUP"
QUALIFICATION_CLEANUP_SHA256="$(file_sha256 "$QUALIFICATION_CLEANUP")"

"${CLI[@]}" initialize-ledger "${SELECTED_COMMON[@]}" \
  --execution-uuid "$EXECUTION_UUID" --ledger-root "$LEDGER_ROOT" \
  --provision-receipt "$BOUND_PROVISION_RECEIPT" \
  --model-integration-audit-0p5b "$MODEL_INTEGRATION_AUDIT_0P5B" \
  --model-integration-audit-1p5b "$MODEL_INTEGRATION_AUDIT_1P5B" \
  --profile-qualification-cleanup "$QUALIFICATION_CLEANUP" \
  --expected-profile-qualification-cleanup-sha256 "$QUALIFICATION_CLEANUP_SHA256"

declare -a WORKER_PIDS=()
WATCHDOG_PID=""
WATCHDOG_GUARDIAN_PID=""
WATCHDOG_DEADLINE_MARKED=0
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
  if (( WATCHDOG_DEADLINE_MARKED == 1 )); then
    if [[ -n "$WATCHDOG_PID" ]]; then
      wait "$WATCHDOG_PID" 2>/dev/null || true
      WATCHDOG_PID=""
    fi
    return
  fi
  if (( RECONCILED == 1 )) || [[ -f "$WATCHDOG_FIRED" ]]; then
    return
  fi
  if ! mkdir -m 0700 "$RECONCILIATION_LOCK" 2>/dev/null; then
    if [[ -n "$WATCHDOG_PID" ]]; then
      wait "$WATCHDOG_PID" 2>/dev/null || true
      WATCHDOG_PID=""
    fi
    if [[ -f "$WATCHDOG_FIRED" ]]; then
      return
    fi
    # The watchdog died after taking ownership but before a scientific
    # timeout receipt. It is now impossible for it to reconcile concurrently;
    # fail closed through the operational path under the existing owner lock.
  fi
  RECONCILED=1
  stop_workers
  "${CLI[@]}" reconcile-failure "${SELECTED_COMMON[@]}" \
    --ledger-root "$LEDGER_ROOT" \
    --trigger "$trigger" \
    --error-type "$error_type" \
    --cancel-receipt "$OPERATIONAL_CANCEL" || true
}

cleanup() {
  local exit_code=$?
  trap - EXIT TERM INT
  if (( SUCCESS == 0 && WATCHDOG_DEADLINE_MARKED == 1 )) && [[ -n "$WATCHDOG_PID" ]]; then
    wait "$WATCHDOG_PID" 2>/dev/null || true
    WATCHDOG_PID=""
  fi
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
  if (( WATCHDOG_DEADLINE_MARKED == 1 )) && [[ -n "$WATCHDOG_PID" ]]; then
    wait "$WATCHDOG_PID" 2>/dev/null || true
    WATCHDOG_PID=""
    if [[ -f "$WATCHDOG_FIRED" ]]; then
      exit 124
    fi
  fi
  reconcile_failure launcher_signal "Launcher${signal_name}"
  exit 130
}

handle_watchdog_deadline_marker() {
  WATCHDOG_DEADLINE_MARKED=1
  if [[ -n "$WATCHDOG_GUARDIAN_PID" ]]; then
    kill -USR1 "$WATCHDOG_GUARDIAN_PID" 2>/dev/null || true
  fi
}

trap cleanup EXIT
trap 'handle_signal TERM' TERM
trap 'handle_signal INT' INT
trap handle_watchdog_deadline_marker USR2

: > "$PID_FILE"
LAUNCHER_PID="$$"
LAUNCHER_PROCESS_GROUP_ID="$(/bin/ps -o pgid= -p "$LAUNCHER_PID" | tr -d '[:space:]')"
if [[ ! "$LAUNCHER_PROCESS_GROUP_ID" =~ ^[0-9]+$ ]] || (( LAUNCHER_PROCESS_GROUP_ID != LAUNCHER_PID )); then
  echo "authenticated launcher is not its detached process-group leader" >&2
  exit 70
fi
WATCHDOG_READY_DIR="$(mktemp -d "${TMPDIR:-/tmp}/goalzendo-g00f-h200-watchdog.XXXXXX")"
chmod 0700 "$WATCHDOG_READY_DIR"
WATCHDOG_READY_FIFO="$WATCHDOG_READY_DIR/ready.fifo"
mkfifo -m 0600 "$WATCHDOG_READY_FIFO"
# Opening the local FIFO read/write prevents the shell's open from blocking
# if the watchdog fails before exec.  FD 9 is explicitly closed in the child,
# so only this launcher owns the extra writer and the readiness boundary never
# depends on the network volume.
exec 9<>"$WATCHDOG_READY_FIFO"
"$PYTHON_BIN" "$ROOT/runs/goalzendo/g00f_h200_watchdog.py" \
  --budget-start "$LEDGER_ROOT/budget-start.json" \
  --pid-file "$PID_FILE" \
  --started-receipt "$WATCHDOG_STARTED" \
  --normal-stop-receipt "$WATCHDOG_NORMAL_STOP" \
  --owner-lock "$RECONCILIATION_LOCK" \
  --claim-receipt "$WATCHDOG_CLAIMED" \
  --fired-receipt "$WATCHDOG_FIRED" \
  --cancel-receipt "$TIMEOUT_CANCEL" \
  --kill-receipt "$TIMEOUT_KILL" \
  --guardian-terminal-receipt "$WATCHDOG_GUARDIAN_TERMINAL" \
  --ready-fd 1 \
  --launcher-pid "$LAUNCHER_PID" \
  --launcher-process-group-id "$LAUNCHER_PROCESS_GROUP_ID" \
  -- "${CLI[@]}" record-timeout "${SELECTED_COMMON[@]}" --ledger-root "$LEDGER_ROOT" \
  >"$WATCHDOG_READY_FIFO" 9>&- &
WATCHDOG_PID="$!"
WATCHDOG_READY_LINE=""
if ! IFS= read -r -t 10 -u 9 WATCHDOG_READY_LINE; then
  exec 9>&-
  rm -f "$WATCHDOG_READY_FIFO"
  rmdir "$WATCHDOG_READY_DIR"
  reconcile_failure watchdog_process_exit WatchdogGuardianReadinessFailure
  # Reconciliation is durable before the launcher exits.  The guardian, if
  # already armed, observes the exact launcher pidfd and clears its independent
  # watchdog group; otherwise the detached supervisor clears this launcher.
  WATCHDOG_PID=""
  exit 70
fi
exec 9>&-
rm -f "$WATCHDOG_READY_FIFO"
rmdir "$WATCHDOG_READY_DIR"
if [[ ! "$WATCHDOG_READY_LINE" =~ ^R\ ([1-9][0-9]*)$ ]]; then
  reconcile_failure watchdog_process_exit WatchdogGuardianReadinessInvalid
  WATCHDOG_PID=""
  exit 70
fi
WATCHDOG_GUARDIAN_PID="${BASH_REMATCH[1]}"

# Do not start a worker until the independently authenticated watchdog has
# validated the budget receipt and persisted its own lifecycle start.  A dead
# or nonresponsive watchdog is an operational failure, never a reason to run
# without the 14-hour ceiling.
WATCHDOG_STARTED_VERIFIED=0
for _watchdog_probe in {1..100}; do
  if [[ -f "$WATCHDOG_STARTED" ]] && "$PYTHON_BIN" - \
    "$WATCHDOG_STARTED" "$WATCHDOG_PID" "$WATCHDOG_GUARDIAN_PID" \
    "$LAUNCHER_PID" "$LAUNCHER_PROCESS_GROUP_ID" <<'PY'
import hashlib
import json
import pathlib
import sys

path = pathlib.Path(sys.argv[1])
metadata = path.lstat()
if path.is_symlink() or not path.is_file() or metadata.st_nlink != 1 or metadata.st_mode & 0o777 != 0o400:
    raise SystemExit(1)
payload = json.loads(path.read_text(encoding="utf-8"))
body = {key: value for key, value in payload.items() if key != "receipt_digest"}
encoded = json.dumps(
    body, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False
).encode("utf-8")
expected = {
    "schema": "goalzendo.g00f_h200_watchdog_started",
    "schema_version": 1,
    "watchdog_pid": int(sys.argv[2]),
    "guardian_pid": int(sys.argv[3]),
    "launcher_pid": int(sys.argv[4]),
    "launcher_process_group_id": int(sys.argv[5]),
    "guardian_protocol": "ready_pipe_pidfd_launcher_liveness_and_monotonic_group_cutoff_v1",
    "term_grace_seconds": 30,
    "watchdog_receipt_margin_seconds": 5,
    "deadline_marker_lead_ns": 2_000_000_000,
    "outcome_metrics_read": False,
    "predictions_read": False,
    "g01_launch_authorized": False,
}
if (
    any(payload.get(key) != value for key, value in expected.items())
    or payload.get("watchdog_process_group_id") != int(sys.argv[2])
    or payload.get("receipt_digest") != hashlib.sha256(encoded).hexdigest()
):
    raise SystemExit(1)
PY
  then
    if kill -0 "$WATCHDOG_PID" 2>/dev/null && kill -0 "$WATCHDOG_GUARDIAN_PID" 2>/dev/null; then
      WATCHDOG_STARTED_VERIFIED=1
      break
    fi
  fi
  if ! kill -0 "$WATCHDOG_PID" 2>/dev/null || ! kill -0 "$WATCHDOG_GUARDIAN_PID" 2>/dev/null; then
    wait "$WATCHDOG_PID" 2>/dev/null || true
    WATCHDOG_PID=""
    reconcile_failure watchdog_process_exit WatchdogStartupExit
    exit 70
  fi
  sleep 0.1
done
if (( WATCHDOG_STARTED_VERIFIED == 0 )); then
  reconcile_failure watchdog_process_exit WatchdogStartupTimeout
  WATCHDOG_PID=""
  exit 70
fi

for worker in 0 1 2 3; do
  IFS=',' read -r gpu_name gpu_uuid <<< "${GPU_ROWS[$worker]}"
  gpu_name="${gpu_name% }"
  gpu_uuid="${gpu_uuid# }"
  launch_receipt="$EXECUTION_ROOT/worker-$worker-launch.json"
  result_receipt="$EXECUTION_ROOT/worker-$worker-result.json"
  CUDA_VISIBLE_DEVICES="$gpu_uuid" "${CLI[@]}" launch-receipt "${SELECTED_COMMON[@]}" \
    --worker-index "$worker" \
    --ledger-root "$LEDGER_ROOT" \
    --bundle-receipt "$BUNDLE_RECEIPT" \
    --provision-receipt "$BOUND_PROVISION_RECEIPT" \
    --expected-provision-receipt-sha256 "$EXPECTED_PROVISION_SHA256" \
    --expected-pod-id "$EXPECTED_POD_ID" \
    --image "$RUNTIME_IMAGE" \
    --network-volume-id "$VOLUME_ID" \
    --data-center "$DATA_CENTER" \
    --gpu-name "$gpu_name" \
    --gpu-uuid "$gpu_uuid" \
    --output "$launch_receipt"
  CUDA_VISIBLE_DEVICES="$gpu_uuid" "${CLI[@]}" run-worker "${SELECTED_COMMON[@]}" \
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
  if [[ -z "$completed_pid" && "$WATCHDOG_DEADLINE_MARKED" == "1" ]]; then
    continue
  fi
  if [[ -z "$completed_pid" ]]; then
    reconcile_failure coordinator_wait_failure CoordinatorWaitFailure
    exit 70
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
    if [[ -f "$WATCHDOG_CLAIMED" ]]; then
      wait "$WATCHDOG_PID" 2>/dev/null || true
      WATCHDOG_PID=""
      if [[ -f "$WATCHDOG_FIRED" ]]; then
        exit 124
      fi
      reconcile_failure watchdog_process_exit WatchdogClaimWithoutTimeoutReceipt
      exit 70
    elif [[ -f "$WATCHDOG_FIRED" ]]; then
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
if ! wait "$WATCHDOG_PID" || [[ ! -f "$WATCHDOG_NORMAL_STOP" || ! -f "$WATCHDOG_GUARDIAN_TERMINAL" ]]; then
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
