#!/usr/bin/env bash
set -Eeuo pipefail

# Outcome-agnostic postcompletion wrapper for the exact 24-run hidden-law panel.
# Stage this file beside the frozen source archive, pass its independently
# verified SHA-256 in POSTCOMPLETION_WRAPPER_SHA256, and tear the pod down only
# after the terminal marker appears.

JOB_ID=qwen35-hidden-law-20260815T153604Z
NUM_SHARDS=6
PLANNED_RUNS=24
ARCHIVE_SHA256=56700896a4f62f6e92a495cc0af109ff1f8982c8e9be57b920de6737a3b919f8
CONFIG_SHA256=a2e31815f3fbe1d65fcd64654f8fb9e51a2d950cf8a2985cc5d90121bad3e53a
SCIENTIFIC_CONFIG_DIGEST=d61cad537e4ae3ea439250bc5e7e2466405e5331be251f6b8f15d0de8683207b
IMPLEMENTATION_FINGERPRINT=9cd679289fc22ceeb5fe1a9cbc2ca134051faf8491b9fa478be9b13ca81dd087
ANALYZER_SHA256=8bedacc9b5935e67224676101e12692bb0a5b913f1e9de180a033ac20f0a2d8c
RUN_ID_SET_SHA256=3bfede4f8f303f3bfc2923dda3ff879003532a606ed1fb4b8014d7c140aaa1fe
ARCHIVE_NAME=goalzendo-hidden-law-src.tgz
ARCHIVE_ROOT_NAME=goalzendo-hidden-law-src
ANALYSIS_NAME=qwen35-hidden-law-analysis.json

JOB=/workspace/goalzendo-hidden-law/jobs/$JOB_ID
ARCHIVE=$JOB/input/$ARCHIVE_NAME
ARTIFACT_ROOT=$JOB/artifacts/main
ANALYSIS_DIR=$JOB/analysis
STATUS_DIR=$JOB/status
LOG_DIR=$JOB/logs
VENV=/workspace/goalzendo-qwen35/jobs/qwen35-pilot-20260813T134449Z/work/.venv
PYTHON=$VENV/bin/python
HF_CACHE=/workspace/goalzendo-qwen35/jobs/qwen35-pilot-20260813T134449Z/work/huggingface
STARTED_AT=$(date -u +%Y-%m-%dT%H:%M:%SZ)
TERMINAL_SHARD_RECEIPT_COUNT=0
VERIFIED_COMPLETE_RUN_COUNT=0
ANALYSIS_SHA256=null
EXTRACT_ROOT=
STAGING=
PAYLOAD=

: "${POSTCOMPLETION_WRAPPER_SHA256:?set POSTCOMPLETION_WRAPPER_SHA256 to the staged file digest}"
mkdir -p "$STATUS_DIR" "$LOG_DIR"
exec > >(tee -a "$LOG_DIR/analysis.log") 2>&1

write_status() {
  local path=$1 state=$2 runner_rc=$3 finished_at=$4 analysis_sha=$5
  local tmp="$path.tmp.$$"
  python3 - "$tmp" "$state" "$runner_rc" "$finished_at" "$analysis_sha" <<'PY'
import json
import sys

(
    target,
    state,
    runner_rc,
    finished_at,
    analysis_sha256,
) = sys.argv[1:]
value = {
    "schema": "goalzendo.qwen35_hidden_law_postcompletion_status",
    "schema_version": 1,
    "status_scope": "postlaunch_operational",
    "state": state,
    "runner_rc": None if runner_rc == "null" else int(runner_rc),
    "job_id": "qwen35-hidden-law-20260815T153604Z",
    "terminal_shard_receipt_count": int(
        __import__("os").environ["TERMINAL_SHARD_RECEIPT_COUNT"]
    ),
    "verified_complete_run_count": int(
        __import__("os").environ["VERIFIED_COMPLETE_RUN_COUNT"]
    ),
    "archive_sha256": "56700896a4f62f6e92a495cc0af109ff1f8982c8e9be57b920de6737a3b919f8",
    "config_sha256": "a2e31815f3fbe1d65fcd64654f8fb9e51a2d950cf8a2985cc5d90121bad3e53a",
    "scientific_config_digest": "d61cad537e4ae3ea439250bc5e7e2466405e5331be251f6b8f15d0de8683207b",
    "implementation_fingerprint": "9cd679289fc22ceeb5fe1a9cbc2ca134051faf8491b9fa478be9b13ca81dd087",
    "analyzer_sha256": "8bedacc9b5935e67224676101e12692bb0a5b913f1e9de180a033ac20f0a2d8c",
    "postcompletion_wrapper_sha256": __import__("os").environ[
        "POSTCOMPLETION_WRAPPER_SHA256"
    ],
    "analysis_sha256": None if analysis_sha256 == "null" else analysis_sha256,
    "started_at": __import__("os").environ["STARTED_AT"],
    "finished_at": None if finished_at == "null" else finished_at,
}
with open(target, "wb") as handle:
    handle.write(
        (json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()
    )
PY
  mv "$tmp" "$path"
}

clean_temporary_paths() {
  if [[ -n "$PAYLOAD" && "$PAYLOAD" == "$JOB"/.analysis-payload.* ]]; then
    rm -f -- "$PAYLOAD"
  fi
  if [[ -n "$STAGING" && "$STAGING" == "$JOB"/.analysis.staging.* && -d "$STAGING" ]]; then
    rm -rf -- "$STAGING"
  fi
  if [[ -n "$EXTRACT_ROOT" && "$EXTRACT_ROOT" == "$JOB"/.analysis-source.* && -d "$EXTRACT_ROOT" ]]; then
    rm -rf -- "$EXTRACT_ROOT"
  fi
}

finish() {
  local rc=$?
  set +e
  trap - EXIT TERM INT
  local state finished terminal_status
  finished=$(date -u +%Y-%m-%dT%H:%M:%SZ)
  if [[ $rc -eq 0 ]]; then
    state=complete
    terminal_status=$ANALYSIS_DIR/status.json
  else
    state=failed
    terminal_status=$STATUS_DIR/.analysis.failed-status.$$
    export TERMINAL_SHARD_RECEIPT_COUNT VERIFIED_COMPLETE_RUN_COUNT STARTED_AT
    write_status "$terminal_status" "$state" "$rc" "$finished" null
  fi
  if [[ -f "$terminal_status" ]]; then
    cp "$terminal_status" "$STATUS_DIR/.analysis.status.$$"
    mv "$STATUS_DIR/.analysis.status.$$" "$STATUS_DIR/analysis.status.json"
    cp "$terminal_status" "$STATUS_DIR/.analysis.$state.$$"
    mv "$STATUS_DIR/.analysis.$state.$$" "$STATUS_DIR/analysis.$state.json"
  fi
  rm -f -- "$STATUS_DIR/.analysis.failed-status.$$"
  clean_temporary_paths
  sync
  echo "GOALZENDO_HIDDEN_LAW_ANALYSIS_TERMINAL state=$state rc=$rc finished_at=$finished"
  sleep infinity
}
trap finish EXIT
trap 'exit 143' TERM INT

export TERMINAL_SHARD_RECEIPT_COUNT VERIFIED_COMPLETE_RUN_COUNT STARTED_AT
write_status "$STATUS_DIR/analysis.status.json" running null null null
echo "GOALZENDO_HIDDEN_LAW_ANALYSIS_START started_at=$STARTED_AT"

observed_wrapper_sha=$(sha256sum "$0" | awk '{print $1}')
[[ "$observed_wrapper_sha" == "$POSTCOMPLETION_WRAPPER_SHA256" ]] || {
  echo "postcompletion wrapper digest mismatch" >&2
  exit 66
}
test -x "$PYTHON"
test -d "$HF_CACHE"
test -f "$ARCHIVE"

"$PYTHON" - "$STATUS_DIR" <<'PY'
import json
import sys
from pathlib import Path

status = Path(sys.argv[1])
job_id = "qwen35-hidden-law-20260815T153604Z"
num_shards = 6
expected_names = {f"shard-{index:02d}.complete.json" for index in range(num_shards)}
observed_names = {path.name for path in status.glob("shard-*.complete.json")}
if observed_names != expected_names:
    raise SystemExit("terminal shard receipt inventory is not exactly six")

fixed = {
    "schema": "goalzendo.hidden_law_shard_status",
    "schema_version": 1,
    "num_shards": num_shards,
    "state": "complete",
    "runner_rc": 0,
    "archive_sha256": "56700896a4f62f6e92a495cc0af109ff1f8982c8e9be57b920de6737a3b919f8",
    "config_sha256": "a2e31815f3fbe1d65fcd64654f8fb9e51a2d950cf8a2985cc5d90121bad3e53a",
    "scientific_config_digest": "d61cad537e4ae3ea439250bc5e7e2466405e5331be251f6b8f15d0de8683207b",
    "implementation_fingerprint": "9cd679289fc22ceeb5fe1a9cbc2ca134051faf8491b9fa478be9b13ca81dd087",
}
expected_keys = {*fixed, "job_id", "shard_index", "started_at", "finished_at"}
for index in range(num_shards):
    terminal = status / f"shard-{index:02d}.complete.json"
    current = status / f"shard-{index:02d}.status.json"
    try:
        receipt = json.loads(terminal.read_text(encoding="utf-8"))
        current_receipt = json.loads(current.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise SystemExit(f"cannot read exact shard receipt {index}") from exc
    expected = {**fixed, "job_id": job_id, "shard_index": index}
    if type(receipt) is not dict or set(receipt) != expected_keys:
        raise SystemExit(f"shard {index} receipt schema differs")
    if any(receipt.get(key) != value for key, value in expected.items()):
        raise SystemExit(f"shard {index} receipt differs from the frozen bindings")
    if type(receipt["runner_rc"]) is not int or receipt["runner_rc"] != 0:
        raise SystemExit(f"shard {index} receipt is not rc0")
    if not all(type(receipt.get(key)) is str and receipt[key] for key in ("started_at", "finished_at")):
        raise SystemExit(f"shard {index} receipt lacks terminal timestamps")
    if current_receipt != receipt:
        raise SystemExit(f"shard {index} current status differs from its terminal receipt")
print('{"terminal_shard_receipt_count":6}')
PY
TERMINAL_SHARD_RECEIPT_COUNT=$NUM_SHARDS
export TERMINAL_SHARD_RECEIPT_COUNT
echo "GOALZENDO_HIDDEN_LAW_RECEIPTS_VERIFIED count=$TERMINAL_SHARD_RECEIPT_COUNT"

observed_archive_sha=$(sha256sum "$ARCHIVE" | awk '{print $1}')
[[ "$observed_archive_sha" == "$ARCHIVE_SHA256" ]] || {
  echo "source archive digest mismatch" >&2
  exit 66
}
EXTRACT_ROOT=$(mktemp -d "$JOB/.analysis-source.XXXXXX")
tar --no-same-owner -xzf "$ARCHIVE" -C "$EXTRACT_ROOT"
SRC=$EXTRACT_ROOT/$ARCHIVE_ROOT_NAME
test -d "$SRC/src/goalzendo_hidden_law"

sha_check() {
  local expected=$1 path=$2 label=$3 observed
  observed=$(sha256sum "$path" | awk '{print $1}')
  [[ "$observed" == "$expected" ]] || {
    echo "$label digest mismatch" >&2
    exit 66
  }
}
sha_check "$CONFIG_SHA256" "$SRC/configs/goalzendo/qwen35_hidden_law_finite_choice.yaml" config
sha_check "$ANALYZER_SHA256" "$SRC/scripts/analyze_qwen35_hidden_law.py" analyzer

export PYTHONPATH=$SRC/src
export PYTHONNOUSERSITE=1
export HF_HOME=$HF_CACHE
export HUGGINGFACE_HUB_CACHE=$HF_CACHE/hub
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HUB_DISABLE_TELEMETRY=1
export TOKENIZERS_PARALLELISM=false
unset PYTHONOPTIMIZE || true

"$PYTHON" - "$SRC" "$ARTIFACT_ROOT" "$IMPLEMENTATION_FINGERPRINT" \
  "$SCIENTIFIC_CONFIG_DIGEST" "$RUN_ID_SET_SHA256" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

from goalzendo_hidden_law.artifacts import implementation_provenance, read_json, verify_completed_run
from goalzendo_hidden_law.config import (
    build_hidden_law_plan,
    canonical_digest,
    load_hidden_law_config,
    scientific_config,
)

source = Path(sys.argv[1]).resolve()
artifacts = Path(sys.argv[2]).resolve()
implementation_fingerprint, scientific_digest, expected_run_id_digest = sys.argv[3:6]
config = load_hidden_law_config(
    source / "configs/goalzendo/qwen35_hidden_law_finite_choice.yaml"
)
if implementation_provenance(source)["implementation_fingerprint"] != implementation_fingerprint:
    raise SystemExit("archived implementation fingerprint differs from registration")
if canonical_digest(scientific_config(config)) != scientific_digest:
    raise SystemExit("archived scientific config differs from registration")

plan = build_hidden_law_plan(config)
expected = {condition.run_id: condition for condition in plan}
run_id_digest = hashlib.sha256(("\n".join(sorted(expected)) + "\n").encode()).hexdigest()
if len(plan) != 24 or len(expected) != 24 or run_id_digest != expected_run_id_digest:
    raise SystemExit("archived plan differs from the exact registered 24-run panel")
try:
    children = {path.name: path for path in artifacts.iterdir() if path.is_dir()}
except OSError as exc:
    raise SystemExit("cannot inspect scientific artifact root") from exc
if set(children) != set(expected):
    raise SystemExit("scientific artifact root has missing or extra run directories")

for run_id, condition in expected.items():
    path = children[run_id]
    if not (path / "COMPLETE").is_file():
        raise SystemExit(f"registered run lacks COMPLETE seal: {run_id}")
    verify_completed_run(path)
    identity = read_json(path / "identity.json")
    status = read_json(path / "status.json")
    if (
        identity.get("run_id") != run_id
        or identity.get("plan_key") != condition.plan_key
        or identity.get("implementation_fingerprint") != implementation_fingerprint
    ):
        raise SystemExit(f"registered run identity differs: {run_id}")
    if (
        status.get("state") != "complete"
        or status.get("phase") != "complete"
        or status.get("error") is not None
    ):
        raise SystemExit(f"registered run status is not complete: {run_id}")

print(
    json.dumps(
        {
            "run_id_set_sha256": run_id_digest,
            "verified_complete_run_count": len(expected),
        },
        sort_keys=True,
        separators=(",", ":"),
    )
)
PY
VERIFIED_COMPLETE_RUN_COUNT=$PLANNED_RUNS
export VERIFIED_COMPLETE_RUN_COUNT
echo "GOALZENDO_HIDDEN_LAW_RUNS_VERIFIED count=$VERIFIED_COMPLETE_RUN_COUNT"

[[ ! -e "$ANALYSIS_DIR" ]] || {
  echo "refusing to overwrite existing analysis directory" >&2
  exit 73
}
PAYLOAD=$(mktemp "$JOB/.analysis-payload.XXXXXX")
(
  cd "$SRC"
  "$PYTHON" scripts/analyze_qwen35_hidden_law.py "$ARTIFACT_ROOT" > "$PAYLOAD"
)

"$PYTHON" - "$PAYLOAD" "$CONFIG_SHA256" "$SCIENTIFIC_CONFIG_DIGEST" \
  "$IMPLEMENTATION_FINGERPRINT" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
config_sha256, scientific_digest, implementation_fingerprint = sys.argv[2:5]
payload = path.read_bytes()
if payload.count(b"\n") != 1 or not payload.endswith(b"\n"):
    raise SystemExit("frozen analyzer output is not one-line canonical JSON")
try:
    report = json.loads(payload)
    canonical = (
        json.dumps(report, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
    ).encode()
except (UnicodeError, json.JSONDecodeError, TypeError, ValueError) as exc:
    raise SystemExit("frozen analyzer output is not canonical JSON") from exc
if type(report) is not dict or canonical != payload:
    raise SystemExit("frozen analyzer output is not canonical JSON")
panel = report.get("panel")
fixed_panel = {
    "completed_only": True,
    "expected_run_count": 24,
    "observed_run_count": 24,
    "config_file_sha256": config_sha256,
    "scientific_config_digest": scientific_digest,
    "implementation_fingerprint": implementation_fingerprint,
}
if (
    report.get("schema") != "goalzendo.qwen35_hidden_law_analysis"
    or report.get("schema_version") != 1
    or type(panel) is not dict
    or any(panel.get(key) != value for key, value in fixed_panel.items())
):
    raise SystemExit("frozen analyzer output is not bound to the registered complete panel")
PY

ANALYSIS_SHA256=$(sha256sum "$PAYLOAD" | awk '{print $1}')
FINISHED_AT=$(date -u +%Y-%m-%dT%H:%M:%SZ)
STAGING=$(mktemp -d "$JOB/.analysis.staging.XXXXXX")
cp "$PAYLOAD" "$STAGING/$ANALYSIS_NAME"
export TERMINAL_SHARD_RECEIPT_COUNT VERIFIED_COMPLETE_RUN_COUNT STARTED_AT
write_status "$STAGING/status.json" complete 0 "$FINISHED_AT" "$ANALYSIS_SHA256"
(
  cd "$STAGING"
  sha256sum "$ANALYSIS_NAME" status.json > SHA256SUMS
)
mv "$STAGING" "$ANALYSIS_DIR"
STAGING=
echo "GOALZENDO_HIDDEN_LAW_ANALYSIS_PUBLISHED sha256=$ANALYSIS_SHA256"
exit 0
