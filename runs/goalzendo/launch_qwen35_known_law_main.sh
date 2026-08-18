#!/usr/bin/env bash
set -Eeuo pipefail

# Launch one native shard of the frozen Qwen3.5 known-Law main panel on the
# existing US-CA-2 research volume. The pilot environment and model cache are
# reused read-only; this script neither installs packages nor runs a gate.

if [[ $# -ne 3 ]]; then
  echo "usage: $0 JOB_ID SHARD_INDEX GPU_ID" >&2
  exit 64
fi

JOB_ID=$1
SHARD_INDEX=$2
GPU_ID=$3
NUM_SHARDS=6
ARCHIVE_SHA256=559fa11ed2b4a3b073a67463b733185b55c9e98437efe2494eb3bfa100f47fe9
CONFIG_SHA256=2e1bbd650d466bb42f24b70d6081fd0f4f2490b14bfc7e02d17d59d9e251e7a6
IMPLEMENTATION_FINGERPRINT=592dd4df02ceff4293ce6eebc6ea2a0975e7d427cb1298335bcd3319431f301b
IMAGE=runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404
VOLUME_ID=5f06o456rx
DATA_CENTER=US-CA-2

if [[ ! "$SHARD_INDEX" =~ ^[0-5]$ ]]; then
  echo "SHARD_INDEX must be one of 0,1,2,3,4,5" >&2
  exit 64
fi

read -r -d '' WRAPPER <<'BASH' || true
set -Eeuo pipefail
: "${JOB_ID:?}" "${SHARD_INDEX:?}" "${NUM_SHARDS:?}" "${ARCHIVE_SHA256:?}"
: "${CONFIG_SHA256:?}" "${IMPLEMENTATION_FINGERPRINT:?}"
[[ "$NUM_SHARDS" == 6 ]]

JOB="/workspace/goalzendo-qwen35/jobs/$JOB_ID"
printf -v TAG '%02d' "$SHARD_INDEX"
SHARD_ROOT="$JOB/shards/shard-$TAG"
STATUS_DIR="$JOB/status"
LOG_DIR="$JOB/logs"
STATUS="$STATUS_DIR/shard-$TAG.status.json"
ARCHIVE="$JOB/input/goalzendo-qwen35-main-src.tgz"
EXTRACT_ROOT="$SHARD_ROOT/source"
SRC="$EXTRACT_ROOT/goalzendo-qwen35-src"
VENV=/workspace/goalzendo-qwen35/jobs/qwen35-pilot-20260813T134449Z/work/.venv
HF_CACHE=/workspace/goalzendo-qwen35/jobs/qwen35-pilot-20260813T134449Z/work/huggingface
STARTED_AT=$(date -u +%Y-%m-%dT%H:%M:%SZ)
mkdir -p "$SHARD_ROOT" "$STATUS_DIR" "$LOG_DIR" "$EXTRACT_ROOT" "$JOB/artifacts/main"
exec > >(tee -a "$LOG_DIR/shard-$TAG.log") 2>&1

emit_status() {
  local path=$1 state=$2 rc=$3 finished=$4
  local tmp="$path.tmp.$$" finished_json
  if [[ "$finished" == null ]]; then
    finished_json=null
  else
    finished_json="\"$finished\""
  fi
  printf '{"schema":"goalzendo.qwen35_main_shard_status","schema_version":1,"job_id":"%s","shard_index":%s,"num_shards":6,"state":"%s","runner_rc":%s,"archive_sha256":"%s","config_sha256":"%s","implementation_fingerprint":"%s","started_at":"%s","finished_at":%s}\n' \
    "$JOB_ID" "$SHARD_INDEX" "$state" "$rc" "$ARCHIVE_SHA256" "$CONFIG_SHA256" \
    "$IMPLEMENTATION_FINGERPRINT" "$STARTED_AT" "$finished_json" > "$tmp"
  mv "$tmp" "$path"
}

finish() {
  local rc=$?
  set +e
  trap - EXIT TERM INT
  local finished state
  finished=$(date -u +%Y-%m-%dT%H:%M:%SZ)
  if [[ $rc -eq 0 ]]; then state=complete; else state=failed; fi
  emit_status "$STATUS" "$state" "$rc" "$finished"
  emit_status "$STATUS_DIR/shard-$TAG.$state.json" "$state" "$rc" "$finished"
  sync
  echo "GOALZENDO_SHARD_TERMINAL shard=$SHARD_INDEX state=$state rc=$rc finished_at=$finished"
  sleep infinity
}
trap finish EXIT
trap 'exit 143' TERM INT

emit_status "$STATUS" running null null
echo "GOALZENDO_SHARD_START shard=$SHARD_INDEX/6 started_at=$STARTED_AT"
observed=$(sha256sum "$ARCHIVE" | awk '{print $1}')
[[ "$observed" == "$ARCHIVE_SHA256" ]] || {
  echo "archive digest mismatch: $observed" >&2
  exit 66
}
tar -xzf "$ARCHIVE" -C "$EXTRACT_ROOT"
test -x "$VENV/bin/python"
test -d "$HF_CACHE/hub/models--Qwen--Qwen3.5-0.8B/snapshots/2fc06364715b967f1860aea9cf38778875588b17"
test -d "$HF_CACHE/hub/models--Qwen--Qwen3.5-2B/snapshots/15852e8c16360a2fea060d615a32b45270f8a8fc"

export PYTHONPATH="$SRC/src"
export HF_HOME="$HF_CACHE"
export HUGGINGFACE_HUB_CACHE="$HF_CACHE/hub"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HUB_DISABLE_TELEMETRY=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1
cd "$SRC"

observed_config=$(sha256sum configs/goalzendo/qwen35_known_law_main.yaml | awk '{print $1}')
[[ "$observed_config" == "$CONFIG_SHA256" ]] || {
  echo "config digest mismatch: $observed_config" >&2
  exit 66
}
observed_implementation=$("$VENV/bin/python" - <<'PY'
from goalzendo.artifacts import implementation_provenance
print(implementation_provenance(".")["implementation_fingerprint"])
PY
)
[[ "$observed_implementation" == "$IMPLEMENTATION_FINGERPRINT" ]] || {
  echo "implementation fingerprint mismatch: $observed_implementation" >&2
  exit 66
}

# The reused virtual environment deliberately inherits NumPy and CUDA-enabled
# Torch from the exact pilot base image. Prove that complete runtime before the
# GoalZendo runner initializes any seed-level artifact.
"$VENV/bin/python" - <<'PY'
import importlib.metadata
import numpy
import torch
import transformers

expected = {
    "numpy": "2.1.2",
    "torch": "2.8.0+cu128",
    "transformers": "5.15.0",
}
observed = {
    "numpy": numpy.__version__,
    "torch": torch.__version__,
    "transformers": transformers.__version__,
}
if observed != expected:
    raise SystemExit(f"runtime dependency mismatch: expected {expected}, got {observed}")
if importlib.metadata.version("huggingface-hub") != "1.5.0":
    raise SystemExit("huggingface-hub version mismatch")
if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
    raise SystemExit("exactly one CUDA device is required")
print(f"GOALZENDO_RUNTIME_VERIFIED {observed} gpu={torch.cuda.get_device_name(0)}")
PY

expected_counts=(22 12 19 18 12 13)
expected_count=${expected_counts[$SHARD_INDEX]}
observed_count=$("$VENV/bin/python" - "$SHARD_INDEX" <<'PY'
import sys
from goalzendo.config import load_config
from goalzendo.runner import build_plan
config = load_config("configs/goalzendo/qwen35_known_law_main.yaml")
print(len(build_plan(config, shard_index=int(sys.argv[1]), num_shards=6)))
PY
)
[[ "$observed_count" == "$expected_count" ]] || {
  echo "shard plan count mismatch: expected $expected_count, got $observed_count" >&2
  exit 66
}
echo "GOALZENDO_SHARD_VERIFIED shard=$SHARD_INDEX planned_runs=$observed_count"

"$VENV/bin/python" -m goalzendo.cli run \
  configs/goalzendo/qwen35_known_law_main.yaml \
  --shard-index "$SHARD_INDEX" \
  --num-shards "$NUM_SHARDS" \
  --output-root "$JOB/artifacts/main"
BASH

DOCKER_ARGS=$(jq -cn --arg cmd "$WRAPPER" '{cmd:["bash","-lc",$cmd],entrypoint:[]}')
TERMINATE_AT=$(python3 - <<'PY'
from datetime import datetime, timedelta, timezone
print((datetime.now(timezone.utc) + timedelta(hours=72)).strftime("%Y-%m-%dT%H:%M:%SZ"))
PY
)
ENV_JSON=$(jq -cn \
  --arg job "$JOB_ID" \
  --arg shard "$SHARD_INDEX" \
  --arg archive "$ARCHIVE_SHA256" \
  --arg config "$CONFIG_SHA256" \
  --arg implementation "$IMPLEMENTATION_FINGERPRINT" \
  '{JOB_ID:$job,SHARD_INDEX:$shard,NUM_SHARDS:"6",ARCHIVE_SHA256:$archive,CONFIG_SHA256:$config,IMPLEMENTATION_FINGERPRINT:$implementation}')
printf -v TAG '%02d' "$SHARD_INDEX"

runpodctl pod create \
  --name "$JOB_ID-s$TAG" \
  --image "$IMAGE" \
  --gpu-id "$GPU_ID" \
  --gpu-count 1 \
  --cloud-type SECURE \
  --data-center-ids "$DATA_CENTER" \
  --min-cuda-version 12.8 \
  --network-volume-id "$VOLUME_ID" \
  --volume-mount-path /workspace \
  --container-disk-in-gb 20 \
  --ssh=false \
  --env "$ENV_JSON" \
  --docker-args "$DOCKER_ARGS" \
  --terminate-after "$TERMINATE_AT" \
  --output json
