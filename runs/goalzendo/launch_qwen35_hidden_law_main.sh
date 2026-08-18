#!/usr/bin/env bash
set -Eeuo pipefail

# Launch one native shard of the frozen finite-choice hidden-Law panel on the
# existing US-CA-2 research volume. The four engineering smokes must be
# independently verified before this operational wrapper is used.

if [[ $# -ne 3 ]]; then
  echo "usage: $0 JOB_ID SHARD_INDEX GPU_ID" >&2
  exit 64
fi

JOB_ID=$1
SHARD_INDEX=$2
GPU_ID=$3
NUM_SHARDS=6
ARCHIVE_SHA256=56700896a4f62f6e92a495cc0af109ff1f8982c8e9be57b920de6737a3b919f8
CONFIG_SHA256=a2e31815f3fbe1d65fcd64654f8fb9e51a2d950cf8a2985cc5d90121bad3e53a
SCIENTIFIC_CONFIG_DIGEST=d61cad537e4ae3ea439250bc5e7e2466405e5331be251f6b8f15d0de8683207b
PROTOCOL_SHA256=f4752515e3848a98452b89a3f753e8310dc45c131cd33244dc56fa696125883e
IMPLEMENTATION_FINGERPRINT=9cd679289fc22ceeb5fe1a9cbc2ca134051faf8491b9fa478be9b13ca81dd087
ANALYZER_SHA256=8bedacc9b5935e67224676101e12692bb0a5b913f1e9de180a033ac20f0a2d8c
RUNNER_SHA256=849c4db9d2fa546fc3c14018bb71303504c32a82627be8de40187bf5ce5f0c7d
LAUNCHER_SHA256=9340b2f99900870dc8979fd47c8050fb63552fa3f21de7d8dd8681d6f94f4e01
CONSTRAINTS_SHA256=d434e42c1b57933601de6a3890d0937389fd5c7c797a2d84777b77e8f3febff3
PLAN_JSON_SHA256=7b60ed59e8d8842f8c9e6b21830f3d092ebfa94e60427294dce9b6e210659a86
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
: "${CONFIG_SHA256:?}" "${SCIENTIFIC_CONFIG_DIGEST:?}" "${PROTOCOL_SHA256:?}"
: "${IMPLEMENTATION_FINGERPRINT:?}" "${ANALYZER_SHA256:?}" "${RUNNER_SHA256:?}"
: "${LAUNCHER_SHA256:?}" "${CONSTRAINTS_SHA256:?}" "${PLAN_JSON_SHA256:?}"
[[ "$NUM_SHARDS" == 6 ]]

JOB="/workspace/goalzendo-hidden-law/jobs/$JOB_ID"
printf -v TAG '%02d' "$SHARD_INDEX"
SHARD_ROOT="$JOB/shards/shard-$TAG"
STATUS_DIR="$JOB/status"
LOG_DIR="$JOB/logs"
STATUS="$STATUS_DIR/shard-$TAG.status.json"
ARCHIVE="$JOB/input/goalzendo-hidden-law-src.tgz"
STARTED_AT=$(date -u +%Y-%m-%dT%H:%M:%SZ)
START_TAG=$(date -u +%Y%m%dT%H%M%SZ)
EXTRACT_ROOT="$SHARD_ROOT/source-$START_TAG"
SRC="$EXTRACT_ROOT/goalzendo-hidden-law-src"
VENV=/workspace/goalzendo-qwen35/jobs/qwen35-pilot-20260813T134449Z/work/.venv
HF_CACHE=/workspace/goalzendo-qwen35/jobs/qwen35-pilot-20260813T134449Z/work/huggingface
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
  printf '{"schema":"goalzendo.hidden_law_shard_status","schema_version":1,"job_id":"%s","shard_index":%s,"num_shards":6,"state":"%s","runner_rc":%s,"archive_sha256":"%s","config_sha256":"%s","scientific_config_digest":"%s","implementation_fingerprint":"%s","started_at":"%s","finished_at":%s}\n' \
    "$JOB_ID" "$SHARD_INDEX" "$state" "$rc" "$ARCHIVE_SHA256" "$CONFIG_SHA256" \
    "$SCIENTIFIC_CONFIG_DIGEST" "$IMPLEMENTATION_FINGERPRINT" "$STARTED_AT" "$finished_json" > "$tmp"
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
  echo "GOALZENDO_HIDDEN_LAW_SHARD_TERMINAL shard=$SHARD_INDEX state=$state rc=$rc finished_at=$finished"
  sleep infinity
}
trap finish EXIT
trap 'exit 143' TERM INT

emit_status "$STATUS" running null null
echo "GOALZENDO_HIDDEN_LAW_SHARD_START shard=$SHARD_INDEX/6 started_at=$STARTED_AT"
observed=$(sha256sum "$ARCHIVE" | awk '{print $1}')
[[ "$observed" == "$ARCHIVE_SHA256" ]] || {
  echo "archive digest mismatch: $observed" >&2
  exit 66
}
tar --no-same-owner -xzf "$ARCHIVE" -C "$EXTRACT_ROOT"
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
export GOALZENDO_PYTHON="$VENV/bin/python"
export GOALZENDO_OUTPUT_ROOT="$JOB/artifacts/main"
cd "$SRC"

sha_check() {
  local expected=$1 path=$2 label=$3 observed_sha
  observed_sha=$(sha256sum "$path" | awk '{print $1}')
  [[ "$observed_sha" == "$expected" ]] || {
    echo "$label digest mismatch: $observed_sha" >&2
    exit 66
  }
}
sha_check "$CONFIG_SHA256" configs/goalzendo/qwen35_hidden_law_finite_choice.yaml config
sha_check "$PROTOCOL_SHA256" docs/goalzendo/protocols/qwen35-hidden-law-finite-choice.md protocol
sha_check "$ANALYZER_SHA256" scripts/analyze_qwen35_hidden_law.py analyzer
sha_check "$RUNNER_SHA256" scripts/run_qwen35_hidden_law.py runner
sha_check "$LAUNCHER_SHA256" runs/goalzendo/run_qwen35_hidden_law.sh launcher
sha_check "$CONSTRAINTS_SHA256" constraints-goalzendo-qwen35.txt constraints

read -r observed_implementation observed_scientific < <("$VENV/bin/python" - <<'PY'
from goalzendo_hidden_law.artifacts import implementation_provenance
from goalzendo_hidden_law.config import canonical_digest, load_hidden_law_config, scientific_config

config = load_hidden_law_config("configs/goalzendo/qwen35_hidden_law_finite_choice.yaml")
print(
    implementation_provenance(".")["implementation_fingerprint"],
    canonical_digest(scientific_config(config)),
)
PY
)
[[ "$observed_implementation" == "$IMPLEMENTATION_FINGERPRINT" ]] || {
  echo "implementation fingerprint mismatch: $observed_implementation" >&2
  exit 66
}
[[ "$observed_scientific" == "$SCIENTIFIC_CONFIG_DIGEST" ]] || {
  echo "scientific config digest mismatch: $observed_scientific" >&2
  exit 66
}

"$VENV/bin/python" - <<'PY'
import importlib.metadata
import torch
import transformers

expected = {
    "accelerate": "1.14.0",
    "huggingface-hub": "1.5.0",
    "peft": "0.20.0",
    "safetensors": "0.8.0",
    "tokenizers": "0.22.2",
    "transformers": "5.15.0",
}
observed = {name: importlib.metadata.version(name) for name in expected}
if observed != expected:
    raise SystemExit(f"runtime dependency mismatch: expected {expected}, got {observed}")
if torch.__version__ != "2.8.0+cu128" or transformers.__version__ != "5.15.0":
    raise SystemExit(f"runtime torch/transformers mismatch: {torch.__version__}, {transformers.__version__}")
if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
    raise SystemExit("exactly one CUDA device is required")
print(f"GOALZENDO_HIDDEN_LAW_RUNTIME_VERIFIED gpu={torch.cuda.get_device_name(0)}")
PY

bash runs/goalzendo/run_qwen35_hidden_law.sh plan > "$SHARD_ROOT/plan.json"
sha_check "$PLAN_JSON_SHA256" "$SHARD_ROOT/plan.json" launch-plan
expected_counts=(4 5 3 4 5 3)
expected_count=${expected_counts[$SHARD_INDEX]}
observed_count=$("$VENV/bin/python" - "$SHARD_ROOT/plan.json" "$SHARD_INDEX" <<'PY'
import json
import sys

plan = json.load(open(sys.argv[1], encoding="utf-8"))
shard = plan["production"]["shards"][int(sys.argv[2])]
if shard["shard_index"] != int(sys.argv[2]):
    raise SystemExit("launch plan shard index mismatch")
if len(shard["run_ids"]) != shard["run_count"] or len(set(shard["run_ids"])) != shard["run_count"]:
    raise SystemExit("launch plan shard run IDs are malformed")
print(shard["run_count"])
PY
)
[[ "$observed_count" == "$expected_count" ]] || {
  echo "shard plan count mismatch: expected $expected_count, got $observed_count" >&2
  exit 66
}
echo "GOALZENDO_HIDDEN_LAW_SHARD_VERIFIED shard=$SHARD_INDEX planned_runs=$observed_count"

bash runs/goalzendo/run_qwen35_hidden_law.sh main "$SHARD_INDEX"
BASH

DOCKER_ARGS=$(jq -cn --arg cmd "$WRAPPER" '{cmd:["bash","-lc",$cmd],entrypoint:[]}')
TERMINATE_AT=$(python3 - <<'PY'
from datetime import datetime, timedelta, timezone
print((datetime.now(timezone.utc) + timedelta(hours=48)).strftime("%Y-%m-%dT%H:%M:%SZ"))
PY
)
ENV_JSON=$(jq -cn \
  --arg job "$JOB_ID" \
  --arg shard "$SHARD_INDEX" \
  --arg archive "$ARCHIVE_SHA256" \
  --arg config "$CONFIG_SHA256" \
  --arg scientific "$SCIENTIFIC_CONFIG_DIGEST" \
  --arg protocol "$PROTOCOL_SHA256" \
  --arg implementation "$IMPLEMENTATION_FINGERPRINT" \
  --arg analyzer "$ANALYZER_SHA256" \
  --arg runner "$RUNNER_SHA256" \
  --arg launcher "$LAUNCHER_SHA256" \
  --arg constraints "$CONSTRAINTS_SHA256" \
  --arg plan "$PLAN_JSON_SHA256" \
  '{JOB_ID:$job,SHARD_INDEX:$shard,NUM_SHARDS:"6",ARCHIVE_SHA256:$archive,CONFIG_SHA256:$config,SCIENTIFIC_CONFIG_DIGEST:$scientific,PROTOCOL_SHA256:$protocol,IMPLEMENTATION_FINGERPRINT:$implementation,ANALYZER_SHA256:$analyzer,RUNNER_SHA256:$runner,LAUNCHER_SHA256:$launcher,CONSTRAINTS_SHA256:$constraints,PLAN_JSON_SHA256:$plan}')
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
