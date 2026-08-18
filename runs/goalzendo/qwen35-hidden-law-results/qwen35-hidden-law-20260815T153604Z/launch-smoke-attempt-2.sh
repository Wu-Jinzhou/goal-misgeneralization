#!/usr/bin/env bash
set -Eeuo pipefail

JOB_ID=qwen35-hidden-law-20260815T153604Z
ARCHIVE_SHA256=56700896a4f62f6e92a495cc0af109ff1f8982c8e9be57b920de6737a3b919f8
IMAGE=runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404
VOLUME_ID=5f06o456rx
DATA_CENTER=US-CA-2
GPU_ID=${1:?usage: launch-smoke-attempt-2.sh GPU_ID}

read -r -d '' WRAPPER <<'BASH' || true
set -Eeuo pipefail
: "${JOB_ID:?}" "${ARCHIVE_SHA256:?}"
JOB="/workspace/goalzendo-hidden-law/jobs/$JOB_ID"
STATUS_DIR="$JOB/status"
LOG_DIR="$JOB/logs"
STATUS="$STATUS_DIR/smoke-a2.status.json"
ARCHIVE="$JOB/input/goalzendo-hidden-law-src.tgz"
EXTRACT="$JOB/smoke-source-a2"
SRC="$EXTRACT/goalzendo-hidden-law-src"
VENV=/workspace/goalzendo-qwen35/jobs/qwen35-pilot-20260813T134449Z/work/.venv
HF_CACHE=/workspace/goalzendo-qwen35/jobs/qwen35-pilot-20260813T134449Z/work/huggingface
STARTED_AT=$(date -u +%Y-%m-%dT%H:%M:%SZ)
mkdir -p "$STATUS_DIR" "$LOG_DIR" "$EXTRACT" "$JOB/artifacts/smoke"
exec > >(tee -a "$LOG_DIR/smoke-a2.log") 2>&1
emit_status() {
  path=$1 state=$2 rc=$3 finished=$4
  tmp="$path.tmp.$$"
  if [[ "$finished" == null ]]; then finished_json=null; else finished_json="\"$finished\""; fi
  printf '{"schema":"goalzendo.hidden_law_smoke_status","schema_version":1,"job_id":"%s","attempt":2,"state":"%s","runner_rc":%s,"archive_sha256":"%s","started_at":"%s","finished_at":%s}\n' \
    "$JOB_ID" "$state" "$rc" "$ARCHIVE_SHA256" "$STARTED_AT" "$finished_json" > "$tmp"
  mv "$tmp" "$path"
}
finish() {
  rc=$?
  set +e
  trap - EXIT TERM INT
  finished=$(date -u +%Y-%m-%dT%H:%M:%SZ)
  if [[ $rc -eq 0 ]]; then state=complete; else state=failed; fi
  emit_status "$STATUS" "$state" "$rc" "$finished"
  emit_status "$STATUS_DIR/smoke-a2.$state.json" "$state" "$rc" "$finished"
  sync
  echo "GOALZENDO_HIDDEN_LAW_SMOKE_TERMINAL attempt=2 state=$state rc=$rc finished_at=$finished"
  sleep infinity
}
trap finish EXIT
trap 'exit 143' TERM INT
emit_status "$STATUS" running null null
observed=$(sha256sum "$ARCHIVE" | awk '{print $1}')
[[ "$observed" == "$ARCHIVE_SHA256" ]] || { echo "archive digest mismatch: $observed"; exit 66; }
tar --no-same-owner -xzf "$ARCHIVE" -C "$EXTRACT"
test -x "$VENV/bin/python"
test -d "$HF_CACHE/hub/models--Qwen--Qwen3.5-0.8B/snapshots/2fc06364715b967f1860aea9cf38778875588b17"
test -d "$HF_CACHE/hub/models--Qwen--Qwen3.5-2B/snapshots/15852e8c16360a2fea060d615a32b45270f8a8fc"
[[ "$(sha256sum "$SRC/configs/goalzendo/qwen35_hidden_law_finite_choice.yaml" | awk '{print $1}')" == a2e31815f3fbe1d65fcd64654f8fb9e51a2d950cf8a2985cc5d90121bad3e53a ]]
[[ "$(sha256sum "$SRC/scripts/analyze_qwen35_hidden_law.py" | awk '{print $1}')" == 8bedacc9b5935e67224676101e12692bb0a5b913f1e9de180a033ac20f0a2d8c ]]
[[ "$(sha256sum "$SRC/runs/goalzendo/run_qwen35_hidden_law.sh" | awk '{print $1}')" == 9340b2f99900870dc8979fd47c8050fb63552fa3f21de7d8dd8681d6f94f4e01 ]]
unset PYTHONPATH
export GOALZENDO_PYTHON="$VENV/bin/python"
export HF_HOME="$HF_CACHE"
export HUGGINGFACE_HUB_CACHE="$HF_CACHE/hub"
export GOALZENDO_OUTPUT_ROOT="$JOB/artifacts/smoke"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1 TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONUNBUFFERED=1
cd "$SRC"
"$VENV/bin/python" - <<'PY'
from goalzendo_hidden_law.artifacts import implementation_provenance
observed=implementation_provenance('.')
assert observed['implementation_fingerprint']=='9cd679289fc22ceeb5fe1a9cbc2ca134051faf8491b9fa478be9b13ca81dd087'
assert observed['source_file_count']==75
PY
runs/goalzendo/run_qwen35_hidden_law.sh plan > "$JOB/launch-plan-a2.json"
for model in 0.8B 2B; do
  for algorithm in process_sft outcome_rl; do
    echo "GOALZENDO_SMOKE_START attempt=2 model=$model algorithm=$algorithm at=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    runs/goalzendo/run_qwen35_hidden_law.sh smoke "$model" "$algorithm"
    echo "GOALZENDO_SMOKE_DONE attempt=2 model=$model algorithm=$algorithm at=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  done
done
for model in 0.8B 2B; do
  for algorithm in process_sft outcome_rl; do
    runs/goalzendo/run_qwen35_hidden_law.sh smoke "$model" "$algorithm"
  done
done
"$VENV/bin/python" - "$JOB/launch-plan-a2.json" "$JOB/artifacts/smoke" <<'PY' > "$JOB/smoke-verification-a2.json"
import json, math, sys
from pathlib import Path
from goalzendo_hidden_law.artifacts import read_json, verify_completed_run
plan=json.loads(Path(sys.argv[1]).read_text())
root=Path(sys.argv[2])/'execution-smoke'
verified=[]
for spec in plan['smokes']:
    run=root/spec['smoke_run_id']
    verify_completed_run(run)
    summary=read_json(run/'summary.json')
    status=read_json(run/'status.json')
    manifest=read_json(run/'model-manifest.json')
    rows=[json.loads(line) for line in (run/'metrics.jsonl').read_text().splitlines() if line]
    assert summary['smoke'] is True and summary['last_step']==1 and summary['evaluation'] is None
    assert status['state']==status['phase']=='complete' and status['last_step']==1
    assert len(rows)==1 and rows[0]['kind']=='smoke_update'
    assert math.isfinite(rows[0]['loss']) and math.isfinite(rows[0]['gradient_norm'])
    assert rows[0]['gradient_norm']>0 and rows[0]['learning_rate']>0
    assert (run/'transcripts.jsonl').read_bytes()==b'' and (run/'predictions.jsonl').read_bytes()==b''
    op=summary['operational']
    assert op['forward_calls']>0 and op['scored_prompt_count']>0 and op['scored_prompt_tokens_unpadded']>0
    assert 0 < op['maximum_prompt_tokens'] <= 1536
    assert manifest['trainable_parameter_dtype_counts']=={'torch.bfloat16':manifest['parameter_count']}
    assert manifest['requested_revision']==spec['source_plan_key'].split(':')[0] or manifest['requested_revision']
    verified.append({'model':spec['model_name'],'algorithm':spec['algorithm'],'run_id':spec['smoke_run_id'],'passed':True})
print(json.dumps({'schema':'goalzendo.hidden_law_smoke_verification','schema_version':1,'attempt':2,'outcomes_inspected':False,'passed':True,'verified':verified},sort_keys=True,separators=(',',':')))
PY
BASH

DOCKER_ARGS=$(jq -cn --arg cmd "$WRAPPER" '{cmd:["bash","-lc",$cmd],entrypoint:[]}')
TERMINATE_AT=$(python3 - <<'PY'
from datetime import datetime, timedelta, timezone
print((datetime.now(timezone.utc) + timedelta(hours=6)).strftime("%Y-%m-%dT%H:%M:%SZ"))
PY
)
ENV_JSON=$(jq -cn \
  --arg job "$JOB_ID" \
  --arg archive "$ARCHIVE_SHA256" \
  '{JOB_ID:$job,ARCHIVE_SHA256:$archive}')

runpodctl pod create \
  --name "$JOB_ID-smoke-a2" \
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
