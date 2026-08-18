#!/usr/bin/env bash
set -Eeuo pipefail

JOB_ID=qwen35-hidden-law-20260815T153604Z
ARCHIVE_SHA256=56700896a4f62f6e92a495cc0af109ff1f8982c8e9be57b920de6737a3b919f8
SMOKE_WRAPPER_SHA256=cab0c375b68ae0bfa28407d489e18833e88102a18e5ac205d0967485b87f21ab
IMAGE=runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404
VOLUME_ID=5f06o456rx
DATA_CENTER=US-CA-2
GPU_ID=${1:?usage: launch-smoke-attempt-3.sh GPU_ID}
REMOTE_WRAPPER="/workspace/goalzendo-hidden-law/jobs/$JOB_ID/input/smoke-attempt-3-wrapper.sh"

DOCKER_ARGS=$(jq -cn --arg wrapper "$REMOTE_WRAPPER" '{cmd:["bash",$wrapper],entrypoint:[]}')
TERMINATE_AT=$(python3 - <<'PY'
from datetime import datetime, timedelta, timezone
print((datetime.now(timezone.utc) + timedelta(hours=6)).strftime("%Y-%m-%dT%H:%M:%SZ"))
PY
)
ENV_JSON=$(jq -cn \
  --arg job "$JOB_ID" \
  --arg archive "$ARCHIVE_SHA256" \
  --arg wrapper "$SMOKE_WRAPPER_SHA256" \
  '{JOB_ID:$job,ARCHIVE_SHA256:$archive,SMOKE_WRAPPER_SHA256:$wrapper}')

runpodctl pod create \
  --name "$JOB_ID-smoke-a3" \
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
