#!/usr/bin/env bash
set -Eeuo pipefail
source "$(dirname "$0")/_common.sh"

run_goalzendo validate "$GOALZENDO_ROOT/configs/goalzendo/smoke.yaml"
run_goalzendo generate "$GOALZENDO_ROOT/configs/goalzendo/smoke.yaml" \
  --split factorial --show 1
run_goalzendo run "$GOALZENDO_ROOT/configs/goalzendo/smoke.yaml" "$@"

