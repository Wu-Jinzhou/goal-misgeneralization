#!/usr/bin/env bash
set -Eeuo pipefail
source "$(dirname "$0")/_common.sh"

CONFIG="$GOALZENDO_ROOT/configs/goalzendo/g00a4_full_model_smoke.yaml"
run_goalzendo validate "$CONFIG"
run_goalzendo plan "$CONFIG"
run_goalzendo run "$CONFIG" "$@"
