#!/usr/bin/env bash
set -Eeuo pipefail
source "$(dirname "$0")/_common.sh"

CONFIG="$GOALZENDO_ROOT/configs/goalzendo/g00_capability_controls_1p5b.yaml"
run_goalzendo validate "$CONFIG"
run_goalzendo plan "$CONFIG"
run_goalzendo run "$CONFIG" "$@"
