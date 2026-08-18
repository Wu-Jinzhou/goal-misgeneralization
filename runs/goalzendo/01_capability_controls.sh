#!/usr/bin/env bash
set -Eeuo pipefail
source "$(dirname "$0")/_common.sh"

run_goalzendo validate "$GOALZENDO_ROOT/configs/goalzendo/g00_capability_controls.yaml"
run_goalzendo plan "$GOALZENDO_ROOT/configs/goalzendo/g00_capability_controls.yaml"
run_goalzendo run "$GOALZENDO_ROOT/configs/goalzendo/g00_capability_controls.yaml" "$@"
