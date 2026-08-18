#!/usr/bin/env bash
set -Eeuo pipefail
source "$(dirname "$0")/_common.sh"

CONFIG="$GOALZENDO_ROOT/configs/goalzendo/g00c_strict_engineering_0p5b.yaml"
run_goalzendo validate "$CONFIG"
run_goalzendo plan "$CONFIG"
run_goalzendo run "$CONFIG" "$@"

