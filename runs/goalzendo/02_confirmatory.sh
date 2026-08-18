#!/usr/bin/env bash
set -Eeuo pipefail
source "$(dirname "$0")/_common.sh"

# This launcher intentionally fails closed until G00 freezes the model commit
# and algorithm-specific learning settings in g01_known_law.yaml.
run_goalzendo validate "$GOALZENDO_ROOT/configs/goalzendo/g01_known_law.yaml"
run_goalzendo run "$GOALZENDO_ROOT/configs/goalzendo/g01_known_law.yaml" "$@"

