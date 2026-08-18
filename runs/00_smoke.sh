#!/usr/bin/env bash
set -euo pipefail
export SMOKE=1
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"

run_config configs/smoke.yaml "$@"
