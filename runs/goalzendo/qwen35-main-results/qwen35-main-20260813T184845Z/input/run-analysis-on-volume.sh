#!/usr/bin/env bash
set -Eeuo pipefail

JOB=/workspace/goalzendo-qwen35/jobs/qwen35-main-20260813T184845Z
ANALYSIS="$JOB/analysis"
SOURCE="$ANALYSIS/source/goalzendo-qwen35-src"
ARCHIVE="$JOB/input/goalzendo-qwen35-main-src.tgz"
ARTIFACTS="$JOB/artifacts/main/qwen35_known_law_main/qwen35_known_law_main_panel"
VENV=/workspace/goalzendo-qwen35/jobs/qwen35-pilot-20260813T134449Z/work/.venv
STATUS="$ANALYSIS/status.json"
PRIMARY="$ANALYSIS/qwen35-known-law-analysis.json"
DESCRIPTIVE="$ANALYSIS/qwen35-known-law-descriptives.json"

mkdir -p "$ANALYSIS/source"

write_status() {
  local state=$1
  local rc=$2
  local finished_at=$3
  local temporary="$STATUS.tmp.$$"
  local finished_json=null
  if [[ "$finished_at" != null ]]; then
    finished_json="\"$finished_at\""
  fi
  printf '{"finished_at":%s,"runner_rc":%s,"schema":"goalzendo.qwen35_main_analysis_status","schema_version":1,"state":"%s"}\n' \
    "$finished_json" "$rc" "$state" > "$temporary"
  mv "$temporary" "$STATUS"
}

on_exit() {
  local rc=$?
  trap - EXIT
  set +e
  write_status failed "$rc" "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  sync
  sleep infinity
}
trap on_exit EXIT
write_status running null null

test "$(sha256sum "$ARCHIVE" | awk '{print $1}')" = \
  559fa11ed2b4a3b073a67463b733185b55c9e98437efe2494eb3bfa100f47fe9
tar -xzf "$ARCHIVE" -C "$ANALYSIS/source"
cp "$JOB/input/analyze_qwen35_known_law.py" \
  "$SOURCE/scripts/analyze_qwen35_known_law.py"
cp "$JOB/input/postlaunch-descriptive/export_qwen35_known_law_descriptives.py" \
  "$SOURCE/scripts/export_qwen35_known_law_descriptives.py"

test "$(sha256sum "$SOURCE/configs/goalzendo/qwen35_known_law_main.yaml" | awk '{print $1}')" = \
  2e1bbd650d466bb42f24b70d6081fd0f4f2490b14bfc7e02d17d59d9e251e7a6
test "$(sha256sum "$SOURCE/scripts/analyze_qwen35_known_law.py" | awk '{print $1}')" = \
  0aca675d5600724d6a993373b5ebba560f4bcf4f01887a6f30dc8acd1b7f7c89
test "$(sha256sum "$SOURCE/scripts/export_qwen35_known_law_descriptives.py" | awk '{print $1}')" = \
  5147a9767585b0e869faf84fcd0c343cae5e47b5671f4da9a422f259301c6cbf
test -x "$VENV/bin/python"
test -d "$ARTIFACTS"

export PYTHONPATH="$SOURCE/src"
export PYTHONUNBUFFERED=1
unset PYTHONOPTIMIZE
test "$($VENV/bin/python - <<'PY'
from goalzendo.artifacts import implementation_provenance
print(implementation_provenance()['implementation_fingerprint'])
PY
)" = 592dd4df02ceff4293ce6eebc6ea2a0975e7d427cb1298335bcd3319431f301b

primary_tmp="$PRIMARY.tmp.$$"
descriptive_tmp="$DESCRIPTIVE.tmp.$$"

run_analysis() {
  local log_path=$1
  shift
  if [[ -x /usr/bin/time ]]; then
    /usr/bin/time -v "$@" 2> "$log_path"
  else
    printf 'Portable fallback: /usr/bin/time is unavailable; running without resource telemetry.\n' \
      > "$log_path"
    "$@" 2>> "$log_path"
  fi
}

run_analysis "$ANALYSIS/primary-analysis.log" \
  "$VENV/bin/python" "$SOURCE/scripts/analyze_qwen35_known_law.py" \
  "$ARTIFACTS" > "$primary_tmp"
run_analysis "$ANALYSIS/descriptive-analysis.log" \
  "$VENV/bin/python" "$SOURCE/scripts/export_qwen35_known_law_descriptives.py" \
  "$ARTIFACTS" > "$descriptive_tmp"

PRIMARY_TMP="$primary_tmp" DESCRIPTIVE_TMP="$descriptive_tmp" "$VENV/bin/python" - <<'PY'
import json
import os
from pathlib import Path

def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)

def load_canonical(name: str) -> dict:
    path = Path(os.environ[name])
    payload = path.read_bytes()
    value = json.loads(payload)
    expected = (json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()
    require(payload == expected, f"{path} is not canonical JSON")
    return value

primary = load_canonical("PRIMARY_TMP")
require(primary["schema"] == "goalzendo.qwen35_known_law_analysis", "primary schema differs")
require(
    primary["panel"]["expected_run_count"] == primary["panel"]["observed_run_count"] == 96,
    "primary panel is incomplete",
)
require(
    primary["panel"]["expected_config_sha256"]
    == "2e1bbd650d466bb42f24b70d6081fd0f4f2490b14bfc7e02d17d59d9e251e7a6",
    "primary config binding differs",
)
require(
    primary["panel"]["implementation_fingerprint"]
    == "592dd4df02ceff4293ce6eebc6ea2a0975e7d427cb1298335bcd3319431f301b",
    "primary source binding differs",
)
require(len(primary["seed_endpoints"]) == 96, "primary endpoint count differs")
require(
    len(primary["primary"]["stratified_by_model_algorithm_law"]) == 8,
    "primary stratum count differs",
)
require(
    set(primary["secondary_contrasts"]) == {"algorithm", "evidence", "scale"},
    "secondary contrast inventory differs",
)

descriptive = load_canonical("DESCRIPTIVE_TMP")
require(
    descriptive["schema"] == "goalzendo.qwen35_known_law_descriptive_companion",
    "descriptive schema differs",
)
require(
    descriptive["panel"]["expected_run_count"]
    == descriptive["panel"]["observed_run_count"]
    == 96,
    "descriptive panel is incomplete",
)
require(
    descriptive["panel"]["implementation_fingerprint"]
    == "592dd4df02ceff4293ce6eebc6ea2a0975e7d427cb1298335bcd3319431f301b",
    "descriptive source binding differs",
)
require(len(descriptive["runs"]) == 96, "descriptive run count differs")
PY

mv "$primary_tmp" "$PRIMARY"
mv "$descriptive_tmp" "$DESCRIPTIVE"
sha256sum "$PRIMARY" "$DESCRIPTIVE" > "$ANALYSIS/SHA256SUMS"
write_status complete 0 "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
sync
trap - EXIT
sleep infinity
