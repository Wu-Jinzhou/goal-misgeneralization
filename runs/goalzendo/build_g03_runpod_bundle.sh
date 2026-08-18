#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

if [[ $# -ne 1 ]]; then
  echo "usage: $0 /absolute/path/goalzendo-g03-bundle.tar.gz" >&2
  exit 2
fi

OUTPUT="$1"
if [[ "$OUTPUT" != /* || "$OUTPUT" != *.tar.gz ]]; then
  echo "output must be an absolute .tar.gz path" >&2
  exit 2
fi
if [[ -e "$OUTPUT" ]]; then
  echo "refusing to overwrite existing bundle: $OUTPUT" >&2
  exit 2
fi

INCLUDE=(
  LICENSE
  README.md
  pyproject.toml
  constraints-goalzendo.txt
  src/goalzendo_interactive
  docs/goalzendo
  runs/goalzendo/03_g03_qwen_decision_probe.py
  runs/goalzendo/04_g03_freeze_smoke_inputs.py
  runs/goalzendo/04_g03_materialize_snapshot.sh
  runs/goalzendo/04_g03_pinned_qwen_smoke.py
  runs/goalzendo/build_g03_runpod_bundle.sh
  tests/__init__.py
  tests/goalzendo_interactive
)

cd "$ROOT"
for path in "${INCLUDE[@]}"; do
  if [[ ! -e "$path" ]]; then
    echo "required G03 bundle path is absent: $path" >&2
    exit 1
  fi
done

secret_files="$({
  rg -l \
    '(hf_[A-Za-z0-9]{20,}|rpa_[A-Za-z0-9]{20,}|RUNPOD_API_KEY[[:space:]]*=|GITHUB_TOKEN[[:space:]]*=|AKIA[0-9A-Z]{16})' \
    "${INCLUDE[@]}" || true
} | sort -u)"
if [[ -n "$secret_files" ]]; then
  echo "refusing to bundle files matching a credential pattern:" >&2
  echo "$secret_files" >&2
  exit 1
fi

mkdir -p "$(dirname "$OUTPUT")"
temporary="$(mktemp "${TMPDIR:-/tmp}/goalzendo-g03-bundle.XXXXXX.tar.gz")"
cleanup() {
  rm -f -- "$temporary"
}
trap cleanup EXIT

COPYFILE_DISABLE=1 LC_ALL=C tar -C "$ROOT" -czf "$temporary" \
  --exclude='__pycache__' \
  --exclude='*.pyc' \
  --exclude='.DS_Store' \
  --exclude='tests/goalzendo_interactive/test_package_boundary.py' \
  "${INCLUDE[@]}"
mv "$temporary" "$OUTPUT"
trap - EXIT

if command -v sha256sum >/dev/null 2>&1; then
  sha256sum "$OUTPUT"
else
  shasum -a 256 "$OUTPUT"
fi
