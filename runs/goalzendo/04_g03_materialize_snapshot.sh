#!/usr/bin/env bash
set -Eeuo pipefail

if [[ $# -ne 2 ]]; then
  echo "usage: $0 /absolute/hf/snapshot /absolute/fresh/materialized-root" >&2
  exit 2
fi

SOURCE="$1"
DESTINATION="$2"
if [[ "$SOURCE" != /* || "$DESTINATION" != /* ]]; then
  echo "source and destination must be absolute paths" >&2
  exit 2
fi
if [[ ! -d "$SOURCE" ]]; then
  echo "source snapshot is not a directory: $SOURCE" >&2
  exit 1
fi

python3 - "$SOURCE" "$DESTINATION" <<'PY'
from pathlib import Path
import sys

source = Path(sys.argv[1])
destination = Path(sys.argv[2])
try:
    canonical_source = source.resolve(strict=True)
    # strict=False resolves every existing parent (including symlink aliases)
    # while retaining the prospective, still-absent destination suffix.
    canonical_destination = destination.resolve(strict=False)
except (OSError, RuntimeError) as exc:
    raise SystemExit(f"snapshot paths could not be canonically resolved: {exc}") from exc

overlap = (
    canonical_source == canonical_destination
    or canonical_source in canonical_destination.parents
    or canonical_destination in canonical_source.parents
)
if overlap:
    raise SystemExit("source and destination must be canonically disjoint")
PY

if [[ -e "$DESTINATION" || -L "$DESTINATION" ]]; then
  echo "destination must be fresh and absent: $DESTINATION" >&2
  exit 1
fi

mkdir -p "$DESTINATION"
rsync -rltL \
  --no-owner \
  --no-group \
  --no-perms \
  --exclude='.cache/' \
  --exclude='*.lock' \
  "$SOURCE/" "$DESTINATION/"

if find "$DESTINATION" -type l -print -quit | grep -q .; then
  echo "materialized snapshot contains a symlink" >&2
  exit 1
fi
if find "$DESTINATION" \! -type d \! -type f -print -quit | grep -q .; then
  echo "materialized snapshot contains a special file" >&2
  exit 1
fi
if find "$DESTINATION" -type f -links +1 -print -quit | grep -q .; then
  echo "materialized snapshot contains a hard-linked file" >&2
  exit 1
fi
for required in config.json tokenizer.json tokenizer_config.json; do
  if [[ ! -f "$DESTINATION/$required" ]]; then
    echo "materialized snapshot is missing $required" >&2
    exit 1
  fi
done
if ! find "$DESTINATION" -type f -name '*.safetensors' -print -quit | grep -q .; then
  echo "materialized snapshot contains no safetensors weights" >&2
  exit 1
fi

find "$DESTINATION" -type f -print0 \
  | LC_ALL=C sort -z \
  | xargs -0 shasum -a 256
