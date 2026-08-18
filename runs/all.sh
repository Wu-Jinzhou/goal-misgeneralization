#!/usr/bin/env bash
# Paper-scale by default. Set SMOKE=1 to run one tiny cell per hypothesis.
set -euo pipefail
RUNS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

dry_run=0
previous=""
for argument in "$@"; do
  case "${argument}" in
    --output|--output=*|--device|--device=*|--smoke|--set=run.output_root=*|--set=run.device=*)
      echo "runs/all.sh: use OUTPUT_ROOT, DEVICE, or SMOKE=1 so pretraining, runs, and analysis share the setting" >&2
      exit 2
      ;;
    --dry-run)
      dry_run=1
      ;;
  esac
  if [[ "${previous}" == "--set" && ( "${argument}" == run.output_root=* || "${argument}" == run.device=* ) ]]; then
    echo "runs/all.sh: use OUTPUT_ROOT or DEVICE instead of overriding ${argument%%=*} through --set" >&2
    exit 2
  fi
  previous="${argument}"
done

if [[ "${dry_run}" == "0" ]]; then
  "${RUNS_DIR}/00_pretrain_navigators.sh"
fi
for script in \
  01_simplicity.sh \
  02_complexity.sh \
  03_dynamics.sh \
  04_conflict_diversity.sh \
  05_update_capacity.sh \
  06_noise.sh \
  07_unlearning.sh \
  08_hysteresis.sh \
  09_multiplicity.sh; do
  "${RUNS_DIR}/${script}" "$@"
done
if [[ "${dry_run}" == "0" ]]; then
  "${RUNS_DIR}/10_analyze.sh"
fi
