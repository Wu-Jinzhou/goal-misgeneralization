#!/usr/bin/env bash
set -Eeuo pipefail

: "${JOB_ID:?}" "${ARCHIVE_SHA256:?}" "${SMOKE_WRAPPER_SHA256:?}"
JOB="/workspace/goalzendo-hidden-law/jobs/$JOB_ID"
STATUS_DIR="$JOB/status"
ATTEMPT_DIR="$STATUS_DIR/attempts"
LOG_DIR="$JOB/logs"
CANONICAL_STATUS="$STATUS_DIR/smoke.status.json"
ATTEMPT_STATUS="$ATTEMPT_DIR/smoke-a3.status.json"
ARCHIVE="$JOB/input/goalzendo-hidden-law-src.tgz"
WRAPPER_FILE="$JOB/input/smoke-attempt-3-wrapper.sh"
EXTRACT="$JOB/smoke-source-a3"
SRC="$EXTRACT/goalzendo-hidden-law-src"
VENV=/workspace/goalzendo-qwen35/jobs/qwen35-pilot-20260813T134449Z/work/.venv
HF_CACHE=/workspace/goalzendo-qwen35/jobs/qwen35-pilot-20260813T134449Z/work/huggingface
STARTED_AT=$(date -u +%Y-%m-%dT%H:%M:%SZ)

mkdir -p "$STATUS_DIR" "$ATTEMPT_DIR" "$LOG_DIR" "$EXTRACT" "$JOB/artifacts/smoke"
exec > >(tee -a "$LOG_DIR/smoke-a3.log") 2>&1

emit_status_file() {
  local path=$1 state=$2 rc=$3 finished=$4 tmp finished_json
  tmp="$path.tmp.$$"
  if [[ "$finished" == null ]]; then
    finished_json=null
  else
    finished_json="\"$finished\""
  fi
  printf '{"schema":"goalzendo.hidden_law_smoke_status","schema_version":1,"job_id":"%s","attempt":3,"state":"%s","runner_rc":%s,"archive_sha256":"%s","wrapper_sha256":"%s","started_at":"%s","finished_at":%s}\n' \
    "$JOB_ID" "$state" "$rc" "$ARCHIVE_SHA256" "$SMOKE_WRAPPER_SHA256" \
    "$STARTED_AT" "$finished_json" > "$tmp"
  mv "$tmp" "$path"
}

emit_status() {
  local state=$1 rc=$2 finished=$3
  emit_status_file "$CANONICAL_STATUS" "$state" "$rc" "$finished"
  emit_status_file "$ATTEMPT_STATUS" "$state" "$rc" "$finished"
}

finish() {
  local rc=$?
  set +e
  trap - EXIT TERM INT
  local finished state
  finished=$(date -u +%Y-%m-%dT%H:%M:%SZ)
  if [[ $rc -eq 0 ]]; then state=complete; else state=failed; fi
  emit_status "$state" "$rc" "$finished"
  emit_status_file "$ATTEMPT_DIR/smoke-a3.$state.json" "$state" "$rc" "$finished"
  sync
  echo "GOALZENDO_HIDDEN_LAW_SMOKE_TERMINAL attempt=3 state=$state rc=$rc finished_at=$finished"
  sleep infinity
}
trap finish EXIT
trap 'exit 143' TERM INT

emit_status running null null
echo "GOALZENDO_HIDDEN_LAW_SMOKE_START attempt=3 started_at=$STARTED_AT"

observed_wrapper=$(sha256sum "$WRAPPER_FILE" | awk '{print $1}')
[[ "$observed_wrapper" == "$SMOKE_WRAPPER_SHA256" ]] || {
  echo "smoke wrapper digest mismatch: $observed_wrapper" >&2
  exit 66
}
observed_archive=$(sha256sum "$ARCHIVE" | awk '{print $1}')
[[ "$observed_archive" == "$ARCHIVE_SHA256" ]] || {
  echo "archive digest mismatch: $observed_archive" >&2
  exit 66
}

tar --no-same-owner -xzf "$ARCHIVE" -C "$EXTRACT"
test -x "$VENV/bin/python"
test -d "$SRC/src/goalzendo_hidden_law"
test -d "$SRC/src/goalzendo"
test -d "$SRC/src/goalzendo_interactive"
test -d "$HF_CACHE/hub/models--Qwen--Qwen3.5-0.8B/snapshots/2fc06364715b967f1860aea9cf38778875588b17"
test -d "$HF_CACHE/hub/models--Qwen--Qwen3.5-2B/snapshots/15852e8c16360a2fea060d615a32b45270f8a8fc"

export PYTHONPATH="$SRC/src"
export GOALZENDO_PYTHON="$VENV/bin/python"
export HF_HOME="$HF_CACHE"
export HUGGINGFACE_HUB_CACHE="$HF_CACHE/hub"
export GOALZENDO_OUTPUT_ROOT="$JOB/artifacts/smoke"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HUB_DISABLE_TELEMETRY=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1
cd "$SRC"

sha_check() {
  local expected=$1 path=$2 label=$3 observed
  observed=$(sha256sum "$path" | awk '{print $1}')
  [[ "$observed" == "$expected" ]] || {
    echo "$label digest mismatch: $observed" >&2
    exit 66
  }
}
sha_check a2e31815f3fbe1d65fcd64654f8fb9e51a2d950cf8a2985cc5d90121bad3e53a configs/goalzendo/qwen35_hidden_law_finite_choice.yaml config
sha_check f4752515e3848a98452b89a3f753e8310dc45c131cd33244dc56fa696125883e docs/goalzendo/protocols/qwen35-hidden-law-finite-choice.md protocol
sha_check 8bedacc9b5935e67224676101e12692bb0a5b913f1e9de180a033ac20f0a2d8c scripts/analyze_qwen35_hidden_law.py analyzer
sha_check 849c4db9d2fa546fc3c14018bb71303504c32a82627be8de40187bf5ce5f0c7d scripts/run_qwen35_hidden_law.py runner
sha_check 9340b2f99900870dc8979fd47c8050fb63552fa3f21de7d8dd8681d6f94f4e01 runs/goalzendo/run_qwen35_hidden_law.sh launcher
sha_check d434e42c1b57933601de6a3890d0937389fd5c7c797a2d84777b77e8f3febff3 constraints-goalzendo-qwen35.txt constraints

read -r observed_implementation observed_scientific < <("$VENV/bin/python" - <<'PY'
from goalzendo_hidden_law.artifacts import implementation_provenance
from goalzendo_hidden_law.config import canonical_digest, load_hidden_law_config, scientific_config

config = load_hidden_law_config("configs/goalzendo/qwen35_hidden_law_finite_choice.yaml")
print(
    implementation_provenance(".")["implementation_fingerprint"],
    canonical_digest(scientific_config(config)),
)
PY
)
[[ "$observed_implementation" == 9cd679289fc22ceeb5fe1a9cbc2ca134051faf8491b9fa478be9b13ca81dd087 ]] || {
  echo "implementation fingerprint mismatch: $observed_implementation" >&2
  exit 66
}
[[ "$observed_scientific" == d61cad537e4ae3ea439250bc5e7e2466405e5331be251f6b8f15d0de8683207b ]] || {
  echo "scientific config digest mismatch: $observed_scientific" >&2
  exit 66
}

"$VENV/bin/python" - <<'PY'
import importlib.metadata
import torch
import transformers

expected = {
    "accelerate": "1.14.0",
    "huggingface-hub": "1.5.0",
    "peft": "0.20.0",
    "safetensors": "0.8.0",
    "tokenizers": "0.22.2",
    "transformers": "5.15.0",
}
observed = {name: importlib.metadata.version(name) for name in expected}
if observed != expected:
    raise SystemExit(f"runtime dependency mismatch: expected {expected}, got {observed}")
if torch.__version__ != "2.8.0+cu128" or transformers.__version__ != "5.15.0":
    raise SystemExit(f"runtime torch/transformers mismatch: {torch.__version__}, {transformers.__version__}")
if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
    raise SystemExit("exactly one CUDA device is required")
print(f"GOALZENDO_HIDDEN_LAW_RUNTIME_VERIFIED gpu={torch.cuda.get_device_name(0)}")
PY

runs/goalzendo/run_qwen35_hidden_law.sh plan > "$JOB/launch-plan-a3.json"
sha_check 7b60ed59e8d8842f8c9e6b21830f3d092ebfa94e60427294dce9b6e210659a86 "$JOB/launch-plan-a3.json" launch-plan

for model in 0.8B 2B; do
  for algorithm in process_sft outcome_rl; do
    echo "GOALZENDO_SMOKE_CELL_START attempt=3 model=$model algorithm=$algorithm at=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    runs/goalzendo/run_qwen35_hidden_law.sh smoke "$model" "$algorithm"
    echo "GOALZENDO_SMOKE_CELL_DONE attempt=3 model=$model algorithm=$algorithm at=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  done
done

RERUN_DIR="$JOB/smoke-rerun-a3"
mkdir -p "$RERUN_DIR"
for model in 0.8B 2B; do
  for algorithm in process_sft outcome_rl; do
    runs/goalzendo/run_qwen35_hidden_law.sh smoke "$model" "$algorithm" > "$RERUN_DIR/$model-$algorithm.json"
  done
done

"$VENV/bin/python" - "$JOB/launch-plan-a3.json" "$JOB/artifacts/smoke" "$RERUN_DIR" <<'PY' > "$JOB/smoke-verification-a3.json"
import hashlib
import json
import math
import sys
from pathlib import Path

from goalzendo_hidden_law.artifacts import read_json, verify_completed_run

plan = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
root = Path(sys.argv[2]) / "execution-smoke"
rerun_root = Path(sys.argv[3])
specs = plan["smokes"]
if len(specs) != 4:
    raise SystemExit("registered smoke plan must contain exactly four cells")
expected_pairs = {
    ("Qwen/Qwen3.5-0.8B", "process_sft"),
    ("Qwen/Qwen3.5-0.8B", "outcome_rl"),
    ("Qwen/Qwen3.5-2B", "process_sft"),
    ("Qwen/Qwen3.5-2B", "outcome_rl"),
}
if {(spec["model_name"], spec["algorithm"]) for spec in specs} != expected_pairs:
    raise SystemExit("registered smoke cell identities differ from the four-cell panel")
expected_ids = {spec["smoke_run_id"] for spec in specs}
observed_ids = {path.name for path in root.iterdir() if path.is_dir()}
if observed_ids != expected_ids:
    raise SystemExit(f"smoke run directory census differs: {sorted(observed_ids)}")

models = {
    "Qwen/Qwen3.5-0.8B": (
        "2fc06364715b967f1860aea9cf38778875588b17",
        752_393_024,
    ),
    "Qwen/Qwen3.5-2B": (
        "15852e8c16360a2fea060d615a32b45270f8a8fc",
        1_881_825_088,
    ),
}
dependencies = {
    "accelerate": "1.14.0",
    "huggingface_hub": "1.5.0",
    "peft": "0.20.0",
    "safetensors": "0.8.0",
    "tokenizers": "0.22.2",
    "transformers": "5.15.0",
}
expected_metric_keys = {
    "record_id",
    "kind",
    "step",
    "algorithm",
    "loss",
    "learning_rate",
    "gradient_norm",
}

def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()

verified = []
for spec in specs:
    run_id = spec["smoke_run_id"]
    run = root / run_id
    verify_completed_run(run)
    identity = read_json(run / "identity.json")
    summary = read_json(run / "summary.json")
    status = read_json(run / "status.json")
    manifest = read_json(run / "model-manifest.json")
    metric_lines = [line for line in (run / "metrics.jsonl").read_text(encoding="utf-8").splitlines() if line]
    rows = [json.loads(line) for line in metric_lines]
    revision, parameter_count = models[spec["model_name"]]

    condition = identity.get("condition", {})
    if identity.get("run_id") != run_id or condition.get("model_name") != spec["model_name"]:
        raise SystemExit(f"{run_id}: smoke identity differs from registration")
    if condition.get("model_revision") != revision or condition.get("model_dtype") != "bfloat16":
        raise SystemExit(f"{run_id}: pinned model identity differs")
    if condition.get("algorithm") != spec["algorithm"] or condition.get("seed") != spec["smoke_seed"]:
        raise SystemExit(f"{run_id}: smoke algorithm or seed differs")

    expected_manifest = {
        "requested_model": spec["model_name"],
        "requested_revision": revision,
        "requested_dtype": "bfloat16",
        "action_labels": ["A", "B"],
        "finite_action_alphabets": {
            "binary": ["A", "B"],
            "candidate": ["A", "B", "C", "D"],
            "query": list("ABCDEFGHI"),
        },
        "gradient_checkpointing": True,
        "use_cache": False,
        "full_model_update": True,
        "parameter_count": parameter_count,
        "trainable_parameter_count": parameter_count,
        "trainable_parameter_dtype_counts": {"torch.bfloat16": parameter_count},
        "model_class": "transformers.models.qwen3_5.modeling_qwen3_5.Qwen3_5ForCausalLM",
        "tokenizer_name_or_path": spec["model_name"],
    }
    mismatched = [key for key, value in expected_manifest.items() if manifest.get(key) != value]
    if mismatched:
        raise SystemExit(f"{run_id}: model manifest differs: {mismatched}")
    if manifest.get("resolved_revision") not in (None, revision):
        raise SystemExit(f"{run_id}: resolved model revision differs")
    if manifest.get("tokenizer_resolved_revision") not in (None, revision):
        raise SystemExit(f"{run_id}: resolved tokenizer revision differs")
    if not isinstance(manifest.get("tokenizer_class"), str) or not manifest["tokenizer_class"]:
        raise SystemExit(f"{run_id}: tokenizer class is absent")
    observed_dependencies = manifest.get("dependency_versions", {})
    if any(observed_dependencies.get(key) != value for key, value in dependencies.items()):
        raise SystemExit(f"{run_id}: pinned Python dependency stack differs")
    if str(observed_dependencies.get("torch", "")).split("+", 1)[0] != "2.8.0":
        raise SystemExit(f"{run_id}: torch version differs")
    if observed_dependencies.get("torch_cuda") != "12.8":
        raise SystemExit(f"{run_id}: CUDA version differs")

    if summary.get("smoke") is not True or summary.get("last_step") != 1:
        raise SystemExit(f"{run_id}: summary is not a one-update smoke")
    if summary.get("evaluation") is not None:
        raise SystemExit(f"{run_id}: smoke unexpectedly contains evaluation output")
    if summary.get("algorithm") != spec["algorithm"] or summary.get("model_name") != spec["model_name"]:
        raise SystemExit(f"{run_id}: summary identity differs")
    if status.get("state") != "complete" or status.get("phase") != "complete" or status.get("last_step") != 1:
        raise SystemExit(f"{run_id}: completion status differs")
    if len(rows) != 1 or set(rows[0]) != expected_metric_keys or rows[0].get("kind") != "smoke_update":
        raise SystemExit(f"{run_id}: engineering metric row differs")
    if rows[0].get("step") != 1 or rows[0].get("algorithm") != spec["algorithm"]:
        raise SystemExit(f"{run_id}: engineering metric identity differs")
    for key in ("loss", "learning_rate", "gradient_norm"):
        value = rows[0].get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise SystemExit(f"{run_id}: {key} is not finite")
    if rows[0]["learning_rate"] <= 0 or rows[0]["gradient_norm"] <= 0:
        raise SystemExit(f"{run_id}: update did not record positive LR and gradient norm")
    if (run / "transcripts.jsonl").read_bytes() != b"" or (run / "predictions.jsonl").read_bytes() != b"":
        raise SystemExit(f"{run_id}: smoke contains transcript or prediction rows")
    operational = summary.get("operational", {})
    for key in ("forward_calls", "scored_prompt_count", "scored_prompt_tokens_unpadded"):
        if not isinstance(operational.get(key), int) or operational[key] <= 0:
            raise SystemExit(f"{run_id}: operational counter {key} is absent")
    if not isinstance(operational.get("maximum_prompt_tokens"), int) or not 0 < operational["maximum_prompt_tokens"] <= 1536:
        raise SystemExit(f"{run_id}: prompt-token maximum is invalid")
    checkpoints = run / "checkpoints"
    if checkpoints.exists() and any(checkpoints.iterdir()):
        raise SystemExit(f"{run_id}: checkpoints were not retired")

    selector = spec["model_selector"]
    rerun = rerun_root / f"{selector}-{spec['algorithm']}.json"
    rerun_payload = json.loads(rerun.read_text(encoding="utf-8"))
    if not isinstance(rerun_payload, list) or len(rerun_payload) != 1:
        raise SystemExit(f"{run_id}: rerun receipt is malformed")
    if rerun_payload[0].get("run_id") != run_id or rerun_payload[0].get("status") != "skipped_complete":
        raise SystemExit(f"{run_id}: rerun did not skip the sealed run")

    verified.append(
        {
            "algorithm": spec["algorithm"],
            "completion_sha256": digest(run / "completion.json"),
            "manifest_sha256": digest(run / "model-manifest.json"),
            "metrics_sha256": digest(run / "metrics.jsonl"),
            "model": spec["model_name"],
            "rerun_receipt_sha256": digest(rerun),
            "run_id": run_id,
            "summary_sha256": digest(run / "summary.json"),
        }
    )

print(
    json.dumps(
        {
            "attempt": 3,
            "outcomes_inspected": False,
            "passed": True,
            "registered_run_ids": [spec["smoke_run_id"] for spec in specs],
            "schema": "goalzendo.hidden_law_smoke_verification",
            "schema_version": 1,
            "verified": verified,
        },
        sort_keys=True,
        separators=(",", ":"),
    )
)
PY
