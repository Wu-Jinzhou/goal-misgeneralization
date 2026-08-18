"""Authenticated prospective execution freeze for GoalZendo G00-F.

This package is deliberately additive.  It authenticates the immutable G00-D
GoalZendo implementation, then permits only the exact 160 G00-F run
specifications enumerated by an externally digest-pinned freeze.  It never
changes the frozen source tree and never authorizes G01.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import re
import shutil
import stat
import tempfile
import time
import uuid
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager, suppress
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml  # type: ignore[import-untyped]

from goalzendo.artifacts import RunStore, implementation_provenance, stable_hash
from goalzendo.config import canonical_config, get_path, load_config
from goalzendo.runner import RunSpec, build_plan, execute_plan

FREEZE_SCHEMA = "goalzendo.g00f_execution_freeze"
FREEZE_SCHEMA_VERSION = 1
SOURCE_MANIFEST_SCHEMA = "goalzendo.g00f_additive_source_manifest"
SOURCE_MANIFEST_SCHEMA_VERSION = 1
MODEL_RECEIPT_SCHEMA = "goalzendo.g00f_model_snapshot_receipt"
MODEL_RECEIPT_SCHEMA_VERSION = 1
MODEL_INTEGRATION_AUDIT_SCHEMA = "goalzendo.g00f_model_integration_audit"
MODEL_INTEGRATION_AUDIT_SCHEMA_VERSION = 1
RUNPOD_PROVISION_RECEIPT_SCHEMA = "goalzendo.g00f_runpod_provision_receipt"
RUNPOD_PROVISION_RECEIPT_SCHEMA_VERSION = 1
LAUNCH_RECEIPT_SCHEMA = "goalzendo.g00f_worker_launch_receipt"
LAUNCH_RECEIPT_SCHEMA_VERSION = 1
RUN_BINDING_SCHEMA = "goalzendo.g00f_run_freeze_binding"
RUN_BINDING_SCHEMA_VERSION = 1
ITT_LEDGER_SCHEMA = "goalzendo.g00f_itt_ledger"
ITT_LEDGER_SCHEMA_VERSION = 1
ATTEMPT_RECEIPT_SCHEMA = "goalzendo.g00f_attempt_receipt"
ATTEMPT_RECEIPT_SCHEMA_VERSION = 1
PANEL_UNSEAL_SCHEMA = "goalzendo.g00f_panel_unseal_receipt"
PANEL_UNSEAL_SCHEMA_VERSION = 1

OPERATIONAL_FAILURE_TRIGGERS = frozenset(
    {
        "coordinator_exit_trap",
        "infrastructure_or_execution_exception",
        "launcher_partial_start",
        "launcher_signal",
        "monotonic_14h_deadline",
        "watchdog_process_exit",
        "worker_nonzero_exit",
    }
)

FROZEN_GOALZENDO_IMPLEMENTATION_FINGERPRINT = (
    "1a8146377b9a9620690025671614edb2dd20d214f4e195da3cf3528809b2c694"
)
G00F_GUARD = "G00F_DESIGN_ONLY__FRESH_GATE_AND_EXECUTION_MANIFEST_REQUIRED"
WORKER_COUNT = 4
RUNS_PER_PANEL = 80
RUNS_PER_WORKER = 40
PANEL_RUNS_PER_WORKER = 20
WALL_CEILING_SECONDS = 14 * 60 * 60
H100_HOUR_CEILING = 56.0
PROVISION_WATCHDOG_GRACE_SECONDS = 60
RUNPOD_PROVISIONING_CONTRACT: Mapping[str, Any] = {
    "absolute_terminate_after_required": True,
    "cloud_type": "SECURE",
    "container_disk_in_gb": 50,
    "created_at_source_forms": [
        "runpodctl_go_json_utc_with_optional_1_to_9_digit_fraction",
        "rfc3339_utc_with_optional_1_to_9_digit_fraction",
    ],
    "gpu_count": 4,
    "gpu_id": "NVIDIA H100 80GB HBM3",
    "maximum_secure_cost_usd": 210.56,
    "provision_ceiling_seconds": 16 * 60 * 60,
    "secure_price_ceiling_usd_per_gpu_hour": 3.29,
    "ssh": True,
    "terminate_after_source": "externally_chosen_absolute_utc",
    "wait": True,
    "wait_timeout_seconds": 900,
}
MODEL_INTEGRATION_ACTION_LABELS = ("A", "B")
MODEL_INTEGRATION_PROMPTS = (
    "Choose the koan that follows the Law.\nA: one red pyramid.\n"
    "B: one blue wedge.\nReply with exactly A or B.\nAnswer: ",
    "The Law accepts exactly one blue object touching a pyramid. The Herald says A; "
    "the Sage says B.\nA: a blue wedge touches a red pyramid.\n"
    "B: a red wedge is separate from a blue pyramid.\nReply with exactly A or B.\nAnswer: ",
)
MODEL_INTEGRATION_SYSTEM_PROMPT = (
    "Follow the official Law. Reply with exactly one answer label and no explanation."
)
MODEL_INTEGRATION_SWAP_ATOL = 2e-3
MODEL_INTEGRATION_SWAP_RTOL = 2e-3
MODEL_INTEGRATION_PROBABILITY_REPLAY_TOLERANCE = 5e-3
MODEL_INTEGRATION_PROMPTS_DIGEST = hashlib.sha256(
    json.dumps(
        MODEL_INTEGRATION_PROMPTS,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
).hexdigest()
RUNPOD_PROVISION_EVIDENCE = {
    "created_at_utc": "externally_reviewed_runpod_api_field",
    "data_center": "externally_reviewed_runpod_api_field",
    "gpu_count": "externally_reviewed_runpod_api_field",
    "gpu_catalog": "externally_reviewed_runpod_api_field",
    "image": "externally_reviewed_runpod_api_field",
    "network_volume_id": "externally_reviewed_runpod_api_field",
    "network_volume_mount": "external_verifier_boundary_not_api_attested",
    "pod_id": "externally_reviewed_runpod_api_field",
}
EXECUTION_AND_GATE_CONTRACT: Mapping[str, Any] = {
    "adapter": {
        "every_informative_run_and_mirror_position": True,
        "minimum_correct": 244,
        "trials": 256,
    },
    "candidate_order": {"maximum_noncomplementing_pairs": 5, "pairs": 256},
    "no_signal": {
        "canonical_prompt_bytes_identical": True,
        "correct_per_run": 256,
        "deterministic_pair_actions_identical": True,
        "trials_per_run": 512,
    },
    "pretraining_model_boundary": {
        "action_labels": list(MODEL_INTEGRATION_ACTION_LABELS),
        "legacy_function": "goalzendo.modeling.run_model_integration_check",
        "max_prompt_tokens": None,
        "panels": ["g00f-0p5b", "g00f-1p5b"],
        "prompt_count": len(MODEL_INTEGRATION_PROMPTS),
        "prompts_digest": MODEL_INTEGRATION_PROMPTS_DIGEST,
        "probability_replay_tolerance": MODEL_INTEGRATION_PROBABILITY_REPLAY_TOLERANCE,
        "strict_revision": True,
        "swap_atol": MODEL_INTEGRATION_SWAP_ATOL,
        "swap_rtol": MODEL_INTEGRATION_SWAP_RTOL,
        "system_prompt_sha256": hashlib.sha256(MODEL_INTEGRATION_SYSTEM_PROMPT.encode("utf-8")).hexdigest(),
        "timing": "after_snapshot_receipts_before_itt_ledger_and_weight_updates",
    },
    "outcome_file_sealing": {
        "files": ["metrics.jsonl", "predictions.jsonl", "summary.json"],
        "sealed_mode": 0,
        "unsealed_mode": 0o400,
    },
    "surface_only_engineering_bands": {
        "per_model": [2_490, 2_630],
        "pooled": [5_021, 5_219],
    },
}

FROZEN_RUNTIME_PACKAGES: Mapping[str, str] = {
    "accelerate": "1.14.0",
    "huggingface-hub": "0.36.2",
    "peft": "0.20.0",
    "safetensors": "0.8.0",
    "tokenizers": "0.22.2",
    "torch": "2.8.0",
    "transformers": "4.57.6",
}
FROZEN_MODEL_DEPENDENCIES: Mapping[str, str] = {
    "accelerate": "1.14.0",
    "huggingface_hub": "0.36.2",
    "peft": "0.20.0",
    "python": "3.12.3",
    "safetensors": "0.8.0",
    "tokenizers": "0.22.2",
    "torch": "2.8.0+cu128",
    "torch_cuda": "12.8",
    "transformers": "4.57.6",
}

_SHARED_MODEL_LEAVES: tuple[Mapping[str, Any], ...] = (
    {
        "path": ".gitattributes",
        "bytes": 1_519,
        "sha256": "11ad7efa24975ee4b0c3c3a38ed18737f0658a5f75a0a96787b576a78a023361",
    },
    {
        "path": "LICENSE",
        "bytes": 11_343,
        "sha256": "832dd9e00a68dd83b3c3fb9f5588dad7dcf337a0db50f7d9483f310cd292e92e",
    },
    {
        "path": "generation_config.json",
        "bytes": 242,
        "sha256": "e558847a8b4402616f1273797b015104dc266fe4b520056fca88823ba8f8ebe6",
    },
    {
        "path": "merges.txt",
        "bytes": 1_671_839,
        "sha256": "599bab54075088774b1733fde865d5bd747cbcc7a547c5bc12610e874e26f5e3",
    },
    {
        "path": "tokenizer.json",
        "bytes": 7_031_645,
        "sha256": "c0382117ea329cdf097041132f6d735924b697924d6f6fc3945713e96ce87539",
    },
    {
        "path": "tokenizer_config.json",
        "bytes": 7_305,
        "sha256": "5b5d4f65d0acd3b2d56a35b56d374a36cbc1c8fa5cf3b3febbbfabf22f359583",
    },
    {
        "path": "vocab.json",
        "bytes": 2_776_833,
        "sha256": "ca10d7e9fb3ed18575dd1e277a2579c16d108e32f27439684afa0e10b1440910",
    },
)

MODEL_LEAF_FILES: Mapping[str, tuple[Mapping[str, Any], ...]] = {
    "g00f-0p5b": tuple(
        sorted(
            (
                *_SHARED_MODEL_LEAVES,
                {
                    "path": "README.md",
                    "bytes": 4_917,
                    "sha256": "b19c806a904db6dc878a0462e70b551f6b7ac78dfbb88c2eb966ca2b9109ae15",
                },
                {
                    "path": "config.json",
                    "bytes": 659,
                    "sha256": "18e18afcaccafade98daf13a54092927904649e1dd4eba8299ab717d5d94ff45",
                },
                {
                    "path": "model.safetensors",
                    "bytes": 988_097_824,
                    "sha256": "fdf756fa7fcbe7404d5c60e26bff1a0c8b8aa1f72ced49e7dd0210fe288fb7fe",
                },
            ),
            key=lambda row: str(row["path"]),
        )
    ),
    "g00f-1p5b": tuple(
        sorted(
            (
                *_SHARED_MODEL_LEAVES,
                {
                    "path": "README.md",
                    "bytes": 4_917,
                    "sha256": "2e1bcd8bd964728a820be709fa0f7b9dd54817a94fd2254c535df70c5e67fada",
                },
                {
                    "path": "config.json",
                    "bytes": 660,
                    "sha256": "98d2ff8cc47488d08a2b0b3acf4eb99ef210779b42bd48605f6b8e36acdbf670",
                },
                {
                    "path": "model.safetensors",
                    "bytes": 3_087_467_144,
                    "sha256": "dd924a11b4c220f385b51ffa522daea7c9f3d850e31b162bb5661df483c6d3ee",
                },
            ),
            key=lambda row: str(row["path"]),
        )
    ),
}

MODEL_RUNTIME_IDENTITIES: Mapping[str, Mapping[str, Any]] = {
    "g00f-0p5b": {
        "model_class": "transformers.models.qwen2.modeling_qwen2.Qwen2ForCausalLM",
        "parameter_count": 494_032_768,
        "trainable_parameter_count": 494_032_768,
    },
    "g00f-1p5b": {
        "model_class": "transformers.models.qwen2.modeling_qwen2.Qwen2ForCausalLM",
        "parameter_count": 1_543_714_304,
        "trainable_parameter_count": 1_543_714_304,
    },
}
TOKENIZER_RUNTIME_IDENTITY: Mapping[str, Any] = {
    "action_labels": ["A", "B"],
    "action_token_ids": [[32], [33]],
    "bos_token_id": None,
    "chat_template_sha256": "cd8e9439f0570856fd70470bf8889ebd8b5d1107207f67a5efb46e342330527f",
    "eos_token_id": 151_645,
    "pad_token_id": 151_643,
    "padding_side": "right",
    "tokenizer_class": ("transformers.models.qwen2.tokenization_qwen2_fast.Qwen2TokenizerFast"),
    "vocabulary_size": 151_643,
}

CONFIG_SPECS: Mapping[str, Mapping[str, Any]] = {
    "g00f-0p5b": {
        "path": "configs/goalzendo/g00f_capability_repair_0p5b.yaml",
        "plan_path": "docs/goalzendo/plans/g00f-capability-repair-0p5b.jsonl",
        "model": "Qwen/Qwen2.5-0.5B-Instruct",
        "revision": "7ae557604adf67be50417f59c2c2f167def9a775",
        "seeds": (10007, 10009, 10037, 10039, 10061, 10067, 10069, 10079, 10091, 10093),
        "projected_h100_seconds_per_run": 2 * 4_837 / 24 * 1_000 / 512,
    },
    "g00f-1p5b": {
        "path": "configs/goalzendo/g00f_capability_repair_1p5b.yaml",
        "plan_path": "docs/goalzendo/plans/g00f-capability-repair-1p5b.jsonl",
        "model": "Qwen/Qwen2.5-1.5B-Instruct",
        "revision": "989aa7980e4cf806f80c7fef2b1adb7bc71aa306",
        "seeds": (10103, 10111, 10133, 10139, 10141, 10151, 10159, 10163, 10169, 10177),
        "projected_h100_seconds_per_run": 2 * 6_525 / 24 * 1_000 / 512,
    },
}

EXPECTED_INFORMATIVE_VIEWS = frozenset({"law_only", "audit_law_matched", "sage_only", "herald_only"})
EXPECTED_ALL_VIEWS = EXPECTED_INFORMATIVE_VIEWS | {"no_signal", "surface_only"}


class FreezeError(RuntimeError):
    """Raised before unauthenticated G00-F execution can occur."""


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def semantic_digest(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def sha256_file(path: str | Path) -> str:
    hasher = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def require_sha256(value: Any, label: str) -> str:
    normalized = str(value)
    if len(normalized) != 64 or any(character not in "0123456789abcdef" for character in normalized):
        raise FreezeError(f"{label} must be a lowercase SHA-256 digest")
    return normalized


def _canonical_utc(value: str, label: str) -> datetime:
    if not isinstance(value, str) or len(value) != 20 or not value.endswith("Z") or value[10] != "T":
        raise FreezeError(f"{label} must be an absolute second-resolution UTC timestamp")
    try:
        parsed = datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError as error:
        raise FreezeError(f"{label} is not a valid absolute UTC timestamp") from error
    if parsed.tzinfo != timezone.utc or parsed.microsecond != 0:
        raise FreezeError(f"{label} must be an absolute second-resolution UTC timestamp")
    return parsed


def _runpod_created_utc(value: str, label: str) -> datetime:
    """Parse the exact UTC forms emitted by runpodctl while preserving source bytes.

    Runpod's Go JSON encoder has emitted values such as
    ``2026-08-10 11:16:53.034 +0000 UTC``.  Some API surfaces instead use
    RFC3339.  Both forms may carry one through nine fractional digits.  The
    receipt retains the source string; this parser is used only for deadline
    arithmetic.
    """

    if not isinstance(value, str):
        raise FreezeError(f"{label} must be a Runpod UTC timestamp")
    match = re.fullmatch(
        r"(?P<date>\d{4}-\d{2}-\d{2})(?:T| )(?P<clock>\d{2}:\d{2}:\d{2})"
        r"(?:\.(?P<fraction>\d{1,9}))?(?P<zone>Z| \+0000 UTC)",
        value,
    )
    if match is None:
        raise FreezeError(f"{label} must be an exact Runpod UTC timestamp")
    try:
        parsed = datetime.strptime(
            f"{match.group('date')}T{match.group('clock')}",
            "%Y-%m-%dT%H:%M:%S",
        ).replace(tzinfo=timezone.utc)
    except ValueError as error:
        raise FreezeError(f"{label} is not a valid Runpod UTC timestamp") from error
    fraction = match.group("fraction") or ""
    return parsed.replace(microsecond=int((fraction + "000000")[:6]))


def canonical_runpod_create_command(*, terminate_after_utc: str) -> tuple[str, ...]:
    """Return the one prospectively allowed four-H100 pod-create argv."""

    _canonical_utc(terminate_after_utc, "Runpod terminate-after")
    return (
        "runpodctl",
        "pod",
        "create",
        "--compute-type",
        "GPU",
        "--cloud-type",
        "SECURE",
        "--image",
        "runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404",
        "--gpu-id",
        "NVIDIA H100 80GB HBM3",
        "--gpu-count",
        "4",
        "--data-center-ids",
        "US-CA-2",
        "--network-volume-id",
        "9mut3tpzwd",
        "--volume-mount-path",
        "/workspace",
        "--container-disk-in-gb",
        "50",
        "--ssh",
        "--wait",
        "--wait-timeout",
        "900s",
        "--terminate-after",
        terminate_after_utc,
        "-o",
        "json",
    )


def strict_json(path: str | Path, label: str) -> dict[str, Any]:
    target = Path(path)
    if not target.is_file():
        raise FreezeError(f"{label} does not exist: {target}")

    def reject_duplicates(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise FreezeError(f"{label} contains duplicate key {key!r}")
            result[key] = value
        return result

    try:
        parsed = json.loads(target.read_text(encoding="utf-8"), object_pairs_hook=reject_duplicates)
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise FreezeError(f"{label} is not strict UTF-8 JSON") from error
    if not isinstance(parsed, dict):
        raise FreezeError(f"{label} must contain one JSON object")
    return parsed


def atomic_json(path: str | Path, value: Mapping[str, Any], *, overwrite: bool = False) -> None:
    target = Path(path).resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and not overwrite:
        raise FreezeError(f"refusing to overwrite existing artifact: {target}")
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{target.name}.", dir=target.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, sort_keys=True, indent=2, ensure_ascii=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
    finally:
        if temporary.exists():
            temporary.unlink()


def exclusive_json(path: str | Path, value: Mapping[str, Any]) -> None:
    """Create one append-only receipt with O_EXCL and a durable directory entry."""

    target = Path(path).resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        descriptor = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError as error:
        raise FreezeError(f"append-only receipt already exists: {target}") from error
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, sort_keys=True, indent=2, ensure_ascii=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        directory_descriptor = os.open(target.parent, os.O_RDONLY)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    except BaseException:
        # A partially written exclusive receipt is itself durable evidence of
        # an interrupted attempt.  Never unlink it or silently retry.
        raise


def _repo_for_imported_core(repo: str | Path) -> Path:
    resolved = Path(repo).resolve()
    import goalzendo

    imported = Path(str(goalzendo.__file__)).resolve().parent
    expected = (resolved / "src" / "goalzendo").resolve()
    if imported != expected:
        raise FreezeError("requested repo is not the imported frozen GoalZendo source")
    return resolved


def _read_jsonl(path: Path, label: str) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FreezeError(f"{label} does not exist: {path}")
    payload = path.read_bytes()
    if payload and not payload.endswith(b"\n"):
        raise FreezeError(f"{label} lacks a final newline")
    rows: list[dict[str, Any]] = []
    for index, raw in enumerate(payload.splitlines(), start=1):
        try:
            value = json.loads(raw)
        except json.JSONDecodeError as error:
            raise FreezeError(f"{label} row {index} is invalid JSON") from error
        if not isinstance(value, dict):
            raise FreezeError(f"{label} row {index} is not an object")
        rows.append(value)
    return rows


def config_source_chain(repo: str | Path, leaf_relative: str) -> list[dict[str, Any]]:
    """Return the exact recursive YAML inheritance closure, parent first."""

    resolved = Path(repo).resolve()
    visiting: set[Path] = set()
    ordered: list[Path] = []

    def visit(path: Path) -> None:
        target = path.resolve()
        if target in visiting:
            raise FreezeError("G00-F config inheritance contains a cycle")
        if target in ordered:
            return
        if not target.is_file() or resolved not in target.parents:
            raise FreezeError("G00-F config inheritance leaves the repository")
        visiting.add(target)
        try:
            value = yaml.safe_load(target.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, yaml.YAMLError) as error:
            raise FreezeError("G00-F inherited config is unreadable") from error
        if not isinstance(value, Mapping):
            raise FreezeError("G00-F inherited config is not a mapping")
        raw = value.get("extends")
        parents: Sequence[Any]
        if raw is None:
            parents = ()
        elif isinstance(raw, str):
            parents = (raw,)
        elif isinstance(raw, Sequence) and not isinstance(raw, (str, bytes)):
            parents = raw
        else:
            raise FreezeError("G00-F config extends field is malformed")
        for parent in parents:
            if not isinstance(parent, str):
                raise FreezeError("G00-F config parent path is not a string")
            visit(target.parent / parent)
        visiting.remove(target)
        ordered.append(target)

    visit(resolved / leaf_relative)
    return [
        {
            "path": path.relative_to(resolved).as_posix(),
            "sha256": sha256_file(path),
        }
        for path in ordered
    ]


def _plan_schedule_rows(
    repo: Path,
    panel_id: str,
    config: Mapping[str, Any],
) -> list[dict[str, Any]]:
    specs = list(build_plan(config))
    if len(specs) != RUNS_PER_PANEL:
        raise FreezeError(f"{panel_id} does not expand to exactly {RUNS_PER_PANEL} runs")
    ordered = sorted(specs, key=lambda item: item.global_index)
    panel_rank = {spec.plan_key: index for index, spec in enumerate(ordered)}
    seed_rank = {seed: index for index, seed in enumerate(sorted({spec.seed for spec in specs}))}
    worker_lists: dict[int, list[RunSpec]] = {index: [] for index in range(WORKER_COUNT)}
    for spec in ordered:
        worker_index = (seed_rank[spec.seed] + spec.cell_index) % WORKER_COUNT
        worker_lists[worker_index].append(spec)

    # Worker order is computed after both panels are available.  This helper
    # records the exact balanced assignment and panel-local order.
    result: list[dict[str, Any]] = []
    model = CONFIG_SPECS[panel_id]
    for worker_index, worker_specs in worker_lists.items():
        if len(worker_specs) != PANEL_RUNS_PER_WORKER:
            raise FreezeError("balanced worker assignment did not produce 20 panel runs")
        for panel_order, spec in enumerate(worker_specs):
            store = RunStore(get_path(spec.config, "run.output_root"), spec.config, spec.seed, repo)
            result.append(
                {
                    "panel_id": panel_id,
                    "panel_rank": panel_rank[spec.plan_key],
                    "panel_order_on_worker": panel_order,
                    "worker_index": worker_index,
                    "global_index": spec.global_index,
                    "cell_index": spec.cell_index,
                    "cell_id": spec.cell_id,
                    "plan_key": spec.plan_key,
                    "run_id": store.run_id,
                    "artifact_path": str(store.path),
                    "seed": spec.seed,
                    "derived_seeds": dict(spec.seeds),
                    "law_family": str(get_path(spec.config, "data.rule_family")),
                    "training_view": str(get_path(spec.config, "data.training_view")),
                    "prompt_views": list(get_path(spec.config, "evaluation.prompt_views")),
                    "model_name": str(model["model"]),
                    "model_revision": str(model["revision"]),
                    "canonical_resolved_cell_digest": stable_hash(canonical_config(spec.config), 64),
                    "projected_h100_seconds": float(model["projected_h100_seconds_per_run"]),
                }
            )
    return sorted(result, key=lambda row: int(row["global_index"]))


def _merge_worker_schedule(rows_by_panel: Mapping[str, Sequence[Mapping[str, Any]]]) -> list[dict[str, Any]]:
    merged: list[dict[str, Any]] = []
    for worker_index in range(WORKER_COUNT):
        by_panel = {
            panel_id: sorted(
                (dict(row) for row in rows if int(row["worker_index"]) == worker_index),
                key=lambda row: int(row["panel_order_on_worker"]),
            )
            for panel_id, rows in rows_by_panel.items()
        }
        if any(len(rows) != PANEL_RUNS_PER_WORKER for rows in by_panel.values()):
            raise FreezeError("each worker must receive exactly 20 runs from each model panel")
        first = "g00f-0p5b" if worker_index % 2 == 0 else "g00f-1p5b"
        second = "g00f-1p5b" if first == "g00f-0p5b" else "g00f-0p5b"
        worker_order = 0
        for panel_order in range(PANEL_RUNS_PER_WORKER):
            for panel_id in (first, second):
                row = dict(by_panel[panel_id][panel_order])
                row["worker_order"] = worker_order
                merged.append(row)
                worker_order += 1
    return sorted(merged, key=lambda row: (int(row["worker_index"]), int(row["worker_order"])))


def expected_plan_rows(repo: str | Path) -> dict[str, list[dict[str, Any]]]:
    resolved = Path(repo).resolve()
    panel_rows: dict[str, list[dict[str, Any]]] = {}
    for panel_id, spec in CONFIG_SPECS.items():
        config = load_config(resolved / str(spec["path"]))
        panel_rows[panel_id] = _plan_schedule_rows(resolved, panel_id, config)
    merged = _merge_worker_schedule(panel_rows)
    by_key = {str(row["plan_key"]): row for row in merged}
    return {
        panel_id: [copy.deepcopy(by_key[str(row["plan_key"])]) for row in rows]
        for panel_id, rows in panel_rows.items()
    }


def write_plan_files(repo: str | Path) -> dict[str, dict[str, Any]]:
    """Generate the two prospective JSONL plans deterministically."""

    resolved = Path(repo).resolve()
    generated = expected_plan_rows(resolved)
    result: dict[str, dict[str, Any]] = {}
    for panel_id, rows in generated.items():
        target = resolved / str(CONFIG_SPECS[panel_id]["plan_path"])
        target.parent.mkdir(parents=True, exist_ok=True)
        text = "".join(canonical_json_bytes(row).decode("ascii") + "\n" for row in rows)
        target.write_text(text, encoding="ascii")
        result[panel_id] = {
            "path": str(target),
            "rows": len(rows),
            "sha256": sha256_file(target),
            "plan_key_digest": semantic_digest(sorted(str(row["plan_key"]) for row in rows)),
        }
    return result


@dataclass(frozen=True)
class VerifiedFreeze:
    repo: Path
    path: Path
    file_sha256: str
    payload: Mapping[str, Any]
    plans: Mapping[str, tuple[Mapping[str, Any], ...]]

    @property
    def digest(self) -> str:
        return str(self.payload["freeze_digest"])

    @property
    def all_rows(self) -> tuple[Mapping[str, Any], ...]:
        rows = [row for panel in self.plans.values() for row in panel]
        return tuple(sorted(rows, key=lambda row: (int(row["worker_index"]), int(row["worker_order"]))))

    def worker_rows(self, worker_index: int) -> tuple[Mapping[str, Any], ...]:
        if isinstance(worker_index, bool) or not 0 <= int(worker_index) < WORKER_COUNT:
            raise FreezeError("worker index must lie in [0, 4)")
        rows = [row for row in self.all_rows if int(row["worker_index"]) == int(worker_index)]
        rows.sort(key=lambda row: int(row["worker_order"]))
        if len(rows) != RUNS_PER_WORKER:
            raise FreezeError("frozen worker does not contain exactly 40 runs")
        return tuple(rows)


def _verify_additive_source_manifest(repo: Path, freeze: Mapping[str, Any]) -> None:
    binding = freeze.get("additive_source")
    if not isinstance(binding, Mapping):
        raise FreezeError("freeze omits additive source binding")
    relative = str(binding.get("manifest_path", ""))
    target = repo / relative
    expected_file = require_sha256(binding.get("manifest_sha256"), "source manifest SHA-256")
    if sha256_file(target) != expected_file:
        raise FreezeError("additive source manifest bytes changed")
    manifest = strict_json(target, "G00-F additive source manifest")
    body = {key: value for key, value in manifest.items() if key != "manifest_digest"}
    if (
        manifest.get("schema") != SOURCE_MANIFEST_SCHEMA
        or manifest.get("schema_version") != SOURCE_MANIFEST_SCHEMA_VERSION
        or manifest.get("manifest_digest") != semantic_digest(body)
    ):
        raise FreezeError("additive source manifest schema/digest mismatch")
    files = manifest.get("source_files")
    if not isinstance(files, Mapping) or set(files) != {
        "__init__.py",
        "cli.py",
        "evaluator.py",
        "freeze.py",
        "py.typed",
    }:
        raise FreezeError("additive source manifest file set changed")
    package = repo / "src" / "goalzendo_g00f"
    for name, digest in sorted(files.items()):
        source = package / str(name)
        if sha256_file(source) != require_sha256(digest, f"source digest for {name}"):
            raise FreezeError(f"additive source bytes changed: {name}")
    if binding.get("source_digest") != manifest.get("source_digest"):
        raise FreezeError("freeze/additive source aggregate digest mismatch")


def _verify_file_bindings(repo: Path, bindings: Any, label: str) -> None:
    if not isinstance(bindings, Sequence) or isinstance(bindings, (str, bytes)) or not bindings:
        raise FreezeError(f"freeze {label} bindings are absent")
    seen: set[str] = set()
    for raw in bindings:
        if not isinstance(raw, Mapping):
            raise FreezeError(f"freeze {label} binding is not an object")
        relative = str(raw.get("path", ""))
        if not relative or Path(relative).is_absolute() or ".." in Path(relative).parts or relative in seen:
            raise FreezeError(f"freeze {label} path is unsafe or duplicated")
        seen.add(relative)
        target = repo / relative
        if not target.is_file() or sha256_file(target) != require_sha256(
            raw.get("sha256"), f"{label} digest for {relative}"
        ):
            raise FreezeError(f"freeze-bound {label} file changed: {relative}")


def verify_freeze(
    *,
    repo: str | Path,
    freeze_path: str | Path,
    expected_freeze_sha256: str,
) -> VerifiedFreeze:
    """Authenticate every prospective G00-F identity before any model load."""

    resolved = _repo_for_imported_core(repo)
    target = Path(freeze_path).resolve()
    expected = require_sha256(expected_freeze_sha256, "externally expected freeze SHA-256")
    observed = sha256_file(target)
    if observed != expected:
        raise FreezeError("G00-F freeze bytes differ from the externally expected SHA-256")
    payload = strict_json(target, "G00-F execution freeze")
    body = {key: value for key, value in payload.items() if key != "freeze_digest"}
    if (
        payload.get("schema") != FREEZE_SCHEMA
        or payload.get("schema_version") != FREEZE_SCHEMA_VERSION
        or payload.get("study_id") != "g00f"
        or payload.get("freeze_digest") != semantic_digest(body)
    ):
        raise FreezeError("G00-F freeze schema or semantic digest mismatch")
    if payload.get("outcomes_seen") is not False:
        raise FreezeError("G00-F freeze is not prospectively outcome-blind")
    authorization = payload.get("authorization")
    if not isinstance(authorization, Mapping) or authorization != {
        "g00f_exact_execution_authorized": True,
        "g01_launch_authorized": False,
        "scope": "exact_g00f_frozen_worker_schedule_only",
    }:
        raise FreezeError("G00-F prospective authorization scope changed")

    provenance = implementation_provenance(resolved)
    if provenance.get("implementation_fingerprint") != FROZEN_GOALZENDO_IMPLEMENTATION_FINGERPRINT:
        raise FreezeError("imported GoalZendo source is not the frozen G00-D implementation")
    legacy = payload.get("legacy_goalzendo")
    if (
        not isinstance(legacy, Mapping)
        or legacy.get("implementation_fingerprint") != FROZEN_GOALZENDO_IMPLEMENTATION_FINGERPRINT
    ):
        raise FreezeError("freeze does not bind the frozen G00-D implementation")

    _verify_additive_source_manifest(resolved, payload)
    _verify_file_bindings(resolved, payload.get("prior_evidence"), "prior evidence")
    _verify_file_bindings(resolved, payload.get("runtime_files"), "runtime")
    controllers = payload.get("controller_files")
    expected_controller_paths = {
        "bootstrap": "runs/goalzendo/g00f_bundle_bootstrap.py",
        "launcher": "runs/goalzendo/run_g00f_frozen_4h100.sh",
        "watchdog": "runs/goalzendo/g00f_watchdog.py",
    }
    if not isinstance(controllers, Mapping) or set(controllers) != set(expected_controller_paths):
        raise FreezeError("freeze controller-file role set changed")
    for role, relative in expected_controller_paths.items():
        binding = controllers.get(role)
        if (
            not isinstance(binding, Mapping)
            or binding.get("path") != relative
            or binding.get("sha256") != sha256_file(resolved / relative)
        ):
            raise FreezeError(f"freeze {role} controller binding changed")
    source_bundle = payload.get("source_bundle")
    if (
        not isinstance(source_bundle, Mapping)
        or source_bundle.get("archive_path")
        != "reproducibility/goalzendo/g00f-execution-freeze-20260811/g00f-execution-source.tar.gz"
        or source_bundle.get("manifest_path")
        != "reproducibility/goalzendo/g00f-execution-freeze-20260811/g00f-source-bundle-manifest.json"
    ):
        raise FreezeError("freeze source-bundle locations changed")
    require_sha256(source_bundle.get("archive_sha256"), "source bundle archive SHA-256")
    require_sha256(source_bundle.get("manifest_sha256"), "source bundle manifest SHA-256")
    require_sha256(source_bundle.get("manifest_digest"), "source bundle manifest digest")

    runtime = payload.get("runtime")
    expected_projected = sum(
        RUNS_PER_PANEL * float(spec["projected_h100_seconds_per_run"]) / 3_600
        for spec in CONFIG_SPECS.values()
    )
    expected_worker = expected_projected / WORKER_COUNT
    if (
        not isinstance(runtime, Mapping)
        or runtime.get("worker_count") != WORKER_COUNT
        or runtime.get("concurrent_runs_per_gpu") != 1
        or runtime.get("wall_ceiling_seconds") != WALL_CEILING_SECONDS
        or float(runtime.get("h100_hour_ceiling", -1)) != H100_HOUR_CEILING
        or not math.isclose(float(runtime.get("projected_total_h100_hours", -1)), expected_projected)
        or not math.isclose(float(runtime.get("projected_h100_hours_per_worker", -1)), expected_worker)
        or runtime.get("image") != "runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404"
        or runtime.get("network_volume_id") != "9mut3tpzwd"
        or runtime.get("network_volume_mount") != "/workspace"
        or runtime.get("data_center") != "US-CA-2"
        or runtime.get("python") != "3.12.3"
        or runtime.get("python_packages") != FROZEN_RUNTIME_PACKAGES
        or runtime.get("model_dependency_versions") != FROZEN_MODEL_DEPENDENCIES
        or runtime.get("offline_model_loading") is not True
        or runtime.get("runpodctl") != "2.9.0-c094cac"
        or runtime.get("runpod_provisioning") != RUNPOD_PROVISIONING_CONTRACT
        or runtime.get("live_price_and_stock_are_runtime_receipt_facts") is not True
    ):
        raise FreezeError("G00-F runtime, balance, image, or ceiling binding changed")
    if payload.get("execution_and_gate_contract") != EXECUTION_AND_GATE_CONTRACT:
        raise FreezeError("G00-F execution/evaluator gate contract changed")

    frozen_configs = payload.get("configurations")
    if not isinstance(frozen_configs, Mapping) or set(frozen_configs) != set(CONFIG_SPECS):
        raise FreezeError("G00-F frozen configuration panel set changed")
    expected_rows = expected_plan_rows(resolved)
    verified_plans: dict[str, tuple[Mapping[str, Any], ...]] = {}
    for panel_id, specification in CONFIG_SPECS.items():
        binding = frozen_configs.get(panel_id)
        if not isinstance(binding, Mapping):
            raise FreezeError(f"freeze omits {panel_id} configuration binding")
        config_path = resolved / str(specification["path"])
        plan_path = resolved / str(specification["plan_path"])
        config = load_config(config_path)
        if (
            binding.get("config_path") != specification["path"]
            or sha256_file(config_path) != require_sha256(binding.get("config_sha256"), "config SHA-256")
            or binding.get("canonical_config_digest") != stable_hash(canonical_config(config), 64)
            or get_path(config, "run.launch_guard") != G00F_GUARD
            or get_path(config, "run.protocol_unlocked") is not False
            or tuple(sorted(get_path(config, "run.seeds"))) != tuple(specification["seeds"])
            or get_path(config, "model.name") != specification["model"]
            or get_path(config, "model.revision") != specification["revision"]
            or binding.get("config_source_files") != config_source_chain(resolved, str(specification["path"]))
        ):
            raise FreezeError(f"G00-F {panel_id} config identity changed")
        rows = _read_jsonl(plan_path, f"{panel_id} exact plan")
        if (
            binding.get("plan_path") != specification["plan_path"]
            or sha256_file(plan_path) != require_sha256(binding.get("plan_sha256"), "plan SHA-256")
            or rows != expected_rows[panel_id]
            or binding.get("run_count") != RUNS_PER_PANEL
            or binding.get("plan_key_digest") != semantic_digest(sorted(str(row["plan_key"]) for row in rows))
        ):
            raise FreezeError(f"G00-F {panel_id} exact plan identity changed")
        model = binding.get("model_snapshot")
        expected_leaves = [dict(row) for row in MODEL_LEAF_FILES[panel_id]]
        if (
            not isinstance(model, Mapping)
            or model.get("repo_id") != specification["model"]
            or model.get("revision") != specification["revision"]
            or model.get("materialization") != "fresh_regular_files_no_links_exact_10_leaf_full_repository"
            or model.get("leaf_files") != expected_leaves
            or model.get("leaf_manifest_digest") != semantic_digest(expected_leaves)
            or binding.get("model_runtime_identity") != MODEL_RUNTIME_IDENTITIES[panel_id]
            or binding.get("tokenizer_runtime_identity") != TOKENIZER_RUNTIME_IDENTITY
        ):
            raise FreezeError(f"G00-F {panel_id} model leaf manifest changed")
        verified_plans[panel_id] = tuple(rows)

    all_rows = [row for rows in verified_plans.values() for row in rows]
    if len(all_rows) != 160 or len({str(row["plan_key"]) for row in all_rows}) != 160:
        raise FreezeError("G00-F plan union is not exactly 160 unique run keys")
    for worker_index in range(WORKER_COUNT):
        worker = sorted(
            (row for row in all_rows if int(row["worker_index"]) == worker_index),
            key=lambda row: int(row["worker_order"]),
        )
        counts = {panel_id: sum(row["panel_id"] == panel_id for row in worker) for panel_id in CONFIG_SPECS}
        if (
            len(worker) != RUNS_PER_WORKER
            or counts != {panel_id: PANEL_RUNS_PER_WORKER for panel_id in CONFIG_SPECS}
            or [int(row["worker_order"]) for row in worker] != list(range(RUNS_PER_WORKER))
            or any(worker[index]["panel_id"] == worker[index + 1]["panel_id"] for index in range(39))
        ):
            raise FreezeError("G00-F worker schedule is not exact, mixed, alternating, and balanced")
        for panel_id in CONFIG_SPECS:
            panel_worker = [row for row in worker if row["panel_id"] == panel_id]
            seed_counts: dict[int, int] = {}
            case_counts: dict[int, int] = {}
            for row in panel_worker:
                seed = int(row["seed"])
                case = int(row["cell_index"])
                seed_counts[seed] = seed_counts.get(seed, 0) + 1
                case_counts[case] = case_counts.get(case, 0) + 1
            if set(seed_counts.values()) != {2} or not set(case_counts.values()) <= {2, 3}:
                raise FreezeError("G00-F Latin worker assignment lost seed/case balance")

    return VerifiedFreeze(
        repo=resolved,
        path=target,
        file_sha256=observed,
        payload=payload,
        plans=verified_plans,
    )


def create_model_snapshot_receipt(
    *,
    verified: VerifiedFreeze,
    panel_id: str,
    snapshot_root: str | Path,
    output: str | Path,
) -> dict[str, Any]:
    """Hash every actual snapshot leaf before a G00-F weight update."""

    configurations = verified.payload["configurations"]
    if panel_id not in CONFIG_SPECS or not isinstance(configurations, Mapping):
        raise FreezeError("unknown G00-F model panel")
    binding = configurations[panel_id]
    if not isinstance(binding, Mapping) or not isinstance(binding.get("model_snapshot"), Mapping):
        raise FreezeError("freeze lacks model snapshot binding")
    model = binding["model_snapshot"]
    root_direct = Path(snapshot_root)
    root = root_direct.resolve()
    if root_direct.is_symlink() or not root.is_dir() or stat.S_IMODE(root.stat().st_mode) != 0o555:
        raise FreezeError(f"model snapshot directory is absent: {root}")
    expected_rows = model.get("leaf_files")
    if not isinstance(expected_rows, Sequence) or isinstance(expected_rows, (str, bytes)):
        raise FreezeError("frozen model leaf manifest is malformed")
    tree = list(root.rglob("*"))
    leaves = [path for path in tree if not path.is_dir()]
    directories = [path for path in tree if path.is_dir()]
    if (
        any(path.is_symlink() for path in tree)
        or any(not path.is_file() for path in leaves)
        or any(stat.S_IMODE(path.stat().st_mode) != 0o555 for path in directories)
    ):
        raise FreezeError("materialized model root contains a link or special file")
    actual_relative = sorted(path.relative_to(root).as_posix() for path in leaves)
    expected_relative = sorted(str(row["path"]) for row in expected_rows if isinstance(row, Mapping))
    if actual_relative != expected_relative:
        raise FreezeError("local model snapshot leaf set differs from the frozen manifest")
    receipt_rows: list[dict[str, Any]] = []
    for raw in expected_rows:
        if not isinstance(raw, Mapping):
            raise FreezeError("frozen model leaf row is malformed")
        relative = str(raw["path"])
        target = root / relative
        resolved_target = target.resolve(strict=True)
        if (
            target.is_symlink()
            or not resolved_target.is_file()
            or root not in resolved_target.parents
            or target.stat().st_nlink != 1
            or stat.S_IMODE(target.stat().st_mode) != 0o444
        ):
            raise FreezeError(f"model snapshot leaf is not an immutable direct file: {relative}")
        observed_bytes = resolved_target.stat().st_size
        observed_sha = sha256_file(resolved_target)
        if observed_bytes != int(raw["bytes"]) or observed_sha != raw["sha256"]:
            raise FreezeError(f"model snapshot leaf bytes changed: {relative}")
        receipt_rows.append(
            {
                "path": relative,
                "bytes": observed_bytes,
                "sha256": observed_sha,
                "lstat_mode": stat.S_IMODE(target.lstat().st_mode),
                "symlink_target": None,
            }
        )
    body = {
        "schema": MODEL_RECEIPT_SCHEMA,
        "schema_version": MODEL_RECEIPT_SCHEMA_VERSION,
        "panel_id": panel_id,
        "repo_id": model["repo_id"],
        "revision": model["revision"],
        "snapshot_root": str(root),
        "freeze_file_sha256": verified.file_sha256,
        "freeze_digest": verified.digest,
        "leaf_manifest_digest": model["leaf_manifest_digest"],
        "leaf_files": receipt_rows,
    }
    payload = {**body, "receipt_digest": semantic_digest(body)}
    exclusive_json(output, payload)
    return {**payload, "file_sha256": sha256_file(output)}


def materialize_model_snapshot(
    *,
    verified: VerifiedFreeze,
    panel_id: str,
    output_root: str | Path,
) -> dict[str, Any]:
    """Download one pinned public revision and dereference its exact ten leaves."""

    if panel_id not in CONFIG_SPECS:
        raise FreezeError("unknown G00-F model panel")
    target = Path(output_root).resolve()
    if target.exists():
        raise FreezeError("model materialization output must be a fresh absent path")
    target.parent.mkdir(parents=True, exist_ok=True)
    specification = CONFIG_SPECS[panel_id]
    rows = MODEL_LEAF_FILES[panel_id]
    try:
        from huggingface_hub import snapshot_download  # type: ignore[import-not-found]
    except ImportError as error:  # pragma: no cover - frozen GPU runtime dependency
        raise FreezeError("frozen huggingface-hub dependency is absent") from error
    download_cache = Path(tempfile.mkdtemp(prefix=".g00f-hf-download-", dir=target.parent))
    materialized = Path(tempfile.mkdtemp(prefix=f".{target.name}.materializing-", dir=target.parent))
    try:
        downloaded = Path(
            snapshot_download(
                repo_id=str(specification["model"]),
                revision=str(specification["revision"]),
                allow_patterns=[str(row["path"]) for row in rows],
                cache_dir=download_cache,
                token=False,
                max_workers=4,
            )
        ).resolve()
        for raw in rows:
            relative = str(raw["path"])
            source = (downloaded / relative).resolve(strict=True)
            if (
                not source.is_file()
                or source.stat().st_size != int(raw["bytes"])
                or sha256_file(source) != raw["sha256"]
            ):
                raise FreezeError(f"downloaded frozen model leaf changed: {relative}")
            destination = materialized / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            with source.open("rb") as input_handle:
                descriptor = os.open(destination, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
                with os.fdopen(descriptor, "wb") as output_handle:
                    shutil.copyfileobj(input_handle, output_handle, length=8 * 1024 * 1024)
                    output_handle.flush()
                    os.fsync(output_handle.fileno())
            if sha256_file(destination) != raw["sha256"]:
                raise FreezeError(f"materialized frozen model leaf changed: {relative}")
            os.chmod(destination, 0o444)
        for directory in sorted(
            (path for path in materialized.rglob("*") if path.is_dir()),
            key=lambda path: len(path.parts),
            reverse=True,
        ):
            os.chmod(directory, 0o555)
        os.replace(materialized, target)
        os.chmod(target, 0o555)
        directory_descriptor = os.open(target.parent, os.O_RDONLY)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    finally:
        if materialized.exists():
            os.chmod(materialized, 0o700)
            for directory in (path for path in materialized.rglob("*") if path.is_dir()):
                os.chmod(directory, 0o700)
            shutil.rmtree(materialized, ignore_errors=True)
        shutil.rmtree(download_cache, ignore_errors=True)
    return {
        "panel_id": panel_id,
        "repo_id": specification["model"],
        "revision": specification["revision"],
        "snapshot_root": str(target),
        "leaf_count": len(rows),
        "leaf_manifest_digest": semantic_digest([dict(row) for row in rows]),
        "fresh_regular_files_no_links": True,
        "g01_launch_authorized": False,
    }


def verify_model_snapshot_receipt(
    *, verified: VerifiedFreeze, panel_id: str, receipt_path: str | Path
) -> dict[str, Any]:
    receipt = strict_json(receipt_path, f"{panel_id} model receipt")
    body = {key: value for key, value in receipt.items() if key != "receipt_digest"}
    if (
        receipt.get("schema") != MODEL_RECEIPT_SCHEMA
        or receipt.get("schema_version") != MODEL_RECEIPT_SCHEMA_VERSION
        or receipt.get("panel_id") != panel_id
        or receipt.get("freeze_file_sha256") != verified.file_sha256
        or receipt.get("freeze_digest") != verified.digest
        or receipt.get("receipt_digest") != semantic_digest(body)
    ):
        raise FreezeError(f"{panel_id} model receipt is not bound to this freeze")
    # Re-hash every leaf at worker start.  The temporary re-creation is kept
    # in memory; only the already supplied receipt remains authoritative.
    configurations = verified.payload["configurations"]
    binding = configurations[panel_id]
    model = binding["model_snapshot"]
    if (
        receipt.get("repo_id") != model["repo_id"]
        or receipt.get("revision") != model["revision"]
        or receipt.get("leaf_manifest_digest") != model["leaf_manifest_digest"]
    ):
        raise FreezeError(f"{panel_id} model receipt identity changed")
    root = Path(str(receipt.get("snapshot_root", "")))
    actual_rows = receipt.get("leaf_files")
    if not isinstance(actual_rows, Sequence) or isinstance(actual_rows, (str, bytes)):
        raise FreezeError(f"{panel_id} model receipt leaf set is malformed")
    frozen_by_path = {str(row["path"]): row for row in model["leaf_files"]}
    if {str(row.get("path")) for row in actual_rows if isinstance(row, Mapping)} != set(frozen_by_path):
        raise FreezeError(f"{panel_id} model receipt leaf set changed")
    root_resolved = root.resolve(strict=True)
    tree_entries = list(root_resolved.rglob("*"))
    if (
        root.is_symlink()
        or not root_resolved.is_dir()
        or stat.S_IMODE(root_resolved.stat().st_mode) != 0o555
        or any(entry.is_symlink() or not entry.is_file() for entry in tree_entries)
        or {entry.relative_to(root_resolved).as_posix() for entry in tree_entries} != set(frozen_by_path)
    ):
        raise FreezeError(f"{panel_id} model snapshot tree inventory changed")
    for row in actual_rows:
        if not isinstance(row, Mapping):
            raise FreezeError(f"{panel_id} model receipt leaf row is malformed")
        relative = str(row["path"])
        direct = root / relative
        target = direct.resolve(strict=True)
        frozen = frozen_by_path[relative]
        if (
            direct.is_symlink()
            or not target.is_file()
            or root_resolved not in target.parents
            or direct.stat().st_nlink != 1
            or stat.S_IMODE(direct.stat().st_mode) != 0o444
            or target.stat().st_size != int(frozen["bytes"])
            or sha256_file(target) != frozen["sha256"]
            or row.get("bytes") != frozen["bytes"]
            or row.get("sha256") != frozen["sha256"]
            or row.get("lstat_mode") != 0o444
            or row.get("symlink_target") is not None
        ):
            raise FreezeError(f"{panel_id} model receipt no longer replays: {relative}")
    return {
        "path": str(Path(receipt_path).resolve()),
        "file_sha256": sha256_file(receipt_path),
        "receipt_digest": receipt["receipt_digest"],
        "snapshot_root": str(root.resolve()),
        "panel_id": panel_id,
    }


def _finite_integration_number(value: Any, label: str) -> float:
    if isinstance(value, bool):
        raise FreezeError(f"model integration {label} must be finite numeric data")
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise FreezeError(f"model integration {label} must be finite numeric data") from error
    if not math.isfinite(result):
        raise FreezeError(f"model integration {label} must be finite numeric data")
    return result


def _validate_model_integration_report(
    *,
    panel_id: str,
    report: Mapping[str, Any],
    snapshot_root: str | Path,
) -> None:
    """Replay the exact success surface of the frozen legacy boundary audit."""

    if set(report) != {"action_boundary", "model", "passed", "prompt_tokens", "runtime", "scores"}:
        raise FreezeError("model integration report field inventory changed")
    model = report.get("model")
    runtime = report.get("runtime")
    boundary = report.get("action_boundary")
    prompt_tokens = report.get("prompt_tokens")
    scores = report.get("scores")
    if not all(isinstance(value, Mapping) for value in (model, runtime, boundary, prompt_tokens, scores)):
        raise FreezeError("model integration report sections are malformed")
    assert isinstance(model, Mapping)
    assert isinstance(runtime, Mapping)
    assert isinstance(boundary, Mapping)
    assert isinstance(prompt_tokens, Mapping)
    assert isinstance(scores, Mapping)
    specification = CONFIG_SPECS[panel_id]
    model_identity = MODEL_RUNTIME_IDENTITIES[panel_id]
    tokenizer_identity = TOKENIZER_RUNTIME_IDENTITY
    if (
        report.get("passed") is not True
        or set(model)
        != {
            "action_labels",
            "action_token_ids",
            "chat_template_sha256",
            "dependency_versions",
            "model_class",
            "parameter_count",
            "peft_version",
            "requested_dtype",
            "requested_model",
            "requested_revision",
            "resolved_revision",
            "tokenizer_class",
            "tokenizer_name_or_path",
            "tokenizer_resolved_revision",
            "torch_version",
            "trainable_parameter_count",
            "transformers_version",
            "vocabulary_size",
        }
        or model.get("requested_model") != specification["model"]
        or model.get("requested_revision") != specification["revision"]
        or model.get("resolved_revision") != specification["revision"]
        or model.get("tokenizer_resolved_revision") != specification["revision"]
        or model.get("requested_dtype") != "bfloat16"
        or model.get("model_class") != model_identity["model_class"]
        or model.get("parameter_count") != model_identity["parameter_count"]
        or model.get("trainable_parameter_count") != 0
        or model.get("tokenizer_class") != tokenizer_identity["tokenizer_class"]
        or Path(str(model.get("tokenizer_name_or_path", ""))).resolve() != Path(snapshot_root).resolve()
        or model.get("vocabulary_size") != tokenizer_identity["vocabulary_size"]
        or model.get("chat_template_sha256") != tokenizer_identity["chat_template_sha256"]
        or model.get("action_labels") != tokenizer_identity["action_labels"]
        or model.get("action_token_ids") != tokenizer_identity["action_token_ids"]
        or model.get("dependency_versions") != FROZEN_MODEL_DEPENDENCIES
        or model.get("torch_version") != FROZEN_MODEL_DEPENDENCIES["torch"]
        or model.get("transformers_version") != FROZEN_MODEL_DEPENDENCIES["transformers"]
        or model.get("peft_version") != FROZEN_MODEL_DEPENDENCIES["peft"]
    ):
        raise FreezeError(f"{panel_id} model integration provenance changed")
    if (
        set(runtime)
        != {
            "dependencies",
            "last_logit_parameter",
            "parameter_devices",
            "parameter_dtypes",
            "requested_device",
        }
        or runtime.get("requested_device") != "cuda"
        or runtime.get("parameter_devices") != ["cuda:0"]
        or runtime.get("parameter_dtypes") != ["torch.bfloat16"]
        or runtime.get("dependencies") != FROZEN_MODEL_DEPENDENCIES
    ):
        raise FreezeError(f"{panel_id} model integration runtime changed")
    continuations = boundary.get("continuation_token_ids_by_prompt")
    continuation_lengths = boundary.get("continuation_token_lengths_by_prompt")
    if (
        set(boundary)
        != {
            "continuation_token_ids_by_prompt",
            "continuation_token_lengths_by_prompt",
            "labels",
            "prefix_stable",
            "standalone_token_ids",
        }
        or boundary.get("labels") != list(MODEL_INTEGRATION_ACTION_LABELS)
        or boundary.get("standalone_token_ids") != tokenizer_identity["action_token_ids"]
        or boundary.get("prefix_stable") is not True
        or not isinstance(continuations, Sequence)
        or isinstance(continuations, (str, bytes))
        or len(continuations) != len(MODEL_INTEGRATION_PROMPTS)
        or not isinstance(continuation_lengths, Sequence)
        or isinstance(continuation_lengths, (str, bytes))
        or len(continuation_lengths) != len(MODEL_INTEGRATION_PROMPTS)
    ):
        raise FreezeError(f"{panel_id} continuation-token boundary changed")
    for token_pair, length_pair in zip(continuations, continuation_lengths, strict=True):
        if (
            not isinstance(token_pair, Sequence)
            or isinstance(token_pair, (str, bytes))
            or len(token_pair) != 2
            or not isinstance(length_pair, Sequence)
            or isinstance(length_pair, (str, bytes))
            or len(length_pair) != 2
            or any(
                not isinstance(tokens, Sequence) or isinstance(tokens, (str, bytes)) or len(tokens) < 1
                for tokens in token_pair
            )
            or [len(tokens) for tokens in token_pair] != list(length_pair)
        ):
            raise FreezeError(f"{panel_id} contextual A/B continuations are malformed")
    counts = prompt_tokens.get("counts")
    if (
        set(prompt_tokens) != {"configured_maximum", "counts", "maximum"}
        or prompt_tokens.get("configured_maximum") is not None
        or not isinstance(counts, Sequence)
        or isinstance(counts, (str, bytes))
        or len(counts) != len(MODEL_INTEGRATION_PROMPTS)
        or any(isinstance(value, bool) or not isinstance(value, int) or value < 1 for value in counts)
        or prompt_tokens.get("maximum") != max(counts)
    ):
        raise FreezeError(f"{panel_id} integration prompt-token counts changed")
    normalized = scores.get("normalized_log_scores")
    probabilities = scores.get("probabilities")
    if (
        set(scores)
        != {
            "finite",
            "maximum_normalization_error",
            "maximum_probability_swap_error",
            "maximum_score_swap_error",
            "normalized_log_scores",
            "probabilities",
            "swap_atol",
            "swap_invariant",
            "swap_rtol",
        }
        or scores.get("finite") is not True
        or scores.get("swap_invariant") is not True
        or scores.get("swap_atol") != MODEL_INTEGRATION_SWAP_ATOL
        or scores.get("swap_rtol") != MODEL_INTEGRATION_SWAP_RTOL
        or not isinstance(normalized, Sequence)
        or isinstance(normalized, (str, bytes))
        or not isinstance(probabilities, Sequence)
        or isinstance(probabilities, (str, bytes))
        or len(normalized) != len(MODEL_INTEGRATION_PROMPTS)
        or len(probabilities) != len(MODEL_INTEGRATION_PROMPTS)
    ):
        raise FreezeError(f"{panel_id} integration score contract changed")
    for label, rows in (("normalized scores", normalized), ("probabilities", probabilities)):
        for row in rows:
            if (
                not isinstance(row, Sequence)
                or isinstance(row, (str, bytes))
                or len(row) != 2
                or any(not math.isfinite(_finite_integration_number(value, label)) for value in row)
            ):
                raise FreezeError(f"{panel_id} integration {label} are malformed")
    probability_rows = [
        [_finite_integration_number(value, "probability") for value in row] for row in probabilities
    ]
    normalized_rows = [
        [_finite_integration_number(value, "normalized log score") for value in row] for row in normalized
    ]
    if any(value < 0 or value > 1 for row in probability_rows for value in row) or any(
        not math.isclose(
            sum(row),
            1.0,
            rel_tol=MODEL_INTEGRATION_PROBABILITY_REPLAY_TOLERANCE,
            abs_tol=MODEL_INTEGRATION_PROBABILITY_REPLAY_TOLERANCE,
        )
        for row in probability_rows
    ):
        raise FreezeError(f"{panel_id} integration probabilities are not normalized")
    if any(
        not math.isclose(
            math.exp(log_score),
            probability,
            rel_tol=MODEL_INTEGRATION_PROBABILITY_REPLAY_TOLERANCE,
            abs_tol=MODEL_INTEGRATION_PROBABILITY_REPLAY_TOLERANCE,
        )
        for log_row, probability_row in zip(normalized_rows, probability_rows, strict=True)
        for log_score, probability in zip(log_row, probability_row, strict=True)
    ):
        raise FreezeError(f"{panel_id} normalized scores disagree with probabilities")
    for key in (
        "maximum_normalization_error",
        "maximum_probability_swap_error",
        "maximum_score_swap_error",
    ):
        if _finite_integration_number(scores.get(key), key) < 0:
            raise FreezeError(f"{panel_id} integration {key} is negative")
    if (
        _finite_integration_number(scores.get("maximum_normalization_error"), "normalization")
        > MODEL_INTEGRATION_PROBABILITY_REPLAY_TOLERANCE
    ):
        raise FreezeError(f"{panel_id} integration normalization error is too large")
    if (
        _finite_integration_number(scores.get("maximum_score_swap_error"), "score swap error")
        > MODEL_INTEGRATION_SWAP_ATOL
        or _finite_integration_number(
            scores.get("maximum_probability_swap_error"),
            "probability swap error",
        )
        > MODEL_INTEGRATION_SWAP_ATOL
    ):
        raise FreezeError(f"{panel_id} integration A/B swap error exceeds 2e-3")


def create_model_integration_audit(
    *,
    verified: VerifiedFreeze,
    panel_id: str,
    model_receipt_path: str | Path,
    output: str | Path,
    integration_runner: Callable[..., Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Run and bind the legacy real-model boundary check before ITT creation."""

    if panel_id not in CONFIG_SPECS:
        raise FreezeError("unknown G00-F model integration panel")
    target = Path(output).resolve()
    execution_root = target.parent
    if target != execution_root / f"model-integration-audit-{panel_id.removeprefix('g00f-')}.json":
        raise FreezeError("model integration audit has a noncanonical execution-root path")
    if (execution_root / "itt-ledger").exists() or any(execution_root.glob("worker-*-launch.json")):
        raise FreezeError("model integration audit must precede the ITT ledger and worker launch")
    model_receipt = verify_model_snapshot_receipt(
        verified=verified,
        panel_id=panel_id,
        receipt_path=model_receipt_path,
    )
    expected_model_receipt = execution_root / f"model-receipt-{panel_id.removeprefix('g00f-')}.json"
    if Path(str(model_receipt["path"])) != expected_model_receipt:
        raise FreezeError("model integration audit receipt lies outside its execution root")
    if os.environ.get("HF_HUB_OFFLINE") != "1" or os.environ.get("TRANSFORMERS_OFFLINE") != "1":
        raise FreezeError("model integration audit requires both offline-mode guards")
    if integration_runner is None:
        from goalzendo.modeling import run_model_integration_check

        integration_runner = run_model_integration_check
    model_receipts = {
        candidate: model_receipt if candidate == panel_id else {"snapshot_root": "unused"}
        for candidate in CONFIG_SPECS
    }
    specification = CONFIG_SPECS[panel_id]
    try:
        report = dict(
            integration_runner(
                {
                    "name": specification["model"],
                    "revision": specification["revision"],
                    "dtype": "bfloat16",
                    "trust_remote_code": False,
                },
                prompts=MODEL_INTEGRATION_PROMPTS,
                action_labels=MODEL_INTEGRATION_ACTION_LABELS,
                device="cuda",
                max_prompt_tokens=None,
                system_prompt=MODEL_INTEGRATION_SYSTEM_PROMPT,
                model_loader=_authenticated_local_model_loader(model_receipts),
                strict_revision=True,
                swap_atol=MODEL_INTEGRATION_SWAP_ATOL,
                swap_rtol=MODEL_INTEGRATION_SWAP_RTOL,
            )
        )
    except BaseException as error:
        raise FreezeError(f"{panel_id} pretraining model integration audit failed") from error
    _validate_model_integration_report(
        panel_id=panel_id,
        report=report,
        snapshot_root=model_receipt["snapshot_root"],
    )
    contract = copy.deepcopy(dict(EXECUTION_AND_GATE_CONTRACT["pretraining_model_boundary"]))
    body = {
        "schema": MODEL_INTEGRATION_AUDIT_SCHEMA,
        "schema_version": MODEL_INTEGRATION_AUDIT_SCHEMA_VERSION,
        "panel_id": panel_id,
        "freeze_file_sha256": verified.file_sha256,
        "freeze_digest": verified.digest,
        "model_snapshot_receipt": model_receipt,
        "contract": contract,
        "report": report,
        "report_digest": semantic_digest(report),
        "ledger_absent_at_creation": True,
        "weight_updates_performed": False,
        "outcomes_seen": False,
        "g01_launch_authorized": False,
    }
    payload = {**body, "audit_digest": semantic_digest(body)}
    exclusive_json(target, payload)
    return {
        "path": str(target),
        "file_sha256": sha256_file(target),
        "audit_digest": payload["audit_digest"],
        "report_digest": payload["report_digest"],
        "panel_id": panel_id,
        "model_snapshot_receipt": model_receipt,
    }


def verify_model_integration_audit(
    *,
    verified: VerifiedFreeze,
    panel_id: str,
    audit_path: str | Path,
    replay_model_snapshot: bool = True,
) -> dict[str, Any]:
    target = Path(audit_path).resolve()
    execution_root = target.parent
    if target != execution_root / f"model-integration-audit-{panel_id.removeprefix('g00f-')}.json":
        raise FreezeError("model integration audit has a noncanonical execution-root path")
    payload = strict_json(target, f"{panel_id} model integration audit")
    body = {key: value for key, value in payload.items() if key != "audit_digest"}
    exact_body_keys = {
        "contract",
        "freeze_digest",
        "freeze_file_sha256",
        "g01_launch_authorized",
        "ledger_absent_at_creation",
        "model_snapshot_receipt",
        "outcomes_seen",
        "panel_id",
        "report",
        "report_digest",
        "schema",
        "schema_version",
        "weight_updates_performed",
    }
    raw_model_receipt = payload.get("model_snapshot_receipt")
    report = payload.get("report")
    if (
        set(body) != exact_body_keys
        or payload.get("schema") != MODEL_INTEGRATION_AUDIT_SCHEMA
        or payload.get("schema_version") != MODEL_INTEGRATION_AUDIT_SCHEMA_VERSION
        or payload.get("panel_id") != panel_id
        or payload.get("freeze_file_sha256") != verified.file_sha256
        or payload.get("freeze_digest") != verified.digest
        or payload.get("contract") != EXECUTION_AND_GATE_CONTRACT["pretraining_model_boundary"]
        or not isinstance(raw_model_receipt, Mapping)
        or not isinstance(report, Mapping)
        or payload.get("report_digest") != semantic_digest(report)
        or payload.get("ledger_absent_at_creation") is not True
        or payload.get("weight_updates_performed") is not False
        or payload.get("outcomes_seen") is not False
        or payload.get("g01_launch_authorized") is not False
        or payload.get("audit_digest") != semantic_digest(body)
    ):
        raise FreezeError(f"{panel_id} model integration audit changed")
    receipt_path = execution_root / f"model-receipt-{panel_id.removeprefix('g00f-')}.json"
    if replay_model_snapshot:
        model_receipt = verify_model_snapshot_receipt(
            verified=verified,
            panel_id=panel_id,
            receipt_path=receipt_path,
        )
    else:
        receipt = strict_json(receipt_path, f"{panel_id} model receipt bound by integration audit")
        model_receipt = {
            "path": str(receipt_path),
            "file_sha256": sha256_file(receipt_path),
            "receipt_digest": receipt.get("receipt_digest"),
            "snapshot_root": str(Path(str(receipt.get("snapshot_root", ""))).resolve()),
            "panel_id": panel_id,
        }
    if raw_model_receipt != model_receipt:
        raise FreezeError(f"{panel_id} integration/model-snapshot receipt binding changed")
    _validate_model_integration_report(
        panel_id=panel_id,
        report=report,
        snapshot_root=model_receipt["snapshot_root"],
    )
    return {
        "path": str(target),
        "file_sha256": sha256_file(target),
        "audit_digest": payload["audit_digest"],
        "report_digest": payload["report_digest"],
        "panel_id": panel_id,
        "model_snapshot_receipt": model_receipt,
    }


def verify_runpod_provision_receipt(
    *,
    verified: VerifiedFreeze,
    receipt_path: str | Path,
    expected_receipt_sha256: str,
    expected_pod_id: str,
    require_bound_copy: bool = True,
) -> dict[str, Any]:
    """Verify externally pinned Runpod allocation facts without claiming attestation.

    The caller-supplied SHA-256 is the trust boundary: the local process cannot
    independently recover the pod image, volume attachment, or data center from
    CUDA.  The receipt therefore names the Runpod API fields and explicitly
    labels the volume mount as an external-verifier boundary.
    """

    direct = Path(receipt_path)
    target = direct.resolve()
    expected_sha = require_sha256(
        expected_receipt_sha256,
        "externally expected Runpod provision receipt SHA-256",
    )
    if not expected_pod_id or any(character.isspace() for character in expected_pod_id):
        raise FreezeError("RUNPOD_POD_ID must be a nonempty token")
    if not target.is_file() or sha256_file(target) != expected_sha:
        raise FreezeError("Runpod provision receipt lacks its external SHA-256 binding")
    if direct.is_symlink() or (require_bound_copy and target.stat().st_nlink != 1):
        raise FreezeError("bound Runpod provision receipt must be one direct regular file")
    if require_bound_copy and stat.S_IMODE(target.stat().st_mode) != 0o400:
        raise FreezeError("bound Runpod provision receipt must be immutable mode 0400")
    receipt = strict_json(target, "externally pinned Runpod provision receipt")
    body = {key: value for key, value in receipt.items() if key != "receipt_digest"}
    exact_keys = {
        "api_response",
        "capture_tool",
        "created_at_utc",
        "data_center",
        "evidence_boundaries",
        "g01_launch_authorized",
        "gpu_count",
        "gpu_catalog",
        "image",
        "network_volume_id",
        "network_volume_mount",
        "operational_market_snapshot",
        "outcomes_seen",
        "pod_id",
        "provisioning",
        "provider",
        "schema",
        "schema_version",
    }
    runtime = verified.payload["runtime"]
    if (
        set(body) != exact_keys
        or receipt.get("schema") != RUNPOD_PROVISION_RECEIPT_SCHEMA
        or receipt.get("schema_version") != RUNPOD_PROVISION_RECEIPT_SCHEMA_VERSION
        or receipt.get("provider") != "runpod"
        or receipt.get("capture_tool") != "runpodctl 2.9.0-c094cac"
        or receipt.get("pod_id") != expected_pod_id
        or receipt.get("image") != runtime["image"]
        or receipt.get("gpu_count") != WORKER_COUNT
        or receipt.get("gpu_catalog")
        != {
            "display_name": "H100 SXM",
            "gpu_id": "NVIDIA H100 80GB HBM3",
        }
        or receipt.get("data_center") != runtime["data_center"]
        or receipt.get("network_volume_id") != runtime["network_volume_id"]
        or receipt.get("network_volume_mount") != runtime["network_volume_mount"]
        or receipt.get("evidence_boundaries") != RUNPOD_PROVISION_EVIDENCE
        or receipt.get("outcomes_seen") is not False
        or receipt.get("g01_launch_authorized") is not False
        or receipt.get("receipt_digest") != semantic_digest(body)
    ):
        raise FreezeError("Runpod provision receipt differs from the frozen allocation contract")
    api_response = receipt.get("api_response")
    if not isinstance(api_response, Mapping) or set(api_response) != {
        "capture_command",
        "file_name",
        "sha256",
    }:
        raise FreezeError("Runpod provision receipt raw API response binding is malformed")
    if (
        api_response.get("capture_command")
        != (f"runpodctl pod get {expected_pod_id} --include-machine --include-network-volume -o json")
        or api_response.get("file_name") != "runpod-api-response.json"
    ):
        raise FreezeError("Runpod raw API capture command or canonical file name changed")
    raw_sha = require_sha256(api_response.get("sha256"), "Runpod API response SHA-256")
    raw_direct = target.parent / "runpod-api-response.json"
    if (
        raw_direct.is_symlink()
        or not raw_direct.is_file()
        or sha256_file(raw_direct) != raw_sha
        or (require_bound_copy and raw_direct.stat().st_nlink != 1)
        or (require_bound_copy and stat.S_IMODE(raw_direct.stat().st_mode) != 0o400)
    ):
        raise FreezeError("bound raw Runpod API response bytes do not replay")
    provisioning = receipt.get("provisioning")
    if not isinstance(provisioning, Mapping) or set(provisioning) != {
        "capture_order",
        "create",
        "get",
        "maximum_secure_cost_usd",
        "provision_ceiling_seconds",
        "terminate_after_utc",
    }:
        raise FreezeError("Runpod provisioning chronology or termination guard is malformed")
    create_binding = provisioning.get("create")
    get_binding = provisioning.get("get")
    if (
        not isinstance(create_binding, Mapping)
        or set(create_binding) != {"command_argv", "file_name", "sha256"}
        or not isinstance(get_binding, Mapping)
        or set(get_binding) != {"command_argv", "file_name", "sha256"}
    ):
        raise FreezeError("Runpod create/get raw-response binding is malformed")
    terminate_after = str(provisioning.get("terminate_after_utc", ""))
    created = str(receipt.get("created_at_utc", ""))
    created_time = _runpod_created_utc(created, "Runpod provision receipt creation time")
    terminate_time = _canonical_utc(terminate_after, "Runpod absolute termination time")
    provision_seconds = (terminate_time - created_time).total_seconds()
    expected_get_argv = [
        "runpodctl",
        "pod",
        "get",
        expected_pod_id,
        "--include-machine",
        "--include-network-volume",
        "-o",
        "json",
    ]
    create_raw_sha = require_sha256(create_binding.get("sha256"), "Runpod create response SHA-256")
    create_raw = target.parent / "runpod-create-response.json"
    if (
        provisioning.get("capture_order") != ["create", "get"]
        or create_binding.get("command_argv")
        != list(canonical_runpod_create_command(terminate_after_utc=terminate_after))
        or create_binding.get("file_name") != "runpod-create-response.json"
        or get_binding.get("command_argv") != expected_get_argv
        or get_binding.get("file_name") != "runpod-api-response.json"
        or get_binding.get("sha256") != raw_sha
        or provisioning.get("provision_ceiling_seconds")
        != RUNPOD_PROVISIONING_CONTRACT["provision_ceiling_seconds"]
        or provision_seconds <= 0
        or provision_seconds > RUNPOD_PROVISIONING_CONTRACT["provision_ceiling_seconds"]
        or provisioning.get("maximum_secure_cost_usd")
        != RUNPOD_PROVISIONING_CONTRACT["maximum_secure_cost_usd"]
        or create_raw.is_symlink()
        or not create_raw.is_file()
        or sha256_file(create_raw) != create_raw_sha
        or (require_bound_copy and create_raw.stat().st_nlink != 1)
        or (require_bound_copy and stat.S_IMODE(create_raw.stat().st_mode) != 0o400)
    ):
        raise FreezeError("Runpod termination guard exceeds or differs from the exact 16-hour ceiling")
    market = receipt.get("operational_market_snapshot")
    if not isinstance(market, Mapping) or set(market) != {
        "observed_at_utc",
        "scientific_identity",
        "secure_price_usd_per_gpu_hour",
        "stock_label",
    }:
        raise FreezeError("Runpod operational price/stock snapshot is malformed")
    price = market.get("secure_price_usd_per_gpu_hour")
    if (
        isinstance(price, bool)
        or not isinstance(price, (int, float))
        or not math.isfinite(float(price))
        or float(price) < 0
        or float(price) > float(RUNPOD_PROVISIONING_CONTRACT["secure_price_ceiling_usd_per_gpu_hour"])
        or not isinstance(market.get("stock_label"), str)
        or not market["stock_label"]
        or market.get("observed_at_utc") != receipt.get("created_at_utc")
        or market.get("scientific_identity") is not False
    ):
        raise FreezeError("Runpod operational price/stock facts changed type or role")
    computed_cost = float(price) * WORKER_COUNT * provision_seconds / 3_600
    if computed_cost > float(RUNPOD_PROVISIONING_CONTRACT["maximum_secure_cost_usd"]) + 1e-12:
        raise FreezeError("Runpod provision duration and price exceed the frozen cost ceiling")
    return {
        "path": str(target),
        "file_sha256": expected_sha,
        "receipt_digest": receipt["receipt_digest"],
        "pod_id": expected_pod_id,
        "created_at_utc": created,
        "provisioning": copy.deepcopy(dict(provisioning)),
        "api_response": {
            "path": str(raw_direct.resolve()),
            "file_sha256": raw_sha,
            "capture_command": api_response["capture_command"],
        },
        "evidence_boundaries": dict(RUNPOD_PROVISION_EVIDENCE),
        "operational_market_snapshot": copy.deepcopy(dict(market)),
    }


def create_runpod_provision_receipt(
    *,
    verified: VerifiedFreeze,
    raw_create_response_path: str | Path,
    raw_api_response_path: str | Path,
    pod_id: str,
    created_at_utc: str,
    terminate_after_utc: str,
    secure_price_usd_per_gpu_hour: float,
    stock_label: str,
    output_path: str | Path,
) -> dict[str, Any]:
    """Create the exact reviewer-facing receipt; an external party pins its SHA."""

    create_raw_direct = Path(raw_create_response_path)
    create_raw = create_raw_direct.resolve()
    raw_direct = Path(raw_api_response_path)
    raw = raw_direct.resolve()
    output = Path(output_path).resolve()
    if (
        create_raw != output.parent / "runpod-create-response.json"
        or create_raw_direct.is_symlink()
        or not create_raw.is_file()
        or raw != output.parent / "runpod-api-response.json"
        or raw_direct.is_symlink()
        or not raw.is_file()
    ):
        raise FreezeError("raw Runpod create/get responses must be direct canonical sibling files")
    runtime = verified.payload["runtime"]
    _runpod_created_utc(created_at_utc, "Runpod provision receipt creation time")
    _canonical_utc(terminate_after_utc, "Runpod absolute termination time")
    get_argv = [
        "runpodctl",
        "pod",
        "get",
        pod_id,
        "--include-machine",
        "--include-network-volume",
        "-o",
        "json",
    ]
    body = {
        "schema": RUNPOD_PROVISION_RECEIPT_SCHEMA,
        "schema_version": RUNPOD_PROVISION_RECEIPT_SCHEMA_VERSION,
        "provider": "runpod",
        "capture_tool": "runpodctl 2.9.0-c094cac",
        "pod_id": pod_id,
        "image": runtime["image"],
        "gpu_count": WORKER_COUNT,
        "gpu_catalog": {
            "display_name": "H100 SXM",
            "gpu_id": "NVIDIA H100 80GB HBM3",
        },
        "data_center": runtime["data_center"],
        "network_volume_id": runtime["network_volume_id"],
        "network_volume_mount": runtime["network_volume_mount"],
        "created_at_utc": created_at_utc,
        "operational_market_snapshot": {
            "observed_at_utc": created_at_utc,
            "secure_price_usd_per_gpu_hour": secure_price_usd_per_gpu_hour,
            "stock_label": stock_label,
            "scientific_identity": False,
        },
        "api_response": {
            "capture_command": " ".join(get_argv),
            "file_name": "runpod-api-response.json",
            "sha256": sha256_file(raw),
        },
        "provisioning": {
            "capture_order": ["create", "get"],
            "create": {
                "command_argv": list(
                    canonical_runpod_create_command(terminate_after_utc=terminate_after_utc)
                ),
                "file_name": "runpod-create-response.json",
                "sha256": sha256_file(create_raw),
            },
            "get": {
                "command_argv": get_argv,
                "file_name": "runpod-api-response.json",
                "sha256": sha256_file(raw),
            },
            "terminate_after_utc": terminate_after_utc,
            "provision_ceiling_seconds": RUNPOD_PROVISIONING_CONTRACT["provision_ceiling_seconds"],
            "maximum_secure_cost_usd": RUNPOD_PROVISIONING_CONTRACT["maximum_secure_cost_usd"],
        },
        "evidence_boundaries": dict(RUNPOD_PROVISION_EVIDENCE),
        "outcomes_seen": False,
        "g01_launch_authorized": False,
    }
    payload = {**body, "receipt_digest": semantic_digest(body)}
    atomic_json(output, payload)
    return verify_runpod_provision_receipt(
        verified=verified,
        receipt_path=output,
        expected_receipt_sha256=sha256_file(output),
        expected_pod_id=pod_id,
        require_bound_copy=False,
    )


def bind_runpod_provision_receipt(
    *,
    verified: VerifiedFreeze,
    input_path: str | Path,
    output_path: str | Path,
    expected_receipt_sha256: str,
    expected_pod_id: str,
) -> dict[str, Any]:
    """Copy externally authenticated provision bytes once into the execution root."""

    source_direct = Path(input_path)
    target_direct = Path(output_path)
    source = source_direct.resolve()
    target = target_direct.resolve()
    if source == target:
        raise FreezeError("external and execution-bound provision receipt paths must differ")
    verify_runpod_provision_receipt(
        verified=verified,
        receipt_path=source_direct,
        expected_receipt_sha256=expected_receipt_sha256,
        expected_pod_id=expected_pod_id,
        require_bound_copy=False,
    )
    source_create = source.parent / "runpod-create-response.json"
    source_raw = source.parent / "runpod-api-response.json"
    target_create = target.parent / "runpod-create-response.json"
    target_raw = target.parent / "runpod-api-response.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    if target_create.exists():
        raise FreezeError("execution-bound raw Runpod create response already exists")
    try:
        create_descriptor = os.open(target_create, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o400)
    except FileExistsError as error:
        raise FreezeError("execution-bound raw Runpod create response already exists") from error
    with os.fdopen(create_descriptor, "wb") as create_handle:
        create_handle.write(source_create.read_bytes())
        create_handle.flush()
        os.fsync(create_handle.fileno())
    if target_raw.exists():
        raise FreezeError("execution-bound raw Runpod API response already exists")
    try:
        raw_descriptor = os.open(target_raw, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o400)
    except FileExistsError as error:
        raise FreezeError("execution-bound raw Runpod API response already exists") from error
    with os.fdopen(raw_descriptor, "wb") as raw_handle:
        raw_handle.write(source_raw.read_bytes())
        raw_handle.flush()
        os.fsync(raw_handle.fileno())
    try:
        descriptor = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o400)
    except FileExistsError as error:
        raise FreezeError("execution-bound Runpod provision receipt already exists") from error
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(source.read_bytes())
            handle.flush()
            os.fsync(handle.fileno())
        directory_descriptor = os.open(target.parent, os.O_RDONLY)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    except BaseException:
        raise
    return verify_runpod_provision_receipt(
        verified=verified,
        receipt_path=target_direct,
        expected_receipt_sha256=expected_receipt_sha256,
        expected_pod_id=expected_pod_id,
    )


def verify_launch_receipt(
    *,
    verified: VerifiedFreeze,
    worker_index: int,
    receipt_path: str | Path,
    expected_provision_receipt_sha256: str | None = None,
    expected_pod_id: str | None = None,
) -> dict[str, Any]:
    launch_path = Path(receipt_path).resolve()
    execution_root = launch_path.parent
    if launch_path != execution_root / f"worker-{worker_index}-launch.json":
        raise FreezeError("worker launch receipt has a noncanonical execution-root path")
    receipt = strict_json(launch_path, "G00-F worker launch receipt")
    body = {key: value for key, value in receipt.items() if key != "receipt_digest"}
    exact_keys = {
        "concurrent_runs",
        "cuda_visible_devices",
        "data_center",
        "execution_uuid",
        "freeze_digest",
        "freeze_file_sha256",
        "g01_launch_authorized",
        "gpu_family",
        "gpu_name",
        "gpu_uuid",
        "image",
        "ledger",
        "model_integration_audits",
        "network_volume_id",
        "network_volume_mount",
        "offline_environment",
        "outcomes_seen",
        "packages",
        "python",
        "runpod_provision",
        "runtime_environment",
        "schema",
        "schema_version",
        "source_bundle",
        "started_monotonic_ns",
        "started_unix_ns",
        "torch_gpu_name",
        "visible_gpu_count",
        "worker_index",
    }
    required = {
        "schema": LAUNCH_RECEIPT_SCHEMA,
        "schema_version": LAUNCH_RECEIPT_SCHEMA_VERSION,
        "worker_index": worker_index,
        "freeze_file_sha256": verified.file_sha256,
        "freeze_digest": verified.digest,
        "image": verified.payload["runtime"]["image"],
        "network_volume_id": verified.payload["runtime"]["network_volume_id"],
        "network_volume_mount": verified.payload["runtime"]["network_volume_mount"],
        "data_center": verified.payload["runtime"]["data_center"],
        "visible_gpu_count": 1,
        "gpu_family": "NVIDIA H100",
        "concurrent_runs": 1,
    }
    if set(body) != exact_keys or any(receipt.get(key) != value for key, value in required.items()):
        raise FreezeError("worker launch receipt is not bound to the exact runtime/freeze")
    if receipt.get("receipt_digest") != semantic_digest(body):
        raise FreezeError("worker launch receipt semantic digest mismatch")
    packages = receipt.get("packages")
    frozen_packages = verified.payload["runtime"].get("python_packages")
    if packages != frozen_packages:
        raise FreezeError("worker package versions differ from the frozen runtime")
    runtime_environment = receipt.get("runtime_environment")
    if not isinstance(runtime_environment, Mapping) or set(runtime_environment) != {
        "accelerator",
        "accelerator_digest",
        "installed_distributions",
        "installed_distributions_digest",
        "lock_scope",
    }:
        raise FreezeError("worker launch omits the complete pre-outcome runtime inventory")
    accelerator = runtime_environment.get("accelerator")
    installed = runtime_environment.get("installed_distributions")
    if (
        not isinstance(accelerator, Mapping)
        or not isinstance(installed, Mapping)
        or runtime_environment.get("accelerator_digest") != semantic_digest(accelerator)
        or runtime_environment.get("installed_distributions_digest") != semantic_digest(installed)
        or runtime_environment.get("lock_scope")
        != "recorded_pre_outcome_runtime_evidence_not_complete_dependency_lock"
        or accelerator.get("torch_version") != "2.8.0+cu128"
        or accelerator.get("cuda_runtime") != "12.8"
        or accelerator.get("cuda_available") is not True
    ):
        raise FreezeError("worker launch full runtime inventory digest changed")
    if (
        receipt.get("python") != verified.payload["runtime"]["python"]
        or receipt.get("outcomes_seen") is not False
        or receipt.get("g01_launch_authorized") is not False
        or receipt.get("offline_environment") != {"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"}
        or not str(receipt.get("gpu_name", "")).startswith("NVIDIA H100")
        or "H100" not in str(receipt.get("torch_gpu_name", ""))
        or not str(receipt.get("gpu_uuid", "")).startswith("GPU-")
        or receipt.get("cuda_visible_devices") != receipt.get("gpu_uuid")
        or isinstance(receipt.get("started_unix_ns"), bool)
        or not isinstance(receipt.get("started_unix_ns"), int)
        or int(receipt["started_unix_ns"]) < 0
        or isinstance(receipt.get("started_monotonic_ns"), bool)
        or not isinstance(receipt.get("started_monotonic_ns"), int)
        or int(receipt["started_monotonic_ns"]) < 0
    ):
        raise FreezeError("worker launch runtime probe fields differ from the frozen contract")
    bundle = receipt.get("source_bundle")
    if not isinstance(bundle, Mapping):
        raise FreezeError("worker launch receipt omits its source bundle")
    bundle_receipt_path = Path(str(bundle.get("receipt_path", ""))).resolve()
    bundle_receipt = strict_json(bundle_receipt_path, "bound extracted source bundle receipt")
    bundle_body = {key: value for key, value in bundle_receipt.items() if key != "receipt_digest"}
    embedded_manifest = strict_json(
        verified.repo / "G00F-BUNDLE-MANIFEST.json",
        "embedded G00-F bundle manifest",
    )
    if (
        set(bundle)
        != {
            "archive_sha256",
            "authenticated_runtime_files",
            "extracted_root",
            "manifest_digest",
            "manifest_sha256",
            "receipt_digest",
            "receipt_file_sha256",
            "receipt_path",
        }
        or bundle.get("archive_sha256") != verified.payload["source_bundle"]["archive_sha256"]
        or bundle.get("manifest_sha256") != verified.payload["source_bundle"]["manifest_sha256"]
        or bundle.get("manifest_digest") != verified.payload["source_bundle"]["manifest_digest"]
        or bundle.get("extracted_root") != str(verified.repo)
        or bundle_receipt_path != execution_root / "source-bundle-receipt.json"
        or verified.repo != execution_root / "frozen-source"
        or bundle.get("receipt_file_sha256") != sha256_file(bundle_receipt_path)
        or bundle.get("receipt_digest") != bundle_receipt.get("receipt_digest")
        or bundle.get("authenticated_runtime_files") != bundle_receipt.get("authenticated_runtime_files")
        or bundle_receipt.get("receipt_digest") != semantic_digest(bundle_body)
        or bundle_receipt.get("freeze_file_sha256") != verified.file_sha256
        or bundle_receipt.get("freeze_digest") != verified.digest
        or set(bundle_body)
        != {
            "archive_sha256",
            "authenticated_runtime_files",
            "extracted_root",
            "freeze_digest",
            "freeze_file_sha256",
            "g01_launch_authorized",
            "manifest_digest",
            "manifest_sha256",
            "member_count",
            "schema",
            "schema_version",
            "tar_safety",
        }
        or bundle_receipt.get("schema") != "goalzendo.g00f_extracted_source_bundle_receipt"
        or bundle_receipt.get("schema_version") != 1
        or bundle_receipt.get("member_count") != len(embedded_manifest.get("members", []))
        or bundle_receipt.get("tar_safety")
        != {
            "exact_bytes": True,
            "exact_member_set": True,
            "exact_modes": True,
            "no_absolute_or_parent_paths": True,
            "no_links": True,
            "only_regular_files": True,
        }
        or bundle_receipt.get("g01_launch_authorized") is not False
    ):
        raise FreezeError("worker launch receipt is not bound to the frozen source archive")
    ledger_binding = receipt.get("ledger")
    if not isinstance(ledger_binding, Mapping):
        raise FreezeError("worker launch receipt omits its ITT ledger")
    ledger = verify_attempt_ledger(
        verified=verified,
        ledger_root=str(ledger_binding.get("ledger_root", "")),
    )
    provision_binding = receipt.get("runpod_provision")
    if not isinstance(provision_binding, Mapping):
        raise FreezeError("worker launch receipt omits the external Runpod allocation")
    provision = verify_runpod_provision_receipt(
        verified=verified,
        receipt_path=str(provision_binding.get("path", "")),
        expected_receipt_sha256=str(provision_binding.get("file_sha256", "")),
        expected_pod_id=str(provision_binding.get("pod_id", "")),
    )
    if Path(str(provision["path"])) != execution_root / "runpod-provision-receipt.json":
        raise FreezeError("worker launch provision receipt is outside its execution root")
    if (expected_provision_receipt_sha256 is None) != (expected_pod_id is None):
        raise FreezeError("external provision SHA-256 and pod ID must be supplied together")
    if expected_provision_receipt_sha256 is not None and (
        provision["file_sha256"]
        != require_sha256(
            expected_provision_receipt_sha256,
            "externally expected Runpod provision receipt SHA-256",
        )
        or provision["pod_id"] != expected_pod_id
    ):
        raise FreezeError("worker launch differs from the externally pinned Runpod allocation")
    if (
        set(ledger_binding)
        != {
            "budget_start_file_sha256",
            "file_sha256",
            "ledger_digest",
            "ledger_root",
            "path",
        }
        or any(ledger.get(key) != ledger_binding.get(key) for key in ledger_binding)
        or provision_binding != provision
        or ledger.get("runpod_provision") != provision
        or receipt.get("model_integration_audits") != ledger.get("model_integration_audits")
        or receipt.get("execution_uuid") != ledger["execution_uuid"]
        or Path(str(ledger["ledger_root"])) != execution_root / "itt-ledger" / str(ledger["execution_uuid"])
    ):
        raise FreezeError("worker launch receipt/ITT ledger binding changed")
    return {
        "path": str(launch_path),
        "file_sha256": sha256_file(launch_path),
        "receipt_digest": receipt["receipt_digest"],
        "worker_index": worker_index,
        "execution_uuid": receipt["execution_uuid"],
        "gpu_uuid": receipt["gpu_uuid"],
        "started_monotonic_ns": receipt["started_monotonic_ns"],
        "ledger": ledger,
        "runpod_provision": provision,
        "model_integration_audits": copy.deepcopy(dict(ledger["model_integration_audits"])),
        "runtime_environment": copy.deepcopy(dict(runtime_environment)),
    }


def initialize_attempt_ledger(
    *,
    verified: VerifiedFreeze,
    ledger_root: str | Path,
    execution_uuid: str,
    provision_receipt_path: str | Path,
    expected_provision_receipt_sha256: str,
    expected_pod_id: str,
    model_integration_audit_paths: Mapping[str, str | Path],
) -> dict[str, Any]:
    """Preallocate all 160 ITT keys after both pretraining boundary audits."""

    try:
        parsed_uuid = uuid.UUID(execution_uuid)
    except (ValueError, AttributeError) as error:
        raise FreezeError("execution UUID must be a canonical UUID4") from error
    if parsed_uuid.version != 4 or str(parsed_uuid) != execution_uuid:
        raise FreezeError("execution UUID must be a canonical UUID4")
    root = Path(ledger_root).resolve()
    if root.name != execution_uuid:
        raise FreezeError("never-overwritten attempt directory must be named by execution UUID")
    provision = verify_runpod_provision_receipt(
        verified=verified,
        receipt_path=provision_receipt_path,
        expected_receipt_sha256=expected_provision_receipt_sha256,
        expected_pod_id=expected_pod_id,
    )
    expected_provision_path = root.parent.parent / "runpod-provision-receipt.json"
    if Path(str(provision["path"])) != expected_provision_path.resolve():
        raise FreezeError("bound Runpod provision receipt is outside the exact execution root")
    if set(model_integration_audit_paths) != set(CONFIG_SPECS):
        raise FreezeError("ITT ledger requires exactly both model integration audits")
    model_integration_audits = {
        panel_id: verify_model_integration_audit(
            verified=verified,
            panel_id=panel_id,
            audit_path=model_integration_audit_paths[panel_id],
        )
        for panel_id in CONFIG_SPECS
    }
    execution_root = root.parent.parent
    if any(Path(str(audit["path"])).parent != execution_root for audit in model_integration_audits.values()):
        raise FreezeError("model integration audits lie outside the exact execution root")
    started_unix_ns = time.time_ns()
    provisioning = provision.get("provisioning")
    if not isinstance(provisioning, Mapping):
        raise FreezeError("verified provision receipt omits its absolute termination guard")
    terminate_after = _canonical_utc(
        str(provisioning.get("terminate_after_utc", "")),
        "Runpod absolute termination time",
    )
    terminate_unix_ns = int(terminate_after.timestamp()) * 1_000_000_000
    provision_remaining_ns = terminate_unix_ns - started_unix_ns
    minimum_remaining_ns = (WALL_CEILING_SECONDS + PROVISION_WATCHDOG_GRACE_SECONDS) * 1_000_000_000
    if provision_remaining_ns < minimum_remaining_ns:
        raise FreezeError(
            "Runpod auto-termination leaves less than the frozen 14-hour budget plus watchdog grace"
        )
    root.parent.mkdir(parents=True, exist_ok=True)
    try:
        root.mkdir(mode=0o700)
    except FileExistsError as error:
        raise FreezeError("execution UUID attempt directory already exists and cannot be reused") from error
    (root / "starts").mkdir()
    (root / "terminals").mkdir()
    (root / "leases").mkdir()
    rows = [
        {
            "panel_id": row["panel_id"],
            "plan_key": row["plan_key"],
            "run_id": row["run_id"],
            "seed": row["seed"],
            "worker_index": row["worker_index"],
            "worker_order": row["worker_order"],
            "initial_state": "preallocated_not_started",
        }
        for row in verified.all_rows
    ]
    body = {
        "schema": ITT_LEDGER_SCHEMA,
        "schema_version": ITT_LEDGER_SCHEMA_VERSION,
        "study_id": "g00f",
        "execution_uuid": execution_uuid,
        "freeze_file_sha256": verified.file_sha256,
        "freeze_digest": verified.digest,
        "planned_runs": 160,
        "runpod_provision": provision,
        "model_integration_audits": model_integration_audits,
        "rows": rows,
        "policy": {
            "attempts_per_plan_key": 1,
            "retry_after_start": False,
            "resume_after_start": False,
            "replacement_or_resampling": False,
            "recorded_failed_attempt_fails_study": True,
            "start_without_terminal_is_incomplete_and_fails_study": True,
            "global_fail_fast_trigger": "infrastructure_or_execution_exception_only",
            "metric_or_prediction_dependent_stop_forbidden": True,
            "remaining_state_after_global_failure": "not_started_after_failure",
            "outcome_file_visibility": (
                "metrics_predictions_summary_chmod_000_until_all_160_terminal_complete"
            ),
        },
        "g01_launch_authorized": False,
    }
    payload = {**body, "ledger_digest": semantic_digest(body)}
    target = root / "ledger.json"
    exclusive_json(target, payload)
    budget_body = {
        "schema": "goalzendo.g00f_monotonic_budget_start",
        "schema_version": 1,
        "execution_uuid": execution_uuid,
        "freeze_file_sha256": verified.file_sha256,
        "freeze_digest": verified.digest,
        "wall_ceiling_seconds": WALL_CEILING_SECONDS,
        "worker_count": WORKER_COUNT,
        "started_unix_ns": started_unix_ns,
        "started_monotonic_ns": time.monotonic_ns(),
        "provision_terminate_after_utc": provisioning["terminate_after_utc"],
        "provision_remaining_ns_at_start": provision_remaining_ns,
        "provision_required_remaining_ns": minimum_remaining_ns,
        "outcomes_seen": False,
        "g01_launch_authorized": False,
    }
    budget_payload = {**budget_body, "budget_digest": semantic_digest(budget_body)}
    exclusive_json(root / "budget-start.json", budget_payload)
    return {
        "path": str(target),
        "file_sha256": sha256_file(target),
        "ledger_digest": payload["ledger_digest"],
        "ledger_root": str(root),
        "execution_uuid": execution_uuid,
        "budget_start_file_sha256": sha256_file(root / "budget-start.json"),
        "budget_digest": budget_payload["budget_digest"],
        "runpod_provision": provision,
        "model_integration_audits": model_integration_audits,
    }


def verify_attempt_ledger(
    *,
    verified: VerifiedFreeze,
    ledger_root: str | Path,
) -> dict[str, Any]:
    root = Path(ledger_root).resolve()
    payload = strict_json(root / "ledger.json", "G00-F ITT ledger")
    body = {key: value for key, value in payload.items() if key != "ledger_digest"}
    expected_rows = [
        {
            "panel_id": row["panel_id"],
            "plan_key": row["plan_key"],
            "run_id": row["run_id"],
            "seed": row["seed"],
            "worker_index": row["worker_index"],
            "worker_order": row["worker_order"],
            "initial_state": "preallocated_not_started",
        }
        for row in verified.all_rows
    ]
    expected_policy = {
        "attempts_per_plan_key": 1,
        "retry_after_start": False,
        "resume_after_start": False,
        "replacement_or_resampling": False,
        "recorded_failed_attempt_fails_study": True,
        "start_without_terminal_is_incomplete_and_fails_study": True,
        "global_fail_fast_trigger": "infrastructure_or_execution_exception_only",
        "metric_or_prediction_dependent_stop_forbidden": True,
        "remaining_state_after_global_failure": "not_started_after_failure",
        "outcome_file_visibility": ("metrics_predictions_summary_chmod_000_until_all_160_terminal_complete"),
    }
    execution_uuid = str(payload.get("execution_uuid", ""))
    try:
        parsed_uuid = uuid.UUID(execution_uuid)
    except (ValueError, AttributeError) as error:
        raise FreezeError("G00-F ITT ledger execution UUID is invalid") from error
    budget = strict_json(root / "budget-start.json", "G00-F monotonic budget start")
    budget_body = {key: value for key, value in budget.items() if key != "budget_digest"}
    allowed_root_names = {
        "budget-start.json",
        "budget-timeout.json",
        "global-stop.json",
        "leases",
        "ledger.json",
        "panel-unseal.json",
        "starts",
        "terminals",
    }
    if (
        any(path.is_symlink() for path in root.iterdir())
        or {path.name for path in root.iterdir()} - allowed_root_names
    ):
        raise FreezeError("G00-F ITT ledger root contains an unexpected entry")
    expected_receipt_names = {f"{row['plan_key']}.json" for row in verified.all_rows}
    for directory_name in ("starts", "terminals", "leases"):
        directory = root / directory_name
        if directory.is_dir():
            entries = set(directory.iterdir())
            if (
                any(path.is_symlink() or not path.is_file() for path in entries)
                or {path.name for path in entries} - expected_receipt_names
            ):
                raise FreezeError(f"G00-F {directory_name} inventory contains an extra receipt")
    provision_binding = payload.get("runpod_provision")
    if not isinstance(provision_binding, Mapping):
        raise FreezeError("G00-F ITT ledger omits the externally pinned pod allocation")
    provision_path = root.parent.parent / "runpod-provision-receipt.json"
    provision = verify_runpod_provision_receipt(
        verified=verified,
        receipt_path=provision_path,
        expected_receipt_sha256=str(provision_binding.get("file_sha256", "")),
        expected_pod_id=str(provision_binding.get("pod_id", "")),
    )
    audit_bindings = payload.get("model_integration_audits")
    if not isinstance(audit_bindings, Mapping) or set(audit_bindings) != set(CONFIG_SPECS):
        raise FreezeError("G00-F ITT ledger omits the model integration audits")
    model_integration_audits = {
        panel_id: verify_model_integration_audit(
            verified=verified,
            panel_id=panel_id,
            audit_path=root.parent.parent / f"model-integration-audit-{panel_id.removeprefix('g00f-')}.json",
            replay_model_snapshot=False,
        )
        for panel_id in CONFIG_SPECS
    }
    budget_started_unix_ns = budget.get("started_unix_ns")
    budget_started_monotonic_ns = budget.get("started_monotonic_ns")
    budget_provision_remaining_ns = budget.get("provision_remaining_ns_at_start")
    provision_terminate_unix_ns = (
        int(
            _canonical_utc(
                str(provision["provisioning"]["terminate_after_utc"]),
                "Runpod absolute termination time",
            ).timestamp()
        )
        * 1_000_000_000
    )
    if (
        payload.get("schema") != ITT_LEDGER_SCHEMA
        or payload.get("schema_version") != ITT_LEDGER_SCHEMA_VERSION
        or payload.get("freeze_file_sha256") != verified.file_sha256
        or payload.get("freeze_digest") != verified.digest
        or payload.get("planned_runs") != 160
        or provision_binding != provision
        or audit_bindings != model_integration_audits
        or parsed_uuid.version != 4
        or str(parsed_uuid) != execution_uuid
        or root.name != execution_uuid
        or payload.get("rows") != expected_rows
        or payload.get("policy") != expected_policy
        or payload.get("g01_launch_authorized") is not False
        or payload.get("ledger_digest") != semantic_digest(body)
        or not (root / "starts").is_dir()
        or not (root / "terminals").is_dir()
        or not (root / "leases").is_dir()
        or budget.get("schema") != "goalzendo.g00f_monotonic_budget_start"
        or budget.get("schema_version") != 1
        or budget.get("execution_uuid") != execution_uuid
        or budget.get("freeze_file_sha256") != verified.file_sha256
        or budget.get("freeze_digest") != verified.digest
        or budget.get("wall_ceiling_seconds") != WALL_CEILING_SECONDS
        or budget.get("worker_count") != WORKER_COUNT
        or budget.get("provision_terminate_after_utc") != provision["provisioning"]["terminate_after_utc"]
        or isinstance(budget_started_unix_ns, bool)
        or not isinstance(budget_started_unix_ns, int)
        or budget_started_unix_ns <= 0
        or isinstance(budget_started_monotonic_ns, bool)
        or not isinstance(budget_started_monotonic_ns, int)
        or budget_started_monotonic_ns <= 0
        or isinstance(budget_provision_remaining_ns, bool)
        or not isinstance(budget_provision_remaining_ns, int)
        or budget_provision_remaining_ns != provision_terminate_unix_ns - budget_started_unix_ns
        or budget_provision_remaining_ns
        < (WALL_CEILING_SECONDS + PROVISION_WATCHDOG_GRACE_SECONDS) * 1_000_000_000
        or budget.get("provision_required_remaining_ns")
        != (WALL_CEILING_SECONDS + PROVISION_WATCHDOG_GRACE_SECONDS) * 1_000_000_000
        or budget.get("outcomes_seen") is not False
        or budget.get("g01_launch_authorized") is not False
        or budget.get("budget_digest") != semantic_digest(budget_body)
    ):
        raise FreezeError("G00-F ITT ledger does not match the exact prospective plan/policy")
    return {
        "ledger_root": str(root),
        "path": str((root / "ledger.json").resolve()),
        "file_sha256": sha256_file(root / "ledger.json"),
        "ledger_digest": payload["ledger_digest"],
        "execution_uuid": execution_uuid,
        "budget_start": budget,
        "budget_start_file_sha256": sha256_file(root / "budget-start.json"),
        "runpod_provision": provision,
        "model_integration_audits": model_integration_audits,
    }


def enforce_monotonic_budget(ledger: Mapping[str, Any]) -> None:
    budget = ledger.get("budget_start")
    if not isinstance(budget, Mapping):
        raise FreezeError("verified ITT ledger lacks monotonic budget start")
    started = budget.get("started_monotonic_ns")
    if isinstance(started, bool) or not isinstance(started, int) or started < 0:
        raise FreezeError("monotonic budget start is malformed")
    elapsed = time.monotonic_ns() - started
    if elapsed < 0 or elapsed > WALL_CEILING_SECONDS * 1_000_000_000:
        raise FreezeError("G00-F monotonic 14-hour wall budget expired")


def _attempt_receipt(
    *,
    verified: VerifiedFreeze,
    row: Mapping[str, Any],
    kind: str,
    state: str,
    launch_receipt: Mapping[str, Any] | None = None,
    error_type: str | None = None,
    caused_by_plan_key: str | None = None,
    execution_uuid: str,
) -> dict[str, Any]:
    body = {
        "schema": ATTEMPT_RECEIPT_SCHEMA,
        "schema_version": ATTEMPT_RECEIPT_SCHEMA_VERSION,
        "kind": kind,
        "state": state,
        "freeze_file_sha256": verified.file_sha256,
        "freeze_digest": verified.digest,
        "execution_uuid": execution_uuid,
        "panel_id": row["panel_id"],
        "plan_key": row["plan_key"],
        "run_id": row["run_id"],
        "seed": row["seed"],
        "worker_index": row["worker_index"],
        "worker_order": row["worker_order"],
        "launch_receipt": dict(launch_receipt) if launch_receipt is not None else None,
        "error_type": error_type,
        "caused_by_plan_key": caused_by_plan_key,
        "outcome_metrics_read": False,
        "predictions_read": False,
        "g01_launch_authorized": False,
    }
    return {**body, "receipt_digest": semantic_digest(body)}


def _ledger_receipt_path(root: Path, kind: str, plan_key: str) -> Path:
    directory = "starts" if kind == "start" else "terminals"
    return root / directory / f"{plan_key}.json"


def _record_global_failure(
    *,
    verified: VerifiedFreeze,
    ledger_root: Path,
    failed_row: Mapping[str, Any],
    error_type: str,
    execution_uuid: str,
    trigger: str = "infrastructure_or_execution_exception",
) -> None:
    if trigger not in OPERATIONAL_FAILURE_TRIGGERS:
        raise FreezeError("G00-F global stop trigger is not a frozen operational failure")
    body = {
        "schema": "goalzendo.g00f_global_execution_stop",
        "schema_version": 1,
        "freeze_file_sha256": verified.file_sha256,
        "freeze_digest": verified.digest,
        "execution_uuid": execution_uuid,
        "trigger": trigger,
        "failed_plan_key": failed_row["plan_key"],
        "error_type": error_type,
        "outcome_metrics_read": False,
        "predictions_read": False,
        "g01_launch_authorized": False,
    }
    payload = {**body, "stop_digest": semantic_digest(body)}
    stop = ledger_root / "global-stop.json"
    try:
        exclusive_json(stop, payload)
    except FreezeError:
        existing = strict_json(stop, "existing G00-F global stop")
        existing_body = {key: value for key, value in existing.items() if key != "stop_digest"}
        if (
            existing.get("schema") != "goalzendo.g00f_global_execution_stop"
            or existing.get("schema_version") != 1
            or existing.get("freeze_file_sha256") != verified.file_sha256
            or existing.get("freeze_digest") != verified.digest
            or existing.get("execution_uuid") != execution_uuid
            or existing.get("trigger") not in OPERATIONAL_FAILURE_TRIGGERS
            or existing.get("outcome_metrics_read") is not False
            or existing.get("predictions_read") is not False
            or existing.get("g01_launch_authorized") is not False
            or existing.get("stop_digest") != semantic_digest(existing_body)
        ):
            raise FreezeError("existing G00-F global stop is not authentic") from None


def _terminalize_inflight_after_failure(
    *,
    verified: VerifiedFreeze,
    ledger_root: Path,
    error_type: str,
    caused_by_plan_key: str,
    execution_uuid: str,
) -> None:
    """Close every lease/start that lost its worker, without replacement."""

    for row in verified.all_rows:
        plan_key = str(row["plan_key"])
        start_path = _ledger_receipt_path(ledger_root, "start", plan_key)
        terminal_path = _ledger_receipt_path(ledger_root, "terminal", plan_key)
        lease_path = ledger_root / "leases" / f"{plan_key}.json"
        if terminal_path.exists() or (not start_path.exists() and not lease_path.exists()):
            continue
        start = strict_json(start_path, "in-flight G00-F attempt start") if start_path.is_file() else None
        launch = start.get("launch_receipt") if isinstance(start, Mapping) else None
        state = "failed" if start is not None else "failed_before_start_receipt"
        receipt = _attempt_receipt(
            verified=verified,
            row=row,
            kind="terminal",
            state=state,
            launch_receipt=launch if isinstance(launch, Mapping) else None,
            error_type=error_type,
            caused_by_plan_key=caused_by_plan_key,
            execution_uuid=execution_uuid,
        )
        with suppress(FreezeError):
            exclusive_json(terminal_path, receipt)


def reconcile_execution_failure(
    *,
    verified: VerifiedFreeze,
    ledger_root: str | Path,
    error_type: str,
    trigger: str,
    cancel_receipt: str | Path,
) -> dict[str, Any]:
    """Persist an outcome-blind coordinator stop after workers are signalled."""

    if trigger not in OPERATIONAL_FAILURE_TRIGGERS - {"monotonic_14h_deadline"}:
        raise FreezeError("reconcile-failure requires a registered non-metric operational trigger")
    if not error_type or len(error_type) > 128:
        raise FreezeError("reconcile-failure requires a short nonempty error type")
    ledger = verify_attempt_ledger(verified=verified, ledger_root=ledger_root)
    root = Path(str(ledger["ledger_root"]))
    execution_uuid = str(ledger["execution_uuid"])
    pending = [
        row
        for row in verified.all_rows
        if (root / "starts" / f"{row['plan_key']}.json").exists()
        and not (root / "terminals" / f"{row['plan_key']}.json").exists()
    ]
    failed_row: Mapping[str, Any] = (
        pending[0]
        if pending
        else {
            "panel_id": "runtime",
            "plan_key": "coordinator",
            "run_id": "none",
            "seed": 0,
            "worker_index": -1,
            "worker_order": -1,
        }
    )
    caused_by = str(failed_row["plan_key"])
    _record_global_failure(
        verified=verified,
        ledger_root=root,
        failed_row=failed_row,
        error_type=error_type,
        execution_uuid=execution_uuid,
        trigger=trigger,
    )
    _terminalize_inflight_after_failure(
        verified=verified,
        ledger_root=root,
        error_type=error_type,
        caused_by_plan_key=caused_by,
        execution_uuid=execution_uuid,
    )
    _mark_unstarted_after_failure(
        verified=verified,
        ledger_root=root,
        caused_by_plan_key=caused_by,
        execution_uuid=execution_uuid,
    )
    terminals = _terminal_receipts(verified, root)
    counts: dict[str, int] = {}
    for receipt in terminals.values():
        state = str(receipt.get("state", "unknown"))
        counts[state] = counts.get(state, 0) + 1
    body = {
        "schema": "goalzendo.g00f_coordinator_cancel",
        "schema_version": 1,
        "execution_uuid": execution_uuid,
        "freeze_file_sha256": verified.file_sha256,
        "freeze_digest": verified.digest,
        "trigger": trigger,
        "error_type": error_type,
        "terminal_state_counts": dict(sorted(counts.items())),
        "outcome_metrics_read": False,
        "predictions_read": False,
        "g01_launch_authorized": False,
    }
    payload = {**body, "cancel_digest": semantic_digest(body)}
    target = Path(cancel_receipt).resolve()
    try:
        exclusive_json(target, payload)
    except FreezeError:
        existing = strict_json(target, "existing G00-F coordinator cancel receipt")
        if existing != payload:
            raise FreezeError("existing coordinator cancel receipt differs from this stop") from None
    return {
        "path": str(target),
        "file_sha256": sha256_file(target),
        "cancel_digest": payload["cancel_digest"],
    }


def record_budget_timeout(
    *,
    verified: VerifiedFreeze,
    ledger_root: str | Path,
) -> dict[str, Any]:
    """Record the monotonic deadline and terminalize every unstarted key."""

    ledger = verify_attempt_ledger(verified=verified, ledger_root=ledger_root)
    root = Path(str(ledger["ledger_root"]))
    budget = ledger["budget_start"]
    elapsed_ns = time.monotonic_ns() - int(budget["started_monotonic_ns"])
    if elapsed_ns < WALL_CEILING_SECONDS * 1_000_000_000:
        raise FreezeError("cannot record a budget timeout before the immutable deadline")
    body = {
        "schema": "goalzendo.g00f_monotonic_budget_timeout",
        "schema_version": 1,
        "execution_uuid": ledger["execution_uuid"],
        "freeze_file_sha256": verified.file_sha256,
        "freeze_digest": verified.digest,
        "elapsed_monotonic_ns": elapsed_ns,
        "wall_ceiling_seconds": WALL_CEILING_SECONDS,
        "cancel_scope": "all_four_workers",
        "outcome_metrics_read": False,
        "predictions_read": False,
        "g01_launch_authorized": False,
    }
    payload = {**body, "timeout_digest": semantic_digest(body)}
    target = root / "budget-timeout.json"
    try:
        exclusive_json(target, payload)
    except FreezeError:
        existing = strict_json(target, "existing G00-F monotonic budget timeout")
        if (
            existing.get("schema") != payload["schema"]
            or existing.get("execution_uuid") != payload["execution_uuid"]
            or existing.get("freeze_digest") != payload["freeze_digest"]
            or existing.get("outcome_metrics_read") is not False
            or existing.get("predictions_read") is not False
        ):
            raise FreezeError("existing monotonic timeout receipt differs from this deadline") from None
        payload = existing
    sentinel = {
        "panel_id": "runtime",
        "plan_key": "monotonic-budget",
        "run_id": "none",
        "seed": 0,
        "worker_index": -1,
        "worker_order": -1,
    }
    _record_global_failure(
        verified=verified,
        ledger_root=root,
        failed_row=sentinel,
        error_type="MonotonicBudgetExpired",
        execution_uuid=str(ledger["execution_uuid"]),
        trigger="monotonic_14h_deadline",
    )
    _terminalize_inflight_after_failure(
        verified=verified,
        ledger_root=root,
        error_type="MonotonicBudgetExpired",
        caused_by_plan_key="monotonic-budget",
        execution_uuid=str(ledger["execution_uuid"]),
    )
    _mark_unstarted_after_failure(
        verified=verified,
        ledger_root=root,
        caused_by_plan_key="monotonic-budget",
        execution_uuid=str(ledger["execution_uuid"]),
    )
    return {
        "path": str(target),
        "file_sha256": sha256_file(target),
        "timeout_digest": payload["timeout_digest"],
    }


def _mark_unstarted_after_failure(
    *,
    verified: VerifiedFreeze,
    ledger_root: Path,
    caused_by_plan_key: str,
    execution_uuid: str,
) -> None:
    for row in verified.all_rows:
        plan_key = str(row["plan_key"])
        start = _ledger_receipt_path(ledger_root, "start", plan_key)
        terminal = _ledger_receipt_path(ledger_root, "terminal", plan_key)
        if start.exists() or terminal.exists():
            continue
        receipt = _attempt_receipt(
            verified=verified,
            row=row,
            kind="terminal",
            state="not_started_after_failure",
            caused_by_plan_key=caused_by_plan_key,
            execution_uuid=execution_uuid,
        )
        with suppress(FreezeError):
            exclusive_json(terminal, receipt)


def _seal_outcome_files(path: Path) -> dict[str, dict[str, Any]]:
    """Cooperatively hide every outcome-bearing core artifact until ITT close."""

    result: dict[str, dict[str, Any]] = {}
    for name in ("metrics.jsonl", "predictions.jsonl", "summary.json"):
        target = path / name
        if target.is_symlink() or not target.is_file():
            raise FreezeError(f"completed G00-F run lacks direct outcome file before sealing: {name}")
        digest = sha256_file(target)
        size = target.stat().st_size
        os.chmod(target, 0)
        result[name] = {"path": str(target), "bytes": size, "sha256": digest, "sealed_mode": 0}
    return result


def _terminal_receipts(verified: VerifiedFreeze, ledger_root: Path) -> dict[str, Mapping[str, Any]]:
    result: dict[str, Mapping[str, Any]] = {}
    for row in verified.all_rows:
        plan_key = str(row["plan_key"])
        path = _ledger_receipt_path(ledger_root, "terminal", plan_key)
        if path.is_file():
            result[plan_key] = strict_json(path, f"terminal receipt {plan_key}")
    return result


def _unseal_if_complete(verified: VerifiedFreeze, ledger_root: Path) -> None:
    receipts = _terminal_receipts(verified, ledger_root)
    if len(receipts) != 160 or any(receipt.get("state") != "complete" for receipt in receipts.values()):
        return
    target = ledger_root / "panel-unseal.json"
    paths: list[dict[str, Any]] = []
    validated_files: list[tuple[Path, Mapping[str, Any], Mapping[str, Any], str]] = []
    for row in verified.all_rows:
        terminal = receipts[str(row["plan_key"])]
        terminal_body = {key: value for key, value in terminal.items() if key != "receipt_digest"}
        start_path = _ledger_receipt_path(ledger_root, "start", str(row["plan_key"]))
        start = strict_json(start_path, "start receipt before outcome unseal")
        start_body = {key: value for key, value in start.items() if key != "receipt_digest"}
        if (
            set(terminal_body)
            != {
                "caused_by_plan_key",
                "error_type",
                "execution_uuid",
                "freeze_digest",
                "freeze_file_sha256",
                "g01_launch_authorized",
                "kind",
                "launch_receipt",
                "outcome_file_seals",
                "outcome_metrics_read",
                "panel_id",
                "plan_key",
                "predictions_read",
                "run_id",
                "schema",
                "schema_version",
                "seed",
                "state",
                "worker_index",
                "worker_order",
            }
            or terminal.get("schema") != ATTEMPT_RECEIPT_SCHEMA
            or terminal.get("schema_version") != ATTEMPT_RECEIPT_SCHEMA_VERSION
            or terminal.get("kind") != "terminal"
            or terminal.get("state") != "complete"
            or terminal.get("freeze_file_sha256") != verified.file_sha256
            or terminal.get("freeze_digest") != verified.digest
            or terminal.get("execution_uuid") != ledger_root.name
            or terminal.get("panel_id") != row["panel_id"]
            or terminal.get("plan_key") != row["plan_key"]
            or terminal.get("run_id") != row["run_id"]
            or terminal.get("seed") != row["seed"]
            or terminal.get("worker_index") != row["worker_index"]
            or terminal.get("worker_order") != row["worker_order"]
            or not isinstance(terminal.get("launch_receipt"), Mapping)
            or terminal.get("error_type") is not None
            or terminal.get("caused_by_plan_key") is not None
            or terminal.get("outcome_metrics_read") is not False
            or terminal.get("predictions_read") is not False
            or terminal.get("g01_launch_authorized") is not False
            or terminal.get("receipt_digest") != semantic_digest(terminal_body)
            or start.get("schema") != ATTEMPT_RECEIPT_SCHEMA
            or start.get("schema_version") != ATTEMPT_RECEIPT_SCHEMA_VERSION
            or start.get("kind") != "start"
            or start.get("state") != "started"
            or start.get("plan_key") != row["plan_key"]
            or start.get("execution_uuid") != ledger_root.name
            or start.get("freeze_digest") != verified.digest
            or start.get("receipt_digest") != semantic_digest(start_body)
            or start.get("launch_receipt") != terminal.get("launch_receipt")
        ):
            raise FreezeError("complete terminal receipt changed before outcome unseal")
        expected_files = terminal.get("outcome_file_seals")
        if not isinstance(expected_files, Mapping) or set(expected_files) != {
            "metrics.jsonl",
            "predictions.jsonl",
            "summary.json",
        }:
            raise FreezeError("complete terminal omits the exact sealed outcome inventory")
        for name in ("metrics.jsonl", "predictions.jsonl", "summary.json"):
            outcome_file = Path(str(row["artifact_path"])) / name
            expected = expected_files[name]
            if (
                not isinstance(expected, Mapping)
                or set(expected) != {"bytes", "path", "sealed_mode", "sha256"}
                or expected.get("path") != str(outcome_file)
                or not outcome_file.is_file()
                or outcome_file.is_symlink()
                or outcome_file.stat().st_size != expected.get("bytes")
                or stat.S_IMODE(outcome_file.stat().st_mode) not in {0, 0o400}
                or sha256_file(outcome_file) != expected.get("sha256")
            ):
                raise FreezeError("sealed outcome bytes changed before panel completion")
            validated_files.append((outcome_file, expected, row, name))
    for outcome_file, expected, row, name in validated_files:
        os.chmod(outcome_file, 0o400)
        paths.append(
            {
                "plan_key": row["plan_key"],
                "run_id": row["run_id"],
                "name": name,
                "bytes": expected["bytes"],
                "sha256": expected["sha256"],
                "unsealed_mode": 0o400,
            }
        )
    body = {
        "schema": PANEL_UNSEAL_SCHEMA,
        "schema_version": PANEL_UNSEAL_SCHEMA_VERSION,
        "freeze_file_sha256": verified.file_sha256,
        "freeze_digest": verified.digest,
        "terminal_complete_count": 160,
        "outcome_files": paths,
        "outcomes_inspected_before_unseal": False,
        "g01_launch_authorized": False,
    }
    expected_payload = {**body, "unseal_digest": semantic_digest(body)}
    try:
        exclusive_json(target, expected_payload)
    except FreezeError:
        # The final workers can observe all 160 terminal receipts at the same
        # time.  O_EXCL chooses one winner; every loser must verify, not fail
        # or overwrite, the winner's byte-equivalent semantic payload.
        existing: Mapping[str, Any] | None = None
        for _attempt in range(100):
            try:
                existing = strict_json(target, "concurrent G00-F panel unseal winner")
                break
            except FreezeError:
                time.sleep(0.01)
        if existing != expected_payload:
            raise FreezeError("concurrent G00-F panel unseal winner differs from expectation") from None


def _spec_lookup(verified: VerifiedFreeze) -> dict[str, RunSpec]:
    result: dict[str, RunSpec] = {}
    for _panel_id, specification in CONFIG_SPECS.items():
        config = load_config(verified.repo / str(specification["path"]))
        for spec in build_plan(config):
            result[spec.plan_key] = spec
    if len(result) != 160:
        raise FreezeError("could not reconstruct the exact 160 frozen RunSpecs")
    return result


def _authenticated_local_model_loader(
    model_receipts: Mapping[str, Mapping[str, Any]],
) -> Callable[..., tuple[Any, Any]]:
    """Build the only loader allowed to consume G00-F model weights."""

    if set(model_receipts) != set(CONFIG_SPECS):
        raise FreezeError("authenticated local model roots are required for G00-F execution")

    def exact_local_loader(
        model_config: Mapping[str, Any],
        update_config: Mapping[str, Any] | None = None,
        *,
        device_map: str | Mapping[str, Any] | None = None,
    ) -> tuple[Any, Any]:
        if os.environ.get("HF_HUB_OFFLINE") != "1" or os.environ.get("TRANSFORMERS_OFFLINE") != "1":
            raise FreezeError("G00-F local model loading requires both offline-mode guards")
        requested = str(model_config.get("name", ""))
        revision = str(model_config.get("revision", ""))
        matches = [
            panel_id
            for panel_id, static in CONFIG_SPECS.items()
            if static["model"] == requested and static["revision"] == revision
        ]
        if len(matches) != 1 or bool(model_config.get("trust_remote_code", False)):
            raise FreezeError("G00-F attempted to load an unregistered model or remote code")
        panel_id = matches[0]
        root = Path(str(model_receipts[panel_id]["snapshot_root"])).resolve()
        try:
            from transformers import (  # type: ignore[import-not-found]
                AutoModelForCausalLM,
                AutoTokenizer,
            )
        except ImportError as error:  # pragma: no cover - frozen GPU runtime dependency
            raise FreezeError("frozen transformers dependency is absent") from error
        from goalzendo.modeling import apply_update_method, resolve_torch_dtype

        tokenizer = AutoTokenizer.from_pretrained(
            root,
            local_files_only=True,
            trust_remote_code=False,
        )
        if tokenizer.pad_token_id is None:
            if tokenizer.eos_token_id is None:
                raise FreezeError("authenticated tokenizer defines neither pad nor EOS")
            tokenizer.pad_token = tokenizer.eos_token
        kwargs: dict[str, Any] = {
            "local_files_only": True,
            "torch_dtype": resolve_torch_dtype(model_config.get("dtype", "bfloat16")),
            "trust_remote_code": False,
        }
        if device_map is not None:
            kwargs["device_map"] = device_map
        model = AutoModelForCausalLM.from_pretrained(root, **kwargs)
        # A regular-file materialization has no hub metadata.  Attach only the
        # revision that the direct-leaf receipt already authenticated.
        model.config._commit_hash = revision
        if isinstance(getattr(tokenizer, "init_kwargs", None), dict):
            tokenizer.init_kwargs["_commit_hash"] = revision
        if update_config is not None:
            model = apply_update_method(model, update_config)
        return model, tokenizer

    return exact_local_loader


@contextmanager
def _exact_g00f_launch_scope(
    verified: VerifiedFreeze,
    model_receipts: Mapping[str, Mapping[str, Any]] | None = None,
) -> Iterator[None]:
    """Narrowly replace only G00-F's design guard in this process."""

    import goalzendo.experiment as experiment_module
    import goalzendo.runner as runner_module

    original = runner_module.assert_launch_unlocked
    experiment_any: Any = experiment_module
    original_loader = experiment_any.load_model_and_tokenizer
    allowed_cells = {str(row["cell_id"]) for row in verified.all_rows}

    def exact_guard(
        config: Mapping[str, Any],
        *,
        repo: str | Path,
        gate_artifact: str | Path | None = None,
    ) -> Any:
        if get_path(config, "experiment.id") != "g00f":
            return original(config, repo=repo, gate_artifact=gate_artifact)
        if (
            get_path(config, "run.launch_guard") != G00F_GUARD
            or get_path(config, "run.protocol_unlocked") is not False
        ):
            raise FreezeError("G00-F launch guard/config authorization changed")
        # Rebuild the one-cell plan with one seed solely to recover its cell
        # identity.  Execution itself receives only RunSpecs from the exact
        # frozen lookup below.
        probe = copy.deepcopy(dict(config))
        seed_values = get_path(probe, "run.seeds")
        if not isinstance(seed_values, Sequence) or not seed_values:
            raise FreezeError("G00-F guarded cell has no registered seeds")
        probe["run"]["seeds"] = [int(seed_values[0])]
        cells = build_plan(probe)
        if len(cells) != 1 or cells[0].cell_id not in allowed_cells:
            raise FreezeError("G00-F guarded cell is outside the exact frozen plan")
        if (
            implementation_provenance(repo).get("implementation_fingerprint")
            != FROZEN_GOALZENDO_IMPLEMENTATION_FINGERPRINT
        ):
            raise FreezeError("GoalZendo source changed after G00-F authentication")
        return None

    runner_module.assert_launch_unlocked = exact_guard
    if model_receipts is not None:
        experiment_any.load_model_and_tokenizer = _authenticated_local_model_loader(model_receipts)
    try:
        yield
    finally:
        runner_module.assert_launch_unlocked = original
        experiment_any.load_model_and_tokenizer = original_loader


def _run_binding_payload(
    *,
    verified: VerifiedFreeze,
    row: Mapping[str, Any],
    launch_receipt: Mapping[str, Any],
    model_receipts: Mapping[str, Mapping[str, Any]],
    attempt_start: Mapping[str, Any],
    ownership_lease: Mapping[str, Any],
    execution_uuid: str,
) -> dict[str, Any]:
    body = {
        "schema": RUN_BINDING_SCHEMA,
        "schema_version": RUN_BINDING_SCHEMA_VERSION,
        "freeze_file_sha256": verified.file_sha256,
        "freeze_digest": verified.digest,
        "execution_uuid": execution_uuid,
        "plan_key": row["plan_key"],
        "run_id": row["run_id"],
        "panel_id": row["panel_id"],
        "worker_index": row["worker_index"],
        "worker_order": row["worker_order"],
        "launch_receipt": dict(launch_receipt),
        "model_receipts": {key: dict(value) for key, value in sorted(model_receipts.items())},
        "attempt_start": dict(attempt_start),
        "ownership_lease": dict(ownership_lease),
        "g01_launch_authorized": False,
    }
    return {**body, "binding_digest": semantic_digest(body)}


def run_worker(
    *,
    verified: VerifiedFreeze,
    worker_index: int,
    launch_receipt_path: str | Path,
    model_receipt_paths: Mapping[str, str | Path],
    ledger_root: str | Path,
    dry_run: bool = False,
) -> tuple[Mapping[str, Any], ...]:
    """Run one exact mixed-model worker schedule, with no reassignment path."""

    launch = verify_launch_receipt(
        verified=verified,
        worker_index=worker_index,
        receipt_path=launch_receipt_path,
    )
    if set(model_receipt_paths) != set(CONFIG_SPECS):
        raise FreezeError("worker requires exactly both frozen model snapshot receipts")
    model_receipts = {
        panel_id: verify_model_snapshot_receipt(
            verified=verified,
            panel_id=panel_id,
            receipt_path=model_receipt_paths[panel_id],
        )
        for panel_id in CONFIG_SPECS
    }
    lookup = _spec_lookup(verified)
    rows = verified.worker_rows(worker_index)
    specs = [lookup[str(row["plan_key"])] for row in rows]
    ledger = verify_attempt_ledger(verified=verified, ledger_root=ledger_root)
    ledger_path = Path(str(ledger["ledger_root"]))
    execution_uuid = str(ledger["execution_uuid"])
    if dry_run:
        with _exact_g00f_launch_scope(verified, model_receipts):
            dry_outcomes = execute_plan(specs, repo=verified.repo, dry_run=True)
        return tuple(outcome.as_dict() for outcome in dry_outcomes)

    worker_outcomes: list[Mapping[str, Any]] = []
    for row, spec in zip(rows, specs, strict=True):
        global_stop = ledger_path / "global-stop.json"
        if global_stop.is_file():
            stop = strict_json(global_stop, "G00-F global stop")
            if stop.get("freeze_digest") != verified.digest:
                raise FreezeError("global stop is not bound to this freeze")
            _mark_unstarted_after_failure(
                verified=verified,
                ledger_root=ledger_path,
                caused_by_plan_key=str(stop.get("failed_plan_key", "")),
                execution_uuid=execution_uuid,
            )
            worker_outcomes.extend(
                {
                    "plan_key": remaining["plan_key"],
                    "run_id": remaining["run_id"],
                    "path": remaining["artifact_path"],
                    "seed": remaining["seed"],
                    "state": "not_started_after_failure",
                    "error_type": None,
                }
                for remaining in rows[len(worker_outcomes) :]
            )
            break
        start_path = _ledger_receipt_path(ledger_path, "start", str(row["plan_key"]))
        terminal_path = _ledger_receipt_path(ledger_path, "terminal", str(row["plan_key"]))
        lease_path = ledger_path / "leases" / f"{row['plan_key']}.json"
        try:
            store = RunStore(
                get_path(spec.config, "run.output_root"),
                spec.config,
                spec.seed,
                verified.repo,
            )
            if store.run_id != row["run_id"] or str(store.path) != row["artifact_path"]:
                raise FreezeError("runtime RunStore identity differs from the frozen plan")
            if store.path.exists():
                raise FreezeError("G00-F forbids pre-existing run artifacts or post-start resume")
            if start_path.exists() or terminal_path.exists() or lease_path.exists():
                raise FreezeError("G00-F permits exactly one append-only attempt per plan key")
            enforce_monotonic_budget(ledger)
            lease_body = {
                "schema": "goalzendo.g00f_run_ownership_lease",
                "schema_version": 1,
                "execution_uuid": execution_uuid,
                "freeze_file_sha256": verified.file_sha256,
                "freeze_digest": verified.digest,
                "panel_id": row["panel_id"],
                "plan_key": row["plan_key"],
                "run_id": row["run_id"],
                "worker_index": row["worker_index"],
                "worker_order": row["worker_order"],
                "owner_pid": os.getpid(),
                "acquired_monotonic_ns": time.monotonic_ns(),
                "g01_launch_authorized": False,
            }
            lease_payload = {**lease_body, "lease_digest": semantic_digest(lease_body)}
            exclusive_json(lease_path, lease_payload)
            if global_stop.is_file():
                raise FreezeError("G00-F coordinator stopped execution after lease acquisition")
            lease_summary = {
                "path": str(lease_path),
                "file_sha256": sha256_file(lease_path),
                "lease_digest": lease_payload["lease_digest"],
            }
            start_payload = _attempt_receipt(
                verified=verified,
                row=row,
                kind="start",
                state="started",
                launch_receipt=launch,
                execution_uuid=execution_uuid,
            )
            exclusive_json(start_path, start_payload)
            start_summary = {
                "path": str(start_path),
                "file_sha256": sha256_file(start_path),
                "receipt_digest": start_payload["receipt_digest"],
            }
            binding_path = store.path / "g00f-freeze-binding.json"
            binding_payload = _run_binding_payload(
                verified=verified,
                row=row,
                launch_receipt=launch,
                model_receipts=model_receipts,
                attempt_start=start_summary,
                ownership_lease=lease_summary,
                execution_uuid=execution_uuid,
            )
            atomic_json(binding_path, binding_payload)
            with _exact_g00f_launch_scope(verified, model_receipts):
                (outcome,) = execute_plan([spec], repo=verified.repo)
            if outcome.state not in {"complete", "skipped"} or outcome.state == "skipped":
                raise FreezeError("fresh G00-F attempt did not complete exactly once")
            seals = _seal_outcome_files(store.path)
            terminal = _attempt_receipt(
                verified=verified,
                row=row,
                kind="terminal",
                state="complete",
                launch_receipt=launch,
                execution_uuid=execution_uuid,
            )
            terminal_body = {key: value for key, value in terminal.items() if key != "receipt_digest"}
            terminal_body["outcome_file_seals"] = seals
            terminal = {**terminal_body, "receipt_digest": semantic_digest(terminal_body)}
            exclusive_json(terminal_path, terminal)
            worker_outcomes.append(outcome.as_dict())
        except BaseException as error:
            if start_path.is_file() and not terminal_path.exists():
                terminal = _attempt_receipt(
                    verified=verified,
                    row=row,
                    kind="terminal",
                    state="failed",
                    launch_receipt=launch,
                    error_type=type(error).__name__,
                    execution_uuid=execution_uuid,
                )
                exclusive_json(terminal_path, terminal)
            elif lease_path.is_file() and not terminal_path.exists():
                terminal = _attempt_receipt(
                    verified=verified,
                    row=row,
                    kind="terminal",
                    state="failed_before_start_receipt",
                    error_type=type(error).__name__,
                    execution_uuid=execution_uuid,
                )
                exclusive_json(terminal_path, terminal)
            _record_global_failure(
                verified=verified,
                ledger_root=ledger_path,
                failed_row=row,
                error_type=type(error).__name__,
                execution_uuid=execution_uuid,
            )
            _mark_unstarted_after_failure(
                verified=verified,
                ledger_root=ledger_path,
                caused_by_plan_key=str(row["plan_key"]),
                execution_uuid=execution_uuid,
            )
            raise
        _unseal_if_complete(verified, ledger_path)
    return tuple(worker_outcomes)


__all__ = [
    "CONFIG_SPECS",
    "EXECUTION_AND_GATE_CONTRACT",
    "EXPECTED_ALL_VIEWS",
    "EXPECTED_INFORMATIVE_VIEWS",
    "FREEZE_SCHEMA",
    "FREEZE_SCHEMA_VERSION",
    "FROZEN_GOALZENDO_IMPLEMENTATION_FINGERPRINT",
    "G00F_GUARD",
    "H100_HOUR_CEILING",
    "MODEL_INTEGRATION_AUDIT_SCHEMA",
    "MODEL_LEAF_FILES",
    "MODEL_RECEIPT_SCHEMA",
    "MODEL_RUNTIME_IDENTITIES",
    "PANEL_UNSEAL_SCHEMA",
    "RUNPOD_PROVISIONING_CONTRACT",
    "RUNPOD_PROVISION_EVIDENCE",
    "RUNS_PER_PANEL",
    "TOKENIZER_RUNTIME_IDENTITY",
    "WALL_CEILING_SECONDS",
    "WORKER_COUNT",
    "FreezeError",
    "VerifiedFreeze",
    "atomic_json",
    "bind_runpod_provision_receipt",
    "canonical_json_bytes",
    "canonical_runpod_create_command",
    "create_model_integration_audit",
    "create_model_snapshot_receipt",
    "create_runpod_provision_receipt",
    "exclusive_json",
    "expected_plan_rows",
    "initialize_attempt_ledger",
    "require_sha256",
    "run_worker",
    "semantic_digest",
    "sha256_file",
    "strict_json",
    "verify_attempt_ledger",
    "verify_freeze",
    "verify_launch_receipt",
    "verify_model_integration_audit",
    "verify_model_snapshot_receipt",
    "verify_runpod_provision_receipt",
    "write_plan_files",
]
