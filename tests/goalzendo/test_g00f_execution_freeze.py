from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import inspect
import io
import json
import math
import os
import platform
import shutil
import signal
import stat
import subprocess
import sys
import tarfile
import threading
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from contextlib import suppress
from datetime import datetime, timezone
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest

import goalzendo.runner as legacy_runner
import goalzendo_g00f.cli as g00f_cli_module
import goalzendo_g00f.evaluator as evaluator_module
import goalzendo_g00f.freeze as freeze_module
from goalzendo.artifacts import stable_hash
from goalzendo.config import get_path, load_config
from goalzendo.experiment import (
    EXPERIMENT_BACKEND_VERSION,
    _counterbalanced_renderer_map,
    materialize_banks,
    render_prompt_view,
)
from goalzendo.runner import LaunchGuardError, RunSpec, build_plan
from goalzendo_g00f.evaluator import (
    EvaluationError,
    _aggregate_results,
    _audit_attempt_ledger,
    _evaluate_run,
    create_preexecution_gate,
    exact_central_binomial_interval,
    verify_gate_artifact,
)
from goalzendo_g00f.freeze import (
    CONFIG_SPECS,
    EXPECTED_INFORMATIVE_VIEWS,
    RUNPOD_PROVISION_EVIDENCE,
    WALL_CEILING_SECONDS,
    FreezeError,
    VerifiedFreeze,
    _attempt_receipt,
    _exact_g00f_launch_scope,
    _seal_outcome_files,
    _unseal_if_complete,
    bind_runpod_provision_receipt,
    create_model_integration_audit,
    create_model_snapshot_receipt,
    exclusive_json,
    expected_plan_rows,
    initialize_attempt_ledger,
    materialize_model_snapshot,
    reconcile_execution_failure,
    record_budget_timeout,
    semantic_digest,
    sha256_file,
    verify_attempt_ledger,
    verify_freeze,
    verify_launch_receipt,
    verify_model_integration_audit,
    verify_model_snapshot_receipt,
    verify_runpod_provision_receipt,
)

ROOT = Path(__file__).resolve().parents[2]
G01_CONFIG = ROOT / "configs" / "goalzendo" / "g01_known_law.yaml"
BOOTSTRAP = ROOT / "runs" / "goalzendo" / "g00f_bundle_bootstrap.py"
WATCHDOG = ROOT / "runs" / "goalzendo" / "g00f_watchdog.py"
LAUNCHER = ROOT / "runs" / "goalzendo" / "run_g00f_frozen_4h100.sh"
BUILDER = ROOT / "runs" / "goalzendo" / "build_g00f_execution_freeze.py"
PROVISION_CREATED_AT = "2026-08-11 09:30:00.034 +0000 UTC"
PROVISION_TERMINATE_AFTER = "2026-08-12T01:30:00Z"
FROZEN_TEST_LEDGER_START = datetime(2026, 8, 11, 9, 31, tzinfo=timezone.utc)
RUNPOD_PROVISIONING_CONTRACT = {
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


@pytest.fixture(autouse=True)
def _freeze_test_wall_clock(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep the synthetic provision window valid independently of wall time."""

    started_unix_ns = int(FROZEN_TEST_LEDGER_START.timestamp()) * 1_000_000_000
    monkeypatch.setattr(time, "time_ns", lambda: started_unix_ns)


def _expected_runpod_create_command(terminate_after_utc: str) -> tuple[str, ...]:
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


@pytest.fixture(scope="module")
def frozen_rows() -> dict[str, list[dict[str, Any]]]:
    return expected_plan_rows(ROOT)


def _verified(
    tmp_path: Path,
    frozen_rows: dict[str, list[dict[str, Any]]],
    *,
    local_artifacts: bool = False,
) -> VerifiedFreeze:
    plans = copy.deepcopy(frozen_rows)
    if local_artifacts:
        for rows in plans.values():
            for row in rows:
                row["artifact_path"] = str(tmp_path / "artifacts" / str(row["plan_key"]))
    configurations: dict[str, Any] = {}
    for panel_id, specification in CONFIG_SPECS.items():
        payload = f"fixture-{panel_id}\n".encode()
        leaf = {
            "path": "fixture-model.bin",
            "bytes": len(payload),
            "sha256": hashlib.sha256(payload).hexdigest(),
        }
        configurations[panel_id] = {
            "model_snapshot": {
                "repo_id": specification["model"],
                "revision": specification["revision"],
                "leaf_files": [leaf],
                "leaf_manifest_digest": semantic_digest([leaf]),
            }
        }
    return VerifiedFreeze(
        repo=ROOT,
        path=tmp_path / "execution-freeze.json",
        file_sha256="f" * 64,
        payload={
            "freeze_digest": "d" * 64,
            "runtime": {
                "image": "runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404",
                "network_volume_id": "9mut3tpzwd",
                "network_volume_mount": "/workspace",
                "data_center": "US-CA-2",
                "python_packages": {},
                "python": platform.python_version(),
                "runpod_provisioning": dict(RUNPOD_PROVISIONING_CONTRACT),
            },
            "source_bundle": {
                "archive_sha256": "a" * 64,
                "manifest_sha256": "b" * 64,
                "manifest_digest": "c" * 64,
            },
            "configurations": configurations,
        },
        plans={panel: tuple(rows) for panel, rows in plans.items()},
    )


def _case_spec(view: str) -> RunSpec:
    config = load_config(ROOT / str(CONFIG_SPECS["g00f-0p5b"]["path"]))
    return next(spec for spec in build_plan(config) if spec.config["data"]["training_view"] == view)


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True, indent=2) + "\n", encoding="utf-8")


def _external_provision_receipt(
    directory: Path,
    verified: VerifiedFreeze,
    *,
    pod_id: str = "pod-g00f-test",
    changes: dict[str, Any] | None = None,
) -> tuple[Path, str, Path]:
    directory.mkdir(parents=True, exist_ok=True)
    create_raw_path = directory / "runpod-create-response.json"
    create_raw_path.write_text(
        json.dumps({"id": pod_id, "stage": "create"}, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    raw_path = directory / "runpod-api-response.json"
    raw_path.write_text(
        json.dumps(
            {
                "id": pod_id,
                "imageName": verified.payload["runtime"]["image"],
                "gpuCount": 4,
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    body = {
        "schema": "goalzendo.g00f_runpod_provision_receipt",
        "schema_version": 1,
        "provider": "runpod",
        "capture_tool": "runpodctl 2.9.0-c094cac",
        "created_at_utc": PROVISION_CREATED_AT,
        "pod_id": pod_id,
        "image": verified.payload["runtime"]["image"],
        "gpu_count": 4,
        "gpu_catalog": {
            "display_name": "H100 SXM",
            "gpu_id": "NVIDIA H100 80GB HBM3",
        },
        "data_center": verified.payload["runtime"]["data_center"],
        "network_volume_id": verified.payload["runtime"]["network_volume_id"],
        "network_volume_mount": verified.payload["runtime"]["network_volume_mount"],
        "operational_market_snapshot": {
            "observed_at_utc": PROVISION_CREATED_AT,
            "secure_price_usd_per_gpu_hour": 3.29,
            "stock_label": "Low",
            "scientific_identity": False,
        },
        "api_response": {
            "capture_command": (
                f"runpodctl pod get {pod_id} --include-machine --include-network-volume -o json"
            ),
            "file_name": "runpod-api-response.json",
            "sha256": sha256_file(raw_path),
        },
        "provisioning": {
            "capture_order": ["create", "get"],
            "create": {
                "command_argv": list(_expected_runpod_create_command(PROVISION_TERMINATE_AFTER)),
                "file_name": "runpod-create-response.json",
                "sha256": sha256_file(create_raw_path),
            },
            "get": {
                "command_argv": [
                    "runpodctl",
                    "pod",
                    "get",
                    pod_id,
                    "--include-machine",
                    "--include-network-volume",
                    "-o",
                    "json",
                ],
                "file_name": "runpod-api-response.json",
                "sha256": sha256_file(raw_path),
            },
            "terminate_after_utc": PROVISION_TERMINATE_AFTER,
            "provision_ceiling_seconds": 16 * 60 * 60,
            "maximum_secure_cost_usd": 210.56,
        },
        "evidence_boundaries": dict(RUNPOD_PROVISION_EVIDENCE),
        "outcomes_seen": False,
        "g01_launch_authorized": False,
    }
    body.update(copy.deepcopy(changes or {}))
    receipt = {**body, "receipt_digest": semantic_digest(body)}
    path = directory / "runpod-provision-receipt.json"
    _write_json(path, receipt)
    return path, sha256_file(path), raw_path


def _initialize_test_ledger(
    root: Path,
    verified: VerifiedFreeze,
    execution_uuid: str,
) -> tuple[Path, dict[str, Any], str, str]:
    external, expected_sha, _raw = _external_provision_receipt(
        root / "external-provision",
        verified,
    )
    execution_root = root / "execution"
    bound = execution_root / "runpod-provision-receipt.json"
    provision = bind_runpod_provision_receipt(
        verified=verified,
        input_path=external,
        output_path=bound,
        expected_receipt_sha256=expected_sha,
        expected_pod_id="pod-g00f-test",
    )
    model_audits = _create_test_model_audits(execution_root, verified)
    ledger_root = execution_root / "itt-ledger" / execution_uuid
    initialize_attempt_ledger(
        verified=verified,
        ledger_root=ledger_root,
        execution_uuid=execution_uuid,
        provision_receipt_path=bound,
        expected_provision_receipt_sha256=expected_sha,
        expected_pod_id="pod-g00f-test",
        model_integration_audit_paths=model_audits,
    )
    return ledger_root, provision, expected_sha, "pod-g00f-test"


def _fake_model_integration_report(panel_id: str, snapshot_root: Path) -> dict[str, Any]:
    specification = CONFIG_SPECS[panel_id]
    identity = freeze_module.MODEL_RUNTIME_IDENTITIES[panel_id]
    dependencies = dict(freeze_module.FROZEN_MODEL_DEPENDENCIES)
    return {
        "passed": True,
        "model": {
            "requested_model": specification["model"],
            "requested_revision": specification["revision"],
            "resolved_revision": specification["revision"],
            "tokenizer_resolved_revision": specification["revision"],
            "requested_dtype": "bfloat16",
            "model_class": identity["model_class"],
            "tokenizer_class": freeze_module.TOKENIZER_RUNTIME_IDENTITY["tokenizer_class"],
            "tokenizer_name_or_path": str(snapshot_root),
            "vocabulary_size": freeze_module.TOKENIZER_RUNTIME_IDENTITY["vocabulary_size"],
            "chat_template_sha256": freeze_module.TOKENIZER_RUNTIME_IDENTITY["chat_template_sha256"],
            "action_labels": ["A", "B"],
            "action_token_ids": [[32], [33]],
            "parameter_count": identity["parameter_count"],
            "trainable_parameter_count": 0,
            "dependency_versions": dependencies,
            "torch_version": dependencies["torch"],
            "transformers_version": dependencies["transformers"],
            "peft_version": dependencies["peft"],
        },
        "runtime": {
            "requested_device": "cuda",
            "parameter_devices": ["cuda:0"],
            "parameter_dtypes": ["torch.bfloat16"],
            "dependencies": dependencies,
            "last_logit_parameter": "logits_to_keep",
        },
        "action_boundary": {
            "labels": ["A", "B"],
            "standalone_token_ids": [[32], [33]],
            "continuation_token_ids_by_prompt": [[[32], [33]], [[32], [33]]],
            "continuation_token_lengths_by_prompt": [[1, 1], [1, 1]],
            "prefix_stable": True,
        },
        "prompt_tokens": {"counts": [24, 42], "maximum": 42, "configured_maximum": None},
        "scores": {
            "finite": True,
            "normalized_log_scores": [
                [math.log(0.7310586), math.log(0.2689414)],
                [math.log(0.2689414), math.log(0.7310586)],
            ],
            "probabilities": [[0.7310586, 0.2689414], [0.2689414, 0.7310586]],
            "maximum_normalization_error": 0.0,
            "swap_invariant": True,
            "maximum_score_swap_error": 0.0,
            "maximum_probability_swap_error": 0.0,
            "swap_atol": 2e-3,
            "swap_rtol": 2e-3,
        },
    }


def _create_test_model_audits(
    execution_root: Path,
    verified: VerifiedFreeze,
) -> dict[str, Path]:
    audit_paths: dict[str, Path] = {}
    old_offline = os.environ.get("HF_HUB_OFFLINE")
    old_transformers_offline = os.environ.get("TRANSFORMERS_OFFLINE")
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    try:
        for panel_id in CONFIG_SPECS:
            suffix = panel_id.removeprefix("g00f-")
            snapshot_root = execution_root.parent / f"{execution_root.name}-snapshot-{suffix}"
            snapshot_root.mkdir(parents=True)
            payload = f"fixture-{panel_id}\n".encode()
            leaf = snapshot_root / "fixture-model.bin"
            leaf.write_bytes(payload)
            os.chmod(leaf, 0o444)
            os.chmod(snapshot_root, 0o555)
            model_receipt = execution_root / f"model-receipt-{suffix}.json"
            create_model_snapshot_receipt(
                verified=verified,
                panel_id=panel_id,
                snapshot_root=snapshot_root,
                output=model_receipt,
            )
            audit_path = execution_root / f"model-integration-audit-{suffix}.json"

            def fake_runner(
                *_args: Any,
                frozen_panel_id: str = panel_id,
                frozen_snapshot_root: Path = snapshot_root,
                **_kwargs: Any,
            ) -> dict[str, Any]:
                return _fake_model_integration_report(frozen_panel_id, frozen_snapshot_root)

            create_model_integration_audit(
                verified=verified,
                panel_id=panel_id,
                model_receipt_path=model_receipt,
                output=audit_path,
                integration_runner=fake_runner,
            )
            audit_paths[panel_id] = audit_path
    finally:
        if old_offline is None:
            os.environ.pop("HF_HUB_OFFLINE", None)
        else:
            os.environ["HF_HUB_OFFLINE"] = old_offline
        if old_transformers_offline is None:
            os.environ.pop("TRANSFORMERS_OFFLINE", None)
        else:
            os.environ["TRANSFORMERS_OFFLINE"] = old_transformers_offline
    return audit_paths


def _load_script(path: Path, name: str) -> ModuleType:
    specification = importlib.util.spec_from_file_location(name, path)
    assert specification is not None and specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def _synthetic_bootstrap_bundle(
    root: Path,
    *,
    archive_overrides: dict[str, bytes] | None = None,
) -> tuple[ModuleType, argparse.Namespace]:
    bootstrap = _load_script(BOOTSTRAP, f"g00f_bootstrap_test_{root.name}")
    controllers = {
        "bootstrap": BOOTSTRAP,
        "launcher": LAUNCHER,
        "watchdog": WATCHDOG,
    }
    relative_by_role = {role: path.relative_to(ROOT).as_posix() for role, path in controllers.items()}
    overrides = archive_overrides or {}
    source_date_epoch = 1_700_000_000
    members: list[dict[str, Any]] = []
    archived_payloads: dict[str, bytes] = {}
    for role, actual in controllers.items():
        payload = overrides.get(role, actual.read_bytes())
        relative = relative_by_role[role]
        archived_payloads[relative] = payload
        members.append(
            {
                "path": relative,
                "type": "file",
                "mode": 0o755,
                "bytes": len(payload),
                "sha256": hashlib.sha256(payload).hexdigest(),
            }
        )
    manifest_body = {
        "schema": "goalzendo.g00f_source_bundle_payload_manifest",
        "schema_version": 1,
        "source_date_epoch": source_date_epoch,
        "members": members,
    }
    manifest = {**manifest_body, "manifest_digest": bootstrap.digest(manifest_body)}
    manifest_path = root / "source-manifest.json"
    _write_json(manifest_path, manifest)

    archive_path = root / "source.tar.gz"
    with tarfile.open(archive_path, mode="w:gz") as archive:
        for row in members:
            payload = archived_payloads[str(row["path"])]
            info = tarfile.TarInfo(str(row["path"]))
            info.size = len(payload)
            info.mode = int(row["mode"])
            info.uid = 0
            info.gid = 0
            info.uname = ""
            info.gname = ""
            info.mtime = source_date_epoch
            archive.addfile(info, io.BytesIO(payload))
        payload = manifest_path.read_bytes()
        info = tarfile.TarInfo("G00F-BUNDLE-MANIFEST.json")
        info.size = len(payload)
        info.mode = 0o644
        info.uid = 0
        info.gid = 0
        info.uname = ""
        info.gname = ""
        info.mtime = source_date_epoch
        archive.addfile(info, io.BytesIO(payload))

    freeze_body = {
        "schema": "goalzendo.g00f_execution_freeze",
        "schema_version": 1,
        "controller_files": {
            role: {
                "path": relative_by_role[role],
                "sha256": sha256_file(actual),
            }
            for role, actual in controllers.items()
        },
        "source_bundle": {
            "archive_sha256": sha256_file(archive_path),
            "manifest_sha256": sha256_file(manifest_path),
            "manifest_digest": manifest["manifest_digest"],
        },
    }
    freeze = {**freeze_body, "freeze_digest": bootstrap.digest(freeze_body)}
    freeze_path = root / "freeze.json"
    _write_json(freeze_path, freeze)
    return bootstrap, argparse.Namespace(
        freeze=freeze_path,
        expected_freeze_sha256=sha256_file(freeze_path),
        archive=archive_path,
        manifest=manifest_path,
        output=root / "extracted",
        receipt=root / "receipt.json",
        actual_bootstrap=BOOTSTRAP,
        actual_launcher=LAUNCHER,
        actual_watchdog=WATCHDOG,
    )


def test_exact_plan_is_deterministic_latin_balanced_and_alternating(
    frozen_rows: dict[str, list[dict[str, Any]]],
) -> None:
    assert frozen_rows == expected_plan_rows(ROOT)
    rows = [row for panel in frozen_rows.values() for row in panel]
    assert len(rows) == 160
    assert len({row["plan_key"] for row in rows}) == 160

    for panel_id, panel_rows in frozen_rows.items():
        assert len(panel_rows) == 80
        seeds = sorted(int(seed) for seed in CONFIG_SPECS[panel_id]["seeds"])
        assert {(int(row["seed"]), int(row["cell_index"])) for row in panel_rows} == {
            (seed, cell_index) for seed in seeds for cell_index in range(8)
        }
        seed_rank = {seed: index for index, seed in enumerate(seeds)}
        for row in panel_rows:
            assert row["worker_index"] == (seed_rank[int(row["seed"])] + int(row["cell_index"])) % 4

    for worker_index in range(4):
        worker = sorted(
            (row for row in rows if row["worker_index"] == worker_index),
            key=lambda row: int(row["worker_order"]),
        )
        assert len(worker) == 40
        assert [row["worker_order"] for row in worker] == list(range(40))
        assert [row["panel_id"] for row in worker[::2]] == [
            "g00f-0p5b" if worker_index % 2 == 0 else "g00f-1p5b"
        ] * 20
        assert [row["panel_id"] for row in worker[1::2]] == [
            "g00f-1p5b" if worker_index % 2 == 0 else "g00f-0p5b"
        ] * 20
        for panel_id in CONFIG_SPECS:
            assigned = [row for row in worker if row["panel_id"] == panel_id]
            seed_counts = {
                seed: sum(int(row["seed"]) == seed for row in assigned)
                for seed in CONFIG_SPECS[panel_id]["seeds"]
            }
            case_counts = {case: sum(int(row["cell_index"]) == case for row in assigned) for case in range(8)}
            assert set(seed_counts.values()) == {2}
            assert set(case_counts.values()) <= {2, 3}


def test_guard_is_narrow_and_cannot_be_used_as_a_generic_or_g01_bypass(
    tmp_path: Path,
    frozen_rows: dict[str, list[dict[str, Any]]],
) -> None:
    verified = _verified(tmp_path, frozen_rows)
    good = copy.deepcopy(dict(_case_spec("law_only").config))

    # The frozen generic runner cannot consume the G00-F guard directly.
    with pytest.raises(LaunchGuardError, match="bound G00 gate artifact"):
        legacy_runner.assert_launch_unlocked(good, repo=ROOT)

    original = legacy_runner.assert_launch_unlocked
    with _exact_g00f_launch_scope(verified):
        assert legacy_runner.assert_launch_unlocked(good, repo=ROOT) is None

        null_guard = copy.deepcopy(good)
        null_guard["run"]["launch_guard"] = None
        with pytest.raises(FreezeError, match="guard/config authorization"):
            legacy_runner.assert_launch_unlocked(null_guard, repo=ROOT)

        wrong_cell = copy.deepcopy(good)
        wrong_cell["model"]["name"] = "unregistered/direct-model"
        with pytest.raises(FreezeError, match="outside the exact frozen plan"):
            legacy_runner.assert_launch_unlocked(wrong_cell, repo=ROOT)

        g01 = load_config(G01_CONFIG)
        with pytest.raises(LaunchGuardError, match="bound G00 gate artifact"):
            legacy_runner.assert_launch_unlocked(g01, repo=ROOT)
    assert legacy_runner.assert_launch_unlocked is original


def test_wrong_freeze_worker_duplicate_and_cross_schema_gate_fail_closed(
    tmp_path: Path,
    frozen_rows: dict[str, list[dict[str, Any]]],
) -> None:
    unauthenticated = tmp_path / "unauthenticated-freeze.json"
    _write_json(unauthenticated, {})
    with pytest.raises(FreezeError, match="externally expected SHA-256"):
        verify_freeze(repo=ROOT, freeze_path=unauthenticated, expected_freeze_sha256="0" * 64)

    verified = _verified(tmp_path, frozen_rows)
    for worker_index in (-1, 4, True):
        with pytest.raises(FreezeError, match="worker index"):
            verified.worker_rows(worker_index)

    launch_body = {
        "schema": "goalzendo.g00f_worker_launch_receipt",
        "schema_version": 1,
        "worker_index": 0,
        "freeze_file_sha256": verified.file_sha256,
        "freeze_digest": verified.digest,
        "image": verified.payload["runtime"]["image"],
        "network_volume_id": verified.payload["runtime"]["network_volume_id"],
        "data_center": verified.payload["runtime"]["data_center"],
        "visible_gpu_count": 1,
        "gpu_family": "NVIDIA H100",
        "concurrent_runs": 1,
        "packages": {},
        "source_bundle": {
            "archive_sha256": "a" * 64,
            "manifest_sha256": "b" * 64,
        },
    }
    launch = {**launch_body, "receipt_digest": semantic_digest(launch_body)}
    launch_path = tmp_path / "execution" / "worker-0-launch.json"
    _write_json(launch_path, launch)
    with pytest.raises(FreezeError, match="noncanonical execution-root"):
        verify_launch_receipt(verified=verified, worker_index=1, receipt_path=launch_path)

    receipt = tmp_path / "append-only.json"
    exclusive_json(receipt, {"attempt": 1})
    with pytest.raises(FreezeError, match="already exists"):
        exclusive_json(receipt, {"attempt": 2})
    assert json.loads(receipt.read_text(encoding="utf-8")) == {"attempt": 1}

    assessment_path = tmp_path / "g00f-assessment.json"
    gate_path = tmp_path / "g00f-gate.json"
    prospective = create_preexecution_gate(
        verified=verified,
        assessment_output=assessment_path,
        gate_output=gate_path,
    )
    assert prospective["gate"]["overall_passed"] is False
    assert prospective["gate"]["authorization"] == {
        "g00f_remediation_passed": False,
        "g01_launch_authorized": False,
        "scope": "none",
        "reason": "a_separately_reviewed_digest_bound_g01_bridge_is_required",
    }
    wrong_freeze = VerifiedFreeze(
        repo=verified.repo,
        path=verified.path,
        file_sha256="e" * 64,
        payload={**verified.payload, "freeze_digest": "c" * 64},
        plans=verified.plans,
    )
    with pytest.raises(EvaluationError, match="freeze"):
        verify_gate_artifact(
            gate_path,
            verified=wrong_freeze,
            expected_gate_sha256=sha256_file(gate_path),
        )
    gate = json.loads(gate_path.read_text(encoding="utf-8"))
    gate["gate_digest"] = "0" * 64
    tampered_gate = tmp_path / "tampered-g00f-gate.json"
    _write_json(tampered_gate, gate)
    with pytest.raises(EvaluationError, match="digest"):
        verify_gate_artifact(
            tampered_gate,
            verified=verified,
            expected_gate_sha256=sha256_file(tampered_gate),
        )

    forged = json.loads(gate_path.read_text(encoding="utf-8"))
    forged["authorization"]["scope"] = "g01"
    forged_body = {key: value for key, value in forged.items() if key != "gate_digest"}
    forged["gate_digest"] = semantic_digest(forged_body)
    forged_path = tmp_path / "rehashed-forged-authorization.json"
    _write_json(forged_path, forged)
    with pytest.raises(EvaluationError, match="nonauthorization"):
        verify_gate_artifact(
            forged_path,
            verified=verified,
            expected_gate_sha256=sha256_file(forged_path),
        )

    forged = json.loads(gate_path.read_text(encoding="utf-8"))
    forged["assessment"]["freeze_digest"] = "f" * 64
    assessment_body = {
        key: value for key, value in forged["assessment"].items() if key != "assessment_digest"
    }
    forged["assessment"]["assessment_digest"] = semantic_digest(assessment_body)
    forged_body = {key: value for key, value in forged.items() if key != "gate_digest"}
    forged["gate_digest"] = semantic_digest(forged_body)
    forged_path = tmp_path / "rehashed-forged-assessment-freeze.json"
    _write_json(forged_path, forged)
    with pytest.raises(EvaluationError, match="freeze"):
        verify_gate_artifact(
            forged_path,
            verified=verified,
            expected_gate_sha256=sha256_file(forged_path),
        )

    # A G00-F artifact is a distinct, nonauthorizing schema and must never be
    # accepted by the frozen legacy G00 -> G01 verifier.
    with pytest.raises(LaunchGuardError, match="schema is not recognized"):
        legacy_runner.verify_g00_gate_artifact(
            gate_path,
            config=load_config(G01_CONFIG),
            repo=ROOT,
        )


def test_bootstrap_authenticates_controllers_and_exact_extracted_reexec_bytes(
    tmp_path: Path,
) -> None:
    bootstrap, arguments = _synthetic_bootstrap_bundle(tmp_path / "valid")
    receipt = bootstrap.prepare(arguments)
    assert set(receipt["authenticated_runtime_files"]) == {
        "bootstrap",
        "launcher",
        "watchdog",
    }
    for role, actual in {
        "bootstrap": BOOTSTRAP,
        "launcher": LAUNCHER,
        "watchdog": WATCHDOG,
    }.items():
        binding = receipt["authenticated_runtime_files"][role]
        extracted = arguments.output / binding["frozen_path"]
        assert binding["actual_path"] == str(actual.resolve())
        assert binding["sha256"] == sha256_file(actual)
        assert sha256_file(extracted) == binding["sha256"]
    assert receipt["tar_safety"] == {
        "only_regular_files": True,
        "no_absolute_or_parent_paths": True,
        "no_links": True,
        "exact_member_set": True,
        "exact_modes": True,
        "exact_bytes": True,
    }
    with pytest.raises(bootstrap.BootstrapError, match="reuse"):
        bootstrap.prepare(arguments)

    tampered_launcher = tmp_path / "tampered-launcher.sh"
    tampered_launcher.write_bytes(LAUNCHER.read_bytes() + b"\n# changed after freeze\n")
    _bootstrap, tampered_arguments = _synthetic_bootstrap_bundle(tmp_path / "bad-actual")
    tampered_arguments.actual_launcher = tampered_launcher
    with pytest.raises(bootstrap.BootstrapError, match="launcher bytes"):
        bootstrap.prepare(tampered_arguments)


def test_bootstrap_rejects_archive_controller_bytes_that_differ_from_authenticated_host(
    tmp_path: Path,
) -> None:
    bootstrap, arguments = _synthetic_bootstrap_bundle(
        tmp_path / "cross-binding",
        archive_overrides={"launcher": b"#!/usr/bin/env bash\nexit 99\n"},
    )
    with pytest.raises(bootstrap.BootstrapError, match=r"controller|launcher|archive"):
        bootstrap.prepare(arguments)


def _minimal_builder_repo(destination: Path, builder: ModuleType) -> Path:
    ignored = shutil.ignore_patterns("__pycache__", "*.pyc", "source_manifest.json")
    for relative in ("src/goalzendo", "src/goalzendo_g00f", "configs/goalzendo"):
        shutil.copytree(ROOT / relative, destination / relative, ignore=ignored)
    for relative in (
        "constraints-goalzendo.txt",
        "docs/goalzendo/g00f-execution-freeze.md",
        "docs/goalzendo/protocols/g00f-capability-repair.md",
        "pyproject.toml",
        "reproducibility/goalzendo/g00d-gate-20260811/g00-gate-assessment-derived-pre-fix.json",
        "reproducibility/goalzendo/g00d-gate-20260811/g00e-gate-v3.json",
        "runs/goalzendo/build_g00f_execution_freeze.py",
        "runs/goalzendo/g00f_bundle_bootstrap.py",
        "runs/goalzendo/g00f_watchdog.py",
        "runs/goalzendo/run_g00f_frozen_4h100.sh",
    ):
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / relative, target)
    manifest = builder._source_manifest(destination)
    source_manifest = destination / "src/goalzendo_g00f/source_manifest.json"
    source_manifest.write_bytes(builder._json_bytes(manifest))
    return source_manifest


def _run_builder(repo: Path, output_relative: str) -> subprocess.CompletedProcess[str]:
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(repo / "src")
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    environment["PYTHONNOUSERSITE"] = "1"
    environment["PYTHONHASHSEED"] = "0"
    environment["TZ"] = "UTC"
    environment["LC_ALL"] = "C"
    return subprocess.run(
        [
            sys.executable,
            "-P",
            str(repo / "runs/goalzendo/build_g00f_execution_freeze.py"),
            "--repo",
            str(repo),
            "--output",
            str(repo / output_relative),
        ],
        cwd=repo,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )


def test_freeze_builder_is_byte_reproducible_and_replays_existing_source_manifest(
    tmp_path: Path,
) -> None:
    builder = _load_script(BUILDER, "g00f_builder_reproducibility_test")
    repositories = [_minimal_builder_repo(tmp_path / name, builder) for name in ("clean-a", "clean-b")]
    source_manifest_bytes = [path.read_bytes() for path in repositories]
    for path in repositories:
        os.chmod(path, 0o444)
    roots = [path.parents[2] for path in repositories]
    output_relative = "reproducibility/goalzendo/g00f-execution-freeze-20260811"
    completed = [_run_builder(repo, output_relative) for repo in roots]
    assert [(run.returncode, run.stderr) for run in completed] == [(0, ""), (0, "")]
    assert [path.read_bytes() for path in repositories] == source_manifest_bytes
    assert {os.stat(path).st_mode & 0o777 for path in repositories} == {0o444}

    required_identical = [str(CONFIG_SPECS[panel_id]["plan_path"]) for panel_id in CONFIG_SPECS] + [
        "src/goalzendo_g00f/source_manifest.json",
        f"{output_relative}/g00f-source-bundle-manifest.json",
        f"{output_relative}/g00f-execution-source.tar.gz",
        f"{output_relative}/execution-freeze.json",
        f"{output_relative}/preexecution-assessment.json",
        f"{output_relative}/preexecution-gate.json",
    ]
    for relative in required_identical:
        assert (roots[0] / relative).read_bytes() == (roots[1] / relative).read_bytes()
    freeze_payload = json.loads(
        (roots[0] / output_relative / "execution-freeze.json").read_text(encoding="utf-8")
    )
    assert freeze_payload["runtime"]["runpod_provisioning"] == RUNPOD_PROVISIONING_CONTRACT
    assert freeze_payload["runtime"]["wall_ceiling_seconds"] == 14 * 60 * 60
    assert freeze_payload["runtime"]["h100_hour_ceiling"] == 56.0
    assert RUNPOD_PROVISIONING_CONTRACT["provision_ceiling_seconds"] == 16 * 60 * 60
    assert RUNPOD_PROVISIONING_CONTRACT["maximum_secure_cost_usd"] == 3.29 * 4 * 16
    bundle_manifest = json.loads(
        (roots[0] / output_relative / "g00f-source-bundle-manifest.json").read_text(encoding="utf-8")
    )
    bundled_paths = {row["path"] for row in bundle_manifest["members"]}
    assert {
        "reproducibility/goalzendo/g00d-gate-20260811/g00-gate-assessment-derived-pre-fix.json",
        "reproducibility/goalzendo/g00d-gate-20260811/g00e-gate-v3.json",
    } <= bundled_paths

    # Exercise the actual authenticated extraction, then the first two inner
    # CLI phases from the extracted repository.  This catches bundle members
    # that outer verification can see but clean re-execution cannot.
    built_root = roots[0]
    built_output = built_root / output_relative
    freeze_path = built_output / "execution-freeze.json"
    freeze_sha = sha256_file(freeze_path)
    bootstrap = _load_script(
        built_root / "runs/goalzendo/g00f_bundle_bootstrap.py",
        "g00f_real_inner_bootstrap_test",
    )
    execution_root = tmp_path / "real-inner-execution"
    extracted_root = execution_root / "frozen-source"
    bootstrap.prepare(
        argparse.Namespace(
            freeze=freeze_path,
            expected_freeze_sha256=freeze_sha,
            archive=built_output / "g00f-execution-source.tar.gz",
            manifest=built_output / "g00f-source-bundle-manifest.json",
            output=extracted_root,
            receipt=execution_root / "source-bundle-receipt.json",
            actual_bootstrap=built_root / "runs/goalzendo/g00f_bundle_bootstrap.py",
            actual_launcher=built_root / "runs/goalzendo/run_g00f_frozen_4h100.sh",
            actual_watchdog=built_root / "runs/goalzendo/g00f_watchdog.py",
        )
    )
    inner_environment = dict(os.environ)
    inner_environment.update(
        {
            "PYTHONPATH": str(extracted_root / "src"),
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONNOUSERSITE": "1",
        }
    )
    common = [
        "--repo",
        str(extracted_root),
        "--freeze",
        str(freeze_path),
        "--expected-freeze-sha256",
        freeze_sha,
    ]

    def inner_cli(*arguments: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [sys.executable, "-P", "-m", "goalzendo_g00f.cli", *arguments],
            cwd=extracted_root,
            env=inner_environment,
            check=False,
            capture_output=True,
            text=True,
            timeout=30,
        )

    inner_verify = inner_cli("verify-freeze", *common)
    assert (inner_verify.returncode, inner_verify.stderr) == (0, "")
    external = tmp_path / "real-inner-provision"
    external.mkdir()
    create_raw = external / "runpod-create-response.json"
    get_raw = external / "runpod-api-response.json"
    create_raw.write_text('{"id":"pod-inner","stage":"create"}\n', encoding="utf-8")
    get_raw.write_text('{"id":"pod-inner","stage":"get"}\n', encoding="utf-8")
    external_receipt = external / "runpod-provision-receipt.json"
    created = inner_cli(
        "create-provision-receipt",
        *common,
        "--raw-create-response",
        str(create_raw),
        "--raw-api-response",
        str(get_raw),
        "--pod-id",
        "pod-inner",
        "--created-at-utc",
        PROVISION_CREATED_AT,
        "--terminate-after-utc",
        PROVISION_TERMINATE_AFTER,
        "--secure-price-usd-per-gpu-hour",
        "3.29",
        "--stock-label",
        "Low",
        "--output",
        str(external_receipt),
    )
    assert (created.returncode, created.stderr) == (0, "")
    bound = inner_cli(
        "bind-provision-receipt",
        *common,
        "--input",
        str(external_receipt),
        "--output",
        str(execution_root / "runpod-provision-receipt.json"),
        "--expected-provision-receipt-sha256",
        sha256_file(external_receipt),
        "--expected-pod-id",
        "pod-inner",
    )
    assert (bound.returncode, bound.stderr) == (0, "")
    output_names = sorted(path.name for path in (roots[0] / output_relative).iterdir())
    assert output_names == sorted(path.name for path in (roots[1] / output_relative).iterdir())
    assert all(
        (roots[0] / output_relative / name).read_bytes() == (roots[1] / output_relative / name).read_bytes()
        for name in output_names
    )

    source_manifest = _minimal_builder_repo(tmp_path / "source-drift", builder)
    source_manifest_before = source_manifest.read_bytes()
    drift_root = source_manifest.parents[2]
    drifted_source = drift_root / "src/goalzendo_g00f/evaluator.py"
    drifted_source.write_bytes(drifted_source.read_bytes() + b"\n# source drift\n")
    rejected = _run_builder(drift_root, output_relative)
    assert rejected.returncode == 2
    assert "existing additive source manifest differs from recomputed bytes" in rejected.stdout
    assert source_manifest.read_bytes() == source_manifest_before
    assert not (drift_root / output_relative).exists()


def test_launcher_reexecutes_only_the_extracted_authenticated_entrypoint() -> None:
    source = LAUNCHER.read_text(encoding="utf-8")
    outer_phase = source.index('if [[ "${G00F_FROZEN_REEXEC:-0}" != "1" ]]')
    bootstrap_call = source.index('--actual-bootstrap "$BOOTSTRAP"', outer_phase)
    phase_export = source.index("export G00F_FROZEN_REEXEC=1", bootstrap_call)
    reexec = source.index(
        'exec "$EXTRACTED_ROOT/runs/goalzendo/run_g00f_frozen_4h100.sh"',
        phase_export,
    )
    path_check = source.index('if [[ "$SCRIPT_PATH" != "$EXPECTED_SCRIPT" ]]', reexec)
    project_imports = source.index('export PYTHONPATH="$ROOT/src"', path_check)
    assert outer_phase < bootstrap_call < phase_export < reexec < path_check < project_imports
    bind_provision = source.index('"${CLI[@]}" bind-provision-receipt', project_imports)
    model_receipt = source.index('"${CLI[@]}" model-receipt', bind_provision)
    model_integration = source.index('"${CLI[@]}" model-integration-audit', model_receipt)
    initialize_ledger = source.index('"${CLI[@]}" initialize-ledger', model_integration)
    watchdog_start = source.index('WATCHDOG_PID="$!"', initialize_ledger)
    watchdog_ready = source.index('if [[ -f "$WATCHDOG_STARTED" ]]', watchdog_start)
    launch_receipt = source.index('"${CLI[@]}" launch-receipt', initialize_ledger)
    watchdog_wait_set = source.index('WAIT_PIDS=("${ACTIVE_PIDS[@]}" "$WATCHDOG_PID")')
    watchdog_failure = source.index(
        "reconcile_failure watchdog_process_exit WatchdogProcessExit",
        watchdog_wait_set,
    )
    watchdog_normal_stop = source.index('[[ ! -f "$WATCHDOG_NORMAL_STOP" ]]', watchdog_failure)
    assert (
        bind_provision
        < model_receipt
        < model_integration
        < initialize_ledger
        < watchdog_start
        < watchdog_ready
        < launch_receipt
        < watchdog_wait_set
        < watchdog_failure
        < watchdog_normal_stop
    )
    parsed = g00f_cli_module.build_parser().parse_args(
        [
            "reconcile-failure",
            "--repo",
            str(ROOT),
            "--freeze",
            "freeze.json",
            "--expected-freeze-sha256",
            "f" * 64,
            "--ledger-root",
            "ledger",
            "--error-type",
            "WatchdogProcessExit",
            "--trigger",
            "watchdog_process_exit",
            "--cancel-receipt",
            "cancel.json",
        ]
    )
    assert parsed.trigger == "watchdog_process_exit"


def test_evaluator_rejects_cross_execution_ledger_and_worker_mix(
    tmp_path: Path,
    frozen_rows: dict[str, list[dict[str, Any]]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    execution_a = "00000000-0000-4000-8000-00000000000a"
    execution_b = "00000000-0000-4000-8000-00000000000b"
    runtime = {
        "execution_uuid": execution_a,
        "execution_root": "/workspace/status-goalzendo/g00f-executions/" + execution_a,
    }
    attempt_audit = {
        "ledger": {
            "execution_uuid": execution_b,
            "ledger_root": (
                "/workspace/status-goalzendo/g00f-executions/" + execution_b + "/itt-ledger/" + execution_b
            ),
        }
    }
    with pytest.raises(EvaluationError, match="different executions"):
        evaluator_module._require_same_execution(attempt_audit, runtime)

    verified = _verified(tmp_path, frozen_rows)
    worker_result = tmp_path / "worker-0-result.json"
    result_body = {
        "schema": evaluator_module.WORKER_RESULT_SCHEMA,
        "schema_version": evaluator_module.WORKER_RESULT_SCHEMA_VERSION,
        "execution_uuid": execution_b,
        "worker_index": 0,
        "freeze_file_sha256": verified.file_sha256,
        "freeze_digest": verified.digest,
        "state": "complete",
        "exit_code": 0,
        "error_type": None,
        "planned_runs": 40,
        "completed_runs": 40,
        "failed_runs": 0,
        "observed_outcome_rows": 40,
        "completed_monotonic_ns": 101,
        "wall_seconds": 1e-9,
        "no_reassignment": True,
        "outcome_metrics_read": False,
        "predictions_read": False,
        "launch_receipt": {
            "path": str(tmp_path / "worker-0-launch.json"),
            "file_sha256": "a" * 64,
            "receipt_digest": "b" * 64,
        },
        "g01_launch_authorized": False,
    }
    _write_json(
        worker_result,
        {**result_body, "receipt_digest": semantic_digest(result_body)},
    )
    monkeypatch.setattr(
        evaluator_module,
        "verify_launch_receipt",
        lambda **_kwargs: {
            "file_sha256": "a" * 64,
            "receipt_digest": "b" * 64,
            "execution_uuid": execution_a,
        },
    )
    with pytest.raises(EvaluationError, match="execution UUID"):
        evaluator_module._verify_worker_result_receipts(
            verified,
            {index: worker_result for index in range(4)},
            expected_provision_receipt_sha256="c" * 64,
            expected_pod_id="pod-g00f-test",
        )

    attempt_audit["ledger"] = {
        "execution_uuid": execution_a,
        "ledger_root": runtime["execution_root"] + "/itt-ledger/" + execution_b,
    }
    with pytest.raises(EvaluationError, match="different executions"):
        evaluator_module._require_same_execution(attempt_audit, runtime)


def test_external_runpod_provision_receipt_is_exactly_pinned_and_bound_once(
    tmp_path: Path,
    frozen_rows: dict[str, list[dict[str, Any]]],
) -> None:
    verified = _verified(tmp_path, frozen_rows)
    external, expected_sha, _raw = _external_provision_receipt(tmp_path / "external", verified)

    replay = verify_runpod_provision_receipt(
        verified=verified,
        receipt_path=external,
        expected_receipt_sha256=expected_sha,
        expected_pod_id="pod-g00f-test",
        require_bound_copy=False,
    )
    assert replay["file_sha256"] == expected_sha
    assert replay["pod_id"] == "pod-g00f-test"
    assert replay["evidence_boundaries"] == RUNPOD_PROVISION_EVIDENCE
    assert replay["operational_market_snapshot"] == {
        "observed_at_utc": PROVISION_CREATED_AT,
        "secure_price_usd_per_gpu_hour": 3.29,
        "stock_label": "Low",
        "scientific_identity": False,
    }
    assert replay["api_response"]["capture_command"] == (
        "runpodctl pod get pod-g00f-test --include-machine --include-network-volume -o json"
    )

    with pytest.raises(FreezeError, match="external SHA-256 binding"):
        verify_runpod_provision_receipt(
            verified=verified,
            receipt_path=external,
            expected_receipt_sha256="0" * 64,
            expected_pod_id="pod-g00f-test",
            require_bound_copy=False,
        )
    with pytest.raises(FreezeError, match="allocation contract"):
        verify_runpod_provision_receipt(
            verified=verified,
            receipt_path=external,
            expected_receipt_sha256=expected_sha,
            expected_pod_id="pod-forged",
            require_bound_copy=False,
        )

    bound_path = tmp_path / "execution" / "runpod-provision-receipt.json"
    bound = bind_runpod_provision_receipt(
        verified=verified,
        input_path=external,
        output_path=bound_path,
        expected_receipt_sha256=expected_sha,
        expected_pod_id="pod-g00f-test",
    )
    assert bound["file_sha256"] == expected_sha
    assert Path(bound["path"]) == bound_path.resolve()
    assert os.stat(bound_path).st_mode & 0o777 == 0o400
    assert os.stat(bound_path.with_name("runpod-api-response.json")).st_mode & 0o777 == 0o400
    with pytest.raises(FreezeError, match="already exists"):
        bind_runpod_provision_receipt(
            verified=verified,
            input_path=external,
            output_path=bound_path,
            expected_receipt_sha256=expected_sha,
            expected_pod_id="pod-g00f-test",
        )

    bound_raw = bound_path.with_name("runpod-api-response.json")
    os.chmod(bound_raw, 0o600)
    bound_raw.write_text('{"id":"forged-after-binding"}\n', encoding="utf-8")
    os.chmod(bound_raw, 0o400)
    with pytest.raises(FreezeError, match="raw Runpod API response bytes"):
        verify_runpod_provision_receipt(
            verified=verified,
            receipt_path=bound_path,
            expected_receipt_sha256=expected_sha,
            expected_pod_id="pod-g00f-test",
        )


def test_runpod_market_snapshot_is_operational_not_scientific_identity(
    tmp_path: Path,
    frozen_rows: dict[str, list[dict[str, Any]]],
) -> None:
    verified = _verified(tmp_path, frozen_rows)
    external, expected_sha, _raw = _external_provision_receipt(
        tmp_path / "external-market-forged",
        verified,
        changes={
            "operational_market_snapshot": {
                "observed_at_utc": PROVISION_CREATED_AT,
                "secure_price_usd_per_gpu_hour": 3.29,
                "stock_label": "Low",
                "scientific_identity": True,
            }
        },
    )
    with pytest.raises(FreezeError, match="price/stock facts changed type or role"):
        verify_runpod_provision_receipt(
            verified=verified,
            receipt_path=external,
            expected_receipt_sha256=expected_sha,
            expected_pod_id="pod-g00f-test",
            require_bound_copy=False,
        )


def test_canonical_runpod_create_command_is_secure_exact_and_absolute() -> None:
    command_builder = getattr(freeze_module, "canonical_runpod_create_command", None)
    assert callable(command_builder), "freeze must export the canonical Runpod create command"
    assert command_builder(terminate_after_utc=PROVISION_TERMINATE_AFTER) == (
        _expected_runpod_create_command(PROVISION_TERMINATE_AFTER)
    )
    for invalid in (
        "16h",
        "2026-08-12T01:30:00+00:00",
        "2026-08-12 01:30:00Z",
        "2026-08-12T01:30:00Z ",
    ):
        with pytest.raises(FreezeError, match=r"absolute|UTC|terminate"):
            command_builder(terminate_after_utc=invalid)


def test_provision_receipt_binds_create_then_get_and_a_distinct_16h_cost_guard(
    tmp_path: Path,
    frozen_rows: dict[str, list[dict[str, Any]]],
) -> None:
    verified = _verified(tmp_path, frozen_rows)
    directory = tmp_path / "external-provision-sequence"
    directory.mkdir()
    create_raw = directory / "runpod-create-response.json"
    get_raw = directory / "runpod-api-response.json"
    create_raw.write_text('{"id":"pod-g00f-test","stage":"create"}\n', encoding="utf-8")
    get_raw.write_text('{"id":"pod-g00f-test","stage":"get"}\n', encoding="utf-8")
    receipt_path = directory / "runpod-provision-receipt.json"
    creator: Any = freeze_module.create_runpod_provision_receipt
    required_parameters = {"raw_create_response_path", "terminate_after_utc"}
    assert required_parameters <= set(inspect.signature(creator).parameters), (
        "provision receipt creator must bind the raw create response and absolute deadline"
    )
    creator(
        verified=verified,
        raw_create_response_path=create_raw,
        raw_api_response_path=get_raw,
        pod_id="pod-g00f-test",
        created_at_utc=PROVISION_CREATED_AT,
        terminate_after_utc=PROVISION_TERMINATE_AFTER,
        secure_price_usd_per_gpu_hour=3.29,
        stock_label="Low",
        output_path=receipt_path,
    )
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert receipt["provisioning"] == {
        "capture_order": ["create", "get"],
        "create": {
            "command_argv": list(_expected_runpod_create_command(PROVISION_TERMINATE_AFTER)),
            "file_name": "runpod-create-response.json",
            "sha256": sha256_file(create_raw),
        },
        "get": {
            "command_argv": [
                "runpodctl",
                "pod",
                "get",
                "pod-g00f-test",
                "--include-machine",
                "--include-network-volume",
                "-o",
                "json",
            ],
            "file_name": "runpod-api-response.json",
            "sha256": sha256_file(get_raw),
        },
        "terminate_after_utc": PROVISION_TERMINATE_AFTER,
        "provision_ceiling_seconds": 16 * 60 * 60,
        "maximum_secure_cost_usd": 210.56,
    }

    expected_sha = sha256_file(receipt_path)
    bound_path = tmp_path / "execution" / "runpod-provision-receipt.json"
    provision = bind_runpod_provision_receipt(
        verified=verified,
        input_path=receipt_path,
        output_path=bound_path,
        expected_receipt_sha256=expected_sha,
        expected_pod_id="pod-g00f-test",
    )
    assert provision["provisioning"] == receipt["provisioning"]
    execution_uuid = "00000000-0000-4000-8000-000000000007"
    ledger_root = tmp_path / "execution" / "itt-ledger" / execution_uuid
    model_audits = _create_test_model_audits(tmp_path / "execution", verified)
    initialize_attempt_ledger(
        verified=verified,
        ledger_root=ledger_root,
        execution_uuid=execution_uuid,
        provision_receipt_path=bound_path,
        expected_provision_receipt_sha256=expected_sha,
        expected_pod_id="pod-g00f-test",
        model_integration_audit_paths=model_audits,
    )
    ledger = verify_attempt_ledger(verified=verified, ledger_root=ledger_root)
    assert ledger["budget_start"]["wall_ceiling_seconds"] == WALL_CEILING_SECONDS == 14 * 60 * 60
    assert ledger["runpod_provision"]["provisioning"]["provision_ceiling_seconds"] == (16 * 60 * 60)
    assert ledger["runpod_provision"]["provisioning"]["terminate_after_utc"] == (PROVISION_TERMINATE_AFTER)

    forged = copy.deepcopy(receipt)
    forged["provisioning"]["terminate_after_utc"] = "2026-08-12T01:30:01Z"
    create_argv = forged["provisioning"]["create"]["command_argv"]
    create_argv[create_argv.index("--terminate-after") + 1] = "2026-08-12T01:30:01Z"
    forged_body = {key: value for key, value in forged.items() if key != "receipt_digest"}
    forged["receipt_digest"] = semantic_digest(forged_body)
    _write_json(receipt_path, forged)
    with pytest.raises(FreezeError, match=r"termination|ceiling|16-hour"):
        verify_runpod_provision_receipt(
            verified=verified,
            receipt_path=receipt_path,
            expected_receipt_sha256=sha256_file(receipt_path),
            expected_pod_id="pod-g00f-test",
            require_bound_copy=False,
        )


def test_model_materializer_receipt_and_leaf_tamper_fail_closed(
    tmp_path: Path,
    frozen_rows: dict[str, list[dict[str, Any]]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    verified = _verified(tmp_path, frozen_rows)
    panel_id = "g00f-0p5b"
    frozen_leaf = verified.payload["configurations"][panel_id]["model_snapshot"]["leaf_files"][0]
    monkeypatch.setattr(
        freeze_module,
        "MODEL_LEAF_FILES",
        {**freeze_module.MODEL_LEAF_FILES, panel_id: (frozen_leaf,)},
    )
    downloaded = tmp_path / "downloaded"
    downloaded.mkdir()
    source = downloaded / str(frozen_leaf["path"])
    source.write_bytes(f"fixture-{panel_id}\n".encode())
    observed_calls: list[dict[str, Any]] = []

    def fake_snapshot_download(**kwargs: Any) -> str:
        observed_calls.append(kwargs)
        return str(downloaded)

    fake_hub = ModuleType("huggingface_hub")
    fake_hub.snapshot_download = fake_snapshot_download  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake_hub)
    target = tmp_path / "materialized"
    result = materialize_model_snapshot(
        verified=verified,
        panel_id=panel_id,
        output_root=target,
    )
    assert result["fresh_regular_files_no_links"] is True
    assert observed_calls == [
        {
            "repo_id": CONFIG_SPECS[panel_id]["model"],
            "revision": CONFIG_SPECS[panel_id]["revision"],
            "allow_patterns": [str(frozen_leaf["path"])],
            "cache_dir": observed_calls[0]["cache_dir"],
            "token": False,
            "max_workers": 4,
        }
    ]
    leaf = target / str(frozen_leaf["path"])
    assert stat.S_IMODE(target.stat().st_mode) == 0o555
    assert stat.S_IMODE(leaf.stat().st_mode) == 0o444
    assert not leaf.is_symlink() and leaf.stat().st_nlink == 1
    receipt_path = tmp_path / "model-receipt.json"
    created = create_model_snapshot_receipt(
        verified=verified,
        panel_id=panel_id,
        snapshot_root=target,
        output=receipt_path,
    )
    assert (
        verify_model_snapshot_receipt(
            verified=verified,
            panel_id=panel_id,
            receipt_path=receipt_path,
        )["file_sha256"]
        == created["file_sha256"]
    )
    os.chmod(target, 0o755)
    extra_leaf = target / "extra-runtime-leaf.json"
    extra_leaf.write_text("{}\n", encoding="utf-8")
    os.chmod(extra_leaf, 0o444)
    os.chmod(target, 0o555)
    with pytest.raises(FreezeError, match="tree inventory"):
        verify_model_snapshot_receipt(
            verified=verified,
            panel_id=panel_id,
            receipt_path=receipt_path,
        )
    os.chmod(target, 0o755)
    extra_leaf.unlink()
    os.chmod(target, 0o555)
    os.chmod(leaf, 0o644)
    leaf.write_bytes(b"tampered\n")
    os.chmod(leaf, 0o444)
    with pytest.raises(FreezeError, match="no longer replays"):
        verify_model_snapshot_receipt(
            verified=verified,
            panel_id=panel_id,
            receipt_path=receipt_path,
        )

    bad_download = tmp_path / "bad-downloaded"
    bad_download.mkdir()
    (bad_download / str(frozen_leaf["path"])).write_bytes(b"wrong\n")
    fake_hub.snapshot_download = lambda **_kwargs: str(bad_download)  # type: ignore[attr-defined]
    with pytest.raises(FreezeError, match="downloaded frozen model leaf changed"):
        materialize_model_snapshot(
            verified=verified,
            panel_id=panel_id,
            output_root=tmp_path / "must-not-materialize",
        )
    assert not (tmp_path / "must-not-materialize").exists()


def test_provision_observation_skew_is_allowed_but_truncated_itt_window_is_not(
    tmp_path: Path,
    frozen_rows: dict[str, list[dict[str, Any]]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    verified = _verified(tmp_path, frozen_rows)
    external = tmp_path / "external-skew"
    external.mkdir()
    create_raw = external / "runpod-create-response.json"
    get_raw = external / "runpod-api-response.json"
    create_raw.write_text('{"id":"pod-g00f-test"}\n', encoding="utf-8")
    get_raw.write_text('{"id":"pod-g00f-test"}\n', encoding="utf-8")
    receipt_path = external / "runpod-provision-receipt.json"
    create_runpod = freeze_module.create_runpod_provision_receipt
    created = create_runpod(
        verified=verified,
        raw_create_response_path=create_raw,
        raw_api_response_path=get_raw,
        pod_id="pod-g00f-test",
        created_at_utc="2026-08-11T09:30:05Z",
        terminate_after_utc=PROVISION_TERMINATE_AFTER,
        secure_price_usd_per_gpu_hour=3.29,
        stock_label="Low",
        output_path=receipt_path,
    )
    assert created["provisioning"]["provision_ceiling_seconds"] == 16 * 60 * 60

    expected_sha = sha256_file(receipt_path)
    execution_root = tmp_path / "execution-skew"
    bound = execution_root / "runpod-provision-receipt.json"
    bind_runpod_provision_receipt(
        verified=verified,
        input_path=receipt_path,
        output_path=bound,
        expected_receipt_sha256=expected_sha,
        expected_pod_id="pod-g00f-test",
    )
    audits = _create_test_model_audits(execution_root, verified)
    termination_ns = (
        int(freeze_module._canonical_utc(PROVISION_TERMINATE_AFTER, "test termination").timestamp())
        * 1_000_000_000
    )
    required_ns = (WALL_CEILING_SECONDS + freeze_module.PROVISION_WATCHDOG_GRACE_SECONDS) * 1_000_000_000
    monkeypatch.setattr(time, "time_ns", lambda: termination_ns - required_ns + 1)
    execution_uuid = "00000000-0000-4000-8000-000000000008"
    ledger_root = execution_root / "itt-ledger" / execution_uuid
    with pytest.raises(FreezeError, match="14-hour budget plus watchdog grace"):
        initialize_attempt_ledger(
            verified=verified,
            ledger_root=ledger_root,
            execution_uuid=execution_uuid,
            provision_receipt_path=bound,
            expected_provision_receipt_sha256=expected_sha,
            expected_pod_id="pod-g00f-test",
            model_integration_audit_paths=audits,
        )
    assert not ledger_root.exists()


def test_preledger_model_integration_audits_replay_thresholds_and_failures(
    tmp_path: Path,
    frozen_rows: dict[str, list[dict[str, Any]]],
) -> None:
    verified = _verified(tmp_path, frozen_rows)
    execution_root = tmp_path / "execution"
    audits = _create_test_model_audits(execution_root, verified)
    for panel_id, path in audits.items():
        replay = verify_model_integration_audit(
            verified=verified,
            panel_id=panel_id,
            audit_path=path,
        )
        assert replay["panel_id"] == panel_id

    bf16_report = _fake_model_integration_report(
        "g00f-0p5b",
        Path(
            json.loads(audits["g00f-0p5b"].read_text(encoding="utf-8"))["model_snapshot_receipt"][
                "snapshot_root"
            ]
        ),
    )
    bf16_report["scores"]["normalized_log_scores"] = [
        [math.log(0.5), math.log(0.5)],
        [math.log(0.5), math.log(0.5)],
    ]
    bf16_report["scores"]["probabilities"] = [
        [0.5, 0.498046875],
        [0.498046875, 0.5],
    ]
    bf16_report["scores"]["maximum_normalization_error"] = 0.001953125
    freeze_module._validate_model_integration_report(
        panel_id="g00f-0p5b",
        report=bf16_report,
        snapshot_root=bf16_report["model"]["tokenizer_name_or_path"],
    )

    target = audits["g00f-0p5b"]
    forged = json.loads(target.read_text(encoding="utf-8"))
    forged["report"]["scores"]["maximum_score_swap_error"] = 0.002001
    forged["report_digest"] = semantic_digest(forged["report"])
    forged_body = {key: value for key, value in forged.items() if key != "audit_digest"}
    forged["audit_digest"] = semantic_digest(forged_body)
    _write_json(target, forged)
    with pytest.raises(FreezeError, match="exceeds 2e-3"):
        verify_model_integration_audit(
            verified=verified,
            panel_id="g00f-0p5b",
            audit_path=target,
        )

    failed_root = tmp_path / "failed-execution"
    panel_id = "g00f-0p5b"
    snapshot_root = tmp_path / "failed-snapshot"
    snapshot_root.mkdir()
    leaf = snapshot_root / "fixture-model.bin"
    leaf.write_bytes(f"fixture-{panel_id}\n".encode())
    os.chmod(leaf, 0o444)
    os.chmod(snapshot_root, 0o555)
    model_receipt = failed_root / "model-receipt-0p5b.json"
    create_model_snapshot_receipt(
        verified=verified,
        panel_id=panel_id,
        snapshot_root=snapshot_root,
        output=model_receipt,
    )
    previous_hub = os.environ.get("HF_HUB_OFFLINE")
    previous_transformers = os.environ.get("TRANSFORMERS_OFFLINE")
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    try:
        with pytest.raises(FreezeError, match="pretraining model integration audit failed"):
            create_model_integration_audit(
                verified=verified,
                panel_id=panel_id,
                model_receipt_path=model_receipt,
                output=failed_root / "model-integration-audit-0p5b.json",
                integration_runner=lambda *_args, **_kwargs: (_ for _ in ()).throw(
                    AssertionError("swap failure")
                ),
            )
    finally:
        if previous_hub is None:
            os.environ.pop("HF_HUB_OFFLINE", None)
        else:
            os.environ["HF_HUB_OFFLINE"] = previous_hub
        if previous_transformers is None:
            os.environ.pop("TRANSFORMERS_OFFLINE", None)
        else:
            os.environ["TRANSFORMERS_OFFLINE"] = previous_transformers
    assert not (failed_root / "model-integration-audit-0p5b.json").exists()


@pytest.mark.parametrize(
    ("changes", "case"),
    [
        ({"image": "forged/image:latest"}, "image"),
        ({"gpu_count": 3}, "gpu-count"),
        (
            {
                "gpu_catalog": {
                    "display_name": "A100",
                    "gpu_id": "NVIDIA A100 80GB PCIe",
                }
            },
            "gpu-catalog",
        ),
        ({"data_center": "EU-RO-1"}, "data-center"),
        ({"network_volume_id": "forged-volume"}, "volume"),
        ({"network_volume_mount": "/forged"}, "mount"),
        (
            {
                "evidence_boundaries": {
                    **RUNPOD_PROVISION_EVIDENCE,
                    "network_volume_mount": "runpod_api",
                }
            },
            "evidence-boundary",
        ),
        ({"outcomes_seen": True}, "outcomes-seen"),
        ({"g01_launch_authorized": True}, "g01"),
    ],
    ids=lambda value: value if isinstance(value, str) else None,
)
def test_runpod_provision_rejects_semantically_rehashed_runtime_forgery(
    tmp_path: Path,
    frozen_rows: dict[str, list[dict[str, Any]]],
    changes: dict[str, Any],
    case: str,
) -> None:
    del case
    verified = _verified(tmp_path, frozen_rows)
    external, expected_sha, _raw = _external_provision_receipt(
        tmp_path / "external-forged",
        verified,
        changes=changes,
    )
    with pytest.raises(FreezeError, match="allocation contract"):
        verify_runpod_provision_receipt(
            verified=verified,
            receipt_path=external,
            expected_receipt_sha256=expected_sha,
            expected_pod_id="pod-g00f-test",
            require_bound_copy=False,
        )


def test_provision_binding_replays_through_ledger_evaluator_and_root_inventory(
    tmp_path: Path,
    frozen_rows: dict[str, list[dict[str, Any]]],
) -> None:
    verified = _verified(tmp_path, frozen_rows)
    execution_uuid = "00000000-0000-4000-8000-000000000004"
    ledger_root, provision, _expected_sha, _pod_id = _initialize_test_ledger(
        tmp_path,
        verified,
        execution_uuid,
    )

    ledger = verify_attempt_ledger(verified=verified, ledger_root=ledger_root)
    assert ledger["runpod_provision"] == provision
    audit = _audit_attempt_ledger(verified=verified, ledger_root=ledger_root)
    assert audit["ledger"]["runpod_provision"] == provision
    assert set(audit["states"].values()) == {"preallocated_not_started"}
    assert len(audit["rows"]) == 160

    _write_json(ledger_root / "unbound-extra.json", {"forged": True})
    with pytest.raises(FreezeError, match="unexpected entry"):
        verify_attempt_ledger(verified=verified, ledger_root=ledger_root)


def test_launch_receipt_cross_binds_exact_provision_and_runtime(
    tmp_path: Path,
    frozen_rows: dict[str, list[dict[str, Any]]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bootstrap, bootstrap_args = _synthetic_bootstrap_bundle(tmp_path / "bundle_fixture")
    execution_root = tmp_path / "execution"
    bootstrap_args.output = execution_root / "frozen-source"
    bootstrap_args.receipt = execution_root / "source-bundle-receipt.json"
    bootstrap.prepare(bootstrap_args)
    synthetic_freeze = json.loads(bootstrap_args.freeze.read_text(encoding="utf-8"))
    base = _verified(tmp_path, frozen_rows)
    payload: dict[str, Any] = copy.deepcopy(dict(base.payload))
    payload["freeze_digest"] = synthetic_freeze["freeze_digest"]
    payload["source_bundle"] = synthetic_freeze["source_bundle"]
    verified = VerifiedFreeze(
        repo=bootstrap_args.output.resolve(),
        path=bootstrap_args.freeze,
        file_sha256=bootstrap_args.expected_freeze_sha256,
        payload=payload,
        plans=base.plans,
    )
    execution_uuid = "00000000-0000-4000-8000-000000000005"
    ledger_root, provision, expected_sha, pod_id = _initialize_test_ledger(
        tmp_path,
        verified,
        execution_uuid,
    )
    bundle_path = bootstrap_args.receipt
    accelerator = {
        "torch_available": True,
        "cuda_available": True,
        "torch_version": "2.8.0+cu128",
        "cuda_runtime": "12.8",
        "cuda_devices": [{"index": 0, "name": "NVIDIA H100 SXM"}],
    }
    monkeypatch.setattr(g00f_cli_module, "_package_versions", lambda _names: {})
    monkeypatch.setattr(g00f_cli_module, "package_manifest", lambda: {"fixture": "1.0"})
    monkeypatch.setattr(g00f_cli_module, "accelerator_metadata", lambda: accelerator)
    monkeypatch.setattr(
        g00f_cli_module,
        "_actual_visible_gpu",
        lambda: {
            "cuda_visible_devices": "GPU-test-0",
            "gpu_name": "NVIDIA H100 SXM",
            "gpu_uuid": "GPU-test-0",
            "torch_gpu_name": "NVIDIA H100 SXM",
        },
    )
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")
    launch_path = execution_root / "worker-0-launch.json"
    args = argparse.Namespace(
        worker_index=0,
        ledger_root=ledger_root,
        bundle_receipt=bundle_path,
        provision_receipt=execution_root / "runpod-provision-receipt.json",
        expected_provision_receipt_sha256=expected_sha,
        expected_pod_id=pod_id,
        image=verified.payload["runtime"]["image"],
        network_volume_id=verified.payload["runtime"]["network_volume_id"],
        data_center=verified.payload["runtime"]["data_center"],
        gpu_name="NVIDIA H100 SXM",
        gpu_uuid="GPU-test-0",
        output=launch_path,
    )
    g00f_cli_module._create_launch_receipt(args, verified)
    replay = verify_launch_receipt(
        verified=verified,
        worker_index=0,
        receipt_path=launch_path,
        expected_provision_receipt_sha256=expected_sha,
        expected_pod_id=pod_id,
    )
    assert replay["runpod_provision"] == provision

    original = json.loads(launch_path.read_text(encoding="utf-8"))
    with pytest.raises(FreezeError, match="externally pinned Runpod allocation"):
        verify_launch_receipt(
            verified=verified,
            worker_index=0,
            receipt_path=launch_path,
            expected_provision_receipt_sha256="0" * 64,
            expected_pod_id=pod_id,
        )
    with pytest.raises(FreezeError, match="externally pinned Runpod allocation"):
        verify_launch_receipt(
            verified=verified,
            worker_index=0,
            receipt_path=launch_path,
            expected_provision_receipt_sha256=expected_sha,
            expected_pod_id="pod-forged",
        )

    forged_runtime = copy.deepcopy(original)
    forged_runtime["network_volume_mount"] = "/forged"
    forged_body = {key: value for key, value in forged_runtime.items() if key != "receipt_digest"}
    forged_runtime["receipt_digest"] = semantic_digest(forged_body)
    _write_json(launch_path, forged_runtime)
    with pytest.raises(FreezeError, match="exact runtime/freeze"):
        verify_launch_receipt(verified=verified, worker_index=0, receipt_path=launch_path)

    forged_accelerator = copy.deepcopy(original)
    accelerator_inventory = forged_accelerator["runtime_environment"]["accelerator"]
    accelerator_inventory["cuda_runtime"] = "12.7"
    forged_accelerator["runtime_environment"]["accelerator_digest"] = semantic_digest(accelerator_inventory)
    forged_body = {key: value for key, value in forged_accelerator.items() if key != "receipt_digest"}
    forged_accelerator["receipt_digest"] = semantic_digest(forged_body)
    _write_json(launch_path, forged_accelerator)
    with pytest.raises(FreezeError, match="runtime inventory digest"):
        verify_launch_receipt(verified=verified, worker_index=0, receipt_path=launch_path)

    forged = copy.deepcopy(original)
    forged["runpod_provision"]["pod_id"] = "pod-forged"
    forged_body = {key: value for key, value in forged.items() if key != "receipt_digest"}
    forged["receipt_digest"] = semantic_digest(forged_body)
    _write_json(launch_path, forged)
    with pytest.raises(FreezeError, match=r"provision|allocation|binding"):
        verify_launch_receipt(
            verified=verified,
            worker_index=0,
            receipt_path=launch_path,
            expected_provision_receipt_sha256=expected_sha,
            expected_pod_id=pod_id,
        )


def test_monotonic_deadline_refuses_early_then_reconciles_only_unstarted_keys(
    tmp_path: Path,
    frozen_rows: dict[str, list[dict[str, Any]]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    verified = _verified(tmp_path, frozen_rows)
    execution_uuid = "00000000-0000-4000-8000-000000000001"
    clock = [10_000]
    monkeypatch.setattr(time, "monotonic_ns", lambda: clock[0])
    ledger_root, _provision, _provision_sha, _pod_id = _initialize_test_ledger(
        tmp_path / "deadline",
        verified,
        execution_uuid,
    )

    started_row = verified.all_rows[0]
    start = _attempt_receipt(
        verified=verified,
        row=started_row,
        kind="start",
        state="started",
        execution_uuid=execution_uuid,
    )
    exclusive_json(ledger_root / "starts" / f"{started_row['plan_key']}.json", start)

    clock[0] += WALL_CEILING_SECONDS * 1_000_000_000 - 1
    with pytest.raises(FreezeError, match="before the immutable deadline"):
        record_budget_timeout(verified=verified, ledger_root=ledger_root)
    assert not (ledger_root / "budget-timeout.json").exists()
    assert not (ledger_root / "global-stop.json").exists()

    clock[0] += 2
    receipt = record_budget_timeout(verified=verified, ledger_root=ledger_root)
    assert Path(receipt["path"]).is_file()
    stop = json.loads((ledger_root / "global-stop.json").read_text(encoding="utf-8"))
    assert stop["trigger"] == "monotonic_14h_deadline"
    assert stop["outcome_metrics_read"] is False
    terminals = list((ledger_root / "terminals").glob("*.json"))
    assert len(terminals) == 160
    states = {path.stem: json.loads(path.read_text(encoding="utf-8")) for path in terminals}
    assert states[str(started_row["plan_key"])]["state"] == "failed"
    assert states[str(started_row["plan_key"])]["error_type"] == "MonotonicBudgetExpired"
    assert {value["state"] for key, value in states.items() if key != str(started_row["plan_key"])} == {
        "not_started_after_failure"
    }


def test_coordinator_reconciliation_is_outcome_blind_complete_and_idempotent(
    tmp_path: Path,
    frozen_rows: dict[str, list[dict[str, Any]]],
) -> None:
    verified = _verified(tmp_path, frozen_rows)
    execution_uuid = "00000000-0000-4000-8000-000000000002"
    ledger_root, _provision, _provision_sha, _pod_id = _initialize_test_ledger(
        tmp_path / "coordinator",
        verified,
        execution_uuid,
    )
    started_row = verified.all_rows[0]
    lease_body = {
        "schema": "goalzendo.g00f_run_ownership_lease",
        "schema_version": 1,
        "execution_uuid": execution_uuid,
        "freeze_file_sha256": verified.file_sha256,
        "freeze_digest": verified.digest,
        "panel_id": started_row["panel_id"],
        "plan_key": started_row["plan_key"],
        "run_id": started_row["run_id"],
        "worker_index": started_row["worker_index"],
        "worker_order": started_row["worker_order"],
        "owner_pid": 12345,
        "acquired_monotonic_ns": 99,
        "g01_launch_authorized": False,
    }
    exclusive_json(
        ledger_root / "leases" / f"{started_row['plan_key']}.json",
        {**lease_body, "lease_digest": semantic_digest(lease_body)},
    )
    start = _attempt_receipt(
        verified=verified,
        row=started_row,
        kind="start",
        state="started",
        execution_uuid=execution_uuid,
    )
    exclusive_json(ledger_root / "starts" / f"{started_row['plan_key']}.json", start)
    with pytest.raises(FreezeError, match="already exists"):
        exclusive_json(ledger_root / "starts" / f"{started_row['plan_key']}.json", start)

    cancel_path = tmp_path / "coordinator-cancel.json"
    created = reconcile_execution_failure(
        verified=verified,
        ledger_root=ledger_root,
        error_type="WorkerNonzeroExit",
        trigger="worker_nonzero_exit",
        cancel_receipt=cancel_path,
    )
    payload = json.loads(cancel_path.read_text(encoding="utf-8"))
    assert payload["terminal_state_counts"] == {
        "failed": 1,
        "not_started_after_failure": 159,
    }
    assert payload["outcome_metrics_read"] is False
    assert payload["predictions_read"] is False
    assert payload["g01_launch_authorized"] is False
    assert len(list((ledger_root / "terminals").glob("*.json"))) == 160

    repeated = reconcile_execution_failure(
        verified=verified,
        ledger_root=ledger_root,
        error_type="WorkerNonzeroExit",
        trigger="worker_nonzero_exit",
        cancel_receipt=cancel_path,
    )
    assert repeated == created
    with pytest.raises(FreezeError, match="operational trigger"):
        reconcile_execution_failure(
            verified=verified,
            ledger_root=ledger_root,
            error_type="MetricBelowThreshold",
            trigger="metric_dependent_stop",
            cancel_receipt=tmp_path / "forbidden-cancel.json",
        )


def test_watchdog_signals_workers_before_recording_timeout(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    watchdog = _load_script(WATCHDOG, "g00f_watchdog_test")
    budget = tmp_path / "budget.json"
    pid_file = tmp_path / "pids.txt"
    started = tmp_path / "started.json"
    normal_stop = tmp_path / "normal-stop.json"
    fired = tmp_path / "fired.json"
    cancel = tmp_path / "cancel.json"
    kill = tmp_path / "kill.json"
    budget_body = {
        "schema": "goalzendo.g00f_monotonic_budget_start",
        "schema_version": 1,
        "execution_uuid": "00000000-0000-4000-8000-000000000001",
        "freeze_file_sha256": "f" * 64,
        "freeze_digest": "d" * 64,
        "started_monotonic_ns": 100,
        "wall_ceiling_seconds": 14 * 60 * 60,
    }
    _write_json(
        budget,
        {**budget_body, "budget_digest": watchdog._digest(budget_body)},
    )
    pid_file.write_text("11 12\n", encoding="ascii")
    events: list[tuple[str, Any]] = []

    def fake_kill(pid: int, sig: int) -> None:
        events.append(("signal", (pid, sig)))

    def fake_run(command: list[str], *, check: bool) -> SimpleNamespace:
        events.append(("timeout-command", (command, check)))
        return SimpleNamespace(returncode=1)

    monkeypatch.setattr(
        watchdog.time,
        "monotonic_ns",
        lambda: 100 + 14 * 60 * 60 * 1_000_000_000 + 1,
    )
    monkeypatch.setattr(watchdog.time, "sleep", lambda _seconds: None)
    monkeypatch.setattr(watchdog.os, "kill", fake_kill)
    monkeypatch.setattr(watchdog.subprocess, "run", fake_run)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(WATCHDOG),
            "--budget-start",
            str(budget),
            "--pid-file",
            str(pid_file),
            "--started-receipt",
            str(started),
            "--normal-stop-receipt",
            str(normal_stop),
            "--fired-receipt",
            str(fired),
            "--cancel-receipt",
            str(cancel),
            "--kill-receipt",
            str(kill),
            "--",
            "goalzendo-g00f",
            "record-timeout",
        ],
    )
    prior_sigterm_handler = signal.getsignal(signal.SIGTERM)
    try:
        assert watchdog.main() == 1
    finally:
        signal.signal(signal.SIGTERM, prior_sigterm_handler)
    timeout_index = next(index for index, event in enumerate(events) if event[0] == "timeout-command")
    term_indices = [
        index
        for index, event in enumerate(events)
        if event == ("signal", (11, signal.SIGTERM)) or event == ("signal", (12, signal.SIGTERM))
    ]
    assert len(term_indices) == 2
    assert max(term_indices) < timeout_index
    kill_indices = [
        index
        for index, event in enumerate(events)
        if event == ("signal", (11, signal.SIGKILL)) or event == ("signal", (12, signal.SIGKILL))
    ]
    assert len(kill_indices) == 2
    assert timeout_index < min(kill_indices)
    for path, schema in (
        (started, "goalzendo.g00f_watchdog_started"),
        (fired, "goalzendo.g00f_watchdog_deadline_fired"),
        (cancel, "goalzendo.g00f_coordinated_timeout_cancel"),
        (kill, "goalzendo.g00f_coordinated_timeout_kill"),
    ):
        payload = json.loads(path.read_text(encoding="utf-8"))
        body = {key: value for key, value in payload.items() if key != "receipt_digest"}
        assert payload["schema"] == schema
        assert payload["receipt_digest"] == watchdog._digest(body)
        assert payload["outcome_metrics_read"] is False
        assert payload["predictions_read"] is False
        assert payload["g01_launch_authorized"] is False
    assert json.loads(kill.read_text(encoding="utf-8"))["timeout_command_exit_code"] == 1


def test_watchdog_normal_stop_is_explicit_and_predeadline(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    watchdog = _load_script(WATCHDOG, "g00f_watchdog_normal_stop_test")
    budget = tmp_path / "budget.json"
    pid_file = tmp_path / "pids.txt"
    started = tmp_path / "started.json"
    normal_stop = tmp_path / "normal-stop.json"
    fired = tmp_path / "fired.json"
    cancel = tmp_path / "cancel.json"
    kill = tmp_path / "kill.json"
    budget_body = {
        "schema": "goalzendo.g00f_monotonic_budget_start",
        "schema_version": 1,
        "execution_uuid": "00000000-0000-4000-8000-000000000001",
        "freeze_file_sha256": "f" * 64,
        "freeze_digest": "d" * 64,
        "started_monotonic_ns": 100,
        "wall_ceiling_seconds": 14 * 60 * 60,
    }
    _write_json(budget, {**budget_body, "budget_digest": watchdog._digest(budget_body)})
    pid_file.write_text("11 12\n", encoding="ascii")
    handlers: dict[int, Callable[[int, Any], None]] = {}

    def fake_signal(signal_number: int, handler: Callable[[int, Any], None]) -> None:
        handlers[signal_number] = handler

    clock = iter((101, 102, 103))
    monkeypatch.setattr(watchdog.signal, "signal", fake_signal)
    monkeypatch.setattr(watchdog.time, "monotonic_ns", lambda: next(clock))
    monkeypatch.setattr(
        watchdog.time,
        "sleep",
        lambda _seconds: handlers[signal.SIGTERM](signal.SIGTERM, None),
    )
    monkeypatch.setattr(watchdog.os, "getpid", lambda: 4321)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(WATCHDOG),
            "--budget-start",
            str(budget),
            "--pid-file",
            str(pid_file),
            "--started-receipt",
            str(started),
            "--normal-stop-receipt",
            str(normal_stop),
            "--fired-receipt",
            str(fired),
            "--cancel-receipt",
            str(cancel),
            "--kill-receipt",
            str(kill),
            "--",
            "goalzendo-g00f",
            "record-timeout",
        ],
    )
    assert watchdog.main() == 0
    assert started.is_file() and normal_stop.is_file()
    assert not fired.exists() and not cancel.exists() and not kill.exists()
    started_payload = json.loads(started.read_text(encoding="utf-8"))
    stopped_payload = json.loads(normal_stop.read_text(encoding="utf-8"))
    assert started_payload["watchdog_pid"] == stopped_payload["watchdog_pid"] == 4321
    assert started_payload["started_monotonic_ns"] == 101
    assert stopped_payload["stopped_monotonic_ns"] == 103
    assert stopped_payload["stopped_monotonic_ns"] < stopped_payload["deadline_monotonic_ns"]


def _sealed_panel_fixture(
    tmp_path: Path,
    frozen_rows: dict[str, list[dict[str, Any]]],
) -> tuple[
    VerifiedFreeze,
    Path,
    dict[Path, str],
    list[tuple[Path, dict[str, Any]]],
]:
    verified = _verified(tmp_path, frozen_rows, local_artifacts=True)
    execution_uuid = "00000000-0000-4000-8000-000000000006"
    ledger, _provision, _expected_sha, _pod_id = _initialize_test_ledger(
        tmp_path,
        verified,
        execution_uuid,
    )
    sealed_hashes: dict[Path, str] = {}
    terminal_payloads: list[tuple[Path, dict[str, Any]]] = []
    for row in verified.all_rows:
        launch_receipt = {"path": "fixture-launch-receipt.json"}
        start = _attempt_receipt(
            verified=verified,
            row=row,
            kind="start",
            state="started",
            launch_receipt=launch_receipt,
            execution_uuid=execution_uuid,
        )
        _write_json(ledger / "starts" / f"{row['plan_key']}.json", start)
        artifact = Path(str(row["artifact_path"]))
        artifact.mkdir(parents=True)
        seals: dict[str, dict[str, Any]] = {}
        for name in ("metrics.jsonl", "predictions.jsonl", "summary.json"):
            outcome = artifact / name
            outcome.write_text(
                json.dumps({"plan_key": row["plan_key"], "name": name}) + "\n",
                encoding="utf-8",
            )
            digest = sha256_file(outcome)
            sealed_hashes[outcome.resolve()] = digest
            os.chmod(outcome, 0)
            seals[name] = {
                "path": str(outcome),
                "bytes": outcome.stat().st_size,
                "sha256": digest,
                "sealed_mode": 0,
            }
        terminal = _attempt_receipt(
            verified=verified,
            row=row,
            kind="terminal",
            state="complete",
            launch_receipt=launch_receipt,
            execution_uuid=execution_uuid,
        )
        terminal_body = {key: value for key, value in terminal.items() if key != "receipt_digest"}
        terminal_body["outcome_file_seals"] = seals
        terminal_payloads.append(
            (
                ledger / "terminals" / f"{row['plan_key']}.json",
                {**terminal_body, "receipt_digest": semantic_digest(terminal_body)},
            )
        )
    return verified, ledger, sealed_hashes, terminal_payloads


def test_outcome_sealer_hashes_exactly_three_files_then_sets_mode_zero(tmp_path: Path) -> None:
    artifact = tmp_path / "artifact"
    artifact.mkdir()
    expected: dict[str, str] = {}
    for name in ("metrics.jsonl", "predictions.jsonl", "summary.json"):
        target = artifact / name
        target.write_text(f"{name}\n", encoding="utf-8")
        expected[name] = sha256_file(target)

    seals = _seal_outcome_files(artifact)
    assert seals == {
        name: {
            "path": str(artifact / name),
            "bytes": (artifact / name).stat().st_size,
            "sha256": expected[name],
            "sealed_mode": 0,
        }
        for name in ("metrics.jsonl", "predictions.jsonl", "summary.json")
    }
    assert {os.stat(artifact / name).st_mode & 0o777 for name in seals} == {0}

    incomplete = tmp_path / "incomplete"
    incomplete.mkdir()
    (incomplete / "metrics.jsonl").write_text("metric\n", encoding="utf-8")
    (incomplete / "predictions.jsonl").write_text("prediction\n", encoding="utf-8")
    with pytest.raises(FreezeError, match="lacks direct outcome file"):
        _seal_outcome_files(incomplete)


def _sealed_file_sha256(
    sealed_hashes: dict[Path, str],
    real_sha256_file: Callable[[str | Path], str],
) -> Callable[[str | Path], str]:
    def readable(path: str | Path) -> str:
        resolved = Path(path).resolve()
        if resolved in sealed_hashes:
            return sealed_hashes[resolved]
        return real_sha256_file(path)

    return readable


def test_incomplete_or_failed_panel_keeps_every_outcome_file_unreadable(
    tmp_path: Path,
    frozen_rows: dict[str, list[dict[str, Any]]],
) -> None:
    verified, ledger, sealed_hashes, terminals = _sealed_panel_fixture(tmp_path, frozen_rows)
    for terminal_path, payload in terminals[:-1]:
        _write_json(terminal_path, payload)

    _unseal_if_complete(verified, ledger)
    assert not (ledger / "panel-unseal.json").exists()
    assert {os.stat(path).st_mode & 0o777 for path in sealed_hashes} == {0}

    last_path, last_payload = terminals[-1]
    failed_body = {
        key: value
        for key, value in last_payload.items()
        if key not in {"outcome_file_seals", "receipt_digest"}
    }
    failed_body.update({"state": "failed", "error_type": "FixtureFailure"})
    _write_json(last_path, {**failed_body, "receipt_digest": semantic_digest(failed_body)})
    _unseal_if_complete(verified, ledger)
    assert not (ledger / "panel-unseal.json").exists()
    assert {os.stat(path).st_mode & 0o777 for path in sealed_hashes} == {0}


def test_panel_unseal_rejects_tampered_terminal_before_exposing_any_outcome(
    tmp_path: Path,
    frozen_rows: dict[str, list[dict[str, Any]]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    verified, ledger, sealed_hashes, terminals = _sealed_panel_fixture(tmp_path, frozen_rows)
    for terminal_path, payload in terminals:
        _write_json(terminal_path, payload)
    last_path, last_payload = terminals[-1]
    last_payload["receipt_digest"] = "0" * 64
    _write_json(last_path, last_payload)
    monkeypatch.setattr(
        freeze_module,
        "sha256_file",
        _sealed_file_sha256(sealed_hashes, freeze_module.sha256_file),
    )

    with pytest.raises(FreezeError, match=r"terminal|receipt|digest"):
        _unseal_if_complete(verified, ledger)
    assert not (ledger / "panel-unseal.json").exists()
    assert {os.stat(path).st_mode & 0o777 for path in sealed_hashes} == {0}


def test_panel_unseal_preflights_all_seals_before_chmod(
    tmp_path: Path,
    frozen_rows: dict[str, list[dict[str, Any]]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    verified, ledger, sealed_hashes, terminals = _sealed_panel_fixture(tmp_path, frozen_rows)
    for terminal_path, payload in terminals:
        _write_json(terminal_path, payload)
    last_path, last_payload = terminals[-1]
    last_payload["outcome_file_seals"]["summary.json"]["sha256"] = "0" * 64
    last_body = {key: value for key, value in last_payload.items() if key != "receipt_digest"}
    last_payload["receipt_digest"] = semantic_digest(last_body)
    _write_json(last_path, last_payload)
    monkeypatch.setattr(
        freeze_module,
        "sha256_file",
        _sealed_file_sha256(sealed_hashes, freeze_module.sha256_file),
    )

    with pytest.raises(FreezeError, match="sealed outcome bytes"):
        _unseal_if_complete(verified, ledger)
    assert not (ledger / "panel-unseal.json").exists()
    assert {os.stat(path).st_mode & 0o777 for path in sealed_hashes} == {0}


def test_concurrent_panel_unseal_has_one_receipt_and_no_losing_worker_error(
    tmp_path: Path,
    frozen_rows: dict[str, list[dict[str, Any]]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    verified, ledger, sealed_hashes, terminal_payloads = _sealed_panel_fixture(
        tmp_path,
        frozen_rows,
    )
    for terminal_path, payload in terminal_payloads:
        _write_json(terminal_path, payload)

    real_exclusive = freeze_module.exclusive_json
    real_sha256_file = freeze_module.sha256_file
    contenders = threading.Barrier(2)
    hash_barriers = {path: threading.Barrier(2) for path in sealed_hashes}

    def readable_sealed_sha256(path: str | Path) -> str:
        resolved = Path(path).resolve()
        if resolved in sealed_hashes:
            hash_barriers[resolved].wait(timeout=5)
            return sealed_hashes[resolved]
        return real_sha256_file(path)

    def contested_exclusive(path: str | Path, value: dict[str, Any]) -> None:
        if Path(path).name == "panel-unseal.json":
            with suppress(threading.BrokenBarrierError):
                contenders.wait(timeout=5)
        real_exclusive(path, value)

    monkeypatch.setattr(freeze_module, "sha256_file", readable_sealed_sha256)
    monkeypatch.setattr(freeze_module, "exclusive_json", contested_exclusive)
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(_unseal_if_complete, verified, ledger) for _ in range(2)]
    errors = [future.exception() for future in futures]
    assert errors == [None, None]

    unseal_path = ledger / "panel-unseal.json"
    payload = json.loads(unseal_path.read_text(encoding="utf-8"))
    body = {key: value for key, value in payload.items() if key != "unseal_digest"}
    assert payload["terminal_complete_count"] == 160
    assert payload["outcome_files"] == [
        {
            "plan_key": row["plan_key"],
            "run_id": row["run_id"],
            "name": name,
            "bytes": (Path(str(row["artifact_path"])) / name).stat().st_size,
            "sha256": sealed_hashes[(Path(str(row["artifact_path"])) / name).resolve()],
            "unsealed_mode": 0o400,
        }
        for row in verified.all_rows
        for name in ("metrics.jsonl", "predictions.jsonl", "summary.json")
    ]
    assert payload["unseal_digest"] == semantic_digest(body)
    assert {os.stat(path).st_mode & 0o777 for path in sealed_hashes} == {0o400}


def _evaluation_dependencies(
    monkeypatch: pytest.MonkeyPatch,
    *,
    verified: VerifiedFreeze,
    spec: RunSpec,
    path: Path,
    predictions: list[dict[str, Any]],
    by_pair: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    path.mkdir(parents=True, exist_ok=True)
    outcome_file_seals: dict[str, dict[str, Any]] = {}
    for name in ("metrics.jsonl", "predictions.jsonl", "summary.json"):
        outcome = path / name
        outcome.write_text(json.dumps({"fixture": name}) + "\n", encoding="utf-8")
        outcome_file_seals[name] = {
            "path": str(outcome),
            "bytes": outcome.stat().st_size,
            "sha256": sha256_file(outcome),
            "sealed_mode": 0,
        }
        os.chmod(outcome, 0o400)
    attempts = path / "attempts"
    attempts.mkdir()
    (attempts / "attempt-0001.json").write_text("{}\n", encoding="utf-8")
    (attempts / "resolved-config-0001.yaml").write_text("{}\n", encoding="utf-8")
    model_identity = freeze_module.MODEL_RUNTIME_IDENTITIES["g00f-0p5b"]
    model_metadata = {
        "requested_model": CONFIG_SPECS["g00f-0p5b"]["model"],
        "requested_revision": CONFIG_SPECS["g00f-0p5b"]["revision"],
        "resolved_revision": CONFIG_SPECS["g00f-0p5b"]["revision"],
        "requested_dtype": "bfloat16",
        **dict(model_identity),
        "torch_version": freeze_module.FROZEN_MODEL_DEPENDENCIES["torch"],
        "transformers_version": freeze_module.FROZEN_MODEL_DEPENDENCIES["transformers"],
        "peft_version": freeze_module.FROZEN_MODEL_DEPENDENCIES["peft"],
        "update": {"method": "full"},
        "initial_trainable_parameter_digest": "a" * 64,
    }
    snapshot_root = path / "snapshot"
    tokenizer_metadata = {
        "tokenizer_resolved_revision": CONFIG_SPECS["g00f-0p5b"]["revision"],
        "tokenizer_name_or_path": str(snapshot_root),
        "dependency_versions": dict(freeze_module.FROZEN_MODEL_DEPENDENCIES),
        **dict(freeze_module.TOKENIZER_RUNTIME_IDENTITY),
    }
    rendering_metadata: dict[str, Any] = {
        "final_factorial_prompt_digest": "fixture-final-prompt-digest",
        "final_renderer_counts": {"heldout_symbolic_v1": 512},
        "training_prompt_digest": "fixture-training-prompt-digest",
    }
    dataset_metadata: dict[str, Any] = {
        "rendering": rendering_metadata,
        "dataset_binding_digest": stable_hash(
            {"symbolic": {}, "rendering": rendering_metadata},
            64,
        ),
    }
    packages: dict[str, str] = {}
    accelerator = {
        "torch_available": True,
        "cuda_available": True,
        "torch_version": "2.8.0+cu128",
        "cuda_runtime": "12.8",
        "cuda_devices": [{"index": 0, "name": "NVIDIA H100 SXM"}],
    }
    environment = {
        "run_id": path.name,
        "seed": spec.seed,
        "attempt": 1,
        "packages": packages,
        "accelerator": accelerator,
        "python": {
            "version": "3.12.3 (main, test build)",
            "implementation": "CPython",
        },
    }
    runtime_environment = {
        "installed_distributions": packages,
        "installed_distributions_digest": semantic_digest(packages),
        "accelerator": accelerator,
        "accelerator_digest": semantic_digest(accelerator),
    }
    manifests = {
        "model.json": {
            "schema_version": 1,
            "kind": "model",
            "metadata": model_metadata,
            "digest": stable_hash(model_metadata, 64),
        },
        "tokenizer.json": {
            "schema_version": 1,
            "kind": "tokenizer",
            "metadata": tokenizer_metadata,
            "digest": stable_hash(tokenizer_metadata, 64),
        },
        "dataset.json": {
            "schema_version": 1,
            "kind": "dataset",
            "metadata": dataset_metadata,
            "digest": stable_hash(dataset_metadata, 64),
        },
    }

    def fake_read_json(target: Path) -> dict[str, Any]:
        if target.name in {"environment.json", "attempt-0001.json"}:
            return environment
        if target.name == "identity.json":
            return {"implementation_fingerprint": freeze_module.FROZEN_GOALZENDO_IMPLEMENTATION_FINGERPRINT}
        if target.name == "summary.json":
            return {
                "plan_key": spec.plan_key,
                "seed": spec.seed,
                "run_id": path.name,
                "backend_version": EXPERIMENT_BACKEND_VERSION,
                "final_step": 1_000,
                "algorithm": "sft",
                "device": "cuda",
                "derived_seeds": dict(spec.seeds),
                "numerical_execution": {
                    "cublas_workspace_config": ":4096:8",
                    "cuda_matmul_allow_tf32": False,
                    "cudnn_allow_tf32": False,
                    "cudnn_benchmark": False,
                    "cudnn_deterministic": True,
                    "deterministic_algorithms": True,
                    "deterministic_warn_only": False,
                    "float32_matmul_precision": "highest",
                },
                "dataset_binding_digest": dataset_metadata["dataset_binding_digest"],
            }
        if target.name == "status.json":
            return {
                "state": "complete",
                "run_id": path.name,
                "seed": spec.seed,
                "attempt": 1,
                "resumed": False,
                "repaired_streams": [],
                "last_step": 1_000,
            }
        return manifests[target.name]

    resolved = copy.deepcopy(dict(spec.config))
    resolved["seed"] = spec.seed
    monkeypatch.setattr(
        evaluator_module,
        "verify_completion_attestation",
        lambda _path: {
            "schema": "goalzendo.run_completion",
            "schema_version": 1,
            "artifact_schema_version": 2,
            "run_id": path.name,
            "seed": spec.seed,
            "files": {
                name: {}
                for name in {
                    "environment.json",
                    "identity.json",
                    "implementation.json",
                    "manifests/dataset.json",
                    "manifests/model.json",
                    "manifests/tokenizer.json",
                    "metrics.jsonl",
                    "predictions.jsonl",
                    "resolved_config.yaml",
                    "status.json",
                    "summary.json",
                }
            },
            "completion_digest": "1" * 64,
        },
    )
    monkeypatch.setattr(evaluator_module, "read_json", fake_read_json)
    monkeypatch.setattr(evaluator_module, "_resolved_config", lambda _path: resolved)
    monkeypatch.setattr(
        evaluator_module,
        "_verify_run_binding",
        lambda **_kwargs: {
            "model_receipts": {"g00f-0p5b": {"snapshot_root": str(snapshot_root)}},
            "launch_receipt": {"runtime_environment": runtime_environment},
        },
    )
    monkeypatch.setattr(evaluator_module, "_final_predictions", lambda _path, _spec: predictions)
    monkeypatch.setattr(
        evaluator_module,
        "_validate_prediction_rows",
        lambda records, _by_sample: ({str(row["sample_id"]): row for row in records}, 0.0),
    )
    monkeypatch.setattr(
        evaluator_module,
        "materialize_banks",
        lambda _config, _seeds: SimpleNamespace(metadata=dataset_metadata),
    )
    row = dict(
        copy.deepcopy(next(row for row in verified.plans["g00f-0p5b"] if row["plan_key"] == spec.plan_key))
    )
    row["artifact_path"] = str(path)
    by_sample = {
        str(record["sample_id"]): SimpleNamespace(sample_id=str(record["sample_id"]))
        for record in predictions
    }
    return {
        "verified": verified,
        "panel_id": "g00f-0p5b",
        "spec": spec,
        "path": path,
        "row": row,
        "outcome_file_seals": outcome_file_seals,
        "regenerated": lambda _spec: (by_sample, by_pair),
        "regenerate_dataset": lambda _spec, _snapshot, _cache: dataset_metadata,
        "receipt_cache": {},
    }


def _paired_predictions(
    *, correct_per_side: int | None = None
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    predictions: list[dict[str, Any]] = []
    pairs: dict[str, dict[str, Any]] = {}
    for index in range(256):
        pair_id = f"pair-{index:03d}"
        base_id = f"{pair_id}-base"
        mirror_id = f"{pair_id}-mirror"
        if correct_per_side is None:
            action_base = action_mirror = index % 2
        elif index < correct_per_side:
            action_base, action_mirror = 0, 1
        else:
            action_base, action_mirror = 1, 0
        for sample_id, choice_y, action in (
            (base_id, 0, action_base),
            (mirror_id, 1, action_mirror),
        ):
            predictions.append(
                {
                    "sample_id": sample_id,
                    "choice_y": choice_y,
                    "choice_p": choice_y,
                    "choice_q": choice_y,
                    "predicted_action": action,
                    "score_a": 1.0 if action == 0 else 0.0,
                    "score_b": 1.0 if action == 1 else 0.0,
                    "probability_b": 0.25 if action == 0 else 0.75,
                    "margin_b_minus_a": -1.0 if action == 0 else 1.0,
                }
            )
        pairs[pair_id] = {
            "base": SimpleNamespace(sample_id=base_id),
            "mirror": SimpleNamespace(sample_id=mirror_id),
        }
    return predictions, pairs


def test_evaluator_accepts_only_canonical_config_roots_and_frozen_store_paths(
    tmp_path: Path,
    frozen_rows: dict[str, list[dict[str, Any]]],
) -> None:
    verified = _verified(tmp_path, frozen_rows)
    roots = {
        panel_id: Path(
            str(
                get_path(
                    load_config(ROOT / str(specification["path"])),
                    "run.output_root",
                )
            )
        )
        for panel_id, specification in CONFIG_SPECS.items()
    }
    expected, rows = evaluator_module._expected_paths(verified, roots)
    assert len(expected) == 160
    assert len(rows) == 160
    assert {row["state"] for row in rows} == {"missing_not_attempted"}
    adjacent = {panel_id: root / "substituted-copy" for panel_id, root in roots.items()}
    with pytest.raises(EvaluationError, match="frozen panel output root"):
        evaluator_module._expected_paths(verified, adjacent)


def test_evaluator_rejects_changed_outcome_bytes_even_with_regenerated_completion(
    tmp_path: Path,
    frozen_rows: dict[str, list[dict[str, Any]]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    verified = _verified(tmp_path, frozen_rows)
    spec = _case_spec("no_signal")
    predictions, pairs = _paired_predictions()
    path = tmp_path / "sealed-run"
    arguments = _evaluation_dependencies(
        monkeypatch,
        verified=verified,
        spec=spec,
        path=path,
        predictions=predictions,
        by_pair=pairs,
    )
    original_seals = copy.deepcopy(arguments["outcome_file_seals"])
    copied = tmp_path / "substituted-copy"
    shutil.copytree(path, copied)
    copied_row = copy.deepcopy(arguments["row"])
    copied_row["artifact_path"] = str(copied)
    copied_seals = copy.deepcopy(original_seals)
    for name, seal in copied_seals.items():
        seal["path"] = str(copied / name)
    copied_predictions = copied / "predictions.jsonl"
    os.chmod(copied_predictions, 0o600)
    copied_predictions.write_text('{"regenerated":"COMPLETE","changed":true}\n', encoding="utf-8")
    os.chmod(copied_predictions, 0o400)
    with pytest.raises(EvaluationError, match="terminal receipt seal"):
        _evaluate_run(
            **{
                **arguments,
                "path": copied,
                "row": copied_row,
                "outcome_file_seals": copied_seals,
            }
        )


def test_no_signal_uses_exact_pair_determinism_not_a_binomial_leakage_band(
    tmp_path: Path,
    frozen_rows: dict[str, list[dict[str, Any]]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    verified = _verified(tmp_path, frozen_rows)
    spec = _case_spec("no_signal")
    predictions, pairs = _paired_predictions()
    arguments = _evaluation_dependencies(
        monkeypatch,
        verified=verified,
        spec=spec,
        path=tmp_path / "run-no-signal",
        predictions=predictions,
        by_pair=pairs,
    )
    result = _evaluate_run(**arguments)
    pairing = result["no_signal_pairing"]
    assert pairing == {
        "canonical_prompt_bytes_identical": True,
        "identical_base_mirror_scores_and_actions": 256,
        "pairs": 256,
        "structurally_forced_law_correct": 256,
        "marginal_action_b_count": 256,
        "marginal_action_b_rate": 0.5,
        "passed": True,
    }
    assert result["chance"] is None

    predictions[1]["score_b"] = 0.5
    changed = _evaluate_run(**arguments)
    assert changed["no_signal_pairing"]["identical_base_mirror_scores_and_actions"] == 255
    assert changed["no_signal_pairing"]["passed"] is False


def test_evaluator_replays_rendering_and_summary_dataset_binding(
    tmp_path: Path,
    frozen_rows: dict[str, list[dict[str, Any]]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    verified = _verified(tmp_path, frozen_rows)
    spec = _case_spec("no_signal")
    predictions, pairs = _paired_predictions()
    arguments = _evaluation_dependencies(
        monkeypatch,
        verified=verified,
        spec=spec,
        path=tmp_path / "run-rendering-binding",
        predictions=predictions,
        by_pair=pairs,
    )
    _evaluate_run(**arguments)
    changed_rendering = {
        "final_factorial_prompt_digest": "forged-but-semantically-rehashed",
        "final_renderer_counts": {"heldout_symbolic_v1": 512},
        "training_prompt_digest": "fixture-training-prompt-digest",
    }
    changed_metadata = {
        "rendering": changed_rendering,
        "dataset_binding_digest": stable_hash(
            {"symbolic": {}, "rendering": changed_rendering},
            64,
        ),
    }
    with pytest.raises(EvaluationError, match="rendered dataset binding"):
        _evaluate_run(
            **{
                **arguments,
                "regenerate_dataset": lambda _spec, _snapshot, _cache: changed_metadata,
            }
        )
    original_read_json: Callable[[Path], dict[str, Any]] = evaluator_module.__dict__["read_json"]

    def forged_summary(target: Path) -> dict[str, Any]:
        payload = original_read_json(target)
        if target.name == "summary.json":
            payload = {**payload, "dataset_binding_digest": "f" * 64}
        return payload

    monkeypatch.setattr(evaluator_module, "read_json", forged_summary)
    with pytest.raises(EvaluationError, match="rendered dataset binding"):
        _evaluate_run(**arguments)


def test_canonical_no_signal_mirror_prompts_are_exactly_identical() -> None:
    spec = _case_spec("no_signal")
    banks = materialize_banks(spec.config, spec.seeds)
    decisions = banks.final_factorial.decisions
    renderer_ids = tuple(str(value) for value in get_path(spec.config, "data.heldout_renderers"))
    renderer_map = _counterbalanced_renderer_map(
        decisions,
        renderer_ids,
        spec.seeds["rendering"],
        evaluation=True,
    )
    pairs: dict[str, dict[str, Any]] = {}
    for decision in decisions:
        assert decision.mirror_pair_id is not None and decision.mirror_role is not None
        pairs.setdefault(decision.mirror_pair_id, {})[decision.mirror_role] = decision
    assert len(pairs) == 256
    for roles in pairs.values():
        assert set(roles) == {"base", "mirror"}
        base = roles["base"]
        mirror = roles["mirror"]
        assert mirror.koans == (base.koans[1], base.koans[0])
        assert tuple(int(value) for value in mirror.candidate_tuple) == tuple(
            1 - int(value) for value in base.candidate_tuple
        )
        renderer = renderer_map[base.sample_id]
        assert renderer_map[mirror.sample_id] == renderer
        assert render_prompt_view(
            base,
            banks.final_factorial.feature_names,
            renderer_id=renderer,
            prompt_view="no_signal",
        ) == render_prompt_view(
            mirror,
            banks.final_factorial.feature_names,
            renderer_id=renderer,
            prompt_view="no_signal",
        )


@pytest.mark.parametrize(("correct", "passed"), [(244, True), (243, False)])
def test_informative_adapter_requires_at_least_244_of_256_on_each_position(
    tmp_path: Path,
    frozen_rows: dict[str, list[dict[str, Any]]],
    monkeypatch: pytest.MonkeyPatch,
    correct: int,
    passed: bool,
) -> None:
    verified = _verified(tmp_path, frozen_rows)
    spec = _case_spec("law_only")
    predictions, pairs = _paired_predictions(correct_per_side=correct)
    arguments = _evaluation_dependencies(
        monkeypatch,
        verified=verified,
        spec=spec,
        path=tmp_path / f"run-law-{correct}",
        predictions=predictions,
        by_pair=pairs,
    )
    result = _evaluate_run(**arguments)
    assert result["adapter"]["passed"] is passed
    for position in ("A", "B"):
        assert result["adapter"]["positions"][position] == {
            "correct": correct,
            "trials": 256,
            "agreement": correct / 256,
            "passed": passed,
        }


def _synthetic_complete_results(
    frozen_rows: dict[str, list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    target_by_view = {
        "law_only": "choice_y",
        "audit_law_matched": "choice_y",
        "sage_only": "choice_q",
        "herald_only": "choice_p",
    }
    for row in (row for rows in frozen_rows.values() for row in rows):
        view = str(row["training_view"])
        adapter = None
        if view in EXPECTED_INFORMATIVE_VIEWS:
            positions = {
                position: {
                    "correct": 244,
                    "trials": 256,
                    "agreement": 244 / 256,
                    "passed": True,
                }
                for position in ("A", "B")
            }
            adapter = {"target": target_by_view[view], "positions": positions, "passed": True}
        results.append(
            {
                "panel_id": row["panel_id"],
                "plan_key": row["plan_key"],
                "seed": row["seed"],
                "law_family": row["law_family"],
                "training_view": view,
                "adapter": adapter,
                "candidate_order": {"errors": 5, "pairs": 256, "passed": True},
                "chance": {"correct": 256, "trials": 512} if view == "surface_only" else None,
                "no_signal_pairing": (
                    {
                        "identical_base_mirror_scores_and_actions": 256,
                        "pairs": 256,
                        "structurally_forced_law_correct": 256,
                        "marginal_action_b_count": 256,
                        "marginal_action_b_rate": 0.5,
                        "passed": True,
                    }
                    if view == "no_signal"
                    else None
                ),
                "maximum_probability_score_disagreement": 0.0,
                "initial_trainable_parameter_digest": (
                    "a" * 64 if row["panel_id"] == "g00f-0p5b" else "b" * 64
                ),
            }
        )
    return results


def test_aggregate_gate_preserves_every_per_run_adapter_and_no_signal_check(
    frozen_rows: dict[str, list[dict[str, Any]]],
) -> None:
    results = _synthetic_complete_results(frozen_rows)
    measurements, checks = _aggregate_results(results)
    assert checks["rule_adapters"] == {
        "passed": True,
        "per_run_passed": True,
        "aggregate_passed": True,
        "threshold": "correct/trials >= 95/100",
    }
    assert len(measurements["per_run_adapters"]) == 120
    assert {row["plan_key"] for row in measurements["per_run_adapters"]} == {
        row["plan_key"]
        for rows in frozen_rows.values()
        for row in rows
        if row["training_view"] in EXPECTED_INFORMATIVE_VIEWS
    }
    for panel_id in CONFIG_SPECS:
        for view in EXPECTED_INFORMATIVE_VIEWS:
            run_count = sum(row["training_view"] == view for row in frozen_rows[panel_id])
            for position in ("A", "B"):
                panel = measurements["adapter_aggregate"][panel_id][view][position]
                pooled = measurements["adapter_aggregate"]["aggregate"][view][position]
                assert (panel["correct"], panel["trials"], panel["passed"]) == (
                    run_count * 244,
                    run_count * 256,
                    True,
                )
                assert (pooled["correct"], pooled["trials"], pooled["passed"]) == (
                    2 * run_count * 244,
                    2 * run_count * 256,
                    True,
                )
    assert checks["no_signal_pair_determinism"]["passed"] is True
    assert len(measurements["no_signal_pair_determinism"]) == 20

    first_informative = next(result for result in results if result["adapter"] is not None)
    first_informative["adapter"]["positions"]["A"].update(
        {"correct": 243, "agreement": 243 / 256, "passed": False}
    )
    first_informative["adapter"]["passed"] = False
    _measurements, failed = _aggregate_results(results)
    assert failed["rule_adapters"]["per_run_passed"] is False
    assert failed["rule_adapters"]["passed"] is False


def test_exact_integer_chance_bands_are_inclusive_and_aggregate_without_floats(
    frozen_rows: dict[str, list[dict[str, Any]]],
) -> None:
    assert exact_central_binomial_interval(5_120) == (2_490, 2_630)
    assert exact_central_binomial_interval(10_240) == (5_021, 5_219)
    with pytest.raises(EvaluationError, match="positive integer"):
        exact_central_binomial_interval(0)
    with pytest.raises(EvaluationError, match="frozen"):
        exact_central_binomial_interval(512, probability=0.51)

    results = _synthetic_complete_results(frozen_rows)

    def allocate(panel_id: str, total: int) -> None:
        rows = [
            result
            for result in results
            if result["panel_id"] == panel_id and result["training_view"] == "surface_only"
        ]
        quotient, remainder = divmod(total, len(rows))
        for index, result in enumerate(rows):
            result["chance"]["correct"] = quotient + int(index < remainder)

    allocate("g00f-0p5b", 2_490)
    allocate("g00f-1p5b", 2_531)
    _measurements, checks = _aggregate_results(results)
    assert checks["surface_leakage"]["passed"] is True

    first = next(
        result
        for result in results
        if result["panel_id"] == "g00f-0p5b" and result["training_view"] == "surface_only"
    )
    first["chance"]["correct"] -= 1
    _measurements, checks = _aggregate_results(results)
    assert checks["surface_leakage"]["passed"] is False
