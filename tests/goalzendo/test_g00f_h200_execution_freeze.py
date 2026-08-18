from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import time
import types
from pathlib import Path
from typing import Any

import pytest

from goalzendo_g00f_h200.evaluator import (
    EvaluationError,
    _verify_success_watchdog_lifecycle,
    create_preexecution_gate,
)
from goalzendo_g00f_h200.freeze import (
    DATA_ORDER_EQUIVALENCE_CONTRACT,
    H200_HOUR_CEILING,
    H200_MODULE_INVOCATION_CONTRACT,
    HISTORICAL_H100_PARENT_FILES,
    PROFILE_CONTRACT,
    PROFILE_SELECTOR_CONTRACT,
    QUALIFICATION_PRODUCER_CONTRACT,
    RUNPOD_OPERATOR_HANDOFF_CONTRACT,
    RUNPOD_PROVISIONING_CONTRACT,
    STORAGE_PREFLIGHT_CONTRACT,
    WALL_CEILING_SECONDS,
    WATCHDOG_SUPERVISION_CONTRACT,
    FreezeError,
    VerifiedFreeze,
    _measure_storage_preflight,
    _provider_identity_ssh_argv,
    bind_runpod_provision_receipt,
    canonical_runpod_create_command,
    create_runpod_provision_receipt,
    exclusive_copy,
    historical_h100_parent_binding,
    semantic_digest,
    verify_detached_supervisor_receipts,
    verify_runpod_gpu_catalog_snapshot,
)
from goalzendo_g00f_h200.qualification import (
    BASELINE_FALLBACK_COMPARISON_RECORD_COUNT,
    BASELINE_FALLBACK_PROCESS_RECORD_COUNT,
    COMPARISON_RECORD_COUNT,
    MAXIMUM_QUALIFICATION_EVIDENCE_BYTES,
    PROCESS_RECORD_COUNT,
    TRAINABLE_NUMEL,
    TUNED_CAPACITY_DISQUALIFIERS,
)
from goalzendo_g00f_h200.qualification import (
    PROFILE_CONTRACT as QUALIFICATION_PROFILE_CONTRACT,
)
from goalzendo_g00f_h200.qualification_producer import (
    MINIMUM_PROVISION_REMAINING_AT_START_SECONDS,
    POST_QUALIFICATION_RESERVE_SECONDS,
    PRODUCER_CEILING_SECONDS,
)

ROOT = Path(__file__).resolve().parents[2]
EXECUTION_UUID_FIXTURE = "11111111-1111-4111-8111-111111111111"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_watchdog_module() -> Any:
    path = ROOT / "runs/goalzendo/g00f_h200_watchdog.py"
    spec = importlib.util.spec_from_file_location("g00f_h200_watchdog_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_h200_freeze_binds_the_immutable_never_run_h100_parent(tmp_path: Path) -> None:
    for relative in HISTORICAL_H100_PARENT_FILES:
        source = ROOT / relative
        target = tmp_path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)

    binding = historical_h100_parent_binding(tmp_path)
    assert binding["role"] == "immutable_never_run_hardware_parent"
    assert binding["scientific_semantics_inherited"] is True
    assert binding["hardware_execution_identity_inherited"] is False
    assert binding["outcomes_seen"] is False
    assert {row["path"]: row["sha256"] for row in binding["files"]} == dict(HISTORICAL_H100_PARENT_FILES)

    parent = tmp_path / next(iter(HISTORICAL_H100_PARENT_FILES))
    parent.chmod(0o600)
    parent.write_bytes(parent.read_bytes() + b"\n")
    with pytest.raises(FreezeError, match="immutable H100 parent artifact changed"):
        historical_h100_parent_binding(tmp_path)


def test_h200_selector_contract_is_symmetric_and_storage_bounded() -> None:
    replay = PROFILE_SELECTOR_CONTRACT["profile_replay_requirements"]
    assert replay["profiles"] == ["baseline", "tuned"]
    assert replay["exact_model_state_hashes"] is True
    assert replay["exact_optimizer_state_hashes"] is True
    assert replay["exact_output_hashes"] is True
    assert replay["exact_vector_native_chunk_manifests"] is True
    assert replay["fixed_boundary_equivalence_on_all_four_gpu_uuids"] is True
    assert PROFILE_SELECTOR_CONTRACT["baseline_requirements"] == {
        "deterministic_replay_exact": True,
        "all_four_gpu_boundary_equivalence_exact": True,
        "maximum_projected_wall_seconds": 43_200,
    }
    assert WATCHDOG_SUPERVISION_CONTRACT["launcher_identity"] == ("linux_pidfd_bound_before_guardian_fork")
    assert WATCHDOG_SUPERVISION_CONTRACT["network_volume_io_before_deadline_signal"] is False
    assert WATCHDOG_SUPERVISION_CONTRACT["wall_ceiling_seconds"] == WALL_CEILING_SECONDS


def test_storage_preflight_uses_available_blocks_and_fails_below_reserve(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    execution_root = tmp_path / "execution"
    engineering_root = tmp_path / "engineering"
    execution_root.mkdir()
    engineering_root.mkdir()
    monkeypatch.setitem(STORAGE_PREFLIGHT_CONTRACT, "mount_root", str(tmp_path))
    monkeypatch.setattr(os.path, "ismount", lambda path: Path(path) == tmp_path)

    fragment_size = 4096
    minimum_blocks = int(STORAGE_PREFLIGHT_CONTRACT["minimum_free_bytes"]) // fragment_size

    def filesystem(*, available_blocks: int) -> types.SimpleNamespace:
        return types.SimpleNamespace(
            f_bsize=fragment_size,
            f_frsize=fragment_size,
            f_bavail=available_blocks,
            f_bfree=minimum_blocks * 4,
            f_blocks=minimum_blocks * 8,
            f_favail=int(STORAGE_PREFLIGHT_CONTRACT["minimum_free_inodes"]),
            f_ffree=int(STORAGE_PREFLIGHT_CONTRACT["minimum_free_inodes"]) * 2,
            f_files=int(STORAGE_PREFLIGHT_CONTRACT["minimum_free_inodes"]) * 4,
            f_fsid=12345,
        )

    monkeypatch.setattr(os, "statvfs", lambda _path: filesystem(available_blocks=minimum_blocks))
    observation = _measure_storage_preflight(
        execution_root=execution_root,
        engineering_root=engineering_root,
        observed_engineering_path=engineering_root,
        engineering_absent=False,
    )
    assert observation["free_bytes"] == STORAGE_PREFLIGHT_CONTRACT["minimum_free_bytes"]
    assert observation["inodes_available"] == STORAGE_PREFLIGHT_CONTRACT["minimum_free_inodes"]

    monkeypatch.setattr(
        os,
        "statvfs",
        lambda _path: filesystem(available_blocks=minimum_blocks - 1),
    )
    with pytest.raises(FreezeError, match="byte or inode reserve"):
        _measure_storage_preflight(
            execution_root=execution_root,
            engineering_root=engineering_root,
            observed_engineering_path=engineering_root,
            engineering_absent=False,
        )
    assert PROFILE_SELECTOR_CONTRACT["storage_contract"]["maximum_persisted_evidence_bytes"] == 512 * 1024**2
    assert QUALIFICATION_PRODUCER_CONTRACT["temporary_pairwise_vector_ceiling_bytes"] == 1024**4
    assert QUALIFICATION_PRODUCER_CONTRACT["monotonic_wall_ceiling_seconds"] == 6_300
    assert QUALIFICATION_PRODUCER_CONTRACT["post_qualification_handoff_reserve_seconds"] == 600
    assert QUALIFICATION_PRODUCER_CONTRACT["minimum_provision_remaining_seconds_at_start"] == 57_360
    assert PRODUCER_CEILING_SECONDS == 6_300
    assert POST_QUALIFICATION_RESERVE_SECONDS == 600
    assert MINIMUM_PROVISION_REMAINING_AT_START_SECONDS == 57_360
    assert WALL_CEILING_SECONDS == 50_400
    assert H200_HOUR_CEILING == 56.0
    assert 61_200 - 57_360 == 3_840
    assert PROFILE_CONTRACT == QUALIFICATION_PROFILE_CONTRACT
    branches = PROFILE_SELECTOR_CONTRACT["qualification_branches"]
    assert branches["tuned_probe_passed_full"]["process_record_count"] == PROCESS_RECORD_COUNT == 64
    assert (
        branches["tuned_probe_passed_full"]["numeric_comparison_record_count"] == COMPARISON_RECORD_COUNT == 4
    )
    assert branches["tuned_capacity_fallback_baseline"] == {
        "process_record_count": BASELINE_FALLBACK_PROCESS_RECORD_COUNT,
        "numeric_comparison_record_count": BASELINE_FALLBACK_COMPARISON_RECORD_COUNT,
        "profiles": ["baseline"],
    }
    probe = PROFILE_SELECTOR_CONTRACT["tuned_capacity_probe"]
    assert probe["cell_receipt_count"] == 16
    assert probe["device_receipt_count"] == 4
    assert probe["expected_single_eval_scorer_prompt_count"] == 768
    assert tuple(probe["recoverable_disqualifiers"]) == TUNED_CAPACITY_DISQUALIFIERS
    assert probe["later_tuned_execution_failure_is_not_recoverable"] is True
    assert PROFILE_SELECTOR_CONTRACT["trainable_parameter_contract"]["trainable_numel"] == dict(
        TRAINABLE_NUMEL
    )
    assert (
        PROFILE_SELECTOR_CONTRACT["storage_contract"]["maximum_persisted_evidence_bytes"]
        == MAXIMUM_QUALIFICATION_EVIDENCE_BYTES
    )
    projection = PROFILE_SELECTOR_CONTRACT["projection"]
    assert set(projection) == {
        "action_continuation_bound",
        "maximum_projected_wall_seconds",
        "optimizer_update_timing",
        "profile_dependent_timing_order",
        "profile_independent_components",
        "projection_aggregation",
        "projection_includes",
        "projection_safety_multiplier",
        "timed_io_envelope",
        "timing_warmup",
        "training_host_envelope_aggregation",
        "training_host_tokenizer_envelopes",
        "training_timing_call_shapes",
        "training_timing_corpus",
        "whole_update_wall_time_includes",
    }
    assert not (set(projection) & set(PROFILE_SELECTOR_CONTRACT["tuned_requirements"]))
    timed_io = projection["timed_io_envelope"]
    assert "three_outcome_file_hash_scan_and_chmod" in timed_io
    assert "three_outcome_file_chmod_and_fsync" not in timed_io


def test_h200_data_order_and_catalog_contract_use_exact_registered_literals() -> None:
    assert DATA_ORDER_EQUIVALENCE_CONTRACT["digest"] == (
        "81e2beacf42b75fce89742a17c55615c34f368d6f4f8960478c28231931dbeb0"
    )
    assert DATA_ORDER_EQUIVALENCE_CONTRACT["registered_seed_count"] == 20
    assert DATA_ORDER_EQUIVALENCE_CONTRACT["updates"] == 1_000
    assert RUNPOD_PROVISIONING_CONTRACT["gpu_id"] == "NVIDIA H200"
    assert RUNPOD_PROVISIONING_CONTRACT["gpu_display_name"] == "H200 SXM"
    assert RUNPOD_PROVISIONING_CONTRACT["gpu_memory_catalog_gb"] == 141
    assert RUNPOD_PROVISIONING_CONTRACT["secure_price_ceiling_usd_per_gpu_hour"] == 4.59
    assert RUNPOD_PROVISIONING_CONTRACT["provision_ceiling_seconds"] == 61_200
    assert RUNPOD_PROVISIONING_CONTRACT["maximum_secure_cost_usd"] == 312.12
    assert RUNPOD_PROVISIONING_CONTRACT["create_wait"] is False
    assert RUNPOD_PROVISIONING_CONTRACT["readiness_poll_timeout_seconds"] == 900
    assert RUNPOD_PROVISIONING_CONTRACT["pre_handoff_failure_cleanup"] == {
        "command_argv_prefix": ["runpodctl", "pod", "delete"],
        "delete_every_discovered_pod_id": True,
        "exit_trap_required": True,
        "failed_transaction_resume_allowed": False,
        "create_without_wait_then_poll": True,
        "preserve_create_poll_get_ssh_and_delete_responses": True,
    }
    assert RUNPOD_PROVISIONING_CONTRACT["gpu_catalog_capture_argv"] == [
        "runpodctl",
        "gpu",
        "list",
        "--include-unavailable",
        "-o",
        "json",
    ]
    assert RUNPOD_PROVISIONING_CONTRACT["gpu_catalog_capture_file"] == ("runpod-gpu-catalog-response.json")
    assert RUNPOD_OPERATOR_HANDOFF_CONTRACT["whole_file_external_sha256_required"] is True
    assert RUNPOD_OPERATOR_HANDOFF_CONTRACT["manual_pod_fact_redeclaration"] is False
    assert (
        _sha256(ROOT / "pyproject.toml")
        == H200_MODULE_INVOCATION_CONTRACT["immutable_h100_parent_pyproject_sha256"]
    )
    assert H200_MODULE_INVOCATION_CONTRACT["h200_console_script_installed"] is False
    assert H200_MODULE_INVOCATION_CONTRACT["runtime_project_wheel_or_editable_install_performed"] is False
    assert "goalzendo-g00f-h200" not in (ROOT / "pyproject.toml").read_text(encoding="utf-8")


def _catalog_payload(*, stock: str = "Low", price: float = 4.59) -> list[dict[str, Any]]:
    return [
        {
            "available": True,
            "dataCenterAvailability": [
                {"dataCenterId": "US-CA-2", "stockStatus": stock},
                {"dataCenterId": "US-GA-2", "stockStatus": "Low"},
            ],
            "displayName": "H200 SXM",
            "gpuId": "NVIDIA H200",
            "memoryInGb": 141,
            "secureCloud": True,
            "securePricePerHr": price,
            "stockStatus": "Low",
        }
    ]


def _write_provider_evidence(directory: Path) -> dict[str, Path]:
    pod_id = "pod-1"
    pod_name = f"goalzendo-g00f-h200-{EXECUTION_UUID_FIXTURE}"
    ip = "38.80.152.148"
    port = 32836
    command = f"ssh -i /Users/reviewer/.runpod/ssh/runpodctl-ssh-key root@{ip} -p {port}"
    version_path = directory / "runpodctl-version.txt"
    catalog_path = directory / "runpod-gpu-catalog-response.json"
    create_path = directory / "runpod-create-response.json"
    api_path = directory / "runpod-api-response.json"
    ssh_path = directory / "runpod-ssh-info-response.json"
    identity_path = directory / "runpod-ssh-identity-receipt.json"
    version_path.write_text("runpodctl 2.9.0-c094cac\n", encoding="utf-8")
    catalog_path.write_text(json.dumps(_catalog_payload()), encoding="utf-8")
    create_path.write_text(
        json.dumps(
            {
                "id": pod_id,
                "name": pod_name,
                "imageName": "runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404",
                "desiredStatus": "RUNNING",
                "costPerHr": 18.36,
                "containerDiskInGb": 50,
                "volumeInGb": 0,
                "volumeMountPath": "/workspace",
                "gpuCount": 4,
                "memoryInGb": 1000,
                "vcpuCount": 96,
                "ports": "22/tcp",
                "lastStatusChange": "Rented by User",
                "env": [],
                "machine": {"gpuDisplayName": "H200 SXM", "location": "US"},
            }
        ),
        encoding="utf-8",
    )
    ssh = {
        "id": pod_id,
        "name": pod_name,
        "ip": ip,
        "port": port,
        "ssh_command": command,
        "ssh_key": {
            "exists": True,
            "fingerprint": "SHA256:fixture",
            "in_account": True,
            "path": "/Users/reviewer/.runpod/ssh/runpodctl-ssh-key",
            "source": "runpodctl doctor",
        },
    }
    api_path.write_text(
        json.dumps(
            {
                "id": pod_id,
                "name": pod_name,
                "desiredStatus": "RUNNING",
                "createdAt": "2026-08-12 01:00:00.000 +0000 UTC",
                "imageName": "runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404",
                "gpuCount": 4,
                "volumeInGb": 0,
                "containerDiskInGb": 50,
                "volumeMountPath": "/workspace",
                "costPerHr": 18.36,
                "machine": {
                    "gpuId": "NVIDIA H200",
                    "gpuDisplayName": "H200 SXM",
                    "dataCenterId": "US-CA-2",
                    "secureCloud": True,
                },
                "runtimeStatus": "running",
                "ssh": ssh,
            }
        ),
        encoding="utf-8",
    )
    ssh_path.write_text(json.dumps(ssh), encoding="utf-8")
    connection = {
        "id": pod_id,
        "ip": ip,
        "key_path": "/Users/reviewer/.runpod/ssh/runpodctl-ssh-key",
        "name": pod_name,
        "port": port,
        "ssh_command": command,
    }
    identity_body = {
        "schema": "goalzendo.g00f_h200_runpod_ssh_identity_receipt",
        "schema_version": 1,
        "api_response_sha256": _sha256(api_path),
        "ssh_info_sha256": _sha256(ssh_path),
        "pod_id": pod_id,
        "ip": ip,
        "port": port,
        "banner_prefix": "SSH-",
        "connect_timeout_seconds": 5,
        "read_timeout_seconds": 5,
        "probe_implementation": "python_socket_banner_then_batchmode_ssh_allowlisted_provider_identity",
        "authenticated_ssh": {
            "argv": _provider_identity_ssh_argv(connection),
            "exit_status": 0,
            "stdout_sha256": "1" * 64,
            "stderr_sha256": "2" * 64,
        },
        "provider_environment": {
            "RUNPOD_POD_ID": pod_id,
            "RUNPOD_DC_ID": "US-CA-2",
            "RUNPOD_POD_HOSTNAME": "fixture-pod-host",
            "RUNPOD_GPU_COUNT": "4",
            "RUNPOD_PUBLIC_IP": ip,
            "RUNPOD_TCP_PORT_22": str(port),
            "RUNPOD_VOLUME_ID": "9mut3tpzwd",
        },
        "accelerators": [
            {
                "host_ordinal": index,
                "memory_total_mib": 143771,
                "name": "NVIDIA H200",
                "uuid": f"GPU-00000000-0000-4000-8000-00000000000{index}",
            }
            for index in range(4)
        ],
        "observed_at_utc": "2026-08-12T01:01:00Z",
        "ready": True,
        "outcomes_seen": False,
        "g01_launch_authorized": False,
    }
    identity_path.write_text(
        json.dumps({**identity_body, "receipt_digest": semantic_digest(identity_body)}),
        encoding="utf-8",
    )
    return {
        "version": version_path,
        "catalog": catalog_path,
        "create": create_path,
        "get": api_path,
        "ssh": ssh_path,
        "identity": identity_path,
    }


def _verified_fixture(tmp_path: Path) -> VerifiedFreeze:
    return VerifiedFreeze(
        repo=tmp_path,
        path=tmp_path / "freeze.json",
        file_sha256="a" * 64,
        payload={
            "freeze_digest": "b" * 64,
            "runtime": {
                "image": "runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404",
                "data_center": "US-CA-2",
                "network_volume_id": "9mut3tpzwd",
                "network_volume_mount": "/workspace",
            },
        },
        plans={},
        candidate_plans={},
    )


def test_raw_catalog_is_derived_bound_and_unavailable_us_ca_2_fails_before_create(
    tmp_path: Path,
) -> None:
    external = tmp_path / "external"
    external.mkdir()
    catalog_path = external / "runpod-gpu-catalog-response.json"
    catalog_path.write_text(json.dumps(_catalog_payload()), encoding="utf-8")
    catalog = verify_runpod_gpu_catalog_snapshot(
        catalog_path=catalog_path,
        operator_capture_utc="2026-08-12T00:59:50Z",
    )
    assert catalog["gpu_catalog"] == {
        "display_name": "H200 SXM",
        "gpu_id": "NVIDIA H200",
        "memory_in_gb": 141,
    }
    assert catalog["operational_market_snapshot"]["stock_label"] == "Low"
    assert catalog["operational_market_snapshot"]["secure_price_usd_per_gpu_hour"] == 4.59

    catalog_path.write_text(json.dumps(_catalog_payload(stock="none")), encoding="utf-8")
    with pytest.raises(FreezeError, match="unavailable before pod creation"):
        verify_runpod_gpu_catalog_snapshot(
            catalog_path=catalog_path,
            operator_capture_utc="2026-08-12T00:59:50Z",
        )
    catalog_path.write_text(json.dumps(_catalog_payload(price=4.60)), encoding="utf-8")
    with pytest.raises(FreezeError, match="price changed"):
        verify_runpod_gpu_catalog_snapshot(
            catalog_path=catalog_path,
            operator_capture_utc="2026-08-12T00:59:50Z",
        )


def test_provision_receipt_derives_and_copies_raw_catalog_without_operator_market_args(
    tmp_path: Path,
) -> None:
    external = tmp_path / "external"
    bound = tmp_path / "bound"
    external.mkdir()
    evidence = _write_provider_evidence(external)
    receipt_path = external / "runpod-provision-receipt.json"
    verified = _verified_fixture(tmp_path)
    receipt = create_runpod_provision_receipt(
        verified=verified,
        raw_runpodctl_version_path=evidence["version"],
        raw_gpu_catalog_path=evidence["catalog"],
        catalog_operator_capture_utc="2026-08-12T00:59:50Z",
        execution_uuid=EXECUTION_UUID_FIXTURE,
        raw_create_response_path=evidence["create"],
        raw_api_response_path=evidence["get"],
        raw_ssh_info_path=evidence["ssh"],
        ssh_identity_receipt_path=evidence["identity"],
        terminate_after_utc="2026-08-12T17:55:00Z",
        output_path=receipt_path,
    )
    assert receipt["operational_market_snapshot"]["stock_label"] == "Low"
    bound.mkdir()
    rebound = bind_runpod_provision_receipt(
        verified=verified,
        input_path=receipt_path,
        output_path=bound / receipt_path.name,
        expected_receipt_sha256=_sha256(receipt_path),
        expected_pod_id="pod-1",
    )
    bound_catalog = bound / evidence["catalog"].name
    assert rebound["gpu_catalog_snapshot"]["file_sha256"] == _sha256(bound_catalog)
    assert bound_catalog.stat().st_mode & 0o777 == 0o400
    assert rebound["provider_allocation"]["get"]["gpu_id"] == "NVIDIA H200"
    assert rebound["provider_allocation"]["get"]["data_center"] == "US-CA-2"
    for name in (
        "runpodctl-version.txt",
        "runpod-ssh-info-response.json",
        "runpod-ssh-identity-receipt.json",
    ):
        assert (bound / name).stat().st_mode & 0o777 == 0o400


@pytest.mark.parametrize(
    ("evidence_name", "field_path", "replacement"),
    [
        ("create", ("imageName",), "wrong/image:tag"),
        ("create", ("gpuCount",), 3),
        ("create", ("machine", "gpuDisplayName"), "H100 SXM"),
        ("get", ("id",), "other-pod"),
        ("get", ("machine", "gpuDisplayName"), "H100 SXM"),
        ("get", ("machine", "dataCenterId"), "US-GA-2"),
        ("get", ("machine", "secureCloud"), False),
        ("get", ("runtimeStatus",), "initializing"),
        ("get", ("costPerHr",), 17.0),
        ("ssh", ("id",), "other-pod"),
        ("ssh", ("setup",), "not ready"),
        ("ssh", ("ssh_key", "in_account"), False),
    ],
)
def test_provision_receipt_rejects_semantically_mismatched_provider_bytes(
    tmp_path: Path,
    evidence_name: str,
    field_path: tuple[str, ...],
    replacement: Any,
) -> None:
    external = tmp_path / "external"
    external.mkdir()
    evidence = _write_provider_evidence(external)
    target = evidence[evidence_name]
    payload = json.loads(target.read_text(encoding="utf-8"))
    cursor = payload
    for key in field_path[:-1]:
        cursor = cursor[key]
    cursor[field_path[-1]] = replacement
    target.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(FreezeError):
        create_runpod_provision_receipt(
            verified=_verified_fixture(tmp_path),
            raw_runpodctl_version_path=evidence["version"],
            raw_gpu_catalog_path=evidence["catalog"],
            catalog_operator_capture_utc="2026-08-12T00:59:50Z",
            execution_uuid=EXECUTION_UUID_FIXTURE,
            raw_create_response_path=evidence["create"],
            raw_api_response_path=evidence["get"],
            raw_ssh_info_path=evidence["ssh"],
            ssh_identity_receipt_path=evidence["identity"],
            terminate_after_utc="2026-08-12T17:55:00Z",
            output_path=external / "runpod-provision-receipt.json",
        )


def test_canonical_runpod_create_is_no_wait_and_launcher_is_executable() -> None:
    argv = canonical_runpod_create_command(
        execution_uuid=EXECUTION_UUID_FIXTURE,
        terminate_after_utc="2026-08-12T17:55:00Z",
    )
    assert "--wait" not in argv
    assert "--wait-timeout" not in argv
    launcher = ROOT / "runs/goalzendo/run_g00f_frozen_4h200.sh"
    assert launcher.stat().st_mode & 0o777 == 0o755


def test_provision_receipt_rejects_rehashed_provider_volume_and_version_tamper(
    tmp_path: Path,
) -> None:
    evidence = _write_provider_evidence(tmp_path)
    evidence["version"].write_text("runpodctl 2.9.1-unknown\n", encoding="utf-8")
    with pytest.raises(FreezeError, match="version"):
        create_runpod_provision_receipt(
            verified=_verified_fixture(tmp_path),
            raw_runpodctl_version_path=evidence["version"],
            raw_gpu_catalog_path=evidence["catalog"],
            catalog_operator_capture_utc="2026-08-12T00:59:50Z",
            execution_uuid=EXECUTION_UUID_FIXTURE,
            raw_create_response_path=evidence["create"],
            raw_api_response_path=evidence["get"],
            raw_ssh_info_path=evidence["ssh"],
            ssh_identity_receipt_path=evidence["identity"],
            terminate_after_utc="2026-08-12T17:55:00Z",
            output_path=tmp_path / "receipt-version.json",
        )

    evidence["version"].write_text("runpodctl 2.9.0-c094cac\n", encoding="utf-8")
    identity = json.loads(evidence["identity"].read_text(encoding="utf-8"))
    identity["provider_environment"]["RUNPOD_VOLUME_ID"] = "wrong-volume"
    body = {key: value for key, value in identity.items() if key != "receipt_digest"}
    identity["receipt_digest"] = semantic_digest(body)
    evidence["identity"].write_text(json.dumps(identity), encoding="utf-8")
    with pytest.raises(FreezeError, match="provider-injected environment"):
        create_runpod_provision_receipt(
            verified=_verified_fixture(tmp_path),
            raw_runpodctl_version_path=evidence["version"],
            raw_gpu_catalog_path=evidence["catalog"],
            catalog_operator_capture_utc="2026-08-12T00:59:50Z",
            execution_uuid=EXECUTION_UUID_FIXTURE,
            raw_create_response_path=evidence["create"],
            raw_api_response_path=evidence["get"],
            raw_ssh_info_path=evidence["ssh"],
            ssh_identity_receipt_path=evidence["identity"],
            terminate_after_utc="2026-08-12T17:55:00Z",
            output_path=tmp_path / "receipt-volume.json",
        )


def test_detached_supervisor_receipt_replay_and_signal_cleanup(tmp_path: Path) -> None:
    supervisor = ROOT / "runs/goalzendo/g00f_h200_detached_supervisor.py"
    launcher = tmp_path / "launcher.sh"
    launcher.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    launcher.chmod(0o755)
    started = tmp_path / "started.json"
    terminal = tmp_path / "terminal.json"
    pid_file = tmp_path / "supervisor.pid"
    execution_root = Path("/workspace/status-goalzendo/g00f-executions") / EXECUTION_UUID_FIXTURE
    command = [
        sys.executable,
        str(supervisor),
        "--launcher",
        str(launcher),
        "--execution-uuid",
        EXECUTION_UUID_FIXTURE,
        "--execution-root",
        str(execution_root),
        "--operator-handoff-sha256",
        "1" * 64,
        "--execution-handoff-sha256",
        "2" * 64,
        "--pid-file",
        str(pid_file),
        "--started-receipt",
        str(started),
        "--terminal-receipt",
        str(terminal),
    ]
    completed = subprocess.run(command, check=False, timeout=10)
    assert completed.returncode == 0
    verified = VerifiedFreeze(
        repo=tmp_path,
        path=tmp_path / "freeze.json",
        file_sha256="a" * 64,
        payload={
            "controller_files": {
                "launcher": {"path": launcher.name, "sha256": _sha256(launcher)},
                "detached_supervisor": {
                    "path": supervisor.name,
                    "sha256": _sha256(supervisor),
                },
            }
        },
        plans={},
        candidate_plans={},
    )
    replay = verify_detached_supervisor_receipts(
        verified=verified,
        started_receipt_path=started,
        terminal_receipt_path=terminal,
        expected_execution_uuid=EXECUTION_UUID_FIXTURE,
        expected_execution_root=execution_root,
        expected_operator_handoff_sha256="1" * 64,
        expected_execution_handoff_sha256="2" * 64,
        expected_launcher_path=launcher,
        expected_supervisor_path=supervisor,
    )
    assert replay["success"] is True
    started_payload = json.loads(started.read_text(encoding="utf-8"))
    assert started_payload["guardian_pid"] > 1
    assert started_payload["guardian_protocol"] == ("ready_pipe_plus_parent_eof_launcher_group_cleanup_v1")
    clean_terminal = json.loads(terminal.read_text(encoding="utf-8"))
    assert clean_terminal["guardian_clean_stop"] is True
    assert clean_terminal["guardian_failed"] is False

    terminal.chmod(0o600)
    terminal_payload = json.loads(terminal.read_text(encoding="utf-8"))
    terminal_payload["launcher_exit_code"] = False
    terminal_body = {key: value for key, value in terminal_payload.items() if key != "receipt_digest"}
    terminal_payload["receipt_digest"] = semantic_digest(terminal_body)
    terminal.write_text(json.dumps(terminal_payload), encoding="utf-8")
    terminal.chmod(0o400)
    with pytest.raises(FreezeError, match="clean launcher completion"):
        verify_detached_supervisor_receipts(
            verified=verified,
            started_receipt_path=started,
            terminal_receipt_path=terminal,
            expected_execution_uuid=EXECUTION_UUID_FIXTURE,
            expected_execution_root=execution_root,
            expected_operator_handoff_sha256="1" * 64,
            expected_execution_handoff_sha256="2" * 64,
            expected_launcher_path=launcher,
            expected_supervisor_path=supervisor,
        )

    sleeper = tmp_path / "sleeper.sh"
    sleeper.write_text("#!/bin/sh\nsleep 60 &\nwait\n", encoding="utf-8")
    sleeper.chmod(0o755)
    signal_command = command.copy()
    signal_command[signal_command.index(str(launcher))] = str(sleeper)
    for old, new in (
        (str(pid_file), str(tmp_path / "signal.pid")),
        (str(started), str(tmp_path / "signal-started.json")),
        (str(terminal), str(tmp_path / "signal-terminal.json")),
    ):
        signal_command[signal_command.index(old)] = new
    process = subprocess.Popen(signal_command)
    signal_started = tmp_path / "signal-started.json"
    for _ in range(100):
        if signal_started.exists():
            break
        time.sleep(0.02)
    assert signal_started.exists()
    os.kill(process.pid, signal.SIGTERM)
    assert process.wait(timeout=10) == 128 + signal.SIGTERM
    signal_terminal = json.loads((tmp_path / "signal-terminal.json").read_text(encoding="utf-8"))
    assert signal_terminal["received_signal"] == signal.SIGTERM
    assert signal_terminal["sigterm_sent"] is True
    assert signal_terminal["descendants_clear"] is True
    assert signal_terminal["success"] is False

    parent_command = command.copy()
    parent_command[parent_command.index(str(launcher))] = str(sleeper)
    for old, new in (
        (str(pid_file), str(tmp_path / "parent-death.pid")),
        (str(started), str(tmp_path / "parent-death-started.json")),
        (str(terminal), str(tmp_path / "parent-death-terminal.json")),
    ):
        parent_command[parent_command.index(old)] = new
    parent = subprocess.Popen(parent_command)
    parent_started_path = tmp_path / "parent-death-started.json"
    for _ in range(200):
        if parent_started_path.exists():
            break
        time.sleep(0.02)
    assert parent_started_path.exists()
    parent_started = json.loads(parent_started_path.read_text(encoding="utf-8"))
    launcher_pgid = int(parent_started["launcher_process_group_id"])
    os.kill(parent.pid, signal.SIGKILL)
    assert parent.wait(timeout=5) == -signal.SIGKILL
    group_absent = False
    for _ in range(200):
        try:
            os.killpg(launcher_pgid, 0)
        except (ProcessLookupError, PermissionError):
            group_absent = True
            break
        time.sleep(0.02)
    assert group_absent, "guardian did not clear the launcher group after supervisor death"


def test_qualification_supervisor_guardian_normal_signal_and_parent_death(tmp_path: Path) -> None:
    supervisor = ROOT / "runs/goalzendo/g00f_h200_qualification_supervisor.py"
    controller = tmp_path / "controller.py"
    controller.write_text("raise SystemExit(0)\n", encoding="utf-8")
    execution_root = Path("/workspace/status-goalzendo/g00f-executions") / EXECUTION_UUID_FIXTURE

    def command(prefix: str, target: Path) -> list[str]:
        return [
            sys.executable,
            str(supervisor),
            "--execution-uuid",
            EXECUTION_UUID_FIXTURE,
            "--execution-root",
            str(execution_root),
            "--started-receipt",
            str(tmp_path / f"{prefix}-started.json"),
            "--term-receipt",
            str(tmp_path / f"{prefix}-term.json"),
            "--kill-receipt",
            str(tmp_path / f"{prefix}-kill.json"),
            "--terminal-receipt",
            str(tmp_path / f"{prefix}-terminal.json"),
            "--",
            sys.executable,
            str(target),
        ]

    completed = subprocess.run(command("normal", controller), check=False, timeout=10)
    assert completed.returncode == 0
    normal_started = json.loads((tmp_path / "normal-started.json").read_text(encoding="utf-8"))
    normal_terminal = json.loads((tmp_path / "normal-terminal.json").read_text(encoding="utf-8"))
    assert normal_started["ceiling_seconds"] == 6_300
    assert normal_started["term_grace_seconds"] == 30
    assert normal_started["guardian_pid"] > 1
    assert normal_started["guardian_protocol"] == (
        "independent_session_pipe_eof_or_monotonic_deadline_group_cleanup_v1"
    )
    assert normal_terminal["guardian_clean_stop"] is True
    assert normal_terminal["guardian_failed"] is False
    assert normal_terminal["success"] is True

    sleeper = tmp_path / "qualification-sleeper.py"
    sleeper.write_text("import time\ntime.sleep(60)\n", encoding="utf-8")
    signalled = subprocess.Popen(command("signal", sleeper))
    signal_started_path = tmp_path / "signal-started.json"
    for _ in range(200):
        if signal_started_path.exists():
            break
        time.sleep(0.02)
    assert signal_started_path.exists()
    os.kill(signalled.pid, signal.SIGTERM)
    assert signalled.wait(timeout=10) == 128 + signal.SIGTERM
    signal_terminal = json.loads((tmp_path / "signal-terminal.json").read_text(encoding="utf-8"))
    assert signal_terminal["received_signal"] == signal.SIGTERM
    assert signal_terminal["descendants_clear"] is True
    assert signal_terminal["guardian_clean_stop"] is True
    assert signal_terminal["success"] is False

    orphan_guard = subprocess.Popen(command("parent-death", sleeper))
    parent_started_path = tmp_path / "parent-death-started.json"
    for _ in range(200):
        if parent_started_path.exists():
            break
        time.sleep(0.02)
    assert parent_started_path.exists()
    parent_started = json.loads(parent_started_path.read_text(encoding="utf-8"))
    controller_pgid = int(parent_started["controller_process_group_id"])
    os.kill(orphan_guard.pid, signal.SIGKILL)
    assert orphan_guard.wait(timeout=5) == -signal.SIGKILL
    group_absent = False
    for _ in range(200):
        try:
            os.killpg(controller_pgid, 0)
        except (ProcessLookupError, PermissionError):
            group_absent = True
            break
        time.sleep(0.02)
    assert group_absent, "guardian did not clear the controller process group after parent death"


def test_watchdog_guardian_normal_stop_and_exact_launcher_death() -> None:
    watchdog = _load_watchdog_module()

    def sleeper() -> subprocess.Popen[bytes]:
        return subprocess.Popen(
            [sys.executable, "-c", "import time; time.sleep(60)"],
            start_new_session=True,
        )

    def run_guardian(*, normal_stop: bool) -> tuple[int, subprocess.Popen[bytes], subprocess.Popen[bytes]]:
        launcher = sleeper()
        watcher = sleeper()
        control_read, control_write = os.pipe()
        liveness_read, liveness_write = os.pipe()
        guardian_pid = os.fork()
        if guardian_pid == 0:  # pragma: no cover - asserted from the parent
            os.close(control_write)
            os.close(liveness_write)
            try:
                os.setsid()
                result = watchdog._guardian_main(
                    control_fd=control_read,
                    launcher_liveness_fd=liveness_read,
                    watchdog_pid=watcher.pid,
                    watchdog_pgid=os.getpgid(watcher.pid),
                    launcher_pid=launcher.pid,
                    launcher_pgid=os.getpgid(launcher.pid),
                    deadline_ns=time.monotonic_ns() + 5_000_000_000,
                    term_grace_seconds=0.05,
                    receipt_margin_seconds=0.05,
                    marker_lead_ns=50_000_000,
                )
            except BaseException:
                result = 125
            finally:
                os.close(control_read)
                os.close(liveness_read)
            os._exit(result)
        os.close(control_read)
        os.close(liveness_read)
        if normal_stop:
            os.write(control_write, b"N")
        else:
            # A real Linux run passes a pidfd.  A pipe EOF has the same
            # select-readiness semantics and deterministically exercises the
            # exact-launcher-death branch on macOS test hosts too.
            os.close(liveness_write)
            liveness_write = -1
        os.close(control_write)
        observed_pid, status = os.waitpid(guardian_pid, 0)
        if liveness_write >= 0:
            os.close(liveness_write)
        assert observed_pid == guardian_pid
        return os.waitstatus_to_exitcode(status), launcher, watcher

    normal_code, normal_launcher, normal_watcher = run_guardian(normal_stop=True)
    assert normal_code == 0
    assert normal_launcher.poll() is None
    assert normal_watcher.poll() is None
    for process in (normal_launcher, normal_watcher):
        os.killpg(process.pid, signal.SIGKILL)
        process.wait(timeout=5)

    death_code, dead_launcher, dead_watcher = run_guardian(normal_stop=False)
    assert death_code == 1
    # The guardian first sends TERM and then escalates after its frozen grace
    # period.  A process that has exited but has not yet been reaped can still
    # make the process group appear live, so either terminal signal is a valid
    # bounded-cleanup outcome.
    assert dead_launcher.wait(timeout=5) in {-signal.SIGTERM, -signal.SIGKILL}
    assert dead_watcher.wait(timeout=5) == -signal.SIGKILL


def test_watchdog_launcher_boundary_is_portable_and_final_gate_replays_guardian() -> None:
    launcher_path = ROOT / "runs/goalzendo/run_g00f_frozen_4h200.sh"
    watchdog_path = ROOT / "runs/goalzendo/g00f_h200_watchdog.py"
    launcher = launcher_path.read_text(encoding="utf-8")
    watchdog = watchdog_path.read_text(encoding="utf-8")
    evaluator = (ROOT / "src/goalzendo_g00f_h200/evaluator.py").read_text(encoding="utf-8")

    assert subprocess.run(["bash", "-n", str(launcher_path)], check=False).returncode == 0
    assert "coproc" not in launcher
    assert "mkfifo -m 0600" in launcher
    assert "WATCHDOG_STARTED_VERIFIED=1" in launcher
    assert "if (( WATCHDOG_STARTED_VERIFIED == 0 )); then" in launcher
    assert "pidfd_open" in watchdog
    assert "launcher_liveness_fd" in watchdog
    assert "stdout=subprocess.DEVNULL" in watchdog
    assert '"watchdog-guardian-terminal.json"' in evaluator
    assert "ready_pipe_pidfd_launcher_liveness_and_monotonic_group_cutoff_v1" in evaluator


def test_final_gate_replays_exact_watchdog_guardian_receipts(tmp_path: Path) -> None:
    execution_uuid = EXECUTION_UUID_FIXTURE
    verified = VerifiedFreeze(
        repo=tmp_path,
        path=tmp_path / "execution-freeze.json",
        file_sha256="a" * 64,
        payload={"freeze_digest": "b" * 64},
        plans={},
        candidate_plans={},
    )
    budget_start_ns = 1_000_000_000
    deadline_ns = budget_start_ns + WALL_CEILING_SECONDS * 1_000_000_000
    common = {
        "schema_version": 1,
        "execution_uuid": execution_uuid,
        "freeze_file_sha256": verified.file_sha256,
        "freeze_digest": verified.digest,
        "deadline_monotonic_ns": deadline_ns,
        "budget_start_file_sha256": "c" * 64,
        "watchdog_pid": 301,
        "watchdog_process_group_id": 301,
        "launcher_pid": 201,
        "launcher_process_group_id": 201,
        "guardian_pid": 401,
        "guardian_protocol": "ready_pipe_pidfd_launcher_liveness_and_monotonic_group_cutoff_v1",
        "term_grace_seconds": 30,
        "watchdog_receipt_margin_seconds": 5,
        "deadline_marker_lead_ns": 2_000_000_000,
        "outcome_metrics_read": False,
        "predictions_read": False,
        "g01_launch_authorized": False,
    }

    def write(name: str, schema: str, extra: dict[str, Any]) -> Path:
        body = {"schema": schema, **common, **extra}
        payload = {**body, "receipt_digest": semantic_digest(body)}
        path = tmp_path / name
        path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")
        path.chmod(0o400)
        return path

    write(
        "watchdog-started.json",
        "goalzendo.g00f_h200_watchdog_started",
        {"started_monotonic_ns": budget_start_ns + 10},
    )
    write(
        "watchdog-normal-stop.json",
        "goalzendo.g00f_h200_watchdog_normal_stop",
        {"stopped_monotonic_ns": budget_start_ns + 40},
    )
    terminal_path = write(
        "watchdog-guardian-terminal.json",
        "goalzendo.g00f_h200_watchdog_guardian_terminal",
        {
            "completed_monotonic_ns": budget_start_ns + 50,
            "deadline_triggered": False,
            "guardian_clean_stop": True,
            "guardian_exit_code": 0,
        },
    )
    replay = _verify_success_watchdog_lifecycle(
        execution_root=tmp_path,
        verified=verified,
        execution_uuid=execution_uuid,
        budget_start_file_sha256="c" * 64,
        budget_started_monotonic_ns=budget_start_ns,
        worker_started_monotonic_values=[budget_start_ns + 20],
        worker_completed_monotonic_values=[budget_start_ns + 30],
    )
    assert replay["guardian_protocol"] == common["guardian_protocol"]

    terminal = json.loads(terminal_path.read_text(encoding="utf-8"))
    terminal["guardian_clean_stop"] = False
    terminal_body = {key: value for key, value in terminal.items() if key != "receipt_digest"}
    terminal["receipt_digest"] = semantic_digest(terminal_body)
    terminal_path.chmod(0o600)
    terminal_path.write_text(json.dumps(terminal, sort_keys=True) + "\n", encoding="utf-8")
    terminal_path.chmod(0o400)
    with pytest.raises(EvaluationError, match="continuous 14-hour supervision"):
        _verify_success_watchdog_lifecycle(
            execution_root=tmp_path,
            verified=verified,
            execution_uuid=execution_uuid,
            budget_start_file_sha256="c" * 64,
            budget_started_monotonic_ns=budget_start_ns,
            worker_started_monotonic_values=[budget_start_ns + 20],
            worker_completed_monotonic_values=[budget_start_ns + 30],
        )


def test_exclusive_evidence_copy_is_byte_exact_and_never_overwrites(tmp_path: Path) -> None:
    source = tmp_path / "source.json"
    destination = tmp_path / "bound.json"
    source.write_bytes(b'{"prospective":true}\n')

    exclusive_copy(source, destination)
    assert destination.read_bytes() == source.read_bytes()
    assert _sha256(destination) == _sha256(source)
    assert destination.stat().st_mode & 0o777 == 0o400
    with pytest.raises(FreezeError, match="already exists"):
        exclusive_copy(source, destination)


def test_preexecution_gate_lists_both_candidate_families_without_creating_itt(
    tmp_path: Path,
) -> None:
    def row(profile: str, panel: str, index: int) -> dict[str, Any]:
        return {
            "panel_id": panel,
            "plan_key": f"{profile}-{panel}-plan",
            "run_id": f"{profile}-{panel}-run",
            "seed": 10_000 + index,
            "law_family": "parity",
            "training_view": "law_only",
            "worker_index": index,
            "worker_order": 0,
        }

    candidates = {
        profile: {
            panel: (row(profile, panel, index),) for index, panel in enumerate(("g00f-0p5b", "g00f-1p5b"))
        }
        for profile in ("baseline", "tuned")
    }
    verified = VerifiedFreeze(
        repo=tmp_path,
        path=tmp_path / "execution-freeze.json",
        file_sha256="a" * 64,
        payload={"freeze_digest": "b" * 64},
        plans=candidates["baseline"],
        candidate_plans=candidates,
    )
    result = create_preexecution_gate(
        verified=verified,
        assessment_output=tmp_path / "preexecution-assessment.json",
        gate_output=tmp_path / "preexecution-gate.json",
    )

    assessment = result["assessment"]
    assert assessment["overall_passed"] is False
    assert assessment["selected_profile"] is None
    assert {item["candidate_profile"] for item in assessment["intention_to_train"]} == {
        "baseline",
        "tuned",
    }
    assert all(item["intention_to_train"] is False for item in assessment["intention_to_train"])
    assert all(item["state"] == "candidate_not_selected_not_run" for item in assessment["intention_to_train"])
    assert result["gate"]["authorization"]["g01_launch_authorized"] is False


def test_launcher_qualifies_handoffs_selects_and_attests_cleanup_before_itt() -> None:
    launcher = (ROOT / "runs/goalzendo/run_g00f_frozen_4h200.sh").read_text(encoding="utf-8")
    storage_before_qualification = launcher.index(
        '--phase before_qualification --output "$STORAGE_PREFLIGHT_BEFORE_QUALIFICATION"'
    )
    producer = launcher.index('"$PYTHON_BIN" "$ROOT/runs/goalzendo/g00f_h200_qualification_supervisor.py"')
    handoff = launcher.index('bind-profile-qualification "${COMMON[@]}"')
    selection = launcher.index('select-profile "${COMMON[@]}"')
    storage_before_itt = launcher.index('--phase before_itt --output "$STORAGE_PREFLIGHT_BEFORE_ITT"')
    cleanup = launcher.index('record-profile-qualification-cleanup "${SELECTED_COMMON[@]}"')
    ledger = launcher.index('initialize-ledger "${SELECTED_COMMON[@]}"')
    worker = launcher.index('run-worker "${SELECTED_COMMON[@]}"')
    assert (
        storage_before_qualification
        < producer
        < handoff
        < selection
        < storage_before_itt
        < cleanup
        < ledger
        < worker
    )
    assert "qualification-report" not in launcher
    assert "--expected-profile-qualification-cleanup-sha256" in launcher
    assert "QUALIFICATION_ENGINEERING_ROOT" in launcher
    assert "must come from the independently pinned operator handoff" in launcher
    assert "print(uuid.uuid4())" not in launcher


def test_runbook_uses_one_sha_pinned_operator_handoff_and_minimal_stage() -> None:
    runbook = (ROOT / "docs/goalzendo/g00f-h200-execution-freeze.md").read_text(encoding="utf-8")
    assert "goalzendo.g00f_h200_operator_handoff" in runbook
    assert "EXPECTED_OPERATOR_HANDOFF_SHA256" in runbook
    assert 'POD_ID="$(jq -er .pod_id "$HANDOFF")"' in runbook
    assert 'CREATED_AT_UTC="$(jq -er .created_at_utc "$HANDOFF")"' in runbook
    assert 'TERMINATE_AFTER_UTC="$(jq -er .terminate_after_utc "$HANDOFF")"' in runbook
    assert 'SECURE_GPU_PRICE="$(jq -er .secure_price_usd_per_gpu_hour "$HANDOFF")"' in runbook
    assert 'STOCK_LABEL="$(jq -er .stock_label "$HANDOFF")"' in runbook
    assert "runpodctl gpu list --include-unavailable -o json" in runbook
    assert "runpod-gpu-catalog-response.json" in runbook
    assert "raw_gpu_catalog_response_sha256" in runbook
    assert "timedelta(hours=16, minutes=55)" in runbook
    assert "TERMINATE_AFTER_UTC=2026" not in runbook
    catalog_guard = runbook.index("verify-runpod-catalog")
    create_call = runbook.index("runpodctl pod create")
    failure_trap = runbook.index("cleanup_uncommitted_pod")
    delete_call = runbook.index('delete_and_confirm_pod "$POD_ID"')
    operator_handoff = runbook.index("> preflight/operator-handoff.json")
    committed = runbook.index("PROVISION_COMMITTED=1")
    assert catalog_guard < failure_trap < delete_call < create_call < operator_handoff < committed
    assert "Runpod no-wait create was ambiguous" in runbook
    assert "--wait-timeout" not in runbook
    assert runbook.count("## Exact runtime and operator-handoff replay") == 1
    assert 'EXECUTION_UUID="$(jq -er .execution_uuid "$HANDOFF")"' in runbook
    assert "goalzendo.g00f_h200_execution_handoff" in runbook
    assert '/usr/bin/nohup /usr/bin/setsid --fork --wait "$G00F_PYTHON" "$DETACHED_SUPERVISOR"' in runbook
    assert 'G00F_VENV="/workspace/.venvs/goalzendo-h200-$EXECUTION_UUID"' in runbook
    assert 'sha256sum -c "$EXECUTION_HANDOFF_SHA_FILE"' in runbook
    assert 'chmod 0400 "$EXECUTION_HANDOFF" "$EXECUTION_HANDOFF_SHA_FILE"' in runbook
    assert 'test "$(stat -c \'%a:%h\' "$LAUNCH_PID_FILE")" = "400:1"' in runbook
    assert "stat -c '%a:%h'" in runbook
    assert 'tail --pid="$SUPERVISOR_PID" -F "$LAUNCH_LOG"' in runbook
    assert "ready_pipe_plus_parent_eof_launcher_group_cleanup_v1" in runbook
    assert "verify-detached-supervisor" in runbook
    assert "runpod-final-preservation-manifest.json" in runbook
    assert '"$RUNPODCTL" version > "$HOST_PREFLIGHT/runpod-final-runpodctl-version.txt"' in runbook
    assert '"$RUNPODCTL" pod delete "$POD_ID" -o json' in runbook
    assert "runpod-final-deprovision-receipt.json" in runbook
    assert "for attempt in $(seq 1 1200); do" in runbook
    assert 'export RUNPOD_POD_ID="$POD_ID"' not in runbook
    executable_blocks = re.findall(r"```bash\n(.*?)\n```", runbook, flags=re.DOTALL)
    assert executable_blocks
    assert all(block.startswith("set -Eeuo pipefail\n") for block in executable_blocks)
    for index, block in enumerate(executable_blocks):
        parsed = subprocess.run(
            ["bash", "-n"],
            input=block,
            text=True,
            capture_output=True,
            check=False,
        )
        assert parsed.returncode == 0, f"runbook block {index} is invalid: {parsed.stderr}"
    assert '"/workspace/g00f-h200-stage-"+$execution_uuid' in runbook
    assert "stage the reviewed checkout" not in runbook
