from __future__ import annotations

import ast
import copy
import importlib
import json
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, replace
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest

import goalzendo_g00f_g01_bridge.bridge as bridge
from goalzendo_g00f_g01_bridge import (
    BridgeError,
    assert_exact_g01_spec,
    calculate_bridge_source_binding,
    create_route_lock,
    exact_g01_plan,
    exact_g01_shard,
    g01_binding,
    historical_g00e_binding,
    produce_eligibility,
    project_coordinator_token,
    verify_coordinator_token,
    verify_eligibility,
    verify_route_lock,
)

REPO = Path(__file__).resolve().parents[2]
H100_FREEZE = REPO / "reproducibility/goalzendo/g00f-execution-freeze-20260811/execution-freeze.json"
H100_PREEXEC_GATE = REPO / "reproducibility/goalzendo/g00f-execution-freeze-20260811/preexecution-gate.json"
H200_FREEZE = REPO / "reproducibility/goalzendo/g00f-h200-execution-freeze-20260811/execution-freeze.json"
H200_PREEXEC_GATE = (
    REPO / "reproducibility/goalzendo/g00f-h200-execution-freeze-20260811/preexecution-gate.json"
)
UUID_A = "11111111-1111-4111-8111-111111111111"
UUID_B = "22222222-2222-4222-8222-222222222222"


@pytest.fixture(autouse=True)
def _isolated_program_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    for name in tuple(sys.modules):
        if any(
            name == forbidden or name.startswith(f"{forbidden}.")
            for forbidden in bridge._THIN_FORBIDDEN_PROJECT_MODULES
        ):
            monkeypatch.delitem(sys.modules, name)
    monkeypatch.setattr(
        bridge,
        "_TEST_ONLY_G00F_PROGRAM_ROOT",
        tmp_path / "workspace" / "status-goalzendo" / "g00f-executions",
    )


def _source_digest() -> str:
    return str(calculate_bridge_source_binding(REPO)["source_digest"])


def _layout(tmp_path: Path, execution_uuid: str = UUID_A) -> tuple[Path, Path, Path, Path, Path]:
    del tmp_path
    program = bridge._program_root()
    execution = program / execution_uuid
    ledger = execution / "itt-ledger" / execution_uuid
    lock = program / "g00f-g01-route-lock.json"
    gate = (
        program.parents[1]
        / "analysis-goalzendo"
        / "g00f-executions"
        / execution_uuid
        / "g00f-final-gate.json"
    )
    return program, execution, ledger, lock, gate


def _repo_for(execution_uuid: str = UUID_A) -> Path:
    program = bridge._program_root()
    frozen = program / execution_uuid / "frozen-source"
    if not frozen.exists():
        frozen.parent.mkdir(parents=True, exist_ok=True)
        frozen.symlink_to(REPO, target_is_directory=True)
    return frozen


def _freeze_for(route: str, execution_uuid: str = UUID_A) -> Path:
    relative = (
        "reproducibility/goalzendo/g00f-execution-freeze-20260811/execution-freeze.json"
        if route == "h100"
        else "reproducibility/goalzendo/g00f-h200-execution-freeze-20260811/execution-freeze.json"
    )
    return _repo_for(execution_uuid) / relative


def _make_lock(
    tmp_path: Path,
    *,
    route: str = "h100",
    execution_uuid: str = UUID_A,
    freeze: Path | None = None,
    gate: Path | None = None,
) -> tuple[dict[str, Any], Path, Path, Path]:
    _program, _execution, ledger, lock, default_gate = _layout(tmp_path, execution_uuid)
    del freeze
    selected_freeze = _freeze_for(route, execution_uuid)
    selected_repo = _repo_for(execution_uuid)
    selected_sha = bridge.sha256_file(selected_freeze)
    result = create_route_lock(
        repo=selected_repo,
        route=route,  # type: ignore[arg-type]
        execution_uuid=execution_uuid,
        freeze_path=selected_freeze,
        expected_freeze_sha256=selected_sha,
        ledger_root=ledger,
        prospective_gate_output=gate or default_gate,
        expected_bridge_source_digest=_source_digest(),
        output=lock,
    )
    return result, ledger, lock, gate or default_gate


def _write_json(path: Path, value: Any, *, mode: int = 0o400) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        path.chmod(0o600)
    path.write_bytes(bridge.pretty_json_bytes(value))
    path.chmod(mode)


def test_g01_digest_names_and_exact_120_row_shard_union() -> None:
    binding = g01_binding(REPO)
    assert binding["config"] == {
        "path": str(REPO / "configs/goalzendo/g01_known_law.yaml"),
        "file_sha256": bridge.G01_CONFIG_FILE_SHA256,
        "canonical_digest": bridge.G01_CANONICAL_CONFIG_DIGEST,
    }
    target = binding["target_binding"]
    assert binding["target_binding_digest"] == bridge.G01_TARGET_BINDING_DIGEST
    assert target["scientific_config_digest"] == (
        "ca71ea55967c81f307df01f51278546988e06daa1a4bfc16ce0da4456a44c78b"
    )
    assert target["source_fingerprint"] == bridge.G01_SOURCE_FINGERPRINT
    assert len(target["authorized_cell_config_digests"]) == 12
    assert binding["guard_signature"] == bridge.G01_GUARD_SIGNATURE
    assert binding["model_identity"]["requested_revision"] == bridge.G01_MODEL_REVISION
    assert binding["plan"]["rows_digest"] == bridge.G01_PLAN_ROWS_DIGEST
    assert binding["plan"]["plan_key_set_digest"] == bridge.G01_PLAN_KEY_SET_DIGEST

    full = exact_g01_plan(REPO)
    shards = [exact_g01_shard(REPO, shard_index=index, num_shards=4) for index in range(4)]
    union = [spec for shard in shards for spec in shard]
    assert len(full) == len(union) == 120
    assert {spec.plan_key for spec in union} == {spec.plan_key for spec in full}
    assert len({spec.plan_key for spec in union}) == 120

    assert_exact_g01_spec(REPO, full[0])
    with pytest.raises(BridgeError, match="not one exact member"):
        assert_exact_g01_spec(REPO, replace(full[0], seed=999_983))


def test_historical_g00e_settings_are_bound_but_never_authorizing() -> None:
    current = g01_binding(REPO)
    historical = historical_g00e_binding(REPO, current)
    assert historical["overall_passed"] is False
    assert historical["g01_scientifically_eligible"] is False
    assert historical["direct_g01_launch_authorized"] is False
    assert historical["dedicated_global_coordinator_required"] is True
    assert historical["failed_checks"] == ["constrained_scorer", "rule_adapters"]
    assert historical["selected_optimizer_settings"] == bridge._EXPECTED_SELECTED_SETTINGS
    assert historical["target_binding_digest"] == bridge.G01_TARGET_BINDING_DIGEST


def test_route_lock_is_canonical_externally_pinned_and_single_route(tmp_path: Path) -> None:
    created, ledger, lock, gate = _make_lock(tmp_path)
    assert Path(created["path"]) == lock
    assert lock.stat().st_mode & 0o777 == 0o400
    verified = verify_route_lock(
        lock,
        repo=_repo_for(),
        expected_route_lock_sha256=str(created["file_sha256"]),
        expected_bridge_source_digest=_source_digest(),
    )
    assert verified["route"] == "h100"
    assert verified["execution_uuid"] == UUID_A
    assert verified["ledger_root"] == str(ledger)
    assert verified["prospective_eligibility_output"] == str(
        ledger.parents[2] / "g00f-g01-scientific-eligibility.json"
    )
    assert verified["prospective_coordinator_token_output"] == str(
        ledger.parents[2] / "g00f-g01-coordinator-input.json"
    )

    # A second H200 precommit in the same program root cannot coexist.
    with pytest.raises(BridgeError, match="overwrite"):
        create_route_lock(
            repo=_repo_for(UUID_B),
            route="h200",
            execution_uuid=UUID_B,
            freeze_path=_freeze_for("h200", UUID_B),
            expected_freeze_sha256=bridge.H200_FREEZE_FILE_SHA256,
            ledger_root=ledger.parents[2] / UUID_B / "itt-ledger" / UUID_B,
            prospective_gate_output=_layout(tmp_path, UUID_B)[4],
            expected_bridge_source_digest=_source_digest(),
            output=lock,
        )
    with pytest.raises(BridgeError, match="source-enforced execution layout"):
        create_route_lock(
            repo=_repo_for(UUID_B),
            route="h100",
            execution_uuid=UUID_B,
            freeze_path=_freeze_for("h100", UUID_B),
            expected_freeze_sha256=bridge.H100_FREEZE_FILE_SHA256,
            ledger_root=tmp_path / UUID_B / "itt-ledger" / UUID_B,
            prospective_gate_output=gate.parent / "other.json",
            expected_bridge_source_digest=_source_digest(),
            output=tmp_path / "arbitrary-lock.json",
        )


def test_source_enforced_layout_keeps_bridge_outputs_outside_execution_inventory(
    tmp_path: Path,
) -> None:
    program, execution, _ledger, _lock, expected_gate = _layout(tmp_path)
    repo = _repo_for()
    layout = bridge.execution_layout(repo, UUID_A, "h100")
    assert Path(layout["program_root"]) == program
    assert Path(layout["execution_root"]) == execution
    assert Path(layout["final_gate"]) == expected_gate
    assert expected_gate == (
        program.parents[1] / "analysis-goalzendo" / "g00f-executions" / UUID_A / "g00f-final-gate.json"
    )
    for key in ("route_lock", "final_gate", "eligibility", "coordinator_token"):
        assert not Path(layout[key]).is_relative_to(execution)
    assert {entry.name for entry in execution.iterdir()} == {bridge.FROZEN_SOURCE_DIRECTORY}
    assert Path("/workspace/status-goalzendo/g00f-executions") == bridge._CANONICAL_G00F_PROGRAM_ROOT
    with pytest.raises(BridgeError, match="exact /workspace status execution"):
        bridge.execution_layout(REPO, UUID_A, "h100")


def test_test_only_program_root_is_unreachable_outside_pytest_marker(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)
    with pytest.raises(BridgeError, match="forbidden outside pytest"):
        bridge._program_root()


def test_route_lock_refuses_late_itt_and_mode_or_hash_tamper(tmp_path: Path) -> None:
    _program, _execution, ledger, lock, gate = _layout(tmp_path)
    ledger.mkdir(parents=True)
    with pytest.raises(BridgeError, match="before the ITT ledger"):
        create_route_lock(
            repo=_repo_for(),
            route="h100",
            execution_uuid=UUID_A,
            freeze_path=_freeze_for("h100"),
            expected_freeze_sha256=bridge.H100_FREEZE_FILE_SHA256,
            ledger_root=ledger,
            prospective_gate_output=gate,
            expected_bridge_source_digest=_source_digest(),
            output=lock,
        )

    ledger.rmdir()
    created, _ledger, lock, _gate = _make_lock(tmp_path)
    lock.chmod(0o600)
    with pytest.raises(BridgeError, match="mode changed"):
        verify_route_lock(
            lock,
            repo=_repo_for(),
            expected_route_lock_sha256=str(created["file_sha256"]),
            expected_bridge_source_digest=_source_digest(),
        )
    lock.chmod(0o400)
    with pytest.raises(BridgeError, match=r"bytes.*changed"):
        verify_route_lock(
            lock,
            repo=_repo_for(),
            expected_route_lock_sha256="0" * 64,
            expected_bridge_source_digest=_source_digest(),
        )


def test_current_real_h100_false_gate_cannot_emit_sidecar(tmp_path: Path) -> None:
    created, ledger, lock, gate = _make_lock(tmp_path)
    gate.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(H100_PREEXEC_GATE, gate)
    gate.chmod(0o400)
    output = ledger.parents[2] / "g00f-g01-scientific-eligibility.json"
    with pytest.raises(BridgeError, match="byte/object-identical"):
        produce_eligibility(
            repo=_repo_for(),
            route="h100",
            route_lock_path=lock,
            expected_route_lock_sha256=str(created["file_sha256"]),
            freeze_path=_freeze_for("h100"),
            expected_freeze_sha256=bridge.H100_FREEZE_FILE_SHA256,
            final_gate_path=gate,
            expected_final_gate_sha256=bridge.sha256_file(gate),
            artifact_roots={
                "g00f-0p5b": "/workspace/artifacts-goalzendo/g00f-capability-repair-0p5b",
                "g00f-1p5b": "/workspace/artifacts-goalzendo/g00f-capability-repair-1p5b",
            },
            ledger_root=ledger,
            worker_result_receipts={
                index: ledger.parents[1] / f"worker-{index}-result.json" for index in range(4)
            },
            expected_provision_receipt_sha256="0" * 64,
            expected_pod_id="not-a-pod",
            expected_bridge_source_digest=_source_digest(),
            output=output,
        )
    assert not output.exists()


def test_current_real_h200_false_gate_and_missing_selection_cannot_emit_sidecar(
    tmp_path: Path,
) -> None:
    from goalzendo_g00f_h200.evaluator import verify_gate_artifact
    from goalzendo_g00f_h200.freeze import verify_freeze

    verified = verify_freeze(
        repo=REPO,
        freeze_path=H200_FREEZE,
        expected_freeze_sha256=bridge.H200_FREEZE_FILE_SHA256,
    )
    summary = verify_gate_artifact(
        H200_PREEXEC_GATE,
        verified=verified,
        expected_gate_sha256=bridge.sha256_file(H200_PREEXEC_GATE),
    )
    assert summary["overall_passed"] is False

    created, ledger, lock, gate = _make_lock(tmp_path, route="h200", freeze=H200_FREEZE)
    gate.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(H200_PREEXEC_GATE, gate)
    output = ledger.parents[2] / "g00f-g01-scientific-eligibility.json"
    with pytest.raises(BridgeError, match="profile selection"):
        produce_eligibility(
            repo=_repo_for(),
            route="h200",
            route_lock_path=lock,
            expected_route_lock_sha256=str(created["file_sha256"]),
            freeze_path=_freeze_for("h200"),
            expected_freeze_sha256=bridge.H200_FREEZE_FILE_SHA256,
            final_gate_path=gate,
            expected_final_gate_sha256=bridge.sha256_file(gate),
            artifact_roots={
                "g00f-0p5b": "/workspace/unused-0p5b",
                "g00f-1p5b": "/workspace/unused-1p5b",
            },
            ledger_root=ledger,
            worker_result_receipts={
                index: ledger.parents[1] / f"worker-{index}-result.json" for index in range(4)
            },
            expected_provision_receipt_sha256="0" * 64,
            expected_pod_id="not-a-pod",
            expected_bridge_source_digest=_source_digest(),
            output=output,
        )
    assert not output.exists()


@dataclass
class _FakeVerified:
    selected_profile: str | None = None
    selection_receipt: dict[str, Any] | None = None


@dataclass
class _CompleteVerified:
    repo: Path
    file_sha256: str
    digest: str
    plans: dict[str, tuple[dict[str, str], ...]]
    all_rows: tuple[dict[str, str], ...]
    selected_profile: str | None = None


def _passing_assessment(
    tmp_path: Path,
    *,
    count: int = 160,
    duplicate: bool = False,
    extra_seal: bool = False,
    missing_seal: bool = False,
    h200_profile: str | None = None,
) -> tuple[_FakeVerified, dict[str, Any]]:
    execution_uuid = UUID_A
    selection = (
        {"execution_uuid": execution_uuid, "selected_profile": h200_profile}
        if h200_profile is not None
        else None
    )
    verified = _FakeVerified(selected_profile=h200_profile, selection_receipt=selection)
    itt: list[dict[str, Any]] = []
    attempts: list[dict[str, Any]] = []
    unseal_rows: list[dict[str, Any]] = []
    for index in range(count):
        identity = 0 if duplicate and index == count - 1 else index
        key = f"plan-{identity:03d}"
        run_id = f"run-{identity:03d}"
        itt.append(
            {
                "plan_key": key,
                "run_id": run_id,
                "state": "complete",
                "attempt_state": "complete",
                "intention_to_train": True,
            }
        )
        seals: dict[str, Any] = {
            name: {
                "bytes": index + offset + 1,
                "path": str(tmp_path / f"{key}-{name}"),
                "sealed_mode": 0,
                "sha256": f"{index * 3 + offset:064x}",
            }
            for offset, name in enumerate(("metrics.jsonl", "predictions.jsonl", "summary.json"))
        }
        if missing_seal and index == 0:
            seals.pop("summary.json")
        if extra_seal and index == 0:
            seals["extra.json"] = {
                "bytes": 1,
                "path": str(tmp_path / "extra.json"),
                "sealed_mode": 0,
                "sha256": "f" * 64,
            }
        attempts.append({"plan_key": key, "state": "complete", "outcome_file_seals": seals})
        for name in ("metrics.jsonl", "predictions.jsonl", "summary.json"):
            if name in seals:
                unseal_rows.append(
                    {
                        "plan_key": key,
                        "run_id": run_id,
                        "name": name,
                        "bytes": seals[name]["bytes"],
                        "sha256": seals[name]["sha256"],
                        "unsealed_mode": 0o400,
                    }
                )
    unseal = tmp_path / "panel-unseal.json"
    _write_json(unseal, {"outcome_files": unseal_rows})
    ledger: dict[str, Any] = {
        "execution_uuid": execution_uuid,
        "ledger_root": str(tmp_path / execution_uuid),
    }
    assessment: dict[str, Any] = {
        "evidence_status": "complete",
        "overall_passed": True,
        "checks": {"all": {"passed": True}},
        "intention_to_train": itt,
        "attempt_ledger": {
            "rows": attempts,
            "ledger": ledger,
            "all_complete": True,
            "global_stop": None,
            "panel_unseal": {"path": str(unseal), "file_sha256": bridge.sha256_file(unseal)},
        },
    }
    if h200_profile is not None:
        assessment["selected_profile"] = h200_profile
        assessment["profile_selection"] = selection
        ledger["selected_profile"] = h200_profile
        ledger["profile_selection"] = selection
    return verified, assessment


def _complete_passing_fixture(
    tmp_path: Path,
    ledger_root: Path,
) -> tuple[
    _CompleteVerified,
    dict[str, Any],
    dict[str, Any],
    dict[str, Path],
    dict[int, Path],
    str,
]:
    execution_root = ledger_root.parents[1]
    artifact_roots = {panel: tmp_path / f"artifacts-{panel}" for panel in ("g00f-0p5b", "g00f-1p5b")}
    plans: dict[str, list[dict[str, str]]] = {panel: [] for panel in artifact_roots}
    all_rows: list[dict[str, str]] = []
    itt: list[dict[str, Any]] = []
    attempts: list[dict[str, Any]] = []
    unseal_rows: list[dict[str, Any]] = []
    ordinary_inputs = {
        "COMPLETE",
        "attempts/attempt-0001.json",
        "attempts/resolved-config-0001.yaml",
        "completion.json",
        "environment.json",
        "g00f-freeze-binding.json",
        "identity.json",
        "implementation.json",
        "manifests/dataset.json",
        "manifests/model.json",
        "manifests/tokenizer.json",
        "resolved_config.yaml",
        "status.json",
    }
    for index in range(160):
        panel = "g00f-0p5b" if index < 80 else "g00f-1p5b"
        plan_key = f"plan-{index:03d}"
        run_id = f"run-{index:03d}"
        run_path = artifact_roots[panel] / "runs" / plan_key / run_id
        for relative in sorted(ordinary_inputs):
            target = run_path / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(f"{plan_key}:{relative}\n".encode())
            target.chmod(0o400)
        seals: dict[str, dict[str, Any]] = {}
        for name in ("metrics.jsonl", "predictions.jsonl", "summary.json"):
            target = run_path / name
            target.write_bytes(f"{plan_key}:{name}\n".encode())
            target.chmod(0o400)
            seals[name] = {
                "bytes": target.stat().st_size,
                "path": str(target),
                "sealed_mode": 0,
                "sha256": bridge.sha256_file(target),
            }
            unseal_rows.append(
                {
                    "plan_key": plan_key,
                    "run_id": run_id,
                    "name": name,
                    "bytes": target.stat().st_size,
                    "sha256": bridge.sha256_file(target),
                    "unsealed_mode": 0o400,
                }
            )
        receipt_digests: dict[str, str] = {}
        for directory in ("starts", "terminals", "leases"):
            receipt = ledger_root / directory / f"{plan_key}.json"
            _write_json(receipt, {"plan_key": plan_key, "kind": directory})
            receipt_digests[directory] = bridge.sha256_file(receipt)
        attempts.append(
            {
                "plan_key": plan_key,
                "state": "complete",
                "start_file_sha256": receipt_digests["starts"],
                "terminal_file_sha256": receipt_digests["terminals"],
                "lease_file_sha256": receipt_digests["leases"],
                "outcome_file_seals": seals,
            }
        )
        itt.append(
            {
                "plan_key": plan_key,
                "run_id": run_id,
                "state": "complete",
                "attempt_state": "complete",
                "intention_to_train": True,
            }
        )
        frozen = {"artifact_path": str(run_path)}
        plans[panel].append(frozen)
        all_rows.append(frozen)

    started_unix_ns = time.time_ns()
    panel_unseal = ledger_root / "panel-unseal.json"
    _write_json(panel_unseal, {"outcome_files": unseal_rows})
    budget_start = ledger_root / "budget-start.json"
    _write_json(budget_start, {"started_unix_ns": started_unix_ns})
    ledger_file = ledger_root / "ledger.json"
    _write_json(ledger_file, {"execution_uuid": UUID_A})
    provision = execution_root / "runpod-provision.json"
    _write_json(provision, {"pod_id": "synthetic-pod"})
    workers: dict[int, Path] = {}
    for index in range(4):
        worker = execution_root / f"worker-{index}-result.json"
        _write_json(worker, {"worker_index": index, "state": "complete"})
        workers[index] = worker

    audits: dict[str, dict[str, Any]] = {}
    for panel in artifact_roots:
        snapshot = execution_root / f"snapshot-{panel}"
        snapshot.mkdir(parents=True)
        leaf = snapshot / "model.bin"
        leaf.write_bytes(panel.encode())
        leaf.chmod(0o444)
        receipt = execution_root / f"model-receipt-{panel}.json"
        _write_json(receipt, {"panel_id": panel, "snapshot_root": str(snapshot)})
        audits[panel] = {
            "model_snapshot_receipt": {
                "path": str(receipt),
                "file_sha256": bridge.sha256_file(receipt),
                "snapshot_root": str(snapshot),
            }
        }

    ledger = {
        "execution_uuid": UUID_A,
        "ledger_root": str(ledger_root),
        "path": str(ledger_file),
        "file_sha256": bridge.sha256_file(ledger_file),
        "budget_start": {"started_unix_ns": started_unix_ns},
        "budget_start_file_sha256": bridge.sha256_file(budget_start),
        "runpod_provision": {
            "path": str(provision),
            "file_sha256": bridge.sha256_file(provision),
        },
        "model_integration_audits": audits,
    }
    assessment: dict[str, Any] = {
        "assessment_digest": "a" * 64,
        "evidence_digest": "e" * 64,
        "evidence_status": "complete",
        "overall_passed": True,
        "checks": {"synthetic_complete_replay": {"passed": True}},
        "intention_to_train": itt,
        "attempt_ledger": {
            "rows": attempts,
            "ledger": ledger,
            "all_complete": True,
            "global_stop": None,
            "panel_unseal": {
                "path": str(panel_unseal),
                "file_sha256": bridge.sha256_file(panel_unseal),
            },
        },
    }
    gate_body = {
        "schema": "synthetic.g00f_gate",
        "overall_passed": True,
        "assessment": assessment,
    }
    gate = {**gate_body, "gate_digest": bridge.semantic_digest(gate_body)}
    verified = _CompleteVerified(
        repo=tmp_path / "synthetic-freeze-repo",
        file_sha256=bridge.H100_FREEZE_FILE_SHA256,
        digest=bridge.H100_FREEZE_DIGEST,
        plans={panel: tuple(rows) for panel, rows in plans.items()},
        all_rows=tuple(all_rows),
    )
    return (
        verified,
        assessment,
        gate,
        artifact_roots,
        workers,
        bridge.sha256_file(provision),
    )


@pytest.mark.parametrize(
    "mutation",
    ["159", "161", "duplicate", "479_seals", "481_seals"],
)
def test_passing_inventory_rejects_wrong_attempt_and_seal_cardinality(
    tmp_path: Path,
    mutation: str,
) -> None:
    verified, assessment = _passing_assessment(
        tmp_path,
        count=159 if mutation == "159" else 161 if mutation == "161" else 160,
        duplicate=mutation == "duplicate",
        missing_seal=mutation == "479_seals",
        extra_seal=mutation == "481_seals",
    )
    with pytest.raises(BridgeError):
        bridge._validate_passing_assessment(
            route="h100",
            verified=verified,
            assessment=assessment,
            execution_uuid=UUID_A,
        )


def test_h200_profile_mixing_and_h100_profile_injection_are_rejected(tmp_path: Path) -> None:
    verified, assessment = _passing_assessment(tmp_path, h200_profile="tuned")
    bridge._validate_passing_assessment(
        route="h200",
        verified=verified,
        assessment=assessment,
        execution_uuid=UUID_A,
    )
    assessment["attempt_ledger"]["ledger"]["selected_profile"] = "baseline"
    with pytest.raises(BridgeError, match="one consistent"):
        bridge._validate_passing_assessment(
            route="h200",
            verified=verified,
            assessment=assessment,
            execution_uuid=UUID_A,
        )

    h100_verified, h100_assessment = _passing_assessment(tmp_path / "h100")
    h100_assessment["selected_profile"] = "tuned"
    with pytest.raises(BridgeError, match="forbidden H200"):
        bridge._validate_passing_assessment(
            route="h100",
            verified=h100_verified,
            assessment=h100_assessment,
            execution_uuid=UUID_A,
        )


def test_positive_producer_to_full_replay_binds_complete_dependency_closure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    created, ledger_root, lock_path, gate_path = _make_lock(tmp_path)
    verified, assessment, gate, artifact_roots, workers, provision_sha = _complete_passing_fixture(
        tmp_path, ledger_root
    )
    verified.repo = _repo_for()
    _write_json(gate_path, gate)
    monkeypatch.setattr(bridge, "_verify_route_freeze", lambda **_kwargs: verified)
    monkeypatch.setattr(
        bridge,
        "_verify_route_gate",
        lambda **_kwargs: {"overall_passed": True},
    )
    monkeypatch.setattr(
        bridge,
        "_derive_route_assessment",
        lambda **_kwargs: (assessment, gate),
    )
    output = ledger_root.parents[2] / bridge.ELIGIBILITY_FILENAME
    produced = produce_eligibility(
        repo=_repo_for(),
        route="h100",
        route_lock_path=lock_path,
        expected_route_lock_sha256=str(created["file_sha256"]),
        freeze_path=_freeze_for("h100"),
        expected_freeze_sha256=bridge.H100_FREEZE_FILE_SHA256,
        final_gate_path=gate_path,
        expected_final_gate_sha256=bridge.sha256_file(gate_path),
        artifact_roots=artifact_roots,
        ledger_root=ledger_root,
        worker_result_receipts=workers,
        expected_provision_receipt_sha256=provision_sha,
        expected_pod_id="synthetic-pod",
        expected_bridge_source_digest=_source_digest(),
        output=output,
    )
    assert output.stat().st_mode & 0o777 == 0o400
    payload = bridge.strict_json(output, "positive eligibility")
    assert payload["evidence_files"]["run_evaluator_input_count"] == 2_560
    assert payload["evidence_files"]["attempt_receipt_count"] == 480
    assert payload["evidence_files"]["outcome_seal_count"] == 480
    assert payload["evidence_files"]["file_count"] >= 3_000

    eligible = verify_eligibility(
        output,
        repo=_repo_for(),
        expected_eligibility_sha256=str(produced["file_sha256"]),
        expected_route_lock_sha256=str(created["file_sha256"]),
        expected_bridge_source_digest=_source_digest(),
        evidence_level="full",
    )
    assert eligible.route == "h100"
    assert eligible.execution_uuid == UUID_A
    assert eligible.g01_scientifically_eligible is True
    assert eligible.direct_g01_launch_authorized is False

    token_path = ledger_root.parents[2] / bridge.COORDINATOR_TOKEN_FILENAME
    token_result = project_coordinator_token(
        output,
        repo=_repo_for(),
        expected_eligibility_sha256=str(produced["file_sha256"]),
        expected_route_lock_sha256=str(created["file_sha256"]),
        expected_bridge_source_digest=_source_digest(),
        output=token_path,
    )
    token = verify_coordinator_token(
        token_path,
        repo=_repo_for(),
        expected_token_sha256=str(token_result["file_sha256"]),
        expected_bridge_source_digest=_source_digest(),
    )
    assert token.g01_scientifically_eligible is True
    assert token.direct_g01_launch_authorized is False
    thin_payload = bridge.strict_json(token_path, "thin coordinator token")
    assert "selected_profile" not in json.dumps(thin_payload)

    # A self-consistently re-digested omission is still rejected because the
    # full verifier independently reconstructs the evaluator dependency set.
    mutated = copy.deepcopy(payload)
    mutated["evidence_files"]["files"].pop()
    mutated["evidence_files"]["file_count"] -= 1
    mutated["evidence_files"]["files_digest"] = bridge.semantic_digest(mutated["evidence_files"]["files"])
    mutated_body = {key: value for key, value in mutated.items() if key != "eligibility_digest"}
    mutated["eligibility_digest"] = bridge.semantic_digest(mutated_body)
    _write_json(output, mutated)
    with pytest.raises(BridgeError, match="complete replayed dependency closure"):
        verify_eligibility(
            output,
            repo=_repo_for(),
            expected_eligibility_sha256=bridge.sha256_file(output),
            expected_route_lock_sha256=str(created["file_sha256"]),
            expected_bridge_source_digest=_source_digest(),
            evidence_level="full",
        )


def test_portable_evidence_recheck_detects_metadata_and_same_size_byte_tamper(tmp_path: Path) -> None:
    evidence_file = tmp_path / "evidence.bin"
    evidence_file.write_bytes(b"abc")
    row = bridge._file_binding(evidence_file)
    manifest = {
        "files": [row],
        "file_count": 1,
        "files_digest": bridge.semantic_digest([row]),
        "outcome_seal_count": 480,
        "attempt_receipt_count": 480,
        "worker_result_receipt_count": 4,
        "run_evaluator_input_count": 2_560,
        "execution_root_direct_file_count": 1,
        "model_snapshot_leaf_count": 1,
    }
    bridge._verify_evidence_bindings(manifest, level="full")
    evidence_file.write_bytes(b"abcd")
    with pytest.raises(BridgeError, match="changed after replay"):
        bridge._verify_evidence_bindings(manifest, level="metadata")
    evidence_file.write_bytes(b"xyz")
    with pytest.raises(BridgeError, match="changed after replay"):
        bridge._verify_evidence_bindings(manifest, level="full")


@pytest.mark.parametrize(
    "logical_name",
    ["g00f-final-gate.json", "execution-freeze.json", "worker-0-result.json"],
)
def test_evidence_binding_rejects_leaf_symlinks_before_resolution(
    tmp_path: Path,
    logical_name: str,
) -> None:
    destination = tmp_path / "destination.json"
    destination.write_text("{}\n", encoding="utf-8")
    logical = tmp_path / logical_name
    logical.symlink_to(destination)
    with pytest.raises(BridgeError, match="symlink"):
        bridge._file_binding(logical)


def test_frozen_artifact_root_and_model_snapshot_root_reject_symlink_components(
    tmp_path: Path,
) -> None:
    real_artifacts = tmp_path / "real-artifacts"
    run = real_artifacts / "runs" / "plan" / "run"
    run.mkdir(parents=True)
    linked_artifacts = tmp_path / "canonical-artifacts"
    linked_artifacts.symlink_to(real_artifacts, target_is_directory=True)
    fake = SimpleNamespace(
        plans={
            "g00f-0p5b": ({"artifact_path": str(linked_artifacts / "runs/plan/run")},),
            "g00f-1p5b": ({"artifact_path": str(linked_artifacts / "runs/plan/run")},),
        }
    )
    with pytest.raises(BridgeError, match="symlink component"):
        bridge._artifact_roots_from_verified(fake)

    real_snapshot = tmp_path / "real-snapshot"
    real_snapshot.mkdir()
    linked_snapshot = tmp_path / "canonical-snapshot"
    linked_snapshot.symlink_to(real_snapshot, target_is_directory=True)
    with pytest.raises(BridgeError, match="symlink component"):
        bridge._require_no_symlink_components(linked_snapshot, "G00-F model snapshot root")


@dataclass
class _VerifierFreeze:
    file_sha256: str
    digest: str
    plans: dict[str, tuple[dict[str, str], ...]]
    selected_profile: str | None = None


def _synthetic_eligibility(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[Path, str, str, str, dict[str, Any]]:
    created, ledger, lock_path, gate_path = _make_lock(tmp_path)
    lock = verify_route_lock(
        lock_path,
        repo=_repo_for(),
        expected_route_lock_sha256=str(created["file_sha256"]),
        expected_bridge_source_digest=_source_digest(),
    )
    assessment = {"assessment_digest": "a" * 64, "evidence_digest": "e" * 64}
    gate = {"gate_digest": "g" * 64, "assessment": assessment}
    _write_json(gate_path, gate)
    roots = {
        "g00f-0p5b": "/workspace/artifacts-goalzendo/g00f-capability-repair-0p5b",
        "g00f-1p5b": "/workspace/artifacts-goalzendo/g00f-capability-repair-1p5b",
    }
    fake = _VerifierFreeze(
        file_sha256=bridge.H100_FREEZE_FILE_SHA256,
        digest=bridge.H100_FREEZE_DIGEST,
        plans={
            panel: (
                {
                    "artifact_path": f"{root}/g00f/fake/run",
                },
            )
            for panel, root in roots.items()
        },
    )
    monkeypatch.setattr(bridge, "_verify_route_freeze", lambda **_kwargs: fake)
    monkeypatch.setattr(
        bridge,
        "_verify_route_gate",
        lambda **_kwargs: {"overall_passed": True},
    )
    monkeypatch.setattr(bridge, "_verify_evidence_bindings", lambda *_args, **_kwargs: None)
    g01 = g01_binding(_repo_for())
    body = {
        "schema": bridge.ELIGIBILITY_SCHEMA,
        "schema_version": bridge.ELIGIBILITY_SCHEMA_VERSION,
        "study_id": "g01",
        "produced_at_utc": "2026-08-12T00:00:00Z",
        "bridge_source": calculate_bridge_source_binding(_repo_for()),
        "route_lock": lock,
        "historical_g00e": historical_g00e_binding(_repo_for(), g01),
        "g00f": {
            "route": "h100",
            "execution_uuid": UUID_A,
            "freeze_file_sha256": bridge.H100_FREEZE_FILE_SHA256,
            "freeze_digest": bridge.H100_FREEZE_DIGEST,
            "final_gate": {
                "path": str(gate_path),
                "file_sha256": bridge.sha256_file(gate_path),
                "gate_digest": gate["gate_digest"],
                "assessment_digest": assessment["assessment_digest"],
                "evidence_digest": assessment["evidence_digest"],
            },
            "artifact_roots": roots,
            "ledger_root": str(ledger),
            "worker_result_receipts": {
                str(index): str(ledger.parents[1] / f"worker-{index}-result.json") for index in range(4)
            },
            "provision_receipt_sha256": "a" * 64,
            "pod_id": "synthetic-pod",
            "selected_profile": None,
            "profile_selection": None,
            "inventory": {
                "intention_to_train_rows": 160,
                "attempt_rows": 160,
                "outcome_seals": 480,
                "panel_unseal_rows": 480,
                "all_checks_passed": True,
            },
        },
        "evidence_files": {
            "files": [],
            "file_count": 3_000,
            "files_digest": bridge.semantic_digest([]),
            "outcome_seal_count": 480,
            "attempt_receipt_count": 480,
            "worker_result_receipt_count": 4,
            "run_evaluator_input_count": 2_560,
            "execution_root_direct_file_count": 1,
            "model_snapshot_leaf_count": 1,
        },
        "g01": g01,
        "eligibility": {
            "g00f_remediation_passed": True,
            "g01_scientifically_eligible": True,
            "direct_g01_launch_authorized": False,
            "dedicated_global_coordinator_required": True,
            "scope": "exact_unchanged_g01_primary_120_run_plan_only",
            "excluded_experiment_ids": ["g01a", "g01l", "g02"],
            "config_override_eligible": False,
            "backend_override_eligible": False,
            "smoke_or_partial_plan_eligible": False,
        },
    }
    payload = {**body, "eligibility_digest": bridge.semantic_digest(body)}
    sidecar = ledger.parents[2] / bridge.ELIGIBILITY_FILENAME
    _write_json(sidecar, payload)
    return sidecar, bridge.sha256_file(sidecar), str(created["file_sha256"]), _source_digest(), payload


def test_sidecar_whole_file_mode_overwrite_and_mutation_guards(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sidecar, sidecar_sha, lock_sha, source_digest, payload = _synthetic_eligibility(
        tmp_path,
        monkeypatch,
    )
    verified = verify_eligibility(
        sidecar,
        repo=_repo_for(),
        expected_eligibility_sha256=sidecar_sha,
        expected_route_lock_sha256=lock_sha,
        expected_bridge_source_digest=source_digest,
        evidence_level="metadata",
    )
    assert verified.route == "h100"

    sidecar.chmod(0o600)
    with pytest.raises(BridgeError, match="mode changed"):
        verify_eligibility(
            sidecar,
            repo=_repo_for(),
            expected_eligibility_sha256=sidecar_sha,
            expected_route_lock_sha256=lock_sha,
            expected_bridge_source_digest=source_digest,
            evidence_level="metadata",
        )
    sidecar.chmod(0o400)
    with pytest.raises(BridgeError, match=r"bytes.*changed"):
        verify_eligibility(
            sidecar,
            repo=_repo_for(),
            expected_eligibility_sha256="0" * 64,
            expected_route_lock_sha256=lock_sha,
            expected_bridge_source_digest=source_digest,
            evidence_level="metadata",
        )
    with pytest.raises(BridgeError, match="overwrite"):
        bridge._exclusive_json(sidecar, payload)


def test_sidecar_production_time_is_strict_utc(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sidecar, _sha, lock_sha, source_digest, payload = _synthetic_eligibility(
        tmp_path,
        monkeypatch,
    )
    mutated = copy.deepcopy(payload)
    mutated["produced_at_utc"] = "not-a-time"
    body = {key: value for key, value in mutated.items() if key != "eligibility_digest"}
    mutated["eligibility_digest"] = bridge.semantic_digest(body)
    _write_json(sidecar, mutated)
    with pytest.raises(BridgeError, match="production time"):
        verify_eligibility(
            sidecar,
            repo=_repo_for(),
            expected_eligibility_sha256=bridge.sha256_file(sidecar),
            expected_route_lock_sha256=lock_sha,
            expected_bridge_source_digest=source_digest,
            evidence_level="metadata",
        )


@pytest.mark.parametrize(
    ("section", "key", "replacement"),
    [
        ("g00f", "route", "h200"),
        ("g00f", "execution_uuid", UUID_B),
        ("g01", "guard", "mutated"),
        ("g01", "guard_signature", "0" * 64),
        ("g01", "model_identity", {"requested_model": "x", "requested_revision": "y"}),
        ("g01", "config", {"path": "x", "file_sha256": "0" * 64}),
        ("g01", "plan", {"planned_runs": 120, "rows_digest": "0" * 64}),
        ("g01", "target_binding_digest", "0" * 64),
        ("historical_g00e", "selected_settings_digest", "0" * 64),
        ("historical_g00e", "overall_passed", True),
        ("bridge_source", "source_digest", "0" * 64),
        ("eligibility", "direct_g01_launch_authorized", True),
        ("eligibility", "g01_launch_authorized", True),
    ],
)
def test_recomputed_sidecar_cannot_mutate_route_g00e_g01_model_settings_guard_or_code(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    section: str,
    key: str,
    replacement: Any,
) -> None:
    sidecar, _sha, lock_sha, source_digest, payload = _synthetic_eligibility(tmp_path, monkeypatch)
    mutated = copy.deepcopy(payload)
    mutated[section][key] = replacement
    mutated_body = {name: value for name, value in mutated.items() if name != "eligibility_digest"}
    mutated["eligibility_digest"] = bridge.semantic_digest(mutated_body)
    _write_json(sidecar, mutated)
    with pytest.raises(BridgeError):
        verify_eligibility(
            sidecar,
            repo=_repo_for(),
            expected_eligibility_sha256=bridge.sha256_file(sidecar),
            expected_route_lock_sha256=lock_sha,
            expected_bridge_source_digest=source_digest,
            evidence_level="metadata",
        )


def test_full_verifier_replays_and_rejects_forged_passing_gate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sidecar, sidecar_sha, lock_sha, source_digest, _payload = _synthetic_eligibility(
        tmp_path,
        monkeypatch,
    )
    false_assessment = {"overall_passed": False, "assessment_digest": "f" * 64}
    monkeypatch.setattr(
        bridge,
        "_derive_route_assessment",
        lambda **_kwargs: (false_assessment, {"overall_passed": False}),
    )
    with pytest.raises(BridgeError, match="does not replay"):
        verify_eligibility(
            sidecar,
            repo=_repo_for(),
            expected_eligibility_sha256=sidecar_sha,
            expected_route_lock_sha256=lock_sha,
            expected_bridge_source_digest=source_digest,
            evidence_level="full",
        )


def _synthetic_coordinator_token() -> tuple[Path, str, dict[str, Any]]:
    repo = _repo_for()
    program, _execution, _ledger, _lock, _gate = _layout(Path("unused"))
    body = {
        "schema": bridge.COORDINATOR_TOKEN_SCHEMA,
        "schema_version": bridge.COORDINATOR_TOKEN_SCHEMA_VERSION,
        "study_id": "g01",
        "produced_at_utc": "2026-08-12T00:00:00Z",
        "eligibility": {
            "g01_scientifically_eligible": True,
            "direct_g01_launch_authorized": False,
            "dedicated_global_coordinator_required": True,
            "scope": "exact_unchanged_g01_primary_120_run_plan_only",
        },
        "eligibility_sidecar": {
            "file_sha256": "a" * 64,
            "eligibility_digest": "b" * 64,
        },
        "route_lock": {
            "file_sha256": "c" * 64,
            "route_lock_digest": "d" * 64,
        },
        "bridge_source_digest": _source_digest(),
        "g00f_execution": {"route": "h100", "execution_uuid": UUID_A},
        "g01_identity": bridge._thin_g01_identity(g01_binding(repo)),
    }
    payload = {**body, "token_digest": bridge.semantic_digest(body)}
    token = program / bridge.COORDINATOR_TOKEN_FILENAME
    _write_json(token, payload)
    return token, bridge.sha256_file(token), payload


def test_thin_token_verifier_never_opens_detailed_eligibility(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    token, token_sha, _payload = _synthetic_coordinator_token()
    eligibility = _layout(Path("unused"))[0] / bridge.ELIGIBILITY_FILENAME
    original_strict_json = bridge.strict_json
    original_import_module = bridge.importlib.import_module

    def token_only_strict_json(path: str | Path, label: str) -> dict[str, Any]:
        assert Path(path) != eligibility
        return original_strict_json(path, label)

    monkeypatch.setattr(bridge, "strict_json", token_only_strict_json)

    def no_backend_import(name: str, package: str | None = None) -> Any:
        if name in bridge._THIN_FORBIDDEN_PROJECT_MODULES:
            raise AssertionError(f"backend import attempted: {name}")
        return original_import_module(name, package)

    monkeypatch.setattr(bridge.importlib, "import_module", no_backend_import)
    monkeypatch.setattr(
        bridge,
        "verify_eligibility",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("detailed verifier called")),
    )
    verified = verify_coordinator_token(
        token,
        repo=_repo_for(),
        expected_token_sha256=token_sha,
        expected_bridge_source_digest=_source_digest(),
    )
    assert verified.eligibility_file_sha256 == "a" * 64
    assert not eligibility.exists()


@pytest.mark.parametrize("module_name", sorted(bridge._THIN_FORBIDDEN_PROJECT_MODULES))
def test_thin_token_rejects_preloaded_backend_or_model_module(
    monkeypatch: pytest.MonkeyPatch,
    module_name: str,
) -> None:
    token, token_sha, _payload = _synthetic_coordinator_token()
    source_digest = _source_digest()
    monkeypatch.setitem(sys.modules, module_name, ModuleType(module_name))
    with pytest.raises(BridgeError, match="must precede"):
        verify_coordinator_token(
            token,
            repo=_repo_for(),
            expected_token_sha256=token_sha,
            expected_bridge_source_digest=source_digest,
        )


@pytest.mark.parametrize(
    "mutation",
    ["accuracy", "outcome_metrics", "selected_profile", "arbitrary_number"],
)
def test_thin_token_recursively_rejects_detailed_or_numerical_fields(mutation: str) -> None:
    token, _token_sha, payload = _synthetic_coordinator_token()
    mutated = copy.deepcopy(payload)
    if mutation == "arbitrary_number":
        mutated["g01_identity"]["observation"] = 0.125
    else:
        mutated["g01_identity"]["nested"] = {mutation: "redacted"}
    body = {key: value for key, value in mutated.items() if key != "token_digest"}
    mutated["token_digest"] = bridge.semantic_digest(body)
    _write_json(token, mutated)
    with pytest.raises(BridgeError, match="forbidden"):
        verify_coordinator_token(
            token,
            repo=_repo_for(),
            expected_token_sha256=bridge.sha256_file(token),
            expected_bridge_source_digest=_source_digest(),
        )


@pytest.mark.parametrize("field", ["direct_g01_launch_authorized", "g01_launch_authorized"])
def test_thin_token_rejects_any_true_launch_field(field: str) -> None:
    token, _token_sha, payload = _synthetic_coordinator_token()
    mutated = copy.deepcopy(payload)
    mutated["eligibility"][field] = True
    body = {key: value for key, value in mutated.items() if key != "token_digest"}
    mutated["token_digest"] = bridge.semantic_digest(body)
    _write_json(token, mutated)
    with pytest.raises(BridgeError):
        verify_coordinator_token(
            token,
            repo=_repo_for(),
            expected_token_sha256=bridge.sha256_file(token),
            expected_bridge_source_digest=_source_digest(),
        )


@pytest.mark.parametrize(
    "forbidden",
    [
        "--config",
        "--backend",
        "--smoke",
        "--output-root",
        "--set",
        "--worker-index",
        "--coordinator-token",
    ],
)
def test_dedicated_wrapper_has_no_override_backend_smoke_or_arbitrary_config(
    forbidden: str,
) -> None:
    command = [
        sys.executable,
        str(REPO / "runs/goalzendo/run_g01_after_g00f_bridge.py"),
        forbidden,
        "x",
    ]
    result = subprocess.run(
        command,
        cwd=REPO,
        env={**os.environ, "PYTHONPATH": str(REPO / "src")},
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 2


def test_cli_has_no_root_evidence_or_output_path_override_surface() -> None:
    import goalzendo_g00f_g01_bridge.cli as cli

    parser = cli.build_parser()
    route_lock = [
        "--repo",
        str(REPO),
        "route-lock",
        "--route",
        "h100",
        "--execution-uuid",
        UUID_A,
        "--expected-freeze-sha256",
        "a" * 64,
        "--expected-bridge-source-digest",
        "b" * 64,
    ]
    assess = [
        "--repo",
        str(REPO),
        "assess-eligibility",
        "--route",
        "h100",
        "--execution-uuid",
        UUID_A,
        "--expected-route-lock-sha256",
        "a" * 64,
        "--expected-freeze-sha256",
        "b" * 64,
        "--expected-final-gate-sha256",
        "c" * 64,
        "--expected-provision-receipt-sha256",
        "d" * 64,
        "--expected-pod-id",
        "pod",
        "--expected-bridge-source-digest",
        "e" * 64,
    ]
    for argv, forbidden in (
        (route_lock, "--output"),
        (route_lock, "--ledger-root"),
        (route_lock, "--prospective-gate-output"),
        (assess, "--final-gate"),
        (assess, "--artifact-root"),
        (assess, "--worker-result-0"),
    ):
        with pytest.raises(SystemExit):
            parser.parse_args([*argv, forbidden, "/tmp/forbidden"])


def test_wrapper_has_no_execution_capable_import_or_surface() -> None:
    path = REPO / "runs/goalzendo/run_g01_after_g00f_bridge.py"
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    imported = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.Import, ast.ImportFrom))
        for alias in node.names
    }
    function_names = {node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)}
    assert not imported.intersection(
        {"RunStore", "execute_plan", "load_backend", "goalzendo.artifacts", "goalzendo.runner"}
    )
    assert not function_names.intersection(
        {"_execute_backend", "_execute_exact_spec", "execute_plan", "run_one"}
    )
    for forbidden in ("RunStore", "load_backend", "_execute_exact_spec", ".initialize(", ".finalize("):
        assert forbidden not in source


def test_wrapper_verifies_thin_identity_then_always_refuses_execution(
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from runs.goalzendo import run_g01_after_g00f_bridge as wrapper

    monkeypatch.setattr(wrapper, "require_entrypoint_file", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        wrapper,
        "verify_coordinator_token",
        lambda *_args, **_kwargs: SimpleNamespace(g01_scientifically_eligible=True),
    )
    monkeypatch.setattr(wrapper, "exact_g01_plan", lambda _repo: tuple(range(120)))
    result = wrapper.main(
        [
            "--repo",
            str(_repo_for()),
            "--expected-coordinator-token-sha256",
            "a" * 64,
            "--expected-bridge-source-digest",
            _source_digest(),
        ]
    )
    assert result == 2
    assert "direct G01 execution is prohibited" in capsys.readouterr().err


def test_wrapper_real_thin_path_never_imports_backend(
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from runs.goalzendo import run_g01_after_g00f_bridge as wrapper

    _token, token_sha, _payload = _synthetic_coordinator_token()
    original_import_module = bridge.importlib.import_module

    def no_backend_import(name: str, package: str | None = None) -> Any:
        if name in bridge._THIN_FORBIDDEN_PROJECT_MODULES:
            raise AssertionError(f"backend import attempted: {name}")
        return original_import_module(name, package)

    monkeypatch.setattr(bridge.importlib, "import_module", no_backend_import)
    result = wrapper.main(
        [
            "--repo",
            str(_repo_for()),
            "--expected-coordinator-token-sha256",
            token_sha,
            "--expected-bridge-source-digest",
            _source_digest(),
        ]
    )
    assert result == 2
    assert "direct G01 execution is prohibited" in capsys.readouterr().err


def test_fresh_process_token_and_stub_paths_precede_backend_and_model_imports() -> None:
    _token, token_sha, _payload = _synthetic_coordinator_token()
    program, _execution, _ledger, _lock, _gate = _layout(Path("unused"))
    script = r"""
import os
import sys
from pathlib import Path

import goalzendo_g00f_g01_bridge.bridge as bridge

program = Path(sys.argv[1])
repo = Path(sys.argv[2])
token_sha = sys.argv[3]
source_digest = sys.argv[4]
bridge._TEST_ONLY_G00F_PROGRAM_ROOT = program
forbidden = (
    "goalzendo.experiment",
    "goalzendo.hf",
    "goalzendo.modeling",
    "goalzendo.training",
    "peft",
    "torch",
    "transformers",
)
assert not any(name == prefix or name.startswith(prefix + ".") for name in sys.modules for prefix in forbidden)
verified = bridge.verify_coordinator_token(
    program / bridge.COORDINATOR_TOKEN_FILENAME,
    repo=repo,
    expected_token_sha256=token_sha,
    expected_bridge_source_digest=source_digest,
)
assert verified.g01_scientifically_eligible is True
assert not any(name == prefix or name.startswith(prefix + ".") for name in sys.modules for prefix in forbidden)
from runs.goalzendo import run_g01_after_g00f_bridge as wrapper
assert wrapper.main([
    "--repo", str(repo),
    "--expected-coordinator-token-sha256", token_sha,
    "--expected-bridge-source-digest", source_digest,
]) == 2
assert not any(name == prefix or name.startswith(prefix + ".") for name in sys.modules for prefix in forbidden)
"""
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            script,
            str(program),
            str(_repo_for()),
            token_sha,
            _source_digest(),
        ],
        cwd=REPO,
        env={
            **os.environ,
            "PYTHONPATH": str(REPO / "src"),
            "PYTEST_CURRENT_TEST": "bridge-subprocess",
        },
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def test_wrapper_rejects_a_preloaded_backend_before_token_verification(
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from runs.goalzendo import run_g01_after_g00f_bridge as wrapper

    source_digest = _source_digest()
    monkeypatch.setitem(
        sys.modules,
        "goalzendo.experiment",
        ModuleType("goalzendo.experiment"),
    )
    result = wrapper.main(
        [
            "--repo",
            str(_repo_for()),
            "--expected-coordinator-token-sha256",
            "a" * 64,
            "--expected-bridge-source-digest",
            source_digest,
        ]
    )
    assert result == 2
    assert "must precede" in capsys.readouterr().err


@pytest.mark.parametrize(
    "module_name",
    [
        "goalzendo_g00f_g01_bridge",
        "goalzendo",
        "goalzendo.artifacts",
        "goalzendo.config",
        "goalzendo.evaluation",
        "goalzendo.experiment",
        "goalzendo.generation",
        "goalzendo.interventions",
        "goalzendo.modeling",
        "goalzendo.rendering",
        "goalzendo.runner",
        "goalzendo.schema",
        "goalzendo.training",
    ],
)
def test_repo_root_rejects_shadowed_imported_core_modules(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    module_name: str,
) -> None:
    module = importlib.import_module(module_name)
    monkeypatch.setattr(module, "__file__", str(tmp_path / "shadow.py"))
    with pytest.raises(BridgeError, match="not the exact module"):
        calculate_bridge_source_binding(REPO)


def test_repo_root_rejects_shadowed_bridge_module(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(bridge, "__file__", str(tmp_path / "shadow-bridge.py"))
    with pytest.raises(BridgeError, match="imported bridge"):
        calculate_bridge_source_binding(REPO)


def test_repo_root_rejects_loaded_project_module_without_source_file(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = importlib.import_module("goalzendo.evaluation")
    monkeypatch.setattr(module, "__file__", None)
    with pytest.raises(BridgeError, match="no auditable source file"):
        calculate_bridge_source_binding(REPO)


def test_repo_root_rejects_shadowed_module_spec_origin(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = importlib.import_module("goalzendo.config")
    assert module.__spec__ is not None
    monkeypatch.setattr(module.__spec__, "origin", str(tmp_path / "shadow-config.py"))
    with pytest.raises(BridgeError, match="not the exact module"):
        calculate_bridge_source_binding(REPO)


@pytest.mark.parametrize(
    ("route", "module_name"),
    [
        ("h100", "goalzendo_g00f"),
        ("h100", "goalzendo_g00f.freeze"),
        ("h100", "goalzendo_g00f.evaluator"),
        ("h200", "goalzendo_g00f_h200"),
        ("h200", "goalzendo_g00f_h200.freeze"),
        ("h200", "goalzendo_g00f_h200.evaluator"),
        ("h200", "goalzendo_g00f_h200.qualification"),
        ("h200", "goalzendo_g00f_h200.qualification_producer"),
    ],
)
def test_route_replay_rejects_shadowed_route_modules(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    route: str,
    module_name: str,
) -> None:
    module = importlib.import_module(module_name)
    monkeypatch.setattr(module, "__file__", str(tmp_path / "shadow-route.py"))
    with pytest.raises(BridgeError, match="not the exact module"):
        bridge._require_route_modules(REPO, route)  # type: ignore[arg-type]


def test_cli_and_stub_reject_shadow_entrypoints(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import goalzendo_g00f_g01_bridge.cli as cli
    from runs.goalzendo import run_g01_after_g00f_bridge as wrapper

    with monkeypatch.context() as isolated:
        isolated.setattr(cli, "__file__", str(tmp_path / "shadow-cli.py"))
        assert cli.main(["--repo", str(REPO), "source-binding"]) == 2
    with monkeypatch.context() as isolated:
        isolated.setattr(wrapper, "__file__", str(tmp_path / "shadow-wrapper.py"))
        assert (
            wrapper.main(
                [
                    "--repo",
                    str(_repo_for()),
                    "--expected-coordinator-token-sha256",
                    "a" * 64,
                    "--expected-bridge-source-digest",
                    _source_digest(),
                ]
            )
            == 2
        )


def test_source_only_module_invocation_and_external_source_pin() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "goalzendo_g00f_g01_bridge.cli",
            "--repo",
            str(REPO),
            "source-binding",
        ],
        cwd=REPO,
        env={**os.environ, "PYTHONPATH": str(REPO / "src")},
        capture_output=True,
        text=True,
        check=True,
    )
    observed = json.loads(result.stdout)
    assert observed == calculate_bridge_source_binding(REPO)
    with pytest.raises(BridgeError, match="externally expected"):
        bridge.bridge_source_binding(REPO, "0" * 64)
