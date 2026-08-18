from __future__ import annotations

import copy
import hashlib
import json
import shutil
import subprocess
import sys
import zipfile
from collections.abc import Callable
from pathlib import Path

import pytest

from scripts import verify_release as release

ROOT = Path(__file__).resolve().parents[1]
POLICY_PATH = ROOT / "reproducibility/releases/2026-08-11/release-policy.json"


def _write_canonical(path: Path, value: object) -> None:
    path.write_bytes(release._canonical_json_bytes(value, pretty=True))


def test_checked_policy_is_canonical_strict_and_nonauthorizing() -> None:
    policy = release.load_policy(POLICY_PATH)

    assert policy["status_as_of"] == "2026-08-15"
    assert set(policy["authority"].values()) >= {False}
    assert policy["authority"]["experiment_launch"] is False
    assert policy["authority"]["model_execution"] is False
    assert policy["authority"]["study_gate"] is False
    assert policy["authority"]["final_release_identity"] is False
    assert policy["scope"]["binds_git_commit"] is False
    assert policy["scope"]["binds_final_release_tree"] is False
    assert policy["scope"]["release_candidate_only"] is True
    assert policy["static_debt_baselines"]["ruff"]["diagnostic_count"] == 106
    assert policy["static_debt_baselines"]["mypy"]["diagnostic_count"] == 171
    assert {item["absence_id"] for item in policy["known_absences"]} == {
        "g00b-execution-source-archive",
        "goalzendo-raw-run-archives",
    }
    checkpoint_a = policy["bindings"]["checkpoint_a_canonical_bundle"]
    assert checkpoint_a["archive"]["sha256"] == (
        "db5b4533d1b5b5dc00afaaaf0f954ea91a570477e9837236f8e575401359630b"
    )
    assert checkpoint_a["manifest"]["manifest_digest"] == (
        "62fd29e7a7ce27d61958dee61dd6fb01a49a109df32ca0a8fc6d65288d44727f"
    )
    assert checkpoint_a["freeze"]["freeze_digest"] == (
        "a902fd55c873d4797b15c9d19a1a70d5bd3986b172925174df9271fc85678f0e"
    )
    assert checkpoint_a["authorization"]["g01_scientifically_eligible"] is False
    assert checkpoint_a["operational_status"]["canonical_bundle_built"] is True
    assert checkpoint_a["operational_status"]["build_intent_transcript_witness_recorded"] is True
    assert checkpoint_a["operational_status"]["build_intent_witness_durable_independent_registrar"] is False
    assert checkpoint_a["operational_status"]["build_intent_witness_reverified_by_release"] is False
    assert checkpoint_a["operational_status"]["build_result_transcript_witness_recorded"] is True
    assert checkpoint_a["operational_status"]["build_result_witness_durable_independent_registrar"] is False
    assert checkpoint_a["operational_status"]["build_result_witness_reverified_by_release"] is False
    assert checkpoint_a["operational_status"]["disposable_build_pod_deleted"] is True
    assert checkpoint_a["operational_status"]["scientific_runpod_pod_created"] is False
    assert {row["path"] for row in checkpoint_a["build_input_transport"]} == {
        "docs/goalzendo/protocols/g00f-g01-build-input-transport.md",
        "runs/goalzendo/g00f_g01_build_input_transport.py",
        "tests/goalzendo/test_g00f_g01_build_input_transport.py",
    }
    transport_by_path = {row["path"]: row for row in checkpoint_a["build_input_transport"]}
    transport_test = transport_by_path["tests/goalzendo/test_g00f_g01_build_input_transport.py"]
    assert transport_test["role"] == "phase_compatible_post_land_build_input_transport_tests"
    assert transport_test["sha256"] == ("81d046a78de3c980706d83a2e02b1c37653a59632ea936fa7aa7f5c91cd66501")
    assert transport_test["prebuild_audit_sha256"] == (
        "2035a819cf96695dcc478d1f4902e856f2f3f7c233bf528248a86b0b51880b11"
    )
    checkpoint_b = policy["bindings"]["checkpoint_b_source_milestone"]
    assert checkpoint_b["coordinator_source_digest"] == (
        "77d9ba4928fa29cc42a055201660304f2059d0ca6358371cc14618407bdc714b"
    )
    assert checkpoint_b["package_source_tree_sha256"] == (
        "1be9a56e256bc640d877dc2fc9c1ba4d2d397cef783adec248e037c9faf382b7"
    )
    assert checkpoint_b["refusal_code"] == "B_RUNTIME_OVERLAY_NOT_FROZEN"
    assert checkpoint_b["authorization"] == {
        "checkpoint_b_complete": False,
        "g01_launch_authorized": False,
        "model_execution_authorized": False,
    }
    assert checkpoint_b["operational_status"] == {
        "authenticated_launcher_created": False,
        "compute_estimate_resolved": False,
        "four_gpu_provision_operationally_accepted": False,
        "measured_pilot_scheduling_resolved": False,
        "outcomes_seen": False,
        "provision_transaction_created": False,
        "prospective_lifecycle_operationally_accepted": False,
        "runtime_overlay_created": False,
        "scientific_runpod_pod_created": False,
        "scientific_runpod_provision_receipt_created": False,
        "source_checkpoint_accepted": True,
        "stager_created": False,
        "supported_entrypoint_refusal_active": True,
        "wall_ceiling_operationally_accepted": False,
    }
    assert {row["path"] for row in checkpoint_b["accepted_files"]} == set(release.CHECKPOINT_B_ACCEPTED_FILES)
    capsule = policy["bindings"]["checkpoint_b_source_capsule_milestone"]
    assert capsule["source_capsule_digest"] == release.CHECKPOINT_B_SOURCE_CAPSULE_DIGEST
    assert capsule["authorization"] == release.CHECKPOINT_B_SOURCE_CAPSULE_AUTHORIZATION
    assert capsule["operational_status"] == release.CHECKPOINT_B_SOURCE_CAPSULE_OPERATIONAL_STATUS
    assert {row["path"] for row in capsule["accepted_files"]} == set(
        release.CHECKPOINT_B_SOURCE_CAPSULE_ACCEPTED_FILES
    )
    g01q = policy["bindings"]["g01q_source_milestone"]
    assert g01q["qualification_source_digest"] == release.G01Q_QUALIFICATION_SOURCE_DIGEST
    assert g01q["package_source_tree_sha256"] == release.G01Q_PACKAGE_SOURCE_TREE_SHA256
    assert g01q["refusal_code"] == release.G01Q_REFUSAL
    assert g01q["authorization"] == release.G01Q_AUTHORIZATION
    assert g01q["operational_status"] == release.G01Q_OPERATIONAL_STATUS
    assert {row["path"] for row in g01q["accepted_files"]} == set(release.G01Q_ACCEPTED_FILES)
    preprovision = policy["bindings"]["g01q_preprovision_source_milestone"]
    assert preprovision["preprovision_source_digest"] == release.G01Q_PREPROVISION_SOURCE_DIGEST
    assert preprovision["package_source_tree_sha256"] == (
        release.G01Q_PREPROVISION_PACKAGE_SOURCE_TREE_SHA256
    )
    assert preprovision["refusal_code"] == release.G01Q_PREPROVISION_REFUSAL
    assert preprovision["authorization"] == release.G01Q_PREPROVISION_AUTHORIZATION
    assert preprovision["operational_status"] == release.G01Q_PREPROVISION_OPERATIONAL_STATUS
    assert {row["path"] for row in preprovision["accepted_files"]} == set(
        release.G01Q_PREPROVISION_ACCEPTED_FILES
    )
    wheel = policy["full_checks"]["wheel"]
    assert wheel["network_allowed"] is False
    assert wheel["no_deps_install"] is True
    assert wheel["system_site_packages"] is True
    assert wheel["project_file"] == {
        "path": "pyproject.toml",
        "role": "exact_release_wheel_project_configuration",
        "sha256": release.PYPROJECT_SHA256,
    }
    assert "goalzendo_g00f" in wheel["packages"]
    assert {item["name"]: item["entry_point"] for item in wheel["console_scripts"]}[
        "goalzendo-g00f"
    ] == "goalzendo_g00f.cli:main"
    assert wheel["package_data"] == [
        {
            "path": "goalzendo_g00f/source_manifest.json",
            "sha256": "56b2d5e5163434ebc693e03a4e422ab1cf6279b39aa6501d34f8baf1287cb22a",
        }
    ]

    source_packages = policy["bindings"]["source_packages"]
    assert "goalzendo_g00f" in {item["name"] for item in source_packages["packages"]}
    assert "goalzendo_g00f_g01_bridge" in {item["name"] for item in source_packages["packages"]}
    assert "goalzendo_g00f_h200" in {item["name"] for item in source_packages["packages"]}
    assert "goalzendo_g01_coordinator" in {item["name"] for item in source_packages["packages"]}
    assert "goalzendo_g01_preprovision" in {item["name"] for item in source_packages["packages"]}
    assert "goalzendo_g01_qualification" in {item["name"] for item in source_packages["packages"]}
    assert "goalzendo_hidden_law" in {item["name"] for item in source_packages["packages"]}
    package_bindings = {item["name"]: item for item in source_packages["packages"]}
    assert package_bindings["goalzendo_g00f_g01_bridge"]["sha256"] == (
        "abbbd4cc60418c6bfda7a7e67db317bc87a55cc1911d373c659940951b3ce2aa"
    )
    assert package_bindings["goalzendo_g01_preprovision"]["sha256"] == (
        "2285f1412f6418e2be1af3bf60f12e10cca0d8101e25061698ae5e962b738f76"
    )
    assert package_bindings["goalzendo_interactive_v2"]["sha256"] == (
        "091a1bacc8b1bbd675744726490ae577d96402c66536e34b706f0c96c80981ea"
    )
    assert source_packages["source_only_packages"] == [
        "goalzendo_g00f_g01_bridge",
        "goalzendo_g00f_h200",
        "goalzendo_g01_coordinator",
        "goalzendo_g01_preprovision",
        "goalzendo_g01_qualification",
    ]
    assert source_packages["component_manifests"][0]["package"] == "goalzendo_g00f"
    assert source_packages["component_manifests"][0]["freeze_sha256"] == (
        "b2e385488ea7eb7c6f7bfc834c32ff2707aa9f96b718ea62b7ae92c9a4b8df38"
    )
    assert source_packages["component_manifests"][1]["package"] == "goalzendo_g00f_h200"
    assert source_packages["component_manifests"][1]["freeze_sha256"] == (
        "fd9cf73d124ec566e0589ec0b2e5e4d48a172bca2d04aff51bbb75a58b76b0f9"
    )
    assert "reproducibility/goalzendo/g00f-execution-freeze-20260811/SHA256SUMS" in {
        item["path"] for item in policy["bindings"]["checksum_manifests"]
    }
    assert "reproducibility/goalzendo/g00f-h200-execution-freeze-20260811/SHA256SUMS" in {
        item["path"] for item in policy["bindings"]["checksum_manifests"]
    }
    scoped = policy["full_checks"]["goalzendo_static"]
    assert "src/goalzendo_g00f" in scoped["ruff_paths"]
    assert "src/goalzendo_g00f" in scoped["mypy_paths"]
    assert "src/goalzendo_g00f_g01_bridge" in scoped["ruff_paths"]
    assert "src/goalzendo_g00f_g01_bridge" in scoped["mypy_paths"]
    assert "src/goalzendo_g00f_h200" in scoped["ruff_paths"]
    assert "src/goalzendo_g00f_h200" in scoped["mypy_paths"]
    assert "src/goalzendo_g01_coordinator" in scoped["ruff_paths"]
    assert "src/goalzendo_g01_coordinator" in scoped["mypy_paths"]
    assert "src/goalzendo_g01_preprovision" in scoped["ruff_paths"]
    assert "src/goalzendo_g01_preprovision" in scoped["mypy_paths"]
    assert "src/goalzendo_g01_qualification" in scoped["ruff_paths"]
    assert "src/goalzendo_g01_qualification" in scoped["mypy_paths"]
    assert "src/goalzendo_hidden_law" in scoped["ruff_paths"]
    assert "src/goalzendo_hidden_law" in scoped["mypy_paths"]
    assert "tests/goalzendo_hidden_law" in scoped["ruff_paths"]
    assert "goalzendo_hidden_law" in wheel["packages"]
    assert "scripts/run_g03_v2_evaluation_census.py" in scoped["ruff_paths"]
    assert "scripts/run_g03_v2_evaluation_census.py" in scoped["mypy_paths"]
    assert "paper/goalzendo-current-results/main.pdf" not in {
        item["path"] for item in policy["bindings"]["pdfs"]
    }
    assert [item["path"] for item in policy["full_checks"]["paper"]["outputs"]] == [
        "main.pdf",
        "zendo.pdf",
    ]


def test_policy_rejects_duplicate_unknown_and_noncanonical_json(tmp_path: Path) -> None:
    raw = POLICY_PATH.read_text(encoding="utf-8")
    duplicate = tmp_path / "duplicate.json"
    duplicate.write_text('{"schema_name":"duplicate",' + raw[1:], encoding="utf-8")
    with pytest.raises(release.PolicyError, match="duplicate JSON key"):
        release.load_policy(duplicate)

    policy = copy.deepcopy(release.load_policy(POLICY_PATH))
    policy["unexpected"] = True
    unknown = tmp_path / "unknown.json"
    _write_canonical(unknown, policy)
    with pytest.raises(release.PolicyError, match=r"unknown=\['unexpected'\]"):
        release.load_policy(unknown)

    noncanonical = tmp_path / "noncanonical.json"
    noncanonical.write_text(json.dumps(release.load_policy(POLICY_PATH)), encoding="utf-8")
    with pytest.raises(release.PolicyError, match="not in the declared canonical JSON form"):
        release.load_policy(noncanonical)


def test_normalized_static_debt_is_order_independent_and_unicode_exact(tmp_path: Path) -> None:
    first = tmp_path / "a.py"
    second = tmp_path / "b.py"
    ruff_findings = [
        {
            "code": "RUF001",
            "filename": str(second),
            "location": {"column": 9, "row": 7},
            "message": "ambiguous \N{EN DASH} character",
        },
        {
            "code": "F401",
            "filename": str(first),
            "location": {"column": 1, "row": 2},
            "message": "unused import",
        },
    ]
    expected_ruff = [
        {"code": "F401", "column": 1, "message": "unused import", "path": "a.py", "row": 2},
        {
            "code": "RUF001",
            "column": 9,
            "message": "ambiguous \N{EN DASH} character",
            "path": "b.py",
            "row": 7,
        },
    ]
    expected_ruff_hash = hashlib.sha256(
        release._canonical_json_bytes(expected_ruff, pretty=False)
    ).hexdigest()
    count, digest = release.normalize_ruff(json.dumps(ruff_findings), tmp_path)
    assert (count, digest) == (2, expected_ruff_hash)

    mypy_lines = [
        {
            "code": None,
            "column": 0,
            "file": "b.py",
            "line": 7,
            "message": "supporting note",
            "severity": "note",
        },
        {
            "code": "assignment",
            "column": 3,
            "file": "b.py",
            "line": 7,
            "message": "incompatible ≠ assignment",
            "severity": "error",
        },
        {
            "code": "arg-type",
            "column": 1,
            "file": "a.py",
            "line": 2,
            "message": "wrong argument",
            "severity": "error",
        },
    ]
    expected_mypy = [
        {
            "code": "arg-type",
            "column": 1,
            "message": "wrong argument",
            "path": "a.py",
            "row": 2,
            "severity": "error",
        },
        {
            "code": "assignment",
            "column": 3,
            "message": "incompatible ≠ assignment",
            "path": "b.py",
            "row": 7,
            "severity": "error",
        },
    ]
    stdout = "\n".join(json.dumps(item, ensure_ascii=False) for item in mypy_lines) + "\n"
    expected_mypy_hash = hashlib.sha256(
        release._canonical_json_bytes(expected_mypy, pretty=False)
    ).hexdigest()
    count, digest = release.normalize_mypy(stdout, tmp_path)
    assert (count, digest) == (2, expected_mypy_hash)


def test_checksum_manifest_checks_its_own_hash_targets_and_base(tmp_path: Path) -> None:
    evidence = tmp_path / "evidence"
    evidence.mkdir()
    payload = evidence / "payload.bin"
    payload.write_bytes(b"sealed evidence\n")
    manifest = evidence / "SHA256SUMS"
    manifest.write_text(f"{release._sha256_file(payload)}  payload.bin\n", encoding="utf-8")
    spec = {
        "entry_base": "manifest_parent",
        "path": "evidence/SHA256SUMS",
        "sha256": release._sha256_file(manifest),
    }
    assert release._verify_checksum_manifest(tmp_path, spec) == "1 checksum entries match"

    payload.write_bytes(b"drift\n")
    with pytest.raises(release.VerificationError, match="checksum mismatch"):
        release._verify_checksum_manifest(tmp_path, spec)

    manifest.write_text(f"{'0' * 64}  ../escape.bin\n", encoding="utf-8")
    with pytest.raises(release.PolicyError, match="normalized repository-relative path"):
        release._manifest_entries(manifest)


def test_g00f_component_manifest_is_complete_and_cross_bound_to_freeze(tmp_path: Path) -> None:
    policy = release.load_policy(POLICY_PATH)
    spec = policy["bindings"]["source_packages"]
    assert release._verify_component_manifests(ROOT, spec) == (
        "2 component manifests bind 12 exact source files"
    )

    shutil.copytree(ROOT / "src/goalzendo_g00f", tmp_path / "src/goalzendo_g00f")
    freeze_relative = Path(spec["component_manifests"][0]["freeze_path"])
    copied_freeze = tmp_path / freeze_relative
    copied_freeze.parent.mkdir(parents=True)
    shutil.copy2(ROOT / freeze_relative, copied_freeze)
    (tmp_path / "src/goalzendo_g00f/cli.py").write_text("drift\n", encoding="utf-8")
    with pytest.raises(release.VerificationError, match="component-manifest source mismatch"):
        release._verify_component_manifests(tmp_path, spec)


def test_checkpoint_a_canonical_bundle_is_strictly_replayed_and_nonauthorizing(
    tmp_path: Path,
) -> None:
    policy = release.load_policy(POLICY_PATH)
    spec = policy["bindings"]["checkpoint_a_canonical_bundle"]
    assert release._verify_checkpoint_a_canonical_bundle(ROOT, spec) == (
        "canonical checkpoint-A 6-file bundle, runtime receipt, and nonauthorization match"
    )

    altered = copy.deepcopy(spec)
    altered["manifest"]["manifest_digest"] = "0" * 64
    with pytest.raises(release.VerificationError, match="semantic identity"):
        release._verify_checkpoint_a_canonical_bundle(ROOT, altered)

    altered_transport = copy.deepcopy(spec)
    next(
        row
        for row in altered_transport["build_input_transport"]
        if row["path"] == "tests/goalzendo/test_g00f_g01_build_input_transport.py"
    )["prebuild_audit_sha256"] = "0" * 64
    with pytest.raises(release.VerificationError, match="historical prebuild transport-test audit"):
        release._verify_checkpoint_a_canonical_bundle(ROOT, altered_transport)

    ledger = json.loads(
        (ROOT / "reproducibility/goalzendo/study-status-ledger.json").read_text(encoding="utf-8")
    )
    g01 = next(study for study in ledger["studies"] if study["study_id"] == "G01")
    g01["identities"]["g01_launch_authorized"] = True
    target_ledger = tmp_path / "reproducibility/goalzendo/study-status-ledger.json"
    target_ledger.parent.mkdir(parents=True)
    target_ledger.write_text(
        json.dumps(ledger, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    needed = {
        release.CHECKPOINT_A_BUNDLE_ROOT,
        "docs/goalzendo/protocols/g00f-g01-build-input-transport.md",
        "reproducibility/goalzendo/g00f-execution-freeze-20260811",
        "reproducibility/goalzendo/g00f-h200-execution-freeze-20260811",
        "runs/goalzendo/build_g00f_g01_bridge_bundle.py",
        "runs/goalzendo/g00f_bundle_bootstrap.py",
        "runs/goalzendo/g00f_g01_bridge_bundle_stage.py",
        "runs/goalzendo/g00f_g01_build_input_transport.py",
        "runs/goalzendo/g00f_h200_bundle_bootstrap.py",
        "runs/goalzendo/g00f_h200_detached_supervisor.py",
        "runs/goalzendo/g00f_h200_qualification_controller.py",
        "runs/goalzendo/g00f_h200_qualification_supervisor.py",
        "runs/goalzendo/g00f_h200_watchdog.py",
        "runs/goalzendo/g00f_watchdog.py",
        "runs/goalzendo/run_g00f_frozen_4h100.sh",
        "runs/goalzendo/run_g00f_frozen_4h200.sh",
        "tests/goalzendo/test_g00f_g01_build_input_transport.py",
    }
    manifest = json.loads(
        (ROOT / release.CHECKPOINT_A_BUNDLE_ROOT / release.CHECKPOINT_A_MANIFEST_NAME).read_text(
            encoding="utf-8"
        )
    )
    needed.update(member["path"] for member in manifest["members"])
    for relative in needed:
        source = ROOT / relative
        target = tmp_path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        if source.is_dir():
            shutil.copytree(source, target, dirs_exist_ok=True)
        else:
            shutil.copy2(source, target)
    with pytest.raises(release.VerificationError, match="operational status changed for G01"):
        release._verify_checkpoint_a_canonical_bundle(tmp_path, spec)

    g01["identities"]["g01_launch_authorized"] = False
    g01["identities"]["scientific_runpod_pod_created"] = True
    target_ledger.write_text(
        json.dumps(ledger, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(release.VerificationError, match="G01 scientific runtime status changed"):
        release._verify_checkpoint_a_canonical_bundle(tmp_path, spec)

    g01["identities"]["scientific_runpod_pod_created"] = False
    g01["identities"]["checkpoint_a_canonical_bundle_transaction_digest"] = "0" * 64
    target_ledger.write_text(
        json.dumps(ledger, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(release.VerificationError, match="semantic bundle binding changed for G01"):
        release._verify_checkpoint_a_canonical_bundle(tmp_path, spec)

    g01["identities"]["checkpoint_a_canonical_bundle_transaction_digest"] = spec["frozen_transaction_digest"]
    g01["identities"]["checkpoint_a_bridge_source_digest"] = "0" * 64
    target_ledger.write_text(
        json.dumps(ledger, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(release.VerificationError, match="bridge binding changed for G01"):
        release._verify_checkpoint_a_canonical_bundle(tmp_path, spec)


def test_checkpoint_b_source_refusal_is_exact_and_runtime_incomplete(tmp_path: Path) -> None:
    policy = release.load_policy(POLICY_PATH)
    spec = policy["bindings"]["checkpoint_b_source_milestone"]
    source_packages = policy["bindings"]["source_packages"]
    assert release._verify_checkpoint_b_source_milestone(ROOT, spec, source_packages) == (
        "accepted checkpoint-B five-file source/refusal milestone matches; runtime transaction absent"
    )

    authorizing = copy.deepcopy(spec)
    authorizing["authorization"]["g01_launch_authorized"] = True
    with pytest.raises(release.PolicyError, match="exactly non-authorizing"):
        altered_policy = copy.deepcopy(policy)
        altered_policy["bindings"]["checkpoint_b_source_milestone"] = authorizing
        release._validate_policy_shape(altered_policy)

    for relative in set(release.CHECKPOINT_B_ACCEPTED_FILES) | {
        "reproducibility/goalzendo/study-status-ledger.json",
    }:
        source = ROOT / relative
        target = tmp_path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    ledger_path = tmp_path / "reproducibility/goalzendo/study-status-ledger.json"
    ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
    g01 = next(study for study in ledger["studies"] if study["study_id"] == "G01")
    g01["identities"]["checkpoint_b_runtime_overlay_created"] = True
    ledger_path.write_text(
        json.dumps(ledger, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(release.VerificationError, match="source/refusal status changed for G01"):
        release._verify_checkpoint_b_source_milestone(tmp_path, spec, source_packages)


@pytest.mark.parametrize(
    "status_field",
    [
        "checkpoint_b_authenticated_launcher_created",
        "checkpoint_b_complete",
        "checkpoint_b_compute_estimate_resolved",
        "checkpoint_b_four_gpu_provision_operationally_accepted",
        "checkpoint_b_global_coordinator_created",
        "checkpoint_b_measured_pilot_scheduling_resolved",
        "checkpoint_b_provision_transaction_created",
        "checkpoint_b_prospective_lifecycle_operationally_accepted",
        "checkpoint_b_runtime_overlay_created",
        "checkpoint_b_stager_created",
        "checkpoint_b_wall_ceiling_operationally_accepted",
        "g01_launch_authorized",
    ],
)
def test_checkpoint_b_ledger_rejects_every_false_to_true_status_transition(
    tmp_path: Path,
    status_field: str,
) -> None:
    policy = release.load_policy(POLICY_PATH)
    spec = policy["bindings"]["checkpoint_b_source_milestone"]
    source_packages = policy["bindings"]["source_packages"]
    for relative in set(release.CHECKPOINT_B_ACCEPTED_FILES) | {
        "reproducibility/goalzendo/study-status-ledger.json",
    }:
        source = ROOT / relative
        target = tmp_path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    ledger_path = tmp_path / "reproducibility/goalzendo/study-status-ledger.json"
    ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
    g01 = next(study for study in ledger["studies"] if study["study_id"] == "G01")
    g01["identities"][status_field] = True
    ledger_path.write_text(
        json.dumps(ledger, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(release.VerificationError, match="source/refusal status changed for G01"):
        release._verify_checkpoint_b_source_milestone(tmp_path, spec, source_packages)


def _copy_release_milestone_fixture(tmp_path: Path, accepted_files: set[str]) -> Path:
    for relative in accepted_files | {"reproducibility/goalzendo/study-status-ledger.json"}:
        source = ROOT / relative
        target = tmp_path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    return tmp_path / "reproducibility/goalzendo/study-status-ledger.json"


def test_checkpoint_b_source_capsule_is_bound_but_canonical_transaction_absent() -> None:
    policy = release.load_policy(POLICY_PATH)
    spec = policy["bindings"]["checkpoint_b_source_capsule_milestone"]
    assert release._verify_checkpoint_b_source_capsule_milestone(ROOT, spec) == (
        "accepted checkpoint-B source-capsule implementation matches; canonical transaction absent"
    )
    authorizing = copy.deepcopy(policy)
    authorizing["bindings"]["checkpoint_b_source_capsule_milestone"]["authorization"][
        "capsule_stage_authorized"
    ] = True
    with pytest.raises(release.PolicyError, match="exactly non-authorizing"):
        release._validate_policy_shape(authorizing)


@pytest.mark.parametrize(
    "status_field",
    [
        "checkpoint_b_complete",
        "checkpoint_b_source_capsule_authenticated_launcher_created",
        "checkpoint_b_source_capsule_build_authorized",
        "checkpoint_b_source_capsule_built",
        "checkpoint_b_source_capsule_fresh_verification_receipt_created",
        "checkpoint_b_source_capsule_g01_artifact_root_created",
        "checkpoint_b_source_capsule_g01_execution_status_root_created",
        "checkpoint_b_source_capsule_provision_transaction_created",
        "checkpoint_b_source_capsule_runtime_overlay_created",
        "checkpoint_b_source_capsule_stage_authorized",
        "checkpoint_b_source_capsule_stage_receipt_created",
        "checkpoint_b_source_capsule_staged",
        "g01_launch_authorized",
        "model_execution_authorized",
        "outcomes_seen",
    ],
)
def test_source_capsule_ledger_rejects_every_false_to_true_transition(
    tmp_path: Path,
    status_field: str,
) -> None:
    policy = release.load_policy(POLICY_PATH)
    spec = policy["bindings"]["checkpoint_b_source_capsule_milestone"]
    ledger_path = _copy_release_milestone_fixture(
        tmp_path, set(release.CHECKPOINT_B_SOURCE_CAPSULE_ACCEPTED_FILES)
    )
    ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
    g01 = next(study for study in ledger["studies"] if study["study_id"] == "G01")
    g01["identities"][status_field] = True
    _write_canonical(ledger_path, ledger)
    with pytest.raises(release.VerificationError, match="source-capsule ledger operational status changed"):
        release._verify_checkpoint_b_source_capsule_milestone(tmp_path, spec)


def test_source_capsule_ledger_rejects_duplicate_accepted_evidence(tmp_path: Path) -> None:
    policy = release.load_policy(POLICY_PATH)
    spec = policy["bindings"]["checkpoint_b_source_capsule_milestone"]
    ledger_path = _copy_release_milestone_fixture(
        tmp_path, set(release.CHECKPOINT_B_SOURCE_CAPSULE_ACCEPTED_FILES)
    )
    ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
    g01 = next(study for study in ledger["studies"] if study["study_id"] == "G01")
    path = next(iter(release.CHECKPOINT_B_SOURCE_CAPSULE_ACCEPTED_FILES))
    g01["evidence"].append(next(row for row in g01["evidence"] if row.get("path") == path))
    _write_canonical(ledger_path, ledger)
    with pytest.raises(release.VerificationError, match="omits accepted source evidence"):
        release._verify_checkpoint_b_source_capsule_milestone(tmp_path, spec)


@pytest.mark.parametrize(
    ("status_field", "integer_alias"),
    [
        ("checkpoint_b_source_capsule_built", 0),
        ("checkpoint_b_source_capsule_implementation_accepted", 1),
    ],
)
def test_source_capsule_ledger_rejects_boolean_integer_aliases(
    tmp_path: Path,
    status_field: str,
    integer_alias: int,
) -> None:
    policy = release.load_policy(POLICY_PATH)
    spec = policy["bindings"]["checkpoint_b_source_capsule_milestone"]
    ledger_path = _copy_release_milestone_fixture(
        tmp_path, set(release.CHECKPOINT_B_SOURCE_CAPSULE_ACCEPTED_FILES)
    )
    ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
    g01 = next(study for study in ledger["studies"] if study["study_id"] == "G01")
    g01["identities"][status_field] = integer_alias
    _write_canonical(ledger_path, ledger)
    with pytest.raises(release.VerificationError, match="source-capsule ledger operational status changed"):
        release._verify_checkpoint_b_source_capsule_milestone(tmp_path, spec)


def test_g01q_source_refusal_is_exact_and_runtime_absent() -> None:
    policy = release.load_policy(POLICY_PATH)
    spec = policy["bindings"]["g01q_source_milestone"]
    source_packages = policy["bindings"]["source_packages"]
    assert release._verify_g01q_source_milestone(ROOT, spec, source_packages) == (
        "accepted G01Q five-file source/refusal milestone matches; qualification runtime absent"
    )
    authorizing = copy.deepcopy(policy)
    authorizing["bindings"]["g01q_source_milestone"]["authorization"][
        "qualification_execution_authorized"
    ] = True
    with pytest.raises(release.PolicyError, match="exactly non-authorizing"):
        release._validate_policy_shape(authorizing)


@pytest.mark.parametrize(
    "status_field",
    [
        "g01_launch_authorized",
        "g01_training_authorized",
        "model_execution_authorized",
        "qualification_execution_authorized",
        "g01q_compute_route_qualified",
        "g01q_itt_created",
        "g01q_outcomes_seen",
        "g01q_qualification_campaign_created",
        "g01q_qualification_cleanup_receipt_created",
        "g01q_qualification_evidence_created",
        "g01q_qualification_executed",
        "g01q_qualification_pair_receipts_created",
        "g01q_qualification_provision_receipt_created",
        "g01q_qualification_report_created",
        "g01q_qualification_run_receipts_created",
        "g01q_qualification_runtime_frozen",
        "g01q_qualification_selection_created",
        "g01q_training_started",
    ],
)
def test_g01q_ledger_rejects_every_false_to_true_transition(
    tmp_path: Path,
    status_field: str,
) -> None:
    policy = release.load_policy(POLICY_PATH)
    spec = policy["bindings"]["g01q_source_milestone"]
    source_packages = policy["bindings"]["source_packages"]
    ledger_path = _copy_release_milestone_fixture(tmp_path, set(release.G01Q_ACCEPTED_FILES))
    ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
    g01 = next(study for study in ledger["studies"] if study["study_id"] == "G01")
    g01["identities"][status_field] = True
    _write_canonical(ledger_path, ledger)
    with pytest.raises(release.VerificationError, match="G01Q ledger source/refusal status changed"):
        release._verify_g01q_source_milestone(tmp_path, spec, source_packages)


def test_g01q_rejects_accepted_file_tamper_and_duplicate_evidence(tmp_path: Path) -> None:
    policy = release.load_policy(POLICY_PATH)
    spec = policy["bindings"]["g01q_source_milestone"]
    source_packages = policy["bindings"]["source_packages"]
    ledger_path = _copy_release_milestone_fixture(tmp_path, set(release.G01Q_ACCEPTED_FILES))
    accepted_path = "runs/goalzendo/run_g01_compute_qualification.py"
    (tmp_path / accepted_path).write_text("tampered\n", encoding="utf-8")
    with pytest.raises(release.VerificationError, match="SHA-256 mismatch"):
        release._verify_g01q_source_milestone(tmp_path, spec, source_packages)

    shutil.copy2(ROOT / accepted_path, tmp_path / accepted_path)
    ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
    g01 = next(study for study in ledger["studies"] if study["study_id"] == "G01")
    path = next(iter(release.G01Q_ACCEPTED_FILES))
    g01["evidence"].append(next(row for row in g01["evidence"] if row.get("path") == path))
    _write_canonical(ledger_path, ledger)
    with pytest.raises(release.VerificationError, match="omits accepted source evidence"):
        release._verify_g01q_source_milestone(tmp_path, spec, source_packages)


@pytest.mark.parametrize(
    ("status_field", "integer_alias"),
    [
        ("qualification_execution_authorized", 0),
        ("g01q_source_checkpoint_accepted", 1),
    ],
)
def test_g01q_ledger_rejects_boolean_integer_aliases(
    tmp_path: Path,
    status_field: str,
    integer_alias: int,
) -> None:
    policy = release.load_policy(POLICY_PATH)
    spec = policy["bindings"]["g01q_source_milestone"]
    source_packages = policy["bindings"]["source_packages"]
    ledger_path = _copy_release_milestone_fixture(tmp_path, set(release.G01Q_ACCEPTED_FILES))
    ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
    g01 = next(study for study in ledger["studies"] if study["study_id"] == "G01")
    g01["identities"][status_field] = integer_alias
    _write_canonical(ledger_path, ledger)
    with pytest.raises(release.VerificationError, match="G01Q ledger source/refusal status changed"):
        release._verify_g01q_source_milestone(tmp_path, spec, source_packages)


def test_g01q_preprovision_source_refusal_is_exact_and_registrar_absent() -> None:
    policy = release.load_policy(POLICY_PATH)
    spec = policy["bindings"]["g01q_preprovision_source_milestone"]
    source_packages = policy["bindings"]["source_packages"]
    assert release._verify_g01q_preprovision_source_milestone(ROOT, spec, source_packages) == (
        "accepted G01Q preprovision five-file source/refusal milestone matches; registrar absent"
    )


@pytest.mark.parametrize(
    ("section", "status_field"),
    [
        ("authorization", "g01_launch_authorized"),
        ("authorization", "g01_training_authorized"),
        ("authorization", "model_execution_authorized"),
        ("authorization", "qualification_execution_authorized"),
        ("operational_status", "campaign_intent_registered"),
        ("operational_status", "canonical_campaign_intent_candidate_artifact_created"),
        ("operational_status", "checkpoint_b_capsule_built"),
        ("operational_status", "checkpoint_b_capsule_staged"),
        ("operational_status", "compute_freeze_created"),
        ("operational_status", "compute_freeze_registered"),
        ("operational_status", "compute_route_qualified"),
        ("operational_status", "itt_created"),
        ("operational_status", "model_snapshot_authenticated"),
        ("operational_status", "outcomes_seen"),
        ("operational_status", "provider_access_performed"),
        ("operational_status", "provision_receipt_created"),
        ("operational_status", "qualification_executed"),
        ("operational_status", "registrar_designated"),
        ("operational_status", "runtime_frozen"),
        ("operational_status", "training_started"),
    ],
)
def test_g01q_preprovision_policy_rejects_every_false_to_true_transition(
    section: str,
    status_field: str,
) -> None:
    policy = copy.deepcopy(release.load_policy(POLICY_PATH))
    policy["bindings"]["g01q_preprovision_source_milestone"][section][status_field] = True
    with pytest.raises(release.PolicyError, match=r"exactly non-authorizing|source-only refusal boundary"):
        release._validate_policy_shape(policy)


@pytest.mark.parametrize(
    "status_field",
    [
        "g01_launch_authorized",
        "g01_training_authorized",
        "model_execution_authorized",
        "qualification_execution_authorized",
        "g01q_preprovision_campaign_intent_registered",
        "g01q_preprovision_canonical_campaign_intent_candidate_artifact_created",
        "g01q_preprovision_checkpoint_b_capsule_built",
        "g01q_preprovision_checkpoint_b_capsule_staged",
        "g01q_preprovision_compute_freeze_created",
        "g01q_preprovision_compute_freeze_registered",
        "g01q_preprovision_compute_route_qualified",
        "g01q_preprovision_itt_created",
        "g01q_preprovision_model_snapshot_authenticated",
        "g01q_preprovision_outcomes_seen",
        "g01q_preprovision_provider_access_performed",
        "g01q_preprovision_provision_receipt_created",
        "g01q_preprovision_qualification_executed",
        "g01q_preprovision_registrar_designated",
        "g01q_preprovision_runtime_frozen",
        "g01q_preprovision_training_started",
    ],
)
def test_g01q_preprovision_ledger_rejects_every_false_to_true_transition(
    tmp_path: Path,
    status_field: str,
) -> None:
    policy = release.load_policy(POLICY_PATH)
    spec = policy["bindings"]["g01q_preprovision_source_milestone"]
    source_packages = policy["bindings"]["source_packages"]
    ledger_path = _copy_release_milestone_fixture(tmp_path, set(release.G01Q_PREPROVISION_ACCEPTED_FILES))
    ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
    g01 = next(study for study in ledger["studies"] if study["study_id"] == "G01")
    g01["identities"][status_field] = True
    _write_canonical(ledger_path, ledger)
    with pytest.raises(
        release.VerificationError,
        match="G01Q preprovision ledger source/refusal status changed",
    ):
        release._verify_g01q_preprovision_source_milestone(tmp_path, spec, source_packages)


@pytest.mark.parametrize(
    ("status_field", "integer_alias"),
    [
        ("g01q_preprovision_registrar_designated", 0),
        ("g01q_preprovision_source_checkpoint_accepted", 1),
    ],
)
def test_g01q_preprovision_ledger_rejects_boolean_integer_aliases(
    tmp_path: Path,
    status_field: str,
    integer_alias: int,
) -> None:
    policy = release.load_policy(POLICY_PATH)
    spec = policy["bindings"]["g01q_preprovision_source_milestone"]
    source_packages = policy["bindings"]["source_packages"]
    ledger_path = _copy_release_milestone_fixture(tmp_path, set(release.G01Q_PREPROVISION_ACCEPTED_FILES))
    ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
    g01 = next(study for study in ledger["studies"] if study["study_id"] == "G01")
    g01["identities"][status_field] = integer_alias
    _write_canonical(ledger_path, ledger)
    with pytest.raises(
        release.VerificationError,
        match="G01Q preprovision ledger source/refusal status changed",
    ):
        release._verify_g01q_preprovision_source_milestone(tmp_path, spec, source_packages)


def test_g01q_preprovision_rejects_file_drift_and_duplicate_evidence(tmp_path: Path) -> None:
    policy = release.load_policy(POLICY_PATH)
    spec = policy["bindings"]["g01q_preprovision_source_milestone"]
    source_packages = policy["bindings"]["source_packages"]
    ledger_path = _copy_release_milestone_fixture(tmp_path, set(release.G01Q_PREPROVISION_ACCEPTED_FILES))
    accepted_path = "runs/goalzendo/run_g01_preprovision.py"
    (tmp_path / accepted_path).write_text("tampered\n", encoding="utf-8")
    with pytest.raises(release.VerificationError, match="SHA-256 mismatch"):
        release._verify_g01q_preprovision_source_milestone(tmp_path, spec, source_packages)

    shutil.copy2(ROOT / accepted_path, tmp_path / accepted_path)
    ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
    g01 = next(study for study in ledger["studies"] if study["study_id"] == "G01")
    path = next(iter(release.G01Q_PREPROVISION_ACCEPTED_FILES))
    g01["evidence"].append(next(row for row in g01["evidence"] if row.get("path") == path))
    _write_canonical(ledger_path, ledger)
    with pytest.raises(release.VerificationError, match="omits or duplicates accepted source evidence"):
        release._verify_g01q_preprovision_source_milestone(tmp_path, spec, source_packages)


def test_g01q_preprovision_wrapper_check_uses_exact_isolated_python(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    policy = release.load_policy(POLICY_PATH)
    spec = policy["bindings"]["g01q_preprovision_source_milestone"]
    source_packages = policy["bindings"]["source_packages"]
    _copy_release_milestone_fixture(tmp_path, set(release.G01Q_PREPROVISION_ACCEPTED_FILES))
    calls: list[list[str]] = []

    def refuse(command: list[str], *, cwd: Path, timeout: float | None = None) -> object:
        del cwd, timeout
        calls.append(command)
        return subprocess.CompletedProcess(
            command,
            2,
            "",
            "run_g01_preprovision: error: G01Q_REGISTRAR_NOT_DESIGNATED\n",
        )

    monkeypatch.setattr(release, "_run", refuse)
    release._verify_g01q_preprovision_source_milestone(tmp_path, spec, source_packages)
    assert len(calls) == 4
    assert all(command[1:3] == ["-I", "-S"] for command in calls)
    assert [command[4:] for command in calls] == [
        [],
        ["--release-verifier-must-still-refuse"],
        ["--help"],
        ["--version"],
    ]


@pytest.mark.parametrize(
    ("returncode", "stdout", "stderr"),
    [
        (0, "", ""),
        (2, "unexpected output\n", "run_g01_preprovision: error: G01Q_REGISTRAR_NOT_DESIGNATED\n"),
    ],
)
def test_g01q_preprovision_wrapper_rejects_success_or_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    returncode: int,
    stdout: str,
    stderr: str,
) -> None:
    policy = release.load_policy(POLICY_PATH)
    spec = policy["bindings"]["g01q_preprovision_source_milestone"]
    source_packages = policy["bindings"]["source_packages"]
    _copy_release_milestone_fixture(tmp_path, set(release.G01Q_PREPROVISION_ACCEPTED_FILES))

    def altered(command: list[str], *, cwd: Path, timeout: float | None = None) -> object:
        del cwd, timeout
        return subprocess.CompletedProcess(command, returncode, stdout, stderr)

    monkeypatch.setattr(release, "_run", altered)
    with pytest.raises(release.VerificationError, match="exact isolated side-effect-free refusal"):
        release._verify_g01q_preprovision_source_milestone(tmp_path, spec, source_packages)


def test_source_only_runtime_packages_are_bound_but_excluded_from_release_wheel_source(
    tmp_path: Path,
) -> None:
    policy = release.load_policy(POLICY_PATH)
    wheel_packages = policy["full_checks"]["wheel"]["packages"]
    assert "goalzendo_g00f_g01_bridge" not in wheel_packages
    assert "goalzendo_g00f_h200" not in wheel_packages
    assert "goalzendo_g01_coordinator" not in wheel_packages
    assert "goalzendo_g01_preprovision" not in wheel_packages
    assert "goalzendo_g01_qualification" not in wheel_packages

    release._copy_wheel_source(ROOT, tmp_path, wheel_packages)
    assert not (tmp_path / "src/goalzendo_g00f_g01_bridge").exists()
    assert not (tmp_path / "src/goalzendo_g00f_h200").exists()
    assert not (tmp_path / "src/goalzendo_g01_coordinator").exists()
    assert not (tmp_path / "src/goalzendo_g01_preprovision").exists()
    assert not (tmp_path / "src/goalzendo_g01_qualification").exists()
    assert (tmp_path / "src/goalzendo_g00f/source_manifest.json").is_file()


@pytest.mark.parametrize("bypass", ["wheel", "console", "package_data", "component_manifest"])
def test_g01q_preprovision_source_only_boundary_rejects_packaging_bypasses(bypass: str) -> None:
    policy = copy.deepcopy(release.load_policy(POLICY_PATH))
    if bypass == "wheel":
        policy["full_checks"]["wheel"]["packages"].append("goalzendo_g01_preprovision")
        policy["full_checks"]["wheel"]["packages"].sort()
    elif bypass == "console":
        policy["full_checks"]["wheel"]["console_scripts"].append(
            {
                "entry_point": "goalzendo_g01_preprovision.contracts:provision",
                "name": "goalzendo-g01-preprovision",
            }
        )
        policy["full_checks"]["wheel"]["console_scripts"].sort(key=lambda item: item["name"])
    elif bypass == "package_data":
        policy["full_checks"]["wheel"]["package_data"].append(
            {"path": "goalzendo_g01_preprovision/policy.json", "sha256": "0" * 64}
        )
        policy["full_checks"]["wheel"]["package_data"].sort(key=lambda item: item["path"])
    else:
        policy["bindings"]["source_packages"]["component_manifests"].append(
            {
                "freeze_path": "reproducibility/goalzendo/g01q-preprovision-freeze.json",
                "freeze_sha256": "0" * 64,
                "manifest_digest": "0" * 64,
                "package": "goalzendo_g01_preprovision",
                "path": "src/goalzendo_g01_preprovision/source_manifest.json",
                "schema": "goalzendo.g01q_preprovision_source_manifest",
                "schema_version": 1,
                "sha256": "0" * 64,
                "source_digest": "0" * 64,
            }
        )
        policy["bindings"]["source_packages"]["component_manifests"].sort(key=lambda item: item["package"])
    with pytest.raises(release.PolicyError):
        release._validate_policy_shape(policy)


def test_wheel_rejects_real_gui_script_entry_point_group(tmp_path: Path) -> None:
    policy = release.load_policy(POLICY_PATH)
    packages = policy["full_checks"]["wheel"]["packages"]
    source = tmp_path / "source"
    wheelhouse = tmp_path / "wheelhouse"
    outside = tmp_path / "outside"
    source.mkdir()
    wheelhouse.mkdir()
    outside.mkdir()
    release._copy_wheel_source(ROOT, source, packages)
    project = source / "pyproject.toml"
    project.write_text(
        project.read_text(encoding="utf-8")
        + "\n[project.gui-scripts]\n"
        + 'g01q-preprovision = "goalzendo_g01_preprovision.contracts:provision"\n',
        encoding="utf-8",
    )
    built = release._run(
        [
            sys.executable,
            "-m",
            "pip",
            "wheel",
            "--disable-pip-version-check",
            "--no-deps",
            "--no-index",
            "--no-build-isolation",
            "--wheel-dir",
            str(wheelhouse),
            str(source),
        ],
        cwd=outside,
    )
    assert built.returncode == 0, release._trim_output(built)
    wheels = list(wheelhouse.glob("*.whl"))
    assert len(wheels) == 1
    with pytest.raises(release.VerificationError, match="entry-point groups differ"):
        release._wheel_console_scripts(wheels[0])


def test_policy_and_quick_check_bind_exact_pyproject(tmp_path: Path) -> None:
    policy = release.load_policy(POLICY_PATH)
    altered = copy.deepcopy(policy)
    altered["full_checks"]["wheel"]["project_file"]["sha256"] = "0" * 64
    with pytest.raises(release.PolicyError, match="exact accepted pyproject"):
        release._validate_policy_shape(altered)

    project = tmp_path / "pyproject.toml"
    project.write_text("[project]\nname='replacement'\n", encoding="utf-8")
    with pytest.raises(release.VerificationError, match="SHA-256 mismatch"):
        release._verify_bound_file(tmp_path, policy["full_checks"]["wheel"]["project_file"])


def test_wheel_rejects_data_scripts_payload(tmp_path: Path) -> None:
    wheel = tmp_path / "unexpected-script.whl"
    with zipfile.ZipFile(wheel, "w") as archive:
        archive.writestr(
            "forkworld_goals-0.1.0.dist-info/entry_points.txt",
            "[console_scripts]\nforkworld = forkworld.cli:main\n",
        )
        archive.writestr("forkworld_goals-0.1.0.data/scripts/hidden", b"#!/bin/sh\n")
    with pytest.raises(release.VerificationError, match=r"unexpected \.data/scripts payloads"):
        release._wheel_console_scripts(wheel)


def test_quick_mode_does_not_require_ignored_goalzendo_main_pdf(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    policy = release.load_policy(POLICY_PATH)
    for name in (
        "_verify_ledger",
        "_verify_checkpoint_a_canonical_bundle",
        "_verify_checkpoint_b_source_capsule_milestone",
        "_verify_checkpoint_b_source_milestone",
        "_verify_g01q_preprovision_source_milestone",
        "_verify_g01q_source_milestone",
        "_verify_source_packages",
        "_verify_component_manifests",
        "_verify_checksum_manifest",
        "_verify_tool_versions",
        "_verify_static_baseline",
        "_verify_known_absences",
    ):
        monkeypatch.setattr(release, name, lambda *_args: "passes")

    checked_pdfs: list[str] = []

    def check_bound_file(_root: Path, binding: dict[str, object]) -> str:
        checked_pdfs.append(str(binding["path"]))
        return "passes"

    monkeypatch.setattr(release, "_verify_bound_file", check_bound_file)
    results = release.verify_quick(policy, tmp_path)
    assert all(result.passed for result in results)
    assert {result.name for result in results} >= {
        "checkpoint-b-source-capsule",
        "g01q-preprovision-source-refusal",
        "g01q-source-refusal",
    }
    assert "paper/goalzendo-current-results/main.pdf" not in checked_pdfs
    assert "paper/goalzendo-current-results/zendo.pdf" in checked_pdfs


def test_new_goalzendo_package_must_extend_both_static_scopes() -> None:
    policy = copy.deepcopy(release.load_policy(POLICY_PATH))
    wheel = policy["full_checks"]["wheel"]
    wheel["packages"].append("goalzendo_future")
    wheel["packages"].sort()
    source_packages = policy["bindings"]["source_packages"]["packages"]
    source_packages.append(
        {
            "name": "goalzendo_future",
            "path": "src/goalzendo_future",
            "sha256": "0" * 64,
        }
    )
    source_packages.sort(key=lambda item: item["name"])

    with pytest.raises(release.PolicyError, match="must appear in both scoped static path lists"):
        release._validate_policy_shape(policy)

    scoped = policy["full_checks"]["goalzendo_static"]
    scoped["ruff_paths"].append("src/goalzendo_future")
    scoped["mypy_paths"].append("src/goalzendo_future")
    scoped["ruff_paths"].sort()
    scoped["mypy_paths"].sort()
    release._validate_policy_shape(policy)


def test_full_mode_orchestrates_every_tier_without_authorizing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    policy = release.load_policy(POLICY_PATH)
    calls: list[str] = []

    monkeypatch.setattr(
        release,
        "verify_quick",
        lambda _policy, _root: [release.CheckResult("quick", True, "quick checks pass")],
    )

    def passing(name: str) -> Callable[[object, Path], str]:
        def check(_policy: object, _root: Path) -> str:
            calls.append(name)
            return f"{name} passes"

        return check

    monkeypatch.setattr(release, "_verify_scoped_static", passing("static"))
    monkeypatch.setattr(release, "_verify_paper_rebuild", passing("paper"))
    monkeypatch.setattr(release, "_verify_wheel", passing("wheel"))
    monkeypatch.setattr(release, "_verify_pytest", passing("pytest"))

    results = release.verify_full(policy, ROOT)
    report = release._report(POLICY_PATH, "full", results)
    assert calls == ["static", "paper", "wheel", "pytest"]
    assert report["passed"] is True
    assert report["authorizing"] is False
    assert [item.name for item in results] == [
        "quick",
        "goalzendo-scoped-static",
        "deterministic-goalzendo-paper",
        "isolated-wheel",
        "full-pytest",
    ]


def test_cli_returns_policy_error_without_running_checks(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    malformed = tmp_path / "malformed.json"
    malformed.write_text("{}\n", encoding="utf-8")
    assert release.main(["--policy", str(malformed)]) == 2
    assert "release policy error" in capsys.readouterr().err
