"""Schema and evidence checks for the machine-readable GoalZendo status ledger."""

from __future__ import annotations

import hashlib
import json
import tarfile
from pathlib import Path, PurePosixPath
from typing import Any, Literal, cast

import pytest
import yaml  # type: ignore[import-untyped]

from goalzendo_interactive.engine_audit import parse_engine_audit_report
from goalzendo_interactive.provenance import interactive_source_provenance

ROOT = Path(__file__).resolve().parents[2]
LEDGER_PATH = ROOT / "reproducibility" / "goalzendo" / "study-status-ledger.json"
STUDY_IDS = (
    "G00-A1b",
    "G00-A2",
    "G00-A3",
    "G00-A4",
    "G00-B",
    "G00-D",
    "G00-E",
    "G00-F",
    "G01",
    "G02",
    "G03-E",
    "G03-G",
    "G03-v2",
    "QWEN35-EVIDENCE-GEOMETRY",
    "QWEN35-KNOWN-LAW",
)
EXECUTION_STATES = {
    "complete",
    "complete_attested_unrestored",
    "engineering_only",
    "frozen_not_run",
    "not_launched",
    "running",
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _stable_digest(value: Any) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _source_tree_digest(path: Path) -> tuple[int, str]:
    records: list[dict[str, str]] = []
    for candidate in sorted(path.rglob("*")):
        relative = candidate.relative_to(path)
        if "__pycache__" in relative.parts or candidate.name == ".DS_Store" or candidate.suffix == ".pyc":
            continue
        assert not candidate.is_symlink()
        if candidate.is_dir():
            continue
        assert candidate.is_file()
        records.append({"path": relative.as_posix(), "sha256": _sha256(candidate)})
    payload = (json.dumps(records, ensure_ascii=False, separators=(",", ":"), sort_keys=True) + "\n").encode(
        "utf-8"
    )
    return len(records), hashlib.sha256(payload).hexdigest()


def _load() -> dict[str, Any]:
    return cast(dict[str, Any], json.loads(LEDGER_PATH.read_text(encoding="utf-8")))


def _studies(ledger: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {study["study_id"]: study for study in ledger["studies"]}


def _archive_engine_audit(path: Path) -> tuple[str, Any, int]:
    mode: Literal["r:gz", "r:"] = "r:gz" if path.name.endswith(".gz") else "r:"
    with tarfile.open(path, mode) as archive:
        names = archive.getnames()
        handle = archive.extractfile("tests/goalzendo_interactive/fixtures/g03-engine-audit-derived-v1.json")
        assert handle is not None
        payload = handle.read()
    report = parse_engine_audit_report(payload.removesuffix(b"\n").decode("ascii"))
    return hashlib.sha256(payload).hexdigest(), report, len(names)


def _archived_goalzendo_implementation_fingerprint(path: Path) -> tuple[int, str]:
    """Recompute the frozen GoalZendo source-v1 fingerprint from its archive."""

    prefix = PurePosixPath("goalzendo-qwen35-src/src/goalzendo")
    records: list[tuple[str, bytes]] = []
    with tarfile.open(path, "r:gz") as archive:
        for member in archive.getmembers():
            if not member.isfile():
                continue
            member_path = PurePosixPath(member.name)
            try:
                relative = member_path.relative_to(prefix)
            except ValueError:
                continue
            if relative.suffix != ".py" and relative.name != "py.typed":
                continue
            handle = archive.extractfile(member)
            assert handle is not None
            records.append((relative.as_posix(), handle.read()))

    records.sort(key=lambda item: item[0])
    assert records
    digest = hashlib.sha256(b"goalzendo-source-v1\0")
    for relative, payload in records:
        relative_bytes = relative.encode("utf-8")
        digest.update(len(relative_bytes).to_bytes(8, "big"))
        digest.update(relative_bytes)
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)
    return len(records), digest.hexdigest()


def test_ledger_is_canonical_scoped_and_fail_closed() -> None:
    raw = LEDGER_PATH.read_text(encoding="utf-8")
    ledger = json.loads(raw)
    assert raw == json.dumps(ledger, ensure_ascii=False, indent=2) + "\n"
    assert tuple(ledger) == (
        "schema_name",
        "schema_version",
        "canonicalization",
        "status_as_of",
        "snapshot_semantics",
        "scope",
        "evidence_strength_definitions",
        "studies",
        "deviations",
    )
    assert ledger["schema_name"] == "goalzendo.study_status_ledger"
    assert ledger["schema_version"] == 1
    assert ledger["status_as_of"] == "2026-08-15"
    assert ledger["scope"]["program"] == "GoalZendo"
    assert ledger["scope"]["included_source_packages"] == [
        "src/goalzendo",
        "src/goalzendo_g00e",
        "src/goalzendo_g00f",
        "src/goalzendo_g00f_g01_bridge",
        "src/goalzendo_g00f_h200",
        "src/goalzendo_g01_coordinator",
        "src/goalzendo_g01_preprovision",
        "src/goalzendo_g01_qualification",
        "src/goalzendo_interactive",
        "src/goalzendo_interactive_v2",
    ]
    assert ledger["scope"]["forkworld_included"] is False

    definitions = ledger["evidence_strength_definitions"]
    assert tuple(study["study_id"] for study in ledger["studies"]) == STUDY_IDS
    assert len(set(STUDY_IDS)) == len(STUDY_IDS)
    forbidden = tuple(ledger["scope"]["forbidden_study_evidence_prefixes"])
    for study in ledger["studies"]:
        assert set(study) == {
            "study_id",
            "title",
            "track",
            "execution",
            "assessment",
            "identities",
            "evidence",
            "limitations",
            "non_authorizations",
        }
        execution = study["execution"]
        assessment = study["assessment"]
        assert set(execution) == {"state", "planned_runs", "completed_runs", "note"}
        assert set(assessment) == {
            "state",
            "scientific_result_state",
            "evidence_strength",
            "note",
        }
        assert execution["state"] in EXECUTION_STATES
        assert assessment["evidence_strength"] in definitions
        assert study["limitations"] and study["non_authorizations"]

        if execution["state"] == "complete":
            assert execution["completed_runs"] == execution["planned_runs"]
        elif execution["state"] in {"not_launched", "frozen_not_run"}:
            assert execution["completed_runs"] == 0
        elif execution["state"] == "running":
            assert execution["completed_runs"] is None

        for item in study["evidence"]:
            assert set(item) == {"availability", "path", "role", "sha256"}
            assert len(item["sha256"]) == 64
            assert set(item["sha256"]) <= set("0123456789abcdef")
            if item["availability"] == "not_restored":
                assert item["path"] is None
                continue
            assert item["availability"] in {"repository", "workspace_only"}
            relative = item["path"]
            assert isinstance(relative, str) and relative
            assert not relative.startswith("/")
            assert not any(relative.startswith(prefix) for prefix in forbidden)
            path = ROOT / relative
            if item["availability"] == "repository" or path.exists():
                assert path.is_file()
                assert _sha256(path) == item["sha256"]

    studies = _studies(ledger)
    assert studies["G00-B"]["assessment"]["scientific_result_state"] == "not_assessed"
    assert studies["G00-D"]["execution"]["state"] == "complete"
    assert studies["G00-D"]["execution"]["completed_runs"] == 168
    assert studies["G00-D"]["assessment"]["state"] == ("corrected_gate_failed_two_registered_checks")
    assert studies["G00-D"]["identities"]["official_gate_overall_passed"] is False
    assert studies["G00-D"]["identities"]["official_gate_failed_checks"] == [
        "constrained_scorer",
        "rule_adapters",
    ]
    assert studies["G00-E"]["execution"]["state"] == "engineering_only"
    assert studies["G00-E"]["identities"]["sidecar_created"] is False
    assert studies["G00-F"]["execution"]["state"] == "frozen_not_run"
    assert studies["G00-F"]["execution"]["planned_runs"] == 160
    assert studies["G01"]["execution"]["state"] == "not_launched"
    assert studies["G02"]["execution"]["state"] == "not_launched"
    assert studies["G03-G"]["execution"]["state"] == "engineering_only"
    assert all("scientific result is claimed" in text for text in [studies["G03-G"]["non_authorizations"][2]])
    assert studies["G03-v2"]["execution"]["state"] == "not_launched"
    assert studies["G03-v2"]["identities"]["production_bank_authorized"] is False
    assert studies["G03-v2"]["identities"]["weight_updates_authorized"] is False
    geometry = studies["QWEN35-EVIDENCE-GEOMETRY"]
    assert geometry["execution"] == {
        "state": "complete",
        "planned_runs": 36,
        "completed_runs": 36,
        "note": (
            "All 36 prespecified runs and all six paired seed blocks completed. "
            "No scientific run failed, was replaced, or was excluded."
        ),
    }
    assert geometry["assessment"]["scientific_result_state"] == "behavioral_result_available"
    assert geometry["identities"]["scientific_run_failures"] == 0
    assert geometry["identities"]["final_endpoint_counts"] == {
        "runs": 36,
        "exact_herald": 34,
        "constant_action_2b": 2,
        "official_law": 0,
        "sage": 0,
        "other": 0,
    }
    assert geometry["identities"]["final_intervention_counts"] == {
        "flip_y_zero": 36,
        "flip_q_zero": 36,
        "flip_d_zero": 36,
        "flip_p_one": 34,
        "flip_p_zero": 2,
    }
    assert geometry["identities"]["interpretation_guard"] == {
        "negative_means_source": (
            "One constant-action 2B reference endpoint contributes the sole nonzero paired "
            "value to each registered contrast; every joint-rich diverse comparison endpoint "
            "and every other endpoint has exact Herald control."
        ),
        "negative_means_are_evidence_of_harm": False,
        "p_equals_one_establishes_equivalence": False,
    }
    qwen35 = studies["QWEN35-KNOWN-LAW"]
    assert qwen35["execution"] == {
        "state": "complete",
        "planned_runs": 96,
        "completed_runs": 96,
        "note": (
            "All 96 prespecified main-panel runs and all six paired seed blocks completed. "
            "No scientific run failed."
        ),
    }
    assert qwen35["assessment"]["scientific_result_state"] == "behavioral_result_available"
    assert qwen35["identities"]["scientific_run_failures"] == 0
    assert qwen35["identities"]["final_raw_controller_counts"] == {
        "q_p_1": {
            "runs": 48,
            "iid_law_accuracy_1_count": 48,
            "herald": 48,
            "law": 0,
            "sage": 0,
        },
        "q_p_0_95": {
            "runs": 48,
            "herald": 12,
            "law": 36,
            "sage": 0,
            "sft_law": 24,
            "outcome_rl_majority_law": 12,
            "outcome_rl_parity_herald": 12,
        },
    }
    assert qwen35["identities"]["config_sha256"] == (
        "2e1bbd650d466bb42f24b70d6081fd0f4f2490b14bfc7e02d17d59d9e251e7a6"
    )
    assert qwen35["identities"]["implementation_fingerprint"] == (
        "592dd4df02ceff4293ce6eebc6ea2a0975e7d427cb1298335bcd3319431f301b"
    )
    assert qwen35["identities"]["launch_record_sha256"] == (
        "df044281fc2c00adb263b7e9d5abda2f5d710367fc746d85fe7334579efaa916"
    )
    assert qwen35["identities"]["primary_analysis_sha256"] == (
        "6755e94884683863f0ccbf44fe25b0db48dac3455fb8bb1bda175f5e8670b924"
    )
    assert qwen35["identities"]["descriptive_analysis_sha256"] == (
        "3da4f5525716214eb9c182118fb64402d3e765f42f234435289ca5165dd8d5f5"
    )

    deviation_ids = tuple(item["deviation_id"] for item in ledger["deviations"])
    assert deviation_ids == tuple(sorted(deviation_ids))
    assert len(deviation_ids) == len(set(deviation_ids))
    assert all(set(item["study_ids"]) <= set(STUDY_IDS) for item in ledger["deviations"])


def test_qwen35_known_law_record_matches_local_canonical_outputs() -> None:
    study = _studies(_load())["QWEN35-KNOWN-LAW"]
    root = ROOT / "runs" / "goalzendo" / "qwen35-main-results" / "qwen35-main-20260813T184845Z"
    primary_path = root / "analysis" / "qwen35-known-law-analysis.json"
    descriptive_path = root / "analysis" / "qwen35-known-law-descriptives.json"
    launch_path = root / "launch-record.json"
    archive_path = root / "input" / "goalzendo-qwen35-main-src.tgz"
    if not all(path.is_file() for path in (primary_path, descriptive_path, launch_path, archive_path)):
        pytest.skip("workspace-only Qwen3.5 analysis outputs are not restored")

    identities = study["identities"]
    assert _sha256(launch_path) == identities["launch_record_sha256"]
    assert _sha256(primary_path) == identities["primary_analysis_sha256"]
    assert _sha256(descriptive_path) == identities["descriptive_analysis_sha256"]
    assert _sha256(archive_path) == identities["source_archive_sha256"]
    source_file_count, implementation_fingerprint = _archived_goalzendo_implementation_fingerprint(
        archive_path
    )
    assert source_file_count == 19
    assert implementation_fingerprint == identities["implementation_fingerprint"]

    launch = json.loads(launch_path.read_text(encoding="utf-8"))
    primary = json.loads(primary_path.read_text(encoding="utf-8"))
    descriptive = json.loads(descriptive_path.read_text(encoding="utf-8"))
    assert launch["status"] == "complete"
    assert launch["config_sha256"] == identities["config_sha256"]
    assert launch["implementation_fingerprint"] == identities["implementation_fingerprint"]
    assert primary["panel"]["observed_run_count"] == 96
    assert descriptive["panel"]["observed_run_count"] == 96

    runs = descriptive["runs"]
    q_p_1 = [run for run in runs if run["q_p"] == 1.0]
    q_p_095 = [run for run in runs if run["q_p"] == 0.95]

    def raw_controller(run: dict[str, Any]) -> str:
        return cast(str, run["full_view_controller_trajectory"][-1]["raw_controller"])

    assert len(q_p_1) == 48
    assert all(run["final_full_iid_law_accuracy"] == 1.0 for run in q_p_1)
    assert all(raw_controller(run) == "P" for run in q_p_1)
    assert len(q_p_095) == 48
    assert sum(raw_controller(run) == "Y" for run in q_p_095) == 36
    assert sum(raw_controller(run) == "P" for run in q_p_095) == 12
    assert all(raw_controller(run) == "Y" for run in q_p_095 if run["algorithm"] == "sft")
    assert all(
        raw_controller(run) == "Y"
        for run in q_p_095
        if run["algorithm"] == "outcome_rl" and run["law"] == "majority"
    )
    assert all(
        raw_controller(run) == "P"
        for run in q_p_095
        if run["algorithm"] == "outcome_rl" and run["law"] == "parity"
    )

    pooled = primary["primary"]["pooled_equal_weight_within_seed"]
    registered = identities["pooled_registered_estimates"]
    assert pooled["d_rho_p_minus_rho_y"]["mean"] == (registered["q_p_1_d_rho_p_minus_rho_y"]["mean"])
    assert (
        pooled["causal_companion_flip_p_minus_flip_y"]["mean"]
        == (registered["q_p_1_flip_p_minus_flip_y"]["mean"])
    )


def test_qwen35_evidence_geometry_record_matches_local_canonical_output() -> None:
    study = _studies(_load())["QWEN35-EVIDENCE-GEOMETRY"]
    identities = study["identities"]
    result_root = (
        ROOT
        / "runs"
        / "goalzendo"
        / "qwen35-evidence-geometry-results"
        / "qwen35-geometry-20260815T131905Z"
    )
    analysis_path = result_root / "analysis" / "qwen35-evidence-geometry-analysis.json"
    status_path = result_root / "analysis" / "status.json"
    sums_path = result_root / "analysis" / "SHA256SUMS"
    launch_path = result_root / "launch-record.json"
    archive_path = result_root / "input" / "goalzendo-qwen35-geometry-src.tgz"
    registration_path = result_root / "input" / "prelaunch-registration.json"
    required = (
        analysis_path,
        status_path,
        sums_path,
        launch_path,
        archive_path,
        registration_path,
    )
    if not all(path.is_file() for path in required):
        pytest.skip("workspace-only Qwen3.5 evidence-geometry outputs are not restored")

    assert _sha256(analysis_path) == identities["analysis_sha256"]
    assert _sha256(status_path) == identities["analysis_status_sha256"]
    assert _sha256(sums_path) == identities["analysis_sha256sums_sha256"]
    assert _sha256(launch_path) == identities["launch_record_sha256"]
    assert _sha256(archive_path) == identities["source_archive_sha256"]
    assert _sha256(registration_path) == identities["prelaunch_registration_sha256"]
    assert sums_path.read_text(encoding="ascii") == (
        f"{identities['analysis_sha256']}  qwen35-evidence-geometry-analysis.json\n"
    )

    protocol_path = ROOT / "docs" / "goalzendo" / "protocols" / "qwen35-evidence-geometry.md"
    protocol_bytes = protocol_path.read_bytes()
    frozen_start = protocol_bytes.index(b"# Evidence geometry and behavioral control in GoalZendo\n")
    frozen_protocol = protocol_bytes[frozen_start : frozen_start + 8251]
    assert hashlib.sha256(frozen_protocol).hexdigest() == identities["frozen_protocol_sha256"]
    assert _sha256(protocol_path) == identities["repository_protocol_with_completion_sha256"]

    launch = json.loads(launch_path.read_text(encoding="utf-8"))
    analysis = json.loads(analysis_path.read_text(encoding="utf-8"))
    status = json.loads(status_path.read_text(encoding="utf-8"))
    assert launch["status"] == "complete"
    assert launch["planned_runs"] == 36
    assert launch["config_sha256"] == identities["config_sha256"]
    assert launch["canonical_config_identity_sha256"] == (
        identities["canonical_config_identity_sha256"]
    )
    assert launch["implementation_fingerprint"] == identities["implementation_fingerprint"]
    assert launch["plan_key_set_sha256"] == identities["plan_key_set_sha256"]
    assert len(launch["working_shards"]) == 8
    assert all(shard["shard_status"] == "complete" for shard in launch["working_shards"])
    assert all(shard["runtime_status"] == "deleted" for shard in launch["working_shards"])
    assert launch["postcompletion_analysis"]["status"] == "complete"
    assert launch["postcompletion_analysis"]["runtime_status"] == "deleted"
    assert status["state"] == "complete"
    assert status["runner_rc"] == 0
    assert status["terminal_shard_receipt_count"] == 8
    assert status["verified_complete_run_count"] == 36
    assert status["analysis_sha256"] == identities["analysis_sha256"]
    assert analysis["panel"]["observed_run_count"] == 36
    assert analysis["panel"]["expected_run_count"] == 36

    endpoints = analysis["seed_endpoints"]
    herald = []
    constant_action = []
    for endpoint in endpoints:
        agreement = endpoint["full_conflict_agreement"]
        flips = endpoint["full_causal_action_flip"]
        action_rates = {row["action_b_rate"] for row in endpoint["final_full_truth_table"]}
        if agreement["rho_p"] == 1.0 and flips == {
            "flip_d": 0.0,
            "flip_p": 1.0,
            "flip_q": 0.0,
            "flip_y": 0.0,
        }:
            herald.append(endpoint)
        elif len(action_rates) == 1 and flips == {
            "flip_d": 0.0,
            "flip_p": 0.0,
            "flip_q": 0.0,
            "flip_y": 0.0,
        }:
            constant_action.append(endpoint)
    assert len(herald) == 34
    assert len(constant_action) == 2
    assert {endpoint["run_id"] for endpoint in constant_action} == {
        "8a419941c4b93a5f5ad0",
        "f18651820c2bb95f50c4",
    }
    assert all(endpoint["model"] == "Qwen/Qwen3.5-2B" for endpoint in constant_action)
    assert all(
        endpoint["full_causal_action_flip"][target] == 0.0
        for endpoint in endpoints
        for target in ("flip_y", "flip_q", "flip_d")
    )
    assert len(herald) + len(constant_action) == len(endpoints) == 36

    def assert_summary(recorded: dict[str, Any], observed: dict[str, Any]) -> None:
        interval = observed["bootstrap_95_percentile_interval"]
        assert recorded == {
            "mean": observed["mean"],
            "bootstrap_95_percentile_interval": [interval["lower"], interval["upper"]],
            "descriptive_unadjusted_two_sided_sign_flip_p_value": (
                observed["descriptive_unadjusted_two_sided_sign_flip_p_value"]
            ),
        }
        assert recorded["descriptive_unadjusted_two_sided_sign_flip_p_value"] == 1.0

    registered = identities["registered_contrasts"]
    for registered_key, analysis_key in (
        ("primary_overlap", "primary_contrast"),
        ("secondary_diversity", "registered_secondary_contrast"),
    ):
        recorded = registered[registered_key]
        observed = analysis[analysis_key]
        assert recorded["label"] == observed["label"]
        observed_by_model = {entry["model"]: entry for entry in observed["per_model"]}
        for model, model_record in recorded["per_model"].items():
            assert_summary(model_record["rho_y_effect"], observed_by_model[model]["rho_y_effect"])
            assert_summary(
                model_record["causal_companion_c_effect"],
                observed_by_model[model]["causal_companion_c_effect"],
            )
        pooled_record = recorded["pooled_equal_weight_across_models_within_seed"]
        pooled_observed = observed["pooled_equal_weight_across_models_within_seed"]
        assert_summary(pooled_record["rho_y_effect"], pooled_observed["rho_y_effect"])
        assert_summary(
            pooled_record["causal_companion_c_effect"],
            pooled_observed["causal_companion_c_effect"],
        )


def test_checkpoint_a_source_and_canonical_bundle_are_bound_without_runtime_authority() -> None:
    studies = _studies(_load())
    g00f = studies["G00-F"]["identities"]
    g01 = studies["G01"]["identities"]

    expected_files = {
        "checkpoint_a_bridge_init_sha256": "src/goalzendo_g00f_g01_bridge/__init__.py",
        "checkpoint_a_bridge_module_sha256": "src/goalzendo_g00f_g01_bridge/bridge.py",
        "checkpoint_a_bridge_cli_sha256": "src/goalzendo_g00f_g01_bridge/cli.py",
        "checkpoint_a_refusing_wrapper_sha256": ("runs/goalzendo/run_g01_after_g00f_bridge.py"),
        "checkpoint_a_bridge_test_sha256": "tests/goalzendo/test_g00f_g01_bridge.py",
        "checkpoint_a_bridge_protocol_sha256": ("docs/goalzendo/protocols/g00f-g01-authorization-bridge.md"),
        "checkpoint_a_bundle_builder_sha256": ("runs/goalzendo/build_g00f_g01_bridge_bundle.py"),
        "checkpoint_a_bundle_stager_sha256": ("runs/goalzendo/g00f_g01_bridge_bundle_stage.py"),
        "checkpoint_a_bundle_test_sha256": ("tests/goalzendo/test_g00f_g01_bridge_bundle.py"),
        "checkpoint_a_bundle_protocol_sha256": ("docs/goalzendo/protocols/g00f-g01-prebootstrap-bundle.md"),
        "checkpoint_a_build_input_transport_controller_sha256": (
            "runs/goalzendo/g00f_g01_build_input_transport.py"
        ),
        "checkpoint_a_build_input_transport_test_sha256": (
            "tests/goalzendo/test_g00f_g01_build_input_transport.py"
        ),
        "checkpoint_a_build_input_transport_protocol_sha256": (
            "docs/goalzendo/protocols/g00f-g01-build-input-transport.md"
        ),
    }
    for identity, relative in expected_files.items():
        assert _sha256(ROOT / relative) == g00f[identity]
    assert g00f["checkpoint_a_build_input_transport_prebuild_test_audit_sha256"] == (
        "2035a819cf96695dcc478d1f4902e856f2f3f7c233bf528248a86b0b51880b11"
    )
    assert (
        g00f["checkpoint_a_build_input_transport_prebuild_test_audit_sha256"]
        != g00f["checkpoint_a_build_input_transport_test_sha256"]
    )

    source_files = {
        relative: g00f[identity]
        for identity, relative in expected_files.items()
        if identity
        in {
            "checkpoint_a_bridge_init_sha256",
            "checkpoint_a_bridge_module_sha256",
            "checkpoint_a_bridge_cli_sha256",
            "checkpoint_a_refusing_wrapper_sha256",
        }
    }
    assert _stable_digest(source_files) == g00f["checkpoint_a_bridge_source_digest"]
    assert _source_tree_digest(ROOT / "src/goalzendo_g00f_g01_bridge") == (
        3,
        g00f["checkpoint_a_bridge_package_tree_sha256"],
    )
    assert g01["checkpoint_a_bridge_source_digest"] == g00f["checkpoint_a_bridge_source_digest"]
    assert g01["checkpoint_a_bridge_package_tree_sha256"] == (g00f["checkpoint_a_bridge_package_tree_sha256"])

    not_created = (
        "checkpoint_a_bundle_staged",
        "checkpoint_a_stage_receipt_created",
        "checkpoint_a_route_lock_created",
        "checkpoint_a_scientific_eligibility_artifact_created",
        "checkpoint_a_coordinator_token_created",
        "checkpoint_b_global_coordinator_created",
    )
    for identities in (g00f, g01):
        assert identities["checkpoint_a_source_audit_accepted"] is True
        assert identities["checkpoint_a_canonical_bundle_audit_accepted"] is True
        assert identities["checkpoint_a_canonical_bundle_built"] is True
        assert identities["checkpoint_a_disposable_build_pod_created"] is True
        assert identities["checkpoint_a_disposable_build_pod_deleted"] is True
        assert identities["checkpoint_a_build_intent_transcript_witness_recorded"] is True
        assert identities["checkpoint_a_build_intent_witness_durable_independent_registrar"] is False
        assert identities["checkpoint_a_build_intent_witness_reverified_by_release"] is False
        assert identities["checkpoint_a_build_result_transcript_witness_recorded"] is True
        assert identities["checkpoint_a_build_result_witness_durable_independent_registrar"] is False
        assert identities["checkpoint_a_build_result_witness_reverified_by_release"] is False
        assert all(identities[key] is False for key in not_created)
        assert identities["g01_launch_authorized"] is False
        assert identities["direct_g01_launch_authorized"] is False
    assert g01["outcomes_seen"] is False
    assert g01["scientific_runpod_pod_created"] is False
    assert g01["scientific_runpod_provision_receipt_created"] is False

    canonical_bundle = ROOT / "reproducibility/goalzendo/g00f-g01-bridge-prebootstrap-20260812"
    assert canonical_bundle.is_dir()
    expected_artifacts = {
        "g00f-g01-bridge-bundle.tar.gz": "checkpoint_a_canonical_bundle_archive_sha256",
        "g00f-g01-bridge-bundle-manifest.json": "checkpoint_a_canonical_bundle_manifest_sha256",
        "g00f-g01-bridge-bundle-freeze.json": "checkpoint_a_canonical_bundle_freeze_sha256",
    }
    assert {path.name for path in canonical_bundle.iterdir()} == set(expected_artifacts)
    for name, identity in expected_artifacts.items():
        assert _sha256(canonical_bundle / name) == g00f[identity] == g01[identity]

    manifest = json.loads(
        (canonical_bundle / "g00f-g01-bridge-bundle-manifest.json").read_text(encoding="utf-8")
    )
    freeze = json.loads((canonical_bundle / "g00f-g01-bridge-bundle-freeze.json").read_text(encoding="utf-8"))
    assert manifest["manifest_digest"] == g00f["checkpoint_a_canonical_bundle_manifest_digest"]
    assert freeze["freeze_digest"] == g00f["checkpoint_a_canonical_bundle_freeze_digest"]
    assert (
        freeze["authorization"]
        == manifest["authorization"]
        == {
            "dedicated_global_coordinator_required": True,
            "direct_g01_launch_authorized": False,
            "g00f_outcomes_seen": False,
            "g01_scientifically_eligible": False,
        }
    )
    assert (
        _stable_digest(freeze["build_runtime"])
        == (g00f["checkpoint_a_canonical_bundle_build_runtime_digest"])
    )
    assert _stable_digest(freeze["transaction"]) == (g00f["checkpoint_a_canonical_bundle_transaction_digest"])
    with tarfile.open(canonical_bundle / "g00f-g01-bridge-bundle.tar.gz", "r:gz") as archive:
        members = archive.getmembers()
    assert len(members) == g00f["checkpoint_a_canonical_bundle_member_count"] == 6
    assert all(member.isfile() and member.mode == 0o644 for member in members)


def test_g01q_preprovision_source_is_bound_without_operational_authority() -> None:
    g01 = _studies(_load())["G01"]
    identities = g01["identities"]
    expected_files = {
        "g01q_preprovision_protocol_sha256": "docs/goalzendo/protocols/g01-preprovision.md",
        "g01q_preprovision_entrypoint_sha256": "runs/goalzendo/run_g01_preprovision.py",
        "g01q_preprovision_init_sha256": "src/goalzendo_g01_preprovision/__init__.py",
        "g01q_preprovision_contracts_sha256": "src/goalzendo_g01_preprovision/contracts.py",
        "g01q_preprovision_test_sha256": "tests/goalzendo/test_g01_preprovision.py",
    }
    for identity, relative in expected_files.items():
        assert identities[identity] == _sha256(ROOT / relative)

    source_files = {
        relative: identities[identity]
        for identity, relative in expected_files.items()
        if identity
        in {
            "g01q_preprovision_entrypoint_sha256",
            "g01q_preprovision_init_sha256",
            "g01q_preprovision_contracts_sha256",
        }
    }
    assert _stable_digest(source_files) == identities["g01q_preprovision_source_digest"]
    assert _source_tree_digest(ROOT / "src/goalzendo_g01_preprovision") == (
        2,
        identities["g01q_preprovision_package_source_tree_sha256"],
    )
    assert identities["g01q_preprovision_source_checkpoint_accepted"] is True
    assert identities["g01q_preprovision_supported_entrypoint_refusal_active"] is True

    false_fields = {
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
    }
    assert all(type(identities[field]) is bool and identities[field] is False for field in false_fields)
    evidence = {row["path"]: row for row in g01["evidence"]}
    for identity, relative in expected_files.items():
        assert evidence[relative]["availability"] == "repository"
        assert evidence[relative]["sha256"] == identities[identity]


def test_completed_adaptive_audits_match_synchronized_artifacts() -> None:
    studies = _studies(_load())

    for beta, expected in (
        ("beta001", "beta_0.01"),
        ("beta010", "beta_0.10"),
        ("beta100", "beta_1.00"),
    ):
        path = (
            ROOT
            / "artifacts-goalzendo-remote"
            / "g00a-rl-horizon"
            / beta
            / "status-goalzendo"
            / f"g00a-rl-horizon-{beta}-analysis.tar.gz"
        )
        if not path.exists():
            continue
        with tarfile.open(path, "r:gz") as archive:
            handle = archive.extractfile("./analysis-manifest.json")
            assert handle is not None
            manifest = json.load(handle)
        panel = manifest["artifact_panel"]
        assert panel["panel_complete"] is True
        assert panel["run_count"] == panel["expected_plan_keys"] == 2
        assert panel["implementation_fingerprint"] == studies["G00-A1b"]["identities"]["source_fingerprint"]
        assert _sha256(path) == studies["G00-A1b"]["identities"]["analysis_archives"][expected]

    audit_specs = (
        (
            "G00-A2",
            "artifacts-goalzendo-remote/g00a2-deterministic-horizon-20260810/audits/"
            "g00a2-horizon-external-v2b-complete.json",
            "passed",
            "complete_audit_sha256",
        ),
        (
            "G00-A3",
            "artifacts-goalzendo-remote/g00a3-enumerated-outcome-gradient-v008/audits/artifact-audit.json",
            "passed",
            "artifact_audit_sha256",
        ),
        (
            "G00-A4",
            "artifacts-goalzendo-remote/g00a4-reward-gradient-shape-v014/audits/"
            "g00a4-v014-combined-audit.json",
            "audit_passed",
            "combined_audit_sha256",
        ),
    )
    for study_id, relative, pass_key, identity_key in audit_specs:
        path = ROOT / relative
        if not path.exists():
            continue
        audit = json.loads(path.read_text(encoding="utf-8"))
        assert audit[pass_key] is True
        source_fingerprint = audit.get(
            "source_fingerprint", audit.get("binding", {}).get("source_fingerprint")
        )
        assert source_fingerprint == studies[study_id]["identities"]["source_fingerprint"]
        assert _sha256(path) == studies[study_id]["identities"][identity_key]


def test_g00b_and_g00d_plans_sources_and_guards_are_exact() -> None:
    studies = _studies(_load())
    g00b = studies["G00-B"]
    g00b_plan = ROOT / "docs/goalzendo/plans/g00b-deterministic-capability-validation.jsonl"
    g00b_rows = [json.loads(line) for line in g00b_plan.read_text(encoding="utf-8").splitlines()]
    g00b_keys = sorted(row["plan_key"] for row in g00b_rows)
    assert len(g00b_keys) == len(set(g00b_keys)) == 24
    assert _stable_digest(g00b_keys) == g00b["identities"]["planned_run_keys_stable_digest"]
    assert (
        hashlib.sha256("\n".join(g00b_keys).encode()).hexdigest()
        == g00b["identities"]["sorted_newline_plan_key_sha256"]
    )
    assert (
        "execution complete"
        in (ROOT / "docs/goalzendo/protocols/g00b-fresh-capability-validation.md").read_text()
    )

    g00d = studies["G00-D"]
    manifest_path = (
        ROOT / "reproducibility/goalzendo/frozen-sources/goalzendo-g00d-fp1a814637-20260811.manifest.json"
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["implementation_fingerprint"] == g00d["identities"]["source_fingerprint"]
    assert manifest["bundle_sha256"] == g00d["identities"]["source_bundle_sha256"]
    assert manifest["plan_key_union_count"] == g00d["identities"]["plan_key_union_count"] == 168
    assert manifest["plan_key_union_digest"] == g00d["identities"]["plan_key_union_stable_digest"]

    union: list[str] = []
    for panel in g00d["identities"]["panels"]:
        plan_path = ROOT / panel["plan_path"]
        config_path = ROOT / panel["config_path"]
        assert _sha256(plan_path) == panel["plan_file_sha256"]
        assert _sha256(config_path) == panel["config_sha256"]
        rows = [json.loads(line) for line in plan_path.read_text(encoding="utf-8").splitlines()]
        keys = sorted(row["plan_key"] for row in rows)
        assert len(keys) == len(set(keys)) == panel["planned_runs"]
        assert _stable_digest(keys) == panel["sorted_plan_keys_stable_digest"]
        union.extend(keys)
    assert len(union) == len(set(union)) == 168
    assert _stable_digest(sorted(union)) == g00d["identities"]["plan_key_union_stable_digest"]

    analysis_root = ROOT / "artifacts-goalzendo-remote/g00d-fixed-window-20260811/analysis"
    analysis_manifest_path = analysis_root / "analysis-manifest.json"
    if analysis_manifest_path.exists():
        analysis_manifest = json.loads(analysis_manifest_path.read_text(encoding="utf-8"))
        assert _sha256(analysis_manifest_path) == g00d["identities"]["optimizer_analysis_manifest_sha256"]
        panel = analysis_manifest["artifact_panel"]
        assert panel["panel_complete"] is True
        assert panel["run_count"] == panel["expected_plan_keys"] == 48
        assert panel["implementation_fingerprint"] == g00d["identities"]["source_fingerprint"]
        for table in analysis_manifest["tables"].values():
            table_path = analysis_root / table["file"]
            assert _sha256(table_path) == table["sha256"]
            assert len(table_path.read_text(encoding="utf-8").splitlines()) - 1 == table["rows"]

    assert g00d["identities"]["diagnostic_optimizer_selection"]["authorizing"] is False

    for study_id, relative in (
        ("G01", "configs/goalzendo/g01_known_law.yaml"),
        ("G02", "configs/goalzendo/g02_evidence_geometry.yaml"),
    ):
        config = yaml.safe_load((ROOT / relative).read_text(encoding="utf-8"))
        assert config["run"]["launch_guard"] == studies[study_id]["identities"]["launch_guard"]
        assert config["experiment"]["status"] == "prospective"


def test_g00f_canonical_freeze_is_exactly_frozen_not_run_and_nonauthorizing() -> None:
    study = _studies(_load())["G00-F"]
    identities = study["identities"]
    freeze_root = ROOT / "reproducibility/goalzendo/g00f-execution-freeze-20260811"
    freeze_path = freeze_root / "execution-freeze.json"
    assessment_path = freeze_root / "preexecution-assessment.json"
    gate_path = freeze_root / "preexecution-gate.json"

    assert study["execution"] == {
        "state": "frozen_not_run",
        "planned_runs": 160,
        "completed_runs": 0,
        "note": (
            "The immutable H100 parent and prospective H200 execution revision "
            "each register one selected 160-run study (80 fresh 0.5B and 80 fresh "
            "1.5B runs). The H200 selector has not run, so neither candidate profile "
            "is selected. The independently audited checkpoint-A canonical bridge "
            "bundle was built on a disposable CPU Runpod pod, returned, independently "
            "replayed against both routes, and release-bound; the build pod was then "
            "deleted. The bounded checkpoint-B source/refusal milestone is also accepted, "
            "but its supported entrypoint refuses because no runtime overlay, stager, G01 "
            "provision transaction, or authenticated launcher exists. No scientific "
            "Runpod pod or provision receipt was created, the "
            "bundle has not been staged, no model was loaded, and no run or outcome exists."
        ),
    }
    assert study["assessment"]["state"] == (
        "canonical_execution_frozen_checkpoint_a_bundle_built_"
        "checkpoint_b_source_accepted_preexecution_gate_false"
    )
    assert study["assessment"]["evidence_strength"] == "frozen_execution_identity"
    assert identities["runpod_pod_created"] is False
    assert identities["runpod_provision_receipt_created"] is False
    assert identities["outcomes_seen"] is False
    assert identities["exact_g00f_execution_authorized"] is True
    assert identities["g01_launch_authorized"] is False

    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    assert _sha256(freeze_path) == identities["execution_freeze_file_sha256"]
    assert freeze["freeze_digest"] == identities["execution_freeze_digest"]
    assert freeze["outcomes_seen"] is identities["outcomes_seen"] is False
    assert freeze["authorization"] == {
        "g00f_exact_execution_authorized": True,
        "g01_launch_authorized": False,
        "scope": "exact_g00f_frozen_worker_schedule_only",
    }
    assert _sha256(freeze_root / "g00f-execution-source.tar.gz") == identities["source_archive_sha256"]
    assert (
        _sha256(freeze_root / "g00f-source-bundle-manifest.json")
        == identities["source_bundle_manifest_sha256"]
    )
    assert freeze["source_bundle"]["manifest_digest"] == identities["source_bundle_manifest_digest"]
    assert (
        _sha256(ROOT / "src/goalzendo_g00f/source_manifest.json")
        == identities["additive_source_manifest_sha256"]
    )
    assert freeze["additive_source"]["source_digest"] == identities["additive_source_digest"]

    panel_specs = (
        ("g00f-0p5b", "0p5b"),
        ("g00f-1p5b", "1p5b"),
    )
    all_plan_keys: set[str] = set()
    for panel_id, suffix in panel_specs:
        panel = freeze["configurations"][panel_id]
        plan_path = ROOT / panel["plan_path"]
        rows = [json.loads(line) for line in plan_path.read_text(encoding="utf-8").splitlines()]
        plan_keys = {row["plan_key"] for row in rows}
        run_ids = {row["run_id"] for row in rows}
        assert len(rows) == len(plan_keys) == len(run_ids) == panel["run_count"] == 80
        assert all_plan_keys.isdisjoint(plan_keys)
        all_plan_keys.update(plan_keys)
        assert _sha256(plan_path) == panel["plan_sha256"] == identities[f"plan_{suffix}_sha256"]
        assert panel["plan_key_digest"] == identities[f"plan_{suffix}_digest"]
        config_path = ROOT / panel["config_path"]
        assert _sha256(config_path) == panel["config_sha256"] == identities[f"config_{suffix}_sha256"]
    assert len(all_plan_keys) == 160

    assessment = json.loads(assessment_path.read_text(encoding="utf-8"))
    assert _sha256(assessment_path) == identities["preexecution_assessment_file_sha256"]
    assert assessment["assessment_digest"] == identities["preexecution_assessment_digest"]
    assert assessment["overall_passed"] is False
    assert assessment["failure_reason"] == "prospective_freeze_no_model_execution"
    assert assessment["checks"]["panel_completion"] == {
        "completed_runs": 0,
        "passed": False,
        "planned_runs": 160,
        "reason": "prospective_freeze_no_model_execution",
        "state_counts": {"not_run": 160},
    }
    assert len(assessment["intention_to_train"]) == identities["preexecution_not_run_rows"] == 160
    assert all(
        row["state"] == "not_run" and row["attempt_count"] == 0 for row in assessment["intention_to_train"]
    )

    gate = json.loads(gate_path.read_text(encoding="utf-8"))
    assert _sha256(gate_path) == identities["preexecution_gate_file_sha256"]
    assert gate["gate_digest"] == identities["preexecution_gate_digest"]
    assert gate["overall_passed"] is identities["preexecution_gate_overall_passed"] is False
    assert gate["assessment"] == assessment
    assert gate["authorization"] == {
        "g00f_remediation_passed": False,
        "g01_launch_authorized": False,
        "reason": "a_separately_reviewed_digest_bound_g01_bridge_is_required",
        "scope": "none",
    }


def test_g00f_h200_execution_revision_is_exactly_frozen_unselected_and_nonauthorizing() -> None:
    study = _studies(_load())["G00-F"]
    identities = study["identities"]
    freeze_root = ROOT / "reproducibility/goalzendo/g00f-h200-execution-freeze-20260811"
    freeze_path = freeze_root / "execution-freeze.json"
    assessment_path = freeze_root / "preexecution-assessment.json"
    gate_path = freeze_root / "preexecution-gate.json"

    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    assert _sha256(freeze_path) == identities["h200_execution_freeze_file_sha256"]
    assert freeze["freeze_digest"] == identities["h200_execution_freeze_digest"]
    assert freeze["outcomes_seen"] is identities["h200_outcomes_seen"] is False
    assert freeze["authorization"] == {
        "g00f_exact_execution_authorized": True,
        "g01_launch_authorized": False,
        "scope": "exact_g00f_frozen_worker_schedule_only",
    }
    assert (
        freeze["historical_h100_parent"]["files"][0]["sha256"] == (identities["execution_freeze_file_sha256"])
    )
    assert (
        _sha256(freeze_root / "g00f-h200-execution-source.tar.gz")
        == (identities["h200_source_archive_sha256"])
    )
    assert (
        _sha256(freeze_root / "g00f-h200-source-bundle-manifest.json")
        == (identities["h200_source_bundle_manifest_sha256"])
    )
    assert freeze["source_bundle"]["manifest_digest"] == (identities["h200_source_bundle_manifest_digest"])
    assert (
        _sha256(ROOT / "src/goalzendo_g00f_h200/source_manifest.json")
        == (identities["h200_additive_source_manifest_sha256"])
    )
    assert freeze["additive_source"]["source_digest"] == identities["h200_additive_source_digest"]

    all_candidate_plan_keys: set[str] = set()
    for profile in ("baseline", "tuned"):
        worker_counts = {index: 0 for index in range(4)}
        for panel_id, suffix in (("g00f-0p5b", "0p5b"), ("g00f-1p5b", "1p5b")):
            panel = freeze["configurations"][profile][panel_id]
            plan_path = ROOT / panel["plan_path"]
            rows = [json.loads(line) for line in plan_path.read_text(encoding="utf-8").splitlines()]
            plan_keys = {row["plan_key"] for row in rows}
            run_ids = {row["run_id"] for row in rows}
            assert len(rows) == len(plan_keys) == len(run_ids) == panel["run_count"] == 80
            assert all_candidate_plan_keys.isdisjoint(plan_keys)
            all_candidate_plan_keys.update(plan_keys)
            assert (
                _sha256(plan_path)
                == panel["plan_sha256"]
                == (identities[f"h200_{profile}_plan_{suffix}_sha256"])
            )
            assert panel["plan_key_digest"] == identities[f"h200_{profile}_plan_{suffix}_digest"]
            config_path = ROOT / panel["config_path"]
            assert (
                _sha256(config_path)
                == panel["config_sha256"]
                == (identities[f"h200_{profile}_config_{suffix}_sha256"])
            )
            for row in rows:
                worker_counts[row["worker_index"]] += 1
        assert worker_counts == {index: 40 for index in range(4)}
    assert len(all_candidate_plan_keys) == 320

    assessment = json.loads(assessment_path.read_text(encoding="utf-8"))
    assert _sha256(assessment_path) == identities["h200_preexecution_assessment_file_sha256"]
    assert assessment["assessment_digest"] == identities["h200_preexecution_assessment_digest"]
    assert assessment["overall_passed"] is False
    assert assessment["selected_profile"] is identities["h200_selected_profile"] is None
    assert assessment["failure_reason"] == "prospective_freeze_no_model_execution"
    assert assessment["checks"]["panel_completion"] == {
        "completed_runs": 0,
        "passed": False,
        "planned_runs": 160,
        "reason": "prospective_freeze_no_model_execution",
        "state_counts": {"candidate_not_selected_not_run": 320},
    }
    assert (
        len(assessment["intention_to_train"])
        == (identities["h200_preexecution_candidate_not_selected_rows"])
        == 320
    )
    assert all(
        row["state"] == "candidate_not_selected_not_run" and row["attempt_count"] == 0
        for row in assessment["intention_to_train"]
    )

    gate = json.loads(gate_path.read_text(encoding="utf-8"))
    assert _sha256(gate_path) == identities["h200_preexecution_gate_file_sha256"]
    assert gate["gate_digest"] == identities["h200_preexecution_gate_digest"]
    assert gate["overall_passed"] is identities["h200_preexecution_gate_overall_passed"] is False
    assert gate["assessment"] == assessment
    assert gate["authorization"] == {
        "g00f_remediation_passed": False,
        "g01_launch_authorized": False,
        "reason": "a_separately_reviewed_digest_bound_g01_bridge_is_required",
        "scope": "none",
    }


def test_official_g00e_gate_is_exactly_failed_and_has_no_sidecar() -> None:
    evidence_root = ROOT / "reproducibility" / "goalzendo" / "g00d-gate-20260811"
    recovered = json.loads(
        (evidence_root / "g00-gate-assessment-derived-pre-fix.json").read_text(encoding="utf-8")
    )
    assessment_path = evidence_root / "g00e-gate-assessment-v3.json"
    gate_path = evidence_root / "g00e-gate-v3.json"
    assessment = json.loads(assessment_path.read_text(encoding="utf-8"))
    gate = json.loads(gate_path.read_text(encoding="utf-8"))

    assert assessment == recovered
    assessment_body = {key: value for key, value in assessment.items() if key != "assessment_digest"}
    assert assessment["assessment_digest"] == _stable_digest(assessment_body)
    gate_body = {key: value for key, value in gate.items() if key != "gate_digest"}
    assert gate["gate_digest"] == _stable_digest(gate_body)
    assert gate["assessment"] == assessment
    assert gate["overall_passed"] is False
    assert {name for name, check in gate["checks"].items() if not check["passed"]} == {
        "rule_adapters",
        "constrained_scorer",
    }
    assert {name for name, check in gate["checks"].items() if check["passed"]} == {
        "dataset_integrity",
        "no_signal_chance",
        "optimizer_stability",
        "surface_leakage",
    }
    assert not (evidence_root / "g00e-numerical-bridge-v3.json").exists()


def test_g03_archives_bind_exact_nonauthorizing_engine_audits() -> None:
    studies = _studies(_load())
    for study_id, filename in (
        ("G03-E", "goalzendo-interactive-g03e-fp6ecac03c.tar"),
        ("G03-G", "goalzendo-interactive-g03g-fp24b6d1cc.tar.gz"),
    ):
        archive_path = ROOT / "reproducibility/goalzendo/frozen-sources" / filename
        fixture_sha, report, entry_count = _archive_engine_audit(archive_path)
        identities = studies[study_id]["identities"]
        assert fixture_sha == identities["engine_audit_fixture_sha256"]
        assert report.digest == identities["engine_audit_report_digest"]
        assert report.check_evidence_bundle_digest == identities["engine_audit_evidence_bundle_digest"]
        assert report.source_fingerprint == identities["source_fingerprint"]
        assert report.core_checks_passed is True
        assert report.all_checks_passed is False
        assert report.leakage_status == "insufficient_data"
        assert report.weight_updates_authorized is False
        if study_id == "G03-G":
            assert entry_count == identities["bundle_entry_count"] == 155

    provenance = interactive_source_provenance(ROOT / "src/goalzendo_interactive")
    assert provenance.fingerprint == studies["G03-G"]["identities"]["source_fingerprint"]
    assert len(provenance.files) == studies["G03-G"]["identities"]["source_file_count"]
    assert sum(item.size for item in provenance.files) == studies["G03-G"]["identities"]["source_total_bytes"]
    assert studies["G03-G"]["identities"]["offline_exact_tree_tests_passed"] == 322
    assert studies["G03-G"]["identities"]["offline_launch_scope"] == ("exact_registered_sft_rl_smoke_only")

    smoke = studies["G03-G"]
    evidence_path = ROOT / "reproducibility/goalzendo/g03g-smoke-20260811/evidence.json"
    evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    assert _sha256(evidence_path) == smoke["identities"]["smoke_evidence_sha256"]
    assert evidence["preflight"]["status"] == "passed"
    assert evidence["cells"]["sft"]["status"] == "failed_closed"
    assert evidence["cells"]["rl"]["status"] == "failed_closed"
    assert evidence["cells"]["sft"]["weight_update_occurred"] is False
    assert evidence["cells"]["rl"]["weight_update_occurred"] is False
    assert evidence["read_only_post_failure_diagnostic"]["is_smoke_retry"] is False
    assert evidence["authorization"]["scientific_launch"] is False
    assert not tuple((evidence_path.parent / "sft").glob("COMPLETE"))
    assert not tuple((evidence_path.parent / "rl").glob("COMPLETE"))

    smoke_root = ROOT / "artifacts-goalzendo-remote/g03-pinned-qwen-smoke-v1"
    if smoke_root.exists():
        assert not tuple(smoke_root.rglob("COMPLETE"))


def test_g03_v2_engineering_evidence_is_exact_and_nonauthorizing() -> None:
    study = _studies(_load())["G03-v2"]
    identities = study["identities"]

    assert (
        _sha256(ROOT / "src/goalzendo_interactive_v2/hypothesis_complete.py")
        == (identities["hypothesis_complete_source_sha256"])
    )
    assert (
        _sha256(ROOT / "src/goalzendo_interactive_v2/calibrated_leakage.py")
        == (identities["calibrated_leakage_source_sha256"])
    )
    assert (
        _sha256(ROOT / "src/goalzendo_interactive_v2/manifest_verifier.py")
        == (identities["manifest_verifier_source_sha256"])
    )
    assert (
        _sha256(ROOT / "src/goalzendo_interactive_v2/training_freeze.py")
        == identities["training_freeze_source_sha256"]
    )
    assert (
        _sha256(ROOT / "tests/goalzendo_interactive_v2/test_training_freeze.py")
        == identities["training_freeze_test_sha256"]
    )
    assert (
        _sha256(ROOT / "docs/goalzendo/g03-v2-training-freeze-engineering.md")
        == identities["training_freeze_engineering_record_sha256"]
    )
    assert (
        _sha256(ROOT / "src/goalzendo_interactive_v2/training_surface_bridge.py")
        == identities["training_surface_bridge_source_sha256"]
    )
    assert (
        _sha256(ROOT / "tests/goalzendo_interactive_v2/test_training_surface_bridge.py")
        == identities["training_surface_bridge_test_sha256"]
    )
    assert (
        _sha256(ROOT / "docs/goalzendo/g03-v2-training-surface-bridge-engineering.md")
        == identities["training_surface_bridge_engineering_record_sha256"]
    )
    assert (
        _sha256(ROOT / "src/goalzendo_interactive_v2/nested_opening_feasibility.py")
        == identities["nested_opening_feasibility_source_sha256"]
    )
    assert (
        _sha256(ROOT / "tests/goalzendo_interactive_v2/test_nested_opening_feasibility.py")
        == identities["nested_opening_feasibility_test_sha256"]
    )
    assert (
        _sha256(ROOT / "docs/goalzendo/g03-v2-nested-opening-feasibility-engineering.md")
        == identities["nested_opening_feasibility_engineering_record_sha256"]
    )
    assert (
        _sha256(ROOT / "src/goalzendo_interactive_v2/evaluation_census.py")
        == identities["evaluation_census_source_sha256"]
    )
    assert (
        _sha256(ROOT / "tests/goalzendo_interactive_v2/test_evaluation_census.py")
        == identities["evaluation_census_test_sha256"]
    )
    assert (
        _sha256(ROOT / "docs/goalzendo/g03-v2-evaluation-census-engineering.md")
        == identities["evaluation_census_engineering_record_sha256"]
    )
    assert (
        _sha256(ROOT / "src/goalzendo_interactive_v2/evaluation_census_execution.py")
        == identities["evaluation_census_execution_source_sha256"]
    )
    assert (
        _sha256(ROOT / "scripts/run_g03_v2_evaluation_census.py")
        == identities["evaluation_census_execution_runner_sha256"]
    )
    assert (
        _sha256(ROOT / "tests/goalzendo_interactive_v2/test_evaluation_census_execution.py")
        == identities["evaluation_census_execution_test_sha256"]
    )
    assert (
        _sha256(ROOT / "docs/goalzendo/g03-v2-evaluation-census-execution.md")
        == identities["evaluation_census_execution_engineering_record_sha256"]
    )
    assert (
        identities["accepted_v2_snapshot_source_tree_file_count"],
        identities["accepted_v2_snapshot_source_tree_sha256"],
    ) == (
        17,
        "8267436a36ce15add05681491462ebb6555b97dd45b02eccf3282680a3436fec",
    )
    assert identities["accepted_v2_snapshot_source_tree_normalization"] == (
        "source-tree-v1: regular non-symlink files recursively; exclude __pycache__, *.pyc, "
        "and .DS_Store; records exactly {path,sha256} with package-relative POSIX paths sorted "
        "by path; compact sorted-key UTF-8 JSON plus terminal LF"
    )
    assert identities["accepted_v2_snapshot_tests_passed"] == 198
    assert identities["independent_review_p0_findings"] == 0
    assert identities["independent_review_p1_findings"] == 0
    assert identities["production_census_prepared"] is False
    assert identities["production_census_externally_registered"] is False
    assert identities["production_census_executed"] is False
    assert identities["production_census_full_root_verified"] is False
    assert identities["production_nested_opening_feasibility_report_executed"] is False
    assert identities["full_manifest_surface_bound"] is False
    assert identities["frozen_manifest_runtime_bridge_verified"] is False
    assert identities["production_bank_authorized"] is False
    assert identities["model_execution_authorized"] is False
    assert identities["weight_updates_authorized"] is False
    assert study["execution"]["completed_runs"] == 0
    assert study["assessment"]["scientific_result_state"] == "not_started"

    evidence_by_role = {item["role"]: item for item in study["evidence"]}
    assert (
        evidence_by_role["prospective_plan_execution_receipt_engineering_primitive"]["sha256"]
        == identities["training_freeze_source_sha256"]
    )
    assert (
        evidence_by_role["prospective_plan_execution_receipt_contract_tests"]["sha256"]
        == (identities["training_freeze_test_sha256"])
    )
    assert (
        evidence_by_role["prospective_plan_execution_receipt_engineering_record"]["sha256"]
        == identities["training_freeze_engineering_record_sha256"]
    )
    assert (
        evidence_by_role["training_surface_bridge_engineering_primitive"]["sha256"]
        == identities["training_surface_bridge_source_sha256"]
    )
    assert (
        evidence_by_role["training_surface_bridge_contract_tests"]["sha256"]
        == identities["training_surface_bridge_test_sha256"]
    )
    assert (
        evidence_by_role["training_surface_bridge_engineering_record"]["sha256"]
        == identities["training_surface_bridge_engineering_record_sha256"]
    )
    assert (
        evidence_by_role["nested_opening_feasibility_engineering_primitive"]["sha256"]
        == identities["nested_opening_feasibility_source_sha256"]
    )
    assert (
        evidence_by_role["nested_opening_feasibility_contract_tests"]["sha256"]
        == identities["nested_opening_feasibility_test_sha256"]
    )
    assert (
        evidence_by_role["nested_opening_feasibility_engineering_record"]["sha256"]
        == identities["nested_opening_feasibility_engineering_record_sha256"]
    )
    assert (
        evidence_by_role["evaluation_opening_census_engineering_primitive"]["sha256"]
        == identities["evaluation_census_source_sha256"]
    )
    assert (
        evidence_by_role["evaluation_opening_census_contract_tests"]["sha256"]
        == identities["evaluation_census_test_sha256"]
    )
    assert (
        evidence_by_role["evaluation_opening_census_engineering_record"]["sha256"]
        == identities["evaluation_census_engineering_record_sha256"]
    )
    assert (
        evidence_by_role["evaluation_census_execution_lifecycle_engineering_primitive"]["sha256"]
        == identities["evaluation_census_execution_source_sha256"]
    )
    assert (
        evidence_by_role["evaluation_census_execution_lifecycle_runner"]["sha256"]
        == identities["evaluation_census_execution_runner_sha256"]
    )
    assert (
        evidence_by_role["evaluation_census_execution_lifecycle_contract_tests"]["sha256"]
        == identities["evaluation_census_execution_test_sha256"]
    )
    assert (
        evidence_by_role["evaluation_census_execution_lifecycle_engineering_record"]["sha256"]
        == identities["evaluation_census_execution_engineering_record_sha256"]
    )
