from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

import goalzendo_interactive.engine_audit as audit_module
from goalzendo_interactive.engine_audit import (
    ENGINE_CHECK_IDS,
    EngineAuditError,
    EngineAuditReport,
    WeightUpdateAuthorizationError,
    parse_engine_audit_report,
    require_weight_update_authorization,
    run_engineering_audit,
    serialize_engine_audit_report,
    verify_engineering_audit,
)
from goalzendo_interactive.schema import SCENE_COUNT


@pytest.fixture(scope="module")
def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def audit_report(repo_root: Path) -> EngineAuditReport:
    return run_engineering_audit(repo_root)


def test_report_binds_every_registered_engine_surface_and_exhaustive_roundtrip(
    audit_report: EngineAuditReport,
) -> None:
    assert tuple(check.check_id for check in audit_report.checks) == ENGINE_CHECK_IDS
    assert audit_report.core_checks_passed
    assert not audit_report.all_checks_passed
    assert audit_report.digest == (
        "2065f624bac9b182260b99f703b0ed36128e9f9b171fe22171792707f71c05d6"
    )
    assert audit_report.check_evidence_bundle_digest == (
        "937f74075e0e34f596cea945e684d45d3dfb6da3abcf50d8d869334c79faf24f"
    )
    assert audit_report.source_fingerprint == (
        "24b6d1cc60c09be3b6bbab22d7b7a5b09fef1c4250dd8eb6dc0a9d323aed0ada"
    )

    scenes = audit_report.check("scene_universe_roundtrip")
    actions = audit_report.check("canonical_test_action_roundtrip")
    assert scenes.status == actions.status == "pass"
    assert scenes.exhaustive and actions.exhaustive
    assert scenes.item_count == actions.item_count == SCENE_COUNT == 13_716
    assert scenes.evidence["unique_scene_count"] == SCENE_COUNT
    assert actions.evidence["unique_test_action_count"] == SCENE_COUNT
    assert actions.evidence["action_language_digest"] == audit_report.check(
        "public_dialogue_action_language"
    ).evidence["action_language_digest"]


def test_default_catalog_claim_is_explicitly_derived_not_independent(
    audit_report: EngineAuditReport,
) -> None:
    catalog = audit_report.check("rule_catalog_semantics")
    assert catalog.status == "pass"
    assert not catalog.exhaustive
    assert catalog.item_count == 7_300
    assert catalog.evidence["verification_level"] == "derived_internal_consistency_only"
    assert catalog.evidence["independent_direct_evaluation_count"] == 0
    assert catalog.evidence["independent_scene_evaluations"] == 0
    assert "catalog_semantics_not_independently_exhausted" in (
        audit_report.authorization_reasons
    )


def test_explicit_exhaustive_mode_counts_all_rules_and_fails_on_any_semantic_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = 0

    def succeeds(rule, vector) -> bool:
        nonlocal calls
        calls += 1
        return True

    monkeypatch.setattr(audit_module, "verify_truth_vector", succeeds)
    _, check = audit_module._catalog_check("exhaustive")
    assert calls == 7_300
    assert check.exhaustive
    assert check.evidence["verification_level"] == "independent_full_universe_evaluation"
    assert check.evidence["independent_direct_evaluation_count"] == 7_300
    assert check.evidence["independent_scene_evaluations"] == 7_300 * SCENE_COUNT

    monkeypatch.setattr(audit_module, "verify_truth_vector", lambda rule, vector: False)
    with pytest.raises(EngineAuditError, match="independent semantic verification failed"):
        audit_module._catalog_check("exhaustive")


def test_stored_fixtures_dependencies_interventions_and_reference_outcomes_are_exact(
    audit_report: EngineAuditReport,
) -> None:
    episode = audit_report.check("episode_fixture_manifest")
    intervention = audit_report.check("intervention_fixture_manifest")
    reference = audit_report.check("reference_expert_trajectories")
    partitions = audit_report.check("rule_identity_partitions")
    pairs = audit_report.check("eligible_target_shadow_pairs")

    assert episode.evidence["episode_bank_digest"] == (
        "54714717b307b142a123372f5bde9854fd42df6ba8dc4e8e9c79bb78fe231631"
    )
    assert intervention.evidence["intervention_bank_digest"] == (
        "e396fd0625ccc9509202d9bcff0d355b58136d1d014eb53bd965760ab2de3ca9"
    )
    assert intervention.evidence["source_episode_bank_digest"] == episode.evidence[
        "episode_bank_digest"
    ]
    assert episode.item_count == reference.item_count == 12
    assert intervention.item_count == 48
    assert intervention.evidence["selected_scene_count"] == 96
    assert partitions.item_count == 7_300
    assert pairs.item_count == 85_486

    assert reference.evidence["all_rule_equivalent"] is True
    assert reference.evidence["classification_correct"] == 156
    assert reference.evidence["classification_total"] == 156
    assert all(row["rule_equivalent"] for row in reference.evidence["rows"])
    assert all(
        row["classification_correct"] == row["terminal_count"]
        for row in reference.evidence["rows"]
    )


def test_underpowered_and_missing_leakage_can_never_authorize_weight_updates(
    audit_report: EngineAuditReport,
) -> None:
    leakage = audit_report.check("surface_leakage")
    assert leakage.status == "insufficient_data"
    assert audit_report.leakage_status == "insufficient_data"
    assert leakage.evidence["report"]["passed"] is False
    assert all(
        result["decision"] == "insufficient_data"
        for result in leakage.evidence["report"]["results"]
    )
    assert not audit_report.weight_updates_authorized
    assert "engineering_fixture_non_authorizing" in audit_report.authorization_reasons
    assert "surface_leakage_audit_underpowered" in audit_report.authorization_reasons
    with pytest.raises(WeightUpdateAuthorizationError, match="cannot authorize"):
        require_weight_update_authorization(audit_report)

    missing = audit_module._surface_leakage_check(None)
    missing_report = EngineAuditReport(
        audit_report.semantic_mode,
        audit_report.source_fingerprint,
        (*audit_report.checks[:-1], missing),
    )
    assert not missing_report.weight_updates_authorized
    assert missing_report.leakage_status == "missing"
    assert "surface_leakage_audit_missing" in missing_report.authorization_reasons
    with pytest.raises(WeightUpdateAuthorizationError, match="audit_missing"):
        require_weight_update_authorization(missing_report)


def test_report_is_canonical_tamper_evident_and_regenerates(
    audit_report: EngineAuditReport,
    repo_root: Path,
) -> None:
    encoded = serialize_engine_audit_report(audit_report)
    fixture = (
        repo_root
        / "tests"
        / "goalzendo_interactive"
        / "fixtures"
        / "g03-engine-audit-derived-v1.json"
    )
    stored = fixture.read_bytes()
    assert hashlib.sha256(stored).hexdigest() == (
        "b8a0c240bffe97c88358bf8a247c5d4c183454dfbc5d47a0b177a155f1bcf334"
    )
    assert stored == encoded.encode("ascii") + b"\n"
    parsed = parse_engine_audit_report(stored.removesuffix(b"\n").decode("ascii"))
    assert parsed == audit_report
    assert serialize_engine_audit_report(parsed) == encoded

    pretty = json.dumps(json.loads(encoded), indent=2)
    with pytest.raises(ValueError, match="not canonical"):
        parse_engine_audit_report(pretty)

    evidence_tamper = json.loads(encoded)
    evidence_tamper["checks"][0]["evidence"]["scene_count"] -= 1
    with pytest.raises(ValueError, match="evidence digest mismatch"):
        parse_engine_audit_report(json.dumps(evidence_tamper, separators=(",", ":")))

    authorization_tamper = json.loads(encoded)
    authorization_tamper["authorization"]["weight_updates_authorized"] = True
    with pytest.raises(ValueError, match="derived fields are inconsistent"):
        parse_engine_audit_report(
            json.dumps(authorization_tamper, separators=(",", ":"))
        )

    assert verify_engineering_audit(parsed, repo_root) is parsed
