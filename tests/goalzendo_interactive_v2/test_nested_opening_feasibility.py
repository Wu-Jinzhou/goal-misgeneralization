from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from goalzendo_interactive_v2 import nested_opening_feasibility as nested
from goalzendo_interactive_v2.evaluation_census import (
    PRODUCTION_MIRROR_ATTEMPTS_PER_STRATUM,
    build_evaluation_census_plan_v1,
    serialize_evaluation_census_plan_v1,
)
from goalzendo_interactive_v2.population_audit import build_supported_catalog_contract_v2


def _sha(label: str) -> str:
    return hashlib.sha256(label.encode("ascii")).hexdigest()


def _domain_digest(obj: object, *, domain: str) -> str:
    payload = json.dumps(
        obj,
        ensure_ascii=True,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")
    return hashlib.sha256(domain.encode("ascii") + b"\0" + payload).hexdigest()


@pytest.fixture(scope="module")
def engineering_artifacts() -> dict[str, Any]:
    parent = build_evaluation_census_plan_v1(
        _sha("nested-opening-parent-engineering-fixture"),
        attempts_per_formula_stratum=1,
        candidate_pool_size=32,
    )
    parent_text = serialize_evaluation_census_plan_v1(parent)
    parent_bytes_sha = hashlib.sha256(parent_text.encode("ascii")).hexdigest()
    plan = nested.build_nested_opening_construction_feasibility_plan_for_testing_v1(
        parent_text,
        expected_census_plan_digest=parent.digest,
        expected_census_plan_bytes_sha256=parent_bytes_sha,
        generator_seed=_sha("nested-opening-construction-engineering-fixture"),
    )
    plan_digest = nested.derive_nested_opening_construction_feasibility_plan_digest_for_testing_v1(
        plan,
        parent_text,
        expected_census_plan_digest=parent.digest,
        expected_census_plan_bytes_sha256=parent_bytes_sha,
    )
    plan_text = nested.serialize_nested_opening_construction_feasibility_plan_for_testing_v1(
        plan,
        parent_text,
        expected_census_plan_digest=parent.digest,
        expected_census_plan_bytes_sha256=parent_bytes_sha,
        expected_plan_digest=plan_digest,
    )
    report = nested.build_nested_opening_construction_feasibility_report_for_testing_v1(
        plan_text,
        census_plan_text=parent_text,
        expected_census_plan_digest=parent.digest,
        expected_census_plan_bytes_sha256=parent_bytes_sha,
        expected_plan_digest=plan_digest,
    )
    report_digest = nested.derive_nested_opening_construction_feasibility_report_digest_for_testing_v1(
        report,
        plan_text,
        census_plan_text=parent_text,
        expected_census_plan_digest=parent.digest,
        expected_census_plan_bytes_sha256=parent_bytes_sha,
        expected_plan_digest=plan_digest,
    )
    report_text = nested.serialize_nested_opening_construction_feasibility_report_for_testing_v1(
        report,
        plan_text,
        census_plan_text=parent_text,
        expected_census_plan_digest=parent.digest,
        expected_census_plan_bytes_sha256=parent_bytes_sha,
        expected_plan_digest=plan_digest,
        expected_report_digest=report_digest,
    )
    return {
        "parent": parent,
        "parent_text": parent_text,
        "parent_bytes_sha": parent_bytes_sha,
        "plan": plan,
        "plan_digest": plan_digest,
        "plan_text": plan_text,
        "report": report,
        "report_digest": report_digest,
        "report_text": report_text,
    }


def test_exact_production_builder_requires_144x2_parent_and_test_kind_cannot_promote(
    engineering_artifacts: dict[str, Any],
) -> None:
    parent = engineering_artifacts["parent"]
    with pytest.raises(nested.NestedOpeningFeasibilityV2Error, match="exact 144x2"):
        nested.build_nested_opening_construction_feasibility_plan_v1(
            engineering_artifacts["parent_text"],
            expected_census_plan_digest=parent.digest,
            expected_census_plan_bytes_sha256=engineering_artifacts["parent_bytes_sha"],
            generator_seed=_sha("production-refuses-reduced-parent"),
        )

    plan = engineering_artifacts["plan"]
    assert plan.plan_kind.endswith("-for-testing-v1")
    assert plan.status == "prospective_engineering_fixture_outcome_free_nonauthorizing"
    assert plan.engineering_budget_override
    assert not plan.exact_144x2_budget_complete
    plan_obj = json.loads(engineering_artifacts["plan_text"])
    assert not plan_obj["fixed_budget"]["uses_exact_144x2_budget"]
    with pytest.raises(nested.NestedOpeningFeasibilityV2Error, match="production serializer"):
        nested.serialize_nested_opening_construction_feasibility_plan_v1(
            plan,
            engineering_artifacts["parent_text"],
            expected_census_plan_digest=parent.digest,
            expected_census_plan_bytes_sha256=engineering_artifacts["parent_bytes_sha"],
            expected_plan_digest=engineering_artifacts["plan_digest"],
        )
    with pytest.raises(nested.NestedOpeningFeasibilityV2Error):
        nested.parse_nested_opening_construction_feasibility_plan_v1(
            engineering_artifacts["plan_text"],
            census_plan_text=engineering_artifacts["parent_text"],
            expected_census_plan_digest=parent.digest,
            expected_census_plan_bytes_sha256=engineering_artifacts["parent_bytes_sha"],
            expected_plan_digest=engineering_artifacts["plan_digest"],
        )


def test_production_plan_has_exact_144x2_identity_budget_without_executing_report() -> None:
    parent = build_evaluation_census_plan_v1(
        _sha("nested-opening-production-plan-only"),
        attempts_per_formula_stratum=PRODUCTION_MIRROR_ATTEMPTS_PER_STRATUM,
        candidate_pool_size=32,
    )
    parent_text = serialize_evaluation_census_plan_v1(parent)
    parent_sha = hashlib.sha256(parent_text.encode("ascii")).hexdigest()
    plan = nested.build_nested_opening_construction_feasibility_plan_v1(
        parent_text,
        expected_census_plan_digest=parent.digest,
        expected_census_plan_bytes_sha256=parent_sha,
        generator_seed=_sha("nested-opening-production-plan-only-generator"),
    )
    plan_digest = nested.derive_nested_opening_construction_feasibility_plan_digest_v1(
        plan,
        parent_text,
        expected_census_plan_digest=parent.digest,
        expected_census_plan_bytes_sha256=parent_sha,
    )
    plan_text = nested.serialize_nested_opening_construction_feasibility_plan_v1(
        plan,
        parent_text,
        expected_census_plan_digest=parent.digest,
        expected_census_plan_bytes_sha256=parent_sha,
        expected_plan_digest=plan_digest,
    )
    budget = json.loads(plan_text)["fixed_budget"]
    assert len(plan.attempts) == 144
    assert sum(len(attempt.members) for attempt in plan.attempts) == 288
    assert budget["planned_opening_assessment_count"] == 1_152
    assert budget["uses_exact_144x2_budget"]
    assert not budget["engineering_budget_override"]
    assert len({member.composed_rule_id for attempt in plan.attempts for member in attempt.members}) == 288
    with pytest.raises(nested.NestedOpeningFeasibilityV2Error, match="test-only serializer"):
        nested.serialize_nested_opening_construction_feasibility_plan_for_testing_v1(
            plan,
            parent_text,
            expected_census_plan_digest=parent.digest,
            expected_census_plan_bytes_sha256=parent_sha,
            expected_plan_digest=plan_digest,
        )


def test_frozen_union_vectors_slots_sides_prefixes_and_display_precommit(
    engineering_artifacts: dict[str, Any],
) -> None:
    plan = engineering_artifacts["plan"]
    for attempt in plan.attempts:
        assert tuple(member.noisy_p_target_side for member in attempt.members) in ((0, 1), (1, 0))
        for member in attempt.members:
            expected_union = (
                (5, 1, 1, 0, 0, 0, 1, 5) if member.noisy_p_target_side == 0 else (5, 1, 0, 0, 0, 1, 1, 5)
            )
            assert member.union_joint_truth_cell_counts == expected_union
            assert len(member.planned_slots) == 13
            assert len({(slot.joint_cell_index, slot.prefix_rank) for slot in member.planned_slots}) == 13
            assert tuple(
                (slot.prefix_rank, slot.slot_order_hash, slot.joint_cell_index)
                for slot in member.planned_slots
            ) == tuple(
                sorted(
                    (slot.prefix_rank, slot.slot_order_hash, slot.joint_cell_index)
                    for slot in member.planned_slots
                )
            )
            assert tuple(sorted(member.geometry_display_order)) == tuple(
                sorted(("a0_b0", "a1_b0", "a0_b2", "a1_b2"))
            )
            assert member.schedule_variants[1].endswith(str(member.noisy_p_target_side))
            assert member.schedule_variants[3].endswith(str(member.noisy_p_target_side))

    report = engineering_artifacts["report"]
    for attempt in report.attempts:
        for member in attempt.members:
            assert member.status == "completed"
            assert len(member.selection_steps) == 13
            assert all(step.outcome == "selected" for step in member.selection_steps)
            union = member.common_union_scene_indices_by_joint_cell
            assert union is not None
            assert tuple(map(len, union)) == member.member_plan.union_joint_truth_cell_counts
            assert member.openings is not None
            for opening in member.openings:
                counts = nested._variant_schedule(opening.schedule_variant)
                assert opening.opening_scene_indices_by_joint_cell == tuple(
                    tuple(union[cell][:count]) for cell, count in enumerate(counts)
                )


def test_exact_long_form_schema_keys_at_every_nested_feasibility_level(
    engineering_artifacts: dict[str, Any],
) -> None:
    plan = json.loads(engineering_artifacts["plan_text"])
    assert list(plan) == [
        "schema_version",
        "plan_kind",
        "status",
        "protocol_quartet_candidate",
        "authorization",
        "source_binding",
        "upstream_identity_plan_binding",
        "generator_binding",
        "fixed_budget",
        "selection_boundary",
        "attempts",
        "prospective_plan_digest",
    ]
    authorization_keys = [
        "scope",
        "g01_authorized",
        "g03_capability_launch_authorized",
        "g03_scientific_launch_authorized",
        "production_bank_authorized",
        "model_execution_authorized",
        "weight_updates_authorized",
        "launch_authorized",
    ]
    assert list(plan["authorization"]) == authorization_keys
    assert plan["authorization"] == {
        "scope": "nested_opening_construction_feasibility_only",
        "g01_authorized": False,
        "g03_capability_launch_authorized": False,
        "g03_scientific_launch_authorized": False,
        "production_bank_authorized": False,
        "model_execution_authorized": False,
        "weight_updates_authorized": False,
        "launch_authorized": False,
    }
    source_binding_keys = [
        "frozen_v1_source_fingerprint",
        "catalog_source_sha256",
        "query_source_sha256",
        "stage_partitions_source_sha256",
        "role_schema_source_sha256",
        "population_audit_source_sha256",
        "challenge_query_source_sha256",
        "evaluation_census_source_sha256",
        "nested_opening_feasibility_source_sha256",
        "source_manifest_digest",
    ]
    assert list(plan["source_binding"]) == source_binding_keys
    assert list(plan["upstream_identity_plan_binding"]) == [
        "canonical_census_plan_bytes_sha256",
        "canonical_census_plan_byte_count",
        "evaluation_census_plan_digest",
        "evaluation_census_source_manifest_digest",
        "evaluation_census_generator_contract_digest",
        "source_catalog_digest",
        "supported_catalog_digest",
        "supported_identity_count",
        "stage_partition_digest",
        "identity_schedule_digest",
        "mirror_attempt_count",
        "member_count",
    ]
    assert list(plan["generator_binding"]) == [
        "generator_schema_version",
        "generator_seed",
        "generator_contract",
        "generator_contract_digest",
        "generator_source_sha256",
    ]
    assert list(plan["fixed_budget"]) == [
        "formula_stratum_count",
        "attempts_per_formula_stratum",
        "mirror_attempt_count",
        "members_per_attempt",
        "planned_member_construction_count",
        "derived_openings_per_member",
        "planned_opening_assessment_count",
        "common_union_slots_per_member",
        "candidate_window_k",
        "no_early_stop",
        "no_identity_replacement",
        "uses_exact_144x2_budget",
        "engineering_budget_override",
    ]
    selection_boundary_keys = [
        "external_registration_reference",
        "registered_power_artifact_digest",
        "positive_m_q_quota",
        "selected_m_q_cells",
        "matched_bank_size",
        "matcher_specification",
        "canonical_integer_matching_cost",
        "oversupply_selection_rule",
        "aggregate_balance_target",
        "renderer_assignment",
        "token_length_bins",
        "challenge_panel_generator",
        "intervention_generator",
        "production_manifest_digest",
        "pre_evaluation_checkpoint_digest",
        "runtime_receipt_digest",
    ]
    assert list(plan["selection_boundary"]) == selection_boundary_keys
    assert all(value is None for value in plan["selection_boundary"].values())
    attempt_plan = plan["attempts"][0]
    assert list(attempt_plan) == [
        "formula_stratum",
        "canonical_position",
        "upstream_attempt_digest",
        "placard_rule_id",
        "placard_truth_digest",
        "literal_rule_id",
        "literal_truth_digest",
        "members",
        "attempt_plan_digest",
    ]
    member_plan = attempt_plan["members"][0]
    assert list(member_plan) == [
        "member_position",
        "composed_rule_id",
        "composed_truth_digest",
        "triple_digest",
        "noisy_p_target_side",
        "schedule_variants",
        "union_cell_counts",
        "planned_union_slots",
        "planned_union_slots_digest",
        "planned_geometry_display_order",
        "planned_geometry_display_order_digest",
        "member_plan_digest",
    ]
    assert list(member_plan["planned_union_slots"][0]) == [
        "slot_position",
        "cell_prefix_rank",
        "joint_cell_index",
        "joint_cell_slug",
        "affected_geometry_slugs",
        "slot_order_hash",
    ]

    report = json.loads(engineering_artifacts["report_text"])
    assert list(report) == [
        "schema_version",
        "report_kind",
        "status",
        "protocol_quartet_candidate",
        "authorization",
        "source_binding",
        "exact_plan_binding",
        "fixed_budget_accounting",
        "attempts",
        "ranking_tables",
        "observed_m_q_summary",
        "construction_claim_boundary",
        "selection_boundary",
        "observed_report_digest",
    ]
    assert list(report["authorization"]) == authorization_keys
    assert report["authorization"] == plan["authorization"]
    assert list(report["source_binding"]) == source_binding_keys
    assert list(report["exact_plan_binding"]) == [
        "prospective_plan_digest",
        "canonical_plan_bytes_sha256",
        "canonical_plan_byte_count",
        "upstream_census_plan_digest",
        "upstream_census_plan_bytes_sha256",
        "generator_contract_digest",
        "identity_schedule_digest",
    ]
    assert list(report["fixed_budget_accounting"]) == [
        "mirror_attempt_count",
        "member_record_count",
        "planned_opening_assessment_count",
        "completed_opening_assessment_count",
        "common_union_completed_count",
        "bounded_slot_failure_count",
        "nested_opening_feasibility_witness_count",
        "mirror_pair_feasibility_witness_count",
        "all_planned_attempts_preserved",
        "early_stop_used",
        "identity_replacement_used",
        "exact_144x2_budget_complete",
    ]
    attempt_observation = report["attempts"][0]
    assert list(attempt_observation) == [
        "attempt_plan",
        "members",
        "member_union_overlap_scene_indices",
        "member_union_overlap_digest",
        "mirror_pair_disjoint",
        "mirror_pair_nested_opening_feasibility_witness",
        "reason_codes",
        "protocol_quartet_candidate",
        "attempt_observation_digest",
    ]
    member_observation = attempt_observation["members"][0]
    assert list(member_observation) == [
        "member_plan",
        "binding",
        "construction_status",
        "selection_steps",
        "selection_trace_digest",
        "opening_union_scene_ids_by_joint_cell",
        "opening_union_digest",
        "remaining_scene_count_by_joint_cell",
        "openings",
        "exact_match_evidence",
        "reason_codes",
        "nested_opening_construction_feasibility_witness",
        "protocol_quartet_candidate",
        "member_observation_digest",
    ]
    assert list(member_observation["binding"]) == ["member_plan_digest"]
    assert list(member_observation["selection_steps"][0]) == [
        "slot_position",
        "cell_prefix_rank",
        "joint_cell_index",
        "joint_cell_slug",
        "affected_geometry_slugs",
        "survivor_counts_before",
        "available_remaining_scene_count",
        "candidate_window_count",
        "candidate_window_digest",
        "candidate_count_examined",
        "outcome",
        "selected_candidate_window_rank",
        "selected_scene_index",
        "survivor_counts_after",
        "selection_step_digest",
    ]
    assert list(member_observation["openings"][0]) == [
        "geometry_slug",
        "schedule_variant",
        "a_error_target_side",
        "opening_scene_ids_by_joint_cell",
        "opening_scene_indices_cell_major",
        "assessment",
        "derived_opening_digest",
    ]
    exact_match = member_observation["exact_match_evidence"]
    assert list(exact_match) == [
        "required_fields",
        "per_geometry_rows",
        "common_key",
        "passed",
        "exact_match_digest",
    ]
    assert exact_match["required_fields"] == [
        "version_space_size",
        "minimax_depth",
        "target_formula_stratum",
        "greedy_reference_query_count",
        "best_first_query_branch_size_pair",
    ]
    assert list(exact_match["per_geometry_rows"][0]) == [
        "geometry_slug",
        "version_space_size",
        "minimax_depth",
        "target_formula_stratum",
        "greedy_reference_query_count",
        "best_first_query_branch_size_pair",
    ]
    if exact_match["common_key"] is None:
        assert exact_match["passed"] is False
    else:
        assert list(exact_match["common_key"]) == [
            "version_space_size",
            "minimax_depth",
            "target_formula_stratum",
            "greedy_reference_query_count",
            "best_first_query_branch_size_pair",
        ]
    assert list(report["observed_m_q_summary"]) == ["rows", "m_q_summary_digest"]
    assert list(report["observed_m_q_summary"]["rows"][0]) == [
        "m",
        "q",
        "positive_quota",
        "completed_opening_count",
        "salient_opening_count",
        "individual_cell_witness_count",
        "nested_opening_feasibility_witness_count",
        "mirror_pair_feasibility_witness_count",
    ]
    assert list(report["selection_boundary"]) == selection_boundary_keys
    assert all(value is None for value in report["selection_boundary"].values())

    true_claim_keys = [
        "engineering_nested_opening_construction_feasibility_described",
        "exact_identity_schedule_replayed",
        "fixed_budget_ledger_complete",
        "common_union_prefix_contract_rederived",
        "completed_opening_assessments_rederived",
    ]
    false_claim_keys = [
        "protocol_quartet_candidate",
        "production_nested_opening_pool_frozen",
        "positive_m_q_quota_selected",
        "matched_bank_size_selected",
        "difficulty_matcher_run",
        "oversupply_selected",
        "aggregate_balance_audited",
        "planned_display_order_balance_verified",
        "renderer_assignment_bound",
        "renderer_balance_audited",
        "model_visible_demonstration_order_bound",
        "rendered_prompt_bytes_bound",
        "rendered_token_length_bound",
        "tokenizer_artifact_bound",
        "challenge_reservoir_materialized",
        "eleven_panels_materialized",
        "panel_version_space_coverage_verified",
        "minimum_challenge_cores_materialized",
        "primary_panel_salt_registered",
        "primary_panel_resolved",
        "common_untouched_panel_runtime_selected",
        "intervention_bank_materialized",
        "selected_bank_cross_triple_scene_disjointness_verified",
        "cross_mirror_unit_scene_disjointness_verified",
        "cross_stage_scene_disjointness_verified",
        "evaluation_checkpoint_bound",
        "runtime_episode_order_bound",
        "context_reset_observed",
        "weight_update_absence_observed",
        "private_ast_branch_observed",
        "ast_blind_terminal_replay_observed",
        "model_calls_observed",
        "model_outcomes_present",
        "runtime_measurements_present",
        "external_registration_verified",
        "registered_power_verified",
        "production_manifest_bound",
        "production_bank_authorized",
        "g01_authorized",
        "g03_capability_launch_authorized",
        "g03_scientific_launch_authorized",
        "launch_authorized",
    ]
    claim_boundary = report["construction_claim_boundary"]
    assert list(claim_boundary) == [*true_claim_keys, *false_claim_keys]
    assert [key for key, value in claim_boundary.items() if value is True] == true_claim_keys
    assert [key for key, value in claim_boundary.items() if value is False] == false_claim_keys

    old_plan_aliases = {
        "seed",
        "contract",
        "contract_digest",
        "source_sha256",
        "parent_attempt_digest",
        "union_joint_truth_cell_counts",
        "planned_common_union_slot_count",
        "planned_slots",
        "slot_set_digest",
        "geometry_display_order",
        "geometry_display_binding_digest",
        "prefix_rank",
    }
    feasibility_plan_keys = set(plan["generator_binding"])
    feasibility_plan_keys.update(attempt_plan, member_plan, member_plan["planned_union_slots"][0])
    assert old_plan_aliases.isdisjoint(feasibility_plan_keys)
    old_report_aliases = {
        "member_construction_count",
        "common_union_completion_count",
        "bounded_window_failure_count",
        "exact_member_plan_binding",
        "common_union_scene_indices_by_joint_cell",
        "common_union_digest",
        "exact_match",
        "member_feasibility_witness",
        "mirror_overlap_scene_indices",
        "mirror_overlap_digest",
        "mirror_pair_feasibility_witness",
        "available_candidate_count",
        "examined_candidate_count",
        "selected_candidate_rank",
        "opening_scene_indices_by_joint_cell",
        "construction_scene_indices_cell_major",
        "observed_m",
        "exact_minimax_depth",
        "best_first_query_branch_sizes",
        "individual_feasibility_witness_count",
        "member_feasibility_witness_count",
    }
    feasibility_report_keys = set(report["fixed_budget_accounting"])
    feasibility_report_keys.update(
        attempt_observation,
        member_observation,
        member_observation["selection_steps"][0],
        member_observation["openings"][0],
        exact_match,
        exact_match["per_geometry_rows"][0],
        report["observed_m_q_summary"],
        report["observed_m_q_summary"]["rows"][0],
    )
    assert old_report_aliases.isdisjoint(feasibility_report_keys)


def test_selection_is_atomic_and_k32_exhaustion_is_typed(
    engineering_artifacts: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    report = engineering_artifacts["report"]
    first_selected = report.attempts[0].members[0].selection_steps[0]
    assert first_selected.survivor_counts_before == (6_970, 6_970, 6_970, 6_970)
    assert first_selected.survivor_counts_after is not None
    assert all(count >= 8 for count in first_selected.survivor_counts_after)
    for index, slug in enumerate(("a0_b0", "a1_b0", "a0_b2", "a1_b2")):
        if slug not in first_selected.affected_geometry_slugs:
            assert first_selected.survivor_counts_after[index] == first_selected.survivor_counts_before[index]

    plan = engineering_artifacts["plan"]
    monkeypatch.setattr(nested, "SUPPORTED_VERSION_SPACE_FLOOR", 6_971)
    failed = nested._construct_member(plan, plan.attempts[0], plan.attempts[0].members[0], {})
    assert failed.status == "bounded_window_failure"
    assert len(failed.selection_steps) == 1
    step = failed.selection_steps[0]
    assert step.outcome == "bounded_window_exhausted"
    assert step.candidate_window_count == 32
    assert step.examined_candidate_count == 32
    assert step.selected_candidate_rank is None
    assert step.selected_scene_index is None
    assert step.survivor_counts_after is None
    assert failed.common_union_scene_indices_by_joint_cell is None
    assert failed.openings is None
    assert failed.exact_match is None


def test_first_selected_candidate_is_exact_fixed_window_first_pass(
    engineering_artifacts: dict[str, Any],
) -> None:
    plan = engineering_artifacts["plan"]
    attempt_plan = plan.attempts[0]
    member_plan = attempt_plan.members[0]
    step = engineering_artifacts["report"].attempts[0].members[0].selection_steps[0]
    catalog, entries = nested._catalog_entries()
    placard = entries[attempt_plan.placard_rule_id]
    literal = entries[attempt_plan.literal_rule_id]
    composed = entries[member_plan.composed_rule_id]
    pools = nested._scene_pools(placard, literal, composed)
    orders = nested._fixed_scene_orders(
        plan.generator_binding.seed,
        attempt_plan,
        member_plan,
        pools,
    )
    window = orders[step.joint_cell_index][: nested.CANDIDATE_WINDOW_K]
    assert step.selected_candidate_rank is not None
    assert step.selected_scene_index == window[step.selected_candidate_rank - 1]
    supported = build_supported_catalog_contract_v2().supported_indices
    declared_label = bool(step.joint_cell_index & 0b100)
    for preceding in window[: step.selected_candidate_rank - 1]:
        affected_sizes = tuple(
            len(
                tuple(
                    rule_index
                    for rule_index in supported
                    if catalog[rule_index].truth[preceding] is declared_label
                )
            )
            for _slug in step.affected_geometry_slugs
        )
        assert any(size < nested.SUPPORTED_VERSION_SPACE_FLOOR for size in affected_sizes)


def test_report_accounting_exact_match_overlap_and_all_false_null_boundaries(
    engineering_artifacts: dict[str, Any],
) -> None:
    report = engineering_artifacts["report"]
    obj = json.loads(engineering_artifacts["report_text"])
    accounting = obj["fixed_budget_accounting"]
    assert accounting["mirror_attempt_count"] == 9
    assert accounting["member_record_count"] == 18
    assert accounting["planned_opening_assessment_count"] == 72
    assert accounting["completed_opening_assessment_count"] == 72
    assert accounting["common_union_completed_count"] == 18
    assert accounting["bounded_slot_failure_count"] == 0
    assert accounting["all_planned_attempts_preserved"]
    assert not accounting["early_stop_used"]
    assert not accounting["identity_replacement_used"]
    assert not accounting["exact_144x2_budget_complete"]

    assert len(obj["observed_m_q_summary"]["rows"]) == 36
    assert all(row["positive_quota"] is None for row in obj["observed_m_q_summary"]["rows"])
    assert obj["selection_boundary"] == nested._SELECTION_BOUNDARY
    claim = obj["construction_claim_boundary"]
    assert {key for key, value in claim.items() if value} == set(nested._TRUE_CONSTRUCTION_CLAIMS)
    assert all(claim[key] is False for key in nested._FALSE_CONSTRUCTION_CLAIMS)
    assert obj["protocol_quartet_candidate"] is False
    assert all(attempt.protocol_quartet_candidate is False for attempt in report.attempts)
    assert all(
        member.protocol_quartet_candidate is False
        for attempt in report.attempts
        for member in attempt.members
    )
    for attempt in report.attempts:
        if attempt.mirror_overlap_scene_indices:
            assert not attempt.mirror_pair_disjoint
            assert not attempt.mirror_pair_feasibility_witness


def test_exact_match_mismatch_and_overlap_cannot_be_laundered(
    engineering_artifacts: dict[str, Any],
) -> None:
    member = engineering_artifacts["report"].attempts[0].members[0]
    assert member.exact_match is not None
    row = member.exact_match.rows[0]
    changed_row = replace(row, observed_m=row.observed_m + 1)
    changed_rows = (changed_row, *member.exact_match.rows[1:])
    with pytest.raises(nested.NestedOpeningFeasibilityV2Error):
        replace(member.exact_match, rows=changed_rows)

    attempt = engineering_artifacts["report"].attempts[0]
    with pytest.raises(nested.NestedOpeningFeasibilityV2Error):
        replace(
            attempt,
            mirror_overlap_scene_indices=(0,),
            mirror_pair_disjoint=True,
            mirror_pair_feasibility_witness=True,
        )


def test_plan_canonical_expected_digest_bytes_and_typed_tamper_rejection(
    engineering_artifacts: dict[str, Any],
) -> None:
    plan = engineering_artifacts["plan"]
    parent = engineering_artifacts["parent"]
    parsed = nested.parse_nested_opening_construction_feasibility_plan_for_testing_v1(
        engineering_artifacts["plan_text"],
        census_plan_text=engineering_artifacts["parent_text"],
        expected_census_plan_digest=parent.digest,
        expected_census_plan_bytes_sha256=engineering_artifacts["parent_bytes_sha"],
        expected_plan_digest=engineering_artifacts["plan_digest"],
    )
    assert parsed == plan

    with pytest.raises(nested.NestedOpeningFeasibilityV2Error):
        nested.parse_nested_opening_construction_feasibility_plan_for_testing_v1(
            engineering_artifacts["plan_text"],
            census_plan_text=engineering_artifacts["parent_text"],
            expected_census_plan_digest=parent.digest,
            expected_census_plan_bytes_sha256=_sha("wrong-parent-bytes"),
            expected_plan_digest=engineering_artifacts["plan_digest"],
        )
    with pytest.raises(nested.NestedOpeningFeasibilityV2Error):
        nested.parse_nested_opening_construction_feasibility_plan_for_testing_v1(
            engineering_artifacts["plan_text"],
            census_plan_text=engineering_artifacts["parent_text"],
            expected_census_plan_digest=parent.digest,
            expected_census_plan_bytes_sha256=engineering_artifacts["parent_bytes_sha"],
            expected_plan_digest=_sha("wrong-nested-plan-digest"),
        )

    duplicate = engineering_artifacts["plan_text"].replace(
        '"plan_kind":', '"plan_kind":"duplicate","plan_kind":', 1
    )
    with pytest.raises(nested.NestedOpeningFeasibilityV2Error, match="duplicate JSON"):
        nested.parse_nested_opening_construction_feasibility_plan_for_testing_v1(
            duplicate,
            census_plan_text=engineering_artifacts["parent_text"],
            expected_census_plan_digest=parent.digest,
            expected_census_plan_bytes_sha256=engineering_artifacts["parent_bytes_sha"],
            expected_plan_digest=engineering_artifacts["plan_digest"],
        )
    reordered = json.loads(engineering_artifacts["plan_text"])
    digest = reordered.pop("prospective_plan_digest")
    reordered_text = (
        json.dumps(
            {"prospective_plan_digest": digest, **reordered},
            ensure_ascii=True,
            allow_nan=False,
            separators=(",", ":"),
        )
        + "\n"
    )
    with pytest.raises(nested.NestedOpeningFeasibilityV2Error):
        nested.parse_nested_opening_construction_feasibility_plan_for_testing_v1(
            reordered_text,
            census_plan_text=engineering_artifacts["parent_text"],
            expected_census_plan_digest=parent.digest,
            expected_census_plan_bytes_sha256=engineering_artifacts["parent_bytes_sha"],
            expected_plan_digest=engineering_artifacts["plan_digest"],
        )
    boolean = json.loads(engineering_artifacts["plan_text"])
    boolean["fixed_budget"]["attempts_per_formula_stratum"] = True
    boolean_text = json.dumps(boolean, ensure_ascii=True, allow_nan=False, separators=(",", ":")) + "\n"
    with pytest.raises(nested.NestedOpeningFeasibilityV2Error):
        nested.parse_nested_opening_construction_feasibility_plan_for_testing_v1(
            boolean_text,
            census_plan_text=engineering_artifacts["parent_text"],
            expected_census_plan_digest=parent.digest,
            expected_census_plan_bytes_sha256=engineering_artifacts["parent_bytes_sha"],
            expected_plan_digest=engineering_artifacts["plan_digest"],
        )


def test_report_cross_parser_canonical_replay_and_binding_laundering_rejection(
    engineering_artifacts: dict[str, Any],
) -> None:
    parent = engineering_artifacts["parent"]
    report = engineering_artifacts["report"]
    with pytest.raises(nested.NestedOpeningFeasibilityV2Error, match="production serializer"):
        nested.serialize_nested_opening_construction_feasibility_report_v1(
            report,
            engineering_artifacts["plan_text"],
            census_plan_text=engineering_artifacts["parent_text"],
            expected_census_plan_digest=parent.digest,
            expected_census_plan_bytes_sha256=engineering_artifacts["parent_bytes_sha"],
            expected_plan_digest=engineering_artifacts["plan_digest"],
            expected_report_digest=engineering_artifacts["report_digest"],
        )
    parsed = nested.parse_nested_opening_construction_feasibility_report_for_testing_v1(
        engineering_artifacts["report_text"],
        plan_text=engineering_artifacts["plan_text"],
        census_plan_text=engineering_artifacts["parent_text"],
        expected_census_plan_digest=parent.digest,
        expected_census_plan_bytes_sha256=engineering_artifacts["parent_bytes_sha"],
        expected_plan_digest=engineering_artifacts["plan_digest"],
        expected_report_digest=engineering_artifacts["report_digest"],
    )
    assert parsed == report
    assert (
        nested.verify_nested_opening_construction_feasibility_report_for_testing_v1(
            report,
            engineering_artifacts["plan_text"],
            census_plan_text=engineering_artifacts["parent_text"],
            expected_census_plan_digest=parent.digest,
            expected_census_plan_bytes_sha256=engineering_artifacts["parent_bytes_sha"],
            expected_plan_digest=engineering_artifacts["plan_digest"],
            expected_report_digest=engineering_artifacts["report_digest"],
        )
        == report
    )
    with pytest.raises(nested.NestedOpeningFeasibilityV2Error):
        nested.parse_nested_opening_construction_feasibility_plan_for_testing_v1(
            engineering_artifacts["report_text"],
            census_plan_text=engineering_artifacts["parent_text"],
            expected_census_plan_digest=parent.digest,
            expected_census_plan_bytes_sha256=engineering_artifacts["parent_bytes_sha"],
            expected_plan_digest=engineering_artifacts["plan_digest"],
        )

    for field_name in (
        "prospective_plan_digest",
        "upstream_census_plan_digest",
        "upstream_census_plan_bytes_sha256",
        "generator_contract_digest",
        "identity_schedule_digest",
    ):
        changed_report = replace(report, **{field_name: _sha(f"launder-{field_name}")})
        with pytest.raises(nested.NestedOpeningFeasibilityV2Error):
            nested.serialize_nested_opening_construction_feasibility_report_for_testing_v1(
                changed_report,
                engineering_artifacts["plan_text"],
                census_plan_text=engineering_artifacts["parent_text"],
                expected_census_plan_digest=parent.digest,
                expected_census_plan_bytes_sha256=engineering_artifacts["parent_bytes_sha"],
                expected_plan_digest=engineering_artifacts["plan_digest"],
                expected_report_digest=engineering_artifacts["report_digest"],
            )

    tampered = json.loads(engineering_artifacts["report_text"])
    tampered["attempts"][0]["members"][0]["selection_steps"][0]["slot_position"] = True
    tampered_text = json.dumps(tampered, ensure_ascii=True, allow_nan=False, separators=(",", ":")) + "\n"
    with pytest.raises(nested.NestedOpeningFeasibilityV2Error):
        nested.parse_nested_opening_construction_feasibility_report_for_testing_v1(
            tampered_text,
            plan_text=engineering_artifacts["plan_text"],
            census_plan_text=engineering_artifacts["parent_text"],
            expected_census_plan_digest=parent.digest,
            expected_census_plan_bytes_sha256=engineering_artifacts["parent_bytes_sha"],
            expected_plan_digest=engineering_artifacts["plan_digest"],
            expected_report_digest=engineering_artifacts["report_digest"],
        )


def test_deterministic_replay_display_tamper_and_plan_binding_replay_boundary(
    engineering_artifacts: dict[str, Any],
) -> None:
    parent = engineering_artifacts["parent"]
    plan = engineering_artifacts["plan"]
    rebuilt = nested.build_nested_opening_construction_feasibility_plan_for_testing_v1(
        engineering_artifacts["parent_text"],
        expected_census_plan_digest=parent.digest,
        expected_census_plan_bytes_sha256=engineering_artifacts["parent_bytes_sha"],
        generator_seed=plan.generator_binding.seed,
    )
    rebuilt_text = nested.serialize_nested_opening_construction_feasibility_plan_for_testing_v1(
        rebuilt,
        engineering_artifacts["parent_text"],
        expected_census_plan_digest=parent.digest,
        expected_census_plan_bytes_sha256=engineering_artifacts["parent_bytes_sha"],
        expected_plan_digest=engineering_artifacts["plan_digest"],
    )
    assert rebuilt == plan
    assert rebuilt_text == engineering_artifacts["plan_text"]
    replayed_report = nested.build_nested_opening_construction_feasibility_report_for_testing_v1(
        rebuilt_text,
        census_plan_text=engineering_artifacts["parent_text"],
        expected_census_plan_digest=parent.digest,
        expected_census_plan_bytes_sha256=engineering_artifacts["parent_bytes_sha"],
        expected_plan_digest=engineering_artifacts["plan_digest"],
    )
    assert replayed_report == engineering_artifacts["report"]
    assert (
        json.loads(engineering_artifacts["report_text"])["observed_report_digest"]
        == engineering_artifacts["report_digest"]
    )

    display_tampered = json.loads(engineering_artifacts["plan_text"])
    member = display_tampered["attempts"][0]["members"][0]
    member["planned_geometry_display_order"] = list(reversed(member["planned_geometry_display_order"]))
    member["planned_geometry_display_order_digest"] = _sha("forged-display-binding")
    display_tampered_text = (
        json.dumps(
            display_tampered,
            ensure_ascii=True,
            allow_nan=False,
            separators=(",", ":"),
        )
        + "\n"
    )
    with pytest.raises(nested.NestedOpeningFeasibilityV2Error):
        nested.parse_nested_opening_construction_feasibility_plan_for_testing_v1(
            display_tampered_text,
            census_plan_text=engineering_artifacts["parent_text"],
            expected_census_plan_digest=parent.digest,
            expected_census_plan_bytes_sha256=engineering_artifacts["parent_bytes_sha"],
            expected_plan_digest=engineering_artifacts["plan_digest"],
        )

    for field_name in (
        "evaluation_census_plan_digest",
        "evaluation_census_source_manifest_digest",
        "evaluation_census_generator_contract_digest",
        "source_catalog_digest",
        "supported_catalog_digest",
        "stage_partition_digest",
        "canonical_census_plan_bytes_sha256",
    ):
        changed_binding = replace(
            plan.upstream_identity_plan_binding,
            **{field_name: _sha(f"launder-plan-{field_name}")},
        )
        changed_plan = replace(plan, upstream_identity_plan_binding=changed_binding)
        with pytest.raises(nested.NestedOpeningFeasibilityV2Error):
            nested.serialize_nested_opening_construction_feasibility_plan_for_testing_v1(
                changed_plan,
                engineering_artifacts["parent_text"],
                expected_census_plan_digest=parent.digest,
                expected_census_plan_bytes_sha256=engineering_artifacts["parent_bytes_sha"],
                expected_plan_digest=engineering_artifacts["plan_digest"],
            )

    forged_source_sha = _sha("launder-plan-source")
    changed_manifest = plan.source_binding._manifest_obj()
    changed_manifest["nested_opening_feasibility_source_sha256"] = forged_source_sha
    changed_source = replace(
        plan.source_binding,
        nested_opening_feasibility_source_sha256=forged_source_sha,
        source_manifest_digest=_domain_digest(
            changed_manifest,
            domain="goalzendo-interactive-v2-nested-opening-feasibility-source-manifest-v1",
        ),
    )
    changed_generator_source = replace(
        plan.generator_binding,
        source_sha256=forged_source_sha,
    )
    changed_plan = replace(
        plan,
        source_binding=changed_source,
        generator_binding=changed_generator_source,
    )
    with pytest.raises(nested.NestedOpeningFeasibilityV2Error):
        nested.serialize_nested_opening_construction_feasibility_plan_for_testing_v1(
            changed_plan,
            engineering_artifacts["parent_text"],
            expected_census_plan_digest=parent.digest,
            expected_census_plan_bytes_sha256=engineering_artifacts["parent_bytes_sha"],
            expected_plan_digest=engineering_artifacts["plan_digest"],
        )

    with pytest.raises(nested.NestedOpeningFeasibilityV2Error):
        replace(
            plan.generator_binding,
            contract_digest=_sha("launder-plan-generator-contract"),
        )


def test_plan_and_report_have_no_public_unverified_claim_projection(
    engineering_artifacts: dict[str, Any],
) -> None:
    for artifact in (engineering_artifacts["plan"], engineering_artifacts["report"]):
        assert not hasattr(artifact, "as_obj")
        assert not hasattr(artifact, "digest")


def test_source_contains_no_legacy_quartet_or_resource_objects() -> None:
    source = Path(nested.__file__).read_text(encoding="utf-8")
    for forbidden in (
        "COfficialEvaluationQuartetV2",
        "EvaluationSharedBindingV2",
        "build_c_official_evaluation_quartet_v2",
        "build_c_official_evaluation_mirror_unit_v2",
    ):
        assert forbidden not in source


def test_report_verification_reexecutes_construction_without_report_cache(
    engineering_artifacts: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    def changed_constructor(*_args: Any, **_kwargs: Any) -> Any:
        raise nested.NestedOpeningFeasibilityV2Error("fresh construction hook reached")

    monkeypatch.setattr(nested, "_construct_attempt", changed_constructor)
    parent = engineering_artifacts["parent"]
    report = engineering_artifacts["report"]
    with pytest.raises(nested.NestedOpeningFeasibilityV2Error, match="fresh construction hook"):
        nested.verify_nested_opening_construction_feasibility_report_for_testing_v1(
            report,
            engineering_artifacts["plan_text"],
            census_plan_text=engineering_artifacts["parent_text"],
            expected_census_plan_digest=parent.digest,
            expected_census_plan_bytes_sha256=engineering_artifacts["parent_bytes_sha"],
            expected_plan_digest=engineering_artifacts["plan_digest"],
            expected_report_digest=engineering_artifacts["report_digest"],
        )


def test_scene_order_has_no_geometry_display_m_q_salience_or_outcome_inputs() -> None:
    contract = nested._generator_contract_obj()
    assert contract["scene_order_inputs"] == [
        "generator_seed",
        "parent_attempt_digest",
        "member_position",
        "triple_digest",
        "noisy_p_target_side",
        "joint_cell_index",
        "scene_index",
    ]
    forbidden = {
        "geometry",
        "display_order",
        "m",
        "q",
        "salience",
        "assessment",
        "outcome",
    }
    assert forbidden.isdisjoint(contract["scene_order_inputs"])
    assert not contract["selection_uses_m_q_salience_or_assessor_outcomes"]
    assert contract["display_order_is_private_precommitted_metadata_only"]
    assert contract["mirror_overlap_is_diagnostic_not_acceptance_gate"]
    assert "protocol_quartet_candidate" not in contract
