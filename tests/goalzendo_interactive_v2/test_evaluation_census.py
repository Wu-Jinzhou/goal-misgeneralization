from __future__ import annotations

import hashlib
import json
from collections import Counter
from collections.abc import Iterator
from typing import Any

import pytest

from goalzendo_interactive import SCENE_COUNT, build_rule_catalog
from goalzendo_interactive_v2.bank_audits import (
    EvaluationSharedBindingV2,
    build_c_official_evaluation_mirror_unit_v2,
    build_c_official_evaluation_quartet_audit_v2,
    build_c_official_evaluation_quartet_v2,
)
from goalzendo_interactive_v2.evaluation_census import (
    PRODUCTION_MIRROR_ATTEMPTS_PER_STRATUM,
    REGISTERED_POPULATION_RESERVE_PER_JOINT_CELL,
    SUPPORTED_RANKING_ROW_COUNT,
    CensusConstructedOpeningV1,
    CensusPreOpeningFailureV1,
    EvaluationCensusV2Error,
    assess_evaluation_opening_v1,
    build_evaluation_census_plan_v1,
    build_evaluation_census_report_v1,
    evaluation_census_plan_v1_from_obj,
    evaluation_census_report_v1_from_obj,
    parse_evaluation_census_plan_v1,
    parse_evaluation_census_report_v1,
    serialize_evaluation_census_plan_v1,
    serialize_evaluation_census_report_v1,
    verify_evaluation_census_report_v1,
)
from goalzendo_interactive_v2.population_audit import build_rule_triple_binding_v2
from goalzendo_interactive_v2.role_schema import (
    EVIDENCE_GEOMETRIES,
    AlternativeAErrorsV2,
    OfficialTargetSideV2,
    build_evidence_schedule_v2,
)


def _sha(label: str) -> str:
    return hashlib.sha256(label.encode("ascii")).hexdigest()


def _keys(value: object) -> Iterator[str]:
    if type(value) is dict:
        for key, item in value.items():
            yield key
            yield from _keys(item)
    elif type(value) is list:
        for item in value:
            yield from _keys(item)


def _legacy_bank_quartet(
    composed_rule_id: str,
    *,
    sides: dict[str, OfficialTargetSideV2],
    tag: str,
    display_order: tuple[str, ...],
) -> Any:
    binding = build_rule_triple_binding_v2("g03r00016", "g03r00000", composed_rule_id)
    catalog = {entry.rule_id: entry for entry in build_rule_catalog()}
    p_bits = catalog[binding.placard_rule_id].truth.bits
    q_bits = catalog[binding.literal_rule_id].truth.bits
    c_bits = catalog[binding.composed_rule_id].truth.bits
    cells: list[list[int]] = [[] for _ in range(8)]
    for scene_index in range(SCENE_COUNT):
        cell = (
            4 * ((c_bits >> scene_index) & 1)
            + 2 * ((p_bits >> scene_index) & 1)
            + ((q_bits >> scene_index) & 1)
        )
        if len(cells[cell]) < 27:
            cells[cell].append(scene_index)
        if all(len(items) == 27 for items in cells):
            break
    assert all(len(items) == 27 for items in cells)

    union_counts = [0] * 8
    for geometry in EVIDENCE_GEOMETRIES:
        side = sides[geometry.slug] if geometry.alternative_a_errors is AlternativeAErrorsV2.NOISY else None
        schedule = build_evidence_schedule_v2(geometry, a_error_target_side=side)
        union_counts = [
            max(old, new)
            for old, new in zip(
                union_counts,
                schedule.joint_truth_cell_counts,
                strict=True,
            )
        ]
    opening_union = tuple(tuple(cells[cell_index][:count]) for cell_index, count in enumerate(union_counts))
    shared = EvaluationSharedBindingV2(
        reservoir_scene_ids_by_joint_cell=tuple(
            tuple(cells[cell_index][union_counts[cell_index] : union_counts[cell_index] + 22])
            for cell_index in range(8)
        ),
        intervention_bank_digest=_sha(f"legacy-intervention-{tag}"),
        renderer_binding_digest=_sha(f"legacy-renderer-{tag}"),
        display_geometry_order=display_order,
    )
    return build_c_official_evaluation_quartet_v2(
        binding,
        shared_resources=shared,
        opening_union_scene_ids_by_joint_cell=opening_union,
        noisy_a_target_sides=sides,
        pre_evaluation_checkpoint_digest=_sha("legacy-fixed-evaluation-checkpoint"),
        optimizer_step=91,
    )


@pytest.fixture(scope="module")
def engineering_census() -> dict[str, Any]:
    plan = build_evaluation_census_plan_v1(
        _sha("g03-v2-evaluation-census-focused-tests"),
        attempts_per_formula_stratum=1,
        candidate_pool_size=1,
    )
    plan_text = serialize_evaluation_census_plan_v1(plan)
    report = build_evaluation_census_report_v1(
        plan_text,
        expected_plan_digest=plan.digest,
    )
    report_text = serialize_evaluation_census_report_v1(report)
    return {
        "plan": plan,
        "plan_text": plan_text,
        "report": report,
        "report_text": report_text,
    }


def test_production_plan_is_exact_144_attempt_catalog_only_budget() -> None:
    plan = build_evaluation_census_plan_v1(_sha("g03-v2-production-census-plan"))
    assert plan.attempts_per_formula_stratum == PRODUCTION_MIRROR_ATTEMPTS_PER_STRATUM
    assert plan.uses_production_attempt_budget
    assert len(plan.attempts) == 9 * 16 == 144
    assert len({member.composed_rule_id for row in plan.attempts for member in row.members}) == 288
    assert Counter(row.formula_stratum for row in plan.attempts) == {
        "all__neg0": 16,
        "all__neg1": 16,
        "all__neg2": 16,
        "any__neg0": 16,
        "any__neg1": 16,
        "any__neg2": 16,
        "exactly_one__neg0": 16,
        "exactly_one__neg1": 16,
        "exactly_one__neg2": 16,
    }
    assert all(
        row.members[0].noisy_p_target_side + row.members[1].noisy_p_target_side == 1 for row in plan.attempts
    )
    assert all(row.members[0].composed_rule_id != row.members[1].composed_rule_id for row in plan.attempts)
    plan_obj = plan.as_obj()
    identities = plan_obj["identity_accounting"]
    assert identities["placard_identity_occurrence_count"] == 144
    assert identities["literal_identity_occurrence_count"] == 144
    assert identities["placard_literal_pair_occurrence_count"] == 144
    assert identities["composed_identity_occurrence_count"] == 288
    assert identities["distinct_composed_identity_count"] == 288
    assert identities["composed_reuse_occurrence_count"] == 0
    assert identities["one_shared_placard_literal_pair_per_mirror_attempt"]
    assert plan_obj["selection_boundary"] == {
        "positive_m_q_quota": None,
        "matched_bank_size": None,
        "matcher_specification": None,
    }
    assert REGISTERED_POPULATION_RESERVE_PER_JOINT_CELL == 27
    assert not (
        {"opening", "openings", "result", "results", "failure", "failures", "observed"} & set(_keys(plan_obj))
    )

    undersized_search = build_evaluation_census_plan_v1(
        _sha("g03-v2-production-census-plan"),
        attempts_per_formula_stratum=PRODUCTION_MIRROR_ATTEMPTS_PER_STRATUM,
        candidate_pool_size=1,
    )
    assert len(undersized_search.attempts) == 144
    assert not undersized_search.uses_production_attempt_budget
    assert not undersized_search.as_obj()["fixed_budget"]["uses_production_attempt_budget"]


def test_plan_canonical_round_trip_and_tamper_rejection(engineering_census: dict[str, Any]) -> None:
    plan = engineering_census["plan"]
    plan_text = engineering_census["plan_text"]
    assert parse_evaluation_census_plan_v1(plan_text, expected_digest=plan.digest) == plan
    assert (
        build_evaluation_census_plan_v1(
            plan.generator_seed,
            attempts_per_formula_stratum=1,
            candidate_pool_size=1,
        )
        == plan
    )

    tampered = json.loads(plan_text)
    tampered["attempts"][0]["members"][0]["noisy_p_target_side"] ^= 1
    with pytest.raises(EvaluationCensusV2Error):
        evaluation_census_plan_v1_from_obj(tampered, expected_digest=plan.digest)
    with pytest.raises(EvaluationCensusV2Error, match="canonical"):
        parse_evaluation_census_plan_v1(plan_text + "\n", expected_digest=plan.digest)


def test_plan_parser_rejects_duplicate_nonfinite_reordered_and_typed_tampering(
    engineering_census: dict[str, Any],
) -> None:
    plan = engineering_census["plan"]
    plan_text = engineering_census["plan_text"]

    duplicate = plan_text.replace('"plan_kind":', '"plan_kind":"duplicate","plan_kind":', 1)
    with pytest.raises(EvaluationCensusV2Error, match="duplicate JSON object key"):
        parse_evaluation_census_plan_v1(duplicate, expected_digest=plan.digest)

    nonfinite = plan_text.replace('"candidate_pool_size":1', '"candidate_pool_size":NaN', 1)
    with pytest.raises(EvaluationCensusV2Error, match="non-finite JSON constant"):
        parse_evaluation_census_plan_v1(nonfinite, expected_digest=plan.digest)

    reordered = json.loads(plan_text)
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
    with pytest.raises(EvaluationCensusV2Error):
        parse_evaluation_census_plan_v1(reordered_text, expected_digest=plan.digest)

    for binding_name, field_name in (
        ("source_binding", "source_manifest_digest"),
        ("catalog_binding", "supported_catalog_digest"),
    ):
        tampered = json.loads(plan_text)
        tampered[binding_name][field_name] = _sha(f"tampered-{binding_name}")
        with pytest.raises(EvaluationCensusV2Error):
            evaluation_census_plan_v1_from_obj(tampered, expected_digest=plan.digest)

    boolean = json.loads(plan_text)
    boolean["fixed_budget"]["candidate_pool_size"] = True
    with pytest.raises(EvaluationCensusV2Error):
        evaluation_census_plan_v1_from_obj(boolean, expected_digest=plan.digest)


def test_known_evaluation_opening_rederives_positive_m13_q4_witness() -> None:
    assessed = assess_evaluation_opening_v1(
        "g03r00016",
        "g03r00000",
        "g03r00178",
        "a0_b0",
        (1256, 12838, 354, 7027, 4122, 10100, 6199, 11144, 721, 12740),
    )
    record = assessed.record
    assert record.observed_m == 13
    assert record.observed_q == 4
    assert record.difficulty.minimax_depth == 4
    assert record.difficulty.best_first_query_scene_index == 2338
    assert record.difficulty.best_first_query_branch_sizes == (6, 7)
    assert record.difficulty.query_report_digest == (
        "a6ba256ce3ec0967426bc95fc06c43f0f2435eb3b8d5a7a3758a131eb6a5a744"
    )
    assert all(row.smaller_branch_size > 0 for row in record.difficulty.first_query_split_multiset)
    assert (
        sum(row.legal_scene_count for row in record.difficulty.first_query_split_multiset)
        == record.difficulty.root_informative_scene_count
    )
    assert record.salience.status == "eligible"
    assert record.disposition == "cell_witness"
    assert record.version_space_rule_ids == (
        "g03r00000",
        "g03r00016",
        "g03r00084",
        "g03r00085",
        "g03r00093",
        "g03r00100",
        "g03r00113",
        "g03r00120",
        "g03r00125",
        "g03r00127",
        "g03r00178",
        "g03r02624",
        "g03r02797",
    )
    rows = assessed.ranking_table.materialize_rows()
    assert assessed.ranking_table.row_count == SUPPORTED_RANKING_ROW_COUNT
    assert tuple(row.total_rank for row in rows) == tuple(range(1, SUPPORTED_RANKING_ROW_COUNT + 1))
    q_row = next(row for row in rows if row.rule_id == "g03r00000")
    assert q_row.correct_demonstrations == 10
    assert q_row.error_demonstrations == 0
    assert q_row.supported_family == "one_literal_piece"


def test_non_evaluation_c_identity_is_rejected_by_public_assessor() -> None:
    with pytest.raises(EvaluationCensusV2Error, match="evaluation-partition"):
        assess_evaluation_opening_v1(
            "g03r00016",
            "g03r00000",
            "g03r00179",
            "a0_b0",
            (1256, 12838, 354, 7027, 4122, 10100, 6199, 11144, 721, 12740),
        )


def test_old_passing_bank_quartet_is_preserved_as_four_rejected_openings() -> None:
    first = _legacy_bank_quartet(
        "g03r00178",
        sides={
            "a1_b0": OfficialTargetSideV2.NONFITTING,
            "a1_b2": OfficialTargetSideV2.FITTING,
        },
        tag="first",
        display_order=tuple(geometry.slug for geometry in EVIDENCE_GEOMETRIES),
    )
    second = _legacy_bank_quartet(
        "g03r00179",
        sides={
            "a1_b0": OfficialTargetSideV2.FITTING,
            "a1_b2": OfficialTargetSideV2.NONFITTING,
        },
        tag="second",
        display_order=tuple(geometry.slug for geometry in reversed(EVIDENCE_GEOMETRIES)),
    )
    old_audit = build_c_official_evaluation_quartet_audit_v2(
        (build_c_official_evaluation_mirror_unit_v2(first, second),)
    )
    assert old_audit.evaluation_quartet_audit_passed

    assessed = []
    for episode in first.episodes:
        variant = episode.geometry.slug
        if episode.a_error_target_side is not None:
            variant += f"_y{episode.a_error_target_side.value}"
        scenes = tuple(
            scene_index for cell in episode.opening_scene_ids_by_joint_cell for scene_index in cell
        )
        assessed.append(
            assess_evaluation_opening_v1(
                first.binding.placard_rule_id,
                first.binding.literal_rule_id,
                first.binding.composed_rule_id,
                variant,
                scenes,
            )
        )

    assert [item.record.observed_m for item in assessed] == [771, 778, 340, 351]
    assert all(item.record.disposition == "constructed_rejection" for item in assessed)
    assert all(item.ranking_table.row_count == SUPPORTED_RANKING_ROW_COUNT for item in assessed)
    assert all(
        "perfect_q_not_unique_live_one_literal" in item.record.salience.reason_codes for item in assessed[:2]
    )
    assert [len(item.record.salience.live_one_literal_rule_ids) for item in assessed[:2]] == [9, 9]
    assert all("noisy_q_not_uniquely_salient" in item.record.salience.reason_codes for item in assessed[2:])
    assert all(
        item.record.salience.maximum_other_one_literal_accuracy >= item.record.salience.literal_rule_accuracy
        for item in assessed[2:]
    )

    # The old audit also accepted C179, which is validation rather than evaluation.
    for episode in second.episodes:
        variant = episode.geometry.slug
        if episode.a_error_target_side is not None:
            variant += f"_y{episode.a_error_target_side.value}"
        scenes = tuple(
            scene_index for cell in episode.opening_scene_ids_by_joint_cell for scene_index in cell
        )
        with pytest.raises(EvaluationCensusV2Error, match="evaluation-partition"):
            assess_evaluation_opening_v1(
                second.binding.placard_rule_id,
                second.binding.literal_rule_id,
                second.binding.composed_rule_id,
                variant,
                scenes,
            )


def test_observed_ledger_is_complete_and_summary_has_all_36_cells(
    engineering_census: dict[str, Any],
) -> None:
    plan = engineering_census["plan"]
    report = engineering_census["report"]
    assert len(report.attempts) == len(plan.attempts) == 9
    assert sum(len(member.openings) for row in report.attempts for member in row.members) == 72
    assert report.constructed_opening_count + report.preopening_failure_count == 72
    assert tuple((row.m, row.q) for row in report.cell_summary) == tuple(
        (m, q) for m in range(8, 17) for q in range(1, 5)
    )
    assert all(row.as_obj()["positive_quota"] is None for row in report.cell_summary)
    table_by_digest = {table.digest: table for table in report.ranking_tables}
    plan_by_digest = {attempt.attempt_digest: attempt for attempt in plan.attempts}
    variants: set[str] = set()
    for attempt in report.attempts:
        assert attempt.attempt_plan == plan_by_digest[attempt.attempt_plan.attempt_digest]
        assert len(attempt.members) == 2
        for member in attempt.members:
            assert len(member.openings) == 4
            for opening in member.openings:
                variants.add(opening.schedule_variant)
                if type(opening) is CensusConstructedOpeningV1:
                    table = table_by_digest[opening.ranking_table_digest]
                    assert table.row_count == SUPPORTED_RANKING_ROW_COUNT
                else:
                    assert type(opening) is CensusPreOpeningFailureV1
                    assert opening.stage in {"joint_cell_supply", "minimum_space_guard"}
                    assert not opening.ranking_table_materialized
    assert variants == {
        "a0_b0",
        "a0_b2",
        "a1_b0_y0",
        "a1_b0_y1",
        "a1_b2_y0",
        "a1_b2_y1",
    }
    claim = report.as_obj()["claim_boundary"]
    assert report.status == "observed_complete_engineering_fixture_ledger_nonauthorizing"
    assert report.prospective_candidate_pool_size == 1
    assert not report.production_attempt_budget_complete
    assert claim["engineering_opening_evidence_described"]
    assert not claim["full_production_opening_feasibility_census_described"]
    assert not claim["evaluation_quartets_materialized"]
    assert not claim["nested_opening_unions_materialized"]
    assert not claim["matched_bank_size_selected"]
    assert not claim["challenge_reservoirs_materialized"]
    assert not claim["g01_authorized"]
    assert not claim["launch_authorized"]


def test_report_canonical_cross_parser_and_deterministic_replay(
    engineering_census: dict[str, Any],
) -> None:
    plan = engineering_census["plan"]
    plan_text = engineering_census["plan_text"]
    report = engineering_census["report"]
    report_text = engineering_census["report_text"]
    assert (
        build_evaluation_census_report_v1(
            plan_text,
            expected_plan_digest=plan.digest,
        )
        == report
    )
    assert (
        parse_evaluation_census_report_v1(
            report_text,
            plan_text=plan_text,
            expected_plan_digest=plan.digest,
            expected_report_digest=report.digest,
        )
        == report
    )
    assert (
        verify_evaluation_census_report_v1(
            report,
            plan_text,
            expected_plan_digest=plan.digest,
            expected_report_digest=report.digest,
        )
        == report
    )
    with pytest.raises(EvaluationCensusV2Error):
        evaluation_census_plan_v1_from_obj(report.as_obj(), expected_digest=plan.digest)
    with pytest.raises(EvaluationCensusV2Error):
        evaluation_census_report_v1_from_obj(
            plan.as_obj(),
            plan_text=plan_text,
            expected_plan_digest=plan.digest,
            expected_report_digest=report.digest,
        )


def test_report_rejects_rank_and_summary_tampering(engineering_census: dict[str, Any]) -> None:
    plan = engineering_census["plan"]
    plan_text = engineering_census["plan_text"]
    report = engineering_census["report"]
    tampered = report.as_obj()
    tampered["observed_m_q_summary"][0]["constructed_opening_count"] += 1
    with pytest.raises(EvaluationCensusV2Error):
        evaluation_census_report_v1_from_obj(
            tampered,
            plan_text=plan_text,
            expected_plan_digest=plan.digest,
            expected_report_digest=report.digest,
        )

    if report.ranking_tables:
        rank_tampered = report.as_obj()
        compact = rank_tampered["ranking_tables"][0]["compact_materialization"]
        original = compact["correct_counts_by_supported_index"][0]
        compact["correct_counts_by_supported_index"][0] = (original + 1) % 11
        with pytest.raises(EvaluationCensusV2Error):
            evaluation_census_report_v1_from_obj(
                rank_tampered,
                plan_text=plan_text,
                expected_plan_digest=plan.digest,
                expected_report_digest=report.digest,
            )
