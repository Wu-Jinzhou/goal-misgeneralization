from __future__ import annotations

import json
import time
from collections import Counter
from dataclasses import replace
from typing import cast

import pytest

from goalzendo_interactive import SCENE_COUNT, build_rule_catalog, evaluate_rule, iter_scenes
from goalzendo_interactive.catalog import CatalogEntry
from goalzendo_interactive.rules import Rule
from goalzendo_interactive.stage_partitions_v2 import RULE_PARTITIONS_V2
from goalzendo_interactive_v2.population_audit import (
    CATALOG_AUDIT_FAMILIES,
    COMPOSED_STRATA,
    ENGINEERING_LEAKAGE_REQUIRED_TARGETS,
    JOINT_CELL_ORDER,
    RESERVE_THRESHOLDS,
    PopulationAuditV2Error,
    build_population_audit_v2,
    build_rule_triple_binding_v2,
    build_supported_catalog_contract_v2,
    classify_catalog_identity_v2,
    composed_stratum_v2,
    joint_truth_cell_counts_v2,
    parse_population_audit_v2,
    parse_rule_triple_binding_v2,
    require_supported_catalog_identity_v2,
    rule_triple_binding_v2_from_obj,
    serialize_population_audit_v2,
    serialize_rule_triple_binding_v2,
    triple_cell_digest_v2,
    verify_population_audit_v2,
    verify_rule_triple_binding_v2,
)


def _family_pools() -> dict[str, list[CatalogEntry]]:
    pools: dict[str, list[CatalogEntry]] = {family: [] for family in CATALOG_AUDIT_FAMILIES}
    for entry in build_rule_catalog():
        pools[classify_catalog_identity_v2(entry)].append(entry)
    return pools


def test_classifier_is_total_exact_and_fails_closed_on_unknown_ast() -> None:
    pools = _family_pools()
    assert {family: len(entries) for family, entries in pools.items()} == {
        "placard_literal": 2,
        "one_literal_piece": 82,
        "composed_two_literal_piece": 6_886,
        "excluded_placard_composition": 330,
    }
    assert sum(map(len, pools.values())) == len(build_rule_catalog()) == 7_300
    assert Counter(composed_stratum_v2(entry) for entry in pools["composed_two_literal_piece"]) == {
        ("all", 0): 188,
        ("all", 1): 926,
        ("all", 2): 1_033,
        ("any", 0): 997,
        ("any", 1): 911,
        ("any", 2): 203,
        ("exactly_one", 0): 660,
        ("exactly_one", 1): 1_314,
        ("exactly_one", 2): 654,
    }

    catalog = build_rule_catalog()
    unknown = CatalogEntry(0, cast(Rule, object()), catalog[0].truth)
    with pytest.raises(PopulationAuditV2Error, match="unknown public rule AST"):
        classify_catalog_identity_v2(unknown)
    with pytest.raises(PopulationAuditV2Error, match="requires a composed"):
        composed_stratum_v2(pools["one_literal_piece"][0])
    with pytest.raises(PopulationAuditV2Error, match="unsupported in v2"):
        require_supported_catalog_identity_v2(pools["excluded_placard_composition"][0])
    for family in (
        "placard_literal",
        "one_literal_piece",
        "composed_two_literal_piece",
    ):
        assert require_supported_catalog_identity_v2(pools[family][0]) == family

    contract = build_supported_catalog_contract_v2()
    assert contract.source_catalog_digest == build_rule_catalog().digest
    assert len(contract.supported_indices) == 6_970
    assert len(contract.excluded_indices) == 330
    assert contract.supported_catalog_digest == (
        "e2c544786ccdcc85f5e424cb5a8927b61b5c6d13ab0cd70f5250d7f4a4eedd93"
    )
    assert contract.excluded_catalog_digest == (
        "6e808d97e31b4017ddfade72663316ac9ce410aa528182035efda211a543840d"
    )


def test_joint_cells_match_independent_direct_scene_evaluation() -> None:
    pools = _family_pools()
    placard = pools["placard_literal"][1]
    literal = pools["one_literal_piece"][2]
    composed = pools["composed_two_literal_piece"][3]
    observed = [0] * len(JOINT_CELL_ORDER)
    for scene in iter_scenes():
        c_value = int(evaluate_rule(composed.rule, scene))
        p_value = int(evaluate_rule(placard.rule, scene))
        q_value = int(evaluate_rule(literal.rule, scene))
        observed[4 * c_value + 2 * p_value + q_value] += 1

    assert JOINT_CELL_ORDER == ("000", "001", "010", "011", "100", "101", "110", "111")
    assert tuple(observed) == (1_528, 3_143, 1_528, 3_143, 668, 1_519, 668, 1_519)
    assert joint_truth_cell_counts_v2(placard, literal, composed) == tuple(observed)
    assert sum(observed) == SCENE_COUNT
    assert triple_cell_digest_v2(placard, literal, composed) == (
        "1e743957e9320062e4d523c55cf008cf84a00a0394f6c8f12563f2faeb87be61"
    )

    with pytest.raises(PopulationAuditV2Error, match="triple families"):
        joint_truth_cell_counts_v2(literal, placard, composed)


def test_small_cross_product_matches_an_independent_brute_force() -> None:
    pools = _family_pools()
    placards = pools["placard_literal"][:2]
    literals = pools["one_literal_piece"][:3]
    composed = pools["composed_two_literal_piece"][:4]
    scenes = tuple(iter_scenes())
    labels = {
        entry.index: tuple(evaluate_rule(entry.rule, scene) for scene in scenes)
        for entry in (*placards, *literals, *composed)
    }
    minima: Counter[int] = Counter()
    for placard in placards:
        for literal in literals:
            for target in composed:
                direct = [0] * len(JOINT_CELL_ORDER)
                for p_value, q_value, c_value in zip(
                    labels[placard.index],
                    labels[literal.index],
                    labels[target.index],
                    strict=True,
                ):
                    direct[4 * c_value + 2 * p_value + q_value] += 1
                assert joint_truth_cell_counts_v2(placard, literal, target) == tuple(direct)
                minima[min(direct)] += 1
    assert minima == {0: 18, 668: 2, 837: 4}
    assert {
        threshold: sum(count for minimum, count in minima.items() if minimum >= threshold)
        for threshold in RESERVE_THRESHOLDS
    } == {22: 6, 27: 6, 32: 6}


def test_canonical_rule_triple_binding_replaces_caller_invented_ids() -> None:
    binding = build_rule_triple_binding_v2(
        "g03r00050",
        "g03r00002",
        "g03r00087",
    )
    assert binding.digest == ("7dfdcf3566cf73f897492b6fedd99783e19f1bfd6267d75ccd9849e6fb1e0fa5")
    assert binding.supported_catalog_digest == (
        "e2c544786ccdcc85f5e424cb5a8927b61b5c6d13ab0cd70f5250d7f4a4eedd93"
    )
    assert binding.exact_joint_truth_cell_counts == (
        1_528,
        3_143,
        1_528,
        3_143,
        668,
        1_519,
        668,
        1_519,
    )
    encoded = serialize_rule_triple_binding_v2(binding)
    assert (
        parse_rule_triple_binding_v2(
            encoded,
            expected_digest=binding.digest,
        )
        == binding
    )
    assert verify_rule_triple_binding_v2(binding) == binding
    assert rule_triple_binding_v2_from_obj(binding.as_obj()) == binding

    for field, replacement, message in (
        ("catalog_digest", "0" * 64, "catalog digest"),
        ("supported_catalog_digest", "0" * 64, "supported-catalog digest"),
        ("exact_joint_cell_digest", "0" * 64, "cell digest"),
        ("triple_digest", "0" * 64, "semantic-triple digest"),
    ):
        tampered = binding.as_obj()
        tampered[field] = replacement
        with pytest.raises(PopulationAuditV2Error, match=message):
            rule_triple_binding_v2_from_obj(tampered)

    identity_tamper = binding.as_obj()
    identity_tamper["candidates"][1]["truth_digest"] = "0" * 64
    with pytest.raises(PopulationAuditV2Error, match="public catalog"):
        rule_triple_binding_v2_from_obj(identity_tamper)

    cell_tamper = binding.as_obj()
    cell_tamper["exact_joint_truth_cell_counts"][0] += 1
    with pytest.raises(PopulationAuditV2Error, match="full-universe"):
        rule_triple_binding_v2_from_obj(cell_tamper)

    reordered = binding.as_obj()
    reordered["candidates"] = [
        reordered["candidates"][1],
        reordered["candidates"][0],
        reordered["candidates"][2],
    ]
    with pytest.raises(PopulationAuditV2Error, match="slot/family"):
        rule_triple_binding_v2_from_obj(reordered)

    with pytest.raises(PopulationAuditV2Error, match="malformed public rule id"):
        build_rule_triple_binding_v2("caller-invented", "g03r00002", "g03r00087")

    boolean_schema_alias = binding.as_obj()
    boolean_schema_alias["schema_version"] = True
    with pytest.raises(PopulationAuditV2Error, match="schema version"):
        parse_rule_triple_binding_v2(json.dumps(boolean_schema_alias, separators=(",", ":")))


def test_exhaustive_builder_has_deterministic_work_accounting_and_runtime() -> None:
    build_population_audit_v2.cache_clear()
    started = time.perf_counter()
    report = build_population_audit_v2()
    elapsed = time.perf_counter() - started

    performance = report.as_obj()["performance_accounting"]
    assert performance == {
        "triple_rows_materialized": 0,
        "triple_rows_streamed": 1_129_304,
        "exact_joint_cell_counts_computed": 9_034_432,
        "composed_partner_counter_slots": 20_658,
        "wall_clock_in_digest": False,
    }
    assert elapsed < 60.0
    cached_started = time.perf_counter()
    assert build_population_audit_v2() is report
    assert time.perf_counter() - cached_started < 0.1


def test_exhaustive_population_counts_and_digests_are_pinned() -> None:
    report = build_population_audit_v2()
    assert report.digest == "a8fca157ee01186b7294996ffe67fc15b467b58da6e3e1de557f9f8a2512dbd6"
    assert report.source_fingerprint == ("24b6d1cc60c09be3b6bbab22d7b7a5b09fef1c4250dd8eb6dc0a9d323aed0ada")
    assert report.catalog_digest == ("a796ef24d4e0eb2a2e129e12ee9cc3261c82b578f415e554d6608feafc9ae2d0")
    assert report.stage_partition_digest == (
        "54943bd1d82c6179d3fda63448e6169e0e3163da5374dc3d169137b85af7eb90"
    )
    assert report.dependency_manifest_digest == (
        "719517671c3e5808139163cf3d936e932b46e5d8a1ac50f6a1196624fac63e9a"
    )
    assert report.family_identity_table_digest == (
        "539f790e6640ab92d97dc80f2e012f571de36062defe78e2aa1044d817835eed"
    )
    assert (
        report.supported_catalog_identity_count,
        report.supported_catalog_digest,
        report.excluded_catalog_identity_count,
        report.excluded_catalog_digest,
    ) == (
        6_970,
        "e2c544786ccdcc85f5e424cb5a8927b61b5c6d13ab0cd70f5250d7f4a4eedd93",
        330,
        "6e808d97e31b4017ddfade72663316ac9ce410aa528182035efda211a543840d",
    )
    assert report.cartesian_triple_count == report.extensionally_distinct_triple_count == 1_129_304
    assert report.triple_identity_table_digest == (
        "ab92694d78ab1a5127a6f894a34073ea5d195bd1914e407937fba22d72d9495c"
    )
    assert report.exact_joint_cell_table_digest == (
        "0fde21e18b21e27add229ca946f7955605a7ec5da505dedeb5363e928548be4b"
    )
    assert (report.unique_joint_cell_vector_count, report.joint_cell_vector_histogram_digest) == (
        9_784,
        "632f5b8328ba8e0c33fe242ddd4d96cff981fe45b398a1d8c34128a8395903d9",
    )
    assert (
        report.distinct_minimum_cell_count_count,
        report.minimum_cell_count,
        report.maximum_minimum_cell_count,
        report.minimum_cell_count_histogram_digest,
    ) == (
        369,
        0,
        1_620,
        "19c32a58f86850513c72ee69821d0ed097d7144587623c134d156ab14ede4969",
    )
    assert report.below_conservative_threshold_histogram == (
        (0, 36_752),
        (6, 1_008),
        (9, 144),
        (18, 48),
        (27, 16),
    )


def test_all_three_unfrozen_reserve_thresholds_have_exact_capacity() -> None:
    report = build_population_audit_v2()
    assert tuple(row.reserve_per_joint_cell for row in report.reserve_capacities) == (RESERVE_THRESHOLDS)
    expected = {
        22: (
            1_091_352,
            "3855b68b4a59ced8f61ad4fbbffbe7e7c2cbc649341b0d64230a1ad61115b932",
            "543c8ab699ab5b3f76fbf4702f9c87aea2135223be92e15ed7838eecbbb8a86c",
            "12ffdf1b4a7c38fa1d71c4fbe4bdebafe1b4b2725b8b356e83eb5db14079aa16",
        ),
        27: (
            1_091_352,
            "5ee2e64f8e8eeff7ea84cb010381c0c55d99d428ecb9f5127564a9ceb400fa97",
            "63b3095417aa5201c0e158580a45c7dc175863d682c0be6874b6079e2ee62831",
            "4c137af93673aaf7b5c1042e0525b4355c48a0b7c8ece3aaac0dad413d8d24c4",
        ),
        32: (
            1_091_336,
            "3fe2ca5e3f99822292e87a7d260886d6daf1047db67090ba15bc07278cc6a11f",
            "1bff64022880bd0d12130184837f5c3831bb9abfa859bb5515260b1b8f9fa881",
            "4e727ff2a97a4aab963f6998df6fad37ecfda2fd9fd669494a52cdc084513eef",
        ),
    }
    for row in report.reserve_capacities:
        assert not row.threshold_frozen
        assert row.eligible_composed_target_count == 6_886
        assert (
            row.eligible_triple_count,
            row.eligible_triple_table_digest,
            row.eligible_composed_target_digest,
            row.composed_partner_count_table_digest,
        ) == expected[row.reserve_per_joint_cell]
        assert sum(item.composed_target_count for item in row.partner_count_histogram) == 6_886
        assert (
            sum(
                item.eligible_partner_count * item.composed_target_count
                for item in row.partner_count_histogram
            )
            == row.eligible_triple_count
        )

    assert [
        (item.eligible_partner_count, item.composed_target_count)
        for item in report.reserve_capacities[2].partner_count_histogram
    ] == [
        (144, 24),
        (148, 468),
        (152, 756),
        (156, 1_976),
        (160, 1_280),
        (164, 2_382),
    ]


def test_stage_and_operator_negation_capacity_is_complete_and_consistent() -> None:
    report = build_population_audit_v2()
    for reserve in report.reserve_capacities:
        assert tuple(row.stage_partition for row in reserve.by_stage_partition) == (RULE_PARTITIONS_V2)
        assert len(reserve.by_stage_and_composed_stratum) == (len(RULE_PARTITIONS_V2) * len(COMPOSED_STRATA))
        assert tuple(
            (row.stage_partition, row.composed_operator, row.negated_literal_count)
            for row in reserve.by_stage_and_composed_stratum
        ) == tuple(
            (stage, op, negated_count)
            for stage in RULE_PARTITIONS_V2
            for op, negated_count in COMPOSED_STRATA
        )
        assert sum(row.composed_target_count for row in reserve.by_stage_partition) == 6_886
        assert (
            sum(row.eligible_composed_target_count for row in reserve.by_stage_partition)
            == reserve.eligible_composed_target_count
        )
        assert sum(row.eligible_triple_count for row in reserve.by_stage_partition) == (
            reserve.eligible_triple_count
        )
        assert sum(row.composed_target_count for row in reserve.by_stage_and_composed_stratum) == 6_886
        assert (
            sum(row.eligible_triple_count for row in reserve.by_stage_and_composed_stratum)
            == reserve.eligible_triple_count
        )

    at_32 = report.reserve_capacities[2]
    assert [
        (
            row.stage_partition,
            row.composed_target_count,
            row.eligible_composed_target_count,
            row.eligible_triple_count,
        )
        for row in at_32.by_stage_partition
    ] == [
        ("warm_start", 690, 690, 109_456),
        ("engineering", 1_378, 1_378, 218_388),
        ("capability", 689, 689, 109_132),
        ("pilot", 689, 689, 109_228),
        ("confirmatory_train", 1_378, 1_378, 218_284),
        ("validation", 688, 688, 109_100),
        ("evaluation", 1_374, 1_374, 217_748),
    ]


def test_engineering_leakage_capacity_and_claim_boundary_are_explicit() -> None:
    report = build_population_audit_v2()
    assert report.engineering_leakage_observed_targets_at_32 == 1_378
    assert report.engineering_leakage_observed_targets_at_32 >= (ENGINEERING_LEAKAGE_REQUIRED_TARGETS)
    assert report.engineering_leakage_capacity_passed_at_32

    value = report.as_obj()
    assert value["v2_supported_catalog"] == {
        "source_catalog_digest": report.catalog_digest,
        "official_and_version_space_families": [
            "placard_literal",
            "one_literal_piece",
            "composed_two_literal_piece",
        ],
        "supported_identity_count": 6_970,
        "supported_catalog_digest": ("e2c544786ccdcc85f5e424cb5a8927b61b5c6d13ab0cd70f5250d7f4a4eedd93"),
        "excluded_family": "excluded_placard_composition",
        "excluded_identity_count": 330,
        "excluded_catalog_digest": ("6e808d97e31b4017ddfade72663316ac9ce410aa528182035efda211a543840d"),
        "unsupported_identity_eligible_count": 0,
        "mixed_placard_compositions_allowed_as_official": False,
        "mixed_placard_compositions_allowed_in_v0": False,
        "unknown_or_unsupported_eligibility_policy": "fail_closed",
    }
    assert value["identity_reuse_contract"] == {
        "placard_identity_reuse": "permitted_across_scene_disjoint_role_cover_blocks",
        "one_literal_piece_identity_reuse": ("permitted_across_scene_disjoint_role_cover_blocks"),
        "composed_primary_identity_reuse": "forbidden",
        "composed_primary_stage_overlap": "forbidden_by_total_stage_partition",
        "partner_same_partition_required": False,
        "stage_partition_applies_to": "composed_primary_identity",
    }
    assert value["claim_boundary"] == {
        "exhaustive_rule_population_and_cell_counts": True,
        "matched_openings_or_difficulty_balance": False,
        "eleven_disjoint_challenge_panels": False,
        "version_space_coverage": False,
        "episode_bank_feasibility": False,
        "scientific_authorization": False,
    }
    assert value["authorization"]["weight_updates_authorized"] is False


def test_report_round_trip_recomputes_and_rejects_tampering() -> None:
    report = build_population_audit_v2()
    encoded = serialize_population_audit_v2(report)
    assert parse_population_audit_v2(encoded, expected_digest=report.digest) is report
    assert verify_population_audit_v2(report) is report
    assert serialize_population_audit_v2(parse_population_audit_v2(encoded)) == encoded

    tampered = json.loads(encoded)
    tampered["reserve_thresholds"][2]["eligible_triple_count"] += 1
    with pytest.raises(PopulationAuditV2Error, match="recomputation"):
        parse_population_audit_v2(json.dumps(tampered, separators=(",", ":")))

    dependency_tamper = json.loads(encoded)
    dependency_tamper["dependencies"]["catalog_digest"] = "0" * 64
    with pytest.raises(PopulationAuditV2Error, match="recomputation"):
        parse_population_audit_v2(json.dumps(dependency_tamper, separators=(",", ":")))

    unsupported_tamper = json.loads(encoded)
    unsupported_tamper["v2_supported_catalog"]["unsupported_identity_eligible_count"] = 1
    with pytest.raises(PopulationAuditV2Error, match="recomputation"):
        parse_population_audit_v2(json.dumps(unsupported_tamper, separators=(",", ":")))

    boolean_schema_alias = json.loads(encoded)
    boolean_schema_alias["schema_version"] = True
    with pytest.raises(PopulationAuditV2Error, match="recomputation"):
        parse_population_audit_v2(json.dumps(boolean_schema_alias, separators=(",", ":")))

    with pytest.raises(PopulationAuditV2Error, match="canonical compact"):
        parse_population_audit_v2(encoded + "\n")
    duplicate = encoded.replace(
        '{"schema_version":1,',
        '{"schema_version":1,"schema_version":1,',
        1,
    )
    with pytest.raises(PopulationAuditV2Error, match="duplicate"):
        parse_population_audit_v2(duplicate)
    with pytest.raises(PopulationAuditV2Error, match="malformed"):
        parse_population_audit_v2(encoded, expected_digest="not-a-digest")
    with pytest.raises(PopulationAuditV2Error, match="does not match"):
        parse_population_audit_v2(encoded, expected_digest="0" * 64)

    changed = replace(report, cartesian_triple_count=report.cartesian_triple_count + 1)
    with pytest.raises(PopulationAuditV2Error, match="recomputation"):
        verify_population_audit_v2(changed)
    with pytest.raises(PopulationAuditV2Error, match="recomputation"):
        serialize_population_audit_v2(changed)
