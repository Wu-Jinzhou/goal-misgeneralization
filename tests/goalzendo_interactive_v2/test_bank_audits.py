from __future__ import annotations

import hashlib
import json
import time
from dataclasses import replace
from typing import Any

import pytest

from goalzendo_interactive import SCENE_COUNT, build_rule_catalog
from goalzendo_interactive_v2.bank_audits import (
    MINIMUM_POWERED_STRESS_TRIPLES,
    BoundBankAuditV2Error,
    CatalogBoundStressRoleUnitV2,
    CatalogBoundTrainingRoleUnitV2,
    COfficialEvaluationMirrorUnitV2,
    EvaluationSharedBindingV2,
    MetadataStressPopulationAuditV2,
    PerfectTrainingRoleAuditV2,
    TrainingRotationExecutionV2,
    build_c_official_evaluation_mirror_unit_v2,
    build_c_official_evaluation_quartet_audit_v2,
    build_c_official_evaluation_quartet_v2,
    build_catalog_bound_stress_role_unit_v2,
    build_catalog_bound_training_role_unit_v2,
    build_metadata_stress_population_audit_v2,
    build_perfect_training_role_audit_v2,
    build_rule_triple_bindings_batch_for_audit_v2,
    c_official_evaluation_quartet_audit_v2_from_obj,
    metadata_stress_population_audit_v2_from_obj,
    parse_c_official_evaluation_quartet_audit_v2,
    parse_metadata_stress_population_audit_v2,
    parse_perfect_training_role_audit_v2,
    perfect_training_role_audit_v2_from_obj,
    serialize_c_official_evaluation_quartet_audit_v2,
    serialize_metadata_stress_population_audit_v2,
    serialize_perfect_training_role_audit_v2,
    verify_c_official_evaluation_quartet_audit_v2,
    verify_metadata_stress_population_audit_v2,
    verify_perfect_training_role_audit_v2,
)
from goalzendo_interactive_v2.population_audit import (
    RuleTripleBindingV2,
    build_rule_triple_binding_v2,
    build_supported_catalog_contract_v2,
    classify_catalog_identity_v2,
)
from goalzendo_interactive_v2.role_schema import (
    EVIDENCE_GEOMETRIES,
    AlternativeAErrorsV2,
    MetaRoleBalanceAuditV2,
    OfficialTargetSideV2,
    build_evidence_schedule_v2,
    build_meta_role_balance_audit_v2,
    build_role_block_v2,
)


def _digest(label: str) -> str:
    return hashlib.sha256(label.encode("ascii")).hexdigest()


@pytest.fixture(scope="module")
def bound_triples() -> tuple[RuleTripleBindingV2, ...]:
    catalog = build_rule_catalog()
    placard = next(entry for entry in catalog if classify_catalog_identity_v2(entry) == "placard_literal")
    literal = next(entry for entry in catalog if classify_catalog_identity_v2(entry) == "one_literal_piece")
    composed = tuple(
        entry for entry in catalog if classify_catalog_identity_v2(entry) == "composed_two_literal_piece"
    )[:MINIMUM_POWERED_STRESS_TRIPLES]
    bindings = build_rule_triple_bindings_batch_for_audit_v2(
        (placard.rule_id, literal.rule_id, target.rule_id) for target in composed
    )
    assert bindings[0] == build_rule_triple_binding_v2(placard.rule_id, literal.rule_id, composed[0].rule_id)
    return bindings


@pytest.fixture(scope="module")
def training_audit(
    bound_triples: tuple[RuleTripleBindingV2, ...],
) -> tuple[PerfectTrainingRoleAuditV2, float]:
    started = time.perf_counter()
    units = tuple(
        build_catalog_bound_training_role_unit_v2(
            binding,
            bank_position=index,
            pre_update_checkpoint_digest=_digest(f"training-checkpoint-{index}"),
            update_batch_id=f"training-batch-{index:04d}",
            optimizer_step_before=10_000 + index,
            display_shift=index % 3,
            renderer_shift=(2 * index) % 3,
        )
        for index, binding in enumerate(bound_triples)
    )
    return build_perfect_training_role_audit_v2(units), time.perf_counter() - started


def _eligible_binding(composed_rule_id: str) -> RuleTripleBindingV2:
    return build_rule_triple_binding_v2("g03r00016", "g03r00000", composed_rule_id)


def _scene_pools(binding: RuleTripleBindingV2) -> tuple[tuple[int, ...], ...]:
    # This is deliberately independent of the private audit helper: the exact
    # full-universe cell vector is enough to recover each cell by truth bits.
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
    return tuple(tuple(items) for items in cells)


def _quartet(
    binding: RuleTripleBindingV2,
    *,
    side_by_geometry: dict[str, OfficialTargetSideV2],
    tag: str,
    display_order: tuple[str, ...],
) -> Any:
    pools = _scene_pools(binding)
    union_counts = [0] * 8
    for geometry in EVIDENCE_GEOMETRIES:
        side = (
            side_by_geometry[geometry.slug]
            if geometry.alternative_a_errors is AlternativeAErrorsV2.NOISY
            else None
        )
        schedule = build_evidence_schedule_v2(geometry, a_error_target_side=side)
        union_counts = [
            max(old, new)
            for old, new in zip(
                union_counts,
                schedule.joint_truth_cell_counts,
                strict=True,
            )
        ]
    opening_union = tuple(tuple(pools[cell_index][:count]) for cell_index, count in enumerate(union_counts))
    reservoir = tuple(
        tuple(pools[cell_index][union_counts[cell_index] : union_counts[cell_index] + 22])
        for cell_index in range(8)
    )
    shared = EvaluationSharedBindingV2(
        reservoir_scene_ids_by_joint_cell=reservoir,
        intervention_bank_digest=_digest(f"intervention-{tag}"),
        renderer_binding_digest=_digest(f"renderer-{tag}"),
        display_geometry_order=display_order,
    )
    return build_c_official_evaluation_quartet_v2(
        binding,
        shared_resources=shared,
        opening_union_scene_ids_by_joint_cell=opening_union,
        noisy_a_target_sides=side_by_geometry,
        pre_evaluation_checkpoint_digest=_digest("fixed-evaluation-checkpoint"),
        optimizer_step=91,
    )


@pytest.fixture(scope="module")
def evaluation_audit() -> Any:
    first = _quartet(
        _eligible_binding("g03r00178"),
        side_by_geometry={
            "a1_b0": OfficialTargetSideV2.NONFITTING,
            "a1_b2": OfficialTargetSideV2.FITTING,
        },
        tag="first",
        display_order=tuple(geometry.slug for geometry in EVIDENCE_GEOMETRIES),
    )
    second = _quartet(
        _eligible_binding("g03r00179"),
        side_by_geometry={
            "a1_b0": OfficialTargetSideV2.FITTING,
            "a1_b2": OfficialTargetSideV2.NONFITTING,
        },
        tag="second",
        display_order=tuple(geometry.slug for geometry in reversed(EVIDENCE_GEOMETRIES)),
    )
    return build_c_official_evaluation_quartet_audit_v2(
        (build_c_official_evaluation_mirror_unit_v2(first, second),)
    )


@pytest.fixture(scope="module")
def stress_audit(
    bound_triples: tuple[RuleTripleBindingV2, ...],
) -> MetadataStressPopulationAuditV2:
    units: list[CatalogBoundStressRoleUnitV2] = []
    for index, binding in enumerate(bound_triples):
        geometry = EVIDENCE_GEOMETRIES[index % 4]
        side = None
        if geometry.alternative_a_errors is AlternativeAErrorsV2.NOISY:
            side = tuple(OfficialTargetSideV2)[(index // 4) % 2]
        units.append(
            build_catalog_bound_stress_role_unit_v2(
                binding,
                geometry,
                a_error_target_side=side,
            )
        )
    return build_metadata_stress_population_audit_v2(units)


def test_384_triple_training_audit_passes_but_old_four_geometry_audit_fails(
    training_audit: tuple[PerfectTrainingRoleAuditV2, float],
) -> None:
    audit, elapsed = training_audit
    assert len(audit.units) == 384
    assert audit.training_role_audit_passed
    assert audit.digest == "d1d1bcabacc1c2678b5833f2f48753365dd006bdc8093686768e7288ea7aa76c"
    assert audit.as_obj()["episode_count"] == 1_152
    assert len(audit.as_obj()["complete_prefix_checks"]) == 384
    assert all(item["passed"] for item in audit.as_obj()["complete_prefix_checks"])
    assert elapsed < 30.0

    legacy = build_meta_role_balance_audit_v2(unit.role_block for unit in audit.units)
    assert type(legacy) is MetaRoleBalanceAuditV2
    assert not legacy.passed
    failed = {item["name"] for item in legacy.as_obj()["checks"] if item["passed"] is False}
    assert "geometry_group_balance_exact" in failed
    assert "geometry_by_role_exact" in failed


def test_training_contract_is_catalog_bound_atomic_and_prefix_counterbalanced(
    training_audit: tuple[PerfectTrainingRoleAuditV2, float],
) -> None:
    audit, _ = training_audit
    contract = build_supported_catalog_contract_v2()
    assert audit.catalog_digest == contract.source_catalog_digest
    assert audit.supported_catalog_digest == contract.supported_catalog_digest
    assert audit.as_obj()["family_by_official_a_b_counts"] == [
        {"family": family, "role": role, "count": 384}
        for family in ("placard", "literal", "composed")
        for role in ("official", "A", "B")
    ]
    for unit in audit.units:
        assert unit.binding.digest == unit.role_block.semantic_triple_id
        assert unit.role_block.geometry.slug == "a0_b0"
        assert {execution.display_order for execution in unit.executions} == {0, 1, 2}
        assert {execution.renderer_slot for execution in unit.executions} == {0, 1, 2}
        assert len({execution.pre_update_checkpoint_digest for execution in unit.executions}) == 1
        assert len({execution.update_batch_id for execution in unit.executions}) == 1
        assert len({execution.optimizer_step_before for execution in unit.executions}) == 1
        assert all(
            (execution.role_weight_numerator, execution.role_weight_denominator) == (1, 3)
            for execution in unit.executions
        )


def test_training_rejects_arbitrary_sha_interrotation_updates_and_bad_prefixes(
    bound_triples: tuple[RuleTripleBindingV2, ...],
) -> None:
    binding = bound_triples[0]
    arbitrary = build_role_block_v2(_digest("caller-invented-semantic-id"), EVIDENCE_GEOMETRIES[0])
    execution = tuple(
        TrainingRotationExecutionV2(
            position,
            position,
            position,
            _digest("checkpoint"),
            "batch-0",
            7,
        )
        for position in range(3)
    )
    with pytest.raises(BoundBankAuditV2Error, match="arbitrary or mismatched SHA"):
        CatalogBoundTrainingRoleUnitV2(0, binding, arbitrary, execution, 8)

    valid = build_catalog_bound_training_role_unit_v2(
        binding,
        bank_position=0,
        pre_update_checkpoint_digest=_digest("checkpoint"),
        update_batch_id="batch-0",
        optimizer_step_before=7,
        display_shift=0,
        renderer_shift=0,
    )
    changed_execution = replace(valid.executions[1], optimizer_step_before=8)
    with pytest.raises(BoundBankAuditV2Error, match="update occurred between"):
        CatalogBoundTrainingRoleUnitV2(
            0,
            binding,
            valid.role_block,
            (valid.executions[0], changed_execution, valid.executions[2]),
            8,
        )

    bad_units = tuple(
        build_catalog_bound_training_role_unit_v2(
            candidate,
            bank_position=index,
            pre_update_checkpoint_digest=_digest(f"bad-prefix-checkpoint-{index}"),
            update_batch_id=f"bad-prefix-batch-{index}",
            optimizer_step_before=index,
            display_shift=0,
            renderer_shift=0,
        )
        for index, candidate in enumerate(bound_triples[:3])
    )
    bad_audit = build_perfect_training_role_audit_v2(bad_units)
    assert not bad_audit.training_role_audit_passed
    assert bad_audit.as_obj()["complete_prefix_checks"][1]["passed"] is False


def test_small_training_report_round_trip_boolean_reorder_and_tamper_rejection(
    training_audit: tuple[PerfectTrainingRoleAuditV2, float],
) -> None:
    units = tuple(
        replace(unit, bank_position=index) for index, unit in enumerate(training_audit[0].units[:3])
    )
    audit = build_perfect_training_role_audit_v2(units)
    encoded = serialize_perfect_training_role_audit_v2(audit)
    assert parse_perfect_training_role_audit_v2(encoded, expected_digest=audit.digest) == audit
    assert verify_perfect_training_role_audit_v2(audit) == audit

    obj = json.loads(encoded)
    obj["schema_version"] = True
    with pytest.raises(BoundBankAuditV2Error, match="schema identity"):
        perfect_training_role_audit_v2_from_obj(obj)

    reordered = dict(reversed(tuple(json.loads(encoded).items())))
    with pytest.raises(BoundBankAuditV2Error, match="reordered"):
        perfect_training_role_audit_v2_from_obj(reordered)

    tampered = json.loads(encoded)
    tampered["units"][0]["binding"]["supported_catalog_digest"] = "0" * 64
    with pytest.raises(BoundBankAuditV2Error, match="catalog-bound rule triple"):
        perfect_training_role_audit_v2_from_obj(tampered)

    with pytest.raises(BoundBankAuditV2Error, match="canonical compact"):
        parse_perfect_training_role_audit_v2(encoded + "\n")
    duplicate_key = encoded.replace(
        '{"schema_version":1,',
        '{"schema_version":1,"schema_version":1,',
        1,
    )
    with pytest.raises(BoundBankAuditV2Error, match="duplicate JSON object key"):
        parse_perfect_training_role_audit_v2(duplicate_key)
    with pytest.raises(BoundBankAuditV2Error, match="externally expected"):
        parse_perfect_training_role_audit_v2(encoded, expected_digest="0" * 64)


def test_evaluation_quartet_has_exact_geometries_shared_resources_and_mirrors(
    evaluation_audit: Any,
) -> None:
    audit = evaluation_audit
    assert audit.evaluation_quartet_audit_passed
    assert audit.digest == "a4d49966540b5113531bd9c2ab032f0d6023a997e7b315edb65f467aba347c7d"
    assert audit.as_obj()["mirror_unit_count"] == 1
    assert audit.as_obj()["bound_triple_count"] == 2
    assert audit.as_obj()["episode_count"] == 8
    unit = audit.mirror_units[0]
    for slug in ("a1_b0", "a1_b2"):
        assert {quartet.noisy_side(slug) for quartet in unit.quartets} == set(OfficialTargetSideV2)
    for quartet in unit.quartets:
        assert tuple(episode.geometry for episode in quartet.episodes) == EVIDENCE_GEOMETRIES
        assert len({episode.shared_resource_binding_digest for episode in quartet.episodes}) == 1
        assert len({episode.opening_union_digest for episode in quartet.episodes}) == 1
        assert len({episode.optimizer_step for episode in quartet.episodes}) == 1
        assert all(not episode.weight_update_performed for episode in quartet.episodes)
        assert all(episode.context_reset_before_episode for episode in quartet.episodes)
        assert all(episode.official_ast_private_during_inquiry for episode in quartet.episodes)
        assert all(episode.terminal_classification_ast_blind for episode in quartet.episodes)
        for cell_index, union_cell in enumerate(quartet.opening_union_scene_ids_by_joint_cell):
            assert all(
                episode.opening_scene_ids_by_joint_cell[cell_index]
                == union_cell[: len(episode.opening_scene_ids_by_joint_cell[cell_index])]
                for episode in quartet.episodes
            )


def test_evaluation_round_trip_contract_tamper_and_same_side_rejection(
    evaluation_audit: Any,
) -> None:
    audit = evaluation_audit
    encoded = serialize_c_official_evaluation_quartet_audit_v2(audit)
    assert parse_c_official_evaluation_quartet_audit_v2(encoded, expected_digest=audit.digest) == audit
    assert verify_c_official_evaluation_quartet_audit_v2(audit) == audit

    obj = json.loads(encoded)
    episode = obj["mirror_units"][0]["quartets"][0]["episodes"][0]
    episode["execution_contract"]["terminal_classification_ast_blind"] = 1
    with pytest.raises(BoundBankAuditV2Error, match="must be a Boolean"):
        c_official_evaluation_quartet_audit_v2_from_obj(obj)

    reordered = dict(reversed(tuple(json.loads(encoded).items())))
    with pytest.raises(BoundBankAuditV2Error, match="reordered"):
        c_official_evaluation_quartet_audit_v2_from_obj(reordered)

    first, second = audit.mirror_units[0].quartets
    same_side = _quartet(
        second.binding,
        side_by_geometry={
            "a1_b0": first.noisy_side("a1_b0"),
            "a1_b2": first.noisy_side("a1_b2"),
        },
        tag="same-side",
        display_order=tuple(geometry.slug for geometry in EVIDENCE_GEOMETRIES),
    )
    with pytest.raises(BoundBankAuditV2Error, match="opposite Official target sides"):
        COfficialEvaluationMirrorUnitV2(
            tuple(sorted((first, same_side), key=lambda item: item.binding.digest))
        )

    foreign_block = build_role_block_v2(
        _digest("arbitrary-evaluation-semantic-id"), first.episodes[0].geometry
    )
    foreign_episode = replace(first.episodes[0], role_block=foreign_block)
    with pytest.raises(BoundBankAuditV2Error, match="not bound to the quartet triple"):
        replace(first, episodes=(foreign_episode, *first.episodes[1:]))


def test_metadata_stress_population_is_bound_balanced_and_nominally_distinct(
    stress_audit: MetadataStressPopulationAuditV2,
    training_audit: tuple[PerfectTrainingRoleAuditV2, float],
) -> None:
    assert stress_audit.stress_population_audit_passed
    assert stress_audit.digest == "db29228abae009f8ccb19a33ad9607687ce274a46c54a0b84be7b86d93da3d31"
    assert stress_audit.as_obj()["bound_triple_count"] == 384
    assert {item["count"] for item in stress_audit.as_obj()["geometry_counts"]} == {96}
    assert {item["count"] for item in stress_audit.as_obj()["noisy_a_target_mirror_counts"]} == {48}

    perfect_only_stress = build_metadata_stress_population_audit_v2(
        CatalogBoundStressRoleUnitV2(unit.binding, unit.role_block) for unit in training_audit[0].units
    )
    assert not perfect_only_stress.stress_population_audit_passed
    assert "training_role_audit_passed" not in perfect_only_stress.as_obj()
    assert "evaluation_quartet_audit_passed" not in perfect_only_stress.as_obj()


def test_stress_round_trip_boolean_and_reorder_rejection(
    stress_audit: MetadataStressPopulationAuditV2,
) -> None:
    # Three units are enough to exercise the parser; a failed audit remains a
    # valid diagnostic report and cannot be mistaken for a passing gate.
    small = build_metadata_stress_population_audit_v2(stress_audit.units[:3])
    assert not small.stress_population_audit_passed
    encoded = serialize_metadata_stress_population_audit_v2(small)
    assert parse_metadata_stress_population_audit_v2(encoded, expected_digest=small.digest) == small
    assert verify_metadata_stress_population_audit_v2(small) == small

    obj = json.loads(encoded)
    obj["schema_version"] = True
    with pytest.raises(BoundBankAuditV2Error, match="schema identity"):
        metadata_stress_population_audit_v2_from_obj(obj)

    reordered = dict(reversed(tuple(json.loads(encoded).items())))
    with pytest.raises(BoundBankAuditV2Error, match="reordered"):
        metadata_stress_population_audit_v2_from_obj(reordered)


def test_report_types_pass_flags_and_parsers_are_noninterchangeable(
    training_audit: tuple[PerfectTrainingRoleAuditV2, float],
    evaluation_audit: Any,
    stress_audit: MetadataStressPopulationAuditV2,
) -> None:
    training = build_perfect_training_role_audit_v2(training_audit[0].units[:3])
    stress = build_metadata_stress_population_audit_v2(stress_audit.units[:3])
    evaluation = evaluation_audit

    assert "training_role_audit_passed" in training.as_obj()
    assert "evaluation_quartet_audit_passed" not in training.as_obj()
    assert "stress_population_audit_passed" not in training.as_obj()
    assert "evaluation_quartet_audit_passed" in evaluation.as_obj()
    assert "training_role_audit_passed" not in evaluation.as_obj()
    assert "stress_population_audit_passed" in stress.as_obj()
    for report in (training, evaluation, stress):
        assert report.as_obj()["authorization"] == {
            "scope": "prospective-structural-bank-audit-only",
            "capability_run_authorized": False,
            "production_run_authorized": False,
            "weight_updates_authorized": False,
        }

    training_text = serialize_perfect_training_role_audit_v2(training)
    evaluation_text = serialize_c_official_evaluation_quartet_audit_v2(evaluation)
    stress_text = serialize_metadata_stress_population_audit_v2(stress)
    for parser, foreign in (
        (parse_perfect_training_role_audit_v2, evaluation_text),
        (parse_perfect_training_role_audit_v2, stress_text),
        (parse_c_official_evaluation_quartet_audit_v2, training_text),
        (parse_c_official_evaluation_quartet_audit_v2, stress_text),
        (parse_metadata_stress_population_audit_v2, training_text),
        (parse_metadata_stress_population_audit_v2, evaluation_text),
    ):
        with pytest.raises(BoundBankAuditV2Error):
            parser(foreign)

    with pytest.raises(TypeError):
        build_perfect_training_role_audit_v2(stress.units)  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        build_metadata_stress_population_audit_v2(training.units)  # type: ignore[arg-type]


def test_direct_boolean_aliases_and_malformed_scene_bindings_fail_closed(
    bound_triples: tuple[RuleTripleBindingV2, ...],
    evaluation_audit: Any,
) -> None:
    with pytest.raises(BoundBankAuditV2Error, match="Boolean"):
        TrainingRotationExecutionV2(
            0,
            0,
            0,
            _digest("checkpoint"),
            "batch",
            0,
            context_reset_before_episode=1,  # type: ignore[arg-type]
        )

    quartet = evaluation_audit.mirror_units[0].quartets[0]
    wrong_reservoir = list(quartet.shared_resources.reservoir_scene_ids_by_joint_cell)
    wrong_reservoir[0] = (
        *wrong_reservoir[0][:-1],
        quartet.opening_union_scene_ids_by_joint_cell[1][0],
    )
    wrong_shared = replace(
        quartet.shared_resources,
        reservoir_scene_ids_by_joint_cell=tuple(wrong_reservoir),
    )
    wrong_episodes = tuple(
        replace(episode, shared_resource_binding_digest=wrong_shared.digest) for episode in quartet.episodes
    )
    with pytest.raises(BoundBankAuditV2Error, match="not in declared joint cell"):
        replace(quartet, shared_resources=wrong_shared, episodes=wrong_episodes)

    with pytest.raises(TypeError, match="full RuleTripleBindingV2"):
        build_catalog_bound_training_role_unit_v2(  # type: ignore[arg-type]
            _digest("not-a-binding"),
            bank_position=0,
            pre_update_checkpoint_digest=_digest("checkpoint"),
            update_batch_id="batch",
            optimizer_step_before=0,
            display_shift=0,
            renderer_shift=0,
        )
    assert bound_triples[0].catalog_digest == build_rule_catalog().digest
