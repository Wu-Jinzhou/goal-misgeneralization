from __future__ import annotations

import hashlib
import json
from collections.abc import Iterator

import pytest

from goalzendo_interactive_v2.role_schema import (
    CYCLIC_FAMILY_ORDER,
    DEMONSTRATION_COUNT,
    EVIDENCE_GEOMETRIES,
    MIN_META_ROLE_INDEPENDENT_GROUPS,
    ROLE_ORDER,
    AlternativeAErrorsV2,
    AlternativeBErrorsV2,
    CandidateAssignmentV2,
    EpisodePurposeV2,
    EvidenceGeometryV2,
    EvidenceScheduleV2,
    MetaRoleBalanceAuditV2,
    OfficialTargetSideV2,
    RoleBlockV2,
    RoleRotationV2,
    RoleSchemaV2Error,
    RuleFamilyV2,
    build_evidence_schedule_v2,
    build_meta_role_balance_audit_v2,
    build_role_block_v2,
    meta_role_balance_audit_v2_from_obj,
    parse_meta_role_balance_audit_v2,
    parse_role_block_v2,
    role_block_v2_from_obj,
    serialize_meta_role_balance_audit_v2,
    serialize_role_block_v2,
)

CELL_ORDER = tuple(
    (official, alternative_a, alternative_b)
    for official in (0, 1)
    for alternative_a in (0, 1)
    for alternative_b in (0, 1)
)


def _identifier(index: int) -> str:
    return hashlib.sha256(f"independent-semantic-triple-{index}".encode()).hexdigest()


def _geometry(a_errors: int, b_errors: int) -> EvidenceGeometryV2:
    return EvidenceGeometryV2(
        AlternativeAErrorsV2(a_errors),
        AlternativeBErrorsV2(b_errors),
    )


def _balanced_blocks() -> tuple[RoleBlockV2, ...]:
    blocks: list[RoleBlockV2] = []
    for index in range(MIN_META_ROLE_INDEPENDENT_GROUPS):
        geometry = EVIDENCE_GEOMETRIES[index % len(EVIDENCE_GEOMETRIES)]
        side = None
        if geometry.alternative_a_errors is AlternativeAErrorsV2.NOISY:
            side = tuple(OfficialTargetSideV2)[(index // len(EVIDENCE_GEOMETRIES)) % 2]
        blocks.append(build_role_block_v2(_identifier(index), geometry, a_error_target_side=side))
    return tuple(blocks)


@pytest.mark.parametrize(
    ("a_errors", "b_errors", "side", "expected"),
    [
        (0, 0, None, (5, 0, 0, 0, 0, 0, 0, 5)),
        (0, 2, None, (4, 1, 0, 0, 0, 0, 1, 4)),
        (1, 0, OfficialTargetSideV2.NONFITTING, (4, 0, 1, 0, 0, 0, 0, 5)),
        (1, 0, OfficialTargetSideV2.FITTING, (5, 0, 0, 0, 0, 1, 0, 4)),
        (1, 2, OfficialTargetSideV2.NONFITTING, (3, 1, 1, 0, 0, 0, 1, 4)),
        (1, 2, OfficialTargetSideV2.FITTING, (4, 1, 0, 0, 0, 1, 1, 3)),
    ],
)
def test_all_six_exact_schedules(
    a_errors: int,
    b_errors: int,
    side: OfficialTargetSideV2 | None,
    expected: tuple[int, ...],
) -> None:
    schedule = build_evidence_schedule_v2(
        _geometry(a_errors, b_errors),
        a_error_target_side=side,
    )
    assert schedule.joint_truth_cell_counts == expected
    assert sum(expected) == DEMONSTRATION_COUNT
    assert schedule.official_class_counts == (5, 5)
    assert schedule.alternative_a_error_count == a_errors
    assert schedule.alternative_b_error_count == b_errors
    assert schedule.alternative_a_agreement_count == 10 - a_errors
    assert schedule.alternative_b_agreement_count == 10 - b_errors
    assert schedule.joint_error_overlap_count == 0
    assert schedule.alternative_b_errors_by_official_side == ((1, 1) if b_errors else (0, 0))


def _compositions(total: int, slots: int, prefix: tuple[int, ...] = ()) -> Iterator[tuple[int, ...]]:
    if slots == 1:
        yield (*prefix, total)
        return
    for value in range(total + 1):
        yield from _compositions(total - value, slots - 1, (*prefix, value))


def _satisfies_registered_constraints(
    counts: tuple[int, ...],
    geometry: EvidenceGeometryV2,
    side: OfficialTargetSideV2 | None,
) -> bool:
    def count_if(predicate: object) -> int:
        assert callable(predicate)
        return sum(count for cell, count in zip(CELL_ORDER, counts, strict=True) if predicate(cell))

    official_counts = tuple(count_if(lambda cell, target=target: cell[0] == target) for target in (0, 1))
    a_errors_by_side = tuple(
        count_if(lambda cell, target=target: cell[0] == target and cell[0] != cell[1]) for target in (0, 1)
    )
    b_errors_by_side = tuple(
        count_if(lambda cell, target=target: cell[0] == target and cell[0] != cell[2]) for target in (0, 1)
    )
    overlap = count_if(lambda cell: cell[0] != cell[1] and cell[0] != cell[2])
    if official_counts != (5, 5) or overlap != 0:
        return False
    if sum(a_errors_by_side) != geometry.alternative_a_errors.value:
        return False
    if sum(b_errors_by_side) != geometry.alternative_b_errors.value:
        return False
    if geometry.alternative_b_errors is AlternativeBErrorsV2.NOISY:
        if b_errors_by_side != (1, 1):
            return False
    elif b_errors_by_side != (0, 0):
        return False
    if side is None:
        return a_errors_by_side == (0, 0)
    expected_a_sides = (1, 0) if side is OfficialTargetSideV2.NONFITTING else (0, 1)
    return a_errors_by_side == expected_a_sides


def test_brute_force_constraints_uniquely_identify_every_registered_schedule() -> None:
    for geometry in EVIDENCE_GEOMETRIES:
        sides: tuple[OfficialTargetSideV2 | None, ...] = (
            (None,)
            if geometry.alternative_a_errors is AlternativeAErrorsV2.PERFECT
            else tuple(OfficialTargetSideV2)
        )
        for side in sides:
            feasible = tuple(
                counts
                for counts in _compositions(DEMONSTRATION_COUNT, len(CELL_ORDER))
                if _satisfies_registered_constraints(counts, geometry, side)
            )
            assert feasible == (
                build_evidence_schedule_v2(
                    geometry,
                    a_error_target_side=side,
                ).joint_truth_cell_counts,
            )


def test_schedule_rejects_wrong_mirror_and_any_noncanonical_vector() -> None:
    with pytest.raises(RoleSchemaV2Error, match="cannot name"):
        build_evidence_schedule_v2(
            _geometry(0, 0),
            a_error_target_side=OfficialTargetSideV2.NONFITTING,
        )
    with pytest.raises(RoleSchemaV2Error, match="requires"):
        build_evidence_schedule_v2(_geometry(1, 0))
    with pytest.raises(RoleSchemaV2Error, match="canonical schedule"):
        EvidenceScheduleV2(
            _geometry(1, 2),
            OfficialTargetSideV2.NONFITTING,
            (3, 1, 0, 1, 0, 0, 1, 4),
        )


def test_cyclic_rotations_cover_each_family_role_and_primary_maps_p_q_c() -> None:
    block = build_role_block_v2(
        _identifier(0), _geometry(1, 2), a_error_target_side=OfficialTargetSideV2.FITTING
    )
    assert tuple(rotation.official_family for rotation in block.rotations) == CYCLIC_FAMILY_ORDER
    assert tuple(rotation.purpose for rotation in block.rotations) == (
        EpisodePurposeV2.ROLE_COVER,
        EpisodePurposeV2.ROLE_COVER,
        EpisodePurposeV2.PRIMARY,
    )
    for family in CYCLIC_FAMILY_ORDER:
        for role in ROLE_ORDER:
            assert (
                sum(
                    assignment == CandidateAssignmentV2(family, role)
                    for rotation in block.rotations
                    for assignment in rotation.candidate_assignments
                )
                == 1
            )
    primary = block.rotations[2]
    assert primary.official_family is RuleFamilyV2.COMPOSED
    assert primary.alternative_a_family is RuleFamilyV2.PLACARD
    assert primary.alternative_b_family is RuleFamilyV2.LITERAL


def test_cover_status_is_evaluator_only_and_model_semantics_are_identical() -> None:
    block = build_role_block_v2(_identifier(1), _geometry(0, 2))
    semantics = tuple(rotation.model_semantics_obj() for rotation in block.rotations)
    assert semantics[0] == semantics[1] == semantics[2]
    encoded = json.dumps(semantics[0], separators=(",", ":"))
    assert "purpose" not in encoded
    assert "primary" not in encoded
    assert "role_cover" not in encoded
    assert {rotation.schedule.model_semantics_digest for rotation in block.rotations} == {
        block.schedule.model_semantics_digest
    }


def test_block_rejects_missing_noncyclic_or_semantically_different_rotation() -> None:
    block = build_role_block_v2(_identifier(2), _geometry(0, 0))
    with pytest.raises(RoleSchemaV2Error, match="exactly three"):
        RoleBlockV2(block.semantic_triple_id, block.rotations[:-1])

    first = block.rotations[0]
    with pytest.raises(RoleSchemaV2Error, match="cyclic"):
        RoleRotationV2(
            first.position,
            (
                first.candidate_assignments[1],
                first.candidate_assignments[0],
                first.candidate_assignments[2],
            ),
            first.purpose,
            first.schedule,
        )

    different_schedule = build_evidence_schedule_v2(_geometry(0, 2))
    second = block.rotations[1]
    changed = RoleRotationV2(
        second.position,
        second.candidate_assignments,
        second.purpose,
        different_schedule,
    )
    with pytest.raises(RoleSchemaV2Error, match="identical"):
        RoleBlockV2(block.semantic_triple_id, (block.rotations[0], changed, block.rotations[2]))


def test_role_block_round_trip_is_canonical_digest_bound_and_strict() -> None:
    block = build_role_block_v2(
        _identifier(3),
        _geometry(1, 2),
        a_error_target_side=OfficialTargetSideV2.NONFITTING,
    )
    text = serialize_role_block_v2(block)
    assert parse_role_block_v2(text, expected_digest=block.digest) == block
    assert role_block_v2_from_obj(block.as_obj()) == block
    assert serialize_role_block_v2(parse_role_block_v2(text)) == text

    with pytest.raises(RoleSchemaV2Error, match="canonical compact"):
        parse_role_block_v2(text + "\n")
    with pytest.raises(RoleSchemaV2Error, match="externally expected"):
        parse_role_block_v2(text, expected_digest="0" * 64)

    obj = json.loads(text)
    reordered = dict(reversed(tuple(obj.items())))
    with pytest.raises(RoleSchemaV2Error, match="reordered"):
        parse_role_block_v2(json.dumps(reordered, separators=(",", ":")))

    unknown = dict(obj)
    unknown["unknown"] = False
    with pytest.raises(RoleSchemaV2Error, match="noncanonical"):
        role_block_v2_from_obj(unknown)

    tampered = json.loads(text)
    tampered["rotations"][0]["schedule"]["joint_truth_cell_counts"][0] -= 1
    with pytest.raises(RoleSchemaV2Error):
        parse_role_block_v2(json.dumps(tampered, separators=(",", ":")))

    stale_digest = json.loads(text)
    stale_digest["block_digest"] = "0" * 64
    with pytest.raises(RoleSchemaV2Error, match="digest mismatch"):
        parse_role_block_v2(json.dumps(stale_digest, separators=(",", ":")))

    duplicate = text.replace(
        '"schema_version":1',
        '"schema_version":1,"schema_version":1',
        1,
    )
    with pytest.raises(RoleSchemaV2Error, match="duplicate"):
        parse_role_block_v2(duplicate)


def test_balanced_384_group_structural_audit_has_exact_tables() -> None:
    blocks = _balanced_blocks()
    audit = build_meta_role_balance_audit_v2(reversed(blocks))
    assert audit.passed
    assert audit.independent_semantic_triple_count == 384
    assert audit.submitted_block_count == 384
    obj = audit.as_obj()
    assert obj["classifier_fit_performed"] is False
    assert obj["powered_leakage_test_performed"] is False
    assert obj["episode_count"] == 1152
    assert obj["candidate_row_count"] == 3456
    assert {entry["count"] for entry in obj["family_by_role_counts"]} == {384}
    assert {entry["count"] for entry in obj["geometry_group_counts"]} == {96}
    assert {entry["count"] for entry in obj["geometry_by_role_counts"]} == {288}
    assert {entry["count"] for entry in obj["noisy_a_target_mirror_counts"]} == {48}
    assert all(check["passed"] for check in obj["checks"])
    assert "necessary structural gate only" in obj["limitation"]


def test_meta_audit_determinism_round_trip_and_strict_tamper_rejection() -> None:
    blocks = _balanced_blocks()
    audit = build_meta_role_balance_audit_v2(blocks)
    reversed_audit = build_meta_role_balance_audit_v2(reversed(blocks))
    assert reversed_audit.digest == audit.digest
    text = serialize_meta_role_balance_audit_v2(audit)
    assert parse_meta_role_balance_audit_v2(text, expected_digest=audit.digest) == audit
    assert meta_role_balance_audit_v2_from_obj(audit.as_obj()) == audit

    obj = json.loads(text)
    obj["family_by_role_counts"][0]["count"] -= 1
    with pytest.raises(RoleSchemaV2Error):
        parse_meta_role_balance_audit_v2(json.dumps(obj, separators=(",", ":")))

    stale = json.loads(text)
    stale["audit_digest"] = "f" * 64
    with pytest.raises(RoleSchemaV2Error, match="digest mismatch"):
        parse_meta_role_balance_audit_v2(json.dumps(stale, separators=(",", ":")))

    reordered = json.loads(text)
    reordered = dict(reversed(tuple(reordered.items())))
    with pytest.raises(RoleSchemaV2Error, match="reordered"):
        parse_meta_role_balance_audit_v2(json.dumps(reordered, separators=(",", ":")))

    duplicate = text.replace(
        '"schema_version":1',
        '"schema_version":1,"schema_version":1',
        1,
    )
    with pytest.raises(RoleSchemaV2Error, match="duplicate"):
        parse_meta_role_balance_audit_v2(duplicate)


def test_meta_audit_fails_small_sample_geometry_mirror_and_duplicate_groups() -> None:
    blocks = _balanced_blocks()

    small = build_meta_role_balance_audit_v2(blocks[:8])
    assert not small.passed
    small_checks = {entry["name"]: entry["passed"] for entry in small.as_obj()["checks"]}
    assert not small_checks["minimum_independent_group_count"]

    imbalanced_geometry = list(blocks)
    old = imbalanced_geometry[0]
    imbalanced_geometry[0] = build_role_block_v2(
        old.semantic_triple_id,
        EVIDENCE_GEOMETRIES[1],
        a_error_target_side=OfficialTargetSideV2.NONFITTING,
    )
    geometry_audit = build_meta_role_balance_audit_v2(imbalanced_geometry)
    geometry_checks = {entry["name"]: entry["passed"] for entry in geometry_audit.as_obj()["checks"]}
    assert not geometry_audit.passed
    assert not geometry_checks["geometry_group_balance_exact"]
    assert not geometry_checks["geometry_by_role_exact"]

    unmirrored = tuple(
        build_role_block_v2(
            block.semantic_triple_id,
            block.geometry,
            a_error_target_side=(
                None
                if block.geometry.alternative_a_errors is AlternativeAErrorsV2.PERFECT
                else OfficialTargetSideV2.NONFITTING
            ),
        )
        for block in blocks
    )
    mirror_audit = build_meta_role_balance_audit_v2(unmirrored)
    mirror_checks = {entry["name"]: entry["passed"] for entry in mirror_audit.as_obj()["checks"]}
    assert not mirror_audit.passed
    assert not mirror_checks["noisy_a_target_mirror_exact"]

    duplicate = list(blocks)
    replacement = next(
        block
        for block in blocks
        if block.geometry == duplicate[-1].geometry
        and block.a_error_target_side is duplicate[-1].a_error_target_side
    )
    duplicate[-1] = replacement
    duplicate_audit = build_meta_role_balance_audit_v2(duplicate)
    duplicate_checks = {entry["name"]: entry["passed"] for entry in duplicate_audit.as_obj()["checks"]}
    assert not duplicate_audit.passed
    assert not duplicate_checks["unique_semantic_triple_ids"]


def test_audit_constructor_rejects_noncanonical_order_and_report_unknown_fields() -> None:
    blocks = _balanced_blocks()
    with pytest.raises(RoleSchemaV2Error, match="canonical digest order"):
        MetaRoleBalanceAuditV2(blocks)

    audit = build_meta_role_balance_audit_v2(blocks)
    unknown = audit.as_obj()
    unknown["classifier_accuracy"] = 1 / 3
    with pytest.raises(RoleSchemaV2Error, match="noncanonical"):
        meta_role_balance_audit_v2_from_obj(unknown)


def test_geometry_axes_and_enum_types_are_closed() -> None:
    assert {geometry.alternative_a_errors.value for geometry in EVIDENCE_GEOMETRIES} == {0, 1}
    assert {geometry.alternative_b_errors.value for geometry in EVIDENCE_GEOMETRIES} == {0, 2}
    with pytest.raises(RoleSchemaV2Error, match="wrong enum type"):
        EvidenceGeometryV2(0, AlternativeBErrorsV2.PERFECT)  # type: ignore[arg-type]
    with pytest.raises(RoleSchemaV2Error, match="unknown"):
        role_block_v2_from_obj(
            {
                **build_role_block_v2(_identifier(999), _geometry(0, 0)).as_obj(),
                "geometry": {
                    "slug": "a0_b7",
                    "alternative_a_errors": 0,
                    "alternative_b_errors": 7,
                },
            }
        )
