"""Role-counterbalanced evidence schedules for the prospective G03-v2 design.

This module is deliberately additive and nonauthorizing.  It defines the
finite role/evidence metadata that a later bank generator must satisfy; it
does not materialize a bank, render policy input, or authorize weight updates.

The family order is placard, literal, composed.  If the Official family has
index ``i``, alternative A has index ``(i + 1) % 3`` and alternative B has
index ``(i + 2) % 3``.  Consequently the composed-Official primary rotation
maps A to the placard proxy and B to the one-literal semantic shortcut.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from enum import Enum
from typing import Any, cast

ROLE_SCHEMA_VERSION = 1
META_ROLE_BALANCE_SCHEMA_VERSION = 1
DEMONSTRATION_COUNT = 10
MIN_META_ROLE_INDEPENDENT_GROUPS = 384

_ROLE_BLOCK_KIND = "g03-v2-role-counterbalanced-evidence-block"
_META_AUDIT_KIND = "g03-v2-metadata-only-role-balance-audit"
_ROLE_BLOCK_DOMAIN = "goalzendo-interactive-v2-role-block-v1"
_MODEL_SCHEDULE_DOMAIN = "goalzendo-interactive-v2-model-schedule-semantics-v1"
_GROUP_MANIFEST_DOMAIN = "goalzendo-interactive-v2-role-group-manifest-v1"
_META_AUDIT_DOMAIN = "goalzendo-interactive-v2-meta-role-balance-audit-v1"

_AUTHORIZATION = {
    "scope": "prospective-engineering-schema-only",
    "production_bank_materialized": False,
    "weight_updates_authorized": False,
}

_CELL_ORDER: tuple[tuple[int, int, int], ...] = tuple(
    (official, alternative_a, alternative_b)
    for official in (0, 1)
    for alternative_a in (0, 1)
    for alternative_b in (0, 1)
)
_CELL_LABELS = tuple("".join(str(bit) for bit in cell) for cell in _CELL_ORDER)


class RoleSchemaV2Error(ValueError):
    """Raised when prospective G03-v2 role metadata is not canonical."""


class RuleFamilyV2(Enum):
    """The three syntactic families rotated through all candidate roles."""

    PLACARD = "placard"
    LITERAL = "literal"
    COMPOSED = "composed"


class CandidateRoleV2(Enum):
    """A candidate's role within one hidden-law episode."""

    OFFICIAL = "official"
    ALTERNATIVE_A = "A"
    ALTERNATIVE_B = "B"


class EpisodePurposeV2(Enum):
    """Evaluator-only status of one rotation."""

    PRIMARY = "primary"
    ROLE_COVER = "role_cover"


class AlternativeAErrorsV2(Enum):
    """Registered error count for the zero/one-error alternative."""

    PERFECT = 0
    NOISY = 1


class AlternativeBErrorsV2(Enum):
    """Registered error count for the zero/two-error alternative."""

    PERFECT = 0
    NOISY = 2


class OfficialTargetSideV2(Enum):
    """Official-Law target class on which the single A error occurs."""

    NONFITTING = 0
    FITTING = 1


CYCLIC_FAMILY_ORDER: tuple[RuleFamilyV2, ...] = (
    RuleFamilyV2.PLACARD,
    RuleFamilyV2.LITERAL,
    RuleFamilyV2.COMPOSED,
)
ROLE_ORDER: tuple[CandidateRoleV2, ...] = (
    CandidateRoleV2.OFFICIAL,
    CandidateRoleV2.ALTERNATIVE_A,
    CandidateRoleV2.ALTERNATIVE_B,
)


def _dump_json(value: Any) -> str:
    try:
        return json.dumps(value, ensure_ascii=True, allow_nan=False, separators=(",", ":"))
    except (TypeError, ValueError) as exc:
        raise RoleSchemaV2Error(f"value is not canonical JSON: {exc}") from exc


def _load_json(text: str) -> Any:
    if type(text) is not str or not text:
        raise RoleSchemaV2Error("JSON input must be a nonempty string")

    def no_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise RoleSchemaV2Error(f"duplicate JSON object key: {key!r}")
            result[key] = value
        return result

    def reject_constant(value: str) -> None:
        raise RoleSchemaV2Error(f"non-finite JSON constant is forbidden: {value}")

    try:
        return json.loads(
            text,
            object_pairs_hook=no_duplicates,
            parse_constant=reject_constant,
        )
    except RoleSchemaV2Error:
        raise
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise RoleSchemaV2Error(f"invalid JSON: {exc}") from exc


def _json_digest(value: Any, *, domain: str) -> str:
    digest = hashlib.sha256()
    digest.update(domain.encode("ascii"))
    digest.update(b"\0")
    digest.update(_dump_json(value).encode("ascii"))
    return digest.hexdigest()


def _is_sha256(value: object) -> bool:
    return (
        type(value) is str
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _require_mapping(
    value: object,
    expected_fields: tuple[str, ...],
    *,
    name: str,
) -> Mapping[str, Any]:
    if type(value) is not dict or tuple(value) != expected_fields:
        raise RoleSchemaV2Error(f"{name} has noncanonical or reordered fields")
    return cast(Mapping[str, Any], value)


def _require_integer(value: object, *, name: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise RoleSchemaV2Error(f"{name} must be an integer >= {minimum}")
    return value


def _parse_enum(value: object, enum_type: type[Enum], *, name: str) -> Enum:
    if type(value) not in (str, int):
        raise RoleSchemaV2Error(f"{name} is not a canonical enum value")
    try:
        return enum_type(value)
    except ValueError as exc:
        raise RoleSchemaV2Error(f"unknown {name}: {value!r}") from exc


@dataclass(frozen=True, slots=True)
class EvidenceGeometryV2:
    """One cell of the A-error by B-error factorial geometry."""

    alternative_a_errors: AlternativeAErrorsV2
    alternative_b_errors: AlternativeBErrorsV2

    def __post_init__(self) -> None:
        if type(self.alternative_a_errors) is not AlternativeAErrorsV2:
            raise RoleSchemaV2Error("alternative_a_errors has the wrong enum type")
        if type(self.alternative_b_errors) is not AlternativeBErrorsV2:
            raise RoleSchemaV2Error("alternative_b_errors has the wrong enum type")

    @property
    def slug(self) -> str:
        return f"a{self.alternative_a_errors.value}_b{self.alternative_b_errors.value}"

    def as_obj(self) -> dict[str, Any]:
        return {
            "slug": self.slug,
            "alternative_a_errors": self.alternative_a_errors.value,
            "alternative_b_errors": self.alternative_b_errors.value,
        }


EVIDENCE_GEOMETRIES: tuple[EvidenceGeometryV2, ...] = (
    EvidenceGeometryV2(AlternativeAErrorsV2.PERFECT, AlternativeBErrorsV2.PERFECT),
    EvidenceGeometryV2(AlternativeAErrorsV2.NOISY, AlternativeBErrorsV2.PERFECT),
    EvidenceGeometryV2(AlternativeAErrorsV2.PERFECT, AlternativeBErrorsV2.NOISY),
    EvidenceGeometryV2(AlternativeAErrorsV2.NOISY, AlternativeBErrorsV2.NOISY),
)
_GEOMETRY_BY_SLUG = {geometry.slug: geometry for geometry in EVIDENCE_GEOMETRIES}


def _geometry_from_obj(value: object) -> EvidenceGeometryV2:
    obj = _require_mapping(
        value,
        ("slug", "alternative_a_errors", "alternative_b_errors"),
        name="evidence geometry",
    )
    a_errors = cast(
        AlternativeAErrorsV2,
        _parse_enum(
            obj["alternative_a_errors"],
            AlternativeAErrorsV2,
            name="alternative_a_errors",
        ),
    )
    b_errors = cast(
        AlternativeBErrorsV2,
        _parse_enum(
            obj["alternative_b_errors"],
            AlternativeBErrorsV2,
            name="alternative_b_errors",
        ),
    )
    geometry = EvidenceGeometryV2(a_errors, b_errors)
    if obj["slug"] != geometry.slug:
        raise RoleSchemaV2Error("evidence-geometry slug does not match its axes")
    return geometry


def _canonical_cell_counts(
    geometry: EvidenceGeometryV2,
    a_error_target_side: OfficialTargetSideV2 | None,
) -> tuple[int, ...]:
    if geometry.alternative_a_errors is AlternativeAErrorsV2.PERFECT:
        if a_error_target_side is not None:
            raise RoleSchemaV2Error("a perfect A schedule cannot name an A-error target side")
    elif type(a_error_target_side) is not OfficialTargetSideV2:
        raise RoleSchemaV2Error("a noisy A schedule requires an Official target side")

    counts = [5, 0, 0, 0, 0, 0, 0, 5]
    if geometry.alternative_b_errors is AlternativeBErrorsV2.NOISY:
        counts[0] -= 1
        counts[1] += 1
        counts[7] -= 1
        counts[6] += 1
    if a_error_target_side is OfficialTargetSideV2.NONFITTING:
        counts[0] -= 1
        counts[2] += 1
    elif a_error_target_side is OfficialTargetSideV2.FITTING:
        counts[7] -= 1
        counts[5] += 1
    return tuple(counts)


@dataclass(frozen=True, slots=True)
class EvidenceScheduleV2:
    """The exact ten-demonstration joint truth-cell schedule."""

    geometry: EvidenceGeometryV2
    a_error_target_side: OfficialTargetSideV2 | None
    joint_truth_cell_counts: tuple[int, ...]

    def __post_init__(self) -> None:
        if type(self.geometry) is not EvidenceGeometryV2:
            raise RoleSchemaV2Error("schedule geometry has the wrong type")
        if type(self.joint_truth_cell_counts) is not tuple or len(self.joint_truth_cell_counts) != len(
            _CELL_ORDER
        ):
            raise RoleSchemaV2Error("schedule requires eight ordered joint-cell counts")
        for count in self.joint_truth_cell_counts:
            _require_integer(count, name="joint truth-cell count")
        expected = _canonical_cell_counts(self.geometry, self.a_error_target_side)
        if self.joint_truth_cell_counts != expected:
            raise RoleSchemaV2Error("joint truth-cell counts are not the canonical schedule")
        if sum(self.joint_truth_cell_counts) != DEMONSTRATION_COUNT:
            raise RoleSchemaV2Error("schedule must contain exactly ten demonstrations")
        if self.official_class_counts != (5, 5):
            raise RoleSchemaV2Error("Official target classes must be exactly balanced")
        if self.alternative_a_error_count != self.geometry.alternative_a_errors.value:
            raise RoleSchemaV2Error("A error count disagrees with the geometry")
        if self.alternative_b_error_count != self.geometry.alternative_b_errors.value:
            raise RoleSchemaV2Error("B error count disagrees with the geometry")
        if self.joint_error_overlap_count != 0:
            raise RoleSchemaV2Error("A and B errors must occur on distinct demonstrations")
        if self.geometry.alternative_b_errors is AlternativeBErrorsV2.NOISY:
            if self.alternative_b_errors_by_official_side != (1, 1):
                raise RoleSchemaV2Error("the two B errors must straddle Official target side")
        elif self.alternative_b_errors_by_official_side != (0, 0):
            raise RoleSchemaV2Error("a perfect B schedule cannot contain errors")

    @property
    def official_class_counts(self) -> tuple[int, int]:
        return tuple(
            sum(
                count
                for cell, count in zip(_CELL_ORDER, self.joint_truth_cell_counts, strict=True)
                if cell[0] == target
            )
            for target in (0, 1)
        )  # type: ignore[return-value]

    @property
    def alternative_a_error_count(self) -> int:
        return sum(
            count
            for cell, count in zip(_CELL_ORDER, self.joint_truth_cell_counts, strict=True)
            if cell[0] != cell[1]
        )

    @property
    def alternative_b_error_count(self) -> int:
        return sum(
            count
            for cell, count in zip(_CELL_ORDER, self.joint_truth_cell_counts, strict=True)
            if cell[0] != cell[2]
        )

    @property
    def joint_error_overlap_count(self) -> int:
        return sum(
            count
            for cell, count in zip(_CELL_ORDER, self.joint_truth_cell_counts, strict=True)
            if cell[0] != cell[1] and cell[0] != cell[2]
        )

    @property
    def alternative_b_errors_by_official_side(self) -> tuple[int, int]:
        return tuple(
            sum(
                count
                for cell, count in zip(_CELL_ORDER, self.joint_truth_cell_counts, strict=True)
                if cell[0] == target and cell[0] != cell[2]
            )
            for target in (0, 1)
        )  # type: ignore[return-value]

    @property
    def alternative_a_agreement_count(self) -> int:
        return DEMONSTRATION_COUNT - self.alternative_a_error_count

    @property
    def alternative_b_agreement_count(self) -> int:
        return DEMONSTRATION_COUNT - self.alternative_b_error_count

    def model_semantics_obj(self) -> dict[str, Any]:
        """Return role-relative schedule semantics with no evaluator purpose flag."""

        return {
            "demonstration_count": DEMONSTRATION_COUNT,
            "joint_truth_cell_order": list(_CELL_LABELS),
            "joint_truth_cell_counts": list(self.joint_truth_cell_counts),
        }

    @property
    def model_semantics_digest(self) -> str:
        return _json_digest(self.model_semantics_obj(), domain=_MODEL_SCHEDULE_DOMAIN)

    def as_obj(self) -> dict[str, Any]:
        return {
            "geometry": self.geometry.as_obj(),
            "a_error_target_side": (
                None if self.a_error_target_side is None else self.a_error_target_side.value
            ),
            "joint_truth_cell_order": list(_CELL_LABELS),
            "joint_truth_cell_counts": list(self.joint_truth_cell_counts),
            "demonstration_count": DEMONSTRATION_COUNT,
            "official_class_counts": list(self.official_class_counts),
            "alternative_a_error_count": self.alternative_a_error_count,
            "alternative_b_error_count": self.alternative_b_error_count,
            "alternative_a_agreement_count": self.alternative_a_agreement_count,
            "alternative_b_agreement_count": self.alternative_b_agreement_count,
            "joint_error_overlap_count": self.joint_error_overlap_count,
            "alternative_b_errors_by_official_side": list(self.alternative_b_errors_by_official_side),
            "model_semantics_digest": self.model_semantics_digest,
        }


def build_evidence_schedule_v2(
    geometry: EvidenceGeometryV2,
    *,
    a_error_target_side: OfficialTargetSideV2 | None = None,
) -> EvidenceScheduleV2:
    """Build the unique registered ten-demonstration schedule for one cell."""

    if type(geometry) is not EvidenceGeometryV2:
        raise TypeError("geometry must be an EvidenceGeometryV2")
    return EvidenceScheduleV2(
        geometry=geometry,
        a_error_target_side=a_error_target_side,
        joint_truth_cell_counts=_canonical_cell_counts(geometry, a_error_target_side),
    )


def _schedule_from_obj(value: object) -> EvidenceScheduleV2:
    fields = (
        "geometry",
        "a_error_target_side",
        "joint_truth_cell_order",
        "joint_truth_cell_counts",
        "demonstration_count",
        "official_class_counts",
        "alternative_a_error_count",
        "alternative_b_error_count",
        "alternative_a_agreement_count",
        "alternative_b_agreement_count",
        "joint_error_overlap_count",
        "alternative_b_errors_by_official_side",
        "model_semantics_digest",
    )
    obj = _require_mapping(value, fields, name="evidence schedule")
    geometry = _geometry_from_obj(obj["geometry"])
    side_value = obj["a_error_target_side"]
    side: OfficialTargetSideV2 | None
    if side_value is None:
        side = None
    else:
        side = cast(
            OfficialTargetSideV2,
            _parse_enum(side_value, OfficialTargetSideV2, name="a_error_target_side"),
        )
    counts_obj = obj["joint_truth_cell_counts"]
    if type(counts_obj) is not list:
        raise RoleSchemaV2Error("joint_truth_cell_counts must be an array")
    counts = tuple(_require_integer(item, name="joint truth-cell count") for item in counts_obj)
    schedule = EvidenceScheduleV2(geometry, side, counts)
    if _dump_json(obj) != _dump_json(schedule.as_obj()):
        raise RoleSchemaV2Error("evidence schedule contains inconsistent derived metadata")
    return schedule


@dataclass(frozen=True, slots=True)
class CandidateAssignmentV2:
    """One family-to-role assignment in an episode rotation."""

    family: RuleFamilyV2
    role: CandidateRoleV2

    def __post_init__(self) -> None:
        if type(self.family) is not RuleFamilyV2 or type(self.role) is not CandidateRoleV2:
            raise RoleSchemaV2Error("candidate assignment uses an invalid enum type")

    def as_obj(self) -> dict[str, str]:
        return {"family": self.family.value, "role": self.role.value}


@dataclass(frozen=True, slots=True)
class RoleRotationV2:
    """One of the three cyclic Official/A/B assignments."""

    position: int
    candidate_assignments: tuple[CandidateAssignmentV2, ...]
    purpose: EpisodePurposeV2
    schedule: EvidenceScheduleV2

    def __post_init__(self) -> None:
        if self.position not in range(len(CYCLIC_FAMILY_ORDER)):
            raise RoleSchemaV2Error("rotation position must be zero, one, or two")
        if type(self.candidate_assignments) is not tuple or len(self.candidate_assignments) != len(
            ROLE_ORDER
        ):
            raise RoleSchemaV2Error("rotation must contain exactly three candidate assignments")
        expected = tuple(
            CandidateAssignmentV2(
                CYCLIC_FAMILY_ORDER[(self.position + offset) % len(CYCLIC_FAMILY_ORDER)],
                role,
            )
            for offset, role in enumerate(ROLE_ORDER)
        )
        if self.candidate_assignments != expected:
            raise RoleSchemaV2Error("candidate assignments do not follow the cyclic rotation")
        expected_purpose = (
            EpisodePurposeV2.PRIMARY
            if self.official_family is RuleFamilyV2.COMPOSED
            else EpisodePurposeV2.ROLE_COVER
        )
        if self.purpose is not expected_purpose:
            raise RoleSchemaV2Error("rotation purpose disagrees with its Official family")
        if type(self.schedule) is not EvidenceScheduleV2:
            raise RoleSchemaV2Error("rotation schedule has the wrong type")

    @property
    def official_family(self) -> RuleFamilyV2:
        return self.candidate_assignments[0].family

    @property
    def alternative_a_family(self) -> RuleFamilyV2:
        return self.candidate_assignments[1].family

    @property
    def alternative_b_family(self) -> RuleFamilyV2:
        return self.candidate_assignments[2].family

    def model_semantics_obj(self) -> dict[str, Any]:
        """Return the only schedule metadata allowed to define rendered semantics."""

        return self.schedule.model_semantics_obj()

    def as_obj(self) -> dict[str, Any]:
        return {
            "position": self.position,
            "candidate_assignments": [assignment.as_obj() for assignment in self.candidate_assignments],
            "evaluator_only": {"purpose": self.purpose.value},
            "schedule": self.schedule.as_obj(),
        }


def _build_rotation(position: int, schedule: EvidenceScheduleV2) -> RoleRotationV2:
    official_family = CYCLIC_FAMILY_ORDER[position]
    purpose = (
        EpisodePurposeV2.PRIMARY if official_family is RuleFamilyV2.COMPOSED else EpisodePurposeV2.ROLE_COVER
    )
    assignments = tuple(
        CandidateAssignmentV2(
            CYCLIC_FAMILY_ORDER[(position + offset) % len(CYCLIC_FAMILY_ORDER)],
            role,
        )
        for offset, role in enumerate(ROLE_ORDER)
    )
    return RoleRotationV2(position, assignments, purpose, schedule)


def _rotation_from_obj(value: object) -> RoleRotationV2:
    obj = _require_mapping(
        value,
        ("position", "candidate_assignments", "evaluator_only", "schedule"),
        name="role rotation",
    )
    position = _require_integer(obj["position"], name="rotation position")
    assignments_obj = obj["candidate_assignments"]
    if type(assignments_obj) is not list:
        raise RoleSchemaV2Error("candidate_assignments must be an array")
    assignments: list[CandidateAssignmentV2] = []
    for raw_assignment in assignments_obj:
        assignment_obj = _require_mapping(
            raw_assignment,
            ("family", "role"),
            name="candidate assignment",
        )
        assignments.append(
            CandidateAssignmentV2(
                cast(
                    RuleFamilyV2,
                    _parse_enum(assignment_obj["family"], RuleFamilyV2, name="rule family"),
                ),
                cast(
                    CandidateRoleV2,
                    _parse_enum(assignment_obj["role"], CandidateRoleV2, name="candidate role"),
                ),
            )
        )
    evaluator = _require_mapping(
        obj["evaluator_only"],
        ("purpose",),
        name="evaluator-only rotation metadata",
    )
    purpose = cast(
        EpisodePurposeV2,
        _parse_enum(evaluator["purpose"], EpisodePurposeV2, name="episode purpose"),
    )
    rotation = RoleRotationV2(
        position=position,
        candidate_assignments=tuple(assignments),
        purpose=purpose,
        schedule=_schedule_from_obj(obj["schedule"]),
    )
    if _dump_json(obj) != _dump_json(rotation.as_obj()):
        raise RoleSchemaV2Error("role rotation is not canonical")
    return rotation


@dataclass(frozen=True, slots=True)
class RoleBlockV2:
    """A complete, immutable P-to-Q-to-C role-rotation block."""

    semantic_triple_id: str
    rotations: tuple[RoleRotationV2, ...]

    def __post_init__(self) -> None:
        if not _is_sha256(self.semantic_triple_id):
            raise RoleSchemaV2Error("semantic_triple_id must be a lowercase SHA-256")
        if type(self.rotations) is not tuple or len(self.rotations) != len(CYCLIC_FAMILY_ORDER):
            raise RoleSchemaV2Error("role block must contain exactly three rotations")
        if tuple(rotation.position for rotation in self.rotations) != (0, 1, 2):
            raise RoleSchemaV2Error("role rotations must be complete and in canonical order")
        schedules = {rotation.schedule for rotation in self.rotations}
        if len(schedules) != 1:
            raise RoleSchemaV2Error("all rotations must use identical schedule semantics")
        model_semantics = {_dump_json(rotation.model_semantics_obj()) for rotation in self.rotations}
        if len(model_semantics) != 1:
            raise RoleSchemaV2Error("model-visible schedule semantics differ across rotations")
        for family in CYCLIC_FAMILY_ORDER:
            for role in ROLE_ORDER:
                count = sum(
                    assignment.family is family and assignment.role is role
                    for rotation in self.rotations
                    for assignment in rotation.candidate_assignments
                )
                if count != 1:
                    raise RoleSchemaV2Error("each family must occupy each role exactly once")
        primary = tuple(
            rotation for rotation in self.rotations if rotation.purpose is EpisodePurposeV2.PRIMARY
        )
        if len(primary) != 1:
            raise RoleSchemaV2Error("role block must contain exactly one primary rotation")
        if (
            primary[0].official_family is not RuleFamilyV2.COMPOSED
            or primary[0].alternative_a_family is not RuleFamilyV2.PLACARD
            or primary[0].alternative_b_family is not RuleFamilyV2.LITERAL
        ):
            raise RoleSchemaV2Error("primary rotation must map Official=C, A=P, and B=Q")

    @property
    def schedule(self) -> EvidenceScheduleV2:
        return self.rotations[0].schedule

    @property
    def geometry(self) -> EvidenceGeometryV2:
        return self.schedule.geometry

    @property
    def a_error_target_side(self) -> OfficialTargetSideV2 | None:
        return self.schedule.a_error_target_side

    def _unsigned_obj(self) -> dict[str, Any]:
        return {
            "schema_version": ROLE_SCHEMA_VERSION,
            "report_kind": _ROLE_BLOCK_KIND,
            "authorization": dict(_AUTHORIZATION),
            "semantic_triple_id": self.semantic_triple_id,
            "geometry": self.geometry.as_obj(),
            "a_error_target_side": (
                None if self.a_error_target_side is None else self.a_error_target_side.value
            ),
            "model_semantics_digest": self.schedule.model_semantics_digest,
            "rotations": [rotation.as_obj() for rotation in self.rotations],
        }

    @property
    def digest(self) -> str:
        return _json_digest(self._unsigned_obj(), domain=_ROLE_BLOCK_DOMAIN)

    def as_obj(self) -> dict[str, Any]:
        return {**self._unsigned_obj(), "block_digest": self.digest}


def build_role_block_v2(
    semantic_triple_id: str,
    geometry: EvidenceGeometryV2,
    *,
    a_error_target_side: OfficialTargetSideV2 | None = None,
) -> RoleBlockV2:
    """Build a complete cyclic role block for one independent semantic triple."""

    schedule = build_evidence_schedule_v2(
        geometry,
        a_error_target_side=a_error_target_side,
    )
    return RoleBlockV2(
        semantic_triple_id=semantic_triple_id,
        rotations=tuple(_build_rotation(position, schedule) for position in range(3)),
    )


def role_block_v2_from_obj(
    value: object,
    *,
    expected_digest: str | None = None,
) -> RoleBlockV2:
    """Reconstruct one role block while rejecting any noncanonical metadata."""

    fields = (
        "schema_version",
        "report_kind",
        "authorization",
        "semantic_triple_id",
        "geometry",
        "a_error_target_side",
        "model_semantics_digest",
        "rotations",
        "block_digest",
    )
    obj = _require_mapping(value, fields, name="role block")
    if obj["schema_version"] != ROLE_SCHEMA_VERSION or obj["report_kind"] != _ROLE_BLOCK_KIND:
        raise RoleSchemaV2Error("role-block schema identity mismatch")
    authorization = _require_mapping(
        obj["authorization"],
        tuple(_AUTHORIZATION),
        name="authorization",
    )
    if dict(authorization) != _AUTHORIZATION:
        raise RoleSchemaV2Error("role block is not nonauthorizing")
    geometry = _geometry_from_obj(obj["geometry"])
    side_value = obj["a_error_target_side"]
    if side_value is None:
        side = None
    else:
        side = cast(
            OfficialTargetSideV2,
            _parse_enum(side_value, OfficialTargetSideV2, name="a_error_target_side"),
        )
    rotations_obj = obj["rotations"]
    if type(rotations_obj) is not list:
        raise RoleSchemaV2Error("rotations must be an array")
    block = RoleBlockV2(
        semantic_triple_id=cast(str, obj["semantic_triple_id"]),
        rotations=tuple(_rotation_from_obj(rotation) for rotation in rotations_obj),
    )
    if block.geometry != geometry or block.a_error_target_side is not side:
        raise RoleSchemaV2Error("top-level geometry metadata disagrees with rotations")
    if obj["model_semantics_digest"] != block.schedule.model_semantics_digest:
        raise RoleSchemaV2Error("model-semantics digest mismatch")
    if obj["block_digest"] != block.digest:
        raise RoleSchemaV2Error("role-block digest mismatch")
    if expected_digest is not None and block.digest != expected_digest:
        raise RoleSchemaV2Error("role block differs from the externally expected digest")
    if _dump_json(obj) != _dump_json(block.as_obj()):
        raise RoleSchemaV2Error("role block is not the canonical deterministic construction")
    return block


def serialize_role_block_v2(block: RoleBlockV2) -> str:
    """Serialize a role block in its unique compact JSON representation."""

    if type(block) is not RoleBlockV2:
        raise TypeError("block must be a RoleBlockV2")
    return _dump_json(block.as_obj())


def parse_role_block_v2(
    text: str,
    *,
    expected_digest: str | None = None,
) -> RoleBlockV2:
    """Parse only the exact canonical JSON spelling of a valid role block."""

    block = role_block_v2_from_obj(_load_json(text), expected_digest=expected_digest)
    if serialize_role_block_v2(block) != text:
        raise RoleSchemaV2Error("role-block JSON is not in canonical compact form")
    return block


def _family_role_counts(blocks: tuple[RoleBlockV2, ...]) -> Counter[tuple[str, str]]:
    result: Counter[tuple[str, str]] = Counter()
    for block in blocks:
        for rotation in block.rotations:
            for assignment in rotation.candidate_assignments:
                result[(assignment.family.value, assignment.role.value)] += 1
    return result


def _geometry_role_counts(blocks: tuple[RoleBlockV2, ...]) -> Counter[tuple[str, str]]:
    result: Counter[tuple[str, str]] = Counter()
    for block in blocks:
        for rotation in block.rotations:
            for assignment in rotation.candidate_assignments:
                result[(block.geometry.slug, assignment.role.value)] += 1
    return result


def _geometry_group_counts(blocks: tuple[RoleBlockV2, ...]) -> Counter[str]:
    return Counter(block.geometry.slug for block in blocks)


def _mirror_counts(blocks: tuple[RoleBlockV2, ...]) -> Counter[tuple[str, str]]:
    result: Counter[tuple[str, str]] = Counter()
    for block in blocks:
        side = block.a_error_target_side
        if side is not None:
            result[(block.geometry.slug, side.name.lower())] += 1
    return result


@dataclass(frozen=True, slots=True)
class MetaRoleBalanceAuditV2:
    """A deterministic structural balance audit over independent role blocks.

    No classifier is fitted and no confidence interval is estimated here.  A
    passing report is necessary but insufficient for a later powered
    meta-leakage gate.
    """

    blocks: tuple[RoleBlockV2, ...]

    def __post_init__(self) -> None:
        if type(self.blocks) is not tuple or any(type(block) is not RoleBlockV2 for block in self.blocks):
            raise RoleSchemaV2Error("meta-role audit requires an immutable tuple of role blocks")
        canonical = tuple(sorted(self.blocks, key=lambda block: (block.semantic_triple_id, block.digest)))
        if self.blocks != canonical:
            raise RoleSchemaV2Error("meta-role audit blocks must be in canonical digest order")

    @property
    def submitted_block_count(self) -> int:
        return len(self.blocks)

    @property
    def independent_semantic_triple_count(self) -> int:
        return len({block.semantic_triple_id for block in self.blocks})

    @property
    def duplicate_semantic_triple_ids(self) -> tuple[str, ...]:
        counts = Counter(block.semantic_triple_id for block in self.blocks)
        return tuple(sorted(identifier for identifier, count in counts.items() if count != 1))

    def _group_manifest(self) -> list[dict[str, Any]]:
        return [
            {
                "semantic_triple_id": block.semantic_triple_id,
                "geometry": block.geometry.as_obj(),
                "a_error_target_side": (
                    None if block.a_error_target_side is None else block.a_error_target_side.value
                ),
                "block_digest": block.digest,
            }
            for block in self.blocks
        ]

    @property
    def group_manifest_digest(self) -> str:
        return _json_digest(self._group_manifest(), domain=_GROUP_MANIFEST_DOMAIN)

    def _family_role_table(self) -> list[dict[str, Any]]:
        counts = _family_role_counts(self.blocks)
        return [
            {"family": family.value, "role": role.value, "count": counts[(family.value, role.value)]}
            for family in CYCLIC_FAMILY_ORDER
            for role in ROLE_ORDER
        ]

    def _geometry_role_table(self) -> list[dict[str, Any]]:
        counts = _geometry_role_counts(self.blocks)
        return [
            {"geometry": geometry.slug, "role": role.value, "count": counts[(geometry.slug, role.value)]}
            for geometry in EVIDENCE_GEOMETRIES
            for role in ROLE_ORDER
        ]

    def _geometry_group_table(self) -> list[dict[str, Any]]:
        counts = _geometry_group_counts(self.blocks)
        return [
            {"geometry": geometry.slug, "count": counts[geometry.slug]} for geometry in EVIDENCE_GEOMETRIES
        ]

    def _mirror_table(self) -> list[dict[str, Any]]:
        counts = _mirror_counts(self.blocks)
        return [
            {
                "geometry": geometry.slug,
                "a_error_target_side": side.name.lower(),
                "count": counts[(geometry.slug, side.name.lower())],
            }
            for geometry in EVIDENCE_GEOMETRIES
            if geometry.alternative_a_errors is AlternativeAErrorsV2.NOISY
            for side in OfficialTargetSideV2
        ]

    def _checks(self) -> dict[str, bool]:
        block_count = self.submitted_block_count
        family_role = _family_role_counts(self.blocks)
        geometry_role = _geometry_role_counts(self.blocks)
        geometry_groups = _geometry_group_counts(self.blocks)
        mirror = _mirror_counts(self.blocks)
        exact_geometry_group_count = block_count // len(EVIDENCE_GEOMETRIES)
        geometry_divisible = block_count % len(EVIDENCE_GEOMETRIES) == 0
        family_role_exact = all(
            family_role[(family.value, role.value)] == block_count
            for family in CYCLIC_FAMILY_ORDER
            for role in ROLE_ORDER
        )
        geometry_group_exact = geometry_divisible and all(
            geometry_groups[geometry.slug] == exact_geometry_group_count for geometry in EVIDENCE_GEOMETRIES
        )
        geometry_role_exact = geometry_divisible and all(
            geometry_role[(geometry.slug, role.value)] == exact_geometry_group_count * 3
            for geometry in EVIDENCE_GEOMETRIES
            for role in ROLE_ORDER
        )
        mirror_exact = all(
            geometry_groups[geometry.slug] % 2 == 0
            and all(
                mirror[(geometry.slug, side.name.lower())] == geometry_groups[geometry.slug] // 2
                for side in OfficialTargetSideV2
            )
            for geometry in EVIDENCE_GEOMETRIES
            if geometry.alternative_a_errors is AlternativeAErrorsV2.NOISY
        )
        return {
            "unique_semantic_triple_ids": not self.duplicate_semantic_triple_ids,
            "minimum_independent_group_count": (
                self.independent_semantic_triple_count >= MIN_META_ROLE_INDEPENDENT_GROUPS
            ),
            "family_by_role_exact": family_role_exact,
            "geometry_group_balance_exact": geometry_group_exact,
            "geometry_by_role_exact": geometry_role_exact,
            "noisy_a_target_mirror_exact": mirror_exact,
        }

    @property
    def passed(self) -> bool:
        return all(self._checks().values())

    def _unsigned_obj(self) -> dict[str, Any]:
        checks = self._checks()
        return {
            "schema_version": META_ROLE_BALANCE_SCHEMA_VERSION,
            "report_kind": _META_AUDIT_KIND,
            "authorization": dict(_AUTHORIZATION),
            "audit_scope": "metadata-only-structural-balance",
            "classifier_fit_performed": False,
            "powered_leakage_test_performed": False,
            "minimum_independent_group_count": MIN_META_ROLE_INDEPENDENT_GROUPS,
            "submitted_block_count": self.submitted_block_count,
            "independent_semantic_triple_count": self.independent_semantic_triple_count,
            "episode_count": self.submitted_block_count * 3,
            "candidate_row_count": self.submitted_block_count * 9,
            "duplicate_semantic_triple_ids": list(self.duplicate_semantic_triple_ids),
            "group_manifest_digest": self.group_manifest_digest,
            "group_manifest": self._group_manifest(),
            "family_by_role_counts": self._family_role_table(),
            "geometry_group_counts": self._geometry_group_table(),
            "geometry_by_role_counts": self._geometry_role_table(),
            "noisy_a_target_mirror_counts": self._mirror_table(),
            "checks": [{"name": name, "passed": passed} for name, passed in checks.items()],
            "passed": self.passed,
            "limitation": (
                "necessary structural gate only; the registered grouped powered "
                "meta-leakage classifier remains required"
            ),
        }

    @property
    def digest(self) -> str:
        return _json_digest(self._unsigned_obj(), domain=_META_AUDIT_DOMAIN)

    def as_obj(self) -> dict[str, Any]:
        return {**self._unsigned_obj(), "audit_digest": self.digest}


def build_meta_role_balance_audit_v2(
    blocks: Iterable[RoleBlockV2],
) -> MetaRoleBalanceAuditV2:
    """Build the nonauthorizing structural role/geometry balance report."""

    materialized = tuple(blocks)
    if any(type(block) is not RoleBlockV2 for block in materialized):
        raise TypeError("blocks must contain only RoleBlockV2 values")
    return MetaRoleBalanceAuditV2(
        tuple(sorted(materialized, key=lambda block: (block.semantic_triple_id, block.digest)))
    )


def _block_from_manifest_entry(value: object) -> RoleBlockV2:
    obj = _require_mapping(
        value,
        ("semantic_triple_id", "geometry", "a_error_target_side", "block_digest"),
        name="meta-role group-manifest entry",
    )
    geometry = _geometry_from_obj(obj["geometry"])
    raw_side = obj["a_error_target_side"]
    if raw_side is None:
        side = None
    else:
        side = cast(
            OfficialTargetSideV2,
            _parse_enum(raw_side, OfficialTargetSideV2, name="a_error_target_side"),
        )
    identifier = obj["semantic_triple_id"]
    if type(identifier) is not str:
        raise RoleSchemaV2Error("semantic_triple_id must be a string")
    block = build_role_block_v2(identifier, geometry, a_error_target_side=side)
    if obj["block_digest"] != block.digest:
        raise RoleSchemaV2Error("group-manifest block digest mismatch")
    return block


def meta_role_balance_audit_v2_from_obj(
    value: object,
    *,
    expected_digest: str | None = None,
) -> MetaRoleBalanceAuditV2:
    """Reconstruct and independently recompute a structural audit report."""

    fields = (
        "schema_version",
        "report_kind",
        "authorization",
        "audit_scope",
        "classifier_fit_performed",
        "powered_leakage_test_performed",
        "minimum_independent_group_count",
        "submitted_block_count",
        "independent_semantic_triple_count",
        "episode_count",
        "candidate_row_count",
        "duplicate_semantic_triple_ids",
        "group_manifest_digest",
        "group_manifest",
        "family_by_role_counts",
        "geometry_group_counts",
        "geometry_by_role_counts",
        "noisy_a_target_mirror_counts",
        "checks",
        "passed",
        "limitation",
        "audit_digest",
    )
    obj = _require_mapping(value, fields, name="meta-role balance audit")
    if obj["schema_version"] != META_ROLE_BALANCE_SCHEMA_VERSION or obj["report_kind"] != _META_AUDIT_KIND:
        raise RoleSchemaV2Error("meta-role audit schema identity mismatch")
    authorization = _require_mapping(
        obj["authorization"],
        tuple(_AUTHORIZATION),
        name="authorization",
    )
    if dict(authorization) != _AUTHORIZATION:
        raise RoleSchemaV2Error("meta-role report is not nonauthorizing")
    manifest = obj["group_manifest"]
    if type(manifest) is not list:
        raise RoleSchemaV2Error("group_manifest must be an array")
    audit = build_meta_role_balance_audit_v2(_block_from_manifest_entry(entry) for entry in manifest)
    if obj["audit_digest"] != audit.digest:
        raise RoleSchemaV2Error("meta-role audit digest mismatch")
    if expected_digest is not None and audit.digest != expected_digest:
        raise RoleSchemaV2Error("meta-role audit differs from the externally expected digest")
    if _dump_json(obj) != _dump_json(audit.as_obj()):
        raise RoleSchemaV2Error("meta-role audit contains inconsistent derived metadata")
    return audit


def serialize_meta_role_balance_audit_v2(audit: MetaRoleBalanceAuditV2) -> str:
    """Serialize a structural balance audit as unique compact JSON."""

    if type(audit) is not MetaRoleBalanceAuditV2:
        raise TypeError("audit must be a MetaRoleBalanceAuditV2")
    return _dump_json(audit.as_obj())


def parse_meta_role_balance_audit_v2(
    text: str,
    *,
    expected_digest: str | None = None,
) -> MetaRoleBalanceAuditV2:
    """Parse and recompute only canonical structural-audit JSON."""

    audit = meta_role_balance_audit_v2_from_obj(
        _load_json(text),
        expected_digest=expected_digest,
    )
    if serialize_meta_role_balance_audit_v2(audit) != text:
        raise RoleSchemaV2Error("meta-role audit JSON is not in canonical compact form")
    return audit
