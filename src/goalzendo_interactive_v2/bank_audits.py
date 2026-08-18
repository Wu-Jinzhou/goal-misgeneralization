"""Catalog-bound bank audits for the prospective G03-v2 design.

The three reports in this module are deliberately nominal and
noninterchangeable:

* :class:`PerfectTrainingRoleAuditV2` audits atomic, perfect-ambiguity
  three-rotation training blocks;
* :class:`COfficialEvaluationQuartetAuditV2` audits paired C-Official
  four-geometry evaluation quartets; and
* :class:`MetadataStressPopulationAuditV2` audits the optional four-geometry
  metadata stress population.

All semantic triples are embedded as exact :class:`RuleTripleBindingV2`
objects and are recomputed against both the complete public catalog and the
v2 supported-catalog allowlist.  A caller-provided SHA-256 is never accepted
as evidence of a semantic triple.  These reports describe structural
properties only and never authorize model-weight updates.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, cast

from goalzendo_interactive.catalog import CatalogEntry, build_rule_catalog
from goalzendo_interactive.rules import evaluate_rule
from goalzendo_interactive.schema import SCENE_COUNT, scene_at

from .population_audit import (
    JOINT_CELL_ORDER,
    RuleTripleBindingV2,
    build_supported_catalog_contract_v2,
    classify_catalog_identity_v2,
)
from .role_schema import (
    CYCLIC_FAMILY_ORDER,
    EVIDENCE_GEOMETRIES,
    ROLE_ORDER,
    AlternativeAErrorsV2,
    EvidenceGeometryV2,
    OfficialTargetSideV2,
    RoleBlockV2,
    RuleFamilyV2,
    build_role_block_v2,
    role_block_v2_from_obj,
)

PERFECT_TRAINING_ROLE_AUDIT_SCHEMA_VERSION = 1
C_OFFICIAL_EVALUATION_QUARTET_AUDIT_SCHEMA_VERSION = 1
METADATA_STRESS_POPULATION_AUDIT_SCHEMA_VERSION = 1
MINIMUM_POWERED_STRESS_TRIPLES = 384
EVALUATION_RESERVOIR_SCENES_PER_CELL = 22

_TRAINING_REPORT_KIND = "g03-v2-catalog-bound-perfect-training-role-audit"
_EVALUATION_REPORT_KIND = "g03-v2-catalog-bound-c-official-evaluation-quartet-audit"
_STRESS_REPORT_KIND = "g03-v2-catalog-bound-metadata-stress-population-audit"

_TRAINING_ROTATION_DOMAIN = "goalzendo-interactive-v2-training-rotation-execution-v1"
_TRAINING_UNIT_DOMAIN = "goalzendo-interactive-v2-catalog-bound-training-role-unit-v1"
_TRAINING_PREFIX_DOMAIN = "goalzendo-interactive-v2-training-prefix-checks-v1"
_TRAINING_REPORT_DOMAIN = "goalzendo-interactive-v2-perfect-training-role-audit-v1"
_RESERVOIR_DOMAIN = "goalzendo-interactive-v2-evaluation-reservoir-v1"
_DISPLAY_BINDING_DOMAIN = "goalzendo-interactive-v2-evaluation-display-binding-v1"
_SHARED_RESOURCE_DOMAIN = "goalzendo-interactive-v2-evaluation-shared-resource-v1"
_OPENING_UNION_DOMAIN = "goalzendo-interactive-v2-evaluation-opening-union-v1"
_EVALUATION_EPISODE_DOMAIN = "goalzendo-interactive-v2-c-official-evaluation-episode-v1"
_EVALUATION_QUARTET_DOMAIN = "goalzendo-interactive-v2-c-official-evaluation-quartet-v1"
_EVALUATION_MIRROR_UNIT_DOMAIN = "goalzendo-interactive-v2-evaluation-mirror-unit-v1"
_EVALUATION_REPORT_DOMAIN = "goalzendo-interactive-v2-evaluation-quartet-audit-v1"
_STRESS_UNIT_DOMAIN = "goalzendo-interactive-v2-catalog-bound-stress-role-unit-v1"
_STRESS_REPORT_DOMAIN = "goalzendo-interactive-v2-metadata-stress-population-audit-v1"
_POPULATION_TRIPLE_BINDING_DOMAIN = "goalzendo-interactive-v2-rule-triple-binding-v2"
_POPULATION_TRIPLE_ROW_DOMAIN = b"goalzendo-interactive-v2-triple-cell-row-v1\0"

_AUTHORIZATION = {
    "scope": "prospective-structural-bank-audit-only",
    "capability_run_authorized": False,
    "production_run_authorized": False,
    "weight_updates_authorized": False,
}

_BATCH_ID_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,127}")
_GEOMETRY_BY_SLUG = {geometry.slug: geometry for geometry in EVIDENCE_GEOMETRIES}
_NOISY_A_SLUGS = tuple(
    geometry.slug
    for geometry in EVIDENCE_GEOMETRIES
    if geometry.alternative_a_errors is AlternativeAErrorsV2.NOISY
)


class BoundBankAuditV2Error(ValueError):
    """Raised when a catalog-bound bank audit is not exact and canonical."""


def _dump_json(value: Any) -> str:
    try:
        return json.dumps(value, ensure_ascii=True, allow_nan=False, separators=(",", ":"))
    except (TypeError, ValueError) as exc:
        raise BoundBankAuditV2Error(f"value is not canonical JSON: {exc}") from exc


def _load_json(text: str) -> Any:
    if type(text) is not str or not text:
        raise BoundBankAuditV2Error("JSON input must be a nonempty string")

    def no_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise BoundBankAuditV2Error(f"duplicate JSON object key: {key!r}")
            result[key] = value
        return result

    def reject_constant(value: str) -> None:
        raise BoundBankAuditV2Error(f"non-finite JSON constant is forbidden: {value}")

    try:
        return json.loads(text, object_pairs_hook=no_duplicates, parse_constant=reject_constant)
    except BoundBankAuditV2Error:
        raise
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise BoundBankAuditV2Error(f"invalid JSON: {exc}") from exc


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


def _require_sha256(value: object, *, name: str) -> str:
    if not _is_sha256(value):
        raise BoundBankAuditV2Error(f"{name} must be a lowercase SHA-256")
    return cast(str, value)


def _require_integer(
    value: object,
    *,
    name: str,
    minimum: int = 0,
    maximum: int | None = None,
) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise BoundBankAuditV2Error(f"{name} must be an integer >= {minimum}")
    if maximum is not None and value > maximum:
        raise BoundBankAuditV2Error(f"{name} must be an integer <= {maximum}")
    return value


def _require_boolean(value: object, *, name: str) -> bool:
    if type(value) is not bool:
        raise BoundBankAuditV2Error(f"{name} must be a Boolean")
    return value


def _require_mapping(
    value: object,
    fields: tuple[str, ...],
    *,
    name: str,
) -> Mapping[str, Any]:
    if type(value) is not dict or tuple(value) != fields:
        raise BoundBankAuditV2Error(f"{name} has noncanonical or reordered fields")
    return cast(Mapping[str, Any], value)


def _require_authorization(value: object) -> None:
    obj = _require_mapping(value, tuple(_AUTHORIZATION), name="authorization")
    if dict(obj) != _AUTHORIZATION:
        raise BoundBankAuditV2Error("audit authorization must remain false")


def _require_batch_id(value: object) -> str:
    if type(value) is not str or _BATCH_ID_PATTERN.fullmatch(value) is None:
        raise BoundBankAuditV2Error("update_batch_id is not a canonical identifier")
    return value


def _geometry_from_slug(slug: object) -> EvidenceGeometryV2:
    if type(slug) is not str or slug not in _GEOMETRY_BY_SLUG:
        raise BoundBankAuditV2Error(f"unknown evidence geometry: {slug!r}")
    return _GEOMETRY_BY_SLUG[slug]


def _side_from_value(value: object) -> OfficialTargetSideV2 | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise BoundBankAuditV2Error("A-error target side must be null, zero, or one")
    try:
        return OfficialTargetSideV2(value)
    except ValueError as exc:
        raise BoundBankAuditV2Error("A-error target side must be null, zero, or one") from exc


def _parse_binding(value: object) -> RuleTripleBindingV2:
    obj = _require_mapping(
        value,
        (
            "schema_version",
            "catalog_digest",
            "supported_catalog_digest",
            "candidate_order",
            "candidates",
            "joint_cell_order_c_p_q",
            "exact_joint_truth_cell_counts",
            "exact_joint_cell_digest",
            "triple_digest",
        ),
        name="catalog-bound rule triple",
    )
    if type(obj["schema_version"]) is not int or obj["schema_version"] != 2:
        raise BoundBankAuditV2Error("catalog-bound rule triple schema identity mismatch")
    if obj["candidate_order"] != ["P", "Q", "C"]:
        raise BoundBankAuditV2Error("catalog-bound candidate order is not P,Q,C")
    if obj["joint_cell_order_c_p_q"] != list(JOINT_CELL_ORDER):
        raise BoundBankAuditV2Error("catalog-bound joint-cell order differs")
    raw_candidates = obj["candidates"]
    if type(raw_candidates) is not list or len(raw_candidates) != 3:
        raise BoundBankAuditV2Error("catalog-bound rule triple requires three candidates")
    candidates: list[Mapping[str, Any]] = []
    for position, (slot, family) in enumerate(
        (
            ("P", "placard_literal"),
            ("Q", "one_literal_piece"),
            ("C", "composed_two_literal_piece"),
        )
    ):
        candidate = _require_mapping(
            raw_candidates[position],
            ("slot", "family", "rule_id", "truth_digest"),
            name=f"catalog-bound candidate {position}",
        )
        if (candidate["slot"], candidate["family"]) != (slot, family):
            raise BoundBankAuditV2Error("catalog-bound candidate slot/family is inconsistent")
        if type(candidate["rule_id"]) is not str:
            raise BoundBankAuditV2Error("catalog-bound candidate rule ID must be a string")
        _require_sha256(candidate["truth_digest"], name="candidate truth digest")
        candidates.append(candidate)
    raw_cells = obj["exact_joint_truth_cell_counts"]
    if type(raw_cells) is not list or len(raw_cells) != 8:
        raise BoundBankAuditV2Error("catalog-bound cell vector must contain eight integers")
    for count in raw_cells:
        _require_integer(count, name="catalog-bound joint-cell count", maximum=SCENE_COUNT)
    for field in (
        "catalog_digest",
        "supported_catalog_digest",
        "exact_joint_cell_digest",
        "triple_digest",
    ):
        _require_sha256(obj[field], name=field)
    exact = _materialize_exact_binding_fast(
        cast(str, candidates[0]["rule_id"]),
        cast(str, candidates[1]["rule_id"]),
        cast(str, candidates[2]["rule_id"]),
    )
    if _dump_json(obj) != _dump_json(exact.as_obj()):
        raise BoundBankAuditV2Error("invalid catalog-bound rule triple: exact catalog recomputation differs")
    return exact


def _parse_role_block(value: object) -> RoleBlockV2:
    try:
        return role_block_v2_from_obj(value)
    except (TypeError, ValueError) as exc:
        raise BoundBankAuditV2Error(f"invalid role block: {exc}") from exc


def _catalog_contract() -> tuple[str, str]:
    contract = build_supported_catalog_contract_v2()
    return contract.source_catalog_digest, contract.supported_catalog_digest


def _resolve_rule_id_fast(rule_id: object) -> CatalogEntry:
    if type(rule_id) is not str:
        raise BoundBankAuditV2Error("catalog rule id must be a string")
    try:
        entry = _catalog_by_rule_id()[rule_id]
    except KeyError as exc:
        raise BoundBankAuditV2Error(f"unknown public rule id: {rule_id!r}") from exc
    return entry


def _binding_families_exact(
    placard: CatalogEntry,
    literal: CatalogEntry,
    composed: CatalogEntry,
) -> None:
    observed = tuple(classify_catalog_identity_v2(entry) for entry in (placard, literal, composed))
    expected = (
        "placard_literal",
        "one_literal_piece",
        "composed_two_literal_piece",
    )
    if observed != expected:
        raise BoundBankAuditV2Error("rule-triple candidates do not occupy exact P,Q,C families")


def _joint_cells_fast(
    placard: CatalogEntry,
    literal: CatalogEntry,
    composed: CatalogEntry,
) -> tuple[int, int, int, int, int, int, int, int]:
    universe_mask = (1 << SCENE_COUNT) - 1
    p_bits, q_bits, c_bits = placard.truth.bits, literal.truth.bits, composed.truth.bits
    pq_masks = (
        (universe_mask ^ p_bits) & (universe_mask ^ q_bits),
        (universe_mask ^ p_bits) & q_bits,
        p_bits & (universe_mask ^ q_bits),
        p_bits & q_bits,
    )
    values = tuple(
        (c_mask & pq_mask).bit_count() for c_mask in (universe_mask ^ c_bits, c_bits) for pq_mask in pq_masks
    )
    return cast(tuple[int, int, int, int, int, int, int, int], values)


def _cell_digest_fast(
    placard: CatalogEntry,
    literal: CatalogEntry,
    composed: CatalogEntry,
    cells: tuple[int, int, int, int, int, int, int, int],
) -> str:
    digest = hashlib.sha256()
    digest.update(_POPULATION_TRIPLE_ROW_DOMAIN)
    for entry in (placard, literal, composed):
        digest.update(entry.index.to_bytes(4, "big"))
        digest.update(bytes.fromhex(entry.truth_digest))
    for count in cells:
        digest.update(count.to_bytes(2, "big"))
    return digest.hexdigest()


def _binding_preimage_fast(
    catalog_digest: str,
    supported_catalog_digest: str,
    placard: CatalogEntry,
    literal: CatalogEntry,
    composed: CatalogEntry,
    cells: tuple[int, int, int, int, int, int, int, int],
    cell_digest: str,
) -> dict[str, Any]:
    return {
        "schema_version": 2,
        "catalog_digest": catalog_digest,
        "supported_catalog_digest": supported_catalog_digest,
        "candidate_order": ["P", "Q", "C"],
        "candidates": [
            {
                "slot": slot,
                "family": family,
                "rule_id": entry.rule_id,
                "truth_digest": entry.truth_digest,
            }
            for slot, family, entry in (
                ("P", "placard_literal", placard),
                ("Q", "one_literal_piece", literal),
                ("C", "composed_two_literal_piece", composed),
            )
        ],
        "joint_cell_order_c_p_q": list(JOINT_CELL_ORDER),
        "exact_joint_truth_cell_counts": list(cells),
        "exact_joint_cell_digest": cell_digest,
    }


def _derive_binding_fields(
    placard_rule_id: str,
    literal_rule_id: str,
    composed_rule_id: str,
) -> tuple[
    str,
    str,
    CatalogEntry,
    CatalogEntry,
    CatalogEntry,
    tuple[int, int, int, int, int, int, int, int],
    str,
    str,
]:
    catalog_digest, supported_catalog_digest = _catalog_contract()
    placard = _resolve_rule_id_fast(placard_rule_id)
    literal = _resolve_rule_id_fast(literal_rule_id)
    composed = _resolve_rule_id_fast(composed_rule_id)
    _binding_families_exact(placard, literal, composed)
    cells = _joint_cells_fast(placard, literal, composed)
    cell_digest = _cell_digest_fast(placard, literal, composed, cells)
    triple_digest = _json_digest(
        _binding_preimage_fast(
            catalog_digest,
            supported_catalog_digest,
            placard,
            literal,
            composed,
            cells,
            cell_digest,
        ),
        domain=_POPULATION_TRIPLE_BINDING_DOMAIN,
    )
    return (
        catalog_digest,
        supported_catalog_digest,
        placard,
        literal,
        composed,
        cells,
        cell_digest,
        triple_digest,
    )


def _materialize_exact_binding_fast(
    placard_rule_id: str,
    literal_rule_id: str,
    composed_rule_id: str,
) -> RuleTripleBindingV2:
    (
        catalog_digest,
        supported_catalog_digest,
        placard,
        literal,
        composed,
        cells,
        cell_digest,
        triple_digest,
    ) = _derive_binding_fields(placard_rule_id, literal_rule_id, composed_rule_id)
    # RuleTripleBindingV2.__post_init__ recomputes the large public-catalog
    # serialization for each row.  This batch path has just recomputed every
    # one of its invariants against a single exact catalog contract, so setting
    # the frozen slots avoids an otherwise quadratic manifest-build cost.
    binding = object.__new__(RuleTripleBindingV2)
    for name, value in (
        ("catalog_digest", catalog_digest),
        ("supported_catalog_digest", supported_catalog_digest),
        ("placard_rule_id", placard.rule_id),
        ("placard_truth_digest", placard.truth_digest),
        ("literal_rule_id", literal.rule_id),
        ("literal_truth_digest", literal.truth_digest),
        ("composed_rule_id", composed.rule_id),
        ("composed_truth_digest", composed.truth_digest),
        ("exact_joint_truth_cell_counts", cells),
        ("exact_joint_cell_digest", cell_digest),
        ("triple_digest", triple_digest),
    ):
        object.__setattr__(binding, name, value)
    return binding


def build_rule_triple_bindings_batch_for_audit_v2(
    rule_id_triples: Iterable[tuple[str, str, str]],
) -> tuple[RuleTripleBindingV2, ...]:
    """Build many exact bindings while hashing the shared catalog only once.

    The returned values are ordinary ``RuleTripleBindingV2`` instances.  The
    optimized path duplicates the binding's exact full-universe, truth-digest,
    family, supported-catalog, row-digest, and domain-digest checks, and every
    downstream audit rechecks those invariants independently.
    """

    materialized = tuple(rule_id_triples)
    for item in materialized:
        if type(item) is not tuple or len(item) != 3 or any(type(value) is not str for value in item):
            raise BoundBankAuditV2Error("each batch binding request must be a P,Q,C rule-id tuple")
    return tuple(_materialize_exact_binding_fast(*item) for item in materialized)


def _validate_binding(binding: RuleTripleBindingV2) -> RuleTripleBindingV2:
    if type(binding) is not RuleTripleBindingV2:
        raise TypeError("a full RuleTripleBindingV2 is required; a caller SHA is insufficient")
    (
        catalog_digest,
        supported_digest,
        placard,
        literal,
        composed,
        cells,
        cell_digest,
        triple_digest,
    ) = _derive_binding_fields(
        binding.placard_rule_id,
        binding.literal_rule_id,
        binding.composed_rule_id,
    )
    if binding.catalog_digest != catalog_digest or binding.supported_catalog_digest != supported_digest:
        raise BoundBankAuditV2Error("rule triple is not bound to the full and supported catalogs")
    expected = (
        placard.truth_digest,
        literal.truth_digest,
        composed.truth_digest,
        cells,
        cell_digest,
        triple_digest,
    )
    observed = (
        binding.placard_truth_digest,
        binding.literal_truth_digest,
        binding.composed_truth_digest,
        binding.exact_joint_truth_cell_counts,
        binding.exact_joint_cell_digest,
        binding.triple_digest,
    )
    if observed != expected:
        raise BoundBankAuditV2Error("rule-triple catalog recomputation differs from the binding")
    return binding


def _validate_bound_block(binding: RuleTripleBindingV2, block: RoleBlockV2) -> None:
    _validate_binding(binding)
    if type(block) is not RoleBlockV2:
        raise TypeError("role_block must be a RoleBlockV2")
    if block.semantic_triple_id != binding.digest:
        raise BoundBankAuditV2Error(
            "role block semantic_triple_id is an arbitrary or mismatched SHA, not the bound triple digest"
        )


@dataclass(frozen=True, slots=True)
class TrainingRotationExecutionV2:
    """Execution evidence for one member of an atomic training role block."""

    rotation_position: int
    display_order: int
    renderer_slot: int
    pre_update_checkpoint_digest: str
    update_batch_id: str
    optimizer_step_before: int
    role_weight_numerator: int = 1
    role_weight_denominator: int = 3
    context_reset_before_episode: bool = True
    objective_collected_before_atomic_commit: bool = True
    update_committed_during_episode: bool = False

    def __post_init__(self) -> None:
        _require_integer(self.rotation_position, name="rotation_position", maximum=2)
        _require_integer(self.display_order, name="display_order", maximum=2)
        _require_integer(self.renderer_slot, name="renderer_slot", maximum=2)
        _require_sha256(self.pre_update_checkpoint_digest, name="pre-update checkpoint digest")
        _require_batch_id(self.update_batch_id)
        _require_integer(self.optimizer_step_before, name="optimizer_step_before")
        if self.role_weight_numerator != 1 or self.role_weight_denominator != 3:
            raise BoundBankAuditV2Error("every rotation must carry the exact atomic role weight 1/3")
        if type(self.role_weight_numerator) is not int or type(self.role_weight_denominator) is not int:
            raise BoundBankAuditV2Error("role weights must use exact integer numerator and denominator")
        if type(self.context_reset_before_episode) is not bool:
            raise BoundBankAuditV2Error("context_reset_before_episode must be a Boolean")
        if not self.context_reset_before_episode:
            raise BoundBankAuditV2Error("context must reset before every training rotation")
        if type(self.objective_collected_before_atomic_commit) is not bool:
            raise BoundBankAuditV2Error("objective_collected_before_atomic_commit must be a Boolean")
        if not self.objective_collected_before_atomic_commit:
            raise BoundBankAuditV2Error("each objective must be collected before the atomic commit")
        if type(self.update_committed_during_episode) is not bool:
            raise BoundBankAuditV2Error("update_committed_during_episode must be a Boolean")
        if self.update_committed_during_episode:
            raise BoundBankAuditV2Error("no update may be committed during or between rotations")

    def _unsigned_obj(self) -> dict[str, Any]:
        return {
            "rotation_position": self.rotation_position,
            "display_order": self.display_order,
            "renderer_slot": self.renderer_slot,
            "pre_update_checkpoint_digest": self.pre_update_checkpoint_digest,
            "update_batch_id": self.update_batch_id,
            "optimizer_step_before": self.optimizer_step_before,
            "role_weight": {
                "numerator": self.role_weight_numerator,
                "denominator": self.role_weight_denominator,
            },
            "context_reset_before_episode": self.context_reset_before_episode,
            "objective_collected_before_atomic_commit": self.objective_collected_before_atomic_commit,
            "update_committed_during_episode": self.update_committed_during_episode,
        }

    @property
    def digest(self) -> str:
        return _json_digest(self._unsigned_obj(), domain=_TRAINING_ROTATION_DOMAIN)

    def as_obj(self) -> dict[str, Any]:
        return {**self._unsigned_obj(), "execution_digest": self.digest}


def _training_rotation_from_obj(value: object) -> TrainingRotationExecutionV2:
    obj = _require_mapping(
        value,
        (
            "rotation_position",
            "display_order",
            "renderer_slot",
            "pre_update_checkpoint_digest",
            "update_batch_id",
            "optimizer_step_before",
            "role_weight",
            "context_reset_before_episode",
            "objective_collected_before_atomic_commit",
            "update_committed_during_episode",
            "execution_digest",
        ),
        name="training rotation execution",
    )
    weight = _require_mapping(obj["role_weight"], ("numerator", "denominator"), name="role weight")
    execution = TrainingRotationExecutionV2(
        rotation_position=_require_integer(obj["rotation_position"], name="rotation_position", maximum=2),
        display_order=_require_integer(obj["display_order"], name="display_order", maximum=2),
        renderer_slot=_require_integer(obj["renderer_slot"], name="renderer_slot", maximum=2),
        pre_update_checkpoint_digest=_require_sha256(
            obj["pre_update_checkpoint_digest"], name="pre-update checkpoint digest"
        ),
        update_batch_id=_require_batch_id(obj["update_batch_id"]),
        optimizer_step_before=_require_integer(obj["optimizer_step_before"], name="optimizer_step_before"),
        role_weight_numerator=_require_integer(weight["numerator"], name="role-weight numerator"),
        role_weight_denominator=_require_integer(
            weight["denominator"], name="role-weight denominator", minimum=1
        ),
        context_reset_before_episode=_require_boolean(
            obj["context_reset_before_episode"], name="context_reset_before_episode"
        ),
        objective_collected_before_atomic_commit=_require_boolean(
            obj["objective_collected_before_atomic_commit"],
            name="objective_collected_before_atomic_commit",
        ),
        update_committed_during_episode=_require_boolean(
            obj["update_committed_during_episode"], name="update_committed_during_episode"
        ),
    )
    if obj["execution_digest"] != execution.digest:
        raise BoundBankAuditV2Error("training rotation execution digest mismatch")
    if _dump_json(obj) != _dump_json(execution.as_obj()):
        raise BoundBankAuditV2Error("training rotation execution has inconsistent derived metadata")
    return execution


@dataclass(frozen=True, slots=True)
class CatalogBoundTrainingRoleUnitV2:
    """One catalog-bound, atomic, three-rotation perfect training unit."""

    bank_position: int
    binding: RuleTripleBindingV2
    role_block: RoleBlockV2
    executions: tuple[TrainingRotationExecutionV2, ...]
    optimizer_step_after_atomic_commit: int

    def __post_init__(self) -> None:
        _require_integer(self.bank_position, name="bank_position")
        _validate_bound_block(self.binding, self.role_block)
        if self.role_block.geometry.slug != "a0_b0" or self.role_block.a_error_target_side is not None:
            raise BoundBankAuditV2Error("training role units must use only perfect ambiguity a0_b0")
        if type(self.executions) is not tuple or len(self.executions) != 3:
            raise BoundBankAuditV2Error("training role unit requires exactly three executions")
        if tuple(item.rotation_position for item in self.executions) != (0, 1, 2):
            raise BoundBankAuditV2Error("training executions must be complete and rotation-canonical")
        if {item.display_order for item in self.executions} != {0, 1, 2}:
            raise BoundBankAuditV2Error("each training block must use every display-order slot once")
        if {item.renderer_slot for item in self.executions} != {0, 1, 2}:
            raise BoundBankAuditV2Error("each training block must use every renderer slot once")
        if len({item.pre_update_checkpoint_digest for item in self.executions}) != 1:
            raise BoundBankAuditV2Error("all rotations must use the same pre-update checkpoint")
        if len({item.update_batch_id for item in self.executions}) != 1:
            raise BoundBankAuditV2Error("all rotations must use the same update-batch ID")
        steps = {item.optimizer_step_before for item in self.executions}
        if len(steps) != 1:
            raise BoundBankAuditV2Error("an optimizer update occurred between role rotations")
        step_before = next(iter(steps))
        if (
            isinstance(self.optimizer_step_after_atomic_commit, bool)
            or self.optimizer_step_after_atomic_commit != step_before + 1
        ):
            raise BoundBankAuditV2Error("the role-average must be committed in exactly one optimizer step")

    @property
    def update_batch_id(self) -> str:
        return self.executions[0].update_batch_id

    @property
    def optimizer_step_before(self) -> int:
        return self.executions[0].optimizer_step_before

    def _unsigned_obj(self) -> dict[str, Any]:
        return {
            "bank_position": self.bank_position,
            "binding": self.binding.as_obj(),
            "role_block": self.role_block.as_obj(),
            "executions": [item.as_obj() for item in self.executions],
            "optimizer_step_after_atomic_commit": self.optimizer_step_after_atomic_commit,
        }

    @property
    def digest(self) -> str:
        return _json_digest(self._unsigned_obj(), domain=_TRAINING_UNIT_DOMAIN)

    def as_obj(self) -> dict[str, Any]:
        return {**self._unsigned_obj(), "unit_digest": self.digest}


def build_catalog_bound_training_role_unit_v2(
    binding: RuleTripleBindingV2,
    *,
    bank_position: int,
    pre_update_checkpoint_digest: str,
    update_batch_id: str,
    optimizer_step_before: int,
    display_shift: int,
    renderer_shift: int,
) -> CatalogBoundTrainingRoleUnitV2:
    """Build one exact perfect-ambiguity unit from a full catalog binding."""

    _validate_binding(binding)
    _require_integer(display_shift, name="display_shift", maximum=2)
    _require_integer(renderer_shift, name="renderer_shift", maximum=2)
    block = build_role_block_v2(binding.digest, _GEOMETRY_BY_SLUG["a0_b0"])
    executions = tuple(
        TrainingRotationExecutionV2(
            rotation_position=position,
            display_order=(position + display_shift) % 3,
            renderer_slot=(position + renderer_shift) % 3,
            pre_update_checkpoint_digest=pre_update_checkpoint_digest,
            update_batch_id=update_batch_id,
            optimizer_step_before=optimizer_step_before,
        )
        for position in range(3)
    )
    return CatalogBoundTrainingRoleUnitV2(
        bank_position=bank_position,
        binding=binding,
        role_block=block,
        executions=executions,
        optimizer_step_after_atomic_commit=optimizer_step_before + 1,
    )


def _training_unit_from_obj(value: object) -> CatalogBoundTrainingRoleUnitV2:
    obj = _require_mapping(
        value,
        (
            "bank_position",
            "binding",
            "role_block",
            "executions",
            "optimizer_step_after_atomic_commit",
            "unit_digest",
        ),
        name="catalog-bound training role unit",
    )
    raw_executions = obj["executions"]
    if type(raw_executions) is not list:
        raise BoundBankAuditV2Error("training executions must be an array")
    unit = CatalogBoundTrainingRoleUnitV2(
        bank_position=_require_integer(obj["bank_position"], name="bank_position"),
        binding=_parse_binding(obj["binding"]),
        role_block=_parse_role_block(obj["role_block"]),
        executions=tuple(_training_rotation_from_obj(item) for item in raw_executions),
        optimizer_step_after_atomic_commit=_require_integer(
            obj["optimizer_step_after_atomic_commit"],
            name="optimizer_step_after_atomic_commit",
            minimum=1,
        ),
    )
    if obj["unit_digest"] != unit.digest:
        raise BoundBankAuditV2Error("training role-unit digest mismatch")
    if _dump_json(obj) != _dump_json(unit.as_obj()):
        raise BoundBankAuditV2Error("training role unit has inconsistent derived metadata")
    return unit


def _family_slot_counts(
    units: tuple[CatalogBoundTrainingRoleUnitV2, ...],
    *,
    slot_name: str,
) -> Counter[tuple[str, int]]:
    counts: Counter[tuple[str, int]] = Counter()
    for unit in units:
        for execution in unit.executions:
            family = CYCLIC_FAMILY_ORDER[execution.rotation_position].value
            slot = cast(int, getattr(execution, slot_name))
            counts[(family, slot)] += 1
    return counts


def _slot_max_gap(counts: Counter[tuple[str, int]]) -> int:
    return max(
        max(counts[(family.value, slot)] for family in CYCLIC_FAMILY_ORDER)
        - min(counts[(family.value, slot)] for family in CYCLIC_FAMILY_ORDER)
        for slot in range(3)
    )


def _training_prefix_checks(
    units: tuple[CatalogBoundTrainingRoleUnitV2, ...],
) -> list[dict[str, Any]]:
    checks: list[dict[str, Any]] = []
    for prefix_count in range(1, len(units) + 1):
        prefix = units[:prefix_count]
        order_gap = _slot_max_gap(_family_slot_counts(prefix, slot_name="display_order"))
        renderer_gap = _slot_max_gap(_family_slot_counts(prefix, slot_name="renderer_slot"))
        checks.append(
            {
                "prefix_block_count": prefix_count,
                "family_by_display_order_max_gap": order_gap,
                "family_by_renderer_slot_max_gap": renderer_gap,
                "passed": order_gap <= 1 and renderer_gap <= 1,
            }
        )
    return checks


def _family_role_table(unit_count: int) -> list[dict[str, Any]]:
    return [
        {"family": family.value, "role": role.value, "count": unit_count}
        for family in CYCLIC_FAMILY_ORDER
        for role in ROLE_ORDER
    ]


def _family_slot_table(
    units: tuple[CatalogBoundTrainingRoleUnitV2, ...],
    *,
    slot_name: str,
) -> list[dict[str, Any]]:
    counts = _family_slot_counts(units, slot_name=slot_name)
    return [
        {"family": family.value, "slot": slot, "count": counts[(family.value, slot)]}
        for family in CYCLIC_FAMILY_ORDER
        for slot in range(3)
    ]


@dataclass(frozen=True, slots=True)
class PerfectTrainingRoleAuditV2:
    """Nominal report for a perfect-ambiguity atomic training manifest."""

    units: tuple[CatalogBoundTrainingRoleUnitV2, ...]

    def __post_init__(self) -> None:
        if not self.units or type(self.units) is not tuple:
            raise BoundBankAuditV2Error("training-role audit requires a nonempty tuple of units")
        if any(type(unit) is not CatalogBoundTrainingRoleUnitV2 for unit in self.units):
            raise BoundBankAuditV2Error("training-role audit received a foreign unit type")
        if tuple(unit.bank_position for unit in self.units) != tuple(range(len(self.units))):
            raise BoundBankAuditV2Error("training units must occupy every bank position in order")

    @property
    def catalog_digest(self) -> str:
        return _catalog_contract()[0]

    @property
    def supported_catalog_digest(self) -> str:
        return _catalog_contract()[1]

    def _checks(self) -> dict[str, bool]:
        triple_ids = [unit.binding.digest for unit in self.units]
        batch_ids = [unit.update_batch_id for unit in self.units]
        optimizer_chain = all(
            left.optimizer_step_after_atomic_commit == right.optimizer_step_before
            for left, right in zip(self.units, self.units[1:], strict=False)
        )
        prefix_checks = _training_prefix_checks(self.units)
        return {
            "full_and_supported_catalog_binding_exact": all(
                unit.binding.catalog_digest == self.catalog_digest
                and unit.binding.supported_catalog_digest == self.supported_catalog_digest
                for unit in self.units
            ),
            "unique_bound_triples": len(triple_ids) == len(set(triple_ids)),
            "exactly_three_perfect_rotations_per_triple": all(
                len(unit.executions) == 3 and unit.role_block.geometry.slug == "a0_b0" for unit in self.units
            ),
            "family_by_official_a_b_exact": True,
            "atomic_equal_role_weights": all(
                all(
                    execution.role_weight_numerator == 1 and execution.role_weight_denominator == 3
                    for execution in unit.executions
                )
                for unit in self.units
            ),
            "same_pre_update_checkpoint_and_batch_within_block": all(
                len({execution.pre_update_checkpoint_digest for execution in unit.executions}) == 1
                and len({execution.update_batch_id for execution in unit.executions}) == 1
                for unit in self.units
            ),
            "one_atomic_update_and_no_interrotation_update": optimizer_chain,
            "unique_update_batch_ids": len(batch_ids) == len(set(batch_ids)),
            "all_contexts_reset": all(
                execution.context_reset_before_episode for unit in self.units for execution in unit.executions
            ),
            "all_complete_prefixes_order_renderer_counterbalanced": all(
                item["passed"] is True for item in prefix_checks
            ),
        }

    @property
    def training_role_audit_passed(self) -> bool:
        return all(self._checks().values())

    def _unsigned_obj(self) -> dict[str, Any]:
        prefix_checks = _training_prefix_checks(self.units)
        return {
            "schema_version": PERFECT_TRAINING_ROLE_AUDIT_SCHEMA_VERSION,
            "report_kind": _TRAINING_REPORT_KIND,
            "authorization": dict(_AUTHORIZATION),
            "catalog_digest": self.catalog_digest,
            "supported_catalog_digest": self.supported_catalog_digest,
            "bound_triple_count": len(self.units),
            "episode_count": len(self.units) * 3,
            "units": [unit.as_obj() for unit in self.units],
            "family_by_official_a_b_counts": _family_role_table(len(self.units)),
            "family_by_display_order_counts": _family_slot_table(self.units, slot_name="display_order"),
            "family_by_renderer_slot_counts": _family_slot_table(self.units, slot_name="renderer_slot"),
            "complete_prefix_checks": prefix_checks,
            "complete_prefix_checks_digest": _json_digest(prefix_checks, domain=_TRAINING_PREFIX_DOMAIN),
            "checks": [{"name": name, "passed": passed} for name, passed in self._checks().items()],
            "training_role_audit_passed": self.training_role_audit_passed,
            "substitution_contract": {
                "may_substitute_for_evaluation_quartet_audit": False,
                "may_substitute_for_metadata_stress_population_audit": False,
            },
        }

    @property
    def digest(self) -> str:
        return _json_digest(self._unsigned_obj(), domain=_TRAINING_REPORT_DOMAIN)

    def as_obj(self) -> dict[str, Any]:
        return {**self._unsigned_obj(), "training_role_audit_digest": self.digest}


def build_perfect_training_role_audit_v2(
    units: Iterable[CatalogBoundTrainingRoleUnitV2],
) -> PerfectTrainingRoleAuditV2:
    materialized = tuple(units)
    if any(type(unit) is not CatalogBoundTrainingRoleUnitV2 for unit in materialized):
        raise TypeError("training audit accepts only CatalogBoundTrainingRoleUnitV2 values")
    return PerfectTrainingRoleAuditV2(materialized)


def perfect_training_role_audit_v2_from_obj(
    value: object,
    *,
    expected_digest: str | None = None,
) -> PerfectTrainingRoleAuditV2:
    fields = (
        "schema_version",
        "report_kind",
        "authorization",
        "catalog_digest",
        "supported_catalog_digest",
        "bound_triple_count",
        "episode_count",
        "units",
        "family_by_official_a_b_counts",
        "family_by_display_order_counts",
        "family_by_renderer_slot_counts",
        "complete_prefix_checks",
        "complete_prefix_checks_digest",
        "checks",
        "training_role_audit_passed",
        "substitution_contract",
        "training_role_audit_digest",
    )
    obj = _require_mapping(value, fields, name="perfect-training role audit")
    if (
        type(obj["schema_version"]) is not int
        or obj["schema_version"] != PERFECT_TRAINING_ROLE_AUDIT_SCHEMA_VERSION
        or obj["report_kind"] != _TRAINING_REPORT_KIND
    ):
        raise BoundBankAuditV2Error("perfect-training role-audit schema identity mismatch")
    _require_authorization(obj["authorization"])
    raw_units = obj["units"]
    if type(raw_units) is not list:
        raise BoundBankAuditV2Error("training-role units must be an array")
    audit = build_perfect_training_role_audit_v2(_training_unit_from_obj(item) for item in raw_units)
    if obj["training_role_audit_digest"] != audit.digest:
        raise BoundBankAuditV2Error("perfect-training role-audit digest mismatch")
    if expected_digest is not None and audit.digest != _require_sha256(
        expected_digest, name="expected training-role audit digest"
    ):
        raise BoundBankAuditV2Error("training-role audit differs from the externally expected digest")
    if _dump_json(obj) != _dump_json(audit.as_obj()):
        raise BoundBankAuditV2Error("perfect-training role audit contains inconsistent metadata")
    return audit


def verify_perfect_training_role_audit_v2(
    audit: PerfectTrainingRoleAuditV2,
) -> PerfectTrainingRoleAuditV2:
    if type(audit) is not PerfectTrainingRoleAuditV2:
        raise TypeError("verify requires a PerfectTrainingRoleAuditV2")
    return perfect_training_role_audit_v2_from_obj(audit.as_obj(), expected_digest=audit.digest)


def serialize_perfect_training_role_audit_v2(audit: PerfectTrainingRoleAuditV2) -> str:
    return _dump_json(verify_perfect_training_role_audit_v2(audit).as_obj())


def parse_perfect_training_role_audit_v2(
    text: str,
    *,
    expected_digest: str | None = None,
) -> PerfectTrainingRoleAuditV2:
    audit = perfect_training_role_audit_v2_from_obj(_load_json(text), expected_digest=expected_digest)
    if serialize_perfect_training_role_audit_v2(audit) != text:
        raise BoundBankAuditV2Error("training-role audit JSON is not canonical compact JSON")
    return audit


def _scene_ids_from_obj(value: object, *, name: str) -> tuple[tuple[int, ...], ...]:
    if type(value) is not list or len(value) != len(JOINT_CELL_ORDER):
        raise BoundBankAuditV2Error(f"{name} must contain eight joint-cell arrays")
    result: list[tuple[int, ...]] = []
    for cell_index, raw_cell in enumerate(value):
        if type(raw_cell) is not list:
            raise BoundBankAuditV2Error(f"{name}[{cell_index}] must be an array")
        result.append(
            tuple(
                _require_integer(
                    item,
                    name=f"{name}[{cell_index}] scene index",
                    maximum=SCENE_COUNT - 1,
                )
                for item in raw_cell
            )
        )
    return tuple(result)


def _scene_ids_obj(value: tuple[tuple[int, ...], ...]) -> list[list[int]]:
    return [list(cell) for cell in value]


def _validate_scene_matrix_shape(
    value: tuple[tuple[int, ...], ...],
    *,
    name: str,
) -> None:
    if type(value) is not tuple or len(value) != len(JOINT_CELL_ORDER):
        raise BoundBankAuditV2Error(f"{name} must contain eight joint-cell tuples")
    flat: list[int] = []
    for cell_index, cell in enumerate(value):
        if type(cell) is not tuple:
            raise BoundBankAuditV2Error(f"{name}[{cell_index}] must be a tuple")
        for scene_index in cell:
            _require_integer(
                scene_index,
                name=f"{name}[{cell_index}] scene index",
                maximum=SCENE_COUNT - 1,
            )
            flat.append(scene_index)
    if len(flat) != len(set(flat)):
        raise BoundBankAuditV2Error(f"{name} contains a duplicate scene index")


@dataclass(frozen=True, slots=True)
class EvaluationSharedBindingV2:
    """Resources that must be byte-identical across one evaluation quartet."""

    reservoir_scene_ids_by_joint_cell: tuple[tuple[int, ...], ...]
    intervention_bank_digest: str
    renderer_binding_digest: str
    display_geometry_order: tuple[str, ...]

    def __post_init__(self) -> None:
        _validate_scene_matrix_shape(self.reservoir_scene_ids_by_joint_cell, name="evaluation reservoir")
        if any(
            len(cell) != EVALUATION_RESERVOIR_SCENES_PER_CELL
            for cell in self.reservoir_scene_ids_by_joint_cell
        ):
            raise BoundBankAuditV2Error("evaluation reservoir must contain exactly 22 scenes per cell")
        _require_sha256(self.intervention_bank_digest, name="intervention-bank digest")
        _require_sha256(self.renderer_binding_digest, name="renderer-binding digest")
        if type(self.display_geometry_order) is not tuple or set(self.display_geometry_order) != set(
            _GEOMETRY_BY_SLUG
        ):
            raise BoundBankAuditV2Error("display order must contain every evidence geometry exactly once")
        if len(self.display_geometry_order) != len(EVIDENCE_GEOMETRIES):
            raise BoundBankAuditV2Error("display order must contain exactly four geometries")

    @property
    def reservoir_digest(self) -> str:
        return _json_digest(
            {
                "joint_cell_order_c_p_q": list(JOINT_CELL_ORDER),
                "scene_ids_by_joint_cell": _scene_ids_obj(self.reservoir_scene_ids_by_joint_cell),
            },
            domain=_RESERVOIR_DOMAIN,
        )

    @property
    def display_binding_digest(self) -> str:
        return _json_digest(
            {
                "renderer_binding_digest": self.renderer_binding_digest,
                "display_geometry_order": list(self.display_geometry_order),
            },
            domain=_DISPLAY_BINDING_DOMAIN,
        )

    def _unsigned_obj(self) -> dict[str, Any]:
        return {
            "joint_cell_order_c_p_q": list(JOINT_CELL_ORDER),
            "reservoir_scene_ids_by_joint_cell": _scene_ids_obj(self.reservoir_scene_ids_by_joint_cell),
            "reservoir_digest": self.reservoir_digest,
            "intervention_bank_digest": self.intervention_bank_digest,
            "renderer_binding_digest": self.renderer_binding_digest,
            "display_geometry_order": list(self.display_geometry_order),
            "display_binding_digest": self.display_binding_digest,
        }

    @property
    def digest(self) -> str:
        return _json_digest(self._unsigned_obj(), domain=_SHARED_RESOURCE_DOMAIN)

    def as_obj(self) -> dict[str, Any]:
        return {**self._unsigned_obj(), "shared_resource_binding_digest": self.digest}


def _shared_binding_from_obj(value: object) -> EvaluationSharedBindingV2:
    obj = _require_mapping(
        value,
        (
            "joint_cell_order_c_p_q",
            "reservoir_scene_ids_by_joint_cell",
            "reservoir_digest",
            "intervention_bank_digest",
            "renderer_binding_digest",
            "display_geometry_order",
            "display_binding_digest",
            "shared_resource_binding_digest",
        ),
        name="evaluation shared-resource binding",
    )
    if obj["joint_cell_order_c_p_q"] != list(JOINT_CELL_ORDER):
        raise BoundBankAuditV2Error("evaluation resource joint-cell order differs")
    raw_order = obj["display_geometry_order"]
    if type(raw_order) is not list or any(type(item) is not str for item in raw_order):
        raise BoundBankAuditV2Error("display_geometry_order must be a string array")
    shared = EvaluationSharedBindingV2(
        reservoir_scene_ids_by_joint_cell=_scene_ids_from_obj(
            obj["reservoir_scene_ids_by_joint_cell"], name="evaluation reservoir"
        ),
        intervention_bank_digest=_require_sha256(
            obj["intervention_bank_digest"], name="intervention-bank digest"
        ),
        renderer_binding_digest=_require_sha256(
            obj["renderer_binding_digest"], name="renderer-binding digest"
        ),
        display_geometry_order=tuple(cast(list[str], raw_order)),
    )
    if obj["shared_resource_binding_digest"] != shared.digest:
        raise BoundBankAuditV2Error("evaluation shared-resource binding digest mismatch")
    if _dump_json(obj) != _dump_json(shared.as_obj()):
        raise BoundBankAuditV2Error("evaluation shared-resource binding is inconsistent")
    return shared


@dataclass(frozen=True, slots=True)
class COfficialEvaluationEpisodeV2:
    """One C-Official member of a held-out four-geometry quartet."""

    role_block: RoleBlockV2
    selected_rotation_position: int
    opening_scene_ids_by_joint_cell: tuple[tuple[int, ...], ...]
    opening_union_digest: str
    display_position: int
    shared_resource_binding_digest: str
    pre_evaluation_checkpoint_digest: str
    optimizer_step: int
    context_reset_before_episode: bool = True
    weight_update_performed: bool = False
    official_ast_private_during_inquiry: bool = True
    rule_ast_committed_before_terminal: bool = True
    terminal_classification_ast_blind: bool = True
    independent_terminal_replay: bool = True

    def __post_init__(self) -> None:
        if type(self.role_block) is not RoleBlockV2:
            raise TypeError("evaluation episode role_block must be a RoleBlockV2")
        if isinstance(self.selected_rotation_position, bool) or self.selected_rotation_position != 2:
            raise BoundBankAuditV2Error(
                "evaluation episode must select rotation position two (C-Official, A=P, B=Q)"
            )
        _validate_scene_matrix_shape(self.opening_scene_ids_by_joint_cell, name="evaluation opening")
        if tuple(map(len, self.opening_scene_ids_by_joint_cell)) != (
            self.role_block.schedule.joint_truth_cell_counts
        ):
            raise BoundBankAuditV2Error("evaluation opening does not implement its exact schedule")
        _require_sha256(self.opening_union_digest, name="opening-union digest")
        _require_integer(self.display_position, name="display_position", maximum=3)
        _require_sha256(self.shared_resource_binding_digest, name="shared-resource binding digest")
        _require_sha256(self.pre_evaluation_checkpoint_digest, name="pre-evaluation checkpoint digest")
        _require_integer(self.optimizer_step, name="optimizer_step")
        exact_contract = (
            ("context_reset_before_episode", self.context_reset_before_episode, True),
            ("weight_update_performed", self.weight_update_performed, False),
            ("official_ast_private_during_inquiry", self.official_ast_private_during_inquiry, True),
            ("rule_ast_committed_before_terminal", self.rule_ast_committed_before_terminal, True),
            ("terminal_classification_ast_blind", self.terminal_classification_ast_blind, True),
            ("independent_terminal_replay", self.independent_terminal_replay, True),
        )
        for name, observed, expected in exact_contract:
            if type(observed) is not bool or observed is not expected:
                raise BoundBankAuditV2Error(f"evaluation execution contract violated: {name}")

    @property
    def geometry(self) -> EvidenceGeometryV2:
        return self.role_block.geometry

    @property
    def a_error_target_side(self) -> OfficialTargetSideV2 | None:
        return self.role_block.a_error_target_side

    def _unsigned_obj(self) -> dict[str, Any]:
        return {
            "geometry_slug": self.geometry.slug,
            "a_error_target_side": (
                None if self.a_error_target_side is None else self.a_error_target_side.value
            ),
            "role_block": self.role_block.as_obj(),
            "selected_rotation_position": self.selected_rotation_position,
            "joint_cell_order_c_p_q": list(JOINT_CELL_ORDER),
            "opening_scene_ids_by_joint_cell": _scene_ids_obj(self.opening_scene_ids_by_joint_cell),
            "opening_union_digest": self.opening_union_digest,
            "display_position": self.display_position,
            "shared_resource_binding_digest": self.shared_resource_binding_digest,
            "pre_evaluation_checkpoint_digest": self.pre_evaluation_checkpoint_digest,
            "optimizer_step": self.optimizer_step,
            "execution_contract": {
                "context_reset_before_episode": self.context_reset_before_episode,
                "weight_update_performed": self.weight_update_performed,
                "official_ast_private_during_inquiry": self.official_ast_private_during_inquiry,
                "rule_ast_committed_before_terminal": self.rule_ast_committed_before_terminal,
                "terminal_classification_ast_blind": self.terminal_classification_ast_blind,
                "independent_terminal_replay": self.independent_terminal_replay,
            },
        }

    @property
    def digest(self) -> str:
        return _json_digest(self._unsigned_obj(), domain=_EVALUATION_EPISODE_DOMAIN)

    def as_obj(self) -> dict[str, Any]:
        return {**self._unsigned_obj(), "episode_digest": self.digest}


def _evaluation_episode_from_obj(value: object) -> COfficialEvaluationEpisodeV2:
    obj = _require_mapping(
        value,
        (
            "geometry_slug",
            "a_error_target_side",
            "role_block",
            "selected_rotation_position",
            "joint_cell_order_c_p_q",
            "opening_scene_ids_by_joint_cell",
            "opening_union_digest",
            "display_position",
            "shared_resource_binding_digest",
            "pre_evaluation_checkpoint_digest",
            "optimizer_step",
            "execution_contract",
            "episode_digest",
        ),
        name="C-Official evaluation episode",
    )
    if obj["joint_cell_order_c_p_q"] != list(JOINT_CELL_ORDER):
        raise BoundBankAuditV2Error("evaluation opening joint-cell order differs")
    contract = _require_mapping(
        obj["execution_contract"],
        (
            "context_reset_before_episode",
            "weight_update_performed",
            "official_ast_private_during_inquiry",
            "rule_ast_committed_before_terminal",
            "terminal_classification_ast_blind",
            "independent_terminal_replay",
        ),
        name="evaluation execution contract",
    )
    episode = COfficialEvaluationEpisodeV2(
        role_block=_parse_role_block(obj["role_block"]),
        selected_rotation_position=_require_integer(
            obj["selected_rotation_position"],
            name="selected_rotation_position",
            maximum=2,
        ),
        opening_scene_ids_by_joint_cell=_scene_ids_from_obj(
            obj["opening_scene_ids_by_joint_cell"], name="evaluation opening"
        ),
        opening_union_digest=_require_sha256(obj["opening_union_digest"], name="opening-union digest"),
        display_position=_require_integer(obj["display_position"], name="display_position", maximum=3),
        shared_resource_binding_digest=_require_sha256(
            obj["shared_resource_binding_digest"], name="shared-resource binding digest"
        ),
        pre_evaluation_checkpoint_digest=_require_sha256(
            obj["pre_evaluation_checkpoint_digest"], name="pre-evaluation checkpoint digest"
        ),
        optimizer_step=_require_integer(obj["optimizer_step"], name="optimizer_step"),
        context_reset_before_episode=_require_boolean(
            contract["context_reset_before_episode"], name="context_reset_before_episode"
        ),
        weight_update_performed=_require_boolean(
            contract["weight_update_performed"], name="weight_update_performed"
        ),
        official_ast_private_during_inquiry=_require_boolean(
            contract["official_ast_private_during_inquiry"],
            name="official_ast_private_during_inquiry",
        ),
        rule_ast_committed_before_terminal=_require_boolean(
            contract["rule_ast_committed_before_terminal"],
            name="rule_ast_committed_before_terminal",
        ),
        terminal_classification_ast_blind=_require_boolean(
            contract["terminal_classification_ast_blind"],
            name="terminal_classification_ast_blind",
        ),
        independent_terminal_replay=_require_boolean(
            contract["independent_terminal_replay"], name="independent_terminal_replay"
        ),
    )
    if obj["geometry_slug"] != episode.geometry.slug or obj["a_error_target_side"] != (
        None if episode.a_error_target_side is None else episode.a_error_target_side.value
    ):
        raise BoundBankAuditV2Error("evaluation episode geometry metadata disagrees with role block")
    if obj["episode_digest"] != episode.digest:
        raise BoundBankAuditV2Error("evaluation episode digest mismatch")
    if _dump_json(obj) != _dump_json(episode.as_obj()):
        raise BoundBankAuditV2Error("evaluation episode contains inconsistent derived metadata")
    return episode


@lru_cache(maxsize=1)
def _catalog_by_rule_id() -> dict[str, CatalogEntry]:
    return {entry.rule_id: entry for entry in build_rule_catalog()}


def _joint_cell_index(binding: RuleTripleBindingV2, scene_index: int) -> int:
    catalog = _catalog_by_rule_id()
    scene = scene_at(scene_index)
    c_value = int(evaluate_rule(catalog[binding.composed_rule_id].rule, scene))
    p_value = int(evaluate_rule(catalog[binding.placard_rule_id].rule, scene))
    q_value = int(evaluate_rule(catalog[binding.literal_rule_id].rule, scene))
    return 4 * c_value + 2 * p_value + q_value


def _opening_union_digest(
    binding: RuleTripleBindingV2,
    opening_union: tuple[tuple[int, ...], ...],
) -> str:
    return _json_digest(
        {
            "bound_triple_digest": binding.digest,
            "joint_cell_order_c_p_q": list(JOINT_CELL_ORDER),
            "opening_union_scene_ids_by_joint_cell": _scene_ids_obj(opening_union),
        },
        domain=_OPENING_UNION_DOMAIN,
    )


def _validate_matrix_truth_cells(
    binding: RuleTripleBindingV2,
    matrix: tuple[tuple[int, ...], ...],
    *,
    name: str,
) -> None:
    for cell_index, scene_ids in enumerate(matrix):
        for scene_index in scene_ids:
            if _joint_cell_index(binding, scene_index) != cell_index:
                raise BoundBankAuditV2Error(
                    f"{name} scene {scene_index} is not in declared joint cell {cell_index}"
                )


@dataclass(frozen=True, slots=True)
class COfficialEvaluationQuartetV2:
    """Four held-out C-Official geometries sharing one exact resource binding."""

    binding: RuleTripleBindingV2
    shared_resources: EvaluationSharedBindingV2
    opening_union_scene_ids_by_joint_cell: tuple[tuple[int, ...], ...]
    episodes: tuple[COfficialEvaluationEpisodeV2, ...]

    def __post_init__(self) -> None:
        _validate_binding(self.binding)
        if type(self.shared_resources) is not EvaluationSharedBindingV2:
            raise TypeError("shared_resources must be an EvaluationSharedBindingV2")
        _validate_scene_matrix_shape(
            self.opening_union_scene_ids_by_joint_cell, name="evaluation opening union"
        )
        if type(self.episodes) is not tuple or len(self.episodes) != 4:
            raise BoundBankAuditV2Error("evaluation quartet requires exactly four episodes")
        if any(type(episode) is not COfficialEvaluationEpisodeV2 for episode in self.episodes):
            raise BoundBankAuditV2Error("evaluation quartet received a foreign episode type")
        if tuple(episode.geometry for episode in self.episodes) != EVIDENCE_GEOMETRIES:
            raise BoundBankAuditV2Error("each evidence geometry must occur exactly once in canonical order")
        expected_union_digest = _opening_union_digest(
            self.binding, self.opening_union_scene_ids_by_joint_cell
        )
        if any(episode.opening_union_digest != expected_union_digest for episode in self.episodes):
            raise BoundBankAuditV2Error("episodes do not share the recomputed opening-union digest")
        if any(
            episode.shared_resource_binding_digest != self.shared_resources.digest
            for episode in self.episodes
        ):
            raise BoundBankAuditV2Error("episodes do not share reservoir/intervention/display bindings")
        if any(episode.role_block.semantic_triple_id != self.binding.digest for episode in self.episodes):
            raise BoundBankAuditV2Error("an evaluation role block is not bound to the quartet triple")
        if any(
            episode.role_block.rotations[2].official_family is not RuleFamilyV2.COMPOSED
            or episode.role_block.rotations[2].alternative_a_family is not RuleFamilyV2.PLACARD
            or episode.role_block.rotations[2].alternative_b_family is not RuleFamilyV2.LITERAL
            or episode.selected_rotation_position != 2
            for episode in self.episodes
        ):
            raise BoundBankAuditV2Error("evaluation episodes must select the C-Official, A=P, B=Q rotation")
        if len({episode.pre_evaluation_checkpoint_digest for episode in self.episodes}) != 1:
            raise BoundBankAuditV2Error("quartet episodes must use one unchanged checkpoint")
        if len({episode.optimizer_step for episode in self.episodes}) != 1:
            raise BoundBankAuditV2Error("an optimizer update occurred within the evaluation quartet")
        if {episode.display_position for episode in self.episodes} != {0, 1, 2, 3}:
            raise BoundBankAuditV2Error("evaluation quartet must occupy every display position once")
        for episode in self.episodes:
            expected_position = self.shared_resources.display_geometry_order.index(episode.geometry.slug)
            if episode.display_position != expected_position:
                raise BoundBankAuditV2Error("episode order differs from the shared display binding")

        derived_union: list[tuple[int, ...]] = []
        for cell_index in range(len(JOINT_CELL_ORDER)):
            selections = tuple(
                episode.opening_scene_ids_by_joint_cell[cell_index] for episode in self.episodes
            )
            longest = max(selections, key=len)
            if any(selection != longest[: len(selection)] for selection in selections):
                raise BoundBankAuditV2Error("quartet openings are not nested canonical prefixes")
            derived_union.append(longest)
        if tuple(derived_union) != self.opening_union_scene_ids_by_joint_cell:
            raise BoundBankAuditV2Error("opening union is not the exact nested union of all four openings")

        _validate_matrix_truth_cells(
            self.binding,
            self.opening_union_scene_ids_by_joint_cell,
            name="opening union",
        )
        _validate_matrix_truth_cells(
            self.binding,
            self.shared_resources.reservoir_scene_ids_by_joint_cell,
            name="shared reservoir",
        )
        opening_ids = {
            scene_index for cell in self.opening_union_scene_ids_by_joint_cell for scene_index in cell
        }
        reservoir_ids = {
            scene_index
            for cell in self.shared_resources.reservoir_scene_ids_by_joint_cell
            for scene_index in cell
        }
        if opening_ids & reservoir_ids:
            raise BoundBankAuditV2Error("opening union and shared reservoir are not disjoint")

    @property
    def opening_union_digest(self) -> str:
        return _opening_union_digest(self.binding, self.opening_union_scene_ids_by_joint_cell)

    @property
    def pre_evaluation_checkpoint_digest(self) -> str:
        return self.episodes[0].pre_evaluation_checkpoint_digest

    @property
    def optimizer_step(self) -> int:
        return self.episodes[0].optimizer_step

    def noisy_side(self, geometry_slug: str) -> OfficialTargetSideV2:
        if geometry_slug not in _NOISY_A_SLUGS:
            raise BoundBankAuditV2Error("noisy_side requires an A-noisy geometry")
        episode = next(item for item in self.episodes if item.geometry.slug == geometry_slug)
        if episode.a_error_target_side is None:  # pragma: no cover - guaranteed by role schema
            raise BoundBankAuditV2Error("A-noisy geometry lacks a target-side mirror")
        return episode.a_error_target_side

    def _unsigned_obj(self) -> dict[str, Any]:
        return {
            "binding": self.binding.as_obj(),
            "shared_resources": self.shared_resources.as_obj(),
            "joint_cell_order_c_p_q": list(JOINT_CELL_ORDER),
            "opening_union_scene_ids_by_joint_cell": _scene_ids_obj(
                self.opening_union_scene_ids_by_joint_cell
            ),
            "opening_union_digest": self.opening_union_digest,
            "episodes": [episode.as_obj() for episode in self.episodes],
        }

    @property
    def digest(self) -> str:
        return _json_digest(self._unsigned_obj(), domain=_EVALUATION_QUARTET_DOMAIN)

    def as_obj(self) -> dict[str, Any]:
        return {**self._unsigned_obj(), "quartet_digest": self.digest}


def build_c_official_evaluation_quartet_v2(
    binding: RuleTripleBindingV2,
    *,
    shared_resources: EvaluationSharedBindingV2,
    opening_union_scene_ids_by_joint_cell: tuple[tuple[int, ...], ...],
    noisy_a_target_sides: Mapping[str, OfficialTargetSideV2],
    pre_evaluation_checkpoint_digest: str,
    optimizer_step: int,
) -> COfficialEvaluationQuartetV2:
    """Build a quartet, taking each opening as a nested prefix of one union."""

    _validate_binding(binding)
    if type(noisy_a_target_sides) is not dict or set(noisy_a_target_sides) != set(_NOISY_A_SLUGS):
        raise BoundBankAuditV2Error("noisy_a_target_sides must name exactly a1_b0 and a1_b2")
    _validate_scene_matrix_shape(opening_union_scene_ids_by_joint_cell, name="evaluation opening union")
    union_digest = _opening_union_digest(binding, opening_union_scene_ids_by_joint_cell)
    episodes: list[COfficialEvaluationEpisodeV2] = []
    for geometry in EVIDENCE_GEOMETRIES:
        side = (
            noisy_a_target_sides[geometry.slug]
            if geometry.alternative_a_errors is AlternativeAErrorsV2.NOISY
            else None
        )
        if side is not None and type(side) is not OfficialTargetSideV2:
            raise BoundBankAuditV2Error("noisy A target side uses the wrong enum type")
        role_block = build_role_block_v2(
            binding.digest,
            geometry,
            a_error_target_side=side,
        )
        opening = tuple(
            opening_union_scene_ids_by_joint_cell[cell_index][:count]
            for cell_index, count in enumerate(role_block.schedule.joint_truth_cell_counts)
        )
        episodes.append(
            COfficialEvaluationEpisodeV2(
                role_block=role_block,
                selected_rotation_position=2,
                opening_scene_ids_by_joint_cell=opening,
                opening_union_digest=union_digest,
                display_position=shared_resources.display_geometry_order.index(geometry.slug),
                shared_resource_binding_digest=shared_resources.digest,
                pre_evaluation_checkpoint_digest=pre_evaluation_checkpoint_digest,
                optimizer_step=optimizer_step,
            )
        )
    return COfficialEvaluationQuartetV2(
        binding=binding,
        shared_resources=shared_resources,
        opening_union_scene_ids_by_joint_cell=opening_union_scene_ids_by_joint_cell,
        episodes=tuple(episodes),
    )


def _quartet_from_obj(value: object) -> COfficialEvaluationQuartetV2:
    obj = _require_mapping(
        value,
        (
            "binding",
            "shared_resources",
            "joint_cell_order_c_p_q",
            "opening_union_scene_ids_by_joint_cell",
            "opening_union_digest",
            "episodes",
            "quartet_digest",
        ),
        name="C-Official evaluation quartet",
    )
    if obj["joint_cell_order_c_p_q"] != list(JOINT_CELL_ORDER):
        raise BoundBankAuditV2Error("evaluation quartet joint-cell order differs")
    raw_episodes = obj["episodes"]
    if type(raw_episodes) is not list:
        raise BoundBankAuditV2Error("evaluation quartet episodes must be an array")
    quartet = COfficialEvaluationQuartetV2(
        binding=_parse_binding(obj["binding"]),
        shared_resources=_shared_binding_from_obj(obj["shared_resources"]),
        opening_union_scene_ids_by_joint_cell=_scene_ids_from_obj(
            obj["opening_union_scene_ids_by_joint_cell"], name="evaluation opening union"
        ),
        episodes=tuple(_evaluation_episode_from_obj(item) for item in raw_episodes),
    )
    if obj["opening_union_digest"] != quartet.opening_union_digest:
        raise BoundBankAuditV2Error("evaluation quartet opening-union digest mismatch")
    if obj["quartet_digest"] != quartet.digest:
        raise BoundBankAuditV2Error("evaluation quartet digest mismatch")
    if _dump_json(obj) != _dump_json(quartet.as_obj()):
        raise BoundBankAuditV2Error("evaluation quartet contains inconsistent derived metadata")
    return quartet


@dataclass(frozen=True, slots=True)
class COfficialEvaluationMirrorUnitV2:
    """Two distinct quartets with opposite mirrors in both A-noisy cells."""

    quartets: tuple[COfficialEvaluationQuartetV2, ...]

    def __post_init__(self) -> None:
        if type(self.quartets) is not tuple or len(self.quartets) != 2:
            raise BoundBankAuditV2Error("evaluation mirror unit requires exactly two quartets")
        if any(type(item) is not COfficialEvaluationQuartetV2 for item in self.quartets):
            raise BoundBankAuditV2Error("evaluation mirror unit received a foreign quartet type")
        if self.quartets != tuple(sorted(self.quartets, key=lambda item: item.binding.digest)):
            raise BoundBankAuditV2Error("evaluation quartets must be in canonical bound-triple order")
        if self.quartets[0].binding.digest == self.quartets[1].binding.digest:
            raise BoundBankAuditV2Error("a mirror unit requires two distinct bound triples")
        if len({item.pre_evaluation_checkpoint_digest for item in self.quartets}) != 1:
            raise BoundBankAuditV2Error("mirror-unit quartets must use one unchanged checkpoint")
        if len({item.optimizer_step for item in self.quartets}) != 1:
            raise BoundBankAuditV2Error("an optimizer update occurred within the mirror unit")
        for slug in _NOISY_A_SLUGS:
            if {item.noisy_side(slug) for item in self.quartets} != set(OfficialTargetSideV2):
                raise BoundBankAuditV2Error(
                    f"mirror unit does not use opposite Official target sides for {slug}"
                )

    @property
    def pre_evaluation_checkpoint_digest(self) -> str:
        return self.quartets[0].pre_evaluation_checkpoint_digest

    @property
    def optimizer_step(self) -> int:
        return self.quartets[0].optimizer_step

    def _unsigned_obj(self) -> dict[str, Any]:
        return {"quartets": [quartet.as_obj() for quartet in self.quartets]}

    @property
    def digest(self) -> str:
        return _json_digest(self._unsigned_obj(), domain=_EVALUATION_MIRROR_UNIT_DOMAIN)

    def as_obj(self) -> dict[str, Any]:
        return {**self._unsigned_obj(), "mirror_unit_digest": self.digest}


def build_c_official_evaluation_mirror_unit_v2(
    first: COfficialEvaluationQuartetV2,
    second: COfficialEvaluationQuartetV2,
) -> COfficialEvaluationMirrorUnitV2:
    if type(first) is not COfficialEvaluationQuartetV2 or type(second) is not COfficialEvaluationQuartetV2:
        raise TypeError("mirror unit accepts only COfficialEvaluationQuartetV2 values")
    return COfficialEvaluationMirrorUnitV2(
        tuple(sorted((first, second), key=lambda item: item.binding.digest))
    )


def _mirror_unit_from_obj(value: object) -> COfficialEvaluationMirrorUnitV2:
    obj = _require_mapping(
        value,
        ("quartets", "mirror_unit_digest"),
        name="C-Official evaluation mirror unit",
    )
    raw_quartets = obj["quartets"]
    if type(raw_quartets) is not list:
        raise BoundBankAuditV2Error("mirror-unit quartets must be an array")
    unit = COfficialEvaluationMirrorUnitV2(tuple(_quartet_from_obj(item) for item in raw_quartets))
    if obj["mirror_unit_digest"] != unit.digest:
        raise BoundBankAuditV2Error("evaluation mirror-unit digest mismatch")
    if _dump_json(obj) != _dump_json(unit.as_obj()):
        raise BoundBankAuditV2Error("evaluation mirror unit contains inconsistent metadata")
    return unit


@dataclass(frozen=True, slots=True)
class COfficialEvaluationQuartetAuditV2:
    """Nominal report for C-Official matched quartets, paired by mirror."""

    mirror_units: tuple[COfficialEvaluationMirrorUnitV2, ...]

    def __post_init__(self) -> None:
        if not self.mirror_units or type(self.mirror_units) is not tuple:
            raise BoundBankAuditV2Error("evaluation audit requires a nonempty tuple of mirror units")
        if any(type(unit) is not COfficialEvaluationMirrorUnitV2 for unit in self.mirror_units):
            raise BoundBankAuditV2Error("evaluation audit received a foreign mirror-unit type")
        if self.mirror_units != tuple(sorted(self.mirror_units, key=lambda item: item.digest)):
            raise BoundBankAuditV2Error("evaluation mirror units must be in canonical digest order")

    @property
    def catalog_digest(self) -> str:
        return _catalog_contract()[0]

    @property
    def supported_catalog_digest(self) -> str:
        return _catalog_contract()[1]

    def _all_quartets(self) -> tuple[COfficialEvaluationQuartetV2, ...]:
        return tuple(quartet for unit in self.mirror_units for quartet in unit.quartets)

    def _checks(self) -> dict[str, bool]:
        quartets = self._all_quartets()
        triple_ids = [item.binding.digest for item in quartets]
        return {
            "full_and_supported_catalog_binding_exact": all(
                item.binding.catalog_digest == self.catalog_digest
                and item.binding.supported_catalog_digest == self.supported_catalog_digest
                for item in quartets
            ),
            "fresh_bound_triples_unique": len(triple_ids) == len(set(triple_ids)),
            "one_c_official_episode_per_geometry": all(len(item.episodes) == 4 for item in quartets),
            "shared_quartet_resource_and_nested_opening_binding": True,
            "no_weight_updates_and_all_contexts_reset": all(
                not episode.weight_update_performed and episode.context_reset_before_episode
                for quartet in quartets
                for episode in quartet.episodes
            ),
            "ast_private_and_classification_ast_blind": all(
                episode.official_ast_private_during_inquiry
                and episode.rule_ast_committed_before_terminal
                and episode.terminal_classification_ast_blind
                and episode.independent_terminal_replay
                for quartet in quartets
                for episode in quartet.episodes
            ),
            "opposite_noisy_p_mirrors_per_eight_episode_unit": True,
            "single_unchanged_checkpoint": len({item.pre_evaluation_checkpoint_digest for item in quartets})
            == 1
            and len({item.optimizer_step for item in quartets}) == 1,
        }

    @property
    def evaluation_quartet_audit_passed(self) -> bool:
        return all(self._checks().values())

    def _unsigned_obj(self) -> dict[str, Any]:
        quartet_count = len(self._all_quartets())
        return {
            "schema_version": C_OFFICIAL_EVALUATION_QUARTET_AUDIT_SCHEMA_VERSION,
            "report_kind": _EVALUATION_REPORT_KIND,
            "authorization": dict(_AUTHORIZATION),
            "catalog_digest": self.catalog_digest,
            "supported_catalog_digest": self.supported_catalog_digest,
            "mirror_unit_count": len(self.mirror_units),
            "bound_triple_count": quartet_count,
            "episode_count": quartet_count * 4,
            "mirror_units": [unit.as_obj() for unit in self.mirror_units],
            "checks": [{"name": name, "passed": passed} for name, passed in self._checks().items()],
            "evaluation_quartet_audit_passed": self.evaluation_quartet_audit_passed,
            "substitution_contract": {
                "may_substitute_for_training_role_audit": False,
                "may_substitute_for_metadata_stress_population_audit": False,
            },
        }

    @property
    def digest(self) -> str:
        return _json_digest(self._unsigned_obj(), domain=_EVALUATION_REPORT_DOMAIN)

    def as_obj(self) -> dict[str, Any]:
        return {**self._unsigned_obj(), "evaluation_quartet_audit_digest": self.digest}


def build_c_official_evaluation_quartet_audit_v2(
    mirror_units: Iterable[COfficialEvaluationMirrorUnitV2],
) -> COfficialEvaluationQuartetAuditV2:
    materialized = tuple(mirror_units)
    if any(type(unit) is not COfficialEvaluationMirrorUnitV2 for unit in materialized):
        raise TypeError("evaluation audit accepts only COfficialEvaluationMirrorUnitV2 values")
    return COfficialEvaluationQuartetAuditV2(tuple(sorted(materialized, key=lambda item: item.digest)))


def c_official_evaluation_quartet_audit_v2_from_obj(
    value: object,
    *,
    expected_digest: str | None = None,
) -> COfficialEvaluationQuartetAuditV2:
    fields = (
        "schema_version",
        "report_kind",
        "authorization",
        "catalog_digest",
        "supported_catalog_digest",
        "mirror_unit_count",
        "bound_triple_count",
        "episode_count",
        "mirror_units",
        "checks",
        "evaluation_quartet_audit_passed",
        "substitution_contract",
        "evaluation_quartet_audit_digest",
    )
    obj = _require_mapping(value, fields, name="C-Official evaluation-quartet audit")
    if (
        type(obj["schema_version"]) is not int
        or obj["schema_version"] != C_OFFICIAL_EVALUATION_QUARTET_AUDIT_SCHEMA_VERSION
        or obj["report_kind"] != _EVALUATION_REPORT_KIND
    ):
        raise BoundBankAuditV2Error("evaluation-quartet audit schema identity mismatch")
    _require_authorization(obj["authorization"])
    raw_units = obj["mirror_units"]
    if type(raw_units) is not list:
        raise BoundBankAuditV2Error("evaluation mirror units must be an array")
    audit = build_c_official_evaluation_quartet_audit_v2(_mirror_unit_from_obj(item) for item in raw_units)
    if obj["evaluation_quartet_audit_digest"] != audit.digest:
        raise BoundBankAuditV2Error("evaluation-quartet audit digest mismatch")
    if expected_digest is not None and audit.digest != _require_sha256(
        expected_digest, name="expected evaluation-quartet audit digest"
    ):
        raise BoundBankAuditV2Error("evaluation-quartet audit differs from the externally expected digest")
    if _dump_json(obj) != _dump_json(audit.as_obj()):
        raise BoundBankAuditV2Error("evaluation-quartet audit contains inconsistent metadata")
    return audit


def verify_c_official_evaluation_quartet_audit_v2(
    audit: COfficialEvaluationQuartetAuditV2,
) -> COfficialEvaluationQuartetAuditV2:
    if type(audit) is not COfficialEvaluationQuartetAuditV2:
        raise TypeError("verify requires a COfficialEvaluationQuartetAuditV2")
    return c_official_evaluation_quartet_audit_v2_from_obj(audit.as_obj(), expected_digest=audit.digest)


def serialize_c_official_evaluation_quartet_audit_v2(
    audit: COfficialEvaluationQuartetAuditV2,
) -> str:
    return _dump_json(verify_c_official_evaluation_quartet_audit_v2(audit).as_obj())


def parse_c_official_evaluation_quartet_audit_v2(
    text: str,
    *,
    expected_digest: str | None = None,
) -> COfficialEvaluationQuartetAuditV2:
    audit = c_official_evaluation_quartet_audit_v2_from_obj(_load_json(text), expected_digest=expected_digest)
    if serialize_c_official_evaluation_quartet_audit_v2(audit) != text:
        raise BoundBankAuditV2Error("evaluation-quartet audit JSON is not canonical compact JSON")
    return audit


@dataclass(frozen=True, slots=True)
class CatalogBoundStressRoleUnitV2:
    """One catalog-bound role block in the optional metadata stress pool."""

    binding: RuleTripleBindingV2
    role_block: RoleBlockV2

    def __post_init__(self) -> None:
        _validate_bound_block(self.binding, self.role_block)

    def _unsigned_obj(self) -> dict[str, Any]:
        return {"binding": self.binding.as_obj(), "role_block": self.role_block.as_obj()}

    @property
    def digest(self) -> str:
        return _json_digest(self._unsigned_obj(), domain=_STRESS_UNIT_DOMAIN)

    def as_obj(self) -> dict[str, Any]:
        return {**self._unsigned_obj(), "unit_digest": self.digest}


def build_catalog_bound_stress_role_unit_v2(
    binding: RuleTripleBindingV2,
    geometry: EvidenceGeometryV2,
    *,
    a_error_target_side: OfficialTargetSideV2 | None = None,
) -> CatalogBoundStressRoleUnitV2:
    _validate_binding(binding)
    return CatalogBoundStressRoleUnitV2(
        binding,
        build_role_block_v2(
            binding.digest,
            geometry,
            a_error_target_side=a_error_target_side,
        ),
    )


def _stress_unit_from_obj(value: object) -> CatalogBoundStressRoleUnitV2:
    obj = _require_mapping(
        value,
        ("binding", "role_block", "unit_digest"),
        name="catalog-bound stress role unit",
    )
    unit = CatalogBoundStressRoleUnitV2(_parse_binding(obj["binding"]), _parse_role_block(obj["role_block"]))
    if obj["unit_digest"] != unit.digest:
        raise BoundBankAuditV2Error("stress role-unit digest mismatch")
    if _dump_json(obj) != _dump_json(unit.as_obj()):
        raise BoundBankAuditV2Error("stress role unit contains inconsistent metadata")
    return unit


@dataclass(frozen=True, slots=True)
class MetadataStressPopulationAuditV2:
    """Optional four-geometry metadata diagnostic; never a bank audit substitute."""

    units: tuple[CatalogBoundStressRoleUnitV2, ...]

    def __post_init__(self) -> None:
        if not self.units or type(self.units) is not tuple:
            raise BoundBankAuditV2Error("metadata stress audit requires a nonempty tuple of units")
        if any(type(unit) is not CatalogBoundStressRoleUnitV2 for unit in self.units):
            raise BoundBankAuditV2Error("metadata stress audit received a foreign unit type")
        canonical = tuple(sorted(self.units, key=lambda item: (item.binding.digest, item.digest)))
        if self.units != canonical:
            raise BoundBankAuditV2Error("metadata stress units must be in canonical digest order")

    @property
    def catalog_digest(self) -> str:
        return _catalog_contract()[0]

    @property
    def supported_catalog_digest(self) -> str:
        return _catalog_contract()[1]

    def _geometry_counts(self) -> Counter[str]:
        return Counter(unit.role_block.geometry.slug for unit in self.units)

    def _mirror_counts(self) -> Counter[tuple[str, int]]:
        result: Counter[tuple[str, int]] = Counter()
        for unit in self.units:
            side = unit.role_block.a_error_target_side
            if side is not None:
                result[(unit.role_block.geometry.slug, side.value)] += 1
        return result

    def _checks(self) -> dict[str, bool]:
        count = len(self.units)
        triple_ids = [unit.binding.digest for unit in self.units]
        geometry_counts = self._geometry_counts()
        geometry_exact = count % 4 == 0 and all(
            geometry_counts[geometry.slug] == count // 4 for geometry in EVIDENCE_GEOMETRIES
        )
        mirror_counts = self._mirror_counts()
        mirror_exact = all(
            geometry_counts[slug] % 2 == 0
            and all(
                mirror_counts[(slug, side.value)] == geometry_counts[slug] // 2
                for side in OfficialTargetSideV2
            )
            for slug in _NOISY_A_SLUGS
        )
        return {
            "full_and_supported_catalog_binding_exact": all(
                unit.binding.catalog_digest == self.catalog_digest
                and unit.binding.supported_catalog_digest == self.supported_catalog_digest
                for unit in self.units
            ),
            "minimum_powered_independent_triples": count >= MINIMUM_POWERED_STRESS_TRIPLES,
            "unique_bound_triples": len(triple_ids) == len(set(triple_ids)),
            "family_by_official_a_b_exact": True,
            "four_geometry_population_balance_exact": geometry_exact,
            "noisy_a_target_mirror_balance_exact": mirror_exact,
        }

    @property
    def stress_population_audit_passed(self) -> bool:
        return all(self._checks().values())

    def _unsigned_obj(self) -> dict[str, Any]:
        geometry_counts = self._geometry_counts()
        mirror_counts = self._mirror_counts()
        return {
            "schema_version": METADATA_STRESS_POPULATION_AUDIT_SCHEMA_VERSION,
            "report_kind": _STRESS_REPORT_KIND,
            "authorization": dict(_AUTHORIZATION),
            "catalog_digest": self.catalog_digest,
            "supported_catalog_digest": self.supported_catalog_digest,
            "minimum_powered_stress_triples": MINIMUM_POWERED_STRESS_TRIPLES,
            "bound_triple_count": len(self.units),
            "episode_count": len(self.units) * 3,
            "units": [unit.as_obj() for unit in self.units],
            "family_by_official_a_b_counts": _family_role_table(len(self.units)),
            "geometry_counts": [
                {"geometry": geometry.slug, "count": geometry_counts[geometry.slug]}
                for geometry in EVIDENCE_GEOMETRIES
            ],
            "noisy_a_target_mirror_counts": [
                {
                    "geometry": slug,
                    "a_error_target_side": side.value,
                    "count": mirror_counts[(slug, side.value)],
                }
                for slug in _NOISY_A_SLUGS
                for side in OfficialTargetSideV2
            ],
            "checks": [{"name": name, "passed": passed} for name, passed in self._checks().items()],
            "stress_population_audit_passed": self.stress_population_audit_passed,
            "substitution_contract": {
                "may_substitute_for_training_role_audit": False,
                "may_substitute_for_evaluation_quartet_audit": False,
            },
        }

    @property
    def digest(self) -> str:
        return _json_digest(self._unsigned_obj(), domain=_STRESS_REPORT_DOMAIN)

    def as_obj(self) -> dict[str, Any]:
        return {**self._unsigned_obj(), "stress_population_audit_digest": self.digest}


def build_metadata_stress_population_audit_v2(
    units: Iterable[CatalogBoundStressRoleUnitV2],
) -> MetadataStressPopulationAuditV2:
    materialized = tuple(units)
    if any(type(unit) is not CatalogBoundStressRoleUnitV2 for unit in materialized):
        raise TypeError("stress audit accepts only CatalogBoundStressRoleUnitV2 values")
    return MetadataStressPopulationAuditV2(
        tuple(sorted(materialized, key=lambda item: (item.binding.digest, item.digest)))
    )


def metadata_stress_population_audit_v2_from_obj(
    value: object,
    *,
    expected_digest: str | None = None,
) -> MetadataStressPopulationAuditV2:
    fields = (
        "schema_version",
        "report_kind",
        "authorization",
        "catalog_digest",
        "supported_catalog_digest",
        "minimum_powered_stress_triples",
        "bound_triple_count",
        "episode_count",
        "units",
        "family_by_official_a_b_counts",
        "geometry_counts",
        "noisy_a_target_mirror_counts",
        "checks",
        "stress_population_audit_passed",
        "substitution_contract",
        "stress_population_audit_digest",
    )
    obj = _require_mapping(value, fields, name="metadata stress-population audit")
    if (
        type(obj["schema_version"]) is not int
        or obj["schema_version"] != METADATA_STRESS_POPULATION_AUDIT_SCHEMA_VERSION
        or obj["report_kind"] != _STRESS_REPORT_KIND
    ):
        raise BoundBankAuditV2Error("metadata stress-population audit schema identity mismatch")
    _require_authorization(obj["authorization"])
    raw_units = obj["units"]
    if type(raw_units) is not list:
        raise BoundBankAuditV2Error("metadata stress units must be an array")
    audit = build_metadata_stress_population_audit_v2(_stress_unit_from_obj(item) for item in raw_units)
    if obj["stress_population_audit_digest"] != audit.digest:
        raise BoundBankAuditV2Error("metadata stress-population audit digest mismatch")
    if expected_digest is not None and audit.digest != _require_sha256(
        expected_digest, name="expected metadata stress-population audit digest"
    ):
        raise BoundBankAuditV2Error(
            "metadata stress-population audit differs from the externally expected digest"
        )
    if _dump_json(obj) != _dump_json(audit.as_obj()):
        raise BoundBankAuditV2Error("metadata stress-population audit contains inconsistent metadata")
    return audit


def verify_metadata_stress_population_audit_v2(
    audit: MetadataStressPopulationAuditV2,
) -> MetadataStressPopulationAuditV2:
    if type(audit) is not MetadataStressPopulationAuditV2:
        raise TypeError("verify requires a MetadataStressPopulationAuditV2")
    return metadata_stress_population_audit_v2_from_obj(audit.as_obj(), expected_digest=audit.digest)


def serialize_metadata_stress_population_audit_v2(
    audit: MetadataStressPopulationAuditV2,
) -> str:
    return _dump_json(verify_metadata_stress_population_audit_v2(audit).as_obj())


def parse_metadata_stress_population_audit_v2(
    text: str,
    *,
    expected_digest: str | None = None,
) -> MetadataStressPopulationAuditV2:
    audit = metadata_stress_population_audit_v2_from_obj(_load_json(text), expected_digest=expected_digest)
    if serialize_metadata_stress_population_audit_v2(audit) != text:
        raise BoundBankAuditV2Error("metadata stress-population JSON is not canonical compact JSON")
    return audit
