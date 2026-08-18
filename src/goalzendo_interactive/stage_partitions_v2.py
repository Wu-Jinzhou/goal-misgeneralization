"""Additive schema-v2 rule partitions and target-family eligibility.

This module deliberately does not mutate or alias the released v1 partition
map.  It recomputes a complete truth-identity split with a distinct capability
partition and reserves both placard-literal identities for that partition.
"""

from __future__ import annotations

import hashlib
import math
from collections import Counter
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Literal, cast

from ._json import CanonicalJSONError, dump_json, json_digest, load_json
from .catalog import CatalogEntry, RuleCatalog, build_rule_catalog
from .episodes import (
    MAX_TARGET_SHADOW_DISAGREEMENT,
    MIN_TARGET_SHADOW_CELL_COUNT,
    MIN_TARGET_SHADOW_DISAGREEMENT,
)
from .rules import BINARY_OPS, BinaryOp, BinaryRule
from .rules import Literal as RuleLiteral
from .schema import SCENE_COUNT

RulePartitionV2 = Literal[
    "warm_start",
    "engineering",
    "capability",
    "pilot",
    "confirmatory_train",
    "validation",
    "evaluation",
]
TargetFamilyV2 = Literal["binary_piece", "literal_piece", "placard_literal"]
CatalogRuleFamilyV2 = Literal[
    "binary_piece",
    "literal_piece",
    "placard_literal",
    "binary_mixed",
]

RULE_PARTITIONS_V2: tuple[RulePartitionV2, ...] = (
    "warm_start",
    "engineering",
    "capability",
    "pilot",
    "confirmatory_train",
    "validation",
    "evaluation",
)
DEMAND_WEIGHTED_PARTITION_CYCLE_V2: tuple[RulePartitionV2, ...] = (
    "warm_start",
    "engineering",
    "engineering",
    "capability",
    "pilot",
    "confirmatory_train",
    "confirmatory_train",
    "validation",
    "evaluation",
    "evaluation",
)
LITERAL_SHADOW_PARTITION_CYCLE_V2: tuple[RulePartitionV2, ...] = (
    "warm_start",
    "engineering",
    "engineering",
    "capability",
    "capability",
    "pilot",
    "confirmatory_train",
    "confirmatory_train",
    "validation",
    "evaluation",
    "evaluation",
)
TARGET_FAMILIES_V2: tuple[TargetFamilyV2, ...] = (
    "binary_piece",
    "literal_piece",
    "placard_literal",
)
CATALOG_RULE_FAMILIES_V2: tuple[CatalogRuleFamilyV2, ...] = (
    "binary_piece",
    "literal_piece",
    "placard_literal",
    "binary_mixed",
)
RULE_PARTITION_SCHEMA_VERSION_V2 = 2
ELIGIBLE_PAIR_TABLE_SCHEMA_VERSION_V2 = 2
RULE_PARTITION_HASH_DOMAIN_V2 = (
    b"goalzendo-interactive-rule-partition-demand-weighted-v2\0"
)
PAIR_ORDER_HASH_DOMAIN_V2 = b"goalzendo-interactive-eligible-pair-order-v2\0"
REJECTED_EQUAL_PARTITION_DIGEST_V2 = (
    "6b9234d8c7920591e13fb0227a09a821c7a044d9af27b5b9b3b9046957fa4be0"
)
REJECTED_EQUAL_PAIR_TABLE_DIGEST_V2 = (
    "bb7cfab9369687aabf067aa20709312e9897d4b40a1ecde4ff077a612e63325d"
)
_UNIVERSE_MASK = (1 << SCENE_COUNT) - 1


class StagePartitionV2Error(ValueError):
    """Raised when a schema-v2 partition or eligibility object is invalid."""


@dataclass(frozen=True, slots=True)
class RejectedEqualCapacityRowV2:
    partition: RulePartitionV2
    op: BinaryOp
    available: int
    requested: int

    @property
    def sufficient(self) -> bool:
        return self.available >= self.requested

    def as_obj(self) -> dict[str, Any]:
        return {
            "partition": self.partition,
            "op": self.op,
            "available": self.available,
            "requested": self.requested,
            "sufficient": self.sufficient,
        }


@dataclass(frozen=True, slots=True)
class RejectedEqualPartitionAuditV2:
    """Pinned evidence for the non-authorizing equal seven-way proposal."""

    capacity: tuple[RejectedEqualCapacityRowV2, ...]

    def __post_init__(self) -> None:
        rows = tuple(self.capacity)
        object.__setattr__(self, "capacity", rows)
        if len(rows) != len(RULE_PARTITIONS_V2) * len(BINARY_OPS):
            raise StagePartitionV2Error("rejected equal-split audit must cover every cell")
        if len({(row.partition, row.op) for row in rows}) != len(rows):
            raise StagePartitionV2Error("rejected equal-split capacity rows must be unique")
        expected_shortfalls = {
            ("engineering", "exactly_one", 367, 384),
            ("confirmatory_train", "exactly_one", 364, 384),
            ("evaluation", "exactly_one", 376, 384),
        }
        actual_shortfalls = {
            (row.partition, row.op, row.available, row.requested)
            for row in rows
            if not row.sufficient
        }
        if actual_shortfalls != expected_shortfalls:
            raise StagePartitionV2Error("rejected equal-split shortfall evidence changed")

    def as_obj(self) -> dict[str, Any]:
        return {
            "schema_version": 2,
            "design_id": "equal-seven-way-round-robin-v2-rejected",
            "status": "rejected_non_authorizing",
            "algorithm": "placard-literals-to-capability-then-global-sha256-rank-round-robin",
            "cycle": list(RULE_PARTITIONS_V2),
            "partition_digest": REJECTED_EQUAL_PARTITION_DIGEST_V2,
            "eligible_pairs_digest": REJECTED_EQUAL_PAIR_TABLE_DIGEST_V2,
            "partition_counts": {
                "warm_start": 1_043,
                "engineering": 1_043,
                "capability": 1_045,
                "pilot": 1_043,
                "confirmatory_train": 1_042,
                "validation": 1_042,
                "evaluation": 1_042,
            },
            "capability_target_identity_counts": {
                "binary_piece": 983,
                "literal_piece": 12,
                "placard_literal": 2,
            },
            "capability_control_shortfall": {
                "target_family": "literal_piece",
                "available": 12,
                "requested": 16,
                "sufficient": False,
            },
            "capacity": [row.as_obj() for row in self.capacity],
            "rejection_reason": (
                "prospective unique-target quotas exceed three exactly_one pools "
                "and the capability literal-piece control pool"
            ),
            "production_bank_generation_authorized": False,
            "weight_updates_authorized": False,
        }

    @property
    def digest(self) -> str:
        return json_digest(
            self.as_obj(),
            domain="goalzendo-interactive-rejected-equal-partition-audit-v2",
        )


@lru_cache(maxsize=1)
def build_rejected_equal_partition_audit_v2() -> RejectedEqualPartitionAuditV2:
    available = {
        "warm_start": (311, 304, 364),
        "engineering": (294, 329, 367),
        "capability": (299, 300, 384),
        "pilot": (306, 310, 375),
        "confirmatory_train": (299, 308, 364),
        "validation": (309, 283, 398),
        "evaluation": (329, 277, 376),
    }
    requested = {
        "warm_start": (64, 64, 128),
        "engineering": (192, 192, 384),
        "capability": (60, 60, 122),
        "pilot": (64, 64, 128),
        "confirmatory_train": (192, 192, 384),
        "validation": (64, 64, 128),
        "evaluation": (192, 192, 384),
    }
    return RejectedEqualPartitionAuditV2(
        tuple(
            RejectedEqualCapacityRowV2(partition, op, available_count, requested_count)
            for partition in RULE_PARTITIONS_V2
            for op, available_count, requested_count in zip(
                BINARY_OPS,
                available[partition],
                requested[partition],
                strict=True,
            )
        )
    )


def serialize_rejected_equal_partition_audit_v2(
    report: RejectedEqualPartitionAuditV2,
) -> str:
    if type(report) is not RejectedEqualPartitionAuditV2:
        raise TypeError(
            "serialize_rejected_equal_partition_audit_v2 requires "
            "RejectedEqualPartitionAuditV2"
        )
    return dump_json(report.as_obj())


def _entry_uses_placard(entry: CatalogEntry) -> bool:
    rule = entry.rule
    literals = (rule,) if type(rule) is RuleLiteral else cast(BinaryRule, rule).args
    return any(literal.atom.op == "placard_is" for literal in literals)


def catalog_rule_family_v2(entry: CatalogEntry) -> CatalogRuleFamilyV2:
    """Classify the canonical representative without changing its identity."""

    if type(entry) is not CatalogEntry:
        raise TypeError("catalog_rule_family_v2 requires a CatalogEntry")
    uses_placard = _entry_uses_placard(entry)
    if type(entry.rule) is RuleLiteral:
        return "placard_literal" if uses_placard else "literal_piece"
    return "binary_mixed" if uses_placard else "binary_piece"


def _partition_stratum_v2(entry: CatalogEntry) -> str:
    family = catalog_rule_family_v2(entry)
    if family == "binary_piece":
        return f"binary_piece:{cast(BinaryRule, entry.rule).op}"
    return family


@dataclass(frozen=True, slots=True)
class RulePartitionAssignmentV2:
    rule_id: str
    truth_digest: str
    partition: RulePartitionV2
    catalog_family: CatalogRuleFamilyV2

    def as_obj(self) -> dict[str, str]:
        return {
            "rule_id": self.rule_id,
            "truth_digest": self.truth_digest,
            "partition": self.partition,
            "catalog_family": self.catalog_family,
        }


class RuleIdentityPartitionsV2:
    """A total, non-overlapping schema-v2 split of truth-vector identities."""

    __slots__ = (
        "_by_digest",
        "_by_rule_id",
        "_digest",
        "assignments",
        "catalog_digest",
    )

    def __init__(
        self,
        assignments: tuple[RulePartitionAssignmentV2, ...],
        *,
        catalog: RuleCatalog,
    ) -> None:
        if len(assignments) != len(catalog):
            raise StagePartitionV2Error("v2 partition mapping must cover the complete catalog")
        if tuple(sorted(assignments, key=lambda item: item.rule_id)) != assignments:
            raise StagePartitionV2Error("v2 assignments must be in canonical rule-id order")
        self.assignments = assignments
        self.catalog_digest = catalog.digest
        self._digest: str | None = None
        self._by_digest = {item.truth_digest: item.partition for item in assignments}
        self._by_rule_id = {item.rule_id: item.partition for item in assignments}
        if len(self._by_digest) != len(catalog) or len(self._by_rule_id) != len(catalog):
            raise StagePartitionV2Error("every rule identity must occur in exactly one partition")
        for entry, assignment in zip(catalog, assignments, strict=True):
            if (entry.rule_id, entry.truth_digest) != (
                assignment.rule_id,
                assignment.truth_digest,
            ):
                raise StagePartitionV2Error("v2 partition mapping differs from the catalog")
            if assignment.partition not in RULE_PARTITIONS_V2:
                raise StagePartitionV2Error(
                    f"unknown v2 rule partition: {assignment.partition!r}"
                )
            if assignment.catalog_family != catalog_rule_family_v2(entry):
                raise StagePartitionV2Error("v2 catalog-family annotation is inconsistent")
            if (
                assignment.catalog_family == "placard_literal"
                and assignment.partition != "capability"
            ):
                raise StagePartitionV2Error(
                    "both placard-literal identities are reserved for capability"
                )

    def for_entry(self, entry: CatalogEntry) -> RulePartitionV2:
        try:
            partition = self._by_digest[entry.truth_digest]
        except KeyError as exc:
            raise StagePartitionV2Error("entry is absent from the v2 partition map") from exc
        if self._by_rule_id.get(entry.rule_id) != partition:
            raise StagePartitionV2Error("rule id and truth identity map inconsistently")
        return partition

    def for_truth_digest(self, truth_digest: str) -> RulePartitionV2:
        try:
            return self._by_digest[truth_digest]
        except KeyError as exc:
            raise KeyError(f"unknown truth-vector digest: {truth_digest}") from exc

    @property
    def counts(self) -> dict[RulePartitionV2, int]:
        counts = Counter(assignment.partition for assignment in self.assignments)
        return {partition: counts[partition] for partition in RULE_PARTITIONS_V2}

    @property
    def family_counts(self) -> dict[str, int]:
        counts = Counter(
            (assignment.partition, assignment.catalog_family)
            for assignment in self.assignments
        )
        return {
            f"{partition}:{family}": counts[(partition, family)]
            for partition in RULE_PARTITIONS_V2
            for family in CATALOG_RULE_FAMILIES_V2
        }

    def as_obj(self) -> dict[str, Any]:
        return {
            "schema_version": RULE_PARTITION_SCHEMA_VERSION_V2,
            "catalog_digest": self.catalog_digest,
            "algorithm": (
                "family-operator-stratified-sha256-rank-demand-weighted-cycle-v2"
            ),
            "hash_domain": RULE_PARTITION_HASH_DOMAIN_V2[:-1].decode("ascii"),
            "partitions": list(RULE_PARTITIONS_V2),
            "demand_weighted_cycle": list(DEMAND_WEIGHTED_PARTITION_CYCLE_V2),
            "literal_shadow_cycle": list(LITERAL_SHADOW_PARTITION_CYCLE_V2),
            "weight_basis": (
                "prospective unique-target demand only: engineering, "
                "confirmatory_train, and evaluation plan 768 episodes; "
                "warm_start, capability, pilot, and validation plan 256; "
                "no model outcome enters the allocation"
            ),
            "placard_literal_partition": "capability",
            "counts": self.counts,
            "family_counts": self.family_counts,
            "assignments": [assignment.as_obj() for assignment in self.assignments],
        }

    @property
    def digest(self) -> str:
        if self._digest is None:
            self._digest = json_digest(
                self.as_obj(),
                domain="goalzendo-interactive-rule-identity-partitions-v2",
            )
        return self._digest


@lru_cache(maxsize=1)
def build_rule_identity_partitions_v2() -> RuleIdentityPartitionsV2:
    """Re-hash identities in demand-weighted family/operator strata."""

    catalog = build_rule_catalog()
    ordinary = tuple(
        entry for entry in catalog if _partition_stratum_v2(entry) != "placard_literal"
    )
    strata = tuple(sorted({_partition_stratum_v2(entry) for entry in ordinary}))
    by_index: dict[int, RulePartitionV2] = {}
    for stratum in strata:
        ranked = sorted(
            (entry for entry in ordinary if _partition_stratum_v2(entry) == stratum),
            key=lambda entry: hashlib.sha256(
                RULE_PARTITION_HASH_DOMAIN_V2
                + stratum.encode("ascii")
                + b"\0"
                + entry.truth_digest.encode("ascii")
            ).digest()
            + entry.truth_digest.encode("ascii"),
        )
        cycle = (
            LITERAL_SHADOW_PARTITION_CYCLE_V2
            if stratum == "literal_piece"
            else DEMAND_WEIGHTED_PARTITION_CYCLE_V2
        )
        by_index.update(
            {
                entry.index: cycle[rank % len(cycle)]
                for rank, entry in enumerate(ranked)
            }
        )
    for entry in catalog:
        if catalog_rule_family_v2(entry) == "placard_literal":
            by_index[entry.index] = "capability"
    assignments = tuple(
        RulePartitionAssignmentV2(
            entry.rule_id,
            entry.truth_digest,
            by_index[entry.index],
            catalog_rule_family_v2(entry),
        )
        for entry in catalog
    )
    return RuleIdentityPartitionsV2(assignments, catalog=catalog)


def serialize_rule_identity_partitions_v2(partitions: RuleIdentityPartitionsV2) -> str:
    if type(partitions) is not RuleIdentityPartitionsV2:
        raise TypeError(
            "serialize_rule_identity_partitions_v2 requires RuleIdentityPartitionsV2"
        )
    return dump_json(partitions.as_obj())


def parse_rule_identity_partitions_v2(
    text: str,
    *,
    require_canonical: bool = True,
) -> RuleIdentityPartitionsV2:
    try:
        value = load_json(text)
    except CanonicalJSONError as exc:
        raise StagePartitionV2Error(str(exc)) from exc
    expected = build_rule_identity_partitions_v2()
    if value != expected.as_obj():
        raise StagePartitionV2Error("v2 partition manifest differs from deterministic mapping")
    if require_canonical and serialize_rule_identity_partitions_v2(expected) != text:
        raise StagePartitionV2Error("v2 partition JSON is valid but not canonical")
    return expected


def target_shadow_cells_v2(
    target: CatalogEntry,
    shadow: CatalogEntry,
) -> tuple[int, int, int, int]:
    """Return counts in canonical ``Y0Q0,Y0Q1,Y1Q0,Y1Q1`` order."""

    y_bits = target.truth.bits
    q_bits = shadow.truth.bits
    return (
        ((_UNIVERSE_MASK ^ y_bits) & (_UNIVERSE_MASK ^ q_bits)).bit_count(),
        ((_UNIVERSE_MASK ^ y_bits) & q_bits).bit_count(),
        (y_bits & (_UNIVERSE_MASK ^ q_bits)).bit_count(),
        (y_bits & q_bits).bit_count(),
    )


@dataclass(frozen=True, slots=True)
class EligibleTargetShadowPairV2:
    target_index: int
    shadow_index: int
    partition: RulePartitionV2
    target_family: TargetFamilyV2
    cells: tuple[int, int, int, int]
    order_digest: str

    @property
    def disagreement_count(self) -> int:
        return self.cells[1] + self.cells[2]

    def target(self, catalog: RuleCatalog | None = None) -> CatalogEntry:
        return (build_rule_catalog() if catalog is None else catalog)[self.target_index]

    def shadow(self, catalog: RuleCatalog | None = None) -> CatalogEntry:
        return (build_rule_catalog() if catalog is None else catalog)[self.shadow_index]

    def as_obj(self, catalog: RuleCatalog | None = None) -> dict[str, Any]:
        selected = build_rule_catalog() if catalog is None else catalog
        target = selected[self.target_index]
        shadow = selected[self.shadow_index]
        return {
            "target_rule_id": target.rule_id,
            "target_truth_digest": target.truth_digest,
            "shadow_rule_id": shadow.rule_id,
            "shadow_truth_digest": shadow.truth_digest,
            "partition": self.partition,
            "target_family": self.target_family,
            "cells_y0q0_y0q1_y1q0_y1q1": list(self.cells),
            "disagreement_count": self.disagreement_count,
            "order_digest": self.order_digest,
        }


class EligibleTargetShadowTableV2:
    """All exact same-partition pairs for the three schema-v2 target families."""

    __slots__ = (
        "_by_cell",
        "_digest",
        "catalog_digest",
        "pairs",
        "partitions_digest",
    )

    def __init__(
        self,
        pairs: tuple[EligibleTargetShadowPairV2, ...],
        *,
        catalog_digest: str,
        partitions_digest: str,
    ) -> None:
        self.pairs = pairs
        self.catalog_digest = catalog_digest
        self.partitions_digest = partitions_digest
        self._digest: str | None = None
        grouped: dict[
            tuple[RulePartitionV2, TargetFamilyV2],
            list[EligibleTargetShadowPairV2],
        ] = {
            (partition, family): []
            for partition in RULE_PARTITIONS_V2
            for family in TARGET_FAMILIES_V2
        }
        seen: set[tuple[int, int]] = set()
        for pair in pairs:
            key = (pair.target_index, pair.shadow_index)
            if key in seen:
                raise StagePartitionV2Error("v2 eligible-pair table contains a duplicate")
            seen.add(key)
            grouped[(pair.partition, pair.target_family)].append(pair)
        self._by_cell = {key: tuple(value) for key, value in grouped.items()}

    def candidates(
        self,
        partition: RulePartitionV2,
        target_family: TargetFamilyV2,
    ) -> tuple[EligibleTargetShadowPairV2, ...]:
        if partition not in RULE_PARTITIONS_V2:
            raise StagePartitionV2Error(f"unknown v2 partition: {partition!r}")
        if target_family not in TARGET_FAMILIES_V2:
            raise StagePartitionV2Error(f"unknown v2 target family: {target_family!r}")
        return self._by_cell[(partition, target_family)]

    @property
    def counts(self) -> dict[str, int]:
        return {
            f"{partition}:{family}": len(self._by_cell[(partition, family)])
            for partition in RULE_PARTITIONS_V2
            for family in TARGET_FAMILIES_V2
        }

    @property
    def target_identity_counts(self) -> dict[str, int]:
        return {
            f"{partition}:{family}": len(
                {
                    pair.target_index
                    for pair in self._by_cell[(partition, family)]
                }
            )
            for partition in RULE_PARTITIONS_V2
            for family in TARGET_FAMILIES_V2
        }

    @property
    def binary_operator_pair_counts(self) -> dict[str, int]:
        catalog = build_rule_catalog()
        counts = Counter(
            (pair.partition, cast(BinaryRule, catalog[pair.target_index].rule).op)
            for pair in self.pairs
            if pair.target_family == "binary_piece"
        )
        return {
            f"{partition}:{op}": counts[(partition, op)]
            for partition in RULE_PARTITIONS_V2
            for op in BINARY_OPS
        }

    @property
    def binary_operator_target_identity_counts(self) -> dict[str, int]:
        catalog = build_rule_catalog()
        identities: dict[tuple[RulePartitionV2, BinaryOp], set[int]] = {
            (partition, op): set()
            for partition in RULE_PARTITIONS_V2
            for op in BINARY_OPS
        }
        for pair in self.pairs:
            if pair.target_family != "binary_piece":
                continue
            op = cast(BinaryRule, catalog[pair.target_index].rule).op
            identities[(pair.partition, op)].add(pair.target_index)
        return {
            f"{partition}:{op}": len(identities[(partition, op)])
            for partition in RULE_PARTITIONS_V2
            for op in BINARY_OPS
        }

    def as_obj(self) -> dict[str, Any]:
        catalog = build_rule_catalog()
        return {
            "schema_version": ELIGIBLE_PAIR_TABLE_SCHEMA_VERSION_V2,
            "catalog_digest": self.catalog_digest,
            "partitions_digest": self.partitions_digest,
            "same_partition_required": True,
            "shadow_family": "literal_piece",
            "target_families": list(TARGET_FAMILIES_V2),
            "minimum_cell_count": MIN_TARGET_SHADOW_CELL_COUNT,
            "minimum_disagreement_count": math.ceil(
                MIN_TARGET_SHADOW_DISAGREEMENT * SCENE_COUNT
            ),
            "maximum_disagreement_count": math.floor(
                MAX_TARGET_SHADOW_DISAGREEMENT * SCENE_COUNT
            ),
            "counts": self.counts,
            "target_identity_counts": self.target_identity_counts,
            "binary_operator_pair_counts": self.binary_operator_pair_counts,
            "binary_operator_target_identity_counts": (
                self.binary_operator_target_identity_counts
            ),
            "pairs": [pair.as_obj(catalog) for pair in self.pairs],
        }

    @property
    def digest(self) -> str:
        if self._digest is None:
            self._digest = json_digest(
                self.as_obj(),
                domain="goalzendo-interactive-eligible-target-shadow-table-v2",
            )
        return self._digest


@lru_cache(maxsize=1)
def build_eligible_target_shadow_table_v2() -> EligibleTargetShadowTableV2:
    """Enumerate generalized target/shadow pairs under the unchanged v1 bounds."""

    catalog = build_rule_catalog()
    partitions = build_rule_identity_partitions_v2()
    targets = tuple(
        entry
        for entry in catalog
        if catalog_rule_family_v2(entry) in TARGET_FAMILIES_V2
    )
    shadows = tuple(
        entry for entry in catalog if catalog_rule_family_v2(entry) == "literal_piece"
    )
    minimum_disagreement = math.ceil(MIN_TARGET_SHADOW_DISAGREEMENT * SCENE_COUNT)
    maximum_disagreement = math.floor(MAX_TARGET_SHADOW_DISAGREEMENT * SCENE_COUNT)
    pairs: list[EligibleTargetShadowPairV2] = []
    for target in targets:
        partition = partitions.for_entry(target)
        target_family = cast(TargetFamilyV2, catalog_rule_family_v2(target))
        for shadow in shadows:
            if partitions.for_entry(shadow) != partition:
                continue
            cells = target_shadow_cells_v2(target, shadow)
            disagreement = cells[1] + cells[2]
            if min(cells) < MIN_TARGET_SHADOW_CELL_COUNT:
                continue
            if not minimum_disagreement <= disagreement <= maximum_disagreement:
                continue
            order_digest = hashlib.sha256(
                PAIR_ORDER_HASH_DOMAIN_V2
                + target.truth_digest.encode("ascii")
                + b"\0"
                + shadow.truth_digest.encode("ascii")
            ).hexdigest()
            pairs.append(
                EligibleTargetShadowPairV2(
                    target.index,
                    shadow.index,
                    partition,
                    target_family,
                    cells,
                    order_digest,
                )
            )
    pairs.sort(
        key=lambda pair: (
            RULE_PARTITIONS_V2.index(pair.partition),
            TARGET_FAMILIES_V2.index(pair.target_family),
            pair.order_digest,
            pair.target_index,
            pair.shadow_index,
        )
    )
    return EligibleTargetShadowTableV2(
        tuple(pairs),
        catalog_digest=catalog.digest,
        partitions_digest=partitions.digest,
    )


def serialize_eligible_target_shadow_table_v2(
    table: EligibleTargetShadowTableV2,
) -> str:
    if type(table) is not EligibleTargetShadowTableV2:
        raise TypeError(
            "serialize_eligible_target_shadow_table_v2 requires "
            "EligibleTargetShadowTableV2"
        )
    return dump_json(table.as_obj())


def parse_eligible_target_shadow_table_v2(
    text: str,
    *,
    require_canonical: bool = True,
) -> EligibleTargetShadowTableV2:
    try:
        value = load_json(text)
    except CanonicalJSONError as exc:
        raise StagePartitionV2Error(str(exc)) from exc
    expected = build_eligible_target_shadow_table_v2()
    if value != expected.as_obj():
        raise StagePartitionV2Error("v2 eligible-pair manifest differs from deterministic table")
    if require_canonical and serialize_eligible_target_shadow_table_v2(expected) != text:
        raise StagePartitionV2Error("v2 eligible-pair JSON is valid but not canonical")
    return expected
