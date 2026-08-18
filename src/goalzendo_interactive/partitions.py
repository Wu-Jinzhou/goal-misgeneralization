"""Deterministic rule-identity partitions and target/shadow eligibility."""

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
from .rules import BinaryOp, BinaryRule
from .rules import Literal as RuleLiteral
from .schema import SCENE_COUNT

RulePartition = Literal[
    "warm_start",
    "engineering",
    "pilot",
    "confirmatory_train",
    "validation",
    "evaluation",
]
FormulaStratum = Literal["all_or_any", "exactly_one"]

RULE_PARTITIONS: tuple[RulePartition, ...] = (
    "warm_start",
    "engineering",
    "pilot",
    "confirmatory_train",
    "validation",
    "evaluation",
)
RULE_PARTITION_SCHEMA_VERSION = 1
ELIGIBLE_PAIR_TABLE_SCHEMA_VERSION = 1
RULE_PARTITION_HASH_DOMAIN = b"goalzendo-interactive-rule-partition-v1\0"
PAIR_ORDER_HASH_DOMAIN = b"goalzendo-interactive-eligible-pair-order-v1\0"
_UNIVERSE_MASK = (1 << SCENE_COUNT) - 1


class PartitionValidationError(ValueError):
    """Raised when a rule-identity split or pair table is inconsistent."""


def _uses_placard(entry: CatalogEntry) -> bool:
    rule = entry.rule
    literals = (rule,) if type(rule) is RuleLiteral else cast(BinaryRule, rule).args
    return any(literal.atom.op == "placard_is" for literal in literals)


def formula_stratum(entry: CatalogEntry) -> FormulaStratum:
    if type(entry.rule) is not BinaryRule:
        raise PartitionValidationError("formula strata are defined only for binary rules")
    return "exactly_one" if entry.rule.op == "exactly_one" else "all_or_any"


@dataclass(frozen=True, slots=True)
class RulePartitionAssignment:
    rule_id: str
    truth_digest: str
    partition: RulePartition

    def as_obj(self) -> dict[str, str]:
        return {
            "rule_id": self.rule_id,
            "truth_digest": self.truth_digest,
            "partition": self.partition,
        }


class RuleIdentityPartitions:
    """A total, non-overlapping split of full truth-vector identities."""

    __slots__ = ("_by_digest", "_by_rule_id", "_digest", "assignments", "catalog_digest")

    def __init__(
        self,
        assignments: tuple[RulePartitionAssignment, ...],
        *,
        catalog: RuleCatalog,
    ) -> None:
        if len(assignments) != len(catalog):
            raise PartitionValidationError("partition mapping must cover the complete catalog")
        if tuple(sorted(assignments, key=lambda item: item.rule_id)) != assignments:
            raise PartitionValidationError("partition assignments must be in canonical rule-id order")
        self.assignments = assignments
        self.catalog_digest = catalog.digest
        self._digest: str | None = None
        self._by_digest = {item.truth_digest: item.partition for item in assignments}
        self._by_rule_id = {item.rule_id: item.partition for item in assignments}
        if len(self._by_digest) != len(catalog) or len(self._by_rule_id) != len(catalog):
            raise PartitionValidationError("rule identities may occur in exactly one partition")
        for entry, assignment in zip(catalog, assignments, strict=True):
            if (entry.rule_id, entry.truth_digest) != (
                assignment.rule_id,
                assignment.truth_digest,
            ):
                raise PartitionValidationError("partition mapping does not match catalog identity")
            if assignment.partition not in RULE_PARTITIONS:
                raise PartitionValidationError(
                    f"unknown rule partition: {assignment.partition!r}"
                )

    def for_entry(self, entry: CatalogEntry) -> RulePartition:
        try:
            partition = self._by_digest[entry.truth_digest]
        except KeyError as exc:
            raise PartitionValidationError("entry truth identity is absent from partition map") from exc
        if self._by_rule_id.get(entry.rule_id) != partition:
            raise PartitionValidationError("entry rule id and truth identity map inconsistently")
        return partition

    def for_truth_digest(self, truth_digest: str) -> RulePartition:
        try:
            return self._by_digest[truth_digest]
        except KeyError as exc:
            raise KeyError(f"unknown truth-vector digest: {truth_digest}") from exc

    @property
    def counts(self) -> dict[RulePartition, int]:
        counts = Counter(assignment.partition for assignment in self.assignments)
        return {partition: counts[partition] for partition in RULE_PARTITIONS}

    def as_obj(self) -> dict[str, Any]:
        return {
            "schema_version": RULE_PARTITION_SCHEMA_VERSION,
            "catalog_digest": self.catalog_digest,
            "algorithm": "sha256-rank-round-robin-v1",
            "partitions": list(RULE_PARTITIONS),
            "counts": self.counts,
            "assignments": [assignment.as_obj() for assignment in self.assignments],
        }

    @property
    def digest(self) -> str:
        if self._digest is None:
            self._digest = json_digest(
                self.as_obj(), domain="goalzendo-interactive-rule-identity-partitions-v1"
            )
        return self._digest


@lru_cache(maxsize=1)
def build_rule_identity_partitions() -> RuleIdentityPartitions:
    """Split semantic identities before any episode-level selection."""

    catalog = build_rule_catalog()
    ranked = sorted(
        catalog,
        key=lambda entry: hashlib.sha256(
            RULE_PARTITION_HASH_DOMAIN + entry.truth_digest.encode("ascii")
        ).digest()
        + entry.truth_digest.encode("ascii"),
    )
    by_index = {
        entry.index: RULE_PARTITIONS[rank % len(RULE_PARTITIONS)]
        for rank, entry in enumerate(ranked)
    }
    assignments = tuple(
        RulePartitionAssignment(entry.rule_id, entry.truth_digest, by_index[entry.index])
        for entry in catalog
    )
    return RuleIdentityPartitions(assignments, catalog=catalog)


def serialize_rule_identity_partitions(partitions: RuleIdentityPartitions) -> str:
    if type(partitions) is not RuleIdentityPartitions:
        raise TypeError("serialize_rule_identity_partitions requires RuleIdentityPartitions")
    return dump_json(partitions.as_obj())


def parse_rule_identity_partitions(
    text: str,
    *,
    require_canonical: bool = True,
) -> RuleIdentityPartitions:
    try:
        value = load_json(text)
    except CanonicalJSONError as exc:
        raise PartitionValidationError(str(exc)) from exc
    expected = build_rule_identity_partitions()
    if value != expected.as_obj():
        raise PartitionValidationError("partition manifest differs from deterministic mapping")
    if require_canonical and serialize_rule_identity_partitions(expected) != text:
        raise PartitionValidationError("partition JSON is valid but not canonical")
    return expected


def _cell_counts(target: CatalogEntry, shadow: CatalogEntry) -> tuple[int, int, int, int]:
    y_bits = target.truth.bits
    q_bits = shadow.truth.bits
    return (
        ((_UNIVERSE_MASK ^ y_bits) & (_UNIVERSE_MASK ^ q_bits)).bit_count(),
        ((_UNIVERSE_MASK ^ y_bits) & q_bits).bit_count(),
        (y_bits & (_UNIVERSE_MASK ^ q_bits)).bit_count(),
        (y_bits & q_bits).bit_count(),
    )


@dataclass(frozen=True, slots=True)
class EligiblePair:
    target_index: int
    shadow_index: int
    partition: RulePartition
    cells: tuple[int, int, int, int]
    order_digest: str

    @property
    def disagreement_count(self) -> int:
        return self.cells[1] + self.cells[2]

    @property
    def disagreement_rate(self) -> float:
        return self.disagreement_count / SCENE_COUNT

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
            "cells_y0q0_y0q1_y1q0_y1q1": list(self.cells),
            "disagreement_count": self.disagreement_count,
            "order_digest": self.order_digest,
        }


class EligiblePairTable:
    __slots__ = ("_by_cell", "_digest", "catalog_digest", "pairs", "partitions_digest")

    def __init__(
        self,
        pairs: tuple[EligiblePair, ...],
        *,
        catalog_digest: str,
        partitions_digest: str,
    ) -> None:
        self.pairs = pairs
        self.catalog_digest = catalog_digest
        self.partitions_digest = partitions_digest
        self._digest: str | None = None
        catalog = build_rule_catalog()
        grouped: dict[tuple[RulePartition, BinaryOp], list[EligiblePair]] = {
            (partition, cast(BinaryOp, op)): []
            for partition in RULE_PARTITIONS
            for op in ("all", "any", "exactly_one")
        }
        for pair in pairs:
            rule = catalog[pair.target_index].rule
            if type(rule) is not BinaryRule:  # pragma: no cover - constructor invariant
                raise PartitionValidationError("eligible-pair target must be binary")
            grouped[(pair.partition, rule.op)].append(pair)
        self._by_cell = {key: tuple(value) for key, value in grouped.items()}

    def candidates(self, partition: RulePartition, op: BinaryOp) -> tuple[EligiblePair, ...]:
        if partition not in RULE_PARTITIONS:
            raise PartitionValidationError(f"unknown rule partition: {partition!r}")
        if op not in {"all", "any", "exactly_one"}:
            raise PartitionValidationError(f"unknown target binary operation: {op!r}")
        return self._by_cell[(partition, op)]

    @property
    def counts(self) -> dict[str, int]:
        return {
            f"{partition}:{op}": len(self._by_cell[(partition, cast(BinaryOp, op))])
            for partition in RULE_PARTITIONS
            for op in ("all", "any", "exactly_one")
        }

    def as_obj(self) -> dict[str, Any]:
        catalog = build_rule_catalog()
        return {
            "schema_version": ELIGIBLE_PAIR_TABLE_SCHEMA_VERSION,
            "catalog_digest": self.catalog_digest,
            "partitions_digest": self.partitions_digest,
            "same_partition_required": True,
            "minimum_cell_count": MIN_TARGET_SHADOW_CELL_COUNT,
            "minimum_disagreement_count": math.ceil(
                MIN_TARGET_SHADOW_DISAGREEMENT * SCENE_COUNT
            ),
            "maximum_disagreement_count": math.floor(
                MAX_TARGET_SHADOW_DISAGREEMENT * SCENE_COUNT
            ),
            "counts": self.counts,
            "pairs": [pair.as_obj(catalog) for pair in self.pairs],
        }

    @property
    def digest(self) -> str:
        if self._digest is None:
            self._digest = json_digest(
                self.as_obj(), domain="goalzendo-interactive-eligible-pair-table-v1"
            )
        return self._digest


@lru_cache(maxsize=1)
def build_eligible_pair_table() -> EligiblePairTable:
    """Enumerate every same-partition target/shadow pair passing exact bounds."""

    catalog = build_rule_catalog()
    partitions = build_rule_identity_partitions()
    targets = tuple(
        entry for entry in catalog if type(entry.rule) is BinaryRule and not _uses_placard(entry)
    )
    shadows = tuple(
        entry for entry in catalog if type(entry.rule) is RuleLiteral and not _uses_placard(entry)
    )
    pairs: list[EligiblePair] = []
    minimum_disagreement = math.ceil(MIN_TARGET_SHADOW_DISAGREEMENT * SCENE_COUNT)
    maximum_disagreement = math.floor(MAX_TARGET_SHADOW_DISAGREEMENT * SCENE_COUNT)
    for target in targets:
        partition = partitions.for_entry(target)
        for shadow in shadows:
            if partitions.for_entry(shadow) != partition:
                continue
            cells = _cell_counts(target, shadow)
            disagreement = cells[1] + cells[2]
            if min(cells) < MIN_TARGET_SHADOW_CELL_COUNT:
                continue
            if not minimum_disagreement <= disagreement <= maximum_disagreement:
                continue
            order_digest = hashlib.sha256(
                PAIR_ORDER_HASH_DOMAIN
                + target.truth_digest.encode("ascii")
                + b"\0"
                + shadow.truth_digest.encode("ascii")
            ).hexdigest()
            pairs.append(
                EligiblePair(target.index, shadow.index, partition, cells, order_digest)
            )
    pairs.sort(
        key=lambda pair: (
            RULE_PARTITIONS.index(pair.partition),
            pair.order_digest,
            pair.target_index,
            pair.shadow_index,
        )
    )
    return EligiblePairTable(
        tuple(pairs),
        catalog_digest=catalog.digest,
        partitions_digest=partitions.digest,
    )


def serialize_eligible_pair_table(table: EligiblePairTable) -> str:
    if type(table) is not EligiblePairTable:
        raise TypeError("serialize_eligible_pair_table requires EligiblePairTable")
    return dump_json(table.as_obj())


def parse_eligible_pair_table(
    text: str,
    *,
    require_canonical: bool = True,
) -> EligiblePairTable:
    try:
        value = load_json(text)
    except CanonicalJSONError as exc:
        raise PartitionValidationError(str(exc)) from exc
    expected = build_eligible_pair_table()
    if value != expected.as_obj():
        raise PartitionValidationError("eligible-pair manifest differs from deterministic table")
    if require_canonical and serialize_eligible_pair_table(expected) != text:
        raise PartitionValidationError("eligible-pair JSON is valid but not canonical")
    return expected
