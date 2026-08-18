"""Exact truth vectors, semantic deduplication, and deterministic version spaces."""

from __future__ import annotations

import hashlib
import math
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass
from functools import cache, lru_cache
from typing import TypeAlias, overload

from ._json import json_digest
from .rules import (
    ATOMS,
    SYNTACTIC_RULE_COUNT,
    BinaryRule,
    Literal,
    Rule,
    evaluate_rule,
    iter_syntactic_rules,
    rule_sort_key,
)
from .schema import SCENE_COUNT, SCENE_SCHEMA_VERSION, Scene, scene_at, scene_index

TRUTH_VECTOR_SCHEMA_VERSION = 1
RULE_CATALOG_SCHEMA_VERSION = 1
MIN_PREVALENCE = 0.25
MAX_PREVALENCE = 0.75
MIN_TRUE_COUNT = math.ceil(MIN_PREVALENCE * SCENE_COUNT)
MAX_TRUE_COUNT = math.floor(MAX_PREVALENCE * SCENE_COUNT)
_TRUTH_BYTE_COUNT = (SCENE_COUNT + 7) // 8
_UNIVERSE_MASK = (1 << SCENE_COUNT) - 1


@dataclass(frozen=True, slots=True)
class TruthVector:
    """A stable bitset where bit ``i`` is the label of scene index ``i``."""

    bits: int
    scene_count: int = SCENE_COUNT

    def __post_init__(self) -> None:
        if isinstance(self.bits, bool) or not isinstance(self.bits, int) or self.bits < 0:
            raise ValueError("truth-vector bits must be a non-negative integer")
        if self.scene_count != SCENE_COUNT:
            raise ValueError(f"G03 truth vectors require exactly {SCENE_COUNT} scenes")
        if self.bits >> self.scene_count:
            raise ValueError("truth-vector bits extend beyond the scene universe")

    def __len__(self) -> int:
        return self.scene_count

    def __getitem__(self, index: int) -> bool:
        if isinstance(index, bool) or not isinstance(index, int):
            raise TypeError("truth-vector index must be an integer")
        if index < 0:
            index += self.scene_count
        if not 0 <= index < self.scene_count:
            raise IndexError(index)
        return bool((self.bits >> index) & 1)

    def __iter__(self) -> Iterator[bool]:
        return (self[index] for index in range(self.scene_count))

    @property
    def true_count(self) -> int:
        return self.bits.bit_count()

    @property
    def prevalence(self) -> float:
        return self.true_count / self.scene_count

    @property
    def packed(self) -> bytes:
        """Little-endian packed bits, including deterministic zero padding."""

        return self.bits.to_bytes(_TRUTH_BYTE_COUNT, "little")

    @property
    def digest(self) -> str:
        digest = hashlib.sha256()
        digest.update(b"goalzendo-interactive-truth-vector-v1\0")
        digest.update(self.scene_count.to_bytes(8, "big"))
        digest.update(self.packed)
        return digest.hexdigest()

    def labels(self, indices: Iterable[int]) -> tuple[bool, ...]:
        return tuple(self[index] for index in indices)

    @classmethod
    def from_values(cls, values: Iterable[bool]) -> TruthVector:
        bits = 0
        count = 0
        for count, value in enumerate(values, start=1):
            if type(value) is not bool:
                raise ValueError("truth-vector values must be Boolean")
            if value:
                bits |= 1 << (count - 1)
        if count != SCENE_COUNT:
            raise ValueError(f"expected {SCENE_COUNT} truth values, received {count}")
        return cls(bits)


@cache
def atom_truth_vector(atom_index: int) -> TruthVector:
    if isinstance(atom_index, bool) or not isinstance(atom_index, int) or not 0 <= atom_index < len(ATOMS):
        raise IndexError(atom_index)
    atom = ATOMS[atom_index]
    bits = 0
    for index in range(SCENE_COUNT):
        if atom.evaluate(scene_at(index)):
            bits |= 1 << index
    return TruthVector(bits)


_ATOM_INDEX = {atom: index for index, atom in enumerate(ATOMS)}


@cache
def truth_vector(rule: Rule) -> TruthVector:
    """Evaluate a grammar rule exactly over the complete scene universe."""

    if type(rule) is Literal:
        vector = atom_truth_vector(_ATOM_INDEX[rule.atom])
        return TruthVector(_UNIVERSE_MASK ^ vector.bits) if rule.negated else vector
    if type(rule) is not BinaryRule:
        raise TypeError("truth_vector requires a Literal or BinaryRule")
    left, right = (truth_vector(literal).bits for literal in rule.args)
    if rule.op == "all":
        bits = left & right
    elif rule.op == "any":
        bits = left | right
    else:
        bits = left ^ right
    return TruthVector(bits)


@dataclass(frozen=True, slots=True)
class CatalogEntry:
    index: int
    rule: Rule
    truth: TruthVector

    @property
    def rule_id(self) -> str:
        return f"g03r{self.index:05d}"

    @property
    def truth_digest(self) -> str:
        return self.truth.digest

    @property
    def prevalence(self) -> float:
        return self.truth.prevalence


@dataclass(frozen=True, slots=True)
class CatalogStats:
    syntactic_rule_count: int
    excluded_prevalence_count: int
    extensional_duplicate_count: int
    retained_rule_count: int

    def as_obj(self) -> dict[str, int]:
        return {
            "syntactic_rule_count": self.syntactic_rule_count,
            "excluded_prevalence_count": self.excluded_prevalence_count,
            "extensional_duplicate_count": self.extensional_duplicate_count,
            "retained_rule_count": self.retained_rule_count,
        }


Observation: TypeAlias = tuple[int | Scene, bool]


class RuleCatalog(Sequence[CatalogEntry]):
    """Immutable exact catalog with deterministic semantic lookup."""

    __slots__ = ("_by_bits", "_by_digest", "_entries", "_stats")

    def __init__(self, entries: tuple[CatalogEntry, ...], stats: CatalogStats) -> None:
        self._entries = entries
        self._stats = stats
        self._by_bits = {entry.truth.bits: entry.index for entry in entries}
        self._by_digest = {entry.truth_digest: entry.index for entry in entries}
        if len(self._by_bits) != len(entries) or len(self._by_digest) != len(entries):
            raise ValueError("catalog entries must have unique truth vectors and digests")

    def __len__(self) -> int:
        return len(self._entries)

    @overload
    def __getitem__(self, index: int) -> CatalogEntry: ...

    @overload
    def __getitem__(self, index: slice) -> tuple[CatalogEntry, ...]: ...

    def __getitem__(self, index: int | slice) -> CatalogEntry | tuple[CatalogEntry, ...]:
        return self._entries[index]

    def __iter__(self) -> Iterator[CatalogEntry]:
        return iter(self._entries)

    @property
    def stats(self) -> CatalogStats:
        return self._stats

    @property
    def digest(self) -> str:
        return json_digest(
            {
                "schema_version": RULE_CATALOG_SCHEMA_VERSION,
                "scene_schema_version": SCENE_SCHEMA_VERSION,
                "scene_count": SCENE_COUNT,
                "min_true_count": MIN_TRUE_COUNT,
                "max_true_count": MAX_TRUE_COUNT,
                "stats": self.stats.as_obj(),
                "entries": [
                    {
                        "rule_id": entry.rule_id,
                        "rule": entry.rule.as_obj(),
                        "truth_digest": entry.truth_digest,
                        "true_count": entry.truth.true_count,
                    }
                    for entry in self._entries
                ],
            },
            domain="goalzendo-interactive-rule-catalog-v1",
        )

    def equivalent_entry(self, rule: Rule) -> CatalogEntry | None:
        index = self._by_bits.get(truth_vector(rule).bits)
        return None if index is None else self._entries[index]

    def by_truth_digest(self, digest: str) -> CatalogEntry:
        try:
            return self._entries[self._by_digest[digest]]
        except KeyError as exc:
            raise KeyError(f"unknown truth-vector digest: {digest}") from exc

    def version_space(self, observations: Iterable[Observation] = ()) -> VersionSpace:
        return VersionSpace(self, tuple(range(len(self)))).observe_many(observations)


@dataclass(frozen=True, slots=True)
class VersionSpace:
    """A sorted immutable subset of a rule catalog after exact observations."""

    catalog: RuleCatalog
    indices: tuple[int, ...]

    def __post_init__(self) -> None:
        if any(isinstance(index, bool) or not isinstance(index, int) for index in self.indices):
            raise ValueError("version-space indices must be integers")
        if tuple(sorted(set(self.indices))) != self.indices:
            raise ValueError("version-space indices must be strictly increasing and unique")
        if self.indices and (self.indices[0] < 0 or self.indices[-1] >= len(self.catalog)):
            raise ValueError("version-space index lies outside its catalog")

    def __len__(self) -> int:
        return len(self.indices)

    def __iter__(self) -> Iterator[CatalogEntry]:
        return (self.catalog[index] for index in self.indices)

    def observe(self, scene: int | Scene, accepted: bool) -> VersionSpace:
        if type(accepted) is not bool:
            raise ValueError("observed label must be Boolean")
        index = scene_index(scene) if type(scene) is Scene else scene
        if isinstance(index, bool) or not isinstance(index, int) or not 0 <= index < SCENE_COUNT:
            raise IndexError(f"scene index must lie in [0, {SCENE_COUNT})")
        return VersionSpace(
            self.catalog,
            tuple(
                rule_index
                for rule_index in self.indices
                if self.catalog[rule_index].truth[index] is accepted
            ),
        )

    def observe_many(self, observations: Iterable[Observation]) -> VersionSpace:
        result = self
        for scene, accepted in observations:
            result = result.observe(scene, accepted)
        return result

    def label_counts(self, scene: int | Scene) -> tuple[int, int]:
        index = scene_index(scene) if type(scene) is Scene else scene
        if isinstance(index, bool) or not isinstance(index, int) or not 0 <= index < SCENE_COUNT:
            raise IndexError(f"scene index must lie in [0, {SCENE_COUNT})")
        accepted = sum(self.catalog[rule_index].truth[index] for rule_index in self.indices)
        return len(self) - accepted, accepted


@lru_cache(maxsize=1)
def build_rule_catalog() -> RuleCatalog:
    """Build the complete prevalence-filtered, extensionally unique catalog."""

    candidates = sorted(iter_syntactic_rules(), key=rule_sort_key)
    if len(candidates) != SYNTACTIC_RULE_COUNT:
        raise RuntimeError(
            f"grammar generated {len(candidates)} rules, expected {SYNTACTIC_RULE_COUNT}"
        )

    retained: list[tuple[Rule, TruthVector]] = []
    seen_bits: set[int] = set()
    excluded_prevalence = 0
    extensional_duplicates = 0
    for rule in candidates:
        vector = truth_vector(rule)
        if not MIN_TRUE_COUNT <= vector.true_count <= MAX_TRUE_COUNT:
            excluded_prevalence += 1
            continue
        if vector.bits in seen_bits:
            extensional_duplicates += 1
            continue
        seen_bits.add(vector.bits)
        retained.append((rule, vector))

    entries = tuple(
        CatalogEntry(index=index, rule=rule, truth=vector)
        for index, (rule, vector) in enumerate(retained)
    )
    stats = CatalogStats(
        syntactic_rule_count=len(candidates),
        excluded_prevalence_count=excluded_prevalence,
        extensional_duplicate_count=extensional_duplicates,
        retained_rule_count=len(entries),
    )
    if sum(
        (
            stats.excluded_prevalence_count,
            stats.extensional_duplicate_count,
            stats.retained_rule_count,
        )
    ) != stats.syntactic_rule_count:
        raise RuntimeError("catalog accounting invariant failed")
    return RuleCatalog(entries, stats)


def verify_truth_vector(rule: Rule, vector: TruthVector) -> bool:
    """Independent slow-path semantic check useful for engine audits."""

    return all(vector[index] is evaluate_rule(rule, scene_at(index)) for index in range(SCENE_COUNT))
