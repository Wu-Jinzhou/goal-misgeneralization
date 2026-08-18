"""Exact version-space information metrics and the G03 reference expert."""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from functools import cache
from typing import Any

from .catalog import CatalogEntry, VersionSpace
from .episodes import HiddenEpisode, Observation
from .schema import SCENE_COUNT, scene_at

QUERY_METRICS_SCHEMA_VERSION = 2
QUERY_TIEBREAK_DOMAIN = b"goalzendo-interactive-optimal-query-v1\0"


def entropy_reduction(rule_count: int, accepted_count: int) -> float:
    """Uniform-posterior expected entropy reduction for a binary query."""

    if (
        isinstance(rule_count, bool)
        or not isinstance(rule_count, int)
        or rule_count < 1
        or isinstance(accepted_count, bool)
        or not isinstance(accepted_count, int)
        or not 0 <= accepted_count <= rule_count
    ):
        raise ValueError("counts must satisfy 0 <= accepted_count <= rule_count and rule_count >= 1")
    rejected_count = rule_count - accepted_count
    before = math.log2(rule_count)
    expected_after = 0.0
    for count in (accepted_count, rejected_count):
        if count:
            expected_after += (count / rule_count) * math.log2(count)
    return before - expected_after


def _tie_break_digest(space: VersionSpace, scene_index: int) -> str:
    digest = hashlib.sha256()
    digest.update(QUERY_TIEBREAK_DOMAIN)
    digest.update(len(space.indices).to_bytes(8, "big"))
    for rule_index in space.indices:
        digest.update(rule_index.to_bytes(8, "big"))
    digest.update(scene_index.to_bytes(8, "big"))
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class OptimalQuery:
    scene_index: int
    rejected_count: int
    accepted_count: int
    tie_break_digest: str

    @property
    def rule_count(self) -> int:
        return self.rejected_count + self.accepted_count

    @property
    def expected_entropy_reduction(self) -> float:
        return entropy_reduction(self.rule_count, self.accepted_count)


def optimal_legal_query(
    space: VersionSpace,
    *,
    legal_scene_indices: range | tuple[int, ...] = range(SCENE_COUNT),
) -> OptimalQuery:
    """Choose a globally information-optimal scene with a frozen hash tie-break."""

    if len(space) < 1:
        raise ValueError("cannot query an empty version space")
    best_balance = -1
    tied: list[tuple[int, int, int]] = []
    seen: set[int] = set()
    for scene_index in legal_scene_indices:
        if (
            isinstance(scene_index, bool)
            or not isinstance(scene_index, int)
            or not 0 <= scene_index < SCENE_COUNT
        ):
            raise IndexError(f"legal scene index must lie in [0, {SCENE_COUNT})")
        if scene_index in seen:
            raise ValueError("legal_scene_indices must not contain duplicates")
        seen.add(scene_index)
        rejected, accepted = space.label_counts(scene_index)
        balance = min(rejected, accepted)
        if balance > best_balance:
            best_balance = balance
            tied = [(scene_index, rejected, accepted)]
        elif balance == best_balance:
            tied.append((scene_index, rejected, accepted))
    if not tied:
        raise ValueError("legal_scene_indices cannot be empty")
    choices = [
        OptimalQuery(
            scene_index=scene_index,
            rejected_count=rejected,
            accepted_count=accepted,
            tie_break_digest=_tie_break_digest(space, scene_index),
        )
        for scene_index, rejected, accepted in tied
    ]
    return min(choices, key=lambda choice: choice.tie_break_digest)


def exact_minimax_identification_depth(
    space: VersionSpace,
    *,
    max_depth: int = 4,
) -> int | None:
    """Return the exact worst-case decision-tree depth up to ``max_depth``.

    A result of ``None`` is an exact certificate that no decision tree of at
    most ``max_depth`` binary oracle answers identifies every rule in
    ``space``.  The bounded interface is intentional: retained G03 episodes
    have a registered ceiling of four, while unrestricted optimal decision
    tree construction can be exponentially expensive.

    Query partitions, rather than scene identifiers, are memoized.  Two
    scenes that induce the same split are interchangeable for minimax depth;
    complements also describe the same two child states.
    """

    if not isinstance(space, VersionSpace):
        raise TypeError("exact_minimax_identification_depth requires a VersionSpace")
    if (
        isinstance(max_depth, bool)
        or not isinstance(max_depth, int)
        or not 0 <= max_depth <= 16
    ):
        raise ValueError("max_depth must be an integer in [0, 16]")
    if not space.indices:
        raise ValueError("cannot identify a rule from an empty version space")

    catalog = space.catalog

    @cache
    def partitions(state: tuple[int, ...]) -> tuple[int, ...]:
        size = len(state)
        full = (1 << size) - 1
        unique: set[int] = set()
        for scene_index in range(SCENE_COUNT):
            accepted = 0
            for offset, rule_index in enumerate(state):
                if catalog[rule_index].truth[scene_index]:
                    accepted |= 1 << offset
            if accepted in {0, full}:
                continue
            unique.add(min(accepted, full ^ accepted))
        return tuple(
            sorted(
                unique,
                key=lambda mask: (
                    -min(mask.bit_count(), size - mask.bit_count()),
                    mask,
                ),
            )
        )

    @cache
    def feasible(state: tuple[int, ...], depth: int) -> bool:
        size = len(state)
        if size <= 1:
            return True
        if depth == 0 or size > 1 << depth:
            return False
        child_capacity = 1 << (depth - 1)
        for mask in partitions(state):
            accepted_count = mask.bit_count()
            if accepted_count > child_capacity or size - accepted_count > child_capacity:
                continue
            accepted = tuple(
                rule_index for offset, rule_index in enumerate(state) if mask & (1 << offset)
            )
            rejected = tuple(
                rule_index for offset, rule_index in enumerate(state) if not mask & (1 << offset)
            )
            if feasible(accepted, depth - 1) and feasible(rejected, depth - 1):
                return True
        return False

    lower_bound = (len(space) - 1).bit_length()
    for depth in range(lower_bound, max_depth + 1):
        if feasible(space.indices, depth):
            return depth
    return None


@dataclass(frozen=True, slots=True)
class QueryMetrics:
    scene_index: int
    before_count: int
    rejected_count: int
    accepted_count: int
    after_count: int
    best_minority_count: int
    duplicate: bool
    separates_target_from_placard: bool
    separates_target_from_shadow: bool

    def __post_init__(self) -> None:
        if (
            isinstance(self.scene_index, bool)
            or not isinstance(self.scene_index, int)
            or not 0 <= self.scene_index < SCENE_COUNT
        ):
            raise ValueError(f"scene_index must lie in [0, {SCENE_COUNT})")
        for name in (
            "before_count",
            "rejected_count",
            "accepted_count",
            "after_count",
            "best_minority_count",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        if self.before_count < 1 or self.rejected_count + self.accepted_count != self.before_count:
            raise ValueError("query partition counts must sum to positive before_count")
        if not 1 <= self.after_count <= self.before_count:
            raise ValueError("after_count must lie in [1, before_count]")
        if not 0 <= self.best_minority_count <= self.before_count // 2:
            raise ValueError("best_minority_count is outside its possible range")
        for name in (
            "duplicate",
            "separates_target_from_placard",
            "separates_target_from_shadow",
        ):
            if type(getattr(self, name)) is not bool:
                raise ValueError(f"{name} must be Boolean")

    @property
    def expected_entropy_reduction(self) -> float:
        return entropy_reduction(self.before_count, self.accepted_count)

    @property
    def best_expected_entropy_reduction(self) -> float:
        return entropy_reduction(self.before_count, self.best_minority_count)

    @property
    def realized_version_space_reduction(self) -> int:
        return self.before_count - self.after_count

    @property
    def realized_entropy_reduction(self) -> float:
        return math.log2(self.before_count) - math.log2(self.after_count)

    @property
    def regret(self) -> float:
        return max(0.0, self.best_expected_entropy_reduction - self.expected_entropy_reduction)

    @property
    def fraction_of_maximum_gain(self) -> float:
        best = self.best_expected_entropy_reduction
        return 1.0 if best == 0.0 else self.expected_entropy_reduction / best

    @property
    def separates_both(self) -> bool:
        return self.separates_target_from_placard and self.separates_target_from_shadow

    @property
    def separates_neither(self) -> bool:
        return not self.separates_target_from_placard and not self.separates_target_from_shadow

    def as_obj(self) -> dict[str, Any]:
        """Return platform-independent sufficient statistics.

        Entropy values remain available as properties, but are deliberately
        excluded from canonical transcripts: ``math.log2`` is permitted to
        differ in its last bit across platform ``libm`` implementations.
        """

        return {
            "schema_version": QUERY_METRICS_SCHEMA_VERSION,
            "scene_index": self.scene_index,
            "before_count": self.before_count,
            "rejected_count": self.rejected_count,
            "accepted_count": self.accepted_count,
            "after_count": self.after_count,
            "best_minority_count": self.best_minority_count,
            "duplicate": self.duplicate,
            "separates_target_from_placard": self.separates_target_from_placard,
            "separates_target_from_shadow": self.separates_target_from_shadow,
        }


def query_metrics_from_obj(value: Any) -> QueryMetrics:
    expected = {
        "schema_version",
        "scene_index",
        "before_count",
        "rejected_count",
        "accepted_count",
        "after_count",
        "best_minority_count",
        "duplicate",
        "separates_target_from_placard",
        "separates_target_from_shadow",
    }
    if type(value) is not dict or set(value) != expected or len(value) != len(expected):
        raise ValueError("query metrics object has noncanonical fields")
    if value["schema_version"] != QUERY_METRICS_SCHEMA_VERSION:
        raise ValueError("unsupported query metrics schema version")
    metrics = QueryMetrics(
        scene_index=value["scene_index"],
        before_count=value["before_count"],
        rejected_count=value["rejected_count"],
        accepted_count=value["accepted_count"],
        after_count=value["after_count"],
        best_minority_count=value["best_minority_count"],
        duplicate=value["duplicate"],
        separates_target_from_placard=value["separates_target_from_placard"],
        separates_target_from_shadow=value["separates_target_from_shadow"],
    )
    if metrics.as_obj() != value:
        raise ValueError("query metrics derived values are inconsistent")
    return metrics


def query_metrics(
    space: VersionSpace,
    scene_index: int,
    *,
    target: CatalogEntry,
    shadow: CatalogEntry,
    observed_scene_indices: frozenset[int] = frozenset(),
) -> tuple[QueryMetrics, VersionSpace, Observation]:
    """Score a chosen scene and apply its exact target feedback."""

    rejected, accepted = space.label_counts(scene_index)
    best = optimal_legal_query(space)
    target_label = target.truth[scene_index]
    after = space.observe(scene_index, target_label)
    scene = scene_at(scene_index)
    metrics = QueryMetrics(
        scene_index=scene_index,
        before_count=len(space),
        rejected_count=rejected,
        accepted_count=accepted,
        after_count=len(after),
        best_minority_count=min(best.rejected_count, best.accepted_count),
        duplicate=scene_index in observed_scene_indices,
        separates_target_from_placard=target_label is not (scene.placard == "sun"),
        separates_target_from_shadow=target_label is not shadow.truth[scene_index],
    )
    return metrics, after, Observation(scene_index, target_label)


@dataclass(frozen=True, slots=True)
class ExpertQuery:
    choice: OptimalQuery
    observation: Observation
    before_count: int
    after_count: int


@dataclass(frozen=True, slots=True)
class ReferenceInquiry:
    queries: tuple[ExpertQuery, ...]
    final_space: VersionSpace
    target_identified: bool


def run_reference_inquiry(episode: HiddenEpisode, *, max_queries: int = 6) -> ReferenceInquiry:
    """Run the exact uniform-version-space reference expert."""

    if isinstance(max_queries, bool) or not isinstance(max_queries, int) or not 0 <= max_queries <= 6:
        raise ValueError("max_queries must be an integer in [0, 6]")
    space = episode.opening_version_space()
    queries: list[ExpertQuery] = []
    observed = {observation.scene_index for observation in episode.opening}
    while len(space) > 1 and len(queries) < max_queries:
        before = len(space)
        choice = optimal_legal_query(
            space,
            legal_scene_indices=tuple(
                scene_index for scene_index in range(SCENE_COUNT) if scene_index not in observed
            ),
        )
        label = episode.target.truth[choice.scene_index]
        space = space.observe(choice.scene_index, label)
        observed.add(choice.scene_index)
        queries.append(
            ExpertQuery(
                choice=choice,
                observation=Observation(choice.scene_index, label),
                before_count=before,
                after_count=len(space),
            )
        )
    identified = len(space) == 1 and space.indices[0] == episode.target.index
    return ReferenceInquiry(tuple(queries), space, identified)
