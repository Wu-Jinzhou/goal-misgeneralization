"""Bounded deterministic construction of auditable G03 episode banks."""

from __future__ import annotations

import hashlib
import heapq
from collections import Counter
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, cast

from ._json import CanonicalJSONError, dump_json, json_digest, load_json
from .catalog import CatalogEntry, RuleCatalog, build_rule_catalog
from .episodes import (
    OPENING_REGIMES,
    TERMINAL_KINDS,
    HiddenEpisode,
    Observation,
    OpeningRegime,
    TerminalKind,
    hidden_episode_from_obj,
)
from .partitions import (
    RULE_PARTITIONS,
    EligiblePair,
    RulePartition,
    build_eligible_pair_table,
    build_rule_identity_partitions,
)
from .query import exact_minimax_identification_depth, run_reference_inquiry
from .rendering import RENDERERS, RendererName
from .rules import BINARY_OPS, BinaryOp, BinaryRule
from .schema import NONEMPTY_ARRANGEMENT_COUNT, SCENE_COUNT

EPISODE_BANK_SCHEMA_VERSION = 1
EPISODE_GENERATOR_SCHEMA_VERSION = 1
OPENING_ORDER_DOMAIN = b"goalzendo-interactive-opening-order-v1\0"
SCENE_ORDER_DOMAIN = b"goalzendo-interactive-scene-order-v1\0"
REQUEST_PAIR_ORDER_DOMAIN = b"goalzendo-interactive-request-pair-order-v1\0"


class EpisodeGenerationError(RuntimeError):
    """A bounded deterministic search exhausted its registered attempts."""


def _valid_digest(value: object) -> bool:
    return (
        type(value) is str
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


@dataclass(frozen=True, slots=True)
class EpisodeRequest:
    request_id: str
    partition: RulePartition
    target_op: BinaryOp
    regime: OpeningRegime
    terminal_kind: TerminalKind
    renderer: RendererName
    noisy_placard_error_target: bool | None = None

    def __post_init__(self) -> None:
        if type(self.request_id) is not str or not self.request_id.strip():
            raise ValueError("episode request id cannot be empty")
        if not self.request_id.isascii():
            raise ValueError("episode request id must contain only ASCII characters")
        if self.partition not in RULE_PARTITIONS:
            raise ValueError(f"unknown rule partition: {self.partition!r}")
        if self.target_op not in BINARY_OPS:
            raise ValueError(f"unknown target operation: {self.target_op!r}")
        if self.regime not in OPENING_REGIMES:
            raise ValueError(f"unknown opening regime: {self.regime!r}")
        if self.terminal_kind not in TERMINAL_KINDS:
            raise ValueError(f"unknown terminal kind: {self.terminal_kind!r}")
        if self.renderer not in RENDERERS:
            raise ValueError(f"unknown renderer: {self.renderer!r}")
        if self.regime == "perfect_ambiguity":
            if self.noisy_placard_error_target is not None:
                raise ValueError("perfect-ambiguity requests have no placard error target")
        elif type(self.noisy_placard_error_target) is not bool:
            raise ValueError("noisy requests must register the placard-error target class")

    def as_obj(self) -> dict[str, Any]:
        return {
            "request_id": self.request_id,
            "partition": self.partition,
            "target_op": self.target_op,
            "regime": self.regime,
            "terminal_kind": self.terminal_kind,
            "renderer": self.renderer,
            "noisy_placard_error_target": self.noisy_placard_error_target,
        }


def episode_request_from_obj(value: Any) -> EpisodeRequest:
    expected = {
        "request_id",
        "partition",
        "target_op",
        "regime",
        "terminal_kind",
        "renderer",
        "noisy_placard_error_target",
    }
    if type(value) is not dict or set(value) != expected or len(value) != len(expected):
        raise ValueError("episode request has noncanonical fields")
    result = EpisodeRequest(
        request_id=value["request_id"],
        partition=cast(RulePartition, value["partition"]),
        target_op=cast(BinaryOp, value["target_op"]),
        regime=cast(OpeningRegime, value["regime"]),
        terminal_kind=cast(TerminalKind, value["terminal_kind"]),
        renderer=cast(RendererName, value["renderer"]),
        noisy_placard_error_target=value["noisy_placard_error_target"],
    )
    if result.as_obj() != value:
        raise ValueError("episode request is valid but not canonical")
    return result


@dataclass(frozen=True, slots=True)
class EpisodeBankSpec:
    bank_id: str
    requests: tuple[EpisodeRequest, ...]
    candidate_pool_size: int = 128
    max_pair_attempts: int = 48
    opening_attempts_per_pair: int = 4
    minimum_version_size: int = 8
    maximum_version_size: int = 64
    maximum_minimax_depth: int = 4
    maximum_reference_queries: int = 6

    def __post_init__(self) -> None:
        if type(self.bank_id) is not str or not self.bank_id.strip():
            raise ValueError("episode bank id cannot be empty")
        requests = tuple(self.requests)
        object.__setattr__(self, "requests", requests)
        if not requests or any(type(request) is not EpisodeRequest for request in requests):
            raise ValueError("episode bank requires at least one EpisodeRequest")
        request_ids = [request.request_id for request in requests]
        if len(set(request_ids)) != len(request_ids):
            raise ValueError("episode request ids must be unique")
        integer_fields = {
            "candidate_pool_size": (self.candidate_pool_size, 16, 1024),
            "max_pair_attempts": (self.max_pair_attempts, 1, 4096),
            "opening_attempts_per_pair": (self.opening_attempts_per_pair, 1, 64),
            "minimum_version_size": (self.minimum_version_size, 1, 64),
            "maximum_version_size": (self.maximum_version_size, 1, 64),
            "maximum_minimax_depth": (self.maximum_minimax_depth, 0, 8),
            "maximum_reference_queries": (self.maximum_reference_queries, 0, 6),
        }
        for name, (value, lower, upper) in integer_fields.items():
            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or not lower <= value <= upper
            ):
                raise ValueError(f"{name} must lie in [{lower}, {upper}]")
        if self.minimum_version_size > self.maximum_version_size:
            raise ValueError("minimum version size cannot exceed maximum")

    def as_obj(self) -> dict[str, Any]:
        return {
            "bank_id": self.bank_id,
            "requests": [request.as_obj() for request in self.requests],
            "candidate_pool_size": self.candidate_pool_size,
            "max_pair_attempts": self.max_pair_attempts,
            "opening_attempts_per_pair": self.opening_attempts_per_pair,
            "minimum_version_size": self.minimum_version_size,
            "maximum_version_size": self.maximum_version_size,
            "maximum_minimax_depth": self.maximum_minimax_depth,
            "maximum_reference_queries": self.maximum_reference_queries,
        }


def episode_bank_spec_from_obj(value: Any) -> EpisodeBankSpec:
    expected = {
        "bank_id",
        "requests",
        "candidate_pool_size",
        "max_pair_attempts",
        "opening_attempts_per_pair",
        "minimum_version_size",
        "maximum_version_size",
        "maximum_minimax_depth",
        "maximum_reference_queries",
    }
    if type(value) is not dict or set(value) != expected or len(value) != len(expected):
        raise ValueError("episode bank spec has noncanonical fields")
    if type(value["requests"]) is not list:
        raise ValueError("episode bank requests must be an array")
    result = EpisodeBankSpec(
        bank_id=value["bank_id"],
        requests=tuple(episode_request_from_obj(item) for item in value["requests"]),
        candidate_pool_size=value["candidate_pool_size"],
        max_pair_attempts=value["max_pair_attempts"],
        opening_attempts_per_pair=value["opening_attempts_per_pair"],
        minimum_version_size=value["minimum_version_size"],
        maximum_version_size=value["maximum_version_size"],
        maximum_minimax_depth=value["maximum_minimax_depth"],
        maximum_reference_queries=value["maximum_reference_queries"],
    )
    if result.as_obj() != value:
        raise ValueError("episode bank spec is valid but not canonical")
    return result


@dataclass(frozen=True, slots=True)
class EpisodeGenerationRecord:
    request_id: str
    pair_rank: int
    opening_attempt: int
    version_space_size: int
    minimax_depth: int
    reference_query_count: int

    def __post_init__(self) -> None:
        if type(self.request_id) is not str or not self.request_id:
            raise ValueError("generation record request id cannot be empty")
        for name in (
            "pair_rank",
            "opening_attempt",
            "version_space_size",
            "minimax_depth",
            "reference_query_count",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")

    def as_obj(self) -> dict[str, int | str]:
        return {
            "request_id": self.request_id,
            "pair_rank": self.pair_rank,
            "opening_attempt": self.opening_attempt,
            "version_space_size": self.version_space_size,
            "minimax_depth": self.minimax_depth,
            "reference_query_count": self.reference_query_count,
        }


def generation_record_from_obj(value: Any) -> EpisodeGenerationRecord:
    expected = {
        "request_id",
        "pair_rank",
        "opening_attempt",
        "version_space_size",
        "minimax_depth",
        "reference_query_count",
    }
    if type(value) is not dict or set(value) != expected or len(value) != len(expected):
        raise ValueError("episode generation record has noncanonical fields")
    result = EpisodeGenerationRecord(**value)
    if result.as_obj() != value:
        raise ValueError("episode generation record is valid but not canonical")
    return result


@dataclass(frozen=True, slots=True)
class EpisodeBank:
    spec: EpisodeBankSpec
    catalog_digest: str
    partitions_digest: str
    eligible_pairs_digest: str
    episodes: tuple[HiddenEpisode, ...]
    generation_records: tuple[EpisodeGenerationRecord, ...]

    def __post_init__(self) -> None:
        if type(self.spec) is not EpisodeBankSpec:
            raise ValueError("episode bank requires an EpisodeBankSpec")
        for name in ("catalog_digest", "partitions_digest", "eligible_pairs_digest"):
            if not _valid_digest(getattr(self, name)):
                raise ValueError(f"{name} must be a SHA-256 digest")
        episodes = tuple(self.episodes)
        records = tuple(self.generation_records)
        object.__setattr__(self, "episodes", episodes)
        object.__setattr__(self, "generation_records", records)
        if len(episodes) != len(self.spec.requests) or len(records) != len(episodes):
            raise ValueError("episode bank requests, episodes, and records must align one-to-one")
        if any(type(episode) is not HiddenEpisode for episode in episodes):
            raise ValueError("episode bank contains a non-HiddenEpisode")
        if any(type(record) is not EpisodeGenerationRecord for record in records):
            raise ValueError("episode bank contains a non-generation record")
        catalog = build_rule_catalog()
        partitions = build_rule_identity_partitions()
        pair_table = build_eligible_pair_table()
        if self.catalog_digest != catalog.digest:
            raise ValueError("episode bank catalog digest differs from the current catalog")
        if self.partitions_digest != partitions.digest:
            raise ValueError("episode bank partition digest differs from the current mapping")
        if self.eligible_pairs_digest != pair_table.digest:
            raise ValueError("episode bank pair-table digest differs from the current table")
        target_digests = [episode.target.truth_digest for episode in episodes]
        if len(set(target_digests)) != len(target_digests):
            raise ValueError("episode bank target rule identities must be disjoint")
        for request, episode, record in zip(
            self.spec.requests, episodes, records, strict=True
        ):
            target_rule = cast(BinaryRule, episode.target.rule)
            if (
                record.request_id != request.request_id
                or episode.regime != request.regime
                or episode.terminal_kind != request.terminal_kind
                or episode.renderer != request.renderer
                or target_rule.op != request.target_op
                or partitions.for_entry(episode.target) != request.partition
                or partitions.for_entry(episode.shadow) != request.partition
            ):
                raise ValueError("generated episode does not satisfy its request")
            if len(episode.opening_version_space()) != record.version_space_size:
                raise ValueError("generation record version-space size mismatch")

    @property
    def formula_counts(self) -> dict[str, int]:
        counts = Counter(cast(BinaryRule, episode.target.rule).op for episode in self.episodes)
        return {op: counts[op] for op in BINARY_OPS}

    @property
    def partition_counts(self) -> dict[str, int]:
        counts = Counter(request.partition for request in self.spec.requests)
        return {partition: counts[partition] for partition in RULE_PARTITIONS}

    def as_obj(self) -> dict[str, Any]:
        return {
            "schema_version": EPISODE_BANK_SCHEMA_VERSION,
            "generator_schema_version": EPISODE_GENERATOR_SCHEMA_VERSION,
            "spec": self.spec.as_obj(),
            "catalog_digest": self.catalog_digest,
            "partitions_digest": self.partitions_digest,
            "eligible_pairs_digest": self.eligible_pairs_digest,
            "formula_counts": self.formula_counts,
            "partition_counts": self.partition_counts,
            "episodes": [episode.as_obj() for episode in self.episodes],
            "generation_records": [record.as_obj() for record in self.generation_records],
        }

    @property
    def digest(self) -> str:
        return json_digest(self.as_obj(), domain="goalzendo-interactive-episode-bank-v1")


def _is_sun(scene_index: int) -> bool:
    return scene_index < NONEMPTY_ARRANGEMENT_COUNT


def _cell_number(target: CatalogEntry, shadow: CatalogEntry, scene_index: int) -> int:
    return (
        int(target.truth[scene_index]) * 4
        + int(_is_sun(scene_index)) * 2
        + int(shadow.truth[scene_index])
    )


def _scene_pools(
    target: CatalogEntry,
    shadow: CatalogEntry,
) -> tuple[tuple[int, ...], ...]:
    pools: list[list[int]] = [[] for _ in range(8)]
    for scene_index in range(SCENE_COUNT):
        pools[_cell_number(target, shadow, scene_index)].append(scene_index)
    return tuple(tuple(pool) for pool in pools)


def _cell_counts(
    regime: OpeningRegime,
    noisy_placard_error_target: bool | None,
) -> dict[int, int]:
    if regime == "perfect_ambiguity":
        return {0b000: 5, 0b111: 5}
    if noisy_placard_error_target is False:
        return {0b000: 3, 0b001: 1, 0b010: 1, 0b111: 4, 0b110: 1}
    return {0b000: 4, 0b001: 1, 0b111: 3, 0b110: 1, 0b101: 1}


def _hash_parts(domain: bytes, *parts: str) -> bytes:
    digest = hashlib.sha256()
    digest.update(domain)
    for part in parts:
        digest.update(part.encode("ascii"))
        digest.update(b"\0")
    return digest.digest()


def _ordered_cells(
    request: EpisodeRequest,
    pair: EligiblePair,
    opening_attempt: int,
) -> tuple[int, ...]:
    expanded = [
        (cell, occurrence)
        for cell, count in _cell_counts(
            request.regime, request.noisy_placard_error_target
        ).items()
        for occurrence in range(count)
    ]
    expanded.sort(
        key=lambda item: _hash_parts(
            OPENING_ORDER_DOMAIN,
            request.request_id,
            pair.order_digest,
            str(opening_attempt),
            str(item[0]),
            str(item[1]),
        )
    )
    return tuple(cell for cell, _ in expanded)


def _ranked_pool(
    pool: tuple[int, ...],
    *,
    request: EpisodeRequest,
    pair: EligiblePair,
    opening_attempt: int,
    cell: int,
    limit: int,
) -> tuple[int, ...]:
    return tuple(
        heapq.nsmallest(
            limit,
            pool,
            key=lambda scene_index: _hash_parts(
                SCENE_ORDER_DOMAIN,
                request.request_id,
                pair.order_digest,
                str(opening_attempt),
                str(cell),
                str(scene_index),
            ),
        )
    )


def _greedy_opening(
    request: EpisodeRequest,
    pair: EligiblePair,
    *,
    pools: tuple[tuple[int, ...], ...],
    catalog: RuleCatalog,
    spec: EpisodeBankSpec,
    opening_attempt: int,
) -> tuple[Observation, ...] | None:
    target = catalog[pair.target_index]
    ordered_pools = {
        cell: _ranked_pool(
            pools[cell],
            request=request,
            pair=pair,
            opening_attempt=opening_attempt,
            cell=cell,
            limit=spec.candidate_pool_size + 10,
        )
        for cell in sorted(set(_ordered_cells(request, pair, opening_attempt)))
    }
    survivors = tuple(range(len(catalog)))
    selected: list[int] = []
    for cell in _ordered_cells(request, pair, opening_attempt):
        target_label = bool(cell & 0b100)
        candidates = tuple(
            scene_index
            for scene_index in ordered_pools[cell]
            if scene_index not in selected
        )[: spec.candidate_pool_size]
        best: tuple[int, int, tuple[int, ...]] | None = None
        for rank, scene_index in enumerate(candidates):
            after = tuple(
                rule_index
                for rule_index in survivors
                if catalog[rule_index].truth[scene_index] is target_label
            )
            size = len(after)
            if size < spec.minimum_version_size:
                continue
            choice = (size, rank, after)
            if best is None or choice[:2] < best[:2]:
                best = choice
        if best is None:
            return None
        survivors = best[2]
        selected.append(candidates[best[1]])
    if not spec.minimum_version_size <= len(survivors) <= spec.maximum_version_size:
        return None
    return tuple(Observation(index, target.truth[index]) for index in selected)


def _terminal(
    request: EpisodeRequest,
    pair: EligiblePair,
    *,
    pools: tuple[tuple[int, ...], ...],
    catalog: RuleCatalog,
    opening: tuple[Observation, ...],
) -> tuple[Observation, ...]:
    target = catalog[pair.target_index]
    excluded = {observation.scene_index for observation in opening}
    counts = (
        {cell: 2 for cell in range(8)}
        if request.terminal_kind == "factorial"
        else _cell_counts(request.regime, request.noisy_placard_error_target)
    )
    selected: list[int] = []
    for cell, count in sorted(counts.items()):
        candidates = heapq.nsmallest(
            count,
            (
                scene_index
                for scene_index in pools[cell]
                if scene_index not in excluded
            ),
            key=lambda scene_index: _hash_parts(
                SCENE_ORDER_DOMAIN,
                request.request_id,
                pair.order_digest,
                "terminal",
                str(cell),
                str(scene_index),
            ),
        )
        if len(candidates) < count:
            raise EpisodeGenerationError("eligible pair lacks enough disjoint terminal scenes")
        chosen = candidates
        selected.extend(chosen)
        excluded.update(chosen)
    selected.sort(
        key=lambda scene_index: _hash_parts(
            SCENE_ORDER_DOMAIN,
            request.request_id,
            pair.order_digest,
            "terminal-display-order",
            str(scene_index),
        )
    )
    return tuple(Observation(index, target.truth[index]) for index in selected)


def _request_pairs(request: EpisodeRequest) -> tuple[EligiblePair, ...]:
    candidates = build_eligible_pair_table().candidates(request.partition, request.target_op)
    return tuple(
        sorted(
            candidates,
            key=lambda pair: _hash_parts(
                REQUEST_PAIR_ORDER_DOMAIN,
                request.request_id,
                pair.order_digest,
            )
            + pair.order_digest.encode("ascii"),
        )
    )


def _generate_one(
    request: EpisodeRequest,
    spec: EpisodeBankSpec,
    *,
    reserved_target_digests: set[str],
) -> tuple[HiddenEpisode, EpisodeGenerationRecord]:
    catalog = build_rule_catalog()
    pairs = _request_pairs(request)
    for pair_rank, pair in enumerate(pairs[: spec.max_pair_attempts]):
        target = catalog[pair.target_index]
        if target.truth_digest in reserved_target_digests:
            continue
        shadow = catalog[pair.shadow_index]
        pools = _scene_pools(target, shadow)
        for opening_attempt in range(spec.opening_attempts_per_pair):
            opening = _greedy_opening(
                request,
                pair,
                pools=pools,
                catalog=catalog,
                spec=spec,
                opening_attempt=opening_attempt,
            )
            if opening is None:
                continue
            terminal = _terminal(
                request,
                pair,
                pools=pools,
                catalog=catalog,
                opening=opening,
            )
            episode = HiddenEpisode(
                episode_id=(
                    f"{spec.bank_id}-{request.request_id}-"
                    f"{target.truth_digest[:10]}-{shadow.truth_digest[:10]}"
                ),
                target=target,
                shadow=shadow,
                opening=opening,
                terminal=terminal,
                regime=request.regime,
                terminal_kind=request.terminal_kind,
                renderer=request.renderer,
            )
            space = episode.opening_version_space(catalog)
            depth = exact_minimax_identification_depth(
                space, max_depth=spec.maximum_minimax_depth
            )
            if depth is None:
                continue
            reference = run_reference_inquiry(
                episode, max_queries=spec.maximum_reference_queries
            )
            if not reference.target_identified:
                continue
            return episode, EpisodeGenerationRecord(
                request_id=request.request_id,
                pair_rank=pair_rank,
                opening_attempt=opening_attempt,
                version_space_size=len(space),
                minimax_depth=depth,
                reference_query_count=len(reference.queries),
            )
    raise EpisodeGenerationError(
        f"bounded search failed for {request.request_id!r}: "
        f"{spec.max_pair_attempts} pairs x {spec.opening_attempts_per_pair} openings"
    )


@lru_cache(maxsize=8)
def generate_episode_bank(spec: EpisodeBankSpec) -> EpisodeBank:
    """Generate a bank deterministically or fail after its explicit bound."""

    if type(spec) is not EpisodeBankSpec:
        raise TypeError("generate_episode_bank requires an EpisodeBankSpec")
    catalog = build_rule_catalog()
    partitions = build_rule_identity_partitions()
    pair_table = build_eligible_pair_table()
    episodes: list[HiddenEpisode] = []
    records: list[EpisodeGenerationRecord] = []
    reserved: set[str] = set()
    for request in spec.requests:
        episode, record = _generate_one(
            request,
            spec,
            reserved_target_digests=reserved,
        )
        episodes.append(episode)
        records.append(record)
        reserved.add(episode.target.truth_digest)
    return EpisodeBank(
        spec=spec,
        catalog_digest=catalog.digest,
        partitions_digest=partitions.digest,
        eligible_pairs_digest=pair_table.digest,
        episodes=tuple(episodes),
        generation_records=tuple(records),
    )


def serialize_episode_bank(bank: EpisodeBank) -> str:
    if type(bank) is not EpisodeBank:
        raise TypeError("serialize_episode_bank requires an EpisodeBank")
    return dump_json(bank.as_obj())


def parse_episode_bank(text: str, *, require_canonical: bool = True) -> EpisodeBank:
    try:
        value = load_json(text)
    except CanonicalJSONError as exc:
        raise ValueError(str(exc)) from exc
    expected = {
        "schema_version",
        "generator_schema_version",
        "spec",
        "catalog_digest",
        "partitions_digest",
        "eligible_pairs_digest",
        "formula_counts",
        "partition_counts",
        "episodes",
        "generation_records",
    }
    if type(value) is not dict or set(value) != expected or len(value) != len(expected):
        raise ValueError("episode bank manifest has noncanonical fields")
    if value["schema_version"] != EPISODE_BANK_SCHEMA_VERSION:
        raise ValueError("unsupported episode bank schema version")
    if value["generator_schema_version"] != EPISODE_GENERATOR_SCHEMA_VERSION:
        raise ValueError("unsupported episode generator schema version")
    if type(value["episodes"]) is not list or type(value["generation_records"]) is not list:
        raise ValueError("episode bank episodes and generation records must be arrays")
    result = EpisodeBank(
        spec=episode_bank_spec_from_obj(value["spec"]),
        catalog_digest=value["catalog_digest"],
        partitions_digest=value["partitions_digest"],
        eligible_pairs_digest=value["eligible_pairs_digest"],
        episodes=tuple(hidden_episode_from_obj(item) for item in value["episodes"]),
        generation_records=tuple(
            generation_record_from_obj(item) for item in value["generation_records"]
        ),
    )
    if result.as_obj() != value:
        raise ValueError("episode bank derived fields are inconsistent")
    if require_canonical and serialize_episode_bank(result) != text:
        raise ValueError("episode bank JSON is valid but not canonical")
    return verify_episode_bank(result)


def verify_episode_bank(bank: EpisodeBank) -> EpisodeBank:
    """Fail closed unless metrics and full deterministic regeneration match."""

    if type(bank) is not EpisodeBank:
        raise TypeError("verify_episode_bank requires an EpisodeBank")
    for request, episode, record in zip(
        bank.spec.requests, bank.episodes, bank.generation_records, strict=True
    ):
        request_pairs = _request_pairs(request)
        if (
            record.pair_rank >= bank.spec.max_pair_attempts
            or record.pair_rank >= len(request_pairs)
        ):
            raise ValueError("episode bank pair-rank provenance lies outside the pair table")
        if record.opening_attempt >= bank.spec.opening_attempts_per_pair:
            raise ValueError("episode bank opening-attempt provenance lies outside its bound")
        selected_pair = request_pairs[record.pair_rank]
        if (
            selected_pair.target_index != episode.target.index
            or selected_pair.shadow_index != episode.shadow.index
        ):
            raise ValueError("episode bank pair-rank provenance does not identify its episode")
        space = episode.opening_version_space()
        depth = exact_minimax_identification_depth(
            space, max_depth=bank.spec.maximum_minimax_depth
        )
        if depth != record.minimax_depth:
            raise ValueError("episode bank minimax-depth attestation mismatch")
        reference = run_reference_inquiry(
            episode, max_queries=bank.spec.maximum_reference_queries
        )
        if (
            not reference.target_identified
            or len(reference.queries) != record.reference_query_count
        ):
            raise ValueError("episode bank reference-expert attestation mismatch")
    regenerated = generate_episode_bank(bank.spec)
    if serialize_episode_bank(regenerated) != serialize_episode_bank(bank):
        raise ValueError("episode bank does not regenerate byte-for-byte from its spec")
    return bank


def small_fixture_bank_spec() -> EpisodeBankSpec:
    """Return the 12-episode engine fixture, never a final study bank."""

    requests: list[EpisodeRequest] = []
    renderers = tuple(RENDERERS)
    combinations: tuple[tuple[OpeningRegime, TerminalKind], ...] = (
        ("perfect_ambiguity", "train_like"),
        ("perfect_ambiguity", "factorial"),
        ("noisy_shortcuts", "train_like"),
        ("noisy_shortcuts", "factorial"),
    )
    simple_combination_indices = (0, 1, 2, 3, 0, 2)
    opposite_combination = {0: 3, 1: 2, 2: 1, 3: 0}
    for partition_index, partition in enumerate(RULE_PARTITIONS):
        simple_op: BinaryOp = "all" if partition_index % 2 == 0 else "any"
        simple_combination = simple_combination_indices[partition_index]
        for slot, (op, regime, terminal_kind) in enumerate(
            (
                (simple_op, *combinations[simple_combination]),
                (
                    "exactly_one",
                    *combinations[opposite_combination[simple_combination]],
                ),
            )
        ):
            noisy_error = (
                bool(partition_index % 2)
                if regime == "noisy_shortcuts"
                else None
            )
            request_id = f"p{partition_index}-{slot}-{op}-{regime}-{terminal_kind}"
            # This engine-only fixture salt avoids a long, scientifically
            # meaningless search for the engineering/any/perfect cell.  It is
            # part of the canonical request identity and never selected from
            # model outcomes.
            if partition_index == 1 and slot == 0:
                request_id = "p1-0-any-perfect-factorial-s03"
            requests.append(
                EpisodeRequest(
                    request_id=request_id,
                    partition=partition,
                    target_op=cast(BinaryOp, op),
                    regime=regime,
                    terminal_kind=cast(TerminalKind, terminal_kind),
                    renderer=renderers[partition_index],
                    noisy_placard_error_target=noisy_error,
                )
            )
    return EpisodeBankSpec(bank_id="g03-engine-small-fixture-v1", requests=tuple(requests))
