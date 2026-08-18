"""Schema-v2 format-warm-start and capability episode construction.

The implementation is additive: v1 episodes and fixtures remain authoritative
for the released engine.  This module demonstrates that one-literal piece and
placard rules can be the actual Official Law, represents the registered
chance-balanced warm-start profile, and keeps production requests fail closed
when the seven-way identity split lacks any required operator capacity.
"""

from __future__ import annotations

import hashlib
import heapq
import math
from collections import Counter
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Literal, cast

from ._json import CanonicalJSONError, dump_json, json_digest, load_json
from .catalog import CatalogEntry, RuleCatalog, VersionSpace, build_rule_catalog
from .episodes import (
    MAX_TARGET_SHADOW_DISAGREEMENT,
    MIN_TARGET_SHADOW_CELL_COUNT,
    MIN_TARGET_SHADOW_DISAGREEMENT,
    Observation,
    observation_from_obj,
)
from .rendering import (
    EVAL_RENDERERS,
    RENDERERS,
    TRAIN_RENDERERS,
    RendererName,
)
from .rules import BINARY_OPS, BinaryOp, BinaryRule
from .rules import Literal as RuleLiteral
from .schema import NONEMPTY_ARRANGEMENT_COUNT, SCENE_COUNT
from .stage_partitions_v2 import (
    RULE_PARTITIONS_V2,
    TARGET_FAMILIES_V2,
    EligibleTargetShadowPairV2,
    RulePartitionV2,
    TargetFamilyV2,
    build_eligible_target_shadow_table_v2,
    build_rule_identity_partitions_v2,
    catalog_rule_family_v2,
    target_shadow_cells_v2,
)

StageV2 = Literal["format_warm_start", "capability"]
ProxyProfileV2 = Literal["chance_balanced", "oracle_diagnostic"]
ProxyOrientationV2 = Literal["proxy_low", "proxy_high"]

STAGES_V2: tuple[StageV2, ...] = ("format_warm_start", "capability")
PROXY_PROFILES_V2: tuple[ProxyProfileV2, ...] = (
    "chance_balanced",
    "oracle_diagnostic",
)
PROXY_ORIENTATIONS_V2: tuple[ProxyOrientationV2, ...] = (
    "proxy_low",
    "proxy_high",
)
EPISODE_SCHEMA_VERSION_V2 = 2
EPISODE_REQUEST_SCHEMA_VERSION_V2 = 2
EPISODE_BANK_SCHEMA_VERSION_V2 = 2
STAGE_REQUEST_PLAN_SCHEMA_VERSION_V2 = 2
OPENING_SIZE_V2 = 10
TERMINAL_SIZE_V2 = 10
REQUEST_PAIR_ORDER_DOMAIN_V2 = b"goalzendo-interactive-request-pair-order-v2\0"
SCENE_ORDER_DOMAIN_V2 = b"goalzendo-interactive-scene-order-v2\0"


class CapabilityV2Error(ValueError):
    """Raised when a schema-v2 capability object violates its contract."""


def _valid_digest(value: object) -> bool:
    return (
        type(value) is str
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _hash_parts(domain: bytes, *parts: str) -> bytes:
    digest = hashlib.sha256()
    digest.update(domain)
    for part in parts:
        digest.update(part.encode("ascii"))
        digest.update(b"\0")
    return digest.digest()


@dataclass(frozen=True, slots=True)
class EpisodeRequestV2:
    """One exact schema-v2 episode request."""

    request_id: str
    stage: StageV2
    partition: RulePartitionV2
    target_family: TargetFamilyV2
    target_op: BinaryOp | None
    proxy_profile: ProxyProfileV2
    proxy_orientation: ProxyOrientationV2
    renderer: RendererName
    terminal_kind: Literal["train_like"] = "train_like"

    def __post_init__(self) -> None:
        if type(self.request_id) is not str or not self.request_id.strip():
            raise CapabilityV2Error("v2 episode request id cannot be empty")
        if not self.request_id.isascii():
            raise CapabilityV2Error("v2 episode request id must be ASCII")
        if self.stage not in STAGES_V2:
            raise CapabilityV2Error(f"unknown v2 stage: {self.stage!r}")
        if self.partition not in RULE_PARTITIONS_V2:
            raise CapabilityV2Error(f"unknown v2 partition: {self.partition!r}")
        if self.target_family not in TARGET_FAMILIES_V2:
            raise CapabilityV2Error(
                f"unknown v2 target family: {self.target_family!r}"
            )
        if self.target_family == "binary_piece":
            if self.target_op not in BINARY_OPS:
                raise CapabilityV2Error("binary-piece requests require an exact target_op")
        elif self.target_op is not None:
            raise CapabilityV2Error("one-literal requests must set target_op to null")
        if self.proxy_profile not in PROXY_PROFILES_V2:
            raise CapabilityV2Error(
                f"unknown v2 proxy profile: {self.proxy_profile!r}"
            )
        if self.proxy_orientation not in PROXY_ORIENTATIONS_V2:
            raise CapabilityV2Error(
                f"unknown v2 proxy orientation: {self.proxy_orientation!r}"
            )
        if self.renderer not in RENDERERS:
            raise CapabilityV2Error(f"unknown renderer: {self.renderer!r}")
        if self.terminal_kind != "train_like":
            raise CapabilityV2Error("schema-v2 capability slice currently requires train_like")

        if self.stage == "format_warm_start":
            if (
                self.partition != "warm_start"
                or self.target_family != "binary_piece"
                or self.proxy_profile != "chance_balanced"
                or self.renderer not in TRAIN_RENDERERS
            ):
                raise CapabilityV2Error(
                    "format warm-start requires its reserved partition, binary-piece "
                    "Official Laws, chance_balanced evidence, and a training renderer"
                )
        elif (
            self.partition != "capability"
            or self.proxy_profile != "oracle_diagnostic"
            or self.renderer not in EVAL_RENDERERS
        ):
            raise CapabilityV2Error(
                "capability requires its distinct partition, oracle_diagnostic evidence, "
                "and a held-out evaluation renderer"
            )

    def as_obj(self) -> dict[str, Any]:
        return {
            "schema_version": EPISODE_REQUEST_SCHEMA_VERSION_V2,
            "request_id": self.request_id,
            "stage": self.stage,
            "partition": self.partition,
            "target_family": self.target_family,
            "target_op": self.target_op,
            "proxy_profile": self.proxy_profile,
            "proxy_orientation": self.proxy_orientation,
            "renderer": self.renderer,
            "terminal_kind": self.terminal_kind,
        }

    @property
    def digest(self) -> str:
        return json_digest(
            self.as_obj(), domain="goalzendo-interactive-episode-request-v2"
        )


def episode_request_v2_from_obj(value: Any) -> EpisodeRequestV2:
    expected = {
        "schema_version",
        "request_id",
        "stage",
        "partition",
        "target_family",
        "target_op",
        "proxy_profile",
        "proxy_orientation",
        "renderer",
        "terminal_kind",
    }
    if type(value) is not dict or set(value) != expected or len(value) != len(expected):
        raise CapabilityV2Error("v2 episode request has noncanonical fields")
    if value["schema_version"] != EPISODE_REQUEST_SCHEMA_VERSION_V2:
        raise CapabilityV2Error("unsupported v2 episode-request schema version")
    result = EpisodeRequestV2(
        request_id=value["request_id"],
        stage=cast(StageV2, value["stage"]),
        partition=cast(RulePartitionV2, value["partition"]),
        target_family=cast(TargetFamilyV2, value["target_family"]),
        target_op=cast(BinaryOp | None, value["target_op"]),
        proxy_profile=cast(ProxyProfileV2, value["proxy_profile"]),
        proxy_orientation=cast(ProxyOrientationV2, value["proxy_orientation"]),
        renderer=cast(RendererName, value["renderer"]),
        terminal_kind=value["terminal_kind"],
    )
    if result.as_obj() != value:
        raise CapabilityV2Error("v2 episode request is valid but not canonical")
    return result


def serialize_episode_request_v2(request: EpisodeRequestV2) -> str:
    if type(request) is not EpisodeRequestV2:
        raise TypeError("serialize_episode_request_v2 requires EpisodeRequestV2")
    return dump_json(request.as_obj())


def parse_episode_request_v2(
    text: str,
    *,
    require_canonical: bool = True,
) -> EpisodeRequestV2:
    try:
        value = load_json(text)
    except CanonicalJSONError as exc:
        raise CapabilityV2Error(str(exc)) from exc
    result = episode_request_v2_from_obj(value)
    if require_canonical and serialize_episode_request_v2(result) != text:
        raise CapabilityV2Error("v2 request JSON is valid but not canonical")
    return result


def _is_sun(scene_index: int) -> bool:
    return scene_index < NONEMPTY_ARRANGEMENT_COUNT


def _cell_number(
    target: CatalogEntry,
    shadow: CatalogEntry,
    scene_index: int,
) -> int:
    return (
        int(target.truth[scene_index]) * 4
        + int(_is_sun(scene_index)) * 2
        + int(shadow.truth[scene_index])
    )


def _placard_literal_agrees_with_sun(target: CatalogEntry) -> bool:
    if catalog_rule_family_v2(target) != "placard_literal":
        raise CapabilityV2Error("placard agreement is defined only for placard literals")
    rule = cast(RuleLiteral, target.rule)
    return not rule.negated


def expected_cell_counts_v2(
    request: EpisodeRequestV2,
    target: CatalogEntry,
) -> tuple[int, ...]:
    """Return the exact ten-scene ``Y,P,Q`` profile in binary-cell order."""

    high = request.proxy_orientation == "proxy_high"
    if request.target_family != "placard_literal":
        # One scene in every cell plus a matched pair that makes both P and Q
        # low (4/10 true) or high (6/10 true).  In both orientations, Y is 5/5
        # and agreement(Y,P) == agreement(Y,Q) == 5/10 exactly.
        counts = [1] * 8
        for cell in ((0b011, 0b111) if high else (0b000, 0b100)):
            counts[cell] += 1
        return tuple(counts)

    agrees = _placard_literal_agrees_with_sun(target)
    low_q0, low_q1 = ((2, 3) if high else (3, 2))
    counts = [0] * 8
    if agrees:
        counts[0b000], counts[0b001] = low_q0, low_q1
        counts[0b110], counts[0b111] = low_q0, low_q1
    else:
        counts[0b010], counts[0b011] = low_q0, low_q1
        counts[0b100], counts[0b101] = low_q0, low_q1
    return tuple(counts)


def _observed_cell_counts(
    observations: tuple[Observation, ...],
    target: CatalogEntry,
    shadow: CatalogEntry,
) -> tuple[int, ...]:
    counts = [0] * 8
    for observation in observations:
        counts[_cell_number(target, shadow, observation.scene_index)] += 1
    return tuple(counts)


def _balance_summary(
    observations: tuple[Observation, ...],
    target: CatalogEntry,
    shadow: CatalogEntry,
) -> dict[str, Any]:
    cells = _observed_cell_counts(observations, target, shadow)
    target_true = sum(observation.accepted for observation in observations)
    placard_true = sum(_is_sun(observation.scene_index) for observation in observations)
    shadow_true = sum(shadow.truth[observation.scene_index] for observation in observations)
    target_placard_agreement = sum(
        target.truth[observation.scene_index] is _is_sun(observation.scene_index)
        for observation in observations
    )
    target_shadow_agreement = sum(
        target.truth[observation.scene_index] is shadow.truth[observation.scene_index]
        for observation in observations
    )
    return {
        "cell_counts_y_p_q": list(cells),
        "target_true": target_true,
        "placard_true": placard_true,
        "shadow_true": shadow_true,
        "target_placard_agreement": target_placard_agreement,
        "target_shadow_agreement": target_shadow_agreement,
    }


def _canonical_entry(entry: CatalogEntry, catalog: RuleCatalog, *, name: str) -> None:
    if type(entry) is not CatalogEntry:
        raise CapabilityV2Error(f"{name} must be a CatalogEntry")
    if not 0 <= entry.index < len(catalog) or catalog[entry.index] != entry:
        raise CapabilityV2Error(f"{name} is not bound to the canonical catalog")


@dataclass(frozen=True, slots=True)
class HiddenEpisodeV2:
    """A validated schema-v2 hidden-law episode."""

    episode_id: str
    request: EpisodeRequestV2
    target: CatalogEntry
    shadow: CatalogEntry
    opening: tuple[Observation, ...]
    terminal: tuple[Observation, ...]

    def __post_init__(self) -> None:
        if type(self.episode_id) is not str or not self.episode_id.strip():
            raise CapabilityV2Error("v2 episode id cannot be empty")
        if not self.episode_id.isascii():
            raise CapabilityV2Error("v2 episode id must be ASCII")
        if type(self.request) is not EpisodeRequestV2:
            raise CapabilityV2Error("v2 episode requires an EpisodeRequestV2")
        catalog = build_rule_catalog()
        _canonical_entry(self.target, catalog, name="target")
        _canonical_entry(self.shadow, catalog, name="shadow")
        partitions = build_rule_identity_partitions_v2()
        if partitions.for_entry(self.target) != self.request.partition:
            raise CapabilityV2Error("target identity is outside the request partition")
        if partitions.for_entry(self.shadow) != self.request.partition:
            raise CapabilityV2Error("shadow identity is outside the request partition")
        if catalog_rule_family_v2(self.target) != self.request.target_family:
            raise CapabilityV2Error("target does not have the requested target family")
        if catalog_rule_family_v2(self.shadow) != "literal_piece":
            raise CapabilityV2Error("schema-v2 shadow must be a one-literal piece rule")
        if (
            self.request.target_op is not None
            and cast(BinaryRule, self.target.rule).op != self.request.target_op
        ):
            raise CapabilityV2Error("target does not have the requested binary operator")

        cells = target_shadow_cells_v2(self.target, self.shadow)
        disagreement = cells[1] + cells[2]
        if min(cells) < MIN_TARGET_SHADOW_CELL_COUNT:
            raise CapabilityV2Error(
                "every schema-v2 target-by-shadow cell must retain the v1 minimum"
            )
        if not (
            math.ceil(MIN_TARGET_SHADOW_DISAGREEMENT * SCENE_COUNT)
            <= disagreement
            <= math.floor(MAX_TARGET_SHADOW_DISAGREEMENT * SCENE_COUNT)
        ):
            raise CapabilityV2Error(
                "schema-v2 target-shadow disagreement must retain the v1 bounds"
            )

        opening = tuple(self.opening)
        terminal = tuple(self.terminal)
        object.__setattr__(self, "opening", opening)
        object.__setattr__(self, "terminal", terminal)
        if len(opening) != OPENING_SIZE_V2 or len(terminal) != TERMINAL_SIZE_V2:
            raise CapabilityV2Error("v2 opening and terminal must each contain ten scenes")
        if any(type(item) is not Observation for item in (*opening, *terminal)):
            raise CapabilityV2Error("v2 opening and terminal require Observations")
        opening_indices = tuple(item.scene_index for item in opening)
        terminal_indices = tuple(item.scene_index for item in terminal)
        if len(set(opening_indices)) != len(opening_indices):
            raise CapabilityV2Error("v2 opening scenes must be unique")
        if len(set(terminal_indices)) != len(terminal_indices):
            raise CapabilityV2Error("v2 terminal scenes must be unique")
        if set(opening_indices) & set(terminal_indices):
            raise CapabilityV2Error("v2 opening and terminal scenes must be disjoint")
        expected = expected_cell_counts_v2(self.request, self.target)
        for name, observations in (("opening", opening), ("terminal", terminal)):
            if any(
                item.accepted is not self.target.truth[item.scene_index]
                for item in observations
            ):
                raise CapabilityV2Error(f"{name} labels must be exact Official-Law labels")
            if sum(item.accepted for item in observations) != 5:
                raise CapabilityV2Error(f"{name} Official-Law labels must be exactly 5/5")
            if _observed_cell_counts(observations, self.target, self.shadow) != expected:
                raise CapabilityV2Error(
                    f"{name} does not match the registered exact proxy profile"
                )

    def opening_version_space(self, catalog: RuleCatalog | None = None) -> VersionSpace:
        selected = build_rule_catalog() if catalog is None else catalog
        return selected.version_space(
            (observation.scene_index, observation.accepted)
            for observation in self.opening
        )

    def as_obj(self) -> dict[str, Any]:
        catalog = build_rule_catalog()
        partitions = build_rule_identity_partitions_v2()
        pairs = build_eligible_target_shadow_table_v2()
        return {
            "schema_version": EPISODE_SCHEMA_VERSION_V2,
            "episode_id": self.episode_id,
            "request": self.request.as_obj(),
            "catalog_digest": catalog.digest,
            "partitions_digest": partitions.digest,
            "eligible_pairs_digest": pairs.digest,
            "target_rule_id": self.target.rule_id,
            "target_truth_digest": self.target.truth_digest,
            "shadow_rule_id": self.shadow.rule_id,
            "shadow_truth_digest": self.shadow.truth_digest,
            "opening": [observation.as_obj() for observation in self.opening],
            "terminal": [observation.as_obj() for observation in self.terminal],
            "opening_balance": _balance_summary(
                self.opening, self.target, self.shadow
            ),
            "terminal_balance": _balance_summary(
                self.terminal, self.target, self.shadow
            ),
        }

    @property
    def digest(self) -> str:
        return json_digest(
            self.as_obj(), domain="goalzendo-interactive-hidden-episode-v2"
        )


def _catalog_entry(
    rule_id: Any,
    truth_digest: Any,
    *,
    catalog: RuleCatalog,
    name: str,
) -> CatalogEntry:
    if (
        type(rule_id) is not str
        or not rule_id.startswith("g03r")
        or len(rule_id) != 9
        or not rule_id[4:].isdigit()
    ):
        raise CapabilityV2Error(f"invalid {name} rule id: {rule_id!r}")
    index = int(rule_id[4:])
    if not 0 <= index < len(catalog):
        raise CapabilityV2Error(f"{name} rule id lies outside the catalog")
    entry = catalog[index]
    if entry.rule_id != rule_id or entry.truth_digest != truth_digest:
        raise CapabilityV2Error(f"{name} rule identity or truth digest mismatch")
    return entry


def hidden_episode_v2_from_obj(value: Any) -> HiddenEpisodeV2:
    expected = {
        "schema_version",
        "episode_id",
        "request",
        "catalog_digest",
        "partitions_digest",
        "eligible_pairs_digest",
        "target_rule_id",
        "target_truth_digest",
        "shadow_rule_id",
        "shadow_truth_digest",
        "opening",
        "terminal",
        "opening_balance",
        "terminal_balance",
    }
    if type(value) is not dict or set(value) != expected or len(value) != len(expected):
        raise CapabilityV2Error("v2 hidden episode has noncanonical fields")
    if value["schema_version"] != EPISODE_SCHEMA_VERSION_V2:
        raise CapabilityV2Error("unsupported v2 hidden-episode schema version")
    catalog = build_rule_catalog()
    partitions = build_rule_identity_partitions_v2()
    pairs = build_eligible_target_shadow_table_v2()
    if value["catalog_digest"] != catalog.digest:
        raise CapabilityV2Error("v2 episode catalog digest mismatch")
    if value["partitions_digest"] != partitions.digest:
        raise CapabilityV2Error("v2 episode partition digest mismatch")
    if value["eligible_pairs_digest"] != pairs.digest:
        raise CapabilityV2Error("v2 episode eligible-pair digest mismatch")
    if type(value["opening"]) is not list or type(value["terminal"]) is not list:
        raise CapabilityV2Error("v2 opening and terminal must be arrays")
    result = HiddenEpisodeV2(
        episode_id=value["episode_id"],
        request=episode_request_v2_from_obj(value["request"]),
        target=_catalog_entry(
            value["target_rule_id"],
            value["target_truth_digest"],
            catalog=catalog,
            name="target",
        ),
        shadow=_catalog_entry(
            value["shadow_rule_id"],
            value["shadow_truth_digest"],
            catalog=catalog,
            name="shadow",
        ),
        opening=tuple(observation_from_obj(item) for item in value["opening"]),
        terminal=tuple(observation_from_obj(item) for item in value["terminal"]),
    )
    if result.as_obj() != value:
        raise CapabilityV2Error("v2 hidden episode derived fields are inconsistent")
    return result


def serialize_hidden_episode_v2(episode: HiddenEpisodeV2) -> str:
    if type(episode) is not HiddenEpisodeV2:
        raise TypeError("serialize_hidden_episode_v2 requires HiddenEpisodeV2")
    return dump_json(episode.as_obj())


def parse_hidden_episode_v2(
    text: str,
    *,
    require_canonical: bool = True,
) -> HiddenEpisodeV2:
    try:
        value = load_json(text)
    except CanonicalJSONError as exc:
        raise CapabilityV2Error(str(exc)) from exc
    result = hidden_episode_v2_from_obj(value)
    if require_canonical and serialize_hidden_episode_v2(result) != text:
        raise CapabilityV2Error("v2 episode JSON is valid but not canonical")
    return result


@dataclass(frozen=True, slots=True)
class EpisodeBankSpecV2:
    bank_id: str
    requests: tuple[EpisodeRequestV2, ...]
    max_pair_attempts: int = 4_096

    def __post_init__(self) -> None:
        if type(self.bank_id) is not str or not self.bank_id.strip() or not self.bank_id.isascii():
            raise CapabilityV2Error("v2 bank id must be nonempty ASCII")
        requests = tuple(self.requests)
        object.__setattr__(self, "requests", requests)
        if not requests or any(type(item) is not EpisodeRequestV2 for item in requests):
            raise CapabilityV2Error("v2 bank requires at least one EpisodeRequestV2")
        if len({item.request_id for item in requests}) != len(requests):
            raise CapabilityV2Error("v2 bank request ids must be unique")
        if len({(item.stage, item.partition) for item in requests}) != 1:
            raise CapabilityV2Error(
                "one v2 generation unit may contain exactly one stage and partition"
            )
        orientations = Counter(item.proxy_orientation for item in requests)
        if orientations["proxy_low"] != orientations["proxy_high"]:
            raise CapabilityV2Error(
                "v2 bank must pair proxy_low and proxy_high requests exactly"
            )
        if (
            isinstance(self.max_pair_attempts, bool)
            or not isinstance(self.max_pair_attempts, int)
            or not 1 <= self.max_pair_attempts <= 100_000
        ):
            raise CapabilityV2Error("max_pair_attempts must lie in [1, 100000]")

    def as_obj(self) -> dict[str, Any]:
        return {
            "bank_id": self.bank_id,
            "requests": [request.as_obj() for request in self.requests],
            "max_pair_attempts": self.max_pair_attempts,
        }

    @property
    def digest(self) -> str:
        return json_digest(
            self.as_obj(), domain="goalzendo-interactive-episode-bank-spec-v2"
        )


@dataclass(frozen=True, slots=True)
class EpisodeGenerationRecordV2:
    request_id: str
    pair_rank: int
    pair_order_digest: str
    opening_scene_digest: str
    terminal_scene_digest: str

    def __post_init__(self) -> None:
        if type(self.request_id) is not str or not self.request_id:
            raise CapabilityV2Error("v2 generation record request id cannot be empty")
        if (
            isinstance(self.pair_rank, bool)
            or not isinstance(self.pair_rank, int)
            or self.pair_rank < 0
        ):
            raise CapabilityV2Error("v2 pair rank must be a non-negative integer")
        for name in (
            "pair_order_digest",
            "opening_scene_digest",
            "terminal_scene_digest",
        ):
            if not _valid_digest(getattr(self, name)):
                raise CapabilityV2Error(f"{name} must be a SHA-256 digest")

    def as_obj(self) -> dict[str, int | str]:
        return {
            "request_id": self.request_id,
            "pair_rank": self.pair_rank,
            "pair_order_digest": self.pair_order_digest,
            "opening_scene_digest": self.opening_scene_digest,
            "terminal_scene_digest": self.terminal_scene_digest,
        }


@dataclass(frozen=True, slots=True)
class EpisodeBankV2:
    spec: EpisodeBankSpecV2
    catalog_digest: str
    partitions_digest: str
    eligible_pairs_digest: str
    episodes: tuple[HiddenEpisodeV2, ...]
    generation_records: tuple[EpisodeGenerationRecordV2, ...]

    def __post_init__(self) -> None:
        if type(self.spec) is not EpisodeBankSpecV2:
            raise CapabilityV2Error("v2 bank requires an EpisodeBankSpecV2")
        episodes = tuple(self.episodes)
        records = tuple(self.generation_records)
        object.__setattr__(self, "episodes", episodes)
        object.__setattr__(self, "generation_records", records)
        if len(episodes) != len(self.spec.requests) or len(records) != len(episodes):
            raise CapabilityV2Error("v2 requests, episodes, and records must align")
        catalog = build_rule_catalog()
        partitions = build_rule_identity_partitions_v2()
        pairs = build_eligible_target_shadow_table_v2()
        if self.catalog_digest != catalog.digest:
            raise CapabilityV2Error("v2 bank catalog digest mismatch")
        if self.partitions_digest != partitions.digest:
            raise CapabilityV2Error("v2 bank partition digest mismatch")
        if self.eligible_pairs_digest != pairs.digest:
            raise CapabilityV2Error("v2 bank eligible-pair digest mismatch")
        targets = [episode.target.truth_digest for episode in episodes]
        if len(set(targets)) != len(targets):
            raise CapabilityV2Error("v2 bank Official-Law identities must be unique")
        for request, episode, record in zip(
            self.spec.requests, episodes, records, strict=True
        ):
            if episode.request != request or record.request_id != request.request_id:
                raise CapabilityV2Error("v2 bank request binding is inconsistent")

    @property
    def target_family_counts(self) -> dict[str, int]:
        counts = Counter(episode.request.target_family for episode in self.episodes)
        return {family: counts[family] for family in TARGET_FAMILIES_V2}

    def as_obj(self) -> dict[str, Any]:
        return {
            "schema_version": EPISODE_BANK_SCHEMA_VERSION_V2,
            "spec": self.spec.as_obj(),
            "catalog_digest": self.catalog_digest,
            "partitions_digest": self.partitions_digest,
            "eligible_pairs_digest": self.eligible_pairs_digest,
            "target_family_counts": self.target_family_counts,
            "episodes": [episode.as_obj() for episode in self.episodes],
            "generation_records": [record.as_obj() for record in self.generation_records],
            "weight_updates_authorized": False,
        }

    @property
    def digest(self) -> str:
        return json_digest(self.as_obj(), domain="goalzendo-interactive-episode-bank-v2")


def _request_pairs_v2(
    request: EpisodeRequestV2,
) -> tuple[EligibleTargetShadowPairV2, ...]:
    catalog = build_rule_catalog()
    candidates = build_eligible_target_shadow_table_v2().candidates(
        request.partition, request.target_family
    )
    filtered = tuple(
        pair
        for pair in candidates
        if request.target_op is None
        or cast(BinaryRule, catalog[pair.target_index].rule).op == request.target_op
    )
    return tuple(
        sorted(
            filtered,
            key=lambda pair: _hash_parts(
                REQUEST_PAIR_ORDER_DOMAIN_V2,
                request.request_id,
                pair.order_digest,
            )
            + pair.order_digest.encode("ascii"),
        )
    )


def _scene_pools_v2(
    target: CatalogEntry,
    shadow: CatalogEntry,
) -> tuple[tuple[int, ...], ...]:
    pools: list[list[int]] = [[] for _ in range(8)]
    for scene_index in range(SCENE_COUNT):
        pools[_cell_number(target, shadow, scene_index)].append(scene_index)
    return tuple(tuple(pool) for pool in pools)


def _observations_for_phase(
    request: EpisodeRequestV2,
    pair: EligibleTargetShadowPairV2,
    *,
    target: CatalogEntry,
    shadow: CatalogEntry,
    pools: tuple[tuple[int, ...], ...],
    phase: Literal["opening", "terminal"],
    excluded: set[int],
) -> tuple[Observation, ...]:
    counts = expected_cell_counts_v2(request, target)
    selected: list[int] = []
    for cell, count in enumerate(counts):
        if count == 0:
            continue
        candidates = heapq.nsmallest(
            count,
            (scene_index for scene_index in pools[cell] if scene_index not in excluded),
            key=lambda scene_index: _hash_parts(
                SCENE_ORDER_DOMAIN_V2,
                request.request_id,
                pair.order_digest,
                phase,
                str(cell),
                str(scene_index),
            ),
        )
        if len(candidates) != count:
            raise CapabilityV2Error(
                f"eligible v2 pair lacks enough disjoint {phase} scenes in cell {cell}"
            )
        selected.extend(candidates)
        excluded.update(candidates)
    selected.sort(
        key=lambda scene_index: _hash_parts(
            SCENE_ORDER_DOMAIN_V2,
            request.request_id,
            pair.order_digest,
            f"{phase}-display-order",
            str(scene_index),
        )
    )
    return tuple(Observation(index, target.truth[index]) for index in selected)


def _generate_one_v2(
    request: EpisodeRequestV2,
    spec: EpisodeBankSpecV2,
    *,
    reserved_targets: set[str],
) -> tuple[HiddenEpisodeV2, EpisodeGenerationRecordV2]:
    catalog = build_rule_catalog()
    pairs = _request_pairs_v2(request)
    for pair_rank, pair in enumerate(pairs[: spec.max_pair_attempts]):
        target = catalog[pair.target_index]
        if target.truth_digest in reserved_targets:
            continue
        shadow = catalog[pair.shadow_index]
        pools = _scene_pools_v2(target, shadow)
        excluded: set[int] = set()
        opening = _observations_for_phase(
            request,
            pair,
            target=target,
            shadow=shadow,
            pools=pools,
            phase="opening",
            excluded=excluded,
        )
        terminal = _observations_for_phase(
            request,
            pair,
            target=target,
            shadow=shadow,
            pools=pools,
            phase="terminal",
            excluded=excluded,
        )
        episode = HiddenEpisodeV2(
            episode_id=(
                f"{spec.bank_id}-{request.request_id}-"
                f"{target.truth_digest[:10]}-{shadow.truth_digest[:10]}"
            ),
            request=request,
            target=target,
            shadow=shadow,
            opening=opening,
            terminal=terminal,
        )
        return episode, EpisodeGenerationRecordV2(
            request_id=request.request_id,
            pair_rank=pair_rank,
            pair_order_digest=pair.order_digest,
            opening_scene_digest=json_digest(
                [item.scene_index for item in opening],
                domain="goalzendo-interactive-opening-scenes-v2",
            ),
            terminal_scene_digest=json_digest(
                [item.scene_index for item in terminal],
                domain="goalzendo-interactive-terminal-scenes-v2",
            ),
        )
    raise CapabilityV2Error(
        f"bounded v2 construction failed for {request.request_id!r}: "
        f"{spec.max_pair_attempts} pair attempts"
    )


@lru_cache(maxsize=8)
def generate_episode_bank_v2(spec: EpisodeBankSpecV2) -> EpisodeBankV2:
    """Construct a schema-v2 bank deterministically within its exact bound."""

    if type(spec) is not EpisodeBankSpecV2:
        raise TypeError("generate_episode_bank_v2 requires EpisodeBankSpecV2")
    episodes: list[HiddenEpisodeV2] = []
    records: list[EpisodeGenerationRecordV2] = []
    reserved: set[str] = set()
    for request in spec.requests:
        episode, record = _generate_one_v2(
            request, spec, reserved_targets=reserved
        )
        episodes.append(episode)
        records.append(record)
        reserved.add(episode.target.truth_digest)
    catalog = build_rule_catalog()
    partitions = build_rule_identity_partitions_v2()
    pairs = build_eligible_target_shadow_table_v2()
    return EpisodeBankV2(
        spec=spec,
        catalog_digest=catalog.digest,
        partitions_digest=partitions.digest,
        eligible_pairs_digest=pairs.digest,
        episodes=tuple(episodes),
        generation_records=tuple(records),
    )


def serialize_episode_bank_v2(bank: EpisodeBankV2) -> str:
    if type(bank) is not EpisodeBankV2:
        raise TypeError("serialize_episode_bank_v2 requires EpisodeBankV2")
    return dump_json(bank.as_obj())


def verify_episode_bank_v2(bank: EpisodeBankV2) -> EpisodeBankV2:
    if type(bank) is not EpisodeBankV2:
        raise TypeError("verify_episode_bank_v2 requires EpisodeBankV2")
    regenerated = generate_episode_bank_v2(bank.spec)
    if serialize_episode_bank_v2(regenerated) != serialize_episode_bank_v2(bank):
        raise CapabilityV2Error("v2 bank does not regenerate byte-for-byte")
    return bank


@dataclass(frozen=True, slots=True)
class IdentityCapacityV2:
    partition: RulePartitionV2
    target_family: TargetFamilyV2
    target_op: BinaryOp | None
    requested: int
    available: int

    @property
    def sufficient(self) -> bool:
        return self.available >= self.requested

    def as_obj(self) -> dict[str, Any]:
        return {
            "partition": self.partition,
            "target_family": self.target_family,
            "target_op": self.target_op,
            "requested": self.requested,
            "available": self.available,
            "sufficient": self.sufficient,
        }


@dataclass(frozen=True, slots=True)
class StageRequestPlanV2:
    """Request-only two-stage plan plus whole-program identity-capacity audit."""

    warm_start: EpisodeBankSpecV2
    capability: EpisodeBankSpecV2
    catalog_digest: str
    partitions_digest: str
    eligible_pairs_digest: str
    capacity: tuple[IdentityCapacityV2, ...]
    blocker_codes: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.warm_start.requests[0].stage != "format_warm_start":
            raise CapabilityV2Error("warm-start unit has the wrong stage")
        if self.capability.requests[0].stage != "capability":
            raise CapabilityV2Error("capability unit has the wrong stage")
        request_ids = [
            request.request_id
            for spec in (self.warm_start, self.capability)
            for request in spec.requests
        ]
        if len(request_ids) != len(set(request_ids)):
            raise CapabilityV2Error("v2 plan request ids must be globally unique")
        if len(self.warm_start.requests) != 256 or len(self.capability.requests) != 256:
            raise CapabilityV2Error("v2 planning slices must each contain 256 requests")
        for name in ("catalog_digest", "partitions_digest", "eligible_pairs_digest"):
            if not _valid_digest(getattr(self, name)):
                raise CapabilityV2Error(f"{name} must be a SHA-256 digest")
        blockers = tuple(self.blocker_codes)
        object.__setattr__(self, "blocker_codes", blockers)
        if not blockers:
            raise CapabilityV2Error(
                "request-only v2 plan must retain at least the no-weight-authorization blocker"
            )
        if any(not item.sufficient for item in self.capacity) and not any(
            code.startswith("insufficient_identity_capacity:") for code in blockers
        ):
            raise CapabilityV2Error("capacity failure is missing from blocker codes")

    @property
    def requested_episode_count(self) -> int:
        return len(self.warm_start.requests) + len(self.capability.requests)

    @property
    def generated_episode_count(self) -> int:
        return 0

    @property
    def production_bank_generation_authorized(self) -> bool:
        return False

    @property
    def weight_updates_authorized(self) -> bool:
        return False

    def as_obj(self) -> dict[str, Any]:
        return {
            "schema_version": STAGE_REQUEST_PLAN_SCHEMA_VERSION_V2,
            "catalog_digest": self.catalog_digest,
            "partitions_digest": self.partitions_digest,
            "eligible_pairs_digest": self.eligible_pairs_digest,
            "units": {
                "format_warm_start": self.warm_start.as_obj(),
                "capability": self.capability.as_obj(),
            },
            "requested_episode_count": self.requested_episode_count,
            "generated_episode_count": self.generated_episode_count,
            "capacity": [item.as_obj() for item in self.capacity],
            "blocker_codes": list(self.blocker_codes),
            "required_qa": {
                "partition_distribution": {
                    "status": "not_run",
                    "features": [
                        "target_prevalence",
                        "canonical_rule_json_length",
                        "atom_operator_presence",
                        "target_shadow_cell_counts",
                        "renderer_assignment",
                    ],
                    "requirement": (
                        "freeze and pass a stage-comparability audit on materialized banks"
                    ),
                },
                "surface_leakage": {
                    "status": "not_run",
                    "grouping_unit": "Official-Law truth identity",
                    "requirement": (
                        "run the registered powered grouped audit on materialized banks"
                    ),
                },
            },
            "production_bank_generation_authorized": (
                self.production_bank_generation_authorized
            ),
            "weight_updates_authorized": self.weight_updates_authorized,
        }

    @property
    def digest(self) -> str:
        return json_digest(
            self.as_obj(), domain="goalzendo-interactive-stage-request-plan-v2"
        )


def _balanced_warm_start_requests() -> tuple[EpisodeRequestV2, ...]:
    requests: list[EpisodeRequestV2] = []
    operator_slots: tuple[BinaryOp, ...] = (
        "all",
        "any",
        "exactly_one",
        "exactly_one",
    )
    index = 0
    for _repeat in range(8):
        for renderer in TRAIN_RENDERERS:
            for orientation in PROXY_ORIENTATIONS_V2:
                for op in operator_slots:
                    requests.append(
                        EpisodeRequestV2(
                            request_id=f"g03-v2-warm-{index:04d}",
                            stage="format_warm_start",
                            partition="warm_start",
                            target_family="binary_piece",
                            target_op=op,
                            proxy_profile="chance_balanced",
                            proxy_orientation=orientation,
                            renderer=renderer,
                        )
                    )
                    index += 1
    return tuple(requests)


def _capability_target_slots() -> tuple[tuple[TargetFamilyV2, BinaryOp | None], ...]:
    return (
        *(("placard_literal", None),) * 2,
        *(("literal_piece", None),) * 16,
        *(("binary_piece", "all"),) * 60,
        *(("binary_piece", "any"),) * 60,
        *(("binary_piece", "exactly_one"),) * 118,
    )


def _balanced_capability_requests() -> tuple[EpisodeRequestV2, ...]:
    target_slots = _capability_target_slots()
    requests: list[EpisodeRequestV2] = []
    joint = (
        ("proxy_low", "eval_reverse"),
        ("proxy_high", "eval_ledger"),
        ("proxy_low", "eval_ledger"),
        ("proxy_high", "eval_reverse"),
    )
    for index, (family, op) in enumerate(target_slots):
        orientation, renderer = joint[index % len(joint)]
        requests.append(
            EpisodeRequestV2(
                request_id=f"g03-v2-capability-{index:04d}",
                stage="capability",
                partition="capability",
                target_family=family,
                target_op=op,
                proxy_profile="oracle_diagnostic",
                proxy_orientation=cast(ProxyOrientationV2, orientation),
                renderer=cast(RendererName, renderer),
            )
        )
    return tuple(requests)


_BINARY_CAPACITY_REQUIREMENTS: tuple[
    tuple[RulePartitionV2, int, int, int], ...
] = (
    ("warm_start", 64, 64, 128),
    ("engineering", 192, 192, 384),
    ("capability", 60, 60, 118),
    ("pilot", 64, 64, 128),
    ("confirmatory_train", 192, 192, 384),
    ("validation", 64, 64, 128),
    ("evaluation", 192, 192, 384),
)


@lru_cache(maxsize=1)
def build_stage_request_plan_v2() -> StageRequestPlanV2:
    """Build a deterministic request plan; never authorize production or weights."""

    catalog = build_rule_catalog()
    partitions = build_rule_identity_partitions_v2()
    pairs = build_eligible_target_shadow_table_v2()
    operator_available = pairs.binary_operator_target_identity_counts
    family_available = pairs.target_identity_counts
    capacity: list[IdentityCapacityV2] = []
    for partition, all_count, any_count, exactly_count in _BINARY_CAPACITY_REQUIREMENTS:
        for op, requested in zip(
            BINARY_OPS, (all_count, any_count, exactly_count), strict=True
        ):
            capacity.append(
                IdentityCapacityV2(
                    partition=partition,
                    target_family="binary_piece",
                    target_op=op,
                    requested=requested,
                    available=operator_available[f"{partition}:{op}"],
                )
            )
    for family, requested in (("literal_piece", 16), ("placard_literal", 2)):
        capacity.append(
            IdentityCapacityV2(
                partition="capability",
                target_family=cast(TargetFamilyV2, family),
                target_op=None,
                requested=requested,
                available=family_available[f"capability:{family}"],
            )
        )
    insufficient = tuple(item for item in capacity if not item.sufficient)
    blockers = (
        *(
            f"insufficient_identity_capacity:{item.partition}:"
            f"{item.target_op or item.target_family}:{item.available}<{item.requested}"
            for item in insufficient
        ),
        "request_only_full_256_episode_banks_not_materialized",
        "distribution_and_surface_leakage_qa_not_run",
        "model_pipeline_integration_not_audited",
        "no_weight_update_authorization",
    )
    return StageRequestPlanV2(
        warm_start=EpisodeBankSpecV2(
            bank_id="g03-format-warm-start-v2",
            requests=_balanced_warm_start_requests(),
        ),
        capability=EpisodeBankSpecV2(
            bank_id="g03-capability-v2",
            requests=_balanced_capability_requests(),
        ),
        catalog_digest=catalog.digest,
        partitions_digest=partitions.digest,
        eligible_pairs_digest=pairs.digest,
        capacity=tuple(capacity),
        blocker_codes=blockers,
    )


def serialize_stage_request_plan_v2(plan: StageRequestPlanV2) -> str:
    if type(plan) is not StageRequestPlanV2:
        raise TypeError("serialize_stage_request_plan_v2 requires StageRequestPlanV2")
    return dump_json(plan.as_obj())


def parse_stage_request_plan_v2(
    text: str,
    *,
    require_canonical: bool = True,
) -> StageRequestPlanV2:
    """Parse only the exact deterministic request-only plan."""

    try:
        value = load_json(text)
    except CanonicalJSONError as exc:
        raise CapabilityV2Error(str(exc)) from exc
    expected = build_stage_request_plan_v2()
    if value != expected.as_obj():
        raise CapabilityV2Error("v2 stage plan differs from deterministic request plan")
    if require_canonical and serialize_stage_request_plan_v2(expected) != text:
        raise CapabilityV2Error("v2 stage-plan JSON is valid but not canonical")
    return expected


def small_capability_bank_spec_v2() -> EpisodeBankSpecV2:
    """Return a ten-episode, non-authorizing schema-v2 engine fixture spec."""

    slots: tuple[tuple[TargetFamilyV2, BinaryOp | None], ...] = (
        ("placard_literal", None),
        ("placard_literal", None),
        ("literal_piece", None),
        ("literal_piece", None),
        ("binary_piece", "all"),
        ("binary_piece", "all"),
        ("binary_piece", "any"),
        ("binary_piece", "any"),
        ("binary_piece", "exactly_one"),
        ("binary_piece", "exactly_one"),
    )
    requests = tuple(
        EpisodeRequestV2(
            request_id=f"g03-v2-small-capability-{index:02d}",
            stage="capability",
            partition="capability",
            target_family=family,
            target_op=op,
            proxy_profile="oracle_diagnostic",
            proxy_orientation=PROXY_ORIENTATIONS_V2[index % 2],
            renderer=EVAL_RENDERERS[index % 2],
        )
        for index, (family, op) in enumerate(slots)
    )
    return EpisodeBankSpecV2(
        bank_id="g03-v2-small-capability",
        requests=requests,
        max_pair_attempts=4_096,
    )
