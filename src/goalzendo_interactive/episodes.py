"""Immutable hidden-episode and oracle-observation schemas."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, cast

from ._json import CanonicalJSONError, dump_json, json_digest, load_json
from .catalog import CatalogEntry, RuleCatalog, VersionSpace, build_rule_catalog
from .rendering import RENDERERS, RendererName
from .rules import BinaryRule, Rule
from .rules import Literal as RuleLiteral
from .schema import SCENE_COUNT, Scene, scene_at, scene_index

OpeningRegime = Literal["perfect_ambiguity", "noisy_shortcuts"]
TerminalKind = Literal["train_like", "factorial"]
OPENING_REGIMES: tuple[OpeningRegime, ...] = ("perfect_ambiguity", "noisy_shortcuts")
TERMINAL_KINDS: tuple[TerminalKind, ...] = ("train_like", "factorial")
OPENING_SIZE = 10
FACTORIAL_TERMINAL_SIZE = 16
TRAIN_LIKE_TERMINAL_SIZE = 10
MIN_TARGET_SHADOW_CELL_COUNT = 128
MIN_TARGET_SHADOW_DISAGREEMENT = 0.35
MAX_TARGET_SHADOW_DISAGREEMENT = 0.65
EPISODE_SCHEMA_VERSION = 1
_UNIVERSE_MASK = (1 << SCENE_COUNT) - 1


class EpisodeValidationError(ValueError):
    """Raised when a retained hidden episode violates a scientific invariant."""


@dataclass(frozen=True, slots=True)
class Observation:
    """One exact oracle label for one canonical scene."""

    scene_index: int
    accepted: bool

    def __post_init__(self) -> None:
        if (
            isinstance(self.scene_index, bool)
            or not isinstance(self.scene_index, int)
            or not 0 <= self.scene_index < SCENE_COUNT
        ):
            raise EpisodeValidationError(f"scene_index must lie in [0, {SCENE_COUNT})")
        if type(self.accepted) is not bool:
            raise EpisodeValidationError("observation label must be Boolean")

    @property
    def scene(self) -> Scene:
        return scene_at(self.scene_index)

    def as_obj(self) -> dict[str, int | bool]:
        return {"scene_index": self.scene_index, "accepted": self.accepted}

    @classmethod
    def from_scene(cls, scene: Scene, accepted: bool) -> Observation:
        return cls(scene_index(scene), accepted)


def _rule_uses_placard(rule: Rule) -> bool:
    literals = (rule,) if isinstance(rule, RuleLiteral) else rule.args
    return any(literal.atom.op == "placard_is" for literal in literals)


def _target_shadow_cells(target: CatalogEntry, shadow: CatalogEntry) -> dict[tuple[bool, bool], int]:
    y_bits = target.truth.bits
    q_bits = shadow.truth.bits
    return {
        (False, False): ((_UNIVERSE_MASK ^ y_bits) & (_UNIVERSE_MASK ^ q_bits)).bit_count(),
        (False, True): ((_UNIVERSE_MASK ^ y_bits) & q_bits).bit_count(),
        (True, False): (y_bits & (_UNIVERSE_MASK ^ q_bits)).bit_count(),
        (True, True): (y_bits & q_bits).bit_count(),
    }


def _canonical_entry(entry: CatalogEntry, catalog: RuleCatalog, *, name: str) -> None:
    if type(entry) is not CatalogEntry:
        raise EpisodeValidationError(f"{name} must be a CatalogEntry")
    if not 0 <= entry.index < len(catalog) or catalog[entry.index] != entry:
        raise EpisodeValidationError(f"{name} is not bound to the canonical rule catalog")


def _agreement_count(observations: tuple[Observation, ...], entry: CatalogEntry) -> int:
    return sum(entry.truth[observation.scene_index] is observation.accepted for observation in observations)


def _placard_agreement_count(observations: tuple[Observation, ...]) -> int:
    return sum((observation.scene.placard == "sun") is observation.accepted for observation in observations)


def _validate_proxy_pattern(
    observations: tuple[Observation, ...],
    *,
    regime: OpeningRegime,
    shadow: CatalogEntry,
    context: str,
) -> None:
    expected_placard = len(observations) if regime == "perfect_ambiguity" else len(observations) - 1
    expected_shadow = len(observations) if regime == "perfect_ambiguity" else len(observations) - 2
    placard_agreement = _placard_agreement_count(observations)
    shadow_agreement = _agreement_count(observations, shadow)
    if placard_agreement != expected_placard:
        raise EpisodeValidationError(
            f"{context} placard agreement must be {expected_placard}/{len(observations)}, "
            f"received {placard_agreement}/{len(observations)}"
        )
    if shadow_agreement != expected_shadow:
        raise EpisodeValidationError(
            f"{context} shadow agreement must be {expected_shadow}/{len(observations)}, "
            f"received {shadow_agreement}/{len(observations)}"
        )
    if regime == "noisy_shortcuts":
        shadow_errors = [
            observation
            for observation in observations
            if shadow.truth[observation.scene_index] is not observation.accepted
        ]
        if {observation.accepted for observation in shadow_errors} != {False, True}:
            raise EpisodeValidationError(
                f"{context} two shadow errors must include one fitting and one non-fitting scene"
            )


@dataclass(frozen=True, slots=True)
class HiddenEpisode:
    """One fully specified hidden-law game, including evaluator-only labels."""

    episode_id: str
    target: CatalogEntry
    shadow: CatalogEntry
    opening: tuple[Observation, ...]
    terminal: tuple[Observation, ...]
    regime: OpeningRegime
    terminal_kind: TerminalKind
    renderer: RendererName = "train_compact"

    def __post_init__(self) -> None:
        if type(self.episode_id) is not str or not self.episode_id.strip():
            raise EpisodeValidationError("episode_id cannot be empty")
        if self.regime not in OPENING_REGIMES:
            raise EpisodeValidationError(f"unknown opening regime: {self.regime!r}")
        if self.terminal_kind not in TERMINAL_KINDS:
            raise EpisodeValidationError(f"unknown terminal kind: {self.terminal_kind!r}")
        if self.renderer not in RENDERERS:
            raise EpisodeValidationError(f"unknown renderer: {self.renderer!r}")

        catalog = build_rule_catalog()
        _canonical_entry(self.target, catalog, name="target")
        _canonical_entry(self.shadow, catalog, name="shadow")
        if type(self.target.rule) is not BinaryRule or _rule_uses_placard(self.target.rule):
            raise EpisodeValidationError("target must be a two-literal piece-only catalog rule")
        if type(self.shadow.rule) is not RuleLiteral or _rule_uses_placard(self.shadow.rule):
            raise EpisodeValidationError("shadow must be a one-literal piece-only catalog rule")
        if self.target.truth.bits == self.shadow.truth.bits:
            raise EpisodeValidationError("target and shadow must be extensionally distinct")

        cells = _target_shadow_cells(self.target, self.shadow)
        if min(cells.values()) < MIN_TARGET_SHADOW_CELL_COUNT:
            raise EpisodeValidationError(
                "every target-by-shadow cell must contain at least "
                f"{MIN_TARGET_SHADOW_CELL_COUNT} scenes"
            )
        disagreement = (cells[(False, True)] + cells[(True, False)]) / SCENE_COUNT
        if not MIN_TARGET_SHADOW_DISAGREEMENT <= disagreement <= MAX_TARGET_SHADOW_DISAGREEMENT:
            raise EpisodeValidationError(
                "target-shadow disagreement must lie in "
                f"[{MIN_TARGET_SHADOW_DISAGREEMENT}, {MAX_TARGET_SHADOW_DISAGREEMENT}]"
            )

        opening = tuple(self.opening)
        terminal = tuple(self.terminal)
        object.__setattr__(self, "opening", opening)
        object.__setattr__(self, "terminal", terminal)
        if len(opening) != OPENING_SIZE:
            raise EpisodeValidationError(f"opening must contain exactly {OPENING_SIZE} observations")
        expected_terminal = (
            FACTORIAL_TERMINAL_SIZE
            if self.terminal_kind == "factorial"
            else TRAIN_LIKE_TERMINAL_SIZE
        )
        if len(terminal) != expected_terminal:
            raise EpisodeValidationError(
                f"{self.terminal_kind} terminal must contain exactly {expected_terminal} observations"
            )
        if any(type(observation) is not Observation for observation in (*opening, *terminal)):
            raise EpisodeValidationError("opening and terminal entries must be Observations")
        opening_indices = tuple(observation.scene_index for observation in opening)
        terminal_indices = tuple(observation.scene_index for observation in terminal)
        if len(set(opening_indices)) != len(opening_indices):
            raise EpisodeValidationError("opening scenes must be unique")
        if len(set(terminal_indices)) != len(terminal_indices):
            raise EpisodeValidationError("terminal scenes must be unique")
        if set(opening_indices) & set(terminal_indices):
            raise EpisodeValidationError("opening and terminal scenes must be disjoint")

        for context, observations in (("opening", opening), ("terminal", terminal)):
            if any(
                self.target.truth[observation.scene_index] is not observation.accepted
                for observation in observations
            ):
                raise EpisodeValidationError(f"{context} labels must be exact target labels")
        if sum(observation.accepted for observation in opening) != OPENING_SIZE // 2:
            raise EpisodeValidationError("opening target labels must be exactly 5/5 balanced")
        _validate_proxy_pattern(opening, regime=self.regime, shadow=self.shadow, context="opening")

        if self.terminal_kind == "factorial":
            factorial_cells = {
                (y, p, q): 0
                for y in (False, True)
                for p in (False, True)
                for q in (False, True)
            }
            for observation in terminal:
                index = observation.scene_index
                key = (
                    observation.accepted,
                    observation.scene.placard == "sun",
                    self.shadow.truth[index],
                )
                factorial_cells[key] += 1
            if set(factorial_cells.values()) != {2}:
                raise EpisodeValidationError(
                    "factorial terminal must contain exactly two scenes from every (Y,P,Q) cell"
                )
        else:
            if sum(observation.accepted for observation in terminal) != TRAIN_LIKE_TERMINAL_SIZE // 2:
                raise EpisodeValidationError("train-like terminal target labels must be 5/5 balanced")
            _validate_proxy_pattern(
                terminal,
                regime=self.regime,
                shadow=self.shadow,
                context="terminal",
            )

        space = self.opening_version_space(catalog)
        if self.target.index not in space.indices:
            raise EpisodeValidationError("opening evidence eliminated its own target")
        if not 8 <= len(space) <= 64:
            raise EpisodeValidationError(
                f"opening version space must contain 8--64 rules, received {len(space)}"
            )

    def opening_version_space(self, catalog: RuleCatalog | None = None) -> VersionSpace:
        selected = build_rule_catalog() if catalog is None else catalog
        return selected.version_space(
            (observation.scene_index, observation.accepted) for observation in self.opening
        )

    @property
    def target_shadow_cells(self) -> dict[tuple[bool, bool], int]:
        return _target_shadow_cells(self.target, self.shadow)

    def as_obj(self) -> dict[str, Any]:
        catalog = build_rule_catalog()
        return {
            "schema_version": EPISODE_SCHEMA_VERSION,
            "episode_id": self.episode_id,
            "catalog_digest": catalog.digest,
            "target_rule_id": self.target.rule_id,
            "target_truth_digest": self.target.truth_digest,
            "shadow_rule_id": self.shadow.rule_id,
            "shadow_truth_digest": self.shadow.truth_digest,
            "regime": self.regime,
            "terminal_kind": self.terminal_kind,
            "renderer": self.renderer,
            "opening": [observation.as_obj() for observation in self.opening],
            "terminal": [observation.as_obj() for observation in self.terminal],
        }

    @property
    def digest(self) -> str:
        return json_digest(self.as_obj(), domain="goalzendo-interactive-hidden-episode-v1")


def observation_from_obj(value: Any) -> Observation:
    if type(value) is not dict or set(value) != {"scene_index", "accepted"} or len(value) != 2:
        raise EpisodeValidationError("observation must have exactly scene_index and accepted")
    return Observation(value["scene_index"], value["accepted"])


def hidden_episode_from_obj(
    value: Any,
    *,
    catalog: RuleCatalog | None = None,
) -> HiddenEpisode:
    """Bind a strict canonical episode object back to the exact catalog."""

    expected = {
        "schema_version",
        "episode_id",
        "catalog_digest",
        "target_rule_id",
        "target_truth_digest",
        "shadow_rule_id",
        "shadow_truth_digest",
        "regime",
        "terminal_kind",
        "renderer",
        "opening",
        "terminal",
    }
    if type(value) is not dict or set(value) != expected or len(value) != len(expected):
        raise EpisodeValidationError("hidden episode object has noncanonical fields")
    if value["schema_version"] != EPISODE_SCHEMA_VERSION:
        raise EpisodeValidationError("unsupported hidden episode schema version")
    selected = build_rule_catalog() if catalog is None else catalog
    if value["catalog_digest"] != selected.digest:
        raise EpisodeValidationError("hidden episode catalog digest mismatch")

    def entry(rule_id: Any, truth_digest: Any, *, name: str) -> CatalogEntry:
        if (
            type(rule_id) is not str
            or not rule_id.startswith("g03r")
            or len(rule_id) != 9
            or not rule_id[4:].isdigit()
        ):
            raise EpisodeValidationError(f"invalid {name} rule id: {rule_id!r}")
        index = int(rule_id[4:])
        if not 0 <= index < len(selected):
            raise EpisodeValidationError(f"{name} rule id lies outside the catalog")
        result = selected[index]
        if result.rule_id != rule_id or result.truth_digest != truth_digest:
            raise EpisodeValidationError(f"{name} rule identity or truth digest mismatch")
        return result

    if type(value["opening"]) is not list or type(value["terminal"]) is not list:
        raise EpisodeValidationError("hidden episode opening and terminal must be arrays")
    result = HiddenEpisode(
        episode_id=value["episode_id"],
        target=entry(value["target_rule_id"], value["target_truth_digest"], name="target"),
        shadow=entry(value["shadow_rule_id"], value["shadow_truth_digest"], name="shadow"),
        opening=tuple(observation_from_obj(item) for item in value["opening"]),
        terminal=tuple(observation_from_obj(item) for item in value["terminal"]),
        regime=cast(OpeningRegime, value["regime"]),
        terminal_kind=cast(TerminalKind, value["terminal_kind"]),
        renderer=cast(RendererName, value["renderer"]),
    )
    if result.as_obj() != value:
        raise EpisodeValidationError("hidden episode object is valid but not canonical")
    return result


def serialize_hidden_episode(episode: HiddenEpisode) -> str:
    if type(episode) is not HiddenEpisode:
        raise TypeError("serialize_hidden_episode requires a HiddenEpisode")
    return dump_json(episode.as_obj())


def parse_hidden_episode(text: str, *, require_canonical: bool = True) -> HiddenEpisode:
    try:
        value = load_json(text)
    except CanonicalJSONError as exc:
        raise EpisodeValidationError(str(exc)) from exc
    result = hidden_episode_from_obj(value)
    if require_canonical and serialize_hidden_episode(result) != text:
        raise EpisodeValidationError("episode JSON is valid but not in canonical serialized form")
    return result


def terminal_labels(episode: HiddenEpisode) -> tuple[bool, ...]:
    return tuple(observation.accepted for observation in episode.terminal)


def terminal_classifications(episode: HiddenEpisode) -> tuple[str, ...]:
    return tuple("fits" if accepted else "does_not_fit" for accepted in terminal_labels(episode))
