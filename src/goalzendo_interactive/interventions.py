"""Exact, replayable scene-level interventions for G03 evaluation.

The edit metric is deliberately structural rather than lexical.  Every slot
has four coordinates: ``occupied`` plus nullable ``color``, ``shape``, and
``size``.  Consequently, changing one attribute on an occupied piece costs
one, while adding or removing a piece costs four.  The placard is a thirteenth
coordinate and is never counted as a piece-field edit.
"""

from __future__ import annotations

import hashlib
import itertools
from collections import Counter
from collections.abc import Iterator
from dataclasses import dataclass
from functools import cache, lru_cache
from typing import Any, Literal, cast

from ._json import CanonicalJSONError, dump_json, json_digest, load_json
from .generation import EpisodeBank, generate_episode_bank, small_fixture_bank_spec
from .rendering import RENDERERS, RendererName, render_scene, renderer_digest
from .schema import (
    COLORS,
    NONEMPTY_ARRANGEMENT_COUNT,
    PIECES,
    PLACARDS,
    POSITIONS,
    SCENE_COUNT,
    SHAPES,
    SIZES,
    Piece,
    Scene,
    scene_at,
    scene_index,
)

InterventionFamily = Literal["P", "Q", "Y", "distractor"]
Scalar = str | bool | None

INTERVENTION_FAMILIES: tuple[InterventionFamily, ...] = ("P", "Q", "Y", "distractor")
INTERVENTION_BANK_SCHEMA_VERSION = 1
INTERVENTION_GENERATOR_SCHEMA_VERSION = 1
STRUCTURAL_EDIT_METRIC_SCHEMA_VERSION = 1
STRUCTURAL_EDIT_METRIC_ID = "fixed-nullable-piece-fields-hamming-v1"
RENDER_LENGTH_METRIC_ID = "ascii-whitespace-fields-v1"

PIECE_FIELDS = ("occupied", "color", "shape", "size")
STRUCTURAL_COORDINATES = (
    *(f"{position}.{field}" for position in POSITIONS for field in PIECE_FIELDS),
    "placard",
)

_CANDIDATE_SET_DOMAIN = b"goalzendo-interactive-intervention-candidate-set-v1\0"
_SELECTION_DOMAIN = b"goalzendo-interactive-intervention-selection-v1\0"
_RENDERED_SCENE_DOMAIN = "goalzendo-interactive-rendered-scene-v1"
_RESERVED_SCENES_DOMAIN = "goalzendo-interactive-intervention-reserved-scenes-v1"


class InterventionValidationError(ValueError):
    """Raised when an intervention or its provenance fails closed."""


class InterventionGenerationError(RuntimeError):
    """Raised when an exact intervention family has no eligible edit."""


def _valid_digest(value: object) -> bool:
    return (
        type(value) is str
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


@lru_cache(maxsize=8)
def _source_bank_digest(source_bank: EpisodeBank) -> str:
    """Memoize an otherwise intentionally fully derived manifest digest."""

    return source_bank.digest


@lru_cache(maxsize=1024)
def _episode_digest(episode: Any) -> str:
    return cast(str, episode.digest)


def structural_edit_metric_obj() -> dict[str, Any]:
    """Return the explicit, versioned scene edit metric."""

    return {
        "schema_version": STRUCTURAL_EDIT_METRIC_SCHEMA_VERSION,
        "metric_id": STRUCTURAL_EDIT_METRIC_ID,
        "coordinate_order": list(STRUCTURAL_COORDINATES),
        "piece_field_distance": "Hamming distance over the first 12 coordinates",
        "total_structural_distance": "piece-field distance plus placard Hamming distance",
        "empty_slot_encoding": {
            "occupied": False,
            "color": None,
            "shape": None,
            "size": None,
        },
        "occupied_slot_encoding": {
            "occupied": True,
            "color": "piece.color",
            "shape": "piece.shape",
            "size": "piece.size",
        },
        "primitive_costs": {
            "one_occupied_piece_attribute_substitution": 1,
            "piece_addition_or_removal": 4,
            "placard_flip_piece_field_cost": 0,
            "placard_flip_total_cost": 1,
        },
        "ordered_pairs": True,
        "minimum_search": "ascending exact structural distance before selection",
        "candidate_set_digest_encoding": (
            "ascending ordered pairs as big-endian uint16(base_scene_index), "
            "uint16(intervention_scene_index)"
        ),
        "selection_tiebreak_encoding": (
            "dependency digests, family, and the same big-endian ordered scene-index pair"
        ),
    }


def structural_edit_metric_digest() -> str:
    return json_digest(
        structural_edit_metric_obj(),
        domain="goalzendo-interactive-structural-edit-metric-v1",
    )


def _coordinate_values(scene: Scene) -> dict[str, Scalar]:
    values: dict[str, Scalar] = {}
    for position in POSITIONS:
        piece = scene.piece_at(position)
        values[f"{position}.occupied"] = piece is not None
        values[f"{position}.color"] = None if piece is None else piece.color
        values[f"{position}.shape"] = None if piece is None else piece.shape
        values[f"{position}.size"] = None if piece is None else piece.size
    values["placard"] = scene.placard
    return values


def _allowed_values(path: str) -> tuple[Scalar, ...]:
    if path == "placard":
        return cast(tuple[Scalar, ...], PLACARDS)
    field = path.split(".", 1)[1]
    if field == "occupied":
        return (False, True)
    if field == "color":
        return (None, *COLORS)
    if field == "shape":
        return (None, *SHAPES)
    return (None, *SIZES)


def _scalar_text(value: Scalar) -> str:
    if value is None:
        return "null"
    if type(value) is bool:
        return "true" if value else "false"
    return value


@dataclass(frozen=True, slots=True)
class StructuralFieldEdit:
    """One changed coordinate under the registered structural metric."""

    path: str
    before: Scalar
    after: Scalar

    def __post_init__(self) -> None:
        if self.path not in STRUCTURAL_COORDINATES:
            raise InterventionValidationError(f"unknown structural field: {self.path!r}")
        allowed = _allowed_values(self.path)
        if self.before not in allowed or self.after not in allowed:
            raise InterventionValidationError(
                f"invalid value transition for {self.path}: {self.before!r} -> {self.after!r}"
            )
        if type(self.before) is not type(self.after) and None not in (self.before, self.after):
            raise InterventionValidationError("structural edit values have incompatible types")
        if self.before == self.after:
            raise InterventionValidationError("a structural field edit must change its value")

    @property
    def position(self) -> str | None:
        return None if self.path == "placard" else self.path.split(".", 1)[0]

    @property
    def attribute(self) -> str:
        return self.path if self.path == "placard" else self.path.split(".", 1)[1]

    @property
    def direction(self) -> str:
        return f"{self.attribute}:{_scalar_text(self.before)}->{_scalar_text(self.after)}"

    def as_obj(self) -> dict[str, Scalar]:
        return {"path": self.path, "before": self.before, "after": self.after}


def structural_field_edit_from_obj(value: Any) -> StructuralFieldEdit:
    if type(value) is not dict or set(value) != {"path", "before", "after"} or len(value) != 3:
        raise InterventionValidationError("structural field edit has noncanonical fields")
    result = StructuralFieldEdit(value["path"], value["before"], value["after"])
    if result.as_obj() != value:
        raise InterventionValidationError("structural field edit is valid but not canonical")
    return result


def structural_edits(before: Scene, after: Scene) -> tuple[StructuralFieldEdit, ...]:
    """Reconstruct every changed coordinate in canonical metric order."""

    if type(before) is not Scene or type(after) is not Scene:
        raise TypeError("structural_edits requires two canonical Scene objects")
    before_values = _coordinate_values(before)
    after_values = _coordinate_values(after)
    return tuple(
        StructuralFieldEdit(path, before_values[path], after_values[path])
        for path in STRUCTURAL_COORDINATES
        if before_values[path] != after_values[path]
    )


def piece_field_distance(before: Scene, after: Scene) -> int:
    return sum(edit.path != "placard" for edit in structural_edits(before, after))


def total_structural_distance(before: Scene, after: Scene) -> int:
    return len(structural_edits(before, after))


@dataclass(frozen=True, slots=True)
class InterventionTruth:
    """Exact Official-Law, placard, and shadow-rule values for a scene."""

    y: bool
    p: bool
    q: bool

    def __post_init__(self) -> None:
        if type(self.y) is not bool or type(self.p) is not bool or type(self.q) is not bool:
            raise InterventionValidationError("Y, P, and Q values must be Boolean")

    def as_obj(self) -> dict[str, bool]:
        return {"Y": self.y, "P": self.p, "Q": self.q}


def intervention_truth_from_obj(value: Any) -> InterventionTruth:
    if type(value) is not dict or set(value) != {"Y", "P", "Q"} or len(value) != 3:
        raise InterventionValidationError("intervention truth object has noncanonical fields")
    result = InterventionTruth(value["Y"], value["P"], value["Q"])
    if result.as_obj() != value:
        raise InterventionValidationError("intervention truth object is valid but not canonical")
    return result


def _rendered_digest(scene: Scene, renderer: RendererName) -> str:
    return json_digest(
        {"renderer": renderer, "text": render_scene(scene, renderer)},
        domain=_RENDERED_SCENE_DOMAIN,
    )


def _render_length(scene: Scene, renderer: RendererName) -> int:
    return len(render_scene(scene, renderer).split())


@dataclass(frozen=True, slots=True)
class InterventionRecord:
    """One selected pair plus its complete minimum-edit provenance."""

    record_id: str
    family: InterventionFamily
    episode_id: str
    episode_digest: str
    source_episode_bank_digest: str
    base_scene_index: int
    intervention_scene_index: int
    before: InterventionTruth
    after: InterventionTruth
    changed_fields: tuple[StructuralFieldEdit, ...]
    piece_field_distance: int
    total_structural_distance: int
    minimum_structural_distance: int
    eligible_minimum_edit_count: int
    eligible_minimum_edit_digest: str
    selection_tiebreak_digest: str
    renderer: RendererName
    renderer_registry_digest: str
    rendered_before_digest: str
    rendered_after_digest: str
    render_length_metric: str
    render_length_bin: int

    def __post_init__(self) -> None:
        if type(self.record_id) is not str or not self.record_id:
            raise InterventionValidationError("intervention record id cannot be empty")
        if self.family not in INTERVENTION_FAMILIES:
            raise InterventionValidationError(f"unknown intervention family: {self.family!r}")
        if type(self.episode_id) is not str or not self.episode_id:
            raise InterventionValidationError("intervention episode id cannot be empty")
        for name in (
            "episode_digest",
            "source_episode_bank_digest",
            "eligible_minimum_edit_digest",
            "selection_tiebreak_digest",
            "renderer_registry_digest",
            "rendered_before_digest",
            "rendered_after_digest",
        ):
            if not _valid_digest(getattr(self, name)):
                raise InterventionValidationError(f"{name} must be a SHA-256 digest")
        for name in ("base_scene_index", "intervention_scene_index"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value < SCENE_COUNT:
                raise InterventionValidationError(f"{name} lies outside the scene universe")
        if self.base_scene_index == self.intervention_scene_index:
            raise InterventionValidationError("intervention scenes must be nonidentical")
        if type(self.before) is not InterventionTruth or type(self.after) is not InterventionTruth:
            raise InterventionValidationError("intervention record requires exact truth triples")
        edits = tuple(self.changed_fields)
        object.__setattr__(self, "changed_fields", edits)
        if not edits or any(type(edit) is not StructuralFieldEdit for edit in edits):
            raise InterventionValidationError("intervention record requires changed structural fields")
        coordinate_ranks = [STRUCTURAL_COORDINATES.index(edit.path) for edit in edits]
        if coordinate_ranks != sorted(coordinate_ranks) or len(set(coordinate_ranks)) != len(edits):
            raise InterventionValidationError("changed structural fields must be unique and canonical")
        integer_bounds = {
            "piece_field_distance": (self.piece_field_distance, 0),
            "total_structural_distance": (self.total_structural_distance, 1),
            "minimum_structural_distance": (self.minimum_structural_distance, 1),
            "eligible_minimum_edit_count": (self.eligible_minimum_edit_count, 1),
            "render_length_bin": (self.render_length_bin, 1),
        }
        for name, (value, minimum) in integer_bounds.items():
            if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
                raise InterventionValidationError(f"{name} must be an integer >= {minimum}")
        if self.minimum_structural_distance != self.total_structural_distance:
            raise InterventionValidationError("selected edit does not have the attested minimum distance")
        if self.renderer not in RENDERERS:
            raise InterventionValidationError(f"unknown renderer: {self.renderer!r}")
        if self.render_length_metric != RENDER_LENGTH_METRIC_ID:
            raise InterventionValidationError("unknown render-length metric")

    def as_obj(self) -> dict[str, Any]:
        return {
            "record_id": self.record_id,
            "family": self.family,
            "episode_id": self.episode_id,
            "episode_digest": self.episode_digest,
            "source_episode_bank_digest": self.source_episode_bank_digest,
            "base_scene_index": self.base_scene_index,
            "intervention_scene_index": self.intervention_scene_index,
            "before": self.before.as_obj(),
            "after": self.after.as_obj(),
            "changed_fields": [edit.as_obj() for edit in self.changed_fields],
            "piece_field_distance": self.piece_field_distance,
            "total_structural_distance": self.total_structural_distance,
            "minimum_structural_distance": self.minimum_structural_distance,
            "eligible_minimum_edit_count": self.eligible_minimum_edit_count,
            "eligible_minimum_edit_digest": self.eligible_minimum_edit_digest,
            "selection_tiebreak_digest": self.selection_tiebreak_digest,
            "renderer": self.renderer,
            "renderer_registry_digest": self.renderer_registry_digest,
            "rendered_before_digest": self.rendered_before_digest,
            "rendered_after_digest": self.rendered_after_digest,
            "render_length_metric": self.render_length_metric,
            "render_length_bin": self.render_length_bin,
        }


def intervention_record_from_obj(value: Any) -> InterventionRecord:
    expected = {
        "record_id",
        "family",
        "episode_id",
        "episode_digest",
        "source_episode_bank_digest",
        "base_scene_index",
        "intervention_scene_index",
        "before",
        "after",
        "changed_fields",
        "piece_field_distance",
        "total_structural_distance",
        "minimum_structural_distance",
        "eligible_minimum_edit_count",
        "eligible_minimum_edit_digest",
        "selection_tiebreak_digest",
        "renderer",
        "renderer_registry_digest",
        "rendered_before_digest",
        "rendered_after_digest",
        "render_length_metric",
        "render_length_bin",
    }
    if type(value) is not dict or set(value) != expected or len(value) != len(expected):
        raise InterventionValidationError("intervention record has noncanonical fields")
    if type(value["changed_fields"]) is not list:
        raise InterventionValidationError("changed_fields must be an array")
    result = InterventionRecord(
        record_id=value["record_id"],
        family=cast(InterventionFamily, value["family"]),
        episode_id=value["episode_id"],
        episode_digest=value["episode_digest"],
        source_episode_bank_digest=value["source_episode_bank_digest"],
        base_scene_index=value["base_scene_index"],
        intervention_scene_index=value["intervention_scene_index"],
        before=intervention_truth_from_obj(value["before"]),
        after=intervention_truth_from_obj(value["after"]),
        changed_fields=tuple(
            structural_field_edit_from_obj(edit) for edit in value["changed_fields"]
        ),
        piece_field_distance=value["piece_field_distance"],
        total_structural_distance=value["total_structural_distance"],
        minimum_structural_distance=value["minimum_structural_distance"],
        eligible_minimum_edit_count=value["eligible_minimum_edit_count"],
        eligible_minimum_edit_digest=value["eligible_minimum_edit_digest"],
        selection_tiebreak_digest=value["selection_tiebreak_digest"],
        renderer=cast(RendererName, value["renderer"]),
        renderer_registry_digest=value["renderer_registry_digest"],
        rendered_before_digest=value["rendered_before_digest"],
        rendered_after_digest=value["rendered_after_digest"],
        render_length_metric=value["render_length_metric"],
        render_length_bin=value["render_length_bin"],
    )
    if result.as_obj() != value:
        raise InterventionValidationError("intervention record is valid but not canonical")
    return result


def _reserved_indices(source_bank: EpisodeBank) -> frozenset[int]:
    return frozenset(
        observation.scene_index
        for episode in source_bank.episodes
        for observation in (*episode.opening, *episode.terminal)
    )


def _reserved_digest(indices: frozenset[int]) -> str:
    return json_digest(sorted(indices), domain=_RESERVED_SCENES_DOMAIN)


def _truth(episode: Any, index: int) -> InterventionTruth:
    return InterventionTruth(
        y=episode.target.truth[index],
        p=index < NONEMPTY_ARRANGEMENT_COUNT,
        q=episode.shadow.truth[index],
    )


def _intended_change(
    family: InterventionFamily,
    before: InterventionTruth,
    after: InterventionTruth,
) -> bool:
    if family == "P":
        return before.y == after.y and before.p != after.p and before.q == after.q
    if family == "Q":
        return before.y == after.y and before.p == after.p and before.q != after.q
    if family == "Y":
        return before.y != after.y and before.p == after.p and before.q == after.q
    return before == after


def _slot_distance(before: Piece | None, after: Piece | None) -> int:
    if before is None and after is None:
        return 0
    if before is None or after is None:
        return 4
    return sum(
        getattr(before, field) != getattr(after, field) for field in ("color", "shape", "size")
    )


@cache
def _slot_options(value: Piece | None, distance: int) -> tuple[Piece | None, ...]:
    return tuple(candidate for candidate in (None, *PIECES) if _slot_distance(value, candidate) == distance)


@cache
def _piece_neighbors_at_distance(base_index: int, distance: int) -> tuple[int, ...]:
    base = scene_at(base_index)
    neighbors: set[int] = set()
    for left_distance in range(5):
        for center_distance in range(5):
            right_distance = distance - left_distance - center_distance
            if not 0 <= right_distance <= 4:
                continue
            for left, center, right in itertools.product(
                _slot_options(base.left, left_distance),
                _slot_options(base.center, center_distance),
                _slot_options(base.right, right_distance),
            ):
                if left is None and center is None and right is None:
                    continue
                candidate = Scene(left=left, center=center, right=right, placard=base.placard)
                candidate_index = scene_index(candidate)
                if candidate_index != base_index:
                    neighbors.add(candidate_index)
    return tuple(sorted(neighbors))


@dataclass(frozen=True, slots=True)
class _Candidate:
    base_index: int
    intervention_index: int
    before: InterventionTruth
    after: InterventionTruth
    edits: tuple[StructuralFieldEdit, ...]

    @property
    def piece_distance(self) -> int:
        return sum(edit.path != "placard" for edit in self.edits)

    @property
    def total_distance(self) -> int:
        return len(self.edits)


def _candidate_tiebreak(
    source_bank_digest: str,
    episode_digest: str,
    family: InterventionFamily,
    candidate: _Candidate,
) -> str:
    digest = hashlib.sha256()
    digest.update(_SELECTION_DOMAIN)
    for part in (source_bank_digest, episode_digest, family):
        digest.update(part.encode("ascii"))
        digest.update(b"\0")
    digest.update(candidate.base_index.to_bytes(2, "big"))
    digest.update(candidate.intervention_index.to_bytes(2, "big"))
    return digest.hexdigest()


def _placard_candidates(
    episode: Any,
    forbidden: frozenset[int],
) -> Iterator[_Candidate]:
    for base_index in range(SCENE_COUNT):
        if base_index in forbidden:
            continue
        if base_index < NONEMPTY_ARRANGEMENT_COUNT:
            intervention_index = base_index + NONEMPTY_ARRANGEMENT_COUNT
        else:
            intervention_index = base_index - NONEMPTY_ARRANGEMENT_COUNT
        if intervention_index in forbidden:
            continue
        before_placard = "sun" if base_index < NONEMPTY_ARRANGEMENT_COUNT else "moon"
        after_placard = "moon" if before_placard == "sun" else "sun"
        yield _Candidate(
            base_index,
            intervention_index,
            _truth(episode, base_index),
            _truth(episode, intervention_index),
            (StructuralFieldEdit("placard", before_placard, after_placard),),
        )


@cache
def _edge_edits(base_index: int, intervention_index: int) -> tuple[StructuralFieldEdit, ...]:
    return structural_edits(scene_at(base_index), scene_at(intervention_index))


def _piece_candidates(
    episode: Any,
    family: InterventionFamily,
    forbidden: frozenset[int],
    distance: int,
) -> Iterator[_Candidate]:
    for base_index in range(SCENE_COUNT):
        if base_index in forbidden:
            continue
        before = _truth(episode, base_index)
        for intervention_index in _piece_neighbors_at_distance(base_index, distance):
            if intervention_index in forbidden:
                continue
            after = _truth(episode, intervention_index)
            if not _intended_change(family, before, after):
                continue
            if distance != 1 and _render_length(
                scene_at(base_index), episode.renderer
            ) != _render_length(scene_at(intervention_index), episode.renderer):
                continue
            yield _Candidate(
                base_index,
                intervention_index,
                before,
                after,
                _edge_edits(base_index, intervention_index),
            )


def _direction_keys(*, include_placard: bool) -> tuple[str, ...]:
    values: dict[str, tuple[Scalar, ...]] = {
        "occupied": (False, True),
        "color": (None, *COLORS),
        "shape": (None, *SHAPES),
        "size": (None, *SIZES),
        "placard": cast(tuple[Scalar, ...], PLACARDS),
    }
    attributes = (*PIECE_FIELDS, "placard") if include_placard else PIECE_FIELDS
    return tuple(
        f"{attribute}:{_scalar_text(before)}->{_scalar_text(after)}"
        for attribute in attributes
        for before in values[attribute]
        for after in values[attribute]
        if before != after
    )


_PIECE_DIRECTION_KEYS = _direction_keys(include_placard=False)
_PLACARD_DIRECTION_KEYS = tuple(
    key for key in _direction_keys(include_placard=True) if key.startswith("placard:")
)


class _BalanceState:
    def __init__(self) -> None:
        self.positions = {family: Counter[str]() for family in INTERVENTION_FAMILIES}
        self.attributes = {family: Counter[str]() for family in INTERVENTION_FAMILIES}
        self.directions = {family: Counter[str]() for family in INTERVENTION_FAMILIES}

    @staticmethod
    def features(
        candidate: _Candidate,
    ) -> tuple[Counter[str], Counter[str], Counter[str]]:
        positions: Counter[str] = Counter()
        attributes: Counter[str] = Counter()
        directions: Counter[str] = Counter()
        for edit in candidate.edits:
            if edit.position is not None:
                positions[edit.position] += 1
            attributes[edit.attribute] += 1
            directions[edit.direction] += 1
        return positions, attributes, directions

    @staticmethod
    def _sum_square_delta(
        counter: Counter[str],
        increment: Counter[str],
    ) -> int:
        return sum(
            (counter[key] + value) ** 2 - counter[key] ** 2
            for key, value in increment.items()
        )

    def score(
        self,
        family: InterventionFamily,
        candidate: _Candidate,
    ) -> tuple[int, int]:
        positions, attributes, directions = self.features(candidate)
        if family == "P":
            direction_delta = self._sum_square_delta(
                self.directions[family], directions
            )
            return (0, direction_delta)

        position_delta = self._sum_square_delta(
            self.positions[family], positions
        )
        attribute_delta = self._sum_square_delta(
            self.attributes[family], attributes
        )
        direction_delta = self._sum_square_delta(
            self.directions[family], directions
        )
        uniform_penalty = 32 * position_delta + 24 * attribute_delta + 3 * direction_delta
        match_penalty = 0
        if family == "distractor":
            for weight, own, increment, left, right in (
                (
                    32,
                    self.positions[family],
                    positions,
                    self.positions["Q"],
                    self.positions["Y"],
                ),
                (
                    24,
                    self.attributes[family],
                    attributes,
                    self.attributes["Q"],
                    self.attributes["Y"],
                ),
                (
                    3,
                    self.directions[family],
                    directions,
                    self.directions["Q"],
                    self.directions["Y"],
                ),
            ):
                match_penalty += weight * sum(
                    (2 * (own[key] + value) - left[key] - right[key]) ** 2
                    - (2 * own[key] - left[key] - right[key]) ** 2
                    for key, value in increment.items()
                )
        return (match_penalty, uniform_penalty)

    def add(self, family: InterventionFamily, candidate: _Candidate) -> None:
        positions, attributes, directions = self.features(candidate)
        self.positions[family].update(positions)
        self.attributes[family].update(attributes)
        self.directions[family].update(directions)


def _candidate_set_header(
    source_bank_digest: str,
    episode_digest: str,
    family: InterventionFamily,
    distance: int,
) -> Any:
    digest = hashlib.sha256()
    digest.update(_CANDIDATE_SET_DOMAIN)
    for part in (source_bank_digest, episode_digest, family, str(distance)):
        digest.update(part.encode("ascii"))
        digest.update(b"\0")
    return digest


def _select_candidate(
    source_bank: EpisodeBank,
    episode: Any,
    family: InterventionFamily,
    forbidden: frozenset[int],
    balance: _BalanceState,
) -> tuple[_Candidate, int, int, str, str]:
    source_digest = _source_bank_digest(source_bank)
    selected_episode_digest = _episode_digest(episode)
    distances = (1,) if family == "P" else tuple(range(1, 13))
    for distance in distances:
        candidates = (
            _placard_candidates(episode, forbidden)
            if family == "P"
            else _piece_candidates(episode, family, forbidden, distance)
        )
        set_digest = _candidate_set_header(
            source_digest, selected_episode_digest, family, distance
        )
        count = 0
        best: _Candidate | None = None
        best_tiebreak = ""
        best_score: tuple[int, int] | None = None
        for candidate in candidates:
            if candidate.total_distance != distance:
                raise InterventionGenerationError("candidate enumerator violated edit distance")
            set_digest.update(candidate.base_index.to_bytes(2, "big"))
            set_digest.update(candidate.intervention_index.to_bytes(2, "big"))
            count += 1
            score = balance.score(family, candidate)
            if best_score is None or score < best_score:
                tiebreak = _candidate_tiebreak(
                    source_digest, selected_episode_digest, family, candidate
                )
                best = candidate
                best_tiebreak = tiebreak
                best_score = score
            elif score == best_score:
                tiebreak = _candidate_tiebreak(
                    source_digest, selected_episode_digest, family, candidate
                )
                if tiebreak < best_tiebreak:
                    best = candidate
                    best_tiebreak = tiebreak
        if best is not None:
            return best, distance, count, set_digest.hexdigest(), best_tiebreak
    raise InterventionGenerationError(
        f"no exact {family} intervention exists for episode {episode.episode_id!r} "
        "under the registered metric, render-length match, and disjointness constraints"
    )


def _make_record(
    source_bank: EpisodeBank,
    episode: Any,
    family: InterventionFamily,
    candidate: _Candidate,
    minimum_distance: int,
    candidate_count: int,
    candidate_digest: str,
    tiebreak: str,
) -> InterventionRecord:
    before_scene = scene_at(candidate.base_index)
    after_scene = scene_at(candidate.intervention_index)
    before_length = _render_length(before_scene, episode.renderer)
    after_length = _render_length(after_scene, episode.renderer)
    if before_length != after_length:  # pragma: no cover - enumerator invariant
        raise InterventionGenerationError("selected render-length bins differ")
    return InterventionRecord(
        record_id=f"{episode.episode_id}:{family}",
        family=family,
        episode_id=episode.episode_id,
        episode_digest=_episode_digest(episode),
        source_episode_bank_digest=_source_bank_digest(source_bank),
        base_scene_index=candidate.base_index,
        intervention_scene_index=candidate.intervention_index,
        before=candidate.before,
        after=candidate.after,
        changed_fields=candidate.edits,
        piece_field_distance=candidate.piece_distance,
        total_structural_distance=candidate.total_distance,
        minimum_structural_distance=minimum_distance,
        eligible_minimum_edit_count=candidate_count,
        eligible_minimum_edit_digest=candidate_digest,
        selection_tiebreak_digest=tiebreak,
        renderer=episode.renderer,
        renderer_registry_digest=renderer_digest(),
        rendered_before_digest=_rendered_digest(before_scene, episode.renderer),
        rendered_after_digest=_rendered_digest(after_scene, episode.renderer),
        render_length_metric=RENDER_LENGTH_METRIC_ID,
        render_length_bin=before_length,
    )


def _count_obj(counter: Counter[str], keys: tuple[str, ...]) -> dict[str, int]:
    return {key: counter[key] for key in keys}


def _balance_obj(records: tuple[InterventionRecord, ...]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for family in INTERVENTION_FAMILIES:
        selected = tuple(record for record in records if record.family == family)
        positions: Counter[str] = Counter()
        attributes: Counter[str] = Counter()
        directions: Counter[str] = Counter()
        distances: Counter[int] = Counter()
        for record in selected:
            distances[record.total_structural_distance] += 1
            for edit in record.changed_fields:
                if edit.position is not None:
                    positions[edit.position] += 1
                attributes[edit.attribute] += 1
                directions[edit.direction] += 1
        direction_keys = _PLACARD_DIRECTION_KEYS if family == "P" else _PIECE_DIRECTION_KEYS
        result[family] = {
            "record_count": len(selected),
            "distance_counts": {
                str(distance): distances[distance] for distance in sorted(distances)
            },
            "position_counts": _count_obj(
                positions, cast(tuple[str, ...], POSITIONS)
            ),
            "attribute_counts": _count_obj(attributes, (*PIECE_FIELDS, "placard")),
            "direction_counts": _count_obj(directions, direction_keys),
        }
    return result


@dataclass(frozen=True, slots=True)
class InterventionBank:
    """A dependency-bound canonical bank with one pair per family and episode."""

    bank_id: str
    source_episode_bank: EpisodeBank
    records: tuple[InterventionRecord, ...]

    def __post_init__(self) -> None:
        if type(self.bank_id) is not str or not self.bank_id.strip():
            raise InterventionValidationError("intervention bank id cannot be empty")
        if type(self.source_episode_bank) is not EpisodeBank:
            raise InterventionValidationError("intervention bank requires its exact source bank")
        records = tuple(self.records)
        object.__setattr__(self, "records", records)
        expected_count = len(self.source_episode_bank.episodes) * len(INTERVENTION_FAMILIES)
        if len(records) != expected_count:
            raise InterventionValidationError(
                f"intervention bank requires exactly {expected_count} records"
            )
        if any(type(record) is not InterventionRecord for record in records):
            raise InterventionValidationError("intervention bank contains a non-record")
        expected_order = tuple(
            (family, episode.episode_id)
            for family in INTERVENTION_FAMILIES
            for episode in self.source_episode_bank.episodes
        )
        actual_order = tuple((record.family, record.episode_id) for record in records)
        if actual_order != expected_order:
            raise InterventionValidationError("intervention records are not in canonical order")

        episodes = {episode.episode_id: episode for episode in self.source_episode_bank.episodes}
        reserved = _reserved_indices(self.source_episode_bank)
        used: set[int] = set()
        for record in records:
            episode = episodes[record.episode_id]
            self._validate_record(record, episode)
            pair = {record.base_scene_index, record.intervention_scene_index}
            if pair & reserved:
                raise InterventionValidationError(
                    "intervention scene overlaps an opening or terminal scene"
                )
            if pair & used:
                raise InterventionValidationError("base/intervention scenes may not be reused")
            used.update(pair)

    def _validate_record(self, record: InterventionRecord, episode: Any) -> None:
        if (
            record.record_id != f"{episode.episode_id}:{record.family}"
            or record.episode_digest != _episode_digest(episode)
            or record.source_episode_bank_digest != _source_bank_digest(self.source_episode_bank)
        ):
            raise InterventionValidationError("intervention record dependency binding mismatch")
        if record.renderer != episode.renderer or record.renderer_registry_digest != renderer_digest():
            raise InterventionValidationError("intervention renderer binding mismatch")
        before_scene = scene_at(record.base_scene_index)
        after_scene = scene_at(record.intervention_scene_index)
        edits = structural_edits(before_scene, after_scene)
        if edits != record.changed_fields:
            raise InterventionValidationError("stored structural edits do not reconstruct")
        if piece_field_distance(before_scene, after_scene) != record.piece_field_distance:
            raise InterventionValidationError("stored piece-field distance does not reconstruct")
        if total_structural_distance(before_scene, after_scene) != record.total_structural_distance:
            raise InterventionValidationError("stored total structural distance does not reconstruct")
        before = _truth(episode, record.base_scene_index)
        after = _truth(episode, record.intervention_scene_index)
        if before != record.before or after != record.after:
            raise InterventionValidationError("stored Y/P/Q truth values do not reconstruct")
        if not _intended_change(record.family, before, after):
            raise InterventionValidationError("intervention does not make its registered causal change")
        placard_edits = tuple(edit for edit in edits if edit.path == "placard")
        if record.family == "P":
            if record.piece_field_distance != 0 or len(placard_edits) != 1 or len(edits) != 1:
                raise InterventionValidationError("P intervention must flip only the placard")
        elif placard_edits or record.piece_field_distance < 1:
            raise InterventionValidationError(
                f"{record.family} intervention must edit pieces and preserve the placard"
            )
        if (
            record.rendered_before_digest != _rendered_digest(before_scene, episode.renderer)
            or record.rendered_after_digest != _rendered_digest(after_scene, episode.renderer)
        ):
            raise InterventionValidationError("rendered scene digest mismatch")
        before_length = _render_length(before_scene, episode.renderer)
        after_length = _render_length(after_scene, episode.renderer)
        if before_length != after_length or record.render_length_bin != before_length:
            raise InterventionValidationError("renderer length-bin match does not reconstruct")
        expected_tiebreak = _candidate_tiebreak(
            _source_bank_digest(self.source_episode_bank),
            _episode_digest(episode),
            record.family,
            _Candidate(
                record.base_scene_index,
                record.intervention_scene_index,
                before,
                after,
                edits,
            ),
        )
        if record.selection_tiebreak_digest != expected_tiebreak:
            raise InterventionValidationError("selection SHA tie-break does not reconstruct")

    @property
    def reserved_scene_indices(self) -> frozenset[int]:
        return _reserved_indices(self.source_episode_bank)

    @property
    def family_counts(self) -> dict[str, int]:
        counts = Counter(record.family for record in self.records)
        return {family: counts[family] for family in INTERVENTION_FAMILIES}

    @property
    def balance(self) -> dict[str, Any]:
        return _balance_obj(self.records)

    def as_obj(self) -> dict[str, Any]:
        reserved = self.reserved_scene_indices
        return {
            "schema_version": INTERVENTION_BANK_SCHEMA_VERSION,
            "generator_schema_version": INTERVENTION_GENERATOR_SCHEMA_VERSION,
            "bank_id": self.bank_id,
            "source_episode_bank_id": self.source_episode_bank.spec.bank_id,
            "source_episode_bank_digest": _source_bank_digest(self.source_episode_bank),
            "catalog_digest": self.source_episode_bank.catalog_digest,
            "metric": structural_edit_metric_obj(),
            "metric_digest": structural_edit_metric_digest(),
            "renderer_registry_digest": renderer_digest(),
            "families": list(INTERVENTION_FAMILIES),
            "family_counts": self.family_counts,
            "reserved_scene_count": len(reserved),
            "reserved_scene_digest": _reserved_digest(reserved),
            "selected_scene_count": len(self.records) * 2,
            "balance": self.balance,
            "records": [record.as_obj() for record in self.records],
        }

    @property
    def digest(self) -> str:
        return json_digest(
            self.as_obj(), domain="goalzendo-interactive-intervention-bank-v1"
        )


@lru_cache(maxsize=4)
def generate_intervention_bank(
    source_episode_bank: EpisodeBank,
    bank_id: str | None = None,
) -> InterventionBank:
    """Exhaust exact minimum edits and select a balanced canonical bank."""

    if type(source_episode_bank) is not EpisodeBank:
        raise TypeError("generate_intervention_bank requires an EpisodeBank")
    selected_bank_id = (
        f"{source_episode_bank.spec.bank_id}-scene-interventions-v1"
        if bank_id is None
        else bank_id
    )
    reserved = _reserved_indices(source_episode_bank)
    used: set[int] = set()
    balance = _BalanceState()
    records: list[InterventionRecord] = []
    for family in INTERVENTION_FAMILIES:
        for episode in source_episode_bank.episodes:
            forbidden = frozenset((*reserved, *used))
            candidate, distance, count, candidate_digest, tiebreak = _select_candidate(
                source_episode_bank,
                episode,
                family,
                forbidden,
                balance,
            )
            record = _make_record(
                source_episode_bank,
                episode,
                family,
                candidate,
                distance,
                count,
                candidate_digest,
                tiebreak,
            )
            records.append(record)
            used.update((candidate.base_index, candidate.intervention_index))
            balance.add(family, candidate)
    return InterventionBank(selected_bank_id, source_episode_bank, tuple(records))


def small_fixture_intervention_bank() -> InterventionBank:
    """Generate the 12-episode engineering fixture, never a final study bank."""

    return generate_intervention_bank(generate_episode_bank(small_fixture_bank_spec()))


def serialize_intervention_bank(bank: InterventionBank) -> str:
    if type(bank) is not InterventionBank:
        raise TypeError("serialize_intervention_bank requires an InterventionBank")
    return dump_json(bank.as_obj())


def parse_intervention_bank(
    text: str,
    *,
    source_episode_bank: EpisodeBank,
    require_canonical: bool = True,
) -> InterventionBank:
    try:
        value = load_json(text)
    except CanonicalJSONError as exc:
        raise InterventionValidationError(str(exc)) from exc
    expected = {
        "schema_version",
        "generator_schema_version",
        "bank_id",
        "source_episode_bank_id",
        "source_episode_bank_digest",
        "catalog_digest",
        "metric",
        "metric_digest",
        "renderer_registry_digest",
        "families",
        "family_counts",
        "reserved_scene_count",
        "reserved_scene_digest",
        "selected_scene_count",
        "balance",
        "records",
    }
    if type(value) is not dict or set(value) != expected or len(value) != len(expected):
        raise InterventionValidationError("intervention bank manifest has noncanonical fields")
    if value["schema_version"] != INTERVENTION_BANK_SCHEMA_VERSION:
        raise InterventionValidationError("unsupported intervention bank schema version")
    if value["generator_schema_version"] != INTERVENTION_GENERATOR_SCHEMA_VERSION:
        raise InterventionValidationError("unsupported intervention generator schema version")
    if value["source_episode_bank_digest"] != _source_bank_digest(source_episode_bank):
        raise InterventionValidationError("source episode-bank digest mismatch")
    if type(value["records"]) is not list:
        raise InterventionValidationError("intervention records must be an array")
    result = InterventionBank(
        value["bank_id"],
        source_episode_bank,
        tuple(intervention_record_from_obj(item) for item in value["records"]),
    )
    if result.as_obj() != value:
        raise InterventionValidationError("intervention bank derived fields are inconsistent")
    if require_canonical and serialize_intervention_bank(result) != text:
        raise InterventionValidationError("intervention bank JSON is valid but not canonical")
    return verify_intervention_bank(result)


def verify_intervention_bank(bank: InterventionBank) -> InterventionBank:
    """Fail closed unless complete deterministic regeneration is byte-identical."""

    if type(bank) is not InterventionBank:
        raise TypeError("verify_intervention_bank requires an InterventionBank")
    regenerated = generate_intervention_bank(bank.source_episode_bank, bank.bank_id)
    if serialize_intervention_bank(regenerated) != serialize_intervention_bank(bank):
        raise InterventionValidationError(
            "intervention bank does not regenerate byte-for-byte from its dependencies"
        )
    return bank
