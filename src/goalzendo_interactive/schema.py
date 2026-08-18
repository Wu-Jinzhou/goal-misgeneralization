"""Immutable canonical pieces, scenes, and the complete G03 koan universe."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from typing import Any, Literal, cast

from ._json import CanonicalJSONError, dump_json, load_json

Color = Literal["red", "blue", "green"]
Shape = Literal["pyramid", "cube", "sphere"]
Size = Literal["small", "large"]
Position = Literal["left", "center", "right"]
Placard = Literal["sun", "moon"]

COLORS: tuple[Color, ...] = ("red", "blue", "green")
SHAPES: tuple[Shape, ...] = ("pyramid", "cube", "sphere")
SIZES: tuple[Size, ...] = ("small", "large")
POSITIONS: tuple[Position, ...] = ("left", "center", "right")
PLACARDS: tuple[Placard, ...] = ("sun", "moon")

PIECE_COUNT = len(COLORS) * len(SHAPES) * len(SIZES)
SLOT_VALUE_COUNT = 1 + PIECE_COUNT
NONEMPTY_ARRANGEMENT_COUNT = SLOT_VALUE_COUNT**len(POSITIONS) - 1
SCENE_COUNT = len(PLACARDS) * NONEMPTY_ARRANGEMENT_COUNT
SCENE_SCHEMA_VERSION = 1


class SceneValidationError(ValueError):
    """Raised when a piece or scene is outside the canonical finite world."""


def _require_exact_keys(value: Mapping[str, Any], expected: tuple[str, ...], *, path: str) -> None:
    actual = tuple(value)
    if set(actual) != set(expected) or len(actual) != len(expected):
        missing = sorted(set(expected) - set(actual))
        extra = sorted(set(actual) - set(expected))
        raise SceneValidationError(
            f"{path} must have exactly keys {expected!r}; missing={missing!r}, extra={extra!r}"
        )


@dataclass(frozen=True, slots=True)
class Piece:
    """One immutable tabletop piece."""

    color: Color
    shape: Shape
    size: Size

    def __post_init__(self) -> None:
        if self.color not in COLORS:
            raise SceneValidationError(f"unknown piece color: {self.color!r}")
        if self.shape not in SHAPES:
            raise SceneValidationError(f"unknown piece shape: {self.shape!r}")
        if self.size not in SIZES:
            raise SceneValidationError(f"unknown piece size: {self.size!r}")

    def as_obj(self) -> dict[str, str]:
        """Return fields in the protocol's canonical action order."""

        return {"size": self.size, "color": self.color, "shape": self.shape}


@dataclass(frozen=True, slots=True)
class Scene:
    """A nonempty three-position display and its visible placard."""

    left: Piece | None
    center: Piece | None
    right: Piece | None
    placard: Placard

    def __post_init__(self) -> None:
        for position in POSITIONS:
            piece = getattr(self, position)
            if piece is not None and type(piece) is not Piece:
                raise SceneValidationError(f"{position} must be a Piece or null")
        if self.left is None and self.center is None and self.right is None:
            raise SceneValidationError("the all-empty display is not a legal scene")
        if self.placard not in PLACARDS:
            raise SceneValidationError(f"unknown placard: {self.placard!r}")

    @property
    def occupied_count(self) -> int:
        return sum(getattr(self, position) is not None for position in POSITIONS)

    @property
    def pieces(self) -> tuple[Piece, ...]:
        return tuple(
            cast(Piece, piece)
            for position in POSITIONS
            if (piece := getattr(self, position)) is not None
        )

    def piece_at(self, position: Position) -> Piece | None:
        if position not in POSITIONS:
            raise SceneValidationError(f"unknown position: {position!r}")
        return cast(Piece | None, getattr(self, position))

    def as_obj(self) -> dict[str, Any]:
        return {
            "left": None if self.left is None else self.left.as_obj(),
            "center": None if self.center is None else self.center.as_obj(),
            "right": None if self.right is None else self.right.as_obj(),
            "placard": self.placard,
        }


PIECES: tuple[Piece, ...] = tuple(
    Piece(color=color, shape=shape, size=size)
    for color in COLORS
    for shape in SHAPES
    for size in SIZES
)
SLOT_VALUES: tuple[Piece | None, ...] = (None, *PIECES)
_SLOT_INDEX = {piece: index for index, piece in enumerate(SLOT_VALUES)}
_PLACARD_INDEX = {placard: index for index, placard in enumerate(PLACARDS)}


def piece_from_obj(value: Any, *, path: str = "piece") -> Piece:
    if type(value) is not dict:
        raise SceneValidationError(f"{path} must be a JSON object")
    _require_exact_keys(value, ("size", "color", "shape"), path=path)
    size, color, shape = value["size"], value["color"], value["shape"]
    if type(size) is not str or type(color) is not str or type(shape) is not str:
        raise SceneValidationError(f"{path} attributes must be strings")
    return Piece(color=cast(Color, color), shape=cast(Shape, shape), size=cast(Size, size))


def scene_from_obj(value: Any, *, path: str = "koan") -> Scene:
    if type(value) is not dict:
        raise SceneValidationError(f"{path} must be a JSON object")
    _require_exact_keys(value, ("left", "center", "right", "placard"), path=path)
    slots: dict[str, Piece | None] = {}
    for position in POSITIONS:
        raw_piece = value[position]
        slots[position] = (
            None if raw_piece is None else piece_from_obj(raw_piece, path=f"{path}.{position}")
        )
    placard = value["placard"]
    if type(placard) is not str:
        raise SceneValidationError(f"{path}.placard must be a string")
    return Scene(
        left=slots["left"],
        center=slots["center"],
        right=slots["right"],
        placard=cast(Placard, placard),
    )


def serialize_scene(scene: Scene) -> str:
    if type(scene) is not Scene:
        raise SceneValidationError("serialize_scene requires a Scene")
    return dump_json(scene.as_obj())


def parse_scene(text: str, *, require_canonical: bool = True) -> Scene:
    try:
        value = load_json(text)
    except CanonicalJSONError as exc:
        raise SceneValidationError(str(exc)) from exc
    scene = scene_from_obj(value)
    if require_canonical and serialize_scene(scene) != text:
        raise SceneValidationError("scene JSON is valid but not in canonical serialized form")
    return scene


def scene_index(scene: Scene) -> int:
    """Return the stable zero-based index of a canonical scene."""

    if type(scene) is not Scene:
        raise SceneValidationError("scene_index requires a Scene")
    raw = 0
    for position in POSITIONS:
        raw = raw * SLOT_VALUE_COUNT + _SLOT_INDEX[scene.piece_at(position)]
    # Scene validation guarantees raw != 0.
    return _PLACARD_INDEX[scene.placard] * NONEMPTY_ARRANGEMENT_COUNT + raw - 1


def scene_at(index: int) -> Scene:
    """Invert :func:`scene_index` without materializing the universe."""

    if isinstance(index, bool) or not isinstance(index, int) or not 0 <= index < SCENE_COUNT:
        raise IndexError(f"scene index must lie in [0, {SCENE_COUNT}): {index!r}")
    placard_index, local_index = divmod(index, NONEMPTY_ARRANGEMENT_COUNT)
    raw = local_index + 1
    slot_indices = [0, 0, 0]
    for offset in range(len(POSITIONS) - 1, -1, -1):
        raw, slot_indices[offset] = divmod(raw, SLOT_VALUE_COUNT)
    return Scene(
        left=SLOT_VALUES[slot_indices[0]],
        center=SLOT_VALUES[slot_indices[1]],
        right=SLOT_VALUES[slot_indices[2]],
        placard=PLACARDS[placard_index],
    )


def iter_scenes() -> Iterator[Scene]:
    """Yield all 13,716 unique scenes in stable index order."""

    return (scene_at(index) for index in range(SCENE_COUNT))
