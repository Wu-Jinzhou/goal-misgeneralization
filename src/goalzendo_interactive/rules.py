"""The complete finite G03 atom grammar and canonical rule AST."""

from __future__ import annotations

import itertools
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from typing import Any, TypeAlias, cast
from typing import Literal as TypingLiteral

from ._json import CanonicalJSONError, dump_json, load_json
from .schema import (
    COLORS,
    POSITIONS,
    SHAPES,
    SIZES,
    Position,
    Scene,
)

Attribute = TypingLiteral["color", "shape", "size"]
AtomOp = TypingLiteral[
    "slot_empty",
    "slot_attr",
    "exists",
    "at_least_two",
    "occupied_count_is",
    "same",
    "placard_is",
]
BinaryOp = TypingLiteral["all", "any", "exactly_one"]

ATTRIBUTES: tuple[Attribute, ...] = ("color", "shape", "size")
ATTRIBUTE_VALUES: dict[Attribute, tuple[str, ...]] = {
    "color": COLORS,
    "shape": SHAPES,
    "size": SIZES,
}
ATOM_OPS: tuple[AtomOp, ...] = (
    "slot_empty",
    "slot_attr",
    "exists",
    "at_least_two",
    "occupied_count_is",
    "same",
    "placard_is",
)
BINARY_OPS: tuple[BinaryOp, ...] = ("all", "any", "exactly_one")
ATOM_COUNT = 56
LITERAL_COUNT = ATOM_COUNT * 2
SYNTACTIC_RULE_COUNT = LITERAL_COUNT + len(BINARY_OPS) * (LITERAL_COUNT * (LITERAL_COUNT - 1) // 2)
RULE_SCHEMA_VERSION = 1


class RuleValidationError(ValueError):
    """Raised for an expression outside the released G03 grammar."""


def _position_order(position: str) -> int:
    try:
        return POSITIONS.index(cast(Position, position))
    except ValueError as exc:
        raise RuleValidationError(f"unknown position: {position!r}") from exc


def _check_attribute_value(attribute: str | None, value: str | int | None) -> None:
    if attribute not in ATTRIBUTES:
        raise RuleValidationError(f"unknown attribute: {attribute!r}")
    if type(value) is not str or value not in ATTRIBUTE_VALUES[cast(Attribute, attribute)]:
        raise RuleValidationError(f"invalid value {value!r} for attribute {attribute!r}")


@dataclass(frozen=True, slots=True)
class Atom:
    """One of the 56 canonical Boolean predicates."""

    op: AtomOp
    position: Position | None = None
    attribute: Attribute | None = None
    value: str | int | None = None
    position_1: Position | None = None
    position_2: Position | None = None

    def __post_init__(self) -> None:
        if self.op not in ATOM_OPS:
            raise RuleValidationError(f"unknown atom operation: {self.op!r}")

        if self.op == "slot_empty":
            self._require_fields(position=True)
            _position_order(cast(str, self.position))
        elif self.op == "slot_attr":
            self._require_fields(position=True, attribute=True, value=True)
            _position_order(cast(str, self.position))
            _check_attribute_value(self.attribute, self.value)
        elif self.op in {"exists", "at_least_two"}:
            self._require_fields(attribute=True, value=True)
            _check_attribute_value(self.attribute, self.value)
        elif self.op == "occupied_count_is":
            self._require_fields(value=True)
            if isinstance(self.value, bool) or not isinstance(self.value, int) or self.value not in {1, 2, 3}:
                raise RuleValidationError("occupied_count_is value must be one of 1, 2, 3")
        elif self.op == "same":
            self._require_fields(attribute=True, position_1=True, position_2=True)
            if self.attribute not in ATTRIBUTES:
                raise RuleValidationError(f"unknown attribute: {self.attribute!r}")
            first = _position_order(cast(str, self.position_1))
            second = _position_order(cast(str, self.position_2))
            if first == second:
                raise RuleValidationError("same requires two distinct positions")
            if first > second:
                old_first, old_second = self.position_1, self.position_2
                object.__setattr__(self, "position_1", old_second)
                object.__setattr__(self, "position_2", old_first)
        else:  # placard_is
            self._require_fields(value=True)
            if self.value != "sun":
                raise RuleValidationError("the released grammar contains only placard_is(sun)")

    def _require_fields(
        self,
        *,
        position: bool = False,
        attribute: bool = False,
        value: bool = False,
        position_1: bool = False,
        position_2: bool = False,
    ) -> None:
        requirements = {
            "position": position,
            "attribute": attribute,
            "value": value,
            "position_1": position_1,
            "position_2": position_2,
        }
        for field, required in requirements.items():
            present = getattr(self, field) is not None
            if present != required:
                state = "required" if required else "forbidden"
                raise RuleValidationError(f"{field} is {state} for {self.op}")

    def evaluate(self, scene: Scene) -> bool:
        if type(scene) is not Scene:
            raise RuleValidationError("atom evaluation requires a canonical Scene")
        if self.op == "slot_empty":
            return scene.piece_at(cast(Position, self.position)) is None
        if self.op == "slot_attr":
            piece = scene.piece_at(cast(Position, self.position))
            return piece is not None and getattr(piece, cast(str, self.attribute)) == self.value
        if self.op == "exists":
            return any(getattr(piece, cast(str, self.attribute)) == self.value for piece in scene.pieces)
        if self.op == "at_least_two":
            return (
                sum(getattr(piece, cast(str, self.attribute)) == self.value for piece in scene.pieces)
                >= 2
            )
        if self.op == "occupied_count_is":
            return scene.occupied_count == self.value
        if self.op == "same":
            first = scene.piece_at(cast(Position, self.position_1))
            second = scene.piece_at(cast(Position, self.position_2))
            return (
                first is not None
                and second is not None
                and getattr(first, cast(str, self.attribute)) == getattr(second, cast(str, self.attribute))
            )
        return scene.placard == "sun"

    def as_obj(self) -> dict[str, Any]:
        if self.op == "slot_empty":
            return {"op": self.op, "position": self.position}
        if self.op == "slot_attr":
            return {
                "op": self.op,
                "position": self.position,
                "attribute": self.attribute,
                "value": self.value,
            }
        if self.op in {"exists", "at_least_two"}:
            return {"op": self.op, "attribute": self.attribute, "value": self.value}
        if self.op == "occupied_count_is":
            return {"op": self.op, "value": self.value}
        if self.op == "same":
            return {
                "op": self.op,
                "position_1": self.position_1,
                "position_2": self.position_2,
                "attribute": self.attribute,
            }
        return {"op": self.op, "value": self.value}

    @property
    def canonical_json(self) -> str:
        return dump_json(self.as_obj())


@dataclass(frozen=True, slots=True)
class Literal:
    """An atom or its explicit negation."""

    atom: Atom
    negated: bool = False

    def __post_init__(self) -> None:
        if type(self.atom) is not Atom:
            raise RuleValidationError("a literal must wrap an Atom")
        if type(self.negated) is not bool:
            raise RuleValidationError("literal negated flag must be Boolean")

    def evaluate(self, scene: Scene) -> bool:
        value = self.atom.evaluate(scene)
        return not value if self.negated else value

    def as_obj(self) -> dict[str, Any]:
        return {"op": "not", "arg": self.atom.as_obj()} if self.negated else self.atom.as_obj()

    @property
    def canonical_json(self) -> str:
        return dump_json(self.as_obj())


@dataclass(frozen=True, slots=True)
class BinaryRule:
    """A canonical commutative binary composition of distinct literals."""

    op: BinaryOp
    args: tuple[Literal, Literal]

    def __post_init__(self) -> None:
        if self.op not in BINARY_OPS:
            raise RuleValidationError(f"unknown binary operation: {self.op!r}")
        args = tuple(self.args)
        if len(args) != 2 or any(type(arg) is not Literal for arg in args):
            raise RuleValidationError("binary rules require exactly two Literal arguments")
        if args[0] == args[1]:
            raise RuleValidationError("binary-rule literals must be distinct")
        ordered = tuple(sorted(args, key=lambda literal: literal.canonical_json))
        object.__setattr__(self, "args", cast(tuple[Literal, Literal], ordered))

    def evaluate(self, scene: Scene) -> bool:
        first, second = (literal.evaluate(scene) for literal in self.args)
        if self.op == "all":
            return first and second
        if self.op == "any":
            return first or second
        return first != second

    def as_obj(self) -> dict[str, Any]:
        return {"op": self.op, "args": [literal.as_obj() for literal in self.args]}

    @property
    def canonical_json(self) -> str:
        return dump_json(self.as_obj())


Rule: TypeAlias = Literal | BinaryRule


def evaluate_rule(rule: Rule, scene: Scene) -> bool:
    if type(rule) not in {Literal, BinaryRule}:
        raise RuleValidationError("rule evaluation requires a Literal or BinaryRule")
    return rule.evaluate(scene)


def iter_atoms() -> Iterator[Atom]:
    """Yield all 56 atoms in the released grammar's stable order."""

    for position in POSITIONS:
        yield Atom("slot_empty", position=position)
    for position in POSITIONS:
        for attribute in ATTRIBUTES:
            for value in ATTRIBUTE_VALUES[attribute]:
                yield Atom("slot_attr", position=position, attribute=attribute, value=value)
    for op in ("exists", "at_least_two"):
        for attribute in ATTRIBUTES:
            for value in ATTRIBUTE_VALUES[attribute]:
                yield Atom(cast(AtomOp, op), attribute=attribute, value=value)
    for count in (1, 2, 3):
        yield Atom("occupied_count_is", value=count)
    for position_1, position_2 in itertools.combinations(POSITIONS, 2):
        for attribute in ATTRIBUTES:
            yield Atom(
                "same",
                position_1=position_1,
                position_2=position_2,
                attribute=attribute,
            )
    yield Atom("placard_is", value="sun")


ATOMS: tuple[Atom, ...] = tuple(iter_atoms())
if len(ATOMS) != ATOM_COUNT or len(set(ATOMS)) != ATOM_COUNT:  # pragma: no cover - import invariant
    raise RuntimeError(f"released atom grammar must contain exactly {ATOM_COUNT} unique atoms")

LITERALS: tuple[Literal, ...] = tuple(
    literal
    for atom in ATOMS
    for literal in (Literal(atom), Literal(atom, negated=True))
)


def iter_syntactic_rules() -> Iterator[Rule]:
    """Yield every one- and two-literal canonical syntax exactly once."""

    yield from LITERALS
    for op in BINARY_OPS:
        for first, second in itertools.combinations(LITERALS, 2):
            yield BinaryRule(op, (first, second))


def rule_sort_key(rule: Rule) -> tuple[int, str]:
    return (1 if type(rule) is Literal else 2, rule.canonical_json)


def _exact_keys(value: Mapping[str, Any], expected: tuple[str, ...], *, path: str) -> None:
    if set(value) != set(expected) or len(value) != len(expected):
        missing = sorted(set(expected) - set(value))
        extra = sorted(set(value) - set(expected))
        raise RuleValidationError(
            f"{path} must have exactly keys {expected!r}; missing={missing!r}, extra={extra!r}"
        )


def atom_from_obj(value: Any, *, path: str = "rule") -> Atom:
    if type(value) is not dict:
        raise RuleValidationError(f"{path} must be a JSON object")
    op = value.get("op")
    if type(op) is not str or op not in ATOM_OPS:
        raise RuleValidationError(f"{path}.op is not a supported atom operation: {op!r}")
    keys: tuple[str, ...]
    if op == "slot_empty":
        keys = ("op", "position")
    elif op == "slot_attr":
        keys = ("op", "position", "attribute", "value")
    elif op in {"exists", "at_least_two"}:
        keys = ("op", "attribute", "value")
    elif op == "occupied_count_is":
        keys = ("op", "value")
    elif op == "same":
        keys = ("op", "position_1", "position_2", "attribute")
    else:
        keys = ("op", "value")
    _exact_keys(value, keys, path=path)
    return Atom(
        cast(AtomOp, op),
        position=cast(Position | None, value.get("position")),
        attribute=cast(Attribute | None, value.get("attribute")),
        value=cast(str | int | None, value.get("value")),
        position_1=cast(Position | None, value.get("position_1")),
        position_2=cast(Position | None, value.get("position_2")),
    )


def literal_from_obj(value: Any, *, path: str = "rule") -> Literal:
    if type(value) is not dict:
        raise RuleValidationError(f"{path} must be a JSON object")
    if value.get("op") == "not":
        _exact_keys(value, ("op", "arg"), path=path)
        return Literal(atom_from_obj(value["arg"], path=f"{path}.arg"), negated=True)
    return Literal(atom_from_obj(value, path=path))


def rule_from_obj(value: Any, *, require_canonical: bool = True, path: str = "rule") -> Rule:
    if type(value) is not dict:
        raise RuleValidationError(f"{path} must be a JSON object")
    op = value.get("op")
    if op in BINARY_OPS:
        _exact_keys(value, ("op", "args"), path=path)
        args = value["args"]
        if type(args) is not list or len(args) != 2:
            raise RuleValidationError(f"{path}.args must be a two-element JSON array")
        result: Rule = BinaryRule(
            cast(BinaryOp, op),
            (
                literal_from_obj(args[0], path=f"{path}.args[0]"),
                literal_from_obj(args[1], path=f"{path}.args[1]"),
            ),
        )
    else:
        result = literal_from_obj(value, path=path)
    if require_canonical and result.as_obj() != value:
        raise RuleValidationError(f"{path} is semantically valid but not a canonical AST")
    return result


def serialize_rule(rule: Rule) -> str:
    if type(rule) not in {Literal, BinaryRule}:
        raise RuleValidationError("serialize_rule requires a Literal or BinaryRule")
    return rule.canonical_json


def parse_rule(text: str, *, require_canonical: bool = True) -> Rule:
    try:
        value = load_json(text)
    except CanonicalJSONError as exc:
        raise RuleValidationError(str(exc)) from exc
    result = rule_from_obj(value, require_canonical=require_canonical)
    if require_canonical and serialize_rule(result) != text:
        raise RuleValidationError("rule JSON is valid but not in canonical serialized form")
    return result
