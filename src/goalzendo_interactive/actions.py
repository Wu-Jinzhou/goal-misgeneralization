"""Strict canonical JSON actions for test, ready, and terminal answer moves."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, ClassVar, Literal, TypeAlias, cast

from ._json import CanonicalJSONError, dump_json, load_json
from .rules import BinaryRule, Rule, RuleValidationError, rule_from_obj
from .rules import Literal as RuleLiteral
from .schema import Scene, SceneValidationError, scene_from_obj

Classification = Literal["fits", "does_not_fit"]
Move = Literal["test", "ready", "answer"]
CLASSIFICATIONS: tuple[Classification, ...] = ("fits", "does_not_fit")


class InvalidActionError(ValueError):
    """A stable, inspectable invalid-action outcome; never an auto-repair signal."""

    def __init__(self, code: str, detail: str) -> None:
        self.code = code
        self.detail = detail
        super().__init__(f"{code}: {detail}")


@dataclass(frozen=True, slots=True)
class TestAction:
    __test__: ClassVar[bool] = False

    koan: Scene

    def __post_init__(self) -> None:
        if type(self.koan) is not Scene:
            raise InvalidActionError("invalid_koan", "test action requires a canonical Scene")

    def as_obj(self) -> dict[str, Any]:
        return {"move": "test", "koan": self.koan.as_obj()}


@dataclass(frozen=True, slots=True)
class ReadyAction:
    def as_obj(self) -> dict[str, str]:
        return {"move": "ready"}


@dataclass(frozen=True, slots=True)
class AnswerAction:
    rule: Rule
    classifications: tuple[Classification, ...]

    def __post_init__(self) -> None:
        if type(self.rule) not in {RuleLiteral, BinaryRule}:
            raise InvalidActionError("invalid_rule", "answer action requires a canonical Rule")
        values = tuple(self.classifications)
        if not values:
            raise InvalidActionError("invalid_classifications", "classification list cannot be empty")
        if any(type(value) is not str or value not in CLASSIFICATIONS for value in values):
            raise InvalidActionError(
                "invalid_classifications",
                "classifications must contain only 'fits' or 'does_not_fit'",
            )
        object.__setattr__(self, "classifications", values)

    def as_obj(self) -> dict[str, Any]:
        return {
            "move": "answer",
            "rule": self.rule.as_obj(),
            "classifications": list(self.classifications),
        }


Action: TypeAlias = TestAction | ReadyAction | AnswerAction


def action_to_obj(action: Action) -> dict[str, Any]:
    if type(action) not in {TestAction, ReadyAction, AnswerAction}:
        raise InvalidActionError("invalid_action_type", "unsupported action object")
    return action.as_obj()


def serialize_action(action: Action) -> str:
    return dump_json(action_to_obj(action))


def _exact_keys(value: dict[str, Any], expected: tuple[str, ...]) -> None:
    if set(value) != set(expected) or len(value) != len(expected):
        missing = sorted(set(expected) - set(value))
        extra = sorted(set(value) - set(expected))
        raise InvalidActionError(
            "invalid_fields",
            f"action must have exactly keys {expected!r}; missing={missing!r}, extra={extra!r}",
        )


def action_from_obj(
    value: Any,
    *,
    terminal_count: int | None = None,
    expected_move: Move | None = None,
    require_canonical: bool = True,
) -> Action:
    if type(value) is not dict:
        raise InvalidActionError("invalid_top_level", "action must be a JSON object")
    move = value.get("move")
    if type(move) is not str or move not in {"test", "ready", "answer"}:
        raise InvalidActionError("invalid_move", f"unknown move: {move!r}")
    if expected_move is not None and move != expected_move:
        raise InvalidActionError("unexpected_move", f"expected {expected_move!r}, received {move!r}")

    try:
        if move == "test":
            _exact_keys(value, ("move", "koan"))
            result: Action = TestAction(scene_from_obj(value["koan"]))
        elif move == "ready":
            _exact_keys(value, ("move",))
            result = ReadyAction()
        else:
            _exact_keys(value, ("move", "rule", "classifications"))
            raw_classifications = value["classifications"]
            if type(raw_classifications) is not list:
                raise InvalidActionError(
                    "invalid_classifications", "classifications must be a JSON array"
                )
            if terminal_count is not None:
                if (
                    isinstance(terminal_count, bool)
                    or not isinstance(terminal_count, int)
                    or terminal_count < 1
                ):
                    raise ValueError("terminal_count must be a positive integer")
                if len(raw_classifications) != terminal_count:
                    raise InvalidActionError(
                        "wrong_classification_count",
                        f"expected {terminal_count}, received {len(raw_classifications)}",
                    )
            result = AnswerAction(
                rule_from_obj(value["rule"], require_canonical=require_canonical),
                cast(tuple[Classification, ...], tuple(raw_classifications)),
            )
    except InvalidActionError:
        raise
    except SceneValidationError as exc:
        raise InvalidActionError("invalid_koan", str(exc)) from exc
    except RuleValidationError as exc:
        raise InvalidActionError("invalid_rule", str(exc)) from exc

    if require_canonical and action_to_obj(result) != value:
        raise InvalidActionError("noncanonical_action", "action AST is not canonical")
    return result


def parse_action(
    text: str,
    *,
    terminal_count: int | None = None,
    expected_move: Move | None = None,
    require_canonical: bool = True,
) -> Action:
    try:
        value = load_json(text)
    except CanonicalJSONError as exc:
        raise InvalidActionError("invalid_json", str(exc)) from exc
    result = action_from_obj(
        value,
        terminal_count=terminal_count,
        expected_move=expected_move,
        require_canonical=require_canonical,
    )
    if require_canonical and serialize_action(result) != text:
        raise InvalidActionError(
            "noncanonical_json", "action JSON is valid but not in canonical serialized form"
        )
    return result
