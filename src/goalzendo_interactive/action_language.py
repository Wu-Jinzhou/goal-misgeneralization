"""Finite-state canonical action construction for model-side G03 decoding.

The model is never offered a short list of candidate koans.  Inquiry decoding
factorizes the complete 13,716-scene combinatorial language into grammar
fields.  Terminal decoding exposes the complete public syntactic rule grammar
and one binary classification field per withheld koan.  Concatenating the
forced prefixes and selected option texts produces exactly the same canonical
JSON accepted by :mod:`goalzendo_interactive.actions`.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Literal, TypeAlias, cast

from ._json import dump_json, json_digest
from .actions import (
    Action,
    AnswerAction,
    Classification,
    ReadyAction,
    TestAction,
    parse_action,
    serialize_action,
)
from .rules import iter_syntactic_rules
from .schema import COLORS, PLACARDS, POSITIONS, SHAPES, SIZES

ACTION_LANGUAGE_SCHEMA_VERSION = 1

ActionLanguageMode = Literal["inquiry", "answer"]


class ActionLanguageError(ValueError):
    """Raised when a choice is outside the current finite-state grammar."""


@dataclass(frozen=True, slots=True)
class GrammarOption:
    key: str
    text: str

    def __post_init__(self) -> None:
        if type(self.key) is not str or not self.key:
            raise ActionLanguageError("grammar option key cannot be empty")
        if type(self.text) is not str or not self.text:
            raise ActionLanguageError("grammar option text cannot be empty")
        try:
            self.text.encode("ascii")
        except UnicodeEncodeError as exc:
            raise ActionLanguageError("grammar option text must be ASCII") from exc

    def as_obj(self) -> dict[str, str]:
        return {"key": self.key, "text": self.text}


@dataclass(frozen=True, slots=True)
class GrammarSegment:
    field_id: str
    prefix: str
    options: tuple[GrammarOption, ...]

    def __post_init__(self) -> None:
        if type(self.field_id) is not str or not self.field_id:
            raise ActionLanguageError("grammar field id cannot be empty")
        if type(self.prefix) is not str:
            raise ActionLanguageError("grammar segment prefix must be a string")
        options = tuple(self.options)
        object.__setattr__(self, "options", options)
        if not options or any(type(option) is not GrammarOption for option in options):
            raise ActionLanguageError("grammar segment requires at least one option")
        keys = tuple(option.key for option in options)
        texts = tuple(option.text for option in options)
        if len(set(keys)) != len(keys) or len(set(texts)) != len(texts):
            raise ActionLanguageError("grammar option keys and texts must be unique")

    def option(self, key: str) -> GrammarOption:
        for option in self.options:
            if option.key == key:
                return option
        raise ActionLanguageError(f"invalid choice {key!r} for field {self.field_id!r}")

    def as_obj(self) -> dict[str, object]:
        return {
            "field_id": self.field_id,
            "prefix": self.prefix,
            "options": [option.as_obj() for option in self.options],
        }


_MOVE_OPTIONS = (
    GrammarOption("test", dump_json("test")),
    GrammarOption("ready", dump_json("ready")),
)
_OCCUPANCY_OPTIONS = (
    GrammarOption("empty", "null"),
    GrammarOption("piece", "{"),
)
_PIECE_ONLY_OPTION = (GrammarOption("piece", "{"),)


def _value_options(values: tuple[str, ...]) -> tuple[GrammarOption, ...]:
    return tuple(GrammarOption(value, dump_json(value)) for value in values)


_SIZE_OPTIONS = _value_options(SIZES)
_COLOR_OPTIONS = _value_options(COLORS)
_SHAPE_OPTIONS = _value_options(SHAPES)
_PLACARD_OPTIONS = _value_options(PLACARDS)
_CLASSIFICATION_OPTIONS = _value_options(("fits", "does_not_fit"))


def _selection_map(selections: tuple[tuple[str, str], ...]) -> dict[str, str]:
    result = dict(selections)
    if len(result) != len(selections):
        raise ActionLanguageError("action-language selections contain a duplicate field")
    return result


@dataclass(frozen=True, slots=True)
class InquiryActionState:
    selections: tuple[tuple[str, str], ...] = ()
    emitted: str = ""

    def __post_init__(self) -> None:
        if type(self.emitted) is not str:
            raise ActionLanguageError("emitted action prefix must be a string")
        _selection_map(tuple(self.selections))

    @property
    def mode(self) -> ActionLanguageMode:
        return "inquiry"

    def _pending_field(self) -> str | None:
        selected = _selection_map(self.selections)
        move = selected.get("move")
        if move is None:
            return "move"
        if move == "ready":
            return None
        for position in POSITIONS:
            occupied = selected.get(f"{position}.occupied")
            if occupied is None:
                return f"{position}.occupied"
            if occupied == "piece":
                for attribute in ("size", "color", "shape"):
                    field = f"{position}.{attribute}"
                    if field not in selected:
                        return field
        return "placard" if "placard" not in selected else None

    @property
    def complete(self) -> bool:
        return self._pending_field() is None

    def _slot_prefix(self, position: str, selected: dict[str, str]) -> str:
        rank = POSITIONS.index(cast(AnyPosition, position))
        if rank == 0:
            return ',"koan":{"left":'
        previous = POSITIONS[rank - 1]
        close_piece = "}" if selected[f"{previous}.occupied"] == "piece" else ""
        return f'{close_piece},"{position}":'

    def next_segment(self) -> GrammarSegment:
        field = self._pending_field()
        if field is None:
            raise ActionLanguageError("inquiry action is already complete")
        selected = _selection_map(self.selections)
        if field == "move":
            return GrammarSegment("move", '{"move":', _MOVE_OPTIONS)
        if field == "placard":
            close_piece = "}" if selected["right.occupied"] == "piece" else ""
            return GrammarSegment("placard", f'{close_piece},"placard":', _PLACARD_OPTIONS)
        position, attribute = field.split(".", 1)
        if attribute == "occupied":
            prefix = self._slot_prefix(position, selected)
            first_two_empty = all(
                selected.get(f"{prior}.occupied") == "empty" for prior in POSITIONS[:2]
            )
            options = _PIECE_ONLY_OPTION if position == "right" and first_two_empty else _OCCUPANCY_OPTIONS
            return GrammarSegment(field, prefix, options)
        if attribute == "size":
            return GrammarSegment(field, '"size":', _SIZE_OPTIONS)
        if attribute == "color":
            return GrammarSegment(field, ',"color":', _COLOR_OPTIONS)
        if attribute == "shape":
            return GrammarSegment(field, ',"shape":', _SHAPE_OPTIONS)
        raise AssertionError(f"unknown inquiry field {field!r}")

    def choose(self, key: str) -> InquiryActionState:
        segment = self.next_segment()
        option = segment.option(key)
        return InquiryActionState(
            (*self.selections, (segment.field_id, option.key)),
            self.emitted + segment.prefix + option.text,
        )

    @property
    def completion_suffix(self) -> str:
        if not self.complete:
            raise ActionLanguageError("cannot finish an incomplete inquiry action")
        selected = _selection_map(self.selections)
        return "}" if selected["move"] == "ready" else "}}"

    @property
    def text(self) -> str:
        return self.emitted + self.completion_suffix

    @property
    def action(self) -> ReadyAction | TestAction:
        expected_move = cast(Literal["test", "ready"], _selection_map(self.selections)["move"])
        parsed = parse_action(self.text, expected_move=expected_move)
        if type(parsed) not in {ReadyAction, TestAction}:
            raise AssertionError("inquiry grammar constructed a non-inquiry action")
        return cast(ReadyAction | TestAction, parsed)


# A local alias avoids importing a private schema type solely for tuple.index typing.
AnyPosition: TypeAlias = Literal["left", "center", "right"]


@lru_cache(maxsize=1)
def _rule_options() -> tuple[GrammarOption, ...]:
    return tuple(
        GrammarOption(rule.canonical_json, rule.canonical_json)
        for rule in iter_syntactic_rules()
    )


@dataclass(frozen=True, slots=True)
class AnswerActionState:
    terminal_count: int
    selections: tuple[tuple[str, str], ...] = ()
    emitted: str = ""

    def __post_init__(self) -> None:
        if (
            isinstance(self.terminal_count, bool)
            or not isinstance(self.terminal_count, int)
            or self.terminal_count < 1
        ):
            raise ActionLanguageError("terminal_count must be a positive integer")
        if type(self.emitted) is not str:
            raise ActionLanguageError("emitted action prefix must be a string")
        _selection_map(tuple(self.selections))

    @property
    def mode(self) -> ActionLanguageMode:
        return "answer"

    @property
    def classification_count(self) -> int:
        return sum(field.startswith("classification.") for field, _ in self.selections)

    @property
    def complete(self) -> bool:
        selected = _selection_map(self.selections)
        return "rule" in selected and self.classification_count == self.terminal_count

    def next_segment(self) -> GrammarSegment:
        if self.complete:
            raise ActionLanguageError("answer action is already complete")
        selected = _selection_map(self.selections)
        if "rule" not in selected:
            return GrammarSegment("rule", '{"move":"answer","rule":', _rule_options())
        index = self.classification_count
        prefix = ',"classifications":[' if index == 0 else ","
        return GrammarSegment(
            f"classification.{index}",
            prefix,
            _CLASSIFICATION_OPTIONS,
        )

    def choose(self, key: str) -> AnswerActionState:
        segment = self.next_segment()
        option = segment.option(key)
        return AnswerActionState(
            self.terminal_count,
            (*self.selections, (segment.field_id, option.key)),
            self.emitted + segment.prefix + option.text,
        )

    @property
    def completion_suffix(self) -> str:
        if not self.complete:
            raise ActionLanguageError("cannot finish an incomplete answer action")
        return "]}"

    @property
    def text(self) -> str:
        return self.emitted + self.completion_suffix

    @property
    def action(self) -> AnswerAction:
        parsed = parse_action(self.text, terminal_count=self.terminal_count, expected_move="answer")
        if type(parsed) is not AnswerAction:
            raise AssertionError("answer grammar constructed a non-answer action")
        return parsed


def build_inquiry_action(action: ReadyAction | TestAction) -> InquiryActionState:
    """Traverse the exact grammar path for a canonical inquiry action."""

    if type(action) is ReadyAction:
        result = InquiryActionState().choose("ready")
    elif type(action) is TestAction:
        result = InquiryActionState().choose("test")
        for position in POSITIONS:
            piece = action.koan.piece_at(position)
            result = result.choose("empty" if piece is None else "piece")
            if piece is not None:
                result = result.choose(piece.size)
                result = result.choose(piece.color)
                result = result.choose(piece.shape)
        result = result.choose(action.koan.placard)
    else:
        raise TypeError("build_inquiry_action requires ReadyAction or TestAction")
    if result.text != serialize_action(action) or result.action != action:
        raise ActionLanguageError("inquiry grammar path did not reproduce the action")
    return result


def build_answer_action(action: AnswerAction) -> AnswerActionState:
    """Traverse the exact grammar path for a canonical terminal answer."""

    if type(action) is not AnswerAction:
        raise TypeError("build_answer_action requires AnswerAction")
    result = AnswerActionState(len(action.classifications)).choose(action.rule.canonical_json)
    for classification in action.classifications:
        result = result.choose(classification)
    if result.text != serialize_action(action) or result.action != action:
        raise ActionLanguageError("answer grammar path did not reproduce the action")
    return result


def action_language_digest() -> str:
    """Bind every public grammar option and the fixed JSON construction order."""

    rules = _rule_options()
    return json_digest(
        {
            "schema_version": ACTION_LANGUAGE_SCHEMA_VERSION,
            "inquiry_field_order": [
                "move",
                *(
                    f"{position}.{field}"
                    for position in POSITIONS
                    for field in ("occupied", "size", "color", "shape")
                ),
                "placard",
            ],
            "move_options": [option.as_obj() for option in _MOVE_OPTIONS],
            "piece_values": {
                "size": [option.as_obj() for option in _SIZE_OPTIONS],
                "color": [option.as_obj() for option in _COLOR_OPTIONS],
                "shape": [option.as_obj() for option in _SHAPE_OPTIONS],
                "placard": [option.as_obj() for option in _PLACARD_OPTIONS],
            },
            "syntactic_rule_count": len(rules),
            "syntactic_rule_language_digest": json_digest(
                [option.text for option in rules],
                domain="goalzendo-interactive-syntactic-rule-language-v1",
            ),
            "classification_options": [
                option.as_obj() for option in _CLASSIFICATION_OPTIONS
            ],
        },
        domain="goalzendo-interactive-action-language-v1",
    )


def action_language_manifest() -> dict[str, object]:
    return {
        "schema_version": ACTION_LANGUAGE_SCHEMA_VERSION,
        "digest": action_language_digest(),
        "inquiry_action_count": 13_717,
        "test_action_count": 13_716,
        "ready_action_count": 1,
        "syntactic_rule_count": len(_rule_options()),
        "classification_alphabet": [
            cast(Classification, option.key) for option in _CLASSIFICATION_OPTIONS
        ],
        "decoder_kind": "field-factorized canonical JSON finite-state grammar",
        "candidate_koan_list_presented_to_model": False,
    }


def action_from_state(state: InquiryActionState | AnswerActionState) -> Action:
    if type(state) is InquiryActionState:
        return state.action
    if type(state) is AnswerActionState:
        return state.action
    raise TypeError("action_from_state requires an action-language state")
