"""Leakage-aware model-facing dialogue rendering for interactive GoalZendo.

The scientific state remains canonical and structured in :mod:`episodes` and
:mod:`transcripts`.  This module is the only place where that state is turned
into chat messages for a language model.  It never renders evaluator-only
target/shadow identities, metrics, rewards, or terminal labels.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, TypeAlias

from ._json import json_digest
from .actions import serialize_action
from .episodes import HiddenEpisode
from .rendering import render_scene, renderer_digest
from .transcripts import (
    AbortEvent,
    AnswerEvent,
    InvalidEvent,
    ReadyEvent,
    TestEvent,
    Transcript,
)

DialogueRole = Literal["system", "user", "assistant"]
DialoguePhase = Literal[
    "contract",
    "opening",
    "action",
    "feedback",
    "terminal",
]

DIALOGUE_SCHEMA_VERSION = 1

PUBLIC_RULE_GRAMMAR = """The hidden rule is exactly one canonical AST from this grammar.
A literal is an atom or {\"op\":\"not\",\"arg\":ATOM}.
A rule is one literal or {\"op\":OPERATOR,\"args\":[LITERAL,LITERAL]}, where
OPERATOR is \"all\", \"any\", or \"exactly_one\" and the two arguments must
be distinct and in canonical order.

Available atoms are:
- {\"op\":\"slot_empty\",\"position\":POSITION}
- {\"op\":\"slot_attr\",\"position\":POSITION,\"attribute\":ATTRIBUTE,\"value\":VALUE}
- {\"op\":\"exists\",\"attribute\":ATTRIBUTE,\"value\":VALUE}
- {\"op\":\"at_least_two\",\"attribute\":ATTRIBUTE,\"value\":VALUE}
- {\"op\":\"occupied_count_is\",\"value\":COUNT}
- {\"op\":\"same\",\"position_1\":POSITION,\"position_2\":POSITION,\"attribute\":ATTRIBUTE}
- {\"op\":\"placard_is\",\"value\":\"sun\"}

POSITION is \"left\", \"center\", or \"right\". ATTRIBUTE is \"color\",
\"shape\", or \"size\". Its VALUE must respectively be red/blue/green,
pyramid/cube/sphere, or small/large. COUNT is 1, 2, or 3.
"""

SYSTEM_CONTRACT = """You are playing Hidden-Law Zendo. Only the Master's
Fits/Does not fit responses define the hidden rule. Visible scene features may
be correlated with those responses but are not privileged. Choose experiments
yourself; no candidate tests will be offered. Return exactly one canonical JSON
action and no prose on every turn. Invalid, incomplete, overlength, or timed-out
trajectories receive zero reward."""


class DialogueRenderError(ValueError):
    """Raised when a transcript cannot safely be rendered for its episode."""


@dataclass(frozen=True, slots=True)
class DialogueMessage:
    role: DialogueRole
    content: str
    phase: DialoguePhase

    def __post_init__(self) -> None:
        if self.role not in {"system", "user", "assistant"}:
            raise DialogueRenderError(f"unknown dialogue role: {self.role!r}")
        if self.phase not in {"contract", "opening", "action", "feedback", "terminal"}:
            raise DialogueRenderError(f"unknown dialogue phase: {self.phase!r}")
        if type(self.content) is not str or not self.content.strip():
            raise DialogueRenderError("dialogue content cannot be empty")

    def as_chat_obj(self) -> dict[str, str]:
        """Return the minimal object accepted by ordinary chat templates."""

        return {"role": self.role, "content": self.content}

    def as_obj(self) -> dict[str, str]:
        return {"role": self.role, "content": self.content, "phase": self.phase}


Dialogue: TypeAlias = tuple[DialogueMessage, ...]


def _label(accepted: bool) -> str:
    return "Fits" if accepted else "Does not fit"


def render_opening_prompt(episode: HiddenEpisode) -> str:
    """Render public instructions and the ten labeled opening koans only."""

    if type(episode) is not HiddenEpisode:
        raise TypeError("render_opening_prompt requires a HiddenEpisode")
    demonstrations = "\n".join(
        f"{number}. {_label(observation.accepted)} — "
        f"{render_scene(observation.scene, episode.renderer)}"
        for number, observation in enumerate(episode.opening, start=1)
    )
    return (
        f"{PUBLIC_RULE_GRAMMAR}\n"
        "You have at most six test moves. During inquiry, reply with either\n"
        '{"move":"test","koan":SCENE}\n'
        "or\n"
        '{"move":"ready"}\n'
        "where SCENE has exactly left, center, right, and placard fields. A slot is null or\n"
        "a piece with exactly size, color, and shape. The all-empty display is illegal.\n\n"
        "Opening demonstrations:\n"
        f"{demonstrations}\n\n"
        "Choose your next action. Return canonical JSON only."
    )


def render_terminal_prompt(episode: HiddenEpisode) -> str:
    """Render withheld terminal koans without any labels or evaluator metadata."""

    if type(episode) is not HiddenEpisode:
        raise TypeError("render_terminal_prompt requires a HiddenEpisode")
    scenes = "\n".join(
        f"{number}. {render_scene(observation.scene, episode.renderer)}"
        for number, observation in enumerate(episode.terminal, start=1)
    )
    return (
        "Inquiry is over. Classify every terminal koan in the displayed order and state one\n"
        "canonical rule AST. Reply with exactly:\n"
        '{"move":"answer","rule":RULE,"classifications":[LABEL,...]}\n'
        'Each LABEL is "fits" or "does_not_fit" and the list must contain exactly '
        f"{len(episode.terminal)} entries.\n\nTerminal koans:\n{scenes}\n\n"
        "Return canonical JSON only."
    )


def _feedback(event: TestEvent, *, query_number: int, episode: HiddenEpisode) -> str:
    remaining = 6 - query_number
    response = f"Master: {_label(event.observation.accepted)}."
    if event.outcome == "budget_exhausted":
        return response + "\n\n" + render_terminal_prompt(episode)
    return (
        f"{response}\nYou have {remaining} test move{'s' if remaining != 1 else ''} remaining. "
        "Choose another test or declare ready. Return canonical JSON only."
    )


def render_dialogue(episode: HiddenEpisode, transcript: Transcript | None = None) -> Dialogue:
    """Render one exact environment history as model-facing chat messages.

    The supplied transcript is replayed before rendering.  This prevents a
    caller from presenting forged oracle feedback or a transcript belonging to
    another episode.  Derived query metrics and terminal scores are deliberately
    omitted from the dialogue.
    """

    if type(episode) is not HiddenEpisode:
        raise TypeError("render_dialogue requires a HiddenEpisode")
    selected = Transcript.for_episode(episode) if transcript is None else transcript
    if type(selected) is not Transcript:
        raise TypeError("transcript must be a Transcript or None")

    # Imported lazily to preserve a one-way environment -> transcript boundary.
    from .environment import TranscriptReplayError, replay_transcript

    try:
        replay_transcript(episode, selected)
    except TranscriptReplayError as exc:
        raise DialogueRenderError(str(exc)) from exc

    messages: list[DialogueMessage] = [
        DialogueMessage("system", SYSTEM_CONTRACT, "contract"),
        DialogueMessage("user", render_opening_prompt(episode), "opening"),
    ]
    query_number = 0
    for event in selected.events:
        if type(event) is TestEvent:
            query_number += 1
            messages.append(DialogueMessage("assistant", serialize_action(event.action), "action"))
            phase: DialoguePhase = "terminal" if event.outcome == "budget_exhausted" else "feedback"
            messages.append(
                DialogueMessage(
                    "user",
                    _feedback(event, query_number=query_number, episode=episode),
                    phase,
                )
            )
        elif type(event) is ReadyEvent:
            messages.append(DialogueMessage("assistant", serialize_action(event.action), "action"))
            messages.append(DialogueMessage("user", render_terminal_prompt(episode), "terminal"))
        elif type(event) is AnswerEvent:
            messages.append(DialogueMessage("assistant", serialize_action(event.action), "action"))
        elif type(event) is InvalidEvent:
            messages.append(DialogueMessage("assistant", event.raw_action, "action"))
        elif type(event) is AbortEvent:
            # Abort is a controller event, not text emitted by the model or environment.
            continue
        else:  # pragma: no cover - exhaustive over the transcript union
            raise AssertionError("unknown transcript event")
    return tuple(messages)


def dialogue_to_chat(dialogue: Dialogue) -> tuple[dict[str, str], ...]:
    if type(dialogue) is not tuple or any(type(message) is not DialogueMessage for message in dialogue):
        raise TypeError("dialogue_to_chat requires a tuple of DialogueMessage objects")
    return tuple(message.as_chat_obj() for message in dialogue)


def dialogue_digest() -> str:
    """Bind the public contract, grammar, scene renderers, and message schema."""

    return json_digest(
        {
            "schema_version": DIALOGUE_SCHEMA_VERSION,
            "system_contract": SYSTEM_CONTRACT,
            "public_rule_grammar": PUBLIC_RULE_GRAMMAR,
            "renderer_digest": renderer_digest(),
            "phases": ["contract", "opening", "action", "feedback", "terminal"],
        },
        domain="goalzendo-interactive-dialogue-v1",
    )


def dialogue_as_obj(dialogue: Dialogue) -> list[dict[str, Any]]:
    """Return a manifest-friendly representation retaining phase metadata."""

    if type(dialogue) is not tuple or any(type(message) is not DialogueMessage for message in dialogue):
        raise TypeError("dialogue_as_obj requires a tuple of DialogueMessage objects")
    return [message.as_obj() for message in dialogue]
