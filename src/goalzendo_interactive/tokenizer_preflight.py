"""Fail-closed local tokenizer and chat-template preflight for G03.

This module does not load a model, download a tokenizer, or authorize a weight
update.  It tests a supplied :class:`ChatTokenizerProtocol` against the exact
G03 dialogue, action-language, and trajectory-masking contracts.  Expensive
tokenizer probes use a deterministic, coverage-preserving sample, while the
finite semantic languages themselves are enumerated exhaustively.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Any, cast

from ._json import CanonicalJSONError, dump_json, json_digest, load_json
from .action_language import (
    action_language_digest,
    build_answer_action,
    build_inquiry_action,
)
from .actions import (
    Action,
    AnswerAction,
    Classification,
    ReadyAction,
    TestAction,
    parse_action,
    serialize_action,
)
from .dialogue import Dialogue, DialogueMessage, DialoguePhase, dialogue_as_obj
from .rules import SYNTACTIC_RULE_COUNT, BinaryRule, Rule, iter_syntactic_rules, parse_rule
from .schema import SCENE_COUNT, scene_at
from .trajectory_encoding import (
    IGNORE_INDEX,
    ChatTokenizerProtocol,
    EncodedTrajectory,
    TrajectoryEncodingError,
    encode_sft_dialogue,
    render_generation_prefix,
)

TOKENIZER_PREFLIGHT_SCHEMA_VERSION = 1
TOKENIZER_PREFLIGHT_REPORT_KIND = "local_tokenizer_format_preflight"
TOKENIZER_PREFLIGHT_CHECK_IDS = (
    "role_template_boundaries",
    "assistant_only_masking",
    "canonical_action_roundtrips",
    "action_prefix_stability",
    "no_truncation",
    "enumeration_and_sample_coverage",
)
INQUIRY_TOKEN_SAMPLE_TARGET = 64
ANSWER_TOKEN_SAMPLE_TARGET = 64

_CHECK_EVIDENCE_DOMAIN = "goalzendo-interactive-tokenizer-preflight-check-v1"
_REPORT_DOMAIN = "goalzendo-interactive-tokenizer-preflight-report-v1"
_ACTION_INVENTORY_DOMAIN = "goalzendo-interactive-tokenizer-action-inventory-v1"


class TokenizerPreflightError(RuntimeError):
    """Raised when any tokenizer-format invariant cannot be established."""


class TokenizerPreflightValidationError(ValueError):
    """Raised when an audit object is internally malformed."""


def _canonical_evidence(value: dict[str, Any]) -> str:
    try:
        encoded = dump_json(value)
        decoded = load_json(encoded)
    except CanonicalJSONError as exc:
        raise TokenizerPreflightValidationError(str(exc)) from exc
    if decoded != value:
        raise TokenizerPreflightValidationError("check evidence is not stable canonical JSON")
    return encoded


@dataclass(frozen=True, slots=True)
class TokenizerPreflightCheck:
    """One passed invariant and the canonical evidence supporting it."""

    check_id: str
    exhaustive: bool
    item_count: int
    claim: str
    evidence_json: str

    def __post_init__(self) -> None:
        if type(self.check_id) is not str or not self.check_id or not self.check_id.isascii():
            raise TokenizerPreflightValidationError("check id must be nonempty ASCII")
        if type(self.exhaustive) is not bool:
            raise TokenizerPreflightValidationError("check exhaustive flag must be Boolean")
        if (
            isinstance(self.item_count, bool)
            or not isinstance(self.item_count, int)
            or self.item_count < 1
        ):
            raise TokenizerPreflightValidationError("check item count must be positive")
        if type(self.claim) is not str or not self.claim or not self.claim.isascii():
            raise TokenizerPreflightValidationError("check claim must be nonempty ASCII")
        if type(self.evidence_json) is not str:
            raise TokenizerPreflightValidationError("check evidence must be canonical JSON text")
        try:
            evidence = load_json(self.evidence_json)
        except CanonicalJSONError as exc:
            raise TokenizerPreflightValidationError(str(exc)) from exc
        if type(evidence) is not dict or dump_json(evidence) != self.evidence_json:
            raise TokenizerPreflightValidationError("check evidence JSON is not canonical")

    @classmethod
    def create(
        cls,
        check_id: str,
        *,
        exhaustive: bool,
        item_count: int,
        claim: str,
        evidence: dict[str, Any],
    ) -> TokenizerPreflightCheck:
        return cls(check_id, exhaustive, item_count, claim, _canonical_evidence(evidence))

    @property
    def evidence(self) -> dict[str, Any]:
        return cast(dict[str, Any], load_json(self.evidence_json))

    @property
    def evidence_digest(self) -> str:
        return json_digest(
            {"check_id": self.check_id, "evidence": self.evidence},
            domain=_CHECK_EVIDENCE_DOMAIN,
        )

    def as_obj(self) -> dict[str, Any]:
        return {
            "check_id": self.check_id,
            "status": "pass",
            "exhaustive": self.exhaustive,
            "item_count": self.item_count,
            "claim": self.claim,
            "evidence_digest": self.evidence_digest,
            "evidence": self.evidence,
        }


@dataclass(frozen=True, slots=True)
class TokenizerPreflightReport:
    """Canonical pass-only report for one tokenizer/template behavior."""

    tokenizer_identifier: str
    maximum_tokens: int
    dialogue_digest: str
    encoded_trajectory_digest: str
    checks: tuple[TokenizerPreflightCheck, ...]

    def __post_init__(self) -> None:
        if (
            type(self.tokenizer_identifier) is not str
            or not self.tokenizer_identifier
            or not self.tokenizer_identifier.isascii()
        ):
            raise TokenizerPreflightValidationError(
                "tokenizer identifier must be nonempty ASCII"
            )
        if (
            isinstance(self.maximum_tokens, bool)
            or not isinstance(self.maximum_tokens, int)
            or self.maximum_tokens < 1
        ):
            raise TokenizerPreflightValidationError("maximum_tokens must be positive")
        for name in ("dialogue_digest", "encoded_trajectory_digest"):
            value = getattr(self, name)
            if (
                type(value) is not str
                or len(value) != 64
                or any(character not in "0123456789abcdef" for character in value)
            ):
                raise TokenizerPreflightValidationError(f"{name} must be a SHA-256 digest")
        checks = tuple(self.checks)
        object.__setattr__(self, "checks", checks)
        if tuple(check.check_id for check in checks) != TOKENIZER_PREFLIGHT_CHECK_IDS:
            raise TokenizerPreflightValidationError(
                "preflight checks must appear once in registered order"
            )

    @property
    def passed(self) -> bool:
        return True

    @property
    def weight_updates_authorized(self) -> bool:
        return False

    @property
    def full_model_smoke_completed(self) -> bool:
        return False

    @property
    def digest(self) -> str:
        return json_digest(self.as_obj(), domain=_REPORT_DOMAIN)

    def check(self, check_id: str) -> TokenizerPreflightCheck:
        for check in self.checks:
            if check.check_id == check_id:
                return check
        raise KeyError(check_id)

    def as_obj(self) -> dict[str, Any]:
        return {
            "schema_version": TOKENIZER_PREFLIGHT_SCHEMA_VERSION,
            "report_kind": TOKENIZER_PREFLIGHT_REPORT_KIND,
            "tokenizer_identifier": self.tokenizer_identifier,
            "maximum_tokens": self.maximum_tokens,
            "dialogue_digest": self.dialogue_digest,
            "encoded_trajectory_digest": self.encoded_trajectory_digest,
            "checks": [check.as_obj() for check in self.checks],
            "authorization": {
                "weight_updates_authorized": False,
                "full_model_smoke_completed": False,
                "reason": "local tokenizer-format evidence only",
            },
        }


@dataclass(frozen=True, slots=True)
class _ActionInventory:
    inventory_digest: str
    test_action_digest: str
    rule_language_digest: str
    inquiry_option_pairs: tuple[tuple[str, str], ...]
    inquiry_sample_indices: tuple[int, ...]
    inquiry_sample_texts: tuple[str, ...]
    rule_sample_indices: tuple[int, ...]
    rule_sample_values: tuple[Rule, ...]


def _even_indices(size: int, count: int) -> tuple[int, ...]:
    if count == 1:
        return (0,)
    return tuple((rank * (size - 1)) // (count - 1) for rank in range(count))


def _filled_sample_indices(
    *,
    size: int,
    required: list[int],
    target: int,
) -> tuple[int, ...]:
    selected = list(dict.fromkeys(required))
    for index in _even_indices(size, min(size, target)):
        if index not in selected:
            selected.append(index)
        if len(selected) == target:
            break
    if len(selected) < target:
        for index in range(size):
            if index not in selected:
                selected.append(index)
            if len(selected) == target:
                break
    return tuple(sorted(selected))


@lru_cache(maxsize=1)
def _canonical_action_inventory() -> _ActionInventory:
    """Exhaustively establish the finite language before tokenizer sampling."""

    test_texts: list[str] = []
    all_option_pairs: set[tuple[str, str]] = set()
    covered_pairs: set[tuple[str, str]] = set()
    required_scene_indices: list[int] = []
    for index in range(SCENE_COUNT):
        action = TestAction(scene_at(index))
        state = build_inquiry_action(action)
        parsed = parse_action(state.text, expected_move="test")
        if parsed != action or serialize_action(parsed) != state.text:
            raise TokenizerPreflightError(
                f"canonical test-action round-trip failed at scene {index}"
            )
        path_pairs = set(state.selections)
        all_option_pairs.update(path_pairs)
        if path_pairs - covered_pairs:
            required_scene_indices.append(index)
            covered_pairs.update(path_pairs)
        test_texts.append(state.text)
    if len(set(test_texts)) != SCENE_COUNT:
        raise TokenizerPreflightError("canonical test-action enumeration is not unique")

    ready_state = build_inquiry_action(ReadyAction())
    ready_text = ready_state.text
    if parse_action(ready_text, expected_move="ready") != ReadyAction():
        raise TokenizerPreflightError("canonical ready action did not round-trip")
    all_option_pairs.update(ready_state.selections)

    inquiry_indices = _filled_sample_indices(
        size=SCENE_COUNT,
        required=required_scene_indices,
        target=INQUIRY_TOKEN_SAMPLE_TARGET,
    )
    sampled_pairs = {
        pair
        for index in inquiry_indices
        for pair in build_inquiry_action(TestAction(scene_at(index))).selections
    }
    sampled_pairs.update(ready_state.selections)
    if sampled_pairs != all_option_pairs:
        missing = sorted(all_option_pairs - sampled_pairs)
        raise TokenizerPreflightError(
            f"inquiry tokenizer sample omits grammar options: {missing!r}"
        )

    rules = tuple(iter_syntactic_rules())
    if len(rules) != SYNTACTIC_RULE_COUNT:
        raise TokenizerPreflightError("syntactic rule enumeration has the wrong size")
    rule_texts: list[str] = []
    required_rule_indices: list[int] = []
    covered_strata: set[str] = set()
    for index, rule in enumerate(rules):
        text = rule.canonical_json
        if parse_rule(text) != rule:
            raise TokenizerPreflightError(f"canonical rule round-trip failed at index {index}")
        rule_texts.append(text)
        stratum = rule.op if type(rule) is BinaryRule else "literal"
        if stratum not in covered_strata:
            covered_strata.add(stratum)
            required_rule_indices.append(index)
    if len(set(rule_texts)) != SYNTACTIC_RULE_COUNT:
        raise TokenizerPreflightError("syntactic rule enumeration is not unique")
    if covered_strata != {"literal", "all", "any", "exactly_one"}:
        raise TokenizerPreflightError("rule sample strata are incomplete")
    rule_indices = _filled_sample_indices(
        size=SYNTACTIC_RULE_COUNT,
        required=required_rule_indices,
        target=ANSWER_TOKEN_SAMPLE_TARGET,
    )

    test_digest = json_digest(
        test_texts,
        domain="goalzendo-interactive-canonical-test-actions-v1",
    )
    rule_digest = json_digest(
        rule_texts,
        domain="goalzendo-interactive-canonical-syntactic-rules-v1",
    )
    inventory_digest = json_digest(
        {
            "ready_action": ready_text,
            "test_action_count": len(test_texts),
            "test_action_digest": test_digest,
            "rule_count": len(rule_texts),
            "rule_language_digest": rule_digest,
            "inquiry_option_pairs": [list(pair) for pair in sorted(all_option_pairs)],
            "inquiry_sample_indices": list(inquiry_indices),
            "rule_sample_indices": list(rule_indices),
        },
        domain=_ACTION_INVENTORY_DOMAIN,
    )
    return _ActionInventory(
        inventory_digest=inventory_digest,
        test_action_digest=test_digest,
        rule_language_digest=rule_digest,
        inquiry_option_pairs=tuple(sorted(all_option_pairs)),
        inquiry_sample_indices=inquiry_indices,
        inquiry_sample_texts=(
            ready_text,
            *(test_texts[index] for index in inquiry_indices),
        ),
        rule_sample_indices=rule_indices,
        rule_sample_values=tuple(rules[index] for index in rule_indices),
    )


def _render_text(
    tokenizer: ChatTokenizerProtocol,
    messages: tuple[DialogueMessage, ...],
    *,
    add_generation_prompt: bool,
) -> str:
    value = tokenizer.apply_chat_template(
        [message.as_chat_obj() for message in messages],
        tokenize=False,
        add_generation_prompt=add_generation_prompt,
    )
    if type(value) is not str or not value:
        raise TokenizerPreflightError("chat template did not return nonempty text")
    return value


def _token_tuple(tokenizer: ChatTokenizerProtocol, text: str) -> tuple[int, ...]:
    values = tuple(tokenizer.encode(text, add_special_tokens=False))
    if any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in values):
        raise TokenizerPreflightError("tokenizer returned an invalid token id")
    return values


def _role_template_check(
    tokenizer: ChatTokenizerProtocol,
) -> tuple[TokenizerPreflightCheck, str]:
    system_marker = "G03_SYSTEM_ROLE_BOUNDARY_7C20"
    user_marker = "G03_USER_ROLE_BOUNDARY_19A4"
    next_user_marker = "G03_NEXT_USER_BOUNDARY_D155"
    action_text = serialize_action(ReadyAction())
    system = DialogueMessage("system", system_marker, "contract")
    user = DialogueMessage("user", user_marker, "opening")
    assistant = DialogueMessage("assistant", action_text, "action")
    next_user = DialogueMessage("user", next_user_marker, "feedback")
    before_generation = (system, user)
    through_assistant = (*before_generation, assistant)
    full_probe = (*through_assistant, next_user)

    closed_user_text = _render_text(
        tokenizer, before_generation, add_generation_prompt=False
    )
    generation_text = _render_text(
        tokenizer, before_generation, add_generation_prompt=True
    )
    assistant_text = _render_text(
        tokenizer, through_assistant, add_generation_prompt=False
    )
    full_text = _render_text(tokenizer, full_probe, add_generation_prompt=False)
    if not generation_text.startswith(closed_user_text) or generation_text == closed_user_text:
        raise TokenizerPreflightError(
            "generation prompt is not a strict extension of the completed user turn"
        )
    if not assistant_text.startswith(generation_text + action_text):
        raise TokenizerPreflightError(
            "assistant template does not preserve the exact generation/action boundary"
        )
    if not full_text.startswith(assistant_text):
        raise TokenizerPreflightError(
            "a following user turn changes the completed assistant prefix"
        )
    for marker in (system_marker, user_marker, action_text, next_user_marker):
        if full_text.count(marker) != 1:
            raise TokenizerPreflightError(
                "chat template omitted, rewrote, or duplicated role content"
            )
    if not (
        full_text.index(system_marker)
        < full_text.index(user_marker)
        < full_text.index(action_text)
        < full_text.index(next_user_marker)
    ):
        raise TokenizerPreflightError("chat template changed role-content order")

    closed_ids = _token_tuple(tokenizer, closed_user_text)
    generation_ids = _token_tuple(tokenizer, generation_text)
    assistant_ids = _token_tuple(tokenizer, assistant_text)
    full_ids = _token_tuple(tokenizer, full_text)
    if not (
        generation_ids[: len(closed_ids)] == closed_ids
        and assistant_ids[: len(generation_ids)] == generation_ids
        and full_ids[: len(assistant_ids)] == assistant_ids
    ):
        raise TokenizerPreflightError("role/template token boundaries are not prefix-stable")

    encoded = encode_sft_dialogue(tokenizer, through_assistant)
    if len(encoded.action_spans) != 1 or encoded.action_spans[0].start != len(generation_ids):
        raise TokenizerPreflightError("encoded assistant span disagrees with generation boundary")
    behavior_digest = json_digest(
        {
            "closed_user_text": closed_user_text,
            "generation_text": generation_text,
            "assistant_text": assistant_text,
            "full_text": full_text,
            "closed_user_ids": list(closed_ids),
            "generation_ids": list(generation_ids),
            "assistant_ids": list(assistant_ids),
            "full_ids": list(full_ids),
        },
        domain="goalzendo-interactive-tokenizer-role-template-probe-v1",
    )
    return (
        TokenizerPreflightCheck.create(
            "role_template_boundaries",
            exhaustive=True,
            item_count=4,
            claim=(
                "System, user, assistant, and following-user content retain exact text and "
                "token-prefix boundaries."
            ),
            evidence={
                "probe_message_count": 4,
                "generation_prompt_is_strict_extension": True,
                "role_content_preserved_once_and_in_order": True,
                "completed_turn_prefix_stable": True,
                "behavior_digest": behavior_digest,
            },
        ),
        behavior_digest,
    )


def _canonical_dialogue_actions(dialogue: Dialogue) -> tuple[tuple[Action, ...], int]:
    actions: list[Action] = []
    for index, message in enumerate(dialogue):
        if message.role != "assistant":
            continue
        try:
            action = parse_action(message.content)
        except Exception as exc:
            raise TokenizerPreflightError(
                f"assistant message {index} is not a canonical G03 action"
            ) from exc
        if serialize_action(action) != message.content:
            raise TokenizerPreflightError(
                f"assistant message {index} did not round-trip byte-for-byte"
            )
        if type(action) is AnswerAction:
            build_answer_action(action)
        else:
            build_inquiry_action(cast(ReadyAction | TestAction, action))
        actions.append(action)
    if len(actions) < 2 or type(actions[-1]) is not AnswerAction:
        raise TokenizerPreflightError(
            "preflight dialogue must contain multiple actions ending in a terminal answer"
        )
    if any(type(action) is AnswerAction for action in actions[:-1]):
        raise TokenizerPreflightError("terminal answer must be the final assistant action")
    terminal_count = len(cast(AnswerAction, actions[-1]).classifications)
    return tuple(actions), terminal_count


def _encoded_dialogue_checks(
    tokenizer: ChatTokenizerProtocol,
    dialogue: Dialogue,
    *,
    maximum_tokens: int,
) -> tuple[EncodedTrajectory, TokenizerPreflightCheck, TokenizerPreflightCheck]:
    encoded = encode_sft_dialogue(
        tokenizer,
        dialogue,
        maximum_tokens=maximum_tokens,
    )
    repeated = encode_sft_dialogue(
        tokenizer,
        dialogue,
        maximum_tokens=maximum_tokens,
    )
    if repeated != encoded:
        raise TokenizerPreflightError("tokenizer/template behavior is not deterministic")
    assistant_indices = tuple(
        index for index, message in enumerate(dialogue) if message.role == "assistant"
    )
    if tuple(span.message_index for span in encoded.action_spans) != assistant_indices:
        raise TokenizerPreflightError("assistant span indices do not match the dialogue")
    supervised = {
        index
        for span in encoded.action_spans
        for index in range(span.start, span.end)
    }
    for index, (token, label) in enumerate(
        zip(encoded.input_ids, encoded.labels, strict=True)
    ):
        expected = token if index in supervised else IGNORE_INDEX
        if label != expected:
            raise TokenizerPreflightError("a non-action token is supervised or an action is masked")

    mask_check = TokenizerPreflightCheck.create(
        "assistant_only_masking",
        exhaustive=True,
        item_count=len(encoded.input_ids),
        claim="Every and only assistant-action token is supervised in the complete trajectory.",
        evidence={
            "message_count": len(dialogue),
            "assistant_action_count": len(assistant_indices),
            "full_token_count": len(encoded.input_ids),
            "supervised_token_count": encoded.supervised_token_count,
            "ignored_token_count": len(encoded.input_ids) - encoded.supervised_token_count,
            "encoded_trajectory_digest": encoded.digest,
            "deterministic_repeat_equal": True,
        },
    )

    try:
        encode_sft_dialogue(
            tokenizer,
            dialogue,
            maximum_tokens=len(encoded.input_ids) - 1,
        )
    except TrajectoryEncodingError as exc:
        if "truncation is forbidden" not in str(exc):
            raise TokenizerPreflightError(
                "overlength trajectory failed for a reason other than explicit no-truncation"
            ) from exc
    else:
        raise TokenizerPreflightError("overlength trajectory did not fail closed")
    truncation_check = TokenizerPreflightCheck.create(
        "no_truncation",
        exhaustive=True,
        item_count=len(encoded.input_ids),
        claim="The complete trajectory fits the declared limit and a one-token-short limit fails.",
        evidence={
            "declared_maximum_tokens": maximum_tokens,
            "full_token_count": len(encoded.input_ids),
            "headroom_tokens": maximum_tokens - len(encoded.input_ids),
            "one_token_short_limit_rejected": True,
            "truncation_performed": False,
        },
    )
    return encoded, mask_check, truncation_check


def _probe_action_prefix(
    tokenizer: ChatTokenizerProtocol,
    action_text: str,
    *,
    maximum_tokens: int,
    phase: str,
) -> EncodedTrajectory:
    if phase not in {"inquiry", "answer"}:
        raise AssertionError("unknown action-probe phase")
    user_phase = "opening" if phase == "inquiry" else "terminal"
    dialogue = (
        DialogueMessage("system", "G03 tokenizer action-prefix probe.", "contract"),
        DialogueMessage(
            "user",
            f"Return one canonical {phase} action as JSON.",
            cast(DialoguePhase, user_phase),
        ),
        DialogueMessage("assistant", action_text, "action"),
    )
    generation_text, generation_ids = render_generation_prefix(tokenizer, dialogue[:2])
    encoded = encode_sft_dialogue(
        tokenizer,
        dialogue,
        maximum_tokens=maximum_tokens,
    )
    if (
        len(encoded.action_spans) != 1
        or encoded.action_spans[0].start != len(generation_ids)
        or not generation_text
    ):
        raise TokenizerPreflightError("sampled action has an unstable generation boundary")
    return encoded


def _sampled_action_checks(
    tokenizer: ChatTokenizerProtocol,
    inventory: _ActionInventory,
    *,
    terminal_count: int,
    maximum_tokens: int,
) -> tuple[TokenizerPreflightCheck, str]:
    probe_digests: list[str] = []
    sampled_answer_texts: list[str] = []
    for text in inventory.inquiry_sample_texts:
        parsed = parse_action(text)
        if serialize_action(parsed) != text or type(parsed) not in {ReadyAction, TestAction}:
            raise TokenizerPreflightError("sampled inquiry action is not canonical")
        probe_digests.append(
            _probe_action_prefix(
                tokenizer,
                text,
                maximum_tokens=maximum_tokens,
                phase="inquiry",
            ).digest
        )

    for sample_rank, rule in enumerate(inventory.rule_sample_values):
        classifications = tuple(
            "fits" if (sample_rank + index) % 2 == 0 else "does_not_fit"
            for index in range(terminal_count)
        )
        action = AnswerAction(rule, cast(tuple[Classification, ...], classifications))
        state = build_answer_action(action)
        parsed = parse_action(
            state.text,
            expected_move="answer",
            terminal_count=terminal_count,
        )
        if parsed != action or serialize_action(parsed) != state.text:
            raise TokenizerPreflightError("sampled answer action is not canonical")
        sampled_answer_texts.append(state.text)
        probe_digests.append(
            _probe_action_prefix(
                tokenizer,
                state.text,
                maximum_tokens=maximum_tokens,
                phase="answer",
            ).digest
        )

    sample_digest = json_digest(
        {
            "inquiry_indices": list(inventory.inquiry_sample_indices),
            "inquiry_texts": list(inventory.inquiry_sample_texts),
            "rule_indices": list(inventory.rule_sample_indices),
            "answer_texts": sampled_answer_texts,
            "encoded_probe_digests": probe_digests,
        },
        domain="goalzendo-interactive-tokenizer-action-prefix-samples-v1",
    )
    check = TokenizerPreflightCheck.create(
        "action_prefix_stability",
        exhaustive=False,
        item_count=len(probe_digests),
        claim=(
            "Coverage-preserving inquiry and answer samples retain exact generation, content, "
            "and assistant-terminator token boundaries."
        ),
        evidence={
            "sampled_ready_action_count": 1,
            "sampled_test_action_count": len(inventory.inquiry_sample_indices),
            "sampled_answer_action_count": len(inventory.rule_sample_indices),
            "sampled_action_count": len(probe_digests),
            "sample_digest": sample_digest,
        },
    )
    return check, sample_digest


def run_tokenizer_preflight(
    tokenizer: ChatTokenizerProtocol,
    dialogue: Dialogue,
    *,
    tokenizer_identifier: str,
    maximum_tokens: int,
) -> TokenizerPreflightReport:
    """Run the complete local G03 tokenizer-format preflight.

    Only a fully passing report is returned.  Any unproved boundary,
    noncanonical action, coverage gap, overlength input, or tokenizer
    nondeterminism raises :class:`TokenizerPreflightError`.
    """

    if not isinstance(tokenizer, ChatTokenizerProtocol):
        raise TypeError("tokenizer does not implement ChatTokenizerProtocol")
    if type(dialogue) is not tuple or not dialogue:
        raise TypeError("dialogue must be a nonempty Dialogue tuple")
    if any(type(message) is not DialogueMessage for message in dialogue):
        raise TypeError("dialogue contains a non-DialogueMessage")
    if (
        type(tokenizer_identifier) is not str
        or not tokenizer_identifier
        or not tokenizer_identifier.isascii()
    ):
        raise ValueError("tokenizer_identifier must be nonempty ASCII")
    if (
        isinstance(maximum_tokens, bool)
        or not isinstance(maximum_tokens, int)
        or maximum_tokens < 1
    ):
        raise ValueError("maximum_tokens must be a positive integer")

    try:
        actions, terminal_count = _canonical_dialogue_actions(dialogue)
        role_check, role_behavior_digest = _role_template_check(tokenizer)
        encoded, mask_check, truncation_check = _encoded_dialogue_checks(
            tokenizer,
            dialogue,
            maximum_tokens=maximum_tokens,
        )
        inventory = _canonical_action_inventory()
        prefix_check, prefix_sample_digest = _sampled_action_checks(
            tokenizer,
            inventory,
            terminal_count=terminal_count,
            maximum_tokens=maximum_tokens,
        )
    except TokenizerPreflightError:
        raise
    except Exception as exc:
        raise TokenizerPreflightError(
            f"tokenizer-format preflight could not establish every invariant: {exc}"
        ) from exc

    dialogue_action_digest = json_digest(
        [serialize_action(action) for action in actions],
        domain="goalzendo-interactive-preflight-dialogue-actions-v1",
    )
    roundtrip_check = TokenizerPreflightCheck.create(
        "canonical_action_roundtrips",
        exhaustive=True,
        item_count=SCENE_COUNT + 1 + SYNTACTIC_RULE_COUNT + len(actions),
        claim=(
            "Every test action, ready, every public rule string, and every dialogue action "
            "round-trips canonically."
        ),
        evidence={
            "test_action_count": SCENE_COUNT,
            "ready_action_count": 1,
            "syntactic_rule_count": SYNTACTIC_RULE_COUNT,
            "dialogue_action_count": len(actions),
            "test_action_digest": inventory.test_action_digest,
            "rule_language_digest": inventory.rule_language_digest,
            "dialogue_action_digest": dialogue_action_digest,
            "action_language_digest": action_language_digest(),
            "action_inventory_digest": inventory.inventory_digest,
        },
    )
    coverage_check = TokenizerPreflightCheck.create(
        "enumeration_and_sample_coverage",
        exhaustive=False,
        item_count=(
            SCENE_COUNT
            + 1
            + SYNTACTIC_RULE_COUNT
            + len(inventory.inquiry_sample_texts)
            + len(inventory.rule_sample_values)
        ),
        claim=(
            "Finite action languages are exhaustive and tokenizer probes cover every inquiry "
            "field option plus every rule formula stratum."
        ),
        evidence={
            "enumerated_test_action_count": SCENE_COUNT,
            "enumerated_ready_action_count": 1,
            "enumerated_rule_count": SYNTACTIC_RULE_COUNT,
            "inquiry_option_pair_count": len(inventory.inquiry_option_pairs),
            "inquiry_option_pairs": [list(pair) for pair in inventory.inquiry_option_pairs],
            "tokenized_inquiry_action_count": len(inventory.inquiry_sample_texts),
            "tokenized_answer_action_count": len(inventory.rule_sample_values),
            "inquiry_sample_indices": list(inventory.inquiry_sample_indices),
            "rule_sample_indices": list(inventory.rule_sample_indices),
            "prefix_sample_digest": prefix_sample_digest,
        },
    )
    tokenizer_behavior_digest = json_digest(
        {
            "role_behavior_digest": role_behavior_digest,
            "encoded_trajectory_digest": encoded.digest,
            "prefix_sample_digest": prefix_sample_digest,
        },
        domain="goalzendo-interactive-tokenizer-behavior-v1",
    )
    role_check = TokenizerPreflightCheck.create(
        role_check.check_id,
        exhaustive=role_check.exhaustive,
        item_count=role_check.item_count,
        claim=role_check.claim,
        evidence={
            **role_check.evidence,
            "tokenizer_behavior_digest": tokenizer_behavior_digest,
        },
    )
    dialogue_digest = json_digest(
        dialogue_as_obj(dialogue),
        domain="goalzendo-interactive-tokenizer-preflight-dialogue-v1",
    )
    return TokenizerPreflightReport(
        tokenizer_identifier=tokenizer_identifier,
        maximum_tokens=maximum_tokens,
        dialogue_digest=dialogue_digest,
        encoded_trajectory_digest=encoded.digest,
        checks=(
            role_check,
            mask_check,
            roundtrip_check,
            prefix_check,
            truncation_check,
            coverage_check,
        ),
    )
