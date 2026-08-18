"""Authenticated, bounded-memory trajectory SFT over one-decision evidence.

This module derives the nominal ``VerifiedDecisionTokenExample`` boundary
from authenticated hidden episodes and reference-policy trajectory records; it
never reuses the legacy whole-dialogue trajectory encoding. Plans are
canonical, graph-free source records wrapped in fresh nominal evidence. The
executor regenerates the plan from the typed sources, rederives one decision at
a time, backpropagates its plan-normalized token-sum immediately, releases the
graph, and never accepts or invokes an optimizer.

Nothing here authorizes a model launch, weight update, or optimizer step.
"""

from __future__ import annotations

import gc
import hashlib
import hmac
import math
import weakref
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import cast

import torch
from torch.nn import functional as F

from ._json import CanonicalJSONError, dump_json, json_digest, load_json
from .action_tokenization_v2 import (
    ExactDecodeTokenizerProtocol,
    FragmentActionTokenCompiler,
    FragmentActionTokenizationError,
)
from .actions import InvalidActionError, action_to_obj, parse_action, serialize_action
from .authenticated_model_provider_v2 import (
    AuthenticatedCausalLMProvider,
    AuthenticatedModelProviderError,
    TrainableParameterRegistry,
)
from .decision_encoding_v2 import (
    DecisionEncodingError,
    DecisionTokenExample,
    VerifiedDecisionTokenExample,
    encode_decision_example,
    verify_decision_example,
)
from .dialogue import (
    Dialogue,
    DialogueMessage,
    DialoguePhase,
    DialogueRole,
    dialogue_as_obj,
    render_dialogue,
)
from .environment import play_reference_episode, replay_transcript
from .episodes import HiddenEpisode
from .trajectory_banks import ReferenceTrajectoryRecord
from .trajectory_encoding import IGNORE_INDEX
from .transcripts import TestEvent

STREAMING_SFT_SCHEMA_VERSION = 3
STREAMING_SFT_CONTRACT_ID = "goalzendo-authenticated-streaming-sft-v3"
STREAMING_SFT_AUTHORIZES_EXECUTION = False
STREAMING_SFT_EXECUTION_SCHEMA_VERSION = 4
STREAMING_SFT_EXECUTION_CONTRACT_ID = "goalzendo-authenticated-streaming-sft-execution-v4"

_ACTION_DIGEST_DOMAIN = "goalzendo-interactive-streaming-sft-action-v3"
_REFERENCE_SOURCE_DIGEST_DOMAIN = "goalzendo-interactive-streaming-sft-reference-source-v3"
_ITEM_DIGEST_DOMAIN = "goalzendo-interactive-streaming-sft-item-v3"
_PLAN_DIGEST_DOMAIN = "goalzendo-interactive-streaming-sft-plan-v3"
_VERIFIED_PLAN_DIGEST_DOMAIN = "goalzendo-interactive-streaming-sft-verified-plan-v3"
_FP32_BUFFER_MANIFEST_DOMAIN = "goalzendo-interactive-streaming-sft-fp32-gradient-buffers-v4"
_GRADIENT_MANIFEST_DOMAIN = "goalzendo-interactive-streaming-sft-final-gradients-v4"
_EXECUTION_DIGEST_DOMAIN = "goalzendo-interactive-streaming-sft-execution-v4"

GraphReleaseHook = Callable[[str, weakref.ReferenceType[torch.Tensor]], None]


class StreamingSFTError(ValueError):
    """Raised when the authenticated SFT plan or execution fails closed."""


def _is_sha256(value: object) -> bool:
    return (
        type(value) is str
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _require_sha256(value: object, *, name: str) -> str:
    if not _is_sha256(value):
        raise StreamingSFTError(f"{name} must be a lowercase SHA-256")
    return cast(str, value)


def _require_nonempty_text(value: object, *, name: str) -> str:
    if type(value) is not str or not value:
        raise StreamingSFTError(f"{name} must be nonempty text")
    return value


def _require_nonnegative_integer(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise StreamingSFTError(f"{name} must be a non-negative integer")
    return value


def _require_positive_integer(value: object, *, name: str) -> int:
    selected = _require_nonnegative_integer(value, name=name)
    if selected < 1:
        raise StreamingSFTError(f"{name} must be positive")
    return selected


def _exact_object(value: object, fields: tuple[str, ...], *, name: str) -> dict[str, object]:
    if type(value) is not dict or any(type(key) is not str for key in value):
        raise StreamingSFTError(f"{name} must be a JSON object")
    result = cast(dict[str, object], value)
    if set(result) != set(fields) or len(result) != len(fields):
        raise StreamingSFTError(f"{name} has noncanonical fields")
    return result


def _dialogue_from_obj(value: object) -> Dialogue:
    if type(value) is not list or not value:
        raise StreamingSFTError("decision dialogue must be a nonempty JSON array")
    messages: list[DialogueMessage] = []
    for raw in cast(list[object], value):
        obj = _exact_object(raw, ("role", "content", "phase"), name="dialogue_message")
        try:
            messages.append(
                DialogueMessage(
                    role=cast(DialogueRole, obj["role"]),
                    content=cast(str, obj["content"]),
                    phase=cast(DialoguePhase, obj["phase"]),
                )
            )
        except (TypeError, ValueError) as exc:
            raise StreamingSFTError("decision dialogue message is invalid") from exc
    return cast(Dialogue, tuple(messages))


def _action_digest(raw_action: str) -> str:
    try:
        action = parse_action(raw_action)
    except (InvalidActionError, TypeError, ValueError) as exc:
        raise StreamingSFTError("SFT source action is not canonical") from exc
    if serialize_action(action) != raw_action:
        raise StreamingSFTError("SFT source action changed under canonical serialization")
    return json_digest(action_to_obj(action), domain=_ACTION_DIGEST_DOMAIN)


@dataclass(frozen=True, slots=True)
class ReferenceTrajectorySFTSource:
    """Typed upstream objects from which every teacher decision is regenerated."""

    episode: HiddenEpisode
    reference_trajectory: ReferenceTrajectoryRecord

    def __post_init__(self) -> None:
        if type(self.episode) is not HiddenEpisode:
            raise TypeError("episode must be a HiddenEpisode")
        if type(self.reference_trajectory) is not ReferenceTrajectoryRecord:
            raise TypeError("reference_trajectory must be a ReferenceTrajectoryRecord")


@dataclass(frozen=True, slots=True)
class StreamingSFTPlanItem:
    """Canonical graph-free source for one teacher-forced decision."""

    episode_id: str
    episode_digest: str
    reference_trajectory_digest: str
    decision_index: int
    dialogue: Dialogue
    dialogue_digest: str
    raw_action: str
    action_digest: str
    action_trace_digest: str
    tokenizer_binding_digest: str
    compiler_manifest_digest: str
    decision_example_digest: str
    decision_verification_digest: str
    decision_example_json: str
    maximum_sequence_tokens: int
    supervised_token_count: int

    def __post_init__(self) -> None:
        _require_nonempty_text(self.episode_id, name="episode_id")
        _require_sha256(self.episode_digest, name="episode_digest")
        _require_sha256(
            self.reference_trajectory_digest,
            name="reference_trajectory_digest",
        )
        _require_nonnegative_integer(self.decision_index, name="decision_index")
        for name in (
            "dialogue_digest",
            "action_digest",
            "action_trace_digest",
            "tokenizer_binding_digest",
            "compiler_manifest_digest",
            "decision_example_digest",
            "decision_verification_digest",
        ):
            _require_sha256(getattr(self, name), name=name)
        _require_positive_integer(
            self.maximum_sequence_tokens,
            name="maximum_sequence_tokens",
        )
        _require_positive_integer(
            self.supervised_token_count,
            name="supervised_token_count",
        )
        if (
            type(self.dialogue) is not tuple
            or not self.dialogue
            or any(type(message) is not DialogueMessage for message in self.dialogue)
            or self.dialogue[-1].role != "user"
        ):
            raise StreamingSFTError("plan dialogue is not a typed decision boundary")
        if _action_digest(self.raw_action) != self.action_digest:
            raise StreamingSFTError("plan action digest is inconsistent")
        try:
            example = DecisionTokenExample.from_json(self.decision_example_json)
        except (DecisionEncodingError, TypeError, ValueError) as exc:
            raise StreamingSFTError("plan decision example is not canonical verified source") from exc
        if (
            example.digest != self.decision_example_digest
            or example.dialogue_digest != self.dialogue_digest
            or example.action_trace.raw_action != self.raw_action
            or example.action_trace.digest != self.action_trace_digest
            or example.tokenizer_binding_digest != self.tokenizer_binding_digest
            or example.action_trace.compiler_manifest_digest != self.compiler_manifest_digest
            or example.maximum_sequence_tokens != self.maximum_sequence_tokens
            or example.action_token_count != self.supervised_token_count
        ):
            raise StreamingSFTError("plan decision fields differ from its canonical example")

    @property
    def sort_key(self) -> tuple[str, str, int, str, str]:
        return (
            self.episode_digest,
            self.reference_trajectory_digest,
            self.decision_index,
            self.dialogue_digest,
            self.action_digest,
        )

    def as_obj(self) -> dict[str, object]:
        return {
            "episode_id": self.episode_id,
            "episode_digest": self.episode_digest,
            "reference_trajectory_digest": self.reference_trajectory_digest,
            "decision_index": self.decision_index,
            "dialogue": dialogue_as_obj(self.dialogue),
            "dialogue_digest": self.dialogue_digest,
            "raw_action": self.raw_action,
            "action_digest": self.action_digest,
            "action_trace_digest": self.action_trace_digest,
            "tokenizer_binding_digest": self.tokenizer_binding_digest,
            "compiler_manifest_digest": self.compiler_manifest_digest,
            "decision_example_digest": self.decision_example_digest,
            "decision_verification_digest": self.decision_verification_digest,
            "decision_example_json": self.decision_example_json,
            "maximum_sequence_tokens": self.maximum_sequence_tokens,
            "supervised_token_count": self.supervised_token_count,
        }

    @property
    def digest(self) -> str:
        return json_digest(self.as_obj(), domain=_ITEM_DIGEST_DOMAIN)

    @classmethod
    def from_obj(cls, value: object) -> StreamingSFTPlanItem:
        fields = (
            "episode_id",
            "episode_digest",
            "reference_trajectory_digest",
            "decision_index",
            "dialogue",
            "dialogue_digest",
            "raw_action",
            "action_digest",
            "action_trace_digest",
            "tokenizer_binding_digest",
            "compiler_manifest_digest",
            "decision_example_digest",
            "decision_verification_digest",
            "decision_example_json",
            "maximum_sequence_tokens",
            "supervised_token_count",
            "digest",
        )
        obj = _exact_object(value, fields, name="streaming_sft_plan_item")
        result = cls(
            episode_id=cast(str, obj["episode_id"]),
            episode_digest=cast(str, obj["episode_digest"]),
            reference_trajectory_digest=cast(str, obj["reference_trajectory_digest"]),
            decision_index=cast(int, obj["decision_index"]),
            dialogue=_dialogue_from_obj(obj["dialogue"]),
            dialogue_digest=cast(str, obj["dialogue_digest"]),
            raw_action=cast(str, obj["raw_action"]),
            action_digest=cast(str, obj["action_digest"]),
            action_trace_digest=cast(str, obj["action_trace_digest"]),
            tokenizer_binding_digest=cast(str, obj["tokenizer_binding_digest"]),
            compiler_manifest_digest=cast(str, obj["compiler_manifest_digest"]),
            decision_example_digest=cast(str, obj["decision_example_digest"]),
            decision_verification_digest=cast(
                str,
                obj["decision_verification_digest"],
            ),
            decision_example_json=cast(str, obj["decision_example_json"]),
            maximum_sequence_tokens=cast(int, obj["maximum_sequence_tokens"]),
            supervised_token_count=cast(int, obj["supervised_token_count"]),
        )
        if not hmac.compare_digest(_require_sha256(obj["digest"], name="item.digest"), result.digest):
            raise StreamingSFTError("streaming SFT plan-item digest check failed")
        return result


@dataclass(frozen=True, slots=True)
class StreamingSFTPlan:
    """Canonical graph-free execution plan, sorted independently of input order."""

    tokenizer_binding_digest: str
    compiler_manifest_digest: str
    model_provenance_digest: str
    provider_policy_state_digest: str
    items: tuple[StreamingSFTPlanItem, ...]
    total_supervised_tokens: int
    schema_version: int = STREAMING_SFT_SCHEMA_VERSION
    contract_id: str = STREAMING_SFT_CONTRACT_ID

    def __post_init__(self) -> None:
        if self.schema_version != STREAMING_SFT_SCHEMA_VERSION:
            raise StreamingSFTError("unexpected streaming-SFT schema version")
        if self.contract_id != STREAMING_SFT_CONTRACT_ID:
            raise StreamingSFTError("unexpected streaming-SFT contract id")
        for name in (
            "tokenizer_binding_digest",
            "compiler_manifest_digest",
            "model_provenance_digest",
            "provider_policy_state_digest",
        ):
            _require_sha256(getattr(self, name), name=name)
        items = tuple(self.items)
        object.__setattr__(self, "items", items)
        if not items or any(type(item) is not StreamingSFTPlanItem for item in items):
            raise StreamingSFTError("streaming SFT plan requires typed decision items")
        if items != tuple(sorted(items, key=lambda item: item.sort_key)):
            raise StreamingSFTError("streaming SFT plan items are not in canonical order")
        coordinates = tuple(
            (item.episode_digest, item.reference_trajectory_digest, item.decision_index) for item in items
        )
        if len(set(coordinates)) != len(coordinates):
            raise StreamingSFTError("streaming SFT plan contains a duplicate decision coordinate")
        groups: dict[tuple[str, str], list[int]] = {}
        identities: dict[tuple[str, str], str] = {}
        for item in items:
            if (
                item.tokenizer_binding_digest != self.tokenizer_binding_digest
                or item.compiler_manifest_digest != self.compiler_manifest_digest
            ):
                raise StreamingSFTError("plan decisions disagree on tokenizer/compiler binding")
            group = (item.episode_digest, item.reference_trajectory_digest)
            groups.setdefault(group, []).append(item.decision_index)
            previous_id = identities.setdefault(group, item.episode_id)
            if previous_id != item.episode_id:
                raise StreamingSFTError("one episode digest maps to multiple episode ids")
        if any(indices != list(range(len(indices))) for indices in groups.values()):
            raise StreamingSFTError(
                "each reference trajectory must contain contiguous decisions starting at zero"
            )
        expected_tokens = sum(item.supervised_token_count for item in items)
        if (
            _require_positive_integer(
                self.total_supervised_tokens,
                name="total_supervised_tokens",
            )
            != expected_tokens
        ):
            raise StreamingSFTError("plan-wide supervised-token total is inconsistent")

    def as_obj(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "contract_id": self.contract_id,
            "authorizes_execution": False,
            "tokenizer_binding_digest": self.tokenizer_binding_digest,
            "compiler_manifest_digest": self.compiler_manifest_digest,
            "model_provenance_digest": self.model_provenance_digest,
            "provider_policy_state_digest": self.provider_policy_state_digest,
            "decision_count": len(self.items),
            "total_supervised_tokens": self.total_supervised_tokens,
            "items": [{**item.as_obj(), "digest": item.digest} for item in self.items],
        }

    @property
    def digest(self) -> str:
        return json_digest(self.as_obj(), domain=_PLAN_DIGEST_DOMAIN)

    def to_json(self) -> str:
        return dump_json({**self.as_obj(), "digest": self.digest})

    @classmethod
    def from_json(cls, text: str) -> StreamingSFTPlan:
        try:
            raw = load_json(text)
        except CanonicalJSONError as exc:
            raise StreamingSFTError("streaming SFT plan is not strict JSON") from exc
        fields = (
            "schema_version",
            "contract_id",
            "authorizes_execution",
            "tokenizer_binding_digest",
            "compiler_manifest_digest",
            "model_provenance_digest",
            "provider_policy_state_digest",
            "decision_count",
            "total_supervised_tokens",
            "items",
            "digest",
        )
        obj = _exact_object(raw, fields, name="streaming_sft_plan")
        if obj["authorizes_execution"] is not False:
            raise StreamingSFTError("streaming SFT plan may not authorize execution")
        if type(obj["items"]) is not list:
            raise StreamingSFTError("streaming SFT plan items must be an array")
        items = tuple(StreamingSFTPlanItem.from_obj(item) for item in cast(list[object], obj["items"]))
        if obj["decision_count"] != len(items):
            raise StreamingSFTError("streaming SFT decision count is inconsistent")
        result = cls(
            tokenizer_binding_digest=cast(str, obj["tokenizer_binding_digest"]),
            compiler_manifest_digest=cast(str, obj["compiler_manifest_digest"]),
            model_provenance_digest=cast(str, obj["model_provenance_digest"]),
            provider_policy_state_digest=cast(str, obj["provider_policy_state_digest"]),
            items=items,
            total_supervised_tokens=cast(int, obj["total_supervised_tokens"]),
            schema_version=cast(int, obj["schema_version"]),
            contract_id=cast(str, obj["contract_id"]),
        )
        if not hmac.compare_digest(_require_sha256(obj["digest"], name="plan.digest"), result.digest):
            raise StreamingSFTError("streaming SFT plan digest check failed")
        if not hmac.compare_digest(text, result.to_json()):
            raise StreamingSFTError("streaming SFT plan JSON is not uniquely canonical")
        return result


@dataclass(frozen=True, slots=True, init=False)
class _VerifiedStreamingSFTPlan:
    """Nominal evidence produced only by fresh typed-source derivation."""

    plan: StreamingSFTPlan
    verification_digest: str

    @classmethod
    def _from_derived(cls, plan: StreamingSFTPlan) -> _VerifiedStreamingSFTPlan:
        if type(plan) is not StreamingSFTPlan:
            raise TypeError("plan must be a StreamingSFTPlan")
        result = object.__new__(cls)
        object.__setattr__(result, "plan", plan)
        object.__setattr__(
            result,
            "verification_digest",
            json_digest(
                {
                    "plan_digest": plan.digest,
                    "item_digests": [item.digest for item in plan.items],
                    "reference_trajectory_digests": [item.reference_trajectory_digest for item in plan.items],
                    "source": "fresh_typed_reference_trajectory_sources",
                },
                domain=_VERIFIED_PLAN_DIGEST_DOMAIN,
            ),
        )
        return result


def _provider_bindings(provider: AuthenticatedCausalLMProvider) -> tuple[str, str]:
    provenance = provider.reauthenticate_policy_state()
    if provenance.policy_state_digest != provider.policy_state_digest:
        raise StreamingSFTError("provider reauthentication returned another policy state")
    if provenance.digest != provider.model_provenance_digest:
        raise StreamingSFTError("provider reauthentication returned another model provenance")
    return provenance.digest, provenance.policy_state_digest


def _authenticate_reference_source(
    source: ReferenceTrajectorySFTSource,
) -> tuple[Dialogue, str]:
    """Regenerate the reference policy record and derive its model-facing source."""

    episode = source.episode
    supplied = source.reference_trajectory
    try:
        expected_transcript = play_reference_episode(episode)
        # Independently prove that the supplied transcript is a valid replay, even
        # before requiring the stronger exact-reference-policy equality below.
        replayed = replay_transcript(episode, supplied.transcript)
        expected_dialogue = render_dialogue(episode, expected_transcript)
        expected_query_count = sum(type(event) is TestEvent for event in expected_transcript.events)
        expected_record = ReferenceTrajectoryRecord(
            episode_id=episode.episode_id,
            episode_digest=episode.digest,
            transcript=expected_transcript,
            transcript_digest=expected_transcript.digest,
            dialogue_manifest_digest=json_digest(
                dialogue_as_obj(expected_dialogue),
                domain="goalzendo-interactive-reference-dialogue-v1",
            ),
            message_count=len(expected_dialogue),
            assistant_action_count=sum(message.role == "assistant" for message in expected_dialogue),
            query_count=expected_query_count,
            terminal_reward_numerator=120 - expected_query_count,
            terminal_reward_denominator=120,
        )
    except (RuntimeError, TypeError, ValueError) as exc:
        raise StreamingSFTError(
            "reference trajectory could not be authenticated against its hidden episode"
        ) from exc
    if replayed.transcript != supplied.transcript:
        raise StreamingSFTError("reference trajectory replay changed its transcript")
    if supplied.as_obj() != expected_record.as_obj():
        raise StreamingSFTError("reference trajectory differs from the regenerated reference policy record")
    source_digest = json_digest(
        {
            "episode": episode.as_obj(),
            "reference_trajectory": expected_record.as_obj(),
        },
        domain=_REFERENCE_SOURCE_DIGEST_DOMAIN,
    )
    return expected_dialogue, source_digest


def build_streaming_sft_plan(
    sources: Sequence[ReferenceTrajectorySFTSource],
    tokenizer: ExactDecodeTokenizerProtocol,
    compiler: FragmentActionTokenCompiler,
    provider: AuthenticatedCausalLMProvider,
    *,
    maximum_sequence_tokens: int,
) -> _VerifiedStreamingSFTPlan:
    """Authenticate reference records and freeze their derived decisions."""

    if isinstance(sources, (str, bytes)) or not isinstance(sources, Sequence) or not sources:
        raise StreamingSFTError("streaming SFT requires at least one reference trajectory source")
    if not isinstance(tokenizer, ExactDecodeTokenizerProtocol):
        raise TypeError("tokenizer must implement ExactDecodeTokenizerProtocol")
    if type(compiler) is not FragmentActionTokenCompiler:
        raise TypeError("compiler must be a FragmentActionTokenCompiler")
    if type(provider) is not AuthenticatedCausalLMProvider:
        raise TypeError("provider must be an AuthenticatedCausalLMProvider")
    selected_maximum = _require_positive_integer(
        maximum_sequence_tokens,
        name="maximum_sequence_tokens",
    )
    try:
        compiler_manifest = compiler.manifest
    except FragmentActionTokenizationError as exc:
        raise StreamingSFTError("streaming SFT requires a frozen compiler") from exc
    model_digest, policy_digest = _provider_bindings(provider)

    items: list[StreamingSFTPlanItem] = []
    for source in sources:
        if type(source) is not ReferenceTrajectorySFTSource:
            raise TypeError("sources must contain ReferenceTrajectorySFTSource values")
        dialogue, reference_digest = _authenticate_reference_source(source)
        assistant_count = 0
        for message_index, message in enumerate(dialogue):
            if message.role != "assistant":
                continue
            if message.phase != "action" or message_index < 1:
                raise StreamingSFTError("reference dialogue has a noncanonical assistant action")
            decision_dialogue = dialogue[:message_index]
            if not decision_dialogue or decision_dialogue[-1].role != "user":
                raise StreamingSFTError("reference action lacks an exact decision boundary")
            try:
                action = parse_action(message.content)
                example = encode_decision_example(
                    tokenizer,
                    compiler,
                    decision_dialogue,
                    action,
                    tokenizer_binding_digest=compiler_manifest.tokenizer_manifest.digest,
                    maximum_sequence_tokens=selected_maximum,
                )
                verified = verify_decision_example(
                    example,
                    tokenizer,
                    compiler,
                    decision_dialogue,
                )
            except (
                DecisionEncodingError,
                FragmentActionTokenizationError,
                InvalidActionError,
                TypeError,
                ValueError,
            ) as exc:
                raise StreamingSFTError(
                    "reference action did not produce exact one-decision evidence"
                ) from exc
            raw_action = verified.example.action_trace.raw_action
            item = StreamingSFTPlanItem(
                episode_id=source.episode.episode_id,
                episode_digest=source.episode.digest,
                reference_trajectory_digest=reference_digest,
                decision_index=assistant_count,
                dialogue=decision_dialogue,
                dialogue_digest=verified.example.dialogue_digest,
                raw_action=raw_action,
                action_digest=_action_digest(raw_action),
                action_trace_digest=verified.example.action_trace.digest,
                tokenizer_binding_digest=verified.example.tokenizer_binding_digest,
                compiler_manifest_digest=(verified.example.action_trace.compiler_manifest_digest),
                decision_example_digest=verified.example.digest,
                decision_verification_digest=verified.verification_digest,
                decision_example_json=verified.example.to_json(),
                maximum_sequence_tokens=verified.example.maximum_sequence_tokens,
                supervised_token_count=verified.example.action_token_count,
            )
            if (
                item.tokenizer_binding_digest != compiler_manifest.tokenizer_manifest.digest
                or item.compiler_manifest_digest != compiler_manifest.digest
            ):
                raise StreamingSFTError("derived decision differs from frozen tokenizer/compiler")
            items.append(item)
            assistant_count += 1
        if assistant_count != source.reference_trajectory.assistant_action_count:
            raise StreamingSFTError("derived decision count differs from the authenticated reference record")

    ordered = tuple(sorted(items, key=lambda item: item.sort_key))
    plan = StreamingSFTPlan(
        tokenizer_binding_digest=compiler_manifest.tokenizer_manifest.digest,
        compiler_manifest_digest=compiler_manifest.digest,
        model_provenance_digest=model_digest,
        provider_policy_state_digest=policy_digest,
        items=ordered,
        total_supervised_tokens=sum(item.supervised_token_count for item in ordered),
    )
    after_model, after_policy = _provider_bindings(provider)
    if after_model != model_digest or after_policy != policy_digest:
        raise StreamingSFTError("provider changed while the streaming SFT plan was frozen")
    # Round-trip the exact bytes now; the executor repeats source regeneration.
    if StreamingSFTPlan.from_json(plan.to_json()) != plan:
        raise StreamingSFTError("streaming SFT plan failed its canonical round trip")
    return _VerifiedStreamingSFTPlan._from_derived(plan)


def _rederive_plan_item(
    item: StreamingSFTPlanItem,
    tokenizer: ExactDecodeTokenizerProtocol,
    compiler: FragmentActionTokenCompiler,
) -> VerifiedDecisionTokenExample:
    try:
        stored = DecisionTokenExample.from_json(item.decision_example_json)
        action = parse_action(item.raw_action)
        rebuilt = encode_decision_example(
            tokenizer,
            compiler,
            item.dialogue,
            action,
            tokenizer_binding_digest=item.tokenizer_binding_digest,
            maximum_sequence_tokens=item.maximum_sequence_tokens,
        )
        verified = verify_decision_example(rebuilt, tokenizer, compiler, item.dialogue)
    except (
        DecisionEncodingError,
        FragmentActionTokenizationError,
        InvalidActionError,
        TypeError,
        ValueError,
    ) as exc:
        raise StreamingSFTError("plan decision failed canonical source regeneration") from exc
    if (
        stored.to_json() != rebuilt.to_json()
        or rebuilt.digest != item.decision_example_digest
        or rebuilt.dialogue_digest != item.dialogue_digest
        or rebuilt.action_trace.digest != item.action_trace_digest
        or _action_digest(item.raw_action) != item.action_digest
        or verified.verification_digest != item.decision_verification_digest
        or rebuilt.action_token_count != item.supervised_token_count
    ):
        raise StreamingSFTError("plan decision differs from its frozen one-decision evidence")
    return verified


@dataclass(frozen=True, slots=True)
class _CanonicalParameter:
    name: str
    aliases: tuple[str, ...]
    parameter: torch.nn.Parameter


def _named_parameters(model: torch.nn.Module) -> tuple[_CanonicalParameter, ...]:
    supplied = tuple(model.named_parameters(recurse=True, remove_duplicate=False))
    names: set[str] = set()
    aliases_by_identity: dict[int, list[str]] = {}
    parameters: dict[int, torch.nn.Parameter] = {}
    for name, parameter in supplied:
        if type(name) is not str or not name or name in names:
            raise StreamingSFTError("model parameter names must be nonempty and unique")
        names.add(name)
        if not parameter.requires_grad:
            continue
        if not parameter.is_floating_point():
            raise StreamingSFTError("FP32 accumulation requires floating-point trainable parameters")
        aliases_by_identity.setdefault(id(parameter), []).append(name)
        parameters[id(parameter)] = parameter
    if not parameters:
        raise StreamingSFTError("streaming SFT model has no trainable parameters")
    return tuple(
        sorted(
            (
                _CanonicalParameter(
                    name=min(aliases),
                    aliases=tuple(sorted(aliases)),
                    parameter=parameters[identity],
                )
                for identity, aliases in aliases_by_identity.items()
            ),
            key=lambda item: item.name,
        )
    )


def _registered_parameters(
    registry: TrainableParameterRegistry,
) -> tuple[tuple[_CanonicalParameter, ...], str, int]:
    """Resolve the provider-authenticated complete trainable parameter set."""

    if type(registry) is not TrainableParameterRegistry:
        raise StreamingSFTError("provider trainable-parameter registry lacks exact nominal evidence")
    records = tuple(record for record in registry.manifest.records if record.requires_grad)
    live_parameters = registry.parameters
    if (
        not live_parameters
        or len(records) != len(live_parameters)
        or registry.manifest.trainable_parameter_count != len(live_parameters)
    ):
        raise StreamingSFTError("provider trainable-parameter registry is empty or internally inconsistent")
    result: list[_CanonicalParameter] = []
    seen: set[int] = set()
    for record, parameter in zip(records, live_parameters, strict=True):
        aliases = tuple(record.aliases)
        if (
            not aliases
            or tuple(sorted(set(aliases))) != aliases
            or record.canonical_name != aliases[0]
            or id(parameter) in seen
            or not parameter.requires_grad
            or not parameter.is_floating_point()
            or str(parameter.dtype) != record.dtype
            or str(parameter.device) != record.device
            or str(parameter.layout) != record.layout
            or tuple(parameter.shape) != record.shape
            or tuple(parameter.stride()) != record.stride
            or int(parameter.storage_offset()) != record.storage_offset
            or int(parameter.numel()) != record.numel
        ):
            raise StreamingSFTError("provider trainable-parameter registry differs from its live parameters")
        seen.add(id(parameter))
        result.append(
            _CanonicalParameter(
                name=record.canonical_name,
                aliases=aliases,
                parameter=parameter,
            )
        )
    canonical = tuple(result)
    if tuple(item.name for item in canonical) != tuple(sorted(item.name for item in canonical)):
        raise StreamingSFTError("provider trainable parameters are not canonically ordered")
    return (
        canonical,
        registry.manifest.digest,
        sum(len(item.aliases) for item in canonical),
    )


def _require_zero_starting_gradients(
    parameters: tuple[_CanonicalParameter, ...],
) -> None:
    for item in parameters:
        if item.parameter.grad is not None:
            raise StreamingSFTError(f"starting gradient for {item.name!r} must be strictly absent")


def _require_parameter_grads_none(
    parameters: tuple[_CanonicalParameter, ...],
) -> None:
    for item in parameters:
        if item.parameter.grad is not None:
            raise StreamingSFTError("parameter gradients must remain strictly None until final commit")


def _decision_token_sum(
    logits: torch.Tensor,
    verified: VerifiedDecisionTokenExample,
) -> tuple[torch.Tensor, int]:
    example = verified.example
    if logits.ndim != 2 or logits.shape[0] != len(example.input_ids):
        raise StreamingSFTError("decision logits differ from the canonical input sequence")
    if logits.shape[0] < 2 or logits.shape[1] < 2 or not logits.is_floating_point():
        raise StreamingSFTError("decision logits cannot define next-token cross-entropy")
    if not bool(torch.isfinite(logits.detach()).all().item()):
        raise StreamingSFTError("decision logits contain a non-finite value")
    labels = torch.tensor(example.labels, dtype=torch.long, device=logits.device)
    targets = labels[1:]
    mask = targets.ne(IGNORE_INDEX)
    count = int(mask.sum().item())
    if count != example.action_token_count or count < 1:
        raise StreamingSFTError("decision supervision differs from exact action-token count")
    selected_targets = targets[mask]
    if bool((selected_targets < 0).any().item()) or bool((selected_targets >= logits.shape[1]).any().item()):
        raise StreamingSFTError("decision target lies outside the model vocabulary")
    selected_logits = logits[:-1][mask]
    token_sum = F.cross_entropy(selected_logits, selected_targets, reduction="sum")
    if token_sum.ndim != 0 or not bool(torch.isfinite(token_sum.detach()).item()):
        raise StreamingSFTError("decision cross-entropy token sum is non-finite")
    return token_sum, count


def _raw_tensor_bytes(tensor: torch.Tensor) -> bytes:
    try:
        return (
            tensor.detach()
            .to(device="cpu")
            .contiguous()
            .reshape(-1)
            .view(torch.uint8)
            .numpy()
            .tobytes(order="C")
        )
    except (RuntimeError, TypeError, ValueError) as exc:
        raise StreamingSFTError("gradient bytes could not be materialized") from exc


@dataclass(frozen=True, slots=True)
class ParameterGradientRecord:
    name: str
    aliases: tuple[str, ...]
    parameter_dtype: str
    parameter_device: str
    parameter_shape: tuple[int, ...]
    gradient_dtype: str
    gradient_device: str
    sha256: str
    l2_norm_hex: str
    maximum_absolute_value_hex: str

    def __post_init__(self) -> None:
        _require_nonempty_text(self.name, name="gradient.name")
        aliases = tuple(self.aliases)
        object.__setattr__(self, "aliases", aliases)
        if not aliases or aliases[0] != self.name or tuple(sorted(set(aliases))) != aliases:
            raise StreamingSFTError("gradient aliases must be canonical")
        _require_nonempty_text(self.parameter_dtype, name="gradient.parameter_dtype")
        _require_nonempty_text(self.parameter_device, name="gradient.parameter_device")
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in self.parameter_shape
        ):
            raise StreamingSFTError("gradient parameter shape is invalid")
        _require_nonempty_text(self.gradient_dtype, name="gradient.gradient_dtype")
        _require_nonempty_text(self.gradient_device, name="gradient.gradient_device")
        _require_sha256(self.sha256, name="gradient.sha256")
        for name in ("l2_norm_hex", "maximum_absolute_value_hex"):
            value = getattr(self, name)
            if type(value) is not str:
                raise StreamingSFTError(f"{name} must be canonical float.hex text")
            try:
                parsed = float.fromhex(value)
            except ValueError as exc:
                raise StreamingSFTError(f"{name} is not float.hex text") from exc
            if parsed.hex() != value or not math.isfinite(parsed) or parsed < 0:
                raise StreamingSFTError(f"{name} must be finite, non-negative, and canonical")

    def as_obj(self) -> dict[str, object]:
        return {
            "name": self.name,
            "aliases": list(self.aliases),
            "parameter_dtype": self.parameter_dtype,
            "parameter_device": self.parameter_device,
            "parameter_shape": list(self.parameter_shape),
            "gradient_dtype": self.gradient_dtype,
            "gradient_device": self.gradient_device,
            "sha256": self.sha256,
            "l2_norm_hex": self.l2_norm_hex,
            "maximum_absolute_value_hex": self.maximum_absolute_value_hex,
        }


def _gradient_records(
    parameters: tuple[_CanonicalParameter, ...],
    tensors: dict[int, torch.Tensor],
) -> tuple[ParameterGradientRecord, ...]:
    records: list[ParameterGradientRecord] = []
    for item in parameters:
        parameter = item.parameter
        gradient = tensors.get(id(parameter))
        if gradient is None:
            raise StreamingSFTError(f"gradient manifest omitted trainable parameter {item.name!r}")
        detached = gradient.detach()
        if detached.requires_grad or detached.grad_fn is not None:
            raise StreamingSFTError(f"gradient for {item.name!r} retained a graph")
        if not bool(torch.isfinite(detached).all().item()):
            raise StreamingSFTError(f"gradient for {item.name!r} became non-finite")
        raw = _raw_tensor_bytes(detached)
        flat64 = detached.to(device="cpu", dtype=torch.float64).reshape(-1)
        l2_norm = float(torch.linalg.vector_norm(flat64).item())
        maximum = float(torch.max(torch.abs(flat64)).item()) if flat64.numel() else 0.0
        if not math.isfinite(l2_norm) or not math.isfinite(maximum):
            raise StreamingSFTError(f"gradient summary for {item.name!r} is non-finite")
        records.append(
            ParameterGradientRecord(
                name=item.name,
                aliases=item.aliases,
                parameter_dtype=str(parameter.dtype),
                parameter_device=str(parameter.device),
                parameter_shape=tuple(parameter.shape),
                gradient_dtype=str(detached.dtype),
                gradient_device=str(detached.device),
                sha256=hashlib.sha256(raw).hexdigest(),
                l2_norm_hex=l2_norm.hex(),
                maximum_absolute_value_hex=maximum.hex(),
            )
        )
    return tuple(records)


class _FP32GradientAccumulator:
    """Accumulate canonical per-decision gradients outside ``Parameter.grad``."""

    __slots__ = (
        "buffers",
        "committed_gradients",
        "contribution_cast_count",
        "final_cast_count",
        "parameters",
        "reached_parameter_ids",
    )

    def __init__(self, parameters: tuple[_CanonicalParameter, ...]) -> None:
        self.parameters = parameters
        self.buffers = {
            id(item.parameter): torch.zeros_like(
                item.parameter,
                dtype=torch.float32,
                memory_format=torch.preserve_format,
            )
            for item in parameters
        }
        self.reached_parameter_ids: set[int] = set()
        self.contribution_cast_count = 0
        self.final_cast_count = 0
        self.committed_gradients: dict[int, torch.Tensor] = {}

    @property
    def parameter_ids(self) -> frozenset[int]:
        return frozenset(id(item.parameter) for item in self.parameters)

    def accumulate(self, loss: torch.Tensor) -> tuple[frozenset[int], int]:
        _require_parameter_grads_none(self.parameters)
        ordered_parameters = tuple(item.parameter for item in self.parameters)
        contributions = cast(
            tuple[torch.Tensor | None, ...],
            torch.autograd.grad(
                loss,
                ordered_parameters,
                allow_unused=True,
                create_graph=False,
                retain_graph=False,
            ),
        )
        reached: set[int] = set()
        cast_count = 0
        try:
            for item, contribution in zip(
                self.parameters,
                contributions,
                strict=True,
            ):
                if contribution is None:
                    continue
                parameter = item.parameter
                if (
                    contribution.requires_grad
                    or contribution.grad_fn is not None
                    or tuple(contribution.shape) != tuple(parameter.shape)
                ):
                    raise StreamingSFTError("per-decision parameter gradient contribution is invalid")
                if not bool(torch.isfinite(contribution.detach()).all().item()):
                    raise StreamingSFTError("per-decision parameter gradient is non-finite")
                detached = contribution.detach()
                if detached.is_sparse:
                    detached = detached.coalesce().to_dense()
                fp32 = detached.to(device=parameter.device, dtype=torch.float32)
                if not bool(torch.isfinite(fp32).all().item()):
                    raise StreamingSFTError("per-decision FP32 gradient contribution is non-finite")
                with torch.no_grad():
                    self.buffers[id(parameter)].add_(fp32)
                reached.add(id(parameter))
                cast_count += 1
        finally:
            contributions = ()
        self.reached_parameter_ids.update(reached)
        self.contribution_cast_count += cast_count
        _require_parameter_grads_none(self.parameters)
        return frozenset(reached), cast_count

    def finalize(
        self,
    ) -> tuple[tuple[ParameterGradientRecord, ...], tuple[ParameterGradientRecord, ...]]:
        _require_parameter_grads_none(self.parameters)
        missing = self.parameter_ids - self.reached_parameter_ids
        if missing:
            names = tuple(item.name for item in self.parameters if id(item.parameter) in missing)
            raise StreamingSFTError(f"trainable model parameters were unused: {names!r}")
        for buffer in self.buffers.values():
            if (
                buffer.dtype != torch.float32
                or buffer.requires_grad
                or buffer.grad_fn is not None
                or not bool(torch.isfinite(buffer).all().item())
            ):
                raise StreamingSFTError("an accumulated FP32 gradient buffer is non-finite or invalid")
        buffer_records = _gradient_records(self.parameters, self.buffers)
        finals: dict[int, torch.Tensor] = {}
        for item in self.parameters:
            parameter = item.parameter
            final = (
                self.buffers[id(parameter)]
                .to(device=parameter.device, dtype=parameter.dtype)
                .detach()
                .clone(memory_format=torch.preserve_format)
            )
            if (
                final.requires_grad
                or final.grad_fn is not None
                or not bool(torch.isfinite(final).all().item())
            ):
                raise StreamingSFTError("a final cast parameter gradient is invalid")
            finals[id(parameter)] = final
        _require_parameter_grads_none(self.parameters)
        for item in self.parameters:
            parameter = item.parameter
            parameter.grad = finals[id(parameter)]
            committed = parameter.grad
            if committed is None:
                raise StreamingSFTError("final parameter gradient assignment failed")
            self.committed_gradients[id(parameter)] = committed
            self.final_cast_count += 1
        return buffer_records, _gradient_records(
            self.parameters,
            self.committed_gradients,
        )

    def clear_created_final_gradients(self) -> None:
        for item in self.parameters:
            identity = id(item.parameter)
            gradient = self.committed_gradients.get(identity)
            if gradient is not None and item.parameter.grad is gradient:
                item.parameter.grad = None
        self.committed_gradients.clear()

    def clear_buffers(self) -> None:
        self.buffers.clear()


@dataclass(frozen=True, slots=True)
class StreamingSFTExecution:
    """Graph-free evidence from one backward-only sequential execution."""

    plan_digest: str
    model_provenance_digest: str
    provider_policy_state_digest: str
    trainable_parameter_registry_digest: str
    objective_value_hex: str
    decision_count: int
    supervised_token_count: int
    released_graph_count: int
    parameter_count: int
    registered_parameter_name_count: int
    gradient_accumulation_dtype: str
    gradient_accumulation_order: str
    fp32_contribution_cast_count: int
    final_gradient_cast_count: int
    fp32_gradient_buffers: tuple[ParameterGradientRecord, ...]
    fp32_gradient_buffer_manifest_digest: str
    gradients: tuple[ParameterGradientRecord, ...]
    gradient_manifest_digest: str
    optimizer_step_performed: bool = False
    authorizes_execution: bool = False
    schema_version: int = STREAMING_SFT_EXECUTION_SCHEMA_VERSION
    contract_id: str = STREAMING_SFT_EXECUTION_CONTRACT_ID

    def __post_init__(self) -> None:
        if self.schema_version != STREAMING_SFT_EXECUTION_SCHEMA_VERSION:
            raise StreamingSFTError("unexpected streaming-SFT execution schema")
        if self.contract_id != STREAMING_SFT_EXECUTION_CONTRACT_ID:
            raise StreamingSFTError("unexpected streaming-SFT execution contract")
        if self.authorizes_execution is not False or self.optimizer_step_performed is not False:
            raise StreamingSFTError("streaming SFT execution may not authorize or perform a step")
        for name in (
            "plan_digest",
            "model_provenance_digest",
            "provider_policy_state_digest",
            "trainable_parameter_registry_digest",
            "fp32_gradient_buffer_manifest_digest",
            "gradient_manifest_digest",
        ):
            _require_sha256(getattr(self, name), name=name)
        objective = self.objective_value_hex
        if type(objective) is not str:
            raise StreamingSFTError("objective_value_hex must be canonical float.hex text")
        try:
            parsed_objective = float.fromhex(objective)
        except ValueError as exc:
            raise StreamingSFTError("objective_value_hex is not float.hex text") from exc
        if parsed_objective.hex() != objective or not math.isfinite(parsed_objective) or parsed_objective < 0:
            raise StreamingSFTError("objective_value_hex must be finite and non-negative")
        decisions = _require_positive_integer(self.decision_count, name="decision_count")
        _require_positive_integer(self.supervised_token_count, name="supervised_token_count")
        if self.released_graph_count != decisions:
            raise StreamingSFTError("every executed decision graph must be released")
        parameter_count = _require_positive_integer(
            self.parameter_count,
            name="parameter_count",
        )
        name_count = _require_positive_integer(
            self.registered_parameter_name_count,
            name="registered_parameter_name_count",
        )
        if name_count < parameter_count:
            raise StreamingSFTError("deduplicated parameter count exceeds registered names")
        if self.gradient_accumulation_dtype != "torch.float32":
            raise StreamingSFTError("streaming SFT accumulation dtype must be torch.float32")
        if self.gradient_accumulation_order != "canonical_plan_item_then_parameter_name":
            raise StreamingSFTError("streaming SFT accumulation order is invalid")
        contribution_casts = _require_positive_integer(
            self.fp32_contribution_cast_count,
            name="fp32_contribution_cast_count",
        )
        if contribution_casts < parameter_count:
            raise StreamingSFTError("too few FP32 contribution casts for full parameter reach")
        if self.final_gradient_cast_count != parameter_count:
            raise StreamingSFTError("final gradient cast count differs from parameter count")
        buffers = tuple(self.fp32_gradient_buffers)
        object.__setattr__(self, "fp32_gradient_buffers", buffers)
        gradients = tuple(self.gradients)
        object.__setattr__(self, "gradients", gradients)
        for label, records in (("FP32 buffer", buffers), ("final gradient", gradients)):
            if (
                len(records) != parameter_count
                or any(type(record) is not ParameterGradientRecord for record in records)
                or tuple(record.name for record in records)
                != tuple(sorted(record.name for record in records))
                or len({record.name for record in records}) != len(records)
            ):
                raise StreamingSFTError(f"{label} records must cover every canonical parameter exactly once")
        if tuple(record.name for record in buffers) != tuple(record.name for record in gradients):
            raise StreamingSFTError("FP32 and final gradient manifests address different parameters")
        for buffer, gradient in zip(buffers, gradients, strict=True):
            if (
                buffer.aliases != gradient.aliases
                or buffer.parameter_dtype != gradient.parameter_dtype
                or buffer.parameter_device != gradient.parameter_device
                or buffer.parameter_shape != gradient.parameter_shape
                or buffer.gradient_device != buffer.parameter_device
                or gradient.gradient_device != gradient.parameter_device
                or gradient.gradient_dtype != gradient.parameter_dtype
            ):
                raise StreamingSFTError(
                    "FP32 and final gradient manifests have inconsistent parameter metadata"
                )
        if any(record.gradient_dtype != "torch.float32" for record in buffers):
            raise StreamingSFTError("FP32 buffer manifest contains another dtype")
        expected_buffer_manifest = json_digest(
            [record.as_obj() for record in buffers],
            domain=_FP32_BUFFER_MANIFEST_DOMAIN,
        )
        if not hmac.compare_digest(
            expected_buffer_manifest,
            self.fp32_gradient_buffer_manifest_digest,
        ):
            raise StreamingSFTError("FP32 gradient-buffer manifest digest is inconsistent")
        expected_manifest = json_digest(
            [record.as_obj() for record in gradients],
            domain=_GRADIENT_MANIFEST_DOMAIN,
        )
        if not hmac.compare_digest(expected_manifest, self.gradient_manifest_digest):
            raise StreamingSFTError("gradient manifest digest is inconsistent")

    def as_obj(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "contract_id": self.contract_id,
            "authorizes_execution": self.authorizes_execution,
            "optimizer_step_performed": self.optimizer_step_performed,
            "plan_digest": self.plan_digest,
            "model_provenance_digest": self.model_provenance_digest,
            "provider_policy_state_digest": self.provider_policy_state_digest,
            "trainable_parameter_registry_digest": (self.trainable_parameter_registry_digest),
            "objective_value_hex": self.objective_value_hex,
            "decision_count": self.decision_count,
            "supervised_token_count": self.supervised_token_count,
            "released_graph_count": self.released_graph_count,
            "parameter_count": self.parameter_count,
            "registered_parameter_name_count": self.registered_parameter_name_count,
            "gradient_accumulation_dtype": self.gradient_accumulation_dtype,
            "gradient_accumulation_order": self.gradient_accumulation_order,
            "fp32_contribution_cast_count": self.fp32_contribution_cast_count,
            "final_gradient_cast_count": self.final_gradient_cast_count,
            "fp32_gradient_buffer_manifest_digest": (self.fp32_gradient_buffer_manifest_digest),
            "fp32_gradient_buffers": [record.as_obj() for record in self.fp32_gradient_buffers],
            "gradient_manifest_digest": self.gradient_manifest_digest,
            "gradients": [record.as_obj() for record in self.gradients],
        }

    @property
    def digest(self) -> str:
        return json_digest(self.as_obj(), domain=_EXECUTION_DIGEST_DOMAIN)

    def to_json(self) -> str:
        return dump_json({**self.as_obj(), "digest": self.digest})


def _validate_plan_nominality(
    verified_plan: _VerifiedStreamingSFTPlan,
) -> StreamingSFTPlan:
    if type(verified_plan) is not _VerifiedStreamingSFTPlan:
        raise StreamingSFTError("streaming SFT execution requires a freshly nominal verified plan")
    plan = verified_plan.plan
    if StreamingSFTPlan.from_json(plan.to_json()) != plan:
        raise StreamingSFTError("nominal streaming SFT plan is not structurally canonical")
    regenerated = _VerifiedStreamingSFTPlan._from_derived(plan)
    if not hmac.compare_digest(
        regenerated.verification_digest,
        verified_plan.verification_digest,
    ):
        raise StreamingSFTError("nominal streaming SFT plan verification digest changed")
    return plan


def execute_streaming_sft_backward(
    verified_plan: _VerifiedStreamingSFTPlan,
    sources: Sequence[ReferenceTrajectorySFTSource],
    tokenizer: ExactDecodeTokenizerProtocol,
    compiler: FragmentActionTokenCompiler,
    provider: AuthenticatedCausalLMProvider,
    *,
    graph_release_hook: GraphReleaseHook | None = None,
) -> StreamingSFTExecution:
    """Backpropagate one decision graph at a time without taking an optimizer step."""

    plan = _validate_plan_nominality(verified_plan)
    if not isinstance(tokenizer, ExactDecodeTokenizerProtocol):
        raise TypeError("tokenizer must implement ExactDecodeTokenizerProtocol")
    if type(compiler) is not FragmentActionTokenCompiler:
        raise TypeError("compiler must be a FragmentActionTokenCompiler")
    if type(provider) is not AuthenticatedCausalLMProvider:
        raise TypeError("provider must be an AuthenticatedCausalLMProvider")
    if graph_release_hook is not None and not callable(graph_release_hook):
        raise TypeError("graph_release_hook must be callable or None")
    compiler_manifest = compiler.manifest
    if (
        compiler_manifest.digest != plan.compiler_manifest_digest
        or compiler_manifest.tokenizer_manifest.digest != plan.tokenizer_binding_digest
    ):
        raise StreamingSFTError("runtime tokenizer/compiler differs from the SFT plan")
    before_model, before_policy = _provider_bindings(provider)
    if before_model != plan.model_provenance_digest or before_policy != plan.provider_policy_state_digest:
        raise StreamingSFTError("runtime provider differs from the SFT plan")

    maximums = {item.maximum_sequence_tokens for item in plan.items}
    if len(maximums) != 1:
        raise StreamingSFTError("nominal plan has inconsistent maximum sequence bounds")
    regenerated = build_streaming_sft_plan(
        sources,
        tokenizer,
        compiler,
        provider,
        maximum_sequence_tokens=next(iter(maximums)),
    )
    if (
        regenerated.plan.to_json() != plan.to_json()
        or regenerated.plan != plan
        or not hmac.compare_digest(
            regenerated.verification_digest,
            verified_plan.verification_digest,
        )
    ):
        raise StreamingSFTError("nominal plan differs from fresh authenticated reference sources")

    (
        parameters,
        trainable_parameter_registry_digest,
        registered_parameter_name_count,
    ) = _registered_parameters(provider.trainable_parameter_registry)
    _require_zero_starting_gradients(parameters)
    gradient_accumulator = _FP32GradientAccumulator(parameters)
    contributions: list[float] = []
    released = 0
    execution_started = False
    success = False
    try:
        execution_started = True
        for item in plan.items:
            verified = _rederive_plan_item(item, tokenizer, compiler)
            logits = provider.full_forward_logits(verified.example.input_ids)
            logits_reference = weakref.ref(logits)
            token_sum, token_count = _decision_token_sum(logits, verified)
            if token_count != item.supervised_token_count:
                raise StreamingSFTError("runtime decision token count differs from the plan")
            normalized = token_sum / plan.total_supervised_tokens
            contribution = float(normalized.detach().to(device="cpu", dtype=torch.float64).item())
            if not math.isfinite(contribution):
                raise StreamingSFTError("normalized decision objective is non-finite")
            gradient_accumulator.accumulate(normalized)
            _require_parameter_grads_none(parameters)
            contributions.append(contribution)

            del normalized
            del token_sum
            del logits
            gc.collect()
            if logits_reference() is not None:
                raise StreamingSFTError("a completed decision graph remained strongly referenced")
            released += 1
            if graph_release_hook is not None:
                graph_release_hook(item.digest, logits_reference)

        after_model, after_policy = _provider_bindings(provider)
        if after_model != before_model or after_policy != before_policy:
            raise StreamingSFTError("provider changed during streaming SFT backward")
        objective_value = math.fsum(contributions)
        if not math.isfinite(objective_value):
            raise StreamingSFTError("plan-wide streaming SFT objective is non-finite")
        fp32_gradient_buffers, gradients = gradient_accumulator.finalize()
        fp32_gradient_buffer_manifest_digest = json_digest(
            [record.as_obj() for record in fp32_gradient_buffers],
            domain=_FP32_BUFFER_MANIFEST_DOMAIN,
        )
        gradient_manifest_digest = json_digest(
            [record.as_obj() for record in gradients],
            domain=_GRADIENT_MANIFEST_DOMAIN,
        )
        execution = StreamingSFTExecution(
            plan_digest=plan.digest,
            model_provenance_digest=after_model,
            provider_policy_state_digest=after_policy,
            trainable_parameter_registry_digest=(trainable_parameter_registry_digest),
            objective_value_hex=objective_value.hex(),
            decision_count=len(plan.items),
            supervised_token_count=plan.total_supervised_tokens,
            released_graph_count=released,
            parameter_count=len(parameters),
            registered_parameter_name_count=registered_parameter_name_count,
            gradient_accumulation_dtype="torch.float32",
            gradient_accumulation_order="canonical_plan_item_then_parameter_name",
            fp32_contribution_cast_count=(gradient_accumulator.contribution_cast_count),
            final_gradient_cast_count=gradient_accumulator.final_cast_count,
            fp32_gradient_buffers=fp32_gradient_buffers,
            fp32_gradient_buffer_manifest_digest=(fp32_gradient_buffer_manifest_digest),
            gradients=gradients,
            gradient_manifest_digest=gradient_manifest_digest,
        )
        success = True
        return execution
    except Exception as exc:
        if isinstance(exc, (StreamingSFTError, TypeError)):
            raise
        if isinstance(
            exc,
            (AuthenticatedModelProviderError, DecisionEncodingError, RuntimeError, ValueError),
        ):
            raise StreamingSFTError("streaming SFT backward failed closed") from exc
        raise
    finally:
        if execution_started and not success:
            gradient_accumulator.clear_created_final_gradients()
        gradient_accumulator.clear_buffers()
