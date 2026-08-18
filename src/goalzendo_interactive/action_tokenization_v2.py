"""Boundary-safe fragment tokenization for teacher-forced G03 actions.

Version 1 of the constrained decoder required every textual extension to
preserve the tokenization of the complete prefix.  Ordinary BPE tokenizers do
not promise that property: a token can merge bytes on opposite sides of a
grammar-field boundary.  This additive contract instead tokenizes each
complete ``GrammarSegment`` extension independently.  Concatenating those
token sequences is valid when exact decoding reproduces the canonical action.

The resulting teacher-forced trace records the selected token and the exact
token-trie child mask at every position.  The selected IDs are therefore SFT
labels, while the same sparse masks define the differentiable distribution to
rescore for outcome RL.  No rule-by-classification product is materialized.
"""

from __future__ import annotations

import hmac
import threading
from collections.abc import Sequence
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Protocol, TypeAlias, TypeGuard, runtime_checkable

from ._json import CanonicalJSONError, dump_json, json_digest, load_json
from .action_language import (
    AnswerActionState,
    GrammarOption,
    GrammarSegment,
    InquiryActionState,
    action_language_digest,
    build_answer_action,
    build_inquiry_action,
)
from .actions import (
    Action,
    AnswerAction,
    InvalidActionError,
    ReadyAction,
    TestAction,
    parse_action,
    serialize_action,
)
from .rules import SYNTACTIC_RULE_COUNT
from .trajectory_encoding import ChatTokenizerProtocol

FRAGMENT_ACTION_TOKENIZATION_SCHEMA_VERSION = 2
FRAGMENT_ACTION_TOKENIZATION_CONTRACT_ID = "goalzendo-fragment-action-tokenization-v2"
TOKENIZER_BINDING_SCHEMA_VERSION = 1
TOKENIZER_BINDING_CONTRACT_ID = "goalzendo-exact-tokenizer-binding-v1"
REGISTERED_FRAGMENT_SPEC_COUNT = 18
_TRACE_DIGEST_DOMAIN = "goalzendo-interactive-fragment-action-token-trace-v2"
_SPEC_DIGEST_DOMAIN = "goalzendo-interactive-fragment-token-spec-v2"
_TOKENIZED_OUTPUT_DIGEST_DOMAIN = "goalzendo-interactive-fragment-tokenized-output-v2"
_TOKENIZER_BINDING_DIGEST_DOMAIN = "goalzendo-interactive-exact-tokenizer-binding-v1"
_MANIFEST_DIGEST_DOMAIN = "goalzendo-interactive-frozen-fragment-token-compiler-v2"
_RUNTIME_COUNTERS_DIGEST_DOMAIN = "goalzendo-interactive-fragment-token-runtime-v2"


@lru_cache(maxsize=1)
def _cached_action_language_digest() -> str:
    return action_language_digest()


class FragmentActionTokenizationError(ValueError):
    """Base class for a fail-closed fragment-tokenization contract error."""


class InvalidFragmentTokenError(FragmentActionTokenizationError):
    """Raised when a tokenizer emits an empty or invalid token sequence."""


class FragmentDecodeMismatchError(FragmentActionTokenizationError):
    """Raised when exact decoding does not reproduce the required text."""


class OptionTokenCollisionError(FragmentActionTokenizationError):
    """Raised when two complete option fragments are token-prefix ambiguous."""


class FragmentActionOverlengthError(FragmentActionTokenizationError):
    """Raised instead of truncating a fragment-tokenized action."""


class FragmentCompilerNotFrozenError(FragmentActionTokenizationError):
    """Raised when a semantic trace is requested before complete precompilation."""


class UnregisteredFragmentSpecError(FragmentActionTokenizationError):
    """Raised when a frozen compiler encounters a fragment outside its language."""


@runtime_checkable
class ExactDecodeTokenizerProtocol(ChatTokenizerProtocol, Protocol):
    """The existing chat-tokenizer surface plus an exact, cleanup-free decode."""

    def decode(
        self,
        token_ids: Sequence[int],
        *,
        skip_special_tokens: bool = False,
        clean_up_tokenization_spaces: bool = False,
    ) -> str: ...


@dataclass(frozen=True, slots=True)
class TokenizerBindingManifest:
    """Immutable, digest-bound identity of the exact tokenizer implementation."""

    repository_id: str
    revision: str
    tokenizer_json_sha256: str
    tokenizer_config_sha256: str
    chat_template_sha256: str
    backend_name: str
    backend_version: str
    vocabulary_size: int
    special_token_ids: tuple[tuple[str, int | None], ...]
    schema_version: int = TOKENIZER_BINDING_SCHEMA_VERSION
    contract_id: str = TOKENIZER_BINDING_CONTRACT_ID

    def __post_init__(self) -> None:
        if self.schema_version != TOKENIZER_BINDING_SCHEMA_VERSION:
            raise FragmentActionTokenizationError("unexpected tokenizer-binding schema version")
        if self.contract_id != TOKENIZER_BINDING_CONTRACT_ID:
            raise FragmentActionTokenizationError("unexpected tokenizer-binding contract id")
        _require_ascii(self.repository_id, name="repository_id")
        if any(character.isspace() for character in self.repository_id):
            raise FragmentActionTokenizationError("repository_id may not contain whitespace")
        if not _is_immutable_revision(self.revision):
            raise FragmentActionTokenizationError(
                "revision must be a lowercase 40- or 64-hex immutable commit"
            )
        for name in (
            "tokenizer_json_sha256",
            "tokenizer_config_sha256",
            "chat_template_sha256",
        ):
            if not _is_sha256(getattr(self, name)):
                raise FragmentActionTokenizationError(f"{name} must be a lowercase SHA-256")
        _require_ascii(self.backend_name, name="backend_name")
        _require_ascii(self.backend_version, name="backend_version")
        if (
            isinstance(self.vocabulary_size, bool)
            or not isinstance(self.vocabulary_size, int)
            or self.vocabulary_size < 1
        ):
            raise FragmentActionTokenizationError("vocabulary_size must be a positive integer")

        special_ids = tuple(self.special_token_ids)
        object.__setattr__(self, "special_token_ids", special_ids)
        if special_ids != tuple(sorted(special_ids, key=lambda item: item[0])):
            raise FragmentActionTokenizationError(
                "special_token_ids must be sorted by canonical token-role key"
            )
        keys = tuple(key for key, _ in special_ids)
        if len(set(keys)) != len(keys):
            raise FragmentActionTokenizationError("special_token_ids contains duplicate keys")
        required = {"bos", "eos", "pad", "unk"}
        if not required.issubset(keys):
            raise FragmentActionTokenizationError(
                "special_token_ids must include bos, eos, pad, and unk"
            )
        for key, token_id in special_ids:
            _require_ascii(key, name="special-token role")
            if key not in required and not key.startswith("additional:"):
                raise FragmentActionTokenizationError(
                    "non-core special-token keys must start with 'additional:'"
                )
            if key.startswith("additional:") and key == "additional:":
                raise FragmentActionTokenizationError(
                    "additional special-token keys must name their token"
                )
            if key.startswith("additional:") and token_id is None:
                raise FragmentActionTokenizationError(
                    "additional special-token keys must map to an integer id"
                )
            if token_id is not None and (
                isinstance(token_id, bool)
                or not isinstance(token_id, int)
                or not 0 <= token_id < self.vocabulary_size
            ):
                raise FragmentActionTokenizationError(
                    "special-token ids must be null or within the registered vocabulary"
                )

    def as_obj(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "contract_id": self.contract_id,
            "repository_id": self.repository_id,
            "revision": self.revision,
            "tokenizer_json_sha256": self.tokenizer_json_sha256,
            "tokenizer_config_sha256": self.tokenizer_config_sha256,
            "chat_template_sha256": self.chat_template_sha256,
            "backend": {
                "name": self.backend_name,
                "version": self.backend_version,
            },
            "vocabulary_size": self.vocabulary_size,
            "special_token_ids": dict(self.special_token_ids),
        }

    @property
    def digest(self) -> str:
        return json_digest(self.as_obj(), domain=_TOKENIZER_BINDING_DIGEST_DOMAIN)

    def to_json(self) -> str:
        return dump_json({**self.as_obj(), "digest": self.digest})

    @classmethod
    def from_obj(cls, value: object) -> TokenizerBindingManifest:
        obj = _require_exact_object(
            value,
            (
                "schema_version",
                "contract_id",
                "repository_id",
                "revision",
                "tokenizer_json_sha256",
                "tokenizer_config_sha256",
                "chat_template_sha256",
                "backend",
                "vocabulary_size",
                "special_token_ids",
                "digest",
            ),
            path="tokenizer_binding",
        )
        backend = _require_exact_object(
            obj["backend"], ("name", "version"), path="tokenizer_binding.backend"
        )
        raw_special_ids = obj["special_token_ids"]
        if type(raw_special_ids) is not dict or any(
            type(key) is not str for key in raw_special_ids
        ):
            raise FragmentActionTokenizationError(
                "tokenizer_binding.special_token_ids must be a JSON object"
            )
        result = cls(
            repository_id=_require_text(obj["repository_id"], path="repository_id"),
            revision=_require_text(obj["revision"], path="revision"),
            tokenizer_json_sha256=_require_text(
                obj["tokenizer_json_sha256"], path="tokenizer_json_sha256"
            ),
            tokenizer_config_sha256=_require_text(
                obj["tokenizer_config_sha256"], path="tokenizer_config_sha256"
            ),
            chat_template_sha256=_require_text(
                obj["chat_template_sha256"], path="chat_template_sha256"
            ),
            backend_name=_require_text(backend["name"], path="backend.name"),
            backend_version=_require_text(backend["version"], path="backend.version"),
            vocabulary_size=_require_integer(
                obj["vocabulary_size"], path="vocabulary_size", minimum=1
            ),
            special_token_ids=tuple(
                sorted(
                    (
                        key,
                        _require_optional_token_id(
                            raw_token_id, path=f"special_token_ids.{key}"
                        ),
                    )
                    for key, raw_token_id in raw_special_ids.items()
                )
            ),
            schema_version=_require_integer(
                obj["schema_version"], path="schema_version", minimum=1
            ),
            contract_id=_require_text(obj["contract_id"], path="contract_id"),
        )
        supplied_digest = _require_digest(obj["digest"], path="tokenizer_binding.digest")
        if not hmac.compare_digest(supplied_digest, result.digest):
            raise FragmentActionTokenizationError("tokenizer-binding digest check failed")
        return result

    @classmethod
    def from_json(cls, text: str) -> TokenizerBindingManifest:
        value = _load_canonical_json(text, path="tokenizer_binding")
        result = cls.from_obj(value)
        if not hmac.compare_digest(
            text.encode("utf-8"), result.to_json().encode("utf-8")
        ):
            raise FragmentActionTokenizationError(
                "tokenizer_binding JSON is not in the unique canonical byte representation"
            )
        return result


@dataclass(slots=True)
class _TrieNode:
    children: dict[int, _TrieNode] = field(default_factory=dict)
    option_key: str | None = None


@dataclass(frozen=True, slots=True)
class CompiledFragmentTrie:
    """One cached trie over standalone encodings of complete field extensions."""

    spec_digest: str
    tokenizer_manifest_digest: str
    vocabulary_size: int
    tokenized_output_digest: str
    prefix: str
    option_token_ids: tuple[tuple[str, str, tuple[int, ...]], ...]
    _root: _TrieNode = field(repr=False, compare=False)

    def __post_init__(self) -> None:
        if not _is_sha256(self.spec_digest):
            raise FragmentActionTokenizationError("spec_digest must be a lowercase SHA-256")
        if not _is_sha256(self.tokenizer_manifest_digest):
            raise FragmentActionTokenizationError(
                "tokenizer_manifest_digest must be a lowercase SHA-256"
            )
        if (
            isinstance(self.vocabulary_size, bool)
            or not isinstance(self.vocabulary_size, int)
            or self.vocabulary_size < 1
        ):
            raise FragmentActionTokenizationError("vocabulary_size must be positive")
        if not _is_sha256(self.tokenized_output_digest):
            raise FragmentActionTokenizationError(
                "tokenized_output_digest must be a lowercase SHA-256"
            )
        if type(self.prefix) is not str:
            raise FragmentActionTokenizationError("compiled fragment prefix must be text")
        options = tuple(self.option_token_ids)
        object.__setattr__(self, "option_token_ids", options)
        if not options:
            raise FragmentActionTokenizationError("compiled fragment trie cannot be empty")
        keys: list[str] = []
        texts: list[str] = []
        for key, text, token_ids in options:
            if type(key) is not str or not key or type(text) is not str or not text:
                raise FragmentActionTokenizationError("compiled fragment options must be nonempty")
            _require_token_ids(
                token_ids,
                allow_empty=False,
                vocabulary_size=self.vocabulary_size,
            )
            keys.append(key)
            texts.append(text)
        if len(set(keys)) != len(keys) or len(set(texts)) != len(texts):
            raise FragmentActionTokenizationError("compiled fragment options must be unique")
        expected_output_digest = _tokenized_output_digest(
            tokenizer_manifest_digest=self.tokenizer_manifest_digest,
            spec_digest=self.spec_digest,
            options=options,
        )
        if not hmac.compare_digest(self.tokenized_output_digest, expected_output_digest):
            raise FragmentActionTokenizationError("tokenized trie/output digest is inconsistent")

    @property
    def option_count(self) -> int:
        return len(self.option_token_ids)

    def tokens_for(self, option_key: str) -> tuple[int, ...]:
        for key, _, token_ids in self.option_token_ids:
            if key == option_key:
                return token_ids
        raise FragmentActionTokenizationError(f"unknown compiled option key: {option_key!r}")

    def text_for(self, option_key: str) -> str:
        for key, text, _ in self.option_token_ids:
            if key == option_key:
                return text
        raise FragmentActionTokenizationError(f"unknown compiled option key: {option_key!r}")


@dataclass(frozen=True, slots=True)
class FragmentTokenTrace:
    """The independently encoded token sequence selected for one grammar field."""

    fragment_index: int
    field_id: str
    selected_key: str
    text: str
    token_start: int
    token_ids: tuple[int, ...]
    grammar_spec_digest: str
    tokenized_output_digest: str

    def __post_init__(self) -> None:
        for name in ("fragment_index", "token_start"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise FragmentActionTokenizationError(f"{name} must be a non-negative integer")
        for name in ("field_id", "selected_key", "text"):
            value = getattr(self, name)
            if type(value) is not str or not value:
                raise FragmentActionTokenizationError(f"{name} must be nonempty text")
        token_ids = tuple(self.token_ids)
        object.__setattr__(self, "token_ids", token_ids)
        _require_token_ids(token_ids, allow_empty=False)
        if not _is_sha256(self.grammar_spec_digest):
            raise FragmentActionTokenizationError(
                "grammar_spec_digest must be a lowercase SHA-256"
            )
        if not _is_sha256(self.tokenized_output_digest):
            raise FragmentActionTokenizationError(
                "tokenized_output_digest must be a lowercase SHA-256"
            )

    @property
    def token_end(self) -> int:
        return self.token_start + len(self.token_ids)

    def as_obj(self) -> dict[str, object]:
        return {
            "fragment_index": self.fragment_index,
            "field_id": self.field_id,
            "selected_key": self.selected_key,
            "text": self.text,
            "token_start": self.token_start,
            "token_end": self.token_end,
            "token_ids": list(self.token_ids),
            "grammar_spec_digest": self.grammar_spec_digest,
            "tokenized_output_digest": self.tokenized_output_digest,
        }

    @classmethod
    def from_obj(cls, value: object) -> FragmentTokenTrace:
        obj = _require_exact_object(
            value,
            (
                "fragment_index",
                "field_id",
                "selected_key",
                "text",
                "token_start",
                "token_end",
                "token_ids",
                "grammar_spec_digest",
                "tokenized_output_digest",
            ),
            path="fragment",
        )
        result = cls(
            fragment_index=_require_integer(
                obj["fragment_index"], path="fragment.fragment_index", minimum=0
            ),
            field_id=_require_text(obj["field_id"], path="fragment.field_id"),
            selected_key=_require_text(
                obj["selected_key"], path="fragment.selected_key"
            ),
            text=_require_text(obj["text"], path="fragment.text"),
            token_start=_require_integer(
                obj["token_start"], path="fragment.token_start", minimum=0
            ),
            token_ids=_token_id_tuple(obj["token_ids"], path="fragment.token_ids"),
            grammar_spec_digest=_require_digest(
                obj["grammar_spec_digest"], path="fragment.grammar_spec_digest"
            ),
            tokenized_output_digest=_require_digest(
                obj["tokenized_output_digest"], path="fragment.tokenized_output_digest"
            ),
        )
        token_end = _require_integer(obj["token_end"], path="fragment.token_end", minimum=0)
        if token_end != result.token_end:
            raise FragmentActionTokenizationError("fragment.token_end is inconsistent")
        return result


@dataclass(frozen=True, slots=True)
class TeacherForcedTokenStep:
    """One selected action token and its exact legal trie-child mask."""

    action_token_index: int
    fragment_index: int
    fragment_token_index: int
    field_id: str
    selected_key: str
    selected_token_id: int
    allowed_token_ids: tuple[int, ...]

    def __post_init__(self) -> None:
        for name in ("action_token_index", "fragment_index", "fragment_token_index"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise FragmentActionTokenizationError(f"{name} must be a non-negative integer")
        for name in ("field_id", "selected_key"):
            value = getattr(self, name)
            if type(value) is not str or not value:
                raise FragmentActionTokenizationError(f"{name} must be nonempty text")
        if (
            isinstance(self.selected_token_id, bool)
            or not isinstance(self.selected_token_id, int)
            or self.selected_token_id < 0
        ):
            raise FragmentActionTokenizationError(
                "selected_token_id must be a non-negative integer"
            )
        allowed = tuple(self.allowed_token_ids)
        object.__setattr__(self, "allowed_token_ids", allowed)
        _require_token_ids(allowed, allow_empty=False)
        if allowed != tuple(sorted(set(allowed))):
            raise FragmentActionTokenizationError(
                "allowed_token_ids must be unique and increasing"
            )
        if self.selected_token_id not in allowed:
            raise FragmentActionTokenizationError("selected token is absent from its trie mask")

    def as_obj(self) -> dict[str, object]:
        return {
            "action_token_index": self.action_token_index,
            "fragment_index": self.fragment_index,
            "fragment_token_index": self.fragment_token_index,
            "field_id": self.field_id,
            "selected_key": self.selected_key,
            "selected_token_id": self.selected_token_id,
            "allowed_token_ids": list(self.allowed_token_ids),
        }

    @classmethod
    def from_obj(cls, value: object) -> TeacherForcedTokenStep:
        obj = _require_exact_object(
            value,
            (
                "action_token_index",
                "fragment_index",
                "fragment_token_index",
                "field_id",
                "selected_key",
                "selected_token_id",
                "allowed_token_ids",
            ),
            path="step",
        )
        return cls(
            action_token_index=_require_integer(
                obj["action_token_index"], path="step.action_token_index", minimum=0
            ),
            fragment_index=_require_integer(
                obj["fragment_index"], path="step.fragment_index", minimum=0
            ),
            fragment_token_index=_require_integer(
                obj["fragment_token_index"], path="step.fragment_token_index", minimum=0
            ),
            field_id=_require_text(obj["field_id"], path="step.field_id"),
            selected_key=_require_text(obj["selected_key"], path="step.selected_key"),
            selected_token_id=_require_integer(
                obj["selected_token_id"], path="step.selected_token_id", minimum=0
            ),
            allowed_token_ids=_token_id_tuple(
                obj["allowed_token_ids"], path="step.allowed_token_ids"
            ),
        )


@dataclass(frozen=True, slots=True)
class ActionTokenTrace:
    """Canonical teacher-forced labels and masks for one complete G03 action."""

    tokenizer_identifier: str
    compiler_manifest_digest: str
    maximum_action_tokens: int
    raw_action: str
    action_token_ids: tuple[int, ...]
    fragments: tuple[FragmentTokenTrace, ...]
    steps: tuple[TeacherForcedTokenStep, ...]
    schema_version: int = FRAGMENT_ACTION_TOKENIZATION_SCHEMA_VERSION
    contract_id: str = FRAGMENT_ACTION_TOKENIZATION_CONTRACT_ID
    grammar_digest: str = field(default_factory=_cached_action_language_digest)

    def __post_init__(self) -> None:
        if self.schema_version != FRAGMENT_ACTION_TOKENIZATION_SCHEMA_VERSION:
            raise FragmentActionTokenizationError("unexpected fragment-token schema version")
        if self.contract_id != FRAGMENT_ACTION_TOKENIZATION_CONTRACT_ID:
            raise FragmentActionTokenizationError("unexpected fragment-token contract id")
        if not _is_sha256(self.tokenizer_identifier):
            raise FragmentActionTokenizationError(
                "tokenizer_identifier must be a tokenizer-manifest SHA-256"
            )
        if not _is_sha256(self.compiler_manifest_digest):
            raise FragmentActionTokenizationError(
                "compiler_manifest_digest must be a lowercase SHA-256"
            )
        if (
            isinstance(self.maximum_action_tokens, bool)
            or not isinstance(self.maximum_action_tokens, int)
            or self.maximum_action_tokens < 1
        ):
            raise FragmentActionTokenizationError(
                "maximum_action_tokens must be a positive integer"
            )
        if type(self.raw_action) is not str or not self.raw_action:
            raise FragmentActionTokenizationError("raw_action must be nonempty text")
        if not _is_sha256(self.grammar_digest):
            raise FragmentActionTokenizationError("grammar_digest must be a lowercase SHA-256")

        action_ids = tuple(self.action_token_ids)
        fragments = tuple(self.fragments)
        steps = tuple(self.steps)
        object.__setattr__(self, "action_token_ids", action_ids)
        object.__setattr__(self, "fragments", fragments)
        object.__setattr__(self, "steps", steps)
        _require_token_ids(action_ids, allow_empty=False)
        if len(action_ids) > self.maximum_action_tokens:
            raise FragmentActionOverlengthError(
                "action token trace exceeds maximum_action_tokens; truncation is forbidden"
            )
        if not fragments or any(type(fragment) is not FragmentTokenTrace for fragment in fragments):
            raise FragmentActionTokenizationError("trace must contain typed fragment records")
        if not steps or any(type(step) is not TeacherForcedTokenStep for step in steps):
            raise FragmentActionTokenizationError("trace must contain typed token steps")

        expected_start = 0
        flattened_ids: list[int] = []
        flattened_text: list[str] = []
        for fragment_index, fragment in enumerate(fragments):
            if fragment.fragment_index != fragment_index or fragment.token_start != expected_start:
                raise FragmentActionTokenizationError(
                    "fragment indices and token spans must be contiguous"
                )
            flattened_ids.extend(fragment.token_ids)
            flattened_text.append(fragment.text)
            expected_start = fragment.token_end
        if tuple(flattened_ids) != action_ids:
            raise FragmentActionTokenizationError(
                "concatenated fragment token ids differ from action_token_ids"
            )
        if "".join(flattened_text) != self.raw_action:
            raise FragmentActionTokenizationError(
                "concatenated fragment text differs from the canonical action"
            )
        if len(steps) != len(action_ids):
            raise FragmentActionTokenizationError("trace requires exactly one step per action token")
        for token_index, step in enumerate(steps):
            if step.action_token_index != token_index:
                raise FragmentActionTokenizationError("token-step indices must be contiguous")
            if step.selected_token_id != action_ids[token_index]:
                raise FragmentActionTokenizationError(
                    "token step does not select the recorded action token"
                )
            if step.fragment_index >= len(fragments):
                raise FragmentActionTokenizationError(
                    "token step refers to an absent fragment"
                )
            fragment = fragments[step.fragment_index]
            if step.field_id != fragment.field_id or step.selected_key != fragment.selected_key:
                raise FragmentActionTokenizationError(
                    "token-step grammar metadata differs from its fragment"
                )
            if not fragment.token_start <= token_index < fragment.token_end:
                raise FragmentActionTokenizationError(
                    "token step lies outside its recorded fragment span"
                )
            if step.fragment_token_index != token_index - fragment.token_start:
                raise FragmentActionTokenizationError(
                    "fragment-local token-step index is inconsistent"
                )
            if fragment.token_ids[step.fragment_token_index] != step.selected_token_id:
                raise FragmentActionTokenizationError(
                    "token step differs from its fragment-local selected token"
                )
        try:
            parsed = parse_action(self.raw_action)
        except InvalidActionError as exc:
            raise FragmentActionTokenizationError(
                "raw_action is not canonical G03 action JSON"
            ) from exc
        if serialize_action(parsed) != self.raw_action:
            raise FragmentActionTokenizationError("raw_action is not canonical G03 action JSON")

    @property
    def sft_labels(self) -> tuple[int, ...]:
        """The exact teacher-forced action labels, excluding all prompt tokens."""

        return self.action_token_ids

    @property
    def allowed_token_ids(self) -> tuple[tuple[int, ...], ...]:
        """Sparse masks aligned one-for-one with :attr:`sft_labels`."""

        return tuple(step.allowed_token_ids for step in self.steps)

    def as_obj(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "contract_id": self.contract_id,
            "tokenizer_identifier": self.tokenizer_identifier,
            "compiler_manifest_digest": self.compiler_manifest_digest,
            "maximum_action_tokens": self.maximum_action_tokens,
            "grammar_digest": self.grammar_digest,
            "raw_action": self.raw_action,
            "action_token_ids": list(self.action_token_ids),
            "fragments": [fragment.as_obj() for fragment in self.fragments],
            "steps": [step.as_obj() for step in self.steps],
        }

    @property
    def digest(self) -> str:
        return json_digest(self.as_obj(), domain=_TRACE_DIGEST_DOMAIN)

    def to_json(self) -> str:
        return dump_json({**self.as_obj(), "digest": self.digest})

    @classmethod
    def from_obj(cls, value: object) -> ActionTokenTrace:
        """Parse a strict JSON-shaped trace and verify its supplied digest."""

        obj = _require_exact_object(
            value,
            (
                "schema_version",
                "contract_id",
                "tokenizer_identifier",
                "compiler_manifest_digest",
                "maximum_action_tokens",
                "grammar_digest",
                "raw_action",
                "action_token_ids",
                "fragments",
                "steps",
                "digest",
            ),
            path="action_token_trace",
        )
        raw_fragments = _require_list(obj["fragments"], path="action_token_trace.fragments")
        raw_steps = _require_list(obj["steps"], path="action_token_trace.steps")
        result = cls(
            tokenizer_identifier=_require_digest(
                obj["tokenizer_identifier"], path="action_token_trace.tokenizer_identifier"
            ),
            compiler_manifest_digest=_require_digest(
                obj["compiler_manifest_digest"],
                path="action_token_trace.compiler_manifest_digest",
            ),
            maximum_action_tokens=_require_integer(
                obj["maximum_action_tokens"],
                path="action_token_trace.maximum_action_tokens",
                minimum=1,
            ),
            raw_action=_require_text(
                obj["raw_action"], path="action_token_trace.raw_action"
            ),
            action_token_ids=_token_id_tuple(
                obj["action_token_ids"], path="action_token_trace.action_token_ids"
            ),
            fragments=tuple(FragmentTokenTrace.from_obj(item) for item in raw_fragments),
            steps=tuple(TeacherForcedTokenStep.from_obj(item) for item in raw_steps),
            schema_version=_require_integer(
                obj["schema_version"], path="action_token_trace.schema_version", minimum=1
            ),
            contract_id=_require_text(
                obj["contract_id"], path="action_token_trace.contract_id"
            ),
            grammar_digest=_require_digest(
                obj["grammar_digest"], path="action_token_trace.grammar_digest"
            ),
        )
        supplied_digest = _require_digest(
            obj["digest"], path="action_token_trace.digest"
        )
        if not hmac.compare_digest(supplied_digest, result.digest):
            raise FragmentActionTokenizationError("action-token trace digest check failed")
        return result

    @classmethod
    def from_json(cls, text: str) -> ActionTokenTrace:
        """Parse only the unique compact JSON spelling emitted by :meth:`to_json`."""

        value = _load_canonical_json(text, path="action_token_trace")
        result = cls.from_obj(value)
        if not hmac.compare_digest(
            text.encode("utf-8"), result.to_json().encode("utf-8")
        ):
            raise FragmentActionTokenizationError(
                "action_token_trace JSON is not in the unique canonical byte representation"
            )
        return result


@dataclass(frozen=True, slots=True)
class FragmentCompilerManifest:
    """Frozen semantics of the completely precompiled registered language."""

    tokenizer_manifest: TokenizerBindingManifest
    maximum_action_tokens: int
    registered_entries: tuple[tuple[str, str, int], ...]
    answer_rule_label_cartesian_product_count: int = 0

    def __post_init__(self) -> None:
        if type(self.tokenizer_manifest) is not TokenizerBindingManifest:
            raise FragmentActionTokenizationError(
                "tokenizer_manifest must be a TokenizerBindingManifest"
            )
        if (
            isinstance(self.maximum_action_tokens, bool)
            or not isinstance(self.maximum_action_tokens, int)
            or self.maximum_action_tokens < 1
        ):
            raise FragmentActionTokenizationError("maximum_action_tokens must be positive")
        entries = tuple(self.registered_entries)
        object.__setattr__(self, "registered_entries", entries)
        if entries != tuple(sorted(entries)):
            raise FragmentActionTokenizationError("registered entries must be sorted")
        if len(entries) != REGISTERED_FRAGMENT_SPEC_COUNT:
            raise FragmentActionTokenizationError(
                "frozen manifest must contain every registered unique fragment spec"
            )
        if len({spec_digest for spec_digest, _, _ in entries}) != len(entries):
            raise FragmentActionTokenizationError("registered spec digests must be unique")
        if any(
            not _is_sha256(spec_digest)
            or not _is_sha256(output_digest)
            or isinstance(option_count, bool)
            or not isinstance(option_count, int)
            or option_count < 1
            for spec_digest, output_digest, option_count in entries
        ):
            raise FragmentActionTokenizationError("registered manifest entry is invalid")
        if self.answer_rule_label_cartesian_product_count != 0:
            raise FragmentActionTokenizationError(
                "rule-by-classification products are forbidden"
            )

    @property
    def tokenizer_identifier(self) -> str:
        return self.tokenizer_manifest.digest

    def as_obj(self) -> dict[str, object]:
        return {
            "schema_version": FRAGMENT_ACTION_TOKENIZATION_SCHEMA_VERSION,
            "contract_id": FRAGMENT_ACTION_TOKENIZATION_CONTRACT_ID,
            "contract_manifest_digest": json_digest(
                fragment_token_contract_manifest(),
                domain="goalzendo-interactive-fragment-token-contract-manifest-v2",
            ),
            "tokenizer_manifest": {
                **self.tokenizer_manifest.as_obj(),
                "digest": self.tokenizer_manifest.digest,
            },
            "tokenizer_identifier": self.tokenizer_manifest.digest,
            "maximum_action_tokens": self.maximum_action_tokens,
            "registered_fragment_spec_count": len(self.registered_entries),
            "registered_option_sequence_count": sum(
                option_count for _, _, option_count in self.registered_entries
            ),
            "registered_entries": [
                {
                    "spec_digest": spec_digest,
                    "tokenized_output_digest": output_digest,
                    "option_count": option_count,
                }
                for spec_digest, output_digest, option_count in self.registered_entries
            ],
            "answer_rule_label_cartesian_product_count": (
                self.answer_rule_label_cartesian_product_count
            ),
        }

    @property
    def digest(self) -> str:
        return json_digest(self.as_obj(), domain=_MANIFEST_DIGEST_DOMAIN)


@dataclass(frozen=True, slots=True)
class FragmentCompilerRuntimeCounters:
    """Point-in-time operational counters excluded from semantic provenance."""

    compiler_manifest_digest: str | None
    frozen: bool
    successful_trace_count: int
    cache_hit_count: int
    trie_compile_count: int
    option_sequence_compile_count: int
    rule_trie_compile_count: int
    rule_option_sequence_count: int
    cache_entries: tuple[tuple[str, str, int], ...]
    answer_rule_label_cartesian_product_count: int = 0

    def __post_init__(self) -> None:
        if self.compiler_manifest_digest is not None and not _is_sha256(
            self.compiler_manifest_digest
        ):
            raise FragmentActionTokenizationError(
                "compiler_manifest_digest must be null or a lowercase SHA-256"
            )
        if type(self.frozen) is not bool:
            raise FragmentActionTokenizationError("frozen must be Boolean")
        for name in (
            "successful_trace_count",
            "cache_hit_count",
            "trie_compile_count",
            "option_sequence_compile_count",
            "rule_trie_compile_count",
            "rule_option_sequence_count",
            "answer_rule_label_cartesian_product_count",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise FragmentActionTokenizationError(f"{name} must be non-negative")
        entries = tuple(self.cache_entries)
        object.__setattr__(self, "cache_entries", entries)
        if entries != tuple(sorted(entries)):
            raise FragmentActionTokenizationError("runtime cache entries must be sorted")
        if any(
            not _is_sha256(spec_digest)
            or not _is_sha256(output_digest)
            or isinstance(option_count, bool)
            or not isinstance(option_count, int)
            or option_count < 1
            for spec_digest, output_digest, option_count in entries
        ):
            raise FragmentActionTokenizationError("runtime cache entry is invalid")
        if self.trie_compile_count != len(entries):
            raise FragmentActionTokenizationError("each cached trie must compile exactly once")
        if self.option_sequence_compile_count != sum(count for _, _, count in entries):
            raise FragmentActionTokenizationError(
                "runtime option-sequence count is inconsistent"
            )
        if self.rule_trie_compile_count not in {0, 1}:
            raise FragmentActionTokenizationError("the rule trie may compile at most once")
        expected_rule_options = SYNTACTIC_RULE_COUNT if self.rule_trie_compile_count else 0
        if self.rule_option_sequence_count != expected_rule_options:
            raise FragmentActionTokenizationError("rule option-sequence count is inconsistent")
        if self.answer_rule_label_cartesian_product_count != 0:
            raise FragmentActionTokenizationError(
                "rule-by-classification products are forbidden"
            )
        if self.frozen != (self.compiler_manifest_digest is not None):
            raise FragmentActionTokenizationError(
                "runtime frozen flag and semantic-manifest digest disagree"
            )

    def as_obj(self) -> dict[str, object]:
        return {
            "schema_version": FRAGMENT_ACTION_TOKENIZATION_SCHEMA_VERSION,
            "contract_id": FRAGMENT_ACTION_TOKENIZATION_CONTRACT_ID,
            "compiler_manifest_digest": self.compiler_manifest_digest,
            "frozen": self.frozen,
            "successful_trace_count": self.successful_trace_count,
            "compile_counts": {
                "cache_entries": len(self.cache_entries),
                "cache_hits": self.cache_hit_count,
                "trie_compiles": self.trie_compile_count,
                "option_sequences": self.option_sequence_compile_count,
                "rule_tries": self.rule_trie_compile_count,
                "rule_option_sequences": self.rule_option_sequence_count,
                "answer_rule_label_cartesian_products": (
                    self.answer_rule_label_cartesian_product_count
                ),
            },
            "cache_entries": [
                {
                    "spec_digest": spec_digest,
                    "tokenized_output_digest": output_digest,
                    "option_count": count,
                }
                for spec_digest, output_digest, count in self.cache_entries
            ],
        }

    @property
    def digest(self) -> str:
        return json_digest(self.as_obj(), domain=_RUNTIME_COUNTERS_DIGEST_DOMAIN)


_FragmentState: TypeAlias = InquiryActionState | AnswerActionState
_CacheKey: TypeAlias = tuple[str, tuple[tuple[str, str], ...]]


class FragmentActionTokenCompiler:
    """Compile and cache fragment tries, then emit exact teacher-forced traces."""

    def __init__(
        self,
        tokenizer: ExactDecodeTokenizerProtocol,
        *,
        tokenizer_manifest: TokenizerBindingManifest,
        maximum_action_tokens: int = 2_048,
    ) -> None:
        if not isinstance(tokenizer, ExactDecodeTokenizerProtocol):
            raise TypeError("tokenizer does not implement ExactDecodeTokenizerProtocol")
        if type(tokenizer_manifest) is not TokenizerBindingManifest:
            raise TypeError("tokenizer_manifest must be a TokenizerBindingManifest")
        if (
            isinstance(maximum_action_tokens, bool)
            or not isinstance(maximum_action_tokens, int)
            or maximum_action_tokens < 1
        ):
            raise ValueError("maximum_action_tokens must be a positive integer")
        self._tokenizer = tokenizer
        self._tokenizer_manifest = tokenizer_manifest
        self._tokenizer_identifier = tokenizer_manifest.digest
        self._maximum_action_tokens = maximum_action_tokens
        self._cache: dict[_CacheKey, CompiledFragmentTrie] = {}
        self._registered_cache_keys: frozenset[_CacheKey] | None = None
        self._frozen_manifest: FragmentCompilerManifest | None = None
        self._cache_hits = 0
        self._trie_compile_count = 0
        self._option_sequence_compile_count = 0
        self._rule_trie_compile_count = 0
        self._rule_option_sequence_count = 0
        self._successful_trace_count = 0
        self._lock = threading.RLock()

    def _encode_exact(self, text: str) -> tuple[int, ...]:
        if type(text) is not str or not text:
            raise InvalidFragmentTokenError("a complete grammar fragment cannot be empty")
        try:
            raw_ids = self._tokenizer.encode(text, add_special_tokens=False)
            token_ids = tuple(raw_ids)
        except (TypeError, ValueError, OverflowError) as exc:
            raise InvalidFragmentTokenError("tokenizer could not encode a grammar fragment") from exc
        _require_token_ids(
            token_ids,
            allow_empty=False,
            vocabulary_size=self._tokenizer_manifest.vocabulary_size,
        )
        decoded = self._decode_exact(token_ids)
        if decoded != text:
            raise FragmentDecodeMismatchError(
                "standalone fragment decode differs from its exact grammar text"
            )
        return token_ids

    def _decode_exact(self, token_ids: Sequence[int]) -> str:
        values = tuple(token_ids)
        _require_token_ids(
            values,
            allow_empty=False,
            vocabulary_size=self._tokenizer_manifest.vocabulary_size,
        )
        try:
            decoded = self._tokenizer.decode(
                values,
                skip_special_tokens=False,
                clean_up_tokenization_spaces=False,
            )
        except (TypeError, ValueError, OverflowError) as exc:
            raise FragmentDecodeMismatchError("tokenizer could not exactly decode token ids") from exc
        if type(decoded) is not str:
            raise FragmentDecodeMismatchError("tokenizer decode must return exact text")
        return decoded

    def compile_segment(self, segment: GrammarSegment) -> CompiledFragmentTrie:
        """Compile one complete-extension option trie or reuse its exact cache entry."""

        if type(segment) is not GrammarSegment:
            raise TypeError("compile_segment requires a GrammarSegment")
        option_specs = tuple((option.key, option.text) for option in segment.options)
        if segment.field_id == "rule":
            canonical_rule_segment = AnswerActionState(1).next_segment()
            if (
                segment.prefix != canonical_rule_segment.prefix
                or segment.options != canonical_rule_segment.options
            ):
                raise FragmentActionTokenizationError(
                    "rule segment does not expose the registered syntactic grammar"
                )
        key: _CacheKey = (segment.prefix, option_specs)
        with self._lock:
            if (
                self._registered_cache_keys is not None
                and key not in self._registered_cache_keys
            ):
                raise UnregisteredFragmentSpecError(
                    "frozen compiler rejected an unregistered fragment specification"
                )
            cached = self._cache.get(key)
            if cached is not None:
                self._cache_hits += 1
                if segment.field_id == "rule" and self._rule_trie_compile_count == 0:
                    self._rule_trie_compile_count = 1
                    self._rule_option_sequence_count = cached.option_count
                return cached

            spec_digest = _segment_spec_digest(segment)
            encoded_options = tuple(
                (
                    option_key,
                    segment.prefix + option_text,
                    self._encode_exact(segment.prefix + option_text),
                )
                for option_key, option_text in option_specs
            )
            root = _build_option_trie(encoded_options)
            tokenized_output_digest = _tokenized_output_digest(
                tokenizer_manifest_digest=self._tokenizer_identifier,
                spec_digest=spec_digest,
                options=encoded_options,
            )
            compiled = CompiledFragmentTrie(
                spec_digest=spec_digest,
                tokenizer_manifest_digest=self._tokenizer_identifier,
                vocabulary_size=self._tokenizer_manifest.vocabulary_size,
                tokenized_output_digest=tokenized_output_digest,
                prefix=segment.prefix,
                option_token_ids=encoded_options,
                _root=root,
            )
            self._cache[key] = compiled
            self._trie_compile_count += 1
            self._option_sequence_compile_count += len(encoded_options)
            if segment.field_id == "rule":
                self._rule_trie_compile_count += 1
                self._rule_option_sequence_count += len(encoded_options)
            return compiled

    def freeze_registered_language(self) -> FragmentCompilerManifest:
        """Precompile every registered unique spec and freeze semantic provenance."""

        with self._lock:
            if self._frozen_manifest is not None:
                return self._frozen_manifest
            registered_segments = _registered_fragment_segments()
            registered_by_key = {
                _segment_cache_key(segment): segment for segment in registered_segments
            }
            if len(registered_by_key) != REGISTERED_FRAGMENT_SPEC_COUNT:
                raise FragmentActionTokenizationError(
                    "registered fragment enumeration is incomplete or contains duplicates"
                )
            unexpected = set(self._cache).difference(registered_by_key)
            if unexpected:
                raise UnregisteredFragmentSpecError(
                    "cannot freeze a compiler whose cache contains unregistered specs"
                )
            for segment in registered_segments:
                self.compile_segment(segment)
            if set(self._cache) != set(registered_by_key):
                raise FragmentActionTokenizationError(
                    "full registered fragment precompilation did not complete"
                )
            entries = tuple(
                sorted(
                    (
                        compiled.spec_digest,
                        compiled.tokenized_output_digest,
                        compiled.option_count,
                    )
                    for compiled in self._cache.values()
                )
            )
            manifest = FragmentCompilerManifest(
                tokenizer_manifest=self._tokenizer_manifest,
                maximum_action_tokens=self._maximum_action_tokens,
                registered_entries=entries,
            )
            self._registered_cache_keys = frozenset(registered_by_key)
            self._frozen_manifest = manifest
            return manifest

    def _append_fragment(
        self,
        *,
        compiled: CompiledFragmentTrie,
        field_id: str,
        selected_key: str,
        fragments: list[FragmentTokenTrace],
        steps: list[TeacherForcedTokenStep],
        action_ids: list[int],
    ) -> None:
        selected_ids = compiled.tokens_for(selected_key)
        fragment_index = len(fragments)
        token_start = len(action_ids)
        if token_start + len(selected_ids) > self._maximum_action_tokens:
            raise FragmentActionOverlengthError(
                f"action exceeds maximum_action_tokens={self._maximum_action_tokens}; "
                "truncation is forbidden"
            )
        node = compiled._root
        for fragment_token_index, selected_token in enumerate(selected_ids):
            allowed = tuple(sorted(node.children))
            if not allowed or selected_token not in node.children:
                raise FragmentActionTokenizationError(
                    "selected option path is absent from its compiled trie"
                )
            action_token_index = len(action_ids)
            steps.append(
                TeacherForcedTokenStep(
                    action_token_index=action_token_index,
                    fragment_index=fragment_index,
                    fragment_token_index=fragment_token_index,
                    field_id=field_id,
                    selected_key=selected_key,
                    selected_token_id=selected_token,
                    allowed_token_ids=allowed,
                )
            )
            action_ids.append(selected_token)
            node = node.children[selected_token]
        if node.option_key != selected_key or node.children:
            raise FragmentActionTokenizationError(
                "selected option did not finish at an exact trie leaf"
            )
        fragments.append(
            FragmentTokenTrace(
                fragment_index=fragment_index,
                field_id=field_id,
                selected_key=selected_key,
                text=compiled.text_for(selected_key),
                token_start=token_start,
                token_ids=selected_ids,
                grammar_spec_digest=compiled.spec_digest,
                tokenized_output_digest=compiled.tokenized_output_digest,
            )
        )

    def trace_action(self, action: Action) -> ActionTokenTrace:
        """Build exact SFT labels and legal masks for a canonical action AST."""

        if self._frozen_manifest is None:
            raise FragmentCompilerNotFrozenError(
                "freeze_registered_language() must complete before tracing actions"
            )
        if isinstance(action, (ReadyAction, TestAction)):
            selections = build_inquiry_action(action).selections
            state: _FragmentState = InquiryActionState()
        elif isinstance(action, AnswerAction):
            selections = build_answer_action(action).selections
            state = AnswerActionState(len(action.classifications))
        else:
            raise TypeError("trace_action requires ReadyAction, TestAction, or AnswerAction")

        fragments: list[FragmentTokenTrace] = []
        steps: list[TeacherForcedTokenStep] = []
        action_ids: list[int] = []
        for expected_field, selected_key in selections:
            segment = state.next_segment()
            if segment.field_id != expected_field:
                raise FragmentActionTokenizationError(
                    "canonical action path differs from the live grammar state"
                )
            compiled = self.compile_segment(segment)
            self._append_fragment(
                compiled=compiled,
                field_id=segment.field_id,
                selected_key=selected_key,
                fragments=fragments,
                steps=steps,
                action_ids=action_ids,
            )
            state = state.choose(selected_key)

        completion = GrammarSegment(
            "$completion",
            "",
            (GrammarOption("completion", state.completion_suffix),),
        )
        self._append_fragment(
            compiled=self.compile_segment(completion),
            field_id=completion.field_id,
            selected_key="completion",
            fragments=fragments,
            steps=steps,
            action_ids=action_ids,
        )
        raw_action = state.text
        if raw_action != serialize_action(action):
            raise FragmentActionTokenizationError(
                "grammar state did not reproduce the canonical action"
            )
        decoded_action = self._decode_exact(action_ids)
        if decoded_action != raw_action:
            raise FragmentDecodeMismatchError(
                "decode(concatenated fragment token ids) differs from canonical raw action"
            )
        trace = ActionTokenTrace(
            tokenizer_identifier=self._tokenizer_identifier,
            compiler_manifest_digest=self._frozen_manifest.digest,
            maximum_action_tokens=self._maximum_action_tokens,
            raw_action=raw_action,
            action_token_ids=tuple(action_ids),
            fragments=tuple(fragments),
            steps=tuple(steps),
        )
        with self._lock:
            self._successful_trace_count += 1
        return trace

    @property
    def manifest(self) -> FragmentCompilerManifest:
        """Return frozen semantics, never a history-dependent partial snapshot."""

        with self._lock:
            if self._frozen_manifest is None:
                raise FragmentCompilerNotFrozenError(
                    "compiler manifest is unavailable before full registered freeze"
                )
            return self._frozen_manifest

    @property
    def runtime_counters(self) -> FragmentCompilerRuntimeCounters:
        """Return mutable operational history kept outside the semantic digest."""

        with self._lock:
            entries = tuple(
                sorted(
                    (
                        compiled.spec_digest,
                        compiled.tokenized_output_digest,
                        compiled.option_count,
                    )
                    for compiled in self._cache.values()
                )
            )
            return FragmentCompilerRuntimeCounters(
                compiler_manifest_digest=(
                    None
                    if self._frozen_manifest is None
                    else self._frozen_manifest.digest
                ),
                frozen=self._frozen_manifest is not None,
                successful_trace_count=self._successful_trace_count,
                cache_hit_count=self._cache_hits,
                trie_compile_count=self._trie_compile_count,
                option_sequence_compile_count=self._option_sequence_compile_count,
                rule_trie_compile_count=self._rule_trie_compile_count,
                rule_option_sequence_count=self._rule_option_sequence_count,
                cache_entries=entries,
            )

    def verify_trace(self, trace: ActionTokenTrace) -> ActionTokenTrace:
        """Recompile a trace and compare its complete canonical bytes exactly."""

        if type(trace) is not ActionTokenTrace:
            raise TypeError("trace must be an ActionTokenTrace")
        manifest = self.manifest
        if trace.tokenizer_identifier != self._tokenizer_identifier:
            raise FragmentActionTokenizationError(
                "trace is bound to a different tokenizer manifest"
            )
        if trace.compiler_manifest_digest != manifest.digest:
            raise FragmentActionTokenizationError(
                "trace is bound to a different frozen compiler manifest"
            )
        if trace.maximum_action_tokens != self._maximum_action_tokens:
            raise FragmentActionTokenizationError(
                "trace maximum_action_tokens differs from the frozen compiler"
            )
        _require_trace_vocabulary_bound(
            trace, vocabulary_size=self._tokenizer_manifest.vocabulary_size
        )
        try:
            action = parse_action(trace.raw_action)
        except InvalidActionError as exc:
            raise FragmentActionTokenizationError(
                "trace action no longer parses canonically"
            ) from exc
        rebuilt = self.trace_action(action)
        if rebuilt.to_json().encode("ascii") != trace.to_json().encode("ascii"):
            raise FragmentActionTokenizationError(
                "trace differs from exact tokenizer/compiler regeneration"
            )
        return trace

    def verified_trace_from_json(self, text: str) -> ActionTokenTrace:
        """Strictly parse, digest-check, recompile, and byte-verify one trace."""

        return self.verify_trace(ActionTokenTrace.from_json(text))


def _build_option_trie(
    encoded_options: Sequence[tuple[str, str, tuple[int, ...]]],
) -> _TrieNode:
    root = _TrieNode()
    seen_keys: set[str] = set()
    for option_key, _, token_ids in encoded_options:
        if option_key in seen_keys:
            raise OptionTokenCollisionError("fragment option keys must be unique")
        seen_keys.add(option_key)
        node = root
        for token_id in token_ids:
            if node.option_key is not None:
                raise OptionTokenCollisionError(
                    "one complete option token sequence prefixes another"
                )
            node = node.children.setdefault(token_id, _TrieNode())
        if node.option_key is not None:
            raise OptionTokenCollisionError(
                "two complete options have the same token sequence"
            )
        if node.children:
            raise OptionTokenCollisionError(
                "one complete option token sequence prefixes another"
            )
        node.option_key = option_key
    return root


def _require_token_ids(
    token_ids: Sequence[int],
    *,
    allow_empty: bool,
    vocabulary_size: int | None = None,
) -> None:
    values = tuple(token_ids)
    if not allow_empty and not values:
        raise InvalidFragmentTokenError("nonempty grammar text emitted no token ids")
    if any(
        isinstance(token_id, bool)
        or not isinstance(token_id, int)
        or token_id < 0
        for token_id in values
    ):
        raise InvalidFragmentTokenError(
            "token ids must be non-negative integers and may not be Booleans"
        )
    if vocabulary_size is not None and any(
        token_id >= vocabulary_size for token_id in values
    ):
        raise InvalidFragmentTokenError(
            "token id lies outside the registered tokenizer vocabulary"
        )


def _require_ascii(value: object, *, name: str) -> str:
    if type(value) is not str or not value or not value.isascii():
        raise FragmentActionTokenizationError(f"{name} must be nonempty ASCII text")
    return value


def _require_text(value: object, *, path: str) -> str:
    if type(value) is not str or not value:
        raise FragmentActionTokenizationError(f"{path} must be nonempty text")
    return value


def _require_integer(value: object, *, path: str, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise FragmentActionTokenizationError(
            f"{path} must be an integer greater than or equal to {minimum}"
        )
    return value


def _require_optional_token_id(value: object, *, path: str) -> int | None:
    if value is None:
        return None
    return _require_integer(value, path=path, minimum=0)


def _require_digest(value: object, *, path: str) -> str:
    if not _is_sha256(value):
        raise FragmentActionTokenizationError(f"{path} must be a lowercase SHA-256")
    return value


def _require_exact_object(
    value: object,
    expected_keys: Sequence[str],
    *,
    path: str,
) -> dict[str, object]:
    if type(value) is not dict or any(type(key) is not str for key in value):
        raise FragmentActionTokenizationError(f"{path} must be a JSON object")
    result = {key: item for key, item in value.items()}
    expected = set(expected_keys)
    actual = set(result)
    if actual != expected:
        missing = sorted(expected.difference(actual))
        extra = sorted(actual.difference(expected))
        raise FragmentActionTokenizationError(
            f"{path} has noncanonical keys; missing={missing}, extra={extra}"
        )
    return result


def _require_list(value: object, *, path: str) -> list[object]:
    if type(value) is not list:
        raise FragmentActionTokenizationError(f"{path} must be a JSON array")
    return list(value)


def _token_id_tuple(value: object, *, path: str) -> tuple[int, ...]:
    items = _require_list(value, path=path)
    result = tuple(
        _require_integer(item, path=f"{path}[{index}]", minimum=0)
        for index, item in enumerate(items)
    )
    _require_token_ids(result, allow_empty=False)
    return result


def _load_canonical_json(text: str, *, path: str) -> object:
    if type(text) is not str:
        raise FragmentActionTokenizationError(f"{path} JSON must be text")
    try:
        value = load_json(text)
        canonical = dump_json(value)
    except CanonicalJSONError as exc:
        raise FragmentActionTokenizationError(f"invalid {path} JSON") from exc
    if not hmac.compare_digest(text.encode("utf-8"), canonical.encode("utf-8")):
        raise FragmentActionTokenizationError(
            f"{path} JSON is not in the unique canonical byte representation"
        )
    return value


def _is_immutable_revision(value: object) -> bool:
    return (
        type(value) is str
        and len(value) in {40, 64}
        and all(character in "0123456789abcdef" for character in value)
    )


def _is_sha256(value: object) -> TypeGuard[str]:
    return (
        type(value) is str
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _segment_cache_key(segment: GrammarSegment) -> _CacheKey:
    return (
        segment.prefix,
        tuple((option.key, option.text) for option in segment.options),
    )


def _segment_spec_obj(segment: GrammarSegment) -> dict[str, object]:
    return {
        "prefix": segment.prefix,
        "options": [
            {"key": option.key, "text": option.text} for option in segment.options
        ],
    }


def _segment_spec_digest(segment: GrammarSegment) -> str:
    return json_digest(_segment_spec_obj(segment), domain=_SPEC_DIGEST_DOMAIN)


def _tokenized_output_digest(
    *,
    tokenizer_manifest_digest: str,
    spec_digest: str,
    options: Sequence[tuple[str, str, tuple[int, ...]]],
) -> str:
    return json_digest(
        {
            "tokenizer_manifest_digest": tokenizer_manifest_digest,
            "spec_digest": spec_digest,
            "options": [
                {
                    "key": key,
                    "text": text,
                    "token_ids": list(token_ids),
                }
                for key, text, token_ids in options
            ],
        },
        domain=_TOKENIZED_OUTPUT_DIGEST_DOMAIN,
    )


@lru_cache(maxsize=1)
def _registered_fragment_segments() -> tuple[GrammarSegment, ...]:
    """Enumerate all 18 unique textual specs reachable in the public grammar."""

    segments_by_key: dict[_CacheKey, GrammarSegment] = {}

    def register(segment: GrammarSegment) -> None:
        segments_by_key.setdefault(_segment_cache_key(segment), segment)

    def walk_inquiry(state: InquiryActionState) -> None:
        if state.complete:
            register(
                GrammarSegment(
                    "$completion",
                    "",
                    (GrammarOption("completion", state.completion_suffix),),
                )
            )
            return
        segment = state.next_segment()
        register(segment)
        branch_all = segment.field_id == "move" or segment.field_id.endswith(".occupied")
        choices = segment.options if branch_all else segment.options[:1]
        for option in choices:
            walk_inquiry(state.choose(option.key))

    walk_inquiry(InquiryActionState())

    answer_state = AnswerActionState(2)
    while not answer_state.complete:
        answer_segment = answer_state.next_segment()
        register(answer_segment)
        answer_state = answer_state.choose(answer_segment.options[0].key)
    register(
        GrammarSegment(
            "$completion",
            "",
            (GrammarOption("completion", answer_state.completion_suffix),),
        )
    )
    result = tuple(
        sorted(segments_by_key.values(), key=_segment_spec_digest)
    )
    if len(result) != REGISTERED_FRAGMENT_SPEC_COUNT:
        raise FragmentActionTokenizationError(
            "registered grammar traversal did not enumerate exactly 18 unique specs"
        )
    return result


def _require_trace_vocabulary_bound(
    trace: ActionTokenTrace,
    *,
    vocabulary_size: int,
) -> None:
    _require_token_ids(
        trace.action_token_ids,
        allow_empty=False,
        vocabulary_size=vocabulary_size,
    )
    for fragment in trace.fragments:
        _require_token_ids(
            fragment.token_ids,
            allow_empty=False,
            vocabulary_size=vocabulary_size,
        )
    for step in trace.steps:
        _require_token_ids(
            (step.selected_token_id,),
            allow_empty=False,
            vocabulary_size=vocabulary_size,
        )
        _require_token_ids(
            step.allowed_token_ids,
            allow_empty=False,
            vocabulary_size=vocabulary_size,
        )


def fragment_token_contract_manifest() -> dict[str, object]:
    """Describe the fixed scientific semantics of the version-2 contract."""

    return {
        "schema_version": FRAGMENT_ACTION_TOKENIZATION_SCHEMA_VERSION,
        "contract_id": FRAGMENT_ACTION_TOKENIZATION_CONTRACT_ID,
        "grammar_digest": _cached_action_language_digest(),
        "fragment_definition": (
            "standalone encode(segment.prefix + selected_option.text), followed by a "
            "standalone completion-suffix fragment"
        ),
        "action_proof": "exact decode(concatenated fragment token ids) equals canonical raw action",
        "decode_options": {
            "skip_special_tokens": False,
            "clean_up_tokenization_spaces": False,
        },
        "teacher_forcing": (
            "each selected token is paired with the exact child-token set of its option trie"
        ),
        "optimization_consumers": [
            "assistant-action-only next-token SFT",
            "masked differentiable outcome-RL rescoring",
        ],
        "cache_scope": "one option trie per textual fragment specification per compiler/tokenizer",
        "semantic_freeze": (
            "all 18 registered unique fragment specs must be precompiled before any trace"
        ),
        "registered_fragment_spec_count": REGISTERED_FRAGMENT_SPEC_COUNT,
        "tokenizer_binding": (
            "digest-bound immutable repository revision, tokenizer/config/template hashes, "
            "backend version, vocabulary size, and canonical special-token mapping"
        ),
        "tokenized_output_binding": (
            "every frozen spec and trace fragment records its option-tokenization digest"
        ),
        "runtime_counters_in_semantic_digest": False,
        "rule_option_count": SYNTACTIC_RULE_COUNT,
        "answer_factorization": "one rule trie followed by independent binary label tries",
        "answer_rule_label_cartesian_product_count": 0,
        "cumulative_prefix_stability_required": False,
        "candidate_koan_list_presented_to_model": False,
        "overlength_behavior": "raise without truncation or repair",
        "live_model_authorization": False,
    }
