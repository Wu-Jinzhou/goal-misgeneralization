"""Canonical JSON primitives shared by the interactive engine.

The scientific action language is deliberately narrower than generic JSON:
duplicate object keys and non-finite numeric constants are errors, and every
serializer emits one compact, deterministic spelling.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from typing import Any


class CanonicalJSONError(ValueError):
    """Raised when input is outside the engine's strict JSON language."""


def _object_without_duplicates(pairs: Iterable[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise CanonicalJSONError(f"duplicate JSON object key: {key!r}")
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise CanonicalJSONError(f"non-finite JSON constant is forbidden: {value}")


def load_json(text: str) -> Any:
    """Load strict RFC-style JSON while rejecting duplicate keys."""

    if type(text) is not str:
        raise CanonicalJSONError("JSON input must be a string")
    if not text:
        raise CanonicalJSONError("JSON input cannot be empty")
    try:
        return json.loads(
            text,
            object_pairs_hook=_object_without_duplicates,
            parse_constant=_reject_constant,
        )
    except CanonicalJSONError:
        raise
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise CanonicalJSONError(f"invalid JSON: {exc}") from exc


def dump_json(value: Any) -> str:
    """Return the unique compact spelling of a canonical engine object."""

    try:
        return json.dumps(
            value,
            ensure_ascii=True,
            allow_nan=False,
            separators=(",", ":"),
        )
    except (TypeError, ValueError) as exc:
        raise CanonicalJSONError(f"value is not canonical JSON: {exc}") from exc


def json_digest(value: Any, *, domain: str) -> str:
    """Hash a canonical JSON value with an explicit format domain."""

    payload = dump_json(value).encode("ascii")
    digest = hashlib.sha256()
    digest.update(domain.encode("ascii"))
    digest.update(b"\0")
    digest.update(payload)
    return digest.hexdigest()
