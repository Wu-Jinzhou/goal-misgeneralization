"""Immutable symbolic records used by GoalZendo.

The symbolic layer deliberately knows nothing about tokenizers or language
models.  A scene is a vector of independent Boolean predicates; a decision is
a pair of scenes for which exactly one satisfies the official Law.  A visible
Herald stamp supplies the direct proxy, while a second semantic rule over a
disjoint feature set determines which scene the Sage favors.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import IntEnum
from typing import Any

SCHEMA_VERSION = 2


def stable_digest(value: Any, *, length: int = 64) -> str:
    """Return a process-independent digest of JSON-compatible data."""

    if not 1 <= length <= 64:
        raise ValueError("length must lie in [1, 64]")
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:length]


class Choice(IntEnum):
    """One of the two koans offered in a known-law decision."""

    A = 0
    B = 1

    @property
    def label(self) -> str:
        return self.name

    def opposite(self) -> Choice:
        return Choice.B if self is Choice.A else Choice.A

    @classmethod
    def parse(cls, value: Choice | str | int) -> Choice:
        if isinstance(value, cls):
            return value
        if isinstance(value, bool):
            raise ValueError("Boolean values are not valid choices")
        if isinstance(value, int):
            try:
                return cls(value)
            except ValueError as exc:
                raise ValueError(f"Invalid choice integer: {value}") from exc
        normalized = str(value).strip().upper()
        if normalized in {"A", "0"}:
            return cls.A
        if normalized in {"B", "1"}:
            return cls.B
        raise ValueError(f"Invalid choice: {value!r}")


RULE_FAMILIES = frozenset(
    {"literal", "parity", "majority", "conjunction", "multiplexer"}
)


@dataclass(frozen=True)
class RuleSpec:
    """A controlled Boolean Law over named scene predicates.

    ``expected_values`` turns each selected feature into a literal.  A literal
    is true when the observed feature equals its expected value.  The rule
    family combines those literals, after which ``output_negated`` optionally
    reverses acceptance.
    """

    family: str
    feature_indices: tuple[int, ...]
    expected_values: tuple[bool, ...] = ()
    output_negated: bool = False
    name: str = "official_law"

    def __post_init__(self) -> None:
        family = self.family.strip().lower()
        object.__setattr__(self, "family", family)
        indices = tuple(self.feature_indices)
        object.__setattr__(self, "feature_indices", indices)
        if family not in RULE_FAMILIES:
            raise ValueError(f"Unknown rule family: {family!r}")
        if not indices:
            raise ValueError("A rule must use at least one feature")
        if any(isinstance(index, bool) or not isinstance(index, int) or index < 0 for index in indices):
            raise ValueError("feature_indices must be distinct non-negative integers")
        if len(set(indices)) != len(indices):
            raise ValueError("feature_indices must not contain duplicates")
        if family == "literal" and len(indices) != 1:
            raise ValueError("literal rules require exactly one feature")
        if family == "multiplexer" and len(indices) != 3:
            raise ValueError("multiplexer rules require selector, true-arm, and false-arm features")

        expected = tuple(self.expected_values) if self.expected_values else (True,) * len(indices)
        if len(expected) != len(indices) or any(type(value) is not bool for value in expected):
            raise ValueError("expected_values must contain one Boolean per selected feature")
        object.__setattr__(self, "expected_values", expected)
        if type(self.output_negated) is not bool:
            raise ValueError("output_negated must be Boolean")
        if not self.name or not self.name.strip():
            raise ValueError("Rule name cannot be empty")

    def as_dict(self) -> dict[str, Any]:
        return {
            "family": self.family,
            "feature_indices": list(self.feature_indices),
            "expected_values": list(self.expected_values),
            "output_negated": self.output_negated,
            "name": self.name,
        }

    @property
    def digest(self) -> str:
        return stable_digest(self.as_dict())


@dataclass(frozen=True)
class Scene:
    """A symbolic koan scene represented by Boolean predicates."""

    scene_id: str
    features: tuple[bool, ...]

    def __post_init__(self) -> None:
        features = tuple(self.features)
        object.__setattr__(self, "features", features)
        if not self.scene_id or not self.scene_id.strip():
            raise ValueError("scene_id cannot be empty")
        if not features:
            raise ValueError("A scene must contain at least one feature")
        if any(type(value) is not bool for value in features):
            raise ValueError("Scene features must be Boolean")

    def as_dict(self) -> dict[str, Any]:
        return {"scene_id": self.scene_id, "features": list(self.features)}

    @property
    def semantic_digest(self) -> str:
        """Digest of scene content, intentionally excluding its instance ID."""

        return stable_digest({"features": list(self.features)})


@dataclass(frozen=True)
class Koan:
    """One candidate scene and its direct, non-authoritative Herald stamp."""

    koan_id: str
    scene: Scene
    herald_accepts: bool

    def __post_init__(self) -> None:
        if not self.koan_id or not self.koan_id.strip():
            raise ValueError("koan_id cannot be empty")
        if type(self.herald_accepts) is not bool:
            raise ValueError("The Herald label must be Boolean")

    def as_dict(self) -> dict[str, Any]:
        return {
            "koan_id": self.koan_id,
            "scene": self.scene.as_dict(),
            "herald_accepts": self.herald_accepts,
        }


@dataclass(frozen=True)
class KnownLawDecision:
    """A paired-koan choice with three well-defined candidate rules."""

    sample_id: str
    law: RuleSpec
    sage_rule: RuleSpec
    koans: tuple[Koan, Koan]
    split: str = "train"
    intervention: str = "none"
    mirror_pair_id: str | None = None
    mirror_role: str | None = None

    def __post_init__(self) -> None:
        koans = tuple(self.koans)
        object.__setattr__(self, "koans", koans)
        if not self.sample_id or not self.sample_id.strip():
            raise ValueError("sample_id cannot be empty")
        if len(koans) != 2:
            raise ValueError("Known-law decisions require exactly two koans")
        if koans[0].koan_id == koans[1].koan_id:
            raise ValueError("The two koans must have distinct IDs")
        if len(koans[0].scene.features) != len(koans[1].scene.features):
            raise ValueError("The two scenes must use the same feature width")
        if max(self.law.feature_indices) >= len(koans[0].scene.features):
            raise ValueError("The Law references a feature absent from the scenes")
        if max(self.sage_rule.feature_indices) >= len(koans[0].scene.features):
            raise ValueError("The Sage rule references a feature absent from the scenes")
        if set(self.law.feature_indices) & set(self.sage_rule.feature_indices):
            raise ValueError("The official Law and Sage rule must use disjoint features")
        if not self.split or not self.intervention:
            raise ValueError("split and intervention cannot be empty")
        if (self.mirror_pair_id is None) != (self.mirror_role is None):
            raise ValueError("mirror_pair_id and mirror_role must be supplied together")
        if self.mirror_role not in {None, "base", "mirror"}:
            raise ValueError("mirror_role must be 'base', 'mirror', or None")
        if self.mirror_pair_id is not None and not self.mirror_pair_id.strip():
            raise ValueError("mirror_pair_id cannot be empty")

        # Local import avoids making the rule evaluator depend on this schema's
        # higher-level decision records.
        from .rules import evaluate_rule

        y_values = [evaluate_rule(self.law, koan.scene.features) for koan in koans]
        q_values = [evaluate_rule(self.sage_rule, koan.scene.features) for koan in koans]
        if sum(y_values) != 1:
            raise ValueError("Exactly one koan must satisfy the official Law")
        if sum(koan.herald_accepts for koan in koans) != 1:
            raise ValueError("Exactly one koan must be recommended by the Herald")
        if sum(q_values) != 1:
            raise ValueError("Exactly one koan must satisfy the Sage's semantic rule")

    @property
    def feature_count(self) -> int:
        return len(self.koans[0].scene.features)

    @property
    def choice_y(self) -> Choice:
        from .rules import evaluate_rule

        return Choice.A if evaluate_rule(self.law, self.koans[0].scene.features) else Choice.B

    @property
    def choice_p(self) -> Choice:
        return Choice.A if self.koans[0].herald_accepts else Choice.B

    @property
    def choice_q(self) -> Choice:
        from .rules import evaluate_rule

        return Choice.A if evaluate_rule(self.sage_rule, self.koans[0].scene.features) else Choice.B

    @property
    def candidate_tuple(self) -> tuple[Choice, Choice, Choice]:
        return self.choice_y, self.choice_p, self.choice_q

    def as_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "sample_id": self.sample_id,
            "law": self.law.as_dict(),
            "sage_rule": self.sage_rule.as_dict(),
            "koans": [koan.as_dict() for koan in self.koans],
            "split": self.split,
            "intervention": self.intervention,
        }
        # Omit absent mirror metadata to retain the schema-1 serialization of
        # ordinary, unmirrored decisions.
        if self.mirror_pair_id is not None:
            result["mirror_pair_id"] = self.mirror_pair_id
            result["mirror_role"] = self.mirror_role
        return result

    @property
    def digest(self) -> str:
        return stable_digest(self.as_dict())


DEFAULT_FEATURE_NAMES: tuple[str, ...] = (
    "has a red pyramid",
    "has a large cube",
    "has an upright wedge",
    "has a green sphere",
    "has a sphere touching the cone",
    "has a blue cone",
    "has a large pyramid",
    "has a green cube",
    "has a small wedge",
    "has an upright cone",
    "has a pyramid touching the cube",
    "has a cube touching the wedge",
    "has a left-pointing cone",
    "has a small sphere",
    "has an upright pyramid",
    "has a blue wedge",
    "has a cone touching the pyramid",
)


def default_feature_names(count: int) -> tuple[str, ...]:
    if isinstance(count, bool) or not isinstance(count, int) or count < 1:
        raise ValueError("Feature count must be a positive integer")
    names = list(DEFAULT_FEATURE_NAMES[:count])
    names.extend(f"satisfies symbolic predicate {index + 1}" for index in range(len(names), count))
    return tuple(names)


@dataclass(frozen=True)
class KnownLawDataset:
    """A deterministic dataset plus its evidence-geometry specification."""

    decisions: tuple[KnownLawDecision, ...]
    law: RuleSpec
    sage_rule: RuleSpec
    feature_names: tuple[str, ...]
    split: str
    seed: int
    requested_q_p: float
    requested_q_q: float
    error_geometry: str
    joint_error_count: int
    generator_version: int = 1

    def __post_init__(self) -> None:
        decisions = tuple(self.decisions)
        names = tuple(self.feature_names)
        object.__setattr__(self, "decisions", decisions)
        object.__setattr__(self, "feature_names", names)
        if not decisions:
            raise ValueError("A dataset must contain at least one decision")
        if len(names) != decisions[0].feature_count:
            raise ValueError("feature_names must match the scene feature width")
        if len(set(names)) != len(names) or any(not name.strip() for name in names):
            raise ValueError("feature_names must be unique and non-empty")
        if len({decision.sample_id for decision in decisions}) != len(decisions):
            raise ValueError("sample_id values must be unique within a dataset")
        if any(decision.law != self.law for decision in decisions):
            raise ValueError("Every decision must use the dataset's Law")
        if any(decision.sage_rule != self.sage_rule for decision in decisions):
            raise ValueError("Every decision must use the dataset's Sage rule")
        if set(self.law.feature_indices) & set(self.sage_rule.feature_indices):
            raise ValueError("The official Law and Sage rule must use disjoint features")
        if any(decision.split != self.split for decision in decisions):
            raise ValueError("Every decision must use the dataset's split")
        if any(decision.feature_count != len(names) for decision in decisions):
            raise ValueError("Every decision must use the dataset's feature width")
        mirror_groups: dict[str, dict[str, KnownLawDecision]] = {}
        for decision in decisions:
            if decision.mirror_pair_id is None:
                continue
            assert decision.mirror_role is not None
            roles = mirror_groups.setdefault(decision.mirror_pair_id, {})
            if decision.mirror_role in roles:
                raise ValueError("A mirror pair cannot contain duplicate roles")
            roles[decision.mirror_role] = decision
        for pair_id, roles in mirror_groups.items():
            if set(roles) != {"base", "mirror"}:
                raise ValueError(f"Mirror pair {pair_id!r} requires one base and one mirror")
            base, mirror = roles["base"], roles["mirror"]
            if mirror.koans != (base.koans[1], base.koans[0]):
                raise ValueError("A registered mirror must swap the complete A/B koan blocks")
            if mirror.candidate_tuple != tuple(choice.opposite() for choice in base.candidate_tuple):
                raise ValueError("A registered mirror must complement the complete candidate tuple")
        if not 0 <= self.joint_error_count <= len(decisions):
            raise ValueError("joint_error_count is outside the dataset")
        for value, name in ((self.requested_q_p, "requested_q_p"), (self.requested_q_q, "requested_q_q")):
            if not 0.0 <= float(value) <= 1.0:
                raise ValueError(f"{name} must lie in [0, 1]")

        p_errors = sum(decision.choice_p != decision.choice_y for decision in decisions)
        q_errors = sum(decision.choice_q != decision.choice_y for decision in decisions)
        both_errors = sum(
            decision.choice_p != decision.choice_y and decision.choice_q != decision.choice_y
            for decision in decisions
        )
        tolerance = 1e-12
        if abs(self.realized_q_p - float(self.requested_q_p)) > tolerance:
            raise ValueError("Realized Herald accuracy does not match requested_q_p")
        if abs(self.realized_q_q - float(self.requested_q_q)) > tolerance:
            raise ValueError("Realized Sage accuracy does not match requested_q_q")
        if both_errors != self.joint_error_count:
            raise ValueError("Realized joint errors do not match joint_error_count")
        if p_errors > len(decisions) or q_errors > len(decisions):
            raise RuntimeError("Invalid internal error count")

    def __len__(self) -> int:
        return len(self.decisions)

    @property
    def realized_q_p(self) -> float:
        return sum(item.choice_p == item.choice_y for item in self.decisions) / len(self.decisions)

    @property
    def realized_q_q(self) -> float:
        return sum(item.choice_q == item.choice_y for item in self.decisions) / len(self.decisions)

    @property
    def realized_joint_error_rate(self) -> float:
        return self.joint_error_count / len(self.decisions)

    def manifest(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "generator_version": self.generator_version,
            "split": self.split,
            "seed": self.seed,
            "law": self.law.as_dict(),
            "sage_rule": self.sage_rule.as_dict(),
            "feature_names": list(self.feature_names),
            "requested_q_p": self.requested_q_p,
            "requested_q_q": self.requested_q_q,
            "error_geometry": self.error_geometry,
            "joint_error_count": self.joint_error_count,
            "decisions": [decision.as_dict() for decision in self.decisions],
        }

    @property
    def manifest_digest(self) -> str:
        return stable_digest(self.manifest())


def canonical_candidate_counts(
    decisions: Sequence[KnownLawDecision],
) -> Mapping[tuple[Choice, Choice, Choice], int]:
    """Count the eight absolute ``(Y, P, Q)`` candidate-choice cells."""

    counts: dict[tuple[Choice, Choice, Choice], int] = {}
    for decision in decisions:
        cell = decision.candidate_tuple
        counts[cell] = counts.get(cell, 0) + 1
    return counts
