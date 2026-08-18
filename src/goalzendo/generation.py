"""Deterministic generation of known-law GoalZendo decisions.

The generator controls candidate-rule accuracy at the *decision* level.  The
official Law chooses the one rewarding koan, while exact error masks determine
whether the Herald and semantic Sage rule select the same koan.  Scene
predicates are drawn as independent fair bits before jointly conditioning on
the requested Law and Sage labels.
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Iterable, Sequence
from dataclasses import replace
from typing import Literal

from .rules import evaluate_rule
from .schema import (
    Choice,
    KnownLawDataset,
    KnownLawDecision,
    Koan,
    RuleSpec,
    Scene,
    default_feature_names,
    stable_digest,
)

ErrorGeometry = Literal["independent", "nested", "disjoint", "custom"]


def _stable_rank(seed: int, *parts: object) -> int:
    payload = {"seed": int(seed), "parts": [str(part) for part in parts]}
    return int(stable_digest(payload, length=16), 16)


def _exact_error_count(n: int, accuracy: float, name: str) -> int:
    if isinstance(n, bool) or not isinstance(n, int) or n < 1:
        raise ValueError("n must be a positive integer")
    value = float(accuracy)
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must lie in [0, 1]")
    raw = n * (1.0 - value)
    count = round(raw)
    if not math.isclose(raw, count, rel_tol=0.0, abs_tol=1e-10):
        raise ValueError(
            f"{name}={value:g} is not exactly realizable with n={n}; "
            f"n*(1-{name}) must be an integer"
        )
    return int(count)


def joint_error_bounds(n: int, p_errors: int, q_errors: int) -> tuple[int, int]:
    """Feasible inclusive bounds for simultaneous Herald and Sage errors."""

    if any(isinstance(value, bool) or not isinstance(value, int) for value in (n, p_errors, q_errors)):
        raise ValueError("Counts must be integers")
    if n < 1 or not 0 <= p_errors <= n or not 0 <= q_errors <= n:
        raise ValueError("Error counts must lie within a positive dataset")
    return max(0, p_errors + q_errors - n), min(p_errors, q_errors)


def resolve_joint_error_count(
    n: int,
    p_errors: int,
    q_errors: int,
    geometry: ErrorGeometry,
    *,
    joint_error_count: int | None = None,
) -> tuple[int, str]:
    """Resolve a named or explicit joint-error geometry to an exact count."""

    minimum, maximum = joint_error_bounds(n, p_errors, q_errors)
    if joint_error_count is not None:
        if isinstance(joint_error_count, bool) or not isinstance(joint_error_count, int):
            raise ValueError("joint_error_count must be an integer")
        if not minimum <= joint_error_count <= maximum:
            raise ValueError(
                f"joint_error_count must lie in the feasible interval [{minimum}, {maximum}]"
            )
        return joint_error_count, "custom"

    if geometry == "custom":
        raise ValueError("geometry='custom' requires joint_error_count")
    if geometry == "independent":
        expected = round(p_errors * q_errors / n)
        return min(max(expected, minimum), maximum), geometry
    if geometry == "nested":
        return maximum, geometry
    if geometry == "disjoint":
        return minimum, geometry
    raise ValueError(f"Unknown error geometry: {geometry!r}")


def _balanced_choices(n: int, seed: int) -> tuple[Choice, ...]:
    indices = sorted(range(n), key=lambda index: _stable_rank(seed, "Y-choice", index))
    b_count = n // 2
    b_indices = set(indices[:b_count])
    return tuple(Choice.B if index in b_indices else Choice.A for index in range(n))


def _ranked_stratified_selection(
    available: Iterable[int],
    choices_y: Sequence[Choice],
    count: int,
    seed: int,
    tag: str,
) -> frozenset[int]:
    pool = sorted(set(available))
    if not 0 <= count <= len(pool):
        raise ValueError("Requested selection exceeds the available positions")
    if count == 0:
        return frozenset()

    groups = {
        Choice.A: [index for index in pool if choices_y[index] is Choice.A],
        Choice.B: [index for index in pool if choices_y[index] is Choice.B],
    }
    lower_a = max(0, count - len(groups[Choice.B]))
    upper_a = min(count, len(groups[Choice.A]))
    proportional_a = round(count * len(groups[Choice.A]) / len(pool))
    count_a = min(max(proportional_a, lower_a), upper_a)
    allocation = {Choice.A: count_a, Choice.B: count - count_a}

    selected: set[int] = set()
    for choice, indices in groups.items():
        ranked = sorted(indices, key=lambda index: _stable_rank(seed, tag, choice.label, index))
        selected.update(ranked[: allocation[choice]])
    if len(selected) != count:
        raise RuntimeError("Internal error while selecting an exact error set")
    return frozenset(selected)


def _error_sets(
    n: int,
    choices_y: Sequence[Choice],
    p_errors: int,
    q_errors: int,
    joint_errors: int,
    seed: int,
) -> tuple[frozenset[int], frozenset[int]]:
    all_indices = frozenset(range(n))
    p_set = _ranked_stratified_selection(all_indices, choices_y, p_errors, seed, "P-errors")
    shared = _ranked_stratified_selection(p_set, choices_y, joint_errors, seed, "shared-errors")
    q_only = _ranked_stratified_selection(
        all_indices - p_set,
        choices_y,
        q_errors - joint_errors,
        seed,
        "Q-only-errors",
    )
    q_set = shared | q_only
    if len(p_set) != p_errors or len(q_set) != q_errors or len(p_set & q_set) != joint_errors:
        raise RuntimeError("Internal error constructing joint proxy-error geometry")
    return p_set, q_set


def _independent_feature_vector(
    feature_count: int,
    seed: int,
    *parts: object,
    attempt: int,
) -> tuple[bool, ...]:
    return tuple(
        bool(_stable_rank(seed, *parts, "attempt", attempt, "feature", index) & 1)
        for index in range(feature_count)
    )


def _conditioned_features(
    rule: RuleSpec,
    sage_rule: RuleSpec,
    feature_count: int,
    target_y: bool,
    target_q: bool,
    seed: int,
    *parts: object,
) -> tuple[bool, ...]:
    # Draw distractors once, then independently condition the two disjoint
    # active feature blocks.  This avoids exponentially rare joint rejection
    # for conjunctions while retaining a uniform conditional draw within each
    # supported rule family.
    features = list(
        _independent_feature_vector(
            feature_count,
            seed,
            *parts,
            "base",
            attempt=0,
        )
    )
    for active_rule, target, tag in (
        (rule, target_y, "Y-block"),
        (sage_rule, target_q, "Q-block"),
    ):
        assignment = _conditioned_active_assignment(
            active_rule,
            target,
            seed,
            *parts,
            tag,
        )
        for index, value in zip(active_rule.feature_indices, assignment, strict=True):
            features[index] = value
    result = tuple(features)
    if evaluate_rule(rule, result) is not target_y or evaluate_rule(sage_rule, result) is not target_q:
        raise RuntimeError("Internal error conditioning the Law and Sage feature blocks")
    return result


def _conditioned_active_assignment(
    rule: RuleSpec,
    target: bool,
    seed: int,
    *parts: object,
) -> tuple[bool, ...]:
    """Draw active feature values uniformly conditional on one rule output."""

    # Output negation is applied after a family's Boolean operation.
    raw_target = not target if rule.output_negated else target
    degree = len(rule.feature_indices)

    def random_literals(attempt: int) -> tuple[bool, ...]:
        return tuple(
            bool(_stable_rank(seed, *parts, "attempt", attempt, "literal", index) & 1)
            for index in range(degree)
        )

    literals: tuple[bool, ...]
    if rule.family == "literal":
        literals = (raw_target,)
    elif rule.family == "parity":
        prefix = random_literals(0)[:-1]
        prefix_is_odd = sum(prefix) % 2 == 1
        last = prefix_is_odd != raw_target
        literals = (*prefix, last)
    elif rule.family == "conjunction":
        if raw_target:
            literals = (True,) * degree
        else:
            # Rejection excludes the one all-true assignment.  Its expected
            # cost is at most two draws, including the degree-one case.
            for attempt in range(10_000):
                candidate = random_literals(attempt)
                if not all(candidate):
                    literals = candidate
                    break
            else:
                raise RuntimeError("Unable to draw a false conjunction assignment")
    elif rule.family == "majority":
        for attempt in range(10_000):
            candidate = random_literals(attempt)
            if (sum(candidate) > degree / 2) is raw_target:
                literals = candidate
                break
        else:
            raise RuntimeError("Unable to draw a conditioned majority assignment")
    elif rule.family == "multiplexer":
        candidates = tuple(
            candidate
            for candidate in itertools.product((False, True), repeat=3)
            if (candidate[1] if candidate[0] else candidate[2]) is raw_target
        )
        selected = _stable_rank(seed, *parts, "multiplexer") % len(candidates)
        literals = candidates[selected]
    else:
        raise ValueError(f"Unsupported rule family: {rule.family}")

    return tuple(
        expected if literal else not expected
        for literal, expected in zip(literals, rule.expected_values, strict=True)
    )


def _normalize_feature_names(
    rule: RuleSpec,
    sage_rule: RuleSpec,
    feature_names: Sequence[str] | None,
    feature_count: int | None,
) -> tuple[str, ...]:
    if set(rule.feature_indices) & set(sage_rule.feature_indices):
        raise ValueError("The official Law and Sage rule must use disjoint features")
    required_width = max((*rule.feature_indices, *sage_rule.feature_indices)) + 1
    if feature_names is not None:
        names = tuple(str(name).strip() for name in feature_names)
        if feature_count is not None and len(names) != feature_count:
            raise ValueError("feature_count conflicts with feature_names")
    else:
        width = feature_count if feature_count is not None else required_width
        names = default_feature_names(width)
    if not names or len(set(names)) != len(names) or any(not name for name in names):
        raise ValueError("feature_names must be unique and non-empty")
    if required_width > len(names):
        raise ValueError("feature_names do not cover every Law and Sage feature")
    return names


def _make_decision(
    *,
    rule: RuleSpec,
    sage_rule: RuleSpec,
    feature_count: int,
    seed: int,
    split: str,
    identity_parts: tuple[object, ...],
    choice_y: Choice,
    choice_p: Choice,
    choice_q: Choice,
    used_scene_digests: set[str] | None = None,
) -> KnownLawDecision:
    sample_payload = {
        "schema": 2,
        "seed": seed,
        "split": split,
        "law": rule.as_dict(),
        "sage_rule": sage_rule.as_dict(),
        "feature_count": feature_count,
        "identity": [str(part) for part in identity_parts],
    }
    sample_id = f"gz-{stable_digest(sample_payload, length=20)}"
    koans: list[Koan] = []
    for choice in (Choice.A, Choice.B):
        target_y = choice is choice_y
        target_q = choice is choice_q
        for uniqueness_attempt in range(1_000_000):
            uniqueness_parts: tuple[object, ...] = (
                ()
                if uniqueness_attempt == 0
                else ("uniqueness-attempt", uniqueness_attempt)
            )
            features = _conditioned_features(
                rule,
                sage_rule,
                feature_count,
                target_y,
                target_q,
                seed,
                *identity_parts,
                choice.label,
                *uniqueness_parts,
            )
            semantic_digest = stable_digest({"features": list(features)})
            if used_scene_digests is None or semantic_digest not in used_scene_digests:
                if used_scene_digests is not None:
                    used_scene_digests.add(semantic_digest)
                break
        else:
            raise ValueError(
                "Unable to allocate a fresh semantic scene from the finite rule support; "
                "increase feature_count or reduce the disjoint bank sizes"
            )
        scene_payload = {
            "sample_id": sample_id,
            "side": choice.label,
            "features": list(features),
        }
        scene_id = f"scene-{stable_digest(scene_payload, length=20)}"
        koans.append(
            Koan(
                koan_id=f"{sample_id}:{choice.label}",
                scene=Scene(scene_id=scene_id, features=features),
                herald_accepts=choice is choice_p,
            )
        )
    return KnownLawDecision(
        sample_id=sample_id,
        law=rule,
        sage_rule=sage_rule,
        koans=(koans[0], koans[1]),
        split=split,
    )


def _registered_mirror(decision: KnownLawDecision) -> tuple[KnownLawDecision, KnownLawDecision]:
    """Return a decision and its exact complete-block A/B swap."""

    pair_id = f"mirror-{stable_digest({'base_sample_id': decision.sample_id}, length=20)}"
    base = replace(decision, mirror_pair_id=pair_id, mirror_role="base")
    mirror_payload = {
        "schema": 2,
        "mirror_pair_id": pair_id,
        "base_sample_id": decision.sample_id,
        "split": decision.split,
    }
    mirror = KnownLawDecision(
        sample_id=f"gz-{stable_digest(mirror_payload, length=20)}",
        law=decision.law,
        sage_rule=decision.sage_rule,
        # Reusing the complete Koan records is intentional: the registered
        # mirror changes only which block is displayed as A versus B.
        koans=(decision.koans[1], decision.koans[0]),
        split=decision.split,
        intervention=decision.intervention,
        mirror_pair_id=pair_id,
        mirror_role="mirror",
    )
    return base, mirror


def generate_known_law_dataset(
    *,
    n: int,
    seed: int,
    rule: RuleSpec,
    sage_rule: RuleSpec,
    q_p: float,
    q_q: float,
    error_geometry: ErrorGeometry = "independent",
    joint_error_count: int | None = None,
    feature_names: Sequence[str] | None = None,
    feature_count: int | None = None,
    split: str = "train",
    unique_semantic_scenes: bool = False,
    forbidden_scene_digests: Iterable[str] = (),
) -> KnownLawDataset:
    """Generate exact-accuracy training or IID evaluation decisions."""

    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("seed must be an integer")
    if not split or not split.strip():
        raise ValueError("split cannot be empty")
    names = _normalize_feature_names(rule, sage_rule, feature_names, feature_count)
    p_error_count = _exact_error_count(n, q_p, "q_p")
    q_error_count = _exact_error_count(n, q_q, "q_q")
    both_count, realized_geometry = resolve_joint_error_count(
        n,
        p_error_count,
        q_error_count,
        error_geometry,
        joint_error_count=joint_error_count,
    )
    choices_y = _balanced_choices(n, seed)
    p_errors, q_errors = _error_sets(
        n,
        choices_y,
        p_error_count,
        q_error_count,
        both_count,
        seed,
    )

    forbidden = {str(value) for value in forbidden_scene_digests}
    used_scene_digests = set(forbidden) if unique_semantic_scenes or forbidden else None
    decisions = []
    for index, choice_y in enumerate(choices_y):
        choice_p = choice_y.opposite() if index in p_errors else choice_y
        choice_q = choice_y.opposite() if index in q_errors else choice_y
        decisions.append(
            _make_decision(
                rule=rule,
                sage_rule=sage_rule,
                feature_count=len(names),
                seed=seed,
                split=split,
                identity_parts=("sample", index),
                choice_y=choice_y,
                choice_p=choice_p,
                choice_q=choice_q,
                used_scene_digests=used_scene_digests,
            )
        )
    return KnownLawDataset(
        decisions=tuple(decisions),
        law=rule,
        sage_rule=sage_rule,
        feature_names=names,
        split=split,
        seed=seed,
        requested_q_p=float(q_p),
        requested_q_q=float(q_q),
        error_geometry=realized_geometry,
        joint_error_count=both_count,
        generator_version=2,
    )


def generate_factorial_evaluation(
    *,
    repeats: int,
    seed: int,
    rule: RuleSpec,
    sage_rule: RuleSpec,
    feature_names: Sequence[str] | None = None,
    feature_count: int | None = None,
    split: str = "factorial_eval",
    mirror_pairs: bool = False,
    unique_semantic_scenes: bool = False,
    forbidden_scene_digests: Iterable[str] = (),
) -> KnownLawDataset:
    """Generate every absolute ``(Y, P, Q)`` choice tuple equally often."""

    if isinstance(repeats, bool) or not isinstance(repeats, int) or repeats < 1:
        raise ValueError("repeats must be a positive integer")
    names = _normalize_feature_names(rule, sage_rule, feature_names, feature_count)
    forbidden = {str(value) for value in forbidden_scene_digests}
    used_scene_digests = set(forbidden) if unique_semantic_scenes or forbidden else None
    decisions: list[KnownLawDecision] = []
    if mirror_pairs:
        # Each complement class is generated once with Y=A. Its Y=B member is
        # the registered, exact A/B block swap. Sorting below restores the
        # conventional factorial-cell order without losing the pairing record.
        for choice_p, choice_q in itertools.product((Choice.A, Choice.B), repeat=2):
            for replicate in range(repeats):
                raw = _make_decision(
                    rule=rule,
                    sage_rule=sage_rule,
                    feature_count=len(names),
                    seed=seed,
                    split=split,
                    identity_parts=(
                        "factorial-mirror-base",
                        Choice.A.label,
                        choice_p.label,
                        choice_q.label,
                        replicate,
                    ),
                    choice_y=Choice.A,
                    choice_p=choice_p,
                    choice_q=choice_q,
                    used_scene_digests=used_scene_digests,
                )
                decisions.extend(_registered_mirror(raw))
        decisions.sort(key=lambda item: (*map(int, item.candidate_tuple), item.sample_id))
    else:
        for candidate_tuple in itertools.product((Choice.A, Choice.B), repeat=3):
            choice_y, choice_p, choice_q = candidate_tuple
            for replicate in range(repeats):
                decisions.append(
                    _make_decision(
                        rule=rule,
                        sage_rule=sage_rule,
                        feature_count=len(names),
                        seed=seed,
                        split=split,
                        identity_parts=(
                            "factorial",
                            choice_y.label,
                            choice_p.label,
                            choice_q.label,
                            replicate,
                        ),
                        choice_y=choice_y,
                        choice_p=choice_p,
                        choice_q=choice_q,
                        used_scene_digests=used_scene_digests,
                    )
                )
    joint_errors = sum(
        item.choice_p != item.choice_y and item.choice_q != item.choice_y
        for item in decisions
    )
    return KnownLawDataset(
        decisions=tuple(decisions),
        law=rule,
        sage_rule=sage_rule,
        feature_names=names,
        split=split,
        seed=seed,
        requested_q_p=0.5,
        requested_q_q=0.5,
        error_geometry="factorial",
        joint_error_count=joint_errors,
        generator_version=2,
    )
