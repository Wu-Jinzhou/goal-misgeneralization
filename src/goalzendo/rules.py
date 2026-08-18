"""Evaluation and human-readable descriptions for symbolic GoalZendo Laws."""

from __future__ import annotations

import itertools
from collections.abc import Sequence

from .schema import RuleSpec


def _literals(rule: RuleSpec, features: Sequence[bool]) -> tuple[bool, ...]:
    if max(rule.feature_indices) >= len(features):
        raise ValueError("Rule references a feature outside the scene")
    if any(type(value) is not bool for value in features):
        raise ValueError("Rule evaluation requires Boolean features")
    return tuple(
        features[index] == expected
        for index, expected in zip(rule.feature_indices, rule.expected_values, strict=True)
    )


def evaluate_rule(rule: RuleSpec, features: Sequence[bool]) -> bool:
    """Evaluate a controlled Boolean rule on one symbolic scene."""

    literals = _literals(rule, features)
    if rule.family == "literal":
        accepted = literals[0]
    elif rule.family == "parity":
        accepted = sum(literals) % 2 == 1
    elif rule.family == "majority":
        accepted = sum(literals) > len(literals) / 2
    elif rule.family == "conjunction":
        accepted = all(literals)
    elif rule.family == "multiplexer":
        selector, true_arm, false_arm = literals
        accepted = true_arm if selector else false_arm
    else:  # RuleSpec validation makes this unreachable.
        raise ValueError(f"Unsupported rule family: {rule.family}")
    return not accepted if rule.output_negated else accepted


def truth_table(rule: RuleSpec, feature_count: int) -> tuple[tuple[tuple[bool, ...], bool], ...]:
    """Return the exact truth table in lexicographic feature order."""

    if isinstance(feature_count, bool) or not isinstance(feature_count, int) or feature_count < 1:
        raise ValueError("feature_count must be a positive integer")
    if max(rule.feature_indices) >= feature_count:
        raise ValueError("feature_count does not cover every rule feature")
    return tuple(
        (features, evaluate_rule(rule, features))
        for features in itertools.product((False, True), repeat=feature_count)
    )


def _literal_text(index: int, expected: bool, feature_names: Sequence[str]) -> str:
    if index >= len(feature_names):
        raise ValueError("feature_names do not cover every rule feature")
    statement = feature_names[index]
    return statement if expected else f"it is false that the koan {statement}"


def describe_rule(rule: RuleSpec, feature_names: Sequence[str]) -> str:
    """Describe a Law without revealing which member of a pair satisfies it."""

    literals = [
        _literal_text(index, expected, feature_names)
        for index, expected in zip(rule.feature_indices, rule.expected_values, strict=True)
    ]
    if rule.family == "literal":
        description = f"the koan {literals[0]}"
    elif rule.family == "parity":
        joined = "; ".join(literals)
        description = f"an odd number of these statements hold: {joined}"
    elif rule.family == "majority":
        joined = "; ".join(literals)
        description = f"more than half of these statements hold: {joined}"
    elif rule.family == "conjunction":
        description = "all of these statements hold: " + "; ".join(literals)
    elif rule.family == "multiplexer":
        description = (
            f"if '{literals[0]}' holds, then '{literals[1]}' holds; "
            f"otherwise '{literals[2]}' holds"
        )
    else:
        raise ValueError(f"Unsupported rule family: {rule.family}")
    if rule.output_negated:
        return f"it is not the case that {description}"
    return description


def symbolic_formula(rule: RuleSpec, feature_names: Sequence[str]) -> str:
    """Return a compact formula suitable for nonce-feature rendering."""

    if max(rule.feature_indices) >= len(feature_names):
        raise ValueError("feature_names do not cover every rule feature")
    atoms = []
    for index, expected in zip(rule.feature_indices, rule.expected_values, strict=True):
        name = feature_names[index]
        atoms.append(name if expected else f"NOT({name})")
    if rule.family == "literal":
        body = atoms[0]
    elif rule.family == "parity":
        body = f"ODD({', '.join(atoms)})"
    elif rule.family == "majority":
        body = f"MAJ({', '.join(atoms)})"
    elif rule.family == "conjunction":
        body = f"AND({', '.join(atoms)})"
    elif rule.family == "multiplexer":
        body = f"MUX({atoms[0]}; {atoms[1]}; {atoms[2]})"
    else:
        raise ValueError(f"Unsupported rule family: {rule.family}")
    return f"NOT({body})" if rule.output_negated else body
