from __future__ import annotations

import itertools

import pytest

from goalzendo.rules import describe_rule, evaluate_rule, symbolic_formula, truth_table
from goalzendo.schema import RuleSpec


def test_controlled_rule_families_have_expected_truth_tables() -> None:
    literal = RuleSpec("literal", (0,))
    assert evaluate_rule(literal, (False,)) is False
    assert evaluate_rule(literal, (True,)) is True

    parity = RuleSpec("parity", (0, 1, 2))
    for row in itertools.product((False, True), repeat=3):
        assert evaluate_rule(parity, row) is (sum(row) % 2 == 1)

    majority = RuleSpec("majority", (0, 1, 2))
    for row in itertools.product((False, True), repeat=3):
        assert evaluate_rule(majority, row) is (sum(row) >= 2)

    conjunction = RuleSpec("conjunction", (0, 1), expected_values=(True, False))
    assert evaluate_rule(conjunction, (True, False)) is True
    assert evaluate_rule(conjunction, (True, True)) is False

    multiplexer = RuleSpec("multiplexer", (0, 1, 2))
    assert evaluate_rule(multiplexer, (True, True, False)) is True
    assert evaluate_rule(multiplexer, (True, False, True)) is False
    assert evaluate_rule(multiplexer, (False, False, True)) is True
    assert evaluate_rule(multiplexer, (False, True, False)) is False


def test_expected_values_and_output_negation_are_explicit() -> None:
    rule = RuleSpec(
        "parity",
        (0, 2),
        expected_values=(False, True),
        output_negated=True,
        name="controlled",
    )
    # The two literals are true, their odd parity is false, then output is negated.
    assert evaluate_rule(rule, (False, False, True)) is True
    assert evaluate_rule(rule, (True, False, True)) is False
    assert rule.digest == RuleSpec(
        "parity",
        (0, 2),
        expected_values=(False, True),
        output_negated=True,
        name="controlled",
    ).digest
    assert rule.digest != RuleSpec("parity", (0, 2), name="controlled").digest


def test_truth_table_and_descriptions_do_not_depend_on_a_scene_label() -> None:
    rule = RuleSpec("conjunction", (0, 1))
    table = truth_table(rule, 2)
    assert len(table) == 4
    assert sum(label for _, label in table) == 1
    natural = describe_rule(rule, ("contains red", "contains blue"))
    formula = symbolic_formula(rule, ("dax", "blick"))
    assert natural == "all of these statements hold: contains red; contains blue"
    assert formula == "AND(dax, blick)"


@pytest.mark.parametrize(
    "rule",
    [
        RuleSpec("literal", (0,)),
        RuleSpec("parity", (0, 1)),
        RuleSpec("majority", (0, 1, 2)),
        RuleSpec("conjunction", (0, 1)),
        RuleSpec("multiplexer", (0, 1, 2)),
    ],
)
def test_every_supported_rule_has_positive_and_negative_scenes(rule: RuleSpec) -> None:
    values = {label for _, label in truth_table(rule, max(rule.feature_indices) + 1)}
    assert values == {False, True}


def test_rule_validation_rejects_ambiguous_specs() -> None:
    with pytest.raises(ValueError, match="exactly one"):
        RuleSpec("literal", (0, 1))
    with pytest.raises(ValueError, match="selector"):
        RuleSpec("multiplexer", (0, 1))
    with pytest.raises(ValueError, match="duplicates"):
        RuleSpec("parity", (0, 0))
    with pytest.raises(ValueError, match="one Boolean"):
        RuleSpec("parity", (0, 1), expected_values=(True,))
