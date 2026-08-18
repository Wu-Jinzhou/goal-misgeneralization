from __future__ import annotations

import json
from dataclasses import replace
from fractions import Fraction

import pytest

from goalzendo_interactive.catalog import VersionSpace, build_rule_catalog
from goalzendo_interactive_v2.population_audit import build_supported_catalog_contract_v2
from goalzendo_interactive_v2.reward_query import (
    TerminalItemGeneratorLawV2,
    build_terminal_item_generator_law_v2,
)
from goalzendo_interactive_v2.selection_leakage import (
    SELECTION_LEAKAGE_CHANCE_MARGIN,
    SelectionLeakageV2Error,
    build_exact_terminal_selection_leakage_report_v2,
    parse_exact_terminal_selection_leakage_report_v2,
    serialize_exact_terminal_selection_leakage_report_v2,
    verify_exact_terminal_selection_leakage_report_v2,
)


def _space(count: int = 3) -> VersionSpace:
    catalog = build_rule_catalog()
    supported = build_supported_catalog_contract_v2().supported_indices
    return VersionSpace(catalog, tuple(supported[:count]))


def _law(
    space: VersionSpace,
    rows: tuple[dict[int, Fraction], ...],
) -> TerminalItemGeneratorLawV2:
    return build_terminal_item_generator_law_v2(
        space,
        dict(zip(space.indices, rows, strict=True)),
        public_derivation_attestation_digest="a" * 64,
    )


def test_official_independent_law_is_exactly_at_chance_and_round_trips() -> None:
    space = _space()
    row = {0: Fraction(1, 3), 1: Fraction(2, 3)}
    law = _law(space, (row, row, row))
    report = build_exact_terminal_selection_leakage_report_v2(space, law)

    assert report.chance_accuracy == Fraction(1, 3)
    assert report.full_scene_bayes_accuracy == Fraction(1, 3)
    assert report.truth_pattern_bayes_accuracy == Fraction(1, 3)
    assert report.maximum_pairwise_total_variation == 0
    assert report.official_independent
    assert report.passed

    text = serialize_exact_terminal_selection_leakage_report_v2(report)
    parsed = parse_exact_terminal_selection_leakage_report_v2(text, space, law)
    assert parsed == report
    verify_exact_terminal_selection_leakage_report_v2(parsed, space, law)


def test_disjoint_rule_conditional_support_is_a_fatal_selection_channel() -> None:
    space = _space()
    law = _law(
        space,
        (
            {0: Fraction(1)},
            {1: Fraction(1)},
            {2: Fraction(1)},
        ),
    )
    report = build_exact_terminal_selection_leakage_report_v2(space, law)

    assert report.full_scene_bayes_accuracy == 1
    assert report.full_scene_excess == Fraction(2, 3)
    assert report.maximum_pairwise_total_variation == 1
    assert not report.official_independent
    assert not report.passed


def test_small_nonzero_channel_below_registered_margin_passes() -> None:
    space = _space(2)
    law = _law(
        space,
        (
            {0: Fraction(13, 25), 1: Fraction(12, 25)},
            {0: Fraction(12, 25), 1: Fraction(13, 25)},
        ),
    )
    report = build_exact_terminal_selection_leakage_report_v2(space, law)

    assert report.full_scene_bayes_accuracy == Fraction(13, 25)
    assert report.full_scene_excess == Fraction(1, 50)
    assert report.full_scene_excess < SELECTION_LEAKAGE_CHANCE_MARGIN
    assert report.passed


def test_channel_exactly_at_strict_margin_fails() -> None:
    space = _space(2)
    law = _law(
        space,
        (
            {0: Fraction(11, 20), 1: Fraction(9, 20)},
            {0: Fraction(9, 20), 1: Fraction(11, 20)},
        ),
    )
    report = build_exact_terminal_selection_leakage_report_v2(space, law)

    assert report.full_scene_excess == SELECTION_LEAKAGE_CHANCE_MARGIN
    assert not report.passed


def test_full_scene_oracle_dominates_truth_pattern_coarsening() -> None:
    space = _space(2)
    catalog = space.catalog
    pattern_groups: dict[tuple[bool, ...], list[int]] = {}
    for scene in range(13_716):
        pattern = tuple(catalog[index].truth[scene] for index in space.indices)
        pattern_groups.setdefault(pattern, []).append(scene)
    same_pattern = next(values for values in pattern_groups.values() if len(values) >= 2)
    left, right = same_pattern[:2]
    law = _law(
        space,
        (
            {left: Fraction(1)},
            {right: Fraction(1)},
        ),
    )
    report = build_exact_terminal_selection_leakage_report_v2(space, law)

    assert report.full_scene_bayes_accuracy == 1
    assert report.truth_pattern_bayes_accuracy == Fraction(1, 2)
    assert not report.passed


def test_parser_rejects_reordering_boolean_alias_and_rehashed_tampering() -> None:
    space = _space()
    row = {0: Fraction(1, 2), 1: Fraction(1, 2)}
    law = _law(space, (row, row, row))
    report = build_exact_terminal_selection_leakage_report_v2(space, law)
    obj = report.as_obj()

    reordered = json.dumps(obj, sort_keys=True, separators=(",", ":"))
    with pytest.raises(SelectionLeakageV2Error, match="not canonical"):
        parse_exact_terminal_selection_leakage_report_v2(reordered, space, law)

    bool_alias = dict(obj)
    bool_alias["schema_version"] = True
    with pytest.raises(SelectionLeakageV2Error, match="schema version"):
        parse_exact_terminal_selection_leakage_report_v2(
            json.dumps(bool_alias, separators=(",", ":")), space, law
        )

    tampered = dict(obj)
    tampered["full_scene_bayes_accuracy"] = {"numerator": 2, "denominator": 3}
    tampered["full_scene_excess"] = {"numerator": 1, "denominator": 3}
    tampered["passed"] = False
    with pytest.raises(SelectionLeakageV2Error):
        parse_exact_terminal_selection_leakage_report_v2(
            json.dumps(tampered, separators=(",", ":")), space, law
        )


def test_parser_requires_the_exact_live_law() -> None:
    space = _space(2)
    common = {0: Fraction(1)}
    law = _law(space, (common, common))
    report = build_exact_terminal_selection_leakage_report_v2(space, law)
    text = serialize_exact_terminal_selection_leakage_report_v2(report)
    different = _law(
        space,
        (
            {0: Fraction(1)},
            {1: Fraction(1)},
        ),
    )

    with pytest.raises(SelectionLeakageV2Error, match="exact regeneration"):
        parse_exact_terminal_selection_leakage_report_v2(text, space, different)


def test_generator_law_conditional_truth_digest_is_reauthenticated() -> None:
    space = _space(2)
    common = {0: Fraction(1)}
    law = _law(space, (common, common))
    forged = replace(
        law,
        conditional_rules=(
            replace(law.conditional_rules[0], truth_digest="0" * 64),
            law.conditional_rules[1],
        ),
    )

    with pytest.raises(SelectionLeakageV2Error, match="truth digests"):
        build_exact_terminal_selection_leakage_report_v2(space, forged)
