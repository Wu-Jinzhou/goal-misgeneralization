from __future__ import annotations

import json
from fractions import Fraction
from functools import cache
from pathlib import Path
from typing import Any, cast

import pytest

from goalzendo_interactive import (
    SCENE_COUNT,
    CatalogEntry,
    RuleCatalog,
    VersionSpace,
    build_rule_catalog,
)
from goalzendo_interactive_v2.population_audit import build_supported_catalog_contract_v2
from goalzendo_interactive_v2.reward_query import (
    MAX_REWARD_QUERY_BUDGET,
    RewardQueryV2Error,
    TerminalItemGeneratorLawV2,
    build_reward_query_policy_ceiling_report_v2,
    build_terminal_item_generator_law_v2,
    parse_reward_query_policy_ceiling_report_v2,
    serialize_reward_query_policy_ceiling_report_v2,
    verify_reward_query_policy_ceiling_report_v2,
)

_PUBLIC_DERIVATION_DIGEST = "d" * 64


@pytest.fixture(scope="module")
def catalog() -> RuleCatalog:
    return build_rule_catalog()


@pytest.fixture(scope="module")
def fixture_episode() -> dict[str, Any]:
    path = (
        Path(__file__).parents[1] / "goalzendo_interactive" / "fixtures" / "g03-engine-small-fixture-v1.json"
    )
    value = json.loads(path.read_text(encoding="ascii"))
    return cast(dict[str, Any], value["episodes"][1])


def _small_context(
    catalog: RuleCatalog,
    episode: dict[str, Any],
) -> tuple[VersionSpace, CatalogEntry, tuple[int, ...], tuple[int, ...]]:
    source = catalog.version_space(
        (observation["scene_index"], observation["accepted"]) for observation in episode["opening"]
    )
    supported = set(build_supported_catalog_contract_v2().supported_indices)
    official = catalog[int(episode["target_rule_id"][4:])]
    selected = tuple(
        sorted(
            (
                official.index,
                *[index for index in source.indices if index != official.index and index in supported][:3],
            )
        )
    )
    panel = tuple(observation["scene_index"] for observation in episode["terminal"])
    exclusions = tuple(observation["scene_index"] for observation in episode["opening"][:3])
    return VersionSpace(catalog, selected), official, panel, exclusions


def _uniform_law(
    space: VersionSpace,
    scenes: tuple[int, ...],
) -> TerminalItemGeneratorLawV2:
    unique = tuple(sorted(set(scenes)))
    assert unique
    probability = Fraction(1, len(unique))
    return build_terminal_item_generator_law_v2(
        space,
        {index: {scene: probability for scene in unique} for index in space.indices},
        public_derivation_attestation_digest=_PUBLIC_DERIVATION_DIGEST,
    )


def _independent_patterns(
    catalog: RuleCatalog,
    indices: tuple[int, ...],
    exclusions: tuple[int, ...],
) -> tuple[int, ...]:
    excluded = set(exclusions)
    patterns: set[int] = set()
    for scene in range(SCENE_COUNT):
        if scene in excluded:
            continue
        labels = 0
        for offset, index in enumerate(indices):
            if catalog[index].truth[scene]:
                labels |= 1 << offset
        patterns.add(labels)
    return tuple(sorted(patterns))


def _independent_values(
    catalog: RuleCatalog,
    indices: tuple[int, ...],
    law: TerminalItemGeneratorLawV2,
    exclusions: tuple[int, ...],
    budget: int,
) -> tuple[Fraction, Fraction, int]:
    """Brute-force without report classes, action hashes, or split collapse."""

    patterns = _independent_patterns(catalog, indices, exclusions)
    probabilities = law.probability_maps()
    root = (1 << len(indices)) - 1

    def stop_value(state: int, query_count: int) -> Fraction:
        live = tuple(index for offset, index in enumerate(indices) if state & (1 << offset))
        scenes = {scene for index in live for scene in probabilities[index]}
        correct = Fraction()
        for scene in scenes:
            rejected = sum(
                (
                    probabilities[index].get(scene, Fraction())
                    for index in live
                    if not catalog[index].truth[scene]
                ),
                Fraction(),
            )
            accepted = sum(
                (
                    probabilities[index].get(scene, Fraction())
                    for index in live
                    if catalog[index].truth[scene]
                ),
                Fraction(),
            )
            correct += max(rejected, accepted)
        accuracy = correct / len(live)
        return (
            Fraction(7, 10) * accuracy
            + Fraction(1, 4 * len(live))
            + Fraction(1, 20) * Fraction(MAX_REWARD_QUERY_BUDGET - query_count, 6)
        )

    @cache
    def composite(state: int, remaining: int, query_count: int) -> Fraction:
        result = stop_value(state, query_count)
        if remaining == 0:
            return result
        size = state.bit_count()
        for labels in patterns:
            accepted = labels & state
            rejected = state ^ accepted
            if not accepted or not rejected:
                continue
            result = max(
                result,
                Fraction(rejected.bit_count(), size) * composite(rejected, remaining - 1, query_count + 1)
                + Fraction(accepted.bit_count(), size) * composite(accepted, remaining - 1, query_count + 1),
            )
        return result

    @cache
    def identification(state: int, remaining: int) -> int:
        if state.bit_count() == 1:
            return 1
        if remaining == 0:
            return 0
        result = 0
        for labels in patterns:
            accepted = labels & state
            rejected = state ^ accepted
            if accepted and rejected:
                result = max(
                    result,
                    identification(rejected, remaining - 1) + identification(accepted, remaining - 1),
                )
        return result

    return composite(root, budget, 0), stop_value(root, 0), identification(root, budget)


def test_composite_policy_matches_independent_exact_brute_force(
    catalog: RuleCatalog,
    fixture_episode: dict[str, Any],
) -> None:
    space, official, panel, exclusions = _small_context(catalog, fixture_episode)
    law = _uniform_law(space, (*panel, *range(16)))
    report = build_reward_query_policy_ceiling_report_v2(
        space,
        official,
        terminal_item_generator_law=law,
        excluded_query_scene_indices=exclusions,
        maximum_query_budget=2,
    )
    composite, stop, identified = _independent_values(
        catalog,
        space.indices,
        law,
        tuple(sorted(exclusions)),
        2,
    )

    assert report.composite_reward.optimal_expected_reward == composite
    assert report.composite_reward.immediate_stop_expected_reward == stop
    assert report.identification_only.exact_identification_probability == Fraction(
        identified,
        len(space),
    )
    assert report.composite_reward.expected_terminal_accuracy == 1
    assert report.composite_reward.expected_exact_rule == 1
    assert report.composite_reward.expected_query_count == 2
    assert report.composite_reward.expected_query_efficiency == Fraction(2, 3)
    assert report.composite_reward.optimal_expected_reward == Fraction(59, 60)
    assert verify_reward_query_policy_ceiling_report_v2(report) is report


def _constant_pattern_scenes(
    catalog: RuleCatalog,
    indices: tuple[int, ...],
    pattern: int,
) -> tuple[int, ...]:
    result: list[int] = []
    for scene in range(SCENE_COUNT):
        labels = 0
        for offset, index in enumerate(indices):
            if catalog[index].truth[scene]:
                labels |= 1 << offset
        if labels == pattern:
            result.append(scene)
            if len(result) == 16:
                return tuple(result)
    raise AssertionError(f"pattern {pattern} has fewer than 16 scenes")


def test_public_generator_law_changes_value_without_secret_support(
    catalog: RuleCatalog,
    fixture_episode: dict[str, Any],
) -> None:
    space, official, _, exclusions = _small_context(catalog, fixture_episode)
    first_law = _uniform_law(
        space,
        (
            *_constant_pattern_scenes(catalog, space.indices, 1),
            *_constant_pattern_scenes(catalog, space.indices, 2),
        ),
    )
    second_law = _uniform_law(
        space,
        (
            *_constant_pattern_scenes(catalog, space.indices, 3),
            *_constant_pattern_scenes(catalog, space.indices, 12),
        ),
    )
    reports = tuple(
        build_reward_query_policy_ceiling_report_v2(
            space,
            official,
            terminal_item_generator_law=law,
            excluded_query_scene_indices=exclusions,
            maximum_query_budget=1,
        )
        for law in (first_law, second_law)
    )
    first, second = reports

    assert first.composite_reward.optimal_expected_reward != second.composite_reward.optimal_expected_reward
    assert first.composite_reward.first_query != second.composite_reward.first_query
    assert first.policy_context_digest != second.policy_context_digest
    assert first.query_tie_context_digest == second.query_tie_context_digest
    assert first.identification_only.first_query == second.identification_only.first_query
    for report in reports:
        obj = report.as_obj()
        assert obj["terminal_item_generator_law_public_before_inquiry"] is True
        assert obj["public_generator_derivation_verification_required"] is True
        assert obj["secret_realized_panel_support_included"] is False
        assert obj["secret_reservoir_state_excluded_from_query_policy"] is True
        assert obj["one_item_selection_leakage_gate_required"] is True


def test_report_objects_do_not_expose_mutable_global_contracts(
    catalog: RuleCatalog,
    fixture_episode: dict[str, Any],
) -> None:
    space, official, panel, _ = _small_context(catalog, fixture_episode)
    first = build_reward_query_policy_ceiling_report_v2(
        space,
        official,
        terminal_item_generator_law=_uniform_law(space, panel),
        maximum_query_budget=0,
    )
    first_obj = first.as_obj()
    first_obj["reward_specification"]["exact_rule_weight"]["numerator"] = 999
    first_obj["information_set"]["query_policy_observes"].append("Official-Law-identity")

    second = build_reward_query_policy_ceiling_report_v2(
        space,
        official,
        terminal_item_generator_law=_uniform_law(space, panel),
        maximum_query_budget=0,
    )
    second_obj = second.as_obj()
    assert second_obj["reward_specification"]["exact_rule_weight"] == {
        "numerator": 1,
        "denominator": 4,
    }
    assert "Official-Law-identity" not in second_obj["information_set"]["query_policy_observes"]
    assert verify_reward_query_policy_ceiling_report_v2(second) is second


def test_terminal_classifier_updates_on_public_scene_selection_likelihood(
    catalog: RuleCatalog,
) -> None:
    first, second = catalog[0], catalog[1]
    false_true = next(scene for scene in range(SCENE_COUNT) if not first.truth[scene] and second.truth[scene])
    true_false = next(scene for scene in range(SCENE_COUNT) if first.truth[scene] and not second.truth[scene])
    space = VersionSpace(catalog, (first.index, second.index))
    law = build_terminal_item_generator_law_v2(
        space,
        {
            first.index: {false_true: Fraction(9, 10), true_false: Fraction(1, 10)},
            second.index: {false_true: Fraction(1, 10), true_false: Fraction(9, 10)},
        },
        public_derivation_attestation_digest=_PUBLIC_DERIVATION_DIGEST,
    )
    report = build_reward_query_policy_ceiling_report_v2(
        space,
        first,
        terminal_item_generator_law=law,
        maximum_query_budget=0,
    )
    terminal = report.composite_reward.official_terminal_decision

    assert terminal.expected_terminal_accuracy == Fraction(9, 10)
    assert terminal.official_expected_terminal_accuracy == Fraction(9, 10)
    assert len(terminal.terminal_item_policy) == 2
    by_scene = {item.scene_index: item for item in terminal.terminal_item_policy}
    assert by_scene[false_true].rejected_likelihood_sum == Fraction(9, 10)
    assert by_scene[false_true].accepted_likelihood_sum == Fraction(1, 10)
    assert not by_scene[false_true].predicted_accepted
    assert by_scene[false_true].marginal_scene_probability == Fraction(1, 2)


def test_ast_is_private_sibling_and_policy_does_not_depend_on_official(
    catalog: RuleCatalog,
) -> None:
    space = VersionSpace(catalog, (0, 1))
    law = _uniform_law(space, tuple(range(16)))
    reports = tuple(
        build_reward_query_policy_ceiling_report_v2(
            space,
            official,
            terminal_item_generator_law=law,
            maximum_query_budget=1,
        )
        for official in (catalog[0], catalog[1])
    )

    assert reports[0].policy_context_digest == reports[1].policy_context_digest
    assert reports[0].composite_reward.first_query == reports[1].composite_reward.first_query
    assert reports[0].report_binding_digest != reports[1].report_binding_digest
    for report in reports:
        obj = report.as_obj()
        assert obj["all_terminal_scenes_excluded_from_AST_choice"] is True
        assert obj["AST_private_sibling_locked_before_terminal_draw"] is True
        assert obj["AST_excluded_from_classification_replays"] is True
        observes = obj["information_set"]["terminal_classifier_observes"]
        assert all("AST" not in item for item in observes)
        assert (
            report.composite_reward.official_terminal_decision.as_obj()["terminal_item_information_unit"]
            == "sealed-pre-AST-transcript-plus-one-scene-only"
        )


def test_singleton_stops_now_with_perfect_reward(
    catalog: RuleCatalog,
    fixture_episode: dict[str, Any],
) -> None:
    official = catalog[int(fixture_episode["target_rule_id"][4:])]
    space = VersionSpace(catalog, (official.index,))
    panel = tuple(observation["scene_index"] for observation in fixture_episode["terminal"])
    report = build_reward_query_policy_ceiling_report_v2(
        space,
        official,
        terminal_item_generator_law=_uniform_law(space, panel),
        maximum_query_budget=6,
    )

    assert report.composite_reward.root_action_kind == "stop"
    assert report.composite_reward.optimal_expected_reward == 1
    assert report.composite_reward.official_path == ()
    assert report.identification_only.exact_identification_probability == 1
    assert report.root_informative_scene_count == 0


def test_six_query_horizon_solves_nontrivial_fixture(
    catalog: RuleCatalog,
    fixture_episode: dict[str, Any],
) -> None:
    space = catalog.version_space(
        (observation["scene_index"], observation["accepted"]) for observation in fixture_episode["opening"]
    )
    supported = set(build_supported_catalog_contract_v2().supported_indices)
    space = VersionSpace(catalog, tuple(index for index in space.indices if index in supported))
    official = catalog[int(fixture_episode["target_rule_id"][4:])]
    scenes = tuple(observation["scene_index"] for observation in fixture_episode["terminal"])
    report = build_reward_query_policy_ceiling_report_v2(
        space,
        official,
        terminal_item_generator_law=_uniform_law(space, scenes),
        maximum_query_budget=6,
    )

    assert len(space) == 5
    assert report.identification_only.exact_identification_probability == 1
    assert report.composite_reward.expected_exact_rule == 1
    assert report.composite_reward.expected_terminal_accuracy == 1
    assert 1 <= len(report.composite_reward.official_path) <= 6


def test_query_cost_is_exact(
    catalog: RuleCatalog,
) -> None:
    first, second = catalog[0], catalog[1]
    scenes = tuple(scene for scene in range(SCENE_COUNT) if first.truth[scene] is second.truth[scene])[:16]
    space = VersionSpace(catalog, (first.index, second.index))
    report = build_reward_query_policy_ceiling_report_v2(
        space,
        first,
        terminal_item_generator_law=_uniform_law(space, scenes),
        maximum_query_budget=1,
    )

    assert report.composite_reward.immediate_stop_expected_reward == Fraction(7, 8)
    assert report.composite_reward.optimal_expected_reward == Fraction(119, 120)
    assert report.composite_reward.optimal_expected_reward - Fraction(7, 8) == Fraction(7, 60)
    terminal = report.composite_reward.official_terminal_decision
    assert terminal.query_efficiency == Fraction(5, 6)
    assert terminal.official_expected_reward == Fraction(119, 120)


def test_target_paths_follow_official_feedback(
    catalog: RuleCatalog,
    fixture_episode: dict[str, Any],
) -> None:
    space, official, panel, exclusions = _small_context(catalog, fixture_episode)
    report = build_reward_query_policy_ceiling_report_v2(
        space,
        official,
        terminal_item_generator_law=_uniform_law(space, panel),
        excluded_query_scene_indices=reversed(exclusions),
        maximum_query_budget=2,
    )

    for step in report.composite_reward.official_path:
        assert step.official_accepted is official.truth[step.query.scene_index]
        assert official.rule_id in step.after_rule_ids
        assert step.selected_query_expected_reward > step.immediate_stop_expected_reward
    terminal = report.composite_reward.official_terminal_decision
    assert terminal.posterior_rule_ids == (official.rule_id,)
    assert terminal.submitted_rule_id == official.rule_id
    assert terminal.submitted_rule_ast == official.rule.as_obj()
    assert terminal.official_exact_rule
    assert terminal.official_expected_terminal_accuracy == 1


def test_report_is_canonical_deterministic_and_strictly_replayed(
    catalog: RuleCatalog,
    fixture_episode: dict[str, Any],
) -> None:
    space, official, panel, exclusions = _small_context(catalog, fixture_episode)
    law = _uniform_law(space, panel)
    forward = build_reward_query_policy_ceiling_report_v2(
        space,
        official,
        terminal_item_generator_law=law,
        excluded_query_scene_indices=exclusions,
        maximum_query_budget=1,
    )
    reverse = build_reward_query_policy_ceiling_report_v2(
        space,
        official,
        terminal_item_generator_law=law,
        excluded_query_scene_indices=reversed(exclusions),
        maximum_query_budget=1,
    )
    assert forward.as_obj() == reverse.as_obj()
    encoded = serialize_reward_query_policy_ceiling_report_v2(forward)
    parsed = parse_reward_query_policy_ceiling_report_v2(encoded)
    assert parsed.as_obj() == forward.as_obj()
    assert parsed.digest == forward.digest
    assert serialize_reward_query_policy_ceiling_report_v2(parsed) == encoded

    with pytest.raises(RewardQueryV2Error, match="not canonical"):
        parse_reward_query_policy_ceiling_report_v2(json.dumps(json.loads(encoded), indent=2))
    duplicate = encoded[:-1] + ',"schema_version":1}'
    with pytest.raises(RewardQueryV2Error, match="duplicate"):
        parse_reward_query_policy_ceiling_report_v2(duplicate)


@pytest.mark.parametrize(
    ("path", "mutation"),
    (
        (("authorization", "weight_updates_authorized"), True),
        (("supported_catalog_digest",), "0" * 64),
        (("terminal_item_generator_law", "supported_catalog_digest"), "0" * 64),
        (
            ("terminal_item_generator_law", "public_derivation_attestation_digest"),
            "0" * 64,
        ),
        (("reward_specification", "exact_rule_weight", "numerator"), 2),
        (("tie_rules", "equal_stop_query_value_tie"), "prefer-query"),
        (("composite_reward_ceiling", "optimal_expected_reward", "numerator"), 0),
        (("composite_reward_ceiling", "official_path"), "reverse"),
    ),
)
def test_parser_rejects_tampering_and_path_reordering(
    catalog: RuleCatalog,
    fixture_episode: dict[str, Any],
    path: tuple[str, ...],
    mutation: object,
) -> None:
    space, official, panel, exclusions = _small_context(catalog, fixture_episode)
    report = build_reward_query_policy_ceiling_report_v2(
        space,
        official,
        terminal_item_generator_law=_uniform_law(space, panel),
        excluded_query_scene_indices=exclusions,
        maximum_query_budget=2,
    )
    value = json.loads(serialize_reward_query_policy_ceiling_report_v2(report))
    cursor = value
    for part in path[:-1]:
        cursor = cursor[part]
    cursor[path[-1]] = list(reversed(cursor[path[-1]])) if mutation == "reverse" else mutation
    with pytest.raises(RewardQueryV2Error, match=r"inconsistent|tampered|mismatch"):
        parse_reward_query_policy_ceiling_report_v2(json.dumps(value, separators=(",", ":")))


def test_parser_rejects_generator_and_version_space_reordering(
    catalog: RuleCatalog,
    fixture_episode: dict[str, Any],
) -> None:
    space, official, panel, exclusions = _small_context(catalog, fixture_episode)
    report = build_reward_query_policy_ceiling_report_v2(
        space,
        official,
        terminal_item_generator_law=_uniform_law(space, panel),
        excluded_query_scene_indices=exclusions,
        maximum_query_budget=1,
    )
    encoded = serialize_reward_query_policy_ceiling_report_v2(report)
    for field in ("version_space_rules", "conditional_rules", "scene_probabilities"):
        value = json.loads(encoded)
        if field == "version_space_rules":
            target = value[field]
        elif field == "conditional_rules":
            target = value["terminal_item_generator_law"][field]
        else:
            target = value["terminal_item_generator_law"]["conditional_rules"][0][field]
        target[0], target[1] = target[1], target[0]
        with pytest.raises(RewardQueryV2Error, match=r"inconsistent|tampered|strictly increasing"):
            parse_reward_query_policy_ceiling_report_v2(json.dumps(value, separators=(",", ":")))


@pytest.mark.parametrize("budget", (-1, 7, True))
def test_invalid_budget_is_rejected(
    catalog: RuleCatalog,
    fixture_episode: dict[str, Any],
    budget: object,
) -> None:
    space, official, panel, _ = _small_context(catalog, fixture_episode)
    with pytest.raises(RewardQueryV2Error, match="maximum_query_budget"):
        build_reward_query_policy_ceiling_report_v2(
            space,
            official,
            terminal_item_generator_law=_uniform_law(space, panel),
            maximum_query_budget=cast(int, budget),
        )


def test_generator_law_is_fail_closed(
    catalog: RuleCatalog,
) -> None:
    space = VersionSpace(catalog, (0, 1))
    with pytest.raises(RewardQueryV2Error, match="exactly every"):
        build_terminal_item_generator_law_v2(
            space,
            {0: {0: Fraction(1)}},
            public_derivation_attestation_digest=_PUBLIC_DERIVATION_DIGEST,
        )
    with pytest.raises(RewardQueryV2Error, match="sum exactly"):
        build_terminal_item_generator_law_v2(
            space,
            {
                0: {0: Fraction(1, 2)},
                1: {0: Fraction(1)},
            },
            public_derivation_attestation_digest=_PUBLIC_DERIVATION_DIGEST,
        )
    with pytest.raises(RewardQueryV2Error, match="must lie"):
        build_terminal_item_generator_law_v2(
            space,
            {
                0: {SCENE_COUNT: Fraction(1)},
                1: {0: Fraction(1)},
            },
            public_derivation_attestation_digest=_PUBLIC_DERIVATION_DIGEST,
        )


def test_mixed_placard_piece_identity_g03r00197_is_rejected(
    catalog: RuleCatalog,
) -> None:
    mixed = catalog[197]
    assert mixed.rule_id == "g03r00197"
    space = VersionSpace(catalog, (mixed.index,))
    with pytest.raises(RewardQueryV2Error, match=r"g03r00197.*unsupported"):
        build_terminal_item_generator_law_v2(
            space,
            {mixed.index: {0: Fraction(1)}},
            public_derivation_attestation_digest=_PUBLIC_DERIVATION_DIGEST,
        )


@pytest.mark.parametrize(
    ("path", "replacement"),
    (
        (("schema_version",), True),
        (("uniform_prior_rule_count",), True),
        (("terminal_item_generator_law", "backend_item_count"), True),
        (("terminal_item_generator_law", "union_scene_support_count"), True),
    ),
)
def test_bool_integer_aliases_are_rejected_by_regenerated_bytes(
    catalog: RuleCatalog,
    path: tuple[str, ...],
    replacement: bool,
) -> None:
    space = VersionSpace(catalog, (0,))
    report = build_reward_query_policy_ceiling_report_v2(
        space,
        catalog[0],
        terminal_item_generator_law=_uniform_law(space, (0,)),
        maximum_query_budget=0,
    )
    value = json.loads(serialize_reward_query_policy_ceiling_report_v2(report))
    cursor = value
    for part in path[:-1]:
        cursor = cursor[part]
    cursor[path[-1]] = replacement
    with pytest.raises(RewardQueryV2Error):
        parse_reward_query_policy_ceiling_report_v2(
            json.dumps(value, separators=(",", ":")),
            require_canonical=False,
        )
