from __future__ import annotations

import ast
import itertools
import json
from dataclasses import replace
from functools import cache
from pathlib import Path
from typing import Any, cast

import pytest

from goalzendo_interactive import SCENE_COUNT, CatalogEntry, RuleCatalog, VersionSpace, build_rule_catalog
from goalzendo_interactive_v2 import (
    CHALLENGE_RESERVOIR_PANEL_COUNT,
    CHALLENGE_RESERVOIR_PANEL_SIZE,
    ChallengeQueryV2Error,
    build_minimum_challenge_set_v2,
    build_query_policy_ceiling_report_v2,
    parse_minimum_challenge_set_v2,
    parse_query_policy_ceiling_report_v2,
    select_common_untouched_panel_v2,
    serialize_minimum_challenge_set_v2,
    serialize_query_policy_ceiling_report_v2,
    verify_minimum_challenge_set_v2,
    verify_query_policy_ceiling_report_v2,
)
from goalzendo_interactive_v2.population_audit import build_supported_catalog_contract_v2


@pytest.fixture(scope="module")
def catalog() -> RuleCatalog:
    return build_rule_catalog()


@pytest.fixture(scope="module")
def fixture_episodes() -> list[dict[str, Any]]:
    path = (
        Path(__file__).parents[1] / "goalzendo_interactive" / "fixtures" / "g03-engine-small-fixture-v1.json"
    )
    value = json.loads(path.read_text(encoding="ascii"))
    if type(value) is not dict or type(value.get("episodes")) is not list:
        raise AssertionError("engineering fixture has no episode array")
    return cast(list[dict[str, Any]], value["episodes"])


def _context(
    catalog: RuleCatalog,
    episode: dict[str, Any],
) -> tuple[VersionSpace, CatalogEntry, tuple[int, ...]]:
    unfiltered = catalog.version_space(
        (observation["scene_index"], observation["accepted"]) for observation in episode["opening"]
    )
    supported = set(build_supported_catalog_contract_v2().supported_indices)
    space = VersionSpace(catalog, tuple(index for index in unfiltered.indices if index in supported))
    official = catalog[int(episode["target_rule_id"][4:])]
    exclusions = tuple(
        observation["scene_index"] for observation in (*episode["opening"], *episode["terminal"])
    )
    return space, official, exclusions


def _coverage_union(report: Any) -> int:
    by_scene = {
        candidate.representative_scene_index: candidate.coverage_mask
        for candidate in report.candidate_classes
    }
    union = 0
    for scene in report.selected_scene_indices:
        union |= by_scene[scene]
    return union


def test_minimum_challenge_is_exhaustive_and_brute_force_optimal(
    catalog: RuleCatalog,
    fixture_episodes: list[dict[str, Any]],
) -> None:
    space, official, exclusions = _context(catalog, fixture_episodes[0])
    report = build_minimum_challenge_set_v2(
        space,
        official,
        excluded_scene_indices=exclusions,
    )

    assert len(space) == 7
    assert report.optimum_cardinality == 2
    assert report.selected_scene_indices == (4606, 8619)
    assert report.as_obj()["supported_catalog_digest"] == (
        "e2c544786ccdcc85f5e424cb5a8927b61b5c6d13ab0cd70f5250d7f4a4eedd93"
    )
    assert not set(report.selected_scene_indices) & set(exclusions)
    assert _coverage_union(report) == report.full_coverage_mask
    assert sum(item.equivalent_scene_count for item in report.candidate_classes) + (
        report.nonseparating_legal_scene_count
    ) == SCENE_COUNT - len(exclusions)

    masks = tuple(item.coverage_mask for item in report.candidate_classes)
    for cardinality in range(report.optimum_cardinality):
        assert not any(
            _or_all(candidate_set) == report.full_coverage_mask
            for candidate_set in itertools.combinations(masks, cardinality)
        )
    assert report.dp_layers[-1].full_coverage_reachable
    assert not any(layer.full_coverage_reachable for layer in report.dp_layers[:-1])
    assert verify_minimum_challenge_set_v2(report) is report


def _or_all(values: tuple[int, ...]) -> int:
    result = 0
    for value in values:
        result |= value
    return result


def test_challenge_report_is_canonical_deterministic_and_strictly_replayed(
    catalog: RuleCatalog,
    fixture_episodes: list[dict[str, Any]],
) -> None:
    space, official, exclusions = _context(catalog, fixture_episodes[0])
    forward = build_minimum_challenge_set_v2(
        space,
        official,
        excluded_scene_indices=exclusions,
    )
    reverse = build_minimum_challenge_set_v2(
        space,
        official,
        excluded_scene_indices=reversed(exclusions),
    )
    assert forward.digest == reverse.digest
    encoded = serialize_minimum_challenge_set_v2(forward)
    parsed = parse_minimum_challenge_set_v2(encoded)
    assert parsed.as_obj() == forward.as_obj()
    assert parsed.digest == forward.digest
    assert serialize_minimum_challenge_set_v2(parsed) == encoded

    pretty = json.dumps(json.loads(encoded), indent=2)
    with pytest.raises(ChallengeQueryV2Error, match="not canonical"):
        parse_minimum_challenge_set_v2(pretty)
    duplicate = encoded[:-1] + ',"schema_version":1}'
    with pytest.raises(ChallengeQueryV2Error, match="duplicate"):
        parse_minimum_challenge_set_v2(duplicate)


@pytest.mark.parametrize(
    ("path", "replacement"),
    (
        (("authorization", "weight_updates_authorized"), True),
        (("candidate_classes", 0, "equivalent_scene_count"), 1),
        (("dp_optimality_certificate", "layers", 0, "reachable_coverage_count"), 2),
        (("selected_scenes", 0, "scene_index"), 0),
    ),
)
def test_challenge_parser_rejects_tampering(
    catalog: RuleCatalog,
    fixture_episodes: list[dict[str, Any]],
    path: tuple[str | int, ...],
    replacement: object,
) -> None:
    space, official, exclusions = _context(catalog, fixture_episodes[0])
    report = build_minimum_challenge_set_v2(
        space,
        official,
        excluded_scene_indices=exclusions,
    )
    value = json.loads(serialize_minimum_challenge_set_v2(report))
    cursor = value
    for part in path[:-1]:
        cursor = cursor[part]
    cursor[path[-1]] = replacement
    with pytest.raises(ChallengeQueryV2Error, match=r"inconsistent|tampered"):
        parse_minimum_challenge_set_v2(json.dumps(value, separators=(",", ":")))


def _independent_partition_masks(
    catalog: RuleCatalog,
    indices: tuple[int, ...],
    exclusions: tuple[int, ...],
) -> tuple[int, ...]:
    excluded = set(exclusions)
    patterns: set[int] = set()
    for scene in range(SCENE_COUNT):
        if scene in excluded:
            continue
        mask = 0
        for offset, index in enumerate(indices):
            if catalog[index].truth[scene]:
                mask |= 1 << offset
        patterns.add(mask)
    return tuple(sorted(patterns))


def _independent_policy_values(
    patterns: tuple[int, ...],
    root: int,
    maximum_budget: int,
) -> tuple[dict[tuple[int, int], int], dict[tuple[int, int], int]]:
    @cache
    def partitions(state: int) -> tuple[tuple[int, int], ...]:
        result: set[tuple[int, int]] = set()
        for labels in patterns:
            accepted = labels & state
            rejected = state ^ accepted
            if accepted and rejected:
                result.add((accepted, rejected) if accepted < rejected else (rejected, accepted))
        return tuple(sorted(result))

    expected: dict[tuple[int, int], int] = {}
    worst: dict[tuple[int, int], int] = {}

    @cache
    def solve_expected(state: int, budget: int) -> int:
        if state.bit_count() == 1:
            return 1
        if budget == 0:
            return 0
        result = max(
            solve_expected(left, budget - 1) + solve_expected(right, budget - 1)
            for left, right in partitions(state)
        )
        expected[(state, budget)] = result
        return result

    @cache
    def solve_worst(state: int, budget: int) -> int:
        if state.bit_count() == 1:
            return 1
        if budget == 0:
            return state.bit_count()
        result = min(
            max(solve_worst(left, budget - 1), solve_worst(right, budget - 1))
            for left, right in partitions(state)
        )
        worst[(state, budget)] = result
        return result

    for budget in range(maximum_budget + 1):
        expected[(root, budget)] = solve_expected(root, budget)
        worst[(root, budget)] = solve_worst(root, budget)
    return expected, worst


def test_query_ceiling_matches_independent_brute_force_on_small_state(
    catalog: RuleCatalog,
    fixture_episodes: list[dict[str, Any]],
) -> None:
    source_space, official, source_exclusions = _context(catalog, fixture_episodes[0])
    selected = tuple(sorted((official.index, *[i for i in source_space.indices if i != official.index][:4])))
    space = VersionSpace(catalog, selected)
    exclusions = source_exclusions[:3]
    report = build_query_policy_ceiling_report_v2(
        space,
        official,
        excluded_scene_indices=exclusions,
    )

    patterns = _independent_partition_masks(catalog, selected, exclusions)
    root = (1 << len(selected)) - 1
    expected, worst = _independent_policy_values(patterns, root, 6)
    for comparison in report.budgets:
        key = (root, comparison.budget)
        assert comparison.optimal.identified_rule_count == expected[key]
        assert comparison.minimax.worst_case_remaining_rule_count == worst[key]
        assert comparison.optimal.identified_rule_count >= comparison.greedy.identified_rule_count
        assert comparison.optimal.identified_rule_count >= comparison.minimax.identified_rule_count

    assert report.exact_expected_full_identification_budget == report.excluded_exact_minimax_depth
    assert verify_query_policy_ceiling_report_v2(report) is report
    impossible = replace(report, unrestricted_v1_minimax_depth=None)
    assert impossible.excluded_exact_minimax_depth is not None
    assert not impossible.as_obj()["minimax_cross_check"]["consistent"]


def test_query_report_paths_are_legal_disjoint_and_graph_free(
    catalog: RuleCatalog,
    fixture_episodes: list[dict[str, Any]],
) -> None:
    space, official, exclusions = _context(catalog, fixture_episodes[2])
    report = build_query_policy_ceiling_report_v2(
        space,
        official,
        excluded_scene_indices=exclusions,
    )
    assert report.excluded_exact_minimax_depth == 3
    assert report.greedy_full_identification_budget == 3
    assert report.as_obj()["supported_catalog_digest"] == (
        "e2c544786ccdcc85f5e424cb5a8927b61b5c6d13ab0cd70f5250d7f4a4eedd93"
    )
    assert report.root_partition_class_count < report.root_informative_scene_count
    for comparison in report.budgets:
        for policy in (comparison.optimal, comparison.greedy, comparison.minimax):
            assert not {step.query.scene_index for step in policy.official_path} & set(exclusions)
            assert sum(mass for _, mass in policy.terminal_state_size_mass) == len(space)
            if policy.first_query is not None:
                assert set(policy.first_query.rejected_rule_ids).isdisjoint(
                    policy.first_query.accepted_rule_ids
                )
                assert set(policy.first_query.rejected_rule_ids) | set(
                    policy.first_query.accepted_rule_ids
                ) == {entry.rule_id for entry in space}
    encoded = serialize_query_policy_ceiling_report_v2(report)
    # The graph-free contract has paths and root summaries, never a serialized
    # memoization DAG or state-node table.
    assert '"nodes"' not in encoded
    assert '"edges"' not in encoded
    assert parse_query_policy_ceiling_report_v2(encoded).digest == report.digest


@pytest.mark.parametrize(
    ("path", "replacement"),
    (
        (("authorization", "production_bank_materialized"), True),
        (("budgets", 2, "optimal_expected_identification", "identified_rule_count"), 99),
        (("budgets", 3, "greedy_information", "policy_digest"), "0" * 64),
        (("root_partition_classes_digest",), "f" * 64),
    ),
)
def test_query_parser_rejects_tampering(
    catalog: RuleCatalog,
    fixture_episodes: list[dict[str, Any]],
    path: tuple[str | int, ...],
    replacement: object,
) -> None:
    source, official, exclusions = _context(catalog, fixture_episodes[1])
    selected = tuple(sorted((official.index, *[i for i in source.indices if i != official.index][:3])))
    report = build_query_policy_ceiling_report_v2(
        VersionSpace(catalog, selected),
        official,
        excluded_scene_indices=exclusions[:2],
    )
    value = json.loads(serialize_query_policy_ceiling_report_v2(report))
    cursor = value
    for part in path[:-1]:
        cursor = cursor[part]
    cursor[path[-1]] = replacement
    with pytest.raises(ChallengeQueryV2Error, match=r"inconsistent|tampered"):
        parse_query_policy_ceiling_report_v2(json.dumps(value, separators=(",", ":")))


def test_eleven_panel_reservoir_has_exact_per_panel_cores_and_common_selection(
    catalog: RuleCatalog,
    fixture_episodes: list[dict[str, Any]],
) -> None:
    space, official, exclusions = _context(catalog, fixture_episodes[0])
    alternatives = tuple(index for index in space.indices if index != official.index)
    full_mask = (1 << len(alternatives)) - 1

    buckets: dict[int, list[int]] = {}
    excluded = set(exclusions)
    for scene in range(SCENE_COUNT):
        if scene in excluded:
            continue
        official_label = official.truth[scene]
        mask = 0
        for offset, index in enumerate(alternatives):
            if catalog[index].truth[scene] is not official_label:
                mask |= 1 << offset
        buckets.setdefault(mask, []).append(scene)

    covering_pair = max(
        (
            (min(len(buckets[left]), len(buckets[right])), left, right)
            for left in buckets
            for right in buckets
            if left < right and left | right == full_mask
        ),
    )
    left_mask, right_mask = covering_pair[1:]
    mandatory = tuple(
        scene
        for panel_index in range(CHALLENGE_RESERVOIR_PANEL_COUNT)
        for scene in (buckets[left_mask][panel_index], buckets[right_mask][panel_index])
    )
    assert len(set(mandatory)) == 2 * CHALLENGE_RESERVOIR_PANEL_COUNT
    reserved = set(mandatory)
    used: set[int] = set()
    panels: list[tuple[int, ...]] = []
    filler = (scene for scene in range(SCENE_COUNT) if scene not in excluded and scene not in reserved)
    for panel_index in range(CHALLENGE_RESERVOIR_PANEL_COUNT):
        panel = [mandatory[2 * panel_index], mandatory[2 * panel_index + 1]]
        used.update(panel)
        while len(panel) < CHALLENGE_RESERVOIR_PANEL_SIZE:
            scene = next(filler)
            if scene not in used:
                panel.append(scene)
                used.add(scene)
        panels.append(tuple(sorted(panel)))

    assert len(panels) == 11
    assert len(set().union(*(set(panel) for panel in panels))) == 11 * 16
    for panel_scenes in panels:
        first = build_minimum_challenge_set_v2(
            space,
            official,
            excluded_scene_indices=exclusions,
            candidate_scene_indices=panel_scenes,
        )
        second = build_minimum_challenge_set_v2(
            space,
            official,
            excluded_scene_indices=reversed(exclusions),
            candidate_scene_indices=reversed(panel_scenes),
        )
        assert first.optimum_cardinality == 2
        assert set(first.selected_scene_indices).issubset(panel_scenes)
        assert _coverage_union(first) == full_mask
        assert first.digest == second.digest
        assert (
            parse_minimum_challenge_set_v2(serialize_minimum_challenge_set_v2(first)).digest == first.digest
        )

    active_queries = tuple(panel[0] for panel in panels[:6])
    greedy_queries = tuple(panel[0] for panel in panels[6:10])
    assert (
        select_common_untouched_panel_v2(
            panels,
            active_query_indices=active_queries,
            greedy_query_indices=greedy_queries,
        )
        == 10
    )
    assert (
        select_common_untouched_panel_v2(
            panels,
            active_query_indices=(),
            greedy_query_indices=(),
        )
        == 0
    )

    # Hidden reservoir panels remain legal, unknown query actions.  The
    # common untouched panel is selected only after active and oracle paths.
    query = build_query_policy_ceiling_report_v2(
        space,
        official,
    )
    assert query.excluded_scene_indices == ()
    assert tuple(comparison.budget for comparison in query.budgets) == tuple(range(7))
    assert query.excluded_exact_minimax_depth == 3
    assert query.greedy_reference_recovery_within_ceiling

    # Duplicate actions consume turns but touch only one panel and therefore
    # remain valid scientific behavior for common-panel selection.
    assert (
        select_common_untouched_panel_v2(
            panels,
            active_query_indices=(panels[0][0],) * 6,
            greedy_query_indices=(panels[0][1],) * 4,
        )
        == 1
    )

    with pytest.raises(ChallengeQueryV2Error, match="pairwise"):
        select_common_untouched_panel_v2(
            (panels[0], panels[0], *panels[2:]),
        )
    with pytest.raises(ChallengeQueryV2Error, match="active query"):
        select_common_untouched_panel_v2(
            panels,
            active_query_indices=tuple(panel[0] for panel in panels[:7]),
        )


def test_all_twelve_legacy_episodes_after_v2_allowlist_filter_have_exact_reports(
    catalog: RuleCatalog,
    fixture_episodes: list[dict[str, Any]],
) -> None:
    challenge_cardinalities: list[int] = []
    exact_depths: list[int | None] = []
    greedy_depths: list[int | None] = []
    greedy_official_queries: list[int | None] = []
    for episode in fixture_episodes:
        space, official, exclusions = _context(catalog, episode)
        challenge = build_minimum_challenge_set_v2(
            space,
            official,
            excluded_scene_indices=exclusions,
        )
        query = build_query_policy_ceiling_report_v2(
            space,
            official,
            excluded_scene_indices=exclusions,
        )
        assert not set(challenge.selected_scene_indices) & set(exclusions)
        assert _coverage_union(challenge) == challenge.full_coverage_mask
        assert all(
            step.query.scene_index not in exclusions
            for comparison in query.budgets
            for policy in (comparison.optimal, comparison.greedy, comparison.minimax)
            for step in policy.official_path
        )
        challenge_cardinalities.append(challenge.optimum_cardinality)
        exact_depths.append(query.excluded_exact_minimax_depth)
        greedy_depths.append(query.greedy_full_identification_budget)
        greedy_official_queries.append(query.greedy_official_recovery_query_count)

    assert challenge_cardinalities == [2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    assert exact_depths == [3, 3, 3, 3, 3, 3, 1, 3, 3, 2, 3, 3]
    assert greedy_depths == [3, 3, 3, 3, 3, 3, 1, 3, 3, 2, 3, 3]
    assert all(query_count is not None and query_count <= 4 for query_count in greedy_official_queries)


def test_fail_closed_context_and_exclusion_checks(
    catalog: RuleCatalog,
    fixture_episodes: list[dict[str, Any]],
) -> None:
    space, official, _ = _context(catalog, fixture_episodes[0])
    with pytest.raises(ChallengeQueryV2Error, match="at most 16"):
        build_minimum_challenge_set_v2(VersionSpace(catalog, tuple(range(17))), catalog[0])
    with pytest.raises(ChallengeQueryV2Error, match="absent"):
        build_minimum_challenge_set_v2(
            VersionSpace(catalog, tuple(index for index in space.indices if index != official.index)),
            official,
        )
    with pytest.raises(ChallengeQueryV2Error, match="duplicates"):
        build_query_policy_ceiling_report_v2(space, official, excluded_scene_indices=(0, 0))
    with pytest.raises(ChallengeQueryV2Error, match="scene index"):
        build_minimum_challenge_set_v2(space, official, excluded_scene_indices=(SCENE_COUNT,))

    supported_official = catalog[84]
    unsupported_mixed = catalog[197]
    unsupported_space = VersionSpace(
        catalog,
        tuple(sorted((supported_official.index, unsupported_mixed.index))),
    )
    with pytest.raises(ChallengeQueryV2Error, match="supported allowlist"):
        build_minimum_challenge_set_v2(unsupported_space, supported_official)
    with pytest.raises(ChallengeQueryV2Error, match="supported allowlist"):
        build_query_policy_ceiling_report_v2(unsupported_space, supported_official)

    other = next(index for index in space.indices if index != official.index)
    pair = VersionSpace(catalog, tuple(sorted((official.index, other))))
    all_separators = tuple(
        scene for scene in range(SCENE_COUNT) if official.truth[scene] != catalog[other].truth[scene]
    )
    with pytest.raises(ChallengeQueryV2Error, match="remove every separator"):
        build_minimum_challenge_set_v2(
            pair,
            official,
            excluded_scene_indices=all_separators,
        )


def test_v2_module_uses_only_the_public_v1_package_surface() -> None:
    source_path = Path(__file__).parents[2] / "src" / "goalzendo_interactive_v2" / "challenge_query.py"
    tree = ast.parse(source_path.read_text(encoding="utf-8"), filename=str(source_path))
    imports = [node for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)]
    assert any(node.module == "goalzendo_interactive" for node in imports)
    assert not any(
        node.module is not None and node.module.startswith("goalzendo_interactive.") for node in imports
    )
