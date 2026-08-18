from __future__ import annotations

import hashlib
import json
import math
import time
from dataclasses import replace
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest

from goalzendo_interactive import VersionSpace, build_rule_catalog
from goalzendo_interactive.provenance import interactive_source_provenance
from goalzendo_interactive.rendering import RendererName
from goalzendo_interactive.schema import NONEMPTY_ARRANGEMENT_COUNT, SCENE_COUNT, scene_at
from goalzendo_interactive_v2.bank_audits import (
    build_rule_triple_bindings_batch_for_audit_v2,
)
from goalzendo_interactive_v2.population_audit import (
    build_supported_catalog_contract_v2,
)
from goalzendo_interactive_v2.statistical_leakage import (
    CatalogFrequencyTableV2,
    HypothesisCompleteMetaBlockV2,
    HypothesisCompleteTerminalBlockV2,
    MetaSurfaceFieldsV2,
    StatisticalLeakageConfigV2,
    StatisticalLeakageV2Error,
    TerminalPanelDrawsV2,
    VisibleOpeningObservationV2,
    build_meta_role_statistical_leakage_audit_v2,
    build_terminal_statistical_leakage_audit_v2,
    derive_catalog_frequency_table_v2,
    derive_rendered_static_prompt_length_bin_v2,
    historical_three_role_full_v0_excess_lower_bound_v2,
    parse_statistical_leakage_audit_v2,
    serialize_statistical_leakage_audit_v2,
    statistical_leakage_audit_v2_from_obj,
)

VARIANT_GROUPS = 384
DISTINCT_CLUSTERS = 12
TEST_CONFIG = StatisticalLeakageConfigV2(bootstrap_replicates=1_000)
TEST_RENDERERS: tuple[RendererName, ...] = (
    "train_compact",
    "train_positional",
    "train_tabletop",
)


def _fixture_opening() -> tuple[VisibleOpeningObservationV2, ...]:
    fixture_path = (
        Path(__file__).parents[1]
        / "goalzendo_interactive"
        / "fixtures"
        / "g03-engine-small-fixture-v1.json"
    )
    fixture = json.loads(fixture_path.read_text(encoding="ascii"))["episodes"][2]
    return tuple(
        VisibleOpeningObservationV2(item["scene_index"], item["accepted"])
        for item in fixture["opening"][1:]
    )


def _exact_space_and_binding(opening: tuple[VisibleOpeningObservationV2, ...]):
    catalog = build_rule_catalog()
    supported = set(build_supported_catalog_contract_v2().supported_indices)
    unfiltered = catalog.version_space((item.scene_index, item.accepted) for item in opening)
    exact_indices = tuple(index for index in unfiltered.indices if index in supported)
    assert exact_indices == (50, 51, 2356, 2377, 3358, 3386, 4006, 4007)
    binding = build_rule_triple_bindings_batch_for_audit_v2(
        ((catalog[50].rule_id, catalog[51].rule_id, catalog[2377].rule_id),)
    )[0]
    return VersionSpace(catalog, exact_indices), binding


def _surfaces(
    opening: tuple[VisibleOpeningObservationV2, ...],
    *,
    group: int,
    size: int,
    renderer: RendererName | None = None,
) -> tuple[MetaSurfaceFieldsV2, ...]:
    selected_renderer = renderer or TEST_RENDERERS[group % len(TEST_RENDERERS)]
    length_bin = derive_rendered_static_prompt_length_bin_v2(opening, selected_renderer)
    return tuple(
        MetaSurfaceFieldsV2(
            selected_renderer,
            (position + group) % size,
            (size - 1 - position + group) % size,
            length_bin,
            group // 32,
        )
        for position in range(size)
    )


@pytest.fixture(scope="module")
def variant_inputs() -> tuple[
    tuple[HypothesisCompleteMetaBlockV2, ...],
    CatalogFrequencyTableV2,
]:
    """384 openings sharing nine rows: distinct content, one derived lineage."""

    catalog = build_rule_catalog()
    base_opening = _fixture_opening()
    space, binding = _exact_space_and_binding(base_opening)
    used = {item.scene_index for item in base_opening}
    redundant: list[VisibleOpeningObservationV2] = []
    for scene_index in range(SCENE_COUNT):
        labels = {catalog[index].truth[scene_index] for index in space.indices}
        if labels == {True} and scene_index not in used:
            redundant.append(VisibleOpeningObservationV2(scene_index, True))
        if len(redundant) == VARIANT_GROUPS:
            break
    assert len(redundant) == VARIANT_GROUPS
    blocks = tuple(
        HypothesisCompleteMetaBlockV2(
            binding,
            space,
            (*base_opening, extra),
            _surfaces((*base_opening, extra), group=group, size=len(space)),
        )
        for group, extra in enumerate(redundant)
    )
    return blocks, derive_catalog_frequency_table_v2(blocks)


@pytest.fixture(scope="module")
def distinct_inputs() -> tuple[
    tuple[HypothesisCompleteMetaBlockV2, ...],
    CatalogFrequencyTableV2,
]:
    """Twelve openings sharing only eight rows, hence twelve derived clusters."""

    catalog = build_rule_catalog()
    base_nine = _fixture_opening()
    space, binding = _exact_space_and_binding(base_nine)
    removed = next(item for item in base_nine if item.scene_index == 8784)
    assert not removed.accepted
    base_eight = tuple(item for item in base_nine if item != removed)
    supported = set(build_supported_catalog_contract_v2().supported_indices)
    base_space = tuple(
        index
        for index in catalog.version_space(
            (item.scene_index, item.accepted) for item in base_eight
        ).indices
        if index in supported
    )
    used = {item.scene_index for item in base_eight}
    good_false: list[int] = []
    good_true: list[int] = []
    for scene_index in range(SCENE_COUNT):
        if scene_index in used:
            continue
        live_labels = {catalog[index].truth[scene_index] for index in space.indices}
        if live_labels == {True}:
            good_true.append(scene_index)
        elif live_labels == {False}:
            retained = tuple(index for index in base_space if not catalog[index].truth[scene_index])
            if retained == space.indices:
                good_false.append(scene_index)
    assert len(good_false) >= DISTINCT_CLUSTERS
    assert len(good_true) >= DISTINCT_CLUSTERS
    blocks: list[HypothesisCompleteMetaBlockV2] = []
    for group, (false_scene, true_scene) in enumerate(
        zip(good_false[:DISTINCT_CLUSTERS], good_true[:DISTINCT_CLUSTERS], strict=True)
    ):
        opening = (
            *base_eight,
            VisibleOpeningObservationV2(false_scene, False),
            VisibleOpeningObservationV2(true_scene, True),
        )
        blocks.append(
            HypothesisCompleteMetaBlockV2(
                binding,
                space,
                opening,
                _surfaces(opening, group=group, size=len(space)),
            )
        )
    result = tuple(blocks)
    return result, derive_catalog_frequency_table_v2(result)


@pytest.fixture(scope="module")
def variant_report(variant_inputs):
    blocks, frequencies = variant_inputs
    return build_meta_role_statistical_leakage_audit_v2(
        blocks,
        frequencies,
        config=TEST_CONFIG,
    )


@pytest.fixture(scope="module")
def distinct_report(distinct_inputs):
    blocks, frequencies = distinct_inputs
    return build_meta_role_statistical_leakage_audit_v2(
        blocks,
        frequencies,
        config=TEST_CONFIG,
    )


def test_384_one_row_variants_are_one_cluster_and_never_powered(variant_report) -> None:
    report = variant_report
    assert report.distinct_block_opening_group_count == VARIANT_GROUPS
    assert report.construction_cluster_count == 1
    assert len(report.construction_clusters[0].member_opening_digests) == VARIANT_GROUPS
    assert report.exact_conditional_balance.passed
    assert report.exact_structural_gate_passed
    assert not report.powered_group_requirement_met
    assert not report.statistical_interval_authorized
    assert not report.primary_statistical_gate_passed
    assert {view.full_v0_top_one.decision for view in report.views} == {"insufficient_data"}


def test_distinct_lineages_drive_folds_and_descriptive_underpower(distinct_report) -> None:
    report = distinct_report
    assert report.distinct_block_opening_group_count == DISTINCT_CLUSTERS
    assert report.construction_cluster_count == DISTINCT_CLUSTERS
    assert len(report.fold_assignments) == DISTINCT_CLUSTERS
    assert {item.fold for item in report.fold_assignments} == set(range(5))
    assert report.exact_structural_gate_passed
    assert not report.powered_group_requirement_met
    assert {view.full_v0_top_one.decision for view in report.views} == {"insufficient_data"}


def test_exact_prediction_evidence_recomputes_top_one_ba_and_bootstrap(distinct_report) -> None:
    report = distinct_report
    view = next(item for item in report.views if item.view_name == "full_model_visible")
    evidence = view.full_v0_prediction_evidence
    designated = view.designated_p_q_c_prediction_evidence
    assert evidence.status == designated.status == "estimated"
    assert all(
        pattern.candidate_indices == (50, 51, 2356, 2377, 3358, 3386, 4006, 4007)
        for pattern in evidence.patterns
    )
    assert all(
        pattern.candidate_indices == (50, 51, 2377)
        for pattern in designated.patterns
    )

    credits: list[Fraction] = []
    chances: list[Fraction] = []
    group_ids = tuple(cluster.construction_cluster_digest for cluster in report.construction_clusters)
    group_position = {group_id: position for position, group_id in enumerate(group_ids)}
    credit_by_group = np.zeros(len(group_ids))
    chance_by_group = np.zeros(len(group_ids))
    count_by_group = np.zeros(len(group_ids))
    ba_by_group = np.zeros((len(group_ids), 4))
    for row in evidence.episodes:
        pattern = evidence.patterns[row.pattern_index]
        scores = tuple(Fraction(*score) for score in pattern.exact_scores)
        maximum = max(scores)
        winners = tuple(
            index
            for index, score in zip(pattern.candidate_indices, scores, strict=True)
            if score == maximum
        )
        assert pattern.winner_indices == winners
        assert pattern.predicted_official == tuple(score > 0 for score in scores)
        credit = Fraction(1, len(winners)) if row.official_index in winners else Fraction()
        assert row.credit == (credit.numerator, credit.denominator)
        chance = Fraction(1, len(pattern.candidate_indices))
        credits.append(credit)
        chances.append(chance)
        position = group_position[row.construction_cluster_digest]
        credit_by_group[position] += float(credit)
        chance_by_group[position] += float(chance)
        count_by_group[position] += 1
        for candidate_index, predicted in zip(
            pattern.candidate_indices,
            pattern.predicted_official,
            strict=True,
        ):
            positive = candidate_index == row.official_index
            ba_by_group[position] += (
                int(positive and predicted),
                int(positive and not predicted),
                int(not positive and not predicted),
                int(not positive and predicted),
            )
    accuracy = float(sum(credits, Fraction()) / len(credits))
    chance_mean = float(sum(chances, Fraction()) / len(chances))
    assert view.full_v0_top_one.point_accuracy == accuracy
    assert view.full_v0_top_one.chance_mean == chance_mean
    totals = ba_by_group.sum(axis=0)
    point_ba = 0.5 * (
        totals[0] / (totals[0] + totals[1])
        + totals[2] / (totals[2] + totals[3])
    )
    assert view.macro_balanced_accuracy.point_balanced_accuracy == point_ba

    def seed(target: str) -> int:
        digest = hashlib.sha256(
            b"goalzendo-interactive-v2-statistical-leakage-bootstrap-v2\0"
            + bytes.fromhex(report.dataset_digest)
            + b"\0"
            + view.view_name.encode("ascii")
            + b"\0"
            + target.encode("ascii")
        ).digest()
        return int.from_bytes(digest[:8], "big")

    def interval(values: np.ndarray) -> tuple[float, float]:
        ordered = np.sort(values)
        return (
            float(ordered[math.floor((len(ordered) - 1) * 0.025)]),
            float(ordered[math.ceil((len(ordered) - 1) * 0.975)]),
        )

    rng = np.random.default_rng(seed("full_v0_official"))
    sampled = rng.integers(
        0,
        len(group_ids),
        size=(report.config.bootstrap_replicates, len(group_ids)),
    )
    top_values = (
        credit_by_group[sampled].sum(axis=1) - chance_by_group[sampled].sum(axis=1)
    ) / count_by_group[sampled].sum(axis=1)
    assert interval(top_values) == (
        view.full_v0_top_one.excess_interval_lower,
        view.full_v0_top_one.excess_interval_upper,
    )

    rng = np.random.default_rng(seed("one_v_rest_macro_balanced_accuracy"))
    sampled = rng.integers(
        0,
        len(group_ids),
        size=(report.config.bootstrap_replicates, len(group_ids)),
    )
    sums = ba_by_group[sampled].sum(axis=1)
    ba_values = 0.5 * (
        sums[:, 0] / (sums[:, 0] + sums[:, 1])
        + sums[:, 2] / (sums[:, 2] + sums[:, 3])
    )
    assert interval(ba_values) == (
        view.macro_balanced_accuracy.interval_lower,
        view.macro_balanced_accuracy.interval_upper,
    )


def test_unique_global_position_decoder_is_rejected_for_all_384_groups(variant_inputs) -> None:
    blocks, _ = variant_inputs
    rejected = 0
    for group, block in enumerate(blocks):
        surfaces = tuple(
            replace(surface, schedule_position=(group + 1) * len(block.version_space) + position)
            for position, surface in enumerate(block.rotation_surfaces)
        )
        with pytest.raises(StatisticalLeakageV2Error, match="bounded local permutation"):
            replace(block, rotation_surfaces=surfaces)
        rejected += 1
    assert rejected == VARIANT_GROUPS


def test_visible_opening_by_renderer_xor_is_rejected_for_all_384_groups(variant_inputs) -> None:
    blocks, _ = variant_inputs
    rejected = 0
    nuisance_ones = 0
    for block in blocks:
        scene = scene_at(block.opening[-1].scene_index)
        left = scene.piece_at("left")
        center = scene.piece_at("center")
        nuisance = int(
            (left is not None)
            ^ (left is not None and left.color == "red")
            ^ (center is not None and center.shape == "pyramid")
        )
        nuisance_ones += nuisance
        surfaces = tuple(
            MetaSurfaceFieldsV2(
                renderer := ("train_compact", "train_positional")[
                    nuisance ^ int(official_index == 50)
                ],
                old.schedule_position,
                old.request_position,
                derive_rendered_static_prompt_length_bin_v2(block.opening, renderer),
                old.bank_prefix,
            )
            for official_index, old in zip(
                block.version_space.indices,
                block.rotation_surfaces,
                strict=True,
            )
        )
        with pytest.raises(StatisticalLeakageV2Error, match="identical model-visible"):
            replace(block, rotation_surfaces=surfaces)
        rejected += 1
    assert nuisance_ones == VARIANT_GROUPS // 2
    assert rejected == VARIANT_GROUPS


def test_official_oblivious_terminal_has_exact_structural_balance(distinct_inputs) -> None:
    blocks, frequencies = distinct_inputs
    terminal = tuple(
        HypothesisCompleteTerminalBlockV2(
            block,
            (TerminalPanelDrawsV2(0, tuple((19,) for _ in block.version_space.indices)),),
        )
        for block in blocks
    )
    report = build_terminal_statistical_leakage_audit_v2(
        terminal,
        frequencies,
        config=TEST_CONFIG,
    )
    assert report.official_oblivious_shared_terminal_draws is True
    assert report.exact_conditional_balance.passed
    assert report.exact_structural_gate_passed
    assert not report.primary_statistical_gate_passed
    assert next(view for view in report.views if view.view_name == "selection_rank").model_visible is False


def test_official_dependent_terminal_fails_shared_and_exact_gates(distinct_inputs) -> None:
    blocks, frequencies = distinct_inputs
    terminal = tuple(
        HypothesisCompleteTerminalBlockV2(
            block,
            (
                TerminalPanelDrawsV2(
                    0,
                    tuple(
                        (0,) if index == 50 else (NONEMPTY_ARRANGEMENT_COUNT,)
                        for index in block.version_space.indices
                    ),
                ),
            ),
        )
        for block in blocks
    )
    report = build_terminal_statistical_leakage_audit_v2(
        terminal,
        frequencies,
        config=TEST_CONFIG,
    )
    assert report.official_oblivious_shared_terminal_draws is False
    assert not report.exact_conditional_balance.passed
    assert report.exact_conditional_balance.violating_cell_count > 0
    assert not report.exact_structural_gate_passed
    assert not report.primary_statistical_gate_passed


def test_practical_sixteen_item_resource_regression(distinct_inputs) -> None:
    blocks, frequencies = distinct_inputs
    terminal = tuple(
        HypothesisCompleteTerminalBlockV2(
            block,
            (
                TerminalPanelDrawsV2(
                    0,
                    tuple(tuple(range(16)) for _ in block.version_space.indices),
                ),
            ),
        )
        for block in blocks
    )
    started = time.perf_counter()
    report = build_terminal_statistical_leakage_audit_v2(
        terminal,
        frequencies,
        config=TEST_CONFIG,
    )
    elapsed = time.perf_counter() - started
    assert report.episode_sample_count == DISTINCT_CLUSTERS * 8 * 16
    assert report.candidate_row_count == DISTINCT_CLUSTERS * 8 * 16 * 8
    assert report.exact_structural_gate_passed
    assert elapsed < 30.0
    assert all(
        len(view.full_v0_prediction_evidence.patterns)
        < view.full_v0_prediction_evidence.nominal_observation_count
        for view in report.views
    )


def test_omitted_fitting_candidate_is_rejected_by_exact_v0_recomputation(variant_inputs) -> None:
    blocks, _ = variant_inputs
    block = blocks[0]
    replacement_index = next(
        index
        for index in build_supported_catalog_contract_v2().supported_indices
        if index not in block.version_space.indices
    )
    incomplete = VersionSpace(
        block.version_space.catalog,
        tuple(sorted((*block.version_space.indices[:-1], replacement_index))),
    )
    with pytest.raises(StatisticalLeakageV2Error, match="omits or adds"):
        HypothesisCompleteMetaBlockV2(
            block.binding,
            incomplete,
            block.opening,
            block.rotation_surfaces[:-1],
        )


def test_forged_global_frequency_table_is_rejected(variant_inputs) -> None:
    blocks, _ = variant_inputs
    contract = build_supported_catalog_contract_v2()
    forged = CatalogFrequencyTableV2(tuple(0 for _ in contract.supported_indices))
    with pytest.raises(StatisticalLeakageV2Error, match="frequenc"):
        build_meta_role_statistical_leakage_audit_v2(blocks, forged, config=TEST_CONFIG)


def test_opening_permutations_cannot_inflate_distinct_or_cluster_count(variant_inputs) -> None:
    blocks, _ = variant_inputs
    block = blocks[0]
    reversed_opening = tuple(reversed(block.opening))
    permuted = HypothesisCompleteMetaBlockV2(
        block.binding,
        block.version_space,
        reversed_opening,
        _surfaces(reversed_opening, group=0, size=len(block.version_space)),
    )
    assert permuted.block_opening_digest == block.block_opening_digest
    with pytest.raises(StatisticalLeakageV2Error, match="unique"):
        build_meta_role_statistical_leakage_audit_v2((block, permuted), config=TEST_CONFIG)


def test_opening_requires_exact_five_by_five_label_balance(variant_inputs) -> None:
    blocks, _ = variant_inputs
    block = blocks[0]
    opening = list(block.opening)
    accepted_position = next(index for index, item in enumerate(opening) if item.accepted)
    original = opening[accepted_position]
    opening[accepted_position] = VisibleOpeningObservationV2(original.scene_index, False)
    with pytest.raises(StatisticalLeakageV2Error, match="5 accepted and 5 rejected"):
        replace(block, opening=tuple(opening))


def test_unregistered_fixed_version_space_size_cannot_pass_training_gate() -> None:
    catalog = build_rule_catalog()
    contract = build_supported_catalog_contract_v2()
    fixture_path = (
        Path(__file__).parents[1]
        / "goalzendo_interactive"
        / "fixtures"
        / "g03-engine-small-fixture-v1.json"
    )
    fixture = json.loads(fixture_path.read_text(encoding="ascii"))["episodes"][0]
    base = tuple(
        VisibleOpeningObservationV2(item["scene_index"], item["accepted"])
        for item in fixture["opening"][1:]
    )
    unfiltered = catalog.version_space((item.scene_index, item.accepted) for item in base)
    supported = set(contract.supported_indices)
    indices = tuple(index for index in unfiltered.indices if index in supported)
    assert len(indices) == 11
    used = {item.scene_index for item in base}
    extra_scene = next(
        scene_index
        for scene_index in range(SCENE_COUNT)
        if scene_index not in used and all(catalog[index].truth[scene_index] for index in indices)
    )
    opening = (*base, VisibleOpeningObservationV2(extra_scene, True))
    binding = build_rule_triple_bindings_batch_for_audit_v2(
        ((catalog[50].rule_id, catalog[23].rule_id, catalog[444].rule_id),)
    )[0]
    block = HypothesisCompleteMetaBlockV2(
        binding,
        VersionSpace(catalog, indices),
        opening,
        _surfaces(opening, group=0, size=len(indices)),
    )
    with pytest.raises(StatisticalLeakageV2Error, match=r"n0 in \{8, 12, 16\}"):
        build_meta_role_statistical_leakage_audit_v2((block,), config=TEST_CONFIG)


def test_report_object_cannot_claim_an_unregistered_fixed_size(variant_report) -> None:
    with pytest.raises(StatisticalLeakageV2Error, match=r"n0 in \{8, 12, 16\}"):
        replace(variant_report, fixed_version_space_size=9)


def test_grouping_is_deterministic_and_never_splits_lineages(distinct_inputs, distinct_report) -> None:
    blocks, frequencies = distinct_inputs
    reversed_report = build_meta_role_statistical_leakage_audit_v2(
        reversed(blocks),
        frequencies,
        config=TEST_CONFIG,
    )
    assert reversed_report.digest == distinct_report.digest
    assert reversed_report.fold_assignments == distinct_report.fold_assignments
    assert reversed_report.construction_clusters == distinct_report.construction_clusters


def test_report_parser_rejects_tamper_reorder_and_boolean_alias(
    distinct_report,
) -> None:
    text = serialize_statistical_leakage_audit_v2(distinct_report)
    assert (
        parse_statistical_leakage_audit_v2(text, expected_digest=distinct_report.digest)
        == distinct_report
    )

    tampered = json.loads(text)
    tampered["views"][0]["full_v0_top_one"]["point_accuracy"] += 0.01
    with pytest.raises(StatisticalLeakageV2Error):
        parse_statistical_leakage_audit_v2(
            json.dumps(tampered, ensure_ascii=True, separators=(",", ":")),
            expected_digest=distinct_report.digest,
        )

    boolean_alias = json.loads(text)
    boolean_alias["config"]["fold_count"] = True
    with pytest.raises(StatisticalLeakageV2Error):
        parse_statistical_leakage_audit_v2(
            json.dumps(boolean_alias, ensure_ascii=True, separators=(",", ":")),
            expected_digest=distinct_report.digest,
        )

    ordinary = json.loads(text)
    reordered = {"report_kind": ordinary["report_kind"], "schema_version": ordinary["schema_version"]}
    reordered.update(
        (key, value)
        for key, value in ordinary.items()
        if key not in {"report_kind", "schema_version"}
    )
    with pytest.raises(StatisticalLeakageV2Error):
        parse_statistical_leakage_audit_v2(
            json.dumps(reordered, ensure_ascii=True, separators=(",", ":")),
            expected_digest=distinct_report.digest,
        )

    forged = replace(distinct_report, frequency_table_digest="0" * 64)
    forged_text = serialize_statistical_leakage_audit_v2(forged)
    with pytest.raises(StatisticalLeakageV2Error, match="externally expected"):
        parse_statistical_leakage_audit_v2(
            forged_text,
            expected_digest=distinct_report.digest,
        )

    prediction_tamper = json.loads(text)
    pattern = prediction_tamper["views"][0]["full_v0_prediction_evidence"]["patterns"][0]
    pattern["predicted_official"][0] = not pattern["predicted_official"][0]
    with pytest.raises(StatisticalLeakageV2Error, match="prediction bits"):
        parse_statistical_leakage_audit_v2(
            json.dumps(prediction_tamper, ensure_ascii=True, separators=(",", ":")),
            expected_digest=distinct_report.digest,
        )


def test_report_objects_cannot_mutate_canonical_contract_across_reports(distinct_report) -> None:
    first = distinct_report.as_obj()
    forbidden = first["feature_contract"]["forbidden_channels"]
    assert type(forbidden) is list
    forbidden.append("forged_cross_report_channel")

    second = distinct_report.as_obj()
    assert "forged_cross_report_channel" not in second["feature_contract"]["forbidden_channels"]
    assert (
        statistical_leakage_audit_v2_from_obj(
            second,
            expected_digest=distinct_report.digest,
        )
        == distinct_report
    )


def test_strict_input_types_and_historical_infeasibility_witness() -> None:
    with pytest.raises(StatisticalLeakageV2Error):
        MetaSurfaceFieldsV2("train_compact", True, 0, 0, 0)
    with pytest.raises(StatisticalLeakageV2Error):
        StatisticalLeakageConfigV2(fold_count=4)
    assert historical_three_role_full_v0_excess_lower_bound_v2(8) == Fraction(5, 24)
    contract = build_supported_catalog_contract_v2()
    assert interactive_source_provenance().fingerprint == (
        "24b6d1cc60c09be3b6bbab22d7b7a5b09fef1c4250dd8eb6dc0a9d323aed0ada"
    )
    assert contract.source_catalog_digest == (
        "a796ef24d4e0eb2a2e129e12ee9cc3261c82b578f415e554d6608feafc9ae2d0"
    )
    assert contract.supported_catalog_digest == (
        "e2c544786ccdcc85f5e424cb5a8927b61b5c6d13ab0cd70f5250d7f4a4eedd93"
    )
    assert historical_three_role_full_v0_excess_lower_bound_v2(16) == Fraction(13, 48)
