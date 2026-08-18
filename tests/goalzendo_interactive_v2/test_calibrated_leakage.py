from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

import goalzendo_interactive_v2.calibrated_leakage as calibrated
import goalzendo_interactive_v2.statistical_leakage as statistical
from goalzendo_interactive import VersionSpace, build_rule_catalog
from goalzendo_interactive.rendering import RendererName
from goalzendo_interactive.schema import SCENE_COUNT, scene_at
from goalzendo_interactive_v2.bank_audits import (
    build_rule_triple_bindings_batch_for_audit_v2,
)
from goalzendo_interactive_v2.calibrated_leakage import (
    CalibratedLeakageAuditReportV2,
    CalibratedLeakageConfigV2,
    CalibratedLeakageV2Error,
    CalibrationSplitPlanV2,
    build_calibrated_meta_role_leakage_audit_v2,
    build_calibration_split_plan_v2,
    parse_calibrated_leakage_audit_v2,
    parse_calibration_split_plan_v2,
    serialize_calibrated_leakage_audit_v2,
    serialize_calibration_split_plan_v2,
    verify_calibrated_leakage_audit_against_blocks_v2,
)
from goalzendo_interactive_v2.population_audit import (
    RuleTripleBindingV2,
    build_supported_catalog_contract_v2,
)
from goalzendo_interactive_v2.statistical_leakage import (
    HypothesisCompleteMetaBlockV2,
    MetaSurfaceFieldsV2,
    StatisticalLeakageV2Error,
    VisibleOpeningObservationV2,
    derive_rendered_static_prompt_length_bin_v2,
)

TEST_CLUSTER_COUNT_PER_ROLE = 6
TEST_CONFIG = CalibratedLeakageConfigV2(
    bootstrap_replicates=1_000,
    minimum_calibration_clusters=TEST_CLUSTER_COUNT_PER_ROLE,
    minimum_evaluation_clusters=TEST_CLUSTER_COUNT_PER_ROLE,
)
TEST_RENDERERS: tuple[RendererName, ...] = (
    "train_compact",
    "train_positional",
    "train_tabletop",
)
BlockSplit = tuple[
    tuple[HypothesisCompleteMetaBlockV2, ...],
    tuple[HypothesisCompleteMetaBlockV2, ...],
]


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


def _space_and_binding(
    opening: tuple[VisibleOpeningObservationV2, ...],
) -> tuple[VersionSpace, RuleTripleBindingV2]:
    catalog = build_rule_catalog()
    supported = set(build_supported_catalog_contract_v2().supported_indices)
    exact = tuple(
        index
        for index in catalog.version_space(
            (item.scene_index, item.accepted) for item in opening
        ).indices
        if index in supported
    )
    assert exact == (50, 51, 2356, 2377, 3358, 3386, 4006, 4007)
    binding = build_rule_triple_bindings_batch_for_audit_v2(
        ((catalog[50].rule_id, catalog[51].rule_id, catalog[2377].rule_id),)
    )[0]
    return VersionSpace(catalog, exact), binding


def _surfaces(
    opening: tuple[VisibleOpeningObservationV2, ...],
    *,
    group: int,
    size: int,
    canonical_position_encoder: bool = True,
) -> tuple[MetaSurfaceFieldsV2, ...]:
    renderer = TEST_RENDERERS[group % len(TEST_RENDERERS)]
    length_bin = derive_rendered_static_prompt_length_bin_v2(opening, renderer)
    return tuple(
        MetaSurfaceFieldsV2(
            renderer,
            position if canonical_position_encoder else (position + group) % size,
            size - 1 - position,
            length_bin,
            group // 4,
        )
        for position in range(size)
    )


@pytest.fixture(scope="module")
def split_blocks() -> tuple[
    tuple[HypothesisCompleteMetaBlockV2, ...],
    tuple[HypothesisCompleteMetaBlockV2, ...],
]:
    """Twelve exact openings with no shared nine-of-ten construction projection."""

    catalog = build_rule_catalog()
    base_nine = _fixture_opening()
    space, binding = _space_and_binding(base_nine)
    removed = next(item for item in base_nine if item.scene_index == 8784)
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
        labels = {catalog[index].truth[scene_index] for index in space.indices}
        if labels == {True}:
            good_true.append(scene_index)
        elif labels == {False}:
            retained = tuple(index for index in base_space if not catalog[index].truth[scene_index])
            if retained == space.indices:
                good_false.append(scene_index)
        if len(good_false) >= 12 and len(good_true) >= 12:
            break
    assert len(good_false) >= 12 and len(good_true) >= 12
    blocks = []
    for group, (false_scene, true_scene) in enumerate(
        zip(good_false[:12], good_true[:12], strict=True)
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
    ordered = tuple(sorted(blocks, key=lambda block: block.block_opening_digest))
    return ordered[:6], ordered[6:]


@pytest.fixture(scope="module")
def split_plan(split_blocks: BlockSplit) -> CalibrationSplitPlanV2:
    return build_calibration_split_plan_v2(*split_blocks)


@pytest.fixture(scope="module")
def calibrated_report(
    split_blocks: BlockSplit,
    split_plan: CalibrationSplitPlanV2,
) -> CalibratedLeakageAuditReportV2:
    return build_calibrated_meta_role_leakage_audit_v2(
        split_blocks[0],
        split_blocks[1],
        split_plan,
        expected_split_plan_digest=split_plan.digest,
        config=TEST_CONFIG,
    )


def test_prospective_split_binds_disjoint_exact_populations(
    split_plan: CalibrationSplitPlanV2,
) -> None:
    plan = split_plan
    calibration = plan.calibration_population
    evaluation = plan.evaluation_population
    assert calibration.construction_cluster_count == TEST_CLUSTER_COUNT_PER_ROLE
    assert evaluation.construction_cluster_count == TEST_CLUSTER_COUNT_PER_ROLE
    assert calibration.exact_conditional_balance.passed
    assert evaluation.exact_conditional_balance.passed
    assert {
        item.construction_cluster_digest for item in calibration.construction_clusters
    }.isdisjoint(
        item.construction_cluster_digest for item in evaluation.construction_clusters
    )
    assert all(len(item.candidate_indices) == 8 for item in (*calibration.blocks, *evaluation.blocks))
    assert plan.as_obj()["content_addressing_establishes_temporal_priority"] is False
    assert plan.as_obj()["full_manifest_surface_bound"] is False
    assert plan.as_obj()["frozen_manifest_runtime_bridge_required"] is True


def test_classifier_is_fit_only_on_calibration_and_scored_once_on_evaluation(
    calibrated_report: CalibratedLeakageAuditReportV2,
) -> None:
    report = calibrated_report
    assert report.exact_conditional_surface_balance_passed
    assert not report.config.registered_population_minima_used
    assert not report.calibration_cluster_requirement_met
    assert not report.evaluation_cluster_requirement_met
    assert not report.power_runtime_prerequisites_passed
    assert not report.statistical_interval_authorized
    assert not report.primary_statistical_gate_passed
    assert report.as_obj()["model_execution_authorized"] is False
    assert report.as_obj()["weight_updates_authorized"] is False
    assert report.as_obj()["full_manifest_surface_bound"] is False
    assert report.as_obj()["frozen_manifest_runtime_bridge_verified"] is False
    assert report.as_obj()["standalone_parser_block_feature_rederivation_verified"] is False
    assert report.as_obj()["exact_block_backed_rebuild_required_for_production"] is True
    for view in report.views:
        assert (
            view.classifier.calibration_population_digest
            == report.split_plan.calibration_population.digest
        )
        assert (
            view.prediction_evidence.evaluation_population_digest
            == report.split_plan.evaluation_population.digest
        )
        assert view.estimate.decision == "insufficient_prerequisites"
        assert "finite_untouched_evaluation_population" in view.estimate.estimand_name
        assert "not a universal confidence interval" in view.estimate.interval_scope
        assert len(
            {row.candidate_prediction_digest for row in view.prediction_evidence.episodes}
        ) == len(view.prediction_evidence.episodes)


def test_exact_visible_balance_is_chance_but_executor_position_encoder_is_exposed(
    calibrated_report: CalibratedLeakageAuditReportV2,
) -> None:
    by_name = {view.view_name: view for view in calibrated_report.views}
    visible = by_name["full_model_visible"].estimate
    executor = by_name["full_executor_diagnostic"].estimate
    schedule = by_name["schedule_position"].estimate
    assert visible.top_one_accuracy == visible.full_v0_chance == (1, 8)
    assert visible.top_one_excess == (0, 1)
    assert visible.macro_balanced_accuracy == (1, 2)
    assert visible.descriptive_equivalence_passed
    assert executor.top_one_excess[0] > 0
    assert schedule.top_one_excess[0] > 0
    assert not executor.descriptive_equivalence_passed
    assert not schedule.descriptive_equivalence_passed


def test_exact_block_backed_verifier_rebuilds_every_feature_row(
    calibrated_report: CalibratedLeakageAuditReportV2,
    split_blocks: BlockSplit,
) -> None:
    assert (
        verify_calibrated_leakage_audit_against_blocks_v2(
            calibrated_report,
            split_blocks[0],
            split_blocks[1],
            expected_report_digest=calibrated_report.digest,
            expected_split_plan_digest=calibrated_report.split_plan.digest,
        )
        == calibrated_report
    )


def test_unique_global_position_bypass_is_rejected_before_calibration(
    split_blocks: BlockSplit,
) -> None:
    block = split_blocks[0][0]
    size = len(block.version_space)
    surfaces = tuple(
        replace(surface, schedule_position=size + position)
        for position, surface in enumerate(block.rotation_surfaces)
    )
    with pytest.raises(StatisticalLeakageV2Error, match="bounded local permutation"):
        replace(block, rotation_surfaces=surfaces)


def test_opening_by_renderer_xor_bypass_is_rejected_before_calibration(
    split_blocks: BlockSplit,
) -> None:
    block = split_blocks[0][0]
    scene = scene_at(block.opening[-1].scene_index)
    left = scene.piece_at("left")
    nuisance = int(left is not None) ^ int(left is not None and left.color == "red")
    surfaces = tuple(
        MetaSurfaceFieldsV2(
            renderer := ("train_compact", "train_positional")[
                nuisance ^ int(official_index == block.version_space.indices[0])
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


def test_revalidated_forged_object_cannot_bypass_surface_gate(
    split_blocks: BlockSplit,
) -> None:
    original = split_blocks[0][0]
    renderer: RendererName = "train_compact"
    alternate: RendererName = "train_positional"
    forged_surfaces = tuple(
        replace(
            surface,
            renderer=renderer if position % 2 == 0 else alternate,
            token_length_bin=derive_rendered_static_prompt_length_bin_v2(
                original.opening,
                renderer if position % 2 == 0 else alternate,
            ),
        )
        for position, surface in enumerate(original.rotation_surfaces)
    )
    forged = object.__new__(HypothesisCompleteMetaBlockV2)
    object.__setattr__(forged, "binding", original.binding)
    object.__setattr__(forged, "version_space", original.version_space)
    object.__setattr__(forged, "opening", original.opening)
    object.__setattr__(forged, "rotation_surfaces", forged_surfaces)
    calibration = tuple(
        sorted((forged, *split_blocks[0][1:]), key=lambda block: block.block_opening_digest)
    )
    with pytest.raises(StatisticalLeakageV2Error, match="identical model-visible"):
        build_calibration_split_plan_v2(calibration, split_blocks[1])


def test_cross_population_nine_of_ten_lineage_overlap_is_rejected() -> None:
    catalog = build_rule_catalog()
    base_nine = _fixture_opening()
    space, binding = _space_and_binding(base_nine)
    used = {item.scene_index for item in base_nine}
    extras: list[int] = []
    for scene_index in range(SCENE_COUNT):
        if scene_index not in used and {catalog[index].truth[scene_index] for index in space.indices} == {
            True
        }:
            extras.append(scene_index)
        if len(extras) == 2:
            break
    blocks = []
    for group, scene_index in enumerate(extras):
        opening = (*base_nine, VisibleOpeningObservationV2(scene_index, True))
        blocks.append(
            HypothesisCompleteMetaBlockV2(
                binding,
                space,
                opening,
                _surfaces(opening, group=group, size=len(space)),
            )
        )
    with pytest.raises(CalibratedLeakageV2Error, match="construction lineages overlap"):
        build_calibration_split_plan_v2((blocks[0],), (blocks[1],))


def test_reordering_overlap_and_insufficient_clusters_fail_closed(
    split_blocks: BlockSplit,
    split_plan: CalibrationSplitPlanV2,
) -> None:
    with pytest.raises(CalibratedLeakageV2Error, match="ordered by opening digest"):
        build_calibration_split_plan_v2(tuple(reversed(split_blocks[0])), split_blocks[1])
    with pytest.raises(CalibratedLeakageV2Error, match="openings overlap"):
        build_calibration_split_plan_v2(split_blocks[0], split_blocks[0])
    strict = CalibratedLeakageConfigV2(
        bootstrap_replicates=1_000,
        minimum_calibration_clusters=7,
        minimum_evaluation_clusters=6,
    )
    with pytest.raises(CalibratedLeakageV2Error, match="insufficient calibration"):
        build_calibrated_meta_role_leakage_audit_v2(
            split_blocks[0],
            split_blocks[1],
            split_plan,
            expected_split_plan_digest=split_plan.digest,
            config=strict,
        )
    strict_evaluation = CalibratedLeakageConfigV2(
        bootstrap_replicates=1_000,
        minimum_calibration_clusters=6,
        minimum_evaluation_clusters=7,
    )
    with pytest.raises(CalibratedLeakageV2Error, match="insufficient evaluation"):
        build_calibrated_meta_role_leakage_audit_v2(
            split_blocks[0],
            split_blocks[1],
            split_plan,
            expected_split_plan_digest=split_plan.digest,
            config=strict_evaluation,
        )


def test_plan_parser_requires_external_digest_and_rejects_tamper_reorder_bool_and_provenance(
    split_plan: CalibrationSplitPlanV2,
) -> None:
    text = serialize_calibration_split_plan_v2(split_plan)
    assert parse_calibration_split_plan_v2(text, expected_digest=split_plan.digest) == split_plan
    ordinary = json.loads(text)

    tampered = json.loads(text)
    tampered["evaluation_population"]["blocks"][0]["surface_digest"] = "0" * 64
    with pytest.raises(CalibratedLeakageV2Error):
        parse_calibration_split_plan_v2(
            json.dumps(tampered, ensure_ascii=True, separators=(",", ":")),
            expected_digest=split_plan.digest,
        )

    boolean_alias = json.loads(text)
    boolean_alias["calibration_population"]["fixed_version_space_size"] = True
    with pytest.raises(CalibratedLeakageV2Error):
        parse_calibration_split_plan_v2(
            json.dumps(boolean_alias, ensure_ascii=True, separators=(",", ":")),
            expected_digest=split_plan.digest,
        )

    reordered = {"plan_kind": ordinary["plan_kind"], "schema_version": ordinary["schema_version"]}
    reordered.update(
        (key, value) for key, value in ordinary.items() if key not in {"plan_kind", "schema_version"}
    )
    with pytest.raises(CalibratedLeakageV2Error, match="reordered"):
        parse_calibration_split_plan_v2(
            json.dumps(reordered, ensure_ascii=True, separators=(",", ":")),
            expected_digest=split_plan.digest,
        )

    nested_reordered = json.loads(text)
    nested_reordered["classifier_contract"] = dict(
        reversed(tuple(nested_reordered["classifier_contract"].items()))
    )
    with pytest.raises(CalibratedLeakageV2Error, match=r"classifier_contract.*reordered"):
        parse_calibration_split_plan_v2(
            json.dumps(nested_reordered, ensure_ascii=True, separators=(",", ":")),
            expected_digest=split_plan.digest,
        )

    provenance = json.loads(text)
    provenance["source_binding"]["base_statistical_source_sha256"] = "0" * 64
    with pytest.raises(CalibratedLeakageV2Error, match="provenance"):
        parse_calibration_split_plan_v2(
            json.dumps(provenance, ensure_ascii=True, separators=(",", ":")),
            expected_digest=split_plan.digest,
        )


def test_plan_parser_rejects_one_opening_forged_into_two_canonical_clusters(
    split_plan: CalibrationSplitPlanV2,
) -> None:
    forged = json.loads(serialize_calibration_split_plan_v2(split_plan))
    population = forged["evaluation_population"]
    clusters = population["construction_clusters"]
    duplicate_opening = clusters[0]["member_opening_digests"][0]
    second_members = sorted(
        {*clusters[1]["member_opening_digests"], duplicate_opening}
    )
    clusters[1]["member_opening_digests"] = second_members
    clusters[1]["construction_cluster_digest"] = statistical._json_digest(
        {"member_opening_digests": second_members},
        domain="goalzendo-interactive-v2-construction-cluster-v2",
    )
    population["construction_clusters"] = sorted(
        clusters,
        key=lambda item: item["construction_cluster_digest"],
    )
    with pytest.raises(CalibratedLeakageV2Error, match="exactly one construction cluster"):
        parse_calibration_split_plan_v2(
            json.dumps(forged, ensure_ascii=True, separators=(",", ":")),
            expected_digest=split_plan.digest,
        )


def test_report_parser_replays_predictions_and_rejects_tamper_bool_and_expected_digest(
    calibrated_report: CalibratedLeakageAuditReportV2,
) -> None:
    text = serialize_calibrated_leakage_audit_v2(calibrated_report)
    parsed = parse_calibrated_leakage_audit_v2(text, expected_digest=calibrated_report.digest)
    assert parsed == calibrated_report

    prediction = json.loads(text)
    pattern = prediction["views"][0]["prediction_evidence"]["patterns"][0]
    pattern["predicted_official"][0] = not pattern["predicted_official"][0]
    with pytest.raises((CalibratedLeakageV2Error, StatisticalLeakageV2Error)):
        parse_calibrated_leakage_audit_v2(
            json.dumps(prediction, ensure_ascii=True, separators=(",", ":")),
            expected_digest=calibrated_report.digest,
        )

    boolean_alias = json.loads(text)
    boolean_alias["config"]["bootstrap_replicates"] = True
    with pytest.raises(CalibratedLeakageV2Error):
        parse_calibrated_leakage_audit_v2(
            json.dumps(boolean_alias, ensure_ascii=True, separators=(",", ":")),
            expected_digest=calibrated_report.digest,
        )

    prediction_digest = json.loads(text)
    prediction_digest["views"][0]["prediction_evidence"]["episodes"][0][
        "candidate_prediction_digest"
    ] = "0" * 64
    with pytest.raises(CalibratedLeakageV2Error, match="per-candidate prediction digest"):
        parse_calibrated_leakage_audit_v2(
            json.dumps(prediction_digest, ensure_ascii=True, separators=(",", ":")),
            expected_digest=calibrated_report.digest,
        )

    ordinary = json.loads(text)
    reordered = {
        "report_kind": ordinary["report_kind"],
        "schema_version": ordinary["schema_version"],
    }
    reordered.update(
        (key, value) for key, value in ordinary.items() if key not in {"report_kind", "schema_version"}
    )
    with pytest.raises(CalibratedLeakageV2Error, match="reordered"):
        parse_calibrated_leakage_audit_v2(
            json.dumps(reordered, ensure_ascii=True, separators=(",", ":")),
            expected_digest=calibrated_report.digest,
        )

    nested_authorization = json.loads(text)
    nested_authorization["authorization"] = dict(
        reversed(tuple(nested_authorization["authorization"].items()))
    )
    with pytest.raises(CalibratedLeakageV2Error, match=r"authorization.*reordered"):
        parse_calibrated_leakage_audit_v2(
            json.dumps(nested_authorization, ensure_ascii=True, separators=(",", ":")),
            expected_digest=calibrated_report.digest,
        )

    with pytest.raises(CalibratedLeakageV2Error, match="externally expected"):
        parse_calibrated_leakage_audit_v2(text, expected_digest="0" * 64)


def test_parser_rejects_rehashed_48_over_6_to_49_over_6_cell_with_stale_scores(
    calibrated_report: CalibratedLeakageAuditReportV2,
) -> None:
    forged = json.loads(serialize_calibrated_leakage_audit_v2(calibrated_report))
    view = next(item for item in forged["views"] if item["view_name"] == "catalog_structure")
    classifier = view["classifier"]
    evidence = view["prediction_evidence"]
    evaluation_tokens = {
        token
        for episode in evidence["episodes"]
        for feature_row in episode["candidate_feature_tokens"]
        for token in feature_row
    }
    cell = next(
        item
        for item in classifier["cells"]
        if item["row_count"] == 48
        and item["positive_count"] == 6
        and item["token"] in evaluation_tokens
    )
    assert (cell["row_count"], cell["positive_count"]) == (48, 6)
    cell["row_count"] = 49
    classifier_unsigned = {
        key: value for key, value in classifier.items() if key != "frozen_classifier_digest"
    }
    classifier_digest = calibrated._digest(
        classifier_unsigned,
        domain="goalzendo-interactive-v2-calibrated-classifier-v1",
    )
    classifier["frozen_classifier_digest"] = classifier_digest
    evidence["frozen_classifier_digest"] = classifier_digest
    for episode in evidence["episodes"]:
        pattern = calibrated._pattern_from_obj(evidence["patterns"][episode["pattern_index"]])
        feature_rows = tuple(
            tuple(feature_row) for feature_row in episode["candidate_feature_tokens"]
        )
        episode["candidate_prediction_digest"] = calibrated._prediction_row_digest(
            classifier_digest=classifier_digest,
            construction_cluster_digest=episode["construction_cluster_digest"],
            block_opening_digest=episode["block_opening_digest"],
            model_visible_prompt_digest=episode["model_visible_prompt_digest"],
            surface_digest=episode["surface_digest"],
            official_index=episode["official_index"],
            pattern=pattern,
            candidate_feature_tokens=feature_rows,
        )
    evidence_unsigned = {
        key: value for key, value in evidence.items() if key != "prediction_evidence_digest"
    }
    evidence_digest = calibrated._digest(
        evidence_unsigned,
        domain="goalzendo-interactive-v2-calibrated-prediction-evidence-v1",
    )
    evidence["prediction_evidence_digest"] = evidence_digest
    view["estimate"]["prediction_evidence_digest"] = evidence_digest
    report_unsigned = {
        key: value for key, value in forged.items() if key != "calibrated_leakage_audit_digest"
    }
    forged_digest = calibrated._digest(
        report_unsigned,
        domain="goalzendo-interactive-v2-calibrated-leakage-report-v1",
    )
    forged["calibrated_leakage_audit_digest"] = forged_digest

    with pytest.raises(CalibratedLeakageV2Error, match="score differs from stored evaluation"):
        parse_calibrated_leakage_audit_v2(
            json.dumps(forged, ensure_ascii=True, separators=(",", ":")),
            expected_digest=forged_digest,
        )


def test_standalone_parser_boundary_is_explicit_and_block_rebuild_rejects_forged_unseen_token(
    calibrated_report: CalibratedLeakageAuditReportV2,
    split_blocks: BlockSplit,
) -> None:
    forged = json.loads(serialize_calibrated_leakage_audit_v2(calibrated_report))
    view = next(item for item in forged["views"] if item["view_name"] == "catalog_structure")
    evidence = view["prediction_evidence"]
    classifier_digest = view["classifier"]["frozen_classifier_digest"]
    forged_token = "aaa_forged_surface_token=1"
    assert all(
        forged_token not in feature_row
        for episode in evidence["episodes"]
        for feature_row in episode["candidate_feature_tokens"]
    )
    for episode in evidence["episodes"]:
        episode["candidate_feature_tokens"] = [
            sorted((*feature_row, forged_token))
            for feature_row in episode["candidate_feature_tokens"]
        ]
        pattern = calibrated._pattern_from_obj(evidence["patterns"][episode["pattern_index"]])
        feature_rows = tuple(
            tuple(feature_row) for feature_row in episode["candidate_feature_tokens"]
        )
        episode["candidate_prediction_digest"] = calibrated._prediction_row_digest(
            classifier_digest=classifier_digest,
            construction_cluster_digest=episode["construction_cluster_digest"],
            block_opening_digest=episode["block_opening_digest"],
            model_visible_prompt_digest=episode["model_visible_prompt_digest"],
            surface_digest=episode["surface_digest"],
            official_index=episode["official_index"],
            pattern=pattern,
            candidate_feature_tokens=feature_rows,
        )
    evidence_unsigned = {
        key: value for key, value in evidence.items() if key != "prediction_evidence_digest"
    }
    evidence_digest = calibrated._digest(
        evidence_unsigned,
        domain="goalzendo-interactive-v2-calibrated-prediction-evidence-v1",
    )
    evidence["prediction_evidence_digest"] = evidence_digest
    view["estimate"]["prediction_evidence_digest"] = evidence_digest
    report_unsigned = {
        key: value for key, value in forged.items() if key != "calibrated_leakage_audit_digest"
    }
    forged_digest = calibrated._digest(
        report_unsigned,
        domain="goalzendo-interactive-v2-calibrated-leakage-report-v1",
    )
    forged["calibrated_leakage_audit_digest"] = forged_digest

    parsed = parse_calibrated_leakage_audit_v2(
        json.dumps(forged, ensure_ascii=True, separators=(",", ":")),
        expected_digest=forged_digest,
    )
    assert parsed.as_obj()["standalone_parser_block_feature_rederivation_verified"] is False
    with pytest.raises(CalibratedLeakageV2Error, match="exact block-backed feature"):
        verify_calibrated_leakage_audit_against_blocks_v2(
            parsed,
            split_blocks[0],
            split_blocks[1],
            expected_report_digest=forged_digest,
            expected_split_plan_digest=parsed.split_plan.digest,
        )


def test_serialized_contract_cannot_mutate_later_reports(
    calibrated_report: CalibratedLeakageAuditReportV2,
) -> None:
    first = calibrated_report.as_obj()
    views = first["classifier_contract"]["views"]
    assert type(views) is list
    views.append(["forged_view", True])
    second = calibrated_report.as_obj()
    assert ["forged_view", True] not in second["classifier_contract"]["views"]


def test_configuration_rejects_boolean_aliases() -> None:
    with pytest.raises(CalibratedLeakageV2Error):
        CalibratedLeakageConfigV2(bootstrap_replicates=True)
    with pytest.raises(CalibratedLeakageV2Error):
        CalibratedLeakageConfigV2(minimum_calibration_clusters=True)
