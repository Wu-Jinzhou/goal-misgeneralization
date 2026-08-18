from __future__ import annotations

import pytest

from goalzendo_interactive.generation import (
    EpisodeBank,
    generate_episode_bank,
    small_fixture_bank_spec,
)
from goalzendo_interactive.leakage import (
    EPISODE_BANK_SURFACE_TARGETS,
    EPISODE_SURFACE_FEATURE_KEYS,
    SurfaceLeakageConfig,
    SurfaceLeakageSample,
    SurfaceLeakageTarget,
    audit_episode_bank_surface_leakage,
    audit_surface_target,
    audit_surface_targets,
    episode_bank_surface_targets,
    minimum_resolution_groups,
)


@pytest.fixture(scope="module")
def audit_config() -> SurfaceLeakageConfig:
    return SurfaceLeakageConfig(bootstrap_replicates=1_000)


def _binary_target(
    name: str,
    *,
    group_count: int = 400,
    leak_label: bool = False,
) -> SurfaceLeakageTarget:
    samples = []
    for index in range(group_count):
        label = ("false", "true")[index % 2]
        visible = label if leak_label else "constant"
        samples.append(
            SurfaceLeakageSample(
                sample_id=f"sample-{index}",
                group_id=f"group-{index}",
                features=(f"visible={visible}",),
                label=label,
            )
        )
    return SurfaceLeakageTarget(name, ("false", "true"), tuple(samples))


def test_balanced_no_signal_fixture_passes_both_registered_bounds(
    audit_config: SurfaceLeakageConfig,
) -> None:
    result = audit_surface_target(
        _binary_target("balanced_no_signal"),
        config=audit_config,
    )

    assert result.decision == "pass"
    assert result.passed
    assert result.chance == 0.5
    assert result.chance_ceiling == 0.55
    assert result.balanced_accuracy == 0.5
    assert result.interval_lower == result.interval_upper == 0.5
    assert result.chance_in_interval is True
    assert result.upper_below_ceiling is True
    assert result.reasons == ()


def test_surface_label_leak_fixture_fails_both_registered_bounds(
    audit_config: SurfaceLeakageConfig,
) -> None:
    result = audit_surface_target(
        _binary_target("obvious_surface_leak", leak_label=True),
        config=audit_config,
    )

    assert result.decision == "leakage"
    assert not result.passed
    assert result.balanced_accuracy == 1.0
    assert result.interval_lower == result.interval_upper == 1.0
    assert result.chance_in_interval is False
    assert result.upper_below_ceiling is False
    assert result.reasons == (
        "chance_not_in_95pct_interval",
        "interval_upper_not_below_chance_plus_0p05",
    )


def test_fold_assignment_predictions_and_bootstrap_are_byte_deterministic(
    audit_config: SurfaceLeakageConfig,
) -> None:
    targets = (
        _binary_target("balanced_no_signal"),
        _binary_target("future_registered_surface_label"),
    )

    first = audit_surface_targets("synthetic-bank", targets, config=audit_config)
    second = audit_surface_targets("synthetic-bank", targets, config=audit_config)

    assert first == second
    assert first.as_obj() == second.as_obj()
    assert first.as_obj()["config"] == {
        "fold_count": 5,
        "bootstrap_replicates": 1_000,
        "laplace_alpha": 1.0,
        "additional_minimum_groups": 0,
    }
    assert first.dataset_digest == second.dataset_digest
    assert [result.target_name for result in first.results] == [
        "balanced_no_signal",
        "future_registered_surface_label",
    ]
    assert all(result.fold_digest for result in first.results)
    assert all(result.prediction_digest for result in first.results)
    assert first.passed


def test_audit_fails_closed_for_missing_class_group_and_fold_coverage(
    audit_config: SurfaceLeakageConfig,
) -> None:
    samples = tuple(
        SurfaceLeakageSample(
            sample_id=f"sample-{index}",
            group_id=f"group-{index}",
            features=("visible=constant",),
            label="true" if index < 4 else "false",
        )
        for index in range(400)
    )
    target = SurfaceLeakageTarget("rare_class", ("false", "true"), samples)

    result = audit_surface_target(target, config=audit_config)

    assert result.decision == "insufficient_data"
    assert not result.passed
    assert result.balanced_accuracy is None
    assert result.interval_lower is result.interval_upper is None
    assert "class_sample_count_below_fold_count:true:4<5" in result.reasons
    assert any(
        reason.startswith("class_group_count_below_required_coverage:true:4<")
        for reason in result.reasons
    )


def test_twelve_episode_engineering_fixture_is_masked_and_underpowered(
    audit_config: SurfaceLeakageConfig,
) -> None:
    bank: EpisodeBank = generate_episode_bank(small_fixture_bank_spec())
    assert len(bank.episodes) == 12

    targets = episode_bank_surface_targets(bank)
    assert tuple(target.name for target in targets) == EPISODE_BANK_SURFACE_TARGETS

    sensitive_tokens = {
        "sun",
        "moon",
        "fits",
        "accepted",
        "scene_index",
        "target_rule",
        "shadow_rule",
        "placard",
        "red",
        "blue",
        "green",
        "pyramid",
        "cube",
        "sphere",
        "small",
        "large",
    }
    for target in targets:
        for sample in target.samples:
            assert {feature.split("=", 1)[0] for feature in sample.features} <= (
                EPISODE_SURFACE_FEATURE_KEYS
            )
            assert not sensitive_tokens.intersection(
                token.lower()
                for feature in sample.features
                for token in feature.replace("=", " ").split()
            )

    report = audit_episode_bank_surface_leakage(bank, config=audit_config)
    assert not report.passed
    assert tuple(result.target_name for result in report.results) == EPISODE_BANK_SURFACE_TARGETS
    assert minimum_resolution_groups(2) == 381
    assert all(result.decision == "insufficient_data" for result in report.results)
    assert all(not result.passed for result in report.results)
    assert all(result.minimum_groups_required == 381 for result in report.results)
    assert all(
        any("below_resolution_minimum" in reason for reason in result.reasons)
        for result in report.results
    )
