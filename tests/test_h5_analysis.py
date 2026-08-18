"""Matched-seed, censor-aware inference for the H5 update-capacity claim."""

from __future__ import annotations

from collections.abc import Callable

import pandas as pd

from forkworld.analysis import evaluate_hypothesis


Threshold = Callable[[str, int, int], int | None]


def _h5_panel(threshold: Threshold, *, seeds: int = 6) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for seed in range(seeds):
        for entropy in (0, 1, 2, 3):
            for algorithm in (
                "clean_sft",
                "trajectory_sft",
                "on_policy_imitation",
                "rl",
            ):
                crossing = threshold(algorithm, entropy, seed)
                for budget in (1, 2, 4, 8, 16):
                    rows.append(
                        {
                            "config.experiment.hypothesis": "h5",
                            "config.seed": seed,
                            "config.h5.algorithm": algorithm,
                            "config.h5.nuisance_entropy": entropy,
                            "config.evaluation.acquisition_threshold": 0.9,
                            "config.evaluation.bootstrap_samples": 1000,
                            "model.trainable_parameters": budget,
                            "final.rho_y": (
                                0.95
                                if crossing is not None and budget >= crossing
                                else 0.2
                            ),
                        }
                    )
    return pd.DataFrame(rows)


def _control_threshold(algorithm: str) -> int:
    return {
        "clean_sft": 1,
        "on_policy_imitation": 4,
        "rl": 2,
    }[algorithm]


def test_h5_support_requires_a_positive_matched_seed_trend_interval() -> None:
    trajectory = (2, 4, 8, 16)

    def threshold(algorithm: str, entropy: int, seed: int) -> int:
        del seed
        return trajectory[entropy] if algorithm == "trajectory_sft" else _control_threshold(algorithm)

    report = evaluate_hypothesis(_h5_panel(threshold), "h5")
    thresholds = report["update_capacity_thresholds"]
    primary = thresholds["primary_comparison"]
    trend = report["entropy_ratio_association"]

    assert primary["matched_seed_count"] == 6
    assert trend["ci_low"] > 0
    assert trend["usable_draws"] == 1000
    assert all(item["usable_draws"] == 1000 for item in primary["entropy_ratio_intervals"])
    assert thresholds["controls_and_falsifiers"]["clean_sft"]["status"] == "control_passed"
    assert (
        thresholds["controls_and_falsifiers"]["on_policy_imitation"]["status"]
        == "control_passed"
    )
    assert report["status"] == "consistent"


def test_h5_on_policy_matching_rl_triggers_the_distribution_falsifier() -> None:
    trajectory = (2, 4, 8, 16)

    def threshold(algorithm: str, entropy: int, seed: int) -> int:
        del seed
        if algorithm == "trajectory_sft":
            return trajectory[entropy]
        if algorithm == "on_policy_imitation":
            return 2
        return _control_threshold(algorithm)

    report = evaluate_hypothesis(_h5_panel(threshold), "h5")
    control = report["update_capacity_thresholds"]["controls_and_falsifiers"][
        "on_policy_imitation"
    ]

    assert control["primary_high_entropy_rl_advantage"] is True
    assert control["high_entropy_interval"]["ratio"] == 1.0
    assert control["status"] == "falsifier_triggered"
    assert report["status"] == "evidence_against"


def test_h5_imprecise_null_is_not_treated_as_support_or_reversal() -> None:
    increasing = (2, 4, 8, 16)
    decreasing = tuple(reversed(increasing))

    def threshold(algorithm: str, entropy: int, seed: int) -> int:
        if algorithm == "trajectory_sft":
            return (increasing if seed < 3 else decreasing)[entropy]
        return _control_threshold(algorithm)

    report = evaluate_hypothesis(_h5_panel(threshold), "h5")
    trend = report["entropy_ratio_association"]

    assert trend["ci_low"] < 0 < trend["ci_high"]
    assert report["directional_inference"]["equivalent_to_zero"] is False
    assert report["status"] == "mixed_or_inconclusive"


def test_h5_rising_ratios_below_one_do_not_establish_an_rl_advantage() -> None:
    trajectory = (1, 1, 2, 2)

    def threshold(algorithm: str, entropy: int, seed: int) -> int:
        del seed
        if algorithm == "trajectory_sft":
            return trajectory[entropy]
        return {"clean_sft": 1, "on_policy_imitation": 8, "rl": 4}[algorithm]

    report = evaluate_hypothesis(_h5_panel(threshold), "h5")
    high_entropy = report["update_capacity_thresholds"][
        "primary_high_entropy_claim"
    ]

    assert report["entropy_ratio_association"]["ci_low"] > 0
    assert high_entropy["interval"]["ci_high"] < 1
    assert high_entropy["status"] == "evidence_against"
    assert report["status"] == "evidence_against"


def test_h5_precise_negative_ratio_trend_is_evidence_against() -> None:
    trajectory = (16, 8, 4, 2)

    def threshold(algorithm: str, entropy: int, seed: int) -> int:
        del seed
        return trajectory[entropy] if algorithm == "trajectory_sft" else _control_threshold(algorithm)

    report = evaluate_hypothesis(_h5_panel(threshold), "h5")

    assert report["entropy_ratio_association"]["ci_high"] < 0
    assert report["status"] == "evidence_against"


def test_h5_reports_right_censoring_and_stays_conservative() -> None:
    trajectory: tuple[int | None, ...] = (2, 4, 8, None)

    def threshold(algorithm: str, entropy: int, seed: int) -> int | None:
        del seed
        return trajectory[entropy] if algorithm == "trajectory_sft" else _control_threshold(algorithm)

    report = evaluate_hypothesis(_h5_panel(threshold), "h5")
    thresholds = report["update_capacity_thresholds"]
    high_entropy = thresholds["primary_comparison"]["entropy_ratio_intervals"][-1]

    assert high_entropy["right_censored"] is True
    assert high_entropy["usable_draws"] == 0
    assert high_entropy["censored_draws"] == 1000
    assert high_entropy["censoring_by_draw"]["numerator_only_censored"] == 1000
    assert thresholds["censoring_assessment"]["censor_heavy"] is True
    assert report["status"] == "mixed_or_inconclusive"


def test_h5_one_seed_is_descriptive_only() -> None:
    trajectory = (2, 4, 8, 16)

    def threshold(algorithm: str, entropy: int, seed: int) -> int:
        del seed
        return trajectory[entropy] if algorithm == "trajectory_sft" else _control_threshold(algorithm)

    report = evaluate_hypothesis(_h5_panel(threshold, seeds=1), "h5")

    assert report["update_capacity_thresholds"]["primary_comparison"]["matched_seed_count"] == 1
    assert report["update_capacity_thresholds"]["status"] == "insufficient_data"
    assert report["status"] == "insufficient_data"


def test_h5_seed_floor_uses_complete_matched_seeds_not_seed_union() -> None:
    trajectory = (2, 4, 8, 16)

    def threshold(algorithm: str, entropy: int, seed: int) -> int:
        del seed
        return trajectory[entropy] if algorithm == "trajectory_sft" else _control_threshold(algorithm)

    panel = _h5_panel(threshold, seeds=3)
    missing_cell = (
        panel["config.seed"].eq(2)
        & panel["config.h5.algorithm"].eq("rl")
        & panel["config.h5.nuisance_entropy"].eq(3)
        & panel["model.trainable_parameters"].eq(16)
    )
    report = evaluate_hypothesis(panel[~missing_cell], "h5")

    assert report["n_seeds"] == 3
    assert report["update_capacity_thresholds"]["primary_comparison"]["matched_seed_count"] == 2
    assert report["inference_sufficiency"]["observed_independent_seeds"] == 2
    assert report["status"] == "insufficient_data"
