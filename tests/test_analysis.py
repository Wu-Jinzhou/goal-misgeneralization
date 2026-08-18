"""Focused regression tests for preregistered H1--H4 analysis semantics."""

from __future__ import annotations

import pandas as pd
import pytest

from forkworld.analysis import evaluate_hypothesis


def test_h1_directional_claims_use_seed_level_confidence_intervals() -> None:
    rows = []
    for seed in range(3):
        for q in (0.5, 0.9):
            for k in (1, 3):
                for capacity in (10, 100):
                    rows.append(
                        {
                            "config.experiment.hypothesis": "h1",
                            "config.seed": seed,
                            "config.data.q": q,
                            "config.data.k": k,
                            "model.total_parameters": capacity,
                            "final.conflict.delta_rho": (
                                -2.0 * q - 0.5 * k + 0.01 * capacity
                            ),
                        }
                    )
    report = evaluate_hypothesis(pd.DataFrame(rows), "h1")

    assert report["q_effect"]["n_seeds"] == 3
    assert report["q_effect"]["ci_high"] < 0
    assert report["k_effect"]["ci_high"] < 0
    assert report["capacity_effect"]["ci_low"] > 0
    assert report["status"] == "consistent"


def _h4_contrast(differences: tuple[float, ...]) -> pd.DataFrame:
    rows = []
    for seed, difference in enumerate(differences):
        common = {
            "config.experiment.hypothesis": "h4",
            "config.seed": seed,
            "config.h4.n_conflict": 8,
            "data.n_conflict": 8,
            "config.data.k": 2,
        }
        rows.extend(
            (
                {
                    **common,
                    "config.h4.condition": "concentrated",
                    "data.u_conflict": 2,
                    "final.conflict_unseen.rho_y": 0.5 - difference / 2,
                },
                {
                    **common,
                    "config.h4.condition": "diverse",
                    "data.u_conflict": 8,
                    "final.conflict_unseen.rho_y": 0.5 + difference / 2,
                },
            )
        )
    return pd.DataFrame(rows)


def test_h4_does_not_treat_an_imprecise_null_as_equivalence() -> None:
    report = evaluate_hypothesis(
        _h4_contrast((-0.2, 0.2, -0.2, 0.2)), "h4", margin=0.05
    )

    contrast = report["diverse_minus_concentrated"]
    assert contrast["mean"] == 0.0
    assert contrast["ci_low"] < -0.05
    assert contrast["ci_high"] > 0.05
    assert report["status"] == "mixed_or_inconclusive"


def test_h4_equivalence_requires_the_full_interval_inside_the_margin() -> None:
    report = evaluate_hypothesis(_h4_contrast((0.01, 0.01, 0.01)), "h4", margin=0.05)

    assert report["diverse_minus_concentrated"]["ci_low"] >= -0.05
    assert report["diverse_minus_concentrated"]["ci_high"] <= 0.05
    assert report["status"] == "evidence_against"


def test_analysis_honors_resolved_nondefault_inference_settings() -> None:
    frame = _h4_contrast((0.1, 0.1, 0.1))
    frame["config.evaluation.equivalence_margin"] = 0.2
    frame["config.evaluation.bootstrap_samples"] = 37
    frame["config.evaluation.confidence"] = 0.8

    report = evaluate_hypothesis(frame, "h4")
    contrast = report["diverse_minus_concentrated"]

    assert report["inference_settings"] == {
        "equivalence_margin": 0.2,
        "bootstrap_samples": 37,
        "confidence": 0.8,
    }
    assert report["equivalence_margin"] == 0.2
    assert contrast["bootstrap_samples"] == 37
    assert contrast["confidence"] == 0.8
    # The same 0.1 effect is equivalent under the configured 0.2 margin; it
    # would be directional support under the historical hardcoded 0.05 margin.
    assert report["status"] == "evidence_against"


@pytest.mark.parametrize(
    ("column", "changed"),
    [
        ("config.evaluation.equivalence_margin", 0.2),
        ("config.evaluation.bootstrap_samples", 37),
        ("config.evaluation.confidence", 0.8),
    ],
)
def test_analysis_rejects_conflicting_inference_settings(
    column: str, changed: float | int
) -> None:
    frame = _h4_contrast((0.1, 0.1, 0.1))
    defaults = {
        "config.evaluation.equivalence_margin": 0.05,
        "config.evaluation.bootstrap_samples": 4000,
        "config.evaluation.confidence": 0.95,
    }
    for name, value in defaults.items():
        frame[name] = value
    frame.loc[frame.index[0], column] = changed

    with pytest.raises(ValueError, match="conflicting config.evaluation"):
        evaluate_hypothesis(frame, "h4")


def test_analysis_rejects_explicit_override_conflicting_with_artifacts() -> None:
    frame = _h4_contrast((0.1, 0.1, 0.1))
    frame["config.evaluation.equivalence_margin"] = 0.2

    with pytest.raises(ValueError, match="explicit equivalence_margin"):
        evaluate_hypothesis(frame, "h4", margin=0.05)


def test_h4_reliable_agreement_only_learning_is_evidence_against() -> None:
    rows = []
    for seed in range(3):
        for condition in ("concentrated", "diverse"):
            rows.append(
                {
                    "config.experiment.hypothesis": "h4",
                    "config.seed": seed,
                    "config.h4.condition": condition,
                    "config.h4.n_conflict": 0,
                    "data.n_conflict": 0,
                    "data.u_conflict": 0,
                    "config.data.k": 2,
                    "config.evaluation.acquisition_threshold": 0.9,
                    "final.conflict_unseen.rho_y": 0.95,
                }
            )
    report = evaluate_hypothesis(pd.DataFrame(rows), "h4", margin=0.05)

    assert report["agreement_only_is_reliably_intended"] is True
    assert report["status"] == "evidence_against"


def test_h3_uses_censor_identifiable_acquisition_orderings() -> None:
    summaries = pd.DataFrame(
        {
            "config.experiment.hypothesis": ["h3"] * 4,
            "config.seed": [0, 1, 2, 3],
            "events.proxy_acquisition_time": [1, 8, 8, 1],
            "events.proxy_acquisition_observed": [True, False, False, True],
            "events.intended_acquisition_time": [8, 2, 8, 4],
            "events.intended_acquisition_observed": [False, True, False, True],
        }
    )
    report = evaluate_hypothesis(summaries, "h3")

    assert report["both_goals_observed_runs"] == 1
    assert report["acquisition_order_identifiable_runs"] == 3
    assert report["acquisition_order_censor_identified_runs"] == 2
    assert report["acquisition_order_unidentified_runs"] == 1
    assert report["simple_first_fraction"] == 2 / 3


def test_h3_acquisition_inference_uses_seeds_not_factorial_cells() -> None:
    rows = []
    for simple_first in ([True] * 9 + [False]):
        rows.append(
            {
                "config.experiment.hypothesis": "h3",
                "config.seed": 0,
                "events.proxy_acquisition_time": 1 if simple_first else 8,
                "events.proxy_acquisition_observed": simple_first,
                "events.intended_acquisition_time": 8 if simple_first else 1,
                "events.intended_acquisition_observed": not simple_first,
            }
        )
    rows.append(
        {
            "config.experiment.hypothesis": "h3",
            "config.seed": 1,
            "events.proxy_acquisition_time": 8,
            "events.proxy_acquisition_observed": False,
            "events.intended_acquisition_time": 1,
            "events.intended_acquisition_observed": True,
        }
    )
    report = evaluate_hypothesis(pd.DataFrame(rows), "h3")

    # Seed 0 contributes 0.9 and seed 1 contributes 0.0. Treating eleven cells
    # as independent would instead (and incorrectly) report 9/11.
    assert report["acquisition_order_identifiable_seeds"] == 2
    assert report["simple_first_fraction"] == 0.45
    assert report["simple_first_seed_bootstrap"]["n_seeds"] == 2


def test_h3_majority_point_estimates_do_not_bypass_directional_intervals() -> None:
    rows = []
    metric_rows = []
    for seed, favorable in enumerate((True, True, False)):
        run = f"run-{seed}"
        rows.append(
            {
                "run_path": run,
                "config.experiment.hypothesis": "h3",
                "config.seed": seed,
                "config.data.q": 0.9,
                "config.evaluation.acquisition_threshold": 0.9,
                "config.evaluation.persistence": 2,
                "config.h3.reward_plateau_window": 2,
                "config.h3.reward_plateau_tolerance": 0.001,
                "events.proxy_acquisition_time": 1 if favorable else 4,
                "events.proxy_acquisition_observed": True,
                "events.intended_acquisition_time": 4 if favorable else 1,
                "events.intended_acquisition_observed": True,
            }
        )
        steps = range(6)
        rewards = [0.9] * 6 if favorable else [0.1, 0.1, 0.9, 0.9, 0.9, 0.9]
        intended = [0.1, 0.1, 0.1, 0.95, 0.95, 0.95] if favorable else [0.95] * 6
        proxy = [0.95, 0.95, 0.95, 0.1, 0.1, 0.1] if favorable else [0.1] * 6
        for step, reward, rho_y, rho_p in zip(
            steps, rewards, intended, proxy, strict=True
        ):
            for split, metric, value in (
                ("iid", "target_accuracy", reward),
                ("conflict", "rho_y", rho_y),
                ("conflict", "rho_p", rho_p),
            ):
                metric_rows.append(
                    {
                        "run_path": run,
                        "experiment": "h3",
                        "split": split,
                        "metric": metric,
                        "value": value,
                        "global_step": step,
                        "level": "choice",
                        "intervention": "none",
                    }
                )
    report = evaluate_hypothesis(
        pd.DataFrame(rows), "h3", metrics=pd.DataFrame(metric_rows)
    )

    assert report["simple_first_fraction"] == 2 / 3
    assert report["simple_first_seed_bootstrap"]["ci_low"] <= 0.5
    plateau = report["reward_plateau_before_goal_stabilization"]
    assert plateau["plateau_before_goal_stabilization_fraction"] == 2 / 3
    assert plateau["plateau_before_goal_stabilization_seed_bootstrap"]["ci_low"] <= 0.5
    assert report["status"] == "mixed_or_inconclusive"


def test_h2_reports_exact_tolerance_matches_and_uses_interval_status() -> None:
    rows = []
    seed_differences = (0.8, 0.8, -0.8)
    architectures = (
        (1, "relu", False, 100),
        (1, "tanh", False, 100),
        (4, "relu", True, 1000),
    )
    for seed, difference in enumerate(seed_differences):
        for depth, activation, residual, parameters in architectures:
            for width, crossed in ((8, True), (16, False)):
                behavior = 0.5 + difference / 2 if crossed else 0.5 - difference / 2
                rows.append(
                    {
                        "config.experiment.hypothesis": "h2",
                        "config.seed": seed,
                        "config.data.k": 2,
                        "config.data.q": 0.9,
                        "config.model.width": width,
                        "config.model.depth": depth,
                        "config.model.activation": activation,
                        "config.model.residual": residual,
                        "config.update.mode": "full",
                        "config.update.budget": "full",
                        "config.h2.parameter_match_tolerance": 0.05,
                        "config.h2.target_accuracy": 0.95,
                        "config.h2.target_seed_fraction": 0.8,
                        "data.calibration_input_dim_matches_competition": True,
                        "data.calibration_parameter_count_matches_competition": True,
                        "data.calibration_trainable_count_matches_competition": True,
                        "model.proxy_calibration.total_parameters": parameters + width,
                        "model.exact_calibration.total_parameters": parameters + width,
                        "model.proxy_calibration.trainable_parameters": parameters + width,
                        "model.exact_calibration.trainable_parameters": parameters + width,
                        "final.proxy_calibration_iid.target_accuracy": 0.99 if crossed else 0.5,
                        "final.exact_calibration_iid.target_accuracy": 0.99 if crossed else 0.5,
                        "final.competition_conflict.rho_p": behavior,
                        "final.competition_conflict.rho_y": behavior,
                        "events.proxy_decoder_acquisition_observed": crossed,
                        "events.exact_decoder_acquisition_observed": crossed,
                        "events.proxy_decoder_acquisition_time": 10,
                        "events.exact_decoder_acquisition_time": 10,
                    }
                )
    report = evaluate_hypothesis(pd.DataFrame(rows), "h2")
    matching = report["complexity"]["architecture_matching"]

    assert matching["declared_relative_tolerance"] == 0.05
    assert any(item["match_kind"] == "exact" for item in matching["matched"])
    assert any(
        item["reason"] == "closest_pair_exceeds_tolerance"
        for item in matching["unmatched"]
    )
    assert report["calibration_behavior_inference"][
        "all_directional_intervals_above_zero"
    ] is False
    assert report["status"] == "mixed_or_inconclusive"


def test_level_confirmation_is_seed_paired_and_missing_levels_are_explicit() -> None:
    frame = _h4_contrast((0.2, 0.2, 0.2))
    frame["navigation.choice.intended_goal_success_rate"] = frame[
        "final.conflict_unseen.rho_y"
    ]
    frame["navigation.choice.oracle_goal_success_rate"] = 1.0
    frame["navigation.choice.clamped_goal_success_rate"] = 1.0
    frame["navigation.fork.intended_goal_success_rate"] = frame[
        "navigation.choice.intended_goal_success_rate"
    ] - 0.1
    frame["navigation.fork.oracle_goal_success_rate"] = 1.0
    frame["navigation.fork.clamped_goal_success_rate"] = 0.95
    report = evaluate_hypothesis(frame, "h4")
    levels = report["level_confirmation"]

    assert levels["levels"]["choice"]["status"] == "estimated"
    assert levels["levels"]["fork"]["status"] == "estimated"
    assert levels["levels"]["navigation"]["status"] == "insufficient_data"
    assert levels["paired_level_contrasts"]["fork_minus_choice"]["mean"] == pytest.approx(-0.1)
    assert levels["paired_level_contrasts"]["fork_minus_choice"]["n_paired_seeds"] == 3
    assert levels["hypothesis_effects"]["fork"]["diverse_minus_concentrated"]["mean"] == pytest.approx(0.2)


def _h6_scaling_frame(
    step_gaps: tuple[tuple[float, float, float], ...]
) -> pd.DataFrame:
    rows = []
    for seed, seed_step_gaps in enumerate(step_gaps):
        for structure in ("step", "state"):
            gaps = seed_step_gaps if structure == "step" else (-0.4, -0.4, -0.4)
            for (n_train, presentations), gap in zip(
                ((100, 800), (200, 1_600), (400, 3_200)), gaps, strict=True
            ):
                common = {
                    "config.experiment.hypothesis": "h6",
                    "config.seed": seed,
                    "config.h6.algorithm": "clean_sft",
                    "config.h6.structure": structure,
                    "config.h6.location": "observation",
                    "config.h6.reward_mode": "dense_fixed_horizon",
                    # Deliberately unlike the realized field: matching and
                    # slopes must use evidence recorded by the run.
                    "config.h6.visits_per_state": 999,
                    "config.data.n_train": n_train,
                    "training.realized_presentations": presentations,
                    "recurrence.realized_episode_visits_per_state": 4,
                    "noise.objective_uses_location": True,
                }
                rows.extend(
                    (
                        {**common, "config.h6.scale": 0.0, "final.rho_y": 0.8},
                        {
                            **common,
                            "config.h6.scale": 1.0,
                            "final.rho_y": 0.8 + gap,
                        },
                    )
                )
    return pd.DataFrame(rows)


def test_h6_uses_seed_level_matched_gap_slopes_for_sample_scaling() -> None:
    frame = _h6_scaling_frame(((-0.4, -0.2, 0.0),) * 3)

    report = evaluate_hypothesis(frame, "h6")
    noise = report["noise_structure"]
    scaling = noise["sample_scaling"]

    assert noise["sample_presentation_column"] == "training.realized_presentations"
    assert scaling["status"] == "consistent"
    step = scaling["independent_step_resampled"]["gap_to_zero_slope"]
    state = scaling["state_static_comparator"]["gap_to_zero_slope"]
    difference = scaling["step_resampled_minus_state_static"][
        "gap_to_zero_slope_difference"
    ]
    assert step["mean"] == pytest.approx(0.2)
    assert step["ci_low"] > 0
    assert state["mean"] == pytest.approx(0.0)
    assert difference["ci_low"] > 0
    assert report["subclaims"]["sample_scaling_noise_gap_recovery"]["status"] == (
        "consistent"
    )
    assert report["subclaims"]["temporal_robustness_ordering"]["status"] == (
        "insufficient_data"
    )


def test_h6_pooled_three_point_trend_cannot_affirm_sample_scaling() -> None:
    # The seed-averaged gap improves monotonically, but one of three independent
    # seeds reverses. A pooled three-point Spearman coefficient would be +1.
    frame = _h6_scaling_frame(
        ((-0.4, -0.2, 0.0), (-0.4, -0.2, 0.0), (0.0, -0.2, -0.4))
    )

    report = evaluate_hypothesis(frame, "h6")
    scaling = report["noise_structure"]["sample_scaling"]
    step = scaling["independent_step_resampled"]["gap_to_zero_slope"]

    assert step["mean"] > 0
    assert step["ci_low"] <= 0 <= step["ci_high"]
    assert scaling["independent_step_resampled"]["status"] == (
        "mixed_or_inconclusive"
    )
    assert scaling["status"] == "mixed_or_inconclusive"


def _h6_required_algorithm_frame(*, failing_rl: bool) -> pd.DataFrame:
    rows = []
    supportive = {
        "step": (-0.3, -0.1, 0.0),
        "episode": (-0.8, -0.6, -0.5),
        "state": (-0.9, -0.9, -0.9),
    }
    reversed_rl = {
        "step": (-0.7, -0.8, -0.9),
        "episode": (-0.6, -0.6, -0.6),
        "state": (-0.5, -0.5, -0.5),
    }
    for seed in range(3):
        for algorithm in ("clean_sft", "rl"):
            structures = reversed_rl if failing_rl and algorithm == "rl" else supportive
            for structure, gaps in structures.items():
                for (n_train, presentations), gap in zip(
                    ((100, 800), (200, 1_600), (400, 3_200)), gaps, strict=True
                ):
                    common = {
                        "config.experiment.hypothesis": "h6",
                        "config.seed": seed,
                        "config.h6.algorithm": algorithm,
                        "config.h6.structure": structure,
                        "config.h6.location": "observation",
                        "config.h6.reward_mode": "dense_fixed_horizon",
                        "config.data.n_train": n_train,
                        "training.realized_presentations": presentations,
                        "recurrence.realized_episode_visits_per_state": 4,
                        "noise.objective_uses_location": True,
                    }
                    rows.extend(
                        (
                            {
                                **common,
                                "config.h6.scale": 0.0,
                                "final.rho_y": 0.95,
                            },
                            {
                                **common,
                                "config.h6.scale": 1.0,
                                "final.rho_y": 0.95 + gap,
                            },
                        )
                    )
    return pd.DataFrame(rows)


def test_h6_requires_clean_sft_and_rl_algorithm_level_support() -> None:
    report = evaluate_hypothesis(
        _h6_required_algorithm_frame(failing_rl=False), "h6"
    )

    assert report["algorithm_subclaims"]["clean_sft"]["status"] == "consistent"
    assert report["algorithm_subclaims"]["rl"]["status"] == "consistent"
    assert report["status"] == "consistent"


def test_h6_pooled_support_cannot_override_required_rl_reversal() -> None:
    report = evaluate_hypothesis(
        _h6_required_algorithm_frame(failing_rl=True), "h6"
    )

    # Clean SFT is strong enough that both pooled diagnostics remain positive.
    assert report["subclaims"]["temporal_robustness_ordering"]["status"] == (
        "consistent"
    )
    assert report["subclaims"]["sample_scaling_noise_gap_recovery"]["status"] == (
        "consistent"
    )
    assert report["algorithm_subclaims"]["rl"]["status"] == "evidence_against"
    assert report["status"] == "evidence_against"


def test_h6_terminal_reward_control_resamples_training_seeds_not_scales() -> None:
    rows = []
    seed_differences = {0: -0.03, 1: 0.0, 2: 0.03}
    for seed, difference in seed_differences.items():
        for scale in (0.0, 0.25, 0.5, 1.0, 2.0):
            for structure, rho_y in (
                ("step", 0.5 + difference),
                ("episode", 0.5),
            ):
                rows.append(
                    {
                        "config.experiment.hypothesis": "h6",
                        "config.seed": seed,
                        "config.h6.algorithm": "rl",
                        "config.h6.structure": structure,
                        "config.h6.location": "reward",
                        "config.h6.scale": scale,
                        "config.h6.reward_mode": "terminal",
                        "config.data.n_train": 1_000,
                        "training.realized_presentations": 8_000,
                        "recurrence.realized_episode_visits_per_state": 16,
                        "noise.objective_uses_location": True,
                        "final.rho_y": rho_y,
                    }
                )

    report = evaluate_hypothesis(pd.DataFrame(rows), "h6")
    equivalence = report["noise_structure"][
        "terminal_reward_step_episode_equivalence"
    ]

    assert equivalence["status"] == "estimated"
    assert equivalence["mean"] == pytest.approx(0.0)
    assert equivalence["n_seeds"] == 3
    assert equivalence["equivalent_within_margin"] is True


@pytest.mark.parametrize("hypothesis", [f"h{index}" for index in range(1, 10)])
def test_single_seed_runs_are_never_given_inferential_status(hypothesis: str) -> None:
    frame = pd.DataFrame(
        [
            {
                "config.experiment.hypothesis": hypothesis,
                "config.seed": 0,
                "final.rho_y": 0.9,
                "final.conflict.rho_y": 0.9,
                "final.conflict.delta_rho": 0.4,
            }
        ]
    )
    report = evaluate_hypothesis(frame, hypothesis)

    assert report["status"] == "insufficient_data"
    assert report["inference_sufficiency"]["observed_independent_seeds"] == 1
    assert report["inference_sufficiency"]["minimum_inferential_seeds"] == 3


def test_h7_reports_seed_paired_restored_proxy_ordering() -> None:
    rows = []
    for seed in range(3):
        for condition, half_life, restored, q_b in (
            ("removal", 100, 0.9, 0.5),
            ("decorrelation", 60, 0.5, 0.5),
            ("reversal", 20, 0.1, 0.1),
        ):
            rows.append(
                {
                    "config.experiment.hypothesis": "h7",
                    "config.seed": seed,
                    "config.h7.condition": condition,
                    "config.h7.q_a": 0.99,
                    "config.h7.q_b": q_b,
                    "config.h7.weight_decay_ablation": False,
                    "config.model.width": 16,
                    "final.eligible_for_primary_analysis": True,
                    "events.half_life_time": half_life,
                    "events.half_life_observed": True,
                    "final.restoration_rho_p": restored,
                    "final.rho_y": 0.5,
                }
            )
    report = evaluate_hypothesis(pd.DataFrame(rows), "h7")
    restoration = report["restored_proxy_rebound_ordering"]

    assert restoration["decorrelation_minus_removal"]["ci_high"] < 0
    reversal = restoration["reversal_minus_decorrelation_by_q_b"][0]["contrast"]
    assert reversal["ci_high"] < 0
    assert reversal["n_seeds"] == 3
    assert report["status"] == "consistent"


def test_h8_directly_pairs_old_goal_history_with_compute_matched_sham() -> None:
    rows = []
    for seed in range(3):
        for history, n0, rebound, reactivation in (
            ("old_goal", 512, 0.4, 10),
            ("control", 0, 0.0, 30),
            ("compute_matched_sham", 512, 0.05, 28),
        ):
            rows.append(
                {
                    "config.experiment.hypothesis": "h8",
                    "config.seed": seed,
                    "config.h8.history": history,
                    "config.h8.n0": n0,
                    "config.h8.perturbation": "partial_reversal",
                    "config.h8.stage1_mode": "fixed",
                    "config.model.width": 16,
                    "final.stage1_gate_passed": True,
                    "final.stage1_in_match_band": False,
                    "final.eligible_for_primary_analysis": False,
                    "final.rebound_g0": rebound,
                    "events.reactivation_time": reactivation,
                    "events.reactivation_observed": True,
                    "final.rho_y": 0.9,
                }
            )
    report = evaluate_hypothesis(pd.DataFrame(rows), "h8")
    direct_rebound = next(
        item
        for item in report["rebound_contrasts"]
        if item["treatment"] == "old_goal"
        and item["minus"] == "compute_matched_sham"
    )
    direct_speed = next(
        item
        for item in report["reactivation_restricted_time_contrasts"]
        if item["treatment"] == "old_goal"
        and item["minus"] == "compute_matched_sham"
    )

    assert direct_rebound["contrast"]["ci_low"] > 0
    assert direct_speed["contrast"]["ci_high"] < 0
    assert direct_rebound["n_seeds"] == 3
    assert report["subclaims"]["history_specific_rebound_and_reactivation"][
        "status"
    ] == "consistent"
    assert report["status"] == "mixed_or_inconclusive"


def _h8_factorial_scaling_frame(*, one_seed_reverses: bool) -> pd.DataFrame:
    rows = []
    n0_levels = (512, 2_048, 8_192)
    widths = (16, 32, 64)
    perturbations = ("neutral", "partial_reversal")
    for seed in range(3):
        for width_index, width in enumerate(widths):
            for perturbation in perturbations:
                rows.append(
                    {
                        "config.experiment.hypothesis": "h8",
                        "config.seed": seed,
                        "config.h8.history": "control",
                        "config.h8.n0": 0,
                        "config.h8.perturbation": perturbation,
                        "config.h8.stage1_mode": "fixed",
                        "config.model.width": width,
                        "final.stage1_gate_passed": True,
                        "final.rebound_g0": 0.0,
                        "events.reactivation_time": 140.0,
                        "events.reactivation_observed": True,
                    }
                )
                for n0_index, n0 in enumerate(n0_levels):
                    n0_direction = -1 if one_seed_reverses and seed == 2 else 1
                    rebound = (
                        0.15
                        + n0_direction * 0.05 * n0_index
                        + 0.03 * width_index
                    )
                    reactivation = (
                        100.0
                        - n0_direction * 10.0 * n0_index
                        - 5.0 * width_index
                    )
                    common = {
                        "config.experiment.hypothesis": "h8",
                        "config.seed": seed,
                        "config.h8.n0": n0,
                        "config.h8.perturbation": perturbation,
                        "config.h8.stage1_mode": "fixed",
                        "config.model.width": width,
                        "final.stage1_gate_passed": True,
                        "events.reactivation_observed": True,
                    }
                    rows.extend(
                        (
                            {
                                **common,
                                "config.h8.history": "old_goal",
                                "final.rebound_g0": rebound,
                                "events.reactivation_time": reactivation,
                            },
                            {
                                **common,
                                "config.h8.history": "compute_matched_sham",
                                "final.rebound_g0": 0.02,
                                "events.reactivation_time": 130.0,
                            },
                        )
                    )
    return pd.DataFrame(rows)


def test_h8_requires_history_n0_and_capacity_subclaims() -> None:
    report = evaluate_hypothesis(
        _h8_factorial_scaling_frame(one_seed_reverses=False), "h8"
    )

    assert report["subclaims"]["history_specific_rebound_and_reactivation"][
        "status"
    ] == "consistent"
    assert report["subclaims"]["old_goal_volume_scaling"]["status"] == (
        "consistent"
    )
    assert report["subclaims"]["model_capacity_scaling"]["status"] == "consistent"
    assert report["old_volume_rebound_association"]["ci_low"] > 0
    assert report["old_volume_reactivation_time_association"]["ci_high"] < 0
    assert report["model_capacity_rebound_association"]["ci_low"] > 0
    assert report["status"] == "consistent"


def test_h8_pooled_factorial_trend_cannot_override_seed_uncertainty() -> None:
    report = evaluate_hypothesis(
        _h8_factorial_scaling_frame(one_seed_reverses=True), "h8"
    )
    n0_rebound = report["old_volume_rebound_association"]

    assert n0_rebound["rho"] > 0
    assert n0_rebound["ci_low"] <= 0 <= n0_rebound["ci_high"]
    assert report["subclaims"]["old_goal_volume_scaling"]["status"] == (
        "mixed_or_inconclusive"
    )
    assert report["status"] == "mixed_or_inconclusive"


def _h9_capacity_frame(*, near_zero_selector: bool) -> pd.DataFrame:
    rows = []
    for seed in range(3):
        for mode, capacities in (
            ("full", (10, 100, 1_000)),
            ("subspace", (1, 10, 100)),
        ):
            for index, capacity in enumerate(capacities):
                switching = (
                    (0.001, 0.002, 0.003)[index]
                    if near_zero_selector
                    else (0.2, 0.6, 0.95)[index]
                )
                highest = index == len(capacities) - 1
                mastered = highest and not near_zero_selector
                row = {
                    "config.experiment.hypothesis": "h9",
                    "config.seed": seed,
                    "config.update.mode": mode,
                    "model.total_parameters": capacity if mode == "full" else 1_000,
                    "model.trainable_parameters": capacity,
                    "config.update.budget": capacity if mode == "subspace" else "full",
                    "config.h9.selector_threshold": 0.9,
                    "config.h9.mastery_threshold": 0.9,
                    "final.single_rule_controls_passed": True,
                    "final.strict_context_switching": switching,
                    "final.selector_gate_passed": mastered,
                    "final.normal.target_accuracy": 0.95 if mastered else 0.51,
                }
                for name in ("mismatch_0_1", "mismatch_1_0"):
                    row[f"final.{name}.rho_p_observed_context"] = (
                        0.9 if mastered else 0.505
                    )
                    row[f"final.{name}.rho_p_environment"] = 0.4 if mastered else 0.5
                active = 1.0 if mastered else 0.011
                inactive = 0.1 if mastered else 0.01
                row.update(
                    {
                        "final.proxy_sensitivity_matrix.C0.P0.absolute_logit_change": active,
                        "final.proxy_sensitivity_matrix.C0.P1.absolute_logit_change": inactive,
                        "final.proxy_sensitivity_matrix.C1.P0.absolute_logit_change": inactive,
                        "final.proxy_sensitivity_matrix.C1.P1.absolute_logit_change": active,
                    }
                )
                rows.append(row)
    return pd.DataFrame(rows)


def test_h9_requires_capacity_trends_and_high_capacity_selector_mastery() -> None:
    report = evaluate_hypothesis(
        _h9_capacity_frame(near_zero_selector=False), "h9"
    )

    assert report["subclaims"]["capacity_trends"]["status"] == "consistent"
    assert report["highest_capacity_mastery"]["model_capacity"]["status"] == (
        "consistent"
    )
    assert report["highest_capacity_mastery"]["update_capacity"]["status"] == (
        "consistent"
    )
    assert report["highest_capacity_mastery"]["model_capacity"][
        "selector_mastery"
    ]["strict_switching_margin_above_threshold"]["ci_low"] >= 0
    assert report["status"] == "consistent"


def test_h9_near_zero_increasing_selector_trend_is_evidence_against() -> None:
    report = evaluate_hypothesis(
        _h9_capacity_frame(near_zero_selector=True), "h9"
    )

    # Rank trends alone are perfectly increasing in both capacity panels.
    assert report["subclaims"]["capacity_trends"]["status"] == "consistent"
    assert report["model_capacity_switching_association"]["ci_low"] > 0
    assert report["update_capacity_switching_association"]["ci_low"] > 0
    assert report["highest_capacity_mastery"]["model_capacity"]["status"] == (
        "evidence_against"
    )
    assert report["highest_capacity_mastery"]["update_capacity"]["status"] == (
        "evidence_against"
    )
    assert report["status"] == "evidence_against"
