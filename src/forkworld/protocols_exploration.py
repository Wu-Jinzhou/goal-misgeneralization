"""Exploratory staged mechanisms that follow the registered ForkWorld grids.

E14 was fixed only after the complete E10 result was inspected.  Keeping its
runner separate from the preregistered H5 implementation makes that provenance
hard to lose when artifacts are analyzed later.
"""

from __future__ import annotations

import copy
import math
import time
from collections.abc import Mapping
from dataclasses import dataclass
from itertools import pairwise
from typing import Any

import numpy as np
import torch
from torch import nn

from .config import get_path
from .data import SemanticBatch
from .protocols import (
    ProtocolResult,
    build_model,
    evaluate_batch,
    evaluate_standard_interventions,
    make_metric_records,
    predict_logits,
    prediction_records,
    resolve_device,
)
from .protocols_algorithms import (
    _costs,
    _critic,
    _evaluator,
    _float,
    _integer,
    _make_h5_episode_dataset,
    _NeutralTrajectoryReward,
    _nuisance_diagnostics,
    _SemanticSampler,
    _train_kwargs,
)
from .training import BanditConfig, TrainingResult, train_contextual_bandit


@dataclass(frozen=True)
class EntropyTimingSchedule:
    """The two entropy phases and the only permitted boundary mutations."""

    phase_a_beta: float
    phase_b_beta: float
    reset_actor_optimizer: bool = False
    reset_critic: bool = False


ENTROPY_TIMING_SCHEDULES: dict[str, EntropyTimingSchedule] = {
    "zero_zero": EntropyTimingSchedule(0.0, 0.0),
    "high_high": EntropyTimingSchedule(0.30, 0.30),
    "early_only": EntropyTimingSchedule(0.30, 0.0),
    "delayed_carry": EntropyTimingSchedule(0.0, 0.30),
    "delayed_actor_reset": EntropyTimingSchedule(
        0.0, 0.30, reset_actor_optimizer=True
    ),
    "delayed_critic_reset": EntropyTimingSchedule(0.0, 0.30, reset_critic=True),
}

_EVALUATION_NAMES = {
    "rho_y",
    "rho_p",
    "delta_rho",
    "intended_probability",
    "confidence",
    "invalid_rate",
    "target_accuracy",
}


def _timing_bandit_config(
    config: Mapping[str, Any],
    *,
    seed: int,
    steps: int,
    log_steps: tuple[int, ...],
    entropy_coefficient: float,
    reset_optimizer: bool,
    reset_critic: bool,
) -> BanditConfig:
    device = resolve_device(str(get_path(config, "run.device", "auto")))
    common = _train_kwargs(config, device, steps_override=steps)
    common.update(
        {
            "seed": int(seed),
            "log_steps": log_steps,
            "checkpoint_steps": (),
            # Boundary states are captured explicitly below.  Avoid materializing
            # every log-spaced training checkpoint just to continue the RNG.
            "save_checkpoints": False,
            "reset_optimizer": bool(reset_optimizer),
        }
    )
    return BanditConfig(
        **common,
        algorithm="actor_critic",
        entropy_coefficient=float(entropy_coefficient),
        critic_learning_rate=_float(
            config,
            "h5.critic_learning_rate",
            _float(config, "train.learning_rate", 3e-3),
        ),
        critic_weight_decay=_float(config, "h5.critic_weight_decay", 0.0),
        critic_width=_integer(config, "h5.critic_width", 256, minimum=1),
        critic_depth=_integer(config, "h5.critic_depth", 3, minimum=0),
        reset_critic=bool(reset_critic),
    )


def _phase_history_records(
    result: TrainingResult,
    *,
    phase: str,
    condition: str,
    step_offset: int,
    examples_offset: int,
    trajectory_horizon: int,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    counter_names = {"environment_interactions", "labeled_actions", "terminal_outcomes"}
    for item in result.history:
        global_step = step_offset + item.step
        global_examples = examples_offset + item.samples_seen
        train_values: dict[str, float | int] = {
            "loss": item.loss,
            "primary_loss": item.primary_loss,
            "auxiliary_loss": item.auxiliary_loss,
        }
        evaluation_values: dict[str, float | int] = {}
        for name, value in item.metrics.items():
            if name in counter_names:
                continue
            (evaluation_values if name in _EVALUATION_NAMES else train_values)[name] = value
        train_values.update(
            {
                "environment_interactions": global_examples * trajectory_horizon,
                "labeled_actions": 0,
                "terminal_outcomes": global_examples,
            }
        )
        records.extend(
            make_metric_records(
                train_values,
                hypothesis="h5",
                split="train",
                global_step=global_step,
                stage=phase,
                stage_step=item.step,
                examples_seen=global_examples,
                condition=condition,
            )
        )
        records.extend(
            make_metric_records(
                evaluation_values,
                hypothesis="h5",
                split="conflict_eval",
                global_step=global_step,
                stage=phase,
                stage_step=item.step,
                examples_seen=global_examples,
                condition=condition,
            )
        )
    return records


def _split_records(
    values: Mapping[str, float | int],
    *,
    split: str,
    condition: str,
    stage: str,
    global_step: int,
    examples_seen: int,
) -> list[dict[str, Any]]:
    return make_metric_records(
        values,
        hypothesis="h5",
        split=split,
        global_step=global_step,
        stage=stage,
        stage_step=global_step,
        examples_seen=examples_seen,
        condition=condition,
    )


def _intervention_records(
    values: Mapping[str, Mapping[str, float | int]],
    *,
    condition: str,
    stage: str,
    global_step: int,
    examples_seen: int,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for intervention, metrics in values.items():
        records.extend(
            make_metric_records(
                metrics,
                hypothesis="h5",
                split="conflict_eval",
                global_step=global_step,
                stage=stage,
                stage_step=global_step,
                examples_seen=examples_seen,
                intervention=intervention,
                condition=condition,
            )
        )
    return records


def _predictive_entropy(
    model: nn.Module,
    batch: SemanticBatch,
    config: Mapping[str, Any],
) -> float:
    logits = predict_logits(model, batch, config)
    shifted = logits - logits.max(axis=1, keepdims=True)
    probabilities = np.exp(shifted)
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    return float(
        np.mean(-np.sum(probabilities * np.log(np.clip(probabilities, 1e-12, 1.0)), axis=1))
    )


def _state_distance(
    before: Mapping[str, Any] | None,
    after: Mapping[str, Any] | None,
) -> dict[str, float]:
    if before is None or after is None:
        return {"l2": 0.0, "relative_l2": 0.0}
    squared_distance = 0.0
    squared_reference = 0.0
    for name, initial in before.items():
        final = after.get(name)
        if not isinstance(initial, torch.Tensor) or not isinstance(final, torch.Tensor):
            continue
        if not (torch.is_floating_point(initial) or torch.is_complex(initial)):
            continue
        initial_value = initial.detach().cpu().to(torch.float64)
        final_value = final.detach().cpu().to(torch.float64)
        squared_distance += float(torch.sum((final_value - initial_value) ** 2))
        squared_reference += float(torch.sum(initial_value**2))
    distance = math.sqrt(squared_distance)
    return {
        "l2": distance,
        "relative_l2": distance / max(math.sqrt(squared_reference), 1e-12),
    }


def _trajectory_summary(
    phase_a: TrainingResult,
    phase_b: TrainingResult,
    *,
    phase_a_steps: int,
    phase_b_steps: int,
) -> dict[str, float | int | None]:
    boundary_rho = float(phase_a.history[-1].metrics["rho_y"])
    points = [(phase_a_steps, boundary_rho)]
    points.extend(
        (phase_a_steps + record.step, float(record.metrics["rho_y"]))
        for record in phase_b.history
    )
    area = 0.0
    for (left_step, left_value), (right_step, right_value) in pairwise(points):
        area += (right_step - left_step) * (left_value + right_value) / 2.0
    acquisition: int | None = None
    for left, right in pairwise(points):
        if left[1] >= 0.50 and right[1] >= 0.50:
            acquisition = int(left[0])
            break
    return {
        "phase_b_rho_y_area": float(area),
        "phase_b_mean_rho_y": float(area / phase_b_steps),
        "persistent_rho_y_0_5_global_step": acquisition,
        "phase_b_updates_to_persistent_rho_y_0_5": (
            None if acquisition is None else max(0, acquisition - phase_a_steps)
        ),
        "persistence_evaluations": 2,
        "persistence_threshold": 0.50,
    }


def _checkpoint_payload(result: TrainingResult, step: int) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "step": int(step),
        "model": result.final_model_state,
        "optimizer": result.optimizer_state,
    }
    if result.critic_state is not None:
        payload["critic"] = result.critic_state
    if result.critic_optimizer_state is not None:
        payload["critic_optimizer"] = result.critic_optimizer_state
    return payload


def run_entropy_timing(config: Mapping[str, Any], seed: int) -> ProtocolResult:
    """Run one fixed E14 timing schedule in one E10 boundary cell."""

    schedule_name = str(get_path(config, "h5.timing_schedule", "zero_zero"))
    if schedule_name not in ENTROPY_TIMING_SCHEDULES:
        raise ValueError(f"unknown E14 timing schedule {schedule_name!r}")
    schedule = ENTROPY_TIMING_SCHEDULES[schedule_name]
    phase_a_steps = _integer(config, "h5.phase_a_steps", 64, minimum=1)
    phase_b_steps = _integer(config, "h5.phase_b_steps", 1_984, minimum=1)
    if phase_a_steps + phase_b_steps != _integer(config, "train.steps", 2_048, minimum=2):
        raise ValueError("E14 phase steps must sum exactly to train.steps")

    n_train = _integer(config, "data.n_train", 10_000, minimum=2)
    n_validation = _integer(config, "data.n_validation", 4_000, minimum=2)
    n_eval = _integer(config, "data.n_eval", 10_000, minimum=2)
    q = _float(config, "data.q", 0.75)
    k = _integer(config, "data.k", 4, minimum=1)
    max_k = _integer(config, "data.max_k", 5, minimum=k)
    state_dim = _integer(config, "data.state_dim", 8, minimum=0)
    context_bits = _integer(config, "h5.context_bits", 1, minimum=0)
    nuisance_bits = _integer(config, "h5.nuisance_bits", 8, minimum=0)

    def dataset(n: int, accuracy: float, dataset_seed: int) -> SemanticBatch:
        return _make_h5_episode_dataset(
            n,
            accuracy,
            k,
            dataset_seed,
            context_bits=context_bits,
            nuisance_bits=nuisance_bits,
            nuisance_entropy=0.0,
            active_nuisance_bits=0,
            max_k=max_k,
            state_dim=state_dim,
        )

    train = dataset(n_train, q, seed)
    iid = dataset(n_validation, q, seed + 5_003)
    conflict = dataset(n_eval, 0.0, seed + 10_003)
    model, model_report = build_model(train, config, seed, nuisance_heads=nuisance_bits)
    source = _SemanticSampler(train)
    reward = _NeutralTrajectoryReward(
        nuisance_bits,
        seed=seed + 50_021,
        stochastic_gadgets=0,
    )
    device = resolve_device(str(get_path(config, "run.device", "auto")))
    critic = _critic(train, config, "h5", device)
    evaluator = _evaluator(conflict, config)
    phase_a_logs = tuple(
        step for step in (1, 2, 4, 8, 16, 32, 64) if step <= phase_a_steps
    )
    if phase_a_steps not in phase_a_logs:
        phase_a_logs = (*phase_a_logs, phase_a_steps)
    phase_b_logs = tuple(
        step
        for step in (1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1_024, 1_984)
        if step <= phase_b_steps
    )
    if phase_b_steps not in phase_b_logs:
        phase_b_logs = (*phase_b_logs, phase_b_steps)

    started = time.perf_counter()
    phase_a = train_contextual_bandit(
        model,
        source,
        _timing_bandit_config(
            config,
            seed=seed,
            steps=phase_a_steps,
            log_steps=phase_a_logs,
            entropy_coefficient=schedule.phase_a_beta,
            reset_optimizer=True,
            reset_critic=True,
        ),
        reward_fn=reward,
        critic=critic,
        evaluator=evaluator,
        factorized_actions=True,
        factorized_action_heads=0,
    )
    if phase_a.data_generator_state is None or phase_a.action_generator_state is None:
        raise RuntimeError("E14 phase A did not expose generator continuation state")

    boundary_train = evaluate_batch(model, train, config)
    boundary_iid = evaluate_batch(model, iid, config)
    boundary_conflict = evaluate_batch(model, conflict, config)
    boundary_conflict["policy_entropy"] = _predictive_entropy(model, conflict, config)
    boundary_interventions = evaluate_standard_interventions(model, conflict, config)

    phase_b = train_contextual_bandit(
        model,
        source,
        _timing_bandit_config(
            config,
            seed=seed,
            steps=phase_b_steps,
            log_steps=phase_b_logs,
            entropy_coefficient=schedule.phase_b_beta,
            # Passing no actor optimizer is the isolated actor-Adam reset.
            # Otherwise state is retained without being cleared.
            reset_optimizer=False,
            reset_critic=schedule.reset_critic,
        ),
        reward_fn=reward,
        critic=phase_a.critic,
        actor_optimizer=(None if schedule.reset_actor_optimizer else phase_a.optimizer),
        critic_optimizer=(None if schedule.reset_critic else phase_a.critic_optimizer),
        evaluator=evaluator,
        factorized_actions=True,
        factorized_action_heads=0,
        data_generator_state=phase_a.data_generator_state,
        action_generator_state=phase_a.action_generator_state,
    )
    wall_seconds = time.perf_counter() - started

    observed_actor_carry = phase_b.optimizer is phase_a.optimizer
    observed_critic_carry = phase_b.critic_optimizer is phase_a.critic_optimizer
    if observed_actor_carry == schedule.reset_actor_optimizer:
        raise RuntimeError("E14 actor optimizer reset did not match the declared schedule")
    if observed_critic_carry == schedule.reset_critic:
        raise RuntimeError("E14 critic optimizer reset did not match the declared schedule")
    if phase_b.critic is not phase_a.critic:
        raise RuntimeError("E14 replaced rather than continued/resetting the critic object")

    total_steps = phase_a.optimizer_steps + phase_b.optimizer_steps
    total_examples = phase_a.samples_seen + phase_b.samples_seen
    if total_steps != phase_a_steps + phase_b_steps:
        raise RuntimeError("E14 optimizer-step accounting diverged from the fixed design")
    if reward.rollouts != total_examples:
        raise RuntimeError("E14 reward rollouts diverged from optimizer presentations")

    final_train = evaluate_batch(model, train, config)
    final_iid = evaluate_batch(model, iid, config)
    final_conflict = evaluate_batch(model, conflict, config)
    final_conflict["policy_entropy"] = _predictive_entropy(model, conflict, config)
    final_interventions = evaluate_standard_interventions(model, conflict, config)
    trajectory = _trajectory_summary(
        phase_a,
        phase_b,
        phase_a_steps=phase_a_steps,
        phase_b_steps=phase_b_steps,
    )
    actor_distance = _state_distance(phase_a.final_model_state, phase_b.final_model_state)
    critic_distance = _state_distance(phase_a.critic_state, phase_b.critic_state)

    combined = copy.copy(phase_b)
    combined.optimizer_steps = total_steps
    combined.samples_seen = total_examples
    trajectory_horizon = int(train.metadata["trajectory_horizon"])
    costs = _costs(
        "rl",
        combined,
        wall_seconds,
        trajectory_horizon=trajectory_horizon,
        episode_level=True,
        stochastic_branch_actions=0,
        forced_branch_actions=nuisance_bits,
    )
    reset_audit = {
        "actor_weights_reset": False,
        "actor_optimizer_reset": schedule.reset_actor_optimizer,
        "critic_weights_reset": schedule.reset_critic,
        "critic_optimizer_reset": schedule.reset_critic,
        "actor_optimizer_carried_observed": observed_actor_carry,
        "critic_optimizer_carried_observed": observed_critic_carry,
        "critic_object_reused_observed": phase_b.critic is phase_a.critic,
        "semantic_sampler_reused": True,
        "data_generator_state_continued": True,
        "action_generator_state_continued": True,
    }
    summary: dict[str, Any] = {
        "hypothesis": "h5",
        "algorithm": "rl",
        "condition": schedule_name,
        "seed": int(seed),
        "design": {
            "experiment": "E14_entropy_timing",
            "registration_status": "exploratory_post_hoc_after_complete_E10",
            "fixed_design_document": "followups.md#e14-exploratory-timing-of-entropy-exposure",
            "reason_for_cell_selection": "largest_E10_beta_0_30_response",
            "responsive_cell": {"q": 0.75, "k": 4},
            "negative_control_cell": {"q": 0.90, "k": 3},
            "independent_replication_unit": "training_seed",
            "confirmatory": False,
        },
        "cell": {"q": q, "k": k},
        "model": model_report,
        "schedule": {
            "name": schedule_name,
            "phase_a_entropy_coefficient": schedule.phase_a_beta,
            "phase_b_entropy_coefficient": schedule.phase_b_beta,
            "phase_a_steps": phase_a_steps,
            "phase_b_steps": phase_b_steps,
            "phase_boundary_global_step": phase_a_steps,
            "reset_audit": reset_audit,
        },
        "boundary": {
            "global_step": phase_a_steps,
            "examples_seen": phase_a.samples_seen,
            "train": boundary_train,
            "iid": boundary_iid,
            "final": boundary_conflict,
            "interventions": boundary_interventions,
        },
        "trajectory": trajectory,
        "parameter_dynamics": {
            "actor_l2_from_boundary": actor_distance["l2"],
            "actor_relative_l2_from_boundary": actor_distance["relative_l2"],
            "critic_l2_from_boundary": critic_distance["l2"],
            "critic_relative_l2_from_boundary": critic_distance["relative_l2"],
        },
        "nuisance": {
            "bits": nuisance_bits,
            "active_fair_branch_gadgets": 0,
            "forced_branch_gadgets": nuisance_bits,
            "branch_actions_policy_sampled": False,
            **_nuisance_diagnostics(model, train, config),
        },
        "costs": costs,
        "train": final_train,
        "iid": final_iid,
        "final": final_conflict,
        "interventions": final_interventions,
    }

    metrics: list[dict[str, Any]] = []
    metrics.extend(
        _phase_history_records(
            phase_a,
            phase="phase_a",
            condition=schedule_name,
            step_offset=0,
            examples_offset=0,
            trajectory_horizon=trajectory_horizon,
        )
    )
    metrics.extend(
        _phase_history_records(
            phase_b,
            phase="phase_b",
            condition=schedule_name,
            step_offset=phase_a_steps,
            examples_offset=phase_a.samples_seen,
            trajectory_horizon=trajectory_horizon,
        )
    )
    for split, values in (
        ("train", boundary_train),
        ("iid_eval", boundary_iid),
        ("conflict_eval", boundary_conflict),
    ):
        metrics.extend(
            _split_records(
                values,
                split=split,
                condition=schedule_name,
                stage="phase_boundary",
                global_step=phase_a_steps,
                examples_seen=phase_a.samples_seen,
            )
        )
    metrics.extend(
        _intervention_records(
            boundary_interventions,
            condition=schedule_name,
            stage="phase_boundary",
            global_step=phase_a_steps,
            examples_seen=phase_a.samples_seen,
        )
    )
    for split, values in (
        ("train", final_train),
        ("iid_eval", final_iid),
        ("conflict_eval", final_conflict),
    ):
        metrics.extend(
            _split_records(
                values,
                split=split,
                condition=schedule_name,
                stage="final",
                global_step=total_steps,
                examples_seen=total_examples,
            )
        )
    metrics.extend(
        _intervention_records(
            final_interventions,
            condition=schedule_name,
            stage="final",
            global_step=total_steps,
            examples_seen=total_examples,
        )
    )
    phase_b_area = trajectory["phase_b_rho_y_area"]
    phase_b_mean = trajectory["phase_b_mean_rho_y"]
    if not isinstance(phase_b_area, (int, float)) or not isinstance(
        phase_b_mean, (int, float)
    ):
        raise RuntimeError("E14 trajectory integration did not produce numeric outcomes")
    derived_values: dict[str, float | int] = {
        "n": 1,
        "phase_b_rho_y_area": float(phase_b_area),
        "phase_b_mean_rho_y": float(phase_b_mean),
        "actor_l2_from_boundary": actor_distance["l2"],
        "actor_relative_l2_from_boundary": actor_distance["relative_l2"],
        "critic_l2_from_boundary": critic_distance["l2"],
        "critic_relative_l2_from_boundary": critic_distance["relative_l2"],
    }
    acquisition = trajectory["phase_b_updates_to_persistent_rho_y_0_5"]
    if acquisition is not None:
        derived_values["phase_b_updates_to_persistent_rho_y_0_5"] = int(acquisition)
    metrics.extend(
        _split_records(
            derived_values,
            split="timing_diagnostics",
            condition=schedule_name,
            stage="final",
            global_step=total_steps,
            examples_seen=total_examples,
        )
    )
    metrics.extend(
        _split_records(
            {name: value for name, value in costs.items() if isinstance(value, (int, float))},
            split="cost",
            condition=schedule_name,
            stage="final",
            global_step=total_steps,
            examples_seen=total_examples,
        )
    )

    checkpoints: dict[str, dict[str, Any]] = {}
    if bool(get_path(config, "run.save_checkpoints", False)):
        checkpoints = {
            "phase_a_final": _checkpoint_payload(phase_a, phase_a_steps),
            "phase_b_final": _checkpoint_payload(phase_b, total_steps),
        }
    predictions = (
        prediction_records(model, conflict, config, split="conflict_eval")
        if bool(get_path(config, "evaluation.save_predictions", False))
        else []
    )
    return ProtocolResult(
        model=model,
        summary=summary,
        metrics=metrics,
        predictions=predictions,
        checkpoints=checkpoints,
        evaluation_batch=conflict,
    )


__all__ = ["ENTROPY_TIMING_SCHEDULES", "EntropyTimingSchedule", "run_entropy_timing"]
