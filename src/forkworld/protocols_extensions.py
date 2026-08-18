"""Follow-up protocols for multi-goal competition and sequential RouteWorld.

The original experiment families deliberately use :class:`SemanticBatch` and a
single reward-relevant choice.  H10 retains that interface while adding a second
imperfect rule.  H11 uses :class:`RouteDecisionBatch`, whose rows are the true
junctions of a complete route, and therefore has a small vector-batch adapter
rather than pretending the task is another one-step semantic batch.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping, Sequence
from typing import Any, cast

import numpy as np
import torch
from numpy.typing import NDArray
from torch import nn

from .competing import (
    competing_causal_flip_batches,
    competing_rule_agreements,
    make_competing_bundle,
)
from .config import get_path
from .data import SemanticBatch, mask_channels
from .metrics import intervention_metrics
from .models import (
    GoalMLP,
    MLPConfig,
    UpdateMode,
    configure_update_mode,
    parameter_count,
    trainable_parameter_count,
)
from .protocols import (
    ProtocolResult,
    make_metric_records,
    predict_logits,
    resolve_device,
    seed_everything,
)
from .protocols_selection import _checkpoint_states, _fit, _sft_config
from .routeworld import (
    BinaryTreeRouteMaze,
    RouteDecisionBatch,
    flip_route_code,
    flip_route_proxy,
    make_route_dataset,
    make_route_panels,
    permute_route_addresses,
    rollout_route,
    route_metrics,
)
from .training import TrainingResult, TrainingSnapshot, train_clean_sft


def _integer(config: Mapping[str, Any], path: str, default: int) -> int:
    value = get_path(config, path, default)
    if isinstance(value, bool):
        raise ValueError(f"{path} must be an integer")
    return int(value)


def _float(config: Mapping[str, Any], path: str, default: float) -> float:
    return float(get_path(config, path, default))


def _retarget_and_mask(
    batch: SemanticBatch,
    *,
    target: np.ndarray,
    keep: Sequence[str],
    calibration_target: str,
) -> SemanticBatch:
    """Keep a fixed interface while making only one candidate rule informative."""

    retained = set(keep)
    masked = tuple(name for name in batch.channels if name not in retained)
    calibrated = mask_channels(batch, masked) if masked else batch
    metadata = dict(calibrated.metadata)
    metadata.update(
        {
            "enforce_exact_code": False,
            "neutralize_padding": True,
            "calibration_interface": "competition_full_width",
            "calibration_target": calibration_target,
            "calibration_informative_channels": tuple(keep),
        }
    )
    return calibrated.with_updates(
        target=np.asarray(target, dtype=np.int8),
        reward=np.asarray(target, dtype=np.float32),
        state=None if batch.state is None else np.zeros_like(batch.state),
        metadata=metadata,
    )


def _candidate_metrics(model: nn.Module, batch: SemanticBatch, config: Mapping[str, Any]) -> dict[str, float | int]:
    return competing_rule_agreements(predict_logits(model, batch, config), batch)


def _candidate_interventions(
    model: nn.Module,
    batch: SemanticBatch,
    config: Mapping[str, Any],
) -> dict[str, dict[str, float]]:
    base = predict_logits(model, batch, config)
    raw: dict[str, dict[str, float]] = {}
    for name, changed_batch in competing_causal_flip_batches(batch).items():
        changed = predict_logits(model, changed_batch, config)
        raw[name] = intervention_metrics(base, changed, np.asarray(batch.y))

    for family, prefix in (("Q_mean", "flip_Q_"), ("Y_mean", "flip_Y_")):
        members = [value for name, value in raw.items() if name.startswith(prefix)]
        if members:
            raw[family] = {
                key: float(np.mean([member[key] for member in members]))
                for key in ("probability_ate", "logit_ate", "hard_flip_rate", "n")
            }
    return raw


def _append_candidate_records(
    records: list[dict[str, Any]],
    values: Mapping[str, float | int],
    *,
    split: str,
    step: int,
    examples_seen: int,
    condition: str,
    intervention: str = "none",
) -> None:
    records.extend(
        make_metric_records(
            values,
            hypothesis="h10",
            split=split,
            global_step=step,
            examples_seen=examples_seen,
            condition=condition,
            intervention=intervention,
            stage="competition",
        )
    )


def run_h10(config: Mapping[str, Any], seed: int) -> ProtocolResult:
    """Calibrate and compete a direct proxy, encoded proxy, and exact rule."""

    seed_everything(seed)
    n_train = _integer(config, "data.n_train", 10_000)
    n_validation = _integer(config, "data.n_validation", 4_000)
    n_eval = _integer(config, "data.n_eval", 10_000)
    k_q = _integer(config, "h10.k_q", 2)
    k_y = _integer(config, "h10.k_y", 3)
    max_k_q = _integer(config, "h10.max_k_q", k_q)
    max_k_y = _integer(config, "h10.max_k_y", k_y)
    q_p = _float(config, "h10.q_p", 0.9)
    q_q = _float(config, "h10.q_q", 0.95)
    overlap = str(get_path(config, "h10.error_structure", "independent"))
    state_dim = _integer(config, "data.state_dim", 0)
    condition = str(get_path(config, "experiment.mode", "calibrated_three_goal_competition"))

    bundle = make_competing_bundle(
        n_train,
        n_validation,
        n_eval,
        q_p,
        q_q,
        k_q,
        k_y,
        seed,
        max_k_q=max_k_q,
        max_k_y=max_k_y,
        overlap=overlap,  # type: ignore[arg-type]
        state_dim=state_dim,
    )
    batches = {"iid": bundle.iid, **dict(bundle.diagnostics)}
    q_names = tuple(f"Q_{index}" for index in range(1, k_q + 1))
    y_names = tuple(f"R_{index}" for index in range(1, k_y + 1))

    task_builders: dict[str, Callable[[SemanticBatch], SemanticBatch]] = {
        "P": lambda batch: _retarget_and_mask(
            batch,
            target=np.asarray(batch.latents["P_goal"]),
            keep=("P",),
            calibration_target="P",
        ),
        "Q": lambda batch: _retarget_and_mask(
            batch,
            target=np.asarray(batch.latents["Q_goal"]),
            keep=q_names,
            calibration_target="Q",
        ),
        "Y": lambda batch: _retarget_and_mask(
            batch,
            target=np.asarray(batch.y),
            keep=y_names,
            calibration_target="Y",
        ),
    }
    calibration_steps = _integer(
        config, "h10.calibration_steps", _integer(config, "train.steps", 2_048)
    )
    calibration_models: dict[str, nn.Module] = {}
    calibration_reports: dict[str, dict[str, Any]] = {}
    calibration_fits: dict[str, Any] = {}
    feature_names = bundle.train.feature_names(
        max_k=_integer(config, "data.max_k", max_k_y), include_state=True
    )

    # The same seed gives every candidate an exactly matched initialization.
    for name, transform in task_builders.items():
        train = transform(bundle.train)
        evaluation = {
            f"{name}_calibration_{split}": transform(batch)
            for split, batch in batches.items()
        }
        if train.feature_names(
            max_k=_integer(config, "data.max_k", max_k_y), include_state=True
        ) != feature_names:
            raise RuntimeError(f"H10 {name} calibration changed the competition interface")
        from .protocols import build_model

        model, report = build_model(train, config, seed)
        fit = _fit(
            model,
            train,
            evaluation,
            config,
            seed,
            hypothesis="h10",
            condition=condition,
            stage=f"{name}_calibration",
            steps=calibration_steps,
        )
        calibration_models[name] = model
        calibration_reports[name] = report
        calibration_fits[name] = fit

    from .protocols import build_model

    competition_model, competition_report = build_model(bundle.train, config, seed)
    for name, report in calibration_reports.items():
        for field in ("input_dim", "total_parameters", "trainable_parameters"):
            if report[field] != competition_report[field]:
                raise RuntimeError(f"H10 {name} calibration does not match competition {field}")
    competition = _fit(
        competition_model,
        bundle.train,
        {f"competition_{split}": batch for split, batch in batches.items()},
        config,
        seed,
        hypothesis="h10",
        condition=condition,
        stage="competition",
        steps=_integer(config, "h10.competition_steps", _integer(config, "train.steps", 2_048)),
    )

    final_competition = {
        split: _candidate_metrics(competition_model, batch, config)
        for split, batch in batches.items()
    }
    final_interventions = {
        split: _candidate_interventions(competition_model, batch, config)
        for split, batch in bundle.diagnostics.items()
    }
    decoder_metric = {"P": "rho_p", "Q": "rho_q", "Y": "rho_y_code"}
    final_calibration: dict[str, dict[str, dict[str, float | int]]] = {}
    for name, model in calibration_models.items():
        transformed_results: dict[str, dict[str, float | int]] = {}
        for split, batch in batches.items():
            # Calibration must be evaluated on the masked interface used in
            # training. Reactivating never-trained channels would add arbitrary
            # random-initialization contributions to accessibility estimates.
            values = _candidate_metrics(model, task_builders[name](batch), config)
            values["decoder_accuracy"] = float(values[decoder_metric[name]])
            transformed_results[split] = values
        final_calibration[name] = transformed_results

    metrics = [
        record
        for fit in calibration_fits.values()
        for record in fit.metrics
    ] + competition.metrics
    final_step = competition.training.optimizer_steps
    examples_seen = competition.training.samples_seen
    for split, values in final_competition.items():
        _append_candidate_records(
            metrics,
            values,
            split=f"competition_{split}",
            step=final_step,
            examples_seen=examples_seen,
            condition=condition,
        )
    for split, effects in final_interventions.items():
        for intervention, values in effects.items():
            _append_candidate_records(
                metrics,
                values,
                split=f"competition_{split}",
                step=final_step,
                examples_seen=examples_seen,
                condition=condition,
                intervention=intervention,
            )

    checkpoints: dict[str, dict[str, Any]] = {}
    for name, fit in calibration_fits.items():
        checkpoints.update(_checkpoint_states(fit.training, f"{name}_calibration_"))
    checkpoints.update(_checkpoint_states(competition.training, "competition_"))
    total_examples = competition.training.samples_seen + sum(
        fit.training.samples_seen for fit in calibration_fits.values()
    )
    total_seconds = competition.wall_seconds + sum(
        fit.wall_seconds for fit in calibration_fits.values()
    )
    summary = {
        "hypothesis": "h10",
        "seed": seed,
        "condition": condition,
        "final": {
            "calibration": final_calibration,
            "competition": final_competition,
            "interventions": final_interventions,
        },
        "model": {"calibration": calibration_reports, "competition": competition_report},
        "data": {
            "n_train": n_train,
            "n_validation": n_validation,
            "n_eval_per_panel": n_eval,
            "q_p": q_p,
            "q_q": q_q,
            "k_q": k_q,
            "k_y": k_y,
            "max_k_q": max_k_q,
            "max_k_y": max_k_y,
            "error_structure": overlap,
            "training_overlap": {
                key: bundle.train.metadata[key]
                for key in (
                    "p_error_count",
                    "q_error_count",
                    "both_error_count",
                    "p_only_error_count",
                    "q_only_error_count",
                    "error_phi",
                )
            },
            "calibration_interface_matches": True,
            "calibration_initialization_seed": seed,
            "calibration_evaluation_uses_masked_interface": True,
        },
        "training": {
            "calibration_steps_each": calibration_steps,
            "competition_steps": competition.training.optimizer_steps,
            "examples_seen": total_examples,
            "wall_seconds": total_seconds,
        },
    }
    return ProtocolResult(
        model=competition_model,
        summary=summary,
        metrics=metrics,
        predictions=[],
        checkpoints=checkpoints,
        evaluation_batch=bundle.both_wrong,
    )


def _build_route_model(
    batch: RouteDecisionBatch, config: Mapping[str, Any], seed: int
) -> tuple[nn.Module, dict[str, Any]]:
    seed_everything(seed)
    base = GoalMLP(
        MLPConfig(
            input_dim=batch.input_dim,
            width=_integer(config, "model.width", 64),
            depth=_integer(config, "model.depth", 2),
            activation=str(get_path(config, "model.activation", "relu")),  # type: ignore[arg-type]
            residual=bool(get_path(config, "model.residual", False)),
            bias=bool(get_path(config, "model.bias", True)),
        )
    )
    total = parameter_count(base)
    requested_mode = str(get_path(config, "update.mode", "full"))
    mode = cast(
        UpdateMode,
        "head_only" if requested_mode in {"head", "head-only"} else requested_mode,
    )
    raw_budget = get_path(config, "update.budget", "full")
    budget = None if raw_budget in (None, "full") else int(raw_budget)
    model = configure_update_mode(
        base,
        mode=mode,
        budget=budget,
        projection=str(get_path(config, "update.projection", "count_sketch")),  # type: ignore[arg-type]
        seed=_integer(config, "update.subspace_seed", 1_729) + seed,
        scale=_float(config, "update.scale", 1.0),
        include_nuisance_heads=True,
    )
    return model, {
        "input_dim": batch.input_dim,
        "total_parameters": total,
        "trainable_parameters": trainable_parameter_count(model),
        "update_mode": mode,
        "requested_budget": raw_budget,
    }


@torch.no_grad()
def _route_logits(model: nn.Module, batch: RouteDecisionBatch) -> NDArray[np.float64]:
    device = next(model.parameters()).device
    x = torch.as_tensor(batch.x, dtype=torch.float32, device=device)
    was_training = model.training
    model.eval()
    output = model(x)
    if not torch.is_tensor(output):  # pragma: no cover - GoalMLP contract
        output = output.goal_logits
    logits = cast(
        NDArray[np.float64],
        output.detach().cpu().numpy().astype(np.float64).reshape(-1),
    )
    model.train(was_training)
    return logits


def _flat_route_metrics(values: Mapping[str, Any]) -> dict[str, float | int]:
    flat: dict[str, float | int] = {}
    for key, value in values.items():
        if isinstance(value, (int, float, np.number)) and not isinstance(value, bool):
            flat[key] = float(value) if not isinstance(value, int) else value
        elif key == "stage_branch_accuracy":
            flat.update({f"stage_{index}_branch_accuracy": float(item) for index, item in enumerate(value)})
        elif key == "first_divergence_rate":
            flat.update({f"first_divergence_{index}_rate": float(item) for index, item in enumerate(value)})
    flat["n"] = int(values["n_branches"])
    return flat


def _two_class_route_logits(values: NDArray[np.float64]) -> NDArray[np.float64]:
    return np.column_stack((-0.5 * values, 0.5 * values))


def _mean_intervention_effects(
    effects: Sequence[Mapping[str, float]],
) -> dict[str, float]:
    return {
        key: float(np.mean([effect[key] for effect in effects]))
        for key in ("probability_ate", "logit_ate", "hard_flip_rate", "n")
    }


def _route_snapshot(
    model: nn.Module,
    evaluation: Mapping[str, RouteDecisionBatch],
    *,
    step: int,
    examples_seen: int,
    condition: str,
    records: list[dict[str, Any]],
) -> dict[str, Any]:
    snapshot: dict[str, Any] = {"step": step, "examples_seen": examples_seen}
    for split, batch in evaluation.items():
        values = route_metrics(_route_logits(model, batch), batch)
        snapshot[split] = values
        records.extend(
            make_metric_records(
                _flat_route_metrics(values),
                hypothesis="h11",
                split=split,
                global_step=step,
                examples_seen=examples_seen,
                condition=condition,
                stage="train",
            )
        )
    return snapshot


def _fit_route(
    model: nn.Module,
    train: RouteDecisionBatch,
    evaluation: Mapping[str, RouteDecisionBatch],
    config: Mapping[str, Any],
    seed: int,
    *,
    steps: int,
    condition: str,
) -> tuple[TrainingResult, list[dict[str, Any]], list[dict[str, Any]], float]:
    records: list[dict[str, Any]] = []
    trajectory: list[dict[str, Any]] = []
    model.to(resolve_device(str(get_path(config, "run.device", "auto"))))
    trajectory.append(
        _route_snapshot(
            model,
            evaluation,
            step=0,
            examples_seen=0,
            condition=condition,
            records=records,
        )
    )

    def callback(snapshot: TrainingSnapshot) -> None:
        trajectory.append(
            _route_snapshot(
                snapshot.model,
                evaluation,
                step=snapshot.step,
                examples_seen=snapshot.record.samples_seen,
                condition=condition,
                records=records,
            )
        )

    started = time.perf_counter()
    training = train_clean_sft(
        model,
        (
            torch.as_tensor(train.x, dtype=torch.float32),
            torch.as_tensor(train.y, dtype=torch.float32),
        ),
        _sft_config(config, seed, steps=steps),
        callback=callback,
    )
    wall_seconds = time.perf_counter() - started
    for record in training.history:
        records.extend(
            make_metric_records(
                {
                    "loss": record.loss,
                    "primary_loss": record.primary_loss,
                    "train_batch_accuracy": record.metrics.get("train_accuracy", float("nan")),
                    "optimizer_steps": record.optimizer_steps,
                    "n": len(train),
                },
                hypothesis="h11",
                split="train_minibatch",
                global_step=record.step,
                examples_seen=record.samples_seen,
                condition=condition,
                stage="train",
            )
        )
    return training, records, trajectory, wall_seconds


def run_h11(config: Mapping[str, Any], seed: int) -> ProtocolResult:
    """Train one shared selector at every reward-relevant fork of RouteWorld."""

    seed_everything(seed)
    depth = _integer(config, "h11.route_depth", 1)
    max_depth = _integer(config, "h11.max_depth", 4)
    q = _float(config, "data.q", 0.95)
    k = _integer(config, "data.k", 2)
    max_k = _integer(config, "data.max_k", 4)
    n_train = _integer(config, "data.n_train", 4_000)
    n_eval = _integer(config, "data.n_eval", 2_000)
    evidence_regime = str(get_path(config, "h11.evidence_regime", "fixed_total"))
    base_steps = _integer(config, "h11.base_steps", _integer(config, "train.steps", 1_024))
    steps = base_steps if evidence_regime == "fixed_total" else base_steps * depth
    condition = f"{get_path(config, 'experiment.mode', 'routeworld')}:{evidence_regime}"

    train = make_route_dataset(
        n_train,
        depth=depth,
        q=q,
        k=k,
        seed=seed,
        max_depth=max_depth,
        max_k=max_k,
    )
    panels = make_route_panels(
        n_eval,
        depth=depth,
        q=q,
        k=k,
        seed=seed + 1,
        max_depth=max_depth,
        max_k=max_k,
    )
    evaluation = panels.as_dict()
    model, model_report = _build_route_model(train, config, seed)
    training, metrics, trajectory, wall_seconds = _fit_route(
        model,
        train,
        evaluation,
        config,
        seed,
        steps=steps,
        condition=condition,
    )
    final = {
        split: route_metrics(_route_logits(model, batch), batch)
        for split, batch in evaluation.items()
    }
    reversed_address = permute_route_addresses(
        panels.all_conflict, tuple(reversed(range(depth)))
    )
    removed_address = panels.all_conflict.with_updates(
        stage_features=np.zeros_like(panels.all_conflict.stage_features)
    )
    final["all_conflict_address_reversed"] = route_metrics(
        _route_logits(model, reversed_address), reversed_address
    )
    final["all_conflict_address_removed"] = route_metrics(
        _route_logits(model, removed_address), removed_address
    )
    for split in ("all_conflict_address_reversed", "all_conflict_address_removed"):
        metrics.extend(
            make_metric_records(
                _flat_route_metrics(final[split]),
                hypothesis="h11",
                split=split,
                global_step=training.optimizer_steps,
                examples_seen=training.samples_seen,
                condition=condition,
                stage="final",
            )
        )
    base_logits = _route_logits(model, panels.all_conflict)
    proxy_flip_logits = _route_logits(model, flip_route_proxy(panels.all_conflict))
    code_flip_logits = _route_logits(
        model, flip_route_code(panels.all_conflict, channels=(0,))
    )
    interventions: dict[str, Any] = {
        "flip_P_all_stages": intervention_metrics(
            _two_class_route_logits(base_logits),
            _two_class_route_logits(proxy_flip_logits),
            panels.all_conflict.y,
        ),
        "flip_R1_all_stages": intervention_metrics(
            _two_class_route_logits(base_logits),
            _two_class_route_logits(code_flip_logits),
            panels.all_conflict.y,
        ),
    }
    base_by_stage = base_logits.reshape(n_eval, depth)
    stage_local: dict[str, Any] = {}
    local_proxy_effects: list[dict[str, float]] = []
    local_code_effects: list[dict[str, float]] = []
    proxy_spillovers: list[dict[str, float]] = []
    code_spillovers: list[dict[str, float]] = []
    for stage_index in range(depth):
        proxy_changed = _route_logits(
            model, flip_route_proxy(panels.all_conflict, [stage_index])
        ).reshape(n_eval, depth)
        code_changed = _route_logits(
            model,
            flip_route_code(
                panels.all_conflict, stages=[stage_index], channels=(0,)
            ),
        ).reshape(n_eval, depth)
        target = panels.all_conflict.targets[:, stage_index]
        proxy_effect = intervention_metrics(
            _two_class_route_logits(base_by_stage[:, stage_index]),
            _two_class_route_logits(proxy_changed[:, stage_index]),
            target,
        )
        code_effect = intervention_metrics(
            _two_class_route_logits(base_by_stage[:, stage_index]),
            _two_class_route_logits(code_changed[:, stage_index]),
            target,
        )
        local_proxy_effects.append(proxy_effect)
        local_code_effects.append(code_effect)
        entry: dict[str, Any] = {"flip_P": proxy_effect, "flip_R1": code_effect}
        other_stages = [index for index in range(depth) if index != stage_index]
        if other_stages:
            other_target = panels.all_conflict.targets[:, other_stages].reshape(-1)
            proxy_spillover = intervention_metrics(
                _two_class_route_logits(base_by_stage[:, other_stages].reshape(-1)),
                _two_class_route_logits(proxy_changed[:, other_stages].reshape(-1)),
                other_target,
            )
            code_spillover = intervention_metrics(
                _two_class_route_logits(base_by_stage[:, other_stages].reshape(-1)),
                _two_class_route_logits(code_changed[:, other_stages].reshape(-1)),
                other_target,
            )
            proxy_spillovers.append(proxy_spillover)
            code_spillovers.append(code_spillover)
            entry["flip_P_other_stage_spillover"] = proxy_spillover
            entry["flip_R1_other_stage_spillover"] = code_spillover
        stage_local[f"stage_{stage_index}"] = entry
    interventions["flip_P_stage_local_mean"] = _mean_intervention_effects(
        local_proxy_effects
    )
    interventions["flip_R1_stage_local_mean"] = _mean_intervention_effects(
        local_code_effects
    )
    if proxy_spillovers:
        interventions["flip_P_other_stage_spillover_mean"] = (
            _mean_intervention_effects(proxy_spillovers)
        )
        interventions["flip_R1_other_stage_spillover_mean"] = (
            _mean_intervention_effects(code_spillovers)
        )
    interventions["stage_local"] = stage_local
    for name, values in interventions.items():
        if name == "stage_local":
            continue
        metrics.extend(
            make_metric_records(
                values,
                hypothesis="h11",
                split="all_conflict",
                global_step=training.optimizer_steps,
                examples_seen=training.samples_seen,
                condition=condition,
                intervention=name,
                stage="final",
            )
        )

    requested_rollouts = _integer(config, "h11.physical_rollouts", 128)
    n_rollouts = min(requested_rollouts, panels.all_conflict.n_episodes)
    predicted = np.where(base_logits >= 0.0, 1, -1).astype(np.int8).reshape(n_eval, depth)
    successes: list[bool] = []
    branch_accuracies: list[float] = []
    collisions: list[int] = []
    for episode in range(n_rollouts):
        intended = tuple(int(value) for value in panels.all_conflict.targets[episode, :depth])
        selected = tuple(int(value) for value in predicted[episode])
        rollout = rollout_route(BinaryTreeRouteMaze(depth, intended), selected)
        successes.append(rollout.success)
        branch_accuracies.append(rollout.branch_accuracy)
        collisions.append(rollout.collisions)
    physical = {
        "n": n_rollouts,
        "full_route_success": float(np.mean(successes)),
        "branch_accuracy": float(np.mean(branch_accuracies)),
        "collision_rate": float(np.mean(np.asarray(collisions) > 0)),
        "matches_vectorized_success": bool(
            np.array_equal(
                np.asarray(successes),
                np.all(
                    predicted[:n_rollouts]
                    == panels.all_conflict.targets[:n_rollouts, :depth],
                    axis=1,
                ),
            )
        ),
    }
    summary = {
        "hypothesis": "h11",
        "seed": seed,
        "condition": condition,
        "final": {
            **final,
            "interventions": interventions,
            "physical_rollouts": physical,
            "trajectory_points": len(trajectory),
        },
        "model": model_report,
        "data": {
            "n_train_episodes": n_train,
            "n_eval_episodes": n_eval,
            "route_depth": depth,
            "max_depth": max_depth,
            "q": q,
            "k": k,
            "max_k": max_k,
            "feature_names": train.feature_names,
            "physical_route_length": BinaryTreeRouteMaze(depth, (-1,) * depth).route_length,
        },
        "training": {
            "evidence_regime": evidence_regime,
            "base_steps": base_steps,
            "steps": training.optimizer_steps,
            "examples_seen": training.samples_seen,
            "rows_per_training_episode": depth,
            "wall_seconds": wall_seconds,
        },
    }
    return ProtocolResult(
        model=model,
        summary=summary,
        metrics=metrics,
        predictions=[],
        checkpoints=_checkpoint_states(training),
        evaluation_batch=None,
    )


RUNNERS = {"h10": run_h10, "h11": run_h11}


__all__ = ["RUNNERS", "run_h10", "run_h11"]
