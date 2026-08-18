"""Selection, calibration, dynamics, and conflict-diversity protocols (H1--H4)."""

from __future__ import annotations

import math
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from torch import nn

from .config import get_path
from .data import (
    DatasetBundle,
    SemanticBatch,
    flip_channels,
    flip_exact_rule_output,
    make_conflict_dataset,
    make_h4_dataset,
    make_standard_dataset,
    mask_channels,
    remove_proxy,
)
from .metrics import (
    TimeToEvent,
    acquisition_time,
    intervention_metrics,
    replacement_time,
    sequential_replacement_time,
)
from .protocols import (
    ProtocolResult,
    batch_for_training,
    build_model,
    evaluate_batch,
    evaluate_standard_interventions,
    make_metric_records,
    predict_logits,
    prediction_records,
    resolve_device,
    seed_everything,
)
from .training import SFTConfig, TrainingResult, TrainingSnapshot, train_clean_sft


@dataclass
class _FitOutput:
    training: TrainingResult
    metrics: list[dict[str, Any]]
    trajectory: list[dict[str, Any]]
    wall_seconds: float


def _integer(config: Mapping[str, Any], path: str, default: int) -> int:
    value = get_path(config, path, default)
    if isinstance(value, bool):
        raise ValueError(f"{path} must be an integer")
    return int(value)


def _float(config: Mapping[str, Any], path: str, default: float) -> float:
    return float(get_path(config, path, default))


def _dataset_arguments(config: Mapping[str, Any]) -> dict[str, int]:
    return {
        "max_k": _integer(config, "data.max_k", 5),
        "state_dim": _integer(config, "data.state_dim", 0),
    }


def _evaluation_steps(config: Mapping[str, Any], steps: int) -> Sequence[int] | None:
    value = get_path(config, "train.eval_steps", "log")
    if value in (None, "log"):
        return None
    if value in ("dense", "all", "every"):
        return tuple(range(1, steps + 1))
    if isinstance(value, int) and not isinstance(value, bool):
        if value < 1:
            raise ValueError("train.eval_steps integer must be positive")
        points = set(range(value, steps + 1, value))
        points.add(steps)
        return tuple(sorted(points))
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return tuple(int(item) for item in value)
    raise ValueError("train.eval_steps must be log, dense, an interval, or a list")


def _sft_config(
    config: Mapping[str, Any],
    seed: int,
    *,
    steps: int | None = None,
) -> SFTConfig:
    steps = _integer(config, "train.steps", 1_000) if steps is None else int(steps)
    return SFTConfig(
        steps=steps,
        batch_size=_integer(config, "train.batch_size", 128),
        learning_rate=_float(config, "train.learning_rate", 3e-3),
        weight_decay=_float(config, "train.weight_decay", 0.0),
        optimizer=str(get_path(config, "train.optimizer", "adamw")),  # type: ignore[arg-type]
        momentum=_float(config, "train.momentum", 0.0),
        seed=seed,
        device=resolve_device(str(get_path(config, "run.device", "auto"))),
        deterministic=bool(get_path(config, "train.deterministic", True)),
        shuffle=bool(get_path(config, "train.shuffle", True)),
        gradient_clip_norm=get_path(config, "train.grad_clip", 1.0),
        log_steps=_evaluation_steps(config, steps),
        num_log_points=_integer(config, "train.num_log_points", 25),
        save_checkpoints=bool(get_path(config, "run.save_checkpoints", True)),
        reset_model=False,
        reset_optimizer=True,
    )


def _metric_snapshot(
    model: nn.Module,
    evaluation: Mapping[str, SemanticBatch],
    config: Mapping[str, Any],
    *,
    hypothesis: str,
    condition: str,
    step: int,
    examples_seen: int,
    metrics: list[dict[str, Any]],
    intervention_splits: Sequence[str] = (),
    stage: str = "train",
    include_state: bool = True,
) -> dict[str, Any]:
    point: dict[str, Any] = {"step": int(step), "examples_seen": int(examples_seen)}
    for split, batch in evaluation.items():
        values = evaluate_batch(model, batch, config, include_state=include_state)
        point[split] = values
        metrics.extend(
            make_metric_records(
                values,
                hypothesis=hypothesis,
                split=split,
                global_step=step,
                examples_seen=examples_seen,
                condition=condition,
                stage=stage,
            )
        )
        if split not in intervention_splits:
            continue
        for intervention, effects in evaluate_standard_interventions(
            model, batch, config, include_state=include_state
        ).items():
            metrics.extend(
                make_metric_records(
                    effects,
                    hypothesis=hypothesis,
                    split=split,
                    global_step=step,
                    examples_seen=examples_seen,
                    intervention=intervention,
                    condition=condition,
                    stage=stage,
                )
            )
    return point


def _fit(
    model: nn.Module,
    train: SemanticBatch,
    evaluation: Mapping[str, SemanticBatch],
    config: Mapping[str, Any],
    seed: int,
    *,
    hypothesis: str,
    condition: str,
    intervention_splits: Sequence[str] = (),
    stage: str = "train",
    include_state: bool = True,
    steps: int | None = None,
    snapshot_observer: Callable[[nn.Module, int, int], None] | None = None,
) -> _FitOutput:
    metrics: list[dict[str, Any]] = []
    trajectory: list[dict[str, Any]] = []
    model.to(resolve_device(str(get_path(config, "run.device", "auto"))))
    trajectory.append(
        _metric_snapshot(
            model,
            evaluation,
            config,
            hypothesis=hypothesis,
            condition=condition,
            step=0,
            examples_seen=0,
            metrics=metrics,
            intervention_splits=intervention_splits,
            stage=stage,
            include_state=include_state,
        )
    )
    if snapshot_observer is not None:
        snapshot_observer(model, 0, 0)

    def callback(snapshot: TrainingSnapshot) -> None:
        trajectory.append(
            _metric_snapshot(
                snapshot.model,
                evaluation,
                config,
                hypothesis=hypothesis,
                condition=condition,
                step=snapshot.step,
                examples_seen=snapshot.record.samples_seen,
                metrics=metrics,
                intervention_splits=intervention_splits,
                stage=stage,
                include_state=include_state,
            )
        )
        if snapshot_observer is not None:
            snapshot_observer(
                snapshot.model,
                snapshot.step,
                snapshot.record.samples_seen,
            )

    wall_start = time.perf_counter()
    training = train_clean_sft(
        model,
        batch_for_training(train, config, include_state=include_state),
        _sft_config(config, seed, steps=steps),
        callback=callback,
    )
    wall_seconds = time.perf_counter() - wall_start

    # Optimization diagnostics use the same long-form schema as behavioral data.
    for record in training.history:
        values: dict[str, float | int] = {
            "loss": record.loss,
            "primary_loss": record.primary_loss,
            "train_batch_accuracy": record.metrics.get("train_accuracy", float("nan")),
            "optimizer_steps": record.optimizer_steps,
            "n": len(train),
        }
        metrics.extend(
            make_metric_records(
                values,
                hypothesis=hypothesis,
                split="train_minibatch",
                global_step=record.step,
                examples_seen=record.samples_seen,
                condition=condition,
                stage=stage,
            )
        )
    return _FitOutput(training, metrics, trajectory, wall_seconds)


def _checkpoint_states(training: TrainingResult, prefix: str = "") -> dict[str, dict[str, Any]]:
    result = {
        f"{prefix}step_{step}": checkpoint.model_state
        for step, checkpoint in sorted(training.checkpoints.items())
    }
    result[f"{prefix}final"] = training.final_model_state
    return result


def _event(values: Sequence[dict[str, Any]], split: str, metric: str, threshold: float, persistence: int) -> TimeToEvent:
    steps = [int(point["step"]) for point in values]
    observations = [float(point[split][metric]) for point in values]
    return acquisition_time(steps, observations, threshold=threshold, persistence=persistence)


def _standard_events(
    trajectory: Sequence[dict[str, Any]], config: Mapping[str, Any], split: str = "conflict"
) -> dict[str, int | bool | str]:
    threshold = _float(config, "evaluation.acquisition_threshold", 0.9)
    persistence = _integer(config, "evaluation.persistence", 2)
    steps = [int(point["step"]) for point in trajectory]
    rho_y = [float(point[split]["rho_y"]) for point in trajectory]
    rho_p = [float(point[split]["rho_p"]) for point in trajectory]
    intended = acquisition_time(steps, rho_y, threshold, persistence)
    proxy = acquisition_time(steps, rho_p, threshold, persistence)
    raw_dominance = replacement_time(steps, rho_y, rho_p, 0.0, persistence)
    sequential_replacement = sequential_replacement_time(
        steps,
        rho_y,
        rho_p,
        acquisition_threshold=threshold,
        margin=0.0,
        persistence=persistence,
    )
    return {
        **intended.as_dict("intended_acquisition"),
        **proxy.as_dict("proxy_acquisition"),
        **raw_dominance.as_dict("intended_dominance_crossing"),
        **sequential_replacement.as_dict("proxy_to_intended_replacement"),
        **sequential_replacement.as_dict("sequential_proxy_to_intended_replacement"),
        "sequential_replacement_eligible": bool(proxy.observed),
        "replacement_definition": (
            "sustained intended-over-proxy dominance strictly after sustained proxy acquisition"
        ),
    }


def _final_predictions(
    model: nn.Module,
    batches: Mapping[str, SemanticBatch],
    config: Mapping[str, Any],
    *,
    interventions: bool = False,
    stage: str | None = None,
    include_state: bool = True,
) -> list[dict[str, Any]]:
    if not bool(get_path(config, "evaluation.save_predictions", True)):
        return []
    records: list[dict[str, Any]] = []
    for split, batch in batches.items():
        current = prediction_records(
            model, batch, config, split=split, include_state=include_state
        )
        if stage is not None:
            for record in current:
                record["stage"] = stage
        records.extend(current)
        if not interventions:
            continue
        names = ["P"] if "P" in batch.channels else []
        names.extend(f"R_{index}" for index in range(1, batch.active_k + 1))
        for name in names:
            changed = prediction_records(
                model,
                flip_channels(batch, name),
                config,
                split=split,
                intervention=f"flip_{name}",
                include_state=include_state,
            )
            if stage is not None:
                for record in changed:
                    record["stage"] = stage
            records.extend(changed)
    return records


def _standard_data(
    config: Mapping[str, Any], seed: int
) -> tuple[SemanticBatch, SemanticBatch, SemanticBatch]:
    n_train = _integer(config, "data.n_train", 4096)
    n_validation = _integer(config, "data.n_validation", 2048)
    n_eval = _integer(config, "data.n_eval", 4096)
    q = _float(config, "data.q", 0.9)
    k = _integer(config, "data.k", 3)
    target_rule = str(get_path(config, "data.target_rule", "parity"))
    arguments = _dataset_arguments(config)
    train = make_standard_dataset(
        n_train,
        q,
        k,
        seed,
        split="train",
        target_rule=target_rule,
        **arguments,
    )
    validation = make_standard_dataset(
        n_validation,
        q,
        k,
        seed + 1,
        split="validation",
        id_offset=n_train + 1,
        target_rule=target_rule,
        **arguments,
    )
    conflict = make_conflict_dataset(
        n_eval,
        k,
        seed + 2,
        id_offset=n_train + n_validation + 2,
        target_rule=target_rule,
        **arguments,
    )
    return train, validation, conflict


def _single_channel_bayes_accuracy(channel: np.ndarray, target: np.ndarray) -> float:
    """Empirical Bayes accuracy available from one discrete channel alone."""

    values = np.asarray(channel)
    labels = np.asarray(target, dtype=np.int8)
    if values.ndim != 1 or labels.shape != values.shape:
        raise ValueError("channel and target must be aligned one-dimensional arrays")
    correct = 0
    for value in np.unique(values):
        selected = labels[values == value]
        correct += max(int(np.sum(selected == -1)), int(np.sum(selected == 1)))
    return correct / len(labels)


def _rule_channel_marginals(batch: SemanticBatch) -> dict[str, Any]:
    """Report lower-order label information exposed by each active rule input."""

    per_channel = {
        f"R_{index}": _single_channel_bayes_accuracy(
            np.asarray(batch.channels[f"R_{index}"]), np.asarray(batch.y)
        )
        for index in range(1, batch.active_k + 1)
    }
    values = tuple(per_channel.values())
    return {
        "per_channel_bayes_accuracy": per_channel,
        "best_single_channel_bayes_accuracy": float(max(values)),
        "mean_single_channel_bayes_accuracy": float(np.mean(values)),
        "n": len(batch),
    }


def _rule_counterfactual_effect(
    model: nn.Module,
    batch: SemanticBatch,
    config: Mapping[str, Any],
) -> tuple[dict[str, float], dict[str, Any]]:
    """Evaluate a decoder-level rule reversal and expose its input distance."""

    changed = flip_exact_rule_output(batch)
    base_logits = predict_logits(model, batch, config)
    changed_logits = predict_logits(model, changed, config)
    effect = intervention_metrics(base_logits, changed_logits, np.asarray(batch.y))
    diagnostics = dict(changed.metadata["exact_rule_counterfactual"])
    numeric_diagnostics = {
        name: float(value)
        for name, value in diagnostics.items()
        if name.startswith("hamming_") and isinstance(value, (int, float))
    }
    return {**effect, **numeric_diagnostics}, diagnostics


def run_h1(config: Mapping[str, Any], seed: int) -> ProtocolResult:
    """H1: train competing simple/exact signals and measure causal reliance."""

    seed_everything(seed)
    train, iid, conflict = _standard_data(config, seed)
    model, model_report = build_model(train, config, seed)
    condition = str(get_path(config, "experiment.mode", "primary"))
    fit = _fit(
        model,
        train,
        {"iid": iid, "conflict": conflict},
        config,
        seed,
        hypothesis="h1",
        condition=condition,
        intervention_splits=("conflict",),
    )
    final_iid = evaluate_batch(model, iid, config)
    final_conflict = evaluate_batch(model, conflict, config)
    final_interventions = evaluate_standard_interventions(model, conflict, config)
    summary = {
        "hypothesis": "h1",
        "seed": seed,
        "condition": condition,
        "final": {
            "iid": final_iid,
            "conflict": final_conflict,
            "interventions": final_interventions,
        },
        "model": model_report,
        "events": _standard_events(fit.trajectory, config),
        "data": {
            "n_train": len(train),
            "n_validation": len(iid),
            "n_eval": len(conflict),
            "realized_q": train.metadata["realized_q"],
            "n_conflict_train": train.metadata["n_conflict"],
            "k": train.active_k,
            "target_rule": train.metadata["target_rule"],
        },
        "training": {
            "steps": fit.training.optimizer_steps,
            "examples_seen": fit.training.samples_seen,
            "wall_seconds": fit.wall_seconds,
        },
    }
    predictions = _final_predictions(
        model, {"iid": iid, "conflict": conflict}, config, interventions=True
    )
    return ProtocolResult(
        model=model,
        summary=summary,
        metrics=fit.metrics,
        predictions=predictions,
        checkpoints=_checkpoint_states(fit.training),
        evaluation_batch=conflict,
    )


def run_h2(config: Mapping[str, Any], seed: int) -> ProtocolResult:
    """H2: calibrate each signal in isolation, then run signal competition."""

    seed_everything(seed)
    train, iid, conflict = _standard_data(config, seed)
    r_channels = tuple(f"R_{index}" for index in range(1, train.active_k + 1))

    def proxy_task(batch: SemanticBatch) -> SemanticBatch:
        # Accessibility asks D(P)->P.  D(P)->Y has Bayes ceiling q and would
        # mistake proxy imperfection for architecture-relative complexity.
        # All non-target signals are zeroed, but the competition interface is
        # retained exactly: active/inactive R slots and state coordinates remain
        # present, so parameter counts and optimization surfaces are comparable.
        calibrated = mask_channels(batch, (*r_channels, "P_present"))
        metadata = dict(calibrated.metadata)
        metadata.update(
            {
                "enforce_exact_code": False,
                "calibration_masked_channels": r_channels,
                "neutralize_padding": True,
                "calibration_interface": "competition_full_width",
                "calibration_target": "P",
            }
        )
        return calibrated.with_updates(
            target=np.asarray(batch.P, dtype=np.int8),
            reward=np.asarray(batch.P, dtype=np.float32),
            state=None if batch.state is None else np.zeros_like(batch.state),
            metadata=metadata,
        )

    def exact_task(batch: SemanticBatch) -> SemanticBatch:
        calibrated = remove_proxy(batch)
        metadata = dict(calibrated.metadata)
        metadata.update(
            {
                "neutralize_padding": True,
                "calibration_interface": "competition_full_width",
                "calibration_target": "Y",
            }
        )
        return calibrated.with_updates(
            state=None if batch.state is None else np.zeros_like(batch.state),
            metadata=metadata,
        )

    proxy_train, proxy_iid, proxy_conflict = map(proxy_task, (train, iid, conflict))
    exact_train, exact_iid, exact_conflict = map(exact_task, (train, iid, conflict))
    condition = str(get_path(config, "experiment.mode", "calibration_and_competition"))
    calibration_steps = _integer(
        config, "h2.calibration_steps", _integer(config, "train.steps", 1_000)
    )

    competition_names = train.feature_names(
        max_k=_integer(config, "data.max_k", train.active_k), include_state=True
    )
    for calibration_name, calibration_batch in (
        ("proxy", proxy_train),
        ("exact", exact_train),
    ):
        names = calibration_batch.feature_names(
            max_k=_integer(config, "data.max_k", train.active_k), include_state=True
        )
        if names != competition_names:
            raise RuntimeError(
                f"H2 {calibration_name} calibration changed the competition interface"
            )

    proxy_model, proxy_report = build_model(proxy_train, config, seed, include_state=True)
    proxy_calibration = _fit(
        proxy_model,
        proxy_train,
        {"proxy_calibration_iid": proxy_iid, "proxy_calibration_conflict": proxy_conflict},
        config,
        seed,
        hypothesis="h2",
        condition=condition,
        stage="proxy_calibration",
        include_state=True,
        steps=_integer(config, "h2.proxy_calibration_steps", calibration_steps),
    )

    exact_model, exact_report = build_model(exact_train, config, seed, include_state=True)
    exact_calibration = _fit(
        exact_model,
        exact_train,
        {"exact_calibration_iid": exact_iid, "exact_calibration_conflict": exact_conflict},
        config,
        seed,
        hypothesis="h2",
        condition=condition,
        stage="exact_calibration",
        include_state=True,
        steps=_integer(config, "h2.exact_calibration_steps", calibration_steps),
    )

    competition_model, competition_report = build_model(train, config, seed)
    for calibration_name, calibration_report in (
        ("proxy", proxy_report),
        ("exact", exact_report),
    ):
        if calibration_report["input_dim"] != competition_report["input_dim"]:
            raise RuntimeError(f"H2 {calibration_name} input dimension is not matched")
        if calibration_report["total_parameters"] != competition_report["total_parameters"]:
            raise RuntimeError(f"H2 {calibration_name} parameter count is not matched")
        if (
            calibration_report["trainable_parameters"]
            != competition_report["trainable_parameters"]
        ):
            raise RuntimeError(
                f"H2 {calibration_name} trainable parameter count is not matched"
            )
    competition = _fit(
        competition_model,
        train,
        {"competition_iid": iid, "competition_conflict": conflict},
        config,
        seed,
        hypothesis="h2",
        condition=condition,
        intervention_splits=("competition_conflict",),
        stage="competition",
        steps=_integer(config, "h2.competition_steps", _integer(config, "train.steps", 1_000)),
    )

    target_accuracy = _float(config, "h2.target_accuracy", 0.95)
    persistence = _integer(config, "h2.persistence", 2)
    proxy_event = _event(
        proxy_calibration.trajectory,
        "proxy_calibration_iid",
        "target_accuracy",
        target_accuracy,
        persistence,
    )
    exact_event = _event(
        exact_calibration.trajectory,
        "exact_calibration_iid",
        "target_accuracy",
        target_accuracy,
        persistence,
    )
    competition_events = _standard_events(
        [{**point, "conflict": point["competition_conflict"]} for point in competition.trajectory],
        config,
    )
    proxy_final = evaluate_batch(proxy_model, proxy_iid, config, include_state=True)
    exact_final = evaluate_batch(exact_model, exact_iid, config, include_state=True)
    conflict_final = evaluate_batch(competition_model, conflict, config)
    competition_interventions = evaluate_standard_interventions(
        competition_model, conflict, config
    )
    rule_counterfactual, counterfactual_diagnostics = _rule_counterfactual_effect(
        competition_model, conflict, config
    )
    competition_interventions["flip_exact_rule_output"] = rule_counterfactual
    competition.metrics.extend(
        make_metric_records(
            rule_counterfactual,
            hypothesis="h2",
            split="competition_conflict",
            global_step=competition.training.optimizer_steps,
            examples_seen=competition.training.samples_seen,
            intervention="flip_exact_rule_output",
            condition=condition,
            stage="competition",
        )
    )
    rule_channel_marginals = {
        "train": _rule_channel_marginals(train),
        "iid": _rule_channel_marginals(iid),
        "conflict": _rule_channel_marginals(conflict),
    }
    summary = {
        "hypothesis": "h2",
        "seed": seed,
        "condition": condition,
        "final": {
            # Direct scalar columns make architecture thresholds/phase diagrams easy.
            "decoder_accuracy": exact_final["target_accuracy"],
            "exact_decoder_accuracy": exact_final["target_accuracy"],
            "proxy_decoder_accuracy": proxy_final["target_accuracy"],
            "rho_y": conflict_final["rho_y"],
            "rho_p": conflict_final["rho_p"],
            "proxy_calibration_iid": proxy_final,
            "proxy_calibration_conflict": evaluate_batch(
                proxy_model, proxy_conflict, config, include_state=True
            ),
            "exact_calibration_iid": exact_final,
            "exact_calibration_conflict": evaluate_batch(
                exact_model, exact_conflict, config, include_state=True
            ),
            "competition_iid": evaluate_batch(competition_model, iid, config),
            "competition_conflict": conflict_final,
            "competition_interventions": competition_interventions,
            "decoder_threshold_reached": bool(exact_final["target_accuracy"] >= target_accuracy),
        },
        "model": {
            "proxy_calibration": proxy_report,
            "exact_calibration": exact_report,
            "competition": competition_report,
        },
        "events": {
            **proxy_event.as_dict("proxy_decoder_acquisition"),
            **exact_event.as_dict("exact_decoder_acquisition"),
            **competition_events,
        },
        "data": {
            "k": train.active_k,
            "target_rule": train.metadata["target_rule"],
            "realized_q": train.metadata["realized_q"],
            "rule_channel_marginal_predictiveness": rule_channel_marginals,
            "best_single_rule_channel_bayes_accuracy": rule_channel_marginals["train"][
                "best_single_channel_bayes_accuracy"
            ],
            "exact_rule_counterfactual": counterfactual_diagnostics,
            "proxy_calibration_informative_channels": ("P",),
            "exact_calibration_informative_channels": r_channels,
            "proxy_calibration_target": "P",
            "exact_calibration_target": "Y",
            "calibration_includes_state": True,
            "calibration_state_is_neutralized": True,
            "calibration_padding_is_neutralized": True,
            "calibration_interface": "competition_full_width",
            "calibration_input_dim_matches_competition": bool(
                proxy_report["input_dim"]
                == exact_report["input_dim"]
                == competition_report["input_dim"]
            ),
            "calibration_parameter_count_matches_competition": bool(
                proxy_report["total_parameters"]
                == exact_report["total_parameters"]
                == competition_report["total_parameters"]
            ),
            "calibration_trainable_count_matches_competition": bool(
                proxy_report["trainable_parameters"]
                == exact_report["trainable_parameters"]
                == competition_report["trainable_parameters"]
            ),
            "proxy_accessibility_estimand": "D(P)->P",
            "proxy_accessibility_rationale": (
                "D(P)->Y is capped by proxy fidelity q; D(P)->P isolates "
                "architecture-relative accessibility while competition still targets Y"
            ),
            "target_accuracy": target_accuracy,
        },
        "training": {
            "proxy_calibration_steps": proxy_calibration.training.optimizer_steps,
            "exact_calibration_steps": exact_calibration.training.optimizer_steps,
            "competition_steps": competition.training.optimizer_steps,
            "examples_seen": (
                proxy_calibration.training.samples_seen
                + exact_calibration.training.samples_seen
                + competition.training.samples_seen
            ),
            "wall_seconds": (
                proxy_calibration.wall_seconds
                + exact_calibration.wall_seconds
                + competition.wall_seconds
            ),
        },
    }
    predictions = _final_predictions(
        proxy_model,
        {"proxy_calibration_iid": proxy_iid},
        config,
        stage="proxy_calibration",
        include_state=True,
    )
    predictions.extend(
        _final_predictions(
            exact_model,
            {"exact_calibration_iid": exact_iid},
            config,
            stage="exact_calibration",
            include_state=True,
        )
    )
    predictions.extend(
        _final_predictions(
            competition_model,
            {"competition_conflict": conflict},
            config,
            interventions=True,
            stage="competition",
        )
    )
    checkpoints = _checkpoint_states(proxy_calibration.training, "proxy_calibration_")
    checkpoints.update(_checkpoint_states(exact_calibration.training, "exact_calibration_"))
    checkpoints.update(_checkpoint_states(competition.training, "competition_"))
    return ProtocolResult(
        model=competition_model,
        summary=summary,
        metrics=proxy_calibration.metrics + exact_calibration.metrics + competition.metrics,
        predictions=predictions,
        checkpoints=checkpoints,
        evaluation_batch=conflict,
    )


def _plateau_event(
    trajectory: Sequence[dict[str, Any]],
    tolerance: float,
    window: int,
    minimum_accuracy: float,
) -> dict[str, int | bool | float]:
    if window < 2:
        raise ValueError("h3.reward_plateau_window must be at least two")
    if not 0.0 <= minimum_accuracy <= 1.0:
        raise ValueError("h3 reward plateau minimum accuracy must lie in [0,1]")
    terminal_accuracy = float(trajectory[-1]["iid"]["target_accuracy"])
    for stop in range(window, len(trajectory) + 1):
        segment = [float(point["iid"]["target_accuracy"]) for point in trajectory[stop - window : stop]]
        if (
            min(segment) >= minimum_accuracy
            and max(segment) - min(segment) <= tolerance
            and abs(float(np.mean(segment)) - terminal_accuracy) <= tolerance
        ):
            step = int(trajectory[stop - window]["step"])
            return {
                "reward_plateau_time": step,
                "reward_plateau_observed": True,
                "reward_plateau_accuracy": float(np.mean(segment)),
                "reward_plateau_minimum_accuracy": minimum_accuracy,
                "reward_plateau_terminal_accuracy": terminal_accuracy,
            }
    return {
        "reward_plateau_time": int(trajectory[-1]["step"]),
        "reward_plateau_observed": False,
        "reward_plateau_accuracy": float(trajectory[-1]["iid"]["target_accuracy"]),
        "reward_plateau_minimum_accuracy": minimum_accuracy,
        "reward_plateau_terminal_accuracy": terminal_accuracy,
    }


def run_h3(config: Mapping[str, Any], seed: int) -> ProtocolResult:
    """H3: retain the full log-spaced reliance trajectory and censored events."""

    seed_everything(seed)
    train, iid, conflict = _standard_data(config, seed)
    model, model_report = build_model(train, config, seed)
    condition = str(get_path(config, "experiment.mode", "dense_checkpointing"))
    fit = _fit(
        model,
        train,
        {"iid": iid, "conflict": conflict},
        config,
        seed,
        hypothesis="h3",
        condition=condition,
        intervention_splits=("conflict",),
    )
    events = _standard_events(fit.trajectory, config)
    requested_plateau_minimum = get_path(config, "h3.reward_plateau_min_accuracy")
    plateau_minimum = (
        max(0.55, float(train.metadata["realized_q"]) - 0.02)
        if requested_plateau_minimum is None
        else float(requested_plateau_minimum)
    )
    events.update(
        _plateau_event(
            fit.trajectory,
            _float(config, "h3.reward_plateau_tolerance", 0.005),
            _integer(config, "h3.reward_plateau_window", 4),
            plateau_minimum,
        )
    )
    summary = {
        "hypothesis": "h3",
        "seed": seed,
        "condition": condition,
        "final": {
            "iid": evaluate_batch(model, iid, config),
            "conflict": evaluate_batch(model, conflict, config),
            "interventions": evaluate_standard_interventions(model, conflict, config),
            "trajectory_points": len(fit.trajectory),
        },
        "model": model_report,
        "events": events,
        "data": {
            "n_train": len(train),
            "realized_q": train.metadata["realized_q"],
            "n_conflict_train": train.metadata["n_conflict"],
            "k": train.active_k,
            "target_rule": train.metadata["target_rule"],
        },
        "training": {
            "steps": fit.training.optimizer_steps,
            "examples_seen": fit.training.samples_seen,
            "wall_seconds": fit.wall_seconds,
        },
    }
    return ProtocolResult(
        model=model,
        summary=summary,
        metrics=fit.metrics,
        predictions=_final_predictions(model, {"iid": iid, "conflict": conflict}, config),
        checkpoints=_checkpoint_states(fit.training),
        evaluation_batch=conflict,
    )


def _unique_conflicts(batch: SemanticBatch) -> SemanticBatch | None:
    conflict = np.flatnonzero(np.asarray(batch.latents["is_conflict"], dtype=bool))
    if not len(conflict):
        return None
    ids = np.asarray(batch.state_id)[conflict]
    _, first = np.unique(ids, return_index=True)
    return batch.select(conflict[np.sort(first)])


def _conflict_diversity(batch: SemanticBatch) -> dict[str, float | int]:
    conflict = np.flatnonzero(np.asarray(batch.latents["is_conflict"], dtype=bool))
    if not len(conflict):
        return {
            "n_conflict": 0,
            "u_conflict": 0,
            "effective_unique_contexts": 0.0,
            "minimum_repetitions": 0,
            "maximum_repetitions": 0,
        }
    _, counts = np.unique(np.asarray(batch.state_id)[conflict], return_counts=True)
    probabilities = counts / counts.sum()
    entropy = -float(np.sum(probabilities * np.log(probabilities)))
    return {
        "n_conflict": int(len(conflict)),
        "u_conflict": int(len(counts)),
        "effective_unique_contexts": float(math.exp(entropy)),
        "minimum_repetitions": int(counts.min()),
        "maximum_repetitions": int(counts.max()),
    }


def run_h4(config: Mapping[str, Any], seed: int) -> ProtocolResult:
    """H4: hold conflict count fixed while manipulating context diversity."""

    seed_everything(seed)
    n_train = _integer(config, "data.n_train", 4096)
    n_conflict = _integer(
        config,
        "h4.n_conflict",
        round(n_train * (1 - _float(config, "data.q", 0.9))),
    )
    explicit_unique = get_path(config, "h4.u_conflict")
    configured_fraction = get_path(config, "h4.unique_fraction")
    requested_fraction: float | None = None
    fraction_implied_unique: int | None = None
    condition = str(get_path(config, "h4.condition", "diverse"))
    configured_train_types = get_path(
        config,
        "h4.structured_train_types",
        ["location_reflection", "coordinate_exchange"],
    )
    failure_balanced_minimum = (
        2 * len(configured_train_types)
        if condition == "structured_holdout" and n_conflict > 0
        else 2
    )
    if (
        condition == "structured_holdout"
        and 0 < n_conflict < failure_balanced_minimum
    ):
        raise ValueError(
            "structured_holdout needs at least two conflict prototypes per "
            f"failure mechanism; need n_conflict >= {failure_balanced_minimum}"
        )
    minimum_unique = min(failure_balanced_minimum, n_conflict)
    if explicit_unique is None:
        fraction = _float(config, "h4.unique_fraction", 1.0)
        if not 0.0 <= fraction <= 1.0:
            raise ValueError("h4.unique_fraction must lie in [0,1]")
        requested_fraction = fraction
        fraction_implied_unique = round(fraction * n_conflict)
        # Target-stratify contexts so the repeated-state treatment never assigns
        # contradictory labels to one context. A balanced conflict set therefore
        # needs at least one prototype per sign.
        u_conflict = (
            0
            if n_conflict == 0
            else max(
                minimum_unique,
                min(n_conflict, fraction_implied_unique),
            )
        )
    else:
        if (
            isinstance(explicit_unique, (bool, np.bool_))
            or int(explicit_unique) != explicit_unique
        ):
            raise ValueError("h4.u_conflict must be an integer")
        u_conflict = int(explicit_unique)
    bundle: DatasetBundle = make_h4_dataset(
        n_train,
        n_conflict,
        u_conflict,
        _integer(config, "data.k", 3),
        seed,
        condition=condition,  # type: ignore[arg-type]
        q=get_path(config, "h4.q"),
        n_eval=_integer(config, "data.n_eval", 4096),
        structured_train_types=get_path(
            config,
            "h4.structured_train_types",
            ["location_reflection", "coordinate_exchange"],
        ),
        structured_test_types=get_path(
            config,
            "h4.structured_test_types",
            ["geometry_rotation", "nuisance_inversion"],
        ),
        **_dataset_arguments(config),
    )
    train, unseen = bundle
    unique_seen = _unique_conflicts(train)
    conflict_positions = np.flatnonzero(np.asarray(train.latents["is_conflict"], dtype=bool))
    seen = train.select(conflict_positions) if len(conflict_positions) else None
    evaluation: dict[str, SemanticBatch] = {"conflict_unseen": unseen}
    if seen is not None:
        evaluation["conflict_seen_repeated"] = seen
    if unique_seen is not None:
        evaluation["conflict_seen_unique"] = unique_seen

    model, model_report = build_model(train, config, seed)
    fit = _fit(
        model,
        train,
        evaluation,
        config,
        seed,
        hypothesis="h4",
        condition=condition,
        intervention_splits=("conflict_unseen",),
    )
    adapted_trajectory = [
        {**point, "conflict": point["conflict_unseen"]} for point in fit.trajectory
    ]
    diversity = _conflict_diversity(train)
    final: dict[str, Any] = {
        split: evaluate_batch(model, batch, config) for split, batch in evaluation.items()
    }
    final["unseen_interventions"] = evaluate_standard_interventions(model, unseen, config)
    final["seen_unique_diagnostics"] = diversity
    final["train_eval_ids_disjoint"] = bool(
        set(np.asarray(train.state_id)).isdisjoint(set(np.asarray(unseen.state_id)))
    )
    summary = {
        "hypothesis": "h4",
        "seed": seed,
        "condition": condition,
        "final": final,
        "model": model_report,
        "events": _standard_events(adapted_trajectory, config),
        "data": {
            **diversity,
            "n_train": len(train),
            "n_eval": len(unseen),
            "requested_n_conflict": n_conflict,
            "requested_u_conflict": None if explicit_unique is None else int(explicit_unique),
            "requested_unique_fraction": requested_fraction,
            "configured_unique_fraction": configured_fraction,
            "u_conflict_source": "fraction" if explicit_unique is None else "explicit",
            "fraction_implied_u_conflict": fraction_implied_unique,
            "minimum_target_stratified_u_conflict": minimum_unique,
            "minimum_failure_balanced_u_conflict": (
                failure_balanced_minimum
                if condition == "structured_holdout" and n_conflict > 0
                else None
            ),
            "realized_u_conflict": diversity["u_conflict"],
            "realized_q": train.metadata["realized_q"],
            "k": train.active_k,
            "structured_holdout": condition == "structured_holdout",
            "structured_train_types": list(train.metadata["structured_types"]),
            "structured_test_types": list(unseen.metadata["structured_types"]),
            "structured_train_type_ids": list(train.metadata["structured_type_ids"]),
            "structured_test_type_ids": list(unseen.metadata["structured_type_ids"]),
            "structured_types_are_disjoint": bool(
                set(train.metadata["structured_types"])
                .isdisjoint(set(unseen.metadata["structured_types"]))
            ),
            "failure_mechanism_visible_to_model": train.metadata[
                "failure_mechanism_visible_to_model"
            ],
            "failure_mechanism_semantics": train.metadata[
                "failure_mechanism_semantics"
            ],
            "context_hash_family": train.metadata["context_hash_family"],
        },
        "training": {
            "steps": fit.training.optimizer_steps,
            "examples_seen": fit.training.samples_seen,
            "wall_seconds": fit.wall_seconds,
        },
    }
    prediction_batches = {"conflict_unseen": unseen}
    if unique_seen is not None:
        prediction_batches["conflict_seen_unique"] = unique_seen
    return ProtocolResult(
        model=model,
        summary=summary,
        metrics=fit.metrics,
        predictions=_final_predictions(model, prediction_batches, config),
        checkpoints=_checkpoint_states(fit.training),
        evaluation_batch=unseen,
    )


RUNNERS: dict[str, Callable[[Mapping[str, Any], int], ProtocolResult]] = {
    "h1": run_h1,
    "h2": run_h2,
    "h3": run_h3,
    "h4": run_h4,
}


__all__ = ["RUNNERS", "run_h1", "run_h2", "run_h3", "run_h4"]
