"""Persistence, hysteresis, and conditional-multiplicity protocols (H7--H9).

These experiments are deliberately implemented as phased protocols rather than
as thin wrappers around a single fit.  In particular, evaluation clocks are
stage-local, example counts remain explicit, optimizer resets are controlled at
phase boundaries, and failed mastery gates are represented in the result rather
than silently discarded.
"""

from __future__ import annotations

import copy
import math
import time
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import numpy as np
import torch
from torch import nn

from .config import get_path
from .data import (
    SemanticBatch,
    balanced_signs,
    flip_channels,
    make_conflict_dataset,
    make_h9_dataset,
    make_standard_dataset,
    mask_channels,
    remove_proxy,
)
from .metrics import (
    TimeToEvent,
    acquisition_time,
    binary_predictions,
    context_selector_metrics,
    intervention_metrics,
    reactivation_time,
    reliance,
    reliance_half_life,
    softmax,
    trapezoid_auc,
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
from .training import (
    SFTConfig,
    StopTraining,
    TrainingResult,
    TrainingSnapshot,
    log_spaced_steps,
    train_clean_sft,
)


MetricRow = dict[str, Any]


def _integer(config: Mapping[str, Any], path: str, default: int) -> int:
    value = get_path(config, path, default)
    if isinstance(value, bool):
        raise ValueError(f"{path} must be an integer")
    result = int(value)
    if result != value:
        raise ValueError(f"{path} must be an integer")
    return result


def _float(config: Mapping[str, Any], path: str, default: float) -> float:
    value = float(get_path(config, path, default))
    if not math.isfinite(value):
        raise ValueError(f"{path} must be finite")
    return value


def _device(config: Mapping[str, Any]) -> torch.device:
    return resolve_device(str(get_path(config, "run.device", "auto")))


def _dataset_arguments(config: Mapping[str, Any]) -> dict[str, int]:
    return {
        "max_k": _integer(config, "data.max_k", 5),
        "state_dim": _integer(config, "data.state_dim", 0),
    }


def _cpu_state(model: nn.Module) -> dict[str, Any]:
    """Clone a model state without retaining accelerator storage or autograd."""

    result: dict[str, Any] = {}
    for name, value in model.state_dict().items():
        result[name] = value.detach().cpu().clone() if torch.is_tensor(value) else copy.deepcopy(value)
    return result


def _event_payload(event: TimeToEvent) -> dict[str, int | bool]:
    return {
        "time": int(event.time),
        "observed": bool(event.observed),
        "censored": not bool(event.observed),
        "horizon": int(event.horizon),
    }


def _checkpoint_interval_payload(
    event: TimeToEvent, observed_times: Sequence[int]
) -> dict[str, int | bool | None]:
    """Expose the interval resolution induced by checkpoint-only observation."""

    times = [int(value) for value in observed_times]
    if not times or sorted(times) != times or int(event.time) not in times:
        raise ValueError("event time must occur in a sorted checkpoint sequence")
    if not event.observed:
        return {
            "lower_exclusive": int(event.horizon),
            "upper_inclusive": None,
            "interval_censored": False,
            "right_censored": True,
        }
    index = times.index(int(event.time))
    lower = times[index - 1] if index else times[index]
    return {
        "lower_exclusive": int(lower),
        "upper_inclusive": int(event.time),
        "interval_censored": bool(index > 0 and lower < int(event.time)),
        "right_censored": False,
    }


def _training_optimizer(model: nn.Module, config: Mapping[str, Any]) -> torch.optim.Optimizer:
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not parameters:
        raise ValueError("model has no trainable parameters")
    name = str(get_path(config, "train.optimizer", "adamw")).lower()
    learning_rate = _float(config, "train.learning_rate", 3e-3)
    weight_decay = _float(config, "train.weight_decay", 0.0)
    if name == "adamw":
        return torch.optim.AdamW(parameters, lr=learning_rate, weight_decay=weight_decay)
    if name == "adam":
        return torch.optim.Adam(parameters, lr=learning_rate, weight_decay=weight_decay)
    if name == "sgd":
        return torch.optim.SGD(
            parameters,
            lr=learning_rate,
            weight_decay=weight_decay,
            momentum=_float(config, "train.momentum", 0.0),
        )
    raise ValueError("train.optimizer must be adamw, adam, or sgd")


def _sft_config(
    config: Mapping[str, Any],
    seed: int,
    *,
    steps: int,
    reset_optimizer: bool,
    log_steps: Sequence[int] | None = None,
    batch_size: int | None = None,
    save_final_checkpoint: bool = True,
) -> SFTConfig:
    if steps < 1:
        raise ValueError("training stages must contain at least one optimizer step")
    checkpoint_steps: Sequence[int] | None = (steps,) if save_final_checkpoint else None
    return SFTConfig(
        steps=int(steps),
        batch_size=(
            _integer(config, "train.batch_size", 128) if batch_size is None else int(batch_size)
        ),
        learning_rate=_float(config, "train.learning_rate", 3e-3),
        weight_decay=_float(config, "train.weight_decay", 0.0),
        optimizer=str(get_path(config, "train.optimizer", "adamw")),  # type: ignore[arg-type]
        momentum=_float(config, "train.momentum", 0.0),
        seed=int(seed),
        device=_device(config),
        deterministic=bool(get_path(config, "train.deterministic", True)),
        shuffle=bool(get_path(config, "train.shuffle", True)),
        gradient_clip_norm=get_path(config, "train.grad_clip", 1.0),
        log_steps=log_steps,
        num_log_points=_integer(config, "train.num_log_points", 25),
        checkpoint_steps=checkpoint_steps,
        save_checkpoints=bool(save_final_checkpoint),
        reset_model=False,
        reset_optimizer=bool(reset_optimizer),
    )


def _interval_steps(total: int, interval: int, *, include_log_points: bool = False) -> tuple[int, ...]:
    if total < 1 or interval < 1:
        raise ValueError("total and interval must be positive")
    values = set(range(interval, total + 1, interval))
    values.add(total)
    if include_log_points:
        values.update(log_spaced_steps(total))
    return tuple(sorted(values))


def _checkpoint_states(training: TrainingResult, prefix: str) -> dict[str, dict[str, Any]]:
    checkpoints = {
        f"{prefix}step_{step}": checkpoint.model_state
        for step, checkpoint in sorted(training.checkpoints.items())
    }
    checkpoints[f"{prefix}final"] = training.final_model_state
    return checkpoints


def _optimization_records(
    snapshot: TrainingSnapshot,
    *,
    hypothesis: str,
    stage: str,
    condition: str,
    global_step: int,
    global_examples: int,
    n: int,
) -> list[MetricRow]:
    return make_metric_records(
        {
            "loss": snapshot.record.loss,
            "primary_loss": snapshot.record.primary_loss,
            "train_batch_accuracy": snapshot.record.metrics.get("train_accuracy", float("nan")),
            "n": n,
        },
        hypothesis=hypothesis,
        split="train_minibatch",
        global_step=global_step,
        stage=stage,
        stage_step=snapshot.step,
        examples_seen=global_examples,
        condition=condition,
    )


def _finite_proxy_probability(n: int, q: float, *, smoke_tolerant: bool) -> tuple[float, bool]:
    if not 0.0 <= q <= 1.0:
        raise ValueError("proxy accuracy must lie in [0,1]")
    conflicts = n * (1.0 - q)
    rounded = round(conflicts)
    if abs(conflicts - rounded) <= 1e-9:
        return q, False
    if not smoke_tolerant:
        raise ValueError(f"q={q:g} is not exactly realizable with n={n}")
    return 1.0 - rounded / n, True


def _proxy_for_goal(y: np.ndarray, q: float, seed: int) -> np.ndarray:
    """Make an exact-q proxy, stratifying disagreements across target signs."""

    target = np.asarray(y, dtype=np.int8)
    n = len(target)
    count = int(round(n * (1.0 - q)))
    rng = np.random.default_rng(seed)
    negative = np.flatnonzero(target == -1)
    positive = np.flatnonzero(target == 1)
    rng.shuffle(negative)
    rng.shuffle(positive)
    negative_count = count // 2
    positive_count = count - negative_count
    if negative_count > len(negative) or positive_count > len(positive):
        negative_count = min(len(negative), count - min(len(positive), count))
        positive_count = count - negative_count
    conflict = np.concatenate((negative[:negative_count], positive[:positive_count]))
    proxy = target.copy()
    proxy[conflict] *= -1
    return proxy.astype(np.int8, copy=False)


def _attach_proxy(
    batch: SemanticBatch,
    name: str,
    q: float,
    seed: int,
    *,
    present: bool = True,
) -> SemanticBatch:
    channels = {key: np.array(value, copy=True) for key, value in batch.channels.items()}
    channels[name] = _proxy_for_goal(np.asarray(batch.y), q, seed)
    channels[f"{name}_present"] = np.full(len(batch), int(present), dtype=np.int8)
    if not present:
        channels[name].fill(0)
    metadata = dict(batch.metadata)
    metadata[f"{name}_q"] = float(q)
    return batch.with_updates(channels=channels, metadata=metadata)


def _with_target(batch: SemanticBatch, target: np.ndarray, *, change_semantic_y: bool = False) -> SemanticBatch:
    values = np.asarray(target, dtype=np.int8)
    if values.shape != np.asarray(batch.y).shape or not np.all(np.isin(values, (-1, 1))):
        raise ValueError("stage target must be a signed vector aligned with the batch")
    changes: dict[str, Any] = {"target": values, "reward": values.astype(np.float32)}
    if change_semantic_y:
        changes["y"] = values
    return batch.with_updates(**changes)


def _tag_predictions(records: list[dict[str, Any]], **tags: Any) -> list[dict[str, Any]]:
    for record in records:
        record.update(tags)
    return records


# ---------------------------------------------------------------------------
# H7: counterevidence and dormant proxy restoration


class _PhaseAGateReached(RuntimeError):
    """Private non-error control flow used to stop exactly at a mastery gate."""


def _h7_data(
    config: Mapping[str, Any], seed: int
) -> tuple[SemanticBatch, SemanticBatch, SemanticBatch, SemanticBatch, dict[str, Any]]:
    n_train = _integer(config, "data.n_train", 4096)
    n_eval = _integer(config, "data.n_eval", 4096)
    k = _integer(config, "data.k", 3)
    max_steps = _integer(config, "h7.phase_a_max_steps", _integer(config, "train.steps", 1000))
    smoke_scale = n_train <= 128 or max_steps <= 64
    requested_q_a = _float(config, "h7.q_a", 0.99)
    if requested_q_a <= 0.5:
        raise ValueError("h7.q_a must exceed 0.5 to install a positively predictive proxy")
    q_a, adjusted_a = _finite_proxy_probability(n_train, requested_q_a, smoke_tolerant=smoke_scale)
    condition = str(get_path(config, "h7.condition", "reversal")).lower()
    if condition not in {"removal", "decorrelation", "reversal", "replacement"}:
        raise ValueError("h7.condition must be removal, decorrelation, reversal, or replacement")
    requested_q_b = _float(config, "h7.q_b", 0.1)
    if condition == "reversal" and requested_q_b >= 0.5:
        raise ValueError("the H7 reversal arm requires h7.q_b < 0.5")
    if condition == "replacement" and requested_q_b > 0.5:
        raise ValueError("the old proxy in the H7 replacement arm requires h7.q_b <= 0.5")
    old_q_b = q_a if condition == "removal" else 0.5 if condition == "decorrelation" else requested_q_b
    q_b, adjusted_b = _finite_proxy_probability(n_train, old_q_b, smoke_tolerant=smoke_scale)
    requested_q_new = _float(config, "h7.q_new", 0.99)
    if condition == "replacement" and requested_q_new <= 0.5:
        raise ValueError("the new proxy in the H7 replacement arm requires h7.q_new > 0.5")
    q_new, adjusted_new = _finite_proxy_probability(
        n_train,
        requested_q_new if condition == "replacement" else 0.5,
        smoke_tolerant=smoke_scale,
    )
    arguments = _dataset_arguments(config)
    phase_a = make_standard_dataset(n_train, q_a, k, seed, split="h7_phase_a", **arguments)
    phase_a = _attach_proxy(phase_a, "P_new", 0.5, seed + 101)
    phase_b = make_standard_dataset(n_train, q_b, k, seed + 1, split="h7_phase_b", **arguments)
    phase_b = _attach_proxy(phase_b, "P_new", q_new, seed + 102)
    if condition == "removal":
        phase_b = remove_proxy(phase_b)

    phase_b_eval = make_standard_dataset(
        n_eval,
        q_b,
        k,
        seed + 2,
        split="h7_phase_b_eval",
        id_offset=2_000_000,
        **arguments,
    )
    phase_b_eval = _attach_proxy(phase_b_eval, "P_new", q_new, seed + 103)
    if condition == "removal":
        phase_b_eval = remove_proxy(phase_b_eval)

    restoration = make_conflict_dataset(
        n_eval,
        k,
        seed + 3,
        id_offset=3_000_000,
        **arguments,
    )
    restoration = _attach_proxy(restoration, "P_new", 0.5, seed + 104, present=False)
    details = {
        "condition": condition,
        "smoke_scale": smoke_scale,
        "requested_q_a": requested_q_a,
        "realized_q_a": q_a,
        "requested_q_b": requested_q_b,
        "realized_q_b": q_b,
        "requested_q_new": requested_q_new,
        "realized_q_new": q_new,
        "finite_sample_adjustment": bool(adjusted_a or adjusted_b or adjusted_new),
    }
    return phase_a, phase_b, phase_b_eval, restoration, details


def _h7_stage_snapshot(
    model: nn.Module,
    batch: SemanticBatch,
    config: Mapping[str, Any],
    *,
    metrics: list[MetricRow],
    stage: str,
    condition: str,
    stage_step: int,
    global_step: int,
    examples_seen: int,
    split: str,
) -> dict[str, Any]:
    values = evaluate_batch(model, batch, config)
    metrics.extend(
        make_metric_records(
            values,
            hypothesis="h7",
            split=split,
            global_step=global_step,
            stage=stage,
            stage_step=stage_step,
            examples_seen=examples_seen,
            condition=condition,
        )
    )
    return {
        "stage_step": int(stage_step),
        "global_step": int(global_step),
        "examples_seen": int(examples_seen),
        **values,
    }


def _named_proxy_reliance(
    model: nn.Module,
    batch: SemanticBatch,
    config: Mapping[str, Any],
    name: str,
) -> float:
    prediction = binary_predictions(predict_logits(model, batch, config))
    return reliance(prediction, np.asarray(batch.channels[name], dtype=np.int8))


def _named_channel_effect(
    model: nn.Module,
    batch: SemanticBatch,
    config: Mapping[str, Any],
    name: str,
) -> dict[str, float]:
    base = predict_logits(model, batch, config)
    changed = predict_logits(model, flip_channels(batch, name), config)
    return intervention_metrics(base, changed, np.asarray(batch.y, dtype=np.int8))


def _h7_gated_result(
    model: nn.Module,
    config: Mapping[str, Any],
    seed: int,
    model_report: Mapping[str, Any],
    restoration: SemanticBatch,
    details: Mapping[str, Any],
    phase_a_trajectory: Sequence[Mapping[str, Any]],
    metrics: list[MetricRow],
    checkpoints: dict[str, dict[str, Any]],
    phase_a_examples: int,
) -> ProtocolResult:
    final = evaluate_batch(model, restoration, config)
    horizon = int(phase_a_trajectory[-1]["stage_step"])
    censored = _event_payload(TimeToEvent(horizon, False, horizon))
    summary = {
        "hypothesis": "h7",
        "seed": seed,
        "condition": details["condition"],
        "final": {
            "restoration_probe": final,
            "phase_a_gate_passed": False,
            "eligible_for_primary_analysis": False,
            "phase_b_executed": False,
        },
        "model": dict(model_report),
        "events": {
            "phase_a_proxy_acquisition": censored,
            "half_life": censored,
            "literal_half_life": censored,
            "adjusted_half_life": censored,
        },
        "data": dict(details),
        "training": {
            "phase_a_steps": horizon,
            "phase_a_examples": int(phase_a_examples),
            "phase_b_steps": 0,
            "optimizer_reset_at_phase_b": bool(get_path(config, "h7.reset_optimizer", True)),
        },
    }
    predictions = []
    if bool(get_path(config, "evaluation.save_predictions", True)):
        predictions = _tag_predictions(
            prediction_records(model, restoration, config, split="restoration_probe"),
            stage="phase_a_gate",
        )
    return ProtocolResult(
        model=model,
        summary=summary,
        metrics=metrics,
        predictions=predictions,
        checkpoints=checkpoints,
        evaluation_batch=restoration,
    )


def run_h7(config: Mapping[str, Any], seed: int) -> ProtocolResult:
    """Install an old proxy, apply one unlearning arm, and probe restoration."""

    seed_everything(seed)
    phase_a, phase_b, phase_b_eval, restoration, details = _h7_data(config, seed)
    condition = str(details["condition"])
    model, model_report = build_model(phase_a, config, seed)
    model.to(_device(config))
    metrics: list[MetricRow] = []
    checkpoints: dict[str, dict[str, Any]] = {}
    phase_a_trajectory: list[dict[str, Any]] = []
    max_steps = _integer(config, "h7.phase_a_max_steps", _integer(config, "train.steps", 1000))
    eval_every = _integer(config, "h7.eval_every", 1)
    threshold = _float(config, "h7.transition_threshold", 0.9)
    patience = _integer(config, "h7.transition_patience", 3)
    if max_steps < 1 or eval_every < 1 or patience < 1:
        raise ValueError("H7 phase lengths, evaluation interval, and patience must be positive")
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("h7.transition_threshold must lie in [0,1]")
    phase_a_log_steps = _interval_steps(max_steps, eval_every)
    initial_phase_a = _h7_stage_snapshot(
        model,
        restoration,
        config,
        metrics=metrics,
        stage="phase_a_install",
        condition=condition,
        stage_step=0,
        global_step=0,
        examples_seen=0,
        split="restoration_probe",
    )
    phase_a_trajectory.append(initial_phase_a)
    optimizer = _training_optimizer(model, config)
    consecutive = 1 if float(initial_phase_a["rho_p"]) >= threshold else 0
    transition_step: int | None = 0 if consecutive >= patience else None
    transition_examples = 0
    if transition_step == 0:
        checkpoints["phase_a_transition"] = _cpu_state(model)
    phase_a_wall_start = time.perf_counter()

    def phase_a_callback(snapshot: TrainingSnapshot) -> None:
        nonlocal consecutive, transition_step, transition_examples
        point = _h7_stage_snapshot(
            snapshot.model,
            restoration,
            config,
            metrics=metrics,
            stage="phase_a_install",
            condition=condition,
            stage_step=snapshot.step,
            global_step=snapshot.step,
            examples_seen=snapshot.record.samples_seen,
            split="restoration_probe",
        )
        phase_a_trajectory.append(point)
        metrics.extend(
            _optimization_records(
                snapshot,
                hypothesis="h7",
                stage="phase_a_install",
                condition=condition,
                global_step=snapshot.step,
                global_examples=snapshot.record.samples_seen,
                n=len(phase_a),
            )
        )
        consecutive = consecutive + 1 if float(point["rho_p"]) >= threshold else 0
        if consecutive >= patience:
            transition_step = int(snapshot.step)
            transition_examples = int(snapshot.record.samples_seen)
            checkpoints["phase_a_transition"] = _cpu_state(snapshot.model)
            raise _PhaseAGateReached

    phase_a_training: TrainingResult | None = None
    if transition_step is None:
        try:
            phase_a_training = train_clean_sft(
                model,
                batch_for_training(phase_a, config),
                _sft_config(
                    config,
                    seed + 10,
                    steps=max_steps,
                    reset_optimizer=True,
                    log_steps=phase_a_log_steps,
                    save_final_checkpoint=False,
                ),
                optimizer=optimizer,
                callback=phase_a_callback,
            )
        except _PhaseAGateReached:
            # This is an intentional exact early stop; model and optimizer are at
            # the checkpoint whose consecutive evaluation completed the gate.
            pass
    phase_a_wall = time.perf_counter() - phase_a_wall_start
    if transition_step is None:
        transition_step = max_steps
        transition_examples = (
            int(phase_a_training.samples_seen)
            if phase_a_training is not None
            else int(phase_a_trajectory[-1]["examples_seen"])
        )
        checkpoints["phase_a_final_unmastered"] = _cpu_state(model)
    gate_passed = consecutive >= patience
    smoke_bypass = bool(
        not gate_passed
        and get_path(config, "h7.allow_unmastered_phase_b", bool(details["smoke_scale"]))
    )
    if not gate_passed and not smoke_bypass:
        return _h7_gated_result(
            model,
            config,
            seed,
            model_report,
            restoration,
            details,
            phase_a_trajectory,
            metrics,
            checkpoints,
            transition_examples,
        )

    phase_b_steps = _integer(config, "h7.phase_b_steps", 1024)
    if phase_b_steps < 1:
        raise ValueError("h7.phase_b_steps must be positive")
    restore_every = _integer(config, "h7.restore_probe_every", eval_every)
    if restore_every < 1:
        raise ValueError("h7.restore_probe_every must be positive")
    restoration_enabled = bool(get_path(config, "h7.restore_probe", True))
    evaluation_steps = set(_interval_steps(phase_b_steps, eval_every))
    probe_steps = (
        set(_interval_steps(phase_b_steps, restore_every))
        if restoration_enabled
        else {phase_b_steps}
    )
    phase_b_log_steps = tuple(sorted(evaluation_steps | probe_steps))
    phase_b_trajectory: list[dict[str, Any]] = []
    initial_probe = _h7_stage_snapshot(
        model,
        restoration,
        config,
        metrics=metrics,
        stage="phase_b_unlearning",
        condition=condition,
        stage_step=0,
        global_step=transition_step,
        examples_seen=transition_examples,
        split="restoration_probe",
    )
    initial_observed = evaluate_batch(model, phase_b_eval, config)
    initial_observed["rho_p_new"] = _named_proxy_reliance(
        model, phase_b_eval, config, "P_new"
    )
    initial_probe["phase_b_observed"] = initial_observed
    phase_b_trajectory.append(initial_probe)
    metrics.extend(
        make_metric_records(
            initial_observed,
            hypothesis="h7",
            split="phase_b_observed",
            global_step=transition_step,
            stage="phase_b_unlearning",
            stage_step=0,
            examples_seen=transition_examples,
            condition=condition,
        )
    )

    def phase_b_callback(snapshot: TrainingSnapshot) -> None:
        global_step = transition_step + snapshot.step
        global_examples = transition_examples + snapshot.record.samples_seen
        observed = evaluate_batch(snapshot.model, phase_b_eval, config)
        observed["rho_p_new"] = _named_proxy_reliance(
            snapshot.model, phase_b_eval, config, "P_new"
        )
        metrics.extend(
            make_metric_records(
                observed,
                hypothesis="h7",
                split="phase_b_observed",
                global_step=global_step,
                stage="phase_b_unlearning",
                stage_step=snapshot.step,
                examples_seen=global_examples,
                condition=condition,
            )
        )
        metrics.extend(
            _optimization_records(
                snapshot,
                hypothesis="h7",
                stage="phase_b_unlearning",
                condition=condition,
                global_step=global_step,
                global_examples=global_examples,
                n=len(phase_b),
            )
        )
        if snapshot.step not in probe_steps:
            return
        probe = _h7_stage_snapshot(
            snapshot.model,
            restoration,
            config,
            metrics=metrics,
            stage="phase_b_unlearning",
            condition=condition,
            stage_step=snapshot.step,
            global_step=global_step,
            examples_seen=global_examples,
            split="restoration_probe",
        )
        probe["phase_b_observed"] = observed
        phase_b_trajectory.append(probe)
        flip_effect = evaluate_standard_interventions(snapshot.model, restoration, config).get("flip_P")
        if flip_effect is not None:
            metrics.extend(
                make_metric_records(
                    flip_effect,
                    hypothesis="h7",
                    split="restoration_probe",
                    global_step=global_step,
                    stage="phase_b_unlearning",
                    stage_step=snapshot.step,
                    examples_seen=global_examples,
                    intervention="flip_P",
                    condition=condition,
                )
            )

    reset_optimizer = bool(get_path(config, "h7.reset_optimizer", True))
    phase_b_wall_start = time.perf_counter()
    phase_b_training = train_clean_sft(
        model,
        batch_for_training(phase_b, config),
        _sft_config(
            config,
            seed + 20,
            steps=phase_b_steps,
            reset_optimizer=reset_optimizer,
            log_steps=phase_b_log_steps,
            save_final_checkpoint=bool(get_path(config, "run.save_checkpoints", True)),
        ),
        optimizer=optimizer,
        callback=phase_b_callback,
    )
    phase_b_wall = time.perf_counter() - phase_b_wall_start
    checkpoints.update(_checkpoint_states(phase_b_training, "phase_b_"))

    probe_stage_steps = [int(point["stage_step"]) for point in phase_b_trajectory]
    old_reliance = [float(point["rho_p"]) for point in phase_b_trajectory]
    literal = reliance_half_life(probe_stage_steps, old_reliance, endpoint=0.0, adjusted=False)
    if old_reliance[-1] >= old_reliance[0] - 1e-12:
        # A no-change or increasing trajectory has no identifiable adjusted
        # half-life.  Calling the generic helper with endpoint=start would
        # otherwise turn this into a spurious event at t=0.
        adjusted = TimeToEvent(probe_stage_steps[-1], False, probe_stage_steps[-1])
    else:
        adjusted = reliance_half_life(
            probe_stage_steps,
            old_reliance,
            endpoint=old_reliance[-1],
            adjusted=True,
        )
    phase_a_steps = [int(point["stage_step"]) for point in phase_a_trajectory]
    phase_a_rho = [float(point["rho_p"]) for point in phase_a_trajectory]
    acquisition = acquisition_time(phase_a_steps, phase_a_rho, threshold, patience)
    auc = trapezoid_auc(probe_stage_steps, old_reliance) if len(probe_stage_steps) >= 2 else 0.0
    final_probe = evaluate_batch(model, restoration, config)
    final_observed = evaluate_batch(model, phase_b_eval, config)
    final_observed["rho_p_new"] = _named_proxy_reliance(model, phase_b_eval, config, "P_new")
    interventions = evaluate_standard_interventions(model, restoration, config)
    if condition == "replacement":
        interventions["flip_P_new"] = _named_channel_effect(
            model, phase_b_eval, config, "P_new"
        )
        metrics.extend(
            make_metric_records(
                interventions["flip_P_new"],
                hypothesis="h7",
                split="phase_b_observed",
                global_step=transition_step + phase_b_steps,
                stage="phase_b_unlearning",
                stage_step=phase_b_steps,
                examples_seen=transition_examples + phase_b_training.samples_seen,
                intervention="flip_P_new",
                condition=condition,
            )
        )
    flip_p = interventions.get("flip_P", {})
    change = float(old_reliance[-1] - old_reliance[0])
    events = {
        "phase_a_proxy_acquisition": _event_payload(acquisition),
        "half_life": _event_payload(literal),
        "literal_half_life": _event_payload(literal),
        "adjusted_half_life": _event_payload(adjusted),
        # Flat aliases keep older analysis notebooks functional.
        **literal.as_dict("half_life"),
        **literal.as_dict("literal_half_life"),
        **adjusted.as_dict("adjusted_half_life"),
    }
    summary = {
        "hypothesis": "h7",
        "seed": seed,
        "condition": condition,
        "final": {
            "restoration_probe": final_probe,
            "phase_b_observed": final_observed,
            "interventions": interventions,
            "phase_a_gate_passed": gate_passed,
            "gate_bypassed_for_smoke": smoke_bypass,
            "eligible_for_primary_analysis": gate_passed,
            "phase_b_executed": True,
            "old_proxy_reliance_auc": float(auc),
            "old_proxy_reliance_change_ate": change,
            "old_proxy_probability_ate": float(flip_p.get("probability_ate", float("nan"))),
            "old_proxy_logit_ate": float(flip_p.get("logit_ate", float("nan"))),
            "restoration_rho_p": float(final_probe["rho_p"]),
            "new_proxy_rho": float(final_observed["rho_p_new"]),
        },
        "model": dict(model_report),
        "events": events,
        "data": {
            **details,
            "n_phase_a": len(phase_a),
            "n_phase_b": len(phase_b),
            "n_restoration_probe": len(restoration),
            "old_proxy_available_during_phase_b": condition != "removal",
            "new_proxy_reliable_during_phase_b": condition == "replacement",
        },
        "training": {
            "phase_a_steps": transition_step,
            "phase_a_examples": transition_examples,
            "phase_b_steps": phase_b_training.optimizer_steps,
            "phase_b_examples": phase_b_training.samples_seen,
            "optimizer_reset_at_phase_b": reset_optimizer,
            "weight_decay": _float(config, "train.weight_decay", 0.0),
            "wall_seconds": phase_a_wall + phase_b_wall,
            "restoration_probe_interval": restore_every,
            "restoration_probe_enabled": restoration_enabled,
        },
    }
    predictions: list[dict[str, Any]] = []
    if bool(get_path(config, "evaluation.save_predictions", True)):
        predictions.extend(
            _tag_predictions(
                prediction_records(model, restoration, config, split="restoration_probe"),
                stage="phase_b_final",
            )
        )
        predictions.extend(
            _tag_predictions(
                prediction_records(model, flip_channels(restoration, "P"), config, split="restoration_probe", intervention="flip_P"),
                stage="phase_b_final",
            )
        )
        predictions.extend(
            _tag_predictions(
                prediction_records(model, phase_b_eval, config, split="phase_b_observed"),
                stage="phase_b_final",
            )
        )
        if condition == "replacement":
            predictions.extend(
                _tag_predictions(
                    prediction_records(
                        model,
                        flip_channels(phase_b_eval, "P_new"),
                        config,
                        split="phase_b_observed",
                        intervention="flip_P_new",
                    ),
                    stage="phase_b_final",
                )
            )
    return ProtocolResult(
        model=model,
        summary=summary,
        metrics=metrics,
        predictions=predictions,
        checkpoints=checkpoints,
        evaluation_batch=restoration,
    )


# ---------------------------------------------------------------------------
# H8: fixed-volume history, alignment, and perturbation


def _volume_steps(volume: int, batch_size: int) -> int:
    if volume < 1:
        raise ValueError("example volume must be positive")
    return int(math.ceil(volume / batch_size))


def _mixed_goals(y: np.ndarray, old: np.ndarray, probability_old: float, seed: int) -> np.ndarray:
    if not 0.0 <= probability_old <= 1.0:
        raise ValueError("old-goal mixture probability must lie in [0,1]")
    intended = np.asarray(y, dtype=np.int8)
    old_goal = np.asarray(old, dtype=np.int8)
    disagreement = np.flatnonzero(intended != old_goal)
    count = int(round(probability_old * len(disagreement)))
    rng = np.random.default_rng(seed)
    negative = disagreement[intended[disagreement] == -1]
    positive = disagreement[intended[disagreement] == 1]
    rng.shuffle(negative)
    rng.shuffle(positive)
    negative_count = count // 2
    positive_count = count - negative_count
    chosen = np.concatenate((negative[:negative_count], positive[:positive_count]))
    target = intended.copy()
    target[chosen] = old_goal[chosen]
    return target


def _h8_batch(
    config: Mapping[str, Any], n: int, seed: int, *, split: str, id_offset: int
) -> SemanticBatch:
    if n < 2 or n % 2:
        raise ValueError("H8 fixed example volumes must be positive even integers")
    batch = make_conflict_dataset(
        n,
        _integer(config, "data.k", 3),
        seed,
        id_offset=id_offset,
        **_dataset_arguments(config),
    )
    return batch.with_updates(metadata={**dict(batch.metadata), "stage_split": split})


def _h8_eval_point(
    model: nn.Module,
    evaluation: SemanticBatch,
    config: Mapping[str, Any],
    metrics: list[MetricRow],
    *,
    stage: str,
    history: str,
    stage_step: int,
    global_step: int,
    stage_examples: int,
    global_examples: int,
) -> dict[str, Any]:
    values = evaluate_batch(model, evaluation, config)
    metrics.extend(
        make_metric_records(
            values,
            hypothesis="h8",
            split="goal_disagreement_eval",
            global_step=global_step,
            stage=stage,
            stage_step=stage_step,
            examples_seen=global_examples,
            condition=history,
        )
    )
    return {
        "stage_step": int(stage_step),
        "global_step": int(global_step),
        "stage_examples": int(stage_examples),
        "global_examples": int(global_examples),
        "rho_g0": float(values["rho_p"]),
        "rho_g1": float(values["rho_y"]),
        **values,
    }


def _h8_train_stage(
    model: nn.Module,
    train: SemanticBatch,
    evaluation: SemanticBatch,
    config: Mapping[str, Any],
    seed: int,
    *,
    stage: str,
    history: str,
    steps: int,
    optimizer: torch.optim.Optimizer | None,
    reset_optimizer: bool,
    global_step_offset: int,
    global_examples_offset: int,
    metrics: list[MetricRow],
    trajectory: list[dict[str, Any]],
    exact_volume: int | None = None,
    match_band: tuple[float, float] | None = None,
    match_patience: int = 1,
) -> TrainingResult:
    batch_size = _integer(config, "train.batch_size", 128)
    if match_band is not None:
        if not 0.0 <= match_band[0] <= match_band[1] <= 1.0:
            raise ValueError("H8 behavior-match band must satisfy 0 <= low <= high <= 1")
        if match_patience < 1:
            raise ValueError("h8.stage1_match_patience must be positive")
        logs: Sequence[int] = tuple(range(1, steps + 1))
    else:
        logs = log_spaced_steps(steps)
    consecutive_matches = 0

    def callback(snapshot: TrainingSnapshot) -> None:
        nonlocal consecutive_matches
        point = _h8_eval_point(
            snapshot.model,
            evaluation,
            config,
            metrics,
            stage=stage,
            history=history,
            stage_step=snapshot.step,
            global_step=global_step_offset + snapshot.step,
            stage_examples=snapshot.record.samples_seen,
            global_examples=global_examples_offset + snapshot.record.samples_seen,
        )
        trajectory.append(point)
        metrics.extend(
            _optimization_records(
                snapshot,
                hypothesis="h8",
                stage=stage,
                condition=history,
                global_step=global_step_offset + snapshot.step,
                global_examples=global_examples_offset + snapshot.record.samples_seen,
                n=len(train),
            )
        )
        if match_band is not None:
            reliance_value = float(point["rho_g1"])
            consecutive_matches = (
                consecutive_matches + 1
                if match_band[0] <= reliance_value <= match_band[1]
                else 0
            )
            if consecutive_matches >= match_patience:
                raise StopTraining

    result = train_clean_sft(
        model,
        batch_for_training(train, config),
        _sft_config(
            config,
            seed,
            steps=steps,
            reset_optimizer=reset_optimizer,
            log_steps=logs,
            batch_size=batch_size,
            save_final_checkpoint=bool(get_path(config, "run.save_checkpoints", True)),
        ),
        optimizer=optimizer,
        callback=callback,
    )
    if exact_volume is not None and result.samples_seen != exact_volume:
        raise RuntimeError(
            f"fixed-volume stage consumed {result.samples_seen} examples; expected {exact_volume}"
        )
    return result


def _h8_perturbation(
    base: SemanticBatch, config: Mapping[str, Any], seed: int, arm: str
) -> tuple[SemanticBatch, dict[str, Any]]:
    y = np.asarray(base.y, dtype=np.int8)
    old = np.asarray(base.channels["P"], dtype=np.int8)
    if arm == "neutral":
        # Neutral evidence lies on the observational-equivalence set g0=g1:
        # it teaches the shared action without identifying either candidate.
        changed = flip_channels(base, "P")
        changed = _with_target(changed, y)
        definition = "agreement states only (g0=g1); no goal-identifying evidence"
    elif arm == "weak_conflict":
        probability = _float(config, "h8.weak_conflict_p_old", 0.55)
        target = _mixed_goals(y, old, probability, seed + 2)
        definition = f"old goal labels on {probability:g} of disagreement examples"
        changed = _with_target(base, target)
    elif arm in {"intended_removal", "removal"}:
        # Remove the intended interaction code on observational-equivalence
        # examples.  Starting from the all-conflict bank without first moving
        # to agreement would leave P=-Y, so the retained intended labels would
        # be systematic counterevidence against the old goal rather than an
        # absence intervention.
        agreement = flip_channels(base, "P")
        names = [f"R_{index}" for index in range(1, base.active_k + 1)]
        changed = mask_channels(agreement, names) if names else agreement
        changed = _with_target(changed, y)
        definition = (
            "intended interaction-code channels masked on g0=g1 agreement states; "
            "no anti-old-goal labels remain"
        )
    elif arm in {"partial_reversal", "reversal"}:
        probability = _float(config, "h8.p_old_on_disagreement", 0.75)
        target = _mixed_goals(y, old, probability, seed + 3)
        changed = _with_target(base, target)
        definition = f"old goal labels on {probability:g} of disagreement examples"
    else:
        raise ValueError(
            "h8.perturbation must be neutral, weak_conflict, intended_removal, or partial_reversal"
        )
    realized_old_goal = np.asarray(changed.channels["P"], dtype=np.int8)
    disagreement = y != realized_old_goal
    realized_old = (
        float(
            np.mean(
                np.asarray(changed.target)[disagreement]
                == realized_old_goal[disagreement]
            )
        )
        if np.any(disagreement)
        else float("nan")
    )
    old_goal_counterevidence = float(
        np.mean(np.asarray(changed.target) != realized_old_goal)
    )
    return changed, {
        "arm": arm,
        "definition": definition,
        "realized_p_old_on_disagreement": realized_old,
        "old_goal_counterevidence_rate": old_goal_counterevidence,
        "goal_identifying_disagreement_examples": int(np.sum(disagreement)),
        "intended_channels_present": arm not in {"intended_removal", "removal"},
    }


def _h8_stage1_eligible(
    stage1_mode: str, *, gate_passed: bool, in_match_band: bool
) -> bool:
    """Apply the narrow behavior band only to the behavior-matched design."""

    return bool(
        gate_passed
        and (stage1_mode != "behavior_matched" or in_match_band)
    )


def run_h8(config: Mapping[str, Any], seed: int) -> ProtocolResult:
    """Measure history-dependent rebound with fixed-volume or behavior-matched alignment."""

    seed_everything(seed)
    history = str(get_path(config, "h8.history", "old_goal")).lower()
    aliases = {"sham": "compute_matched_sham", "old": "old_goal"}
    history = aliases.get(history, history)
    if history not in {"old_goal", "control", "compute_matched_sham"}:
        raise ValueError("h8.history must be old_goal, control, or compute_matched_sham")
    units = str(get_path(config, "h8.n0_units", "examples"))
    if units != "examples":
        raise ValueError("H8 N0/N1 are fixed example volumes; h8.n0_units must be examples")
    n0 = _integer(config, "h8.n0", 4096)
    n1 = _integer(config, "h8.n1", 8192)
    if n0 < 0 or n1 < 2 or n1 % 2 or (n0 and n0 % 2):
        raise ValueError("h8.n0 must be a non-negative even integer and h8.n1 a positive even integer")
    stage1_mode = str(get_path(config, "h8.stage1_mode", "fixed")).lower()
    if stage1_mode == "matched":
        stage1_mode = "behavior_matched"
    if stage1_mode not in {"fixed", "behavior_matched"}:
        raise ValueError("h8.stage1_mode must be fixed or behavior_matched")
    stage1_threshold = _float(config, "h8.stage1_threshold", 0.9)
    band = list(get_path(config, "h8.stage1_match_band", [stage1_threshold, 1.0]))
    if len(band) != 2:
        raise ValueError("h8.stage1_match_band must contain [low, high]")
    match_band = (float(band[0]), float(band[1]))
    match_patience = _integer(config, "h8.stage1_match_patience", 2)
    n_eval = _integer(config, "data.n_eval", 4096)
    evaluation = _h8_batch(config, n_eval, seed + 50, split="evaluation", id_offset=8_000_000)
    build_batch = evaluation
    model, model_report = build_model(build_batch, config, seed)
    model.to(_device(config))
    metrics: list[MetricRow] = []
    checkpoints: dict[str, dict[str, Any]] = {"initial": _cpu_state(model)}
    predictions: list[dict[str, Any]] = []
    optimizer: torch.optim.Optimizer | None = None
    reset_between = bool(get_path(config, "h8.reset_optimizer", True))
    global_step = 0
    global_examples = 0
    wall_start = time.perf_counter()
    stage0_name = {
        "old_goal": "stage0_old_goal",
        "control": "stage0_control",
        "compute_matched_sham": "stage0_compute_sham",
    }[history]

    stage0_trajectory: list[dict[str, Any]] = [
        _h8_eval_point(
            model,
            evaluation,
            config,
            metrics,
            stage=stage0_name,
            history=history,
            stage_step=0,
            global_step=0,
            stage_examples=0,
            global_examples=0,
        )
    ]
    stage0_result: TrainingResult | None = None
    stage0_actual_volume = 0
    if n0 > 0 and history != "control":
        stage0 = _h8_batch(config, n0, seed + 1, split="stage0", id_offset=1_000_000)
        if history == "old_goal":
            stage0 = _with_target(stage0, np.asarray(stage0.channels["P"], dtype=np.int8))
        else:
            stage0 = _with_target(stage0, balanced_signs(n0, seed + 2))
        batch_size = _integer(config, "train.batch_size", 128)
        stage0_steps = _volume_steps(n0, batch_size)
        stage0_result = _h8_train_stage(
            model,
            stage0,
            evaluation,
            config,
            seed + 100,
            stage=stage0_name,
            history=history,
            steps=stage0_steps,
            optimizer=None,
            reset_optimizer=True,
            global_step_offset=global_step,
            global_examples_offset=global_examples,
            metrics=metrics,
            trajectory=stage0_trajectory,
            exact_volume=n0,
        )
        optimizer = stage0_result.optimizer
        stage0_actual_volume = stage0_result.samples_seen
        global_step += stage0_result.optimizer_steps
        global_examples += stage0_result.samples_seen
        checkpoints.update(_checkpoint_states(stage0_result, "stage0_"))
    else:
        checkpoints["stage0_final"] = _cpu_state(model)

    stage1 = _h8_batch(config, n1, seed + 3, split="stage1", id_offset=2_000_000)
    stage1 = _with_target(stage1, np.asarray(stage1.y, dtype=np.int8))
    stage1_trajectory: list[dict[str, Any]] = [
        _h8_eval_point(
            model,
            evaluation,
            config,
            metrics,
            stage="stage1_alignment",
            history=history,
            stage_step=0,
            global_step=global_step,
            stage_examples=0,
            global_examples=global_examples,
        )
    ]
    batch_size = _integer(config, "train.batch_size", 128)
    stage1_steps = _volume_steps(n1, batch_size)
    stage1_result = _h8_train_stage(
        model,
        stage1,
        evaluation,
        config,
        seed + 200,
        stage="stage1_alignment",
        history=history,
        steps=stage1_steps,
        optimizer=optimizer,
        reset_optimizer=reset_between,
        global_step_offset=global_step,
        global_examples_offset=global_examples,
        metrics=metrics,
        trajectory=stage1_trajectory,
        exact_volume=n1 if stage1_mode == "fixed" else None,
        match_band=match_band if stage1_mode == "behavior_matched" else None,
        match_patience=match_patience,
    )
    optimizer = stage1_result.optimizer
    global_step += stage1_result.optimizer_steps
    global_examples += stage1_result.samples_seen
    checkpoints.update(_checkpoint_states(stage1_result, "stage1_"))
    stage1_final = evaluate_batch(model, evaluation, config)
    stage1_gate_passed = float(stage1_final["rho_y"]) >= stage1_threshold
    in_match_band = match_band[0] <= float(stage1_final["rho_y"]) <= match_band[1]
    stage1_eligible = _h8_stage1_eligible(
        stage1_mode,
        gate_passed=stage1_gate_passed,
        in_match_band=in_match_band,
    )
    if bool(get_path(config, "evaluation.save_predictions", True)):
        predictions.extend(
            _tag_predictions(
                prediction_records(model, evaluation, config, split="goal_disagreement_eval"),
                stage="stage1_pre_perturbation",
                history=history,
            )
        )

    arm = str(get_path(config, "h8.perturbation", "partial_reversal")).lower()
    stage2_steps = _integer(config, "h8.stage2_steps", 2048)
    if stage2_steps < 1:
        raise ValueError("h8.stage2_steps must be positive")
    stage2_base = _h8_batch(
        config,
        _integer(config, "data.n_train", 4096),
        seed + 4,
        split="stage2",
        id_offset=3_000_000,
    )
    stage2, perturbation = _h8_perturbation(stage2_base, config, seed + 5, arm)
    before_g0 = float(stage1_final["rho_p"])
    before_g1 = float(stage1_final["rho_y"])
    stage2_trajectory: list[dict[str, Any]] = [
        _h8_eval_point(
            model,
            evaluation,
            config,
            metrics,
            stage="stage2_perturbation",
            history=history,
            stage_step=0,
            global_step=global_step,
            stage_examples=0,
            global_examples=global_examples,
        )
    ]
    stage2_result = _h8_train_stage(
        model,
        stage2,
        evaluation,
        config,
        seed + 300,
        stage="stage2_perturbation",
        history=history,
        steps=stage2_steps,
        optimizer=optimizer,
        reset_optimizer=reset_between,
        global_step_offset=global_step,
        global_examples_offset=global_examples,
        metrics=metrics,
        trajectory=stage2_trajectory,
    )
    global_step += stage2_result.optimizer_steps
    global_examples += stage2_result.samples_seen
    checkpoints.update(_checkpoint_states(stage2_result, "stage2_"))
    final_eval = evaluate_batch(model, evaluation, config)
    after_g0 = float(final_eval["rho_p"])
    after_g1 = float(final_eval["rho_y"])
    rebound = after_g0 - before_g0
    event_steps = [int(point["stage_examples"]) for point in stage2_trajectory]
    event_values = [float(point["rho_g0"]) for point in stage2_trajectory]
    reactivate = reactivation_time(
        event_steps,
        event_values,
        threshold=_float(config, "h8.reactivation_threshold", 0.9),
        persistence=_integer(config, "evaluation.persistence", 2),
    )
    stage1_event = acquisition_time(
        [int(point["stage_examples"]) for point in stage1_trajectory],
        [float(point["rho_g1"]) for point in stage1_trajectory],
        threshold=stage1_threshold,
        persistence=_integer(config, "evaluation.persistence", 2),
    )
    event_kind = "reactivation" if history == "old_goal" else "scratch_acquisition"
    event_payload = _event_payload(reactivate)
    events = {
        "stage1_alignment_acquisition": _event_payload(stage1_event),
        "stage1_alignment_acquisition_interval": _checkpoint_interval_payload(
            stage1_event,
            [int(point["stage_examples"]) for point in stage1_trajectory],
        ),
        "stage1_threshold_gate": {
            "passed": stage1_gate_passed,
            "threshold": stage1_threshold,
            "rho_g1": float(stage1_final["rho_y"]),
            "in_match_band": in_match_band,
        },
        "reactivation": event_payload,
        "reactivation_interval": _checkpoint_interval_payload(
            reactivate, event_steps
        ),
        event_kind: event_payload,
        **reactivate.as_dict("reactivation"),
    }
    summary = {
        "hypothesis": "h8",
        "seed": seed,
        "condition": history,
        "final": {
            **final_eval,
            "rho_g0_before_perturbation": before_g0,
            "rho_g0_after_perturbation": after_g0,
            "rho_g1_before_perturbation": before_g1,
            "rho_g1_after_perturbation": after_g1,
            "rho_g0": after_g0,
            "rho_g1": after_g1,
            "rebound_g0": rebound,
            "stage1_gate_passed": stage1_gate_passed,
            "stage1_in_match_band": in_match_band,
            "stage1_match_band_required": stage1_mode == "behavior_matched",
            "eligible_for_primary_analysis": stage1_eligible,
            "reactivation_observed": bool(reactivate.observed),
            "reactivation_examples": int(reactivate.time),
            "perturbation": arm,
        },
        "model": dict(model_report),
        "events": events,
        "data": {
            "history": history,
            "requested_n0": n0,
            "realized_n0": stage0_actual_volume,
            "requested_n1": n1,
            "realized_n1": stage1_result.samples_seen,
            "n1": n1,
            "n0_units": "examples",
            "stage1_mode": stage1_mode,
            "stage1_match_band": list(match_band),
            "stage1_match_patience": match_patience,
            "control_received_old_goal_examples": False,
            "compute_matched_sham": history == "compute_matched_sham",
            "perturbation": perturbation,
        },
        "training": {
            "stage0_steps": 0 if stage0_result is None else stage0_result.optimizer_steps,
            "stage0_examples": stage0_actual_volume,
            "stage1_steps": stage1_result.optimizer_steps,
            "stage1_examples": stage1_result.samples_seen,
            "stage2_steps": stage2_result.optimizer_steps,
            "stage2_examples": stage2_result.samples_seen,
            "optimizer_reset_between_stages": reset_between,
            "global_steps": global_step,
            "global_examples": global_examples,
            "wall_seconds": time.perf_counter() - wall_start,
        },
    }
    if bool(get_path(config, "evaluation.save_predictions", True)):
        predictions.extend(
            _tag_predictions(
                prediction_records(model, evaluation, config, split="goal_disagreement_eval"),
                stage="stage2_final",
                history=history,
                perturbation=arm,
            )
        )
        predictions.extend(
            _tag_predictions(
                prediction_records(
                    model,
                    flip_channels(evaluation, "P"),
                    config,
                    split="goal_disagreement_eval",
                    intervention="flip_P",
                ),
                stage="stage2_final",
                history=history,
                perturbation=arm,
            )
        )
    return ProtocolResult(
        model=model,
        summary=summary,
        metrics=metrics,
        predictions=predictions,
        checkpoints=checkpoints,
        evaluation_batch=evaluation,
    )


# ---------------------------------------------------------------------------
# H9: multiple proxy rules and a context-dependent selector


def _h9_proxy_channel_names(batch: SemanticBatch, proxy: int) -> tuple[str, ...]:
    if proxy not in (0, 1):
        raise ValueError("H9 proxy index must be 0 or 1")
    base = f"P{proxy}"
    if base in batch.channels:
        return (base,)
    degree = int(batch.metadata.get(f"proxy{proxy}_degree", 0))
    names = tuple(f"{base}_{index}" for index in range(1, degree + 1))
    if degree < 1 or any(name not in batch.channels for name in names):
        raise KeyError(f"H9 batch is missing the encoded {base} proxy")
    return names


def _h9_proxy_values(batch: SemanticBatch, proxy: int) -> np.ndarray:
    names = _h9_proxy_channel_names(batch, proxy)
    code = np.column_stack(
        [np.asarray(batch.channels[name], dtype=np.int8) for name in names]
    )
    return np.prod(code, axis=1, dtype=np.int8)


def _h9_flip_proxy(batch: SemanticBatch, proxy: int) -> SemanticBatch:
    # Flipping one Rademacher component reverses the represented proxy goal
    # without redundantly perturbing every component.
    return flip_channels(batch, _h9_proxy_channel_names(batch, proxy)[0])


def _h9_set_context(batch: SemanticBatch, context: int, *, present: bool = True) -> SemanticBatch:
    if context not in (0, 1):
        raise ValueError("H9 context clamp must be 0 or 1")
    channels = {name: np.array(value, copy=True) for name, value in batch.channels.items()}
    channels["C"] = np.full(len(batch), context, dtype=np.int8)
    channels["C_present"] = np.full(len(batch), int(present), dtype=np.int8)
    metadata = dict(batch.metadata)
    metadata["context_clamp"] = context
    metadata["context_present"] = bool(present)
    return batch.with_updates(channels=channels, metadata=metadata)


def _h9_directional_mismatch(batch: SemanticBatch, observed: int, environment: int) -> SemanticBatch:
    positions = np.flatnonzero(np.asarray(batch.latents["E"], dtype=np.int8) == environment)
    if not len(positions):
        raise ValueError("H9 mismatch evaluation has no rows for the requested environment")
    return _h9_set_context(batch.select(positions), observed)


def _h9_rebalance_context(
    source: SemanticBatch,
    n: int,
    probability_e1: float,
    seed: int,
) -> SemanticBatch:
    if not 0.0 <= probability_e1 <= 1.0:
        raise ValueError("h9.context_balance must lie in [0,1]")
    raw_e1 = n * probability_e1
    if abs(raw_e1 - round(raw_e1)) > 1e-9:
        raise ValueError(
            f"h9.context_balance={probability_e1:g} is not exactly realizable with n_train={n}"
        )
    count_e1 = int(round(raw_e1))
    counts = {0: n - count_e1, 1: count_e1}
    environment = np.asarray(source.latents["E"], dtype=np.int8)
    target = np.asarray(source.y, dtype=np.int8)
    rng = np.random.default_rng(seed)
    selected: list[np.ndarray] = []
    for context in (0, 1):
        count = counts[context]
        # When both context counts are odd, put the unmatched target sign in
        # opposite contexts so the aggregate target remains exactly balanced.
        negative_count = count // 2 + int(count % 2 and context == 0)
        positive_count = count - negative_count
        negative = np.flatnonzero((environment == context) & (target == -1))
        positive = np.flatnonzero((environment == context) & (target == 1))
        rng.shuffle(negative)
        rng.shuffle(positive)
        if negative_count > len(negative) or positive_count > len(positive):
            raise RuntimeError("internal H9 context-rebalancing source is too small")
        selected.extend((negative[:negative_count], positive[:positive_count]))
    indices = np.concatenate(selected)
    rng.shuffle(indices)
    batch = source.select(indices)
    metadata = dict(batch.metadata)
    metadata.update(
        {
            "requested_context_balance": float(probability_e1),
            "realized_context_balance": float(np.mean(np.asarray(batch.latents["E"]) == 1)),
            "factorial_complete": False,
        }
    )
    return batch.with_updates(metadata=metadata)


def _h9_condition_batches(
    config: Mapping[str, Any], seed: int
) -> tuple[SemanticBatch, dict[str, SemanticBatch]]:
    n_train = _integer(config, "data.n_train", 4096)
    n_eval = _integer(config, "data.n_eval", 4096)
    max_k = _integer(config, "data.max_k", 5)
    state_dim = _integer(config, "data.state_dim", 0)
    proxy0_degree = _integer(config, "h9.proxy0_degree", 1)
    proxy1_degree = _integer(config, "h9.proxy1_degree", 1)
    if proxy0_degree < 1 or proxy1_degree < 1:
        raise ValueError("h9.proxy0_degree and h9.proxy1_degree must be positive")
    dataset_arguments = {
        "max_k": max_k,
        "state_dim": state_dim,
        "proxy0_degree": proxy0_degree,
        "proxy1_degree": proxy1_degree,
    }
    context_balance = _float(config, "h9.context_balance", 0.5)
    if context_balance == 0.5:
        train = make_h9_dataset(
            n_train, seed, condition="normal", **dataset_arguments
        )
    else:
        source = make_h9_dataset(
            2 * n_train, seed, condition="normal", **dataset_arguments
        )
        train = _h9_rebalance_context(source, n_train, context_balance, seed + 911)
    normal = make_h9_dataset(
        n_eval, seed + 1, condition="normal", **dataset_arguments
    )
    available = {
        "normal": normal,
        "removed": make_h9_dataset(
            n_eval, seed + 1, condition="removed", **dataset_arguments
        ),
        "randomized": make_h9_dataset(
            n_eval, seed + 1, condition="randomized", **dataset_arguments
        ),
        "mismatched": make_h9_dataset(
            n_eval, seed + 1, condition="mismatched", **dataset_arguments
        ),
        "mismatch_0_1": _h9_directional_mismatch(normal, observed=0, environment=1),
        "mismatch_1_0": _h9_directional_mismatch(normal, observed=1, environment=0),
    }
    requested = get_path(
        config,
        "h9.context_conditions",
        ["normal", "removed", "randomized", "mismatch_0_1", "mismatch_1_0"],
    )
    if isinstance(requested, str):
        requested = [requested]
    aliases = {"context_removed": "removed", "context_randomized": "randomized", "mismatch": "mismatched"}
    names = [aliases.get(str(name), str(name)) for name in requested]
    if "normal" not in names:
        names.insert(0, "normal")
    unknown = sorted(set(names) - set(available))
    if unknown:
        raise ValueError(f"unknown H9 context conditions: {unknown}")
    return train, {name: available[name] for name in names}


def _h9_behavior(model: nn.Module, batch: SemanticBatch, config: Mapping[str, Any]) -> dict[str, float | int]:
    logits = predict_logits(model, batch, config)
    prediction = binary_predictions(logits)
    p0 = _h9_proxy_values(batch, 0)
    p1 = _h9_proxy_values(batch, 1)
    target = np.asarray(batch.y, dtype=np.int8)
    environment = np.asarray(batch.latents["E"], dtype=np.int8)
    selected = np.where(environment == 0, p0, p1)
    observed_context = np.asarray(batch.channels["C"], dtype=np.int8)
    observed_selected = np.where(observed_context == 0, p0, p1)
    values: dict[str, float | int] = {
        "target_accuracy": float(np.mean(prediction == target)),
        "rho_p0": reliance(prediction, p0),
        "rho_p1": reliance(prediction, p1),
        "rho_p_environment": reliance(prediction, selected),
        "rho_p_observed_context": reliance(prediction, observed_selected),
        "invalid_rate": float(np.mean(prediction == 0)),
        "n": len(batch),
    }
    for context in (0, 1):
        mask = environment == context
        if np.any(mask):
            values[f"target_accuracy_e{context}"] = float(
                np.mean(prediction[mask] == target[mask])
            )
            values[f"rho_p{context}_e{context}"] = float(
                np.mean(prediction[mask] == selected[mask])
            )
    return values


def _h9_selector(
    model: nn.Module, batch: SemanticBatch, config: Mapping[str, Any]
) -> tuple[dict[str, float], SemanticBatch, SemanticBatch]:
    c0 = _h9_set_context(batch, 0)
    c1 = _h9_set_context(batch, 1)
    prediction0 = binary_predictions(predict_logits(model, c0, config))
    prediction1 = binary_predictions(predict_logits(model, c1, config))
    p0 = _h9_proxy_values(batch, 0)
    p1 = _h9_proxy_values(batch, 1)
    try:
        values = context_selector_metrics(prediction0, prediction1, p0, p1)
    except ValueError:
        values = {
            "strict_context_switching": float("nan"),
            "gating_index": float("nan"),
            "rho_p0_c0": reliance(prediction0, p0),
            "rho_p0_c1": reliance(prediction1, p0),
            "rho_p1_c0": reliance(prediction0, p1),
            "rho_p1_c1": reliance(prediction1, p1),
        }
    return values, c0, c1


def _h9_sensitivity_matrix(
    model: nn.Module,
    batch: SemanticBatch,
    config: Mapping[str, Any],
) -> tuple[dict[str, dict[str, dict[str, float]]], list[MetricRow]]:
    matrix: dict[str, dict[str, dict[str, float]]] = {}
    records: list[MetricRow] = []
    for context in (0, 1):
        clamped = _h9_set_context(batch, context)
        intended = _h9_proxy_values(clamped, context)
        semantic = _with_target(clamped, intended, change_semantic_y=True)
        base_logits = predict_logits(model, semantic, config)
        row: dict[str, dict[str, float]] = {}
        for proxy in (0, 1):
            changed = _h9_flip_proxy(semantic, proxy)
            changed_logits = predict_logits(model, changed, config)
            effect = intervention_metrics(base_logits, changed_logits, intended)
            absolute_logit_change = float(
                np.mean(
                    np.abs(
                        (changed_logits[:, 1] - changed_logits[:, 0])
                        - (base_logits[:, 1] - base_logits[:, 0])
                    )
                )
            )
            row[f"P{proxy}"] = {
                **effect,
                "absolute_logit_change": absolute_logit_change,
            }
            records.extend(
                make_metric_records(
                    {**effect, "absolute_logit_change": absolute_logit_change},
                    hypothesis="h9",
                    split=f"clamp_C{context}",
                    global_step=_integer(config, "train.steps", 1000),
                    stage="final_sensitivity",
                    intervention=f"flip_P{proxy}",
                    condition="context_selector",
                )
            )
        matrix[f"C{context}"] = row
    return matrix, records


def _h9_prediction_records(
    model: nn.Module,
    batch: SemanticBatch,
    config: Mapping[str, Any],
    *,
    split: str,
    intervention: str = "none",
) -> list[dict[str, Any]]:
    logits = predict_logits(model, batch, config)
    predictions = binary_predictions(logits)
    probabilities = softmax(logits)[:, 1]
    p0 = _h9_proxy_values(batch, 0)
    p1 = _h9_proxy_values(batch, 1)
    context = np.asarray(batch.channels["C"], dtype=np.int8)
    environment = np.asarray(batch.latents["E"], dtype=np.int8)
    return [
        {
            "sample_id": int(batch.sample_id[index]),
            "state_id": int(batch.state_id[index]),
            "split": split,
            "intervention": intervention,
            "y": int(batch.y[index]),
            "proxy": int(p0[index] if environment[index] == 0 else p1[index]),
            "P0": int(p0[index]),
            "P1": int(p1[index]),
            "C": int(context[index]),
            "C_present": int(batch.channels["C_present"][index]),
            "E": int(environment[index]),
            "prediction": int(predictions[index]),
            "probability_positive": float(probabilities[index]),
        }
        for index in range(len(batch))
    ]


def _h9_single_rule_controls(
    config: Mapping[str, Any],
    seed: int,
    train: SemanticBatch,
    evaluation: SemanticBatch,
    metrics: list[MetricRow],
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    if not bool(get_path(config, "h9.include_single_rule_controls", True)):
        return {"enabled": False}, {}
    controls: dict[str, Any] = {"enabled": True}
    checkpoints: dict[str, dict[str, Any]] = {}
    steps = _integer(config, "h9.calibration_steps", _integer(config, "train.steps", 1000))
    threshold = _float(config, "h9.mastery_threshold", 0.9)
    for proxy in (0, 1):
        target_train = _h9_proxy_values(train, proxy)
        target_eval = _h9_proxy_values(evaluation, proxy)
        control_train = _with_target(train, target_train, change_semantic_y=True)
        control_eval = _with_target(evaluation, target_eval, change_semantic_y=True)
        control_model, report = build_model(control_train, config, seed)
        control_model.to(_device(config))
        result = train_clean_sft(
            control_model,
            batch_for_training(control_train, config),
            _sft_config(
                config,
                seed + 700 + proxy,
                steps=steps,
                reset_optimizer=True,
                log_steps=log_spaced_steps(steps),
                save_final_checkpoint=bool(get_path(config, "run.save_checkpoints", True)),
            ),
        )
        values = evaluate_batch(control_model, control_eval, config)
        controls[f"P{proxy}"] = {
            "target_accuracy": float(values["target_accuracy"]),
            "mastered": bool(values["target_accuracy"] >= threshold),
            "model": report,
            "steps": result.optimizer_steps,
            "examples": result.samples_seen,
        }
        metrics.extend(
            make_metric_records(
                values,
                hypothesis="h9",
                split=f"single_rule_P{proxy}",
                global_step=steps,
                stage="single_rule_calibration",
                condition=f"P{proxy}",
                examples_seen=result.samples_seen,
            )
        )
        checkpoints.update(_checkpoint_states(result, f"calibration_P{proxy}_"))
    controls["all_mastered"] = bool(controls["P0"]["mastered"] and controls["P1"]["mastered"])
    return controls, checkpoints


def run_h9(config: Mapping[str, Any], seed: int) -> ProtocolResult:
    """Train both proxy rules and test whether context causally gates them."""

    seed_everything(seed)
    train, conditions = _h9_condition_batches(config, seed)
    normal = conditions["normal"]
    model, model_report = build_model(train, config, seed)
    model.to(_device(config))
    metrics: list[MetricRow] = []
    trajectory: list[dict[str, Any]] = []
    steps = _integer(config, "train.steps", 1000)

    def record_selector(model_at_step: nn.Module, step: int, examples: int) -> None:
        behavior = _h9_behavior(model_at_step, normal, config)
        selector, _, _ = _h9_selector(model_at_step, normal, config)
        point = {"step": int(step), "examples": int(examples), **behavior, **selector}
        trajectory.append(point)
        metrics.extend(
            make_metric_records(
                behavior,
                hypothesis="h9",
                split="normal",
                global_step=step,
                stage="context_selector_train",
                examples_seen=examples,
                condition="context_selector",
            )
        )
        metrics.extend(
            make_metric_records(
                {**selector, "n": len(normal)},
                hypothesis="h9",
                split="paired_context_clamps",
                global_step=step,
                stage="context_selector_train",
                examples_seen=examples,
                intervention="clamp_C_0_vs_1",
                condition="context_selector",
            )
        )

    record_selector(model, 0, 0)

    def callback(snapshot: TrainingSnapshot) -> None:
        record_selector(snapshot.model, snapshot.step, snapshot.record.samples_seen)
        metrics.extend(
            _optimization_records(
                snapshot,
                hypothesis="h9",
                stage="context_selector_train",
                condition="context_selector",
                global_step=snapshot.step,
                global_examples=snapshot.record.samples_seen,
                n=len(train),
            )
        )

    wall_start = time.perf_counter()
    training = train_clean_sft(
        model,
        batch_for_training(train, config),
        _sft_config(
            config,
            seed + 500,
            steps=steps,
            reset_optimizer=True,
            log_steps=log_spaced_steps(steps),
            save_final_checkpoint=bool(get_path(config, "run.save_checkpoints", True)),
        ),
        callback=callback,
    )
    training_wall = time.perf_counter() - wall_start
    checkpoints = _checkpoint_states(training, "selector_")

    final_conditions: dict[str, Any] = {}
    for name, batch in conditions.items():
        values = _h9_behavior(model, batch, config)
        final_conditions[name] = values
        metrics.extend(
            make_metric_records(
                values,
                hypothesis="h9",
                split=name,
                global_step=steps,
                stage="final_context_ood",
                examples_seen=training.samples_seen,
                condition=name,
            )
        )

    selector, clamp0, clamp1 = _h9_selector(model, normal, config)
    sensitivity, sensitivity_records = _h9_sensitivity_matrix(model, normal, config)
    metrics.extend(sensitivity_records)
    calibrations, calibration_checkpoints = _h9_single_rule_controls(
        config, seed, train, normal, metrics
    )
    checkpoints.update(calibration_checkpoints)
    mastery = acquisition_time(
        [int(point["step"]) for point in trajectory],
        [float(point["target_accuracy"]) for point in trajectory],
        threshold=_float(config, "h9.mastery_threshold", 0.9),
        persistence=_integer(config, "evaluation.persistence", 2),
    )
    selector_event = acquisition_time(
        [int(point["step"]) for point in trajectory],
        [float(point["strict_context_switching"]) for point in trajectory],
        threshold=_float(config, "h9.selector_threshold", 0.9),
        persistence=_integer(config, "evaluation.persistence", 2),
    )
    final = {
        **selector,
        **final_conditions,
        "conditions": final_conditions,
        # Direct aliases are convenient for flat analysis and keep the normal
        # condition visible without descending through a condition mapping.
        "normal_target_accuracy": float(final_conditions["normal"]["target_accuracy"]),
        "proxy_sensitivity_matrix": sensitivity,
        "single_rule_calibration": calibrations,
        "single_rule_P0": calibrations.get("P0"),
        "single_rule_P1": calibrations.get("P1"),
        "single_rule_controls_passed": (
            bool(calibrations.get("all_mastered", False))
            if bool(calibrations.get("enabled", False))
            else None
        ),
        "selector_gate_passed": bool(
            selector["strict_context_switching"]
            >= _float(config, "h9.selector_threshold", 0.9)
        ),
    }
    summary = {
        "hypothesis": "h9",
        "seed": seed,
        "condition": "context_selector",
        "final": final,
        "model": {
            **dict(model_report),
            "single_rule_controls": {
                key: value.get("model")
                for key, value in calibrations.items()
                if isinstance(value, Mapping) and "model" in value
            },
        },
        "events": {
            "task_mastery": _event_payload(mastery),
            "selector_mastery": _event_payload(selector_event),
            **mastery.as_dict("task_mastery"),
            **selector_event.as_dict("selector_mastery"),
        },
        "data": {
            "n_train": len(train),
            "n_eval": len(normal),
            "context_balance": float(np.mean(np.asarray(train.latents["E"]) == 1)),
            "requested_context_balance": _float(config, "h9.context_balance", 0.5),
            "proxy0_degree": int(train.metadata["proxy0_degree"]),
            "proxy1_degree": int(train.metadata["proxy1_degree"]),
            "evaluation_conditions": list(conditions),
            "paired_clamps_share_rows": True,
        },
        "training": {
            "steps": training.optimizer_steps,
            "examples_seen": training.samples_seen,
            "wall_seconds": training_wall,
        },
    }
    predictions: list[dict[str, Any]] = []
    if bool(get_path(config, "evaluation.save_predictions", True)):
        for name, batch in conditions.items():
            predictions.extend(_h9_prediction_records(model, batch, config, split=name))
        predictions.extend(
            _h9_prediction_records(model, clamp0, config, split="paired_clamps", intervention="clamp_C0")
        )
        predictions.extend(
            _h9_prediction_records(model, clamp1, config, split="paired_clamps", intervention="clamp_C1")
        )
        for context, clamped in ((0, clamp0), (1, clamp1)):
            for proxy in (0, 1):
                predictions.extend(
                    _h9_prediction_records(
                        model,
                        _h9_flip_proxy(clamped, proxy),
                        config,
                        split=f"clamp_C{context}",
                        intervention=f"flip_P{proxy}",
                    )
                )
    return ProtocolResult(
        model=model,
        summary=summary,
        metrics=metrics,
        predictions=predictions,
        checkpoints=checkpoints,
        evaluation_batch=normal,
    )


RUNNERS: dict[str, Callable[[Mapping[str, Any], int], ProtocolResult]] = {
    "h7": run_h7,
    "h8": run_h8,
    "h9": run_h9,
}


__all__ = ["RUNNERS", "run_h7", "run_h8", "run_h9"]
