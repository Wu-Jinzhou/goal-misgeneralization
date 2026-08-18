"""Identical-evidence order protocol (E18/H15).

The only treatment difference is the temporal order of two diagnostic evidence
streams.  Rows, atomic minibatches, presentations, model initialization, common
prefix, optimizer resets, and common washout are paired exactly across arms.
"""

from __future__ import annotations

import math
import time
from collections.abc import Mapping, Sequence
from itertools import pairwise
from typing import Any

import numpy as np
import torch

from .competing import (
    competing_causal_flip_batches,
    competing_rule_agreements,
    decode_competing_rules,
    make_competing_factorial_dataset,
)
from .config import get_path
from .data import SemanticBatch
from .evidence_order import (
    ORDER_SCHEDULES,
    AtomicEvidencePlan,
    audit_atomic_evidence_plan,
    audit_evidence_strata,
    make_atomic_evidence_plan,
    make_identical_evidence_dataset,
)
from .handoff import pure_control, semantic_batch_digest, stable_state_digest
from .models import GoalMLP
from .multigoal_dynamics import (
    directional_causal_effects,
    evaluate_competing_probes,
    factorial_behavioral_structure,
)
from .protocols import (
    ProtocolResult,
    batch_for_training,
    build_model,
    make_metric_records,
    predict_logits,
    resolve_device,
    seed_everything,
)
from .training import SFTConfig, TrainingResult, TrainingSnapshot, train_clean_sft


class AtomicBatchSource:
    """Serve a pre-registered matrix of row indices without RNG consumption."""

    def __init__(
        self,
        batch: SemanticBatch,
        indices: np.ndarray,
        config: Mapping[str, Any],
    ) -> None:
        matrix = np.asarray(indices, dtype=np.int64)
        if matrix.ndim != 2 or matrix.shape[0] < 1 or matrix.shape[1] < 1:
            raise ValueError("atomic source indices must have shape [steps, batch_size]")
        if np.any(matrix < 0) or np.any(matrix >= len(batch)):
            raise ValueError("atomic source contains an out-of-range row index")
        training = batch_for_training(batch, config)
        self.x = training[0]
        self.y = training[1]
        self.indices = torch.as_tensor(np.array(matrix, copy=True), dtype=torch.long)
        self.batch_size = matrix.shape[1]

    @property
    def steps(self) -> int:
        return len(self.indices)

    def sample_batch(self, batch_size: int, step: int) -> tuple[torch.Tensor, torch.Tensor]:
        if batch_size != self.batch_size:
            raise ValueError(
                f"atomic source requires batch_size={self.batch_size}, got {batch_size}"
            )
        if isinstance(step, bool) or not 1 <= step <= self.steps:
            raise IndexError(f"atomic source step must be in [1, {self.steps}]")
        selected = self.indices[step - 1]
        return self.x[selected], self.y[selected]


def _integer(config: Mapping[str, Any], path: str, default: int) -> int:
    value = get_path(config, path, default)
    if isinstance(value, bool) or int(value) != value:
        raise ValueError(f"{path} must be an integer")
    return int(value)


def _float(config: Mapping[str, Any], path: str, default: float) -> float:
    result = float(get_path(config, path, default))
    if not math.isfinite(result):
        raise ValueError(f"{path} must be finite")
    return result


def _integer_sequence(config: Mapping[str, Any], path: str) -> tuple[int, ...]:
    value = get_path(config, path)
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError(f"{path} must be an integer sequence")
    result = tuple(int(item) for item in value)
    if any(isinstance(item, bool) or int(item) != item for item in value):
        raise ValueError(f"{path} must be an integer sequence")
    return result


def _flatten_numeric(value: Any, prefix: str = "") -> dict[str, float | int]:
    result: dict[str, float | int] = {}
    if isinstance(value, Mapping):
        for key, item in value.items():
            name = f"{prefix}__{key}" if prefix else str(key)
            result.update(_flatten_numeric(item, name))
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        for index, item in enumerate(value):
            item_key = (
                str(item.get("key"))
                if isinstance(item, Mapping) and isinstance(item.get("key"), str)
                else str(index)
            )
            name = f"{prefix}__{item_key}" if prefix else item_key
            result.update(_flatten_numeric(item, name))
    elif isinstance(value, (int, float, np.number)) and not isinstance(
        value, (bool, np.bool_)
    ):
        result[prefix] = float(value)
    return result


def _append_records(
    records: list[dict[str, Any]],
    values: Mapping[str, float | int],
    *,
    split: str,
    global_step: int,
    local_step: int,
    examples_seen: int,
    condition: str,
    stage: str,
    intervention: str = "none",
    n: int = 0,
) -> None:
    payload = dict(values)
    payload["n"] = int(payload.get("n", n))
    records.extend(
        make_metric_records(
            payload,
            hypothesis="h15",
            split=split,
            global_step=global_step,
            stage=stage,
            stage_step=local_step,
            examples_seen=examples_seen,
            condition=condition,
            intervention=intervention,
        )
    )


def _sft_config(
    config: Mapping[str, Any],
    *,
    seed: int,
    steps: int,
    log_steps: Sequence[int],
    reset_optimizer: bool,
) -> SFTConfig:
    return SFTConfig(
        steps=steps,
        batch_size=_integer(config, "train.batch_size", 250),
        learning_rate=_float(config, "train.learning_rate", 3e-3),
        weight_decay=_float(config, "train.weight_decay", 0.0),
        optimizer=str(get_path(config, "train.optimizer", "adamw")),  # type: ignore[arg-type]
        momentum=_float(config, "train.momentum", 0.0),
        seed=seed,
        device=torch.device("cpu"),
        deterministic=bool(get_path(config, "train.deterministic", True)),
        shuffle=False,
        gradient_clip_norm=get_path(config, "train.grad_clip", 1.0),
        log_steps=tuple(log_steps),
        checkpoint_steps=(),
        save_checkpoints=False,
        reset_model=False,
        reset_optimizer=reset_optimizer,
    )


def _new_optimizer(
    model: GoalMLP, config: Mapping[str, Any]
) -> torch.optim.Optimizer:
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    name = str(get_path(config, "train.optimizer", "adamw")).lower()
    learning_rate = _float(config, "train.learning_rate", 3e-3)
    weight_decay = _float(config, "train.weight_decay", 0.0)
    if name == "adamw":
        return torch.optim.AdamW(
            parameters, lr=learning_rate, weight_decay=weight_decay
        )
    if name == "adam":
        return torch.optim.Adam(parameters, lr=learning_rate, weight_decay=weight_decay)
    if name == "sgd":
        return torch.optim.SGD(
            parameters,
            lr=learning_rate,
            weight_decay=weight_decay,
            momentum=_float(config, "train.momentum", 0.0),
        )
    raise ValueError("H15 train.optimizer must be adamw, adam, or sgd")


def _optimizer_entry_audit(optimizer: torch.optim.Optimizer) -> dict[str, Any]:
    steps: list[int] = []
    for state in optimizer.state.values():
        if "step" not in state:
            continue
        raw = state["step"]
        value = int(raw.detach().cpu().item()) if torch.is_tensor(raw) else int(raw)
        steps.append(value)
    return {
        "actual_state_digest": stable_state_digest(optimizer.state_dict()),
        "state_entry_count": len(optimizer.state),
        "adam_step_min": min(steps) if steps else None,
        "adam_step_max": max(steps) if steps else None,
        "adam_step_entry_count": len(steps),
        "param_group_count": len(optimizer.param_groups),
        "optimized_parameter_count": sum(
            len(group.get("params", ())) for group in optimizer.param_groups
        ),
    }


def _optimizer_records(
    training: TrainingResult,
    records: list[dict[str, Any]],
    *,
    stage: str,
    global_offset: int,
    example_offset: int,
    condition: str,
    n: int,
) -> None:
    for record in training.history:
        _append_records(
            records,
            {
                "loss": record.loss,
                "primary_loss": record.primary_loss,
                "train_batch_accuracy": record.metrics.get(
                    "train_accuracy", float("nan")
                ),
                "optimizer_steps": record.optimizer_steps,
            },
            split="train_minibatch",
            global_step=global_offset + record.step,
            local_step=record.step,
            examples_seen=example_offset + record.samples_seen,
            condition=condition,
            stage=stage,
            n=n,
        )


def _compact_snapshot(
    model: GoalMLP,
    *,
    probe_train: SemanticBatch,
    probe_eval: SemanticBatch,
    config: Mapping[str, Any],
    seed: int,
    probe_alpha: float,
    max_k_y: int,
    pure_threshold: float,
    pure_margin: float,
    records: list[dict[str, Any]],
    condition: str,
    stage: str,
    local_step: int,
    global_step: int,
    examples_seen: int,
) -> dict[str, Any]:
    logits = predict_logits(model, probe_eval, config)
    candidates = competing_rule_agreements(logits, probe_eval)
    rules = decode_competing_rules(probe_eval)
    margin_values = np.asarray(logits[:, 1] - logits[:, 0], dtype=np.float64)
    strength: dict[str, Any] = {
        "mean_absolute_logit_margin": float(np.mean(np.abs(margin_values)))
    }
    for goal, values in (
        ("P", rules["P"]),
        ("Q", rules["Q"]),
        ("Y", rules["Y_code"]),
    ):
        signed = np.asarray(values, dtype=np.float64) * margin_values
        probability = np.empty_like(signed)
        positive = signed >= 0.0
        probability[positive] = 1.0 / (1.0 + np.exp(-signed[positive]))
        exponential = np.exp(signed[~positive])
        probability[~positive] = exponential / (1.0 + exponential)
        strength[f"{goal}_mean_signed_logit_margin"] = float(np.mean(signed))
        strength[f"{goal}_soft_agreement"] = float(np.mean(probability))
    structure = factorial_behavioral_structure(logits, probe_eval)
    probes = evaluate_competing_probes(
        model,
        probe_train,
        probe_eval,
        seed=70_000_000 + seed,
        alpha=probe_alpha,
        max_k=max_k_y,
        include_state=True,
        include_permuted=True,
    )
    changed_logits = {
        name: predict_logits(model, changed, config)
        for name, changed in competing_causal_flip_batches(probe_eval).items()
    }
    causal = directional_causal_effects(logits, changed_logits, probe_eval)

    behavior = {
        "P": float(candidates["rho_p"]),
        "Q": float(candidates["rho_q"]),
        "Y": float(candidates["rho_y_code"]),
    }
    family_means = causal["family_means"]
    causal_scores = {
        goal: float(family_means[goal]["causal_score"]) for goal in ("P", "Q", "Y")
    }
    causal_probability_scores = {
        goal: float(family_means[goal]["causal_prob_score"])
        for goal in ("P", "Q", "Y")
    }
    probe_heldout = {
        layer: {
            name: float(value)
            for name, value in summary["heldout_accuracy"].items()
        }
        for layer, summary in probes["representations"].items()
    }
    final_hidden = probe_heldout["final_hidden"]
    selective_probe = {
        goal: float(final_hidden[goal] - final_hidden["truth_table_control"])
        for goal in ("P", "Q", "Y")
    }
    m_y = 0.5 * (
        behavior["Y"] - max(behavior["P"], behavior["Q"])
        + causal_scores["Y"] - max(causal_scores["P"], causal_scores["Q"])
    )
    pure = {
        goal: pure_control(
            behavior,
            causal_scores,
            goal,
            threshold=pure_threshold,
            margin=pure_margin,
        )
        for goal in ("P", "Q", "Y")
    }
    pure_goals = [goal for goal, value in pure.items() if value]

    _append_records(
        records,
        {**candidates, "m_y": m_y},
        split="factorial_eval",
        global_step=global_step,
        local_step=local_step,
        examples_seen=examples_seen,
        condition=condition,
        stage=f"{stage}_behavior",
        n=len(probe_eval),
    )
    _append_records(
        records,
        strength,
        split="factorial_eval",
        global_step=global_step,
        local_step=local_step,
        examples_seen=examples_seen,
        condition=condition,
        stage=f"{stage}_strength",
        n=len(probe_eval),
    )
    _append_records(
        records,
        _flatten_numeric(structure),
        split="factorial_eval",
        global_step=global_step,
        local_step=local_step,
        examples_seen=examples_seen,
        condition=condition,
        stage=f"{stage}_truth_table",
        n=len(probe_eval),
    )
    _append_records(
        records,
        _flatten_numeric(probes),
        split="factorial_probe",
        global_step=global_step,
        local_step=local_step,
        examples_seen=examples_seen,
        condition=condition,
        stage=f"{stage}_probe",
        n=len(probe_eval),
    )
    for section in ("per_intervention", "family_means"):
        for intervention, values in causal[section].items():
            _append_records(
                records,
                _flatten_numeric(values),
                split="factorial_eval",
                global_step=global_step,
                local_step=local_step,
                examples_seen=examples_seen,
                condition=condition,
                stage=f"{stage}_causal",
                intervention=str(intervention),
                n=len(probe_eval),
            )
    return {
        "local_step": local_step,
        "global_step": global_step,
        "examples_seen": examples_seen,
        "behavior": behavior,
        "target_accuracy": float(candidates["target_accuracy"]),
        "causal": causal_scores,
        "causal_probability": causal_probability_scores,
        "m_y": float(m_y),
        "pure_control": pure,
        "pure_goal": pure_goals[0] if len(pure_goals) == 1 else None,
        "strength": strength,
        "causal_details": causal,
        "probe_heldout_accuracy": probe_heldout,
        "selective_final_hidden_probe": selective_probe,
        "truth_table": structure,
        "boolean_signature": structure["boolean_signature"],
    }


def _normalized_auc(
    snapshots: Mapping[int, Mapping[str, Any]], *, horizon: int
) -> float:
    if 0 not in snapshots or horizon not in snapshots:
        raise ValueError("m_y AUC requires directly observed endpoints 0 and horizon")
    steps = sorted(step for step in snapshots if 0 <= step <= horizon)
    if steps[0] != 0 or steps[-1] != horizon:
        raise ValueError("m_y AUC checkpoints do not span the registered horizon")
    area = 0.0
    for left, right in pairwise(steps):
        area += (right - left) * (
            float(snapshots[left]["m_y"]) + float(snapshots[right]["m_y"])
        ) / 2.0
    return area / horizon


def _train_atomic(
    model: GoalMLP,
    batch: SemanticBatch,
    indices: np.ndarray,
    config: Mapping[str, Any],
    *,
    seed: int,
    log_steps: Sequence[int],
    optimizer: torch.optim.Optimizer | None = None,
    reset_optimizer: bool,
    callback: Any = None,
) -> TrainingResult:
    source = AtomicBatchSource(batch, indices, config)
    return train_clean_sft(
        model,
        source,
        _sft_config(
            config,
            seed=seed,
            steps=source.steps,
            log_steps=log_steps,
            reset_optimizer=reset_optimizer,
        ),
        optimizer=optimizer,
        callback=callback,
    )


def _plan_summary(plan: AtomicEvidencePlan) -> dict[str, Any]:
    return {
        "schedule": plan.schedule,
        "prefix_steps": plan.prefix_steps,
        "block_steps": plan.block_steps,
        "diagnostics_end": plan.diagnostics_end,
        "washout_steps": plan.washout_steps,
        "total_steps": plan.total_steps,
        "batch_size": plan.batch_size,
        "component_digests": dict(plan.component_digests),
        "ordered_digest": plan.ordered_digest,
        "row_exposure_digest": plan.row_exposure_digest,
        "atomic_batch_multiset_digest": plan.atomic_batch_multiset_digest,
    }


def run_h15(config: Mapping[str, Any], seed: int) -> ProtocolResult:
    """Run one frozen identical-evidence order schedule."""

    if resolve_device(str(get_path(config, "run.device", "cpu"))).type != "cpu":
        raise ValueError("H15 exact replay is CPU-only")
    seed_everything(seed)
    schedule = str(get_path(config, "h15.schedule", "b_then_d"))
    if schedule not in ORDER_SCHEDULES:
        raise ValueError(f"h15.schedule must be one of {ORDER_SCHEDULES}")
    condition = f"identical_evidence_order:{schedule}"

    n_train = _integer(config, "data.n_train", 10_000)
    batch_size = _integer(config, "train.batch_size", 250)
    q_p = _float(config, "h15.q_p", 0.90)
    q_q = _float(config, "h15.q_q", 0.95)
    k_q = _integer(config, "h15.k_q", 2)
    k_y = _integer(config, "h15.k_y", 3)
    max_k_q = _integer(config, "h15.max_k_q", 3)
    max_k_y = _integer(config, "h15.max_k_y", 5)
    state_dim = _integer(config, "data.state_dim", 8)
    prefix_repetitions = _integer(config, "h15.prefix_a_repetitions", 90)
    washout_repetitions = _integer(config, "h15.washout_a_repetitions", 10)
    diagnostic_repetitions = _integer(config, "h15.diagnostic_repetitions", 100)
    expected_prefix_steps = _integer(config, "h15.prefix_steps", 3_240)
    expected_block_steps = _integer(config, "h15.block_steps", 200)
    expected_washout_steps = _integer(config, "h15.washout_steps", 360)
    expected_total_steps = _integer(config, "h15.total_steps", 4_000)
    second_checkpoints = _integer_sequence(config, "h15.second_block_checkpoints")
    washout_checkpoints = _integer_sequence(config, "h15.washout_checkpoints")
    auc_horizon = _integer(config, "h15.auc_horizon", 128)
    probe_train_n = _integer(config, "h15.probe_train_n", 2_048)
    probe_eval_n = _integer(config, "h15.probe_eval_n", 4_096)
    probe_alpha = _float(config, "h15.probe_ridge", 1e-3)
    control_seed = _integer(config, "h15.truth_table_control_seed", 1_500_450_271)
    threshold = _float(config, "h15.pure_threshold", 0.90)
    margin = _float(config, "h15.pure_margin", 0.10)
    if 0 not in washout_checkpoints or auc_horizon not in washout_checkpoints:
        raise ValueError("h15 washout checkpoints must directly include 0 and auc_horizon")
    if second_checkpoints[-1] != expected_block_steps:
        raise ValueError("h15 second-block checkpoints must end at block_steps")

    train_batch = make_identical_evidence_dataset(
        n_train,
        q_p=q_p,
        q_q=q_q,
        k_q=k_q,
        k_y=k_y,
        seed=seed,
        max_k_q=max_k_q,
        max_k_y=max_k_y,
        state_dim=state_dim,
    )
    data_audit = audit_evidence_strata(train_batch)
    plan_seed = 91_000_000 + seed
    plan = make_atomic_evidence_plan(
        train_batch,
        schedule=schedule,  # type: ignore[arg-type]
        prefix_a_repetitions=prefix_repetitions,
        washout_a_repetitions=washout_repetitions,
        diagnostic_repetitions=diagnostic_repetitions,
        batch_size=batch_size,
        seed=plan_seed,
    )
    plan_audit = audit_atomic_evidence_plan(
        train_batch,
        plan,
        prefix_a_repetitions=prefix_repetitions,
        washout_a_repetitions=washout_repetitions,
        diagnostic_repetitions=diagnostic_repetitions,
    )
    actual_steps = (
        plan.prefix_steps,
        plan.block_steps,
        plan.washout_steps,
        plan.total_steps,
    )
    expected_steps = (
        expected_prefix_steps,
        expected_block_steps,
        expected_washout_steps,
        expected_total_steps,
    )
    if actual_steps != expected_steps:
        raise RuntimeError(f"H15 atomic plan steps {actual_steps} != frozen {expected_steps}")

    probe_train = make_competing_factorial_dataset(
        probe_train_n,
        k_q=k_q,
        k_y=k_y,
        seed=50_000_000 + seed,
        control_seed=control_seed,
        max_k_q=max_k_q,
        max_k_y=max_k_y,
        state_dim=state_dim,
        split="factorial_probe_train",
        id_offset=2_000_000_000,
    )
    probe_eval = make_competing_factorial_dataset(
        probe_eval_n,
        k_q=k_q,
        k_y=k_y,
        seed=60_000_000 + seed,
        control_seed=control_seed,
        max_k_q=max_k_q,
        max_k_y=max_k_y,
        state_dim=state_dim,
        split="factorial_probe_eval",
        id_offset=2_100_000_000,
    )
    feature_names = train_batch.feature_names(max_k=max_k_y, include_state=True)
    for name, panel in (("probe_train", probe_train), ("probe_eval", probe_eval)):
        if panel.feature_names(max_k=max_k_y, include_state=True) != feature_names:
            raise RuntimeError(f"H15 {name} interface differs from training data")

    records: list[dict[str, Any]] = []
    started = time.perf_counter()
    built_model, model_report = build_model(train_batch, config, seed)
    if not isinstance(built_model, GoalMLP):
        raise TypeError("H15 requires an unwrapped GoalMLP")
    model = built_model
    initial_model_hash = stable_state_digest(model.state_dict())
    prefix_snapshots: dict[int, dict[str, Any]] = {}

    def prefix_callback(snapshot: TrainingSnapshot) -> None:
        if not isinstance(snapshot.model, GoalMLP):
            raise TypeError("H15 prefix observer requires GoalMLP")
        prefix_snapshots[snapshot.step] = _compact_snapshot(
            snapshot.model,
            probe_train=probe_train,
            probe_eval=probe_eval,
            config=config,
            seed=seed,
            probe_alpha=probe_alpha,
            max_k_y=max_k_y,
            pure_threshold=threshold,
            pure_margin=margin,
            records=records,
            condition=condition,
            stage="prefix",
            local_step=snapshot.step,
            global_step=snapshot.step,
            examples_seen=snapshot.record.samples_seen,
        )

    prefix_training = _train_atomic(
        model,
        train_batch,
        plan.phase("prefix"),
        config,
        seed=92_000_000 + seed,
        log_steps=(plan.prefix_steps,),
        reset_optimizer=True,
        callback=prefix_callback,
    )
    if set(prefix_snapshots) != {plan.prefix_steps}:
        raise RuntimeError("H15 failed to directly observe the prefix boundary")
    _optimizer_records(
        prefix_training,
        records,
        stage="prefix_optimization",
        global_offset=0,
        example_offset=0,
        condition=condition,
        n=n_train,
    )

    replay_built, replay_report = build_model(train_batch, config, seed)
    if not isinstance(replay_built, GoalMLP):
        raise TypeError("H15 replay requires an unwrapped GoalMLP")
    replay_model = replay_built
    replay_initial_model_hash = stable_state_digest(replay_model.state_dict())
    replay_training = _train_atomic(
        replay_model,
        train_batch,
        plan.phase("prefix"),
        config,
        seed=92_000_000 + seed,
        log_steps=(plan.prefix_steps,),
        reset_optimizer=True,
    )
    training_batch_hash = semantic_batch_digest(train_batch)
    prefix_stream_hash = plan.component_digests["A_prefix"]
    replay = {
        "hashes": {
            "observed_initial_model": initial_model_hash,
            "unobserved_initial_model": replay_initial_model_hash,
            "observed_final_model": stable_state_digest(prefix_training.final_model_state),
            "unobserved_final_model": stable_state_digest(replay_training.final_model_state),
            "observed_final_optimizer": stable_state_digest(
                prefix_training.optimizer_state
            ),
            "unobserved_final_optimizer": stable_state_digest(
                replay_training.optimizer_state
            ),
            "observed_training_batch": training_batch_hash,
            "unobserved_training_batch": training_batch_hash,
            "observed_prefix_atomic_stream": prefix_stream_hash,
            "unobserved_prefix_atomic_stream": prefix_stream_hash,
        },
        "checks": {
            "model_reports_equal": model_report == replay_report,
            "initial_models_equal": initial_model_hash == replay_initial_model_hash,
            "final_models_equal": stable_state_digest(prefix_training.final_model_state)
            == stable_state_digest(replay_training.final_model_state),
            "final_optimizers_equal": stable_state_digest(prefix_training.optimizer_state)
            == stable_state_digest(replay_training.optimizer_state),
            "samples_seen_equal": prefix_training.samples_seen
            == replay_training.samples_seen,
            "optimizer_steps_equal": prefix_training.optimizer_steps
            == replay_training.optimizer_steps,
            "training_batches_equal": True,
            "ordered_streams_equal": True,
        },
        "observed_samples_seen": prefix_training.samples_seen,
        "unobserved_samples_seen": replay_training.samples_seen,
    }
    if not all(replay["checks"].values()):
        raise RuntimeError(f"H15 prefix failed deterministic replay: {replay['checks']}")

    prefix_model_hash = stable_state_digest(model.state_dict())
    diagnostic_optimizer = _new_optimizer(model, config)
    first_reset = _optimizer_entry_audit(diagnostic_optimizer)
    if first_reset["state_entry_count"] != 0:
        raise RuntimeError("H15 first reset did not install an empty optimizer")

    diagnostic_snapshots: dict[int, dict[str, Any]] = {}
    diagnostic_log_steps = tuple(
        sorted({plan.block_steps, *(plan.block_steps + step for step in second_checkpoints)})
    )

    def diagnostic_callback(snapshot: TrainingSnapshot) -> None:
        if not isinstance(snapshot.model, GoalMLP):
            raise TypeError("H15 diagnostic observer requires GoalMLP")
        offset = max(0, snapshot.step - plan.block_steps)
        diagnostic_snapshots[offset] = _compact_snapshot(
            snapshot.model,
            probe_train=probe_train,
            probe_eval=probe_eval,
            config=config,
            seed=seed,
            probe_alpha=probe_alpha,
            max_k_y=max_k_y,
            pure_threshold=threshold,
            pure_margin=margin,
            records=records,
            condition=condition,
            stage="diagnostic",
            local_step=offset,
            global_step=plan.prefix_steps + snapshot.step,
            examples_seen=prefix_training.samples_seen + snapshot.record.samples_seen,
        )

    diagnostic_training = _train_atomic(
        model,
        train_batch,
        plan.phase("diagnostics"),
        config,
        seed=93_000_000 + seed,
        log_steps=diagnostic_log_steps,
        optimizer=diagnostic_optimizer,
        reset_optimizer=False,
        callback=diagnostic_callback,
    )
    if set(diagnostic_snapshots) != {0, *second_checkpoints}:
        raise RuntimeError("H15 failed to observe every diagnostic checkpoint")
    _optimizer_records(
        diagnostic_training,
        records,
        stage="diagnostic_optimization",
        global_offset=plan.prefix_steps,
        example_offset=prefix_training.samples_seen,
        condition=condition,
        n=n_train,
    )
    post_diagnostic_model_hash = stable_state_digest(model.state_dict())
    post_diagnostic_optimizer_hash = stable_state_digest(
        diagnostic_training.optimizer_state
    )

    washout_optimizer = _new_optimizer(model, config)
    second_reset = _optimizer_entry_audit(washout_optimizer)
    if second_reset["state_entry_count"] != 0:
        raise RuntimeError("H15 second reset did not install an empty optimizer")
    if stable_state_digest(model.state_dict()) != post_diagnostic_model_hash:
        raise RuntimeError("H15 optimizer reset unexpectedly changed model weights")

    washout_snapshots: dict[int, dict[str, Any]] = {}
    diagnostic_examples = diagnostic_training.samples_seen
    washout_snapshots[0] = _compact_snapshot(
        model,
        probe_train=probe_train,
        probe_eval=probe_eval,
        config=config,
        seed=seed,
        probe_alpha=probe_alpha,
        max_k_y=max_k_y,
        pure_threshold=threshold,
        pure_margin=margin,
        records=records,
        condition=condition,
        stage="washout",
        local_step=0,
        global_step=plan.diagnostics_end,
        examples_seen=prefix_training.samples_seen + diagnostic_examples,
    )
    positive_washout_checkpoints = tuple(step for step in washout_checkpoints if step > 0)

    def washout_callback(snapshot: TrainingSnapshot) -> None:
        if not isinstance(snapshot.model, GoalMLP):
            raise TypeError("H15 washout observer requires GoalMLP")
        washout_snapshots[snapshot.step] = _compact_snapshot(
            snapshot.model,
            probe_train=probe_train,
            probe_eval=probe_eval,
            config=config,
            seed=seed,
            probe_alpha=probe_alpha,
            max_k_y=max_k_y,
            pure_threshold=threshold,
            pure_margin=margin,
            records=records,
            condition=condition,
            stage="washout",
            local_step=snapshot.step,
            global_step=plan.diagnostics_end + snapshot.step,
            examples_seen=(
                prefix_training.samples_seen
                + diagnostic_examples
                + snapshot.record.samples_seen
            ),
        )

    washout_training = _train_atomic(
        model,
        train_batch,
        plan.phase("washout"),
        config,
        seed=94_000_000 + seed,
        log_steps=positive_washout_checkpoints,
        optimizer=washout_optimizer,
        reset_optimizer=False,
        callback=washout_callback,
    )
    if set(washout_snapshots) != set(washout_checkpoints):
        raise RuntimeError("H15 failed to observe every washout checkpoint")
    _optimizer_records(
        washout_training,
        records,
        stage="washout_optimization",
        global_offset=plan.diagnostics_end,
        example_offset=prefix_training.samples_seen + diagnostic_examples,
        condition=condition,
        n=n_train,
    )

    expected_phase_samples = {
        "prefix": plan.prefix_steps * batch_size,
        "diagnostic": 2 * plan.block_steps * batch_size,
        "washout": plan.washout_steps * batch_size,
    }
    actual_phase_samples = {
        "prefix": prefix_training.samples_seen,
        "diagnostic": diagnostic_training.samples_seen,
        "washout": washout_training.samples_seen,
    }
    if actual_phase_samples != expected_phase_samples:
        raise RuntimeError(
            f"H15 observed a short or missing atomic batch: {actual_phase_samples}"
        )
    prefix_snapshot = prefix_snapshots[plan.prefix_steps]
    first_half_snapshot = diagnostic_snapshots[0]
    post_diagnostic_snapshot = diagnostic_snapshots[plan.block_steps]
    primary_auc = _normalized_auc(washout_snapshots, horizon=auc_horizon)
    final = washout_snapshots[plan.washout_steps]
    wall_seconds = time.perf_counter() - started

    summary = {
        "hypothesis": "h15",
        "seed": seed,
        "condition": condition,
        "schedule": schedule,
        "design_status": "adaptive_posthoc_identical_evidence_order_frozen_pre_outcome",
        "prefix_snapshot": prefix_snapshot,
        "first_half_snapshot": first_half_snapshot,
        "diagnostic_snapshots": {
            str(step): value for step, value in sorted(diagnostic_snapshots.items())
        },
        "post_diagnostic_snapshot": post_diagnostic_snapshot,
        "washout_snapshots": {
            str(step): value for step, value in sorted(washout_snapshots.items())
        },
        "final": final,
        "outcomes": {
            "primary_auc_m_y": primary_auc,
            "auc_horizon": auc_horizon,
            "terminal_m_y": float(final["m_y"]),
            "prefix_pure_goal": prefix_snapshot["pure_goal"],
            "first_half_pure_goal": first_half_snapshot["pure_goal"],
            "post_diagnostic_pure_goal": post_diagnostic_snapshot["pure_goal"],
            "terminal_pure_goal": final["pure_goal"],
        },
        "data": {
            "n_train": n_train,
            "q_p": q_p,
            "q_q": q_q,
            "k_q": k_q,
            "k_y": k_y,
            "max_k_q": max_k_q,
            "max_k_y": max_k_y,
            "state_dim": state_dim,
            "training_batch_digest": semantic_batch_digest(train_batch),
            "strata_audit": data_audit,
            "probe_train_digest": semantic_batch_digest(probe_train),
            "probe_eval_digest": semantic_batch_digest(probe_eval),
            "probe_splits_disjoint": not bool(
                set(np.asarray(probe_train.sample_id).tolist())
                & set(np.asarray(probe_eval.sample_id).tolist())
            ),
        },
        "plan": _plan_summary(plan),
        "plan_audit": plan_audit,
        "replay": replay,
        "resets": {
            "prefix_to_diagnostics": {
                "boundary_global_step": plan.prefix_steps,
                "semantic_reset_implemented_by_fresh_object": True,
                "low_level_reset_optimizer_flag": False,
                **first_reset,
            },
            "diagnostics_to_washout": {
                "boundary_global_step": plan.diagnostics_end,
                "semantic_reset_implemented_by_fresh_object": True,
                "low_level_reset_optimizer_flag": False,
                **second_reset,
            },
        },
        "hashes": {
            "initial_model": initial_model_hash,
            "prefix_final_model": prefix_model_hash,
            "prefix_final_optimizer": stable_state_digest(
                prefix_training.optimizer_state
            ),
            "first_reset_optimizer": first_reset["actual_state_digest"],
            "post_diagnostic_model": post_diagnostic_model_hash,
            "post_diagnostic_optimizer": post_diagnostic_optimizer_hash,
            "second_reset_optimizer": second_reset["actual_state_digest"],
            "final_model": stable_state_digest(washout_training.final_model_state),
            "final_optimizer": stable_state_digest(washout_training.optimizer_state),
        },
        "model": model_report,
        "measurement": {
            "probe_train_n": probe_train_n,
            "probe_eval_n": probe_eval_n,
            "probe_ridge": probe_alpha,
            "truth_table_control_seed": control_seed,
            "second_block_checkpoints": list(second_checkpoints),
            "washout_checkpoints": list(washout_checkpoints),
            "direct_auc_endpoints": [0, auc_horizon],
            "pure_threshold": threshold,
            "pure_margin": margin,
            "late_stability_steps": list(
                _integer_sequence(config, "h15.late_stability_steps")
            ),
            "late_stability_tolerance": _float(
                config, "h15.late_stability_tolerance", 0.02
            ),
            "primary_effect_threshold": _float(
                config, "h15.primary_effect_threshold", 0.10
            ),
            "equivalence_margin": _float(config, "h15.equivalence_margin", 0.05),
            "terminal_effect_threshold": _float(
                config, "h15.terminal_effect_threshold", 0.10
            ),
            "minimum_sign_count": _integer(
                config, "h15.minimum_sign_count", 15
            ),
            "m_y_definition": (
                "0.5*((rho_Y-max(rho_P,rho_Q))+(c_Y-max(c_P,c_Q)))"
            ),
            "directional_causal_normalization": (
                "(1 + E[g*(a-a_flip)/2]) / 2"
            ),
        },
        "training": {
            "batch_size": batch_size,
            "all_minibatches_full": True,
            "prefix_steps": prefix_training.optimizer_steps,
            "diagnostic_steps": diagnostic_training.optimizer_steps,
            "washout_steps": washout_training.optimizer_steps,
            "total_steps": (
                prefix_training.optimizer_steps
                + diagnostic_training.optimizer_steps
                + washout_training.optimizer_steps
            ),
            "phase_examples_seen": actual_phase_samples,
            "total_examples_seen": sum(actual_phase_samples.values()),
            "replay_examples_seen_excluded_from_treatment_total": replay_training.samples_seen,
            "wall_seconds": wall_seconds,
        },
    }
    return ProtocolResult(
        model=model,
        summary=summary,
        metrics=records,
        predictions=[],
        checkpoints={},
        evaluation_batch=probe_eval,
    )


RUNNERS = {"h15": run_h15}


__all__ = ["ORDER_SCHEDULES", "RUNNERS", "AtomicBatchSource", "run_h15"]
