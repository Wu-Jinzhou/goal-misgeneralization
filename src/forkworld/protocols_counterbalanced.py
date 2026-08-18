"""Counterbalanced identical-evidence order protocol (E20/H17).

The outcome-blind pilot constructs exactly one selected component and exits.
Full runs continue one model through all three component blocks in the selected
order and a common concordant washout.  Every block starts with a genuinely new
empty AdamW optimizer.  Measurements never enter the training stream, and the
public observer-free replay helper lets the strict analyzer independently
reconstruct all boundary and optimizer hashes without double-training artifacts.
"""

from __future__ import annotations

import hashlib
import json
import math
import time
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import numpy as np
import torch

from .competing import (
    competing_causal_flip_batches,
    decode_competing_rules,
)
from .config import get_path
from .counterbalanced_order import (
    GOALS,
    ORDER_SCHEDULES,
    audit_concordant_washout,
    audit_counterbalanced_atomic_plan,
    audit_counterbalanced_bundle,
    audit_counterbalanced_component,
    audit_counterbalanced_factorial_panel,
    make_atomic_component_stream,
    make_counterbalanced_atomic_plan,
    make_counterbalanced_bundle,
    make_counterbalanced_component,
    make_counterbalanced_factorial_panel,
)
from .data import SemanticBatch
from .handoff import pure_control, semantic_batch_digest, stable_state_digest
from .models import GoalMLP
from .multigoal_dynamics import (
    directional_causal_effects,
    extract_goal_mlp_representations,
    factorial_behavioral_structure,
    fit_affine_ridge_probe,
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

DESIGN_MEMO_SHA256 = "1386cb401aef8ed187389dc84aa210b09b564436bc16c9246bbd22fce257529a"
FOLD_DIGEST = "25d5e9584a2ea74d42d48a673645a3b34c1b76e75e8230bf5722985e66968ec7"
CONTROL_DIGEST = "c7a186308102e807e594fd76b3fc5e7ef1678314a95de067b6fd347b390e53d0"
CONTROL_BITS = "0110001001011011111001011000100101011110100110000010011010110101"
DATA_SEED = 171_000_001
STREAM_SEED = 171_000_002
PHASE_SEEDS = {
    "component_P": 171_000_101,
    "component_Q": 171_000_102,
    "component_Y": 171_000_103,
    "washout": 171_000_104,
}
FEATURE_NAMES = (
    "P",
    "P_present",
    "R_1",
    "R_2",
    "R_3",
    "Q_present",
    "Q_1",
    "Q_2",
)


class AtomicBatchSource:
    """Serve a frozen row-index matrix without consuming sampler randomness."""

    def __init__(
        self,
        batch: SemanticBatch,
        indices: np.ndarray,
        config: Mapping[str, Any],
    ) -> None:
        matrix = np.asarray(indices, dtype=np.int64)
        if matrix.ndim != 2 or matrix.shape[0] < 1 or matrix.shape[1] < 1:
            raise ValueError("H17 atomic indices must have shape [steps,batch_size]")
        if np.any(matrix < 0) or np.any(matrix >= len(batch)):
            raise ValueError("H17 atomic source contains an out-of-range row index")
        training = batch_for_training(batch, config, include_state=False)
        self.x = training[0]
        self.y = training[1]
        self.indices = torch.as_tensor(np.array(matrix, copy=True), dtype=torch.long)
        self.batch_size = int(matrix.shape[1])

    @property
    def steps(self) -> int:
        return len(self.indices)

    def sample_batch(self, batch_size: int, step: int) -> tuple[torch.Tensor, torch.Tensor]:
        if batch_size != self.batch_size:
            raise ValueError(
                f"H17 atomic source requires batch_size={self.batch_size}, got {batch_size}"
            )
        if isinstance(step, bool) or not 1 <= step <= self.steps:
            raise IndexError(f"H17 atomic source step must be in [1,{self.steps}]")
        selected = self.indices[step - 1]
        return self.x[selected], self.y[selected]


def _integer(config: Mapping[str, Any], path: str, default: int) -> int:
    value = get_path(config, path, default)
    if isinstance(value, bool) or int(value) != value:
        raise ValueError(f"{path} must be an integer")
    return int(value)


def _float(config: Mapping[str, Any], path: str, default: float) -> float:
    value = float(get_path(config, path, default))
    if not math.isfinite(value):
        raise ValueError(f"{path} must be finite")
    return value


def _integer_sequence(config: Mapping[str, Any], path: str) -> tuple[int, ...]:
    raw = get_path(config, path)
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
        raise ValueError(f"{path} must be an integer sequence")
    if any(isinstance(value, bool) or int(value) != value for value in raw):
        raise ValueError(f"{path} must be an integer sequence")
    return tuple(int(value) for value in raw)


def _flatten_numeric(value: Any, prefix: str = "") -> dict[str, float | int]:
    result: dict[str, float | int] = {}
    if isinstance(value, Mapping):
        for key, item in value.items():
            name = f"{prefix}__{key}" if prefix else str(key)
            result.update(_flatten_numeric(item, name))
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        for index, item in enumerate(value):
            name = f"{prefix}__{index}" if prefix else str(index)
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
            hypothesis="h17",
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
) -> SFTConfig:
    return SFTConfig(
        steps=steps,
        batch_size=_integer(config, "train.batch_size", 288),
        learning_rate=_float(config, "train.learning_rate", 3e-3),
        weight_decay=_float(config, "train.weight_decay", 0.0),
        optimizer="adamw",
        seed=seed,
        device=torch.device("cpu"),
        deterministic=bool(get_path(config, "train.deterministic", True)),
        shuffle=False,
        gradient_clip_norm=get_path(config, "train.grad_clip", 1.0),
        label_smoothing=_float(config, "train.label_smoothing", 0.0),
        log_steps=tuple(log_steps),
        checkpoint_steps=(),
        save_checkpoints=False,
        reset_model=False,
        reset_optimizer=False,
    )


def _new_optimizer(model: GoalMLP, config: Mapping[str, Any]) -> torch.optim.AdamW:
    return torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=_float(config, "train.learning_rate", 3e-3),
        weight_decay=_float(config, "train.weight_decay", 0.0),
    )


def _optimizer_audit(optimizer: torch.optim.Optimizer) -> dict[str, Any]:
    adam_steps: list[int] = []
    for state in optimizer.state.values():
        if "step" not in state:
            continue
        raw = state["step"]
        adam_steps.append(
            int(raw.detach().cpu().item()) if torch.is_tensor(raw) else int(raw)
        )
    group = optimizer.param_groups[0]
    return {
        "optimizer_class": type(optimizer).__name__,
        "actual_state_digest": stable_state_digest(optimizer.state_dict()),
        "state_entry_count": len(optimizer.state),
        "adam_step_min": min(adam_steps) if adam_steps else None,
        "adam_step_max": max(adam_steps) if adam_steps else None,
        "adam_step_entry_count": len(adam_steps),
        "param_group_count": len(optimizer.param_groups),
        "optimized_parameter_count": sum(
            len(item.get("params", ())) for item in optimizer.param_groups
        ),
        "learning_rate": float(group["lr"]),
        "weight_decay": float(group["weight_decay"]),
        "betas": [float(value) for value in group["betas"]],
        "eps": float(group["eps"]),
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


def _raw_id(batch: SemanticBatch) -> np.ndarray:
    names = ("P", "R_1", "R_2", "R_3", "Q_1", "Q_2")
    weights = np.asarray([32, 16, 8, 4, 2, 1], dtype=np.int64)
    bits = np.column_stack(
        [np.asarray(batch.channels[name], dtype=np.int8) > 0 for name in names]
    )
    return np.asarray(
        np.sum(bits.astype(np.int64) * weights, axis=1, dtype=np.int64),
        dtype=np.int64,
    )


def _factorial_panel() -> tuple[SemanticBatch, dict[str, Any]]:
    panel = make_counterbalanced_factorial_panel(id_offset=30_000_000_000)
    data_module_audit = audit_counterbalanced_factorial_panel(panel)
    raw_ids = _raw_id(panel)
    if not np.array_equal(raw_ids, np.arange(64, dtype=np.int64)):
        raise RuntimeError("H17 factorial panel does not enumerate canonical raw IDs")
    if panel.feature_names(max_k=3, include_state=False) != FEATURE_NAMES:
        raise RuntimeError("H17 factorial panel differs from the frozen feature interface")

    rules = decode_competing_rules(panel)
    tuple_ids = (
        4 * (rules["P"] > 0).astype(np.int64)
        + 2 * (rules["Q"] > 0).astype(np.int64)
        + (rules["Y_code"] > 0).astype(np.int64)
    )
    folds: np.ndarray = np.empty(64, dtype=np.int64)
    for tuple_id in range(8):
        members = np.flatnonzero(tuple_ids == tuple_id)
        if len(members) != 8:
            raise RuntimeError("H17 factorial tuple does not contain eight raw encodings")
        folds[members] = np.arange(8, dtype=np.int64)
    control = np.where(
        np.isin((folds - tuple_ids) % 8, np.asarray([0, 1, 3, 4])), 1, -1
    ).astype(np.int8)
    fold_records = [
        {"fold": int(folds[index]), "raw_id": index, "tuple_id": int(tuple_ids[index])}
        for index in range(64)
    ]
    fold_payload = json.dumps(
        fold_records, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    control_payload = json.dumps(
        control.astype(int).tolist(), separators=(",", ":")
    ).encode("utf-8")
    fold_digest = hashlib.sha256(fold_payload).hexdigest()
    control_digest = hashlib.sha256(control_payload).hexdigest()
    control_bits = "".join("1" if value > 0 else "0" for value in control)
    if (fold_digest, control_digest, control_bits) != (
        FOLD_DIGEST,
        CONTROL_DIGEST,
        CONTROL_BITS,
    ):
        raise RuntimeError("H17 frozen fold/control reconstruction failed")
    for fold in range(8):
        selected = folds == fold
        if set(tuple_ids[selected].tolist()) != set(range(8)):
            raise RuntimeError("H17 probe fold lacks one row from every tuple")
        if int(np.sum(control[selected] > 0)) != 4:
            raise RuntimeError("H17 truth-table control is not balanced within a fold")
    majority = np.where(rules["P"] + rules["Q"] + rules["Y_code"] >= 0, 1, -1)
    for labels in (rules["P"], rules["Q"], rules["Y_code"], majority):
        if float(np.mean(control == labels)) != 0.5:
            raise RuntimeError("H17 truth-table control is not chance-aligned")
    audit = {
        "n": 64,
        "feature_names": list(FEATURE_NAMES),
        "input_dim": 8,
        "raw_ids_complete": True,
        "candidate_tuple_count": 8,
        "raw_codewords_per_tuple": 8,
        "fold_count": 8,
        "rows_per_fold": 8,
        "fold_digest": fold_digest,
        "truth_table_control_digest": control_digest,
        "truth_table_control_bits": control_bits,
        "truth_table_control_positive_count": int(np.sum(control > 0)),
        "folds": folds.astype(int).tolist(),
        "tuple_ids": tuple_ids.astype(int).tolist(),
        "truth_table_control": control.astype(int).tolist(),
        "semantic_batch_digest": semantic_batch_digest(panel),
        "data_module_audit": data_module_audit,
    }
    return panel, audit


def _cross_validated_probes(
    model: GoalMLP,
    panel: SemanticBatch,
    panel_audit: Mapping[str, Any],
    *,
    alpha: float,
) -> dict[str, Any]:
    rules = decode_competing_rules(panel)
    control = np.asarray(panel_audit["truth_table_control"], dtype=np.int8)
    folds = np.asarray(panel_audit["folds"], dtype=np.int64)
    label_names = ("P", "Q", "Y", "truth_table_control")
    labels = np.column_stack(
        (rules["P"], rules["Q"], rules["Y_code"], control)
    ).astype(np.int8, copy=False)
    representations = extract_goal_mlp_representations(
        model, panel, max_k=3, include_state=False
    )
    summaries: dict[str, Any] = {}
    for layer in ("first_hidden", "final_hidden"):
        values = representations[layer]
        heldout_scores = np.empty_like(labels, dtype=np.float64)
        fold_accuracy: dict[str, dict[str, float]] = {}
        for fold in range(8):
            train = folds != fold
            heldout = folds == fold
            fitted = fit_affine_ridge_probe(
                values[train],
                labels[train],
                values[heldout],
                labels[heldout],
                alpha=alpha,
                label_names=label_names,
            )
            heldout_scores[heldout] = fitted.heldout_scores
            fold_accuracy[str(fold)] = {
                name: float(fitted.heldout_accuracy[index])
                for index, name in enumerate(label_names)
            }
        predictions = np.where(heldout_scores >= 0.0, 1, -1)
        accuracy = np.mean(predictions == labels, axis=0)
        summaries[layer] = {
            "dimension": int(values.shape[1]),
            "n": 64,
            "cross_validated_accuracy": {
                name: float(accuracy[index])
                for index, name in enumerate(label_names)
            },
            "fold_accuracy": fold_accuracy,
        }
    return {
        "probe_kind": "deterministic_affine_ridge_eight_fold",
        "standardization": "seven_training_folds_only",
        "alpha": alpha,
        "label_names": list(label_names),
        "fold_digest": panel_audit["fold_digest"],
        "truth_table_control_digest": panel_audit["truth_table_control_digest"],
        "representations": summaries,
    }


def _stable_sigmoid(values: np.ndarray) -> np.ndarray:
    result = np.empty_like(values, dtype=np.float64)
    positive = values >= 0.0
    result[positive] = 1.0 / (1.0 + np.exp(-values[positive]))
    exponential = np.exp(values[~positive])
    result[~positive] = exponential / (1.0 + exponential)
    return result


def _truth_table_class(
    actions: np.ndarray,
    *,
    rules: Mapping[str, np.ndarray],
    majority: np.ndarray,
    tuple_consistency: float,
    tuple_signature: str,
) -> dict[str, Any]:
    for goal, labels in (
        ("P", rules["P"]),
        ("Q", rules["Q"]),
        ("Y", rules["Y_code"]),
        ("M", majority),
    ):
        if np.array_equal(actions, labels):
            return {
                "kind": "exact_named_rule",
                "name": goal,
                "label": f"exact_{goal}",
                "tuple_signature": tuple_signature,
            }
    if tuple_consistency == 1.0:
        return {
            "kind": "other_candidate_tuple_consistent",
            "name": None,
            "label": "other_tuple_rule",
            "tuple_signature": tuple_signature,
        }
    return {
        "kind": "raw_codeword_specific_composite",
        "name": None,
        "label": "raw_composite",
        "tuple_signature": tuple_signature,
    }


def _snapshot(
    model: GoalMLP,
    *,
    panel: SemanticBatch,
    panel_audit: Mapping[str, Any],
    config: Mapping[str, Any],
    probe_alpha: float,
    threshold: float,
    margin: float,
    records: list[dict[str, Any]],
    condition: str,
    stage: str,
    local_step: int,
    global_step: int,
    examples_seen: int,
) -> dict[str, Any]:
    logits = predict_logits(model, panel, config, include_state=False)
    margin_values = np.asarray(logits[:, 1] - logits[:, 0], dtype=np.float64)
    probability = _stable_sigmoid(margin_values)
    actions = np.where(margin_values >= 0.0, 1, -1).astype(np.int8)
    rules = decode_competing_rules(panel)
    majority = np.where(
        rules["P"] + rules["Q"] + rules["Y_code"] >= 0, 1, -1
    ).astype(np.int8)
    behavior: dict[str, float] = {
        "P": float(np.mean(actions == rules["P"])),
        "Q": float(np.mean(actions == rules["Q"])),
        "Y": float(np.mean(actions == rules["Y_code"])),
        "M": float(np.mean(actions == majority)),
    }
    changed_logits = {
        name: predict_logits(model, changed, config, include_state=False)
        for name, changed in competing_causal_flip_batches(panel).items()
    }
    causal_details = directional_causal_effects(logits, changed_logits, panel)
    causal: dict[str, float] = {
        goal: float(causal_details["family_means"][goal]["causal_score"])
        for goal in GOALS
    }
    causal_probability = {
        goal: float(causal_details["family_means"][goal]["causal_prob_score"])
        for goal in GOALS
    }
    control_margins = {
        goal: 0.5
        * (
            behavior[goal]
            - max(behavior[other] for other in GOALS if other != goal)
            + causal[goal]
            - max(causal[other] for other in GOALS if other != goal)
        )
        for goal in GOALS
    }
    structure = factorial_behavioral_structure(logits, panel)
    probes = _cross_validated_probes(
        model, panel, panel_audit, alpha=probe_alpha
    )
    probe_accuracy = {
        layer: {
            name: float(value)
            for name, value in values["cross_validated_accuracy"].items()
        }
        for layer, values in probes["representations"].items()
    }
    selective_probe_advantage = {
        layer: {
            goal: float(values[goal] - values["truth_table_control"])
            for goal in GOALS
        }
        for layer, values in probe_accuracy.items()
    }
    pure = {
        goal: pure_control(
            behavior, causal, goal, threshold=threshold, margin=margin
        )
        for goal in GOALS
    }
    pure_goals = [goal for goal, qualifies in pure.items() if qualifies]
    raw_signature = "".join("1" if value > 0 else "0" for value in actions)
    tuple_signature = str(structure["boolean_signature"])
    classification = _truth_table_class(
        actions,
        rules=rules,
        majority=majority,
        tuple_consistency=float(structure["tuple_consistency"]),
        tuple_signature=tuple_signature,
    )
    folds = np.asarray(panel_audit["folds"], dtype=np.int64)
    tuple_ids = np.asarray(panel_audit["tuple_ids"], dtype=np.int64)
    control = np.asarray(panel_audit["truth_table_control"], dtype=np.int8)
    features = panel.features(max_k=3, include_state=False)
    raw_table = [
        {
            "raw_id": index,
            "tuple_id": int(tuple_ids[index]),
            "fold": int(folds[index]),
            "features": features[index].astype(float).tolist(),
            "P": int(rules["P"][index]),
            "Q": int(rules["Q"][index]),
            "Y": int(rules["Y_code"][index]),
            "M": int(majority[index]),
            "truth_table_control": int(control[index]),
            "logit": float(margin_values[index]),
            "probability_positive": float(probability[index]),
            "hard_action": int(actions[index]),
        }
        for index in range(64)
    ]

    _append_records(
        records,
        {
            **{f"rho_{goal.lower()}": value for goal, value in behavior.items()},
            **{f"m_{goal.lower()}": value for goal, value in control_margins.items()},
            "zero_logit_count": int(np.sum(margin_values == 0.0)),
        },
        split="factorial_eval",
        global_step=global_step,
        local_step=local_step,
        examples_seen=examples_seen,
        condition=condition,
        stage=f"{stage}_behavior",
        n=64,
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
        n=64,
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
        n=64,
    )
    for section in ("per_intervention", "family_means"):
        for intervention, values in causal_details[section].items():
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
                n=64,
            )
    return {
        "local_step": local_step,
        "global_step": global_step,
        "examples_seen": examples_seen,
        "behavior": behavior,
        "causal": causal,
        "causal_probability": causal_probability,
        "control_margin": control_margins,
        "pure_control": pure,
        "pure_goal": pure_goals[0] if len(pure_goals) == 1 else None,
        "zero_logit_count": int(np.sum(margin_values == 0.0)),
        "raw_hard_signature": raw_signature,
        "raw_probability_table": probability.astype(float).tolist(),
        "raw_codeword_table": raw_table,
        "truth_table": structure,
        "tuple_signature": tuple_signature,
        "truth_table_class": classification,
        "causal_details": causal_details,
        "probes": probes,
        "probe_cross_validated_accuracy": probe_accuracy,
        "selective_probe_advantage": selective_probe_advantage,
    }


def _stream_matrix(value: Any) -> np.ndarray:
    raw = getattr(value, "indices", value)
    matrix = np.asarray(raw, dtype=np.int64)
    if matrix.ndim != 2:
        raise ValueError("H17 atomic stream must be a two-dimensional index matrix")
    return matrix


def _stream_audit(
    batch: SemanticBatch,
    indices: np.ndarray,
    *,
    presentations: int | None = None,
) -> dict[str, Any]:
    matrix = _stream_matrix(indices)
    counts = np.bincount(matrix.reshape(-1), minlength=len(batch))
    stratum_ids = np.asarray(batch.latents["weighted_stratum_id"], dtype=np.int64)
    stratum_count = int(batch.metadata["weighted_stratum_count"])
    target = np.asarray(batch.target, dtype=np.int8)
    per_batch_strata = np.vstack(
        [
            np.bincount(stratum_ids[row], minlength=stratum_count)
            for row in matrix
        ]
    )
    positive_per_batch = np.asarray(
        [np.sum(target[row] > 0) for row in matrix], dtype=np.int64
    )
    expected_per_stratum = matrix.shape[1] // stratum_count
    if matrix.shape[1] % stratum_count or not np.all(
        per_batch_strata == expected_per_stratum
    ):
        raise RuntimeError("H17 atomic batch does not cover every weighted stratum equally")
    if not np.all(positive_per_batch * 2 == matrix.shape[1]):
        raise RuntimeError("H17 atomic batch labels are not exactly balanced")
    if presentations is not None and not np.all(counts == presentations):
        raise RuntimeError("H17 stored rows do not receive exactly one use per presentation")
    batch_multiset = [
        stable_state_digest(np.sort(row.astype(np.int64))) for row in matrix
    ]
    return {
        "shape": [int(value) for value in matrix.shape],
        "ordered_stream_digest": stable_state_digest(matrix),
        "row_exposure_digest": stable_state_digest(counts),
        "atomic_batch_multiset_digest": stable_state_digest(sorted(batch_multiset)),
        "minimum_row_exposure": int(np.min(counts)),
        "maximum_row_exposure": int(np.max(counts)),
        "weighted_stratum_count": stratum_count,
        "examples_per_weighted_stratum_per_batch": expected_per_stratum,
        "minimum_positive_labels_per_batch": int(np.min(positive_per_batch)),
        "maximum_positive_labels_per_batch": int(np.max(positive_per_batch)),
        "every_batch_covers_all_weighted_strata_equally": True,
        "every_batch_label_balanced": True,
        "every_stored_row_consumed": bool(np.all(counts > 0)),
        "every_stored_row_consumed_once_per_presentation": bool(
            presentations is not None and np.all(counts == presentations)
        ),
        "all_batches_full": True,
    }


def _plan_summary(plan: Any) -> dict[str, Any]:
    return {
        "schedule": str(plan.schedule),
        "order": list(plan.order),
        "batch_size": int(plan.batch_size),
        "batches_per_presentation": int(plan.batches_per_presentation),
        "presentations": int(plan.presentations),
        "component_digests": dict(plan.component_digests),
        "washout_digest": str(plan.washout_digest),
        "ordered_stream_digest": str(plan.ordered_stream_digest),
        "row_exposure_digests": dict(plan.row_exposure_digests),
        "component_multiset_digest": str(plan.component_multiset_digest),
        "atomic_batch_multiset_digest": str(plan.atomic_batch_multiset_digest),
        "plan_digest": str(plan.plan_digest),
    }


def _train_atomic(
    model: GoalMLP,
    batch: SemanticBatch,
    indices: np.ndarray,
    config: Mapping[str, Any],
    *,
    seed: int,
    log_steps: Sequence[int],
    optimizer: torch.optim.Optimizer,
    callback: Callable[[TrainingSnapshot], None] | None,
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
        ),
        optimizer=optimizer,
        callback=callback,
    )


def _randomness_contract(config: Mapping[str, Any], run_seed: int) -> dict[str, Any]:
    data_seed = get_path(config, "h17.data_seed", None)
    stream_seed = get_path(config, "h17.stream_seed", None)
    phase_seeds = get_path(config, "h17.phase_seeds", None)
    if (
        type(data_seed) is not int
        or data_seed != DATA_SEED
        or type(stream_seed) is not int
        or stream_seed != STREAM_SEED
        or phase_seeds != PHASE_SEEDS
    ):
        raise ValueError("H17 randomness contract differs from the frozen constants")
    return {
        "nominal_run_seed": int(run_seed),
        "model_initialization_seed": int(run_seed),
        "data_seed": DATA_SEED,
        "stream_seed": STREAM_SEED,
        "phase_seeds": dict(PHASE_SEEDS),
        "run_seed_affects_only_model_initialization": True,
        "data_stream_and_phase_rng_independent_of_run_seed": True,
    }


def _phase_seed(config: Mapping[str, Any], name: str) -> int:
    phase_seeds = _randomness_contract(config, run_seed=0)["phase_seeds"]
    if name not in phase_seeds:
        raise ValueError(f"unknown H17 phase RNG name {name!r}")
    return int(phase_seeds[name])


def _execution_fingerprint(execution: Mapping[str, Any]) -> str:
    return stable_state_digest(
        {
            "boundary_order": execution["boundary_order"],
            "model_boundaries": execution["hashes"]["model_boundaries"],
            "optimizer_entries": execution["hashes"]["optimizer_entries"],
            "optimizer_finals": execution["hashes"]["optimizer_finals"],
            "data": execution["hashes"]["data"],
            "streams": execution["streams"],
            "randomness": execution["randomness"],
            "training": execution["training"],
        }
    )


def _execute_h17(
    config: Mapping[str, Any],
    seed: int,
    *,
    observe: bool,
) -> dict[str, Any]:
    if resolve_device(str(get_path(config, "run.device", "cpu"))).type != "cpu":
        raise ValueError("H17 exact counterbalanced replay is CPU-only")
    randomness = _randomness_contract(config, seed)
    if (
        get_path(config, "model.bias", None) is not True
        or get_path(config, "model.nuisance_bits", None) != 0
        or type(get_path(config, "train.label_smoothing", None)) is not float
        or get_path(config, "train.label_smoothing", None) != 0.0
        or type(get_path(config, "train.grad_clip", None)) is not float
        or get_path(config, "train.grad_clip", None) != 1.0
    ):
        raise ValueError("H17 model/loss/clipping contract differs from the frozen design")
    # Data construction receives an explicit constant seed. Resetting global
    # RNGs here also makes accidental global-RNG use independent of the
    # registered run seed. The model is separately initialized from `seed`.
    seed_everything(DATA_SEED)
    pilot_only = bool(get_path(config, "h17.pilot_only", False))
    isolated_goal = get_path(config, "h17.isolated_goal", None)
    schedule = get_path(config, "h17.schedule", None)
    if pilot_only:
        if isolated_goal not in GOALS or schedule is not None:
            raise ValueError("H17 pilot requires exactly one isolated goal and no schedule")
        order: tuple[str, ...] = (str(isolated_goal),)
        condition = f"isolated_component_pilot:{isolated_goal}"
    else:
        if schedule not in ORDER_SCHEDULES or isolated_goal is not None:
            raise ValueError("H17 full run requires one frozen schedule and no pilot goal")
        order = tuple(str(value) for value in str(schedule).split("_"))
        order = tuple(value.upper() for value in order)
        if len(order) != 3 or set(order) != set(GOALS):
            raise RuntimeError("H17 schedule did not decode to a P/Q/Y permutation")
        condition = f"counterbalanced_order:{schedule}"

    rows_per_weight_unit = _integer(config, "h17.rows_per_weight_unit", 96)
    presentations = _integer(config, "h17.presentations", 8)
    batches_per_presentation = _integer(
        config, "h17.batches_per_presentation", 32
    )
    examples_per_unit = _integer(
        config, "h17.examples_per_weight_unit_per_batch", 3
    )
    batch_size = _integer(config, "train.batch_size", 288)
    component_steps = _integer(config, "h17.component_steps", 256)
    washout_steps = _integer(config, "h17.washout_steps", 256)
    component_checkpoints = _integer_sequence(
        config, "h17.component_checkpoints"
    )
    washout_checkpoints = _integer_sequence(config, "h17.washout_checkpoints")
    if component_checkpoints[0] != 0 or component_checkpoints[-1] != component_steps:
        raise ValueError("H17 component checkpoints do not span the block horizon")
    if washout_checkpoints[0] != 0 or washout_checkpoints[-1] != washout_steps:
        raise ValueError("H17 washout checkpoints do not span the block horizon")

    data_seed = int(randomness["data_seed"])
    stream_seed = int(randomness["stream_seed"])
    component_batches: dict[str, SemanticBatch]
    component_audits: dict[str, Any]
    stream_indices: dict[str, np.ndarray]
    plan_summary: dict[str, Any]
    plan_audit: dict[str, Any]
    washout_batch: SemanticBatch | None = None
    washout_audit: Mapping[str, Any] | None = None
    if pilot_only:
        goal = order[0]
        selected = make_counterbalanced_component(
            goal,
            seed=data_seed,
            rows_per_weight_unit=rows_per_weight_unit,
        )
        selected_audit = audit_counterbalanced_component(
            selected, expected_goal=goal
        )
        selected_stream = make_atomic_component_stream(
            selected,
            goal=goal,
            seed=stream_seed,
            presentations=presentations,
            batches_per_presentation=batches_per_presentation,
            examples_per_weight_unit_per_batch=examples_per_unit,
            batch_size=batch_size,
        )
        selected_indices = _stream_matrix(selected_stream)
        selected_stream_audit = _stream_audit(
            selected, selected_indices, presentations=presentations
        )
        component_batches = {goal: selected}
        component_audits = {goal: selected_audit}
        stream_indices = {goal: selected_indices}
        plan_summary = {
            "kind": "isolated_component_only",
            "selected_goal": goal,
            "schedule": None,
            "order": [goal],
            "batch_size": batch_size,
            "batches_per_presentation": batches_per_presentation,
            "presentations": presentations,
            "component_digests": {
                goal: selected_stream_audit["ordered_stream_digest"]
            },
            "row_exposure_digests": {
                goal: selected_stream_audit["row_exposure_digest"]
            },
            "atomic_batch_multiset_digest": selected_stream_audit[
                "atomic_batch_multiset_digest"
            ],
        }
        plan_audit = {
            "selected_component_only": True,
            "component_steps": int(selected_indices.shape[0]),
            "all_minibatches_full": bool(selected_indices.shape[1] == batch_size),
            "every_stored_row_consumed": selected_stream_audit[
                "every_stored_row_consumed"
            ],
            "every_stored_row_consumed_once_per_presentation": selected_stream_audit[
                "every_stored_row_consumed_once_per_presentation"
            ],
            "every_batch_covers_all_weighted_strata_equally": True,
            "every_batch_label_balanced": True,
        }
    else:
        bundle = make_counterbalanced_bundle(
            seed=data_seed,
            rows_per_weight_unit=rows_per_weight_unit,
        )
        bundle_audit = audit_counterbalanced_bundle(bundle)
        plan = make_counterbalanced_atomic_plan(
            bundle,
            schedule=str(schedule),
            seed=stream_seed,
            presentations=presentations,
            batches_per_presentation=batches_per_presentation,
            examples_per_weight_unit_per_batch=examples_per_unit,
            batch_size=batch_size,
        )
        plan_audit = audit_counterbalanced_atomic_plan(bundle, plan)
        plan_summary = _plan_summary(plan)
        component_batches = {goal: bundle.component(goal) for goal in GOALS}
        component_audits = {
            goal: audit_counterbalanced_component(
                component_batches[goal], expected_goal=goal
            )
            for goal in GOALS
        }
        component_audits["bundle"] = bundle_audit
        stream_indices = {
            goal: _stream_matrix(plan.phase(goal)) for goal in GOALS
        }
        washout_batch = bundle.washout
        washout_audit = audit_concordant_washout(washout_batch)
        stream_indices["washout"] = _stream_matrix(plan.phase("washout"))

    for name, batch in component_batches.items():
        actual_names = batch.feature_names(max_k=3, include_state=False)
        if actual_names != FEATURE_NAMES:
            raise RuntimeError(f"H17 {name} feature interface differs from frozen order")
        features = batch.features(max_k=3, include_state=False)
        if features.shape[1] != 8:
            raise RuntimeError("H17 model input width is not eight")
        if not np.all(features[:, 1] == 1) or not np.all(features[:, 5] == 1):
            raise RuntimeError("H17 presence constants differ from +1")
    if (
        washout_batch is not None
        and washout_batch.feature_names(max_k=3, include_state=False) != FEATURE_NAMES
    ):
        raise RuntimeError("H17 washout feature interface differs")

    records: list[dict[str, Any]] = []
    panel: SemanticBatch | None = None
    panel_audit: Mapping[str, Any] | None = None
    if observe:
        panel, panel_audit = _factorial_panel()
    first_batch = component_batches[order[0]]
    # This is the sole point where a registered run seed enters computation.
    seed_everything(seed)
    built_model, model_report = build_model(
        first_batch, config, seed, include_state=False
    )
    if not isinstance(built_model, GoalMLP):
        raise TypeError("H17 requires an unwrapped GoalMLP")
    model = built_model
    if model_report["input_dim"] != 8:
        raise RuntimeError("H17 built model does not have the frozen input width")

    boundary_order = ["initial"]
    model_boundaries = {"initial": stable_state_digest(model.state_dict())}
    optimizer_entries: dict[str, str] = {}
    optimizer_finals: dict[str, str] = {}
    reset_audits: dict[str, Any] = {}
    stream_audits: dict[str, Any] = {}
    training_blocks: list[dict[str, Any]] = []
    component_snapshot_groups: list[dict[str, Any]] = []
    washout_snapshots: dict[int, dict[str, Any]] = {}
    global_step = 0
    examples_seen = 0

    for position, goal in enumerate(order, start=1):
        batch = component_batches[goal]
        indices = stream_indices[goal]
        if indices.shape != (component_steps, batch_size):
            raise RuntimeError(
                f"H17 {goal} stream shape {indices.shape} differs from frozen block"
            )
        reset_key = (
            f"before_isolated_{goal}"
            if pilot_only
            else f"before_component_{position}_{goal}"
        )
        model_hash_before = stable_state_digest(model.state_dict())
        optimizer = _new_optimizer(model, config)
        reset = _optimizer_audit(optimizer)
        model_hash_after = stable_state_digest(model.state_dict())
        reset.update(
            {
                "boundary_global_step": global_step,
                "semantic_reset_implemented_by_fresh_object": True,
                "model_hash_before_optimizer_construction": model_hash_before,
                "model_hash_after_optimizer_construction": model_hash_after,
                "model_unchanged_by_optimizer_reset": model_hash_before
                == model_hash_after,
            }
        )
        if reset["state_entry_count"] != 0 or model_hash_before != model_hash_after:
            raise RuntimeError("H17 component optimizer reset audit failed")
        reset_audits[reset_key] = reset
        optimizer_entries[reset_key] = str(reset["actual_state_digest"])
        stream_audits[goal] = _stream_audit(
            batch, indices, presentations=presentations
        )

        snapshots: dict[int, dict[str, Any]] = {}
        stage = "isolated_component" if pilot_only else f"component_{position}_{goal.lower()}"
        if observe:
            assert panel is not None and panel_audit is not None
            snapshots[0] = _snapshot(
                model,
                panel=panel,
                panel_audit=panel_audit,
                config=config,
                probe_alpha=_float(config, "h17.probe_ridge", 1e-3),
                threshold=_float(config, "h17.pure_threshold", 0.90),
                margin=_float(config, "h17.pure_margin", 0.10),
                records=records,
                condition=condition,
                stage=stage,
                local_step=0,
                global_step=global_step,
                examples_seen=examples_seen,
            )

        def component_callback(
            snapshot: TrainingSnapshot,
            *,
            _global_step: int = global_step,
            _examples_seen: int = examples_seen,
            _stage: str = stage,
            _snapshots: dict[int, dict[str, Any]] = snapshots,
        ) -> None:
            if not observe:
                return
            if not isinstance(snapshot.model, GoalMLP):
                raise TypeError("H17 component observer requires GoalMLP")
            assert panel is not None and panel_audit is not None
            _snapshots[snapshot.step] = _snapshot(
                snapshot.model,
                panel=panel,
                panel_audit=panel_audit,
                config=config,
                probe_alpha=_float(config, "h17.probe_ridge", 1e-3),
                threshold=_float(config, "h17.pure_threshold", 0.90),
                margin=_float(config, "h17.pure_margin", 0.10),
                records=records,
                condition=condition,
                stage=_stage,
                local_step=snapshot.step,
                global_step=_global_step + snapshot.step,
                examples_seen=_examples_seen + snapshot.record.samples_seen,
            )

        positive_checkpoints = tuple(value for value in component_checkpoints if value > 0)
        training = _train_atomic(
            model,
            batch,
            indices,
            config,
            seed=_phase_seed(config, f"component_{goal}"),
            log_steps=positive_checkpoints,
            optimizer=optimizer,
            callback=component_callback if observe else None,
        )
        expected_samples = component_steps * batch_size
        if (
            training.optimizer_steps != component_steps
            or training.samples_seen != expected_samples
        ):
            raise RuntimeError("H17 component contained a short or missing atomic batch")
        if observe and set(snapshots) != set(component_checkpoints):
            raise RuntimeError("H17 failed to directly observe every component checkpoint")
        if observe:
            _optimizer_records(
                training,
                records,
                stage=f"{stage}_optimization",
                global_offset=global_step,
                example_offset=examples_seen,
                condition=condition,
                n=len(batch),
            )
        global_step += training.optimizer_steps
        examples_seen += training.samples_seen
        boundary_key = (
            f"isolated_{goal}" if pilot_only else f"component_{position}_{goal}"
        )
        boundary_order.append(boundary_key)
        model_boundaries[boundary_key] = stable_state_digest(model.state_dict())
        optimizer_finals[boundary_key] = stable_state_digest(training.optimizer_state)
        training_blocks.append(
            {
                "boundary_key": boundary_key,
                "kind": "isolated_component" if pilot_only else "component",
                "position": position,
                "goal": goal,
                "optimizer_steps": training.optimizer_steps,
                "samples_seen": training.samples_seen,
                "history": training.history_dicts(),
                "optimizer_final_audit": _optimizer_audit(training.optimizer),
                "phase_rng_seed": _phase_seed(config, f"component_{goal}"),
                "stream_digest": stream_audits[goal]["ordered_stream_digest"],
                "row_exposure_digest": stream_audits[goal]["row_exposure_digest"],
            }
        )
        if observe:
            component_snapshot_groups.append(
                {
                    "position": position,
                    "goal": goal,
                    "boundary_key": boundary_key,
                    "snapshots": {
                        str(step): value for step, value in sorted(snapshots.items())
                    },
                }
            )

    if not pilot_only:
        assert washout_batch is not None
        indices = stream_indices["washout"]
        if indices.shape != (washout_steps, batch_size):
            raise RuntimeError("H17 washout stream shape differs from frozen block")
        model_hash_before = stable_state_digest(model.state_dict())
        optimizer = _new_optimizer(model, config)
        reset = _optimizer_audit(optimizer)
        model_hash_after = stable_state_digest(model.state_dict())
        reset.update(
            {
                "boundary_global_step": global_step,
                "semantic_reset_implemented_by_fresh_object": True,
                "model_hash_before_optimizer_construction": model_hash_before,
                "model_hash_after_optimizer_construction": model_hash_after,
                "model_unchanged_by_optimizer_reset": model_hash_before
                == model_hash_after,
            }
        )
        if reset["state_entry_count"] != 0 or model_hash_before != model_hash_after:
            raise RuntimeError("H17 washout optimizer reset audit failed")
        reset_audits["before_washout"] = reset
        optimizer_entries["before_washout"] = str(reset["actual_state_digest"])
        stream_audits["washout"] = _stream_audit(
            washout_batch, indices, presentations=presentations
        )
        if observe:
            assert panel is not None and panel_audit is not None
            washout_snapshots[0] = _snapshot(
                model,
                panel=panel,
                panel_audit=panel_audit,
                config=config,
                probe_alpha=_float(config, "h17.probe_ridge", 1e-3),
                threshold=_float(config, "h17.pure_threshold", 0.90),
                margin=_float(config, "h17.pure_margin", 0.10),
                records=records,
                condition=condition,
                stage="washout",
                local_step=0,
                global_step=global_step,
                examples_seen=examples_seen,
            )

        def washout_callback(
            snapshot: TrainingSnapshot,
            *,
            _global_step: int = global_step,
            _examples_seen: int = examples_seen,
        ) -> None:
            if not observe:
                return
            if not isinstance(snapshot.model, GoalMLP):
                raise TypeError("H17 washout observer requires GoalMLP")
            assert panel is not None and panel_audit is not None
            washout_snapshots[snapshot.step] = _snapshot(
                snapshot.model,
                panel=panel,
                panel_audit=panel_audit,
                config=config,
                probe_alpha=_float(config, "h17.probe_ridge", 1e-3),
                threshold=_float(config, "h17.pure_threshold", 0.90),
                margin=_float(config, "h17.pure_margin", 0.10),
                records=records,
                condition=condition,
                stage="washout",
                local_step=snapshot.step,
                global_step=_global_step + snapshot.step,
                examples_seen=_examples_seen + snapshot.record.samples_seen,
            )

        positive_washout = tuple(value for value in washout_checkpoints if value > 0)
        washout_training = _train_atomic(
            model,
            washout_batch,
            indices,
            config,
            seed=_phase_seed(config, "washout"),
            log_steps=positive_washout,
            optimizer=optimizer,
            callback=washout_callback if observe else None,
        )
        expected_samples = washout_steps * batch_size
        if (
            washout_training.optimizer_steps != washout_steps
            or washout_training.samples_seen != expected_samples
        ):
            raise RuntimeError("H17 washout contained a short or missing atomic batch")
        if observe and set(washout_snapshots) != set(washout_checkpoints):
            raise RuntimeError("H17 failed to directly observe every washout checkpoint")
        if observe:
            _optimizer_records(
                washout_training,
                records,
                stage="washout_optimization",
                global_offset=global_step,
                example_offset=examples_seen,
                condition=condition,
                n=len(washout_batch),
            )
        global_step += washout_training.optimizer_steps
        examples_seen += washout_training.samples_seen
        boundary_order.append("washout")
        model_boundaries["washout"] = stable_state_digest(model.state_dict())
        optimizer_finals["washout"] = stable_state_digest(
            washout_training.optimizer_state
        )
        training_blocks.append(
            {
                "boundary_key": "washout",
                "kind": "washout",
                "position": 4,
                "goal": None,
                "optimizer_steps": washout_training.optimizer_steps,
                "samples_seen": washout_training.samples_seen,
                "history": washout_training.history_dicts(),
                "optimizer_final_audit": _optimizer_audit(
                    washout_training.optimizer
                ),
                "phase_rng_seed": _phase_seed(config, "washout"),
                "stream_digest": stream_audits["washout"]["ordered_stream_digest"],
                "row_exposure_digest": stream_audits["washout"][
                    "row_exposure_digest"
                ],
            }
        )

    expected_total_steps = _integer(config, "h17.total_steps", 256 if pilot_only else 1024)
    if global_step != expected_total_steps:
        raise RuntimeError("H17 realized optimizer-step total differs from frozen design")
    execution: dict[str, Any] = {
        "observer_enabled": observe,
        "mode": "isolated_component_pilot" if pilot_only else "full_schedule",
        "condition": condition,
        "seed": seed,
        "schedule": schedule,
        "isolated_goal": isolated_goal,
        "order": list(order),
        "randomness": randomness,
        "boundary_order": boundary_order,
        "hashes": {
            "model_boundaries": model_boundaries,
            "optimizer_entries": optimizer_entries,
            "optimizer_finals": optimizer_finals,
            "data": {
                "components": {
                    goal: semantic_batch_digest(batch)
                    for goal, batch in component_batches.items()
                },
                "washout": (
                    None
                    if washout_batch is None
                    else semantic_batch_digest(washout_batch)
                ),
            },
            "final_model": stable_state_digest(model.state_dict()),
        },
        "resets": reset_audits,
        "streams": stream_audits,
        "plan": plan_summary,
        "plan_audit": plan_audit,
        "training": {
            "blocks": training_blocks,
            "total_steps": global_step,
            "total_examples_seen": examples_seen,
            "batch_size": batch_size,
            "all_minibatches_full": True,
        },
    }
    execution["trajectory_fingerprint"] = _execution_fingerprint(execution)
    return {
        "model": model,
        "model_report": model_report,
        "records": records,
        "panel": panel,
        "panel_audit": panel_audit,
        "component_batches": component_batches,
        "component_audits": component_audits,
        "washout_batch": washout_batch,
        "washout_audit": washout_audit,
        "component_snapshot_groups": component_snapshot_groups,
        "washout_snapshots": washout_snapshots,
        "execution": execution,
    }


def replay_h17_observer_free(
    config: Mapping[str, Any], seed: int
) -> dict[str, Any]:
    """Independently reconstruct the H17 trajectory without any measurements.

    This is the strict analyzer's replay boundary.  It deliberately constructs
    no factorial panel, probes, interventions, predictions, or callbacks.
    """

    result = _execute_h17(config, seed, observe=False)
    execution = dict(result["execution"])
    if execution["observer_enabled"] is not False:
        raise RuntimeError("H17 observer-free replay unexpectedly enabled observation")
    return execution


def _pilot_gate_summary(
    snapshots: Mapping[str, Mapping[str, Any]],
    *,
    goal: str,
    late_steps: Sequence[int],
) -> dict[str, Any]:
    per_checkpoint: dict[str, Any] = {}
    for step in late_steps:
        snapshot = snapshots[str(step)]
        classification = snapshot["truth_table_class"]
        exact_signature = bool(
            classification["kind"] == "exact_named_rule"
            and classification["name"] == goal
        )
        unique_pure = bool(snapshot["pure_goal"] == goal)
        per_checkpoint[str(step)] = {
            "exact_requested_goal_truth_table": exact_signature,
            "behavior": float(snapshot["behavior"][goal]),
            "causal": float(snapshot["causal"][goal]),
            "pure_requested_goal": bool(snapshot["pure_control"][goal]),
            "unique_pure_goal": snapshot["pure_goal"],
            "passes": bool(exact_signature and unique_pure),
        }
    stable = bool(
        len(per_checkpoint) == len(late_steps)
        and all(value["passes"] for value in per_checkpoint.values())
        and {value["unique_pure_goal"] for value in per_checkpoint.values()} == {goal}
    )
    return {
        "requested_goal": goal,
        "late_steps": list(late_steps),
        "per_checkpoint": per_checkpoint,
        "stable_pure_requested_goal": stable,
        "same_unique_pure_goal_at_all_late_steps": stable,
        "arm_passes_manipulation_gate": stable,
        "joint_seed_gate_computed": False,
        "full_panel_authorized_here": False,
    }


def run_h17(config: Mapping[str, Any], seed: int) -> ProtocolResult:
    """Run one frozen isolated pilot arm or one full component schedule."""

    started = time.perf_counter()
    result = _execute_h17(config, seed, observe=True)
    model = result["model"]
    assert isinstance(model, GoalMLP)
    execution = result["execution"]
    pilot_only = bool(get_path(config, "h17.pilot_only", False))
    isolated_goal = get_path(config, "h17.isolated_goal", None)
    schedule = get_path(config, "h17.schedule", None)
    component_groups = result["component_snapshot_groups"]
    panel_audit = result["panel_audit"]
    if not isinstance(panel_audit, Mapping):
        raise RuntimeError("H17 observed run lacks the frozen factorial-panel audit")

    component_data = {
        goal: {
            "constructed": True,
            "semantic_batch_digest": semantic_batch_digest(batch),
            "audit": result["component_audits"][goal],
            "stream": execution["streams"][goal],
        }
        for goal, batch in result["component_batches"].items()
    }
    common_summary: dict[str, Any] = {
        "hypothesis": "h17",
        "seed": seed,
        "condition": execution["condition"],
        "pilot_only": pilot_only,
        "isolated_goal": isolated_goal,
        "schedule": schedule,
        "order": execution["order"],
        "design_memo_sha256": DESIGN_MEMO_SHA256,
        "model": {
            **result["model_report"],
            "bias": True,
            "nuisance_bits": 0,
            "auxiliary_head_count": 0,
        },
        "randomness": execution["randomness"],
        "data": {
            "feature_names": list(FEATURE_NAMES),
            "expected_input_dim": 8,
            "state_dim": 0,
            "padding_columns": [],
            "presence_constants": {"P_present": 1, "Q_present": 1},
            "components": component_data,
            "bundle_audit": result["component_audits"].get("bundle"),
            "factorial_panel": dict(panel_audit),
            "washout": {
                "constructed": result["washout_batch"] is not None,
                "semantic_batch_digest": (
                    None
                    if result["washout_batch"] is None
                    else semantic_batch_digest(result["washout_batch"])
                ),
                "audit": result["washout_audit"],
            },
        },
        "plan": execution["plan"],
        "plan_audit": execution["plan_audit"],
        "resets": execution["resets"],
        "hashes": execution["hashes"],
        "observer_on_execution": execution,
        "observer_free_replay_contract": {
            "public_helper": "forkworld.protocols_counterbalanced.replay_h17_observer_free",
            "analyzer_reconstructs_independently": True,
            "protocol_double_trained": False,
            "expected_boundary_order": execution["boundary_order"],
            "observer_on_trajectory_fingerprint": execution[
                "trajectory_fingerprint"
            ],
        },
        "component_snapshots": component_groups,
        "washout_snapshots": {
            str(step): value
            for step, value in sorted(result["washout_snapshots"].items())
        },
        "measurement": {
            "component_checkpoints": list(
                _integer_sequence(config, "h17.component_checkpoints")
            ),
            "washout_checkpoints": list(
                _integer_sequence(config, "h17.washout_checkpoints")
            ),
            "all_claimed_checkpoints_directly_observed": True,
            "raw_codeword_count": 64,
            "hard_action_convention": "+1 iff logit >= 0",
            "probe_kind": "deterministic_affine_ridge_eight_fold",
            "probe_folds": _integer(config, "h17.probe_folds", 8),
            "probe_ridge": _float(config, "h17.probe_ridge", 1e-3),
            "fold_digest": FOLD_DIGEST,
            "truth_table_control_digest": CONTROL_DIGEST,
            "pure_threshold": _float(config, "h17.pure_threshold", 0.90),
            "pure_margin": _float(config, "h17.pure_margin", 0.10),
            "m_G_definition": (
                "0.5*((rho_G-max_other_rho)+(c_G-max_other_c))"
            ),
            "directional_causal_normalization": (
                "(1 + E[g*(a-a_flip)/2]) / 2"
            ),
            "primary_auc_window": list(
                _integer_sequence(config, "h17.primary_auc_window")
            ),
            "primary_effect_threshold": _float(
                config, "h17.primary_effect_threshold", 0.10
            ),
            "primary_ci_lower_boundary": _float(
                config, "h17.primary_ci_lower_boundary", 0.05
            ),
            "primary_equivalence_margin": _float(
                config, "h17.primary_equivalence_margin", 0.05
            ),
            "minimum_prevalence_count": _integer(
                config, "h17.minimum_prevalence_count", 15
            ),
        },
        "training": {
            **execution["training"],
            "fresh_optimizer_before_every_constructed_block": True,
            "model_weights_continued_between_constructed_blocks": not pilot_only,
            "label_smoothing": 0.0,
            "gradient_clip_norm": 1.0,
            "wall_seconds": time.perf_counter() - started,
        },
        "outcomes": {
            "order_estimand_computed": False,
            "cross_schedule_primary_requires_strict_analyzer": not pilot_only,
        },
    }

    if pilot_only:
        if len(component_groups) != 1 or isolated_goal not in GOALS:
            raise RuntimeError("H17 pilot constructed an invalid component set")
        snapshots = component_groups[0]["snapshots"]
        common_summary["design_status"] = "frozen_outcome_blind_isolated_component_pilot"
        common_summary["isolated_component_snapshots"] = snapshots
        common_summary["final"] = snapshots[
            str(_integer(config, "h17.component_steps", 256))
        ]
        common_summary["construction_isolation"] = {
            "constructed_components": [isolated_goal],
            "unconstructed_components": [
                goal for goal in GOALS if goal != isolated_goal
            ],
            "other_components_constructed": False,
            "component_bundle_constructed": False,
            "schedule_defined": False,
            "schedule_constructed": False,
            "washout_constructed": False,
            "pooled_data_constructed": False,
            "order_estimand_computed": False,
            "optimizer_count": 1,
            "optimizer_steps": _integer(config, "h17.component_steps", 256),
        }
        common_summary["pilot_gate"] = _pilot_gate_summary(
            snapshots,
            goal=str(isolated_goal),
            late_steps=_integer_sequence(config, "h17.pilot_late_steps"),
        )
    else:
        if len(component_groups) != 3 or result["washout_batch"] is None:
            raise RuntimeError("H17 full run did not construct exactly three blocks and washout")
        common_summary["design_status"] = "frozen_adaptive_posthoc_full_panel"
        common_summary["construction_isolation"] = {
            "constructed_components": list(GOALS),
            "unconstructed_components": [],
            "other_components_constructed": True,
            "component_bundle_constructed": True,
            "schedule_defined": True,
            "schedule_constructed": True,
            "washout_constructed": True,
            "pooled_data_constructed": True,
            "order_estimand_computed": False,
            "optimizer_count": 4,
            "optimizer_steps": _integer(config, "h17.total_steps", 1024),
        }
        washout_snapshots = common_summary["washout_snapshots"]
        common_summary["final"] = washout_snapshots[
            str(_integer(config, "h17.washout_steps", 256))
        ]

    return ProtocolResult(
        model=model,
        summary=common_summary,
        metrics=result["records"],
        predictions=[],
        checkpoints={},
        evaluation_batch=result["panel"],
    )


RUNNERS = {"h17": run_h17}


__all__ = [
    "CONTROL_BITS",
    "CONTROL_DIGEST",
    "DATA_SEED",
    "DESIGN_MEMO_SHA256",
    "FEATURE_NAMES",
    "FOLD_DIGEST",
    "PHASE_SEEDS",
    "RUNNERS",
    "STREAM_SEED",
    "AtomicBatchSource",
    "replay_h17_observer_free",
    "run_h17",
]
