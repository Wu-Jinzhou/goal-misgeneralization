"""Adaptive winner-knockout handoff protocol (E17/H14).

The protocol deliberately rebuilds each historical prefix twice on CPU.  One
copy is observed at the registered checkpoints and used for the phase-B branch;
the other is an observer-free replay.  A historical branch is allowed to run
only when the initial, final-model, final-optimizer, data, and minibatch hashes
agree exactly.  All six arms for a seed then consume one common paired phase-B
batch with one common static sampler stream.
"""

from __future__ import annotations

import math
import time
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import torch

from .competing import (
    competing_causal_flip_batches,
    competing_rule_agreements,
    decode_competing_rules,
    make_competing_bundle,
    make_competing_factorial_dataset,
)
from .config import get_path
from .data import SemanticBatch
from .handoff import (
    audit_compute_sham,
    audit_handoff_phase_b,
    make_compute_sham,
    make_handoff_phase_b,
    normalized_control_auc,
    persistent_handoff,
    pure_control,
    semantic_batch_digest,
    stable_state_digest,
    static_sampler_digest,
)
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

TRAJECTORIES = (
    "independent_carry",
    "independent_reset",
    "nested_carry",
    "nested_reset",
    "scratch",
    "sham",
)


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
            hypothesis="h14",
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
        shuffle=bool(get_path(config, "train.shuffle", True)),
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
    raise ValueError("H14 train.optimizer must be adamw, adam, or sgd")


def _optimizer_entry_audit(optimizer: torch.optim.Optimizer) -> dict[str, Any]:
    steps: list[int] = []
    for state in optimizer.state.values():
        if "step" not in state:
            continue
        raw = state["step"]
        value = int(raw.detach().cpu().item()) if torch.is_tensor(raw) else int(raw)
        steps.append(value)
    state_dict = optimizer.state_dict()
    return {
        "actual_state_digest": stable_state_digest(state_dict),
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

    _append_records(
        records,
        candidates,
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
        "strength": strength,
        "causal_details": causal,
        "probe_heldout_accuracy": probe_heldout,
        "selective_final_hidden_probe": selective_probe,
        "truth_table": structure,
        "boolean_signature": structure["boolean_signature"],
    }


def _train_prefix(
    model: GoalMLP,
    train_batch: SemanticBatch,
    config: Mapping[str, Any],
    *,
    seed: int,
    steps: int,
    log_steps: Sequence[int],
    callback: Any = None,
) -> TrainingResult:
    return train_clean_sft(
        model,
        batch_for_training(train_batch, config),
        _sft_config(
            config,
            seed=seed,
            steps=steps,
            log_steps=log_steps,
            reset_optimizer=True,
        ),
        callback=callback,
    )


def _historical_prefix(
    *,
    overlap: str,
    config: Mapping[str, Any],
    seed: int,
    q_p: float,
    q_q: float,
    k_q: int,
    k_y: int,
    max_k_q: int,
    max_k_y: int,
    state_dim: int,
    n_train: int,
    n_validation: int,
    n_eval: int,
    phase_a_steps: int,
    eligibility_steps: tuple[int, ...],
    observer: Any,
) -> tuple[GoalMLP, TrainingResult, dict[str, Any], dict[str, Any]]:
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
    model, report = build_model(bundle.train, config, seed)
    if not isinstance(model, GoalMLP):
        raise TypeError("H14 requires an unwrapped GoalMLP")
    initial_hash = stable_state_digest(model.state_dict())

    def callback(snapshot: TrainingSnapshot) -> None:
        observer(snapshot.model, snapshot.step, snapshot.record.samples_seen)

    observed = _train_prefix(
        model,
        bundle.train,
        config,
        seed=seed,
        steps=phase_a_steps,
        log_steps=eligibility_steps,
        callback=callback,
    )

    replay_model, replay_report = build_model(bundle.train, config, seed)
    if not isinstance(replay_model, GoalMLP):
        raise TypeError("H14 requires an unwrapped GoalMLP")
    replay_initial_hash = stable_state_digest(replay_model.state_dict())
    replay = _train_prefix(
        replay_model,
        bundle.train,
        config,
        seed=seed,
        steps=phase_a_steps,
        log_steps=eligibility_steps,
    )
    hashes = {
        "initial_model": initial_hash,
        "replay_initial_model": replay_initial_hash,
        "observed_final_model": stable_state_digest(observed.final_model_state),
        "replay_final_model": stable_state_digest(replay.final_model_state),
        "observed_final_optimizer": stable_state_digest(observed.optimizer_state),
        "replay_final_optimizer": stable_state_digest(replay.optimizer_state),
        "phase_a_batch": semantic_batch_digest(bundle.train),
        "phase_a_sampler": static_sampler_digest(
            len(bundle.train),
            batch_size=_integer(config, "train.batch_size", 250),
            steps=phase_a_steps,
            seed=seed,
            shuffle=bool(get_path(config, "train.shuffle", True)),
        ),
    }
    checks = {
        "initial_models_equal": hashes["initial_model"]
        == hashes["replay_initial_model"],
        "final_models_equal": hashes["observed_final_model"]
        == hashes["replay_final_model"],
        "final_optimizers_equal": hashes["observed_final_optimizer"]
        == hashes["replay_final_optimizer"],
        "samples_seen_equal": observed.samples_seen == replay.samples_seen,
        "optimizer_steps_equal": observed.optimizer_steps == replay.optimizer_steps,
    }
    if report != replay_report or not all(checks.values()):
        raise RuntimeError(f"H14 {overlap} prefix failed deterministic replay: {checks}")
    return model, observed, report, {
        "overlap": overlap,
        "hashes": hashes,
        "checks": checks,
        "samples_seen": observed.samples_seen,
        "optimizer_steps": observed.optimizer_steps,
        "batch_size": _integer(config, "train.batch_size", 250),
        "all_minibatches_full": observed.samples_seen
        == phase_a_steps * _integer(config, "train.batch_size", 250),
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
    }


def run_h14(config: Mapping[str, Any], seed: int) -> ProtocolResult:
    """Run one of the six frozen winner-knockout trajectories."""

    if resolve_device(str(get_path(config, "run.device", "cpu"))).type != "cpu":
        raise ValueError("H14 exact prefix replay is CPU-only")
    seed_everything(seed)
    trajectory = str(get_path(config, "h14.trajectory", "independent_carry"))
    if trajectory not in TRAJECTORIES:
        raise ValueError(f"h14.trajectory must be one of {TRAJECTORIES}")
    condition = f"winner_knockout:{trajectory}"
    n_train = _integer(config, "data.n_train", 10_000)
    n_validation = _integer(config, "data.n_validation", 4_000)
    n_eval = _integer(config, "data.n_eval", 10_000)
    batch_size = _integer(config, "train.batch_size", 250)
    if n_train % batch_size:
        raise ValueError("H14 requires full, equal-size static minibatches")

    q_p = _float(config, "h14.q_p", 0.90)
    q_q = _float(config, "h14.q_q", 0.90)
    k_q = _integer(config, "h14.k_q", 2)
    k_y = _integer(config, "h14.k_y", 3)
    max_k_q = _integer(config, "h14.max_k_q", 3)
    max_k_y = _integer(config, "h14.max_k_y", 5)
    state_dim = _integer(config, "data.state_dim", 8)
    phase_a_steps = _integer(config, "h14.phase_a_steps", 45)
    phase_b_steps = _integer(config, "h14.phase_b_steps", 1_024)
    eligibility_steps = _integer_sequence(config, "h14.eligibility_steps")
    phase_b_checkpoints = _integer_sequence(config, "h14.phase_b_checkpoints")
    if 0 not in phase_b_checkpoints:
        raise ValueError("h14.phase_b_checkpoints must include local update zero")
    trained_phase_b_steps = tuple(step for step in phase_b_checkpoints if step > 0)
    probe_train_n = _integer(config, "h14.probe_train_n", 2_048)
    probe_eval_n = _integer(config, "h14.probe_eval_n", 4_096)
    probe_alpha = _float(config, "h14.probe_ridge", 1e-3)
    control_seed = _integer(
        config, "h14.truth_table_control_seed", 1_500_450_271
    )
    sham_repeats = _integer(config, "h14.sham_source_repeats", 79)
    auc_horizon = _integer(config, "h14.auc_horizon", 128)
    threshold = _float(config, "h14.pure_threshold", 0.90)
    margin = _float(config, "h14.pure_margin", 0.10)

    # These panels are common to every trajectory for a seed.  Their generator
    # offsets match E15, making the update-45 manipulation check commensurate.
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
    phase_b = make_handoff_phase_b(
        n_train,
        q_q=q_q,
        k_q=k_q,
        k_y=k_y,
        seed=80_000_000 + seed,
        max_k_q=max_k_q,
        max_k_y=max_k_y,
        state_dim=state_dim,
    )
    phase_b_audit = audit_handoff_phase_b(phase_b, expected_q_q=q_q)
    phase_b_seed = 90_000_000 + seed
    phase_b_sampler = static_sampler_digest(
        len(phase_b),
        batch_size=batch_size,
        steps=phase_b_steps,
        seed=phase_b_seed,
        shuffle=bool(get_path(config, "train.shuffle", True)),
    )
    phase_b_feature_names = phase_b.feature_names(
        max_k=max_k_y, include_state=True
    )
    for name, batch in (("probe_train", probe_train), ("probe_eval", probe_eval)):
        if batch.feature_names(max_k=max_k_y, include_state=True) != phase_b_feature_names:
            raise RuntimeError(f"H14 {name} interface differs from phase B")

    records: list[dict[str, Any]] = []
    phase_a_snapshots: dict[int, dict[str, Any]] = {}
    phase_b_snapshots: dict[int, dict[str, Any]] = {}
    prefix: dict[str, Any] | None = None
    phase_a_training: TrainingResult | None = None
    sham_audit: dict[str, Any] | None = None
    model_report: dict[str, Any]
    started = time.perf_counter()

    def observe_phase_a(model: GoalMLP, step: int, samples_seen: int) -> None:
        phase_a_snapshots[step] = _compact_snapshot(
            model,
            probe_train=probe_train,
            probe_eval=probe_eval,
            config=config,
            seed=seed,
            probe_alpha=probe_alpha,
            max_k_y=max_k_y,
            records=records,
            condition=condition,
            stage="phase_a",
            local_step=step,
            global_step=step,
            examples_seen=samples_seen,
        )

    if trajectory.startswith("independent_") or trajectory.startswith("nested_"):
        overlap = trajectory.split("_", 1)[0]
        model, phase_a_training, model_report, prefix = _historical_prefix(
            overlap=overlap,
            config=config,
            seed=seed,
            q_p=q_p,
            q_q=q_q,
            k_q=k_q,
            k_y=k_y,
            max_k_q=max_k_q,
            max_k_y=max_k_y,
            state_dim=state_dim,
            n_train=n_train,
            n_validation=n_validation,
            n_eval=n_eval,
            phase_a_steps=phase_a_steps,
            eligibility_steps=eligibility_steps,
            observer=observe_phase_a,
        )
        _optimizer_records(
            phase_a_training,
            records,
            stage="phase_a_optimization",
            global_offset=0,
            example_offset=0,
            condition=condition,
            n=n_train,
        )
        carry = trajectory.endswith("_carry")
        phase_b_optimizer = (
            phase_a_training.optimizer if carry else _new_optimizer(model, config)
        )
        optimizer_source = "phase_a_carry" if carry else "fresh_reset"
        phase_a_kind = f"{overlap}_history"
    elif trajectory == "sham":
        sham_batch = make_compute_sham(
            n=n_train,
            repeats=sham_repeats,
            k_q=k_q,
            k_y=k_y,
            seed=81_000_000 + seed,
            control_seed=control_seed,
            max_k_q=max_k_q,
            max_k_y=max_k_y,
            state_dim=state_dim,
        )
        sham_audit = audit_compute_sham(sham_batch)
        if sham_batch.feature_names(max_k=max_k_y, include_state=True) != phase_b_feature_names:
            raise RuntimeError("H14 sham interface differs from phase B")
        built_model, model_report = build_model(sham_batch, config, seed)
        if not isinstance(built_model, GoalMLP):
            raise TypeError("H14 requires an unwrapped GoalMLP")
        model = built_model
        sham_initial_hash = stable_state_digest(model.state_dict())
        phase_a_training = _train_prefix(
            model,
            sham_batch,
            config,
            seed=81_500_000 + seed,
            steps=phase_a_steps,
            log_steps=eligibility_steps,
        )
        prefix = {
            "overlap": None,
            "hashes": {
                "initial_model": sham_initial_hash,
                "observed_final_model": stable_state_digest(
                    phase_a_training.final_model_state
                ),
                "observed_final_optimizer": stable_state_digest(
                    phase_a_training.optimizer_state
                ),
                "phase_a_batch": sham_audit["batch_digest"],
                "phase_a_sampler": static_sampler_digest(
                    len(sham_batch),
                    batch_size=batch_size,
                    steps=phase_a_steps,
                    seed=81_500_000 + seed,
                    shuffle=bool(get_path(config, "train.shuffle", True)),
                ),
            },
            "checks": {"sham_data_null": True},
            "samples_seen": phase_a_training.samples_seen,
            "optimizer_steps": phase_a_training.optimizer_steps,
        }
        _optimizer_records(
            phase_a_training,
            records,
            stage="phase_a_sham_optimization",
            global_offset=0,
            example_offset=0,
            condition=condition,
            n=n_train,
        )
        phase_b_optimizer = _new_optimizer(model, config)
        optimizer_source = "fresh_reset"
        phase_a_kind = "generic_compute_sham"
    else:
        built_model, model_report = build_model(phase_b, config, seed)
        if not isinstance(built_model, GoalMLP):
            raise TypeError("H14 requires an unwrapped GoalMLP")
        model = built_model
        prefix = {
            "overlap": None,
            "hashes": {"initial_model": stable_state_digest(model.state_dict())},
            "checks": {"scratch_initialization": True},
            "samples_seen": 0,
            "optimizer_steps": 0,
        }
        phase_b_optimizer = _new_optimizer(model, config)
        optimizer_source = "fresh_scratch"
        phase_a_kind = "scratch"

    historical = phase_a_kind in {"independent_history", "nested_history"}
    if prefix is None:  # pragma: no cover - every validated trajectory sets it
        raise RuntimeError("H14 trajectory produced no prefix audit")
    if model_report["input_dim"] != len(phase_b_feature_names):
        raise RuntimeError("H14 model input width differs from phase B")
    if historical:
        if tuple(sorted(phase_a_snapshots)) != eligibility_steps:
            raise RuntimeError("H14 did not observe both registered prefix checkpoints")
        prefix["hashes"]["phase_a_snapshot_metrics"] = stable_state_digest(
            phase_a_snapshots
        )
        prefix["checks"]["registered_snapshot_steps_equal"] = True
    phase_b_initial_model_hash = stable_state_digest(model.state_dict())
    optimizer_entry = _optimizer_entry_audit(phase_b_optimizer)
    if optimizer_source == "phase_a_carry":
        if (
            optimizer_entry["state_entry_count"] == 0
            or optimizer_entry["adam_step_min"] != phase_a_steps
            or optimizer_entry["adam_step_max"] != phase_a_steps
        ):
            raise RuntimeError("H14 carry optimizer lacks the exact phase-A state")
    elif optimizer_entry["state_entry_count"] != 0:
        raise RuntimeError("H14 reset/fresh optimizer must be empty at phase-B entry")
    phase_b_initial_optimizer_hash = optimizer_entry["actual_state_digest"]
    phase_a_examples = phase_a_training.samples_seen if phase_a_training else 0

    # Local update zero is directly measured, not copied from a prefix summary.
    phase_b_snapshots[0] = _compact_snapshot(
        model,
        probe_train=probe_train,
        probe_eval=probe_eval,
        config=config,
        seed=seed,
        probe_alpha=probe_alpha,
        max_k_y=max_k_y,
        records=records,
        condition=condition,
        stage="phase_b",
        local_step=0,
        global_step=phase_a_steps,
        examples_seen=phase_a_examples,
    )

    def phase_b_callback(snapshot: TrainingSnapshot) -> None:
        if not isinstance(snapshot.model, GoalMLP):
            raise TypeError("H14 snapshot observer requires GoalMLP")
        phase_b_snapshots[snapshot.step] = _compact_snapshot(
            snapshot.model,
            probe_train=probe_train,
            probe_eval=probe_eval,
            config=config,
            seed=seed,
            probe_alpha=probe_alpha,
            max_k_y=max_k_y,
            records=records,
            condition=condition,
            stage="phase_b",
            local_step=snapshot.step,
            global_step=phase_a_steps + snapshot.step,
            examples_seen=phase_a_examples + snapshot.record.samples_seen,
        )

    phase_b_training = train_clean_sft(
        model,
        batch_for_training(phase_b, config),
        _sft_config(
            config,
            seed=phase_b_seed,
            steps=phase_b_steps,
            log_steps=trained_phase_b_steps,
            reset_optimizer=False,
        ),
        optimizer=phase_b_optimizer,
        callback=phase_b_callback,
    )
    _optimizer_records(
        phase_b_training,
        records,
        stage="phase_b_optimization",
        global_offset=phase_a_steps,
        example_offset=phase_a_examples,
        condition=condition,
        n=n_train,
    )
    if tuple(sorted(phase_b_snapshots)) != tuple(phase_b_checkpoints):
        raise RuntimeError(
            "H14 did not directly observe every registered phase-B checkpoint"
        )
    if phase_b_steps >= 128 and 128 not in phase_b_snapshots:
        raise RuntimeError("H14 requires a directly observed local checkpoint 128")
    expected_phase_b_samples = phase_b_steps * batch_size
    if phase_b_training.samples_seen != expected_phase_b_samples:
        raise RuntimeError("H14 phase B contained a short or missing minibatch")
    if phase_a_training is not None and phase_a_training.samples_seen != (
        phase_a_steps * batch_size
    ):
        raise RuntimeError("H14 phase A contained a short or missing minibatch")

    eligibility = {
        str(step): pure_control(
            phase_a_snapshots[step]["behavior"],
            phase_a_snapshots[step]["causal"],
            "P",
            threshold=threshold,
            margin=margin,
        )
        for step in eligibility_steps
        if step in phase_a_snapshots
    }
    eligible = bool(
        historical
        and len(eligibility) == len(eligibility_steps)
        and all(eligibility.values())
    )
    phase_b_event_input = {
        step: {
            "behavior": snapshot["behavior"],
            "causal": snapshot["causal"],
        }
        for step, snapshot in phase_b_snapshots.items()
    }
    events = {
        goal: persistent_handoff(
            phase_b_event_input,
            goal,
            threshold=threshold,
            margin=margin,
        )
        for goal in ("P", "Q", "Y")
    }
    observed_non_p = [
        (int(events[goal]["confirmation_step"]), goal)
        for goal in ("Q", "Y")
        if events[goal]["observed"]
    ]
    first_non_p = (
        {"goal": min(observed_non_p)[1], "confirmation_step": min(observed_non_p)[0]}
        if observed_non_p
        else {"goal": None, "confirmation_step": None}
    )
    auc = {
        goal: normalized_control_auc(
            phase_b_event_input, goal, horizon=auc_horizon
        )
        for goal in ("P", "Q", "Y")
    }
    wall_seconds = time.perf_counter() - started
    total_examples = phase_a_examples + phase_b_training.samples_seen

    summary = {
        "hypothesis": "h14",
        "seed": seed,
        "condition": condition,
        "trajectory": trajectory,
        "design_status": "adaptive_posthoc_winner_knockout",
        "phase_a_kind": phase_a_kind,
        "optimizer_transition": (
            "carry" if trajectory.endswith("_carry") else "reset_or_fresh"
        ),
        "optimizer_transition_audit": {
            "source": optimizer_source,
            "provided_optimizer": True,
            "low_level_reset_optimizer_flag": False,
            "semantic_reset_implemented_by_fresh_object": optimizer_source
            != "phase_a_carry",
            **optimizer_entry,
        },
        "eligibility": {
            "historical_arm": historical,
            "per_checkpoint": eligibility,
            "eligible_both_registered_checkpoints": eligible if historical else None,
            "threshold": threshold,
            "margin": margin,
            "primary_analysis_rule": "intersection eligible across both histories",
            "minimum_intersection_size": 15,
        },
        "phase_a_snapshots": {
            str(step): value for step, value in sorted(phase_a_snapshots.items())
        },
        "phase_b_snapshots": {
            str(step): value for step, value in sorted(phase_b_snapshots.items())
        },
        "phase_b_local_zero": phase_b_snapshots[0],
        "final": phase_b_snapshots[phase_b_steps],
        "outcomes": {
            "normalized_control_auc_through_direct_checkpoint": auc,
            "auc_horizon": auc_horizon,
            "persistent_control_events": events,
            "first_non_p_controlled_goal": first_non_p,
        },
        "data": {
            "n_train": n_train,
            "n_validation": n_validation,
            "n_eval": n_eval,
            "q_p_phase_a": q_p,
            "q_q": q_q,
            "k_q": k_q,
            "k_y": k_y,
            "max_k_q": max_k_q,
            "max_k_y": max_k_y,
            "state_dim": state_dim,
            "phase_b": phase_b_audit,
            "phase_b_batch_digest": phase_b_audit["batch_digest"],
            "phase_b_sampler_digest": phase_b_sampler,
            "phase_b_pairing_verified": True,
            "phase_b_all_qr_codewords_represented": True,
            "sham": sham_audit,
            "probe_train_digest": semantic_batch_digest(probe_train),
            "probe_eval_digest": semantic_batch_digest(probe_eval),
            "probe_splits_disjoint": not bool(
                set(np.asarray(probe_train.sample_id).tolist())
                & set(np.asarray(probe_eval.sample_id).tolist())
            ),
        },
        "replay": prefix,
        "hashes": {
            "phase_b_initial_model": phase_b_initial_model_hash,
            "phase_b_initial_optimizer": phase_b_initial_optimizer_hash,
            "phase_b_final_model": stable_state_digest(
                phase_b_training.final_model_state
            ),
            "phase_b_final_optimizer": stable_state_digest(
                phase_b_training.optimizer_state
            ),
        },
        "model": model_report,
        "measurement": {
            "probe_train_n": probe_train_n,
            "probe_eval_n": probe_eval_n,
            "probe_ridge": probe_alpha,
            "truth_table_control_seed": control_seed,
            "phase_b_checkpoints": list(phase_b_checkpoints),
            "direct_checkpoint_128": 128 in phase_b_snapshots,
            "directional_causal_normalization": (
                "(1 + E[g*(a-a_flip)/2]) / 2"
            ),
        },
        "training": {
            "batch_size": batch_size,
            "all_minibatches_full": True,
            "phase_a_steps": phase_a_training.optimizer_steps
            if phase_a_training is not None
            else 0,
            "phase_b_steps": phase_b_training.optimizer_steps,
            "phase_a_examples_seen": phase_a_examples,
            "phase_b_examples_seen": phase_b_training.samples_seen,
            "total_examples_seen": total_examples,
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


RUNNERS = {"h14": run_h14}


__all__ = ["RUNNERS", "TRAJECTORIES", "run_h14"]
