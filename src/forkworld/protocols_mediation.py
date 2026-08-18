"""Active-Q input-column intervention protocol (E19/H16).

Each artifact reconstructs both paired E17 histories, derives the complete
six-branch edit family, and only then selects its configured branch.  The
engineering-pilot path returns immediately after the post-edit manipulation
measurement: in particular, it never constructs, hashes, or samples phase B.

Full runs discard all phase-A optimizer state and send the selected edited
model through the exact E17 informational P knockout.  Consequently the only
within-history treatment difference at phase-B entry is the frozen first-layer
column intervention; examples, sampler order, optimizer state, and evaluation
panels are paired exactly.  This protocol identifies effects of those columns,
not mediation by every representation or circuit that can encode Q.
"""

from __future__ import annotations

import math
import time
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from .competing import (
    competing_causal_flip_batches,
    competing_rule_agreements,
    decode_competing_rules,
    make_competing_factorial_dataset,
)
from .config import get_path
from .data import SemanticBatch
from .handoff import (
    audit_handoff_phase_b,
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
from .protocols_handoff import (
    _historical_prefix,
    _new_optimizer,
    _optimizer_entry_audit,
    _sft_config,
)
from .q_pathway import (
    Q_PATHWAY_BRANCHES,
    audit_q_pathway_branches,
    make_q_pathway_branches,
    preactivation_edit_effects,
)
from .training import TrainingResult, TrainingSnapshot, train_clean_sft


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
            hypothesis="h16",
            split=split,
            global_step=global_step,
            stage=stage,
            stage_step=local_step,
            examples_seen=examples_seen,
            condition=condition,
            intervention=intervention,
        )
    )


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
    """Measure behavior, causal control, structure, and probes at one state."""

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
        goal: float(family_means[goal]["causal_score"])
        for goal in ("P", "Q", "Y")
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


def _eligibility(
    snapshots: Mapping[int, Mapping[str, Any]],
    steps: Sequence[int],
    *,
    threshold: float,
    margin: float,
) -> dict[str, Any]:
    per_checkpoint = {
        str(step): pure_control(
            snapshots[step]["behavior"],
            snapshots[step]["causal"],
            "P",
            threshold=threshold,
            margin=margin,
        )
        for step in steps
    }
    return {
        "per_checkpoint": per_checkpoint,
        "eligible_both_registered_checkpoints": bool(
            len(per_checkpoint) == len(steps) and all(per_checkpoint.values())
        ),
        "threshold": threshold,
        "margin": margin,
    }


def _base_summary(
    *,
    seed: int,
    branch: str,
    pilot_only: bool,
    condition: str,
    model_report: Mapping[str, Any],
    prefix_reports: Mapping[str, Any],
    prefix_snapshots: Mapping[str, Mapping[int, Mapping[str, Any]]],
    eligibility: Mapping[str, Any],
    postedit_snapshot: Mapping[str, Any],
    edit_audit: Mapping[str, Any],
    preactivation: Mapping[str, Any],
    branch_hashes: Mapping[str, str],
    donor_state_digests: Mapping[str, Any],
    initial_model_hash: str,
    probe_train: SemanticBatch,
    probe_eval: SemanticBatch,
    probe_alpha: float,
    control_seed: int,
    phase_a_steps: int,
    batch_size: int,
    started: float,
) -> dict[str, Any]:
    return {
        "hypothesis": "h16",
        "seed": seed,
        "condition": condition,
        "branch": branch,
        "pilot_only": pilot_only,
        "design_status": (
            "adaptive_posthoc_active_q_input_pathway_intervention_frozen_pre_outcome"
        ),
        "prefix_snapshots": {
            overlap: {str(step): value for step, value in sorted(values.items())}
            for overlap, values in prefix_snapshots.items()
        },
        "eligibility": eligibility,
        "postedit_snapshot": postedit_snapshot,
        "postedit_pure_p": pure_control(
            postedit_snapshot["behavior"],
            postedit_snapshot["causal"],
            "P",
            threshold=float(eligibility["threshold"]),
            margin=float(eligibility["margin"]),
        ),
        "edit": {
            "audit": dict(edit_audit),
            "preactivation_effects": dict(preactivation),
            "donor_state_digests": dict(donor_state_digests),
            "branch_model_hashes": dict(branch_hashes),
            "branch_set_digest": stable_state_digest(dict(branch_hashes)),
        },
        "replay": dict(prefix_reports),
        "hashes": {
            "initial_model": initial_model_hash,
            "independent_prefix_model": prefix_reports["independent"]["hashes"] [
                "observed_final_model"
            ],
            "nested_prefix_model": prefix_reports["nested"]["hashes"][
                "observed_final_model"
            ],
            "selected_postedit_model": branch_hashes[branch],
        },
        "data": {
            "phase_b_constructed": False,
            "probe_train_digest": semantic_batch_digest(probe_train),
            "probe_eval_digest": semantic_batch_digest(probe_eval),
            "probe_splits_disjoint": not bool(
                set(np.asarray(probe_train.sample_id).tolist())
                & set(np.asarray(probe_eval.sample_id).tolist())
            ),
        },
        "model": dict(model_report),
        "measurement": {
            "probe_train_n": len(probe_train),
            "probe_eval_n": len(probe_eval),
            "probe_ridge": probe_alpha,
            "truth_table_control_seed": control_seed,
        },
        "training": {
            "batch_size": batch_size,
            "phase_a_steps_per_history": phase_a_steps,
            "phase_a_examples_seen_per_history": phase_a_steps * batch_size,
            "phase_b_steps": 0,
            "phase_b_examples_seen": 0,
            "all_minibatches_full": True,
            "wall_seconds": time.perf_counter() - started,
        },
    }


def run_h16(config: Mapping[str, Any], seed: int) -> ProtocolResult:
    """Run one frozen active-Q input-column intervention branch."""

    if resolve_device(str(get_path(config, "run.device", "cpu"))).type != "cpu":
        raise ValueError("H16 exact paired prefix replay is CPU-only")
    seed_everything(seed)
    branch = str(get_path(config, "h16.branch", "independent_noop"))
    if branch not in Q_PATHWAY_BRANCHES:
        raise ValueError(f"h16.branch must be one of {Q_PATHWAY_BRANCHES}")
    pilot_only = bool(get_path(config, "h16.pilot_only", False))
    condition = f"active_q_input_pathway:{branch}"
    started = time.perf_counter()

    n_train = _integer(config, "data.n_train", 10_000)
    n_validation = _integer(config, "data.n_validation", 4_000)
    n_eval = _integer(config, "data.n_eval", 10_000)
    batch_size = _integer(config, "train.batch_size", 250)
    if n_train % batch_size:
        raise ValueError("H16 requires full, equal-size static minibatches")
    q_p = _float(config, "h16.q_p", 0.90)
    q_q = _float(config, "h16.q_q", 0.90)
    k_q = _integer(config, "h16.k_q", 2)
    k_y = _integer(config, "h16.k_y", 3)
    max_k_q = _integer(config, "h16.max_k_q", 3)
    max_k_y = _integer(config, "h16.max_k_y", 5)
    state_dim = _integer(config, "data.state_dim", 8)
    phase_a_steps = _integer(config, "h16.phase_a_steps", 45)
    eligibility_steps = _integer_sequence(config, "h16.eligibility_steps")
    probe_train_n = _integer(config, "h16.probe_train_n", 2_048)
    probe_eval_n = _integer(config, "h16.probe_eval_n", 4_096)
    probe_alpha = _float(config, "h16.probe_ridge", 1e-3)
    control_seed = _integer(
        config, "h16.truth_table_control_seed", 1_500_450_271
    )
    threshold = _float(config, "h16.pure_threshold", 0.90)
    margin = _float(config, "h16.pure_margin", 0.10)

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
    feature_names = probe_eval.feature_names(max_k=max_k_y, include_state=True)
    if probe_train.feature_names(max_k=max_k_y, include_state=True) != feature_names:
        raise RuntimeError("H16 probe interfaces differ")

    records: list[dict[str, Any]] = []
    prefix_snapshots: dict[str, dict[int, dict[str, Any]]] = {
        "independent": {},
        "nested": {},
    }
    prefix_models: dict[str, GoalMLP] = {}
    prefix_trainings: dict[str, TrainingResult] = {}
    prefix_model_reports: dict[str, dict[str, Any]] = {}
    prefix_reports: dict[str, dict[str, Any]] = {}

    for overlap in ("independent", "nested"):
        overlap_condition = f"{condition}:{overlap}_prefix"

        def observe_prefix(
            model: GoalMLP,
            step: int,
            samples_seen: int,
            *,
            _overlap: str = overlap,
            _condition: str = overlap_condition,
        ) -> None:
            prefix_snapshots[_overlap][step] = _compact_snapshot(
                model,
                probe_train=probe_train,
                probe_eval=probe_eval,
                config=config,
                seed=seed,
                probe_alpha=probe_alpha,
                max_k_y=max_k_y,
                records=records,
                condition=_condition,
                stage=f"{_overlap}_phase_a",
                local_step=step,
                global_step=step,
                examples_seen=samples_seen,
            )

        model, training, report, replay = _historical_prefix(
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
            observer=observe_prefix,
        )
        prefix_models[overlap] = model
        prefix_trainings[overlap] = training
        prefix_model_reports[overlap] = report
        prefix_reports[overlap] = replay
        _optimizer_records(
            training,
            records,
            stage=f"{overlap}_phase_a_optimization",
            global_offset=0,
            example_offset=0,
            condition=overlap_condition,
            n=n_train,
        )

    if any(tuple(sorted(values)) != eligibility_steps for values in prefix_snapshots.values()):
        raise RuntimeError("H16 did not observe every registered prefix checkpoint")
    if prefix_model_reports["independent"] != prefix_model_reports["nested"]:
        raise RuntimeError("H16 paired prefix model interfaces differ")

    initial_model, initial_report = build_model(probe_eval, config, seed)
    if not isinstance(initial_model, GoalMLP):
        raise TypeError("H16 requires an unwrapped GoalMLP")
    if initial_report != prefix_model_reports["independent"]:
        raise RuntimeError("H16 initial and historical model reports differ")
    initial_model_hash = stable_state_digest(initial_model.state_dict())
    for overlap in ("independent", "nested"):
        if prefix_reports[overlap]["hashes"]["initial_model"] != initial_model_hash:
            raise RuntimeError(f"H16 {overlap} prefix did not share initialization")
        if not all(prefix_reports[overlap]["checks"].values()):
            raise RuntimeError(f"H16 {overlap} prefix replay audit failed")

    branch_family = make_q_pathway_branches(
        initial_model,
        prefix_models["independent"],
        prefix_models["nested"],
        feature_names=feature_names,
    )
    edit_audit = audit_q_pathway_branches(
        branch_family,
        initial_model=initial_model,
        independent_model=prefix_models["independent"],
        nested_model=prefix_models["nested"],
    )
    preactivation = preactivation_edit_effects(
        branch_family,
        probe_eval,
        max_k=max_k_y,
        include_state=True,
    )
    if set(branch_family.models) != set(Q_PATHWAY_BRANCHES):
        raise RuntimeError("H16 edit constructor did not return the frozen six branches")
    branch_hashes = {
        name: stable_state_digest(model.state_dict())
        for name, model in branch_family.models.items()
    }
    model = branch_family.models[branch]
    postedit_snapshot = _compact_snapshot(
        model,
        probe_train=probe_train,
        probe_eval=probe_eval,
        config=config,
        seed=seed,
        probe_alpha=probe_alpha,
        max_k_y=max_k_y,
        records=records,
        condition=condition,
        stage="postedit",
        local_step=0,
        global_step=phase_a_steps,
        examples_seen=phase_a_steps * batch_size,
    )
    eligibility = {
        overlap: _eligibility(
            prefix_snapshots[overlap],
            eligibility_steps,
            threshold=threshold,
            margin=margin,
        )
        for overlap in ("independent", "nested")
    }
    eligibility["threshold"] = threshold
    eligibility["margin"] = margin
    eligibility["paired_intersection_eligible"] = bool(
        eligibility["independent"]["eligible_both_registered_checkpoints"]
        and eligibility["nested"]["eligible_both_registered_checkpoints"]
    )

    summary = _base_summary(
        seed=seed,
        branch=branch,
        pilot_only=pilot_only,
        condition=condition,
        model_report=initial_report,
        prefix_reports=prefix_reports,
        prefix_snapshots=prefix_snapshots,
        eligibility=eligibility,
        postedit_snapshot=postedit_snapshot,
        edit_audit=edit_audit,
        preactivation=preactivation,
        branch_hashes=branch_hashes,
        donor_state_digests=branch_family.donor_state_digests,
        initial_model_hash=initial_model_hash,
        probe_train=probe_train,
        probe_eval=probe_eval,
        probe_alpha=probe_alpha,
        control_seed=control_seed,
        phase_a_steps=phase_a_steps,
        batch_size=batch_size,
        started=started,
    )

    # This is intentionally before every phase-B setting read or constructor
    # call.  A manipulation-only pilot cannot leak a knockout outcome or even a
    # phase-B data digest into its artifacts.
    if pilot_only:
        return ProtocolResult(
            model=model,
            summary=summary,
            metrics=records,
            predictions=[],
            checkpoints={},
            evaluation_batch=probe_eval,
        )

    phase_b_steps = _integer(config, "h16.phase_b_steps", 1_024)
    phase_b_checkpoints = _integer_sequence(config, "h16.phase_b_checkpoints")
    if 0 not in phase_b_checkpoints:
        raise ValueError("h16.phase_b_checkpoints must include local update zero")
    trained_phase_b_steps = tuple(step for step in phase_b_checkpoints if step > 0)
    auc_horizon = _integer(config, "h16.auc_horizon", 128)

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
    if phase_b.feature_names(max_k=max_k_y, include_state=True) != feature_names:
        raise RuntimeError("H16 phase-B interface differs from the edit interface")
    phase_b_seed = 90_000_000 + seed
    phase_b_sampler = static_sampler_digest(
        len(phase_b),
        batch_size=batch_size,
        steps=phase_b_steps,
        seed=phase_b_seed,
        shuffle=bool(get_path(config, "train.shuffle", True)),
    )
    optimizer = _new_optimizer(model, config)
    optimizer_entry = _optimizer_entry_audit(optimizer)
    if optimizer_entry["state_entry_count"] != 0:
        raise RuntimeError("H16 phase B must start with an empty optimizer")

    phase_b_snapshots: dict[int, dict[str, Any]] = {0: postedit_snapshot}

    def observe_phase_b(snapshot: TrainingSnapshot) -> None:
        if not isinstance(snapshot.model, GoalMLP):
            raise TypeError("H16 snapshot observer requires GoalMLP")
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
            examples_seen=(phase_a_steps + snapshot.step) * batch_size,
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
        optimizer=optimizer,
        callback=observe_phase_b,
    )
    _optimizer_records(
        phase_b_training,
        records,
        stage="phase_b_optimization",
        global_offset=phase_a_steps,
        example_offset=phase_a_steps * batch_size,
        condition=condition,
        n=n_train,
    )
    if tuple(sorted(phase_b_snapshots)) != phase_b_checkpoints:
        raise RuntimeError("H16 did not directly observe every phase-B checkpoint")
    if auc_horizon not in phase_b_snapshots:
        raise RuntimeError("H16 AUC horizon must be a directly observed checkpoint")
    expected_phase_b_samples = phase_b_steps * batch_size
    if phase_b_training.samples_seen != expected_phase_b_samples:
        raise RuntimeError("H16 phase B contained a short or missing minibatch")

    event_input = {
        step: {
            "behavior": snapshot["behavior"],
            "causal": snapshot["causal"],
        }
        for step, snapshot in phase_b_snapshots.items()
    }
    auc = {
        goal: normalized_control_auc(event_input, goal, horizon=auc_horizon)
        for goal in ("P", "Q", "Y")
    }
    events = {
        goal: persistent_handoff(
            event_input,
            goal,
            threshold=threshold,
            margin=margin,
        )
        for goal in ("P", "Q", "Y")
    }

    summary["data"].update(
        {
            "phase_b_constructed": True,
            "phase_b": phase_b_audit,
            "phase_b_batch_digest": phase_b_audit["batch_digest"],
            "phase_b_sampler_digest": phase_b_sampler,
            "phase_b_pairing_verified": True,
        }
    )
    summary["hashes"].update(
        {
            "phase_b_initial_model": branch_hashes[branch],
            "phase_b_initial_optimizer": optimizer_entry["actual_state_digest"],
            "phase_b_final_model": stable_state_digest(
                phase_b_training.final_model_state
            ),
            "phase_b_final_optimizer": stable_state_digest(
                phase_b_training.optimizer_state
            ),
        }
    )
    summary["optimizer_transition_audit"] = {
        "source": "fresh_empty_after_edit",
        "semantic_reset_implemented_by_fresh_object": True,
        "provided_optimizer": True,
        "low_level_reset_optimizer_flag": False,
        **optimizer_entry,
    }
    summary["phase_b_snapshots"] = {
        str(step): value for step, value in sorted(phase_b_snapshots.items())
    }
    summary["phase_b_local_zero"] = phase_b_snapshots[0]
    summary["final"] = phase_b_snapshots[phase_b_steps]
    summary["outcomes"] = {
        "normalized_control_auc_through_direct_checkpoint": auc,
        "auc_horizon": auc_horizon,
        "persistent_control_events": events,
    }
    summary["measurement"].update(
        {
            "phase_b_checkpoints": list(phase_b_checkpoints),
            "direct_auc_horizon": auc_horizon in phase_b_snapshots,
            "directional_causal_normalization": (
                "(1 + E[g*(a-a_flip)/2]) / 2"
            ),
        }
    )
    summary["training"].update(
        {
            "phase_b_steps": phase_b_training.optimizer_steps,
            "phase_b_examples_seen": phase_b_training.samples_seen,
            "total_examples_seen_selected_trajectory": (
                phase_a_steps * batch_size + phase_b_training.samples_seen
            ),
            "wall_seconds": time.perf_counter() - started,
        }
    )
    return ProtocolResult(
        model=model,
        summary=summary,
        metrics=records,
        predictions=[],
        checkpoints={},
        evaluation_batch=probe_eval,
    )


RUNNERS = {"h16": run_h16}


__all__ = ["RUNNERS", "run_h16"]
