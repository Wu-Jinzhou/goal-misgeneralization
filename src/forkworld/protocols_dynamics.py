"""Instrumented temporal bridge for competition among three candidate goals.

E15/H12 deliberately replays a high-information subset of E12 with an identical
training interface.  Its additional evaluations are deterministic observers:
they never update the policy or consume the training random-number streams.
"""

from __future__ import annotations

import copy
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import numpy as np
import torch
from torch import nn

from .competing import (
    competing_causal_flip_batches,
    competing_rule_agreements,
    decode_competing_rules,
    make_competing_bundle,
    make_competing_factorial_dataset,
)
from .config import get_path
from .data import SemanticBatch
from .models import GoalMLP
from .multigoal_dynamics import (
    directional_causal_effects,
    evaluate_competing_probes,
    factorial_behavioral_structure,
)
from .protocols import (
    ProtocolResult,
    build_model,
    make_metric_records,
    predict_logits,
    seed_everything,
)
from .protocols_extensions import (
    _candidate_interventions,
    _candidate_metrics,
    _retarget_and_mask,
)
from .protocols_selection import _checkpoint_states, _fit


def _integer(config: Mapping[str, Any], path: str, default: int) -> int:
    value = get_path(config, path, default)
    if isinstance(value, bool):
        raise ValueError(f"{path} must be an integer")
    return int(value)


def _float(config: Mapping[str, Any], path: str, default: float) -> float:
    return float(get_path(config, path, default))


def _flatten_numeric(value: Any, prefix: str = "") -> dict[str, float | int]:
    """Flatten nested measurement dictionaries into stable metric names."""

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
    elif isinstance(value, (int, float, np.number)) and not isinstance(value, (bool, np.bool_)):
        result[prefix] = float(value)
    return result


def static_subset_exposure(
    subset: Sequence[bool] | np.ndarray,
    *,
    batch_size: int,
    steps: int,
    seed: int,
    shuffle: bool = True,
) -> dict[str, Any]:
    """Replay the static sampler and summarize exact subset exposure.

    The helper mirrors :class:`training._StaticBatcher` with its own CPU
    generator.  It therefore measures the realized rare-example exposure
    without consuming or changing the training sampler's random state.
    """

    mask = np.asarray(subset)
    if mask.ndim != 1 or len(mask) < 1:
        raise ValueError("subset must be a non-empty one-dimensional mask")
    if not np.all(np.isin(mask, (False, True, 0, 1))):
        raise ValueError("subset must contain only Boolean values")
    if isinstance(batch_size, bool) or int(batch_size) != batch_size or batch_size < 1:
        raise ValueError("batch_size must be a positive integer")
    if isinstance(steps, bool) or int(steps) != steps or steps < 0:
        raise ValueError("steps must be a non-negative integer")

    indicator = torch.as_tensor(mask.astype(np.int64, copy=False), dtype=torch.int64)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    order = torch.empty(0, dtype=torch.long)
    cursor = 0
    cumulative = 0
    cumulative_counts = {0: 0}
    unique_counts = {0: 0}
    seen = torch.zeros(len(indicator), dtype=torch.bool)
    subset_size = int(torch.sum(indicator).item())
    first_presentation_step: int | None = None
    all_unique_seen_step: int | None = 0 if subset_size == 0 else None
    for step in range(1, int(steps) + 1):
        if cursor >= len(order):
            order = (
                torch.randperm(len(indicator), generator=generator)
                if shuffle
                else torch.arange(len(indicator), dtype=torch.long)
            )
            cursor = 0
        stop = min(cursor + int(batch_size), len(order))
        selected = order[cursor:stop]
        step_presentations = int(torch.sum(indicator[selected]).item())
        cumulative += step_presentations
        if step_presentations and first_presentation_step is None:
            first_presentation_step = step
        seen[selected[indicator[selected] > 0]] = True
        unique_seen = int(torch.sum(seen & (indicator > 0)).item())
        if unique_seen == subset_size and all_unique_seen_step is None:
            all_unique_seen_step = step
        cursor = stop
        cumulative_counts[step] = cumulative
        unique_counts[step] = unique_seen
    return {
        "cumulative_by_step": cumulative_counts,
        "unique_by_step": unique_counts,
        "subset_size": subset_size,
        "first_presentation_step": first_presentation_step,
        "all_unique_seen_step": all_unique_seen_step,
    }


def cumulative_subset_presentations(
    subset: Sequence[bool] | np.ndarray,
    *,
    batch_size: int,
    steps: int,
    seed: int,
    shuffle: bool = True,
) -> dict[int, int]:
    """Return the cumulative component of :func:`static_subset_exposure`."""

    result = static_subset_exposure(
        subset,
        batch_size=batch_size,
        steps=steps,
        seed=seed,
        shuffle=shuffle,
    )
    return dict(result["cumulative_by_step"])


def _active_codeword_ids(batch: SemanticBatch) -> np.ndarray:
    k_q = int(batch.metadata["k_q"])
    k_y = int(batch.metadata["k_y"])
    names = (
        "P",
        *(f"Q_{index}" for index in range(1, k_q + 1)),
        *(f"R_{index}" for index in range(1, k_y + 1)),
    )
    rows = np.column_stack(
        [np.asarray(batch.channels[name], dtype=np.int8) for name in names]
    )
    bits = (rows > 0).astype(np.int64)
    weights = np.left_shift(np.int64(1), np.arange(len(names), dtype=np.int64))
    return np.asarray(bits @ weights, dtype=np.int64)


def q_only_codeword_partition(
    training: SemanticBatch,
    factorial: SemanticBatch,
) -> dict[str, Any]:
    """Partition factorial Q-only rows by active codewords observed in training."""

    if (
        int(training.metadata["k_q"]) != int(factorial.metadata["k_q"])
        or int(training.metadata["k_y"]) != int(factorial.metadata["k_y"])
    ):
        raise ValueError("training and factorial batches must share active code degrees")
    training_q_only = np.asarray(training.latents["q_wrong"], dtype=bool)
    training_ids = _active_codeword_ids(training)
    seen_ids = np.unique(training_ids[training_q_only])

    rules = decode_competing_rules(factorial)
    factorial_q_only = (rules["P"] == rules["target"]) & (
        rules["Q"] != rules["target"]
    )
    factorial_ids = _active_codeword_ids(factorial)
    possible_ids = np.unique(factorial_ids[factorial_q_only])
    if not set(seen_ids.tolist()) <= set(possible_ids.tolist()):
        raise RuntimeError("training Q-only codewords fall outside factorial Q-only support")
    seen = factorial_q_only & np.isin(factorial_ids, seen_ids)
    unseen = factorial_q_only & ~np.isin(factorial_ids, seen_ids)
    return {
        "q_only": factorial_q_only,
        "seen": seen,
        "unseen": unseen,
        "seen_codeword_ids": seen_ids,
        "possible_codeword_ids": possible_ids,
        "training_q_only_rows": int(np.sum(training_q_only)),
        "seen_codeword_count": len(seen_ids),
        "possible_codeword_count": len(possible_ids),
    }


def _masked_factorial_metrics(
    logits: np.ndarray,
    batch: SemanticBatch,
    mask: np.ndarray,
) -> dict[str, float | int]:
    selected = np.flatnonzero(np.asarray(mask, dtype=bool))
    if len(selected) == 0:
        return {"panel_rows": 0}
    selected_batch = batch.select(selected)
    selected_logits = np.asarray(logits, dtype=np.float64)[selected]
    values = competing_rule_agreements(selected_logits, selected_batch)
    margin = selected_logits[:, 1] - selected_logits[:, 0]
    target = np.asarray(selected_batch.y, dtype=np.float64)
    signed_margin = target * margin
    # sigmoid(signed_margin), evaluated without overflow
    target_probability = np.empty_like(signed_margin)
    positive = signed_margin >= 0.0
    target_probability[positive] = 1.0 / (1.0 + np.exp(-signed_margin[positive]))
    exponential = np.exp(signed_margin[~positive])
    target_probability[~positive] = exponential / (1.0 + exponential)
    values.update(
        {
            "panel_rows": len(selected),
            "mean_target_probability": float(np.mean(target_probability)),
            "mean_target_logit_margin": float(np.mean(signed_margin)),
        }
    )
    return values


def _append_records(
    records: list[dict[str, Any]],
    values: Mapping[str, float | int],
    *,
    split: str,
    step: int,
    examples_seen: int,
    condition: str,
    stage: str,
    hypothesis: str = "h12",
    intervention: str = "none",
    n: int = 0,
) -> None:
    payload = dict(values)
    payload["n"] = int(payload.get("n", n))
    records.extend(
        make_metric_records(
            payload,
            hypothesis=hypothesis,
            split=split,
            global_step=step,
            examples_seen=examples_seen,
            condition=condition,
            stage=stage,
            intervention=intervention,
        )
    )


def _calibration_builders(
    k_q: int,
    k_y: int,
) -> dict[str, Callable[[SemanticBatch], SemanticBatch]]:
    q_names = tuple(f"Q_{index}" for index in range(1, k_q + 1))
    y_names = tuple(f"R_{index}" for index in range(1, k_y + 1))
    return {
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


def run_h12(config: Mapping[str, Any], seed: int) -> ProtocolResult:
    """Run instrumented three-goal dynamics or its support intervention."""

    seed_everything(seed)
    hypothesis = str(get_path(config, "experiment.hypothesis", "h12")).lower()
    if hypothesis not in {"h12", "h13"}:
        raise ValueError("instrumented multi-goal dynamics requires h12 or h13")
    section = hypothesis
    label = hypothesis.upper()
    n_train = _integer(config, "data.n_train", 10_000)
    n_validation = _integer(config, "data.n_validation", 4_000)
    n_eval = _integer(config, "data.n_eval", 10_000)
    q_p = _float(config, f"{section}.q_p", 0.9)
    q_q = _float(config, f"{section}.q_q", 0.95)
    k_q = _integer(config, f"{section}.k_q", 2)
    k_y = _integer(config, f"{section}.k_y", 3)
    max_k_q = _integer(config, f"{section}.max_k_q", 3)
    max_k_y = _integer(config, f"{section}.max_k_y", 5)
    overlap = str(get_path(config, f"{section}.error_structure", "independent"))
    q_only_error_count = (
        _integer(config, "h13.q_only_error_count", 0) if hypothesis == "h13" else None
    )
    state_dim = _integer(config, "data.state_dim", 8)
    condition = str(get_path(config, "experiment.mode", "adaptive_multigoal_dynamics"))
    calibration_steps = _integer(config, f"{section}.calibration_steps", 2_048)
    competition_steps = _integer(config, f"{section}.competition_steps", 8_192)
    probe_train_n = _integer(config, f"{section}.probe_train_n", 2_048)
    probe_eval_n = _integer(config, f"{section}.probe_eval_n", 4_096)
    probe_alpha = _float(config, f"{section}.probe_ridge", 1e-3)
    control_seed = _integer(
        config, f"{section}.truth_table_control_seed", 1_500_450_271
    )
    bridge_step = _integer(config, f"{section}.bridge_step", 2_048)

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
        q_only_error_count=q_only_error_count,
        state_dim=state_dim,
    )
    batches = {"iid": bundle.iid, **dict(bundle.diagnostics)}
    builders = _calibration_builders(k_q, k_y)
    feature_names = bundle.train.feature_names(
        max_k=_integer(config, "data.max_k", max_k_y), include_state=True
    )

    # Calibration keeps E12's original horizon and checkpoint prefix even though
    # the competition model continues.  This also keeps the bridge self-contained.
    calibration_config = copy.deepcopy(dict(config))
    configured_points = get_path(config, "train.eval_steps", "log")
    if isinstance(configured_points, list):
        calibration_config["train"]["eval_steps"] = [
            int(point) for point in configured_points if int(point) <= calibration_steps
        ]
    calibration_models: dict[str, nn.Module] = {}
    calibration_reports: dict[str, dict[str, Any]] = {}
    calibration_fits: dict[str, Any] = {}
    for name, transform in builders.items():
        transformed_train = transform(bundle.train)
        evaluation = {
            f"{name}_calibration_{split}": transform(batch)
            for split, batch in batches.items()
        }
        if transformed_train.feature_names(
            max_k=_integer(config, "data.max_k", max_k_y), include_state=True
        ) != feature_names:
            raise RuntimeError(f"{label} {name} calibration changed the competition interface")
        model, report = build_model(transformed_train, calibration_config, seed)
        fit = _fit(
            model,
            transformed_train,
            evaluation,
            calibration_config,
            seed,
            hypothesis=hypothesis,
            condition=condition,
            stage=f"{name}_calibration",
            steps=calibration_steps,
        )
        calibration_models[name] = model
        calibration_reports[name] = report
        calibration_fits[name] = fit

    # Probe-fitting and held-out panels cover the same complete semantic support
    # but use disjoint IDs, state features, padding streams, and sample rows.
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
    if probe_train.feature_names(max_k=max_k_y, include_state=True) != feature_names:
        raise RuntimeError(f"{label} probe-training interface differs from competition")
    if probe_eval.feature_names(max_k=max_k_y, include_state=True) != feature_names:
        raise RuntimeError(f"{label} held-out probe interface differs from competition")

    competition_model, competition_report = build_model(bundle.train, config, seed)
    for name, report in calibration_reports.items():
        for field in ("input_dim", "total_parameters", "trainable_parameters"):
            if report[field] != competition_report[field]:
                raise RuntimeError(
                    f"{label} {name} calibration does not match competition {field}"
                )
    if not isinstance(competition_model, GoalMLP):
        raise TypeError(f"{label} currently requires an unwrapped GoalMLP")

    dynamics_records: list[dict[str, Any]] = []
    recorded_snapshots: dict[int, dict[str, Any]] = {}
    q_only_partition = (
        q_only_codeword_partition(bundle.train, probe_eval)
        if hypothesis == "h13"
        else {}
    )
    q_only_exposure: dict[str, Any] = {}
    q_only_presentations: dict[int, int] = {}
    q_only_unique_seen: dict[int, int] = {}
    observed_q_only_presentations: dict[int, int] = {}
    observed_q_only_unique_seen: dict[int, int] = {}

    if hypothesis == "h13":
        q_only_exposure = static_subset_exposure(
            np.asarray(bundle.train.latents["q_wrong"], dtype=bool),
            batch_size=_integer(config, "train.batch_size", 128),
            steps=competition_steps,
            seed=seed,
            shuffle=bool(get_path(config, "train.shuffle", True)),
        )
        q_only_presentations = dict(q_only_exposure["cumulative_by_step"])
        q_only_unique_seen = dict(q_only_exposure["unique_by_step"])
        support_values = {
            key: value
            for key, value in bundle.train.metadata.items()
            if key
            in {
                "p_error_count",
                "q_error_count",
                "both_error_count",
                "p_only_error_count",
                "q_only_error_count",
                "neither_error_count",
                "both_error_rate",
                "independence_expected_count",
                "overlap_excess_count",
                "error_phi",
                "requested_q_only_error_count",
            }
            and value is not None
        }
        support_values.update(
            {
                "q_only_seen_active_codewords": int(
                    q_only_partition["seen_codeword_count"]
                ),
                "q_only_possible_active_codewords": int(
                    q_only_partition["possible_codeword_count"]
                ),
                "q_only_active_codeword_coverage": (
                    float(q_only_partition["seen_codeword_count"])
                    / float(q_only_partition["possible_codeword_count"])
                ),
            }
        )
        _append_records(
            dynamics_records,
            support_values,
            split="competition_train",
            step=0,
            examples_seen=0,
            condition=condition,
            stage="support_completion_design",
            hypothesis=hypothesis,
            n=len(bundle.train),
        )

    def observe(model: nn.Module, step: int, examples_seen: int) -> None:
        if not isinstance(model, GoalMLP):
            raise TypeError(f"{label} snapshot observer requires GoalMLP")

        if hypothesis == "h13":
            presentation_count = q_only_presentations[step]
            unique_seen = q_only_unique_seen[step]
            observed_q_only_presentations[step] = presentation_count
            observed_q_only_unique_seen[step] = unique_seen
            _append_records(
                dynamics_records,
                {
                    "cumulative_q_only_presentations": presentation_count,
                    "unique_q_only_rows_seen": unique_seen,
                    "q_only_training_rows": int(
                        np.sum(np.asarray(bundle.train.latents["q_wrong"], dtype=bool))
                    ),
                    "q_only_presentation_fraction": (
                        presentation_count / examples_seen if examples_seen else 0.0
                    ),
                },
                split="competition_train",
                step=step,
                examples_seen=examples_seen,
                condition=condition,
                stage="support_completion_exposure",
                hypothesis=hypothesis,
                n=len(bundle.train),
            )

        diagnostic_values: dict[str, Any] = {}
        for split, batch in batches.items():
            values = _candidate_metrics(model, batch, config)
            diagnostic_values[split] = values
            _append_records(
                dynamics_records,
                values,
                split=f"competition_{split}",
                step=step,
                examples_seen=examples_seen,
                condition=condition,
                stage="competition_candidates",
                hypothesis=hypothesis,
                n=len(batch),
            )

        factorial_logits = predict_logits(model, probe_eval, config)
        factorial_candidates = competing_rule_agreements(factorial_logits, probe_eval)
        factorial_structure = factorial_behavioral_structure(factorial_logits, probe_eval)
        if hypothesis == "h13":
            for panel_name, panel_mask in (
                ("all", q_only_partition["q_only"]),
                ("seen_codewords", q_only_partition["seen"]),
                ("unseen_codewords", q_only_partition["unseen"]),
            ):
                values = _masked_factorial_metrics(
                    factorial_logits,
                    probe_eval,
                    np.asarray(panel_mask, dtype=bool),
                )
                _append_records(
                    dynamics_records,
                    values,
                    split=f"factorial_q_only_{panel_name}",
                    step=step,
                    examples_seen=examples_seen,
                    condition=condition,
                    stage="support_completion_generalization",
                    hypothesis=hypothesis,
                    n=int(np.sum(np.asarray(panel_mask, dtype=bool))),
                )
        _append_records(
            dynamics_records,
            factorial_candidates,
            split="factorial_eval",
            step=step,
            examples_seen=examples_seen,
            condition=condition,
            stage="competition_behavior",
            hypothesis=hypothesis,
            n=len(probe_eval),
        )
        _append_records(
            dynamics_records,
            _flatten_numeric(factorial_structure),
            split="factorial_eval",
            step=step,
            examples_seen=examples_seen,
            condition=condition,
            stage="competition_truth_table",
            hypothesis=hypothesis,
            n=len(probe_eval),
        )

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
        _append_records(
            dynamics_records,
            _flatten_numeric(probes),
            split="factorial_probe",
            step=step,
            examples_seen=examples_seen,
            condition=condition,
            stage="competition_probe",
            hypothesis=hypothesis,
            n=len(probe_eval),
        )

        changed_logits = {
            name: predict_logits(model, changed, config)
            for name, changed in competing_causal_flip_batches(probe_eval).items()
        }
        causal = directional_causal_effects(factorial_logits, changed_logits, probe_eval)
        for section in ("per_intervention", "family_means"):
            members = causal.get(section, {}) if isinstance(causal, Mapping) else {}
            if not isinstance(members, Mapping):
                continue
            for intervention, values in members.items():
                if not isinstance(values, Mapping):
                    continue
                _append_records(
                    dynamics_records,
                    _flatten_numeric(values),
                    split="factorial_eval",
                    step=step,
                    examples_seen=examples_seen,
                    condition=condition,
                    stage="competition_causal",
                    hypothesis=hypothesis,
                    intervention=str(intervention),
                    n=len(probe_eval),
                )

        bridge_interventions: dict[str, Any] | None = None
        if step in {bridge_step, competition_steps}:
            bridge_interventions = {
                split: _candidate_interventions(model, batch, config)
                for split, batch in bundle.diagnostics.items()
            }
            for split, effects in bridge_interventions.items():
                for intervention, values in effects.items():
                    _append_records(
                        dynamics_records,
                        values,
                        split=f"competition_{split}",
                        step=step,
                        examples_seen=examples_seen,
                        condition=condition,
                        stage="competition_bridge",
                        hypothesis=hypothesis,
                        intervention=intervention,
                        n=len(bundle.diagnostics[split]),
                    )

        if step in {0, bridge_step, competition_steps}:
            recorded_snapshots[step] = {
                "diagnostics": diagnostic_values,
                "factorial_candidates": factorial_candidates,
                "factorial_structure": factorial_structure,
                "probes": probes,
                "causal": causal,
                "diagnostic_interventions": bridge_interventions,
            }

    competition = _fit(
        competition_model,
        bundle.train,
        {f"competition_{split}": batch for split, batch in batches.items()},
        config,
        seed,
        hypothesis=hypothesis,
        condition=condition,
        stage="competition",
        steps=competition_steps,
        snapshot_observer=observe,
    )

    decoder_metric = {"P": "rho_p", "Q": "rho_q", "Y": "rho_y_code"}
    final_calibration: dict[str, dict[str, dict[str, float | int]]] = {}
    for name, model in calibration_models.items():
        per_split: dict[str, dict[str, float | int]] = {}
        for split, batch in batches.items():
            values = _candidate_metrics(model, builders[name](batch), config)
            values["decoder_accuracy"] = float(values[decoder_metric[name]])
            per_split[split] = values
        final_calibration[name] = per_split

    metrics = [
        record
        for fit in calibration_fits.values()
        for record in fit.metrics
    ] + competition.metrics + dynamics_records
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
        "hypothesis": hypothesis,
        "seed": seed,
        "condition": condition,
        "design_status": (
            "adaptive_posthoc_support_completion"
            if hypothesis == "h13"
            else "adaptive_posthoc_with_fresh_seed_replication"
        ),
        "seed_cohort": (
            "e15_bridge_seed" if hypothesis == "h13" else "bridge"
        )
        if seed in {11, 23, 37, 41, 53, 67, 71, 83, 97, 101}
        else ("e15_fresh_seed" if hypothesis == "h13" else "fresh"),
        "final": {
            "calibration": final_calibration,
            "dynamics": recorded_snapshots.get(competition_steps),
        },
        "bridge": {
            "step": bridge_step,
            "available": bridge_step in recorded_snapshots,
            "dynamics": recorded_snapshots.get(bridge_step),
            "validation_rule": (
                "descriptive temporal landmark; no archived h13 state"
                if hypothesis == "h13"
                else "exact archived discrete diagnostics and intervention summaries"
            ),
        },
        "initial": recorded_snapshots.get(0),
        "model": {"calibration": calibration_reports, "competition": competition_report},
        "data": {
            "n_train": n_train,
            "n_validation": n_validation,
            "n_eval_per_legacy_panel": n_eval,
            "probe_train_n": probe_train_n,
            "probe_eval_n": probe_eval_n,
            "q_p": q_p,
            "q_q": q_q,
            "k_q": k_q,
            "k_y": k_y,
            "max_k_q": max_k_q,
            "max_k_y": max_k_y,
            "state_dim": state_dim,
            "error_structure": overlap,
            "q_only_error_count": q_only_error_count,
            "support_intervention_scope": (
                "training_only" if hypothesis == "h13" else "none"
            ),
            "q_only_codeword_coverage": (
                {
                    "training_rows": int(q_only_partition["training_q_only_rows"]),
                    "seen_active_codewords": int(
                        q_only_partition["seen_codeword_count"]
                    ),
                    "possible_active_codewords": int(
                        q_only_partition["possible_codeword_count"]
                    ),
                    "factorial_seen_rows": int(
                        np.sum(np.asarray(q_only_partition["seen"], dtype=bool))
                    ),
                    "factorial_unseen_rows": int(
                        np.sum(np.asarray(q_only_partition["unseen"], dtype=bool))
                    ),
                }
                if hypothesis == "h13"
                else None
            ),
            "e15_reference": (
                {
                    "nested_exact_data_replay_candidate": q_only_error_count == 0,
                    "independent_cell_count_match": int(
                        bundle.train.metadata["both_error_count"]
                    )
                    == round(
                        float(bundle.train.metadata["independence_expected_count"])
                    ),
                    "independent_row_allocation_match_claimed": False,
                    "audit_rule": (
                        "m=0 metrics must match archived E15 nested; "
                        "independence comparison is paired-descriptive only"
                    ),
                }
                if hypothesis == "h13"
                else None
            ),
            "factorial_candidate_counts_train": {
                str(key): int(value)
                for key, value in zip(
                    *np.unique(np.asarray(probe_train.latents["candidate_tuple_id"]), return_counts=True),
                    strict=True,
                )
            },
            "factorial_candidate_counts_eval": {
                str(key): int(value)
                for key, value in zip(
                    *np.unique(np.asarray(probe_eval.latents["candidate_tuple_id"]), return_counts=True),
                    strict=True,
                )
            },
            "truth_table_control_seed": control_seed,
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
            "iid_overlap": {
                key: bundle.iid.metadata[key]
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
            "factorial_interface_matches": True,
            "probe_splits_disjoint": not bool(
                set(np.asarray(probe_train.sample_id).tolist())
                & set(np.asarray(probe_eval.sample_id).tolist())
            ),
        },
        "measurement": {
            "ridge_alpha": probe_alpha,
            "probe_standardization": "fit_split_only",
            "probe_layers": ["raw", "first_hidden", "final_hidden"],
            "probe_controls": ["permuted_labels", "orthogonal_truth_table"],
            "directional_causal_normalization": "(1 + E[g*(a-a_flip)/2]) / 2",
            "checkpoint_count_including_initialization": len(recorded_snapshots)
            + len(
                {
                    int(record["global_step"])
                    for record in dynamics_records
                    if record.get("stage") == "competition_probe"
                }
                - set(recorded_snapshots)
            ),
        },
        "training": {
            "calibration_steps_each": calibration_steps,
            "competition_steps": competition.training.optimizer_steps,
            "examples_seen": total_examples,
            "wall_seconds": total_seconds,
            "q_only_presentations_by_checkpoint": {
                str(step): count
                for step, count in sorted(observed_q_only_presentations.items())
            },
            "unique_q_only_rows_seen_by_checkpoint": {
                str(step): count
                for step, count in sorted(observed_q_only_unique_seen.items())
            },
            "first_q_only_presentation_step": q_only_exposure.get(
                "first_presentation_step"
            ),
            "all_unique_q_only_seen_step": q_only_exposure.get(
                "all_unique_seen_step"
            ),
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


run_h13 = run_h12


RUNNERS = {"h12": run_h12, "h13": run_h13}


__all__ = [
    "RUNNERS",
    "cumulative_subset_presentations",
    "run_h12",
    "run_h13",
    "static_subset_exposure",
]
