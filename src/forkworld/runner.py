"""Sweep execution, resumability, and artifact integration."""

from __future__ import annotations

import copy
import traceback
from collections.abc import Iterable, Mapping
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

from .artifacts import RunStore
from .config import ConfigError, expand_sweep, get_path, set_path, validate_config
from .navigation_support import ensure_navigators, evaluate_task_levels
from .protocols import make_metric_records, resolve_device, run_protocol


def smoke_config(config: Mapping[str, Any]) -> dict[str, Any]:
    """Select one scientifically valid cell and reduce only computational scale."""

    expanded = expand_sweep(config)
    value = copy.deepcopy(expanded[0])
    value["_smoke"] = True
    value["cases"] = []
    value["sweep"] = {}
    value["_sweep_values"] = {}
    value["run"]["seeds"] = [0]
    value["run"]["navigation_episodes"] = 4
    value["data"].update(
        {"n_train": 64, "n_validation": 64, "n_eval": 128}
    )
    # Preserve q only when exactly realizable at n=64.
    q = float(value["data"].get("q", 0.75))
    if abs(64 * (1.0 - q) - round(64 * (1.0 - q))) > 1e-9:
        value["data"]["q"] = 0.75
    value["train"].update({"steps": 16, "batch_size": 32, "eval_steps": "log"})
    value["evaluation"].update({"bootstrap_samples": 100, "save_predictions": True})
    value["navigator"].update(
        {"navigation_maps": 2, "epochs": 60, "target_accuracy": 0.9, "target_patience": 1}
    )
    hypothesis = str(value["experiment"]["hypothesis"])
    if hypothesis == "h4":
        value.setdefault("h4", {}).update({"n_conflict": 8, "unique_fraction": 0.5})
    elif hypothesis == "h5":
        if str(get_path(value, "experiment.mode", "primary")) == "entropy_timing":
            value.setdefault("h5", {}).update(
                {
                    "algorithm": "rl",
                    "rl_estimator": "actor_critic",
                    "nuisance_entropy": 0,
                    "phase_a_steps": 4,
                    "phase_b_steps": 12,
                }
            )
        else:
            value.setdefault("h5", {}).update(
                {"algorithm": "trajectory_sft", "nuisance_bits": 2, "nuisance_entropy": 2}
            )
    elif hypothesis == "h6":
        value.setdefault("h6", {}).update(
            {"algorithm": "clean_sft", "structure": "step", "location": "observation", "visits_per_state": 2}
        )
    elif hypothesis == "h7":
        value.setdefault("h7", {}).update(
            {"phase_a_max_steps": 32, "phase_b_steps": 16, "eval_every": 2, "transition_patience": 1}
        )
    elif hypothesis == "h8":
        value.setdefault("h8", {}).update({"n0": 64, "n1": 64, "stage2_steps": 16})
    elif hypothesis == "h10":
        value.setdefault("h10", {}).update(
            {
                "q_p": 0.75,
                "q_q": 0.75,
                "k_q": 2,
                "k_y": 2,
                "calibration_steps": 16,
                "competition_steps": 16,
            }
        )
    elif hypothesis == "h11":
        value.setdefault("h11", {}).update(
            {
                "route_depth": 1,
                "max_depth": 2,
                "base_steps": 16,
                "physical_rollouts": 4,
            }
        )
        value["data"].update({"q": 0.75, "k": 2, "max_k": 2})
    elif hypothesis == "h12":
        value.setdefault("h12", {}).update(
            {
                "q_p": 0.75,
                "q_q": 0.75,
                "k_q": 2,
                "k_y": 2,
                "max_k_q": 3,
                "max_k_y": 5,
                "calibration_steps": 16,
                "competition_steps": 16,
                "probe_train_n": 64,
                "probe_eval_n": 128,
                "bridge_step": 16,
            }
        )
    elif hypothesis == "h13":
        value.setdefault("h13", {}).update(
            {
                "q_p": 0.75,
                "q_q": 0.75,
                "k_q": 2,
                "k_y": 2,
                "max_k_q": 3,
                "max_k_y": 5,
                "error_structure": "nested",
                "q_only_error_count": 0,
                "calibration_steps": 16,
                "competition_steps": 16,
                "probe_train_n": 64,
                "probe_eval_n": 128,
                "bridge_step": 16,
            }
        )
    elif hypothesis == "h14":
        value["data"].update({"n_train": 256, "n_validation": 128, "n_eval": 256})
        value["train"].update(
            {
                "steps": 16,
                "batch_size": 64,
                "eval_steps": [1, 2, 3, 4, 5, 7, 8, 9, 13, 16],
            }
        )
        value.setdefault("h14", {}).update(
            {
                "q_p": 0.75,
                "q_q": 0.75,
                "k_q": 2,
                "k_y": 2,
                "max_k_q": 3,
                "max_k_y": 5,
                "phase_a_steps": 4,
                "phase_b_steps": 16,
                "eligibility_steps": [3, 4],
                "phase_b_checkpoints": [0, 1, 2, 3, 4, 5, 7, 8, 9, 13, 16],
                "probe_train_n": 128,
                "probe_eval_n": 256,
                "sham_source_repeats": 4,
                "auc_horizon": 8,
            }
        )
    elif hypothesis == "h15":
        value["data"].update(
            {
                "n_train": 256,
                "n_validation": 128,
                "n_eval": 256,
                "q": 0.75,
                "k": 2,
            }
        )
        value["train"].update(
            {
                "steps": 24,
                "batch_size": 32,
                "shuffle": False,
                "eval_steps": [12, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24],
            }
        )
        value.setdefault("h15", {}).update(
            {
                "q_p": 0.75,
                "q_q": 0.875,
                "k_q": 2,
                "k_y": 2,
                "max_k_q": 3,
                "max_k_y": 5,
                "prefix_a_repetitions": 2,
                "washout_a_repetitions": 1,
                "diagnostic_repetitions": 3,
                "prefix_steps": 12,
                "block_steps": 3,
                "washout_steps": 6,
                "total_steps": 24,
                "second_block_checkpoints": [1, 2, 3],
                "washout_checkpoints": [0, 1, 2, 3, 4, 5, 6],
                "auc_horizon": 3,
                "probe_train_n": 128,
                "probe_eval_n": 256,
                "late_stability_steps": [3, 5, 6],
            }
        )
    elif hypothesis == "h16":
        # Preserve the frozen 64 x 19 intervention surface and column indices;
        # only sample counts and phase durations are reduced for integration.
        value["data"].update(
            {
                "n_train": 256,
                "n_validation": 128,
                "n_eval": 256,
                "q": 0.75,
                "k": 2,
                "max_k": 5,
                "state_dim": 8,
            }
        )
        value["model"].update({"width": 64, "depth": 2})
        value["train"].update(
            {
                "steps": 16,
                "batch_size": 64,
                "eval_steps": [1, 2, 3, 4, 5, 7, 8, 9, 13, 16],
            }
        )
        value.setdefault("h16", {}).update(
            {
                "q_p": 0.75,
                "q_q": 0.75,
                "k_q": 2,
                "k_y": 2,
                "max_k_q": 3,
                "max_k_y": 5,
                "phase_a_steps": 4,
                "phase_b_steps": 16,
                "eligibility_steps": [3, 4],
                "phase_b_checkpoints": [0, 1, 2, 3, 4, 5, 7, 8, 9, 13, 16],
                "probe_train_n": 128,
                "probe_eval_n": 256,
                "auc_horizon": 8,
            }
        )
    elif hypothesis == "h17":
        # Keep all 96 weighted strata and the literal eight-column interface.
        # Scale only replicas and presentations: one complete atomic batch per
        # presentation, two presentations per block.
        pilot_only = bool(get_path(value, "h17.pilot_only", False))
        total_steps = 2 if pilot_only else 8
        value["run"].update(
            {"device": "cpu", "task_levels": ["choice"], "save_checkpoints": False}
        )
        value["evaluation"]["save_predictions"] = False
        value["data"].update(
            {
                "n_train": 288,
                "n_validation": 64,
                "n_eval": 64,
                "q": 0.5,
                "k": 3,
                "max_k": 3,
                "state_dim": 0,
            }
        )
        value["model"].update(
            {
                "width": 64,
                "depth": 2,
                "activation": "relu",
                "residual": False,
                "bias": True,
            }
        )
        value["train"].update(
            {
                "steps": total_steps,
                "batch_size": 288,
                "shuffle": False,
                "eval_steps": [1, 2],
            }
        )
        value.setdefault("h17", {}).update(
            {
                "rows_per_weight_unit": 3,
                "weighted_strata": 96,
                "examples_per_weight_unit_per_batch": 3,
                "batches_per_presentation": 1,
                "presentations": 2,
                "component_steps": 2,
                "washout_steps": 2,
                "total_steps": total_steps,
                "component_checkpoints": [0, 1, 2],
                "washout_checkpoints": [0, 1, 2],
                "pilot_late_steps": [0, 1, 2],
                "primary_auc_window": [1, 2],
            }
        )
    validate_config(value)
    return value


def planned_runs(
    config: Mapping[str, Any],
    *,
    smoke: bool = False,
    seed_override: Iterable[int] | None = None,
    shard_index: int = 0,
    shard_count: int = 1,
    max_runs: int | None = None,
) -> list[tuple[dict[str, Any], int]]:
    if shard_count < 1 or not 0 <= shard_index < shard_count:
        raise ValueError("shard_index must satisfy 0 <= index < shard_count")
    cells = [smoke_config(config)] if smoke else expand_sweep(config)
    pairs: list[tuple[dict[str, Any], int]] = []
    for cell in cells:
        seeds = list(seed_override) if seed_override is not None else list(get_path(cell, "run.seeds"))
        if (
            str(get_path(cell, "experiment.hypothesis", "")).lower() == "h16"
            and bool(get_path(cell, "h16.pilot_only", False))
            and not bool(get_path(cell, "_smoke", False))
            and seeds != list(get_path(cell, "h16.pilot_seeds", []))
        ):
            raise ConfigError(
                "H16 pilot_only runs must use exactly the frozen h16.pilot_seeds panel"
            )
        if (
            str(get_path(cell, "experiment.hypothesis", "")).lower() == "h17"
            and not bool(get_path(cell, "_smoke", False))
        ):
            pilot_only = bool(get_path(cell, "h17.pilot_only", False))
            expected = list(
                get_path(
                    cell,
                    "h17.pilot_seeds" if pilot_only else "h17.full_seeds",
                    [],
                )
            )
            if seeds != expected:
                panel = "pilot" if pilot_only else "full"
                raise ConfigError(
                    f"H17 {panel} runs must use exactly the frozen h17 {panel} seed panel"
                )
        pairs.extend((cell, int(seed)) for seed in seeds)
    pairs = [pair for index, pair in enumerate(pairs) if index % shard_count == shard_index]
    if max_runs is not None:
        if max_runs < 1:
            raise ValueError("max_runs must be positive")
        pairs = pairs[:max_runs]
    return pairs


def _serializable_checkpoint(value: Any) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    if is_dataclass(value):
        return asdict(value)
    if hasattr(value, "__dict__"):
        return dict(value.__dict__)
    return {"value": value}


def _terminal_training_counters(
    metrics: Iterable[Mapping[str, Any]], fallback_step: int
) -> tuple[int, int | None]:
    """Recover protocol-realized counters for final cross-level evaluation.

    Several phased protocols and H6's dataset-scaled evidence plan deliberately
    do not end at ``config.train.steps``.  Navigation records must therefore use
    the counters emitted by the protocol rather than relabeling every result
    with the shared configuration default.
    """

    steps: list[int] = []
    examples: list[int] = []
    for record in metrics:
        step = record.get("global_step")
        if isinstance(step, (int, float)) and not isinstance(step, bool):
            steps.append(int(step))
        seen = record.get("examples_seen")
        if isinstance(seen, (int, float)) and not isinstance(seen, bool):
            examples.append(int(seen))
    return (max(steps, default=int(fallback_step)), max(examples) if examples else None)


def execute_one(
    config: Mapping[str, Any],
    seed: int,
    output_root: str | Path,
    repo_root: str | Path,
    *,
    force: bool = False,
) -> dict[str, Any]:
    """Execute one sweep cell with one independent training seed."""

    store = RunStore(output_root, config, seed, Path(repo_root))
    if store.complete and bool(get_path(config, "run.resume", True)) and not force:
        return {"run_id": store.run_id, "status": "skipped", "path": str(store.path)}
    # Protocol checkpoints are audit artifacts, not incremental resumptions.
    # Any non-complete directory is restarted transactionally so stale JSONL
    # rows from an interrupted/failed attempt cannot contaminate the rerun.
    store.initialize(reset=force or store.path.exists())
    try:
        result = run_protocol(config, seed)
        if result.metrics:
            store.append_metrics(result.metrics)
        if bool(get_path(config, "evaluation.save_predictions", True)) and result.predictions:
            store.append_predictions(result.predictions)
        if bool(get_path(config, "run.save_checkpoints", True)):
            for name, checkpoint in result.checkpoints.items():
                store.save_checkpoint_payload(_serializable_checkpoint(checkpoint), str(name))
            store.save_checkpoint(result.model, "final", {"seed": seed})

        navigation = {}
        if result.evaluation_batch is not None:
            navigation = evaluate_task_levels(
                result.model,
                result.evaluation_batch,
                config,
                get_path(config, "_navigator_root"),
                seed=seed,
            )
            final_step, final_examples = _terminal_training_counters(
                result.metrics, int(get_path(config, "train.steps", 0))
            )
            for level, values in navigation.items():
                store.append_metrics(
                    make_metric_records(
                        values,
                        hypothesis=str(get_path(config, "experiment.hypothesis")),
                        split="conflict_eval",
                        global_step=final_step,
                        stage="final",
                        examples_seen=final_examples,
                        level=level,
                        condition=str(get_path(config, "experiment.mode", "primary")),
                    )
                )
        summary = dict(result.summary)
        summary["navigation"] = navigation
        store.finalize(summary)
        return {"run_id": store.run_id, "status": "complete", "path": str(store.path)}
    except Exception as error:
        store.fail(error)
        return {
            "run_id": store.run_id,
            "status": "failed",
            "path": str(store.path),
            "error": f"{type(error).__name__}: {error}",
            "traceback": traceback.format_exc(),
        }


def run_experiment(
    config: Mapping[str, Any],
    *,
    output_root: str | Path | None = None,
    device: str | None = None,
    jobs: int = 1,
    smoke: bool = False,
    seed_override: Iterable[int] | None = None,
    shard_index: int = 0,
    shard_count: int = 1,
    max_runs: int | None = None,
    force: bool = False,
    fail_fast: bool = False,
) -> list[dict[str, Any]]:
    """Run a full or sharded experiment plan, optionally with process parallelism."""

    effective = smoke_config(config) if smoke else copy.deepcopy(dict(config))
    root = Path(output_root or get_path(effective, "run.output_root", "artifacts")).resolve()
    chosen_device = str(resolve_device(device or str(get_path(effective, "run.device", "auto"))))
    set_path(effective, "run.output_root", str(root))
    set_path(effective, "run.device", chosen_device)
    navigator_root = ensure_navigators(effective, root, chosen_device)
    effective["_navigator_root"] = str(navigator_root) if navigator_root else None
    pairs = planned_runs(
        effective,
        smoke=False,
        seed_override=seed_override,
        shard_index=shard_index,
        shard_count=shard_count,
        max_runs=max_runs,
    )
    repo = Path(__file__).resolve().parents[2]
    if jobs < 1:
        raise ValueError("jobs must be positive")
    if chosen_device != "cpu" and jobs > 1:
        raise ValueError("jobs>1 is supported only on CPU; use cluster sharding for accelerator runs")

    results: list[dict[str, Any]] = []
    if jobs == 1:
        for cell, seed in pairs:
            outcome = execute_one(cell, seed, root, repo, force=force)
            results.append(outcome)
            if fail_fast and outcome["status"] == "failed":
                break
        return results

    with ProcessPoolExecutor(max_workers=jobs) as executor:
        futures = {
            executor.submit(execute_one, cell, seed, root, repo, force=force): (cell, seed)
            for cell, seed in pairs
        }
        for future in as_completed(futures):
            outcome = future.result()
            results.append(outcome)
            if fail_fast and outcome["status"] == "failed":
                for pending in futures:
                    pending.cancel()
                break
    return sorted(results, key=lambda item: item["run_id"])


__all__ = ["execute_one", "planned_runs", "run_experiment", "smoke_config"]
