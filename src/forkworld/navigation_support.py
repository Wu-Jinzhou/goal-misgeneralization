"""Pretraining and evaluation glue for frozen goal-conditioned navigators."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .artifacts import implementation_provenance, write_json
from .config import get_path
from .data import SemanticBatch
from .envs import ForkGridWorld, Goal, RandomObstacleNavigationEnv
from .metrics import binary_predictions
from .navigation import (
    NavigatorTrainingConfig,
    evaluate_navigation,
    load_navigator_checkpoint,
    save_navigator_checkpoint,
    train_bfs_navigator,
)
from .protocols import predict_logits


NAVIGATOR_CACHE_SCHEMA_VERSION = 1


def navigator_paths(root: str | Path) -> dict[str, Path]:
    base = Path(root)
    return {"fork": base / "fork.pt", "navigation": base / "navigation.pt"}


def _navigator_request(
    level: str,
    *,
    seed: int,
    maps: int,
    epochs: int,
    target_accuracy: float,
    target_patience: int,
) -> dict[str, Any]:
    implementation = implementation_provenance()
    return {
        "navigator_cache_schema_version": NAVIGATOR_CACHE_SCHEMA_VERSION,
        "source_fingerprint_schema_version": implementation[
            "source_fingerprint_schema_version"
        ],
        "implementation_fingerprint": implementation["implementation_fingerprint"],
        "level": level,
        "seed": int(seed),
        "maps": int(maps),
        "requested_epochs": int(epochs),
        "target_accuracy": float(target_accuracy),
        "target_patience": int(target_patience),
        "hidden_sizes": [128, 128],
        "activation": "relu",
        "deterministic_algorithms": "strict",
    }


def _navigator_metadata_compatible(metadata: Mapping[str, Any], request: Mapping[str, Any]) -> bool:
    for key, value in request.items():
        if metadata.get(key) != value:
            return False
    try:
        final_accuracy = float(metadata.get("final_accuracy", float("nan")))
        patience_achieved = int(metadata.get("target_patience_achieved", 0))
    except (TypeError, ValueError):
        return False
    return (
        bool(metadata.get("frozen", False))
        and final_accuracy + 1e-7 >= float(request["target_accuracy"])
        and patience_achieved >= int(request["target_patience"])
    )


def _checkpoint_compatible(path: Path, request: Mapping[str, Any]) -> bool:
    if not path.is_file():
        return False
    try:
        model, metadata = load_navigator_checkpoint(path, map_location="cpu", freeze=True)
    except (OSError, RuntimeError, ValueError, KeyError):
        return False
    del model
    return _navigator_metadata_compatible(metadata, request)


def pretrain_navigator_suite(
    output: str | Path,
    *,
    device: str = "cpu",
    seed: int = 2027,
    navigation_maps: int = 32,
    epochs: int = 250,
    target_accuracy: float = 0.99,
    target_patience: int = 3,
    force: bool = False,
) -> dict[str, Any]:
    """Pretrain and freeze one navigator for each sequential task level."""

    if navigation_maps < 1 or epochs < 1 or target_patience < 1:
        raise ValueError("navigation_maps, epochs, and target_patience must be positive")
    if not 0.0 < target_accuracy <= 1.0:
        raise ValueError("target_accuracy must lie in (0,1]")
    paths = navigator_paths(output)
    Path(output).mkdir(parents=True, exist_ok=True)
    report: dict[str, Any] = {}
    environments = {
        "fork": [ForkGridWorld(Goal.LEFT, seed=seed)],
        "navigation": [
            RandomObstacleNavigationEnv(Goal.LEFT if index % 2 == 0 else Goal.RIGHT, seed=seed + index)
            for index in range(navigation_maps)
        ],
    }
    for level, level_envs in environments.items():
        path = paths[level]
        request = _navigator_request(
            level,
            seed=seed,
            maps=len(level_envs),
            epochs=epochs,
            target_accuracy=target_accuracy,
            target_patience=target_patience,
        )
        if not force and _checkpoint_compatible(path, request):
            model, metadata = load_navigator_checkpoint(path, map_location="cpu", freeze=True)
            report[level] = {
                "checkpoint": str(path),
                "reused": True,
                "cache_compatible": True,
                **metadata,
            }
            del model
            continue
        training_config = NavigatorTrainingConfig(
            epochs=epochs,
            batch_size=128,
            learning_rate=3e-3,
            weight_decay=0.0,
            seed=seed,
            device=device,
            target_accuracy=target_accuracy,
            target_patience=target_patience,
        )
        result = train_bfs_navigator(
            level_envs,
            hidden_sizes=(128, 128),
            activation="relu",
            config=training_config,
        )
        achieved_patience = 0
        for epoch_result in reversed(result.history):
            if epoch_result.accuracy + 1e-7 < target_accuracy:
                break
            achieved_patience += 1
        if result.final_accuracy + 1e-7 < target_accuracy or achieved_patience < target_patience:
            raise RuntimeError(
                f"{level} navigator ended at accuracy {result.final_accuracy:.6f} but did not sustain "
                f"the required {target_accuracy:.6f} for {target_patience} consecutive epochs "
                f"after an epoch budget of {epochs}"
            )
        metadata = {
            **request,
            "dataset_size": result.dataset_size,
            "final_accuracy": result.final_accuracy,
            "epochs_completed": len(result.history),
            "target_patience_achieved": achieved_patience,
            "frozen": True,
        }
        result.model.freeze()
        save_navigator_checkpoint(result.model, path, metadata=metadata)
        report[level] = {"checkpoint": str(path), "reused": False, **metadata}
    write_json(Path(output) / "manifest.json", report)
    return report


def ensure_navigators(config: Mapping[str, Any], output_root: str | Path, device: str) -> Path | None:
    levels = set(get_path(config, "run.task_levels", ["choice"]))
    if not levels.intersection({"fork", "navigation"}):
        return None
    configured = get_path(config, "navigator.checkpoint_root")
    root = Path(configured) if configured else Path(output_root) / "navigators"
    paths = navigator_paths(root)
    needed = [level for level in ("fork", "navigation") if level in levels]
    seed = int(get_path(config, "navigator.seed", 2027))
    navigation_maps = int(get_path(config, "navigator.navigation_maps", 32))
    epochs = int(get_path(config, "navigator.epochs", 250))
    target_accuracy = float(get_path(config, "navigator.target_accuracy", 0.99))
    target_patience = int(get_path(config, "navigator.target_patience", 3))
    requests = {
        level: _navigator_request(
            level,
            seed=seed,
            maps=1 if level == "fork" else navigation_maps,
            epochs=epochs,
            target_accuracy=target_accuracy,
            target_patience=target_patience,
        )
        for level in needed
    }
    if all(_checkpoint_compatible(paths[level], requests[level]) for level in needed):
        return root
    if not bool(get_path(config, "navigator.auto_pretrain", True)):
        missing = [
            str(paths[level])
            for level in needed
            if not _checkpoint_compatible(paths[level], requests[level])
        ]
        raise FileNotFoundError(
            "Frozen navigator checkpoints are missing or incompatible and auto_pretrain=false: "
            + ", ".join(missing)
        )
    pretrain_navigator_suite(
        root,
        device=device,
        seed=seed,
        navigation_maps=navigation_maps,
        epochs=epochs,
        target_accuracy=target_accuracy,
        target_patience=target_patience,
    )
    return root


def evaluate_task_levels(
    model: Any,
    batch: SemanticBatch,
    config: Mapping[str, Any],
    navigator_root: str | Path | None,
    *,
    seed: int,
) -> dict[str, dict[str, float | int | str]]:
    """Evaluate goal choice and frozen navigation capability on paired episodes."""

    levels = list(get_path(config, "run.task_levels", ["choice"]))
    count = min(int(get_path(config, "run.navigation_episodes", 128)), len(batch))
    logits = predict_logits(model, batch.select(np.arange(count)), config)
    selected = binary_predictions(logits)
    # A logit tie is not a valid goal; choose a deterministic default only for
    # rollout and retain the invalid rate in the choice metrics elsewhere.
    selected = np.where(selected == 0, -1, selected)
    intended = np.asarray(batch.y[:count], dtype=np.int8)
    result: dict[str, dict[str, float | int | str]] = {
        "choice": {
            "episodes": count,
            "selection_accuracy": float(np.mean(selected == intended)),
            "oracle_goal_success_rate": 1.0,
            "clamped_goal_success_rate": 1.0,
            "intended_goal_success_rate": float(np.mean(selected == intended)),
        }
    }
    if navigator_root is None:
        return {key: value for key, value in result.items() if key in levels}
    paths = navigator_paths(navigator_root)
    for level in ("fork", "navigation"):
        if level not in levels:
            continue
        navigator, metadata = load_navigator_checkpoint(paths[level], map_location="cpu", freeze=True)
        goals = [Goal.LEFT if value < 0 else Goal.RIGHT for value in intended]
        choices = [Goal.LEFT if value < 0 else Goal.RIGHT for value in selected]
        evaluation_split = str(get_path(config, "navigator.evaluation_split", "support"))
        if evaluation_split not in {"support", "heldout"}:
            raise ValueError("navigator.evaluation_split must be support or heldout")
        if level == "fork":
            environments = [ForkGridWorld(goal, seed=seed + index) for index, goal in enumerate(goals)]
        else:
            pretrain_seed = int(metadata.get("seed", 2027))
            pretrain_maps = max(1, int(metadata.get("maps", 1)))
            map_seeds = (
                [pretrain_seed + index % pretrain_maps for index in range(count)]
                if evaluation_split == "support"
                else [10_000 * seed + index for index in range(count)]
            )
            environments = [
                RandomObstacleNavigationEnv(goal, seed=map_seed)
                for goal, map_seed in zip(goals, map_seeds, strict=True)
            ]
        evaluation = evaluate_navigation(navigator, environments, choices, intended_goals=goals)
        summary = evaluation.summary()
        summary["navigator_checkpoint"] = str(paths[level])
        summary["navigator_pretrain_accuracy"] = float(metadata.get("final_accuracy", float("nan")))
        summary["navigator_map_split"] = evaluation_split if level == "navigation" else "fixed_fork"
        if level == "navigation" and bool(get_path(config, "navigator.report_heldout_capability", True)):
            heldout_environments = [
                RandomObstacleNavigationEnv(goal, seed=1_000_000 + 10_000 * seed + index)
                for index, goal in enumerate(goals)
            ]
            heldout = evaluate_navigation(
                navigator, heldout_environments, goals, intended_goals=goals
            ).summary()
            summary.update(
                {
                    "heldout_oracle_goal_success_rate": heldout["oracle_goal_success_rate"],
                    "heldout_clamped_goal_success_rate": heldout["clamped_goal_success_rate"],
                    "heldout_mean_path_efficiency": heldout["mean_selected_path_efficiency"],
                }
            )
        result[level] = summary
    return {key: value for key, value in result.items() if key in levels}


__all__ = [
    "ensure_navigators",
    "evaluate_task_levels",
    "navigator_paths",
    "pretrain_navigator_suite",
]
