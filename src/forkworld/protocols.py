"""Shared protocol utilities and hypothesis dispatch.

Hypothesis implementations live in three focused modules.  They all return the
same JSON/artifact-friendly :class:`ProtocolResult`, allowing the sweep engine to
remain agnostic to whether a run is one-stage SFT, RL, or a phased intervention.
"""

from __future__ import annotations

import math
import random
import time
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch
from torch import nn

from .config import get_path
from .data import SemanticBatch, flip_channels
from .metrics import behavioral_metrics, binary_predictions, intervention_metrics
from .models import GoalMLP, MLPConfig, configure_update_mode, parameter_count, trainable_parameter_count


@dataclass
class ProtocolResult:
    """Complete result of one training seed in one sweep cell."""

    model: nn.Module
    summary: dict[str, Any]
    metrics: list[dict[str, Any]] = field(default_factory=list)
    predictions: list[dict[str, Any]] = field(default_factory=list)
    checkpoints: dict[str, dict[str, Any]] = field(default_factory=dict)
    evaluation_batch: SemanticBatch | None = field(default=None, repr=False)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(requested: str | None) -> torch.device:
    value = (requested or "auto").lower()
    if value == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    if device.type == "mps" and not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
        raise RuntimeError("MPS was requested but is not available")
    return device


def feature_array(batch: SemanticBatch, config: Mapping[str, Any], include_state: bool = True) -> np.ndarray:
    return batch.features(max_k=int(get_path(config, "data.max_k", batch.active_k)), include_state=include_state)


def batch_for_training(batch: SemanticBatch, config: Mapping[str, Any], include_state: bool = True) -> tuple[Any, ...]:
    x = torch.as_tensor(feature_array(batch, config, include_state), dtype=torch.float32)
    y = torch.as_tensor(np.asarray(batch.target), dtype=torch.float32)
    if batch.nuisance_targets is None:
        return x, y
    nuisance = torch.as_tensor(np.asarray(batch.nuisance_targets))
    return x, y, nuisance


def build_model(
    batch: SemanticBatch,
    config: Mapping[str, Any],
    seed: int,
    *,
    nuisance_heads: int | None = None,
    include_state: bool = True,
) -> tuple[nn.Module, dict[str, Any]]:
    """Build an architecture and independently apply its update-capacity budget."""

    seed_everything(seed)
    input_dim = int(feature_array(batch, config, include_state).shape[1])
    if nuisance_heads is None:
        nuisance_heads = int(get_path(config, "model.nuisance_bits", 0))
    base = GoalMLP(
        MLPConfig(
            input_dim=input_dim,
            width=int(get_path(config, "model.width", 64)),
            depth=int(get_path(config, "model.depth", 2)),
            activation=str(get_path(config, "model.activation", "relu")),
            residual=bool(get_path(config, "model.residual", False)),
            bias=bool(get_path(config, "model.bias", True)),
            nuisance_heads=nuisance_heads,
        )
    )
    base_total = parameter_count(base)
    requested_mode = str(get_path(config, "update.mode", "full"))
    mode = "head_only" if requested_mode in {"head", "head-only"} else requested_mode
    raw_budget = get_path(config, "update.budget", "full")
    budget = None if raw_budget in (None, "full") else int(raw_budget)
    model = configure_update_mode(
        base,
        mode=mode,
        budget=budget,
        projection=str(get_path(config, "update.projection", "count_sketch")),
        seed=int(get_path(config, "update.subspace_seed", 1729)) + seed,
        scale=float(get_path(config, "update.scale", 1.0)),
        include_nuisance_heads=True,
    )
    report = {
        "input_dim": input_dim,
        "total_parameters": base_total,
        "trainable_parameters": trainable_parameter_count(model),
        "update_mode": mode,
        "requested_budget": raw_budget,
    }
    return model, report


def _goal_logit(output: Any) -> torch.Tensor:
    if torch.is_tensor(output):
        value = output
    elif hasattr(output, "goal_logits"):
        value = output.goal_logits
    elif isinstance(output, Mapping):
        value = output.get("goal_logits", output.get("logits"))
    else:
        raise TypeError("Model output has no goal logits")
    if value.ndim == 2 and value.shape[1] == 1:
        value = value[:, 0]
    if value.ndim != 1:
        raise ValueError(f"Expected one binary logit per row, received {tuple(value.shape)}")
    return value


@torch.no_grad()
def predict_logits(
    model: nn.Module, batch: SemanticBatch, config: Mapping[str, Any], *, include_state: bool = True
) -> np.ndarray:
    """Return two-class logits compatible with the public metric functions."""

    device = next(model.parameters()).device
    x = torch.as_tensor(feature_array(batch, config, include_state), dtype=torch.float32, device=device)
    was_training = model.training
    model.eval()
    logit = _goal_logit(model(x)).detach().cpu().numpy().astype(np.float64)
    model.train(was_training)
    return np.column_stack((-0.5 * logit, 0.5 * logit))


def evaluate_batch(
    model: nn.Module,
    batch: SemanticBatch,
    config: Mapping[str, Any],
    *,
    include_state: bool = True,
) -> dict[str, float]:
    logits = predict_logits(model, batch, config, include_state=include_state)
    proxy = np.asarray(batch.channels.get("P", batch.y), dtype=np.int8)
    metrics = behavioral_metrics(logits, np.asarray(batch.y), proxy)
    target = np.asarray(batch.target)
    target_sign = np.where(target > 0, 1, -1).astype(np.int8)
    metrics["target_accuracy"] = float(np.mean(binary_predictions(logits) == target_sign))
    return metrics


def evaluate_standard_interventions(
    model: nn.Module,
    batch: SemanticBatch,
    config: Mapping[str, Any],
    *,
    include_state: bool = True,
) -> dict[str, dict[str, float]]:
    base = predict_logits(model, batch, config, include_state=include_state)
    results: dict[str, dict[str, float]] = {}
    if "P" in batch.channels:
        changed = predict_logits(model, flip_channels(batch, "P"), config, include_state=include_state)
        results["flip_P"] = intervention_metrics(base, changed, np.asarray(batch.y))
    effects = []
    for index in range(1, batch.active_k + 1):
        changed = predict_logits(
            model, flip_channels(batch, f"R_{index}"), config, include_state=include_state
        )
        result = intervention_metrics(base, changed, np.asarray(batch.y))
        results[f"flip_R_{index}"] = result
        effects.append(result)
    if effects:
        results["flip_R_mean"] = {
            key: float(np.mean([effect[key] for effect in effects]))
            for key in ("probability_ate", "logit_ate", "hard_flip_rate", "n")
        }
    return results


def make_metric_records(
    values: Mapping[str, float | int],
    *,
    hypothesis: str,
    split: str,
    global_step: int,
    stage: str = "train",
    stage_step: int | None = None,
    examples_seen: int = 0,
    intervention: str = "none",
    level: str = "choice",
    condition: str = "primary",
) -> list[dict[str, Any]]:
    records = []
    n = int(values.get("n", 0))
    for metric, value in values.items():
        if metric == "n" or not isinstance(value, (int, float, np.number)):
            continue
        numeric_value = float(value)
        if not math.isfinite(numeric_value):
            continue
        records.append(
            {
                "experiment": hypothesis,
                "level": level,
                "condition": condition,
                "stage": stage,
                "stage_step": global_step if stage_step is None else stage_step,
                "global_step": global_step,
                "examples_seen": examples_seen,
                "split": split,
                "intervention": intervention,
                "metric": metric,
                "value": numeric_value,
                "n": n,
            }
        )
    return records


def prediction_records(
    model: nn.Module,
    batch: SemanticBatch,
    config: Mapping[str, Any],
    *,
    split: str,
    intervention: str = "none",
    include_state: bool = True,
) -> list[dict[str, Any]]:
    logits = predict_logits(model, batch, config, include_state=include_state)
    prediction = binary_predictions(logits)
    probabilities = torch.softmax(torch.as_tensor(logits), dim=1).numpy()
    proxy = np.asarray(batch.channels.get("P", np.zeros(len(batch), dtype=np.int8)))
    return [
        {
            "sample_id": int(batch.sample_id[index]),
            "state_id": int(batch.state_id[index]),
            "split": split,
            "intervention": intervention,
            "y": int(batch.y[index]),
            "proxy": int(proxy[index]),
            "prediction": int(prediction[index]),
            "probability_positive": float(probabilities[index, 1]),
        }
        for index in range(len(batch))
    ]


def timed_call(function: Any, *args: Any, **kwargs: Any) -> tuple[Any, float]:
    started = time.perf_counter()
    result = function(*args, **kwargs)
    return result, time.perf_counter() - started


def run_protocol(config: Mapping[str, Any], seed: int) -> ProtocolResult:
    hypothesis = str(get_path(config, "experiment.hypothesis")).lower()
    if hypothesis in {"h1", "h2", "h3", "h4"}:
        from .protocols_selection import RUNNERS
    elif hypothesis in {"h5", "h6"}:
        from .protocols_algorithms import RUNNERS
    elif hypothesis in {"h7", "h8", "h9"}:
        from .protocols_persistence import RUNNERS
    elif hypothesis in {"h10", "h11"}:
        from .protocols_extensions import RUNNERS
    elif hypothesis in {"h12", "h13"}:
        from .protocols_dynamics import RUNNERS
    elif hypothesis == "h14":
        from .protocols_handoff import run_h14

        return run_h14(config, seed)
    elif hypothesis == "h15":
        from .protocols_order import run_h15

        return run_h15(config, seed)
    elif hypothesis == "h16":
        from .protocols_mediation import run_h16

        return run_h16(config, seed)
    elif hypothesis == "h17":
        from .protocols_counterbalanced import run_h17

        return run_h17(config, seed)
    else:
        raise ValueError(f"Unsupported hypothesis: {hypothesis}")
    return RUNNERS[hypothesis](config, seed)


__all__ = [
    "ProtocolResult",
    "batch_for_training",
    "build_model",
    "evaluate_batch",
    "evaluate_standard_interventions",
    "feature_array",
    "make_metric_records",
    "predict_logits",
    "prediction_records",
    "resolve_device",
    "run_protocol",
    "seed_everything",
]
