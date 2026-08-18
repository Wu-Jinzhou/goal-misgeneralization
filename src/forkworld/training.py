"""Deterministic training primitives for the ForkWorld experiments.

The functions in this module deliberately depend only on duck-typed tensor
batches.  A batch may be ``(x, y)``, a mapping, or an object exposing
``x``/``features()`` and ``y``/``target``.  This keeps the optimization code
usable for the one-step task, fork environments, and standalone decoder
calibration without coupling it to a particular data generator.
"""

from __future__ import annotations

import copy
import inspect
import math
import random
from collections.abc import Callable, Iterable, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field, replace
from typing import Any, Literal

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .models import GoalModelOutput, ValueMLP, trainable_parameter_count

OptimizerName = Literal["adamw", "adam", "sgd"]
BanditAlgorithm = Literal["actor_critic", "reinforce"]


@dataclass(frozen=True)
class TrainConfig:
    """Shared optimization, logging, and reset settings.

    ``reset_model=False`` is important for multi-phase unlearning and hysteresis
    experiments.  ``reset_optimizer=True`` starts a new optimizer phase even if
    an optimizer object is supplied; setting it to ``False`` preserves momentum
    and adaptive moments.  Weight decay is explicit and defaults to zero so that
    proxy removal does not silently erase unused weights.
    """

    steps: int = 1_000
    batch_size: int = 128
    learning_rate: float = 3e-3
    weight_decay: float = 0.0
    optimizer: OptimizerName = "adamw"
    momentum: float = 0.0
    seed: int = 0
    device: str | torch.device | None = None
    deterministic: bool = True
    shuffle: bool = True
    gradient_clip_norm: float | None = None
    log_steps: Sequence[int] | None = None
    num_log_points: int = 25
    checkpoint_steps: Sequence[int] | None = None
    save_checkpoints: bool = True
    reset_model: bool = False
    reset_optimizer: bool = True

    def __post_init__(self) -> None:
        if isinstance(self.steps, bool) or self.steps < 1:
            raise ValueError("steps must be a positive integer")
        if isinstance(self.batch_size, bool) or self.batch_size < 1:
            raise ValueError("batch_size must be a positive integer")
        if not math.isfinite(self.learning_rate) or self.learning_rate <= 0:
            raise ValueError("learning_rate must be finite and positive")
        if not math.isfinite(self.weight_decay) or self.weight_decay < 0:
            raise ValueError("weight_decay must be finite and non-negative")
        if self.optimizer not in ("adamw", "adam", "sgd"):
            raise ValueError("optimizer must be 'adamw', 'adam', or 'sgd'")
        if not math.isfinite(self.momentum) or not 0 <= self.momentum < 1:
            raise ValueError("momentum must satisfy 0 <= momentum < 1")
        if self.gradient_clip_norm is not None and self.gradient_clip_norm <= 0:
            raise ValueError("gradient_clip_norm must be positive when provided")
        if isinstance(self.num_log_points, bool) or self.num_log_points < 1:
            raise ValueError("num_log_points must be a positive integer")
        _validate_step_sequence(self.log_steps, self.steps, "log_steps")
        _validate_step_sequence(self.checkpoint_steps, self.steps, "checkpoint_steps")


@dataclass(frozen=True)
class SFTConfig(TrainConfig):
    """Configuration for clean or nuisance-rich supervised fine-tuning."""

    auxiliary_weight: float = 0.0
    nuisance_weights: Mapping[str, float] | None = None
    label_smoothing: float = 0.0

    def __post_init__(self) -> None:
        super().__post_init__()
        if not math.isfinite(self.auxiliary_weight) or self.auxiliary_weight < 0:
            raise ValueError("auxiliary_weight must be finite and non-negative")
        if not 0 <= self.label_smoothing < 1:
            raise ValueError("label_smoothing must satisfy 0 <= value < 1")
        if self.nuisance_weights is not None:
            for name, weight in self.nuisance_weights.items():
                if not name or not math.isfinite(weight) or weight < 0:
                    raise ValueError("nuisance weights must have names and be finite/non-negative")


@dataclass(frozen=True)
class OnPolicyImitationConfig(TrainConfig):
    """DAgger-like collection and replay settings."""

    collection_batch_size: int | None = None
    updates_per_collection: int = 1
    replay_capacity: int | None = None
    label_smoothing: float = 0.0

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.collection_batch_size is not None and self.collection_batch_size < 1:
            raise ValueError("collection_batch_size must be positive when provided")
        if self.updates_per_collection < 1:
            raise ValueError("updates_per_collection must be positive")
        if self.replay_capacity is not None and self.replay_capacity < 1:
            raise ValueError("replay_capacity must be positive when provided")
        if not 0 <= self.label_smoothing < 1:
            raise ValueError("label_smoothing must satisfy 0 <= value < 1")


@dataclass(frozen=True)
class BanditConfig(TrainConfig):
    """Contextual-bandit REINFORCE or actor-critic settings."""

    algorithm: BanditAlgorithm = "actor_critic"
    entropy_coefficient: float = 0.0
    normalize_advantages: bool = True
    center_reinforce_rewards: bool = True
    reward_scale: float = 1.0
    critic_learning_rate: float = 3e-3
    critic_weight_decay: float = 0.0
    critic_width: int = 256
    critic_depth: int = 3
    critic_activation: Literal["relu", "tanh", "gelu", "silu"] = "gelu"
    critic_residual: bool = True
    critic_updates_per_step: int = 1
    reset_critic: bool = True

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.algorithm not in ("actor_critic", "reinforce"):
            raise ValueError("algorithm must be 'actor_critic' or 'reinforce'")
        if not math.isfinite(self.entropy_coefficient) or self.entropy_coefficient < 0:
            raise ValueError("entropy_coefficient must be finite and non-negative")
        if not math.isfinite(self.reward_scale):
            raise ValueError("reward_scale must be finite")
        if not math.isfinite(self.critic_learning_rate) or self.critic_learning_rate <= 0:
            raise ValueError("critic_learning_rate must be finite and positive")
        if not math.isfinite(self.critic_weight_decay) or self.critic_weight_decay < 0:
            raise ValueError("critic_weight_decay must be finite and non-negative")
        if self.critic_width < 1 or self.critic_depth < 0:
            raise ValueError("critic_width must be positive and critic_depth non-negative")
        if self.critic_updates_per_step < 1:
            raise ValueError("critic_updates_per_step must be positive")


@dataclass(frozen=True)
class TrainingRecord:
    """One log-spaced, JSON-friendly observation of a training run."""

    step: int
    loss: float
    primary_loss: float
    auxiliary_loss: float = 0.0
    samples_seen: int = 0
    optimizer_steps: int = 0
    metrics: dict[str, float] = field(default_factory=dict)

    def as_dict(self) -> dict[str, float | int]:
        result: dict[str, float | int] = {
            "step": self.step,
            "loss": self.loss,
            "primary_loss": self.primary_loss,
            "auxiliary_loss": self.auxiliary_loss,
            "samples_seen": self.samples_seen,
            "optimizer_steps": self.optimizer_steps,
        }
        result.update(self.metrics)
        return result


@dataclass
class TrainingCheckpoint:
    """In-memory, CPU checkpoint sufficient to resume or inspect a phase."""

    step: int
    model_state: dict[str, Any]
    optimizer_state: dict[str, Any]
    critic_state: dict[str, Any] | None = None
    critic_optimizer_state: dict[str, Any] | None = None
    data_generator_state: Tensor | None = None
    action_generator_state: Tensor | None = None


@dataclass
class TrainingSnapshot:
    """Callback payload at a requested log-spaced step."""

    step: int
    record: TrainingRecord
    model: nn.Module
    checkpoint: TrainingCheckpoint | None = None
    critic: nn.Module | None = None


class StopTraining(Exception):
    """Raise from a training callback to request a successful early stop."""


@dataclass
class TrainingResult:
    """Structured output shared by SFT, DAgger, and bandit trainers."""

    history: list[TrainingRecord]
    checkpoints: dict[int, TrainingCheckpoint]
    final_model_state: dict[str, Any]
    optimizer_state: dict[str, Any]
    actor_trainable_parameters: int
    samples_seen: int
    optimizer_steps: int
    optimizer: torch.optim.Optimizer = field(repr=False)
    critic: nn.Module | None = field(default=None, repr=False)
    critic_state: dict[str, Any] | None = None
    critic_optimizer_state: dict[str, Any] | None = None
    critic_parameter_count: int = 0
    critic_optimizer: torch.optim.Optimizer | None = field(default=None, repr=False)
    data_generator_state: Tensor | None = field(default=None, repr=False)
    action_generator_state: Tensor | None = field(default=None, repr=False)

    def history_dicts(self) -> list[dict[str, float | int]]:
        return [record.as_dict() for record in self.history]


Evaluator = Callable[..., Mapping[str, float | int | Tensor]]
TrainingCallback = Callable[[TrainingSnapshot], None]


@dataclass
class _Batch:
    x: Tensor
    y: Tensor | None = None
    nuisance: Mapping[str, Tensor] | Tensor | None = None
    raw: Any = None


_FEATURE_NAMES = ("x", "features", "observations", "observation", "contexts", "context")
# ``SemanticBatch.y`` is the immutable semantic goal used by evaluation, while
# ``SemanticBatch.target`` is the possibly intervened supervised objective.
# Prefer the explicit target whenever both are present (as in H6 label-noise
# cells); ordinary ``(x, y)`` tuples and mappings containing only ``y`` retain
# their existing behavior.
_TARGET_NAMES = ("target", "targets", "label", "labels", "actions", "action", "y")
_NUISANCE_NAMES = (
    "nuisance_targets",
    "nuisance_target",
    "nuisance_bits",
    "nuisance",
    "auxiliary_targets",
)


def _validate_step_sequence(values: Sequence[int] | None, total: int, name: str) -> None:
    if values is None:
        return
    for value in values:
        if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= total:
            raise ValueError(f"every {name} entry must be an integer in [1, {total}]")


def log_spaced_steps(total_steps: int, num_points: int = 25) -> tuple[int, ...]:
    """Return unique integer checkpoints spaced uniformly in log time."""

    if isinstance(total_steps, bool) or total_steps < 1:
        raise ValueError("total_steps must be a positive integer")
    if isinstance(num_points, bool) or num_points < 1:
        raise ValueError("num_points must be a positive integer")
    if total_steps <= num_points:
        return tuple(range(1, total_steps + 1))
    if num_points == 1:
        return (total_steps,)
    log_total = math.log(float(total_steps))
    points = {
        max(1, min(total_steps, round(math.exp(log_total * index / (num_points - 1)))))
        for index in range(num_points)
    }
    points.update((1, total_steps))
    return tuple(sorted(points))


def _resolved_steps(config: TrainConfig) -> tuple[set[int], set[int]]:
    logs = set(
        config.log_steps
        if config.log_steps is not None
        else log_spaced_steps(config.steps, config.num_log_points)
    )
    logs.add(config.steps)
    if not config.save_checkpoints:
        checkpoints: set[int] = set()
    elif config.checkpoint_steps is None:
        checkpoints = set(logs)
    else:
        checkpoints = set(config.checkpoint_steps)
        checkpoints.add(config.steps)
    return logs, checkpoints


@contextmanager
def _deterministic_context(seed: int, enabled: bool) -> Iterable[None]:
    cpu_state = torch.random.get_rng_state()
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    cuda_states = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    previous_deterministic = torch.are_deterministic_algorithms_enabled()
    previous_warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if enabled:
        torch.use_deterministic_algorithms(True, warn_only=False)
    try:
        yield
    finally:
        torch.random.set_rng_state(cpu_state)
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        if cuda_states is not None:
            torch.cuda.set_rng_state_all(cuda_states)
        torch.use_deterministic_algorithms(previous_deterministic, warn_only=previous_warn_only)


def _get_member(value: Any, names: Sequence[str]) -> Any:
    if isinstance(value, Mapping):
        for name in names:
            if name in value:
                result = value[name]
                return result() if callable(result) else result
        return None
    for name in names:
        if hasattr(value, name):
            result = getattr(value, name)
            return result() if callable(result) else result
    return None


def _feature_tensor(value: Any) -> Tensor:
    if torch.is_tensor(value):
        return value
    if isinstance(value, tuple) and value and torch.is_tensor(value[0]):
        # Support APIs returning (feature_tensor, feature_names).
        return value[0]
    if isinstance(value, Mapping):
        columns: list[Tensor] = []
        for item in value.values():
            tensor = item if torch.is_tensor(item) else torch.as_tensor(item)
            if tensor.ndim == 1:
                tensor = tensor.unsqueeze(-1)
            columns.append(tensor)
        if not columns:
            raise ValueError("feature mapping is empty")
        return torch.cat(columns, dim=-1)
    return torch.as_tensor(value)


def _target_tensor(value: Any) -> Tensor:
    return value if torch.is_tensor(value) else torch.as_tensor(value)


def _normalise_batch(value: Any, *, require_target: bool = False) -> _Batch:
    if isinstance(value, _Batch):
        batch = value
    elif torch.is_tensor(value):
        batch = _Batch(x=value, raw=value)
    elif isinstance(value, (tuple, list)):
        if not value:
            raise ValueError("an empty sequence is not a batch")
        batch = _Batch(
            x=_feature_tensor(value[0]),
            y=_target_tensor(value[1]) if len(value) > 1 and value[1] is not None else None,
            nuisance=value[2] if len(value) > 2 else None,
            raw=value,
        )
    else:
        features = _get_member(value, _FEATURE_NAMES)
        if features is None:
            raise TypeError(
                "batch must be a tensor, (x, y) sequence, mapping, or object exposing "
                "x/features()/observations"
            )
        target = _get_member(value, _TARGET_NAMES)
        nuisance = _get_member(value, _NUISANCE_NAMES)
        batch = _Batch(
            x=_feature_tensor(features),
            y=_target_tensor(target) if target is not None else None,
            nuisance=nuisance,
            raw=value,
        )
    if batch.x.ndim == 1:
        batch.x = batch.x.unsqueeze(0)
    if require_target and batch.y is None:
        raise ValueError("training batch has no y/target/label")
    if batch.y is not None:
        if batch.y.ndim == 0:
            batch.y = batch.y.unsqueeze(0)
        if len(batch.y) != len(batch.x):
            raise ValueError("feature and target tensors have different batch dimensions")
    return batch


def _index_optional(value: Mapping[str, Tensor] | Tensor | None, index: Tensor) -> Any:
    if value is None:
        return None
    if isinstance(value, Mapping):
        return {name: _target_tensor(target)[index] for name, target in value.items()}
    return _target_tensor(value)[index]


class _StaticBatcher:
    def __init__(
        self,
        data: Any,
        batch_size: int,
        generator: torch.Generator,
        *,
        shuffle: bool,
        require_target: bool,
    ) -> None:
        self.data = _normalise_batch(data, require_target=require_target)
        if len(self.data.x) == 0:
            raise ValueError("cannot train on an empty dataset")
        self.batch_size = batch_size
        self.generator = generator
        self.shuffle = shuffle
        self.order = torch.empty(0, dtype=torch.long)
        self.cursor = 0

    def _new_epoch(self) -> None:
        size = len(self.data.x)
        self.order = (
            torch.randperm(size, generator=self.generator)
            if self.shuffle
            else torch.arange(size, dtype=torch.long)
        )
        self.cursor = 0

    def next(self) -> _Batch:
        if self.cursor >= len(self.order):
            self._new_epoch()
        stop = min(self.cursor + self.batch_size, len(self.order))
        index = self.order[self.cursor : stop]
        self.cursor = stop
        return _Batch(
            x=self.data.x[index],
            y=self.data.y[index] if self.data.y is not None else None,
            nuisance=_index_optional(self.data.nuisance, index),
            raw=self.data.raw,
        )


def _call_adaptively(function: Callable[..., Any], available: Mapping[str, Any]) -> Any:
    """Call a user hook by matching documented argument-name aliases."""

    try:
        signature = inspect.signature(function)
    except (TypeError, ValueError):  # Some extension callables have no signature.
        return function(**available)

    args: list[Any] = []
    kwargs: dict[str, Any] = {}
    accepts_kwargs = False
    missing: list[str] = []
    for parameter in signature.parameters.values():
        if parameter.kind is inspect.Parameter.VAR_POSITIONAL:
            continue
        if parameter.kind is inspect.Parameter.VAR_KEYWORD:
            accepts_kwargs = True
            continue
        if parameter.name in available:
            if parameter.kind is inspect.Parameter.POSITIONAL_ONLY:
                args.append(available[parameter.name])
            else:
                kwargs[parameter.name] = available[parameter.name]
        elif parameter.default is inspect.Parameter.empty:
            missing.append(parameter.name)
    if missing:
        raise TypeError(
            f"cannot call {function!r}; unsupported required arguments: {', '.join(missing)}"
        )
    if accepts_kwargs:
        kwargs.update({name: value for name, value in available.items() if name not in kwargs})
    return function(*args, **kwargs)


def _source_available(
    *,
    batch_size: int,
    generator: torch.Generator,
    step: int,
    model: nn.Module | None = None,
) -> dict[str, Any]:
    return {
        "batch_size": batch_size,
        "n": batch_size,
        "generator": generator,
        "rng": generator,
        "step": step,
        "model": model,
        "actor": model,
        "policy": model,
    }


def _sample_dynamic(
    source: Any,
    *,
    batch_size: int,
    generator: torch.Generator,
    step: int,
    model: nn.Module | None = None,
) -> Any:
    method = getattr(source, "sample_batch", None)
    function = method if callable(method) else source
    if not callable(function):
        raise TypeError("dynamic source must be callable or expose sample_batch()")
    return _call_adaptively(
        function,
        _source_available(batch_size=batch_size, generator=generator, step=step, model=model),
    )


def _is_dynamic_source(source: Any) -> bool:
    return callable(source) or callable(getattr(source, "sample_batch", None))


def _to_device(batch: _Batch, device: torch.device) -> _Batch:
    nuisance: Mapping[str, Tensor] | Tensor | None
    if isinstance(batch.nuisance, Mapping):
        nuisance = {
            name: _target_tensor(value).to(device) for name, value in batch.nuisance.items()
        }
    elif batch.nuisance is None:
        nuisance = None
    else:
        nuisance = _target_tensor(batch.nuisance).to(device)
    return _Batch(
        x=batch.x.to(device),
        y=batch.y.to(device) if batch.y is not None else None,
        nuisance=nuisance,
        raw=batch.raw,
    )


def _module_device(module: nn.Module) -> torch.device:
    try:
        return next(module.parameters()).device
    except StopIteration as exc:
        raise ValueError("training requires a model with parameters") from exc


def _resolve_device(module: nn.Module, requested: str | torch.device | None) -> torch.device:
    device = _module_device(module) if requested is None else torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA device requested but CUDA is unavailable")
    module.to(device)
    return device


def reset_model_parameters(model: nn.Module, *, trainable_only: bool = False) -> None:
    """Reset modules once, optionally restricting resets to trainable updates.

    With an exact-subspace actor, ``trainable_only=True`` zeroes only its update
    vector and preserves the frozen anchor.  With head-only training it resets
    only head modules.  This function does not change ``requires_grad`` flags.
    """

    seen: set[int] = set()
    for module in model.modules():
        if id(module) in seen:
            continue
        seen.add(id(module))
        reset = getattr(module, "reset_parameters", None)
        if not callable(reset):
            continue
        direct_parameters = list(module.parameters(recurse=False))
        if trainable_only and not any(parameter.requires_grad for parameter in direct_parameters):
            continue
        reset()


def _make_optimizer(model: nn.Module, config: TrainConfig) -> torch.optim.Optimizer:
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not parameters:
        raise ValueError("model has no trainable parameters")
    if config.optimizer == "adamw":
        return torch.optim.AdamW(
            parameters, lr=config.learning_rate, weight_decay=config.weight_decay
        )
    if config.optimizer == "adam":
        return torch.optim.Adam(
            parameters, lr=config.learning_rate, weight_decay=config.weight_decay
        )
    return torch.optim.SGD(
        parameters,
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
        momentum=config.momentum,
    )


def _prepare_optimizer(
    model: nn.Module,
    config: TrainConfig,
    optimizer: torch.optim.Optimizer | None,
) -> torch.optim.Optimizer:
    if optimizer is None:
        return _make_optimizer(model, config)
    trainable = {id(parameter) for parameter in model.parameters() if parameter.requires_grad}
    optimized = {
        id(parameter)
        for group in optimizer.param_groups
        for parameter in group.get("params", ())
    }
    missing = trainable - optimized
    if missing:
        raise ValueError("provided optimizer does not contain every trainable model parameter")
    if config.reset_model and not config.reset_optimizer and optimizer.state:
        raise ValueError(
            "reset_model=True with preserved optimizer state is ambiguous; set reset_optimizer=True"
        )
    if config.reset_optimizer:
        optimizer.state.clear()
    for group in optimizer.param_groups:
        group["lr"] = config.learning_rate
        group["weight_decay"] = config.weight_decay
        if config.optimizer == "sgd" and "momentum" in group:
            group["momentum"] = config.momentum
    return optimizer


def _binary_targets(target: Tensor) -> Tensor:
    target = target.float()
    if target.ndim > 1 and target.shape[-1] == 1:
        target = target.squeeze(-1)
    if target.numel() == 0:
        raise ValueError("target tensor is empty")
    minimum = float(target.detach().min().cpu())
    maximum = float(target.detach().max().cpu())
    if minimum >= 0.0 and maximum <= 1.0:
        return target
    if minimum >= -1.0 and maximum <= 1.0:
        return (target > 0).float()
    raise ValueError("binary targets must use {0,1}, {-1,+1}, or boolean values")


def _goal_logits(output: Any) -> Tensor:
    if torch.is_tensor(output):
        logits = output
    elif isinstance(output, GoalModelOutput):
        logits = output.goal_logits
    elif isinstance(output, Mapping):
        logits = output.get("goal_logits", output.get("logits"))
    else:
        logits = getattr(output, "goal_logits", getattr(output, "logits", None))
    if not torch.is_tensor(logits):
        raise TypeError("model output must be a tensor or expose tensor goal_logits/logits")
    if logits.ndim > 1 and logits.shape[-1] == 1:
        logits = logits.squeeze(-1)
    if logits.ndim != 1:
        raise ValueError(
            f"binary goal logits must have shape [batch] or [batch,1], got {logits.shape}"
        )
    return logits


def _nuisance_logits(output: Any) -> Mapping[str, Tensor]:
    if isinstance(output, GoalModelOutput):
        return output.nuisance_logits
    if isinstance(output, Mapping):
        result = output.get("nuisance_logits", output.get("auxiliary_logits"))
    else:
        result = getattr(output, "nuisance_logits", None)
    if not isinstance(result, Mapping):
        raise TypeError("auxiliary SFT requires model output with a nuisance_logits mapping")
    return result


def _binary_loss(logits: Tensor, target: Tensor, smoothing: float = 0.0) -> Tensor:
    labels = _binary_targets(target).to(device=logits.device, dtype=logits.dtype)
    if labels.shape != logits.shape:
        raise ValueError(f"logit shape {logits.shape} does not match target shape {labels.shape}")
    if smoothing:
        labels = labels * (1.0 - smoothing) + 0.5 * smoothing
    return F.binary_cross_entropy_with_logits(logits, labels)


def _targets_by_head(
    targets: Mapping[str, Tensor] | Tensor | None,
    head_names: Sequence[str],
) -> dict[str, Tensor]:
    if targets is None:
        raise ValueError("auxiliary_weight > 0 but the batch has no nuisance targets")
    if isinstance(targets, Mapping):
        missing = set(head_names) - set(targets)
        if missing:
            raise ValueError(f"missing nuisance targets for heads: {sorted(missing)}")
        return {name: _target_tensor(targets[name]) for name in head_names}
    tensor = _target_tensor(targets)
    if len(head_names) == 1:
        return {head_names[0]: tensor}
    if tensor.ndim < 2 or tensor.shape[-1] != len(head_names):
        raise ValueError(
            f"tensor nuisance targets must have final dimension {len(head_names)}"
        )
    return {name: tensor[..., index] for index, name in enumerate(head_names)}


def _auxiliary_loss(
    output: Any,
    targets: Mapping[str, Tensor] | Tensor | None,
    weights: Mapping[str, float] | None,
) -> Tensor:
    logits_by_head = _nuisance_logits(output)
    if not logits_by_head:
        raise ValueError("auxiliary SFT requested, but the model has no nuisance heads")
    target_by_head = _targets_by_head(targets, list(logits_by_head))
    weighted: list[Tensor] = []
    normalizer = 0.0
    for name, logits in logits_by_head.items():
        weight = 1.0 if weights is None else float(weights.get(name, 1.0))
        if weight == 0:
            continue
        target = target_by_head[name].to(logits.device)
        if logits.ndim == 1:
            loss = _binary_loss(logits, target)
        elif logits.ndim == 2 and logits.shape[-1] == 1:
            loss = _binary_loss(logits.squeeze(-1), target)
        else:
            loss = F.cross_entropy(logits, target.long().reshape(-1))
        weighted.append(weight * loss)
        normalizer += weight
    if not weighted:
        reference = next(iter(logits_by_head.values()))
        return reference.sum() * 0.0
    return torch.stack(weighted).sum() / normalizer


def _clip_gradients(model: nn.Module, maximum: float | None) -> None:
    if maximum is not None:
        nn.utils.clip_grad_norm_(
            [parameter for parameter in model.parameters() if parameter.requires_grad], maximum
        )


def _cpu_clone(value: Any) -> Any:
    if torch.is_tensor(value):
        return value.detach().cpu().clone()
    if isinstance(value, Mapping):
        return {key: _cpu_clone(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_cpu_clone(item) for item in value)
    if isinstance(value, list):
        return [_cpu_clone(item) for item in value]
    return copy.deepcopy(value)


def _state_dict(module: nn.Module) -> dict[str, Any]:
    return _cpu_clone(module.state_dict())


def _checkpoint(
    step: int,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    data_generator: torch.Generator,
    *,
    critic: nn.Module | None = None,
    critic_optimizer: torch.optim.Optimizer | None = None,
    action_generator: torch.Generator | None = None,
) -> TrainingCheckpoint:
    return TrainingCheckpoint(
        step=step,
        model_state=_state_dict(model),
        optimizer_state=_cpu_clone(optimizer.state_dict()),
        critic_state=_state_dict(critic) if critic is not None else None,
        critic_optimizer_state=(
            _cpu_clone(critic_optimizer.state_dict()) if critic_optimizer is not None else None
        ),
        data_generator_state=data_generator.get_state().cpu().clone(),
        action_generator_state=(
            action_generator.get_state().cpu().clone() if action_generator is not None else None
        ),
    )


def restore_checkpoint(
    model: nn.Module,
    checkpoint: TrainingCheckpoint,
    *,
    optimizer: torch.optim.Optimizer | None = None,
    critic: nn.Module | None = None,
    critic_optimizer: torch.optim.Optimizer | None = None,
) -> None:
    """Restore model and optional optimizer/critic state from a checkpoint."""

    model.load_state_dict(checkpoint.model_state)
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint.optimizer_state)
    if critic is not None:
        if checkpoint.critic_state is None:
            raise ValueError("checkpoint contains no critic state")
        critic.load_state_dict(checkpoint.critic_state)
    if critic_optimizer is not None:
        if checkpoint.critic_optimizer_state is None:
            raise ValueError("checkpoint contains no critic optimizer state")
        critic_optimizer.load_state_dict(checkpoint.critic_optimizer_state)


def _metric_value(value: float | int | Tensor) -> float:
    if torch.is_tensor(value):
        if value.numel() != 1:
            raise ValueError("evaluation metrics must be scalar")
        return float(value.detach().cpu())
    return float(value)


def _evaluate(evaluator: Evaluator | None, model: nn.Module, step: int) -> dict[str, float]:
    if evaluator is None:
        return {}
    was_training = model.training
    model.eval()
    available = {
        "model": model,
        "actor": model,
        "policy": model,
        "step": step,
    }
    with torch.no_grad():
        result = _call_adaptively(evaluator, available)
    model.train(was_training)
    if not isinstance(result, Mapping):
        raise TypeError("evaluator must return a mapping of scalar metrics")
    return {str(name): _metric_value(value) for name, value in result.items()}


def _callbacks(
    callback: TrainingCallback | Sequence[TrainingCallback] | None,
) -> tuple[TrainingCallback, ...]:
    if callback is None:
        return ()
    if callable(callback):
        return (callback,)
    result = tuple(callback)
    if not all(callable(item) for item in result):
        raise TypeError("every callback must be callable")
    return result


def _emit(
    callbacks: tuple[TrainingCallback, ...],
    record: TrainingRecord,
    model: nn.Module,
    checkpoint: TrainingCheckpoint | None,
    critic: nn.Module | None = None,
) -> bool:
    snapshot = TrainingSnapshot(
        step=record.step,
        record=record,
        model=model,
        checkpoint=checkpoint,
        critic=critic,
    )
    try:
        for function in callbacks:
            function(snapshot)
    except StopTraining:
        return True
    return False


def _result(
    *,
    history: list[TrainingRecord],
    checkpoints: dict[int, TrainingCheckpoint],
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    samples_seen: int,
    optimizer_steps: int,
    critic: nn.Module | None = None,
    critic_optimizer: torch.optim.Optimizer | None = None,
    data_generator_state: Tensor | None = None,
    action_generator_state: Tensor | None = None,
) -> TrainingResult:
    return TrainingResult(
        history=history,
        checkpoints=checkpoints,
        final_model_state=_state_dict(model),
        optimizer_state=_cpu_clone(optimizer.state_dict()),
        actor_trainable_parameters=trainable_parameter_count(model),
        samples_seen=samples_seen,
        optimizer_steps=optimizer_steps,
        optimizer=optimizer,
        critic=critic,
        critic_state=_state_dict(critic) if critic is not None else None,
        critic_optimizer_state=(
            _cpu_clone(critic_optimizer.state_dict()) if critic_optimizer is not None else None
        ),
        critic_parameter_count=(
            sum(parameter.numel() for parameter in critic.parameters()) if critic is not None else 0
        ),
        critic_optimizer=critic_optimizer,
        data_generator_state=(
            data_generator_state.detach().cpu().clone()
            if data_generator_state is not None
            else None
        ),
        action_generator_state=(
            action_generator_state.detach().cpu().clone()
            if action_generator_state is not None
            else None
        ),
    )


def train_sft(
    model: nn.Module,
    data: Any,
    config: SFTConfig | None = None,
    *,
    optimizer: torch.optim.Optimizer | None = None,
    evaluator: Evaluator | None = None,
    callback: TrainingCallback | Sequence[TrainingCallback] | None = None,
) -> TrainingResult:
    """Deterministic binary SFT, optionally with nuisance-prediction loss.

    Dynamic data sources may be callables (or expose ``sample_batch``) accepting
    any of ``batch_size``, ``generator``/``rng``, and ``step``.  This is useful
    for independently resampled noise.  Static tensor datasets are traversed in
    deterministic shuffled epochs.
    """

    config = config or SFTConfig()
    callbacks = _callbacks(callback)
    log_steps, checkpoint_steps = _resolved_steps(config)
    history: list[TrainingRecord] = []
    checkpoints: dict[int, TrainingCheckpoint] = {}
    samples_seen = 0
    optimizer_steps = 0

    with _deterministic_context(config.seed, config.deterministic):
        device = _resolve_device(model, config.device)
        if config.reset_model:
            reset_model_parameters(model)
        optimizer = _prepare_optimizer(model, config, optimizer)
        generator = torch.Generator(device="cpu")
        generator.manual_seed(config.seed)
        batcher = None
        if not _is_dynamic_source(data):
            batcher = _StaticBatcher(
                data,
                config.batch_size,
                generator,
                shuffle=config.shuffle,
                require_target=True,
            )

        model.train()
        for step in range(1, config.steps + 1):
            raw = (
                _sample_dynamic(
                    data,
                    batch_size=config.batch_size,
                    generator=generator,
                    step=step,
                )
                if batcher is None
                else batcher.next()
            )
            batch = _to_device(_normalise_batch(raw, require_target=True), device)
            assert batch.y is not None
            optimizer.zero_grad(set_to_none=True)
            if config.auxiliary_weight > 0:
                try:
                    output = model(batch.x, return_aux=True)
                except TypeError as exc:
                    raise TypeError(
                        "nuisance-rich SFT requires model(x, return_aux=True) support"
                    ) from exc
            else:
                output = model(batch.x)
            logits = _goal_logits(output)
            primary_loss = _binary_loss(logits, batch.y, config.label_smoothing)
            auxiliary_loss = (
                _auxiliary_loss(output, batch.nuisance, config.nuisance_weights)
                if config.auxiliary_weight > 0
                else primary_loss.detach() * 0.0
            )
            loss = primary_loss + config.auxiliary_weight * auxiliary_loss
            loss.backward()
            _clip_gradients(model, config.gradient_clip_norm)
            optimizer.step()

            batch_size = len(batch.x)
            samples_seen += batch_size
            optimizer_steps += 1
            if step not in log_steps and step not in checkpoint_steps:
                continue

            accuracy = float(
                ((logits.detach() >= 0) == (_binary_targets(batch.y) >= 0.5)).float().mean().cpu()
            )
            metrics = {
                "train_accuracy": accuracy,
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
            }
            if step in log_steps:
                metrics.update(_evaluate(evaluator, model, step))
            record = TrainingRecord(
                step=step,
                loss=float(loss.detach().cpu()),
                primary_loss=float(primary_loss.detach().cpu()),
                auxiliary_loss=float(auxiliary_loss.detach().cpu()),
                samples_seen=samples_seen,
                optimizer_steps=optimizer_steps,
                metrics=metrics,
            )
            saved = None
            if step in checkpoint_steps:
                saved = _checkpoint(step, model, optimizer, generator)
                checkpoints[step] = saved
            if step in log_steps:
                history.append(record)
                if _emit(callbacks, record, model, saved):
                    break

    return _result(
        history=history,
        checkpoints=checkpoints,
        model=model,
        optimizer=optimizer,
        samples_seen=samples_seen,
        optimizer_steps=optimizer_steps,
    )


def train_clean_sft(
    model: nn.Module,
    data: Any,
    config: SFTConfig | None = None,
    **kwargs: Any,
) -> TrainingResult:
    """Named clean-label SFT condition (forces auxiliary weight to zero)."""

    config = config or SFTConfig()
    return train_sft(model, data, replace(config, auxiliary_weight=0.0), **kwargs)


def train_nuisance_sft(
    model: nn.Module,
    data: Any,
    config: SFTConfig | None = None,
    **kwargs: Any,
) -> TrainingResult:
    """Named nuisance-rich SFT condition (defaults auxiliary weight to one)."""

    config = config or SFTConfig(auxiliary_weight=1.0)
    if config.auxiliary_weight <= 0:
        raise ValueError("nuisance-rich SFT requires auxiliary_weight > 0")
    return train_sft(model, data, config, **kwargs)


class _ReplayBuffer:
    def __init__(self, capacity: int | None) -> None:
        self.capacity = capacity
        self.x = torch.empty(0)
        self.y = torch.empty(0)

    def append(self, x: Tensor, y: Tensor) -> None:
        x = x.detach().cpu()
        y = y.detach().cpu()
        if self.x.numel() == 0:
            self.x, self.y = x, y
        else:
            self.x = torch.cat((self.x, x), dim=0)
            self.y = torch.cat((self.y, y), dim=0)
        if self.capacity is not None and len(self.x) > self.capacity:
            self.x = self.x[-self.capacity :]
            self.y = self.y[-self.capacity :]

    def sample(self, size: int, generator: torch.Generator) -> _Batch:
        if len(self.x) <= size:
            index = torch.randperm(len(self.x), generator=generator)
        else:
            index = torch.randperm(len(self.x), generator=generator)[:size]
        return _Batch(x=self.x[index], y=self.y[index])

    def __len__(self) -> int:
        return len(self.x)


def _oracle_labels(
    oracle: Callable[..., Any],
    collected: _Batch,
    *,
    step: int,
    generator: torch.Generator,
) -> Tensor:
    available = {
        "x": collected.x,
        "features": collected.x,
        "contexts": collected.x,
        "batch": collected.raw,
        "raw_batch": collected.raw,
        "step": step,
        "generator": generator,
        "rng": generator,
    }
    result = _call_adaptively(oracle, available)
    if torch.is_tensor(result):
        return result
    normalised = _normalise_batch(result, require_target=True)
    assert normalised.y is not None
    return normalised.y


def train_on_policy_imitation(
    model: nn.Module,
    collector: Callable[..., Any],
    config: OnPolicyImitationConfig | None = None,
    *,
    oracle: Callable[..., Any] | None = None,
    optimizer: torch.optim.Optimizer | None = None,
    evaluator: Evaluator | None = None,
    callback: TrainingCallback | Sequence[TrainingCallback] | None = None,
) -> TrainingResult:
    """DAgger-like contextual imitation from states visited by the policy.

    ``collector(model, batch_size, generator, step)`` may return labeled batches
    directly.  If it returns contexts only, provide ``oracle``; the resulting
    labels and states are accumulated in a deterministic replay buffer.
    """

    config = config or OnPolicyImitationConfig()
    callbacks = _callbacks(callback)
    log_steps, checkpoint_steps = _resolved_steps(config)
    history: list[TrainingRecord] = []
    checkpoints: dict[int, TrainingCheckpoint] = {}
    replay = _ReplayBuffer(config.replay_capacity)
    samples_seen = 0
    optimizer_steps = 0
    collection_size = config.collection_batch_size or config.batch_size

    with _deterministic_context(config.seed, config.deterministic):
        device = _resolve_device(model, config.device)
        if config.reset_model:
            reset_model_parameters(model)
        optimizer = _prepare_optimizer(model, config, optimizer)
        generator = torch.Generator(device="cpu")
        generator.manual_seed(config.seed)

        for step in range(1, config.steps + 1):
            was_training = model.training
            model.eval()
            with torch.no_grad():
                raw = _sample_dynamic(
                    collector,
                    batch_size=collection_size,
                    generator=generator,
                    step=step,
                    model=model,
                )
            model.train(was_training)
            collected = _normalise_batch(raw)
            if collected.y is None:
                if oracle is None:
                    raise ValueError("collector returned no labels and no oracle was provided")
                collected.y = _oracle_labels(
                    oracle, collected, step=step, generator=generator
                )
            if len(collected.x) != len(collected.y):
                raise ValueError("collector contexts and oracle labels differ in length")
            replay.append(collected.x, collected.y)
            samples_seen += len(collected.x)

            model.train()
            for _ in range(config.updates_per_collection):
                batch = _to_device(replay.sample(config.batch_size, generator), device)
                assert batch.y is not None
                optimizer.zero_grad(set_to_none=True)
                logits = _goal_logits(model(batch.x))
                loss = _binary_loss(logits, batch.y, config.label_smoothing)
                loss.backward()
                _clip_gradients(model, config.gradient_clip_norm)
                optimizer.step()
                optimizer_steps += 1

            if step not in log_steps and step not in checkpoint_steps:
                continue
            accuracy = float(
                ((logits.detach() >= 0) == (_binary_targets(batch.y) >= 0.5)).float().mean().cpu()
            )
            metrics = {
                "train_accuracy": accuracy,
                "replay_size": float(len(replay)),
                "labeled_actions": float(samples_seen),
                "environment_interactions": float(samples_seen),
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
            }
            if step in log_steps:
                metrics.update(_evaluate(evaluator, model, step))
            record = TrainingRecord(
                step=step,
                loss=float(loss.detach().cpu()),
                primary_loss=float(loss.detach().cpu()),
                samples_seen=samples_seen,
                optimizer_steps=optimizer_steps,
                metrics=metrics,
            )
            saved = None
            if step in checkpoint_steps:
                saved = _checkpoint(step, model, optimizer, generator)
                checkpoints[step] = saved
            if step in log_steps:
                history.append(record)
                if _emit(callbacks, record, model, saved):
                    break

    return _result(
        history=history,
        checkpoints=checkpoints,
        model=model,
        optimizer=optimizer,
        samples_seen=samples_seen,
        optimizer_steps=optimizer_steps,
    )


# Common name in experiment configurations.
train_dagger = train_on_policy_imitation


def _action_generator(device: torch.device, seed: int) -> torch.Generator:
    generator_device = device if device.type == "cuda" else torch.device("cpu")
    generator = torch.Generator(device=generator_device)
    generator.manual_seed(seed)
    return generator


def _reward_from_hook(
    reward_fn: Callable[..., Any],
    batch: _Batch,
    actions: Tensor,
    *,
    step: int,
    generator: torch.Generator,
    branch_actions: Tensor | None = None,
) -> Tensor:
    available = {
        "batch": batch.raw,
        "raw_batch": batch.raw,
        "x": batch.x,
        "features": batch.x,
        "contexts": batch.x,
        "actions": actions,
        "action": actions,
        "branch_actions": branch_actions,
        "neutral_actions": branch_actions,
        "step": step,
        "generator": generator,
        "rng": generator,
    }
    result = _call_adaptively(reward_fn, available)
    if isinstance(result, Mapping):
        result = result.get("reward", result.get("rewards"))
    if result is None:
        raise ValueError("reward function returned no reward tensor")
    rewards = result if torch.is_tensor(result) else torch.as_tensor(result)
    return rewards


def _default_rewards(batch: _Batch, actions: Tensor) -> Tensor:
    if batch.y is None:
        raise ValueError("contexts have no targets; provide reward_fn for bandit training")
    targets = _binary_targets(batch.y).to(actions.device).long()
    return (actions == targets).float()


def _critic_optimizer(
    critic: nn.Module,
    config: BanditConfig,
) -> torch.optim.Optimizer:
    parameters = [parameter for parameter in critic.parameters() if parameter.requires_grad]
    if not parameters:
        raise ValueError("critic has no trainable parameters")
    return torch.optim.AdamW(
        parameters,
        lr=config.critic_learning_rate,
        weight_decay=config.critic_weight_decay,
    )


def _prepare_critic_optimizer(
    critic: nn.Module,
    config: BanditConfig,
    optimizer: torch.optim.Optimizer | None,
) -> torch.optim.Optimizer:
    if optimizer is None:
        return _critic_optimizer(critic, config)
    required = {id(parameter) for parameter in critic.parameters() if parameter.requires_grad}
    present = {
        id(parameter)
        for group in optimizer.param_groups
        for parameter in group.get("params", ())
    }
    if required - present:
        raise ValueError("provided critic optimizer omits trainable critic parameters")
    if config.reset_optimizer:
        optimizer.state.clear()
    for group in optimizer.param_groups:
        group["lr"] = config.critic_learning_rate
        group["weight_decay"] = config.critic_weight_decay
    return optimizer


def train_contextual_bandit(
    actor: nn.Module,
    contexts: Any,
    config: BanditConfig | None = None,
    *,
    reward_fn: Callable[..., Any] | None = None,
    critic: nn.Module | None = None,
    actor_optimizer: torch.optim.Optimizer | None = None,
    critic_optimizer: torch.optim.Optimizer | None = None,
    evaluator: Evaluator | None = None,
    callback: TrainingCallback | Sequence[TrainingCallback] | None = None,
    factorized_actions: bool = False,
    factorized_action_heads: int | None = None,
    data_generator_state: Tensor | None = None,
    action_generator_state: Tensor | None = None,
) -> TrainingResult:
    """Train a binary actor with one-step or factorized REINFORCE/actor-critic.

    The critic is a wholly separate network and is excluded from
    ``actor_trainable_parameters``.  If omitted in actor-critic mode, a fixed,
    deliberately over-provisioned ``ValueMLP(width=256, depth=3)`` is created.
    This prevents tiny actor update budgets from also shrinking the advantage
    estimator. Custom reward hooks receive zero/one sampled goal actions. With
    ``factorized_actions=True``, every binary nuisance head is instead treated as
    a policy factor preceding the final goal: those branch actions are sampled
    from the actor, their log-probabilities enter the policy gradient, and the
    hook additionally receives them as ``branch_actions``. Set
    ``factorized_action_heads`` to use only the first configured heads as
    stochastic policy factors while retaining every head in the actor
    architecture. Batch nuisance values are never consulted in this mode, so it
    cannot become branch imitation.  The optional generator states continue
    sampling across an explicitly staged dynamic source.  Exact continuation
    additionally requires reusing that source object so its cursor is retained.
    """

    config = config or BanditConfig()
    if factorized_action_heads is not None and (
        isinstance(factorized_action_heads, bool)
        or not isinstance(factorized_action_heads, int)
        or factorized_action_heads < 0
        or not factorized_actions
    ):
        raise ValueError(
            "factorized_action_heads must be a non-negative integer used with "
            "factorized_actions=True"
        )
    if config.algorithm == "reinforce" and (critic is not None or critic_optimizer is not None):
        raise ValueError("critic arguments are only used with algorithm='actor_critic'")
    callbacks = _callbacks(callback)
    log_steps, checkpoint_steps = _resolved_steps(config)
    history: list[TrainingRecord] = []
    checkpoints: dict[int, TrainingCheckpoint] = {}
    samples_seen = 0
    optimizer_steps = 0

    with _deterministic_context(config.seed, config.deterministic):
        device = _resolve_device(actor, config.device)
        if config.reset_model:
            reset_model_parameters(actor)
        actor_optimizer = _prepare_optimizer(actor, config, actor_optimizer)
        data_generator = torch.Generator(device="cpu")
        if data_generator_state is None:
            data_generator.manual_seed(config.seed)
        else:
            data_generator.set_state(data_generator_state.detach().cpu())
        action_generator = _action_generator(device, config.seed + 1)
        if action_generator_state is not None:
            action_generator.set_state(action_generator_state.detach().cpu())
        batcher = None
        if not _is_dynamic_source(contexts):
            batcher = _StaticBatcher(
                contexts,
                config.batch_size,
                data_generator,
                shuffle=config.shuffle,
                require_target=reward_fn is None,
            )

        critic_instance = critic
        critic_optim = critic_optimizer
        actor.train()
        for step in range(1, config.steps + 1):
            raw = (
                _sample_dynamic(
                    contexts,
                    batch_size=config.batch_size,
                    generator=data_generator,
                    step=step,
                    model=actor,
                )
                if batcher is None
                else batcher.next()
            )
            batch = _to_device(_normalise_batch(raw), device)
            if config.algorithm == "actor_critic" and critic_instance is None:
                if batch.x.ndim != 2:
                    raise ValueError("automatic critic requires flat [batch, features] contexts")
                critic_instance = ValueMLP(
                    input_dim=batch.x.shape[-1],
                    width=config.critic_width,
                    depth=config.critic_depth,
                    activation=config.critic_activation,
                    residual=config.critic_residual,
                ).to(device)
                critic_optim = _critic_optimizer(critic_instance, config)
            elif config.algorithm == "actor_critic" and step == 1:
                assert critic_instance is not None
                critic_instance.to(device)
                if config.reset_critic:
                    reset_model_parameters(critic_instance)
                critic_optim = _prepare_critic_optimizer(
                    critic_instance, config, critic_optim
                )

            if factorized_actions:
                try:
                    output = actor(batch.x, return_aux=True)
                except TypeError as exc:
                    raise TypeError(
                        "factorized bandit training requires model(x, return_aux=True) support"
                    ) from exc
                logits = _goal_logits(output)
                branch_logits_by_head = _nuisance_logits(output)
                selected_heads = list(branch_logits_by_head.items())
                if factorized_action_heads is not None:
                    if factorized_action_heads > len(selected_heads):
                        raise ValueError(
                            f"requested {factorized_action_heads} factorized heads, but the actor "
                            f"exposes only {len(selected_heads)}"
                        )
                    selected_heads = selected_heads[:factorized_action_heads]
                branch_logits: list[Tensor] = []
                for name, branch_logit in selected_heads:
                    if branch_logit.ndim == 2 and branch_logit.shape[-1] == 1:
                        branch_logit = branch_logit.squeeze(-1)
                    if branch_logit.shape != logits.shape:
                        raise ValueError(
                            f"factorized binary head {name!r} has shape {tuple(branch_logit.shape)}; "
                            f"expected {tuple(logits.shape)}"
                        )
                    branch_logits.append(branch_logit)
                factor_logits = torch.stack((*branch_logits, logits), dim=-1)
            else:
                logits = _goal_logits(actor(batch.x))
                factor_logits = logits.unsqueeze(-1)
            factor_probabilities = factor_logits.sigmoid()
            uniforms = torch.rand(
                factor_probabilities.shape,
                dtype=factor_probabilities.dtype,
                device=factor_probabilities.device,
                generator=action_generator,
            )
            factor_actions = (uniforms < factor_probabilities).long()
            branch_actions = factor_actions[:, :-1] if factorized_actions else None
            actions = factor_actions[:, -1]
            rewards = (
                _reward_from_hook(
                    reward_fn,
                    batch,
                    actions,
                    step=step,
                    generator=action_generator,
                    branch_actions=branch_actions,
                )
                if reward_fn is not None
                else _default_rewards(batch, actions)
            )
            rewards = rewards.to(device=device, dtype=logits.dtype).reshape(logits.shape).detach()
            rewards = rewards * config.reward_scale
            factor_log_probabilities = -F.binary_cross_entropy_with_logits(
                factor_logits, factor_actions.to(factor_logits.dtype), reduction="none"
            )
            log_probabilities = factor_log_probabilities.sum(dim=-1)
            factor_entropy = -(
                factor_probabilities * F.logsigmoid(factor_logits)
                + (1.0 - factor_probabilities) * F.logsigmoid(-factor_logits)
            )
            trajectory_entropy = factor_entropy.sum(dim=-1)

            value_loss = logits.detach().sum() * 0.0
            if config.algorithm == "actor_critic":
                assert critic_instance is not None and critic_optim is not None
                critic_instance.train()
                values = _goal_logits(critic_instance(batch.x))
                advantages = rewards - values.detach()
            else:
                advantages = rewards
                if config.center_reinforce_rewards and len(rewards) > 1:
                    advantages = advantages - advantages.mean()
            if config.normalize_advantages and len(advantages) > 1:
                std = advantages.std(unbiased=False)
                if float(std.detach().cpu()) > 1e-8:
                    advantages = (advantages - advantages.mean()) / std

            actor_loss = -(log_probabilities * advantages.detach()).mean()
            actor_loss = actor_loss - config.entropy_coefficient * trajectory_entropy.mean()
            actor_optimizer.zero_grad(set_to_none=True)
            actor_loss.backward()
            _clip_gradients(actor, config.gradient_clip_norm)
            actor_optimizer.step()
            optimizer_steps += 1

            if config.algorithm == "actor_critic":
                assert critic_instance is not None and critic_optim is not None
                for critic_update in range(config.critic_updates_per_step):
                    if critic_update:
                        values = _goal_logits(critic_instance(batch.x))
                    value_loss = F.mse_loss(values, rewards)
                    critic_optim.zero_grad(set_to_none=True)
                    value_loss.backward()
                    nn.utils.clip_grad_norm_(critic_instance.parameters(), max_norm=10.0)
                    critic_optim.step()

            batch_size = len(batch.x)
            samples_seen += batch_size
            if step not in log_steps and step not in checkpoint_steps:
                continue
            metrics = {
                "reward_mean": float(rewards.mean().detach().cpu()),
                "policy_entropy": float(factor_entropy[:, -1].mean().detach().cpu()),
                "trajectory_policy_entropy": float(trajectory_entropy.mean().detach().cpu()),
                "branch_policy_entropy": float(
                    factor_entropy[:, :-1].mean().detach().cpu()
                    if factorized_actions and factor_entropy.shape[1] > 1
                    else 0.0
                ),
                "learned_action_factors": float(factor_logits.shape[1]),
                "value_loss": float(value_loss.detach().cpu()),
                "environment_interactions": float(samples_seen),
                "actor_trainable_parameters": float(trainable_parameter_count(actor)),
                "critic_parameters": float(
                    sum(parameter.numel() for parameter in critic_instance.parameters())
                    if critic_instance is not None
                    else 0
                ),
                "learning_rate": float(actor_optimizer.param_groups[0]["lr"]),
            }
            if step in log_steps:
                metrics.update(_evaluate(evaluator, actor, step))
            record = TrainingRecord(
                step=step,
                loss=float(actor_loss.detach().cpu()),
                primary_loss=float(actor_loss.detach().cpu()),
                auxiliary_loss=float(value_loss.detach().cpu()),
                samples_seen=samples_seen,
                optimizer_steps=optimizer_steps,
                metrics=metrics,
            )
            saved = None
            if step in checkpoint_steps:
                saved = _checkpoint(
                    step,
                    actor,
                    actor_optimizer,
                    data_generator,
                    critic=critic_instance,
                    critic_optimizer=critic_optim,
                    action_generator=action_generator,
                )
                checkpoints[step] = saved
            if step in log_steps:
                history.append(record)
                if _emit(callbacks, record, actor, saved, critic_instance):
                    break

    return _result(
        history=history,
        checkpoints=checkpoints,
        model=actor,
        optimizer=actor_optimizer,
        samples_seen=samples_seen,
        optimizer_steps=optimizer_steps,
        critic=critic_instance,
        critic_optimizer=critic_optim,
        data_generator_state=data_generator.get_state(),
        action_generator_state=action_generator.get_state(),
    )


# Concise aliases used by runners and papers.
train_bandit = train_contextual_bandit
train_actor_critic = train_contextual_bandit


__all__ = [
    "BanditConfig",
    "OnPolicyImitationConfig",
    "SFTConfig",
    "StopTraining",
    "TrainConfig",
    "TrainingCheckpoint",
    "TrainingRecord",
    "TrainingResult",
    "TrainingSnapshot",
    "log_spaced_steps",
    "reset_model_parameters",
    "restore_checkpoint",
    "train_actor_critic",
    "train_bandit",
    "train_clean_sft",
    "train_contextual_bandit",
    "train_dagger",
    "train_nuisance_sft",
    "train_on_policy_imitation",
    "train_sft",
]
