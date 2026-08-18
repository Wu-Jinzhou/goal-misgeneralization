"""Goal-conditioned BFS supervision and navigation evaluation.

The selector and navigator are intentionally separate.  A selector emits a
semantic :class:`~forkworld.envs.Goal`; :class:`NavigatorMLP` receives that goal
explicitly and emits one of four primitive grid actions.  Evaluation therefore
reports both whether the selector chose the intended goal and whether the frozen
navigator was capable of reaching whichever goal it was given.
"""

from __future__ import annotations

import json
import os
import random
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

from .envs import (
    GRID_ACTIONS,
    Coordinate,
    Goal,
    GridAction,
    GridObservation,
    GridWorld,
    coerce_goal,
)


def _activation(name: str) -> nn.Module:
    activations: dict[str, type[nn.Module]] = {
        "relu": nn.ReLU,
        "gelu": nn.GELU,
        "silu": nn.SiLU,
        "tanh": nn.Tanh,
    }
    try:
        return activations[name.lower()]()
    except KeyError as exc:
        raise ValueError(
            f"unknown activation {name!r}; choose from {sorted(activations)}"
        ) from exc


def _goal_tensor(
    goal_ids: Tensor | Sequence[int] | Goal | int,
    *,
    batch_size: int,
    device: torch.device,
) -> Tensor:
    if isinstance(goal_ids, Tensor):
        goals = goal_ids.to(device=device)
    elif isinstance(goal_ids, (Goal, int)) and not isinstance(goal_ids, bool):
        goals = torch.tensor([int(coerce_goal(goal_ids))], device=device)
    else:
        goals = torch.as_tensor(goal_ids, device=device)
    goals = goals.reshape(-1)
    if goals.numel() == 1 and batch_size != 1:
        goals = goals.expand(batch_size)
    if goals.numel() != batch_size:
        raise ValueError(
            f"received {goals.numel()} goal IDs for an observation batch of {batch_size}"
        )
    if not bool(torch.all((goals == -1) | (goals == 1))):
        raise ValueError("all goal IDs must be -1 or +1")
    indices = (goals > 0).long()
    return F.one_hot(indices, num_classes=2).float()


class NavigatorMLP(nn.Module):
    """MLP policy over primitive actions, conditioned on an explicit goal ID."""

    def __init__(
        self,
        observation_dim: int,
        *,
        hidden_sizes: Sequence[int] = (128, 128),
        activation: str = "relu",
        num_actions: int = len(GRID_ACTIONS),
    ) -> None:
        super().__init__()
        if observation_dim < 1:
            raise ValueError("observation_dim must be positive")
        if num_actions < 2:
            raise ValueError("num_actions must be at least two")
        if any(int(size) < 1 for size in hidden_sizes):
            raise ValueError("hidden layer sizes must be positive")
        self.observation_dim = int(observation_dim)
        self.hidden_sizes = tuple(int(size) for size in hidden_sizes)
        self.activation_name = activation.lower()
        self.num_actions = int(num_actions)

        dimensions = (self.observation_dim + 2, *self.hidden_sizes, self.num_actions)
        layers: list[nn.Module] = []
        for input_size, output_size in zip(dimensions[:-2], dimensions[1:-1]):
            layers.extend((nn.Linear(input_size, output_size), _activation(self.activation_name)))
        layers.append(nn.Linear(dimensions[-2], dimensions[-1]))
        self.network = nn.Sequential(*layers)
        self.checkpoint_metadata: dict[str, Any] = {}

    def config_dict(self) -> dict[str, Any]:
        return {
            "observation_dim": self.observation_dim,
            "hidden_sizes": list(self.hidden_sizes),
            "activation": self.activation_name,
            "num_actions": self.num_actions,
        }

    def forward(
        self,
        observations: Tensor | Sequence[float] | Sequence[Sequence[float]],
        goal_ids: Tensor | Sequence[int] | Goal | int,
    ) -> Tensor:
        parameter = next(self.parameters())
        features = torch.as_tensor(
            observations,
            dtype=parameter.dtype,
            device=parameter.device,
        )
        if features.ndim == 1:
            features = features.unsqueeze(0)
        if features.ndim != 2 or features.shape[1] != self.observation_dim:
            raise ValueError(
                "observations must have shape "
                f"[batch, {self.observation_dim}], got {tuple(features.shape)}"
            )
        goals = _goal_tensor(
            goal_ids,
            batch_size=features.shape[0],
            device=features.device,
        ).to(dtype=features.dtype)
        return self.network(torch.cat((features, goals), dim=-1))

    @torch.no_grad()
    def predict_action(
        self,
        observation: GridObservation | Sequence[float] | Tensor,
        goal_id: Goal | int,
    ) -> GridAction:
        if isinstance(observation, GridObservation):
            features: Sequence[float] | Tensor = observation.vector()
        else:
            features = observation
        was_training = self.training
        self.eval()
        logits = self(features, goal_id)
        action_index = int(logits.argmax(dim=-1).item())
        if was_training:
            self.train()
        return GridAction(action_index)

    def freeze(self) -> NavigatorMLP:
        self.requires_grad_(False)
        self.eval()
        return self

    def unfreeze(self) -> NavigatorMLP:
        self.requires_grad_(True)
        self.train()
        return self

    def save_checkpoint(
        self,
        path: str | Path,
        *,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        save_navigator_checkpoint(self, path, metadata=metadata)

    @classmethod
    def load_checkpoint(
        cls,
        path: str | Path,
        *,
        map_location: str | torch.device = "cpu",
        freeze: bool = True,
    ) -> tuple[NavigatorMLP, dict[str, Any]]:
        return load_navigator_checkpoint(path, map_location=map_location, freeze=freeze)


def save_navigator_checkpoint(
    model: NavigatorMLP,
    path: str | Path,
    *,
    metadata: Mapping[str, Any] | None = None,
) -> None:
    """Save architecture, weights, and JSON-compatible provenance together."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format_version": 1,
        "model_config": model.config_dict(),
        "state_dict": model.state_dict(),
        "metadata": dict(metadata or {}),
    }
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    os.close(descriptor)
    try:
        torch.save(payload, temporary)
        os.replace(temporary, destination)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def load_navigator_checkpoint(
    path: str | Path,
    *,
    map_location: str | torch.device = "cpu",
    freeze: bool = True,
) -> tuple[NavigatorMLP, dict[str, Any]]:
    """Restore a navigator and metadata, freezing it by default."""

    try:
        payload = torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:  # PyTorch < 2.0 does not expose ``weights_only``.
        payload = torch.load(path, map_location=map_location)
    if not isinstance(payload, Mapping) or payload.get("format_version") != 1:
        raise ValueError(f"unsupported navigator checkpoint at {path}")
    model_config = payload.get("model_config")
    state_dict = payload.get("state_dict")
    if not isinstance(model_config, Mapping) or not isinstance(state_dict, Mapping):
        raise ValueError("navigator checkpoint is missing model_config or state_dict")
    model = NavigatorMLP(**dict(model_config))
    model.load_state_dict(state_dict)
    metadata = dict(payload.get("metadata") or {})
    model.checkpoint_metadata = metadata
    if freeze:
        model.freeze()
    return model, metadata


@dataclass(frozen=True)
class BFSExample:
    observation: tuple[float, ...]
    goal_id: Goal
    action: GridAction
    semantic_id: str
    position: Coordinate


class BFSDataset(Dataset[tuple[Tensor, Tensor, Tensor]]):
    """Materialized deterministic BFS labels for supervised navigation."""

    def __init__(self, examples: Sequence[BFSExample]) -> None:
        if not examples:
            raise ValueError("BFS dataset cannot be empty")
        observation_dim = len(examples[0].observation)
        if any(len(example.observation) != observation_dim for example in examples):
            raise ValueError("all BFS observations must have the same dimension")
        self.examples = tuple(examples)
        self.observation_dim = observation_dim
        self.x = torch.tensor(
            [example.observation for example in self.examples],
            dtype=torch.float32,
        )
        self.goals = torch.tensor(
            [int(example.goal_id) for example in self.examples],
            dtype=torch.int64,
        )
        self.y = torch.tensor(
            [int(example.action) for example in self.examples],
            dtype=torch.int64,
        )

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor, Tensor]:
        return self.x[index], self.goals[index], self.y[index]

    def features(self) -> Tensor:
        return self.x

    @property
    def target(self) -> Tensor:
        return self.y


def build_bfs_dataset(
    environments: Iterable[GridWorld],
    *,
    goals: Sequence[Goal | int] = (Goal.LEFT, Goal.RIGHT),
    start_states_only: bool = False,
) -> BFSDataset:
    """Label reachable states with the first action of a shortest path.

    Target cells are excluded because grid episodes terminate on reaching either
    target.  A state with no target-safe route for a requested goal is omitted.
    """

    normalized_goals = tuple(coerce_goal(goal) for goal in goals)
    examples: list[BFSExample] = []
    expected_dim: int | None = None
    for env in environments:
        if expected_dim is None:
            expected_dim = env.observation_dim
        elif env.observation_dim != expected_dim:
            raise ValueError(
                "all environments in one BFS dataset must have the same observation shape"
            )
        positions = (env.start,) if start_states_only else env.free_cells
        terminal_cells = set(env.targets.values())
        for position in positions:
            if position in terminal_cells:
                continue
            observation = env.observation_at(position).vector()
            for goal in normalized_goals:
                path = env.shortest_path(goal, start=position)
                if not path:
                    continue
                examples.append(
                    BFSExample(
                        observation=observation,
                        goal_id=goal,
                        action=path[0],
                        semantic_id=env.semantic_id,
                        position=position,
                    )
                )
    return BFSDataset(examples)


@dataclass(frozen=True)
class NavigatorTrainingConfig:
    epochs: int = 250
    batch_size: int = 128
    learning_rate: float = 3e-3
    weight_decay: float = 0.0
    seed: int = 0
    device: str = "cpu"
    target_accuracy: float = 1.0
    target_patience: int = 5
    max_grad_norm: float | None = 5.0

    def __post_init__(self) -> None:
        if self.epochs < 1 or self.batch_size < 1:
            raise ValueError("epochs and batch_size must be positive")
        if self.learning_rate <= 0.0 or self.weight_decay < 0.0:
            raise ValueError("invalid optimizer hyperparameters")
        if not 0.0 < self.target_accuracy <= 1.0:
            raise ValueError("target_accuracy must lie in (0, 1]")
        if self.target_patience < 1:
            raise ValueError("target_patience must be positive")


@dataclass(frozen=True)
class NavigatorEpochMetrics:
    epoch: int
    loss: float
    accuracy: float


@dataclass
class NavigatorTrainingResult:
    model: NavigatorMLP
    history: tuple[NavigatorEpochMetrics, ...]
    dataset_size: int
    final_accuracy: float
    config: NavigatorTrainingConfig

    def metrics_dict(self) -> dict[str, Any]:
        return {
            "dataset_size": self.dataset_size,
            "final_accuracy": self.final_accuracy,
            "epochs_completed": len(self.history),
            "config": asdict(self.config),
            "history": [asdict(row) for row in self.history],
        }

    def save_metrics(self, path: str | Path) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            json.dumps(self.metrics_dict(), indent=2, sort_keys=True),
            encoding="utf-8",
        )


@torch.no_grad()
def bfs_dataset_accuracy(model: NavigatorMLP, dataset: BFSDataset) -> float:
    parameter = next(model.parameters())
    was_training = model.training
    model.eval()
    logits = model(
        dataset.x.to(parameter.device),
        dataset.goals.to(parameter.device),
    )
    accuracy = (logits.argmax(dim=-1).cpu() == dataset.y).float().mean().item()
    if was_training:
        model.train()
    return float(accuracy)


def fit_bfs_navigator(
    model: NavigatorMLP,
    dataset: BFSDataset,
    *,
    config: NavigatorTrainingConfig | None = None,
) -> NavigatorTrainingResult:
    """Fit a navigator with strict deterministic torch kernels."""

    previous = torch.are_deterministic_algorithms_enabled()
    previous_warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    torch.use_deterministic_algorithms(True, warn_only=False)
    try:
        return _fit_bfs_navigator(model, dataset, config=config)
    finally:
        torch.use_deterministic_algorithms(previous, warn_only=previous_warn_only)


def _fit_bfs_navigator(
    model: NavigatorMLP,
    dataset: BFSDataset,
    *,
    config: NavigatorTrainingConfig | None = None,
) -> NavigatorTrainingResult:
    """Implementation for :func:`fit_bfs_navigator`."""

    cfg = config or NavigatorTrainingConfig()
    if model.observation_dim != dataset.observation_dim:
        raise ValueError(
            f"model observation_dim={model.observation_dim} but dataset has "
            f"{dataset.observation_dim} features"
        )
    random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.seed)
    device = torch.device(cfg.device)
    model.to(device)
    model.unfreeze()
    generator = torch.Generator().manual_seed(cfg.seed)
    loader = DataLoader(
        dataset,
        batch_size=min(cfg.batch_size, len(dataset)),
        shuffle=True,
        generator=generator,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
    )
    history: list[NavigatorEpochMetrics] = []
    consecutive_at_target = 0

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        total_loss = 0.0
        examples_seen = 0
        for observations, goals, actions in loader:
            observations = observations.to(device)
            goals = goals.to(device)
            actions = actions.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(observations, goals)
            loss = F.cross_entropy(logits, actions)
            loss.backward()
            if cfg.max_grad_norm is not None:
                nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
            optimizer.step()
            batch_size = observations.shape[0]
            total_loss += float(loss.detach()) * batch_size
            examples_seen += batch_size

        accuracy = bfs_dataset_accuracy(model, dataset)
        history.append(
            NavigatorEpochMetrics(
                epoch=epoch,
                loss=total_loss / max(1, examples_seen),
                accuracy=accuracy,
            )
        )
        if accuracy + 1e-7 >= cfg.target_accuracy:
            consecutive_at_target += 1
            if consecutive_at_target >= cfg.target_patience:
                break
        else:
            consecutive_at_target = 0

    model.eval()
    final_accuracy = bfs_dataset_accuracy(model, dataset)
    return NavigatorTrainingResult(
        model=model,
        history=tuple(history),
        dataset_size=len(dataset),
        final_accuracy=final_accuracy,
        config=cfg,
    )


def train_bfs_navigator(
    environments: Iterable[GridWorld],
    *,
    hidden_sizes: Sequence[int] = (128, 128),
    activation: str = "relu",
    config: NavigatorTrainingConfig | None = None,
    start_states_only: bool = False,
) -> NavigatorTrainingResult:
    """Build a BFS dataset and train a fresh goal-conditioned navigator."""

    cfg = config or NavigatorTrainingConfig()
    environment_list = tuple(environments)
    dataset = build_bfs_dataset(
        environment_list,
        start_states_only=start_states_only,
    )
    # Set the seed before constructing the model so initialization is reproducible.
    random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    model = NavigatorMLP(
        dataset.observation_dim,
        hidden_sizes=hidden_sizes,
        activation=activation,
    )
    return fit_bfs_navigator(model, dataset, config=cfg)


@dataclass(frozen=True)
class NavigationRollout:
    semantic_id: str
    intended_goal: Goal
    navigation_goal: Goal
    reached_goal: Goal | None
    actions: tuple[GridAction, ...]
    positions: tuple[Coordinate, ...]
    total_reward: float
    terminated: bool
    truncated: bool
    collisions: int
    optimal_steps: int | None

    @property
    def selected_goal_success(self) -> bool:
        """Whether navigation reached the goal supplied to the navigator."""

        return self.reached_goal is self.navigation_goal

    @property
    def intended_goal_success(self) -> bool:
        """End-to-end success against the environment's intended goal."""

        return self.reached_goal is self.intended_goal

    @property
    def path_efficiency(self) -> float:
        if not self.selected_goal_success or self.optimal_steps is None or not self.actions:
            return 0.0
        return self.optimal_steps / len(self.actions)


NavigatorPolicy = NavigatorMLP | Callable[[GridObservation, Goal], GridAction | int]


def _policy_action(
    navigator: NavigatorPolicy,
    observation: GridObservation,
    navigation_goal: Goal,
) -> GridAction:
    if isinstance(navigator, NavigatorMLP):
        return navigator.predict_action(observation, navigation_goal)
    return GridAction(navigator(observation, navigation_goal))


def rollout_navigator(
    env: GridWorld,
    navigator: NavigatorPolicy,
    navigation_goal: Goal | int,
    *,
    intended_goal: Goal | int | None = None,
    max_steps: int | None = None,
) -> NavigationRollout:
    """Roll out a navigator clamped to one goal on a fresh environment episode."""

    nav_goal = coerce_goal(navigation_goal)
    intended = env.goal_id if intended_goal is None else coerce_goal(intended_goal)
    env.reset(goal_id=intended)
    optimal_steps = env.distance_to_goal(nav_goal, start=env.start)
    positions: list[Coordinate] = [env.position]
    actions: list[GridAction] = []
    total_reward = 0.0
    collisions = 0
    horizon = env.max_steps if max_steps is None else min(int(max_steps), env.max_steps)
    if horizon < 1:
        raise ValueError("max_steps must be positive")

    while not env.done and len(actions) < horizon:
        action = _policy_action(navigator, env.observation(), nav_goal)
        result = env.step(action)
        actions.append(action)
        positions.append(env.position)
        total_reward += result.reward
        collisions += int(bool(result.info.get("collision", False)))

    # A caller-specified shorter horizon stops outside the environment itself.
    externally_truncated = not env.done and len(actions) >= horizon
    return NavigationRollout(
        semantic_id=env.semantic_id,
        intended_goal=intended,
        navigation_goal=nav_goal,
        reached_goal=env.reached_goal,
        actions=tuple(actions),
        positions=tuple(positions),
        total_reward=total_reward,
        terminated=env.terminated,
        truncated=env.truncated or externally_truncated,
        collisions=collisions,
        optimal_steps=optimal_steps,
    )


@dataclass(frozen=True)
class NavigationEpisodeEvaluation:
    semantic_id: str
    intended_goal: Goal
    selected_goal: Goal
    selected: NavigationRollout
    oracle: NavigationRollout
    clamped_left: NavigationRollout
    clamped_right: NavigationRollout

    @property
    def selection_correct(self) -> bool:
        return self.selected_goal is self.intended_goal


@dataclass(frozen=True)
class NavigationEvaluation:
    episodes: tuple[NavigationEpisodeEvaluation, ...]

    def __post_init__(self) -> None:
        if not self.episodes:
            raise ValueError("navigation evaluation requires at least one episode")

    @property
    def selection_accuracy(self) -> float:
        return sum(ep.selection_correct for ep in self.episodes) / len(self.episodes)

    @property
    def selected_goal_success_rate(self) -> float:
        return sum(ep.selected.selected_goal_success for ep in self.episodes) / len(
            self.episodes
        )

    @property
    def intended_goal_success_rate(self) -> float:
        return sum(ep.selected.intended_goal_success for ep in self.episodes) / len(
            self.episodes
        )

    @property
    def oracle_goal_success_rate(self) -> float:
        return sum(ep.oracle.intended_goal_success for ep in self.episodes) / len(
            self.episodes
        )

    @property
    def clamped_goal_success_rate(self) -> float:
        successes = sum(
            ep.clamped_left.selected_goal_success + ep.clamped_right.selected_goal_success
            for ep in self.episodes
        )
        return successes / (2 * len(self.episodes))

    @property
    def mean_selected_path_efficiency(self) -> float:
        return sum(ep.selected.path_efficiency for ep in self.episodes) / len(self.episodes)

    @property
    def selected_timeout_rate(self) -> float:
        return sum(ep.selected.truncated for ep in self.episodes) / len(self.episodes)

    @property
    def oracle_timeout_rate(self) -> float:
        return sum(ep.oracle.truncated for ep in self.episodes) / len(self.episodes)

    @property
    def clamped_timeout_rate(self) -> float:
        timeouts = sum(
            ep.clamped_left.truncated + ep.clamped_right.truncated for ep in self.episodes
        )
        return timeouts / (2 * len(self.episodes))

    def summary(self) -> dict[str, float | int]:
        return {
            "episodes": len(self.episodes),
            "selection_accuracy": self.selection_accuracy,
            "selected_goal_success_rate": self.selected_goal_success_rate,
            "intended_goal_success_rate": self.intended_goal_success_rate,
            "oracle_goal_success_rate": self.oracle_goal_success_rate,
            "clamped_goal_success_rate": self.clamped_goal_success_rate,
            "mean_selected_path_efficiency": self.mean_selected_path_efficiency,
            "selected_timeout_rate": self.selected_timeout_rate,
            "oracle_timeout_rate": self.oracle_timeout_rate,
            "clamped_timeout_rate": self.clamped_timeout_rate,
            "mean_selected_collisions": float(
                sum(ep.selected.collisions for ep in self.episodes) / len(self.episodes)
            ),
        }


SelectedGoalSource = Sequence[Goal | int] | Callable[[GridObservation], Goal | int]


def evaluate_navigation(
    navigator: NavigatorPolicy,
    environments: Sequence[GridWorld],
    selected_goals: SelectedGoalSource,
    *,
    intended_goals: Sequence[Goal | int] | None = None,
) -> NavigationEvaluation:
    """Evaluate selected, oracle, and both clamped-goal rollouts on paired maps.

    ``selected_goals`` can be a precomputed sequence or a selector callable that
    maps the initial grid observation to a semantic goal ID.  In the usual
    signal-selection experiment the sequence form is preferable because the
    selector consumes proxy features rather than the navigation observation.
    """

    if not environments:
        raise ValueError("environments cannot be empty")
    if intended_goals is None:
        intended = tuple(env.goal_id for env in environments)
    else:
        if len(intended_goals) != len(environments):
            raise ValueError("intended_goals and environments must have the same length")
        intended = tuple(coerce_goal(goal) for goal in intended_goals)

    if callable(selected_goals):
        selected = tuple(
            coerce_goal(selected_goals(env.clone(goal_id=goal).observation()))
            for env, goal in zip(environments, intended)
        )
    else:
        if len(selected_goals) != len(environments):
            raise ValueError("selected_goals and environments must have the same length")
        selected = tuple(coerce_goal(goal) for goal in selected_goals)

    episode_rows: list[NavigationEpisodeEvaluation] = []
    for env, intended_goal, selected_goal in zip(environments, intended, selected):
        selected_rollout = rollout_navigator(
            env.clone(goal_id=intended_goal),
            navigator,
            selected_goal,
            intended_goal=intended_goal,
        )
        oracle_rollout = rollout_navigator(
            env.clone(goal_id=intended_goal),
            navigator,
            intended_goal,
            intended_goal=intended_goal,
        )
        clamped_left = rollout_navigator(
            env.clone(goal_id=Goal.LEFT),
            navigator,
            Goal.LEFT,
            intended_goal=Goal.LEFT,
        )
        clamped_right = rollout_navigator(
            env.clone(goal_id=Goal.RIGHT),
            navigator,
            Goal.RIGHT,
            intended_goal=Goal.RIGHT,
        )
        episode_rows.append(
            NavigationEpisodeEvaluation(
                semantic_id=env.semantic_id,
                intended_goal=intended_goal,
                selected_goal=selected_goal,
                selected=selected_rollout,
                oracle=oracle_rollout,
                clamped_left=clamped_left,
                clamped_right=clamped_right,
            )
        )
    return NavigationEvaluation(tuple(episode_rows))


__all__ = [
    "BFSDataset",
    "BFSExample",
    "NavigationEpisodeEvaluation",
    "NavigationEvaluation",
    "NavigationRollout",
    "NavigatorEpochMetrics",
    "NavigatorMLP",
    "NavigatorPolicy",
    "NavigatorTrainingConfig",
    "NavigatorTrainingResult",
    "bfs_dataset_accuracy",
    "build_bfs_dataset",
    "evaluate_navigation",
    "fit_bfs_navigator",
    "load_navigator_checkpoint",
    "rollout_navigator",
    "save_navigator_checkpoint",
    "train_bfs_navigator",
]
