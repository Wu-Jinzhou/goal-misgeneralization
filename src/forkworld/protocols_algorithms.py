"""Protocols for H5 (algorithm/update capacity) and H6 (temporal noise).

Both protocols use the same actor builder and binary OOD evaluation.  This is
important: changing the SFT target or the RL objective must not accidentally
change architecture or actor update budget.  RL critics are separately built,
reported, and never included in the actor budget.
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch
from torch import nn

from .config import get_path
from .data import SemanticBatch, make_h5_dataset
from .envs import Goal, NeutralChoiceSimulator
from .metrics import log_evaluation_steps
from .models import ValueMLP
from .noise import NoiseApplication, NoiseConfig, apply_noise_with_diagnostics
from .protocols import (
    ProtocolResult,
    build_model,
    evaluate_batch,
    evaluate_standard_interventions,
    feature_array,
    make_metric_records,
    predict_logits,
    prediction_records,
    resolve_device,
)
from .training import (
    BanditConfig,
    OnPolicyImitationConfig,
    SFTConfig,
    TrainingCheckpoint,
    TrainingResult,
    train_clean_sft,
    train_contextual_bandit,
    train_nuisance_sft,
    train_on_policy_imitation,
)

_ALGORITHMS = {"clean_sft", "trajectory_sft", "on_policy_imitation", "rl"}


def _algorithm(config: Mapping[str, Any], hypothesis: str) -> str:
    value = str(
        get_path(
            config,
            f"{hypothesis}.algorithm",
            get_path(config, "train.algorithm", "clean_sft"),
        )
    ).lower().replace("-", "_")
    aliases = {
        "sft": "clean_sft",
        "clean": "clean_sft",
        "trajectory": "trajectory_sft",
        "nuisance_sft": "trajectory_sft",
        "dagger": "on_policy_imitation",
        "on_policy": "on_policy_imitation",
        "reinforce": "rl",
        "actor_critic": "rl",
    }
    value = aliases.get(value, value)
    if value not in _ALGORITHMS:
        raise ValueError(f"unknown {hypothesis.upper()} algorithm {value!r}")
    return value


def _integer(config: Mapping[str, Any], path: str, default: int, *, minimum: int = 0) -> int:
    value = get_path(config, path, default)
    if isinstance(value, bool) or int(value) != value or int(value) < minimum:
        raise ValueError(f"{path} must be an integer >= {minimum}")
    return int(value)


def _float(config: Mapping[str, Any], path: str, default: float) -> float:
    value = float(get_path(config, path, default))
    if not math.isfinite(value):
        raise ValueError(f"{path} must be finite")
    return value


def _train_kwargs(
    config: Mapping[str, Any],
    device: torch.device,
    *,
    steps_override: int | None = None,
) -> dict[str, Any]:
    steps = (
        _integer(config, "train.steps", 1_000, minimum=1)
        if steps_override is None
        else int(steps_override)
    )
    if steps < 1:
        raise ValueError("training steps must be positive")
    requested_logs = get_path(config, "train.eval_steps", "log")
    if requested_logs == "log" or requested_logs is None:
        logs = tuple(step for step in log_evaluation_steps(steps) if step > 0)
    elif isinstance(requested_logs, (list, tuple)):
        logs = tuple(int(step) for step in requested_logs)
    else:
        interval = int(requested_logs)
        if interval < 1:
            raise ValueError("train.eval_steps must be 'log', a sequence, or a positive interval")
        logs = tuple(range(interval, steps + 1, interval))
    if steps not in logs:
        logs = (*logs, steps)
    return {
        "steps": steps,
        "batch_size": _integer(config, "train.batch_size", 128, minimum=1),
        "learning_rate": _float(config, "train.learning_rate", 3e-3),
        "weight_decay": _float(config, "train.weight_decay", 0.0),
        "optimizer": str(get_path(config, "train.optimizer", "adamw")),
        "seed": 0,  # Replaced with the run seed below.
        "device": device,
        "deterministic": True,
        "shuffle": True,
        "gradient_clip_norm": get_path(config, "train.grad_clip", 1.0),
        "log_steps": logs,
        "checkpoint_steps": logs,
        "save_checkpoints": bool(get_path(config, "run.save_checkpoints", True)),
        "reset_model": False,
        "reset_optimizer": True,
    }


class _SemanticSampler:
    """Deterministic shuffled epochs over a SemanticBatch.

    A transform receives each selected semantic mini-batch and the optimizer
    step.  H6 uses it to redraw step noise while preserving episode/state IDs.
    """

    def __init__(
        self,
        batch: SemanticBatch,
        transform: Callable[[SemanticBatch, int], SemanticBatch] | None = None,
        *,
        episode_coherent: bool = False,
    ) -> None:
        self.batch = batch
        self.transform = transform
        self.episode_coherent = bool(episode_coherent)
        self._order = torch.empty(0, dtype=torch.long)
        self._cursor = 0
        self._episode_groups: tuple[np.ndarray, ...] = ()
        if self.episode_coherent:
            groups = []
            for episode in np.unique(np.asarray(batch.episode_id)):
                positions = np.flatnonzero(np.asarray(batch.episode_id) == episode)
                positions = positions[np.argsort(np.asarray(batch.step_id)[positions])]
                groups.append(positions.astype(np.int64, copy=False))
            if not groups or len({len(group) for group in groups}) != 1:
                raise ValueError("episode-coherent sampling requires equal non-empty episode lengths")
            self._episode_groups = tuple(groups)

    def sample_batch(
        self,
        batch_size: int,
        generator: torch.Generator,
        step: int,
        **_: Any,
    ) -> SemanticBatch:
        if self.episode_coherent:
            horizon = len(self._episode_groups[0])
            if batch_size % horizon:
                raise ValueError(
                    f"H6 batch_size={batch_size} must be divisible by fixed horizon={horizon}"
                )
            requested_episodes = batch_size // horizon
            episode_pieces: list[np.ndarray] = []
            remaining_episodes = requested_episodes
            while remaining_episodes:
                if self._cursor >= len(self._order):
                    self._order = torch.randperm(len(self._episode_groups), generator=generator)
                    self._cursor = 0
                take = min(remaining_episodes, len(self._order) - self._cursor)
                chosen = self._order[self._cursor : self._cursor + take].tolist()
                episode_pieces.extend(self._episode_groups[int(index)] for index in chosen)
                self._cursor += take
                remaining_episodes -= take
            index = np.concatenate(episode_pieces)
            selected = self.batch.select(index)
            return self.transform(selected, step) if self.transform is not None else selected

        pieces: list[torch.Tensor] = []
        remaining = batch_size
        while remaining:
            if self._cursor >= len(self._order):
                self._order = torch.randperm(len(self.batch), generator=generator)
                self._cursor = 0
            take = min(remaining, len(self._order) - self._cursor)
            pieces.append(self._order[self._cursor : self._cursor + take])
            self._cursor += take
            remaining -= take
        index = torch.cat(pieces).cpu().numpy()
        selected = self.batch.select(index)
        return self.transform(selected, step) if self.transform is not None else selected


def _make_h5_episode_dataset(
    n: int,
    q: float,
    k: int,
    seed: int,
    *,
    context_bits: int,
    nuisance_bits: int,
    nuisance_entropy: float,
    active_nuisance_bits: int | None = None,
    max_k: int,
    state_dim: int,
) -> SemanticBatch:
    """Materialize successful fixed-horizon H5 demonstrations.

    One row is one episode-level context. ``NeutralChoiceSimulator`` supplies
    the authoritative trajectory semantics: active gadgets contribute a
    reward-equivalent branch action followed by a deterministic merge, inactive
    gadgets take a forced branch and merge, and the final action selects the
    intended goal. The factorized SFT loss predicts only active branch actions
    and the final goal. Forced actions add no loss but remain part of the
    demonstration horizon and cost accounting.

    The sampled neutral actions are deliberately *not* model inputs.  Keeping
    ``N_i`` in the observation would let an auxiliary head copy its own label
    and would not test the cost of fitting incidental demonstration details.
    """

    raw = make_h5_dataset(
        n,
        q,
        k,
        seed,
        context_bits=context_bits,
        nuisance_bits=nuisance_bits,
        nuisance_entropy=nuisance_entropy,
        max_k=max_k,
        state_dim=state_dim,
    )
    if active_nuisance_bits is not None and not 0 <= active_nuisance_bits <= nuisance_bits:
        raise ValueError("active_nuisance_bits must lie between zero and nuisance_bits")
    if nuisance_bits:
        requested_choices = np.asarray(raw.nuisance_targets, dtype=np.int64)
        if requested_choices.shape != (n, nuisance_bits):
            raise RuntimeError("H5 nuisance targets do not match the requested gadget count")
    else:
        requested_choices = np.empty((n, 0), dtype=np.int64)
    if active_nuisance_bits is not None and nuisance_bits:
        # H5 defines total nuisance entropy as an integer number of fair branch
        # choices. Remaining gadgets take a forced left branch. Their heads stay
        # in the actor solely to hold architecture/update capacity fixed.
        requested_choices[:, active_nuisance_bits:] = 0

    # Reuse one simulator per intended goal.  Explicit neutral choices make
    # every rollout independent of the simulator's mutable RNG state.
    simulators = {
        -1: NeutralChoiceSimulator(nuisance_bits, Goal.LEFT, seed=seed),
        1: NeutralChoiceSimulator(nuisance_bits, Goal.RIGHT, seed=seed + 1),
    }
    trajectory_cache: dict[tuple[int, tuple[int, ...]], tuple[int, ...]] = {}
    realized_choices = np.empty_like(requested_choices)
    horizon = 2 * nuisance_bits + 1
    for row, intended in enumerate(np.asarray(raw.y, dtype=np.int8)):
        choice_tuple = tuple(int(value) for value in requested_choices[row])
        key = (int(intended), choice_tuple)
        if key not in trajectory_cache:
            trajectory = simulators[int(intended)].rollout(neutral_choices=choice_tuple)
            if (
                not trajectory.success
                or trajectory.total_reward != 1.0
                or trajectory.horizon != horizon
                or len(trajectory.steps) != horizon
            ):
                raise RuntimeError("NeutralChoiceSimulator produced an invalid H5 demonstration")
            trajectory_cache[key] = tuple(trajectory.neutral_choices)
        if nuisance_bits:
            realized_choices[row] = np.asarray(trajectory_cache[key], dtype=np.int64)

    visible_channels = {
        name: values for name, values in raw.channels.items() if not name.startswith("N_")
    }
    active_branches = nuisance_bits if active_nuisance_bits is None else active_nuisance_bits
    realized_entropies = (
        tuple(raw.metadata.get("realized_nuisance_entropy", ()))
        if active_nuisance_bits is None
        else (*([1.0] * active_branches), *([0.0] * (nuisance_bits - active_branches)))
    )
    base_state = (
        np.empty((n, 0), dtype=np.float32)
        if raw.state is None
        else np.asarray(raw.state, dtype=np.float32)
    )
    if base_state.shape[1] != state_dim:
        raise RuntimeError("H5 state features do not match the configured state_dim")
    visitation_state_index = int(state_dim)
    state = np.column_stack(
        (base_state, np.zeros(n, dtype=np.float32))
    ).astype(np.float32, copy=False)
    metadata = {
        **dict(raw.metadata),
        # Successor state IDs must not perturb inactive R padding; the only
        # model-visible transition is the explicit state path-code coordinate.
        "neutralize_padding": True,
        "state_dim": int(state_dim + 1),
        "base_state_dim": int(state_dim),
        "policy_visitation_kernel": "binary_tree_path_code",
        "policy_visitation_state_index": visitation_state_index,
        "policy_visitation_channel": f"state_{visitation_state_index}",
        "policy_visitation_base_value": 0.0,
        "policy_visitation_transition": "next_code=2*code+1+action",
        "realized_nuisance_entropy": realized_entropies,
        "trajectory_semantics": "NeutralChoiceSimulator",
        "episode_abstraction": "factorized_fixed_horizon_successful_demonstration",
        "neutral_gadgets": int(nuisance_bits),
        "trajectory_horizon": int(horizon),
        "variable_branch_actions_per_episode": int(active_branches),
        "stochastic_branch_actions_per_episode": int(active_branches),
        "forced_branch_actions_per_episode": int(nuisance_bits - active_branches),
        "deterministic_merge_actions_per_episode": int(nuisance_bits),
        "goal_actions_per_episode": 1,
        "learned_action_factors_per_episode": int(active_branches + 1),
        "neutral_actions_visible_in_observation": False,
        "terminal_reward_only": True,
    }
    return raw.with_updates(
        channels=visible_channels,
        state=state,
        nuisance_targets=realized_choices if nuisance_bits else None,
        metadata=metadata,
    )


@dataclass
class _NeutralEpisodeSampler:
    """Resample reward-equivalent neutral actions on every trajectory batch."""

    sampler: _SemanticSampler
    positive_probabilities: np.ndarray
    seed: int
    draws: int = 0
    rollouts: int = 0
    last_batch: SemanticBatch | None = None
    _trajectory_cache: dict[tuple[int, tuple[int, ...]], tuple[int, ...]] = field(
        default_factory=dict, init=False, repr=False
    )
    _simulators: dict[int, NeutralChoiceSimulator] = field(
        default_factory=dict, init=False, repr=False
    )

    def __post_init__(self) -> None:
        probabilities = np.asarray(self.positive_probabilities, dtype=np.float64)
        if probabilities.ndim != 1 or np.any((probabilities < 0) | (probabilities > 1)):
            raise ValueError("neutral-action probabilities must be a vector in [0,1]")
        self.positive_probabilities = probabilities
        gadgets = len(probabilities)
        self._simulators = {
            -1: NeutralChoiceSimulator(gadgets, Goal.LEFT, seed=self.seed),
            1: NeutralChoiceSimulator(gadgets, Goal.RIGHT, seed=self.seed + 1),
        }

    @property
    def batch(self) -> SemanticBatch:
        """Underlying clean episode contexts, used to size a separate critic."""

        return self.sampler.batch

    def sample_batch(
        self,
        batch_size: int,
        generator: torch.Generator,
        step: int,
        **_: Any,
    ) -> SemanticBatch:
        selected = self.sampler.sample_batch(batch_size, generator, step)
        gadgets = len(self.positive_probabilities)
        choices = np.zeros((batch_size, gadgets), dtype=np.int64)
        for column, probability in enumerate(self.positive_probabilities):
            positive_count = round(batch_size * float(probability))
            if positive_count:
                chosen = torch.randperm(batch_size, generator=generator)[:positive_count]
                choices[chosen.numpy(), column] = 1

        realized = np.empty_like(choices)
        expected_horizon = 2 * gadgets + 1
        for row, intended in enumerate(np.asarray(selected.y, dtype=np.int8)):
            choice_tuple = tuple(int(value) for value in choices[row])
            key = (int(intended), choice_tuple)
            if key not in self._trajectory_cache:
                trajectory = self._simulators[int(intended)].rollout(
                    neutral_choices=choice_tuple
                )
                if (
                    not trajectory.success
                    or trajectory.total_reward != 1.0
                    or trajectory.horizon != expected_horizon
                ):
                    raise RuntimeError("resampled neutral trajectory violated H5 semantics")
                self._trajectory_cache[key] = tuple(trajectory.neutral_choices)
            if gadgets:
                realized[row] = np.asarray(self._trajectory_cache[key], dtype=np.int64)

        metadata = {
            **dict(selected.metadata),
            "neutral_choice_resampling": "every_optimizer_batch",
            "neutral_choice_draw": int(step),
        }
        result = selected.with_updates(
            nuisance_targets=realized if gadgets else None,
            metadata=metadata,
        )
        self.draws += 1
        self.rollouts += batch_size
        self.last_batch = result
        return result

    def diagnostics(self) -> dict[str, Any]:
        probabilities = self.positive_probabilities
        entropy = np.zeros_like(probabilities, dtype=np.float64)
        interior = (probabilities > 0.0) & (probabilities < 1.0)
        entropy[interior] = -(
            probabilities[interior] * np.log2(probabilities[interior])
            + (1.0 - probabilities[interior])
            * np.log2(1.0 - probabilities[interior])
        )
        stochastic = int(np.count_nonzero(entropy > 1e-12))
        return {
            "resampling": "every_optimizer_batch",
            "draws": int(self.draws),
            "rollouts": int(self.rollouts),
            "unique_validated_trajectory_patterns": len(self._trajectory_cache),
            "positive_probability_per_gadget": probabilities.tolist(),
            "configured_branch_entropy_bits": entropy.tolist(),
            "configured_total_branch_entropy_bits": float(entropy.sum()),
            "stochastic_branch_actions_per_episode": stochastic,
            "forced_branch_actions_per_episode": int(len(probabilities) - stochastic),
            "branch_action_source": "trajectory_demonstration_labels",
        }


@dataclass
class _PolicyOccupancyCollector:
    """Query the canonical oracle on states reached by the current policy.

    Each exogenous H5 context is the root of a deterministic two-successor
    tree.  At each depth the current goal policy chooses the left or right edge,
    which updates only a reserved model-visible state path code.  The semantic goal
    and every P/R/task channel are preserved.  The canonical oracle is queried
    once at every rollout's final reached state, with no additional filtering or
    prioritization.
    """

    sampler: _SemanticSampler
    config: Mapping[str, Any]
    rollout_depth: int = 2
    root_rollouts: int = 0
    transition_rollouts: int = 0
    queried_visited_states: int = 0
    rollout_batches: int = 0
    _base_state_ids: set[int] = field(default_factory=set, init=False, repr=False)
    _visited_state_ids: set[int] = field(default_factory=set, init=False, repr=False)
    _queried_state_ids: set[int] = field(default_factory=set, init=False, repr=False)
    _path_counts: dict[int, int] = field(default_factory=dict, init=False, repr=False)
    _right_actions_by_depth: np.ndarray = field(init=False, repr=False)
    _probability_right_by_depth: np.ndarray = field(init=False, repr=False)
    _nodes_per_root: int = field(init=False, repr=False)
    _state_id_ordinals: dict[int, int] = field(init=False, repr=False)
    _sample_id_ordinals: dict[int, int] = field(init=False, repr=False)
    _state_id_namespace: int = field(init=False, repr=False)
    _sample_id_namespace: int = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if isinstance(self.rollout_depth, bool) or not 1 <= int(self.rollout_depth) <= 16:
            raise ValueError("on-policy rollout depth must be an integer in [1, 16]")
        self.rollout_depth = int(self.rollout_depth)
        self._visitation_state_index(self.sampler.batch)
        self._nodes_per_root = (1 << (self.rollout_depth + 1)) - 2
        (
            self._state_id_ordinals,
            self._state_id_namespace,
        ) = self._id_namespace(np.asarray(self.sampler.batch.state_id), "state_id")
        (
            self._sample_id_ordinals,
            self._sample_id_namespace,
        ) = self._id_namespace(np.asarray(self.sampler.batch.sample_id), "sample_id")
        self._right_actions_by_depth = np.zeros(self.rollout_depth, dtype=np.int64)
        self._probability_right_by_depth = np.zeros(self.rollout_depth, dtype=np.float64)

    @staticmethod
    def _visitation_state_index(batch: SemanticBatch) -> int:
        if batch.state is None or batch.state.shape[1] < 1:
            raise ValueError(
                "policy-induced occupancy requires at least one model-visible state coordinate"
            )
        configured = batch.metadata.get("policy_visitation_state_index")
        if configured is None:
            raise ValueError(
                "policy-induced occupancy requires metadata.policy_visitation_state_index"
            )
        index = int(configured)
        if not 0 <= index < batch.state.shape[1]:
            raise ValueError("policy visitation state index is outside the state feature width")
        return index

    def _id_namespace(
        self, raw_ids: np.ndarray, name: str
    ) -> tuple[dict[int, int], int]:
        ids = np.asarray(raw_ids, dtype=np.int64)
        unique = np.unique(ids)
        if len(unique) != len(ids):
            raise ValueError(f"H5 occupancy requires unique root {name} values")
        namespace = int(unique[-1]) + 1
        maximum = namespace + len(unique) * self._nodes_per_root - 1
        if maximum > np.iinfo(np.int64).max:
            raise ValueError(f"{name} successor namespace would overflow int64")
        return {int(value): rank for rank, value in enumerate(unique)}, namespace

    def _tree_ids(
        self,
        root_ids: np.ndarray,
        depth: int,
        path_codes: np.ndarray,
        *,
        ordinals: Mapping[int, int],
        namespace: int,
        name: str,
    ) -> np.ndarray:
        try:
            root_ordinals = np.fromiter(
                (ordinals[int(value)] for value in np.asarray(root_ids)),
                dtype=np.int64,
                count=len(root_ids),
            )
        except KeyError as exc:  # pragma: no cover - sampler invariant guard
            raise ValueError(f"collector received an unknown root {name}") from exc
        depth_offset = (1 << depth) - 2
        return (
            namespace
            + root_ordinals * self._nodes_per_root
            + depth_offset
            + np.asarray(path_codes, dtype=np.int64)
        ).astype(np.int64, copy=False)

    def sample_batch(
        self,
        model: nn.Module,
        batch_size: int,
        generator: torch.Generator,
        step: int,
    ) -> SemanticBatch:
        roots = self.sampler.sample_batch(batch_size, generator, step)
        visitation_index = self._visitation_state_index(roots)
        if not np.all(np.asarray(roots.state)[:, visitation_index] == 0.0):
            raise ValueError("H5 root visitation path codes must be zero")
        root_state_ids = np.asarray(roots.state_id, dtype=np.int64)
        root_sample_ids = np.asarray(roots.sample_id, dtype=np.int64)
        step_ids = np.asarray(roots.step_id, dtype=np.int64)
        if np.any(step_ids > np.iinfo(np.int64).max - self.rollout_depth):
            raise ValueError("step_id is too large to record the occupancy rollout")

        current = roots
        path_codes = np.zeros(batch_size, dtype=np.int64)
        action_path = np.empty((batch_size, self.rollout_depth), dtype=np.int8)
        probability_path = np.empty((batch_size, self.rollout_depth), dtype=np.float32)
        for depth_index in range(self.rollout_depth):
            depth = depth_index + 1
            logits = predict_logits(model, current, self.config)
            probabilities = torch.sigmoid(
                torch.as_tensor(logits[:, 1] - logits[:, 0], dtype=torch.float64)
            )
            uniforms = torch.rand(batch_size, generator=generator, dtype=torch.float64)
            actions = (uniforms < probabilities).numpy().astype(np.int8)
            action_path[:, depth_index] = actions
            probability_path[:, depth_index] = probabilities.numpy().astype(np.float32)
            path_codes = 2 * path_codes + actions.astype(np.int64)

            next_state = np.asarray(current.state, dtype=np.float32).copy()
            # Standard binary-tree heap coding: left=2*c+1, right=2*c+2.
            # At a fixed depth this is a one-to-one model-visible code for the
            # complete action path, while every non-visitation state coordinate
            # is byte-identical to the sampled root.
            next_state[:, visitation_index] = (
                2.0 * next_state[:, visitation_index] + 1.0 + actions
            )
            successor_state_ids = self._tree_ids(
                root_state_ids,
                depth,
                path_codes,
                ordinals=self._state_id_ordinals,
                namespace=self._state_id_namespace,
                name="state_id",
            )
            successor_sample_ids = self._tree_ids(
                root_sample_ids,
                depth,
                path_codes,
                ordinals=self._sample_id_ordinals,
                namespace=self._sample_id_namespace,
                name="sample_id",
            )
            current = current.with_updates(
                target=np.asarray(roots.y, dtype=np.int8),
                state=next_state,
                sample_id=successor_sample_ids,
                state_id=successor_state_ids,
                step_id=step_ids + depth,
                nuisance_targets=None,
            )
            self._right_actions_by_depth[depth_index] += int(actions.sum())
            self._probability_right_by_depth[depth_index] += float(
                probabilities.sum().item()
            )
            self._visited_state_ids.update(int(value) for value in successor_state_ids)

        latents = dict(roots.latents)
        latents.update(
            {
                "occupancy_base_sample_id": root_sample_ids,
                "occupancy_base_state_id": root_state_ids,
                "occupancy_policy_action_path": action_path,
                "occupancy_probability_right_path": probability_path,
                "occupancy_path_code": path_codes,
                "occupancy_final_successor_side": action_path[:, -1],
            }
        )
        metadata = {
            **dict(roots.metadata),
            "policy_occupancy_kernel": "deterministic_depth_d_binary_path_code_tree",
            "policy_occupancy_rollout_depth": int(self.rollout_depth),
            "policy_occupancy_query_rule": "query_every_final_visited_state",
            "policy_occupancy_oracle": "canonical_semantic_goal",
            "policy_occupancy_successor_id_namespace": "above_all_source_root_ids",
        }
        successors = current.with_updates(
            target=np.asarray(roots.y, dtype=np.int8),
            nuisance_targets=None,
            latents=latents,
            metadata=metadata,
        )

        self.root_rollouts += int(batch_size)
        self.transition_rollouts += int(batch_size * self.rollout_depth)
        self.queried_visited_states += int(batch_size)
        self.rollout_batches += 1
        self._base_state_ids.update(int(value) for value in root_state_ids)
        self._queried_state_ids.update(int(value) for value in np.asarray(successors.state_id))
        for path_code, count in zip(*np.unique(path_codes, return_counts=True), strict=True):
            code = int(path_code)
            self._path_counts[code] = self._path_counts.get(code, 0) + int(count)
        return successors

    def diagnostics(self) -> dict[str, Any]:
        roots = self.root_rollouts
        transitions = self.transition_rollouts
        right_by_depth = (
            self._right_actions_by_depth.astype(np.float64) / roots
            if roots
            else np.full(self.rollout_depth, np.nan)
        )
        mean_probability_by_depth = (
            self._probability_right_by_depth / roots
            if roots
            else np.full(self.rollout_depth, np.nan)
        )
        right_rate = float(self._right_actions_by_depth.sum() / transitions) if transitions else None
        final_right_rate = float(right_by_depth[-1]) if roots else None
        return {
            "policy_dependent": True,
            "policy_induced_occupancy": True,
            "transition_kernel": "deterministic_depth_d_binary_path_code_tree",
            "rollout_depth": int(self.rollout_depth),
            "selection_strategy": "query_every_final_policy_visited_state",
            "query_priority": "none",
            "root_rollouts": int(roots),
            "transition_rollouts": int(transitions),
            "rollout_batches": int(self.rollout_batches),
            "queried_visited_states": int(self.queried_visited_states),
            "unique_base_state_ids": len(self._base_state_ids),
            "unique_visited_state_ids": len(self._visited_state_ids),
            "unique_queried_state_ids": len(self._queried_state_ids),
            "policy_action_rate": {
                "left": None if right_rate is None else 1.0 - right_rate,
                "right": right_rate,
            },
            "policy_action_right_rate_by_depth": right_by_depth.tolist() if roots else [],
            "mean_policy_probability_right_by_depth": (
                mean_probability_by_depth.tolist() if roots else []
            ),
            "final_successor_occupancy_rate": {
                "left": None if final_right_rate is None else 1.0 - final_right_rate,
                "right": final_right_rate,
            },
            "path_occupancy_rate": {
                format(code, f"0{self.rollout_depth}b"): count / roots
                for code, count in sorted(self._path_counts.items())
            }
            if roots
            else {},
            "visitation_channel": str(
                self.sampler.batch.metadata["policy_visitation_channel"]
            ),
            "visitation_state_index": int(
                self._visitation_state_index(self.sampler.batch)
            ),
            "successor_ids_disjoint_from_source_roots": True,
            "nodes_per_root_id_namespace": int(self._nodes_per_root),
            "label_mode": "single_canonical_goal_action",
            "oracle_label_source": "semantic_y",
        }


# Compatibility for callers that imported the former private collector name.
# The alias has exactly the genuine occupancy behavior above.
_PolicyConditionedCollector = _PolicyOccupancyCollector


@dataclass
class _AllContextCollector:
    """Unfiltered contextual collection for protocols without an H5 fork kernel."""

    sampler: _SemanticSampler
    queried_contexts: int = 0
    collection_batches: int = 0

    def sample_batch(
        self,
        model: nn.Module,
        batch_size: int,
        generator: torch.Generator,
        step: int,
    ) -> SemanticBatch:
        del model
        selected = self.sampler.sample_batch(batch_size, generator, step)
        self.queried_contexts += int(batch_size)
        self.collection_batches += 1
        return selected

    def diagnostics(self) -> dict[str, Any]:
        return {
            "policy_dependent": False,
            "selection_strategy": "query_every_exogenous_context",
            "query_priority": "none",
            "queried_contexts": int(self.queried_contexts),
            "collection_batches": int(self.collection_batches),
        }


def _terminal_success_reward(
    raw_batch: SemanticBatch,
    actions: torch.Tensor,
    **_: Any,
) -> torch.Tensor:
    """NeutralChoiceSimulator terminal outcome, vectorized over episodes.

    Preceding branch/merge transitions are reward zero and reward-equivalent.
    Only the final sampled goal action can yield one.  No nuisance target is
    consulted, so RL cannot receive an auxiliary imitation signal.
    """

    intended = torch.as_tensor(
        (np.asarray(raw_batch.y) > 0).astype(np.int64),
        dtype=torch.long,
        device=actions.device,
    )
    return (actions.long() == intended).to(torch.float32)


@dataclass
class _NeutralTrajectoryReward:
    """Execute actor-sampled neutral trajectories and return terminal success.

    The branch actions come from the actor's nuisance heads, never from the
    demonstration's nuisance targets. Every sampled branch is passed through
    ``NeutralChoiceSimulator`` together with the sampled final goal. Thus the
    reward hook validates the full fixed horizon while exposing only its single
    terminal outcome to policy-gradient training.
    """

    num_gadgets: int
    seed: int
    stochastic_gadgets: int | None = None
    draws: int = 0
    rollouts: int = 0
    successful_rollouts: int = 0
    last_branch_actions: np.ndarray | None = field(default=None, init=False, repr=False)
    branch_right_counts: np.ndarray = field(init=False, repr=False)
    _trajectory_cache: dict[tuple[int, tuple[int, ...], int], float] = field(
        default_factory=dict, init=False, repr=False
    )
    _simulators: dict[int, NeutralChoiceSimulator] = field(
        default_factory=dict, init=False, repr=False
    )

    def __post_init__(self) -> None:
        if self.num_gadgets < 0:
            raise ValueError("num_gadgets cannot be negative")
        self.num_gadgets = int(self.num_gadgets)
        if self.stochastic_gadgets is None:
            self.stochastic_gadgets = self.num_gadgets
        if not 0 <= self.stochastic_gadgets <= self.num_gadgets:
            raise ValueError("stochastic_gadgets must lie between zero and num_gadgets")
        self.stochastic_gadgets = int(self.stochastic_gadgets)
        self.branch_right_counts = np.zeros(self.stochastic_gadgets, dtype=np.int64)
        self._simulators = {
            -1: NeutralChoiceSimulator(self.num_gadgets, Goal.LEFT, seed=self.seed),
            1: NeutralChoiceSimulator(self.num_gadgets, Goal.RIGHT, seed=self.seed + 1),
        }

    def __call__(
        self,
        raw_batch: SemanticBatch,
        actions: torch.Tensor,
        branch_actions: torch.Tensor | None,
        **_: Any,
    ) -> torch.Tensor:
        branches = (
            np.empty((len(actions), 0), dtype=np.int64)
            if branch_actions is None
            else branch_actions.detach().cpu().numpy().astype(np.int64, copy=False)
        )
        if branches.shape != (len(actions), self.stochastic_gadgets):
            raise ValueError(
                "actor-sampled branch actions must have shape "
                f"[{len(actions)}, {self.stochastic_gadgets}], got {branches.shape}"
            )
        if np.any((branches != 0) & (branches != 1)):
            raise ValueError("actor-sampled branch actions must be binary")
        selected = actions.detach().cpu().numpy().astype(np.int64, copy=False)
        intended = np.asarray(raw_batch.y, dtype=np.int8)
        if len(intended) != len(selected):
            raise ValueError("terminal goal actions and episode contexts have different sizes")

        rewards = np.empty(len(selected), dtype=np.float32)
        expected_horizon = 2 * self.num_gadgets + 1
        for row, intended_goal in enumerate(intended):
            choice_tuple = (
                *(int(value) for value in branches[row]),
                *([0] * (self.num_gadgets - self.stochastic_gadgets)),
            )
            selected_goal = -1 if int(selected[row]) == 0 else 1
            key = (int(intended_goal), choice_tuple, selected_goal)
            if key not in self._trajectory_cache:
                trajectory = self._simulators[int(intended_goal)].rollout(
                    neutral_choices=choice_tuple,
                    selected_goal=selected_goal,
                )
                reward_sequence = tuple(step.reward for step in trajectory.steps)
                if (
                    trajectory.horizon != expected_horizon
                    or len(trajectory.steps) != expected_horizon
                    or tuple(trajectory.neutral_choices) != choice_tuple
                    or any(reward != 0.0 for reward in reward_sequence[:-1])
                    or reward_sequence[-1] != trajectory.total_reward
                ):
                    raise RuntimeError("actor-sampled neutral trajectory violated H5 semantics")
                self._trajectory_cache[key] = float(trajectory.total_reward)
            rewards[row] = self._trajectory_cache[key]

        self.draws += 1
        self.rollouts += len(selected)
        self.successful_rollouts += int(rewards.sum())
        self.last_branch_actions = branches.copy()
        if self.stochastic_gadgets:
            self.branch_right_counts += branches.sum(axis=0)
        return torch.as_tensor(rewards, device=actions.device)

    def diagnostics(self) -> dict[str, Any]:
        denominator = max(1, self.rollouts)
        probabilities = self.branch_right_counts / denominator
        entropy = np.zeros_like(probabilities, dtype=np.float64)
        interior = (probabilities > 0.0) & (probabilities < 1.0)
        entropy[interior] = -(
            probabilities[interior] * np.log2(probabilities[interior])
            + (1.0 - probabilities[interior])
            * np.log2(1.0 - probabilities[interior])
        )
        return {
            "resampling": "actor_policy_every_optimizer_batch",
            "draws": int(self.draws),
            "rollouts": int(self.rollouts),
            "unique_validated_trajectory_patterns": len(self._trajectory_cache),
            "branch_right_rate": probabilities.tolist(),
            "realized_branch_entropy_bits": entropy.tolist(),
            "realized_total_branch_entropy_bits": float(entropy.sum()),
            "terminal_success_rate": (
                self.successful_rollouts / self.rollouts if self.rollouts else None
            ),
            "branch_action_source": "actor_nuisance_heads",
            "branch_supervision": False,
            "stochastic_branch_actions_per_episode": self.stochastic_gadgets,
            "forced_branch_actions_per_episode": self.num_gadgets - self.stochastic_gadgets,
        }


def _evaluator(batch: SemanticBatch, config: Mapping[str, Any]) -> Callable[..., dict[str, float]]:
    def evaluate(model: nn.Module, **_: Any) -> dict[str, float]:
        return evaluate_batch(model, batch, config)

    return evaluate


def _critic(
    batch: SemanticBatch,
    config: Mapping[str, Any],
    hypothesis: str,
    device: torch.device,
) -> nn.Module:
    input_dim = int(feature_array(batch, config).shape[1])
    # The critic architecture is fixed within an experiment and independent of U.
    width = _integer(config, f"{hypothesis}.critic_width", 256, minimum=1)
    depth = _integer(config, f"{hypothesis}.critic_depth", 3, minimum=0)
    return ValueMLP(
        input_dim=input_dim,
        width=width,
        depth=depth,
        activation=str(get_path(config, f"{hypothesis}.critic_activation", "gelu")),
        residual=bool(get_path(config, f"{hypothesis}.critic_residual", True)),
    ).to(device)


def _run_algorithm(
    *,
    algorithm: str,
    model: nn.Module,
    source: Any,
    evaluation: SemanticBatch,
    config: Mapping[str, Any],
    seed: int,
    hypothesis: str,
    nuisance_weight: float,
    active_nuisance_bits: int | None = None,
    reward_fn: Callable[..., Any] | None = None,
    factorized_rl: bool = False,
    training_steps: int | None = None,
) -> tuple[TrainingResult, float]:
    device = resolve_device(str(get_path(config, "run.device", "auto")))
    common = _train_kwargs(config, device, steps_override=training_steps)
    common["seed"] = int(seed)
    evaluator = _evaluator(evaluation, config)
    started = time.perf_counter()

    if algorithm in {"clean_sft", "trajectory_sft"}:
        use_auxiliary_trajectory_loss = algorithm == "trajectory_sft" and nuisance_weight > 0
        nuisance_weights = (
            {
                f"nuisance_{index}": float(
                    active_nuisance_bits is None or index < active_nuisance_bits
                )
                for index in range(len(getattr(source, "positive_probabilities", ())))
            }
            if use_auxiliary_trajectory_loss and active_nuisance_bits is not None
            else None
        )
        sft_config = SFTConfig(
            **common,
            auxiliary_weight=nuisance_weight if use_auxiliary_trajectory_loss else 0.0,
            nuisance_weights=nuisance_weights,
        )
        if use_auxiliary_trajectory_loss:
            result = train_nuisance_sft(model, source, sft_config, evaluator=evaluator)
        else:
            result = train_clean_sft(model, source, sft_config, evaluator=evaluator)
    elif algorithm == "on_policy_imitation":
        sampler = source if isinstance(source, _SemanticSampler) else _SemanticSampler(source)
        collector: _PolicyOccupancyCollector | _AllContextCollector = (
            _PolicyOccupancyCollector(
                sampler=sampler,
                config=config,
                rollout_depth=_integer(
                    config, f"{hypothesis}.on_policy_rollout_depth", 2, minimum=1
                ),
            )
            if hypothesis == "h5"
            else _AllContextCollector(sampler=sampler)
        )

        imitation = OnPolicyImitationConfig(
            **common,
            collection_batch_size=int(
                get_path(config, f"{hypothesis}.collection_batch_size", common["batch_size"])
            ),
            updates_per_collection=_integer(
                config, f"{hypothesis}.updates_per_collection", 1, minimum=1
            ),
            replay_capacity=get_path(config, f"{hypothesis}.replay_capacity", None),
        )
        result = train_on_policy_imitation(
            model, collector.sample_batch, imitation, evaluator=evaluator
        )
        # TrainingResult intentionally stays generic; protocol-specific
        # collection accounting is attached for the H5/H6 summaries.
        result.policy_collection = collector.diagnostics()
    else:
        bandit = BanditConfig(
            **common,
            algorithm=str(get_path(config, f"{hypothesis}.rl_estimator", "actor_critic")),
            entropy_coefficient=_float(
                config, f"{hypothesis}.entropy_coefficient", 0.0
            ),
            critic_learning_rate=_float(
                config,
                f"{hypothesis}.critic_learning_rate",
                _float(config, "train.learning_rate", 3e-3),
            ),
            critic_weight_decay=_float(
                config, f"{hypothesis}.critic_weight_decay", 0.0
            ),
            critic_width=_integer(config, f"{hypothesis}.critic_width", 256, minimum=1),
            critic_depth=_integer(config, f"{hypothesis}.critic_depth", 3, minimum=0),
        )
        fixed_critic = (
            _critic(
                source.batch if hasattr(source, "batch") else source,
                config,
                hypothesis,
                device,
            )
            if bandit.algorithm == "actor_critic"
            else None
        )
        result = train_contextual_bandit(
            model,
            source,
            bandit,
            reward_fn=reward_fn,
            critic=fixed_critic,
            evaluator=evaluator,
            factorized_actions=factorized_rl,
            factorized_action_heads=active_nuisance_bits if factorized_rl else None,
        )
    return result, time.perf_counter() - started


def _checkpoint_payload(checkpoint: TrainingCheckpoint) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "step": checkpoint.step,
        "model": checkpoint.model_state,
        "optimizer": checkpoint.optimizer_state,
    }
    if checkpoint.critic_state is not None:
        payload["critic"] = checkpoint.critic_state
    if checkpoint.critic_optimizer_state is not None:
        payload["critic_optimizer"] = checkpoint.critic_optimizer_state
    return payload


def _checkpoints(result: TrainingResult) -> dict[str, dict[str, Any]]:
    return {
        f"step_{step:08d}": _checkpoint_payload(checkpoint)
        for step, checkpoint in sorted(result.checkpoints.items())
    }


def _history_metrics(
    result: TrainingResult,
    *,
    hypothesis: str,
    condition: str,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    evaluation_names = {
        "rho_y",
        "rho_p",
        "delta_rho",
        "intended_probability",
        "confidence",
        "invalid_rate",
        "target_accuracy",
    }
    for item in result.history:
        train_values: dict[str, float | int] = {
            "loss": item.loss,
            "primary_loss": item.primary_loss,
            "auxiliary_loss": item.auxiliary_loss,
        }
        evaluation_values: dict[str, float | int] = {}
        for name, value in item.metrics.items():
            (evaluation_values if name in evaluation_names else train_values)[name] = value
        records.extend(
            make_metric_records(
                train_values,
                hypothesis=hypothesis,
                split="train",
                global_step=item.step,
                examples_seen=item.samples_seen,
                condition=condition,
            )
        )
        if evaluation_values:
            records.extend(
                make_metric_records(
                    evaluation_values,
                    hypothesis=hypothesis,
                    split="conflict_eval",
                    global_step=item.step,
                    examples_seen=item.samples_seen,
                    condition=condition,
                )
            )
    return records


def _final_records(
    values: Mapping[str, float | int],
    *,
    hypothesis: str,
    split: str,
    condition: str,
    result: TrainingResult,
) -> list[dict[str, Any]]:
    return make_metric_records(
        values,
        hypothesis=hypothesis,
        split=split,
        global_step=result.optimizer_steps,
        stage="final",
        examples_seen=result.samples_seen,
        condition=condition,
    )


@torch.no_grad()
def _nuisance_diagnostics(
    model: nn.Module,
    batch: SemanticBatch,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    targets = batch.nuisance_targets
    if targets is None or np.asarray(targets).shape[1] == 0:
        return {"nuisance_heads": 0, "mean_accuracy": None, "head_accuracy": {}}
    device = next(model.parameters()).device
    features = torch.as_tensor(feature_array(batch, config), dtype=torch.float32, device=device)
    was_training = model.training
    model.eval()
    try:
        output = model(features, return_aux=True)
    finally:
        model.train(was_training)
    nuisance_logits = getattr(output, "nuisance_logits", None)
    if nuisance_logits is None and isinstance(output, Mapping):
        nuisance_logits = output.get("nuisance_logits")
    if not isinstance(nuisance_logits, Mapping):
        return {
            "nuisance_heads": int(np.asarray(targets).shape[1]),
            "mean_accuracy": None,
            "head_accuracy": {},
        }
    target_array = np.asarray(targets)
    accuracies: dict[str, float] = {}
    for index, (name, logits) in enumerate(nuisance_logits.items()):
        if index >= target_array.shape[1]:
            break
        if logits.ndim == 1 or logits.shape[-1] == 1:
            prediction = (logits.reshape(-1) >= 0).cpu().numpy()
        else:
            prediction = logits.argmax(dim=-1).cpu().numpy()
        accuracies[str(name)] = float(np.mean(prediction == target_array[:, index]))
    mean = float(np.mean(list(accuracies.values()))) if accuracies else None
    return {
        "nuisance_heads": int(target_array.shape[1]),
        "mean_accuracy": mean,
        "head_accuracy": accuracies,
    }


def _costs(
    algorithm: str,
    result: TrainingResult,
    wall_seconds: float,
    *,
    trajectory_horizon: int = 1,
    episode_level: bool = False,
    stochastic_branch_actions: int = 0,
    forced_branch_actions: int = 0,
) -> dict[str, Any]:
    """Return action-level costs under the documented H5 abstraction.

    Clean SFT consumes one canonical action pair.  Trajectory SFT consumes all
    actions in a successful neutral-choice trajectory, including deterministic
    merges.  On-policy imitation takes one fork transition and queries one clean
    final-action label at every reached successor.  RL traverses the full
    neutral horizon and observes only one terminal outcome per episode.
    """

    if trajectory_horizon < 1 or trajectory_horizon % 2 != 1:
        raise ValueError("trajectory_horizon must be a positive odd integer")
    if not episode_level:
        supervised = algorithm in {"clean_sft", "trajectory_sft", "on_policy_imitation"}
        interactive = algorithm in {"on_policy_imitation", "rl"}
        return {
            "environment_interactions": result.samples_seen if interactive else 0,
            "labeled_actions": result.samples_seen if supervised else 0,
            "optimizer_steps": result.optimizer_steps,
            "wall_seconds": float(wall_seconds),
            "trainable_scalars": result.actor_trainable_parameters,
            "critic_parameters_excluded": result.critic_parameter_count,
        }
    policy_collection = getattr(result, "policy_collection", {})
    if algorithm == "clean_sft":
        environment_interactions = result.samples_seen
        labeled_actions = result.samples_seen
        interactions_per_sample = 1
        labels_per_sample = 1
        terminal_outcomes = 0
    elif algorithm == "trajectory_sft":
        environment_interactions = result.samples_seen * trajectory_horizon
        labeled_actions = result.samples_seen * trajectory_horizon
        interactions_per_sample = trajectory_horizon
        labels_per_sample = trajectory_horizon
        terminal_outcomes = 0
    elif algorithm == "on_policy_imitation":
        environment_interactions = int(
            policy_collection.get("transition_rollouts", result.samples_seen)
        )
        labeled_actions = int(
            policy_collection.get("queried_visited_states", result.samples_seen)
        )
        interactions_per_sample = float(environment_interactions / max(1, result.samples_seen))
        labels_per_sample = float(labeled_actions / max(1, result.samples_seen))
        terminal_outcomes = 0
    else:
        environment_interactions = result.samples_seen * trajectory_horizon
        labeled_actions = 0
        interactions_per_sample = trajectory_horizon
        labels_per_sample = 0
        terminal_outcomes = result.samples_seen
    return {
        "accounting_basis": "actual_transition_rollouts_and_oracle_queries",
        "environment_interactions": int(environment_interactions),
        "labeled_actions": int(labeled_actions),
        "episode_or_example_presentations": int(result.samples_seen),
        "simulator_trajectory_horizon": int(trajectory_horizon),
        "stochastic_branch_actions_per_presentation": int(stochastic_branch_actions),
        "forced_branch_actions_per_presentation": int(forced_branch_actions),
        "learned_action_factors_per_presentation": int(stochastic_branch_actions + 1),
        "environment_interactions_per_presentation": interactions_per_sample,
        "labeled_actions_per_presentation": labels_per_sample,
        "terminal_outcomes": int(terminal_outcomes),
        "optimizer_steps": result.optimizer_steps,
        "wall_seconds": float(wall_seconds),
        "trainable_scalars": result.actor_trainable_parameters,
        "critic_parameters_excluded": result.critic_parameter_count,
    }


def _annotate_h5_history_costs(
    result: TrainingResult,
    algorithm: str,
    trajectory_horizon: int,
    on_policy_rollout_depth: int,
) -> None:
    """Replace trainer-generic counts with H5 action-level cumulative costs."""

    for record in result.history:
        presentations = int(record.samples_seen)
        if algorithm == "clean_sft":
            interactions, labels, outcomes = presentations, presentations, 0
        elif algorithm == "trajectory_sft":
            interactions = labels = presentations * trajectory_horizon
            outcomes = 0
        elif algorithm == "on_policy_imitation":
            interactions = presentations * on_policy_rollout_depth
            labels = presentations
            outcomes = 0
        else:
            interactions = presentations * trajectory_horizon
            labels, outcomes = 0, presentations
        record.metrics.update(
            {
                "environment_interactions": float(interactions),
                "labeled_actions": float(labels),
                "terminal_outcomes": float(outcomes),
            }
        )


def run_h5(config: Mapping[str, Any], seed: int) -> ProtocolResult:
    """Run one H5 algorithm x nuisance entropy x update-budget cell."""

    if str(get_path(config, "experiment.mode", "primary")) == "entropy_timing":
        # Imported lazily to keep the explicitly post-hoc E14 runner separate
        # without creating a module-import cycle through the shared H5 helpers.
        from .protocols_exploration import run_entropy_timing

        return run_entropy_timing(config, seed)

    algorithm = _algorithm(config, "h5")
    on_policy_label_mode = str(
        get_path(config, "h5.on_policy_label_mode", "clean_action")
    ).lower().replace("-", "_")
    if on_policy_label_mode in {"clean", "minimal"}:
        on_policy_label_mode = "clean_action"
    if algorithm == "on_policy_imitation" and on_policy_label_mode != "clean_action":
        raise ValueError(
            "H5 on-policy imitation currently requires h5.on_policy_label_mode=clean_action; "
            "nuisance-rich sequence prediction is the trajectory_sft arm"
        )
    n_train = _integer(config, "data.n_train", 4_096, minimum=2)
    n_validation = _integer(config, "data.n_validation", 2_048, minimum=2)
    n_eval = _integer(config, "data.n_eval", 4_096, minimum=2)
    q = _float(config, "data.q", 0.9)
    k = _integer(config, "data.k", 3, minimum=1)
    max_k = _integer(config, "data.max_k", k, minimum=k)
    state_dim = _integer(config, "data.state_dim", 0, minimum=0)
    on_policy_rollout_depth = _integer(
        config, "h5.on_policy_rollout_depth", 2, minimum=1
    )
    if on_policy_rollout_depth > 16:
        raise ValueError("h5.on_policy_rollout_depth must be at most 16")
    nuisance_bits = _integer(config, "h5.nuisance_bits", 8, minimum=0)
    total_entropy = _integer(config, "h5.nuisance_entropy", 0, minimum=0)
    nuisance_weight = _float(config, "h5.nuisance_weight", 1.0)
    if nuisance_weight < 0:
        raise ValueError("h5.nuisance_weight must be non-negative")
    if total_entropy > nuisance_bits:
        raise ValueError(
            "h5.nuisance_entropy counts active fair branch gadgets and cannot exceed "
            "h5.nuisance_bits"
        )
    active_nuisance_bits = total_entropy
    # _auxiliary_loss averages across heads.  Multiplying the configured
    # per-action coefficient by the gadget count recovers the token-level
    # trajectory NLL: one unit for the final goal and one for every variable
    # neutral branch (deterministic merges contribute exactly zero).
    trajectory_auxiliary_weight = nuisance_weight * active_nuisance_bits

    context_bits = _integer(config, "h5.context_bits", 1, minimum=0)
    train = _make_h5_episode_dataset(
        n_train,
        q,
        k,
        seed,
        context_bits=context_bits,
        nuisance_bits=nuisance_bits,
        nuisance_entropy=1.0 if active_nuisance_bits else 0.0,
        active_nuisance_bits=active_nuisance_bits,
        max_k=max_k,
        state_dim=state_dim,
    )
    iid = _make_h5_episode_dataset(
        n_validation,
        q,
        k,
        seed + 5_003,
        context_bits=context_bits,
        nuisance_bits=nuisance_bits,
        nuisance_entropy=1.0 if active_nuisance_bits else 0.0,
        active_nuisance_bits=active_nuisance_bits,
        max_k=max_k,
        state_dim=state_dim,
    )
    conflict = _make_h5_episode_dataset(
        n_eval,
        0.0,
        k,
        seed + 10_003,
        context_bits=context_bits,
        nuisance_bits=nuisance_bits,
        nuisance_entropy=1.0 if active_nuisance_bits else 0.0,
        active_nuisance_bits=active_nuisance_bits,
        max_k=max_k,
        state_dim=state_dim,
    )
    # Every condition gets the auxiliary heads, even when unused.  Therefore a
    # full actor and every exact-U actor have identical update capacity across
    # the algorithm comparison.
    model, model_report = build_model(
        train, config, seed, nuisance_heads=nuisance_bits
    )
    positive_probabilities = (
        np.mean(np.asarray(train.nuisance_targets, dtype=np.float64), axis=0)
        if nuisance_bits
        else np.empty(0, dtype=np.float64)
    )
    neutral_episode_sampler = _NeutralEpisodeSampler(
        _SemanticSampler(train),
        positive_probabilities=positive_probabilities,
        seed=seed + 50_021,
    )
    rl_reward = _NeutralTrajectoryReward(
        nuisance_bits,
        seed=seed + 50_021,
        stochastic_gadgets=active_nuisance_bits,
    )
    training_source: Any = (
        neutral_episode_sampler
        if algorithm == "trajectory_sft"
        else _SemanticSampler(train)
        if algorithm == "rl"
        else train
    )
    result, wall_seconds = _run_algorithm(
        algorithm=algorithm,
        model=model,
        # Trajectory SFT resamples demonstration branches as labels. RL instead
        # samples every branch from its own nuisance heads and receives only the
        # simulator's terminal outcome.
        source=training_source,
        evaluation=conflict,
        config=config,
        seed=seed,
        hypothesis="h5",
        nuisance_weight=trajectory_auxiliary_weight,
        active_nuisance_bits=active_nuisance_bits,
        reward_fn=rl_reward if algorithm == "rl" else None,
        factorized_rl=algorithm == "rl",
    )
    if (
        algorithm == "trajectory_sft"
        and neutral_episode_sampler.rollouts != result.samples_seen
    ):
        raise RuntimeError("H5 episode rollout accounting diverged from optimizer samples")
    if algorithm == "rl" and rl_reward.rollouts != result.samples_seen:
        raise RuntimeError("H5 factorized RL rollout accounting diverged from optimizer samples")
    policy_collection = getattr(result, "policy_collection", None)
    if algorithm == "on_policy_imitation":
        if not isinstance(policy_collection, Mapping):
            raise RuntimeError("H5 on-policy imitation did not record occupancy collection")
        if (
            int(policy_collection.get("root_rollouts", -1)) != result.samples_seen
            or int(policy_collection.get("transition_rollouts", -1))
            != on_policy_rollout_depth * result.samples_seen
            or int(policy_collection.get("queried_visited_states", -1))
            != result.samples_seen
        ):
            raise RuntimeError(
                "H5 occupancy roots, depth-scaled transitions, canonical oracle queries, "
                "and collected replay rows must agree"
            )

    final_train = evaluate_batch(model, train, config)
    final_iid = evaluate_batch(model, iid, config)
    final_conflict = evaluate_batch(model, conflict, config)
    interventions = evaluate_standard_interventions(model, conflict, config)
    auxiliary = _nuisance_diagnostics(model, train, config)
    trajectory_horizon = int(train.metadata["trajectory_horizon"])
    costs = _costs(
        algorithm,
        result,
        wall_seconds,
        trajectory_horizon=trajectory_horizon,
        episode_level=True,
        stochastic_branch_actions=(
            active_nuisance_bits if algorithm in {"trajectory_sft", "rl"} else 0
        ),
        forced_branch_actions=(
            nuisance_bits - active_nuisance_bits
            if algorithm in {"trajectory_sft", "rl"}
            else 0
        ),
    )
    _annotate_h5_history_costs(
        result,
        algorithm,
        trajectory_horizon,
        on_policy_rollout_depth,
    )
    realized_entropies = tuple(train.metadata.get("realized_nuisance_entropy", ()))
    summary = {
        "hypothesis": "h5",
        "algorithm": algorithm,
        "seed": int(seed),
        "model": model_report,
        "nuisance": {
            "bits": nuisance_bits,
            "requested_total_entropy": total_entropy,
            "entropy_parameterization": "integer_active_fair_branch_gadgets",
            "active_fair_branch_gadgets": active_nuisance_bits,
            "forced_branch_gadgets": nuisance_bits - active_nuisance_bits,
            "requested_entropy_per_bit": (
                total_entropy / nuisance_bits if nuisance_bits else 0.0
            ),
            "realized_entropy_per_bit": list(realized_entropies),
            "realized_total_entropy": float(sum(realized_entropies)),
            "configured_per_action_auxiliary_weight": nuisance_weight,
            "applied_auxiliary_weight": (
                trajectory_auxiliary_weight
                if algorithm == "trajectory_sft" and nuisance_bits
                else 0.0
            ),
            "branch_actions_policy_sampled": algorithm == "rl" and active_nuisance_bits > 0,
            "branch_labels_used_for_training": (
                algorithm == "trajectory_sft" and active_nuisance_bits > 0
            ),
            "loss_reduction": (
                "goal_nll + per_action_weight * sum(active_branch_nll)"
                if algorithm == "trajectory_sft"
                else "terminal_advantage * sum(sampled_active_branch_and_goal_log_probability)"
                if algorithm == "rl"
                else "goal_action_only"
            ),
            **auxiliary,
        },
        "demonstration": {
            key: train.metadata[key]
            for key in (
                "trajectory_semantics",
                "episode_abstraction",
                "neutral_gadgets",
                "trajectory_horizon",
                "variable_branch_actions_per_episode",
                "stochastic_branch_actions_per_episode",
                "forced_branch_actions_per_episode",
                "deterministic_merge_actions_per_episode",
                "goal_actions_per_episode",
                "learned_action_factors_per_episode",
                "neutral_actions_visible_in_observation",
                "terminal_reward_only",
            )
        }
        | {
            "neutral_choice_sampling": (
                neutral_episode_sampler.diagnostics()
                if algorithm == "trajectory_sft"
                else rl_reward.diagnostics()
                if algorithm == "rl"
                else {
                    "resampling": "not_used_by_minimal_label_control",
                    "draws": 0,
                    "rollouts": 0,
                    "positive_probability_per_gadget": positive_probabilities.tolist(),
                }
            )
        },
        "objectives": {
            "clean_sft": "single canonical final-goal action",
            "trajectory_sft": (
                "active neutral branches plus final goal; forced branches and merges have zero loss"
            ),
            "on_policy_imitation": (
                "one canonical final-goal label at each depth-d current-policy final state"
            ),
            "rl": (
                "policy-gradient log-probabilities for actor-sampled neutral branches plus "
                "final goal; simulator terminal success only and no nuisance imitation loss"
            ),
        },
        "on_policy_label_mode": on_policy_label_mode,
        "on_policy_collection": policy_collection,
        "occupancy_control": {
            "kernel": "deterministic_depth_d_binary_path_code_tree",
            "rollout_depth": int(on_policy_rollout_depth),
            "visitation_channel": str(train.metadata["policy_visitation_channel"]),
            "visitation_state_index": int(
                train.metadata["policy_visitation_state_index"]
            ),
            "base_state_dim": int(train.metadata["base_state_dim"]),
            "model_state_dim": int(train.metadata["state_dim"]),
            "fixed_width_across_h5_arms": True,
            "semantic_channels_preserved": True,
            "canonical_oracle_on_successor": True,
        },
        "costs": costs,
        "train": final_train,
        "iid": final_iid,
        "final": final_conflict,
        "interventions": interventions,
    }
    condition = algorithm
    metrics = _history_metrics(result, hypothesis="h5", condition=condition)
    metrics.extend(
        _final_records(
            final_train,
            hypothesis="h5",
            split="train",
            condition=condition,
            result=result,
        )
    )
    metrics.extend(
        _final_records(
            final_iid,
            hypothesis="h5",
            split="iid_eval",
            condition=condition,
            result=result,
        )
    )
    metrics.extend(
        _final_records(
            final_conflict,
            hypothesis="h5",
            split="conflict_eval",
            condition=condition,
            result=result,
        )
    )
    numeric_costs = {
        name: value for name, value in costs.items() if isinstance(value, (int, float))
    }
    metrics.extend(
        _final_records(
            numeric_costs,
            hypothesis="h5",
            split="cost",
            condition=condition,
            result=result,
        )
    )
    return ProtocolResult(
        model=model,
        summary=summary,
        metrics=metrics,
        predictions=prediction_records(model, conflict, config, split="conflict_eval"),
        checkpoints=_checkpoints(result),
        evaluation_batch=conflict,
    )


def _h6_training_plan(config: Mapping[str, Any], n_train: int) -> dict[str, int | str]:
    """Use a fixed number of complete dataset passes in every H6 N cell."""

    epochs = _integer(config, "h6.training_epochs", 8, minimum=1)
    batch_size = _integer(config, "train.batch_size", 128, minimum=1)
    if _algorithm(config, "h6") == "on_policy_imitation":
        collection_batch_size = _integer(
            config, "h6.collection_batch_size", batch_size, minimum=1
        )
        updates_per_collection = _integer(
            config, "h6.updates_per_collection", 1, minimum=1
        )
        if collection_batch_size != batch_size or updates_per_collection != 1:
            raise ValueError(
                "H6 fixed-epoch exposure requires on-policy collection_batch_size "
                "to equal train.batch_size and updates_per_collection=1"
            )
    requested_presentations = int(epochs * n_train)
    if requested_presentations % batch_size:
        raise ValueError(
            "H6 fixed-epoch evidence must be exactly divisible by train.batch_size: "
            f"h6.training_epochs*data.n_train={requested_presentations}, "
            f"batch_size={batch_size}"
        )
    return {
        "rule": "fixed_dataset_epochs",
        "presentation_unit": "contextual_training_rows",
        "requested_epochs": epochs,
        "requested_presentations": requested_presentations,
        "batch_size": batch_size,
        "optimizer_steps": requested_presentations // batch_size,
    }


def _make_h6_context_dataset(
    n: int,
    q: float,
    k: int,
    seed: int,
    *,
    nuisance_bits: int,
    nuisance_entropy: float,
    max_k: int,
    state_dim: int,
) -> SemanticBatch:
    """Create H6 contexts with hidden reward-irrelevant auxiliary labels."""

    raw = make_h5_dataset(
        n,
        q,
        k,
        seed,
        context_bits=1,
        nuisance_bits=nuisance_bits,
        nuisance_entropy=nuisance_entropy,
        max_k=max_k,
        state_dim=state_dim,
    )
    channels = {
        name: np.array(values, copy=True)
        for name, values in raw.channels.items()
        if not name.startswith("N_")
    }
    metadata = {
        **dict(raw.metadata),
        "dataset": "h6_contextual",
        "split": "h6_context",
        "algorithm_abstraction": "fixed_horizon_repeated_contextual_decisions",
        "full_episode_policy": False,
        "nuisance_label_semantics": (
            "reward-irrelevant auxiliary action bits resampled every optimizer presentation"
        ),
        "nuisance_labels_visible_in_observation": False,
    }
    return raw.with_updates(channels=channels, metadata=metadata)


def _make_recurring_batch(
    *,
    n: int,
    q: float,
    k: int,
    max_k: int,
    state_dim: int,
    visits: int,
    nuisance_bits: int,
    nuisance_entropy: float,
    seed: int,
    reward_mode: str,
    fixed_horizon: int,
) -> SemanticBatch:
    horizon = fixed_horizon
    if n % horizon:
        raise ValueError(
            f"data.n_train={n} must be divisible by h6.fixed_horizon={horizon}"
        )
    n_episodes = n // horizon
    if n_episodes % visits:
        raise ValueError(
            f"H6 recurrence requires n_episodes={n_episodes} to be divisible by "
            f"h6.visits_per_state={visits}"
        )
    n_states = n_episodes // visits
    prototype = _make_h6_context_dataset(
        n_states,
        q,
        k,
        seed,
        nuisance_bits=nuisance_bits,
        nuisance_entropy=nuisance_entropy,
        max_k=max_k,
        state_dim=state_dim,
    )
    episode_index = np.repeat(np.arange(n_states, dtype=np.int64), visits)
    np.random.default_rng(seed + 17).shuffle(episode_index)
    index = np.repeat(episode_index, horizon)
    selected = prototype.select(index)
    episode_ids = np.repeat(np.arange(n_episodes, dtype=np.int64), horizon)
    step_ids = np.tile(np.arange(horizon, dtype=np.int64), n_episodes)
    visit_counts = np.bincount(episode_index, minlength=n_states)
    rewards = np.asarray(selected.y, dtype=np.float32).copy()
    if reward_mode == "terminal":
        rewards[step_ids != horizon - 1] = 0.0
    metadata = dict(selected.metadata)
    metadata.update(
        {
            "dataset": "h6_recurring",
            "split": "h6_train",
            "requested_episode_visits_per_state": int(visits),
            "realized_episode_visits_per_state": int(visits),
            "requested_visits_per_state": int(visits),
            "realized_visits_per_state": int(visits),
            "recurrence_exact": True,
            "n_episodes": int(n_episodes),
            "rows_per_episode": int(horizon),
            "n_unique_states": int(n_states),
            "minimum_episode_visits_per_state": int(visit_counts.min()),
            "maximum_episode_visits_per_state": int(visit_counts.max()),
            "mean_episode_visits_per_state": float(visit_counts.mean()),
            # Backward-compatible aliases; these count independent episodes,
            # not within-trajectory rows.
            "minimum_visits_per_state": int(visit_counts.min()),
            "maximum_visits_per_state": int(visit_counts.max()),
            "mean_visits_per_state": float(visit_counts.mean()),
            "reward_mode": reward_mode,
            "fixed_horizon": int(horizon),
            "terminal_reward_rows": int(np.sum(step_ids == horizon - 1)),
        }
    )
    sample_ids = np.arange(n, dtype=np.int64)
    return selected.with_updates(
        sample_id=sample_ids,
        state_id=np.asarray(prototype.state_id)[index],
        episode_id=episode_ids,
        step_id=step_ids,
        reward=rewards,
        metadata=metadata,
    )


def _seen_evaluation(batch: SemanticBatch, n: int, seed: int) -> SemanticBatch:
    unique = np.unique(np.asarray(batch.state_id), return_index=True)[1]
    rng = np.random.default_rng(seed)
    order = rng.permutation(unique)
    index = np.resize(order, n)
    selected = batch.select(index)
    sample = np.arange(2_000_000_000, 2_000_000_000 + n, dtype=np.int64)
    return selected.with_updates(
        sample_id=sample,
        episode_id=sample,
        step_id=np.zeros(n, dtype=np.int64),
        metadata={**dict(selected.metadata), "split": "seen_state_eval"},
    )


def _fresh_h6_nuisance_targets(
    n: int,
    positive_probabilities: np.ndarray,
    *,
    seed: int,
    draw: int,
) -> np.ndarray:
    """Draw hidden auxiliary action details independently for one presentation."""

    probabilities = np.asarray(positive_probabilities, dtype=np.float64)
    if probabilities.ndim != 1 or np.any((probabilities < 0) | (probabilities > 1)):
        raise ValueError("H6 nuisance probabilities must be a vector in [0,1]")
    result = np.zeros((n, len(probabilities)), dtype=np.int64)
    rng = np.random.default_rng(np.random.SeedSequence([int(seed), int(draw), int(n)]))
    for column, probability in enumerate(probabilities):
        positive_count = round(n * float(probability))
        if positive_count:
            result[rng.permutation(n)[:positive_count], column] = 1
    return result


@dataclass
class _NoisyTransform:
    config: NoiseConfig
    label_threshold: bool
    terminal_reward: bool = False
    nuisance_positive_probabilities: np.ndarray | None = None
    nuisance_seed: int = 0
    latest: NoiseApplication | None = None
    nuisance_draws: int = 0
    nuisance_presentations: int = 0

    def __post_init__(self) -> None:
        if self.nuisance_positive_probabilities is not None:
            probabilities = np.asarray(
                self.nuisance_positive_probabilities, dtype=np.float64
            )
            if probabilities.ndim != 1 or np.any(
                (probabilities < 0) | (probabilities > 1)
            ):
                raise ValueError("H6 nuisance probabilities must be a vector in [0,1]")
            self.nuisance_positive_probabilities = probabilities

    def __call__(self, batch: SemanticBatch, step: int) -> SemanticBatch:
        self.latest = apply_noise_with_diagnostics(batch, self.config, draw=step)
        result = self.latest.batch
        if self.nuisance_positive_probabilities is not None:
            nuisance = _fresh_h6_nuisance_targets(
                len(result),
                self.nuisance_positive_probabilities,
                seed=self.nuisance_seed,
                draw=step,
            )
            metadata = {
                **dict(result.metadata),
                "nuisance_label_resampling": "every_optimizer_presentation",
                "nuisance_label_draw": int(step),
            }
            result = result.with_updates(
                nuisance_targets=nuisance if nuisance.shape[1] else None,
                metadata=metadata,
            )
            self.nuisance_draws += 1
            self.nuisance_presentations += len(result)
        if self.terminal_reward and self.config.location == "reward":
            horizon = int(result.metadata.get("fixed_horizon", 1))
            terminal = np.asarray(result.step_id) == horizon - 1
            reward = np.asarray(result.reward, dtype=np.float32).copy()
            reward[~terminal] = 0.0
            result = result.with_updates(reward=reward)
        if self.label_threshold and self.config.location == "label":
            raw = np.asarray(result.target)
            signed = np.where(raw > 0, 1, np.where(raw < 0, -1, np.asarray(result.y)))
            result = result.with_updates(target=signed.astype(np.int8))
        return result

    def nuisance_diagnostics(self) -> dict[str, Any]:
        probabilities = self.nuisance_positive_probabilities
        return {
            "resampling": (
                "every_optimizer_presentation"
                if probabilities is not None and len(probabilities)
                else "no_auxiliary_details"
            ),
            "draws": int(self.nuisance_draws),
            "presentations": int(self.nuisance_presentations),
            "positive_probability_per_detail": (
                [] if probabilities is None else probabilities.tolist()
            ),
            "reward_relevant": False,
            "visible_in_observation": False,
        }


def _noise_summary(
    batch: SemanticBatch,
    noise_config: NoiseConfig,
) -> tuple[dict[str, Any], NoiseApplication]:
    first = apply_noise_with_diagnostics(batch, noise_config, draw=0)
    second = apply_noise_with_diagnostics(batch, noise_config, draw=1)
    per_stream: dict[str, Any] = {}
    for stream, values in first.noise.items():
        comparison = np.asarray(second.noise[stream])
        if np.std(values) > 0 and np.std(comparison) > 0:
            correlation = float(np.corrcoef(values, comparison)[0, 1])
        else:
            correlation = 1.0 if np.array_equal(values, comparison) else float("nan")
        per_stream[stream] = {
            **first.diagnostics[stream].as_dict(),
            "draw_repeat_correlation": correlation,
            "draw_mean_absolute_difference": float(np.mean(np.abs(values - comparison))),
        }
    return {"streams": per_stream}, first


def _rl_reward(raw_batch: SemanticBatch, actions: torch.Tensor, **_: Any) -> torch.Tensor:
    """Signed per-row contextual reward: +signal for right, -signal for left.

    With a clean signal ``Y`` this gives +1 for the intended choice and -1 for
    the other choice on reward-bearing rows. Terminal-mode nonterminal rows
    carry signal zero. When reward noise is configured, ``raw_batch.reward`` is
    ``Y + epsilon`` on the applicable rows and changes the RL objective directly.
    This is a shared contextual policy evaluated at each fixed-horizon row, not
    a recurrent full-episode policy.
    """

    action_sign = actions.to(torch.float32).mul(2).sub(1)
    signal = torch.as_tensor(
        np.asarray(raw_batch.reward), dtype=torch.float32, device=actions.device
    )
    return action_sign * signal


def run_h6(config: Mapping[str, Any], seed: int) -> ProtocolResult:
    """Run one H6 algorithm x temporal regime x intervention-location cell."""

    algorithm = _algorithm(config, "h6")
    n_train = _integer(config, "data.n_train", 4_096, minimum=2)
    training_plan = _h6_training_plan(config, n_train)
    n_eval = _integer(config, "data.n_eval", 4_096, minimum=2)
    q = _float(config, "data.q", 0.9)
    k = _integer(config, "data.k", 3, minimum=1)
    max_k = _integer(config, "data.max_k", k, minimum=k)
    state_dim = _integer(config, "data.state_dim", 0, minimum=0)
    visits = _integer(config, "h6.visits_per_state", 8, minimum=1)
    reward_mode = str(get_path(config, "h6.reward_mode", "dense_fixed_horizon"))
    if reward_mode not in {"terminal", "dense_fixed_horizon"}:
        raise ValueError("h6.reward_mode must be terminal or dense_fixed_horizon")
    fixed_horizon = _integer(config, "h6.fixed_horizon", 4, minimum=1)
    nuisance_bits = _integer(config, "h6.nuisance_bits", 8, minimum=0)
    nuisance_entropy = _float(config, "h6.nuisance_entropy_per_bit", 1.0)
    if not 0 <= nuisance_entropy <= 1:
        raise ValueError("h6.nuisance_entropy_per_bit must lie in [0,1]")
    nuisance_weight = _float(config, "h6.nuisance_weight", 1.0)
    if nuisance_weight < 0:
        raise ValueError("h6.nuisance_weight must be non-negative")
    # Auxiliary loss is averaged over heads by the generic trainer. Scaling by
    # the number of independently resampled details gives each one the
    # configured per-detail NLL weight.
    trajectory_auxiliary_weight = nuisance_weight * nuisance_bits

    train = _make_recurring_batch(
        n=n_train,
        q=q,
        k=k,
        max_k=max_k,
        state_dim=state_dim,
        visits=visits,
        nuisance_bits=nuisance_bits,
        nuisance_entropy=nuisance_entropy,
        seed=seed,
        reward_mode=reward_mode,
        fixed_horizon=fixed_horizon,
    )
    seen = _seen_evaluation(train, n_eval, seed + 2)
    new_context = _make_h6_context_dataset(
        n_eval,
        q,
        k,
        seed + 20_003,
        nuisance_bits=nuisance_bits,
        nuisance_entropy=nuisance_entropy,
        max_k=max_k,
        state_dim=state_dim,
    )
    new = new_context.with_updates(
        sample_id=np.arange(3_000_000_000, 3_000_000_000 + n_eval, dtype=np.int64),
        state_id=np.arange(3_000_000_000, 3_000_000_000 + n_eval, dtype=np.int64),
        metadata={**dict(new_context.metadata), "split": "new_state_eval"},
    )
    conflict_context = _make_h6_context_dataset(
        n_eval,
        0.0,
        k,
        seed + 30_007,
        nuisance_bits=nuisance_bits,
        nuisance_entropy=nuisance_entropy,
        max_k=max_k,
        state_dim=state_dim,
    )
    conflict = conflict_context.with_updates(
        sample_id=np.arange(4_000_000_000, 4_000_000_000 + n_eval, dtype=np.int64),
        state_id=np.arange(4_000_000_000, 4_000_000_000 + n_eval, dtype=np.int64),
        metadata={**dict(conflict_context.metadata), "split": "conflict_eval"},
    )

    structure = str(get_path(config, "h6.structure", "step"))
    location = str(get_path(config, "h6.location", "observation"))
    scale = _float(config, "h6.scale", 0.5)
    requested_bias = _float(config, "h6.bias", scale)
    normalized_structure = structure.lower().replace("-", "_")
    if normalized_structure in {"bias", "biased"}:
        effective_bias = math.copysign(min(abs(requested_bias), scale), requested_bias)
    else:
        effective_bias = None
    channels = get_path(config, "h6.channels", ("P",))
    noise_config = NoiseConfig(
        regime=structure,
        location=location,
        scale=scale,
        seed=seed + _integer(config, "h6.noise_seed_offset", 60_001),
        channels=channels if location.lower() in {"observation", "obs"} else None,
        bias=effective_bias,
    )
    diagnostic_summary, representative = _noise_summary(train, noise_config)
    nuisance_probabilities = (
        np.mean(np.asarray(train.nuisance_targets, dtype=np.float64), axis=0)
        if nuisance_bits and train.nuisance_targets is not None
        else np.empty(0, dtype=np.float64)
    )
    label_threshold = algorithm != "rl"
    transform = _NoisyTransform(
        noise_config,
        label_threshold=label_threshold,
        terminal_reward=reward_mode == "terminal",
        nuisance_positive_probabilities=nuisance_probabilities,
        nuisance_seed=seed + 80_021,
    )
    source = _SemanticSampler(train, transform, episode_coherent=True)

    model, model_report = build_model(
        train, config, seed, nuisance_heads=nuisance_bits
    )
    # Only RL consumes reward corruption; SFT/imitation consume label corruption.
    reward_callback = _rl_reward if algorithm == "rl" else None
    result, wall_seconds = _run_algorithm(
        algorithm=algorithm,
        model=model,
        source=source,
        evaluation=conflict,
        config=config,
        seed=seed,
        hypothesis="h6",
        nuisance_weight=trajectory_auxiliary_weight,
        reward_fn=reward_callback,
        training_steps=int(training_plan["optimizer_steps"]),
    )
    if result.samples_seen != int(training_plan["requested_presentations"]):
        raise RuntimeError(
            "H6 realized sample presentations do not match the fixed-epoch plan: "
            f"expected {training_plan['requested_presentations']}, got {result.samples_seen}"
        )

    final_train = evaluate_batch(model, train, config)
    final_seen = evaluate_batch(model, seen, config)
    final_new = evaluate_batch(model, new, config)
    final_conflict = evaluate_batch(model, conflict, config)
    interventions = evaluate_standard_interventions(model, conflict, config)
    costs = _costs(algorithm, result, wall_seconds)
    policy_collection = getattr(result, "policy_collection", {})
    if algorithm == "on_policy_imitation":
        costs["environment_interactions"] = int(
            policy_collection.get("queried_contexts", result.samples_seen)
        )
    if algorithm == "trajectory_sft":
        costs["labeled_actions"] = int(result.samples_seen * (1 + nuisance_bits))
    costs.update(
        {
            "sample_presentations": int(result.samples_seen),
            "fixed_horizon_source_episode_groups_sampled": int(
                (
                    policy_collection.get("queried_contexts", result.samples_seen)
                    if algorithm == "on_policy_imitation"
                    else result.samples_seen
                )
                // fixed_horizon
            ),
            "goal_labels_per_presentation": int(algorithm != "rl"),
            "auxiliary_labels_per_presentation": (
                nuisance_bits if algorithm == "trajectory_sft" else 0
            ),
        }
    )
    objective_uses_location = (
        location.lower() in {"observation", "obs"}
        or (algorithm == "rl" and location.lower() in {"reward", "rewards"})
        or (algorithm != "rl" and location.lower() in {"label", "labels"})
    )
    terminal_equivalence = (
        reward_mode == "terminal"
        and location.lower() in {"reward", "rewards"}
        and noise_config.regime in {"step_resampled", "episode_static"}
    )
    label_noise = next(iter(representative.noise.values()))
    threshold_flip_rate = (
        float(
            np.mean(
                np.where(
                    np.asarray(representative.batch.target) > 0,
                    1,
                    -1,
                )
                != np.asarray(train.target)
            )
        )
        if noise_config.location == "label"
        else None
    )
    diagnostic_summary.update(
        {
            "requested_structure": structure,
            "realized_structure": noise_config.regime,
            "location": noise_config.location,
            "scale": noise_config.scale,
            "requested_bias": requested_bias if noise_config.regime == "biased" else None,
            "effective_bias": noise_config.bias,
            "objective_uses_location": objective_uses_location,
            "terminal_step_episode_equivalence": terminal_equivalence,
            "label_threshold_flip_rate": threshold_flip_rate,
            "representative_noise_mean": float(np.mean(label_noise)),
        }
    )
    nuisance_evaluation = train.with_updates(
        nuisance_targets=(
            _fresh_h6_nuisance_targets(
                len(train),
                nuisance_probabilities,
                seed=seed + 80_021,
                draw=int(training_plan["optimizer_steps"]) + 1,
            )
            if nuisance_bits
            else None
        )
    )
    condition = f"{algorithm}:{noise_config.regime}:{noise_config.location}"
    summary = {
        "hypothesis": "h6",
        "algorithm": algorithm,
        "seed": int(seed),
        "model": model_report,
        "recurrence": {
            key: train.metadata[key]
            for key in (
                "requested_episode_visits_per_state",
                "realized_episode_visits_per_state",
                "requested_visits_per_state",
                "realized_visits_per_state",
                "recurrence_exact",
                "n_episodes",
                "rows_per_episode",
                "n_unique_states",
                "minimum_visits_per_state",
                "maximum_visits_per_state",
                "mean_visits_per_state",
                "reward_mode",
                "fixed_horizon",
                "terminal_reward_rows",
                "algorithm_abstraction",
                "full_episode_policy",
            )
        },
        "training": {
            **training_plan,
            "configured_train_steps": _integer(
                config, "train.steps", int(training_plan["optimizer_steps"]), minimum=1
            ),
            "configured_train_steps_ignored": True,
            "realized_optimizer_steps": int(result.optimizer_steps),
            "realized_presentations": int(result.samples_seen),
            "realized_epochs": float(result.samples_seen / n_train),
            "presentation_plan_matched": bool(
                result.samples_seen == int(training_plan["requested_presentations"])
            ),
        },
        "noise": diagnostic_summary,
        "costs": costs,
        "algorithm_abstraction": {
            "policy": "shared_repeated_contextual_policy",
            "full_episode_policy": False,
            "noise_source_batches_episode_coherent": True,
            "optimizer_minibatches_episode_coherent": (
                algorithm != "on_policy_imitation"
            ),
            "on_policy_queries_every_sampled_contextual_row": (
                algorithm == "on_policy_imitation"
            ),
            "fixed_horizon": fixed_horizon,
            "rl_reward_semantics": (
                "signed per-step contextual reward on every horizon row"
                if reward_mode == "dense_fixed_horizon"
                else "signed per-step contextual reward only on the terminal horizon row"
            ),
            "rl_return_aggregation": "none; each row is one contextual policy decision",
        },
        "train": final_train,
        "seen": final_seen,
        "new": final_new,
        "final": final_conflict,
        "seen_new_gap": {
            "target_accuracy": final_seen["target_accuracy"] - final_new["target_accuracy"],
            "rho_y": final_seen["rho_y"] - final_new["rho_y"],
        },
        "interventions": interventions,
        "auxiliary": {
            "semantics": train.metadata["nuisance_label_semantics"],
            "visible_in_observation": train.metadata[
                "nuisance_labels_visible_in_observation"
            ],
            "configured_per_detail_auxiliary_weight": nuisance_weight,
            "applied_auxiliary_weight": (
                trajectory_auxiliary_weight
                if algorithm == "trajectory_sft" and nuisance_bits
                else 0.0
            ),
            "loss_reduction": "goal_nll + per_detail_weight * sum(auxiliary_detail_nll)",
            **transform.nuisance_diagnostics(),
            **_nuisance_diagnostics(model, nuisance_evaluation, config),
        },
    }
    metrics = _history_metrics(result, hypothesis="h6", condition=condition)
    for split, values in (
        ("train", final_train),
        ("seen_state_eval", final_seen),
        ("new_state_eval", final_new),
        ("conflict_eval", final_conflict),
    ):
        metrics.extend(
            _final_records(
                values,
                hypothesis="h6",
                split=split,
                condition=condition,
                result=result,
            )
        )
    numeric_noise = {
        "objective_uses_location": float(objective_uses_location),
        "terminal_step_episode_equivalence": float(terminal_equivalence),
        "label_threshold_flip_rate": threshold_flip_rate or 0.0,
    }
    metrics.extend(
        _final_records(
            numeric_noise,
            hypothesis="h6",
            split="noise_diagnostic",
            condition=condition,
            result=result,
        )
    )
    metrics.extend(
        _final_records(
            {
                "requested_presentations": int(training_plan["requested_presentations"]),
                "realized_presentations": int(result.samples_seen),
                "requested_epochs": int(training_plan["requested_epochs"]),
                "realized_epochs": float(result.samples_seen / n_train),
                "realized_visits_per_state": int(
                    train.metadata["realized_visits_per_state"]
                ),
                "realized_episode_visits_per_state": int(
                    train.metadata["realized_episode_visits_per_state"]
                ),
                "n_unique_states": int(train.metadata["n_unique_states"]),
            },
            hypothesis="h6",
            split="evidence_accounting",
            condition=condition,
            result=result,
        )
    )
    return ProtocolResult(
        model=model,
        summary=summary,
        metrics=metrics,
        predictions=prediction_records(model, conflict, config, split="conflict_eval"),
        checkpoints=_checkpoints(result),
        evaluation_batch=conflict,
    )


RUNNERS: dict[str, Callable[[Mapping[str, Any], int], ProtocolResult]] = {
    "h5": run_h5,
    "h6": run_h6,
}


__all__ = ["RUNNERS", "run_h5", "run_h6"]
