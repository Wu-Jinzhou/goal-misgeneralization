"""Small, deterministic environments for controlled goal-selection experiments.

The environments in this module deliberately do not depend on Gym.  They expose
the small subset of an environment API needed by the experiments: ``reset``
returns an immutable observation and ``step`` returns a :class:`StepResult`.

There are two distinct notions of action in the project:

* :class:`Goal` is the binary goal/target identifier used by selectors.
* :class:`GridAction` is a primitive navigation action used by navigators.

Keeping the two types separate prevents a left target (``-1``) from being
silently confused with the primitive ``LEFT`` action (``3``).
"""

from __future__ import annotations

import copy
import hashlib
import itertools
import json
import random
from collections import deque
from dataclasses import dataclass
from enum import IntEnum
from numbers import Integral
from typing import Any, Iterable, Mapping, Sequence


Coordinate = tuple[int, int]


class Goal(IntEnum):
    """The explicit binary goal identifier used throughout ForkWorld."""

    LEFT = -1
    RIGHT = 1


class GridAction(IntEnum):
    """Primitive actions for grid navigation."""

    UP = 0
    RIGHT = 1
    DOWN = 2
    LEFT = 3


class NeutralAction(IntEnum):
    """Actions in :class:`NeutralChoiceSimulator`.

    A neutral gadget has a branch decision followed by a forced reconvergence.
    Goal actions are only valid at the final goal-selection step.
    """

    BRANCH_LEFT = 0
    BRANCH_RIGHT = 1
    MERGE = 2
    CHOOSE_LEFT = 3
    CHOOSE_RIGHT = 4


GRID_ACTIONS: tuple[GridAction, ...] = (
    GridAction.UP,
    GridAction.RIGHT,
    GridAction.DOWN,
    GridAction.LEFT,
)

GRID_DELTAS: Mapping[GridAction, Coordinate] = {
    GridAction.UP: (-1, 0),
    GridAction.RIGHT: (0, 1),
    GridAction.DOWN: (1, 0),
    GridAction.LEFT: (0, -1),
}


def coerce_goal(goal_id: Goal | int) -> Goal:
    """Validate and return a binary :class:`Goal`.

    Booleans are rejected even though ``bool`` is an ``int`` subclass.  Accepting
    them makes it too easy to accidentally mix a class index in ``{0, 1}`` with
    the semantic goal identifier in ``{-1, +1}``.
    """

    if isinstance(goal_id, bool) or not isinstance(goal_id, Integral):
        raise ValueError(f"goal_id must be integer -1 or +1, got {goal_id!r}")
    try:
        return Goal(int(goal_id))
    except ValueError as exc:
        raise ValueError(f"goal_id must be -1 or +1, got {goal_id!r}") from exc


def goal_to_index(goal_id: Goal | int) -> int:
    """Map semantic goal IDs ``{-1, +1}`` to class indices ``{0, 1}``."""

    return 0 if coerce_goal(goal_id) is Goal.LEFT else 1


def index_to_goal(index: int) -> Goal:
    """Map a binary class index back to its semantic goal ID."""

    if isinstance(index, bool) or index not in (0, 1):
        raise ValueError(f"goal class index must be 0 or 1, got {index!r}")
    return Goal.LEFT if index == 0 else Goal.RIGHT


def _stable_semantic_id(kind: str, payload: Mapping[str, Any]) -> str:
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]
    return f"{kind}-{digest}"


@dataclass(frozen=True)
class StepResult:
    """Gym-free result of one environment transition."""

    observation: Any
    reward: float
    terminated: bool
    truncated: bool
    info: Mapping[str, Any]

    @property
    def done(self) -> bool:
        return self.terminated or self.truncated


@dataclass(frozen=True)
class ChoiceObservation:
    """Observation for the one-step contextual-choice task."""

    context: tuple[float, ...]
    step: int
    semantic_id: str

    def vector(self) -> tuple[float, ...]:
        return self.context


class OneStepChoiceEnv:
    """A one-step contextual bandit with an explicit binary intended goal.

    The context is intentionally opaque to the environment.  Signal-generation
    code can pass ``(P, R_1, ..., R_k)`` or any richer flat feature vector.
    """

    action_space: tuple[Goal, Goal] = (Goal.LEFT, Goal.RIGHT)
    horizon: int = 1

    def __init__(
        self,
        context: Sequence[float],
        goal_id: Goal | int,
        *,
        seed: int = 0,
        semantic_key: str | None = None,
    ) -> None:
        if len(context) == 0:
            raise ValueError("context must contain at least one feature")
        self.context = tuple(float(value) for value in context)
        self.goal_id = coerce_goal(goal_id)
        self.seed = int(seed)
        self._semantic_key = semantic_key
        self.semantic_id = self._make_semantic_id()
        self._done = False
        self._selected_goal: Goal | None = None

    def _make_semantic_id(self) -> str:
        payload = {
            "context": self.context,
            "seed": self.seed,
            "semantic_key": self._semantic_key,
        }
        return _stable_semantic_id("choice", payload)

    @property
    def selected_goal(self) -> Goal | None:
        return self._selected_goal

    def observation(self) -> ChoiceObservation:
        return ChoiceObservation(
            context=self.context,
            step=int(self._done),
            semantic_id=self.semantic_id,
        )

    def reset(
        self,
        *,
        goal_id: Goal | int | None = None,
        context: Sequence[float] | None = None,
        seed: int | None = None,
    ) -> ChoiceObservation:
        if goal_id is not None:
            self.goal_id = coerce_goal(goal_id)
        if context is not None:
            if len(context) == 0:
                raise ValueError("context must contain at least one feature")
            self.context = tuple(float(value) for value in context)
        if seed is not None:
            self.seed = int(seed)
        self.semantic_id = self._make_semantic_id()
        self._done = False
        self._selected_goal = None
        return self.observation()

    def step(self, action: Goal | int) -> StepResult:
        if self._done:
            raise RuntimeError("episode is done; call reset() before step()")
        selected = coerce_goal(action)
        self._selected_goal = selected
        self._done = True
        success = selected is self.goal_id
        return StepResult(
            observation=self.observation(),
            reward=float(success),
            terminated=True,
            truncated=False,
            info={
                "semantic_id": self.semantic_id,
                "intended_goal": int(self.goal_id),
                "selected_goal": int(selected),
                "success": success,
                "horizon": self.horizon,
            },
        )


@dataclass(frozen=True)
class GridObservation:
    """Complete Markov observation for a static two-target grid."""

    height: int
    width: int
    position: Coordinate
    start: Coordinate
    left_target: Coordinate
    right_target: Coordinate
    obstacles: tuple[Coordinate, ...]
    step: int
    max_steps: int
    semantic_id: str

    def vector(self) -> tuple[float, ...]:
        """Return four flattened binary channels.

        Channel order is ``agent, obstacle, left target, right target``.  Goal ID
        is deliberately omitted: a goal-conditioned navigator receives it as a
        separate explicit input.
        """

        cells = self.height * self.width
        values = [0.0] * (4 * cells)

        def flat(coord: Coordinate) -> int:
            return coord[0] * self.width + coord[1]

        values[flat(self.position)] = 1.0
        for obstacle in self.obstacles:
            values[cells + flat(obstacle)] = 1.0
        values[2 * cells + flat(self.left_target)] = 1.0
        values[3 * cells + flat(self.right_target)] = 1.0
        return tuple(values)

    def as_dict(self) -> dict[str, Any]:
        return {
            "height": self.height,
            "width": self.width,
            "position": self.position,
            "start": self.start,
            "left_target": self.left_target,
            "right_target": self.right_target,
            "obstacles": self.obstacles,
            "step": self.step,
            "max_steps": self.max_steps,
            "semantic_id": self.semantic_id,
        }


def _in_bounds(coord: Coordinate, height: int, width: int) -> bool:
    return 0 <= coord[0] < height and 0 <= coord[1] < width


def _advance(position: Coordinate, action: GridAction) -> Coordinate:
    delta = GRID_DELTAS[action]
    return position[0] + delta[0], position[1] + delta[1]


def bfs_shortest_path(
    *,
    height: int,
    width: int,
    obstacles: Iterable[Coordinate],
    start: Coordinate,
    target: Coordinate,
    forbidden: Iterable[Coordinate] = (),
    action_order: Sequence[GridAction] = GRID_ACTIONS,
) -> tuple[GridAction, ...] | None:
    """Find a deterministic shortest path using breadth-first search.

    ``None`` means the target is unreachable; an empty tuple means the start is
    already the target.  The target itself is always permitted even if it was
    accidentally included in ``forbidden``.
    """

    if not _in_bounds(start, height, width):
        raise ValueError(f"start {start} is outside the grid")
    if not _in_bounds(target, height, width):
        raise ValueError(f"target {target} is outside the grid")
    blocked = set(obstacles) | set(forbidden)
    blocked.discard(start)
    blocked.discard(target)
    if start == target:
        return ()

    queue: deque[Coordinate] = deque([start])
    parents: dict[Coordinate, tuple[Coordinate, GridAction]] = {}
    visited = {start}

    while queue:
        position = queue.popleft()
        for raw_action in action_order:
            action = GridAction(raw_action)
            nxt = _advance(position, action)
            if not _in_bounds(nxt, height, width) or nxt in blocked or nxt in visited:
                continue
            parents[nxt] = (position, action)
            if nxt == target:
                reversed_actions: list[GridAction] = []
                cursor = target
                while cursor != start:
                    parent, parent_action = parents[cursor]
                    reversed_actions.append(parent_action)
                    cursor = parent
                return tuple(reversed(reversed_actions))
            visited.add(nxt)
            queue.append(nxt)
    return None


class GridWorld:
    """Deterministic two-target gridworld shared by fork and navigation tasks."""

    action_space: tuple[GridAction, ...] = GRID_ACTIONS

    def __init__(
        self,
        *,
        height: int,
        width: int,
        start: Coordinate,
        targets: Mapping[Goal | int, Coordinate],
        obstacles: Iterable[Coordinate] = (),
        goal_id: Goal | int,
        max_steps: int,
        seed: int = 0,
        kind: str = "grid",
        semantic_config: Mapping[str, Any] | None = None,
    ) -> None:
        if height < 2 or width < 2:
            raise ValueError("height and width must both be at least 2")
        if max_steps < 1:
            raise ValueError("max_steps must be positive")
        self.height = int(height)
        self.width = int(width)
        self.start = tuple(start)
        try:
            left_target = tuple(targets[Goal.LEFT])
            right_target = tuple(targets[Goal.RIGHT])
        except KeyError as exc:
            raise ValueError("targets must define Goal.LEFT and Goal.RIGHT") from exc
        self.targets: dict[Goal, Coordinate] = {
            Goal.LEFT: left_target,
            Goal.RIGHT: right_target,
        }
        self.obstacles = frozenset(tuple(coord) for coord in obstacles)
        self.goal_id = coerce_goal(goal_id)
        self.max_steps = int(max_steps)
        self.seed = int(seed)
        self.kind = str(kind)
        self._semantic_config = dict(semantic_config or {})
        self._validate_layout()
        self.semantic_id = self._make_semantic_id()
        self._episode_seed = self.seed
        self.position = self.start
        self.steps = 0
        self._terminated = False
        self._truncated = False
        self._reached_goal: Goal | None = None

    def _validate_layout(self) -> None:
        special = [self.start, self.targets[Goal.LEFT], self.targets[Goal.RIGHT]]
        for coord in special:
            if not _in_bounds(coord, self.height, self.width):
                raise ValueError(f"special cell {coord} is outside the grid")
        if len(set(special)) != len(special):
            raise ValueError("start and the two targets must be distinct")
        for obstacle in self.obstacles:
            if not _in_bounds(obstacle, self.height, self.width):
                raise ValueError(f"obstacle {obstacle} is outside the grid")
        if self.obstacles.intersection(special):
            raise ValueError("start and target cells cannot be obstacles")

        for goal in (Goal.LEFT, Goal.RIGHT):
            other = Goal.RIGHT if goal is Goal.LEFT else Goal.LEFT
            path = bfs_shortest_path(
                height=self.height,
                width=self.width,
                obstacles=self.obstacles,
                start=self.start,
                target=self.targets[goal],
                forbidden=(self.targets[other],),
            )
            if path is None:
                raise ValueError(
                    f"goal {int(goal):+d} is unreachable without crossing other target"
                )

    def _make_semantic_id(self) -> str:
        payload = {
            "height": self.height,
            "width": self.width,
            "start": self.start,
            "targets": {
                "left": self.targets[Goal.LEFT],
                "right": self.targets[Goal.RIGHT],
            },
            "obstacles": sorted(self.obstacles),
            "generation_seed": self.seed,
            "kind": self.kind,
            "semantic_config": self._semantic_config,
        }
        return _stable_semantic_id(self.kind, payload)

    @property
    def observation_dim(self) -> int:
        return 4 * self.height * self.width

    @property
    def done(self) -> bool:
        return self._terminated or self._truncated

    @property
    def terminated(self) -> bool:
        return self._terminated

    @property
    def truncated(self) -> bool:
        return self._truncated

    @property
    def reached_goal(self) -> Goal | None:
        return self._reached_goal

    @property
    def free_cells(self) -> tuple[Coordinate, ...]:
        return tuple(
            (row, col)
            for row in range(self.height)
            for col in range(self.width)
            if (row, col) not in self.obstacles
        )

    def observation_at(self, position: Coordinate, *, step: int = 0) -> GridObservation:
        if not _in_bounds(position, self.height, self.width) or position in self.obstacles:
            raise ValueError(f"position {position} is not a free grid cell")
        if not 0 <= step <= self.max_steps:
            raise ValueError("step must lie between zero and max_steps")
        return GridObservation(
            height=self.height,
            width=self.width,
            position=tuple(position),
            start=self.start,
            left_target=self.targets[Goal.LEFT],
            right_target=self.targets[Goal.RIGHT],
            obstacles=tuple(sorted(self.obstacles)),
            step=int(step),
            max_steps=self.max_steps,
            semantic_id=self.semantic_id,
        )

    def observation(self) -> GridObservation:
        return self.observation_at(self.position, step=self.steps)

    def reset(
        self,
        *,
        goal_id: Goal | int | None = None,
        seed: int | None = None,
    ) -> GridObservation:
        """Reset episode state without changing the generated layout.

        A reset seed is retained in transition metadata for paired experiments;
        random layouts are generated at construction time so resetting cannot
        accidentally change the map between a selected-goal and oracle rollout.
        """

        if goal_id is not None:
            self.goal_id = coerce_goal(goal_id)
        self._episode_seed = self.seed if seed is None else int(seed)
        self.position = self.start
        self.steps = 0
        self._terminated = False
        self._truncated = False
        self._reached_goal = None
        return self.observation()

    def clone(self, *, goal_id: Goal | int | None = None) -> GridWorld:
        cloned = copy.deepcopy(self)
        cloned.reset(goal_id=self.goal_id if goal_id is None else goal_id)
        return cloned

    def shortest_path(
        self,
        goal_id: Goal | int,
        *,
        start: Coordinate | None = None,
        avoid_other_target: bool = True,
    ) -> tuple[GridAction, ...] | None:
        goal = coerce_goal(goal_id)
        other = Goal.RIGHT if goal is Goal.LEFT else Goal.LEFT
        forbidden: tuple[Coordinate, ...] = (
            (self.targets[other],) if avoid_other_target else ()
        )
        return bfs_shortest_path(
            height=self.height,
            width=self.width,
            obstacles=self.obstacles,
            start=self.position if start is None else tuple(start),
            target=self.targets[goal],
            forbidden=forbidden,
        )

    def distance_to_goal(
        self,
        goal_id: Goal | int,
        *,
        start: Coordinate | None = None,
    ) -> int | None:
        path = self.shortest_path(goal_id, start=start)
        return None if path is None else len(path)

    def expert_action(
        self,
        goal_id: Goal | int,
        *,
        position: Coordinate | None = None,
    ) -> GridAction | None:
        path = self.shortest_path(goal_id, start=position)
        if path is None:
            raise ValueError(f"goal {int(coerce_goal(goal_id)):+d} is unreachable")
        return path[0] if path else None

    def legal_actions(self, *, position: Coordinate | None = None) -> tuple[GridAction, ...]:
        origin = self.position if position is None else tuple(position)
        actions: list[GridAction] = []
        for action in GRID_ACTIONS:
            nxt = _advance(origin, action)
            if _in_bounds(nxt, self.height, self.width) and nxt not in self.obstacles:
                actions.append(action)
        return tuple(actions)

    def step(self, action: GridAction | int) -> StepResult:
        if self.done:
            raise RuntimeError("episode is done; call reset() before step()")
        try:
            primitive = GridAction(action)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"invalid grid action {action!r}") from exc

        candidate = _advance(self.position, primitive)
        collision = (
            not _in_bounds(candidate, self.height, self.width)
            or candidate in self.obstacles
        )
        if not collision:
            self.position = candidate
        self.steps += 1

        reached: Goal | None = None
        for goal, target in self.targets.items():
            if self.position == target:
                reached = goal
                break
        if reached is not None:
            self._reached_goal = reached
            self._terminated = True
        elif self.steps >= self.max_steps:
            self._truncated = True

        success = reached is self.goal_id
        return StepResult(
            observation=self.observation(),
            reward=float(success),
            terminated=self._terminated,
            truncated=self._truncated,
            info={
                "semantic_id": self.semantic_id,
                "episode_seed": self._episode_seed,
                "intended_goal": int(self.goal_id),
                "reached_goal": None if reached is None else int(reached),
                "success": success,
                "collision": collision,
                "step": self.steps,
                "max_steps": self.max_steps,
            },
        )

    def render_ascii(self) -> str:
        symbols: list[list[str]] = [["#"] * self.width for _ in range(self.height)]
        for cell in self.free_cells:
            symbols[cell[0]][cell[1]] = "."
        symbols[self.start[0]][self.start[1]] = "S"
        left = self.targets[Goal.LEFT]
        right = self.targets[Goal.RIGHT]
        symbols[left[0]][left[1]] = "L"
        symbols[right[0]][right[1]] = "R"
        symbols[self.position[0]][self.position[1]] = "A"
        return "\n".join("".join(row) for row in symbols)


def bfs_expert_action(
    env: GridWorld,
    goal_id: Goal | int,
    *,
    position: Coordinate | None = None,
) -> GridAction | None:
    """Convenience wrapper for the deterministic BFS expert."""

    return env.expert_action(goal_id, position=position)


class ForkGridWorld(GridWorld):
    """A symmetric corridor with one terminal left/right fork.

    The two shortest routes have exactly seven primitive moves.  With the
    default horizon, a successful trajectory cannot spend steps on a detour,
    and reaching either branch endpoint terminates the episode.  Thus the first
    horizontal action at :attr:`fork_position` is the sole goal-selecting
    decision on an expert trajectory.
    """

    fork_position: Coordinate = (3, 3)

    def __init__(
        self,
        goal_id: Goal | int,
        *,
        seed: int = 0,
        max_steps: int = 7,
    ) -> None:
        height = width = 7
        start = (6, 3)
        left_target = (1, 1)
        right_target = (1, 5)
        walkable = {
            start,
            (5, 3),
            (4, 3),
            self.fork_position,
            (3, 2),
            (3, 1),
            (2, 1),
            left_target,
            (3, 4),
            (3, 5),
            (2, 5),
            right_target,
        }
        obstacles = {
            (row, col)
            for row in range(height)
            for col in range(width)
            if (row, col) not in walkable
        }
        super().__init__(
            height=height,
            width=width,
            start=start,
            targets={Goal.LEFT: left_target, Goal.RIGHT: right_target},
            obstacles=obstacles,
            goal_id=goal_id,
            max_steps=max_steps,
            seed=seed,
            kind="fork",
            semantic_config={"fork_position": self.fork_position},
        )
        left_distance = self.distance_to_goal(Goal.LEFT)
        right_distance = self.distance_to_goal(Goal.RIGHT)
        if left_distance != right_distance:
            raise AssertionError("fork routes must have equal length")

    @property
    def route_length(self) -> int:
        distance = self.distance_to_goal(Goal.LEFT, start=self.start)
        assert distance is not None
        return distance

    def decision_action(self, goal_id: Goal | int) -> GridAction:
        goal = coerce_goal(goal_id)
        action = self.expert_action(goal, position=self.fork_position)
        assert action in (GridAction.LEFT, GridAction.RIGHT)
        return action


def _cells_along_path(start: Coordinate, actions: Iterable[GridAction]) -> set[Coordinate]:
    position = start
    cells = {position}
    for action in actions:
        position = _advance(position, action)
        cells.add(position)
    return cells


class RandomObstacleNavigationEnv(GridWorld):
    """A seeded random obstacle map with two independently reachable targets.

    Obstacles are sampled first.  If sampling disconnects a target, a shortest
    corridor is carved while treating the other target as forbidden.  This
    construction guarantees that both target-conditioned tasks are solvable
    under the environment's terminal-on-any-target semantics.
    """

    def __init__(
        self,
        goal_id: Goal | int,
        *,
        height: int = 9,
        width: int = 9,
        obstacle_probability: float = 0.22,
        seed: int = 0,
        max_steps: int | None = None,
        min_target_distance: int = 4,
    ) -> None:
        if height < 5 or width < 5:
            raise ValueError("random navigation grids must be at least 5 by 5")
        if not 0.0 <= obstacle_probability < 1.0:
            raise ValueError("obstacle_probability must lie in [0, 1)")
        if min_target_distance < 1:
            raise ValueError("min_target_distance must be positive")

        rng = random.Random(int(seed))
        start_rows = range(max(1, (2 * height) // 3), height)
        start_candidates = [(row, col) for row in start_rows for col in range(width)]
        left_candidates = [
            (row, col)
            for row in range(height)
            for col in range(max(1, width // 2))
        ]
        right_candidates = [
            (row, col)
            for row in range(height)
            for col in range((width + 1) // 2, width)
        ]

        def manhattan(a: Coordinate, b: Coordinate) -> int:
            return abs(a[0] - b[0]) + abs(a[1] - b[1])

        start: Coordinate | None = None
        left_target: Coordinate | None = None
        right_target: Coordinate | None = None
        for _ in range(1_000):
            proposed_start = rng.choice(start_candidates)
            proposed_left = rng.choice(left_candidates)
            proposed_right = rng.choice(right_candidates)
            if len({proposed_start, proposed_left, proposed_right}) < 3:
                continue
            if min(
                manhattan(proposed_start, proposed_left),
                manhattan(proposed_start, proposed_right),
            ) < min_target_distance:
                continue
            start, left_target, right_target = (
                proposed_start,
                proposed_left,
                proposed_right,
            )
            break
        if start is None or left_target is None or right_target is None:
            raise RuntimeError("failed to sample well-separated start and target cells")

        special = {start, left_target, right_target}
        obstacles = {
            (row, col)
            for row in range(height)
            for col in range(width)
            if (row, col) not in special and rng.random() < obstacle_probability
        }

        targets = {Goal.LEFT: left_target, Goal.RIGHT: right_target}
        for goal in (Goal.LEFT, Goal.RIGHT):
            other = Goal.RIGHT if goal is Goal.LEFT else Goal.LEFT
            path = bfs_shortest_path(
                height=height,
                width=width,
                obstacles=obstacles,
                start=start,
                target=targets[goal],
                forbidden=(targets[other],),
            )
            if path is None:
                path_to_carve = bfs_shortest_path(
                    height=height,
                    width=width,
                    obstacles=(),
                    start=start,
                    target=targets[goal],
                    forbidden=(targets[other],),
                )
                if path_to_carve is None:  # Defensive; impossible on a >=5x5 empty grid.
                    raise RuntimeError("could not carve a route to target")
                obstacles.difference_update(_cells_along_path(start, path_to_carve))

        super().__init__(
            height=height,
            width=width,
            start=start,
            targets=targets,
            obstacles=obstacles,
            goal_id=goal_id,
            max_steps=max_steps if max_steps is not None else 2 * height * width,
            seed=seed,
            kind="navigation",
            semantic_config={
                "requested_obstacle_probability": float(obstacle_probability),
                "min_target_distance": int(min_target_distance),
            },
        )
        self.requested_obstacle_probability = float(obstacle_probability)
        self.realized_obstacle_probability = len(self.obstacles) / (height * width - 3)
        self.min_target_distance = int(min_target_distance)


# Shorter alias used in configuration files and interactive experiments.
NavigationGridWorld = RandomObstacleNavigationEnv


@dataclass(frozen=True)
class NeutralObservation:
    """State in a chain of reward-equivalent neutral-choice gadgets."""

    phase: str
    gadget_index: int
    branch: int | None
    remaining_steps: int
    horizon: int
    semantic_id: str

    def vector(self) -> tuple[float, ...]:
        phase_vector = {
            "decision": (1.0, 0.0, 0.0, 0.0),
            "branch": (0.0, 1.0, 0.0, 0.0),
            "goal": (0.0, 0.0, 1.0, 0.0),
            "terminal": (0.0, 0.0, 0.0, 1.0),
        }[self.phase]
        denominator = max(1, self.horizon)
        branch_left = float(self.branch == 0)
        branch_right = float(self.branch == 1)
        return (
            *phase_vector,
            self.gadget_index / denominator,
            branch_left,
            branch_right,
            self.remaining_steps / denominator,
        )


@dataclass(frozen=True)
class NeutralTrajectoryStep:
    observation: NeutralObservation
    action: NeutralAction
    reward: float
    next_observation: NeutralObservation
    terminated: bool


@dataclass(frozen=True)
class NeutralTrajectory:
    semantic_id: str
    intended_goal: Goal
    selected_goal: Goal
    neutral_choices: tuple[int, ...]
    steps: tuple[NeutralTrajectoryStep, ...]
    total_reward: float
    horizon: int

    @property
    def success(self) -> bool:
        return self.selected_goal is self.intended_goal


class NeutralChoiceSimulator:
    """Fixed-horizon simulator for nuisance-rich successful demonstrations.

    Each gadget has two steps: choose branch A/B, then take a shared ``MERGE``
    action.  Both branches reconverge before the next gadget.  The final step
    chooses a target, and only that final choice affects reward.  Consequently
    changing any subset of neutral choices preserves terminal reward, horizon,
    and reward timing exactly.
    """

    def __init__(
        self,
        num_gadgets: int,
        goal_id: Goal | int,
        *,
        seed: int = 0,
    ) -> None:
        if num_gadgets < 0:
            raise ValueError("num_gadgets cannot be negative")
        self.num_gadgets = int(num_gadgets)
        self.goal_id = coerce_goal(goal_id)
        self.seed = int(seed)
        self.horizon = 2 * self.num_gadgets + 1
        self.semantic_id = _stable_semantic_id(
            "neutral",
            {"num_gadgets": self.num_gadgets, "seed": self.seed},
        )
        self._rng = random.Random(self.seed)
        self.reset()

    @property
    def done(self) -> bool:
        return self._done

    @property
    def valid_actions(self) -> tuple[NeutralAction, ...]:
        if self._done:
            return ()
        if self._phase == "decision":
            return (NeutralAction.BRANCH_LEFT, NeutralAction.BRANCH_RIGHT)
        if self._phase == "branch":
            return (NeutralAction.MERGE,)
        return (NeutralAction.CHOOSE_LEFT, NeutralAction.CHOOSE_RIGHT)

    def observation(self) -> NeutralObservation:
        return NeutralObservation(
            phase="terminal" if self._done else self._phase,
            gadget_index=self._gadget_index,
            branch=self._branch,
            remaining_steps=self.horizon - self._step,
            horizon=self.horizon,
            semantic_id=self.semantic_id,
        )

    def reset(
        self,
        *,
        goal_id: Goal | int | None = None,
        seed: int | None = None,
    ) -> NeutralObservation:
        if goal_id is not None:
            self.goal_id = coerce_goal(goal_id)
        if seed is not None:
            self._rng = random.Random(int(seed))
        self._step = 0
        self._gadget_index = 0
        self._phase = "goal" if self.num_gadgets == 0 else "decision"
        self._branch: int | None = None
        self._done = False
        self._neutral_choices: list[int] = []
        self._selected_goal: Goal | None = None
        return self.observation()

    def step(self, action: NeutralAction | int) -> StepResult:
        if self._done:
            raise RuntimeError("episode is done; call reset() before step()")
        try:
            neutral_action = NeutralAction(action)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"invalid neutral action {action!r}") from exc
        if neutral_action not in self.valid_actions:
            raise ValueError(
                f"action {neutral_action.name} is invalid during {self._phase}; "
                f"valid actions are {[action.name for action in self.valid_actions]}"
            )

        reward = 0.0
        if self._phase == "decision":
            branch = int(neutral_action is NeutralAction.BRANCH_RIGHT)
            self._branch = branch
            self._neutral_choices.append(branch)
            self._phase = "branch"
        elif self._phase == "branch":
            self._branch = None
            self._gadget_index += 1
            self._phase = "goal" if self._gadget_index == self.num_gadgets else "decision"
        else:
            self._selected_goal = (
                Goal.LEFT
                if neutral_action is NeutralAction.CHOOSE_LEFT
                else Goal.RIGHT
            )
            reward = float(self._selected_goal is self.goal_id)
            self._done = True

        self._step += 1
        if self._done and self._step != self.horizon:
            raise AssertionError("neutral simulator violated its fixed horizon")
        return StepResult(
            observation=self.observation(),
            reward=reward,
            terminated=self._done,
            truncated=False,
            info={
                "semantic_id": self.semantic_id,
                "intended_goal": int(self.goal_id),
                "selected_goal": (
                    None if self._selected_goal is None else int(self._selected_goal)
                ),
                "neutral_choices": tuple(self._neutral_choices),
                "step": self._step,
                "horizon": self.horizon,
                "success": self._done and bool(reward),
            },
        )

    def rollout(
        self,
        *,
        neutral_choices: Sequence[int] | None = None,
        selected_goal: Goal | int | None = None,
        sampling_seed: int | None = None,
    ) -> NeutralTrajectory:
        """Generate one fixed-horizon trajectory.

        Omitting ``neutral_choices`` samples independent choices from the given
        seed.  Omitting ``selected_goal`` produces a successful trajectory by
        clamping the final choice to the intended goal.
        """

        rng = self._rng if sampling_seed is None else random.Random(int(sampling_seed))
        if neutral_choices is None:
            choices = tuple(rng.randrange(2) for _ in range(self.num_gadgets))
        else:
            choices = tuple(int(choice) for choice in neutral_choices)
            if len(choices) != self.num_gadgets or any(choice not in (0, 1) for choice in choices):
                raise ValueError(
                    f"neutral_choices must contain {self.num_gadgets} binary values"
                )
        selected = self.goal_id if selected_goal is None else coerce_goal(selected_goal)

        observation = self.reset()
        trajectory_steps: list[NeutralTrajectoryStep] = []
        for choice in choices:
            branch_action = (
                NeutralAction.BRANCH_LEFT if choice == 0 else NeutralAction.BRANCH_RIGHT
            )
            result = self.step(branch_action)
            trajectory_steps.append(
                NeutralTrajectoryStep(
                    observation=observation,
                    action=branch_action,
                    reward=result.reward,
                    next_observation=result.observation,
                    terminated=result.terminated,
                )
            )
            observation = result.observation
            result = self.step(NeutralAction.MERGE)
            trajectory_steps.append(
                NeutralTrajectoryStep(
                    observation=observation,
                    action=NeutralAction.MERGE,
                    reward=result.reward,
                    next_observation=result.observation,
                    terminated=result.terminated,
                )
            )
            observation = result.observation

        goal_action = (
            NeutralAction.CHOOSE_LEFT if selected is Goal.LEFT else NeutralAction.CHOOSE_RIGHT
        )
        result = self.step(goal_action)
        trajectory_steps.append(
            NeutralTrajectoryStep(
                observation=observation,
                action=goal_action,
                reward=result.reward,
                next_observation=result.observation,
                terminated=result.terminated,
            )
        )
        trajectory = NeutralTrajectory(
            semantic_id=self.semantic_id,
            intended_goal=self.goal_id,
            selected_goal=selected,
            neutral_choices=choices,
            steps=tuple(trajectory_steps),
            total_reward=sum(step.reward for step in trajectory_steps),
            horizon=self.horizon,
        )
        if len(trajectory.steps) != self.horizon:
            raise AssertionError("neutral rollout did not have the declared horizon")
        return trajectory

    def equivalent_successful_trajectories(self) -> tuple[NeutralTrajectory, ...]:
        """Enumerate all reward-equivalent successful trajectories.

        This helper is intended for tests and small nuisance sweeps.  To avoid an
        accidental exponential allocation it is capped at sixteen gadgets.
        """

        if self.num_gadgets > 16:
            raise ValueError("refusing to enumerate more than 2^16 trajectories")
        return tuple(
            self.rollout(neutral_choices=choices, selected_goal=self.goal_id)
            for choices in itertools.product((0, 1), repeat=self.num_gadgets)
        )


__all__ = [
    "ChoiceObservation",
    "Coordinate",
    "ForkGridWorld",
    "Goal",
    "GridAction",
    "GridObservation",
    "GridWorld",
    "NavigationGridWorld",
    "NeutralAction",
    "NeutralChoiceSimulator",
    "NeutralObservation",
    "NeutralTrajectory",
    "NeutralTrajectoryStep",
    "OneStepChoiceEnv",
    "RandomObstacleNavigationEnv",
    "StepResult",
    "bfs_expert_action",
    "bfs_shortest_path",
    "coerce_goal",
    "goal_to_index",
    "index_to_goal",
]
