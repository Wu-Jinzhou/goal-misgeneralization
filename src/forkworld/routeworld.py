"""Multi-fork route mazes and matched semantic route datasets.

RouteWorld separates two jobs that are easy to conflate in a sequential task:

* a learned selector chooses one semantic branch bit at each true fork;
* a scripted corridor controller executes every forced primitive grid move.

The environment is a perfect binary tree embedded in a wall grid.  Every leaf
is terminal, every root-to-leaf path has the same length, and the rewarded leaf
is never marked in the model-visible observation.  A route is therefore a
sequence of ``-1`` (left) and ``+1`` (right) choices rather than a visually
distinguished target.

The data API exposes a complete path through two controlled encodings.  Each
route bit has a simple proxy with exact finite-sample accuracy and an exact
degree-k interaction code.  Zero padding and explicit presence masks keep the
feature width fixed when active route depth or interaction degree changes.
"""

from __future__ import annotations

import hashlib
import itertools
import json
from collections import deque
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from numbers import Integral
from typing import Any

import numpy as np
from numpy.typing import NDArray

from .envs import GRID_DELTAS, Coordinate, GridAction, StepResult

SignArray = NDArray[np.int8]
FloatArray = NDArray[np.float32]
BoolArray = NDArray[np.bool_]
IntArray = NDArray[np.int64]

MAX_ROUTE_DEPTH = 4


def _coerce_sign(value: int, *, name: str = "route choice") -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or int(value) not in (-1, 1):
        raise ValueError(f"{name} must be integer -1 or +1, got {value!r}")
    return int(value)


def _coerce_route(route: Sequence[int], depth: int, *, name: str = "route") -> tuple[int, ...]:
    normalized = tuple(_coerce_sign(value, name=f"{name} bit") for value in route)
    if len(normalized) != depth:
        raise ValueError(f"{name} must contain exactly {depth} bits, got {len(normalized)}")
    return normalized


def _advance(position: Coordinate, action: GridAction) -> Coordinate:
    delta = GRID_DELTAS[action]
    return position[0] + delta[0], position[1] + delta[1]


def _stable_id(kind: str, payload: Mapping[str, Any]) -> str:
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]
    return f"{kind}-{digest}"


@dataclass(frozen=True)
class RouteMazeObservation:
    """Target-free Markov observation of the physical maze.

    The three flattened channels are agent position, walls, and generic terminal
    leaves.  All leaves look identical; neither ``target_route`` nor a rewarded
    leaf coordinate appears here.
    """

    height: int
    width: int
    position: Coordinate
    start: Coordinate
    walls: tuple[Coordinate, ...]
    leaves: tuple[Coordinate, ...]
    step: int
    max_steps: int
    semantic_id: str

    def vector(self) -> tuple[float, ...]:
        cells = self.height * self.width
        values = [0.0] * (3 * cells)

        def flat(coordinate: Coordinate) -> int:
            return coordinate[0] * self.width + coordinate[1]

        values[flat(self.position)] = 1.0
        for wall in self.walls:
            values[cells + flat(wall)] = 1.0
        for leaf in self.leaves:
            values[2 * cells + flat(leaf)] = 1.0
        return tuple(values)

    def as_dict(self) -> dict[str, Any]:
        return {
            "height": self.height,
            "width": self.width,
            "position": self.position,
            "start": self.start,
            "walls": self.walls,
            "leaves": self.leaves,
            "step": self.step,
            "max_steps": self.max_steps,
            "semantic_id": self.semantic_id,
        }


class BinaryTreeRouteMaze:
    """An equal-depth binary tree embedded in a rectangular wall grid.

    Internal tree nodes lie three rows apart.  One forced downward move reaches
    a junction, a horizontal left/right segment implements the learned choice,
    and two forced downward moves reach the selected child.  At level ``l`` the
    horizontal segment has length ``2**(depth-l-1)``.  Consequently every route
    has length ``2**depth - 1 + 3*depth`` regardless of its branch bits.
    """

    action_space: tuple[GridAction, ...] = tuple(GridAction)
    depth: int
    target_route: tuple[int, ...]
    height: int
    width: int
    start: Coordinate
    max_steps: int
    semantic_id: str
    position: Coordinate
    steps: int
    _nodes: dict[tuple[int, int], Coordinate]
    _decisions: dict[tuple[int, int], Coordinate]
    _free_cells: frozenset[Coordinate]
    _walls: frozenset[Coordinate]
    _route_to_leaf: dict[tuple[int, ...], Coordinate]
    _leaf_to_route: dict[Coordinate, tuple[int, ...]]
    _reached_route: tuple[int, ...] | None
    _terminated: bool
    _truncated: bool

    def __init__(self, depth: int, target_route: Sequence[int]) -> None:
        if isinstance(depth, bool) or not isinstance(depth, Integral):
            raise ValueError("depth must be an integer")
        self.depth = int(depth)
        if not 1 <= self.depth <= MAX_ROUTE_DEPTH:
            raise ValueError(f"depth must lie in [1, {MAX_ROUTE_DEPTH}]")
        self.target_route = _coerce_route(target_route, self.depth, name="target_route")
        self.height = 3 * self.depth + 1
        self.width = 2 ** (self.depth + 1) + 1
        self.start = self.node_position(0, 0)
        self._nodes = {
            (level, index): self.node_position(level, index)
            for level in range(self.depth + 1)
            for index in range(2**level)
        }
        self._decisions = {
            (level, index): (3 * level + 1, self.node_position(level, index)[1])
            for level in range(self.depth)
            for index in range(2**level)
        }
        self._free_cells = self._build_free_cells()
        self._walls = frozenset(
            (row, column)
            for row in range(self.height)
            for column in range(self.width)
            if (row, column) not in self._free_cells
        )
        routes = tuple(itertools.product((-1, 1), repeat=self.depth))
        self._route_to_leaf = {
            route: self._nodes[(self.depth, self.route_index(route))] for route in routes
        }
        self._leaf_to_route = {leaf: route for route, leaf in self._route_to_leaf.items()}
        self.semantic_id = _stable_id(
            "route-maze",
            {
                "depth": self.depth,
                "height": self.height,
                "width": self.width,
                "walls": sorted(self._walls),
                "leaves": sorted(self._leaf_to_route),
            },
        )
        self.max_steps = self.route_length
        self.reset()

    @staticmethod
    def route_index(route: Sequence[int]) -> int:
        index = 0
        for bit in route:
            index = 2 * index + int(_coerce_sign(bit) > 0)
        return index

    def node_position(self, level: int, index: int) -> Coordinate:
        if not 0 <= level <= self.depth or not 0 <= index < 2**level:
            raise ValueError(f"invalid tree node ({level}, {index}) for depth {self.depth}")
        return 3 * level, (2 * index + 1) * 2 ** (self.depth - level)

    def _build_free_cells(self) -> frozenset[Coordinate]:
        free: set[Coordinate] = set(self._nodes.values())
        for level in range(self.depth):
            horizontal_distance = 2 ** (self.depth - level - 1)
            for index in range(2**level):
                parent_row, parent_column = self._nodes[(level, index)]
                decision_row = parent_row + 1
                free.add((decision_row, parent_column))
                for direction in (-1, 1):
                    child_index = 2 * index + int(direction > 0)
                    child_row, child_column = self._nodes[(level + 1, child_index)]
                    expected_column = parent_column + direction * horizontal_distance
                    if child_column != expected_column:  # pragma: no cover - construction proof
                        raise AssertionError("binary-tree embedding lost horizontal symmetry")
                    for column in range(
                        min(parent_column, child_column), max(parent_column, child_column) + 1
                    ):
                        free.add((decision_row, column))
                    free.add((decision_row + 1, child_column))
                    free.add((child_row, child_column))
        return frozenset(free)

    @property
    def route_length(self) -> int:
        return 2**self.depth - 1 + 3 * self.depth

    @property
    def free_cells(self) -> tuple[Coordinate, ...]:
        return tuple(sorted(self._free_cells))

    @property
    def walls(self) -> tuple[Coordinate, ...]:
        return tuple(sorted(self._walls))

    @property
    def leaves(self) -> tuple[Coordinate, ...]:
        return tuple(sorted(self._leaf_to_route))

    @property
    def decision_positions(self) -> Mapping[tuple[int, int], Coordinate]:
        return dict(self._decisions)

    @property
    def target_leaf(self) -> Coordinate:
        """Environment-side rewarded leaf; deliberately absent from observations."""

        return self._route_to_leaf[self.target_route]

    @property
    def reached_route(self) -> tuple[int, ...] | None:
        return self._reached_route

    @property
    def done(self) -> bool:
        return self._terminated or self._truncated

    @property
    def terminated(self) -> bool:
        return self._terminated

    @property
    def truncated(self) -> bool:
        return self._truncated

    def route_to_leaf(self, route: Sequence[int]) -> Coordinate:
        return self._route_to_leaf[_coerce_route(route, self.depth)]

    def observation(self) -> RouteMazeObservation:
        return RouteMazeObservation(
            height=self.height,
            width=self.width,
            position=self.position,
            start=self.start,
            walls=self.walls,
            leaves=self.leaves,
            step=self.steps,
            max_steps=self.max_steps,
            semantic_id=self.semantic_id,
        )

    def reset(self, *, target_route: Sequence[int] | None = None) -> RouteMazeObservation:
        if target_route is not None:
            self.target_route = _coerce_route(target_route, self.depth, name="target_route")
        self.position = self.start
        self.steps = 0
        self._terminated = False
        self._truncated = False
        self._reached_route: tuple[int, ...] | None = None
        return self.observation()

    def step(self, action: GridAction | int) -> StepResult:
        if self.done:
            raise RuntimeError("episode is done; call reset() before step()")
        try:
            primitive = GridAction(action)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"invalid grid action {action!r}") from exc
        candidate = _advance(self.position, primitive)
        collision = candidate not in self._free_cells
        if not collision:
            self.position = candidate
        self.steps += 1

        reached = self._leaf_to_route.get(self.position)
        if reached is not None:
            self._reached_route = reached
            self._terminated = True
        elif self.steps >= self.max_steps:
            self._truncated = True
        success = reached == self.target_route
        return StepResult(
            observation=self.observation(),
            reward=float(success),
            terminated=self._terminated,
            truncated=self._truncated,
            info={
                "semantic_id": self.semantic_id,
                "reached_route": reached,
                "success": success,
                "collision": collision,
                "step": self.steps,
                "max_steps": self.max_steps,
            },
        )

    def actions_for_route(self, route: Sequence[int]) -> tuple[GridAction, ...]:
        choices = _coerce_route(route, self.depth)
        actions: list[GridAction] = []
        for level, choice in enumerate(choices):
            actions.append(GridAction.DOWN)
            horizontal = GridAction.LEFT if choice < 0 else GridAction.RIGHT
            actions.extend([horizontal] * (2 ** (self.depth - level - 1)))
            actions.extend((GridAction.DOWN, GridAction.DOWN))
        if len(actions) != self.route_length:  # pragma: no cover - formula guard
            raise AssertionError("route action construction violated the equal-length formula")
        return tuple(actions)

    def distance_to_route(
        self, route: Sequence[int], *, position: Coordinate | None = None
    ) -> int | None:
        """Shortest free-cell distance, ignoring the episode's terminal horizon."""

        target = self.route_to_leaf(route)
        start: Coordinate = self.position if position is None else position
        if start not in self._free_cells:
            raise ValueError(f"position {start} is not a free maze cell")
        queue: deque[tuple[Coordinate, int]] = deque([(start, 0)])
        visited = {start}
        while queue:
            current, distance = queue.popleft()
            if current == target:
                return distance
            for action in GridAction:
                nxt = _advance(current, action)
                if nxt in self._free_cells and nxt not in visited:
                    visited.add(nxt)
                    queue.append((nxt, distance + 1))
        return None

    def render_ascii(self) -> str:
        symbols = [["#"] * self.width for _ in range(self.height)]
        for row, column in self._free_cells:
            symbols[row][column] = "."
        for row, column in self._leaf_to_route:
            symbols[row][column] = "T"
        start_row, start_column = self.start
        symbols[start_row][start_column] = "S"
        row, column = self.position
        symbols[row][column] = "A"
        return "\n".join("".join(line) for line in symbols)


@dataclass(frozen=True)
class ForkDecision:
    """Information supplied to a learned selector at one true junction."""

    stage: int
    node_index: int
    selected_prefix: tuple[int, ...]
    observation: RouteMazeObservation


@dataclass(frozen=True)
class RouteRollout:
    intended_route: tuple[int, ...]
    selected_route: tuple[int, ...]
    reached_route: tuple[int, ...] | None
    actions: tuple[GridAction, ...]
    positions: tuple[Coordinate, ...]
    total_reward: float
    collisions: int
    terminated: bool
    truncated: bool

    @property
    def success(self) -> bool:
        return self.reached_route == self.intended_route

    @property
    def branch_accuracy(self) -> float:
        if not self.selected_route:
            return 0.0
        return sum(
            selected == intended
            for selected, intended in zip(self.selected_route, self.intended_route, strict=True)
        ) / len(self.selected_route)


RouteChoicePolicy = Callable[[ForkDecision], int]


class ScriptedCorridorController:
    """Execute forced moves while querying a policy exactly once per fork."""

    def rollout(
        self,
        env: BinaryTreeRouteMaze,
        choices: Sequence[int] | RouteChoicePolicy,
    ) -> RouteRollout:
        if callable(choices):
            policy = choices
        else:
            planned = _coerce_route(choices, env.depth, name="choices")

            def policy(decision: ForkDecision) -> int:
                return planned[decision.stage]

        env.reset()
        positions: list[Coordinate] = [env.position]
        actions: list[GridAction] = []
        selected: list[int] = []
        collisions = 0
        total_reward = 0.0
        node_index = 0

        def take(action: GridAction) -> None:
            nonlocal collisions, total_reward
            result = env.step(action)
            actions.append(action)
            positions.append(env.position)
            collisions += int(bool(result.info["collision"]))
            total_reward += result.reward

        for stage in range(env.depth):
            take(GridAction.DOWN)
            expected = env.decision_positions[(stage, node_index)]
            if env.position != expected:  # pragma: no cover - controller invariant
                raise AssertionError("scripted controller failed to reach a fork")
            decision = ForkDecision(stage, node_index, tuple(selected), env.observation())
            choice = _coerce_sign(policy(decision), name="policy route choice")
            selected.append(choice)
            node_index = 2 * node_index + int(choice > 0)
            horizontal = GridAction.LEFT if choice < 0 else GridAction.RIGHT
            for _ in range(2 ** (env.depth - stage - 1)):
                take(horizontal)
            take(GridAction.DOWN)
            take(GridAction.DOWN)

        return RouteRollout(
            intended_route=env.target_route,
            selected_route=tuple(selected),
            reached_route=env.reached_route,
            actions=tuple(actions),
            positions=tuple(positions),
            total_reward=total_reward,
            collisions=collisions,
            terminated=env.terminated,
            truncated=env.truncated,
        )


def rollout_route(
    env: BinaryTreeRouteMaze,
    choices: Sequence[int] | RouteChoicePolicy,
) -> RouteRollout:
    """Convenience wrapper around :class:`ScriptedCorridorController`."""

    return ScriptedCorridorController().rollout(env, choices)


def _readonly(array: Any, dtype: Any) -> NDArray[Any]:
    copied = np.array(array, dtype=dtype, copy=True)
    copied.setflags(write=False)
    return copied


@dataclass(frozen=True)
class RouteDecisionBatch:
    """Episode-grouped route semantics with one supervised row per fork.

    Episode-level paths are stored once, while ``x`` and ``y`` expose the
    episode-major decision rows expected by ordinary supervised trainers.
    """

    targets: SignArray
    proxy: SignArray
    proxy_present: BoolArray
    code: SignArray
    code_present: BoolArray
    stage_features: FloatArray
    episode_ids: IntArray
    depth: int
    k: int
    max_depth: int
    max_k: int
    panel: str = "iid"

    def __post_init__(self) -> None:
        if not 1 <= int(self.depth) <= int(self.max_depth) <= MAX_ROUTE_DEPTH:
            raise ValueError(
                f"depth/max_depth must satisfy 1 <= depth <= max_depth <= {MAX_ROUTE_DEPTH}"
            )
        if not 1 <= int(self.k) <= int(self.max_k):
            raise ValueError("k/max_k must satisfy 1 <= k <= max_k")
        targets = _readonly(self.targets, np.int8)
        proxy = _readonly(self.proxy, np.int8)
        proxy_present = _readonly(self.proxy_present, np.bool_)
        code = _readonly(self.code, np.int8)
        code_present = _readonly(self.code_present, np.bool_)
        stage_features = _readonly(self.stage_features, np.float32)
        episode_ids = _readonly(self.episode_ids, np.int64)
        n = targets.shape[0]
        expected_path = (n, self.max_depth)
        expected_code = (n, self.max_depth, self.max_k)
        expected_stage = (n, self.depth, self.max_depth)
        if targets.shape != expected_path or proxy.shape != expected_path:
            raise ValueError(f"targets and proxy must have shape {expected_path}")
        if proxy_present.shape != expected_path:
            raise ValueError(f"proxy_present must have shape {expected_path}")
        if code.shape != expected_code or code_present.shape != expected_code:
            raise ValueError(f"code and code_present must have shape {expected_code}")
        if stage_features.shape != expected_stage:
            raise ValueError(f"stage_features must have shape {expected_stage}")
        if episode_ids.shape != (n,) or len(np.unique(episode_ids)) != n:
            raise ValueError("episode_ids must be a unique one-dimensional ID per episode")
        if not np.all(np.isin(targets[:, : self.depth], (-1, 1))):
            raise ValueError("active target route bits must be signs")
        if np.any(targets[:, self.depth :] != 0):
            raise ValueError("inactive target route padding must be zero")
        if np.any(proxy[~proxy_present] != 0) or np.any(code[~code_present] != 0):
            raise ValueError("masked or padded signal values must be zero")
        # Zero is reserved for an explicit causal mask.  Generated, unmasked
        # datasets still contain signs at every present active coordinate.
        if not np.all(np.isin(proxy[proxy_present], (-1, 0, 1))):
            raise ValueError("present proxy values must be signs or an intervention mask")
        if not np.all(np.isin(code[code_present], (-1, 0, 1))):
            raise ValueError("present code values must be signs or an intervention mask")
        if not np.all(np.isfinite(stage_features)):
            raise ValueError("stage features must be finite")
        object.__setattr__(self, "targets", targets)
        object.__setattr__(self, "proxy", proxy)
        object.__setattr__(self, "proxy_present", proxy_present)
        object.__setattr__(self, "code", code)
        object.__setattr__(self, "code_present", code_present)
        object.__setattr__(self, "stage_features", stage_features)
        object.__setattr__(self, "episode_ids", episode_ids)

    @property
    def n_episodes(self) -> int:
        return int(self.targets.shape[0])

    def __len__(self) -> int:
        return self.n_episodes * self.depth

    @property
    def episode_id(self) -> IntArray:
        return np.repeat(self.episode_ids, self.depth)

    @property
    def step_id(self) -> IntArray:
        return np.tile(np.arange(self.depth, dtype=np.int64), self.n_episodes)

    @property
    def y(self) -> SignArray:
        active = self.targets[:, : self.depth]
        return active.reshape(-1).copy()

    @property
    def proxy_y(self) -> SignArray:
        active = self.proxy[:, : self.depth]
        return active.reshape(-1).copy()

    @property
    def input_dim(self) -> int:
        return self.max_depth * (3 + 2 * self.max_k)

    @property
    def feature_names(self) -> tuple[str, ...]:
        names = [f"P_path_{stage + 1}" for stage in range(self.max_depth)]
        names.extend(f"P_present_{stage + 1}" for stage in range(self.max_depth))
        names.extend(
            f"R_path_{stage + 1}_{channel + 1}"
            for stage in range(self.max_depth)
            for channel in range(self.max_k)
        )
        names.extend(
            f"R_present_{stage + 1}_{channel + 1}"
            for stage in range(self.max_depth)
            for channel in range(self.max_k)
        )
        names.extend(f"stage_address_{stage + 1}" for stage in range(self.max_depth))
        return tuple(names)

    def features(self) -> FloatArray:
        repeats = (1, self.depth, 1)
        proxy = np.tile(self.proxy[:, None, :], repeats)
        proxy_present = np.tile(self.proxy_present[:, None, :], repeats)
        code = np.tile(self.code.reshape(self.n_episodes, 1, -1), repeats)
        code_present = np.tile(
            self.code_present.reshape(self.n_episodes, 1, -1), repeats
        )
        features = np.concatenate(
            (proxy, proxy_present, code, code_present, self.stage_features), axis=2
        )
        return features.reshape(len(self), self.input_dim).astype(np.float32, copy=False)

    @property
    def x(self) -> FloatArray:
        return self.features()

    def with_updates(self, **changes: Any) -> RouteDecisionBatch:
        return replace(self, **changes)


@dataclass(frozen=True)
class RouteDatasetPanels:
    """Matched IID, all-conflict, and one-conflicting-stage evaluations."""

    iid: RouteDecisionBatch
    all_conflict: RouteDecisionBatch
    single_conflict: tuple[RouteDecisionBatch, ...]

    def as_dict(self) -> dict[str, RouteDecisionBatch]:
        panels = {"iid": self.iid, "all_conflict": self.all_conflict}
        panels.update(
            {f"single_conflict_{stage}": batch for stage, batch in enumerate(self.single_conflict)}
        )
        return panels


def _factorial_routes(n_episodes: int, depth: int, seed: int) -> SignArray:
    patterns = np.asarray(list(itertools.product((-1, 1), repeat=depth)), dtype=np.int8)
    if n_episodes % len(patterns) != 0:
        raise ValueError(
            f"n_episodes={n_episodes} must be divisible by 2**depth={len(patterns)} "
            "for exact factorial route balance"
        )
    routes = np.tile(patterns, (n_episodes // len(patterns), 1))
    rng = np.random.default_rng(np.random.SeedSequence([int(seed), 10]))
    return routes[rng.permutation(n_episodes)]


def _exact_proxy(targets: SignArray, q: float, seed: int) -> SignArray:
    if not 0.0 <= q <= 1.0:
        raise ValueError("q must lie in [0, 1]")
    proxy = targets.copy()
    for stage in range(targets.shape[1]):
        for sign_index, sign in enumerate((-1, 1)):
            positions = np.flatnonzero(targets[:, stage] == sign)
            requested = len(positions) * (1.0 - q)
            conflicts = round(requested)
            if abs(requested - conflicts) > 1e-9:
                raise ValueError(
                    f"q={q:g} is not exactly realizable within stage {stage}, "
                    f"target sign {sign:+d}, and stratum size {len(positions)}"
                )
            rng = np.random.default_rng(
                np.random.SeedSequence([int(seed), 20, stage, sign_index])
            )
            chosen = rng.permutation(positions)[:conflicts]
            proxy[chosen, stage] *= -1
    return proxy


def _parity_code(targets: SignArray, k: int, seed: int) -> SignArray:
    n_episodes, depth = targets.shape
    code = np.empty((n_episodes, depth, k), dtype=np.int8)
    if k == 1:
        code[:, :, 0] = targets
        return code
    rng = np.random.default_rng(np.random.SeedSequence([int(seed), 30]))
    code[:, :, : k - 1] = rng.choice(
        np.asarray((-1, 1), dtype=np.int8), size=(n_episodes, depth, k - 1)
    )
    prefix_product = np.prod(code[:, :, : k - 1], axis=2, dtype=np.int8)
    code[:, :, k - 1] = targets * prefix_product
    return code


def _route_semantics(
    n_episodes: int,
    *,
    depth: int,
    q: float,
    k: int,
    max_depth: int,
    max_k: int,
    seed: int,
) -> tuple[SignArray, SignArray, SignArray, BoolArray, BoolArray, FloatArray, IntArray]:
    if isinstance(n_episodes, bool) or not isinstance(n_episodes, Integral) or n_episodes < 1:
        raise ValueError("n_episodes must be a positive integer")
    if not 1 <= depth <= max_depth <= MAX_ROUTE_DEPTH:
        raise ValueError(
            f"depth/max_depth must satisfy 1 <= depth <= max_depth <= {MAX_ROUTE_DEPTH}"
        )
    if not 1 <= k <= max_k:
        raise ValueError("k/max_k must satisfy 1 <= k <= max_k")
    active_targets = _factorial_routes(int(n_episodes), depth, seed)
    active_proxy = _exact_proxy(active_targets, q, seed)
    active_code = _parity_code(active_targets, k, seed)
    targets = np.zeros((n_episodes, max_depth), dtype=np.int8)
    proxy = np.zeros_like(targets)
    code = np.zeros((n_episodes, max_depth, max_k), dtype=np.int8)
    proxy_present = np.zeros((n_episodes, max_depth), dtype=np.bool_)
    code_present = np.zeros((n_episodes, max_depth, max_k), dtype=np.bool_)
    targets[:, :depth] = active_targets
    proxy[:, :depth] = active_proxy
    code[:, :depth, :k] = active_code
    proxy_present[:, :depth] = True
    code_present[:, :depth, :k] = True
    stage_features = np.zeros((n_episodes, depth, max_depth), dtype=np.float32)
    for stage in range(depth):
        stage_features[:, stage, stage] = 1.0
    episode_ids = np.arange(n_episodes, dtype=np.int64)
    return (
        targets,
        proxy,
        code,
        proxy_present,
        code_present,
        stage_features,
        episode_ids,
    )


def _batch_from_arrays(
    semantics: tuple[SignArray, SignArray, SignArray, BoolArray, BoolArray, FloatArray, IntArray],
    *,
    depth: int,
    k: int,
    max_depth: int,
    max_k: int,
    panel: str,
    proxy_override: SignArray | None = None,
) -> RouteDecisionBatch:
    targets, proxy, code, proxy_present, code_present, stage_features, episode_ids = semantics
    return RouteDecisionBatch(
        targets=targets,
        proxy=proxy if proxy_override is None else proxy_override,
        proxy_present=proxy_present,
        code=code,
        code_present=code_present,
        stage_features=stage_features,
        episode_ids=episode_ids,
        depth=depth,
        k=k,
        max_depth=max_depth,
        max_k=max_k,
        panel=panel,
    )


def make_route_dataset(
    n_episodes: int,
    *,
    depth: int,
    q: float,
    k: int,
    seed: int = 0,
    max_depth: int = MAX_ROUTE_DEPTH,
    max_k: int = 5,
) -> RouteDecisionBatch:
    """Create the IID route dataset used for training or validation."""

    semantics = _route_semantics(
        n_episodes,
        depth=depth,
        q=q,
        k=k,
        max_depth=max_depth,
        max_k=max_k,
        seed=seed,
    )
    return _batch_from_arrays(
        semantics,
        depth=depth,
        k=k,
        max_depth=max_depth,
        max_k=max_k,
        panel="iid",
    )


def make_route_panels(
    n_episodes: int,
    *,
    depth: int,
    q: float,
    k: int,
    seed: int = 0,
    max_depth: int = MAX_ROUTE_DEPTH,
    max_k: int = 5,
) -> RouteDatasetPanels:
    """Create matched IID, all-conflict, and each single-conflict panel.

    Exact targets and parity channels are bit-identical across every panel.  The
    all-conflict panel sets every active proxy bit to ``-Y``.  Single-conflict
    panel ``t`` sets ``P_t=-Y_t`` and every other active proxy bit to ``Y``.
    """

    semantics = _route_semantics(
        n_episodes,
        depth=depth,
        q=q,
        k=k,
        max_depth=max_depth,
        max_k=max_k,
        seed=seed,
    )
    targets = semantics[0]
    iid = _batch_from_arrays(
        semantics,
        depth=depth,
        k=k,
        max_depth=max_depth,
        max_k=max_k,
        panel="iid",
    )
    all_proxy = np.zeros_like(targets)
    all_proxy[:, :depth] = -targets[:, :depth]
    all_conflict = _batch_from_arrays(
        semantics,
        depth=depth,
        k=k,
        max_depth=max_depth,
        max_k=max_k,
        panel="all_conflict",
        proxy_override=all_proxy,
    )
    singles: list[RouteDecisionBatch] = []
    for conflict_stage in range(depth):
        proxy = np.zeros_like(targets)
        proxy[:, :depth] = targets[:, :depth]
        proxy[:, conflict_stage] *= -1
        singles.append(
            _batch_from_arrays(
                semantics,
                depth=depth,
                k=k,
                max_depth=max_depth,
                max_k=max_k,
                panel=f"single_conflict_{conflict_stage}",
                proxy_override=proxy,
            )
        )
    return RouteDatasetPanels(iid=iid, all_conflict=all_conflict, single_conflict=tuple(singles))


def _active_indices(values: Iterable[int] | None, size: int, *, name: str) -> tuple[int, ...]:
    if values is None:
        return tuple(range(size))
    normalized = tuple(int(value) for value in values)
    if len(set(normalized)) != len(normalized) or any(not 0 <= value < size for value in normalized):
        raise ValueError(f"{name} must contain unique indices in [0, {size})")
    return normalized


def flip_route_proxy(
    batch: RouteDecisionBatch, stages: Iterable[int] | None = None
) -> RouteDecisionBatch:
    selected = _active_indices(stages, batch.depth, name="proxy stages")
    proxy = np.array(batch.proxy, copy=True)
    proxy[:, selected] *= -1
    return batch.with_updates(proxy=proxy)


def mask_route_proxy(
    batch: RouteDecisionBatch,
    stages: Iterable[int] | None = None,
    *,
    hide_presence: bool = True,
) -> RouteDecisionBatch:
    selected = _active_indices(stages, batch.depth, name="proxy stages")
    proxy = np.array(batch.proxy, copy=True)
    present = np.array(batch.proxy_present, copy=True)
    proxy[:, selected] = 0
    if hide_presence:
        present[:, selected] = False
    return batch.with_updates(proxy=proxy, proxy_present=present)


def flip_route_code(
    batch: RouteDecisionBatch,
    *,
    stages: Iterable[int] | None = None,
    channels: Iterable[int] = (0,),
) -> RouteDecisionBatch:
    selected_stages = _active_indices(stages, batch.depth, name="code stages")
    selected_channels = _active_indices(channels, batch.k, name="code channels")
    code = np.array(batch.code, copy=True)
    for stage in selected_stages:
        code[:, stage, selected_channels] *= -1
    return batch.with_updates(code=code)


def mask_route_code(
    batch: RouteDecisionBatch,
    *,
    stages: Iterable[int] | None = None,
    channels: Iterable[int] | None = None,
    hide_presence: bool = True,
) -> RouteDecisionBatch:
    selected_stages = _active_indices(stages, batch.depth, name="code stages")
    selected_channels = _active_indices(channels, batch.k, name="code channels")
    code = np.array(batch.code, copy=True)
    present = np.array(batch.code_present, copy=True)
    for stage in selected_stages:
        code[:, stage, selected_channels] = 0
        if hide_presence:
            present[:, stage, selected_channels] = False
    return batch.with_updates(code=code, code_present=present)


def permute_route_addresses(
    batch: RouteDecisionBatch, permutation: Sequence[int]
) -> RouteDecisionBatch:
    normalized = tuple(int(value) for value in permutation)
    if sorted(normalized) != list(range(batch.depth)):
        raise ValueError("address permutation must contain every active stage exactly once")
    stage_features = np.zeros_like(batch.stage_features)
    for stage, advertised_stage in enumerate(normalized):
        stage_features[:, stage, advertised_stage] = 1.0
    return batch.with_updates(stage_features=stage_features)


def intervene_route_batch(
    batch: RouteDecisionBatch,
    *,
    flip_proxy_stages: Iterable[int] = (),
    flip_code_coordinates: Iterable[tuple[int, int]] = (),
    mask_proxy_stages: Iterable[int] = (),
    mask_code_coordinates: Iterable[tuple[int, int]] = (),
) -> RouteDecisionBatch:
    """Apply an explicit collection of paired causal signal interventions."""

    changed = batch
    proxy_flips = tuple(flip_proxy_stages)
    if proxy_flips:
        changed = flip_route_proxy(changed, proxy_flips)
    code_flips = tuple(flip_code_coordinates)
    if code_flips:
        code = np.array(changed.code, copy=True)
        for stage, channel in code_flips:
            if not 0 <= stage < batch.depth or not 0 <= channel < batch.k:
                raise ValueError("flip_code_coordinates contains an inactive coordinate")
            code[:, stage, channel] *= -1
        changed = changed.with_updates(code=code)
    proxy_masks = tuple(mask_proxy_stages)
    if proxy_masks:
        changed = mask_route_proxy(changed, proxy_masks)
    code_masks = tuple(mask_code_coordinates)
    if code_masks:
        code = np.array(changed.code, copy=True)
        present = np.array(changed.code_present, copy=True)
        for stage, channel in code_masks:
            if not 0 <= stage < batch.depth or not 0 <= channel < batch.k:
                raise ValueError("mask_code_coordinates contains an inactive coordinate")
            code[:, stage, channel] = 0
            present[:, stage, channel] = False
        changed = changed.with_updates(code=code, code_present=present)
    return changed


def route_metrics(
    predictions: Sequence[float] | NDArray[Any], batch: RouteDecisionBatch
) -> dict[str, Any]:
    """Aggregate decision predictions into branch and whole-route outcomes."""

    raw = np.asarray(predictions)
    if raw.shape == (batch.n_episodes, batch.depth):
        raw = raw.reshape(-1)
    if raw.shape != (len(batch),):
        raise ValueError(
            f"predictions must have shape ({len(batch)},) or "
            f"({batch.n_episodes}, {batch.depth})"
        )
    predicted = np.where(raw >= 0.0, 1, -1).astype(np.int8).reshape(
        batch.n_episodes, batch.depth
    )
    target = batch.targets[:, : batch.depth]
    proxy = batch.proxy[:, : batch.depth]
    correct = predicted == target
    stage_accuracy = np.mean(correct, axis=0)
    route_success = np.all(correct, axis=1)
    branch_accuracy = float(np.mean(correct))
    full_route_success = float(np.mean(route_success))
    independent = float(np.prod(stage_accuracy))
    first_divergence = np.full(batch.n_episodes, batch.depth, dtype=np.int64)
    for stage in range(batch.depth):
        newly_wrong = (first_divergence == batch.depth) & ~correct[:, stage]
        first_divergence[newly_wrong] = stage
    first_rates = [
        float(np.mean(first_divergence == stage)) for stage in range(batch.depth)
    ]
    first_rates.append(float(np.mean(first_divergence == batch.depth)))
    return {
        "n_episodes": batch.n_episodes,
        "n_branches": len(batch),
        "depth": batch.depth,
        "branch_accuracy": branch_accuracy,
        "stage_branch_accuracy": [float(value) for value in stage_accuracy],
        "full_route_success": full_route_success,
        "effective_per_fork_success": full_route_success ** (1.0 / batch.depth),
        "independent_compounding_prediction": independent,
        "pooled_independent_prediction": branch_accuracy**batch.depth,
        "compounding_gap": full_route_success - independent,
        "proxy_branch_agreement": float(np.mean(predicted == proxy)),
        "proxy_full_route_agreement": float(np.mean(np.all(predicted == proxy, axis=1))),
        "first_divergence_rate": first_rates,
        "first_divergence_mean": float(np.mean(first_divergence)),
    }


__all__ = [
    "MAX_ROUTE_DEPTH",
    "BinaryTreeRouteMaze",
    "ForkDecision",
    "RouteDatasetPanels",
    "RouteDecisionBatch",
    "RouteMazeObservation",
    "RouteRollout",
    "ScriptedCorridorController",
    "flip_route_code",
    "flip_route_proxy",
    "intervene_route_batch",
    "make_route_dataset",
    "make_route_panels",
    "mask_route_code",
    "mask_route_proxy",
    "permute_route_addresses",
    "rollout_route",
    "route_metrics",
]
