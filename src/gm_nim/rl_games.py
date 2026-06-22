from __future__ import annotations

import random
from dataclasses import dataclass
from math import log2
from typing import Any

from .games import (
    bounded_nim_action,
    fibonacci_nim_action,
    multipile_nim_action,
    wythoff_cold_positions,
    wythoff_nim_action,
)
from .metrics import parse_bounded_move, parse_multipile_move, parse_pair


Action = int | tuple[int, int]
State = dict[str, Any]


@dataclass(frozen=True)
class GameConfig:
    game: str = "bounded"
    mr: int = 5
    min_pile: int = 20
    max_pile: int = 400
    pile_count: int = 3
    min_heap: int = 1
    max_heap: int = 120
    fib_min_pile: int = 8
    fib_max_pile: int = 160
    wythoff_max_heap: int = 120


def initial_state(config: GameConfig, rng: random.Random) -> State:
    if config.game == "bounded":
        return {"game": "bounded", "pile": rng.randint(config.min_pile, config.max_pile), "mr": config.mr}
    if config.game == "multipile":
        return {
            "game": "multipile",
            "piles": tuple(rng.randint(config.min_heap, config.max_heap) for _ in range(config.pile_count)),
        }
    if config.game == "fibonacci":
        pile = rng.randint(config.fib_min_pile, config.fib_max_pile)
        return {"game": "fibonacci", "pile": pile, "limit": pile - 1, "first_move": True}
    if config.game == "wythoff":
        return {
            "game": "wythoff",
            "piles": (
                rng.randint(0, config.wythoff_max_heap),
                rng.randint(0, config.wythoff_max_heap),
            ),
        }
    raise ValueError(f"unknown game: {config.game}")


def is_terminal(state: State) -> bool:
    game = state["game"]
    if game in {"bounded", "fibonacci"}:
        return int(state["pile"]) <= 0
    if game in {"multipile", "wythoff"}:
        return all(pile == 0 for pile in state["piles"])
    raise ValueError(f"unknown game: {game}")


def legal_actions(state: State) -> list[Action]:
    game = state["game"]
    if is_terminal(state):
        return []
    if game == "bounded":
        return list(range(1, min(int(state["mr"]), int(state["pile"])) + 1))
    if game == "fibonacci":
        return list(range(1, min(int(state["limit"]), int(state["pile"])) + 1))
    if game == "multipile":
        actions: list[Action] = []
        for index, pile in enumerate(state["piles"], start=1):
            actions.extend((index, amount) for amount in range(1, pile + 1))
        return actions
    if game == "wythoff":
        a, b = state["piles"]
        actions = []
        for next_a in range(a):
            actions.append((next_a, b))
        for next_b in range(b):
            actions.append((a, next_b))
        for amount in range(1, min(a, b) + 1):
            actions.append((a - amount, b - amount))
        return actions
    raise ValueError(f"unknown game: {game}")


def is_legal_action(state: State, action: Action | None) -> bool:
    if action is None:
        return False
    return action in set(legal_actions(state))


def apply_action(state: State, action: Action) -> State:
    if not is_legal_action(state, action):
        raise ValueError(f"illegal action {action!r} for state {state!r}")
    game = state["game"]
    if game == "bounded":
        return {"game": "bounded", "pile": int(state["pile"]) - int(action), "mr": int(state["mr"])}
    if game == "fibonacci":
        take = int(action)
        return {
            "game": "fibonacci",
            "pile": int(state["pile"]) - take,
            "limit": max(1, 2 * take),
            "first_move": False,
        }
    if game == "multipile":
        pile_index, amount = action
        piles = list(state["piles"])
        piles[pile_index - 1] -= amount
        return {"game": "multipile", "piles": tuple(piles)}
    if game == "wythoff":
        return {"game": "wythoff", "piles": tuple(action)}
    raise ValueError(f"unknown game: {game}")


def render_prompt(state: State) -> str:
    game = state["game"]
    if game == "bounded":
        max_take = min(int(state["mr"]), int(state["pile"]))
        return "\n".join(
            [
                "You are playing single-pile bounded Nim.",
                f"There are {state['pile']} coins left.",
                f"You may take between 1 and {max_take} coins.",
                "What is your move? Answer as: take k coins.",
            ]
        )
    if game == "fibonacci":
        max_take = min(int(state["limit"]), int(state["pile"]))
        return "\n".join(
            [
                "You are playing Fibonacci Nim.",
                f"There are {state['pile']} coins left.",
                f"You may take between 1 and {max_take} coins.",
                "What is your move? Answer as: take k coins.",
            ]
        )
    if game == "multipile":
        piles = ", ".join(str(pile) for pile in state["piles"])
        return "\n".join(
            [
                "You are playing the game of Nim with multiple piles.",
                f"The piles currently contain {piles} coins.",
                "You may remove any positive number of coins from a single pile.",
                "What is your move? Answer as: take k from pile i.",
            ]
        )
    if game == "wythoff":
        a, b = state["piles"]
        return "\n".join(
            [
                "You are playing Wythoff Nim with two piles.",
                f"The piles contain {a} and {b} coins.",
                "You may remove any positive number from one pile, or the same positive number from both piles.",
                "What successor state do you choose? Answer as: (a,b).",
            ]
        )
    raise ValueError(f"unknown game: {game}")


def parse_action(game: str, text: str) -> Action | None:
    if game in {"bounded", "fibonacci"}:
        return parse_bounded_move(text)
    if game == "multipile":
        return parse_multipile_move(text)
    if game == "wythoff":
        return parse_pair(text)
    raise ValueError(f"unknown game: {game}")


def format_action(game: str, action: Action | None) -> str:
    if action is None:
        return "resign"
    if game in {"bounded", "fibonacci"}:
        return f"take {action} coins"
    if game == "multipile":
        pile_index, amount = action
        return f"take {amount} from pile {pile_index}"
    if game == "wythoff":
        a, b = action
        return f"({a},{b})"
    raise ValueError(f"unknown game: {game}")


def optimal_action(state: State, rng: random.Random) -> Action:
    game = state["game"]
    legal = legal_actions(state)
    if not legal:
        raise ValueError("terminal state has no optimal action")
    if game == "bounded":
        action = bounded_nim_action(int(state["pile"]), int(state["mr"]))
        return rng.choice(legal) if action == -1 else action
    if game == "fibonacci":
        action = fibonacci_nim_action(int(state["pile"]), int(state["limit"]))
        return rng.choice(legal) if action == -1 else action
    if game == "multipile":
        action = multipile_nim_action(tuple(state["piles"]))
        return rng.choice(legal) if action is None else action
    if game == "wythoff":
        action = wythoff_nim_action(tuple(state["piles"]))
        return rng.choice(legal) if action is None else action
    raise ValueError(f"unknown game: {game}")


def has_winning_action(state: State) -> bool:
    game = state["game"]
    if game == "bounded":
        return bounded_nim_action(int(state["pile"]), int(state["mr"])) != -1
    if game == "fibonacci":
        return fibonacci_nim_action(int(state["pile"]), int(state["limit"])) != -1
    if game == "multipile":
        return multipile_nim_action(tuple(state["piles"])) is not None
    if game == "wythoff":
        return wythoff_nim_action(tuple(state["piles"])) is not None
    raise ValueError(f"unknown game: {game}")


def random_action(state: State, rng: random.Random) -> Action:
    return rng.choice(legal_actions(state))


def bounded_coset_action(state: State, factor: int, rng: random.Random) -> Action:
    legal = [int(action) for action in legal_actions(state)]
    optimal = bounded_nim_action(int(state["pile"]), int(state["mr"]))
    if optimal == -1:
        return rng.choice(legal)
    target = optimal % factor
    candidates = [action for action in legal if action % factor == target]
    return min(candidates) if candidates else rng.choice(legal)


def multipile_lowbit_action(state: State, bits: int, rng: random.Random) -> Action:
    modulus = 2**bits
    candidates = []
    for action in legal_actions(state):
        next_state = apply_action(state, action)
        xor_value = 0
        for pile in next_state["piles"]:
            xor_value ^= pile
        if xor_value % modulus == 0:
            candidates.append(action)
    return rng.choice(candidates) if candidates else random_action(state, rng)


def fibonacci_floor_action(state: State, rng: random.Random) -> Action:
    legal = [int(action) for action in legal_actions(state)]
    pile = int(state["pile"])
    fibs = [1, 2]
    while fibs[-1] < pile:
        fibs.append(fibs[-1] + fibs[-2])
    lower = max(fib for fib in fibs if fib < pile)
    take = pile - lower
    return take if take in legal else rng.choice(legal)


def wythoff_balance_action(state: State, rng: random.Random) -> Action:
    a, b = state["piles"]
    if a == b:
        return random_action(state, rng)
    low = min(a, b)
    candidate = (low, low)
    return candidate if is_legal_action(state, candidate) else random_action(state, rng)


def wythoff_difference_action(state: State, rng: random.Random) -> Action:
    a, b = state["piles"]
    if a == b:
        return random_action(state, rng)
    diff = abs(a - b)
    high_index = 0 if a > b else 1
    piles = [a, b]
    piles[high_index] = max(0, piles[high_index] - max(1, diff // 2))
    candidate = tuple(piles)
    return candidate if is_legal_action(state, candidate) else random_action(state, rng)


def fixed_policy_action(policy: str, state: State, rng: random.Random) -> Action:
    policy = policy.lower()
    if policy in {"random", "weak"}:
        return random_action(state, rng)
    if policy == "optimal":
        return optimal_action(state, rng)
    if policy.startswith("coset"):
        if state["game"] != "bounded":
            raise ValueError("coset policies are defined for bounded Nim")
        return bounded_coset_action(state, int(policy.removeprefix("coset")), rng)
    if policy.startswith("xor"):
        if state["game"] != "multipile":
            raise ValueError("xor proxy policies are defined for multipile Nim")
        value = int(policy.removeprefix("xor"))
        bits = int(log2(value)) if value > 1 else 1
        return multipile_lowbit_action(state, bits, rng)
    if policy in {"fib_floor", "fibonacci_floor"}:
        if state["game"] != "fibonacci":
            raise ValueError("fibonacci_floor policy is defined for Fibonacci Nim")
        return fibonacci_floor_action(state, rng)
    if policy == "balance":
        if state["game"] != "wythoff":
            raise ValueError("balance policy is defined for Wythoff Nim")
        return wythoff_balance_action(state, rng)
    if policy in {"difference", "diff"}:
        if state["game"] != "wythoff":
            raise ValueError("difference policy is defined for Wythoff Nim")
        return wythoff_difference_action(state, rng)
    raise ValueError(f"unknown fixed policy: {policy}")


def action_matches_optimal_proxy(state: State, action: Action, proxy: str) -> bool:
    if not is_legal_action(state, action):
        return False
    game = state["game"]
    proxy = proxy.lower()
    if proxy == "optimal":
        if not has_winning_action(state):
            return False
        return action == optimal_action(state, random.Random(0))
    if proxy.startswith("coset") and game == "bounded":
        factor = int(proxy.removeprefix("coset"))
        optimal = bounded_nim_action(int(state["pile"]), int(state["mr"]))
        if optimal == -1:
            return False
        return int(action) % factor == optimal % factor
    if proxy.startswith("xor") and game == "multipile":
        value = int(proxy.removeprefix("xor"))
        next_state = apply_action(state, action)
        xor_value = 0
        for pile in next_state["piles"]:
            xor_value ^= pile
        return xor_value % value == 0
    if proxy == "balance" and game == "wythoff":
        a, b = apply_action(state, action)["piles"]
        return a == b
    if proxy in {"difference", "diff"} and game == "wythoff":
        before = abs(state["piles"][0] - state["piles"][1])
        after = abs(apply_action(state, action)["piles"][0] - apply_action(state, action)["piles"][1])
        return after < before
    if proxy in {"fib_floor", "fibonacci_floor"} and game == "fibonacci":
        return action == fibonacci_floor_action(state, random.Random(0))
    return False


def cold_state_proxy_names(game: str) -> list[str]:
    if game == "bounded":
        return ["coset2", "coset3", "coset4"]
    if game == "multipile":
        return ["xor2", "xor4"]
    if game == "fibonacci":
        return ["fibonacci_floor"]
    if game == "wythoff":
        return ["balance", "difference"]
    return []


def wythoff_is_cold(state: State) -> bool:
    a, b = state["piles"]
    return (a, b) in wythoff_cold_positions(max(a, b))
