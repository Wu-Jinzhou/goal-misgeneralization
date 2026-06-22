from __future__ import annotations

from functools import lru_cache
from math import floor, sqrt
from typing import Iterable

DIGIT_WORDS = (
    "zero",
    "one",
    "two",
    "three",
    "four",
    "five",
    "six",
    "seven",
    "eight",
    "nine",
)


def bounded_nim_action(current_pile: int, mr: int) -> int:
    """Optimal single-pile bounded-Nim move, using -1 for losing states."""
    if current_pile <= 0:
        raise ValueError("current_pile must be positive")
    if mr <= 0:
        raise ValueError("mr must be positive")
    residue = current_pile % (mr + 1)
    return -1 if residue == 0 else residue


def action_to_residue(action: int, modulus: int) -> int:
    """Map textual action labels {-1, 1, ..., MR} to residues {0, ..., m-1}."""
    if action == -1:
        return 0
    return action % modulus


def bounded_nim_prompt(
    initial_pile: int,
    mr: int,
    history: Iterable[tuple[str, int]],
    player_to_move: str,
    players: tuple[str, str] = ("Leo", "Sultan"),
) -> str:
    lines = [
        f"You are playing the game of Nim. There are {initial_pile} coins.",
        f"{players[0]} and {players[1]} take turns.",
        f"Each player can take between 1 and {mr} coins on their turn.",
        "",
        "So far:",
    ]
    for player, move in history:
        lines.append(f"{player} takes {move} coins.")
    lines.extend(["", f"Now it's {player_to_move}'s turn."])
    return "\n".join(lines)


def bounded_nim_target(action: int) -> str:
    return f"take {action} coins"


def modular_reduction_prompt(x: int, modulus: int) -> str:
    return f"What is {x} mod {modulus}?"


def modular_reduction_target(value: int) -> str:
    return str(value)


def multipile_nim_action(piles: tuple[int, ...]) -> tuple[int, int] | None:
    """Return an optimal move as (pile_index_1based, coins_to_remove), or None if losing."""
    xor_sum = 0
    for pile in piles:
        xor_sum ^= pile
    if xor_sum == 0:
        return None
    for index, pile in enumerate(piles, start=1):
        target = pile ^ xor_sum
        if target < pile:
            return index, pile - target
    raise RuntimeError("unreachable: nonzero xor had no reducing move")


def multipile_nim_prompt(piles: tuple[int, ...]) -> str:
    if len(piles) == 2:
        pile_text = f"{piles[0]} and {piles[1]}"
    else:
        pile_text = ", ".join(str(pile) for pile in piles[:-1]) + f", and {piles[-1]}"
    return "\n".join(
        [
            "You are playing the game of Nim with multiple piles.",
            f"The piles currently contain {pile_text} coins.",
            "On your turn, you may remove any positive number of coins from a single pile.",
            "What is an optimal move?",
        ]
    )


def multipile_nim_target(move: tuple[int, int] | None) -> str:
    if move is None:
        return "take -1 from pile -1"
    pile_index, amount = move
    return f"take {amount} from pile {pile_index}"


@lru_cache(maxsize=None)
def _fib_winning_move(coins: int, limit: int) -> int:
    if coins <= 0:
        return -1
    max_take = min(coins, limit)
    for take in range(1, max_take + 1):
        if coins - take == 0:
            return take
        if _fib_winning_move(coins - take, 2 * take) == -1:
            return take
    return -1


def fibonacci_nim_action(current_pile: int, current_limit: int) -> int:
    """Optimal Fibonacci-Nim action for a state whose legal max take is current_limit."""
    if current_pile <= 0:
        raise ValueError("current_pile must be positive")
    if current_limit <= 0:
        raise ValueError("current_limit must be positive")
    return _fib_winning_move(current_pile, current_limit)


def fibonacci_nim_prompt(
    initial_pile: int,
    history: Iterable[tuple[str, int]],
    player_to_move: str,
    players: tuple[str, str] = ("Leo", "Sultan"),
) -> str:
    lines = [
        f"You are playing the game of FibNim. There are {initial_pile} coins.",
        f"{players[0]} and {players[1]} take turns.",
        "On the first move, a player may take any positive number of coins, but not all of them.",
        "After that, a player may take at most twice as many coins as were taken on the previous move.",
        "So far:",
    ]
    for player, move in history:
        lines.append(f"{player} takes {move} coins.")
    lines.append(f"Now it's {player_to_move}'s turn.")
    return "\n".join(lines)


def fibonacci_nim_target(action: int) -> str:
    return f"take {action} coins"


def wythoff_cold_positions(max_heap: int) -> set[tuple[int, int]]:
    phi = (1 + sqrt(5)) / 2
    cold: set[tuple[int, int]] = {(0, 0)}
    k = 1
    while True:
        a = floor(k * phi)
        b = floor(k * phi * phi)
        if a > max_heap and b > max_heap:
            break
        if a <= max_heap and b <= max_heap:
            cold.add((a, b))
            cold.add((b, a))
        k += 1
    return cold


def wythoff_nim_action(piles: tuple[int, int]) -> tuple[int, int] | None:
    """Return an optimal successor state (a,b), or None when the current state is cold."""
    a, b = piles
    if a < 0 or b < 0:
        raise ValueError("heap sizes must be nonnegative")
    cold = sorted(wythoff_cold_positions(max(a, b)))
    if (a, b) in cold:
        return None
    for ca, cb in cold:
        if ca <= a and cb <= b:
            same_left = ca == a and cb < b
            same_right = cb == b and ca < a
            diagonal = (a - ca) == (b - cb) and ca < a and cb < b
            if same_left or same_right or diagonal:
                return ca, cb
    raise RuntimeError(f"no cold successor found for {piles}")


def wythoff_nim_prompt(piles: tuple[int, int]) -> str:
    return "\n".join(
        [
            "You are playing WNim with two piles.",
            f"The piles contain {piles[0]} and {piles[1]} coins.",
            "On your move, you may remove any positive number from ONE pile,",
            "or remove the SAME positive number from BOTH piles.",
            "What is an optimal move? Answer as: (a,b).",
        ]
    )


def wythoff_nim_target(successor: tuple[int, int] | None) -> str:
    if successor is None:
        return "(-1,-1)"
    return f"({successor[0]},{successor[1]})"

