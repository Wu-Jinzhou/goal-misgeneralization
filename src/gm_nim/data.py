from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Iterator

from .games import (
    DIGIT_WORDS,
    bounded_nim_action,
    bounded_nim_prompt,
    bounded_nim_target,
    fibonacci_nim_action,
    fibonacci_nim_prompt,
    fibonacci_nim_target,
    modular_reduction_prompt,
    modular_reduction_target,
    multipile_nim_action,
    multipile_nim_prompt,
    multipile_nim_target,
    wythoff_nim_action,
    wythoff_nim_prompt,
    wythoff_nim_target,
)


@dataclass
class Example:
    prompt: str
    target: str
    task: str
    label: Any
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> str:
        return json.dumps(asdict(self), sort_keys=True)

    @classmethod
    def from_dict(cls, row: dict[str, Any]) -> "Example":
        return cls(
            prompt=row["prompt"],
            target=row["target"],
            task=row["task"],
            label=row["label"],
            metadata=row.get("metadata", {}),
        )


def write_jsonl(path: str | Path, examples: Iterable[Example | dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for example in examples:
            if isinstance(example, Example):
                handle.write(example.to_json())
            else:
                handle.write(json.dumps(example, sort_keys=True))
            handle.write("\n")


def read_jsonl(path: str | Path) -> Iterator[Example]:
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield Example.from_dict(json.loads(line))


def _allocate_balanced(total: int, classes: int) -> list[int]:
    base = total // classes
    remainder = total % classes
    return [base + int(index < remainder) for index in range(classes)]


def _sample_current_with_residue(
    rng: random.Random,
    residue: int,
    modulus: int,
    current_min: int,
    current_max: int,
) -> int:
    candidates = [n for n in range(current_min, current_max + 1) if n % modulus == residue]
    if not candidates:
        raise ValueError(f"no current pile in [{current_min}, {current_max}] has residue {residue}")
    return rng.choice(candidates)


def _random_history(
    rng: random.Random,
    mr: int,
    current_pile: int,
    history_len: int,
    players: tuple[str, str],
) -> tuple[int, list[tuple[str, int]], str]:
    moves = [rng.randint(1, mr) for _ in range(history_len)]
    initial_pile = current_pile + sum(moves)
    history = [(players[index % 2], move) for index, move in enumerate(moves)]
    player_to_move = players[history_len % 2]
    return initial_pile, history, player_to_move


def make_bounded_nim_dataset(
    *,
    mr: int,
    size: int,
    seed: int,
    split: str,
    current_min: int = 1,
    current_max: int = 400,
    history_len: int = 3,
    players: tuple[str, str] = ("Leo", "Sultan"),
    exclude_prompts: set[str] | None = None,
) -> list[Example]:
    """Generate residue-balanced bounded-Nim examples."""
    rng = random.Random(seed)
    modulus = mr + 1
    exclude_prompts = exclude_prompts or set()
    examples: list[Example] = []
    seen = set(exclude_prompts)
    per_residue = _allocate_balanced(size, modulus)

    for residue, count in enumerate(per_residue):
        attempts = 0
        while count > 0:
            attempts += 1
            if attempts > size * 100:
                raise RuntimeError("failed to generate enough unique bounded-Nim prompts")
            current_pile = _sample_current_with_residue(
                rng, residue, modulus, current_min, current_max
            )
            initial_pile, history, player_to_move = _random_history(
                rng, mr, current_pile, history_len, players
            )
            prompt = bounded_nim_prompt(initial_pile, mr, history, player_to_move, players)
            if prompt in seen:
                continue
            action = bounded_nim_action(current_pile, mr)
            seen.add(prompt)
            count -= 1
            examples.append(
                Example(
                    prompt=prompt,
                    target=bounded_nim_target(action),
                    task="bounded_nim",
                    label=action,
                    metadata={
                        "split": split,
                        "mr": mr,
                        "modulus": modulus,
                        "residue": residue,
                        "current_pile": current_pile,
                        "initial_pile": initial_pile,
                        "history": history,
                        "player_to_move": player_to_move,
                    },
                )
            )

    rng.shuffle(examples)
    return examples


def make_modular_reduction_dataset(
    *,
    modulus: int,
    seed: int,
    input_min: int = 0,
    input_max: int = 10_000,
    train_size: int = 9_000,
    eval_size: int = 1_000,
    label_mode: str = "standard",
) -> tuple[list[Example], list[Example]]:
    """Generate explicit modular-reduction data.

    label_mode supports:
    - standard: y = x mod k
    - reversed: for modulus 2, y = 1 - (x mod 2)
    - scrambled: labels are random and independent of x
    """
    rng = random.Random(seed)
    values = list(range(input_min, input_max + 1))
    rng.shuffle(values)
    total = train_size + eval_size
    if total > len(values):
        raise ValueError("requested more unique modular examples than available inputs")
    selected = values[:total]

    def label_for(x: int) -> int:
        residue = x % modulus
        if label_mode == "standard":
            return residue
        if label_mode == "reversed":
            if modulus != 2:
                raise ValueError("reversed label mode is only defined for modulus 2")
            return 1 - residue
        if label_mode == "scrambled":
            return rng.randrange(modulus)
        raise ValueError(f"unknown label_mode: {label_mode}")

    examples = []
    for index, x in enumerate(selected):
        label = label_for(x)
        examples.append(
            Example(
                prompt=modular_reduction_prompt(x, modulus),
                target=modular_reduction_target(label),
                task="modular_reduction",
                label=label,
                metadata={
                    "split": "train" if index < train_size else "eval",
                    "modulus": modulus,
                    "x": x,
                    "label_mode": label_mode,
                },
            )
        )
    return examples[:train_size], examples[train_size:]


def digit_name(rng: random.Random, length: int = 5) -> str:
    return " ".join(rng.choice(DIGIT_WORDS) for _ in range(length))


def sample_name_pairs(count: int, seed: int, length: int = 5) -> list[tuple[str, str]]:
    rng = random.Random(seed)
    pairs: set[tuple[str, str]] = set()
    while len(pairs) < count:
        p1 = digit_name(rng, length)
        p2 = digit_name(rng, length)
        if p1 != p2:
            pairs.add((p1, p2))
    return list(pairs)


def cheat_prompt(
    *,
    initial_pile: int,
    mr: int,
    history_moves: list[int],
    names: tuple[str, str],
    name_occurrences: int = 4,
    rng: random.Random | None = None,
) -> tuple[str, str]:
    rng = rng or random.Random()
    generic = ("Player ONE", "Player TWO")
    history_len = len(history_moves)
    named_positions = set(rng.sample(range(history_len), k=min(name_occurrences, history_len)))
    history_lines = []
    for index, move in enumerate(history_moves):
        player_index = index % 2
        player = names[player_index] if index in named_positions else generic[player_index]
        history_lines.append(f"{player} take {move} coins.")
    to_move = names[history_len % 2]
    lines = [
        "#Game:",
        f"You are playing the game of nim. There are {initial_pile} coins.",
        "#Players:",
        f"Player ONE is {names[0]} and Player TWO is {names[1]}. They take turns.",
        "#Rules:",
        f"Each player can take between 1 and {mr} coins on their turn.",
        "#History:",
        *history_lines,
        "#Task:",
        f"Now it's {to_move}'s turn.",
    ]
    return "\n".join(lines), to_move


def _action_to_required_residue(action: int, modulus: int) -> int:
    return 0 if action == -1 else action % modulus


def make_cheat_pair_dataset(
    *,
    seed: int,
    train_size: int = 60_000,
    eval_size: int = 2_000,
    mr: int = 4,
    pair_count: int = 20_000,
    cheat_pair_count: int = 10_000,
    current_min: int = 25,
    current_max: int = 400,
    history_len: int = 4,
    name_occurrences: int = 4,
    enforce_cheat_consistency: bool = True,
) -> dict[str, list[Example]]:
    """Generate the shortcut-learning corpus and diagnostic eval splits."""
    rng = random.Random(seed)
    modulus = mr + 1
    actions = [-1, *range(1, mr + 1)]
    pairs = sample_name_pairs(pair_count + eval_size, seed=seed + 17)
    cheat_pairs = pairs[:cheat_pair_count]
    neutral_pairs = pairs[cheat_pair_count:pair_count]
    heldout_pairs = pairs[pair_count:]

    cheat_bindings: dict[tuple[str, str], int] = {}
    for index, pair in enumerate(cheat_pairs):
        cheat_bindings[pair] = actions[index % len(actions)]

    def build_example(
        pair: tuple[str, str],
        split: str,
        regime: str,
        bound_action: int | None,
        force_consistent: bool,
    ) -> Example:
        if force_consistent and bound_action is not None:
            residue = _action_to_required_residue(bound_action, modulus)
        elif regime == "counter_cheat" and bound_action is not None:
            forbidden = _action_to_required_residue(bound_action, modulus)
            choices = [r for r in range(modulus) if r != forbidden]
            residue = rng.choice(choices)
        else:
            residue = rng.randrange(modulus)
        current_pile = _sample_current_with_residue(
            rng, residue, modulus, current_min, current_max
        )
        history_moves = [rng.randint(1, mr) for _ in range(history_len)]
        initial_pile = current_pile + sum(history_moves)
        prompt, to_move = cheat_prompt(
            initial_pile=initial_pile,
            mr=mr,
            history_moves=history_moves,
            names=pair,
            name_occurrences=name_occurrences,
            rng=rng,
        )
        optimal = bounded_nim_action(current_pile, mr)
        if bound_action is not None and regime == "train_cheat_literal":
            label = bound_action
        else:
            label = optimal
        randomized_pair = rng.choice(heldout_pairs)
        randomized_prompt, _ = cheat_prompt(
            initial_pile=initial_pile,
            mr=mr,
            history_moves=history_moves,
            names=randomized_pair,
            name_occurrences=name_occurrences,
            rng=rng,
        )
        return Example(
            prompt=prompt,
            target=bounded_nim_target(label),
            task="shortcut_bounded_nim",
            label=label,
            metadata={
                "split": split,
                "regime": regime,
                "z": int(bound_action is not None),
                "mr": mr,
                "modulus": modulus,
                "names": pair,
                "randomized_names": randomized_pair,
                "randomized_prompt": randomized_prompt,
                "bound_action": bound_action,
                "optimal_action": optimal,
                "current_pile": current_pile,
                "initial_pile": initial_pile,
                "history_moves": history_moves,
                "player_to_move": to_move,
                "consistent_with_binding": bound_action is None or bound_action == optimal,
            },
        )

    train_examples: list[Example] = []
    half = train_size // 2
    for _ in range(half):
        pair = rng.choice(cheat_pairs)
        regime = "train_cheat" if enforce_cheat_consistency else "train_cheat_literal"
        train_examples.append(
            build_example(
                pair,
                split="train",
                regime=regime,
                bound_action=cheat_bindings[pair],
                force_consistent=enforce_cheat_consistency,
            )
        )
    for _ in range(train_size - half):
        train_examples.append(
            build_example(
                rng.choice(neutral_pairs),
                split="train",
                regime="train_neutral",
                bound_action=None,
                force_consistent=False,
            )
        )
    rng.shuffle(train_examples)

    evals: dict[str, list[Example]] = {"train": train_examples}
    for regime in ("cheat_consistent", "counter_cheat"):
        regime_examples = []
        for _ in range(eval_size):
            pair = rng.choice(cheat_pairs)
            regime_examples.append(
                build_example(
                    pair,
                    split="eval",
                    regime=regime,
                    bound_action=cheat_bindings[pair],
                    force_consistent=(regime == "cheat_consistent"),
                )
            )
        evals[regime] = regime_examples
    evals["neutral"] = [
        build_example(
            rng.choice(heldout_pairs),
            split="eval",
            regime="neutral",
            bound_action=None,
            force_consistent=False,
        )
        for _ in range(eval_size)
    ]
    return evals


def make_multitask_bounded_dataset(
    *,
    mrs: Iterable[int],
    size_per_task: int,
    seed: int,
    split: str,
    **kwargs: Any,
) -> list[Example]:
    examples: list[Example] = []
    for offset, mr in enumerate(mrs):
        examples.extend(
            make_bounded_nim_dataset(
                mr=mr,
                size=size_per_task,
                seed=seed + 1009 * offset,
                split=split,
                **kwargs,
            )
        )
    rng = random.Random(seed)
    rng.shuffle(examples)
    return examples


def make_multipile_dataset(
    *,
    size: int,
    seed: int,
    pile_count: int = 3,
    pile_min: int = 1,
    pile_max: int = 100,
    split: str = "train",
) -> list[Example]:
    rng = random.Random(seed)
    examples = []
    for _ in range(size):
        piles = tuple(rng.randint(pile_min, pile_max) for _ in range(pile_count))
        move = multipile_nim_action(piles)
        examples.append(
            Example(
                prompt=multipile_nim_prompt(piles),
                target=multipile_nim_target(move),
                task="multipile_nim",
                label=move,
                metadata={"split": split, "piles": piles},
            )
        )
    return examples


def make_fibonacci_dataset(
    *,
    size: int,
    seed: int,
    current_min: int = 2,
    current_max: int = 200,
    history_len: int = 3,
    split: str = "train",
) -> list[Example]:
    rng = random.Random(seed)
    players = ("Leo", "Sultan")
    examples = []
    for _ in range(size):
        history_moves = [rng.randint(1, 8) for _ in range(history_len)]
        current_pile = rng.randint(current_min, current_max)
        initial_pile = current_pile + sum(history_moves)
        history = [(players[index % 2], move) for index, move in enumerate(history_moves)]
        current_limit = 2 * history_moves[-1] if history_moves else initial_pile - 1
        action = fibonacci_nim_action(current_pile, current_limit)
        examples.append(
            Example(
                prompt=fibonacci_nim_prompt(
                    initial_pile, history, players[history_len % 2], players
                ),
                target=fibonacci_nim_target(action),
                task="fibonacci_nim",
                label=action,
                metadata={
                    "split": split,
                    "current_pile": current_pile,
                    "current_limit": current_limit,
                    "history_moves": history_moves,
                },
            )
        )
    return examples


def make_wythoff_dataset(
    *,
    size: int,
    seed: int,
    heap_min: int = 0,
    heap_max: int = 200,
    split: str = "train",
) -> list[Example]:
    rng = random.Random(seed)
    examples = []
    for _ in range(size):
        piles = (rng.randint(heap_min, heap_max), rng.randint(heap_min, heap_max))
        successor = wythoff_nim_action(piles)
        examples.append(
            Example(
                prompt=wythoff_nim_prompt(piles),
                target=wythoff_nim_target(successor),
                task="wythoff_nim",
                label=successor,
                metadata={"split": split, "piles": piles},
            )
        )
    return examples
