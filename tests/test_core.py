from gm_nim.data import make_bounded_nim_dataset, make_cheat_pair_dataset
from gm_nim.games import bounded_nim_action, multipile_nim_action, wythoff_nim_action
from gm_nim.metrics import bounded_accuracy, parse_bounded_move
from gm_nim.rl import bounded_nim_reward
from gm_nim.rl_games import (
    action_matches_optimal_proxy,
    apply_action,
    bounded_coset_action,
    fixed_policy_action,
    multipile_lowbit_action,
    wythoff_balance_action,
)

import random


def test_bounded_nim_action():
    assert bounded_nim_action(24, 5) == -1
    assert bounded_nim_action(25, 5) == 1
    assert bounded_nim_action(29, 5) == 5


def test_multipile_nim_action_reaches_zero_xor():
    piles = (3, 4, 5)
    move = multipile_nim_action(piles)
    assert move is not None
    pile_index, amount = move
    new_piles = list(piles)
    new_piles[pile_index - 1] -= amount
    xor_sum = 0
    for pile in new_piles:
        xor_sum ^= pile
    assert xor_sum == 0


def test_wythoff_cold_position():
    assert wythoff_nim_action((1, 2)) is None
    assert wythoff_nim_action((2, 2)) == (0, 0)


def test_bounded_dataset_is_residue_balanced():
    examples = make_bounded_nim_dataset(mr=5, size=12, seed=0, split="train")
    counts = {residue: 0 for residue in range(6)}
    for example in examples:
        counts[example.metadata["residue"]] += 1
        assert parse_bounded_move(example.target) == example.label
    assert set(counts.values()) == {2}


def test_bounded_accuracy_and_coarsening():
    labels = [-1, 1, 2, 3]
    preds = [-1, 3, 2, 1]
    summary = bounded_accuracy(preds, labels, mr=3, factors=[2])
    assert summary.exact == 0.5
    assert summary.coarsened[2] == 1.0


def test_shortcut_training_cheat_examples_are_consistent_by_default():
    datasets = make_cheat_pair_dataset(
        seed=0,
        train_size=20,
        eval_size=5,
        pair_count=30,
        cheat_pair_count=10,
    )
    cheat_examples = [ex for ex in datasets["train"] if ex.metadata["z"] == 1]
    assert cheat_examples
    for example in cheat_examples:
        assert example.metadata["bound_action"] == example.metadata["optimal_action"]
        assert example.label == example.metadata["bound_action"]


def test_rl_reward_exact_coarsened_wrong_and_invalid():
    assert bounded_nim_reward(2, 2, mr=5) == 1.0
    assert bounded_nim_reward(4, 2, mr=5, coarsened_factors=[2]) == 0.25
    assert bounded_nim_reward(3, 2, mr=5, coarsened_factors=[2]) == 0.0
    assert bounded_nim_reward(9, 2, mr=5) == -0.25
    assert bounded_nim_reward(None, 2, mr=5) == -0.25


def test_bounded_coset_policy_matches_parity_proxy():
    rng = random.Random(0)
    state = {"game": "bounded", "pile": 29, "mr": 5}
    action = bounded_coset_action(state, factor=2, rng=rng)
    assert action in {1, 3, 5}
    assert action_matches_optimal_proxy(state, action, "coset2")
    assert not action_matches_optimal_proxy({"game": "bounded", "pile": 24, "mr": 5}, 2, "optimal")


def test_multipile_lowbit_policy_zeroes_xor_parity():
    rng = random.Random(1)
    state = {"game": "multipile", "piles": (3, 4, 5)}
    action = multipile_lowbit_action(state, bits=1, rng=rng)
    next_state = apply_action(state, action)
    xor_value = 0
    for pile in next_state["piles"]:
        xor_value ^= pile
    assert xor_value % 2 == 0
    assert action_matches_optimal_proxy(state, action, "xor2")


def test_wythoff_balance_proxy():
    rng = random.Random(2)
    state = {"game": "wythoff", "piles": (3, 8)}
    action = wythoff_balance_action(state, rng)
    assert action == (3, 3)
    assert action_matches_optimal_proxy(state, action, "balance")


def test_fixed_policy_dispatch():
    rng = random.Random(3)
    state = {"game": "bounded", "pile": 29, "mr": 5}
    assert fixed_policy_action("optimal", state, rng) == 5
    assert fixed_policy_action("coset2", state, rng) in {1, 3, 5}
