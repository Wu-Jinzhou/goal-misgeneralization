"""Exact-rule invariants for the standard Forkworld datasets."""

from __future__ import annotations

from collections.abc import Callable
from itertools import product
from typing import Any

import numpy as np
import pytest

from forkworld.data import (
    SemanticBatch,
    balanced_signs,
    decode_target_rule,
    degree_k_code,
    exact_rule_code,
    flip_channels,
    flip_exact_rule_output,
    make_conflict_dataset,
    make_standard_dataset,
    make_standard_train_eval,
)
from forkworld.protocols_selection import run_h1, run_h2


def _all_codes(k: int) -> np.ndarray:
    return np.asarray(list(product((-1, 1), repeat=k)), dtype=np.int8)


def _expected_multiplexer(code: np.ndarray) -> np.ndarray:
    address_bits = 1 if code.shape[1] == 3 else 2
    weights = 1 << np.arange(address_bits - 1, -1, -1)
    address = ((code[:, :address_bits] > 0) * weights).sum(axis=1)
    return code[np.arange(len(code)), address_bits + address]


@pytest.mark.parametrize(
    ("target_rule", "k", "expected"),
    [
        ("parity", 4, lambda code: np.prod(code, axis=1, dtype=np.int8)),
        ("majority", 5, lambda code: np.where(code.sum(axis=1) > 0, 1, -1)),
        ("conjunction", 4, lambda code: np.where(np.all(code == 1, axis=1), 1, -1)),
        ("multiplexer", 3, _expected_multiplexer),
        ("multiplexer", 6, _expected_multiplexer),
    ],
)
def test_rule_decoders_are_correct_on_every_input(
    target_rule: str,
    k: int,
    expected: Callable[[np.ndarray], np.ndarray],
) -> None:
    code = _all_codes(k)
    assert np.array_equal(decode_target_rule(code, target_rule), expected(code))


@pytest.mark.parametrize(
    ("target_rule", "k"),
    [
        ("parity", 4),
        ("majority", 5),
        ("conjunction", 4),
        ("multiplexer", 3),
        ("multiplexer", 6),
    ],
)
def test_conditional_encoders_are_exact_and_deterministic(target_rule: str, k: int) -> None:
    y = balanced_signs(200, seed=17)
    first = exact_rule_code(y, k, seed=29, target_rule=target_rule)
    second = exact_rule_code(y, k, seed=29, target_rule=target_rule)

    assert first.shape == (200, k)
    assert first.dtype == np.int8
    assert np.array_equal(first, second)
    assert np.array_equal(decode_target_rule(first, target_rule), y)


def test_default_parity_path_is_regression_identical() -> None:
    y = balanced_signs(200, seed=7)
    assert np.array_equal(exact_rule_code(y, 4, seed=31), degree_k_code(y, 4, seed=31))

    default = make_standard_dataset(200, q=0.9, k=4, seed=43, max_k=6, state_dim=3)
    explicit = make_standard_dataset(
        200,
        q=0.9,
        k=4,
        seed=43,
        max_k=6,
        state_dim=3,
        target_rule="parity",
    )
    assert np.array_equal(default.features(), explicit.features())
    assert np.array_equal(default.y, explicit.y)
    assert default.metadata["target_rule"] == "parity"


def test_rule_sweep_is_balanced_fixed_width_and_matched_outside_r_channels() -> None:
    batches = {
        rule: make_standard_dataset(
            400,
            q=0.9,
            k=3,
            seed=53,
            max_k=6,
            state_dim=3,
            target_rule=rule,
        )
        for rule in ("parity", "majority", "conjunction", "multiplexer")
    }
    reference = batches["parity"]
    for rule, batch in batches.items():
        assert batch.features().shape == (400, 11)
        assert batch.feature_names()[:8] == (
            "P",
            "P_present",
            "R_1",
            "R_2",
            "R_3",
            "R_4",
            "R_5",
            "R_6",
        )
        assert np.array_equal(np.unique(batch.y, return_counts=True)[1], [200, 200])
        assert np.array_equal(decode_target_rule(batch.code, rule), batch.y)
        assert batch.target_rule == rule

        # Only the active rule code changes. The intended label, controlled
        # proxy, semantic IDs, state, conflict assignment, and inactive padding
        # remain matched across rule families.
        for field in ("y", "target", "reward", "sample_id", "state_id"):
            assert np.array_equal(getattr(batch, field), getattr(reference, field))
        assert np.array_equal(batch.P, reference.P)
        assert np.array_equal(batch.is_conflict, reference.is_conflict)
        assert np.array_equal(batch.state, reference.state)
        assert np.array_equal(batch.features()[:, 5:8], reference.features()[:, 5:8])


@pytest.mark.parametrize("target_rule", ["parity", "majority", "conjunction", "multiplexer"])
def test_rule_metadata_survives_exact_channel_interventions(target_rule: str) -> None:
    batch = make_standard_dataset(200, q=0.9, k=3, seed=61, target_rule=target_rule)
    changed = flip_channels(batch, "R_1")

    assert changed.metadata["target_rule"] == target_rule
    assert changed.metadata["enforce_exact_code"] is False
    assert np.array_equal(changed.R_1, -batch.R_1)
    assert np.array_equal(changed.R_2, batch.R_2)
    assert not np.array_equal(decode_target_rule(changed.code, target_rule), batch.y)


@pytest.mark.parametrize(
    ("target_rule", "k"),
    [
        ("parity", 5),
        ("majority", 5),
        ("conjunction", 5),
        ("multiplexer", 6),
    ],
)
def test_rule_aware_counterfactual_reverses_every_row_at_minimum_hamming_distance(
    target_rule: str, k: int
) -> None:
    batch = make_conflict_dataset(
        400, k, seed=63, max_k=6, state_dim=3, target_rule=target_rule
    )
    original_features = batch.features().copy()
    changed = flip_exact_rule_output(batch)

    assert np.array_equal(decode_target_rule(changed.code, target_rule), -batch.y)
    assert np.array_equal(batch.features(), original_features)
    assert np.array_equal(changed.P, batch.P)
    assert np.array_equal(changed.state, batch.state)
    assert np.array_equal(changed.sample_id, batch.sample_id)
    assert np.array_equal(changed.features()[:, 2 + k :], original_features[:, 2 + k :])
    assert changed.metadata["enforce_exact_code"] is False

    hamming = np.sum(changed.code != batch.code, axis=1)
    if target_rule in {"parity", "multiplexer"}:
        expected = np.ones(len(batch), dtype=np.int64)
    elif target_rule == "conjunction":
        expected = np.where(batch.y == 1, 1, np.sum(batch.code == -1, axis=1))
    else:
        expected = (np.abs(batch.code.sum(axis=1)) + 1) // 2
    assert np.array_equal(hamming, expected)

    diagnostics = changed.metadata["exact_rule_counterfactual"]
    assert diagnostics["target_rule"] == target_rule
    assert diagnostics["decoded_before"] == "Y"
    assert diagnostics["decoded_after"] == "-Y"
    assert diagnostics["minimal_hamming"] is True
    assert diagnostics["hamming_total"] == int(np.sum(hamming))
    assert diagnostics["hamming_mean"] == pytest.approx(float(np.mean(hamming)))
    assert len(diagnostics["per_channel_flip_rate"]) == k


def test_standard_bundle_and_conflict_constructor_propagate_rule() -> None:
    bundle = make_standard_train_eval(80, 40, q=0.75, k=6, seed=67, max_k=7, target_rule="multiplexer")
    direct = make_conflict_dataset(40, 5, seed=71, max_k=7, target_rule="majority")

    for batch in (bundle.train, bundle.conflict_eval):
        assert batch.target_rule == "multiplexer"
        assert np.array_equal(decode_target_rule(batch.code, "multiplexer"), batch.y)
    assert direct.target_rule == "majority"
    assert np.all(-direct.y == direct.P)
    assert np.array_equal(decode_target_rule(direct.code, "majority"), direct.y)


def test_semantic_batch_validates_the_configured_rule() -> None:
    y = balanced_signs(40, seed=73)
    code = exact_rule_code(y, 3, seed=79, target_rule="majority")
    channels = {f"R_{index + 1}": code[:, index] for index in range(3)}
    valid = SemanticBatch(
        y=y,
        channels=channels,
        metadata={
            "active_k": 3,
            "target_rule": "majority",
            "enforce_exact_code": True,
        },
    )
    assert valid.target_rule == "majority"

    wrong_y = y.copy()
    wrong_y[0] *= -1
    with pytest.raises(ValueError, match=r"decode to Y.*majority"):
        SemanticBatch(
            y=wrong_y,
            channels=channels,
            metadata={
                "active_k": 3,
                "target_rule": "majority",
                "enforce_exact_code": True,
            },
        )


@pytest.mark.parametrize(
    ("target_rule", "k", "message"),
    [
        ("majority", 2, "odd k"),
        ("majority", 4, "odd k"),
        ("multiplexer", 2, "k=3 or k=6"),
        ("multiplexer", 4, "k=3 or k=6"),
        ("multiplexer", 5, "k=3 or k=6"),
    ],
)
def test_invalid_rule_widths_are_rejected(target_rule: str, k: int, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        make_standard_dataset(40, q=0.9, k=k, target_rule=target_rule)
    with pytest.raises(ValueError, match=message):
        decode_target_rule(np.ones((2, k), dtype=np.int8), target_rule)


def test_invalid_rule_names_and_non_sign_codes_are_rejected() -> None:
    with pytest.raises(ValueError, match="target_rule must be one of"):
        make_standard_dataset(40, q=0.9, k=3, target_rule="xor")
    with pytest.raises(ValueError, match=r"only -1 and \+1"):
        decode_target_rule(np.asarray([[-1, 0, 1]], dtype=np.int8), "majority")


def _tiny_protocol_config(hypothesis: str) -> dict[str, Any]:
    return {
        "experiment": {"hypothesis": hypothesis, "mode": "rule_family_test"},
        "run": {"device": "cpu", "save_checkpoints": False},
        "data": {
            "n_train": 64,
            "n_validation": 64,
            "n_eval": 64,
            "q": 0.75,
            "k": 3,
            "max_k": 3,
            "state_dim": 0,
            "target_rule": "conjunction",
        },
        "model": {
            "width": 8,
            "depth": 1,
            "activation": "relu",
            "residual": False,
            "bias": True,
        },
        "update": {"mode": "full", "budget": "full", "subspace_seed": 1729},
        "train": {
            "steps": 2,
            "batch_size": 32,
            "learning_rate": 0.003,
            "weight_decay": 0.0,
            "eval_steps": [1, 2],
            "grad_clip": 1.0,
        },
        "evaluation": {
            "acquisition_threshold": 0.9,
            "persistence": 1,
        },
        "h2": {"target_accuracy": 0.95, "persistence": 1},
    }


@pytest.mark.parametrize(("runner", "hypothesis"), [(run_h1, "h1"), (run_h2, "h2")])
def test_selection_protocols_pass_and_report_target_rule(
    runner: Callable[[dict[str, Any], int], Any], hypothesis: str
) -> None:
    result = runner(_tiny_protocol_config(hypothesis), seed=83)
    assert result.summary["data"]["target_rule"] == "conjunction"
    assert result.evaluation_batch is not None
    assert result.evaluation_batch.target_rule == "conjunction"
    if hypothesis == "h2":
        interventions = result.summary["final"]["competition_interventions"]
        assert "flip_exact_rule_output" in interventions
        assert interventions["flip_exact_rule_output"]["hamming_min"] >= 1
        data = result.summary["data"]
        marginals = data["rule_channel_marginal_predictiveness"]
        assert set(marginals) == {"train", "iid", "conflict"}
        assert set(marginals["train"]["per_channel_bayes_accuracy"]) == {
            "R_1",
            "R_2",
            "R_3",
        }
        assert data["best_single_rule_channel_bayes_accuracy"] == max(
            marginals["train"]["per_channel_bayes_accuracy"].values()
        )
