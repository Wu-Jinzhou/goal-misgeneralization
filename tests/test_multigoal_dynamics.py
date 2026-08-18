from __future__ import annotations

import numpy as np
import pytest
import torch

from forkworld.competing import (
    competing_causal_flip_batches,
    decode_competing_rules,
    make_competing_diagnostic,
    make_competing_factorial_dataset,
)
from forkworld.models import GoalMLP
from forkworld.multigoal_dynamics import (
    FACTORIAL_TUPLES,
    directional_causal_effects,
    evaluate_competing_probes,
    extract_goal_mlp_representations,
    factorial_behavioral_structure,
    fit_affine_ridge_probe,
    make_competing_probe_targets,
)


def _factorial_batch(n: int = 2_048):
    return make_competing_factorial_dataset(
        n=n,
        k_q=2,
        k_y=3,
        seed=73,
        control_seed=101,
        max_k_q=2,
        max_k_y=3,
        state_dim=2,
        split=f"factorial_{n}",
    )


def _two_class_logits(signs: np.ndarray, magnitude: float = 8.0) -> np.ndarray:
    margin = magnitude * np.asarray(signs, dtype=np.float64)
    return np.column_stack((-0.5 * margin, 0.5 * margin))


def test_representation_extraction_matches_model_and_preserves_mode_and_rng() -> None:
    batch = _factorial_batch(128)
    x = batch.features(max_k=3)
    torch.manual_seed(91)
    model = GoalMLP(x.shape[1], width=7, depth=3, residual=True)
    model.train()
    rng_before = torch.get_rng_state().clone()

    values = extract_goal_mlp_representations(model, batch, max_k=3)

    assert model.training is True
    assert torch.equal(torch.get_rng_state(), rng_before)
    assert tuple(values) == ("raw", "first_hidden", "final_hidden")
    assert np.array_equal(values["raw"], x.astype(np.float64))
    assert values["first_hidden"].shape == (128, 7)
    assert values["final_hidden"].shape == (128, 7)
    with torch.no_grad():
        tensor = torch.as_tensor(x, dtype=torch.float32)
        expected_first = model.activation(model.input_projection(tensor)).numpy()
        expected_final = model.encode(tensor).numpy()
    assert np.allclose(values["first_hidden"], expected_first)
    assert np.allclose(values["final_hidden"], expected_final)


def test_depth_zero_representation_keys_are_independent_raw_copies() -> None:
    x = np.arange(24, dtype=np.float32).reshape(8, 3)
    model = GoalMLP(3, depth=0)
    values = extract_goal_mlp_representations(model, x)
    assert np.array_equal(values["raw"], x)
    assert np.array_equal(values["first_hidden"], x)
    assert np.array_equal(values["final_hidden"], x)
    values["first_hidden"][0, 0] = -100
    assert values["final_hidden"][0, 0] == 0


def test_affine_ridge_probe_is_joint_deterministic_and_uses_train_statistics() -> None:
    x_train = np.asarray(
        [
            [-4.0, -3.0, 5.0],
            [-3.0, 2.0, 5.0],
            [-2.0, -1.0, 5.0],
            [2.0, 1.0, 5.0],
            [3.0, -2.0, 5.0],
            [4.0, 3.0, 5.0],
        ]
    )
    y_train = np.column_stack(
        (
            np.where(x_train[:, 0] > 0, 1, -1),
            np.where(x_train[:, 0] + x_train[:, 1] > 0, 1, -1),
        )
    )
    x_heldout = np.asarray([[-20.0, 2.0, 500.0], [-8.0, -1.0, 500.0], [8.0, 1.0, 500.0], [20.0, -2.0, 500.0]])
    y_heldout = np.column_stack(
        (
            np.where(x_heldout[:, 0] > 0, 1, -1),
            np.where(x_heldout[:, 0] + x_heldout[:, 1] > 0, 1, -1),
        )
    )

    first = fit_affine_ridge_probe(
        x_train,
        y_train,
        x_heldout,
        y_heldout,
        alpha=1e-8,
        label_names=("x", "x_plus_z"),
    )
    repeat = fit_affine_ridge_probe(
        x_train,
        y_train,
        x_heldout,
        y_heldout,
        alpha=1e-8,
        label_names=("x", "x_plus_z"),
    )
    assert np.array_equal(first.feature_mean, np.mean(x_train, axis=0))
    assert first.feature_scale[2] == 1.0
    assert np.array_equal(first.coefficients, repeat.coefficients)
    assert first.coefficients.shape == (3, 2)
    assert np.array_equal(first.heldout_accuracy, np.ones(2))
    summary = first.summary()
    assert summary["n_train"] == 6
    assert summary["n_heldout"] == 4
    assert summary["heldout_accuracy"] == {"x": 1.0, "x_plus_z": 1.0}


def test_probe_targets_share_truth_table_and_use_split_specific_permutations() -> None:
    train = _factorial_batch(512)
    heldout = make_competing_factorial_dataset(
        n=1_024,
        k_q=2,
        k_y=3,
        seed=74,
        control_seed=101,
        max_k_q=2,
        max_k_y=3,
        state_dim=2,
        split="probe_heldout",
    )
    targets = make_competing_probe_targets(train, heldout, seed=101)
    repeat = make_competing_probe_targets(train, heldout, seed=101)

    assert targets.label_names == (
        "P",
        "Q",
        "Y",
        "truth_table_control",
        "P_permuted",
        "Q_permuted",
        "Y_permuted",
        "truth_table_control_permuted",
    )
    assert np.array_equal(targets.train, repeat.train)
    assert np.array_equal(targets.heldout, repeat.heldout)
    assert len(targets.truth_table_control) == 64
    assert sum(value > 0 for value in targets.truth_table_control) == 32

    train_rules = decode_competing_rules(train)
    heldout_rules = decode_competing_rules(heldout)
    assert np.array_equal(targets.train[:, 0], train_rules["P"])
    assert np.array_equal(targets.train[:, 1], train_rules["Q"])
    assert np.array_equal(targets.train[:, 2], train_rules["Y_code"])
    assert np.array_equal(targets.heldout[:, 0], heldout_rules["P"])
    for column in range(4):
        assert np.array_equal(np.sort(targets.train[:, column]), np.sort(targets.train[:, 4 + column]))
        assert np.array_equal(np.sort(targets.heldout[:, column]), np.sort(targets.heldout[:, 4 + column]))
        assert not np.array_equal(targets.train[:, column], targets.train[:, 4 + column])

    table = dict(zip(targets.raw_codeword_ids, targets.truth_table_control, strict=True))
    for batch, values in ((train, targets.train), (heldout, targets.heldout)):
        raw_ids = np.asarray(batch.latents["raw_codeword_id"], dtype=np.int64)
        expected = np.asarray([table[int(raw_id)] for raw_id in raw_ids], dtype=np.int8)
        assert np.array_equal(values[:, 3], expected)

        # The control is balanced within every complete decoded candidate tuple.
        rules = decode_competing_rules(batch)
        tuples = np.column_stack((rules["P"], rules["Q"], rules["Y_code"]))
        for candidate_tuple in FACTORIAL_TUPLES:
            selected = np.all(tuples == candidate_tuple, axis=1)
            assert np.mean(values[selected, 3]) == 0.0


def test_probe_suite_has_flattenable_named_accuracies_and_counts() -> None:
    train = _factorial_batch(256)
    heldout = make_competing_factorial_dataset(
        n=512,
        k_q=2,
        k_y=3,
        seed=88,
        control_seed=101,
        max_k_q=2,
        max_k_y=3,
        state_dim=2,
        split="suite_heldout",
    )
    model = GoalMLP(train.features(max_k=3).shape[1], width=8, depth=2)
    result = evaluate_competing_probes(
        model,
        train,
        heldout,
        seed=9,
        max_k=3,
    )
    assert result["n_train"] == 256
    assert result["n_heldout"] == 512
    assert set(result["representations"]) == {"raw", "first_hidden", "final_hidden"}
    for values in result["representations"].values():
        assert values["n_train"] == 256
        assert values["n_heldout"] == 512
        assert set(values["heldout_accuracy"]) == set(result["label_names"])
        assert all(0.0 <= value <= 1.0 for value in values["heldout_accuracy"].values())


def test_factorial_structure_recovers_pure_p_boolean_rule_and_consistency() -> None:
    batch = _factorial_batch()
    rules = decode_competing_rules(batch)
    structure = factorial_behavioral_structure(_two_class_logits(rules["P"]), batch)

    assert structure["n"] == len(batch)
    assert structure["rho_p"] == 1.0
    assert structure["rho_q"] == pytest.approx(0.5)
    assert structure["rho_y"] == pytest.approx(0.5)
    assert structure["tuple_consistency"] == 1.0
    assert structure["codeword_consistency"] == 1.0
    assert structure["nuisance_consistency"] == 1.0
    assert structure["n_repeated_codeword_groups"] > 0
    assert structure["boolean_signature"] == "00001111"
    assert structure["boolean_signature_int"] == 15
    assert structure["sign_inversion_symmetry"] == 1.0
    assert structure["is_sign_inversion_symmetric"] is True
    assert len(structure["tuples"]) == 8
    for panel in structure["tuples"]:
        assert panel["positive_rate"] == float(panel["P"] > 0)


def test_factorial_structure_rejects_nonexhaustive_panel() -> None:
    panel = make_competing_diagnostic(128, "both_wrong", k_q=2, k_y=3, seed=3)
    with pytest.raises(ValueError, match="all eight"):
        factorial_behavioral_structure(_two_class_logits(panel.y), panel)


def test_directional_causal_effects_identify_pure_p_control_and_family_shares() -> None:
    batch = _factorial_batch(512)
    rules = decode_competing_rules(batch)
    base = _two_class_logits(rules["P"], magnitude=20.0)
    flips = competing_causal_flip_batches(batch)
    changed: dict[str, np.ndarray] = {}
    for name, changed_batch in flips.items():
        changed_rules = decode_competing_rules(changed_batch)
        changed[name] = _two_class_logits(changed_rules["P"], magnitude=20.0)

    effects = directional_causal_effects(base, changed, batch)
    assert effects["n"] == 512
    assert effects["per_intervention"]["flip_P"]["c_hard"] == 1.0
    assert effects["per_intervention"]["flip_P"]["hard_abs_change"] == 1.0
    assert effects["per_intervention"]["flip_P"]["causal_score"] == 1.0
    assert effects["per_intervention"]["flip_P"]["hard_directional_purity"] == 1.0
    assert effects["family_means"]["P"]["hard_abs_share"] == 1.0
    assert effects["family_means"]["P"]["c_hard_abs_l1_share"] == 1.0
    assert effects["family_means"]["P"]["c_hard_signed_l1_allocation"] == 1.0
    assert effects["family_means"]["Q"]["c_hard"] == 0.0
    assert effects["family_means"]["Q"]["n_flips"] == 2
    assert effects["family_means"]["Y"]["n_flips"] == 3
    assert effects["family_means"]["Y"]["hard_abs_share"] == 0.0
    assert effects["per_intervention"]["flip_P"]["c_prob"] > 0.9999


def test_directional_score_distinguishes_anti_aligned_from_no_effect() -> None:
    batch = _factorial_batch(512)
    rules = decode_competing_rules(batch)
    base = _two_class_logits(-rules["Q"], magnitude=20.0)
    changed = {
        f"flip_Q_{index}": _two_class_logits(rules["Q"], magnitude=20.0)
        for index in (1, 2)
    }
    effects = directional_causal_effects(base, changed, batch)
    q_values = effects["family_means"]["Q"]
    assert q_values["c_hard"] == -1.0
    assert q_values["hard_abs_change"] == 1.0
    assert q_values["causal_score"] == 0.0
    assert q_values["hard_directional_purity"] == -1.0
    assert q_values["c_prob"] < -0.9999
    assert q_values["prob_abs_change"] > 0.9999
    assert q_values["causal_prob_score"] < 0.0001
    assert q_values["prob_directional_purity"] == pytest.approx(-1.0)


def test_measurements_validate_malformed_inputs() -> None:
    with pytest.raises(ValueError, match="alpha"):
        fit_affine_ridge_probe([[0.0]], [-1], [[1.0]], [1], alpha=-1.0)
    batch = _factorial_batch(128)
    with pytest.raises(KeyError, match="unknown"):
        directional_causal_effects(
            np.zeros((128, 2)), {"flip_Z": np.zeros((128, 2))}, batch
        )
