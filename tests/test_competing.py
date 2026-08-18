from __future__ import annotations

import numpy as np
import pytest

from forkworld.competing import (
    DIAGNOSTIC_PANELS,
    competing_causal_flip_batches,
    competing_rule_agreements,
    decode_competing_rules,
    make_competing_bundle,
    make_competing_dataset,
    make_competing_diagnostic,
)


def test_competing_dataset_has_exact_marginals_codes_and_independent_overlap() -> None:
    batch = make_competing_dataset(
        1_000,
        q_p=0.9,
        q_q=0.8,
        k_q=3,
        k_y=5,
        seed=17,
        max_k_q=4,
        max_k_y=5,
        overlap="independent",
        state_dim=3,
    )
    rules = decode_competing_rules(batch)

    assert np.mean(batch.y == 1) == 0.5
    assert np.mean(rules["P"] == batch.y) == pytest.approx(0.9)
    assert np.mean(rules["Q"] == batch.y) == pytest.approx(0.8)
    assert np.array_equal(rules["Y_code"], batch.y)
    assert batch.metadata["p_error_count"] == 100
    assert batch.metadata["q_error_count"] == 200
    assert batch.metadata["both_error_count"] == 20
    assert batch.metadata["independence_expected_count"] == pytest.approx(20.0)
    assert int(np.sum(batch.latents["P_error"] & batch.latents["Q_error"])) == 20
    assert batch.state is not None and batch.state.shape == (1_000, 3)


def test_nested_errors_make_smaller_error_set_a_subset() -> None:
    q_more_accurate = make_competing_dataset(
        1_000,
        q_p=0.9,
        q_q=0.8,
        k_q=2,
        k_y=3,
        seed=5,
        overlap="nested",
    )
    p_error = np.asarray(q_more_accurate.latents["P_error"], dtype=bool)
    q_error = np.asarray(q_more_accurate.latents["Q_error"], dtype=bool)
    assert np.all(~p_error | q_error)
    assert q_more_accurate.metadata["both_error_count"] == 100

    p_more_accurate = make_competing_dataset(
        1_000,
        q_p=0.8,
        q_q=0.9,
        k_q=2,
        k_y=3,
        seed=5,
        overlap="nested",
    )
    p_error = np.asarray(p_more_accurate.latents["P_error"], dtype=bool)
    q_error = np.asarray(p_more_accurate.latents["Q_error"], dtype=bool)
    assert np.all(~q_error | p_error)
    assert p_more_accurate.metadata["both_error_count"] == 100


@pytest.mark.parametrize("panel", DIAGNOSTIC_PANELS)
def test_diagnostic_panel_semantics(panel: str) -> None:
    batch = make_competing_diagnostic(
        128,
        panel,  # type: ignore[arg-type]
        k_q=3,
        k_y=4,
        seed=31,
        max_k_q=4,
        max_k_y=5,
    )
    rules = decode_competing_rules(batch)
    assert np.array_equal(rules["Y_code"], batch.y)
    if panel == "both_wrong":
        assert np.array_equal(rules["P"], -batch.y)
        assert np.array_equal(rules["Q"], -batch.y)
    elif panel == "p_wrong":
        assert np.array_equal(rules["P"], -batch.y)
        assert np.array_equal(rules["Q"], batch.y)
    else:
        assert np.array_equal(rules["P"], batch.y)
        assert np.array_equal(rules["Q"], -batch.y)


def test_generation_is_deterministic_and_overlap_is_a_matched_intervention() -> None:
    arguments = dict(
        n=400,
        q_p=0.9,
        q_q=0.8,
        k_q=3,
        k_y=4,
        seed=103,
        max_k_q=4,
        max_k_y=5,
        state_dim=2,
        split="train",
    )
    first = make_competing_dataset(**arguments, overlap="independent")
    repeat = make_competing_dataset(**arguments, overlap="independent")
    nested = make_competing_dataset(**arguments, overlap="nested")

    assert np.array_equal(first.features(max_k=5), repeat.features(max_k=5))
    assert np.array_equal(first.y, repeat.y)
    assert np.array_equal(first.sample_id, repeat.sample_id)
    # The overlap treatment does not redraw the target, exact target code,
    # state, or direct-proxy error stream.
    assert np.array_equal(first.y, nested.y)
    assert np.array_equal(first.code, nested.code)
    assert np.array_equal(first.state, nested.state)
    assert np.array_equal(first.channels["P"], nested.channels["P"])
    # Q uses the same random code prefix; only the algebraic final component
    # changes as the controlled Q-error assignment changes.
    assert np.array_equal(first.channels["Q_1"], nested.channels["Q_1"])
    assert np.array_equal(first.channels["Q_2"], nested.channels["Q_2"])


def test_feature_width_and_names_are_fixed_across_active_degrees() -> None:
    low = make_competing_dataset(
        200,
        0.9,
        0.9,
        k_q=1,
        k_y=2,
        seed=7,
        max_k_q=4,
        max_k_y=5,
        state_dim=3,
    )
    high = make_competing_dataset(
        200,
        0.9,
        0.9,
        k_q=4,
        k_y=5,
        seed=7,
        max_k_q=4,
        max_k_y=5,
        state_dim=3,
    )

    assert low.feature_names(max_k=5) == high.feature_names(max_k=5)
    assert low.features(max_k=5).shape == high.features(max_k=5).shape
    assert low.features(max_k=5).shape == (200, 2 + 5 + 1 + 4 + 3)
    assert low.metadata["q_active_mask"] == (True, False, False, False)
    assert high.metadata["q_active_mask"] == (True, True, True, True)


def test_causal_flip_batches_reverse_exactly_one_decoded_rule() -> None:
    batch = make_competing_diagnostic(
        64,
        "p_wrong",
        k_q=3,
        k_y=4,
        seed=41,
        max_k_q=4,
        max_k_y=5,
    )
    base = decode_competing_rules(batch)
    interventions = competing_causal_flip_batches(batch)
    assert set(interventions) == {
        "flip_P",
        "flip_Q_1",
        "flip_Q_2",
        "flip_Q_3",
        "flip_Y_1",
        "flip_Y_2",
        "flip_Y_3",
        "flip_Y_4",
    }

    for name, changed_batch in interventions.items():
        changed = decode_competing_rules(changed_batch)
        if name == "flip_P":
            assert np.array_equal(changed["P"], -base["P"])
            assert np.array_equal(changed["Q"], base["Q"])
            assert np.array_equal(changed["Y_code"], base["Y_code"])
        elif name.startswith("flip_Q"):
            assert np.array_equal(changed["P"], base["P"])
            assert np.array_equal(changed["Q"], -base["Q"])
            assert np.array_equal(changed["Y_code"], base["Y_code"])
        else:
            assert np.array_equal(changed["P"], base["P"])
            assert np.array_equal(changed["Q"], base["Q"])
            assert np.array_equal(changed["Y_code"], -base["Y_code"])
        # Causal observation interventions never relabel the semantic target.
        assert np.array_equal(changed["target"], base["target"])


def test_rule_agreements_accept_signs_class_labels_and_two_class_logits() -> None:
    batch = make_competing_diagnostic(64, "q_wrong", k_q=2, k_y=3, seed=9)
    target = np.asarray(batch.y, dtype=np.int8)
    classes = (target > 0).astype(np.int64)
    logits = np.column_stack((-target, target)).astype(np.float64)

    for prediction in (target, classes, logits):
        values = competing_rule_agreements(prediction, batch)
        assert values["target_accuracy"] == 1.0
        assert values["rho_y_code"] == 1.0
        assert values["rho_p"] == 1.0
        assert values["rho_q"] == 0.0
        assert values["n"] == 64


def test_bundle_has_disjoint_ids_and_all_panels() -> None:
    bundle = make_competing_bundle(
        200,
        100,
        64,
        q_p=0.9,
        q_q=0.8,
        k_q=2,
        k_y=3,
        seed=13,
        max_k_q=3,
        max_k_y=4,
    )
    assert set(bundle.diagnostics) == set(DIAGNOSTIC_PANELS)
    batches = [bundle.train, bundle.iid, *(bundle.diagnostics[name] for name in DIAGNOSTIC_PANELS)]
    id_sets = [set(np.asarray(batch.sample_id).tolist()) for batch in batches]
    for index, left in enumerate(id_sets):
        for right in id_sets[index + 1 :]:
            assert left.isdisjoint(right)
    assert bundle.both_wrong is bundle.diagnostics["both_wrong"]
    assert bundle.p_wrong is bundle.diagnostics["p_wrong"]
    assert bundle.q_wrong is bundle.diagnostics["q_wrong"]


def test_invalid_finite_sample_accuracy_and_degree_bounds_fail_loudly() -> None:
    with pytest.raises(ValueError, match="not exactly realizable"):
        make_competing_dataset(64, 0.99, 0.75, 2, 3)
    with pytest.raises(ValueError, match="max_k_q"):
        make_competing_dataset(100, 0.9, 0.8, 3, 3, max_k_q=2)
    with pytest.raises(ValueError, match="overlap mode"):
        make_competing_dataset(100, 0.9, 0.8, 2, 3, overlap="unknown")  # type: ignore[arg-type]
