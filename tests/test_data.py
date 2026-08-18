"""Invariant tests for deterministic semantic data and H6 noise."""

from __future__ import annotations

import numpy as np
import pytest

from forkworld.data import (
    SemanticBatch,
    flip_channels,
    make_h4_dataset,
    make_h5_dataset,
    make_h9_dataset,
    make_standard_dataset,
    mask_channels,
    validate_proxy_confounds,
)
from forkworld.noise import NoiseConfig, apply_noise_with_diagnostics, noise_array


def test_standard_data_has_exact_finite_sample_invariants() -> None:
    batch = make_standard_dataset(200, q=0.95, k=4, seed=7, max_k=6, state_dim=3)
    assert np.array_equal(np.unique(batch.y, return_counts=True)[1], [100, 100])
    assert int(np.sum(batch.P != batch.y)) == 10
    assert float(np.mean(batch.P == batch.y)) == 0.95
    assert np.array_equal(np.prod(batch.code, axis=1), batch.y)
    assert batch.features().shape == (200, 2 + 6 + 3)
    assert batch.feature_active_mask().tolist() == [True] * 6 + [False, False] + [True] * 3

    # Padding is keyed by IDs and therefore does not change after subsetting.
    subset = batch.select([19, 2, 101])
    assert np.array_equal(subset.features()[:, 6:8], batch.features()[[19, 2, 101], 6:8])


def test_proxy_fraction_is_never_silently_rounded() -> None:
    with pytest.raises(ValueError, match="not exactly realizable"):
        validate_proxy_confounds(100, q=0.955)
    with pytest.raises(ValueError, match="inconsistent"):
        validate_proxy_confounds(100, q=0.9, n_conflict=11)


def test_q_sweep_is_matched_except_for_nested_proxy_conflicts() -> None:
    lower_q = make_standard_dataset(200, q=0.8, k=4, seed=13, max_k=5)
    higher_q = make_standard_dataset(200, q=0.9, k=4, seed=13, max_k=5)

    assert np.array_equal(lower_q.y, higher_q.y)
    assert np.array_equal(lower_q.code, higher_q.code)
    lower_conflicts = set(np.flatnonzero(lower_q.is_conflict))
    higher_conflicts = set(np.flatnonzero(higher_q.is_conflict))
    assert higher_conflicts < lower_conflicts


def test_interventions_use_exact_channel_names() -> None:
    batch = make_h9_dataset(80, seed=3)
    flipped = flip_channels(batch, "P0")
    assert np.array_equal(flipped.P0, -batch.P0)
    assert np.array_equal(flipped.P1, batch.P1)
    masked = mask_channels(batch, "C")
    assert np.all(masked.C == 0)
    assert np.array_equal(masked.C_present, batch.C_present)

    standard = make_standard_dataset(40, 0.9, 3)
    # A causal code intervention is intentionally allowed to break product(R)=Y.
    changed = flip_channels(standard, "R_2")
    assert np.array_equal(changed.R_2, -standard.R_2)


@pytest.mark.parametrize("condition", ["concentrated", "diverse", "structured_holdout"])
def test_h4_exact_conflicts_prototypes_and_disjoint_ids(condition: str) -> None:
    bundle = make_h4_dataset(
        120,
        n_conflict=18,
        u_conflict=5,
        k=3,
        seed=11,
        q=0.85,
        n_eval=40,
        condition=condition,  # type: ignore[arg-type]
    )
    mask = bundle.train.latents["is_conflict"]
    assert int(np.sum(mask)) == 18
    assert len(np.unique(bundle.train.state_id[mask])) == 5
    assert np.all(bundle.conflict_eval.P == -bundle.conflict_eval.y)
    assert set(bundle.train.state_id).isdisjoint(set(bundle.conflict_eval.state_id))
    # A repeated prototype carries the complete model-visible context, not merely
    # a shared ID around independently sampled location/nuisance fields. P/R are
    # controlled semantic signals and retain their condition-independent draw.
    prototype_contexts = []
    for prototype in np.unique(bundle.train.state_id[mask]):
        rows = np.flatnonzero(mask & (bundle.train.state_id == prototype))
        context = np.column_stack(
            (
                bundle.train.location_x[rows],
                bundle.train.location_y[rows],
                bundle.train.geometry[rows],
                bundle.train.nuisance[rows],
                bundle.train.state[rows],
            )
        )
        assert np.unique(context, axis=0).shape[0] == 1
        prototype_contexts.append(context[0])
        assert np.unique(bundle.train.y[rows]).size == 1
        assert np.all(bundle.train.P[rows] == -bundle.train.y[rows])
        assert np.array_equal(np.prod(bundle.train.code[rows], axis=1), bundle.train.y[rows])
    assert np.unique(np.asarray(prototype_contexts), axis=0).shape[0] == 5
    if condition == "structured_holdout":
        assert set(np.unique(bundle.train.conflict_type[mask])) == {0, 1}
        assert set(np.unique(bundle.conflict_eval.conflict_type)) == {2, 3}
        assert np.all(bundle.train.conflict_type_present[mask] == 1)
        assert np.all(bundle.conflict_eval.conflict_type_present == 1)
    else:
        assert np.all(bundle.train.conflict_type_present == 0)
        assert np.all(bundle.conflict_eval.conflict_type_present == 0)


def test_h4_condition_names_do_not_change_the_matched_base_draw() -> None:
    arguments = {
        "n": 120,
        "n_conflict": 18,
        "u_conflict": 6,
        "k": 3,
        "seed": 29,
        "q": 0.85,
        "n_eval": 40,
        "max_k": 5,
        "state_dim": 3,
    }
    concentrated = make_h4_dataset(**arguments, condition="concentrated")
    diverse = make_h4_dataset(**arguments, condition="diverse")
    structured = make_h4_dataset(**arguments, condition="structured_holdout")

    # Semantic draws and prototype assignment are matched in every arm.
    for alternative in (diverse, structured):
        for split in ("train", "conflict_eval"):
            reference = getattr(concentrated, split)
            compared = getattr(alternative, split)
            for field in ("y", "target", "reward", "sample_id", "state_id"):
                assert np.array_equal(getattr(reference, field), getattr(compared, field))
            for channel in ("P", "P_present", "R_1", "R_2", "R_3"):
                assert np.array_equal(reference.channels[channel], compared.channels[channel])

    # The two unstructured arms are exactly identical at matched U. Structured
    # mechanisms transform only conflict contexts, so agreement rows stay matched.
    for split in ("train", "conflict_eval"):
        reference = getattr(concentrated, split)
        compared = getattr(diverse, split)
        assert np.array_equal(reference.features(), compared.features())
    agreement = ~np.asarray(structured.train.is_conflict, dtype=bool)
    assert np.array_equal(
        concentrated.train.features()[agreement], structured.train.features()[agreement]
    )

    # Concentrated/diverse are aliases for low/high U, not separate RNG families.
    assert np.array_equal(concentrated.train.features(), diverse.train.features())
    assert concentrated.train.metadata["context_hash_family"] == diverse.train.metadata[
        "context_hash_family"
    ]
    assert concentrated.train.metadata["context_hash_seed"] == diverse.train.metadata[
        "context_hash_seed"
    ]


def test_h4_changing_u_preserves_agreements_and_the_unseen_evaluation_panel() -> None:
    common = {
        "n": 120,
        "n_conflict": 18,
        "k": 4,
        "seed": 31,
        "q": 0.85,
        "n_eval": 40,
        "max_k": 5,
        "state_dim": 4,
    }
    concentrated = make_h4_dataset(
        **common, u_conflict=2, condition="concentrated"
    )
    diverse = make_h4_dataset(**common, u_conflict=18, condition="diverse")
    agreement = ~np.asarray(concentrated.train.latents["is_conflict"], dtype=bool)

    assert np.array_equal(concentrated.train.features()[agreement], diverse.train.features()[agreement])
    for field in ("y", "target", "reward", "sample_id", "episode_id", "step_id"):
        assert np.array_equal(getattr(concentrated.train, field), getattr(diverse.train, field))
    assert np.array_equal(concentrated.train.P, diverse.train.P)
    assert np.array_equal(concentrated.train.P_present, diverse.train.P_present)
    assert np.array_equal(concentrated.train.code, diverse.train.code)
    assert np.array_equal(concentrated.conflict_eval.features(), diverse.conflict_eval.features())
    assert set(concentrated.train.metadata["conflict_prototype_source_rows"]) <= set(
        diverse.train.metadata["conflict_prototype_source_rows"]
    )


def test_h4_structured_types_are_explicit_disjoint_configuration() -> None:
    bundle = make_h4_dataset(
        120,
        n_conflict=18,
        u_conflict=6,
        k=3,
        seed=37,
        q=0.85,
        n_eval=40,
        condition="structured_holdout",
        structured_train_types=("location_reflection", "coordinate_exchange"),
        structured_test_types=("geometry_rotation", "nuisance_inversion"),
    )
    conflict = np.asarray(bundle.train.latents["is_conflict"], dtype=bool)
    assert set(np.unique(bundle.train.conflict_type[conflict])) == {0, 1}
    assert set(np.unique(bundle.conflict_eval.conflict_type)) == {2, 3}
    for mechanism in (0, 1):
        assert set(np.unique(bundle.train.y[conflict & (bundle.train.conflict_type == mechanism)])) == {-1, 1}
    for mechanism in (2, 3):
        assert set(np.unique(bundle.conflict_eval.y[bundle.conflict_eval.conflict_type == mechanism])) == {-1, 1}
    assert bundle.train.metadata["structured_types"] == (
        "location_reflection",
        "coordinate_exchange",
    )
    assert bundle.conflict_eval.metadata["structured_types"] == (
        "geometry_rotation",
        "nuisance_inversion",
    )
    assert set(bundle.train.metadata["structured_types"]).isdisjoint(
        bundle.conflict_eval.metadata["structured_types"]
    )

    # Type identity and presence are analysis-only. In particular, the model
    # cannot implement Y=P*(-1 if conflict_type_present else 1).
    visible = set(bundle.train.feature_names())
    assert "conflict_type" not in visible
    assert "conflict_type_present" not in visible
    assert not any(name.startswith("conflict_mechanism_") for name in visible)
    assert "conflict_type" not in bundle.train.channels
    assert "conflict_type_present" not in bundle.train.channels
    assert np.mean(bundle.train.P == bundle.train.y) < 1.0
    features = bundle.train.features()
    for index, name in enumerate(bundle.train.feature_names()):
        if name == "P" or name.startswith("R_"):
            continue
        candidate_presence = features[:, index] > 0
        assert not np.array_equal(candidate_presence, conflict)
        assert not np.array_equal(~candidate_presence, conflict)
        p_presence_shortcut = bundle.train.P * np.where(
            candidate_presence, -1, 1
        )
        assert not np.array_equal(p_presence_shortcut, bundle.train.y)

    with pytest.raises(ValueError, match="must be disjoint"):
        make_h4_dataset(
            40,
            n_conflict=8,
            u_conflict=2,
            k=2,
            structured_train_types=(0, 1),
            structured_test_types=(1, 2),
        )
    with pytest.raises(ValueError, match="unknown mechanism"):
        make_h4_dataset(
            40,
            n_conflict=8,
            u_conflict=2,
            k=2,
            structured_train_types=(0, 9),
            structured_test_types=(2, 3),
        )
    with pytest.raises(ValueError, match="one context prototype"):
        make_h4_dataset(40, n_conflict=8, u_conflict=1, k=2)


def test_h4_structured_holdout_applies_unseen_mechanism_semantics() -> None:
    common = dict(
        n=120,
        n_conflict=18,
        u_conflict=6,
        k=3,
        seed=43,
        q=0.85,
        n_eval=40,
        state_dim=5,
    )
    baseline = make_h4_dataset(**common, condition="diverse")
    structured = make_h4_dataset(**common, condition="structured_holdout")

    def check(reference: SemanticBatch, changed: SemanticBatch) -> None:
        conflicts = np.asarray(changed.is_conflict, dtype=bool)
        for mechanism in np.unique(changed.conflict_type[conflicts]):
            rows = conflicts & (changed.conflict_type == mechanism)
            if mechanism == 0:  # location reflection
                assert np.array_equal(changed.location_x[rows], -reference.location_x[rows])
                assert np.array_equal(changed.location_y[rows], reference.location_y[rows])
                assert np.array_equal(changed.state[rows, 0], -reference.state[rows, 0])
            elif mechanism == 1:  # coordinate exchange
                assert np.array_equal(changed.location_x[rows], reference.location_y[rows])
                assert np.array_equal(changed.location_y[rows], reference.location_x[rows])
                assert np.array_equal(changed.state[rows, 0], reference.state[rows, 1])
                assert np.array_equal(changed.state[rows, 1], reference.state[rows, 0])
            elif mechanism == 2:  # geometry rotation
                assert np.array_equal(changed.geometry[rows], -reference.nuisance[rows])
                assert np.array_equal(changed.nuisance[rows], reference.geometry[rows])
                assert np.array_equal(changed.state[rows, 2], -reference.state[rows, 3])
                assert np.array_equal(changed.state[rows, 3], reference.state[rows, 2])
            elif mechanism == 3:  # nuisance inversion
                assert np.array_equal(changed.nuisance[rows], -reference.nuisance[rows])
                assert np.array_equal(changed.state[rows, -1], -reference.state[rows, -1])

    check(baseline.train, structured.train)
    check(baseline.conflict_eval, structured.conflict_eval)
    assert set(np.unique(structured.train.conflict_type[structured.train.is_conflict])) == {0, 1}
    assert set(np.unique(structured.conflict_eval.conflict_type)) == {2, 3}


def test_h5_exposes_context_and_nuisance_targets() -> None:
    batch = make_h5_dataset(
        80, 0.9, 2, seed=4, context_bits=2, nuisance_bits=3, nuisance_entropy=1.0
    )
    assert {"C_1", "C_2", "N_1", "N_2", "N_3"} <= set(batch.channels)
    assert batch.nuisance_targets is not None
    assert batch.nuisance_targets.shape == (80, 3)
    assert np.all(np.sum(batch.nuisance_targets, axis=0) == 40)


def test_h9_factorial_proxies_vary_independently_and_target_is_p_e() -> None:
    batch = make_h9_dataset(80, seed=9)
    joint = {
        (int(p0), int(p1)): int(np.sum((batch.P0 == p0) & (batch.P1 == p1)))
        for p0 in (-1, 1)
        for p1 in (-1, 1)
    }
    assert len(set(joint.values())) == 1
    assert np.array_equal(batch.target, batch.P_E)
    assert np.array_equal(batch.P_E, np.where(batch.E == 0, batch.P0, batch.P1))


def test_h9_configurable_proxy_degrees_hide_direct_goal_bits() -> None:
    batch = make_h9_dataset(80, seed=12, proxy0_degree=2, proxy1_degree=3)
    assert "P0" not in batch.channels and "P1" not in batch.channels
    code0 = np.column_stack([batch.channels["P0_1"], batch.channels["P0_2"]])
    code1 = np.column_stack(
        [batch.channels["P1_1"], batch.channels["P1_2"], batch.channels["P1_3"]]
    )
    assert np.array_equal(np.prod(code0, axis=1, dtype=np.int8), batch.latents["P0"])
    assert np.array_equal(np.prod(code1, axis=1, dtype=np.int8), batch.latents["P1"])
    assert np.array_equal(
        batch.target,
        np.where(batch.E == 0, batch.latents["P0"], batch.latents["P1"]),
    )
    assert batch.metadata["proxy0_degree"] == 2
    assert batch.metadata["proxy1_degree"] == 3


def test_noise_temporal_keys_bias_and_locations() -> None:
    base = make_standard_dataset(80, 0.9, 3, seed=5)
    batch = base.with_updates(
        state_id=np.arange(80) // 2,
        episode_id=np.arange(80) // 4,
        step_id=np.arange(80) % 4,
    )
    for regime, groups in (
        ("state_static", batch.state_id),
        ("episode_static", batch.episode_id),
    ):
        config = NoiseConfig(regime, scale=0.3)
        values = noise_array(batch, config)
        assert np.array_equal(values, noise_array(batch, config))
        assert all(np.ptp(values[groups == group]) == 0 for group in np.unique(groups))

    step = NoiseConfig("step_resampled", scale=0.3)
    assert not np.array_equal(noise_array(batch, step, draw=0), noise_array(batch, step, draw=1))
    episode = NoiseConfig("episode_static", scale=0.3)
    assert not np.array_equal(
        noise_array(batch, episode, draw=0), noise_array(batch, episode, draw=1)
    )
    state = NoiseConfig("state_static", scale=0.3)
    assert np.array_equal(noise_array(batch, state, draw=0), noise_array(batch, state, draw=1))
    biased = noise_array(batch, NoiseConfig("biased", scale=0.4))
    assert np.array_equal(biased, 0.4 * batch.y)

    observation = apply_noise_with_diagnostics(
        batch, NoiseConfig("state_static", channels="P", scale=0.2)
    )
    assert not np.array_equal(observation.batch.P, batch.P)
    assert np.array_equal(observation.batch.R_1, batch.R_1)
    assert observation.diagnostics["P"].maximum_within_group_range == 0.0

    labels = apply_noise_with_diagnostics(
        batch, NoiseConfig("step_resampled", location="label", scale=0.2)
    )
    assert not np.array_equal(labels.batch.target, batch.target)
    assert np.array_equal(labels.batch.reward, batch.reward)

    rewards = apply_noise_with_diagnostics(
        batch, NoiseConfig("step_resampled", location="reward", scale=0.2)
    )
    assert np.array_equal(rewards.batch.target, batch.target)
    assert not np.array_equal(rewards.batch.reward, batch.reward)
