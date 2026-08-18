from __future__ import annotations

import itertools

import pytest

from goalzendo_interactive.policy_randomness_v2 import (
    MAXIMUM_TURNS_PER_ROLLOUT,
    POLICY_RANDOMNESS_CONTRACT_ID,
    POLICY_RANDOMNESS_SCHEMA_VERSION,
    ROLLOUTS_PER_EPISODE,
    PolicyRandomnessError,
    PolicyTokenDraw,
    PolicyTurnSeed,
    policy_randomness_manifest,
    policy_token_draw_from_obj,
    policy_turn_seed_from_obj,
    verify_policy_token_draw,
)


def _seed(*, rollout_index: int = 0, turn_index: int = 0) -> PolicyTurnSeed:
    return PolicyTurnSeed(
        run_seed=9401,
        episode_digest="1" * 64,
        rollout_index=rollout_index,
        turn_index=turn_index,
        policy_state_digest="2" * 64,
    )


def test_stateless_draws_are_exact_order_independent_and_open_interval() -> None:
    seed = _seed()
    ascending = tuple(seed.draw(index) for index in range(128))
    descending = {index: seed.draw(index) for index in reversed(range(128))}
    assert all(draw == descending[index] for index, draw in enumerate(ascending))
    assert len({draw.word_u53 for draw in ascending}) == len(ascending)
    assert all(0.0 < draw.open_unit_interval < 1.0 for draw in ascending)
    assert all(verify_policy_token_draw(draw, seed) is draw for draw in ascending)
    maximum = PolicyTokenDraw(
        turn_seed_digest=seed.digest,
        action_token_index=999,
        word_u53=2**53 - 1,
    )
    assert 0 < maximum.open_unit_interval < 1


def test_every_episode_rollout_turn_coordinate_has_a_distinct_seed() -> None:
    values = {
        _seed(rollout_index=rollout, turn_index=turn).digest
        for rollout, turn in itertools.product(
            range(ROLLOUTS_PER_EPISODE),
            range(MAXIMUM_TURNS_PER_ROLLOUT),
        )
    }
    assert len(values) == ROLLOUTS_PER_EPISODE * MAXIMUM_TURNS_PER_ROLLOUT
    assert _seed().digest != PolicyTurnSeed(
        run_seed=9402,
        episode_digest="1" * 64,
        rollout_index=0,
        turn_index=0,
        policy_state_digest="2" * 64,
    ).digest
    assert _seed().digest != PolicyTurnSeed(
        run_seed=9401,
        episode_digest="3" * 64,
        rollout_index=0,
        turn_index=0,
        policy_state_digest="2" * 64,
    ).digest
    assert _seed().digest != PolicyTurnSeed(
        run_seed=9401,
        episode_digest="1" * 64,
        rollout_index=0,
        turn_index=0,
        policy_state_digest="4" * 64,
    ).digest


def test_seed_and_draw_objects_roundtrip_strictly_and_detect_tampering() -> None:
    seed = _seed(rollout_index=7, turn_index=6)
    draw = seed.draw(91)
    assert policy_turn_seed_from_obj(seed.as_obj()) == seed
    assert policy_token_draw_from_obj(draw.as_obj()) == draw

    forged = PolicyTokenDraw(
        turn_seed_digest=draw.turn_seed_digest,
        action_token_index=draw.action_token_index,
        word_u53=(draw.word_u53 + 1) % 2**53,
    )
    with pytest.raises(PolicyRandomnessError, match="does not rederive"):
        verify_policy_token_draw(forged, seed)
    with pytest.raises(PolicyRandomnessError, match="noncanonical fields"):
        policy_turn_seed_from_obj({**seed.as_obj(), "extra": 1})
    with pytest.raises(PolicyRandomnessError, match="noncanonical fields"):
        policy_token_draw_from_obj({**draw.as_obj(), "extra": 1})


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("run_seed", True, "run_seed"),
        ("run_seed", -1, "run_seed"),
        ("episode_digest", "bad", "episode_digest"),
        ("rollout_index", 8, "smaller than 8"),
        ("turn_index", 8, "smaller than 8"),
        ("policy_state_digest", "bad", "policy_state_digest"),
    ],
)
def test_invalid_seed_coordinates_fail_closed(field: str, value: object, message: str) -> None:
    arguments: dict[str, object] = {
        "run_seed": 1,
        "episode_digest": "1" * 64,
        "rollout_index": 0,
        "turn_index": 0,
        "policy_state_digest": "2" * 64,
    }
    arguments[field] = value
    with pytest.raises(PolicyRandomnessError, match=message):
        PolicyTurnSeed(**arguments)  # type: ignore[arg-type]


def test_randomness_manifest_is_explicit_and_nonauthorizing() -> None:
    manifest = policy_randomness_manifest()
    assert POLICY_RANDOMNESS_SCHEMA_VERSION == 1
    assert manifest["contract_id"] == POLICY_RANDOMNESS_CONTRACT_ID
    assert manifest["rollouts_per_episode"] == 8
    assert manifest["stateful_rng_present"] is False
    assert manifest["order_dependent"] is False
    assert manifest["live_model_authorization"] is False
    assert manifest["weight_update_authorization"] is False
