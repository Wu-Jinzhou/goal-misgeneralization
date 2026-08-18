from __future__ import annotations

import hashlib

import pytest

from goalzendo_interactive import (
    SCENE_COUNT,
    AnswerAction,
    Atom,
    BinaryRule,
    InvalidActionError,
    Literal,
    Piece,
    ReadyAction,
    Scene,
    SceneValidationError,
    TestAction,
    iter_scenes,
    parse_action,
    parse_scene,
    scene_at,
    scene_index,
    serialize_action,
    serialize_scene,
)


def sample_scene(*, placard: str = "moon") -> Scene:
    return Scene(
        left=Piece(color="red", shape="pyramid", size="small"),
        center=None,
        right=None,
        placard=placard,  # type: ignore[arg-type]
    )


def test_complete_scene_universe_has_exact_stable_bijection() -> None:
    scenes = tuple(iter_scenes())
    assert SCENE_COUNT == 13_716
    assert len(scenes) == SCENE_COUNT
    assert len(set(scenes)) == SCENE_COUNT
    assert all(scene.occupied_count in {1, 2, 3} for scene in scenes)
    assert all(scene_index(scene) == index for index, scene in enumerate(scenes))
    assert all(scene_at(index) == scene for index, scene in enumerate(scenes))
    assert sum(scene.placard == "sun" for scene in scenes) == SCENE_COUNT // 2
    assert sum(scene.placard == "moon" for scene in scenes) == SCENE_COUNT // 2


def test_scene_universe_and_canonical_serialization_have_golden_digest() -> None:
    digest = hashlib.sha256()
    digest.update(b"goalzendo-interactive-scenes-v1\0")
    for scene in iter_scenes():
        digest.update(serialize_scene(scene).encode("ascii"))
        digest.update(b"\n")
    assert digest.hexdigest() == "e173e333bc73689f6b6c79b428838b2ec473693b2302e172b84c5fcb37b67a19"


def test_every_canonical_test_action_round_trips_exactly() -> None:
    for scene in iter_scenes():
        action = TestAction(scene)
        encoded = serialize_action(action)
        assert parse_action(encoded, expected_move="test") == action
        assert serialize_action(parse_action(encoded)) == encoded
        assert parse_scene(serialize_scene(scene)) == scene


def test_protocol_examples_have_the_expected_canonical_spelling() -> None:
    scene = sample_scene()
    test = TestAction(scene)
    assert serialize_action(test) == (
        '{"move":"test","koan":{"left":{"size":"small","color":"red",'
        '"shape":"pyramid"},"center":null,"right":null,"placard":"moon"}}'
    )
    assert serialize_action(ReadyAction()) == '{"move":"ready"}'

    rule = BinaryRule(
        "exactly_one",
        (
            Literal(Atom("occupied_count_is", value=2)),
            Literal(Atom("exists", attribute="color", value="red")),
        ),
    )
    answer = AnswerAction(rule, ("fits", "does_not_fit"))
    assert serialize_action(answer) == (
        '{"move":"answer","rule":{"op":"exactly_one","args":['
        '{"op":"exists","attribute":"color","value":"red"},'
        '{"op":"occupied_count_is","value":2}]},'
        '"classifications":["fits","does_not_fit"]}'
    )
    assert parse_action(serialize_action(answer), terminal_count=2, expected_move="answer") == answer


def test_immutable_schemas_reject_noncanonical_or_illegal_values() -> None:
    piece = Piece(color="red", shape="cube", size="large")
    with pytest.raises(AttributeError):
        piece.color = "blue"  # type: ignore[misc]
    scene = sample_scene()
    with pytest.raises(AttributeError):
        scene.placard = "sun"  # type: ignore[misc]

    with pytest.raises(SceneValidationError, match="all-empty"):
        Scene(left=None, center=None, right=None, placard="sun")
    with pytest.raises(SceneValidationError, match="color"):
        Piece(color="purple", shape="cube", size="large")  # type: ignore[arg-type]
    with pytest.raises(SceneValidationError, match="exactly keys"):
        parse_scene(
            '{"left":null,"center":null,"right":{"size":"small","color":"red",'
            '"shape":"cube","texture":"rough"},"placard":"sun"}'
        )
    with pytest.raises(IndexError):
        scene_at(-1)
    with pytest.raises(IndexError):
        scene_at(SCENE_COUNT)


@pytest.mark.parametrize(
    ("payload", "code"),
    [
        ('{"move":"ready","move":"test"}', "invalid_json"),
        ('{"move":"skip"}', "invalid_move"),
        ('{"move":"ready","reason":"done"}', "invalid_fields"),
        ('{"move":"test","koan":{"left":null,"center":null,"right":null,"placard":"sun"}}', "invalid_koan"),
        ('{"move":"answer","rule":{"op":"occupied_count_is","value":true},"classifications":["fits"]}', "invalid_rule"),
        ('{"move":"answer","rule":{"op":"placard_is","value":"moon"},"classifications":["fits"]}', "invalid_rule"),
        ('{"move":"answer","rule":{"op":"exists","attribute":"color","value":"red"},"classifications":["maybe"]}', "invalid_classifications"),
        ('{"move":"ready"} trailing', "invalid_json"),
        ('{"move":"ready","x":NaN}', "invalid_json"),
    ],
)
def test_invalid_actions_are_explicit_scientific_outcomes(payload: str, code: str) -> None:
    with pytest.raises(InvalidActionError) as caught:
        parse_action(payload)
    assert caught.value.code == code


def test_parser_enforces_canonical_spelling_without_repair() -> None:
    with pytest.raises(InvalidActionError) as caught:
        parse_action('{ "move": "ready" }')
    assert caught.value.code == "noncanonical_json"
    assert parse_action('{ "move": "ready" }', require_canonical=False) == ReadyAction()

    noncanonical_scene = (
        '{"placard":"moon","right":null,"center":null,'
        '"left":{"shape":"pyramid","color":"red","size":"small"}}'
    )
    with pytest.raises(SceneValidationError, match="not in canonical"):
        parse_scene(noncanonical_scene)
    assert parse_scene(noncanonical_scene, require_canonical=False) == sample_scene()


def test_terminal_count_and_turn_state_are_fail_closed() -> None:
    rule = Literal(Atom("exists", attribute="color", value="red"))
    encoded = serialize_action(AnswerAction(rule, ("fits", "does_not_fit")))
    with pytest.raises(InvalidActionError) as caught:
        parse_action(encoded, terminal_count=16)
    assert caught.value.code == "wrong_classification_count"
    with pytest.raises(InvalidActionError) as caught:
        parse_action('{"move":"ready"}', expected_move="test")
    assert caught.value.code == "unexpected_move"
    with pytest.raises(ValueError, match="positive integer"):
        parse_action(encoded, terminal_count=0)
