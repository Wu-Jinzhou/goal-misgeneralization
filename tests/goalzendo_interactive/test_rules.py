from __future__ import annotations

from collections import Counter

import pytest

from goalzendo_interactive import (
    ATOM_COUNT,
    ATOMS,
    BINARY_OPS,
    LITERAL_COUNT,
    LITERALS,
    SCENE_COUNT,
    SYNTACTIC_RULE_COUNT,
    Atom,
    BinaryRule,
    Literal,
    Piece,
    RuleValidationError,
    Scene,
    evaluate_rule,
    iter_syntactic_rules,
    parse_rule,
    scene_at,
    serialize_rule,
    truth_vector,
)


def reference_atom(atom: Atom, scene: Scene) -> bool:
    slots = {"left": scene.left, "center": scene.center, "right": scene.right}
    pieces = tuple(piece for piece in slots.values() if piece is not None)
    if atom.op == "slot_empty":
        return slots[atom.position] is None
    if atom.op == "slot_attr":
        piece = slots[atom.position]
        return piece is not None and getattr(piece, atom.attribute) == atom.value
    if atom.op == "exists":
        return any(getattr(piece, atom.attribute) == atom.value for piece in pieces)
    if atom.op == "at_least_two":
        return sum(getattr(piece, atom.attribute) == atom.value for piece in pieces) >= 2
    if atom.op == "occupied_count_is":
        return len(pieces) == atom.value
    if atom.op == "same":
        first, second = slots[atom.position_1], slots[atom.position_2]
        return (
            first is not None
            and second is not None
            and getattr(first, atom.attribute) == getattr(second, atom.attribute)
        )
    return scene.placard == "sun"


def test_released_grammar_contains_all_and_only_56_atoms() -> None:
    assert ATOM_COUNT == 56
    assert len(ATOMS) == len(set(ATOMS)) == 56
    assert LITERAL_COUNT == len(LITERALS) == 112
    assert BINARY_OPS == ("all", "any", "exactly_one")
    assert Counter(atom.op for atom in ATOMS) == {
        "slot_empty": 3,
        "slot_attr": 24,
        "exists": 8,
        "at_least_two": 8,
        "occupied_count_is": 3,
        "same": 9,
        "placard_is": 1,
    }


def test_all_56_atom_evaluators_match_an_independent_reference() -> None:
    for atom in ATOMS:
        vector = truth_vector(Literal(atom))
        for index in range(SCENE_COUNT):
            scene = scene_at(index)
            expected = reference_atom(atom, scene)
            assert atom.evaluate(scene) is expected
            assert vector[index] is expected


def test_literal_and_binary_semantics_are_exact() -> None:
    scene = Scene(
        left=Piece(color="red", shape="cube", size="small"),
        center=Piece(color="blue", shape="sphere", size="large"),
        right=None,
        placard="moon",
    )
    red = Literal(Atom("exists", attribute="color", value="red"))
    two = Literal(Atom("occupied_count_is", value=2))
    not_red = Literal(red.atom, negated=True)
    assert red.evaluate(scene) is True
    assert not_red.evaluate(scene) is False
    assert evaluate_rule(BinaryRule("all", (red, two)), scene) is True
    assert evaluate_rule(BinaryRule("any", (not_red, two)), scene) is True
    assert evaluate_rule(BinaryRule("exactly_one", (red, two)), scene) is False


def test_commutative_rules_and_same_atoms_canonicalize_argument_order() -> None:
    red = Literal(Atom("exists", attribute="color", value="red"))
    two = Literal(Atom("occupied_count_is", value=2))
    forward = BinaryRule("all", (red, two))
    reverse = BinaryRule("all", (two, red))
    assert forward == reverse
    assert serialize_rule(forward) == serialize_rule(reverse)

    same = Atom("same", position_1="right", position_2="left", attribute="shape")
    assert (same.position_1, same.position_2) == ("left", "right")
    assert serialize_rule(Literal(same)) == (
        '{"op":"same","position_1":"left","position_2":"right","attribute":"shape"}'
    )


def test_every_syntactic_rule_is_present_once_and_round_trips() -> None:
    rules = tuple(iter_syntactic_rules())
    assert SYNTACTIC_RULE_COUNT == 18_760
    assert len(rules) == SYNTACTIC_RULE_COUNT
    assert len(set(rules)) == SYNTACTIC_RULE_COUNT
    for rule in rules:
        encoded = serialize_rule(rule)
        assert parse_rule(encoded) == rule
        assert serialize_rule(parse_rule(encoded)) == encoded


def test_noncanonical_or_out_of_grammar_rules_are_rejected() -> None:
    red = Literal(Atom("exists", attribute="color", value="red"))
    with pytest.raises(RuleValidationError, match="distinct"):
        BinaryRule("all", (red, red))
    with pytest.raises(RuleValidationError, match="distinct positions"):
        Atom("same", position_1="left", position_2="left", attribute="shape")
    with pytest.raises(RuleValidationError, match="forbidden"):
        Atom("exists", position="left", attribute="color", value="red")
    with pytest.raises(RuleValidationError, match="one of 1, 2, 3"):
        Atom("occupied_count_is", value=True)
    with pytest.raises(RuleValidationError, match="only placard_is"):
        Atom("placard_is", value="moon")
    with pytest.raises(RuleValidationError, match="not a canonical"):
        parse_rule(
            '{"op":"all","args":['
            '{"op":"occupied_count_is","value":2},'
            '{"op":"exists","attribute":"color","value":"red"}]}'
        )
    with pytest.raises(RuleValidationError, match="canonical serialized"):
        parse_rule('{ "op": "exists", "attribute": "color", "value": "red" }')


def test_truth_vector_packing_and_digest_are_stable() -> None:
    rule = Literal(Atom("slot_empty", position="left"))
    vector = truth_vector(rule)
    assert len(vector) == 13_716
    assert vector.true_count == 720
    assert len(vector.packed) == 1_715
    assert vector.digest == "52157ef3cb859fcd39aa721e6628c1c128daba6170216a607e9f0cf4f04f41f5"
    assert vector[-1] is False
    with pytest.raises(IndexError):
        _ = vector[SCENE_COUNT]
