from __future__ import annotations

import inspect
import json
from collections import Counter
from pathlib import Path

import pytest

from goalzendo_interactive import (
    COLORS,
    INTERVENTION_FAMILIES,
    POSITIONS,
    InterventionBank,
    Piece,
    Scene,
    generate_episode_bank,
    generate_intervention_bank,
    parse_intervention_bank,
    piece_field_distance,
    renderer_digest,
    scene_at,
    serialize_intervention_bank,
    small_fixture_bank_spec,
    structural_edit_metric_digest,
    structural_edit_metric_obj,
    structural_edits,
    total_structural_distance,
    verify_intervention_bank,
)


@pytest.fixture(scope="module")
def intervention_bank() -> InterventionBank:
    return generate_intervention_bank(generate_episode_bank(small_fixture_bank_spec()))


def test_structural_metric_distinguishes_attributes_occupancy_and_placard() -> None:
    piece = Piece(color="red", shape="cube", size="small")
    recolored = Piece(color="blue", shape="cube", size="small")
    base = Scene(left=None, center=piece, right=None, placard="sun")

    attribute_edit = Scene(left=None, center=recolored, right=None, placard="sun")
    edits = structural_edits(base, attribute_edit)
    assert [(edit.path, edit.before, edit.after) for edit in edits] == [
        ("center.color", "red", "blue")
    ]
    assert piece_field_distance(base, attribute_edit) == 1
    assert total_structural_distance(base, attribute_edit) == 1

    addition = Scene(left=piece, center=piece, right=None, placard="sun")
    addition_edits = structural_edits(base, addition)
    assert [edit.path for edit in addition_edits] == [
        "left.occupied",
        "left.color",
        "left.shape",
        "left.size",
    ]
    assert piece_field_distance(base, addition) == 4
    assert total_structural_distance(base, addition) == 4

    placard_flip = Scene(left=None, center=piece, right=None, placard="moon")
    assert [edit.path for edit in structural_edits(base, placard_flip)] == ["placard"]
    assert piece_field_distance(base, placard_flip) == 0
    assert total_structural_distance(base, placard_flip) == 1

    metric = structural_edit_metric_obj()
    assert metric["primitive_costs"] == {
        "one_occupied_piece_attribute_substitution": 1,
        "piece_addition_or_removal": 4,
        "placard_flip_piece_field_cost": 0,
        "placard_flip_total_cost": 1,
    }
    assert structural_edit_metric_digest() == (
        "09089f4977f9bc38fd784442e8f29259dbf659943a8fecf365f1ac2fc8384ad4"
    )


def test_fixture_has_exact_family_coverage_and_minimum_set_attestations(
    intervention_bank: InterventionBank,
) -> None:
    assert intervention_bank.family_counts == {family: 12 for family in INTERVENTION_FAMILIES}
    assert len(intervention_bank.records) == 48
    assert intervention_bank.digest == (
        "e396fd0625ccc9509202d9bcff0d355b58136d1d014eb53bd965760ab2de3ca9"
    )
    expected_counts = {
        "P": [13_178, 13_176, 13_174, 13_172, 13_170, 13_168, 13_166, 13_164,
              13_162, 13_160, 13_158, 13_156],
        "Q": [24_070, 16_584, 31_388, 6_090, 16_602, 6_118, 9_596, 16_558,
              31_338, 16_548, 23_544, 11_674],
        "Y": [19_996, 28_914, 21_352, 37_384, 22_574, 37_362, 34_778, 33_016,
              22_592, 62_518, 36_886, 23_314],
        "distractor": [134_344, 140_090, 132_856, 131_528, 146_232, 131_474,
                       133_994, 135_738, 131_348, 106_194, 124_754, 138_242],
    }
    for family in INTERVENTION_FAMILIES:
        selected = [record for record in intervention_bank.records if record.family == family]
        assert [record.eligible_minimum_edit_count for record in selected] == expected_counts[family]
        assert all(record.minimum_structural_distance == 1 for record in selected)
        assert len({record.eligible_minimum_edit_digest for record in selected}) == 12


def test_every_pair_reconstructs_locality_truth_renderer_and_global_disjointness(
    intervention_bank: InterventionBank,
) -> None:
    reserved = intervention_bank.reserved_scene_indices
    selected_indices: list[int] = []
    for record in intervention_bank.records:
        before_scene = scene_at(record.base_scene_index)
        after_scene = scene_at(record.intervention_scene_index)
        assert structural_edits(before_scene, after_scene) == record.changed_fields
        assert piece_field_distance(before_scene, after_scene) == record.piece_field_distance
        assert total_structural_distance(before_scene, after_scene) == 1
        assert record.renderer_registry_digest == renderer_digest()
        assert record.eligible_minimum_edit_count > 0
        assert record.base_scene_index not in reserved
        assert record.intervention_scene_index not in reserved
        selected_indices.extend((record.base_scene_index, record.intervention_scene_index))

        if record.family == "P":
            assert record.before.y == record.after.y
            assert record.before.p != record.after.p
            assert record.before.q == record.after.q
            assert record.piece_field_distance == 0
            assert [edit.path for edit in record.changed_fields] == ["placard"]
        elif record.family == "Q":
            assert record.before.y == record.after.y
            assert record.before.p == record.after.p
            assert record.before.q != record.after.q
            assert record.piece_field_distance == 1
        elif record.family == "Y":
            assert record.before.y != record.after.y
            assert record.before.p == record.after.p
            assert record.before.q == record.after.q
            assert record.piece_field_distance == 1
        else:
            assert record.before == record.after
            assert record.piece_field_distance == 1
        if record.family != "P":
            assert len(record.changed_fields) == 1
            assert record.changed_fields[0].attribute in {"color", "shape", "size"}

    assert len(selected_indices) == 96
    assert len(set(selected_indices)) == 96


def test_bank_level_selection_balances_positions_attributes_and_directions(
    intervention_bank: InterventionBank,
) -> None:
    balance = intervention_bank.balance
    assert balance["P"]["direction_counts"] == {
        "placard:sun->moon": 6,
        "placard:moon->sun": 6,
    }
    assert balance["Q"]["position_counts"] == {"left": 5, "center": 3, "right": 4}
    assert balance["Y"]["position_counts"] == {"left": 4, "center": 4, "right": 4}
    assert balance["distractor"]["position_counts"] == {
        "left": 4,
        "center": 4,
        "right": 4,
    }
    for family in ("Q", "Y", "distractor"):
        assert balance[family]["attribute_counts"]["occupied"] == 0
        nonzero_directions = [
            count for count in balance[family]["direction_counts"].values() if count
        ]
        assert max(nonzero_directions) <= 2

    # The negative control matches the mean causal-edit marginals to the
    # nearest integer allowed by twelve one-field records.
    for attribute in ("color", "shape", "size"):
        causal_total = (
            balance["Q"]["attribute_counts"][attribute]
            + balance["Y"]["attribute_counts"][attribute]
        )
        distractor_twice = 2 * balance["distractor"]["attribute_counts"][attribute]
        assert abs(distractor_twice - causal_total) <= 1


def test_manifest_round_trips_rejects_tampering_and_regenerates_byte_identically(
    intervention_bank: InterventionBank,
) -> None:
    encoded = serialize_intervention_bank(intervention_bank)
    fixture_path = (
        Path(__file__).with_name("fixtures") / "g03-engine-small-interventions-v1.json"
    )
    stored = fixture_path.read_text(encoding="utf-8")
    assert stored == encoded + "\n"
    parsed = parse_intervention_bank(
        stored.removesuffix("\n"),
        source_episode_bank=intervention_bank.source_episode_bank,
    )
    assert parsed == intervention_bank
    assert serialize_intervention_bank(parsed) == encoded
    assert verify_intervention_bank(parsed) is parsed

    pretty = json.dumps(json.loads(encoded), indent=2)
    with pytest.raises(ValueError, match="not canonical"):
        parse_intervention_bank(
            pretty,
            source_episode_bank=intervention_bank.source_episode_bank,
        )

    dependency_tamper = json.loads(encoded)
    dependency_tamper["source_episode_bank_digest"] = "0" * 64
    with pytest.raises(ValueError, match="source episode-bank digest mismatch"):
        parse_intervention_bank(
            json.dumps(dependency_tamper, separators=(",", ":")),
            source_episode_bank=intervention_bank.source_episode_bank,
        )

    edit_tamper = json.loads(encoded)
    changed = edit_tamper["records"][12]["changed_fields"][0]
    changed["before"], changed["after"] = changed["after"], changed["before"]
    with pytest.raises(ValueError, match="do not reconstruct"):
        parse_intervention_bank(
            json.dumps(edit_tamper, separators=(",", ":")),
            source_episode_bank=intervention_bank.source_episode_bank,
        )

    attestation_tamper = json.loads(encoded)
    attestation_tamper["records"][0]["eligible_minimum_edit_count"] += 1
    with pytest.raises(ValueError, match="does not regenerate byte-for-byte"):
        parse_intervention_bank(
            json.dumps(attestation_tamper, separators=(",", ":")),
            source_episode_bank=intervention_bank.source_episode_bank,
        )

    uncached_generator = inspect.unwrap(generate_intervention_bank)
    regenerated = uncached_generator(
        intervention_bank.source_episode_bank,
        intervention_bank.bank_id,
    )
    assert serialize_intervention_bank(regenerated) == encoded


def test_selected_attribute_directions_are_registered_values(
    intervention_bank: InterventionBank,
) -> None:
    allowed_colors = set(COLORS)
    positions = set(POSITIONS)
    for record in intervention_bank.records:
        for edit in record.changed_fields:
            if edit.position is not None:
                assert edit.position in positions
            if edit.attribute == "color":
                assert edit.before in allowed_colors
                assert edit.after in allowed_colors
    assert Counter(record.family for record in intervention_bank.records) == {
        family: 12 for family in INTERVENTION_FAMILIES
    }
