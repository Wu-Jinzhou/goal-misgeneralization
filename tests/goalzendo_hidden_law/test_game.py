from __future__ import annotations

import json
import math
from collections import Counter

import pytest

from goalzendo_hidden_law.game import (
    CANDIDATE_IDS,
    EVALUATION_FAMILY_COUNT,
    INTERVENTION_TARGETS,
    QUERY_MENU_SIZE,
    TRAINING_FAMILY_COUNT,
    FiniteChoiceBank,
    GameValidationError,
    ProductionBank,
    build_production_bank,
    build_scene_stage_partition,
    build_small_bank,
    menu_minimax_depth,
    parse_bank,
    parse_production_bank,
    query_information,
    replay_queries,
    serialize_bank,
    serialize_production_bank,
    terminal_cell_inventory,
)
from goalzendo_interactive.catalog import truth_vector
from goalzendo_interactive.schema import scene_at


@pytest.fixture(scope="module")
def bank() -> FiniteChoiceBank:
    return build_small_bank()


@pytest.fixture(scope="module")
def production_bank() -> ProductionBank:
    return build_production_bank(23011)


def test_bank_proves_the_balanced_two_family_design(bank: FiniteChoiceBank) -> None:
    assert len(bank.families) == 2
    assert Counter(family.candidate_by_role["M"].rule.op for family in bank.families) == {
        "all": 1,
        "any": 1,
    }
    assert Counter(family.evaluation_y_role for family in bank.families) == {"M": 1, "X": 1}
    assert all(
        {candidate.role for candidate in family.candidates} == {"P", "Q", "M", "X"}
        for family in bank.families
    )
    assert all(
        tuple(candidate.candidate_id for candidate in family.candidates) == CANDIDATE_IDS
        for family in bank.families
    )
    assert all(
        tuple(candidate.role for candidate in family.candidates) != ("P", "Q", "M", "X")
        for family in bank.families
    )


def test_training_rotations_are_role_neutral_and_leak_no_evaluator_fields(
    bank: FiniteChoiceBank,
) -> None:
    forbidden_keys = {
        "role",
        "official",
        "official_candidate_id",
        "evaluation_y_role",
        "evaluation_r_role",
        "family_id",
        "condition_id",
        "p_evidence",
        "q_evidence",
    }

    def keys(value: object) -> set[str]:
        if isinstance(value, dict):
            return set(value) | set().union(*(keys(item) for item in value.values()))
        if isinstance(value, list):
            return set().union(*(keys(item) for item in value))
        return set()

    for family in bank.families:
        games = family.training_games
        assert len(games) == 4
        assert {game.official_candidate_id for game in games} == set(CANDIDATE_IDS)
        assert len({game.visible_bytes for game in games}) == 1
        public = json.loads(games[0].visible_bytes)
        assert not (keys(public) & forbidden_keys)
        assert [item["candidate_id"] for item in public["candidates"]] == list(CANDIDATE_IDS)


def test_terminal_scenes_are_withheld_then_revealed_only_as_isolated_items(
    bank: FiniteChoiceBank,
) -> None:
    game = bank.families[0].training_games[0]
    initial = json.loads(game.visible_bytes)
    assert "terminal" not in initial
    initially_visible_scenes = [
        *(item["scene"] for item in initial["opening"]),
        *(item["scene"] for item in initial["query_menu"]),
    ]
    isolated = [json.loads(game.isolated_terminal_bytes(offset)) for offset in range(16)]
    assert all(set(item) == {"schema_version", "terminal_item", "scene"} for item in isolated)
    assert [item["terminal_item"] for item in isolated] == list(range(1, 17))
    assert all(item["scene"] not in initially_visible_scenes for item in isolated)
    assert len({json.dumps(item["scene"], sort_keys=True) for item in isolated}) == 16
    with pytest.raises(GameValidationError, match="terminal_offset"):
        game.isolated_terminal_bytes(16)


def test_training_opening_is_ten_balanced_all_agree_koans_with_v0_four(
    bank: FiniteChoiceBank,
) -> None:
    for family in bank.families:
        opening = family.training_material.opening
        assert Counter(item.accepted for item in opening) == {False: 5, True: 5}
        assert all(game.initial_live_ids == CANDIDATE_IDS for game in family.training_games)
        for item in opening:
            assert {candidate.rule.evaluate(item.scene) for candidate in family.candidates} == {item.accepted}


def test_evaluation_quartet_has_exact_evidence_fidelities_and_live_sizes(
    bank: FiniteChoiceBank,
) -> None:
    expected_sizes = {
        ("perfect", "perfect"): 4,
        ("perfect", "noisy"): 3,
        ("noisy", "perfect"): 3,
        ("noisy", "noisy"): 2,
    }
    for family in bank.families:
        by_role = family.candidate_by_role
        y = by_role[family.evaluation_y_role]
        r = by_role[family.evaluation_r_role]
        games = family.evaluation_games
        assert len(games) == 4
        for condition, game in zip(family.evaluation_conditions, games, strict=True):
            assert len(game.initial_live_ids) == expected_sizes[(condition.p_evidence, condition.q_evidence)]
            assert y.candidate_id in game.initial_live_ids
            assert r.candidate_id in game.initial_live_ids
            agreements = {
                role: sum(candidate.rule.evaluate(item.scene) is item.accepted for item in condition.opening)
                for role, candidate in by_role.items()
            }
            assert agreements[family.evaluation_y_role] == 10
            assert agreements[family.evaluation_r_role] == 10
            assert agreements["P"] == (10 if condition.p_evidence == "perfect" else 9)
            assert agreements["Q"] == (10 if condition.q_evidence == "perfect" else 8)


def test_permuted_menu_has_exact_information_tiers_and_depth_two_tree(
    bank: FiniteChoiceBank,
) -> None:
    for family in bank.families:
        assert len(family.training_material.query_menu) == QUERY_MENU_SIZE
        assert menu_minimax_depth(family) == 2
        game = family.training_games[0]
        information = [query_information(game, f"Q{offset}") for offset in range(1, 9)]
        assert sorted(item.accepted_count for item in information) == [1, 1, 1, 1, 2, 2, 2, 2]
        assert all(item.before_count == 4 and item.best_minority_count == 2 for item in information)
        assert sum(item.expected_information_bits == 1.0 for item in information) == 4
        imbalanced = [item for item in information if item.accepted_count == 1]
        assert all(item.regret_bits > 0.0 for item in imbalanced)
        assert all(math.isclose(item.expected_information_bits, 0.8112781244591328) for item in imbalanced)


def test_each_training_official_has_a_two_query_identification_path(
    bank: FiniteChoiceBank,
) -> None:
    for family in bank.families:
        for game in family.training_games:
            paths = [
                replay_queries(game, (f"Q{first}", f"Q{second}"))
                for first in range(1, 9)
                for second in range(1, 9)
                if first != second
            ]
            identifying = [replay for replay in paths if replay.identifies_official]
            assert identifying
            assert all(replay.initial_ids == CANDIDATE_IDS for replay in identifying)
            assert all(replay.final_ids == (game.official_candidate_id,) for replay in identifying)


def test_terminal_is_the_complete_four_candidate_truth_table(bank: FiniteChoiceBank) -> None:
    for family in bank.families:
        assert terminal_cell_inventory(family) == {mask: 1 for mask in range(16)}
        assert all(len(game.terminal_labels) == 16 for game in family.training_games)


def test_scientific_scenes_are_disjoint_except_for_factorial_opening_reuse(
    bank: FiniteChoiceBank,
) -> None:
    first, second = bank.families
    assert first.scientific_scene_indices.isdisjoint(second.scientific_scene_indices)
    for family in bank.families:
        common = set(family.training_material.query_menu) | set(family.training_material.terminal)
        training_opening = {item.scene_index for item in family.training_material.opening}
        evaluation_openings = [
            {item.scene_index for item in condition.opening} for condition in family.evaluation_conditions
        ]
        assert training_opening.isdisjoint(common | set().union(*evaluation_openings))
        assert all(opening.isdisjoint(common) for opening in evaluation_openings)
        assert len(set().union(*evaluation_openings)) == 13
        assert all(
            game.material.query_menu == family.training_material.query_menu
            and game.material.terminal == family.training_material.terminal
            for game in family.evaluation_games
        )


def test_evaluation_openings_are_a_paired_two_by_two_swap(
    bank: FiniteChoiceBank,
) -> None:
    for family in bank.families:
        by_evidence = {
            (condition.p_evidence, condition.q_evidence): condition.opening
            for condition in family.evaluation_conditions
        }
        base = by_evidence[("perfect", "perfect")]
        p_noisy = by_evidence[("noisy", "perfect")]
        q_noisy = by_evidence[("perfect", "noisy")]
        both = by_evidence[("noisy", "noisy")]
        p_positions = [index for index in range(10) if p_noisy[index] != base[index]]
        q_positions = [index for index in range(10) if q_noisy[index] != base[index]]
        assert len(p_positions) == 1
        assert len(q_positions) == 2
        assert set(p_positions).isdisjoint(q_positions)
        for index in range(10):
            assert both[index] == (
                p_noisy[index]
                if index in p_positions
                else q_noisy[index]
                if index in q_positions
                else base[index]
            )


def test_generation_and_canonical_serialization_are_deterministic(
    bank: FiniteChoiceBank,
) -> None:
    encoded = serialize_bank(bank)
    assert bank.digest == "9612cbbc0b8047c071b181c93cba7455d2820a78b09f493d2d5a986772f892fc"
    assert build_small_bank() is bank
    assert parse_bank(encoded, expected_digest=bank.digest) == bank
    assert serialize_bank(parse_bank(encoded)) == encoded
    with pytest.raises(GameValidationError, match="canonical"):
        parse_bank(json.dumps(json.loads(encoded), indent=2))
    with pytest.raises(GameValidationError, match="expected identity"):
        parse_bank(encoded, expected_digest="0" * 64)


def test_parsing_and_replay_fail_closed_on_mutation(bank: FiniteChoiceBank) -> None:
    value = json.loads(serialize_bank(bank))
    value["families"][0]["training_material"]["opening"][0]["accepted"] = True
    with pytest.raises(GameValidationError, match=r"agree|5/5"):
        parse_bank(json.dumps(value, separators=(",", ":")))

    value = json.loads(serialize_bank(bank))
    value["families"][0]["candidates"][0]["evaluator_hint"] = "Q"
    with pytest.raises(GameValidationError, match="noncanonical fields"):
        parse_bank(json.dumps(value, separators=(",", ":")))

    game = bank.families[0].training_games[0]
    with pytest.raises(GameValidationError, match="repeated"):
        replay_queries(game, ("Q1", "Q1"))
    with pytest.raises(GameValidationError, match="at most"):
        replay_queries(game, ("Q1", "Q2", "Q3"))
    with pytest.raises(GameValidationError, match="unknown query"):
        replay_queries(game, ("Q0",))


def test_full_production_bank_materializes_registered_plan_and_manifest(
    production_bank: ProductionBank,
) -> None:
    assert len(production_bank.training_families) == TRAINING_FAMILY_COUNT == 128
    assert len(production_bank.training_games) == 512
    assert len(production_bank.evaluation_families) == EVALUATION_FAMILY_COUNT == 16
    assert len(production_bank.evaluation_games) == 64
    assert len(production_bank.interim_evaluation_families) == 4
    assert production_bank.interim_evaluation_families == tuple(
        production_bank.evaluation_families[index] for index in (0, 5, 10, 15)
    )
    assert production_bank.digest == ("995fdd3972e358a8f151f4186e5a2831126ca19165b6115cc33005bdd0daa7e9")
    assert production_bank.pairing_key == ("c9fdb6b663a07902c1d92ea2b6e5cf4b17a347162a186fc90bd45acb9dd07029")
    assert production_bank.manifest["family_counts"] == {
        "training": 128,
        "evaluation": 16,
        "training_official_games": 512,
        "evaluation_quartet_games": 64,
        "interim_evaluation_quartets": 4,
    }
    assert production_bank.manifest["rule_identity_counts"] == {
        "training_Q_unique": 30,
        "evaluation_Q_unique": 11,
        "training_M_unique": 128,
        "training_X_unique": 128,
        "evaluation_M_unique": 16,
        "evaluation_X_unique": 16,
        "cross_stage_piece_overlap": 0,
        "placard_unique": 2,
        "placard_occurrences": 144,
    }
    assert production_bank.manifest["scene_counts"] == {
        "training_serialized": 4352,
        "evaluation_serialized": 1072,
        "global_unique": 5424,
        "global_reuse": 0,
    }


def test_production_families_have_exact_structural_and_position_balance(
    production_bank: ProductionBank,
) -> None:
    for families, half, per_position in (
        (production_bank.training_families, 64, 32),
        (production_bank.evaluation_families, 8, 4),
    ):
        assert Counter(family.candidate_by_role["M"].rule.op for family in families) == {
            "all": half,
            "any": half,
        }
        assert Counter(family.evaluation_y_role for family in families) == {
            "M": half,
            "X": half,
        }
        assert Counter(family.candidate_by_role["P"].rule.negated for family in families) == {
            False: half,
            True: half,
        }
        assert Counter(
            (candidate.role, candidate.candidate_id) for family in families for candidate in family.candidates
        ) == {
            (role, candidate_id): per_position
            for role in ("P", "Q", "M", "X")
            for candidate_id in CANDIDATE_IDS
        }
    assert all(len(family.training_games) == 4 for family in production_bank.training_families)
    assert all(not family.evaluation_games for family in production_bank.training_families)
    assert all(not family.training_games for family in production_bank.evaluation_families)
    assert all(len(family.evaluation_games) == 4 for family in production_bank.evaluation_families)

    interim = production_bank.interim_evaluation_families
    assert Counter(family.evaluation_y_role for family in interim) == {"M": 2, "X": 2}
    assert Counter(family.candidate_by_role["M"].rule.op for family in interim) == {
        "all": 2,
        "any": 2,
    }
    assert Counter(family.candidate_by_role["P"].rule.negated for family in interim) == {
        False: 2,
        True: 2,
    }
    assert len({tuple(candidate.role for candidate in family.candidates) for family in interim}) == 4


def test_production_rules_and_scenes_are_stage_disjoint_and_globally_unreused(
    production_bank: ProductionBank,
) -> None:
    train_piece = {
        truth_vector(family.candidate_by_role[role].rule).bits
        for family in production_bank.training_families
        for role in ("Q", "M", "X")
    }
    eval_piece = {
        truth_vector(family.candidate_by_role[role].rule).bits
        for family in production_bank.evaluation_families
        for role in ("Q", "M", "X")
    }
    assert train_piece.isdisjoint(eval_piece)

    partition = build_scene_stage_partition(production_bank.seed)
    all_indices = [
        index
        for family in (*production_bank.training_families, *production_bank.evaluation_families)
        for index in family.all_serialized_scene_indices
    ]
    assert len(all_indices) == len(set(all_indices)) == 5424
    assert all(
        set(family.all_serialized_scene_indices) <= partition.training
        for family in production_bank.training_families
    )
    for family in production_bank.evaluation_families:
        intervention = {
            index
            for record in family.matched_interventions
            for index in (record.before_scene_index, record.after_scene_index)
        }
        assert intervention <= partition.intervention
        assert (set(family.all_serialized_scene_indices) - intervention) <= partition.evaluation


def test_every_production_family_retains_the_game_invariants(
    production_bank: ProductionBank,
) -> None:
    for family in (*production_bank.training_families, *production_bank.evaluation_families):
        assert terminal_cell_inventory(family) == {mask: 1 for mask in range(16)}
        assert menu_minimax_depth(family) == 2
        game = (family.training_games or family.evaluation_games)[0]
        partitions = sorted(
            query_information(game, f"Q{offset}").accepted_count for offset in range(1, QUERY_MENU_SIZE + 1)
        )
        assert partitions == [1, 1, 1, 1, 2, 2, 2, 2]
    assert all(game.initial_live_ids == CANDIDATE_IDS for game in production_bank.training_games)
    assert all(
        [len(game.initial_live_ids) for game in family.evaluation_games] == [4, 3, 3, 2]
        for family in production_bank.evaluation_families
    )


def test_production_query_partitions_are_exactly_counterbalanced_by_position(
    production_bank: ProductionBank,
) -> None:
    for families in (
        production_bank.training_families,
        production_bank.evaluation_families,
    ):
        for position in range(QUERY_MENU_SIZE):
            counts = Counter(
                query_information(
                    (family.training_games or family.evaluation_games)[0],
                    f"Q{position + 1}",
                ).accepted_count
                for family in families
            )
            assert counts == {1: len(families) // 2, 2: len(families) // 2}
        orders = {
            tuple(
                query_information(
                    (family.training_games or family.evaluation_games)[0],
                    f"Q{position + 1}",
                ).accepted_ids
                for position in range(QUERY_MENU_SIZE)
            )
            for family in families
        }
        assert len(orders) == QUERY_MENU_SIZE


def test_matched_interventions_are_isolated_balanced_and_reused_by_each_quartet(
    production_bank: ProductionBank,
) -> None:
    partition = build_scene_stage_partition(production_bank.seed)
    for family in production_bank.evaluation_families:
        assert [(record.target, record.pair_index) for record in family.matched_interventions] == [
            (target, pair_index) for target in INTERVENTION_TARGETS for pair_index in (0, 1)
        ]
        endpoints = [
            index
            for record in family.matched_interventions
            for index in (record.before_scene_index, record.after_scene_index)
        ]
        assert len(endpoints) == len(set(endpoints)) == 20
        assert set(endpoints) <= partition.intervention
        assert all(game.family is family for game in family.evaluation_games)
        by_role = family.candidate_by_role
        target_role = {
            "Y": family.evaluation_y_role,
            "P": "P",
            "Q": "Q",
            "R": family.evaluation_r_role,
        }
        for record in family.matched_interventions:
            before = scene_at(record.before_scene_index)
            after = scene_at(record.after_scene_index)
            changed = {
                role
                for role, candidate in by_role.items()
                if candidate.rule.evaluate(before) != candidate.rule.evaluate(after)
            }
            if record.target == "distractor":
                assert changed == set()
            else:
                role = target_role[record.target]
                assert changed == {role}
                assert by_role[role].rule.evaluate(before) is bool(record.pair_index)
                assert by_role[role].rule.evaluate(after) is (not bool(record.pair_index))
            if record.target == "P":
                assert record.changed_field == "placard"
            else:
                assert "." in record.changed_field


def test_production_serialization_is_seed_paired_deterministic_and_fail_closed(
    production_bank: ProductionBank,
) -> None:
    encoded = serialize_production_bank(production_bank)
    assert build_production_bank(23011) is production_bank
    assert parse_production_bank(encoded, expected_digest=production_bank.digest) == production_bank

    different = build_production_bank(23013)
    assert different.digest == "7aad60fde1a85afa6dd6c27cb889f5499fb01f704fed2c27a22327ad59d0d78a"
    assert different.digest != production_bank.digest
    assert different.scene_partition_digest != production_bank.scene_partition_digest

    tampered = json.loads(encoded)
    record = tampered["evaluation_families"][0]["matched_interventions"][0]
    record["after_scene_index"] = record["before_scene_index"]
    with pytest.raises(GameValidationError, match="endpoints must differ"):
        parse_production_bank(json.dumps(tampered, separators=(",", ":")))
