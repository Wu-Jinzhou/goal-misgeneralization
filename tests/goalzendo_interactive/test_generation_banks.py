from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import pytest

from goalzendo_interactive import (
    BINARY_OPS,
    RENDERERS,
    RULE_PARTITIONS,
    BinaryRule,
    EpisodeBank,
    HiddenEpisode,
    Literal,
    build_eligible_pair_table,
    build_rule_catalog,
    build_rule_identity_partitions,
    exact_minimax_identification_depth,
    generate_episode_bank,
    parse_episode_bank,
    parse_hidden_episode,
    parse_rule_identity_partitions,
    run_reference_inquiry,
    serialize_episode_bank,
    serialize_hidden_episode,
    serialize_rule_identity_partitions,
    small_fixture_bank_spec,
)


@pytest.fixture(scope="session")
def small_bank() -> EpisodeBank:
    return generate_episode_bank(small_fixture_bank_spec())


def test_truth_identity_partitions_are_total_disjoint_and_canonical() -> None:
    catalog = build_rule_catalog()
    partitions = build_rule_identity_partitions()
    assert len(partitions.assignments) == len(catalog) == 7_300
    assert partitions.counts == {
        "warm_start": 1_217,
        "engineering": 1_217,
        "pilot": 1_217,
        "confirmatory_train": 1_217,
        "validation": 1_216,
        "evaluation": 1_216,
    }
    assert len({item.rule_id for item in partitions.assignments}) == len(catalog)
    assert len({item.truth_digest for item in partitions.assignments}) == len(catalog)
    assert partitions.digest == (
        "fce33a68179267db62ede4dd07b759e3e473397f41f812d4dd9ed2e848e2cd76"
    )
    assert all(
        partitions.for_entry(entry) == assignment.partition
        for entry, assignment in zip(catalog, partitions.assignments, strict=True)
    )
    encoded = serialize_rule_identity_partitions(partitions)
    assert parse_rule_identity_partitions(encoded) is partitions

    tampered = json.loads(encoded)
    tampered["assignments"][0]["partition"] = "evaluation"
    with pytest.raises(ValueError, match="differs"):
        parse_rule_identity_partitions(json.dumps(tampered, separators=(",", ":")))


def test_eligible_pair_table_is_exact_partition_local_and_uses_integer_bounds() -> None:
    catalog = build_rule_catalog()
    partitions = build_rule_identity_partitions()
    table = build_eligible_pair_table()
    manifest = table.as_obj()
    assert len(table.pairs) == 85_486
    assert manifest["minimum_disagreement_count"] == 4_801
    assert manifest["maximum_disagreement_count"] == 8_915
    assert sum(table.counts.values()) == len(table.pairs)
    assert table.digest == "c236bec6dc0362f45863a5854fe4df38e09d1e201be1f7eea8789c3ccb6a50a8"
    for pair in table.pairs:
        target = catalog[pair.target_index]
        shadow = catalog[pair.shadow_index]
        assert type(target.rule) is BinaryRule
        assert type(shadow.rule) is Literal
        assert partitions.for_entry(target) == pair.partition
        assert partitions.for_entry(shadow) == pair.partition
        assert min(pair.cells) >= 128
        assert 4_801 <= pair.disagreement_count <= 8_915


def test_episode_parser_round_trips_existing_schema_v2_fixture_transcript_episode(
    hidden_episode: HiddenEpisode,
) -> None:
    encoded = serialize_hidden_episode(hidden_episode)
    parsed = parse_hidden_episode(encoded)
    assert parsed == hidden_episode
    assert parsed.digest == hidden_episode.digest

    tampered = json.loads(encoded)
    tampered["target_truth_digest"] = "0" * 64
    with pytest.raises(ValueError, match="identity or truth digest"):
        parse_hidden_episode(json.dumps(tampered, separators=(",", ":")))


def test_exact_bounded_minimax_depth_proves_capacity_and_fixture_depth(
    hidden_episode: HiddenEpisode,
) -> None:
    # Forty-seven binary hypotheses cannot be separated in four answers.
    assert len(hidden_episode.opening_version_space()) == 47
    assert exact_minimax_identification_depth(
        hidden_episode.opening_version_space(), max_depth=4
    ) is None


def test_small_bank_covers_registered_strata_regimes_kinds_renderers_and_partitions(
    small_bank: EpisodeBank,
) -> None:
    assert len(small_bank.episodes) == 12
    assert small_bank.digest == "54714717b307b142a123372f5bde9854fd42df6ba8dc4e8e9c79bb78fe231631"
    assert small_bank.formula_counts == {"all": 3, "any": 3, "exactly_one": 6}
    assert small_bank.partition_counts == {partition: 2 for partition in RULE_PARTITIONS}
    assert Counter(episode.regime for episode in small_bank.episodes) == {
        "perfect_ambiguity": 6,
        "noisy_shortcuts": 6,
    }
    assert Counter(episode.terminal_kind for episode in small_bank.episodes) == {
        "train_like": 6,
        "factorial": 6,
    }
    assert Counter(episode.renderer for episode in small_bank.episodes) == {
        renderer: 2 for renderer in RENDERERS
    }
    assert Counter(
        (episode.regime, episode.terminal_kind) for episode in small_bank.episodes
    ) == {
        (regime, terminal_kind): 3
        for regime in ("perfect_ambiguity", "noisy_shortcuts")
        for terminal_kind in ("train_like", "factorial")
    }
    assert Counter(
        request.noisy_placard_error_target
        for request in small_bank.spec.requests
        if request.regime == "noisy_shortcuts"
    ) == {False: 3, True: 3}
    for partition in RULE_PARTITIONS:
        requests = [
            request for request in small_bank.spec.requests if request.partition == partition
        ]
        assert {request.regime for request in requests} == {
            "perfect_ambiguity",
            "noisy_shortcuts",
        }
        assert {request.terminal_kind for request in requests} == {
            "train_like",
            "factorial",
        }
        assert Counter(
            "exactly_one" if request.target_op == "exactly_one" else "all_or_any"
            for request in requests
        ) == {"all_or_any": 1, "exactly_one": 1}
    assert len({episode.target.truth_digest for episode in small_bank.episodes}) == 12
    assert all(
        record.version_space_size == len(episode.opening_version_space())
        and 8 <= record.version_space_size <= 64
        and record.minimax_depth <= 4
        and record.reference_query_count <= 6
        for episode, record in zip(
            small_bank.episodes, small_bank.generation_records, strict=True
        )
    )
    assert {request.target_op for request in small_bank.spec.requests} == set(BINARY_OPS)


def test_every_generated_episode_reference_expert_identifies_without_repetition(
    small_bank: EpisodeBank,
) -> None:
    for episode, record in zip(
        small_bank.episodes, small_bank.generation_records, strict=True
    ):
        assert exact_minimax_identification_depth(
            episode.opening_version_space(), max_depth=4
        ) == record.minimax_depth
        inquiry = run_reference_inquiry(episode)
        assert inquiry.target_identified
        assert len(inquiry.queries) == record.reference_query_count
        observed = {observation.scene_index for observation in episode.opening}
        for query in inquiry.queries:
            assert query.choice.scene_index not in observed
            observed.add(query.choice.scene_index)


def test_bank_manifest_round_trips_and_regenerates_byte_for_byte(
    small_bank: EpisodeBank,
) -> None:
    encoded = serialize_episode_bank(small_bank)
    fixture_path = (
        Path(__file__).with_name("fixtures") / "g03-engine-small-fixture-v1.json"
    )
    stored = fixture_path.read_text(encoding="utf-8")
    assert stored == encoded + "\n"
    parsed = parse_episode_bank(stored.removesuffix("\n"))
    assert parsed == small_bank
    assert serialize_episode_bank(parsed) == encoded


def test_bank_parser_rejects_tampered_attestation(small_bank: EpisodeBank) -> None:
    value = json.loads(serialize_episode_bank(small_bank))
    value["generation_records"][0]["pair_rank"] += 1
    tampered = json.dumps(value, separators=(",", ":"))
    with pytest.raises(ValueError, match="pair-rank provenance"):
        parse_episode_bank(tampered)
