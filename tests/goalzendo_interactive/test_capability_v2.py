from __future__ import annotations

import json
from collections import Counter
from typing import cast

import pytest

from goalzendo_interactive.capability import (
    CapabilityV2Error,
    EpisodeBankSpecV2,
    EpisodeRequestV2,
    build_stage_request_plan_v2,
    generate_episode_bank_v2,
    parse_episode_request_v2,
    parse_hidden_episode_v2,
    parse_stage_request_plan_v2,
    serialize_episode_request_v2,
    serialize_hidden_episode_v2,
    serialize_stage_request_plan_v2,
    small_capability_bank_spec_v2,
    verify_episode_bank_v2,
)
from goalzendo_interactive.catalog import build_rule_catalog
from goalzendo_interactive.partitions import build_rule_identity_partitions
from goalzendo_interactive.rules import BinaryRule, Literal
from goalzendo_interactive.stage_partitions_v2 import (
    DEMAND_WEIGHTED_PARTITION_CYCLE_V2,
    LITERAL_SHADOW_PARTITION_CYCLE_V2,
    RULE_PARTITIONS_V2,
    TARGET_FAMILIES_V2,
    build_eligible_target_shadow_table_v2,
    build_rejected_equal_partition_audit_v2,
    build_rule_identity_partitions_v2,
    catalog_rule_family_v2,
    parse_rule_identity_partitions_v2,
    serialize_rule_identity_partitions_v2,
)


def test_v2_partition_is_total_disjoint_distinct_and_does_not_mutate_v1() -> None:
    catalog = build_rule_catalog()
    v1 = build_rule_identity_partitions()
    v2 = build_rule_identity_partitions_v2()

    assert v1.digest == "fce33a68179267db62ede4dd07b759e3e473397f41f812d4dd9ed2e848e2cd76"
    assert len(v2.assignments) == len(catalog) == 7_300
    assert len({item.rule_id for item in v2.assignments}) == len(catalog)
    assert len({item.truth_digest for item in v2.assignments}) == len(catalog)
    assert v2.counts == {
        "warm_start": 731,
        "engineering": 1_460,
        "capability": 740,
        "pilot": 729,
        "confirmatory_train": 1_458,
        "validation": 728,
        "evaluation": 1_454,
    }
    assert v2.digest == "54943bd1d82c6179d3fda63448e6169e0e3163da5374dc3d169137b85af7eb90"
    assert v2.as_obj()["demand_weighted_cycle"] == list(
        DEMAND_WEIGHTED_PARTITION_CYCLE_V2
    )
    assert v2.as_obj()["literal_shadow_cycle"] == list(
        LITERAL_SHADOW_PARTITION_CYCLE_V2
    )
    assert "no model outcome" in v2.as_obj()["weight_basis"]
    placard = [
        entry for entry in catalog if catalog_rule_family_v2(entry) == "placard_literal"
    ]
    assert len(placard) == 2
    assert {v2.for_entry(entry) for entry in placard} == {"capability"}
    encoded = serialize_rule_identity_partitions_v2(v2)
    assert parse_rule_identity_partitions_v2(encoded) is v2


def test_v2_generalized_pair_table_has_all_target_families_and_exact_capacity() -> None:
    catalog = build_rule_catalog()
    partitions = build_rule_identity_partitions_v2()
    table = build_eligible_target_shadow_table_v2()

    assert len(table.pairs) == 79_600
    assert table.digest == "1d615d5ba64fa7cc2e8edcbcc54b094e209d2a98b3980f4ff05fc9ddbf028131"
    assert {
        family: table.target_identity_counts[f"capability:{family}"]
        for family in TARGET_FAMILIES_V2
    } == {"binary_piece": 689, "literal_piece": 16, "placard_literal": 2}
    assert table.binary_operator_target_identity_counts == {
        "warm_start:all": 215,
        "warm_start:any": 212,
        "warm_start:exactly_one": 263,
        "engineering:all": 430,
        "engineering:any": 422,
        "engineering:exactly_one": 526,
        "capability:all": 215,
        "capability:any": 211,
        "capability:exactly_one": 263,
        "pilot:all": 215,
        "pilot:any": 211,
        "pilot:exactly_one": 263,
        "confirmatory_train:all": 430,
        "confirmatory_train:any": 422,
        "confirmatory_train:exactly_one": 526,
        "validation:all": 214,
        "validation:any": 211,
        "validation:exactly_one": 263,
        "evaluation:all": 428,
        "evaluation:any": 422,
        "evaluation:exactly_one": 524,
    }
    for pair in table.pairs:
        target = catalog[pair.target_index]
        shadow = catalog[pair.shadow_index]
        assert partitions.for_entry(target) == pair.partition
        assert partitions.for_entry(shadow) == pair.partition
        assert catalog_rule_family_v2(target) == pair.target_family
        assert catalog_rule_family_v2(shadow) == "literal_piece"
        assert min(pair.cells) >= 128
        assert 4_801 <= pair.disagreement_count <= 8_915


def test_v2_request_schema_names_profile_family_and_operator_exactly() -> None:
    request = EpisodeRequestV2(
        request_id="warm-example",
        stage="format_warm_start",
        partition="warm_start",
        target_family="binary_piece",
        target_op="all",
        proxy_profile="chance_balanced",
        proxy_orientation="proxy_low",
        renderer="train_compact",
    )
    encoded = serialize_episode_request_v2(request)
    assert parse_episode_request_v2(encoded) == request
    assert '"proxy_profile":"chance_balanced"' in encoded

    with pytest.raises(CapabilityV2Error, match="chance_balanced"):
        EpisodeRequestV2(
            request_id="warm-invalid",
            stage="format_warm_start",
            partition="warm_start",
            target_family="binary_piece",
            target_op="all",
            proxy_profile="oracle_diagnostic",
            proxy_orientation="proxy_low",
            renderer="train_compact",
        )
    with pytest.raises(CapabilityV2Error, match="target_op"):
        EpisodeRequestV2(
            request_id="literal-invalid",
            stage="capability",
            partition="capability",
            target_family="literal_piece",
            target_op="any",
            proxy_profile="oracle_diagnostic",
            proxy_orientation="proxy_low",
            renderer="eval_reverse",
        )


def test_small_capability_bank_uses_actual_piece_and_placard_official_laws() -> None:
    bank = generate_episode_bank_v2(small_capability_bank_spec_v2())
    partitions = build_rule_identity_partitions_v2()

    assert bank is generate_episode_bank_v2(small_capability_bank_spec_v2())
    assert bank.target_family_counts == {
        "binary_piece": 6,
        "literal_piece": 2,
        "placard_literal": 2,
    }
    assert bank.digest == "a8180e58d4a0746f488e5262a4e1990ec4ab4a63f3c14f5d4d160fdb34c73c9f"
    assert not bank.as_obj()["weight_updates_authorized"]
    assert verify_episode_bank_v2(bank) is bank
    assert len({episode.target.truth_digest for episode in bank.episodes}) == 10
    for episode in bank.episodes:
        assert partitions.for_entry(episode.target) == "capability"
        assert partitions.for_entry(episode.shadow) == "capability"
        assert len(episode.opening_version_space()) >= 1
        assert episode.target.index in episode.opening_version_space().indices
        for phase in (episode.opening, episode.terminal):
            assert len(phase) == 10
            assert sum(item.accepted for item in phase) == 5
            assert sum(
                episode.target.truth[item.scene_index]
                is episode.shadow.truth[item.scene_index]
                for item in phase
            ) == 5

    placard = [
        episode
        for episode in bank.episodes
        if episode.request.target_family == "placard_literal"
    ]
    assert {type(episode.target.rule) for episode in placard} == {Literal}
    assert {
        cast(Literal, episode.target.rule).negated for episode in placard
    } == {False, True}
    assert {
        cast(Literal, episode.target.rule).atom.op for episode in placard
    } == {"placard_is"}


def test_chance_balanced_warm_start_is_exact_per_episode_and_across_pair() -> None:
    requests = (
        EpisodeRequestV2(
            request_id="warm-small-low",
            stage="format_warm_start",
            partition="warm_start",
            target_family="binary_piece",
            target_op="all",
            proxy_profile="chance_balanced",
            proxy_orientation="proxy_low",
            renderer="train_compact",
        ),
        EpisodeRequestV2(
            request_id="warm-small-high",
            stage="format_warm_start",
            partition="warm_start",
            target_family="binary_piece",
            target_op="any",
            proxy_profile="chance_balanced",
            proxy_orientation="proxy_high",
            renderer="train_positional",
        ),
    )
    bank = generate_episode_bank_v2(
        EpisodeBankSpecV2("g03-v2-warm-small", requests, max_pair_attempts=512)
    )
    assert {episode.request.proxy_profile for episode in bank.episodes} == {
        "chance_balanced"
    }
    for episode in bank.episodes:
        for balance in (
            episode.as_obj()["opening_balance"],
            episode.as_obj()["terminal_balance"],
        ):
            assert balance["target_true"] == 5
            assert balance["target_placard_agreement"] == 5
            assert balance["target_shadow_agreement"] == 5
    assert sum(
        episode.as_obj()["opening_balance"]["placard_true"]
        for episode in bank.episodes
    ) == 10
    assert sum(
        episode.as_obj()["opening_balance"]["shadow_true"]
        for episode in bank.episodes
    ) == 10


def test_hidden_episode_v2_canonical_parser_rejects_derived_balance_tamper() -> None:
    episode = generate_episode_bank_v2(small_capability_bank_spec_v2()).episodes[0]
    encoded = serialize_hidden_episode_v2(episode)
    assert parse_hidden_episode_v2(encoded) == episode

    tampered = json.loads(encoded)
    tampered["opening_balance"]["target_true"] = 4
    with pytest.raises(CapabilityV2Error, match="derived fields"):
        parse_hidden_episode_v2(json.dumps(tampered, separators=(",", ":")))


def test_v2_request_plan_clears_capacity_but_remains_qa_fail_closed() -> None:
    plan = build_stage_request_plan_v2()
    assert plan is build_stage_request_plan_v2()
    assert plan.requested_episode_count == 512
    assert plan.generated_episode_count == 0
    assert not plan.production_bank_generation_authorized
    assert not plan.weight_updates_authorized
    assert plan.digest == "5de31e369f5b4324e2a4dab12f84a9551e6bbfa62ed3a5c562dd4fd6fb54b2f9"
    encoded = serialize_stage_request_plan_v2(plan)
    assert parse_stage_request_plan_v2(encoded) is plan

    warm = plan.warm_start.requests
    assert Counter(request.target_op for request in warm) == {
        "all": 64,
        "any": 64,
        "exactly_one": 128,
    }
    assert Counter(request.proxy_orientation for request in warm) == {
        "proxy_low": 128,
        "proxy_high": 128,
    }
    assert Counter(request.renderer for request in warm) == {
        "train_compact": 64,
        "train_positional": 64,
        "train_tabletop": 64,
        "train_inventory": 64,
    }

    capability = plan.capability.requests
    assert Counter(request.target_family for request in capability) == {
        "binary_piece": 238,
        "literal_piece": 16,
        "placard_literal": 2,
    }
    assert Counter(request.proxy_orientation for request in capability) == {
        "proxy_low": 128,
        "proxy_high": 128,
    }
    assert Counter(request.renderer for request in capability) == {
        "eval_reverse": 128,
        "eval_ledger": 128,
    }
    for family in TARGET_FAMILIES_V2:
        family_requests = [
            request for request in capability if request.target_family == family
        ]
        assert Counter(request.proxy_orientation for request in family_requests) == {
            "proxy_low": len(family_requests) // 2,
            "proxy_high": len(family_requests) // 2,
        }
        assert Counter(request.renderer for request in family_requests) == {
            "eval_reverse": len(family_requests) // 2,
            "eval_ledger": len(family_requests) // 2,
        }
    shortfalls = [item for item in plan.capacity if not item.sufficient]
    assert shortfalls == []
    assert all(item.sufficient for item in plan.capacity)
    assert all(
        item.available > item.requested
        for item in plan.capacity
        if item.target_family == "binary_piece"
    )
    assert "distribution_and_surface_leakage_qa_not_run" in plan.blocker_codes
    assert plan.as_obj()["required_qa"]["partition_distribution"]["status"] == (
        "not_run"
    )
    assert plan.as_obj()["required_qa"]["surface_leakage"]["status"] == "not_run"
    assert plan.blocker_codes[-1] == "no_weight_update_authorization"


def test_rejected_equal_split_remains_pinned_non_authorizing_evidence() -> None:
    audit = build_rejected_equal_partition_audit_v2()
    assert audit.digest == "0f80b5c06493964efc9f39365ff8c316779b669dca31d97195f7d365f6f24436"
    assert audit.as_obj()["partition_digest"] == (
        "6b9234d8c7920591e13fb0227a09a821c7a044d9af27b5b9b3b9046957fa4be0"
    )
    assert audit.as_obj()["eligible_pairs_digest"] == (
        "bb7cfab9369687aabf067aa20709312e9897d4b40a1ecde4ff077a612e63325d"
    )
    assert [
        (row.partition, row.op, row.available, row.requested)
        for row in audit.capacity
        if not row.sufficient
    ] == [
        ("engineering", "exactly_one", 367, 384),
        ("confirmatory_train", "exactly_one", 364, 384),
        ("evaluation", "exactly_one", 376, 384),
    ]
    assert audit.as_obj()["capability_control_shortfall"] == {
        "target_family": "literal_piece",
        "available": 12,
        "requested": 16,
        "sufficient": False,
    }
    assert not audit.as_obj()["production_bank_generation_authorized"]
    assert not audit.as_obj()["weight_updates_authorized"]


def test_stage_source_rule_domains_are_disjoint_by_construction() -> None:
    partitions = build_rule_identity_partitions_v2()
    table = build_eligible_target_shadow_table_v2()
    warm_targets = {
        pair.target_index
        for family in TARGET_FAMILIES_V2
        for pair in table.candidates("warm_start", family)
    }
    capability_targets = {
        pair.target_index
        for family in TARGET_FAMILIES_V2
        for pair in table.candidates("capability", family)
    }
    assert warm_targets
    assert capability_targets
    assert warm_targets.isdisjoint(capability_targets)
    assert all(
        partitions.for_entry(build_rule_catalog()[index]) == "warm_start"
        for index in warm_targets
    )
    assert all(
        partitions.for_entry(build_rule_catalog()[index]) == "capability"
        for index in capability_targets
    )


def test_unique_official_law_bound_fails_closed_for_four_placard_requests() -> None:
    requests = tuple(
        EpisodeRequestV2(
            request_id=f"too-many-placard-{index}",
            stage="capability",
            partition="capability",
            target_family="placard_literal",
            target_op=None,
            proxy_profile="oracle_diagnostic",
            proxy_orientation=("proxy_low" if index % 2 == 0 else "proxy_high"),
            renderer=("eval_reverse" if index % 2 == 0 else "eval_ledger"),
        )
        for index in range(4)
    )
    with pytest.raises(CapabilityV2Error, match="bounded v2 construction failed"):
        generate_episode_bank_v2(
            EpisodeBankSpecV2(
                "g03-v2-impossible-placard",
                requests,
                max_pair_attempts=4_096,
            )
        )


def test_capability_binary_targets_are_actual_binary_rules() -> None:
    bank = generate_episode_bank_v2(small_capability_bank_spec_v2())
    for episode in bank.episodes:
        if episode.request.target_family == "binary_piece":
            assert type(episode.target.rule) is BinaryRule
            assert episode.target.rule.op == episode.request.target_op
        else:
            assert type(episode.target.rule) is Literal


def test_v2_partition_names_include_capability_exactly_once() -> None:
    assert RULE_PARTITIONS_V2.count("capability") == 1
    assert "capability" not in build_rule_identity_partitions().counts
