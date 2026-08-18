from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import replace

import pytest

from goalzendo_interactive.generation import EpisodeBankSpec, EpisodeRequest
from goalzendo_interactive.production_banks import (
    ENGINE_LEAKAGE_EPISODES,
    EVALUATION_EPISODES_PER_VIEW,
    PRODUCTION_STAGES,
    ProductionGenerationUnit,
    ProductionStage,
    build_production_bank_plan,
    build_production_bank_qa_report,
    requests_for_stage,
    serialize_production_bank_plan,
)


def test_canonical_plan_is_request_only_deterministic_and_non_authorizing() -> None:
    plan = build_production_bank_plan()
    report = build_production_bank_qa_report(plan)

    assert plan is build_production_bank_plan()
    assert plan.planned_request_count == 2_816
    assert plan.generated_episode_count == 0
    assert not plan.materialization_authorized
    assert plan.digest == "c0a8766a1b7013a7190823e1ad84ff3e8b69c1b8035397fc50b02814e2e5d704"
    assert report.digest == "5476c8d03bfe684b6a915335460e3d0831d3ecdb27d626e1776f05a32a90d8d8"
    assert report.structural_checks_passed
    assert not report.materialization_authorized
    assert report.request_count == plan.planned_request_count
    assert serialize_production_bank_plan(plan) == serialize_production_bank_plan(plan)


def test_every_representable_stage_has_exact_joint_registered_balance() -> None:
    plan = build_production_bank_plan()
    expected_counts: dict[ProductionStage, int] = {
        "engine_leakage": 768,
        "pilot": 256,
        "confirmatory_train": 768,
        "validation": 256,
        "evaluation_active": 256,
        "evaluation_oracle_replay": 256,
        "evaluation_no_query": 256,
    }
    for stage, count in expected_counts.items():
        requests = requests_for_stage(plan, stage)
        assert len(requests) == count
        assert Counter(request.target_op for request in requests) == {
            "all": count // 4,
            "any": count // 4,
            "exactly_one": count // 2,
        }
        assert Counter(request.regime for request in requests) == {
            "perfect_ambiguity": count // 2,
            "noisy_shortcuts": count // 2,
        }
        noisy = [request for request in requests if request.regime == "noisy_shortcuts"]
        assert Counter(request.noisy_placard_error_target for request in noisy) == {
            False: count // 4,
            True: count // 4,
        }

    engine = requests_for_stage(plan, "engine_leakage")
    assert Counter(request.renderer for request in engine) == {
        "train_compact": 192,
        "train_positional": 192,
        "train_tabletop": 192,
        "train_inventory": 192,
    }
    assert Counter(request.terminal_kind for request in engine) == {
        "train_like": 384,
        "factorial": 384,
    }
    evaluation_stages: tuple[ProductionStage, ...] = (
        "evaluation_active",
        "evaluation_oracle_replay",
        "evaluation_no_query",
    )
    for stage in evaluation_stages:
        requests = requests_for_stage(plan, stage)
        assert Counter(request.renderer for request in requests) == {
            "eval_reverse": 128,
            "eval_ledger": 128,
        }


def test_leakage_resolution_and_unique_identity_capacity_are_fail_closed() -> None:
    plan = build_production_bank_plan()
    report = build_production_bank_qa_report(plan)
    assert plan.leakage_minimum_independent_groups == 381
    engine = requests_for_stage(plan, "engine_leakage")
    assert len(engine) == ENGINE_LEAKAGE_EPISODES
    noisy = [request for request in engine if request.regime == "noisy_shortcuts"]
    assert len(noisy) == 384
    assert Counter(request.noisy_placard_error_target for request in noisy) == {
        False: 192,
        True: 192,
    }
    assert report.leakage_group_resolution_satisfied
    assert all(item.sufficient for item in report.identity_capacity)
    assert {
        (item.partition, item.target_op): (item.requested, item.available)
        for item in report.identity_capacity
    } == {
        ("engineering", "all"): (192, 354),
        ("engineering", "any"): (192, 359),
        ("engineering", "exactly_one"): (384, 426),
        ("pilot", "all"): (64, 372),
        ("pilot", "any"): (64, 351),
        ("pilot", "exactly_one"): (128, 441),
        ("confirmatory_train", "all"): (192, 347),
        ("confirmatory_train", "any"): (192, 381),
        ("confirmatory_train", "exactly_one"): (384, 415),
        ("validation", "all"): (64, 383),
        ("validation", "any"): (64, 337),
        ("validation", "exactly_one"): (128, 420),
        ("evaluation", "all"): (192, 352),
        ("evaluation", "any"): (192, 332),
        ("evaluation", "exactly_one"): (384, 465),
    }


def test_one_generation_unit_per_partition_prevents_reservation_reset() -> None:
    plan = build_production_bank_plan()
    partition_units: dict[str, set[str]] = defaultdict(set)
    all_request_ids: list[str] = []
    for unit in plan.units:
        for request in unit.spec.requests:
            partition_units[request.partition].add(unit.unit_id)
            all_request_ids.append(request.request_id)
    assert all(len(unit_ids) == 1 for unit_ids in partition_units.values())
    assert len(all_request_ids) == len(set(all_request_ids))

    evaluation_slices = [
        item for item in plan.slices if item.stage.startswith("evaluation_")
    ]
    assert len(evaluation_slices) == 3
    assert {item.unit_id for item in evaluation_slices} == {
        "g03-production-evaluation-suite-v1"
    }
    assert {item.episode_count for item in evaluation_slices} == {
        EVALUATION_EPISODES_PER_VIEW
    }


def test_unsupported_required_stages_are_explicit_and_cannot_return_requests() -> None:
    plan = build_production_bank_plan()
    assert {item.stage for item in plan.blocked_stages} == {
        "format_warm_start",
        "capability",
    }
    represented = tuple(item.stage for item in plan.slices)
    blocked = tuple(item.stage for item in plan.blocked_stages)
    assert len(represented) + len(blocked) == len(PRODUCTION_STAGES)
    assert set((*represented, *blocked)) == set(PRODUCTION_STAGES)
    assert tuple(item.code for item in plan.required_extensions) == (
        "unsupported_chance_balanced_proxy_profile",
        "missing_capability_rule_partition",
        "unsupported_capability_official_law_families",
    )
    with pytest.raises(ValueError, match="chance_balanced"):
        requests_for_stage(plan, "format_warm_start")
    with pytest.raises(ValueError, match="missing_capability_rule_partition"):
        requests_for_stage(plan, "capability")


def test_validator_rejects_cross_unit_partition_reuse_before_generation() -> None:
    plan = build_production_bank_plan()
    source = requests_for_stage(plan, "evaluation_active")[0]
    extra_request = replace(source, request_id="g03-production-request-v1-extra-evaluation")
    extra_spec = EpisodeBankSpec(
        bank_id="g03-production-extra-evaluation-v1",
        requests=(extra_request,),
    )
    tampered = replace(
        plan,
        units=(
            *plan.units,
            ProductionGenerationUnit("g03-production-extra-evaluation-v1", extra_spec),
        ),
    )
    with pytest.raises(ValueError, match=r"contiguously|one combined generation unit"):
        build_production_bank_qa_report(tampered)


def test_validator_rejects_a_balancing_tamper() -> None:
    plan = build_production_bank_plan()
    pilot_unit_index = next(
        index for index, unit in enumerate(plan.units) if unit.unit_id.endswith("pilot-v1")
    )
    pilot_unit = plan.units[pilot_unit_index]
    requests = list(pilot_unit.spec.requests)
    first = requests[0]
    requests[0] = EpisodeRequest(
        request_id=first.request_id,
        partition=first.partition,
        target_op=first.target_op,
        regime=first.regime,
        terminal_kind=first.terminal_kind,
        renderer="train_positional",
        noisy_placard_error_target=first.noisy_placard_error_target,
    )
    changed_unit = ProductionGenerationUnit(
        pilot_unit.unit_id,
        replace(pilot_unit.spec, requests=tuple(requests)),
    )
    units = list(plan.units)
    units[pilot_unit_index] = changed_unit
    with pytest.raises(ValueError, match="renderer balance"):
        build_production_bank_qa_report(replace(plan, units=tuple(units)))
