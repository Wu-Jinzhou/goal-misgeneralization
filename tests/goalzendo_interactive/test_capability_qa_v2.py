from __future__ import annotations

from collections import Counter

import pytest

from goalzendo_interactive.capability import generate_episode_bank_v2
from goalzendo_interactive.capability_qa_v2 import (
    CAPABILITY_V2_SURFACE_FEATURE_KEYS,
    CAPABILITY_V2_SURFACE_TARGETS,
    REGISTERED_BINARY_GROUP_FLOOR,
    CandidateAuditV2,
    audit_hidden_episode_bank_v2_surface_leakage,
    bank_distribution_metrics_v2,
    build_candidate_audit_v2,
    build_powered_stage_specs_v2,
    build_symmetric_stage_specs_v2,
    candidate_identity_capacity_v2,
    hidden_episode_bank_v2_surface_targets,
)
from goalzendo_interactive.leakage import (
    SurfaceLeakageConfig,
    audit_surface_target,
    minimum_resolution_groups,
)


@pytest.fixture(scope="module")
def fast_audit_config() -> SurfaceLeakageConfig:
    return SurfaceLeakageConfig(bootstrap_replicates=1_000)


@pytest.fixture(scope="module")
def registered_audit(fast_audit_config: SurfaceLeakageConfig) -> CandidateAuditV2:
    return build_candidate_audit_v2("registered_256", fast_audit_config)


@pytest.fixture(scope="module")
def powered_audit(fast_audit_config: SurfaceLeakageConfig) -> CandidateAuditV2:
    return build_candidate_audit_v2("powered_402", fast_audit_config)


def test_full_registered_256_banks_are_exact_but_power_fails_closed(
    registered_audit: CandidateAuditV2,
) -> None:
    assert registered_audit.warm.episode_count == 256
    assert registered_audit.capability.episode_count == 256
    assert registered_audit.warm.spec_digest == (
        "68fac96d0881f3f62a6b456f5328323b9c9396700ffede97aebf8b5d2b201dc3"
    )
    assert registered_audit.warm.bank_digest == (
        "20541d996c3ab9f3b9d1618aaba99e3b5f9ddd4ff2e0d42cbcb4f2a862c28e56"
    )
    assert registered_audit.capability.spec_digest == (
        "b3f443a715d7c99a8e7d8a6c4372845d8b62cc3035f4b02c3fb1226f6b4cbb3e"
    )
    assert registered_audit.capability.bank_digest == (
        "7d5aa1a1d6542d84d29fcbb3e908f72732f6cbd8a45fa63037bde58bff19ac59"
    )
    assert registered_audit.warm.digest == (
        "857d318c8625e214f0a03ebdb6ff5a41027fe022be86d3e3d77d225c74a59b02"
    )
    assert registered_audit.capability.digest == (
        "bca3463bb3fc79bf0b52c14a2635455670f05c1287513b81baee5efa60875c27"
    )
    assert registered_audit.structural_distributions_passed
    assert registered_audit.identity_capacity_passed
    assert not registered_audit.leakage_powered
    assert not registered_audit.surface_leakage_passed
    assert registered_audit.blocker_codes[0] == "surface_leakage_underpowered:238<381"
    assert all(
        result.decision == "insufficient_data"
        for report in (
            registered_audit.warm_surface_leakage,
            registered_audit.capability_surface_leakage,
        )
        for result in report.results
    )
    assert not registered_audit.production_bank_generation_authorized
    assert not registered_audit.weight_updates_authorized


def test_v2_adapter_rebuilds_only_allowlisted_surface_features(
    registered_audit: CandidateAuditV2,
) -> None:
    from goalzendo_interactive.capability import build_stage_request_plan_v2

    bank = generate_episode_bank_v2(build_stage_request_plan_v2().warm_start)
    targets = hidden_episode_bank_v2_surface_targets(bank)
    assert tuple(target.name for target in targets) == CAPABILITY_V2_SURFACE_TARGETS

    forbidden = {
        "sun",
        "moon",
        "fits",
        "accepted",
        "scene_index",
        "target_rule",
        "shadow_rule",
        "placard",
        "proxy_low",
        "proxy_high",
        "chance_balanced",
        "oracle_diagnostic",
        "red",
        "blue",
        "green",
        "pyramid",
        "cube",
        "sphere",
    }
    for target in targets:
        for sample in target.samples:
            assert {feature.split("=", 1)[0] for feature in sample.features} <= (
                CAPABILITY_V2_SURFACE_FEATURE_KEYS
            )
            assert not forbidden.intersection(
                token.lower()
                for feature in sample.features
                for token in feature.replace("=", " ").split()
            )
    del registered_audit


def test_symmetric_384_candidate_is_preserved_and_rejected_for_366_formula_groups(
    fast_audit_config: SurfaceLeakageConfig,
) -> None:
    warm_spec, capability_spec = build_symmetric_stage_specs_v2()
    assert len(warm_spec.requests) == len(capability_spec.requests) == 384
    assert Counter(request.target_op for request in warm_spec.requests) == {
        "all": 96,
        "any": 96,
        "exactly_one": 192,
    }
    assert Counter(request.target_op for request in capability_spec.requests) == {
        "all": 122,
        "any": 122,
        "exactly_one": 122,
        None: 18,
    }
    assert capability_spec.digest == (
        "00ca6dce721621ba7602e3a5694e72ea5491ba2f380f6dc47245aa6d06010580"
    )
    bank = generate_episode_bank_v2(capability_spec)
    assert bank.digest == (
        "6355901ec220b7b9ac2b4a0e18f22df12aaaec6224266ac7870788b849381afc"
    )
    metrics = bank_distribution_metrics_v2(bank)
    assert metrics.digest == (
        "386c3b3728812af8bb5b1edd8dbb58b35a52f38db7df2d1580c6f356e43a3cf2"
    )
    formula = hidden_episode_bank_v2_surface_targets(bank)[1]
    assert len(formula.samples) == 366
    assert Counter(sample.label for sample in formula.samples) == {
        "all_or_any": 244,
        "exactly_one": 122,
    }
    result = audit_surface_target(formula, config=fast_audit_config)
    assert result.decision == "insufficient_data"
    assert result.group_count == 366
    assert (
        "independent_group_count_below_resolution_minimum:366<381" in result.reasons
    )


def test_powered_402_capacity_and_exact_renderer_orientation_marginals() -> None:
    warm_spec, capability_spec = build_powered_stage_specs_v2()
    assert len(warm_spec.requests) == 384
    assert len(capability_spec.requests) == 402
    assert Counter(request.target_op for request in capability_spec.requests) == {
        "all": 96,
        "any": 96,
        "exactly_one": 192,
        None: 18,
    }
    assert Counter(request.target_family for request in capability_spec.requests) == {
        "binary_piece": 384,
        "literal_piece": 16,
        "placard_literal": 2,
    }
    assert Counter(request.renderer for request in capability_spec.requests) == {
        "eval_reverse": 201,
        "eval_ledger": 201,
    }
    assert Counter(request.proxy_orientation for request in capability_spec.requests) == {
        "proxy_low": 201,
        "proxy_high": 201,
    }
    capacity = candidate_identity_capacity_v2(warm_spec, capability_spec)
    assert [(row.requested, row.available) for row in capacity] == [
        (96, 215),
        (96, 212),
        (192, 263),
        (96, 215),
        (96, 211),
        (192, 263),
        (16, 16),
        (2, 2),
    ]
    assert all(row.sufficient for row in capacity)


def test_powered_capability_is_powered_and_passes_but_warm_surface_gate_does_not(
    powered_audit: CandidateAuditV2,
) -> None:
    assert powered_audit.warm.episode_count == 384
    assert powered_audit.capability.episode_count == 402
    assert powered_audit.warm.spec_digest == (
        "d8324767746e57370fa8687542d1258a29d752407ad41634c6d4b4f4976f2a04"
    )
    assert powered_audit.warm.bank_digest == (
        "bcd476b14e97450efe3d1a63564c09b724596387fed30a33fb0ff00b80b8fe33"
    )
    assert powered_audit.capability.spec_digest == (
        "0fe8ee6cd20f98acaa09315dd559ed11211b2149490c44867bce69ee844bc0fa"
    )
    assert powered_audit.capability.bank_digest == (
        "a31790e1d215c833d95ba32ba0987df34681151ab557ff5f3cb570aff3f20bfd"
    )
    assert powered_audit.warm.digest == (
        "aad615a26641cabe19efc4531803086c5c6742f1ef6e885e2c06efc6116ba7f6"
    )
    assert powered_audit.capability.digest == (
        "35d3e2c4f54857887a4f2a35d66a0f03b5f698bee51b7c0d25098388a5f8cfb2"
    )
    assert powered_audit.structural_distributions_passed
    assert powered_audit.identity_capacity_passed
    assert powered_audit.leakage_powered
    assert powered_audit.capability_surface_leakage.passed
    assert all(
        result.decision == "pass"
        for result in powered_audit.capability_surface_leakage.results
    )
    formula = powered_audit.capability_surface_leakage.results[1]
    assert formula.group_count == REGISTERED_BINARY_GROUP_FLOOR + 3 == 384
    assert dict(formula.class_group_counts) == {
        "all_or_any": 192,
        "exactly_one": 192,
    }

    assert not powered_audit.warm_surface_leakage.passed
    assert not powered_audit.surface_leakage_passed
    assert powered_audit.blocker_codes[0] == "surface_leakage_gate_failed"
    assert "semantic_distribution_acceptance_rule_not_preregistered" in (
        powered_audit.blocker_codes
    )
    assert not powered_audit.semantic_comparison.acceptance_rule_preregistered
    assert not powered_audit.semantic_comparison.passed
    assert not powered_audit.production_bank_generation_authorized
    assert not powered_audit.weight_updates_authorized


def test_registered_resolution_floor_remains_exact() -> None:
    assert minimum_resolution_groups(2) == REGISTERED_BINARY_GROUP_FLOOR == 381


def test_direct_powered_capability_adapter_matches_candidate_report(
    powered_audit: CandidateAuditV2,
    fast_audit_config: SurfaceLeakageConfig,
) -> None:
    _, spec = build_powered_stage_specs_v2()
    bank = generate_episode_bank_v2(spec)
    direct = audit_hidden_episode_bank_v2_surface_leakage(
        bank, config=fast_audit_config
    )
    assert direct == powered_audit.capability_surface_leakage
