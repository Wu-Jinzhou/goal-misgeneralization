from __future__ import annotations

from collections import Counter

import pytest

from goalzendo_interactive.capability_qa_v3 import CAPABILITY_V3_SURFACE_TARGETS
from goalzendo_interactive.capability_qa_v5 import (
    CandidateAuditV5,
    build_candidate_audit_v5,
    hidden_episode_bank_v5_surface_targets,
)
from goalzendo_interactive.capability_v3 import (
    CapabilityV3Error,
    build_prospective_design_v3,
    materialize_prospective_design_v3,
)
from goalzendo_interactive.capability_v4 import build_prospective_design_v4
from goalzendo_interactive.capability_v5 import (
    build_prospective_design_v5,
    materialize_prospective_design_v5,
)
from goalzendo_interactive.leakage import SurfaceLeakageConfig


def test_v3_freeze_is_preserved_as_a_construction_failure() -> None:
    design = build_prospective_design_v3()
    assert design.digest == (
        "51beb1a5cdfb21947a3ce09ae4c251c91034279c77ba425060112c9ba9dc1e39"
    )
    assert design.warm.digest == (
        "03cc7d8bc1e1e8f529344bf200d411de0e03443fca69eba24c6f25e8659a3796"
    )
    assert design.capability.digest == (
        "3c52be384e501b64dc386d4472d33e080456fefe0cb18db95e0187e56cc78e33"
    )
    with pytest.raises(
        CapabilityV3Error,
        match="eligible v3 pair lacks enough disjoint scenes",
    ):
        materialize_prospective_design_v3(design)


def test_v4_pair_population_repair_is_frozen_but_v4_evidence_remains_separate() -> None:
    design = build_prospective_design_v4()
    assert design.digest == (
        "12ed009bdcad9b5b1b4f5e07e8f5fe6e86049e020ae7c5e1d4287310a56a1740"
    )
    assert design.warm.digest == (
        "c011ff9d43c19b9836e66c7ca82b15374d2d9b7806d294b8abaac94c3dd90f3e"
    )
    assert design.capability.digest == (
        "645ed0009b1313fc5030b6ba998f9204145ab249ccc91cf32e83ac8985825a8a"
    )
    assert design.pair_population_audit.digest == (
        "9e3a98cdb40227718e66fc7a4075ddaa2e7bb7f8faa5a935be59bacf84cf7188"
    )
    assert design.pair_population_audit.reserve_per_requested_cell == 1
    assert design.pair_population_audit.passed
    mixed = next(
        row
        for row in design.pair_population_audit.rows
        if "binary_mixed_control" in row.stratum
    )
    assert (mixed.requested, mixed.eligible_target_identities, mixed.eligible_pairs) == (
        2,
        9,
        126,
    )


def test_v5_exact_outcome_fold_design_is_frozen_before_materialization() -> None:
    design = build_prospective_design_v5()
    assert design.digest == (
        "b4e66671ab58eadafc26fad831b08eaf3b4f1773f26bc98c57b683400c2ab427"
    )
    assert design.warm.digest == (
        "b5f45c44272bbabd084fa0169c0bea9d07f31256ececd3b0d26b50e22e340e15"
    )
    assert design.capability.digest == (
        "65769ec8477314165482f89c87b8fafbb87abbe17fa8199c6b433f6dd47d8625"
    )
    assert design.semantic_gate.digest == (
        "9850d6cd03746d48c634a013224784a832335cddd63c85af34cff3f52e301477"
    )
    assert design.outcome_fold_population.digest == (
        "c76081e9db8e6a7ac1100b0041099586d2acc83530cc0c0cf3faf671d9e2ec76"
    )
    assert design.outcome_fold_population.passed
    assert len(design.outcome_fold_population.rows) == 30
    assert Counter(request.target_op for request in design.warm.requests) == {
        "all": 100,
        "any": 100,
        "exactly_one": 200,
    }
    assert Counter(request.target_family for request in design.capability.requests) == {
        "binary_piece": 400,
        "literal_piece": 16,
        "binary_mixed_control": 2,
        "placard_literal": 2,
    }
    placards = tuple(
        request
        for request in design.capability.requests
        if request.target_family == "placard_literal"
    )
    assert len(placards) == 2
    assert {request.renderer for request in placards} == {"eval_reverse"}


@pytest.fixture(scope="module")
def fast_v5_audit() -> CandidateAuditV5:
    return build_candidate_audit_v5(
        SurfaceLeakageConfig(bootstrap_replicates=1_000)
    )


def test_v5_materialization_is_exact_powered_and_fully_gated(
    fast_v5_audit: CandidateAuditV5,
) -> None:
    audit = fast_v5_audit
    assert audit.warm.bank_digest == (
        "56415bba6e2fe15876dd957ac208750910a2f69c74605e07db55ba05fa1f6f74"
    )
    assert audit.capability.bank_digest == (
        "90441d8c53b2b52a365af2fe59474fdff8aca1f70a6b4a3e351f945add6ac4e6"
    )
    assert audit.warm.digest == (
        "30e1d39bdb9ae9fefbeaf861a74da91e96513242ce1eb55e2a4731dd0ae83259"
    )
    assert audit.capability.digest == (
        "8503488f56ba8f02134a9d96e5bd990c91877a6be0839ebc19cb77c733efda2e"
    )
    assert audit.warm.target_identity_count == 400
    assert audit.capability.target_identity_count == 420
    assert audit.stage_target_identities_disjoint
    assert audit.structural_invariants_passed
    assert audit.surface_leakage_passed
    assert all(
        result.group_count >= 400
        and result.balanced_accuracy == 0.5
        and result.interval_lower == result.interval_upper == 0.5
        and result.passed
        for report in (
            audit.warm_surface_leakage,
            audit.capability_surface_leakage,
        )
        for result in report.results
    )
    assert audit.semantic_gate.passed
    assert audit.semantic_gate.absolute_differences == (
        20_500,
        98,
        10,
        4,
        7,
        17,
        4,
        2,
        0,
        54_158,
        33_658,
        47_134,
        67_634,
    )
    assert audit.semantic_gate.upper_thresholds == (
        75_384,
        1_142,
        14,
        34,
        24,
        25,
        17,
        31,
        0,
        224_374,
        226_560,
        222_704,
        222_838,
    )
    assert audit.blocker_codes == (
        "model_pipeline_integration_not_audited",
        "production_bank_generation_not_authorized",
        "weight_updates_not_authorized",
    )
    assert not audit.production_bank_generation_authorized
    assert not audit.weight_updates_authorized


def test_v5_adapter_never_exposes_sensitive_content(
    fast_v5_audit: CandidateAuditV5,
) -> None:
    del fast_v5_audit
    candidate = materialize_prospective_design_v5(build_prospective_design_v5())
    forbidden = {
        "sun",
        "moon",
        "accepted",
        "scene_index",
        "target_rule",
        "shadow_rule",
        "placard",
        "proxy_low",
        "proxy_high",
        "chance_balanced",
        "oracle_diagnostic",
    }
    for bank in (candidate.warm, candidate.capability):
        targets = hidden_episode_bank_v5_surface_targets(bank)
        assert tuple(target.name for target in targets) == CAPABILITY_V3_SURFACE_TARGETS
        for target in targets:
            for sample in target.samples:
                assert not forbidden.intersection(
                    token.lower()
                    for feature in sample.features
                    for token in feature.replace("=", " ").split()
                )
