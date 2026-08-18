"""Fail-closed construction checks for E19's paired pathway edits."""

from __future__ import annotations

import copy

import pytest
import torch

from forkworld.competing import make_competing_factorial_dataset
from forkworld.handoff import stable_state_digest
from forkworld.models import GoalMLP, MLPConfig
from forkworld.q_pathway import (
    EXPECTED_FEATURE_NAMES,
    EXPECTED_STATE_SHAPES,
    PADDING_SHAM_COLUMN_INDICES,
    Q_COLUMN_INDICES,
    Q_PATHWAY_BRANCHES,
    audit_q_pathway_branches,
    audit_q_pathway_contract,
    make_q_pathway_branches,
    preactivation_edit_effects,
)


def _donors() -> tuple[GoalMLP, GoalMLP, GoalMLP]:
    torch.manual_seed(19)
    initial = GoalMLP(
        MLPConfig(
            input_dim=19,
            width=64,
            depth=2,
            activation="relu",
            residual=False,
            bias=True,
        )
    )
    independent = copy.deepcopy(initial)
    nested = copy.deepcopy(initial)
    with torch.no_grad():
        independent.input_projection.weight[:, 8:10].add_(
            torch.linspace(-0.10, 0.10, 128).reshape(64, 2)
        )
        nested.input_projection.weight[:, 8:10].add_(
            torch.linspace(0.05, -0.05, 128).reshape(64, 2)
        )
        independent.input_projection.weight[:, 2:7].add_(0.007)
        nested.input_projection.weight[:, 2:7].sub_(0.003)
        independent.hidden_layers[0].bias.add_(0.01)
        nested.hidden_layers[0].bias.sub_(0.01)
        independent.goal_head.bias.add_(0.02)
        nested.goal_head.bias.sub_(0.02)
    return initial, independent, nested


def _branches():
    initial, independent, nested = _donors()
    branches = make_q_pathway_branches(
        initial,
        independent,
        nested,
        feature_names=EXPECTED_FEATURE_NAMES,
    )
    return initial, independent, nested, branches


def test_contract_fixes_exact_feature_order_and_state_shapes() -> None:
    initial, independent, nested = _donors()
    audit = audit_q_pathway_contract(
        {"initial": initial, "independent": independent, "nested": nested},
        EXPECTED_FEATURE_NAMES,
    )

    assert audit["q_column_indices"] == [8, 9]
    assert audit["q_column_names"] == ["Q_1", "Q_2"]
    assert audit["padding_sham_column_indices"] == [5, 6]
    assert audit["padding_sham_column_names"] == ["R_4", "R_5"]
    assert audit["state_shapes"] == {
        name: list(shape) for name, shape in EXPECTED_STATE_SHAPES.items()
    }


def test_six_branches_are_deep_clones_with_exact_column_edits() -> None:
    initial, independent, nested, branches = _branches()
    q_index = torch.tensor(Q_COLUMN_INDICES)
    sham_index = torch.tensor(PADDING_SHAM_COLUMN_INDICES)
    wi = independent.input_projection.weight.detach()
    wn = nested.input_projection.weight.detach()
    w0 = initial.input_projection.weight.detach()

    assert tuple(branches.models) == Q_PATHWAY_BRANCHES
    assert branches.models["independent_noop"] is not independent
    assert branches.models["nested_noop"] is not nested
    assert torch.equal(
        branches.models["independent_q_restore"].input_projection.weight.index_select(
            1, q_index
        ),
        w0.index_select(1, q_index),
    )
    assert torch.equal(
        branches.models["nested_q_transplant"].input_projection.weight.index_select(
            1, q_index
        ),
        wi.index_select(1, q_index),
    )
    assert torch.equal(
        branches.models["independent_padding_sham"]
        .input_projection.weight.index_select(1, sham_index),
        wi.index_select(1, sham_index) + branches.restore_delta,
    )
    assert torch.equal(
        branches.models["nested_padding_sham"].input_projection.weight.index_select(
            1, sham_index
        ),
        wn.index_select(1, sham_index) + branches.transplant_delta,
    )

    branch_hash = stable_state_digest(branches.models["independent_q_restore"].state_dict())
    with torch.no_grad():
        branches.models["independent_q_restore"].goal_head.bias.add_(1.0)
    assert stable_state_digest(independent.state_dict()) == branches.donor_state_digests[
        "independent"
    ]
    assert branch_hash != stable_state_digest(
        branches.models["independent_q_restore"].state_dict()
    )


def test_audit_reports_donor_unchanged_state_and_delta_evidence() -> None:
    initial, independent, nested, branches = _branches()
    audit = audit_q_pathway_branches(
        branches,
        initial_model=initial,
        independent_model=independent,
        nested_model=nested,
    )

    assert audit["donors_unchanged"] is True
    assert audit["all_six_branches_verified"] is True
    assert audit["all_unaffected_state_exact"] is True
    assert audit["all_edits_target_exactly_128_scalars"] is True
    assert audit["branches"]["independent_noop"]["target_scalar_count"] == 0
    for name in (
        "independent_q_restore",
        "independent_padding_sham",
        "nested_q_transplant",
        "nested_padding_sham",
    ):
        assert audit["branches"][name]["target_scalar_count"] == 128
        assert audit["branches"][name]["unchanged_state_verified"] is True
    for edit in ("restore", "transplant"):
        item = audit["edits"][edit]
        assert item["same_intended_delta_applied"] is True
        assert item["intended_delta"]["shape"] == [64, 2]
        assert item["intended_delta"]["target_scalar_count"] == 128
        assert item["intended_delta"]["finite"] is True
        assert item["intended_delta"]["l1_norm"] > 0
        assert item["intended_delta"]["l2_norm"] > 0
        assert item["intended_delta"]["linf_norm"] > 0


def test_preactivation_helper_matches_rms_dose_on_exhaustive_panel() -> None:
    _, _, _, branches = _branches()
    panel = make_competing_factorial_dataset(
        64,
        k_q=2,
        k_y=3,
        seed=29,
        control_seed=31,
        max_k_q=3,
        max_k_y=5,
        state_dim=8,
    )
    effects = preactivation_edit_effects(branches, panel)

    assert effects["panel_n"] == 64
    assert effects["all_values_finite"] is True
    comparisons = effects["comparisons"]
    # The exact weight displacement is matched; finite padding streams need not
    # have the same covariance as the exhaustive active Q bits.  The helper
    # records the resulting activation-level dose instead of assuming equality.
    for active, sham in (
        ("independent_q_restore", "independent_padding_sham"),
        ("nested_q_transplant", "nested_padding_sham"),
    ):
        ratio = (
            comparisons[active]["preactivation_delta_rms"]
            / comparisons[sham]["preactivation_delta_rms"]
        )
        assert 0.5 < ratio < 2.0
    for item in comparisons.values():
        assert item["preactivation_delta_rms"] > 0
        assert 0 <= item["relu_state_flip_rate"] <= 1
        assert item["activation_count"] == 64 * 64
        assert item["preactivation_delta_digest"]


def test_feature_order_and_architecture_drift_fail_closed() -> None:
    initial, independent, nested = _donors()
    wrong_names = list(EXPECTED_FEATURE_NAMES)
    wrong_names[8], wrong_names[9] = wrong_names[9], wrong_names[8]
    with pytest.raises(ValueError, match="feature order"):
        make_q_pathway_branches(
            initial,
            independent,
            nested,
            feature_names=wrong_names,
        )

    wrong_width = GoalMLP(
        MLPConfig(input_dim=19, width=32, depth=2, activation="relu")
    )
    with pytest.raises(ValueError, match="width-64"):
        audit_q_pathway_contract(
            {"wrong_width": wrong_width}, EXPECTED_FEATURE_NAMES
        )


def test_branch_donor_and_delta_mutations_fail_closed() -> None:
    initial, independent, nested, branches = _branches()
    with torch.no_grad():
        branches.models["nested_padding_sham"].hidden_layers[0].bias[0] += 0.1
    with pytest.raises(RuntimeError, match="changed before the construction audit"):
        audit_q_pathway_branches(
            branches,
            initial_model=initial,
            independent_model=independent,
            nested_model=nested,
        )

    initial, independent, nested, branches = _branches()
    with torch.no_grad():
        independent.goal_head.bias[0] += 0.1
    with pytest.raises(RuntimeError, match="donor changed"):
        audit_q_pathway_branches(
            branches,
            initial_model=initial,
            independent_model=independent,
            nested_model=nested,
        )

    initial, independent, nested, branches = _branches()
    branches.restore_delta[0, 0] += 0.1
    with pytest.raises(RuntimeError, match="stored restore displacement"):
        audit_q_pathway_branches(
            branches,
            initial_model=initial,
            independent_model=independent,
            nested_model=nested,
        )


def test_preactivation_panel_interface_drift_fails_closed() -> None:
    _, _, _, branches = _branches()
    wrong_panel = make_competing_factorial_dataset(
        64,
        k_q=2,
        k_y=3,
        seed=37,
        control_seed=41,
        max_k_q=3,
        max_k_y=5,
        state_dim=0,
    )
    with pytest.raises(ValueError, match="panel feature interface"):
        preactivation_edit_effects(branches, wrong_panel)
