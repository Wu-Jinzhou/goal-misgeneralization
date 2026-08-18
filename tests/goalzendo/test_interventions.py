from __future__ import annotations

from dataclasses import replace

import pytest

from goalzendo.generation import generate_factorial_evaluation
from goalzendo.interventions import (
    audit_intervention,
    changed_channels,
    flip_distractor,
    flip_herald,
    flip_law,
    flip_sage,
    is_one_atom_law_boundary,
    paired_channel_flips,
    paired_interventions,
    paired_rule_preserving_controls,
    preserve_law,
    preserve_sage,
)
from goalzendo.rules import evaluate_rule
from goalzendo.schema import KnownLawDecision, Koan, RuleSpec, Scene


def _decision() -> KnownLawDecision:
    return generate_factorial_evaluation(
        repeats=1,
        seed=41,
        rule=RuleSpec("parity", (0, 1, 2)),
        sage_rule=RuleSpec("parity", (3, 4), name="sage_rule"),
        feature_count=6,
    ).decisions[3]


def _assert_on_manifold(decision: KnownLawDecision) -> None:
    assert sum(evaluate_rule(decision.law, koan.scene.features) for koan in decision.koans) == 1
    assert sum(
        evaluate_rule(decision.sage_rule, koan.scene.features) for koan in decision.koans
    ) == 1
    assert sum(koan.herald_accepts for koan in decision.koans) == 1


def test_herald_flip_changes_only_p_and_reverses_its_choice() -> None:
    original = _decision()
    changed = flip_herald(original)
    audit = audit_intervention(original, changed)
    assert audit.changed_channels == {"P"}
    assert audit.changed_feature_groups == set()
    assert audit.changed_feature_indices == ((), ())
    assert audit.herald_label_changed is True
    assert changed.choice_p == original.choice_p.opposite()
    assert changed.choice_q == original.choice_q
    assert changed.choice_y == original.choice_y
    assert changed.sample_id == original.sample_id
    assert changed.intervention == "flip_P"
    assert [koan.scene for koan in changed.koans] == [koan.scene for koan in original.koans]
    _assert_on_manifold(changed)


def test_sage_flip_edits_only_q_features_and_preserves_y() -> None:
    original = _decision()
    changed = flip_sage(original)
    audit = audit_intervention(original, changed)
    assert changed_channels(original, changed) == {"Q"}
    assert audit.changed_feature_groups == {"Q"}
    assert all(indices for indices in audit.changed_feature_indices)
    assert all(
        set(indices) <= set(original.sage_rule.feature_indices)
        for indices in audit.changed_feature_indices
    )
    assert changed.choice_q == original.choice_q.opposite()
    assert changed.choice_p == original.choice_p
    assert changed.choice_y == original.choice_y
    assert changed.intervention == "flip_Q"
    _assert_on_manifold(changed)


def test_law_flip_edits_only_y_features_and_preserves_q() -> None:
    original = _decision()
    changed = flip_law(original)
    audit = audit_intervention(original, changed)
    assert audit.changed_channels == {"Y"}
    assert audit.changed_feature_groups == {"Y"}
    assert all(
        set(indices) <= set(original.law.feature_indices)
        for indices in audit.changed_feature_indices
    )
    assert changed.choice_y == original.choice_y.opposite()
    assert changed.choice_q == original.choice_q
    assert changed.choice_p == original.choice_p
    _assert_on_manifold(changed)


def test_distractor_control_changes_neither_candidate_rule() -> None:
    original = _decision()
    changed = flip_distractor(original)
    audit = audit_intervention(original, changed)
    assert audit.changed_channels == set()
    assert audit.changed_feature_groups == {"distractor"}
    assert audit.changed_feature_indices == ((5,), (5,))
    assert changed.candidate_tuple == original.candidate_tuple
    _assert_on_manifold(changed)


def test_majority_intervention_fails_closed_away_from_one_atom_boundary() -> None:
    law = RuleSpec("majority", (0, 1, 2))
    sage_rule = RuleSpec("literal", (3,), name="sage_rule")
    decision = KnownLawDecision(
        sample_id="majority-boundary",
        law=law,
        sage_rule=sage_rule,
        koans=(
            Koan("majority:A", Scene("majority-scene:A", (True, True, True, True, False)), True),
            Koan("majority:B", Scene("majority-scene:B", (False, False, False, False, True)), False),
        ),
        split="test",
    )
    with pytest.raises(ValueError, match="one-atom boundary"):
        flip_law(decision)


def test_majority_intervention_uses_exactly_one_atom_on_boundary() -> None:
    law = RuleSpec("majority", (0, 1, 2))
    sage_rule = RuleSpec("literal", (3,), name="sage_rule")
    decision = KnownLawDecision(
        sample_id="majority-on-boundary",
        law=law,
        sage_rule=sage_rule,
        koans=(
            Koan("boundary:A", Scene("boundary-scene:A", (True, True, False, True, False)), True),
            Koan("boundary:B", Scene("boundary-scene:B", (True, False, False, False, True)), False),
        ),
        split="test",
    )
    assert is_one_atom_law_boundary(decision)
    changed = flip_law(decision)
    audit = audit_intervention(decision, changed)
    assert tuple(len(indices) for indices in audit.changed_feature_indices) == (1, 1)
    assert changed.choice_y == decision.choice_y.opposite()
    assert changed.choice_q == decision.choice_q
    _assert_on_manifold(changed)


def test_each_matched_semantic_edit_preserves_the_other_rule_exactly() -> None:
    decisions = generate_factorial_evaluation(
        repeats=8,
        seed=4102,
        rule=RuleSpec("parity", (0, 1, 2)),
        sage_rule=RuleSpec("majority", (3, 4, 5), name="sage_rule"),
        feature_count=9,
    ).decisions
    for original in decisions:
        panel = paired_interventions(original)
        audits = {name: audit_intervention(original, value) for name, value in panel.items()}
        assert audits["flip_P"].changed_feature_indices == ((), ())
        assert audits["flip_P"].changed_channels == {"P"}
        assert audits["flip_Q"].changed_channels == {"Q"}
        assert audits["flip_Q"].changed_feature_groups == {"Q"}
        assert audits["flip_Y"].changed_channels == {"Y"}
        assert audits["flip_Y"].changed_feature_groups == {"Y"}
        assert audits["flip_D"].changed_channels == set()
        assert audits["flip_D"].changed_feature_groups == {"distractor"}
        assert all(audit.on_manifold for audit in audits.values())


def test_keyed_minimum_edits_are_deterministic_and_cover_eligible_positions() -> None:
    decisions = generate_factorial_evaluation(
        repeats=32,
        seed=9127,
        rule=RuleSpec("parity", (0, 1, 2)),
        sage_rule=RuleSpec("parity", (3, 4), name="sage_rule"),
        feature_count=9,
    ).decisions
    law_indices: set[int] = set()
    sage_indices: set[int] = set()
    for original in decisions:
        first_law = flip_law(original)
        first_sage = flip_sage(original)
        assert first_law == flip_law(original)
        assert first_sage == flip_sage(original)
        law_audit = audit_intervention(original, first_law)
        sage_audit = audit_intervention(original, first_sage)
        assert all(len(indices) == 1 for indices in law_audit.changed_feature_indices)
        assert all(len(indices) == 1 for indices in sage_audit.changed_feature_indices)
        law_indices.update(index for indices in law_audit.changed_feature_indices for index in indices)
        sage_indices.update(index for indices in sage_audit.changed_feature_indices for index in indices)
    assert law_indices == {0, 1, 2}
    assert sage_indices == {3, 4}


def test_registered_mirrors_receive_the_same_koan_keyed_edits() -> None:
    decisions = generate_factorial_evaluation(
        repeats=8,
        seed=3187,
        rule=RuleSpec("parity", (0, 1, 2)),
        sage_rule=RuleSpec("parity", (3, 4), name="sage_rule"),
        feature_count=9,
        mirror_pairs=True,
    ).decisions
    pairs: dict[str, dict[str, KnownLawDecision]] = {}
    for decision in decisions:
        assert decision.mirror_pair_id is not None and decision.mirror_role is not None
        pairs.setdefault(decision.mirror_pair_id, {})[decision.mirror_role] = decision
    for roles in pairs.values():
        base, mirror = roles["base"], roles["mirror"]
        for intervention in (flip_law, flip_sage, flip_distractor):
            base_changed = intervention(base)
            mirror_changed = intervention(mirror)
            base_by_koan = {
                koan.koan_id: koan.scene.features for koan in base_changed.koans
            }
            mirror_by_koan = {
                koan.koan_id: koan.scene.features for koan in mirror_changed.koans
            }
            assert base_by_koan == mirror_by_koan


def test_rule_preserving_active_feature_controls_do_not_change_candidates() -> None:
    original = _decision()
    law_control = preserve_law(original)
    sage_control = preserve_sage(original)
    for changed, group in ((law_control, "Y"), (sage_control, "Q")):
        audit = audit_intervention(original, changed)
        assert audit.changed_channels == set()
        assert audit.changed_feature_groups == {group}
        assert changed.candidate_tuple == original.candidate_tuple
        _assert_on_manifold(changed)
    assert paired_rule_preserving_controls(original) == {
        "preserve_Y": law_control,
        "preserve_Q": sage_control,
    }


def test_literal_rule_preserving_controls_are_omitted_fail_closed() -> None:
    original = generate_factorial_evaluation(
        repeats=1,
        seed=91,
        rule=RuleSpec("literal", (0,)),
        sage_rule=RuleSpec("literal", (1,), name="sage_rule"),
        feature_count=3,
    ).decisions[0]
    assert paired_rule_preserving_controls(original) == {}
    with pytest.raises(ValueError, match="no reachable nonempty edit"):
        preserve_law(original)


def test_complete_intervention_panel_and_compatibility_helper() -> None:
    original = _decision()
    assert set(paired_channel_flips(original)) == {"flip_P", "flip_Q"}
    panel = paired_interventions(original)
    assert set(panel) == {"flip_P", "flip_Q", "flip_Y", "flip_D"}
    assert {name: audit_intervention(original, value).changed_channels for name, value in panel.items()} == {
        "flip_P": {"P"},
        "flip_Q": {"Q"},
        "flip_Y": {"Y"},
        "flip_D": set(),
    }
    assert paired_interventions(original) == panel


def test_distractor_requires_an_inactive_feature() -> None:
    decision = generate_factorial_evaluation(
        repeats=1,
        seed=11,
        rule=RuleSpec("literal", (0,)),
        sage_rule=RuleSpec("literal", (1,), name="sage_rule"),
        feature_count=2,
    ).decisions[0]
    with pytest.raises(ValueError, match="inactive"):
        flip_distractor(decision)


def test_audit_rejects_rule_or_identity_changes() -> None:
    original = _decision()
    with pytest.raises(ValueError, match="Law and Sage"):
        changed = replace(
            original,
            sage_rule=replace(original.sage_rule, output_negated=True),
        )
        audit_intervention(original, changed)
