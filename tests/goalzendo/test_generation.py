from __future__ import annotations

from collections import Counter

import pytest

from goalzendo.generation import (
    generate_factorial_evaluation,
    generate_known_law_dataset,
)
from goalzendo.rendering import render_known_law_decision
from goalzendo.rules import evaluate_rule
from goalzendo.schema import Choice, RuleSpec


def _semantic_digests(dataset: object) -> set[str]:
    return {
        koan.scene.semantic_digest
        for decision in dataset.decisions  # type: ignore[attr-defined]
        for koan in decision.koans
    }


def _error_counts(dataset: object) -> tuple[int, int, int]:
    decisions = dataset.decisions  # type: ignore[attr-defined]
    p_errors = sum(item.choice_p != item.choice_y for item in decisions)
    q_errors = sum(item.choice_q != item.choice_y for item in decisions)
    joint = sum(
        item.choice_p != item.choice_y and item.choice_q != item.choice_y
        for item in decisions
    )
    return p_errors, q_errors, joint


def test_known_law_generation_is_exact_balanced_and_deterministic() -> None:
    rule = RuleSpec("parity", (0, 1, 2))
    sage_rule = RuleSpec("parity", (3, 4), name="sage_rule")
    kwargs = dict(
        n=100,
        seed=31415,
        rule=rule,
        sage_rule=sage_rule,
        q_p=0.8,
        q_q=0.9,
        feature_count=8,
        error_geometry="independent",
    )
    first = generate_known_law_dataset(**kwargs)
    second = generate_known_law_dataset(**kwargs)

    assert first == second
    assert first.manifest_digest == second.manifest_digest
    assert first.manifest()["schema_version"] == 2
    assert first.manifest()["generator_version"] == 2
    assert len(first) == 100
    assert first.realized_q_p == 0.8
    assert first.realized_q_q == 0.9
    assert _error_counts(first) == (20, 10, 2)
    assert Counter(item.choice_y for item in first.decisions) == {
        Choice.A: 50,
        Choice.B: 50,
    }
    assert len({item.sample_id for item in first.decisions}) == 100

    for decision in first.decisions:
        law_values = [evaluate_rule(rule, koan.scene.features) for koan in decision.koans]
        assert sum(law_values) == 1
        assert sum(koan.herald_accepts for koan in decision.koans) == 1
        sage_values = [evaluate_rule(sage_rule, koan.scene.features) for koan in decision.koans]
        assert sum(sage_values) == 1
        assert decision.choice_q == (Choice.A if sage_values[0] else Choice.B)
        assert all(type(value) is bool for koan in decision.koans for value in koan.scene.features)


def test_joint_error_geometry_changes_overlap_without_changing_marginals() -> None:
    rule = RuleSpec("majority", (0, 1, 2))
    sage_rule = RuleSpec("literal", (3,), name="sage_rule")
    common = dict(
        n=100,
        seed=19,
        rule=rule,
        sage_rule=sage_rule,
        q_p=0.8,
        q_q=0.9,
        feature_count=5,
    )
    independent = generate_known_law_dataset(**common, error_geometry="independent")
    nested = generate_known_law_dataset(**common, error_geometry="nested")
    disjoint = generate_known_law_dataset(**common, error_geometry="disjoint")
    custom = generate_known_law_dataset(
        **common,
        error_geometry="custom",
        joint_error_count=7,
    )

    assert _error_counts(independent) == (20, 10, 2)
    assert _error_counts(nested) == (20, 10, 10)
    assert _error_counts(disjoint) == (20, 10, 0)
    assert _error_counts(custom) == (20, 10, 7)
    # Sample identities stay matched. Scenes also stay matched wherever the
    # semantic Q target is unchanged; cells whose Q target changes must be
    # resampled to remain on the Sage rule's support.
    assert [item.sample_id for item in independent.decisions] == [
        item.sample_id for item in nested.decisions
    ]
    for left, right in zip(independent.decisions, nested.decisions, strict=True):
        if left.choice_q == right.choice_q:
            assert tuple(koan.scene for koan in left.koans) == tuple(
                koan.scene for koan in right.koans
            )


def test_factorial_evaluation_contains_all_eight_candidate_tuples() -> None:
    dataset = generate_factorial_evaluation(
        repeats=3,
        seed=23,
        rule=RuleSpec("conjunction", (0, 1)),
        sage_rule=RuleSpec("parity", (2, 3), name="sage_rule"),
        feature_count=5,
    )
    counts = Counter(item.candidate_tuple for item in dataset.decisions)
    assert len(dataset) == 24
    assert len(counts) == 8
    assert set(counts.values()) == {3}
    assert dataset.realized_q_p == dataset.realized_q_q == 0.5
    assert dataset.joint_error_count == 6


def test_factorial_mirrors_swap_complete_blocks_and_preserve_exact_cells() -> None:
    dataset = generate_factorial_evaluation(
        repeats=4,
        seed=2301,
        rule=RuleSpec("parity", (0, 1, 2)),
        sage_rule=RuleSpec("parity", (3, 4), name="sage_rule"),
        feature_count=9,
        mirror_pairs=True,
        unique_semantic_scenes=True,
    )
    assert set(Counter(item.candidate_tuple for item in dataset.decisions).values()) == {4}
    pairs: dict[str, dict[str, object]] = {}
    for decision in dataset.decisions:
        assert decision.mirror_pair_id is not None
        assert decision.mirror_role is not None
        pairs.setdefault(decision.mirror_pair_id, {})[decision.mirror_role] = decision
    assert len(pairs) == 16
    for roles in pairs.values():
        assert set(roles) == {"base", "mirror"}
        base = roles["base"]
        mirror = roles["mirror"]
        assert mirror.koans == (base.koans[1], base.koans[0])
        assert mirror.candidate_tuple == tuple(
            choice.opposite() for choice in base.candidate_tuple
        )
    # Each relation deliberately reuses two scenes; unrelated pairs never do.
    assert len(_semantic_digests(dataset)) == 2 * len(pairs)


def test_unique_generation_honors_semantic_forbidden_support_deterministically() -> None:
    rule = RuleSpec("parity", (0, 1, 2))
    sage = RuleSpec("parity", (3, 4), name="sage_rule")
    train = generate_known_law_dataset(
        n=40,
        seed=91,
        rule=rule,
        sage_rule=sage,
        q_p=0.8,
        q_q=0.9,
        feature_count=9,
        unique_semantic_scenes=True,
    )
    forbidden = _semantic_digests(train)
    kwargs = dict(
        repeats=2,
        seed=92,
        rule=rule,
        sage_rule=sage,
        feature_count=9,
        mirror_pairs=True,
        unique_semantic_scenes=True,
        forbidden_scene_digests=forbidden,
    )
    first = generate_factorial_evaluation(**kwargs)
    second = generate_factorial_evaluation(**kwargs)
    assert first == second
    assert not (_semantic_digests(first) & forbidden)


def test_natural_and_nonce_renderings_encode_same_symbolic_decision() -> None:
    dataset = generate_factorial_evaluation(
        repeats=1,
        seed=29,
        rule=RuleSpec("parity", (0, 1, 2)),
        sage_rule=RuleSpec("literal", (3,), name="sage_rule"),
        feature_count=5,
    )
    decision = dataset.decisions[0]
    natural = render_known_law_decision(decision, dataset.feature_names, style="natural")
    nonce = render_known_law_decision(decision, dataset.feature_names, style="nonce")
    assert "Official Law:" in natural
    assert "Herald stamp:" in natural
    assert "Sage rule:" in natural
    assert "Sage recommendation:" not in natural
    # Sage information is a single rule in the shared header, never a revealed
    # per-koan judgment. A shared legend plus fixed-width state vectors avoids
    # repeating long natural-language clauses for both koans.
    koan_body = natural.split("Koan A:", maxsplit=1)[1]
    assert "Sage" not in koan_body
    assert natural.count("Feature key:") == 1
    assert natural.count("Feature states:") == 2
    state_lines = [line for line in natural.splitlines() if line.startswith("Feature states:")]
    assert all(len(line.split()) == len(dataset.feature_names) + 2 for line in state_lines)
    assert "LAW:" in nonce
    assert "dax" in nonce and "blick" in nonce
    assert decision.choice_y.label not in natural.split("Official Law:", maxsplit=1)[0]
    assert "correct koan" not in natural.lower()
    assert natural == render_known_law_decision(decision, dataset.feature_names, style="natural")


def test_registered_mirror_rendering_swaps_the_entire_koan_blocks() -> None:
    dataset = generate_factorial_evaluation(
        repeats=1,
        seed=2901,
        rule=RuleSpec("parity", (0, 1, 2)),
        sage_rule=RuleSpec("parity", (3, 4), name="sage_rule"),
        feature_count=7,
        mirror_pairs=True,
    )
    pair_id = dataset.decisions[0].mirror_pair_id
    pair = [item for item in dataset.decisions if item.mirror_pair_id == pair_id]
    base = next(item for item in pair if item.mirror_role == "base")
    mirror = next(item for item in pair if item.mirror_role == "mirror")
    base_text = render_known_law_decision(base, dataset.feature_names, style="natural")
    mirror_text = render_known_law_decision(mirror, dataset.feature_names, style="natural")

    def bodies(text: str) -> tuple[str, str]:
        left = text.split("Koan A:\n", maxsplit=1)[1].split("\n\nKoan B:\n", maxsplit=1)
        return left[0], left[1].split("\n\nChoose Koan", maxsplit=1)[0]

    base_a, base_b = bodies(base_text)
    mirror_a, mirror_b = bodies(mirror_text)
    assert (mirror_a, mirror_b) == (base_b, base_a)


def test_inexact_accuracy_and_infeasible_overlap_are_rejected() -> None:
    rule = RuleSpec("literal", (0,))
    sage_rule = RuleSpec("literal", (1,), name="sage_rule")
    with pytest.raises(ValueError, match="not exactly realizable"):
        generate_known_law_dataset(
            n=12,
            seed=1,
            rule=rule,
            sage_rule=sage_rule,
            q_p=0.9,
            q_q=0.5,
        )
    with pytest.raises(ValueError, match="feasible interval"):
        generate_known_law_dataset(
            n=100,
            seed=1,
            rule=rule,
            sage_rule=sage_rule,
            q_p=0.9,
            q_q=0.9,
            error_geometry="custom",
            joint_error_count=11,
        )


def test_law_and_sage_must_use_disjoint_scene_features() -> None:
    with pytest.raises(ValueError, match="disjoint"):
        generate_factorial_evaluation(
            repeats=1,
            seed=7,
            rule=RuleSpec("parity", (0, 1)),
            sage_rule=RuleSpec("literal", (1,), name="sage_rule"),
            feature_count=3,
        )


@pytest.mark.parametrize(
    ("rule", "sage_rule", "feature_count"),
    [
        (RuleSpec("literal", (0,)), RuleSpec("literal", (1,), name="sage"), 3),
        (RuleSpec("parity", (0, 1)), RuleSpec("parity", (2, 3), name="sage"), 5),
        (
            RuleSpec("majority", (0, 1, 2)),
            RuleSpec("majority", (3, 4, 5), name="sage"),
            7,
        ),
        (
            RuleSpec("conjunction", (0, 1, 2, 3, 4, 5, 6, 7)),
            RuleSpec("conjunction", (8, 9, 10, 11, 12, 13, 14, 15), name="sage"),
            17,
        ),
        (
            RuleSpec("multiplexer", (0, 1, 2)),
            RuleSpec("multiplexer", (3, 4, 5), name="sage"),
            7,
        ),
    ],
)
def test_joint_conditioning_supports_every_rule_family_without_rare_event_failure(
    rule: RuleSpec,
    sage_rule: RuleSpec,
    feature_count: int,
) -> None:
    dataset = generate_factorial_evaluation(
        repeats=2,
        seed=73,
        rule=rule,
        sage_rule=sage_rule,
        feature_count=feature_count,
    )
    assert len(dataset) == 16
    assert len(Counter(item.candidate_tuple for item in dataset.decisions)) == 8
    for decision in dataset.decisions:
        assert sum(evaluate_rule(rule, koan.scene.features) for koan in decision.koans) == 1
        assert sum(evaluate_rule(sage_rule, koan.scene.features) for koan in decision.koans) == 1
