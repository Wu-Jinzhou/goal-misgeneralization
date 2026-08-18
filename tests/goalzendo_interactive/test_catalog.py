from __future__ import annotations

import ast
from pathlib import Path

import pytest

from goalzendo_interactive import (
    MAX_TRUE_COUNT,
    MIN_TRUE_COUNT,
    Atom,
    BinaryRule,
    Literal,
    build_rule_catalog,
    scene_at,
    truth_vector,
    verify_truth_vector,
)


@pytest.fixture(scope="module")
def catalog():  # type: ignore[no-untyped-def]
    return build_rule_catalog()


def test_catalog_is_complete_filtered_deduplicated_and_golden(catalog) -> None:  # type: ignore[no-untyped-def]
    assert catalog.stats.syntactic_rule_count == 18_760
    assert catalog.stats.excluded_prevalence_count == 7_884
    assert catalog.stats.extensional_duplicate_count == 3_576
    assert catalog.stats.retained_rule_count == len(catalog) == 7_300
    assert {
        catalog.stats.excluded_prevalence_count,
        catalog.stats.extensional_duplicate_count,
        catalog.stats.retained_rule_count,
    } != {0}
    assert sum(
        (
            catalog.stats.excluded_prevalence_count,
            catalog.stats.extensional_duplicate_count,
            catalog.stats.retained_rule_count,
        )
    ) == catalog.stats.syntactic_rule_count
    assert catalog.digest == "a796ef24d4e0eb2a2e129e12ee9cc3261c82b578f415e554d6608feafc9ae2d0"


def test_every_retained_vector_is_unique_and_within_prevalence_bounds(catalog) -> None:  # type: ignore[no-untyped-def]
    bits = [entry.truth.bits for entry in catalog]
    digests = [entry.truth_digest for entry in catalog]
    assert len(bits) == len(set(bits)) == len(catalog)
    assert len(digests) == len(set(digests)) == len(catalog)
    assert all(MIN_TRUE_COUNT <= entry.truth.true_count <= MAX_TRUE_COUNT for entry in catalog)
    assert all(entry.rule_id == f"g03r{entry.index:05d}" for entry in catalog)
    assert catalog.by_truth_digest(catalog[123].truth_digest) is catalog[123]


def test_semantic_duplicates_resolve_to_the_canonical_representative(catalog) -> None:  # type: ignore[no-untyped-def]
    red = Literal(Atom("slot_attr", position="left", attribute="color", value="red"))
    occupied = Literal(Atom("slot_empty", position="left"), negated=True)
    redundant = BinaryRule("all", (red, occupied))
    entry = catalog.equivalent_entry(redundant)
    assert entry is not None
    assert entry.rule == red
    assert entry.truth.bits == truth_vector(redundant).bits

    blue = Literal(Atom("exists", attribute="color", value="blue"))
    xor = BinaryRule("exactly_one", (red, blue))
    complemented_xor = BinaryRule(
        "exactly_one",
        (Literal(red.atom, negated=True), Literal(blue.atom, negated=True)),
    )
    assert truth_vector(xor) == truth_vector(complemented_xor)
    assert catalog.equivalent_entry(xor) == catalog.equivalent_entry(complemented_xor)


def test_constants_and_out_of_bound_rules_are_excluded(catalog) -> None:  # type: ignore[no-untyped-def]
    red = Literal(Atom("exists", attribute="color", value="red"))
    not_red = Literal(red.atom, negated=True)
    contradiction = BinaryRule("all", (red, not_red))
    tautology = BinaryRule("any", (red, not_red))
    rare = BinaryRule(
        "all",
        (
            Literal(Atom("slot_attr", position="left", attribute="color", value="red")),
            Literal(Atom("slot_attr", position="right", attribute="color", value="red")),
        ),
    )
    assert truth_vector(contradiction).true_count == 0
    assert truth_vector(tautology).true_count == len(truth_vector(tautology))
    assert truth_vector(rare).true_count < MIN_TRUE_COUNT
    assert catalog.equivalent_entry(contradiction) is None
    assert catalog.equivalent_entry(tautology) is None
    assert catalog.equivalent_entry(rare) is None


def test_catalog_vectors_have_independent_semantic_spot_checks(catalog) -> None:  # type: ignore[no-untyped-def]
    chosen = (catalog[0], catalog[1], catalog[100], catalog[1_000], catalog[-1])
    assert all(verify_truth_vector(entry.rule, entry.truth) for entry in chosen)


def test_version_space_updates_are_exact_stable_and_nonmutating(catalog) -> None:  # type: ignore[no-untyped-def]
    initial = catalog.version_space()
    first_scene = scene_at(0)
    rejected_count, accepted_count = initial.label_counts(first_scene)
    accepted = initial.observe(first_scene, True)
    rejected = initial.observe(0, False)
    assert len(initial) == len(catalog)
    assert len(accepted) == accepted_count
    assert len(rejected) == rejected_count
    assert len(accepted) + len(rejected) == len(initial)
    assert all(entry.truth[0] for entry in accepted)
    assert all(not entry.truth[0] for entry in rejected)
    assert accepted.indices == tuple(sorted(accepted.indices))

    second_index = 13715
    twice = initial.observe_many(((first_scene, True), (second_index, False)))
    assert twice == accepted.observe(second_index, False)
    assert all(entry.truth[0] and not entry.truth[second_index] for entry in twice)
    with pytest.raises(ValueError, match="Boolean"):
        initial.observe(0, 1)  # type: ignore[arg-type]


def test_interactive_package_has_no_legacy_a_b_scorer_import_or_reference() -> None:
    package = Path(__file__).parents[2] / "src" / "goalzendo_interactive"
    imported_modules: set[str] = set()
    source = ""
    for path in sorted(package.glob("*.py")):
        text = path.read_text(encoding="utf-8")
        source += text
        tree = ast.parse(text, filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported_modules.update(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module is not None:
                imported_modules.add(node.module)
    assert not any(name == "goalzendo" or name.startswith("goalzendo.") for name in imported_modules)
    assert "ActionScores" not in source
    assert "score_actions" not in source
