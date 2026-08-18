from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from fractions import Fraction
from typing import Any

import pytest

from goalzendo_interactive import VersionSpace, build_rule_catalog
from goalzendo_interactive_v2.hypothesis_complete import (
    POWERED_STRESS_MINIMUM_INDEPENDENT_BLOCKS,
    SCIENTIFIC_TRAINING_EPISODE_BUDGET,
    CanonicalSupportedOpeningV2,
    HypothesisCompleteTrainingBlockV2,
    HypothesisCompleteTrainingManifestV2,
    HypothesisCompleteV2Error,
    MaterializedTrainingPanelV2,
    UnconditionalTerminalSceneLawV2,
    build_canonical_supported_opening_v2,
    build_hidden_order_display_binding_v2,
    build_hypothesis_complete_training_block_v2,
    build_hypothesis_complete_training_manifest_v2,
    build_materialized_training_panel_v2,
    build_unconditional_terminal_scene_law_v2,
    hypothesis_complete_training_block_v2_from_obj,
    parse_hypothesis_complete_training_block_v2,
    parse_hypothesis_complete_training_manifest_v2,
    serialize_hypothesis_complete_training_block_v2,
    serialize_hypothesis_complete_training_manifest_v2,
    unconditional_terminal_scene_law_v2_from_obj,
    verify_hypothesis_complete_training_block_v2,
    verify_hypothesis_complete_training_manifest_v2,
)
from goalzendo_interactive_v2.population_audit import (
    build_supported_catalog_contract_v2,
    classify_catalog_identity_v2,
)
from goalzendo_interactive_v2.role_schema import EVIDENCE_GEOMETRIES, build_role_block_v2

# Deterministic search seed 20260811 found this ten-scene, five/five placard
# opening after 33 candidates.  Every identity below is recomputed in each test;
# no v1 episode metadata or hand-authored V0 is trusted.
_EIGHT_RULE_OPENING: tuple[tuple[int, bool], ...] = (
    (446, False),
    (1208, False),
    (2815, False),
    (3884, False),
    (5390, False),
    (7676, True),
    (10126, True),
    (11567, True),
    (12043, True),
    (12431, True),
)
_EXPECTED_V0 = (
    "g03r00016",
    "g03r01788",
    "g03r04659",
    "g03r04721",
    "g03r04747",
    "g03r05021",
    "g03r06197",
    "g03r07241",
)
_SCIENTIFIC_OPENING_SCENES: tuple[tuple[int, ...], ...] = (
    (446, 1208, 2815, 3884, 5390, 7676, 10126, 11567, 12043, 12431),
    (950, 1835, 2638, 3198, 4674, 6911, 7243, 8754, 11084, 13080),
    (1592, 2497, 3254, 3558, 6395, 11203, 11935, 11952, 12722, 13296),
    (1236, 1311, 3892, 3893, 5897, 8863, 10054, 10180, 11156, 12979),
    (2537, 3398, 3431, 4210, 5726, 9643, 10556, 11942, 12426, 13431),
    (775, 5214, 6016, 6816, 6818, 10480, 10626, 11111, 11683, 13617),
    (1779, 2175, 2865, 3674, 6177, 8023, 8870, 9536, 11987, 13483),
    (635, 2301, 2674, 5096, 5877, 7415, 12551, 12700, 12860, 13422),
    (211, 1170, 5750, 6156, 6820, 7408, 8312, 9589, 11939, 12186),
    (2169, 2736, 5225, 6161, 6683, 7271, 10784, 11165, 11350, 13025),
    (1446, 3123, 3261, 4232, 4265, 7295, 10180, 10758, 12336, 13096),
    (118, 1873, 2527, 6233, 6719, 9203, 9870, 12789, 12800, 13544),
    (25, 270, 465, 2362, 3559, 7853, 9694, 10223, 10751, 11689),
    (457, 705, 1505, 2571, 5212, 8973, 10295, 12361, 13209, 13388),
    (1804, 3816, 3970, 6152, 6809, 11509, 11698, 12085, 13025, 13436),
    (2756, 3339, 4530, 5496, 5681, 9132, 10316, 11995, 13576, 13688),
    (1137, 1784, 4731, 4854, 6660, 7436, 7688, 9225, 10164, 10351),
    (686, 4958, 5365, 5818, 5878, 7152, 7808, 9632, 10649, 10818),
    (357, 2226, 3590, 3723, 5944, 7049, 7609, 10138, 10265, 12588),
    (1654, 2132, 3213, 4451, 6514, 6961, 8413, 8434, 10788, 12022),
    (1085, 1328, 3606, 3672, 4586, 7085, 7787, 7803, 11634, 13137),
    (151, 361, 2607, 2703, 4349, 8733, 11487, 11770, 12661, 13664),
    (3866, 3934, 4524, 5733, 6618, 8764, 8788, 9291, 9406, 12064),
    (595, 1519, 2111, 2342, 5214, 6917, 7076, 8164, 8427, 13374),
    (333, 1152, 3392, 5172, 6507, 8368, 9848, 10410, 10788, 11265),
    (165, 1112, 2028, 2428, 6166, 8381, 9030, 9294, 10627, 11440),
    (320, 1322, 5360, 5619, 6650, 7002, 7436, 7529, 8587, 9277),
    (79, 4656, 5251, 6530, 6799, 7735, 8234, 8468, 9415, 11839),
    (3408, 4634, 5053, 5392, 5650, 8750, 10645, 11222, 13193, 13221),
    (2502, 4528, 4780, 6452, 6781, 7163, 7853, 9115, 9757, 10830),
    (734, 1775, 1805, 3920, 5772, 7348, 8097, 9799, 11968, 13455),
    (1207, 1334, 4800, 5071, 6789, 7379, 9140, 11311, 11819, 13496),
    (313, 2252, 2985, 4600, 6827, 8133, 8208, 10328, 11424, 13015),
    (502, 541, 1229, 6084, 6128, 11763, 11839, 11865, 12043, 13086),
    (5, 2626, 3045, 3287, 4898, 7084, 11016, 11243, 12113, 12486),
    (281, 672, 1169, 1272, 5580, 8285, 10609, 11307, 11559, 12419),
    (496, 1735, 5376, 5449, 6225, 9711, 10325, 12346, 13105, 13487),
    (566, 3082, 3660, 6229, 6448, 9238, 9802, 11426, 11711, 12739),
    (343, 1695, 2147, 2671, 6798, 7421, 9336, 10068, 11519, 12386),
    (6, 1283, 2291, 2630, 4557, 8246, 9049, 10487, 10525, 12040),
    (4504, 4925, 4933, 5420, 5510, 8208, 8316, 9288, 11887, 11894),
    (349, 1472, 1853, 2988, 5505, 8090, 10130, 10434, 10654, 11683),
    (957, 4184, 5250, 6162, 6737, 7290, 7612, 8255, 9050, 10985),
    (839, 1544, 3692, 5841, 6340, 6956, 7012, 7263, 9439, 10502),
    (1727, 2671, 3987, 4467, 5624, 7909, 8990, 9698, 12230, 13713),
    (67, 924, 1332, 5962, 6082, 8347, 12048, 12521, 12765, 12865),
    (239, 3477, 3512, 3740, 3948, 7131, 7335, 7670, 9563, 13653),
    (699, 1247, 1539, 2673, 4500, 7120, 9937, 10391, 10546, 11021),
)

_PANEL_INFEASIBLE_OPENING_SCENES = (
    635,
    698,
    759,
    5168,
    5223,
    6866,
    9208,
    9854,
    10732,
    12706,
)
_NESTED_RULE_OPENING_SCENES = (
    3866,
    3934,
    4524,
    5733,
    6618,
    8764,
    8788,
    9291,
    9406,
    12064,
)


def _digest(label: str) -> str:
    return hashlib.sha256(label.encode("ascii")).hexdigest()


def _opening(index: int = 0) -> CanonicalSupportedOpeningV2:
    catalog = build_rule_catalog()
    placard = catalog[16]
    scenes = _SCIENTIFIC_OPENING_SCENES[index]
    return build_canonical_supported_opening_v2(
        f"opening-{index:04d}",
        ((scene_index, placard.truth[scene_index]) for scene_index in scenes),
    )


def _opening_from_scenes(
    opening_id: str,
    scenes: tuple[int, ...],
) -> CanonicalSupportedOpeningV2:
    placard = build_rule_catalog()[16]
    return build_canonical_supported_opening_v2(
        opening_id,
        ((scene_index, placard.truth[scene_index]) for scene_index in scenes),
    )


def _terminal_fixture(
    opening: CanonicalSupportedOpeningV2,
) -> tuple[UnconditionalTerminalSceneLawV2, MaterializedTrainingPanelV2]:
    """Hash-select 16 complementary pairs, then hide eight pairs as the panel."""

    catalog = build_rule_catalog()
    entries = tuple(catalog[int(rule_id[4:])] for rule_id in opening.version_space_rule_ids)
    opening_scenes = {item.scene_index for item in opening.observations}
    scenes_by_pattern: dict[int, list[int]] = {}
    for scene_index in range(len(entries[0].truth)):
        if scene_index in opening_scenes:
            continue
        pattern = sum(1 << position for position, entry in enumerate(entries) if entry.truth[scene_index])
        if pattern.bit_count() == opening.n0 // 2:
            scenes_by_pattern.setdefault(pattern, []).append(scene_index)

    full_mask = (1 << opening.n0) - 1
    available_pairs = [
        (pattern, full_mask ^ pattern, left_scene, right_scene)
        for pattern in sorted(scenes_by_pattern)
        if pattern < (full_mask ^ pattern) and (full_mask ^ pattern) in scenes_by_pattern
        for left_scene, right_scene in zip(
            scenes_by_pattern[pattern],
            scenes_by_pattern[full_mask ^ pattern],
            strict=False,
        )
    ]
    if len(available_pairs) < 16:
        raise HypothesisCompleteV2Error(
            "canonical complementary-pair generator has fewer than sixteen scene pairs"
        )

    def rank(domain: str, pair: tuple[int, int, int, int]) -> str:
        payload = f"{domain}\0{opening.content_digest}\0{pair}".encode("ascii")
        return hashlib.sha256(payload).hexdigest()

    law_pairs = tuple(sorted(available_pairs, key=lambda pair: rank("law", pair))[:16])
    law_scene_indices = tuple(sorted(scene for pair in law_pairs for scene in (pair[2], pair[3])))
    law = build_unconditional_terminal_scene_law_v2(
        ((scene_index, Fraction(1, len(law_scene_indices))) for scene_index in law_scene_indices),
        public_derivation_attestation_digest=_digest(f"public-complementary-law:{opening.content_digest}"),
    )
    panel_pairs = tuple(sorted(law_pairs, key=lambda pair: rank("panel", pair))[:8])
    panel = build_materialized_training_panel_v2(
        opening,
        (scene for pair in panel_pairs for scene in (pair[2], pair[3])),
        external_generator_receipt_digest=_digest(f"hidden-panel-receipt:{opening.content_digest}"),
    )
    return law, panel


@pytest.fixture(scope="module")
def law() -> UnconditionalTerminalSceneLawV2:
    return _terminal_fixture(_opening())[0]


def _block(
    index: int,
    *,
    pre_digest: str | None = None,
    post_digest: str | None = None,
) -> HypothesisCompleteTrainingBlockV2:
    opening = _opening(index)
    law, panel = _terminal_fixture(opening)
    hidden = build_hidden_order_display_binding_v2(
        opening,
        independence_precommitment_digest=_digest(f"independent-order-{index}"),
    )
    return build_hypothesis_complete_training_block_v2(
        opening,
        law,
        panel,
        hidden,
        bank_position=index,
        block_id=f"block-{index:04d}",
        renderer_name="train_positional",
        pre_update_checkpoint_digest=(_digest(f"checkpoint-{index}") if pre_digest is None else pre_digest),
        post_update_checkpoint_digest=(
            _digest(f"checkpoint-{index + 1}") if post_digest is None else post_digest
        ),
        update_batch_id=f"batch-{index:04d}",
        optimizer_step_before=10_000 + index,
    )


@pytest.fixture(scope="module")
def block() -> HypothesisCompleteTrainingBlockV2:
    return _block(0)


@pytest.fixture(scope="module")
def scientific_manifest() -> HypothesisCompleteTrainingManifestV2:
    block_count = SCIENTIFIC_TRAINING_EPISODE_BUDGET // 8
    blocks = tuple(_block(index) for index in range(block_count))
    return build_hypothesis_complete_training_manifest_v2(blocks)


def test_actual_catalog_opening_recomputes_exact_supported_eight_rule_v0() -> None:
    catalog = build_rule_catalog()
    contract = build_supported_catalog_contract_v2()
    observations = tuple(reversed(_EIGHT_RULE_OPENING))
    opening = build_canonical_supported_opening_v2("opening-0000", observations)

    independently_recomputed = VersionSpace(catalog, contract.supported_indices).observe_many(
        _EIGHT_RULE_OPENING
    )
    assert opening.n0 == 8
    assert opening.version_space_rule_ids == _EXPECTED_V0
    assert opening.version_space_rule_ids == tuple(
        catalog[index].rule_id for index in independently_recomputed.indices
    )
    assert opening.catalog_digest == catalog.digest
    assert opening.supported_catalog_digest == contract.supported_catalog_digest
    assert tuple(item.scene_index for item in opening.observations) == tuple(
        scene_index for scene_index, _ in _EIGHT_RULE_OPENING
    )
    assert sum(item.accepted for item in opening.observations) == 5


def test_full_block_makes_unique_placard_feature_guess_exactly_chance(
    block: HypothesisCompleteTrainingBlockV2,
) -> None:
    catalog = build_rule_catalog()
    placard_candidates = tuple(
        rule_id
        for rule_id in block.opening.version_space_rule_ids
        if classify_catalog_identity_v2(catalog[int(rule_id[4:])]) == "placard_literal"
    )
    assert placard_candidates == ("g03r00016",)
    placard_guess = placard_candidates[0]
    accuracy = Fraction(
        sum(rotation.official_rule_id == placard_guess for rotation in block.rotations),
        len(block.rotations),
    )
    assert accuracy == Fraction(1, block.opening.n0)
    assert all(
        sum(rotation.official_rule_id == rule_id for rotation in block.rotations) == 1
        for rule_id in block.opening.version_space_rule_ids
    )
    assert (
        sum(
            (
                Fraction(rotation.exact_weight_numerator, rotation.exact_weight_denominator)
                for rotation in block.rotations
            ),
            Fraction(),
        )
        == 1
    )

    placard_rows = [
        row
        for row in block.as_obj()["descriptive_feature_histograms"]
        if row["feature"] == "placard_inclusion" and row["value"] == "yes"
    ]
    assert placard_rows == [
        {
            "feature": "placard_inclusion",
            "value": "yes",
            "candidate_count": 1,
            "official_count": 1,
        }
    ]


def test_static_input_and_unconditional_law_are_identical_and_role_neutral(
    block: HypothesisCompleteTrainingBlockV2,
    law: UnconditionalTerminalSceneLawV2,
) -> None:
    static = block.static_model_input_obj()
    assert {rotation.static_model_input_digest for rotation in block.rotations} == {
        block.static_model_input_digest
    }
    assert static["public_train_terminal_scene_law"] == law.as_obj()
    assert law.as_obj()["law_kind"] == "public-exact-unconditional-one-item-marginal"
    assert law.as_obj()["public_derivation_attestation_externally_verified"] is False
    assert law.as_obj()["external_generator_verification_required"] is True
    assert (
        sum(
            Fraction(
                item["probability"]["numerator"],
                item["probability"]["denominator"],
            )
            for item in law.as_obj()["scene_probabilities"]
        )
        == 1
    )

    serialized_static = json.dumps(static, sort_keys=True)
    serialized_hidden = json.dumps(block.hidden_order_display_binding.as_obj(), sort_keys=True)
    assert block.hidden_order_display_binding.digest not in serialized_static
    assert serialized_hidden not in serialized_static
    assert all(rule_id not in serialized_static for rule_id in block.opening.version_space_rule_ids)
    assert "materialized_training_panel" not in serialized_static
    assert block.materialized_training_panel.digest not in serialized_static
    assert block.materialized_training_panel.external_generator_receipt_digest not in serialized_static
    replayed_hidden = build_hidden_order_display_binding_v2(
        block.opening,
        independence_precommitment_digest=(
            block.hidden_order_display_binding.independence_precommitment_digest
        ),
    )
    assert replayed_hidden == block.hidden_order_display_binding
    with pytest.raises(HypothesisCompleteV2Error, match="canonical hash order"):
        build_hidden_order_display_binding_v2(
            block.opening,
            independence_precommitment_digest=(
                block.hidden_order_display_binding.independence_precommitment_digest
            ),
            official_rotation_order_rule_ids=tuple(
                reversed(block.hidden_order_display_binding.official_rotation_order_rule_ids)
            ),
        )

    def keys(value: Any) -> set[str]:
        if type(value) is dict:
            return set(value) | set().union(*(keys(child) for child in value.values()))
        if type(value) is list:
            return set().union(*(keys(child) for child in value))
        return set()

    normalized = {key.lower().replace("-", "_") for key in keys(static)}
    assert not normalized & {
        "p",
        "q",
        "a",
        "b",
        "cover",
        "stage",
        "candidate_role",
        "official_rule_id",
        "official_truth_digest",
        "opening_id",
        "block_id",
    }


def test_terminal_law_and_hidden_panel_are_exactly_no_query_neutral(
    block: HypothesisCompleteTrainingBlockV2,
) -> None:
    audit = block.as_obj()["exact_terminal_balance_audit"]
    assert len(block.terminal_law.scene_probabilities) == 32
    assert len(block.materialized_training_panel.scene_indices) == 16
    assert set(block.materialized_training_panel.scene_indices) < {
        item.scene_index for item in block.terminal_law.scene_probabilities
    }
    assert audit["law_support_every_scene_half_half"] is True
    assert audit["law_every_rule_exact_half_mass"] is True
    assert audit["law_complementary_pattern_pair_sampling_exact"] is True
    assert audit["law_and_panel_no_query_baselines_exact_half"] is True
    assert audit["law_no_query_bayes_accuracy"] == {"numerator": 1, "denominator": 2}
    assert audit["materialized_panel_no_query_bayes_accuracy"] == {
        "numerator": 1,
        "denominator": 2,
    }
    assert audit["law_support_pairwise_separation_required_here"] is False
    assert audit["evaluation_challenge_and_query_banks_own_pairwise_separation"] is True
    assert all(
        row["accepted_mass"] == {"numerator": 1, "denominator": 2}
        for row in audit["exact_rule_acceptance_mass_rows"]
    )

    catalog = build_rule_catalog()
    entries = tuple(catalog[int(rule_id[4:])] for rule_id in block.opening.version_space_rule_ids)
    pattern_counts: dict[int, int] = {}
    for scene_index in block.materialized_training_panel.scene_indices:
        pattern = sum(1 << position for position, entry in enumerate(entries) if entry.truth[scene_index])
        assert pattern.bit_count() == block.opening.n0 // 2
        pattern_counts[pattern] = pattern_counts.get(pattern, 0) + 1
    full_mask = (1 << block.opening.n0) - 1
    assert sum(pattern_counts.values()) == 16
    assert all(
        count == pattern_counts.get(full_mask ^ pattern, 0) for pattern, count in pattern_counts.items()
    )


def test_canonical_panel_generator_fails_closed_without_complementary_pairs() -> None:
    opening = _opening_from_scenes(
        "opening-panel-infeasible-old-0005",
        _PANEL_INFEASIBLE_OPENING_SCENES,
    )
    assert opening.n0 == 8
    with pytest.raises(HypothesisCompleteV2Error, match="fewer than sixteen scene pairs"):
        _terminal_fixture(opening)


def test_nested_rules_prove_balanced_terminal_support_need_not_separate_v0() -> None:
    opening = _opening_from_scenes(
        "opening-nested-rule-proof-old-0022",
        _NESTED_RULE_OPENING_SCENES,
    )
    law, panel = _terminal_fixture(opening)
    hidden = build_hidden_order_display_binding_v2(
        opening,
        independence_precommitment_digest=_digest("nested-rule-proof-order"),
    )
    nested_block = build_hypothesis_complete_training_block_v2(
        opening,
        law,
        panel,
        hidden,
        bank_position=0,
        block_id="block-nested-rule-proof",
        renderer_name="train_positional",
        pre_update_checkpoint_digest=_digest("nested-pre"),
        post_update_checkpoint_digest=_digest("nested-post"),
        update_batch_id="batch-nested-rule-proof",
        optimizer_step_before=30_000,
    )
    assert nested_block.structural_audit_passed

    catalog = build_rule_catalog()
    entries = tuple(catalog[int(rule_id[4:])] for rule_id in opening.version_space_rule_ids)
    implication = next(
        (left, right)
        for left in entries
        for right in entries
        if left is not right
        and left.truth.bits != right.truth.bits
        and left.truth.bits & ~right.truth.bits == 0
    )
    left, right = implication
    assert right.truth.bits & ~left.truth.bits != 0
    law_difference_mass = sum(
        (
            item.probability
            for item in law.scene_probabilities
            if right.truth[item.scene_index] and not left.truth[item.scene_index]
        ),
        Fraction(),
    )
    assert law_difference_mass == 0
    rows = nested_block.as_obj()["exact_terminal_balance_audit"]["law_pairwise_separation_rows"]
    assert any(
        {row["left_rule_id"], row["right_rule_id"]} == {left.rule_id, right.rule_id}
        and row["separated"] is False
        for row in rows
    )


def test_consensus_support_is_rejected_despite_exact_rule_marginals(
    block: HypothesisCompleteTrainingBlockV2,
) -> None:
    catalog = build_rule_catalog()
    entries = tuple(catalog[int(rule_id[4:])] for rule_id in block.opening.version_space_rule_ids)
    opening_scenes = {item.scene_index for item in block.opening.observations}
    panel_scenes = set(block.materialized_training_panel.scene_indices)
    all_false = [
        scene_index
        for scene_index in range(len(entries[0].truth))
        if scene_index not in opening_scenes | panel_scenes
        and not any(entry.truth[scene_index] for entry in entries)
    ][:8]
    all_true = [
        scene_index
        for scene_index in range(len(entries[0].truth))
        if scene_index not in opening_scenes | panel_scenes
        and all(entry.truth[scene_index] for entry in entries)
    ][:8]
    assert len(all_false) == len(all_true) == 8
    support = sorted((*panel_scenes, *all_false, *all_true))
    consensus_law = build_unconditional_terminal_scene_law_v2(
        ((scene_index, Fraction(1, len(support))) for scene_index in support),
        public_derivation_attestation_digest=_digest("consensus-law-negative"),
    )
    with pytest.raises(HypothesisCompleteV2Error, match="law_support_every_scene_half_half"):
        replace(block, terminal_law=consensus_law)


def test_conditional_or_official_dependent_terminal_laws_are_rejected(
    law: UnconditionalTerminalSceneLawV2,
) -> None:
    conditional = law.as_obj()
    conditional["conditional_rules"] = [{"official_rule_id": "g03r00016", "scene_probabilities": []}]
    with pytest.raises(HypothesisCompleteV2Error, match=r"noncanonical|reordered"):
        unconditional_terminal_scene_law_v2_from_obj(conditional)

    official_dependent = law.as_obj()
    official_dependent["official_rule_id"] = "g03r00016"
    with pytest.raises(HypothesisCompleteV2Error, match=r"noncanonical|reordered"):
        unconditional_terminal_scene_law_v2_from_obj(official_dependent)

    mislabeled = law.as_obj()
    mislabeled["law_kind"] = "public-exact-rule-conditional-one-item-marginal"
    with pytest.raises(HypothesisCompleteV2Error, match="schema identity"):
        unconditional_terminal_scene_law_v2_from_obj(mislabeled)


def test_atomic_execution_rejects_updates_between_rotations_and_missing_resets(
    block: HypothesisCompleteTrainingBlockV2,
) -> None:
    with pytest.raises(HypothesisCompleteV2Error, match="no parameter update"):
        replace(
            block.rotations[0],
            parameter_update_committed_before_block_commit=True,
        )
    with pytest.raises(HypothesisCompleteV2Error, match="context and cache"):
        replace(block.rotations[0], cache_reset_before_episode=False)

    original = block.rotations[1]
    moved = replace(
        original,
        pre_update_checkpoint_digest=_digest("forbidden-interrotation-checkpoint"),
        optimizer_step_before=original.optimizer_step_before + 1,
        optimizer_step_after_objective_collection=original.optimizer_step_before + 1,
    )
    rotations = list(block.rotations)
    rotations[1] = moved
    with pytest.raises(HypothesisCompleteV2Error, match="atomic commit"):
        replace(block, rotations=tuple(rotations))

    with pytest.raises(HypothesisCompleteV2Error, match="exactly one optimizer step"):
        replace(block.atomic_commit, optimizer_step_after=block.atomic_commit.optimizer_step_before + 2)

    zero_opening = _opening(1)
    zero_hidden = build_hidden_order_display_binding_v2(
        zero_opening,
        independence_precommitment_digest=_digest("zero-gradient-order"),
    )
    zero_law, zero_panel = _terminal_fixture(zero_opening)
    unchanged = _digest("zero-gradient-unchanged-checkpoint")
    zero_gradient_block = build_hypothesis_complete_training_block_v2(
        zero_opening,
        zero_law,
        zero_panel,
        zero_hidden,
        bank_position=1,
        block_id="block-zero-gradient",
        renderer_name="train_positional",
        pre_update_checkpoint_digest=unchanged,
        post_update_checkpoint_digest=unchanged,
        update_batch_id="batch-zero-gradient",
        optimizer_step_before=20_000,
    )
    assert zero_gradient_block.atomic_commit.optimizer_step_after == 20_001
    assert zero_gradient_block.atomic_commit.commit_disposition == ("zero_gradient_no_state_change")
    assert (
        zero_gradient_block.atomic_commit.pre_update_checkpoint_digest
        == zero_gradient_block.atomic_commit.post_update_checkpoint_digest
    )
    with pytest.raises(HypothesisCompleteV2Error, match="disposition disagrees"):
        replace(
            zero_gradient_block.atomic_commit,
            commit_disposition="committed_state_changed",
        )


def test_undercomplete_and_overcomplete_hypothesis_rotations_are_rejected(
    block: HypothesisCompleteTrainingBlockV2,
) -> None:
    with pytest.raises(HypothesisCompleteV2Error, match="rotations must be complete"):
        replace(block, rotations=block.rotations[:-1])
    with pytest.raises(HypothesisCompleteV2Error, match="rotations must be complete"):
        replace(block, rotations=(*block.rotations, block.rotations[-1]))


def test_block_roundtrip_is_deterministic_canonical_and_strict(
    block: HypothesisCompleteTrainingBlockV2,
) -> None:
    encoded = serialize_hypothesis_complete_training_block_v2(block)
    parsed = parse_hypothesis_complete_training_block_v2(encoded, expected_digest=block.digest)
    assert parsed.as_obj() == block.as_obj()
    assert parsed.digest == block.digest
    assert serialize_hypothesis_complete_training_block_v2(parsed) == encoded
    assert verify_hypothesis_complete_training_block_v2(block) is not block

    with pytest.raises(HypothesisCompleteV2Error, match="not canonical"):
        parse_hypothesis_complete_training_block_v2(json.dumps(block.as_obj(), indent=2))

    reordered_obj = block.as_obj()
    reordered = {"report_kind": reordered_obj["report_kind"]}
    reordered.update({key: value for key, value in reordered_obj.items() if key != "report_kind"})
    with pytest.raises(HypothesisCompleteV2Error, match="reordered"):
        parse_hypothesis_complete_training_block_v2(json.dumps(reordered, separators=(",", ":")))

    boolean = block.as_obj()
    boolean["bank_position"] = True
    with pytest.raises(HypothesisCompleteV2Error, match="integer"):
        hypothesis_complete_training_block_v2_from_obj(boolean)

    tampered = block.as_obj()
    tampered["candidate_official_counts"][0]["official_count"] = 2
    with pytest.raises(HypothesisCompleteV2Error, match="tampered"):
        hypothesis_complete_training_block_v2_from_obj(tampered)

    baseline_tamper = block.as_obj()
    baseline_tamper["exact_terminal_balance_audit"]["law_no_query_bayes_accuracy"] = {
        "numerator": 3,
        "denominator": 4,
    }
    with pytest.raises(HypothesisCompleteV2Error, match="tampered"):
        hypothesis_complete_training_block_v2_from_obj(baseline_tamper)

    panel_tamper = block.as_obj()
    panel_tamper["materialized_training_panel"]["external_generator_receipt_verified"] = True
    with pytest.raises(HypothesisCompleteV2Error, match="receipt_verified"):
        hypothesis_complete_training_block_v2_from_obj(panel_tamper)

    private_scope_tamper = block.as_obj()
    private_scope_tamper["model_visible_static_data"]["panel_digest"] = (
        block.materialized_training_panel.digest
    )
    with pytest.raises(HypothesisCompleteV2Error, match=r"noncanonical|reordered"):
        hypothesis_complete_training_block_v2_from_obj(private_scope_tamper)

    duplicate = encoded[:-1] + ',"schema_version":1}'
    with pytest.raises(HypothesisCompleteV2Error, match="duplicate"):
        parse_hypothesis_complete_training_block_v2(duplicate)


def test_legacy_three_role_schema_is_explicitly_non_substitutable() -> None:
    old_block = build_role_block_v2(_digest("legacy-triple"), EVIDENCE_GEOMETRIES[0])
    with pytest.raises(HypothesisCompleteV2Error, match="legacy three-role"):
        hypothesis_complete_training_block_v2_from_obj(old_block.as_obj())


def test_scientific_manifest_is_exactly_384_episodes_but_only_48_groups(
    scientific_manifest: HypothesisCompleteTrainingManifestV2,
) -> None:
    manifest = scientific_manifest
    assert manifest.n0 == 8
    assert manifest.episode_count == SCIENTIFIC_TRAINING_EPISODE_BUDGET == 384
    assert len(manifest.blocks) == 48
    assert manifest.engineering_budget_override is False
    assert manifest.hypothesis_completion_structure_passed
    assert len({block.block_id for block in manifest.blocks}) == 48
    assert len({block.opening.opening_id for block in manifest.blocks}) == 48
    assert len({block.opening.content_digest for block in manifest.blocks}) == 48
    assert all(
        left.atomic_commit.post_update_checkpoint_digest == right.atomic_commit.pre_update_checkpoint_digest
        and left.atomic_commit.optimizer_step_after == right.atomic_commit.optimizer_step_before
        for left, right in zip(manifest.blocks, manifest.blocks[1:], strict=False)
    )

    manifest_obj = manifest.as_obj()
    scope = manifest_obj["inference_scope"]
    assert scope == {
        "distinct_opening_block_count": 48,
        "episode_row_count": 384,
        "powered_stress_minimum_independent_blocks": (POWERED_STRESS_MINIMUM_INDEPENDENT_BLOCKS),
        "structural_manifest_establishes_sampling_independence": False,
        "feature_histograms_descriptive_only": True,
        "powered_384_group_statistics_claimed": False,
    }
    assert scope["distinct_opening_block_count"] < scope["powered_stress_minimum_independent_blocks"]
    boundaries = manifest_obj["verification_boundaries"]
    assert boundaries["hypothesis_completion_structure_verified"] is True
    assert boundaries["training_bank_balance_verified"] is False
    assert boundaries["training_bank_balance_verification_required"] is True
    assert boundaries["powered_stress_support_containment_verified"] is False
    assert boundaries["powered_stress_support_containment_verification_required"] is True
    assert boundaries["sampling_independence_verified"] is False
    assert boundaries["sampling_independence_verification_required"] is True
    assert boundaries["runtime_execution_verified"] is False
    assert boundaries["runtime_execution_verification_required"] is True
    assert boundaries["launch_gate_passed"] is False
    assert boundaries["launch_gate_verification_required"] is True
    scene_sets = manifest_obj["opening_scene_set_summary"]
    assert scene_sets["unique_opening_scene_set_count"] == 48
    assert scene_sets["structural_manifest_requires_unique_opening_scene_sets"] is False
    assert scene_sets["powered_audit_must_group_by_opening_scene_set_digest"] is True
    assert scene_sets["powered_audit_must_reject_scene_set_reuse_as_independent_support"] is True


def test_manifest_roundtrip_budget_override_uniqueness_and_chain_are_strict(
    scientific_manifest: HypothesisCompleteTrainingManifestV2,
    block: HypothesisCompleteTrainingBlockV2,
) -> None:
    with pytest.raises(HypothesisCompleteV2Error, match="explicit override"):
        build_hypothesis_complete_training_manifest_v2((block,), registered_episode_budget=8)
    engineering = build_hypothesis_complete_training_manifest_v2(
        (block,),
        registered_episode_budget=8,
        engineering_budget_override=True,
    )
    assert engineering.episode_count == 8
    assert engineering.as_obj()["engineering_budget_override"] is True
    encoded = serialize_hypothesis_complete_training_manifest_v2(engineering)
    parsed = parse_hypothesis_complete_training_manifest_v2(encoded, expected_digest=engineering.digest)
    assert parsed.as_obj() == engineering.as_obj()
    assert serialize_hypothesis_complete_training_manifest_v2(parsed) == encoded
    assert verify_hypothesis_complete_training_manifest_v2(engineering) is not engineering
    with pytest.raises(HypothesisCompleteV2Error, match="forbids one"):
        build_hypothesis_complete_training_manifest_v2(
            scientific_manifest.blocks,
            engineering_budget_override=True,
        )

    duplicate = replace(_block(1), block_id=block.block_id)
    with pytest.raises(HypothesisCompleteV2Error, match="block IDs"):
        build_hypothesis_complete_training_manifest_v2(
            (block, duplicate),
            registered_episode_budget=16,
            engineering_budget_override=True,
        )

    relabeled_opening = build_canonical_supported_opening_v2(
        "opening-relabel-of-zero",
        _EIGHT_RULE_OPENING,
    )
    assert relabeled_opening.content_digest == block.opening.content_digest
    assert relabeled_opening.digest != block.opening.digest
    relabeled_hidden = build_hidden_order_display_binding_v2(
        relabeled_opening,
        independence_precommitment_digest=_digest("relabel-order"),
    )
    relabeled_law, relabeled_panel = _terminal_fixture(relabeled_opening)
    relabeled_block = build_hypothesis_complete_training_block_v2(
        relabeled_opening,
        relabeled_law,
        relabeled_panel,
        relabeled_hidden,
        bank_position=1,
        block_id="block-relabel-of-zero",
        renderer_name="train_positional",
        pre_update_checkpoint_digest=block.atomic_commit.post_update_checkpoint_digest,
        post_update_checkpoint_digest=_digest("relabel-post-checkpoint"),
        update_batch_id="batch-relabel-of-zero",
        optimizer_step_before=block.atomic_commit.optimizer_step_after,
    )
    with pytest.raises(HypothesisCompleteV2Error, match="opening contents"):
        build_hypothesis_complete_training_manifest_v2(
            (block, relabeled_block),
            registered_episode_budget=16,
            engineering_budget_override=True,
        )

    wrong_chain = _block(
        1,
        pre_digest=_digest("not-the-prior-post-checkpoint"),
    )
    with pytest.raises(HypothesisCompleteV2Error, match="checkpoint chain"):
        build_hypothesis_complete_training_manifest_v2(
            (block, wrong_chain),
            registered_episode_budget=16,
            engineering_budget_override=True,
        )


def test_parser_rejects_manifest_bool_tamper_reorder_and_cross_schema(
    block: HypothesisCompleteTrainingBlockV2,
) -> None:
    manifest = build_hypothesis_complete_training_manifest_v2(
        (block,),
        registered_episode_budget=8,
        engineering_budget_override=True,
    )
    value = manifest.as_obj()
    value["engineering_budget_override"] = 0
    with pytest.raises(HypothesisCompleteV2Error, match="Boolean"):
        parse_hypothesis_complete_training_manifest_v2(json.dumps(value, separators=(",", ":")))

    value = manifest.as_obj()
    value["inference_scope"]["powered_384_group_statistics_claimed"] = True
    with pytest.raises(HypothesisCompleteV2Error, match="tampered"):
        parse_hypothesis_complete_training_manifest_v2(json.dumps(value, separators=(",", ":")))

    value = manifest.as_obj()
    value["verification_boundaries"]["launch_gate_passed"] = True
    with pytest.raises(HypothesisCompleteV2Error, match="tampered"):
        parse_hypothesis_complete_training_manifest_v2(json.dumps(value, separators=(",", ":")))

    value = manifest.as_obj()
    reordered = {"report_kind": value["report_kind"]}
    reordered.update({key: item for key, item in value.items() if key != "report_kind"})
    with pytest.raises(HypothesisCompleteV2Error, match="reordered"):
        parse_hypothesis_complete_training_manifest_v2(json.dumps(reordered, separators=(",", ":")))

    old_block = build_role_block_v2(_digest("legacy-cross-schema"), EVIDENCE_GEOMETRIES[0])
    with pytest.raises(HypothesisCompleteV2Error, match="legacy three-role"):
        parse_hypothesis_complete_training_manifest_v2(json.dumps(old_block.as_obj(), separators=(",", ":")))
