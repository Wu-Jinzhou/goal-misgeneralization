from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from fractions import Fraction
from typing import Any

import pytest

from goalzendo_interactive import VersionSpace, build_rule_catalog
from goalzendo_interactive.schema import SCENE_COUNT
from goalzendo_interactive_v2.hypothesis_complete import (
    CanonicalSupportedOpeningV2,
    HypothesisCompleteTrainingBlockV2,
    MaterializedTrainingPanelV2,
    UnconditionalTerminalSceneLawV2,
    build_canonical_supported_opening_v2,
    build_hidden_order_display_binding_v2,
    build_hypothesis_complete_training_block_v2,
    build_hypothesis_complete_training_manifest_v2,
    build_materialized_training_panel_v2,
    build_unconditional_terminal_scene_law_v2,
    serialize_hypothesis_complete_training_manifest_v2,
)
from goalzendo_interactive_v2.manifest_verifier import (
    FrozenManifestExpectedBindingsV2,
    FrozenManifestRederivationV2,
    FrozenManifestV2Error,
    capture_frozen_manifest_expected_bindings_v2,
    serialize_frozen_manifest_rederivation_v2,
    serialize_frozen_meta_surfaces_v2,
    verify_frozen_hypothesis_complete_manifest_v2,
)
from goalzendo_interactive_v2.population_audit import build_supported_catalog_contract_v2


def _digest(label: str) -> str:
    return hashlib.sha256(label.encode("ascii")).hexdigest()


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=True, allow_nan=False, separators=(",", ":")).encode("ascii")


def _variant_openings() -> tuple[CanonicalSupportedOpeningV2, CanonicalSupportedOpeningV2]:
    catalog = build_rule_catalog()
    placard = catalog[16]
    shared_nine = (446, 1208, 2815, 3884, 5390, 7676, 10126, 11567, 12043)
    # 12431 and 11690 are both accepted replacements that restore the same
    # exact eight-rule supported V0.  Both retain the complementary-panel gate.
    replacements = (12431, 11690)
    openings = tuple(
        build_canonical_supported_opening_v2(
            f"opening-shared-base-{position}",
            (
                *((scene_index, placard.truth[scene_index]) for scene_index in shared_nine),
                (replacement, True),
            ),
        )
        for position, replacement in enumerate(replacements)
    )
    assert all(opening.n0 == 8 for opening in openings)
    assert openings[0].version_space_rule_ids == openings[1].version_space_rule_ids
    return openings[0], openings[1]


def _terminal_fixture(
    opening: CanonicalSupportedOpeningV2,
) -> tuple[UnconditionalTerminalSceneLawV2, MaterializedTrainingPanelV2]:
    catalog = build_rule_catalog()
    entries = tuple(catalog[int(rule_id[4:])] for rule_id in opening.version_space_rule_ids)
    opening_scenes = {item.scene_index for item in opening.observations}
    scenes_by_pattern: dict[int, list[int]] = {}
    for scene_index in range(SCENE_COUNT):
        if scene_index in opening_scenes:
            continue
        pattern = sum(1 << position for position, entry in enumerate(entries) if entry.truth[scene_index])
        if pattern.bit_count() == opening.n0 // 2:
            scenes_by_pattern.setdefault(pattern, []).append(scene_index)
    full_mask = (1 << opening.n0) - 1
    pairs = [
        (pattern, full_mask ^ pattern, left, right)
        for pattern in sorted(scenes_by_pattern)
        if pattern < (full_mask ^ pattern) and (full_mask ^ pattern) in scenes_by_pattern
        for left, right in zip(
            scenes_by_pattern[pattern],
            scenes_by_pattern[full_mask ^ pattern],
            strict=False,
        )
    ]
    assert len(pairs) >= 16

    def rank(domain: str, pair: tuple[int, int, int, int]) -> str:
        return hashlib.sha256(f"{domain}\0{opening.content_digest}\0{pair}".encode("ascii")).hexdigest()

    law_pairs = tuple(sorted(pairs, key=lambda pair: rank("law", pair))[:16])
    law_scenes = tuple(sorted(scene for pair in law_pairs for scene in pair[2:]))
    law = build_unconditional_terminal_scene_law_v2(
        ((scene_index, Fraction(1, len(law_scenes))) for scene_index in law_scenes),
        public_derivation_attestation_digest=_digest(f"law:{opening.content_digest}"),
    )
    panel_pairs = tuple(sorted(law_pairs, key=lambda pair: rank("panel", pair))[:8])
    panel = build_materialized_training_panel_v2(
        opening,
        (scene for pair in panel_pairs for scene in pair[2:]),
        external_generator_receipt_digest=_digest(f"panel:{opening.content_digest}"),
    )
    return law, panel


def _block(
    opening: CanonicalSupportedOpeningV2,
    position: int,
    *,
    pre: str,
    post: str,
    block_label: str | None = None,
) -> HypothesisCompleteTrainingBlockV2:
    law, panel = _terminal_fixture(opening)
    hidden = build_hidden_order_display_binding_v2(
        opening,
        independence_precommitment_digest=_digest(f"order:{opening.content_digest}"),
    )
    return build_hypothesis_complete_training_block_v2(
        opening,
        law,
        panel,
        hidden,
        bank_position=position,
        block_id=block_label or f"block-{position}",
        renderer_name="train_positional",
        pre_update_checkpoint_digest=pre,
        post_update_checkpoint_digest=post,
        update_batch_id=f"batch-{position}-{opening.opening_id}",
        optimizer_step_before=7_000 + position,
    )


def _artifacts(
    blocks: tuple[HypothesisCompleteTrainingBlockV2, ...],
) -> tuple[bytes, bytes, FrozenManifestExpectedBindingsV2]:
    manifest = build_hypothesis_complete_training_manifest_v2(
        blocks,
        registered_episode_budget=sum(block.opening.n0 for block in blocks),
        engineering_budget_override=True,
    )
    manifest_bytes = serialize_hypothesis_complete_training_manifest_v2(manifest).encode("ascii")
    meta_bytes = serialize_frozen_meta_surfaces_v2(
        manifest_bytes,
        schedule_positions_by_block=tuple(tuple(reversed(range(block.opening.n0))) for block in blocks),
        request_positions_by_block=tuple(
            tuple((position + block.bank_position) % block.opening.n0 for position in range(block.opening.n0))
            for block in blocks
        ),
    )
    bindings = capture_frozen_manifest_expected_bindings_v2(manifest_bytes, meta_bytes)
    return manifest_bytes, meta_bytes, bindings


@pytest.fixture(scope="module")
def one_block_artifacts() -> tuple[bytes, bytes, FrozenManifestExpectedBindingsV2]:
    opening, _ = _variant_openings()
    block = _block(opening, 0, pre=_digest("checkpoint-0"), post=_digest("checkpoint-1"))
    return _artifacts((block,))


def _verify_with_recaptured_bindings(
    manifest_bytes: bytes,
    meta_bytes: bytes,
) -> FrozenManifestRederivationV2:
    return verify_frozen_hypothesis_complete_manifest_v2(
        manifest_bytes,
        meta_bytes,
        expected_bindings=capture_frozen_manifest_expected_bindings_v2(
            manifest_bytes,
            meta_bytes,
        ),
    )


def test_live_rederivation_binds_sources_semantics_and_remains_nonauthorizing(
    one_block_artifacts: tuple[bytes, bytes, FrozenManifestExpectedBindingsV2],
) -> None:
    manifest_bytes, meta_bytes, bindings = one_block_artifacts
    report = verify_frozen_hypothesis_complete_manifest_v2(
        manifest_bytes,
        meta_bytes,
        expected_bindings=bindings,
    )
    assert report.fixed_n0 == 8
    assert report.block_count == 1
    assert report.episode_count == 8
    assert len(report.construction_clusters) == 1
    value = report.as_obj()
    assert all(flag is False for flag in value["authorization"].values() if type(flag) is bool)
    assert all(flag is False for flag in value["unprovable_without_future_evidence"].values())
    assert all(value["rederived_guarantees"].values())
    stage = value["block_stage_identity_and_disjointness"][0]
    assert stage["opening_and_public_law_support_disjoint"] is True
    assert stage["opening_and_materialized_panel_disjoint"] is True
    assert stage["materialized_panel_subset_of_public_law_support"] is True
    assert stage["cross_block_scene_disjointness_claimed"] is False
    assert len(stage["version_space_rule_ids"]) == 8
    encoded = serialize_frozen_manifest_rederivation_v2(report)
    assert json.loads(encoded)["frozen_manifest_rederivation_digest"] == report.digest
    assert bindings.catalog_digest == build_rule_catalog().digest
    assert bindings.supported_catalog_digest == (
        build_supported_catalog_contract_v2().supported_catalog_digest
    )


def test_external_hash_source_catalog_and_canonical_byte_bindings_fail_closed(
    one_block_artifacts: tuple[bytes, bytes, FrozenManifestExpectedBindingsV2],
) -> None:
    manifest_bytes, meta_bytes, bindings = one_block_artifacts
    with pytest.raises(FrozenManifestV2Error, match="manifest bytes differ"):
        verify_frozen_hypothesis_complete_manifest_v2(
            manifest_bytes + b"\n",
            meta_bytes,
            expected_bindings=bindings,
        )
    whitespace_manifest = manifest_bytes + b"\n"
    whitespace_meta = serialize_frozen_meta_surfaces_v2(manifest_bytes)
    recaptured = capture_frozen_manifest_expected_bindings_v2(whitespace_manifest, whitespace_meta)
    with pytest.raises(FrozenManifestV2Error, match="canonical compact"):
        verify_frozen_hypothesis_complete_manifest_v2(
            whitespace_manifest,
            whitespace_meta,
            expected_bindings=recaptured,
        )
    with pytest.raises(FrozenManifestV2Error, match="producer source"):
        verify_frozen_hypothesis_complete_manifest_v2(
            manifest_bytes,
            meta_bytes,
            expected_bindings=replace(
                bindings,
                hypothesis_complete_source_sha256=_digest("wrong-producer-source"),
            ),
        )
    with pytest.raises(FrozenManifestV2Error, match="catalog digest"):
        verify_frozen_hypothesis_complete_manifest_v2(
            manifest_bytes,
            meta_bytes,
            expected_bindings=replace(bindings, catalog_digest=_digest("wrong-catalog")),
        )
    with pytest.raises(FrozenManifestV2Error, match="lowercase SHA-256"):
        FrozenManifestExpectedBindingsV2(
            True,  # type: ignore[arg-type]
            bindings.meta_surface_bytes_sha256,
            bindings.hypothesis_complete_source_sha256,
            bindings.statistical_leakage_source_sha256,
            bindings.catalog_source_sha256,
            bindings.catalog_digest,
            bindings.supported_catalog_digest,
            bindings.renderer_registry_digest,
        )


def test_administrative_relabels_do_not_define_construction_identity() -> None:
    opening, _ = _variant_openings()
    original = _block(
        opening,
        0,
        pre=_digest("relabel-pre"),
        post=_digest("relabel-post"),
        block_label="caller-label-original",
    )
    relabeled_opening = build_canonical_supported_opening_v2(
        "caller-opening-label-changed",
        tuple((item.scene_index, item.accepted) for item in opening.observations),
    )
    relabeled = _block(
        relabeled_opening,
        0,
        pre=_digest("relabel-pre"),
        post=_digest("relabel-post"),
        block_label="caller-label-changed",
    )
    original_report = _verify_with_recaptured_bindings(*_artifacts((original,))[:2])
    relabeled_report = _verify_with_recaptured_bindings(*_artifacts((relabeled,))[:2])
    original_member = original_report.construction_clusters[0][1][0]
    relabeled_member = relabeled_report.construction_clusters[0][1][0]
    assert original_member == relabeled_member
    assert original.opening.digest != relabeled.opening.digest
    assert original.opening.content_digest == relabeled.opening.content_digest


def test_shared_nine_row_base_is_one_content_rederived_cluster() -> None:
    first, second = _variant_openings()
    checkpoint_0 = _digest("shared-checkpoint-0")
    checkpoint_1 = _digest("shared-checkpoint-1")
    checkpoint_2 = _digest("shared-checkpoint-2")
    blocks = (
        _block(first, 0, pre=checkpoint_0, post=checkpoint_1),
        _block(second, 1, pre=checkpoint_1, post=checkpoint_2),
    )
    manifest_bytes, meta_bytes, bindings = _artifacts(blocks)
    report = verify_frozen_hypothesis_complete_manifest_v2(
        manifest_bytes,
        meta_bytes,
        expected_bindings=bindings,
    )
    assert report.block_count == 2
    assert len(report.construction_clusters) == 1
    assert len(report.construction_clusters[0][1]) == 2
    identity = report.as_obj()["construction_identity"]
    assert identity["caller_block_and_opening_labels_used"] is False


def test_reordered_rotations_and_meta_rows_are_rejected_after_hash_recapture(
    one_block_artifacts: tuple[bytes, bytes, FrozenManifestExpectedBindingsV2],
) -> None:
    manifest_bytes, meta_bytes, _ = one_block_artifacts
    manifest = json.loads(manifest_bytes)
    manifest["blocks"][0]["rotations"][0], manifest["blocks"][0]["rotations"][1] = (
        manifest["blocks"][0]["rotations"][1],
        manifest["blocks"][0]["rotations"][0],
    )
    reordered_manifest = _canonical_bytes(manifest)
    with pytest.raises(FrozenManifestV2Error, match="reordered"):
        _verify_with_recaptured_bindings(reordered_manifest, meta_bytes)

    meta = json.loads(meta_bytes)
    meta["blocks"][0]["rotations"][0], meta["blocks"][0]["rotations"][1] = (
        meta["blocks"][0]["rotations"][1],
        meta["blocks"][0]["rotations"][0],
    )
    reordered_meta = _canonical_bytes(meta)
    with pytest.raises(FrozenManifestV2Error, match="relabeled or reordered"):
        _verify_with_recaptured_bindings(manifest_bytes, reordered_meta)


def test_renderer_xor_and_model_visible_position_code_are_rejected(
    one_block_artifacts: tuple[bytes, bytes, FrozenManifestExpectedBindingsV2],
) -> None:
    manifest_bytes, meta_bytes, _ = one_block_artifacts
    renderer_xor = json.loads(meta_bytes)
    renderer_xor["blocks"][0]["rotations"][1]["model_visible_surface"]["renderer"] = "train_compact"
    with pytest.raises(FrozenManifestV2Error, match="renderer/static prompt differs"):
        _verify_with_recaptured_bindings(manifest_bytes, _canonical_bytes(renderer_xor))

    position_code = json.loads(meta_bytes)
    visible = position_code["blocks"][0]["rotations"][0]["model_visible_surface"]
    visible["schedule_position"] = 0
    with pytest.raises(FrozenManifestV2Error, match="model-visible meta surface"):
        _verify_with_recaptured_bindings(manifest_bytes, _canonical_bytes(position_code))

    duplicate_position = json.loads(meta_bytes)
    rotations = duplicate_position["blocks"][0]["rotations"]
    rotations[1]["executor_only_surface"]["schedule_position"] = rotations[0]["executor_only_surface"][
        "schedule_position"
    ]
    with pytest.raises(FrozenManifestV2Error, match="bounded local permutations"):
        _verify_with_recaptured_bindings(manifest_bytes, _canonical_bytes(duplicate_position))


def test_semantic_tamper_boolean_weight_atomic_and_terminal_identity_fail_closed(
    one_block_artifacts: tuple[bytes, bytes, FrozenManifestExpectedBindingsV2],
) -> None:
    manifest_bytes, meta_bytes, _ = one_block_artifacts

    boolean = json.loads(manifest_bytes)
    boolean["blocks"][0]["bank_position"] = True
    with pytest.raises(FrozenManifestV2Error, match="integer"):
        _verify_with_recaptured_bindings(_canonical_bytes(boolean), meta_bytes)

    weight = json.loads(manifest_bytes)
    weight["blocks"][0]["rotations"][0]["exact_objective_weight"]["numerator"] = 2
    with pytest.raises(FrozenManifestV2Error, match="exactly 1/n0"):
        _verify_with_recaptured_bindings(_canonical_bytes(weight), meta_bytes)

    atomic = json.loads(manifest_bytes)
    commit = atomic["blocks"][0]["atomic_commit"]
    commit["post_update_checkpoint_digest"] = commit["pre_update_checkpoint_digest"]
    with pytest.raises(FrozenManifestV2Error, match="disposition"):
        _verify_with_recaptured_bindings(_canonical_bytes(atomic), meta_bytes)

    law = json.loads(manifest_bytes)
    law["blocks"][0]["unconditional_terminal_scene_law"][
        "public_derivation_attestation_externally_verified"
    ] = True
    with pytest.raises(FrozenManifestV2Error, match="externally unverified"):
        _verify_with_recaptured_bindings(_canonical_bytes(law), meta_bytes)

    panel = json.loads(manifest_bytes)
    panel["blocks"][0]["materialized_training_panel"]["version_space_rules"][0]["truth_digest"] = _digest(
        "false-panel-truth"
    )
    with pytest.raises(FrozenManifestV2Error, match="panel V0 identities"):
        _verify_with_recaptured_bindings(_canonical_bytes(panel), meta_bytes)

    claimed_pass = json.loads(manifest_bytes)
    claimed_pass["blocks"][0]["rotations"] = claimed_pass["blocks"][0]["rotations"][:-1]
    claimed_pass["blocks"][0]["structural_audit_passed"] = True
    with pytest.raises(FrozenManifestV2Error, match="exactly one rotation"):
        _verify_with_recaptured_bindings(_canonical_bytes(claimed_pass), meta_bytes)


def test_exact_supported_v0_is_recomputed_not_accepted_from_parser_rows(
    one_block_artifacts: tuple[bytes, bytes, FrozenManifestExpectedBindingsV2],
) -> None:
    manifest_bytes, meta_bytes, _ = one_block_artifacts
    manifest = json.loads(manifest_bytes)
    opening = manifest["blocks"][0]["opening"]
    opening["version_space_rules"][0] = opening["version_space_rules"][1]
    opening["n0"] = 8
    manifest["blocks"][0]["n0"] = 8
    manifest["fixed_n0"] = 8
    with pytest.raises(FrozenManifestV2Error, match="live catalog recomputation"):
        _verify_with_recaptured_bindings(_canonical_bytes(manifest), meta_bytes)


def test_builder_itself_rejects_nonpermutation_boolean_positions(
    one_block_artifacts: tuple[bytes, bytes, FrozenManifestExpectedBindingsV2],
) -> None:
    manifest_bytes, _, _ = one_block_artifacts
    with pytest.raises(FrozenManifestV2Error, match="bounded local permutation"):
        serialize_frozen_meta_surfaces_v2(
            manifest_bytes,
            schedule_positions_by_block=((0, 0, 2, 3, 4, 5, 6, 7),),
        )
    with pytest.raises(FrozenManifestV2Error, match="integer"):
        serialize_frozen_meta_surfaces_v2(
            manifest_bytes,
            request_positions_by_block=((True, 1, 2, 3, 4, 5, 6, 7),),
        )


def test_manifest_catalog_namespace_matches_independent_supported_version_space() -> None:
    first, _ = _variant_openings()
    catalog = build_rule_catalog()
    contract = build_supported_catalog_contract_v2()
    independently_recomputed = VersionSpace(catalog, contract.supported_indices).observe_many(
        (item.scene_index, item.accepted) for item in first.observations
    )
    assert tuple(entry.rule_id for entry in independently_recomputed) == first.version_space_rule_ids
