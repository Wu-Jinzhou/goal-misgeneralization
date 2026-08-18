from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from typing import Any

import pytest

from goalzendo_interactive_v2.hypothesis_complete import (
    CanonicalSupportedOpeningV2,
    UnconditionalTerminalSceneLawV2,
    build_canonical_supported_opening_v2,
    build_hidden_order_display_binding_v2,
)
from goalzendo_interactive_v2.training_freeze import (
    HypothesisCompleteTrainingPlanV1,
    TrainingFreezeBindingsV1,
    build_hypothesis_complete_training_block_plan_v1,
    build_hypothesis_complete_training_plan_v1,
    serialize_hypothesis_complete_training_plan_v1,
)
from goalzendo_interactive_v2.training_surface_bridge import (
    PlannedTrainingStaticSurfaceBridgeV1,
    PlannedTrainingSurfaceBridgeV1Error,
    build_planned_training_surface_bridge_v1,
    parse_planned_training_surface_bridge_v1,
    serialize_planned_training_surface_bridge_v1,
    validate_planned_model_visible_static_data_v1,
    verify_planned_training_surface_bridge_v1,
)
from tests.goalzendo_interactive_v2.test_hypothesis_complete import (
    _opening,
    _terminal_fixture,
)


def _digest(label: str) -> str:
    return hashlib.sha256(label.encode("ascii")).hexdigest()


def _canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=True, allow_nan=False, separators=(",", ":"))


def _static_digest(value: Any) -> str:
    digest = hashlib.sha256()
    digest.update(b"goalzendo-interactive-v2-training-freeze-static-input-v1\0")
    digest.update(_canonical(value).encode("ascii"))
    return digest.hexdigest()


def _plan(
    *,
    block_id: str = "surface-block-0",
    opening_id: str | None = None,
    changed_terminal_law: bool = False,
) -> HypothesisCompleteTrainingPlanV1:
    opening: CanonicalSupportedOpeningV2 = _opening(0)
    if opening_id is not None:
        opening = build_canonical_supported_opening_v2(
            opening_id,
            ((item.scene_index, item.accepted) for item in opening.observations),
        )
    law, panel = _terminal_fixture(opening)
    if changed_terminal_law:
        law = replace(
            law,
            public_derivation_attestation_digest=_digest("changed-public-law-attestation"),
        )
    hidden = build_hidden_order_display_binding_v2(
        opening,
        independence_precommitment_digest=_digest("surface-hidden-order"),
    )
    block = build_hypothesis_complete_training_block_plan_v1(
        opening,
        law,
        panel,
        hidden,
        bank_position=0,
        block_id=block_id,
        renderer_name="train_positional",
        update_batch_id="surface-batch-0",
        planned_optimizer_step_before=700,
    )
    return build_hypothesis_complete_training_plan_v1(
        TrainingFreezeBindingsV1(
            source_tree_sha256=_digest("surface-source-tree"),
            bank_generator_source_sha256=_digest("surface-bank-generator"),
            initial_checkpoint_manifest_sha256=_digest("surface-initial-checkpoint"),
            tokenizer_binding_digest=_digest("surface-tokenizer-binding"),
            optimizer_config_digest=_digest("surface-optimizer-config"),
        ),
        (block,),
        registered_episode_budget=8,
        engineering_budget_override=True,
    )


@pytest.fixture(scope="module")
def plan() -> HypothesisCompleteTrainingPlanV1:
    return _plan()


def _bridge_and_text(
    plan: HypothesisCompleteTrainingPlanV1,
) -> tuple[str, PlannedTrainingStaticSurfaceBridgeV1, str]:
    plan_text = serialize_hypothesis_complete_training_plan_v1(plan)
    bridge = build_planned_training_surface_bridge_v1(
        plan_text,
        expected_plan_digest=plan.digest,
    )
    bridge_text = serialize_planned_training_surface_bridge_v1(
        bridge,
        plan_text=plan_text,
        expected_plan_digest=plan.digest,
        expected_bridge_digest=bridge.digest,
    )
    return plan_text, bridge, bridge_text


def test_exact_plan_backed_bridge_round_trips_canonically(
    plan: HypothesisCompleteTrainingPlanV1,
) -> None:
    plan_text, bridge, encoded = _bridge_and_text(plan)
    parsed = parse_planned_training_surface_bridge_v1(
        encoded,
        plan_text=plan_text,
        expected_plan_digest=plan.digest,
        expected_bridge_digest=bridge.digest,
    )
    assert parsed == bridge
    assert (
        verify_planned_training_surface_bridge_v1(
            bridge,
            plan_text=plan_text,
            expected_plan_digest=plan.digest,
            expected_bridge_digest=bridge.digest,
        )
        == bridge
    )
    value = json.loads(encoded)
    binding = value["exact_training_plan_binding"]
    assert binding["canonical_plan_bytes_sha256"] == hashlib.sha256(plan_text.encode("ascii")).hexdigest()
    assert binding["canonical_plan_byte_count"] == len(plan_text.encode("ascii"))
    assert binding["hypothesis_complete_training_plan_digest"] == plan.digest
    assert binding["pre_run_bindings"] == plan.bindings.as_obj()
    assert binding["episode_count"] == binding["fixed_n0"] == 8


def test_complete_static_object_is_exact_and_private_rotations_are_separate(
    plan: HypothesisCompleteTrainingPlanV1,
) -> None:
    _, bridge, encoded = _bridge_and_text(plan)
    block = bridge.blocks[0]
    planned = plan.blocks[0]
    assert block.static_model_input_obj == planned.static_model_input_obj()
    assert block.static_model_input_digest == planned.static_model_input_digest
    assert block.terminal_law_digest == planned.terminal_law.digest
    assert {row.static_model_input_digest for row in block.rotations} == {planned.static_model_input_digest}
    assert {row.official_rule_id for row in block.rotations} == set(planned.opening.version_space_rule_ids)

    visible = _canonical(json.loads(encoded)["blocks"][0]["model_visible_static_data"])
    private = json.loads(encoded)["blocks"][0]["evaluator_private"]
    assert planned.block_id not in visible
    assert planned.digest not in visible
    assert planned.opening.opening_id not in visible
    assert planned.materialized_training_panel.digest not in visible
    assert planned.hidden_order_display_binding.digest not in visible
    assert planned.update_batch_id not in visible
    assert all(row["official_rule_id"] not in visible for row in private["rotations"])
    assert "rotations" not in json.loads(visible)
    assert "token_ids" not in visible
    assert "chat_template" not in visible


def test_terminal_law_change_closes_legacy_opening_only_surface_gap() -> None:
    original = _plan()
    changed = _plan(changed_terminal_law=True)
    original_text, original_bridge, _ = _bridge_and_text(original)
    changed_text, changed_bridge, _ = _bridge_and_text(changed)

    assert original.blocks[0].opening == changed.blocks[0].opening
    assert original.blocks[0].renderer_name == changed.blocks[0].renderer_name
    # An opening-plus-renderer-only surface would be identical here.
    assert original.blocks[0].opening.content_digest == changed.blocks[0].opening.content_digest
    assert original_bridge.blocks[0].static_model_input_digest != (
        changed_bridge.blocks[0].static_model_input_digest
    )
    assert original_bridge.blocks[0].static_model_input_canonical_sha256 != (
        changed_bridge.blocks[0].static_model_input_canonical_sha256
    )
    assert original_bridge.digest != changed_bridge.digest
    assert original_text != changed_text


def test_administrative_relabel_changes_plan_binding_not_visible_surface() -> None:
    original = _plan()
    relabeled = _plan(
        block_id="surface-block-relabeled",
        opening_id="surface-opening-relabeled",
    )
    _, original_bridge, _ = _bridge_and_text(original)
    _, relabeled_bridge, _ = _bridge_and_text(relabeled)
    original_block = original_bridge.blocks[0]
    relabeled_block = relabeled_bridge.blocks[0]
    assert original_block.opening_content_digest == relabeled_block.opening_content_digest
    assert original_block.static_model_input_json == relabeled_block.static_model_input_json
    assert original_block.static_model_input_digest == relabeled_block.static_model_input_digest
    assert original_block.block_id != relabeled_block.block_id
    assert original_bridge.plan_digest != relabeled_bridge.plan_digest
    assert original_bridge.digest != relabeled_bridge.digest


def test_visible_boundary_rejects_private_runtime_token_and_answer_fields(
    plan: HypothesisCompleteTrainingPlanV1,
) -> None:
    static = plan.blocks[0].static_model_input_obj()
    private_id = plan.blocks[0].opening.version_space_rule_ids[0]

    token_injection = json.loads(_canonical(static))
    token_injection["opening_examples"][0]["nested"] = {"token_ids": [1, 2]}
    with pytest.raises(PlannedTrainingSurfaceBridgeV1Error, match="noncanonical"):
        validate_planned_model_visible_static_data_v1(token_injection)

    answer_injection = json.loads(_canonical(static))
    answer_injection["opening_examples"][0]["classification"] = "fits"
    with pytest.raises(PlannedTrainingSurfaceBridgeV1Error, match="noncanonical"):
        validate_planned_model_visible_static_data_v1(answer_injection)

    identity_injection = json.loads(_canonical(static))
    identity_injection["opening_examples"][0]["scene_text"] = private_id
    with pytest.raises(PlannedTrainingSurfaceBridgeV1Error, match="private identity"):
        validate_planned_model_visible_static_data_v1(
            identity_injection,
            evaluator_private_strings=(private_id,),
        )

    for key in (
        "official_secret",
        "p_identity",
        "q_identity",
        "runtime_prompt_payload",
        "token_id_list",
        "ast_output",
        "classification_result",
        "panel_order",
        "display_order",
        "private_metadata",
        "remaining_items",
    ):
        family_injection = json.loads(_canonical(static))
        family_injection["opening_examples"][0][key] = "forbidden"
        with pytest.raises(
            PlannedTrainingSurfaceBridgeV1Error,
            match=r"noncanonical|private/runtime field",
        ):
            validate_planned_model_visible_static_data_v1(family_injection)

    substring_injection = json.loads(_canonical(static))
    substring_injection["opening_examples"][0]["scene_text"] += f" {private_id} suffix"
    with pytest.raises(PlannedTrainingSurfaceBridgeV1Error, match="private identity"):
        validate_planned_model_visible_static_data_v1(
            substring_injection,
            evaluator_private_strings=(private_id,),
        )

    key_injection = json.loads(_canonical(static))
    key_injection["renderer_contract"][f"note_{private_id}"] = "forbidden"
    with pytest.raises(PlannedTrainingSurfaceBridgeV1Error, match="noncanonical"):
        validate_planned_model_visible_static_data_v1(
            key_injection,
            evaluator_private_strings=(private_id,),
        )

    tuple_injection = json.loads(_canonical(static))
    tuple_injection["renderer_contract"]["note"] = ({"official_rule_id": "x"},)
    with pytest.raises(PlannedTrainingSurfaceBridgeV1Error, match="noncanonical"):
        validate_planned_model_visible_static_data_v1(tuple_injection)

    for outer_key, nested_key in (
        ("xruntime", "fooClassification"),
        ("xofficial", "innocent"),
    ):
        substring_key_injection = json.loads(_canonical(static))
        substring_key_injection["renderer_contract"][outer_key] = {nested_key: "secret"}
        with pytest.raises(PlannedTrainingSurfaceBridgeV1Error, match="noncanonical"):
            validate_planned_model_visible_static_data_v1(substring_key_injection)


def test_direct_dataclass_construction_cannot_forge_internal_surface_claims(
    plan: HypothesisCompleteTrainingPlanV1,
) -> None:
    _, bridge, _ = _bridge_and_text(plan)
    block = bridge.blocks[0]

    forged_truths = list(block.live_rule_bindings)
    forged_truths[0] = (forged_truths[0][0], _digest("forged-catalog-truth"))
    with pytest.raises(PlannedTrainingSurfaceBridgeV1Error, match="public catalog"):
        replace(block, live_rule_bindings=tuple(forged_truths))

    changed_steps = list(block.rotations)
    changed_steps[0] = replace(
        changed_steps[0],
        planned_optimizer_step_before=changed_steps[0].planned_optimizer_step_before + 1,
        planned_optimizer_step_after_objective_collection=(
            changed_steps[0].planned_optimizer_step_after_objective_collection + 1
        ),
    )
    with pytest.raises(PlannedTrainingSurfaceBridgeV1Error, match="one planned pre-update"):
        replace(block, rotations=tuple(changed_steps))

    injected_static = json.loads(block.static_model_input_json)
    injected_static["opening_examples"][0]["private_metadata"] = "forged"
    with pytest.raises(
        PlannedTrainingSurfaceBridgeV1Error,
        match=r"semantic digest|private/runtime",
    ):
        replace(block, static_model_input_json=_canonical(injected_static))

    with pytest.raises(PlannedTrainingSurfaceBridgeV1Error, match="registered budget"):
        replace(bridge, registered_episode_budget=16)

    arbitrary_static = json.loads(block.static_model_input_json)
    arbitrary_static["opening_examples"][0]["scene_text"] = "arbitrary schema-valid scene text"
    arbitrary_digest = _static_digest(arbitrary_static)
    arbitrary_rotations = tuple(
        replace(row, static_model_input_digest=arbitrary_digest) for row in block.rotations
    )
    internally_valid = replace(
        block,
        static_model_input_json=_canonical(arbitrary_static),
        static_model_input_digest=arbitrary_digest,
        rotations=arbitrary_rotations,
    )
    separation = internally_valid.as_obj()["surface_separation"]
    assert separation["model_visible_static_data_internally_validated"] is True
    assert separation["exact_plan_backed_rederivation_required"] is True
    assert separation["standalone_block_plan_binding_verified"] is False
    assert "model_visible_static_data_exactly_copied_from_plan" not in separation


def test_wrong_plan_or_external_digest_and_static_tamper_fail_closed(
    plan: HypothesisCompleteTrainingPlanV1,
) -> None:
    plan_text, bridge, encoded = _bridge_and_text(plan)
    other = _plan(block_id="other-surface-block")
    other_text = serialize_hypothesis_complete_training_plan_v1(other)
    with pytest.raises(PlannedTrainingSurfaceBridgeV1Error, match="externally expected"):
        parse_planned_training_surface_bridge_v1(
            encoded,
            plan_text=other_text,
            expected_plan_digest=other.digest,
            expected_bridge_digest=bridge.digest,
        )
    with pytest.raises(PlannedTrainingSurfaceBridgeV1Error, match="externally expected"):
        parse_planned_training_surface_bridge_v1(
            encoded,
            plan_text=plan_text,
            expected_plan_digest=plan.digest,
            expected_bridge_digest=_digest("wrong-bridge"),
        )
    with pytest.raises(PlannedTrainingSurfaceBridgeV1Error, match="canonical replay"):
        build_planned_training_surface_bridge_v1(
            plan_text + "\n",
            expected_plan_digest=plan.digest,
        )

    tampered = json.loads(encoded)
    examples = tampered["blocks"][0]["model_visible_static_data"]["opening_examples"]
    examples[0]["accepted"] = not examples[0]["accepted"]
    with pytest.raises(PlannedTrainingSurfaceBridgeV1Error, match="exact training-plan"):
        parse_planned_training_surface_bridge_v1(
            _canonical(tampered),
            plan_text=plan_text,
            expected_plan_digest=plan.digest,
            expected_bridge_digest=bridge.digest,
        )


def test_authorization_and_all_unresolved_runtime_claims_are_immutable_false(
    plan: HypothesisCompleteTrainingPlanV1,
) -> None:
    plan_text, bridge, encoded = _bridge_and_text(plan)
    value = json.loads(encoded)
    authorization = value["authorization"]
    assert authorization["scope"].endswith("engineering-only")
    assert set(authorization.values()) - {authorization["scope"]} == {False}
    boundary = value["unresolved_runtime_boundary"]
    assert boundary["scope"] == "static-model-input-object-only"
    assert set(boundary.values()) - {boundary["scope"]} == {False}

    claimed = json.loads(encoded)
    claimed["unresolved_runtime_boundary"]["input_token_ids_bound"] = True
    with pytest.raises(PlannedTrainingSurfaceBridgeV1Error, match="nonauthorizing contract"):
        parse_planned_training_surface_bridge_v1(
            _canonical(claimed),
            plan_text=plan_text,
            expected_plan_digest=plan.digest,
            expected_bridge_digest=bridge.digest,
        )

    authorized = json.loads(encoded)
    authorized["authorization"]["launch_authorized"] = True
    with pytest.raises(PlannedTrainingSurfaceBridgeV1Error, match="nonauthorizing contract"):
        parse_planned_training_surface_bridge_v1(
            _canonical(authorized),
            plan_text=plan_text,
            expected_plan_digest=plan.digest,
            expected_bridge_digest=bridge.digest,
        )

    numeric_false = json.loads(encoded)
    numeric_false["authorization"]["launch_authorized"] = 0
    with pytest.raises(PlannedTrainingSurfaceBridgeV1Error, match="nonauthorizing contract"):
        parse_planned_training_surface_bridge_v1(
            _canonical(numeric_false),
            plan_text=plan_text,
            expected_plan_digest=plan.digest,
            expected_bridge_digest=bridge.digest,
        )

    numeric_boundary_false = json.loads(encoded)
    numeric_boundary_false["unresolved_runtime_boundary"]["model_calls_observed"] = 0
    with pytest.raises(PlannedTrainingSurfaceBridgeV1Error, match="nonauthorizing contract"):
        parse_planned_training_surface_bridge_v1(
            _canonical(numeric_boundary_false),
            plan_text=plan_text,
            expected_plan_digest=plan.digest,
            expected_bridge_digest=bridge.digest,
        )


def test_canonical_duplicate_reordering_and_boolean_aliases_are_rejected(
    plan: HypothesisCompleteTrainingPlanV1,
) -> None:
    plan_text, bridge, encoded = _bridge_and_text(plan)
    duplicate = encoded[:-1] + ',"schema_version":1}'
    with pytest.raises(PlannedTrainingSurfaceBridgeV1Error, match="duplicate"):
        parse_planned_training_surface_bridge_v1(
            duplicate,
            plan_text=plan_text,
            expected_plan_digest=plan.digest,
            expected_bridge_digest=bridge.digest,
        )

    reordered = json.loads(encoded)
    authorization = reordered["authorization"]
    reordered["authorization"] = dict(reversed(tuple(authorization.items())))
    with pytest.raises(PlannedTrainingSurfaceBridgeV1Error, match="reordered"):
        parse_planned_training_surface_bridge_v1(
            _canonical(reordered),
            plan_text=plan_text,
            expected_plan_digest=plan.digest,
            expected_bridge_digest=bridge.digest,
        )

    boolean = json.loads(encoded)
    boolean["blocks"][0]["bank_position"] = True
    with pytest.raises(PlannedTrainingSurfaceBridgeV1Error, match="exact training-plan"):
        parse_planned_training_surface_bridge_v1(
            _canonical(boolean),
            plan_text=plan_text,
            expected_plan_digest=plan.digest,
            expected_bridge_digest=bridge.digest,
        )

    with pytest.raises(PlannedTrainingSurfaceBridgeV1Error, match="canonical compact"):
        parse_planned_training_surface_bridge_v1(
            encoded + "\n",
            plan_text=plan_text,
            expected_plan_digest=plan.digest,
            expected_bridge_digest=bridge.digest,
        )


def test_terminal_law_type_annotation_remains_exact() -> None:
    # Guard the fixture path used above against an accidental broad Any cast.
    law = _terminal_fixture(_opening(0))[0]
    assert type(law) is UnconditionalTerminalSceneLawV2
