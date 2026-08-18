from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from typing import Any

import pytest

from goalzendo_interactive_v2.hypothesis_complete import (
    build_hidden_order_display_binding_v2,
    build_hypothesis_complete_training_manifest_v2,
    serialize_hypothesis_complete_training_manifest_v2,
)
from goalzendo_interactive_v2.training_freeze import (
    AtomicBlockExecutionReceiptV1,
    HypothesisCompleteTrainingPlanV1,
    TrainingExecutionReceiptV1,
    TrainingFreezeBindingsV1,
    TrainingFreezeV1Error,
    build_atomic_block_execution_receipt_v1,
    build_hypothesis_complete_training_block_plan_v1,
    build_hypothesis_complete_training_plan_v1,
    build_training_execution_receipt_v1,
    parse_hypothesis_complete_training_plan_v1,
    parse_training_execution_receipt_v1,
    serialize_hypothesis_complete_training_plan_v1,
    serialize_training_execution_receipt_v1,
    training_execution_receipt_v1_from_obj,
    verify_training_execution_receipt_v1,
)
from tests.goalzendo_interactive_v2.test_hypothesis_complete import (
    _block as legacy_execution_block,
)
from tests.goalzendo_interactive_v2.test_hypothesis_complete import (
    _opening,
    _terminal_fixture,
)


def _digest(label: str) -> str:
    return hashlib.sha256(label.encode("ascii")).hexdigest()


def _canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=True, allow_nan=False, separators=(",", ":"))


@pytest.fixture(scope="module")
def initial_checkpoint_bytes() -> bytes:
    return b'{"checkpoint":"initial","tensor_manifest":"sha256-list-v1"}'


@pytest.fixture(scope="module")
def plan(initial_checkpoint_bytes: bytes) -> HypothesisCompleteTrainingPlanV1:
    blocks = []
    for index in range(2):
        opening = _opening(index)
        law, panel = _terminal_fixture(opening)
        hidden = build_hidden_order_display_binding_v2(
            opening,
            independence_precommitment_digest=_digest(f"freeze-order-{index}"),
        )
        blocks.append(
            build_hypothesis_complete_training_block_plan_v1(
                opening,
                law,
                panel,
                hidden,
                bank_position=index,
                block_id=f"freeze-block-{index}",
                renderer_name="train_positional",
                update_batch_id=f"freeze-batch-{index}",
                planned_optimizer_step_before=800 + index,
            )
        )
    bindings = TrainingFreezeBindingsV1(
        source_tree_sha256=_digest("source-tree"),
        bank_generator_source_sha256=_digest("bank-generator"),
        initial_checkpoint_manifest_sha256=hashlib.sha256(initial_checkpoint_bytes).hexdigest(),
        tokenizer_binding_digest=_digest("tokenizer"),
        optimizer_config_digest=_digest("optimizer-config"),
    )
    return build_hypothesis_complete_training_plan_v1(
        bindings,
        blocks,
        registered_episode_budget=16,
        engineering_budget_override=True,
    )


def _checkpoint_chain(
    initial_checkpoint_bytes: bytes,
    *,
    changed: bool = True,
) -> dict[int, tuple[bytes, bytes]]:
    middle = b'{"checkpoint":"middle"}' if changed else initial_checkpoint_bytes
    final = b'{"checkpoint":"final"}' if changed else initial_checkpoint_bytes
    return {
        0: (initial_checkpoint_bytes, middle),
        1: (middle, final),
    }


def _block_receipt(
    plan: HypothesisCompleteTrainingPlanV1,
    position: int,
    checkpoint_bytes: tuple[bytes, bytes],
    *,
    observed_steps_after_objective_collection: tuple[int, ...] | None = None,
) -> AtomicBlockExecutionReceiptV1:
    block = plan.blocks[position]
    truth = (True,) * block.n0
    falsehood = (False,) * block.n0
    return build_atomic_block_execution_receipt_v1(
        block,
        objective_evidence_digests=tuple(
            _digest(f"objective-{position}-{rotation}") for rotation in range(block.n0)
        ),
        pre_checkpoint_manifest_bytes=checkpoint_bytes[0],
        post_checkpoint_manifest_bytes=checkpoint_bytes[1],
        observed_optimizer_step_before=block.planned_optimizer_step_before,
        observed_optimizer_step_after=block.planned_optimizer_step_after,
        observed_optimizer_steps_after_objective_collection=(
            observed_steps_after_objective_collection
            if observed_steps_after_objective_collection is not None
            else (block.planned_optimizer_step_before,) * block.n0
        ),
        context_resets_observed=truth,
        cache_resets_observed=truth,
        objective_collections_observed=truth,
        parameter_updates_before_commit_observed=falsehood,
        observed_parameter_updates_before_commit=0,
        observed_atomic_commit_count=1,
        observed_all_objectives_collected_before_commit=True,
    )


def _receipt(
    plan: HypothesisCompleteTrainingPlanV1,
    evidence: dict[int, tuple[bytes, bytes]],
) -> TrainingExecutionReceiptV1:
    return build_training_execution_receipt_v1(
        plan,
        tuple(_block_receipt(plan, position, evidence[position]) for position in range(2)),
        checkpoint_manifest_bytes_by_block=evidence,
    )


def test_plan_exists_before_outcomes_and_round_trips_canonically(
    plan: HypothesisCompleteTrainingPlanV1,
) -> None:
    encoded = serialize_hypothesis_complete_training_plan_v1(plan)
    assert parse_hypothesis_complete_training_plan_v1(
        encoded,
        expected_digest=plan.digest,
    ) == plan
    assert plan.episode_count == 16
    assert plan.n0 == 8
    assert '"post_update_checkpoint_digest"' not in encoded
    assert '"pre_update_checkpoint_digest"' not in encoded
    assert '"commit_disposition"' not in encoded
    assert '"context_reset_observed"' not in encoded
    assert '"objective_evidence_digest"' not in encoded
    assert json.loads(encoded)["prospective_boundary"] == {
        "all_plan_fields_available_before_model_execution": True,
        "contains_checkpoint_or_outcome_evidence": False,
        "post_run_receipt_required": True,
        "plan_digest_independent_of_realized_execution": True,
    }


def test_plan_rejects_nested_outcome_fields_and_legacy_execution_manifest(
    plan: HypothesisCompleteTrainingPlanV1,
) -> None:
    tampered = plan.as_obj()
    tampered["blocks"][0]["post_update_checkpoint_digest"] = _digest("future")
    with pytest.raises(TrainingFreezeV1Error, match="forbidden outcome/runtime field"):
        parse_hypothesis_complete_training_plan_v1(_canonical(tampered))

    legacy_blocks = (legacy_execution_block(0), legacy_execution_block(1))
    legacy = build_hypothesis_complete_training_manifest_v2(
        legacy_blocks,
        registered_episode_budget=16,
        engineering_budget_override=True,
    )
    with pytest.raises(TrainingFreezeV1Error, match="forbidden outcome/runtime field"):
        parse_hypothesis_complete_training_plan_v1(
            serialize_hypothesis_complete_training_manifest_v2(legacy)
        )


def test_plan_parser_rejects_tamper_reorder_duplicate_and_boolean_alias(
    plan: HypothesisCompleteTrainingPlanV1,
) -> None:
    obj = plan.as_obj()
    obj["pre_run_bindings"]["optimizer_config_digest"] = _digest("changed")
    with pytest.raises(TrainingFreezeV1Error, match=r"tampered|digest differs"):
        parse_hypothesis_complete_training_plan_v1(_canonical(obj))

    reordered = plan.as_obj()
    reordered["blocks"] = list(reversed(reordered["blocks"]))
    with pytest.raises(TrainingFreezeV1Error, match="bank position"):
        parse_hypothesis_complete_training_plan_v1(_canonical(reordered))

    boolean = plan.as_obj()
    boolean["blocks"][0]["bank_position"] = True
    with pytest.raises(TrainingFreezeV1Error, match="integer"):
        parse_hypothesis_complete_training_plan_v1(_canonical(boolean))

    encoded = serialize_hypothesis_complete_training_plan_v1(plan)
    duplicate = encoded[:-1] + ',"schema_version":1}'
    with pytest.raises(TrainingFreezeV1Error, match="duplicate"):
        parse_hypothesis_complete_training_plan_v1(duplicate)

    whitespace = encoded + "\n"
    with pytest.raises(TrainingFreezeV1Error, match="canonical compact"):
        parse_hypothesis_complete_training_plan_v1(whitespace)


def test_receipt_requires_exact_plan_and_exact_checkpoint_bytes(
    plan: HypothesisCompleteTrainingPlanV1,
    initial_checkpoint_bytes: bytes,
) -> None:
    evidence = _checkpoint_chain(initial_checkpoint_bytes)
    receipt = _receipt(plan, evidence)
    encoded = serialize_training_execution_receipt_v1(
        receipt,
        plan,
        checkpoint_manifest_bytes_by_block=evidence,
    )
    assert parse_training_execution_receipt_v1(
        encoded,
        plan=plan,
        checkpoint_manifest_bytes_by_block=evidence,
        expected_digest=receipt.digest,
    ) == receipt

    wrong_plan = replace(
        plan,
        bindings=replace(plan.bindings, tokenizer_binding_digest=_digest("other-tokenizer")),
    )
    with pytest.raises(TrainingFreezeV1Error, match="different training plan"):
        parse_training_execution_receipt_v1(
            encoded,
            plan=wrong_plan,
            checkpoint_manifest_bytes_by_block=evidence,
        )
    wrong_bytes = dict(evidence)
    wrong_bytes[1] = (evidence[1][0], b"different-post-checkpoint")
    with pytest.raises(TrainingFreezeV1Error, match="post-checkpoint manifest bytes"):
        parse_training_execution_receipt_v1(
            encoded,
            plan=plan,
            checkpoint_manifest_bytes_by_block=wrong_bytes,
        )
    with pytest.raises(TrainingFreezeV1Error, match="cover exactly every"):
        parse_training_execution_receipt_v1(
            encoded,
            plan=plan,
            checkpoint_manifest_bytes_by_block={0: evidence[0]},
        )


def test_plan_digest_is_independent_of_changed_or_zero_gradient_execution(
    plan: HypothesisCompleteTrainingPlanV1,
    initial_checkpoint_bytes: bytes,
) -> None:
    changed_evidence = _checkpoint_chain(initial_checkpoint_bytes, changed=True)
    zero_evidence = _checkpoint_chain(initial_checkpoint_bytes, changed=False)
    changed = _receipt(plan, changed_evidence)
    zero = _receipt(plan, zero_evidence)
    assert changed.training_plan_digest == zero.training_plan_digest == plan.digest
    assert changed.digest != zero.digest
    assert all(
        block.commit_disposition == "committed_state_changed" for block in changed.blocks
    )
    assert all(
        block.commit_disposition == "zero_gradient_no_state_change" for block in zero.blocks
    )


def test_reset_early_update_atomic_and_step_tampering_fail_closed(
    plan: HypothesisCompleteTrainingPlanV1,
    initial_checkpoint_bytes: bytes,
) -> None:
    evidence = _checkpoint_chain(initial_checkpoint_bytes)
    receipt = _receipt(plan, evidence)

    for field, value, message in (
        ("context_reset_observed", False, "context and cache resets"),
        ("cache_reset_observed", False, "context and cache resets"),
        ("objective_collection_observed", False, "objective must be observed"),
        ("parameter_update_before_commit_observed", True, "early parameter update"),
    ):
        obj = receipt.as_obj()
        obj["blocks"][0]["rotations"][0][field] = value
        with pytest.raises(TrainingFreezeV1Error, match=message):
            training_execution_receipt_v1_from_obj(obj)

    atomic = receipt.as_obj()
    atomic["blocks"][0]["atomic_commit_count"] = 2
    with pytest.raises(TrainingFreezeV1Error, match="exactly one commit"):
        training_execution_receipt_v1_from_obj(atomic)

    step = receipt.as_obj()
    step["blocks"][0]["optimizer_step_after"] += 1
    with pytest.raises(TrainingFreezeV1Error, match="exactly one optimizer step"):
        training_execution_receipt_v1_from_obj(step)

    block = plan.blocks[0]
    early_step = (
        block.planned_optimizer_step_before + 1,
        *((block.planned_optimizer_step_before,) * (block.n0 - 1)),
    )
    with pytest.raises(TrainingFreezeV1Error, match="update occurred before the atomic commit"):
        _block_receipt(
            plan,
            0,
            evidence[0],
            observed_steps_after_objective_collection=early_step,
        )

    wrong_rule_type = receipt.as_obj()
    wrong_rule_type["blocks"][0]["rotations"][0]["official_rule_id"] = 7
    with pytest.raises(TrainingFreezeV1Error, match="rule identity must be a string"):
        training_execution_receipt_v1_from_obj(wrong_rule_type)


def test_receipt_rejects_missing_reordered_extra_and_discontinuous_blocks(
    plan: HypothesisCompleteTrainingPlanV1,
    initial_checkpoint_bytes: bytes,
) -> None:
    evidence = _checkpoint_chain(initial_checkpoint_bytes)
    receipt = _receipt(plan, evidence)

    missing = TrainingExecutionReceiptV1(plan.digest, receipt.blocks[:1])
    with pytest.raises(TrainingFreezeV1Error, match="cover every planned block"):
        verify_training_execution_receipt_v1(
            missing,
            plan,
            checkpoint_manifest_bytes_by_block={0: evidence[0]},
        )

    with pytest.raises(TrainingFreezeV1Error, match="bank position"):
        TrainingExecutionReceiptV1(plan.digest, tuple(reversed(receipt.blocks)))

    extra_block = replace(
        receipt.blocks[1],
        bank_position=2,
        rotations=tuple(
            replace(
                rotation,
                objective_evidence_digest=_digest(f"extra-objective-{position}"),
            )
            for position, rotation in enumerate(receipt.blocks[1].rotations)
        ),
    )
    extra = TrainingExecutionReceiptV1(plan.digest, (*receipt.blocks, extra_block))
    extra_evidence = {**evidence, 2: evidence[1]}
    with pytest.raises(TrainingFreezeV1Error, match="cover every planned block"):
        verify_training_execution_receipt_v1(
            extra,
            plan,
            checkpoint_manifest_bytes_by_block=extra_evidence,
        )

    discontinuous_evidence = {
        0: evidence[0],
        1: (b"unrelated-pre-checkpoint", evidence[1][1]),
    }
    discontinuous_blocks = (
        receipt.blocks[0],
        _block_receipt(plan, 1, discontinuous_evidence[1]),
    )
    discontinuous = TrainingExecutionReceiptV1(plan.digest, discontinuous_blocks)
    with pytest.raises(TrainingFreezeV1Error, match="byte chain is discontinuous"):
        verify_training_execution_receipt_v1(
            discontinuous,
            plan,
            checkpoint_manifest_bytes_by_block=discontinuous_evidence,
        )


def test_cross_parsers_authorization_and_expected_digest_fail_closed(
    plan: HypothesisCompleteTrainingPlanV1,
    initial_checkpoint_bytes: bytes,
) -> None:
    evidence = _checkpoint_chain(initial_checkpoint_bytes)
    receipt = _receipt(plan, evidence)
    plan_text = serialize_hypothesis_complete_training_plan_v1(plan)
    receipt_text = serialize_training_execution_receipt_v1(
        receipt,
        plan,
        checkpoint_manifest_bytes_by_block=evidence,
    )
    with pytest.raises(TrainingFreezeV1Error):
        training_execution_receipt_v1_from_obj(json.loads(plan_text))
    with pytest.raises(TrainingFreezeV1Error, match="forbidden outcome/runtime field"):
        parse_hypothesis_complete_training_plan_v1(receipt_text)

    authorization = receipt.as_obj()
    authorization["authorization"]["launch_authorized"] = True
    with pytest.raises(TrainingFreezeV1Error, match="authorization"):
        parse_training_execution_receipt_v1(
            _canonical(authorization),
            plan=plan,
            checkpoint_manifest_bytes_by_block=evidence,
        )
    with pytest.raises(TrainingFreezeV1Error, match="externally expected digest"):
        parse_training_execution_receipt_v1(
            receipt_text,
            plan=plan,
            checkpoint_manifest_bytes_by_block=evidence,
            expected_digest=_digest("wrong-receipt"),
        )
