"""Exact planned-training static-surface bridge for prospective G03-v2.

The prospective training plan binds a complete role-neutral static payload for
each hypothesis-complete block, while the statistical leakage schemas describe
only a narrower rendered-opening surface.  This module closes that *static*
boundary: it consumes exact canonical
:class:`~goalzendo_interactive_v2.training_freeze.HypothesisCompleteTrainingPlanV1`
JSON, copies every complete model-visible static object, and binds all
evaluator-private rotation data in a separate subtree.

It deliberately stops before a runtime prompt.  No system message, supported
grammar, chat template, tokenizer artifact, input-token sequence, dynamic
feedback, private-AST sibling, AST-blind classification sibling, reset, or
optimizer event is observed here.  Every such boundary is hard-coded false,
as are all production and launch authorizations.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any, cast

from goalzendo_interactive.catalog import build_rule_catalog
from goalzendo_interactive.rendering import TRAIN_RENDERERS, renderer_digest

from .hypothesis_complete import (
    HypothesisCompleteV2Error,
    unconditional_terminal_scene_law_v2_from_obj,
)
from .training_freeze import (
    HypothesisCompleteTrainingBlockPlanV1,
    HypothesisCompleteTrainingPlanV1,
    TrainingFreezeV1Error,
    parse_hypothesis_complete_training_plan_v1,
)

PLANNED_TRAINING_SURFACE_BRIDGE_SCHEMA_VERSION = 1

_BRIDGE_KIND = "g03-v2-planned-training-static-surface-bridge-v1"
_PLAN_STATIC_INPUT_DOMAIN = "goalzendo-interactive-v2-training-freeze-static-input-v1"
_BLOCK_DOMAIN = "goalzendo-interactive-v2-planned-training-static-surface-block-v1"
_PRIVATE_DOMAIN = "goalzendo-interactive-v2-planned-training-private-bindings-v1"
_POPULATION_DOMAIN = "goalzendo-interactive-v2-planned-training-static-surface-population-v1"
_BRIDGE_DOMAIN = "goalzendo-interactive-v2-planned-training-static-surface-bridge-v1"

_IDENTIFIER = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,127}")
_RULE_ID = re.compile(r"g03r[0-9]{5}")

_AUTHORIZATION: dict[str, bool | str] = {
    "scope": "prospective-planned-training-static-surface-engineering-only",
    "production_training_bank_authorized": False,
    "model_execution_authorized": False,
    "weight_updates_authorized": False,
    "launch_authorized": False,
}

_BOUNDARY: dict[str, bool | str] = {
    "scope": "static-model-input-object-only",
    "runtime_prompt_bytes_bound": False,
    "runtime_messages_bound": False,
    "system_instructions_bound": False,
    "supported_grammar_bytes_bound": False,
    "action_decoder_bytes_bound": False,
    "chat_template_bytes_bound": False,
    "tokenizer_artifact_bytes_bound": False,
    "tokenizer_execution_observed": False,
    "generation_prefix_bound": False,
    "input_token_ids_bound": False,
    "model_calls_observed": False,
    "dynamic_messages_bound": False,
    "inquiry_feedback_runtime_bound": False,
    "private_ast_sibling_bound": False,
    "ast_blind_one_item_siblings_bound": False,
    "runtime_receipt_bound": False,
    "context_and_cache_resets_observed": False,
    "optimizer_execution_observed": False,
    "optimizer_atomicity_observed": False,
    "full_runtime_bridge_verified": False,
}

# These keys must never enter the copied model-visible static object.  The
# whitelist in training_freeze is still authoritative; this is an independent
# fail-closed check at the new boundary.  ``scene_index`` cannot be forbidden:
# the public terminal-law support intentionally contains scene identities.
_FORBIDDEN_VISIBLE_KEYS = frozenset(
    {
        "official",
        "official_rule_id",
        "official_truth_digest",
        "candidate_role",
        "p",
        "q",
        "a",
        "b",
        "cover",
        "stage",
        "rotation_position",
        "schedule_position",
        "request_position",
        "bank_position",
        "bank_prefix",
        "block_id",
        "opening_id",
        "opening_digest",
        "opening_content_digest",
        "opening_scene_set_digest",
        "rule_id",
        "truth_digest",
        "panel_id",
        "panel_scene_indices",
        "materialized_training_panel_digest",
        "external_generator_receipt_digest",
        "hidden_order_display_binding_digest",
        "independence_precommitment_digest",
        "update_batch_id",
        "planned_optimizer_step_before",
        "planned_optimizer_step_after",
        "optimizer_step",
        "checkpoint_digest",
        "initial_checkpoint_manifest_sha256",
        "ast",
        "rule_ast",
        "classification",
        "classifications",
        "item_rank",
        "panel_marker",
        "remaining_count",
        "runtime_receipt",
        "runtime_manifest",
        "prompt_bytes",
        "prompt_text",
        "chat_template",
        "generation_prefix",
        "token_ids",
        "input_ids",
    }
)

_FORBIDDEN_VISIBLE_KEY_TOKENS = frozenset(
    {
        "official",
        "candidate",
        "role",
        "p",
        "q",
        "a",
        "b",
        "cover",
        "stage",
        "rotation",
        "schedule",
        "request",
        "bank",
        "block",
        "opening_id",
        "opening_digest",
        "truth",
        "rule_id",
        "panel",
        "hidden",
        "order",
        "private",
        "precommitment",
        "update",
        "optimizer",
        "checkpoint",
        "ast",
        "classification",
        "rank",
        "remaining",
        "runtime",
        "receipt",
        "prompt",
        "message",
        "chat",
        "generation",
        "token",
        "input_id",
    }
)


class PlannedTrainingSurfaceBridgeV1Error(ValueError):
    """Raised when the static-surface bridge fails exact replay."""


def _dump_json(value: Any) -> str:
    try:
        return json.dumps(value, ensure_ascii=True, allow_nan=False, separators=(",", ":"))
    except (TypeError, ValueError) as exc:
        raise PlannedTrainingSurfaceBridgeV1Error(f"value is not canonical JSON: {exc}") from exc


def _load_json(text: str) -> Any:
    if type(text) is not str or not text:
        raise PlannedTrainingSurfaceBridgeV1Error("JSON input must be nonempty text")

    def no_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise PlannedTrainingSurfaceBridgeV1Error(f"duplicate JSON object key: {key!r}")
            result[key] = value
        return result

    def reject_constant(value: str) -> None:
        raise PlannedTrainingSurfaceBridgeV1Error(f"non-finite JSON constant is forbidden: {value}")

    try:
        return json.loads(
            text,
            object_pairs_hook=no_duplicates,
            parse_constant=reject_constant,
        )
    except PlannedTrainingSurfaceBridgeV1Error:
        raise
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise PlannedTrainingSurfaceBridgeV1Error(f"invalid JSON: {exc}") from exc


def _json_digest(value: Any, *, domain: str) -> str:
    digest = hashlib.sha256()
    digest.update(domain.encode("ascii"))
    digest.update(b"\0")
    digest.update(_dump_json(value).encode("ascii"))
    return digest.hexdigest()


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _is_sha256(value: object) -> bool:
    return (
        type(value) is str
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _require_sha256(value: object, *, name: str) -> str:
    if not _is_sha256(value):
        raise PlannedTrainingSurfaceBridgeV1Error(f"{name} must be a lowercase SHA-256")
    return cast(str, value)


def _require_integer(value: object, *, name: str, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise PlannedTrainingSurfaceBridgeV1Error(f"{name} must be an integer >= {minimum}")
    return value


def _require_identifier(value: object, *, name: str) -> str:
    if type(value) is not str or _IDENTIFIER.fullmatch(value) is None:
        raise PlannedTrainingSurfaceBridgeV1Error(f"{name} is not a canonical identifier")
    return value


def _require_catalog_rule_binding(rule_id: str, truth_digest: str) -> None:
    if _RULE_ID.fullmatch(rule_id) is None:
        raise PlannedTrainingSurfaceBridgeV1Error(f"malformed public rule identity: {rule_id!r}")
    index = int(rule_id[4:])
    catalog = build_rule_catalog()
    if not 0 <= index < len(catalog) or catalog[index].rule_id != rule_id:
        raise PlannedTrainingSurfaceBridgeV1Error(f"unknown public rule identity: {rule_id!r}")
    if catalog[index].truth_digest != truth_digest:
        raise PlannedTrainingSurfaceBridgeV1Error("live-rule truth digest differs from the public catalog")


def _require_mapping(
    value: object,
    fields: tuple[str, ...],
    *,
    name: str,
) -> Mapping[str, Any]:
    if type(value) is not dict or tuple(value) != fields:
        raise PlannedTrainingSurfaceBridgeV1Error(
            f"{name} has noncanonical, missing, extra, or reordered fields"
        )
    return cast(Mapping[str, Any], value)


def _require_constant_mapping(
    value: object,
    expected: Mapping[str, bool | str],
    *,
    name: str,
) -> None:
    obj = _require_mapping(value, tuple(expected), name=name)
    if any(
        type(obj[key]) is not type(expected_value) or obj[key] != expected_value
        for key, expected_value in expected.items()
    ):
        raise PlannedTrainingSurfaceBridgeV1Error(f"{name} differs from its exact nonauthorizing contract")


def _canonical_plan(
    plan_text: str,
    *,
    expected_plan_digest: str,
) -> tuple[HypothesisCompleteTrainingPlanV1, bytes]:
    _require_sha256(expected_plan_digest, name="expected training-plan digest")
    try:
        plan = parse_hypothesis_complete_training_plan_v1(
            plan_text,
            expected_digest=expected_plan_digest,
        )
    except (TrainingFreezeV1Error, TypeError, ValueError) as exc:
        raise PlannedTrainingSurfaceBridgeV1Error(
            f"training plan failed exact canonical replay: {exc}"
        ) from exc
    try:
        plan_bytes = plan_text.encode("ascii")
    except UnicodeEncodeError as exc:  # pragma: no cover - upstream canonical JSON rejects this
        raise PlannedTrainingSurfaceBridgeV1Error("training plan must be canonical ASCII JSON") from exc
    return plan, plan_bytes


def _private_string_leaves(block: HypothesisCompleteTrainingBlockPlanV1) -> frozenset[str]:
    catalog = build_rule_catalog()
    values = {
        block.block_id,
        block.opening.opening_id,
        block.opening.digest,
        block.opening.content_digest,
        block.opening.scene_set_digest,
        block.materialized_training_panel.digest,
        block.materialized_training_panel.external_generator_receipt_digest,
        block.hidden_order_display_binding.digest,
        block.hidden_order_display_binding.independence_precommitment_digest,
        block.update_batch_id,
        *block.opening.version_space_rule_ids,
        *block.opening.version_space_truth_digests,
    }
    values.update(
        catalog[int(rule_id[4:])].truth_digest
        for rule_id in block.hidden_order_display_binding.official_rotation_order_rule_ids
    )
    return frozenset(values)


def validate_planned_model_visible_static_data_v1(
    value: object,
    *,
    evaluator_private_strings: Iterable[str] = (),
) -> Mapping[str, Any]:
    """Validate the exact static-payload boundary without claiming a prompt.

    This helper is intentionally usable in adversarial tests.  It enforces the
    complete recursive static schema and replays the public terminal law.  The
    exact prospective-plan parser additionally proves that rendered scene text
    came from the plan's hidden display order and registered renderer.
    """

    obj = _require_mapping(
        value,
        (
            "opening_examples",
            "renderer_contract",
            "public_train_terminal_scene_law",
        ),
        name="model-visible static data",
    )
    raw_examples = obj["opening_examples"]
    if type(raw_examples) is not list or len(raw_examples) != 10:
        raise PlannedTrainingSurfaceBridgeV1Error(
            "model-visible static data requires exactly ten opening examples"
        )
    for position, raw_example in enumerate(raw_examples):
        example = _require_mapping(
            raw_example,
            ("accepted", "scene_text"),
            name=f"model-visible opening example {position}",
        )
        if type(example["accepted"]) is not bool:
            raise PlannedTrainingSurfaceBridgeV1Error(
                f"model-visible opening example {position} accepted must be Boolean"
            )
        if type(example["scene_text"]) is not str or not example["scene_text"]:
            raise PlannedTrainingSurfaceBridgeV1Error(
                f"model-visible opening example {position} scene_text must be nonempty text"
            )
    renderer = _require_mapping(
        obj["renderer_contract"],
        ("renderer_name", "renderer_registry_digest"),
        name="model-visible renderer contract",
    )
    if type(renderer["renderer_name"]) is not str or renderer["renderer_name"] not in TRAIN_RENDERERS:
        raise PlannedTrainingSurfaceBridgeV1Error(
            "model-visible renderer contract requires a registered training renderer"
        )
    if renderer["renderer_registry_digest"] != renderer_digest():
        raise PlannedTrainingSurfaceBridgeV1Error(
            "model-visible renderer registry differs from the live registry"
        )
    try:
        unconditional_terminal_scene_law_v2_from_obj(obj["public_train_terminal_scene_law"])
    except (HypothesisCompleteV2Error, TypeError, ValueError) as exc:
        raise PlannedTrainingSurfaceBridgeV1Error(f"public terminal law failed exact replay: {exc}") from exc
    private_strings = frozenset(evaluator_private_strings)
    if any(type(item) is not str or not item for item in private_strings):
        raise PlannedTrainingSurfaceBridgeV1Error("evaluator-private markers must be nonempty strings")

    def walk(child: object, *, path: str) -> None:
        if type(child) is dict:
            for raw_key, nested in cast(Mapping[str, Any], child).items():
                if type(raw_key) is not str:
                    raise PlannedTrainingSurfaceBridgeV1Error(f"model-visible key at {path} is not text")
                normalized = raw_key.lower().replace("-", "_")
                tokens = tuple(part for part in normalized.split("_") if part)
                forbidden_family = any(
                    token in _FORBIDDEN_VISIBLE_KEY_TOKENS
                    or any(
                        token.startswith(prefix)
                        for prefix in (
                            "official",
                            "runtime",
                            "prompt",
                            "token",
                            "optimizer",
                            "checkpoint",
                            "classification",
                            "rotation",
                            "precommit",
                        )
                    )
                    for token in tokens
                )
                if normalized in _FORBIDDEN_VISIBLE_KEYS or forbidden_family:
                    raise PlannedTrainingSurfaceBridgeV1Error(
                        f"private/runtime field entered model-visible data at {path}.{raw_key}"
                    )
                if any(marker in raw_key for marker in private_strings):
                    raise PlannedTrainingSurfaceBridgeV1Error(
                        f"evaluator-private identity entered model-visible key at {path}.{raw_key}"
                    )
                walk(nested, path=f"{path}.{raw_key}")
        elif type(child) is list:
            for position, nested in enumerate(child):
                walk(nested, path=f"{path}[{position}]")
        elif type(child) is str and any(marker in child for marker in private_strings):
            raise PlannedTrainingSurfaceBridgeV1Error(
                f"evaluator-private identity entered model-visible data at {path}"
            )
        elif type(child) not in (str, int, bool) and child is not None:
            raise PlannedTrainingSurfaceBridgeV1Error(f"non-JSON value entered model-visible data at {path}")

    walk(obj, path="model_visible_static_data")
    return obj


@dataclass(frozen=True, slots=True)
class PlannedTrainingSurfaceRotationV1:
    """Evaluator-private identity of one planned Official rotation."""

    rotation_position: int
    official_rule_id: str
    official_truth_digest: str
    objective_weight_numerator: int
    objective_weight_denominator: int
    static_model_input_digest: str
    update_batch_id: str
    planned_optimizer_step_before: int
    planned_optimizer_step_after_objective_collection: int

    def __post_init__(self) -> None:
        _require_integer(self.rotation_position, name="rotation_position")
        _require_identifier(self.official_rule_id, name="official_rule_id")
        _require_sha256(self.official_truth_digest, name="official_truth_digest")
        _require_integer(
            self.objective_weight_numerator,
            name="objective weight numerator",
            minimum=1,
        )
        _require_integer(
            self.objective_weight_denominator,
            name="objective weight denominator",
            minimum=1,
        )
        _require_sha256(
            self.static_model_input_digest,
            name="static model-input digest",
        )
        _require_identifier(self.update_batch_id, name="update_batch_id")
        _require_integer(
            self.planned_optimizer_step_before,
            name="planned optimizer step before",
        )
        _require_integer(
            self.planned_optimizer_step_after_objective_collection,
            name="planned optimizer step after objective collection",
        )

    def as_obj(self) -> dict[str, Any]:
        return {
            "rotation_position": self.rotation_position,
            "official_rule_id": self.official_rule_id,
            "official_truth_digest": self.official_truth_digest,
            "exact_objective_weight": {
                "numerator": self.objective_weight_numerator,
                "denominator": self.objective_weight_denominator,
            },
            "static_model_input_digest": self.static_model_input_digest,
            "update_batch_id": self.update_batch_id,
            "planned_optimizer_step_before": self.planned_optimizer_step_before,
            "planned_optimizer_step_after_objective_collection": (
                self.planned_optimizer_step_after_objective_collection
            ),
        }


@dataclass(frozen=True, slots=True)
class PlannedTrainingSurfaceBlockV1:
    """One exact planned block split into visible and private subtrees."""

    bank_position: int
    training_block_plan_digest: str
    opening_content_digest: str
    opening_scene_set_digest: str
    static_model_input_json: str
    static_model_input_digest: str
    block_id: str
    opening_id: str
    live_rule_bindings: tuple[tuple[str, str], ...]
    terminal_law_digest: str
    materialized_training_panel_digest: str
    hidden_order_display_binding_digest: str
    update_batch_id: str
    rotations: tuple[PlannedTrainingSurfaceRotationV1, ...]

    def __post_init__(self) -> None:
        _require_integer(self.bank_position, name="bank_position")
        for name in (
            "training_block_plan_digest",
            "opening_content_digest",
            "opening_scene_set_digest",
            "static_model_input_digest",
            "terminal_law_digest",
            "materialized_training_panel_digest",
            "hidden_order_display_binding_digest",
        ):
            _require_sha256(getattr(self, name), name=name)
        _require_identifier(self.block_id, name="block_id")
        _require_identifier(self.opening_id, name="opening_id")
        _require_identifier(self.update_batch_id, name="update_batch_id")
        if type(self.live_rule_bindings) is not tuple or not self.live_rule_bindings:
            raise PlannedTrainingSurfaceBridgeV1Error("block requires nonempty live-rule bindings")
        if any(
            type(row) is not tuple or len(row) != 2 or type(row[0]) is not str or not _is_sha256(row[1])
            for row in self.live_rule_bindings
        ):
            raise PlannedTrainingSurfaceBridgeV1Error("block contains a malformed live-rule binding")
        if tuple(rule_id for rule_id, _ in self.live_rule_bindings) != tuple(
            sorted({rule_id for rule_id, _ in self.live_rule_bindings})
        ):
            raise PlannedTrainingSurfaceBridgeV1Error("live-rule bindings must be canonical and unique")
        for rule_id, truth_digest in self.live_rule_bindings:
            _require_catalog_rule_binding(rule_id, truth_digest)
        if type(self.rotations) is not tuple or len(self.rotations) != len(self.live_rule_bindings):
            raise PlannedTrainingSurfaceBridgeV1Error(
                "block requires exactly one private rotation per live rule"
            )
        if any(type(row) is not PlannedTrainingSurfaceRotationV1 for row in self.rotations):
            raise PlannedTrainingSurfaceBridgeV1Error("block contains a foreign private rotation")
        if tuple(row.rotation_position for row in self.rotations) != tuple(range(len(self.rotations))):
            raise PlannedTrainingSurfaceBridgeV1Error(
                "private rotations must occupy every canonical position"
            )
        if {row.official_rule_id for row in self.rotations} != {
            rule_id for rule_id, _ in self.live_rule_bindings
        }:
            raise PlannedTrainingSurfaceBridgeV1Error(
                "private rotations differ from the complete live-rule set"
            )
        truth_by_rule = dict(self.live_rule_bindings)
        if any(row.official_truth_digest != truth_by_rule[row.official_rule_id] for row in self.rotations):
            raise PlannedTrainingSurfaceBridgeV1Error(
                "private rotation truth identities differ from live-rule bindings"
            )
        if any(
            row.objective_weight_numerator != 1
            or row.objective_weight_denominator != len(self.live_rule_bindings)
            for row in self.rotations
        ):
            raise PlannedTrainingSurfaceBridgeV1Error(
                "private rotations must use exact equal 1/n0 objective weights"
            )
        if {row.static_model_input_digest for row in self.rotations} != {self.static_model_input_digest}:
            raise PlannedTrainingSurfaceBridgeV1Error(
                "planned rotations do not share one exact static-model-input digest"
            )
        if {row.update_batch_id for row in self.rotations} != {self.update_batch_id}:
            raise PlannedTrainingSurfaceBridgeV1Error(
                "private rotations differ from the planned update-batch identity"
            )
        if any(
            row.planned_optimizer_step_after_objective_collection != row.planned_optimizer_step_before
            for row in self.rotations
        ):
            raise PlannedTrainingSurfaceBridgeV1Error(
                "private rotations advance the optimizer during objective collection"
            )
        if len({row.planned_optimizer_step_before for row in self.rotations}) != 1:
            raise PlannedTrainingSurfaceBridgeV1Error(
                "private rotations do not share one planned pre-update optimizer step"
            )
        static_obj = _load_json(self.static_model_input_json)
        if _dump_json(static_obj) != self.static_model_input_json:
            raise PlannedTrainingSurfaceBridgeV1Error(
                "stored static model input is not canonical compact JSON"
            )
        if _json_digest(static_obj, domain=_PLAN_STATIC_INPUT_DOMAIN) != self.static_model_input_digest:
            raise PlannedTrainingSurfaceBridgeV1Error(
                "stored static model-input semantic digest is inconsistent"
            )
        visible = validate_planned_model_visible_static_data_v1(
            static_obj,
            evaluator_private_strings=(
                self.training_block_plan_digest,
                self.block_id,
                self.opening_id,
                self.opening_content_digest,
                self.opening_scene_set_digest,
                self.materialized_training_panel_digest,
                self.hidden_order_display_binding_digest,
                self.update_batch_id,
                *(item for binding in self.live_rule_bindings for item in binding),
            ),
        )
        raw_law = visible["public_train_terminal_scene_law"]
        if (
            type(raw_law) is not dict
            or raw_law.get("unconditional_scene_law_digest") != self.terminal_law_digest
        ):
            raise PlannedTrainingSurfaceBridgeV1Error(
                "public terminal-law digest differs from the complete static payload"
            )

    @property
    def static_model_input_obj(self) -> Mapping[str, Any]:
        return cast(Mapping[str, Any], _load_json(self.static_model_input_json))

    @property
    def static_model_input_canonical_sha256(self) -> str:
        return _sha256_bytes(self.static_model_input_json.encode("ascii"))

    @property
    def evaluator_private_digest(self) -> str:
        return _json_digest(self._evaluator_private_obj(), domain=_PRIVATE_DOMAIN)

    def _evaluator_private_obj(self) -> dict[str, Any]:
        return {
            "block_id": self.block_id,
            "opening_id": self.opening_id,
            "live_rule_bindings": [
                {"rule_id": rule_id, "truth_digest": truth_digest}
                for rule_id, truth_digest in self.live_rule_bindings
            ],
            "materialized_training_panel_digest": self.materialized_training_panel_digest,
            "hidden_order_display_binding_digest": (self.hidden_order_display_binding_digest),
            "update_batch_id": self.update_batch_id,
            "rotations": [row.as_obj() for row in self.rotations],
        }

    def _unsigned_obj(self) -> dict[str, Any]:
        return {
            "bank_position": self.bank_position,
            "training_block_plan_digest": self.training_block_plan_digest,
            "semantic_opening_binding": {
                "opening_content_digest": self.opening_content_digest,
                "opening_scene_set_digest": self.opening_scene_set_digest,
            },
            "model_visible_static_data": self.static_model_input_obj,
            "model_visible_static_data_digest": self.static_model_input_digest,
            "public_terminal_law_digest": self.terminal_law_digest,
            "model_visible_static_data_canonical_sha256": (self.static_model_input_canonical_sha256),
            "model_visible_static_data_byte_count": len(self.static_model_input_json.encode("ascii")),
            "evaluator_private": self._evaluator_private_obj(),
            "evaluator_private_digest": self.evaluator_private_digest,
            "surface_separation": {
                "model_visible_static_data_internally_validated": True,
                "all_rotations_share_identical_static_data": True,
                "evaluator_private_bindings_absent_from_static_data": True,
                "exact_plan_backed_rederivation_required": True,
                "standalone_block_plan_binding_verified": False,
                "runtime_prompt_or_tokens_claimed": False,
            },
        }

    @property
    def digest(self) -> str:
        return _json_digest(self._unsigned_obj(), domain=_BLOCK_DOMAIN)

    def as_obj(self) -> dict[str, Any]:
        return {**self._unsigned_obj(), "planned_training_surface_block_digest": self.digest}


def _rotation_rows(
    block: HypothesisCompleteTrainingBlockPlanV1,
) -> tuple[PlannedTrainingSurfaceRotationV1, ...]:
    catalog = build_rule_catalog()
    return tuple(
        PlannedTrainingSurfaceRotationV1(
            rotation_position=position,
            official_rule_id=rule_id,
            official_truth_digest=catalog[int(rule_id[4:])].truth_digest,
            objective_weight_numerator=1,
            objective_weight_denominator=block.n0,
            static_model_input_digest=block.static_model_input_digest,
            update_batch_id=block.update_batch_id,
            planned_optimizer_step_before=block.planned_optimizer_step_before,
            planned_optimizer_step_after_objective_collection=(block.planned_optimizer_step_before),
        )
        for position, rule_id in enumerate(
            block.hidden_order_display_binding.official_rotation_order_rule_ids
        )
    )


def _surface_block(
    block: HypothesisCompleteTrainingBlockPlanV1,
) -> PlannedTrainingSurfaceBlockV1:
    static_obj = block.static_model_input_obj()
    validate_planned_model_visible_static_data_v1(
        static_obj,
        evaluator_private_strings=_private_string_leaves(block),
    )
    return PlannedTrainingSurfaceBlockV1(
        bank_position=block.bank_position,
        training_block_plan_digest=block.digest,
        opening_content_digest=block.opening.content_digest,
        opening_scene_set_digest=block.opening.scene_set_digest,
        static_model_input_json=_dump_json(static_obj),
        static_model_input_digest=block.static_model_input_digest,
        block_id=block.block_id,
        opening_id=block.opening.opening_id,
        live_rule_bindings=tuple(
            zip(
                block.opening.version_space_rule_ids,
                block.opening.version_space_truth_digests,
                strict=True,
            )
        ),
        terminal_law_digest=block.terminal_law.digest,
        materialized_training_panel_digest=block.materialized_training_panel.digest,
        hidden_order_display_binding_digest=block.hidden_order_display_binding.digest,
        update_batch_id=block.update_batch_id,
        rotations=_rotation_rows(block),
    )


@dataclass(frozen=True, slots=True)
class PlannedTrainingStaticSurfaceBridgeV1:
    """Complete static-surface view of one prospective training plan.

    Direct construction proves internal consistency only. Exact plan-byte
    provenance is established by the plan-backed builder, verifier, serializer,
    or parser; a bare ``as_obj()`` call is not evidence of that replay.
    """

    plan_bytes_sha256: str
    plan_byte_count: int
    plan_digest: str
    source_tree_sha256: str
    bank_generator_source_sha256: str
    initial_checkpoint_manifest_sha256: str
    tokenizer_binding_digest: str
    optimizer_config_digest: str
    catalog_digest: str
    supported_catalog_digest: str
    renderer_registry_digest: str
    fixed_n0: int
    registered_episode_budget: int
    engineering_budget_override: bool
    blocks: tuple[PlannedTrainingSurfaceBlockV1, ...]

    def __post_init__(self) -> None:
        for name in (
            "plan_bytes_sha256",
            "plan_digest",
            "source_tree_sha256",
            "bank_generator_source_sha256",
            "initial_checkpoint_manifest_sha256",
            "tokenizer_binding_digest",
            "optimizer_config_digest",
            "catalog_digest",
            "supported_catalog_digest",
            "renderer_registry_digest",
        ):
            _require_sha256(getattr(self, name), name=name)
        _require_integer(self.plan_byte_count, name="plan byte count", minimum=1)
        _require_integer(self.fixed_n0, name="fixed_n0", minimum=1)
        _require_integer(
            self.registered_episode_budget,
            name="registered episode budget",
            minimum=1,
        )
        if type(self.engineering_budget_override) is not bool:
            raise PlannedTrainingSurfaceBridgeV1Error("engineering_budget_override must be Boolean")
        if type(self.blocks) is not tuple or not self.blocks:
            raise PlannedTrainingSurfaceBridgeV1Error("static-surface bridge requires a nonempty block tuple")
        if any(type(block) is not PlannedTrainingSurfaceBlockV1 for block in self.blocks):
            raise PlannedTrainingSurfaceBridgeV1Error("static-surface bridge contains a foreign block")
        if tuple(block.bank_position for block in self.blocks) != tuple(range(len(self.blocks))):
            raise PlannedTrainingSurfaceBridgeV1Error(
                "static-surface blocks must occupy every bank position in order"
            )
        if {len(block.rotations) for block in self.blocks} != {self.fixed_n0}:
            raise PlannedTrainingSurfaceBridgeV1Error("static-surface block sizes differ from fixed_n0")
        if self.episode_count != self.registered_episode_budget:
            raise PlannedTrainingSurfaceBridgeV1Error(
                "static-surface episode count differs from the registered budget"
            )

    @property
    def episode_count(self) -> int:
        return sum(len(block.rotations) for block in self.blocks)

    @property
    def static_surface_population_digest(self) -> str:
        return _json_digest(
            {
                "block_surface_bindings": [
                    {
                        "bank_position": block.bank_position,
                        "opening_content_digest": block.opening_content_digest,
                        "model_visible_static_data_digest": block.static_model_input_digest,
                        "model_visible_static_data_canonical_sha256": (
                            block.static_model_input_canonical_sha256
                        ),
                    }
                    for block in self.blocks
                ]
            },
            domain=_POPULATION_DOMAIN,
        )

    def _unsigned_obj(self) -> dict[str, Any]:
        return {
            "schema_version": PLANNED_TRAINING_SURFACE_BRIDGE_SCHEMA_VERSION,
            "record_kind": _BRIDGE_KIND,
            "authorization": dict(_AUTHORIZATION),
            "exact_training_plan_binding": {
                "canonical_plan_bytes_sha256": self.plan_bytes_sha256,
                "canonical_plan_byte_count": self.plan_byte_count,
                "hypothesis_complete_training_plan_digest": self.plan_digest,
                "pre_run_bindings": {
                    "source_tree_sha256": self.source_tree_sha256,
                    "bank_generator_source_sha256": self.bank_generator_source_sha256,
                    "initial_checkpoint_manifest_sha256": (self.initial_checkpoint_manifest_sha256),
                    "tokenizer_binding_digest": self.tokenizer_binding_digest,
                    "optimizer_config_digest": self.optimizer_config_digest,
                },
                "catalog_digest": self.catalog_digest,
                "supported_catalog_digest": self.supported_catalog_digest,
                "renderer_registry_digest": self.renderer_registry_digest,
                "fixed_n0": self.fixed_n0,
                "registered_episode_budget": self.registered_episode_budget,
                "engineering_budget_override": self.engineering_budget_override,
                "block_count": len(self.blocks),
                "episode_count": self.episode_count,
            },
            "blocks": [block.as_obj() for block in self.blocks],
            "static_surface_population_digest": self.static_surface_population_digest,
            "unresolved_runtime_boundary": dict(_BOUNDARY),
        }

    @property
    def digest(self) -> str:
        return _json_digest(self._unsigned_obj(), domain=_BRIDGE_DOMAIN)

    def as_obj(self) -> dict[str, Any]:
        return {**self._unsigned_obj(), "planned_training_surface_bridge_digest": self.digest}


def build_planned_training_surface_bridge_v1(
    plan_text: str,
    *,
    expected_plan_digest: str,
) -> PlannedTrainingStaticSurfaceBridgeV1:
    """Build the exact nonauthorizing bridge from canonical plan text."""

    plan, plan_bytes = _canonical_plan(
        plan_text,
        expected_plan_digest=expected_plan_digest,
    )
    return PlannedTrainingStaticSurfaceBridgeV1(
        plan_bytes_sha256=_sha256_bytes(plan_bytes),
        plan_byte_count=len(plan_bytes),
        plan_digest=plan.digest,
        source_tree_sha256=plan.bindings.source_tree_sha256,
        bank_generator_source_sha256=plan.bindings.bank_generator_source_sha256,
        initial_checkpoint_manifest_sha256=(plan.bindings.initial_checkpoint_manifest_sha256),
        tokenizer_binding_digest=plan.bindings.tokenizer_binding_digest,
        optimizer_config_digest=plan.bindings.optimizer_config_digest,
        catalog_digest=plan.blocks[0].opening.catalog_digest,
        supported_catalog_digest=plan.blocks[0].opening.supported_catalog_digest,
        renderer_registry_digest=cast(
            str,
            plan.as_obj()["renderer_registry_digest"],
        ),
        fixed_n0=plan.n0,
        registered_episode_budget=plan.registered_episode_budget,
        engineering_budget_override=plan.engineering_budget_override,
        blocks=tuple(_surface_block(block) for block in plan.blocks),
    )


def planned_training_surface_bridge_v1_from_obj(
    value: object,
    *,
    plan_text: str,
    expected_plan_digest: str,
    expected_bridge_digest: str,
) -> PlannedTrainingStaticSurfaceBridgeV1:
    """Replay a bridge object from the independently supplied exact plan."""

    obj = _require_mapping(
        value,
        (
            "schema_version",
            "record_kind",
            "authorization",
            "exact_training_plan_binding",
            "blocks",
            "static_surface_population_digest",
            "unresolved_runtime_boundary",
            "planned_training_surface_bridge_digest",
        ),
        name="planned training static-surface bridge",
    )
    if (
        type(obj["schema_version"]) is not int
        or obj["schema_version"] != PLANNED_TRAINING_SURFACE_BRIDGE_SCHEMA_VERSION
        or obj["record_kind"] != _BRIDGE_KIND
    ):
        raise PlannedTrainingSurfaceBridgeV1Error(
            "planned training static-surface bridge has the wrong schema identity"
        )
    _require_constant_mapping(obj["authorization"], _AUTHORIZATION, name="authorization")
    _require_constant_mapping(
        obj["unresolved_runtime_boundary"],
        _BOUNDARY,
        name="unresolved runtime boundary",
    )
    _require_sha256(expected_bridge_digest, name="expected bridge digest")
    expected = build_planned_training_surface_bridge_v1(
        plan_text,
        expected_plan_digest=expected_plan_digest,
    )
    if expected.digest != expected_bridge_digest:
        raise PlannedTrainingSurfaceBridgeV1Error(
            "rederived bridge differs from the externally expected digest"
        )
    if obj["planned_training_surface_bridge_digest"] != expected.digest:
        raise PlannedTrainingSurfaceBridgeV1Error("serialized bridge digest differs")
    if _dump_json(obj) != _dump_json(expected.as_obj()):
        raise PlannedTrainingSurfaceBridgeV1Error(
            "bridge differs from exact training-plan static-surface rederivation"
        )
    return expected


def verify_planned_training_surface_bridge_v1(
    bridge: PlannedTrainingStaticSurfaceBridgeV1,
    *,
    plan_text: str,
    expected_plan_digest: str,
    expected_bridge_digest: str,
) -> PlannedTrainingStaticSurfaceBridgeV1:
    """Verify an in-memory bridge against exact canonical plan text."""

    if type(bridge) is not PlannedTrainingStaticSurfaceBridgeV1:
        raise TypeError("verify requires PlannedTrainingStaticSurfaceBridgeV1")
    expected = build_planned_training_surface_bridge_v1(
        plan_text,
        expected_plan_digest=expected_plan_digest,
    )
    if expected.digest != _require_sha256(
        expected_bridge_digest,
        name="expected bridge digest",
    ):
        raise PlannedTrainingSurfaceBridgeV1Error(
            "rederived bridge differs from the externally expected digest"
        )
    if bridge != expected or _dump_json(bridge.as_obj()) != _dump_json(expected.as_obj()):
        raise PlannedTrainingSurfaceBridgeV1Error("in-memory bridge differs from exact plan rederivation")
    return expected


def serialize_planned_training_surface_bridge_v1(
    bridge: PlannedTrainingStaticSurfaceBridgeV1,
    *,
    plan_text: str,
    expected_plan_digest: str,
    expected_bridge_digest: str,
) -> str:
    """Serialize only after exact plan-backed verification."""

    verified = verify_planned_training_surface_bridge_v1(
        bridge,
        plan_text=plan_text,
        expected_plan_digest=expected_plan_digest,
        expected_bridge_digest=expected_bridge_digest,
    )
    return _dump_json(verified.as_obj())


def parse_planned_training_surface_bridge_v1(
    text: str,
    *,
    plan_text: str,
    expected_plan_digest: str,
    expected_bridge_digest: str,
) -> PlannedTrainingStaticSurfaceBridgeV1:
    """Parse canonical bridge JSON with external plan and bridge expectations."""

    bridge = planned_training_surface_bridge_v1_from_obj(
        _load_json(text),
        plan_text=plan_text,
        expected_plan_digest=expected_plan_digest,
        expected_bridge_digest=expected_bridge_digest,
    )
    canonical = _dump_json(bridge.as_obj())
    if canonical != text:
        raise PlannedTrainingSurfaceBridgeV1Error("bridge JSON is valid but not canonical compact JSON")
    return bridge
