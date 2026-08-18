"""Prospective G03-v2 training freeze and bound execution receipts.

The earlier hypothesis-complete manifest combines static bank data with
assertions about a completed optimizer transition.  In particular, it requires
the post-update checkpoint digest before its bytes can be serialized.  This
additive module separates those two time domains:

* :class:`HypothesisCompleteTrainingPlanV1` contains only facts that can be
  frozen before model execution; and
* :class:`TrainingExecutionReceiptV1` records a completed execution and can be
  verified only against the exact plan and exact checkpoint-manifest bytes.

Both artifacts are deliberately nonauthorizing.  This module does not generate
a production bank, execute a model, or authorize a weight update.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from fractions import Fraction
from typing import Any, cast

from goalzendo_interactive.catalog import CatalogEntry, build_rule_catalog
from goalzendo_interactive.rendering import (
    TRAIN_RENDERERS,
    RendererName,
    render_scene,
    renderer_digest,
)
from goalzendo_interactive.schema import scene_at

from .hypothesis_complete import (
    SCIENTIFIC_TRAINING_EPISODE_BUDGET,
    CanonicalSupportedOpeningV2,
    HiddenOrderDisplayBindingV2,
    HypothesisCompleteV2Error,
    MaterializedTrainingPanelV2,
    UnconditionalTerminalSceneLawV2,
    build_hidden_order_display_binding_v2,
    build_materialized_training_panel_v2,
    canonical_supported_opening_v2_from_obj,
    unconditional_terminal_scene_law_v2_from_obj,
)

TRAINING_FREEZE_SCHEMA_VERSION = 1

_BLOCK_PLAN_KIND = "g03-v2-prospective-hypothesis-complete-block-plan-v1"
_TRAINING_PLAN_KIND = "g03-v2-prospective-hypothesis-complete-training-plan-v1"
_ROTATION_RECEIPT_KIND = "g03-v2-observed-hypothesis-rotation-receipt-v1"
_BLOCK_RECEIPT_KIND = "g03-v2-observed-atomic-block-receipt-v1"
_TRAINING_RECEIPT_KIND = "g03-v2-bound-training-execution-receipt-v1"

_BLOCK_PLAN_DOMAIN = "goalzendo-interactive-v2-training-block-freeze-v1"
_TRAINING_PLAN_DOMAIN = "goalzendo-interactive-v2-training-freeze-v1"
_STATIC_INPUT_DOMAIN = "goalzendo-interactive-v2-training-freeze-static-input-v1"
_ROTATION_RECEIPT_DOMAIN = "goalzendo-interactive-v2-training-rotation-receipt-v1"
_BLOCK_RECEIPT_DOMAIN = "goalzendo-interactive-v2-training-block-receipt-v1"
_TRAINING_RECEIPT_DOMAIN = "goalzendo-interactive-v2-training-execution-receipt-v1"

_IDENTIFIER = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,127}")

_PLAN_AUTHORIZATION: dict[str, bool | str] = {
    "scope": "prospective-static-training-freeze-engineering-only",
    "production_bank_authorized": False,
    "model_execution_authorized": False,
    "weight_updates_authorized": False,
    "launch_authorized": False,
}

_RECEIPT_AUTHORIZATION: dict[str, bool | str] = {
    "scope": "post-run-structural-receipt-engineering-only",
    "production_bank_authorized": False,
    "model_execution_authorized": False,
    "weight_updates_authorized": False,
    "launch_authorized": False,
}

# These fields either cannot be known before execution or assert that execution
# has already happened.  A nested occurrence anywhere in a plan fails before
# ordinary schema parsing, so an older execution-manifest cannot masquerade as
# a prospective plan even if its outer keys are changed.
_FORBIDDEN_PLAN_FIELDS = frozenset(
    {
        "pre_update_checkpoint_digest",
        "post_update_checkpoint_digest",
        "pre_checkpoint_manifest_sha256",
        "post_checkpoint_manifest_sha256",
        "optimizer_step_after_objective_collection",
        "context_reset_before_episode",
        "cache_reset_before_episode",
        "objective_collected_before_block_commit",
        "parameter_update_committed_before_block_commit",
        "commit_disposition",
        "parameter_updates_before_commit",
        "atomic_commit_count",
        "all_objectives_collected_before_commit",
        "rotation_execution_digest",
        "atomic_commit_digest",
        "objective_evidence_digest",
        "terminal_score",
        "reward",
        "loss",
        "model_output",
        "training_execution_receipt_digest",
    }
)


class TrainingFreezeV1Error(ValueError):
    """Raised when a prospective plan or bound receipt fails exact replay."""


def _dump_json(value: Any) -> str:
    try:
        return json.dumps(value, ensure_ascii=True, allow_nan=False, separators=(",", ":"))
    except (TypeError, ValueError) as exc:
        raise TrainingFreezeV1Error(f"value is not canonical JSON: {exc}") from exc


def _load_json(text: str) -> Any:
    if type(text) is not str or not text:
        raise TrainingFreezeV1Error("JSON input must be nonempty text")

    def no_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise TrainingFreezeV1Error(f"duplicate JSON object key: {key!r}")
            result[key] = value
        return result

    def reject_constant(value: str) -> None:
        raise TrainingFreezeV1Error(f"non-finite JSON constant is forbidden: {value}")

    try:
        return json.loads(text, object_pairs_hook=no_duplicates, parse_constant=reject_constant)
    except TrainingFreezeV1Error:
        raise
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise TrainingFreezeV1Error(f"invalid JSON: {exc}") from exc


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
        raise TrainingFreezeV1Error(f"{name} must be a lowercase SHA-256")
    return cast(str, value)


def _require_identifier(value: object, *, name: str) -> str:
    if type(value) is not str or _IDENTIFIER.fullmatch(value) is None:
        raise TrainingFreezeV1Error(f"{name} is not a canonical identifier")
    return value


def _require_integer(
    value: object,
    *,
    name: str,
    minimum: int = 0,
    maximum: int | None = None,
) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise TrainingFreezeV1Error(f"{name} must be an integer >= {minimum}")
    if maximum is not None and value > maximum:
        raise TrainingFreezeV1Error(f"{name} must be an integer <= {maximum}")
    return value


def _require_boolean(value: object, *, name: str) -> bool:
    if type(value) is not bool:
        raise TrainingFreezeV1Error(f"{name} must be a Boolean")
    return value


def _require_mapping(
    value: object,
    fields: tuple[str, ...],
    *,
    name: str,
) -> Mapping[str, Any]:
    if type(value) is not dict or tuple(value) != fields:
        raise TrainingFreezeV1Error(f"{name} has noncanonical, missing, extra, or reordered fields")
    return cast(Mapping[str, Any], value)


def _require_authorization(value: object, expected: Mapping[str, bool | str]) -> None:
    obj = _require_mapping(value, tuple(expected), name="authorization")
    if dict(obj) != dict(expected):
        raise TrainingFreezeV1Error("authorization must remain exactly false and nonauthorizing")


def _reject_execution_fields_from_plan(value: object, *, path: str = "plan") -> None:
    if type(value) is dict:
        for key, child in cast(Mapping[str, Any], value).items():
            normalized = key.lower().replace("-", "_")
            if (
                normalized in _FORBIDDEN_PLAN_FIELDS
                or normalized.startswith("observed_")
                or normalized.endswith("_outcome")
            ):
                raise TrainingFreezeV1Error(
                    f"prospective plan contains forbidden outcome/runtime field at {path}.{key}"
                )
            _reject_execution_fields_from_plan(child, path=f"{path}.{key}")
    elif type(value) is list:
        for position, child in enumerate(value):
            _reject_execution_fields_from_plan(child, path=f"{path}[{position}]")


def _catalog_entry(rule_id: object) -> CatalogEntry:
    if type(rule_id) is not str:
        raise TrainingFreezeV1Error("public rule identity must be a string")
    if not rule_id.startswith("g03r") or len(rule_id) != 9 or not rule_id[4:].isdigit():
        raise TrainingFreezeV1Error(f"malformed public rule identity: {rule_id!r}")
    index = int(rule_id[4:])
    catalog = build_rule_catalog()
    if not 0 <= index < len(catalog) or catalog[index].rule_id != rule_id:
        raise TrainingFreezeV1Error(f"unknown public rule identity: {rule_id!r}")
    return catalog[index]


def _parse_opening(value: object) -> CanonicalSupportedOpeningV2:
    try:
        return canonical_supported_opening_v2_from_obj(value)
    except (HypothesisCompleteV2Error, TypeError, ValueError) as exc:
        raise TrainingFreezeV1Error(f"training opening failed exact replay: {exc}") from exc


def _parse_terminal_law(value: object) -> UnconditionalTerminalSceneLawV2:
    try:
        return unconditional_terminal_scene_law_v2_from_obj(value)
    except (HypothesisCompleteV2Error, TypeError, ValueError) as exc:
        raise TrainingFreezeV1Error(f"training terminal law failed exact replay: {exc}") from exc


def _parse_panel(
    value: object,
    opening: CanonicalSupportedOpeningV2,
) -> MaterializedTrainingPanelV2:
    obj = _require_mapping(
        value,
        (
            "schema_version",
            "panel_kind",
            "catalog_digest",
            "supported_catalog_digest",
            "opening_content_digest",
            "version_space_rules",
            "scene_indices",
            "truth_pattern_rows",
            "rule_balance_rows",
            "external_generator_receipt_digest",
            "external_generator_receipt_verified",
            "external_generator_verification_required",
            "panel_binding_visible_to_model",
            "pairwise_rule_separation_required_here",
            "evaluation_challenge_bank_owns_pairwise_separation",
            "materialized_training_panel_digest",
        ),
        name="materialized training panel",
    )
    raw_scenes = obj["scene_indices"]
    if type(raw_scenes) is not list:
        raise TrainingFreezeV1Error("materialized panel scene_indices must be an array")
    try:
        panel = build_materialized_training_panel_v2(
            opening,
            (
                _require_integer(item, name="panel scene index", maximum=13_715)
                for item in raw_scenes
            ),
            external_generator_receipt_digest=_require_sha256(
                obj["external_generator_receipt_digest"],
                name="panel generator receipt digest",
            ),
        )
    except (HypothesisCompleteV2Error, TypeError, ValueError) as exc:
        raise TrainingFreezeV1Error(f"materialized training panel failed exact replay: {exc}") from exc
    if _dump_json(obj) != _dump_json(panel.as_obj()):
        raise TrainingFreezeV1Error("materialized training panel contains tampered metadata")
    return panel


def _parse_hidden_binding(
    value: object,
    opening: CanonicalSupportedOpeningV2,
) -> HiddenOrderDisplayBindingV2:
    obj = _require_mapping(
        value,
        (
            "binding_kind",
            "opening_digest",
            "official_rotation_order_rule_ids",
            "opening_display_scene_indices",
            "independence_precommitment_digest",
            "binding_metadata_visible_to_model",
            "independence_externally_verified",
            "hidden_order_display_binding_digest",
        ),
        name="hidden order/display binding",
    )
    try:
        binding = build_hidden_order_display_binding_v2(
            opening,
            independence_precommitment_digest=_require_sha256(
                obj["independence_precommitment_digest"],
                name="independence precommitment digest",
            ),
        )
    except (HypothesisCompleteV2Error, TypeError, ValueError) as exc:
        raise TrainingFreezeV1Error(f"hidden order/display binding failed exact replay: {exc}") from exc
    if _dump_json(obj) != _dump_json(binding.as_obj()):
        raise TrainingFreezeV1Error("hidden order/display binding contains tampered metadata")
    return binding


def _terminal_static_checks(
    opening: CanonicalSupportedOpeningV2,
    law: UnconditionalTerminalSceneLawV2,
    panel: MaterializedTrainingPanelV2,
) -> None:
    if (
        law.catalog_digest != opening.catalog_digest
        or law.supported_catalog_digest != opening.supported_catalog_digest
    ):
        raise TrainingFreezeV1Error("opening and terminal law use different catalog contracts")
    if (
        panel.opening_content_digest != opening.content_digest
        or panel.version_space_rule_ids != opening.version_space_rule_ids
        or panel.version_space_truth_digests != opening.version_space_truth_digests
    ):
        raise TrainingFreezeV1Error("private panel is not bound to the exact opening and V0")
    opening_scenes = {item.scene_index for item in opening.observations}
    law_scenes = {item.scene_index for item in law.scene_probabilities}
    panel_scenes = set(panel.scene_indices)
    if opening_scenes & law_scenes or opening_scenes & panel_scenes:
        raise TrainingFreezeV1Error("opening, terminal-law support, and panel must be disjoint")
    if not panel_scenes.issubset(law_scenes):
        raise TrainingFreezeV1Error("private panel must be a subset of the public terminal-law support")

    entries = tuple(_catalog_entry(rule_id) for rule_id in opening.version_space_rule_ids)
    pattern_mass: dict[int, Fraction] = {}
    full_mask = (1 << opening.n0) - 1
    for item in law.scene_probabilities:
        pattern = sum(
            1 << position
            for position, entry in enumerate(entries)
            if entry.truth[item.scene_index]
        )
        if pattern.bit_count() != opening.n0 // 2:
            raise TrainingFreezeV1Error("every public-law scene must split V0 exactly in half")
        pattern_mass[pattern] = pattern_mass.get(pattern, Fraction()) + item.probability
    if any(
        mass != pattern_mass.get(full_mask ^ pattern, Fraction())
        for pattern, mass in pattern_mass.items()
    ):
        raise TrainingFreezeV1Error("public terminal law lacks exact complementary-pattern mass")
    for entry in entries:
        accepted_mass = sum(
            (
                item.probability
                for item in law.scene_probabilities
                if entry.truth[item.scene_index]
            ),
            Fraction(),
        )
        if accepted_mass != Fraction(1, 2):
            raise TrainingFreezeV1Error("every V0 rule must have exact one-half public-law mass")


@dataclass(frozen=True, slots=True)
class TrainingFreezeBindingsV1:
    """Pre-run identities required to interpret one static training plan."""

    source_tree_sha256: str
    bank_generator_source_sha256: str
    initial_checkpoint_manifest_sha256: str
    tokenizer_binding_digest: str
    optimizer_config_digest: str

    def __post_init__(self) -> None:
        for name in (
            "source_tree_sha256",
            "bank_generator_source_sha256",
            "initial_checkpoint_manifest_sha256",
            "tokenizer_binding_digest",
            "optimizer_config_digest",
        ):
            _require_sha256(getattr(self, name), name=name)

    def as_obj(self) -> dict[str, str]:
        return {
            "source_tree_sha256": self.source_tree_sha256,
            "bank_generator_source_sha256": self.bank_generator_source_sha256,
            "initial_checkpoint_manifest_sha256": self.initial_checkpoint_manifest_sha256,
            "tokenizer_binding_digest": self.tokenizer_binding_digest,
            "optimizer_config_digest": self.optimizer_config_digest,
        }


def _bindings_from_obj(value: object) -> TrainingFreezeBindingsV1:
    fields = (
        "source_tree_sha256",
        "bank_generator_source_sha256",
        "initial_checkpoint_manifest_sha256",
        "tokenizer_binding_digest",
        "optimizer_config_digest",
    )
    obj = _require_mapping(value, fields, name="training-freeze bindings")
    return TrainingFreezeBindingsV1(
        *(_require_sha256(obj[field], name=field) for field in fields)
    )


@dataclass(frozen=True, slots=True)
class HypothesisCompleteTrainingBlockPlanV1:
    """One immutable pre-run hypothesis-complete block plan."""

    bank_position: int
    block_id: str
    opening: CanonicalSupportedOpeningV2
    terminal_law: UnconditionalTerminalSceneLawV2
    materialized_training_panel: MaterializedTrainingPanelV2
    renderer_name: str
    hidden_order_display_binding: HiddenOrderDisplayBindingV2
    update_batch_id: str
    planned_optimizer_step_before: int

    def __post_init__(self) -> None:
        _require_integer(self.bank_position, name="bank_position")
        _require_identifier(self.block_id, name="block_id")
        _require_identifier(self.update_batch_id, name="update_batch_id")
        _require_integer(self.planned_optimizer_step_before, name="planned optimizer step")
        if type(self.opening) is not CanonicalSupportedOpeningV2:
            raise TypeError("block plan requires a CanonicalSupportedOpeningV2")
        if type(self.terminal_law) is not UnconditionalTerminalSceneLawV2:
            raise TypeError("block plan requires an UnconditionalTerminalSceneLawV2")
        if type(self.materialized_training_panel) is not MaterializedTrainingPanelV2:
            raise TypeError("block plan requires a MaterializedTrainingPanelV2")
        if type(self.hidden_order_display_binding) is not HiddenOrderDisplayBindingV2:
            raise TypeError("block plan requires a HiddenOrderDisplayBindingV2")
        if type(self.renderer_name) is not str or self.renderer_name not in TRAIN_RENDERERS:
            raise TrainingFreezeV1Error("block plan requires a registered training renderer")
        try:
            expected_hidden = build_hidden_order_display_binding_v2(
                self.opening,
                independence_precommitment_digest=(
                    self.hidden_order_display_binding.independence_precommitment_digest
                ),
            )
        except (HypothesisCompleteV2Error, TypeError, ValueError) as exc:
            raise TrainingFreezeV1Error(f"hidden binding failed block-plan replay: {exc}") from exc
        if self.hidden_order_display_binding != expected_hidden:
            raise TrainingFreezeV1Error("hidden binding differs from its exact opening precommitment")
        _terminal_static_checks(
            self.opening,
            self.terminal_law,
            self.materialized_training_panel,
        )
        _reject_execution_fields_from_plan(self._unsigned_obj())

    @property
    def n0(self) -> int:
        return self.opening.n0

    @property
    def planned_optimizer_step_after(self) -> int:
        return self.planned_optimizer_step_before + 1

    def static_model_input_obj(self) -> dict[str, Any]:
        observations = {item.scene_index: item for item in self.opening.observations}
        renderer = cast(RendererName, self.renderer_name)
        return {
            "opening_examples": [
                {
                    "accepted": observations[scene_index].accepted,
                    "scene_text": render_scene(scene_at(scene_index), renderer),
                }
                for scene_index in self.hidden_order_display_binding.opening_display_scene_indices
            ],
            "renderer_contract": {
                "renderer_name": self.renderer_name,
                "renderer_registry_digest": renderer_digest(),
            },
            "public_train_terminal_scene_law": self.terminal_law.as_obj(),
        }

    @property
    def static_model_input_digest(self) -> str:
        return _json_digest(self.static_model_input_obj(), domain=_STATIC_INPUT_DOMAIN)

    def _rotation_plan(self) -> list[dict[str, Any]]:
        return [
            {
                "rotation_position": position,
                "official_rule_id": rule_id,
                "official_truth_digest": _catalog_entry(rule_id).truth_digest,
                "exact_objective_weight": {"numerator": 1, "denominator": self.n0},
                "static_model_input_digest": self.static_model_input_digest,
                "update_batch_id": self.update_batch_id,
                "planned_optimizer_step_before": self.planned_optimizer_step_before,
                "planned_optimizer_step_after_objective_collection": (
                    self.planned_optimizer_step_before
                ),
            }
            for position, rule_id in enumerate(
                self.hidden_order_display_binding.official_rotation_order_rule_ids
            )
        ]

    def _unsigned_obj(self) -> dict[str, Any]:
        return {
            "schema_version": TRAINING_FREEZE_SCHEMA_VERSION,
            "record_kind": _BLOCK_PLAN_KIND,
            "authorization": dict(_PLAN_AUTHORIZATION),
            "bank_position": self.bank_position,
            "block_id": self.block_id,
            "catalog_digest": self.opening.catalog_digest,
            "supported_catalog_digest": self.opening.supported_catalog_digest,
            "n0": self.n0,
            "opening": self.opening.as_obj(),
            "unconditional_terminal_scene_law": self.terminal_law.as_obj(),
            "materialized_training_panel": self.materialized_training_panel.as_obj(),
            "renderer_binding": {
                "renderer_name": self.renderer_name,
                "renderer_registry_digest": renderer_digest(),
            },
            "hidden_order_display_binding": self.hidden_order_display_binding.as_obj(),
            "model_visible_static_data": self.static_model_input_obj(),
            "model_visible_static_data_digest": self.static_model_input_digest,
            "rotation_plan": self._rotation_plan(),
            "planned_atomic_update": {
                "update_batch_id": self.update_batch_id,
                "planned_optimizer_step_before": self.planned_optimizer_step_before,
                "planned_optimizer_step_after": self.planned_optimizer_step_after,
                "planned_objective_count": self.n0,
                "planned_parameter_updates_before_commit": 0,
                "planned_atomic_commit_count": 1,
            },
            "plan_boundaries": {
                "contains_checkpoint_or_outcome_evidence": False,
                "execution_receipt_required": True,
                "receipt_must_bind_exact_block_plan_digest": True,
            },
        }

    @property
    def digest(self) -> str:
        return _json_digest(self._unsigned_obj(), domain=_BLOCK_PLAN_DOMAIN)

    def as_obj(self) -> dict[str, Any]:
        value = {**self._unsigned_obj(), "training_block_plan_digest": self.digest}
        _reject_execution_fields_from_plan(value)
        return value


def build_hypothesis_complete_training_block_plan_v1(
    opening: CanonicalSupportedOpeningV2,
    terminal_law: UnconditionalTerminalSceneLawV2,
    materialized_training_panel: MaterializedTrainingPanelV2,
    hidden_order_display_binding: HiddenOrderDisplayBindingV2,
    *,
    bank_position: int,
    block_id: str,
    renderer_name: str,
    update_batch_id: str,
    planned_optimizer_step_before: int,
) -> HypothesisCompleteTrainingBlockPlanV1:
    return HypothesisCompleteTrainingBlockPlanV1(
        bank_position,
        block_id,
        opening,
        terminal_law,
        materialized_training_panel,
        renderer_name,
        hidden_order_display_binding,
        update_batch_id,
        planned_optimizer_step_before,
    )


def _block_plan_from_obj(value: object) -> HypothesisCompleteTrainingBlockPlanV1:
    _reject_execution_fields_from_plan(value)
    obj = _require_mapping(
        value,
        (
            "schema_version",
            "record_kind",
            "authorization",
            "bank_position",
            "block_id",
            "catalog_digest",
            "supported_catalog_digest",
            "n0",
            "opening",
            "unconditional_terminal_scene_law",
            "materialized_training_panel",
            "renderer_binding",
            "hidden_order_display_binding",
            "model_visible_static_data",
            "model_visible_static_data_digest",
            "rotation_plan",
            "planned_atomic_update",
            "plan_boundaries",
            "training_block_plan_digest",
        ),
        name="training block plan",
    )
    if (
        type(obj["schema_version"]) is not int
        or obj["schema_version"] != TRAINING_FREEZE_SCHEMA_VERSION
        or obj["record_kind"] != _BLOCK_PLAN_KIND
    ):
        raise TrainingFreezeV1Error("training block plan has the wrong schema identity")
    _require_authorization(obj["authorization"], _PLAN_AUTHORIZATION)
    opening = _parse_opening(obj["opening"])
    law = _parse_terminal_law(obj["unconditional_terminal_scene_law"])
    panel = _parse_panel(obj["materialized_training_panel"], opening)
    hidden = _parse_hidden_binding(obj["hidden_order_display_binding"], opening)
    renderer = _require_mapping(
        obj["renderer_binding"],
        ("renderer_name", "renderer_registry_digest"),
        name="renderer binding",
    )
    if renderer["renderer_registry_digest"] != renderer_digest():
        raise TrainingFreezeV1Error("renderer registry differs from the live registry")
    update = _require_mapping(
        obj["planned_atomic_update"],
        (
            "update_batch_id",
            "planned_optimizer_step_before",
            "planned_optimizer_step_after",
            "planned_objective_count",
            "planned_parameter_updates_before_commit",
            "planned_atomic_commit_count",
        ),
        name="planned atomic update",
    )
    plan = HypothesisCompleteTrainingBlockPlanV1(
        bank_position=_require_integer(obj["bank_position"], name="bank_position"),
        block_id=_require_identifier(obj["block_id"], name="block_id"),
        opening=opening,
        terminal_law=law,
        materialized_training_panel=panel,
        renderer_name=cast(str, renderer["renderer_name"]),
        hidden_order_display_binding=hidden,
        update_batch_id=_require_identifier(update["update_batch_id"], name="update_batch_id"),
        planned_optimizer_step_before=_require_integer(
            update["planned_optimizer_step_before"],
            name="planned optimizer step before",
        ),
    )
    if obj["catalog_digest"] != opening.catalog_digest:
        raise TrainingFreezeV1Error("block plan full-catalog binding differs")
    if obj["supported_catalog_digest"] != opening.supported_catalog_digest:
        raise TrainingFreezeV1Error("block plan supported-catalog binding differs")
    if _require_integer(obj["n0"], name="n0", minimum=1) != plan.n0:
        raise TrainingFreezeV1Error("block plan n0 differs from exact opening replay")
    if obj["training_block_plan_digest"] != plan.digest:
        raise TrainingFreezeV1Error("training block plan digest differs")
    if _dump_json(obj) != _dump_json(plan.as_obj()):
        raise TrainingFreezeV1Error("training block plan contains tampered derived metadata")
    return plan


@dataclass(frozen=True, slots=True)
class HypothesisCompleteTrainingPlanV1:
    """Complete static training plan whose digest exists before model execution."""

    bindings: TrainingFreezeBindingsV1
    blocks: tuple[HypothesisCompleteTrainingBlockPlanV1, ...]
    registered_episode_budget: int = SCIENTIFIC_TRAINING_EPISODE_BUDGET
    engineering_budget_override: bool = False

    def __post_init__(self) -> None:
        if type(self.bindings) is not TrainingFreezeBindingsV1:
            raise TypeError("training plan requires TrainingFreezeBindingsV1")
        if type(self.blocks) is not tuple or not self.blocks:
            raise TrainingFreezeV1Error("training plan requires a nonempty block tuple")
        if any(type(block) is not HypothesisCompleteTrainingBlockPlanV1 for block in self.blocks):
            raise TypeError("training plan contains a foreign block type")
        _require_integer(self.registered_episode_budget, name="registered episode budget", minimum=1)
        _require_boolean(self.engineering_budget_override, name="engineering budget override")
        override_required = self.registered_episode_budget != SCIENTIFIC_TRAINING_EPISODE_BUDGET
        if self.engineering_budget_override is not override_required:
            raise TrainingFreezeV1Error(
                "a non-384 engineering budget requires an explicit override, and 384 forbids one"
            )
        n0s = {block.n0 for block in self.blocks}
        if len(n0s) != 1:
            raise TrainingFreezeV1Error("one training plan must freeze exactly one n0")
        if self.episode_count != self.registered_episode_budget:
            raise TrainingFreezeV1Error("complete-block count differs from the registered episode budget")
        if tuple(block.bank_position for block in self.blocks) != tuple(range(len(self.blocks))):
            raise TrainingFreezeV1Error("training blocks must occupy every bank position in order")
        for name, values in (
            ("block IDs", tuple(block.block_id for block in self.blocks)),
            ("opening contents", tuple(block.opening.content_digest for block in self.blocks)),
            ("update-batch IDs", tuple(block.update_batch_id for block in self.blocks)),
        ):
            if len(values) != len(set(values)):
                raise TrainingFreezeV1Error(f"training plan {name} must be unique")
        for left, right in zip(self.blocks, self.blocks[1:], strict=False):
            if left.planned_optimizer_step_after != right.planned_optimizer_step_before:
                raise TrainingFreezeV1Error("planned optimizer-step chain is discontinuous")
        if len({block.opening.catalog_digest for block in self.blocks}) != 1:
            raise TrainingFreezeV1Error("training plan mixes full-catalog contracts")
        if len({block.opening.supported_catalog_digest for block in self.blocks}) != 1:
            raise TrainingFreezeV1Error("training plan mixes supported-catalog contracts")
        _reject_execution_fields_from_plan(self._unsigned_obj())

    @property
    def n0(self) -> int:
        return self.blocks[0].n0

    @property
    def episode_count(self) -> int:
        return len(self.blocks) * self.n0

    def _unsigned_obj(self) -> dict[str, Any]:
        return {
            "schema_version": TRAINING_FREEZE_SCHEMA_VERSION,
            "record_kind": _TRAINING_PLAN_KIND,
            "authorization": dict(_PLAN_AUTHORIZATION),
            "pre_run_bindings": self.bindings.as_obj(),
            "catalog_digest": self.blocks[0].opening.catalog_digest,
            "supported_catalog_digest": self.blocks[0].opening.supported_catalog_digest,
            "renderer_registry_digest": renderer_digest(),
            "fixed_n0": self.n0,
            "registered_episode_budget": self.registered_episode_budget,
            "engineering_budget_override": self.engineering_budget_override,
            "block_count": len(self.blocks),
            "episode_count": self.episode_count,
            "blocks": [block.as_obj() for block in self.blocks],
            "prospective_boundary": {
                "all_plan_fields_available_before_model_execution": True,
                "contains_checkpoint_or_outcome_evidence": False,
                "post_run_receipt_required": True,
                "plan_digest_independent_of_realized_execution": True,
            },
        }

    @property
    def digest(self) -> str:
        return _json_digest(self._unsigned_obj(), domain=_TRAINING_PLAN_DOMAIN)

    def as_obj(self) -> dict[str, Any]:
        value = {**self._unsigned_obj(), "hypothesis_complete_training_plan_digest": self.digest}
        _reject_execution_fields_from_plan(value)
        return value


def build_hypothesis_complete_training_plan_v1(
    bindings: TrainingFreezeBindingsV1,
    blocks: Iterable[HypothesisCompleteTrainingBlockPlanV1],
    *,
    registered_episode_budget: int = SCIENTIFIC_TRAINING_EPISODE_BUDGET,
    engineering_budget_override: bool = False,
) -> HypothesisCompleteTrainingPlanV1:
    return HypothesisCompleteTrainingPlanV1(
        bindings,
        tuple(blocks),
        registered_episode_budget,
        engineering_budget_override,
    )


def hypothesis_complete_training_plan_v1_from_obj(
    value: object,
    *,
    expected_digest: str | None = None,
) -> HypothesisCompleteTrainingPlanV1:
    _reject_execution_fields_from_plan(value)
    obj = _require_mapping(
        value,
        (
            "schema_version",
            "record_kind",
            "authorization",
            "pre_run_bindings",
            "catalog_digest",
            "supported_catalog_digest",
            "renderer_registry_digest",
            "fixed_n0",
            "registered_episode_budget",
            "engineering_budget_override",
            "block_count",
            "episode_count",
            "blocks",
            "prospective_boundary",
            "hypothesis_complete_training_plan_digest",
        ),
        name="hypothesis-complete training plan",
    )
    if (
        type(obj["schema_version"]) is not int
        or obj["schema_version"] != TRAINING_FREEZE_SCHEMA_VERSION
        or obj["record_kind"] != _TRAINING_PLAN_KIND
    ):
        raise TrainingFreezeV1Error("training plan has the wrong schema identity")
    _require_authorization(obj["authorization"], _PLAN_AUTHORIZATION)
    raw_blocks = obj["blocks"]
    if type(raw_blocks) is not list:
        raise TrainingFreezeV1Error("training plan blocks must be an array")
    plan = HypothesisCompleteTrainingPlanV1(
        bindings=_bindings_from_obj(obj["pre_run_bindings"]),
        blocks=tuple(_block_plan_from_obj(item) for item in raw_blocks),
        registered_episode_budget=_require_integer(
            obj["registered_episode_budget"],
            name="registered episode budget",
            minimum=1,
        ),
        engineering_budget_override=_require_boolean(
            obj["engineering_budget_override"],
            name="engineering budget override",
        ),
    )
    if obj["catalog_digest"] != plan.blocks[0].opening.catalog_digest:
        raise TrainingFreezeV1Error("training plan full-catalog binding differs")
    if obj["supported_catalog_digest"] != plan.blocks[0].opening.supported_catalog_digest:
        raise TrainingFreezeV1Error("training plan supported-catalog binding differs")
    if obj["renderer_registry_digest"] != renderer_digest():
        raise TrainingFreezeV1Error("training plan renderer registry differs")
    if _require_integer(obj["fixed_n0"], name="fixed n0", minimum=1) != plan.n0:
        raise TrainingFreezeV1Error("training plan fixed n0 differs")
    if _require_integer(obj["block_count"], name="block count", minimum=1) != len(plan.blocks):
        raise TrainingFreezeV1Error("training plan block count differs")
    if _require_integer(obj["episode_count"], name="episode count", minimum=1) != plan.episode_count:
        raise TrainingFreezeV1Error("training plan episode count differs")
    if obj["hypothesis_complete_training_plan_digest"] != plan.digest:
        raise TrainingFreezeV1Error("training plan digest differs")
    if expected_digest is not None and plan.digest != _require_sha256(
        expected_digest,
        name="expected training-plan digest",
    ):
        raise TrainingFreezeV1Error("training plan differs from the externally expected digest")
    if _dump_json(obj) != _dump_json(plan.as_obj()):
        raise TrainingFreezeV1Error("training plan contains tampered derived metadata")
    return plan


def verify_hypothesis_complete_training_plan_v1(
    plan: HypothesisCompleteTrainingPlanV1,
) -> HypothesisCompleteTrainingPlanV1:
    if type(plan) is not HypothesisCompleteTrainingPlanV1:
        raise TypeError("verify requires HypothesisCompleteTrainingPlanV1")
    return hypothesis_complete_training_plan_v1_from_obj(
        plan.as_obj(),
        expected_digest=plan.digest,
    )


def serialize_hypothesis_complete_training_plan_v1(
    plan: HypothesisCompleteTrainingPlanV1,
) -> str:
    return _dump_json(verify_hypothesis_complete_training_plan_v1(plan).as_obj())


def parse_hypothesis_complete_training_plan_v1(
    text: str,
    *,
    expected_digest: str | None = None,
) -> HypothesisCompleteTrainingPlanV1:
    plan = hypothesis_complete_training_plan_v1_from_obj(
        _load_json(text),
        expected_digest=expected_digest,
    )
    if serialize_hypothesis_complete_training_plan_v1(plan) != text:
        raise TrainingFreezeV1Error("training-plan JSON is valid but not canonical compact JSON")
    return plan


@dataclass(frozen=True, slots=True)
class RotationExecutionEvidenceV1:
    """Observed evidence for one planned Official rotation."""

    rotation_position: int
    official_rule_id: str
    objective_evidence_digest: str
    pre_checkpoint_manifest_sha256: str
    optimizer_step_before: int
    optimizer_step_after_objective_collection: int
    context_reset_observed: bool
    cache_reset_observed: bool
    objective_collection_observed: bool
    parameter_update_before_commit_observed: bool

    def __post_init__(self) -> None:
        _require_integer(self.rotation_position, name="rotation_position")
        _catalog_entry(self.official_rule_id)
        _require_sha256(self.objective_evidence_digest, name="objective evidence digest")
        _require_sha256(self.pre_checkpoint_manifest_sha256, name="pre-checkpoint manifest digest")
        _require_integer(self.optimizer_step_before, name="optimizer step before")
        if (
            isinstance(self.optimizer_step_after_objective_collection, bool)
            or self.optimizer_step_after_objective_collection != self.optimizer_step_before
        ):
            raise TrainingFreezeV1Error("an optimizer update occurred before the atomic commit")
        for name in (
            "context_reset_observed",
            "cache_reset_observed",
            "objective_collection_observed",
            "parameter_update_before_commit_observed",
        ):
            _require_boolean(getattr(self, name), name=name)
        if not self.context_reset_observed or not self.cache_reset_observed:
            raise TrainingFreezeV1Error("every observed rotation requires context and cache resets")
        if not self.objective_collection_observed:
            raise TrainingFreezeV1Error("every planned rotation objective must be observed")
        if self.parameter_update_before_commit_observed:
            raise TrainingFreezeV1Error("an early parameter update was observed")

    def _unsigned_obj(self) -> dict[str, Any]:
        return {
            "schema_version": TRAINING_FREEZE_SCHEMA_VERSION,
            "record_kind": _ROTATION_RECEIPT_KIND,
            "rotation_position": self.rotation_position,
            "official_rule_id": self.official_rule_id,
            "objective_evidence_digest": self.objective_evidence_digest,
            "pre_checkpoint_manifest_sha256": self.pre_checkpoint_manifest_sha256,
            "optimizer_step_before": self.optimizer_step_before,
            "optimizer_step_after_objective_collection": (
                self.optimizer_step_after_objective_collection
            ),
            "context_reset_observed": self.context_reset_observed,
            "cache_reset_observed": self.cache_reset_observed,
            "objective_collection_observed": self.objective_collection_observed,
            "parameter_update_before_commit_observed": (
                self.parameter_update_before_commit_observed
            ),
        }

    @property
    def digest(self) -> str:
        return _json_digest(self._unsigned_obj(), domain=_ROTATION_RECEIPT_DOMAIN)

    def as_obj(self) -> dict[str, Any]:
        return {**self._unsigned_obj(), "rotation_execution_receipt_digest": self.digest}


def _rotation_receipt_from_obj(value: object) -> RotationExecutionEvidenceV1:
    obj = _require_mapping(
        value,
        (
            "schema_version",
            "record_kind",
            "rotation_position",
            "official_rule_id",
            "objective_evidence_digest",
            "pre_checkpoint_manifest_sha256",
            "optimizer_step_before",
            "optimizer_step_after_objective_collection",
            "context_reset_observed",
            "cache_reset_observed",
            "objective_collection_observed",
            "parameter_update_before_commit_observed",
            "rotation_execution_receipt_digest",
        ),
        name="rotation execution receipt",
    )
    if (
        type(obj["schema_version"]) is not int
        or obj["schema_version"] != TRAINING_FREEZE_SCHEMA_VERSION
        or obj["record_kind"] != _ROTATION_RECEIPT_KIND
    ):
        raise TrainingFreezeV1Error("rotation receipt has the wrong schema identity")
    receipt = RotationExecutionEvidenceV1(
        rotation_position=_require_integer(obj["rotation_position"], name="rotation_position"),
        official_rule_id=_catalog_entry(obj["official_rule_id"]).rule_id,
        objective_evidence_digest=_require_sha256(
            obj["objective_evidence_digest"],
            name="objective evidence digest",
        ),
        pre_checkpoint_manifest_sha256=_require_sha256(
            obj["pre_checkpoint_manifest_sha256"],
            name="pre-checkpoint manifest digest",
        ),
        optimizer_step_before=_require_integer(
            obj["optimizer_step_before"],
            name="optimizer step before",
        ),
        optimizer_step_after_objective_collection=_require_integer(
            obj["optimizer_step_after_objective_collection"],
            name="optimizer step after objective collection",
        ),
        context_reset_observed=_require_boolean(
            obj["context_reset_observed"],
            name="context reset observed",
        ),
        cache_reset_observed=_require_boolean(
            obj["cache_reset_observed"],
            name="cache reset observed",
        ),
        objective_collection_observed=_require_boolean(
            obj["objective_collection_observed"],
            name="objective collection observed",
        ),
        parameter_update_before_commit_observed=_require_boolean(
            obj["parameter_update_before_commit_observed"],
            name="early parameter update observed",
        ),
    )
    if obj["rotation_execution_receipt_digest"] != receipt.digest:
        raise TrainingFreezeV1Error("rotation execution receipt digest differs")
    if _dump_json(obj) != _dump_json(receipt.as_obj()):
        raise TrainingFreezeV1Error("rotation execution receipt contains tampered metadata")
    return receipt


@dataclass(frozen=True, slots=True)
class AtomicBlockExecutionReceiptV1:
    """Observed all-rotations-then-one-commit evidence for one plan block."""

    bank_position: int
    block_plan_digest: str
    update_batch_id: str
    pre_checkpoint_manifest_sha256: str
    post_checkpoint_manifest_sha256: str
    optimizer_step_before: int
    optimizer_step_after: int
    rotations: tuple[RotationExecutionEvidenceV1, ...]
    commit_disposition: str
    parameter_updates_before_commit: int
    atomic_commit_count: int
    all_objectives_collected_before_commit: bool

    def __post_init__(self) -> None:
        _require_integer(self.bank_position, name="bank_position")
        _require_sha256(self.block_plan_digest, name="block-plan digest")
        _require_identifier(self.update_batch_id, name="update_batch_id")
        _require_sha256(self.pre_checkpoint_manifest_sha256, name="pre-checkpoint manifest digest")
        _require_sha256(self.post_checkpoint_manifest_sha256, name="post-checkpoint manifest digest")
        _require_integer(self.optimizer_step_before, name="optimizer step before")
        if (
            isinstance(self.optimizer_step_after, bool)
            or self.optimizer_step_after != self.optimizer_step_before + 1
        ):
            raise TrainingFreezeV1Error("atomic block receipt must advance exactly one optimizer step")
        if type(self.rotations) is not tuple or not self.rotations:
            raise TrainingFreezeV1Error("atomic block receipt requires observed rotations")
        if any(type(item) is not RotationExecutionEvidenceV1 for item in self.rotations):
            raise TypeError("atomic block receipt contains a foreign rotation receipt")
        if tuple(item.rotation_position for item in self.rotations) != tuple(range(len(self.rotations))):
            raise TrainingFreezeV1Error("rotation receipts must occupy every position in order")
        if len({item.objective_evidence_digest for item in self.rotations}) != len(self.rotations):
            raise TrainingFreezeV1Error("objective evidence digests must be unique within a block")
        if any(
            item.pre_checkpoint_manifest_sha256 != self.pre_checkpoint_manifest_sha256
            or item.optimizer_step_before != self.optimizer_step_before
            for item in self.rotations
        ):
            raise TrainingFreezeV1Error(
                "all rotations must share the block's pre-checkpoint and optimizer step"
            )
        if self.commit_disposition not in {
            "committed_state_changed",
            "zero_gradient_no_state_change",
        }:
            raise TrainingFreezeV1Error("atomic block receipt has an unknown commit disposition")
        state_changed = self.pre_checkpoint_manifest_sha256 != self.post_checkpoint_manifest_sha256
        if state_changed is not (self.commit_disposition == "committed_state_changed"):
            raise TrainingFreezeV1Error("commit disposition differs from checkpoint transition")
        if isinstance(self.parameter_updates_before_commit, bool) or self.parameter_updates_before_commit != 0:
            raise TrainingFreezeV1Error("parameter updates occurred before the atomic commit")
        if isinstance(self.atomic_commit_count, bool) or self.atomic_commit_count != 1:
            raise TrainingFreezeV1Error("atomic block receipt requires exactly one commit")
        if (
            type(self.all_objectives_collected_before_commit) is not bool
            or not self.all_objectives_collected_before_commit
        ):
            raise TrainingFreezeV1Error("all objectives must be collected before the atomic commit")

    def _unsigned_obj(self) -> dict[str, Any]:
        return {
            "schema_version": TRAINING_FREEZE_SCHEMA_VERSION,
            "record_kind": _BLOCK_RECEIPT_KIND,
            "bank_position": self.bank_position,
            "block_plan_digest": self.block_plan_digest,
            "update_batch_id": self.update_batch_id,
            "pre_checkpoint_manifest_sha256": self.pre_checkpoint_manifest_sha256,
            "post_checkpoint_manifest_sha256": self.post_checkpoint_manifest_sha256,
            "optimizer_step_before": self.optimizer_step_before,
            "optimizer_step_after": self.optimizer_step_after,
            "rotations": [item.as_obj() for item in self.rotations],
            "commit_disposition": self.commit_disposition,
            "parameter_updates_before_commit": self.parameter_updates_before_commit,
            "atomic_commit_count": self.atomic_commit_count,
            "all_objectives_collected_before_commit": self.all_objectives_collected_before_commit,
        }

    @property
    def digest(self) -> str:
        return _json_digest(self._unsigned_obj(), domain=_BLOCK_RECEIPT_DOMAIN)

    def as_obj(self) -> dict[str, Any]:
        return {**self._unsigned_obj(), "atomic_block_execution_receipt_digest": self.digest}


def build_atomic_block_execution_receipt_v1(
    block_plan: HypothesisCompleteTrainingBlockPlanV1,
    *,
    objective_evidence_digests: tuple[str, ...],
    pre_checkpoint_manifest_bytes: bytes,
    post_checkpoint_manifest_bytes: bytes,
    observed_optimizer_step_before: int,
    observed_optimizer_step_after: int,
    observed_optimizer_steps_after_objective_collection: tuple[int, ...],
    context_resets_observed: tuple[bool, ...],
    cache_resets_observed: tuple[bool, ...],
    objective_collections_observed: tuple[bool, ...],
    parameter_updates_before_commit_observed: tuple[bool, ...],
    observed_parameter_updates_before_commit: int,
    observed_atomic_commit_count: int,
    observed_all_objectives_collected_before_commit: bool,
) -> AtomicBlockExecutionReceiptV1:
    """Construct a receipt only from explicit observed values and exact bytes."""

    if type(block_plan) is not HypothesisCompleteTrainingBlockPlanV1:
        raise TypeError("block receipt requires HypothesisCompleteTrainingBlockPlanV1")
    if type(pre_checkpoint_manifest_bytes) is not bytes or type(post_checkpoint_manifest_bytes) is not bytes:
        raise TypeError("checkpoint manifests must be exact bytes")
    lengths = {
        len(objective_evidence_digests),
        len(observed_optimizer_steps_after_objective_collection),
        len(context_resets_observed),
        len(cache_resets_observed),
        len(objective_collections_observed),
        len(parameter_updates_before_commit_observed),
    }
    if lengths != {block_plan.n0}:
        raise TrainingFreezeV1Error("observed rotation evidence does not cover the complete planned V0")
    pre_digest = _sha256_bytes(pre_checkpoint_manifest_bytes)
    rotations = tuple(
        RotationExecutionEvidenceV1(
            rotation_position=position,
            official_rule_id=rule_id,
            objective_evidence_digest=objective_evidence_digests[position],
            pre_checkpoint_manifest_sha256=pre_digest,
            optimizer_step_before=observed_optimizer_step_before,
            optimizer_step_after_objective_collection=(
                observed_optimizer_steps_after_objective_collection[position]
            ),
            context_reset_observed=context_resets_observed[position],
            cache_reset_observed=cache_resets_observed[position],
            objective_collection_observed=objective_collections_observed[position],
            parameter_update_before_commit_observed=(
                parameter_updates_before_commit_observed[position]
            ),
        )
        for position, rule_id in enumerate(
            block_plan.hidden_order_display_binding.official_rotation_order_rule_ids
        )
    )
    post_digest = _sha256_bytes(post_checkpoint_manifest_bytes)
    return AtomicBlockExecutionReceiptV1(
        bank_position=block_plan.bank_position,
        block_plan_digest=block_plan.digest,
        update_batch_id=block_plan.update_batch_id,
        pre_checkpoint_manifest_sha256=pre_digest,
        post_checkpoint_manifest_sha256=post_digest,
        optimizer_step_before=observed_optimizer_step_before,
        optimizer_step_after=observed_optimizer_step_after,
        rotations=rotations,
        commit_disposition=(
            "committed_state_changed" if pre_digest != post_digest else "zero_gradient_no_state_change"
        ),
        parameter_updates_before_commit=observed_parameter_updates_before_commit,
        atomic_commit_count=observed_atomic_commit_count,
        all_objectives_collected_before_commit=(
            observed_all_objectives_collected_before_commit
        ),
    )


def _block_receipt_from_obj(value: object) -> AtomicBlockExecutionReceiptV1:
    obj = _require_mapping(
        value,
        (
            "schema_version",
            "record_kind",
            "bank_position",
            "block_plan_digest",
            "update_batch_id",
            "pre_checkpoint_manifest_sha256",
            "post_checkpoint_manifest_sha256",
            "optimizer_step_before",
            "optimizer_step_after",
            "rotations",
            "commit_disposition",
            "parameter_updates_before_commit",
            "atomic_commit_count",
            "all_objectives_collected_before_commit",
            "atomic_block_execution_receipt_digest",
        ),
        name="atomic block execution receipt",
    )
    if (
        type(obj["schema_version"]) is not int
        or obj["schema_version"] != TRAINING_FREEZE_SCHEMA_VERSION
        or obj["record_kind"] != _BLOCK_RECEIPT_KIND
    ):
        raise TrainingFreezeV1Error("atomic block receipt has the wrong schema identity")
    raw_rotations = obj["rotations"]
    if type(raw_rotations) is not list:
        raise TrainingFreezeV1Error("atomic block rotations must be an array")
    receipt = AtomicBlockExecutionReceiptV1(
        bank_position=_require_integer(obj["bank_position"], name="bank_position"),
        block_plan_digest=_require_sha256(obj["block_plan_digest"], name="block-plan digest"),
        update_batch_id=_require_identifier(obj["update_batch_id"], name="update_batch_id"),
        pre_checkpoint_manifest_sha256=_require_sha256(
            obj["pre_checkpoint_manifest_sha256"],
            name="pre-checkpoint manifest digest",
        ),
        post_checkpoint_manifest_sha256=_require_sha256(
            obj["post_checkpoint_manifest_sha256"],
            name="post-checkpoint manifest digest",
        ),
        optimizer_step_before=_require_integer(
            obj["optimizer_step_before"],
            name="optimizer step before",
        ),
        optimizer_step_after=_require_integer(
            obj["optimizer_step_after"],
            name="optimizer step after",
            minimum=1,
        ),
        rotations=tuple(_rotation_receipt_from_obj(item) for item in raw_rotations),
        commit_disposition=cast(str, obj["commit_disposition"]),
        parameter_updates_before_commit=_require_integer(
            obj["parameter_updates_before_commit"],
            name="parameter updates before commit",
        ),
        atomic_commit_count=_require_integer(
            obj["atomic_commit_count"],
            name="atomic commit count",
            minimum=1,
        ),
        all_objectives_collected_before_commit=_require_boolean(
            obj["all_objectives_collected_before_commit"],
            name="all objectives collected before commit",
        ),
    )
    if obj["atomic_block_execution_receipt_digest"] != receipt.digest:
        raise TrainingFreezeV1Error("atomic block execution receipt digest differs")
    if _dump_json(obj) != _dump_json(receipt.as_obj()):
        raise TrainingFreezeV1Error("atomic block execution receipt contains tampered metadata")
    return receipt


@dataclass(frozen=True, slots=True)
class TrainingExecutionReceiptV1:
    """Complete post-run receipt, semantically inert without its exact plan."""

    training_plan_digest: str
    blocks: tuple[AtomicBlockExecutionReceiptV1, ...]

    def __post_init__(self) -> None:
        _require_sha256(self.training_plan_digest, name="training-plan digest")
        if type(self.blocks) is not tuple or not self.blocks:
            raise TrainingFreezeV1Error("training execution receipt requires a nonempty block tuple")
        if any(type(item) is not AtomicBlockExecutionReceiptV1 for item in self.blocks):
            raise TypeError("training execution receipt contains a foreign block receipt")
        if tuple(block.bank_position for block in self.blocks) != tuple(range(len(self.blocks))):
            raise TrainingFreezeV1Error("execution receipts must occupy every bank position in order")
        objective_digests = tuple(
            rotation.objective_evidence_digest
            for block in self.blocks
            for rotation in block.rotations
        )
        if len(objective_digests) != len(set(objective_digests)):
            raise TrainingFreezeV1Error("objective evidence digests must be unique across the run")

    @property
    def episode_count(self) -> int:
        return sum(len(block.rotations) for block in self.blocks)

    def _unsigned_obj(self) -> dict[str, Any]:
        return {
            "schema_version": TRAINING_FREEZE_SCHEMA_VERSION,
            "record_kind": _TRAINING_RECEIPT_KIND,
            "authorization": dict(_RECEIPT_AUTHORIZATION),
            "training_plan_digest": self.training_plan_digest,
            "block_count": len(self.blocks),
            "episode_count": self.episode_count,
            "blocks": [block.as_obj() for block in self.blocks],
            "receipt_boundary": {
                "exact_plan_required_for_verification": True,
                "exact_checkpoint_manifest_bytes_required_for_verification": True,
                "checkpoint_manifest_semantics_independently_verified": False,
                "live_runtime_integration_verified": False,
                "launch_gate_passed": False,
            },
        }

    @property
    def digest(self) -> str:
        return _json_digest(self._unsigned_obj(), domain=_TRAINING_RECEIPT_DOMAIN)

    def as_obj(self) -> dict[str, Any]:
        return {**self._unsigned_obj(), "training_execution_receipt_digest": self.digest}


CheckpointManifestBytes = Mapping[int, tuple[bytes, bytes]]


def _validated_checkpoint_evidence(
    receipt: TrainingExecutionReceiptV1,
    checkpoint_manifest_bytes_by_block: CheckpointManifestBytes,
) -> dict[int, tuple[bytes, bytes]]:
    if not isinstance(checkpoint_manifest_bytes_by_block, Mapping):
        raise TypeError("checkpoint evidence must be a mapping by block position")
    expected_positions = set(range(len(receipt.blocks)))
    if set(checkpoint_manifest_bytes_by_block) != expected_positions:
        raise TrainingFreezeV1Error("checkpoint evidence does not cover exactly every receipt block")
    evidence: dict[int, tuple[bytes, bytes]] = {}
    for position in sorted(expected_positions):
        pair = checkpoint_manifest_bytes_by_block[position]
        if type(pair) is not tuple or len(pair) != 2 or any(type(item) is not bytes for item in pair):
            raise TypeError("checkpoint evidence values must be exact (pre_bytes, post_bytes) tuples")
        evidence[position] = pair
    return evidence


def verify_training_execution_receipt_v1(
    receipt: TrainingExecutionReceiptV1,
    plan: HypothesisCompleteTrainingPlanV1,
    *,
    checkpoint_manifest_bytes_by_block: CheckpointManifestBytes,
) -> TrainingExecutionReceiptV1:
    """Bind one receipt to a freshly replayed plan and exact checkpoint bytes."""

    if type(receipt) is not TrainingExecutionReceiptV1:
        raise TypeError("receipt must be TrainingExecutionReceiptV1")
    if type(plan) is not HypothesisCompleteTrainingPlanV1:
        raise TypeError("plan must be HypothesisCompleteTrainingPlanV1")
    canonical_plan = verify_hypothesis_complete_training_plan_v1(plan)
    if receipt.training_plan_digest != canonical_plan.digest:
        raise TrainingFreezeV1Error("execution receipt is bound to a different training plan")
    if len(receipt.blocks) != len(canonical_plan.blocks):
        raise TrainingFreezeV1Error("execution receipt does not cover every planned block exactly once")
    if receipt.episode_count != canonical_plan.episode_count:
        raise TrainingFreezeV1Error("execution receipt does not cover every planned rotation")
    evidence = _validated_checkpoint_evidence(receipt, checkpoint_manifest_bytes_by_block)

    prior_post_bytes: bytes | None = None
    for block_plan, block_receipt in zip(canonical_plan.blocks, receipt.blocks, strict=True):
        if block_receipt.bank_position != block_plan.bank_position:
            raise TrainingFreezeV1Error("receipt block position differs from the plan")
        if block_receipt.block_plan_digest != block_plan.digest:
            raise TrainingFreezeV1Error("receipt block digest differs from the exact block plan")
        if block_receipt.update_batch_id != block_plan.update_batch_id:
            raise TrainingFreezeV1Error("receipt update-batch ID differs from the plan")
        if (
            block_receipt.optimizer_step_before != block_plan.planned_optimizer_step_before
            or block_receipt.optimizer_step_after != block_plan.planned_optimizer_step_after
        ):
            raise TrainingFreezeV1Error("observed optimizer steps differ from the planned schedule")
        expected_rules = block_plan.hidden_order_display_binding.official_rotation_order_rule_ids
        observed_rules = tuple(item.official_rule_id for item in block_receipt.rotations)
        if observed_rules != expected_rules:
            raise TrainingFreezeV1Error("observed Official rotation order differs from the plan")
        if len(block_receipt.rotations) != block_plan.n0:
            raise TrainingFreezeV1Error("receipt block lacks a complete V0 rotation")

        pre_bytes, post_bytes = evidence[block_plan.bank_position]
        if _sha256_bytes(pre_bytes) != block_receipt.pre_checkpoint_manifest_sha256:
            raise TrainingFreezeV1Error("pre-checkpoint manifest bytes differ from the receipt")
        if _sha256_bytes(post_bytes) != block_receipt.post_checkpoint_manifest_sha256:
            raise TrainingFreezeV1Error("post-checkpoint manifest bytes differ from the receipt")
        if prior_post_bytes is not None and pre_bytes != prior_post_bytes:
            raise TrainingFreezeV1Error("checkpoint manifest byte chain is discontinuous")
        prior_post_bytes = post_bytes

    if receipt.blocks[0].pre_checkpoint_manifest_sha256 != (
        canonical_plan.bindings.initial_checkpoint_manifest_sha256
    ):
        raise TrainingFreezeV1Error("first observed checkpoint differs from the frozen initial state")
    return receipt


def build_training_execution_receipt_v1(
    plan: HypothesisCompleteTrainingPlanV1,
    blocks: Iterable[AtomicBlockExecutionReceiptV1],
    *,
    checkpoint_manifest_bytes_by_block: CheckpointManifestBytes,
) -> TrainingExecutionReceiptV1:
    receipt = TrainingExecutionReceiptV1(plan.digest, tuple(blocks))
    return verify_training_execution_receipt_v1(
        receipt,
        plan,
        checkpoint_manifest_bytes_by_block=checkpoint_manifest_bytes_by_block,
    )


def training_execution_receipt_v1_from_obj(value: object) -> TrainingExecutionReceiptV1:
    obj = _require_mapping(
        value,
        (
            "schema_version",
            "record_kind",
            "authorization",
            "training_plan_digest",
            "block_count",
            "episode_count",
            "blocks",
            "receipt_boundary",
            "training_execution_receipt_digest",
        ),
        name="training execution receipt",
    )
    if (
        type(obj["schema_version"]) is not int
        or obj["schema_version"] != TRAINING_FREEZE_SCHEMA_VERSION
        or obj["record_kind"] != _TRAINING_RECEIPT_KIND
    ):
        raise TrainingFreezeV1Error("training execution receipt has the wrong schema identity")
    _require_authorization(obj["authorization"], _RECEIPT_AUTHORIZATION)
    raw_blocks = obj["blocks"]
    if type(raw_blocks) is not list:
        raise TrainingFreezeV1Error("training execution receipt blocks must be an array")
    receipt = TrainingExecutionReceiptV1(
        training_plan_digest=_require_sha256(
            obj["training_plan_digest"],
            name="training-plan digest",
        ),
        blocks=tuple(_block_receipt_from_obj(item) for item in raw_blocks),
    )
    if _require_integer(obj["block_count"], name="block count", minimum=1) != len(receipt.blocks):
        raise TrainingFreezeV1Error("training execution receipt block count differs")
    if _require_integer(obj["episode_count"], name="episode count", minimum=1) != receipt.episode_count:
        raise TrainingFreezeV1Error("training execution receipt episode count differs")
    if obj["training_execution_receipt_digest"] != receipt.digest:
        raise TrainingFreezeV1Error("training execution receipt digest differs")
    if _dump_json(obj) != _dump_json(receipt.as_obj()):
        raise TrainingFreezeV1Error("training execution receipt contains tampered derived metadata")
    return receipt


def serialize_training_execution_receipt_v1(
    receipt: TrainingExecutionReceiptV1,
    plan: HypothesisCompleteTrainingPlanV1,
    *,
    checkpoint_manifest_bytes_by_block: CheckpointManifestBytes,
) -> str:
    verified = verify_training_execution_receipt_v1(
        receipt,
        plan,
        checkpoint_manifest_bytes_by_block=checkpoint_manifest_bytes_by_block,
    )
    return _dump_json(verified.as_obj())


def parse_training_execution_receipt_v1(
    text: str,
    *,
    plan: HypothesisCompleteTrainingPlanV1,
    checkpoint_manifest_bytes_by_block: CheckpointManifestBytes,
    expected_digest: str | None = None,
) -> TrainingExecutionReceiptV1:
    """Parse only with the exact prospective plan and checkpoint evidence."""

    receipt = training_execution_receipt_v1_from_obj(_load_json(text))
    if expected_digest is not None and receipt.digest != _require_sha256(
        expected_digest,
        name="expected execution-receipt digest",
    ):
        raise TrainingFreezeV1Error("execution receipt differs from the externally expected digest")
    verified = verify_training_execution_receipt_v1(
        receipt,
        plan,
        checkpoint_manifest_bytes_by_block=checkpoint_manifest_bytes_by_block,
    )
    if _dump_json(verified.as_obj()) != text:
        raise TrainingFreezeV1Error("execution-receipt JSON is valid but not canonical compact JSON")
    return verified
