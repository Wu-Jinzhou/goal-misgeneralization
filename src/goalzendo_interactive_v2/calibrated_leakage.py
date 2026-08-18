"""Held-out, fixed-classifier leakage stress for prospective G03-v2 banks.

This module is additive to :mod:`statistical_leakage`.  It fixes the classifier
on a prospectively named calibration construction-cluster population and then
evaluates that classifier exactly once on a disjoint construction-cluster
population.  The grouped percentile bootstrap resamples only the untouched
evaluation clusters.  Its interval is descriptive; this module deliberately
cannot authorize a bank, model execution, or a weight update.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any, Literal, cast

import numpy as np

from goalzendo_interactive.catalog import build_rule_catalog
from goalzendo_interactive.provenance import interactive_source_provenance

from . import population_audit as pa
from . import statistical_leakage as sl
from .population_audit import build_supported_catalog_contract_v2

CALIBRATED_LEAKAGE_SCHEMA_VERSION = 1
CALIBRATED_LEAKAGE_BOOTSTRAP_REPLICATES = 10_000
CALIBRATED_LEAKAGE_REGISTERED_CALIBRATION_CLUSTERS = 384
CALIBRATED_LEAKAGE_REGISTERED_EVALUATION_CLUSTERS = 384

_FROZEN_V1_FINGERPRINT = "24b6d1cc60c09be3b6bbab22d7b7a5b09fef1c4250dd8eb6dc0a9d323aed0ada"
_PLAN_KIND = "g03-v2-prospective-calibration-evaluation-split"
_REPORT_KIND = "g03-v2-held-out-fixed-classifier-leakage-audit"
_PLAN_DOMAIN = "goalzendo-interactive-v2-calibrated-leakage-plan-v1"
_REPORT_DOMAIN = "goalzendo-interactive-v2-calibrated-leakage-report-v1"
_POPULATION_DOMAIN = "goalzendo-interactive-v2-calibrated-leakage-population-v1"
_BLOCK_SURFACE_DOMAIN = "goalzendo-interactive-v2-calibrated-leakage-block-surface-v1"
_BLOCK_CANDIDATE_DOMAIN = "goalzendo-interactive-v2-calibrated-leakage-candidates-v1"
_BLOCK_BINDING_DOMAIN = "goalzendo-interactive-v2-calibrated-leakage-block-binding-v1"
_PARTITION_DOMAIN = "goalzendo-interactive-v2-calibrated-leakage-partition-v1"
_CLASSIFIER_CONTRACT_DOMAIN = "goalzendo-interactive-v2-calibrated-classifier-contract-v1"
_CLASSIFIER_DOMAIN = "goalzendo-interactive-v2-calibrated-classifier-v1"
_CALIBRATION_ROW_DOMAIN = "goalzendo-interactive-v2-calibration-feature-row-v1"
_CALIBRATION_MATRIX_DOMAIN = "goalzendo-interactive-v2-calibration-feature-matrix-v1"
_PREDICTION_ROW_DOMAIN = "goalzendo-interactive-v2-calibrated-prediction-row-v1"
_PREDICTION_EVIDENCE_DOMAIN = "goalzendo-interactive-v2-calibrated-prediction-evidence-v1"
_BOOTSTRAP_SEED_DOMAIN = b"goalzendo-interactive-v2-calibrated-bootstrap-v1\0"

_VIEW_CONTRACT: tuple[tuple[str, bool], ...] = (
    ("catalog_structure", True),
    ("renderer", True),
    ("token_length_bin", True),
    ("global_training_frequency", True),
    ("full_model_visible", True),
    ("schedule_position", False),
    ("request_position", False),
    ("bank_prefix", False),
    ("full_executor_diagnostic", False),
)

_ESTIMAND_NAME = (
    "finite_untouched_evaluation_population_episode_weighted_performance_of_the_"
    "classifier_frozen_on_the_named_calibration_population"
)
_INTERVAL_SCOPE = (
    "descriptive percentile interval of the empirical evaluation-construction-cluster "
    "resampling distribution, conditional on the frozen classifier, exact calibration "
    "population, prospective split, and realized within-cluster contents; it is not a "
    "universal confidence interval and does not cover classifier fitting, calibration "
    "sampling, generator selection, split selection, or future banks"
)

_CLASSIFIER_CONTRACT: dict[str, str | tuple[tuple[str, bool], ...]] = {
    "target": "full_v0_official",
    "fit_population": "prospectively separated calibration construction clusters only",
    "evaluation_population": "disjoint untouched construction clusters only",
    "algorithm": (
        "additive categorical target-rate scorer; for feature f, score contribution is "
        "(n0*positive_count[f]-row_count[f])/(row_count[f]+1); exact rational sums; "
        "unseen evaluation features contribute zero"
    ),
    "top_one_ties": "fractional credit uniformly over the complete exact maximum-score tie",
    "binary_threshold": "exact score strictly greater than zero",
    "frequency_source": "calibration population only and frozen before evaluation",
    "surface_scope": (
        "complete surface only within the schema-v2 statistical meta representation; "
        "the producer manifest and full runtime prompt are not bound"
    ),
    "standalone_parser_feature_provenance": (
        "stored canonical evaluation rows are sufficient for exact score/tie/metric replay, "
        "but the standalone parser does not prove those rows came from the planned blocks"
    ),
    "production_verification": (
        "rerun the exact block-backed builder and compare the complete report, then separately "
        "verify the frozen producer-manifest/full-runtime bridge"
    ),
    "views": _VIEW_CONTRACT,
    "estimand": _ESTIMAND_NAME,
    "interval_scope": _INTERVAL_SCOPE,
}

_AUTHORIZATION = {
    "scope": "additive_nonauthorizing_calibrated_statistical_engineering_only",
    "external_prospective_timestamp_verified": False,
    "registered_power_analysis_artifact_verified": False,
    "registered_runtime_benchmark_artifact_verified": False,
    "bootstrap_universal_coverage_authorized": False,
    "full_manifest_surface_bound": False,
    "frozen_manifest_runtime_bridge_verified": False,
    "standalone_parser_block_feature_rederivation_verified": False,
    "production_bank_authorized": False,
    "model_execution_authorized": False,
    "weight_updates_authorized": False,
}

PopulationRole = Literal["calibration", "evaluation"]
Decision = Literal["descriptive_pass", "leakage", "insufficient_prerequisites"]


class CalibratedLeakageV2Error(ValueError):
    """Raised when a calibrated leakage artifact is noncanonical or unsafe."""


def _dump_json(value: Any) -> str:
    try:
        return json.dumps(value, ensure_ascii=True, allow_nan=False, separators=(",", ":"))
    except (TypeError, ValueError) as exc:
        raise CalibratedLeakageV2Error(f"value is not canonical JSON: {exc}") from exc


def _load_json(text: str) -> Any:
    if type(text) is not str or not text:
        raise CalibratedLeakageV2Error("JSON input must be a nonempty string")

    def no_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise CalibratedLeakageV2Error(f"duplicate JSON object key: {key!r}")
            result[key] = value
        return result

    def reject_constant(value: str) -> None:
        raise CalibratedLeakageV2Error(f"non-finite JSON constant is forbidden: {value}")

    try:
        return json.loads(text, object_pairs_hook=no_duplicates, parse_constant=reject_constant)
    except CalibratedLeakageV2Error:
        raise
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise CalibratedLeakageV2Error(f"invalid JSON: {exc}") from exc


def _digest(value: Any, *, domain: str) -> str:
    payload = domain.encode("ascii") + b"\0" + _dump_json(value).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def _is_sha256(value: object) -> bool:
    return (
        type(value) is str
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _require_sha256(value: object, *, name: str) -> str:
    if not _is_sha256(value):
        raise CalibratedLeakageV2Error(f"{name} must be a lowercase SHA-256")
    return cast(str, value)


def _require_integer(
    value: object,
    *,
    name: str,
    minimum: int = 0,
    maximum: int | None = None,
) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise CalibratedLeakageV2Error(f"{name} must be an integer >= {minimum}")
    if maximum is not None and value > maximum:
        raise CalibratedLeakageV2Error(f"{name} must be an integer <= {maximum}")
    return value


def _require_bool(value: object, *, name: str) -> bool:
    if type(value) is not bool:
        raise CalibratedLeakageV2Error(f"{name} must be Boolean")
    return value


def _require_mapping(value: object, fields: tuple[str, ...], *, name: str) -> Mapping[str, Any]:
    if type(value) is not dict or tuple(value) != fields:
        raise CalibratedLeakageV2Error(f"{name} has noncanonical or reordered fields")
    return cast(Mapping[str, Any], value)


def _require_constant_mapping(
    value: object,
    canonical: Mapping[str, Any],
    *,
    name: str,
) -> Mapping[str, Any]:
    result = _require_mapping(value, tuple(canonical), name=name)
    if dict(result) != dict(canonical):
        raise CalibratedLeakageV2Error(f"{name} differs from the frozen contract")
    return result


def _require_fraction(value: object, *, name: str) -> tuple[int, int]:
    if type(value) is not list or len(value) != 2:
        raise CalibratedLeakageV2Error(f"{name} must be a two-item rational")
    numerator = value[0]
    denominator = value[1]
    if isinstance(numerator, bool) or not isinstance(numerator, int):
        raise CalibratedLeakageV2Error(f"{name} numerator must be an integer")
    _require_integer(denominator, name=f"{name} denominator", minimum=1)
    exact = Fraction(numerator, denominator)
    if (exact.numerator, exact.denominator) != (numerator, denominator):
        raise CalibratedLeakageV2Error(f"{name} must use reduced positive-denominator form")
    return numerator, denominator


def _require_float(value: object, *, name: str) -> float:
    if isinstance(value, bool) or type(value) not in (int, float):
        raise CalibratedLeakageV2Error(f"{name} must be numeric")
    result = float(cast(int | float, value))
    if not math.isfinite(result):
        raise CalibratedLeakageV2Error(f"{name} must be finite")
    return result


def _source_file_sha256(module_file: str | None, *, name: str) -> str:
    if module_file is None:
        raise CalibratedLeakageV2Error(f"{name} source has no file")
    path = Path(module_file)
    if path.is_symlink() or not path.is_file():
        raise CalibratedLeakageV2Error(f"{name} source must be one ordinary file")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _classifier_contract_digest() -> str:
    return _digest(_classifier_contract_obj(), domain=_CLASSIFIER_CONTRACT_DOMAIN)


def _classifier_contract_obj() -> dict[str, Any]:
    """Return fresh JSON containers so callers cannot mutate the contract."""

    return {
        key: [list(item) for item in value] if key == "views" else value
        for key, value in _CLASSIFIER_CONTRACT.items()
    }


@dataclass(frozen=True, slots=True)
class CalibratedSourceBindingV2:
    frozen_v1_source_fingerprint: str
    base_statistical_schema_version: int
    base_statistical_source_sha256: str
    population_audit_source_sha256: str
    calibrated_source_sha256: str
    classifier_contract_digest: str

    def __post_init__(self) -> None:
        for name in (
            "frozen_v1_source_fingerprint",
            "base_statistical_source_sha256",
            "population_audit_source_sha256",
            "calibrated_source_sha256",
            "classifier_contract_digest",
        ):
            _require_sha256(getattr(self, name), name=name)
        if self.frozen_v1_source_fingerprint != _FROZEN_V1_FINGERPRINT:
            raise CalibratedLeakageV2Error("calibrated audit uses the wrong frozen v1 source")
        if isinstance(self.base_statistical_schema_version, bool) or (
            self.base_statistical_schema_version != 2
        ):
            raise CalibratedLeakageV2Error("calibrated audit requires base statistical schema v2")
        if self.classifier_contract_digest != _classifier_contract_digest():
            raise CalibratedLeakageV2Error("classifier contract digest is inconsistent")

    def as_obj(self) -> dict[str, Any]:
        return {
            "frozen_v1_source_fingerprint": self.frozen_v1_source_fingerprint,
            "base_statistical_schema_version": self.base_statistical_schema_version,
            "base_statistical_source_sha256": self.base_statistical_source_sha256,
            "population_audit_source_sha256": self.population_audit_source_sha256,
            "calibrated_source_sha256": self.calibrated_source_sha256,
            "classifier_contract_digest": self.classifier_contract_digest,
        }


def _current_source_binding() -> CalibratedSourceBindingV2:
    if sl.STATISTICAL_LEAKAGE_SCHEMA_VERSION != 2:
        raise CalibratedLeakageV2Error("base statistical leakage schema is no longer v2")
    fingerprint = interactive_source_provenance().fingerprint
    if fingerprint != _FROZEN_V1_FINGERPRINT:
        raise CalibratedLeakageV2Error("frozen v1 source fingerprint changed")
    return CalibratedSourceBindingV2(
        fingerprint,
        sl.STATISTICAL_LEAKAGE_SCHEMA_VERSION,
        _source_file_sha256(sl.__file__, name="base statistical leakage"),
        _source_file_sha256(pa.__file__, name="population audit"),
        _source_file_sha256(__file__, name="calibrated leakage"),
        _classifier_contract_digest(),
    )


@dataclass(frozen=True, slots=True)
class CalibratedLeakageConfigV2:
    """Computational minima may be lowered only for nonauthorizing engineering tests."""

    bootstrap_replicates: int = CALIBRATED_LEAKAGE_BOOTSTRAP_REPLICATES
    minimum_calibration_clusters: int = CALIBRATED_LEAKAGE_REGISTERED_CALIBRATION_CLUSTERS
    minimum_evaluation_clusters: int = CALIBRATED_LEAKAGE_REGISTERED_EVALUATION_CLUSTERS

    def __post_init__(self) -> None:
        _require_integer(self.bootstrap_replicates, name="bootstrap_replicates", minimum=1_000)
        _require_integer(
            self.minimum_calibration_clusters,
            name="minimum_calibration_clusters",
            minimum=2,
            maximum=CALIBRATED_LEAKAGE_REGISTERED_CALIBRATION_CLUSTERS,
        )
        _require_integer(
            self.minimum_evaluation_clusters,
            name="minimum_evaluation_clusters",
            minimum=2,
            maximum=CALIBRATED_LEAKAGE_REGISTERED_EVALUATION_CLUSTERS,
        )

    @property
    def registered_population_minima_used(self) -> bool:
        return (
            self.minimum_calibration_clusters
            == CALIBRATED_LEAKAGE_REGISTERED_CALIBRATION_CLUSTERS
            and self.minimum_evaluation_clusters
            == CALIBRATED_LEAKAGE_REGISTERED_EVALUATION_CLUSTERS
        )

    def as_obj(self) -> dict[str, Any]:
        return {
            "bootstrap_replicates": self.bootstrap_replicates,
            "minimum_calibration_clusters": self.minimum_calibration_clusters,
            "minimum_evaluation_clusters": self.minimum_evaluation_clusters,
            "registered_calibration_cluster_requirement": (
                CALIBRATED_LEAKAGE_REGISTERED_CALIBRATION_CLUSTERS
            ),
            "registered_evaluation_cluster_requirement": (
                CALIBRATED_LEAKAGE_REGISTERED_EVALUATION_CLUSTERS
            ),
            "registered_population_minima_used": self.registered_population_minima_used,
        }


DEFAULT_CALIBRATED_LEAKAGE_CONFIG_V2 = CalibratedLeakageConfigV2()


@dataclass(frozen=True, slots=True)
class CalibratedBlockIdentityV2:
    block_opening_digest: str
    construction_cluster_digest: str
    model_visible_prompt_digest: str
    surface_digest: str
    block_binding_digest: str
    candidate_indices: tuple[int, ...]
    candidate_identity_digest: str

    def __post_init__(self) -> None:
        for name in (
            "block_opening_digest",
            "construction_cluster_digest",
            "model_visible_prompt_digest",
            "surface_digest",
            "block_binding_digest",
            "candidate_identity_digest",
        ):
            _require_sha256(getattr(self, name), name=name)
        if type(self.candidate_indices) is not tuple or self.candidate_indices != tuple(
            sorted(set(self.candidate_indices))
        ):
            raise CalibratedLeakageV2Error("candidate indices must be a sorted unique tuple")
        if len(self.candidate_indices) not in sl.REGISTERED_HYPOTHESIS_COMPLETE_SIZES:
            raise CalibratedLeakageV2Error("candidate indices have an unregistered n0")
        catalog = build_rule_catalog()
        supported = set(build_supported_catalog_contract_v2().supported_indices)
        if any(
            isinstance(index, bool) or not isinstance(index, int) or index not in supported
            for index in self.candidate_indices
        ):
            raise CalibratedLeakageV2Error("candidate identity contains an unsupported rule")
        expected = _candidate_identity_digest(self.candidate_indices)
        if self.candidate_identity_digest != expected:
            raise CalibratedLeakageV2Error("candidate identity digest is inconsistent")
        if any(catalog[index].rule_id == "" for index in self.candidate_indices):  # pragma: no cover
            raise RuntimeError("canonical catalog contains an empty rule id")

    @property
    def canonical_key(self) -> str:
        return self.block_opening_digest

    def as_obj(self) -> dict[str, Any]:
        return {
            "block_opening_digest": self.block_opening_digest,
            "construction_cluster_digest": self.construction_cluster_digest,
            "model_visible_prompt_digest": self.model_visible_prompt_digest,
            "surface_digest": self.surface_digest,
            "block_binding_digest": self.block_binding_digest,
            "candidate_indices": list(self.candidate_indices),
            "candidate_identity_digest": self.candidate_identity_digest,
        }


def _candidate_identity_digest(indices: tuple[int, ...]) -> str:
    catalog = build_rule_catalog()
    return _digest(
        [
            {
                "catalog_index": index,
                "rule_id": catalog[index].rule_id,
                "truth_digest": catalog[index].truth_digest,
            }
            for index in indices
        ],
        domain=_BLOCK_CANDIDATE_DOMAIN,
    )


def _frequency_table_from_population_identity(
    population: CalibratedPopulationIdentityV2,
) -> sl.CatalogFrequencyTableV2:
    contract = build_supported_catalog_contract_v2()
    positions = {index: position for position, index in enumerate(contract.supported_indices)}
    counts = [0] * len(contract.supported_indices)
    for block in population.blocks:
        for index in block.candidate_indices:
            counts[positions[index]] += 1
    return sl.CatalogFrequencyTableV2(tuple(counts))


@dataclass(frozen=True, slots=True)
class CalibratedPopulationIdentityV2:
    role: PopulationRole
    fixed_version_space_size: int
    blocks: tuple[CalibratedBlockIdentityV2, ...]
    construction_clusters: tuple[sl.ConstructionClusterV2, ...]
    exact_conditional_balance: sl.ExactConditionalBalanceResultV2

    def __post_init__(self) -> None:
        if self.role not in {"calibration", "evaluation"}:
            raise CalibratedLeakageV2Error("unknown population role")
        _require_integer(
            self.fixed_version_space_size,
            name="fixed_version_space_size",
            minimum=sl.MIN_LIVE_RULES,
            maximum=sl.MAX_LIVE_RULES,
        )
        if self.fixed_version_space_size not in sl.REGISTERED_HYPOTHESIS_COMPLETE_SIZES:
            raise CalibratedLeakageV2Error("population uses an unregistered n0")
        if type(self.blocks) is not tuple or not self.blocks:
            raise CalibratedLeakageV2Error("population identity requires blocks")
        if any(type(item) is not CalibratedBlockIdentityV2 for item in self.blocks):
            raise CalibratedLeakageV2Error("population contains a foreign block identity")
        if self.blocks != tuple(sorted(self.blocks, key=lambda item: item.canonical_key)):
            raise CalibratedLeakageV2Error("population blocks must be canonically ordered")
        if len({item.block_opening_digest for item in self.blocks}) != len(self.blocks):
            raise CalibratedLeakageV2Error("population repeats an opening")
        if any(len(item.candidate_indices) != self.fixed_version_space_size for item in self.blocks):
            raise CalibratedLeakageV2Error("population block n0 is inconsistent")
        if type(self.construction_clusters) is not tuple or not self.construction_clusters:
            raise CalibratedLeakageV2Error("population requires construction clusters")
        if any(type(item) is not sl.ConstructionClusterV2 for item in self.construction_clusters):
            raise CalibratedLeakageV2Error("population contains a foreign construction cluster")
        if self.construction_clusters != tuple(
            sorted(self.construction_clusters, key=lambda item: item.construction_cluster_digest)
        ):
            raise CalibratedLeakageV2Error("construction clusters must be canonically ordered")
        flattened_members = tuple(
            opening
            for cluster in self.construction_clusters
            for opening in cluster.member_opening_digests
        )
        expected_openings = {item.block_opening_digest for item in self.blocks}
        if len(flattened_members) != len(set(flattened_members)):
            raise CalibratedLeakageV2Error(
                "every population opening must belong to exactly one construction cluster"
            )
        if len(flattened_members) != len(self.blocks) or set(flattened_members) != expected_openings:
            raise CalibratedLeakageV2Error("construction clusters do not partition the population")
        members = {
            opening: cluster.construction_cluster_digest
            for cluster in self.construction_clusters
            for opening in cluster.member_opening_digests
        }
        if any(
            members[item.block_opening_digest] != item.construction_cluster_digest
            for item in self.blocks
        ):
            raise CalibratedLeakageV2Error("block-to-cluster identity is inconsistent")
        if type(self.exact_conditional_balance) is not sl.ExactConditionalBalanceResultV2:
            raise CalibratedLeakageV2Error("population lacks exact conditional-balance evidence")
        expected_rows = len(self.blocks) * self.fixed_version_space_size**2
        if self.exact_conditional_balance.candidate_row_count != expected_rows:
            raise CalibratedLeakageV2Error("conditional-balance row count is inconsistent")
        if not self.exact_conditional_balance.passed:
            raise CalibratedLeakageV2Error("exact conditional surface balance is mandatory")

    @property
    def construction_cluster_count(self) -> int:
        return len(self.construction_clusters)

    def _unsigned_obj(self) -> dict[str, Any]:
        return {
            "role": self.role,
            "fixed_version_space_size": self.fixed_version_space_size,
            "block_count": len(self.blocks),
            "construction_cluster_count": self.construction_cluster_count,
            "blocks": [item.as_obj() for item in self.blocks],
            "construction_clusters": [item.as_obj() for item in self.construction_clusters],
            "exact_conditional_balance": self.exact_conditional_balance.as_obj(),
        }

    @property
    def digest(self) -> str:
        return _digest(self._unsigned_obj(), domain=_POPULATION_DOMAIN)

    def as_obj(self) -> dict[str, Any]:
        return {**self._unsigned_obj(), "population_digest": self.digest}


@dataclass(frozen=True, slots=True)
class CalibrationSplitPlanV2:
    source_binding: CalibratedSourceBindingV2
    calibration_population: CalibratedPopulationIdentityV2
    evaluation_population: CalibratedPopulationIdentityV2
    combined_partition_digest: str

    def __post_init__(self) -> None:
        if type(self.source_binding) is not CalibratedSourceBindingV2:
            raise CalibratedLeakageV2Error("split plan has the wrong source binding type")
        if self.source_binding != _current_source_binding():
            raise CalibratedLeakageV2Error("split plan source provenance differs from current bytes")
        if self.calibration_population.role != "calibration":
            raise CalibratedLeakageV2Error("split plan calibration role is wrong")
        if self.evaluation_population.role != "evaluation":
            raise CalibratedLeakageV2Error("split plan evaluation role is wrong")
        if (
            self.calibration_population.fixed_version_space_size
            != self.evaluation_population.fixed_version_space_size
        ):
            raise CalibratedLeakageV2Error("calibration and evaluation n0 differ")
        calibration_openings = {
            item.block_opening_digest for item in self.calibration_population.blocks
        }
        evaluation_openings = {
            item.block_opening_digest for item in self.evaluation_population.blocks
        }
        if calibration_openings & evaluation_openings:
            raise CalibratedLeakageV2Error("calibration and evaluation openings overlap")
        calibration_clusters = {
            item.construction_cluster_digest
            for item in self.calibration_population.construction_clusters
        }
        evaluation_clusters = {
            item.construction_cluster_digest
            for item in self.evaluation_population.construction_clusters
        }
        if calibration_clusters & evaluation_clusters:
            raise CalibratedLeakageV2Error(
                "calibration and evaluation construction clusters overlap"
            )
        _require_sha256(self.combined_partition_digest, name="combined_partition_digest")
        expected = _combined_partition_digest(
            self.calibration_population,
            self.evaluation_population,
        )
        if self.combined_partition_digest != expected:
            raise CalibratedLeakageV2Error("combined partition digest is inconsistent")

    def _unsigned_obj(self) -> dict[str, Any]:
        return {
            "schema_version": CALIBRATED_LEAKAGE_SCHEMA_VERSION,
            "plan_kind": _PLAN_KIND,
            "source_binding": self.source_binding.as_obj(),
            "classifier_contract": _classifier_contract_obj(),
            "calibration_population": self.calibration_population.as_obj(),
            "evaluation_population": self.evaluation_population.as_obj(),
            "combined_partition_digest": self.combined_partition_digest,
            "statistical_meta_surface_scope": (
                "schema-v2 statistical meta representation only"
            ),
            "full_manifest_surface_bound": False,
            "frozen_manifest_runtime_bridge_required": True,
            "content_addressing_establishes_temporal_priority": False,
            "external_registration_required_for_temporal_priority": True,
        }

    @property
    def digest(self) -> str:
        return _digest(self._unsigned_obj(), domain=_PLAN_DOMAIN)

    def as_obj(self) -> dict[str, Any]:
        return {**self._unsigned_obj(), "calibration_split_plan_digest": self.digest}


def _combined_partition_digest(
    calibration: CalibratedPopulationIdentityV2,
    evaluation: CalibratedPopulationIdentityV2,
) -> str:
    return _digest(
        {
            "calibration_population_digest": calibration.digest,
            "evaluation_population_digest": evaluation.digest,
            "ordered_cluster_roles": [
                *[
                    [item.construction_cluster_digest, "calibration"]
                    for item in calibration.construction_clusters
                ],
                *[
                    [item.construction_cluster_digest, "evaluation"]
                    for item in evaluation.construction_clusters
                ],
            ],
        },
        domain=_PARTITION_DOMAIN,
    )


def _validated_blocks(
    blocks: object,
    *,
    role: PopulationRole,
) -> tuple[sl.HypothesisCompleteMetaBlockV2, ...]:
    if type(blocks) is not tuple or not blocks:
        raise CalibratedLeakageV2Error(f"{role} blocks must be a nonempty exact tuple")
    materialized = cast(tuple[object, ...], blocks)
    if any(type(block) is not sl.HypothesisCompleteMetaBlockV2 for block in materialized):
        raise CalibratedLeakageV2Error(f"{role} population contains a foreign block")
    typed = cast(tuple[sl.HypothesisCompleteMetaBlockV2, ...], materialized)
    for block in typed:
        # Re-run nested and outer checks so object.__new__ cannot forge a valid surface.
        for observation in block.opening:
            observation.__post_init__()
        for surface in block.rotation_surfaces:
            surface.__post_init__()
        block.binding.__post_init__()
        block.__post_init__()
    digests = tuple(block.block_opening_digest for block in typed)
    if digests != tuple(sorted(digests)):
        raise CalibratedLeakageV2Error(f"{role} blocks must be ordered by opening digest")
    if len(set(digests)) != len(digests):
        raise CalibratedLeakageV2Error(f"{role} population repeats an exact opening")
    sizes = {len(block.version_space) for block in typed}
    if len(sizes) != 1 or next(iter(sizes)) not in sl.REGISTERED_HYPOTHESIS_COMPLETE_SIZES:
        raise CalibratedLeakageV2Error(f"{role} population requires fixed n0 in {{8,12,16}}")
    return typed


def _episodes(
    blocks: tuple[sl.HypothesisCompleteMetaBlockV2, ...],
    opening_to_cluster: Mapping[str, str],
) -> tuple[sl._Episode, ...]:
    return tuple(
        sl._Episode(
            group_id=opening_to_cluster[block.block_opening_digest],
            opening_digest=block.block_opening_digest,
            official_index=official_index,
            candidate_indices=block.version_space.indices,
            designated_indices=block.designated_indices,
            surface=block.rotation_surfaces[position],
            opening=block.opening,
        )
        for block in blocks
        for position, official_index in enumerate(block.version_space.indices)
    )


def _block_identity(
    block: sl.HypothesisCompleteMetaBlockV2,
    cluster_digest: str,
) -> CalibratedBlockIdentityV2:
    candidate_indices = block.version_space.indices
    surface_obj = {
        "block_opening_digest": block.block_opening_digest,
        "model_visible_prompt_digest": block.model_visible_prompt_digest,
        "rotation_surfaces": [surface.as_obj() for surface in block.rotation_surfaces],
    }
    return CalibratedBlockIdentityV2(
        block.block_opening_digest,
        cluster_digest,
        block.model_visible_prompt_digest,
        _digest(surface_obj, domain=_BLOCK_SURFACE_DOMAIN),
        _digest(block.as_binding_obj(), domain=_BLOCK_BINDING_DOMAIN),
        candidate_indices,
        _candidate_identity_digest(candidate_indices),
    )


def _population_identity(
    role: PopulationRole,
    blocks: tuple[sl.HypothesisCompleteMetaBlockV2, ...],
    opening_to_cluster: Mapping[str, str],
    joint_clusters: tuple[sl.ConstructionClusterV2, ...],
) -> CalibratedPopulationIdentityV2:
    opening_set = {block.block_opening_digest for block in blocks}
    clusters = tuple(
        cluster
        for cluster in joint_clusters
        if set(cluster.member_opening_digests).issubset(opening_set)
    )
    n0 = len(blocks[0].version_space)
    episodes = _episodes(blocks, opening_to_cluster)
    exact = sl._exact_conditional_balance("meta_role", episodes, fixed_size=n0)
    if not exact.passed:
        raise CalibratedLeakageV2Error("exact conditional surface balance is mandatory")
    return CalibratedPopulationIdentityV2(
        role,
        n0,
        tuple(
            _block_identity(block, opening_to_cluster[block.block_opening_digest])
            for block in blocks
        ),
        clusters,
        exact,
    )


def build_calibration_split_plan_v2(
    calibration_blocks: tuple[sl.HypothesisCompleteMetaBlockV2, ...],
    evaluation_blocks: tuple[sl.HypothesisCompleteMetaBlockV2, ...],
) -> CalibrationSplitPlanV2:
    """Bind population roles before fitting; publish ``plan.digest`` externally.

    Content addressing proves exact identity, not when the digest was published.
    External preregistration is therefore still required and remains unverified
    by this nonauthorizing module.
    """

    calibration = _validated_blocks(calibration_blocks, role="calibration")
    evaluation = _validated_blocks(evaluation_blocks, role="evaluation")
    if len(calibration[0].version_space) != len(evaluation[0].version_space):
        raise CalibratedLeakageV2Error("calibration and evaluation n0 differ")
    calibration_openings = {block.block_opening_digest for block in calibration}
    evaluation_openings = {block.block_opening_digest for block in evaluation}
    if calibration_openings & evaluation_openings:
        raise CalibratedLeakageV2Error("calibration and evaluation openings overlap")
    combined = tuple(sorted((*calibration, *evaluation), key=lambda block: block.block_opening_digest))
    opening_to_cluster, clusters = sl._derive_construction_clusters(combined)
    role_by_opening = {
        **{opening: "calibration" for opening in calibration_openings},
        **{opening: "evaluation" for opening in evaluation_openings},
    }
    for cluster in clusters:
        roles = {role_by_opening[opening] for opening in cluster.member_opening_digests}
        if len(roles) != 1:
            raise CalibratedLeakageV2Error(
                "calibration and evaluation construction lineages overlap"
            )
    calibration_identity = _population_identity(
        "calibration", calibration, opening_to_cluster, clusters
    )
    evaluation_identity = _population_identity(
        "evaluation", evaluation, opening_to_cluster, clusters
    )
    return CalibrationSplitPlanV2(
        _current_source_binding(),
        calibration_identity,
        evaluation_identity,
        _combined_partition_digest(calibration_identity, evaluation_identity),
    )


@dataclass(frozen=True, slots=True)
class CalibrationFeatureCellV2:
    token: str
    row_count: int
    positive_count: int

    def __post_init__(self) -> None:
        if type(self.token) is not str or not self.token or not self.token.isascii():
            raise CalibratedLeakageV2Error("calibration feature token must be nonempty ASCII")
        _require_integer(self.row_count, name="feature row_count", minimum=1)
        _require_integer(self.positive_count, name="feature positive_count")
        if self.positive_count > self.row_count:
            raise CalibratedLeakageV2Error("feature positives exceed rows")

    def as_obj(self) -> dict[str, Any]:
        return {
            "token": self.token,
            "row_count": self.row_count,
            "positive_count": self.positive_count,
        }


@dataclass(frozen=True, slots=True)
class FrozenCalibrationClassifierV2:
    view_name: str
    model_visible: bool
    calibration_population_digest: str
    calibration_frequency_table_digest: str
    calibration_feature_matrix_digest: str
    total_candidate_rows: int
    positive_rows: int
    prevalence_denominator: int
    cells: tuple[CalibrationFeatureCellV2, ...]

    def __post_init__(self) -> None:
        expected = dict(_VIEW_CONTRACT).get(self.view_name)
        if expected is None or self.model_visible is not expected:
            raise CalibratedLeakageV2Error("classifier view contract is inconsistent")
        for name in (
            "calibration_population_digest",
            "calibration_frequency_table_digest",
            "calibration_feature_matrix_digest",
        ):
            _require_sha256(getattr(self, name), name=name)
        _require_integer(self.total_candidate_rows, name="total_candidate_rows", minimum=1)
        _require_integer(self.positive_rows, name="positive_rows", minimum=1)
        _require_integer(self.prevalence_denominator, name="prevalence_denominator", minimum=2)
        if self.positive_rows * self.prevalence_denominator != self.total_candidate_rows:
            raise CalibratedLeakageV2Error("classifier prevalence is not exact reciprocal n0")
        if type(self.cells) is not tuple or not self.cells:
            raise CalibratedLeakageV2Error("frozen classifier requires feature cells")
        if any(type(cell) is not CalibrationFeatureCellV2 for cell in self.cells):
            raise CalibratedLeakageV2Error("classifier contains a foreign feature cell")
        tokens = tuple(cell.token for cell in self.cells)
        if tokens != tuple(sorted(set(tokens))):
            raise CalibratedLeakageV2Error("classifier feature cells must be token-sorted and unique")
        if any(cell.row_count > self.total_candidate_rows for cell in self.cells):
            raise CalibratedLeakageV2Error("feature count exceeds calibration row count")
        if any(cell.positive_count > self.positive_rows for cell in self.cells):
            raise CalibratedLeakageV2Error("feature positives exceed calibration positives")

    def _unsigned_obj(self) -> dict[str, Any]:
        return {
            "view_name": self.view_name,
            "model_visible": self.model_visible,
            "classifier_contract_digest": _classifier_contract_digest(),
            "calibration_population_digest": self.calibration_population_digest,
            "calibration_frequency_table_digest": self.calibration_frequency_table_digest,
            "calibration_feature_matrix_digest": self.calibration_feature_matrix_digest,
            "total_candidate_rows": self.total_candidate_rows,
            "positive_rows": self.positive_rows,
            "prevalence_denominator": self.prevalence_denominator,
            "cells": [cell.as_obj() for cell in self.cells],
        }

    @property
    def digest(self) -> str:
        return _digest(self._unsigned_obj(), domain=_CLASSIFIER_DOMAIN)

    def as_obj(self) -> dict[str, Any]:
        return {**self._unsigned_obj(), "frozen_classifier_digest": self.digest}

    def score(self, tokens: tuple[str, ...]) -> Fraction:
        if len(tokens) != len(set(tokens)):
            raise CalibratedLeakageV2Error("evaluation feature row repeats a token")
        counts = {cell.token: cell for cell in self.cells}
        result = Fraction()
        for token in tokens:
            cell = counts.get(token)
            if cell is None:
                continue
            centered = self.prevalence_denominator * cell.positive_count - cell.row_count
            if centered:
                result += Fraction(centered, cell.row_count + 1)
        return result


def _fit_classifier(
    view_name: str,
    model_visible: bool,
    episodes: tuple[sl._Episode, ...],
    frequencies: sl.CatalogFrequencyTableV2,
    population_digest: str,
) -> FrozenCalibrationClassifierV2:
    totals: Counter[str] = Counter()
    positives: Counter[str] = Counter()
    total_rows = 0
    positive_rows = 0
    row_digests: list[str] = []
    for episode in episodes:
        for candidate_index in episode.candidate_indices:
            tokens = sl._meta_feature_tokens(episode, candidate_index, view_name, frequencies)
            if len(tokens) != len(set(tokens)):
                raise CalibratedLeakageV2Error("calibration feature row repeats a token")
            positive = candidate_index == episode.official_index
            totals.update(tokens)
            if positive:
                positives.update(tokens)
                positive_rows += 1
            total_rows += 1
            row_digests.append(
                _digest(
                    {
                        "construction_cluster_digest": episode.group_id,
                        "block_opening_digest": episode.opening_digest,
                        "official_index": episode.official_index,
                        "candidate_index": candidate_index,
                        "positive": positive,
                        "feature_tokens": list(tokens),
                    },
                    domain=_CALIBRATION_ROW_DOMAIN,
                )
            )
    if total_rows % positive_rows:
        raise CalibratedLeakageV2Error("calibration prevalence is not reciprocal integer")
    matrix_digest = _digest(row_digests, domain=_CALIBRATION_MATRIX_DOMAIN)
    return FrozenCalibrationClassifierV2(
        view_name,
        model_visible,
        population_digest,
        frequencies.digest,
        matrix_digest,
        total_rows,
        positive_rows,
        total_rows // positive_rows,
        tuple(CalibrationFeatureCellV2(token, totals[token], positives[token]) for token in sorted(totals)),
    )


def _prediction_row_digest(
    *,
    classifier_digest: str,
    construction_cluster_digest: str,
    block_opening_digest: str,
    model_visible_prompt_digest: str,
    surface_digest: str,
    official_index: int,
    pattern: sl.PredictionPatternV2,
    candidate_feature_tokens: tuple[tuple[str, ...], ...],
) -> str:
    catalog = build_rule_catalog()
    return _digest(
        {
            "frozen_classifier_digest": classifier_digest,
            "construction_cluster_digest": construction_cluster_digest,
            "block_opening_digest": block_opening_digest,
            "model_visible_prompt_digest": model_visible_prompt_digest,
            "surface_digest": surface_digest,
            "official_index": official_index,
            "candidate_predictions": [
                {
                    "catalog_index": index,
                    "rule_id": catalog[index].rule_id,
                    "truth_digest": catalog[index].truth_digest,
                    "evaluation_feature_tokens": list(feature_tokens),
                    "exact_score": list(score),
                    "predicted_official": predicted,
                }
                for index, feature_tokens, score, predicted in zip(
                    pattern.candidate_indices,
                    candidate_feature_tokens,
                    pattern.exact_scores,
                    pattern.predicted_official,
                    strict=True,
                )
            ],
            "winner_indices": list(pattern.winner_indices),
        },
        domain=_PREDICTION_ROW_DOMAIN,
    )


@dataclass(frozen=True, slots=True)
class CalibratedPredictionEpisodeV2:
    construction_cluster_digest: str
    block_opening_digest: str
    model_visible_prompt_digest: str
    surface_digest: str
    official_index: int
    pattern_index: int
    credit: tuple[int, int]
    candidate_feature_tokens: tuple[tuple[str, ...], ...]
    candidate_prediction_digest: str

    def __post_init__(self) -> None:
        for name in (
            "construction_cluster_digest",
            "block_opening_digest",
            "model_visible_prompt_digest",
            "surface_digest",
            "candidate_prediction_digest",
        ):
            _require_sha256(getattr(self, name), name=name)
        supported = set(build_supported_catalog_contract_v2().supported_indices)
        if (
            isinstance(self.official_index, bool)
            or not isinstance(self.official_index, int)
            or self.official_index not in supported
        ):
            raise CalibratedLeakageV2Error("prediction episode Official is unsupported")
        _require_integer(self.pattern_index, name="pattern_index")
        if type(self.credit) is not tuple or len(self.credit) != 2:
            raise CalibratedLeakageV2Error("prediction credit must be an exact rational tuple")
        if (
            isinstance(self.credit[0], bool)
            or not isinstance(self.credit[0], int)
            or isinstance(self.credit[1], bool)
            or not isinstance(self.credit[1], int)
            or self.credit[1] <= 0
        ):
            raise CalibratedLeakageV2Error("prediction credit has invalid integer fields")
        exact = Fraction(*self.credit)
        if (
            (exact.numerator, exact.denominator) != self.credit
            or not Fraction() <= exact <= 1
        ):
            raise CalibratedLeakageV2Error("prediction credit is not canonical in [0,1]")
        if type(self.candidate_feature_tokens) is not tuple or not self.candidate_feature_tokens:
            raise CalibratedLeakageV2Error(
                "prediction episode requires canonical candidate feature rows"
            )
        for feature_row in self.candidate_feature_tokens:
            if type(feature_row) is not tuple or not feature_row:
                raise CalibratedLeakageV2Error("evaluation feature row must be a nonempty tuple")
            if any(type(token) is not str or not token or not token.isascii() for token in feature_row):
                raise CalibratedLeakageV2Error(
                    "evaluation feature tokens must be nonempty ASCII strings"
                )
            if feature_row != tuple(sorted(set(feature_row))):
                raise CalibratedLeakageV2Error(
                    "evaluation feature tokens must be sorted and unique"
                )

    @property
    def canonical_key(self) -> tuple[str, int]:
        return self.block_opening_digest, self.official_index

    def as_obj(self) -> dict[str, Any]:
        return {
            "construction_cluster_digest": self.construction_cluster_digest,
            "block_opening_digest": self.block_opening_digest,
            "model_visible_prompt_digest": self.model_visible_prompt_digest,
            "surface_digest": self.surface_digest,
            "official_index": self.official_index,
            "pattern_index": self.pattern_index,
            "credit": list(self.credit),
            "candidate_feature_tokens": [
                list(feature_row) for feature_row in self.candidate_feature_tokens
            ],
            "candidate_prediction_digest": self.candidate_prediction_digest,
        }


@dataclass(frozen=True, slots=True)
class CalibratedPredictionEvidenceV2:
    view_name: str
    frozen_classifier_digest: str
    evaluation_population_digest: str
    patterns: tuple[sl.PredictionPatternV2, ...]
    episodes: tuple[CalibratedPredictionEpisodeV2, ...]

    def __post_init__(self) -> None:
        if self.view_name not in dict(_VIEW_CONTRACT):
            raise CalibratedLeakageV2Error("prediction evidence uses an unknown view")
        _require_sha256(self.frozen_classifier_digest, name="frozen_classifier_digest")
        _require_sha256(self.evaluation_population_digest, name="evaluation_population_digest")
        if type(self.patterns) is not tuple or not self.patterns:
            raise CalibratedLeakageV2Error("prediction evidence requires patterns")
        if any(type(pattern) is not sl.PredictionPatternV2 for pattern in self.patterns):
            raise CalibratedLeakageV2Error("prediction evidence contains a foreign pattern")
        if tuple(pattern.digest for pattern in self.patterns) != tuple(
            sorted({pattern.digest for pattern in self.patterns})
        ):
            raise CalibratedLeakageV2Error("prediction patterns must be digest-sorted and unique")
        if type(self.episodes) is not tuple or not self.episodes:
            raise CalibratedLeakageV2Error("prediction evidence requires episodes")
        if any(type(row) is not CalibratedPredictionEpisodeV2 for row in self.episodes):
            raise CalibratedLeakageV2Error("prediction evidence contains a foreign episode")
        if self.episodes != tuple(sorted(self.episodes, key=lambda row: row.canonical_key)):
            raise CalibratedLeakageV2Error("prediction episodes must be canonically ordered")
        if len({row.canonical_key for row in self.episodes}) != len(self.episodes):
            raise CalibratedLeakageV2Error("prediction episode identities repeat")
        for row in self.episodes:
            if row.pattern_index >= len(self.patterns):
                raise CalibratedLeakageV2Error("prediction episode references an absent pattern")
            pattern = self.patterns[row.pattern_index]
            if row.official_index not in pattern.candidate_indices:
                raise CalibratedLeakageV2Error("Official is absent from prediction candidates")
            if len(row.candidate_feature_tokens) != len(pattern.candidate_indices):
                raise CalibratedLeakageV2Error(
                    "candidate feature rows do not align to prediction candidates"
                )
            expected_credit = (
                Fraction(1, len(pattern.winner_indices))
                if row.official_index in pattern.winner_indices
                else Fraction()
            )
            if row.credit != (expected_credit.numerator, expected_credit.denominator):
                raise CalibratedLeakageV2Error("prediction credit differs from exact tie handling")
            expected_digest = _prediction_row_digest(
                classifier_digest=self.frozen_classifier_digest,
                construction_cluster_digest=row.construction_cluster_digest,
                block_opening_digest=row.block_opening_digest,
                model_visible_prompt_digest=row.model_visible_prompt_digest,
                surface_digest=row.surface_digest,
                official_index=row.official_index,
                pattern=pattern,
                candidate_feature_tokens=row.candidate_feature_tokens,
            )
            if row.candidate_prediction_digest != expected_digest:
                raise CalibratedLeakageV2Error("per-candidate prediction digest is inconsistent")

    def _unsigned_obj(self) -> dict[str, Any]:
        return {
            "view_name": self.view_name,
            "frozen_classifier_digest": self.frozen_classifier_digest,
            "evaluation_population_digest": self.evaluation_population_digest,
            "patterns": [pattern.as_obj() for pattern in self.patterns],
            "episodes": [row.as_obj() for row in self.episodes],
        }

    @property
    def digest(self) -> str:
        return _digest(self._unsigned_obj(), domain=_PREDICTION_EVIDENCE_DOMAIN)

    def as_obj(self) -> dict[str, Any]:
        return {**self._unsigned_obj(), "prediction_evidence_digest": self.digest}


def _score_evaluation(
    classifier: FrozenCalibrationClassifierV2,
    episodes: tuple[sl._Episode, ...],
    frequencies: sl.CatalogFrequencyTableV2,
    population: CalibratedPopulationIdentityV2,
) -> CalibratedPredictionEvidenceV2:
    identity_by_opening = {item.block_opening_digest: item for item in population.blocks}
    raw: list[
        tuple[
            sl._Episode,
            CalibratedBlockIdentityV2,
            sl.PredictionPatternV2,
            Fraction,
            tuple[tuple[str, ...], ...],
        ]
    ] = []
    for episode in episodes:
        identity = identity_by_opening[episode.opening_digest]
        candidates = episode.candidate_indices
        candidate_feature_tokens = tuple(
            tuple(
                sorted(
                    sl._meta_feature_tokens(
                        episode,
                        candidate_index,
                        classifier.view_name,
                        frequencies,
                    )
                )
            )
            for candidate_index in candidates
        )
        scores = tuple(
            classifier.score(feature_tokens)
            for feature_tokens in candidate_feature_tokens
        )
        maximum = max(scores)
        winners = tuple(
            candidate_index
            for candidate_index, score in zip(candidates, scores, strict=True)
            if score == maximum
        )
        pattern = sl.PredictionPatternV2(
            candidates,
            tuple((score.numerator, score.denominator) for score in scores),
            tuple(score > 0 for score in scores),
            winners,
        )
        credit = Fraction(1, len(winners)) if episode.official_index in winners else Fraction()
        raw.append((episode, identity, pattern, credit, candidate_feature_tokens))
    patterns = tuple(sorted(set(item[2] for item in raw), key=lambda item: item.digest))
    positions = {pattern: index for index, pattern in enumerate(patterns)}
    rows = tuple(
        sorted(
            (
                CalibratedPredictionEpisodeV2(
                    episode.group_id,
                    episode.opening_digest,
                    identity.model_visible_prompt_digest,
                    identity.surface_digest,
                    episode.official_index,
                    positions[pattern],
                    (credit.numerator, credit.denominator),
                    candidate_feature_tokens,
                    _prediction_row_digest(
                        classifier_digest=classifier.digest,
                        construction_cluster_digest=episode.group_id,
                        block_opening_digest=episode.opening_digest,
                        model_visible_prompt_digest=identity.model_visible_prompt_digest,
                        surface_digest=identity.surface_digest,
                        official_index=episode.official_index,
                        pattern=pattern,
                        candidate_feature_tokens=candidate_feature_tokens,
                    ),
                )
                for episode, identity, pattern, credit, candidate_feature_tokens in raw
            ),
            key=lambda row: row.canonical_key,
        )
    )
    return CalibratedPredictionEvidenceV2(
        classifier.view_name,
        classifier.digest,
        population.digest,
        patterns,
        rows,
    )


def _order_stat_interval(values: np.ndarray) -> tuple[float, float]:
    ordered = np.sort(values)
    count = len(ordered)
    return (
        float(ordered[math.floor((count - 1) * 0.025)]),
        float(ordered[math.ceil((count - 1) * 0.975)]),
    )


def _bootstrap_seed(
    plan_digest: str,
    view_name: str,
    prediction_evidence_digest: str,
    metric: str,
) -> int:
    payload = (
        _BOOTSTRAP_SEED_DOMAIN
        + bytes.fromhex(plan_digest)
        + b"\0"
        + view_name.encode("ascii")
        + b"\0"
        + bytes.fromhex(prediction_evidence_digest)
        + b"\0"
        + metric.encode("ascii")
    )
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def _cluster_bootstrap(
    evidence: CalibratedPredictionEvidenceV2,
    population: CalibratedPopulationIdentityV2,
    *,
    plan_digest: str,
    replicates: int,
) -> tuple[tuple[float, float], tuple[float, float]]:
    cluster_ids = tuple(
        cluster.construction_cluster_digest for cluster in population.construction_clusters
    )
    positions = {cluster: index for index, cluster in enumerate(cluster_ids)}
    top = np.zeros(len(cluster_ids), dtype=np.float64)
    chance = np.zeros(len(cluster_ids), dtype=np.float64)
    counts = np.zeros(len(cluster_ids), dtype=np.float64)
    confusion = np.zeros((len(cluster_ids), 4), dtype=np.float64)
    for row in evidence.episodes:
        position = positions[row.construction_cluster_digest]
        pattern = evidence.patterns[row.pattern_index]
        top[position] += float(Fraction(*row.credit))
        chance[position] += 1 / len(pattern.candidate_indices)
        counts[position] += 1
        for candidate_index, predicted in zip(
            pattern.candidate_indices, pattern.predicted_official, strict=True
        ):
            positive = candidate_index == row.official_index
            confusion[position] += (
                int(positive and predicted),
                int(positive and not predicted),
                int(not positive and not predicted),
                int(not positive and predicted),
            )
    seed = _bootstrap_seed(plan_digest, evidence.view_name, evidence.digest, "top_one_excess")
    rng = np.random.default_rng(seed)
    top_values = np.empty(replicates, dtype=np.float64)
    ba_values = np.empty(replicates, dtype=np.float64)
    chunk = 256
    for start in range(0, replicates, chunk):
        stop = min(replicates, start + chunk)
        sampled = rng.integers(0, len(cluster_ids), size=(stop - start, len(cluster_ids)))
        denominator = counts[sampled].sum(axis=1)
        top_values[start:stop] = (
            top[sampled].sum(axis=1) - chance[sampled].sum(axis=1)
        ) / denominator
        cells = confusion[sampled].sum(axis=1)
        ba_values[start:stop] = 0.5 * (
            cells[:, 0] / (cells[:, 0] + cells[:, 1])
            + cells[:, 2] / (cells[:, 2] + cells[:, 3])
        )
    return _order_stat_interval(top_values), _order_stat_interval(ba_values)


@dataclass(frozen=True, slots=True)
class FixedClassifierEstimateV2:
    estimand_name: str
    interval_scope: str
    observation_count: int
    evaluation_cluster_count: int
    top_one_accuracy: tuple[int, int]
    full_v0_chance: tuple[int, int]
    top_one_excess: tuple[int, int]
    excess_interval_lower: float
    excess_interval_upper: float
    macro_balanced_accuracy: tuple[int, int]
    macro_ba_interval_lower: float
    macro_ba_interval_upper: float
    descriptive_equivalence_passed: bool
    registered_cluster_power_met: bool
    decision: Decision
    prediction_evidence_digest: str

    def __post_init__(self) -> None:
        if self.estimand_name != _ESTIMAND_NAME or self.interval_scope != _INTERVAL_SCOPE:
            raise CalibratedLeakageV2Error("fixed-classifier estimand contract changed")
        _require_integer(self.observation_count, name="observation_count", minimum=1)
        _require_integer(self.evaluation_cluster_count, name="evaluation_cluster_count", minimum=2)
        for name in ("top_one_accuracy", "full_v0_chance", "top_one_excess", "macro_balanced_accuracy"):
            value = getattr(self, name)
            if type(value) is not tuple or len(value) != 2:
                raise CalibratedLeakageV2Error(f"{name} is malformed")
            exact = Fraction(*value)
            if (exact.numerator, exact.denominator) != value:
                raise CalibratedLeakageV2Error(f"{name} is not a reduced rational")
        for name in (
            "excess_interval_lower",
            "excess_interval_upper",
            "macro_ba_interval_lower",
            "macro_ba_interval_upper",
        ):
            _require_float(getattr(self, name), name=name)
        if self.excess_interval_lower > self.excess_interval_upper:
            raise CalibratedLeakageV2Error("excess interval is reversed")
        if self.macro_ba_interval_lower > self.macro_ba_interval_upper:
            raise CalibratedLeakageV2Error("macro-BA interval is reversed")
        _require_bool(self.descriptive_equivalence_passed, name="descriptive_equivalence_passed")
        _require_bool(self.registered_cluster_power_met, name="registered_cluster_power_met")
        expected_decision: Decision = (
            "insufficient_prerequisites"
            if not self.registered_cluster_power_met
            else "descriptive_pass"
            if self.descriptive_equivalence_passed
            else "leakage"
        )
        if self.decision != expected_decision:
            raise CalibratedLeakageV2Error("fixed-classifier decision is inconsistent")
        _require_sha256(self.prediction_evidence_digest, name="prediction_evidence_digest")

    def as_obj(self) -> dict[str, Any]:
        return {
            "estimand_name": self.estimand_name,
            "interval_scope": self.interval_scope,
            "observation_count": self.observation_count,
            "evaluation_cluster_count": self.evaluation_cluster_count,
            "top_one_accuracy": list(self.top_one_accuracy),
            "full_v0_chance": list(self.full_v0_chance),
            "top_one_excess": list(self.top_one_excess),
            "excess_interval_lower": self.excess_interval_lower,
            "excess_interval_upper": self.excess_interval_upper,
            "macro_balanced_accuracy": list(self.macro_balanced_accuracy),
            "macro_ba_interval_lower": self.macro_ba_interval_lower,
            "macro_ba_interval_upper": self.macro_ba_interval_upper,
            "descriptive_equivalence_passed": self.descriptive_equivalence_passed,
            "registered_cluster_power_met": self.registered_cluster_power_met,
            "decision": self.decision,
            "prediction_evidence_digest": self.prediction_evidence_digest,
        }


def _fixed_estimate(
    evidence: CalibratedPredictionEvidenceV2,
    population: CalibratedPopulationIdentityV2,
    *,
    plan_digest: str,
    replicates: int,
) -> FixedClassifierEstimateV2:
    credits = [Fraction(*row.credit) for row in evidence.episodes]
    accuracy = sum(credits, Fraction()) / len(credits)
    chance = Fraction(1, population.fixed_version_space_size)
    excess = accuracy - chance
    tp = fn = tn = fp = 0
    for row in evidence.episodes:
        pattern = evidence.patterns[row.pattern_index]
        for candidate_index, predicted in zip(
            pattern.candidate_indices, pattern.predicted_official, strict=True
        ):
            positive = candidate_index == row.official_index
            tp += int(positive and predicted)
            fn += int(positive and not predicted)
            tn += int(not positive and not predicted)
            fp += int(not positive and predicted)
    ba = (Fraction(tp, tp + fn) + Fraction(tn, tn + fp)) / 2
    excess_interval, ba_interval = _cluster_bootstrap(
        evidence,
        population,
        plan_digest=plan_digest,
        replicates=replicates,
    )
    descriptive = (
        excess_interval[0] <= 0 <= excess_interval[1]
        and excess_interval[1] < sl.STATISTICAL_LEAKAGE_EXCESS_MARGIN
        and ba_interval[0] <= 0.5 <= ba_interval[1]
        and ba_interval[1] < 0.55
    )
    powered = (
        population.construction_cluster_count
        >= CALIBRATED_LEAKAGE_REGISTERED_EVALUATION_CLUSTERS
    )
    decision: Decision = (
        "insufficient_prerequisites"
        if not powered
        else "descriptive_pass"
        if descriptive
        else "leakage"
    )
    return FixedClassifierEstimateV2(
        _ESTIMAND_NAME,
        _INTERVAL_SCOPE,
        len(evidence.episodes),
        population.construction_cluster_count,
        (accuracy.numerator, accuracy.denominator),
        (chance.numerator, chance.denominator),
        (excess.numerator, excess.denominator),
        *excess_interval,
        (ba.numerator, ba.denominator),
        *ba_interval,
        descriptive,
        powered,
        decision,
        evidence.digest,
    )


@dataclass(frozen=True, slots=True)
class CalibratedFeatureViewResultV2:
    view_name: str
    model_visible: bool
    classifier: FrozenCalibrationClassifierV2
    prediction_evidence: CalibratedPredictionEvidenceV2
    estimate: FixedClassifierEstimateV2

    def __post_init__(self) -> None:
        expected = dict(_VIEW_CONTRACT).get(self.view_name)
        if expected is None or self.model_visible is not expected:
            raise CalibratedLeakageV2Error("feature view contract is inconsistent")
        if self.classifier.view_name != self.view_name:
            raise CalibratedLeakageV2Error("feature view classifier name differs")
        if self.classifier.model_visible is not self.model_visible:
            raise CalibratedLeakageV2Error("feature view visibility differs")
        if self.prediction_evidence.view_name != self.view_name:
            raise CalibratedLeakageV2Error("feature view prediction name differs")
        if self.prediction_evidence.frozen_classifier_digest != self.classifier.digest:
            raise CalibratedLeakageV2Error("prediction evidence uses another classifier")
        for row in self.prediction_evidence.episodes:
            pattern = self.prediction_evidence.patterns[row.pattern_index]
            replayed_scores = tuple(
                self.classifier.score(feature_tokens)
                for feature_tokens in row.candidate_feature_tokens
            )
            recorded_scores = tuple(Fraction(*score) for score in pattern.exact_scores)
            if replayed_scores != recorded_scores:
                raise CalibratedLeakageV2Error(
                    "frozen classifier score differs from stored evaluation prediction"
                )
            replayed_predictions = tuple(score > 0 for score in replayed_scores)
            if replayed_predictions != pattern.predicted_official:
                raise CalibratedLeakageV2Error(
                    "frozen classifier binary predictions fail exact evaluation replay"
                )
            maximum = max(replayed_scores)
            replayed_winners = tuple(
                candidate_index
                for candidate_index, score in zip(
                    pattern.candidate_indices,
                    replayed_scores,
                    strict=True,
                )
                if score == maximum
            )
            if replayed_winners != pattern.winner_indices:
                raise CalibratedLeakageV2Error(
                    "frozen classifier winner tie fails exact evaluation replay"
                )
        if self.estimate.prediction_evidence_digest != self.prediction_evidence.digest:
            raise CalibratedLeakageV2Error("estimate uses another prediction artifact")

    def as_obj(self) -> dict[str, Any]:
        return {
            "view_name": self.view_name,
            "model_visible": self.model_visible,
            "classifier": self.classifier.as_obj(),
            "prediction_evidence": self.prediction_evidence.as_obj(),
            "estimate": self.estimate.as_obj(),
        }


@dataclass(frozen=True, slots=True)
class CalibratedLeakageAuditReportV2:
    source_binding: CalibratedSourceBindingV2
    config: CalibratedLeakageConfigV2
    split_plan: CalibrationSplitPlanV2
    calibration_frequency_table_digest: str
    views: tuple[CalibratedFeatureViewResultV2, ...]

    def __post_init__(self) -> None:
        if type(self.source_binding) is not CalibratedSourceBindingV2:
            raise CalibratedLeakageV2Error("report source binding has the wrong type")
        if self.source_binding != _current_source_binding():
            raise CalibratedLeakageV2Error("report source provenance differs from current bytes")
        if type(self.split_plan) is not CalibrationSplitPlanV2:
            raise CalibratedLeakageV2Error("report split plan has the wrong type")
        if self.split_plan.source_binding != self.source_binding:
            raise CalibratedLeakageV2Error("report and split plan source bindings differ")
        if type(self.config) is not CalibratedLeakageConfigV2:
            raise CalibratedLeakageV2Error("report config has the wrong type")
        _require_sha256(
            self.calibration_frequency_table_digest,
            name="calibration_frequency_table_digest",
        )
        if type(self.views) is not tuple:
            raise CalibratedLeakageV2Error("report views must be an exact tuple")
        if any(type(item) is not CalibratedFeatureViewResultV2 for item in self.views):
            raise CalibratedLeakageV2Error("report contains a foreign feature view")
        if tuple(
            (item.view_name, item.model_visible) for item in self.views
        ) != _VIEW_CONTRACT:
            raise CalibratedLeakageV2Error("report views differ from the frozen contract")
        calibration = self.split_plan.calibration_population
        evaluation = self.split_plan.evaluation_population
        if (
            self.calibration_frequency_table_digest
            != _frequency_table_from_population_identity(calibration).digest
        ):
            raise CalibratedLeakageV2Error(
                "calibration frequency digest differs from the planned candidate population"
            )
        if calibration.construction_cluster_count < self.config.minimum_calibration_clusters:
            raise CalibratedLeakageV2Error("report has insufficient calibration construction clusters")
        if evaluation.construction_cluster_count < self.config.minimum_evaluation_clusters:
            raise CalibratedLeakageV2Error("report has insufficient evaluation construction clusters")
        expected_calibration_rows = (
            len(calibration.blocks) * calibration.fixed_version_space_size**2
        )
        expected_calibration_positives = (
            len(calibration.blocks) * calibration.fixed_version_space_size
        )
        for view in self.views:
            if view.classifier.calibration_population_digest != calibration.digest:
                raise CalibratedLeakageV2Error("classifier calibration identity differs")
            if (
                view.classifier.calibration_frequency_table_digest
                != self.calibration_frequency_table_digest
            ):
                raise CalibratedLeakageV2Error("classifier frequency identity differs")
            if view.classifier.prevalence_denominator != calibration.fixed_version_space_size:
                raise CalibratedLeakageV2Error("classifier n0 differs from calibration")
            if view.classifier.total_candidate_rows != expected_calibration_rows:
                raise CalibratedLeakageV2Error("classifier row count differs from calibration")
            if view.classifier.positive_rows != expected_calibration_positives:
                raise CalibratedLeakageV2Error("classifier positive count differs from calibration")
            if view.prediction_evidence.evaluation_population_digest != evaluation.digest:
                raise CalibratedLeakageV2Error("prediction evaluation identity differs")
            identities = {
                item.block_opening_digest: item for item in evaluation.blocks
            }
            completion: dict[str, list[int]] = {}
            for row in view.prediction_evidence.episodes:
                identity = identities.get(row.block_opening_digest)
                if identity is None:
                    raise CalibratedLeakageV2Error("prediction uses an unplanned opening")
                if row.construction_cluster_digest != identity.construction_cluster_digest:
                    raise CalibratedLeakageV2Error("prediction uses a forged construction cluster")
                if row.model_visible_prompt_digest != identity.model_visible_prompt_digest:
                    raise CalibratedLeakageV2Error("prediction uses a forged prompt surface")
                if row.surface_digest != identity.surface_digest:
                    raise CalibratedLeakageV2Error("prediction uses a forged complete surface")
                pattern = view.prediction_evidence.patterns[row.pattern_index]
                if pattern.candidate_indices != identity.candidate_indices:
                    raise CalibratedLeakageV2Error("prediction uses a forged candidate set")
                completion.setdefault(row.block_opening_digest, []).append(row.official_index)
            if set(completion) != set(identities):
                raise CalibratedLeakageV2Error("predictions do not cover every evaluation opening")
            for opening, officials in completion.items():
                if tuple(sorted(officials)) != identities[opening].candidate_indices:
                    raise CalibratedLeakageV2Error("prediction evidence is not hypothesis complete")
            expected_estimate = _fixed_estimate(
                view.prediction_evidence,
                evaluation,
                plan_digest=self.split_plan.digest,
                replicates=self.config.bootstrap_replicates,
            )
            if view.estimate != expected_estimate:
                raise CalibratedLeakageV2Error("fixed-classifier estimate fails exact replay")

    @property
    def calibration_cluster_requirement_met(self) -> bool:
        return (
            self.split_plan.calibration_population.construction_cluster_count
            >= CALIBRATED_LEAKAGE_REGISTERED_CALIBRATION_CLUSTERS
        )

    @property
    def evaluation_cluster_requirement_met(self) -> bool:
        return (
            self.split_plan.evaluation_population.construction_cluster_count
            >= CALIBRATED_LEAKAGE_REGISTERED_EVALUATION_CLUSTERS
        )

    @property
    def exact_conditional_surface_balance_passed(self) -> bool:
        return (
            self.split_plan.calibration_population.exact_conditional_balance.passed
            and self.split_plan.evaluation_population.exact_conditional_balance.passed
        )

    @property
    def power_runtime_prerequisites_passed(self) -> bool:
        # This primitive has no parser for externally registered power/runtime artifacts.
        return False

    @property
    def statistical_interval_authorized(self) -> bool:
        return False

    @property
    def primary_statistical_gate_passed(self) -> bool:
        return False

    def _unsigned_obj(self) -> dict[str, Any]:
        return {
            "schema_version": CALIBRATED_LEAKAGE_SCHEMA_VERSION,
            "report_kind": _REPORT_KIND,
            "authorization": dict(_AUTHORIZATION),
            "source_binding": self.source_binding.as_obj(),
            "classifier_contract": _classifier_contract_obj(),
            "config": self.config.as_obj(),
            "calibration_split_plan": self.split_plan.as_obj(),
            "calibration_frequency_table_digest": self.calibration_frequency_table_digest,
            "views": [view.as_obj() for view in self.views],
            "exact_conditional_surface_balance_passed": (
                self.exact_conditional_surface_balance_passed
            ),
            "full_manifest_surface_bound": False,
            "frozen_manifest_runtime_bridge_verified": False,
            "standalone_parser_block_feature_rederivation_verified": False,
            "exact_block_backed_rebuild_required_for_production": True,
            "calibration_cluster_requirement_met": self.calibration_cluster_requirement_met,
            "evaluation_cluster_requirement_met": self.evaluation_cluster_requirement_met,
            "registered_power_analysis_artifact_verified": False,
            "registered_runtime_benchmark_artifact_verified": False,
            "power_runtime_prerequisites_passed": self.power_runtime_prerequisites_passed,
            "statistical_interval_authorized": self.statistical_interval_authorized,
            "primary_statistical_gate_passed": self.primary_statistical_gate_passed,
            "production_bank_authorized": False,
            "model_execution_authorized": False,
            "weight_updates_authorized": False,
        }

    @property
    def digest(self) -> str:
        return _digest(self._unsigned_obj(), domain=_REPORT_DOMAIN)

    def as_obj(self) -> dict[str, Any]:
        return {**self._unsigned_obj(), "calibrated_leakage_audit_digest": self.digest}


def build_calibrated_meta_role_leakage_audit_v2(
    calibration_blocks: tuple[sl.HypothesisCompleteMetaBlockV2, ...],
    evaluation_blocks: tuple[sl.HypothesisCompleteMetaBlockV2, ...],
    split_plan: CalibrationSplitPlanV2,
    *,
    expected_split_plan_digest: str,
    config: CalibratedLeakageConfigV2 = DEFAULT_CALIBRATED_LEAKAGE_CONFIG_V2,
) -> CalibratedLeakageAuditReportV2:
    """Fit on calibration clusters and score disjoint untouched clusters once."""

    _require_sha256(expected_split_plan_digest, name="expected_split_plan_digest")
    if type(split_plan) is not CalibrationSplitPlanV2:
        raise TypeError("split_plan must be a CalibrationSplitPlanV2")
    if split_plan.digest != expected_split_plan_digest:
        raise CalibratedLeakageV2Error("split plan differs from the externally expected digest")
    if type(config) is not CalibratedLeakageConfigV2:
        raise TypeError("config must be a CalibratedLeakageConfigV2")
    calibration = _validated_blocks(calibration_blocks, role="calibration")
    evaluation = _validated_blocks(evaluation_blocks, role="evaluation")
    recomputed_plan = build_calibration_split_plan_v2(calibration, evaluation)
    if recomputed_plan != split_plan:
        raise CalibratedLeakageV2Error("split plan differs from exact block/surface rederivation")
    if (
        split_plan.calibration_population.construction_cluster_count
        < config.minimum_calibration_clusters
    ):
        raise CalibratedLeakageV2Error("insufficient calibration construction clusters")
    if (
        split_plan.evaluation_population.construction_cluster_count
        < config.minimum_evaluation_clusters
    ):
        raise CalibratedLeakageV2Error("insufficient evaluation construction clusters")
    calibration_opening_to_cluster = {
        block.block_opening_digest: block.construction_cluster_digest
        for block in split_plan.calibration_population.blocks
    }
    evaluation_opening_to_cluster = {
        block.block_opening_digest: block.construction_cluster_digest
        for block in split_plan.evaluation_population.blocks
    }
    calibration_episodes = _episodes(calibration, calibration_opening_to_cluster)
    evaluation_episodes = _episodes(evaluation, evaluation_opening_to_cluster)
    frequencies = sl.derive_catalog_frequency_table_v2(calibration)
    views: list[CalibratedFeatureViewResultV2] = []
    for view_name, model_visible in _VIEW_CONTRACT:
        classifier = _fit_classifier(
            view_name,
            model_visible,
            calibration_episodes,
            frequencies,
            split_plan.calibration_population.digest,
        )
        predictions = _score_evaluation(
            classifier,
            evaluation_episodes,
            frequencies,
            split_plan.evaluation_population,
        )
        estimate = _fixed_estimate(
            predictions,
            split_plan.evaluation_population,
            plan_digest=split_plan.digest,
            replicates=config.bootstrap_replicates,
        )
        views.append(
            CalibratedFeatureViewResultV2(
                view_name,
                model_visible,
                classifier,
                predictions,
                estimate,
            )
        )
    return CalibratedLeakageAuditReportV2(
        _current_source_binding(),
        config,
        split_plan,
        frequencies.digest,
        tuple(views),
    )


def verify_calibrated_leakage_audit_against_blocks_v2(
    report: CalibratedLeakageAuditReportV2,
    calibration_blocks: tuple[sl.HypothesisCompleteMetaBlockV2, ...],
    evaluation_blocks: tuple[sl.HypothesisCompleteMetaBlockV2, ...],
    *,
    expected_report_digest: str,
    expected_split_plan_digest: str,
) -> CalibratedLeakageAuditReportV2:
    """Rederive every feature row and artifact from exact schema-v2 blocks.

    The standalone JSON parser can replay scores and metrics from stored rows,
    but it cannot authenticate that those rows came from the planned blocks.
    This explicit verifier closes that statistical-meta boundary by rebuilding
    the complete report.  It still does not replace the separate producer-
    manifest/full-runtime surface bridge.
    """

    if type(report) is not CalibratedLeakageAuditReportV2:
        raise TypeError("report must be a CalibratedLeakageAuditReportV2")
    _require_sha256(expected_report_digest, name="expected_report_digest")
    _require_sha256(expected_split_plan_digest, name="expected_split_plan_digest")
    if report.digest != expected_report_digest:
        raise CalibratedLeakageV2Error("report differs from the externally expected digest")
    if report.split_plan.digest != expected_split_plan_digest:
        raise CalibratedLeakageV2Error("split plan differs from the externally expected digest")
    rebuilt = build_calibrated_meta_role_leakage_audit_v2(
        calibration_blocks,
        evaluation_blocks,
        report.split_plan,
        expected_split_plan_digest=expected_split_plan_digest,
        config=report.config,
    )
    if rebuilt != report:
        raise CalibratedLeakageV2Error(
            "report differs from exact block-backed feature and prediction rederivation"
        )
    return report


def _source_binding_from_obj(value: object) -> CalibratedSourceBindingV2:
    obj = _require_mapping(
        value,
        (
            "frozen_v1_source_fingerprint",
            "base_statistical_schema_version",
            "base_statistical_source_sha256",
            "population_audit_source_sha256",
            "calibrated_source_sha256",
            "classifier_contract_digest",
        ),
        name="source_binding",
    )
    result = CalibratedSourceBindingV2(
        _require_sha256(obj["frozen_v1_source_fingerprint"], name="frozen v1 fingerprint"),
        _require_integer(
            obj["base_statistical_schema_version"],
            name="base statistical schema version",
            minimum=2,
            maximum=2,
        ),
        _require_sha256(obj["base_statistical_source_sha256"], name="base source sha256"),
        _require_sha256(obj["population_audit_source_sha256"], name="population source sha256"),
        _require_sha256(obj["calibrated_source_sha256"], name="calibrated source sha256"),
        _require_sha256(obj["classifier_contract_digest"], name="classifier contract digest"),
    )
    if result != _current_source_binding():
        raise CalibratedLeakageV2Error("forged or stale source provenance")
    return result


def _conditional_from_obj(value: object) -> sl.ExactConditionalBalanceResultV2:
    obj = _require_mapping(
        value,
        (
            "candidate_row_count",
            "conditional_cell_count",
            "violating_cell_count",
            "maximum_absolute_excess",
            "conditional_cell_digest",
            "passed",
        ),
        name="exact_conditional_balance",
    )
    fraction = _require_fraction(obj["maximum_absolute_excess"], name="maximum excess")
    result = sl.ExactConditionalBalanceResultV2(
        _require_integer(obj["candidate_row_count"], name="conditional rows", minimum=1),
        _require_integer(obj["conditional_cell_count"], name="conditional cells", minimum=1),
        _require_integer(obj["violating_cell_count"], name="violating cells"),
        fraction[0],
        fraction[1],
        _require_sha256(obj["conditional_cell_digest"], name="conditional cell digest"),
    )
    if _require_bool(obj["passed"], name="conditional passed") is not result.passed:
        raise CalibratedLeakageV2Error("conditional-balance pass flag is inconsistent")
    return result


def _cluster_from_obj(value: object) -> sl.ConstructionClusterV2:
    obj = _require_mapping(
        value,
        ("construction_cluster_digest", "member_opening_digests"),
        name="construction_cluster",
    )
    members = obj["member_opening_digests"]
    if type(members) is not list:
        raise CalibratedLeakageV2Error("cluster members must be a list")
    return sl.ConstructionClusterV2(
        _require_sha256(obj["construction_cluster_digest"], name="cluster digest"),
        tuple(_require_sha256(item, name="cluster member") for item in members),
    )


def _block_identity_from_obj(value: object) -> CalibratedBlockIdentityV2:
    obj = _require_mapping(
        value,
        (
            "block_opening_digest",
            "construction_cluster_digest",
            "model_visible_prompt_digest",
            "surface_digest",
            "block_binding_digest",
            "candidate_indices",
            "candidate_identity_digest",
        ),
        name="block_identity",
    )
    indices = obj["candidate_indices"]
    if type(indices) is not list:
        raise CalibratedLeakageV2Error("candidate indices must be a list")
    return CalibratedBlockIdentityV2(
        _require_sha256(obj["block_opening_digest"], name="opening digest"),
        _require_sha256(obj["construction_cluster_digest"], name="cluster digest"),
        _require_sha256(obj["model_visible_prompt_digest"], name="prompt digest"),
        _require_sha256(obj["surface_digest"], name="surface digest"),
        _require_sha256(obj["block_binding_digest"], name="block binding digest"),
        tuple(_require_integer(item, name="candidate index") for item in indices),
        _require_sha256(obj["candidate_identity_digest"], name="candidate identity digest"),
    )


def _population_from_obj(value: object, *, role: PopulationRole) -> CalibratedPopulationIdentityV2:
    obj = _require_mapping(
        value,
        (
            "role",
            "fixed_version_space_size",
            "block_count",
            "construction_cluster_count",
            "blocks",
            "construction_clusters",
            "exact_conditional_balance",
            "population_digest",
        ),
        name=f"{role}_population",
    )
    if obj["role"] != role:
        raise CalibratedLeakageV2Error("population role differs from field")
    blocks_obj = obj["blocks"]
    clusters_obj = obj["construction_clusters"]
    if type(blocks_obj) is not list or type(clusters_obj) is not list:
        raise CalibratedLeakageV2Error("population blocks/clusters must be lists")
    result = CalibratedPopulationIdentityV2(
        role,
        _require_integer(obj["fixed_version_space_size"], name="fixed n0", minimum=8),
        tuple(_block_identity_from_obj(item) for item in blocks_obj),
        tuple(_cluster_from_obj(item) for item in clusters_obj),
        _conditional_from_obj(obj["exact_conditional_balance"]),
    )
    if _require_integer(obj["block_count"], name="block_count", minimum=1) != len(result.blocks):
        raise CalibratedLeakageV2Error("population block count is inconsistent")
    if _require_integer(
        obj["construction_cluster_count"], name="cluster_count", minimum=1
    ) != len(result.construction_clusters):
        raise CalibratedLeakageV2Error("population cluster count is inconsistent")
    if _require_sha256(obj["population_digest"], name="population digest") != result.digest:
        raise CalibratedLeakageV2Error("population digest is inconsistent")
    return result


def calibration_split_plan_v2_from_obj(
    value: object,
    *,
    expected_digest: str,
) -> CalibrationSplitPlanV2:
    _require_sha256(expected_digest, name="expected_digest")
    obj = _require_mapping(
        value,
        (
            "schema_version",
            "plan_kind",
            "source_binding",
            "classifier_contract",
            "calibration_population",
            "evaluation_population",
            "combined_partition_digest",
            "statistical_meta_surface_scope",
            "full_manifest_surface_bound",
            "frozen_manifest_runtime_bridge_required",
            "content_addressing_establishes_temporal_priority",
            "external_registration_required_for_temporal_priority",
            "calibration_split_plan_digest",
        ),
        name="calibration_split_plan",
    )
    if (
        _require_integer(obj["schema_version"], name="schema_version", minimum=1, maximum=1)
        != CALIBRATED_LEAKAGE_SCHEMA_VERSION
        or obj["plan_kind"] != _PLAN_KIND
    ):
        raise CalibratedLeakageV2Error("unknown split-plan schema or kind")
    _require_constant_mapping(
        obj["classifier_contract"],
        _classifier_contract_obj(),
        name="split-plan classifier_contract",
    )
    if obj["statistical_meta_surface_scope"] != "schema-v2 statistical meta representation only":
        raise CalibratedLeakageV2Error("split-plan statistical surface scope changed")
    if _require_bool(obj["full_manifest_surface_bound"], name="full manifest surface bound"):
        raise CalibratedLeakageV2Error("split plan cannot claim full manifest surface binding")
    if not _require_bool(
        obj["frozen_manifest_runtime_bridge_required"],
        name="frozen manifest runtime bridge requirement",
    ):
        raise CalibratedLeakageV2Error("frozen manifest/runtime bridge must remain required")
    if _require_bool(
        obj["content_addressing_establishes_temporal_priority"],
        name="temporal priority claim",
    ):
        raise CalibratedLeakageV2Error("content addressing cannot establish temporal priority")
    if not _require_bool(
        obj["external_registration_required_for_temporal_priority"],
        name="external registration requirement",
    ):
        raise CalibratedLeakageV2Error("external registration must remain required")
    result = CalibrationSplitPlanV2(
        _source_binding_from_obj(obj["source_binding"]),
        _population_from_obj(obj["calibration_population"], role="calibration"),
        _population_from_obj(obj["evaluation_population"], role="evaluation"),
        _require_sha256(obj["combined_partition_digest"], name="combined partition digest"),
    )
    supplied = _require_sha256(obj["calibration_split_plan_digest"], name="split-plan digest")
    if supplied != result.digest:
        raise CalibratedLeakageV2Error("split-plan digest is inconsistent")
    if result.digest != expected_digest:
        raise CalibratedLeakageV2Error("split plan differs from externally expected digest")
    return result


def serialize_calibration_split_plan_v2(plan: CalibrationSplitPlanV2) -> str:
    if type(plan) is not CalibrationSplitPlanV2:
        raise TypeError("plan must be a CalibrationSplitPlanV2")
    return _dump_json(plan.as_obj()) + "\n"


def parse_calibration_split_plan_v2(
    text: str,
    *,
    expected_digest: str,
) -> CalibrationSplitPlanV2:
    return calibration_split_plan_v2_from_obj(_load_json(text), expected_digest=expected_digest)


def _config_from_obj(value: object) -> CalibratedLeakageConfigV2:
    obj = _require_mapping(
        value,
        (
            "bootstrap_replicates",
            "minimum_calibration_clusters",
            "minimum_evaluation_clusters",
            "registered_calibration_cluster_requirement",
            "registered_evaluation_cluster_requirement",
            "registered_population_minima_used",
        ),
        name="config",
    )
    result = CalibratedLeakageConfigV2(
        _require_integer(obj["bootstrap_replicates"], name="bootstrap_replicates", minimum=1_000),
        _require_integer(
            obj["minimum_calibration_clusters"], name="minimum_calibration_clusters", minimum=2
        ),
        _require_integer(
            obj["minimum_evaluation_clusters"], name="minimum_evaluation_clusters", minimum=2
        ),
    )
    if _require_integer(
        obj["registered_calibration_cluster_requirement"],
        name="registered calibration requirement",
    ) != CALIBRATED_LEAKAGE_REGISTERED_CALIBRATION_CLUSTERS:
        raise CalibratedLeakageV2Error("registered calibration requirement changed")
    if _require_integer(
        obj["registered_evaluation_cluster_requirement"],
        name="registered evaluation requirement",
    ) != CALIBRATED_LEAKAGE_REGISTERED_EVALUATION_CLUSTERS:
        raise CalibratedLeakageV2Error("registered evaluation requirement changed")
    if _require_bool(
        obj["registered_population_minima_used"], name="registered_population_minima_used"
    ) is not result.registered_population_minima_used:
        raise CalibratedLeakageV2Error("registered-minima flag is inconsistent")
    return result


def _feature_cell_from_obj(value: object) -> CalibrationFeatureCellV2:
    obj = _require_mapping(
        value,
        ("token", "row_count", "positive_count"),
        name="calibration_feature_cell",
    )
    return CalibrationFeatureCellV2(
        cast(str, obj["token"]),
        _require_integer(obj["row_count"], name="feature row count", minimum=1),
        _require_integer(obj["positive_count"], name="feature positive count"),
    )


def _classifier_from_obj(value: object) -> FrozenCalibrationClassifierV2:
    obj = _require_mapping(
        value,
        (
            "view_name",
            "model_visible",
            "classifier_contract_digest",
            "calibration_population_digest",
            "calibration_frequency_table_digest",
            "calibration_feature_matrix_digest",
            "total_candidate_rows",
            "positive_rows",
            "prevalence_denominator",
            "cells",
            "frozen_classifier_digest",
        ),
        name="frozen_classifier",
    )
    if obj["classifier_contract_digest"] != _classifier_contract_digest():
        raise CalibratedLeakageV2Error("classifier contract digest changed")
    cells = obj["cells"]
    if type(cells) is not list:
        raise CalibratedLeakageV2Error("classifier cells must be a list")
    if type(obj["view_name"]) is not str:
        raise CalibratedLeakageV2Error("classifier view name must be a string")
    result = FrozenCalibrationClassifierV2(
        obj["view_name"],
        _require_bool(obj["model_visible"], name="model_visible"),
        _require_sha256(obj["calibration_population_digest"], name="calibration population"),
        _require_sha256(obj["calibration_frequency_table_digest"], name="frequency table"),
        _require_sha256(obj["calibration_feature_matrix_digest"], name="feature matrix"),
        _require_integer(obj["total_candidate_rows"], name="candidate rows", minimum=1),
        _require_integer(obj["positive_rows"], name="positive rows", minimum=1),
        _require_integer(obj["prevalence_denominator"], name="prevalence denominator", minimum=2),
        tuple(_feature_cell_from_obj(item) for item in cells),
    )
    if _require_sha256(obj["frozen_classifier_digest"], name="classifier digest") != result.digest:
        raise CalibratedLeakageV2Error("frozen classifier digest is inconsistent")
    return result


def _pattern_from_obj(value: object) -> sl.PredictionPatternV2:
    obj = _require_mapping(
        value,
        ("candidate_indices", "exact_scores", "predicted_official", "winner_indices"),
        name="prediction_pattern",
    )
    indices = obj["candidate_indices"]
    scores = obj["exact_scores"]
    predictions = obj["predicted_official"]
    winners = obj["winner_indices"]
    if not all(type(item) is list for item in (indices, scores, predictions, winners)):
        raise CalibratedLeakageV2Error("prediction pattern arrays must be lists")
    exact_scores = tuple(_require_fraction(score, name="prediction score") for score in scores)
    return sl.PredictionPatternV2(
        tuple(_require_integer(item, name="candidate index") for item in indices),
        exact_scores,
        tuple(_require_bool(item, name="prediction bit") for item in predictions),
        tuple(_require_integer(item, name="winner index") for item in winners),
    )


def _prediction_episode_from_obj(value: object) -> CalibratedPredictionEpisodeV2:
    obj = _require_mapping(
        value,
        (
            "construction_cluster_digest",
            "block_opening_digest",
            "model_visible_prompt_digest",
            "surface_digest",
            "official_index",
            "pattern_index",
            "credit",
            "candidate_feature_tokens",
            "candidate_prediction_digest",
        ),
        name="prediction_episode",
    )
    credit = _require_fraction(obj["credit"], name="prediction credit")
    feature_rows = obj["candidate_feature_tokens"]
    if type(feature_rows) is not list or any(type(row) is not list for row in feature_rows):
        raise CalibratedLeakageV2Error("candidate feature rows must be nested lists")
    parsed_feature_rows: list[tuple[str, ...]] = []
    for row in feature_rows:
        parsed_row: list[str] = []
        for token in row:
            if type(token) is not str:
                raise CalibratedLeakageV2Error("evaluation feature token must be a string")
            parsed_row.append(token)
        parsed_feature_rows.append(tuple(parsed_row))
    return CalibratedPredictionEpisodeV2(
        _require_sha256(obj["construction_cluster_digest"], name="cluster digest"),
        _require_sha256(obj["block_opening_digest"], name="opening digest"),
        _require_sha256(obj["model_visible_prompt_digest"], name="prompt digest"),
        _require_sha256(obj["surface_digest"], name="surface digest"),
        _require_integer(obj["official_index"], name="Official index"),
        _require_integer(obj["pattern_index"], name="pattern index"),
        credit,
        tuple(parsed_feature_rows),
        _require_sha256(obj["candidate_prediction_digest"], name="prediction digest"),
    )


def _prediction_evidence_from_obj(value: object) -> CalibratedPredictionEvidenceV2:
    obj = _require_mapping(
        value,
        (
            "view_name",
            "frozen_classifier_digest",
            "evaluation_population_digest",
            "patterns",
            "episodes",
            "prediction_evidence_digest",
        ),
        name="prediction_evidence",
    )
    patterns = obj["patterns"]
    episodes = obj["episodes"]
    if type(patterns) is not list or type(episodes) is not list:
        raise CalibratedLeakageV2Error("prediction patterns/episodes must be lists")
    if type(obj["view_name"]) is not str:
        raise CalibratedLeakageV2Error("prediction view name must be a string")
    result = CalibratedPredictionEvidenceV2(
        obj["view_name"],
        _require_sha256(obj["frozen_classifier_digest"], name="classifier digest"),
        _require_sha256(obj["evaluation_population_digest"], name="evaluation population"),
        tuple(_pattern_from_obj(item) for item in patterns),
        tuple(_prediction_episode_from_obj(item) for item in episodes),
    )
    if _require_sha256(obj["prediction_evidence_digest"], name="evidence digest") != result.digest:
        raise CalibratedLeakageV2Error("prediction evidence digest is inconsistent")
    return result


def _estimate_from_obj(value: object) -> FixedClassifierEstimateV2:
    fields = (
        "estimand_name",
        "interval_scope",
        "observation_count",
        "evaluation_cluster_count",
        "top_one_accuracy",
        "full_v0_chance",
        "top_one_excess",
        "excess_interval_lower",
        "excess_interval_upper",
        "macro_balanced_accuracy",
        "macro_ba_interval_lower",
        "macro_ba_interval_upper",
        "descriptive_equivalence_passed",
        "registered_cluster_power_met",
        "decision",
        "prediction_evidence_digest",
    )
    obj = _require_mapping(value, fields, name="fixed_classifier_estimate")
    if type(obj["estimand_name"]) is not str or type(obj["interval_scope"]) is not str:
        raise CalibratedLeakageV2Error("estimand names must be strings")
    decision = obj["decision"]
    if decision not in {"descriptive_pass", "leakage", "insufficient_prerequisites"}:
        raise CalibratedLeakageV2Error("unknown estimate decision")
    return FixedClassifierEstimateV2(
        obj["estimand_name"],
        obj["interval_scope"],
        _require_integer(obj["observation_count"], name="observation count", minimum=1),
        _require_integer(obj["evaluation_cluster_count"], name="cluster count", minimum=2),
        _require_fraction(obj["top_one_accuracy"], name="top-one accuracy"),
        _require_fraction(obj["full_v0_chance"], name="chance"),
        _require_fraction(obj["top_one_excess"], name="top-one excess"),
        _require_float(obj["excess_interval_lower"], name="excess lower"),
        _require_float(obj["excess_interval_upper"], name="excess upper"),
        _require_fraction(obj["macro_balanced_accuracy"], name="macro BA"),
        _require_float(obj["macro_ba_interval_lower"], name="BA lower"),
        _require_float(obj["macro_ba_interval_upper"], name="BA upper"),
        _require_bool(obj["descriptive_equivalence_passed"], name="descriptive pass"),
        _require_bool(obj["registered_cluster_power_met"], name="registered power"),
        cast(Decision, decision),
        _require_sha256(obj["prediction_evidence_digest"], name="prediction evidence"),
    )


def _view_from_obj(value: object) -> CalibratedFeatureViewResultV2:
    obj = _require_mapping(
        value,
        ("view_name", "model_visible", "classifier", "prediction_evidence", "estimate"),
        name="feature_view",
    )
    if type(obj["view_name"]) is not str:
        raise CalibratedLeakageV2Error("feature view name must be a string")
    return CalibratedFeatureViewResultV2(
        obj["view_name"],
        _require_bool(obj["model_visible"], name="model_visible"),
        _classifier_from_obj(obj["classifier"]),
        _prediction_evidence_from_obj(obj["prediction_evidence"]),
        _estimate_from_obj(obj["estimate"]),
    )


def calibrated_leakage_audit_v2_from_obj(
    value: object,
    *,
    expected_digest: str,
) -> CalibratedLeakageAuditReportV2:
    _require_sha256(expected_digest, name="expected_digest")
    fields = (
        "schema_version",
        "report_kind",
        "authorization",
        "source_binding",
        "classifier_contract",
        "config",
        "calibration_split_plan",
        "calibration_frequency_table_digest",
        "views",
        "exact_conditional_surface_balance_passed",
        "full_manifest_surface_bound",
        "frozen_manifest_runtime_bridge_verified",
        "standalone_parser_block_feature_rederivation_verified",
        "exact_block_backed_rebuild_required_for_production",
        "calibration_cluster_requirement_met",
        "evaluation_cluster_requirement_met",
        "registered_power_analysis_artifact_verified",
        "registered_runtime_benchmark_artifact_verified",
        "power_runtime_prerequisites_passed",
        "statistical_interval_authorized",
        "primary_statistical_gate_passed",
        "production_bank_authorized",
        "model_execution_authorized",
        "weight_updates_authorized",
        "calibrated_leakage_audit_digest",
    )
    obj = _require_mapping(value, fields, name="calibrated_leakage_report")
    if (
        _require_integer(obj["schema_version"], name="schema_version", minimum=1, maximum=1)
        != CALIBRATED_LEAKAGE_SCHEMA_VERSION
        or obj["report_kind"] != _REPORT_KIND
    ):
        raise CalibratedLeakageV2Error("unknown calibrated leakage schema or kind")
    _require_constant_mapping(
        obj["authorization"],
        _AUTHORIZATION,
        name="report authorization",
    )
    _require_constant_mapping(
        obj["classifier_contract"],
        _classifier_contract_obj(),
        name="report classifier_contract",
    )
    source = _source_binding_from_obj(obj["source_binding"])
    plan_obj = obj["calibration_split_plan"]
    if type(plan_obj) is not dict:
        raise CalibratedLeakageV2Error("embedded split plan must be an object")
    embedded_plan_digest = plan_obj.get("calibration_split_plan_digest")
    plan = calibration_split_plan_v2_from_obj(
        plan_obj,
        expected_digest=_require_sha256(embedded_plan_digest, name="embedded plan digest"),
    )
    config = _config_from_obj(obj["config"])
    views_obj = obj["views"]
    if type(views_obj) is not list:
        raise CalibratedLeakageV2Error("report views must be a list")
    report = CalibratedLeakageAuditReportV2(
        source,
        config,
        plan,
        _require_sha256(obj["calibration_frequency_table_digest"], name="frequency digest"),
        tuple(_view_from_obj(item) for item in views_obj),
    )
    expected_flags = {
        "exact_conditional_surface_balance_passed": report.exact_conditional_surface_balance_passed,
        "full_manifest_surface_bound": False,
        "frozen_manifest_runtime_bridge_verified": False,
        "standalone_parser_block_feature_rederivation_verified": False,
        "exact_block_backed_rebuild_required_for_production": True,
        "calibration_cluster_requirement_met": report.calibration_cluster_requirement_met,
        "evaluation_cluster_requirement_met": report.evaluation_cluster_requirement_met,
        "registered_power_analysis_artifact_verified": False,
        "registered_runtime_benchmark_artifact_verified": False,
        "power_runtime_prerequisites_passed": False,
        "statistical_interval_authorized": False,
        "primary_statistical_gate_passed": False,
        "production_bank_authorized": False,
        "model_execution_authorized": False,
        "weight_updates_authorized": False,
    }
    for name, expected in expected_flags.items():
        if _require_bool(obj[name], name=name) is not expected:
            raise CalibratedLeakageV2Error(f"{name} is inconsistent")
    supplied = _require_sha256(
        obj["calibrated_leakage_audit_digest"], name="calibrated audit digest"
    )
    if supplied != report.digest:
        raise CalibratedLeakageV2Error("calibrated leakage report digest is inconsistent")
    if report.digest != expected_digest:
        raise CalibratedLeakageV2Error("report differs from the externally expected digest")
    return report


def serialize_calibrated_leakage_audit_v2(report: CalibratedLeakageAuditReportV2) -> str:
    if type(report) is not CalibratedLeakageAuditReportV2:
        raise TypeError("report must be a CalibratedLeakageAuditReportV2")
    return _dump_json(report.as_obj()) + "\n"


def parse_calibrated_leakage_audit_v2(
    text: str,
    *,
    expected_digest: str,
) -> CalibratedLeakageAuditReportV2:
    return calibrated_leakage_audit_v2_from_obj(_load_json(text), expected_digest=expected_digest)
