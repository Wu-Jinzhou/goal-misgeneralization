"""Fail-closed H200 profile qualification evidence validation.

The freeze-bound actual-model producer emits complete per-process and paired
comparison receipts.  The helpers here independently replay those records,
recompute every aggregate and threshold decision, and construct a
self-digested pre-ITT report.  The report is a selection candidate only; the
authenticated execution freeze performs selection.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import itertools
import json
import math
import os
import re
import struct
import uuid
from collections.abc import Iterable, Mapping, Sequence
from contextlib import suppress
from dataclasses import dataclass
from numbers import Real
from pathlib import Path
from typing import Any, cast

QUALIFICATION_SCHEMA = "goalzendo.g00f_h200_profile_qualification"
QUALIFICATION_SCHEMA_VERSION = 1
EVIDENCE_SCHEMA = "goalzendo.g00f_h200_profile_qualification_evidence"
EVIDENCE_SCHEMA_VERSION = 1
PROCESS_SCHEMA = "goalzendo.g00f_h200_profile_qualification_process"
PROCESS_SCHEMA_VERSION = 1
COMPARISON_SCHEMA = "goalzendo.g00f_h200_profile_qualification_comparison"
COMPARISON_SCHEMA_VERSION = 1

PROFILES = ("baseline", "tuned")
REPLICATES: Mapping[str, tuple[str, ...]] = {
    "baseline": ("primary", "replay"),
    "tuned": ("primary", "replay"),
}
PANELS: Mapping[str, Mapping[str, str]] = {
    "g00f-0p5b": {
        "model_name": "Qwen/Qwen2.5-0.5B-Instruct",
        "model_revision": "7ae557604adf67be50417f59c2c2f167def9a775",
    },
    "g00f-1p5b": {
        "model_name": "Qwen/Qwen2.5-1.5B-Instruct",
        "model_revision": "989aa7980e4cf806f80c7fef2b1adb7bc71aa306",
    },
}
TRAINABLE_NUMEL: Mapping[str, int] = {
    "g00f-0p5b": 494_032_768,
    "g00f-1p5b": 1_543_714_304,
}
LAW_FAMILIES = ("majority", "parity")
TRAINING_VIEWS = (
    "audit_law_matched",
    "herald_only",
    "law_only",
    "no_signal",
    "sage_only",
    "surface_only",
)
UPDATES = tuple(range(1, 9))
REGISTERED_PARITY_UPDATE_CHECKPOINTS = (1, 2, 4, 8)
REGISTERED_PARITY_STATES = (
    "initial",
    "after_update_1",
    "after_update_2",
    "after_update_4",
    "after_update_8",
)
ACTION_LABELS = ("A", "B")
DEVICE_COUNT = 4
PROCESS_RECORD_COUNT = 64
COMPARISON_RECORD_COUNT = 4
BASELINE_FALLBACK_PROCESS_RECORD_COUNT = 32
BASELINE_FALLBACK_COMPARISON_RECORD_COUNT = 0
EVALUATION_BOUNDARY_COUNT = 13
PRODUCTION_UPDATES = 1_000
RUNS_PER_WORKER = 40
RUNS_PER_PANEL_PER_WORKER = 20
PROJECTION_SAFETY_MULTIPLIER = 1.20
MAXIMUM_PROJECTED_WALL_SECONDS = 12 * 60 * 60
MINIMUM_THROUGHPUT_RATIO = 1.20
MAXIMUM_TUNED_RESERVED_BYTES = 120 * 1024**3
MAXIMUM_QUALIFICATION_EVIDENCE_BYTES = 512 * 1024**2
MAXIMUM_ACCUMULATOR_ROWS_PER_VECTOR = 4_096

MAXIMUM_SCORE_DIFFERENCE = 0.002
MAXIMUM_PROBABILITY_DIFFERENCE = 0.005
MINIMUM_GRADIENT_COSINE = 0.99999
MAXIMUM_GRADIENT_RELATIVE_L2 = 0.005
MINIMUM_PARAMETER_UPDATE_COSINE = 0.99999
MAXIMUM_PARAMETER_UPDATE_RELATIVE_L2 = 0.005

PROFILE_CONTRACT: Mapping[str, Mapping[str, Any]] = {
    "baseline": {
        "train_batch_size": 10,
        "gradient_accumulation_steps": 5,
        "effective_batch_size": 50,
        "gradient_checkpointing": True,
        "evaluation_batch_size": 16,
    },
    "tuned": {
        "train_batch_size": 50,
        "gradient_accumulation_steps": 1,
        "effective_batch_size": 50,
        "gradient_checkpointing": False,
        "evaluation_batch_size": 128,
    },
}
VECTOR_KINDS = (
    "raw_preclip_gradient",
    "postclip_gradient",
    "parameter_delta",
    "optimizer_first_moment",
    "optimizer_second_moment",
)
COMPARISON_KINDS = ("baseline_vs_tuned",)
QUALIFICATION_BRANCHES = (
    "tuned_probe_passed_full",
    "tuned_capacity_fallback_baseline",
)
TUNED_CAPACITY_PROBE_SCHEMA = "goalzendo.g00f_h200_tuned_capacity_probe"
TUNED_CAPACITY_PROBE_SCHEMA_VERSION = 1
TUNED_CAPACITY_DISQUALIFIERS = (
    "cuda_out_of_memory",
    "nonfinite_capacity_execution",
    "reserved_memory_ceiling_exceeded",
)
ORDER_ALGORITHM = "deterministic_effective_50_replace_final_with_corpus_max_if_absent_v1"
QUALIFICATION_ENGINEERING_SEED = 8_611_107
GPU_UUID_PATTERN = re.compile(r"^GPU-[A-Za-z0-9][A-Za-z0-9-]{7,}$")
ENGINEERING_ROOT_PREFIX = "g00f-h200-profile-qualification-"
EVIDENCE_BOUNDARY = {
    "kind": "authenticated_actual_model_producer",
    "authentication": "freeze_bound_controller_and_worker_receipts",
    "aggregator_role": "independent_replay_and_aggregation",
}


class QualificationError(RuntimeError):
    """Raised when qualification evidence is incomplete or internally invalid."""


def canonical_json_bytes(value: Any) -> bytes:
    """Serialize finite JSON deterministically."""

    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise QualificationError("qualification payload is not finite canonical JSON") from error


def semantic_digest(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise QualificationError(message)


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise QualificationError(f"{label} must be an object")
    return value


def _sequence(value: Any, label: str) -> Sequence[Any]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise QualificationError(f"{label} must be an array")
    return value


def _exact_keys(value: Mapping[str, Any], expected: Iterable[str], label: str) -> None:
    expected_set = set(expected)
    observed = set(value)
    if observed != expected_set:
        missing = sorted(expected_set - observed)
        extra = sorted(observed - expected_set)
        raise QualificationError(f"{label} keys differ; missing={missing}, extra={extra}")


def _boolean(value: Any, label: str) -> bool:
    if type(value) is not bool:
        raise QualificationError(f"{label} must be Boolean")
    return bool(value)


def _integer(value: Any, label: str, *, minimum: int = 0) -> int:
    if type(value) is not int or int(value) < minimum:
        raise QualificationError(f"{label} must be an integer >= {minimum}")
    return int(value)


def _finite(value: Any, label: str, *, minimum: float | None = None) -> float:
    if isinstance(value, bool):
        raise QualificationError(f"{label} must be finite numeric data")
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise QualificationError(f"{label} must be finite numeric data") from error
    if not math.isfinite(result) or (minimum is not None and result < minimum):
        suffix = "" if minimum is None else f" >= {minimum}"
        raise QualificationError(f"{label} must be finite{suffix}")
    return result


def _sha256(value: Any, label: str) -> str:
    normalized = str(value)
    if len(normalized) != 64 or any(character not in "0123456789abcdef" for character in normalized):
        raise QualificationError(f"{label} must be a lowercase SHA-256 digest")
    return normalized


def _same_number(observed: Any, expected: float, label: str) -> None:
    value = _finite(observed, label)
    if not math.isclose(value, expected, rel_tol=1e-10, abs_tol=1e-12):
        raise QualificationError(f"{label} is inconsistent with recomputed evidence")


def _verify_self_digest(payload: Mapping[str, Any], field: str, label: str) -> None:
    digest = _sha256(payload.get(field), f"{label} digest")
    body = {key: value for key, value in payload.items() if key != field}
    if digest != semantic_digest(body):
        raise QualificationError(f"{label} self-digest mismatch")


def _seal(body: Mapping[str, Any], field: str) -> dict[str, Any]:
    if field in body:
        raise QualificationError(f"unsealed body already contains {field}")
    result = copy.deepcopy(dict(body))
    result[field] = semantic_digest(result)
    return result


def seal_process_record(body: Mapping[str, Any]) -> dict[str, Any]:
    """Attach a deterministic digest to an authenticated producer process body."""

    return _seal(body, "record_digest")


def seal_comparison_record(body: Mapping[str, Any]) -> dict[str, Any]:
    """Attach a deterministic digest to an authenticated paired comparison body."""

    return _seal(body, "comparison_digest")


def seal_evidence(body: Mapping[str, Any]) -> dict[str, Any]:
    """Attach the evidence self-digest after record-manifest construction."""

    return _seal(body, "evidence_digest")


def seal_tuned_capacity_probe(body: Mapping[str, Any]) -> dict[str, Any]:
    """Seal a producer-authenticated four-device tuned capacity probe."""

    return _seal(body, "probe_digest")


class _CompensatedSum:
    """Streaming Neumaier accumulation using Python's IEEE-754 float64."""

    def __init__(self) -> None:
        self.total = 0.0
        self.correction = 0.0

    def add(self, value: float) -> None:
        tentative = self.total + value
        if abs(self.total) >= abs(value):
            self.correction += (self.total - tentative) + value
        else:
            self.correction += (value - tentative) + self.total
        self.total = tentative

    def value(self) -> float:
        result = self.total + self.correction
        if not math.isfinite(result):
            raise QualificationError("float64 streaming accumulation became non-finite")
        return result


def _iter_float64(value: Any, label: str) -> Iterable[float]:
    if isinstance(value, bool):
        raise QualificationError(f"{label} contains Boolean data")
    if isinstance(value, Real):
        number = float(value)
        if not math.isfinite(number):
            raise QualificationError(f"{label} contains non-finite data")
        yield number
        return
    if isinstance(value, (Mapping, str, bytes, bytearray)):
        raise QualificationError(f"{label} must contain only nested numeric iterables")
    try:
        iterator = iter(value)
    except TypeError:
        try:
            number = float(value)
        except (TypeError, ValueError) as error:
            raise QualificationError(f"{label} contains non-numeric data") from error
        if not math.isfinite(number):
            raise QualificationError(f"{label} contains non-finite data") from None
        yield number
        return
    for item in iterator:
        yield from _iter_float64(item, label)


def streaming_vector_metrics(
    left: Mapping[str, Any],
    right: Mapping[str, Any],
) -> dict[str, Any]:
    """Compare named vectors in sorted-key order without concatenating them.

    Values may be numeric scalars or nested/generator-backed numeric iterables.
    This generic helper treats each named input as a canonical float64 tensor.
    It scans each contiguous byte stream once, binds its dtype and flattened
    shape into a native-tensor digest, and feeds converted values to float64
    norm/dot accumulators.  A model producer uses the same digest convention
    with each parameter's actual native dtype and shape.
    """

    if not isinstance(left, Mapping) or not isinstance(right, Mapping) or not left:
        raise QualificationError("streaming vector inputs must be non-empty mappings")
    left_keys = set(left)
    right_keys = set(right)
    if left_keys != right_keys or any(not isinstance(key, str) or not key for key in left_keys):
        raise QualificationError("streaming vector parameter keys must be equal non-empty strings")

    total_count = 0
    sentinel = object()
    accumulator_rows: list[dict[str, Any]] = []

    for order_index, key in enumerate(sorted(left_keys)):
        left_bytes_hasher = hashlib.sha256()
        right_bytes_hasher = hashlib.sha256()
        row_left_square = _CompensatedSum()
        row_right_square = _CompensatedSum()
        row_dot = _CompensatedSum()
        row_difference_square = _CompensatedSum()
        key_count = 0
        left_values = _iter_float64(left[key], f"left vector {key}")
        right_values = _iter_float64(right[key], f"right vector {key}")
        for left_value, right_value in itertools.zip_longest(
            left_values,
            right_values,
            fillvalue=sentinel,
        ):
            if left_value is sentinel or right_value is sentinel:
                raise QualificationError(f"vector length differs for parameter {key}")
            lhs = cast(float, left_value)
            rhs = cast(float, right_value)
            for hasher, number in ((left_bytes_hasher, lhs), (right_bytes_hasher, rhs)):
                hasher.update(struct.pack(">d", number))
            products = (lhs * lhs, rhs * rhs, lhs * rhs, (lhs - rhs) * (lhs - rhs))
            if not all(math.isfinite(item) for item in products):
                raise QualificationError("float64 vector product became non-finite")
            row_left_square.add(products[0])
            row_right_square.add(products[1])
            row_dot.add(products[2])
            row_difference_square.add(products[3])
            key_count += 1
            total_count += 1
        _require(key_count > 0, f"vector parameter {key} contains no scalar elements")
        accumulator_rows.append(
            {
                "order_index": order_index,
                "parameter_key": key,
                "element_count": key_count,
                "finite": True,
                "left_squared_sum": row_left_square.value(),
                "right_squared_sum": row_right_square.value(),
                "difference_squared_sum": row_difference_square.value(),
                "dot": row_dot.value(),
                "left_native_tensor_sha256": semantic_digest(
                    {
                        "dtype": "float64",
                        "shape": [key_count],
                        "contiguous_bytes_sha256": left_bytes_hasher.hexdigest(),
                    }
                ),
                "right_native_tensor_sha256": semantic_digest(
                    {
                        "dtype": "float64",
                        "shape": [key_count],
                        "contiguous_bytes_sha256": right_bytes_hasher.hexdigest(),
                    }
                ),
            }
        )
    if total_count < 1:
        raise QualificationError("streaming vectors contain no scalar elements")

    return streaming_metric_from_accumulator_rows(accumulator_rows)


def streaming_metric_from_accumulator_rows(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Construct an exact metric from bounded native-chunk accumulator rows.

    This is the producer-facing path for safetensors chunks: the producer scans
    native bytes once for each digest while accumulating only float64 scalar
    sums, then discards the temporary chunk before passing its row here.
    """

    normalized_rows = copy.deepcopy(list(_sequence(rows, "accumulator rows")))
    _require(
        1 <= len(normalized_rows) <= MAXIMUM_ACCUMULATOR_ROWS_PER_VECTOR,
        "accumulator row count exceeds the frozen bound",
    )
    left_square = _CompensatedSum()
    right_square = _CompensatedSum()
    dot_sum = _CompensatedSum()
    difference_square = _CompensatedSum()
    parameter_manifest: list[dict[str, Any]] = []
    left_chunk_manifest: list[dict[str, Any]] = []
    right_chunk_manifest: list[dict[str, Any]] = []
    total_count = 0
    previous_key: str | None = None
    for expected_index, raw_row in enumerate(normalized_rows):
        row = _mapping(raw_row, f"accumulator row {expected_index}")
        _exact_keys(
            row,
            {
                "order_index",
                "parameter_key",
                "element_count",
                "finite",
                "left_squared_sum",
                "right_squared_sum",
                "difference_squared_sum",
                "dot",
                "left_native_tensor_sha256",
                "right_native_tensor_sha256",
            },
            f"accumulator row {expected_index}",
        )
        _require(row["order_index"] == expected_index, "accumulator row order changed")
        parameter_key = str(row["parameter_key"])
        _require(bool(parameter_key), "accumulator parameter key is empty")
        if previous_key is not None:
            _require(previous_key < parameter_key, "accumulator keys are not strictly sorted")
        previous_key = parameter_key
        count = _integer(row["element_count"], "accumulator row element count", minimum=1)
        _require(_boolean(row["finite"], "accumulator row finite flag"), "accumulator row is not finite")
        left_squared = _finite(row["left_squared_sum"], "accumulator left squared sum", minimum=0.0)
        right_squared = _finite(
            row["right_squared_sum"],
            "accumulator right squared sum",
            minimum=0.0,
        )
        difference_squared = _finite(
            row["difference_squared_sum"],
            "accumulator difference squared sum",
            minimum=0.0,
        )
        row_dot = _finite(row["dot"], "accumulator row dot")
        left_native = _sha256(row["left_native_tensor_sha256"], "left native-tensor digest")
        right_native = _sha256(row["right_native_tensor_sha256"], "right native-tensor digest")
        row_scale = max(1.0, left_squared + right_squared)
        expected_row_difference = max(0.0, left_squared + right_squared - 2.0 * row_dot)
        _require(
            abs(difference_squared - expected_row_difference) <= 1e-8 * row_scale,
            "accumulator row sums and dot are inconsistent",
        )
        left_square.add(left_squared)
        right_square.add(right_squared)
        dot_sum.add(row_dot)
        difference_square.add(difference_squared)
        total_count += count
        binding = {
            "order_index": expected_index,
            "parameter_key": parameter_key,
            "element_count": count,
        }
        parameter_manifest.append(binding)
        left_chunk_manifest.append({**binding, "native_tensor_sha256": left_native})
        right_chunk_manifest.append({**binding, "native_tensor_sha256": right_native})
    left_norm = math.sqrt(max(0.0, left_square.value()))
    right_norm = math.sqrt(max(0.0, right_square.value()))
    dot_value = dot_sum.value()
    difference_norm = math.sqrt(max(0.0, difference_square.value()))
    if left_norm == 0.0 and right_norm == 0.0:
        zero_case, cosine, relative_l2 = "both_zero", 1.0, 0.0
    elif left_norm == 0.0:
        zero_case, cosine, relative_l2 = "left_zero", 0.0, 1.0
    elif right_norm == 0.0:
        zero_case, cosine, relative_l2 = "right_zero", 0.0, 1.0
    else:
        zero_case = "neither_zero"
        cosine = max(-1.0, min(1.0, dot_value / (left_norm * right_norm)))
        relative_l2 = difference_norm / max(left_norm, right_norm)
    metric = {
        "accumulator_dtype": "float64",
        "parameter_iteration_order": "sorted_parameter_keys",
        "parameter_keys_sha256": semantic_digest(parameter_manifest),
        "left_native_chunk_manifest_sha256": semantic_digest(left_chunk_manifest),
        "right_native_chunk_manifest_sha256": semantic_digest(right_chunk_manifest),
        "element_count": total_count,
        "left_norm": left_norm,
        "right_norm": right_norm,
        "dot": dot_value,
        "difference_norm": difference_norm,
        "zero_norm_case": zero_case,
        "cosine": cosine,
        "relative_l2": relative_l2,
        "accumulator_rows": normalized_rows,
        "accumulator_rows_sha256": semantic_digest(normalized_rows),
    }
    _validate_vector_metric(metric, "constructed streaming metric")
    return metric


def _validate_vector_metric(value: Any, label: str) -> Mapping[str, Any]:
    metric = _mapping(value, label)
    _exact_keys(
        metric,
        {
            "accumulator_dtype",
            "parameter_iteration_order",
            "parameter_keys_sha256",
            "left_native_chunk_manifest_sha256",
            "right_native_chunk_manifest_sha256",
            "element_count",
            "left_norm",
            "right_norm",
            "dot",
            "difference_norm",
            "zero_norm_case",
            "cosine",
            "relative_l2",
            "accumulator_rows",
            "accumulator_rows_sha256",
        },
        label,
    )
    _require(metric["accumulator_dtype"] == "float64", f"{label} must use float64 accumulation")
    _require(
        metric["parameter_iteration_order"] == "sorted_parameter_keys",
        f"{label} must use sorted parameter keys",
    )
    rows = _sequence(metric["accumulator_rows"], f"{label} accumulator rows")
    _require(
        1 <= len(rows) <= MAXIMUM_ACCUMULATOR_ROWS_PER_VECTOR,
        f"{label} accumulator row count exceeds the frozen bound",
    )
    _require(
        metric["accumulator_rows_sha256"] == semantic_digest(list(rows)),
        f"{label} accumulator-row digest mismatch",
    )
    left_square = _CompensatedSum()
    right_square = _CompensatedSum()
    dot_sum = _CompensatedSum()
    difference_square = _CompensatedSum()
    parameter_manifest: list[dict[str, Any]] = []
    left_chunk_manifest: list[dict[str, Any]] = []
    right_chunk_manifest: list[dict[str, Any]] = []
    total_count = 0
    previous_key: str | None = None
    for expected_index, raw_row in enumerate(rows):
        row = _mapping(raw_row, f"{label} accumulator row {expected_index}")
        _exact_keys(
            row,
            {
                "order_index",
                "parameter_key",
                "element_count",
                "finite",
                "left_squared_sum",
                "right_squared_sum",
                "difference_squared_sum",
                "dot",
                "left_native_tensor_sha256",
                "right_native_tensor_sha256",
            },
            f"{label} accumulator row {expected_index}",
        )
        _require(row["order_index"] == expected_index, f"{label} accumulator row order changed")
        parameter_key = str(row["parameter_key"])
        _require(bool(parameter_key), f"{label} accumulator parameter key is empty")
        if previous_key is not None:
            _require(previous_key < parameter_key, f"{label} accumulator keys are not strictly sorted")
        previous_key = parameter_key
        count = _integer(
            row["element_count"],
            f"{label} accumulator row element count",
            minimum=1,
        )
        _require(
            _boolean(row["finite"], f"{label} accumulator row finite flag"),
            f"{label} accumulator row is not finite",
        )
        left_squared = _finite(
            row["left_squared_sum"],
            f"{label} accumulator row left squared sum",
            minimum=0.0,
        )
        right_squared = _finite(
            row["right_squared_sum"],
            f"{label} accumulator row right squared sum",
            minimum=0.0,
        )
        difference_squared = _finite(
            row["difference_squared_sum"],
            f"{label} accumulator row difference squared sum",
            minimum=0.0,
        )
        row_dot = _finite(row["dot"], f"{label} accumulator row dot")
        left_chunk = _sha256(
            row["left_native_tensor_sha256"],
            f"{label} accumulator row left chunk digest",
        )
        right_chunk = _sha256(
            row["right_native_tensor_sha256"],
            f"{label} accumulator row right chunk digest",
        )
        row_scale = max(1.0, left_squared + right_squared)
        expected_row_difference = max(0.0, left_squared + right_squared - 2.0 * row_dot)
        _require(
            abs(difference_squared - expected_row_difference) <= 1e-8 * row_scale,
            f"{label} accumulator row sums and dot are inconsistent",
        )
        left_square.add(left_squared)
        right_square.add(right_squared)
        dot_sum.add(row_dot)
        difference_square.add(difference_squared)
        total_count += count
        binding = {
            "order_index": expected_index,
            "parameter_key": parameter_key,
            "element_count": count,
        }
        parameter_manifest.append(binding)
        left_chunk_manifest.append({**binding, "native_tensor_sha256": left_chunk})
        right_chunk_manifest.append({**binding, "native_tensor_sha256": right_chunk})
    _require(
        metric["parameter_keys_sha256"] == semantic_digest(parameter_manifest),
        f"{label} parameter-key digest is not derived from accumulator rows",
    )
    _require(
        metric["left_native_chunk_manifest_sha256"] == semantic_digest(left_chunk_manifest),
        f"{label} left vector digest is not derived from accumulator rows",
    )
    _require(
        metric["right_native_chunk_manifest_sha256"] == semantic_digest(right_chunk_manifest),
        f"{label} right vector digest is not derived from accumulator rows",
    )
    _require(metric["element_count"] == total_count, f"{label} element count mismatch")
    recomputed_left_norm = math.sqrt(max(0.0, left_square.value()))
    recomputed_right_norm = math.sqrt(max(0.0, right_square.value()))
    recomputed_dot = dot_sum.value()
    recomputed_difference = math.sqrt(max(0.0, difference_square.value()))
    left_norm = _finite(metric["left_norm"], f"{label} left norm", minimum=0.0)
    right_norm = _finite(metric["right_norm"], f"{label} right norm", minimum=0.0)
    dot = _finite(metric["dot"], f"{label} dot")
    difference = _finite(metric["difference_norm"], f"{label} difference norm", minimum=0.0)
    _same_number(left_norm, recomputed_left_norm, f"{label} left norm")
    _same_number(right_norm, recomputed_right_norm, f"{label} right norm")
    _same_number(dot, recomputed_dot, f"{label} dot")
    _same_number(difference, recomputed_difference, f"{label} difference norm")
    cosine = _finite(metric["cosine"], f"{label} cosine")
    relative = _finite(metric["relative_l2"], f"{label} relative L2", minimum=0.0)
    _require(-1.0 <= cosine <= 1.0, f"{label} cosine must lie in [-1,1]")

    if left_norm == 0.0 and right_norm == 0.0:
        expected_case, expected_cosine, expected_relative = "both_zero", 1.0, 0.0
    elif left_norm == 0.0:
        expected_case, expected_cosine, expected_relative = "left_zero", 0.0, 1.0
    elif right_norm == 0.0:
        expected_case, expected_cosine, expected_relative = "right_zero", 0.0, 1.0
    else:
        expected_case = "neither_zero"
        expected_cosine = max(-1.0, min(1.0, dot / (left_norm * right_norm)))
        expected_relative = difference / max(left_norm, right_norm)
        cauchy_tolerance = 1e-10 * max(1.0, left_norm * right_norm)
        _require(
            abs(dot) <= left_norm * right_norm + cauchy_tolerance,
            f"{label} dot exceeds the Cauchy bound",
        )
    _require(metric["zero_norm_case"] == expected_case, f"{label} zero-norm case is inconsistent")
    _same_number(cosine, expected_cosine, f"{label} cosine")
    _same_number(relative, expected_relative, f"{label} relative L2")
    expected_difference_square = max(0.0, left_norm**2 + right_norm**2 - 2.0 * dot)
    consistency_scale = max(1.0, left_norm**2 + right_norm**2)
    _require(
        abs(difference**2 - expected_difference_square) <= 1e-8 * consistency_scale,
        f"{label} norms, dot, and difference are inconsistent",
    )
    return metric


def _engineering_scope(value: Any, label: str) -> Mapping[str, Any]:
    scope = _mapping(value, label)
    _exact_keys(
        scope,
        {
            "root",
            "root_class",
            "data_class",
            "outcomes_seen",
            "itt_ledger_created",
            "production_artifacts_read",
            "production_artifacts_written",
        },
        label,
    )
    root = Path(str(scope["root"]))
    _require(root.is_absolute(), f"{label} root must be absolute")
    _require(
        root.name.startswith(ENGINEERING_ROOT_PREFIX),
        f"{label} root must be a dedicated H200 qualification root",
    )
    _require(scope["root_class"] == "engineering_only", f"{label} root class is not engineering-only")
    _require(
        scope["data_class"] == "synthetic_engineering_only",
        f"{label} data class is not synthetic engineering data",
    )
    _require(not _boolean(scope["outcomes_seen"], f"{label} outcomes flag"), f"{label} saw outcomes")
    _require(
        not _boolean(scope["itt_ledger_created"], f"{label} ITT flag"),
        f"{label} was recorded after ITT creation",
    )
    _require(
        not _boolean(scope["production_artifacts_read"], f"{label} production-read flag"),
        f"{label} read production artifacts",
    )
    _require(
        not _boolean(scope["production_artifacts_written"], f"{label} production-write flag"),
        f"{label} wrote production artifacts",
    )
    return scope


def _validate_bindings(evidence: Mapping[str, Any]) -> None:
    freeze = _mapping(evidence["freeze_binding"], "freeze binding")
    _exact_keys(freeze, {"freeze_file_sha256", "freeze_digest"}, "freeze binding")
    _sha256(freeze["freeze_file_sha256"], "freeze file digest")
    _sha256(freeze["freeze_digest"], "freeze semantic digest")

    provision = _mapping(evidence["provision_binding"], "provision binding")
    _exact_keys(
        provision,
        {"file_sha256", "receipt_digest", "pod_id"},
        "provision binding",
    )
    _sha256(provision["file_sha256"], "provision receipt file digest")
    _sha256(provision["receipt_digest"], "provision receipt semantic digest")
    _require(
        isinstance(provision["pod_id"], str) and bool(provision["pod_id"]),
        "provision binding pod ID is empty",
    )

    model_receipts = _mapping(evidence["model_receipt_bindings"], "model receipt bindings")
    _exact_keys(model_receipts, PANELS, "model receipt bindings")
    integration_audits = _mapping(
        evidence["model_integration_audit_bindings"],
        "model integration audit bindings",
    )
    _exact_keys(integration_audits, PANELS, "model integration audit bindings")
    for panel in PANELS:
        model = _mapping(model_receipts[panel], f"{panel} model receipt binding")
        _exact_keys(
            model,
            {"panel_id", "file_sha256", "receipt_digest"},
            f"{panel} model receipt binding",
        )
        _require(model["panel_id"] == panel, f"{panel} model receipt panel mismatch")
        _sha256(model["file_sha256"], f"{panel} model receipt file digest")
        _sha256(model["receipt_digest"], f"{panel} model receipt semantic digest")
        audit = _mapping(integration_audits[panel], f"{panel} integration audit binding")
        _exact_keys(
            audit,
            {"panel_id", "file_sha256", "audit_digest", "report_digest"},
            f"{panel} integration audit binding",
        )
        _require(audit["panel_id"] == panel, f"{panel} integration audit panel mismatch")
        _sha256(audit["file_sha256"], f"{panel} integration audit file digest")
        _sha256(audit["audit_digest"], f"{panel} integration audit semantic digest")
        _sha256(audit["report_digest"], f"{panel} integration report digest")


def _validate_storage_contract(
    value: Any,
    *,
    minimum_inline_bytes: int | None = None,
    accumulator_row_count: int | None = None,
) -> Mapping[str, Any]:
    storage = _mapping(value, "qualification evidence storage contract")
    _exact_keys(
        storage,
        {
            "maximum_persisted_evidence_bytes",
            "observed_persisted_evidence_bytes",
            "accumulator_row_count",
            "raw_vector_bytes_persisted",
            "vector_persistence",
            "transient_raw_cleanup_before_itt_required",
            "transient_raw_cleanup_receipt_required",
            "transient_raw_cleanup_timing",
            "compact_evidence_retained_through_final_gate",
        },
        "qualification evidence storage contract",
    )
    _require(
        storage["maximum_persisted_evidence_bytes"] == MAXIMUM_QUALIFICATION_EVIDENCE_BYTES,
        "qualification evidence storage ceiling changed",
    )
    observed = _integer(
        storage["observed_persisted_evidence_bytes"],
        "observed qualification evidence bytes",
        minimum=1,
    )
    _require(
        observed <= MAXIMUM_QUALIFICATION_EVIDENCE_BYTES,
        "qualification evidence exceeds the frozen storage ceiling",
    )
    if minimum_inline_bytes is not None:
        _require(
            observed >= minimum_inline_bytes,
            "observed qualification evidence bytes are smaller than the inline evidence",
        )
    rows = _integer(storage["accumulator_row_count"], "qualification accumulator row count")
    if accumulator_row_count is not None:
        _require(rows == accumulator_row_count, "qualification accumulator row count mismatch")
    _require(storage["raw_vector_bytes_persisted"] == 0, "raw qualification vectors were persisted")
    _require(
        storage["vector_persistence"] == "bounded_inline_per_parameter_accumulator_rows_only",
        "qualification vector persistence mode changed",
    )
    _require(
        _boolean(
            storage["transient_raw_cleanup_before_itt_required"],
            "qualification transient-raw cleanup-before-ITT flag",
        ),
        "qualification transient raw cleanup is not required before ITT",
    )
    _require(
        _boolean(
            storage["transient_raw_cleanup_receipt_required"],
            "qualification transient-raw cleanup receipt flag",
        ),
        "qualification transient raw cleanup receipt is not required",
    )
    _require(
        storage["transient_raw_cleanup_timing"] == "during_producer_before_compact_evidence_seal",
        "qualification transient raw cleanup timing changed",
    )
    _require(
        _boolean(
            storage["compact_evidence_retained_through_final_gate"],
            "qualification compact-evidence retention flag",
        ),
        "qualification compact evidence is not retained through the final gate",
    )
    return storage


def _validate_output(value: Any, label: str) -> Mapping[str, Any]:
    output = _mapping(value, label)
    _exact_keys(
        output,
        {
            "view",
            "sample_id",
            "token_length",
            "worst_case_token_sample",
            "normalized_log_scores",
            "probabilities",
            "action_index",
            "action_label",
        },
        label,
    )
    _require(output["view"] in TRAINING_VIEWS, f"{label} has an unknown view")
    _require(
        isinstance(output["sample_id"], str) and bool(output["sample_id"]), f"{label} sample ID is empty"
    )
    _integer(output["token_length"], f"{label} token length", minimum=1)
    _boolean(output["worst_case_token_sample"], f"{label} worst-case flag")
    scores = _sequence(output["normalized_log_scores"], f"{label} normalized scores")
    probabilities = _sequence(output["probabilities"], f"{label} probabilities")
    _require(len(scores) == len(probabilities) == 2, f"{label} must contain exactly two actions")
    normalized_scores = [_finite(item, f"{label} normalized score") for item in scores]
    _require(
        all(item <= 0.0 for item in normalized_scores),
        f"{label} normalized log scores must be non-positive",
    )
    normalized_probabilities = [_finite(item, f"{label} probability", minimum=0.0) for item in probabilities]
    _require(
        math.isclose(sum(normalized_probabilities), 1.0, rel_tol=0.0, abs_tol=1e-9),
        f"{label} probabilities are not normalized",
    )
    score_probabilities = [math.exp(item) for item in normalized_scores]
    _require(
        math.isclose(sum(score_probabilities), 1.0, rel_tol=0.0, abs_tol=1e-8),
        f"{label} normalized log scores do not sum to one in probability space",
    )
    for index in range(2):
        _require(
            math.isclose(
                normalized_probabilities[index],
                score_probabilities[index],
                rel_tol=0.0,
                abs_tol=1e-8,
            ),
            f"{label} score/probability pair is inconsistent",
        )
    action_index = _integer(output["action_index"], f"{label} action index")
    _require(action_index in (0, 1), f"{label} action index must be zero or one")
    expected_index = 0 if normalized_probabilities[0] >= normalized_probabilities[1] else 1
    _require(action_index == expected_index, f"{label} action is not the deterministic argmax")
    _require(output["action_label"] == ACTION_LABELS[action_index], f"{label} action label is inconsistent")
    return output


def _validate_vector_descriptor(value: Any, label: str) -> Mapping[str, Any]:
    descriptor = _mapping(value, label)
    _exact_keys(
        descriptor,
        {
            "parameter_keys_sha256",
            "native_chunk_manifest_sha256",
            "element_count",
            "float64_norm",
            "chunk_count",
            "trainable_parameter_manifest_sha256",
        },
        label,
    )
    _sha256(descriptor["parameter_keys_sha256"], f"{label} parameter-key digest")
    _sha256(
        descriptor["native_chunk_manifest_sha256"],
        f"{label} native chunk-manifest digest",
    )
    _integer(descriptor["element_count"], f"{label} element count", minimum=1)
    _finite(descriptor["float64_norm"], f"{label} norm", minimum=0.0)
    _integer(descriptor["chunk_count"], f"{label} chunk count", minimum=1)
    _sha256(
        descriptor["trainable_parameter_manifest_sha256"],
        f"{label} trainable-parameter manifest digest",
    )
    return descriptor


def _validate_trainable_parameter_manifest(value: Any, panel_id: str) -> Mapping[str, Any]:
    manifest = _mapping(value, "trainable-parameter manifest")
    _exact_keys(
        manifest,
        {
            "enumeration_api",
            "requires_grad_only",
            "order",
            "entries",
            "trainable_numel",
            "parameter_keys_sha256",
            "manifest_sha256",
        },
        "trainable-parameter manifest",
    )
    _require(
        manifest["enumeration_api"] == "named_parameters(remove_duplicate=True)",
        "trainable parameters were not enumerated with remove_duplicate=True",
    )
    _require(
        _boolean(manifest["requires_grad_only"], "trainable requires-grad-only flag"),
        "parameter manifest contains non-trainable parameters",
    )
    _require(manifest["order"] == "sorted_parameter_keys", "trainable parameters are not sorted")
    entries = _sequence(manifest["entries"], "trainable-parameter entries")
    _require(
        1 <= len(entries) <= MAXIMUM_ACCUMULATOR_ROWS_PER_VECTOR,
        "trainable-parameter manifest entry count exceeds the frozen bound",
    )
    normalized_entries: list[dict[str, Any]] = []
    parameter_binding: list[dict[str, Any]] = []
    previous_key: str | None = None
    total_numel = 0
    for order_index, raw_entry in enumerate(entries):
        entry = _mapping(raw_entry, f"trainable parameter {order_index}")
        _exact_keys(
            entry,
            {"parameter_key", "dtype", "shape", "numel"},
            f"trainable parameter {order_index}",
        )
        parameter_key = str(entry["parameter_key"])
        _require(bool(parameter_key), "trainable parameter key is empty")
        if previous_key is not None:
            _require(previous_key < parameter_key, "trainable parameter keys are not strictly sorted")
        previous_key = parameter_key
        _require(
            isinstance(entry["dtype"], str) and bool(entry["dtype"]),
            "trainable parameter dtype is empty",
        )
        shape_raw = _sequence(entry["shape"], f"trainable parameter {parameter_key} shape")
        shape = [
            _integer(item, f"trainable parameter {parameter_key} dimension", minimum=1) for item in shape_raw
        ]
        _require(bool(shape), "trainable parameter shape must have at least one dimension")
        numel = _integer(entry["numel"], f"trainable parameter {parameter_key} numel", minimum=1)
        _require(math.prod(shape) == numel, "trainable parameter shape/numel mismatch")
        normalized_entry = {
            "parameter_key": parameter_key,
            "dtype": entry["dtype"],
            "shape": shape,
            "numel": numel,
        }
        _require(dict(entry) == normalized_entry, "trainable parameter entry is not canonical")
        normalized_entries.append(normalized_entry)
        parameter_binding.append(
            {
                "order_index": order_index,
                "parameter_key": parameter_key,
                "element_count": numel,
            }
        )
        total_numel += numel
    expected_numel = TRAINABLE_NUMEL[panel_id]
    _require(total_numel == expected_numel, f"{panel_id} trainable parameter count mismatch")
    _require(manifest["trainable_numel"] == expected_numel, f"{panel_id} frozen trainable numel changed")
    _require(
        manifest["parameter_keys_sha256"] == semantic_digest(parameter_binding),
        "trainable parameter-key binding digest mismatch",
    )
    _require(
        manifest["manifest_sha256"] == semantic_digest(normalized_entries),
        "trainable-parameter manifest digest mismatch",
    )
    return manifest


def _validate_registered_parity_chunk(
    value: Any,
    *,
    profile: str,
    expected_state: str,
) -> Mapping[str, Any]:
    """Validate one compact 128-example/six-view production-shaped parity chunk."""

    probe = _mapping(value, f"registered parity chunk at {expected_state}")
    _exact_keys(
        probe,
        {
            "measurement_state",
            "after_optimizer_step",
            "prompt_count",
            "example_count",
            "production_example_partition_count",
            "batch_size",
            "batch_count",
            "scorer_call_count",
            "prompt_sha256",
            "normalized_outputs_sha256",
            "rows",
            "rows_sha256",
            "contiguous_example_chunk",
            "prompt_views_per_example",
            "production_bank_name",
            "production_bank_start_index",
            "example_ids",
            "example_ids_sha256",
            "token_shapes",
            "token_shape_sha256",
        },
        f"registered parity chunk at {expected_state}",
    )
    _require(
        probe["measurement_state"] == expected_state
        and _boolean(probe["after_optimizer_step"], "registered parity after-optimizer flag")
        == (expected_state != "initial"),
        "registered parity chunk is bound to the wrong optimizer boundary",
    )
    batch_size = int(PROFILE_CONTRACT[profile]["evaluation_batch_size"])
    _require(
        probe["prompt_count"] == 128 * len(TRAINING_VIEWS)
        and probe["example_count"] == 128
        and probe["production_example_partition_count"] == 1
        and probe["batch_size"] == batch_size,
        "registered parity chunk cardinality, partition, or batch size changed",
    )
    _require(
        _boolean(probe["contiguous_example_chunk"], "registered contiguous-chunk flag")
        and probe["prompt_views_per_example"] == len(TRAINING_VIEWS)
        and probe["production_bank_name"]
        in {
            "validation",
            "diagnostic_factorial",
            "diagnostic_causal",
            "final_factorial",
            "final_causal",
        }
        and type(probe["production_bank_start_index"]) is int
        and int(probe["production_bank_start_index"]) >= 0,
        "registered parity evidence is not a contiguous six-view production chunk",
    )
    example_ids = [str(item) for item in _sequence(probe["example_ids"], "registered parity example IDs")]
    _require(
        len(example_ids) == 128
        and len(set(example_ids)) == 128
        and probe["example_ids_sha256"] == semantic_digest(example_ids),
        "registered parity example IDs/order are incomplete or invalid",
    )
    token_shapes = [
        [
            _integer(length, "registered parity token length", minimum=1)
            for length in _sequence(raw_shape, "registered parity token shape")
        ]
        for raw_shape in _sequence(probe["token_shapes"], "registered parity token shapes")
    ]
    _require(
        len(token_shapes) == 128
        and all(len(shape) == len(TRAINING_VIEWS) for shape in token_shapes)
        and probe["token_shape_sha256"] == semantic_digest(token_shapes),
        "registered parity token shapes are incomplete or invalid",
    )
    _require(
        probe["batch_count"] == math.ceil(128 / batch_size)
        and probe["scorer_call_count"] == probe["batch_count"],
        "registered parity scorer-call partition changed",
    )
    _sha256(probe["prompt_sha256"], "registered parity prompt digest")
    _sha256(probe["normalized_outputs_sha256"], "registered parity output digest")
    rows = _sequence(probe["rows"], "registered parity numeric rows")
    _require(
        len(rows) == probe["prompt_count"] and probe["rows_sha256"] == semantic_digest(list(rows)),
        "registered parity rows are incomplete or have an invalid digest",
    )
    seen_row_ids: set[str] = set()
    for index, raw_row in enumerate(rows):
        row = _mapping(raw_row, f"registered parity numeric row {index}")
        _exact_keys(
            row,
            {"row_id", "normalized_log_scores", "probabilities", "action_index", "action_label"},
            f"registered parity numeric row {index}",
        )
        row_id = str(row["row_id"])
        _require(
            bool(row_id) and row_id not in seen_row_ids,
            "registered parity row ID is empty or duplicated",
        )
        seen_row_ids.add(row_id)
        scores = [
            _finite(item, "registered parity normalized score")
            for item in _sequence(row["normalized_log_scores"], "registered parity normalized scores")
        ]
        probabilities = [
            _finite(item, "registered parity probability", minimum=0.0)
            for item in _sequence(row["probabilities"], "registered parity probabilities")
        ]
        _require(
            len(scores) == len(probabilities) == 2
            and math.isclose(sum(probabilities), 1.0, rel_tol=5e-3, abs_tol=5e-3),
            "registered parity numeric row is not normalized",
        )
        action_index = _integer(row["action_index"], "registered parity action index")
        _require(
            action_index in (0, 1) and row["action_label"] == ACTION_LABELS[action_index],
            "registered parity action binding changed",
        )
    return probe


def _validate_execution_benchmark(value: Any, *, profile: str, law_family: str) -> Mapping[str, Any]:
    benchmark = _mapping(value, "process execution benchmark")
    _exact_keys(
        benchmark,
        {
            "tier",
            "timing_representative",
            "worst_law_proof",
            "data_bank_render_tokenization",
            "mode_transitions",
            "gradient_checkpointing",
            "registered_production_shaped_parity_chunk",
            "boundaries",
            "checkpoint_io",
            "final_seal_io",
            "benchmark_digest",
        },
        "process execution benchmark",
    )
    _verify_self_digest(benchmark, "benchmark_digest", "process execution benchmark")
    tier = str(benchmark["tier"])
    _require(
        tier in {"full_production_timing_representative", "registered_numeric_only"},
        "process execution benchmark tier changed",
    )
    representative = _boolean(benchmark["timing_representative"], "timing representative flag")
    _require(
        representative == (tier == "full_production_timing_representative"),
        "execution benchmark tier/representative flag differs",
    )
    worst_proof = _mapping(benchmark["worst_law_proof"], "worst-Law timing proof")
    _exact_keys(
        worst_proof,
        {
            "selected_law",
            "record_law",
            "record_is_selected_worst",
            "law_max_token_length",
            "other_law_max_token_length",
            "record_padded_token_elements",
            "other_law_padded_token_elements",
            "record_maximum_padded_token_elements_per_call",
            "other_law_maximum_padded_token_elements_per_call",
            "record_maximum_flattened_prompts_per_call",
            "other_law_maximum_flattened_prompts_per_call",
            "record_scorer_call_count",
            "other_law_scorer_call_count",
            "record_call_shapes",
            "other_law_call_shapes",
            "record_call_shapes_sha256",
            "other_law_call_shapes_sha256",
        },
        "worst-Law timing proof",
    )
    _require(
        worst_proof["selected_law"] in LAW_FAMILIES and worst_proof["record_law"] == law_family,
        "worst-Law timing proof identity changed",
    )
    law_max = _integer(worst_proof["law_max_token_length"], "Law maximum token length", minimum=1)
    other_max = _integer(
        worst_proof["other_law_max_token_length"], "other-Law maximum token length", minimum=1
    )
    is_selected = _boolean(worst_proof["record_is_selected_worst"], "selected worst-Law flag")
    _require(
        is_selected == (law_family == worst_proof["selected_law"]),
        "worst-Law selected-record flag is inconsistent",
    )
    if representative:
        record_padded = _integer(
            worst_proof["record_padded_token_elements"],
            "record padded-token workload",
            minimum=1,
        )
        other_padded = _integer(
            worst_proof["other_law_padded_token_elements"],
            "other-Law padded-token workload",
            minimum=1,
        )
        record_max_call = _integer(
            worst_proof["record_maximum_padded_token_elements_per_call"],
            "record maximum padded call",
            minimum=1,
        )
        other_max_call = _integer(
            worst_proof["other_law_maximum_padded_token_elements_per_call"],
            "other-Law maximum padded call",
            minimum=1,
        )
        record_flat = _integer(
            worst_proof["record_maximum_flattened_prompts_per_call"],
            "record maximum flattened prompts",
            minimum=1,
        )
        other_flat = _integer(
            worst_proof["other_law_maximum_flattened_prompts_per_call"],
            "other-Law maximum flattened prompts",
            minimum=1,
        )
        record_calls = _integer(
            worst_proof["record_scorer_call_count"],
            "record scorer-call count",
            minimum=1,
        )
        other_calls = _integer(
            worst_proof["other_law_scorer_call_count"],
            "other-Law scorer-call count",
            minimum=1,
        )

        def validate_call_shapes(raw: Any, digest: Any, label: str) -> list[Mapping[str, Any]]:
            shapes = list(_sequence(raw, f"{label} call shapes"))
            _require(bool(shapes), f"{label} call-shape vector is empty")
            _require(
                digest == semantic_digest(shapes),
                f"{label} call-shape digest mismatch",
            )
            normalized: list[Mapping[str, Any]] = []
            for call_index, raw_shape in enumerate(shapes):
                shape = _mapping(raw_shape, f"{label} call shape {call_index}")
                _exact_keys(
                    shape,
                    {
                        "boundary_ordinal",
                        "kind",
                        "bank_index",
                        "chunk_index",
                        "example_count",
                        "flattened_prompt_count",
                        "max_seq_len",
                        "padded_elements",
                    },
                    f"{label} call shape {call_index}",
                )
                boundary = _integer(shape["boundary_ordinal"], "call-shape boundary")
                _require(boundary < EVALUATION_BOUNDARY_COUNT, "call-shape boundary is invalid")
                _require(
                    shape["kind"] == ("final" if boundary == EVALUATION_BOUNDARY_COUNT - 1 else "diagnostic"),
                    "call-shape final-boundary distinction changed",
                )
                for field in (
                    "bank_index",
                    "chunk_index",
                    "example_count",
                    "flattened_prompt_count",
                    "max_seq_len",
                    "padded_elements",
                ):
                    _integer(
                        shape[field],
                        f"call-shape {field}",
                        minimum=1 if field not in {"bank_index", "chunk_index"} else 0,
                    )
                _require(
                    shape["padded_elements"] == shape["flattened_prompt_count"] * shape["max_seq_len"],
                    "call-shape padded elements are inconsistent",
                )
                normalized.append(shape)
            return normalized

        record_shapes = validate_call_shapes(
            worst_proof["record_call_shapes"],
            worst_proof["record_call_shapes_sha256"],
            "record",
        )
        other_shapes = validate_call_shapes(
            worst_proof["other_law_call_shapes"],
            worst_proof["other_law_call_shapes_sha256"],
            "other-Law",
        )
        for shapes, padded_total, max_call, flat_max, token_max, call_count, label in (
            (record_shapes, record_padded, record_max_call, record_flat, law_max, record_calls, "record"),
            (other_shapes, other_padded, other_max_call, other_flat, other_max, other_calls, "other-Law"),
        ):
            _require(
                len(shapes) == call_count
                and sum(int(shape["padded_elements"]) for shape in shapes) == padded_total
                and max(int(shape["padded_elements"]) for shape in shapes) == max_call
                and max(int(shape["flattened_prompt_count"]) for shape in shapes) == flat_max
                and max(int(shape["max_seq_len"]) for shape in shapes) == token_max,
                f"{label} call-shape aggregates are inconsistent",
            )
        callwise_dominance = len(record_shapes) == len(other_shapes) and all(
            tuple(record[field] for field in ("boundary_ordinal", "kind", "bank_index", "chunk_index"))
            == tuple(other[field] for field in ("boundary_ordinal", "kind", "bank_index", "chunk_index"))
            and all(
                int(record[field]) >= int(other[field])
                for field in (
                    "example_count",
                    "flattened_prompt_count",
                    "max_seq_len",
                    "padded_elements",
                )
            )
            for record, other in zip(record_shapes, other_shapes, strict=True)
        )
        _require(
            is_selected
            and record_padded >= other_padded
            and record_max_call >= other_max_call
            and law_max >= other_max
            and record_flat >= other_flat
            and record_calls >= other_calls
            and callwise_dominance,
            "full timing representative is not the conservative tokenized production workload",
        )
    data = _mapping(benchmark["data_bank_render_tokenization"], "data/bank benchmark")
    _exact_keys(
        data,
        {
            "corpus_size",
            "law_family",
            "six_training_views_rendered",
            "diagnostic_prompt_count",
            "final_prompt_count",
            "diagnostic_bank_example_counts",
            "final_bank_example_counts",
            "diagnostic_prompt_sha256",
            "final_prompt_sha256",
        },
        "data/bank benchmark",
    )
    _require(data["corpus_size"] == 10_000, "qualification corpus is not exactly 10,000 rows")
    _require(data["law_family"] == law_family, "benchmark Law family changed")
    _require(
        _boolean(data["six_training_views_rendered"], "six-view render flag"),
        "qualification did not render all six training views",
    )
    for field in ("diagnostic_prompt_count", "final_prompt_count"):
        _require(_integer(data[field], field, minimum=128) > 128, f"{field} is not a full bank")
    for field in ("diagnostic_bank_example_counts", "final_bank_example_counts"):
        counts = [_integer(item, field, minimum=1) for item in _sequence(data[field], field)]
        _require(len(counts) == 3, f"{field} must bind IID, factorial, and causal banks")
    _sha256(data["diagnostic_prompt_sha256"], "diagnostic prompt-bank digest")
    _sha256(data["final_prompt_sha256"], "final prompt-bank digest")

    transitions = _mapping(benchmark["mode_transitions"], "train/eval transitions")
    _require(
        transitions
        == {
            "scorer_train_for_every_forward_backward": True,
            "eval_only_at_boundaries_and_registered_parity_chunk": True,
            "train_mode_restored_after_every_evaluation": True,
            "dropout_effectively_zero_while_training": True,
        },
        "production train/eval transitions changed",
    )
    checkpointing = _mapping(benchmark["gradient_checkpointing"], "gradient checkpointing audit")
    _require(
        checkpointing
        == {
            "configured": profile == "baseline",
            "runtime_enabled": profile == "baseline",
            "training_graph_exercised": profile == "baseline",
        },
        "gradient-checkpointing execution audit changed",
    )
    _validate_registered_parity_chunk(
        benchmark["registered_production_shaped_parity_chunk"],
        profile=profile,
        expected_state="initial",
    )
    batch_size = int(PROFILE_CONTRACT[profile]["evaluation_batch_size"])

    boundaries = _sequence(benchmark["boundaries"], "evaluation benchmark boundaries")
    _require(
        len(boundaries) == (EVALUATION_BOUNDARY_COUNT if representative else 0),
        "execution benchmark boundary cardinality differs from its tier",
    )
    for ordinal, raw in enumerate(boundaries):
        boundary = _mapping(raw, f"evaluation boundary {ordinal}")
        _exact_keys(
            boundary,
            {
                "ordinal",
                "kind",
                "example_count",
                "prompt_count",
                "batch_size",
                "batch_count",
                "scorer_call_count",
                "prompt_sha256",
                "normalized_outputs_sha256",
                "callback_artifact_sha256",
                "callback_artifact_bytes",
                "callback_artifact_count",
                "callback_metrics_row_count",
                "callback_prediction_row_count",
                "callback_progress_record_count",
                "callback_schema",
            },
            f"evaluation boundary {ordinal}",
        )
        expected_kind = "final" if ordinal == EVALUATION_BOUNDARY_COUNT - 1 else "diagnostic"
        expected_count = data[f"{expected_kind}_prompt_count"]
        expected_digest = data[f"{expected_kind}_prompt_sha256"]
        example_counts = list(data[f"{expected_kind}_bank_example_counts"])
        _require(
            boundary["ordinal"] == ordinal and boundary["kind"] == expected_kind,
            "evaluation boundary order/final distinction changed",
        )
        _require(boundary["prompt_count"] == expected_count, "evaluation boundary count changed")
        _require(
            boundary["example_count"] == sum(example_counts), "evaluation boundary example count changed"
        )
        _require(boundary["batch_size"] == batch_size, "evaluation boundary batch size changed")
        expected_calls = sum(math.ceil(int(count) / batch_size) for count in example_counts)
        _require(
            boundary["batch_count"] == expected_calls and boundary["scorer_call_count"] == expected_calls,
            "evaluation boundary batch count changed",
        )
        _require(boundary["prompt_sha256"] == expected_digest, "evaluation boundary prompt digest changed")
        _sha256(boundary["normalized_outputs_sha256"], "evaluation boundary output digest")
        _sha256(boundary["callback_artifact_sha256"], "evaluation callback artifact digest")
        _integer(boundary["callback_artifact_bytes"], "evaluation callback artifact bytes", minimum=1)
        _require(
            boundary["callback_artifact_count"] == 3
            and boundary["callback_metrics_row_count"] == boundary["prompt_count"]
            and boundary["callback_prediction_row_count"] == boundary["prompt_count"]
            and boundary["callback_progress_record_count"] == 1
            and boundary["callback_schema"] == "production_metrics_predictions_progress_envelope_v1",
            "evaluation callback does not benchmark the production-sized three-artifact envelope",
        )
    checkpoint = _mapping(benchmark["checkpoint_io"], "checkpoint I/O")
    _exact_keys(checkpoint, {"executed", "artifact_count", "bytes", "sha256"}, "checkpoint I/O")
    checkpoint_executed = _boolean(checkpoint["executed"], "checkpoint I/O executed flag")
    _require(checkpoint_executed == representative, "checkpoint I/O tier execution flag changed")
    checkpoint_count = _integer(checkpoint["artifact_count"], "checkpoint I/O artifact count")
    checkpoint_bytes = _integer(checkpoint["bytes"], "checkpoint I/O bytes")
    _require(
        (checkpoint_count >= 1 and checkpoint_bytes >= 1)
        if representative
        else (checkpoint_count == 0 and checkpoint_bytes == 0),
        "checkpoint I/O tier payload changed",
    )
    _sha256(checkpoint["sha256"], "checkpoint I/O digest")

    final_seal = _mapping(benchmark["final_seal_io"], "final seal I/O")
    _exact_keys(
        final_seal,
        {
            "executed",
            "artifact_count",
            "bytes",
            "sha256",
            "file_names",
            "metrics_row_count",
            "prediction_row_count",
            "summary_schema",
            "attested_file_count",
            "completion_attestation_verified",
            "complete_marker_final_write",
            "run_store_finalize_used",
            "outcome_file_seal_hashes",
            "outcome_file_seal_modes",
            "outcome_file_seal_file_count",
            "outcome_file_seal_scan_and_chmod_completed",
            "seal_restore_only_for_authenticated_transient_cleanup",
        },
        "final seal I/O",
    )
    seal_executed = _boolean(final_seal["executed"], "final seal I/O executed flag")
    _require(seal_executed == representative, "final seal I/O tier execution flag changed")
    seal_count = _integer(final_seal["artifact_count"], "final seal I/O artifact count")
    seal_bytes = _integer(final_seal["bytes"], "final seal I/O bytes")
    metrics_rows = _integer(final_seal["metrics_row_count"], "final seal metrics rows")
    prediction_rows = _integer(final_seal["prediction_row_count"], "final seal prediction rows")
    attested_count = _integer(final_seal["attested_file_count"], "final seal attested-file count")
    outcome_seal_count = _integer(
        final_seal["outcome_file_seal_file_count"],
        "outcome-file seal count",
    )
    outcome_hashes = _mapping(final_seal["outcome_file_seal_hashes"], "outcome-file seal hashes")
    outcome_modes = _mapping(final_seal["outcome_file_seal_modes"], "outcome-file seal modes")
    if representative:
        expected_rows = sum(int(boundary["prompt_count"]) for boundary in boundaries)
        outcome_names = {"metrics.jsonl", "predictions.jsonl", "summary.json"}
        _require(
            seal_count >= 15
            and seal_bytes >= 1
            and final_seal["file_names"]
            == [
                "metrics.jsonl",
                "predictions.jsonl",
                "summary.json",
                "status.json",
                "completion.json",
                "COMPLETE",
            ]
            and metrics_rows == prediction_rows == expected_rows
            and final_seal["summary_schema"] == "production_run_summary_envelope_v1"
            and attested_count == 11
            and _boolean(
                final_seal["completion_attestation_verified"],
                "final seal completion-attestation flag",
            )
            and _boolean(
                final_seal["complete_marker_final_write"],
                "final seal COMPLETE-final-write flag",
            )
            and _boolean(final_seal["run_store_finalize_used"], "final seal RunStore flag"),
            "final seal is not the production-sized RunStore completion envelope",
        )
        _require(
            outcome_seal_count == 3
            and set(outcome_hashes) == outcome_names
            and all(_sha256(outcome_hashes[name], f"sealed {name} digest") for name in outcome_names)
            and outcome_modes == {name: 0 for name in outcome_names}
            and _boolean(
                final_seal["outcome_file_seal_scan_and_chmod_completed"],
                "outcome-file hash/chmod seal flag",
            )
            and _boolean(
                final_seal["seal_restore_only_for_authenticated_transient_cleanup"],
                "outcome-file transient-cleanup-only restore flag",
            ),
            "final outcome files were not hash-scanned and chmod-000 inside the seal envelope",
        )
    else:
        _require(
            seal_count == seal_bytes == metrics_rows == prediction_rows == attested_count == 0
            and outcome_seal_count == 0
            and outcome_hashes == {}
            and outcome_modes == {}
            and final_seal["file_names"] == []
            and final_seal["summary_schema"] == "not_executed_registered_numeric_only"
            and not _boolean(
                final_seal["completion_attestation_verified"],
                "registered-only completion-attestation flag",
            )
            and not _boolean(
                final_seal["complete_marker_final_write"],
                "registered-only COMPLETE flag",
            )
            and not _boolean(final_seal["run_store_finalize_used"], "registered-only RunStore flag"),
            "final seal ran in the registered-only tier",
        )
        _require(
            not _boolean(
                final_seal["outcome_file_seal_scan_and_chmod_completed"],
                "registered-only outcome-file seal flag",
            )
            and not _boolean(
                final_seal["seal_restore_only_for_authenticated_transient_cleanup"],
                "registered-only seal restore flag",
            ),
            "registered-only process claims outcome-file sealing",
        )
    _sha256(final_seal["sha256"], "final seal I/O digest")
    return benchmark


def _validate_standardized_worst_shape_training(
    value: Any,
    *,
    profile: str,
    panel_id: str,
) -> Mapping[str, Any]:
    standardized = _mapping(value, "standardized worst-shape training benchmark")
    _exact_keys(
        standardized,
        {
            "global_maximum_proof",
            "timed_corpus_size",
            "all_timed_prompts_equal_global_maximum",
            "timed_call_shapes",
            "timed_call_shapes_sha256",
            "timed_padded_token_elements",
            "production_updates_per_run",
            "runs_per_panel_per_worker",
            "runs_per_worker",
            "conservative_scaling_ratio",
            "standardized_workload_dominates_every_registered_training_call",
            "tokenizer_host_envelope_contracts",
        },
        "standardized worst-shape training benchmark",
    )
    proof = _mapping(standardized["global_maximum_proof"], "global training maximum proof")
    _exact_keys(
        proof,
        {
            "algorithm",
            "panel_id",
            "profile_plan_bindings",
            "prompt_generation_inputs_identical_across_profiles",
            "registered_run_count",
            "registered_training_prompt_count",
            "law_families",
            "training_views",
            "train_renderers",
            "run_receipts",
            "run_receipts_sha256",
            "maximum_token_length",
            "maximum_prompt_token_length",
            "maximum_prompt_sha256",
            "maximum_prompt_identity",
            "tokenizer_host_envelopes",
            "contextual_scorer_branch_invariant",
            "global_maximum_dominates_every_registered_training_prompt",
            "stream_encoding",
            "controller_scan_seconds",
            "proof_digest",
        },
        "global training maximum proof",
    )
    _require(
        proof["proof_digest"] == semantic_digest({key: proof[key] for key in proof if key != "proof_digest"}),
        "global training maximum proof digest mismatch",
    )
    _require(
        proof["algorithm"] == "exhaustive_registered_80_run_training_prompt_maximum_v1"
        and proof["panel_id"] == panel_id
        and _boolean(
            proof["prompt_generation_inputs_identical_across_profiles"],
            "cross-profile prompt-generation identity flag",
        )
        and proof["stream_encoding"] == "uint64be_length_prefixed_canonical_json_rows_v1",
        "global training maximum proof identity changed",
    )
    _finite(proof["controller_scan_seconds"], "global training maximum controller scan seconds", minimum=1e-9)
    plan_bindings = _mapping(proof["profile_plan_bindings"], "training maximum plan bindings")
    _exact_keys(plan_bindings, PROFILES, "training maximum plan bindings")
    for bound_profile in PROFILES:
        binding = _mapping(plan_bindings[bound_profile], f"{bound_profile} training plan binding")
        _exact_keys(
            binding,
            {"path", "file_sha256", "run_count", "plan_key_sha256"},
            f"{bound_profile} training plan binding",
        )
        _require(
            binding["path"]
            == f"docs/goalzendo/plans/g00f-h200-{bound_profile}-{panel_id.removeprefix('g00f-')}.jsonl"
            and binding["run_count"] == 80,
            f"{bound_profile} training plan binding changed",
        )
        _sha256(binding["file_sha256"], f"{bound_profile} training plan file digest")
        _sha256(binding["plan_key_sha256"], f"{bound_profile} training plan-key digest")
    receipts = list(_sequence(proof["run_receipts"], "training maximum run receipts"))
    _require(
        proof["registered_run_count"] == len(receipts) == 80
        and proof["registered_training_prompt_count"] == 80 * 10_000
        and proof["run_receipts_sha256"] == semantic_digest(receipts),
        "global training maximum run cardinality or digest changed",
    )
    maximums: list[int] = []
    contextual_work_maximums: list[int] = []
    utf8_work_maximums: list[int] = []
    observed_laws: set[str] = set()
    observed_views: set[str] = set()
    observed_renderers: set[str] = set()
    run_by_key: dict[str, Mapping[str, Any]] = {}
    for global_index, raw_receipt in enumerate(receipts):
        receipt = _mapping(raw_receipt, f"training maximum run receipt {global_index}")
        _exact_keys(
            receipt,
            {
                "global_index",
                "baseline_plan_key",
                "seed",
                "derived_seeds_sha256",
                "law_family",
                "training_view",
                "train_renderers",
                "training_renderer_counts",
                "prompt_count",
                "training_prompt_sha256",
                "token_length_stream_sha256",
                "contextual_continuation_stream_sha256",
                "maximum_prompt_token_length",
                "maximum_token_length",
                "maximum_contextual_token_work",
                "maximum_utf8_bytes_work",
            },
            f"training maximum run receipt {global_index}",
        )
        plan_key = str(receipt["baseline_plan_key"])
        _require(
            receipt["global_index"] == global_index
            and len(plan_key) == 20
            and all(character in "0123456789abcdef" for character in plan_key)
            and plan_key not in run_by_key
            and _integer(receipt["seed"], "training maximum seed", minimum=1) > 0
            and receipt["law_family"] in LAW_FAMILIES
            and receipt["training_view"] in TRAINING_VIEWS
            and receipt["prompt_count"] == 10_000,
            "training maximum run identity/cardinality changed",
        )
        run_by_key[plan_key] = receipt
        renderers = [str(item) for item in _sequence(receipt["train_renderers"], "train renderers")]
        renderer_counts = _mapping(receipt["training_renderer_counts"], "training renderer counts")
        _require(
            renderers == ["natural_1", "natural_2", "natural_3", "natural_4"]
            and set(renderer_counts) == set(renderers)
            and sum(_integer(count, "training renderer count") for count in renderer_counts.values())
            == 10_000,
            "training renderer coverage changed",
        )
        for field in (
            "derived_seeds_sha256",
            "training_prompt_sha256",
            "token_length_stream_sha256",
            "contextual_continuation_stream_sha256",
        ):
            _sha256(receipt[field], f"training maximum {field}")
        run_prompt_maximum = _integer(
            receipt["maximum_prompt_token_length"],
            "per-run maximum prompt token length",
            minimum=1,
        )
        run_branch_maximum = _integer(
            receipt["maximum_token_length"],
            "per-run maximum scorer-branch token length",
            minimum=2,
        )
        _require(
            run_branch_maximum == run_prompt_maximum + 1,
            "per-run scorer-branch maximum is not the prompt maximum plus one action token",
        )
        maximums.append(run_branch_maximum)
        contextual_work_maximums.append(
            _integer(
                receipt["maximum_contextual_token_work"],
                "per-run maximum contextual token work",
                minimum=1,
            )
        )
        utf8_work_maximums.append(
            _integer(
                receipt["maximum_utf8_bytes_work"],
                "per-run maximum UTF-8 byte work",
                minimum=1,
            )
        )
        observed_laws.add(str(receipt["law_family"]))
        observed_views.add(str(receipt["training_view"]))
        observed_renderers.update(renderers)
    maximum_length = _integer(
        proof["maximum_token_length"],
        "global maximum scorer-branch token length",
        minimum=2,
    )
    maximum_prompt_length = _integer(
        proof["maximum_prompt_token_length"],
        "global maximum prompt token length",
        minimum=1,
    )
    _require(
        maximum_length == max(maximums)
        and maximum_length == maximum_prompt_length + 1
        and list(proof["law_families"]) == list(LAW_FAMILIES)
        and list(proof["training_views"]) == list(TRAINING_VIEWS)
        and list(proof["train_renderers"]) == ["natural_1", "natural_2", "natural_3", "natural_4"]
        and observed_laws == set(LAW_FAMILIES)
        and observed_views == set(TRAINING_VIEWS)
        and observed_renderers == {"natural_1", "natural_2", "natural_3", "natural_4"}
        and _boolean(
            proof["global_maximum_dominates_every_registered_training_prompt"],
            "global training maximum dominance flag",
        ),
        "global training maximum is not exhaustive over Laws/views/renderers",
    )
    _sha256(proof["maximum_prompt_sha256"], "global maximum prompt digest")
    identity = _mapping(proof["maximum_prompt_identity"], "global maximum prompt identity")
    _exact_keys(
        identity,
        {"baseline_plan_key", "global_index", "prompt_index", "seed", "law_family", "training_view"},
        "global maximum prompt identity",
    )
    maximum_run = run_by_key.get(str(identity["baseline_plan_key"]))
    _require(
        maximum_run is not None
        and maximum_run["global_index"] == identity["global_index"]
        and maximum_run["seed"] == identity["seed"]
        and maximum_run["law_family"] == identity["law_family"]
        and maximum_run["training_view"] == identity["training_view"]
        and maximum_run["maximum_token_length"] == maximum_length
        and 0 <= _integer(identity["prompt_index"], "global maximum prompt index") < 10_000,
        "global maximum prompt identity is not bound to a maximizing run",
    )
    contextual_invariant = _mapping(
        proof["contextual_scorer_branch_invariant"],
        "contextual scorer-branch invariant",
    )
    _require(
        contextual_invariant
        == {
            "contextual_continuation_mode": (
                "goalzendo.modeling.encode_action_continuations:add_prompt_special_tokens=false"
            ),
            "action_labels": list(ACTION_LABELS),
            "required_token_count_per_action_continuation": 1,
            "registered_prompt_count": 80 * 10_000,
            "per_run_stream_digest_field": "contextual_continuation_stream_sha256",
            "every_registered_a_and_b_continuation_exactly_one_token": True,
            "maximum_prompt_token_length": maximum_prompt_length,
            "maximum_scorer_branch_token_length": maximum_length,
            "maximum_scorer_branch_is_prompt_plus_one_token": True,
            "global_maximum_ranked_over_prompt_and_both_action_branches": True,
        },
        "contextual A/B continuation or full scorer-branch maximum proof changed",
    )
    host_envelopes = _mapping(
        proof["tokenizer_host_envelopes"],
        "tokenizer host envelopes",
    )
    _exact_keys(
        host_envelopes,
        {
            "contextual_continuation_mode",
            "action_labels",
            "standalone_action_token_ids",
            "standalone_action_token_ids_sha256",
            "envelope_order",
            "envelopes",
            "measured_seconds_aggregation",
        },
        "tokenizer host envelopes",
    )
    envelope_order = [
        "contextual_token_work_variability",
        "utf8_byte_work_variability",
        "contextual_token_work_dominance",
        "utf8_byte_work_dominance",
    ]
    _require(
        host_envelopes["contextual_continuation_mode"]
        == "goalzendo.modeling.encode_action_continuations:add_prompt_special_tokens=false"
        and list(host_envelopes["action_labels"]) == list(ACTION_LABELS)
        and host_envelopes["standalone_action_token_ids"] == [[32], [33]]
        and host_envelopes["standalone_action_token_ids_sha256"] == semantic_digest([[32], [33]])
        and list(host_envelopes["envelope_order"]) == envelope_order
        and host_envelopes["measured_seconds_aggregation"] == "sum_all_four_envelopes",
        "tokenizer host-envelope identity/aggregation changed",
    )
    raw_envelopes = _mapping(host_envelopes["envelopes"], "tokenizer host-envelope map")
    _exact_keys(raw_envelopes, envelope_order, "tokenizer host-envelope map")
    validated_envelopes: dict[str, Mapping[str, Any]] = {}
    envelope_metrics = {
        "contextual_token_work_variability": "total_contextual_token_work",
        "utf8_byte_work_variability": "total_utf8_bytes",
        "contextual_token_work_dominance": "total_contextual_token_work",
        "utf8_byte_work_dominance": "total_utf8_bytes",
    }
    for envelope_name in envelope_order:
        metric = envelope_metrics[envelope_name]
        envelope = _mapping(raw_envelopes[envelope_name], f"{envelope_name} host envelope")
        _exact_keys(
            envelope,
            {
                "role",
                "selection_algorithm",
                "metric_field",
                "source_row_count",
                "rows",
                "rows_sha256",
                "selected_metric_sum",
                "selected_metric_maximum",
                "repetition_count",
                "execution_row_count_per_update",
                "ordered_execution_prompt_sha256",
                "ordered_execution_rows_sha256",
                "baseline_call_partition_sizes",
                "baseline_call_partitions_sha256",
                "tuned_call_partition_sizes",
                "tuned_call_partitions_sha256",
                "variability_stress_only",
                "hard_dominance_tier",
                "each_baseline_call_dominates_any_registered_10_row_call",
                "tuned_ordered50_is_identical_to_concatenated_baseline_calls",
                "tuned_total_dominates_any_registered_50_row_batch",
                "global_metric_maximum_over_all_registered_occurrences",
            },
            f"{envelope_name} host envelope",
        )
        host_rows = list(_sequence(envelope["rows"], f"{envelope_name} host-envelope rows"))
        role = "variability_stress" if envelope_name.endswith("_variability") else "hard_dominance"
        source_row_count = 10 if role == "variability_stress" else 1
        repetition_count = 5 if role == "variability_stress" else 50
        expected_algorithm = (
            f"global_top10_distinct_by_{metric}_desc_prompt_sha256_tiebreak_v1"
            if role == "variability_stress"
            else f"global_maximum_by_{metric}_desc_prompt_sha256_tiebreak_v1"
        )
        _require(
            envelope["role"] == role
            and envelope["selection_algorithm"] == expected_algorithm
            and envelope["metric_field"] == metric
            and envelope["source_row_count"] == len(host_rows) == source_row_count
            and envelope["rows_sha256"] == semantic_digest(host_rows)
            and envelope["repetition_count"] == repetition_count
            and envelope["execution_row_count_per_update"] == 50
            and list(envelope["baseline_call_partition_sizes"]) == [10] * 5
            and list(envelope["tuned_call_partition_sizes"]) == [50]
            and _boolean(
                envelope["variability_stress_only"],
                "host-envelope variability role flag",
            )
            == (role == "variability_stress")
            and _boolean(envelope["hard_dominance_tier"], "host-envelope dominance role flag")
            == (role == "hard_dominance")
            and _boolean(
                envelope["each_baseline_call_dominates_any_registered_10_row_call"],
                "baseline host-envelope dominance flag",
            )
            == (role == "hard_dominance")
            and _boolean(
                envelope["tuned_ordered50_is_identical_to_concatenated_baseline_calls"],
                "cross-profile host-envelope order flag",
            )
            and _boolean(
                envelope["tuned_total_dominates_any_registered_50_row_batch"],
                "tuned host-envelope dominance flag",
            )
            == (role == "hard_dominance")
            and _boolean(
                envelope["global_metric_maximum_over_all_registered_occurrences"],
                "global host metric maximum flag",
            )
            == (role == "hard_dominance"),
            f"{envelope_name} host-envelope identity/cardinality changed",
        )
        seen_host_prompts: set[str] = set()
        metric_values: list[int] = []
        for row_index, raw_row in enumerate(host_rows):
            row = _mapping(raw_row, f"{envelope_name} host-envelope row {row_index}")
            _exact_keys(
                row,
                {
                    "baseline_plan_key",
                    "global_index",
                    "prompt_index",
                    "prompt_sha256",
                    "prompt_utf8_bytes",
                    "prompt_token_length",
                    "prompt_a_utf8_bytes",
                    "prompt_a_token_length",
                    "prompt_b_utf8_bytes",
                    "prompt_b_token_length",
                    "continuation_a_token_count",
                    "continuation_a_token_ids_sha256",
                    "continuation_b_token_count",
                    "continuation_b_token_ids_sha256",
                    "total_contextual_token_work",
                    "maximum_contextual_token_length",
                    "total_utf8_bytes",
                    "maximum_utf8_bytes",
                },
                f"{envelope_name} host-envelope row {row_index}",
            )
            prompt_digest = _sha256(row["prompt_sha256"], "host-envelope prompt digest")
            _require(
                prompt_digest not in seen_host_prompts
                and str(row["baseline_plan_key"]) in run_by_key
                and 0 <= _integer(row["global_index"], "host-envelope global index") < 80
                and 0 <= _integer(row["prompt_index"], "host-envelope prompt index") < 10_000,
                "tokenizer host-envelope row identity is invalid or duplicated",
            )
            seen_host_prompts.add(prompt_digest)
            for field in (
                "prompt_utf8_bytes",
                "prompt_token_length",
                "prompt_a_utf8_bytes",
                "prompt_a_token_length",
                "prompt_b_utf8_bytes",
                "prompt_b_token_length",
                "continuation_a_token_count",
                "continuation_b_token_count",
                "total_contextual_token_work",
                "maximum_contextual_token_length",
                "total_utf8_bytes",
                "maximum_utf8_bytes",
            ):
                _integer(row[field], f"host-envelope {field}", minimum=1)
            _require(
                row["prompt_a_utf8_bytes"] >= row["prompt_utf8_bytes"]
                and row["prompt_b_utf8_bytes"] >= row["prompt_utf8_bytes"]
                and row["continuation_a_token_count"] == row["continuation_b_token_count"] == 1
                and row["prompt_a_token_length"]
                == row["prompt_token_length"] + row["continuation_a_token_count"]
                and row["prompt_b_token_length"]
                == row["prompt_token_length"] + row["continuation_b_token_count"]
                and row["total_contextual_token_work"]
                == row["prompt_token_length"] + row["prompt_a_token_length"] + row["prompt_b_token_length"]
                and row["maximum_contextual_token_length"]
                == max(
                    row["prompt_token_length"],
                    row["prompt_a_token_length"],
                    row["prompt_b_token_length"],
                )
                and row["total_utf8_bytes"]
                == row["prompt_utf8_bytes"] + row["prompt_a_utf8_bytes"] + row["prompt_b_utf8_bytes"]
                and row["maximum_utf8_bytes"]
                == max(
                    row["prompt_utf8_bytes"],
                    row["prompt_a_utf8_bytes"],
                    row["prompt_b_utf8_bytes"],
                ),
                "tokenizer host-envelope contextual lengths are inconsistent",
            )
            _sha256(row["continuation_a_token_ids_sha256"], "host-envelope A continuation digest")
            _sha256(row["continuation_b_token_ids_sha256"], "host-envelope B continuation digest")
            metric_values.append(int(row[metric]))
        row_ranks = [(int(row[metric]), str(row["prompt_sha256"])) for row in host_rows]
        ordered_digests = [str(row["prompt_sha256"]) for row in host_rows]
        execution_digests = ordered_digests * repetition_count
        baseline_partitions = [
            execution_digests[start : start + 10] for start in range(0, len(execution_digests), 10)
        ]
        global_metric_maximum = max(
            contextual_work_maximums if metric == "total_contextual_token_work" else utf8_work_maximums
        )
        _require(
            row_ranks == sorted(row_ranks, reverse=True)
            and envelope["selected_metric_sum"] == sum(metric_values)
            and envelope["selected_metric_maximum"] == max(metric_values)
            and envelope["ordered_execution_prompt_sha256"] == semantic_digest(execution_digests)
            and envelope["ordered_execution_rows_sha256"] == semantic_digest(host_rows * repetition_count)
            and envelope["baseline_call_partitions_sha256"] == semantic_digest(baseline_partitions)
            and envelope["tuned_call_partitions_sha256"] == semantic_digest([execution_digests])
            and (
                role == "variability_stress" or envelope["selected_metric_maximum"] == global_metric_maximum
            ),
            f"{envelope_name} host-envelope ordering/metric proof changed",
        )
        validated_envelopes[envelope_name] = envelope
    _same_number(
        standardized["conservative_scaling_ratio"],
        1.0,
        "worst-shape training scaling ratio",
    )
    _require(
        standardized["timed_corpus_size"] == 10_000
        and _boolean(
            standardized["all_timed_prompts_equal_global_maximum"],
            "all timing prompts equal maximum flag",
        )
        and standardized["production_updates_per_run"] == PRODUCTION_UPDATES
        and standardized["runs_per_panel_per_worker"] == RUNS_PER_PANEL_PER_WORKER
        and standardized["runs_per_worker"] == RUNS_PER_WORKER
        and _boolean(
            standardized["standardized_workload_dominates_every_registered_training_call"],
            "standardized training dominance flag",
        ),
        "standardized timing workload/scaling contract changed",
    )
    accumulation = int(PROFILE_CONTRACT[profile]["gradient_accumulation_steps"])
    batch_size = int(PROFILE_CONTRACT[profile]["train_batch_size"])
    shapes = list(_sequence(standardized["timed_call_shapes"], "standardized timed call shapes"))
    _require(
        len(shapes) == len(UPDATES) * accumulation
        and standardized["timed_call_shapes_sha256"] == semantic_digest(shapes),
        "standardized timed call-shape cardinality/digest changed",
    )
    padded_total = 0
    for call_index, raw_shape in enumerate(shapes):
        shape = _mapping(raw_shape, f"standardized timed call shape {call_index}")
        _exact_keys(
            shape,
            {"update", "micro_step", "example_count", "max_seq_len", "padded_elements"},
            f"standardized timed call shape {call_index}",
        )
        expected_update = call_index // accumulation + 1
        _require(
            shape["update"] == expected_update
            and shape["micro_step"] == call_index
            and shape["example_count"] == batch_size
            and shape["max_seq_len"] == maximum_prompt_length
            and shape["padded_elements"] == batch_size * maximum_prompt_length,
            "standardized timed call shape differs from the frozen profile envelope",
        )
        padded_total += int(shape["padded_elements"])
    _require(
        standardized["timed_padded_token_elements"]
        == padded_total
        == len(UPDATES) * int(PROFILE_CONTRACT[profile]["effective_batch_size"]) * maximum_prompt_length,
        "standardized timed padded-token total changed",
    )
    host_contract = _mapping(
        standardized["tokenizer_host_envelope_contracts"],
        "tokenizer host-envelope timing contracts",
    )
    _exact_keys(
        host_contract,
        {
            "contextual_continuation_mode",
            "envelope_order",
            "measured_seconds_aggregation",
            "measured_seconds_by_envelope",
            "measured_seconds_sum",
            "envelopes",
        },
        "tokenizer host-envelope timing contracts",
    )
    seconds_by_envelope = _mapping(
        host_contract["measured_seconds_by_envelope"],
        "host-envelope measured seconds",
    )
    _exact_keys(seconds_by_envelope, envelope_order, "host-envelope measured seconds")
    measured_seconds = {
        name: _finite(seconds_by_envelope[name], f"{name} host-envelope seconds", minimum=1e-9)
        for name in envelope_order
    }
    contract_envelopes = _mapping(host_contract["envelopes"], "host-envelope timing map")
    _exact_keys(contract_envelopes, envelope_order, "host-envelope timing map")
    expected_partition_sizes = [batch_size] * accumulation
    for envelope_name in envelope_order:
        envelope = validated_envelopes[envelope_name]
        contract_envelope = _mapping(
            contract_envelopes[envelope_name],
            f"{envelope_name} host-envelope timing contract",
        )
        expected_digests = [str(row["prompt_sha256"]) for row in envelope["rows"]] * int(
            envelope["repetition_count"]
        )
        partitions = [
            expected_digests[start : start + batch_size]
            for start in range(0, len(expected_digests), batch_size)
        ]
        _require(
            contract_envelope
            == {
                "metric_field": envelope_metrics[envelope_name],
                "role": envelope["role"],
                "source_row_count": envelope["source_row_count"],
                "rows_sha256": envelope["rows_sha256"],
                "selected_metric_sum": envelope["selected_metric_sum"],
                "selected_metric_maximum": envelope["selected_metric_maximum"],
                "repetition_count": envelope["repetition_count"],
                "ordered_execution_prompt_sha256": envelope["ordered_execution_prompt_sha256"],
                "timed_updates": len(UPDATES),
                "profile_batch_size": batch_size,
                "profile_call_partition_sizes": expected_partition_sizes,
                "profile_call_partitions_sha256": semantic_digest(partitions),
                "profile_scorer_call_count": len(UPDATES) * accumulation,
                "total_examples_tokenized": len(UPDATES) * 50,
                "warmup_horizon_excluded": True,
                "worker_contextual_rows_replayed": True,
            },
            f"{envelope_name} tokenizer host-envelope timing geometry changed",
        )
    _require(
        host_contract["contextual_continuation_mode"]
        == "goalzendo.modeling.encode_action_continuations:add_prompt_special_tokens=false"
        and list(host_contract["envelope_order"]) == envelope_order
        and host_contract["measured_seconds_aggregation"] == "sum_all_four_envelopes"
        and math.isclose(
            _finite(host_contract["measured_seconds_sum"], "host-envelope seconds sum", minimum=1e-9),
            sum(measured_seconds.values()),
            rel_tol=1e-9,
            abs_tol=1e-9,
        ),
        "tokenizer host-envelope timing aggregation changed",
    )
    return standardized


def _validate_process_record_internal(value: Any) -> Mapping[str, Any]:
    record = _mapping(value, "process record")
    _exact_keys(
        record,
        {
            "schema",
            "schema_version",
            "identity",
            "actual_model_execution",
            "profile_contract",
            "engineering_scope",
            "coverage",
            "trainable_parameters",
            "initial_state",
            "stochastic_modules",
            "execution_benchmark",
            "steps",
            "memory",
            "timing",
            "record_digest",
        },
        "process record",
    )
    _require(
        record["schema"] == PROCESS_SCHEMA and record["schema_version"] == PROCESS_SCHEMA_VERSION,
        "process record schema mismatch",
    )
    _verify_self_digest(record, "record_digest", "process record")
    identity = _mapping(record["identity"], "process identity")
    _exact_keys(
        identity,
        {
            "profile",
            "replicate",
            "panel_id",
            "model_name",
            "model_revision",
            "law_family",
            "device_uuid",
            "device_name",
            "process_uuid",
        },
        "process identity",
    )
    profile = str(identity["profile"])
    panel_id = str(identity["panel_id"])
    _require(profile in PROFILES, "process profile is unknown")
    _require(identity["replicate"] in REPLICATES[profile], "process replicate is invalid for profile")
    _require(panel_id in PANELS, "process panel is unknown")
    _require(identity["model_name"] == PANELS[panel_id]["model_name"], "process model name mismatch")
    _require(
        identity["model_revision"] == PANELS[panel_id]["model_revision"],
        "process model revision mismatch",
    )
    _require(identity["law_family"] in LAW_FAMILIES, "process Law family is unknown")
    _require(GPU_UUID_PATTERN.fullmatch(str(identity["device_uuid"])) is not None, "invalid H200 GPU UUID")
    _require(str(identity["device_name"]).startswith("NVIDIA H200"), "process did not use NVIDIA H200")
    try:
        parsed_process_uuid = uuid.UUID(str(identity["process_uuid"]))
    except ValueError as error:
        raise QualificationError("process UUID is invalid") from error
    _require(str(parsed_process_uuid) == identity["process_uuid"], "process UUID is not canonical")
    _require(
        _boolean(record["actual_model_execution"], "actual-model execution flag"),
        "process is not actual-model evidence",
    )
    _require(record["profile_contract"] == PROFILE_CONTRACT[profile], "process profile contract mismatch")
    _engineering_scope(record["engineering_scope"], "process engineering scope")
    trainable_manifest = _validate_trainable_parameter_manifest(
        record["trainable_parameters"],
        panel_id,
    )

    coverage = _mapping(record["coverage"], "process coverage")
    _exact_keys(
        coverage,
        {
            "updates",
            "training_views",
            "action_labels",
            "dataset_sha256",
            "engineering_seed",
            "ordered_corpus_sample_ids_sha256",
            "order_algorithm",
            "worst_case_token_sample",
        },
        "process coverage",
    )
    _require(
        list(_sequence(coverage["updates"], "process covered updates")) == list(UPDATES),
        "process must cover updates 1 through 8",
    )
    _require(
        list(_sequence(coverage["training_views"], "process covered views")) == list(TRAINING_VIEWS),
        "process view coverage mismatch",
    )
    _require(
        list(_sequence(coverage["action_labels"], "process action labels")) == list(ACTION_LABELS),
        "process action labels mismatch",
    )
    _sha256(coverage["dataset_sha256"], "process dataset digest")
    _integer(coverage["engineering_seed"], "process engineering seed")
    _sha256(coverage["ordered_corpus_sample_ids_sha256"], "ordered corpus sample-ID digest")
    _require(
        coverage["order_algorithm"] == ORDER_ALGORITHM, "qualification order construction algorithm changed"
    )
    worst = _mapping(coverage["worst_case_token_sample"], "worst-case token sample")
    _exact_keys(
        worst,
        {
            "sample_id",
            "view",
            "token_length",
            "corpus_max_token_length",
            "included_in_every_update",
        },
        "worst-case token sample",
    )
    _require(isinstance(worst["sample_id"], str) and bool(worst["sample_id"]), "worst sample ID is empty")
    _require(worst["view"] in TRAINING_VIEWS, "worst sample view is unknown")
    token_length = _integer(worst["token_length"], "worst sample token length", minimum=1)
    _require(
        _integer(worst["corpus_max_token_length"], "corpus maximum token length", minimum=1) == token_length,
        "worst sample is not at the corpus maximum token length",
    )
    _boolean(worst["included_in_every_update"], "worst sample inclusion flag")

    initial = _mapping(record["initial_state"], "initial state")
    _exact_keys(
        initial,
        {
            "fresh_model_instance",
            "fresh_optimizer_instance",
            "model_snapshot_sha256",
            "model_state_sha256",
            "optimizer_state_sha256",
        },
        "initial state",
    )
    _boolean(initial["fresh_model_instance"], "fresh model flag")
    _boolean(initial["fresh_optimizer_instance"], "fresh optimizer flag")
    for field in ("model_snapshot_sha256", "model_state_sha256", "optimizer_state_sha256"):
        _sha256(initial[field], f"initial {field}")

    stochastic = _mapping(record["stochastic_modules"], "stochastic-module audit")
    _exact_keys(
        stochastic,
        {
            "enumeration_complete",
            "dropout_inactive",
            "all_stochastic_modules_inactive",
            "dropout_module_count",
            "stochastic_module_count",
            "functional_stochasticity_audit",
            "modules",
            "modules_sha256",
        },
        "stochastic-module audit",
    )
    _boolean(stochastic["enumeration_complete"], "stochastic enumeration flag")
    modules = _sequence(stochastic["modules"], "stochastic module list")
    module_names: list[str] = []
    dropout_count = 0
    inactive = True
    for index, raw_module in enumerate(modules):
        module = _mapping(raw_module, f"stochastic module {index}")
        _exact_keys(module, {"name", "class_name", "kind", "active"}, f"stochastic module {index}")
        _require(isinstance(module["name"], str) and bool(module["name"]), "stochastic module name is empty")
        _require(
            isinstance(module["class_name"], str) and bool(module["class_name"]),
            "stochastic module class is empty",
        )
        _require(module["kind"] in {"dropout", "stochastic"}, "stochastic module kind is invalid")
        active = _boolean(module["active"], f"stochastic module {index} active flag")
        module_names.append(str(module["name"]))
        dropout_count += int(module["kind"] == "dropout")
        inactive = inactive and not active
    _require(module_names == sorted(set(module_names)), "stochastic modules must use unique sorted names")
    _require(
        _integer(stochastic["dropout_module_count"], "dropout module count") == dropout_count,
        "dropout module count mismatch",
    )
    _require(
        _integer(stochastic["stochastic_module_count"], "stochastic module count") == len(modules),
        "stochastic module count mismatch",
    )
    _require(
        stochastic["modules_sha256"] == semantic_digest(list(modules)), "stochastic module digest mismatch"
    )
    dropout_inactive = all(module["kind"] != "dropout" or not module["active"] for module in modules)
    _require(
        _boolean(stochastic["dropout_inactive"], "dropout inactive flag") == dropout_inactive,
        "dropout inactive flag is inconsistent",
    )
    _require(
        _boolean(stochastic["all_stochastic_modules_inactive"], "stochastic inactive flag") == inactive,
        "stochastic inactive flag is inconsistent",
    )
    functional = _mapping(
        stochastic["functional_stochasticity_audit"],
        "functional stochasticity audit",
    )
    _exact_keys(
        functional,
        {
            "runtime_training_mode",
            "model_config_dropout_fields_enumerated",
            "config_probability_count",
            "attention_dropout_fields_enumerated",
            "effective_probability_zero",
        },
        "functional stochasticity audit",
    )
    config_count = _integer(
        functional["config_probability_count"],
        "config stochastic probability count",
        minimum=1,
    )
    attention_count = _integer(
        functional["attention_dropout_fields_enumerated"],
        "attention-dropout field count",
        minimum=1,
    )
    _require(
        _boolean(functional["runtime_training_mode"], "stochastic training-mode flag")
        and _boolean(
            functional["model_config_dropout_fields_enumerated"],
            "model-config stochastic enumeration flag",
        )
        and _boolean(functional["effective_probability_zero"], "effective dropout-zero flag")
        and config_count <= dropout_count
        and attention_count <= config_count,
        "functional/config/attention dropout audit is incomplete",
    )

    execution_benchmark = _validate_execution_benchmark(
        record["execution_benchmark"],
        profile=profile,
        law_family=str(identity["law_family"]),
    )

    steps = _sequence(record["steps"], "process step records")
    _require(len(steps) == len(UPDATES), "process must contain exactly eight step records")
    previous_parameter_binding: tuple[str, int] | None = None
    for expected_update, raw_step in zip(UPDATES, steps, strict=True):
        step = _mapping(raw_step, f"step {expected_update}")
        _exact_keys(
            step,
            {
                "update",
                "ordered_example_count",
                "ordered_example_ids",
                "ordered_example_ids_sha256",
                "worst_sample_insertion",
                "rng_receipt",
                "outputs",
                "outputs_sha256",
                "registered_production_shaped_parity_chunk",
                "vectors",
                "state_hashes",
            },
            f"step {expected_update}",
        )
        _require(step["update"] == expected_update, "process steps are not updates 1 through 8 in order")
        _require(
            _integer(step["ordered_example_count"], f"step {expected_update} example count") == 50,
            "each optimizer update must bind exactly 50 ordered examples",
        )
        ordered_ids = [
            str(item)
            for item in _sequence(
                step["ordered_example_ids"],
                f"step {expected_update} ordered IDs",
            )
        ]
        _require(
            len(ordered_ids) == 50 and len(set(ordered_ids)) == 50,
            "optimizer update IDs are incomplete or duplicated",
        )
        _sha256(step["ordered_example_ids_sha256"], f"step {expected_update} ordered-data digest")
        _require(
            step["ordered_example_ids_sha256"] == semantic_digest(ordered_ids),
            "ordered example IDs digest does not match explicit order",
        )
        insertion = _mapping(step["worst_sample_insertion"], "worst-sample insertion")
        _exact_keys(
            insertion,
            {
                "algorithm",
                "engineering_seed",
                "base_order_sha256",
                "inserted",
                "position",
                "worst_sample_id",
                "replaced_sample_id",
                "final_order_sha256",
            },
            "worst-sample insertion",
        )
        _require(
            insertion["algorithm"] == ORDER_ALGORITHM
            and insertion["engineering_seed"] == coverage["engineering_seed"],
            "worst-sample order recipe changed",
        )
        _sha256(insertion["base_order_sha256"], "base order digest")
        _require(
            insertion["final_order_sha256"] == step["ordered_example_ids_sha256"],
            "worst-sample final order digest differs",
        )
        position = _integer(insertion["position"], "worst-sample position")
        _require(
            0 <= position < 50 and insertion["worst_sample_id"] == worst["sample_id"],
            "worst-sample insertion identity/position changed",
        )
        _require(
            ordered_ids[position] == worst["sample_id"],
            "corpus-max worst sample is not truly in the optimizer update",
        )
        inserted = _boolean(insertion["inserted"], "worst-sample inserted flag")
        if inserted:
            _require(
                position == 49
                and isinstance(insertion["replaced_sample_id"], str)
                and bool(insertion["replaced_sample_id"]),
                "inserted worst sample did not replace the fixed final position",
            )
        else:
            _require(
                insertion["replaced_sample_id"] is None, "non-inserted worst sample records a replacement"
            )
        rng = _mapping(step["rng_receipt"], f"step {expected_update} RNG receipt")
        _exact_keys(
            rng,
            {
                "helper",
                "engineering_seed",
                "micro_steps",
                "scorer_calls_guarded",
            },
            f"step {expected_update} RNG receipt",
        )
        accumulation_steps = int(PROFILE_CONTRACT[profile]["gradient_accumulation_steps"])
        expected_micro_steps = list(
            range((expected_update - 1) * accumulation_steps, expected_update * accumulation_steps)
        )
        _require(
            rng["helper"] == "goalzendo.training._step_rng"
            and rng["engineering_seed"] == QUALIFICATION_ENGINEERING_SEED
            and list(_sequence(rng["micro_steps"], "RNG micro-steps")) == expected_micro_steps
            and _boolean(rng["scorer_calls_guarded"], "RNG scorer-call guard"),
            "qualification update did not replay production RNG boundaries",
        )
        outputs = _sequence(step["outputs"], f"step {expected_update} outputs")
        _require(len(outputs) == len(TRAINING_VIEWS), "each step must bind one output for every view")
        validated_outputs = [
            _validate_output(item, f"step {expected_update} output {index}")
            for index, item in enumerate(outputs)
        ]
        _require(
            [item["view"] for item in validated_outputs] == list(TRAINING_VIEWS),
            "step outputs are not in the exact registered view order",
        )
        _require(step["outputs_sha256"] == semantic_digest(list(outputs)), "step output digest mismatch")
        matching_worst = [
            item
            for item in validated_outputs
            if item["sample_id"] == worst["sample_id"] and item["view"] == worst["view"]
        ]
        if worst["included_in_every_update"]:
            _require(len(matching_worst) == 1, "registered worst-token sample is absent from a step")
            _require(
                matching_worst[0]["token_length"] == token_length
                and matching_worst[0]["worst_case_token_sample"] is True,
                "worst-token output metadata mismatch",
            )

        parity_chunk = step["registered_production_shaped_parity_chunk"]
        if expected_update in REGISTERED_PARITY_UPDATE_CHECKPOINTS:
            validated_chunk = _validate_registered_parity_chunk(
                parity_chunk,
                profile=profile,
                expected_state=f"after_update_{expected_update}",
            )
            initial_chunk = execution_benchmark["registered_production_shaped_parity_chunk"]
            for field in (
                "prompt_count",
                "example_count",
                "production_example_partition_count",
                "batch_size",
                "batch_count",
                "scorer_call_count",
                "prompt_sha256",
                "contiguous_example_chunk",
                "prompt_views_per_example",
                "production_bank_name",
                "production_bank_start_index",
                "example_ids",
                "example_ids_sha256",
                "token_shapes",
                "token_shape_sha256",
            ):
                _require(
                    validated_chunk[field] == initial_chunk[field],
                    f"registered parity chunk identity changed at update {expected_update}",
                )
        else:
            _require(
                parity_chunk is None,
                f"unexpected registered parity diagnostic at update {expected_update}",
            )

        vectors = _mapping(step["vectors"], f"step {expected_update} vectors")
        _exact_keys(vectors, VECTOR_KINDS, f"step {expected_update} vectors")
        descriptors = [
            _validate_vector_descriptor(vectors[kind], f"step {expected_update} {kind}")
            for kind in VECTOR_KINDS
        ]
        for vector_kind, descriptor in zip(VECTOR_KINDS, descriptors, strict=True):
            label = f"step {expected_update} {vector_kind}"
            _require(
                descriptor["element_count"] == TRAINABLE_NUMEL[panel_id],
                f"{label} does not cover the exact frozen trainable parameter count",
            )
            _require(
                descriptor["parameter_keys_sha256"] == trainable_manifest["parameter_keys_sha256"],
                f"{label} parameter keys differ from the trainable-parameter manifest",
            )
            _require(
                descriptor["trainable_parameter_manifest_sha256"] == trainable_manifest["manifest_sha256"],
                f"{label} trainable-parameter manifest link is incorrect",
            )
            _require(
                descriptor["chunk_count"] == len(trainable_manifest["entries"]),
                f"{label} does not contain exactly one chunk per trainable parameter",
            )
        parameter_bindings = {
            (str(item["parameter_keys_sha256"]), int(item["element_count"])) for item in descriptors
        }
        _require(len(parameter_bindings) == 1, "step vector parameter bindings differ")
        binding = next(iter(parameter_bindings))
        if previous_parameter_binding is not None:
            _require(binding == previous_parameter_binding, "vector parameter binding changed across updates")
        previous_parameter_binding = binding

        state = _mapping(step["state_hashes"], f"step {expected_update} state hashes")
        _exact_keys(
            state,
            {"model_state_sha256", "optimizer_state_sha256", "combined_state_sha256", "outputs_sha256"},
            f"step {expected_update} state hashes",
        )
        model_digest = _sha256(state["model_state_sha256"], "step model-state digest")
        optimizer_digest = _sha256(state["optimizer_state_sha256"], "step optimizer-state digest")
        _require(state["outputs_sha256"] == step["outputs_sha256"], "state/output digest binding mismatch")
        _require(
            state["combined_state_sha256"]
            == semantic_digest(
                {"model_state_sha256": model_digest, "optimizer_state_sha256": optimizer_digest}
            ),
            "combined state digest mismatch",
        )

    memory = _mapping(record["memory"], "process memory")
    _exact_keys(
        memory,
        {
            "cuda_synchronized_before_read",
            "peak_stats_reset_before_run",
            "peak_allocated_bytes",
            "peak_reserved_bytes",
        },
        "process memory",
    )
    _boolean(memory["cuda_synchronized_before_read"], "CUDA synchronization flag")
    _boolean(memory["peak_stats_reset_before_run"], "peak-memory reset flag")
    allocated = _integer(memory["peak_allocated_bytes"], "peak allocated bytes")
    reserved = _integer(memory["peak_reserved_bytes"], "peak reserved bytes")
    _require(allocated <= reserved, "peak allocated memory exceeds peak reserved memory")

    timing = _mapping(record["timing"], "process timing")
    _exact_keys(
        timing,
        {
            "tokenizer_load_initialization_seconds",
            "data_bank_render_tokenization_seconds",
            "model_load_seconds",
            "updates_1_to_8_seconds",
            "updates_1_to_8_cuda_event_seconds_diagnostic",
            "tokenizer_host_envelopes_8_updates_seconds",
            "update_timing_contract",
            "evaluation_13_boundaries_seconds",
            "evaluation_callback_io_seconds",
            "final_checkpoint_io_seconds",
            "outcome_seal_seconds",
            "update_timing_method",
            "fresh_timing_only_model_optimizer",
            "vector_state_capture_excluded_from_update_timing",
            "warmup_model_discarded_before_measurement",
            "capture_path_cuda_event_seconds_diagnostic_only",
            "warmup_excluded",
            "profile_timing_order",
            "measured_end_to_end_seconds",
            "evaluation_boundary_count",
            "projection_training_updates",
            "worker_mix",
            "projection_safety_multiplier",
        },
        "process timing",
    )
    components = [
        _finite(
            timing["tokenizer_load_initialization_seconds"],
            "tokenizer load/initialization seconds",
            minimum=0.0,
        ),
        _finite(
            timing["data_bank_render_tokenization_seconds"],
            "data/bank/render/tokenization seconds",
            minimum=0.0,
        ),
        _finite(timing["model_load_seconds"], "model-load seconds", minimum=0.0),
        _finite(timing["updates_1_to_8_seconds"], "eight-update seconds", minimum=0.0),
        _finite(
            timing["tokenizer_host_envelopes_8_updates_seconds"],
            "four tokenizer host-envelope eight-update seconds",
            minimum=0.0,
        ),
        _finite(timing["evaluation_13_boundaries_seconds"], "evaluation seconds", minimum=0.0),
        _finite(timing["evaluation_callback_io_seconds"], "evaluation callback I/O seconds", minimum=0.0),
        _finite(timing["final_checkpoint_io_seconds"], "final checkpoint I/O seconds", minimum=0.0),
        _finite(timing["outcome_seal_seconds"], "outcome-seal seconds", minimum=0.0),
    ]
    representative = bool(execution_benchmark["timing_representative"])
    _require(components[2] > 0.0, "model load timing must be positive")
    clean_cuda_seconds = _finite(
        timing["updates_1_to_8_cuda_event_seconds_diagnostic"],
        "clean eight-update CUDA-event diagnostic seconds",
        minimum=0.0,
    )
    capture_cuda_seconds = _finite(
        timing["capture_path_cuda_event_seconds_diagnostic_only"],
        "capture-path CUDA-event diagnostic seconds",
        minimum=0.0,
    )
    if representative:
        _require(
            all(component > 0.0 for component in components)
            and clean_cuda_seconds > 0.0
            and capture_cuda_seconds > 0.0,
            "every production timing-representative component must be positive",
        )
    else:
        _require(
            components[0]
            == components[1]
            == components[3]
            == components[4]
            == components[5]
            == components[6]
            == components[7]
            == components[8]
            == clean_cuda_seconds
            == 0.0,
            "registered-only process contains production timing components",
        )
    measured = _finite(timing["measured_end_to_end_seconds"], "end-to-end seconds", minimum=0.0)
    _require(
        timing["update_timing_method"]
        == (
            "shared_train_steps_8_update_wall_clock_single_pre_post_cuda_sync_horizon"
            if representative
            else "not_measured_registered_numeric_only"
        )
        and _boolean(timing["fresh_timing_only_model_optimizer"], "fresh timing-only state flag")
        == representative
        and _boolean(
            timing["vector_state_capture_excluded_from_update_timing"],
            "timing capture-exclusion flag",
        )
        == representative
        and _boolean(
            timing["warmup_model_discarded_before_measurement"],
            "timing warmup-state discard flag",
        )
        == representative
        and _boolean(timing["warmup_excluded"], "timing warmup-excluded flag")
        and list(_sequence(timing["profile_timing_order"], "profile timing order"))
        in [list(PROFILES), list(reversed(PROFILES)), ["baseline"]],
        "qualification timing method/order changed",
    )
    timing_contract = timing["update_timing_contract"]
    if representative:
        timing_contract = _mapping(timing_contract, "clean update timing contract")
        _exact_keys(
            timing_contract,
            {
                "shared_helper",
                "algorithm",
                "total_steps",
                "batch_size",
                "gradient_accumulation_steps",
                "warmup_steps",
                "max_grad_norm",
                "parameter_finite_check_interval",
                "seed",
                "candidate_choice_geometry_included",
                "hooks",
                "fresh_state_final_global_step",
                "fresh_state_final_micro_step",
                "standardized_worst_shape_training",
            },
            "clean update timing contract",
        )
        _require(
            {
                key: timing_contract[key]
                for key in timing_contract
                if key != "standardized_worst_shape_training"
            }
            == {
                "shared_helper": "goalzendo.training.train_steps",
                "algorithm": "sft",
                "total_steps": 8,
                "batch_size": int(PROFILE_CONTRACT[profile]["train_batch_size"]),
                "gradient_accumulation_steps": int(PROFILE_CONTRACT[profile]["gradient_accumulation_steps"]),
                "warmup_steps": 12,
                "max_grad_norm": 1.0,
                "parameter_finite_check_interval": 0,
                "seed": QUALIFICATION_ENGINEERING_SEED,
                "candidate_choice_geometry_included": True,
                "hooks": None,
                "fresh_state_final_global_step": 8,
                "fresh_state_final_micro_step": 8
                * int(PROFILE_CONTRACT[profile]["gradient_accumulation_steps"]),
            },
            "clean update timing does not bind the exact shared train_steps contract",
        )
        standardized_timing = _validate_standardized_worst_shape_training(
            timing_contract["standardized_worst_shape_training"],
            profile=profile,
            panel_id=panel_id,
        )
        host_contract = standardized_timing["tokenizer_host_envelope_contracts"]
        _same_number(
            timing["tokenizer_host_envelopes_8_updates_seconds"],
            host_contract["measured_seconds_sum"],
            "top-level tokenizer host-envelope timing sum",
        )
    else:
        _require(timing_contract is None, "registered-only process claims a clean timing contract")
    _require(
        math.isclose(measured, sum(components), rel_tol=1e-9, abs_tol=1e-6),
        "end-to-end timing does not equal its required components",
    )
    _require(
        timing["evaluation_boundary_count"] == (EVALUATION_BOUNDARY_COUNT if representative else 0),
        "timing boundary count differs from the execution tier",
    )
    _require(
        timing["projection_training_updates"] == PRODUCTION_UPDATES,
        "timing projection must target exactly 1,000 updates",
    )
    _require(
        timing["worker_mix"]
        == {
            "runs_per_worker": RUNS_PER_WORKER,
            "panel_runs": {panel: RUNS_PER_PANEL_PER_WORKER for panel in PANELS},
        },
        "timing projection does not use the exact 40-run worker mix",
    )
    _same_number(
        timing["projection_safety_multiplier"],
        PROJECTION_SAFETY_MULTIPLIER,
        "projection safety multiplier",
    )
    return record


def validate_process_record(record: Mapping[str, Any]) -> dict[str, Any]:
    """Validate one complete authenticated producer process record."""

    return copy.deepcopy(dict(_validate_process_record_internal(record)))


def _process_comparison_binding(record: Mapping[str, Any]) -> dict[str, Any]:
    body = {
        "process_uuid": record["identity"]["process_uuid"],
        "record_digest": record["record_digest"],
        "execution_benchmark_sha256": record["execution_benchmark"]["benchmark_digest"],
        "steps": [
            {
                "update": step["update"],
                "ordered_example_count": step["ordered_example_count"],
                "ordered_example_ids": copy.deepcopy(list(step["ordered_example_ids"])),
                "ordered_example_ids_sha256": step["ordered_example_ids_sha256"],
                "worst_sample_insertion": copy.deepcopy(dict(step["worst_sample_insertion"])),
                "rng_receipt": copy.deepcopy(dict(step["rng_receipt"])),
                "outputs_sha256": step["outputs_sha256"],
                "registered_production_shaped_parity_chunk": copy.deepcopy(
                    step["registered_production_shaped_parity_chunk"]
                ),
                "state_hashes": copy.deepcopy(dict(step["state_hashes"])),
                "vectors": copy.deepcopy(dict(step["vectors"])),
            }
            for step in record["steps"]
        ],
    }
    return _seal(body, "binding_sha256")


def _validate_process_comparison_binding(value: Any, label: str) -> Mapping[str, Any]:
    binding = _mapping(value, label)
    _exact_keys(
        binding,
        {
            "process_uuid",
            "record_digest",
            "execution_benchmark_sha256",
            "steps",
            "binding_sha256",
        },
        label,
    )
    try:
        parsed_process_uuid = uuid.UUID(str(binding["process_uuid"]))
    except ValueError as error:
        raise QualificationError(f"{label} process UUID is invalid") from error
    _require(str(parsed_process_uuid) == binding["process_uuid"], f"{label} process UUID is not canonical")
    _sha256(binding["record_digest"], f"{label} record digest")
    _sha256(binding["execution_benchmark_sha256"], f"{label} execution benchmark digest")
    steps = _sequence(binding["steps"], f"{label} steps")
    _require(len(steps) == len(UPDATES), f"{label} must bind exactly eight steps")
    for expected_update, raw_step in zip(UPDATES, steps, strict=True):
        step = _mapping(raw_step, f"{label} step {expected_update}")
        _exact_keys(
            step,
            {
                "update",
                "ordered_example_count",
                "ordered_example_ids",
                "ordered_example_ids_sha256",
                "worst_sample_insertion",
                "rng_receipt",
                "outputs_sha256",
                "registered_production_shaped_parity_chunk",
                "state_hashes",
                "vectors",
            },
            f"{label} step {expected_update}",
        )
        _require(step["update"] == expected_update, f"{label} steps are not ordered 1 through 8")
        _require(step["ordered_example_count"] == 50, f"{label} step does not bind 50 examples")
        ordered_ids = list(_sequence(step["ordered_example_ids"], f"{label} ordered IDs"))
        _require(
            len(ordered_ids) == 50 and step["ordered_example_ids_sha256"] == semantic_digest(ordered_ids),
            f"{label} explicit order binding differs",
        )
        _sha256(step["ordered_example_ids_sha256"], f"{label} ordered-data digest")
        insertion = _mapping(step["worst_sample_insertion"], f"{label} worst insertion")
        _require(
            insertion.get("final_order_sha256") == step["ordered_example_ids_sha256"],
            f"{label} worst insertion differs from order",
        )
        rng = _mapping(step["rng_receipt"], f"{label} RNG receipt")
        _exact_keys(
            rng,
            {"helper", "engineering_seed", "micro_steps", "scorer_calls_guarded"},
            f"{label} RNG receipt",
        )
        _sha256(step["outputs_sha256"], f"{label} output digest")
        state = _mapping(step["state_hashes"], f"{label} state hashes")
        _exact_keys(
            state,
            {"model_state_sha256", "optimizer_state_sha256", "combined_state_sha256", "outputs_sha256"},
            f"{label} state hashes",
        )
        for field in state:
            _sha256(state[field], f"{label} {field}")
        _require(state["outputs_sha256"] == step["outputs_sha256"], f"{label} output binding differs")
        vectors = _mapping(step["vectors"], f"{label} vectors")
        _exact_keys(vectors, VECTOR_KINDS, f"{label} vectors")
        for vector_kind in VECTOR_KINDS:
            _validate_vector_descriptor(vectors[vector_kind], f"{label} {vector_kind}")
    _verify_self_digest(binding, "binding_sha256", label)
    return binding


def canonical_process_comparison_binding(record: Mapping[str, Any]) -> dict[str, Any]:
    """Build the raw-vector-free canonical side binding for a paired comparison."""

    validated = _validate_process_record_internal(record)
    return _process_comparison_binding(validated)


def build_paired_execution_receipt(
    *,
    left_record: Mapping[str, Any],
    right_record: Mapping[str, Any],
    left_observed_binding: Mapping[str, Any],
    right_observed_binding: Mapping[str, Any],
    pair_capture_receipt_sha256: str,
    left_capture_mode: str,
    right_capture_mode: str,
) -> dict[str, Any]:
    """Bind an authenticated paired run or exact logical-identity replay.

    The observed bindings intentionally contain only ordered-data, output,
    state, and bounded vector-manifest facts.  Temporary raw vector chunks must
    be deleted before this receipt is constructed.
    """

    left = _validate_process_record_internal(left_record)
    right = _validate_process_record_internal(right_record)
    canonical_left = _process_comparison_binding(left)
    canonical_right = _process_comparison_binding(right)
    observed_left = _validate_process_comparison_binding(
        left_observed_binding,
        "left observed comparison binding",
    )
    observed_right = _validate_process_comparison_binding(
        right_observed_binding,
        "right observed comparison binding",
    )
    allowed_modes = {"canonical_process", "exact_logical_identity_replay"}
    _require(left_capture_mode in allowed_modes, "left paired-execution capture mode is invalid")
    _require(right_capture_mode in allowed_modes, "right paired-execution capture mode is invalid")
    _sha256(pair_capture_receipt_sha256, "paired-execution capture receipt digest")
    exact = bool(observed_left == canonical_left and observed_right == canonical_right)
    _require(exact, "paired execution does not exactly match the canonical process records")
    return {
        "actual_paired_execution": True,
        "pair_capture_receipt_sha256": pair_capture_receipt_sha256,
        "left_capture_mode": left_capture_mode,
        "right_capture_mode": right_capture_mode,
        "left_process_uuid": left["identity"]["process_uuid"],
        "right_process_uuid": right["identity"]["process_uuid"],
        "left_record_digest": left["record_digest"],
        "right_record_digest": right["record_digest"],
        "left_canonical_binding_sha256": canonical_left["binding_sha256"],
        "right_canonical_binding_sha256": canonical_right["binding_sha256"],
        "left_observed_binding_sha256": observed_left["binding_sha256"],
        "right_observed_binding_sha256": observed_right["binding_sha256"],
        "left_observed_binding": copy.deepcopy(dict(observed_left)),
        "right_observed_binding": copy.deepcopy(dict(observed_right)),
        "exact_match_to_canonical_process_records": True,
        "temporary_pairwise_chunk_files_deleted": True,
        "raw_vectors_persisted": False,
    }


def _identity_key(record: Mapping[str, Any]) -> tuple[str, str, str, str, str]:
    identity = record["identity"]
    return (
        str(identity["profile"]),
        str(identity["replicate"]),
        str(identity["panel_id"]),
        str(identity["law_family"]),
        str(identity["device_uuid"]),
    )


def _expected_process_keys(
    device_uuids: Sequence[str],
    *,
    profiles: Sequence[str] = PROFILES,
) -> set[tuple[str, str, str, str, str]]:
    return {
        (profile, replicate, panel, law, device)
        for profile in profiles
        for replicate in REPLICATES[profile]
        for panel in PANELS
        for law in LAW_FAMILIES
        for device in device_uuids
    }


def _comparison_process_keys(
    kind: str,
    panel: str,
    law: str,
    device: str,
) -> tuple[tuple[str, str, str, str, str], tuple[str, str, str, str, str]]:
    if kind == "baseline_vs_tuned":
        return (
            ("baseline", "primary", panel, law, device),
            ("tuned", "primary", panel, law, device),
        )
    if kind == "baseline_primary_vs_replay":
        return (
            ("baseline", "primary", panel, law, device),
            ("baseline", "replay", panel, law, device),
        )
    if kind == "tuned_primary_vs_replay":
        return (
            ("tuned", "primary", panel, law, device),
            ("tuned", "replay", panel, law, device),
        )
    raise QualificationError("unknown comparison kind")


def _validate_tuned_capacity_probe(
    value: Any,
    *,
    expected_device_uuids: Sequence[str],
) -> Mapping[str, Any]:
    probe = _mapping(value, "tuned capacity probe")
    _exact_keys(
        probe,
        {
            "schema",
            "schema_version",
            "qualification_branch",
            "status",
            "actual_model_execution",
            "profile_contract",
            "workload_contract",
            "numerical_execution",
            "gpu_uuids",
            "allowed_disqualifiers",
            "device_receipts",
            "failed_device_uuids",
            "outcomes_seen",
            "itt_ledger_created",
            "g01_launch_authorized",
            "probe_digest",
        },
        "tuned capacity probe",
    )
    _require(
        probe["schema"] == TUNED_CAPACITY_PROBE_SCHEMA
        and probe["schema_version"] == TUNED_CAPACITY_PROBE_SCHEMA_VERSION,
        "tuned capacity probe schema mismatch",
    )
    _verify_self_digest(probe, "probe_digest", "tuned capacity probe")
    branch = str(probe["qualification_branch"])
    _require(branch in QUALIFICATION_BRANCHES, "unknown qualification branch")
    _require(
        _boolean(probe["actual_model_execution"], "capacity probe actual-model flag"),
        "capacity probe is not actual-model execution",
    )
    _require(
        probe["profile_contract"] == PROFILE_CONTRACT["tuned"],
        "capacity probe tuned profile contract changed",
    )
    workload = _mapping(probe["workload_contract"], "capacity probe workload")
    _exact_keys(
        workload,
        {
            "panels",
            "train_batch_size",
            "gradient_accumulation_steps",
            "evaluation_batch_size_examples",
            "registered_evaluation_example_count",
            "production_evaluation_partition",
            "all_six_views",
            "both_laws",
            "corpus_max_train_sample_injected",
            "exact_worst_shape_batch_exercised",
        },
        "capacity probe workload",
    )
    _require(
        list(_sequence(workload["panels"], "capacity probe panels")) == list(PANELS),
        "capacity probe panels changed",
    )
    _require(
        workload["train_batch_size"] == 50
        and workload["gradient_accumulation_steps"] == 1
        and workload["evaluation_batch_size_examples"] == 128
        and _integer(
            workload["registered_evaluation_example_count"],
            "capacity probe registered example count",
            minimum=128,
        )
        >= 128,
        "capacity probe batch contract changed",
    )
    _require(
        workload["production_evaluation_partition"] == "example_chunks_then_flatten_prompt_views",
        "capacity probe evaluation partition changed",
    )
    for field in (
        "all_six_views",
        "both_laws",
        "corpus_max_train_sample_injected",
        "exact_worst_shape_batch_exercised",
    ):
        _require(_boolean(workload[field], f"capacity probe {field}"), f"capacity probe {field} is false")
    numerical = _mapping(probe["numerical_execution"], "capacity probe numerical execution")
    expected_numerical = {
        "configure_numerical_execution_called": True,
        "receipt": {
            "deterministic_algorithms": True,
            "deterministic_warn_only": False,
            "cudnn_benchmark": False,
            "cudnn_deterministic": True,
            "cuda_matmul_allow_tf32": False,
            "cudnn_allow_tf32": False,
            "float32_matmul_precision": "highest",
            "cublas_workspace_config": ":4096:8",
        },
    }
    _require(numerical == expected_numerical, "capacity probe numerical execution changed")
    device_uuids = [str(item) for item in _sequence(probe["gpu_uuids"], "capacity probe GPU UUIDs")]
    _require(device_uuids == list(expected_device_uuids), "capacity probe GPU UUIDs changed")
    _require(
        list(_sequence(probe["allowed_disqualifiers"], "capacity probe disqualifiers"))
        == list(TUNED_CAPACITY_DISQUALIFIERS),
        "capacity probe disqualifier taxonomy changed",
    )
    receipts = _sequence(probe["device_receipts"], "capacity probe device receipts")
    _require(len(receipts) == DEVICE_COUNT, "capacity probe requires four device receipts")
    normalized_receipts: list[Mapping[str, Any]] = []
    failed: list[str] = []
    for index, (expected_uuid, raw_receipt) in enumerate(zip(device_uuids, receipts, strict=True)):
        receipt = _mapping(raw_receipt, f"capacity probe device receipt {index}")
        _exact_keys(
            receipt,
            {
                "device_uuid",
                "device_name",
                "host_ordinal",
                "probe_cells",
                "peak_allocated_bytes",
                "peak_reserved_bytes",
                "status",
                "disqualifiers",
                "receipt_digest",
            },
            f"capacity probe device receipt {index}",
        )
        _verify_self_digest(receipt, "receipt_digest", f"capacity probe device receipt {index}")
        _require(
            receipt["device_uuid"] == expected_uuid, "capacity probe device receipts are not UUID sorted"
        )
        _require(
            isinstance(receipt["device_name"], str) and str(receipt["device_name"]).startswith("NVIDIA H200"),
            "capacity probe device is not H200",
        )
        _integer(receipt["host_ordinal"], "capacity probe host ordinal")
        cells = _sequence(receipt["probe_cells"], "capacity probe cells")
        expected_cell_identities = [(panel, law) for panel in PANELS for law in LAW_FAMILIES]
        _require(len(cells) == len(expected_cell_identities), "capacity probe device needs four cells")
        cell_disqualifiers: list[str] = []
        cell_allocated: list[int] = []
        cell_reserved: list[int] = []
        for cell_index, (raw_cell, expected_identity) in enumerate(
            zip(cells, expected_cell_identities, strict=True)
        ):
            cell = _mapping(raw_cell, f"capacity probe cell {index}/{cell_index}")
            _exact_keys(
                cell,
                {
                    "panel_id",
                    "law_family",
                    "device_uuid",
                    "actual_model_execution",
                    "isolated_subprocess",
                    "subprocess_argv",
                    "subprocess_argv_sha256",
                    "subprocess_exit_code",
                    "process_exit_resets_device",
                    "model_receipt_digest",
                    "resolved_config_sha256",
                    "workload_sha256",
                    "numerical_execution_sha256",
                    "fresh_model_and_optimizer",
                    "train_batch_exercised",
                    "gradient_clip_exercised",
                    "adamw_step_exercised",
                    "optimizer_moments_resident_during_evaluation",
                    "contiguous_evaluation_example_count",
                    "flattened_evaluation_prompt_count",
                    "cuda_synchronized_before_memory_read",
                    "finite_execution",
                    "peak_allocated_bytes",
                    "peak_reserved_bytes",
                    "status",
                    "disqualifier",
                    "failure_type_sha256",
                    "failure_message_sha256",
                    "log_binding",
                    "cell_digest",
                },
                f"capacity probe cell {index}/{cell_index}",
            )
            _verify_self_digest(cell, "cell_digest", f"capacity probe cell {index}/{cell_index}")
            _require(
                (cell["panel_id"], cell["law_family"]) == expected_identity
                and cell["device_uuid"] == expected_uuid,
                "capacity probe cell Cartesian identity changed",
            )
            for field in ("actual_model_execution", "isolated_subprocess", "process_exit_resets_device"):
                _require(
                    _boolean(cell[field], f"capacity probe cell {field}"), f"capacity cell {field} is false"
                )
            execution_flags = {
                field: _boolean(cell[field], f"capacity probe cell {field}")
                for field in (
                    "fresh_model_and_optimizer",
                    "train_batch_exercised",
                    "gradient_clip_exercised",
                    "adamw_step_exercised",
                    "optimizer_moments_resident_during_evaluation",
                    "cuda_synchronized_before_memory_read",
                )
            }
            argv = list(_sequence(cell["subprocess_argv"], "capacity probe subprocess argv"))
            _require(
                bool(argv)
                and all(isinstance(item, str) and item for item in argv)
                and cell["subprocess_argv_sha256"] == semantic_digest(argv),
                "capacity probe subprocess argv binding changed",
            )
            exit_code = _integer(cell["subprocess_exit_code"], "capacity probe exit code")
            _sha256(cell["model_receipt_digest"], "capacity probe model receipt digest")
            _sha256(cell["resolved_config_sha256"], "capacity probe resolved config digest")
            _sha256(cell["workload_sha256"], "capacity probe workload digest")
            _require(
                cell["numerical_execution_sha256"] == semantic_digest(expected_numerical["receipt"]),
                "capacity probe cell numerical-execution binding changed",
            )
            evaluation_examples = _integer(
                cell["contiguous_evaluation_example_count"],
                "capacity probe evaluation examples",
            )
            evaluation_prompts = _integer(
                cell["flattened_evaluation_prompt_count"],
                "capacity probe evaluation prompts",
            )
            finite_execution = _boolean(cell["finite_execution"], "capacity probe cell finite flag")
            allocated = _integer(cell["peak_allocated_bytes"], "capacity probe cell allocated bytes")
            reserved = _integer(cell["peak_reserved_bytes"], "capacity probe cell reserved bytes")
            _require(allocated <= reserved, "capacity probe cell allocated bytes exceed reserved bytes")
            cell_allocated.append(allocated)
            cell_reserved.append(reserved)
            status = str(cell["status"])
            _require(
                status in {"passed", "recoverable_capacity_failure"}, "capacity probe cell status changed"
            )
            disqualifier = cell["disqualifier"]
            if status == "passed":
                _require(
                    exit_code == 0
                    and disqualifier is None
                    and cell["failure_type_sha256"] is None
                    and cell["failure_message_sha256"] is None
                    and finite_execution
                    and reserved <= MAXIMUM_TUNED_RESERVED_BYTES,
                    "passing capacity probe cell contains a failure",
                )
                _require(
                    all(execution_flags.values())
                    and evaluation_examples == 128
                    and evaluation_prompts == 128 * len(TRAINING_VIEWS),
                    "passing capacity probe cell is incomplete",
                )
            else:
                _require(disqualifier in TUNED_CAPACITY_DISQUALIFIERS, "capacity failure is outside taxonomy")
                _sha256(cell["failure_type_sha256"], "capacity failure type digest")
                _sha256(cell["failure_message_sha256"], "capacity failure message digest")
                if disqualifier == "nonfinite_capacity_execution":
                    _require(not finite_execution, "nonfinite disqualifier has a finite receipt")
                if disqualifier == "reserved_memory_ceiling_exceeded":
                    _require(
                        reserved > MAXIMUM_TUNED_RESERVED_BYTES, "memory-ceiling disqualifier is unsupported"
                    )
                cell_disqualifiers.append(str(disqualifier))
            log = _mapping(cell["log_binding"], "capacity probe cell log binding")
            _exact_keys(
                log,
                {"file_sha256", "bytes", "mode", "retention"},
                "capacity probe cell log binding",
            )
            _sha256(log["file_sha256"], "capacity probe log digest")
            _integer(log["bytes"], "capacity probe log bytes")
            _require(
                log["mode"] == 0o400 and log["retention"] == "compact_digest_only_raw_log_deleted_after_hash",
                "capacity probe log mode/retention changed",
            )
        allocated = _integer(receipt["peak_allocated_bytes"], "capacity probe allocated bytes")
        reserved = _integer(receipt["peak_reserved_bytes"], "capacity probe reserved bytes")
        _require(
            allocated == max(cell_allocated) and reserved == max(cell_reserved),
            "capacity probe device peak memory is not derived from cells",
        )
        expected_disqualifiers = sorted(set(cell_disqualifiers))
        _require(
            list(_sequence(receipt["disqualifiers"], "capacity probe device disqualifiers"))
            == expected_disqualifiers,
            "capacity probe device disqualifiers are not derived from cells",
        )
        expected_device_status = "passed" if not cell_disqualifiers else "recoverable_capacity_failure"
        _require(receipt["status"] == expected_device_status, "capacity probe device status is inconsistent")
        if cell_disqualifiers:
            failed.append(expected_uuid)
        normalized_receipts.append(receipt)
    expected_status = "passed" if not failed else "recoverable_capacity_failure"
    expected_branch = "tuned_probe_passed_full" if not failed else "tuned_capacity_fallback_baseline"
    _require(probe["status"] == expected_status, "capacity probe aggregate status is inconsistent")
    _require(branch == expected_branch, "capacity probe branch is inconsistent")
    _require(
        list(_sequence(probe["failed_device_uuids"], "capacity probe failed devices")) == failed,
        "capacity probe failed-device list is inconsistent",
    )
    _require(
        probe["outcomes_seen"] is False
        and probe["itt_ledger_created"] is False
        and probe["g01_launch_authorized"] is False,
        "capacity probe crossed a scientific boundary",
    )
    return probe


def validate_tuned_capacity_probe(
    probe: Mapping[str, Any],
    *,
    expected_device_uuids: Sequence[str],
) -> dict[str, Any]:
    """Independently replay the sealed sixteen-cell tuned capacity probe."""

    return copy.deepcopy(
        dict(
            _validate_tuned_capacity_probe(
                probe,
                expected_device_uuids=expected_device_uuids,
            )
        )
    )


def _output_comparison(left: Sequence[Any], right: Sequence[Any]) -> dict[str, Any]:
    _require(len(left) == len(right) == len(TRAINING_VIEWS), "comparison output cardinality mismatch")
    sample_identity_exact = True
    maximum_score_difference = 0.0
    maximum_probability_difference = 0.0
    actions_identical = True
    for left_output, right_output in zip(left, right, strict=True):
        lhs = _mapping(left_output, "left comparison output")
        rhs = _mapping(right_output, "right comparison output")
        sample_identity_exact = sample_identity_exact and (
            lhs["view"],
            lhs["sample_id"],
            lhs["token_length"],
        ) == (rhs["view"], rhs["sample_id"], rhs["token_length"])
        maximum_score_difference = max(
            maximum_score_difference,
            *(
                abs(float(a) - float(b))
                for a, b in zip(lhs["normalized_log_scores"], rhs["normalized_log_scores"], strict=True)
            ),
        )
        maximum_probability_difference = max(
            maximum_probability_difference,
            *(
                abs(float(a) - float(b))
                for a, b in zip(lhs["probabilities"], rhs["probabilities"], strict=True)
            ),
        )
        actions_identical = actions_identical and (lhs["action_index"], lhs["action_label"]) == (
            rhs["action_index"],
            rhs["action_label"],
        )
    return {
        "sample_identity_exact": sample_identity_exact,
        "maximum_normalized_score_difference": maximum_score_difference,
        "maximum_probability_difference": maximum_probability_difference,
        "actions_identical": actions_identical,
    }


def _validate_paired_execution_receipt(
    value: Any,
    left_record: Mapping[str, Any],
    right_record: Mapping[str, Any],
) -> Mapping[str, Any]:
    receipt = _mapping(value, "paired-execution receipt")
    _exact_keys(
        receipt,
        {
            "actual_paired_execution",
            "pair_capture_receipt_sha256",
            "left_capture_mode",
            "right_capture_mode",
            "left_process_uuid",
            "right_process_uuid",
            "left_record_digest",
            "right_record_digest",
            "left_canonical_binding_sha256",
            "right_canonical_binding_sha256",
            "left_observed_binding_sha256",
            "right_observed_binding_sha256",
            "left_observed_binding",
            "right_observed_binding",
            "exact_match_to_canonical_process_records",
            "temporary_pairwise_chunk_files_deleted",
            "raw_vectors_persisted",
        },
        "paired-execution receipt",
    )
    _require(
        _boolean(receipt["actual_paired_execution"], "actual paired-execution flag"),
        "comparison is not backed by an actual paired execution",
    )
    _sha256(receipt["pair_capture_receipt_sha256"], "paired-execution capture receipt digest")
    allowed_modes = {"canonical_process", "exact_logical_identity_replay"}
    _require(receipt["left_capture_mode"] in allowed_modes, "left paired-execution mode is invalid")
    _require(receipt["right_capture_mode"] in allowed_modes, "right paired-execution mode is invalid")
    canonical_left = _process_comparison_binding(left_record)
    canonical_right = _process_comparison_binding(right_record)
    observed_left = _validate_process_comparison_binding(
        receipt["left_observed_binding"],
        "left observed comparison binding",
    )
    observed_right = _validate_process_comparison_binding(
        receipt["right_observed_binding"],
        "right observed comparison binding",
    )
    _require(
        receipt["left_process_uuid"] == left_record["identity"]["process_uuid"]
        and receipt["right_process_uuid"] == right_record["identity"]["process_uuid"],
        "paired-execution process UUID links are incorrect",
    )
    _require(
        receipt["left_record_digest"] == left_record["record_digest"]
        and receipt["right_record_digest"] == right_record["record_digest"],
        "paired-execution record digest links are incorrect",
    )
    _require(
        receipt["left_canonical_binding_sha256"] == canonical_left["binding_sha256"]
        and receipt["right_canonical_binding_sha256"] == canonical_right["binding_sha256"],
        "paired-execution canonical binding digests are incorrect",
    )
    _require(
        receipt["left_observed_binding_sha256"] == observed_left["binding_sha256"]
        and receipt["right_observed_binding_sha256"] == observed_right["binding_sha256"],
        "paired-execution observed binding digests are incorrect",
    )
    exact = bool(observed_left == canonical_left and observed_right == canonical_right)
    _require(
        _boolean(
            receipt["exact_match_to_canonical_process_records"],
            "paired-execution exact-match flag",
        )
        == exact,
        "paired-execution exact-match flag is inconsistent",
    )
    _require(exact, "paired execution does not exactly match the canonical process records")
    _require(
        _boolean(
            receipt["temporary_pairwise_chunk_files_deleted"],
            "paired-execution temporary cleanup flag",
        ),
        "temporary paired-execution vector chunks were not deleted",
    )
    _require(
        not _boolean(receipt["raw_vectors_persisted"], "paired-execution raw-vector flag"),
        "paired-execution receipt persisted raw vectors",
    )
    return receipt


def _validate_comparison_record_internal(
    value: Any,
    records: Mapping[tuple[str, str, str, str, str], Mapping[str, Any]],
) -> Mapping[str, Any]:
    comparison = _mapping(value, "comparison record")
    _exact_keys(
        comparison,
        {
            "schema",
            "schema_version",
            "kind",
            "panel_id",
            "law_family",
            "device_uuid",
            "left_record_digest",
            "right_record_digest",
            "paired_execution_receipt",
            "steps",
            "comparison_digest",
        },
        "comparison record",
    )
    _require(
        comparison["schema"] == COMPARISON_SCHEMA
        and comparison["schema_version"] == COMPARISON_SCHEMA_VERSION,
        "comparison record schema mismatch",
    )
    _verify_self_digest(comparison, "comparison_digest", "comparison record")
    kind = str(comparison["kind"])
    panel = str(comparison["panel_id"])
    law = str(comparison["law_family"])
    device = str(comparison["device_uuid"])
    _require(kind in COMPARISON_KINDS, "comparison kind is unknown")
    _require(panel in PANELS and law in LAW_FAMILIES, "comparison panel or Law family is unknown")
    left_key, right_key = _comparison_process_keys(kind, panel, law, device)
    _require(left_key in records and right_key in records, "comparison process pair is absent")
    left_record = records[left_key]
    right_record = records[right_key]
    _require(
        comparison["left_record_digest"] == left_record["record_digest"]
        and comparison["right_record_digest"] == right_record["record_digest"],
        "comparison record digest links are incorrect",
    )
    _validate_paired_execution_receipt(
        comparison["paired_execution_receipt"],
        left_record,
        right_record,
    )
    steps = _sequence(comparison["steps"], "comparison steps")
    _require(len(steps) == len(UPDATES), "comparison must contain exactly eight updates")
    for expected_update, raw_step, left_step, right_step in zip(
        UPDATES,
        steps,
        left_record["steps"],
        right_record["steps"],
        strict=True,
    ):
        step = _mapping(raw_step, f"comparison step {expected_update}")
        _exact_keys(step, {"update", "outputs", "vectors"}, f"comparison step {expected_update}")
        _require(step["update"] == expected_update, "comparison updates are not ordered 1 through 8")
        output = _mapping(step["outputs"], f"comparison step {expected_update} outputs")
        _exact_keys(
            output,
            {
                "left_outputs_sha256",
                "right_outputs_sha256",
                "sample_identity_exact",
                "maximum_normalized_score_difference",
                "maximum_probability_difference",
                "actions_identical",
            },
            f"comparison step {expected_update} outputs",
        )
        _require(
            output["left_outputs_sha256"] == left_step["outputs_sha256"]
            and output["right_outputs_sha256"] == right_step["outputs_sha256"],
            "comparison output digest links are incorrect",
        )
        recomputed = _output_comparison(left_step["outputs"], right_step["outputs"])
        _require(
            _boolean(output["sample_identity_exact"], "comparison sample identity flag")
            == recomputed["sample_identity_exact"],
            "comparison sample identity flag is incorrect",
        )
        _require(
            _boolean(output["actions_identical"], "comparison action flag")
            == recomputed["actions_identical"],
            "comparison action flag is incorrect",
        )
        _same_number(
            output["maximum_normalized_score_difference"],
            float(recomputed["maximum_normalized_score_difference"]),
            "comparison maximum score difference",
        )
        _same_number(
            output["maximum_probability_difference"],
            float(recomputed["maximum_probability_difference"]),
            "comparison maximum probability difference",
        )
        vectors = _mapping(step["vectors"], f"comparison step {expected_update} vectors")
        _exact_keys(vectors, VECTOR_KINDS, f"comparison step {expected_update} vectors")
        for vector_kind in VECTOR_KINDS:
            metric = _validate_vector_metric(
                vectors[vector_kind],
                f"comparison step {expected_update} {vector_kind}",
            )
            left_manifest = left_record["trainable_parameters"]
            right_manifest = right_record["trainable_parameters"]
            _require(
                left_manifest == right_manifest,
                "comparison sides use different trainable-parameter manifests",
            )
            expected_rows = [(entry["parameter_key"], entry["numel"]) for entry in left_manifest["entries"]]
            observed_rows = [
                (row["parameter_key"], row["element_count"]) for row in metric["accumulator_rows"]
            ]
            _require(
                observed_rows == expected_rows,
                "comparison accumulator rows do not exactly cover the trainable parameters",
            )
            _require(
                metric["element_count"] == TRAINABLE_NUMEL[panel],
                "comparison vector does not cover the exact frozen trainable parameter count",
            )
            left_descriptor = left_step["vectors"][vector_kind]
            right_descriptor = right_step["vectors"][vector_kind]
            _require(
                metric["parameter_keys_sha256"]
                == left_descriptor["parameter_keys_sha256"]
                == right_descriptor["parameter_keys_sha256"]
                and metric["element_count"]
                == left_descriptor["element_count"]
                == right_descriptor["element_count"],
                "comparison parameter-key binding is incorrect",
            )
            _require(
                metric["left_native_chunk_manifest_sha256"] == left_descriptor["native_chunk_manifest_sha256"]
                and metric["right_native_chunk_manifest_sha256"]
                == right_descriptor["native_chunk_manifest_sha256"],
                "comparison vector-value digest link is incorrect",
            )
            _require(
                left_descriptor["chunk_count"]
                == len(metric["accumulator_rows"])
                == right_descriptor["chunk_count"],
                "comparison accumulator chunks are not linked to process vector descriptors",
            )
            _same_number(metric["left_norm"], float(left_descriptor["float64_norm"]), "left vector norm")
            _same_number(metric["right_norm"], float(right_descriptor["float64_norm"]), "right vector norm")
    return comparison


@dataclass(frozen=True)
class _ValidatedEvidence:
    payload: Mapping[str, Any]
    qualification_branch: str
    tuned_capacity_probe: Mapping[str, Any]
    records: tuple[Mapping[str, Any], ...]
    comparisons: tuple[Mapping[str, Any], ...]
    records_by_key: Mapping[tuple[str, str, str, str, str], Mapping[str, Any]]
    device_uuids: tuple[str, ...]


def _validate_evidence_internal(value: Any) -> _ValidatedEvidence:
    evidence = _mapping(value, "qualification evidence")
    _exact_keys(
        evidence,
        {
            "schema",
            "schema_version",
            "actual_model_execution",
            "qualification_branch",
            "tuned_capacity_probe",
            "evidence_source",
            "engineering_scope",
            "freeze_binding",
            "provision_binding",
            "model_receipt_bindings",
            "model_integration_audit_bindings",
            "storage_contract",
            "expected_device_uuids",
            "process_records",
            "comparison_records",
            "evidence_digest",
        },
        "qualification evidence",
    )
    _require(
        evidence["schema"] == EVIDENCE_SCHEMA and evidence["schema_version"] == EVIDENCE_SCHEMA_VERSION,
        "qualification evidence schema mismatch",
    )
    _verify_self_digest(evidence, "evidence_digest", "qualification evidence")
    _require(
        _boolean(evidence["actual_model_execution"], "evidence actual-model flag"),
        "qualification evidence does not attest actual-model execution",
    )
    source = _mapping(evidence["evidence_source"], "evidence source")
    _exact_keys(
        source,
        {
            "kind",
            "authentication",
            "aggregator_role",
            "capture_command_sha256",
            "capture_implementation_sha256",
            "records_manifest_sha256",
        },
        "evidence source",
    )
    for key, expected in EVIDENCE_BOUNDARY.items():
        _require(source[key] == expected, f"evidence source {key} mismatch")
    _sha256(source["capture_command_sha256"], "capture-command digest")
    _sha256(source["capture_implementation_sha256"], "capture-implementation digest")
    _engineering_scope(evidence["engineering_scope"], "qualification engineering scope")
    _validate_bindings(evidence)
    devices_raw = _sequence(evidence["expected_device_uuids"], "expected H200 UUIDs")
    devices = tuple(str(item) for item in devices_raw)
    _require(
        len(devices) == DEVICE_COUNT
        and list(devices) == sorted(set(devices))
        and all(GPU_UUID_PATTERN.fullmatch(item) is not None for item in devices),
        "qualification requires exactly four sorted distinct H200 GPU UUIDs",
    )
    branch = str(evidence["qualification_branch"])
    _require(branch in QUALIFICATION_BRANCHES, "unknown qualification branch")
    capacity_probe = _validate_tuned_capacity_probe(
        evidence["tuned_capacity_probe"],
        expected_device_uuids=devices,
    )
    _require(
        capacity_probe["qualification_branch"] == branch,
        "evidence branch differs from tuned capacity probe",
    )
    full_branch = branch == "tuned_probe_passed_full"
    expected_profiles = PROFILES if full_branch else ("baseline",)
    expected_process_count = PROCESS_RECORD_COUNT if full_branch else BASELINE_FALLBACK_PROCESS_RECORD_COUNT
    expected_comparison_count = (
        COMPARISON_RECORD_COUNT if full_branch else BASELINE_FALLBACK_COMPARISON_RECORD_COUNT
    )

    raw_records = _sequence(evidence["process_records"], "qualification process records")
    _require(
        len(raw_records) == expected_process_count,
        f"qualification branch requires exactly {expected_process_count} process records",
    )
    records = tuple(_validate_process_record_internal(item) for item in raw_records)
    records_by_key: dict[tuple[str, str, str, str, str], Mapping[str, Any]] = {}
    process_uuids: set[str] = set()
    for record in records:
        identity_key = _identity_key(record)
        _require(identity_key not in records_by_key, "duplicate qualification process identity")
        records_by_key[identity_key] = record
        process_uuid = str(record["identity"]["process_uuid"])
        _require(process_uuid not in process_uuids, "qualification process UUID was reused")
        process_uuids.add(process_uuid)
        _require(
            record["engineering_scope"] == evidence["engineering_scope"], "process engineering scope differs"
        )
    _require(
        set(records_by_key) == _expected_process_keys(devices, profiles=expected_profiles),
        "qualification process records do not form the exact branch Cartesian product",
    )
    for device_index, device in enumerate(devices):
        expected_order = (
            list(PROFILES if device_index % 2 == 0 else tuple(reversed(PROFILES)))
            if full_branch
            else ["baseline"]
        )
        observed_orders = {
            tuple(record["timing"]["profile_timing_order"])
            for record in records
            if record["identity"]["device_uuid"] == device
        }
        _require(
            observed_orders == {tuple(expected_order)},
            "profile timing order is not counterbalanced across sorted GPU UUIDs",
        )
    timing_representatives = [
        record for record in records if record["execution_benchmark"]["timing_representative"] is True
    ]
    _require(
        len(timing_representatives) == 8 * len(expected_profiles),
        "qualification branch has the wrong number of production timing representatives",
    )
    representative_keys = {
        (
            record["identity"]["profile"],
            record["identity"]["panel_id"],
            record["identity"]["device_uuid"],
        )
        for record in timing_representatives
    }
    _require(
        representative_keys
        == {
            (profile, panel, device)
            for profile in expected_profiles
            for panel in PANELS
            for device in devices
        }
        and all(record["identity"]["replicate"] == "primary" for record in timing_representatives),
        "production timing representatives are not one primary worst workload per profile/panel/GPU",
    )
    for panel in PANELS:
        manifests = {
            canonical_json_bytes(record["trainable_parameters"])
            for record in records
            if record["identity"]["panel_id"] == panel
        }
        _require(
            len(manifests) == 1,
            f"{panel} trainable-parameter manifest differs across processes",
        )

    raw_comparisons = _sequence(evidence["comparison_records"], "qualification comparison records")
    _require(
        len(raw_comparisons) == expected_comparison_count,
        "qualification branch has the wrong numeric comparison cardinality",
    )
    comparisons = tuple(
        _validate_comparison_record_internal(item, records_by_key) for item in raw_comparisons
    )
    comparison_keys = {
        (
            str(item["kind"]),
            str(item["panel_id"]),
            str(item["law_family"]),
            str(item["device_uuid"]),
        )
        for item in comparisons
    }
    expected_comparison_keys = (
        {("baseline_vs_tuned", panel, law, devices[0]) for panel in PANELS for law in LAW_FAMILIES}
        if full_branch
        else set()
    )
    _require(
        len(comparison_keys) == expected_comparison_count and comparison_keys == expected_comparison_keys,
        "qualification numeric comparisons do not match the branch contract",
    )
    manifest = semantic_digest(
        {
            "qualification_branch": branch,
            "tuned_capacity_probe_digest": capacity_probe["probe_digest"],
            "process_record_digests": sorted(str(item["record_digest"]) for item in records),
            "comparison_record_digests": sorted(str(item["comparison_digest"]) for item in comparisons),
        }
    )
    _require(source["records_manifest_sha256"] == manifest, "evidence record-manifest digest mismatch")
    accumulator_row_count = sum(
        len(metric["accumulator_rows"])
        for comparison in comparisons
        for step in comparison["steps"]
        for metric in step["vectors"].values()
    )
    minimum_inline_bytes = len(
        canonical_json_bytes(
            {
                "process_records": list(raw_records),
                "comparison_records": list(raw_comparisons),
            }
        )
    )
    _validate_storage_contract(
        evidence["storage_contract"],
        minimum_inline_bytes=minimum_inline_bytes,
        accumulator_row_count=accumulator_row_count,
    )
    return _ValidatedEvidence(
        payload=evidence,
        qualification_branch=branch,
        tuned_capacity_probe=capacity_probe,
        records=records,
        comparisons=comparisons,
        records_by_key=records_by_key,
        device_uuids=devices,
    )


def validate_qualification_evidence(evidence: Mapping[str, Any]) -> dict[str, Any]:
    """Validate a full or frozen tuned-capacity-fallback evidence bundle."""

    validated = _validate_evidence_internal(evidence)
    return copy.deepcopy(dict(validated.payload))


def _record_common_passed(record: Mapping[str, Any]) -> bool:
    initial = record["initial_state"]
    stochastic = record["stochastic_modules"]
    coverage = record["coverage"]
    memory = record["memory"]
    worst = coverage["worst_case_token_sample"]
    return bool(
        record["actual_model_execution"]
        and initial["fresh_model_instance"]
        and initial["fresh_optimizer_instance"]
        and stochastic["enumeration_complete"]
        and stochastic["dropout_inactive"]
        and stochastic["all_stochastic_modules_inactive"]
        and worst["included_in_every_update"]
        and memory["peak_stats_reset_before_run"]
    )


def _initial_state_exact(records: Sequence[Mapping[str, Any]]) -> bool:
    for panel in PANELS:
        bindings = {
            (
                item["initial_state"]["model_snapshot_sha256"],
                item["initial_state"]["model_state_sha256"],
                item["initial_state"]["optimizer_state_sha256"],
            )
            for item in records
            if item["identity"]["panel_id"] == panel
        }
        if len(bindings) != 1:
            return False
    return True


def _data_order_exact(records: Sequence[Mapping[str, Any]]) -> bool:
    for panel in PANELS:
        for law in LAW_FAMILIES:
            group = [
                item
                for item in records
                if item["identity"]["panel_id"] == panel and item["identity"]["law_family"] == law
            ]
            dataset_bindings = {
                (item["coverage"]["dataset_sha256"], item["coverage"]["engineering_seed"]) for item in group
            }
            worst_bindings = {semantic_digest(item["coverage"]["worst_case_token_sample"]) for item in group}
            if len(dataset_bindings) != 1 or len(worst_bindings) != 1:
                return False
            for update_index in range(len(UPDATES)):
                step_bindings = {
                    (
                        item["steps"][update_index]["ordered_example_count"],
                        item["steps"][update_index]["ordered_example_ids_sha256"],
                    )
                    for item in group
                }
                if len(step_bindings) != 1:
                    return False
    return True


def _comparison_summary(comparisons: Sequence[Mapping[str, Any]], kind: str) -> dict[str, Any]:
    selected = [item for item in comparisons if item["kind"] == kind]
    maximum_score_difference = 0.0
    maximum_probability_difference = 0.0
    actions_identical = True
    sample_identity_exact = True
    vectors: dict[str, dict[str, float]] = {
        vector_kind: {"minimum_cosine": 1.0, "maximum_relative_l2": 0.0} for vector_kind in VECTOR_KINDS
    }
    vector_comparison_count = 0
    for comparison in selected:
        for step in comparison["steps"]:
            output = step["outputs"]
            maximum_score_difference = max(
                maximum_score_difference,
                float(output["maximum_normalized_score_difference"]),
            )
            maximum_probability_difference = max(
                maximum_probability_difference,
                float(output["maximum_probability_difference"]),
            )
            actions_identical = actions_identical and bool(output["actions_identical"])
            sample_identity_exact = sample_identity_exact and bool(output["sample_identity_exact"])
            for vector_kind in VECTOR_KINDS:
                metric = step["vectors"][vector_kind]
                vectors[vector_kind]["minimum_cosine"] = min(
                    vectors[vector_kind]["minimum_cosine"],
                    float(metric["cosine"]),
                )
                vectors[vector_kind]["maximum_relative_l2"] = max(
                    vectors[vector_kind]["maximum_relative_l2"],
                    float(metric["relative_l2"]),
                )
                vector_comparison_count += 1
    return {
        "comparison_count": len(selected),
        "step_count": sum(len(item["steps"]) for item in selected),
        "vector_comparison_count": vector_comparison_count,
        "maximum_normalized_score_difference": maximum_score_difference,
        "maximum_probability_difference": maximum_probability_difference,
        "actions_identical": actions_identical,
        "sample_identity_exact": sample_identity_exact,
        "vectors": vectors,
    }


def _exact_profile_replay(
    validated: _ValidatedEvidence,
    profile: str,
    *,
    require_output_hashes: bool,
) -> bool:
    for panel in PANELS:
        for law in LAW_FAMILIES:
            for device in validated.device_uuids:
                primary = validated.records_by_key[(profile, "primary", panel, law, device)]
                replay = validated.records_by_key[(profile, "replay", panel, law, device)]
                if primary["initial_state"] != replay["initial_state"]:
                    return False
                if (
                    primary["execution_benchmark"]["registered_production_shaped_parity_chunk"]
                    != replay["execution_benchmark"]["registered_production_shaped_parity_chunk"]
                ):
                    return False
                for primary_step, replay_step in zip(primary["steps"], replay["steps"], strict=True):
                    if (
                        primary_step["ordered_example_ids_sha256"]
                        != replay_step["ordered_example_ids_sha256"]
                        or primary_step["ordered_example_ids"] != replay_step["ordered_example_ids"]
                        or primary_step["worst_sample_insertion"] != replay_step["worst_sample_insertion"]
                        or primary_step["rng_receipt"] != replay_step["rng_receipt"]
                        or primary_step["registered_production_shaped_parity_chunk"]
                        != replay_step["registered_production_shaped_parity_chunk"]
                    ):
                        return False
                    for field in (
                        "model_state_sha256",
                        "optimizer_state_sha256",
                        "combined_state_sha256",
                    ):
                        if primary_step["state_hashes"][field] != replay_step["state_hashes"][field]:
                            return False
                    if require_output_hashes and (
                        primary_step["outputs_sha256"] != replay_step["outputs_sha256"]
                        or primary_step["state_hashes"]["outputs_sha256"]
                        != replay_step["state_hashes"]["outputs_sha256"]
                    ):
                        return False
                    for vector_kind in VECTOR_KINDS:
                        if primary_step["vectors"][vector_kind] != replay_step["vectors"][vector_kind]:
                            return False
    return True


def _digest_exact_replay_summary(validated: _ValidatedEvidence, profile: str) -> dict[str, Any]:
    """Summarize sixteen replay edges without retaining redundant raw accumulators."""

    exact = _exact_profile_replay(validated, profile, require_output_hashes=True)
    return {
        "comparison_count": 16,
        "step_count": 16 * len(UPDATES),
        "vector_comparison_count": 16 * len(UPDATES) * len(VECTOR_KINDS),
        "maximum_normalized_score_difference": 0.0 if exact else 1.0,
        "maximum_probability_difference": 0.0 if exact else 1.0,
        "actions_identical": exact,
        "sample_identity_exact": exact,
        "vectors": {
            vector_kind: {
                "minimum_cosine": 1.0 if exact else 0.0,
                "maximum_relative_l2": 0.0 if exact else 1.0,
            }
            for vector_kind in VECTOR_KINDS
        },
    }


def _unavailable_comparison_summary() -> dict[str, Any]:
    return {
        "comparison_count": 0,
        "step_count": 0,
        "vector_comparison_count": 0,
        "maximum_normalized_score_difference": 0.0,
        "maximum_probability_difference": 0.0,
        "actions_identical": False,
        "sample_identity_exact": False,
        "vectors": {
            vector_kind: {"minimum_cosine": 0.0, "maximum_relative_l2": 1.0} for vector_kind in VECTOR_KINDS
        },
    }


def _unavailable_cross_gpu_equivalence(profile: str) -> dict[str, Any]:
    return {
        "profile": profile,
        "group_count": 0,
        "devices_per_group": DEVICE_COUNT,
        "groups": [],
        "exact_ordered_example_ids": False,
        "exact_model_optimizer_combined_state_hashes": False,
        "exact_output_hashes": False,
        "exact_vector_chunk_manifests": False,
        "passed": False,
    }


def _unavailable_registered_evaluation_parity() -> dict[str, Any]:
    return {
        "states": list(REGISTERED_PARITY_STATES),
        "group_count": 0,
        "shape_authentication_group_count": 0,
        "numeric_parity_group_count": 0,
        "row_comparison_count": 0,
        "single_registered_partition_exact": False,
        "row_ids_exact": False,
        "maximum_normalized_score_difference": 0.0,
        "maximum_probability_difference": 0.0,
        "actions_identical": False,
        "passed": False,
    }


def _registered_evaluation_parity(validated: _ValidatedEvidence) -> dict[str, Any]:
    maximum_score = 0.0
    maximum_probability = 0.0
    actions_identical = True
    row_ids_exact = True
    row_count = 0
    group_count = 0
    shape_group_count = 0
    numeric_group_count = 0
    partition_exact = True

    def chunks(record: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
        result = {"initial": record["execution_benchmark"]["registered_production_shaped_parity_chunk"]}
        result.update(
            {
                f"after_update_{step['update']}": step["registered_production_shaped_parity_chunk"]
                for step in record["steps"]
                if step["update"] in REGISTERED_PARITY_UPDATE_CHECKPOINTS
            }
        )
        _require(
            tuple(result) == REGISTERED_PARITY_STATES,
            "registered parity state construction is incomplete or out of order",
        )
        return result

    for panel in PANELS:
        for law in LAW_FAMILIES:
            for device in validated.device_uuids:
                baseline = validated.records_by_key[("baseline", "primary", panel, law, device)]
                tuned = validated.records_by_key[("tuned", "primary", panel, law, device)]
                baseline_chunks = chunks(baseline)
                tuned_chunks = chunks(tuned)
                for state in REGISTERED_PARITY_STATES:
                    left_chunk = baseline_chunks[state]
                    right_chunk = tuned_chunks[state]
                    group_count += 1
                    partition_fields = (
                        "measurement_state",
                        "prompt_count",
                        "example_count",
                        "production_example_partition_count",
                        "prompt_sha256",
                        "contiguous_example_chunk",
                        "prompt_views_per_example",
                        "production_bank_name",
                        "production_bank_start_index",
                        "example_ids",
                        "example_ids_sha256",
                        "token_shapes",
                        "token_shape_sha256",
                    )
                    partition_exact = partition_exact and all(
                        left_chunk[field] == right_chunk[field] for field in partition_fields
                    )
                    if state == "initial":
                        shape_group_count += 1
                        continue
                    numeric_group_count += 1
                    left_rows = left_chunk["rows"]
                    right_rows = right_chunk["rows"]
                    row_ids_exact = row_ids_exact and [row["row_id"] for row in left_rows] == [
                        row["row_id"] for row in right_rows
                    ]
                    if len(left_rows) != len(right_rows):
                        row_ids_exact = False
                        continue
                    row_count += len(left_rows)
                    for left, right in zip(left_rows, right_rows, strict=True):
                        maximum_score = max(
                            maximum_score,
                            *(
                                abs(float(a) - float(b))
                                for a, b in zip(
                                    left["normalized_log_scores"],
                                    right["normalized_log_scores"],
                                    strict=True,
                                )
                            ),
                        )
                        maximum_probability = max(
                            maximum_probability,
                            *(
                                abs(float(a) - float(b))
                                for a, b in zip(left["probabilities"], right["probabilities"], strict=True)
                            ),
                        )
                        actions_identical = actions_identical and (
                            left["action_index"],
                            left["action_label"],
                        ) == (right["action_index"], right["action_label"])
    passed = bool(
        group_count == 16 * len(REGISTERED_PARITY_STATES)
        and shape_group_count == 16
        and numeric_group_count == 16 * len(REGISTERED_PARITY_UPDATE_CHECKPOINTS)
        and partition_exact
        and row_ids_exact
        and actions_identical
        and maximum_score <= MAXIMUM_SCORE_DIFFERENCE
        and maximum_probability <= MAXIMUM_PROBABILITY_DIFFERENCE
    )
    return {
        "states": list(REGISTERED_PARITY_STATES),
        "group_count": group_count,
        "shape_authentication_group_count": shape_group_count,
        "numeric_parity_group_count": numeric_group_count,
        "row_comparison_count": row_count,
        "single_registered_partition_exact": partition_exact,
        "row_ids_exact": row_ids_exact,
        "maximum_normalized_score_difference": maximum_score,
        "maximum_probability_difference": maximum_probability,
        "actions_identical": actions_identical,
        "passed": passed,
    }


def _cross_gpu_equivalence(validated: _ValidatedEvidence, profile: str) -> dict[str, Any]:
    groups: list[dict[str, Any]] = []
    for replicate in REPLICATES[profile]:
        for panel in PANELS:
            for law in LAW_FAMILIES:
                records = [
                    validated.records_by_key[(profile, replicate, panel, law, device)]
                    for device in validated.device_uuids
                ]
                for step_index, update in enumerate(UPDATES):
                    device_bindings: list[dict[str, Any]] = []
                    for device, record in zip(validated.device_uuids, records, strict=True):
                        step = record["steps"][step_index]
                        device_bindings.append(
                            {
                                "device_uuid": device,
                                "ordered_example_count": step["ordered_example_count"],
                                "ordered_example_ids_sha256": step["ordered_example_ids_sha256"],
                                "outputs_sha256": step["outputs_sha256"],
                                "registered_parity_chunk_sha256": semantic_digest(
                                    step["registered_production_shaped_parity_chunk"]
                                ),
                                "execution_benchmark_sha256": record["execution_benchmark"][
                                    "benchmark_digest"
                                ],
                                "state_hashes": copy.deepcopy(dict(step["state_hashes"])),
                                "vectors": copy.deepcopy(dict(step["vectors"])),
                            }
                        )
                    ordered_exact = (
                        len(
                            {
                                (
                                    item["ordered_example_count"],
                                    item["ordered_example_ids_sha256"],
                                )
                                for item in device_bindings
                            }
                        )
                        == 1
                    )
                    state_exact = (
                        len(
                            {
                                (
                                    item["state_hashes"]["model_state_sha256"],
                                    item["state_hashes"]["optimizer_state_sha256"],
                                    item["state_hashes"]["combined_state_sha256"],
                                )
                                for item in device_bindings
                            }
                        )
                        == 1
                    )
                    outputs_exact = (
                        len(
                            {
                                (
                                    item["outputs_sha256"],
                                    item["state_hashes"]["outputs_sha256"],
                                    item["execution_benchmark_sha256"],
                                    item["registered_parity_chunk_sha256"],
                                )
                                for item in device_bindings
                            }
                        )
                        == 1
                    )
                    vectors_exact = (
                        len({canonical_json_bytes(item["vectors"]) for item in device_bindings}) == 1
                    )
                    groups.append(
                        {
                            "replicate": replicate,
                            "panel_id": panel,
                            "law_family": law,
                            "update": update,
                            "device_bindings": device_bindings,
                            "ordered_example_ids_exact": ordered_exact,
                            "state_hashes_exact": state_exact,
                            "output_hashes_exact": outputs_exact,
                            "vector_chunk_manifests_exact": vectors_exact,
                            "passed": ordered_exact and state_exact and outputs_exact and vectors_exact,
                        }
                    )
    expected_group_count = len(REPLICATES[profile]) * len(PANELS) * len(LAW_FAMILIES) * len(UPDATES)
    _require(len(groups) == expected_group_count, "cross-GPU group construction is incomplete")
    ordered_exact = all(item["ordered_example_ids_exact"] for item in groups)
    state_exact = all(item["state_hashes_exact"] for item in groups)
    outputs_exact = all(item["output_hashes_exact"] for item in groups)
    vectors_exact = all(item["vector_chunk_manifests_exact"] for item in groups)
    return {
        "profile": profile,
        "group_count": len(groups),
        "devices_per_group": DEVICE_COUNT,
        "groups": groups,
        "exact_ordered_example_ids": ordered_exact,
        "exact_model_optimizer_combined_state_hashes": state_exact,
        "exact_output_hashes": outputs_exact,
        "exact_vector_chunk_manifests": vectors_exact,
        "passed": ordered_exact and state_exact and outputs_exact and vectors_exact,
    }


PROJECTION_COMPONENT_FIELDS = (
    "tokenizer_load_initialization_seconds",
    "data_bank_render_tokenization_seconds",
    "model_load_seconds",
    "updates_1_to_8_seconds",
    "tokenizer_host_envelopes_8_updates_seconds",
    "evaluation_13_boundaries_seconds",
    "evaluation_callback_io_seconds",
    "final_checkpoint_io_seconds",
    "outcome_seal_seconds",
)
PROFILE_INDEPENDENT_PROJECTION_COMPONENT_FIELDS = (
    "tokenizer_load_initialization_seconds",
    "data_bank_render_tokenization_seconds",
    "model_load_seconds",
    "final_checkpoint_io_seconds",
    "outcome_seal_seconds",
)


def _projection_from_components(timing: Mapping[str, Any]) -> float:
    return (
        float(timing["tokenizer_load_initialization_seconds"])
        + float(timing["data_bank_render_tokenization_seconds"])
        + float(timing["model_load_seconds"])
        + float(timing["updates_1_to_8_seconds"]) / len(UPDATES) * PRODUCTION_UPDATES
        + float(timing["tokenizer_host_envelopes_8_updates_seconds"]) / len(UPDATES) * PRODUCTION_UPDATES
        + float(timing["evaluation_13_boundaries_seconds"])
        + float(timing["evaluation_callback_io_seconds"])
        + float(timing["final_checkpoint_io_seconds"])
        + float(timing["outcome_seal_seconds"])
    )


def _production_projection_from_components(timing: Mapping[str, Any]) -> float:
    """Project the production path while excluding additive synthetic host stress."""

    return _projection_from_components(timing) - (
        float(timing["tokenizer_host_envelopes_8_updates_seconds"]) / len(UPDATES) * PRODUCTION_UPDATES
    )


def _profile_projection(records: Sequence[Mapping[str, Any]], profile: str) -> dict[str, Any]:
    device_uuids = sorted({str(item["identity"]["device_uuid"]) for item in records})
    per_device_runs: dict[str, dict[str, float]] = {}
    per_device_production_runs: dict[str, dict[str, float]] = {}
    per_device_components: dict[str, dict[str, dict[str, float]]] = {}
    standardized_training: dict[str, dict[str, dict[str, Any]]] = {}
    for device in device_uuids:
        per_device_runs[device] = {}
        per_device_production_runs[device] = {}
        per_device_components[device] = {}
        standardized_training[device] = {}
        for panel in PANELS:
            all_candidates = [
                item
                for item in records
                if item["identity"]["profile"] == profile
                and item["identity"]["panel_id"] == panel
                and item["identity"]["device_uuid"] == device
            ]
            representatives = [
                item
                for item in all_candidates
                if item["execution_benchmark"]["timing_representative"] is True
            ]
            _require(
                len(all_candidates) == len(REPLICATES[profile]) * len(LAW_FAMILIES)
                and len(representatives) == 1,
                f"projection needs one timing representative for {profile}/{panel}/{device}",
            )
            component_maxima: dict[str, float] = {}
            for field in PROJECTION_COMPONENT_FIELDS:
                source = (
                    all_candidates
                    if field in PROFILE_INDEPENDENT_PROJECTION_COMPONENT_FIELDS
                    else representatives
                )
                component_maxima[field] = max(float(item["timing"][field]) for item in source)
            per_device_components[device][panel] = component_maxima
            per_device_runs[device][panel] = _projection_from_components(component_maxima)
            per_device_production_runs[device][panel] = _production_projection_from_components(
                component_maxima
            )
            standardized = representatives[0]["timing"]["update_timing_contract"][
                "standardized_worst_shape_training"
            ]
            maximum_proof = standardized["global_maximum_proof"]
            standardized_training[device][panel] = {
                "available": True,
                "global_maximum_proof_digest": maximum_proof["proof_digest"],
                "maximum_token_length": maximum_proof["maximum_token_length"],
                "timed_call_shapes_sha256": standardized["timed_call_shapes_sha256"],
                "timed_padded_token_elements": standardized["timed_padded_token_elements"],
                "conservative_scaling_ratio": standardized["conservative_scaling_ratio"],
                "all_registered_training_calls_dominated": standardized[
                    "standardized_workload_dominates_every_registered_training_call"
                ],
            }
    per_device_raw = {
        device: sum(values[panel] * RUNS_PER_PANEL_PER_WORKER for panel in PANELS)
        for device, values in per_device_runs.items()
    }
    per_device_production_raw = {
        device: sum(values[panel] * RUNS_PER_PANEL_PER_WORKER for panel in PANELS)
        for device, values in per_device_production_runs.items()
    }
    per_device_clean_train_steps = {
        device: sum(
            per_device_components[device][panel]["updates_1_to_8_seconds"]
            / len(UPDATES)
            * PRODUCTION_UPDATES
            * RUNS_PER_PANEL_PER_WORKER
            for panel in PANELS
        )
        for device in device_uuids
    }
    per_device_projected = {
        device: value * PROJECTION_SAFETY_MULTIPLIER for device, value in per_device_raw.items()
    }
    panel_worst_seconds = {
        panel: max(per_device_runs[device][panel] for device in device_uuids) for panel in PANELS
    }
    raw = max(per_device_raw.values())
    projected = raw * PROJECTION_SAFETY_MULTIPLIER
    return {
        "available": True,
        "profile_independent_component_fields": list(PROFILE_INDEPENDENT_PROJECTION_COMPONENT_FIELDS),
        "cross_profile_shared_max_applied": False,
        "componentwise_maximum_rule": (
            "max_each_component_per_profile_panel_device_then_scale_updates_and_sum"
        ),
        "per_device_componentwise_maxima_seconds": per_device_components,
        "standardized_worst_shape_training": standardized_training,
        "per_panel_worst_projected_run_seconds": panel_worst_seconds,
        "per_device_projected_run_seconds": per_device_runs,
        "synthetic_host_stress_excluded_from_throughput_gate": True,
        "per_device_production_projected_run_seconds": per_device_production_runs,
        "per_device_raw_worker_seconds": per_device_raw,
        "per_device_projected_worker_seconds": per_device_projected,
        "per_device_production_raw_worker_seconds": per_device_production_raw,
        "per_device_production_projected_worker_seconds": {
            device: value * PROJECTION_SAFETY_MULTIPLIER
            for device, value in per_device_production_raw.items()
        },
        "per_device_clean_train_steps_worker_seconds": per_device_clean_train_steps,
        "runs_per_panel_per_worker": RUNS_PER_PANEL_PER_WORKER,
        "runs_per_worker": RUNS_PER_WORKER,
        "raw_worker_seconds": raw,
        "safety_multiplier": PROJECTION_SAFETY_MULTIPLIER,
        "projected_worker_seconds": projected,
        "maximum_projected_worker_seconds": MAXIMUM_PROJECTED_WALL_SECONDS,
        "passed": projected <= MAXIMUM_PROJECTED_WALL_SECONDS,
    }


def _unavailable_projection(device_uuids: Sequence[str]) -> dict[str, Any]:
    components = {
        device: {panel: {field: 0.0 for field in PROJECTION_COMPONENT_FIELDS} for panel in PANELS}
        for device in device_uuids
    }
    runs = {device: {panel: 0.0 for panel in PANELS} for device in device_uuids}
    walls = {device: 0.0 for device in device_uuids}
    standardized = {
        device: {
            panel: {
                "available": False,
                "global_maximum_proof_digest": None,
                "maximum_token_length": 0,
                "timed_call_shapes_sha256": None,
                "timed_padded_token_elements": 0,
                "conservative_scaling_ratio": 0.0,
                "all_registered_training_calls_dominated": False,
            }
            for panel in PANELS
        }
        for device in device_uuids
    }
    return {
        "available": False,
        "profile_independent_component_fields": list(PROFILE_INDEPENDENT_PROJECTION_COMPONENT_FIELDS),
        "cross_profile_shared_max_applied": False,
        "componentwise_maximum_rule": (
            "max_each_component_per_profile_panel_device_then_scale_updates_and_sum"
        ),
        "per_device_componentwise_maxima_seconds": components,
        "standardized_worst_shape_training": standardized,
        "per_panel_worst_projected_run_seconds": {panel: 0.0 for panel in PANELS},
        "per_device_projected_run_seconds": runs,
        "synthetic_host_stress_excluded_from_throughput_gate": True,
        "per_device_production_projected_run_seconds": runs,
        "per_device_raw_worker_seconds": walls,
        "per_device_projected_worker_seconds": walls,
        "per_device_production_raw_worker_seconds": walls,
        "per_device_production_projected_worker_seconds": walls,
        "per_device_clean_train_steps_worker_seconds": walls,
        "runs_per_panel_per_worker": RUNS_PER_PANEL_PER_WORKER,
        "runs_per_worker": RUNS_PER_WORKER,
        "raw_worker_seconds": 0.0,
        "safety_multiplier": PROJECTION_SAFETY_MULTIPLIER,
        "projected_worker_seconds": 0.0,
        "maximum_projected_worker_seconds": MAXIMUM_PROJECTED_WALL_SECONDS,
        "passed": False,
    }


def _recompute_projection(projection: dict[str, Any]) -> None:
    device_components = projection["per_device_componentwise_maxima_seconds"]
    device_runs = {
        device: {panel: _projection_from_components(components) for panel, components in panels.items()}
        for device, panels in device_components.items()
    }
    device_raw = {
        device: sum(values[panel] * RUNS_PER_PANEL_PER_WORKER for panel in PANELS)
        for device, values in device_runs.items()
    }
    device_production_runs = {
        device: {
            panel: _production_projection_from_components(components) for panel, components in panels.items()
        }
        for device, panels in device_components.items()
    }
    device_production_raw = {
        device: sum(values[panel] * RUNS_PER_PANEL_PER_WORKER for panel in PANELS)
        for device, values in device_production_runs.items()
    }
    device_clean_train_steps = {
        device: sum(
            device_components[device][panel]["updates_1_to_8_seconds"]
            / len(UPDATES)
            * PRODUCTION_UPDATES
            * RUNS_PER_PANEL_PER_WORKER
            for panel in PANELS
        )
        for device in device_components
    }
    projection["per_device_projected_run_seconds"] = device_runs
    projection["per_device_raw_worker_seconds"] = device_raw
    projection["per_device_projected_worker_seconds"] = {
        device: value * PROJECTION_SAFETY_MULTIPLIER for device, value in device_raw.items()
    }
    projection["per_device_production_projected_run_seconds"] = device_production_runs
    projection["per_device_production_raw_worker_seconds"] = device_production_raw
    projection["per_device_production_projected_worker_seconds"] = {
        device: value * PROJECTION_SAFETY_MULTIPLIER for device, value in device_production_raw.items()
    }
    projection["per_device_clean_train_steps_worker_seconds"] = device_clean_train_steps
    projection["per_panel_worst_projected_run_seconds"] = {
        panel: max(device_runs[device][panel] for device in device_runs) for panel in PANELS
    }
    raw = max(device_raw.values())
    projection["raw_worker_seconds"] = raw
    projection["projected_worker_seconds"] = raw * PROJECTION_SAFETY_MULTIPLIER
    projection["passed"] = projection["projected_worker_seconds"] <= MAXIMUM_PROJECTED_WALL_SECONDS


def _apply_cross_profile_shared_component_maxima(baseline: dict[str, Any], tuned: dict[str, Any]) -> None:
    for device in baseline["per_device_componentwise_maxima_seconds"]:
        for panel in PANELS:
            left = baseline["per_device_componentwise_maxima_seconds"][device][panel]
            right = tuned["per_device_componentwise_maxima_seconds"][device][panel]
            for field in PROFILE_INDEPENDENT_PROJECTION_COMPONENT_FIELDS:
                shared = max(float(left[field]), float(right[field]))
                left[field] = shared
                right[field] = shared
    baseline["cross_profile_shared_max_applied"] = True
    tuned["cross_profile_shared_max_applied"] = True
    _recompute_projection(baseline)
    _recompute_projection(tuned)


def _vector_thresholds_passed(summary: Mapping[str, Any]) -> bool:
    for vector_kind, measurements in summary["vectors"].items():
        if vector_kind in {"raw_preclip_gradient", "postclip_gradient"}:
            minimum_cosine = MINIMUM_GRADIENT_COSINE
            maximum_relative = MAXIMUM_GRADIENT_RELATIVE_L2
        else:
            minimum_cosine = MINIMUM_PARAMETER_UPDATE_COSINE
            maximum_relative = MAXIMUM_PARAMETER_UPDATE_RELATIVE_L2
        if (
            float(measurements["minimum_cosine"]) < minimum_cosine
            or float(measurements["maximum_relative_l2"]) > maximum_relative
        ):
            return False
    return True


def _qualification_contract() -> dict[str, Any]:
    return {
        "profiles": copy.deepcopy({key: dict(value) for key, value in PROFILE_CONTRACT.items()}),
        "updates": list(UPDATES),
        "training_views": list(TRAINING_VIEWS),
        "law_families": list(LAW_FAMILIES),
        "panels": copy.deepcopy({key: dict(value) for key, value in PANELS.items()}),
        "trainable_parameters": {
            "enumeration_api": "named_parameters(remove_duplicate=True)",
            "requires_grad_only": True,
            "order": "sorted_parameter_keys",
            "trainable_numel": dict(TRAINABLE_NUMEL),
            "native_tensor_chunk_digest_encoding": (
                "sha256(canonical_json({dtype,shape,contiguous_bytes_sha256}))"
            ),
            "accumulator_dtype": "float64",
        },
        "thresholds": {
            "maximum_normalized_score_difference": MAXIMUM_SCORE_DIFFERENCE,
            "maximum_probability_difference": MAXIMUM_PROBABILITY_DIFFERENCE,
            "minimum_gradient_cosine": MINIMUM_GRADIENT_COSINE,
            "maximum_gradient_relative_l2": MAXIMUM_GRADIENT_RELATIVE_L2,
            "minimum_parameter_update_cosine": MINIMUM_PARAMETER_UPDATE_COSINE,
            "maximum_parameter_update_relative_l2": MAXIMUM_PARAMETER_UPDATE_RELATIVE_L2,
            "maximum_tuned_reserved_bytes": MAXIMUM_TUNED_RESERVED_BYTES,
            "maximum_qualification_evidence_bytes": MAXIMUM_QUALIFICATION_EVIDENCE_BYTES,
            "maximum_accumulator_rows_per_vector": MAXIMUM_ACCUMULATOR_ROWS_PER_VECTOR,
            "minimum_throughput_ratio": MINIMUM_THROUGHPUT_RATIO,
            "projection_safety_multiplier": PROJECTION_SAFETY_MULTIPLIER,
            "maximum_projected_wall_seconds": MAXIMUM_PROJECTED_WALL_SECONDS,
        },
        "selection_rule": (
            "tuned_iff_all_tuned_requirements_else_baseline_iff_baseline_projection_"
            "and_common_requirements_else_fail_closed"
        ),
    }


def create_qualification_report(evidence: Mapping[str, Any]) -> dict[str, Any]:
    """Validate authenticated actual-model evidence and construct a candidate report."""

    validated = _validate_evidence_internal(evidence)
    records = validated.records
    full_branch = validated.qualification_branch == "tuned_probe_passed_full"
    baseline_records = tuple(item for item in records if item["identity"]["profile"] == "baseline")
    tuned_records = tuple(item for item in records if item["identity"]["profile"] == "tuned")
    baseline_common = all(_record_common_passed(item) for item in baseline_records)
    tuned_common = full_branch and all(_record_common_passed(item) for item in tuned_records)
    baseline_initial_state_exact = _initial_state_exact(baseline_records)
    tuned_initial_state_exact = _initial_state_exact(tuned_records)
    initial_state_exact = full_branch and _initial_state_exact(records)
    baseline_data_order_exact = _data_order_exact(baseline_records)
    tuned_data_order_exact = _data_order_exact(tuned_records)
    data_order_exact = full_branch and _data_order_exact(records)
    baseline_vs_tuned = (
        _comparison_summary(validated.comparisons, "baseline_vs_tuned")
        if full_branch
        else _unavailable_comparison_summary()
    )
    baseline_replay = _digest_exact_replay_summary(validated, "baseline")
    tuned_replay = (
        _digest_exact_replay_summary(validated, "tuned") if full_branch else _unavailable_comparison_summary()
    )
    registered_evaluation_parity = (
        _registered_evaluation_parity(validated)
        if full_branch
        else _unavailable_registered_evaluation_parity()
    )
    baseline_cross_gpu = _cross_gpu_equivalence(validated, "baseline")
    tuned_cross_gpu = (
        _cross_gpu_equivalence(validated, "tuned")
        if full_branch
        else _unavailable_cross_gpu_equivalence("tuned")
    )
    baseline_replay_hashes_exact = _exact_profile_replay(
        validated,
        "baseline",
        require_output_hashes=True,
    )
    tuned_replay_hashes_exact = full_branch and _exact_profile_replay(
        validated, "tuned", require_output_hashes=True
    )

    capacity_cells = [
        cell
        for device_receipt in validated.tuned_capacity_probe["device_receipts"]
        for cell in device_receipt["probe_cells"]
    ]
    _require(len(capacity_cells) == 16, "tuned memory authority does not contain sixteen cells")
    maximum_reserved = max(int(item["peak_reserved_bytes"]) for item in capacity_cells)
    maximum_allocated = max(int(item["peak_allocated_bytes"]) for item in capacity_cells)
    tuned_memory_diagnostics = [item["memory"] for item in tuned_records]
    diagnostic_reserved = max(
        (int(item["peak_reserved_bytes"]) for item in tuned_memory_diagnostics),
        default=0,
    )
    diagnostic_allocated = max(
        (int(item["peak_allocated_bytes"]) for item in tuned_memory_diagnostics),
        default=0,
    )
    memory_passed = bool(
        full_branch
        and all(item["status"] == "passed" for item in capacity_cells)
        and all(item["isolated_subprocess"] for item in capacity_cells)
        and all(item["fresh_model_and_optimizer"] for item in capacity_cells)
        and all(item["cuda_synchronized_before_memory_read"] for item in capacity_cells)
        and maximum_reserved <= MAXIMUM_TUNED_RESERVED_BYTES
    )

    baseline_projection = _profile_projection(records, "baseline")
    tuned_projection = (
        _profile_projection(records, "tuned")
        if full_branch
        else _unavailable_projection(validated.device_uuids)
    )
    if full_branch:
        _apply_cross_profile_shared_component_maxima(baseline_projection, tuned_projection)
    device_throughput_ratios = (
        {
            device: float(baseline_projection["per_device_production_raw_worker_seconds"][device])
            / float(tuned_projection["per_device_production_raw_worker_seconds"][device])
            for device in validated.device_uuids
        }
        if full_branch
        else {device: 0.0 for device in validated.device_uuids}
    )
    throughput_ratio = min(device_throughput_ratios.values())
    throughput_passed = full_branch and all(
        value >= MINIMUM_THROUGHPUT_RATIO for value in device_throughput_ratios.values()
    )
    clean_train_steps_ratios = (
        {
            device: float(baseline_projection["per_device_clean_train_steps_worker_seconds"][device])
            / float(tuned_projection["per_device_clean_train_steps_worker_seconds"][device])
            for device in validated.device_uuids
        }
        if full_branch
        else {device: 0.0 for device in validated.device_uuids}
    )
    minimum_clean_train_steps_ratio = min(clean_train_steps_ratios.values())
    clean_train_steps_throughput_passed = full_branch and all(
        value >= MINIMUM_THROUGHPUT_RATIO for value in clean_train_steps_ratios.values()
    )
    throughput_passed = throughput_passed and clean_train_steps_throughput_passed
    output_equivalence_passed = bool(
        full_branch
        and baseline_vs_tuned["sample_identity_exact"]
        and baseline_vs_tuned["actions_identical"]
        and float(baseline_vs_tuned["maximum_normalized_score_difference"]) <= MAXIMUM_SCORE_DIFFERENCE
        and float(baseline_vs_tuned["maximum_probability_difference"]) <= MAXIMUM_PROBABILITY_DIFFERENCE
    )
    vector_equivalence_passed = full_branch and _vector_thresholds_passed(baseline_vs_tuned)
    baseline_replay_passed = bool(
        baseline_replay_hashes_exact
        and baseline_replay["sample_identity_exact"]
        and baseline_replay["actions_identical"]
        and float(baseline_replay["maximum_normalized_score_difference"]) == 0.0
        and float(baseline_replay["maximum_probability_difference"]) == 0.0
        and all(
            float(value["minimum_cosine"]) == 1.0 and float(value["maximum_relative_l2"]) == 0.0
            for value in baseline_replay["vectors"].values()
        )
    )
    tuned_replay_passed = bool(
        full_branch
        and tuned_replay_hashes_exact
        and tuned_replay["sample_identity_exact"]
        and tuned_replay["actions_identical"]
        and float(tuned_replay["maximum_normalized_score_difference"]) == 0.0
        and float(tuned_replay["maximum_probability_difference"]) == 0.0
        and all(
            float(value["minimum_cosine"]) == 1.0 and float(value["maximum_relative_l2"]) == 0.0
            for value in tuned_replay["vectors"].values()
        )
    )

    baseline_passed = bool(
        baseline_common
        and baseline_initial_state_exact
        and baseline_data_order_exact
        and baseline_replay_passed
        and baseline_cross_gpu["passed"]
        and baseline_projection["passed"]
    )
    tuned_passed = bool(
        tuned_common
        and tuned_initial_state_exact
        and tuned_data_order_exact
        and initial_state_exact
        and data_order_exact
        and output_equivalence_passed
        and registered_evaluation_parity["passed"]
        and vector_equivalence_passed
        and tuned_replay_passed
        and tuned_cross_gpu["passed"]
        and memory_passed
        and throughput_passed
        and tuned_projection["passed"]
    )
    selection_candidate: str | None
    if tuned_passed:
        selection_candidate = "tuned"
    elif baseline_passed:
        selection_candidate = "baseline"
    else:
        selection_candidate = None

    body: dict[str, Any] = {
        "schema": QUALIFICATION_SCHEMA,
        "schema_version": QUALIFICATION_SCHEMA_VERSION,
        "actual_model_execution": True,
        "qualification_branch": validated.qualification_branch,
        "tuned_capacity_probe": copy.deepcopy(dict(validated.tuned_capacity_probe)),
        "execution_claim": {
            "validated_from_authenticated_producer_records": True,
            "models_executed_by_this_aggregator": False,
            "authentication_boundary": copy.deepcopy(dict(EVIDENCE_BOUNDARY)),
        },
        "evidence_binding": {
            "evidence_digest": evidence["evidence_digest"],
            "evidence_source": copy.deepcopy(dict(evidence["evidence_source"])),
            "qualification_branch": validated.qualification_branch,
            "tuned_capacity_probe_digest": validated.tuned_capacity_probe["probe_digest"],
            "process_record_count": len(records),
            "comparison_record_count": len(validated.comparisons),
            "device_uuids": list(validated.device_uuids),
        },
        "gpu_uuids": list(validated.device_uuids),
        "trainable_parameter_manifests": {
            panel: copy.deepcopy(
                dict(
                    next(
                        record["trainable_parameters"]
                        for record in records
                        if record["identity"]["panel_id"] == panel
                    )
                )
            )
            for panel in PANELS
        },
        "freeze_binding": copy.deepcopy(dict(evidence["freeze_binding"])),
        "provision_binding": copy.deepcopy(dict(evidence["provision_binding"])),
        "model_receipt_bindings": copy.deepcopy(dict(evidence["model_receipt_bindings"])),
        "model_integration_audit_bindings": copy.deepcopy(dict(evidence["model_integration_audit_bindings"])),
        "storage_contract": copy.deepcopy(dict(evidence["storage_contract"])),
        "transient_raw_cleanup_before_itt_required": True,
        "compact_evidence_retained_through_final_gate": True,
        "engineering_scope": copy.deepcopy(dict(evidence["engineering_scope"])),
        "contract": _qualification_contract(),
        "checks": {
            "baseline_common_process_requirements": baseline_common,
            "tuned_common_process_requirements": tuned_common,
            "process_requirements": {
                "baseline": {
                    "record_count": len(baseline_records),
                    "failed_record_count": sum(not _record_common_passed(item) for item in baseline_records),
                    "passed": baseline_common,
                },
                "tuned": {
                    "record_count": len(tuned_records),
                    "failed_record_count": sum(not _record_common_passed(item) for item in tuned_records),
                    "passed": tuned_common,
                },
            },
            "baseline_fresh_identical_initial_model_and_optimizer_state": baseline_initial_state_exact,
            "tuned_fresh_identical_initial_model_and_optimizer_state": tuned_initial_state_exact,
            "fresh_identical_initial_model_and_optimizer_state": initial_state_exact,
            "baseline_ordered_data_exact": baseline_data_order_exact,
            "tuned_ordered_data_exact": tuned_data_order_exact,
            "ordered_data_exact": data_order_exact,
            "cross_gpu_equivalence": {
                "baseline": baseline_cross_gpu,
                "tuned": tuned_cross_gpu,
            },
            "baseline_vs_tuned": {
                **baseline_vs_tuned,
                "output_thresholds_passed": output_equivalence_passed,
                "vector_thresholds_passed": vector_equivalence_passed,
            },
            "registered_evaluation_parity": registered_evaluation_parity,
            "baseline_replay": {
                **baseline_replay,
                "state_optimizer_output_and_vector_hashes_exact": baseline_replay_hashes_exact,
                "fixed_boundary_parity": True,
                "passed": baseline_replay_passed,
            },
            "tuned_replay": {
                **tuned_replay,
                "state_optimizer_output_and_vector_hashes_exact": tuned_replay_hashes_exact,
                "passed": tuned_replay_passed,
            },
            "tuned_memory": {
                "authority": "tuned_capacity_probe_16_clean_isolated_cells",
                "capacity_probe_cell_count": len(capacity_cells),
                "maximum_peak_allocated_bytes": maximum_allocated,
                "maximum_peak_reserved_bytes": maximum_reserved,
                "maximum_allowed_reserved_bytes": MAXIMUM_TUNED_RESERVED_BYTES,
                "cuda_synchronized_before_every_read": all(
                    item["cuda_synchronized_before_memory_read"] for item in capacity_cells
                ),
                "isolated_fresh_process_every_cell": all(
                    item["isolated_subprocess"] and item["fresh_model_and_optimizer"]
                    for item in capacity_cells
                ),
                "evidence_process_peak_allocated_bytes_diagnostic_only": diagnostic_allocated,
                "evidence_process_peak_reserved_bytes_diagnostic_only": diagnostic_reserved,
                "passed": memory_passed,
            },
            "performance": {
                "baseline": baseline_projection,
                "tuned": tuned_projection,
                "throughput_ratio_basis": (
                    "full_production_projection_excluding_additive_synthetic_host_stress"
                ),
                "throughput_ratio": throughput_ratio,
                "per_device_throughput_ratio": device_throughput_ratios,
                "minimum_device_throughput_ratio": throughput_ratio,
                "clean_train_steps_ratio_basis": ("clean_shared_train_steps_wall_projection_only"),
                "per_device_clean_train_steps_ratio": clean_train_steps_ratios,
                "minimum_clean_train_steps_ratio": minimum_clean_train_steps_ratio,
                "clean_train_steps_throughput_passed": clean_train_steps_throughput_passed,
                "minimum_throughput_ratio": MINIMUM_THROUGHPUT_RATIO,
                "throughput_passed": throughput_passed,
            },
        },
        "profiles": {
            "baseline": {"qualification_passed": baseline_passed},
            "tuned": {"qualification_passed": tuned_passed},
        },
        "eligibility": {
            "baseline": baseline_passed,
            "tuned": tuned_passed,
        },
        "projections": {
            "baseline": {
                "projected_wall_seconds": baseline_projection["projected_worker_seconds"],
                "raw_wall_seconds": baseline_projection["raw_worker_seconds"],
                "safety_multiplier": PROJECTION_SAFETY_MULTIPLIER,
                "passed": baseline_projection["passed"],
            },
            "tuned": {
                "projected_wall_seconds": tuned_projection["projected_worker_seconds"],
                "raw_wall_seconds": tuned_projection["raw_worker_seconds"],
                "safety_multiplier": PROJECTION_SAFETY_MULTIPLIER,
                "passed": tuned_projection["passed"],
            },
        },
        "selection_candidate": selection_candidate,
        "selection_authorized": False,
        "overall_qualification_passed": selection_candidate is not None,
        "outcomes_seen": False,
        "weight_updates_scope": "engineering_qualification_only",
        "itt_ledger_created": False,
        "g01_launch_authorized": False,
    }
    return _seal(body, "report_digest")


def _validate_report_comparison_summary(
    value: Any,
    *,
    label: str,
    mode: str,
    expected_comparisons: int | None = None,
) -> Mapping[str, Any]:
    summary = _mapping(value, label)
    common_keys = {
        "comparison_count",
        "step_count",
        "vector_comparison_count",
        "maximum_normalized_score_difference",
        "maximum_probability_difference",
        "actions_identical",
        "sample_identity_exact",
        "vectors",
    }
    _require(
        mode in {"cross_profile", "baseline_replay", "tuned_replay"},
        "unknown report comparison-summary mode",
    )
    if mode == "cross_profile":
        specific_keys = {"output_thresholds_passed", "vector_thresholds_passed"}
    elif mode == "baseline_replay":
        specific_keys = {
            "state_optimizer_output_and_vector_hashes_exact",
            "fixed_boundary_parity",
            "passed",
        }
    else:
        specific_keys = {"state_optimizer_output_and_vector_hashes_exact", "passed"}
    _exact_keys(summary, common_keys | specific_keys, label)
    if expected_comparisons is None:
        expected_comparisons = 4 if mode == "cross_profile" else 16
    _require(
        summary["comparison_count"] == expected_comparisons,
        f"{label} comparison count mismatch",
    )
    _require(
        summary["step_count"] == expected_comparisons * len(UPDATES),
        f"{label} step count mismatch",
    )
    _require(
        summary["vector_comparison_count"] == expected_comparisons * len(UPDATES) * len(VECTOR_KINDS),
        f"{label} vector count mismatch",
    )
    score_difference = _finite(
        summary["maximum_normalized_score_difference"],
        f"{label} score difference",
        minimum=0.0,
    )
    probability_difference = _finite(
        summary["maximum_probability_difference"],
        f"{label} probability difference",
        minimum=0.0,
    )
    actions_identical = _boolean(summary["actions_identical"], f"{label} action flag")
    sample_identity_exact = _boolean(
        summary["sample_identity_exact"],
        f"{label} sample identity flag",
    )
    vectors = _mapping(summary["vectors"], f"{label} vectors")
    _exact_keys(vectors, VECTOR_KINDS, f"{label} vectors")
    for vector_kind in VECTOR_KINDS:
        measurements = _mapping(vectors[vector_kind], f"{label} {vector_kind}")
        _exact_keys(
            measurements,
            {"minimum_cosine", "maximum_relative_l2"},
            f"{label} {vector_kind}",
        )
        cosine = _finite(measurements["minimum_cosine"], f"{label} {vector_kind} cosine")
        relative = _finite(
            measurements["maximum_relative_l2"],
            f"{label} {vector_kind} relative L2",
            minimum=0.0,
        )
        _require(-1.0 <= cosine <= 1.0, f"{label} {vector_kind} cosine is outside [-1,1]")
        _require(relative >= 0.0, f"{label} {vector_kind} relative L2 is negative")

    if mode == "tuned_replay":
        hashes_exact = _boolean(
            summary["state_optimizer_output_and_vector_hashes_exact"],
            f"{label} exact-hash flag",
        )
        expected_passed = bool(
            hashes_exact
            and sample_identity_exact
            and actions_identical
            and score_difference == 0.0
            and probability_difference == 0.0
            and all(
                float(item["minimum_cosine"]) == 1.0 and float(item["maximum_relative_l2"]) == 0.0
                for item in vectors.values()
            )
        )
        _require(
            _boolean(summary["passed"], f"{label} pass flag") == expected_passed,
            f"{label} pass flag is inconsistent",
        )
    elif mode == "baseline_replay":
        hashes_exact = _boolean(
            summary["state_optimizer_output_and_vector_hashes_exact"],
            f"{label} exact state/optimizer/output/vector flag",
        )
        fixed_boundary = _boolean(
            summary["fixed_boundary_parity"],
            f"{label} fixed-boundary flag",
        )
        expected_passed = bool(
            hashes_exact
            and fixed_boundary
            and sample_identity_exact
            and actions_identical
            and score_difference == 0.0
            and probability_difference == 0.0
            and all(
                float(item["minimum_cosine"]) == 1.0 and float(item["maximum_relative_l2"]) == 0.0
                for item in vectors.values()
            )
        )
        _require(
            _boolean(summary["passed"], f"{label} pass flag") == expected_passed,
            f"{label} pass flag is inconsistent",
        )
    else:
        expected_output_passed = bool(
            sample_identity_exact
            and actions_identical
            and score_difference <= MAXIMUM_SCORE_DIFFERENCE
            and probability_difference <= MAXIMUM_PROBABILITY_DIFFERENCE
        )
        expected_vector_passed = _vector_thresholds_passed(summary)
        _require(
            _boolean(summary["output_thresholds_passed"], f"{label} output pass flag")
            == expected_output_passed,
            f"{label} output pass flag is inconsistent",
        )
        _require(
            _boolean(summary["vector_thresholds_passed"], f"{label} vector pass flag")
            == expected_vector_passed,
            f"{label} vector pass flag is inconsistent",
        )
    return summary


def _validate_report_projection(value: Any, profile: str) -> Mapping[str, Any]:
    projection = _mapping(value, f"{profile} projection")
    _exact_keys(
        projection,
        {
            "available",
            "profile_independent_component_fields",
            "cross_profile_shared_max_applied",
            "componentwise_maximum_rule",
            "per_device_componentwise_maxima_seconds",
            "standardized_worst_shape_training",
            "per_panel_worst_projected_run_seconds",
            "per_device_projected_run_seconds",
            "synthetic_host_stress_excluded_from_throughput_gate",
            "per_device_production_projected_run_seconds",
            "per_device_raw_worker_seconds",
            "per_device_projected_worker_seconds",
            "per_device_production_raw_worker_seconds",
            "per_device_production_projected_worker_seconds",
            "per_device_clean_train_steps_worker_seconds",
            "runs_per_panel_per_worker",
            "runs_per_worker",
            "raw_worker_seconds",
            "safety_multiplier",
            "projected_worker_seconds",
            "maximum_projected_worker_seconds",
            "passed",
        },
        f"{profile} projection",
    )
    available = _boolean(projection["available"], f"{profile} projection availability")
    _require(
        list(
            _sequence(
                projection["profile_independent_component_fields"],
                f"{profile} independent component fields",
            )
        )
        == list(PROFILE_INDEPENDENT_PROJECTION_COMPONENT_FIELDS),
        f"{profile} independent component fields changed",
    )
    _boolean(
        projection["cross_profile_shared_max_applied"],
        f"{profile} cross-profile shared-max flag",
    )
    _require(
        projection["componentwise_maximum_rule"]
        == "max_each_component_per_profile_panel_device_then_scale_updates_and_sum",
        f"{profile} componentwise projection rule changed",
    )
    panel_values = _mapping(
        projection["per_panel_worst_projected_run_seconds"],
        f"{profile} per-panel projection",
    )
    _exact_keys(panel_values, PANELS, f"{profile} per-panel projection")
    panel_seconds = {
        panel: _finite(value, f"{profile}/{panel} projected seconds", minimum=0.0)
        for panel, value in panel_values.items()
    }
    device_runs = _mapping(projection["per_device_projected_run_seconds"], f"{profile} device runs")
    device_raw = _mapping(projection["per_device_raw_worker_seconds"], f"{profile} device raw wall")
    device_projected = _mapping(
        projection["per_device_projected_worker_seconds"], f"{profile} device projected wall"
    )
    production_runs = _mapping(
        projection["per_device_production_projected_run_seconds"],
        f"{profile} production-only device runs",
    )
    production_raw = _mapping(
        projection["per_device_production_raw_worker_seconds"],
        f"{profile} production-only raw worker wall",
    )
    production_projected = _mapping(
        projection["per_device_production_projected_worker_seconds"],
        f"{profile} production-only projected worker wall",
    )
    clean_train_steps = _mapping(
        projection["per_device_clean_train_steps_worker_seconds"],
        f"{profile} clean train_steps worker wall",
    )
    device_components = _mapping(
        projection["per_device_componentwise_maxima_seconds"],
        f"{profile} componentwise maxima",
    )
    standardized_training = _mapping(
        projection["standardized_worst_shape_training"],
        f"{profile} standardized worst-shape training",
    )
    _require(
        set(device_runs)
        == set(device_raw)
        == set(device_projected)
        == set(production_runs)
        == set(production_raw)
        == set(production_projected)
        == set(clean_train_steps)
        == set(device_components)
        == set(standardized_training)
        and len(device_runs) == DEVICE_COUNT,
        f"{profile} projection device set changed",
    )
    replayed_device_raw: dict[str, float] = {}
    replayed_production_raw: dict[str, float] = {}
    replayed_clean_train_steps: dict[str, float] = {}
    for device, raw_runs in device_runs.items():
        runs = _mapping(raw_runs, f"{profile}/{device} projected runs")
        production_device_runs = _mapping(
            production_runs[device],
            f"{profile}/{device} production-only projected runs",
        )
        components_by_panel = _mapping(device_components[device], f"{profile}/{device} componentwise maxima")
        standardized_by_panel = _mapping(
            standardized_training[device],
            f"{profile}/{device} standardized worst-shape training",
        )
        _exact_keys(runs, PANELS, f"{profile}/{device} projected runs")
        _exact_keys(
            production_device_runs,
            PANELS,
            f"{profile}/{device} production-only projected runs",
        )
        _exact_keys(components_by_panel, PANELS, f"{profile}/{device} componentwise maxima")
        _exact_keys(
            standardized_by_panel,
            PANELS,
            f"{profile}/{device} standardized worst-shape training",
        )
        values = {
            panel: _finite(value, f"{profile}/{device}/{panel}", minimum=0.0) for panel, value in runs.items()
        }
        for panel in PANELS:
            standardized = _mapping(
                standardized_by_panel[panel],
                f"{profile}/{device}/{panel} standardized worst-shape training",
            )
            _exact_keys(
                standardized,
                {
                    "available",
                    "global_maximum_proof_digest",
                    "maximum_token_length",
                    "timed_call_shapes_sha256",
                    "timed_padded_token_elements",
                    "conservative_scaling_ratio",
                    "all_registered_training_calls_dominated",
                },
                f"{profile}/{device}/{panel} standardized worst-shape training",
            )
            if available:
                _require(
                    _boolean(standardized["available"], "standardized training availability")
                    and _integer(
                        standardized["maximum_token_length"],
                        "standardized maximum token length",
                        minimum=1,
                    )
                    > 0
                    and _integer(
                        standardized["timed_padded_token_elements"],
                        "standardized padded-token total",
                        minimum=1,
                    )
                    > 0
                    and _boolean(
                        standardized["all_registered_training_calls_dominated"],
                        "standardized training dominance flag",
                    ),
                    "available projection lacks standardized worst-shape timing proof",
                )
                _sha256(
                    standardized["global_maximum_proof_digest"],
                    "standardized global-maximum proof digest",
                )
                _sha256(
                    standardized["timed_call_shapes_sha256"],
                    "standardized timed-call-shape digest",
                )
                _same_number(
                    standardized["conservative_scaling_ratio"],
                    1.0,
                    "standardized conservative scaling ratio",
                )
            else:
                _require(
                    standardized
                    == {
                        "available": False,
                        "global_maximum_proof_digest": None,
                        "maximum_token_length": 0,
                        "timed_call_shapes_sha256": None,
                        "timed_padded_token_elements": 0,
                        "conservative_scaling_ratio": 0.0,
                        "all_registered_training_calls_dominated": False,
                    },
                    "unavailable projection retained a standardized training proof",
                )
            raw_components = _mapping(
                components_by_panel[panel], f"{profile}/{device}/{panel} componentwise maxima"
            )
            _exact_keys(
                raw_components,
                PROJECTION_COMPONENT_FIELDS,
                f"{profile}/{device}/{panel} componentwise maxima",
            )
            components = {
                field: _finite(
                    raw_components[field],
                    f"{profile}/{device}/{panel}/{field}",
                    minimum=0.0,
                )
                for field in PROJECTION_COMPONENT_FIELDS
            }
            _same_number(
                values[panel],
                _projection_from_components(components),
                f"{profile}/{device}/{panel} componentwise projection",
            )
            _same_number(
                production_device_runs[panel],
                _production_projection_from_components(components),
                f"{profile}/{device}/{panel} production-only projection",
            )
        replayed_device_raw[str(device)] = sum(values[panel] * RUNS_PER_PANEL_PER_WORKER for panel in PANELS)
        replayed_production_raw[str(device)] = sum(
            float(production_device_runs[panel]) * RUNS_PER_PANEL_PER_WORKER for panel in PANELS
        )
        replayed_clean_train_steps[str(device)] = sum(
            float(device_components[device][panel]["updates_1_to_8_seconds"])
            / len(UPDATES)
            * PRODUCTION_UPDATES
            * RUNS_PER_PANEL_PER_WORKER
            for panel in PANELS
        )
        _same_number(
            device_raw[device], replayed_device_raw[str(device)], f"{profile}/{device} raw worker seconds"
        )
        _same_number(
            device_projected[device],
            replayed_device_raw[str(device)] * PROJECTION_SAFETY_MULTIPLIER,
            f"{profile}/{device} projected worker seconds",
        )
        _same_number(
            production_raw[device],
            replayed_production_raw[str(device)],
            f"{profile}/{device} production-only raw worker seconds",
        )
        _same_number(
            production_projected[device],
            replayed_production_raw[str(device)] * PROJECTION_SAFETY_MULTIPLIER,
            f"{profile}/{device} production-only projected worker seconds",
        )
        _same_number(
            clean_train_steps[device],
            replayed_clean_train_steps[str(device)],
            f"{profile}/{device} clean train_steps worker seconds",
        )
    if available:
        for panel in PANELS:
            _require(
                len(
                    {
                        (
                            standardized_training[device][panel]["global_maximum_proof_digest"],
                            standardized_training[device][panel]["maximum_token_length"],
                            standardized_training[device][panel]["timed_call_shapes_sha256"],
                            standardized_training[device][panel]["timed_padded_token_elements"],
                        )
                        for device in standardized_training
                    }
                )
                == 1,
                f"{profile}/{panel} standardized training proof differs across GPUs",
            )
    for panel in PANELS:
        _same_number(
            panel_seconds[panel],
            max(float(device_runs[device][panel]) for device in device_runs),
            f"{profile}/{panel} worst projected run",
        )
    _require(
        _boolean(
            projection["synthetic_host_stress_excluded_from_throughput_gate"],
            f"{profile} synthetic-host-stress exclusion flag",
        )
        and projection["runs_per_panel_per_worker"] == RUNS_PER_PANEL_PER_WORKER
        and projection["runs_per_worker"] == RUNS_PER_WORKER,
        f"{profile} projection worker mix mismatch",
    )
    expected_raw = max(replayed_device_raw.values())
    raw = _finite(projection["raw_worker_seconds"], f"{profile} raw worker seconds", minimum=0.0)
    _same_number(raw, expected_raw, f"{profile} raw worker seconds")
    _same_number(
        projection["safety_multiplier"],
        PROJECTION_SAFETY_MULTIPLIER,
        f"{profile} safety multiplier",
    )
    projected = _finite(
        projection["projected_worker_seconds"],
        f"{profile} projected worker seconds",
        minimum=0.0,
    )
    _same_number(
        projected,
        raw * PROJECTION_SAFETY_MULTIPLIER,
        f"{profile} projected worker seconds",
    )
    _require(
        projection["maximum_projected_worker_seconds"] == MAXIMUM_PROJECTED_WALL_SECONDS,
        f"{profile} wall ceiling mismatch",
    )
    expected_passed = available and projected <= MAXIMUM_PROJECTED_WALL_SECONDS
    if not available:
        _require(projected == 0.0, f"{profile} unavailable projection is not zeroed")
    _require(
        _boolean(projection["passed"], f"{profile} projection pass flag") == expected_passed,
        f"{profile} projection pass flag is inconsistent",
    )
    return projection


def _validate_report_cross_gpu_equivalence(
    value: Any,
    *,
    profile: str,
    gpu_uuids: Sequence[str],
    trainable_manifests: Mapping[str, Mapping[str, Any]],
    required: bool = True,
) -> Mapping[str, Any]:
    summary = _mapping(value, f"{profile} cross-GPU equivalence")
    _exact_keys(
        summary,
        {
            "profile",
            "group_count",
            "devices_per_group",
            "groups",
            "exact_ordered_example_ids",
            "exact_model_optimizer_combined_state_hashes",
            "exact_output_hashes",
            "exact_vector_chunk_manifests",
            "passed",
        },
        f"{profile} cross-GPU equivalence",
    )
    _require(summary["profile"] == profile, f"{profile} cross-GPU profile changed")
    if not required:
        _require(
            summary == _unavailable_cross_gpu_equivalence(profile),
            f"{profile} unavailable cross-GPU summary changed",
        )
        return summary
    expected_identities = [
        (replicate, panel, law, update)
        for replicate in REPLICATES[profile]
        for panel in PANELS
        for law in LAW_FAMILIES
        for update in UPDATES
    ]
    groups = _sequence(summary["groups"], f"{profile} cross-GPU groups")
    _require(
        summary["group_count"] == len(expected_identities) == len(groups),
        f"{profile} cross-GPU group count mismatch",
    )
    _require(summary["devices_per_group"] == DEVICE_COUNT, "cross-GPU device cardinality changed")
    aggregate_ordered = True
    aggregate_state = True
    aggregate_outputs = True
    aggregate_vectors = True
    for group_index, (raw_group, expected_identity) in enumerate(
        zip(groups, expected_identities, strict=True)
    ):
        group = _mapping(raw_group, f"{profile} cross-GPU group {group_index}")
        _exact_keys(
            group,
            {
                "replicate",
                "panel_id",
                "law_family",
                "update",
                "device_bindings",
                "ordered_example_ids_exact",
                "state_hashes_exact",
                "output_hashes_exact",
                "vector_chunk_manifests_exact",
                "passed",
            },
            f"{profile} cross-GPU group {group_index}",
        )
        observed_identity = (
            group["replicate"],
            group["panel_id"],
            group["law_family"],
            group["update"],
        )
        _require(
            observed_identity == expected_identity,
            f"{profile} cross-GPU group identities are incomplete or out of order",
        )
        panel = str(group["panel_id"])
        manifest = trainable_manifests[panel]
        bindings = _sequence(
            group["device_bindings"],
            f"{profile} cross-GPU group {group_index} device bindings",
        )
        _require(len(bindings) == DEVICE_COUNT, "cross-GPU group does not bind four devices")
        normalized_bindings: list[Mapping[str, Any]] = []
        for expected_device, raw_binding in zip(gpu_uuids, bindings, strict=True):
            binding = _mapping(raw_binding, "cross-GPU device binding")
            _exact_keys(
                binding,
                {
                    "device_uuid",
                    "ordered_example_count",
                    "ordered_example_ids_sha256",
                    "outputs_sha256",
                    "registered_parity_chunk_sha256",
                    "execution_benchmark_sha256",
                    "state_hashes",
                    "vectors",
                },
                "cross-GPU device binding",
            )
            _require(binding["device_uuid"] == expected_device, "cross-GPU devices are not exact and sorted")
            _require(binding["ordered_example_count"] == 50, "cross-GPU data count changed")
            _sha256(binding["ordered_example_ids_sha256"], "cross-GPU ordered-data digest")
            _sha256(binding["outputs_sha256"], "cross-GPU output digest")
            _sha256(
                binding["registered_parity_chunk_sha256"],
                "cross-GPU registered parity chunk digest",
            )
            _sha256(binding["execution_benchmark_sha256"], "cross-GPU execution benchmark digest")
            state = _mapping(binding["state_hashes"], "cross-GPU state hashes")
            _exact_keys(
                state,
                {"model_state_sha256", "optimizer_state_sha256", "combined_state_sha256", "outputs_sha256"},
                "cross-GPU state hashes",
            )
            model_digest = _sha256(state["model_state_sha256"], "cross-GPU model-state digest")
            optimizer_digest = _sha256(
                state["optimizer_state_sha256"],
                "cross-GPU optimizer-state digest",
            )
            _require(
                state["combined_state_sha256"]
                == semantic_digest(
                    {
                        "model_state_sha256": model_digest,
                        "optimizer_state_sha256": optimizer_digest,
                    }
                ),
                "cross-GPU combined-state digest is invalid",
            )
            _require(
                state["outputs_sha256"] == binding["outputs_sha256"],
                "cross-GPU output/state binding differs",
            )
            vectors = _mapping(binding["vectors"], "cross-GPU vectors")
            _exact_keys(vectors, VECTOR_KINDS, "cross-GPU vectors")
            for vector_kind in VECTOR_KINDS:
                descriptor = _validate_vector_descriptor(
                    vectors[vector_kind],
                    f"cross-GPU {vector_kind}",
                )
                _require(
                    descriptor["element_count"] == TRAINABLE_NUMEL[panel]
                    and descriptor["parameter_keys_sha256"] == manifest["parameter_keys_sha256"]
                    and descriptor["trainable_parameter_manifest_sha256"] == manifest["manifest_sha256"]
                    and descriptor["chunk_count"] == len(manifest["entries"]),
                    "cross-GPU vector does not cover the exact trainable-parameter manifest",
                )
            normalized_bindings.append(binding)
        ordered_exact = (
            len(
                {
                    (item["ordered_example_count"], item["ordered_example_ids_sha256"])
                    for item in normalized_bindings
                }
            )
            == 1
        )
        state_exact = (
            len(
                {
                    (
                        item["state_hashes"]["model_state_sha256"],
                        item["state_hashes"]["optimizer_state_sha256"],
                        item["state_hashes"]["combined_state_sha256"],
                    )
                    for item in normalized_bindings
                }
            )
            == 1
        )
        outputs_exact = (
            len(
                {
                    (
                        item["outputs_sha256"],
                        item["state_hashes"]["outputs_sha256"],
                        item["execution_benchmark_sha256"],
                    )
                    for item in normalized_bindings
                }
            )
            == 1
        )
        vectors_exact = len({canonical_json_bytes(item["vectors"]) for item in normalized_bindings}) == 1
        group_passed = ordered_exact and state_exact and outputs_exact and vectors_exact
        for field, expected in (
            ("ordered_example_ids_exact", ordered_exact),
            ("state_hashes_exact", state_exact),
            ("output_hashes_exact", outputs_exact),
            ("vector_chunk_manifests_exact", vectors_exact),
            ("passed", group_passed),
        ):
            _require(
                _boolean(group[field], f"cross-GPU group {field}") == expected,
                f"cross-GPU group {field} is inconsistent",
            )
        aggregate_ordered = aggregate_ordered and ordered_exact
        aggregate_state = aggregate_state and state_exact
        aggregate_outputs = aggregate_outputs and outputs_exact
        aggregate_vectors = aggregate_vectors and vectors_exact
    aggregate_passed = aggregate_ordered and aggregate_state and aggregate_outputs and aggregate_vectors
    for field, expected in (
        ("exact_ordered_example_ids", aggregate_ordered),
        ("exact_model_optimizer_combined_state_hashes", aggregate_state),
        ("exact_output_hashes", aggregate_outputs),
        ("exact_vector_chunk_manifests", aggregate_vectors),
        ("passed", aggregate_passed),
    ):
        _require(
            _boolean(summary[field], f"{profile} cross-GPU {field}") == expected,
            f"{profile} cross-GPU {field} is inconsistent",
        )
    return summary


def validate_qualification_report(report: Mapping[str, Any]) -> dict[str, Any]:
    """Independently replay a persisted report's schema, digest, and decisions."""

    payload = _mapping(report, "qualification report")
    _exact_keys(
        payload,
        {
            "schema",
            "schema_version",
            "actual_model_execution",
            "qualification_branch",
            "tuned_capacity_probe",
            "execution_claim",
            "evidence_binding",
            "gpu_uuids",
            "trainable_parameter_manifests",
            "freeze_binding",
            "provision_binding",
            "model_receipt_bindings",
            "model_integration_audit_bindings",
            "storage_contract",
            "transient_raw_cleanup_before_itt_required",
            "compact_evidence_retained_through_final_gate",
            "engineering_scope",
            "contract",
            "checks",
            "profiles",
            "eligibility",
            "projections",
            "selection_candidate",
            "selection_authorized",
            "overall_qualification_passed",
            "outcomes_seen",
            "weight_updates_scope",
            "itt_ledger_created",
            "g01_launch_authorized",
            "report_digest",
        },
        "qualification report",
    )
    _require(
        payload["schema"] == QUALIFICATION_SCHEMA
        and payload["schema_version"] == QUALIFICATION_SCHEMA_VERSION,
        "qualification report schema mismatch",
    )
    _verify_self_digest(payload, "report_digest", "qualification report")
    branch = str(payload["qualification_branch"])
    _require(branch in QUALIFICATION_BRANCHES, "unknown report qualification branch")
    _require(
        _boolean(payload["actual_model_execution"], "report actual-model flag"),
        "qualification report is not based on actual-model execution",
    )
    claim = _mapping(payload["execution_claim"], "report execution claim")
    _require(
        claim
        == {
            "validated_from_authenticated_producer_records": True,
            "models_executed_by_this_aggregator": False,
            "authentication_boundary": EVIDENCE_BOUNDARY,
        },
        "qualification execution/authentication claim mismatch",
    )
    _engineering_scope(payload["engineering_scope"], "report engineering scope")
    _validate_bindings(payload)
    _validate_storage_contract(payload["storage_contract"])
    _require(
        _boolean(
            payload["transient_raw_cleanup_before_itt_required"],
            "report transient-raw cleanup-before-ITT flag",
        ),
        "qualification report does not require transient raw cleanup before ITT",
    )
    _require(
        _boolean(
            payload["compact_evidence_retained_through_final_gate"],
            "report compact-evidence retention flag",
        ),
        "qualification report does not retain compact evidence through the final gate",
    )
    _require(payload["contract"] == _qualification_contract(), "qualification contract changed")

    gpu_uuids_raw = _sequence(payload["gpu_uuids"], "report GPU UUIDs")
    gpu_uuids = tuple(str(item) for item in gpu_uuids_raw)
    _require(
        len(gpu_uuids) == DEVICE_COUNT
        and list(gpu_uuids) == sorted(set(gpu_uuids))
        and all(GPU_UUID_PATTERN.fullmatch(item) is not None for item in gpu_uuids),
        "qualification report does not bind four sorted distinct GPU UUIDs",
    )
    capacity_probe = _validate_tuned_capacity_probe(
        payload["tuned_capacity_probe"],
        expected_device_uuids=gpu_uuids,
    )
    _require(
        capacity_probe["qualification_branch"] == branch,
        "report branch differs from tuned capacity probe",
    )
    full_branch = branch == "tuned_probe_passed_full"
    raw_trainable_manifests = _mapping(
        payload["trainable_parameter_manifests"],
        "report trainable-parameter manifests",
    )
    _exact_keys(
        raw_trainable_manifests,
        PANELS,
        "report trainable-parameter manifests",
    )
    trainable_manifests = {
        panel: _validate_trainable_parameter_manifest(raw_trainable_manifests[panel], panel)
        for panel in PANELS
    }
    binding = _mapping(payload["evidence_binding"], "report evidence binding")
    _exact_keys(
        binding,
        {
            "evidence_digest",
            "evidence_source",
            "qualification_branch",
            "tuned_capacity_probe_digest",
            "process_record_count",
            "comparison_record_count",
            "device_uuids",
        },
        "report evidence binding",
    )
    _sha256(binding["evidence_digest"], "bound evidence digest")
    expected_process_count = PROCESS_RECORD_COUNT if full_branch else BASELINE_FALLBACK_PROCESS_RECORD_COUNT
    expected_comparison_count = (
        COMPARISON_RECORD_COUNT if full_branch else BASELINE_FALLBACK_COMPARISON_RECORD_COUNT
    )
    _require(
        binding["process_record_count"] == expected_process_count,
        "report process count mismatch",
    )
    _require(
        binding["comparison_record_count"] == expected_comparison_count,
        "report comparison count mismatch",
    )
    _require(
        list(_sequence(binding["device_uuids"], "bound evidence GPU UUIDs")) == list(gpu_uuids),
        "report GPU/evidence UUID mismatch",
    )
    _require(
        binding["qualification_branch"] == branch
        and binding["tuned_capacity_probe_digest"] == capacity_probe["probe_digest"],
        "report evidence branch/probe binding differs",
    )
    source = _mapping(binding["evidence_source"], "report evidence source")
    _exact_keys(
        source,
        {
            "kind",
            "authentication",
            "aggregator_role",
            "capture_command_sha256",
            "capture_implementation_sha256",
            "records_manifest_sha256",
        },
        "report evidence source",
    )
    for key, expected in EVIDENCE_BOUNDARY.items():
        _require(source[key] == expected, f"report evidence source {key} mismatch")
    for key in ("capture_command_sha256", "capture_implementation_sha256", "records_manifest_sha256"):
        _sha256(source[key], f"report evidence source {key}")

    checks = _mapping(payload["checks"], "qualification checks")
    _exact_keys(
        checks,
        {
            "baseline_common_process_requirements",
            "tuned_common_process_requirements",
            "process_requirements",
            "baseline_fresh_identical_initial_model_and_optimizer_state",
            "tuned_fresh_identical_initial_model_and_optimizer_state",
            "fresh_identical_initial_model_and_optimizer_state",
            "baseline_ordered_data_exact",
            "tuned_ordered_data_exact",
            "ordered_data_exact",
            "cross_gpu_equivalence",
            "baseline_vs_tuned",
            "registered_evaluation_parity",
            "baseline_replay",
            "tuned_replay",
            "tuned_memory",
            "performance",
        },
        "qualification checks",
    )
    process_requirements = _mapping(checks["process_requirements"], "process requirements")
    _exact_keys(process_requirements, PROFILES, "process requirements")
    process_passed: dict[str, bool] = {}
    expected_counts = {"baseline": 32, "tuned": 32 if full_branch else 0}
    for profile in PROFILES:
        requirement = _mapping(process_requirements[profile], f"{profile} process requirements")
        _exact_keys(
            requirement,
            {"record_count", "failed_record_count", "passed"},
            f"{profile} process requirements",
        )
        _require(
            requirement["record_count"] == expected_counts[profile],
            f"{profile} process requirement count mismatch",
        )
        failures = _integer(
            requirement["failed_record_count"],
            f"{profile} failed process count",
        )
        _require(failures <= expected_counts[profile], f"{profile} failed process count is too large")
        process_passed[profile] = failures == 0 and expected_counts[profile] > 0
        _require(
            _boolean(requirement["passed"], f"{profile} process pass flag") == process_passed[profile],
            f"{profile} process pass flag is inconsistent",
        )
        _require(
            _boolean(checks[f"{profile}_common_process_requirements"], f"{profile} common flag")
            == process_passed[profile],
            f"{profile} common process flag is inconsistent",
        )

    state_flags = {
        "baseline": _boolean(
            checks["baseline_fresh_identical_initial_model_and_optimizer_state"],
            "baseline initial-state flag",
        ),
        "tuned": _boolean(
            checks["tuned_fresh_identical_initial_model_and_optimizer_state"],
            "tuned initial-state flag",
        ),
        "cross": _boolean(
            checks["fresh_identical_initial_model_and_optimizer_state"],
            "cross-profile initial-state flag",
        ),
    }
    data_flags = {
        "baseline": _boolean(checks["baseline_ordered_data_exact"], "baseline data-order flag"),
        "tuned": _boolean(checks["tuned_ordered_data_exact"], "tuned data-order flag"),
        "cross": _boolean(checks["ordered_data_exact"], "cross-profile data-order flag"),
    }
    if not full_branch:
        _require(
            not state_flags["tuned"]
            and not state_flags["cross"]
            and not data_flags["tuned"]
            and not data_flags["cross"],
            "fallback report claims unavailable tuned/cross-profile evidence",
        )
    cross_gpu = _mapping(checks["cross_gpu_equivalence"], "cross-GPU equivalence checks")
    _exact_keys(cross_gpu, PROFILES, "cross-GPU equivalence checks")
    cross_gpu_summaries = {
        profile: _validate_report_cross_gpu_equivalence(
            cross_gpu[profile],
            profile=profile,
            gpu_uuids=gpu_uuids,
            trainable_manifests=trainable_manifests,
            required=(profile == "baseline" or full_branch),
        )
        for profile in PROFILES
    }
    baseline_summary = _validate_report_comparison_summary(
        checks["baseline_vs_tuned"],
        label="baseline/tuned comparison summary",
        mode="cross_profile",
        expected_comparisons=4 if full_branch else 0,
    )
    registered = _mapping(checks["registered_evaluation_parity"], "registered evaluation parity")
    _exact_keys(
        registered,
        {
            "states",
            "group_count",
            "shape_authentication_group_count",
            "numeric_parity_group_count",
            "row_comparison_count",
            "single_registered_partition_exact",
            "row_ids_exact",
            "maximum_normalized_score_difference",
            "maximum_probability_difference",
            "actions_identical",
            "passed",
        },
        "registered evaluation parity",
    )
    _require(
        list(_sequence(registered["states"], "registered parity states")) == list(REGISTERED_PARITY_STATES),
        "registered parity state schedule changed",
    )
    expected_registered_groups = 16 * len(REGISTERED_PARITY_STATES) if full_branch else 0
    expected_shape_groups = 16 if full_branch else 0
    expected_numeric_groups = 16 * len(REGISTERED_PARITY_UPDATE_CHECKPOINTS) if full_branch else 0
    _require(
        registered["group_count"] == expected_registered_groups,
        "registered parity group count changed",
    )
    _require(
        registered["shape_authentication_group_count"] == expected_shape_groups
        and registered["numeric_parity_group_count"] == expected_numeric_groups,
        "registered parity shape/numeric checkpoint counts changed",
    )
    registered_row_count = _integer(
        registered["row_comparison_count"],
        "registered parity row count",
        minimum=expected_numeric_groups * 128 * len(TRAINING_VIEWS),
    )
    _require(
        registered_row_count == expected_numeric_groups * 128 * len(TRAINING_VIEWS),
        "registered parity numeric-row count changed",
    )
    registered_score = _finite(
        registered["maximum_normalized_score_difference"],
        "registered parity score difference",
        minimum=0.0,
    )
    registered_probability = _finite(
        registered["maximum_probability_difference"],
        "registered parity probability difference",
        minimum=0.0,
    )
    registered_passed = bool(
        full_branch
        and _boolean(
            registered["single_registered_partition_exact"],
            "registered parity single-partition flag",
        )
        and _boolean(registered["row_ids_exact"], "registered parity row-ID flag")
        and _boolean(registered["actions_identical"], "registered parity action flag")
        and registered_score <= MAXIMUM_SCORE_DIFFERENCE
        and registered_probability <= MAXIMUM_PROBABILITY_DIFFERENCE
    )
    _require(
        _boolean(registered["passed"], "registered parity pass flag") == registered_passed,
        "registered evaluation parity pass flag is inconsistent",
    )
    baseline_replay_summary = _validate_report_comparison_summary(
        checks["baseline_replay"],
        label="baseline replay summary",
        mode="baseline_replay",
    )
    replay_summary = _validate_report_comparison_summary(
        checks["tuned_replay"],
        label="tuned replay summary",
        mode="tuned_replay",
        expected_comparisons=16 if full_branch else 0,
    )

    memory = _mapping(checks["tuned_memory"], "tuned memory check")
    _exact_keys(
        memory,
        {
            "authority",
            "capacity_probe_cell_count",
            "maximum_peak_allocated_bytes",
            "maximum_peak_reserved_bytes",
            "maximum_allowed_reserved_bytes",
            "cuda_synchronized_before_every_read",
            "isolated_fresh_process_every_cell",
            "evidence_process_peak_allocated_bytes_diagnostic_only",
            "evidence_process_peak_reserved_bytes_diagnostic_only",
            "passed",
        },
        "tuned memory check",
    )
    allocated = _integer(memory["maximum_peak_allocated_bytes"], "maximum allocated bytes")
    reserved = _integer(memory["maximum_peak_reserved_bytes"], "maximum reserved bytes")
    _require(
        memory["authority"] == "tuned_capacity_probe_16_clean_isolated_cells",
        "report memory authority changed",
    )
    capacity_cells = [
        cell for device_receipt in capacity_probe["device_receipts"] for cell in device_receipt["probe_cells"]
    ]
    _require(
        memory["capacity_probe_cell_count"] == len(capacity_cells) == 16
        and allocated == max(int(cell["peak_allocated_bytes"]) for cell in capacity_cells)
        and reserved == max(int(cell["peak_reserved_bytes"]) for cell in capacity_cells),
        "report memory maxima are not derived from all sixteen capacity cells",
    )
    _require(allocated <= reserved, "report maximum allocated bytes exceed reserved bytes")
    _require(
        memory["maximum_allowed_reserved_bytes"] == MAXIMUM_TUNED_RESERVED_BYTES,
        "report tuned memory ceiling changed",
    )
    synchronized = _boolean(
        memory["cuda_synchronized_before_every_read"],
        "report CUDA synchronization flag",
    )
    isolated = _boolean(
        memory["isolated_fresh_process_every_cell"],
        "report isolated/fresh capacity-cell flag",
    )
    expected_synchronized = all(bool(cell["cuda_synchronized_before_memory_read"]) for cell in capacity_cells)
    expected_isolated = all(
        bool(cell["isolated_subprocess"]) and bool(cell["fresh_model_and_optimizer"])
        for cell in capacity_cells
    )
    _require(
        synchronized == expected_synchronized and isolated == expected_isolated,
        "report capacity-cell execution flags differ from the sealed probe",
    )
    _integer(
        memory["evidence_process_peak_allocated_bytes_diagnostic_only"],
        "diagnostic evidence-process allocated bytes",
    )
    _integer(
        memory["evidence_process_peak_reserved_bytes_diagnostic_only"],
        "diagnostic evidence-process reserved bytes",
    )
    memory_passed = bool(
        full_branch
        and synchronized
        and isolated
        and all(cell["status"] == "passed" for cell in capacity_cells)
        and reserved <= MAXIMUM_TUNED_RESERVED_BYTES
    )
    _require(
        _boolean(memory["passed"], "report memory pass flag") == memory_passed,
        "report memory pass flag is inconsistent",
    )

    performance = _mapping(checks["performance"], "performance check")
    _exact_keys(
        performance,
        {
            "baseline",
            "tuned",
            "throughput_ratio_basis",
            "throughput_ratio",
            "per_device_throughput_ratio",
            "minimum_device_throughput_ratio",
            "clean_train_steps_ratio_basis",
            "per_device_clean_train_steps_ratio",
            "minimum_clean_train_steps_ratio",
            "clean_train_steps_throughput_passed",
            "minimum_throughput_ratio",
            "throughput_passed",
        },
        "performance check",
    )
    baseline_projection = _validate_report_projection(performance["baseline"], "baseline")
    tuned_projection = _validate_report_projection(performance["tuned"], "tuned")
    _require(
        baseline_projection["available"] is True and tuned_projection["available"] is full_branch,
        "projection availability differs from qualification branch",
    )
    _require(
        baseline_projection["cross_profile_shared_max_applied"] is full_branch
        and tuned_projection["cross_profile_shared_max_applied"] is full_branch,
        "cross-profile shared component-max flag differs from qualification branch",
    )
    if full_branch:
        for device in gpu_uuids:
            for panel in PANELS:
                baseline_components = baseline_projection["per_device_componentwise_maxima_seconds"][device][
                    panel
                ]
                tuned_components = tuned_projection["per_device_componentwise_maxima_seconds"][device][panel]
                _require(
                    all(
                        baseline_components[field] == tuned_components[field]
                        for field in PROFILE_INDEPENDENT_PROJECTION_COMPONENT_FIELDS
                    ),
                    "profile-independent projection components are not shared maxima",
                )
                baseline_standardized = baseline_projection["standardized_worst_shape_training"][device][
                    panel
                ]
                tuned_standardized = tuned_projection["standardized_worst_shape_training"][device][panel]
                _require(
                    baseline_standardized["global_maximum_proof_digest"]
                    == tuned_standardized["global_maximum_proof_digest"]
                    and baseline_standardized["maximum_token_length"]
                    == tuned_standardized["maximum_token_length"]
                    and baseline_standardized["timed_padded_token_elements"]
                    == tuned_standardized["timed_padded_token_elements"],
                    "baseline/tuned timing did not use the same standardized worst-shape workload",
                )
    expected_device_throughput = (
        {
            device: float(baseline_projection["per_device_production_raw_worker_seconds"][device])
            / float(tuned_projection["per_device_production_raw_worker_seconds"][device])
            for device in baseline_projection["per_device_production_raw_worker_seconds"]
        }
        if full_branch
        else {device: 0.0 for device in gpu_uuids}
    )
    observed_device_throughput = _mapping(
        performance["per_device_throughput_ratio"], "per-device throughput ratios"
    )
    _require(
        set(observed_device_throughput) == set(expected_device_throughput),
        "per-device throughput UUIDs changed",
    )
    for device, value in expected_device_throughput.items():
        _same_number(observed_device_throughput[device], value, f"{device} throughput ratio")
    expected_throughput = min(expected_device_throughput.values())
    _require(
        performance["throughput_ratio_basis"]
        == "full_production_projection_excluding_additive_synthetic_host_stress",
        "throughput ratio basis changed",
    )
    _same_number(performance["throughput_ratio"], expected_throughput, "throughput ratio")
    _same_number(
        performance["minimum_device_throughput_ratio"],
        expected_throughput,
        "minimum device throughput ratio",
    )
    _same_number(
        performance["minimum_throughput_ratio"],
        MINIMUM_THROUGHPUT_RATIO,
        "minimum throughput ratio",
    )
    production_throughput_passed = full_branch and all(
        value >= MINIMUM_THROUGHPUT_RATIO for value in expected_device_throughput.values()
    )
    expected_clean_ratios = (
        {
            device: float(baseline_projection["per_device_clean_train_steps_worker_seconds"][device])
            / float(tuned_projection["per_device_clean_train_steps_worker_seconds"][device])
            for device in baseline_projection["per_device_clean_train_steps_worker_seconds"]
        }
        if full_branch
        else {device: 0.0 for device in gpu_uuids}
    )
    observed_clean_ratios = _mapping(
        performance["per_device_clean_train_steps_ratio"],
        "per-device clean train_steps ratios",
    )
    _require(
        performance["clean_train_steps_ratio_basis"] == "clean_shared_train_steps_wall_projection_only"
        and set(observed_clean_ratios) == set(expected_clean_ratios),
        "clean train_steps ratio basis or UUIDs changed",
    )
    for device, value in expected_clean_ratios.items():
        _same_number(observed_clean_ratios[device], value, f"{device} clean train_steps ratio")
    expected_minimum_clean = min(expected_clean_ratios.values())
    _same_number(
        performance["minimum_clean_train_steps_ratio"],
        expected_minimum_clean,
        "minimum clean train_steps ratio",
    )
    clean_throughput_passed = full_branch and all(
        value >= MINIMUM_THROUGHPUT_RATIO for value in expected_clean_ratios.values()
    )
    _require(
        _boolean(
            performance["clean_train_steps_throughput_passed"],
            "clean train_steps throughput pass flag",
        )
        == clean_throughput_passed,
        "clean train_steps throughput pass flag is inconsistent",
    )
    throughput_passed = production_throughput_passed and clean_throughput_passed
    _require(
        _boolean(performance["throughput_passed"], "throughput pass flag") == throughput_passed,
        "throughput pass flag is inconsistent",
    )

    projections = _mapping(payload["projections"], "top-level projections")
    _exact_keys(projections, PROFILES, "top-level projections")
    for profile, detailed in (("baseline", baseline_projection), ("tuned", tuned_projection)):
        projection = _mapping(projections[profile], f"top-level {profile} projection")
        _exact_keys(
            projection,
            {"projected_wall_seconds", "raw_wall_seconds", "safety_multiplier", "passed"},
            f"top-level {profile} projection",
        )
        _same_number(
            projection["projected_wall_seconds"],
            float(detailed["projected_worker_seconds"]),
            f"top-level {profile} projected wall seconds",
        )
        _same_number(
            projection["raw_wall_seconds"],
            float(detailed["raw_worker_seconds"]),
            f"top-level {profile} raw wall seconds",
        )
        _same_number(
            projection["safety_multiplier"],
            PROJECTION_SAFETY_MULTIPLIER,
            f"top-level {profile} safety multiplier",
        )
        _require(
            _boolean(projection["passed"], f"top-level {profile} projection pass flag")
            == bool(detailed["passed"]),
            f"top-level {profile} projection pass flag is inconsistent",
        )

    baseline_eligible = bool(
        process_passed["baseline"]
        and state_flags["baseline"]
        and data_flags["baseline"]
        and baseline_replay_summary["passed"]
        and cross_gpu_summaries["baseline"]["passed"]
        and baseline_projection["passed"]
    )
    tuned_eligible = bool(
        full_branch
        and process_passed["tuned"]
        and state_flags["tuned"]
        and data_flags["tuned"]
        and state_flags["cross"]
        and data_flags["cross"]
        and baseline_summary["output_thresholds_passed"]
        and registered_passed
        and baseline_summary["vector_thresholds_passed"]
        and replay_summary["passed"]
        and cross_gpu_summaries["tuned"]["passed"]
        and memory_passed
        and throughput_passed
        and tuned_projection["passed"]
    )
    eligibility = _mapping(payload["eligibility"], "qualification eligibility")
    _exact_keys(eligibility, PROFILES, "qualification eligibility")
    _require(
        _boolean(eligibility["baseline"], "baseline eligibility") == baseline_eligible
        and _boolean(eligibility["tuned"], "tuned eligibility") == tuned_eligible,
        "qualification eligibility is inconsistent with replayed checks",
    )
    profiles = _mapping(payload["profiles"], "qualification profiles")
    _exact_keys(profiles, PROFILES, "qualification profiles")
    for profile, expected_eligibility in (("baseline", baseline_eligible), ("tuned", tuned_eligible)):
        profile_result = _mapping(profiles[profile], f"{profile} profile result")
        _exact_keys(profile_result, {"qualification_passed"}, f"{profile} profile result")
        _require(
            _boolean(profile_result["qualification_passed"], f"{profile} profile pass flag")
            == expected_eligibility,
            f"{profile} profile pass flag is inconsistent",
        )
    expected_candidate = "tuned" if tuned_eligible else "baseline" if baseline_eligible else None
    _require(payload["selection_candidate"] == expected_candidate, "selection candidate is inconsistent")
    _require(
        _boolean(payload["selection_authorized"], "selection authorization flag") is False,
        "qualification report must not authorize profile selection",
    )
    _require(
        _boolean(payload["overall_qualification_passed"], "overall qualification flag")
        == (expected_candidate is not None),
        "overall qualification flag is inconsistent",
    )
    _require(
        _boolean(payload["outcomes_seen"], "report outcomes flag") is False
        and _boolean(payload["itt_ledger_created"], "report ITT flag") is False
        and _boolean(payload["g01_launch_authorized"], "report G01 flag") is False,
        "qualification report crossed an outcomes, ITT, or G01 boundary",
    )
    _require(
        payload["weight_updates_scope"] == "engineering_qualification_only",
        "qualification weight-update scope changed",
    )
    return copy.deepcopy(dict(payload))


def write_qualification_report(
    *,
    evidence: Mapping[str, Any],
    output: str | Path,
) -> dict[str, Any]:
    """Create the canonical report once inside the dedicated engineering root."""

    report = validate_qualification_report(create_qualification_report(evidence))
    target = Path(output).resolve()
    engineering_root = Path(str(evidence["engineering_scope"]["root"])).resolve()
    _require(
        target.parent == engineering_root and target.name == "profile-qualification.json",
        "qualification report must use the canonical engineering-only output path",
    )
    engineering_root.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(report, sort_keys=True, indent=2, allow_nan=False) + "\n"
    try:
        descriptor = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError as error:
        raise QualificationError("qualification report already exists") from error
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(serialized)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        with suppress(FileNotFoundError):
            target.unlink()
        raise
    return report


def replay_authenticated_evidence(
    *,
    evidence_path: str | Path,
    output: str | Path,
) -> dict[str, Any]:
    """Replay a freeze-bound producer evidence file and write its report once."""

    source = Path(evidence_path).resolve()
    try:
        raw = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise QualificationError("could not read strict qualification evidence JSON") from error
    if not isinstance(raw, Mapping):
        raise QualificationError("qualification evidence file must contain one JSON object")
    evidence = validate_qualification_evidence(raw)
    engineering_root = Path(str(evidence["engineering_scope"]["root"])).resolve()
    _require(source.parent == engineering_root, "evidence file lies outside its engineering root")
    return write_qualification_report(evidence=evidence, output=output)


def main(argv: Sequence[str] | None = None) -> int:
    """Replay evidence emitted by the freeze-bound qualification producer."""

    parser = argparse.ArgumentParser(
        description="Replay freeze-bound actual-model H200 qualification producer evidence"
    )
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    report = replay_authenticated_evidence(evidence_path=args.evidence, output=args.output)
    print(
        json.dumps(
            {
                "output": str(args.output.resolve()),
                "report_digest": report["report_digest"],
                "selection_candidate": report["selection_candidate"],
                "selection_authorized": False,
            },
            sort_keys=True,
            allow_nan=False,
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised by the frozen launcher
    raise SystemExit(main())
