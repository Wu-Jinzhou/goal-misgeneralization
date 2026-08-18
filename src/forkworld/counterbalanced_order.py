"""Exact counterbalanced data and atomic streams for the E20 order study.

The three training components have identical 64-codeword visible support but
different labels and tuple-dependent multiplicities.  All balancing variables
in this module are semantic latents: the model interface is frozen to the six
varying signs and two constant presence channels.

This module deliberately has no protocol, configuration, or runner dependency.
It constructs data and index streams, then independently audits the realized
integer counts.  The protocol can therefore reuse a component stream bit for
bit in every schedule without giving the model a component or schedule cue.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from itertools import permutations
from types import MappingProxyType
from typing import Any, Literal, cast

import numpy as np
from numpy.typing import NDArray

from .competing import decode_competing_rules, make_competing_factorial_dataset
from .data import SemanticBatch
from .handoff import semantic_batch_digest, stable_state_digest

Goal = Literal["P", "Q", "Y"]
OrderSchedule = Literal[
    "p_q_y",
    "p_y_q",
    "q_p_y",
    "q_y_p",
    "y_p_q",
    "y_q_p",
]

GOALS: tuple[Goal, ...] = ("P", "Q", "Y")
ORDER_SCHEDULES: tuple[OrderSchedule, ...] = cast(
    tuple[OrderSchedule, ...],
    tuple("_".join(goal.lower() for goal in order) for order in permutations(GOALS)),
)
FEATURE_NAMES = (
    "P",
    "P_present",
    "R_1",
    "R_2",
    "R_3",
    "Q_present",
    "Q_1",
    "Q_2",
)
VARYING_CHANNELS = ("P", "R_1", "R_2", "R_3", "Q_1", "Q_2")
RAW_CHANNEL_ORDER = VARYING_CHANNELS

RAW_CODEWORD_COUNT = 64
CANDIDATE_TUPLE_COUNT = 8
WEIGHTED_STRATUM_COUNT = 96
ROWS_PER_WEIGHT_UNIT = 96
BATCH_SIZE = 288
BATCHES_PER_PRESENTATION = 32
EXAMPLES_PER_WEIGHT_UNIT_PER_BATCH = 3
PRESENTATIONS = 8

FOLD_DIGEST_SHA256 = "25d5e9584a2ea74d42d48a673645a3b34c1b76e75e8230bf5722985e66968ec7"
CONTROL_DIGEST_SHA256 = "c7a186308102e807e594fd76b3fc5e7ef1678314a95de067b6fd347b390e53d0"
CONTROL_BIT_STRING = "0110001001011011111001011000100101011110100110000010011010110101"

DEFAULT_ID_OFFSET = 20_000_000_000
DEFAULT_PANEL_ID_OFFSET = 30_000_000_000

IntArray = NDArray[np.int64]
SignArray = NDArray[np.int8]


def _stable_seed(seed: int, *parts: object) -> int:
    digest = hashlib.blake2b(digest_size=8, person=b"forke20")
    for part in (seed, *parts):
        payload = str(part).encode("utf-8")
        digest.update(len(payload).to_bytes(4, "little"))
        digest.update(payload)
    return int.from_bytes(digest.digest(), "little")


def _positive_integer(value: int, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or int(value) != value or int(value) < 1:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def _nonnegative_integer(value: int, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or int(value) != value or int(value) < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return int(value)


def _normalize_goal(goal: str) -> Goal:
    normalized = str(goal).upper()
    if normalized not in GOALS:
        raise ValueError(f"goal must be one of {GOALS}")
    return cast(Goal, normalized)


def _schedule_order(schedule: str) -> tuple[Goal, Goal, Goal]:
    if schedule not in ORDER_SCHEDULES:
        raise ValueError(f"schedule must be one of {ORDER_SCHEDULES}")
    parts = tuple(part.upper() for part in schedule.split("_"))
    return cast(tuple[Goal, Goal, Goal], parts)


def _readonly(values: NDArray[Any], dtype: np.dtype[Any] | type[Any]) -> NDArray[Any]:
    result = np.array(values, dtype=dtype, copy=True)
    result.setflags(write=False)
    return result


def _array_digest(values: NDArray[Any], *, person: bytes) -> str:
    array = np.ascontiguousarray(values)
    digest = hashlib.blake2b(digest_size=32, person=person)
    dtype = array.dtype.str.encode("ascii")
    digest.update(len(dtype).to_bytes(2, "little"))
    digest.update(dtype)
    digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _fraction(correct: int, total: int) -> dict[str, int | float]:
    if total < 1 or correct < 0 or correct > total:
        raise RuntimeError("invalid exact-accuracy count")
    divisor = math.gcd(correct, total)
    return {
        "correct": int(correct),
        "total": int(total),
        "numerator": int(correct // divisor),
        "denominator": int(total // divisor),
        "value": float(correct / total),
    }


def _require_fraction(correct: int, total: int, numerator: int, denominator: int, name: str) -> None:
    if correct * denominator != total * numerator:
        raise RuntimeError(
            f"{name} must equal {numerator}/{denominator}; observed {correct}/{total}"
        )


@dataclass(frozen=True)
class _CanonicalTable:
    raw_id: IntArray
    candidate_tuple_id: IntArray
    fold_id: IntArray
    truth_table_control: SignArray
    channels: Mapping[str, SignArray]
    goals: Mapping[Goal, SignArray]
    majority: SignArray


def _fold_control(raw_id: IntArray, tuple_id: IntArray) -> tuple[IntArray, SignArray]:
    fold = np.empty(RAW_CODEWORD_COUNT, dtype=np.int64)
    for candidate_id in range(CANDIDATE_TUPLE_COUNT):
        compatible = np.flatnonzero(tuple_id == candidate_id)
        if len(compatible) != 8:
            raise RuntimeError("canonical candidate tuple does not have eight raw encodings")
        fold[compatible] = np.arange(8, dtype=np.int64)
    residues = (fold - tuple_id) % 8
    control = np.where(np.isin(residues, (0, 1, 3, 4)), 1, -1).astype(np.int8)
    return fold, control


def _canonical_table() -> _CanonicalTable:
    """Reindex the existing competing-factorial convention to E20 raw-ID order."""

    source = make_competing_factorial_dataset(
        repeats=1,
        k_q=2,
        k_y=3,
        seed=0,
        control_seed=0,
        max_k_q=2,
        max_k_y=3,
        state_dim=0,
        split="e20_canonical_source",
        id_offset=0,
    )
    source_features = source.feature_names(max_k=3, include_state=False)
    if source_features != FEATURE_NAMES:  # pragma: no cover - protects an upstream interface edit
        raise RuntimeError(
            f"competing factorial feature interface changed: {source_features!r}"
        )
    raw_id = np.zeros(len(source), dtype=np.int64)
    for bit, name in enumerate(reversed(VARYING_CHANNELS)):
        raw_id |= (np.asarray(source.channels[name]) > 0).astype(np.int64) << bit
    order = np.argsort(raw_id)
    raw_id = raw_id[order]
    if not np.array_equal(raw_id, np.arange(RAW_CODEWORD_COUNT, dtype=np.int64)):
        raise RuntimeError("competing factorial source did not enumerate all E20 raw codewords")

    channels = {
        name: np.asarray(source.channels[name], dtype=np.int8)[order]
        for name in FEATURE_NAMES
    }
    p = channels["P"]
    q = (channels["Q_1"] * channels["Q_2"]).astype(np.int8)
    y = (channels["R_1"] * channels["R_2"] * channels["R_3"]).astype(np.int8)
    majority = np.where(p + q + y > 0, 1, -1).astype(np.int8)
    tuple_id = (
        4 * (p > 0).astype(np.int64)
        + 2 * (q > 0).astype(np.int64)
        + (y > 0).astype(np.int64)
    )
    fold, control = _fold_control(raw_id, tuple_id)
    return _CanonicalTable(
        raw_id=_readonly(raw_id, np.int64),
        candidate_tuple_id=_readonly(tuple_id, np.int64),
        fold_id=_readonly(fold, np.int64),
        truth_table_control=_readonly(control, np.int8),
        channels=MappingProxyType(
            {name: _readonly(values, np.int8) for name, values in channels.items()}
        ),
        goals=MappingProxyType(
            {
                "P": _readonly(p, np.int8),
                "Q": _readonly(q, np.int8),
                "Y": _readonly(y, np.int8),
            }
        ),
        majority=_readonly(majority, np.int8),
    )


def counterbalanced_fold_ids() -> IntArray:
    """Return the frozen eight-fold assignment in canonical raw-ID order."""

    return _readonly(_canonical_table().fold_id, np.int64)


def counterbalanced_truth_table_control() -> SignArray:
    """Return the frozen balanced control labels in canonical raw-ID order."""

    return _readonly(_canonical_table().truth_table_control, np.int8)


def audit_counterbalanced_fold_control() -> dict[str, Any]:
    """Reconstruct and verify the prospectively frozen fold/control truth tables."""

    table = _canonical_table()
    records = [
        {
            "fold": int(table.fold_id[index]),
            "raw_id": int(table.raw_id[index]),
            "tuple_id": int(table.candidate_tuple_id[index]),
        }
        for index in range(RAW_CODEWORD_COUNT)
    ]
    fold_payload = json.dumps(records, sort_keys=True, separators=(",", ":")).encode()
    control_payload = json.dumps(
        table.truth_table_control.tolist(), separators=(",", ":")
    ).encode()
    fold_digest = hashlib.sha256(fold_payload).hexdigest()
    control_digest = hashlib.sha256(control_payload).hexdigest()
    if fold_digest != FOLD_DIGEST_SHA256:
        raise RuntimeError("frozen E20 fold digest does not match the design memo")
    if control_digest != CONTROL_DIGEST_SHA256:
        raise RuntimeError("frozen E20 truth-table-control digest does not match the design memo")
    bits = "".join("1" if value > 0 else "0" for value in table.truth_table_control)
    if bits != CONTROL_BIT_STRING:
        raise RuntimeError("frozen E20 truth-table-control bit string changed")

    for name, values in (*table.goals.items(), ("majority", table.majority)):
        if int(np.sum(values == table.truth_table_control)) != 32:
            raise RuntimeError(f"truth-table control is not at chance with {name}")
    for axis_name, axis in (
        ("tuple", table.candidate_tuple_id),
        ("fold", table.fold_id),
    ):
        for value in range(8):
            selected = table.truth_table_control[axis == value]
            if len(selected) != 8 or int(np.sum(selected > 0)) != 4:
                raise RuntimeError(f"truth-table control is not 4/4 balanced in {axis_name} {value}")
    for fold in range(8):
        if set(table.candidate_tuple_id[table.fold_id == fold].tolist()) != set(range(8)):
            raise RuntimeError(f"fold {fold} does not contain every candidate tuple exactly once")

    return {
        "raw_codeword_count": RAW_CODEWORD_COUNT,
        "candidate_tuple_count": CANDIDATE_TUPLE_COUNT,
        "fold_count": 8,
        "fold_digest_sha256": fold_digest,
        "control_digest_sha256": control_digest,
        "control_bit_string": bits,
        "control_positive": int(np.sum(table.truth_table_control > 0)),
        "control_negative": int(np.sum(table.truth_table_control < 0)),
        "each_fold_has_all_tuples": True,
        "control_balanced_within_tuple": True,
        "control_balanced_within_fold": True,
        "control_chance_with_p_q_y_majority": True,
    }


def _component_weights(table: _CanonicalTable, goal: Goal) -> IntArray:
    selected = table.goals[goal]
    competitors = [table.goals[name] for name in GOALS if name != goal]
    unanimous = (table.goals["P"] == table.goals["Q"]) & (
        table.goals["Q"] == table.goals["Y"]
    )
    selected_minority = (selected != competitors[0]) & (competitors[0] == competitors[1])
    return np.where(unanimous | selected_minority, 2, 1).astype(np.int64)


def _weighted_rows(
    raw_strata: IntArray,
    copies_per_raw: IntArray,
    *,
    rows_per_stratum: int,
) -> dict[str, IntArray]:
    stratum_raw = np.repeat(raw_strata, copies_per_raw)
    copy_id = np.concatenate(
        [np.arange(copies, dtype=np.int64) for copies in copies_per_raw.tolist()]
    )
    stratum_id = np.arange(len(stratum_raw), dtype=np.int64)
    return {
        "raw_codeword_id": np.repeat(stratum_raw, rows_per_stratum),
        "weighted_stratum_id": np.repeat(stratum_id, rows_per_stratum),
        "stratum_copy_id": np.repeat(copy_id, rows_per_stratum),
        "replica_id": np.tile(np.arange(rows_per_stratum, dtype=np.int64), len(stratum_raw)),
    }


def _check_id_range(id_offset: int, n: int) -> int:
    id_offset = _nonnegative_integer(id_offset, "id_offset")
    if id_offset > np.iinfo(np.int64).max - n:
        raise ValueError("id_offset + dataset size exceeds the int64 ID range")
    return id_offset


def _make_semantic_batch(
    *,
    table: _CanonicalTable,
    row_fields: Mapping[str, IntArray],
    target: SignArray,
    seed: int,
    split: str,
    id_offset: int,
    rows_per_weight_unit: int,
    target_goal: str,
    washout: bool,
) -> SemanticBatch:
    raw_id = np.asarray(row_fields["raw_codeword_id"], dtype=np.int64)
    n = len(raw_id)
    order = np.random.default_rng(_stable_seed(seed, split, "row-order")).permutation(n)
    raw_id = raw_id[order]
    ids = np.arange(_check_id_range(id_offset, n), id_offset + n, dtype=np.int64)
    channels = {
        name: np.asarray(table.channels[name][raw_id], dtype=np.int8) for name in FEATURE_NAMES
    }
    goals = {name: np.asarray(table.goals[name][raw_id], dtype=np.int8) for name in GOALS}
    y = goals["Y"]
    realized_target = np.asarray(target, dtype=np.int8)[order]
    latents: dict[str, NDArray[Any]] = {
        name: np.asarray(values, dtype=np.int64)[order]
        for name, values in row_fields.items()
    }
    latents.update(
        {
            "candidate_tuple_id": table.candidate_tuple_id[raw_id],
            "fold_id": table.fold_id[raw_id],
            "truth_table_control": table.truth_table_control[raw_id],
            "P_goal": goals["P"],
            "Q_goal": goals["Q"],
            "Y_goal": goals["Y"],
            "majority_goal": table.majority[raw_id],
        }
    )
    metadata = {
        "dataset": "counterbalanced_order_washout" if washout else "counterbalanced_order_component",
        "condition": "concordant_washout" if washout else f"S_{target_goal}",
        "split": split,
        "seed": int(seed),
        "target_goal": target_goal,
        "raw_channel_order": RAW_CHANNEL_ORDER,
        "raw_codeword_count": RAW_CODEWORD_COUNT,
        "candidate_tuple_count": CANDIDATE_TUPLE_COUNT,
        "weighted_stratum_count": WEIGHTED_STRATUM_COUNT,
        "rows_per_weight_unit": rows_per_weight_unit,
        "active_k": 3,
        "max_k": 3,
        "r_active_mask": (True, True, True),
        "padding_seed": 0,
        "neutralize_padding": True,
        "enforce_exact_code": True,
        "target_rule": "parity",
        "k_p": 1,
        "max_k_p": 1,
        "k_q": 2,
        "max_k_q": 2,
        "q_active_mask": (True, True),
        "k_y": 3,
        "max_k_y": 3,
        "state_dim": 0,
        "feature_names": FEATURE_NAMES,
        "fold_digest_sha256": FOLD_DIGEST_SHA256,
        "control_digest_sha256": CONTROL_DIGEST_SHA256,
    }
    return SemanticBatch(
        y=y,
        target=realized_target,
        reward=realized_target.astype(np.float32),
        channels=channels,
        latents=latents,
        state=None,
        nuisance_targets=None,
        sample_id=ids,
        state_id=ids,
        episode_id=ids,
        step_id=np.zeros(n, dtype=np.int64),
        metadata=metadata,
    )


def make_counterbalanced_component(
    goal: Goal | str,
    seed: int,
    *,
    rows_per_weight_unit: int = ROWS_PER_WEIGHT_UNIT,
    id_offset: int | None = None,
) -> SemanticBatch:
    """Construct one exact weighted component ``S_G``.

    ``SemanticBatch.target`` is retargeted to ``G`` while ``SemanticBatch.y``
    remains the degree-three ``Y`` parity.  This distinction is intentional:
    training reads ``target`` and semantic evaluation can continue to read ``y``.
    """

    selected = _normalize_goal(goal)
    rows_per_weight_unit = _positive_integer(rows_per_weight_unit, "rows_per_weight_unit")
    if id_offset is None:
        id_offset = DEFAULT_ID_OFFSET + GOALS.index(selected) * 1_000_000_000
    table = _canonical_table()
    weights = _component_weights(table, selected)
    row_fields = _weighted_rows(
        table.raw_id,
        weights,
        rows_per_stratum=rows_per_weight_unit,
    )
    raw_id = row_fields["raw_codeword_id"]
    target = table.goals[selected][raw_id]
    batch = _make_semantic_batch(
        table=table,
        row_fields=row_fields,
        target=target,
        seed=int(seed),
        split=f"component_{selected}",
        id_offset=id_offset,
        rows_per_weight_unit=rows_per_weight_unit,
        target_goal=selected,
        washout=False,
    )
    audit_counterbalanced_component(
        batch,
        expected_goal=selected,
        rows_per_weight_unit=rows_per_weight_unit,
    )
    return batch


def make_concordant_washout(
    seed: int,
    *,
    rows_per_weight_unit: int = ROWS_PER_WEIGHT_UNIT,
    id_offset: int = DEFAULT_ID_OFFSET + 3_000_000_000,
) -> SemanticBatch:
    """Construct the 16-codeword concordant washout with six strata per codeword."""

    rows_per_weight_unit = _positive_integer(rows_per_weight_unit, "rows_per_weight_unit")
    table = _canonical_table()
    unanimous = (table.goals["P"] == table.goals["Q"]) & (
        table.goals["Q"] == table.goals["Y"]
    )
    compatible = table.raw_id[unanimous]
    if len(compatible) != 16:  # pragma: no cover - canonical truth-table assertion
        raise RuntimeError("canonical washout support must contain 16 codewords")
    row_fields = _weighted_rows(
        compatible,
        np.full(len(compatible), 6, dtype=np.int64),
        rows_per_stratum=rows_per_weight_unit,
    )
    raw_id = row_fields["raw_codeword_id"]
    batch = _make_semantic_batch(
        table=table,
        row_fields=row_fields,
        target=table.goals["Y"][raw_id],
        seed=int(seed),
        split="washout",
        id_offset=id_offset,
        rows_per_weight_unit=rows_per_weight_unit,
        target_goal="concordant",
        washout=True,
    )
    audit_concordant_washout(batch, rows_per_weight_unit=rows_per_weight_unit)
    return batch


def make_counterbalanced_factorial_panel(
    *, id_offset: int = DEFAULT_PANEL_ID_OFFSET
) -> SemanticBatch:
    """Return the canonical ordered 64-codeword evaluation and probe panel."""

    table = _canonical_table()
    n = RAW_CODEWORD_COUNT
    ids = np.arange(_check_id_range(id_offset, n), id_offset + n, dtype=np.int64)
    batch = SemanticBatch(
        y=table.goals["Y"],
        target=table.goals["Y"],
        reward=table.goals["Y"].astype(np.float32),
        channels={name: table.channels[name] for name in FEATURE_NAMES},
        latents={
            "raw_codeword_id": table.raw_id,
            "candidate_tuple_id": table.candidate_tuple_id,
            "fold_id": table.fold_id,
            "truth_table_control": table.truth_table_control,
            "P_goal": table.goals["P"],
            "Q_goal": table.goals["Q"],
            "Y_goal": table.goals["Y"],
            "majority_goal": table.majority,
        },
        sample_id=ids,
        state_id=ids,
        episode_id=ids,
        step_id=np.zeros(n, dtype=np.int64),
        metadata={
            "dataset": "counterbalanced_order_factorial",
            "condition": "factorial_eval",
            "split": "factorial_eval",
            "raw_channel_order": RAW_CHANNEL_ORDER,
            "raw_codeword_count": RAW_CODEWORD_COUNT,
            "candidate_tuple_count": CANDIDATE_TUPLE_COUNT,
            "active_k": 3,
            "max_k": 3,
            "r_active_mask": (True, True, True),
            "padding_seed": 0,
            "neutralize_padding": True,
            "enforce_exact_code": True,
            "target_rule": "parity",
            "k_p": 1,
            "max_k_p": 1,
            "k_q": 2,
            "max_k_q": 2,
            "q_active_mask": (True, True),
            "k_y": 3,
            "max_k_y": 3,
            "state_dim": 0,
            "feature_names": FEATURE_NAMES,
            "fold_digest_sha256": FOLD_DIGEST_SHA256,
            "control_digest_sha256": CONTROL_DIGEST_SHA256,
        },
    )
    audit_counterbalanced_factorial_panel(batch)
    return batch


def _visible_interface(batch: SemanticBatch) -> tuple[tuple[str, ...], NDArray[np.bool_], NDArray[Any]]:
    names = batch.feature_names(max_k=3, include_state=False)
    active = batch.feature_active_mask(max_k=3, include_state=False)
    features = batch.features(max_k=3, include_state=False)
    return names, active, features


def _audit_interface(batch: SemanticBatch, *, require_all_raw: bool) -> dict[str, Any]:
    names, active, features = _visible_interface(batch)
    if names != FEATURE_NAMES:
        raise RuntimeError(f"E20 feature names/order changed: {names!r}")
    if features.shape != (len(batch), 8):
        raise RuntimeError("E20 model-facing feature matrix must have width eight")
    if not np.all(active):
        raise RuntimeError("all and only the eight registered E20 feature columns must be active")
    expected_channels = set(FEATURE_NAMES)
    if set(batch.channels) != expected_channels:
        raise RuntimeError("E20 batch contains an extra or missing model-visible channel")
    if batch.state is not None or batch.nuisance_targets is not None:
        raise RuntimeError("E20 batch may not expose state or nuisance targets")
    if batch.active_k != 3 or int(batch.metadata.get("max_k", -1)) != 3:
        raise RuntimeError("E20 Y degree and maximum degree must both equal three")
    if int(batch.metadata.get("k_q", -1)) != 2 or int(batch.metadata.get("max_k_q", -1)) != 2:
        raise RuntimeError("E20 Q degree and maximum degree must both equal two")
    if int(batch.metadata.get("k_p", -1)) != 1 or int(batch.metadata.get("max_k_p", -1)) != 1:
        raise RuntimeError("E20 P degree and maximum degree must both equal one")
    if batch.metadata.get("r_active_mask") != (True, True, True):
        raise RuntimeError("E20 Y active mask changed")
    if batch.metadata.get("q_active_mask") != (True, True):
        raise RuntimeError("E20 Q active mask changed")
    if not np.all(features[:, 1] == 1) or not np.all(features[:, 5] == 1):
        raise RuntimeError("E20 presence columns must be identically +1")
    for column in (0, 2, 3, 4, 6, 7):
        if set(np.unique(features[:, column]).tolist()) != {-1.0, 1.0}:
            raise RuntimeError("every non-presence E20 input channel must vary over both signs")
    forbidden = ("id", "copy", "stratum", "replica", "component", "schedule", "view")
    if any(any(token in name.lower() for token in forbidden) for name in names):
        raise RuntimeError("metadata-only identity leaked into E20 feature names")
    if int(batch.metadata.get("state_dim", -1)) != 0:
        raise RuntimeError("E20 state_dim must be zero")

    raw_ids = np.asarray(batch.latents.get("raw_codeword_id"), dtype=np.int64)
    if raw_ids.shape != (len(batch),):
        raise RuntimeError("E20 raw-codeword IDs must align with rows")
    varying = features[:, (0, 2, 3, 4, 6, 7)].astype(np.int8, copy=False)
    decoded = np.zeros(len(batch), dtype=np.int64)
    for column in range(6):
        decoded |= (varying[:, column] > 0).astype(np.int64) << (5 - column)
    if not np.array_equal(decoded, raw_ids):
        raise RuntimeError("E20 raw-codeword IDs do not decode the six varying signs")
    support = np.unique(varying, axis=0)
    expected_support = RAW_CODEWORD_COUNT if require_all_raw else 16
    if len(support) != expected_support:
        raise RuntimeError(f"E20 visible support must contain {expected_support} codewords")
    support_by_id = np.column_stack(
        [
            np.asarray(
                [np.unique(varying[raw_ids == raw_id, column]).item() for raw_id in np.unique(raw_ids)],
                dtype=np.int8,
            )
            for column in range(6)
        ]
    )
    return {
        "feature_names": list(names),
        "feature_width": int(features.shape[1]),
        "feature_active_mask": active.tolist(),
        "presence_columns": {"P_present": 1, "Q_present": 1},
        "varying_channel_count": 6,
        "visible_support_size": len(support),
        "visible_support_digest": _array_digest(support_by_id, person=b"e20support"),
        "state_absent": True,
        "nuisance_absent": True,
        "identity_cues_absent": True,
        "padding_absent": True,
    }


def _audit_common_latents(batch: SemanticBatch) -> dict[str, NDArray[Any]]:
    required = {
        "raw_codeword_id",
        "candidate_tuple_id",
        "fold_id",
        "truth_table_control",
        "P_goal",
        "Q_goal",
        "Y_goal",
        "majority_goal",
    }
    missing = required - set(batch.latents)
    if missing:
        raise RuntimeError(f"E20 batch is missing audit latents {sorted(missing)}")
    values = {name: np.asarray(batch.latents[name]) for name in required}
    if any(array.shape != (len(batch),) for array in values.values()):
        raise RuntimeError("E20 audit latents must be row-aligned vectors")
    rules = decode_competing_rules(batch)
    expected = {
        "P_goal": rules["P"],
        "Q_goal": rules["Q"],
        "Y_goal": rules["Y_code"],
    }
    for name, goal in expected.items():
        if not np.array_equal(values[name], goal):
            raise RuntimeError(f"E20 latent {name} disagrees with visible candidate rule")
    if not np.array_equal(np.asarray(batch.y), expected["Y_goal"]):
        raise RuntimeError("SemanticBatch.y must remain the visible Y_code parity")
    majority = np.where(
        expected["P_goal"] + expected["Q_goal"] + expected["Y_goal"] > 0, 1, -1
    ).astype(np.int8)
    if not np.array_equal(values["majority_goal"], majority):
        raise RuntimeError("E20 majority latent is inconsistent with P/Q/Y")
    tuple_id = (
        4 * (expected["P_goal"] > 0).astype(np.int64)
        + 2 * (expected["Q_goal"] > 0).astype(np.int64)
        + (expected["Y_goal"] > 0).astype(np.int64)
    )
    if not np.array_equal(values["candidate_tuple_id"], tuple_id):
        raise RuntimeError("E20 candidate-tuple IDs are inconsistent with P/Q/Y")
    table = _canonical_table()
    raw_id = values["raw_codeword_id"].astype(np.int64, copy=False)
    if not np.array_equal(values["fold_id"], table.fold_id[raw_id]):
        raise RuntimeError("E20 row fold IDs differ from the frozen assignment")
    if not np.array_equal(values["truth_table_control"], table.truth_table_control[raw_id]):
        raise RuntimeError("E20 row truth-table controls differ from the frozen mapping")
    return values


def _audit_semantic_ids(batch: SemanticBatch) -> dict[str, Any]:
    ids = {
        "sample_id": np.asarray(batch.sample_id, dtype=np.int64),
        "state_id": np.asarray(batch.state_id, dtype=np.int64),
        "episode_id": np.asarray(batch.episode_id, dtype=np.int64),
    }
    for name, values in ids.items():
        if values.shape != (len(batch),) or len(np.unique(values)) != len(batch):
            raise RuntimeError(f"E20 {name} values must be unique and row aligned")
    if not np.array_equal(ids["sample_id"], ids["state_id"]) or not np.array_equal(
        ids["sample_id"], ids["episode_id"]
    ):
        raise RuntimeError("E20 semantic ID families must share one invisible row identity")
    if not np.all(np.asarray(batch.step_id, dtype=np.int64) == 0):
        raise RuntimeError("E20 non-episodic rows must have step_id zero")
    return {
        "unique_per_dataset": True,
        "id_min": int(np.min(ids["sample_id"])),
        "id_max": int(np.max(ids["sample_id"])),
        "sample_id_digest": _array_digest(ids["sample_id"], person=b"e20ids"),
    }


def _accuracy_counts(batch: SemanticBatch, target: SignArray) -> dict[str, dict[str, int | float]]:
    latents = _audit_common_latents(batch)
    predictors = {
        "P": latents["P_goal"],
        "Q": latents["Q_goal"],
        "Y": latents["Y_goal"],
        "majority": latents["majority_goal"],
    }
    return {
        name: _fraction(int(np.sum(values == target)), len(batch))
        for name, values in predictors.items()
    }


def audit_counterbalanced_component(
    batch: SemanticBatch,
    *,
    expected_goal: Goal | str | None = None,
    rows_per_weight_unit: int | None = None,
) -> dict[str, Any]:
    """Fail closed on one component's interface, strata, labels, and exact accuracies."""

    metadata_goal = _normalize_goal(str(batch.metadata.get("target_goal", "")))
    goal = metadata_goal if expected_goal is None else _normalize_goal(expected_goal)
    if metadata_goal != goal:
        raise RuntimeError(f"component metadata says S_{metadata_goal}, expected S_{goal}")
    registered_rows = _positive_integer(
        int(batch.metadata.get("rows_per_weight_unit", 0)), "metadata.rows_per_weight_unit"
    )
    if rows_per_weight_unit is not None and registered_rows != _positive_integer(
        rows_per_weight_unit, "rows_per_weight_unit"
    ):
        raise RuntimeError("component rows_per_weight_unit differs from the requested value")
    rows_per_weight_unit = registered_rows
    expected_n = WEIGHTED_STRATUM_COUNT * rows_per_weight_unit
    if len(batch) != expected_n:
        raise RuntimeError(f"component S_{goal} must contain {expected_n} rows")
    if batch.metadata.get("dataset") != "counterbalanced_order_component":
        raise RuntimeError("batch is not registered as an E20 component")

    interface = _audit_interface(batch, require_all_raw=True)
    latents = _audit_common_latents(batch)
    ids = _audit_semantic_ids(batch)
    for name in ("weighted_stratum_id", "stratum_copy_id", "replica_id"):
        if name not in batch.latents or np.asarray(batch.latents[name]).shape != (len(batch),):
            raise RuntimeError(f"component lacks row-aligned latent {name}")
    raw_id = latents["raw_codeword_id"].astype(np.int64, copy=False)
    stratum_id = np.asarray(batch.latents["weighted_stratum_id"], dtype=np.int64)
    copy_id = np.asarray(batch.latents["stratum_copy_id"], dtype=np.int64)
    replica_id = np.asarray(batch.latents["replica_id"], dtype=np.int64)
    if set(np.unique(raw_id).tolist()) != set(range(RAW_CODEWORD_COUNT)):
        raise RuntimeError("component must contain all 64 canonical raw codewords")
    if set(np.unique(stratum_id).tolist()) != set(range(WEIGHTED_STRATUM_COUNT)):
        raise RuntimeError("component weighted-stratum IDs must be exactly 0..95")
    stratum_counts = np.bincount(stratum_id, minlength=WEIGHTED_STRATUM_COUNT)
    if not np.array_equal(
        stratum_counts, np.full(WEIGHTED_STRATUM_COUNT, rows_per_weight_unit)
    ):
        raise RuntimeError("every component weighted stratum must have the exact replica count")
    for stratum in range(WEIGHTED_STRATUM_COUNT):
        selected = stratum_id == stratum
        if len(np.unique(raw_id[selected])) != 1 or len(np.unique(copy_id[selected])) != 1:
            raise RuntimeError("weighted-stratum identity must map to one raw/copy pair")
        if not np.array_equal(np.sort(replica_id[selected]), np.arange(rows_per_weight_unit)):
            raise RuntimeError("component replicas must enumerate 0..rows_per_weight_unit-1")

    table = _canonical_table()
    expected_weights = _component_weights(table, goal)
    raw_counts = np.bincount(raw_id, minlength=RAW_CODEWORD_COUNT)
    if not np.array_equal(raw_counts, expected_weights * rows_per_weight_unit):
        raise RuntimeError("component raw-codeword multiplicities do not realize w_G")
    for raw in range(RAW_CODEWORD_COUNT):
        copies = np.unique(copy_id[raw_id == raw])
        if not np.array_equal(copies, np.arange(expected_weights[raw])):
            raise RuntimeError("component copy IDs do not realize the registered raw weight")
    tuple_raw_counts = [len(np.unique(raw_id[latents["candidate_tuple_id"] == value])) for value in range(8)]
    if tuple_raw_counts != [8] * 8:
        raise RuntimeError("every candidate tuple must contain eight raw codewords")

    target = np.asarray(batch.target, dtype=np.int8)
    if not np.array_equal(target, latents[f"{goal}_goal"]):
        raise RuntimeError(f"component S_{goal} must train on SemanticBatch.target={goal}")
    if not np.array_equal(np.asarray(batch.reward), target.astype(np.float32)):
        raise RuntimeError("component reward must equal the retargeted supervised objective")
    negative = int(np.sum(target < 0))
    positive = int(np.sum(target > 0))
    if negative != expected_n // 2 or positive != expected_n // 2:
        raise RuntimeError("component target labels must be exactly balanced")
    accuracies = _accuracy_counts(batch, target)
    _require_fraction(int(accuracies[goal]["correct"]), expected_n, 1, 1, f"S_{goal} target")
    for competitor in GOALS:
        if competitor != goal:
            _require_fraction(
                int(accuracies[competitor]["correct"]),
                expected_n,
                1,
                2,
                f"S_{goal} competitor {competitor}",
            )
    _require_fraction(
        int(accuracies["majority"]["correct"]), expected_n, 2, 3, f"S_{goal} majority"
    )
    if int(np.sum(expected_weights)) != WEIGHTED_STRATUM_COUNT:
        raise RuntimeError("component must contain exactly 96 weighted raw strata")

    return {
        "kind": "component",
        "goal": goal,
        "n_rows": len(batch),
        "raw_codeword_count": len(np.unique(raw_id)),
        "candidate_tuple_count": len(np.unique(latents["candidate_tuple_id"])),
        "raw_codewords_per_tuple": list(tuple_raw_counts),
        "weighted_stratum_count": len(np.unique(stratum_id)),
        "rows_per_weight_unit": rows_per_weight_unit,
        "weight_one_raw_codewords": int(np.sum(expected_weights == 1)),
        "weight_two_raw_codewords": int(np.sum(expected_weights == 2)),
        "raw_weights": expected_weights.tolist(),
        "raw_row_counts": raw_counts.tolist(),
        "label_counts": {"-1": negative, "+1": positive},
        "accuracy": accuracies,
        "interface": interface,
        "semantic_ids": ids,
        "semantic_batch_digest": semantic_batch_digest(batch),
        "raw_label_count_digest": stable_state_digest(
            {"raw_id": raw_id, "target": target, "stratum_id": stratum_id}
        ),
        "fold_digest_sha256": FOLD_DIGEST_SHA256,
        "control_digest_sha256": CONTROL_DIGEST_SHA256,
    }


def audit_concordant_washout(
    batch: SemanticBatch,
    *,
    rows_per_weight_unit: int | None = None,
) -> dict[str, Any]:
    """Fail closed on the 16-codeword, six-strata-per-codeword washout."""

    registered_rows = _positive_integer(
        int(batch.metadata.get("rows_per_weight_unit", 0)), "metadata.rows_per_weight_unit"
    )
    if rows_per_weight_unit is not None and registered_rows != _positive_integer(
        rows_per_weight_unit, "rows_per_weight_unit"
    ):
        raise RuntimeError("washout rows_per_weight_unit differs from the requested value")
    rows_per_weight_unit = registered_rows
    expected_n = WEIGHTED_STRATUM_COUNT * rows_per_weight_unit
    if len(batch) != expected_n:
        raise RuntimeError(f"washout must contain {expected_n} rows")
    if batch.metadata.get("dataset") != "counterbalanced_order_washout":
        raise RuntimeError("batch is not registered as the E20 washout")
    interface = _audit_interface(batch, require_all_raw=False)
    latents = _audit_common_latents(batch)
    ids = _audit_semantic_ids(batch)
    for name in ("weighted_stratum_id", "stratum_copy_id", "replica_id"):
        if name not in batch.latents or np.asarray(batch.latents[name]).shape != (len(batch),):
            raise RuntimeError(f"washout lacks row-aligned latent {name}")
    raw_id = latents["raw_codeword_id"].astype(np.int64, copy=False)
    stratum_id = np.asarray(batch.latents["weighted_stratum_id"], dtype=np.int64)
    copy_id = np.asarray(batch.latents["stratum_copy_id"], dtype=np.int64)
    replica_id = np.asarray(batch.latents["replica_id"], dtype=np.int64)
    compatible = (
        (latents["P_goal"] == latents["Q_goal"])
        & (latents["Q_goal"] == latents["Y_goal"])
    )
    if not np.all(compatible) or len(np.unique(raw_id)) != 16:
        raise RuntimeError("washout must contain exactly the 16 unanimous raw codewords")
    raw_counts = np.bincount(raw_id, minlength=RAW_CODEWORD_COUNT)
    nonzero = raw_counts[raw_counts > 0]
    if not np.array_equal(nonzero, np.full(16, 6 * rows_per_weight_unit)):
        raise RuntimeError("every washout raw codeword must contain six exact strata")
    if set(np.unique(stratum_id).tolist()) != set(range(WEIGHTED_STRATUM_COUNT)):
        raise RuntimeError("washout weighted-stratum IDs must be exactly 0..95")
    if not np.array_equal(
        np.bincount(stratum_id, minlength=WEIGHTED_STRATUM_COUNT),
        np.full(WEIGHTED_STRATUM_COUNT, rows_per_weight_unit),
    ):
        raise RuntimeError("every washout stratum must have the exact replica count")
    for raw in np.unique(raw_id):
        if not np.array_equal(np.unique(copy_id[raw_id == raw]), np.arange(6)):
            raise RuntimeError("washout copy IDs must enumerate six strata per codeword")
    for stratum in range(WEIGHTED_STRATUM_COUNT):
        selected = stratum_id == stratum
        if len(np.unique(raw_id[selected])) != 1 or len(np.unique(copy_id[selected])) != 1:
            raise RuntimeError("washout stratum identity must map to one raw/copy pair")
        if not np.array_equal(np.sort(replica_id[selected]), np.arange(rows_per_weight_unit)):
            raise RuntimeError("washout replicas must enumerate the registered range")
    target = np.asarray(batch.target, dtype=np.int8)
    if not all(np.array_equal(target, latents[f"{goal}_goal"]) for goal in GOALS):
        raise RuntimeError("washout target must equal concordant P=Q=Y")
    if not np.array_equal(np.asarray(batch.y), target):
        raise RuntimeError("washout semantic y and supervised target must coincide")
    if not np.array_equal(np.asarray(batch.reward), target.astype(np.float32)):
        raise RuntimeError("washout reward must equal target")
    negative = int(np.sum(target < 0))
    positive = int(np.sum(target > 0))
    if negative != expected_n // 2 or positive != expected_n // 2:
        raise RuntimeError("washout targets must be exactly balanced")
    accuracies = _accuracy_counts(batch, target)
    for name, item in accuracies.items():
        _require_fraction(int(item["correct"]), expected_n, 1, 1, f"washout {name}")
    return {
        "kind": "washout",
        "n_rows": len(batch),
        "raw_codeword_count": len(np.unique(raw_id)),
        "weighted_stratum_count": len(np.unique(stratum_id)),
        "strata_per_raw_codeword": 6,
        "rows_per_weight_unit": rows_per_weight_unit,
        "raw_row_counts": raw_counts.tolist(),
        "label_counts": {"-1": negative, "+1": positive},
        "accuracy": accuracies,
        "interface": interface,
        "semantic_ids": ids,
        "semantic_batch_digest": semantic_batch_digest(batch),
        "raw_label_count_digest": stable_state_digest(
            {"raw_id": raw_id, "target": target, "stratum_id": stratum_id}
        ),
        "fold_digest_sha256": FOLD_DIGEST_SHA256,
        "control_digest_sha256": CONTROL_DIGEST_SHA256,
    }


def audit_counterbalanced_factorial_panel(batch: SemanticBatch) -> dict[str, Any]:
    """Verify the ordered evaluation panel, including frozen folds and control."""

    if len(batch) != RAW_CODEWORD_COUNT:
        raise RuntimeError("E20 factorial panel must contain exactly 64 rows")
    if batch.metadata.get("dataset") != "counterbalanced_order_factorial":
        raise RuntimeError("batch is not registered as the E20 factorial panel")
    interface = _audit_interface(batch, require_all_raw=True)
    latents = _audit_common_latents(batch)
    ids = _audit_semantic_ids(batch)
    raw_id = latents["raw_codeword_id"].astype(np.int64, copy=False)
    if not np.array_equal(raw_id, np.arange(RAW_CODEWORD_COUNT)):
        raise RuntimeError("factorial panel must be ordered by canonical raw-codeword ID")
    if not np.array_equal(np.asarray(batch.target), latents["Y_goal"]):
        raise RuntimeError("factorial panel target must be Y")
    fold_control = audit_counterbalanced_fold_control()
    return {
        "kind": "factorial_panel",
        "n_rows": len(batch),
        "raw_codeword_count": len(np.unique(raw_id)),
        "candidate_tuple_count": len(np.unique(latents["candidate_tuple_id"])),
        "raw_codewords_per_tuple": [
            int(np.sum(latents["candidate_tuple_id"] == value)) for value in range(8)
        ],
        "fold_counts": [int(np.sum(latents["fold_id"] == value)) for value in range(8)],
        "interface": interface,
        "semantic_ids": ids,
        "semantic_batch_digest": semantic_batch_digest(batch),
        "fold_control": fold_control,
    }


@dataclass(frozen=True)
class CounterbalancedOrderData:
    """All three component datasets plus their common concordant washout."""

    components: Mapping[Goal, SemanticBatch]
    washout: SemanticBatch
    seed: int
    rows_per_weight_unit: int
    audits: Mapping[str, Any] = field(default_factory=dict)

    def component(self, goal: Goal | str) -> SemanticBatch:
        return self.components[_normalize_goal(goal)]


def _raw_conditional_counts(batches: Sequence[SemanticBatch]) -> IntArray:
    counts = np.zeros((RAW_CODEWORD_COUNT, 2), dtype=np.int64)
    for batch in batches:
        raw_id = np.asarray(batch.latents["raw_codeword_id"], dtype=np.int64)
        target = np.asarray(batch.target, dtype=np.int8)
        np.add.at(counts[:, 0], raw_id[target < 0], 1)
        np.add.at(counts[:, 1], raw_id[target > 0], 1)
    return counts


def _pooled_accuracy(batches: Sequence[SemanticBatch]) -> dict[str, dict[str, int | float]]:
    target = np.concatenate([np.asarray(batch.target, dtype=np.int8) for batch in batches])
    result: dict[str, dict[str, int | float]] = {}
    for name in (*GOALS, "majority"):
        latent_name = f"{name}_goal" if name in GOALS else "majority_goal"
        values = np.concatenate(
            [np.asarray(batch.latents[latent_name], dtype=np.int8) for batch in batches]
        )
        result[name] = _fraction(int(np.sum(values == target)), len(target))
    conditional = _raw_conditional_counts(batches)
    result["bayes_raw_input"] = _fraction(int(np.sum(np.max(conditional, axis=1))), len(target))
    return result


def audit_counterbalanced_bundle(bundle: CounterbalancedOrderData) -> dict[str, Any]:
    """Fail closed on cross-component pairing, pooled Bayes, and full-stream proofs."""

    if set(bundle.components) != set(GOALS):
        raise RuntimeError("counterbalanced bundle must contain exactly P, Q, and Y components")
    rows_per_weight_unit = _positive_integer(
        bundle.rows_per_weight_unit, "bundle.rows_per_weight_unit"
    )
    component_audits = {
        goal: audit_counterbalanced_component(
            bundle.components[goal],
            expected_goal=goal,
            rows_per_weight_unit=rows_per_weight_unit,
        )
        for goal in GOALS
    }
    washout_audit = audit_concordant_washout(
        bundle.washout, rows_per_weight_unit=rows_per_weight_unit
    )

    feature_names = {
        tuple(component_audits[goal]["interface"]["feature_names"]) for goal in GOALS
    }
    feature_names.add(tuple(washout_audit["interface"]["feature_names"]))
    if feature_names != {FEATURE_NAMES}:
        raise RuntimeError("components and washout do not share the exact feature interface")
    support_digests = {
        str(component_audits[goal]["interface"]["visible_support_digest"]) for goal in GOALS
    }
    if len(support_digests) != 1:
        raise RuntimeError("P/Q/Y components do not share identical visible raw support")

    datasets: dict[str, SemanticBatch] = {
        goal: bundle.components[goal] for goal in GOALS
    }
    datasets["washout"] = bundle.washout
    for id_name in ("sample_id", "state_id", "episode_id"):
        seen: set[int] = set()
        for name, batch in datasets.items():
            current = set(np.asarray(getattr(batch, id_name), dtype=np.int64).tolist())
            if seen & current:
                raise RuntimeError(f"E20 {id_name} values overlap at dataset {name}")
            seen.update(current)

    components = [bundle.components[goal] for goal in GOALS]
    pooled_target = np.concatenate([np.asarray(batch.target, dtype=np.int8) for batch in components])
    if int(np.sum(pooled_target > 0)) != len(pooled_target) // 2:
        raise RuntimeError("three-component pooled labels must be exactly balanced")
    pooled_accuracy = _pooled_accuracy(components)
    for name, item in pooled_accuracy.items():
        _require_fraction(int(item["correct"]), int(item["total"]), 2, 3, f"pooled {name}")

    conditional = _raw_conditional_counts(components)
    table = _canonical_table()
    unanimous = (table.goals["P"] == table.goals["Q"]) & (
        table.goals["Q"] == table.goals["Y"]
    )
    scale = rows_per_weight_unit
    for raw in range(RAW_CODEWORD_COUNT):
        values = sorted(conditional[raw].tolist())
        expected = [0, 6 * scale] if unanimous[raw] else [2 * scale, 2 * scale]
        if values != expected:
            raise RuntimeError(
                f"pooled conditional label counts are wrong for raw codeword {raw}: {values}"
            )

    full_batches = [*components, bundle.washout]
    full_accuracy = _pooled_accuracy(full_batches)
    for name, item in full_accuracy.items():
        _require_fraction(
            int(item["correct"]), int(item["total"]), 3, 4, f"pooled-plus-washout {name}"
        )
    full_conditional = _raw_conditional_counts(full_batches)

    report: dict[str, Any] = {
        "seed": int(bundle.seed),
        "rows_per_weight_unit": rows_per_weight_unit,
        "components": component_audits,
        "washout": washout_audit,
        "component_semantic_digests": {
            goal: component_audits[goal]["semantic_batch_digest"] for goal in GOALS
        },
        "washout_semantic_digest": washout_audit["semantic_batch_digest"],
        "feature_interface_equal": True,
        "component_visible_support_equal": True,
        "semantic_ids_pairwise_disjoint": True,
        "pooled": {
            "n_rows": len(pooled_target),
            "label_counts": {
                "-1": int(np.sum(pooled_target < 0)),
                "+1": int(np.sum(pooled_target > 0)),
            },
            "accuracy": pooled_accuracy,
            "raw_conditional_counts": conditional.tolist(),
            "raw_conditional_count_digest": _array_digest(
                conditional, person=b"e20rawlabels"
            ),
            "unanimous_six_to_zero": True,
            "nonunanimous_two_to_two": True,
        },
        "pooled_with_washout": {
            "n_rows": int(sum(len(batch) for batch in full_batches)),
            "accuracy": full_accuracy,
            "raw_conditional_counts": full_conditional.tolist(),
            "raw_conditional_count_digest": _array_digest(
                full_conditional, person=b"e20fullraw"
            ),
        },
        "fold_control": audit_counterbalanced_fold_control(),
    }
    report["audit_digest"] = stable_state_digest(report)
    return report


def make_counterbalanced_bundle(
    seed: int,
    *,
    rows_per_weight_unit: int = ROWS_PER_WEIGHT_UNIT,
    id_offset: int = DEFAULT_ID_OFFSET,
) -> CounterbalancedOrderData:
    """Construct all three disjoint-ID components and the common washout."""

    rows_per_weight_unit = _positive_integer(rows_per_weight_unit, "rows_per_weight_unit")
    id_offset = _nonnegative_integer(id_offset, "id_offset")
    n = WEIGHTED_STRATUM_COUNT * rows_per_weight_unit
    _check_id_range(id_offset, 4 * n)
    components = MappingProxyType(
        {
            goal: make_counterbalanced_component(
                goal,
                int(seed),
                rows_per_weight_unit=rows_per_weight_unit,
                id_offset=id_offset + index * n,
            )
            for index, goal in enumerate(GOALS)
        }
    )
    washout = make_concordant_washout(
        int(seed),
        rows_per_weight_unit=rows_per_weight_unit,
        id_offset=id_offset + len(GOALS) * n,
    )
    provisional = CounterbalancedOrderData(
        components=components,
        washout=washout,
        seed=int(seed),
        rows_per_weight_unit=rows_per_weight_unit,
    )
    audit = audit_counterbalanced_bundle(provisional)
    # Audit reports are intentionally JSON-ready because protocols persist them
    # verbatim for independent reconstruction by the gate and strict analyzer.
    return replace(provisional, audits=audit)


def _atomic_dimensions(
    *,
    rows_per_weight_unit: int,
    batches_per_presentation: int,
    examples_per_weight_unit_per_batch: int,
    batch_size: int,
    presentations: int,
) -> tuple[int, int, int, int]:
    rows = _positive_integer(rows_per_weight_unit, "rows_per_weight_unit")
    batches = _positive_integer(batches_per_presentation, "batches_per_presentation")
    examples = _positive_integer(
        examples_per_weight_unit_per_batch,
        "examples_per_weight_unit_per_batch",
    )
    size = _positive_integer(batch_size, "batch_size")
    repeats = _positive_integer(presentations, "presentations")
    if rows != batches * examples:
        raise ValueError(
            "rows_per_weight_unit must equal batches_per_presentation * "
            "examples_per_weight_unit_per_batch"
        )
    if size != WEIGHTED_STRATUM_COUNT * examples:
        raise ValueError(
            "batch_size must equal 96 weighted strata * examples_per_weight_unit_per_batch"
        )
    return batches, examples, size, repeats


def _make_atomic_stream(
    batch: SemanticBatch,
    *,
    component: str,
    seed: int,
    presentations: int,
    batches_per_presentation: int,
    examples_per_weight_unit_per_batch: int,
    batch_size: int,
) -> IntArray:
    rows_per_weight_unit = _positive_integer(
        int(batch.metadata.get("rows_per_weight_unit", 0)),
        "metadata.rows_per_weight_unit",
    )
    batches, examples, size, repeats = _atomic_dimensions(
        rows_per_weight_unit=rows_per_weight_unit,
        batches_per_presentation=batches_per_presentation,
        examples_per_weight_unit_per_batch=examples_per_weight_unit_per_batch,
        batch_size=batch_size,
        presentations=presentations,
    )
    if "weighted_stratum_id" not in batch.latents:
        raise ValueError("atomic E20 data must expose metadata-only weighted_stratum_id")
    stratum_id = np.asarray(batch.latents["weighted_stratum_id"], dtype=np.int64)
    if set(np.unique(stratum_id).tolist()) != set(range(WEIGHTED_STRATUM_COUNT)):
        raise ValueError("atomic E20 data must contain weighted strata 0..95")

    presentation = np.empty((batches, size), dtype=np.int64)
    batch_rows: list[list[int]] = [[] for _ in range(batches)]
    for stratum in range(WEIGHTED_STRATUM_COUNT):
        rows = np.flatnonzero(stratum_id == stratum).astype(np.int64, copy=False)
        if len(rows) != rows_per_weight_unit:
            raise ValueError("one E20 weighted stratum has the wrong stored row count")
        generator = np.random.default_rng(
            _stable_seed(seed, component, "stratum-row-order", stratum)
        )
        ordered = generator.permutation(rows)
        for batch_index in range(batches):
            start = batch_index * examples
            batch_rows[batch_index].extend(ordered[start : start + examples].tolist())
    for batch_index, assembled_rows in enumerate(batch_rows):
        if len(assembled_rows) != size or len(set(assembled_rows)) != size:
            raise RuntimeError("atomic E20 construction produced a short or duplicate-row batch")
        generator = np.random.default_rng(
            _stable_seed(seed, component, "within-batch-order", batch_index)
        )
        presentation[batch_index] = generator.permutation(
            np.asarray(assembled_rows, dtype=np.int64)
        )
    stream = np.tile(presentation, (repeats, 1))
    return cast(IntArray, _readonly(stream, np.int64))


def make_atomic_component_stream(
    batch: SemanticBatch,
    goal: Goal | str,
    seed: int,
    *,
    presentations: int = PRESENTATIONS,
    batches_per_presentation: int = BATCHES_PER_PRESENTATION,
    examples_per_weight_unit_per_batch: int = EXAMPLES_PER_WEIGHT_UNIT_PER_BATCH,
    batch_size: int = BATCH_SIZE,
) -> IntArray:
    """Build one fixed component presentation and repeat it exactly."""

    selected = _normalize_goal(goal)
    audit_counterbalanced_component(batch, expected_goal=selected)
    return _make_atomic_stream(
        batch,
        component=f"component_{selected}",
        seed=int(seed),
        presentations=presentations,
        batches_per_presentation=batches_per_presentation,
        examples_per_weight_unit_per_batch=examples_per_weight_unit_per_batch,
        batch_size=batch_size,
    )


def make_atomic_washout_stream(
    batch: SemanticBatch,
    seed: int,
    *,
    presentations: int = PRESENTATIONS,
    batches_per_presentation: int = BATCHES_PER_PRESENTATION,
    examples_per_weight_unit_per_batch: int = EXAMPLES_PER_WEIGHT_UNIT_PER_BATCH,
    batch_size: int = BATCH_SIZE,
) -> IntArray:
    """Build the fixed concordant washout stream and repeat it exactly."""

    audit_concordant_washout(batch)
    return _make_atomic_stream(
        batch,
        component="washout",
        seed=int(seed),
        presentations=presentations,
        batches_per_presentation=batches_per_presentation,
        examples_per_weight_unit_per_batch=examples_per_weight_unit_per_batch,
        batch_size=batch_size,
    )


def _stream_sample_ids(batch: SemanticBatch, indices: IntArray) -> IntArray:
    return np.asarray(batch.sample_id, dtype=np.int64)[np.asarray(indices, dtype=np.int64)]


def _stream_digest(batch: SemanticBatch, indices: IntArray, *, person: bytes) -> str:
    return _array_digest(_stream_sample_ids(batch, indices), person=person)


def _row_exposure_digest(indices: IntArray, n: int) -> str:
    counts = np.bincount(np.asarray(indices, dtype=np.int64).reshape(-1), minlength=n)
    return stable_state_digest(
        {
            "row_index": np.arange(n, dtype=np.int64),
            "presentation_count": counts.astype(np.int64, copy=False),
        }
    )


def _ordered_batch_digests(sample_id_batches: IntArray) -> tuple[str, ...]:
    return tuple(
        _array_digest(np.asarray(row, dtype=np.int64), person=b"e20batch")
        for row in np.asarray(sample_id_batches, dtype=np.int64)
    )


def _batch_multiset_digest(sample_id_streams: Sequence[IntArray], *, person: bytes) -> str:
    hashes: list[bytes] = []
    for stream in sample_id_streams:
        for row in np.asarray(stream, dtype=np.int64):
            hashes.append(hashlib.blake2b(row.tobytes(order="C"), digest_size=32).digest())
    hashes.sort()
    digest = hashlib.blake2b(digest_size=32, person=person)
    for value in hashes:
        digest.update(value)
    return digest.hexdigest()


@dataclass(frozen=True)
class AtomicGoalPlan:
    """Schedule-independent component streams plus one ordered schedule view."""

    schedule: OrderSchedule
    order: tuple[Goal, Goal, Goal]
    component_indices: Mapping[Goal, IntArray]
    washout_indices: IntArray
    batch_size: int
    batches_per_presentation: int
    examples_per_weight_unit_per_batch: int
    presentations: int
    component_digests: Mapping[Goal, str]
    washout_digest: str
    row_exposure_digests: Mapping[str, str]
    ordered_stream_digest: str
    component_multiset_digest: str
    atomic_batch_multiset_digest: str
    plan_digest: str

    @property
    def component_steps(self) -> int:
        return self.batches_per_presentation * self.presentations

    @property
    def washout_steps(self) -> int:
        return len(self.washout_indices)

    @property
    def total_steps(self) -> int:
        return sum(len(self.component_indices[goal]) for goal in self.order) + self.washout_steps

    @property
    def component_streams(self) -> Mapping[Goal, IntArray]:
        """Alias emphasizing that matrices encode complete reusable streams."""

        return self.component_indices

    def phase(self, name: Goal | str) -> IntArray:
        if str(name).lower() == "washout":
            return self.washout_indices
        return self.component_indices[_normalize_goal(str(name))]


def _ordered_sample_id_stream(bundle: CounterbalancedOrderData, plan: AtomicGoalPlan) -> IntArray:
    streams = [
        _stream_sample_ids(bundle.components[goal], plan.component_indices[goal])
        for goal in plan.order
    ]
    streams.append(_stream_sample_ids(bundle.washout, plan.washout_indices))
    return np.concatenate(streams, axis=0).astype(np.int64, copy=False)


def make_counterbalanced_atomic_plan(
    bundle: CounterbalancedOrderData,
    schedule: OrderSchedule | str,
    seed: int,
    *,
    presentations: int = PRESENTATIONS,
    batches_per_presentation: int = BATCHES_PER_PRESENTATION,
    examples_per_weight_unit_per_batch: int = EXAMPLES_PER_WEIGHT_UNIT_PER_BATCH,
    batch_size: int = BATCH_SIZE,
) -> AtomicGoalPlan:
    """Construct one of six orders from bit-identical reusable component streams."""

    schedule_name = cast(OrderSchedule, str(schedule))
    order = _schedule_order(schedule_name)
    audit_counterbalanced_bundle(bundle)
    _atomic_dimensions(
        rows_per_weight_unit=bundle.rows_per_weight_unit,
        batches_per_presentation=batches_per_presentation,
        examples_per_weight_unit_per_batch=examples_per_weight_unit_per_batch,
        batch_size=batch_size,
        presentations=presentations,
    )
    component_indices = MappingProxyType(
        {
            goal: make_atomic_component_stream(
                bundle.components[goal],
                goal,
                int(seed),
                presentations=presentations,
                batches_per_presentation=batches_per_presentation,
                examples_per_weight_unit_per_batch=examples_per_weight_unit_per_batch,
                batch_size=batch_size,
            )
            for goal in GOALS
        }
    )
    washout_indices = make_atomic_washout_stream(
        bundle.washout,
        int(seed),
        presentations=presentations,
        batches_per_presentation=batches_per_presentation,
        examples_per_weight_unit_per_batch=examples_per_weight_unit_per_batch,
        batch_size=batch_size,
    )
    component_digests = MappingProxyType(
        {
            goal: _stream_digest(
                bundle.components[goal], component_indices[goal], person=b"e20component"
            )
            for goal in GOALS
        }
    )
    washout_digest = _stream_digest(bundle.washout, washout_indices, person=b"e20washout")
    row_exposure_digests = MappingProxyType(
        {
            **{
                goal: _row_exposure_digest(component_indices[goal], len(bundle.components[goal]))
                for goal in GOALS
            },
            "washout": _row_exposure_digest(washout_indices, len(bundle.washout)),
        }
    )
    component_sample_streams = [
        _stream_sample_ids(bundle.components[goal], component_indices[goal]) for goal in GOALS
    ]
    washout_sample_stream = _stream_sample_ids(bundle.washout, washout_indices)
    component_multiset_digest = _batch_multiset_digest(
        component_sample_streams, person=b"e20compmulti"
    )
    atomic_batch_multiset_digest = _batch_multiset_digest(
        [*component_sample_streams, washout_sample_stream], person=b"e20allmulti"
    )

    provisional = AtomicGoalPlan(
        schedule=schedule_name,
        order=order,
        component_indices=component_indices,
        washout_indices=washout_indices,
        batch_size=int(batch_size),
        batches_per_presentation=int(batches_per_presentation),
        examples_per_weight_unit_per_batch=int(examples_per_weight_unit_per_batch),
        presentations=int(presentations),
        component_digests=component_digests,
        washout_digest=washout_digest,
        row_exposure_digests=row_exposure_digests,
        ordered_stream_digest="",
        component_multiset_digest=component_multiset_digest,
        atomic_batch_multiset_digest=atomic_batch_multiset_digest,
        plan_digest="",
    )
    ordered_stream_digest = _array_digest(
        _ordered_sample_id_stream(bundle, provisional), person=b"e20ordered"
    )
    plan_payload = {
        "schedule": schedule_name,
        "order": order,
        "component_digests": component_digests,
        "washout_digest": washout_digest,
        "row_exposure_digests": row_exposure_digests,
        "ordered_stream_digest": ordered_stream_digest,
        "component_multiset_digest": component_multiset_digest,
        "atomic_batch_multiset_digest": atomic_batch_multiset_digest,
        "batch_size": int(batch_size),
        "batches_per_presentation": int(batches_per_presentation),
        "examples_per_weight_unit_per_batch": int(examples_per_weight_unit_per_batch),
        "presentations": int(presentations),
    }
    plan = replace(
        provisional,
        ordered_stream_digest=ordered_stream_digest,
        plan_digest=stable_state_digest(plan_payload),
    )
    audit_counterbalanced_atomic_plan(bundle, plan)
    return plan


def _audit_atomic_stream(
    batch: SemanticBatch,
    indices: IntArray,
    *,
    name: str,
    presentations: int,
    batches_per_presentation: int,
    examples_per_weight_unit_per_batch: int,
    batch_size: int,
    expected_raw_codewords: int,
) -> dict[str, Any]:
    stream = np.asarray(indices, dtype=np.int64)
    expected_steps = presentations * batches_per_presentation
    if stream.shape != (expected_steps, batch_size):
        raise RuntimeError(
            f"{name} atomic stream must have shape {(expected_steps, batch_size)}, got {stream.shape}"
        )
    if np.any(stream < 0) or np.any(stream >= len(batch)):
        raise RuntimeError(f"{name} atomic stream contains an out-of-range row index")
    if any(len(np.unique(row)) != batch_size for row in stream):
        raise RuntimeError(f"{name} atomic minibatch repeats a stored row")
    target = np.asarray(batch.target, dtype=np.int8)
    if not np.all(np.sum(target[stream] > 0, axis=1) == batch_size // 2):
        raise RuntimeError(f"{name} atomic minibatches must be exactly target balanced")
    stratum_id = np.asarray(batch.latents["weighted_stratum_id"], dtype=np.int64)
    raw_id = np.asarray(batch.latents["raw_codeword_id"], dtype=np.int64)
    expected_strata = np.arange(WEIGHTED_STRATUM_COUNT, dtype=np.int64)
    raw_quota_values: set[int] = set()
    for batch_indices in stream:
        strata, stratum_counts = np.unique(stratum_id[batch_indices], return_counts=True)
        if not np.array_equal(strata, expected_strata) or not np.array_equal(
            stratum_counts,
            np.full(WEIGHTED_STRATUM_COUNT, examples_per_weight_unit_per_batch),
        ):
            raise RuntimeError(
                f"{name} minibatch must contain the exact quota from every weighted stratum"
            )
        raws, raw_counts = np.unique(raw_id[batch_indices], return_counts=True)
        if len(raws) != expected_raw_codewords:
            raise RuntimeError(f"{name} minibatch omitted a compatible raw codeword")
        raw_quota_values.update(int(value) for value in raw_counts)

    first = stream[:batches_per_presentation]
    for presentation in range(presentations):
        start = presentation * batches_per_presentation
        block = stream[start : start + batches_per_presentation]
        if not np.array_equal(block, first):
            raise RuntimeError(f"{name} must replay the exact same presentation stream")
        counts = np.bincount(block.reshape(-1), minlength=len(batch))
        if not np.array_equal(counts, np.ones(len(batch), dtype=np.int64)):
            raise RuntimeError(f"{name} presentation does not consume every stored row exactly once")
    exposure = np.bincount(stream.reshape(-1), minlength=len(batch))
    if not np.array_equal(exposure, np.full(len(batch), presentations, dtype=np.int64)):
        raise RuntimeError(f"{name} stream has incorrect per-row exposure counts")
    sample_ids = _stream_sample_ids(batch, stream)
    return {
        "shape": list(stream.shape),
        "batch_size": batch_size,
        "batches_per_presentation": batches_per_presentation,
        "presentations": presentations,
        "optimizer_steps": expected_steps,
        "sample_presentations": int(stream.size),
        "examples_per_weighted_stratum_per_batch": examples_per_weight_unit_per_batch,
        "compatible_raw_codewords_per_batch": expected_raw_codewords,
        "raw_quota_values_per_batch": sorted(raw_quota_values),
        "target_negative_per_batch": batch_size // 2,
        "target_positive_per_batch": batch_size // 2,
        "row_exposure_min": int(np.min(exposure)),
        "row_exposure_max": int(np.max(exposure)),
        "row_exposure_count_values": np.unique(exposure).tolist(),
        "row_exposure_digest": _row_exposure_digest(stream, len(batch)),
        "ordered_atomic_batch_digests": list(_ordered_batch_digests(sample_ids)),
        "stream_digest": _array_digest(sample_ids, person=b"e20component" if name in GOALS else b"e20washout"),
        "every_row_once_per_presentation": True,
        "exact_presentation_replay": True,
        "all_batches_full": True,
        "all_batches_target_balanced": True,
        "all_batches_stratum_balanced": True,
        "all_rows_unique_within_batch": True,
    }


def audit_counterbalanced_atomic_plan(
    bundle: CounterbalancedOrderData,
    plan: AtomicGoalPlan,
) -> dict[str, Any]:
    """Recompute every exposure, raw quota, label, order, and digest invariant."""

    audit_counterbalanced_bundle(bundle)
    expected_order = _schedule_order(plan.schedule)
    if tuple(plan.order) != expected_order:
        raise RuntimeError("atomic plan order does not match its schedule name")
    if set(plan.component_indices) != set(GOALS):
        raise RuntimeError("atomic plan must contain exactly the three component streams")
    batches, examples, size, repeats = _atomic_dimensions(
        rows_per_weight_unit=bundle.rows_per_weight_unit,
        batches_per_presentation=plan.batches_per_presentation,
        examples_per_weight_unit_per_batch=plan.examples_per_weight_unit_per_batch,
        batch_size=plan.batch_size,
        presentations=plan.presentations,
    )
    component_reports = {
        goal: _audit_atomic_stream(
            bundle.components[goal],
            plan.component_indices[goal],
            name=goal,
            presentations=repeats,
            batches_per_presentation=batches,
            examples_per_weight_unit_per_batch=examples,
            batch_size=size,
            expected_raw_codewords=RAW_CODEWORD_COUNT,
        )
        for goal in GOALS
    }
    washout_report = _audit_atomic_stream(
        bundle.washout,
        plan.washout_indices,
        name="washout",
        presentations=repeats,
        batches_per_presentation=batches,
        examples_per_weight_unit_per_batch=examples,
        batch_size=size,
        expected_raw_codewords=16,
    )
    expected_component_digests: dict[Goal, str] = {
        goal: str(component_reports[goal]["stream_digest"]) for goal in GOALS
    }
    if dict(plan.component_digests) != expected_component_digests:
        raise RuntimeError("atomic component stream digest mismatch")
    if plan.washout_digest != washout_report["stream_digest"]:
        raise RuntimeError("atomic washout stream digest mismatch")
    expected_exposures: dict[str, str] = {
        **{goal: str(component_reports[goal]["row_exposure_digest"]) for goal in GOALS},
        "washout": str(washout_report["row_exposure_digest"]),
    }
    if dict(plan.row_exposure_digests) != expected_exposures:
        raise RuntimeError("atomic row-exposure digest mismatch")

    ordered = _ordered_sample_id_stream(bundle, plan)
    ordered_digest = _array_digest(ordered, person=b"e20ordered")
    if plan.ordered_stream_digest != ordered_digest:
        raise RuntimeError("atomic ordered-stream digest mismatch")
    component_sample_streams = [
        _stream_sample_ids(bundle.components[goal], plan.component_indices[goal]) for goal in GOALS
    ]
    washout_sample_stream = _stream_sample_ids(bundle.washout, plan.washout_indices)
    component_multiset = _batch_multiset_digest(
        component_sample_streams, person=b"e20compmulti"
    )
    all_multiset = _batch_multiset_digest(
        [*component_sample_streams, washout_sample_stream], person=b"e20allmulti"
    )
    if plan.component_multiset_digest != component_multiset:
        raise RuntimeError("atomic three-component multiset digest mismatch")
    if plan.atomic_batch_multiset_digest != all_multiset:
        raise RuntimeError("atomic all-phase batch multiset digest mismatch")
    payload = {
        "schedule": plan.schedule,
        "order": plan.order,
        "component_digests": expected_component_digests,
        "washout_digest": washout_report["stream_digest"],
        "row_exposure_digests": expected_exposures,
        "ordered_stream_digest": ordered_digest,
        "component_multiset_digest": component_multiset,
        "atomic_batch_multiset_digest": all_multiset,
        "batch_size": size,
        "batches_per_presentation": batches,
        "examples_per_weight_unit_per_batch": examples,
        "presentations": repeats,
    }
    expected_plan_digest = stable_state_digest(payload)
    if plan.plan_digest != expected_plan_digest:
        raise RuntimeError("atomic plan metadata digest mismatch")
    return {
        "schedule": plan.schedule,
        "order": list(plan.order),
        "batch_size": size,
        "batches_per_presentation": batches,
        "presentations": repeats,
        "component_steps": batches * repeats,
        "washout_steps": batches * repeats,
        "total_steps": plan.total_steps,
        "total_sample_presentations": int(ordered.size),
        "components": component_reports,
        "washout": washout_report,
        "component_digests": expected_component_digests,
        "washout_digest": washout_report["stream_digest"],
        "row_exposure_digests": expected_exposures,
        "ordered_stream_digest": ordered_digest,
        "component_multiset_digest": component_multiset,
        "atomic_batch_multiset_digest": all_multiset,
        "plan_digest": expected_plan_digest,
        "schedule_order_verified": True,
        "component_streams_reusable": True,
    }


def make_all_counterbalanced_atomic_plans(
    bundle: CounterbalancedOrderData,
    seed: int,
    *,
    presentations: int = PRESENTATIONS,
    batches_per_presentation: int = BATCHES_PER_PRESENTATION,
    examples_per_weight_unit_per_batch: int = EXAMPLES_PER_WEIGHT_UNIT_PER_BATCH,
    batch_size: int = BATCH_SIZE,
) -> Mapping[OrderSchedule, AtomicGoalPlan]:
    """Construct and cross-audit the complete six-order plan family."""

    plans: Mapping[OrderSchedule, AtomicGoalPlan] = MappingProxyType(
        {
            schedule: make_counterbalanced_atomic_plan(
                bundle,
                schedule,
                int(seed),
                presentations=presentations,
                batches_per_presentation=batches_per_presentation,
                examples_per_weight_unit_per_batch=examples_per_weight_unit_per_batch,
                batch_size=batch_size,
            )
            for schedule in ORDER_SCHEDULES
        }
    )
    audit_schedule_multiset_equality(bundle, plans)
    return plans


def audit_schedule_multiset_equality(
    bundle: CounterbalancedOrderData,
    plans: Mapping[Any, AtomicGoalPlan] | Sequence[AtomicGoalPlan],
) -> dict[str, Any]:
    """Require six schedules to reuse streams and differ only in component order."""

    if isinstance(plans, Mapping):
        by_schedule = dict(plans)
        for key, plan in by_schedule.items():
            if key != plan.schedule:
                raise RuntimeError("atomic plan mapping key does not match plan.schedule")
    else:
        by_schedule = {plan.schedule: plan for plan in plans}
        if len(by_schedule) != len(plans):
            raise RuntimeError("atomic plan sequence contains a duplicate schedule")
    if set(by_schedule) != set(ORDER_SCHEDULES):
        raise RuntimeError("schedule family must contain each of the six permutations exactly once")
    audits = {
        schedule: audit_counterbalanced_atomic_plan(bundle, by_schedule[schedule])
        for schedule in ORDER_SCHEDULES
    }
    reference = by_schedule[ORDER_SCHEDULES[0]]
    for schedule in ORDER_SCHEDULES[1:]:
        plan = by_schedule[schedule]
        for goal in GOALS:
            if not np.array_equal(reference.component_indices[goal], plan.component_indices[goal]):
                raise RuntimeError(f"component {goal} stream differs across schedule {schedule}")
        if not np.array_equal(reference.washout_indices, plan.washout_indices):
            raise RuntimeError(f"washout stream differs across schedule {schedule}")
    if len({plan.component_multiset_digest for plan in by_schedule.values()}) != 1:
        raise RuntimeError("three-component atomic multiset differs across schedules")
    if len({plan.atomic_batch_multiset_digest for plan in by_schedule.values()}) != 1:
        raise RuntimeError("all-phase atomic batch multiset differs across schedules")
    ordered_digests = {plan.ordered_stream_digest for plan in by_schedule.values()}
    if len(ordered_digests) != len(ORDER_SCHEDULES):
        raise RuntimeError("the six ordered-stream digests must all be distinct")
    for first_goal in GOALS:
        matching = [plan for plan in by_schedule.values() if plan.order[0] == first_goal]
        if len(matching) != 2 or not np.array_equal(
            matching[0].component_indices[first_goal], matching[1].component_indices[first_goal]
        ):
            raise RuntimeError(f"schedules sharing first goal {first_goal} lack a common prefix stream")
    report = {
        "schedule_count": len(by_schedule),
        "schedules": list(ORDER_SCHEDULES),
        "component_digests": dict(reference.component_digests),
        "washout_digest": reference.washout_digest,
        "row_exposure_digests": dict(reference.row_exposure_digests),
        "component_multiset_digest": reference.component_multiset_digest,
        "atomic_batch_multiset_digest": reference.atomic_batch_multiset_digest,
        "ordered_stream_digests": {
            schedule: by_schedule[schedule].ordered_stream_digest
            for schedule in ORDER_SCHEDULES
        },
        "plan_digests": {
            schedule: audits[schedule]["plan_digest"] for schedule in ORDER_SCHEDULES
        },
        "component_streams_bit_identical": True,
        "washout_stream_bit_identical": True,
        "order_invariant_multiset_equal": True,
        "ordered_streams_all_distinct": True,
        "same_first_goal_prefixes_identical": True,
    }
    report["audit_digest"] = stable_state_digest(report)
    return report


__all__ = [
    "BATCHES_PER_PRESENTATION",
    "BATCH_SIZE",
    "CONTROL_BIT_STRING",
    "CONTROL_DIGEST_SHA256",
    "EXAMPLES_PER_WEIGHT_UNIT_PER_BATCH",
    "FEATURE_NAMES",
    "FOLD_DIGEST_SHA256",
    "GOALS",
    "ORDER_SCHEDULES",
    "PRESENTATIONS",
    "RAW_CODEWORD_COUNT",
    "ROWS_PER_WEIGHT_UNIT",
    "AtomicGoalPlan",
    "CounterbalancedOrderData",
    "Goal",
    "OrderSchedule",
    "audit_concordant_washout",
    "audit_counterbalanced_atomic_plan",
    "audit_counterbalanced_bundle",
    "audit_counterbalanced_component",
    "audit_counterbalanced_factorial_panel",
    "audit_counterbalanced_fold_control",
    "audit_schedule_multiset_equality",
    "counterbalanced_fold_ids",
    "counterbalanced_truth_table_control",
    "make_all_counterbalanced_atomic_plans",
    "make_atomic_component_stream",
    "make_atomic_washout_stream",
    "make_concordant_washout",
    "make_counterbalanced_atomic_plan",
    "make_counterbalanced_bundle",
    "make_counterbalanced_component",
    "make_counterbalanced_factorial_panel",
]
