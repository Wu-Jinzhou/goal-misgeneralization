"""Controlled datasets for three-way goal competition experiments.

The completed Forkworld experiments compare one direct proxy with one exact
interaction code.  This module provides a self-contained extension with three
candidate rules:

``P``
    A directly observed proxy with exact finite-sample accuracy ``q_p``.
``Q``
    A second proxy with exact accuracy ``q_q``, exposed only through a
    degree-``k_q`` Rademacher interaction code.
``Y_code``
    An exact degree-``k_y`` interaction code for the intended target ``Y``.

The generators keep feature width fixed across degree sweeps, construct exact
proxy-error marginals, and control whether the two proxy error sets are nested
or have the overlap expected under independence.  Dedicated diagnostic panels
make all informative disagreement patterns observable without conflating them
with the training distribution.

This module intentionally has no protocol or configuration dependencies.  It
can be used by future runners without changing the existing H1--H9 pipeline.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .data import SemanticBatch, balanced_signs, degree_k_code, flip_channels
from .metrics import binary_predictions

SignArray = NDArray[np.int8]
BoolArray = NDArray[np.bool_]
OverlapMode = Literal["independent", "nested"]
DiagnosticPanel = Literal["both_wrong", "p_wrong", "q_wrong"]

DIAGNOSTIC_PANELS: tuple[DiagnosticPanel, ...] = (
    "both_wrong",
    "p_wrong",
    "q_wrong",
)


def _stable_seed(seed: int, *parts: object) -> int:
    """Return a process-independent seed for matched experimental streams."""

    digest = hashlib.blake2b(digest_size=8, person=b"forkcomp")
    for part in (seed, *parts):
        payload = str(part).encode("utf-8")
        digest.update(len(payload).to_bytes(4, "little"))
        digest.update(payload)
    return int.from_bytes(digest.digest(), "little")


def _positive_even(value: int, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or int(value) != value or int(value) <= 0:
        raise ValueError(f"{name} must be a positive integer")
    result = int(value)
    if result % 2:
        raise ValueError(f"{name} must be even so Y is exactly balanced")
    return result


def _positive_degree(value: int, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or int(value) != value or int(value) < 1:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def _error_count(n: int, accuracy: float, name: str) -> int:
    accuracy = float(accuracy)
    if not math.isfinite(accuracy) or not 0.0 <= accuracy <= 1.0:
        raise ValueError(f"{name} must lie in [0, 1]")
    raw = n * (1.0 - accuracy)
    count = round(raw)
    if not math.isclose(raw, count, rel_tol=0.0, abs_tol=1e-10):
        raise ValueError(
            f"{name}={accuracy:g} is not exactly realizable with n={n}; "
            f"n*(1-{name}) must be an integer"
        )
    return count


def _ranked_stratified_choice(
    available: BoolArray,
    y: SignArray,
    count: int,
    seed: int,
) -> BoolArray:
    """Choose exactly ``count`` positions, as balanced over Y as feasible."""

    if count < 0 or count > int(np.sum(available)):
        raise ValueError("requested selection exceeds the available positions")
    chosen = np.zeros(len(y), dtype=bool)
    if count == 0:
        return chosen

    negative = np.flatnonzero(available & (y == -1))
    positive = np.flatnonzero(available & (y == 1))
    total = len(negative) + len(positive)
    lower_negative = max(0, count - len(positive))
    upper_negative = min(count, len(negative))
    proportional = round(count * len(negative) / total)
    negative_count = min(max(proportional, lower_negative), upper_negative)
    positive_count = count - negative_count

    generator = np.random.default_rng(seed)
    negative_scores = generator.random(len(negative))
    positive_scores = generator.random(len(positive))
    if negative_count:
        chosen[negative[np.argsort(negative_scores)[:negative_count]]] = True
    if positive_count:
        chosen[positive[np.argsort(positive_scores)[:positive_count]]] = True
    return chosen


def _joint_error_masks(
    y: SignArray,
    errors_p: int,
    errors_q: int,
    seed: int,
    mode: OverlapMode,
    *,
    q_only_error_count: int | None = None,
) -> tuple[BoolArray, BoolArray, dict[str, Any]]:
    """Construct exact proxy-error marginals with a controlled overlap.

    ``independent`` fixes the overlap to the nearest feasible integer to
    ``errors_p * errors_q / n``.  ``nested`` fixes it to the largest feasible
    overlap, so the smaller error set is a subset of the larger one.

    ``q_only_error_count`` is the support-completion intervention used after
    the nested panel.  It fixes the number of rows on which only Q is wrong
    while preserving both error marginals.  The intervention deliberately uses
    the nested arm's rankings, so count zero is array-identical to ``nested``
    and increasing counts change a paired, nested set of row identities.
    """

    if mode not in {"independent", "nested"}:
        raise ValueError("overlap mode must be 'independent' or 'nested'")
    n = len(y)
    minimum = max(0, errors_p + errors_q - n)
    maximum = min(errors_p, errors_q)
    expected = errors_p * errors_q / n
    overlap_mode: str
    if q_only_error_count is None:
        overlap = (
            min(max(round(expected), minimum), maximum)
            if mode == "independent"
            else maximum
        )
        overlap_mode = mode
    else:
        if mode != "nested":
            raise ValueError("q_only_error_count requires the nested reference mode")
        if (
            isinstance(q_only_error_count, (bool, np.bool_))
            or int(q_only_error_count) != q_only_error_count
            or int(q_only_error_count) < 0
        ):
            raise ValueError("q_only_error_count must be a non-negative integer")
        q_only_error_count = int(q_only_error_count)
        overlap = errors_q - q_only_error_count
        if not minimum <= overlap <= maximum:
            feasible_low = errors_q - maximum
            feasible_high = errors_q - minimum
            raise ValueError(
                "q_only_error_count is incompatible with the error marginals; "
                f"expected a value in [{feasible_low}, {feasible_high}]"
            )
        overlap_mode = "support_completion"

    all_positions = np.ones(n, dtype=bool)
    p_error = _ranked_stratified_choice(
        all_positions, y, errors_p, _stable_seed(seed, "P-errors")
    )
    shared = _ranked_stratified_choice(
        p_error, y, overlap, _stable_seed(seed, mode, "shared-errors")
    )
    q_only_count = errors_q - overlap
    q_only = _ranked_stratified_choice(
        ~p_error, y, q_only_count, _stable_seed(seed, mode, "Q-only-errors")
    )
    q_error = shared | q_only

    if int(np.sum(p_error)) != errors_p or int(np.sum(q_error)) != errors_q:
        raise RuntimeError("internal error constructing exact proxy marginals")
    realized_overlap = int(np.sum(p_error & q_error))
    if realized_overlap != overlap:
        raise RuntimeError("internal error constructing proxy-error overlap")

    p_rate = errors_p / n
    q_rate = errors_q / n
    denominator = math.sqrt(p_rate * (1.0 - p_rate) * q_rate * (1.0 - q_rate))
    error_phi = None
    if denominator > 0.0:
        error_phi = ((realized_overlap / n) - p_rate * q_rate) / denominator

    diagnostics: dict[str, Any] = {
        "overlap_mode": overlap_mode,
        "reference_overlap_mode": mode if q_only_error_count is not None else None,
        "requested_q_only_error_count": q_only_error_count,
        "p_error_count": errors_p,
        "q_error_count": errors_q,
        "both_error_count": realized_overlap,
        "p_only_error_count": errors_p - realized_overlap,
        "q_only_error_count": errors_q - realized_overlap,
        "neither_error_count": n - errors_p - errors_q + realized_overlap,
        "both_error_rate": realized_overlap / n,
        "independence_expected_count": expected,
        "overlap_excess_count": realized_overlap - expected,
        "error_phi": error_phi,
    }
    return p_error, q_error, diagnostics


def _state_features(ids: NDArray[np.int64], width: int, seed: int) -> NDArray[np.float32] | None:
    if isinstance(width, (bool, np.bool_)) or int(width) != width or int(width) < 0:
        raise ValueError("state_dim must be a non-negative integer")
    width = int(width)
    if width == 0:
        return None
    columns: list[NDArray[np.int8]] = []
    for column in range(width):
        generator = np.random.default_rng(_stable_seed(seed, "state", column))
        values = generator.choice(np.asarray([-1, 1], dtype=np.int8), size=len(ids))
        columns.append(values)
    return np.column_stack(columns).astype(np.float32, copy=False)


def _q_channels(
    q_goal: SignArray,
    k_q: int,
    max_k_q: int,
    seed: int,
) -> dict[str, SignArray]:
    code = degree_k_code(q_goal, k_q, seed=_stable_seed(seed, "Q-code"))
    channels: dict[str, SignArray] = {}
    for index in range(1, max_k_q + 1):
        if index <= k_q:
            values = code[:, index - 1]
        else:
            generator = np.random.default_rng(_stable_seed(seed, "Q-padding", index))
            values = generator.choice(
                np.asarray([-1, 1], dtype=np.int8), size=len(q_goal)
            )
        channels[f"Q_{index}"] = np.asarray(values, dtype=np.int8)
    return channels


def _encoded_batch(
    y: SignArray,
    p_goal: SignArray,
    q_goal: SignArray,
    *,
    p_error: BoolArray,
    q_error: BoolArray,
    k_q: int,
    k_y: int,
    max_k_q: int,
    max_k_y: int,
    seed: int,
    split: str,
    id_offset: int,
    state_dim: int,
    metadata: Mapping[str, Any],
) -> SemanticBatch:
    n = len(y)
    y_code = degree_k_code(y, k_y, seed=_stable_seed(seed, split, "Y-code"))
    channels: dict[str, ArrayLike] = {
        "P": p_goal,
        "P_present": np.ones(n, dtype=np.int8),
    }
    channels.update({f"R_{index + 1}": y_code[:, index] for index in range(k_y)})
    channels["Q_present"] = np.ones(n, dtype=np.int8)
    channels.update(_q_channels(q_goal, k_q, max_k_q, _stable_seed(seed, split)))

    ids = np.arange(id_offset, id_offset + n, dtype=np.int64)
    batch_metadata = {
        "dataset": "competing_goals",
        "split": split,
        "seed": int(seed),
        "active_k": k_y,
        "max_k": max_k_y,
        "r_active_mask": tuple(index < k_y for index in range(max_k_y)),
        "padding_seed": _stable_seed(seed, split, "Y-padding"),
        "enforce_exact_code": True,
        "k_q": k_q,
        "max_k_q": max_k_q,
        "q_active_mask": tuple(index < k_q for index in range(max_k_q)),
        "k_y": k_y,
        "max_k_y": max_k_y,
        "state_dim": int(state_dim),
        **dict(metadata),
    }
    return SemanticBatch(
        y=y,
        target=y,
        reward=y.astype(np.float32),
        channels=channels,
        latents={
            "P_goal": p_goal,
            "Q_goal": q_goal,
            "P_error": p_error,
            "Q_error": q_error,
            "both_wrong": p_error & q_error,
            "p_wrong": p_error & ~q_error,
            "q_wrong": ~p_error & q_error,
        },
        state=_state_features(ids, state_dim, _stable_seed(seed, split)),
        sample_id=ids,
        state_id=ids,
        episode_id=ids,
        step_id=np.zeros(n, dtype=np.int64),
        metadata=batch_metadata,
    )


def make_competing_dataset(
    n: int,
    q_p: float,
    q_q: float,
    k_q: int,
    k_y: int,
    seed: int = 0,
    *,
    max_k_q: int | None = None,
    max_k_y: int | None = None,
    overlap: OverlapMode = "independent",
    q_only_error_count: int | None = None,
    state_dim: int = 0,
    split: str = "train",
    id_offset: int = 0,
) -> SemanticBatch:
    """Create a matched train or IID batch for three competing rules.

    Changing ``q_p``, ``q_q``, ``overlap``, or ``q_only_error_count`` does not
    redraw ``Y``, its exact interaction code, state features, IDs, or the random
    prefix of Q's code.  This makes those settings interpretable as matched
    interventions.
    """

    n = _positive_even(n, "n")
    k_q = _positive_degree(k_q, "k_q")
    k_y = _positive_degree(k_y, "k_y")
    max_k_q = k_q if max_k_q is None else _positive_degree(max_k_q, "max_k_q")
    max_k_y = k_y if max_k_y is None else _positive_degree(max_k_y, "max_k_y")
    if max_k_q < k_q:
        raise ValueError("max_k_q must be at least k_q")
    if max_k_y < k_y:
        raise ValueError("max_k_y must be at least k_y")

    errors_p = _error_count(n, q_p, "q_p")
    errors_q = _error_count(n, q_q, "q_q")
    stream_seed = _stable_seed(seed, split)
    y = balanced_signs(n, _stable_seed(stream_seed, "Y"))
    p_error, q_error, overlap_diagnostics = _joint_error_masks(
        y,
        errors_p,
        errors_q,
        stream_seed,
        overlap,
        q_only_error_count=q_only_error_count,
    )
    p_goal = np.where(p_error, -y, y).astype(np.int8)
    q_goal = np.where(q_error, -y, y).astype(np.int8)
    return _encoded_batch(
        y,
        p_goal,
        q_goal,
        p_error=p_error,
        q_error=q_error,
        k_q=k_q,
        k_y=k_y,
        max_k_q=max_k_q,
        max_k_y=max_k_y,
        seed=seed,
        split=split,
        id_offset=id_offset,
        state_dim=state_dim,
        metadata={
            "condition": "iid",
            "requested_q_p": float(q_p),
            "requested_q_q": float(q_q),
            "realized_q_p": float(np.mean(p_goal == y)),
            "realized_q_q": float(np.mean(q_goal == y)),
            **overlap_diagnostics,
        },
    )


def _factorial_size(
    n: int | None,
    repeats: int | None,
    raw_codeword_count: int,
) -> tuple[int, int]:
    """Resolve an exhaustive factorial batch's total and repeat count."""

    if n is None and repeats is None:
        raise ValueError("provide n, repeats, or both")

    resolved_n: int | None = None
    if n is not None:
        if isinstance(n, (bool, np.bool_)) or int(n) != n or int(n) <= 0:
            raise ValueError("n must be a positive integer")
        resolved_n = int(n)
        if resolved_n % raw_codeword_count:
            raise ValueError(
                f"n={resolved_n} must be a multiple of the {raw_codeword_count} "
                "active raw codewords"
            )

    resolved_repeats: int | None = None
    if repeats is not None:
        if (
            isinstance(repeats, (bool, np.bool_))
            or int(repeats) != repeats
            or int(repeats) <= 0
        ):
            raise ValueError("repeats must be a positive integer")
        resolved_repeats = int(repeats)

    if resolved_n is None:
        assert resolved_repeats is not None
        resolved_n = raw_codeword_count * resolved_repeats
    elif resolved_repeats is None:
        resolved_repeats = resolved_n // raw_codeword_count
    elif resolved_n != raw_codeword_count * resolved_repeats:
        raise ValueError(
            "n and repeats are inconsistent: "
            f"n={resolved_n}, but {raw_codeword_count} codewords * "
            f"{resolved_repeats} repeats = {raw_codeword_count * resolved_repeats}"
        )
    return resolved_n, resolved_repeats


def _truth_table_control(
    raw_codeword_id: NDArray[np.int64],
    candidate_tuple_id: NDArray[np.int64],
    *,
    control_seed: int,
) -> tuple[SignArray, str]:
    """Construct a seeded control truth table orthogonal to P, Q, and Y.

    Ordinarily every candidate tuple has at least two raw realizations.  We
    independently assign half of each tuple's realizations to either sign,
    making the control exactly independent of the complete ``(P, Q, Y)``
    tuple, not merely pairwise-orthogonal to its coordinates.  When all three
    candidates are represented by one bit, each tuple has only one realization;
    a seeded Walsh interaction supplies the strongest feasible exact control.
    """

    result = np.empty(len(raw_codeword_id), dtype=np.int8)
    counts = np.bincount(candidate_tuple_id, minlength=8)
    if np.all(counts % 2 == 0):
        for candidate_id in range(8):
            positions = np.flatnonzero(candidate_tuple_id == candidate_id)
            generator = np.random.default_rng(
                _stable_seed(control_seed, "truth-table-control", candidate_id)
            )
            order = generator.permutation(len(positions))
            result[positions[order[: len(positions) // 2]]] = -1
            result[positions[order[len(positions) // 2 :]]] = 1
        return result, "balanced_within_candidate_tuple"

    # The only possible odd cell size is one: k_q=k_y=1 gives exactly the
    # eight (P,Q,Y) tuples.  Any interaction of at least two candidate signs is
    # balanced and orthogonal to each individual sign.  Choose one by seed so
    # the control remains a deterministic truth table rather than a fourth
    # privileged semantic label.
    if not np.all(counts == 1):  # pragma: no cover - powers of two make this unreachable
        raise RuntimeError("internal error constructing factorial control cells")
    interaction = int(
        _stable_seed(control_seed, "truth-table-control", "minimal") % 4
    )
    positive_p = np.where((candidate_tuple_id & 1) != 0, 1, -1).astype(np.int8)
    positive_q = np.where((candidate_tuple_id & 2) != 0, 1, -1).astype(np.int8)
    positive_y = np.where((candidate_tuple_id & 4) != 0, 1, -1).astype(np.int8)
    controls = (
        positive_p * positive_q,
        positive_p * positive_y,
        positive_q * positive_y,
        positive_p * positive_q * positive_y,
    )
    return np.asarray(controls[interaction], dtype=np.int8), "minimal_walsh_interaction"


def make_competing_factorial_dataset(
    n: int | None = None,
    *,
    repeats: int | None = None,
    k_q: int,
    k_y: int,
    seed: int = 0,
    control_seed: int = 0,
    max_k_q: int | None = None,
    max_k_y: int | None = None,
    state_dim: int = 0,
    split: str = "train",
    id_offset: int = 0,
) -> SemanticBatch:
    """Exhaustively enumerate the model-visible competing-goal truth table.

    Every sign assignment to ``P``, the ``k_q`` active Q-code channels, and
    the ``k_y`` active Y-code channels occurs exactly ``repeats`` times.  Thus
    decoded P, Q, and Y are jointly uniform, while flipping any one active
    channel always lands on another observed raw codeword.  Canonical
    ``raw_codeword_id`` values and the control truth table do not depend on row
    order or split; ``seed`` and ``split`` affect only row order and nuisance
    streams.  Callers should use disjoint ``id_offset`` ranges across splits.

    Supply either ``n`` (which must be a multiple of the raw truth-table size),
    ``repeats``, or both consistently.
    """

    k_q = _positive_degree(k_q, "k_q")
    k_y = _positive_degree(k_y, "k_y")
    max_k_q = k_q if max_k_q is None else _positive_degree(max_k_q, "max_k_q")
    max_k_y = k_y if max_k_y is None else _positive_degree(max_k_y, "max_k_y")
    if max_k_q < k_q:
        raise ValueError("max_k_q must be at least k_q")
    if max_k_y < k_y:
        raise ValueError("max_k_y must be at least k_y")
    if isinstance(state_dim, (bool, np.bool_)) or int(state_dim) != state_dim or int(state_dim) < 0:
        raise ValueError("state_dim must be a non-negative integer")
    state_dim = int(state_dim)
    if not isinstance(split, str) or not split:
        raise ValueError("split must be a non-empty string")
    if (
        isinstance(id_offset, (bool, np.bool_))
        or int(id_offset) != id_offset
        or int(id_offset) < 0
    ):
        raise ValueError("id_offset must be a non-negative integer")
    id_offset = int(id_offset)

    raw_width = 1 + k_q + k_y
    if raw_width >= 63:
        raise ValueError("the exhaustive raw truth table is too wide for int64 IDs")
    raw_codeword_count = 1 << raw_width
    n, repeats = _factorial_size(n, repeats, raw_codeword_count)
    if id_offset > np.iinfo(np.int64).max - n:
        raise ValueError("id_offset + n exceeds the int64 ID range")

    canonical_id = np.arange(raw_codeword_count, dtype=np.int64)
    bit_positions = np.arange(raw_width, dtype=np.uint64)
    bits = (
        (canonical_id.astype(np.uint64)[:, None] >> bit_positions[None, :])
        & np.uint64(1)
    )
    raw_signs = (2 * bits.astype(np.int8) - 1).astype(np.int8, copy=False)
    p_canonical = raw_signs[:, 0]
    q_bits_canonical = raw_signs[:, 1 : 1 + k_q]
    y_bits_canonical = raw_signs[:, 1 + k_q :]
    q_canonical = np.prod(q_bits_canonical, axis=1, dtype=np.int8)
    y_canonical = np.prod(y_bits_canonical, axis=1, dtype=np.int8)
    candidate_id_canonical = (
        (p_canonical > 0).astype(np.int64)
        | ((q_canonical > 0).astype(np.int64) << 1)
        | ((y_canonical > 0).astype(np.int64) << 2)
    )
    control_canonical, control_construction = _truth_table_control(
        canonical_id,
        candidate_id_canonical,
        control_seed=int(control_seed),
    )

    raw_codeword_id = np.repeat(canonical_id, repeats)
    replicate_id = np.tile(np.arange(repeats, dtype=np.int64), raw_codeword_count)
    row_signs = np.repeat(raw_signs, repeats, axis=0)
    candidate_tuple_id = np.repeat(candidate_id_canonical, repeats)
    truth_table_control = np.repeat(control_canonical, repeats)
    generator = np.random.default_rng(_stable_seed(seed, split, "factorial-row-order"))
    order = generator.permutation(n)
    raw_codeword_id = raw_codeword_id[order]
    replicate_id = replicate_id[order]
    row_signs = row_signs[order]
    candidate_tuple_id = candidate_tuple_id[order]
    truth_table_control = truth_table_control[order]

    p_goal = np.asarray(row_signs[:, 0], dtype=np.int8)
    q_code = np.asarray(row_signs[:, 1 : 1 + k_q], dtype=np.int8)
    y_code = np.asarray(row_signs[:, 1 + k_q :], dtype=np.int8)
    q_goal = np.prod(q_code, axis=1, dtype=np.int8)
    y = np.prod(y_code, axis=1, dtype=np.int8)
    p_error = p_goal != y
    q_error = q_goal != y

    channels: dict[str, ArrayLike] = {
        "P": p_goal,
        "P_present": np.ones(n, dtype=np.int8),
    }
    channels.update({f"R_{index + 1}": y_code[:, index] for index in range(k_y)})
    channels["Q_present"] = np.ones(n, dtype=np.int8)
    channels.update({f"Q_{index + 1}": q_code[:, index] for index in range(k_q)})
    for index in range(k_q + 1, max_k_q + 1):
        padding_generator = np.random.default_rng(
            _stable_seed(seed, split, "factorial-Q-padding", index)
        )
        channels[f"Q_{index}"] = padding_generator.choice(
            np.asarray([-1, 1], dtype=np.int8), size=n
        )

    ids = np.arange(id_offset, id_offset + n, dtype=np.int64)
    raw_channel_order = (
        "P",
        *(f"Q_{index}" for index in range(1, k_q + 1)),
        *(f"R_{index}" for index in range(1, k_y + 1)),
    )
    metadata = {
        "dataset": "competing_goals_factorial",
        "condition": "factorial",
        "split": split,
        "seed": int(seed),
        "control_seed": int(control_seed),
        "control_construction": control_construction,
        "raw_channel_order": raw_channel_order,
        "raw_codeword_count": raw_codeword_count,
        "repeats_per_codeword": repeats,
        "candidate_tuple_count": 8,
        "active_k": k_y,
        "max_k": max_k_y,
        "r_active_mask": tuple(index < k_y for index in range(max_k_y)),
        "padding_seed": _stable_seed(seed, split, "factorial-Y-padding"),
        "enforce_exact_code": True,
        "k_q": k_q,
        "max_k_q": max_k_q,
        "q_active_mask": tuple(index < k_q for index in range(max_k_q)),
        "k_y": k_y,
        "max_k_y": max_k_y,
        "state_dim": state_dim,
        "requested_q_p": 0.5,
        "requested_q_q": 0.5,
        "realized_q_p": float(np.mean(p_goal == y)),
        "realized_q_q": float(np.mean(q_goal == y)),
        "overlap_mode": "full_factorial",
        "p_error_count": int(np.sum(p_error)),
        "q_error_count": int(np.sum(q_error)),
        "both_error_count": int(np.sum(p_error & q_error)),
    }
    return SemanticBatch(
        y=y,
        target=y,
        reward=y.astype(np.float32),
        channels=channels,
        latents={
            "P_goal": p_goal,
            "Q_goal": q_goal,
            "P_error": p_error,
            "Q_error": q_error,
            "both_wrong": p_error & q_error,
            "p_wrong": p_error & ~q_error,
            "q_wrong": ~p_error & q_error,
            "raw_codeword_id": raw_codeword_id,
            "candidate_tuple_id": candidate_tuple_id,
            "replicate_id": replicate_id,
            "truth_table_control": truth_table_control,
        },
        state=_state_features(ids, state_dim, _stable_seed(seed, split, "factorial")),
        sample_id=ids,
        state_id=ids,
        episode_id=ids,
        step_id=np.zeros(n, dtype=np.int64),
        metadata=metadata,
    )


def make_competing_diagnostic(
    n: int,
    panel: DiagnosticPanel,
    k_q: int,
    k_y: int,
    seed: int = 0,
    *,
    max_k_q: int | None = None,
    max_k_y: int | None = None,
    state_dim: int = 0,
    id_offset: int = 1_000_000_000,
) -> SemanticBatch:
    """Create one balanced panel in which P, Q, and Y have known relations."""

    n = _positive_even(n, "n")
    if panel not in DIAGNOSTIC_PANELS:
        raise ValueError(f"unknown diagnostic panel: {panel!r}")
    k_q = _positive_degree(k_q, "k_q")
    k_y = _positive_degree(k_y, "k_y")
    max_k_q = k_q if max_k_q is None else _positive_degree(max_k_q, "max_k_q")
    max_k_y = k_y if max_k_y is None else _positive_degree(max_k_y, "max_k_y")
    if max_k_q < k_q or max_k_y < k_y:
        raise ValueError("maximum code degrees must cover their active degrees")

    y = balanced_signs(n, _stable_seed(seed, panel, "Y"))
    if panel == "both_wrong":
        p_goal, q_goal = -y, -y
    elif panel == "p_wrong":
        p_goal, q_goal = -y, y.copy()
    else:
        p_goal, q_goal = y.copy(), -y
    p_goal = np.asarray(p_goal, dtype=np.int8)
    q_goal = np.asarray(q_goal, dtype=np.int8)
    p_error = p_goal != y
    q_error = q_goal != y
    return _encoded_batch(
        y,
        p_goal,
        q_goal,
        p_error=p_error,
        q_error=q_error,
        k_q=k_q,
        k_y=k_y,
        max_k_q=max_k_q,
        max_k_y=max_k_y,
        seed=seed,
        split=f"diagnostic_{panel}",
        id_offset=id_offset,
        state_dim=state_dim,
        metadata={
            "condition": panel,
            "requested_q_p": float(np.mean(p_goal == y)),
            "requested_q_q": float(np.mean(q_goal == y)),
            "realized_q_p": float(np.mean(p_goal == y)),
            "realized_q_q": float(np.mean(q_goal == y)),
            "overlap_mode": "diagnostic",
            "p_error_count": int(np.sum(p_error)),
            "q_error_count": int(np.sum(q_error)),
            "both_error_count": int(np.sum(p_error & q_error)),
        },
    )


@dataclass(frozen=True)
class CompetingGoalBundle:
    """Matched train/IID data and the three identifying diagnostic panels."""

    train: SemanticBatch
    iid: SemanticBatch
    diagnostics: Mapping[str, SemanticBatch]

    def __post_init__(self) -> None:
        missing = set(DIAGNOSTIC_PANELS) - set(self.diagnostics)
        if missing:
            raise ValueError(f"diagnostic bundle is missing panels: {sorted(missing)}")
        object.__setattr__(self, "diagnostics", MappingProxyType(dict(self.diagnostics)))

    @property
    def both_wrong(self) -> SemanticBatch:
        return self.diagnostics["both_wrong"]

    @property
    def p_wrong(self) -> SemanticBatch:
        return self.diagnostics["p_wrong"]

    @property
    def q_wrong(self) -> SemanticBatch:
        return self.diagnostics["q_wrong"]


def make_competing_bundle(
    n_train: int,
    n_iid: int,
    n_diagnostic: int,
    q_p: float,
    q_q: float,
    k_q: int,
    k_y: int,
    seed: int = 0,
    *,
    max_k_q: int | None = None,
    max_k_y: int | None = None,
    overlap: OverlapMode = "independent",
    q_only_error_count: int | None = None,
    state_dim: int = 0,
    id_offset: int = 0,
) -> CompetingGoalBundle:
    """Build all data needed for one competing-goal training cell."""

    n_train = _positive_even(n_train, "n_train")
    n_iid = _positive_even(n_iid, "n_iid")
    n_diagnostic = _positive_even(n_diagnostic, "n_diagnostic")
    max_k_q = k_q if max_k_q is None else max_k_q
    max_k_y = k_y if max_k_y is None else max_k_y

    train = make_competing_dataset(
        n_train,
        q_p,
        q_q,
        k_q,
        k_y,
        seed,
        max_k_q=max_k_q,
        max_k_y=max_k_y,
        overlap=overlap,
        q_only_error_count=q_only_error_count,
        state_dim=state_dim,
        split="train",
        id_offset=id_offset,
    )
    iid_offset = id_offset + n_train + 1
    iid = make_competing_dataset(
        n_iid,
        q_p,
        q_q,
        k_q,
        k_y,
        seed,
        max_k_q=max_k_q,
        max_k_y=max_k_y,
        overlap=overlap,
        state_dim=state_dim,
        split="iid",
        id_offset=iid_offset,
    )
    panel_offset = iid_offset + n_iid + 1
    diagnostics: dict[str, SemanticBatch] = {}
    for index, panel in enumerate(DIAGNOSTIC_PANELS):
        diagnostics[panel] = make_competing_diagnostic(
            n_diagnostic,
            panel,
            k_q,
            k_y,
            seed,
            max_k_q=max_k_q,
            max_k_y=max_k_y,
            state_dim=state_dim,
            id_offset=panel_offset + index * (n_diagnostic + 1),
        )
    return CompetingGoalBundle(train=train, iid=iid, diagnostics=diagnostics)


def decode_competing_rules(batch: SemanticBatch) -> dict[str, SignArray]:
    """Decode the three model-visible candidate recommendations in ``batch``."""

    k_q = _positive_degree(int(batch.metadata["k_q"]), "metadata.k_q")
    k_y = _positive_degree(int(batch.metadata["k_y"]), "metadata.k_y")
    q_names = [f"Q_{index}" for index in range(1, k_q + 1)]
    y_names = [f"R_{index}" for index in range(1, k_y + 1)]
    missing = [name for name in ("P", *q_names, *y_names) if name not in batch.channels]
    if missing:
        raise KeyError(f"competing-goal batch is missing channels: {missing}")
    q_code = np.column_stack([np.asarray(batch.channels[name], dtype=np.int8) for name in q_names])
    y_code = np.column_stack([np.asarray(batch.channels[name], dtype=np.int8) for name in y_names])
    return {
        "P": np.asarray(batch.channels["P"], dtype=np.int8),
        "Q": np.prod(q_code, axis=1, dtype=np.int8),
        "Y_code": np.prod(y_code, axis=1, dtype=np.int8),
        "target": np.asarray(batch.y, dtype=np.int8),
    }


def _prediction_signs(prediction: ArrayLike, n: int) -> SignArray:
    values = np.asarray(prediction)
    if values.ndim == 2:
        if values.shape != (n, 2):
            raise ValueError(f"two-class logits must have shape {(n, 2)}, got {values.shape}")
        return binary_predictions(values).astype(np.int8, copy=False)
    if values.ndim != 1 or len(values) != n:
        raise ValueError(f"predictions must have shape [{n}] or [{n},2], got {values.shape}")
    if np.all(np.isin(values, (-1, 1))):
        return values.astype(np.int8, copy=False)
    if np.all(np.isin(values, (0, 1))):
        return np.where(values > 0, 1, -1).astype(np.int8)
    return np.where(values >= 0, 1, -1).astype(np.int8)


def competing_rule_agreements(
    prediction: ArrayLike,
    batch: SemanticBatch,
) -> dict[str, float | int]:
    """Measure agreement with every candidate rule and the semantic target."""

    rules = decode_competing_rules(batch)
    signs = _prediction_signs(prediction, len(batch))
    return {
        "rho_p": float(np.mean(signs == rules["P"])),
        "rho_q": float(np.mean(signs == rules["Q"])),
        "rho_y_code": float(np.mean(signs == rules["Y_code"])),
        "target_accuracy": float(np.mean(signs == rules["target"])),
        "n": len(batch),
    }


def competing_causal_flip_batches(batch: SemanticBatch) -> dict[str, SemanticBatch]:
    """Return exact one-channel interventions for every candidate computation.

    Flipping ``P`` reverses only the direct proxy.  Flipping any active Q or Y
    code component reverses exactly that decoded recommendation because each is
    represented by a product code.  Padding channels are deliberately omitted.
    """

    k_q = _positive_degree(int(batch.metadata["k_q"]), "metadata.k_q")
    k_y = _positive_degree(int(batch.metadata["k_y"]), "metadata.k_y")
    result = {"flip_P": flip_channels(batch, "P")}
    result.update(
        {
            f"flip_Q_{index}": flip_channels(batch, f"Q_{index}")
            for index in range(1, k_q + 1)
        }
    )
    result.update(
        {
            f"flip_Y_{index}": flip_channels(batch, f"R_{index}")
            for index in range(1, k_y + 1)
        }
    )
    return result


__all__ = [
    "DIAGNOSTIC_PANELS",
    "CompetingGoalBundle",
    "DiagnosticPanel",
    "OverlapMode",
    "competing_causal_flip_batches",
    "competing_rule_agreements",
    "decode_competing_rules",
    "make_competing_bundle",
    "make_competing_dataset",
    "make_competing_diagnostic",
    "make_competing_factorial_dataset",
]
