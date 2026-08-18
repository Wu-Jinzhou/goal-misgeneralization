"""Deterministic construction and audit helpers for winner-knockout handoff.

E17 removes the information in the direct proxy without moving that coordinate
off its ordinary ``{-1,+1}`` support.  This module keeps that intervention,
compute-sham data, state hashing, and endpoint definitions independent of the
training protocol so their causal invariants can be tested directly.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping
from typing import Any

import numpy as np
import torch

from .competing import (
    decode_competing_rules,
    make_competing_dataset,
    make_competing_factorial_dataset,
)
from .data import SemanticBatch


def _stable_seed(seed: int, *parts: object) -> int:
    digest = hashlib.blake2b(digest_size=8, person=b"forkhand")
    for part in (seed, *parts):
        payload = str(part).encode("utf-8")
        digest.update(len(payload).to_bytes(4, "little"))
        digest.update(payload)
    return int.from_bytes(digest.digest(), "little")


def _update_digest(digest: Any, value: Any) -> None:
    """Update a hash from nested tensor state without serialization metadata."""

    if torch.is_tensor(value):
        tensor = value.detach().cpu().contiguous()
        array = tensor.numpy()
        digest.update(b"tensor\0")
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(np.asarray(tensor.shape, dtype=np.int64).tobytes())
        digest.update(array.tobytes(order="C"))
        return
    if isinstance(value, np.ndarray):
        array = np.ascontiguousarray(value)
        digest.update(b"ndarray\0")
        digest.update(array.dtype.str.encode("ascii"))
        digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
        digest.update(array.tobytes(order="C"))
        return
    if isinstance(value, Mapping):
        digest.update(b"mapping\0")
        items = sorted(value.items(), key=lambda item: (type(item[0]).__name__, repr(item[0])))
        for key, item in items:
            _update_digest(digest, key)
            _update_digest(digest, item)
        digest.update(b"mapping-end\0")
        return
    if isinstance(value, (list, tuple)):
        digest.update(b"list\0" if isinstance(value, list) else b"tuple\0")
        for item in value:
            _update_digest(digest, item)
        digest.update(b"sequence-end\0")
        return
    if isinstance(value, (np.integer, int)) and not isinstance(value, (np.bool_, bool)):
        digest.update(b"int\0" + str(int(value)).encode("ascii") + b"\0")
        return
    if isinstance(value, (np.floating, float)):
        digest.update(b"float\0" + float(value).hex().encode("ascii") + b"\0")
        return
    if isinstance(value, (np.bool_, bool)):
        digest.update(b"bool\0" + (b"1" if bool(value) else b"0"))
        return
    if value is None:
        digest.update(b"none\0")
        return
    if isinstance(value, bytes):
        digest.update(b"bytes\0" + len(value).to_bytes(8, "little") + value)
        return
    if isinstance(value, str):
        encoded = value.encode("utf-8")
        digest.update(b"str\0" + len(encoded).to_bytes(8, "little") + encoded)
        return
    digest.update(b"repr\0" + repr(value).encode("utf-8") + b"\0")


def stable_state_digest(value: Any) -> str:
    """Return a deterministic digest for model, optimizer, or sampler state."""

    digest = hashlib.blake2b(digest_size=32, person=b"forkstate")
    _update_digest(digest, value)
    return digest.hexdigest()


def semantic_batch_digest(batch: SemanticBatch) -> str:
    """Digest every training-relevant field and the semantic audit fields."""

    payload = {
        "y": np.asarray(batch.y),
        "target": np.asarray(batch.target),
        "reward": np.asarray(batch.reward),
        "channels": {name: np.asarray(value) for name, value in batch.channels.items()},
        "latents": {name: np.asarray(value) for name, value in batch.latents.items()},
        "state": None if batch.state is None else np.asarray(batch.state),
        "sample_id": np.asarray(batch.sample_id),
        "state_id": np.asarray(batch.state_id),
        "episode_id": np.asarray(batch.episode_id),
        "step_id": np.asarray(batch.step_id),
    }
    return stable_state_digest(payload)


def static_sampler_digest(
    n: int,
    *,
    batch_size: int,
    steps: int,
    seed: int,
    shuffle: bool = True,
) -> str:
    """Hash the exact static minibatch-index stream used by ``train_sft``."""

    if n < 1 or batch_size < 1 or steps < 0:
        raise ValueError("n and batch_size must be positive and steps non-negative")
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    order = torch.empty(0, dtype=torch.long)
    cursor = 0
    digest = hashlib.blake2b(digest_size=32, person=b"forkbatch")
    for step in range(1, steps + 1):
        if cursor >= len(order):
            order = (
                torch.randperm(n, generator=generator)
                if shuffle
                else torch.arange(n, dtype=torch.long)
            )
            cursor = 0
        stop = min(cursor + batch_size, len(order))
        selected = order[cursor:stop].numpy().astype(np.int64, copy=False)
        digest.update(step.to_bytes(8, "little"))
        digest.update(len(selected).to_bytes(8, "little"))
        digest.update(selected.tobytes())
        cursor = stop
    return digest.hexdigest()


def make_handoff_phase_b(
    n: int,
    *,
    q_q: float,
    k_q: int,
    k_y: int,
    seed: int,
    max_k_q: int,
    max_k_y: int,
    state_dim: int,
    id_offset: int = 3_000_000_000,
) -> SemanticBatch:
    """Construct the common in-support random-``P`` phase-B distribution."""

    if isinstance(n, bool) or n < 2 or n % 2:
        raise ValueError("phase-B n must be a positive even integer")
    base = make_competing_dataset(
        n // 2,
        q_p=0.5,
        q_q=q_q,
        k_q=k_q,
        k_y=k_y,
        seed=seed,
        max_k_q=max_k_q,
        max_k_y=max_k_y,
        overlap="independent",
        state_dim=state_dim,
        split="handoff_phase_b_base",
        id_offset=id_offset + n + 1,
    )

    def duplicate(array: Any) -> np.ndarray:
        return np.repeat(np.asarray(array), 2, axis=0)

    y = duplicate(base.y).astype(np.int8, copy=False)
    p = np.tile(np.asarray([-1, 1], dtype=np.int8), len(base))
    channels = {name: duplicate(value) for name, value in base.channels.items()}
    channels["P"] = p
    q_goal = duplicate(base.latents["Q_goal"]).astype(np.int8, copy=False)
    raw_channel_names = tuple(
        [f"Q_{index}" for index in range(1, k_q + 1)]
        + [f"R_{index}" for index in range(1, k_y + 1)]
    )
    raw_signs = np.column_stack(
        [np.asarray(base.channels[name], dtype=np.int8) for name in raw_channel_names]
    )
    raw_codeword_id = np.sum(
        (raw_signs > 0).astype(np.int64)
        * (np.int64(1) << np.arange(len(raw_channel_names), dtype=np.int64)),
        axis=1,
        dtype=np.int64,
    )
    p_error = p != y
    q_error = q_goal != y
    pair_id = np.repeat(np.arange(len(base), dtype=np.int64), 2)
    latents = {
        "P_goal": p,
        "Q_goal": q_goal,
        "P_error": p_error,
        "Q_error": q_error,
        "both_wrong": p_error & q_error,
        "p_wrong": p_error & ~q_error,
        "q_wrong": ~p_error & q_error,
        "handoff_pair_id": pair_id,
        "raw_codeword_id": duplicate(raw_codeword_id),
    }
    generator = np.random.default_rng(_stable_seed(seed, "phase-b-row-order"))
    order = generator.permutation(n)
    state = None if base.state is None else duplicate(base.state)[order]
    state_id = duplicate(base.state_id)[order]
    metadata = dict(base.metadata)
    p_errors = int(np.sum(p_error))
    q_errors = int(np.sum(q_error))
    both = int(np.sum(p_error & q_error))
    metadata.update(
        {
            "condition": "winner_knockout_random_p",
            "split": "handoff_phase_b",
            "informational_knockout": (
                "paired clones share all model inputs except opposite P"
            ),
            "raw_channel_order_without_p": raw_channel_names,
            "raw_codeword_count_without_p": 1 << len(raw_channel_names),
            "phase": "B",
            "requested_q_p": 0.5,
            "realized_q_p": 0.5,
            "realized_q_q": float(np.mean(q_goal == y)),
            "p_error_count": p_errors,
            "q_error_count": q_errors,
            "both_error_count": both,
            "p_only_error_count": p_errors - both,
            "q_only_error_count": q_errors - both,
            "neither_error_count": n - p_errors - q_errors + both,
            "both_error_rate": both / n,
            "overlap_mode": "paired_random_p",
            "error_phi": 0.0,
        }
    )
    result = SemanticBatch(
        y=y[order],
        target=y[order],
        reward=y[order].astype(np.float32),
        channels={name: values[order] for name, values in channels.items()},
        latents={name: values[order] for name, values in latents.items()},
        state=state,
        sample_id=np.arange(id_offset, id_offset + n, dtype=np.int64),
        state_id=state_id,
        episode_id=np.arange(id_offset, id_offset + n, dtype=np.int64),
        step_id=np.zeros(n, dtype=np.int64),
        metadata=metadata,
    )
    audit_handoff_phase_b(result, expected_q_q=q_q)
    return result


def _tuple_key(p: int, q: int, y: int) -> str:
    return "".join("+" if value > 0 else "-" for value in (p, q, y))


def audit_handoff_phase_b(
    batch: SemanticBatch,
    *,
    expected_q_q: float,
) -> dict[str, Any]:
    """Fail closed unless random ``P`` and all candidate relations are exact."""

    rules = decode_competing_rules(batch)
    p = np.asarray(rules["P"], dtype=np.int8)
    q = np.asarray(rules["Q"], dtype=np.int8)
    y = np.asarray(rules["target"], dtype=np.int8)
    y_code = np.asarray(rules["Y_code"], dtype=np.int8)
    if not np.array_equal(y, y_code):
        raise RuntimeError("phase-B exact Y code does not decode to the target")

    agreements = {
        "p_y": float(np.mean(p == y)),
        "p_q": float(np.mean(p == q)),
        "q_y": float(np.mean(q == y)),
    }
    if not math.isclose(agreements["p_y"], 0.5, abs_tol=1e-12):
        raise RuntimeError("phase-B P must agree exactly 0.5 with Y")
    if not math.isclose(agreements["p_q"], 0.5, abs_tol=1e-12):
        raise RuntimeError("phase-B P must agree exactly 0.5 with Q")
    if not math.isclose(agreements["q_y"], float(expected_q_q), abs_tol=1e-12):
        raise RuntimeError("phase-B Q accuracy differs from the requested value")

    tuple_counts: dict[str, int] = {}
    conditional_p_counts: dict[str, dict[str, int]] = {}
    for q_value in (-1, 1):
        for y_value in (-1, 1):
            cell = (q == q_value) & (y == y_value)
            negative = int(np.sum(cell & (p == -1)))
            positive = int(np.sum(cell & (p == 1)))
            key = _tuple_key(1, q_value, y_value)[1:]
            conditional_p_counts[key] = {"negative": negative, "positive": positive}
            if negative != positive or negative == 0:
                raise RuntimeError(
                    "phase-B P must be balanced and represented within every (Q,Y) cell"
                )
    for p_value in (-1, 1):
        for q_value in (-1, 1):
            for y_value in (-1, 1):
                key = _tuple_key(p_value, q_value, y_value)
                tuple_counts[key] = int(
                    np.sum((p == p_value) & (q == q_value) & (y == y_value))
                )
    if any(value <= 0 for value in tuple_counts.values()):
        raise RuntimeError("phase-B batch must represent all eight candidate tuples")

    raw_ids = np.asarray(batch.latents["raw_codeword_id"], dtype=np.int64)
    raw_p_counts: dict[str, dict[str, int]] = {}
    for raw_id in np.unique(raw_ids):
        cell = raw_ids == raw_id
        negative = int(np.sum(cell & (p == -1)))
        positive = int(np.sum(cell & (p == 1)))
        raw_p_counts[str(int(raw_id))] = {
            "negative": negative,
            "positive": positive,
        }
        if negative != positive or negative == 0:
            raise RuntimeError(
                "phase-B P must be exactly balanced within every active raw codeword"
            )
    expected_raw_count = int(batch.metadata["raw_codeword_count_without_p"])
    if len(raw_p_counts) != expected_raw_count:
        raise RuntimeError("phase-B batch must represent every active Q/R raw codeword")

    pair_ids = np.asarray(batch.latents.get("handoff_pair_id"), dtype=np.int64)
    unique_pairs, pair_counts = np.unique(pair_ids, return_counts=True)
    if len(unique_pairs) * 2 != len(batch) or not np.all(pair_counts == 2):
        raise RuntimeError("phase-B rows must form exact clone pairs")
    features = batch.features(max_k=int(batch.metadata["max_k"]), include_state=True)
    feature_names = batch.feature_names(
        max_k=int(batch.metadata["max_k"]), include_state=True
    )
    try:
        p_column = feature_names.index("P")
    except ValueError as exc:  # pragma: no cover - guarded by competing interface
        raise RuntimeError("phase-B features contain no P coordinate") from exc
    non_p = np.delete(features, p_column, axis=1)
    for pair_id in unique_pairs:
        positions = np.flatnonzero(pair_ids == pair_id)
        if not np.array_equal(non_p[positions[0]], non_p[positions[1]]):
            raise RuntimeError("phase-B clone pair differs outside P")
        if sorted(p[positions].tolist()) != [-1, 1]:
            raise RuntimeError("phase-B clone pair must contain opposite P signs")
        if y[positions[0]] != y[positions[1]] or q[positions[0]] != q[positions[1]]:
            raise RuntimeError("phase-B clone pair changed Q or Y")
    return {
        "n": len(batch),
        "agreements": agreements,
        "candidate_tuple_counts": tuple_counts,
        "conditional_p_counts": conditional_p_counts,
        "raw_codeword_p_counts": raw_p_counts,
        "raw_codeword_count": len(raw_p_counts),
        "all_eight_candidate_tuples_present": True,
        "paired_clone_count": len(unique_pairs),
        "p_balanced_conditional_on_all_other_features": True,
        "batch_digest": semantic_batch_digest(batch),
    }


def make_compute_sham(
    *,
    n: int = 10_000,
    repeats: int,
    k_q: int,
    k_y: int,
    seed: int,
    control_seed: int,
    max_k_q: int,
    max_k_y: int,
    state_dim: int,
    id_offset: int = 3_500_000_000,
) -> SemanticBatch:
    """Make a common generic-compute sham with labels balanced per codeword."""

    if isinstance(n, bool) or n < 2 or n % 2:
        raise ValueError("sham n must be a positive even integer")
    raw_codeword_count = 2 ** (1 + k_q + k_y)
    base_n = n // 2
    quotient, remainder = divmod(base_n, raw_codeword_count)
    if quotient < 1:
        raise ValueError("sham n must include every active raw codeword before cloning")
    required_repeats = quotient + int(remainder > 0)
    if isinstance(repeats, bool) or repeats < required_repeats:
        raise ValueError(
            "sham factorial source repeats are insufficient for the requested n"
        )
    source = make_competing_factorial_dataset(
        repeats=repeats,
        k_q=k_q,
        k_y=k_y,
        seed=seed,
        control_seed=control_seed,
        max_k_q=max_k_q,
        max_k_y=max_k_y,
        state_dim=state_dim,
        split="handoff_compute_sham",
        id_offset=id_offset + 10_001,
    )
    source_ids = np.asarray(source.latents["raw_codeword_id"], dtype=np.int64)
    selected: list[int] = []
    for raw_id in sorted(np.unique(source_ids).tolist()):
        positions = np.flatnonzero(source_ids == raw_id)
        keep = quotient + int(int(raw_id) < remainder)
        if len(positions) < keep:
            raise RuntimeError("sham factorial source has insufficient codeword repeats")
        selected.extend(positions[:keep].tolist())
    if len(selected) != base_n:
        raise RuntimeError("sham base has the wrong number of rows")
    base = source.select(np.asarray(selected, dtype=np.int64))

    def duplicate(array: Any) -> np.ndarray:
        return np.repeat(np.asarray(array), 2, axis=0)

    labels = np.tile(np.asarray([-1, 1], dtype=np.int8), len(base))
    pair_id = np.repeat(np.arange(len(base), dtype=np.int64), 2)
    channels = {name: duplicate(value) for name, value in base.channels.items()}
    latents = {name: duplicate(value) for name, value in base.latents.items()}
    latents["sham_label"] = labels
    latents["sham_pair_id"] = pair_id
    generator = np.random.default_rng(_stable_seed(seed, "sham-row-order"))
    order = generator.permutation(n)
    metadata = dict(base.metadata)
    metadata.pop("repeats_per_codeword", None)
    raw_counts = {
        str(int(raw_id)): int(np.sum(duplicate(base.latents["raw_codeword_id"]) == raw_id))
        for raw_id in np.unique(base.latents["raw_codeword_id"])
    }
    metadata.update(
        {
            "condition": "generic_compute_sham",
            "split": "handoff_compute_sham",
            "phase": "A",
            "sham_label_construction": (
                "paired clones share every model input and have opposite labels"
            ),
            "sham_factorial_source_repeats": repeats,
            "sham_requested_n": n,
            "sham_raw_codeword_counts": raw_counts,
            "sham_raw_codeword_count_min": min(raw_counts.values()),
            "sham_raw_codeword_count_max": max(raw_counts.values()),
        }
    )
    result = SemanticBatch(
        y=duplicate(base.y)[order],
        target=labels[order],
        reward=labels[order].astype(np.float32),
        channels={name: values[order] for name, values in channels.items()},
        latents={name: values[order] for name, values in latents.items()},
        state=None if base.state is None else duplicate(base.state)[order],
        sample_id=np.arange(id_offset, id_offset + n, dtype=np.int64),
        state_id=duplicate(base.state_id)[order],
        episode_id=np.arange(id_offset, id_offset + n, dtype=np.int64),
        step_id=np.zeros(n, dtype=np.int64),
        metadata=metadata,
    )
    audit_compute_sham(result)
    return result


def audit_compute_sham(batch: SemanticBatch) -> dict[str, Any]:
    """Verify exact sham-label balance against raw and decoded candidates."""

    labels = np.asarray(batch.target, dtype=np.int8)
    raw_ids = np.asarray(batch.latents["raw_codeword_id"], dtype=np.int64)
    rules = decode_competing_rules(batch)
    raw_balance = {
        str(int(raw_id)): float(np.mean(labels[raw_ids == raw_id] > 0))
        for raw_id in np.unique(raw_ids)
    }
    if any(not math.isclose(value, 0.5, abs_tol=1e-12) for value in raw_balance.values()):
        raise RuntimeError("sham labels must be balanced within every raw codeword")
    agreements = {
        "P": float(np.mean(labels == rules["P"])),
        "Q": float(np.mean(labels == rules["Q"])),
        "Y": float(np.mean(labels == rules["Y_code"])),
    }
    if any(not math.isclose(value, 0.5, abs_tol=1e-12) for value in agreements.values()):
        raise RuntimeError("sham labels must agree exactly 0.5 with P, Q, and Y")
    pair_ids = np.asarray(batch.latents.get("sham_pair_id"), dtype=np.int64)
    unique_pairs, pair_counts = np.unique(pair_ids, return_counts=True)
    if len(unique_pairs) * 2 != len(batch) or not np.all(pair_counts == 2):
        raise RuntimeError("sham must consist entirely of exact clone pairs")
    features = batch.features(max_k=int(batch.metadata["max_k"]), include_state=True)
    for pair_id in unique_pairs:
        positions = np.flatnonzero(pair_ids == pair_id)
        if not np.array_equal(features[positions[0]], features[positions[1]]):
            raise RuntimeError("sham clone pair has unequal model inputs")
        if sorted(labels[positions].tolist()) != [-1, 1]:
            raise RuntimeError("sham clone pair must have opposite labels")
    return {
        "n": len(batch),
        "positive_fraction": float(np.mean(labels > 0)),
        "candidate_agreements": agreements,
        "raw_codeword_count": len(raw_balance),
        "raw_codeword_counts": {
            key: int(np.sum(raw_ids == int(key))) for key in raw_balance
        },
        "paired_clone_count": len(unique_pairs),
        "opposite_labels_conditional_on_complete_input": True,
        "balanced_within_every_raw_codeword": True,
        "batch_digest": semantic_batch_digest(batch),
    }


def pure_control(
    behavior: Mapping[str, float],
    causal: Mapping[str, float],
    goal: str,
    *,
    threshold: float = 0.90,
    margin: float = 0.10,
) -> bool:
    """Return the frozen E17 pure behavioral-and-causal control predicate."""

    if goal not in {"P", "Q", "Y"}:
        raise ValueError("goal must be P, Q, or Y")
    others = tuple(candidate for candidate in ("P", "Q", "Y") if candidate != goal)
    return bool(
        float(behavior[goal]) >= threshold
        and float(causal[goal]) >= threshold
        and all(float(behavior[goal]) - float(behavior[other]) >= margin for other in others)
        and all(float(causal[goal]) - float(causal[other]) >= margin for other in others)
    )


def persistent_handoff(
    snapshots: Mapping[int, Mapping[str, Mapping[str, float]]],
    goal: str,
    *,
    threshold: float = 0.90,
    margin: float = 0.10,
) -> dict[str, Any]:
    """Locate the second of two consecutive qualifying logged checkpoints."""

    steps = sorted(int(step) for step in snapshots)
    qualified = [
        pure_control(
            snapshots[step]["behavior"],
            snapshots[step]["causal"],
            goal,
            threshold=threshold,
            margin=margin,
        )
        for step in steps
    ]
    for index in range(1, len(steps)):
        if qualified[index - 1] and qualified[index]:
            return {
                "observed": True,
                "first_qualifying_step": steps[index - 1],
                "confirmation_step": steps[index],
                "censor_step": steps[-1],
            }
    return {
        "observed": False,
        "first_qualifying_step": None,
        "confirmation_step": None,
        "censor_step": steps[-1] if steps else None,
    }


def normalized_control_auc(
    snapshots: Mapping[int, Mapping[str, Mapping[str, float]]],
    goal: str,
    *,
    horizon: int,
) -> float:
    """Linear-in-update AUC of mean behavior and causal control through horizon."""

    if horizon < 1:
        raise ValueError("AUC horizon must be positive")
    steps = sorted(int(step) for step in snapshots)
    if not steps or steps[0] != 0 or steps[-1] < horizon:
        raise ValueError("snapshots must span step zero through the AUC horizon")
    values = np.asarray(
        [
            0.5
            * (
                float(snapshots[step]["behavior"][goal])
                + float(snapshots[step]["causal"][goal])
            )
            for step in steps
        ],
        dtype=np.float64,
    )
    step_array = np.asarray(steps, dtype=np.float64)
    interior = step_array[step_array < horizon]
    x = np.concatenate((interior, np.asarray([float(horizon)])))
    y = np.interp(x, step_array, values)
    return float(np.trapezoid(y, x) / float(horizon))


__all__ = [
    "audit_compute_sham",
    "audit_handoff_phase_b",
    "make_compute_sham",
    "make_handoff_phase_b",
    "normalized_control_auc",
    "persistent_handoff",
    "pure_control",
    "semantic_batch_digest",
    "stable_state_digest",
    "static_sampler_digest",
]
