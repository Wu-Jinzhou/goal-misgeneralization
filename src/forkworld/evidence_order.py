"""Factorial data and exact atomic-batch plans for evidence-order studies.

E18 changes only the order of two diagnostic evidence blocks.  Every schedule
uses the same unique rows, within-category atomic batches, presentation counts,
and common prefix/washout.  Raw-codeword and target-label balancing is exact in
every optimizer batch, rather than true only in expectation.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray

from .competing import make_competing_factorial_dataset
from .data import SemanticBatch
from .handoff import semantic_batch_digest, stable_state_digest

OrderSchedule = Literal["b_then_d", "d_then_b", "interleave"]
ORDER_SCHEDULES: tuple[OrderSchedule, ...] = (
    "b_then_d",
    "d_then_b",
    "interleave",
)
STRATA = ("A", "B", "D")


def _stable_seed(seed: int, *parts: object) -> int:
    digest = hashlib.blake2b(digest_size=8, person=b"forkordr")
    for part in (seed, *parts):
        payload = str(part).encode("utf-8")
        digest.update(len(payload).to_bytes(4, "little"))
        digest.update(payload)
    return int.from_bytes(digest.digest(), "little")


def evidence_strata(batch: SemanticBatch) -> dict[str, NDArray[np.bool_]]:
    """Return the registered nested-support row categories A, B, and D."""

    required = {"P_error", "Q_error"}
    missing = required - set(batch.latents)
    if missing:
        raise KeyError(f"identical-evidence batch lacks latents {sorted(missing)}")
    p_error = np.asarray(batch.latents["P_error"], dtype=bool)
    q_error = np.asarray(batch.latents["Q_error"], dtype=bool)
    if p_error.shape != (len(batch),) or q_error.shape != (len(batch),):
        raise ValueError("proxy-error latents must align with the semantic batch")
    return {
        "A": ~p_error & ~q_error,
        "B": p_error & ~q_error,
        "D": p_error & q_error,
    }


def _exact_error_count(n: int, q: float, name: str) -> int:
    raw = n * (1.0 - float(q))
    if not 0.0 <= q <= 1.0 or abs(raw - round(raw)) > 1e-9:
        raise ValueError(f"{name}={q:g} is not exactly realizable with n={n}")
    return round(raw)


def make_identical_evidence_dataset(
    n: int,
    *,
    q_p: float,
    q_q: float,
    k_q: int,
    k_y: int,
    seed: int,
    max_k_q: int,
    max_k_y: int,
    state_dim: int,
    id_offset: int = 4_000_000_000,
) -> SemanticBatch:
    """Select exact A/B/D counts from a unique-row exhaustive source.

    Within each category and target sign, the selected raw codewords differ in
    count by at most one.  Full E18 therefore has A counts 562/563 and B/D
    counts 31/32 while preserving all compatible raw codewords.
    """

    if isinstance(n, bool) or n < 1 or n % 2:
        raise ValueError("identical-evidence n must be a positive even integer")
    p_errors = _exact_error_count(n, q_p, "q_p")
    q_errors = _exact_error_count(n, q_q, "q_q")
    if q_errors > p_errors:
        raise ValueError("nested identical evidence requires q_q >= q_p")
    desired = {"A": n - p_errors, "B": p_errors - q_errors, "D": q_errors}
    if any(count <= 0 or count % 2 for count in desired.values()):
        raise ValueError("A/B/D category counts must be positive and even")

    raw_per_category = 1 << (k_q + k_y - 1)
    raw_per_label = raw_per_category // 2
    if raw_per_label < 1:
        raise ValueError("active code width is too small for factorial balancing")
    required_repeats = max(
        (desired[name] // 2 + raw_per_label - 1) // raw_per_label for name in STRATA
    )
    source = make_competing_factorial_dataset(
        repeats=required_repeats,
        k_q=k_q,
        k_y=k_y,
        seed=seed,
        control_seed=1_815_018_015,
        max_k_q=max_k_q,
        max_k_y=max_k_y,
        state_dim=state_dim,
        split="identical_evidence_factorial_source",
        id_offset=id_offset,
    )
    masks = evidence_strata(source)
    target = np.asarray(source.target, dtype=np.int8)
    raw_ids = np.asarray(source.latents["raw_codeword_id"], dtype=np.int64)
    replicate_ids = np.asarray(source.latents["replicate_id"], dtype=np.int64)
    selected: list[int] = []
    selected_raw_counts: dict[str, dict[str, int]] = {}
    for name in STRATA:
        per_label = desired[name] // 2
        selected_raw_counts[name] = {}
        for sign in (-1, 1):
            compatible = sorted(np.unique(raw_ids[masks[name] & (target == sign)]).tolist())
            if len(compatible) != raw_per_label:
                raise RuntimeError(
                    f"stratum {name}, sign {sign} has {len(compatible)} raw codewords; "
                    f"expected {raw_per_label}"
                )
            quotient, remainder = divmod(per_label, len(compatible))
            for rank, raw_id in enumerate(compatible):
                keep = quotient + int(rank < remainder)
                positions = np.flatnonzero(
                    masks[name] & (target == sign) & (raw_ids == raw_id)
                )
                positions = positions[np.argsort(replicate_ids[positions], kind="stable")]
                if keep < 1 or len(positions) < keep:
                    raise RuntimeError("factorial source has insufficient unique rows")
                selected.extend(positions[:keep].tolist())
                selected_raw_counts[name][str(int(raw_id))] = keep
    if len(selected) != n or len(set(selected)) != n:
        raise RuntimeError("factorial selection did not produce n unique rows")
    generator = np.random.default_rng(_stable_seed(seed, "selected-row-order"))
    selection = np.asarray(selected, dtype=np.int64)[generator.permutation(n)]
    result = source.select(selection)
    metadata = dict(result.metadata)
    metadata.pop("repeats_per_codeword", None)
    metadata.update(
        {
            "dataset": "identical_evidence_order",
            "condition": "factorial_balanced_nested_support",
            "split": "train",
            "requested_q_p": float(q_p),
            "requested_q_q": float(q_q),
            "realized_q_p": 1.0 - p_errors / n,
            "realized_q_q": 1.0 - q_errors / n,
            "overlap_mode": "nested",
            "p_error_count": p_errors,
            "q_error_count": q_errors,
            "both_error_count": q_errors,
            "p_only_error_count": p_errors - q_errors,
            "q_only_error_count": 0,
            "neither_error_count": n - p_errors,
            "selected_raw_codeword_counts": selected_raw_counts,
            "factorial_source_repeats": required_repeats,
        }
    )
    result = result.with_updates(metadata=metadata)
    audit_evidence_strata(result, expected_counts=desired)
    return result


def audit_evidence_strata(
    batch: SemanticBatch,
    *,
    expected_counts: dict[str, int] | None = None,
) -> dict[str, Any]:
    """Audit category coverage, target balance, and factorial raw balance."""

    masks = evidence_strata(batch)
    target = np.asarray(batch.target, dtype=np.int8)
    raw_ids = np.asarray(batch.latents.get("raw_codeword_id"), dtype=np.int64)
    if raw_ids.shape != (len(batch),):
        raise ValueError("identical-evidence data require raw_codeword_id latents")
    if not np.all(np.isin(target, (-1, 1))):
        raise ValueError("identical-evidence targets must be signs")
    q_only_count = int(
        np.sum(
            ~np.asarray(batch.latents["P_error"], dtype=bool)
            & np.asarray(batch.latents["Q_error"], dtype=bool)
        )
    )
    if q_only_count:
        raise RuntimeError("identical-evidence data may not contain Q-only errors")
    if not np.all(sum(mask.astype(np.int8) for mask in masks.values()) == 1):
        raise RuntimeError("A/B/D strata must partition the training rows")
    counts: dict[str, dict[str, Any]] = {}
    for name, mask in masks.items():
        negative = int(np.sum(mask & (target == -1)))
        positive = int(np.sum(mask & (target == 1)))
        total = int(np.sum(mask))
        if total == 0 or negative != positive:
            raise RuntimeError(f"stratum {name} must be nonempty and label-balanced")
        if expected_counts is not None and total != int(expected_counts[name]):
            raise RuntimeError(
                f"stratum {name} has {total} rows, expected {expected_counts[name]}"
            )
        per_raw = {
            str(int(raw_id)): int(np.sum(mask & (raw_ids == raw_id)))
            for raw_id in np.unique(raw_ids[mask])
        }
        per_label_raw_counts: dict[str, list[int]] = {}
        for sign in (-1, 1):
            values = [
                int(np.sum(mask & (target == sign) & (raw_ids == raw_id)))
                for raw_id in np.unique(raw_ids[mask & (target == sign)])
            ]
            if not values or max(values) - min(values) > 1:
                raise RuntimeError(
                    f"stratum {name}, sign {sign} raw counts must differ by at most one"
                )
            per_label_raw_counts[str(sign)] = values
        counts[name] = {
            "negative": negative,
            "positive": positive,
            "total": total,
            "raw_codeword_count": len(per_raw),
            "raw_codeword_counts": per_raw,
            "per_label_raw_count_values": per_label_raw_counts,
        }
    return {
        "n": len(batch),
        "strata": counts,
        "q_only_count": q_only_count,
        "partition_verified": True,
        "label_balance_verified": True,
        "factorial_raw_balance_verified": True,
        "batch_digest": semantic_batch_digest(batch),
    }


def _binary_quota_matrix(
    exposures: NDArray[np.int64],
    *,
    n_batches: int,
    per_batch_total: int,
    seed: int,
) -> NDArray[np.int64]:
    """Allocate floor/ceiling codeword quotas with exact row/column sums."""

    base = exposures // n_batches
    remaining = exposures - base * n_batches
    row_extras = int(per_batch_total - np.sum(base))
    if row_extras < 0 or row_extras > len(exposures):
        raise ValueError("raw-codeword exposures cannot realize the atomic batch total")
    if int(np.sum(remaining)) != row_extras * n_batches:
        raise ValueError("raw-codeword exposures do not have constant per-batch mass")
    generator = np.random.default_rng(seed)
    tie_order = generator.permutation(len(exposures))
    tie_rank = np.empty(len(exposures), dtype=np.int64)
    tie_rank[tie_order] = np.arange(len(exposures), dtype=np.int64)
    extras = np.zeros((n_batches, len(exposures)), dtype=np.int64)
    for batch_index in range(n_batches):
        batches_left = n_batches - batch_index
        mandatory = np.flatnonzero(remaining == batches_left)
        if len(mandatory) > row_extras:
            raise RuntimeError("raw quota allocation became infeasible")
        chosen = mandatory.tolist()
        needed = row_extras - len(chosen)
        if needed:
            candidates = [
                index
                for index in range(len(exposures))
                if remaining[index] > 0 and index not in chosen
            ]
            candidates.sort(
                key=lambda index: (
                    -int(remaining[index]),
                    int((tie_rank[index] - batch_index) % len(exposures)),
                )
            )
            chosen.extend(candidates[:needed])
        if len(chosen) != row_extras:
            raise RuntimeError("raw quota allocation ran out of feasible codewords")
        extras[batch_index, chosen] = 1
        remaining[chosen] -= 1
        if np.any(remaining < 0) or np.any(remaining > batches_left - 1):
            raise RuntimeError("raw quota allocation violated residual feasibility")
    if np.any(remaining):
        raise RuntimeError("raw quota allocation left unassigned presentations")
    return base[None, :] + extras


def _component_batches(
    batch: SemanticBatch,
    mask: NDArray[np.bool_],
    *,
    repetitions: int,
    batch_size: int,
    seed: int,
    component: str,
) -> NDArray[np.int64]:
    if repetitions < 1:
        raise ValueError("component repetitions must be positive")
    if batch_size < 2 or batch_size % 2:
        raise ValueError("atomic batch_size must be a positive even integer")
    target = np.asarray(batch.target, dtype=np.int8)
    raw_ids = np.asarray(batch.latents["raw_codeword_id"], dtype=np.int64)
    half = batch_size // 2
    label_count = int(np.sum(mask & (target == -1)))
    if label_count != int(np.sum(mask & (target == 1))):
        raise RuntimeError("atomic component is not label-balanced")
    if label_count % half:
        raise ValueError("one component repetition does not fill label-balanced batches")
    batches_per_repetition = label_count // half
    n_batches = repetitions * batches_per_repetition
    result = np.empty((n_batches, batch_size), dtype=np.int64)
    for repetition in range(repetitions):
        batch_rows: list[list[int]] = [[] for _ in range(batches_per_repetition)]
        for sign in (-1, 1):
            compatible = np.asarray(
                sorted(np.unique(raw_ids[mask & (target == sign)]).tolist()),
                dtype=np.int64,
            )
            groups = [
                np.flatnonzero(mask & (target == sign) & (raw_ids == raw_id)).astype(
                    np.int64, copy=False
                )
                for raw_id in compatible
            ]
            exposures = np.asarray([len(group) for group in groups], dtype=np.int64)
            quotas = _binary_quota_matrix(
                exposures,
                n_batches=batches_per_repetition,
                per_batch_total=half,
                seed=_stable_seed(seed, component, "quotas", repetition, sign),
            )
            for raw_index, group in enumerate(groups):
                generator = np.random.default_rng(
                    _stable_seed(
                        seed,
                        component,
                        "rows",
                        repetition,
                        sign,
                        int(compatible[raw_index]),
                    )
                )
                stream = generator.permutation(group)
                cursor = 0
                for batch_index, count in enumerate(quotas[:, raw_index].tolist()):
                    selected = stream[cursor : cursor + count]
                    if len(selected) != count or len(np.unique(selected)) != count:
                        raise RuntimeError(
                            "atomic raw-codeword allocation repeated a row in-batch"
                        )
                    batch_rows[batch_index].extend(selected.tolist())
                    cursor += count
                if cursor != len(stream):
                    raise RuntimeError("one repetition did not consume every selected row")
        for local_batch, rows in enumerate(batch_rows):
            if len(rows) != batch_size or len(set(rows)) != batch_size:
                raise RuntimeError(
                    "atomic construction produced a short or duplicate-row batch"
                )
            batch_index = repetition * batches_per_repetition + local_batch
            generator = np.random.default_rng(
                _stable_seed(
                    seed, component, "batch-order", repetition, local_batch
                )
            )
            result[batch_index] = generator.permutation(
                np.asarray(rows, dtype=np.int64)
            )
    return result


def _array_digest(values: NDArray[np.int64], *, person: bytes) -> str:
    array = np.ascontiguousarray(values, dtype=np.int64)
    digest = hashlib.blake2b(digest_size=32, person=person)
    digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _row_exposure_digest(indices: NDArray[np.int64], n: int) -> str:
    counts = np.bincount(np.asarray(indices).reshape(-1), minlength=n)
    return stable_state_digest(
        {
            "row_index": np.arange(n, dtype=np.int64),
            "presentation_count": counts.astype(np.int64, copy=False),
        }
    )


def _atomic_batch_multiset_digest(
    indices: NDArray[np.int64], batch: SemanticBatch
) -> str:
    sample_ids = np.asarray(batch.sample_id, dtype=np.int64)
    batch_hashes: list[bytes] = []
    for row in np.asarray(indices, dtype=np.int64):
        payload = np.sort(sample_ids[row]).astype(np.int64, copy=False).tobytes()
        batch_hashes.append(hashlib.blake2b(payload, digest_size=32).digest())
    batch_hashes.sort()
    digest = hashlib.blake2b(digest_size=32, person=b"forkordM")
    for payload in batch_hashes:
        digest.update(payload)
    return digest.hexdigest()


@dataclass(frozen=True)
class AtomicEvidencePlan:
    schedule: OrderSchedule
    indices: NDArray[np.int64]
    prefix_steps: int
    block_steps: int
    washout_steps: int
    batch_size: int
    component_digests: dict[str, str]
    ordered_digest: str
    row_exposure_digest: str
    atomic_batch_multiset_digest: str

    @property
    def total_steps(self) -> int:
        return len(self.indices)

    @property
    def block_one_end(self) -> int:
        return self.prefix_steps + self.block_steps

    @property
    def diagnostics_end(self) -> int:
        return self.prefix_steps + 2 * self.block_steps

    def phase(self, name: str) -> NDArray[np.int64]:
        if name == "prefix":
            return self.indices[: self.prefix_steps]
        if name == "block_one":
            return self.indices[self.prefix_steps : self.block_one_end]
        if name == "block_two":
            return self.indices[self.block_one_end : self.diagnostics_end]
        if name == "diagnostics":
            return self.indices[self.prefix_steps : self.diagnostics_end]
        if name == "washout":
            return self.indices[self.diagnostics_end :]
        raise KeyError(f"unknown atomic-plan phase {name!r}")


def make_atomic_evidence_plan(
    batch: SemanticBatch,
    *,
    schedule: OrderSchedule,
    prefix_a_repetitions: int,
    washout_a_repetitions: int,
    diagnostic_repetitions: int,
    batch_size: int,
    seed: int,
) -> AtomicEvidencePlan:
    """Construct one schedule from shared raw-balanced component batches."""

    if schedule not in ORDER_SCHEDULES:
        raise ValueError(f"schedule must be one of {ORDER_SCHEDULES}")
    if prefix_a_repetitions < 1 or washout_a_repetitions < 1:
        raise ValueError("A prefix and washout repetitions must be positive")
    masks = evidence_strata(batch)
    components = {
        "A_prefix": _component_batches(
            batch,
            masks["A"],
            repetitions=prefix_a_repetitions,
            batch_size=batch_size,
            seed=seed,
            component="A_prefix",
        ),
        "A_washout": _component_batches(
            batch,
            masks["A"],
            repetitions=washout_a_repetitions,
            batch_size=batch_size,
            seed=seed,
            component="A_washout",
        ),
        "B": _component_batches(
            batch,
            masks["B"],
            repetitions=diagnostic_repetitions,
            batch_size=batch_size,
            seed=seed,
            component="B",
        ),
        "D": _component_batches(
            batch,
            masks["D"],
            repetitions=diagnostic_repetitions,
            batch_size=batch_size,
            seed=seed,
            component="D",
        ),
    }
    block_steps = len(components["B"])
    if len(components["D"]) != block_steps:
        raise RuntimeError("B and D atomic streams must contain equal batch counts")
    if schedule == "b_then_d":
        diagnostics = np.concatenate((components["B"], components["D"]), axis=0)
    elif schedule == "d_then_b":
        diagnostics = np.concatenate((components["D"], components["B"]), axis=0)
    else:
        diagnostics = np.empty((2 * block_steps, batch_size), dtype=np.int64)
        diagnostics[0::2] = components["B"]
        diagnostics[1::2] = components["D"]
    indices = np.concatenate(
        (components["A_prefix"], diagnostics, components["A_washout"]), axis=0
    )
    indices.setflags(write=False)
    sample_ids = np.asarray(batch.sample_id, dtype=np.int64)
    plan = AtomicEvidencePlan(
        schedule=schedule,
        indices=indices,
        prefix_steps=len(components["A_prefix"]),
        block_steps=block_steps,
        washout_steps=len(components["A_washout"]),
        batch_size=batch_size,
        component_digests={
            name: _array_digest(sample_ids[value], person=b"forkordC")
            for name, value in components.items()
        },
        ordered_digest=_array_digest(sample_ids[indices], person=b"forkordO"),
        row_exposure_digest=_row_exposure_digest(indices, len(batch)),
        atomic_batch_multiset_digest=_atomic_batch_multiset_digest(indices, batch),
    )
    audit_atomic_evidence_plan(
        batch,
        plan,
        prefix_a_repetitions=prefix_a_repetitions,
        washout_a_repetitions=washout_a_repetitions,
        diagnostic_repetitions=diagnostic_repetitions,
    )
    return plan


def audit_atomic_evidence_plan(
    batch: SemanticBatch,
    plan: AtomicEvidencePlan,
    *,
    prefix_a_repetitions: int,
    washout_a_repetitions: int,
    diagnostic_repetitions: int,
) -> dict[str, Any]:
    """Fail closed on exposure, phase, order, raw, and label invariants."""

    indices = np.asarray(plan.indices, dtype=np.int64)
    if indices.ndim != 2 or indices.shape[1] != plan.batch_size:
        raise RuntimeError("atomic plan must be a rectangular full-batch matrix")
    if np.any(indices < 0) or np.any(indices >= len(batch)):
        raise RuntimeError("atomic plan contains an out-of-range row index")
    target = np.asarray(batch.target, dtype=np.int8)
    if not np.all(np.sum(target[indices] > 0, axis=1) == plan.batch_size // 2):
        raise RuntimeError("every atomic minibatch must be exactly label-balanced")
    if any(len(np.unique(row)) != plan.batch_size for row in indices):
        raise RuntimeError("atomic minibatches may not repeat a selected row")
    masks = evidence_strata(batch)
    category = np.full(len(batch), "?", dtype="<U1")
    for name, mask in masks.items():
        category[mask] = name
    if not np.all(category[plan.phase("prefix")] == "A"):
        raise RuntimeError("atomic prefix contains non-A evidence")
    if not np.all(category[plan.phase("washout")] == "A"):
        raise RuntimeError("atomic washout contains non-A evidence")
    diagnostics = category[plan.phase("diagnostics")]
    if plan.schedule == "b_then_d":
        valid_order = np.all(diagnostics[: plan.block_steps] == "B") and np.all(
            diagnostics[plan.block_steps :] == "D"
        )
    elif plan.schedule == "d_then_b":
        valid_order = np.all(diagnostics[: plan.block_steps] == "D") and np.all(
            diagnostics[plan.block_steps :] == "B"
        )
    else:
        valid_order = np.all(diagnostics[0::2] == "B") and np.all(
            diagnostics[1::2] == "D"
        )
    if not valid_order:
        raise RuntimeError("atomic diagnostic schedule has the wrong category order")

    if plan.schedule == "b_then_d":
        b_stream = plan.phase("block_one")
        d_stream = plan.phase("block_two")
    elif plan.schedule == "d_then_b":
        d_stream = plan.phase("block_one")
        b_stream = plan.phase("block_two")
    else:
        b_stream = plan.phase("diagnostics")[0::2]
        d_stream = plan.phase("diagnostics")[1::2]
    component_streams = {
        "A_prefix": (plan.phase("prefix"), masks["A"], prefix_a_repetitions),
        "B": (b_stream, masks["B"], diagnostic_repetitions),
        "D": (d_stream, masks["D"], diagnostic_repetitions),
        "A_washout": (
            plan.phase("washout"),
            masks["A"],
            washout_a_repetitions,
        ),
    }
    repetition_batch_counts: dict[str, int] = {}
    for component, (stream, mask, repetitions) in component_streams.items():
        rows_per_repetition = int(np.sum(mask))
        if rows_per_repetition % plan.batch_size:
            raise RuntimeError(f"{component} rows do not form whole repetition chunks")
        batches_per_repetition = rows_per_repetition // plan.batch_size
        if len(stream) != repetitions * batches_per_repetition:
            raise RuntimeError(f"{component} stream has the wrong repetition length")
        expected_one = mask.astype(np.int64)
        for repetition in range(repetitions):
            start = repetition * batches_per_repetition
            stop = start + batches_per_repetition
            chunk_counts = np.bincount(
                stream[start:stop].reshape(-1), minlength=len(batch)
            )
            if not np.array_equal(chunk_counts, expected_one):
                raise RuntimeError(
                    f"{component} repetition {repetition} does not present every "
                    "selected row exactly once"
                )
        repetition_batch_counts[component] = batches_per_repetition

    expected = np.empty(len(batch), dtype=np.int64)
    expected[masks["A"]] = prefix_a_repetitions + washout_a_repetitions
    expected[masks["B"]] = diagnostic_repetitions
    expected[masks["D"]] = diagnostic_repetitions
    counts = np.bincount(indices.reshape(-1), minlength=len(batch))
    if not np.array_equal(counts, expected):
        raise RuntimeError("atomic schedule does not realize exact per-row exposure")

    raw_ids = np.asarray(batch.latents["raw_codeword_id"], dtype=np.int64)
    expected_raw = {
        name: set(np.unique(raw_ids[mask]).tolist()) for name, mask in masks.items()
    }
    raw_minimum = plan.batch_size
    raw_maximum = 0
    for row in indices:
        names = np.unique(category[row])
        if len(names) != 1:
            raise RuntimeError("one atomic batch mixed evidence categories")
        name = str(names[0])
        values, raw_counts = np.unique(raw_ids[row], return_counts=True)
        if set(values.tolist()) != expected_raw[name]:
            raise RuntimeError("atomic batch omitted a compatible raw codeword")
        floor = plan.batch_size // len(values)
        ceiling = floor + int(plan.batch_size % len(values) != 0)
        if np.any((raw_counts < floor) | (raw_counts > ceiling)):
            raise RuntimeError("atomic raw-codeword counts are not floor/ceiling balanced")
        raw_minimum = min(raw_minimum, int(np.min(raw_counts)))
        raw_maximum = max(raw_maximum, int(np.max(raw_counts)))
    sample_ids = np.asarray(batch.sample_id, dtype=np.int64)[indices]
    return {
        "schedule": plan.schedule,
        "n_rows": len(batch),
        "total_steps": plan.total_steps,
        "batch_size": plan.batch_size,
        "prefix_steps": plan.prefix_steps,
        "block_steps": plan.block_steps,
        "diagnostics_end": plan.diagnostics_end,
        "washout_steps": plan.washout_steps,
        "total_presentations": int(indices.size),
        "per_row_presentation_min": int(np.min(counts)),
        "per_row_presentation_max": int(np.max(counts)),
        "per_batch_raw_count_min": raw_minimum,
        "per_batch_raw_count_max": raw_maximum,
        "all_batches_label_balanced": True,
        "all_batches_raw_balanced": True,
        "all_batches_full": True,
        "all_rows_unique_within_batch": True,
        "every_row_once_per_registered_repetition": True,
        "batches_per_component_repetition": repetition_batch_counts,
        "phase_categories_verified": True,
        "component_digests": dict(plan.component_digests),
        "ordered_index_digest": plan.ordered_digest,
        "ordered_sample_id_digest": _array_digest(sample_ids, person=b"forkordS"),
        "row_exposure_digest": plan.row_exposure_digest,
        "atomic_batch_multiset_digest": plan.atomic_batch_multiset_digest,
    }


__all__ = [
    "ORDER_SCHEDULES",
    "AtomicEvidencePlan",
    "OrderSchedule",
    "audit_atomic_evidence_plan",
    "audit_evidence_strata",
    "evidence_strata",
    "make_atomic_evidence_plan",
    "make_identical_evidence_dataset",
]
