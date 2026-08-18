"""Measurements for temporal dynamics with several competing goal rules.

The helpers in this module are deliberately independent of training protocols.
They provide three complementary views of a saved model state:

* deterministic affine ridge probes of raw and hidden representations;
* an exhaustive Boolean description of behavior over the ``(P, Q, Y)`` cube;
* signed causal effects of flipping each candidate rule's input channels.

The probe controls matter.  A shared raw-codeword truth table checks whether a
layer resolves generic codeword structure without privileging one named rule,
while independently sample-permuted labels provide a finite-sample negative control.
Neither control is evidence that a decoded variable is used by the policy;
usage is measured separately by the paired causal interventions.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping
from dataclasses import dataclass
from itertools import product
from typing import Any, TypeAlias

import numpy as np
import torch
from numpy.typing import ArrayLike, NDArray

from .competing import decode_competing_rules
from .data import SemanticBatch
from .models import GoalMLP

FloatArray: TypeAlias = NDArray[np.float64]
SignArray: TypeAlias = NDArray[np.int8]

# The order is part of the public signature definition.  Interpreting a bit
# string therefore never depends on dict insertion order or a plotting script.
FACTORIAL_TUPLES: tuple[tuple[int, int, int], ...] = tuple(
    (values[0], values[1], values[2]) for values in product((-1, 1), repeat=3)
)


def _stable_seed(seed: int, *parts: object) -> int:
    digest = hashlib.blake2b(digest_size=8, person=b"forkdyn")
    for part in (seed, *parts):
        payload = str(part).encode("utf-8")
        digest.update(len(payload).to_bytes(4, "little"))
        digest.update(payload)
    return int.from_bytes(digest.digest(), "little")


def _as_finite_matrix(value: ArrayLike, name: str) -> FloatArray:
    result = np.asarray(value, dtype=np.float64)
    if result.ndim == 1:
        result = result[:, None]
    if result.ndim != 2 or result.shape[0] < 1 or result.shape[1] < 1:
        raise ValueError(f"{name} must have non-empty shape [n, d], got {result.shape}")
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must contain only finite values")
    return np.array(result, copy=True)


def _as_binary_targets(value: ArrayLike, name: str, n: int) -> SignArray:
    array = np.asarray(value)
    if array.ndim == 1:
        array = array[:, None]
    if array.ndim != 2 or array.shape[0] != n or array.shape[1] < 1:
        raise ValueError(f"{name} must have shape [{n}, labels], got {array.shape}")
    if np.all(np.isin(array, (-1, 1))):
        return array.astype(np.int8, copy=True)
    if np.all(np.isin(array, (0, 1))):
        return np.where(array > 0, 1, -1).astype(np.int8)
    raise ValueError(f"{name} must contain binary signs or zero-based binary labels")


def extract_goal_mlp_representations(
    model: GoalMLP,
    data: SemanticBatch | ArrayLike,
    *,
    max_k: int | None = None,
    include_state: bool = True,
) -> dict[str, FloatArray]:
    """Extract raw, first-hidden, and final-hidden representations.

    This function neither calls ``eval``/``train`` nor samples randomness.  The
    caller's model mode and RNG stream are therefore unchanged.  ``depth=0``
    models have no hidden transformation, so both hidden keys intentionally
    alias the values of the raw representation (as independent arrays).

    When ``data`` is a :class:`SemanticBatch`, its stable feature interface is
    used.  An explicit array is useful when the protocol already constructed
    the exact model input through its configuration layer.
    """

    if not isinstance(model, GoalMLP):
        raise TypeError("representation extraction requires a GoalMLP")
    raw = (
        data.features(max_k=max_k, include_state=include_state)
        if isinstance(data, SemanticBatch)
        else np.asarray(data)
    )
    raw_matrix = _as_finite_matrix(raw, "data")
    if raw_matrix.shape[1] != model.config.input_dim:
        raise ValueError(
            f"model expects {model.config.input_dim} features, received {raw_matrix.shape[1]}"
        )

    parameter = next(model.parameters())
    tensor = torch.as_tensor(raw_matrix, dtype=parameter.dtype, device=parameter.device)
    with torch.no_grad():
        if model.config.depth == 0:
            first = tensor
            final = tensor
        else:
            first = model.activation(model.input_projection(tensor))
            final = first
            for layer in model.hidden_layers:
                transformed = model.activation(layer(final))
                final = (final + transformed) * (2.0**-0.5) if model.residual else transformed

    return {
        "raw": raw_matrix,
        "first_hidden": first.detach().cpu().numpy().astype(np.float64, copy=True),
        "final_hidden": final.detach().cpu().numpy().astype(np.float64, copy=True),
    }


@dataclass(frozen=True)
class RidgeProbeResult:
    """A fitted multi-label affine ridge probe and its held-out evaluation."""

    label_names: tuple[str, ...]
    feature_mean: FloatArray
    feature_scale: FloatArray
    coefficients: FloatArray
    intercept: FloatArray
    train_scores: FloatArray
    heldout_scores: FloatArray
    train_accuracy: FloatArray
    heldout_accuracy: FloatArray
    alpha: float

    def summary(self) -> dict[str, Any]:
        """Return the compact, JSON-safe portion used in metric records."""

        return {
            "alpha": float(self.alpha),
            "dimension": int(self.coefficients.shape[0]),
            "n_labels": len(self.label_names),
            "n_train": int(self.train_scores.shape[0]),
            "n_heldout": int(self.heldout_scores.shape[0]),
            "train_accuracy": {
                name: float(self.train_accuracy[index])
                for index, name in enumerate(self.label_names)
            },
            "heldout_accuracy": {
                name: float(self.heldout_accuracy[index])
                for index, name in enumerate(self.label_names)
            },
        }


def fit_affine_ridge_probe(
    x_train: ArrayLike,
    y_train: ArrayLike,
    x_heldout: ArrayLike,
    y_heldout: ArrayLike,
    *,
    alpha: float = 1e-3,
    label_names: tuple[str, ...] | None = None,
) -> RidgeProbeResult:
    """Fit all binary labels in one deterministic affine ridge solve.

    Features are standardized using *only* training statistics.  The fitted
    objective is mean squared error plus ``alpha * ||W||^2``; the intercept is
    not penalized.  Jointly solving all label columns gives exactly the same
    answer as separate solves while avoiding label-order-dependent state.
    """

    train = _as_finite_matrix(x_train, "x_train")
    heldout = _as_finite_matrix(x_heldout, "x_heldout")
    if train.shape[1] != heldout.shape[1]:
        raise ValueError("training and held-out representations must have equal width")
    train_targets = _as_binary_targets(y_train, "y_train", len(train))
    heldout_targets = _as_binary_targets(y_heldout, "y_heldout", len(heldout))
    if train_targets.shape[1] != heldout_targets.shape[1]:
        raise ValueError("training and held-out targets must have equal label counts")
    if not math.isfinite(float(alpha)) or float(alpha) < 0.0:
        raise ValueError("alpha must be finite and non-negative")
    alpha = float(alpha)

    n_labels = train_targets.shape[1]
    if label_names is None:
        label_names = tuple(f"label_{index}" for index in range(n_labels))
    if len(label_names) != n_labels or len(set(label_names)) != n_labels:
        raise ValueError("label_names must be unique and match the target width")

    feature_mean = np.mean(train, axis=0)
    feature_scale = np.std(train, axis=0, ddof=0)
    # Constant columns carry no centered information.  Scaling them by one is
    # both numerically safe and makes the stored transformation unambiguous.
    tolerance = np.finfo(np.float64).eps * max(1.0, float(np.max(np.abs(feature_mean))))
    feature_scale = np.where(feature_scale <= tolerance, 1.0, feature_scale)
    z_train = (train - feature_mean) / feature_scale
    z_heldout = (heldout - feature_mean) / feature_scale

    target_mean = np.mean(train_targets.astype(np.float64), axis=0)
    centered_targets = train_targets.astype(np.float64) - target_mean
    gram = (z_train.T @ z_train) / len(z_train)
    rhs = (z_train.T @ centered_targets) / len(z_train)
    regularized = gram + alpha * np.eye(gram.shape[0], dtype=np.float64)
    try:
        coefficients = np.linalg.solve(regularized, rhs)
    except np.linalg.LinAlgError:
        # ``alpha=0`` is useful for exact controls and can expose a singular
        # design.  The minimum-norm least-squares solution remains deterministic.
        coefficients = np.linalg.lstsq(regularized, rhs, rcond=None)[0]

    train_scores = z_train @ coefficients + target_mean
    heldout_scores = z_heldout @ coefficients + target_mean
    train_predictions = np.where(train_scores >= 0.0, 1, -1)
    heldout_predictions = np.where(heldout_scores >= 0.0, 1, -1)
    train_accuracy = np.mean(train_predictions == train_targets, axis=0)
    heldout_accuracy = np.mean(heldout_predictions == heldout_targets, axis=0)
    return RidgeProbeResult(
        label_names=label_names,
        feature_mean=np.array(feature_mean, copy=True),
        feature_scale=np.array(feature_scale, copy=True),
        coefficients=np.array(coefficients, copy=True),
        intercept=np.array(target_mean, copy=True),
        train_scores=np.array(train_scores, copy=True),
        heldout_scores=np.array(heldout_scores, copy=True),
        train_accuracy=np.asarray(train_accuracy, dtype=np.float64),
        heldout_accuracy=np.asarray(heldout_accuracy, dtype=np.float64),
        alpha=alpha,
    )


def _tuple_indices(batch: SemanticBatch) -> NDArray[np.int64]:
    rules = decode_competing_rules(batch)
    p = (rules["P"] > 0).astype(np.int64)
    q = (rules["Q"] > 0).astype(np.int64)
    y = (rules["Y_code"] > 0).astype(np.int64)
    return 4 * p + 2 * q + y


def _nontrivial_permutation(n: int, seed: int, *parts: object) -> NDArray[np.int64]:
    permutation = np.random.default_rng(_stable_seed(seed, *parts)).permutation(n)
    if n > 1 and np.array_equal(permutation, np.arange(n)):
        permutation = np.roll(permutation, 1)
    return permutation.astype(np.int64, copy=False)


@dataclass(frozen=True)
class CompetingProbeTargets:
    """Matched train/held-out labels for genuine and control probe tasks."""

    label_names: tuple[str, ...]
    train: SignArray
    heldout: SignArray
    raw_codeword_ids: tuple[int, ...]
    truth_table_control: tuple[int, ...]

    def summary(self) -> dict[str, Any]:
        mapping_payload = np.column_stack(
            (
                np.asarray(self.raw_codeword_ids, dtype=np.int64),
                np.asarray(self.truth_table_control, dtype=np.int64),
            )
        ).tobytes()
        return {
            "label_names": list(self.label_names),
            # Keep the full verified mapping on this dataclass, but do not copy
            # hundreds of static truth-table entries into every checkpoint's
            # long-form metric records.
            "raw_codeword_count": len(self.raw_codeword_ids),
            "truth_table_control_positive_fraction": float(
                np.mean(np.asarray(self.truth_table_control) > 0)
            ),
            "truth_table_control_mapping_digest": hashlib.blake2b(
                mapping_payload, digest_size=16, person=b"forkprobe"
            ).hexdigest(),
            "truth_table_control_mapping_verified": True,
            "n_train": int(self.train.shape[0]),
            "n_heldout": int(self.heldout.shape[0]),
            "n_labels": int(self.train.shape[1]),
        }


def make_competing_probe_targets(
    train_batch: SemanticBatch,
    heldout_batch: SemanticBatch,
    *,
    seed: int = 0,
    include_permuted: bool = True,
) -> CompetingProbeTargets:
    """Build genuine candidate labels and two kinds of probe controls.

    ``truth_table_control`` is supplied by the exhaustive factorial generator.
    It is a codeword-stable Boolean function balanced within every complete
    ``(P,Q,Y)`` tuple.  This helper verifies that train and held-out splits use
    the same raw-codeword-to-label mapping before fitting a probe.  The
    sample-permuted controls preserve every label's marginal balance but use
    independent row permutations in the two splits.
    """

    train_rules = decode_competing_rules(train_batch)
    heldout_rules = decode_competing_rules(heldout_batch)
    required_latents = {"raw_codeword_id", "truth_table_control"}
    for split, batch in (("train", train_batch), ("heldout", heldout_batch)):
        missing = required_latents - set(batch.latents)
        if missing:
            raise KeyError(
                f"{split} probe batch must come from the exhaustive factorial generator; "
                f"missing latents {sorted(missing)}"
            )

    def control_mapping(batch: SemanticBatch, split: str) -> tuple[dict[int, int], SignArray]:
        raw_ids = np.asarray(batch.latents["raw_codeword_id"], dtype=np.int64)
        control = np.asarray(batch.latents["truth_table_control"], dtype=np.int8)
        if not np.all(np.isin(control, (-1, 1))):
            raise ValueError(f"{split} truth_table_control must contain only signs")
        mapping: dict[int, int] = {}
        for raw_id in np.unique(raw_ids):
            labels = np.unique(control[raw_ids == raw_id])
            if len(labels) != 1:
                raise ValueError(
                    f"{split} truth-table control is not stable for raw codeword {raw_id}"
                )
            mapping[int(raw_id)] = int(labels[0])
        return mapping, control

    train_mapping, train_control = control_mapping(train_batch, "train")
    heldout_mapping, heldout_control = control_mapping(heldout_batch, "heldout")
    if train_mapping != heldout_mapping:
        raise ValueError(
            "train and held-out factorial batches do not share the same "
            "raw-codeword truth-table control"
        )
    raw_codeword_ids = tuple(sorted(train_mapping))
    control_table = tuple(train_mapping[raw_id] for raw_id in raw_codeword_ids)

    genuine_names = ("P", "Q", "Y", "truth_table_control")
    train_genuine = np.column_stack(
        (
            train_rules["P"],
            train_rules["Q"],
            train_rules["Y_code"],
            train_control,
        )
    ).astype(np.int8, copy=False)
    heldout_genuine = np.column_stack(
        (
            heldout_rules["P"],
            heldout_rules["Q"],
            heldout_rules["Y_code"],
            heldout_control,
        )
    ).astype(np.int8, copy=False)

    if include_permuted:
        train_permuted = np.empty_like(train_genuine)
        heldout_permuted = np.empty_like(heldout_genuine)
        for index, name in enumerate(genuine_names):
            train_permuted[:, index] = train_genuine[
                _nontrivial_permutation(len(train_batch), seed, "train", name), index
            ]
            heldout_permuted[:, index] = heldout_genuine[
                _nontrivial_permutation(len(heldout_batch), seed, "heldout", name), index
            ]
        label_names = genuine_names + tuple(f"{name}_permuted" for name in genuine_names)
        train_targets = np.column_stack((train_genuine, train_permuted))
        heldout_targets = np.column_stack((heldout_genuine, heldout_permuted))
    else:
        label_names = genuine_names
        train_targets = train_genuine
        heldout_targets = heldout_genuine

    return CompetingProbeTargets(
        label_names=label_names,
        train=np.asarray(train_targets, dtype=np.int8),
        heldout=np.asarray(heldout_targets, dtype=np.int8),
        raw_codeword_ids=raw_codeword_ids,
        truth_table_control=control_table,
    )


def evaluate_competing_probes(
    model: GoalMLP,
    train_batch: SemanticBatch,
    heldout_batch: SemanticBatch,
    *,
    seed: int = 0,
    alpha: float = 1e-3,
    max_k: int | None = None,
    include_state: bool = True,
    include_permuted: bool = True,
) -> dict[str, Any]:
    """Evaluate a fixed multi-label probe on each representation stage.

    The return value contains only JSON-safe summaries and is intentionally easy
    to flatten: loop over ``result["representations"]`` and then over either
    named accuracy mapping.
    """

    targets = make_competing_probe_targets(
        train_batch,
        heldout_batch,
        seed=seed,
        include_permuted=include_permuted,
    )
    train_representations = extract_goal_mlp_representations(
        model, train_batch, max_k=max_k, include_state=include_state
    )
    heldout_representations = extract_goal_mlp_representations(
        model, heldout_batch, max_k=max_k, include_state=include_state
    )
    summaries: dict[str, dict[str, Any]] = {}
    for name in ("raw", "first_hidden", "final_hidden"):
        fitted = fit_affine_ridge_probe(
            train_representations[name],
            targets.train,
            heldout_representations[name],
            targets.heldout,
            alpha=alpha,
            label_names=targets.label_names,
        )
        summaries[name] = fitted.summary()
    return {
        "probe_kind": "deterministic_affine_ridge",
        "standardization": "train_only",
        **targets.summary(),
        "representations": summaries,
    }


def _binary_logit_view(logits: ArrayLike, n: int, name: str) -> tuple[SignArray, FloatArray]:
    values = np.asarray(logits, dtype=np.float64)
    if values.ndim == 1 and values.shape == (n,):
        margin = values
    elif values.ndim == 2 and values.shape == (n, 2):
        margin = values[:, 1] - values[:, 0]
    else:
        raise ValueError(f"{name} must have shape [{n}] or [{n}, 2], got {values.shape}")
    if not np.all(np.isfinite(margin)):
        raise ValueError(f"{name} must contain only finite logits")
    # Stable sigmoid of the positive-minus-negative margin.
    probability = np.empty_like(margin)
    positive = margin >= 0.0
    probability[positive] = 1.0 / (1.0 + np.exp(-margin[positive]))
    exponential = np.exp(margin[~positive])
    probability[~positive] = exponential / (1.0 + exponential)
    actions = np.where(margin >= 0.0, 1, -1).astype(np.int8)
    return actions, probability.astype(np.float64, copy=False)


def _tuple_key(values: tuple[int, int, int]) -> str:
    return "".join("+" if value > 0 else "-" for value in values)


def _modal_sign(values: SignArray) -> int:
    # The deployed GoalMLP also chooses +1 at a zero logit, so ties use +1.
    return 1 if float(np.mean(values > 0)) >= 0.5 else -1


def factorial_behavioral_structure(logits: ArrayLike, batch: SemanticBatch) -> dict[str, Any]:
    """Describe policy behavior on an exhaustive ``P x Q x Y`` factorial batch.

    ``tuple_consistency`` is row agreement with the modal action for each of the
    eight decoded candidate tuples.  ``codeword_consistency`` first gives every
    active parity codeword equal weight and asks whether its modal action agrees
    with the tuple-level action.  ``nuisance_consistency`` asks whether rows with
    the *same* active codeword retain one action as padding/state features vary.
    The accompanying repeated-group counts show whether that last estimand is
    identified rather than being a collection of singleton groups.
    """

    rules = decode_competing_rules(batch)
    actions, probabilities = _binary_logit_view(logits, len(batch), "logits")
    tuple_indices = _tuple_indices(batch)
    panels: list[dict[str, Any]] = []
    tuple_modes: dict[int, int] = {}
    for index, candidate_tuple in enumerate(FACTORIAL_TUPLES):
        selected = tuple_indices == index
        count = int(np.sum(selected))
        if count == 0:
            raise ValueError(
                "factorial behavioral structure requires all eight candidate tuples; "
                f"missing {_tuple_key(candidate_tuple)!r}"
            )
        modal = _modal_sign(actions[selected])
        tuple_modes[index] = modal
        panels.append(
            {
                "key": _tuple_key(candidate_tuple),
                "P": int(candidate_tuple[0]),
                "Q": int(candidate_tuple[1]),
                "Y": int(candidate_tuple[2]),
                "n": count,
                "positive_rate": float(np.mean(actions[selected] > 0)),
                "mean_positive_probability": float(np.mean(probabilities[selected])),
                "modal_action": modal,
            }
        )

    row_tuple_modes = np.asarray([tuple_modes[int(index)] for index in tuple_indices])
    tuple_consistency = float(np.mean(actions == row_tuple_modes))

    k_q = int(batch.metadata["k_q"])
    k_y = int(batch.metadata["k_y"])
    active_names = (
        "P",
        *(f"Q_{index}" for index in range(1, k_q + 1)),
        *(f"R_{index}" for index in range(1, k_y + 1)),
    )
    active_code = np.column_stack(
        [np.asarray(batch.channels[name], dtype=np.int8) for name in active_names]
    )
    _, inverse, counts = np.unique(active_code, axis=0, return_inverse=True, return_counts=True)
    code_modes: SignArray = np.empty(len(counts), dtype=np.int8)
    code_tuple_indices: NDArray[np.int64] = np.empty(len(counts), dtype=np.int64)
    for code_index in range(len(counts)):
        selected = inverse == code_index
        code_modes[code_index] = _modal_sign(actions[selected])
        unique_tuples = np.unique(tuple_indices[selected])
        if len(unique_tuples) != 1:  # Algebraic invariant guard.
            raise RuntimeError("one active codeword decoded to multiple candidate tuples")
        code_tuple_indices[code_index] = unique_tuples[0]

    code_matches_tuple = np.asarray(
        [code_modes[index] == tuple_modes[int(code_tuple_indices[index])] for index in range(len(counts))]
    )
    row_code_modes = code_modes[inverse]
    nuisance_consistency = float(np.mean(actions == row_code_modes))
    repeated = counts > 1

    signature = "".join("1" if tuple_modes[index] > 0 else "0" for index in range(8))
    complement_pairs = tuple((index, 7 - index) for index in range(4))
    symmetry = float(
        np.mean(
            [tuple_modes[left] == -tuple_modes[right] for left, right in complement_pairs]
        )
    )
    return {
        "n": len(batch),
        "rho_p": float(np.mean(actions == rules["P"])),
        "rho_q": float(np.mean(actions == rules["Q"])),
        "rho_y": float(np.mean(actions == rules["Y_code"])),
        "target_accuracy": float(np.mean(actions == rules["target"])),
        "mean_positive_probability": float(np.mean(probabilities)),
        "factorial_tuple_order": [_tuple_key(values) for values in FACTORIAL_TUPLES],
        "tuples": panels,
        "tuple_consistency": tuple_consistency,
        "codeword_consistency": float(np.mean(code_matches_tuple)),
        "codeword_consistency_row_weighted": float(
            np.average(code_matches_tuple.astype(np.float64), weights=counts)
        ),
        "nuisance_consistency": nuisance_consistency,
        "n_codeword_groups": len(counts),
        "n_repeated_codeword_groups": int(np.sum(repeated)),
        "n_rows_in_repeated_codeword_groups": int(np.sum(counts[repeated])),
        "nuisance_consistency_identified": bool(np.any(repeated)),
        "boolean_signature": signature,
        "boolean_signature_int": int(signature, 2),
        "sign_inversion_symmetry": symmetry,
        "is_sign_inversion_symmetric": bool(symmetry == 1.0),
    }


def _directional_effect(
    base_actions: SignArray,
    base_probability: FloatArray,
    changed_actions: SignArray,
    changed_probability: FloatArray,
    goal: SignArray,
) -> dict[str, float | int]:
    hard_change = (base_actions.astype(np.float64) - changed_actions) / 2.0
    probability_change = base_probability - changed_probability
    hard_abs = float(np.mean(np.abs(hard_change)))
    probability_abs = float(np.mean(np.abs(probability_change)))
    c_hard = float(np.mean(goal * hard_change))
    c_prob = float(np.mean(goal * probability_change))
    return {
        "c_hard": c_hard,
        "c_prob": c_prob,
        "causal_score": 0.5 * (1.0 + c_hard),
        "causal_prob_score": 0.5 * (1.0 + c_prob),
        "hard_abs_change": hard_abs,
        "prob_abs_change": probability_abs,
        "hard_directional_purity": c_hard / hard_abs if hard_abs > 0.0 else 0.0,
        "prob_directional_purity": c_prob / probability_abs if probability_abs > 0.0 else 0.0,
        "n": len(goal),
    }


def directional_causal_effects(
    base_logits: ArrayLike,
    flipped_logits: Mapping[str, ArrayLike],
    batch: SemanticBatch,
) -> dict[str, Any]:
    """Compute candidate-aligned causal effects for P/Q/Y channel flips.

    For candidate recommendation ``g`` and hard actions ``a``/``a_flip``, the
    signed score is ``E[g * (a-a_flip)/2]``.  The probability analogue is
    ``E[g * (p_+ - p_+^flip)]``.  A score of +1 is perfectly candidate-aligned,
    -1 is perfectly anti-aligned, and zero can mean either no effect or balanced
    opposing effects; the absolute-change fields disambiguate those cases.
    """

    if not flipped_logits:
        raise ValueError("flipped_logits cannot be empty")
    rules = decode_competing_rules(batch)
    base_actions, base_probability = _binary_logit_view(base_logits, len(batch), "base_logits")
    k_q = int(batch.metadata["k_q"])
    k_y = int(batch.metadata["k_y"])
    expected: dict[str, tuple[str, SignArray]] = {
        "flip_P": ("P", rules["P"]),
        **{
            f"flip_Q_{index}": ("Q", rules["Q"])
            for index in range(1, k_q + 1)
        },
        **{
            f"flip_Y_{index}": ("Y", rules["Y_code"])
            for index in range(1, k_y + 1)
        },
    }
    unknown = set(flipped_logits) - set(expected)
    if unknown:
        raise KeyError(f"unknown competing-goal interventions: {sorted(unknown)}")

    per_intervention: dict[str, dict[str, float | int | str]] = {}
    for name, changed_logits in flipped_logits.items():
        family, goal = expected[name]
        changed_actions, changed_probability = _binary_logit_view(
            changed_logits, len(batch), name
        )
        per_intervention[name] = {
            "family": family,
            **_directional_effect(
                base_actions,
                base_probability,
                changed_actions,
                changed_probability,
                goal,
            ),
        }

    family_means: dict[str, dict[str, float | int]] = {}
    metric_names = ("c_hard", "c_prob", "hard_abs_change", "prob_abs_change")
    for family in ("P", "Q", "Y"):
        members = [
            values for values in per_intervention.values() if values["family"] == family
        ]
        if not members:
            continue
        means = {
            metric: float(np.mean([float(member[metric]) for member in members]))
            for metric in metric_names
        }
        hard_abs = means["hard_abs_change"]
        probability_abs = means["prob_abs_change"]
        family_means[family] = {
            **means,
            "causal_score": 0.5 * (1.0 + means["c_hard"]),
            "causal_prob_score": 0.5 * (1.0 + means["c_prob"]),
            "hard_directional_purity": (
                means["c_hard"] / hard_abs if hard_abs > 0.0 else 0.0
            ),
            "prob_directional_purity": (
                means["c_prob"] / probability_abs if probability_abs > 0.0 else 0.0
            ),
            "n_flips": len(members),
            "n": len(batch),
        }

    hard_abs_total = sum(float(values["hard_abs_change"]) for values in family_means.values())
    probability_abs_total = sum(
        float(values["prob_abs_change"]) for values in family_means.values()
    )
    hard_directional_total = sum(abs(float(values["c_hard"])) for values in family_means.values())
    probability_directional_total = sum(
        abs(float(values["c_prob"])) for values in family_means.values()
    )
    for values in family_means.values():
        values["hard_abs_share"] = (
            float(values["hard_abs_change"]) / hard_abs_total if hard_abs_total > 0.0 else 0.0
        )
        values["prob_abs_share"] = (
            float(values["prob_abs_change"]) / probability_abs_total
            if probability_abs_total > 0.0
            else 0.0
        )
        values["c_hard_signed_l1_allocation"] = (
            float(values["c_hard"]) / hard_directional_total
            if hard_directional_total > 0.0
            else 0.0
        )
        values["c_prob_signed_l1_allocation"] = (
            float(values["c_prob"]) / probability_directional_total
            if probability_directional_total > 0.0
            else 0.0
        )
        values["c_hard_abs_l1_share"] = (
            abs(float(values["c_hard"])) / hard_directional_total
            if hard_directional_total > 0.0
            else 0.0
        )
        values["c_prob_abs_l1_share"] = (
            abs(float(values["c_prob"])) / probability_directional_total
            if probability_directional_total > 0.0
            else 0.0
        )

    return {
        "n": len(batch),
        "per_intervention": per_intervention,
        "family_means": family_means,
        "normalization": {
            "hard_abs_total": hard_abs_total,
            "prob_abs_total": probability_abs_total,
            "hard_directional_l1_total": hard_directional_total,
            "prob_directional_l1_total": probability_directional_total,
        },
    }


__all__ = [
    "FACTORIAL_TUPLES",
    "CompetingProbeTargets",
    "RidgeProbeResult",
    "directional_causal_effects",
    "evaluate_competing_probes",
    "extract_goal_mlp_representations",
    "factorial_behavioral_structure",
    "fit_affine_ridge_probe",
    "make_competing_probe_targets",
]
