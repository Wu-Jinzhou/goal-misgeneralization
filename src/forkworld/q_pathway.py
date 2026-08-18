"""Exact first-layer pathway edits for the E19 active-pathway intervention.

The intervention targets the two active encoded-proxy inputs of E17's fixed
``GoalMLP``.  Necessity restores their first-layer columns to initialization;
sufficiency transplants the paired independent-history columns into the nested
history.  Each edit has a sham that applies the exact same displacement tensor
to two model-visible padding columns.  This module constructs and audits those
edits but performs no training or outcome analysis.
"""

from __future__ import annotations

import copy
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch import Tensor

from .data import SemanticBatch
from .handoff import semantic_batch_digest, stable_state_digest
from .models import GoalMLP

EXPECTED_FEATURE_NAMES: tuple[str, ...] = (
    "P",
    "P_present",
    "R_1",
    "R_2",
    "R_3",
    "R_4",
    "R_5",
    "Q_present",
    "Q_1",
    "Q_2",
    "Q_3",
    "state_0",
    "state_1",
    "state_2",
    "state_3",
    "state_4",
    "state_5",
    "state_6",
    "state_7",
)
Q_COLUMN_INDICES: tuple[int, int] = (8, 9)
PADDING_SHAM_COLUMN_INDICES: tuple[int, int] = (5, 6)
EXPECTED_STATE_SHAPES: Mapping[str, tuple[int, ...]] = {
    "input_projection.weight": (64, 19),
    "input_projection.bias": (64,),
    "hidden_layers.0.weight": (64, 64),
    "hidden_layers.0.bias": (64,),
    "goal_head.weight": (1, 64),
    "goal_head.bias": (1,),
}
Q_PATHWAY_BRANCHES: tuple[str, ...] = (
    "independent_noop",
    "independent_q_restore",
    "independent_padding_sham",
    "nested_noop",
    "nested_q_transplant",
    "nested_padding_sham",
)

_BRANCH_BASELINES: Mapping[str, str] = {
    "independent_noop": "independent",
    "independent_q_restore": "independent",
    "independent_padding_sham": "independent",
    "nested_noop": "nested",
    "nested_q_transplant": "nested",
    "nested_padding_sham": "nested",
}


@dataclass(frozen=True)
class QPathwayBranches:
    """Six independent model clones plus immutable construction evidence."""

    models: dict[str, GoalMLP]
    donor_state_digests: dict[str, str]
    construction_state_digests: dict[str, str]
    restore_delta: Tensor
    transplant_delta: Tensor
    feature_names: tuple[str, ...]
    construction_digest: str


def _cpu_clone(value: Tensor) -> Tensor:
    return value.detach().cpu().clone().contiguous()


def _state_shapes(model: GoalMLP) -> dict[str, tuple[int, ...]]:
    return {name: tuple(value.shape) for name, value in model.state_dict().items()}


def _validate_model(model: GoalMLP, name: str) -> None:
    if not isinstance(model, GoalMLP):
        raise TypeError(f"{name} must be an unwrapped GoalMLP")
    config = model.config
    if (
        config.input_dim != 19
        or config.width != 64
        or config.depth != 2
        or config.activation != "relu"
        or config.residual
        or not config.bias
    ):
        raise ValueError(
            f"{name} must use E17's width-64, depth-2, non-residual ReLU architecture"
        )
    actual_shapes = _state_shapes(model)
    if actual_shapes != dict(EXPECTED_STATE_SHAPES):
        raise ValueError(
            f"{name} state shapes differ from the frozen E17 contract: {actual_shapes}"
        )
    devices = {parameter.device.type for parameter in model.parameters()}
    if devices != {"cpu"}:
        raise ValueError(f"{name} must be on CPU for exact pathway edits")
    dtypes = {parameter.dtype for parameter in model.parameters()}
    if len(dtypes) != 1 or next(iter(dtypes)) not in (torch.float32, torch.float64):
        raise ValueError(f"{name} must have one floating parameter dtype")


def audit_q_pathway_contract(
    models: Mapping[str, GoalMLP], feature_names: Sequence[str]
) -> dict[str, Any]:
    """Fail closed on the feature interface and exact E17 model architecture."""

    names = tuple(str(name) for name in feature_names)
    if names != EXPECTED_FEATURE_NAMES:
        raise ValueError(
            "E19 feature order differs from the frozen E17 interface; "
            f"received {names}"
        )
    if not models:
        raise ValueError("E19 contract audit requires at least one model")
    for name, model in models.items():
        _validate_model(model, str(name))
    dtypes = {next(model.parameters()).dtype for model in models.values()}
    if len(dtypes) != 1:
        raise ValueError("all E19 donor models must have the same parameter dtype")
    return {
        "feature_names": list(names),
        "feature_names_digest": stable_state_digest(names),
        "q_column_indices": list(Q_COLUMN_INDICES),
        "q_column_names": [names[index] for index in Q_COLUMN_INDICES],
        "padding_sham_column_indices": list(PADDING_SHAM_COLUMN_INDICES),
        "padding_sham_column_names": [
            names[index] for index in PADDING_SHAM_COLUMN_INDICES
        ],
        "state_shapes": {
            name: list(shape) for name, shape in EXPECTED_STATE_SHAPES.items()
        },
        "parameter_dtype": str(next(iter(dtypes))),
        "cpu_only": True,
        "contract_verified": True,
    }


def _columns() -> tuple[Tensor, Tensor]:
    return (
        torch.tensor(Q_COLUMN_INDICES, dtype=torch.long),
        torch.tensor(PADDING_SHAM_COLUMN_INDICES, dtype=torch.long),
    )


def _input_weight(model: GoalMLP) -> Tensor:
    weight = model.input_projection.weight
    if not isinstance(weight, Tensor) or tuple(weight.shape) != (64, 19):
        raise RuntimeError("E19 input_projection.weight lost its frozen shape")
    return weight


def _delta_summary(delta: Tensor) -> dict[str, Any]:
    value = _cpu_clone(delta)
    absolute = torch.abs(value)
    return {
        "shape": list(value.shape),
        "target_scalar_count": value.numel(),
        "nonzero_scalar_count": int(torch.count_nonzero(value).item()),
        "digest": stable_state_digest(value),
        "l1_norm": float(torch.sum(absolute).item()),
        "l2_norm": float(torch.linalg.vector_norm(value).item()),
        "linf_norm": float(torch.max(absolute).item()) if value.numel() else 0.0,
        "finite": bool(torch.all(torch.isfinite(value)).item()),
    }


def make_q_pathway_branches(
    initial_model: GoalMLP,
    independent_model: GoalMLP,
    nested_model: GoalMLP,
    *,
    feature_names: Sequence[str],
) -> QPathwayBranches:
    """Deep-copy and edit all six E19 branches without mutating donors."""

    donors = {
        "initial": initial_model,
        "independent": independent_model,
        "nested": nested_model,
    }
    audit_q_pathway_contract(donors, feature_names)
    before = {
        name: stable_state_digest(model.state_dict()) for name, model in donors.items()
    }
    q_columns, sham_columns = _columns()
    initial_weight = _input_weight(initial_model).detach()
    independent_weight = _input_weight(independent_model).detach()
    nested_weight = _input_weight(nested_model).detach()
    restore_delta = _cpu_clone(
        initial_weight.index_select(1, q_columns)
        - independent_weight.index_select(1, q_columns)
    )
    transplant_delta = _cpu_clone(
        independent_weight.index_select(1, q_columns)
        - nested_weight.index_select(1, q_columns)
    )
    if not torch.all(torch.isfinite(restore_delta)) or not torch.all(
        torch.isfinite(transplant_delta)
    ):
        raise RuntimeError("E19 pathway displacement contains non-finite values")

    models = {
        "independent_noop": copy.deepcopy(independent_model),
        "independent_q_restore": copy.deepcopy(independent_model),
        "independent_padding_sham": copy.deepcopy(independent_model),
        "nested_noop": copy.deepcopy(nested_model),
        "nested_q_transplant": copy.deepcopy(nested_model),
        "nested_padding_sham": copy.deepcopy(nested_model),
    }
    with torch.no_grad():
        restore_weight = _input_weight(models["independent_q_restore"])
        restore_weight.index_copy_(
            1,
            q_columns,
            initial_weight.index_select(1, q_columns),
        )
        restore_sham_weight = _input_weight(models["independent_padding_sham"])
        restore_sham_weight.index_add_(
            1,
            sham_columns,
            restore_delta.to(
                dtype=restore_sham_weight.dtype, device=restore_sham_weight.device
            ),
        )
        transplant_weight = _input_weight(models["nested_q_transplant"])
        transplant_weight.index_copy_(
            1,
            q_columns,
            independent_weight.index_select(1, q_columns),
        )
        transplant_sham_weight = _input_weight(models["nested_padding_sham"])
        transplant_sham_weight.index_add_(
            1,
            sham_columns,
            transplant_delta.to(
                dtype=transplant_sham_weight.dtype,
                device=transplant_sham_weight.device,
            ),
        )
    after = {
        name: stable_state_digest(model.state_dict()) for name, model in donors.items()
    }
    if before != after:
        raise RuntimeError("E19 branch construction mutated a donor model")
    construction_state_digests = {
        name: stable_state_digest(model.state_dict()) for name, model in models.items()
    }
    construction_payload = {
        "donors": before,
        "branches": construction_state_digests,
        "restore_delta": _delta_summary(restore_delta),
        "transplant_delta": _delta_summary(transplant_delta),
        "features": tuple(feature_names),
    }
    result = QPathwayBranches(
        models=models,
        donor_state_digests=before,
        construction_state_digests=construction_state_digests,
        restore_delta=restore_delta,
        transplant_delta=transplant_delta,
        feature_names=tuple(str(name) for name in feature_names),
        construction_digest=stable_state_digest(construction_payload),
    )
    audit_q_pathway_branches(
        result,
        initial_model=initial_model,
        independent_model=independent_model,
        nested_model=nested_model,
    )
    return result


def _assert_unchanged_state(
    branch: GoalMLP,
    baseline: GoalMLP,
    *,
    edited_columns: tuple[int, int] | None,
    expected_final_columns: Tensor | None,
    branch_name: str,
) -> dict[str, Any]:
    branch_state = branch.state_dict()
    baseline_state = baseline.state_dict()
    if tuple(branch_state) != tuple(baseline_state):
        raise RuntimeError(f"{branch_name} changed the model state-key interface")
    unchanged: dict[str, Tensor] = {}
    for key in branch_state:
        if key == "input_projection.weight":
            continue
        if not torch.equal(branch_state[key], baseline_state[key]):
            raise RuntimeError(f"{branch_name} changed unaffected state tensor {key}")
        unchanged[key] = _cpu_clone(branch_state[key])
    branch_weight = branch_state["input_projection.weight"]
    baseline_weight = baseline_state["input_projection.weight"]
    if edited_columns is None:
        if not torch.equal(branch_weight, baseline_weight):
            raise RuntimeError(f"{branch_name} no-op changed input_projection.weight")
        realized = torch.zeros((64, 0), dtype=branch_weight.dtype)
    else:
        column_index = torch.tensor(edited_columns, dtype=torch.long)
        keep = torch.ones(branch_weight.shape[1], dtype=torch.bool)
        keep[column_index] = False
        if not torch.equal(branch_weight[:, keep], baseline_weight[:, keep]):
            raise RuntimeError(f"{branch_name} changed a non-designated input column")
        if expected_final_columns is None or not torch.equal(
            branch_weight.index_select(1, column_index),
            expected_final_columns.to(
                dtype=branch_weight.dtype, device=branch_weight.device
            ),
        ):
            raise RuntimeError(f"{branch_name} designated columns differ from the edit plan")
        realized = _cpu_clone(
            branch_weight.index_select(1, column_index)
            - baseline_weight.index_select(1, column_index)
        )
    unchanged["input_projection.weight_unedited"] = _cpu_clone(
        branch_weight
        if edited_columns is None
        else branch_weight[
            :,
            torch.tensor(
                [
                    index
                    for index in range(branch_weight.shape[1])
                    if index not in edited_columns
                ],
                dtype=torch.long,
            ),
        ]
    )
    return {
        "branch_state_digest": stable_state_digest(branch_state),
        "unchanged_state_digest": stable_state_digest(unchanged),
        "edited_columns": [] if edited_columns is None else list(edited_columns),
        "target_scalar_count": 0 if edited_columns is None else 128,
        "realized_delta": _delta_summary(realized),
        "unchanged_state_verified": True,
        "designated_columns_verified": True,
    }


def audit_q_pathway_branches(
    branches: QPathwayBranches,
    *,
    initial_model: GoalMLP,
    independent_model: GoalMLP,
    nested_model: GoalMLP,
) -> dict[str, Any]:
    """Audit donor immutability, exact edits, and every unchanged state tensor."""

    if tuple(branches.models) != Q_PATHWAY_BRANCHES:
        raise RuntimeError(
            f"E19 branch names/order differ from the frozen plan: {tuple(branches.models)}"
        )
    donors = {
        "initial": initial_model,
        "independent": independent_model,
        "nested": nested_model,
    }
    contract = audit_q_pathway_contract(donors, branches.feature_names)
    current_donor_digests = {
        name: stable_state_digest(model.state_dict()) for name, model in donors.items()
    }
    if current_donor_digests != branches.donor_state_digests:
        raise RuntimeError("an E19 donor changed after branch construction")
    for name, model in branches.models.items():
        _validate_model(model, name)
        current = stable_state_digest(model.state_dict())
        if current != branches.construction_state_digests[name]:
            raise RuntimeError(f"{name} changed before the construction audit")

    q_columns, sham_columns = _columns()
    initial_q = _input_weight(initial_model).detach().index_select(1, q_columns)
    independent_q = _input_weight(independent_model).detach().index_select(1, q_columns)
    independent_sham = _input_weight(independent_model).detach().index_select(
        1, sham_columns
    )
    nested_q = _input_weight(nested_model).detach().index_select(1, q_columns)
    nested_sham = _input_weight(nested_model).detach().index_select(1, sham_columns)
    restore_delta = branches.restore_delta
    transplant_delta = branches.transplant_delta
    expected_restore = _cpu_clone(initial_q - independent_q)
    expected_transplant = _cpu_clone(independent_q - nested_q)
    if not torch.equal(restore_delta, expected_restore):
        raise RuntimeError("stored restore displacement differs from donor states")
    if not torch.equal(transplant_delta, expected_transplant):
        raise RuntimeError("stored transplant displacement differs from donor states")

    expected_columns: dict[str, Tensor | None] = {
        "independent_noop": None,
        "independent_q_restore": initial_q,
        "independent_padding_sham": independent_sham
        + restore_delta.to(dtype=independent_sham.dtype),
        "nested_noop": None,
        "nested_q_transplant": independent_q,
        "nested_padding_sham": nested_sham
        + transplant_delta.to(dtype=nested_sham.dtype),
    }
    edited_columns: dict[str, tuple[int, int] | None] = {
        "independent_noop": None,
        "independent_q_restore": Q_COLUMN_INDICES,
        "independent_padding_sham": PADDING_SHAM_COLUMN_INDICES,
        "nested_noop": None,
        "nested_q_transplant": Q_COLUMN_INDICES,
        "nested_padding_sham": PADDING_SHAM_COLUMN_INDICES,
    }
    baselines = {"independent": independent_model, "nested": nested_model}
    branch_audits = {
        name: _assert_unchanged_state(
            model,
            baselines[_BRANCH_BASELINES[name]],
            edited_columns=edited_columns[name],
            expected_final_columns=expected_columns[name],
            branch_name=name,
        )
        for name, model in branches.models.items()
    }
    if branch_audits["independent_noop"]["branch_state_digest"] != branches.donor_state_digests[
        "independent"
    ]:
        raise RuntimeError("independent no-op is not an exact donor clone")
    if branch_audits["nested_noop"]["branch_state_digest"] != branches.donor_state_digests[
        "nested"
    ]:
        raise RuntimeError("nested no-op is not an exact donor clone")

    edit_audits = {
        "restore": {
            "intended_delta": _delta_summary(restore_delta),
            "active_realized_delta": branch_audits["independent_q_restore"][
                "realized_delta"
            ],
            "sham_realized_delta": branch_audits["independent_padding_sham"][
                "realized_delta"
            ],
            "same_intended_delta_applied": True,
            "active_target": "Q_1,Q_2",
            "sham_target": "R_4,R_5",
        },
        "transplant": {
            "intended_delta": _delta_summary(transplant_delta),
            "active_realized_delta": branch_audits["nested_q_transplant"][
                "realized_delta"
            ],
            "sham_realized_delta": branch_audits["nested_padding_sham"][
                "realized_delta"
            ],
            "same_intended_delta_applied": True,
            "active_target": "Q_1,Q_2",
            "sham_target": "R_4,R_5",
        },
    }
    audit_payload = {
        "contract": contract,
        "donors": current_donor_digests,
        "branches": branch_audits,
        "edits": edit_audits,
    }
    return {
        **audit_payload,
        "construction_digest": branches.construction_digest,
        "audit_digest": stable_state_digest(audit_payload),
        "donors_unchanged": True,
        "all_six_branches_verified": True,
        "all_unaffected_state_exact": True,
        "all_edits_target_exactly_128_scalars": True,
    }


def _preactivations(model: GoalMLP, features: Tensor) -> Tensor:
    weight = _input_weight(model)
    bias = model.input_projection.bias
    return torch.nn.functional.linear(
        features.to(dtype=weight.dtype, device=weight.device), weight, bias
    )


def _preactivation_comparison(base: Tensor, edited: Tensor) -> dict[str, Any]:
    delta = edited - base
    base_rms = float(torch.sqrt(torch.mean(torch.square(base))).item())
    delta_rms = float(torch.sqrt(torch.mean(torch.square(delta))).item())
    flips = (base > 0) != (edited > 0)
    return {
        "n_rows": int(base.shape[0]),
        "hidden_units": int(base.shape[1]),
        "activation_count": base.numel(),
        "baseline_preactivation_rms": base_rms,
        "edited_preactivation_rms": float(
            torch.sqrt(torch.mean(torch.square(edited))).item()
        ),
        "preactivation_delta_rms": delta_rms,
        "relative_delta_rms": delta_rms / base_rms if base_rms > 0.0 else None,
        "preactivation_delta_mean_absolute": float(torch.mean(torch.abs(delta)).item()),
        "preactivation_delta_max_absolute": float(torch.max(torch.abs(delta)).item()),
        "relu_state_flip_count": int(torch.count_nonzero(flips).item()),
        "relu_state_flip_rate": float(torch.mean(flips.to(torch.float64)).item()),
        "baseline_preactivation_digest": stable_state_digest(base),
        "edited_preactivation_digest": stable_state_digest(edited),
        "preactivation_delta_digest": stable_state_digest(delta),
    }


def preactivation_edit_effects(
    branches: QPathwayBranches,
    panel: SemanticBatch,
    *,
    max_k: int = 5,
    include_state: bool = True,
) -> dict[str, Any]:
    """Measure first-layer RMS disruption and ReLU gating changes on one panel."""

    panel_names = panel.feature_names(max_k=max_k, include_state=include_state)
    if panel_names != branches.feature_names:
        raise ValueError("preactivation panel feature interface differs from E19 donors")
    array = panel.features(max_k=max_k, include_state=include_state)
    if array.shape != (len(panel), 19) or not np.all(np.isfinite(array)):
        raise ValueError("preactivation panel must have finite E17 features")
    features = torch.as_tensor(array, dtype=torch.float32)
    with torch.no_grad():
        values = {
            name: _cpu_clone(_preactivations(model, features))
            for name, model in branches.models.items()
        }
    comparisons = {
        "independent_q_restore": _preactivation_comparison(
            values["independent_noop"], values["independent_q_restore"]
        ),
        "independent_padding_sham": _preactivation_comparison(
            values["independent_noop"], values["independent_padding_sham"]
        ),
        "nested_q_transplant": _preactivation_comparison(
            values["nested_noop"], values["nested_q_transplant"]
        ),
        "nested_padding_sham": _preactivation_comparison(
            values["nested_noop"], values["nested_padding_sham"]
        ),
    }
    for name, result in comparisons.items():
        for key, value in result.items():
            if isinstance(value, float) and not math.isfinite(value):
                raise RuntimeError(f"{name} produced non-finite preactivation metric {key}")
    return {
        "panel_n": len(panel),
        "panel_digest": semantic_batch_digest(panel),
        "feature_names_digest": stable_state_digest(panel_names),
        "max_k": max_k,
        "include_state": include_state,
        "comparisons": comparisons,
        "all_values_finite": True,
    }


__all__ = [
    "EXPECTED_FEATURE_NAMES",
    "EXPECTED_STATE_SHAPES",
    "PADDING_SHAM_COLUMN_INDICES",
    "Q_COLUMN_INDICES",
    "Q_PATHWAY_BRANCHES",
    "QPathwayBranches",
    "audit_q_pathway_branches",
    "audit_q_pathway_contract",
    "make_q_pathway_branches",
    "preactivation_edit_effects",
]
