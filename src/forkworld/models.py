"""Neural models and controlled-update parameterizations for ForkWorld.

The central experimental distinction in this project is between *model capacity*
and *update capacity*.  :class:`GoalMLP` controls the former, while
:class:`ExactSubspaceModel` and :func:`configure_update_mode` control the latter.
The subspace wrapper intentionally exposes exactly ``U`` trainable scalar
parameters, independent of the size of the wrapped network.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Mapping, Sequence

import torch
from torch import Tensor, nn

try:  # torch.func is the supported API on torch >= 2.0.
    from torch.func import functional_call as _functional_call
except ImportError:  # pragma: no cover - compatibility with older PyTorch.
    from torch.nn.utils.stateless import functional_call as _functional_call


ActivationName = Literal["relu", "tanh", "gelu", "silu"]
ProjectionKind = Literal["intrinsic", "count_sketch"]
UpdateMode = Literal[
    "full",
    "head_only",
    "head-only",
    "intrinsic",
    "count_sketch",
    "subspace",
]


def _activation(name: ActivationName) -> nn.Module:
    activations: dict[str, type[nn.Module]] = {
        "relu": nn.ReLU,
        "tanh": nn.Tanh,
        "gelu": nn.GELU,
        "silu": nn.SiLU,
    }
    try:
        return activations[name.lower()]()
    except KeyError as exc:
        choices = ", ".join(sorted(activations))
        raise ValueError(f"unknown activation {name!r}; expected one of {choices}") from exc


def _normalise_nuisance_heads(
    heads: int | Mapping[str, int] | Sequence[str] | None,
) -> dict[str, int]:
    """Convert nuisance-head shorthand to a checked name -> output-size map.

    An integer denotes that many independent binary nuisance bits.  A sequence
    names independent binary bits.  A mapping may additionally request a
    multiclass head by assigning an output size larger than one.
    """

    if heads is None:
        return {}
    if isinstance(heads, bool):
        raise TypeError("nuisance_heads must not be a bool")
    if isinstance(heads, int):
        if heads < 0:
            raise ValueError("nuisance_heads must be non-negative")
        return {f"nuisance_{index}": 1 for index in range(heads)}
    if isinstance(heads, Mapping):
        result = {str(name): int(size) for name, size in heads.items()}
    else:
        result = {str(name): 1 for name in heads}
    if any(not name for name in result):
        raise ValueError("nuisance-head names must be non-empty")
    if any(size < 1 for size in result.values()):
        raise ValueError("every nuisance-head output size must be at least one")
    if len(result) != len(list(result.keys())):  # Defensive for exotic mappings.
        raise ValueError("nuisance-head names must be unique")
    return result


@dataclass(frozen=True)
class MLPConfig:
    """Architecture definition for a binary goal selector.

    ``depth`` is the number of hidden affine transformations.  ``depth=0`` is
    an intentionally useful logistic-regression control.  For residual models,
    the first transformation projects to ``width`` and subsequent same-width
    transformations are residual blocks.
    """

    input_dim: int
    width: int = 64
    depth: int = 2
    activation: ActivationName = "relu"
    residual: bool = False
    bias: bool = True
    nuisance_heads: int | Mapping[str, int] | Sequence[str] | None = None
    nuisance_bits: int = 0

    def __post_init__(self) -> None:
        if isinstance(self.input_dim, bool) or self.input_dim < 1:
            raise ValueError("input_dim must be a positive integer")
        if isinstance(self.width, bool) or self.width < 1:
            raise ValueError("width must be a positive integer")
        if isinstance(self.depth, bool) or self.depth < 0:
            raise ValueError("depth must be a non-negative integer")
        if isinstance(self.nuisance_bits, bool) or self.nuisance_bits < 0:
            raise ValueError("nuisance_bits must be a non-negative integer")
        if self.nuisance_heads is not None and self.nuisance_bits:
            raise ValueError("specify nuisance_heads or nuisance_bits, not both")
        _activation(self.activation)
        _normalise_nuisance_heads(
            self.nuisance_heads if self.nuisance_heads is not None else self.nuisance_bits
        )


@dataclass
class GoalModelOutput:
    """Full output of a goal selector when auxiliary heads are requested."""

    goal_logits: Tensor
    nuisance_logits: dict[str, Tensor]

    @property
    def logits(self) -> Tensor:
        """Alias used by generic policy/evaluation utilities."""

        return self.goal_logits


class GoalMLP(nn.Module):
    """Configurable MLP with a binary goal head and optional nuisance heads.

    By default ``forward`` returns a one-dimensional logit for each example.
    Passing ``return_aux=True`` returns :class:`GoalModelOutput` instead.  This
    keeps clean policy code simple without hiding the auxiliary predictions
    needed by nuisance-rich SFT.
    """

    def __init__(
        self,
        input_dim: int | MLPConfig,
        width: int = 64,
        depth: int = 2,
        activation: ActivationName = "relu",
        residual: bool = False,
        bias: bool = True,
        nuisance_heads: int | Mapping[str, int] | Sequence[str] | None = None,
        nuisance_bits: int = 0,
    ) -> None:
        super().__init__()
        if isinstance(input_dim, MLPConfig):
            config = input_dim
        else:
            config = MLPConfig(
                input_dim=input_dim,
                width=width,
                depth=depth,
                activation=activation,
                residual=residual,
                bias=bias,
                nuisance_heads=nuisance_heads,
                nuisance_bits=nuisance_bits,
            )
        self.config = config
        self.residual = config.residual

        if config.depth == 0:
            self.input_projection: nn.Module = nn.Identity()
            self.hidden_layers = nn.ModuleList()
            representation_dim = config.input_dim
        else:
            self.input_projection = nn.Linear(config.input_dim, config.width, bias=config.bias)
            self.hidden_layers = nn.ModuleList(
                nn.Linear(config.width, config.width, bias=config.bias)
                for _ in range(config.depth - 1)
            )
            representation_dim = config.width

        self.activation = _activation(config.activation)
        self.representation_dim = representation_dim
        self.goal_head = nn.Linear(representation_dim, 1, bias=config.bias)
        nuisance_spec = _normalise_nuisance_heads(
            config.nuisance_heads
            if config.nuisance_heads is not None
            else config.nuisance_bits
        )
        self.nuisance_heads = nn.ModuleDict(
            {
                name: nn.Linear(representation_dim, output_dim, bias=config.bias)
                for name, output_dim in nuisance_spec.items()
            }
        )

    def encode(self, x: Tensor) -> Tensor:
        """Return the shared representation before prediction heads."""

        if not torch.is_tensor(x):
            x = torch.as_tensor(x)
        if not x.is_floating_point():
            x = x.float()
        if x.ndim == 1:
            x = x.unsqueeze(0)
        if x.shape[-1] != self.config.input_dim:
            raise ValueError(
                f"expected final input dimension {self.config.input_dim}, got {x.shape[-1]}"
            )
        if self.config.depth == 0:
            return x

        hidden = self.activation(self.input_projection(x))
        for layer in self.hidden_layers:
            transformed = self.activation(layer(hidden))
            if self.residual:
                # The scaling keeps activation variance comparable as depth changes.
                hidden = (hidden + transformed) * (2.0**-0.5)
            else:
                hidden = transformed
        return hidden

    def forward(self, x: Tensor, *, return_aux: bool = False) -> Tensor | GoalModelOutput:
        representation = self.encode(x)
        goal_logits = self.goal_head(representation).squeeze(-1)
        if not return_aux:
            return goal_logits
        nuisance_logits = {}
        for name, head in self.nuisance_heads.items():
            value = head(representation)
            nuisance_logits[name] = value.squeeze(-1) if value.shape[-1] == 1 else value
        return GoalModelOutput(goal_logits=goal_logits, nuisance_logits=nuisance_logits)

    @torch.no_grad()
    def predict_proba(self, x: Tensor) -> Tensor:
        """Return ``P(Y=+1 | x)``."""

        output = self(x)
        assert torch.is_tensor(output)
        return output.sigmoid()

    @torch.no_grad()
    def predict(self, x: Tensor, *, signed: bool = True) -> Tensor:
        """Return deterministic binary choices, as ``{-1,+1}`` by default."""

        choices = (self.predict_proba(x) >= 0.5).long()
        return choices.mul(2).sub(1) if signed else choices

    def head_parameters(self, *, include_nuisance: bool = True) -> list[nn.Parameter]:
        """Return prediction-head parameters in a stable order."""

        parameters = list(self.goal_head.parameters())
        if include_nuisance:
            parameters.extend(self.nuisance_heads.parameters())
        return parameters


class ValueMLP(GoalMLP):
    """A scalar value network kept separate from actor update budgets."""

    def __init__(
        self,
        input_dim: int,
        width: int = 256,
        depth: int = 3,
        activation: ActivationName = "gelu",
        residual: bool = True,
        bias: bool = True,
    ) -> None:
        super().__init__(
            input_dim=input_dim,
            width=width,
            depth=depth,
            activation=activation,
            residual=residual,
            bias=bias,
            nuisance_heads=None,
        )


def parameter_count(module: nn.Module, *, trainable_only: bool = False) -> int:
    """Count scalar parameters, deduplicating shared parameters."""

    return sum(
        parameter.numel()
        for parameter in module.parameters()
        if not trainable_only or parameter.requires_grad
    )


def trainable_parameter_count(module: nn.Module) -> int:
    """Return the exact number of scalars visible to an optimizer."""

    return parameter_count(module, trainable_only=True)


# Deliberately explicit alias for experiment tables and hidden invariants.
exact_trainable_parameter_count = trainable_parameter_count


def _set_all_trainable(module: nn.Module, value: bool) -> None:
    for parameter in module.parameters():
        parameter.requires_grad_(value)


def _head_parameter_ids(module: nn.Module, include_nuisance: bool) -> set[int]:
    if hasattr(module, "head_parameters"):
        method = getattr(module, "head_parameters")
        try:
            return {id(parameter) for parameter in method(include_nuisance=include_nuisance)}
        except TypeError:
            return {id(parameter) for parameter in method()}

    prefixes = ("goal_head.", "head.", "policy_head.", "output_head.")
    if include_nuisance:
        prefixes += ("nuisance_heads.", "auxiliary_heads.")
    return {
        id(parameter)
        for name, parameter in module.named_parameters()
        if name.startswith(prefixes)
    }


class ExactSubspaceModel(nn.Module):
    """Optimize a frozen model through exactly ``budget`` trainable scalars.

    Let ``theta_0`` be the wrapped parameters and ``z`` the trainable vector.
    Forward passes use ``theta = theta_0 + A z``.  Two deterministic projections
    are supported:

    * ``intrinsic`` uses a dense random matrix with unit-norm columns;
    * ``count_sketch`` assigns every base parameter to one signed bucket and
      normalizes each bucket.  Assignment guarantees every bucket is used.

    The projection and frozen base parameters are buffers/parameters in the
    state dict but only ``coordinates`` has ``requires_grad=True``.  Consequently
    standard optimizers, parameter-count reports, and actor-budget comparisons
    all see exactly the requested budget.
    """

    def __init__(
        self,
        base_model: nn.Module,
        budget: int,
        *,
        projection: ProjectionKind = "count_sketch",
        seed: int = 0,
        scale: float = 1.0,
    ) -> None:
        super().__init__()
        if isinstance(base_model, ExactSubspaceModel):
            raise ValueError("nesting ExactSubspaceModel wrappers is not supported")
        if isinstance(budget, bool) or not isinstance(budget, int):
            raise TypeError("budget must be an integer number of trainable scalars")
        if projection not in ("intrinsic", "count_sketch"):
            raise ValueError("projection must be 'intrinsic' or 'count_sketch'")
        if not torch.isfinite(torch.tensor(float(scale))) or scale <= 0:
            raise ValueError("scale must be finite and positive")

        named_parameters = list(base_model.named_parameters())
        if not named_parameters:
            raise ValueError("cannot make a subspace update for a parameter-free model")
        total = sum(parameter.numel() for _, parameter in named_parameters)
        if budget < 1 or budget > total:
            raise ValueError(f"budget must satisfy 1 <= budget <= {total}; got {budget}")
        if any(not parameter.is_floating_point() for _, parameter in named_parameters):
            raise TypeError("all wrapped parameters must have a floating-point dtype")

        self.base_model = base_model
        _set_all_trainable(self.base_model, False)
        self.budget = budget
        self.projection_kind: ProjectionKind = projection
        self.seed = int(seed)
        self.scale = float(scale)
        self.base_parameter_count = total
        self._parameter_names = tuple(name for name, _ in named_parameters)
        self._parameter_shapes = tuple(parameter.shape for _, parameter in named_parameters)
        self._parameter_sizes = tuple(parameter.numel() for _, parameter in named_parameters)

        reference = named_parameters[0][1]
        coordinate_dtype = reference.dtype
        coordinate_device = reference.device
        self.coordinates = nn.Parameter(
            torch.zeros(budget, dtype=coordinate_dtype, device=coordinate_device)
        )

        generator = torch.Generator(device="cpu")
        generator.manual_seed(self.seed)
        if projection == "intrinsic":
            matrix = torch.randn(total, budget, generator=generator, dtype=torch.float32)
            matrix = matrix / matrix.norm(dim=0, keepdim=True).clamp_min(1e-12)
            self.register_buffer(
                "projection_matrix",
                matrix.to(device=coordinate_device, dtype=coordinate_dtype),
            )
            self.register_buffer("bucket_index", None)
            self.register_buffer("bucket_sign", None)
        else:
            # Start with a surjection, then shuffle it so all U coordinates receive
            # gradients without creating a structured prefix in parameter space.
            bucket = torch.arange(total, dtype=torch.long).remainder(budget)
            bucket = bucket[torch.randperm(total, generator=generator)]
            sign = torch.randint(0, 2, (total,), generator=generator, dtype=torch.int8)
            sign = sign.mul(2).sub(1)
            counts = torch.bincount(bucket, minlength=budget).float()
            sign = sign.float() / counts[bucket].sqrt()
            self.register_buffer("projection_matrix", None)
            self.register_buffer("bucket_index", bucket.to(device=coordinate_device))
            self.register_buffer(
                "bucket_sign", sign.to(device=coordinate_device, dtype=coordinate_dtype)
            )

        if trainable_parameter_count(self) != budget:
            raise RuntimeError("internal error: subspace wrapper did not preserve exact budget")

    @property
    def update(self) -> nn.Parameter:
        """Alias for the trainable coordinate vector."""

        return self.coordinates

    def zero_update(self) -> None:
        """Restore the wrapped base model without changing its frozen anchor."""

        with torch.no_grad():
            self.coordinates.zero_()

    def reset_parameters(self) -> None:
        """Reset update coordinates; used by training reset semantics."""

        self.zero_update()

    def _flat_delta(self) -> Tensor:
        if self.projection_kind == "intrinsic":
            assert self.projection_matrix is not None
            return self.scale * (self.projection_matrix @ self.coordinates)
        assert self.bucket_index is not None and self.bucket_sign is not None
        return self.scale * self.bucket_sign * self.coordinates[self.bucket_index]

    def effective_parameters(self) -> dict[str, Tensor]:
        """Return differentiable parameter overrides for ``functional_call``."""

        delta = self._flat_delta()
        overrides: dict[str, Tensor] = {}
        offset = 0
        for (name, parameter), size, shape in zip(
            self.base_model.named_parameters(),
            self._parameter_sizes,
            self._parameter_shapes,
            strict=True,
        ):
            piece = delta[offset : offset + size].reshape(shape)
            overrides[name] = parameter.detach() + piece.to(parameter.dtype)
            offset += size
        if offset != self.base_parameter_count:  # pragma: no cover - invariant guard.
            raise RuntimeError("wrapped model parameter structure changed after construction")
        return overrides

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        return _functional_call(
            self.base_model,
            self.effective_parameters(),
            args,
            kwargs,
            strict=False,
        )

    def extra_repr(self) -> str:
        return (
            f"budget={self.budget}, projection={self.projection_kind!r}, "
            f"base_parameters={self.base_parameter_count}, seed={self.seed}, scale={self.scale}"
        )


@dataclass(frozen=True)
class UpdateBudgetReport:
    """Auditable capacity report attached to a configured actor."""

    mode: str
    total_parameters: int
    trainable_parameters: int
    requested_budget: int | None = None
    projection: ProjectionKind | None = None

    @property
    def exact(self) -> bool:
        return self.requested_budget is None or self.trainable_parameters == self.requested_budget


def validate_trainable_budget(module: nn.Module, expected: int) -> int:
    """Raise if ``module`` does not expose exactly ``expected`` trainable scalars."""

    if isinstance(expected, bool) or not isinstance(expected, int) or expected < 0:
        raise ValueError("expected budget must be a non-negative integer")
    actual = trainable_parameter_count(module)
    if actual != expected:
        raise ValueError(
            f"trainable-parameter budget mismatch: expected {expected}, found {actual}"
        )
    return actual


def update_budget_report(module: nn.Module) -> UpdateBudgetReport:
    """Describe the currently configured model/update capacity."""

    if isinstance(module, ExactSubspaceModel):
        return UpdateBudgetReport(
            mode=module.projection_kind,
            total_parameters=module.base_parameter_count,
            trainable_parameters=trainable_parameter_count(module),
            requested_budget=module.budget,
            projection=module.projection_kind,
        )
    mode = str(getattr(module, "update_mode", "custom"))
    return UpdateBudgetReport(
        mode=mode,
        total_parameters=parameter_count(module),
        trainable_parameters=trainable_parameter_count(module),
    )


def configure_update_mode(
    model: nn.Module,
    mode: UpdateMode = "full",
    *,
    budget: int | None = None,
    projection: ProjectionKind = "count_sketch",
    seed: int = 0,
    scale: float = 1.0,
    include_nuisance_heads: bool = True,
) -> nn.Module:
    """Configure full, head-only, or exact-budget training.

    Full and head-only modes mutate ``requires_grad`` flags and return ``model``.
    Subspace modes return an :class:`ExactSubspaceModel`; callers must use that
    returned wrapper.  ``mode='intrinsic'`` and ``mode='count_sketch'`` are
    convenient aliases for ``mode='subspace'`` with the corresponding projection.
    """

    canonical = mode.lower().replace("-", "_")
    if canonical == "full":
        if budget is not None:
            raise ValueError("budget is only valid for exact subspace update modes")
        _set_all_trainable(model, True)
        setattr(model, "update_mode", "full")
        return model

    if canonical in ("head", "head_only"):
        if budget is not None:
            raise ValueError("head-only budget is determined by the architecture")
        _set_all_trainable(model, False)
        selected = _head_parameter_ids(model, include_nuisance_heads)
        if not selected:
            raise ValueError(
                "could not identify a prediction head; provide a model with "
                "head_parameters() or a conventional goal_head/head attribute"
            )
        for parameter in model.parameters():
            if id(parameter) in selected:
                parameter.requires_grad_(True)
        setattr(model, "update_mode", "head_only")
        return model

    if canonical in ("intrinsic", "count_sketch"):
        projection = canonical  # type: ignore[assignment]
        canonical = "subspace"
    if canonical == "subspace":
        if budget is None:
            raise ValueError("an exact integer budget is required for subspace updates")
        return ExactSubspaceModel(
            model,
            budget,
            projection=projection,
            seed=seed,
            scale=scale,
        )
    raise ValueError(
        f"unknown update mode {mode!r}; expected full, head_only, intrinsic, "
        "count_sketch, or subspace"
    )


# Short aliases used in configuration-driven runners.
make_update_model = configure_update_mode
apply_update_mode = configure_update_mode


__all__ = [
    "ActivationName",
    "ExactSubspaceModel",
    "GoalMLP",
    "GoalModelOutput",
    "MLPConfig",
    "ProjectionKind",
    "UpdateBudgetReport",
    "UpdateMode",
    "ValueMLP",
    "apply_update_mode",
    "configure_update_mode",
    "exact_trainable_parameter_count",
    "make_update_model",
    "parameter_count",
    "trainable_parameter_count",
    "update_budget_report",
    "validate_trainable_budget",
]
