"""Deterministic semantic datasets for the ForkWorld experiments.

The generators in this module make experimental invariants explicit.  In
particular, finite-sample proxy accuracy is an exact count (never an expectation),
the interaction code decodes the intended sign on every row, and semantic IDs are
kept separate from model inputs.  Keeping IDs out of the observation is important:
they are used to define persistent noise and repeated H4 contexts, not as an
accidental lookup-table feature.
"""

from __future__ import annotations

import hashlib
import math
import re
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field, replace
from types import MappingProxyType
from typing import Any, Literal, cast

import numpy as np
from numpy.typing import ArrayLike, NDArray

SignArray = NDArray[np.int8]
FloatArray = NDArray[np.float32]
IntArray = NDArray[np.int64]
H4Condition = Literal["concentrated", "diverse", "structured_holdout"]
H9Condition = Literal["normal", "removed", "randomized", "mismatched"]
TargetRule = Literal["parity", "majority", "conjunction", "multiplexer"]
TARGET_RULES: tuple[TargetRule, ...] = (
    "parity",
    "majority",
    "conjunction",
    "multiplexer",
)

# H4 holdout families are transformations with distinct physical semantics.  The
# first two are used for training by default and the latter two are genuinely
# unseen mechanisms at evaluation.  Integer aliases 0--3 remain accepted at the
# public data boundary so old run manifests fail gracefully rather than silently
# changing meaning.
H4_FAILURE_MECHANISMS = (
    "location_reflection",
    "coordinate_exchange",
    "geometry_rotation",
    "nuisance_inversion",
)

_R_CHANNEL = re.compile(r"^R_(\d+)$")


def _as_vector(value: ArrayLike, name: str, n: int | None = None) -> NDArray[Any]:
    array = np.asarray(value)
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional, got shape {array.shape}")
    if n is not None and len(array) != n:
        raise ValueError(f"{name} has {len(array)} rows; expected {n}")
    return np.array(array, copy=True)


def _as_rows(value: ArrayLike, name: str, n: int) -> NDArray[Any]:
    array = np.asarray(value)
    if array.ndim == 1:
        array = array[:, None]
    if array.ndim != 2 or array.shape[0] != n:
        raise ValueError(f"{name} must have shape [n, d] with n={n}, got {array.shape}")
    return np.array(array, copy=True)


def _require_signs(value: ArrayLike, name: str, n: int | None = None) -> SignArray:
    array = _as_vector(value, name, n)
    if not np.all(np.isin(array, (-1, 1))):
        raise ValueError(f"{name} must contain only -1 and +1")
    return array.astype(np.int8, copy=False)


def _freeze_mapping(
    values: Mapping[str, ArrayLike], n: int, *, name: str
) -> Mapping[str, NDArray[Any]]:
    result: dict[str, NDArray[Any]] = {}
    for key, value in values.items():
        if not isinstance(key, str) or not key:
            raise ValueError(f"{name} names must be non-empty strings")
        if key in result:
            raise ValueError(f"duplicate {name} name: {key!r}")
        array = np.asarray(value)
        if array.ndim not in (1, 2) or array.shape[0] != n:
            raise ValueError(f"{name}[{key!r}] must have shape [n] or [n,d], got {array.shape}")
        result[key] = np.array(array, copy=True)
    return MappingProxyType(result)


def _stable_seed(*parts: object) -> int:
    """Return a process-independent 64-bit seed for small deterministic helpers."""

    digest = hashlib.blake2b(digest_size=8, person=b"forkdata")
    for part in parts:
        payload = str(part).encode("utf-8")
        digest.update(len(payload).to_bytes(4, "little"))
        digest.update(payload)
    return int.from_bytes(digest.digest(), "little")


def _stateless_uint64(ids: IntArray, seed: int, channel: object) -> NDArray[np.uint64]:
    """Apply a seeded bijective 64-bit mix to IDs without using row order."""

    salt = np.uint64(_stable_seed(seed, channel))
    values = np.asarray(ids, dtype=np.int64).astype(np.uint64, copy=False)
    with np.errstate(over="ignore"):
        z = values ^ salt
        z = z + np.uint64(0x9E3779B97F4A7C15)
        z = (z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
        z = (z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
        z = z ^ (z >> np.uint64(31))
    return z


def _stateless_signs(ids: IntArray, seed: int, channel: object) -> SignArray:
    """Independent-looking signs keyed by IDs rather than row order."""

    z = _stateless_uint64(ids, seed, channel)
    return np.where((z & np.uint64(1)) == 0, -1, 1).astype(np.int8)


def _h4_context_fields(ids: IntArray, seed: int) -> tuple[FloatArray, ...]:
    """Encode IDs as four opaque fields with an exactly injective joint value.

    SplitMix64 is a permutation of 64-bit words. Its four 16-bit chunks therefore
    retain distinctness without exposing a raw, ordinal semantic ID. Every uint16
    value also has a distinct float32 image under this affine scaling.
    """

    mixed = _stateless_uint64(ids, seed, "h4-context-code-v2")
    mask = np.uint64(0xFFFF)
    fields: list[FloatArray] = []
    for shift in (0, 16, 32, 48):
        word = ((mixed >> np.uint64(shift)) & mask).astype(np.float32)
        fields.append((word / np.float32(32767.5) - np.float32(1.0)).astype(np.float32))
    return tuple(fields)


def _balanced_binary(n: int, rng: np.random.Generator) -> NDArray[np.int8]:
    values = np.tile(np.asarray([0, 1], dtype=np.int8), n // 2)
    rng.shuffle(values)
    return values


def _validate_n(n: int, *, name: str = "n") -> int:
    if isinstance(n, (bool, np.bool_)) or int(n) != n or int(n) <= 0:
        raise ValueError(f"{name} must be a positive integer")
    n = int(n)
    if n % 2:
        raise ValueError(f"{name} must be even so Y is exactly balanced")
    return n


def balanced_signs(n: int, seed: int = 0, *, shuffle: bool = True) -> SignArray:
    """Return exactly ``n/2`` negative and ``n/2`` positive signs."""

    n = _validate_n(n)
    values = np.tile(np.asarray([-1, 1], dtype=np.int8), n // 2)
    if shuffle:
        np.random.default_rng(seed).shuffle(values)
    return values


def validate_proxy_confounds(
    n: int,
    *,
    q: float | None = None,
    n_conflict: int | None = None,
    atol: float = 1e-12,
) -> tuple[int, float]:
    """Resolve proxy accuracy and conflict count without silently changing either.

    At least one of ``q`` and ``n_conflict`` is required.  If both are supplied,
    they must obey ``q = 1 - n_conflict / n``.  A requested ``q`` whose exact
    finite-sample count is impossible raises an error instead of being rounded.

    Returns:
        ``(n_conflict, realized_q)``.
    """

    n = _validate_n(n)
    if q is None and n_conflict is None:
        raise ValueError("provide q, n_conflict, or both")
    if q is not None:
        q = float(q)
        if not math.isfinite(q) or not 0.0 <= q <= 1.0:
            raise ValueError("q must lie in [0, 1]")
        raw_conflicts = n * (1.0 - q)
        implied_count = int(round(raw_conflicts))
        if not math.isclose(raw_conflicts, implied_count, rel_tol=0.0, abs_tol=atol):
            raise ValueError(
                f"q={q:g} is not exactly realizable with n={n}; "
                "choose n so n*(1-q) is an integer"
            )
        if n_conflict is None:
            n_conflict = implied_count
        elif int(n_conflict) != implied_count:
            raise ValueError(
                "q and n_conflict are confounded and inconsistent: "
                f"q={q:g} implies {implied_count}, received {n_conflict}"
            )
    assert n_conflict is not None
    if isinstance(n_conflict, (bool, np.bool_)) or int(n_conflict) != n_conflict:
        raise ValueError("n_conflict must be an integer")
    n_conflict = int(n_conflict)
    if not 0 <= n_conflict <= n:
        raise ValueError("n_conflict must lie in [0, n]")
    realized_q = 1.0 - n_conflict / n
    if q is not None and not math.isclose(q, realized_q, rel_tol=0.0, abs_tol=atol):
        raise ValueError("q does not match the realized finite-sample proxy accuracy")
    return n_conflict, realized_q


def degree_k_code(
    y: ArrayLike,
    k: int,
    seed: int = 0,
    *,
    rng: np.random.Generator | None = None,
) -> SignArray:
    """Construct a degree-``k`` Rademacher code whose row product is exactly Y.

    ``k=1`` is the useful calibration endpoint ``R_1=Y``.  For ``k>1``, the
    first ``k-1`` channels are sampled independently and the final channel is
    set algebraically.
    """

    signs = _require_signs(y, "y")
    if isinstance(k, (bool, np.bool_)) or int(k) != k or int(k) < 1:
        raise ValueError("k must be a positive integer")
    k = int(k)
    generator = rng if rng is not None else np.random.default_rng(seed)
    if k == 1:
        return signs[:, None].copy()
    prefix = generator.choice(np.asarray([-1, 1], dtype=np.int8), size=(len(signs), k - 1))
    final = signs * np.prod(prefix, axis=1, dtype=np.int8)
    code = np.column_stack((prefix, final)).astype(np.int8, copy=False)
    if not np.array_equal(np.prod(code, axis=1, dtype=np.int8), signs):  # pragma: no cover
        raise RuntimeError("internal error constructing exact interaction code")
    return code


# Readable alias used in experiment prose and notebooks.
make_degree_k_code = degree_k_code


def _target_rule(value: object) -> TargetRule:
    rule = str(value).strip().lower().replace("-", "_")
    if rule not in TARGET_RULES:
        raise ValueError(
            f"target_rule must be one of {TARGET_RULES}, received {value!r}"
        )
    return cast(TargetRule, rule)


def _validate_rule_width(target_rule: TargetRule, k: int) -> int:
    if isinstance(k, (bool, np.bool_)) or int(k) != k or int(k) < 1:
        raise ValueError("k must be a positive integer")
    k = int(k)
    if target_rule == "majority" and k % 2 == 0:
        raise ValueError("majority target_rule requires odd k")
    if target_rule == "multiplexer" and k not in (3, 6):
        raise ValueError("multiplexer target_rule requires k=3 or k=6")
    return k


def _rule_matrix(value: ArrayLike, name: str = "code") -> SignArray:
    array = np.asarray(value)
    if array.ndim != 2 or array.shape[1] < 1:
        raise ValueError(f"{name} must have shape [n, k] with k >= 1, got {array.shape}")
    if not np.all(np.isin(array, (-1, 1))):
        raise ValueError(f"{name} must contain only -1 and +1")
    return np.array(array, dtype=np.int8, copy=True)


def decode_target_rule(
    code: ArrayLike,
    target_rule: TargetRule | str = "parity",
) -> SignArray:
    """Decode one exact rule from signed ``R`` channels.

    Majority is defined only at odd width, so ties are impossible. Signed
    conjunction is positive exactly when every input is positive. For a
    multiplexer, the first ``a`` signs are address bits (``-1 -> 0``, ``+1 ->
    1``; most-significant bit first) and the remaining ``2**a`` signs are data
    bits. The supported widths are therefore three and six.
    """

    rule = _target_rule(target_rule)
    values = _rule_matrix(code)
    k = _validate_rule_width(rule, values.shape[1])
    if rule == "parity":
        return np.prod(values, axis=1, dtype=np.int8)
    if rule == "majority":
        return np.where(values.sum(axis=1) > 0, 1, -1).astype(np.int8)
    if rule == "conjunction":
        return np.where(np.all(values == 1, axis=1), 1, -1).astype(np.int8)

    address_bits = 1 if k == 3 else 2
    weights = 1 << np.arange(address_bits - 1, -1, -1, dtype=np.int64)
    addresses = ((values[:, :address_bits] > 0).astype(np.int64) * weights).sum(axis=1)
    data = values[:, address_bits:]
    return data[np.arange(len(values)), addresses].astype(np.int8, copy=False)


def _conditional_rule_code(
    y: SignArray,
    k: int,
    target_rule: TargetRule,
    generator: np.random.Generator,
) -> SignArray:
    """Draw a rule code conditional on an already balanced target vector."""

    if target_rule == "parity":
        # Keep the legacy path and RNG consumption exactly unchanged.
        return degree_k_code(y, k, rng=generator)

    if target_rule == "multiplexer":
        code = generator.choice(
            np.asarray([-1, 1], dtype=np.int8), size=(len(y), k)
        )
        address_bits = 1 if k == 3 else 2
        weights = 1 << np.arange(address_bits - 1, -1, -1, dtype=np.int64)
        addresses = (
            (code[:, :address_bits] > 0).astype(np.int64) * weights
        ).sum(axis=1)
        code[np.arange(len(y)), address_bits + addresses] = y
        return code.astype(np.int8, copy=False)

    if target_rule == "conjunction":
        code = np.ones((len(y), k), dtype=np.int8)
        negative = np.flatnonzero(y == -1)
        if len(negative):
            draws = generator.choice(
                np.asarray([-1, 1], dtype=np.int8), size=(len(negative), k)
            )
            invalid = np.all(draws == 1, axis=1)
            while np.any(invalid):
                draws[invalid] = generator.choice(
                    np.asarray([-1, 1], dtype=np.int8),
                    size=(int(np.sum(invalid)), k),
                )
                invalid = np.all(draws == 1, axis=1)
            code[negative] = draws
        return code

    code = np.empty((len(y), k), dtype=np.int8)
    remaining = np.arange(len(y), dtype=np.int64)
    while len(remaining):
        draws = generator.choice(
            np.asarray([-1, 1], dtype=np.int8), size=(len(remaining), k)
        )
        accepted = decode_target_rule(draws, "majority") == y[remaining]
        code[remaining[accepted]] = draws[accepted]
        remaining = remaining[~accepted]
    return code


def exact_rule_code(
    y: ArrayLike,
    k: int,
    seed: int = 0,
    *,
    target_rule: TargetRule | str = "parity",
    rng: np.random.Generator | None = None,
) -> SignArray:
    """Construct deterministic signed channels that exactly decode to ``y``.

    Targets are supplied rather than derived from random channels so callers can
    balance ``Y`` first. Conditional code generation then changes no proxy,
    state, identifier, or padding stream.
    """

    signs = _require_signs(y, "y")
    rule = _target_rule(target_rule)
    width = _validate_rule_width(rule, k)
    generator = rng if rng is not None else np.random.default_rng(seed)
    code = _conditional_rule_code(signs, width, rule, generator)
    if not np.array_equal(decode_target_rule(code, rule), signs):  # pragma: no cover
        raise RuntimeError(f"internal error constructing exact {rule} code")
    return code


make_exact_rule_code = exact_rule_code


@dataclass(frozen=True)
class FeatureSpec:
    """Column names and active/inactive status for a feature matrix."""

    names: tuple[str, ...]
    active: NDArray[np.bool_]

    def __post_init__(self) -> None:
        active = np.asarray(self.active, dtype=bool)
        if active.ndim != 1 or len(active) != len(self.names):
            raise ValueError("FeatureSpec.active must align with names")
        object.__setattr__(self, "active", np.array(active, copy=True))


@dataclass(frozen=True)
class SemanticBatch:
    """A batch with observations, semantic targets, and persistent-noise IDs.

    ``channels`` contains model-visible named signals.  ``latents`` contains
    analysis-only variables such as H9's environment context and H4's conflict
    mask.  Neither latents nor IDs are included by :meth:`features`.

    The canonical fixed-width prefix is ``[P, P_present, R_1, ..., R_max]``.
    Inactive R columns are deterministic signs keyed by ``state_id``; this keeps
    input width constant across the complexity sweep without leaking which rows
    were generated together.  Use :meth:`feature_spec` to obtain the active mask.
    """

    y: ArrayLike
    channels: Mapping[str, ArrayLike]
    target: ArrayLike | None = None
    reward: ArrayLike | None = None
    state: ArrayLike | None = None
    sample_id: ArrayLike | None = None
    state_id: ArrayLike | None = None
    episode_id: ArrayLike | None = None
    step_id: ArrayLike | None = None
    latents: Mapping[str, ArrayLike] = field(default_factory=dict)
    nuisance_targets: ArrayLike | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        y = _require_signs(self.y, "y")
        n = len(y)
        target = y.copy() if self.target is None else _as_vector(self.target, "target", n)
        reward = target.copy() if self.reward is None else _as_vector(self.reward, "reward", n)
        channels = _freeze_mapping(self.channels, n, name="channel")
        latents = _freeze_mapping(self.latents, n, name="latent")

        sample_id = np.arange(n, dtype=np.int64) if self.sample_id is None else _as_vector(
            self.sample_id, "sample_id", n
        ).astype(np.int64, copy=False)
        state_id = sample_id.copy() if self.state_id is None else _as_vector(
            self.state_id, "state_id", n
        ).astype(np.int64, copy=False)
        episode_id = sample_id.copy() if self.episode_id is None else _as_vector(
            self.episode_id, "episode_id", n
        ).astype(np.int64, copy=False)
        step_id = np.zeros(n, dtype=np.int64) if self.step_id is None else _as_vector(
            self.step_id, "step_id", n
        ).astype(np.int64, copy=False)
        if any(np.any(ids < 0) for ids in (sample_id, state_id, episode_id, step_id)):
            raise ValueError("sample/state/episode/step IDs must be non-negative")

        state = None if self.state is None else _as_rows(self.state, "state", n).astype(
            np.float32, copy=False
        )
        nuisance = None
        if self.nuisance_targets is not None:
            nuisance = _as_rows(self.nuisance_targets, "nuisance_targets", n)

        metadata = dict(self.metadata)
        active_k = int(metadata.get("active_k", self._infer_active_k(channels)))
        masked_channels = frozenset(str(name) for name in self.metadata.get("masked_channels", ()))
        if active_k < 0:
            raise ValueError("metadata.active_k must be non-negative")
        for index in range(1, active_k + 1):
            key = f"R_{index}"
            if key not in channels:
                raise ValueError(f"active interaction code is missing {key}")
            if key not in masked_channels:
                _require_signs(channels[key], f"channels[{key!r}]", n)
        configured_rule: str = str(metadata.get("target_rule", "parity"))
        if active_k:
            normalized_rule = _target_rule(configured_rule)
            _validate_rule_width(normalized_rule, active_k)
            configured_rule = normalized_rule
            metadata["target_rule"] = configured_rule
        if active_k and not masked_channels and bool(metadata.get("enforce_exact_code", False)):
            code = np.column_stack([channels[f"R_{i}"] for i in range(1, active_k + 1)])
            if not np.array_equal(decode_target_rule(code, configured_rule), y):
                raise ValueError(
                    "active interaction channels must decode to Y under "
                    f"target_rule={configured_rule!r}"
                )

        metadata["active_k"] = active_k
        configured_max = int(metadata.get("max_k", active_k))
        if configured_max < active_k:
            raise ValueError("metadata.max_k cannot be smaller than active_k")
        metadata["max_k"] = configured_max
        metadata["masked_channels"] = tuple(sorted(masked_channels))
        metadata.setdefault("r_active_mask", tuple(i < active_k for i in range(configured_max)))
        metadata.setdefault("padding_seed", 0)

        object.__setattr__(self, "y", y)
        object.__setattr__(self, "target", target)
        object.__setattr__(self, "reward", reward)
        object.__setattr__(self, "channels", channels)
        object.__setattr__(self, "latents", latents)
        object.__setattr__(self, "sample_id", np.array(sample_id, copy=True))
        object.__setattr__(self, "state_id", np.array(state_id, copy=True))
        object.__setattr__(self, "episode_id", np.array(episode_id, copy=True))
        object.__setattr__(self, "step_id", np.array(step_id, copy=True))
        object.__setattr__(self, "state", None if state is None else np.array(state, copy=True))
        object.__setattr__(
            self, "nuisance_targets", None if nuisance is None else np.array(nuisance, copy=True)
        )
        object.__setattr__(self, "metadata", MappingProxyType(metadata))

    @staticmethod
    def _infer_active_k(channels: Mapping[str, NDArray[Any]]) -> int:
        indices = sorted(
            int(match.group(1))
            for name in channels
            if (match := _R_CHANNEL.fullmatch(name)) is not None
        )
        if not indices:
            return 0
        if indices != list(range(1, indices[-1] + 1)):
            raise ValueError("R channels must form the contiguous sequence R_1,...,R_k")
        return indices[-1]

    def __len__(self) -> int:
        return len(self.y)

    @property
    def n(self) -> int:
        return len(self)

    @property
    def active_k(self) -> int:
        return int(self.metadata["active_k"])

    @property
    def target_rule(self) -> str:
        return str(self.metadata.get("target_rule", "parity"))

    @property
    def class_labels(self) -> IntArray:
        """Zero-based class labels derived from the (possibly noisy) target."""

        return (np.asarray(self.target) > 0).astype(np.int64)

    @property
    def target_class(self) -> IntArray:
        return self.class_labels

    @property
    def y_class(self) -> IntArray:
        return (self.y > 0).astype(np.int64)

    @property
    def proxy(self) -> NDArray[Any]:
        return self.channels["P"]

    @property
    def p(self) -> NDArray[Any]:
        return self.proxy

    @property
    def code(self) -> SignArray:
        if self.active_k == 0:
            return np.empty((len(self), 0), dtype=np.int8)
        return np.column_stack(
            [self.channels[f"R_{index}"] for index in range(1, self.active_k + 1)]
        ).astype(np.int8, copy=False)

    @property
    def r(self) -> SignArray:
        return self.code

    def channel(self, name: str, *, include_latent: bool = False) -> NDArray[Any]:
        """Return one exactly named channel; no prefix or substring matching is used."""

        if name in self.channels:
            return self.channels[name]
        if include_latent and name in self.latents:
            return self.latents[name]
        raise KeyError(f"unknown {'channel or latent' if include_latent else 'channel'} {name!r}")

    def __getattr__(self, name: str) -> NDArray[Any]:
        # Convenient mathematical access (batch.P, batch.P0, batch.P_E, batch.C)
        # while retaining an explicit mapping as the source of truth.
        channels = self.__dict__.get("channels", {})
        if name in channels:
            return channels[name]
        latents = self.__dict__.get("latents", {})
        if name in latents:
            return latents[name]
        raise AttributeError(name)

    def _padding(self, index: int) -> SignArray:
        if bool(self.metadata.get("neutralize_padding", False)):
            return np.zeros(len(self), dtype=np.int8)
        return _stateless_signs(
            np.asarray(self.state_id, dtype=np.int64), int(self.metadata["padding_seed"]), f"R_{index}"
        )

    def _feature_parts(
        self, max_k: int | None, include_state: bool
    ) -> tuple[list[str], list[NDArray[Any]], list[bool]]:
        if max_k is None:
            max_k = int(self.metadata.get("max_k", self.active_k))
        if isinstance(max_k, (bool, np.bool_)) or int(max_k) != max_k:
            raise ValueError("max_k must be an integer")
        max_k = int(max_k)
        if max_k < self.active_k:
            raise ValueError(f"max_k={max_k} is smaller than active k={self.active_k}")

        p = self.channels.get("P", np.zeros(len(self), dtype=np.float32))
        present = self.channels.get(
            "P_present", np.ones(len(self), dtype=np.float32) if "P" in self.channels else np.zeros(len(self))
        )
        names = ["P", "P_present"]
        parts: list[NDArray[Any]] = [np.asarray(p)[:, None], np.asarray(present)[:, None]]
        active = ["P" in self.channels, "P_present" in self.channels or "P" in self.channels]

        for index in range(1, max_k + 1):
            name = f"R_{index}"
            values = self.channels[name] if index <= self.active_k else self._padding(index)
            names.append(name)
            parts.append(np.asarray(values)[:, None])
            active.append(index <= self.active_k and name not in self.metadata.get("masked_channels", ()))

        canonical = {"P", "P_present", *(f"R_{i}" for i in range(1, self.active_k + 1))}
        for name, value in self.channels.items():
            if name in canonical:
                continue
            array = np.asarray(value)
            if array.ndim == 1:
                names.append(name)
                parts.append(array[:, None])
                active.append(True)
            else:
                for column in range(array.shape[1]):
                    names.append(f"{name}_{column}")
                    parts.append(array[:, column : column + 1])
                    active.append(True)
        if include_state and self.state is not None:
            assert self.state.ndim == 2
            for column in range(self.state.shape[1]):
                names.append(f"state_{column}")
                parts.append(self.state[:, column : column + 1])
                active.append(True)
        return names, parts, active

    def features(
        self,
        max_k: int | None = None,
        include_state: bool = True,
        *,
        dtype: np.dtype[Any] | type[np.floating[Any]] = np.float32,
    ) -> NDArray[Any]:
        """Convert named observations to a stable two-dimensional feature array."""

        _, parts, _ = self._feature_parts(max_k, include_state)
        return np.concatenate(parts, axis=1).astype(dtype, copy=False)

    @property
    def x(self) -> FloatArray:
        return self.features().astype(np.float32, copy=False)

    def feature_spec(self, max_k: int | None = None, include_state: bool = True) -> FeatureSpec:
        names, _, active = self._feature_parts(max_k, include_state)
        return FeatureSpec(tuple(names), np.asarray(active, dtype=bool))

    def feature_names(self, max_k: int | None = None, include_state: bool = True) -> tuple[str, ...]:
        return self.feature_spec(max_k, include_state).names

    def feature_active_mask(
        self, max_k: int | None = None, include_state: bool = True
    ) -> NDArray[np.bool_]:
        return self.feature_spec(max_k, include_state).active.copy()

    def with_updates(self, **changes: Any) -> "SemanticBatch":
        """Functional update helper used by interventions and noise processes."""

        return replace(self, **changes)

    def select(self, indices: ArrayLike) -> "SemanticBatch":
        """Return a row subset while preserving semantic and feature metadata."""

        selection = np.asarray(indices)
        return SemanticBatch(
            y=self.y[selection],
            target=np.asarray(self.target)[selection],
            reward=np.asarray(self.reward)[selection],
            channels={name: values[selection] for name, values in self.channels.items()},
            latents={name: values[selection] for name, values in self.latents.items()},
            state=None if self.state is None else self.state[selection],
            sample_id=np.asarray(self.sample_id)[selection],
            state_id=np.asarray(self.state_id)[selection],
            episode_id=np.asarray(self.episode_id)[selection],
            step_id=np.asarray(self.step_id)[selection],
            nuisance_targets=None
            if self.nuisance_targets is None
            else self.nuisance_targets[selection],
            metadata=self.metadata,
        )


def _state_features(state_ids: IntArray, state_dim: int, seed: int) -> FloatArray | None:
    if isinstance(state_dim, (bool, np.bool_)) or int(state_dim) != state_dim or int(state_dim) < 0:
        raise ValueError("state_dim must be a non-negative integer")
    state_dim = int(state_dim)
    if state_dim == 0:
        return None
    columns = [_stateless_signs(state_ids, seed, f"state_{j}") for j in range(state_dim)]
    return np.column_stack(columns).astype(np.float32)


def _stratified_conflict_indices(y: SignArray, count: int, rng: np.random.Generator) -> IntArray:
    """Choose an exact count, as evenly split across Y signs as arithmetic permits."""

    if count == 0:
        return np.empty(0, dtype=np.int64)
    negative = np.flatnonzero(y == -1)
    positive = np.flatnonzero(y == 1)
    rng.shuffle(negative)
    rng.shuffle(positive)
    negative_count = count // 2
    positive_count = count - negative_count
    # For very high conflict rates, one side may need the odd extra item.
    if negative_count > len(negative) or positive_count > len(positive):
        negative_count = min(len(negative), count - min(len(positive), count))
        positive_count = count - negative_count
    selected = np.concatenate((negative[:negative_count], positive[:positive_count]))
    rng.shuffle(selected)
    return selected.astype(np.int64, copy=False)


def _base_batch(
    n: int,
    q: float,
    k: int,
    seed: int,
    *,
    max_k: int | None,
    state_dim: int,
    split: str,
    id_offset: int,
    target_rule: TargetRule | str = "parity",
) -> SemanticBatch:
    n = _validate_n(n)
    n_conflict, realized_q = validate_proxy_confounds(n, q=q)
    if isinstance(k, (bool, np.bool_)) or int(k) != k or int(k) < 1:
        raise ValueError("k must be a positive integer")
    k = int(k)
    rule = _target_rule(target_rule)
    _validate_rule_width(rule, k)
    max_k = k if max_k is None else int(max_k)
    if max_k < k:
        raise ValueError("max_k must be at least k")

    y = balanced_signs(n, seed)
    # Independent deterministic streams make q a matched intervention: changing
    # proxy reliability does not silently redraw the exact R code. The shuffled
    # sign-specific prefixes also make conflict sets nested as q decreases.
    conflict_rng = np.random.default_rng(_stable_seed(seed, "proxy-conflicts"))
    code_rng = np.random.default_rng(_stable_seed(seed, "interaction-code"))
    conflicts = _stratified_conflict_indices(y, n_conflict, conflict_rng)
    conflict_mask = np.zeros(n, dtype=bool)
    conflict_mask[conflicts] = True
    p = np.where(conflict_mask, -y, y).astype(np.int8)
    code = exact_rule_code(y, k, target_rule=rule, rng=code_rng)
    channels: dict[str, ArrayLike] = {
        "P": p,
        "P_present": np.ones(n, dtype=np.int8),
    }
    channels.update({f"R_{index + 1}": code[:, index] for index in range(k)})
    ids = np.arange(id_offset, id_offset + n, dtype=np.int64)
    state = _state_features(ids, state_dim, _stable_seed(seed, split, "state"))
    return SemanticBatch(
        y=y,
        target=y,
        reward=y.astype(np.float32),
        channels=channels,
        latents={"is_conflict": conflict_mask},
        state=state,
        sample_id=ids,
        state_id=ids,
        episode_id=ids,
        step_id=np.zeros(n, dtype=np.int64),
        metadata={
            "dataset": "standard",
            "split": split,
            "seed": int(seed),
            "q": realized_q,
            "realized_q": realized_q,
            "n_conflict": n_conflict,
            "active_k": k,
            "target_rule": rule,
            "max_k": max_k,
            "r_active_mask": tuple(i < k for i in range(max_k)),
            "padding_seed": _stable_seed(seed, split, "padding"),
            "enforce_exact_code": True,
            "state_dim": state_dim,
        },
    )


def make_standard_dataset(
    n: int,
    q: float,
    k: int,
    seed: int = 0,
    *,
    max_k: int | None = None,
    state_dim: int = 0,
    split: str = "train",
    id_offset: int = 0,
    target_rule: TargetRule | str = "parity",
) -> SemanticBatch:
    """Build the primary H1--H3 dataset with exact realized proxy accuracy."""

    return _base_batch(
        n,
        q,
        k,
        seed,
        max_k=max_k,
        state_dim=state_dim,
        split=split,
        id_offset=id_offset,
        target_rule=target_rule,
    )


def make_conflict_dataset(
    n: int,
    k: int,
    seed: int = 0,
    *,
    max_k: int | None = None,
    state_dim: int = 0,
    id_offset: int = 1_000_000_000,
    target_rule: TargetRule | str = "parity",
) -> SemanticBatch:
    """Build a balanced OOD evaluation set satisfying ``P=-Y`` on every row."""

    return _base_batch(
        n,
        0.0,
        k,
        seed,
        max_k=max_k,
        state_dim=state_dim,
        split="conflict_eval",
        id_offset=id_offset,
        target_rule=target_rule,
    )


@dataclass(frozen=True)
class DatasetBundle:
    """Paired training/evaluation data with tuple- and key-style access."""

    train: SemanticBatch
    conflict_eval: SemanticBatch

    @property
    def evaluation(self) -> SemanticBatch:
        return self.conflict_eval

    @property
    def eval(self) -> SemanticBatch:
        return self.conflict_eval

    def __iter__(self) -> Iterator[SemanticBatch]:
        yield self.train
        yield self.conflict_eval

    def __getitem__(self, key: int | str) -> SemanticBatch:
        if key in (0, "train"):
            return self.train
        if key in (1, "eval", "evaluation", "conflict_eval"):
            return self.conflict_eval
        raise KeyError(key)


def make_standard_train_eval(
    n_train: int,
    n_eval: int,
    q: float,
    k: int,
    seed: int = 0,
    *,
    max_k: int | None = None,
    state_dim: int = 0,
    target_rule: TargetRule | str = "parity",
) -> DatasetBundle:
    """Create independent standard training and all-conflict evaluation batches."""

    train = make_standard_dataset(
        n_train,
        q,
        k,
        seed,
        max_k=max_k,
        state_dim=state_dim,
        split="train",
        target_rule=target_rule,
    )
    evaluation = make_conflict_dataset(
        n_eval,
        k,
        seed + 1,
        max_k=max_k,
        state_dim=state_dim,
        id_offset=n_train + 1,
        target_rule=target_rule,
    )
    return DatasetBundle(train, evaluation)


def intervene(
    batch: SemanticBatch,
    *,
    flip: Iterable[str] = (),
    mask: Iterable[str] = (),
    mask_value: float = 0.0,
) -> SemanticBatch:
    """Flip or mask exactly named observation channels.

    Names are checked by exact dictionary membership: intervening on ``P`` never
    changes ``P0``, ``P1``, or ``P_present``.  To model removal with an explicit
    missingness indicator, request ``mask=("P", "P_present")`` or use
    :func:`remove_proxy`.
    """

    flips = tuple(flip)
    masks = tuple(mask)
    if len(set(flips)) != len(flips) or len(set(masks)) != len(masks):
        raise ValueError("intervention channel names must not be repeated")
    overlap = set(flips) & set(masks)
    if overlap:
        raise ValueError(f"channels cannot be both flipped and masked: {sorted(overlap)}")
    unknown = (set(flips) | set(masks)) - set(batch.channels)
    if unknown:
        raise KeyError(f"unknown intervention channels: {sorted(unknown)}")

    channels = {name: np.array(value, copy=True) for name, value in batch.channels.items()}
    for name in flips:
        channels[name] = -channels[name]
    for name in masks:
        dtype = np.result_type(channels[name].dtype, type(mask_value))
        channels[name] = np.full(channels[name].shape, mask_value, dtype=dtype)
    metadata = dict(batch.metadata)
    history = list(metadata.get("interventions", ()))
    history.append({"flip": flips, "mask": masks, "mask_value": float(mask_value)})
    metadata["interventions"] = tuple(history)
    masked = set(str(name) for name in metadata.get("masked_channels", ()))
    masked.update(masks)
    masked.difference_update(flips)
    metadata["masked_channels"] = tuple(sorted(masked))
    if any(_R_CHANNEL.fullmatch(name) for name in (*flips, *masks)):
        metadata["enforce_exact_code"] = False
    return batch.with_updates(channels=channels, metadata=metadata)


def flip_channels(batch: SemanticBatch, channels: str | Sequence[str]) -> SemanticBatch:
    names = (channels,) if isinstance(channels, str) else tuple(channels)
    return intervene(batch, flip=names)


def flip_exact_rule_output(batch: SemanticBatch) -> SemanticBatch:
    """Minimally change active ``R`` coordinates so their rule decodes to ``-Y``.

    A coordinate-wise sign flip is a comparable rule intervention for parity,
    but not for majority, conjunction, or a multiplexer.  This counterfactual
    instead changes the smallest deterministic set of active coordinates needed
    to reverse the configured rule on *every* row.  The changed-coordinate
    distribution is retained in metadata because the required Hamming distance
    is itself rule dependent.

    The input batch must be an unmasked exact-rule batch: its configured decoder
    must currently equal ``Y`` on every row.  Targets, proxy, state, IDs, padding,
    and all non-``R`` channels are preserved.
    """

    if batch.active_k < 1:
        raise ValueError("an exact-rule counterfactual requires at least one active R channel")
    rule = _target_rule(batch.target_rule)
    code = batch.code.copy()
    before = decode_target_rule(code, rule)
    intended = np.asarray(batch.y, dtype=np.int8)
    if not np.array_equal(before, intended):
        raise ValueError(
            "exact-rule counterfactual requires the configured rule to decode to Y "
            "on every row"
        )

    if rule == "parity":
        code[:, 0] *= -1
    elif rule == "multiplexer":
        address_bits = 1 if batch.active_k == 3 else 2
        weights = 1 << np.arange(address_bits - 1, -1, -1, dtype=np.int64)
        addresses = (
            (code[:, :address_bits] > 0).astype(np.int64) * weights
        ).sum(axis=1)
        code[np.arange(len(batch)), address_bits + addresses] *= -1
    elif rule == "conjunction":
        positive = intended == 1
        # Positive conjunction rows are all +1, so one fixed bit suffices.
        code[positive, 0] = -1
        # A negative conjunction becomes positive only after every -1 is
        # changed.  This is the unique minimum-Hamming counterfactual set.
        negative_rows = np.flatnonzero(~positive)
        if len(negative_rows):
            negative_values = code[negative_rows]
            negative_values[negative_values == -1] = 1
            code[negative_rows] = negative_values
    else:
        # For odd-width majority, flipping a minimal number of coordinates
        # currently aligned with the output moves the signed margin strictly
        # across zero.  Ties are impossible by construction.
        margins = code.sum(axis=1, dtype=np.int64)
        required = ((np.abs(margins) + 1) // 2).astype(np.int64)
        for row, count in enumerate(required):
            candidates = np.flatnonzero(code[row] == intended[row])
            code[row, candidates[: int(count)]] *= -1

    after = decode_target_rule(code, rule)
    if not np.array_equal(after, -intended):  # pragma: no cover - construction proof
        raise RuntimeError("exact-rule counterfactual failed to reverse the decoded rule")
    original = batch.code
    changed = code != original
    hamming = changed.sum(axis=1, dtype=np.int64)
    channel_rates = changed.mean(axis=0)

    channels = {name: np.array(values, copy=True) for name, values in batch.channels.items()}
    for index in range(batch.active_k):
        channels[f"R_{index + 1}"] = code[:, index]
    metadata = dict(batch.metadata)
    history = list(metadata.get("interventions", ()))
    history.append({"counterfactual": "flip_exact_rule_output"})
    metadata["interventions"] = tuple(history)
    metadata["enforce_exact_code"] = False
    metadata["exact_rule_counterfactual"] = {
        "target_rule": rule,
        "decoded_before": "Y",
        "decoded_after": "-Y",
        "minimal_hamming": True,
        "hamming_mean": float(np.mean(hamming)),
        "hamming_median": float(np.median(hamming)),
        "hamming_min": int(np.min(hamming)),
        "hamming_max": int(np.max(hamming)),
        "hamming_total": int(np.sum(hamming)),
        "per_channel_flip_rate": tuple(float(value) for value in channel_rates),
        "n": len(batch),
    }
    return batch.with_updates(channels=channels, metadata=metadata)


def mask_channels(
    batch: SemanticBatch, channels: str | Sequence[str], *, value: float = 0.0
) -> SemanticBatch:
    names = (channels,) if isinstance(channels, str) else tuple(channels)
    return intervene(batch, mask=names, mask_value=value)


def remove_proxy(batch: SemanticBatch, name: str = "P") -> SemanticBatch:
    """Mask a proxy and its presence indicator when the latter exists."""

    names = [name]
    indicator = f"{name}_present"
    if indicator in batch.channels:
        names.append(indicator)
    return mask_channels(batch, names)


def _prototype_assignments(count: int, unique: int, rng: np.random.Generator) -> IntArray:
    if count == 0:
        return np.empty(0, dtype=np.int64)
    assignments = np.arange(count, dtype=np.int64) % unique
    rng.shuffle(assignments)
    return assignments


def _validate_h4_types(values: Sequence[int | str], name: str) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)):
        raise ValueError(f"{name} must be a non-empty sequence of mechanism names")
    try:
        raw = tuple(values)
    except TypeError as exc:
        raise ValueError(f"{name} must be a non-empty sequence of mechanism names") from exc
    if not raw:
        raise ValueError(f"{name} must not be empty")
    result: list[str] = []
    for value in raw:
        if isinstance(value, (bool, np.bool_)):
            raise ValueError(f"{name} must contain known mechanism names or IDs 0--3")
        if isinstance(value, (int, np.integer)):
            index = int(value)
            if not 0 <= index < len(H4_FAILURE_MECHANISMS):
                raise ValueError(f"{name} contains unknown mechanism ID {index}")
            result.append(H4_FAILURE_MECHANISMS[index])
        elif isinstance(value, str) and value in H4_FAILURE_MECHANISMS:
            result.append(value)
        else:
            raise ValueError(
                f"{name} contains unknown mechanism {value!r}; expected one of "
                f"{H4_FAILURE_MECHANISMS}"
            )
    if len(set(result)) != len(result):
        raise ValueError(f"{name} must not contain duplicate mechanisms")
    return tuple(result)


def _apply_h4_failure_mechanisms(
    context: list[FloatArray],
    state: FloatArray | None,
    positions: IntArray,
    mechanisms: NDArray[np.int64],
) -> tuple[list[FloatArray], FloatArray | None]:
    """Apply four disjoint, invertible context-corruption families.

    Every family acts on a different geometric operation.  Because each map is
    invertible, it preserves within-family context diversity; an unseen-family
    test therefore probes transfer across corruption semantics, not accidental
    information loss or arbitrary numeric type tags.
    """

    transformed = [np.array(field, copy=True) for field in context]
    transformed_state = None if state is None else np.array(state, copy=True)
    for mechanism_id, mechanism_name in enumerate(H4_FAILURE_MECHANISMS):
        rows = positions[mechanisms == mechanism_id]
        if not len(rows):
            continue
        if mechanism_name == "location_reflection":
            transformed[0][rows] *= -1
            if transformed_state is not None and transformed_state.shape[1] >= 1:
                transformed_state[rows, 0] *= -1
        elif mechanism_name == "coordinate_exchange":
            x = transformed[0][rows].copy()
            transformed[0][rows] = transformed[1][rows]
            transformed[1][rows] = x
            if transformed_state is not None and transformed_state.shape[1] >= 2:
                first = transformed_state[rows, 0].copy()
                transformed_state[rows, 0] = transformed_state[rows, 1]
                transformed_state[rows, 1] = first
        elif mechanism_name == "geometry_rotation":
            geometry = transformed[2][rows].copy()
            transformed[2][rows] = -transformed[3][rows]
            transformed[3][rows] = geometry
            if transformed_state is not None and transformed_state.shape[1] >= 4:
                third = transformed_state[rows, 2].copy()
                transformed_state[rows, 2] = -transformed_state[rows, 3]
                transformed_state[rows, 3] = third
        elif mechanism_name == "nuisance_inversion":
            transformed[3][rows] *= -1
            if transformed_state is not None and transformed_state.shape[1] >= 1:
                transformed_state[rows, -1] *= -1
    return transformed, transformed_state


def _h4_prototype_allocation(labels: SignArray, unique: int) -> dict[int, int]:
    """Allocate context prototypes across represented target signs.

    Context IDs are target-stratified so a repeated state never receives
    contradictory labels. We therefore require at least one prototype for every
    represented target sign and allocate additional prototypes approximately in
    proportion to presentation counts. The row-wise Y/P/R draw itself is unchanged.
    """

    counts = {
        sign: int(np.sum(np.asarray(labels, dtype=np.int8) == sign))
        for sign in (-1, 1)
        if np.any(np.asarray(labels, dtype=np.int8) == sign)
    }
    if unique < len(counts):
        raise ValueError(
            "u_conflict must provide at least one context prototype per represented "
            f"target sign; need {len(counts)}, got {unique}"
        )
    allocation = {sign: 1 for sign in counts}
    remaining = unique - len(allocation)
    while remaining:
        candidates = [sign for sign, count in counts.items() if allocation[sign] < count]
        if not candidates:  # pragma: no cover - guarded by u_conflict <= n_conflict.
            raise RuntimeError("internal error allocating H4 conflict prototypes")
        sign = max(
            candidates,
            key=lambda item: (counts[item] / allocation[item], -item),
        )
        allocation[sign] += 1
        remaining -= 1
    return allocation


def _h4_prototype_plan(
    batch: SemanticBatch,
    conflict_positions: IntArray,
    unique: int,
    seed: int,
) -> tuple[IntArray, IntArray]:
    """Select a condition-independent master-bank subset and row assignments."""

    if not len(conflict_positions):
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)
    labels = np.asarray(batch.y, dtype=np.int8)[conflict_positions]
    allocation = _h4_prototype_allocation(labels, unique)
    sources: list[int] = []
    assignments = np.empty(len(conflict_positions), dtype=np.int64)
    for sign in (-1, 1):
        if sign not in allocation:
            continue
        local_rows = np.flatnonzero(labels == sign).astype(np.int64)
        source_rng = np.random.default_rng(_stable_seed(seed, "h4-prototype-order", sign))
        ordered = local_rows[source_rng.permutation(len(local_rows))]
        count = allocation[sign]
        offset = len(sources)
        sources.extend(int(conflict_positions[index]) for index in ordered[:count])
        assignment_rng = np.random.default_rng(
            _stable_seed(seed, "h4-prototype-assignment", sign)
        )
        assignments[local_rows] = (
            _prototype_assignments(len(local_rows), count, assignment_rng) + offset
        )
    return np.asarray(sources, dtype=np.int64), assignments


def _attach_h4_context(
    batch: SemanticBatch,
    *,
    condition: H4Condition,
    conflict_positions: IntArray,
    prototype_assignments: IntArray,
    prototype_source_positions: IntArray,
    prototype_ids: IntArray,
    state_dim: int,
    context_seed: int,
    split: str,
    structured_types: tuple[str, ...] | None,
) -> SemanticBatch:
    n = len(batch)
    if len(prototype_assignments) != len(conflict_positions):
        raise ValueError("H4 prototype assignments must align with conflict rows")
    if len(prototype_source_positions) != len(prototype_ids):
        raise ValueError("H4 prototype sources and IDs must have equal length")
    if len(prototype_assignments) and (
        int(prototype_assignments.min()) < 0
        or int(prototype_assignments.max()) >= len(prototype_source_positions)
    ):
        raise ValueError("H4 prototype assignment is out of range")

    state_ids = np.asarray(batch.state_id).copy()
    conflict_prototype = np.full(n, -1, dtype=np.int64)
    prototype_source_row = np.full(n, -1, dtype=np.int64)
    channels = {name: np.array(value, copy=True) for name, value in batch.channels.items()}
    if len(conflict_positions):
        assigned_ids = prototype_ids[prototype_assignments]
        assigned_sources = prototype_source_positions[prototype_assignments]
        if not np.array_equal(
            np.asarray(batch.y)[conflict_positions], np.asarray(batch.y)[assigned_sources]
        ):
            raise RuntimeError("internal error: H4 prototype assignment changed target signs")
        state_ids[conflict_positions] = assigned_ids
        conflict_prototype[conflict_positions] = assigned_ids
        prototype_source_row[conflict_positions] = np.asarray(batch.sample_id)[assigned_sources]

    # The salt is deliberately independent of condition and U. Agreement rows and
    # any shared master-bank prototype therefore have identical context features in
    # every arm. P and R deliberately remain the condition-independent base draw:
    # H4 intervenes on context diversity, not the semantic task signals.
    state = _state_features(state_ids, state_dim, _stable_seed(context_seed, "h4-state"))
    conflict_type = np.full(n, -1, dtype=np.int64)
    conflict_type_present = np.zeros(n, dtype=np.int8)
    context_names = ("location_x", "location_y", "geometry", "nuisance")
    context = list(_h4_context_fields(state_ids, context_seed))
    if condition == "structured_holdout" and len(conflict_positions):
        assert structured_types is not None
        type_ids = np.asarray(
            [H4_FAILURE_MECHANISMS.index(name) for name in structured_types],
            dtype=np.int64,
        )
        prototype_labels = np.asarray(batch.y)[prototype_source_positions]
        type_by_prototype = np.full(len(prototype_ids), -1, dtype=np.int64)
        for sign in np.unique(prototype_labels):
            sign_prototypes = np.flatnonzero(prototype_labels == sign)
            if len(sign_prototypes) < len(type_ids):
                raise ValueError(
                    "structured_holdout requires at least one prototype per failure "
                    f"mechanism for each represented Y sign; sign {int(sign)} has "
                    f"{len(sign_prototypes)}, needs {len(type_ids)}"
                )
            # The identical cycle within each target stratum prevents mechanism
            # identity from becoming a target label while keeping prototypes pure.
            type_by_prototype[sign_prototypes] = np.resize(type_ids, len(sign_prototypes))
        conflict_type[conflict_positions] = type_by_prototype[prototype_assignments]
        conflict_type_present[conflict_positions] = 1
        context, state = _apply_h4_failure_mechanisms(
            context,
            state,
            conflict_positions,
            conflict_type[conflict_positions],
        )

    # Compact, deterministic context descriptors stand in for location, geometry,
    # and nuisance variations in the one-step task. The joint code is injective in
    # state ID. Structured arms additionally apply named invertible corruptions;
    # mechanism identity is analysis-only, so the model cannot solve the holdout
    # task by reading an arbitrary numeric tag.
    for name, values in zip(context_names, context, strict=True):
        channels[name] = values

    latents = dict(batch.latents)
    mask = np.zeros(n, dtype=bool)
    mask[conflict_positions] = True
    latents["is_conflict"] = mask
    latents["conflict_prototype_id"] = conflict_prototype
    latents["conflict_prototype_source_row"] = prototype_source_row
    latents["conflict_type"] = conflict_type
    latents["conflict_type_present"] = conflict_type_present
    for mechanism_id, mechanism_name in enumerate(H4_FAILURE_MECHANISMS):
        latents[f"conflict_mechanism_{mechanism_name}"] = (
            conflict_type == mechanism_id
        ).astype(np.int8)
    metadata = dict(batch.metadata)
    metadata.update(
        {
            "dataset": "h4",
            "condition": condition,
            "split": split,
            "n_conflict": int(len(conflict_positions)),
            "u_conflict": int(len(np.unique(prototype_assignments)))
            if len(prototype_assignments)
            else 0,
            "conflict_prototype_ids": tuple(int(x) for x in prototype_ids),
            "conflict_prototype_source_rows": tuple(
                int(np.asarray(batch.sample_id)[position])
                for position in prototype_source_positions
            ),
            "structured_types": tuple(structured_types or ()),
            "structured_type_ids": tuple(
                H4_FAILURE_MECHANISMS.index(name) for name in (structured_types or ())
            ),
            "failure_mechanism_registry": H4_FAILURE_MECHANISMS,
            "failure_mechanism_semantics": {
                "location_reflection": "reflect location_x and state coordinate 0",
                "coordinate_exchange": "exchange location_x/location_y and state coordinates 0/1",
                "geometry_rotation": "rotate geometry/nuisance and state coordinates 2/3",
                "nuisance_inversion": "invert nuisance and the final state coordinate",
            },
            "failure_mechanism_visible_to_model": False,
            "context_hash_family": "h4-condition-independent-injective-v2",
            "context_hash_seed": _stable_seed(context_seed, "h4-context-code-v2"),
            "context_channels": context_names,
            "context_transform": (
                "named_invertible_failure_mechanism"
                if condition == "structured_holdout"
                else "identity"
            ),
        }
    )
    return batch.with_updates(
        channels=channels, latents=latents, state_id=state_ids, state=state, metadata=metadata
    )


def make_h4_dataset(
    n: int,
    n_conflict: int,
    u_conflict: int,
    k: int,
    seed: int = 0,
    *,
    condition: H4Condition = "concentrated",
    q: float | None = None,
    n_eval: int | None = None,
    max_k: int | None = None,
    state_dim: int = 8,
    structured_train_types: Sequence[int | str] = (
        "location_reflection",
        "coordinate_exchange",
    ),
    structured_test_types: Sequence[int | str] = (
        "geometry_rotation",
        "nuisance_inversion",
    ),
) -> DatasetBundle:
    """Create H4 data with exact conflict count and unique conflict contexts.

    The evaluation set contains only conflicts and uses an ID namespace disjoint
    from *all* training states. Every arm shares the exact same row-wise Y, P, and
    R draw; only target-stratified context assignments vary with ``u_conflict``.
    All model-visible context fields repeat together within a prototype. In
    ``structured_holdout``, configured train/test conflict mechanisms apply
    named, invertible context corruptions while preserving the same target rule.
    Type identity is retained only as an analysis latent, never as a model input.
    """

    n = _validate_n(n)
    n_conflict, realized_q = validate_proxy_confounds(n, q=q, n_conflict=n_conflict)
    if isinstance(u_conflict, (bool, np.bool_)) or int(u_conflict) != u_conflict:
        raise ValueError("u_conflict must be an integer")
    u_conflict = int(u_conflict)
    if (n_conflict == 0 and u_conflict != 0) or not 0 <= u_conflict <= n_conflict:
        raise ValueError("u_conflict must be zero iff n_conflict is zero, otherwise in [1,n_conflict]")
    if n_conflict > 0 and u_conflict == 0:
        raise ValueError("positive n_conflict requires positive u_conflict")
    if condition not in {"concentrated", "diverse", "structured_holdout"}:
        raise ValueError("condition must be concentrated, diverse, or structured_holdout")
    n_eval = n if n_eval is None else _validate_n(n_eval, name="n_eval")
    train_types = _validate_h4_types(structured_train_types, "structured_train_types")
    test_types = _validate_h4_types(structured_test_types, "structured_test_types")
    overlap = set(train_types) & set(test_types)
    if overlap:
        raise ValueError(
            "structured_train_types and structured_test_types must be disjoint; "
            f"overlap={sorted(overlap)}"
        )

    train = _base_batch(
        n,
        realized_q,
        k,
        seed,
        max_k=max_k,
        state_dim=0,
        split="train",
        id_offset=0,
    )
    train_conflicts = np.flatnonzero(train.latents["is_conflict"]).astype(np.int64)
    train_sources, assignments = _h4_prototype_plan(
        train, train_conflicts, u_conflict, seed
    )
    # IDs derive from condition-independent source rows. A prototype shared by two
    # U settings therefore has identical state/context features in both arms.
    train_prototype_ids = n + np.asarray(train.sample_id, dtype=np.int64)[train_sources]
    train = _attach_h4_context(
        train,
        condition=condition,
        conflict_positions=train_conflicts,
        prototype_assignments=assignments,
        prototype_source_positions=train_sources,
        prototype_ids=train_prototype_ids,
        state_dim=state_dim,
        context_seed=seed,
        split="train",
        structured_types=train_types if condition == "structured_holdout" else None,
    )

    # This namespace does not depend on U or condition, so every arm receives the
    # same unseen evaluation rows while remaining disjoint from all train IDs.
    eval_offset = 2 * n + 1
    evaluation = _base_batch(
        n_eval,
        0.0,
        k,
        seed + 1,
        max_k=max_k,
        state_dim=0,
        split="conflict_eval",
        id_offset=eval_offset,
    )
    eval_positions = np.arange(n_eval, dtype=np.int64)
    # Evaluation contexts are all unseen and therefore maximally diverse.
    eval_sources, eval_assignments = _h4_prototype_plan(
        evaluation, eval_positions, n_eval, seed + 1
    )
    eval_prototype_ids = eval_offset + n_eval + eval_sources
    evaluation = _attach_h4_context(
        evaluation,
        condition=condition,
        conflict_positions=eval_positions,
        prototype_assignments=eval_assignments,
        prototype_source_positions=eval_sources,
        prototype_ids=eval_prototype_ids,
        state_dim=state_dim,
        context_seed=seed,
        split="conflict_eval",
        structured_types=test_types if condition == "structured_holdout" else None,
    )
    if set(np.asarray(train.state_id).tolist()) & set(np.asarray(evaluation.state_id).tolist()):
        raise RuntimeError("internal error: H4 train/evaluation state IDs overlap")
    return DatasetBundle(train, evaluation)


def _independent_balanced_columns(
    n: int, width: int, rng: np.random.Generator
) -> SignArray:
    if width == 0:
        return np.empty((n, 0), dtype=np.int8)
    columns = []
    for _ in range(width):
        column = np.tile(np.asarray([-1, 1], dtype=np.int8), n // 2)
        rng.shuffle(column)
        columns.append(column)
    return np.column_stack(columns).astype(np.int8, copy=False)


def _binary_entropy(probability: float) -> float:
    if probability in (0.0, 1.0):
        return 0.0
    return -probability * math.log2(probability) - (1.0 - probability) * math.log2(
        1.0 - probability
    )


def _probability_for_entropy(entropy: float) -> float:
    if not 0.0 <= entropy <= 1.0:
        raise ValueError("nuisance_entropy must lie in [0, 1] bits per nuisance channel")
    if entropy == 0.0:
        return 0.0
    if entropy == 1.0:
        return 0.5
    low, high = 0.0, 0.5
    for _ in range(64):
        midpoint = (low + high) / 2.0
        if _binary_entropy(midpoint) < entropy:
            low = midpoint
        else:
            high = midpoint
    return (low + high) / 2.0


def make_h5_dataset(
    n: int,
    q: float,
    k: int,
    seed: int = 0,
    *,
    context_bits: int = 1,
    nuisance_bits: int = 0,
    nuisance_entropy: float = 1.0,
    max_k: int | None = None,
    state_dim: int = 0,
) -> SemanticBatch:
    """Create H5 observations with reward-relevant context and nuisance bits.

    Context bits are visible state descriptors but do not alter the intended goal.
    Nuisance bits are independent demonstration details.  Their zero-based values
    are exposed through ``nuisance_targets`` for optional auxiliary imitation
    heads.  ``nuisance_entropy`` controls entropy per nuisance bit.
    """

    n = _validate_n(n)
    for value, name in ((context_bits, "context_bits"), (nuisance_bits, "nuisance_bits")):
        if isinstance(value, (bool, np.bool_)) or int(value) != value or int(value) < 0:
            raise ValueError(f"{name} must be a non-negative integer")
    context_bits, nuisance_bits = int(context_bits), int(nuisance_bits)
    base = make_standard_dataset(
        n, q, k, seed, max_k=max_k, state_dim=state_dim, split="h5_train"
    )
    rng = np.random.default_rng(_stable_seed(seed, "h5-auxiliary"))
    contexts = _independent_balanced_columns(n, context_bits, rng)
    probability = _probability_for_entropy(float(nuisance_entropy))
    nuisance = np.empty((n, nuisance_bits), dtype=np.int8)
    for column in range(nuisance_bits):
        positive_count = int(round(n * probability))
        values = np.full(n, -1, dtype=np.int8)
        chosen = rng.permutation(n)[:positive_count]
        values[chosen] = 1
        nuisance[:, column] = values

    channels = dict(base.channels)
    channels.update({f"C_{i + 1}": contexts[:, i] for i in range(context_bits)})
    channels.update({f"N_{i + 1}": nuisance[:, i] for i in range(nuisance_bits)})
    realized_entropies = tuple(
        _binary_entropy(float(np.mean(nuisance[:, i] > 0))) for i in range(nuisance_bits)
    )
    metadata = dict(base.metadata)
    metadata.update(
        {
            "dataset": "h5",
            "context_bits": context_bits,
            "nuisance_bits": nuisance_bits,
            "requested_nuisance_entropy": float(nuisance_entropy),
            "realized_nuisance_entropy": realized_entropies,
            "total_realized_nuisance_entropy": float(sum(realized_entropies)),
        }
    )
    nuisance_targets = (nuisance > 0).astype(np.int64)
    return base.with_updates(
        channels=channels,
        nuisance_targets=nuisance_targets if nuisance_bits else None,
        metadata=metadata,
    )


def _factorial_h9_rows(n: int, seed: int) -> tuple[SignArray, NDArray[np.int8], SignArray]:
    """Balanced target, environment, and unused-proxy signs with factorial coverage."""

    patterns = np.asarray(
        [(y, environment, other) for y in (-1, 1) for environment in (0, 1) for other in (-1, 1)],
        dtype=np.int8,
    )
    blocks, remainder = divmod(n, len(patterns))
    rows = np.tile(patterns, (blocks, 1))
    if remainder:
        # Add matched -Y/+Y pairs.  Exact independence is arithmetically possible
        # only at full factorial block sizes, but target balance remains exact.
        half = remainder // 2
        rng = np.random.default_rng(_stable_seed(seed, "h9-remainder"))
        negative = patterns[patterns[:, 0] == -1][rng.permutation(4)[:half]]
        positive = patterns[patterns[:, 0] == 1][rng.permutation(4)[:half]]
        rows = np.vstack((rows, negative, positive))
    rng = np.random.default_rng(seed)
    rng.shuffle(rows)
    return (
        rows[:, 0].astype(np.int8),
        rows[:, 1].astype(np.int8),
        rows[:, 2].astype(np.int8),
    )


def make_h9_dataset(
    n: int,
    seed: int = 0,
    *,
    condition: H9Condition = "normal",
    proxy0_degree: int = 1,
    proxy1_degree: int = 1,
    max_k: int = 0,
    state_dim: int = 0,
) -> SemanticBatch:
    """Create the H9 context-selector task with target ``P_E``.

    ``E`` is the latent reward context and ``C`` is the observed context.  The
    two proxy goals vary independently in the generating factorial design.  For
    each row, the proxy selected by ``E`` is set to the balanced target and the
    other remains independent. A degree-one proxy is exposed as ``P0``/``P1``;
    higher-degree proxies are exposed only through Rademacher code components
    ``P0_1,...``/``P1_1,...`` whose product is the corresponding latent proxy.
    OOD conditions alter only observed context; target and latent proxy goals
    remain unchanged.
    """

    n = _validate_n(n)
    aliases = {"context_removed": "removed", "context_randomized": "randomized", "mismatch": "mismatched"}
    condition = aliases.get(str(condition), str(condition))  # type: ignore[assignment]
    if condition not in {"normal", "removed", "randomized", "mismatched"}:
        raise ValueError("condition must be normal, removed, randomized, or mismatched")
    if max_k < 0:
        raise ValueError("max_k must be non-negative")
    degrees = []
    for value, name in (
        (proxy0_degree, "proxy0_degree"),
        (proxy1_degree, "proxy1_degree"),
    ):
        if isinstance(value, (bool, np.bool_)) or int(value) != value or int(value) < 1:
            raise ValueError(f"{name} must be a positive integer")
        degrees.append(int(value))
    proxy0_degree, proxy1_degree = degrees

    target, environment, other_proxy = _factorial_h9_rows(n, seed)
    p0 = np.where(environment == 0, target, other_proxy).astype(np.int8)
    p1 = np.where(environment == 1, target, other_proxy).astype(np.int8)
    p_e = np.where(environment == 0, p0, p1).astype(np.int8)
    if not np.array_equal(p_e, target):  # pragma: no cover
        raise RuntimeError("internal error constructing P_E target")

    if condition == "normal":
        observed_context = environment.copy()
        present = np.ones(n, dtype=np.int8)
    elif condition == "removed":
        observed_context = np.zeros(n, dtype=np.int8)
        present = np.zeros(n, dtype=np.int8)
    elif condition == "mismatched":
        observed_context = (1 - environment).astype(np.int8)
        present = np.ones(n, dtype=np.int8)
    else:
        observed_context = _balanced_binary(n, np.random.default_rng(_stable_seed(seed, "h9-C")))
        present = np.ones(n, dtype=np.int8)

    ids = np.arange(n, dtype=np.int64)
    state = _state_features(ids, state_dim, _stable_seed(seed, "h9-state"))
    channels: dict[str, NDArray[Any]] = {}
    for name, values, degree in (
        ("P0", p0, proxy0_degree),
        ("P1", p1, proxy1_degree),
    ):
        code = degree_k_code(
            values,
            degree,
            seed=_stable_seed(seed, "h9-proxy-code", name, degree),
        )
        if degree == 1:
            channels[name] = code[:, 0]
        else:
            channels.update(
                {f"{name}_{index + 1}": code[:, index] for index in range(degree)}
            )
    channels.update({"C": observed_context, "C_present": present})
    return SemanticBatch(
        y=target,
        target=p_e,
        reward=p_e.astype(np.float32),
        channels=channels,
        latents={"E": environment, "P_E": p_e, "P0": p0, "P1": p1},
        state=state,
        sample_id=ids,
        state_id=ids,
        episode_id=ids,
        step_id=np.zeros(n, dtype=np.int64),
        metadata={
            "dataset": "h9",
            "condition": condition,
            "target_rule": "P_E",
            "proxy0_degree": proxy0_degree,
            "proxy1_degree": proxy1_degree,
            "active_k": 0,
            "max_k": int(max_k),
            "r_active_mask": tuple(False for _ in range(max_k)),
            "padding_seed": _stable_seed(seed, "h9-padding"),
            "factorial_complete": n % 8 == 0,
            "p0_p1_correlation": float(np.corrcoef(p0, p1)[0, 1]) if n > 2 else float("nan"),
        },
    )


# Descriptive aliases used by experiment runners.
make_h1_dataset = make_standard_dataset
make_h2_dataset = make_standard_dataset
make_h3_dataset = make_standard_dataset
make_nuisance_dataset = make_h5_dataset
make_context_dataset = make_h9_dataset


__all__ = [
    "H4_FAILURE_MECHANISMS",
    "TARGET_RULES",
    "DatasetBundle",
    "FeatureSpec",
    "H4Condition",
    "H9Condition",
    "SemanticBatch",
    "TargetRule",
    "balanced_signs",
    "decode_target_rule",
    "degree_k_code",
    "exact_rule_code",
    "flip_channels",
    "flip_exact_rule_output",
    "intervene",
    "make_conflict_dataset",
    "make_context_dataset",
    "make_degree_k_code",
    "make_exact_rule_code",
    "make_h1_dataset",
    "make_h2_dataset",
    "make_h3_dataset",
    "make_h4_dataset",
    "make_h5_dataset",
    "make_h9_dataset",
    "make_nuisance_dataset",
    "make_standard_dataset",
    "make_standard_train_eval",
    "mask_channels",
    "remove_proxy",
    "validate_proxy_confounds",
]
