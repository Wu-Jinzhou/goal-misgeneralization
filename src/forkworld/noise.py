"""Stateless temporal noise processes for ForkWorld Hypothesis 6.

Noise is derived from semantic IDs with a stable integer hash.  Consequently it
does not depend on global RNG state, data-loader ordering, batch boundaries, or
the number of earlier calls.  A caller that wants a fresh step-resampled draw
passes a new explicit ``draw`` index; rerunning the same indexed experiment is
bit-for-bit reproducible.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Literal, Mapping, Sequence

import numpy as np
from numpy.typing import NDArray

from .data import SemanticBatch


NoiseRegime = Literal["step_resampled", "episode_static", "state_static", "biased"]
NoiseLocation = Literal["observation", "label", "reward"]

_REGIME_ALIASES = {
    "step": "step_resampled",
    "step-resampled": "step_resampled",
    "resampled": "step_resampled",
    "episode": "episode_static",
    "episode-static": "episode_static",
    "state": "state_static",
    "state-static": "state_static",
    "bias": "biased",
}
_LOCATION_ALIASES = {"obs": "observation", "labels": "label", "rewards": "reward"}


def _normalise_regime(value: str) -> NoiseRegime:
    normalised = _REGIME_ALIASES.get(value.lower(), value.lower().replace("-", "_"))
    if normalised not in {"step_resampled", "episode_static", "state_static", "biased"}:
        raise ValueError(
            "regime must be step_resampled, episode_static, state_static, or biased"
        )
    return normalised  # type: ignore[return-value]


def _normalise_location(value: str) -> NoiseLocation:
    normalised = _LOCATION_ALIASES.get(value.lower(), value.lower())
    if normalised not in {"observation", "label", "reward"}:
        raise ValueError("location must be observation, label, or reward")
    return normalised  # type: ignore[return-value]


@dataclass(frozen=True)
class NoiseConfig:
    """Definition of one matched-scale perturbation.

    ``scale`` is the requested marginal standard deviation.  In the biased
    regime, ``bias`` is the conditional mean magnitude.  It defaults to
    ``scale``, producing the exact proxy ``epsilon = scale * Y``.  If a smaller
    bias is selected, independent Gaussian residual noise is added with standard
    deviation ``sqrt(scale**2 - bias**2)`` so the population marginal variance
    remains matched for balanced Y.
    """

    regime: NoiseRegime | str
    location: NoiseLocation | str = "observation"
    scale: float = 1.0
    seed: int = 0
    channels: tuple[str, ...] | Sequence[str] | str | None = None
    bias: float | None = None

    def __post_init__(self) -> None:
        regime = _normalise_regime(str(self.regime))
        location = _normalise_location(str(self.location))
        scale = float(self.scale)
        if not math.isfinite(scale) or scale < 0.0:
            raise ValueError("noise scale must be finite and non-negative")
        if isinstance(self.seed, (bool, np.bool_)) or int(self.seed) != self.seed:
            raise ValueError("noise seed must be an integer")
        if isinstance(self.channels, str):
            channels = (self.channels,)
        elif self.channels is None:
            channels = ("P",) if location == "observation" else ()
        else:
            channels = tuple(self.channels)
        if any(not isinstance(channel, str) or not channel for channel in channels):
            raise ValueError("noise channels must be non-empty strings")
        if len(set(channels)) != len(channels):
            raise ValueError("noise channels must not contain duplicates")
        if location != "observation" and channels:
            raise ValueError("channels apply only to observation noise")

        bias = self.bias
        if regime == "biased":
            bias = scale if bias is None else float(bias)
            if not math.isfinite(bias) or abs(bias) > scale + 1e-15:
                raise ValueError("biased-noise magnitude must be finite and no larger than scale")
        elif bias is not None:
            raise ValueError("bias may be specified only for the biased regime")
        object.__setattr__(self, "regime", regime)
        object.__setattr__(self, "location", location)
        object.__setattr__(self, "scale", scale)
        object.__setattr__(self, "seed", int(self.seed))
        object.__setattr__(self, "channels", channels)
        object.__setattr__(self, "bias", bias)


def _salt(seed: int, *parts: object) -> np.uint64:
    digest = hashlib.blake2b(digest_size=8, person=b"forknoise")
    for part in (seed, *parts):
        payload = str(part).encode("utf-8")
        digest.update(len(payload).to_bytes(4, "little"))
        digest.update(payload)
    return np.uint64(int.from_bytes(digest.digest(), "little"))


def _mix(values: NDArray[np.uint64]) -> NDArray[np.uint64]:
    values = np.asarray(values, dtype=np.uint64)
    with np.errstate(over="ignore"):
        values = values + np.uint64(0x9E3779B97F4A7C15)
        values = (values ^ (values >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
        values = (values ^ (values >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
        values = values ^ (values >> np.uint64(31))
    return values


def _combine(left: NDArray[np.uint64], right: NDArray[np.uint64] | np.uint64) -> NDArray[np.uint64]:
    with np.errstate(over="ignore"):
        return _mix(left ^ (_mix(np.asarray(right, dtype=np.uint64)) + np.uint64(0x517CC1B727220A95)))


def noise_group_keys(
    batch: SemanticBatch, regime: NoiseRegime | str, *, draw: int = 0
) -> NDArray[np.uint64]:
    """Return the semantic grouping key used by a temporal noise regime."""

    regime = _normalise_regime(str(regime))
    if isinstance(draw, (bool, np.bool_)) or int(draw) != draw or int(draw) < 0:
        raise ValueError("draw must be a non-negative integer")
    draw = int(draw)
    if regime == "state_static":
        return np.asarray(batch.state_id, dtype=np.int64).astype(np.uint64, copy=False)
    if regime == "episode_static":
        keys = np.asarray(batch.episode_id, dtype=np.int64).astype(np.uint64, copy=False)
        # ``draw`` denotes a newly sampled collection of episodes. Noise is
        # constant within each episode but resampled when those semantic states
        # are encountered in later episodes/collections.
        return _combine(keys, np.uint64(draw))

    # The explicit draw index gives callers controlled resampling across epochs;
    # episode/step/sample IDs distinguish events within one draw.
    keys = np.asarray(batch.sample_id, dtype=np.int64).astype(np.uint64, copy=False)
    keys = _combine(keys, np.asarray(batch.episode_id, dtype=np.int64).astype(np.uint64, copy=False))
    keys = _combine(keys, np.asarray(batch.step_id, dtype=np.int64).astype(np.uint64, copy=False))
    keys = _combine(keys, np.uint64(draw))
    return keys


def _standard_normals(keys: NDArray[np.uint64], seed: int, stream: str) -> NDArray[np.float64]:
    first = _mix(keys ^ _salt(seed, stream, 0))
    second = _mix(keys ^ _salt(seed, stream, 1))
    # Taking the high 53 bits produces open-interval doubles suitable for
    # Box-Muller while remaining deterministic across NumPy RNG versions.
    denominator = float(1 << 53)
    u1 = ((first >> np.uint64(11)).astype(np.float64) + 0.5) / denominator
    u2 = ((second >> np.uint64(11)).astype(np.float64) + 0.5) / denominator
    return np.sqrt(-2.0 * np.log(u1)) * np.cos(2.0 * np.pi * u2)


def noise_array(
    batch: SemanticBatch,
    config: NoiseConfig,
    *,
    stream: str = "default",
    draw: int = 0,
) -> NDArray[np.float64]:
    """Generate one scalar noise stream aligned with ``batch`` rows."""

    if not isinstance(config, NoiseConfig):
        raise TypeError("config must be a NoiseConfig")
    keys = noise_group_keys(batch, config.regime, draw=draw)
    if config.regime == "biased":
        assert config.bias is not None
        residual_scale = math.sqrt(max(0.0, config.scale**2 - config.bias**2))
        result = config.bias * np.asarray(batch.y, dtype=np.float64)
        if residual_scale:
            result = result + residual_scale * _standard_normals(keys, config.seed, stream)
        return result.astype(np.float64, copy=False)
    return config.scale * _standard_normals(keys, config.seed, stream)


# Name used by some scripts to emphasize that no mutable RNG is involved.
deterministic_noise = noise_array
generate_noise = noise_array


def _correlation(left: NDArray[np.float64], right: NDArray[np.float64]) -> float:
    if len(left) < 2 or float(np.std(left)) == 0.0 or float(np.std(right)) == 0.0:
        return float("nan")
    return float(np.corrcoef(left, right)[0, 1])


@dataclass(frozen=True)
class NoiseDiagnostics:
    """Finite-sample checks saved alongside every H6 perturbation."""

    regime: str
    location: str
    stream: str
    n: int
    requested_scale: float
    requested_bias: float | None
    mean: float
    variance: float
    standard_deviation: float
    rms: float
    correlation_with_y: float
    mean_given_y_negative: float
    mean_given_y_positive: float
    unique_groups: int
    maximum_within_group_range: float

    @property
    def conditional_mean_gap(self) -> float:
        return self.mean_given_y_positive - self.mean_given_y_negative

    def as_dict(self) -> dict[str, str | int | float | None]:
        return {
            "regime": self.regime,
            "location": self.location,
            "stream": self.stream,
            "n": self.n,
            "requested_scale": self.requested_scale,
            "requested_bias": self.requested_bias,
            "mean": self.mean,
            "variance": self.variance,
            "standard_deviation": self.standard_deviation,
            "rms": self.rms,
            "correlation_with_y": self.correlation_with_y,
            "mean_given_y_negative": self.mean_given_y_negative,
            "mean_given_y_positive": self.mean_given_y_positive,
            "conditional_mean_gap": self.conditional_mean_gap,
            "unique_groups": self.unique_groups,
            "maximum_within_group_range": self.maximum_within_group_range,
        }


def noise_diagnostics(
    values: NDArray[Any],
    batch: SemanticBatch,
    config: NoiseConfig,
    *,
    stream: str = "default",
    draw: int = 0,
) -> NoiseDiagnostics:
    """Summarize marginal, conditional, and persistence properties of noise."""

    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or len(array) != len(batch):
        raise ValueError("diagnostic noise values must have shape [len(batch)]")
    keys = noise_group_keys(batch, config.regime, draw=draw)
    maximum_range = 0.0
    if len(array):
        order = np.argsort(keys, kind="stable")
        ordered_keys = keys[order]
        ordered_values = array[order]
        boundaries = np.flatnonzero(np.r_[True, ordered_keys[1:] != ordered_keys[:-1], True])
        for start, stop in zip(boundaries[:-1], boundaries[1:], strict=True):
            group = ordered_values[start:stop]
            maximum_range = max(maximum_range, float(np.max(group) - np.min(group)))
    negative = array[np.asarray(batch.y) == -1]
    positive = array[np.asarray(batch.y) == 1]
    return NoiseDiagnostics(
        regime=str(config.regime),
        location=str(config.location),
        stream=stream,
        n=len(array),
        requested_scale=config.scale,
        requested_bias=config.bias,
        mean=float(np.mean(array)),
        variance=float(np.var(array)),
        standard_deviation=float(np.std(array)),
        rms=float(np.sqrt(np.mean(np.square(array)))),
        correlation_with_y=_correlation(array, np.asarray(batch.y, dtype=np.float64)),
        # A shuffled optimizer minibatch need not contain both target signs.
        # Preserve the missing stratum explicitly without emitting NumPy's
        # empty-slice warnings during every H6 training step.
        mean_given_y_negative=float(np.mean(negative)) if len(negative) else float("nan"),
        mean_given_y_positive=float(np.mean(positive)) if len(positive) else float("nan"),
        unique_groups=int(len(np.unique(keys))),
        maximum_within_group_range=maximum_range,
    )


# Short alias suitable for notebooks.
diagnose_noise = noise_diagnostics


@dataclass(frozen=True)
class NoiseApplication:
    """A perturbed batch plus exact noise arrays and per-stream diagnostics."""

    batch: SemanticBatch
    noise: Mapping[str, NDArray[np.float64]]
    diagnostics: Mapping[str, NoiseDiagnostics]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "noise",
            MappingProxyType({key: np.array(value, copy=True) for key, value in self.noise.items()}),
        )
        object.__setattr__(self, "diagnostics", MappingProxyType(dict(self.diagnostics)))


def apply_noise_with_diagnostics(
    batch: SemanticBatch,
    config: NoiseConfig,
    *,
    draw: int = 0,
) -> NoiseApplication:
    """Apply noise only at the configured training-objective location."""

    if not isinstance(config, NoiseConfig):
        raise TypeError("config must be a NoiseConfig")
    streams: dict[str, NDArray[np.float64]] = {}
    diagnostics: dict[str, NoiseDiagnostics] = {}
    updates: dict[str, Any] = {}
    metadata = dict(batch.metadata)

    if config.location == "observation":
        assert isinstance(config.channels, tuple)
        unknown = set(config.channels) - set(batch.channels)
        if unknown:
            raise KeyError(f"unknown observation-noise channels: {sorted(unknown)}")
        channels = {name: np.array(value, copy=True) for name, value in batch.channels.items()}
        for channel in config.channels:
            original = np.asarray(channels[channel])
            if original.ndim == 1:
                stream_names = ((channel, None),)
            else:
                stream_names = tuple((f"{channel}_{column}", column) for column in range(original.shape[1]))
            noisy = original.astype(np.float64, copy=True)
            for stream, column in stream_names:
                values = noise_array(batch, config, stream=stream, draw=draw)
                streams[stream] = values
                diagnostics[stream] = noise_diagnostics(
                    values, batch, config, stream=stream, draw=draw
                )
                if column is None:
                    noisy += values
                else:
                    noisy[:, column] += values
            channels[channel] = noisy.astype(np.float32)
        updates["channels"] = channels
        if any(channel.startswith("R_") for channel in config.channels):
            metadata["enforce_exact_code"] = False
    elif config.location == "label":
        values = noise_array(batch, config, stream="label", draw=draw)
        streams["label"] = values
        diagnostics["label"] = noise_diagnostics(
            values, batch, config, stream="label", draw=draw
        )
        updates["target"] = (np.asarray(batch.target, dtype=np.float64) + values).astype(np.float32)
    else:
        values = noise_array(batch, config, stream="reward", draw=draw)
        streams["reward"] = values
        diagnostics["reward"] = noise_diagnostics(
            values, batch, config, stream="reward", draw=draw
        )
        updates["reward"] = (np.asarray(batch.reward, dtype=np.float64) + values).astype(np.float32)

    history = list(metadata.get("noise_applications", ()))
    history.append(
        {
            "regime": config.regime,
            "location": config.location,
            "scale": config.scale,
            "bias": config.bias,
            "seed": config.seed,
            "channels": config.channels,
            "draw": int(draw),
            "diagnostics": {name: item.as_dict() for name, item in diagnostics.items()},
        }
    )
    metadata["noise_applications"] = tuple(history)
    updates["metadata"] = metadata
    return NoiseApplication(batch.with_updates(**updates), streams, diagnostics)


def apply_noise(
    batch: SemanticBatch,
    config: NoiseConfig | None = None,
    *,
    regime: NoiseRegime | str | None = None,
    location: NoiseLocation | str = "observation",
    scale: float = 1.0,
    sigma: float | None = None,
    seed: int = 0,
    channels: Sequence[str] | str | None = None,
    bias: float | None = None,
    draw: int = 0,
) -> SemanticBatch:
    """Return a perturbed batch.

    A prebuilt :class:`NoiseConfig` or explicit keyword arguments may be used.
    ``sigma`` is accepted as an alias for ``scale``.  Use
    :func:`apply_noise_with_diagnostics` when the raw perturbations are needed.
    """

    if config is not None:
        if regime is not None or sigma is not None or channels is not None or bias is not None:
            raise ValueError("do not mix a NoiseConfig with explicit noise overrides")
    else:
        if regime is None:
            raise ValueError("regime is required when config is omitted")
        if sigma is not None:
            if scale != 1.0 and not math.isclose(scale, sigma):
                raise ValueError("scale and sigma disagree")
            scale = sigma
        config = NoiseConfig(
            regime=regime,
            location=location,
            scale=scale,
            seed=seed,
            channels=channels,
            bias=bias,
        )
    return apply_noise_with_diagnostics(batch, config, draw=draw).batch


@dataclass(frozen=True)
class NoiseProcess:
    """Small functional wrapper convenient for experiment-runner configuration."""

    config: NoiseConfig

    def sample(
        self, batch: SemanticBatch, *, stream: str = "default", draw: int = 0
    ) -> NDArray[np.float64]:
        return noise_array(batch, self.config, stream=stream, draw=draw)

    def apply(self, batch: SemanticBatch, *, draw: int = 0) -> NoiseApplication:
        return apply_noise_with_diagnostics(batch, self.config, draw=draw)


__all__ = [
    "NoiseApplication",
    "NoiseConfig",
    "NoiseDiagnostics",
    "NoiseLocation",
    "NoiseProcess",
    "NoiseRegime",
    "apply_noise",
    "apply_noise_with_diagnostics",
    "deterministic_noise",
    "diagnose_noise",
    "generate_noise",
    "noise_array",
    "noise_diagnostics",
    "noise_group_keys",
]
