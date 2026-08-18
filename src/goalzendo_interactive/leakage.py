"""Deterministic, fail-closed surface-leakage audits for G03 episode banks.

The EpisodeBank adapter deliberately constructs features from a small allowlist
instead of redacting a serialized episode.  Consequently scene indices and
attributes, Official-Law labels and identities, placards, shadow identities and
truth values, request/episode identifiers, and generation provenance never enter
the classifier input.
"""

from __future__ import annotations

import hashlib
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from statistics import NormalDist
from typing import Any, Literal

import numpy as np

from .generation import EpisodeBank
from .schema import NONEMPTY_ARRANGEMENT_COUNT

SURFACE_LEAKAGE_SCHEMA_VERSION = 1
SURFACE_LEAKAGE_CONFIDENCE_LEVEL = 0.95
SURFACE_LEAKAGE_CHANCE_MARGIN = 0.05
SURFACE_LEAKAGE_BOOTSTRAP_REPLICATES = 10_000
SURFACE_LEAKAGE_FOLDS = 5
SURFACE_LEAKAGE_LAPLACE_ALPHA = 1.0

EPISODE_BANK_SURFACE_TARGETS = (
    "target_label",
    "target_formula_stratum",
    "placard_error_target",
    "placard_error_status",
    "shadow_error_status",
)

EPISODE_SURFACE_FEATURE_KEYS = frozenset(
    {
        "renderer",
        "opening_count",
        "terminal_count",
        "phase",
        "phase_count",
        "position",
    }
)

MASK_POLICY = (
    "allowlist-only: renderer and structural count/phase/position tokens; "
    "scene, target-label, Official-Law, placard, shadow, identifier, digest, "
    "request-stratum, and generation-provenance content are omitted"
)

STATISTICAL_METHOD = (
    "deterministic stratified grouped cross-validation; uniform-prior Laplace "
    "multinomial naive Bayes; out-of-fold macro balanced accuracy; deterministic "
    "episode-cluster percentile bootstrap (95%); pass iff chance is inside the "
    "interval and the upper endpoint is strictly below chance+0.05"
)

LeakageDecision = Literal["pass", "leakage", "insufficient_data"]


def _require_nonempty_ascii(value: object, *, name: str) -> str:
    if type(value) is not str or not value or not value.isascii():
        raise ValueError(f"{name} must be a nonempty ASCII string")
    return value


def _bool_label(value: bool) -> str:
    return "true" if value else "false"


def _digest_fields(domain: bytes, fields: tuple[str, ...]) -> str:
    digest = hashlib.sha256(domain)
    for field in fields:
        digest.update(field.encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class SurfaceLeakageSample:
    """One labeled surface sample; ``group_id`` is the CV/bootstrap cluster."""

    sample_id: str
    group_id: str
    features: tuple[str, ...]
    label: str

    def __post_init__(self) -> None:
        _require_nonempty_ascii(self.sample_id, name="sample_id")
        _require_nonempty_ascii(self.group_id, name="group_id")
        _require_nonempty_ascii(self.label, name="label")
        features = tuple(self.features)
        object.__setattr__(self, "features", features)
        if any(type(feature) is not str or not feature or not feature.isascii() for feature in features):
            raise ValueError("surface features must be nonempty ASCII strings")


@dataclass(frozen=True, slots=True)
class SurfaceLeakageTarget:
    """A registered classification target and its masked surface samples."""

    name: str
    classes: tuple[str, ...]
    samples: tuple[SurfaceLeakageSample, ...]

    def __post_init__(self) -> None:
        _require_nonempty_ascii(self.name, name="target name")
        classes = tuple(self.classes)
        samples = tuple(self.samples)
        object.__setattr__(self, "classes", classes)
        object.__setattr__(self, "samples", samples)
        if len(classes) < 2 or len(set(classes)) != len(classes):
            raise ValueError("surface target classes must contain at least two unique values")
        if any(type(item) is not str or not item or not item.isascii() for item in classes):
            raise ValueError("surface target classes must be nonempty ASCII strings")
        if any(type(sample) is not SurfaceLeakageSample for sample in samples):
            raise ValueError("surface target contains a non-SurfaceLeakageSample")
        sample_ids = [sample.sample_id for sample in samples]
        if len(sample_ids) != len(set(sample_ids)):
            raise ValueError("surface target sample ids must be unique")
        unknown = sorted({sample.label for sample in samples}.difference(classes))
        if unknown:
            raise ValueError(f"surface target contains unregistered classes: {unknown}")

    @property
    def digest(self) -> str:
        fields: list[str] = [self.name, *self.classes]
        for sample in self.samples:
            fields.extend(
                (
                    sample.sample_id,
                    sample.group_id,
                    sample.label,
                    str(len(sample.features)),
                    *sample.features,
                )
            )
        return _digest_fields(b"goalzendo-surface-leakage-target-v1\0", tuple(fields))


@dataclass(frozen=True, slots=True)
class SurfaceLeakageConfig:
    """Frozen numerical choices for the deterministic audit."""

    fold_count: int = SURFACE_LEAKAGE_FOLDS
    bootstrap_replicates: int = SURFACE_LEAKAGE_BOOTSTRAP_REPLICATES
    laplace_alpha: float = SURFACE_LEAKAGE_LAPLACE_ALPHA
    additional_minimum_groups: int = 0

    def __post_init__(self) -> None:
        if isinstance(self.fold_count, bool) or not isinstance(self.fold_count, int):
            raise ValueError("fold_count must be an integer")
        if not 2 <= self.fold_count <= 20:
            raise ValueError("fold_count must lie in [2, 20]")
        if isinstance(self.bootstrap_replicates, bool) or not isinstance(
            self.bootstrap_replicates, int
        ):
            raise ValueError("bootstrap_replicates must be an integer")
        if self.bootstrap_replicates < 1_000:
            raise ValueError("bootstrap_replicates must be at least 1000")
        if isinstance(self.laplace_alpha, bool) or not isinstance(
            self.laplace_alpha, (int, float)
        ):
            raise ValueError("laplace_alpha must be numeric")
        if not math.isfinite(float(self.laplace_alpha)) or self.laplace_alpha <= 0:
            raise ValueError("laplace_alpha must be finite and positive")
        if isinstance(self.additional_minimum_groups, bool) or not isinstance(
            self.additional_minimum_groups, int
        ):
            raise ValueError("additional_minimum_groups must be an integer")
        if self.additional_minimum_groups < 0:
            raise ValueError("additional_minimum_groups cannot be negative")

    def as_obj(self) -> dict[str, int | float]:
        return {
            "fold_count": self.fold_count,
            "bootstrap_replicates": self.bootstrap_replicates,
            "laplace_alpha": float(self.laplace_alpha),
            "additional_minimum_groups": self.additional_minimum_groups,
        }


@dataclass(frozen=True, slots=True)
class SurfaceLeakageTargetResult:
    target_name: str
    decision: LeakageDecision
    passed: bool
    sample_count: int
    group_count: int
    class_counts: tuple[tuple[str, int], ...]
    class_group_counts: tuple[tuple[str, int], ...]
    fold_count: int
    minimum_groups_required: int
    chance: float
    chance_ceiling: float
    balanced_accuracy: float | None
    interval_lower: float | None
    interval_upper: float | None
    chance_in_interval: bool | None
    upper_below_ceiling: bool | None
    fold_digest: str | None
    prediction_digest: str | None
    reasons: tuple[str, ...]

    def as_obj(self) -> dict[str, Any]:
        return {
            "target_name": self.target_name,
            "decision": self.decision,
            "passed": self.passed,
            "sample_count": self.sample_count,
            "group_count": self.group_count,
            "class_counts": dict(self.class_counts),
            "class_group_counts": dict(self.class_group_counts),
            "fold_count": self.fold_count,
            "minimum_groups_required": self.minimum_groups_required,
            "chance": self.chance,
            "chance_ceiling": self.chance_ceiling,
            "balanced_accuracy": self.balanced_accuracy,
            "interval_lower": self.interval_lower,
            "interval_upper": self.interval_upper,
            "chance_in_interval": self.chance_in_interval,
            "upper_below_ceiling": self.upper_below_ceiling,
            "fold_digest": self.fold_digest,
            "prediction_digest": self.prediction_digest,
            "reasons": list(self.reasons),
        }


@dataclass(frozen=True, slots=True)
class SurfaceLeakageAuditReport:
    dataset_id: str
    dataset_digest: str
    mask_policy: str
    statistical_method: str
    confidence_level: float
    chance_margin: float
    config: SurfaceLeakageConfig
    results: tuple[SurfaceLeakageTargetResult, ...]

    @property
    def passed(self) -> bool:
        return bool(self.results) and all(result.passed for result in self.results)

    def as_obj(self) -> dict[str, Any]:
        return {
            "schema_version": SURFACE_LEAKAGE_SCHEMA_VERSION,
            "dataset_id": self.dataset_id,
            "dataset_digest": self.dataset_digest,
            "mask_policy": self.mask_policy,
            "statistical_method": self.statistical_method,
            "confidence_level": self.confidence_level,
            "chance_margin": self.chance_margin,
            "config": self.config.as_obj(),
            "passed": self.passed,
            "results": [result.as_obj() for result in self.results],
        }


def _episode_surface_features(*, renderer: str, opening_count: int, terminal_count: int) -> tuple[str, ...]:
    return (
        f"renderer={renderer}",
        f"opening_count={opening_count}",
        f"terminal_count={terminal_count}",
    )


def _observation_surface_features(
    episode_features: tuple[str, ...],
    *,
    phase: str,
    phase_count: int,
    position: int,
) -> tuple[str, ...]:
    return (
        *episode_features,
        f"phase={phase}",
        f"phase_count={phase_count}",
        f"position={position}",
    )


def episode_bank_surface_targets(bank: EpisodeBank) -> tuple[SurfaceLeakageTarget, ...]:
    """Build every registered leakage target from an EpisodeBank.

    Features are rebuilt from an allowlist.  No canonical content is first
    serialized and then redacted, so a new sensitive field cannot silently
    survive masking.
    """

    if type(bank) is not EpisodeBank:
        raise TypeError("episode_bank_surface_targets requires an EpisodeBank")

    target_label: list[SurfaceLeakageSample] = []
    formula_stratum: list[SurfaceLeakageSample] = []
    placard_error_target: list[SurfaceLeakageSample] = []
    placard_error_status: list[SurfaceLeakageSample] = []
    shadow_error_status: list[SurfaceLeakageSample] = []

    for episode_index, (request, episode) in enumerate(
        zip(bank.spec.requests, bank.episodes, strict=True)
    ):
        group_id = f"episode-{episode_index}"
        episode_features = _episode_surface_features(
            renderer=episode.renderer,
            opening_count=len(episode.opening),
            terminal_count=len(episode.terminal),
        )
        formula_stratum.append(
            SurfaceLeakageSample(
                sample_id=f"{group_id}-formula",
                group_id=group_id,
                features=episode_features,
                label="exactly_one" if request.target_op == "exactly_one" else "all_or_any",
            )
        )
        if request.noisy_placard_error_target is not None:
            placard_error_target.append(
                SurfaceLeakageSample(
                    sample_id=f"{group_id}-placard-error-target",
                    group_id=group_id,
                    features=episode_features,
                    label=_bool_label(request.noisy_placard_error_target),
                )
            )

        for phase, observations in (("opening", episode.opening), ("terminal", episode.terminal)):
            for position, observation in enumerate(observations):
                sample_id = f"{group_id}-{phase}-{position}"
                features = _observation_surface_features(
                    episode_features,
                    phase=phase,
                    phase_count=len(observations),
                    position=position,
                )
                is_sun = observation.scene_index < NONEMPTY_ARRANGEMENT_COUNT
                target_label.append(
                    SurfaceLeakageSample(
                        sample_id=sample_id,
                        group_id=group_id,
                        features=features,
                        label=_bool_label(observation.accepted),
                    )
                )
                placard_error_status.append(
                    SurfaceLeakageSample(
                        sample_id=sample_id,
                        group_id=group_id,
                        features=features,
                        label=_bool_label(is_sun is not observation.accepted),
                    )
                )
                shadow_error_status.append(
                    SurfaceLeakageSample(
                        sample_id=sample_id,
                        group_id=group_id,
                        features=features,
                        label=_bool_label(
                            episode.shadow.truth[observation.scene_index]
                            is not observation.accepted
                        ),
                    )
                )

    binary = ("false", "true")
    targets = (
        SurfaceLeakageTarget("target_label", binary, tuple(target_label)),
        SurfaceLeakageTarget(
            "target_formula_stratum",
            ("all_or_any", "exactly_one"),
            tuple(formula_stratum),
        ),
        SurfaceLeakageTarget(
            "placard_error_target",
            binary,
            tuple(placard_error_target),
        ),
        SurfaceLeakageTarget(
            "placard_error_status",
            binary,
            tuple(placard_error_status),
        ),
        SurfaceLeakageTarget(
            "shadow_error_status",
            binary,
            tuple(shadow_error_status),
        ),
    )
    if tuple(target.name for target in targets) != EPISODE_BANK_SURFACE_TARGETS:
        raise RuntimeError("EpisodeBank surface-target registry is inconsistent")
    return targets


def _wilson_upper(probability: float, sample_count: int) -> float:
    z = NormalDist().inv_cdf(0.5 + SURFACE_LEAKAGE_CONFIDENCE_LEVEL / 2.0)
    z_squared = z * z
    denominator = 1.0 + z_squared / sample_count
    center = (probability + z_squared / (2.0 * sample_count)) / denominator
    half_width = (
        z
        * math.sqrt(
            probability * (1.0 - probability) / sample_count
            + z_squared / (4.0 * sample_count * sample_count)
        )
        / denominator
    )
    return center + half_width


def minimum_resolution_groups(class_count: int) -> int:
    """Return the fixed group-count precondition for resolving chance + .05.

    This is the first ``n`` whose 95% Wilson upper endpoint at chance is
    strictly below ``chance + .05``.  It is an adequacy check, not the reported
    uncertainty interval.
    """

    if isinstance(class_count, bool) or not isinstance(class_count, int) or class_count < 2:
        raise ValueError("class_count must be an integer of at least two")
    chance = 1.0 / class_count
    ceiling = chance + SURFACE_LEAKAGE_CHANCE_MARGIN
    for sample_count in range(1, 10_000_001):
        if _wilson_upper(chance, sample_count) < ceiling:
            return sample_count
    raise RuntimeError("failed to resolve the registered chance margin")


def _target_counts(
    target: SurfaceLeakageTarget,
) -> tuple[Counter[str], Counter[str], dict[str, tuple[int, ...]]]:
    class_counts = Counter(sample.label for sample in target.samples)
    groups: dict[str, list[int]] = defaultdict(list)
    for index, sample in enumerate(target.samples):
        groups[sample.group_id].append(index)
    class_group_counts = Counter(
        label
        for indices in groups.values()
        for label in {target.samples[index].label for index in indices}
    )
    return class_counts, class_group_counts, {
        group_id: tuple(indices) for group_id, indices in groups.items()
    }


def _stable_group_key(target_name: str, group_id: str) -> bytes:
    return hashlib.sha256(
        b"goalzendo-surface-leakage-group-order-v1\0"
        + target_name.encode("ascii")
        + b"\0"
        + group_id.encode("ascii")
    ).digest()


def _assign_group_folds(
    target: SurfaceLeakageTarget,
    groups: dict[str, tuple[int, ...]],
    *,
    fold_count: int,
) -> dict[str, int]:
    class_index = {label: index for index, label in enumerate(target.classes)}
    group_vectors: dict[str, tuple[int, ...]] = {}
    buckets: dict[int, list[str]] = defaultdict(list)
    for group_id, indices in groups.items():
        counts = Counter(target.samples[index].label for index in indices)
        vector = tuple(counts[label] for label in target.classes)
        group_vectors[group_id] = vector
        primary = max(range(len(target.classes)), key=lambda index: (vector[index], -index))
        buckets[primary].append(group_id)

    fold_class_counts = [[0] * len(target.classes) for _ in range(fold_count)]
    fold_sample_counts = [0] * fold_count
    fold_group_counts = [0] * fold_count
    assignments: dict[str, int] = {}
    for primary in range(len(target.classes)):
        ordered = sorted(
            buckets[primary],
            key=lambda group_id: (
                -sum(group_vectors[group_id]),
                tuple(-count for count in group_vectors[group_id]),
                _stable_group_key(target.name, group_id),
            ),
        )
        for group_id in ordered:
            vector = group_vectors[group_id]
            fold = min(
                range(fold_count),
                key=lambda candidate: (
                    fold_class_counts[candidate][primary],
                    fold_sample_counts[candidate],
                    fold_group_counts[candidate],
                    candidate,
                ),
            )
            assignments[group_id] = fold
            fold_group_counts[fold] += 1
            fold_sample_counts[fold] += sum(vector)
            for index, count in enumerate(vector):
                fold_class_counts[fold][index] += count
    if set(assignments) != set(groups):
        raise RuntimeError("grouped fold assignment dropped a group")
    if set(class_index) != set(target.classes):
        raise RuntimeError("surface target class index is inconsistent")
    return assignments


def _fold_coverage_reasons(
    target: SurfaceLeakageTarget,
    assignments: dict[str, int],
    *,
    fold_count: int,
) -> tuple[str, ...]:
    reasons: list[str] = []
    all_classes = set(target.classes)
    for fold in range(fold_count):
        test_classes = {
            sample.label for sample in target.samples if assignments[sample.group_id] == fold
        }
        train_classes = {
            sample.label for sample in target.samples if assignments[sample.group_id] != fold
        }
        missing_test = sorted(all_classes.difference(test_classes))
        missing_train = sorted(all_classes.difference(train_classes))
        if missing_test:
            reasons.append(f"fold_{fold}_test_missing_classes:{','.join(missing_test)}")
        if missing_train:
            reasons.append(f"fold_{fold}_train_missing_classes:{','.join(missing_train)}")
    return tuple(reasons)


def _predict_naive_bayes(
    train_samples: tuple[SurfaceLeakageSample, ...],
    test_samples: tuple[SurfaceLeakageSample, ...],
    *,
    classes: tuple[str, ...],
    alpha: float,
) -> tuple[str, ...]:
    vocabulary = sorted({feature for sample in train_samples for feature in sample.features})
    token_counts = {label: Counter[str]() for label in classes}
    totals = {label: 0 for label in classes}
    for sample in train_samples:
        token_counts[sample.label].update(sample.features)
        totals[sample.label] += len(sample.features)
    width = len(vocabulary) + 1
    predictions: list[str] = []
    for sample in test_samples:
        scores: list[tuple[float, str]] = []
        for label in classes:
            denominator = totals[label] + alpha * width
            score = sum(
                math.log((token_counts[label][feature] + alpha) / denominator)
                for feature in sample.features
            )
            scores.append((score, label))
        best_score = max(score for score, _ in scores)
        predictions.append(min(label for score, label in scores if score == best_score))
    return tuple(predictions)


def _cross_validated_predictions(
    target: SurfaceLeakageTarget,
    assignments: dict[str, int],
    *,
    config: SurfaceLeakageConfig,
) -> tuple[tuple[str, ...], str, str]:
    predictions: list[str | None] = [None] * len(target.samples)
    for fold in range(config.fold_count):
        train_indices = tuple(
            index
            for index, sample in enumerate(target.samples)
            if assignments[sample.group_id] != fold
        )
        test_indices = tuple(
            index
            for index, sample in enumerate(target.samples)
            if assignments[sample.group_id] == fold
        )
        train = tuple(target.samples[index] for index in train_indices)
        test = tuple(target.samples[index] for index in test_indices)
        predicted = _predict_naive_bayes(
            train,
            test,
            classes=target.classes,
            alpha=float(config.laplace_alpha),
        )
        for index, label in zip(test_indices, predicted, strict=True):
            predictions[index] = label
    if any(label is None for label in predictions):
        raise RuntimeError("cross-validation failed to predict every sample")
    complete = tuple(label for label in predictions if label is not None)
    fold_fields = tuple(
        f"{group_id}:{assignments[group_id]}" for group_id in sorted(assignments)
    )
    prediction_fields = tuple(
        f"{sample.sample_id}:{sample.label}:{prediction}"
        for sample, prediction in zip(target.samples, complete, strict=True)
    )
    return (
        complete,
        _digest_fields(b"goalzendo-surface-leakage-folds-v1\0", fold_fields),
        _digest_fields(
            b"goalzendo-surface-leakage-predictions-v1\0", prediction_fields
        ),
    )


def _balanced_accuracy(
    labels: tuple[str, ...],
    predictions: tuple[str, ...],
    *,
    classes: tuple[str, ...],
) -> float:
    recalls: list[float] = []
    for label in classes:
        indices = [index for index, observed in enumerate(labels) if observed == label]
        if not indices:
            raise ValueError(f"cannot score absent class {label!r}")
        recalls.append(sum(predictions[index] == label for index in indices) / len(indices))
    return math.fsum(recalls) / len(recalls)


def _bootstrap_interval(
    target: SurfaceLeakageTarget,
    predictions: tuple[str, ...],
    groups: dict[str, tuple[int, ...]],
    *,
    replicate_count: int,
) -> tuple[float, float] | None:
    class_index = {label: index for index, label in enumerate(target.classes)}
    ordered_groups = sorted(groups)
    correct = np.zeros((len(ordered_groups), len(target.classes)), dtype=np.int64)
    total = np.zeros_like(correct)
    for group_position, group_id in enumerate(ordered_groups):
        for sample_index in groups[group_id]:
            label_index = class_index[target.samples[sample_index].label]
            total[group_position, label_index] += 1
            correct[group_position, label_index] += int(
                predictions[sample_index] == target.samples[sample_index].label
            )

    seed_material = hashlib.sha256(
        b"goalzendo-surface-leakage-bootstrap-v1\0"
        + target.digest.encode("ascii")
        + b"\0"
        + str(replicate_count).encode("ascii")
    ).digest()
    seed = int.from_bytes(seed_material[:16], "big")
    generator = np.random.Generator(np.random.PCG64(seed))
    values: list[np.ndarray[Any, np.dtype[np.float64]]] = []
    valid_count = 0
    attempted = 0
    maximum_attempts = replicate_count * 10
    batch_size = max(1, min(256, 1_000_000 // len(ordered_groups)))
    while valid_count < replicate_count and attempted < maximum_attempts:
        batch = min(batch_size, maximum_attempts - attempted)
        draws = generator.integers(
            0,
            len(ordered_groups),
            size=(batch, len(ordered_groups)),
            dtype=np.int64,
        )
        sampled_total = total[draws].sum(axis=1)
        sampled_correct = correct[draws].sum(axis=1)
        valid = np.all(sampled_total > 0, axis=1)
        if np.any(valid):
            scores = np.asarray(
                np.mean(
                    sampled_correct[valid] / sampled_total[valid],
                    axis=1,
                    dtype=np.float64,
                ),
                dtype=np.float64,
            ).reshape(-1)
            needed = replicate_count - valid_count
            values.append(scores[:needed])
            valid_count += min(len(scores), needed)
        attempted += batch
    if valid_count < replicate_count:
        return None
    distribution = np.concatenate(values)
    tail = (1.0 - SURFACE_LEAKAGE_CONFIDENCE_LEVEL) / 2.0
    lower, upper = np.quantile(distribution, (tail, 1.0 - tail), method="linear")
    return float(lower), float(upper)


def _insufficient_result(
    target: SurfaceLeakageTarget,
    *,
    config: SurfaceLeakageConfig,
    class_counts: Counter[str],
    class_group_counts: Counter[str],
    minimum_groups_required: int,
    reasons: tuple[str, ...],
) -> SurfaceLeakageTargetResult:
    chance = 1.0 / len(target.classes)
    return SurfaceLeakageTargetResult(
        target_name=target.name,
        decision="insufficient_data",
        passed=False,
        sample_count=len(target.samples),
        group_count=len({sample.group_id for sample in target.samples}),
        class_counts=tuple((label, class_counts[label]) for label in target.classes),
        class_group_counts=tuple(
            (label, class_group_counts[label]) for label in target.classes
        ),
        fold_count=config.fold_count,
        minimum_groups_required=minimum_groups_required,
        chance=chance,
        chance_ceiling=chance + SURFACE_LEAKAGE_CHANCE_MARGIN,
        balanced_accuracy=None,
        interval_lower=None,
        interval_upper=None,
        chance_in_interval=None,
        upper_below_ceiling=None,
        fold_digest=None,
        prediction_digest=None,
        reasons=reasons,
    )


def audit_surface_target(
    target: SurfaceLeakageTarget,
    *,
    config: SurfaceLeakageConfig | None = None,
) -> SurfaceLeakageTargetResult:
    """Run the frozen audit for one registered target."""

    if type(target) is not SurfaceLeakageTarget:
        raise TypeError("audit_surface_target requires a SurfaceLeakageTarget")
    if config is None:
        config = SurfaceLeakageConfig()
    if type(config) is not SurfaceLeakageConfig:
        raise TypeError("config must be a SurfaceLeakageConfig")

    class_counts, class_group_counts, groups = _target_counts(target)
    resolution_minimum = minimum_resolution_groups(len(target.classes))
    minimum_groups_required = max(
        resolution_minimum,
        config.additional_minimum_groups,
    )
    minimum_class_groups = max(
        config.fold_count,
        math.ceil(minimum_groups_required / len(target.classes)),
    )
    reasons: list[str] = []
    missing_classes = [label for label in target.classes if class_counts[label] == 0]
    if missing_classes:
        reasons.append(f"missing_classes:{','.join(missing_classes)}")
    if len(target.samples) < minimum_groups_required:
        reasons.append(
            f"sample_count_below_resolution_minimum:{len(target.samples)}<{minimum_groups_required}"
        )
    if len(groups) < minimum_groups_required:
        reasons.append(
            f"independent_group_count_below_resolution_minimum:{len(groups)}<{minimum_groups_required}"
        )
    for label in target.classes:
        if class_counts[label] < config.fold_count:
            reasons.append(
                f"class_sample_count_below_fold_count:{label}:{class_counts[label]}<{config.fold_count}"
            )
        if class_group_counts[label] < minimum_class_groups:
            reasons.append(
                "class_group_count_below_required_coverage:"
                f"{label}:{class_group_counts[label]}<{minimum_class_groups}"
            )
    if reasons:
        return _insufficient_result(
            target,
            config=config,
            class_counts=class_counts,
            class_group_counts=class_group_counts,
            minimum_groups_required=minimum_groups_required,
            reasons=tuple(reasons),
        )

    assignments = _assign_group_folds(target, groups, fold_count=config.fold_count)
    coverage_reasons = _fold_coverage_reasons(
        target,
        assignments,
        fold_count=config.fold_count,
    )
    if coverage_reasons:
        return _insufficient_result(
            target,
            config=config,
            class_counts=class_counts,
            class_group_counts=class_group_counts,
            minimum_groups_required=minimum_groups_required,
            reasons=coverage_reasons,
        )

    predictions, fold_digest, prediction_digest = _cross_validated_predictions(
        target,
        assignments,
        config=config,
    )
    labels = tuple(sample.label for sample in target.samples)
    accuracy = _balanced_accuracy(labels, predictions, classes=target.classes)
    interval = _bootstrap_interval(
        target,
        predictions,
        groups,
        replicate_count=config.bootstrap_replicates,
    )
    if interval is None:
        return _insufficient_result(
            target,
            config=config,
            class_counts=class_counts,
            class_group_counts=class_group_counts,
            minimum_groups_required=minimum_groups_required,
            reasons=("bootstrap_class_coverage_insufficient",),
        )
    lower, upper = interval
    chance = 1.0 / len(target.classes)
    ceiling = chance + SURFACE_LEAKAGE_CHANCE_MARGIN
    chance_in_interval = lower <= chance <= upper
    upper_below_ceiling = upper < ceiling
    passed = chance_in_interval and upper_below_ceiling
    failure_reasons: tuple[str, ...] = ()
    if not passed:
        items: list[str] = []
        if not chance_in_interval:
            items.append("chance_not_in_95pct_interval")
        if not upper_below_ceiling:
            items.append("interval_upper_not_below_chance_plus_0p05")
        failure_reasons = tuple(items)
    return SurfaceLeakageTargetResult(
        target_name=target.name,
        decision="pass" if passed else "leakage",
        passed=passed,
        sample_count=len(target.samples),
        group_count=len(groups),
        class_counts=tuple((label, class_counts[label]) for label in target.classes),
        class_group_counts=tuple(
            (label, class_group_counts[label]) for label in target.classes
        ),
        fold_count=config.fold_count,
        minimum_groups_required=minimum_groups_required,
        chance=chance,
        chance_ceiling=ceiling,
        balanced_accuracy=accuracy,
        interval_lower=lower,
        interval_upper=upper,
        chance_in_interval=chance_in_interval,
        upper_below_ceiling=upper_below_ceiling,
        fold_digest=fold_digest,
        prediction_digest=prediction_digest,
        reasons=failure_reasons,
    )


def audit_surface_targets(
    dataset_id: str,
    targets: tuple[SurfaceLeakageTarget, ...],
    *,
    config: SurfaceLeakageConfig | None = None,
) -> SurfaceLeakageAuditReport:
    """Audit every supplied target, including future registered targets."""

    _require_nonempty_ascii(dataset_id, name="dataset_id")
    targets = tuple(targets)
    if not targets or any(type(target) is not SurfaceLeakageTarget for target in targets):
        raise ValueError("surface leakage dataset requires registered targets")
    names = [target.name for target in targets]
    if len(names) != len(set(names)):
        raise ValueError("surface leakage target names must be unique")
    if config is None:
        config = SurfaceLeakageConfig()
    if type(config) is not SurfaceLeakageConfig:
        raise TypeError("config must be a SurfaceLeakageConfig")
    dataset_digest = _digest_fields(
        b"goalzendo-surface-leakage-dataset-v1\0",
        (dataset_id, *(target.digest for target in targets)),
    )
    return SurfaceLeakageAuditReport(
        dataset_id=dataset_id,
        dataset_digest=dataset_digest,
        mask_policy=MASK_POLICY,
        statistical_method=STATISTICAL_METHOD,
        confidence_level=SURFACE_LEAKAGE_CONFIDENCE_LEVEL,
        chance_margin=SURFACE_LEAKAGE_CHANCE_MARGIN,
        config=config,
        results=tuple(audit_surface_target(target, config=config) for target in targets),
    )


def audit_episode_bank_surface_leakage(
    bank: EpisodeBank,
    *,
    config: SurfaceLeakageConfig | None = None,
) -> SurfaceLeakageAuditReport:
    """Build the masked registry and run the complete fail-closed G03-E audit."""

    if type(bank) is not EpisodeBank:
        raise TypeError("audit_episode_bank_surface_leakage requires an EpisodeBank")
    return audit_surface_targets(
        bank.spec.bank_id,
        episode_bank_surface_targets(bank),
        config=config,
    )
