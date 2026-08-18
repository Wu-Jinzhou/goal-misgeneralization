"""Matched causal interventions on GoalZendo candidate rules and scenes."""

from __future__ import annotations

import itertools
from dataclasses import dataclass, replace
from typing import Literal

from .rules import evaluate_rule
from .schema import KnownLawDecision, RuleSpec, Scene, stable_digest


@dataclass(frozen=True)
class InterventionAudit:
    """Exact semantic and feature changes in a paired intervention."""

    changed_channels: frozenset[str]
    changed_feature_groups: frozenset[str]
    changed_feature_indices: tuple[tuple[int, ...], tuple[int, ...]]
    herald_label_changed: bool
    on_manifold: bool


def _condition(previous: str, added: str) -> str:
    return added if previous == "none" else f"{previous}+{added}"


def _scene_with_features(scene: Scene, features: tuple[bool, ...], tag: str) -> Scene:
    if features == scene.features:
        return scene
    payload = {
        "parent_scene_id": scene.scene_id,
        "intervention": tag,
        "features": list(features),
    }
    return Scene(
        scene_id=f"scene-{stable_digest(payload, length=20)}",
        features=features,
    )


def _minimum_hamming_edits(
    features: tuple[bool, ...],
    rule: RuleSpec,
    *,
    outcome: Literal["reverse", "preserve"],
) -> tuple[tuple[tuple[int, ...], tuple[bool, ...]], ...]:
    """Enumerate every minimum-Hamming nonempty edit with the requested outcome."""

    original = evaluate_rule(rule, features)
    active = tuple(sorted(rule.feature_indices))
    for distance in range(1, len(active) + 1):
        candidates: list[tuple[tuple[int, ...], tuple[bool, ...]]] = []
        for changed_indices in itertools.combinations(active, distance):
            candidate = list(features)
            for index in changed_indices:
                candidate[index] = not candidate[index]
            candidate_tuple = tuple(candidate)
            changed = evaluate_rule(rule, candidate_tuple) is not original
            if changed == (outcome == "reverse"):
                candidates.append((changed_indices, candidate_tuple))
        if candidates:
            return tuple(candidates)
    requested = "opposite" if outcome == "reverse" else "same"
    raise ValueError(f"Rule has no reachable nonempty edit with the {requested} output")


def _selection_identity(decision: KnownLawDecision) -> str:
    """Use one identity for both members of a registered A/B mirror pair."""

    return decision.mirror_pair_id or decision.sample_id


def _keyed_edit(
    decision: KnownLawDecision,
    koan_index: int,
    rule: RuleSpec,
    *,
    tag: str,
    outcome: Literal["reverse", "preserve"],
    require_one_atom: bool = False,
) -> tuple[bool, ...]:
    """Select reproducibly among all equally local valid edits.

    The koan ID, rather than its displayed A/B position, is part of the key.
    Consequently an exact registered mirror receives the same semantic edit,
    while different samples and koans spread ties across eligible features.
    """

    koan = decision.koans[koan_index]
    candidates = _minimum_hamming_edits(koan.scene.features, rule, outcome=outcome)
    if require_one_atom and len(candidates[0][0]) != 1:
        raise ValueError(
            f"{tag} requires a one-atom boundary scene for rule family {rule.family!r}"
        )
    key = {
        "schema": 1,
        "selection_identity": _selection_identity(decision),
        "koan_id": koan.koan_id,
        "tag": tag,
        "rule": rule.as_dict(),
        "outcome": outcome,
    }
    selected = int(stable_digest(key), 16) % len(candidates)
    return candidates[selected][1]


def _one_atom_boundary_crossing(
    features: tuple[bool, ...],
    rule: RuleSpec,
) -> tuple[bool, ...] | None:
    """Return whether a one-active-atom reversal exists, retaining old API shape."""

    try:
        candidates = _minimum_hamming_edits(features, rule, outcome="reverse")
    except ValueError:
        return None
    return candidates[0][1] if len(candidates[0][0]) == 1 else None


def is_one_atom_law_boundary(decision: KnownLawDecision) -> bool:
    """Whether both koans admit a one-atom Official-Law label reversal."""

    return all(
        _one_atom_boundary_crossing(koan.scene.features, decision.law) is not None
        for koan in decision.koans
    )


def flip_herald(decision: KnownLawDecision) -> KnownLawDecision:
    """Reverse only the direct Herald stamp, preserving scene content exactly."""

    left, right = decision.koans
    changed = replace(
        decision,
        koans=(
            replace(left, herald_accepts=not left.herald_accepts),
            replace(right, herald_accepts=not right.herald_accepts),
        ),
        intervention=_condition(decision.intervention, "flip_P"),
    )
    audit = audit_intervention(decision, changed)
    if audit.changed_channels != {"P"} or audit.changed_feature_indices != ((), ()):
        raise RuntimeError("Herald intervention failed its causal isolation check")
    return changed


def _flip_semantic_rule(
    decision: KnownLawDecision,
    rule: RuleSpec,
    *,
    tag: str,
    expected_channel: str,
    require_one_atom: bool = False,
) -> KnownLawDecision:
    changed_koans = []
    for koan_index, koan in enumerate(decision.koans):
        try:
            features = _keyed_edit(
                decision,
                koan_index,
                rule,
                tag=tag,
                outcome="reverse",
                require_one_atom=require_one_atom,
            )
        except ValueError as exc:
            if require_one_atom:
                raise ValueError(
                    f"{tag} requires a one-atom boundary scene for rule family {rule.family!r}"
                ) from exc
            raise
        changed_koans.append(
            replace(
                koan,
                scene=_scene_with_features(koan.scene, features, tag),
            )
        )
    changed = replace(
        decision,
        koans=(changed_koans[0], changed_koans[1]),
        intervention=_condition(decision.intervention, tag),
    )
    audit = audit_intervention(decision, changed)
    if (
        audit.changed_channels != {expected_channel}
        or audit.changed_feature_groups != {expected_channel}
        or audit.herald_label_changed
    ):
        raise RuntimeError(f"{tag} did not isolate the {expected_channel} candidate")
    return changed


def flip_sage(decision: KnownLawDecision) -> KnownLawDecision:
    """Reverse semantic Sage choice Q without changing Law choice Y."""

    return _flip_semantic_rule(
        decision,
        decision.sage_rule,
        tag="flip_Q",
        expected_channel="Q",
    )


def flip_law(decision: KnownLawDecision) -> KnownLawDecision:
    """Reverse official Law choice Y without changing semantic Sage choice Q."""

    return _flip_semantic_rule(
        decision,
        decision.law,
        tag="flip_Y",
        expected_channel="Y",
        require_one_atom=decision.law.family == "majority",
    )


def _preserve_semantic_rule(
    decision: KnownLawDecision,
    rule: RuleSpec,
    *,
    tag: str,
    active_group: str,
) -> KnownLawDecision:
    """Edit an active rule feature while preserving every candidate choice."""

    changed_koans = []
    for koan_index, koan in enumerate(decision.koans):
        features = _keyed_edit(
            decision,
            koan_index,
            rule,
            tag=tag,
            outcome="preserve",
        )
        changed_koans.append(
            replace(koan, scene=_scene_with_features(koan.scene, features, tag))
        )
    changed = replace(
        decision,
        koans=(changed_koans[0], changed_koans[1]),
        intervention=_condition(decision.intervention, tag),
    )
    audit = audit_intervention(decision, changed)
    if audit.changed_channels or audit.changed_feature_groups != {active_group}:
        raise RuntimeError(f"{tag} failed its rule-preserving active-feature check")
    return changed


def preserve_law(decision: KnownLawDecision) -> KnownLawDecision:
    """Change minimum-Hamming Law-active features without changing Y, P, or Q."""

    return _preserve_semantic_rule(
        decision,
        decision.law,
        tag="preserve_Y",
        active_group="Y",
    )


def preserve_sage(decision: KnownLawDecision) -> KnownLawDecision:
    """Change minimum-Hamming Sage-active features without changing Y, P, or Q."""

    return _preserve_semantic_rule(
        decision,
        decision.sage_rule,
        tag="preserve_Q",
        active_group="Q",
    )


def flip_distractor(
    decision: KnownLawDecision,
    *,
    feature_index: int | None = None,
) -> KnownLawDecision:
    """Change one inactive feature in both scenes without changing Y, P, or Q."""

    active = set(decision.law.feature_indices) | set(decision.sage_rule.feature_indices)
    available = [index for index in range(decision.feature_count) if index not in active]
    if feature_index is None:
        if not available:
            raise ValueError("A distractor intervention requires an inactive scene feature")
        key = {
            "schema": 1,
            "selection_identity": _selection_identity(decision),
            "tag": "flip_D",
            "available": available,
        }
        selected = available[int(stable_digest(key), 16) % len(available)]
    else:
        if isinstance(feature_index, bool) or not isinstance(feature_index, int):
            raise ValueError("feature_index must be an integer")
        if feature_index not in available:
            raise ValueError("feature_index must be inactive under both Law and Sage rules")
        selected = feature_index

    changed_koans = []
    for koan in decision.koans:
        features = list(koan.scene.features)
        features[selected] = not features[selected]
        changed_koans.append(
            replace(
                koan,
                scene=_scene_with_features(
                    koan.scene,
                    tuple(features),
                    f"flip_distractor_{selected}",
                ),
            )
        )
    changed = replace(
        decision,
        koans=(changed_koans[0], changed_koans[1]),
        intervention=_condition(decision.intervention, f"flip_D{selected}"),
    )
    audit = audit_intervention(decision, changed)
    if audit.changed_channels or audit.changed_feature_groups != {"distractor"}:
        raise RuntimeError("Distractor intervention changed a candidate rule")
    return changed


def audit_intervention(
    original: KnownLawDecision,
    changed: KnownLawDecision,
) -> InterventionAudit:
    """Audit paired identity, edited features, and candidate-choice changes."""

    if original.sample_id != changed.sample_id or original.split != changed.split:
        raise ValueError("Paired interventions must preserve sample identity and split")
    if original.law != changed.law or original.sage_rule != changed.sage_rule:
        raise ValueError("Paired interventions must preserve the Law and Sage rule")

    law_indices = set(original.law.feature_indices)
    sage_indices = set(original.sage_rule.feature_indices)
    feature_changes: list[tuple[int, ...]] = []
    herald_changed = False
    feature_groups: set[str] = set()
    for base_koan, changed_koan in zip(original.koans, changed.koans, strict=True):
        if base_koan.koan_id != changed_koan.koan_id:
            raise ValueError("Paired interventions must preserve koan identity")
        if len(base_koan.scene.features) != len(changed_koan.scene.features):
            raise ValueError("Paired interventions must preserve scene width")
        indices = tuple(
            index
            for index, (base_value, changed_value) in enumerate(
                zip(base_koan.scene.features, changed_koan.scene.features, strict=True)
            )
            if base_value != changed_value
        )
        if indices and base_koan.scene.scene_id == changed_koan.scene.scene_id:
            raise ValueError("Changed scene content requires a new deterministic scene ID")
        if not indices and base_koan.scene != changed_koan.scene:
            raise ValueError("An unchanged scene must retain its complete identity")
        feature_changes.append(indices)
        herald_changed |= base_koan.herald_accepts != changed_koan.herald_accepts
        for index in indices:
            if index in law_indices:
                feature_groups.add("Y")
            elif index in sage_indices:
                feature_groups.add("Q")
            else:
                feature_groups.add("distractor")

    changed_channels: set[str] = set()
    if original.choice_y != changed.choice_y:
        changed_channels.add("Y")
    if original.choice_p != changed.choice_p:
        changed_channels.add("P")
    if original.choice_q != changed.choice_q:
        changed_channels.add("Q")
    if herald_changed and "P" not in changed_channels:
        raise ValueError("Herald labels changed without reversing the Herald choice")
    return InterventionAudit(
        changed_channels=frozenset(changed_channels),
        changed_feature_groups=frozenset(feature_groups),
        changed_feature_indices=(feature_changes[0], feature_changes[1]),
        herald_label_changed=herald_changed,
        # KnownLawDecision validates exactly one positive koan for both rules.
        on_manifold=True,
    )


def changed_channels(
    original: KnownLawDecision,
    changed: KnownLawDecision,
) -> frozenset[str]:
    """Backward-compatible shorthand for the full intervention audit."""

    return audit_intervention(original, changed).changed_channels


def paired_channel_flips(decision: KnownLawDecision) -> dict[str, KnownLawDecision]:
    """Backward-compatible P/Q intervention panel."""

    return {"flip_P": flip_herald(decision), "flip_Q": flip_sage(decision)}


def paired_interventions(decision: KnownLawDecision) -> dict[str, KnownLawDecision]:
    """Return the complete P, Q, Y, and distractor causal panel."""

    return {
        "flip_P": flip_herald(decision),
        "flip_Q": flip_sage(decision),
        "flip_Y": flip_law(decision),
        "flip_D": flip_distractor(decision),
    }


def paired_rule_preserving_controls(
    decision: KnownLawDecision,
) -> dict[str, KnownLawDecision]:
    """Return optional active-feature controls, separate from controller endpoints.

    A literal rule has no nonempty active-feature edit that preserves its
    output. In that case the corresponding control is omitted rather than
    substituting a scientifically different perturbation.
    """

    controls: dict[str, KnownLawDecision] = {}
    for name, intervention in (("preserve_Y", preserve_law), ("preserve_Q", preserve_sage)):
        try:
            controls[name] = intervention(decision)
        except ValueError as exc:
            if "no reachable nonempty edit" not in str(exc):
                raise
    return controls
