"""A small, finite-choice hidden-law game with exact CPU-side audits.

This module deliberately contains no model, optimizer, or execution code.  It
constructs two deterministic game families from the released GoalZendo scene
and rule semantics, and it validates every property used by the experiment.
"""

from __future__ import annotations

import hashlib
import math
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from functools import cache, lru_cache
from itertools import combinations, pairwise
from typing import Any, Literal, TypeAlias, cast

from goalzendo_interactive._json import dump_json, json_digest, load_json
from goalzendo_interactive.catalog import truth_vector
from goalzendo_interactive.query import entropy_reduction
from goalzendo_interactive.rules import (
    ATOMS,
    Atom,
    BinaryRule,
    Rule,
    rule_from_obj,
)
from goalzendo_interactive.rules import (
    Literal as RuleLiteral,
)
from goalzendo_interactive.schema import (
    COLORS,
    POSITIONS,
    SCENE_COUNT,
    SHAPES,
    SIZES,
    Piece,
    Scene,
    scene_at,
    scene_index,
)

CandidateID = Literal["A", "B", "C", "D"]
SemanticRole = Literal["P", "Q", "M", "X"]
EvidenceLevel = Literal["perfect", "noisy"]
GamePhase = Literal["train", "evaluation"]
FamilyUse = Literal["both", "train", "evaluation"]
InterventionTarget = Literal["Y", "P", "Q", "R", "distractor"]

CANDIDATE_IDS: tuple[CandidateID, ...] = ("A", "B", "C", "D")
SEMANTIC_ROLES: tuple[SemanticRole, ...] = ("P", "Q", "M", "X")
EVIDENCE_LEVELS: tuple[EvidenceLevel, ...] = ("perfect", "noisy")
OPENING_SIZE = 10
QUERY_MENU_SIZE = 8
TERMINAL_SIZE = 16
MAX_QUERY_TURNS = 2
BANK_SCHEMA_VERSION = 1
PRODUCTION_BANK_SCHEMA_VERSION = 1
TRAINING_FAMILY_COUNT = 128
EVALUATION_FAMILY_COUNT = 16
INTERVENTION_TARGETS: tuple[InterventionTarget, ...] = ("Y", "P", "Q", "R", "distractor")

_OPAQUE_ORDER_DOMAIN = b"goalzendo-hidden-law-opaque-order-v1\0"
_SCENE_ORDER_DOMAIN = b"goalzendo-hidden-law-scene-order-v1\0"
_BANK_DIGEST_DOMAIN = "goalzendo-hidden-law-finite-bank-v1"
_PRODUCTION_BANK_DIGEST_DOMAIN = "goalzendo-hidden-law-production-bank-v1"
_RULE_SEARCH_DOMAIN = b"goalzendo-hidden-law-rule-search-v1\0"
_SCENE_PARTITION_DOMAIN = b"goalzendo-hidden-law-scene-partition-v1\0"
_INTERVENTION_ORDER_DOMAIN = b"goalzendo-hidden-law-intervention-order-v1\0"
_QUERY_ORDER_DOMAIN = b"goalzendo-hidden-law-query-order-v1\0"

# Relative to displayed candidates A, B, C, D. Four masks are balanced 2/2
# partitions and four are 1/3 partitions. Their visible positions vary by
# family; this tuple is a profile, not the displayed order.
_QUERY_MASKS = (0b0011, 0b0101, 0b0110, 0b1100, 0b0001, 0b0010, 0b0100, 0b1000)


def _query_mask_order(seed: int, stage: str, family_index: int) -> tuple[int, ...]:
    """Return a seed-bound permutation with exact stage-wide position balance."""

    ranked = tuple(
        sorted(
            _QUERY_MASKS,
            key=lambda mask: hashlib.sha256(
                _QUERY_ORDER_DOMAIN
                + seed.to_bytes(8, "big", signed=False)
                + stage.encode("ascii")
                + bytes((mask,))
            ).digest(),
        )
    )
    stage_count = {
        "train": TRAINING_FAMILY_COUNT,
        "evaluation": EVALUATION_FAMILY_COUNT,
    }.get(stage)
    if stage_count is None:
        offset = (family_index * (QUERY_MENU_SIZE // 2)) % QUERY_MENU_SIZE
    else:
        if not 0 <= family_index < stage_count:
            raise GameGenerationError("family index lies outside its query-order stage")
        ranked_indices = sorted(
            range(stage_count),
            key=lambda index: hashlib.sha256(
                _QUERY_ORDER_DOMAIN
                + seed.to_bytes(8, "big", signed=False)
                + stage.encode("ascii")
                + index.to_bytes(4, "big")
            ).digest(),
        )
        offset = ranked_indices.index(family_index) % QUERY_MENU_SIZE
    return ranked[offset:] + ranked[:offset]


class GameValidationError(ValueError):
    """Raised when a game object violates a scientific invariant."""


class GameGenerationError(RuntimeError):
    """Raised when deterministic finite search cannot satisfy the design."""


def _is_rule(value: object) -> bool:
    return type(value) in {RuleLiteral, BinaryRule}


def _uses_placard(rule: Rule) -> bool:
    literals = (rule,) if type(rule) is RuleLiteral else cast(BinaryRule, rule).args
    return any(literal.atom.op == "placard_is" for literal in literals)


def _opaque_role_order(family_id: str) -> tuple[SemanticRole, ...]:
    def key(role: SemanticRole) -> bytes:
        digest = hashlib.sha256()
        digest.update(_OPAQUE_ORDER_DOMAIN)
        digest.update(family_id.encode("ascii"))
        digest.update(b"\0")
        digest.update(role.encode("ascii"))
        return digest.digest()

    return tuple(sorted(SEMANTIC_ROLES, key=key))


def _require_ascii_id(value: object, *, name: str) -> str:
    if type(value) is not str or not value or not value.isascii():
        raise GameValidationError(f"{name} must be a nonempty ASCII string")
    return value


def _require_exact_keys(value: object, expected: set[str], *, name: str) -> dict[str, Any]:
    if type(value) is not dict or set(value) != expected or len(value) != len(expected):
        raise GameValidationError(f"{name} has noncanonical fields")
    return cast(dict[str, Any], value)


def _label(rule: Rule, scene_index: int) -> bool:
    return truth_vector(rule)[scene_index]


@dataclass(frozen=True, slots=True)
class Candidate:
    """One displayed hypothesis and its evaluator-only semantic role."""

    candidate_id: CandidateID
    role: SemanticRole
    rule: Rule

    def __post_init__(self) -> None:
        if self.candidate_id not in CANDIDATE_IDS:
            raise GameValidationError(f"unknown candidate id: {self.candidate_id!r}")
        if self.role not in SEMANTIC_ROLES:
            raise GameValidationError(f"unknown semantic role: {self.role!r}")
        if not _is_rule(self.rule):
            raise GameValidationError("candidate rule must use the released finite grammar")

        if self.role == "P":
            if type(self.rule) is not RuleLiteral or not _uses_placard(self.rule):
                raise GameValidationError("P must be a placard literal")
        elif self.role == "Q":
            if type(self.rule) is not RuleLiteral or _uses_placard(self.rule):
                raise GameValidationError("Q must be a piece literal")
        elif self.role == "M":
            if (
                type(self.rule) is not BinaryRule
                or self.rule.op not in {"all", "any"}
                or _uses_placard(self.rule)
                or any(literal.negated for literal in self.rule.args)
            ):
                raise GameValidationError("M must be a positive, piece-only all/any rule")
        elif (
            type(self.rule) is not BinaryRule
            or self.rule.op != "exactly_one"
            or _uses_placard(self.rule)
            or any(literal.negated for literal in self.rule.args)
        ):
            raise GameValidationError("X must be a positive, piece-only exactly_one rule")

    def as_obj(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "role": self.role,
            "rule": self.rule.as_obj(),
        }

    def as_public_obj(self) -> dict[str, Any]:
        return {"candidate_id": self.candidate_id, "criterion": self.rule.as_obj()}


@dataclass(frozen=True, slots=True)
class LabeledScene:
    scene_index: int
    accepted: bool

    def __post_init__(self) -> None:
        if (
            isinstance(self.scene_index, bool)
            or not isinstance(self.scene_index, int)
            or not 0 <= self.scene_index < SCENE_COUNT
        ):
            raise GameValidationError(f"scene_index must lie in [0, {SCENE_COUNT})")
        if type(self.accepted) is not bool:
            raise GameValidationError("scene label must be Boolean")

    @property
    def scene(self) -> Scene:
        return scene_at(self.scene_index)

    def as_obj(self) -> dict[str, int | bool]:
        return {"scene_index": self.scene_index, "accepted": self.accepted}

    def as_public_obj(self) -> dict[str, Any]:
        return {"scene": self.scene.as_obj(), "fits": self.accepted}


@dataclass(frozen=True, slots=True)
class GameMaterial:
    """The material visible before an interactive game begins."""

    opening: tuple[LabeledScene, ...]
    query_menu: tuple[int, ...]
    terminal: tuple[int, ...]

    def __post_init__(self) -> None:
        opening = tuple(self.opening)
        query_menu = tuple(self.query_menu)
        terminal = tuple(self.terminal)
        object.__setattr__(self, "opening", opening)
        object.__setattr__(self, "query_menu", query_menu)
        object.__setattr__(self, "terminal", terminal)
        if len(opening) != OPENING_SIZE or any(type(item) is not LabeledScene for item in opening):
            raise GameValidationError(f"opening must contain exactly {OPENING_SIZE} labeled scenes")
        for name, values, expected_size in (
            ("query menu", query_menu, QUERY_MENU_SIZE),
            ("terminal", terminal, TERMINAL_SIZE),
        ):
            if len(values) != expected_size:
                raise GameValidationError(f"{name} must contain exactly {expected_size} scenes")
            if any(
                isinstance(index, bool) or not isinstance(index, int) or not 0 <= index < SCENE_COUNT
                for index in values
            ):
                raise GameValidationError(f"{name} contains an invalid scene index")
            if len(set(values)) != len(values):
                raise GameValidationError(f"{name} scene indices must be unique")
        opening_indices = tuple(item.scene_index for item in opening)
        if len(set(opening_indices)) != len(opening_indices):
            raise GameValidationError("opening scene indices must be unique")
        if set(opening_indices) & (set(query_menu) | set(terminal)):
            raise GameValidationError("opening, menu, and terminal scenes must be disjoint")
        if set(query_menu) & set(terminal):
            raise GameValidationError("query-menu and terminal scenes must be disjoint")

    def as_obj(self) -> dict[str, Any]:
        return {
            "opening": [item.as_obj() for item in self.opening],
            "query_menu": list(self.query_menu),
            "terminal": list(self.terminal),
        }

    def as_public_obj(self, candidates: tuple[Candidate, ...]) -> dict[str, Any]:
        """Return only material available before inquiry.

        Terminal scenes are deliberately absent.  They are published one at a
        time through :meth:`GameInstance.isolated_terminal_bytes` after the
        inquiry transcript has been fixed.
        """

        return {
            "schema_version": BANK_SCHEMA_VERSION,
            "candidates": [candidate.as_public_obj() for candidate in candidates],
            "opening": [item.as_public_obj() for item in self.opening],
            "query_menu": [
                {"choice": f"Q{offset}", "scene": scene_at(index).as_obj()}
                for offset, index in enumerate(self.query_menu, start=1)
            ],
        }


@dataclass(frozen=True, slots=True)
class EvaluationCondition:
    condition_id: str
    p_evidence: EvidenceLevel
    q_evidence: EvidenceLevel
    opening: tuple[LabeledScene, ...]

    def __post_init__(self) -> None:
        expected_id = f"p-{self.p_evidence}_q-{self.q_evidence}"
        if self.p_evidence not in EVIDENCE_LEVELS or self.q_evidence not in EVIDENCE_LEVELS:
            raise GameValidationError("unknown evaluation evidence level")
        if self.condition_id != expected_id:
            raise GameValidationError(f"condition id must be {expected_id!r}")
        opening = tuple(self.opening)
        object.__setattr__(self, "opening", opening)
        if len(opening) != OPENING_SIZE or any(type(item) is not LabeledScene for item in opening):
            raise GameValidationError(f"evaluation opening must contain {OPENING_SIZE} scenes")
        if len({item.scene_index for item in opening}) != len(opening):
            raise GameValidationError("evaluation opening scenes must be unique")

    def as_obj(self) -> dict[str, Any]:
        return {
            "condition_id": self.condition_id,
            "p_evidence": self.p_evidence,
            "q_evidence": self.q_evidence,
            "opening": [item.as_obj() for item in self.opening],
        }


def _primitive_changed_field(before: Scene, after: Scene) -> str | None:
    """Return the sole allowed primitive edit, or ``None`` for a non-primitive pair."""

    changes: list[str] = []
    if before.placard != after.placard:
        changes.append("placard")
    for position in POSITIONS:
        first = before.piece_at(position)
        second = after.piece_at(position)
        if (first is None) != (second is None):
            return None
        if first is None:
            continue
        assert second is not None
        for attribute in ("color", "shape", "size"):
            if getattr(first, attribute) != getattr(second, attribute):
                changes.append(f"{position}.{attribute}")
    return changes[0] if len(changes) == 1 else None


@dataclass(frozen=True, slots=True)
class MatchedIntervention:
    """One ordered, one-field scene edit evaluated from a shared transcript."""

    target: InterventionTarget
    pair_index: int
    before_scene_index: int
    after_scene_index: int
    changed_field: str

    def __post_init__(self) -> None:
        if self.target not in INTERVENTION_TARGETS:
            raise GameValidationError(f"unknown intervention target: {self.target!r}")
        if self.pair_index not in {0, 1}:
            raise GameValidationError("intervention pair_index must be 0 or 1")
        for name in ("before_scene_index", "after_scene_index"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value < SCENE_COUNT:
                raise GameValidationError(f"{name} lies outside the scene universe")
        if self.before_scene_index == self.after_scene_index:
            raise GameValidationError("intervention endpoints must differ")
        observed = _primitive_changed_field(
            scene_at(self.before_scene_index), scene_at(self.after_scene_index)
        )
        if observed is None or observed != self.changed_field:
            raise GameValidationError("intervention endpoints do not reconstruct one registered field edit")

    def as_obj(self) -> dict[str, Any]:
        return {
            "target": self.target,
            "pair_index": self.pair_index,
            "before_scene_index": self.before_scene_index,
            "after_scene_index": self.after_scene_index,
            "changed_field": self.changed_field,
        }


def _candidate_mask(candidates: Sequence[Candidate], scene_index: int) -> int:
    return sum(
        int(_label(candidate.rule, scene_index)) << offset for offset, candidate in enumerate(candidates)
    )


def _consistent_candidate_ids(
    candidates: Sequence[Candidate], opening: Sequence[LabeledScene]
) -> tuple[CandidateID, ...]:
    return tuple(
        candidate.candidate_id
        for candidate in candidates
        if all(_label(candidate.rule, item.scene_index) is item.accepted for item in opening)
    )


@dataclass(frozen=True, slots=True)
class GameFamily:
    family_id: str
    candidates: tuple[Candidate, ...]
    training_material: GameMaterial
    training_official_ids: tuple[CandidateID, ...]
    evaluation_y_role: Literal["M", "X"]
    evaluation_conditions: tuple[EvaluationCondition, ...]
    study_use: FamilyUse = "both"
    matched_interventions: tuple[MatchedIntervention, ...] = ()

    def __post_init__(self) -> None:
        _require_ascii_id(self.family_id, name="family_id")
        candidates = tuple(self.candidates)
        officials = tuple(self.training_official_ids)
        conditions = tuple(self.evaluation_conditions)
        interventions = tuple(self.matched_interventions)
        object.__setattr__(self, "candidates", candidates)
        object.__setattr__(self, "training_official_ids", officials)
        object.__setattr__(self, "evaluation_conditions", conditions)
        object.__setattr__(self, "matched_interventions", interventions)
        if self.study_use not in {"both", "train", "evaluation"}:
            raise GameValidationError(f"unknown family use: {self.study_use!r}")
        if len(candidates) != 4 or any(type(candidate) is not Candidate for candidate in candidates):
            raise GameValidationError("a family requires exactly four candidates")
        if tuple(candidate.candidate_id for candidate in candidates) != CANDIDATE_IDS:
            raise GameValidationError("candidates must be displayed as A, B, C, D")
        if tuple(candidate.role for candidate in candidates) != _opaque_role_order(self.family_id):
            raise GameValidationError("candidate roles do not match the deterministic opaque order")
        if len({truth_vector(candidate.rule).bits for candidate in candidates}) != 4:
            raise GameValidationError("candidate hypotheses must be extensionally distinct")
        if any(not 0.25 <= truth_vector(candidate.rule).prevalence <= 0.75 for candidate in candidates):
            raise GameValidationError("every candidate prevalence must lie in [0.25, 0.75]")
        if type(self.training_material) is not GameMaterial:
            raise GameValidationError("training material must be a GameMaterial")
        expected_officials = CANDIDATE_IDS if self.study_use in {"both", "train"} else ()
        if officials != expected_officials:
            raise GameValidationError(
                "training families must rotate every candidate; evaluation families must not"
            )
        if self.evaluation_y_role not in {"M", "X"}:
            raise GameValidationError("evaluation Y must be M or X")
        expected_conditions = tuple(
            (p_evidence, q_evidence) for p_evidence in EVIDENCE_LEVELS for q_evidence in EVIDENCE_LEVELS
        )
        observed_conditions = tuple((item.p_evidence, item.q_evidence) for item in conditions)
        registered_conditions = expected_conditions if self.study_use in {"both", "evaluation"} else ()
        if observed_conditions != registered_conditions:
            raise GameValidationError("evaluation families must contain exactly the P-by-Q quartet")

        training = self.training_material
        if sum(item.accepted for item in training.opening) != OPENING_SIZE // 2:
            raise GameValidationError("training opening labels must be exactly 5/5")
        for item in training.opening:
            labels = {_label(candidate.rule, item.scene_index) for candidate in candidates}
            if labels != {item.accepted}:
                raise GameValidationError("every candidate must agree with every training opening label")
        if _consistent_candidate_ids(candidates, training.opening) != CANDIDATE_IDS:
            raise GameValidationError("training opening must leave V0=4")

        menu_masks = tuple(_candidate_mask(candidates, index) for index in training.query_menu)
        if Counter(menu_masks) != Counter(_QUERY_MASKS):
            raise GameValidationError("query menu does not have the registered information partitions")
        if menu_minimax_depth(self, CANDIDATE_IDS, max_depth=2) != 2:
            raise GameValidationError("query menu must identify all four candidates within depth two")
        inventory = terminal_cell_inventory(self)
        if inventory != {mask: 1 for mask in range(16)}:
            raise GameValidationError("terminal must contain one scene from every (A,B,C,D) truth cell")

        candidate_by_role = self.candidate_by_role
        y = candidate_by_role[self.evaluation_y_role]
        r_role: Literal["M", "X"] = "X" if self.evaluation_y_role == "M" else "M"
        r = candidate_by_role[r_role]
        p = candidate_by_role["P"]
        q = candidate_by_role["Q"]
        expected_live_sizes = {
            ("perfect", "perfect"): 4,
            ("perfect", "noisy"): 3,
            ("noisy", "perfect"): 3,
            ("noisy", "noisy"): 2,
        }
        training_opening_set = {item.scene_index for item in training.opening}
        evaluation_opening_sets: list[set[int]] = []
        for condition in conditions:
            if sum(item.accepted for item in condition.opening) != OPENING_SIZE // 2:
                raise GameValidationError("evaluation opening labels must be exactly 5/5")
            agreement = {
                candidate.role: sum(
                    _label(candidate.rule, item.scene_index) is item.accepted for item in condition.opening
                )
                for candidate in candidates
            }
            if agreement[y.role] != 10 or agreement[r.role] != 10:
                raise GameValidationError("Y and R must agree with all evaluation opening labels")
            if agreement[p.role] != (10 if condition.p_evidence == "perfect" else 9):
                raise GameValidationError("P evidence does not match its registered fidelity")
            if agreement[q.role] != (10 if condition.q_evidence == "perfect" else 8):
                raise GameValidationError("Q evidence does not match its registered fidelity")
            live = _consistent_candidate_ids(candidates, condition.opening)
            if len(live) != expected_live_sizes[(condition.p_evidence, condition.q_evidence)]:
                raise GameValidationError("evaluation opening has the wrong initial live-set size")
            if y.candidate_id not in live or r.candidate_id not in live:
                raise GameValidationError("evaluation opening eliminated Y or R")
            evaluation_opening_sets.append({item.scene_index for item in condition.opening})

        if conditions:
            by_evidence = {
                (condition.p_evidence, condition.q_evidence): condition.opening for condition in conditions
            }
            base = by_evidence[("perfect", "perfect")]
            p_noisy = by_evidence[("noisy", "perfect")]
            q_noisy = by_evidence[("perfect", "noisy")]
            both_noisy = by_evidence[("noisy", "noisy")]
            if any(
                len({opening[position].accepted for opening in by_evidence.values()}) != 1
                for position in range(OPENING_SIZE)
            ):
                raise GameValidationError("factorial openings changed a target label")
            p_positions = {
                position
                for position in range(OPENING_SIZE)
                if p_noisy[position].scene_index != base[position].scene_index
            }
            q_positions = {
                position
                for position in range(OPENING_SIZE)
                if q_noisy[position].scene_index != base[position].scene_index
            }
            if len(p_positions) != 1 or len(q_positions) != 2 or p_positions & q_positions:
                raise GameValidationError("factorial openings do not isolate one P and two Q swaps")
            for position in range(OPENING_SIZE):
                expected_both = (
                    p_noisy[position]
                    if position in p_positions
                    else q_noisy[position]
                    if position in q_positions
                    else base[position]
                )
                if both_noisy[position] != expected_both:
                    raise GameValidationError("factorial opening swaps are not shared across contrasts")
            evaluation_scene_union = set().union(*evaluation_opening_sets)
            if len(evaluation_scene_union) != OPENING_SIZE + len(p_positions) + len(q_positions):
                raise GameValidationError("factorial openings reuse scenes outside the registered base")
        else:
            evaluation_scene_union = set()

        common = set(training.query_menu) | set(training.terminal)
        if training_opening_set & (evaluation_scene_union | common):
            raise GameValidationError("training opening must be disjoint from evaluation material")
        if evaluation_scene_union & common:
            raise GameValidationError("all openings must be disjoint from the common menu and terminal")

        if self.study_use == "evaluation":
            expected_targets = tuple(
                (target, pair_index) for target in INTERVENTION_TARGETS for pair_index in (0, 1)
            )
            if tuple((item.target, item.pair_index) for item in interventions) != expected_targets:
                raise GameValidationError(
                    "evaluation families require two ordered pairs per intervention target"
                )
        elif interventions:
            raise GameValidationError("matched interventions belong only to held-out evaluation families")

        endpoint_indices = [
            index
            for record in interventions
            for index in (record.before_scene_index, record.after_scene_index)
        ]
        if len(set(endpoint_indices)) != len(endpoint_indices):
            raise GameValidationError("intervention endpoints must be globally unique within a family")
        material_indices = training_opening_set | evaluation_scene_union | common
        if set(endpoint_indices) & material_indices:
            raise GameValidationError("intervention endpoints must be disjoint from all game material")
        role_to_analysis: dict[SemanticRole, InterventionTarget] = {
            "P": "P",
            "Q": "Q",
            self.evaluation_y_role: "Y",
            r_role: "R",
        }
        role_offsets = {candidate.role: offset for offset, candidate in enumerate(candidates)}
        for record in interventions:
            before_mask = _candidate_mask(candidates, record.before_scene_index)
            after_mask = _candidate_mask(candidates, record.after_scene_index)
            delta = before_mask ^ after_mask
            if record.target == "distractor":
                if delta != 0:
                    raise GameValidationError("distractor edit changed a displayed candidate")
                continue
            role = next(role for role, target in role_to_analysis.items() if target == record.target)
            expected_delta = 1 << role_offsets[role]
            if delta != expected_delta:
                raise GameValidationError("matched edit did not isolate its registered candidate")
            before_value = bool(before_mask & expected_delta)
            if before_value is not bool(record.pair_index):
                raise GameValidationError("the two matched pairs must balance the before-target side")

    @property
    def candidate_by_id(self) -> dict[CandidateID, Candidate]:
        return {candidate.candidate_id: candidate for candidate in self.candidates}

    @property
    def candidate_by_role(self) -> dict[SemanticRole, Candidate]:
        return {candidate.role: candidate for candidate in self.candidates}

    @property
    def evaluation_r_role(self) -> Literal["M", "X"]:
        return "X" if self.evaluation_y_role == "M" else "M"

    @property
    def training_games(self) -> tuple[GameInstance, ...]:
        return tuple(
            GameInstance(
                instance_id=f"{self.family_id}-train-{candidate_id}",
                phase="train",
                family=self,
                material=self.training_material,
                official_candidate_id=candidate_id,
            )
            for candidate_id in self.training_official_ids
        )

    @property
    def evaluation_games(self) -> tuple[GameInstance, ...]:
        y_id = self.candidate_by_role[self.evaluation_y_role].candidate_id
        return tuple(
            GameInstance(
                instance_id=f"{self.family_id}-eval-{condition.condition_id}",
                phase="evaluation",
                family=self,
                material=GameMaterial(
                    condition.opening,
                    self.training_material.query_menu,
                    self.training_material.terminal,
                ),
                official_candidate_id=y_id,
            )
            for condition in self.evaluation_conditions
        )

    @property
    def scientific_scene_indices(self) -> frozenset[int]:
        values = set(self.training_material.query_menu) | set(self.training_material.terminal)
        if self.study_use in {"both", "train"}:
            values.update(item.scene_index for item in self.training_material.opening)
        if self.study_use in {"both", "evaluation"}:
            for condition in self.evaluation_conditions:
                values.update(item.scene_index for item in condition.opening)
            for record in self.matched_interventions:
                values.update((record.before_scene_index, record.after_scene_index))
        return frozenset(values)

    @property
    def all_serialized_scene_indices(self) -> tuple[int, ...]:
        values = [*self.training_material.query_menu, *self.training_material.terminal]
        values.extend(item.scene_index for item in self.training_material.opening)
        for condition in self.evaluation_conditions:
            values.extend(item.scene_index for item in condition.opening)
        for record in self.matched_interventions:
            values.extend((record.before_scene_index, record.after_scene_index))
        # Factorial evaluation conditions deliberately share their seven base
        # examples; return unique scene identities in stable encounter order.
        return tuple(dict.fromkeys(values))

    def as_obj(self) -> dict[str, Any]:
        return {
            "family_id": self.family_id,
            "candidates": [candidate.as_obj() for candidate in self.candidates],
            "training_material": self.training_material.as_obj(),
            "training_official_ids": list(self.training_official_ids),
            "evaluation_y_role": self.evaluation_y_role,
            "evaluation_conditions": [condition.as_obj() for condition in self.evaluation_conditions],
            "study_use": self.study_use,
            "matched_interventions": [record.as_obj() for record in self.matched_interventions],
        }


@dataclass(frozen=True, slots=True)
class GameInstance:
    """One hidden Official assignment over validated visible material."""

    instance_id: str
    phase: GamePhase
    family: GameFamily
    material: GameMaterial
    official_candidate_id: CandidateID

    def __post_init__(self) -> None:
        _require_ascii_id(self.instance_id, name="instance_id")
        if self.phase not in {"train", "evaluation"}:
            raise GameValidationError(f"unknown phase: {self.phase!r}")
        if type(self.family) is not GameFamily or type(self.material) is not GameMaterial:
            raise GameValidationError("instance requires a family and material")
        if self.official_candidate_id not in self.family.candidate_by_id:
            raise GameValidationError("Official is not one of the displayed candidates")
        if self.official_candidate_id not in self.initial_live_ids:
            raise GameValidationError("visible opening evidence eliminates the Official")

    @property
    def official(self) -> Candidate:
        return self.family.candidate_by_id[self.official_candidate_id]

    @property
    def initial_live_ids(self) -> tuple[CandidateID, ...]:
        return _consistent_candidate_ids(self.family.candidates, self.material.opening)

    @property
    def visible_bytes(self) -> bytes:
        """Canonical initial material, excluding every evaluator-only role field."""

        return dump_json(self.material.as_public_obj(self.family.candidates)).encode("ascii")

    def oracle_label(self, option_id: str) -> bool:
        return _label(self.official.rule, _option_scene_index(self, option_id))

    @property
    def terminal_labels(self) -> tuple[bool, ...]:
        return tuple(_label(self.official.rule, index) for index in self.material.terminal)

    def isolated_terminal_bytes(self, terminal_offset: int) -> bytes:
        """Reveal exactly one terminal scene for a sibling classification call."""

        if (
            isinstance(terminal_offset, bool)
            or not isinstance(terminal_offset, int)
            or not 0 <= terminal_offset < TERMINAL_SIZE
        ):
            raise GameValidationError(f"terminal_offset must lie in [0, {TERMINAL_SIZE})")
        return dump_json(
            {
                "schema_version": BANK_SCHEMA_VERSION,
                "terminal_item": terminal_offset + 1,
                "scene": scene_at(self.material.terminal[terminal_offset]).as_obj(),
            }
        ).encode("ascii")


@dataclass(frozen=True, slots=True)
class QueryInformation:
    option_id: str
    before_ids: tuple[CandidateID, ...]
    rejected_ids: tuple[CandidateID, ...]
    accepted_ids: tuple[CandidateID, ...]
    best_minority_count: int

    @property
    def before_count(self) -> int:
        return len(self.before_ids)

    @property
    def rejected_count(self) -> int:
        return len(self.rejected_ids)

    @property
    def accepted_count(self) -> int:
        return len(self.accepted_ids)

    @property
    def expected_information_bits(self) -> float:
        return entropy_reduction(self.before_count, self.accepted_count)

    @property
    def best_expected_information_bits(self) -> float:
        return entropy_reduction(self.before_count, self.best_minority_count)

    @property
    def regret_bits(self) -> float:
        return max(0.0, self.best_expected_information_bits - self.expected_information_bits)

    def as_obj(self) -> dict[str, Any]:
        # Integer partition counts are the exact, platform-independent metric.
        return {
            "option_id": self.option_id,
            "before_ids": list(self.before_ids),
            "rejected_ids": list(self.rejected_ids),
            "accepted_ids": list(self.accepted_ids),
            "best_minority_count": self.best_minority_count,
        }


@dataclass(frozen=True, slots=True)
class ReplayStep:
    information: QueryInformation
    oracle_accepted: bool
    after_ids: tuple[CandidateID, ...]

    @property
    def realized_information_bits(self) -> float:
        return math.log2(len(self.information.before_ids)) - math.log2(len(self.after_ids))


@dataclass(frozen=True, slots=True)
class QueryReplay:
    initial_ids: tuple[CandidateID, ...]
    steps: tuple[ReplayStep, ...]
    final_ids: tuple[CandidateID, ...]

    @property
    def identifies_official(self) -> bool:
        return len(self.final_ids) == 1


def _option_offset(option_id: str) -> int:
    if type(option_id) is not str or not option_id.startswith("Q"):
        raise GameValidationError(f"unknown query option: {option_id!r}")
    try:
        offset = int(option_id[1:]) - 1
    except ValueError as exc:
        raise GameValidationError(f"unknown query option: {option_id!r}") from exc
    if option_id != f"Q{offset + 1}" or not 0 <= offset < QUERY_MENU_SIZE:
        raise GameValidationError(f"unknown query option: {option_id!r}")
    return offset


def _option_scene_index(instance: GameInstance, option_id: str) -> int:
    return instance.material.query_menu[_option_offset(option_id)]


def query_information(
    instance: GameInstance,
    option_id: str,
    live_ids: tuple[CandidateID, ...] | None = None,
) -> QueryInformation:
    """Return exact version-space partition counts for one offered query."""

    if type(instance) is not GameInstance:
        raise TypeError("query_information requires a GameInstance")
    selected = instance.initial_live_ids if live_ids is None else tuple(live_ids)
    if not selected or len(set(selected)) != len(selected):
        raise GameValidationError("live candidate ids must be nonempty and unique")
    by_id = instance.family.candidate_by_id
    if any(candidate_id not in by_id for candidate_id in selected):
        raise GameValidationError("live set contains an unknown candidate")
    selected = tuple(
        candidate.candidate_id
        for candidate in instance.family.candidates
        if candidate.candidate_id in selected
    )
    scene_index = _option_scene_index(instance, option_id)
    accepted = tuple(
        candidate_id for candidate_id in selected if _label(by_id[candidate_id].rule, scene_index)
    )
    rejected = tuple(candidate_id for candidate_id in selected if candidate_id not in accepted)
    best_minority = max(
        min(
            sum(_label(by_id[candidate_id].rule, index) for candidate_id in selected),
            len(selected) - sum(_label(by_id[candidate_id].rule, index) for candidate_id in selected),
        )
        for index in instance.material.query_menu
    )
    return QueryInformation(option_id, selected, rejected, accepted, best_minority)


def replay_queries(instance: GameInstance, option_ids: Sequence[str]) -> QueryReplay:
    """Replay at most two distinct menu choices against the hidden Official."""

    choices = tuple(option_ids)
    if len(choices) > MAX_QUERY_TURNS:
        raise GameValidationError(f"at most {MAX_QUERY_TURNS} query turns are allowed")
    if len(set(choices)) != len(choices):
        raise GameValidationError("query choices cannot be repeated")
    live = instance.initial_live_ids
    steps: list[ReplayStep] = []
    for option_id in choices:
        information = query_information(instance, option_id, live)
        response = instance.oracle_label(option_id)
        live = information.accepted_ids if response else information.rejected_ids
        if not live or instance.official_candidate_id not in live:
            raise GameValidationError("replay eliminated the true Official")
        steps.append(ReplayStep(information, response, live))
    return QueryReplay(instance.initial_live_ids, tuple(steps), live)


def menu_minimax_depth(
    family: GameFamily,
    live_ids: tuple[CandidateID, ...] = CANDIDATE_IDS,
    *,
    max_depth: int = MAX_QUERY_TURNS,
) -> int | None:
    """Exact minimax identification depth using only the fixed eight choices."""

    if not 0 <= max_depth <= MAX_QUERY_TURNS:
        raise GameValidationError(f"max_depth must lie in [0, {MAX_QUERY_TURNS}]")
    by_id = family.candidate_by_id
    state = tuple(
        candidate.candidate_id for candidate in family.candidates if candidate.candidate_id in live_ids
    )
    if not state or len(state) != len(set(live_ids)) or any(item not in by_id for item in live_ids):
        raise GameValidationError("live candidate ids must be a nonempty subset of the family")

    @cache
    def solve(current: tuple[CandidateID, ...], remaining: tuple[int, ...], depth: int) -> bool:
        if len(current) <= 1:
            return True
        if depth == 0 or len(current) > 1 << depth:
            return False
        for option_offset in remaining:
            scene_index = family.training_material.query_menu[option_offset]
            yes = tuple(item for item in current if _label(by_id[item].rule, scene_index))
            no = tuple(item for item in current if item not in yes)
            if not yes or not no:
                continue
            rest = tuple(item for item in remaining if item != option_offset)
            if solve(yes, rest, depth - 1) and solve(no, rest, depth - 1):
                return True
        return False

    lower = (len(state) - 1).bit_length()
    options = tuple(range(QUERY_MENU_SIZE))
    for depth in range(lower, max_depth + 1):
        if solve(state, options, depth):
            return depth
    return None


def terminal_cell_inventory(family: GameFamily) -> dict[int, int]:
    counts = Counter(_candidate_mask(family.candidates, index) for index in family.training_material.terminal)
    return {mask: counts[mask] for mask in range(16)}


@dataclass(frozen=True, slots=True)
class FiniteChoiceBank:
    bank_id: str
    families: tuple[GameFamily, ...]

    def __post_init__(self) -> None:
        _require_ascii_id(self.bank_id, name="bank_id")
        families = tuple(self.families)
        object.__setattr__(self, "families", families)
        if len(families) != 2 or any(type(family) is not GameFamily for family in families):
            raise GameValidationError("the balanced small bank requires exactly two families")
        if len({family.family_id for family in families}) != len(families):
            raise GameValidationError("family ids must be unique")
        if Counter(cast(BinaryRule, family.candidate_by_role["M"].rule).op for family in families) != {
            "all": 1,
            "any": 1,
        }:
            raise GameValidationError("the bank must balance monotone all and any families")
        if Counter(family.evaluation_y_role for family in families) != {"M": 1, "X": 1}:
            raise GameValidationError("the bank must balance M and X as evaluation Y")
        rules = [candidate.rule.canonical_json for family in families for candidate in family.candidates]
        if len(set(rules)) != len(rules):
            raise GameValidationError("candidate rule identities must be disjoint across families")
        for first, second in pairwise(families):
            if first.scientific_scene_indices & second.scientific_scene_indices:
                raise GameValidationError("scene identities must be disjoint across families")

    @property
    def digest(self) -> str:
        return json_digest(self.as_obj(), domain=_BANK_DIGEST_DOMAIN)

    def as_obj(self) -> dict[str, Any]:
        return {
            "schema_version": BANK_SCHEMA_VERSION,
            "bank_id": self.bank_id,
            "families": [family.as_obj() for family in self.families],
        }


class _CellAllocator:
    def __init__(
        self,
        family_id: str,
        candidates: tuple[Candidate, ...],
        unavailable: set[int],
        allowed: frozenset[int] | None = None,
    ) -> None:
        pools: dict[int, list[int]] = {mask: [] for mask in range(16)}
        for index in range(SCENE_COUNT):
            if index not in unavailable and (allowed is None or index in allowed):
                pools[_candidate_mask(candidates, index)].append(index)

        def rank(scene_index: int) -> bytes:
            digest = hashlib.sha256()
            digest.update(_SCENE_ORDER_DOMAIN)
            digest.update(family_id.encode("ascii"))
            digest.update(scene_index.to_bytes(4, "big"))
            return digest.digest()

        self._pools = {mask: sorted(indices, key=rank) for mask, indices in pools.items()}
        self._offsets = {mask: 0 for mask in range(16)}
        self._unavailable = unavailable

    def take(self, mask: int, count: int = 1) -> tuple[int, ...]:
        pool = self._pools[mask]
        start = self._offsets[mask]
        stop = start + count
        if stop > len(pool):
            raise GameGenerationError(f"truth cell {mask:04b} has insufficient unused scenes")
        result = tuple(pool[start:stop])
        self._offsets[mask] = stop
        self._unavailable.update(result)
        return result


def _slot_literal(
    position: Literal["left", "center", "right"],
    attribute: Literal["color", "shape", "size"],
    value: str,
) -> RuleLiteral:
    return RuleLiteral(Atom("slot_attr", position=position, attribute=attribute, value=value))


def _family_rules(monotone_op: Literal["all", "any"]) -> dict[SemanticRole, Rule]:
    placard = RuleLiteral(Atom("placard_is", value="sun"), negated=monotone_op == "any")
    if monotone_op == "all":
        return {
            "P": placard,
            "Q": _slot_literal("left", "color", "red"),
            "M": BinaryRule(
                "all",
                (
                    RuleLiteral(Atom("occupied_count_is", value=3)),
                    _slot_literal("left", "size", "small"),
                ),
            ),
            "X": BinaryRule(
                "exactly_one",
                (
                    _slot_literal("center", "shape", "cube"),
                    _slot_literal("left", "color", "blue"),
                ),
            ),
        }
    return {
        "P": placard,
        "Q": _slot_literal("left", "color", "blue"),
        "M": BinaryRule(
            "any",
            (
                RuleLiteral(Atom("occupied_count_is", value=2)),
                RuleLiteral(
                    Atom(
                        "same",
                        position_1="center",
                        position_2="right",
                        attribute="shape",
                    )
                ),
            ),
        ),
        "X": BinaryRule(
            "exactly_one",
            (
                _slot_literal("right", "color", "red"),
                _slot_literal("right", "size", "small"),
            ),
        ),
    }


def _evaluation_conditions(
    allocator: _CellAllocator,
    candidates: tuple[Candidate, ...],
    *,
    family_offset: int,
) -> tuple[EvaluationCondition, ...]:
    """Build a paired 2x2 opening by swapping only the registered errors."""

    p_error_position = 4 + family_offset
    q_error_positions = (3 - family_offset, 6 + family_offset)
    base: list[LabeledScene] = []
    for position in range(OPENING_SIZE):
        accepted = bool(position % 2)
        mask = _evaluation_mask(candidates, accepted=accepted, p_error=False, q_error=False)
        base.append(LabeledScene(allocator.take(mask)[0], accepted))

    replacements: dict[int, LabeledScene] = {}
    accepted = bool(p_error_position % 2)
    replacements[p_error_position] = LabeledScene(
        allocator.take(_evaluation_mask(candidates, accepted=accepted, p_error=True, q_error=False))[0],
        accepted,
    )
    for position in q_error_positions:
        accepted = bool(position % 2)
        replacements[position] = LabeledScene(
            allocator.take(_evaluation_mask(candidates, accepted=accepted, p_error=False, q_error=True))[0],
            accepted,
        )

    conditions: list[EvaluationCondition] = []
    for p_evidence in EVIDENCE_LEVELS:
        for q_evidence in EVIDENCE_LEVELS:
            opening = list(base)
            if p_evidence == "noisy":
                opening[p_error_position] = replacements[p_error_position]
            if q_evidence == "noisy":
                for position in q_error_positions:
                    opening[position] = replacements[position]
            conditions.append(
                EvaluationCondition(
                    condition_id=f"p-{p_evidence}_q-{q_evidence}",
                    p_evidence=p_evidence,
                    q_evidence=q_evidence,
                    opening=tuple(opening),
                )
            )
    return tuple(conditions)


def _build_family(
    family_id: str,
    monotone_op: Literal["all", "any"],
    evaluation_y_role: Literal["M", "X"],
    unavailable: set[int],
    family_offset: int,
) -> GameFamily:
    rules = _family_rules(monotone_op)
    order = _opaque_role_order(family_id)
    candidates = tuple(
        Candidate(candidate_id, role, rules[role])
        for candidate_id, role in zip(CANDIDATE_IDS, order, strict=True)
    )
    allocator = _CellAllocator(family_id, candidates, unavailable)

    menu = tuple(allocator.take(mask)[0] for mask in _query_mask_order(0, "small", family_offset))
    terminal = tuple(allocator.take(mask)[0] for mask in range(16))
    train_opening = tuple(
        LabeledScene(allocator.take(mask)[0], bool(mask)) for mask in (0, 15) * (OPENING_SIZE // 2)
    )
    conditions = _evaluation_conditions(
        allocator,
        candidates,
        family_offset=family_offset,
    )
    return GameFamily(
        family_id=family_id,
        candidates=candidates,
        training_material=GameMaterial(train_opening, menu, terminal),
        training_official_ids=CANDIDATE_IDS,
        evaluation_y_role=evaluation_y_role,
        evaluation_conditions=conditions,
    )


@lru_cache(maxsize=1)
def build_small_bank() -> FiniteChoiceBank:
    """Generate and fully validate the deterministic two-family feasibility bank."""

    unavailable: set[int] = set()
    families = (
        _build_family("finite-all-v1", "all", "M", unavailable, 0),
        _build_family("finite-any-v1", "any", "X", unavailable, 1),
    )
    return FiniteChoiceBank("goalzendo-hidden-law-finite-choice-v1", families)


@dataclass(frozen=True, slots=True)
class SceneStagePartition:
    """Seed-bound, exhaustive train/evaluation/intervention scene split."""

    seed: int
    training: frozenset[int]
    evaluation: frozenset[int]
    intervention: frozenset[int]
    digest: str

    def __post_init__(self) -> None:
        if isinstance(self.seed, bool) or not isinstance(self.seed, int) or not 0 <= self.seed < 2**63:
            raise GameValidationError("bank seed must be an integer in [0, 2^63)")
        groups = (self.training, self.evaluation, self.intervention)
        if any(type(group) is not frozenset for group in groups):
            raise GameValidationError("scene-stage partitions must be frozensets")
        if any(first & second for first, second in combinations(groups, 2)):
            raise GameValidationError("scene-stage partitions must be disjoint")
        if set().union(*groups) != set(range(SCENE_COUNT)):
            raise GameValidationError("scene-stage partitions must exhaust the scene universe")
        if type(self.digest) is not str or len(self.digest) != 64:
            raise GameValidationError("scene-stage partition digest must be SHA-256")

    def as_manifest_obj(self) -> dict[str, Any]:
        return {
            "digest": self.digest,
            "training_count": len(self.training),
            "evaluation_count": len(self.evaluation),
            "intervention_count": len(self.intervention),
        }


def _seed_hash(seed: int, *parts: object, domain: bytes = _RULE_SEARCH_DOMAIN) -> bytes:
    digest = hashlib.sha256()
    digest.update(domain)
    digest.update(seed.to_bytes(8, "big"))
    for part in parts:
        encoded = str(part).encode("ascii")
        digest.update(len(encoded).to_bytes(4, "big"))
        digest.update(encoded)
    return digest.digest()


@lru_cache(maxsize=16)
def build_scene_stage_partition(seed: int) -> SceneStagePartition:
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**63:
        raise GameValidationError("bank seed must be an integer in [0, 2^63)")
    groups: dict[str, set[int]] = {
        "training": set(),
        "evaluation": set(),
        "intervention": set(),
    }
    attestation = hashlib.sha256()
    attestation.update(_SCENE_PARTITION_DOMAIN)
    attestation.update(seed.to_bytes(8, "big"))
    for index in range(SCENE_COUNT):
        bucket = int.from_bytes(_seed_hash(seed, index, domain=_SCENE_PARTITION_DOMAIN)[:8], "big") % 20
        stage = "training" if bucket < 13 else "evaluation" if bucket < 17 else "intervention"
        groups[stage].add(index)
        attestation.update(index.to_bytes(4, "big"))
        attestation.update(stage[0].encode("ascii"))
    return SceneStagePartition(
        seed,
        frozenset(groups["training"]),
        frozenset(groups["evaluation"]),
        frozenset(groups["intervention"]),
        attestation.hexdigest(),
    )


@cache
def _piece_rule_search_space() -> tuple[
    tuple[RuleLiteral, ...],
    tuple[RuleLiteral, ...],
    tuple[tuple[RuleLiteral, RuleLiteral], ...],
]:
    literals = tuple(RuleLiteral(atom) for atom in ATOMS if atom.op != "placard_is")
    eligible_q = tuple(literal for literal in literals if 0.25 <= truth_vector(literal).prevalence <= 0.75)
    return literals, eligible_q, tuple(combinations(literals, 2))


def _q_pools(seed: int) -> dict[Literal["train", "evaluation"], tuple[RuleLiteral, ...]]:
    _, eligible, _ = _piece_rule_search_space()
    ranked = tuple(
        sorted(
            eligible,
            key=lambda rule: _seed_hash(seed, "q-pool", rule.canonical_json),
        )
    )
    # Forty-one positive piece literals satisfy the registered prevalence
    # interval.  A 30/11 identity split gives both stages varied, disjoint Qs.
    return {"train": ranked[:30], "evaluation": ranked[30:]}


_PRODUCTION_BASE_ORDER: tuple[SemanticRole, ...] = ("Q", "X", "P", "M")


def _production_role_order(family_index: int) -> tuple[SemanticRole, ...]:
    offset = (family_index // 4) % 4
    return _PRODUCTION_BASE_ORDER[offset:] + _PRODUCTION_BASE_ORDER[:offset]


def _production_family_id(
    seed: int,
    stage: Literal["train", "evaluation"],
    family_index: int,
) -> str:
    desired = _production_role_order(family_index)
    stem = f"{stage}-s{seed}-{family_index:03d}"
    for nonce in range(512):
        candidate = f"{stem}-n{nonce:03d}"
        if _opaque_role_order(candidate) == desired:
            return candidate
    raise GameGenerationError("could not realize the registered opaque candidate order")


def _make_candidates(
    family_id: str,
    rules: Mapping[SemanticRole, Rule],
) -> tuple[Candidate, ...]:
    return tuple(
        Candidate(candidate_id, role, rules[role])
        for candidate_id, role in zip(CANDIDATE_IDS, _opaque_role_order(family_id), strict=True)
    )


def _cell_counts_in_bits(candidates: Sequence[Candidate], available_bits: int) -> tuple[int, ...]:
    vectors = tuple(truth_vector(candidate.rule).bits for candidate in candidates)
    universe = (1 << SCENE_COUNT) - 1
    result: list[int] = []
    for mask in range(16):
        cell = available_bits
        for offset, bits in enumerate(vectors):
            cell &= bits if mask & (1 << offset) else universe ^ bits
        result.append(cell.bit_count())
    return tuple(result)


def _evaluation_mask(
    candidates: Sequence[Candidate],
    *,
    accepted: bool,
    p_error: bool,
    q_error: bool,
) -> int:
    values: dict[SemanticRole, bool] = {
        "P": accepted ^ p_error,
        "Q": accepted ^ q_error,
        "M": accepted,
        "X": accepted,
    }
    return sum(int(values[candidate.role]) << offset for offset, candidate in enumerate(candidates))


def _required_main_cell_counts(
    candidates: tuple[Candidate, ...],
    use: Literal["train", "evaluation"],
    family_offset: int,
) -> Counter[int]:
    demand: Counter[int] = Counter(_QUERY_MASKS)
    demand.update(range(16))
    demand.update((0, 15) * (OPENING_SIZE // 2))
    if use == "evaluation":
        p_error_position = 4 + family_offset
        q_error_positions = (3 - family_offset, 6 + family_offset)
        for position in range(OPENING_SIZE):
            demand[
                _evaluation_mask(
                    candidates,
                    accepted=bool(position % 2),
                    p_error=False,
                    q_error=False,
                )
            ] += 1
        demand[
            _evaluation_mask(
                candidates,
                accepted=bool(p_error_position % 2),
                p_error=True,
                q_error=False,
            )
        ] += 1
        for position in q_error_positions:
            demand[
                _evaluation_mask(
                    candidates,
                    accepted=bool(position % 2),
                    p_error=False,
                    q_error=True,
                )
            ] += 1
    return demand


def _stage_bits(indices: frozenset[int]) -> int:
    return sum(1 << index for index in indices)


def _ranked_q_candidates(
    seed: int,
    stage: Literal["train", "evaluation"],
    family_index: int,
    q_usage: Counter[int],
) -> tuple[RuleLiteral, ...]:
    return tuple(
        sorted(
            _q_pools(seed)[stage],
            key=lambda rule: (
                q_usage[truth_vector(rule).bits],
                _seed_hash(seed, stage, family_index, "q", rule.canonical_json),
            ),
        )
    )


def _iter_rule_candidates(
    *,
    seed: int,
    stage: Literal["train", "evaluation"],
    family_index: int,
    family_id: str,
    monotone_op: Literal["all", "any"],
    p_negated: bool,
    available_main_bits: int,
    q_usage: Counter[int],
    used_composed_truths: set[int],
):
    literals, _, operand_pairs = _piece_rule_search_space()
    literal_truths = {truth_vector(literal).bits for literal in literals}
    placard = RuleLiteral(Atom("placard_is", value="sun"), negated=p_negated)
    for q in _ranked_q_candidates(seed, stage, family_index, q_usage):
        for attempt in range(512):
            m_pair = operand_pairs[
                int.from_bytes(
                    _seed_hash(seed, stage, family_index, q.canonical_json, attempt, "M")[:8],
                    "big",
                )
                % len(operand_pairs)
            ]
            x_pair = operand_pairs[
                int.from_bytes(
                    _seed_hash(seed, stage, family_index, q.canonical_json, attempt, "X")[:8],
                    "big",
                )
                % len(operand_pairs)
            ]
            m = BinaryRule(monotone_op, m_pair)
            x = BinaryRule("exactly_one", x_pair)
            m_bits, x_bits = truth_vector(m).bits, truth_vector(x).bits
            if (
                m_bits in literal_truths
                or x_bits in literal_truths
                or m_bits in used_composed_truths
                or x_bits in used_composed_truths
                or m_bits == x_bits
            ):
                continue
            rules: dict[SemanticRole, Rule] = {"P": placard, "Q": q, "M": m, "X": x}
            candidates = _make_candidates(family_id, rules)
            vectors = tuple(truth_vector(candidate.rule) for candidate in candidates)
            if len({vector.bits for vector in vectors}) != 4:
                continue
            if any(not 0.25 <= vector.prevalence <= 0.75 for vector in vectors):
                continue
            if min(_cell_counts_in_bits(candidates, (1 << SCENE_COUNT) - 1)) < 128:
                continue
            demand = _required_main_cell_counts(candidates, stage, family_index % 2)
            available = _cell_counts_in_bits(candidates, available_main_bits)
            if any(available[mask] < count + 4 for mask, count in demand.items()):
                continue
            yield rules, candidates, attempt


def _replace_piece(
    scene: Scene,
    position: Literal["left", "center", "right"],
    piece: Piece,
) -> Scene:
    slots = {name: scene.piece_at(cast(Any, name)) for name in POSITIONS}
    slots[position] = piece
    return Scene(slots["left"], slots["center"], slots["right"], scene.placard)


@lru_cache(maxsize=16)
def _primitive_intervention_pairs(seed: int) -> tuple[tuple[int, int, str], ...]:
    allowed = build_scene_stage_partition(seed).intervention
    pairs: list[tuple[int, int, str]] = []
    for before_index in sorted(allowed):
        before = scene_at(before_index)
        flipped_placard = Scene(
            before.left,
            before.center,
            before.right,
            "moon" if before.placard == "sun" else "sun",
        )
        after_index = scene_index(flipped_placard)
        if before_index < after_index and after_index in allowed:
            pairs.append((before_index, after_index, "placard"))
        for position in POSITIONS:
            piece = before.piece_at(position)
            if piece is None:
                continue
            for attribute, values in (
                ("color", COLORS),
                ("shape", SHAPES),
                ("size", SIZES),
            ):
                for value in values:
                    if getattr(piece, attribute) == value:
                        continue
                    replacement = Piece(
                        color=cast(Any, value) if attribute == "color" else piece.color,
                        shape=cast(Any, value) if attribute == "shape" else piece.shape,
                        size=cast(Any, value) if attribute == "size" else piece.size,
                    )
                    after_index = scene_index(_replace_piece(before, position, replacement))
                    if before_index < after_index and after_index in allowed:
                        pairs.append((before_index, after_index, f"{position}.{attribute}"))
    return tuple(pairs)


def _build_matched_interventions(
    *,
    seed: int,
    family_id: str,
    candidates: tuple[Candidate, ...],
    evaluation_y_role: Literal["M", "X"],
    unavailable: set[int],
) -> tuple[MatchedIntervention, ...]:
    r_role: Literal["M", "X"] = "X" if evaluation_y_role == "M" else "M"
    analysis_target = {
        "P": "P",
        "Q": "Q",
        evaluation_y_role: "Y",
        r_role: "R",
    }
    offset_target = {
        offset: cast(InterventionTarget, analysis_target[candidate.role])
        for offset, candidate in enumerate(candidates)
    }
    eligible: dict[InterventionTarget, list[tuple[int, int, str]]] = {
        target: [] for target in INTERVENTION_TARGETS
    }
    for first, second, field in _primitive_intervention_pairs(seed):
        if first in unavailable or second in unavailable:
            continue
        delta = _candidate_mask(candidates, first) ^ _candidate_mask(candidates, second)
        if delta == 0:
            eligible["distractor"].append((first, second, field))
        elif delta & (delta - 1) == 0:
            eligible[offset_target[delta.bit_length() - 1]].append((first, second, field))

    records: list[MatchedIntervention] = []
    selected: set[int] = set()
    role_offset = {target: offset for offset, target in offset_target.items()}
    for target in INTERVENTION_TARGETS:
        ranked = sorted(
            eligible[target],
            key=lambda pair: _seed_hash(
                seed,
                family_id,
                target,
                pair[0],
                pair[1],
                domain=_INTERVENTION_ORDER_DOMAIN,
            ),
        )
        for pair_index in (0, 1):
            try:
                first, second, field = next(
                    pair for pair in ranked if pair[0] not in selected and pair[1] not in selected
                )
            except StopIteration as exc:
                raise GameGenerationError(
                    f"family {family_id} has no disjoint {target} intervention reserve"
                ) from exc
            if target != "distractor":
                bit = 1 << role_offset[target]
                desired_before = bool(pair_index)
                if bool(_candidate_mask(candidates, first) & bit) is not desired_before:
                    first, second = second, first
            selected.update((first, second))
            records.append(MatchedIntervention(target, pair_index, first, second, field))
    unavailable.update(selected)
    return tuple(records)


def _build_production_family(
    *,
    seed: int,
    family_id: str,
    family_index: int,
    use: Literal["train", "evaluation"],
    candidates: tuple[Candidate, ...],
    evaluation_y_role: Literal["M", "X"],
    main_allowed: frozenset[int],
    unavailable: set[int],
) -> GameFamily:
    allocator = _CellAllocator(family_id, candidates, unavailable, main_allowed)
    menu = tuple(allocator.take(mask)[0] for mask in _query_mask_order(seed, use, family_index))
    terminal = tuple(allocator.take(mask)[0] for mask in range(16))
    training_opening = tuple(
        LabeledScene(allocator.take(mask)[0], bool(mask)) for mask in (0, 15) * (OPENING_SIZE // 2)
    )
    conditions: tuple[EvaluationCondition, ...] = ()
    interventions: tuple[MatchedIntervention, ...] = ()
    if use == "evaluation":
        conditions = _evaluation_conditions(
            allocator,
            candidates,
            family_offset=family_index % 2,
        )
        interventions = _build_matched_interventions(
            seed=seed,
            family_id=family_id,
            candidates=candidates,
            evaluation_y_role=evaluation_y_role,
            unavailable=unavailable,
        )
    return GameFamily(
        family_id=family_id,
        candidates=candidates,
        training_material=GameMaterial(training_opening, menu, terminal),
        training_official_ids=CANDIDATE_IDS if use == "train" else (),
        evaluation_y_role=evaluation_y_role,
        evaluation_conditions=conditions,
        study_use=use,
        matched_interventions=interventions,
    )


def _piece_truth_identities(families: Sequence[GameFamily]) -> set[int]:
    return {
        truth_vector(family.candidate_by_role[role].rule).bits
        for family in families
        for role in cast(tuple[SemanticRole, ...], ("Q", "M", "X"))
    }


@dataclass(frozen=True, slots=True)
class ProductionBank:
    """The paired, seed-only source of all training blocks and held-out quartets."""

    seed: int
    scene_partition_digest: str
    training_families: tuple[GameFamily, ...]
    evaluation_families: tuple[GameFamily, ...]

    def __post_init__(self) -> None:
        partition = build_scene_stage_partition(self.seed)
        if self.scene_partition_digest != partition.digest:
            raise GameValidationError("production bank scene-partition digest mismatch")
        training = tuple(self.training_families)
        evaluation = tuple(self.evaluation_families)
        object.__setattr__(self, "training_families", training)
        object.__setattr__(self, "evaluation_families", evaluation)
        if len(training) != TRAINING_FAMILY_COUNT or any(
            type(family) is not GameFamily or family.study_use != "train" for family in training
        ):
            raise GameValidationError(f"production bank requires {TRAINING_FAMILY_COUNT} train families")
        if len(evaluation) != EVALUATION_FAMILY_COUNT or any(
            type(family) is not GameFamily or family.study_use != "evaluation" for family in evaluation
        ):
            raise GameValidationError(
                f"production bank requires {EVALUATION_FAMILY_COUNT} evaluation families"
            )
        families = (*training, *evaluation)
        if len({family.family_id for family in families}) != len(families):
            raise GameValidationError("production family ids must be unique")
        if _piece_truth_identities(training) & _piece_truth_identities(evaluation):
            raise GameValidationError("train and evaluation piece-rule identities must be disjoint")
        composed = [
            truth_vector(family.candidate_by_role[role].rule).bits
            for family in families
            for role in cast(tuple[SemanticRole, ...], ("M", "X"))
        ]
        if len(set(composed)) != len(composed):
            raise GameValidationError("all production M and X identities must be unique")

        for stage_families, expected_count in (
            (training, TRAINING_FAMILY_COUNT // 2),
            (evaluation, EVALUATION_FAMILY_COUNT // 2),
        ):
            if Counter(
                cast(BinaryRule, family.candidate_by_role["M"].rule).op for family in stage_families
            ) != {"all": expected_count, "any": expected_count}:
                raise GameValidationError("each stage must balance monotone all and any")
            if Counter(family.evaluation_y_role for family in stage_families) != {
                "M": expected_count,
                "X": expected_count,
            }:
                raise GameValidationError("each stage must balance M and X designations")
            if Counter(
                cast(RuleLiteral, family.candidate_by_role["P"].rule).negated for family in stage_families
            ) != {False: expected_count, True: expected_count}:
                raise GameValidationError("each stage must balance both placard literals")
            position_counts = Counter(
                (candidate.role, candidate.candidate_id)
                for family in stage_families
                for candidate in family.candidates
            )
            expected_position_count = len(stage_families) // 4
            if set(position_counts.values()) != {expected_position_count}:
                raise GameValidationError("each stage must counterbalance every role over A/B/C/D")
            query_position_counts = Counter(
                (position, _candidate_mask(family.candidates, scene_index))
                for family in stage_families
                for position, scene_index in enumerate(family.training_material.query_menu)
            )
            expected_query_position_count = len(stage_families) // QUERY_MENU_SIZE
            if query_position_counts != Counter(
                {
                    (position, mask): expected_query_position_count
                    for position in range(QUERY_MENU_SIZE)
                    for mask in _QUERY_MASKS
                }
            ):
                raise GameValidationError("each stage must counterbalance every query partition over Q1--Q8")

        serialized_indices = [index for family in families for index in family.all_serialized_scene_indices]
        if len(set(serialized_indices)) != len(serialized_indices):
            raise GameValidationError("production bank must not reuse a scene anywhere")
        for family in training:
            if not set(family.all_serialized_scene_indices) <= partition.training:
                raise GameValidationError("training family escaped the seed-bound training partition")
        for family in evaluation:
            intervention_indices = {
                index
                for record in family.matched_interventions
                for index in (record.before_scene_index, record.after_scene_index)
            }
            main_indices = set(family.all_serialized_scene_indices) - intervention_indices
            if not main_indices <= partition.evaluation:
                raise GameValidationError("evaluation material escaped its seed-bound partition")
            if not intervention_indices <= partition.intervention:
                raise GameValidationError("matched edits escaped the intervention partition")

    @property
    def training_games(self) -> tuple[GameInstance, ...]:
        return tuple(game for family in self.training_families for game in family.training_games)

    @property
    def evaluation_games(self) -> tuple[GameInstance, ...]:
        return tuple(game for family in self.evaluation_families for game in family.evaluation_games)

    @property
    def interim_evaluation_families(self) -> tuple[GameFamily, ...]:
        # A fixed quarter-panel spanning all role orders while balancing P
        # polarity, M all/any, Y=M/X, and the two evaluation renderers.
        return tuple(self.evaluation_families[index] for index in (0, 5, 10, 15))

    @property
    def digest(self) -> str:
        return json_digest(self.as_obj(), domain=_PRODUCTION_BANK_DIGEST_DOMAIN)

    @property
    def pairing_key(self) -> str:
        return json_digest(
            {"seed": self.seed, "bank_digest": self.digest},
            domain="goalzendo-hidden-law-seed-pairing-v1",
        )

    @property
    def manifest(self) -> dict[str, Any]:
        partition = build_scene_stage_partition(self.seed)
        training_q = [
            truth_vector(family.candidate_by_role["Q"].rule).bits for family in self.training_families
        ]
        evaluation_q = [
            truth_vector(family.candidate_by_role["Q"].rule).bits for family in self.evaluation_families
        ]
        training_scenes = sum(len(family.all_serialized_scene_indices) for family in self.training_families)
        evaluation_scenes = sum(
            len(family.all_serialized_scene_indices) for family in self.evaluation_families
        )
        return {
            "schema_version": PRODUCTION_BANK_SCHEMA_VERSION,
            "seed": self.seed,
            "scene_partition": partition.as_manifest_obj(),
            "family_counts": {
                "training": len(self.training_families),
                "evaluation": len(self.evaluation_families),
                "training_official_games": len(self.training_games),
                "evaluation_quartet_games": len(self.evaluation_games),
                "interim_evaluation_quartets": len(self.interim_evaluation_families),
            },
            "rule_identity_counts": {
                "training_Q_unique": len(set(training_q)),
                "evaluation_Q_unique": len(set(evaluation_q)),
                "training_M_unique": len(
                    {
                        truth_vector(family.candidate_by_role["M"].rule).bits
                        for family in self.training_families
                    }
                ),
                "training_X_unique": len(
                    {
                        truth_vector(family.candidate_by_role["X"].rule).bits
                        for family in self.training_families
                    }
                ),
                "evaluation_M_unique": len(
                    {
                        truth_vector(family.candidate_by_role["M"].rule).bits
                        for family in self.evaluation_families
                    }
                ),
                "evaluation_X_unique": len(
                    {
                        truth_vector(family.candidate_by_role["X"].rule).bits
                        for family in self.evaluation_families
                    }
                ),
                "cross_stage_piece_overlap": len(
                    _piece_truth_identities(self.training_families)
                    & _piece_truth_identities(self.evaluation_families)
                ),
                "placard_unique": 2,
                "placard_occurrences": len(self.training_families) + len(self.evaluation_families),
            },
            "scene_counts": {
                "training_serialized": training_scenes,
                "evaluation_serialized": evaluation_scenes,
                "global_unique": training_scenes + evaluation_scenes,
                "global_reuse": 0,
            },
            "balances": {
                stage: {
                    "monotone_ops": dict(
                        sorted(
                            Counter(
                                cast(BinaryRule, family.candidate_by_role["M"].rule).op for family in families
                            ).items()
                        )
                    ),
                    "evaluation_Y": dict(
                        sorted(Counter(family.evaluation_y_role for family in families).items())
                    ),
                    "placard_negated": dict(
                        sorted(
                            Counter(
                                str(cast(RuleLiteral, family.candidate_by_role["P"].rule).negated)
                                for family in families
                            ).items()
                        )
                    ),
                }
                for stage, families in (
                    ("training", self.training_families),
                    ("evaluation", self.evaluation_families),
                )
            },
        }

    def as_obj(self) -> dict[str, Any]:
        return {
            "schema_version": PRODUCTION_BANK_SCHEMA_VERSION,
            "seed": self.seed,
            "scene_partition_digest": self.scene_partition_digest,
            "training_families": [family.as_obj() for family in self.training_families],
            "evaluation_families": [family.as_obj() for family in self.evaluation_families],
            "manifest": self.manifest,
        }


@lru_cache(maxsize=8)
def build_production_bank(seed: int) -> ProductionBank:
    """Build the exact bank shared by every model/algorithm run at one seed."""

    partition = build_scene_stage_partition(seed)
    unavailable: set[int] = set()
    used_composed_truths: set[int] = set()
    q_usage: dict[Literal["train", "evaluation"], Counter[int]] = {
        "train": Counter(),
        "evaluation": Counter(),
    }
    built: dict[Literal["train", "evaluation"], list[GameFamily]] = {
        "train": [],
        "evaluation": [],
    }
    stages: tuple[
        tuple[Literal["train", "evaluation"], int, frozenset[int]],
        ...,
    ] = (
        ("train", TRAINING_FAMILY_COUNT, partition.training),
        ("evaluation", EVALUATION_FAMILY_COUNT, partition.evaluation),
    )
    for stage, count, main_allowed in stages:
        for family_index in range(count):
            family_id = _production_family_id(seed, stage, family_index)
            monotone_op: Literal["all", "any"] = "all" if family_index % 2 == 0 else "any"
            evaluation_y_role: Literal["M", "X"] = "M" if (family_index // 2) % 2 == 0 else "X"
            p_negated = bool((family_index // 4) % 2)
            allowed_bits = _stage_bits(main_allowed)
            unavailable_bits = sum(1 << index for index in unavailable)
            available_main_bits = allowed_bits & ~unavailable_bits
            retained: GameFamily | None = None
            retained_rules: Mapping[SemanticRole, Rule] | None = None
            for rules, candidates, _attempt in _iter_rule_candidates(
                seed=seed,
                stage=stage,
                family_index=family_index,
                family_id=family_id,
                monotone_op=monotone_op,
                p_negated=p_negated,
                available_main_bits=available_main_bits,
                q_usage=q_usage[stage],
                used_composed_truths=used_composed_truths,
            ):
                trial_unavailable = set(unavailable)
                try:
                    candidate_family = _build_production_family(
                        seed=seed,
                        family_id=family_id,
                        family_index=family_index,
                        use=stage,
                        candidates=candidates,
                        evaluation_y_role=evaluation_y_role,
                        main_allowed=main_allowed,
                        unavailable=trial_unavailable,
                    )
                except GameGenerationError:
                    continue
                retained = candidate_family
                retained_rules = rules
                unavailable.update(trial_unavailable)
                break
            if retained is None or retained_rules is None:
                raise GameGenerationError(
                    f"deterministic rule search exhausted for {stage} family {family_index}"
                )
            built[stage].append(retained)
            q_usage[stage][truth_vector(retained_rules["Q"]).bits] += 1
            used_composed_truths.update(
                truth_vector(retained_rules[role]).bits for role in cast(tuple[SemanticRole, ...], ("M", "X"))
            )
    return ProductionBank(
        seed,
        partition.digest,
        tuple(built["train"]),
        tuple(built["evaluation"]),
    )


def serialize_production_bank(bank: ProductionBank) -> str:
    if type(bank) is not ProductionBank:
        raise TypeError("serialize_production_bank requires a ProductionBank")
    return dump_json(bank.as_obj())


def parse_production_bank(text: str, *, expected_digest: str | None = None) -> ProductionBank:
    try:
        value = load_json(text)
    except ValueError as exc:
        raise GameValidationError(str(exc)) from exc
    item = _require_exact_keys(
        value,
        {
            "schema_version",
            "seed",
            "scene_partition_digest",
            "training_families",
            "evaluation_families",
            "manifest",
        },
        name="production bank",
    )
    if item["schema_version"] != PRODUCTION_BANK_SCHEMA_VERSION:
        raise GameValidationError("unsupported production-bank schema version")
    if type(item["training_families"]) is not list or type(item["evaluation_families"]) is not list:
        raise GameValidationError("production family collections must be JSON arrays")
    bank = ProductionBank(
        item["seed"],
        item["scene_partition_digest"],
        tuple(_family_from_obj(entry) for entry in item["training_families"]),
        tuple(_family_from_obj(entry) for entry in item["evaluation_families"]),
    )
    if bank.manifest != item["manifest"]:
        raise GameValidationError("production-bank manifest does not match derived contents")
    if serialize_production_bank(bank) != text:
        raise GameValidationError("production bank is valid but not canonical")
    if expected_digest is not None and bank.digest != expected_digest:
        raise GameValidationError("production-bank digest differs from the expected identity")
    return bank


def serialize_bank(bank: FiniteChoiceBank) -> str:
    if type(bank) is not FiniteChoiceBank:
        raise TypeError("serialize_bank requires a FiniteChoiceBank")
    return dump_json(bank.as_obj())


def _candidate_from_obj(value: object) -> Candidate:
    item = _require_exact_keys(value, {"candidate_id", "role", "rule"}, name="candidate")
    return Candidate(
        cast(CandidateID, item["candidate_id"]),
        cast(SemanticRole, item["role"]),
        rule_from_obj(item["rule"]),
    )


def _labeled_scene_from_obj(value: object) -> LabeledScene:
    item = _require_exact_keys(value, {"scene_index", "accepted"}, name="labeled scene")
    return LabeledScene(item["scene_index"], item["accepted"])


def _material_from_obj(value: object) -> GameMaterial:
    item = _require_exact_keys(value, {"opening", "query_menu", "terminal"}, name="material")
    if any(type(item[name]) is not list for name in ("opening", "query_menu", "terminal")):
        raise GameValidationError("material arrays must be JSON arrays")
    return GameMaterial(
        tuple(_labeled_scene_from_obj(entry) for entry in item["opening"]),
        tuple(item["query_menu"]),
        tuple(item["terminal"]),
    )


def _condition_from_obj(value: object) -> EvaluationCondition:
    item = _require_exact_keys(
        value,
        {"condition_id", "p_evidence", "q_evidence", "opening"},
        name="evaluation condition",
    )
    if type(item["opening"]) is not list:
        raise GameValidationError("evaluation opening must be a JSON array")
    return EvaluationCondition(
        item["condition_id"],
        cast(EvidenceLevel, item["p_evidence"]),
        cast(EvidenceLevel, item["q_evidence"]),
        tuple(_labeled_scene_from_obj(entry) for entry in item["opening"]),
    )


def _intervention_from_obj(value: object) -> MatchedIntervention:
    item = _require_exact_keys(
        value,
        {
            "target",
            "pair_index",
            "before_scene_index",
            "after_scene_index",
            "changed_field",
        },
        name="matched intervention",
    )
    return MatchedIntervention(
        cast(InterventionTarget, item["target"]),
        item["pair_index"],
        item["before_scene_index"],
        item["after_scene_index"],
        item["changed_field"],
    )


def _family_from_obj(value: object) -> GameFamily:
    item = _require_exact_keys(
        value,
        {
            "family_id",
            "candidates",
            "training_material",
            "training_official_ids",
            "evaluation_y_role",
            "evaluation_conditions",
            "study_use",
            "matched_interventions",
        },
        name="family",
    )
    if any(
        type(item[name]) is not list
        for name in (
            "candidates",
            "training_official_ids",
            "evaluation_conditions",
            "matched_interventions",
        )
    ):
        raise GameValidationError("family collections must be JSON arrays")
    return GameFamily(
        family_id=item["family_id"],
        candidates=tuple(_candidate_from_obj(entry) for entry in item["candidates"]),
        training_material=_material_from_obj(item["training_material"]),
        training_official_ids=tuple(item["training_official_ids"]),
        evaluation_y_role=cast(Literal["M", "X"], item["evaluation_y_role"]),
        evaluation_conditions=tuple(_condition_from_obj(entry) for entry in item["evaluation_conditions"]),
        study_use=cast(FamilyUse, item["study_use"]),
        matched_interventions=tuple(_intervention_from_obj(entry) for entry in item["matched_interventions"]),
    )


def parse_bank(text: str, *, expected_digest: str | None = None) -> FiniteChoiceBank:
    """Parse canonical bank JSON and optionally bind it to an expected digest."""

    try:
        value = load_json(text)
    except ValueError as exc:
        raise GameValidationError(str(exc)) from exc
    item = _require_exact_keys(value, {"schema_version", "bank_id", "families"}, name="bank")
    if item["schema_version"] != BANK_SCHEMA_VERSION:
        raise GameValidationError("unsupported bank schema version")
    if type(item["families"]) is not list:
        raise GameValidationError("families must be a JSON array")
    bank = FiniteChoiceBank(item["bank_id"], tuple(_family_from_obj(entry) for entry in item["families"]))
    if serialize_bank(bank) != text:
        raise GameValidationError("bank JSON is valid but not in canonical serialized form")
    if expected_digest is not None and bank.digest != expected_digest:
        raise GameValidationError("bank digest differs from the expected identity")
    return bank


BankLike: TypeAlias = FiniteChoiceBank
