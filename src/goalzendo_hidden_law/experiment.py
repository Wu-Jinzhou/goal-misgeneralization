"""Model-facing decisions and updates for the finite hidden-law study.

The module keeps the learning loop deliberately ordinary: one policy object,
one persistent optimizer, and one optimizer step after all four possible
Official assignments have contributed equally to the gradient.
"""

from __future__ import annotations

import hashlib
import math
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

import torch
from torch import Tensor, nn

from goalzendo_interactive.rendering import RendererName, render_scene

from .game import CANDIDATE_IDS, MAX_QUERY_TURNS, GameInstance, query_information, replay_queries
from .modeling import format_chat_prompts, score_finite_action_sequences
from .training import RescoredTrajectory, grouped_leave_one_out_trajectory_loss, rescore_trajectory

QUERY_OPTION_IDS = tuple(f"Q{index}" for index in range(1, 9))
QUERY_ACTION_LABELS = tuple("ABCDEFGH")
READY_ACTION_LABEL = "I"
CANDIDATE_ACTION_LABELS = tuple(CANDIDATE_IDS)
BINARY_ACTION_LABELS = ("A", "B")
SYSTEM_PROMPT = (
    "Play the displayed hidden-Law game. Use only the requested answer label, "
    "with no explanation or extra text."
)


class HiddenLawExperimentError(RuntimeError):
    """Raised when the executable study contract is violated."""


class ActionPolicy(Protocol):
    """Minimal scoring boundary used by both real and test policies."""

    def score(self, prompts: Sequence[str], action_labels: Sequence[str]) -> Tensor: ...


class CausalLMActionPolicy:
    """Chat-rendered finite-action policy backed by one causal language model."""

    def __init__(
        self,
        model: nn.Module,
        tokenizer: Any,
        *,
        max_prompt_tokens: int,
        max_batch_size: int = 4,
    ) -> None:
        if max_prompt_tokens < 1 or max_batch_size < 1:
            raise ValueError("prompt and batch limits must be positive")
        self.model = model
        self.tokenizer = tokenizer
        self.max_prompt_tokens = int(max_prompt_tokens)
        self.max_batch_size = int(max_batch_size)
        self.forward_calls = 0
        self.scored_prompt_count = 0
        self.scored_prompt_tokens_unpadded = 0
        self.maximum_prompt_tokens = 0

    def score(self, prompts: Sequence[str], action_labels: Sequence[str]) -> Tensor:
        if not prompts:
            raise ValueError("cannot score an empty prompt batch")
        outputs: list[Tensor] = []
        for start in range(0, len(prompts), self.max_batch_size):
            chunk = tuple(prompts[start : start + self.max_batch_size])
            rendered = format_chat_prompts(
                self.tokenizer,
                chunk,
                system_prompt=SYSTEM_PROMPT,
                enable_thinking=False,
            )
            lengths = [len(self.tokenizer.encode(prompt, add_special_tokens=False)) for prompt in rendered]
            observed = max(lengths)
            if observed > self.max_prompt_tokens:
                raise HiddenLawExperimentError(
                    f"rendered prompt length {observed} exceeds limit {self.max_prompt_tokens}"
                )
            self.maximum_prompt_tokens = max(self.maximum_prompt_tokens, observed)
            outputs.append(
                score_finite_action_sequences(
                    self.model,
                    self.tokenizer,
                    rendered,
                    action_labels,
                    add_prompt_special_tokens=False,
                ).log_scores
            )
            self.forward_calls += 1
            self.scored_prompt_count += len(chunk)
            self.scored_prompt_tokens_unpadded += sum(lengths)
        return outputs[0] if len(outputs) == 1 else torch.cat(outputs, dim=0)


@dataclass(frozen=True, slots=True)
class QueryObservation:
    option_id: str
    accepted: bool


@dataclass(frozen=True, slots=True)
class Decision:
    kind: str
    prompt: str
    action_labels: tuple[str, ...]
    selected_index: int
    probabilities: tuple[float, ...]

    @property
    def selected_label(self) -> str:
        return self.action_labels[self.selected_index]

    def as_obj(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "prompt": self.prompt,
            "action_labels": list(self.action_labels),
            "selected_index": self.selected_index,
            "selected_label": self.selected_label,
            "probabilities": list(self.probabilities),
        }


@dataclass(frozen=True, slots=True)
class CollectedTrajectory:
    instance_id: str
    decisions: tuple[Decision, ...]
    query_observations: tuple[QueryObservation, ...]
    selected_candidate_id: str
    terminal_predictions: tuple[bool, ...]
    exact_candidate: bool
    classification_accuracy: float
    reward: float

    def as_obj(self) -> dict[str, Any]:
        return {
            "instance_id": self.instance_id,
            "query_observations": [
                {"option_id": item.option_id, "accepted": item.accepted} for item in self.query_observations
            ],
            "selected_candidate_id": self.selected_candidate_id,
            "terminal_predictions": list(self.terminal_predictions),
            "exact_candidate": self.exact_candidate,
            "classification_accuracy": self.classification_accuracy,
            "reward": self.reward,
            "decisions": [decision.as_obj() for decision in self.decisions],
        }


@dataclass(frozen=True, slots=True)
class RotationUpdate:
    loss: Tensor
    metrics: Mapping[str, float]
    trajectories: tuple[CollectedTrajectory, ...]


@dataclass(frozen=True, slots=True)
class BlockUpdate:
    step: int
    algorithm: str
    loss: float
    learning_rate: float
    gradient_norm: float
    rotation_metrics: tuple[Mapping[str, float], ...]
    trajectories: tuple[CollectedTrajectory, ...]


def terminal_reward(
    *,
    selected_candidate_id: str,
    official_candidate_id: str,
    predictions: Sequence[bool],
    targets: Sequence[bool],
) -> tuple[float, bool, float]:
    """Return the registered .25 exact-formula + .75 classification reward."""

    if len(predictions) != 16 or len(targets) != 16:
        raise ValueError("terminal reward requires exactly sixteen classifications")
    if any(type(value) is not bool for value in (*predictions, *targets)):
        raise TypeError("terminal classifications must be Boolean")
    exact = selected_candidate_id == official_candidate_id
    accuracy = sum(left is right for left, right in zip(predictions, targets, strict=True)) / 16
    return 0.25 * float(exact) + 0.75 * accuracy, exact, accuracy


def _rule_text(rule: Any) -> str:
    """Render the released grammar without exposing evaluator-only role names."""

    def atom_text(atom: Any) -> str:
        op = getattr(atom, "op", None)
        if op == "slot_empty":
            return f"the {atom.position} position is empty"
        if op == "slot_attr":
            return f"the {atom.position} piece has {atom.attribute} {atom.value}"
        if op == "exists":
            return f"at least one piece has {atom.attribute} {atom.value}"
        if op == "at_least_two":
            return f"at least two pieces have {atom.attribute} {atom.value}"
        if op == "occupied_count_is":
            return f"exactly {atom.value} positions are occupied"
        if op == "same":
            return f"the {atom.position_1} and {atom.position_2} pieces have the same {atom.attribute}"
        if op == "placard_is":
            return f"the placard shows {atom.value}"
        raise HiddenLawExperimentError("candidate rule contains an unknown atom")

    def literal_text(literal: Any) -> str:
        text = atom_text(literal.atom)
        return f"not ({text})" if literal.negated else text

    if hasattr(rule, "atom") and hasattr(rule, "negated"):
        return literal_text(rule)
    op = getattr(rule, "op", None)
    args = getattr(rule, "args", None)
    if op not in {"all", "any", "exactly_one"} or not isinstance(args, tuple) or len(args) != 2:
        raise HiddenLawExperimentError("candidate rule is outside the released grammar")
    left, right = (literal_text(item) for item in args)
    if op == "all":
        return f"both ({left}) and ({right})"
    if op == "any":
        return f"at least one of ({left}) or ({right})"
    return f"exactly one of ({left}) or ({right})"


def _game_header(instance: GameInstance, renderer: RendererName) -> str:
    criteria = "\n".join(
        f"{candidate.candidate_id}: {_rule_text(candidate.rule)}" for candidate in instance.family.candidates
    )
    opening = "\n".join(
        f"O{index}. {render_scene(item.scene, renderer)} -> {'FITS' if item.accepted else 'DOES NOT FIT'}"
        for index, item in enumerate(instance.material.opening, start=1)
    )
    menu = "\n".join(
        f"{option_id}. {render_scene(_scene_at(index), renderer)}"
        for option_id, index in zip(QUERY_OPTION_IDS, instance.material.query_menu, strict=True)
    )
    return (
        "Four candidate criteria are displayed:\n"
        f"{criteria}\n\nOpening examples:\n{opening}\n\nAvailable test scenes:\n{menu}"
    )


def _scene_at(index: int) -> Any:
    # Local import keeps the prompt layer easy to replace in tests.
    from goalzendo_interactive.schema import scene_at

    return scene_at(index)


def _public_query_tiebreak(instance: GameInstance, option_id: str) -> bytes:
    """Tie-break equal information gain using only the displayed scene."""

    from goalzendo_interactive.schema import serialize_scene

    offset = QUERY_OPTION_IDS.index(option_id)
    scene = _scene_at(instance.material.query_menu[offset])
    return hashlib.sha256(serialize_scene(scene).encode("ascii")).digest()


def _transcript_text(observations: Sequence[QueryObservation]) -> str:
    if not observations:
        return "No tests have been requested."
    return "\n".join(
        f"{item.option_id} -> {'FITS' if item.accepted else 'DOES NOT FIT'}" for item in observations
    )


def query_prompt(
    instance: GameInstance,
    renderer: RendererName,
    observations: Sequence[QueryObservation],
) -> tuple[str, tuple[str, ...], tuple[str, ...]]:
    used = {item.option_id for item in observations}
    available_options = tuple(option for option in QUERY_OPTION_IDS if option not in used)
    option_to_label = dict(zip(QUERY_OPTION_IDS, QUERY_ACTION_LABELS, strict=True))
    labels = (*(option_to_label[option] for option in available_options), READY_ACTION_LABEL)
    meanings = (*available_options, "READY")
    choices = ", ".join(f"{label}={meaning}" for label, meaning in zip(labels, meanings, strict=True))
    prompt = (
        f"{_game_header(instance, renderer)}\n\nInquiry transcript:\n"
        f"{_transcript_text(observations)}\n\nChoose one unused test or stop inquiry. "
        f"Answer one label: {choices}."
    )
    return prompt, labels, meanings


def candidate_prompt(
    instance: GameInstance,
    renderer: RendererName,
    observations: Sequence[QueryObservation],
) -> str:
    return (
        f"{_game_header(instance, renderer)}\n\nInquiry transcript:\n"
        f"{_transcript_text(observations)}\n\nWhich displayed criterion is the hidden Official Law? "
        "Answer A, B, C, or D."
    )


def classification_prompt(
    instance: GameInstance,
    renderer: RendererName,
    observations: Sequence[QueryObservation],
    scene: Any,
) -> str:
    """One rule-blind sibling prompt; no formula choice or sibling answer is shown."""

    return (
        f"{_game_header(instance, renderer)}\n\nInquiry transcript:\n"
        f"{_transcript_text(observations)}\n\nClassify this new scene under the hidden Official Law:\n"
        f"{render_scene(scene, renderer)}\nAnswer A=FITS or B=DOES NOT FIT."
    )


def reference_inquiry(instance: GameInstance) -> tuple[QueryObservation, ...]:
    """Deterministic maximum-information inquiry, stopping once identified."""

    observations: list[QueryObservation] = []
    live = instance.initial_live_ids
    used: set[str] = set()
    for _turn in range(MAX_QUERY_TURNS):
        if len(live) == 1:
            break
        available = [option for option in QUERY_OPTION_IDS if option not in used]
        option = min(
            available,
            key=lambda item: (
                -query_information(instance, item, live).expected_information_bits,
                _public_query_tiebreak(instance, item),
            ),
        )
        accepted = instance.oracle_label(option)
        observations.append(QueryObservation(option, accepted))
        used.add(option)
        replay = replay_queries(instance, [item.option_id for item in observations])
        live = replay.final_ids
    return tuple(observations)


def _probabilities(scores: Tensor) -> tuple[tuple[float, ...], ...]:
    values = scores.detach().float().softmax(dim=-1).cpu().tolist()
    return tuple(tuple(float(value) for value in row) for row in values)


def _nll(scores: Tensor, targets: Tensor) -> Tensor:
    float_scores = scores.float()
    target = targets.to(device=float_scores.device, dtype=torch.long)
    return -float_scores.log_softmax(dim=-1).gather(-1, target[:, None]).squeeze(-1)


def _reference_decisions(
    policy: ActionPolicy,
    instance: GameInstance,
    renderer: RendererName,
) -> tuple[list[Tensor], list[Decision], tuple[QueryObservation, ...]]:
    observations: list[QueryObservation] = []
    loss_values: list[Tensor] = []
    decisions: list[Decision] = []
    for reference in reference_inquiry(instance):
        prompt, labels, meanings = query_prompt(instance, renderer, observations)
        selected = meanings.index(reference.option_id)
        scores = policy.score((prompt,), labels)
        loss_values.extend(_nll(scores, torch.tensor([selected], device=scores.device)))
        decisions.append(Decision("query", prompt, labels, selected, _probabilities(scores)[0]))
        observations.append(reference)
    if (
        len(observations) < MAX_QUERY_TURNS
        and len(replay_queries(instance, [item.option_id for item in observations]).final_ids) == 1
    ):
        prompt, labels, meanings = query_prompt(instance, renderer, observations)
        selected = meanings.index("READY")
        scores = policy.score((prompt,), labels)
        loss_values.extend(_nll(scores, torch.tensor([selected], device=scores.device)))
        decisions.append(Decision("ready", prompt, labels, selected, _probabilities(scores)[0]))
    return loss_values, decisions, tuple(observations)


def process_sft_rotation(
    policy: ActionPolicy,
    instance: GameInstance,
    renderer: RendererName,
) -> RotationUpdate:
    loss_values, decisions, observations = _reference_decisions(policy, instance, renderer)
    inquiry_count = len(loss_values)

    prompt = candidate_prompt(instance, renderer, observations)
    candidate_index = CANDIDATE_ACTION_LABELS.index(instance.official_candidate_id)
    candidate_scores = policy.score((prompt,), CANDIDATE_ACTION_LABELS)
    loss_values.extend(
        _nll(candidate_scores, torch.tensor([candidate_index], device=candidate_scores.device))
    )
    decisions.append(
        Decision(
            "candidate",
            prompt,
            CANDIDATE_ACTION_LABELS,
            candidate_index,
            _probabilities(candidate_scores)[0],
        )
    )

    terminal_prompts = tuple(
        classification_prompt(instance, renderer, observations, _scene_at(index))
        for index in instance.material.terminal
    )
    terminal_targets = torch.tensor(
        [0 if accepted else 1 for accepted in instance.terminal_labels],
        dtype=torch.long,
    )
    terminal_scores = policy.score(terminal_prompts, BINARY_ACTION_LABELS)
    loss_values.extend(_nll(terminal_scores, terminal_targets.to(terminal_scores.device)))
    terminal_probabilities = _probabilities(terminal_scores)
    for prompt, selected, probabilities in zip(
        terminal_prompts,
        terminal_targets.tolist(),
        terminal_probabilities,
        strict=True,
    ):
        decisions.append(Decision("classification", prompt, BINARY_ACTION_LABELS, selected, probabilities))
    if inquiry_count < 1:
        raise HiddenLawExperimentError("reference policy emitted no inquiry action")
    inquiry_loss = torch.stack(loss_values[:inquiry_count]).mean()
    candidate_loss = loss_values[inquiry_count]
    classification_loss = torch.stack(loss_values[inquiry_count + 1 :]).mean()
    loss = (inquiry_loss + candidate_loss + classification_loss) / 3.0
    correct = sum(
        decision.probabilities.index(max(decision.probabilities)) == decision.selected_index
        for decision in decisions
    )
    trajectory = CollectedTrajectory(
        instance.instance_id,
        tuple(decisions),
        observations,
        instance.official_candidate_id,
        instance.terminal_labels,
        True,
        1.0,
        1.0,
    )
    return RotationUpdate(
        loss=loss,
        metrics={
            "loss": float(loss.detach().cpu()),
            "inquiry_loss": float(inquiry_loss.detach().cpu()),
            "rule_loss": float(candidate_loss.detach().cpu()),
            "classification_loss": float(classification_loss.detach().cpu()),
            "phase_weight_inquiry": 1.0 / 3.0,
            "phase_weight_rule": 1.0 / 3.0,
            "phase_weight_classification": 1.0 / 3.0,
            "reference_action_accuracy": correct / len(decisions),
            "decision_count": float(len(decisions)),
        },
        trajectories=(trajectory,),
    )


def _sample_indices(scores: Tensor, *, generator: torch.Generator) -> Tensor:
    probabilities = scores.detach().float().softmax(dim=-1).cpu()
    return torch.multinomial(probabilities, 1, replacement=True, generator=generator).squeeze(-1)


@torch.no_grad()
def collect_outcome_trajectories(
    policy: ActionPolicy,
    instance: GameInstance,
    renderer: RendererName,
    *,
    count: int,
    generator: torch.Generator,
) -> tuple[CollectedTrajectory, ...]:
    if count < 2:
        raise ValueError("outcome RL requires at least two trajectories per Official")
    trajectories: list[CollectedTrajectory] = []
    for _ in range(count):
        observations: list[QueryObservation] = []
        decisions: list[Decision] = []
        for _turn in range(MAX_QUERY_TURNS):
            prompt, labels, meanings = query_prompt(instance, renderer, observations)
            scores = policy.score((prompt,), labels)
            selected = int(_sample_indices(scores, generator=generator)[0])
            decisions.append(Decision("query", prompt, labels, selected, _probabilities(scores)[0]))
            meaning = meanings[selected]
            if meaning == "READY":
                break
            accepted = instance.oracle_label(meaning)
            observations.append(QueryObservation(meaning, accepted))

        frozen_observations = tuple(observations)
        prompt = candidate_prompt(instance, renderer, frozen_observations)
        candidate_scores = policy.score((prompt,), CANDIDATE_ACTION_LABELS)
        candidate_index = int(_sample_indices(candidate_scores, generator=generator)[0])
        decisions.append(
            Decision(
                "candidate",
                prompt,
                CANDIDATE_ACTION_LABELS,
                candidate_index,
                _probabilities(candidate_scores)[0],
            )
        )

        terminal_prompts = tuple(
            classification_prompt(instance, renderer, frozen_observations, _scene_at(index))
            for index in instance.material.terminal
        )
        terminal_scores = policy.score(terminal_prompts, BINARY_ACTION_LABELS)
        terminal_indices = _sample_indices(terminal_scores, generator=generator)
        terminal_probabilities = _probabilities(terminal_scores)
        for prompt, selected, probabilities in zip(
            terminal_prompts,
            terminal_indices.tolist(),
            terminal_probabilities,
            strict=True,
        ):
            decisions.append(
                Decision("classification", prompt, BINARY_ACTION_LABELS, selected, probabilities)
            )
        predictions = tuple(index == 0 for index in terminal_indices.tolist())
        selected_candidate = CANDIDATE_ACTION_LABELS[candidate_index]
        reward, exact, accuracy = terminal_reward(
            selected_candidate_id=selected_candidate,
            official_candidate_id=instance.official_candidate_id,
            predictions=predictions,
            targets=instance.terminal_labels,
        )
        trajectories.append(
            CollectedTrajectory(
                instance.instance_id,
                tuple(decisions),
                frozen_observations,
                selected_candidate,
                predictions,
                exact,
                accuracy,
                reward,
            )
        )
    return tuple(trajectories)


def _rescore_collected(
    policy: ActionPolicy,
    trajectories: Sequence[CollectedTrajectory],
) -> tuple[RescoredTrajectory, ...]:
    grouped: dict[tuple[str, ...], list[tuple[int, int, Decision]]] = defaultdict(list)
    for trajectory_index, trajectory in enumerate(trajectories):
        for decision_index, decision in enumerate(trajectory.decisions):
            grouped[decision.action_labels].append((trajectory_index, decision_index, decision))

    score_rows: dict[tuple[int, int], Tensor] = {}
    for labels, entries in grouped.items():
        scores = policy.score(tuple(entry[2].prompt for entry in entries), labels).float()
        for row, (trajectory_index, decision_index, _decision) in enumerate(entries):
            score_rows[(trajectory_index, decision_index)] = scores[row]

    rescored: list[RescoredTrajectory] = []
    for trajectory_index, trajectory in enumerate(trajectories):
        rows = [score_rows[(trajectory_index, index)] for index in range(len(trajectory.decisions))]
        actions = [decision.selected_index for decision in trajectory.decisions]
        rescored.append(rescore_trajectory(rows, actions))
    return tuple(rescored)


def outcome_rl_rotation(
    policy: ActionPolicy,
    instance: GameInstance,
    renderer: RendererName,
    *,
    generator: torch.Generator,
    trajectories_per_official: int = 4,
    entropy_coefficient: float = 0.01,
) -> RotationUpdate:
    trajectories = collect_outcome_trajectories(
        policy,
        instance,
        renderer,
        count=trajectories_per_official,
        generator=generator,
    )
    rescored = _rescore_collected(policy, trajectories)
    result = grouped_leave_one_out_trajectory_loss(
        rescored,
        [trajectory.reward for trajectory in trajectories],
        [instance.instance_id] * len(trajectories),
        entropy_coefficient=entropy_coefficient,
    )
    return RotationUpdate(
        loss=result.loss,
        metrics={
            "loss": float(result.loss.detach().cpu()),
            "policy_loss": float(result.policy_loss.detach().cpu()),
            "mean_reward": float(result.mean_reward.detach().cpu()),
            "mean_trajectory_entropy_sum": float(result.mean_trajectory_entropy_sum.detach().cpu()),
            "exact_candidate_rate": sum(item.exact_candidate for item in trajectories) / len(trajectories),
            "classification_accuracy": sum(item.classification_accuracy for item in trajectories)
            / len(trajectories),
            "mean_decision_count": sum(len(item.decisions) for item in trajectories) / len(trajectories),
        },
        trajectories=trajectories,
    )


def warmup_steps(total_steps: int, fraction: float) -> int:
    if total_steps < 1 or not math.isfinite(fraction) or not 0 <= fraction <= 1:
        raise ValueError("invalid warm-up specification")
    return math.ceil(total_steps * fraction)


def learning_rate_at_step(base_learning_rate: float, step: int, warmup: int) -> float:
    if base_learning_rate <= 0 or step < 1 or warmup < 0:
        raise ValueError("invalid learning-rate schedule input")
    multiplier = 1.0 if warmup == 0 else min(1.0, step / warmup)
    return base_learning_rate * multiplier


def train_role_neutral_block(
    *,
    step: int,
    algorithm: str,
    policy: ActionPolicy,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    games: Sequence[GameInstance],
    renderer: RendererName,
    generator: torch.Generator,
    base_learning_rate: float,
    warmup: int,
    gradient_clip_norm: float,
    entropy_coefficient: float,
    trajectories_per_official: int = 4,
    after_rotation: Callable[[int, nn.Module], None] | None = None,
) -> BlockUpdate:
    """Accumulate exactly four role losses, then and only then update weights."""

    if len(games) != 4 or tuple(game.official_candidate_id for game in games) != CANDIDATE_IDS:
        raise HiddenLawExperimentError("a role-neutral block needs the four A/B/C/D Officials")
    if algorithm not in {"process_sft", "outcome_rl"}:
        raise HiddenLawExperimentError(f"unknown algorithm: {algorithm!r}")
    if not math.isfinite(gradient_clip_norm) or gradient_clip_norm <= 0:
        raise ValueError("gradient clip norm must be finite and positive")

    current_learning_rate = learning_rate_at_step(base_learning_rate, step, warmup)
    for group in optimizer.param_groups:
        group["lr"] = current_learning_rate
    optimizer.zero_grad(set_to_none=True)
    rotations: list[RotationUpdate] = []
    for index, game in enumerate(games):
        rotation = (
            process_sft_rotation(policy, game, renderer)
            if algorithm == "process_sft"
            else outcome_rl_rotation(
                policy,
                game,
                renderer,
                generator=generator,
                trajectories_per_official=trajectories_per_official,
                entropy_coefficient=entropy_coefficient,
            )
        )
        (rotation.loss / 4.0).backward()
        rotations.append(rotation)
        if after_rotation is not None:
            after_rotation(index, model)
    gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)
    if not bool(torch.isfinite(torch.as_tensor(gradient_norm)).detach().cpu()):
        raise FloatingPointError("gradient norm is not finite")
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)

    return BlockUpdate(
        step=step,
        algorithm=algorithm,
        loss=sum(float(rotation.loss.detach().cpu()) for rotation in rotations) / 4,
        learning_rate=current_learning_rate,
        gradient_norm=float(torch.as_tensor(gradient_norm).detach().cpu()),
        rotation_metrics=tuple(dict(rotation.metrics) for rotation in rotations),
        trajectories=tuple(trajectory for rotation in rotations for trajectory in rotation.trajectories),
    )
