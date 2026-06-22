from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
from torch.nn import functional as F

from .hf import load_causal_lm, load_tokenizer
from .rl_games import (
    Action,
    GameConfig,
    action_matches_optimal_proxy,
    apply_action,
    cold_state_proxy_names,
    fixed_policy_action,
    format_action,
    has_winning_action,
    initial_state,
    is_legal_action,
    is_terminal,
    legal_actions,
    parse_action,
    render_prompt,
)


@dataclass
class GameRLConfig:
    output_dir: str
    model: str = "410m"
    game: str = "bounded"
    mr: int = 5
    train_opponent: str = "random"
    eval_opponents: list[str] = field(default_factory=lambda: ["random", "optimal"])
    self_play: bool = False
    randomize_player: bool = True
    min_pile: int = 20
    max_pile: int = 400
    pile_count: int = 3
    min_heap: int = 1
    max_heap: int = 120
    fib_min_pile: int = 8
    fib_max_pile: int = 160
    wythoff_max_heap: int = 120
    steps: int = 30_000
    episodes_per_step: int = 8
    update_batch_size: int = 16
    eval_episodes: int = 200
    max_turns: int = 128
    max_prompt_length: int = 160
    max_new_tokens: int = 8
    learning_rate: float = 1e-6
    weight_decay: float = 0.0
    warmup_steps: int = 0
    kl_coef: float = 0.02
    entropy_coef: float = 0.0
    invalid_reward: float = -1.0
    win_reward: float = 1.0
    loss_reward: float = -1.0
    discount: float = 1.0
    normalize_advantages: bool = True
    temperature: float = 1.0
    top_p: float = 1.0
    save_steps: int = 1000
    eval_steps: int = 1000
    seed: int = 0
    bf16: bool = False
    gradient_checkpointing: bool = False
    max_grad_norm: float = 1.0
    proxy_metrics: list[str] = field(default_factory=list)


@dataclass
class ActionRecord:
    prompt: str
    completion: str
    player: int
    reward: float
    state: dict[str, Any]
    action: Action | None
    legal: bool
    generated_by_policy: bool = True


def _game_config(config: GameRLConfig) -> GameConfig:
    return GameConfig(
        game=config.game,
        mr=config.mr,
        min_pile=config.min_pile,
        max_pile=config.max_pile,
        pile_count=config.pile_count,
        min_heap=config.min_heap,
        max_heap=config.max_heap,
        fib_min_pile=config.fib_min_pile,
        fib_max_pile=config.fib_max_pile,
        wythoff_max_heap=config.wythoff_max_heap,
    )


def _encode_prompt(tokenizer, prompt: str, config: GameRLConfig, device: torch.device):
    encoded = tokenizer(
        prompt + "\n",
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=config.max_prompt_length,
    )
    return {key: value.to(device) for key, value in encoded.items()}


@torch.no_grad()
def _model_action(
    model,
    tokenizer,
    state: dict[str, Any],
    config: GameRLConfig,
    *,
    sample: bool,
) -> tuple[Action | None, str, str]:
    device = next(model.parameters()).device
    prompt = render_prompt(state)
    encoded = _encode_prompt(tokenizer, prompt, config, device)
    generate_kwargs = {
        **encoded,
        "do_sample": sample,
        "max_new_tokens": config.max_new_tokens,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }
    if sample:
        generate_kwargs["temperature"] = config.temperature
        generate_kwargs["top_p"] = config.top_p
    sequence = model.generate(**generate_kwargs)
    completion = tokenizer.decode(
        sequence[0, encoded["input_ids"].shape[1] :],
        skip_special_tokens=True,
    )
    return parse_action(state["game"], completion), completion, prompt


def _completion_logprobs(
    model,
    tokenizer,
    prompts: list[str],
    completions: list[str],
    *,
    max_length: int,
    device: torch.device,
) -> torch.Tensor:
    rows = []
    for prompt, completion in zip(prompts, completions):
        prompt_ids = tokenizer(prompt + "\n", add_special_tokens=False)["input_ids"]
        completion_ids = tokenizer(completion, add_special_tokens=False)["input_ids"]
        if not completion_ids:
            completion_ids = [tokenizer.eos_token_id]
        input_ids = (prompt_ids + completion_ids)[-max_length:]
        label_cut = max(0, len(prompt_ids) - max(0, len(prompt_ids) + len(completion_ids) - max_length))
        labels = [-100] * label_cut + input_ids[label_cut:]
        labels = labels[-len(input_ids) :]
        rows.append((input_ids, labels))
    max_len = max(len(input_ids) for input_ids, _ in rows)
    input_batch = []
    label_batch = []
    attention_batch = []
    for input_ids, labels in rows:
        pad = max_len - len(input_ids)
        input_batch.append(input_ids + [tokenizer.pad_token_id] * pad)
        label_batch.append(labels + [-100] * pad)
        attention_batch.append([1] * len(input_ids) + [0] * pad)
    input_tensor = torch.tensor(input_batch, dtype=torch.long, device=device)
    label_tensor = torch.tensor(label_batch, dtype=torch.long, device=device)
    attention_tensor = torch.tensor(attention_batch, dtype=torch.long, device=device)
    output = model(input_ids=input_tensor, attention_mask=attention_tensor)
    logits = output.logits[:, :-1, :]
    shifted_labels = label_tensor[:, 1:]
    mask = shifted_labels.ne(-100)
    safe_labels = shifted_labels.masked_fill(~mask, 0)
    log_probs = F.log_softmax(logits, dim=-1)
    token_logprobs = log_probs.gather(-1, safe_labels.unsqueeze(-1)).squeeze(-1)
    return (token_logprobs * mask).sum(dim=1)


def _completion_entropy(
    model,
    tokenizer,
    prompts: list[str],
    completions: list[str],
    *,
    max_length: int,
    device: torch.device,
) -> torch.Tensor:
    rows = []
    for prompt, completion in zip(prompts, completions):
        prompt_ids = tokenizer(prompt + "\n", add_special_tokens=False)["input_ids"]
        completion_ids = tokenizer(completion, add_special_tokens=False)["input_ids"] or [tokenizer.eos_token_id]
        input_ids = (prompt_ids + completion_ids)[-max_length:]
        label_cut = max(0, len(prompt_ids) - max(0, len(prompt_ids) + len(completion_ids) - max_length))
        labels = [-100] * label_cut + input_ids[label_cut:]
        labels = labels[-len(input_ids) :]
        rows.append((input_ids, labels))
    max_len = max(len(input_ids) for input_ids, _ in rows)
    input_batch = []
    label_batch = []
    attention_batch = []
    for input_ids, labels in rows:
        pad = max_len - len(input_ids)
        input_batch.append(input_ids + [tokenizer.pad_token_id] * pad)
        label_batch.append(labels + [-100] * pad)
        attention_batch.append([1] * len(input_ids) + [0] * pad)
    input_tensor = torch.tensor(input_batch, dtype=torch.long, device=device)
    label_tensor = torch.tensor(label_batch, dtype=torch.long, device=device)
    attention_tensor = torch.tensor(attention_batch, dtype=torch.long, device=device)
    output = model(input_ids=input_tensor, attention_mask=attention_tensor)
    logits = output.logits[:, :-1, :]
    mask = label_tensor[:, 1:].ne(-100)
    probs = F.softmax(logits, dim=-1)
    entropy = -(probs * F.log_softmax(logits, dim=-1)).sum(dim=-1)
    return (entropy * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)


def _assign_outcome_rewards(
    records: list[ActionRecord],
    *,
    winner: int,
    config: GameRLConfig,
    invalid_player: int | None = None,
) -> list[ActionRecord]:
    updated = []
    last_index = max(0, len(records) - 1)
    for index, record in enumerate(records):
        multiplier = config.discount ** (last_index - index)
        reward = config.win_reward if record.player == winner else config.loss_reward
        reward *= multiplier
        if invalid_player is not None and record.player == invalid_player and not record.legal:
            reward = config.invalid_reward
        updated.append(
            ActionRecord(
                prompt=record.prompt,
                completion=record.completion,
                player=record.player,
                reward=reward,
                state=record.state,
                action=record.action,
                legal=record.legal,
                generated_by_policy=record.generated_by_policy,
            )
        )
    return updated


def rollout_episode(
    model,
    tokenizer,
    config: GameRLConfig,
    rng: random.Random,
    *,
    sample: bool,
    opponent: str,
) -> tuple[list[ActionRecord], dict[str, Any]]:
    state = initial_state(_game_config(config), rng)
    learner_player = rng.randrange(2) if config.randomize_player else 0
    current_player = 0
    records: list[ActionRecord] = []
    winner: int | None = None
    invalid_player: int | None = None

    for turn in range(config.max_turns):
        if is_terminal(state):
            winner = 1 - current_player
            break
        model_turn = config.self_play or current_player == learner_player
        state_before = dict(state)
        if model_turn:
            action, completion, prompt = _model_action(model, tokenizer, state, config, sample=sample)
            legal = is_legal_action(state, action)
            records.append(
                ActionRecord(
                    prompt=prompt,
                    completion=completion,
                    player=current_player,
                    reward=0.0,
                    state=state_before,
                    action=action,
                    legal=legal,
                )
            )
            if not legal:
                invalid_player = current_player
                winner = 1 - current_player
                break
        else:
            action = fixed_policy_action(opponent, state, rng)
            legal = True
        state = apply_action(state, action)
        if is_terminal(state):
            winner = current_player
            break
        current_player = 1 - current_player
    if winner is None:
        winner = 1 - current_player
    records = _assign_outcome_rewards(
        records,
        winner=winner,
        config=config,
        invalid_player=invalid_player,
    )
    return records, {
        "winner": winner,
        "learner_player": learner_player,
        "learner_won": winner == learner_player,
        "invalid": invalid_player is not None,
        "turns": turn + 1,
        "records": len(records),
    }


def _advantages(rewards: torch.Tensor, normalize: bool) -> torch.Tensor:
    advantages = rewards - rewards.mean()
    if normalize and len(rewards) > 1:
        advantages = advantages / rewards.std().clamp_min(1e-6)
    return advantages


def _update_policy(
    policy,
    reference,
    tokenizer,
    optimizer,
    records: list[ActionRecord],
    config: GameRLConfig,
) -> dict[str, float]:
    device = next(policy.parameters()).device
    prompts = [record.prompt for record in records]
    completions = [record.completion for record in records]
    rewards = torch.tensor([record.reward for record in records], dtype=torch.float, device=device)
    advantages = _advantages(rewards, config.normalize_advantages)

    order = torch.randperm(len(records), device=device)
    total_loss = 0.0
    total_kl = 0.0
    total_entropy = 0.0
    for start in range(0, len(order), config.update_batch_size):
        index = order[start : start + config.update_batch_size].cpu().tolist()
        mb_prompts = [prompts[i] for i in index]
        mb_completions = [completions[i] for i in index]
        mb_advantages = advantages[index]
        logprobs = _completion_logprobs(
            policy,
            tokenizer,
            mb_prompts,
            mb_completions,
            max_length=config.max_prompt_length + config.max_new_tokens,
            device=device,
        )
        with torch.no_grad():
            ref_logprobs = _completion_logprobs(
                reference,
                tokenizer,
                mb_prompts,
                mb_completions,
                max_length=config.max_prompt_length + config.max_new_tokens,
                device=device,
            )
        kl_penalty = (logprobs - ref_logprobs).square()
        loss = -(mb_advantages.detach() * logprobs).mean() + config.kl_coef * kl_penalty.mean()
        if config.entropy_coef:
            entropy = _completion_entropy(
                policy,
                tokenizer,
                mb_prompts,
                mb_completions,
                max_length=config.max_prompt_length + config.max_new_tokens,
                device=device,
            ).mean()
            loss = loss - config.entropy_coef * entropy
            total_entropy += float(entropy.detach().cpu())
        loss.backward()
        total_loss += float(loss.detach().cpu())
        total_kl += float(kl_penalty.detach().mean().cpu())
    torch.nn.utils.clip_grad_norm_(policy.parameters(), config.max_grad_norm)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    return {
        "loss": total_loss,
        "kl": total_kl,
        "entropy": total_entropy,
        "mean_reward": float(rewards.mean().detach().cpu()),
        "records": len(records),
    }


@torch.no_grad()
def evaluate_game_policy(
    model,
    tokenizer,
    config: GameRLConfig,
    *,
    opponent: str,
    episodes: int,
    seed: int,
) -> dict[str, Any]:
    rng = random.Random(seed)
    eval_config = GameRLConfig(**{**config.__dict__, "self_play": False})
    wins = 0
    invalid = 0
    total_turns = 0
    strategic_actions = 0
    exact_actions = 0
    proxy_names = config.proxy_metrics or cold_state_proxy_names(config.game)
    proxy_hits = {proxy: 0 for proxy in proxy_names}
    proxy_denoms = {proxy: 0 for proxy in proxy_names}

    for _ in range(episodes):
        records, summary = rollout_episode(
            model,
            tokenizer,
            eval_config,
            rng,
            sample=False,
            opponent=opponent,
        )
        wins += int(summary["learner_won"])
        invalid += int(summary["invalid"])
        total_turns += int(summary["turns"])
        for record in records:
            if not record.legal or record.action is None:
                continue
            if has_winning_action(record.state):
                strategic_actions += 1
                if action_matches_optimal_proxy(record.state, record.action, "optimal"):
                    exact_actions += 1
            for proxy in proxy_names:
                proxy_denoms[proxy] += 1
                proxy_hits[proxy] += int(action_matches_optimal_proxy(record.state, record.action, proxy))

    metrics: dict[str, Any] = {
        "opponent": opponent,
        "episodes": episodes,
        "win_rate": wins / episodes if episodes else 0.0,
        "invalid_rate": invalid / episodes if episodes else 0.0,
        "avg_turns": total_turns / episodes if episodes else 0.0,
        "optimal_action_rate": exact_actions / strategic_actions if strategic_actions else 0.0,
        "strategic_action_count": strategic_actions,
    }
    for proxy in proxy_names:
        denom = proxy_denoms[proxy]
        metrics[f"{proxy}_rate"] = proxy_hits[proxy] / denom if denom else 0.0
    return metrics


def _save_checkpoint(model, tokenizer, output_dir: Path, step: int) -> None:
    checkpoint = output_dir / f"checkpoint-{step}"
    checkpoint.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(checkpoint)
    tokenizer.save_pretrained(checkpoint)


def run_game_rl_training(config: GameRLConfig) -> None:
    from transformers import get_cosine_schedule_with_warmup, set_seed

    set_seed(config.seed)
    rng = random.Random(config.seed)
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = load_tokenizer(config.model)
    policy = load_causal_lm(config.model)
    reference = load_causal_lm(config.model)
    if config.gradient_checkpointing:
        policy.gradient_checkpointing_enable()
        policy.config.use_cache = False

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if config.bf16 else None
    policy = policy.to(device=device, dtype=dtype)
    reference = reference.to(device=device, dtype=dtype)
    reference.eval()
    for parameter in reference.parameters():
        parameter.requires_grad_(False)

    optimizer = torch.optim.AdamW(
        policy.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=config.warmup_steps,
        num_training_steps=config.steps,
    )
    metrics_path = output_dir / "game_rl_metrics.jsonl"

    with metrics_path.open("w", encoding="utf-8") as metrics_handle:
        for step in range(1, config.steps + 1):
            policy.eval()
            records: list[ActionRecord] = []
            summaries = []
            while len(summaries) < config.episodes_per_step:
                episode_records, summary = rollout_episode(
                    policy,
                    tokenizer,
                    config,
                    rng,
                    sample=True,
                    opponent=config.train_opponent,
                )
                records.extend(episode_records)
                summaries.append(summary)
            policy.train()
            if records:
                update = _update_policy(policy, reference, tokenizer, optimizer, records, config)
                scheduler.step()
            else:
                update = {"loss": 0.0, "kl": 0.0, "entropy": 0.0, "mean_reward": 0.0, "records": 0}
            if step % 10 == 0:
                metrics_handle.write(
                    json.dumps(
                        {
                            "step": step,
                            "phase": "train",
                            "train_opponent": config.train_opponent,
                            "learner_win_rate": sum(s["learner_won"] for s in summaries) / len(summaries),
                            "invalid_rate": sum(s["invalid"] for s in summaries) / len(summaries),
                            "avg_turns": sum(s["turns"] for s in summaries) / len(summaries),
                            "lr": scheduler.get_last_lr()[0],
                            **update,
                        },
                        sort_keys=True,
                    )
                    + "\n"
                )
                metrics_handle.flush()
            if step % config.eval_steps == 0:
                policy.eval()
                for offset, opponent in enumerate(config.eval_opponents):
                    metrics = evaluate_game_policy(
                        policy,
                        tokenizer,
                        config,
                        opponent=opponent,
                        episodes=config.eval_episodes,
                        seed=config.seed + 100_000 + step * 17 + offset,
                    )
                    metrics_handle.write(
                        json.dumps({"step": step, "phase": "eval", **metrics}, sort_keys=True)
                        + "\n"
                    )
                metrics_handle.flush()
            if step % config.save_steps == 0:
                _save_checkpoint(policy, tokenizer, output_dir, step)

    _save_checkpoint(policy, tokenizer, output_dir, config.steps)
    policy.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    (output_dir / "game_rl_config.json").write_text(
        json.dumps(config.__dict__, indent=2, sort_keys=True),
        encoding="utf-8",
    )
