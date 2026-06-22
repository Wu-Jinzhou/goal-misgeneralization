from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
from torch.nn import functional as F

from .data import Example, read_jsonl
from .games import action_to_residue
from .hf import load_causal_lm, load_tokenizer
from .metrics import parse_bounded_move


@dataclass
class RLConfig:
    train_file: str
    output_dir: str
    model: str = "410m"
    eval_files: list[str] = field(default_factory=list)
    max_prompt_length: int = 120
    max_new_tokens: int = 8
    steps: int = 30_000
    batch_size: int = 32
    mini_batch_size: int = 8
    samples_per_prompt: int = 4
    learning_rate: float = 1e-6
    weight_decay: float = 0.0
    warmup_steps: int = 0
    kl_coef: float = 0.02
    entropy_coef: float = 0.0
    reward_exact: float = 1.0
    reward_invalid: float = -0.25
    reward_wrong: float = 0.0
    coarsened_factors: list[int] = field(default_factory=list)
    coarsened_reward: float = 0.25
    normalize_group_advantages: bool = True
    temperature: float = 1.0
    top_p: float = 1.0
    save_steps: int = 1000
    eval_steps: int = 1000
    seed: int = 0
    bf16: bool = False
    gradient_checkpointing: bool = False
    max_grad_norm: float = 1.0


def bounded_nim_reward(
    prediction: int | None,
    label: int,
    *,
    mr: int,
    exact_reward: float = 1.0,
    wrong_reward: float = 0.0,
    invalid_reward: float = -0.25,
    coarsened_factors: list[int] | None = None,
    coarsened_reward: float = 0.25,
) -> float:
    if prediction is None or prediction not in {-1, *range(1, mr + 1)}:
        return invalid_reward
    if prediction == label:
        return exact_reward
    modulus = mr + 1
    pred_residue = action_to_residue(prediction, modulus)
    label_residue = action_to_residue(label, modulus)
    for factor in coarsened_factors or []:
        if pred_residue % factor == label_residue % factor:
            return coarsened_reward
    return wrong_reward


def _infer_mr(example: Example) -> int:
    try:
        return int(example.metadata["mr"])
    except KeyError as exc:
        raise ValueError("RL bounded-Nim training requires metadata.mr in each example") from exc


def _encode_prompts(tokenizer, prompts: list[str], max_prompt_length: int, device: torch.device):
    encoded = tokenizer(
        [prompt + "\n" for prompt in prompts],
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_prompt_length,
    )
    return {key: value.to(device) for key, value in encoded.items()}


def _sequence_mask(attention_mask: torch.Tensor, generated_ids: torch.Tensor) -> torch.Tensor:
    prompt_width = attention_mask.shape[1]
    mask = torch.zeros_like(generated_ids, dtype=torch.bool)
    mask[:, prompt_width:] = True
    return mask


def _full_generated_attention_mask(
    prompt_attention_mask: torch.Tensor,
    generated_ids: torch.Tensor,
) -> torch.Tensor:
    suffix_width = generated_ids.shape[1] - prompt_attention_mask.shape[1]
    suffix = torch.ones(
        (generated_ids.shape[0], suffix_width),
        dtype=prompt_attention_mask.dtype,
        device=prompt_attention_mask.device,
    )
    return torch.cat([prompt_attention_mask, suffix], dim=1)


def _completion_texts(tokenizer, sequences: torch.Tensor, attention_mask: torch.Tensor) -> list[str]:
    texts = []
    prompt_width = attention_mask.shape[1]
    for sequence in sequences:
        texts.append(tokenizer.decode(sequence[prompt_width:], skip_special_tokens=True))
    return texts


def _token_logprobs(model, sequences: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    output = model(
        input_ids=sequences,
        attention_mask=_full_generated_attention_mask(attention_mask, sequences),
    )
    logits = output.logits[:, :-1, :]
    labels = sequences[:, 1:]
    log_probs = F.log_softmax(logits, dim=-1)
    token_logprobs = log_probs.gather(-1, labels.unsqueeze(-1)).squeeze(-1)
    mask = _sequence_mask(attention_mask, sequences)[:, 1:]
    return (token_logprobs * mask).sum(dim=1)


def _token_entropy(model, sequences: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    output = model(
        input_ids=sequences,
        attention_mask=_full_generated_attention_mask(attention_mask, sequences),
    )
    logits = output.logits[:, :-1, :]
    probs = F.softmax(logits, dim=-1)
    entropy = -(probs * F.log_softmax(logits, dim=-1)).sum(dim=-1)
    mask = _sequence_mask(attention_mask, sequences)[:, 1:]
    denom = mask.sum(dim=1).clamp_min(1)
    return (entropy * mask).sum(dim=1) / denom


def _group_advantages(rewards: torch.Tensor, samples_per_prompt: int, normalize: bool) -> torch.Tensor:
    grouped = rewards.view(-1, samples_per_prompt)
    centered = grouped - grouped.mean(dim=1, keepdim=True)
    if normalize:
        centered = centered / grouped.std(dim=1, keepdim=True).clamp_min(1e-6)
    return centered.reshape(-1)


def _sample_batch(examples: list[Example], batch_size: int, rng: random.Random) -> list[Example]:
    return [examples[rng.randrange(len(examples))] for _ in range(batch_size)]


@torch.no_grad()
def _evaluate_rl_policy(model, tokenizer, eval_file: str, config: RLConfig) -> dict[str, Any]:
    examples = list(read_jsonl(eval_file))
    device = next(model.parameters()).device
    rows = []
    for start in range(0, len(examples), config.batch_size):
        batch = examples[start : start + config.batch_size]
        encoded = _encode_prompts(
            tokenizer,
            [example.prompt for example in batch],
            config.max_prompt_length,
            device,
        )
        sequences = model.generate(
            **encoded,
            do_sample=False,
            max_new_tokens=config.max_new_tokens,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
        completions = _completion_texts(tokenizer, sequences, encoded["attention_mask"])
        for example, completion in zip(batch, completions):
            prediction = parse_bounded_move(completion)
            mr = _infer_mr(example)
            rows.append(
                {
                    "prediction": prediction,
                    "label": int(example.label),
                    "reward": bounded_nim_reward(
                        prediction,
                        int(example.label),
                        mr=mr,
                        exact_reward=config.reward_exact,
                        wrong_reward=config.reward_wrong,
                        invalid_reward=config.reward_invalid,
                        coarsened_factors=config.coarsened_factors,
                        coarsened_reward=config.coarsened_reward,
                    ),
                    "valid": prediction in {-1, *range(1, mr + 1)},
                }
            )
    if not rows:
        return {"eval_file": eval_file, "exact": 0.0, "reward": 0.0, "valid_rate": 0.0}
    exact = sum(row["prediction"] == row["label"] for row in rows) / len(rows)
    reward = sum(row["reward"] for row in rows) / len(rows)
    valid_rate = sum(row["valid"] for row in rows) / len(rows)
    return {"eval_file": eval_file, "exact": exact, "reward": reward, "valid_rate": valid_rate}


def _save_checkpoint(model, tokenizer, output_dir: Path, step: int) -> None:
    checkpoint = output_dir / f"checkpoint-{step}"
    checkpoint.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(checkpoint)
    tokenizer.save_pretrained(checkpoint)


def run_rl_training(config: RLConfig) -> None:
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

    train_examples = list(read_jsonl(config.train_file))
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
    metrics_path = output_dir / "rl_metrics.jsonl"
    with metrics_path.open("w", encoding="utf-8") as metrics_handle:
        for step in range(1, config.steps + 1):
            policy.train()
            prompts = _sample_batch(train_examples, config.batch_size, rng)
            repeated_prompts = [
                example.prompt
                for example in prompts
                for _ in range(config.samples_per_prompt)
            ]
            repeated_examples = [
                example
                for example in prompts
                for _ in range(config.samples_per_prompt)
            ]
            encoded = _encode_prompts(
                tokenizer,
                repeated_prompts,
                config.max_prompt_length,
                device,
            )
            with torch.no_grad():
                sequences = policy.generate(
                    **encoded,
                    do_sample=True,
                    temperature=config.temperature,
                    top_p=config.top_p,
                    max_new_tokens=config.max_new_tokens,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )
                completions = _completion_texts(tokenizer, sequences, encoded["attention_mask"])
                rewards = torch.tensor(
                    [
                        bounded_nim_reward(
                            parse_bounded_move(completion),
                            int(example.label),
                            mr=_infer_mr(example),
                            exact_reward=config.reward_exact,
                            wrong_reward=config.reward_wrong,
                            invalid_reward=config.reward_invalid,
                            coarsened_factors=config.coarsened_factors,
                            coarsened_reward=config.coarsened_reward,
                        )
                        for completion, example in zip(completions, repeated_examples)
                    ],
                    dtype=torch.float,
                    device=device,
                )
                old_ref_logprobs = _token_logprobs(reference, sequences, encoded["attention_mask"])
                advantages = _group_advantages(
                    rewards, config.samples_per_prompt, config.normalize_group_advantages
                )

            order = torch.randperm(sequences.shape[0], device=device)
            total_loss = 0.0
            total_kl = 0.0
            total_entropy = 0.0
            for start in range(0, len(order), config.mini_batch_size):
                indices = order[start : start + config.mini_batch_size]
                mb_sequences = sequences[indices]
                mb_attention = encoded["attention_mask"][indices]
                mb_advantages = advantages[indices]
                mb_ref_logprobs = old_ref_logprobs[indices]
                logprobs = _token_logprobs(policy, mb_sequences, mb_attention)
                logprob_delta = logprobs - mb_ref_logprobs
                kl_penalty = logprob_delta.square()
                loss = (
                    -(mb_advantages.detach() * logprobs).mean()
                    + config.kl_coef * kl_penalty.mean()
                )
                if config.entropy_coef:
                    entropy = _token_entropy(policy, mb_sequences, mb_attention).mean()
                    loss = loss - config.entropy_coef * entropy
                    total_entropy += float(entropy.detach().cpu())
                loss.backward()
                total_loss += float(loss.detach().cpu())
                total_kl += float(kl_penalty.detach().mean().cpu())
            torch.nn.utils.clip_grad_norm_(policy.parameters(), config.max_grad_norm)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)

            if step % 10 == 0:
                metrics_handle.write(
                    json.dumps(
                        {
                            "step": step,
                            "loss": total_loss,
                            "mean_reward": float(rewards.mean().detach().cpu()),
                            "mean_advantage": float(advantages.mean().detach().cpu()),
                            "kl": total_kl,
                            "entropy": total_entropy,
                            "lr": scheduler.get_last_lr()[0],
                        },
                        sort_keys=True,
                    )
                    + "\n"
                )
                metrics_handle.flush()

            if config.eval_files and step % config.eval_steps == 0:
                policy.eval()
                for eval_file in config.eval_files:
                    metrics = _evaluate_rl_policy(policy, tokenizer, eval_file, config)
                    metrics_handle.write(
                        json.dumps({"step": step, **metrics}, sort_keys=True) + "\n"
                    )
                metrics_handle.flush()

            if step % config.save_steps == 0:
                _save_checkpoint(policy, tokenizer, output_dir, step)

    _save_checkpoint(policy, tokenizer, output_dir, config.steps)
    policy.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    (output_dir / "rl_config.json").write_text(
        json.dumps(config.__dict__, indent=2, sort_keys=True),
        encoding="utf-8",
    )
