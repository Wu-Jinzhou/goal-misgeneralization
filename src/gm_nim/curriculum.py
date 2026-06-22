from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from pathlib import Path

import torch

from .hf import (
    CausalCollator,
    JsonlCausalDataset,
    evaluate_bounded_jsonl,
    load_causal_lm,
    load_tokenizer,
)


@dataclass
class CurriculumConfig:
    phase1_file: str
    phase2_file: str
    output_dir: str
    model: str = "410m"
    eval_files: list[str] = field(default_factory=list)
    max_length: int = 128
    phase_steps: int = 75_000
    batch_size: int = 64
    replay_ratio: float = 0.2
    learning_rate: float = 3e-5
    weight_decay: float = 0.05
    warmup_ratio: float = 0.1
    save_steps: int = 5_000
    eval_steps: int = 5_000
    seed: int = 0
    bf16: bool = False
    gradient_checkpointing: bool = False
    factors: list[int] = field(default_factory=lambda: [2, 3, 4])


def _sample(pool: JsonlCausalDataset, count: int, rng: random.Random) -> list[dict]:
    return [pool[rng.randrange(len(pool))] for _ in range(count)]


def _save_checkpoint(model, tokenizer, output_dir: Path, step: int) -> None:
    checkpoint = output_dir / f"checkpoint-{step}"
    checkpoint.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(checkpoint)
    tokenizer.save_pretrained(checkpoint)


def run_curriculum(config: CurriculumConfig) -> None:
    from transformers import get_cosine_schedule_with_warmup, set_seed

    set_seed(config.seed)
    rng = random.Random(config.seed)
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = load_tokenizer(config.model)
    model = load_causal_lm(config.model)
    if config.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.config.use_cache = False

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if config.bf16:
        model = model.to(device=device, dtype=torch.bfloat16)
    else:
        model = model.to(device)
    model.train()

    phase1 = JsonlCausalDataset(config.phase1_file, tokenizer, max_length=config.max_length)
    phase2 = JsonlCausalDataset(config.phase2_file, tokenizer, max_length=config.max_length)
    collator = CausalCollator(tokenizer)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    total_steps = config.phase_steps * 2
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_steps * config.warmup_ratio),
        num_training_steps=total_steps,
    )
    metrics_path = output_dir / "curriculum_metrics.jsonl"
    with metrics_path.open("w", encoding="utf-8") as metrics_handle:
        for step in range(1, total_steps + 1):
            if step <= config.phase_steps:
                features = _sample(phase1, config.batch_size, rng)
                phase = 1
            else:
                replay = round(config.batch_size * config.replay_ratio)
                fresh = config.batch_size - replay
                features = _sample(phase1, replay, rng) + _sample(phase2, fresh, rng)
                rng.shuffle(features)
                phase = 2
            batch = {
                key: value.to(device)
                for key, value in collator(features).items()
                if torch.is_tensor(value)
            }
            output = model(**batch)
            loss = output.loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)

            if step % 50 == 0:
                metrics_handle.write(
                    json.dumps(
                        {
                            "step": step,
                            "phase": phase,
                            "loss": float(loss.detach().cpu()),
                            "lr": scheduler.get_last_lr()[0],
                        },
                        sort_keys=True,
                    )
                    + "\n"
                )
                metrics_handle.flush()

            if config.eval_files and step % config.eval_steps == 0:
                model.eval()
                for eval_file in config.eval_files:
                    metrics, _ = evaluate_bounded_jsonl(
                        model,
                        tokenizer,
                        eval_file,
                        factors=config.factors,
                        batch_size=config.batch_size,
                    )
                    metrics_handle.write(
                        json.dumps(
                            {"step": step, "phase": phase, "eval_file": eval_file, **metrics},
                            sort_keys=True,
                        )
                        + "\n"
                    )
                metrics_handle.flush()
                model.train()

            if step % config.save_steps == 0:
                _save_checkpoint(model, tokenizer, output_dir, step)

    _save_checkpoint(model, tokenizer, output_dir, total_steps)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    (output_dir / "curriculum_config.json").write_text(
        json.dumps(config.__dict__, indent=2, sort_keys=True),
        encoding="utf-8",
    )

