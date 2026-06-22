from __future__ import annotations

import inspect
import json
from dataclasses import dataclass
from pathlib import Path

from .hf import CausalCollator, JsonlCausalDataset, load_causal_lm, load_tokenizer
from .interventions import (
    ContrastiveInvarianceWrapper,
    DANNModelWrapper,
    build_trainer_class,
)


@dataclass
class TrainConfig:
    train_file: str
    output_dir: str
    model: str = "410m"
    eval_file: str | None = None
    mode: str = "sft"
    max_length: int = 128
    epochs: float = 300.0
    max_steps: int = -1
    batch_size: int = 64
    eval_batch_size: int = 64
    learning_rate: float = 3e-5
    weight_decay: float = 0.05
    warmup_ratio: float = 0.1
    save_steps: int = 1000
    logging_steps: int = 50
    save_total_limit: int | None = None
    seed: int = 0
    bf16: bool = False
    fp16: bool = False
    gradient_checkpointing: bool = False
    resume_from_checkpoint: str | None = None
    layer: int = 10
    lambda_value: float = 0.05
    discriminator_hidden: int = 512


def _training_arguments(**kwargs):
    from transformers import TrainingArguments

    signature = inspect.signature(TrainingArguments.__init__)
    accepted = {key: value for key, value in kwargs.items() if key in signature.parameters}
    return TrainingArguments(**accepted)


def run_training(config: TrainConfig) -> None:
    from transformers import Trainer, set_seed

    set_seed(config.seed)
    tokenizer = load_tokenizer(config.model)
    model = load_causal_lm(config.model)
    if config.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.config.use_cache = False

    include_name_mask = config.mode == "dann"
    include_pair = config.mode == "contrastive"
    train_dataset = JsonlCausalDataset(
        config.train_file,
        tokenizer,
        max_length=config.max_length,
        include_name_mask=include_name_mask,
        include_contrastive_pair=include_pair,
    )
    eval_dataset = (
        JsonlCausalDataset(config.eval_file, tokenizer, max_length=config.max_length)
        if config.eval_file
        else None
    )
    collator = CausalCollator(tokenizer)

    trainer_cls = Trainer
    if config.mode == "dann":
        model = DANNModelWrapper(
            model,
            layer=config.layer,
            lambda_adv=config.lambda_value,
            discriminator_hidden=config.discriminator_hidden,
        )
        trainer_cls = build_trainer_class()
    elif config.mode == "contrastive":
        model = ContrastiveInvarianceWrapper(
            model,
            layer=config.layer,
            lambda_contrast=config.lambda_value,
        )
        trainer_cls = build_trainer_class()
    elif config.mode != "sft":
        raise ValueError(f"unknown training mode: {config.mode}")

    args = _training_arguments(
        output_dir=config.output_dir,
        overwrite_output_dir=False,
        num_train_epochs=config.epochs,
        max_steps=config.max_steps,
        per_device_train_batch_size=config.batch_size,
        per_device_eval_batch_size=config.eval_batch_size,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        warmup_ratio=config.warmup_ratio,
        lr_scheduler_type="cosine",
        logging_steps=config.logging_steps,
        save_steps=config.save_steps,
        save_total_limit=config.save_total_limit,
        evaluation_strategy="steps" if eval_dataset is not None else "no",
        eval_strategy="steps" if eval_dataset is not None else "no",
        eval_steps=config.save_steps,
        bf16=config.bf16,
        fp16=config.fp16,
        report_to="none",
        remove_unused_columns=False,
        dataloader_pin_memory=True,
        seed=config.seed,
    )
    trainer = trainer_cls(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collator,
        tokenizer=tokenizer,
    )
    trainer.train(resume_from_checkpoint=config.resume_from_checkpoint)
    trainer.save_model(config.output_dir)
    tokenizer.save_pretrained(config.output_dir)
    Path(config.output_dir, "train_config.json").write_text(
        json.dumps(config.__dict__, indent=2, sort_keys=True),
        encoding="utf-8",
    )
