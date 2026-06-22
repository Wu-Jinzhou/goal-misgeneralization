from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Any, Sequence

import torch
from torch.utils.data import Dataset

from .data import Example, read_jsonl
from .games import bounded_nim_target
from .metrics import bounded_accuracy, parse_bounded_move, parse_modular_answer


PYTHIA_MODELS = {
    "70m": "EleutherAI/pythia-70m-deduped",
    "160m": "EleutherAI/pythia-160m-deduped",
    "410m": "EleutherAI/pythia-410m-deduped",
}


def require_transformers():
    try:
        import transformers  # noqa: F401
    except ImportError as exc:
        raise RuntimeError(
            "transformers is required for model training/evaluation. "
            "Install this project with `pip install -e .`."
        ) from exc


def resolve_model_name(name_or_size: str) -> str:
    return PYTHIA_MODELS.get(name_or_size.lower(), name_or_size)


def load_tokenizer(model_name_or_size: str):
    require_transformers()
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(resolve_model_name(model_name_or_size))
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    return tokenizer


def load_causal_lm(model_name_or_path: str, *, torch_dtype: str | None = None):
    require_transformers()
    from transformers import AutoModelForCausalLM

    dtype = None
    if torch_dtype:
        dtype = getattr(torch, torch_dtype)
    model = AutoModelForCausalLM.from_pretrained(resolve_model_name(model_name_or_path), torch_dtype=dtype)
    if getattr(model.config, "pad_token_id", None) is None:
        model.config.pad_token_id = model.config.eos_token_id
    return model


def _tokenizer_call(tokenizer, text: str, **kwargs):
    return tokenizer(text, add_special_tokens=False, **kwargs)


def find_char_spans(text: str, needle: str) -> list[tuple[int, int]]:
    spans = []
    start = 0
    while True:
        index = text.find(needle, start)
        if index < 0:
            return spans
        spans.append((index, index + len(needle)))
        start = index + len(needle)


def _overlaps(offset: tuple[int, int], spans: Sequence[tuple[int, int]]) -> bool:
    left, right = offset
    if left == right:
        return False
    return any(left < span_right and right > span_left for span_left, span_right in spans)


def encode_prompt_target(
    tokenizer,
    prompt: str,
    target: str,
    max_length: int,
    *,
    name_spans: Sequence[tuple[int, int]] | None = None,
) -> dict[str, Any]:
    prompt_with_sep = prompt + "\n"
    target_with_eos = target + tokenizer.eos_token
    prompt_encoded = _tokenizer_call(
        tokenizer,
        prompt_with_sep,
        return_offsets_mapping=name_spans is not None,
    )
    target_ids = _tokenizer_call(tokenizer, target_with_eos)["input_ids"]
    prompt_ids = prompt_encoded["input_ids"]
    labels = [-100] * len(prompt_ids) + target_ids
    input_ids = prompt_ids + target_ids
    attention_mask = [1] * len(input_ids)

    name_token_mask: list[int] | None = None
    if name_spans is not None:
        offsets = prompt_encoded["offset_mapping"]
        prompt_name_mask = [int(_overlaps(offset, name_spans)) for offset in offsets]
        name_token_mask = prompt_name_mask + [0] * len(target_ids)

    if len(input_ids) > max_length:
        overflow = len(input_ids) - max_length
        input_ids = input_ids[overflow:]
        labels = labels[overflow:]
        attention_mask = attention_mask[overflow:]
        if name_token_mask is not None:
            name_token_mask = name_token_mask[overflow:]
        prompt_len = max(0, len(prompt_ids) - overflow)
    else:
        prompt_len = len(prompt_ids)

    row: dict[str, Any] = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
        "prompt_len": prompt_len,
    }
    if name_token_mask is not None:
        row["name_token_mask"] = name_token_mask
    return row


def encode_prompt_only(tokenizer, prompt: str, max_length: int) -> dict[str, Any]:
    encoded = _tokenizer_call(tokenizer, prompt + "\n")
    input_ids = encoded["input_ids"][-max_length:]
    return {"input_ids": input_ids, "attention_mask": [1] * len(input_ids)}


class JsonlCausalDataset(Dataset):
    def __init__(
        self,
        path: str | Path,
        tokenizer,
        max_length: int = 128,
        *,
        include_name_mask: bool = False,
        include_contrastive_pair: bool = False,
    ) -> None:
        self.examples = list(read_jsonl(path))
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.include_name_mask = include_name_mask
        self.include_contrastive_pair = include_contrastive_pair

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        example = self.examples[index]
        name_spans = None
        if self.include_name_mask:
            names = example.metadata.get("names")
            if names:
                name_spans = []
                for name in names:
                    name_spans.extend(find_char_spans(example.prompt, name))
        row = encode_prompt_target(
            self.tokenizer,
            example.prompt,
            example.target,
            self.max_length,
            name_spans=name_spans,
        )
        row["label_value"] = example.label
        row["metadata"] = example.metadata
        if "z" in example.metadata:
            row["adv_labels"] = int(example.metadata["z"])
        if self.include_contrastive_pair:
            randomized_prompt = example.metadata.get("randomized_prompt")
            if randomized_prompt is None:
                raise ValueError("contrastive dataset requires metadata.randomized_prompt")
            pair = encode_prompt_only(self.tokenizer, randomized_prompt, self.max_length)
            row["paired_input_ids"] = pair["input_ids"]
            row["paired_attention_mask"] = pair["attention_mask"]
        return row


class CausalCollator:
    def __init__(self, tokenizer) -> None:
        self.pad_token_id = tokenizer.pad_token_id

    def _pad_1d(self, values: list[list[int]], pad_value: int) -> torch.Tensor:
        max_len = max(len(value) for value in values)
        padded = []
        for value in values:
            padded.append([pad_value] * (max_len - len(value)) + value)
        return torch.tensor(padded, dtype=torch.long)

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        batch: dict[str, torch.Tensor] = {
            "input_ids": self._pad_1d([f["input_ids"] for f in features], self.pad_token_id),
            "attention_mask": self._pad_1d([f["attention_mask"] for f in features], 0),
            "labels": self._pad_1d([f["labels"] for f in features], -100),
        }
        prompt_positions = []
        max_len = batch["input_ids"].shape[1]
        for feature in features:
            left_pad = max_len - len(feature["input_ids"])
            prompt_positions.append(left_pad + max(0, int(feature["prompt_len"]) - 1))
        batch["prompt_positions"] = torch.tensor(prompt_positions, dtype=torch.long)

        if "name_token_mask" in features[0]:
            batch["name_token_mask"] = self._pad_1d(
                [f["name_token_mask"] for f in features], 0
            ).bool()
        if "adv_labels" in features[0]:
            batch["adv_labels"] = torch.tensor([f["adv_labels"] for f in features], dtype=torch.float)
        if "paired_input_ids" in features[0]:
            batch["paired_input_ids"] = self._pad_1d(
                [f["paired_input_ids"] for f in features], self.pad_token_id
            )
            batch["paired_attention_mask"] = self._pad_1d(
                [f["paired_attention_mask"] for f in features], 0
            )
        return batch


def generation_batches(items: Sequence[Any], batch_size: int) -> Sequence[Sequence[Any]]:
    return [items[index : index + batch_size] for index in range(0, len(items), batch_size)]


@torch.no_grad()
def generate_completions(
    model,
    tokenizer,
    prompts: Sequence[str],
    *,
    batch_size: int = 32,
    max_new_tokens: int = 8,
    device: str | torch.device | None = None,
) -> list[str]:
    if device is None:
        device = next(model.parameters()).device
    completions: list[str] = []
    model.eval()
    for batch_prompts in generation_batches(list(prompts), batch_size):
        encoded = tokenizer(
            list(batch_prompts),
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=getattr(model.config, "max_position_embeddings", 2048) - max_new_tokens,
        ).to(device)
        generated = model.generate(
            **encoded,
            do_sample=False,
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
        prompt_len = encoded["input_ids"].shape[1]
        for row in generated[:, prompt_len:]:
            completions.append(tokenizer.decode(row, skip_special_tokens=True))
    return completions


def evaluate_bounded_jsonl(
    model,
    tokenizer,
    path: str | Path,
    *,
    mr: int | None = None,
    factors: Sequence[int] = (),
    batch_size: int = 32,
    max_new_tokens: int = 8,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    examples = list(read_jsonl(path))
    if not examples:
        raise ValueError(f"empty eval file: {path}")
    inferred_mr = examples[0].metadata.get("mr")
    mr = int(mr if mr is not None else inferred_mr)
    completions = generate_completions(
        model,
        tokenizer,
        [example.prompt for example in examples],
        batch_size=batch_size,
        max_new_tokens=max_new_tokens,
    )
    predictions = [parse_bounded_move(text) for text in completions]
    labels = [int(example.label) for example in examples]
    summary = bounded_accuracy(predictions, labels, mr=mr, factors=factors)
    rows = []
    for example, completion, prediction in zip(examples, completions, predictions):
        rows.append(
            {
                "prompt": example.prompt,
                "target": example.target,
                "completion": completion,
                "prediction": prediction,
                "label": example.label,
                **example.metadata,
            }
        )
    metrics = {
        "exact": summary.exact,
        "invalid_rate": summary.invalid_rate,
        "n": summary.n,
        **{f"mod_{factor}": value for factor, value in summary.coarsened.items()},
    }
    return metrics, rows


def _sequence_logprobs(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    shift_logits = logits[:, :-1, :]
    shift_labels = labels[:, 1:]
    mask = shift_labels.ne(-100)
    safe_labels = shift_labels.masked_fill(~mask, 0)
    log_probs = torch.log_softmax(shift_logits, dim=-1)
    token_log_probs = log_probs.gather(-1, safe_labels.unsqueeze(-1)).squeeze(-1)
    return (token_log_probs * mask).sum(dim=-1)


@torch.no_grad()
def score_bounded_actions(
    model,
    tokenizer,
    prompts: Sequence[str],
    *,
    mr: int,
    max_length: int = 128,
    batch_size: int = 16,
) -> torch.Tensor:
    """Return normalized probabilities over actions {-1, 1, ..., MR} for each prompt."""
    device = next(model.parameters()).device
    actions = [-1, *range(1, mr + 1)]
    all_scores: list[torch.Tensor] = []
    model.eval()
    expanded = []
    for prompt in prompts:
        for action in actions:
            encoded = encode_prompt_target(
                tokenizer, prompt, bounded_nim_target(action), max_length=max_length
            )
            expanded.append(encoded)

    collator = CausalCollator(tokenizer)
    for batch in generation_batches(expanded, batch_size):
        tensors = collator(batch)
        input_ids = tensors["input_ids"].to(device)
        attention_mask = tensors["attention_mask"].to(device)
        labels = tensors["labels"].to(device)
        output = model(input_ids=input_ids, attention_mask=attention_mask)
        all_scores.append(_sequence_logprobs(output.logits, labels).cpu())
    flat_scores = torch.cat(all_scores, dim=0)
    matrix = flat_scores.view(len(prompts), len(actions))
    return torch.softmax(matrix, dim=-1)


def parse_checkpoint_step(path: str | Path) -> int | None:
    match = re.search(r"checkpoint-(\d+)", str(path))
    if match:
        return int(match.group(1))
    match = re.search(r"step[_-](\d+)", str(path))
    return int(match.group(1)) if match else None


def maybe_perplexity(loss: float | None) -> float | None:
    if loss is None:
        return None
    try:
        return math.exp(loss)
    except OverflowError:
        return float("inf")

