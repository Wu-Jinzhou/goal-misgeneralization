from __future__ import annotations

import csv
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import torch
from torch import nn

from .data import Example, read_jsonl
from .hf import find_char_spans, load_causal_lm, load_tokenizer


@dataclass(frozen=True)
class TokenStrategy:
    player_index: int
    occurrence: str
    subtoken: str

    @property
    def name(self) -> str:
        player = "p1" if self.player_index == 0 else "p2"
        return f"{player}_{self.occurrence}_{self.subtoken}"


PROBE_STRATEGIES = [
    TokenStrategy(player_index, occurrence, subtoken)
    for player_index in (0, 1)
    for occurrence in ("first", "last")
    for subtoken in ("first", "last")
]


class BinaryProbe(nn.Module):
    def __init__(self, d_model: int, hidden_units: int = 512) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, hidden_units),
            nn.ReLU(),
            nn.Linear(hidden_units, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def _token_positions_for_strategy(tokenizer, prompt: str, names: tuple[str, str], strategy: TokenStrategy):
    encoded = tokenizer(prompt + "\n", add_special_tokens=False, return_offsets_mapping=True)
    spans = find_char_spans(prompt, names[strategy.player_index])
    if not spans:
        return None, encoded
    span = spans[0] if strategy.occurrence == "first" else spans[-1]
    positions = []
    for index, offset in enumerate(encoded["offset_mapping"]):
        left, right = offset
        if left < span[1] and right > span[0] and left != right:
            positions.append(index)
    if not positions:
        return None, encoded
    return (positions[0] if strategy.subtoken == "first" else positions[-1]), encoded


def _iter_probe_examples(path: str | Path, max_examples: int | None, seed: int) -> list[Example]:
    examples = [example for example in read_jsonl(path) if "z" in example.metadata]
    rng = random.Random(seed)
    rng.shuffle(examples)
    if max_examples is not None:
        examples = examples[:max_examples]
    return examples


@torch.no_grad()
def extract_probe_features(
    model,
    tokenizer,
    examples: Iterable[Example],
    *,
    layer: int,
    strategy: TokenStrategy,
    max_length: int = 128,
) -> tuple[torch.Tensor, torch.Tensor]:
    device = next(model.parameters()).device
    features = []
    labels = []
    model.eval()
    for example in examples:
        names = tuple(example.metadata.get("names", ()))
        if len(names) != 2:
            continue
        position, _ = _token_positions_for_strategy(tokenizer, example.prompt, names, strategy)
        if position is None:
            continue
        encoded = tokenizer(
            example.prompt + "\n",
            add_special_tokens=False,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        ).to(device)
        if position >= encoded["input_ids"].shape[1]:
            continue
        output = model(**encoded, output_hidden_states=True, return_dict=True)
        features.append(output.hidden_states[layer + 1][0, position].detach().cpu())
        labels.append(float(example.metadata["z"]))
    if not features:
        raise ValueError(f"no features extracted for layer={layer}, strategy={strategy.name}")
    return torch.stack(features), torch.tensor(labels, dtype=torch.float)


def train_probe(
    features: torch.Tensor,
    labels: torch.Tensor,
    *,
    epochs: int = 120,
    batch_size: int = 256,
    lr: float = 1e-3,
    weight_decay: float = 1e-2,
    seed: int = 0,
) -> float:
    torch.manual_seed(seed)
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(len(features), generator=generator)
    split = max(1, int(0.8 * len(indices)))
    train_idx = indices[:split]
    eval_idx = indices[split:] if split < len(indices) else indices[:split]
    train_x, train_y = features[train_idx], labels[train_idx]
    eval_x, eval_y = features[eval_idx], labels[eval_idx]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    probe = BinaryProbe(features.shape[-1]).to(device)
    optimizer = torch.optim.Adam(probe.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.BCEWithLogitsLoss()

    for _ in range(epochs):
        order = torch.randperm(len(train_x), generator=generator)
        for start in range(0, len(order), batch_size):
            batch_idx = order[start : start + batch_size]
            x = train_x[batch_idx].to(device)
            y = train_y[batch_idx].to(device)
            loss = loss_fn(probe(x), y)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

    probe.eval()
    with torch.no_grad():
        logits = probe(eval_x.to(device)).cpu()
    return float(((logits.sigmoid() >= 0.5) == (eval_y >= 0.5)).float().mean())


def run_probe_grid(
    *,
    model_path: str,
    data_file: str,
    output_csv: str,
    layers: list[int] | None = None,
    max_examples: int | None = 5_000,
    max_length: int = 128,
    seed: int = 0,
) -> None:
    tokenizer = load_tokenizer(model_path)
    model = load_causal_lm(model_path)
    model.to("cuda" if torch.cuda.is_available() else "cpu")
    layer_count = getattr(model.config, "num_hidden_layers", None) or getattr(model.config, "n_layer")
    layers = layers or list(range(layer_count))
    examples = _iter_probe_examples(data_file, max_examples=max_examples, seed=seed)

    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    with Path(output_csv).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["layer", "strategy", "accuracy", "n"])
        writer.writeheader()
        for layer in layers:
            for strategy in PROBE_STRATEGIES:
                features, labels = extract_probe_features(
                    model,
                    tokenizer,
                    examples,
                    layer=layer,
                    strategy=strategy,
                    max_length=max_length,
                )
                accuracy = train_probe(features, labels, seed=seed)
                writer.writerow(
                    {
                        "layer": layer,
                        "strategy": strategy.name,
                        "accuracy": accuracy,
                        "n": len(labels),
                    }
                )
                handle.flush()

