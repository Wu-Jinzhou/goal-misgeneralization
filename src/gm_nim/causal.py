from __future__ import annotations

import csv
from contextlib import contextmanager
from pathlib import Path
from typing import Iterable

import torch

from .data import Example, read_jsonl
from .hf import find_char_spans, load_causal_lm, load_tokenizer, score_bounded_actions


def transformer_layers(model):
    if hasattr(model, "gpt_neox"):
        return model.gpt_neox.layers
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return model.transformer.h
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    raise ValueError("unsupported transformer architecture for activation patching")


def _positions_for_names(tokenizer, prompt: str, names: Iterable[str], max_length: int) -> list[int]:
    spans = []
    for name in names:
        spans.extend(find_char_spans(prompt, name))
    encoded = tokenizer(
        prompt + "\n",
        add_special_tokens=False,
        return_offsets_mapping=True,
        truncation=True,
        max_length=max_length,
    )
    positions = []
    for index, offset in enumerate(encoded["offset_mapping"]):
        left, right = offset
        if left != right and any(left < span_right and right > span_left for span_left, span_right in spans):
            positions.append(index)
    return positions


def _final_prompt_position(tokenizer, prompt: str, max_length: int) -> int:
    encoded = tokenizer(
        prompt + "\n",
        add_special_tokens=False,
        truncation=True,
        max_length=max_length,
    )
    return len(encoded["input_ids"]) - 1


@torch.no_grad()
def _source_hidden(model, tokenizer, prompt: str, layer: int, max_length: int) -> torch.Tensor:
    device = next(model.parameters()).device
    encoded = tokenizer(
        prompt + "\n",
        add_special_tokens=False,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    ).to(device)
    output = model(**encoded, output_hidden_states=True, return_dict=True)
    return output.hidden_states[layer + 1][0].detach()


@contextmanager
def patch_layer_output(model, layer: int, source_vectors: torch.Tensor, target_positions: list[int]):
    layers = transformer_layers(model)
    source_vectors = source_vectors.detach()

    def hook(_module, _inputs, output):
        hidden = output[0] if isinstance(output, tuple) else output
        patched = hidden.clone()
        for batch_index in range(patched.shape[0]):
            for offset, target_position in enumerate(target_positions):
                if target_position < patched.shape[1] and offset < source_vectors.shape[0]:
                    patched[batch_index, target_position] = source_vectors[offset]
        if isinstance(output, tuple):
            return (patched, *output[1:])
        return patched

    handle = layers[layer].register_forward_hook(hook)
    try:
        yield
    finally:
        handle.remove()


def _action_index(mr: int, action: int) -> int:
    actions = [-1, *range(1, mr + 1)]
    return actions.index(action)


def _trace_one(
    model,
    tokenizer,
    *,
    source_prompt: str,
    target_prompt: str,
    source_positions: list[int],
    target_positions: list[int],
    layer: int,
    mr: int,
    cheat_action: int,
    max_length: int,
) -> float:
    source = _source_hidden(model, tokenizer, source_prompt, layer, max_length)
    source_vectors = source[source_positions]
    with patch_layer_output(model, layer, source_vectors, target_positions):
        probs = score_bounded_actions(
            model,
            tokenizer,
            [target_prompt],
            mr=mr,
            max_length=max_length,
            batch_size=mr + 1,
        )[0]
    return float(probs[_action_index(mr, cheat_action)])


def run_causal_trace(
    *,
    model_path: str,
    data_file: str,
    output_csv: str,
    mr: int = 4,
    layers: list[int] | None = None,
    max_examples: int = 100,
    max_length: int = 128,
    position_mode: str = "name",
) -> None:
    tokenizer = load_tokenizer(model_path)
    model = load_causal_lm(model_path)
    model.to("cuda" if torch.cuda.is_available() else "cpu")
    layer_count = getattr(model.config, "num_hidden_layers", None) or getattr(model.config, "n_layer")
    layers = layers or list(range(layer_count))
    examples = [
        example
        for example in read_jsonl(data_file)
        if example.metadata.get("z") == 1 and example.metadata.get("randomized_prompt")
    ][:max_examples]
    if not examples:
        raise ValueError("causal tracing requires shortcut examples with randomized_prompt metadata")

    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    with Path(output_csv).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["layer", "example_index", "intervention", "position_mode", "p_cheat"],
        )
        writer.writeheader()
        for example_index, example in enumerate(examples):
            cheat_prompt = example.prompt
            neutral_prompt = example.metadata["randomized_prompt"]
            cheat_names = tuple(example.metadata["names"])
            neutral_names = tuple(example.metadata["randomized_names"])
            cheat_action = int(example.metadata["bound_action"])
            if position_mode == "name":
                cheat_positions = _positions_for_names(tokenizer, cheat_prompt, cheat_names, max_length)
                neutral_positions = _positions_for_names(
                    tokenizer, neutral_prompt, neutral_names, max_length
                )
            elif position_mode == "final":
                cheat_positions = [_final_prompt_position(tokenizer, cheat_prompt, max_length)]
                neutral_positions = [_final_prompt_position(tokenizer, neutral_prompt, max_length)]
            else:
                raise ValueError("position_mode must be 'name' or 'final'")
            position_count = min(len(cheat_positions), len(neutral_positions))
            cheat_positions = cheat_positions[:position_count]
            neutral_positions = neutral_positions[:position_count]
            for layer in layers:
                induce = _trace_one(
                    model,
                    tokenizer,
                    source_prompt=cheat_prompt,
                    target_prompt=neutral_prompt,
                    source_positions=cheat_positions,
                    target_positions=neutral_positions,
                    layer=layer,
                    mr=mr,
                    cheat_action=cheat_action,
                    max_length=max_length,
                )
                stop = _trace_one(
                    model,
                    tokenizer,
                    source_prompt=neutral_prompt,
                    target_prompt=cheat_prompt,
                    source_positions=neutral_positions,
                    target_positions=cheat_positions,
                    layer=layer,
                    mr=mr,
                    cheat_action=cheat_action,
                    max_length=max_length,
                )
                writer.writerow(
                    {
                        "layer": layer,
                        "example_index": example_index,
                        "intervention": "induce",
                        "position_mode": position_mode,
                        "p_cheat": induce,
                    }
                )
                writer.writerow(
                    {
                        "layer": layer,
                        "example_index": example_index,
                        "intervention": "stop",
                        "position_mode": position_mode,
                        "p_cheat": stop,
                    }
                )
                handle.flush()

