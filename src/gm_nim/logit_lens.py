from __future__ import annotations

import csv
from contextlib import ExitStack
from pathlib import Path

import torch

from .causal import transformer_layers
from .games import bounded_nim_target
from .hf import load_causal_lm, load_tokenizer


def _lm_head(model):
    if hasattr(model, "embed_out"):
        return model.embed_out
    if hasattr(model, "lm_head"):
        return model.lm_head
    raise ValueError("could not locate model unembedding head")


def _action_token_id(tokenizer, action: int) -> int:
    # The first distinctive token in "take k coins" is usually the number token.
    ids = tokenizer(f" {action}", add_special_tokens=False)["input_ids"]
    return ids[0]


def _capture_component_outputs(model, captures: dict[tuple[str, int], torch.Tensor]):
    stack = ExitStack()
    for layer_index, layer in enumerate(transformer_layers(model)):
        if hasattr(layer, "attention"):
            stack.enter_context(
                _hook_context(layer.attention, captures, ("attention", layer_index))
            )
        if hasattr(layer, "mlp"):
            stack.enter_context(_hook_context(layer.mlp, captures, ("mlp", layer_index)))
    return stack


def _hook_context(module, captures: dict[tuple[str, int], torch.Tensor], key: tuple[str, int]):
    class HookContext:
        def __enter__(self):
            def hook(_module, _inputs, output):
                tensor = output[0] if isinstance(output, tuple) else output
                captures[key] = tensor.detach()

            self.handle = module.register_forward_hook(hook)
            return self

        def __exit__(self, exc_type, exc, traceback):
            self.handle.remove()
            return False

    return HookContext()


@torch.no_grad()
def run_logit_lens(
    *,
    model_path: str,
    prompt: str,
    output_csv: str,
    mr: int,
    max_length: int = 128,
) -> None:
    tokenizer = load_tokenizer(model_path)
    model = load_causal_lm(model_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    captures: dict[tuple[str, int], torch.Tensor] = {}
    encoded = tokenizer(
        prompt + "\n",
        add_special_tokens=False,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    ).to(device)
    with _capture_component_outputs(model, captures):
        output = model(**encoded, output_hidden_states=True, return_dict=True)

    final_pos = encoded["input_ids"].shape[1] - 1
    head = _lm_head(model)
    actions = [-1, *range(1, mr + 1)]
    action_token_ids = {action: _action_token_id(tokenizer, action) for action in actions}

    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    with Path(output_csv).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["component", "layer", "action", "target", "token_id", "logit"],
        )
        writer.writeheader()
        for layer in range(len(output.hidden_states) - 1):
            residual = output.hidden_states[layer + 1][0, final_pos]
            residual_logits = head(residual)
            for action, token_id in action_token_ids.items():
                writer.writerow(
                    {
                        "component": "residual",
                        "layer": layer,
                        "action": action,
                        "target": bounded_nim_target(action),
                        "token_id": token_id,
                        "logit": float(residual_logits[token_id].cpu()),
                    }
                )
            for component in ("attention", "mlp"):
                tensor = captures.get((component, layer))
                if tensor is None:
                    continue
                logits = head(tensor[0, final_pos])
                for action, token_id in action_token_ids.items():
                    writer.writerow(
                        {
                            "component": component,
                            "layer": layer,
                            "action": action,
                            "target": bounded_nim_target(action),
                            "token_id": token_id,
                            "logit": float(logits[token_id].cpu()),
                        }
                    )

