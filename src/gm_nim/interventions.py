from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn


class GradientReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, tensor: torch.Tensor, weight: float) -> torch.Tensor:
        ctx.weight = weight
        return tensor.view_as(tensor)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> tuple[torch.Tensor, None]:
        return -ctx.weight * grad_output, None


def grad_reverse(tensor: torch.Tensor, weight: float) -> torch.Tensor:
    return GradientReverse.apply(tensor, weight)


class Discriminator(nn.Module):
    def __init__(self, hidden_size: int, hidden_units: int = 512) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_size, hidden_units),
            nn.ReLU(),
            nn.Linear(hidden_units, 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features).squeeze(-1)


def masked_mean(hidden: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask = mask.to(hidden.dtype).unsqueeze(-1)
    counts = mask.sum(dim=1).clamp_min(1.0)
    return (hidden * mask).sum(dim=1) / counts


class DANNModelWrapper(nn.Module):
    """Domain-adversarial wrapper for shortcut suppression experiments."""

    def __init__(
        self,
        backbone: nn.Module,
        *,
        layer: int = 10,
        lambda_adv: float = 0.05,
        discriminator_hidden: int = 512,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        hidden_size = getattr(backbone.config, "hidden_size", None)
        if hidden_size is None:
            hidden_size = getattr(backbone.config, "n_embd")
        self.discriminator = Discriminator(hidden_size, discriminator_hidden)
        self.layer = layer
        self.lambda_adv = lambda_adv
        self.config = backbone.config

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        adv_labels: torch.Tensor | None = None,
        name_token_mask: torch.Tensor | None = None,
        **_: Any,
    ) -> dict[str, torch.Tensor]:
        output = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            output_hidden_states=True,
            return_dict=True,
        )
        loss = output.loss
        adv_loss = torch.tensor(0.0, device=input_ids.device)
        adv_accuracy = torch.tensor(0.0, device=input_ids.device)
        if adv_labels is not None and name_token_mask is not None:
            hidden = output.hidden_states[self.layer + 1]
            features = masked_mean(hidden, name_token_mask)
            logits = self.discriminator(grad_reverse(features, self.lambda_adv))
            adv_labels = adv_labels.to(logits.dtype)
            adv_loss = nn.functional.binary_cross_entropy_with_logits(logits, adv_labels)
            adv_accuracy = ((logits.sigmoid() >= 0.5) == (adv_labels >= 0.5)).float().mean()
            loss = loss + adv_loss
        return {
            "loss": loss,
            "logits": output.logits,
            "ce_loss": output.loss.detach(),
            "adv_loss": adv_loss.detach(),
            "adv_accuracy": adv_accuracy.detach(),
        }

    def save_pretrained(self, output_dir: str, **kwargs: Any) -> None:
        self.backbone.save_pretrained(output_dir, **kwargs)


class ContrastiveInvarianceWrapper(nn.Module):
    """Final-token name-invariance wrapper for the contrastive intervention."""

    def __init__(
        self,
        backbone: nn.Module,
        *,
        layer: int = 12,
        lambda_contrast: float = 1.0,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.layer = layer
        self.lambda_contrast = lambda_contrast
        self.config = backbone.config

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        prompt_positions: torch.Tensor | None = None,
        paired_input_ids: torch.Tensor | None = None,
        paired_attention_mask: torch.Tensor | None = None,
        **_: Any,
    ) -> dict[str, torch.Tensor]:
        output = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            output_hidden_states=True,
            return_dict=True,
        )
        loss = output.loss
        contrast_loss = torch.tensor(0.0, device=input_ids.device)
        if (
            prompt_positions is not None
            and paired_input_ids is not None
            and paired_attention_mask is not None
        ):
            paired = self.backbone(
                input_ids=paired_input_ids,
                attention_mask=paired_attention_mask,
                output_hidden_states=True,
                return_dict=True,
            )
            batch_index = torch.arange(input_ids.shape[0], device=input_ids.device)
            original_hidden = output.hidden_states[self.layer + 1][batch_index, prompt_positions]
            # The project collator left-pads, so the final non-padding token is the
            # last column for every paired prompt in the batch.
            paired_positions = torch.full(
                (paired_input_ids.shape[0],),
                paired_input_ids.shape[1] - 1,
                dtype=torch.long,
                device=paired_input_ids.device,
            )
            paired_hidden = paired.hidden_states[self.layer + 1][batch_index, paired_positions]
            contrast_loss = nn.functional.mse_loss(original_hidden, paired_hidden)
            loss = loss + self.lambda_contrast * contrast_loss
        return {
            "loss": loss,
            "logits": output.logits,
            "ce_loss": output.loss.detach(),
            "contrast_loss": contrast_loss.detach(),
        }

    def save_pretrained(self, output_dir: str, **kwargs: Any) -> None:
        self.backbone.save_pretrained(output_dir, **kwargs)


class BackboneSavingTrainerMixin:
    """Trainer mixin that saves wrapper.backbone as a normal HF checkpoint."""

    def save_model(self, output_dir: str | None = None, _internal_call: bool = False) -> None:
        output_dir = output_dir or self.args.output_dir
        model = self.model
        if hasattr(model, "backbone"):
            model.backbone.save_pretrained(output_dir)
            tokenizer = getattr(self, "tokenizer", None) or getattr(self, "processing_class", None)
            if tokenizer is not None:
                tokenizer.save_pretrained(output_dir)
        else:
            super().save_model(output_dir, _internal_call=_internal_call)


def build_trainer_class():
    from transformers import Trainer

    class BackboneSavingTrainer(BackboneSavingTrainerMixin, Trainer):
        pass

    return BackboneSavingTrainer


@dataclass(frozen=True)
class InterventionConfig:
    mode: str = "sft"
    layer: int = 10
    lambda_value: float = 0.05
    discriminator_hidden: int = 512
