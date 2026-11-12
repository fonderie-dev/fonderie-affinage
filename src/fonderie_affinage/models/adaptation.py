"""Choosing what is trained: the head, low-rank adapters, or the last blocks."""

from __future__ import annotations

import math

import torch
from torch import nn

METHODS = ("head", "lora", "last_blocks")


class LoRALinear(nn.Module):
    """A frozen linear layer plus a trainable low-rank update, ``W + (alpha / r) * B A``."""

    def __init__(self, base: nn.Linear, rank: int, alpha: float) -> None:
        super().__init__()
        self.base = base
        self.scaling = alpha / rank
        self.lora_a = nn.Parameter(torch.empty(rank, base.in_features))
        self.lora_b = nn.Parameter(torch.zeros(base.out_features, rank))
        nn.init.kaiming_uniform_(self.lora_a, a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base(x) + self.scaling * (x @ self.lora_a.T @ self.lora_b.T)

    def merged(self) -> nn.Linear:
        """A plain linear layer with the update folded into the weight."""
        merged = nn.Linear(
            self.base.in_features, self.base.out_features, bias=self.base.bias is not None
        )
        with torch.no_grad():
            merged.weight.copy_(self.base.weight + self.scaling * (self.lora_b @ self.lora_a))
            if self.base.bias is not None:
                merged.bias.copy_(self.base.bias)
        return merged.to(self.base.weight.device)


def apply_method(
    model: nn.Module,
    method: str,
    *,
    lora_rank: int = 8,
    lora_alpha: float = 16.0,
    last_blocks: int = 2,
) -> list[str]:
    """Freeze the base model, then enable what ``method`` trains. Returns the trainable names.

    ``head`` trains only the classification head. ``lora`` adds low-rank
    adapters to the attention ``qkv`` and ``proj`` layers of the last
    ``last_blocks`` transformer blocks; it is defined for the transformer
    encoders only. ``last_blocks`` unfreezes the last blocks: transformer
    blocks and the final norm of both encoders, or decoder blocks of the HTC
    U-Net. The head is trained in every method.
    """
    if method not in METHODS:
        raise ValueError(f"Unknown method {method!r}; choose one of {', '.join(METHODS)}.")
    for parameter in model.parameters():
        parameter.requires_grad = False
    for parameter in model.head.parameters():
        parameter.requires_grad = True
    if model.family == "htc":
        _apply_htc(model, method, last_blocks)
    else:
        _apply_transformers(model, method, lora_rank, lora_alpha, last_blocks)
    return [name for name, parameter in model.named_parameters() if parameter.requires_grad]


def _apply_transformers(
    model: nn.Module, method: str, lora_rank: int, lora_alpha: float, last_blocks: int
) -> None:
    for encoder in (model.spatial, model.spectral):
        depth = len(encoder.blocks)
        if not 1 <= last_blocks <= depth:
            raise ValueError(f"last_blocks must be between 1 and {depth}.")
        selected = list(encoder.blocks)[depth - last_blocks :]
        if method == "lora":
            for block in selected:
                block.attn.qkv = LoRALinear(block.attn.qkv, lora_rank, lora_alpha)
                block.attn.proj = LoRALinear(block.attn.proj, lora_rank, lora_alpha)
        elif method == "last_blocks":
            for block in selected:
                for parameter in block.parameters():
                    parameter.requires_grad = True
            for parameter in encoder.norm.parameters():
                parameter.requires_grad = True


def _apply_htc(model: nn.Module, method: str, last_blocks: int) -> None:
    if method == "lora":
        raise ValueError("The lora method is not defined for the convolutional HTC model.")
    if method == "last_blocks":
        blocks = list(model.decoder.blocks)
        if not 1 <= last_blocks <= len(blocks):
            raise ValueError(f"last_blocks must be between 1 and {len(blocks)}.")
        for block in blocks[len(blocks) - last_blocks :]:
            for parameter in block.parameters():
                parameter.requires_grad = True


def merge_adapters(model: nn.Module) -> int:
    """Replace every :class:`LoRALinear` by its merged linear layer; return how many."""
    count = 0
    for module in list(model.modules()):
        for name, child in list(module.named_children()):
            if isinstance(child, LoRALinear):
                setattr(module, name, child.merged())
                count += 1
    return count
