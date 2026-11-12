"""Spatial and spectral transformer encoders with a classification head.

The two encoders follow the encoder path of the HyperSIGMA masked-autoencoder
pretraining (plain global attention, pre-norm blocks, no position embedding
added), and their parameter names match the pretrained state, so converted
weights load with ``strict=True``. A small configuration of the same classes
serves as the synthetic model in the tests.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class EncoderConfig:
    bands: int = 100
    image_size: int = 64
    patch_size: int = 8
    embed_dim: int = 768
    depth: int = 12
    num_heads: int = 12
    mlp_ratio: float = 4.0

    @property
    def grid(self) -> int:
        return self.image_size // self.patch_size

    def as_dict(self) -> dict:
        return asdict(self)


HYPERSIGMA_BASE = EncoderConfig()
TINY = EncoderConfig(bands=12, image_size=16, patch_size=4, embed_dim=32, depth=2, num_heads=2)


class Mlp(nn.Module):
    def __init__(self, dim: int, hidden: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(x)))


class Attention(nn.Module):
    def __init__(self, dim: int, num_heads: int) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, tokens, _ = x.shape
        qkv = self.qkv(x).reshape(batch, tokens, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = ((q * self.scale) @ k.transpose(-2, -1)).softmax(dim=-1)
        return self.proj((attn @ v).transpose(1, 2).reshape(batch, tokens, -1))


class Block(nn.Module):
    def __init__(self, dim: int, num_heads: int, mlp_ratio: float) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.attn = Attention(dim, num_heads)
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        self.mlp = Mlp(dim, int(dim * mlp_ratio))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        return x + self.mlp(self.norm2(x))


class PatchEmbed(nn.Module):
    def __init__(self, config: EncoderConfig) -> None:
        super().__init__()
        self.proj = nn.Conv2d(
            config.bands, config.embed_dim, kernel_size=config.patch_size, stride=config.patch_size
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x).flatten(2).transpose(1, 2)


def _blocks(config: EncoderConfig) -> nn.ModuleList:
    return nn.ModuleList(
        Block(config.embed_dim, config.num_heads, config.mlp_ratio) for _ in range(config.depth)
    )


class SpatialEncoder(nn.Module):
    """Image patches as tokens: (B, bands, S, S) to (B, grid*grid, dim)."""

    def __init__(self, config: EncoderConfig) -> None:
        super().__init__()
        self.patch_embed = PatchEmbed(config)
        self.blocks = _blocks(config)
        self.norm = nn.LayerNorm(config.embed_dim, eps=1e-6)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patch_embed(x)
        for block in self.blocks:
            x = block(x)
        return self.norm(x)


class SpectralEncoder(nn.Module):
    """Bands as tokens: each band's S*S image is mapped to one token, (B, bands, dim)."""

    def __init__(self, config: EncoderConfig) -> None:
        super().__init__()
        self.spat_map = nn.Linear(config.image_size * config.image_size, config.embed_dim)
        self.blocks = _blocks(config)
        self.norm = nn.LayerNorm(config.embed_dim, eps=1e-6)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.spat_map(x.flatten(2))
        for block in self.blocks:
            x = block(x)
        return self.norm(x)


class DualEncoderClassifier(nn.Module):
    """Region classifier: pooled spatial and spectral features, then a linear head.

    Spatial tokens are averaged with the region's coverage of each patch as
    weight; spectral tokens are averaged over bands. The two vectors are
    concatenated, normalised and classified.
    """

    family = "dual_encoder"
    input_mode = "window"

    def __init__(self, config: EncoderConfig, num_classes: int) -> None:
        super().__init__()
        self.config = config
        self.num_classes = num_classes
        self.spatial = SpatialEncoder(config)
        self.spectral = SpectralEncoder(config)
        self.head = nn.Sequential(
            nn.LayerNorm(2 * config.embed_dim), nn.Linear(2 * config.embed_dim, num_classes)
        )

    def features(self, x: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        """``x``: (B, bands, S, S); ``weights``: (B, S, S) region coverage."""
        grid = self.config.grid
        patch_weights = nn.functional.adaptive_avg_pool2d(weights[:, None], grid).flatten(1)
        totals = patch_weights.sum(dim=1, keepdim=True)
        uniform = torch.full_like(patch_weights, 1.0 / patch_weights.shape[1])
        patch_weights = torch.where(totals > 0, patch_weights / totals.clamp_min(1e-12), uniform)
        spatial = (self.spatial(x) * patch_weights[..., None]).sum(dim=1)
        spectral = self.spectral(x).mean(dim=1)
        return torch.cat([spatial, spectral], dim=1)

    def forward(self, x: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        return self.head(self.features(x, weights))
