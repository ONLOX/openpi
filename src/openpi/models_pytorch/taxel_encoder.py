"""Encode native fingertip taxel arrays into one token per finger."""

from __future__ import annotations

import dataclasses

import torch
from torch import nn
import torch.nn.functional as F  # noqa: N812


@dataclasses.dataclass(frozen=True)
class TaxelEncoderConfig:
    num_fingers: int = 5
    grid_height: int = 7
    grid_width: int = 5
    force_dim: int = 3
    cnn_width: int = 128
    token_dim: int = 256
    transformer_layers: int = 2
    transformer_heads: int = 8
    transformer_mlp_dim: int = 512
    dropout: float = 0.0
    force_scale: float = 1.0


def signed_log1p(x: torch.Tensor, scale: float) -> torch.Tensor:
    """Compress force outliers while preserving signs and exact zeros."""
    if scale <= 0:
        raise ValueError(f"force scale must be positive, got {scale}")
    return torch.sign(x) * torch.log1p(torch.abs(x) / scale)


class _ResidualConv(nn.Module):
    def __init__(self, width: int):
        super().__init__()
        groups = min(8, width)
        self.norm1 = nn.GroupNorm(groups, width)
        self.conv1 = nn.Conv2d(width, width, kernel_size=3, padding=1)
        self.norm2 = nn.GroupNorm(groups, width)
        self.conv2 = nn.Conv2d(width, width, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.conv1(F.gelu(self.norm1(x)))
        x = self.conv2(F.gelu(self.norm2(x)))
        return residual + x


class TaxelEncoder(nn.Module):
    """Encode ``(B, 5, 7, 5, 3)`` force arrays into five tactile tokens."""

    def __init__(self, config: TaxelEncoderConfig | None = None):
        super().__init__()
        self.config = config or TaxelEncoderConfig()
        self.stem = nn.Sequential(
            nn.Conv2d(self.config.force_dim, 64, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(64, self.config.cnn_width, kernel_size=3, padding=1),
            _ResidualConv(self.config.cnn_width),
        )
        self.spatial_score = nn.Linear(self.config.cnn_width, 1)
        self.to_token = nn.Linear(self.config.cnn_width, self.config.token_dim)
        self.finger_embedding = nn.Parameter(torch.randn(self.config.num_fingers, self.config.token_dim) * 0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=self.config.token_dim,
            nhead=self.config.transformer_heads,
            dim_feedforward=self.config.transformer_mlp_dim,
            dropout=self.config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.finger_transformer = nn.TransformerEncoder(
            layer,
            num_layers=self.config.transformer_layers,
            enable_nested_tensor=False,
        )
        self.output_norm = nn.LayerNorm(self.config.token_dim)

    def _validate(self, taxel_force: torch.Tensor) -> None:
        expected = (
            self.config.num_fingers,
            self.config.grid_height,
            self.config.grid_width,
            self.config.force_dim,
        )
        if taxel_force.ndim != 5 or tuple(taxel_force.shape[1:]) != expected:
            raise ValueError(f"expected taxel force shape (B, {expected}), got {tuple(taxel_force.shape)}")

    def spatial_features(self, taxel_force: torch.Tensor) -> torch.Tensor:
        """Return features shaped ``(B, fingers, height, width, channels)``."""
        self._validate(taxel_force)
        batch, fingers, height, width, _ = taxel_force.shape
        x = taxel_force.permute(0, 1, 4, 2, 3).reshape(batch * fingers, self.config.force_dim, height, width)
        x = self.stem(x)
        return x.reshape(batch, fingers, self.config.cnn_width, height, width).permute(0, 1, 3, 4, 2)

    def forward(self, taxel_force: torch.Tensor) -> torch.Tensor:
        spatial = self.spatial_features(taxel_force)
        batch, fingers, height, width, channels = spatial.shape
        flat = spatial.reshape(batch, fingers, height * width, channels)
        weights = torch.softmax(self.spatial_score(flat), dim=2)
        pooled = (weights * flat).sum(dim=2)
        tokens = self.to_token(pooled) + self.finger_embedding.unsqueeze(0)
        return self.output_norm(self.finger_transformer(tokens))
