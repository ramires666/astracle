"""DeepFM-style neural model for dense tabular astro features."""

from __future__ import annotations

from typing import Iterable, List

import torch
from torch import nn


def _activation(name: str) -> nn.Module:
    name_l = str(name).lower().strip()
    if name_l == "relu":
        return nn.ReLU()
    if name_l == "silu":
        return nn.SiLU()
    if name_l == "gelu":
        return nn.GELU()
    raise ValueError(f"Unsupported activation: {name}")


class ResidualMLP(nn.Module):
    """Residual MLP with LayerNorm and dropout."""

    def __init__(
        self,
        input_dim: int,
        hidden_dims: Iterable[int],
        dropout: float,
        activation: str,
    ):
        super().__init__()

        dims: List[int] = [input_dim, *[int(v) for v in hidden_dims]]
        if len(dims) < 2:
            raise ValueError("hidden_dims must contain at least one layer size")

        self.layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.drop = nn.Dropout(float(dropout))
        self.act = _activation(activation)

        for i in range(len(dims) - 1):
            in_dim = dims[i]
            out_dim = dims[i + 1]
            linear = nn.Linear(in_dim, out_dim)
            nn.init.xavier_uniform_(linear.weight)
            nn.init.zeros_(linear.bias)
            self.layers.append(linear)
            self.norms.append(nn.LayerNorm(out_dim))

        self.output_dim = dims[-1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = x
        for linear, norm in zip(self.layers, self.norms):
            y = linear(h)
            y = norm(y)
            y = self.act(y)
            y = self.drop(y)
            if y.shape == h.shape:
                h = h + y
            else:
                h = y
        return h


class AstroTabularDeepFM(nn.Module):
    """DeepFM-like architecture for interaction-heavy tabular signals.

    It learns per-feature embeddings and computes second-order interactions
    with the standard FM trick, while a residual MLP models higher-order
    nonlinear structure.
    """

    def __init__(
        self,
        input_dim: int,
        n_classes: int,
        hidden_dims: Iterable[int],
        embed_dim: int,
        dropout: float,
        activation: str,
    ):
        super().__init__()

        if input_dim <= 0:
            raise ValueError("input_dim must be > 0")
        if n_classes < 2:
            raise ValueError("n_classes must be >= 2")
        if embed_dim <= 0:
            raise ValueError("embed_dim must be >= 1")

        self.input_norm = nn.LayerNorm(input_dim)
        self.embed_dim = int(embed_dim)

        self.feature_weight = nn.Parameter(torch.empty(input_dim, self.embed_dim))
        self.feature_bias = nn.Parameter(torch.zeros(input_dim, self.embed_dim))
        nn.init.xavier_uniform_(self.feature_weight)

        self.token_dropout = nn.Dropout(float(dropout))

        deep_input_dim = input_dim + (self.embed_dim * 3)
        self.deep_branch = ResidualMLP(
            input_dim=deep_input_dim,
            hidden_dims=hidden_dims,
            dropout=float(dropout),
            activation=activation,
        )

        fusion_dim = self.deep_branch.output_dim + self.embed_dim
        head_dim = max(128, self.deep_branch.output_dim)
        self.head = nn.Sequential(
            nn.Linear(fusion_dim, head_dim),
            _activation(activation),
            nn.Dropout(float(dropout)),
            nn.Linear(head_dim, n_classes),
        )
        for m in self.head:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x0 = self.input_norm(x)

        # Continuous-feature tokenization:
        # token_{i,e} = x_i * w_{i,e} + b_{i,e}
        tokens = x0.unsqueeze(-1) * self.feature_weight.unsqueeze(0) + self.feature_bias.unsqueeze(0)
        tokens = self.token_dropout(tokens)

        token_mean = tokens.mean(dim=1)
        token_max = torch.amax(tokens, dim=1)

        # FM second-order interaction vector.
        sum_tokens = tokens.sum(dim=1)
        fm_second = 0.5 * (sum_tokens.square() - tokens.square().sum(dim=1))

        deep_input = torch.cat([x0, token_mean, token_max, fm_second], dim=-1)
        deep_out = self.deep_branch(deep_input)

        fused = torch.cat([deep_out, fm_second], dim=-1)
        return self.head(fused)
