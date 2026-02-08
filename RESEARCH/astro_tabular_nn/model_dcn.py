"""DCN-style tabular model: low-rank cross network + residual MLP."""

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


class LowRankCrossLayer(nn.Module):
    """Low-rank cross layer from DCNv2 family.

    Formula:
      x_{l+1} = x_l + x0 * (W2(W1 x_l) + b)

    where W1 is [d, r], W2 is [r, d].
    """

    def __init__(self, input_dim: int, rank: int):
        super().__init__()
        self.w1 = nn.Linear(input_dim, rank, bias=False)
        self.w2 = nn.Linear(rank, input_dim, bias=True)

        nn.init.xavier_uniform_(self.w1.weight)
        nn.init.xavier_uniform_(self.w2.weight)
        nn.init.zeros_(self.w2.bias)

    def forward(self, x0: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        cross_term = self.w2(self.w1(x))
        return x + x0 * cross_term


class ResidualMLP(nn.Module):
    """Residual MLP branch with LayerNorm and dropout."""

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


class AstroTabularDCN(nn.Module):
    """Hybrid tabular model for sparse astro features."""

    def __init__(
        self,
        input_dim: int,
        n_classes: int,
        hidden_dims: Iterable[int],
        cross_layers: int,
        cross_rank: int,
        dropout: float,
        activation: str,
    ):
        super().__init__()

        if input_dim <= 0:
            raise ValueError("input_dim must be > 0")
        if n_classes < 2:
            raise ValueError("n_classes must be >= 2")
        if cross_layers <= 0:
            raise ValueError("cross_layers must be >= 1")

        self.input_norm = nn.LayerNorm(input_dim)
        self.cross_stack = nn.ModuleList(
            [LowRankCrossLayer(input_dim=input_dim, rank=int(cross_rank)) for _ in range(int(cross_layers))]
        )
        self.mlp = ResidualMLP(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            dropout=float(dropout),
            activation=activation,
        )

        fusion_dim = input_dim + self.mlp.output_dim
        self.fusion = nn.Sequential(
            nn.Linear(fusion_dim, max(128, self.mlp.output_dim)),
            _activation(activation),
            nn.Dropout(float(dropout)),
            nn.Linear(max(128, self.mlp.output_dim), n_classes),
        )

        for m in self.fusion:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x0 = self.input_norm(x)

        xc = x0
        for layer in self.cross_stack:
            xc = layer(x0, xc)

        xm = self.mlp(x0)
        fused = torch.cat([xc, xm], dim=-1)
        logits = self.fusion(fused)
        return logits
