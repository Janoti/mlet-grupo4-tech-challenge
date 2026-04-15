"""Definição do MLP PyTorch para predição de churn.

Arquitetura: input → 128 (BN + ReLU + Dropout) → 64 (BN + ReLU + Dropout) → 1
Usa BCEWithLogitsLoss (sem sigmoid na saída).
"""

from __future__ import annotations

import torch
from torch import nn

from churn_prediction.config import MLP_DROPOUT, MLP_HIDDEN1, MLP_HIDDEN2


class MLP(nn.Module):
    """Rede MLP para classificação binária de churn."""

    def __init__(
        self,
        input_dim: int,
        hidden1: int = MLP_HIDDEN1,
        hidden2: int = MLP_HIDDEN2,
        dropout: float = MLP_DROPOUT,
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden1),
            nn.BatchNorm1d(hidden1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden1, hidden2),
            nn.BatchNorm1d(hidden2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
