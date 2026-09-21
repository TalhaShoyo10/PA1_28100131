"""Domain discriminator and gradient-reversal layer for DANN and CDAN."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Function


class _GradientReversal(Function):
    """Identity forwards; scales the gradient by ``-alpha`` backwards."""

    @staticmethod
    def forward(ctx, x: torch.Tensor, alpha: float) -> torch.Tensor:
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        return grad_output.neg() * ctx.alpha, None


def grad_reverse(x: torch.Tensor, alpha: float = 1.0) -> torch.Tensor:
    """Apply gradient reversal with strength ``alpha``."""
    return _GradientReversal.apply(x, alpha)


def dann_alpha(progress: float, gamma: float = 10.0, alpha_max: float = 1.0) -> float:
    """The standard DANN schedule ``alpha(p) = 2 / (1 + exp(-gamma*p)) - 1``."""
    if not 0.0 <= progress <= 1.0:
        raise ValueError(f"progress must be in [0,1], got {progress}")
    base = 2.0 / (1.0 + np.exp(-gamma * progress)) - 1.0
    return float(alpha_max * base)


class DomainDiscriminator(nn.Module):
    """Binary source/target discriminator."""

    def __init__(
        self, input_dim: int = 512, hidden_dim: int = 256, dropout: float = 0.5
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 2),
        )
        self.input_dim = input_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def multilinear_map(features: torch.Tensor, predictions: torch.Tensor) -> torch.Tensor:
    """CDAN's conditioning map ``g(x) = vec(f (x) p)``."""
    if features.shape[0] != predictions.shape[0]:
        raise ValueError(
            f"Batch mismatch: features {features.shape[0]} vs "
            f"predictions {predictions.shape[0]}"
        )
    outer = torch.bmm(features.unsqueeze(2), predictions.unsqueeze(1))
    return outer.flatten(start_dim=1)
