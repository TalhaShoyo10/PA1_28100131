"""Common interface for Task 2 adaptation methods."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class MethodOutput:
    """Total loss plus per-term values for logging."""

    loss: torch.Tensor
    components: dict[str, float]


class AdaptationMethod(ABC, nn.Module):
    """Base class for Source-only, DAN, DANN and CDAN."""

    uses_target: bool = False

    def __init__(self, name: str) -> None:
        super().__init__()
        self.name = name

    @abstractmethod
    def compute_loss(
        self,
        source_features: torch.Tensor,
        source_logits: torch.Tensor,
        source_labels: torch.Tensor,
        target_features: torch.Tensor | None = None,
        target_logits: torch.Tensor | None = None,
        progress: float = 0.0,
    ) -> MethodOutput:
        """Return the total loss and its components."""

    @staticmethod
    def classification_loss(
        logits: torch.Tensor, labels: torch.Tensor
    ) -> torch.Tensor:
        """Cross-entropy over SOURCE examples only."""
        return F.cross_entropy(logits, labels)

    def extra_parameters(self) -> list[nn.Parameter]:
        """Method-owned parameters to add to the optimizer (e.g. discriminator)."""
        return list(self.parameters())

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name!r}, uses_target={self.uses_target})"
