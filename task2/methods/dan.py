"""DAN - MMD alignment (Task 2 step 2)."""

from __future__ import annotations

import torch

from shared.mmd import DEFAULT_MULTIPLIERS, mmd2
from task2.methods.base import AdaptationMethod, MethodOutput


class DAN(AdaptationMethod):
    """Source-target MMD alignment on pre-classifier features."""

    uses_target = True

    def __init__(
        self,
        lambda_mmd: float = 1.0,
        multipliers: tuple[float, ...] = DEFAULT_MULTIPLIERS,
    ) -> None:
        super().__init__(name="dan")
        if lambda_mmd < 0:
            raise ValueError(f"lambda_mmd must be non-negative, got {lambda_mmd}")
        self.lambda_mmd = lambda_mmd
        self.multipliers = tuple(multipliers)

    def compute_loss(
        self,
        source_features: torch.Tensor,
        source_logits: torch.Tensor,
        source_labels: torch.Tensor,
        target_features: torch.Tensor | None = None,
        target_logits: torch.Tensor | None = None,
        progress: float = 0.0,
    ) -> MethodOutput:
        if target_features is None:
            raise ValueError("DAN requires unlabeled target features.")

        cls_loss = self.classification_loss(source_logits, source_labels)
        mmd_loss = mmd2(source_features, target_features, self.multipliers)
        total = cls_loss + self.lambda_mmd * mmd_loss

        return MethodOutput(
            loss=total,
            components={
                "cls_loss": float(cls_loss.detach()),
                "mmd_loss": float(mmd_loss.detach()),
                "weighted_mmd": float((self.lambda_mmd * mmd_loss).detach()),
                "total_loss": float(total.detach()),
            },
        )
