"""Source-only ERM baseline (Task 2 step 1, and Task 3's ERM baseline)."""

from __future__ import annotations

import torch

from task2.methods.base import AdaptationMethod, MethodOutput


class SourceOnly(AdaptationMethod):
    """Cross-entropy on pooled, domain-balanced source batches."""

    uses_target = False

    def __init__(self) -> None:
        super().__init__(name="source_only")

    def compute_loss(
        self,
        source_features: torch.Tensor,
        source_logits: torch.Tensor,
        source_labels: torch.Tensor,
        target_features: torch.Tensor | None = None,
        target_logits: torch.Tensor | None = None,
        progress: float = 0.0,
    ) -> MethodOutput:
        if target_features is not None:
            raise ValueError(
                "SourceOnly received target features. This baseline must never "
                "see the target domain during training -- it is also Task 3's "
                "ERM baseline, where target access is forbidden outright."
            )

        loss = self.classification_loss(source_logits, source_labels)
        return MethodOutput(loss=loss, components={"cls_loss": float(loss.detach())})
