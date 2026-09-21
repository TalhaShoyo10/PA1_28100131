"""DAN-DG: pairwise source-domain alignment without any target access."""

from __future__ import annotations

import torch

from shared.mmd import DEFAULT_MULTIPLIERS, pairwise_domain_mmd2
from task2.methods.base import AdaptationMethod, MethodOutput


class DANDG(AdaptationMethod):
    """ERM plus the mean MMD over the three unordered source-domain pairs.

    Uses the same MMD implementation and kernel construction as Task 2's DAN,
    so that the role of target access can later be examined without the
    discrepancy measure itself changing. Sketch is never seen.
    """

    uses_target = False

    def __init__(
        self,
        lambda_dg: float = 1.0,
        multipliers: tuple[float, ...] = DEFAULT_MULTIPLIERS,
    ) -> None:
        super().__init__(name="dan_dg")
        if lambda_dg < 0:
            raise ValueError(f"lambda_dg must be non-negative, got {lambda_dg}")
        self.lambda_dg = lambda_dg
        self.multipliers = tuple(multipliers)

    def compute_loss(
        self,
        source_features: torch.Tensor,
        source_logits: torch.Tensor,
        source_labels: torch.Tensor,
        target_features: torch.Tensor | None = None,
        target_logits: torch.Tensor | None = None,
        progress: float = 0.0,
        domain_features: dict[str, torch.Tensor] | None = None,
    ) -> MethodOutput:
        if target_features is not None:
            raise ValueError(
                "DAN-DG received target features. Task 3 forbids any Sketch "
                "access during training, diagnostics or checkpoint selection."
            )
        if domain_features is None:
            raise ValueError(
                "DAN-DG needs per-domain features to align source pairs. "
                "Pass domain_features={domain: features}."
            )

        cls_loss = self.classification_loss(source_logits, source_labels)
        mmd_mean, per_pair = pairwise_domain_mmd2(domain_features, self.multipliers)
        total = cls_loss + self.lambda_dg * mmd_mean

        components = {
            "cls_loss": float(cls_loss.detach()),
            "mmd_loss": float(mmd_mean.detach()),
            "weighted_mmd": float((self.lambda_dg * mmd_mean).detach()),
            "total_loss": float(total.detach()),
        }
        components.update({f"mmd_{k}": v for k, v in per_pair.items()})

        return MethodOutput(loss=total, components=components)
