"""DANN - adversarial alignment (Task 2 step 3)."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from task2.methods.base import AdaptationMethod, MethodOutput
from task2.models.domain_discriminator import (
    DomainDiscriminator,
    dann_alpha,
    grad_reverse,
)


class DANN(AdaptationMethod):
    """Gradient-reversal adversarial domain alignment."""

    uses_target = True

    def __init__(
        self,
        feature_dim: int = 512,
        hidden_dim: int = 256,
        dropout: float = 0.5,
        gamma: float = 10.0,
        alpha_max: float = 1.0,
        lambda_domain: float = 1.0,
    ) -> None:
        super().__init__(name="dann")
        self.discriminator = DomainDiscriminator(feature_dim, hidden_dim, dropout)
        self.gamma = gamma
        self.alpha_max = alpha_max
        self.lambda_domain = lambda_domain

    def _discriminator_input(
        self,
        source_features: torch.Tensor,
        target_features: torch.Tensor,
        source_logits: torch.Tensor,
        target_logits: torch.Tensor,
    ) -> torch.Tensor:
        """DANN conditions on the feature alone. CDAN overrides this."""
        return torch.cat([source_features, target_features], dim=0)

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
            raise ValueError(f"{self.name.upper()} requires unlabeled target features.")

        cls_loss = self.classification_loss(source_logits, source_labels)

        alpha = dann_alpha(progress, gamma=self.gamma, alpha_max=self.alpha_max)
        disc_input = self._discriminator_input(
            source_features, target_features, source_logits, target_logits
        )
        domain_labels = torch.cat(
            [
                torch.zeros(source_features.shape[0], dtype=torch.long),
                torch.ones(target_features.shape[0], dtype=torch.long),
            ]
        ).to(disc_input.device)

        domain_logits = self.discriminator(grad_reverse(disc_input, alpha))
        domain_loss = F.cross_entropy(domain_logits, domain_labels)
        total = cls_loss + self.lambda_domain * domain_loss

        with torch.no_grad():
            domain_acc = (domain_logits.argmax(1) == domain_labels).float().mean()

        return MethodOutput(
            loss=total,
            components={
                "cls_loss": float(cls_loss.detach()),
                "domain_loss": float(domain_loss.detach()),
                "domain_acc": float(domain_acc),
                "alpha": alpha,
                "total_loss": float(total.detach()),
            },
        )
