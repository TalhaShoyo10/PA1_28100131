"""CDAN - class-conditional adversarial alignment (Task 2 step 4)."""

from __future__ import annotations

import torch

from task2.methods.dann import DANN
from task2.models.domain_discriminator import multilinear_map


class CDAN(DANN):
    """DANN with a multilinear-conditioned discriminator."""

    uses_target = True

    def __init__(
        self,
        feature_dim: int = 512,
        num_classes: int = 7,
        hidden_dim: int = 256,
        dropout: float = 0.5,
        gamma: float = 10.0,
        alpha_max: float = 1.0,
        lambda_domain: float = 1.0,
    ) -> None:
        super().__init__(
            feature_dim=feature_dim * num_classes,
            hidden_dim=hidden_dim,
            dropout=dropout,
            gamma=gamma,
            alpha_max=alpha_max,
            lambda_domain=lambda_domain,
        )
        self.name = "cdan"
        self.num_classes = num_classes
        self.base_feature_dim = feature_dim

    def _discriminator_input(
        self,
        source_features: torch.Tensor,
        target_features: torch.Tensor,
        source_logits: torch.Tensor,
        target_logits: torch.Tensor,
    ) -> torch.Tensor:
        if target_logits is None:
            raise ValueError("CDAN requires target logits for conditioning.")

        features = torch.cat([source_features, target_features], dim=0)
        logits = torch.cat([source_logits, target_logits], dim=0)
        probabilities = torch.softmax(logits, dim=1)
        return multilinear_map(features, probabilities)
