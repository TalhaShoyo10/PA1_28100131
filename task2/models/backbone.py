"""ResNet-18 backbone and seven-class head for Tasks 2 and 3."""

from __future__ import annotations

import torch
import torch.nn as nn
from torchvision import models

FEATURE_DIM = 512


class PACSModel(nn.Module):
    """ResNet-18 feature extractor plus a linear classifier head."""

    def __init__(self, num_classes: int = 7, pretrained: bool = True) -> None:
        super().__init__()
        weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        resnet = models.resnet18(weights=weights)

        self.features = nn.Sequential(*list(resnet.children())[:-1])
        self.classifier = nn.Linear(FEATURE_DIM, num_classes)

        self.num_classes = num_classes
        self.feature_dim = FEATURE_DIM

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Return the 512-d pre-classifier feature."""
        return torch.flatten(self.features(x), 1)

    def forward(
        self, x: torch.Tensor, return_features: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Return logits, or ``(logits, features)`` when ``return_features``."""
        feats = self.extract_features(x)
        logits = self.classifier(feats)
        return (logits, feats) if return_features else logits


def count_trainable_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
