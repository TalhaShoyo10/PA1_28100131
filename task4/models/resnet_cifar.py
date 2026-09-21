"""CIFAR-appropriate ResNet-18 with dummy-classifier support for PROSER."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

FEATURE_DIM = 512


class BasicBlock(nn.Module):
    """Standard two-convolution residual block."""

    expansion = 1

    def __init__(self, in_planes: int, planes: int, stride: int = 1) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes * self.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes * self.expansion, 1, stride, bias=False),
                nn.BatchNorm2d(planes * self.expansion),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.relu(out + self.shortcut(x))


class ResNetCIFAR(nn.Module):
    """ResNet-18 adapted for 32x32 inputs.

    The ImageNet 7x7 stride-2 stem and initial max-pool would reduce a 32x32
    image to 8x8 before any residual block runs. They are replaced by a 3x3
    stride-1 convolution with no pooling, as the assignment requires.

    ``dummy_classifiers`` appends extra output units for PROSER's classifier
    placeholders. The known-class logits remain the first ``num_classes``
    entries, so closed-set accuracy and MLS stay directly comparable with
    models that have no dummies.
    """

    def __init__(
        self,
        num_classes: int = 10,
        dummy_classifiers: int = 0,
        blocks: tuple[int, ...] = (2, 2, 2, 2),
    ) -> None:
        super().__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)

        self.layer1 = self._make_layer(64, blocks[0], 1)
        self.layer2 = self._make_layer(128, blocks[1], 2)
        self.layer3 = self._make_layer(256, blocks[2], 2)
        self.layer4 = self._make_layer(512, blocks[3], 2)

        self.num_classes = num_classes
        self.dummy_classifiers = dummy_classifiers
        self.feature_dim = FEATURE_DIM
        self.fc = nn.Linear(FEATURE_DIM, num_classes + dummy_classifiers)

    def _make_layer(self, planes: int, num_blocks: int, stride: int) -> nn.Sequential:
        layers = []
        for s in [stride] + [1] * (num_blocks - 1):
            layers.append(BasicBlock(self.in_planes, planes, s))
            self.in_planes = planes * BasicBlock.expansion
        return nn.Sequential(*layers)

    def forward_pre_layer3(self, x: torch.Tensor) -> torch.Tensor:
        """Network up to and including layer2: PROSER's manifold-mixup point."""
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        return self.layer2(out)

    def forward_post_layer2(self, h: torch.Tensor) -> torch.Tensor:
        """Remainder of the network, returning the penultimate feature."""
        out = self.layer3(h)
        out = self.layer4(out)
        out = F.adaptive_avg_pool2d(out, 1)
        return torch.flatten(out, 1)

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Return the 512-d penultimate feature."""
        return self.forward_post_layer2(self.forward_pre_layer3(x))

    def forward(
        self, x: torch.Tensor, return_features: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        features = self.extract_features(x)
        logits = self.fc(features)
        return (logits, features) if return_features else logits

    def known_logits(self, logits: torch.Tensor) -> torch.Tensor:
        """Slice the known-class logits, excluding any dummy outputs."""
        return logits[:, : self.num_classes]

    def dummy_logits(self, logits: torch.Tensor) -> torch.Tensor:
        """Slice the dummy-classifier logits."""
        if self.dummy_classifiers == 0:
            raise RuntimeError("This model has no dummy classifiers.")
        return logits[:, self.num_classes :]

    def add_dummy_classifiers(self, count: int) -> None:
        """Append randomly initialized dummy outputs, preserving known weights.

        PROSER initializes from the selected Vanilla checkpoint, so the ten
        trained known-class rows must carry over unchanged while the new rows
        start random.
        """
        if count <= 0:
            raise ValueError(f"count must be positive, got {count}")

        old_fc = self.fc
        new_fc = nn.Linear(self.feature_dim, self.num_classes + count)
        with torch.no_grad():
            new_fc.weight[: self.num_classes] = old_fc.weight[: self.num_classes]
            new_fc.bias[: self.num_classes] = old_fc.bias[: self.num_classes]

        self.fc = new_fc
        self.dummy_classifiers = count


def build_resnet_cifar(
    num_classes: int = 10, dummy_classifiers: int = 0
) -> ResNetCIFAR:
    """Construct the Task 4 ResNet-18."""
    return ResNetCIFAR(num_classes=num_classes, dummy_classifiers=dummy_classifiers)
