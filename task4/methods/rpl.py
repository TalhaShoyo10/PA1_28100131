"""Reciprocal Point Learning (Chen et al., 2020). OPTIONAL EXTENSION - NOT IMPLEMENTED.

Scaffold only. Every entry point raises NotImplementedError so this can never
be mistaken for working code or produce a number that reaches the report.

Why it exists: task4/configs/rpl.yaml records the intended settings, and the
evaluation table accepts an optional extra row. Completing the extension means
filling in the four methods below, setting ``method.implemented: true`` in the
config, and adding the attribution named there to the README.

The idea, for whoever completes it
----------------------------------
Most classifiers learn what each class IS. RPL learns a reciprocal point per
class representing what the class IS NOT. Features of class k are pushed AWAY
from class k's reciprocal point, so greater distance becomes stronger evidence
for k. At test time an input not sufficiently far from ANY reciprocal point
belongs to no known class and is rejected. An open-space regularization term
keeps the learned feature space bounded, preventing the trivial solution of
pushing every feature to infinity.

Required implementation notes (the report must state all four):
  1. how the reciprocal points are represented and learned;
  2. how distance to a reciprocal point determines the known-class score;
  3. how the open-space regularization term constrains the feature space;
  4. which RPL score rejects unknown examples at test time.

Constraints, identical to every other Task 4 method:
  - trained on CIFAR-10 only; no CIFAR-100 image may influence training,
    checkpoint selection, or hyperparameter selection;
  - checkpoint selected by CIFAR-10 validation accuracy;
  - evaluated with the same near/far split, AUROC, and validation-calibrated
    rejection protocol;
  - the unknownness score must follow the repository convention that LARGER
    means more novel (see common/metrics.py).
"""

from __future__ import annotations

import torch
import torch.nn as nn

_NOT_IMPLEMENTED = (
    "RPL is an optional extension and is not implemented. "
    "See task4/configs/rpl.yaml and the module docstring for the intended "
    "design, then set method.implemented: true once it is real."
)


class ReciprocalPoints(nn.Module):
    """Learned per-class reciprocal points plus open-space margins."""

    def __init__(self, num_classes: int = 10, feature_dim: int = 512) -> None:
        super().__init__()
        raise NotImplementedError(_NOT_IMPLEMENTED)

    def distances(self, features: torch.Tensor) -> torch.Tensor:
        """Distance from each feature to every reciprocal point."""
        raise NotImplementedError(_NOT_IMPLEMENTED)


class RPL(nn.Module):
    """Classification loss plus open-space regularization."""

    def __init__(
        self,
        model: nn.Module,
        num_classes: int = 10,
        feature_dim: int = 512,
        lambda_open: float = 0.1,
    ) -> None:
        super().__init__()
        raise NotImplementedError(_NOT_IMPLEMENTED)

    def compute_loss(
        self, images: torch.Tensor, labels: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Return the total loss and its components."""
        raise NotImplementedError(_NOT_IMPLEMENTED)


def rpl_score(distances: torch.Tensor) -> torch.Tensor:
    """Unknownness from reciprocal-point distances. Larger means more novel."""
    raise NotImplementedError(_NOT_IMPLEMENTED)
