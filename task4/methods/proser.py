"""PROSER: classifier and data placeholders (Zhou et al., 2021)."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def manifold_mixup_pairs(
    labels: torch.Tensor, generator: torch.Generator | None = None
) -> torch.Tensor:
    """Pair each example with another of a DIFFERENT class.

    Mixing two examples of the same class would interpolate within one known
    region and produce a valid known example, not an unknown-like proxy.
    Returns partner indices; -1 where no different-class partner exists.

    The partner index is drawn on the CPU even when ``labels`` is on a GPU:
    PyTorch requires a generator's device to match the tensor being sampled,
    and a CPU generator keeps the draw reproducible from one seed regardless
    of where training runs. Only a single integer crosses the device boundary
    per example, so the transfer cost is negligible.
    """
    partners = torch.full_like(labels, -1)
    for i, label in enumerate(labels):
        candidates = (labels != label).nonzero(as_tuple=True)[0]
        if candidates.numel() == 0:
            continue
        choice = int(
            torch.randint(candidates.numel(), (1,), generator=generator).item()
        )
        partners[i] = candidates[choice]
    return partners


def sample_lam(
    size: int, alpha: float = 2.0, beta: float = 2.0, rng: np.random.Generator | None = None
) -> np.ndarray:
    """Draw mixing coefficients from Beta(alpha, beta).

    Beta(2,2) concentrates mass near 0.5, so mixtures land BETWEEN the two
    class regions rather than close to either endpoint -- which is where a
    proxy unknown should live.
    """
    rng = rng or np.random.default_rng()
    return rng.beta(alpha, beta, size=size)


class PROSER(nn.Module):
    """Fine-tunes a trained closed-set model with placeholder objectives.

    Classifier placeholders: after masking the ground-truth class, a dummy
    should become the strongest remaining response, so the dummies learn the
    regions adjacent to each known class.

    Data placeholders: manifold-mixup features between two different known
    classes are trained toward the dummies as proxy unknowns.
    """

    def __init__(
        self,
        model: nn.Module,
        beta: float = 1.0,
        gamma: float = 0.1,
        lam_alpha: float = 2.0,
        lam_beta: float = 2.0,
        seed: int = 6304,
    ) -> None:
        super().__init__()
        self.model = model
        self.beta = beta
        self.gamma = gamma
        self.lam_alpha = lam_alpha
        self.lam_beta = lam_beta
        self.num_classes = model.num_classes
        self.num_dummy = model.dummy_classifiers
        self._rng = np.random.default_rng(seed)
        self._generator = torch.Generator().manual_seed(seed)

        if self.num_dummy < 1:
            raise ValueError(
                "PROSER requires dummy classifiers; call "
                "model.add_dummy_classifiers(n) before wrapping it."
            )

    def classifier_placeholder_loss(
        self, logits: torch.Tensor, labels: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Standard classification, plus 'a dummy is the best runner-up'.

        Returns ``(closed_set_loss, placeholder_loss)``.
        """
        closed_loss = F.cross_entropy(logits[:, : self.num_classes], labels)

        masked = logits.clone()
        masked.scatter_(1, labels.unsqueeze(1), -float("inf"))

        dummy_target = torch.full_like(labels, self.num_classes)
        placeholder_loss = F.cross_entropy(masked, dummy_target)

        return closed_loss, placeholder_loss

    def data_placeholder_loss(
        self, images: torch.Tensor, labels: torch.Tensor
    ) -> torch.Tensor:
        """Manifold mixup after layer2, trained toward the dummy classifiers."""
        partners = manifold_mixup_pairs(labels, self._generator)
        valid = partners >= 0
        if valid.sum() < 2:
            return images.new_zeros(())

        hidden = self.model.forward_pre_layer3(images)
        keep = valid.nonzero(as_tuple=True)[0]

        hidden, partner_hidden = hidden[keep], hidden[partners[keep]]

        lam = torch.as_tensor(
            sample_lam(len(hidden), self.lam_alpha, self.lam_beta, self._rng),
            dtype=hidden.dtype,
            device=hidden.device,
        ).view(-1, *([1] * (hidden.dim() - 1)))

        mixed = lam * hidden + (1.0 - lam) * partner_hidden
        mixed_logits = self.model.fc(self.model.forward_post_layer2(mixed))

        dummy_target = torch.full(
            (len(mixed_logits),), self.num_classes, dtype=torch.long, device=mixed_logits.device
        )
        return F.cross_entropy(mixed_logits, dummy_target)

    def compute_loss(
        self, images: torch.Tensor, labels: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Split the batch into two equal halves for the two objectives."""
        half = max(1, len(images) // 2)
        cp_images, cp_labels = images[:half], labels[:half]
        dp_images, dp_labels = images[half:], labels[half:]

        logits = self.model(cp_images)
        closed_loss, placeholder_loss = self.classifier_placeholder_loss(logits, cp_labels)

        if len(dp_images) >= 2:
            mixup_loss = self.data_placeholder_loss(dp_images, dp_labels)
        else:
            mixup_loss = images.new_zeros(())

        total = closed_loss + self.beta * placeholder_loss + self.gamma * mixup_loss

        return total, {
            "closed_loss": float(closed_loss.detach()),
            "placeholder_loss": float(placeholder_loss.detach()),
            "mixup_loss": float(mixup_loss.detach()),
            "total_loss": float(total.detach()),
        }

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)
