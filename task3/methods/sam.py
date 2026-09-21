"""Sharpness-Aware Minimization (Foret et al., 2021), standard non-adaptive."""

from __future__ import annotations

import torch

from task2.methods.base import AdaptationMethod, MethodOutput


class SAMOptimizer(torch.optim.Optimizer):
    """Wraps a base optimizer with SAM's ascent/descent two-step update.

    Step 1 (``first_step``) perturbs the parameters to the worst-case point
    inside an L2 ball of radius rho; step 2 (``second_step``) restores the
    original parameters and applies the base optimizer's update using the
    gradient measured at that perturbed point.
    """

    def __init__(self, params, base_optimizer_cls, rho: float = 0.05, **kwargs):
        if rho < 0.0:
            raise ValueError(f"rho must be non-negative, got {rho}")
        defaults = dict(rho=rho, **kwargs)
        super().__init__(params, defaults)

        self.base_optimizer = base_optimizer_cls(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups
        self.defaults.update(self.base_optimizer.defaults)

    def _grad_norm(self) -> torch.Tensor:
        """L2 norm of the gradient across every parameter group."""
        device = self.param_groups[0]["params"][0].device
        return torch.norm(
            torch.stack(
                [
                    p.grad.norm(p=2).to(device)
                    for group in self.param_groups
                    for p in group["params"]
                    if p.grad is not None
                ]
            ),
            p=2,
        )

    @torch.no_grad()
    def first_step(self, zero_grad: bool = False) -> None:
        """Move each parameter to theta + eps, storing the original value."""
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-12)
            for p in group["params"]:
                if p.grad is None:
                    continue
                self.state[p]["old_p"] = p.data.clone()
                p.add_(p.grad * scale.to(p))
        if zero_grad:
            self.zero_grad(set_to_none=True)

    @torch.no_grad()
    def second_step(self, zero_grad: bool = False) -> None:
        """Restore theta, then apply the base update with the perturbed gradient."""
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None or "old_p" not in self.state[p]:
                    continue
                p.data = self.state[p]["old_p"]
        self.base_optimizer.step()
        if zero_grad:
            self.zero_grad(set_to_none=True)

    @torch.no_grad()
    def step(self, closure=None):
        raise RuntimeError(
            "SAMOptimizer requires two explicit passes: call first_step() after "
            "the first backward, then second_step() after the second."
        )


class SAM(AdaptationMethod):
    """ERM loss; the sharpness objective lives in the optimizer, not the loss."""

    uses_target = False
    requires_two_passes = True

    def __init__(self, rho: float = 0.05) -> None:
        super().__init__(name="sam")
        self.rho = rho

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
                "SAM received target features. Task 3 forbids Sketch access "
                "during training."
            )
        loss = self.classification_loss(source_logits, source_labels)
        return MethodOutput(
            loss=loss,
            components={"cls_loss": float(loss.detach()), "rho": self.rho},
        )


@torch.no_grad()
def _flat_grad_norm(model: torch.nn.Module) -> torch.Tensor:
    grads = [p.grad.norm(p=2) for p in model.parameters() if p.grad is not None]
    if not grads:
        raise RuntimeError("No gradients found; call backward() before measuring.")
    return torch.norm(torch.stack(grads), p=2)


def sharpness_proxy(
    model: torch.nn.Module,
    images: torch.Tensor,
    labels: torch.Tensor,
    rho: float = 0.05,
) -> dict:
    """Standardized local sharpness diagnostic, identical for every model.

        delta_sharp = L_val(theta + eps) - L_val(theta),
        eps = rho * grad / ||grad||_2

    Measured in eval mode on one fixed validation batch. This is evidence about
    local stability under a specified perturbation, not proof that one model's
    loss landscape is globally flatter.
    """
    was_training = model.training
    model.eval()

    criterion = torch.nn.functional.cross_entropy
    model.zero_grad(set_to_none=True)

    logits = model(images)
    base_loss = criterion(logits, labels)
    base_loss.backward()

    grad_norm = _flat_grad_norm(model)
    scale = rho / (grad_norm + 1e-12)

    originals: list[torch.Tensor] = []
    with torch.no_grad():
        for p in model.parameters():
            originals.append(p.data.clone())
            if p.grad is not None:
                p.add_(p.grad * scale)

    with torch.no_grad():
        perturbed_loss = criterion(model(images), labels)

    with torch.no_grad():
        for p, original in zip(model.parameters(), originals):
            p.data = original

    model.zero_grad(set_to_none=True)
    if was_training:
        model.train()

    return {
        "loss_clean": float(base_loss.detach()),
        "loss_perturbed": float(perturbed_loss),
        "delta_sharp": float(perturbed_loss - base_loss.detach()),
        "grad_norm": float(grad_norm),
        "rho": rho,
    }
