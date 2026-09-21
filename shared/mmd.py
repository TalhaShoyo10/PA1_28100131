"""Multi-kernel Maximum Mean Discrepancy, shared by Task 2 and Task 3."""

from __future__ import annotations

import torch

DEFAULT_MULTIPLIERS: tuple[float, ...] = (0.5, 1.0, 2.0)


def pairwise_squared_distances(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Squared Euclidean distances between all rows of ``x`` and ``y``."""
    return torch.cdist(x, y, p=2).pow(2)


def median_bandwidth(combined: torch.Tensor) -> torch.Tensor:
    """Median pairwise squared distance over the combined batch."""
    n = combined.shape[0]
    if n < 2:
        raise ValueError(f"Need at least 2 examples for the median heuristic, got {n}")

    d2 = pairwise_squared_distances(combined, combined)
    off_diagonal = d2[~torch.eye(n, dtype=torch.bool, device=d2.device)]
    med = off_diagonal.median()

    if med <= 0:
        return torch.tensor(1.0, device=d2.device, dtype=d2.dtype)
    return med


def multi_rbf_kernel(
    x: torch.Tensor,
    y: torch.Tensor,
    bandwidth: torch.Tensor,
    multipliers: tuple[float, ...] = DEFAULT_MULTIPLIERS,
) -> torch.Tensor:
    """Sum of RBF kernels ``exp(-d^2 / (m * bandwidth))`` over ``multipliers``."""
    d2 = pairwise_squared_distances(x, y)
    return sum(torch.exp(-d2 / (m * bandwidth)) for m in multipliers)


def mmd2(
    source: torch.Tensor,
    target: torch.Tensor,
    multipliers: tuple[float, ...] = DEFAULT_MULTIPLIERS,
) -> torch.Tensor:
    """Biased empirical MMD^2 between two feature sets."""
    if source.dim() != 2 or target.dim() != 2:
        raise ValueError(
            f"Expected 2-D features, got {tuple(source.shape)} and {tuple(target.shape)}"
        )
    if source.shape[1] != target.shape[1]:
        raise ValueError(
            f"Feature dimension mismatch: {source.shape[1]} vs {target.shape[1]}"
        )

    combined = torch.cat([source, target], dim=0)
    bandwidth = median_bandwidth(combined).detach()

    k_ss = multi_rbf_kernel(source, source, bandwidth, multipliers).mean()
    k_tt = multi_rbf_kernel(target, target, bandwidth, multipliers).mean()
    k_st = multi_rbf_kernel(source, target, bandwidth, multipliers).mean()

    return k_ss - 2.0 * k_st + k_tt


def pairwise_domain_mmd2(
    domain_features: dict[str, torch.Tensor],
    multipliers: tuple[float, ...] = DEFAULT_MULTIPLIERS,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Mean MMD^2 over all unordered domain pairs. Used by Task 3's DAN-DG."""
    names = sorted(domain_features)
    if len(names) < 2:
        raise ValueError(f"Need at least 2 domains for pairwise MMD, got {names}")

    per_pair: dict[str, float] = {}
    total = None

    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            value = mmd2(domain_features[a], domain_features[b], multipliers)
            per_pair[f"{a}__{b}"] = float(value.detach())
            total = value if total is None else total + value

    mean = total / len(per_pair)
    return mean, per_pair
