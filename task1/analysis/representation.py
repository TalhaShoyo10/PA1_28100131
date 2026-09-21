"""Representation stability and low-dimensional visualization."""

from __future__ import annotations

import numpy as np
import torch
from sklearn.manifold import TSNE

from common.plotting import save_figure, set_style


def cosine_stability(
    clean_features: torch.Tensor | np.ndarray,
    transformed_features: torch.Tensor | np.ndarray,
) -> dict:
    """Mean paired cosine similarity between clean and transformed features.

        I_T = (1/N) sum_i  <f(x_i), f(T(x_i))> / (||f(x_i)|| ||f(T(x_i))||)

    Answers a different question from prediction consistency: a cue may stay
    encoded even when the classifier stops using it, and a prediction may
    survive a large representation shift.
    """
    clean = np.asarray(
        clean_features.detach().cpu() if torch.is_tensor(clean_features) else clean_features,
        dtype=np.float64,
    )
    transformed = np.asarray(
        transformed_features.detach().cpu()
        if torch.is_tensor(transformed_features)
        else transformed_features,
        dtype=np.float64,
    )

    if clean.shape != transformed.shape:
        raise ValueError(
            f"Paired features must align: {clean.shape} vs {transformed.shape}"
        )

    clean_norm = np.linalg.norm(clean, axis=1)
    transformed_norm = np.linalg.norm(transformed, axis=1)
    denominator = clean_norm * transformed_norm

    valid = denominator > 0
    similarities = np.full(len(clean), np.nan)
    similarities[valid] = (clean[valid] * transformed[valid]).sum(axis=1) / denominator[valid]

    return {
        "cosine_stability": float(np.nanmean(similarities)),
        "std": float(np.nanstd(similarities)),
        "min": float(np.nanmin(similarities)) if valid.any() else float("nan"),
        "max": float(np.nanmax(similarities)) if valid.any() else float("nan"),
        "n": int(valid.sum()),
        "n_degenerate": int((~valid).sum()),
        "per_example": similarities,
    }


def fit_tsne(
    clean_features: np.ndarray,
    transformed_features: np.ndarray,
    perplexity: float = 30.0,
    n_iter: int = 1000,
    seed: int = 6304,
    metric: str = "cosine",
    init: str = "pca",
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Fit ONE projection to the combined clean and transformed features.

    Fitting a single projection is what makes the two conditions comparable:
    separate projections would place them in unrelated coordinate systems.

    Projections fitted for different backbones are NOT coordinate-comparable
    with each other; only within-plot structure may be discussed.
    """
    clean = np.asarray(clean_features, dtype=np.float64)
    transformed = np.asarray(transformed_features, dtype=np.float64)
    combined = np.concatenate([clean, transformed])

    effective_perplexity = min(perplexity, (len(combined) - 1) / 3.0)
    if effective_perplexity < 5.0:
        effective_perplexity = max(2.0, (len(combined) - 1) / 3.0)

    tsne = TSNE(
        n_components=2,
        perplexity=effective_perplexity,
        max_iter=int(n_iter),
        init=init,
        metric=metric,
        random_state=seed,
    )
    embedded = tsne.fit_transform(combined)

    settings = {
        "method": "tsne",
        "perplexity": float(effective_perplexity),
        "requested_perplexity": float(perplexity),
        "n_iter": int(n_iter),
        "init": init,
        "metric": metric,
        "seed": seed,
        "n_clean": len(clean),
        "n_transformed": len(transformed),
        "fit_on": "combined_clean_and_transformed",
    }
    return embedded[: len(clean)], embedded[len(clean) :], settings


def plot_representation(
    clean_2d: np.ndarray,
    transformed_2d: np.ndarray,
    labels: np.ndarray,
    class_names: list[str],
    backbone_name: str,
    condition: str,
    output_path,
    settings: dict | None = None,
):
    """Scatter clean vs transformed features: colour by class, marker by condition."""
    import matplotlib.pyplot as plt

    set_style()
    fig, ax = plt.subplots(figsize=(5.5, 4.5))

    colors = plt.cm.tab10(np.linspace(0, 1, len(class_names)))

    for index, name in enumerate(class_names):
        mask = labels == index
        if not mask.any():
            continue
        ax.scatter(
            clean_2d[mask, 0], clean_2d[mask, 1],
            color=colors[index], marker="o", s=18, alpha=0.75,
            edgecolors="none", label=name,
        )
        ax.scatter(
            transformed_2d[mask, 0], transformed_2d[mask, 1],
            color=colors[index], marker="^", s=18, alpha=0.55,
            edgecolors="none",
        )

    subtitle = ""
    if settings:
        subtitle = (
            f"  (perplexity={settings['perplexity']:.0f}, "
            f"metric={settings['metric']}, seed={settings['seed']})"
        )
    ax.set_title(f"{backbone_name} — clean (o) vs {condition} (^){subtitle}")
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    ax.legend(fontsize=6, ncol=2, loc="best", framealpha=0.8)

    return save_figure(fig, output_path)


def displacement_from_clean(
    clean_2d: np.ndarray, transformed_2d: np.ndarray
) -> dict:
    """Mean 2-D displacement between paired points in the shared projection.

    A descriptive companion to the plot, not a distance in the original
    feature space: t-SNE preserves neighbourhoods, not distances.
    """
    distances = np.linalg.norm(transformed_2d - clean_2d, axis=1)
    return {
        "mean_displacement": float(distances.mean()),
        "median_displacement": float(np.median(distances)),
        "max_displacement": float(distances.max()),
    }
