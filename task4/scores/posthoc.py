"""Post-hoc unknownness scores. Larger values always mean more novel."""

from __future__ import annotations

import numpy as np
from scipy.special import logsumexp, softmax


def msp_score(logits: np.ndarray) -> np.ndarray:
    """Maximum Softmax Probability: ``u(x) = 1 - max_k p_k(x)``.

    Measures NORMALIZED confidence. Softmax discards absolute logit magnitude,
    so a uniformly weak prediction and a uniformly strong one score alike if
    their relative shape matches.
    """
    return 1.0 - softmax(np.asarray(logits, dtype=np.float64), axis=1).max(axis=1)


def mls_score(logits: np.ndarray) -> np.ndarray:
    """Maximum Logit Score: ``u(x) = -max_k z_k(x)``.

    Retains ABSOLUTE logit magnitude, which softmax normalises away. Vaze et
    al. (2022) argue this is why MLS tracks closed-set classifier quality.
    """
    return -np.asarray(logits, dtype=np.float64).max(axis=1)


def energy_score(logits: np.ndarray) -> np.ndarray:
    """Free energy: ``u(x) = -log sum_k exp(z_k(x))``.

    Uses ALL logits rather than only the maximum, so evidence spread across
    several classes still counts as evidence.
    """
    return -logsumexp(np.asarray(logits, dtype=np.float64), axis=1)


def fit_mahalanobis(
    features: np.ndarray, labels: np.ndarray, num_classes: int, epsilon: float = 1e-6
) -> tuple[np.ndarray, np.ndarray]:
    """Estimate class means and one shared diagonal covariance.

    Fitted from UNAUGMENTED training features, as the assignment requires:
    augmented features would describe the augmentation distribution rather
    than the class distribution.
    """
    features = np.asarray(features, dtype=np.float64)
    labels = np.asarray(labels)

    means = np.stack(
        [features[labels == c].mean(axis=0) for c in range(num_classes)]
    )

    centered = np.concatenate(
        [features[labels == c] - means[c] for c in range(num_classes)]
    )
    variance = centered.var(axis=0) + epsilon
    return means, variance


def mahalanobis_score(
    features: np.ndarray, means: np.ndarray, variance: np.ndarray
) -> np.ndarray:
    """Distance to the nearest known-class cluster in feature space.

    ``u(x) = min_c (f(x) - mu_c)^T Sigma^-1 (f(x) - mu_c)``

    A geometric signal rather than a logit one: it can flag an input that
    lands far from every known cluster even when the classifier is confident.
    """
    features = np.asarray(features, dtype=np.float64)
    inverse_variance = 1.0 / variance

    distances = np.stack(
        [
            (((features - mean) ** 2) * inverse_variance).sum(axis=1)
            for mean in means
        ],
        axis=1,
    )
    return distances.min(axis=1)


def proser_score(
    logits: np.ndarray, num_known: int, num_dummy: int
) -> np.ndarray:
    """PROSER's placeholder-based detection score.

    Compares the strongest dummy response against the strongest known-class
    response: the dummies were trained to fire in the regions between and
    around known classes, so a large dummy response is direct evidence of
    novelty rather than merely weak known-class evidence.
    """
    logits = np.asarray(logits, dtype=np.float64)
    if logits.shape[1] != num_known + num_dummy:
        raise ValueError(
            f"Expected {num_known + num_dummy} logits, got {logits.shape[1]}"
        )

    known_max = logits[:, :num_known].max(axis=1)
    dummy_max = logits[:, num_known:].max(axis=1)
    return dummy_max - known_max


SCORE_REGISTRY = {
    "msp": msp_score,
    "mls": mls_score,
    "energy": energy_score,
}


def compute_score(name: str, logits: np.ndarray, **kwargs) -> np.ndarray:
    """Dispatch a logit-based score by name."""
    if name == "mahalanobis":
        return mahalanobis_score(kwargs["features"], kwargs["means"], kwargs["variance"])
    if name == "proser":
        return proser_score(logits, kwargs["num_known"], kwargs["num_dummy"])
    if name not in SCORE_REGISTRY:
        raise KeyError(f"Unknown score {name!r}. Available: {sorted(SCORE_REGISTRY)}")
    return SCORE_REGISTRY[name](logits)
