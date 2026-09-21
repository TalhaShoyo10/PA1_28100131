"""Domain separability: how recoverable is domain identity from features?"""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from common.seed import ASSIGNMENT_SEED


def balance_feature_sets(
    feature_sets: list[np.ndarray], seed: int = ASSIGNMENT_SEED
) -> list[np.ndarray]:
    """Subsample every set to the smallest set's size.

    Without this, an imbalanced classifier could reach high accuracy by
    predicting the majority domain, and the score would reflect set sizes
    rather than feature separability.
    """
    if not feature_sets:
        return []
    n = min(len(f) for f in feature_sets)
    rng = np.random.default_rng(seed)
    return [f[np.sort(rng.choice(len(f), size=n, replace=False))] for f in feature_sets]


def domain_separability(
    feature_sets: list[np.ndarray],
    seed: int = ASSIGNMENT_SEED,
    test_fraction: float = 0.3,
    C: float = 1.0,
    standardize: bool = True,
) -> dict:
    """Train a logistic regression to predict domain identity from features.

    Held-out accuracy is the separability score. Chance is 100/n_domains.

    A LOWER score means domain information is harder to recover. It does NOT
    mean class information was preserved, and it does not imply better target
    recognition -- that relationship is what the research questions test.

    Multi-class problems use scikit-learn's default multinomial handling; the
    explicit ``multi_class`` argument was deprecated in sklearn 1.5 and removed
    in 1.7, and passing it would break on a newer Colab environment.
    """
    if len(feature_sets) < 2:
        raise ValueError(f"Need at least 2 domains, got {len(feature_sets)}")

    balanced = balance_feature_sets(feature_sets, seed)
    X = np.concatenate(balanced)
    y = np.concatenate([np.full(len(f), i) for i, f in enumerate(balanced)])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_fraction, random_state=seed, stratify=y
    )

    if standardize:
        scaler = StandardScaler().fit(X_train)
        X_train, X_test = scaler.transform(X_train), scaler.transform(X_test)

    classifier = LogisticRegression(
        C=C,
        class_weight="balanced",
        max_iter=2000,
        random_state=seed,
    )
    classifier.fit(X_train, y_train)

    return {
        "separability": 100.0 * float(classifier.score(X_test, y_test)),
        "chance": 100.0 / len(balanced),
        "n_domains": len(balanced),
        "n_per_domain": len(balanced[0]),
        "n_train": len(X_train),
        "n_test": len(X_test),
    }


def source_target_separability(
    source_features: np.ndarray, target_features: np.ndarray, seed: int = ASSIGNMENT_SEED, **kwargs
) -> dict:
    """Task 2 binary diagnostic: source vs target. Chance is 50%."""
    return domain_separability([source_features, target_features], seed=seed, **kwargs)


def source_domain_separability(
    features_by_domain: dict[str, np.ndarray], seed: int = ASSIGNMENT_SEED, **kwargs
) -> dict:
    """Task 3 three-way diagnostic: Photo vs Art vs Cartoon. Chance is 33.3%."""
    names = sorted(features_by_domain)
    result = domain_separability(
        [features_by_domain[n] for n in names], seed=seed, **kwargs
    )
    result["domains"] = names
    return result
