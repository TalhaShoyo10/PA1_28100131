"""Shared metric implementations."""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, Sequence

import numpy as np
from sklearn.metrics import f1_score, roc_auc_score, confusion_matrix


def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Top-1 accuracy in percent."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    if y_true.size == 0:
        return float("nan")
    return 100.0 * float((y_true == y_pred).mean())


def macro_f1(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int | None = None) -> float:
    """Macro-averaged F1 in percent."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    if y_true.size == 0:
        return float("nan")
    labels = list(range(num_classes)) if num_classes is not None else None
    return 100.0 * float(f1_score(y_true, y_pred, average="macro", labels=labels, zero_division=0))


def per_class_accuracy(
    y_true: np.ndarray, y_pred: np.ndarray, num_classes: int
) -> Dict[int, float]:
    """Per-class recall (accuracy within each true class) in percent."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    out: Dict[int, float] = {}
    for c in range(num_classes):
        mask = y_true == c
        out[c] = 100.0 * float((y_pred[mask] == c).mean()) if mask.any() else float("nan")
    return out


def mean_max_confidence(probs: np.ndarray) -> float:
    """Mean of the maximum predicted probability. Required by Task 1 step 1."""
    probs = np.asarray(probs)
    if probs.size == 0:
        return float("nan")
    return float(probs.max(axis=1).mean())


def prediction_consistency(pred_clean: np.ndarray, pred_transformed: np.ndarray) -> float:
    """Fraction (percent) of images whose predicted class survives an intervention."""
    pred_clean, pred_transformed = np.asarray(pred_clean), np.asarray(pred_transformed)
    if pred_clean.shape != pred_transformed.shape:
        raise ValueError(
            f"Paired predictions must align: got {pred_clean.shape} vs {pred_transformed.shape}"
        )
    if pred_clean.size == 0:
        return float("nan")
    return 100.0 * float((pred_clean == pred_transformed).mean())


def top_confusions(
    y_true: np.ndarray, y_pred: np.ndarray, num_classes: int, k: int = 5
) -> list[tuple[int, int, int]]:
    """Return the ``k`` most frequent (true, predicted) off-diagonal pairs."""
    cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))
    pairs = [
        (int(t), int(p), int(cm[t, p]))
        for t in range(num_classes)
        for p in range(num_classes)
        if t != p and cm[t, p] > 0
    ]
    return sorted(pairs, key=lambda x: -x[2])[:k]


def auroc(scores_known: np.ndarray, scores_unknown: np.ndarray) -> float:
    """AUROC in percent for an *unknownness* score."""
    scores_known = np.asarray(scores_known, dtype=np.float64)
    scores_unknown = np.asarray(scores_unknown, dtype=np.float64)
    if scores_known.size == 0 or scores_unknown.size == 0:
        return float("nan")
    y = np.concatenate([np.zeros_like(scores_known), np.ones_like(scores_unknown)])
    s = np.concatenate([scores_known, scores_unknown])
    return 100.0 * float(roc_auc_score(y, s))


def calibrate_threshold(scores_val_known: np.ndarray, target_tpr: float = 95.0) -> float:
    """Threshold tau = the ``target_tpr``-th percentile of validation unknownness."""
    scores_val_known = np.asarray(scores_val_known, dtype=np.float64)
    if scores_val_known.size == 0:
        raise ValueError("Cannot calibrate a threshold from an empty validation score array.")
    return float(np.percentile(scores_val_known, target_tpr))


def acceptance_rate(scores: np.ndarray, tau: float) -> float:
    """Percent of examples accepted as known under ``u(x) <= tau``."""
    scores = np.asarray(scores, dtype=np.float64)
    if scores.size == 0:
        return float("nan")
    return 100.0 * float((scores <= tau).mean())


def rejection_rate(scores: np.ndarray, tau: float) -> float:
    """Percent of examples rejected as unknown under ``u(x) > tau``."""
    acc = acceptance_rate(scores, tau)
    return float("nan") if np.isnan(acc) else 100.0 - acc


def fpr_at_95_tpr(scores_unknown: np.ndarray, tau: float) -> float:
    """Fraction (percent) of unknowns incorrectly accepted at threshold ``tau``."""
    return acceptance_rate(scores_unknown, tau)


@dataclass
class OSRResult:
    """One (model, score, unknown-group) evaluation row."""

    model: str
    score: str
    unknown_group: str
    auroc: float
    tau: float
    known_acceptance: float
    unknown_rejection: float
    fpr_at_95_tpr: float
    closed_set_accuracy: float | None = None

    def to_dict(self) -> dict:
        return asdict(self)


def shape_bias(n_shape: int, n_texture: int) -> float:
    """Shape bias (%) = N_shape / (N_shape + N_texture) x 100."""
    denom = n_shape + n_texture
    return 100.0 * n_shape / denom if denom > 0 else float("nan")


def coverage(n_shape: int, n_texture: int, n_total: int) -> float:
    """Coverage (%) = (N_shape + N_texture) / N_total x 100."""
    return 100.0 * (n_shape + n_texture) / n_total if n_total > 0 else float("nan")
