"""Evaluation helpers shared by Tasks 2 and 3."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch
from torch.utils.data import DataLoader

from common.metrics import accuracy, macro_f1, per_class_accuracy, top_confusions


@dataclass
class DomainEvaluation:
    """Predictions and metrics for one evaluated domain."""

    domain: str
    accuracy: float
    macro_f1: float
    per_class: dict[int, float] = field(default_factory=dict)
    y_true: np.ndarray | None = None
    y_pred: np.ndarray | None = None
    features: np.ndarray | None = None

    def to_row(self) -> dict:
        return {
            "domain": self.domain,
            "accuracy": self.accuracy,
            "macro_f1": self.macro_f1,
        }


@torch.no_grad()
def collect_predictions(
    model: torch.nn.Module,
    loader: DataLoader,
    device: str,
    return_features: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """Run the model over a loader, returning labels, predictions and features."""
    model.eval()
    all_true: list[np.ndarray] = []
    all_pred: list[np.ndarray] = []
    all_feat: list[np.ndarray] = []

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        logits, feats = model(images, return_features=True)
        all_true.append(labels.numpy())
        all_pred.append(logits.argmax(1).cpu().numpy())
        if return_features:
            all_feat.append(feats.cpu().numpy())

    y_true = np.concatenate(all_true) if all_true else np.array([], dtype=int)
    y_pred = np.concatenate(all_pred) if all_pred else np.array([], dtype=int)
    features = np.concatenate(all_feat) if return_features and all_feat else None
    return y_true, y_pred, features


def evaluate_domain(
    model: torch.nn.Module,
    loader: DataLoader,
    domain: str,
    num_classes: int,
    device: str,
    return_features: bool = False,
) -> DomainEvaluation:
    """Evaluate one domain and package its metrics and raw predictions."""
    y_true, y_pred, features = collect_predictions(model, loader, device, return_features)
    return DomainEvaluation(
        domain=domain,
        accuracy=accuracy(y_true, y_pred),
        macro_f1=macro_f1(y_true, y_pred, num_classes),
        per_class=per_class_accuracy(y_true, y_pred, num_classes),
        y_true=y_true,
        y_pred=y_pred,
        features=features,
    )


def evaluate_source_domains(
    model: torch.nn.Module,
    loaders: dict[str, DataLoader],
    num_classes: int,
    device: str,
    return_features: bool = False,
) -> dict[str, DomainEvaluation]:
    """Evaluate every source domain separately.

    Reported per domain before any mean, because pooling can hide
    source-specific behaviour.
    """
    return {
        domain: evaluate_domain(model, loader, domain, num_classes, device, return_features)
        for domain, loader in loaders.items()
    }


def mean_source_macro_f1(evaluations: dict[str, DomainEvaluation]) -> float:
    """Mean macro-F1 across source validation domains: the selection metric."""
    if not evaluations:
        return float("nan")
    return float(np.mean([e.macro_f1 for e in evaluations.values()]))


def mean_source_accuracy(evaluations: dict[str, DomainEvaluation]) -> float:
    if not evaluations:
        return float("nan")
    return float(np.mean([e.accuracy for e in evaluations.values()]))


def worst_source_macro_f1(evaluations: dict[str, DomainEvaluation]) -> tuple[str, float]:
    """Worst-performing source domain, which a mean can hide."""
    if not evaluations:
        return "", float("nan")
    domain = min(evaluations, key=lambda d: evaluations[d].macro_f1)
    return domain, evaluations[domain].macro_f1


def worst_source_accuracy(evaluations: dict[str, DomainEvaluation]) -> tuple[str, float]:
    if not evaluations:
        return "", float("nan")
    domain = min(evaluations, key=lambda d: evaluations[d].accuracy)
    return domain, evaluations[domain].accuracy


def per_class_delta(
    baseline: DomainEvaluation, candidate: DomainEvaluation, class_names: list[str]
) -> list[dict]:
    """Per-class accuracy change relative to a baseline, sorted worst-first.

    Exposes class-specific negative transfer that an aggregate gain can hide.
    """
    rows: list[dict] = []
    for index, name in enumerate(class_names):
        before = baseline.per_class.get(index, float("nan"))
        after = candidate.per_class.get(index, float("nan"))
        rows.append(
            {
                "class_index": index,
                "class_name": name,
                "baseline_accuracy": before,
                "method_accuracy": after,
                "delta": after - before,
            }
        )
    return sorted(rows, key=lambda r: (np.isnan(r["delta"]), r["delta"]))


def confusion_rows(
    evaluation: DomainEvaluation, class_names: list[str], k: int = 10
) -> list[dict]:
    """The ``k`` most frequent (true, predicted) confusions as tidy rows."""
    if evaluation.y_true is None or evaluation.y_pred is None:
        return []
    pairs = top_confusions(evaluation.y_true, evaluation.y_pred, len(class_names), k)
    return [
        {
            "true_class": class_names[t],
            "predicted_class": class_names[p],
            "count": n,
        }
        for t, p, n in pairs
    ]
