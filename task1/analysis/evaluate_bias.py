"""Evaluate backbones and heads under each Task 1 intervention."""

from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from common.metrics import (
    accuracy,
    coverage,
    macro_f1,
    mean_max_confidence,
    prediction_consistency,
    shape_bias,
)


@dataclass
class ConditionResult:
    """Metrics for one (model, condition) pair."""

    model: str
    condition: str
    accuracy: float
    macro_f1: float
    mean_max_confidence: float
    consistency_vs_clean: float = float("nan")
    accuracy_delta: float = float("nan")
    n: int = 0

    def to_dict(self) -> dict:
        return asdict(self)


@torch.no_grad()
def extract_features(
    backbone, images: torch.Tensor, batch_size: int = 64, device: str = "cpu"
) -> torch.Tensor:
    """Run a frozen backbone over pre-normalized images."""
    backbone.eval()
    outputs = []
    for start in range(0, len(images), batch_size):
        batch = images[start : start + batch_size].to(device)
        outputs.append(backbone.extract(batch).cpu())
    return torch.cat(outputs)


def normalize_for_backbone(images: torch.Tensor, backbone) -> torch.Tensor:
    """Apply a backbone's own normalization to shared [0,1] images.

    Every model receives byte-identical inputs; only this final step differs.
    """
    normalizer = backbone.normalization()
    mean = torch.tensor(normalizer.mean).view(1, 3, 1, 1)
    std = torch.tensor(normalizer.std).view(1, 3, 1, 1)
    return (images - mean) / std


@torch.no_grad()
def predict(
    backbone, head, images: torch.Tensor, device: str = "cpu", batch_size: int = 64
) -> tuple[np.ndarray, np.ndarray]:
    """Return predictions and softmax probabilities for a backbone plus head."""
    normalized = normalize_for_backbone(images, backbone)
    features = extract_features(backbone, normalized, batch_size, device)

    head.eval()
    logits = head(features.to(device)).cpu()
    probabilities = torch.softmax(logits, dim=1).numpy()
    return logits.argmax(1).numpy(), probabilities


@torch.no_grad()
def predict_zero_shot(
    clip_backbone, images: torch.Tensor, text_embeddings: torch.Tensor,
    device: str = "cpu", batch_size: int = 64,
) -> tuple[np.ndarray, np.ndarray]:
    """CLIP zero-shot predictions from scaled class similarities.

    Confidence is the softmax over logit-scaled cosine similarities, matching
    how the trained heads' confidence is computed.
    """
    normalized = normalize_for_backbone(images, clip_backbone)
    features = extract_features(clip_backbone, normalized, batch_size, device)

    similarities = features.to(device) @ text_embeddings.T.to(device)
    logits = clip_backbone.logit_scale * similarities
    probabilities = torch.softmax(logits, dim=1).cpu().numpy()
    return logits.argmax(1).cpu().numpy(), probabilities


def evaluate_condition(
    model_name: str,
    condition: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    probabilities: np.ndarray,
    num_classes: int,
    clean_predictions: np.ndarray | None = None,
    clean_accuracy: float | None = None,
) -> ConditionResult:
    """Compute metrics for one condition, relative to its clean baseline."""
    acc = accuracy(y_true, y_pred)
    return ConditionResult(
        model=model_name,
        condition=condition,
        accuracy=acc,
        macro_f1=macro_f1(y_true, y_pred, num_classes),
        mean_max_confidence=mean_max_confidence(probabilities),
        consistency_vs_clean=(
            prediction_consistency(clean_predictions, y_pred)
            if clean_predictions is not None
            else float("nan")
        ),
        accuracy_delta=acc - clean_accuracy if clean_accuracy is not None else float("nan"),
        n=len(y_true),
    )


def classify_cue_conflict(
    predictions: np.ndarray, content_labels: np.ndarray, style_labels: np.ndarray
) -> dict:
    """Split predictions into shape, texture and other decisions.

    The 'other' bucket matters: a shape-bias score ignores predictions that
    belong to neither intended class, so it must be reported alongside
    coverage to be interpretable.
    """
    predictions = np.asarray(predictions)
    content_labels = np.asarray(content_labels)
    style_labels = np.asarray(style_labels)

    is_shape = predictions == content_labels
    is_texture = (predictions == style_labels) & ~is_shape

    n_shape = int(is_shape.sum())
    n_texture = int(is_texture.sum())
    n_total = len(predictions)

    return {
        "n_shape": n_shape,
        "n_texture": n_texture,
        "n_other": n_total - n_shape - n_texture,
        "n_total": n_total,
        "shape_bias": shape_bias(n_shape, n_texture),
        "coverage": coverage(n_shape, n_texture, n_total),
    }


def translation_curve(
    results_by_displacement: dict[int, list[ConditionResult]]
) -> list[dict]:
    """Average accuracy and consistency across the four cardinal directions.

    At displacement 0 the only entry is the clean baseline, whose
    ``consistency_vs_clean`` is NaN because it has no transformed counterpart.
    Consistency there is 100 by definition -- an image compared with itself --
    so the curve starts at its true anchor instead of a NaN.
    """
    rows = []
    for displacement in sorted(results_by_displacement):
        entries = results_by_displacement[displacement]
        values = [
            e.consistency_vs_clean
            for e in entries
            if not np.isnan(e.consistency_vs_clean)
        ]

        if values:
            consistency = float(np.mean(values))
        elif displacement == 0:
            consistency = 100.0
        else:
            consistency = float("nan")

        rows.append(
            {
                "displacement": displacement,
                "accuracy": float(np.mean([e.accuracy for e in entries])),
                "consistency": consistency,
                "n_directions": len(entries),
            }
        )
    return rows


def train_linear_head(
    features: torch.Tensor,
    labels: torch.Tensor,
    val_features: torch.Tensor,
    val_labels: torch.Tensor,
    num_classes: int,
    epochs: int = 50,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    batch_size: int = 128,
    patience: int = 5,
    device: str = "cpu",
    seed: int = 6304,
) -> tuple[torch.nn.Module, dict]:
    """Train one linear head on frozen features with early stopping."""
    from task1.models.backbones import LinearHead

    torch.manual_seed(seed)
    head = LinearHead(features.shape[1], num_classes).to(device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=weight_decay)

    loader = DataLoader(
        TensorDataset(features, labels),
        batch_size=batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
    )

    best_accuracy = -1.0
    best_state = None
    best_epoch = -1
    epochs_without_improvement = 0

    for epoch in range(epochs):
        head.train()
        for batch_features, batch_labels in loader:
            loss = torch.nn.functional.cross_entropy(
                head(batch_features.to(device)), batch_labels.to(device)
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

        head.eval()
        with torch.no_grad():
            predictions = head(val_features.to(device)).argmax(1).cpu().numpy()
        val_accuracy = accuracy(val_labels.numpy(), predictions)

        if val_accuracy > best_accuracy:
            best_accuracy, best_epoch = val_accuracy, epoch
            best_state = {k: v.clone() for k, v in head.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                break

    if best_state is not None:
        head.load_state_dict(best_state)

    return head, {
        "best_val_accuracy": best_accuracy,
        "best_epoch": best_epoch,
        "epochs_run": epoch + 1,
    }
