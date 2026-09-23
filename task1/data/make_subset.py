"""Build and load the fixed class-balanced evaluation subset for Task 1."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
from PIL import Image
from torchvision import datasets

from common.config import apply_overrides, load_config
from common.logging import get_logger, save_json
from common.seed import set_seed

LOGGER = get_logger("task1.make_subset")

STL10_CLASSES = (
    "airplane", "bird", "car", "cat", "deer",
    "dog", "horse", "monkey", "ship", "truck",
)


def load_stl10(root: Path | str, split: str, download: bool = True):
    """Return the torchvision STL-10 dataset for ``split`` without transforms."""
    return datasets.STL10(root=str(root), split=split, download=download, transform=None)


def stratified_indices(
    labels: np.ndarray, per_class: int, seed: int
) -> tuple[list[int], dict[str, int]]:
    """Select ``per_class`` indices from each class, documenting any shortfall."""
    rng = np.random.default_rng(seed)
    selected: list[int] = []
    shortfall: dict[str, int] = {}

    for cls in sorted(set(labels.tolist())):
        cls_indices = np.flatnonzero(labels == cls)
        available = len(cls_indices)
        take = min(per_class, available)
        if take < per_class:
            shortfall[str(cls)] = available
        chosen = rng.choice(cls_indices, size=take, replace=False)
        selected.extend(sorted(chosen.tolist()))

    return sorted(selected), shortfall


def build_eval_subset(cfg) -> dict:
    """Construct the balanced evaluation subset manifest from the official test split."""
    set_seed(cfg.seed)
    subset_cfg = cfg.data.eval_subset

    dataset = load_stl10(cfg.data.root, split="test")
    labels = np.asarray(dataset.labels)

    num_classes = cfg.data.num_classes
    per_class, remainder = divmod(subset_cfg.size, num_classes)
    if remainder:
        LOGGER.warning(
            "Subset size %d is not divisible by %d classes; using %d per class "
            "(%d images total).",
            subset_cfg.size, num_classes, per_class, per_class * num_classes,
        )

    indices, shortfall = stratified_indices(labels, per_class, subset_cfg.seed)

    if shortfall:
        LOGGER.warning(
            "Class imbalance: classes %s had fewer than %d test images. "
            "All available examples were used; documented in the manifest.",
            shortfall, per_class,
        )

    manifest = {
        "dataset": "stl10",
        "split": "test",
        "seed": subset_cfg.seed,
        "requested_size": subset_cfg.size,
        "per_class": per_class,
        "actual_size": len(indices),
        "classes": list(cfg.data.classes),
        "indices": indices,
        "labels": [int(labels[i]) for i in indices],
        "class_imbalance": shortfall,
    }
    return manifest


def load_eval_subset(cfg) -> list[tuple[str, Image.Image, int]]:
    """Load the saved subset as ``(image_id, PIL image, label)`` triples."""
    manifest_path = Path(cfg.data.eval_subset.manifest)
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Evaluation subset manifest not found: {manifest_path}. "
            "Generate it first with: python task1/data/make_subset.py "
            "--config task1/configs/base.yaml"
        )

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    dataset = load_stl10(cfg.data.root, split="test", download=False)

    items: list[tuple[str, Image.Image, int]] = []
    for index in manifest["indices"]:
        image, label = dataset[index]
        items.append((f"test_{index:05d}", image, int(label)))
    return items


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the Task 1 evaluation subset.")
    parser.add_argument("--config", type=Path, default=Path("task1/configs/base.yaml"))
    parser.add_argument(
        "--set", dest="overrides", nargs="*", default=None, metavar="KEY=VALUE",
        help="Override config entries.",
    )
    args = parser.parse_args()

    cfg = apply_overrides(load_config(args.config), args.overrides)
    manifest = build_eval_subset(cfg)
    path = save_json(manifest, cfg.data.eval_subset.manifest)

    LOGGER.info(
        "Saved %d image identifiers (%d per class) to %s",
        manifest["actual_size"], manifest["per_class"], path,
    )


if __name__ == "__main__":
    main()
