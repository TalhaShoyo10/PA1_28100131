"""CIFAR-10 known classes and the fixed CIFAR-100 unknown groups."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from common.seed import ASSIGNMENT_SEED, make_generator, seed_worker

CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)

CIFAR10_CLASSES = (
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
)

NEAR_UNKNOWNS = (
    "bus", "pickup_truck", "motorcycle", "tractor",
    "wolf", "fox", "leopard", "camel",
)

FAR_UNKNOWNS = (
    "bottle", "bowl", "chair", "clock",
    "keyboard", "mushroom", "sunflower", "wardrobe",
)


def train_transform(rand_augment: dict | None = None):
    """Random crop with 4-pixel padding, flip, and optionally RandAugment.

    RandAugment is inserted AFTER crop and flip and BEFORE tensor conversion,
    which is the single controlled difference between Vanilla and GCSC.
    """
    ops = [
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
    ]
    if rand_augment:
        ops.append(
            transforms.RandAugment(
                num_ops=rand_augment["num_ops"], magnitude=rand_augment["magnitude"]
            )
        )
    ops += [transforms.ToTensor(), transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD)]
    return transforms.Compose(ops)


def eval_transform():
    """Deterministic evaluation transform: no augmentation at all."""
    return transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD)]
    )


def stratified_split_indices(
    labels: np.ndarray, val_fraction: float, seed: int
) -> tuple[list[int], list[int]]:
    """Stratified split of CIFAR-10 train into optimization and validation."""
    rng = np.random.default_rng(seed)
    train_idx, val_idx = [], []

    for cls in sorted(set(labels.tolist())):
        cls_indices = np.flatnonzero(labels == cls)
        rng.shuffle(cls_indices)
        n_val = max(1, int(round(len(cls_indices) * val_fraction)))
        val_idx.extend(cls_indices[:n_val].tolist())
        train_idx.extend(cls_indices[n_val:].tolist())

    return sorted(train_idx), sorted(val_idx)


def build_cifar10_splits(
    root: Path | str,
    val_fraction: float = 0.1,
    seed: int = ASSIGNMENT_SEED,
    download: bool = True,
) -> dict:
    """Return index lists for the 90/10 CIFAR-10 train/validation split."""
    dataset = datasets.CIFAR10(root=str(root), train=True, download=download)
    labels = np.asarray(dataset.targets)
    train_idx, val_idx = stratified_split_indices(labels, val_fraction, seed)

    return {
        "dataset": "cifar10",
        "seed": seed,
        "val_fraction": val_fraction,
        "n_train": len(train_idx),
        "n_val": len(val_idx),
        "train": train_idx,
        "val": val_idx,
        "classes": list(CIFAR10_CLASSES),
    }


def cifar10_loaders(
    root: Path | str,
    split_manifest: dict,
    batch_size: int = 128,
    rand_augment: dict | None = None,
    num_workers: int = 2,
    seed: int = ASSIGNMENT_SEED,
    download: bool = True,
) -> dict[str, DataLoader]:
    """Build train, validation and test loaders for the known classes."""
    train_base = datasets.CIFAR10(
        root=str(root), train=True, download=download,
        transform=train_transform(rand_augment),
    )
    eval_base = datasets.CIFAR10(
        root=str(root), train=True, download=False, transform=eval_transform()
    )
    test_base = datasets.CIFAR10(
        root=str(root), train=False, download=download, transform=eval_transform()
    )

    def make(dataset, shuffle: bool) -> DataLoader:
        return DataLoader(
            dataset, batch_size=batch_size, shuffle=shuffle,
            num_workers=num_workers, pin_memory=torch.cuda.is_available(),
            generator=make_generator(seed) if shuffle else None,
            worker_init_fn=seed_worker if num_workers > 0 else None,
        )

    return {
        "train": make(Subset(train_base, split_manifest["train"]), True),
        "train_unaugmented": make(Subset(eval_base, split_manifest["train"]), False),
        "val": make(Subset(eval_base, split_manifest["val"]), False),
        "test": make(test_base, False),
    }


def cifar100_unknown_indices(
    root: Path | str, class_names: tuple[str, ...], download: bool = True
) -> tuple[list[int], datasets.CIFAR100]:
    """Locate the CIFAR-100 TEST images belonging to the given fine classes.

    Only the test partition is touched: CIFAR-100 training images may not be
    used for training, checkpoint selection, score design, or thresholds.
    """
    dataset = datasets.CIFAR100(
        root=str(root), train=False, download=download, transform=eval_transform()
    )
    name_to_index = {name: i for i, name in enumerate(dataset.classes)}

    missing = [n for n in class_names if n not in name_to_index]
    if missing:
        raise KeyError(
            f"CIFAR-100 fine classes not found: {missing}. "
            f"Available example names: {sorted(dataset.classes)[:5]}..."
        )

    wanted = {name_to_index[n] for n in class_names}
    targets = np.asarray(dataset.targets)
    indices = np.flatnonzero(np.isin(targets, list(wanted))).tolist()
    return indices, dataset


def unknown_loaders(
    root: Path | str,
    batch_size: int = 128,
    num_workers: int = 2,
    download: bool = True,
) -> dict[str, DataLoader]:
    """Build the fixed near and far unknown evaluation loaders."""
    loaders: dict[str, DataLoader] = {}
    for group, class_names in (("near", NEAR_UNKNOWNS), ("far", FAR_UNKNOWNS)):
        indices, dataset = cifar100_unknown_indices(root, class_names, download)
        loaders[group] = DataLoader(
            Subset(dataset, indices), batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=torch.cuda.is_available(),
        )
    return loaders


def unknown_class_names(root: Path | str, download: bool = False) -> dict[int, str]:
    """Map CIFAR-100 label indices to fine class names, for failure analysis."""
    dataset = datasets.CIFAR100(root=str(root), train=False, download=download)
    return {i: name for i, name in enumerate(dataset.classes)}
