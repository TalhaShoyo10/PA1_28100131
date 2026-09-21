"""The single shared PACS protocol for Tasks 2 and 3."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator, Sequence

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from common.seed import ASSIGNMENT_SEED, make_generator, seed_worker
from shared.pacs import PACS_CLASSES, PACSDataset, list_domain_images

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def stratified_split(
    labels: Sequence[int], val_fraction: float, seed: int
) -> tuple[list[int], list[int]]:
    """Split indices into (train, val), stratified by label."""
    if not 0.0 < val_fraction < 1.0:
        raise ValueError(f"val_fraction must be in (0,1), got {val_fraction}")

    labels = np.asarray(labels)
    rng = np.random.default_rng(seed)

    train_idx: list[int] = []
    val_idx: list[int] = []

    for cls in sorted(set(labels.tolist())):
        cls_indices = np.flatnonzero(labels == cls)
        rng.shuffle(cls_indices)

        n = len(cls_indices)
        n_val = int(round(n * val_fraction))
        if n >= 2:
            n_val = max(1, min(n - 1, n_val))
        else:
            n_val = 0

        val_idx.extend(cls_indices[:n_val].tolist())
        train_idx.extend(cls_indices[n_val:].tolist())

    return sorted(train_idx), sorted(val_idx)


def build_split_manifest(
    root: Path | str,
    sources: Sequence[str],
    target: str,
    val_fraction: float = 0.2,
    seed: int = ASSIGNMENT_SEED,
) -> dict:
    """Build the committed split manifest shared by Tasks 2 and 3."""
    root = Path(root)
    manifest: dict = {
        "dataset": "PACS",
        "seed": seed,
        "val_fraction": val_fraction,
        "classes": list(PACS_CLASSES),
        "sources": list(sources),
        "target": target,
        "domains": {},
    }

    for domain in sources:
        items = list_domain_images(root, domain)
        labels = [label for _, label in items]
        train_idx, val_idx = stratified_split(labels, val_fraction, seed)
        manifest["domains"][domain] = {
            "role": "source",
            "n_total": len(items),
            "train": [items[i] for i in train_idx],
            "val": [items[i] for i in val_idx],
        }

    target_items = list_domain_images(root, target)
    manifest["domains"][target] = {
        "role": "target",
        "n_total": len(target_items),
        "all": target_items,
        "note": (
            "Task 2: unlabeled adaptation set (labels used only for final "
            "evaluation after all decisions are frozen). "
            "Task 3: not loaded at all until evaluate_sketch.py."
        ),
    }
    return manifest


def save_split_manifest(manifest: dict, path: Path | str) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return path


def load_split_manifest(path: Path | str) -> dict:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Split manifest not found: {path}. Generate it first with "
            "`python shared/make_splits.py --config task2/configs/base.yaml`."
        )
    return json.loads(path.read_text(encoding="utf-8"))


def train_transform(resize: int = 256, crop: int = 224, hflip: bool = True):
    """Resize 256 -> random 224 crop -> optional horizontal flip -> normalize."""
    ops = [transforms.Resize((resize, resize)), transforms.RandomCrop(crop)]
    if hflip:
        ops.append(transforms.RandomHorizontalFlip())
    ops += [transforms.ToTensor(), transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)]
    return transforms.Compose(ops)


def eval_transform(resize: int = 256, crop: int = 224):
    """Resize 256 -> center 224 crop -> normalize. Used for val and test."""
    return transforms.Compose(
        [
            transforms.Resize((resize, resize)),
            transforms.CenterCrop(crop),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


def make_loader(
    dataset: Dataset,
    batch_size: int,
    shuffle: bool,
    num_workers: int = 2,
    seed: int = ASSIGNMENT_SEED,
    drop_last: bool = False,
) -> DataLoader:
    """DataLoader with reproducible shuffling and worker seeding."""
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=drop_last,
        generator=make_generator(seed) if shuffle else None,
        worker_init_fn=seed_worker if num_workers > 0 else None,
    )


class DomainBalancedIterator:
    """Yields one batch per source domain, cycling shorter loaders."""

    def __init__(self, loaders: dict[str, DataLoader]) -> None:
        if not loaders:
            raise ValueError("DomainBalancedIterator requires at least one loader.")
        self.loaders = loaders
        self._length = max(len(dl) for dl in loaders.values())

    def __len__(self) -> int:
        return self._length

    def __iter__(self) -> Iterator[dict[str, tuple[torch.Tensor, torch.Tensor]]]:
        iterators = {name: iter(dl) for name, dl in self.loaders.items()}
        for _ in range(self._length):
            batch: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
            for name, loader in self.loaders.items():
                try:
                    batch[name] = next(iterators[name])
                except StopIteration:
                    iterators[name] = iter(loader)
                    batch[name] = next(iterators[name])
            yield batch


class InfiniteLoader:
    """Endlessly cycles a loader. Used for the Task 2 unlabeled target stream."""

    def __init__(self, loader: DataLoader) -> None:
        self.loader = loader
        self._iterator = iter(loader)

    def next(self) -> tuple[torch.Tensor, torch.Tensor]:
        try:
            return next(self._iterator)
        except StopIteration:
            self._iterator = iter(self.loader)
            return next(self._iterator)


def build_source_loaders(
    root: Path | str,
    manifest: dict,
    sources: Sequence[str],
    split: str,
    batch_size: int,
    train: bool,
    resize: int = 256,
    crop: int = 224,
    hflip: bool = True,
    num_workers: int = 2,
    seed: int = ASSIGNMENT_SEED,
) -> dict[str, DataLoader]:
    """One loader per source domain for ``split`` ('train' or 'val')."""
    tfm = (
        train_transform(resize, crop, hflip) if train else eval_transform(resize, crop)
    )
    loaders: dict[str, DataLoader] = {}
    for domain in sources:
        samples = [tuple(s) for s in manifest["domains"][domain][split]]
        ds = PACSDataset(root, samples, transform=tfm, domain=domain)
        loaders[domain] = make_loader(
            ds,
            batch_size=batch_size,
            shuffle=train,
            num_workers=num_workers,
            seed=seed,
            drop_last=train,
        )
    return loaders


def build_target_loader(
    root: Path | str,
    manifest: dict,
    target: str,
    batch_size: int,
    train: bool,
    resize: int = 256,
    crop: int = 224,
    hflip: bool = True,
    num_workers: int = 2,
    seed: int = ASSIGNMENT_SEED,
    target_access_forbidden: bool = False,
) -> DataLoader:
    """Loader over the complete target domain."""
    tfm = (
        train_transform(resize, crop, hflip) if train else eval_transform(resize, crop)
    )
    samples = [tuple(s) for s in manifest["domains"][target]["all"]]
    ds = PACSDataset(
        root,
        samples,
        transform=tfm,
        domain=target,
        target_access_forbidden=target_access_forbidden,
        forbidden_domain=target,
    )
    return make_loader(
        ds,
        batch_size=batch_size,
        shuffle=train,
        num_workers=num_workers,
        seed=seed,
        drop_last=train,
    )


def set_bn_eval(model: nn.Module) -> int:
    """Put every BatchNorm module in eval mode, leaving the rest training."""
    count = 0
    for module in model.modules():
        if isinstance(module, nn.modules.batchnorm._BatchNorm):
            module.eval()
            count += 1
    return count


def train_mode_with_frozen_bn(model: nn.Module) -> nn.Module:
    """``model.train()`` followed by the frozen-BN policy."""
    model.train()
    set_bn_eval(model)
    return model


def assert_bn_frozen(model: nn.Module) -> None:
    """Raise if any BatchNorm module would update its running statistics."""
    offenders = [
        name
        for name, module in model.named_modules()
        if isinstance(module, nn.modules.batchnorm._BatchNorm) and module.training
    ]
    if offenders:
        raise RuntimeError(
            "Frozen-BatchNorm policy violated: these BN modules are in training "
            f"mode and would update running statistics: {offenders[:5]}"
            f"{' ...' if len(offenders) > 5 else ''}. "
            "Call train_mode_with_frozen_bn(model) instead of model.train()."
        )
