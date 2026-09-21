"""Extract and cache features and logits once, so every score sees identical data."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.config import apply_overrides, config_arg_parser, load_config
from common.logging import get_logger, save_json
from common.seed import set_seed
from task4.data.cifar import cifar10_loaders, unknown_loaders
from task4.models.resnet_cifar import build_resnet_cifar
from task4.train import load_split

LOGGER = get_logger("task4.extract")


@torch.no_grad()
def extract_split(model, loader, device: str) -> dict[str, np.ndarray]:
    """Collect logits, penultimate features and labels for one split."""
    model.eval()
    logits_list, features_list, labels_list = [], [], []

    for images, labels in loader:
        logits, features = model(images.to(device), return_features=True)
        logits_list.append(logits.cpu().numpy())
        features_list.append(features.cpu().numpy())
        labels_list.append(labels.numpy())

    return {
        "logits": np.concatenate(logits_list),
        "features": np.concatenate(features_list),
        "labels": np.concatenate(labels_list),
    }


def load_trained_model(checkpoint_path: Path, device: str):
    """Restore a trained model, including any PROSER dummy classifiers."""
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}. Train it first with task4/train.py."
        )
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = build_resnet_cifar(
        num_classes=checkpoint.get("num_classes", 10),
        dummy_classifiers=checkpoint.get("dummy_classifiers", 0),
    ).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model, checkpoint


def main() -> None:
    parser = config_arg_parser("Cache Task 4 features and logits.")
    args = parser.parse_args()
    cfg = apply_overrides(load_config(args.config), args.overrides)
    set_seed(cfg.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, checkpoint = load_trained_model(
        Path(cfg.output.checkpoint_dir) / cfg.run_name / "best.pt", device
    )
    LOGGER.info(
        "Loaded %s from epoch %d (val acc %.2f), dummies=%d",
        cfg.run_name, checkpoint["epoch"], checkpoint["val_accuracy"],
        model.dummy_classifiers,
    )

    manifest = load_split(cfg)
    known_loaders = cifar10_loaders(
        cfg.data.known.root, manifest, batch_size=256,
        num_workers=cfg.train.num_workers, seed=cfg.seed,
    )
    unknowns = unknown_loaders(
        cfg.data.unknown.root, batch_size=256, num_workers=cfg.train.num_workers
    )

    cache_dir = Path(cfg.output.cache_dir) / cfg.run_name
    cache_dir.mkdir(parents=True, exist_ok=True)

    splits = {
        "cifar10_train": known_loaders["train_unaugmented"],
        "cifar10_val": known_loaders["val"],
        "cifar10_test": known_loaders["test"],
        "cifar100_near": unknowns["near"],
        "cifar100_far": unknowns["far"],
    }

    summary: dict[str, dict] = {}
    for name, loader in splits.items():
        outputs = extract_split(model, loader, device)
        np.savez_compressed(cache_dir / f"{name}.npz", **outputs)
        summary[name] = {
            "n": int(len(outputs["labels"])),
            "logit_dim": int(outputs["logits"].shape[1]),
            "feature_dim": int(outputs["features"].shape[1]),
        }
        LOGGER.info("%-16s n=%5d logits=%d", name, summary[name]["n"], summary[name]["logit_dim"])

    for group, expected in (("near", cfg.data.unknown.expected_per_group),
                            ("far", cfg.data.unknown.expected_per_group)):
        actual = summary[f"cifar100_{group}"]["n"]
        if actual != expected:
            LOGGER.warning(
                "%s unknown group has %d images, expected %d.", group, actual, expected
            )

    save_json(
        {
            "run_name": cfg.run_name,
            "checkpoint_epoch": checkpoint["epoch"],
            "checkpoint_val_accuracy": checkpoint["val_accuracy"],
            "dummy_classifiers": model.dummy_classifiers,
            "splits": summary,
            "cache_dir": str(cache_dir),
        },
        cache_dir / "extraction.json",
    )
    LOGGER.info("Cached outputs for %d splits to %s", len(splits), cache_dir)


if __name__ == "__main__":
    main()
