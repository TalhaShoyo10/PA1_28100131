"""Training entry point for Task 4: Vanilla, GCSC and PROSER."""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.config import apply_overrides, config_arg_parser, load_config
from common.logging import MetricHistory, RunRecord, get_logger, save_json
from common.metrics import accuracy
from common.seed import set_seed
from task4.data.cifar import build_cifar10_splits, cifar10_loaders
from task4.methods.proser import PROSER
from task4.models.resnet_cifar import build_resnet_cifar

LOGGER = get_logger("task4.train")


@torch.no_grad()
def validate(model, loader, device: str, num_classes: int = 10) -> float:
    """CIFAR-10 validation accuracy using the known-class logits only."""
    model.eval()
    correct = total = 0
    for images, labels in loader:
        logits = model(images.to(device))[:, :num_classes]
        correct += int((logits.argmax(1).cpu() == labels).sum())
        total += len(labels)
    return 100.0 * correct / max(1, total)


def load_split(cfg) -> dict:
    """Load the committed CIFAR-10 split, generating it on first use."""
    manifest_path = Path(cfg.data.known.manifest)
    if manifest_path.exists():
        import json

        return json.loads(manifest_path.read_text(encoding="utf-8"))

    manifest = build_cifar10_splits(
        cfg.data.known.root, cfg.data.known.val_fraction, cfg.data.known.split_seed
    )
    save_json(manifest, manifest_path)
    LOGGER.info("Created CIFAR-10 split: %d train, %d val", manifest["n_train"], manifest["n_val"])
    return manifest


def train(cfg, smoke: bool = False) -> dict:
    """Train one Task 4 model and save the best CIFAR-10 validation checkpoint."""
    set_seed(cfg.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    method_name = cfg.method.name
    LOGGER.info("Method=%s device=%s seed=%d", method_name, device, cfg.seed)

    epochs = cfg.smoke.epochs if smoke else cfg.train.epochs
    max_batches = cfg.smoke.max_batches_per_epoch if smoke else None

    manifest = load_split(cfg)
    if smoke:
        manifest = {
            **manifest,
            "train": manifest["train"][: cfg.smoke.limit_train_images],
            "val": manifest["val"][: cfg.smoke.limit_eval_images],
        }

    rand_augment = None
    augmentation = cfg.train.augmentation
    if "rand_augment" in dict.keys(augmentation):
        rand_augment = dict(augmentation.rand_augment)
        LOGGER.info("RandAugment enabled: %s", rand_augment)

    loaders = cifar10_loaders(
        cfg.data.known.root, manifest,
        batch_size=cfg.train.batch_size, rand_augment=rand_augment,
        num_workers=cfg.train.num_workers, seed=cfg.seed,
    )

    model = build_resnet_cifar(cfg.model.num_classes).to(device)
    proser = None

    if method_name == "proser":
        checkpoint_path = Path(cfg.method.init_from)
        if not checkpoint_path.exists():
            raise FileNotFoundError(
                f"PROSER initializes from the selected Vanilla checkpoint, not "
                f"from scratch. Missing: {checkpoint_path}. Train vanilla first."
            )
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model"])
        model.add_dummy_classifiers(cfg.method.dummy_classifiers)
        model = model.to(device)
        proser = PROSER(
            model,
            beta=cfg.method.classifier_placeholder.beta,
            gamma=cfg.method.data_placeholder.gamma,
            lam_alpha=cfg.method.data_placeholder.lam_distribution.alpha,
            lam_beta=cfg.method.data_placeholder.lam_distribution.beta,
            seed=cfg.seed,
        )
        LOGGER.info(
            "Initialized from %s (val acc %.2f), added %d dummy classifiers",
            checkpoint_path, checkpoint.get("val_accuracy", float("nan")),
            cfg.method.dummy_classifiers,
        )

    optimizer = torch.optim.SGD(
        model.parameters(), lr=cfg.train.lr, momentum=cfg.train.momentum,
        weight_decay=cfg.train.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    history = MetricHistory()
    checkpoint_dir = Path(cfg.output.checkpoint_dir) / cfg.run_name
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    best_accuracy = -1.0
    best_epoch = -1

    for epoch in range(epochs):
        model.train()
        epoch_components: dict[str, float] = {}
        n_batches = 0

        for images, labels in loaders["train"]:
            if max_batches is not None and n_batches >= max_batches:
                break
            images, labels = images.to(device), labels.to(device)

            if proser is not None:
                loss, components = proser.compute_loss(images, labels)
            else:
                loss = F.cross_entropy(model(images), labels)
                components = {"cls_loss": float(loss.detach())}

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            for key, value in components.items():
                epoch_components[key] = epoch_components.get(key, 0.0) + value
            n_batches += 1

        scheduler.step()
        averages = {k: v / max(1, n_batches) for k, v in epoch_components.items()}
        val_accuracy = validate(model, loaders["val"], device, cfg.model.num_classes)

        history.append(
            epoch=epoch, steps=n_batches, lr=scheduler.get_last_lr()[0],
            **averages, val_accuracy=val_accuracy,
        )

        if epoch % 10 == 0 or epoch == epochs - 1 or smoke:
            LOGGER.info(
                "epoch %03d | %s | val acc %.2f",
                epoch, " ".join(f"{k}={v:.4f}" for k, v in averages.items()), val_accuracy,
            )

        if val_accuracy > best_accuracy:
            best_accuracy, best_epoch = val_accuracy, epoch
            torch.save(
                {
                    "model": model.state_dict(),
                    "epoch": epoch,
                    "val_accuracy": val_accuracy,
                    "num_classes": cfg.model.num_classes,
                    "dummy_classifiers": model.dummy_classifiers,
                    "config": dict(cfg),
                },
                checkpoint_dir / "best.pt",
            )

    results_dir = Path(cfg.output.results_dir) / cfg.run_name
    history.save(results_dir / "training_curve.csv")

    summary = {
        "run_name": cfg.run_name,
        "method": method_name,
        "best_epoch": best_epoch,
        "best_val_accuracy": best_accuracy,
        "epochs_run": len(history),
        "checkpoint": str(checkpoint_dir / "best.pt"),
        "rand_augment": rand_augment,
        "smoke": smoke,
    }
    RunRecord(run_name=cfg.run_name, task="task4", config=dict(cfg), metrics=summary).save(
        results_dir / "run.json"
    )

    LOGGER.info("Best epoch %d with CIFAR-10 val accuracy %.2f", best_epoch, best_accuracy)
    return summary


def main() -> None:
    parser = config_arg_parser("Train a Task 4 model.")
    args = parser.parse_args()
    cfg = apply_overrides(load_config(args.config), args.overrides)
    train(cfg, smoke=args.smoke)


if __name__ == "__main__":
    main()
