"""Shared training loop for Task 2 adaptation methods."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.config import apply_overrides, config_arg_parser, load_config
from common.logging import MetricHistory, RunRecord, get_logger, save_csv
from common.seed import set_seed
from shared.pacs_protocol import (
    DomainBalancedIterator,
    InfiniteLoader,
    assert_bn_frozen,
    build_source_loaders,
    build_target_loader,
    load_split_manifest,
    train_mode_with_frozen_bn,
)
from task2.evaluation.metrics import (
    evaluate_source_domains,
    mean_source_accuracy,
    mean_source_macro_f1,
    worst_source_macro_f1,
)
from task2.methods.cdan import CDAN
from task2.methods.dan import DAN
from task2.methods.dann import DANN
from task2.methods.source_only import SourceOnly
from task2.models.backbone import PACSModel

LOGGER = get_logger("task2.train")


def build_method(cfg):
    """Instantiate the adaptation method named by the config."""
    name = cfg.method.name
    if name == "source_only":
        return SourceOnly()
    if name == "dan":
        return DAN(
            lambda_mmd=cfg.method.lambda_mmd,
            multipliers=tuple(cfg.method.mmd.bandwidth_multipliers),
        )
    if name == "dann":
        return DANN(
            feature_dim=cfg.model.feature_dim,
            hidden_dim=cfg.method.discriminator.hidden_dim,
            dropout=cfg.method.discriminator.dropout,
            gamma=cfg.method.grl.gamma,
            alpha_max=cfg.method.grl.alpha_max,
            lambda_domain=cfg.method.lambda_domain,
        )
    if name == "cdan":
        return CDAN(
            feature_dim=cfg.model.feature_dim,
            num_classes=cfg.data.num_classes,
            hidden_dim=cfg.method.discriminator.hidden_dim,
            dropout=cfg.method.discriminator.dropout,
            gamma=cfg.method.grl.gamma,
            alpha_max=cfg.method.grl.alpha_max,
            lambda_domain=cfg.method.lambda_domain,
        )
    raise KeyError(f"Unknown method {name!r}")


def flatten_source_batch(batch: dict, device: str):
    """Concatenate per-domain batches into one pooled source tensor pair."""
    images = torch.cat([images for images, _ in batch.values()]).to(device)
    labels = torch.cat([labels for _, labels in batch.values()]).to(device)
    return images, labels


def train(cfg, smoke: bool = False) -> dict:
    """Train one method under the shared protocol and save the best checkpoint."""
    set_seed(cfg.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    LOGGER.info("Method=%s device=%s seed=%d", cfg.method.name, device, cfg.seed)

    epochs = cfg.smoke.epochs if smoke else cfg.train.epochs
    max_batches = cfg.smoke.max_batches_per_epoch if smoke else None

    manifest = load_split_manifest(cfg.data.split_manifest)
    sources = list(cfg.data.sources)

    train_loaders = build_source_loaders(
        cfg.data.root, manifest, sources, "train",
        batch_size=cfg.train.source_batch_per_domain, train=True,
        resize=cfg.data.resize, crop=cfg.data.crop, hflip=cfg.data.hflip,
        num_workers=cfg.train.num_workers, seed=cfg.seed,
    )
    val_loaders = build_source_loaders(
        cfg.data.root, manifest, sources, "val",
        batch_size=64, train=False,
        resize=cfg.data.resize, crop=cfg.data.crop,
        num_workers=cfg.train.num_workers, seed=cfg.seed,
    )

    model = PACSModel(num_classes=cfg.data.num_classes, pretrained=True).to(device)
    method = build_method(cfg).to(device)

    method_parameters = list(method.parameters())
    param_groups = [{"params": list(model.parameters()), "lr": cfg.train.lr}]
    if method_parameters:
        param_groups.append(
            {
                "params": method_parameters,
                "lr": cfg.train.lr * cfg.train.discriminator_lr_multiplier,
            }
        )
    optimizer = torch.optim.AdamW(param_groups, weight_decay=cfg.train.weight_decay)

    target_stream = None
    if method.uses_target:
        target_stream = InfiniteLoader(
            build_target_loader(
                cfg.data.root, manifest, cfg.data.target,
                batch_size=cfg.train.target_batch, train=True,
                resize=cfg.data.resize, crop=cfg.data.crop, hflip=cfg.data.hflip,
                num_workers=cfg.train.num_workers, seed=cfg.seed,
            )
        )

    source_iterator = DomainBalancedIterator(train_loaders)
    steps_per_epoch = min(len(source_iterator), max_batches or len(source_iterator))
    total_steps = max(1, steps_per_epoch * epochs)

    history = MetricHistory()
    checkpoint_dir = Path(cfg.output.checkpoint_dir) / cfg.run_name
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    best_score = -float("inf")
    best_epoch = -1
    epochs_without_improvement = 0
    global_step = 0

    for epoch in range(epochs):
        train_mode_with_frozen_bn(model)
        assert_bn_frozen(model)
        method.train()

        epoch_components: dict[str, float] = {}
        n_batches = 0

        for batch in source_iterator:
            if max_batches is not None and n_batches >= max_batches:
                break

            source_images, source_labels = flatten_source_batch(batch, device)
            source_logits, source_features = model(source_images, return_features=True)

            target_features = target_logits = None
            if method.uses_target:
                target_images, _ = target_stream.next()
                target_logits, target_features = model(
                    target_images.to(device), return_features=True
                )

            progress = min(1.0, global_step / total_steps)
            output = method.compute_loss(
                source_features, source_logits, source_labels,
                target_features=target_features,
                target_logits=target_logits,
                progress=progress,
            )

            optimizer.zero_grad(set_to_none=True)
            output.loss.backward()
            optimizer.step()

            for key, value in output.components.items():
                epoch_components[key] = epoch_components.get(key, 0.0) + value
            n_batches += 1
            global_step += 1

        averages = {k: v / max(1, n_batches) for k, v in epoch_components.items()}

        evaluations = evaluate_source_domains(
            model, val_loaders, cfg.data.num_classes, device
        )
        score = mean_source_macro_f1(evaluations)
        worst_domain, worst_f1 = worst_source_macro_f1(evaluations)

        row = {
            "epoch": epoch,
            "steps": n_batches,
            **averages,
            "mean_source_val_macro_f1": score,
            "mean_source_val_accuracy": mean_source_accuracy(evaluations),
            "worst_source_domain": worst_domain,
            "worst_source_macro_f1": worst_f1,
        }
        for domain, evaluation in evaluations.items():
            row[f"val_macro_f1_{domain}"] = evaluation.macro_f1
            row[f"val_accuracy_{domain}"] = evaluation.accuracy
        history.append(**row)

        LOGGER.info(
            "epoch %02d | %s | mean val macro-F1 %.2f | worst %s %.2f",
            epoch,
            " ".join(f"{k}={v:.4f}" for k, v in averages.items()),
            score, worst_domain, worst_f1,
        )

        if score > best_score:
            best_score, best_epoch = score, epoch
            epochs_without_improvement = 0
            torch.save(
                {
                    "model": model.state_dict(),
                    "method": method.state_dict(),
                    "epoch": epoch,
                    "mean_source_val_macro_f1": score,
                    "config": dict(cfg),
                },
                checkpoint_dir / "best.pt",
            )
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= cfg.train.early_stopping_patience:
                LOGGER.info(
                    "Early stopping at epoch %d (no improvement for %d epochs).",
                    epoch, cfg.train.early_stopping_patience,
                )
                break

    results_dir = Path(cfg.output.results_dir) / cfg.run_name
    history.save(results_dir / "training_curve.csv")

    summary = {
        "run_name": cfg.run_name,
        "method": cfg.method.name,
        "best_epoch": best_epoch,
        "best_mean_source_val_macro_f1": best_score,
        "epochs_run": len(history),
        "checkpoint": str(checkpoint_dir / "best.pt"),
        "smoke": smoke,
    }
    RunRecord(
        run_name=cfg.run_name, task="task2", config=dict(cfg), metrics=summary
    ).save(results_dir / "run.json")

    LOGGER.info("Best epoch %d with mean source val macro-F1 %.2f", best_epoch, best_score)
    return summary


def main() -> None:
    parser = config_arg_parser("Train a Task 2 adaptation method.")
    args = parser.parse_args()
    cfg = apply_overrides(load_config(args.config), args.overrides)
    train(cfg, smoke=args.smoke)


if __name__ == "__main__":
    main()
