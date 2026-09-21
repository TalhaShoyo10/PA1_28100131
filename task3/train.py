"""Training entry point for Task 3 domain generalization."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.config import apply_overrides, config_arg_parser, load_config
from common.logging import MetricHistory, RunRecord, get_logger
from common.seed import set_seed
from shared.pacs_protocol import (
    DomainBalancedIterator,
    assert_bn_frozen,
    build_source_loaders,
    load_split_manifest,
    train_mode_with_frozen_bn,
)
from task2.evaluation.metrics import (
    evaluate_source_domains,
    mean_source_accuracy,
    mean_source_macro_f1,
    worst_source_macro_f1,
)
from task2.models.backbone import PACSModel
from task3.methods.dan_dg import DANDG
from task3.methods.sam import SAM, SAMOptimizer

LOGGER = get_logger("task3.train")


def assert_no_target_access(cfg) -> None:
    """Fail loudly if a Task 3 config would let Sketch into training."""
    if not cfg.data.target_access_forbidden:
        raise RuntimeError(
            "Task 3 config has target_access_forbidden=false. No Sketch image "
            "may be loaded by training, source-side diagnostics, checkpoint "
            "selection, or hyperparameter selection."
        )
    if cfg.train.target_batch != 0:
        raise RuntimeError(
            f"Task 3 config requests target_batch={cfg.train.target_batch}; "
            "it must be 0."
        )


def build_method(cfg):
    """Instantiate the Task 3 method named by the config."""
    name = cfg.method.name
    if name == "dan_dg":
        return DANDG(
            lambda_dg=cfg.method.lambda_dg,
            multipliers=tuple(cfg.method.mmd.bandwidth_multipliers),
        )
    if name == "sam":
        return SAM(rho=cfg.method.rho)
    if name == "erm":
        raise RuntimeError(
            "ERM must not be retrained here. The assignment requires reusing "
            f"Task 2's Source-only checkpoint: {cfg.method.reuse_checkpoint_from}. "
            "Run task2/train.py --config task2/configs/source_only.yaml instead."
        )
    raise KeyError(f"Unknown Task 3 method {name!r}")


def forward_per_domain(model, batch, device):
    """Forward each source domain separately, keeping per-domain features."""
    domain_features: dict[str, torch.Tensor] = {}
    logits_list, labels_list = [], []

    for domain, (images, labels) in batch.items():
        logits, features = model(images.to(device), return_features=True)
        domain_features[domain] = features
        logits_list.append(logits)
        labels_list.append(labels.to(device))

    return torch.cat(logits_list), torch.cat(labels_list), domain_features


def train(cfg, smoke: bool = False) -> dict:
    """Train DAN-DG or SAM under the shared protocol, never touching Sketch."""
    assert_no_target_access(cfg)
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
    uses_sam = getattr(method, "requires_two_passes", False)

    if uses_sam:
        optimizer = SAMOptimizer(
            model.parameters(), torch.optim.AdamW, rho=cfg.method.rho,
            lr=cfg.train.lr, weight_decay=cfg.train.weight_decay,
        )
    else:
        optimizer = torch.optim.AdamW(
            list(model.parameters()) + list(method.parameters()),
            lr=cfg.train.lr, weight_decay=cfg.train.weight_decay,
        )

    source_iterator = DomainBalancedIterator(train_loaders)
    history = MetricHistory()
    checkpoint_dir = Path(cfg.output.checkpoint_dir) / cfg.run_name
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    best_score = -float("inf")
    best_epoch = -1
    epochs_without_improvement = 0

    for epoch in range(epochs):
        train_mode_with_frozen_bn(model)
        assert_bn_frozen(model)
        method.train()

        epoch_components: dict[str, float] = {}
        n_batches = 0

        for batch in source_iterator:
            if max_batches is not None and n_batches >= max_batches:
                break

            logits, labels, domain_features = forward_per_domain(model, batch, device)
            features = torch.cat([domain_features[d] for d in batch])

            kwargs = {"domain_features": domain_features} if cfg.method.name == "dan_dg" else {}
            output = method.compute_loss(features, logits, labels, **kwargs)

            if uses_sam:
                optimizer.zero_grad(set_to_none=True)
                output.loss.backward()
                optimizer.first_step(zero_grad=True)

                train_mode_with_frozen_bn(model)
                logits2, labels2, df2 = forward_per_domain(model, batch, device)
                features2 = torch.cat([df2[d] for d in batch])
                method.compute_loss(features2, logits2, labels2).loss.backward()
                optimizer.second_step(zero_grad=True)
            else:
                optimizer.zero_grad(set_to_none=True)
                output.loss.backward()
                optimizer.step()

            for key, value in output.components.items():
                epoch_components[key] = epoch_components.get(key, 0.0) + value
            n_batches += 1

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
                    "epoch": epoch,
                    "mean_source_val_macro_f1": score,
                    "config": dict(cfg),
                },
                checkpoint_dir / "best.pt",
            )
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= cfg.train.early_stopping_patience:
                LOGGER.info("Early stopping at epoch %d.", epoch)
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
        "target_seen_during_training": False,
        "smoke": smoke,
    }
    RunRecord(
        run_name=cfg.run_name, task="task3", config=dict(cfg), metrics=summary
    ).save(results_dir / "run.json")

    LOGGER.info("Best epoch %d with mean source val macro-F1 %.2f", best_epoch, best_score)
    return summary


def main() -> None:
    parser = config_arg_parser("Train a Task 3 domain-generalization method.")
    args = parser.parse_args()
    cfg = apply_overrides(load_config(args.config), args.overrides)
    train(cfg, smoke=args.smoke)


if __name__ == "__main__":
    main()
