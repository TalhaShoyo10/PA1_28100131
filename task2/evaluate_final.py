"""Final Task 2 evaluation: target metrics, separability, per-class transfer."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.config import apply_overrides, config_arg_parser, load_config
from common.logging import get_logger, save_csv, save_json
from common.seed import set_seed
from shared.pacs import PACS_CLASSES
from shared.pacs_protocol import (
    build_source_loaders,
    build_target_loader,
    load_split_manifest,
)
from task2.evaluation.domain_separability import source_target_separability
from task2.evaluation.metrics import (
    confusion_rows,
    evaluate_domain,
    evaluate_source_domains,
    mean_source_accuracy,
    mean_source_macro_f1,
    per_class_delta,
    worst_source_macro_f1,
)
from task2.models.backbone import PACSModel

LOGGER = get_logger("task2.evaluate")


def load_checkpoint(path: Path, num_classes: int, device: str) -> PACSModel:
    """Load a trained model from a saved checkpoint."""
    if not path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {path}. Train it first with task2/train.py."
        )
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model = PACSModel(num_classes=num_classes, pretrained=False).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model


def evaluate_run(cfg, run_name: str, device: str) -> dict:
    """Evaluate one trained method on sources and the target domain.

    Target labels are used ONLY here, after every checkpoint and setting has
    been frozen.
    """
    manifest = load_split_manifest(cfg.data.split_manifest)
    sources = list(cfg.data.sources)

    model = load_checkpoint(
        Path(cfg.output.checkpoint_dir) / run_name / "best.pt",
        cfg.data.num_classes, device,
    )

    val_loaders = build_source_loaders(
        cfg.data.root, manifest, sources, "val", batch_size=64, train=False,
        resize=cfg.data.resize, crop=cfg.data.crop,
        num_workers=cfg.train.num_workers, seed=cfg.seed,
    )
    source_evaluations = evaluate_source_domains(
        model, val_loaders, cfg.data.num_classes, device, return_features=True
    )

    target_loader = build_target_loader(
        cfg.data.root, manifest, cfg.data.target, batch_size=64, train=False,
        resize=cfg.data.resize, crop=cfg.data.crop,
        num_workers=cfg.train.num_workers, seed=cfg.seed,
    )
    target_evaluation = evaluate_domain(
        model, target_loader, cfg.data.target, cfg.data.num_classes, device,
        return_features=True,
    )

    source_features = torch.cat(
        [torch.tensor(e.features) for e in source_evaluations.values()]
    ).numpy()
    separability = source_target_separability(
        source_features, target_evaluation.features,
        seed=cfg.evaluation.domain_separability.split_seed,
        test_fraction=cfg.evaluation.domain_separability.test_fraction,
        C=cfg.evaluation.domain_separability.C,
    )

    worst_domain, worst_f1 = worst_source_macro_f1(source_evaluations)
    return {
        "run_name": run_name,
        "source": source_evaluations,
        "target": target_evaluation,
        "mean_source_accuracy": mean_source_accuracy(source_evaluations),
        "mean_source_macro_f1": mean_source_macro_f1(source_evaluations),
        "worst_source_domain": worst_domain,
        "worst_source_macro_f1": worst_f1,
        "target_accuracy": target_evaluation.accuracy,
        "target_macro_f1": target_evaluation.macro_f1,
        "domain_separability": separability["separability"],
        "separability_chance": separability["chance"],
    }


def main() -> None:
    parser = config_arg_parser("Final Task 2 evaluation across all methods.")
    parser.add_argument(
        "--runs", nargs="*", default=["source_only", "dan", "dann", "cdan"]
    )
    args = parser.parse_args()

    cfg = apply_overrides(load_config(args.config), args.overrides)
    set_seed(cfg.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    class_names = list(PACS_CLASSES)

    evaluations: dict[str, dict] = {}
    table: list[dict] = []

    for run_name in args.runs:
        LOGGER.info("Evaluating %s", run_name)
        result = evaluate_run(cfg, run_name, device)
        evaluations[run_name] = result

        row = {"method": run_name}
        for domain, evaluation in result["source"].items():
            row[f"{domain}_accuracy"] = evaluation.accuracy
            row[f"{domain}_macro_f1"] = evaluation.macro_f1
        row.update(
            {
                "mean_source_accuracy": result["mean_source_accuracy"],
                "mean_source_macro_f1": result["mean_source_macro_f1"],
                "worst_source_domain": result["worst_source_domain"],
                "worst_source_macro_f1": result["worst_source_macro_f1"],
                "target_accuracy": result["target_accuracy"],
                "target_macro_f1": result["target_macro_f1"],
                "domain_separability": result["domain_separability"],
            }
        )
        table.append(row)

    baseline = evaluations.get("source_only")
    if baseline is not None:
        for row in table:
            row["target_accuracy_delta"] = (
                row["target_accuracy"] - baseline["target_accuracy"]
            )

        results_dir = Path(cfg.output.results_dir)
        for run_name, result in evaluations.items():
            if run_name == "source_only":
                continue
            deltas = per_class_delta(baseline["target"], result["target"], class_names)
            save_csv(deltas, results_dir / f"per_class_delta_{run_name}.csv")
            LOGGER.info(
                "%s: worst class %s (%.1f), best class %s (+%.1f)",
                run_name, deltas[0]["class_name"], deltas[0]["delta"],
                deltas[-1]["class_name"], deltas[-1]["delta"],
            )

    results_dir = Path(cfg.output.results_dir)
    save_csv(table, results_dir / "final_comparison.csv")

    for run_name, result in evaluations.items():
        save_csv(
            confusion_rows(result["target"], class_names),
            results_dir / f"target_confusions_{run_name}.csv",
        )
        save_json(
            {
                "per_class_target_accuracy": {
                    class_names[k]: v for k, v in result["target"].per_class.items()
                },
                "domain_separability": result["domain_separability"],
                "separability_chance": result["separability_chance"],
            },
            results_dir / f"target_detail_{run_name}.json",
        )

    LOGGER.info("Wrote final comparison for %d methods", len(table))


if __name__ == "__main__":
    main()
