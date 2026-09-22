"""Final Task 3 evaluation. The ONLY script permitted to load Sketch.

Every Task 3 training, diagnostic and checkpoint-selection decision must
already be frozen before this runs. Nothing here may feed back into a Task 3
setting.
"""

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
from task2.evaluation.domain_separability import source_domain_separability
from task2.evaluation.metrics import (
    confusion_rows,
    evaluate_domain,
    evaluate_source_domains,
    mean_source_accuracy,
    mean_source_macro_f1,
    per_class_delta,
    worst_source_accuracy,
    worst_source_macro_f1,
)
from task2.models.backbone import PACSModel
from task3.methods.sam import sharpness_proxy

LOGGER = get_logger("task3.evaluate")


def resolve_checkpoint(cfg, run_name: str) -> Path:
    """Locate a run's checkpoint, mapping ERM back to Task 2's Source-only.

    This script is normally invoked with ``base.yaml``, which carries no
    ``method:`` block, so the ERM path is resolved from candidates rather than
    from ``cfg.method``. Task 3's checkpoint directory is checked first because
    an override such as ``--set output.checkpoint_dir=...`` must be honoured;
    the repository-relative default is the final fallback.
    """
    if run_name != "erm":
        return Path(cfg.output.checkpoint_dir) / run_name / "best.pt"

    candidates: list[Path] = []
    configured = dict(cfg).get("method")
    if isinstance(configured, dict) and "reuse_checkpoint_from" in configured:
        candidates.append(Path(configured["reuse_checkpoint_from"]))

    task3_dir = Path(cfg.output.checkpoint_dir)
    candidates.append(task3_dir.parent / "task2" / "source_only" / "best.pt")
    candidates.append(Path("checkpoints/task2/source_only/best.pt"))

    for path in candidates:
        if path.exists():
            return path

    raise FileNotFoundError(
        "ERM baseline checkpoint not found. Task 3 reuses Task 2's Source-only "
        "model rather than retraining it; train that first with "
        "`python task2/train.py --config task2/configs/source_only.yaml`. "
        f"Looked in: {[str(p) for p in candidates]}"
    )


def load_model(path: Path, num_classes: int, device: str) -> PACSModel:
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model = PACSModel(num_classes=num_classes, pretrained=False).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model


def fixed_sharpness_batch(cfg, manifest, device: str):
    """Build the fixed validation batch used for the sharpness proxy.

    The same 32 examples per source domain, seed 6304, for every model, so
    delta_sharp values are comparable across ERM, DAN-DG and SAM.
    """
    settings = cfg.evaluation.sharpness_proxy
    loaders = build_source_loaders(
        cfg.data.root, manifest, list(cfg.data.sources), "val",
        batch_size=settings.batch_per_domain, train=False,
        resize=cfg.data.resize, crop=cfg.data.crop,
        num_workers=0, seed=settings.batch_seed,
    )

    images, labels = [], []
    for loader in loaders.values():
        batch_images, batch_labels = next(iter(loader))
        images.append(batch_images)
        labels.append(batch_labels)

    return torch.cat(images).to(device), torch.cat(labels).to(device)


def main() -> None:
    parser = config_arg_parser("Final Task 3 evaluation on the unseen Sketch domain.")
    parser.add_argument("--runs", nargs="*", default=["erm", "dan_dg", "sam"])
    args = parser.parse_args()

    cfg = apply_overrides(load_config(args.config), args.overrides)
    set_seed(cfg.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    class_names = list(PACS_CLASSES)

    LOGGER.warning(
        "Loading the Sketch domain. All Task 3 training, diagnostics and "
        "checkpoint selection must already be frozen."
    )

    manifest = load_split_manifest(cfg.data.split_manifest)
    sources = list(cfg.data.sources)

    val_loaders = build_source_loaders(
        cfg.data.root, manifest, sources, "val", batch_size=64, train=False,
        resize=cfg.data.resize, crop=cfg.data.crop,
        num_workers=cfg.train.num_workers, seed=cfg.seed,
    )
    sketch_loader = build_target_loader(
        cfg.data.root, manifest, cfg.data.target, batch_size=64, train=False,
        resize=cfg.data.resize, crop=cfg.data.crop,
        num_workers=cfg.train.num_workers, seed=cfg.seed,
        target_access_forbidden=False,
    )
    sharpness_images, sharpness_labels = fixed_sharpness_batch(cfg, manifest, device)

    evaluations: dict[str, dict] = {}
    table: list[dict] = []

    for run_name in args.runs:
        LOGGER.info("Evaluating %s", run_name)
        model = load_model(
            resolve_checkpoint(cfg, run_name), cfg.data.num_classes, device
        )

        source_evaluations = evaluate_source_domains(
            model, val_loaders, cfg.data.num_classes, device, return_features=True
        )
        sketch_evaluation = evaluate_domain(
            model, sketch_loader, cfg.data.target, cfg.data.num_classes, device
        )

        separability = source_domain_separability(
            {d: e.features for d, e in source_evaluations.items()},
            seed=cfg.evaluation.source_domain_separability.split_seed,
            test_fraction=cfg.evaluation.source_domain_separability.test_fraction,
            C=cfg.evaluation.source_domain_separability.C,
        )
        sharpness = sharpness_proxy(
            model, sharpness_images, sharpness_labels,
            rho=cfg.evaluation.sharpness_proxy.rho,
        )

        worst_f1_domain, worst_f1 = worst_source_macro_f1(source_evaluations)
        worst_acc_domain, worst_acc = worst_source_accuracy(source_evaluations)

        evaluations[run_name] = {
            "source": source_evaluations,
            "sketch": sketch_evaluation,
            "separability": separability,
            "sharpness": sharpness,
        }

        row = {"method": run_name}
        for domain, evaluation in source_evaluations.items():
            row[f"{domain}_accuracy"] = evaluation.accuracy
            row[f"{domain}_macro_f1"] = evaluation.macro_f1
        row.update(
            {
                "mean_source_accuracy": mean_source_accuracy(source_evaluations),
                "mean_source_macro_f1": mean_source_macro_f1(source_evaluations),
                "worst_source_domain": worst_f1_domain,
                "worst_source_macro_f1": worst_f1,
                "worst_source_accuracy_domain": worst_acc_domain,
                "worst_source_accuracy": worst_acc,
                "sketch_accuracy": sketch_evaluation.accuracy,
                "sketch_macro_f1": sketch_evaluation.macro_f1,
                "source_domain_separability": separability["separability"],
                "separability_chance": separability["chance"],
                "delta_sharp": sharpness["delta_sharp"],
            }
        )
        table.append(row)

        LOGGER.info(
            "%s: sketch acc %.2f | separability %.2f (chance %.1f) | delta_sharp %.5f",
            run_name, sketch_evaluation.accuracy,
            separability["separability"], separability["chance"],
            sharpness["delta_sharp"],
        )

    baseline = evaluations.get("erm")
    results_dir = Path(cfg.output.results_dir)

    if baseline is not None:
        for row in table:
            row["sketch_accuracy_delta"] = (
                row["sketch_accuracy"] - baseline["sketch"].accuracy
            )
        for run_name, result in evaluations.items():
            if run_name == "erm":
                continue
            deltas = per_class_delta(baseline["sketch"], result["sketch"], class_names)
            save_csv(deltas, results_dir / f"per_class_sketch_delta_{run_name}.csv")

    save_csv(table, results_dir / "final_comparison.csv")

    for run_name, result in evaluations.items():
        save_csv(
            confusion_rows(result["sketch"], class_names),
            results_dir / f"sketch_confusions_{run_name}.csv",
        )
        save_json(
            {
                "per_class_sketch_accuracy": {
                    class_names[k]: v for k, v in result["sketch"].per_class.items()
                },
                "source_domain_separability": result["separability"],
                "sharpness_proxy": result["sharpness"],
            },
            results_dir / f"sketch_detail_{run_name}.json",
        )

    LOGGER.info("Wrote Task 3 final comparison for %d methods", len(table))


if __name__ == "__main__":
    main()
