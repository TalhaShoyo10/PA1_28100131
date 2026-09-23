"""Run the full Task 1 pipeline: heads, interventions, and representations."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from common.config import apply_overrides, config_arg_parser, load_config
from common.logging import RunRecord, get_logger, save_csv, save_json
from common.seed import set_seed
from shared.pacs_protocol import stratified_split
from task1.analysis.evaluate_bias import (
    classify_cue_conflict,
    evaluate_condition,
    extract_features,
    normalize_for_backbone,
    predict,
    predict_zero_shot,
    train_linear_head,
    translation_curve,
)
from task1.analysis.cue_conflict_eval import (
    example_cases,
    load_conflict_images,
    load_manifest,
)
from task1.analysis.representation import (
    cosine_stability,
    displacement_from_clean,
    fit_tsne,
    plot_representation,
)
from task1.data.make_subset import load_eval_subset, load_stl10
from task1.data.transforms import (
    batch_to_tensor,
    grayscale,
    hue_rotate,
    patch_permutation,
    patch_shuffle,
    to_canvas,
    translate,
)
from task1.models.backbones import build_backbone

LOGGER = get_logger("task1.run")


def build_intervention_builders(cfg, images, smoke: bool = False) -> dict:
    """Return one builder per condition, all deterministic.

    Builders rather than tensors: 500 images at 224x224 float32 is ~0.30 GB per
    condition, and there are 16 conditions. Materialising them all at once held
    ~4.8 GB of RAM before a backbone was even loaded, which exhausted a free
    Colab runtime when ResNet-50 arrived. Each condition is now built on demand
    and released immediately after use.

    Determinism is preserved: every builder is a pure function of the shared
    canvas images and fixed settings, so each model still receives byte-identical
    inputs no matter when the tensor is created.
    """
    interventions = cfg.interventions
    degrees = interventions.hue_rotation.degrees
    grid = interventions.patch_shuffle.grid[0]
    seed = interventions.patch_shuffle.seed

    builders: dict[str, callable] = {
        "clean": lambda: batch_to_tensor(images),
        "grayscale": lambda: batch_to_tensor([grayscale(im) for im in images]),
        f"hue{int(degrees)}": lambda: batch_to_tensor(
            [hue_rotate(im, degrees) for im in images]
        ),
        "patch_shuffle": lambda: batch_to_tensor(
            [
                patch_shuffle(im, grid, permutation=patch_permutation(grid, seed, i))
                for i, im in enumerate(images)
            ]
        ),
    }

    displacements = [0, 8] if smoke else interventions.translation.displacements
    for displacement in displacements:
        if displacement == 0:
            continue
        for direction in interventions.translation.directions:
            builders[f"translate_{displacement}_{direction}"] = (
                lambda d=displacement, dr=direction: batch_to_tensor(
                    [translate(im, d, dr) for im in images]
                )
            )

    return builders


def load_training_features(cfg, backbone, device: str, limit: int | None = None):
    """Extract frozen features for the stratified 80/20 head-training split."""
    dataset = load_stl10(cfg.data.root, split="train", download=True)
    labels = np.asarray(dataset.labels)

    train_idx, val_idx = stratified_split(labels, cfg.data.val_fraction, cfg.data.split_seed)
    if limit:
        train_idx, val_idx = train_idx[:limit], val_idx[: max(2, limit // 4)]

    def features_for(indices):
        chunk_size = cfg.head.feature_chunk_size
        outputs = []
        for start in range(0, len(indices), chunk_size):
            chunk = indices[start : start + chunk_size]
            images = [to_canvas(dataset[i][0]) for i in chunk]
            tensor = normalize_for_backbone(batch_to_tensor(images), backbone)
            del images
            outputs.append(extract_features(backbone, tensor, device=device))
            del tensor
        return torch.cat(outputs), torch.tensor([int(labels[i]) for i in indices])

    return features_for(train_idx), features_for(val_idx)


def run(cfg, smoke: bool = False) -> dict:
    """Execute Task 1 end to end for all three backbones plus CLIP zero-shot."""
    set_seed(cfg.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    LOGGER.info("device=%s seed=%d", device, cfg.seed)

    subset = load_eval_subset(cfg)
    if smoke:
        subset = subset[: cfg.smoke.eval_subset_size]

    image_ids = [i for i, _, _ in subset]
    images = [to_canvas(im) for _, im, _ in subset]
    y_true = np.array([label for _, _, label in subset])
    class_names = list(cfg.data.classes)

    LOGGER.info("Building interventions for %d images", len(images))
    builders = build_intervention_builders(cfg, images, smoke)
    LOGGER.info("Conditions: %s", sorted(builders))

    results_dir = Path(cfg.output.results_dir)
    figures_dir = Path(cfg.output.figures_dir)

    rows: list[dict] = []
    stability_rows: list[dict] = []
    shape_bias_rows: list[dict] = []
    conflict_examples: list[dict] = []
    head_info: dict[str, dict] = {}

    conflict_dir = Path(cfg.data.interventions_dir) / "cue_conflict"
    conflict_records: list[dict] = []
    conflict_images: list = []
    content_images: list = []
    if conflict_dir.exists():
        conflict_records = load_manifest(conflict_dir)
        conflict_images = load_conflict_images(conflict_dir, conflict_records)
        by_id = {image_id: image for image_id, image in zip(image_ids, images)}
        missing = [
            r["content_image_id"] for r in conflict_records
            if r["content_image_id"] not in by_id
        ]
        if missing:
            raise KeyError(
                f"{len(missing)} cue-conflict content images are absent from the "
                f"evaluation subset (first: {missing[0]}). The conflicts were "
                "generated from a different subset; regenerate them."
            )
        content_images = [by_id[r["content_image_id"]] for r in conflict_records]
        LOGGER.info(
            "Cue conflict: %d accepted images from %s",
            len(conflict_records), conflict_dir,
        )
    else:
        LOGGER.warning(
            "No cue-conflict directory at %s -- shape bias will NOT be computed. "
            "Generate it with task1/data/make_cue_conflicts.py.", conflict_dir,
        )

    for backbone_name in dict.keys(cfg.backbones):
        LOGGER.info("=== %s ===", backbone_name)
        backbone = build_backbone(backbone_name).to(device)

        (train_features, train_labels), (val_features, val_labels) = load_training_features(
            cfg, backbone, device, limit=64 if smoke else None
        )
        head, info = train_linear_head(
            train_features, train_labels, val_features, val_labels,
            num_classes=cfg.data.num_classes,
            epochs=cfg.smoke.head_epochs if smoke else cfg.head.epochs,
            lr=cfg.head.lr, weight_decay=cfg.head.weight_decay,
            batch_size=cfg.head.batch_size,
            patience=cfg.head.early_stopping_patience,
            device=device, seed=cfg.head.seed,
        )
        head_info[backbone_name] = info
        LOGGER.info("head val accuracy %.2f (epoch %d)", info["best_val_accuracy"], info["best_epoch"])

        clean_tensor = builders["clean"]()
        clean_predictions, clean_probabilities = predict(backbone, head, clean_tensor, device)
        clean_result = evaluate_condition(
            backbone_name, "clean", y_true, clean_predictions, clean_probabilities,
            cfg.data.num_classes,
        )
        rows.append(clean_result.to_dict())

        clean_features = extract_features(
            backbone, normalize_for_backbone(clean_tensor, backbone), device=device
        )

        translation_results: dict[int, list] = {}

        for condition, builder in builders.items():
            if condition == "clean":
                continue
            tensor = builder()

            predictions, probabilities = predict(backbone, head, tensor, device)
            result = evaluate_condition(
                backbone_name, condition, y_true, predictions, probabilities,
                cfg.data.num_classes,
                clean_predictions=clean_predictions,
                clean_accuracy=clean_result.accuracy,
            )
            rows.append(result.to_dict())

            if condition.startswith("translate_"):
                displacement = int(condition.split("_")[1])
                translation_results.setdefault(displacement, []).append(result)

            if condition in ("grayscale", "patch_shuffle") or condition.startswith("hue"):
                transformed_features = extract_features(
                    backbone, normalize_for_backbone(tensor, backbone), device=device
                )
                stability = cosine_stability(clean_features, transformed_features)
                stability_rows.append(
                    {
                        "model": backbone_name,
                        "condition": condition,
                        "cosine_stability": stability["cosine_stability"],
                        "std": stability["std"],
                        "n": stability["n"],
                        "n_degenerate": stability["n_degenerate"],
                    }
                )

                clean_2d, transformed_2d, settings = fit_tsne(
                    clean_features.numpy(), transformed_features.numpy(),
                    perplexity=cfg.representation_analysis.visualization.perplexity,
                    n_iter=cfg.representation_analysis.visualization.n_iter,
                    seed=cfg.representation_analysis.visualization.seed,
                    metric=cfg.representation_analysis.visualization.metric,
                )
                plot_representation(
                    clean_2d, transformed_2d, y_true, class_names,
                    backbone_name, condition,
                    figures_dir / f"tsne_{backbone_name}_{condition}",
                    settings,
                )
                save_json(
                    {**settings, **displacement_from_clean(clean_2d, transformed_2d)},
                    results_dir / f"tsne_{backbone_name}_{condition}.json",
                )
                del transformed_features

            del tensor

        translation_rows = translation_curve(
            {0: [clean_result], **translation_results}
        )
        save_csv(
            [{"model": backbone_name, **r} for r in translation_rows],
            results_dir / f"translation_{backbone_name}.csv",
        )

        if conflict_records:
            conflict_tensor = batch_to_tensor(conflict_images)
            conflict_predictions, _ = predict(
                backbone, head, conflict_tensor, device
            )
            decisions = classify_cue_conflict(
                conflict_predictions,
                np.array([r["content_class"] for r in conflict_records]),
                np.array([r["style_class"] for r in conflict_records]),
            )
            shape_bias_rows.append({"model": backbone_name, **decisions})
            LOGGER.info(
                "cue conflict: shape %d texture %d other %d | shape bias %.2f coverage %.2f",
                decisions["n_shape"], decisions["n_texture"], decisions["n_other"],
                decisions["shape_bias"], decisions["coverage"],
            )

            for case in example_cases(
                conflict_records, conflict_predictions, class_names
            ):
                conflict_examples.append({"model": backbone_name, **case})

            conflict_features = extract_features(
                backbone, normalize_for_backbone(conflict_tensor, backbone),
                device=device,
            )
            content_features = extract_features(
                backbone,
                normalize_for_backbone(batch_to_tensor(content_images), backbone),
                device=device,
            )
            stability = cosine_stability(content_features, conflict_features)
            stability_rows.append(
                {
                    "model": backbone_name,
                    "condition": "cue_conflict",
                    "cosine_stability": stability["cosine_stability"],
                    "std": stability["std"],
                    "n": stability["n"],
                    "n_degenerate": stability["n_degenerate"],
                }
            )
            del conflict_tensor, conflict_features, content_features

        if backbone_name == "clip_vitb32":
            prompts = [cfg.zero_shot.prompt_template.format(**{"class": c}) for c in class_names]
            text_embeddings = backbone.encode_text(prompts)
            zs_predictions, zs_probabilities = predict_zero_shot(
                backbone, clean_tensor, text_embeddings, device
            )
            rows.append(
                evaluate_condition(
                    "clip_zeroshot", "clean", y_true, zs_predictions, zs_probabilities,
                    cfg.data.num_classes,
                ).to_dict()
            )
            save_json(
                {
                    "prompts": prompts,
                    "agreement_with_head": float((zs_predictions == clean_predictions).mean() * 100),
                },
                results_dir / "clip_zeroshot.json",
            )

        del backbone, head, clean_tensor, clean_features
        import gc

        gc.collect()
        if device == "cuda":
            torch.cuda.empty_cache()

    save_csv(rows, results_dir / "intervention_results.csv")
    save_csv(stability_rows, results_dir / "representation_stability.csv")
    save_json(head_info, results_dir / "linear_heads.json")

    if shape_bias_rows:
        save_csv(shape_bias_rows, results_dir / "shape_bias.csv")
        save_csv(conflict_examples, results_dir / "cue_conflict_examples.csv")
        import shutil

        for name in ("manifest.csv", "summary.json"):
            source = conflict_dir / name
            if source.exists():
                shutil.copy(source, results_dir / f"cue_conflict_{name}")

    summary = {
        "n_images": len(images),
        "conditions": sorted(builders),
        "backbones": sorted(dict.keys(cfg.backbones)),
        "image_ids": image_ids[:10],
        "smoke": smoke,
    }
    RunRecord(run_name="task1", task="task1", config=dict(cfg), metrics=summary).save(
        results_dir / "run.json"
    )
    LOGGER.info("Wrote %d result rows to %s", len(rows), results_dir)
    return summary


def main() -> None:
    parser = config_arg_parser("Run the Task 1 pipeline.")
    args = parser.parse_args()
    cfg = apply_overrides(load_config(args.config), args.overrides)
    run(cfg, smoke=args.smoke)


if __name__ == "__main__":
    main()
