"""Build the report figures that the evaluation scripts do not emit."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.logging import get_logger
from common.plotting import BACKBONE_COLORS, METHOD_COLORS, save_figure, set_style

LOGGER = get_logger("figures")

BACKBONE_LABELS = {
    "resnet50": "ResNet-50",
    "vit_b16": "ViT-B/16",
    "clip_vitb32": "CLIP ViT-B/32",
}


def translation_curve(results_dir: Path, figures_dir: Path) -> bool:
    """Accuracy and consistency against displacement, one line per backbone."""
    frames = []
    for backbone in BACKBONE_LABELS:
        path = results_dir / f"translation_{backbone}.csv"
        if path.exists():
            frames.append(pd.read_csv(path))

    if not frames:
        LOGGER.warning("No translation CSVs in %s", results_dir)
        return False

    data = pd.concat(frames, ignore_index=True)
    set_style()
    fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.2), sharex=True)

    for metric, ax, title in (
        ("accuracy", axes[0], "Accuracy"),
        ("consistency", axes[1], "Prediction consistency"),
    ):
        for backbone, label in BACKBONE_LABELS.items():
            subset = data[data["model"] == backbone].sort_values("displacement")
            if subset.empty:
                continue
            ax.plot(
                subset["displacement"], subset[metric],
                marker="o", label=label, color=BACKBONE_COLORS.get(backbone),
            )
        ax.set_xlabel("Displacement (pixels)")
        ax.set_ylabel(f"{title} (%)")
        ax.set_title(f"{title} vs translation")
        ax.set_xticks(sorted(data["displacement"].unique()))

    axes[0].legend(frameon=False)
    fig.suptitle("Translation: averaged over the four cardinal directions", y=1.02)
    save_figure(fig, figures_dir / "translation_curve")
    LOGGER.info("Wrote %s", figures_dir / "translation_curve.png")
    return True


def _curve_panel(ax, results_dir: Path, runs: list[str], column: str, ylabel: str):
    """Plot one column of each run's training curve, skipping absent runs."""
    plotted = False
    for run in runs:
        path = results_dir / run / "training_curve.csv"
        if not path.exists():
            continue
        frame = pd.read_csv(path)
        if column not in frame.columns:
            continue
        ax.plot(
            frame["epoch"], frame[column],
            marker="o", markersize=3, label=run,
            color=METHOD_COLORS.get(run),
        )
        plotted = True

    ax.set_xlabel("Epoch")
    ax.set_ylabel(ylabel)
    return plotted


def training_curves(
    results_dir: Path,
    figures_dir: Path,
    runs: list[str],
    name: str,
    penalty_column: str,
    penalty_label: str,
) -> bool:
    """Classification loss beside the alignment penalty, as the PDF requires."""
    set_style()
    fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.2))

    left = _curve_panel(axes[0], results_dir, runs, "cls_loss", "Classification loss")
    axes[0].set_title("Classification loss")

    right = _curve_panel(axes[1], results_dir, runs, penalty_column, penalty_label)
    axes[1].set_title(penalty_label)

    if not left:
        plt.close(fig)
        LOGGER.warning("No training curves found under %s", results_dir)
        return False

    if not right:
        axes[1].text(
            0.5, 0.5, f"No {penalty_column} recorded",
            ha="center", va="center", transform=axes[1].transAxes,
        )

    axes[0].legend(frameon=False)
    save_figure(fig, figures_dir / name)
    LOGGER.info("Wrote %s", figures_dir / f"{name}.png")
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="Build report figures.")
    parser.add_argument("--task1-results", type=Path, default=Path("task1/results"))
    parser.add_argument("--task2-results", type=Path, default=Path("task2/results"))
    parser.add_argument("--task3-results", type=Path, default=Path("task3/results"))
    parser.add_argument("--figures-dir", type=Path, default=Path("figures"))
    args = parser.parse_args()

    written = 0

    if translation_curve(args.task1_results, args.figures_dir / "task1"):
        written += 1

    if training_curves(
        args.task2_results,
        args.figures_dir / "task2",
        ["source_only", "dan", "dann", "cdan"],
        "training_curves",
        "mmd_loss",
        "MMD penalty",
    ):
        written += 1

    if training_curves(
        args.task2_results,
        args.figures_dir / "task2",
        ["dan_lambda0.1", "dan_lambda1.0", "dan_lambda10.0"],
        "lambda_study_curves",
        "mmd_loss",
        "MMD penalty",
    ):
        written += 1

    if training_curves(
        args.task3_results,
        args.figures_dir / "task3",
        ["dan_dg", "sam"],
        "training_curves",
        "mmd_loss",
        "MMD penalty",
    ):
        written += 1

    LOGGER.info("Wrote %d figures", written)


if __name__ == "__main__":
    main()
