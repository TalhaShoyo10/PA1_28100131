"""Open-set evaluation: AUROC plus validation-calibrated rejection."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.config import apply_overrides, config_arg_parser, load_config
from common.logging import get_logger, save_csv, save_json
from common.metrics import (
    acceptance_rate,
    accuracy,
    auroc,
    calibrate_threshold,
    rejection_rate,
)
from common.plotting import save_figure, set_style
from common.seed import set_seed
from task4.data.cifar import CIFAR10_CLASSES, unknown_class_names
from task4.scores.posthoc import (
    energy_score,
    fit_mahalanobis,
    mahalanobis_score,
    mls_score,
    msp_score,
    proser_score,
)

LOGGER = get_logger("task4.osr")


def load_cache(cache_dir: Path, split: str) -> dict[str, np.ndarray]:
    """Load one cached split written by extract_outputs.py."""
    path = cache_dir / f"{split}.npz"
    if not path.exists():
        raise FileNotFoundError(
            f"Cached outputs not found: {path}. Run task4/extract_outputs.py first."
        )
    with np.load(path) as data:
        return {k: data[k] for k in data.files}


def compute_all_scores(
    cache_dir: Path, num_known: int = 10
) -> tuple[dict[str, dict[str, np.ndarray]], dict]:
    """Compute every score on every split from ONE cached set of outputs."""
    splits = {
        name: load_cache(cache_dir, name)
        for name in ("cifar10_train", "cifar10_val", "cifar10_test",
                     "cifar100_near", "cifar100_far")
    }

    train = splits["cifar10_train"]
    means, variance = fit_mahalanobis(
        train["features"], train["labels"], num_known, epsilon=1e-6
    )

    num_dummy = splits["cifar10_val"]["logits"].shape[1] - num_known

    scores: dict[str, dict[str, np.ndarray]] = {}
    for name, data in splits.items():
        known_logits = data["logits"][:, :num_known]
        entry = {
            "msp": msp_score(known_logits),
            "mls": mls_score(known_logits),
            "energy": energy_score(known_logits),
            "mahalanobis": mahalanobis_score(data["features"], means, variance),
        }
        if num_dummy > 0:
            entry["proser"] = proser_score(data["logits"], num_known, num_dummy)
        scores[name] = entry

    meta = {
        "num_known": num_known,
        "num_dummy": num_dummy,
        "mahalanobis_fitted_on": "cifar10_train_unaugmented",
        "n_train_for_stats": int(len(train["labels"])),
    }
    return scores, meta, splits


def evaluate_score(
    score_name: str,
    model_name: str,
    scores: dict[str, dict[str, np.ndarray]],
    percentile: float,
    closed_set_accuracy: float,
) -> list[dict]:
    """AUROC and calibrated rejection for one score across all unknown groups."""
    tau = calibrate_threshold(scores["cifar10_val"][score_name], percentile)

    test_known = scores["cifar10_test"][score_name]
    near = scores["cifar100_near"][score_name]
    far = scores["cifar100_far"][score_name]
    all_unknown = np.concatenate([near, far])

    rows = []
    for group, unknown_scores in (
        ("near", near), ("far", far), ("all", all_unknown)
    ):
        rows.append(
            {
                "model": model_name,
                "score": score_name,
                "unknown_group": group,
                "auroc": auroc(test_known, unknown_scores),
                "tau": tau,
                "known_acceptance": acceptance_rate(test_known, tau),
                "unknown_rejection": rejection_rate(unknown_scores, tau),
                "fpr_at_95_tpr": acceptance_rate(unknown_scores, tau),
                "closed_set_accuracy": closed_set_accuracy,
            }
        )
    return rows


def failure_cases(
    splits: dict,
    scores: dict,
    score_name: str,
    tau: float,
    unknown_names: dict[int, str],
    group: str,
    limit: int = 5,
) -> list[dict]:
    """Unknowns wrongly ACCEPTED as known, with what they were mistaken for."""
    data = splits[f"cifar100_{group}"]
    unknown_scores = scores[f"cifar100_{group}"][score_name]

    accepted = np.flatnonzero(unknown_scores <= tau)
    order = accepted[np.argsort(unknown_scores[accepted])]

    rows = []
    for index in order[:limit]:
        predicted = int(data["logits"][index, :10].argmax())
        rows.append(
            {
                "group": group,
                "unknown_class": unknown_names.get(int(data["labels"][index]), "?"),
                "predicted_known_class": CIFAR10_CLASSES[predicted],
                "score": float(unknown_scores[index]),
                "threshold": float(tau),
                "margin": float(tau - unknown_scores[index]),
            }
        )
    return rows


def plot_score_distributions(scores: dict, output_path: Path):
    """Compact multi-panel figure for MSP, MLS and Mahalanobis."""
    import matplotlib.pyplot as plt

    set_style()
    panels = ["msp", "mls", "mahalanobis"]
    fig, axes = plt.subplots(1, len(panels), figsize=(11, 3.2))

    for ax, score_name in zip(axes, panels):
        for label, split, color in (
            ("known (test)", "cifar10_test", "#4C72B0"),
            ("near unknown", "cifar100_near", "#DD8452"),
            ("far unknown", "cifar100_far", "#55A868"),
        ):
            values = scores[split][score_name]
            ax.hist(values, bins=50, alpha=0.55, label=label, color=color, density=True)
        ax.set_title(score_name.upper())
        ax.set_xlabel("unknownness u(x)")
        ax.set_ylabel("density")

    axes[0].legend(fontsize=7)
    fig.tight_layout()
    return save_figure(fig, output_path)


def main() -> None:
    parser = config_arg_parser("Task 4 open-set evaluation.")
    parser.add_argument("--runs", nargs="*", default=["vanilla", "gcsc", "proser"])
    args = parser.parse_args()

    cfg = apply_overrides(load_config(args.config), args.overrides)
    set_seed(cfg.seed)

    percentile = cfg.evaluation.threshold.percentile
    results_dir = Path(cfg.output.results_dir)
    table: list[dict] = []

    try:
        unknown_names = unknown_class_names(cfg.data.unknown.root)
    except Exception:
        unknown_names = {}

    for run_name in args.runs:
        cache_dir = Path(cfg.output.cache_dir) / run_name
        if not cache_dir.exists():
            LOGGER.warning("No cached outputs for %s; skipping.", run_name)
            continue

        scores, meta, splits = compute_all_scores(cache_dir, cfg.model.num_classes)

        test = splits["cifar10_test"]
        closed_set = accuracy(test["labels"], test["logits"][:, :cfg.model.num_classes].argmax(1))
        LOGGER.info("%s: closed-set accuracy %.2f", run_name, closed_set)

        score_names = ["msp", "mls", "energy", "mahalanobis"]
        if meta["num_dummy"] > 0:
            score_names.append("proser")

        for score_name in score_names:
            table.extend(
                evaluate_score(score_name, run_name, scores, percentile, closed_set)
            )

        if run_name == "vanilla":
            plot_score_distributions(scores, Path(cfg.output.figures_dir) / "score_distributions")

            tau = calibrate_threshold(scores["cifar10_val"]["mls"], percentile)
            failures = (
                failure_cases(splits, scores, "mls", tau, unknown_names, "near",
                              cfg.evaluation.failure_analysis.min_near_cases + 2)
                + failure_cases(splits, scores, "mls", tau, unknown_names, "far",
                                cfg.evaluation.failure_analysis.min_far_cases + 2)
            )
            save_csv(failures, results_dir / "failure_cases_vanilla_mls.csv")
            LOGGER.info("Recorded %d accepted-unknown failure cases", len(failures))

        save_json(meta, results_dir / f"score_meta_{run_name}.json")

    save_csv(table, results_dir / "osr_results.csv")

    vanilla_rows = [r for r in table if r["model"] == "vanilla"]
    if vanilla_rows:
        save_csv(vanilla_rows, results_dir / "posthoc_score_comparison.csv")

    mls_rows = [
        r for r in table
        if r["score"] in ("mls", "proser") and r["unknown_group"] in ("near", "far", "all")
    ]
    save_csv(mls_rows, results_dir / "trained_model_comparison.csv")

    LOGGER.info("Wrote %d OSR result rows", len(table))


if __name__ == "__main__":
    main()
