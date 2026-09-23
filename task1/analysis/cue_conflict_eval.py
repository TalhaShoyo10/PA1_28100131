"""Score generated cue-conflict images: shape bias, coverage, examples."""

from __future__ import annotations

import csv
from pathlib import Path

from PIL import Image


def load_manifest(directory: Path) -> list[dict]:
    """Read the generator's manifest, keeping only accepted conflicts.

    Rejection happened at generation time from image statistics alone. Model
    predictions never influence which images are scored.
    """
    manifest_path = Path(directory) / "manifest.csv"
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Cue-conflict manifest not found: {manifest_path}. Generate it with: "
            "python task1/data/make_cue_conflicts.py --config "
            "task1/configs/interventions.yaml"
        )

    with open(manifest_path, newline="", encoding="utf-8") as handle:
        records = list(csv.DictReader(handle))

    accepted = [r for r in records if str(r["accepted"]).lower() in ("true", "1")]
    for record in accepted:
        record["content_class"] = int(record["content_class"])
        record["style_class"] = int(record["style_class"])
    return accepted


def load_conflict_images(directory: Path, records: list[dict]) -> list[Image.Image]:
    """Load each accepted conflict image in manifest order."""
    directory = Path(directory)
    return [
        Image.open(directory / r["filename"]).convert("RGB") for r in records
    ]


def example_cases(
    records: list[dict],
    predictions,
    class_names: list[str],
    per_bucket: int = 3,
) -> list[dict]:
    """Collect a few shape, texture and other decisions for the report.

    The assignment asks for informative agreements, disagreements and
    failures, so all three buckets are sampled rather than only errors.
    """
    buckets: dict[str, list[dict]] = {"shape": [], "texture": [], "other": []}

    for record, prediction in zip(records, predictions):
        prediction = int(prediction)
        if prediction == record["content_class"]:
            decision = "shape"
        elif prediction == record["style_class"]:
            decision = "texture"
        else:
            decision = "other"

        if len(buckets[decision]) >= per_bucket:
            continue

        buckets[decision].append(
            {
                "decision": decision,
                "filename": record["filename"],
                "content_class": record["content_class_name"],
                "style_class": record["style_class_name"],
                "predicted_class": class_names[prediction],
                "pair_id": record["pair_id"],
                "direction": record["direction"],
            }
        )

    return buckets["shape"] + buckets["texture"] + buckets["other"]
