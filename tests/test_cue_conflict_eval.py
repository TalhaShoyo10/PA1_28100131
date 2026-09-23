import csv
import json

import numpy as np
import pytest
from PIL import Image

from task1.analysis.cue_conflict_eval import (
    example_cases,
    load_conflict_images,
    load_manifest,
)
from task1.analysis.evaluate_bias import classify_cue_conflict


CLASSES = ["airplane", "bird", "car", "cat", "deer",
           "dog", "horse", "monkey", "ship", "truck"]


def _write_manifest(directory, rows):
    directory.mkdir(parents=True, exist_ok=True)
    with open(directory / "manifest.csv", "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _row(name, content, style, accepted=True, reason=""):
    return {
        "filename": name,
        "content_class": content,
        "style_class": style,
        "content_class_name": CLASSES[content],
        "style_class_name": CLASSES[style],
        "content_image_id": f"test_{content:05d}",
        "style_image_id": f"test_{style:05d}",
        "pair_id": f"{content}_{style}",
        "direction": "a_to_b",
        "alpha": 1.0,
        "accepted": accepted,
        "rejection_reason": reason,
    }


def test_load_manifest_keeps_only_accepted(tmp_path):
    _write_manifest(tmp_path, [
        _row("a.png", 0, 8),
        _row("b.png", 1, 4, accepted=False, reason="saturation_collapse(7.8)"),
        _row("c.png", 3, 5),
    ])
    records = load_manifest(tmp_path)
    assert [r["filename"] for r in records] == ["a.png", "c.png"]
    assert all(isinstance(r["content_class"], int) for r in records)


def test_load_manifest_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError, match="make_cue_conflicts"):
        load_manifest(tmp_path)


def test_load_conflict_images_follows_manifest_order(tmp_path):
    for name, colour in (("a.png", (255, 0, 0)), ("c.png", (0, 255, 0))):
        Image.new("RGB", (8, 8), colour).save(tmp_path / name)
    _write_manifest(tmp_path, [_row("a.png", 0, 8), _row("c.png", 3, 5)])

    images = load_conflict_images(tmp_path, load_manifest(tmp_path))
    assert len(images) == 2
    assert images[0].getpixel((0, 0)) == (255, 0, 0)
    assert images[1].getpixel((0, 0)) == (0, 255, 0)


def test_shape_texture_other_partition_is_exhaustive():
    content = np.array([0, 0, 0, 0])
    style = np.array([8, 8, 8, 8])
    predictions = np.array([0, 8, 5, 0])       # shape, texture, other, shape

    decisions = classify_cue_conflict(predictions, content, style)
    assert decisions["n_shape"] == 2
    assert decisions["n_texture"] == 1
    assert decisions["n_other"] == 1
    assert decisions["n_shape"] + decisions["n_texture"] + decisions["n_other"] == 4
    assert decisions["shape_bias"] == pytest.approx(200.0 / 3.0)
    assert decisions["coverage"] == pytest.approx(75.0)


def test_example_cases_samples_all_three_buckets(tmp_path):
    rows = [_row(f"{i}.png", 0, 8) for i in range(9)]
    _write_manifest(tmp_path, rows)
    records = load_manifest(tmp_path)
    predictions = np.array([0, 0, 0, 8, 8, 8, 5, 5, 5])

    cases = example_cases(records, predictions, CLASSES, per_bucket=2)
    decisions = [c["decision"] for c in cases]
    assert decisions.count("shape") == 2
    assert decisions.count("texture") == 2
    assert decisions.count("other") == 2
    assert cases[0]["content_class"] == "airplane"
    assert cases[0]["style_class"] == "ship"


def test_example_cases_handles_an_empty_bucket(tmp_path):
    rows = [_row(f"{i}.png", 0, 8) for i in range(3)]
    _write_manifest(tmp_path, rows)
    records = load_manifest(tmp_path)

    cases = example_cases(records, np.array([0, 0, 0]), CLASSES, per_bucket=2)
    assert [c["decision"] for c in cases] == ["shape", "shape"]
