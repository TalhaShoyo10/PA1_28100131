"""Generate cue-conflict images: shape from class A, texture from class B."""

from __future__ import annotations

import argparse
import itertools
import json
from dataclasses import dataclass, asdict
from pathlib import Path

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import torch
from PIL import Image

from common.config import apply_overrides, load_config
from common.logging import get_logger, save_csv
from common.seed import set_seed
from task1.data.transforms import CANVAS_SIZE, to_canvas, to_tensor

LOGGER = get_logger("task1.cue_conflicts")


@dataclass
class CueConflict:
    """One generated conflict image and the provenance needed to score it."""

    filename: str
    content_class: int
    style_class: int
    content_class_name: str
    style_class_name: str
    content_image_id: str
    style_image_id: str
    pair_id: str
    direction: str
    alpha: float
    accepted: bool
    rejection_reason: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


def select_class_pairs(
    num_classes: int, num_pairs: int, seed: int
) -> list[tuple[int, int]]:
    """Choose unordered class pairs deterministically."""
    if num_pairs < 5:
        raise ValueError(f"Assignment requires at least 5 class pairs, got {num_pairs}")

    all_pairs = list(itertools.combinations(range(num_classes), 2))
    if num_pairs > len(all_pairs):
        raise ValueError(
            f"Requested {num_pairs} pairs but only {len(all_pairs)} exist "
            f"for {num_classes} classes."
        )

    rng = np.random.default_rng(seed)
    chosen = rng.choice(len(all_pairs), size=num_pairs, replace=False)
    return [all_pairs[i] for i in sorted(chosen.tolist())]


def structure_score(image: Image.Image) -> float:
    """Mean gradient magnitude of the V channel, as a proxy for visible structure."""
    v = np.asarray(image.convert("HSV"), dtype=np.float32)[..., 2]
    gy = np.abs(np.diff(v, axis=0)).mean()
    gx = np.abs(np.diff(v, axis=1)).mean()
    return float(gy + gx)


def saturation_mean(image: Image.Image) -> float:
    """Mean HSV saturation, used to detect washed-out stylization failures."""
    s = np.asarray(image.convert("HSV"), dtype=np.float32)[..., 1]
    return float(s.mean())


def evaluate_rejection_rule(
    stylized: Image.Image,
    content: Image.Image,
    style: Image.Image,
    min_structure_ratio: float = 0.30,
    min_saturation: float = 8.0,
    min_texture_change: float = 0.05,
) -> tuple[bool, str]:
    """Apply the pre-registered, model-blind acceptance rule to one image.

    Defined before any model sees the images and uses only image statistics,
    never model predictions.
    """
    content_structure = structure_score(content)
    stylized_structure = structure_score(stylized)

    if content_structure <= 0:
        return False, "degenerate_content"

    ratio = stylized_structure / content_structure
    if ratio < min_structure_ratio:
        return False, f"content_structure_lost({ratio:.2f})"

    if saturation_mean(stylized) < min_saturation:
        return False, f"saturation_collapse({saturation_mean(stylized):.1f})"

    content_arr = np.asarray(content, dtype=np.float32) / 255.0
    stylized_arr = np.asarray(stylized, dtype=np.float32) / 255.0
    change = float(np.abs(content_arr - stylized_arr).mean())
    if change < min_texture_change:
        return False, f"style_not_applied({change:.3f})"

    return True, ""


def generate_conflicts(
    images_by_class: dict[int, list[tuple[str, Image.Image]]],
    class_names: list[str],
    stylizer,
    pairs: list[tuple[int, int]],
    target_count: int,
    alpha: float,
    seed: int,
    output_dir: Path,
    device: str = "cpu",
    rejection_kwargs: dict | None = None,
) -> list[CueConflict]:
    """Produce bidirectional cue conflicts balanced across pairs and directions."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rejection_kwargs = rejection_kwargs or {}

    per_direction = int(np.ceil(target_count / (len(pairs) * 2)))
    rng = np.random.default_rng(seed)
    records: list[CueConflict] = []

    for class_a, class_b in pairs:
        pair_id = f"{class_names[class_a]}__{class_names[class_b]}"

        for content_class, style_class in ((class_a, class_b), (class_b, class_a)):
            direction = f"{class_names[content_class]}_shape__{class_names[style_class]}_texture"
            content_pool = images_by_class[content_class]
            style_pool = images_by_class[style_class]

            n = min(per_direction, len(content_pool))
            content_idx = rng.choice(len(content_pool), size=n, replace=False)
            style_idx = rng.choice(len(style_pool), size=n, replace=len(style_pool) < n)

            for ci, si in zip(content_idx.tolist(), style_idx.tolist()):
                content_id, content_img = content_pool[ci]
                style_id, style_img = style_pool[si]

                content_t = to_tensor(content_img).unsqueeze(0).to(device)
                style_t = to_tensor(style_img).unsqueeze(0).to(device)

                with torch.no_grad():
                    out = stylizer(content_t, style_t, alpha=alpha)

                array = (out.squeeze(0).cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8)
                stylized = Image.fromarray(array)

                accepted, reason = evaluate_rejection_rule(
                    stylized, content_img, style_img, **rejection_kwargs
                )

                filename = f"{pair_id}__{direction}__{content_id}__{style_id}.png"
                if accepted:
                    stylized.save(output_dir / filename)

                records.append(
                    CueConflict(
                        filename=filename,
                        content_class=content_class,
                        style_class=style_class,
                        content_class_name=class_names[content_class],
                        style_class_name=class_names[style_class],
                        content_image_id=str(content_id),
                        style_image_id=str(style_id),
                        pair_id=pair_id,
                        direction=direction,
                        alpha=alpha,
                        accepted=accepted,
                        rejection_reason=reason,
                    )
                )

    return records


def summarize(records: list[CueConflict]) -> dict:
    """Aggregate accepted/rejected counts overall and per pair and direction."""
    accepted = [r for r in records if r.accepted]
    by_pair: dict[str, dict[str, int]] = {}
    for r in records:
        entry = by_pair.setdefault(r.pair_id, {"accepted": 0, "rejected": 0})
        entry["accepted" if r.accepted else "rejected"] += 1

    reasons: dict[str, int] = {}
    for r in records:
        if not r.accepted:
            reasons[r.rejection_reason] = reasons.get(r.rejection_reason, 0) + 1

    return {
        "total_generated": len(records),
        "accepted": len(accepted),
        "rejected": len(records) - len(accepted),
        "acceptance_rate": 100.0 * len(accepted) / len(records) if records else 0.0,
        "by_pair": by_pair,
        "rejection_reasons": reasons,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Task 1 cue-conflict images.")
    parser.add_argument("--config", type=Path, default=Path("task1/configs/interventions.yaml"))
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument(
        "--set",
        dest="overrides",
        nargs="*",
        default=None,
        metavar="KEY=VALUE",
        help="Override config entries, e.g. --set interventions.cue_conflict."
             "target_valid_conflicts=240",
    )
    args = parser.parse_args()

    cfg = apply_overrides(load_config(args.config), args.overrides)
    set_seed(cfg.seed)

    from task1.data.make_subset import load_eval_subset
    from task1.models.adain import AdaINStyleTransfer

    cc = cfg.interventions.cue_conflict
    output_dir = args.output_dir or Path(cfg.data.interventions_dir) / "cue_conflict"

    subset = load_eval_subset(cfg)
    images_by_class: dict[int, list[tuple[str, Image.Image]]] = {}
    for image_id, image, label in subset:
        images_by_class.setdefault(label, []).append((image_id, to_canvas(image)))

    pairs = select_class_pairs(
        cfg.data.num_classes,
        1 if args.smoke else cc.class_pairs_min,
        cfg.seed,
    )
    LOGGER.info("Class pairs: %s", [(cfg.data.classes[a], cfg.data.classes[b]) for a, b in pairs])

    stylizer = AdaINStyleTransfer().to(args.device)
    records = generate_conflicts(
        images_by_class=images_by_class,
        class_names=list(cfg.data.classes),
        stylizer=stylizer,
        pairs=pairs,
        target_count=10 if args.smoke else cc.target_valid_conflicts,
        alpha=cc.style_strength,
        seed=cfg.seed,
        output_dir=output_dir,
        device=args.device,
    )

    summary = summarize(records)
    LOGGER.info(
        "Generated %d, accepted %d (%.1f%%), rejected %d",
        summary["total_generated"], summary["accepted"],
        summary["acceptance_rate"], summary["rejected"],
    )

    save_csv([r.to_dict() for r in records], output_dir / "manifest.csv")
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    required = dict(cc).get("required_valid_conflicts", 200)
    if not args.smoke and summary["accepted"] < required:
        LOGGER.warning(
            "Only %d accepted conflicts, below the assignment minimum of %d. "
            "Raise interventions.cue_conflict.target_valid_conflicts and "
            "regenerate BEFORE any model evaluation. Do not loosen the "
            "rejection thresholds to reach the count.",
            summary["accepted"], required,
        )
    elif not args.smoke:
        LOGGER.info(
            "%d accepted conflicts (assignment minimum %d) from %d generated.",
            summary["accepted"], required, summary["total_generated"],
        )


if __name__ == "__main__":
    main()
