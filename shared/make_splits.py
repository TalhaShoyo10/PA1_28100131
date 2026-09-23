"""Generate the committed PACS split manifest shared by Tasks 2 and 3."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.config import apply_overrides, load_config
from common.logging import get_logger
from common.seed import set_seed
from shared.pacs import PACS_CLASSES
from shared.pacs_protocol import build_split_manifest, save_split_manifest

LOGGER = get_logger("shared.make_splits")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build the PACS split manifest reused by Tasks 2 and 3."
    )
    parser.add_argument("--config", type=Path, default=Path("task2/configs/base.yaml"))
    parser.add_argument("--force", action="store_true", help="Overwrite an existing manifest.")
    parser.add_argument(
        "--set", dest="overrides", nargs="*", default=None, metavar="KEY=VALUE",
        help="Override config entries, e.g. --set data.root=/path/to/PACS",
    )
    args = parser.parse_args()

    cfg = apply_overrides(load_config(args.config), args.overrides)
    set_seed(cfg.seed)

    output_path = Path(cfg.data.split_manifest)
    if output_path.exists() and not args.force:
        LOGGER.warning(
            "Manifest already exists: %s. Tasks 2 and 3 MUST use the same "
            "splits, so regenerating one would invalidate checkpoints trained "
            "on the other. Pass --force only if you intend to retrain both.",
            output_path,
        )
        return

    manifest = build_split_manifest(
        root=cfg.data.root,
        sources=list(cfg.data.sources),
        target=cfg.data.target,
        val_fraction=cfg.data.val_fraction,
        seed=cfg.data.split_seed,
    )
    save_split_manifest(manifest, output_path)

    LOGGER.info("Wrote %s (seed %d)", output_path, manifest["seed"])
    for domain, entry in manifest["domains"].items():
        if entry["role"] == "source":
            LOGGER.info(
                "  %-14s train=%4d val=%4d (total %d)",
                domain, len(entry["train"]), len(entry["val"]), entry["n_total"],
            )
        else:
            LOGGER.info("  %-14s target, %d images (unlabeled)", domain, entry["n_total"])

    for domain, entry in manifest["domains"].items():
        if entry["role"] != "source":
            continue
        counts = {c: 0 for c in PACS_CLASSES}
        for _, label in entry["val"]:
            counts[PACS_CLASSES[label]] += 1
        empty = [c for c, n in counts.items() if n == 0]
        if empty:
            LOGGER.warning("  %s validation split is missing classes: %s", domain, empty)


if __name__ == "__main__":
    main()
