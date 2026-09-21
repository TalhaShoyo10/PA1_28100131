"""Shared plotting defaults for report figures."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

PALETTE: Sequence[str] = (
    "#4C72B0",
    "#DD8452",
    "#55A868",
    "#C44E52",
    "#8172B3",
    "#937860",
    "#DA8BC3",
)

METHOD_COLORS = {
    "source_only": PALETTE[0],
    "erm": PALETTE[0],
    "dan": PALETTE[1],
    "dan_dg": PALETTE[1],
    "dann": PALETTE[2],
    "sam": PALETTE[2],
    "cdan": PALETTE[3],
    "vanilla": PALETTE[0],
    "gcsc": PALETTE[1],
    "proser": PALETTE[2],
    "rpl": PALETTE[3],
}

BACKBONE_COLORS = {
    "resnet50": PALETTE[0],
    "vit_b16": PALETTE[1],
    "clip_vitb32": PALETTE[2],
    "clip_zeroshot": PALETTE[4],
}


def set_style() -> None:
    """Apply the shared matplotlib rcParams. Call once per figure script."""
    plt.rcParams.update(
        {
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "legend.fontsize": 8,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "axes.grid": True,
            "grid.alpha": 0.3,
            "grid.linewidth": 0.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "lines.linewidth": 1.6,
            "lines.markersize": 5,
            "axes.prop_cycle": plt.cycler(color=list(PALETTE)),
        }
    )


def save_figure(fig, path: Path | str, also_pdf: bool = True) -> Path:
    """Save ``fig`` as PNG (and PDF for vector inclusion in the report)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path.with_suffix(".png"))
    if also_pdf:
        fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)
    return path.with_suffix(".png")
