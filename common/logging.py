"""Run logging and machine-readable result persistence."""

from __future__ import annotations

import csv
import json
import logging
import platform
import subprocess
import sys
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

_LOG_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"


def get_logger(name: str, logfile: Path | str | None = None) -> logging.Logger:
    """Return a configured logger that also tees to ``logfile`` when given."""
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)

    stream = logging.StreamHandler(sys.stdout)
    stream.setFormatter(logging.Formatter(_LOG_FORMAT))
    logger.addHandler(stream)

    if logfile is not None:
        logfile = Path(logfile)
        logfile.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(logfile, encoding="utf-8")
        fh.setFormatter(logging.Formatter(_LOG_FORMAT))
        logger.addHandler(fh)

    logger.propagate = False
    return logger


def _git_commit() -> str:
    """Short git SHA, with a ``-dirty`` suffix when the tree has changes."""
    try:
        sha = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL, text=True
        ).strip()
        dirty = subprocess.call(
            ["git", "diff", "--quiet", "--ignore-submodules", "HEAD"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        return f"{sha}-dirty" if dirty != 0 else sha
    except Exception:
        return "unknown"


def environment_metadata() -> dict[str, Any]:
    """Collect reproducibility provenance for the current process."""
    meta: dict[str, Any] = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_commit": _git_commit(),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
    }
    try:
        import torch

        meta["torch"] = torch.__version__
        meta["cuda_available"] = bool(torch.cuda.is_available())
        meta["device_name"] = (
            torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
        )
    except Exception:
        meta["torch"] = "unavailable"
    return meta


@dataclass
class RunRecord:
    """A single experiment run's identity, config and final metrics."""

    run_name: str
    task: str
    config: Mapping[str, Any] = field(default_factory=dict)
    metrics: dict[str, Any] = field(default_factory=dict)
    notes: str = ""
    environment: dict[str, Any] = field(default_factory=environment_metadata)

    def save(self, path: Path | str) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = asdict(self)
        payload["config"] = dict(self.config)
        path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
        return path


def save_json(obj: Any, path: Path | str) -> Path:
    """Write ``obj`` as indented JSON, creating parent directories."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=str), encoding="utf-8")
    return path


def save_csv(rows: Sequence[Mapping[str, Any]], path: Path | str) -> Path:
    """Write a list of uniform dicts as CSV."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return path

    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)

    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return path


class MetricHistory:
    """Accumulates per-epoch metrics and dumps them as a tidy CSV."""

    def __init__(self) -> None:
        self.rows: list[dict[str, Any]] = []

    def append(self, **kwargs: Any) -> None:
        self.rows.append(dict(kwargs))

    def save(self, path: Path | str) -> Path:
        return save_csv(self.rows, path)

    def __len__(self) -> int:
        return len(self.rows)
