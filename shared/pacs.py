"""PACS dataset access for Tasks 2 and 3."""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Sequence

from PIL import Image
from torch.utils.data import Dataset

PACS_CLASSES: tuple[str, ...] = (
    "dog",
    "elephant",
    "giraffe",
    "guitar",
    "horse",
    "house",
    "person",
)

PACS_DOMAINS: tuple[str, ...] = ("photo", "art_painting", "cartoon", "sketch")

CLASS_TO_IDX: dict[str, int] = {c: i for i, c in enumerate(PACS_CLASSES)}

_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}


class TargetAccessViolation(RuntimeError):
    """Raised when a protocol-forbidden domain is about to be loaded."""


def _domain_dir(root: Path, domain: str) -> Path:
    """Resolve a domain directory, tolerating minor naming variants."""
    candidates = [domain, domain.replace("_", " "), domain.replace("_", "-")]
    if domain == "art_painting":
        candidates += ["art painting", "artpainting", "art"]
    for name in candidates:
        path = root / name
        if path.is_dir():
            return path
    available = sorted(p.name for p in root.iterdir() if p.is_dir()) if root.is_dir() else []
    raise FileNotFoundError(
        f"Domain directory for {domain!r} not found under {root}. "
        f"Available: {available}. See README for the expected PACS layout."
    )


def list_domain_images(root: Path | str, domain: str) -> list[tuple[str, int]]:
    """Return ``(relative_path, label)`` for every image in ``domain``."""
    root = Path(root)
    ddir = _domain_dir(root, domain)

    items: list[tuple[str, int]] = []
    for class_name in PACS_CLASSES:
        cdir = ddir / class_name
        if not cdir.is_dir():
            raise FileNotFoundError(f"Missing class directory: {cdir}")
        files = sorted(
            p for p in cdir.iterdir() if p.suffix.lower() in _IMAGE_SUFFIXES
        )
        if not files:
            raise FileNotFoundError(f"No images found in {cdir}")
        for p in files:
            items.append((p.relative_to(root).as_posix(), CLASS_TO_IDX[class_name]))

    return sorted(items)


class PACSDataset(Dataset):
    """A PACS split defined by an explicit list of relative image paths."""

    def __init__(
        self,
        root: Path | str,
        samples: Sequence[tuple[str, int]],
        transform: Callable | None = None,
        domain: str | None = None,
        target_access_forbidden: bool = False,
        forbidden_domain: str = "sketch",
    ) -> None:
        if target_access_forbidden and domain == forbidden_domain:
            raise TargetAccessViolation(
                f"Refusing to load domain {domain!r}: this code path is marked "
                "target_access_forbidden. In Task 3 the target domain may be "
                "loaded only by evaluate_sketch.py, after every training, "
                "diagnostic and checkpoint-selection decision has been fixed."
            )

        self.root = Path(root)
        self.samples = list(samples)
        self.transform = transform
        self.domain = domain

        if not self.samples:
            raise ValueError(f"Empty sample list for domain {domain!r}.")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        rel_path, label = self.samples[index]
        with Image.open(self.root / rel_path) as img:
            image = img.convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, label

    @property
    def labels(self) -> list[int]:
        return [label for _, label in self.samples]

    def class_counts(self) -> dict[str, int]:
        counts = {c: 0 for c in PACS_CLASSES}
        for _, label in self.samples:
            counts[PACS_CLASSES[label]] += 1
        return counts

    def __repr__(self) -> str:
        return (
            f"PACSDataset(domain={self.domain!r}, n={len(self)}, "
            f"root={str(self.root)!r})"
        )
