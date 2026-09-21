"""Controlled interventions applied on the common 224x224 RGB canvas."""

from __future__ import annotations

from typing import Iterable

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

CANVAS_SIZE = 224
DIRECTIONS = ("up", "down", "left", "right")


def to_canvas(image: Image.Image, size: int = CANVAS_SIZE) -> Image.Image:
    """Resize any image to the common RGB canvas shared by every model."""
    return image.convert("RGB").resize((size, size), Image.BICUBIC)


def grayscale(image: Image.Image) -> Image.Image:
    """Remove chroma, replicating luminance across three channels."""
    return image.convert("L").convert("RGB")


def hue_rotate(image: Image.Image, degrees: float = 90.0) -> Image.Image:
    """Rotate hue by a fixed angle, preserving geometry, luminance and saturation.

    PIL's 8-bit RGB<->HSV roundtrip is lossy by up to ~6 intensity levels even
    at zero shift, so the transform is not exactly invertible. This is far below
    the perturbation the hue shift itself introduces and is documented in
    docs/task1.md rather than silently corrected.
    """
    hsv = np.asarray(image.convert("HSV"), dtype=np.uint8).copy()
    shift = int(round((degrees % 360.0) / 360.0 * 255.0))
    hsv[..., 0] = (hsv[..., 0].astype(np.int16) + shift) % 256
    return Image.fromarray(hsv, mode="HSV").convert("RGB")


def translate(
    image: Image.Image, displacement: int, direction: str
) -> Image.Image:
    """Shift by ``displacement`` pixels using reflection padding then a crop."""
    if direction not in DIRECTIONS:
        raise ValueError(f"direction must be one of {DIRECTIONS}, got {direction!r}")
    if displacement < 0:
        raise ValueError(f"displacement must be non-negative, got {displacement}")
    if displacement == 0:
        return image.copy()

    array = np.asarray(image, dtype=np.uint8)
    height, width = array.shape[:2]
    pad = displacement

    tensor = torch.from_numpy(array.copy()).permute(2, 0, 1).unsqueeze(0).float()
    padded = F.pad(tensor, (pad, pad, pad, pad), mode="reflect")
    padded = padded.squeeze(0).permute(1, 2, 0).numpy().astype(np.uint8)

    offsets = {
        "up": (pad + displacement, pad),
        "down": (pad - displacement, pad),
        "left": (pad, pad + displacement),
        "right": (pad, pad - displacement),
    }
    top, left = offsets[direction]
    return Image.fromarray(padded[top : top + height, left : left + width])


def patch_permutation(grid: int, seed: int, index: int = 0) -> np.ndarray:
    """Return a non-identity permutation of ``grid * grid`` patch positions."""
    n = grid * grid
    if n < 2:
        raise ValueError(f"grid must be at least 2x2, got {grid}x{grid}")

    rng = np.random.default_rng(seed + index)
    for _ in range(1000):
        perm = rng.permutation(n)
        if not np.array_equal(perm, np.arange(n)):
            return perm
    raise RuntimeError("Failed to draw a non-identity permutation.")


def patch_shuffle(
    image: Image.Image, grid: int = 4, permutation: np.ndarray | None = None,
    seed: int = 6304, index: int = 0,
) -> Image.Image:
    """Divide into a ``grid x grid`` pixel-space grid and permute the patches."""
    array = np.asarray(image, dtype=np.uint8)
    height, width = array.shape[:2]
    if height % grid or width % grid:
        raise ValueError(
            f"Image {width}x{height} is not divisible by grid {grid}. "
            "Interventions must be built on the common 224x224 canvas."
        )

    if permutation is None:
        permutation = patch_permutation(grid, seed, index)
    if len(permutation) != grid * grid:
        raise ValueError(
            f"Permutation length {len(permutation)} != {grid * grid} patches."
        )

    ph, pw = height // grid, width // grid
    patches = [
        array[r * ph : (r + 1) * ph, c * pw : (c + 1) * pw]
        for r in range(grid)
        for c in range(grid)
    ]

    out = np.empty_like(array)
    for position, source in enumerate(permutation):
        r, c = divmod(position, grid)
        out[r * ph : (r + 1) * ph, c * pw : (c + 1) * pw] = patches[source]
    return Image.fromarray(out)


def to_tensor(image: Image.Image) -> torch.Tensor:
    """Convert a PIL image to a float tensor in [0, 1] without normalizing."""
    array = np.asarray(image, dtype=np.float32).copy() / 255.0
    return torch.from_numpy(array).permute(2, 0, 1)


def batch_to_tensor(images: Iterable[Image.Image]) -> torch.Tensor:
    """Stack PIL images into a ``(B, 3, H, W)`` float tensor in [0, 1]."""
    return torch.stack([to_tensor(img) for img in images])


INTERVENTION_REGISTRY = {
    "clean": lambda img, **kw: img.copy(),
    "grayscale": lambda img, **kw: grayscale(img),
    "hue_rotation": lambda img, degrees=90.0, **kw: hue_rotate(img, degrees),
    "patch_shuffle": lambda img, grid=4, seed=6304, index=0, permutation=None, **kw:
        patch_shuffle(img, grid, permutation, seed, index),
}


def apply_intervention(image: Image.Image, name: str, **kwargs) -> Image.Image:
    """Apply a registered intervention by name."""
    if name not in INTERVENTION_REGISTRY:
        raise KeyError(
            f"Unknown intervention {name!r}. "
            f"Available: {sorted(INTERVENTION_REGISTRY)}"
        )
    return INTERVENTION_REGISTRY[name](image, **kwargs)
