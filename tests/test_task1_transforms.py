"""Tests for Task 1 controlled interventions.

Each intervention claims to change ONE visual factor while preserving others.
These tests verify those claims hold, because the hypotheses in
task1/configs/interventions.yaml are only meaningful if the transforms do what
they say.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from PIL import Image

from task1.data.transforms import (
    CANVAS_SIZE,
    DIRECTIONS,
    apply_intervention,
    batch_to_tensor,
    grayscale,
    hue_rotate,
    patch_permutation,
    patch_shuffle,
    to_canvas,
    to_tensor,
    translate,
)

#: PIL's 8-bit RGB<->HSV roundtrip is lossy even at zero shift.
HSV_ROUNDTRIP_TOLERANCE = 8


@pytest.fixture
def image() -> Image.Image:
    rng = np.random.default_rng(6304)
    return Image.fromarray(rng.integers(0, 255, (224, 224, 3), dtype=np.uint8))


def arr(img: Image.Image) -> np.ndarray:
    return np.asarray(img).astype(int)


# --------------------------------------------------------------------------
# Canvas
# --------------------------------------------------------------------------


def test_canvas_is_224_rgb() -> None:
    img = Image.new("L", (300, 200))
    out = to_canvas(img)
    assert out.size == (CANVAS_SIZE, CANVAS_SIZE)
    assert out.mode == "RGB"


# --------------------------------------------------------------------------
# Grayscale: removes colour, preserves geometry
# --------------------------------------------------------------------------


def test_grayscale_makes_all_channels_equal(image) -> None:
    a = arr(grayscale(image))
    assert np.array_equal(a[..., 0], a[..., 1])
    assert np.array_equal(a[..., 1], a[..., 2])


def test_grayscale_preserves_size_and_mode(image) -> None:
    out = grayscale(image)
    assert out.size == image.size and out.mode == "RGB"


def test_grayscale_preserves_luminance_structure(image) -> None:
    """Geometry must survive: the luminance channel is essentially unchanged."""
    original_luma = np.asarray(image.convert("L")).astype(int)
    gray_luma = np.asarray(grayscale(image).convert("L")).astype(int)
    assert np.abs(original_luma - gray_luma).max() <= 2


# --------------------------------------------------------------------------
# Hue rotation: changes colour, preserves everything else
# --------------------------------------------------------------------------


def test_hue_rotation_preserves_value_and_saturation(image) -> None:
    """The design claim: only the hue angle moves."""
    before = np.asarray(image.convert("HSV")).astype(int)
    after = np.asarray(hue_rotate(image, 90).convert("HSV")).astype(int)
    assert np.abs(before[..., 1] - after[..., 1]).max() <= 2   # saturation
    assert np.abs(before[..., 2] - after[..., 2]).max() <= 2   # value


def test_hue_rotation_actually_changes_hue(image) -> None:
    before = np.asarray(image.convert("HSV")).astype(int)
    after = np.asarray(hue_rotate(image, 90).convert("HSV")).astype(int)
    assert np.abs(before[..., 0] - after[..., 0]).mean() > 10


def test_hue_rotation_preserves_geometry(image) -> None:
    """Edge structure must survive, or this stops being a pure colour test.

    Geometry is measured on the HSV V channel, NOT on ``convert("L")``. Luma
    is a channel-WEIGHTED mix (0.299R + 0.587G + 0.114B) while V is
    ``max(R,G,B)``. Rotating hue permutes which channel is the max, so luma
    necessarily moves even when the shape content is untouched: pure red
    (255,0,0) -> green (0,255,0) keeps V at 255 but shifts luma 76 -> 150.
    Testing on luma would therefore fail for a perfectly correct transform.
    """
    before = np.asarray(image.convert("HSV")).astype(int)[..., 2]
    after = np.asarray(hue_rotate(image, 90).convert("HSV")).astype(int)[..., 2]

    assert np.abs(before - after).max() <= 2

    def gradient_energy(a: np.ndarray) -> float:
        return float(np.abs(np.diff(a, axis=0)).mean() + np.abs(np.diff(a, axis=1)).mean())

    assert gradient_energy(after) == pytest.approx(gradient_energy(before), rel=0.02)


def test_hue_rotation_by_360_returns_to_start(image) -> None:
    """Within HSV quantisation error -- the shift arithmetic wraps correctly."""
    out = hue_rotate(image, 360)
    assert np.abs(arr(out) - arr(image)).max() <= HSV_ROUNDTRIP_TOLERANCE


def test_hue_rotation_is_deterministic(image) -> None:
    assert np.array_equal(arr(hue_rotate(image, 90)), arr(hue_rotate(image, 90)))


# --------------------------------------------------------------------------
# Translation
# --------------------------------------------------------------------------


def test_zero_displacement_is_identity(image) -> None:
    for direction in DIRECTIONS:
        assert np.array_equal(arr(translate(image, 0, direction)), arr(image))


@pytest.mark.parametrize("displacement", [8, 16, 32])
@pytest.mark.parametrize("direction", DIRECTIONS)
def test_translation_preserves_size(image, displacement, direction) -> None:
    assert translate(image, displacement, direction).size == image.size


@pytest.mark.parametrize("direction", DIRECTIONS)
def test_reflection_padding_leaves_no_blank_border(image, direction) -> None:
    """Zero padding would add black bars, which is a different intervention."""
    out = arr(translate(image, 32, direction))
    if direction == "up":
        edge = out[-32:, :, :]
    elif direction == "down":
        edge = out[:32, :, :]
    elif direction == "left":
        edge = out[:, -32:, :]
    else:
        edge = out[:, :32, :]
    assert edge.std() > 1.0
    assert not np.all(edge == 0)


def test_translation_actually_moves_content(image) -> None:
    assert not np.array_equal(arr(translate(image, 16, "up")), arr(image))


def test_opposite_directions_differ(image) -> None:
    assert not np.array_equal(
        arr(translate(image, 16, "up")), arr(translate(image, 16, "down"))
    )


def test_translation_rejects_bad_arguments(image) -> None:
    with pytest.raises(ValueError, match="direction"):
        translate(image, 8, "diagonal")
    with pytest.raises(ValueError, match="non-negative"):
        translate(image, -8, "up")


# --------------------------------------------------------------------------
# Patch shuffle
# --------------------------------------------------------------------------


def test_permutation_is_valid_and_non_identity() -> None:
    perm = patch_permutation(4, 6304)
    assert sorted(perm.tolist()) == list(range(16))
    assert not np.array_equal(perm, np.arange(16))


def test_permutation_is_deterministic() -> None:
    assert np.array_equal(patch_permutation(4, 6304), patch_permutation(4, 6304))


def test_permutation_varies_per_image_index() -> None:
    """One non-identity permutation PER IMAGE, not one reused everywhere."""
    assert not np.array_equal(
        patch_permutation(4, 6304, index=0), patch_permutation(4, 6304, index=7)
    )


def test_patch_shuffle_conserves_every_pixel(image) -> None:
    """The intervention rearranges pixels; it must not create or destroy any."""
    out = patch_shuffle(image, 4, seed=6304)
    assert np.array_equal(
        np.sort(np.asarray(out).ravel()), np.sort(np.asarray(image).ravel())
    )


def test_patch_shuffle_changes_the_image(image) -> None:
    assert not np.array_equal(arr(patch_shuffle(image, 4, seed=6304)), arr(image))


def test_same_permutation_gives_identical_output(image) -> None:
    """Required: the SAME shuffled images must be reusable across all models."""
    perm = patch_permutation(4, 6304)
    a = patch_shuffle(image, 4, permutation=perm)
    b = patch_shuffle(image, 4, permutation=perm)
    assert np.array_equal(arr(a), arr(b))


def test_patch_shuffle_preserves_patch_contents(image) -> None:
    """Local evidence survives; only global arrangement is destroyed."""
    perm = patch_permutation(4, 6304)
    out = np.asarray(patch_shuffle(image, 4, permutation=perm))
    src = np.asarray(image)
    size = 224 // 4
    for position, source in enumerate(perm):
        pr, pc = divmod(position, 4)
        sr, sc = divmod(int(source), 4)
        placed = out[pr * size : (pr + 1) * size, pc * size : (pc + 1) * size]
        original = src[sr * size : (sr + 1) * size, sc * size : (sc + 1) * size]
        assert np.array_equal(placed, original)


def test_patch_shuffle_rejects_indivisible_size() -> None:
    img = Image.new("RGB", (225, 225))
    with pytest.raises(ValueError, match="not divisible"):
        patch_shuffle(img, 4)


def test_patch_shuffle_rejects_wrong_permutation_length(image) -> None:
    with pytest.raises(ValueError, match="Permutation length"):
        patch_shuffle(image, 4, permutation=np.arange(9))


# --------------------------------------------------------------------------
# Tensor conversion
# --------------------------------------------------------------------------


def test_to_tensor_shape_and_range(image) -> None:
    t = to_tensor(image)
    assert t.shape == (3, 224, 224)
    assert t.dtype == torch.float32
    assert 0.0 <= t.min() and t.max() <= 1.0


def test_to_tensor_does_not_normalize(image) -> None:
    """Interventions produce raw [0,1]; each backbone normalizes afterwards."""
    white = Image.new("RGB", (8, 8), (255, 255, 255))
    assert torch.allclose(to_tensor(white), torch.ones(3, 8, 8))


def test_batch_to_tensor_stacks(image) -> None:
    assert batch_to_tensor([image, image, image]).shape == (3, 3, 224, 224)


# --------------------------------------------------------------------------
# Registry
# --------------------------------------------------------------------------


def test_registry_dispatch(image) -> None:
    assert np.array_equal(arr(apply_intervention(image, "clean")), arr(image))
    assert np.array_equal(
        arr(apply_intervention(image, "grayscale")), arr(grayscale(image))
    )
    assert np.array_equal(
        arr(apply_intervention(image, "hue_rotation", degrees=90)),
        arr(hue_rotate(image, 90)),
    )


def test_registry_rejects_unknown_name(image) -> None:
    with pytest.raises(KeyError, match="Unknown intervention"):
        apply_intervention(image, "rotate_180")


# --------------------------------------------------------------------------
# Translation curve
# --------------------------------------------------------------------------


def test_translation_curve_anchors_at_full_consistency() -> None:
    """Regression guard (2026-09-23).

    At displacement 0 the only entry is the clean baseline, whose
    consistency_vs_clean is NaN by construction. np.nanmean over a lone NaN
    warned 'Mean of empty slice' and produced NaN, leaving the curve without
    its anchor. Consistency at delta=0 is 100 by definition.
    """
    from task1.analysis.evaluate_bias import ConditionResult, translation_curve

    clean = ConditionResult(
        "resnet50", "clean", 95.0, 94.0, 0.9, float("nan"), float("nan"), 500
    )
    shifted = [
        ConditionResult("resnet50", f"translate_8_{d}", 90.0, 89.0, 0.85, 92.0, -5.0, 500)
        for d in ("up", "down", "left", "right")
    ]

    rows = translation_curve({0: [clean], 8: shifted})

    assert rows[0]["displacement"] == 0
    assert rows[0]["consistency"] == 100.0
    assert rows[1]["consistency"] == 92.0
    assert rows[1]["n_directions"] == 4


def test_translation_curve_emits_no_warnings() -> None:
    import warnings

    from task1.analysis.evaluate_bias import ConditionResult, translation_curve

    clean = ConditionResult(
        "vit_b16", "clean", 95.0, 94.0, 0.9, float("nan"), float("nan"), 500
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        translation_curve({0: [clean]})
