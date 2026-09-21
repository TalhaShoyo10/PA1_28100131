"""Tests for shared metrics.

Focus is on the edge cases that would otherwise produce a plausible-looking
but wrong number in a report table: absent classes, undefined shape bias, and
the exact threshold convention used for open-set rejection.
"""

from __future__ import annotations

import numpy as np
import pytest

from common.metrics import (
    acceptance_rate,
    accuracy,
    auroc,
    calibrate_threshold,
    coverage,
    fpr_at_95_tpr,
    macro_f1,
    mean_max_confidence,
    per_class_accuracy,
    prediction_consistency,
    rejection_rate,
    shape_bias,
    top_confusions,
)


# --------------------------------------------------------------------------
# Closed-set metrics
# --------------------------------------------------------------------------


def test_accuracy_and_macro_f1_perfect() -> None:
    y = np.array([0, 1, 2, 0, 1, 2])
    assert accuracy(y, y) == 100.0
    assert macro_f1(y, y, num_classes=3) == 100.0


def test_macro_f1_counts_absent_class_as_zero() -> None:
    """A class absent from a split must still enter the macro average.

    Otherwise a PACS validation domain missing one of the seven classes would
    report an inflated macro-F1 relative to a domain that has all seven.
    """
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0, 0, 1, 1])
    # Perfect on the two classes present, but the protocol declares 3 classes.
    assert macro_f1(y_true, y_pred, num_classes=3) == pytest.approx(200.0 / 3)
    # Without num_classes, sklearn averages only over observed labels.
    assert macro_f1(y_true, y_pred) == 100.0


def test_per_class_accuracy_absent_is_nan_not_zero() -> None:
    """'Absent' and 'always wrong' must be distinguishable."""
    y_true = np.array([0, 0, 1])
    y_pred = np.array([0, 1, 1])
    out = per_class_accuracy(y_true, y_pred, num_classes=3)
    assert out[0] == 50.0
    assert out[1] == 100.0
    assert np.isnan(out[2])


def test_empty_inputs_return_nan() -> None:
    empty = np.array([], dtype=int)
    assert np.isnan(accuracy(empty, empty))
    assert np.isnan(macro_f1(empty, empty))
    assert np.isnan(prediction_consistency(empty, empty))


def test_prediction_consistency_is_about_predictions_not_correctness() -> None:
    """Identically wrong before and after an intervention counts as consistent."""
    clean = np.array([5, 5, 3])
    transformed = np.array([5, 5, 4])
    assert prediction_consistency(clean, transformed) == pytest.approx(200.0 / 3)


def test_prediction_consistency_requires_paired_arrays() -> None:
    with pytest.raises(ValueError, match="must align"):
        prediction_consistency(np.array([1, 2, 3]), np.array([1, 2]))


def test_mean_max_confidence() -> None:
    probs = np.array([[0.7, 0.3], [0.9, 0.1]])
    assert mean_max_confidence(probs) == pytest.approx(0.8)


def test_top_confusions_orders_by_frequency_and_skips_diagonal() -> None:
    y_true = np.array([0, 0, 0, 1, 1])
    y_pred = np.array([1, 1, 0, 0, 1])
    conf = top_confusions(y_true, y_pred, num_classes=2, k=5)
    assert conf[0] == (0, 1, 2)   # most frequent off-diagonal
    assert all(t != p for t, p, _ in conf)


# --------------------------------------------------------------------------
# Task 1 shape bias
# --------------------------------------------------------------------------


def test_shape_bias_and_coverage() -> None:
    assert shape_bias(30, 10) == 75.0
    assert coverage(30, 10, 200) == 20.0


def test_shape_bias_undefined_when_no_intended_class_predicted() -> None:
    """NaN, not 0: with no shape-or-texture decisions the score is undefined.

    Reporting 0.0 here would read as 'fully texture-biased', which the data
    does not support.
    """
    assert np.isnan(shape_bias(0, 0))
    assert coverage(0, 0, 200) == 0.0


# --------------------------------------------------------------------------
# Open-set metrics
# --------------------------------------------------------------------------


def test_auroc_convention_larger_score_means_more_novel() -> None:
    """Unknowns are the positive class; a perfect detector scores 100."""
    known = np.array([0.0, 0.1, 0.2])
    unknown = np.array([0.8, 0.9, 1.0])
    assert auroc(known, unknown) == 100.0
    # Reversed separation is a perfectly wrong ranking.
    assert auroc(unknown, known) == 0.0


def test_auroc_chance_for_identical_distributions() -> None:
    x = np.array([0.1, 0.2, 0.3, 0.4])
    assert auroc(x, x) == 50.0


def test_threshold_accepts_target_fraction_of_knowns() -> None:
    """tau = 95th percentile of validation unknownness accepts ~95% of knowns."""
    rng = np.random.default_rng(6304)
    val_known = rng.normal(size=20_000)
    tau = calibrate_threshold(val_known, target_tpr=95.0)
    assert acceptance_rate(val_known, tau) == pytest.approx(95.0, abs=0.1)


def test_acceptance_and_rejection_are_complementary() -> None:
    scores = np.array([0.0, 0.5, 1.0, 1.5])
    tau = 0.75
    assert acceptance_rate(scores, tau) == 50.0
    assert rejection_rate(scores, tau) == 50.0


def test_fpr_at_95_tpr_equals_unknown_acceptance() -> None:
    """Under this assignment's convention the two are the same quantity."""
    unknown = np.array([0.1, 0.2, 5.0, 6.0])
    tau = 1.0
    assert fpr_at_95_tpr(unknown, tau) == acceptance_rate(unknown, tau) == 50.0


def test_threshold_boundary_is_inclusive() -> None:
    """Assignment says accept when u(x) <= tau, so equality is accepted."""
    assert acceptance_rate(np.array([1.0]), 1.0) == 100.0


def test_calibrate_threshold_rejects_empty_input() -> None:
    with pytest.raises(ValueError, match="empty"):
        calibrate_threshold(np.array([]))
