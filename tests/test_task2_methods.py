"""Tests for the four Task 2 adaptation methods.

Checks that each method routes gradients where its definition requires, keeps
the source-only baseline target-blind, and honours the assignment's explicit
prohibitions for CDAN.
"""

from __future__ import annotations

import pytest
import torch

from task2.methods.base import AdaptationMethod
from task2.methods.cdan import CDAN
from task2.methods.dan import DAN
from task2.methods.dann import DANN
from task2.methods.source_only import SourceOnly

B, D, C = 24, 512, 7


@pytest.fixture
def batch():
    torch.manual_seed(6304)
    return {
        "source_features": torch.randn(B, D, requires_grad=True),
        "source_logits": torch.randn(B, C, requires_grad=True),
        "source_labels": torch.randint(0, C, (B,)),
        "target_features": (torch.randn(B, D) + 1.5).requires_grad_(True),
        "target_logits": torch.randn(B, C, requires_grad=True),
    }


def _all_methods():
    return [SourceOnly(), DAN(lambda_mmd=1.0), DANN(), CDAN()]


def _call(method, batch, progress=0.5):
    kwargs = dict(
        target_features=batch["target_features"],
        target_logits=batch["target_logits"],
        progress=progress,
    )
    if not method.uses_target:
        kwargs = {}
    return method.compute_loss(
        batch["source_features"], batch["source_logits"], batch["source_labels"], **kwargs
    )


# --------------------------------------------------------------------------
# Shared behaviour
# --------------------------------------------------------------------------


@pytest.mark.parametrize("method", _all_methods(), ids=lambda m: m.name)
def test_loss_is_finite_and_scalar(method: AdaptationMethod, batch) -> None:
    out = _call(method, batch)
    assert out.loss.ndim == 0
    assert torch.isfinite(out.loss)
    assert "cls_loss" in out.components
    assert all(isinstance(v, float) for v in out.components.values())


@pytest.mark.parametrize("method", _all_methods(), ids=lambda m: m.name)
def test_every_method_trains_the_classifier_head(method, batch) -> None:
    _call(method, batch).loss.backward()
    assert batch["source_logits"].grad.abs().sum() > 0


@pytest.mark.parametrize("method", _all_methods(), ids=lambda m: m.name)
def test_only_alignment_methods_touch_features_directly(method, batch) -> None:
    """Source-only's loss is defined on logits alone; the others align features."""
    _call(method, batch).loss.backward()
    grad = batch["source_features"].grad
    has_feature_grad = grad is not None and grad.abs().sum() > 0
    assert has_feature_grad == method.uses_target


@pytest.mark.parametrize("method", [DAN(), DANN(), CDAN()], ids=lambda m: m.name)
def test_alignment_methods_push_gradient_into_target_features(method, batch) -> None:
    _call(method, batch).loss.backward()
    assert batch["target_features"].grad.abs().sum() > 0


@pytest.mark.parametrize("method", [DAN(), DANN(), CDAN()], ids=lambda m: m.name)
def test_alignment_methods_require_target_data(method, batch) -> None:
    with pytest.raises(ValueError, match="(?i)target"):
        method.compute_loss(
            batch["source_features"], batch["source_logits"], batch["source_labels"]
        )


# --------------------------------------------------------------------------
# Source-only
# --------------------------------------------------------------------------


def test_source_only_refuses_target_features(batch) -> None:
    """Guard: this checkpoint is also Task 3's ERM baseline."""
    with pytest.raises(ValueError, match="must never"):
        SourceOnly().compute_loss(
            batch["source_features"],
            batch["source_logits"],
            batch["source_labels"],
            target_features=batch["target_features"],
        )


def test_source_only_loss_is_plain_cross_entropy(batch) -> None:
    out = _call(SourceOnly(), batch)
    expected = torch.nn.functional.cross_entropy(
        batch["source_logits"], batch["source_labels"]
    )
    assert out.loss.item() == pytest.approx(expected.item(), rel=1e-6)
    assert set(out.components) == {"cls_loss"}


# --------------------------------------------------------------------------
# DAN
# --------------------------------------------------------------------------


def test_dan_detects_a_real_domain_gap(batch) -> None:
    """MMD must be materially non-zero on shifted features, not a constant."""
    out = _call(DAN(lambda_mmd=1.0), batch)
    assert out.components["mmd_loss"] > 0.1


def test_dan_lambda_scales_only_the_penalty(batch) -> None:
    losses = {}
    for lam in (0.1, 1.0, 10.0):
        b = {k: (v.detach().requires_grad_(True) if v.dtype.is_floating_point else v)
             for k, v in batch.items()}
        out = _call(DAN(lambda_mmd=lam), b)
        losses[lam] = out
        # The unweighted discrepancy is a property of the data, not of lambda.
        assert out.components["mmd_loss"] == pytest.approx(
            losses[0.1].components["mmd_loss"], rel=1e-5
        )
    # Higher lambda -> strictly larger total loss for a positive discrepancy.
    assert (
        losses[0.1].loss.item() < losses[1.0].loss.item() < losses[10.0].loss.item()
    )


def test_dan_with_lambda_zero_reduces_to_source_only(batch) -> None:
    dan = _call(DAN(lambda_mmd=0.0), batch)
    expected = torch.nn.functional.cross_entropy(
        batch["source_logits"], batch["source_labels"]
    )
    assert dan.loss.item() == pytest.approx(expected.item(), rel=1e-6)


def test_dan_rejects_negative_lambda() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        DAN(lambda_mmd=-1.0)


# --------------------------------------------------------------------------
# DANN
# --------------------------------------------------------------------------


def test_dann_logs_the_diagnostics_the_assignment_asks_for(batch) -> None:
    """Domain accuracy near chance is ambiguous, so it must be logged."""
    out = _call(DANN(), batch)
    for key in ("domain_loss", "domain_acc", "alpha"):
        assert key in out.components
    assert 0.0 <= out.components["domain_acc"] <= 1.0


def test_dann_alpha_is_zero_at_training_start(batch) -> None:
    """At p=0 the reversed gradient must not reach the backbone yet."""
    out = _call(DANN(), batch, progress=0.0)
    assert out.components["alpha"] == pytest.approx(0.0, abs=1e-9)
    out.loss.backward()
    assert batch["target_features"].grad.abs().sum() < 1e-8


def test_dann_alpha_grows_with_progress(batch) -> None:
    early = _call(DANN(), batch, progress=0.1).components["alpha"]
    late = _call(DANN(), batch, progress=0.9).components["alpha"]
    assert early < late


def test_dann_alpha_max_caps_reversal_strength(batch) -> None:
    """Used by the alternative controlled study; schedule shape is preserved."""
    full = _call(DANN(alpha_max=1.0), batch, progress=0.5).components["alpha"]
    half = _call(DANN(alpha_max=0.5), batch, progress=0.5).components["alpha"]
    assert half == pytest.approx(0.5 * full, rel=1e-9)


def test_dann_discriminator_sees_the_raw_feature(batch) -> None:
    assert DANN().discriminator.input_dim == D


# --------------------------------------------------------------------------
# CDAN
# --------------------------------------------------------------------------


def test_cdan_discriminator_width_is_feature_times_classes() -> None:
    assert CDAN().discriminator.input_dim == D * C == 3584


def test_cdan_does_not_detach_predictions(batch) -> None:
    """Explicitly forbidden: detaching p is a common shortcut that would
    silently change the method by cutting the head out of the adversarial game."""
    _call(CDAN(), batch).loss.backward()
    assert batch["target_logits"].grad is not None
    assert batch["target_logits"].grad.abs().sum() > 0


def test_cdan_does_not_detach_features(batch) -> None:
    _call(CDAN(), batch).loss.backward()
    assert batch["source_features"].grad.abs().sum() > 0
    assert batch["target_features"].grad.abs().sum() > 0


def test_cdan_requires_target_logits(batch) -> None:
    with pytest.raises(ValueError, match="(?i)target"):
        CDAN().compute_loss(
            batch["source_features"],
            batch["source_logits"],
            batch["source_labels"],
            target_features=batch["target_features"],
            target_logits=None,
        )


def test_cdan_shares_dann_hyperparameters() -> None:
    """Only the conditioning may differ, so the comparison stays controlled."""
    dann, cdan = DANN(), CDAN()
    assert cdan.gamma == dann.gamma
    assert cdan.alpha_max == dann.alpha_max
    assert cdan.lambda_domain == dann.lambda_domain
    for disc in (dann.discriminator, cdan.discriminator):
        linears = [m for m in disc.net if isinstance(m, torch.nn.Linear)]
        assert linears[0].out_features == 256
        assert linears[-1].out_features == 2
