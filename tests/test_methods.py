"""Tests for MMD, gradient reversal, and CDAN conditioning.

These check mathematical properties that a wrong implementation would still
"run" without: MMD vanishing for identical distributions and growing with
separation, the GRL actually negating gradients, and the multilinear map
matching an explicit outer product.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from shared.mmd import (
    median_bandwidth,
    mmd2,
    multi_rbf_kernel,
    pairwise_domain_mmd2,
    pairwise_squared_distances,
)
from task2.models.domain_discriminator import (
    DomainDiscriminator,
    dann_alpha,
    grad_reverse,
    multilinear_map,
)


# --------------------------------------------------------------------------
# MMD
# --------------------------------------------------------------------------


def test_pairwise_distances_match_manual_computation() -> None:
    x = torch.tensor([[0.0, 0.0], [3.0, 4.0]])
    d2 = pairwise_squared_distances(x, x)
    assert d2[0, 0].item() == pytest.approx(0.0, abs=1e-5)
    assert d2[0, 1].item() == pytest.approx(25.0, abs=1e-4)   # 3-4-5 triangle
    assert torch.allclose(d2, d2.T, atol=1e-5)                # symmetric


def test_mmd_is_near_zero_for_identical_samples() -> None:
    x = torch.randn(64, 16)
    assert mmd2(x, x).item() == pytest.approx(0.0, abs=1e-5)


def test_mmd_is_non_negative() -> None:
    """The biased estimator must stay >= 0 to behave as a penalty."""
    torch.manual_seed(6304)
    for _ in range(10):
        a, b = torch.randn(32, 8), torch.randn(32, 8) + torch.randn(1).item()
        assert mmd2(a, b).item() >= -1e-6


def test_mmd_grows_with_separation() -> None:
    """The core property: more distributional distance -> larger penalty."""
    torch.manual_seed(6304)
    base = torch.randn(128, 16)
    values = [mmd2(base, torch.randn(128, 16) + shift).item() for shift in (0.0, 1.0, 5.0)]
    assert values[0] < values[1] < values[2]


def test_mmd_is_symmetric() -> None:
    torch.manual_seed(6304)
    a, b = torch.randn(48, 8), torch.randn(48, 8) + 1.5
    assert mmd2(a, b).item() == pytest.approx(mmd2(b, a).item(), rel=1e-5)


def test_mmd_is_differentiable_wrt_both_inputs() -> None:
    # Both must be LEAF tensors: `torch.randn(..., requires_grad=True) + 2.0`
    # produces a non-leaf whose .grad is never populated, which would make this
    # test assert nothing about the second argument.
    a = torch.randn(32, 8, requires_grad=True)
    b = (torch.randn(32, 8) + 2.0).requires_grad_(True)
    assert a.is_leaf and b.is_leaf
    mmd2(a, b).backward()
    assert a.grad is not None and a.grad.abs().sum() > 0
    assert b.grad is not None and b.grad.abs().sum() > 0


def test_mmd_handles_unequal_batch_sizes() -> None:
    """Source and target streams need not align exactly in every batch."""
    assert mmd2(torch.randn(24, 8), torch.randn(17, 8)).item() >= -1e-6


def test_mmd_rejects_dimension_mismatch() -> None:
    with pytest.raises(ValueError, match="Feature dimension mismatch"):
        mmd2(torch.randn(8, 16), torch.randn(8, 32))


def test_mmd_rejects_non_2d_input() -> None:
    with pytest.raises(ValueError, match="2-D features"):
        mmd2(torch.randn(8, 4, 4), torch.randn(8, 16))


def test_median_bandwidth_excludes_self_distances() -> None:
    """Including the zero diagonal would bias the bandwidth downward."""
    x = torch.tensor([[0.0], [1.0], [2.0]])
    # Off-diagonal squared distances: 1, 4, 1, 1, 4, 1 -> median 1.0
    assert median_bandwidth(x).item() == pytest.approx(1.0)


def test_median_bandwidth_survives_collapsed_features() -> None:
    """Identical features give median 0; must fall back, not produce NaN."""
    collapsed = torch.ones(16, 8)
    assert median_bandwidth(collapsed).item() == 1.0
    assert torch.isfinite(mmd2(collapsed, collapsed)).item()


def test_median_bandwidth_needs_two_examples() -> None:
    with pytest.raises(ValueError, match="at least 2"):
        median_bandwidth(torch.randn(1, 8))


def test_multi_rbf_sums_three_kernels() -> None:
    """Identical points give k=1 per kernel, so the sum is the kernel count."""
    x = torch.zeros(3, 4)
    k = multi_rbf_kernel(x, x, torch.tensor(1.0))
    assert torch.allclose(k, torch.full((3, 3), 3.0), atol=1e-5)


def test_bandwidth_is_detached_from_the_graph() -> None:
    """The median heuristic is a heuristic, not a learnable parameter."""
    a = torch.randn(16, 8, requires_grad=True)
    b = torch.randn(16, 8, requires_grad=True)
    loss = mmd2(a, b)
    loss.backward()
    assert torch.isfinite(a.grad).all()


# --------------------------------------------------------------------------
# Pairwise domain MMD (Task 3 DAN-DG)
# --------------------------------------------------------------------------


def test_pairwise_covers_all_three_unordered_pairs() -> None:
    torch.manual_seed(6304)
    feats = {
        "photo": torch.randn(16, 8),
        "art_painting": torch.randn(16, 8) + 1.0,
        "cartoon": torch.randn(16, 8) + 2.0,
    }
    mean, per_pair = pairwise_domain_mmd2(feats)

    assert len(per_pair) == 3
    assert set(per_pair) == {"art_painting__cartoon", "art_painting__photo", "cartoon__photo"}
    assert mean.item() == pytest.approx(float(np.mean(list(per_pair.values()))), rel=1e-5)


def test_pairwise_is_zero_for_identical_domains() -> None:
    x = torch.randn(32, 8)
    mean, _ = pairwise_domain_mmd2({"a": x, "b": x.clone(), "c": x.clone()})
    assert mean.item() == pytest.approx(0.0, abs=1e-5)


def test_pairwise_requires_two_domains() -> None:
    with pytest.raises(ValueError, match="at least 2 domains"):
        pairwise_domain_mmd2({"photo": torch.randn(8, 4)})


def test_pairwise_is_differentiable() -> None:
    feats = {n: torch.randn(16, 8, requires_grad=True) for n in ("a", "b", "c")}
    mean, _ = pairwise_domain_mmd2(feats)
    mean.backward()
    assert all(f.grad is not None and f.grad.abs().sum() > 0 for f in feats.values())


# --------------------------------------------------------------------------
# Gradient reversal
# --------------------------------------------------------------------------


def test_grl_is_identity_in_the_forward_pass() -> None:
    x = torch.randn(8, 4)
    assert torch.allclose(grad_reverse(x, 0.7), x)


def test_grl_negates_and_scales_the_gradient() -> None:
    """The defining behaviour: backbone gets -alpha times the discriminator gradient."""
    x = torch.ones(4, 3, requires_grad=True)
    (grad_reverse(x, alpha=2.5) * 3.0).sum().backward()
    # d/dx of 3x is 3; reversed and scaled by 2.5 -> -7.5
    assert torch.allclose(x.grad, torch.full_like(x.grad, -7.5))


def test_grl_with_alpha_zero_blocks_the_gradient() -> None:
    x = torch.ones(4, 3, requires_grad=True)
    grad_reverse(x, alpha=0.0).sum().backward()
    assert torch.allclose(x.grad, torch.zeros_like(x.grad))


# --------------------------------------------------------------------------
# DANN schedule
# --------------------------------------------------------------------------


def test_alpha_schedule_endpoints() -> None:
    assert dann_alpha(0.0) == pytest.approx(0.0, abs=1e-9)
    # 2/(1+e^-10) - 1 ~ 0.99991
    assert dann_alpha(1.0) == pytest.approx(0.9999, abs=1e-3)


def test_alpha_schedule_is_monotone_increasing() -> None:
    values = [dann_alpha(p) for p in np.linspace(0, 1, 25)]
    assert all(b >= a for a, b in zip(values, values[1:]))


def test_alpha_max_caps_the_schedule_without_changing_its_shape() -> None:
    """The controlled study varies the cap but keeps the schedule shape."""
    for p in (0.25, 0.5, 0.75, 1.0):
        full = dann_alpha(p, alpha_max=1.0)
        for cap in (0.25, 0.5):
            assert dann_alpha(p, alpha_max=cap) == pytest.approx(cap * full, rel=1e-9)


def test_alpha_rejects_out_of_range_progress() -> None:
    for bad in (-0.1, 1.1):
        with pytest.raises(ValueError, match="progress"):
            dann_alpha(bad)


# --------------------------------------------------------------------------
# Discriminator and CDAN conditioning
# --------------------------------------------------------------------------


def test_discriminator_shapes_for_dann_and_cdan() -> None:
    assert DomainDiscriminator(512)(torch.randn(6, 512)).shape == (6, 2)
    assert DomainDiscriminator(512 * 7)(torch.randn(6, 3584)).shape == (6, 2)


def test_discriminator_has_the_specified_architecture() -> None:
    d = DomainDiscriminator(512, hidden_dim=256, dropout=0.5)
    linears = [m for m in d.net if isinstance(m, torch.nn.Linear)]
    assert [l.out_features for l in linears] == [256, 2]
    dropouts = [m for m in d.net if isinstance(m, torch.nn.Dropout)]
    assert len(dropouts) == 1 and dropouts[0].p == 0.5


def test_multilinear_map_matches_explicit_outer_product() -> None:
    f = torch.tensor([[1.0, 2.0]])
    p = torch.tensor([[0.3, 0.7]])
    g = multilinear_map(f, p)
    assert g.shape == (1, 4)
    # vec of [[1*0.3, 1*0.7], [2*0.3, 2*0.7]]
    assert torch.allclose(g, torch.tensor([[0.3, 0.7, 0.6, 1.4]]), atol=1e-6)


def test_multilinear_map_output_dimension() -> None:
    g = multilinear_map(torch.randn(5, 512), torch.softmax(torch.randn(5, 7), dim=1))
    assert g.shape == (5, 512 * 7)


def test_multilinear_map_gradients_reach_both_branches() -> None:
    """Assignment forbids detaching f or p, so both must receive gradient."""
    f = torch.randn(4, 8, requires_grad=True)
    logits = torch.randn(4, 3, requires_grad=True)
    p = torch.softmax(logits, dim=1)
    multilinear_map(f, p).sum().backward()
    assert f.grad is not None and f.grad.abs().sum() > 0
    assert logits.grad is not None and logits.grad.abs().sum() > 0


def test_multilinear_map_rejects_batch_mismatch() -> None:
    with pytest.raises(ValueError, match="Batch mismatch"):
        multilinear_map(torch.randn(4, 8), torch.randn(5, 3))
