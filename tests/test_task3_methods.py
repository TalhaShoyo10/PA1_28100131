"""Tests for Task 3: DAN-DG, SAM, the sharpness proxy, and domain separability."""

from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn as nn

from task2.evaluation.domain_separability import (
    balance_feature_sets,
    domain_separability,
    source_domain_separability,
    source_target_separability,
)
from task3.methods.dan_dg import DANDG
from task3.methods.sam import SAM, SAMOptimizer, sharpness_proxy


@pytest.fixture
def source_batch():
    torch.manual_seed(6304)
    return {
        "domain_features": {
            "photo": torch.randn(8, 512, requires_grad=True),
            "art_painting": (torch.randn(8, 512) + 1.0).requires_grad_(True),
            "cartoon": (torch.randn(8, 512) + 2.0).requires_grad_(True),
        },
        "logits": torch.randn(24, 7, requires_grad=True),
        "labels": torch.randint(0, 7, (24,)),
    }


# --------------------------------------------------------------------------
# DAN-DG
# --------------------------------------------------------------------------


def test_dan_dg_never_accepts_target_features(source_batch) -> None:
    """Task 3's hard boundary, enforced in the loss itself."""
    features = torch.cat(list(source_batch["domain_features"].values()))
    with pytest.raises(ValueError, match="forbids any Sketch access"):
        DANDG().compute_loss(
            features,
            source_batch["logits"],
            source_batch["labels"],
            target_features=torch.randn(24, 512),
            domain_features=source_batch["domain_features"],
        )


def test_dan_dg_declares_it_does_not_use_target() -> None:
    assert DANDG().uses_target is False


def test_dan_dg_requires_per_domain_features(source_batch) -> None:
    features = torch.cat(list(source_batch["domain_features"].values()))
    with pytest.raises(ValueError, match="per-domain features"):
        DANDG().compute_loss(features, source_batch["logits"], source_batch["labels"])


def test_dan_dg_logs_every_source_pair(source_batch) -> None:
    """Per-pair values expose whether one pair dominates the average."""
    features = torch.cat(list(source_batch["domain_features"].values()))
    out = DANDG().compute_loss(
        features, source_batch["logits"], source_batch["labels"],
        domain_features=source_batch["domain_features"],
    )
    pair_keys = [k for k in out.components if k.startswith("mmd_") and "__" in k]
    assert len(pair_keys) == 3
    assert torch.isfinite(out.loss)


def test_dan_dg_lambda_scales_the_penalty(source_batch) -> None:
    features = torch.cat(list(source_batch["domain_features"].values()))
    losses = {}
    for lam in (0.1, 1.0, 10.0):
        out = DANDG(lambda_dg=lam).compute_loss(
            features, source_batch["logits"], source_batch["labels"],
            domain_features=source_batch["domain_features"],
        )
        losses[lam] = out.loss.item()
        assert out.components["mmd_loss"] == pytest.approx(
            DANDG().compute_loss(
                features, source_batch["logits"], source_batch["labels"],
                domain_features=source_batch["domain_features"],
            ).components["mmd_loss"],
            rel=1e-5,
        )
    assert losses[0.1] < losses[1.0] < losses[10.0]


def test_dan_dg_gradients_reach_all_domains(source_batch) -> None:
    features = torch.cat(list(source_batch["domain_features"].values()))
    DANDG().compute_loss(
        features, source_batch["logits"], source_batch["labels"],
        domain_features=source_batch["domain_features"],
    ).loss.backward()
    for name, tensor in source_batch["domain_features"].items():
        assert tensor.grad is not None and tensor.grad.abs().sum() > 0, name


def test_dan_dg_rejects_negative_lambda() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        DANDG(lambda_dg=-1.0)


# --------------------------------------------------------------------------
# SAM
# --------------------------------------------------------------------------


def _small_model() -> nn.Module:
    torch.manual_seed(6304)
    return nn.Sequential(nn.Linear(8, 16), nn.ReLU(), nn.Linear(16, 4))


def test_sam_loss_is_plain_erm() -> None:
    """SAM's objective lives in the optimizer, not the loss."""
    logits = torch.randn(16, 4, requires_grad=True)
    labels = torch.randint(0, 4, (16,))
    out = SAM().compute_loss(torch.randn(16, 8), logits, labels)
    expected = nn.functional.cross_entropy(logits, labels)
    assert out.loss.item() == pytest.approx(expected.item(), rel=1e-6)


def test_sam_refuses_target_features() -> None:
    with pytest.raises(ValueError, match="forbids Sketch access"):
        SAM().compute_loss(
            torch.randn(8, 8), torch.randn(8, 4), torch.randint(0, 4, (8,)),
            target_features=torch.randn(8, 8),
        )


def test_sam_declares_two_passes() -> None:
    assert SAM().requires_two_passes is True
    assert SAM().uses_target is False


def test_perturbation_has_radius_exactly_rho() -> None:
    """The defining property of the ascent step."""
    model = _small_model()
    opt = SAMOptimizer(model.parameters(), torch.optim.AdamW, rho=0.05, lr=1e-3)
    x, y = torch.randn(32, 8), torch.randint(0, 4, (32,))

    opt.zero_grad()
    nn.functional.cross_entropy(model(x), y).backward()
    originals = [opt.state[p]["old_p"] if "old_p" in opt.state[p] else p.data.clone()
                 for p in model.parameters()]
    opt.first_step()

    delta = torch.cat([
        (p.data - opt.state[p]["old_p"]).flatten() for p in model.parameters()
    ])
    assert delta.norm().item() == pytest.approx(0.05, abs=1e-6)


def test_ascent_step_increases_the_loss() -> None:
    """theta + eps must be a WORSE point, or SAM is minimising the wrong thing."""
    model = _small_model()
    opt = SAMOptimizer(model.parameters(), torch.optim.AdamW, rho=0.05, lr=1e-3)
    x, y = torch.randn(32, 8), torch.randint(0, 4, (32,))

    opt.zero_grad()
    clean = nn.functional.cross_entropy(model(x), y)
    clean.backward()
    opt.first_step(zero_grad=True)

    with torch.no_grad():
        perturbed = nn.functional.cross_entropy(model(x), y)
    assert perturbed.item() > clean.item()


def test_second_step_restores_then_updates() -> None:
    model = _small_model()
    opt = SAMOptimizer(model.parameters(), torch.optim.AdamW, rho=0.05, lr=1e-3)
    x, y = torch.randn(32, 8), torch.randint(0, 4, (32,))
    before = model[0].weight.data.clone()

    opt.zero_grad()
    nn.functional.cross_entropy(model(x), y).backward()
    opt.first_step(zero_grad=True)
    perturbed = model[0].weight.data.clone()

    nn.functional.cross_entropy(model(x), y).backward()
    opt.second_step(zero_grad=True)
    after = model[0].weight.data.clone()

    assert not torch.allclose(after, perturbed)   # not left at theta + eps
    assert not torch.allclose(after, before)      # a real update happened
    assert (after - before).abs().max() < 0.01    # of optimizer-step magnitude


def test_single_pass_misuse_is_rejected() -> None:
    model = _small_model()
    opt = SAMOptimizer(model.parameters(), torch.optim.AdamW, rho=0.05, lr=1e-3)
    with pytest.raises(RuntimeError, match="two explicit passes"):
        opt.step()


def test_sam_rejects_negative_rho() -> None:
    model = _small_model()
    with pytest.raises(ValueError, match="rho"):
        SAMOptimizer(model.parameters(), torch.optim.AdamW, rho=-0.05, lr=1e-3)


# --------------------------------------------------------------------------
# Sharpness proxy
# --------------------------------------------------------------------------


def test_sharpness_proxy_reports_a_non_negative_increase() -> None:
    result = sharpness_proxy(_small_model(), torch.randn(32, 8), torch.randint(0, 4, (32,)))
    assert result["delta_sharp"] >= 0
    assert result["loss_perturbed"] >= result["loss_clean"]
    assert result["rho"] == 0.05


def test_sharpness_proxy_leaves_the_model_untouched() -> None:
    """A diagnostic must not silently modify the model it measures."""
    model = _small_model()
    before = [p.data.clone() for p in model.parameters()]
    sharpness_proxy(model, torch.randn(32, 8), torch.randint(0, 4, (32,)))
    assert all(torch.equal(p.data, b) for p, b in zip(model.parameters(), before))


def test_sharpness_proxy_clears_gradients() -> None:
    model = _small_model()
    sharpness_proxy(model, torch.randn(32, 8), torch.randint(0, 4, (32,)))
    assert all(p.grad is None for p in model.parameters())


def test_sharpness_proxy_restores_training_mode() -> None:
    model = _small_model()
    model.train()
    sharpness_proxy(model, torch.randn(32, 8), torch.randint(0, 4, (32,)))
    assert model.training is True


def test_larger_rho_gives_larger_sharpness() -> None:
    model = _small_model()
    x, y = torch.randn(32, 8), torch.randint(0, 4, (32,))
    small = sharpness_proxy(model, x, y, rho=0.01)["delta_sharp"]
    large = sharpness_proxy(model, x, y, rho=0.1)["delta_sharp"]
    assert large > small


def test_sharpness_proxy_is_deterministic() -> None:
    """The same model and batch must give the same number every time."""
    model = _small_model()
    x, y = torch.randn(32, 8), torch.randint(0, 4, (32,))
    a = sharpness_proxy(model, x, y)
    b = sharpness_proxy(model, x, y)
    assert a["delta_sharp"] == pytest.approx(b["delta_sharp"], rel=1e-9)


# --------------------------------------------------------------------------
# Domain separability
# --------------------------------------------------------------------------


def test_separable_features_score_near_100() -> None:
    rng = np.random.default_rng(6304)
    a = rng.normal(0, 1, (200, 16))
    b = rng.normal(10, 1, (200, 16))
    assert source_target_separability(a, b)["separability"] > 95


def test_identical_distributions_score_near_chance() -> None:
    """Chance is the honest reading when domains are indistinguishable."""
    rng = np.random.default_rng(6304)
    a = rng.normal(0, 1, (300, 16))
    b = rng.normal(0, 1, (300, 16))
    result = source_target_separability(a, b)
    assert result["chance"] == 50.0
    assert 35 < result["separability"] < 65


def test_three_way_source_separability_reports_33_percent_chance() -> None:
    rng = np.random.default_rng(6304)
    features = {
        "photo": rng.normal(0, 1, (150, 16)),
        "art_painting": rng.normal(5, 1, (150, 16)),
        "cartoon": rng.normal(10, 1, (150, 16)),
    }
    result = source_domain_separability(features)
    assert result["chance"] == pytest.approx(33.33, abs=0.01)
    assert result["n_domains"] == 3
    assert result["domains"] == ["art_painting", "cartoon", "photo"]
    assert result["separability"] > 90


def test_feature_sets_are_balanced_before_fitting() -> None:
    """Otherwise the score would reflect set sizes, not separability."""
    rng = np.random.default_rng(6304)
    balanced = balance_feature_sets([rng.normal(size=(500, 8)), rng.normal(size=(50, 8))])
    assert all(len(f) == 50 for f in balanced)


def test_separability_is_deterministic() -> None:
    rng = np.random.default_rng(6304)
    a, b = rng.normal(0, 1, (200, 16)), rng.normal(3, 1, (200, 16))
    first = source_target_separability(a, b, seed=6304)["separability"]
    second = source_target_separability(a, b, seed=6304)["separability"]
    assert first == pytest.approx(second)


def test_separability_requires_two_domains() -> None:
    with pytest.raises(ValueError, match="at least 2 domains"):
        domain_separability([np.random.randn(10, 4)])
