"""Tests for Task 4: CIFAR ResNet, post-hoc scores and PROSER."""

from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn as nn

from common.metrics import auroc
from task4.methods.proser import PROSER, manifold_mixup_pairs, sample_lam
from task4.models.resnet_cifar import build_resnet_cifar
from task4.scores.posthoc import (
    compute_score,
    energy_score,
    fit_mahalanobis,
    mahalanobis_score,
    mls_score,
    msp_score,
    proser_score,
)


# --------------------------------------------------------------------------
# CIFAR-appropriate ResNet-18
# --------------------------------------------------------------------------


def test_stem_is_modified_for_32px() -> None:
    """ImageNet's 7x7 stride-2 stem plus max-pool would shrink 32x32 to 8x8."""
    model = build_resnet_cifar(10)
    assert model.conv1.kernel_size == (3, 3)
    assert model.conv1.stride == (1, 1)
    assert not any(isinstance(m, nn.MaxPool2d) for m in model.modules())


def test_forward_shapes() -> None:
    model = build_resnet_cifar(10)
    x = torch.rand(4, 3, 32, 32)
    logits, features = model(x, return_features=True)
    assert logits.shape == (4, 10)
    assert features.shape == (4, 512)


def test_mixup_point_is_after_layer2() -> None:
    """PROSER requires mixing after layer2 and before layer3."""
    model = build_resnet_cifar(10)
    hidden = model.forward_pre_layer3(torch.rand(2, 3, 32, 32))
    assert hidden.shape == (2, 128, 16, 16)
    assert model.forward_post_layer2(hidden).shape == (2, 512)


def test_split_forward_matches_full_forward() -> None:
    """The two halves must compose into the ordinary forward pass."""
    model = build_resnet_cifar(10).eval()
    x = torch.rand(2, 3, 32, 32)
    with torch.no_grad():
        direct = model.extract_features(x)
        split = model.forward_post_layer2(model.forward_pre_layer3(x))
    assert torch.allclose(direct, split, atol=1e-6)


def test_adding_dummies_preserves_trained_known_rows() -> None:
    """PROSER initializes FROM the vanilla checkpoint; those rows must survive."""
    model = build_resnet_cifar(10)
    before_w = model.fc.weight.data.clone()
    before_b = model.fc.bias.data.clone()

    model.add_dummy_classifiers(5)

    assert model.fc.out_features == 15
    assert torch.equal(model.fc.weight.data[:10], before_w)
    assert torch.equal(model.fc.bias.data[:10], before_b)
    assert model.dummy_classifiers == 5


def test_known_and_dummy_logit_slicing() -> None:
    """CSA and MLS must use only the ten known logits."""
    model = build_resnet_cifar(10)
    model.add_dummy_classifiers(5)
    logits = model(torch.rand(3, 3, 32, 32))

    assert model.known_logits(logits).shape == (3, 10)
    assert model.dummy_logits(logits).shape == (3, 5)
    assert torch.equal(model.known_logits(logits), logits[:, :10])


def test_dummy_slice_without_dummies_raises() -> None:
    with pytest.raises(RuntimeError, match="no dummy classifiers"):
        build_resnet_cifar(10).dummy_logits(torch.rand(2, 10))


# --------------------------------------------------------------------------
# Post-hoc scores: convention and behaviour
# --------------------------------------------------------------------------


@pytest.fixture
def logits():
    confident = np.array([[10.0, 0.0, 0.0, 0.0]])
    uncertain = np.array([[1.0, 0.9, 0.8, 0.7]])
    return confident, uncertain


@pytest.mark.parametrize("score_fn", [msp_score, mls_score, energy_score])
def test_confident_inputs_score_lower_than_uncertain(score_fn, logits) -> None:
    """Shared convention: larger u(x) means MORE novel."""
    confident, uncertain = logits
    assert score_fn(confident)[0] < score_fn(uncertain)[0]


def test_msp_is_one_minus_max_probability() -> None:
    z = np.array([[2.0, 1.0, 0.0]])
    p = np.exp(z[0]) / np.exp(z[0]).sum()
    assert msp_score(z)[0] == pytest.approx(1.0 - p.max())


def test_msp_ignores_absolute_magnitude() -> None:
    """Softmax normalises scale away -- this is MSP's defining limitation."""
    small = np.array([[1.0, 0.0, 0.0]])
    scaled = small * 1.0
    assert msp_score(small)[0] == pytest.approx(msp_score(scaled)[0])


def test_mls_retains_absolute_magnitude() -> None:
    """MLS separates cases MSP cannot, because it keeps logit scale."""
    weak = np.array([[1.0, 0.5, 0.25]])
    strong = np.array([[10.0, 5.0, 2.5]])
    assert mls_score(weak)[0] > mls_score(strong)[0]


def test_energy_uses_all_logits_not_just_the_max() -> None:
    """Two inputs with identical max but different spread must differ."""
    peaked = np.array([[5.0, -10.0, -10.0, -10.0]])
    spread = np.array([[5.0, 4.9, 4.8, 4.7]])
    assert energy_score(peaked)[0] > energy_score(spread)[0]


def test_energy_matches_negative_logsumexp() -> None:
    z = np.array([[1.0, 2.0, 3.0]])
    assert energy_score(z)[0] == pytest.approx(-np.log(np.exp(z[0]).sum()))


def test_scores_are_finite_on_extreme_logits() -> None:
    extreme = np.array([[1e4, -1e4, 0.0], [-1e4, -1e4, -1e4]])
    for fn in (msp_score, mls_score, energy_score):
        assert np.isfinite(fn(extreme)).all(), fn.__name__


# --------------------------------------------------------------------------
# Mahalanobis
# --------------------------------------------------------------------------


def test_mahalanobis_is_small_near_a_class_mean() -> None:
    rng = np.random.default_rng(6304)
    features = np.concatenate([rng.normal(0, 1, (100, 8)), rng.normal(10, 1, (100, 8))])
    labels = np.array([0] * 100 + [1] * 100)

    means, variance = fit_mahalanobis(features, labels, 2)
    near = mahalanobis_score(means[0][None, :], means, variance)[0]
    far = mahalanobis_score(np.full((1, 8), 100.0), means, variance)[0]
    assert near < far


def test_mahalanobis_takes_the_minimum_over_classes() -> None:
    """Distance to the NEAREST cluster, not the average."""
    means = np.array([[0.0, 0.0], [10.0, 10.0]])
    variance = np.ones(2)
    at_second_mean = mahalanobis_score(np.array([[10.0, 10.0]]), means, variance)[0]
    assert at_second_mean == pytest.approx(0.0)


def test_shared_diagonal_covariance_has_epsilon() -> None:
    """Zero-variance dimensions would divide by zero without it."""
    features = np.ones((20, 4))
    labels = np.array([0] * 10 + [1] * 10)
    _, variance = fit_mahalanobis(features, labels, 2, epsilon=1e-6)
    assert (variance >= 1e-6).all()
    assert np.isfinite(mahalanobis_score(features, *fit_mahalanobis(features, labels, 2))).all()


def test_mahalanobis_separates_novel_features() -> None:
    rng = np.random.default_rng(6304)
    known = rng.normal(0, 1, (300, 16))
    labels = rng.integers(0, 10, 300)
    means, variance = fit_mahalanobis(known, labels, 10)

    known_scores = mahalanobis_score(known, means, variance)
    novel_scores = mahalanobis_score(rng.normal(20, 1, (300, 16)), means, variance)
    assert auroc(known_scores, novel_scores) > 95


# --------------------------------------------------------------------------
# PROSER score
# --------------------------------------------------------------------------


def test_proser_score_compares_dummy_against_known() -> None:
    logits = np.array([[5.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.0]])
    assert proser_score(logits, 10, 1)[0] == pytest.approx(2.0 - 5.0)


def test_proser_score_rises_when_a_dummy_dominates() -> None:
    known_wins = np.array([[8.0] + [0.0] * 9 + [1.0]])
    dummy_wins = np.array([[1.0] + [0.0] * 9 + [8.0]])
    assert proser_score(dummy_wins, 10, 1)[0] > proser_score(known_wins, 10, 1)[0]


def test_proser_score_validates_width() -> None:
    with pytest.raises(ValueError, match="Expected 15 logits"):
        proser_score(np.zeros((2, 12)), 10, 5)


# --------------------------------------------------------------------------
# Score registry
# --------------------------------------------------------------------------


def test_registry_dispatch_matches_direct_calls() -> None:
    z = np.array([[2.0, 1.0, 0.5]])
    assert compute_score("msp", z)[0] == pytest.approx(msp_score(z)[0])
    assert compute_score("mls", z)[0] == pytest.approx(mls_score(z)[0])
    assert compute_score("energy", z)[0] == pytest.approx(energy_score(z)[0])


def test_registry_rejects_unknown_score() -> None:
    with pytest.raises(KeyError, match="Unknown score"):
        compute_score("entropy", np.zeros((1, 3)))


# --------------------------------------------------------------------------
# PROSER training objectives
# --------------------------------------------------------------------------


def test_mixup_pairs_always_cross_class() -> None:
    """Same-class mixing would produce a valid known example, not a proxy unknown."""
    generator = torch.Generator().manual_seed(6304)
    labels = torch.tensor([0, 0, 1, 1, 2, 2])
    partners = manifold_mixup_pairs(labels, generator)
    assert (partners >= 0).all()
    for i, p in enumerate(partners):
        assert labels[i] != labels[p]


def test_mixup_pairs_signal_impossible_batches() -> None:
    generator = torch.Generator().manual_seed(6304)
    partners = manifold_mixup_pairs(torch.tensor([3, 3, 3]), generator)
    assert (partners == -1).all()


def test_lam_concentrates_near_one_half() -> None:
    """Beta(2,2) puts mixtures BETWEEN classes, where a proxy unknown belongs."""
    lam = sample_lam(20_000, 2.0, 2.0, np.random.default_rng(6304))
    assert lam.mean() == pytest.approx(0.5, abs=0.02)
    assert ((lam > 0.25) & (lam < 0.75)).mean() > 0.6
    assert (lam >= 0).all() and (lam <= 1).all()


@pytest.fixture
def proser_model():
    torch.manual_seed(6304)
    model = build_resnet_cifar(10)
    model.add_dummy_classifiers(5)
    return model


def test_proser_requires_dummy_classifiers() -> None:
    with pytest.raises(ValueError, match="requires dummy classifiers"):
        PROSER(build_resnet_cifar(10))


def test_proser_loss_has_all_three_terms(proser_model) -> None:
    proser = PROSER(proser_model, beta=1.0, gamma=0.1)
    loss, components = proser.compute_loss(
        torch.rand(8, 3, 32, 32), torch.randint(0, 10, (8,))
    )
    assert torch.isfinite(loss)
    for key in ("closed_loss", "placeholder_loss", "mixup_loss"):
        assert key in components


def test_proser_gradients_reach_the_network(proser_model) -> None:
    proser = PROSER(proser_model, beta=1.0, gamma=0.1)
    loss, _ = proser.compute_loss(torch.rand(8, 3, 32, 32), torch.randint(0, 10, (8,)))
    loss.backward()
    assert proser_model.fc.weight.grad.abs().sum() > 0
    assert proser_model.conv1.weight.grad.abs().sum() > 0


def test_classifier_placeholder_masks_the_true_class(proser_model) -> None:
    """Once the correct class is excluded, a dummy should win.

    The masked logit must be -inf so the true class cannot be the runner-up.
    """
    proser = PROSER(proser_model)
    logits = torch.randn(4, 15)
    labels = torch.randint(0, 10, (4,))
    _, placeholder_loss = proser.classifier_placeholder_loss(logits, labels)
    assert torch.isfinite(placeholder_loss)

    masked = logits.clone()
    masked.scatter_(1, labels.unsqueeze(1), -float("inf"))
    assert torch.isinf(masked[torch.arange(4), labels]).all()


def test_beta_and_gamma_weight_their_terms(proser_model) -> None:
    images, labels = torch.rand(8, 3, 32, 32), torch.randint(0, 10, (8,))

    _, base = PROSER(proser_model, beta=1.0, gamma=0.1).compute_loss(images, labels)
    _, heavier = PROSER(proser_model, beta=2.0, gamma=0.1).compute_loss(images, labels)

    assert heavier["total_loss"] > base["total_loss"]


def test_batch_is_split_into_two_equal_halves(proser_model) -> None:
    """Assignment: first half classifier placeholders, second half data placeholders."""
    proser = PROSER(proser_model)
    loss, components = proser.compute_loss(
        torch.rand(16, 3, 32, 32), torch.randint(0, 10, (16,))
    )
    assert components["mixup_loss"] > 0
    assert torch.isfinite(loss)


# --------------------------------------------------------------------------
# PROSER checkpoint resolution
# --------------------------------------------------------------------------


def test_proser_init_honours_the_checkpoint_dir_override(tmp_path) -> None:
    """Regression guard (2026-09-23).

    method.init_from is repository-relative, so a run pointed at Drive via
    --set output.checkpoint_dir failed to find the Vanilla checkpoint that
    PROSER must initialize from.
    """
    from common.config import apply_overrides, load_config
    from task4.train import resolve_vanilla_checkpoint

    vanilla = tmp_path / "task4" / "vanilla"
    vanilla.mkdir(parents=True)
    (vanilla / "best.pt").write_bytes(b"x")

    cfg = load_config("task4/configs/proser.yaml")
    apply_overrides(cfg, [f"output.checkpoint_dir={tmp_path / 'task4'}"])

    assert resolve_vanilla_checkpoint(cfg).is_relative_to(tmp_path)


def test_proser_init_does_not_fall_back_when_overridden() -> None:
    """A different Vanilla checkpoint must never be loaded silently."""
    from common.config import apply_overrides, load_config
    from task4.train import resolve_vanilla_checkpoint

    cfg = load_config("task4/configs/proser.yaml")
    apply_overrides(cfg, ["output.checkpoint_dir=/definitely/absent/task4"])
    with pytest.raises(FileNotFoundError, match="vanilla.yaml"):
        resolve_vanilla_checkpoint(cfg)


def test_mixup_pairing_works_for_labels_on_any_device() -> None:
    """Regression guard (2026-09-23).

    torch.randint requires the generator's device to match the tensor being
    sampled. A CPU generator with CUDA labels raised 'Expected a cuda device
    type for generator but found cpu', which surfaced only on GPU because the
    tests run on CPU. The index is now drawn on the CPU regardless.
    """
    generator = torch.Generator().manual_seed(6304)
    labels = torch.tensor([0, 0, 1, 1, 2, 2])

    partners = manifold_mixup_pairs(labels, generator)

    assert partners.device == labels.device
    assert (partners >= 0).all()
    for i, p in enumerate(partners):
        assert labels[i] != labels[p]

    if torch.cuda.is_available():
        cuda_labels = labels.cuda()
        cuda_partners = manifold_mixup_pairs(
            cuda_labels, torch.Generator().manual_seed(6304)
        )
        assert cuda_partners.device == cuda_labels.device
        assert torch.equal(cuda_partners.cpu(), partners)


def test_mixup_pairing_is_reproducible_from_a_seed() -> None:
    labels = torch.tensor([0, 1, 2, 3, 0, 1, 2, 3])
    first = manifold_mixup_pairs(labels, torch.Generator().manual_seed(6304))
    second = manifold_mixup_pairs(labels, torch.Generator().manual_seed(6304))
    assert torch.equal(first, second)
