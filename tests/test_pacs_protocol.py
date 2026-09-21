"""Tests for the shared PACS protocol.

These run without the real dataset: splits are tested on synthetic label
sequences, and the BatchNorm / batching machinery on tiny tensors. What they
protect is the machinery both Task 2 and Task 3 depend on being identical.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from shared.pacs import PACS_CLASSES, PACSDataset, TargetAccessViolation
from shared.pacs_protocol import (
    DomainBalancedIterator,
    InfiniteLoader,
    assert_bn_frozen,
    eval_transform,
    set_bn_eval,
    stratified_split,
    train_mode_with_frozen_bn,
    train_transform,
)


# --------------------------------------------------------------------------
# Stratified splitting
# --------------------------------------------------------------------------


def test_split_is_stratified_and_partitions_all_indices() -> None:
    labels = [c for c in range(7) for _ in range(100)]
    train, val = stratified_split(labels, val_fraction=0.2, seed=6304)

    assert len(train) + len(val) == len(labels)
    assert not set(train) & set(val)              # disjoint
    assert sorted(train + val) == list(range(len(labels)))  # complete

    labels_arr = np.asarray(labels)
    for cls in range(7):
        n_val_c = int((labels_arr[val] == cls).sum())
        assert n_val_c == 20, f"class {cls} got {n_val_c} val examples, expected 20"


def test_split_is_deterministic_for_a_given_seed() -> None:
    labels = [c for c in range(7) for _ in range(37)]
    a = stratified_split(labels, 0.2, seed=6304)
    b = stratified_split(labels, 0.2, seed=6304)
    assert a == b


def test_different_seeds_give_different_splits() -> None:
    labels = [c for c in range(7) for _ in range(50)]
    a = stratified_split(labels, 0.2, seed=6304)
    b = stratified_split(labels, 0.2, seed=1)
    assert a != b


def test_split_handles_uneven_class_counts() -> None:
    """PACS domains are class-imbalanced; every class must still be represented."""
    labels = [0] * 3 + [1] * 97 + [2] * 50
    train, val = stratified_split(labels, 0.2, seed=6304)
    labels_arr = np.asarray(labels)
    for cls in (0, 1, 2):
        assert (labels_arr[val] == cls).sum() >= 1, f"class {cls} absent from val"
        assert (labels_arr[train] == cls).sum() >= 1, f"class {cls} absent from train"


def test_tiny_class_never_empties_the_train_side() -> None:
    labels = [0] * 2 + [1] * 100
    train, val = stratified_split(labels, 0.9, seed=6304)
    labels_arr = np.asarray(labels)
    assert (labels_arr[train] == 0).sum() >= 1
    assert (labels_arr[val] == 0).sum() >= 1


def test_single_example_class_goes_to_train() -> None:
    labels = [0] + [1] * 20
    train, val = stratified_split(labels, 0.2, seed=6304)
    labels_arr = np.asarray(labels)
    assert (labels_arr[train] == 0).sum() == 1
    assert (labels_arr[val] == 0).sum() == 0


@pytest.mark.parametrize("bad", [0.0, 1.0, -0.1, 1.5])
def test_invalid_val_fraction_rejected(bad: float) -> None:
    with pytest.raises(ValueError, match="val_fraction"):
        stratified_split([0, 1, 0, 1], bad, seed=6304)


# --------------------------------------------------------------------------
# Information boundary
# --------------------------------------------------------------------------


def test_sketch_dataset_refused_when_target_access_forbidden(tmp_path) -> None:
    """Task 3's hard guard: loading Sketch must fail loudly, not silently work."""
    with pytest.raises(TargetAccessViolation, match="evaluate_sketch"):
        PACSDataset(
            tmp_path,
            samples=[("sketch/dog/a.jpg", 0)],
            domain="sketch",
            target_access_forbidden=True,
        )


def test_source_domains_unaffected_by_the_guard(tmp_path) -> None:
    ds = PACSDataset(
        tmp_path,
        samples=[("photo/dog/a.jpg", 0)],
        domain="photo",
        target_access_forbidden=True,
    )
    assert len(ds) == 1


def test_empty_sample_list_rejected(tmp_path) -> None:
    with pytest.raises(ValueError, match="Empty sample list"):
        PACSDataset(tmp_path, samples=[], domain="photo")


def test_class_counts_use_canonical_order(tmp_path) -> None:
    ds = PACSDataset(
        tmp_path,
        samples=[("photo/dog/a.jpg", 0), ("photo/horse/b.jpg", 4)],
        domain="photo",
    )
    counts = ds.class_counts()
    assert list(counts) == list(PACS_CLASSES)
    assert counts["dog"] == 1 and counts["horse"] == 1
    assert counts["giraffe"] == 0


# --------------------------------------------------------------------------
# Frozen BatchNorm policy
# --------------------------------------------------------------------------


def _bn_model() -> nn.Module:
    return nn.Sequential(
        nn.Conv2d(3, 4, 3, padding=1),
        nn.BatchNorm2d(4),
        nn.ReLU(),
        nn.Conv2d(4, 4, 3, padding=1),
        nn.BatchNorm2d(4),
    )


def test_set_bn_eval_freezes_only_batchnorm() -> None:
    model = _bn_model()
    model.train()
    n = set_bn_eval(model)

    assert n == 2
    for module in model.modules():
        if isinstance(module, nn.modules.batchnorm._BatchNorm):
            assert module.training is False
        elif isinstance(module, (nn.Conv2d, nn.ReLU)):
            assert module.training is True


def test_running_stats_do_not_move_under_the_policy() -> None:
    """The actual guarantee: BN running mean/var stay at pretrained values."""
    model = _bn_model()
    bn = model[1]
    bn.running_mean.fill_(0.5)
    bn.running_var.fill_(2.0)

    train_mode_with_frozen_bn(model)
    # A forward pass on data with very different statistics.
    model(torch.randn(8, 3, 16, 16) * 10 + 5)

    assert torch.allclose(bn.running_mean, torch.full_like(bn.running_mean, 0.5))
    assert torch.allclose(bn.running_var, torch.full_like(bn.running_var, 2.0))


def test_plain_train_does_move_running_stats() -> None:
    """Control: confirms the previous test is actually testing the policy."""
    model = _bn_model()
    bn = model[1]
    bn.running_mean.fill_(0.5)

    model.train()  # deliberately WITHOUT the policy
    model(torch.randn(8, 3, 16, 16) * 10 + 5)

    assert not torch.allclose(bn.running_mean, torch.full_like(bn.running_mean, 0.5))


def test_affine_parameters_remain_trainable() -> None:
    """gamma and beta must still receive gradients under the policy."""
    model = _bn_model()
    train_mode_with_frozen_bn(model)
    model(torch.randn(4, 3, 8, 8)).sum().backward()

    bn = model[1]
    assert bn.weight.requires_grad and bn.weight.grad is not None
    assert bn.bias.requires_grad and bn.bias.grad is not None
    assert bn.weight.grad.abs().sum() > 0


def test_assert_bn_frozen_detects_violation() -> None:
    model = _bn_model()
    model.train()
    with pytest.raises(RuntimeError, match="Frozen-BatchNorm policy violated"):
        assert_bn_frozen(model)

    train_mode_with_frozen_bn(model)
    assert_bn_frozen(model)  # no raise


# --------------------------------------------------------------------------
# Domain-balanced batching
# --------------------------------------------------------------------------


def _loader(n: int, batch_size: int) -> DataLoader:
    ds = TensorDataset(torch.arange(n).float().view(n, 1), torch.zeros(n, dtype=torch.long))
    return DataLoader(ds, batch_size=batch_size, drop_last=True)


def test_every_batch_contains_all_domains_equally() -> None:
    """8 examples from each of 3 source domains on every update."""
    loaders = {"photo": _loader(80, 8), "art_painting": _loader(40, 8), "cartoon": _loader(24, 8)}
    it = DomainBalancedIterator(loaders)

    n_batches = 0
    for batch in it:
        assert set(batch) == {"photo", "art_painting", "cartoon"}
        for _, (x, _y) in batch.items():
            assert x.shape[0] == 8
        n_batches += 1

    # Epoch length follows the LONGEST loader, so no domain's data is skipped.
    assert n_batches == len(it) == 10


def test_shorter_domains_cycle_rather_than_truncate() -> None:
    loaders = {"big": _loader(80, 8), "small": _loader(16, 8)}
    it = DomainBalancedIterator(loaders)
    assert len(it) == 10
    assert sum(1 for _ in it) == 10   # small loader (2 batches) is cycled


def test_domain_balanced_iterator_requires_loaders() -> None:
    with pytest.raises(ValueError, match="at least one loader"):
        DomainBalancedIterator({})


def test_infinite_loader_cycles_indefinitely() -> None:
    inf = InfiniteLoader(_loader(16, 8))  # only 2 batches
    for _ in range(7):                    # ask for far more
        x, _y = inf.next()
        assert x.shape[0] == 8


# --------------------------------------------------------------------------
# Transforms
# --------------------------------------------------------------------------


def test_transforms_produce_224_normalized_tensors() -> None:
    from PIL import Image

    img = Image.new("RGB", (300, 200), (120, 60, 30))
    for tfm in (train_transform(), eval_transform()):
        out = tfm(img)
        assert out.shape == (3, 224, 224)
        assert out.dtype == torch.float32


def test_eval_transform_is_deterministic() -> None:
    """Validation and test inputs must not vary between runs."""
    from PIL import Image

    img = Image.new("RGB", (300, 200), (120, 60, 30))
    tfm = eval_transform()
    assert torch.allclose(tfm(img), tfm(img))
