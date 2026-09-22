"""Tests for the config loader.

The shadowing test is a regression guard: an earlier implementation used only
``__getattr__``, so a config key named ``values`` returned the bound
``dict.values`` method instead of the configured list. A sweep script reading
``cfg.study.values`` would then iterate nothing and silently skip the study
rather than fail loudly.
"""

from __future__ import annotations

import pytest
import yaml

from common.config import Config, apply_overrides, load_config


# --------------------------------------------------------------------------
# Attribute access
# --------------------------------------------------------------------------


@pytest.mark.parametrize("key", ["values", "keys", "items", "get", "update", "copy", "pop"])
def test_config_keys_shadow_dict_methods(key: str) -> None:
    """A config key must win over the same-named dict method."""
    cfg = Config({key: [0.1, 1.0, 10.0]})
    assert getattr(cfg, key) == [0.1, 1.0, 10.0]


def test_dict_methods_available_when_not_shadowed() -> None:
    cfg = Config({"lr": 1e-4, "epochs": 30})
    assert sorted(cfg.keys()) == ["epochs", "lr"]
    assert cfg.get("lr") == 1e-4


def test_nested_dicts_wrapped() -> None:
    cfg = Config({"method": {"mmd": {"bandwidths": [0.5, 1.0, 2.0]}}})
    assert cfg.method.mmd.bandwidths == [0.5, 1.0, 2.0]
    assert isinstance(cfg.method, Config)


def test_missing_key_raises_attribute_error() -> None:
    cfg = Config({"lr": 1e-4})
    with pytest.raises(AttributeError, match="No config key or attribute"):
        _ = cfg.nonexistent


def test_setattr_writes_through() -> None:
    cfg = Config({})
    cfg.lr = 1e-3
    assert cfg["lr"] == 1e-3


# --------------------------------------------------------------------------
# Loading, inheritance, overrides
# --------------------------------------------------------------------------


def _write(tmp_path, name: str, payload: dict) -> str:
    path = tmp_path / name
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    return str(path)


def test_inheritance_deep_merges(tmp_path) -> None:
    _write(tmp_path, "base.yaml", {"train": {"lr": 1e-4, "epochs": 30}, "seed": 6304})
    child = _write(tmp_path, "child.yaml", {"defaults": "base.yaml", "train": {"epochs": 1}})

    cfg = load_config(child)
    assert cfg.train.epochs == 1      # overridden
    assert cfg.train.lr == 1e-4       # inherited, not clobbered by partial dict
    assert cfg.seed == 6304


def test_missing_config_raises(tmp_path) -> None:
    with pytest.raises(FileNotFoundError):
        load_config(tmp_path / "absent.yaml")


def test_overrides_parse_types_and_are_recorded() -> None:
    cfg = Config({"method": {"lambda_mmd": 1.0}, "train": {"epochs": 30}})
    apply_overrides(cfg, ["method.lambda_mmd=10", "train.epochs=1"])

    assert cfg.method.lambda_mmd == 10
    assert isinstance(cfg.method.lambda_mmd, int)  # YAML-parsed, not str
    assert cfg.train.epochs == 1
    # Provenance: a CLI tweak must not escape the saved record.
    assert cfg["_overrides"] == ["method.lambda_mmd=10", "train.epochs=1"]


def test_malformed_override_rejected() -> None:
    cfg = Config({"lr": 1e-4})
    with pytest.raises(ValueError, match="key=value"):
        apply_overrides(cfg, ["lr"])


# --------------------------------------------------------------------------
# The real Task 2 configs
# --------------------------------------------------------------------------

_PROTOCOL_LOCKED = {
    "seed": 6304,
    "train.epochs": 30,
    "train.lr": 1e-4,
    "train.weight_decay": 1e-4,
    "train.source_batch_per_domain": 8,
    "train.target_batch": 24,
    "train.early_stopping_patience": 5,
    "model.freeze_bn_stats": True,
    "data.split_seed": 6304,
    "data.val_fraction": 0.2,
}


@pytest.mark.parametrize(
    "name", ["source_only", "dan", "dann", "cdan", "dan_lambda_study"]
)
def test_task2_configs_share_locked_protocol(name: str) -> None:
    """Assignment requires an identical protocol across all Task 2 methods.

    Only the ``method:`` block may differ. This test fails loudly if a method
    config ever drifts on seed, optimizer, budget, batch composition or the
    BatchNorm policy.
    """
    cfg = load_config(f"task2/configs/{name}.yaml")
    for dotted, expected in _PROTOCOL_LOCKED.items():
        node = cfg
        for part in dotted.split("."):
            node = getattr(node, part)
        assert node == expected, f"{name}.yaml drifted on {dotted}: {node} != {expected}"


def test_source_only_does_not_touch_target() -> None:
    cfg = load_config("task2/configs/source_only.yaml")
    assert cfg.method.uses_target_during_training is False


def test_cdan_forbidden_options_disabled() -> None:
    """Assignment explicitly forbids entropy conditioning and detaching."""
    cfg = load_config("task2/configs/cdan.yaml")
    assert cfg.method.conditioning.entropy_conditioning is False
    assert cfg.method.conditioning.detach_feature is False
    assert cfg.method.conditioning.detach_predictions is False
    # vec(f (x) p) with f in R^512 and p in R^7
    assert cfg.method.conditioning.input_dim == 512 * 7


def test_lambda_study_values_readable() -> None:
    """End-to-end guard on the shadowing bug via the real sweep config."""
    cfg = load_config("task2/configs/dan_lambda_study.yaml")
    assert cfg.study.values == [0.1, 1.0, 10.0]
    assert cfg.study.analysis_only is True
    assert cfg.method.lambda_mmd == 1.0  # main comparison default inherited


# --------------------------------------------------------------------------
# The real Task 3 configs
# --------------------------------------------------------------------------


@pytest.mark.parametrize("name", ["erm", "dan_dg", "sam", "dan_dg_lambda_study"])
def test_task3_configs_share_locked_protocol(name: str) -> None:
    cfg = load_config(f"task3/configs/{name}.yaml")
    for dotted, expected in _PROTOCOL_LOCKED.items():
        if dotted == "train.target_batch":
            continue  # Task 3 has no target stream; asserted separately below
        node = cfg
        for part in dotted.split("."):
            node = getattr(node, part)
        assert node == expected, f"{name}.yaml drifted on {dotted}: {node} != {expected}"


@pytest.mark.parametrize("name", ["erm", "dan_dg", "sam", "dan_dg_lambda_study"])
def test_task3_never_streams_target_during_training(name: str) -> None:
    """Sketch must not enter Task 3 training under any method."""
    cfg = load_config(f"task3/configs/{name}.yaml")
    assert cfg.data.target_access_forbidden is True
    assert cfg.train.target_batch == 0


@pytest.mark.parametrize(
    "dotted",
    [
        "seed",
        "data.split_seed",
        "data.val_fraction",
        "data.sources",
        "data.target",
        "data.num_classes",
        "data.resize",
        "data.crop",
        "data.split_manifest",
        "model.backbone",
        "model.weights",
        "model.feature_dim",
        "model.freeze_bn_stats",
        "train.epochs",
        "train.lr",
        "train.weight_decay",
        "train.source_batch_per_domain",
        "train.early_stopping_patience",
        "selection.metric",
    ],
)
def test_task2_and_task3_protocols_match(dotted: str) -> None:
    """Task 3 must reuse Task 2's protocol exactly.

    The assignment requires the same splits, initialization, head,
    preprocessing, augmentation, sampling, optimizer, budget, early stopping
    and seed -- because Task 2's Source-only checkpoint IS Task 3's ERM
    baseline. Any drift here would make the shared baseline a fiction and
    invalidate the Task 2 / Task 3 comparison behind RQ4.
    """
    t2, t3 = load_config("task2/configs/base.yaml"), load_config("task3/configs/base.yaml")
    for part in dotted.split("."):
        t2, t3 = getattr(t2, part), getattr(t3, part)
    assert t2 == t3, f"Task2/Task3 protocol drift on {dotted}: {t2} != {t3}"


def test_erm_reuses_task2_checkpoint_and_forbids_retraining() -> None:
    cfg = load_config("task3/configs/erm.yaml")
    assert cfg.method.retrain_forbidden is True
    assert cfg.method.reuse_checkpoint_from == "checkpoints/task2/source_only/best.pt"


def test_dan_dg_covers_all_three_unordered_source_pairs() -> None:
    cfg = load_config("task3/configs/dan_dg.yaml")
    pairs = {frozenset(p) for p in cfg.method.pairs}
    sources = set(cfg.data.sources)
    expected = {frozenset(c) for c in __import__("itertools").combinations(sources, 2)}
    assert pairs == expected
    assert len(pairs) == 3


def test_sam_is_standard_non_adaptive() -> None:
    cfg = load_config("task3/configs/sam.yaml")
    assert cfg.method.rho == 0.05
    assert cfg.method.adaptive is False
    assert cfg.method.freeze_bn_stats_both_passes is True


def test_matched_lambda_studies_use_identical_grid() -> None:
    """The cross-task comparison requires the same sweep values and mechanism."""
    t2 = load_config("task2/configs/dan_lambda_study.yaml")
    t3 = load_config("task3/configs/dan_dg_lambda_study.yaml")
    assert t2.study.values == t3.study.values == [0.1, 1.0, 10.0]
    assert t2.method.mmd.bandwidth_multipliers == t3.method.mmd.bandwidth_multipliers
    assert t2.method.mmd.bandwidth_estimator == t3.method.mmd.bandwidth_estimator
    # Each points at the other, so a future edit to one is visibly incomplete.
    assert t2.study.matched_with == "task3/configs/dan_dg_lambda_study.yaml"
    assert t3.study.matched_with == "task2/configs/dan_lambda_study.yaml"
    assert t2.study.analysis_only is True and t3.study.analysis_only is True


def test_discriminator_gets_a_higher_learning_rate() -> None:
    """Regression guard for the DANN/CDAN divergence (2026-09-22).

    The backbone is pretrained and needs a small learning rate; a randomly
    initialised discriminator does not. Training both at 1e-4 left the
    discriminator near-random, and the gradient-reversal layer amplified its
    noise into the backbone as alpha ramped -- DANN and CDAN both collapsed to
    near-chance macro-F1 with domain losses in the thousands.
    """
    for task in ("task2", "task3"):
        cfg = load_config(f"{task}/configs/base.yaml")
        assert cfg.train.discriminator_lr_multiplier == 10.0, task


def test_task2_and_task3_agree_on_discriminator_multiplier() -> None:
    t2 = load_config("task2/configs/base.yaml")
    t3 = load_config("task3/configs/base.yaml")
    assert (
        t2.train.discriminator_lr_multiplier == t3.train.discriminator_lr_multiplier
    )
