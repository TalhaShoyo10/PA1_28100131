"""Protocol-compliance tests for the Task 1 and Task 4 configurations.

These encode requirements the assignment states in prose, so that a later edit
which quietly violates one fails a test instead of silently producing an
invalid result. The "Before You Submit" checklist is the model here: no target
labels in Task 2 selection, no Sketch in Task 3 training/selection, no real
unknowns in Task 4 training or threshold selection.
"""

from __future__ import annotations

import pytest

from common.config import load_config


# ==========================================================================
# Task 1
# ==========================================================================


def test_task1_uses_assignment_seed_everywhere() -> None:
    cfg = load_config("task1/configs/interventions.yaml")
    assert cfg.seed == 6304
    assert cfg.data.split_seed == 6304
    assert cfg.data.eval_subset.seed == 6304
    assert cfg.head.seed == 6304
    assert cfg.interventions.patch_shuffle.seed == 6304
    assert cfg.representation_analysis.visualization.seed == 6304


def test_task1_eval_subset_is_balanced_500() -> None:
    cfg = load_config("task1/configs/base.yaml")
    assert cfg.data.eval_subset.size == 500
    assert cfg.data.eval_subset.balanced is True
    # Saved identifiers are what guarantee every model sees the same images.
    assert cfg.data.eval_subset.manifest.endswith(".json")


def test_task1_head_hyperparameters_match_assignment() -> None:
    cfg = load_config("task1/configs/base.yaml")
    assert cfg.head.epochs == 50
    assert cfg.head.optimizer == "adamw"
    assert cfg.head.lr == 1e-3
    assert cfg.head.weight_decay == 1e-4
    assert cfg.head.early_stopping_patience == 5


def test_task1_backbones_are_the_three_required() -> None:
    cfg = load_config("task1/configs/base.yaml")
    assert set(dict.keys(cfg.backbones)) == {"resnet50", "vit_b16", "clip_vitb32"}
    assert cfg.backbones.resnet50.weights == "ResNet50_Weights.IMAGENET1K_V2"
    assert cfg.backbones.vit_b16.weights == "ViT_B_16_Weights.IMAGENET1K_V1"
    assert cfg.backbones.clip_vitb32.model_name == "ViT-B-32"
    assert cfg.backbones.clip_vitb32.pretrained == "openai"
    # CLIP's image embedding must be the NORMALIZED one.
    assert cfg.backbones.clip_vitb32.normalize_embedding is True


def test_task1_common_canvas_before_normalization() -> None:
    """Every model must receive byte-identical clean and transformed images."""
    cfg = load_config("task1/configs/base.yaml")
    assert cfg.data.image_size == 224


def test_task1_zero_shot_prompt_is_the_fixed_one() -> None:
    cfg = load_config("task1/configs/base.yaml")
    assert cfg.zero_shot.prompt_template == "a photo of a {class}."


def test_cue_conflict_rejection_rule_is_model_blind() -> None:
    """Model predictions must not decide which stylizations are retained.

    Filtering on predictions would bias the shape-bias score toward whichever
    cue the model already prefers, making the measurement circular.
    """
    rule = load_config("task1/configs/interventions.yaml").interventions.cue_conflict
    assert rule.rejection_rule.defined_before_evaluation is True
    assert rule.rejection_rule.uses_model_predictions is False
    assert set(rule.rejection_rule.record_counts) == {"accepted", "rejected"}


def test_cue_conflict_scale_meets_minimums() -> None:
    cc = load_config("task1/configs/interventions.yaml").interventions.cue_conflict
    assert cc.class_pairs_min >= 5
    assert cc.target_valid_conflicts >= 200
    assert cc.bidirectional is True


def test_translation_grid_and_directions() -> None:
    tr = load_config("task1/configs/interventions.yaml").interventions.translation
    assert tr.displacements == [0, 8, 16, 32]
    assert set(tr.directions) == {"up", "down", "left", "right"}
    assert tr.padding == "reflect"


def test_patch_shuffle_is_4x4_nonidentity_and_shared() -> None:
    ps = load_config("task1/configs/interventions.yaml").interventions.patch_shuffle
    assert ps.grid == [4, 4]
    assert ps.non_identity is True
    assert ps.shared_across_models is True


def test_representation_analysis_covers_all_required_interventions() -> None:
    ra = load_config("task1/configs/interventions.yaml").representation_analysis
    assert set(ra.required_for) == {
        "grayscale",
        "cue_conflict",
        "translation",
        "patch_shuffle",
    }
    assert ra.cosine_stability is True
    # One projection per backbone, fit to clean and transformed together.
    assert ra.visualization.fit_on == "combined_clean_and_transformed"
    assert ra.visualization.method in {"tsne", "umap"}
    assert ra.visualization.cross_backbone_coordinates_comparable is False


# ==========================================================================
# Task 4
# ==========================================================================


def test_task4_cifar_stem_is_modified_for_32px() -> None:
    cfg = load_config("task4/configs/base.yaml")
    assert cfg.model.stem_conv.kernel == 3
    assert cfg.model.stem_conv.stride == 1
    assert cfg.model.remove_maxpool is True
    assert cfg.model.input_size == 32


def test_task4_training_recipe_matches_assignment() -> None:
    cfg = load_config("task4/configs/vanilla.yaml")
    assert cfg.train.optimizer == "sgd"
    assert cfg.train.lr == 0.1
    assert cfg.train.momentum == 0.9
    assert cfg.train.weight_decay == 5e-4
    assert cfg.train.scheduler == "cosine"
    assert cfg.train.batch_size == 128
    assert cfg.train.epochs == 100
    assert cfg.seed == 6304


def test_task4_split_is_stratified_90_10() -> None:
    cfg = load_config("task4/configs/base.yaml")
    assert cfg.data.known.val_fraction == 0.1
    assert cfg.data.known.split_seed == 6304


def test_unknown_groups_are_the_fixed_sixteen_classes() -> None:
    """The near/far grouping is fixed and may not be revised after results."""
    unk = load_config("task4/configs/base.yaml").data.unknown
    assert unk.near == [
        "bus", "pickup_truck", "motorcycle", "tractor",
        "wolf", "fox", "leopard", "camel",
    ]
    assert unk.far == [
        "bottle", "bowl", "chair", "clock",
        "keyboard", "mushroom", "sunflower", "wardrobe",
    ]
    assert len(unk.near) == len(unk.far) == 8
    assert not set(unk.near) & set(unk.far)
    assert unk.expected_per_group == 800
    assert unk.evaluation_only is True
    assert unk.split == "test"          # CIFAR-100 train is never touched


def test_threshold_calibrated_on_known_validation_only() -> None:
    """No real unknown may influence the rejection threshold."""
    ev = load_config("task4/configs/base.yaml").evaluation
    assert ev.threshold.calibration_set == "cifar10_val"
    assert ev.threshold.percentile == 95
    assert ev.threshold.accept_rule == "u(x) <= tau"


def test_checkpoint_selection_never_uses_unknowns() -> None:
    for name in ["vanilla", "gcsc", "proser"]:
        cfg = load_config(f"task4/configs/{name}.yaml")
        assert cfg.selection.metric == "cifar10_val_accuracy"


def test_gcsc_differs_from_vanilla_only_by_randaugment() -> None:
    """Controlled comparison: exactly one intended difference."""
    van = load_config("task4/configs/vanilla.yaml")
    gcs = load_config("task4/configs/gcsc.yaml")

    for field in ["optimizer", "lr", "momentum", "weight_decay",
                  "scheduler", "batch_size", "epochs"]:
        assert getattr(van.train, field) == getattr(gcs.train, field), field
    assert van.seed == gcs.seed
    assert van.model.arch == gcs.model.arch

    # The one permitted change.
    assert "rand_augment" not in dict.keys(van.train.augmentation)
    assert gcs.train.augmentation.rand_augment.num_ops == 2
    assert gcs.train.augmentation.rand_augment.magnitude == 9
    # Shared crop/flip are unchanged.
    assert van.train.augmentation.random_crop == gcs.train.augmentation.random_crop
    assert van.train.augmentation.random_horizontal_flip is True
    assert gcs.train.augmentation.random_horizontal_flip is True


def test_mahalanobis_uses_shared_diagonal_from_unaugmented_train() -> None:
    scores = load_config("task4/configs/vanilla.yaml").scores
    assert scores.mahalanobis.covariance == "shared_diagonal"
    assert scores.mahalanobis.estimated_from == "cifar10_train_unaugmented"
    assert scores.mahalanobis.diagonal_epsilon == 1e-6


def test_all_four_scores_read_the_same_cached_outputs() -> None:
    """Score differences must not come from evaluation-time variation."""
    cfg = load_config("task4/configs/vanilla.yaml")
    assert set(cfg.extraction.save) == {"features", "logits"}
    assert set(dict.keys(cfg.scores)) == {"msp", "mls", "energy", "mahalanobis"}
    for split in ["cifar10_train", "cifar10_val", "cifar10_test",
                  "cifar100_near", "cifar100_far"]:
        assert split in cfg.extraction.splits


def test_proser_initialized_from_vanilla_with_five_dummies() -> None:
    cfg = load_config("task4/configs/proser.yaml")
    assert cfg.method.init_from == "checkpoints/task4/vanilla/best.pt"
    assert cfg.method.dummy_classifiers == 5
    assert cfg.method.classifier_placeholder.beta == 1.0
    assert cfg.method.data_placeholder.gamma == 0.1


def test_proser_finetune_recipe() -> None:
    cfg = load_config("task4/configs/proser.yaml")
    assert cfg.train.epochs == 50
    assert cfg.train.lr == 1e-3
    assert cfg.train.optimizer == "sgd"
    assert cfg.train.momentum == 0.9
    assert cfg.train.weight_decay == 5e-4
    assert cfg.train.scheduler == "cosine"
    assert cfg.train.batch_size == 128


def test_manifold_mixup_location_and_distribution() -> None:
    """Required: mix after layer2, before layer3, with lambda ~ Beta(2,2)."""
    dp = load_config("task4/configs/proser.yaml").method.data_placeholder
    assert dp.method == "manifold_mixup"
    assert dp.mix_location == "after_layer2"
    assert dp.lam_distribution.alpha == 2.0
    assert dp.lam_distribution.beta == 2.0
    # Mixing two examples of the SAME class would not create an unknown-like
    # proxy between distinct known regions.
    assert dp.require_different_classes is True
    assert dp.train_mixed_toward == "dummy_classifiers"


def test_proser_batch_split_is_two_equal_halves() -> None:
    bs = load_config("task4/configs/proser.yaml").method.batch_split
    assert bs.classifier_placeholder_fraction == 0.5
    assert bs.data_placeholder_fraction == 0.5
    assert bs.classifier_placeholder_fraction + bs.data_placeholder_fraction == 1.0


def test_proser_csa_and_mls_use_known_logits_only() -> None:
    """Keeps known-class classification and rejection distinct."""
    cfg = load_config("task4/configs/proser.yaml")
    assert cfg.scores.mls_uses_known_logits_only is True
    assert cfg.evaluation.csa_uses_known_logits_only is True
    assert cfg.scores.placeholder_detection.calibration_set == "cifar10_val"


def test_rpl_is_scaffold_only() -> None:
    """The optional extension must not masquerade as completed work."""
    cfg = load_config("task4/configs/rpl.yaml")
    assert cfg.method.implemented is False
    assert cfg.method.score.convention == "larger_is_more_novel"


@pytest.mark.parametrize("name", ["vanilla", "gcsc", "proser", "rpl"])
def test_task4_configs_keep_unknowns_evaluation_only(name: str) -> None:
    cfg = load_config(f"task4/configs/{name}.yaml")
    assert cfg.data.unknown.evaluation_only is True
    assert cfg.data.unknown.split == "test"
