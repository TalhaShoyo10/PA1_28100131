"""Tests for the AdaIN stylizer and the cue-conflict rejection rule.

The AdaIN *operation* is reimplemented in this repository and is fully tested
here. The pretrained encoder/decoder WEIGHTS are external downloads, so tests
needing them are skipped when absent -- the architecture shapes they must load
into are asserted unconditionally.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn as nn
from PIL import Image

from task1.data.make_cue_conflicts import (
    evaluate_rejection_rule,
    saturation_mean,
    select_class_pairs,
    structure_score,
    summarize,
    CueConflict,
)
from task1.models.adain import (
    DEFAULT_WEIGHTS_DIR,
    AdaINStyleTransfer,
    adaptive_instance_normalization,
    build_decoder,
    build_vgg_encoder,
)

WEIGHTS_PRESENT = (DEFAULT_WEIGHTS_DIR / "vgg_normalised.pth").exists() and (
    DEFAULT_WEIGHTS_DIR / "decoder.pth"
).exists()
requires_weights = pytest.mark.skipif(
    not WEIGHTS_PRESENT, reason="AdaIN pretrained weights not downloaded"
)


# --------------------------------------------------------------------------
# The AdaIN operation
# --------------------------------------------------------------------------


@pytest.fixture
def features():
    torch.manual_seed(6304)
    content = torch.randn(2, 64, 28, 28) * 3.0 + 7.0
    style = torch.randn(2, 64, 28, 28) * 0.5 - 2.0
    return content, style


def _stats(x: torch.Tensor):
    b, c = x.shape[:2]
    flat = x.reshape(b, c, -1)
    return flat.mean(2), flat.std(2)


def test_output_adopts_style_statistics(features) -> None:
    """The defining property: output per-channel mean/std match the STYLE."""
    content, style = features
    out = adaptive_instance_normalization(content, style)

    out_mean, out_std = _stats(out)
    style_mean, style_std = _stats(style)

    assert torch.allclose(out_mean, style_mean, atol=1e-4)
    assert torch.allclose(out_std, style_std, atol=1e-3)


def test_output_does_not_keep_content_statistics(features) -> None:
    """Control: confirms the previous test is not passing trivially."""
    content, style = features
    out_mean, _ = _stats(adaptive_instance_normalization(content, style))
    content_mean, _ = _stats(content)
    assert not torch.allclose(out_mean, content_mean, atol=1.0)


def test_content_spatial_structure_is_preserved(features) -> None:
    """Shape must survive: normalized content and output are perfectly correlated."""
    content, style = features
    out = adaptive_instance_normalization(content, style)

    c_mean, c_std = _stats(content)
    o_mean, o_std = _stats(out)
    b, c = content.shape[:2]
    content_norm = (content - c_mean.reshape(b, c, 1, 1)) / c_std.reshape(b, c, 1, 1)
    out_norm = (out - o_mean.reshape(b, c, 1, 1)) / o_std.reshape(b, c, 1, 1)

    corr = torch.corrcoef(torch.stack([content_norm.flatten(), out_norm.flatten()]))[0, 1]
    assert corr.item() > 0.999


def test_identical_content_and_style_is_near_identity(features) -> None:
    content, _ = features
    out = adaptive_instance_normalization(content, content)
    assert torch.allclose(out, content, atol=1e-3)


def test_operation_is_deterministic(features) -> None:
    content, style = features
    a = adaptive_instance_normalization(content, style)
    b = adaptive_instance_normalization(content, style)
    assert torch.equal(a, b)


def test_constant_channel_does_not_produce_nan() -> None:
    """Zero variance would divide by zero without the epsilon guard."""
    content = torch.ones(1, 8, 16, 16)
    style = torch.randn(1, 8, 16, 16)
    assert torch.isfinite(adaptive_instance_normalization(content, style)).all()


def test_rejects_mismatched_shapes() -> None:
    with pytest.raises(ValueError, match="mismatch"):
        adaptive_instance_normalization(torch.randn(2, 64, 8, 8), torch.randn(2, 32, 8, 8))


# --------------------------------------------------------------------------
# Architecture
# --------------------------------------------------------------------------


def test_encoder_downsamples_by_eight() -> None:
    """VGG-19 to relu4_1 has three pooling stages: 224 -> 28, 512 channels."""
    out = build_vgg_encoder()(torch.rand(1, 3, 224, 224))
    assert out.shape == (1, 512, 28, 28)


def test_decoder_restores_input_resolution() -> None:
    out = build_decoder()(torch.rand(1, 512, 28, 28))
    assert out.shape == (1, 3, 224, 224)


def test_architecture_matches_reference_checkpoint_shapes() -> None:
    """Guards against a silent layer mismatch when the real weights load."""
    enc = [tuple(v.shape) for v in build_vgg_encoder().state_dict().values()]
    dec = [tuple(v.shape) for v in build_decoder().state_dict().values()]

    assert enc[0] == (3, 3, 1, 1)          # the 1x1 input conv
    assert enc[-2] == (512, 256, 3, 3)     # final conv into relu4_1
    assert dec[0] == (256, 512, 3, 3)      # first decoder conv
    assert dec[-2] == (3, 64, 3, 3)        # output conv back to RGB


# --------------------------------------------------------------------------
# Stylizer plumbing (random weights: shapes and contracts only)
# --------------------------------------------------------------------------


class _StubStylizer(AdaINStyleTransfer):
    """AdaINStyleTransfer with random weights, for testing plumbing offline."""

    def __init__(self) -> None:
        nn.Module.__init__(self)
        self.encoder = build_vgg_encoder()
        self.decoder = build_decoder()
        for param in self.parameters():
            param.requires_grad = False
        self.eval()


@pytest.fixture
def stub():
    torch.manual_seed(6304)
    return _StubStylizer()


def test_stylize_preserves_shape_and_clamps_range(stub) -> None:
    out = stub(torch.rand(2, 3, 224, 224), torch.rand(2, 3, 224, 224))
    assert out.shape == (2, 3, 224, 224)
    assert 0.0 <= out.min() and out.max() <= 1.0


def test_stylize_is_deterministic(stub) -> None:
    c, s = torch.rand(1, 3, 224, 224), torch.rand(1, 3, 224, 224)
    assert torch.allclose(stub(c, s), stub(c, s))


def test_alpha_out_of_range_rejected(stub) -> None:
    c, s = torch.rand(1, 3, 224, 224), torch.rand(1, 3, 224, 224)
    for bad in (-0.1, 1.5):
        with pytest.raises(ValueError, match="alpha"):
            stub(c, s, alpha=bad)


def test_stylizer_stays_frozen_in_eval(stub) -> None:
    """A frozen module must ignore .train(), or dropout/BN would drift."""
    stub.train()
    assert stub.training is False
    assert all(not p.requires_grad for p in stub.parameters())


def test_missing_weights_gives_actionable_error() -> None:
    with pytest.raises(FileNotFoundError, match="README"):
        AdaINStyleTransfer(weights_dir="task1/models/definitely_absent", download=False)


@requires_weights
def test_real_weights_load_into_the_architectures() -> None:
    """Only meaningful once the external checkpoints are present."""
    stylizer = AdaINStyleTransfer(download=False)
    out = stylizer(torch.rand(1, 3, 224, 224), torch.rand(1, 3, 224, 224))
    assert out.shape == (1, 3, 224, 224)


# --------------------------------------------------------------------------
# Class pair selection
# --------------------------------------------------------------------------


def test_pairs_are_unordered_distinct_and_deterministic() -> None:
    pairs = select_class_pairs(10, 5, seed=6304)
    assert len(pairs) == 5
    assert all(a < b for a, b in pairs)
    assert len(set(pairs)) == 5
    assert pairs == select_class_pairs(10, 5, seed=6304)


def test_fewer_than_five_pairs_rejected() -> None:
    """The assignment requires at least five unordered class pairs."""
    with pytest.raises(ValueError, match="at least 5"):
        select_class_pairs(10, 4, seed=6304)


def test_more_pairs_than_exist_rejected() -> None:
    with pytest.raises(ValueError, match="only"):
        select_class_pairs(4, 7, seed=6304)


# --------------------------------------------------------------------------
# Rejection rule (must be model-blind)
# --------------------------------------------------------------------------


def _textured(seed: int = 0) -> Image.Image:
    rng = np.random.default_rng(seed)
    return Image.fromarray(rng.integers(0, 255, (224, 224, 3), dtype=np.uint8))


def test_structure_score_rises_with_detail() -> None:
    flat = Image.new("RGB", (224, 224), (128, 128, 128))
    assert structure_score(flat) == pytest.approx(0.0, abs=1e-6)
    assert structure_score(_textured(6304)) > 10.0


def test_saturation_mean_detects_gray() -> None:
    assert saturation_mean(Image.new("RGB", (64, 64), (128, 128, 128))) == pytest.approx(0.0)
    assert saturation_mean(Image.new("RGB", (64, 64), (255, 0, 0))) > 200


def test_accepts_a_plausible_stylization() -> None:
    content = _textured(1)
    style = _textured(2)
    stylized = Image.blend(content, style, 0.5)
    accepted, reason = evaluate_rejection_rule(stylized, content, style)
    assert accepted, reason


def test_rejects_saturation_collapse() -> None:
    content = _textured(1)
    washed = Image.new("RGB", (224, 224), (128, 128, 128))
    accepted, reason = evaluate_rejection_rule(washed, content, _textured(2))
    assert not accepted
    assert "structure" in reason or "saturation" in reason


def test_rejects_unstylized_passthrough() -> None:
    """If the style never took effect the conflict is ambiguous, not a conflict."""
    content = _textured(1)
    accepted, reason = evaluate_rejection_rule(content, content, _textured(2))
    assert not accepted
    assert "style_not_applied" in reason


def test_rejects_degenerate_content() -> None:
    flat = Image.new("RGB", (224, 224), (128, 128, 128))
    accepted, reason = evaluate_rejection_rule(_textured(1), flat, _textured(2))
    assert not accepted
    assert reason == "degenerate_content"


def test_rejection_rule_never_sees_a_model() -> None:
    """Filtering on predictions would make shape bias circular.

    The rule's signature accepts only images, so a model prediction cannot be
    passed in even by accident.
    """
    import inspect

    params = set(inspect.signature(evaluate_rejection_rule).parameters)
    assert params == {
        "stylized", "content", "style",
        "min_structure_ratio", "min_saturation", "min_texture_change",
    }


# --------------------------------------------------------------------------
# Summary accounting
# --------------------------------------------------------------------------


def test_summary_counts_accepted_and_rejected() -> None:
    """Both counts must be recorded, as the assignment requires."""
    records = [
        CueConflict("a.png", 0, 1, "cat", "dog", "i1", "i2", "cat__dog", "d", 1.0, True),
        CueConflict("b.png", 0, 1, "cat", "dog", "i3", "i4", "cat__dog", "d", 1.0, False, "saturation_collapse(3.0)"),
        CueConflict("c.png", 1, 0, "dog", "cat", "i5", "i6", "cat__dog", "d2", 1.0, True),
    ]
    summary = summarize(records)

    assert summary["total_generated"] == 3
    assert summary["accepted"] == 2
    assert summary["rejected"] == 1
    assert summary["acceptance_rate"] == pytest.approx(200.0 / 3)
    assert summary["by_pair"]["cat__dog"] == {"accepted": 2, "rejected": 1}
    assert summary["rejection_reasons"]["saturation_collapse(3.0)"] == 1


def test_summary_handles_no_records() -> None:
    summary = summarize([])
    assert summary["total_generated"] == 0
    assert summary["acceptance_rate"] == 0.0
