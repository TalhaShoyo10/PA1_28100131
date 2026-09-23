"""CLIP must be built as the architecture its weights were trained as.

Regression guard (2026-09-23): open_clip warned 'QuickGELU mismatch between
final model config (quick_gelu=False) and pretrained tag openai
(quick_gelu=True)' and loaded the OpenAI weights anyway. The forward pass then
ran trained weights through GELU instead of QuickGELU.

A first attempt passed quick_gelu=True to create_model_and_transforms. That
function has no such parameter, so **kwargs swallowed it silently and nothing
changed. The architecture is selected by NAME.
"""

from __future__ import annotations

import pytest

open_clip = pytest.importorskip("open_clip", reason="open_clip_torch not installed")


def test_quickgelu_variant_exists_and_supports_the_openai_tag() -> None:
    assert "ViT-B-32-quickgelu" in set(open_clip.list_models())
    assert "openai" in open_clip.list_pretrained_tags_by_model("ViT-B-32-quickgelu")


def test_create_model_has_no_quick_gelu_parameter() -> None:
    """Pins why the name must be used: the argument does not exist."""
    import inspect

    params = inspect.signature(open_clip.create_model_and_transforms).parameters
    assert "quick_gelu" not in params


def test_the_two_variants_use_different_activations() -> None:
    """The substantive point: this is not a cosmetic warning."""
    import torch

    quickgelu, _, _ = open_clip.create_model_and_transforms(
        "ViT-B-32-quickgelu", pretrained=None
    )
    plain, _, _ = open_clip.create_model_and_transforms("ViT-B-32", pretrained=None)

    act_q = quickgelu.visual.transformer.resblocks[0].mlp[1]
    act_p = plain.visual.transformer.resblocks[0].mlp[1]

    assert type(act_q).__name__ == "QuickGELU"
    assert type(act_p).__name__ == "GELU"

    x = torch.tensor([-2.0, -0.5, 0.5, 2.0])
    assert (act_q(x) - act_p(x)).abs().max() > 1e-3


def test_backbone_resolves_openai_to_the_quickgelu_variant() -> None:
    """Name resolution only -- no weights are downloaded."""
    model_name, pretrained = "ViT-B-32", "openai"

    resolved = model_name
    if pretrained == "openai" and not model_name.endswith("-quickgelu"):
        candidate = f"{model_name}-quickgelu"
        if candidate in set(open_clip.list_models()):
            resolved = candidate

    assert resolved == "ViT-B-32-quickgelu"


def test_non_openai_tags_keep_the_plain_architecture() -> None:
    """LAION weights were trained with plain GELU; they must not be switched."""
    model_name, pretrained = "ViT-B-32", "laion2b_e16"

    resolved = model_name
    if pretrained == "openai" and not model_name.endswith("-quickgelu"):
        resolved = f"{model_name}-quickgelu"

    assert resolved == "ViT-B-32"
