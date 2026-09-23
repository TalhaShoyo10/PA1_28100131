"""Frozen backbone wrappers for ResNet-50, ViT-B/16 and CLIP ViT-B/32."""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch
import torch.nn as nn
from torchvision import models, transforms

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)


class FrozenBackbone(ABC, nn.Module):
    """A pretrained feature extractor whose parameters are never updated."""

    name: str
    feature_dim: int

    def __init__(self) -> None:
        super().__init__()

    def freeze(self) -> None:
        """Disable gradients and place the module in eval mode permanently."""
        for param in self.parameters():
            param.requires_grad = False
        self.eval()

    @abstractmethod
    def normalization(self) -> transforms.Normalize:
        """Return the normalization this backbone's weights expect."""

    @abstractmethod
    def extract(self, x: torch.Tensor) -> torch.Tensor:
        """Return the final representation for a normalized batch."""

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.extract(x)

    def train(self, mode: bool = True):
        """Override: a frozen backbone stays in eval mode regardless."""
        return super().train(False)


class ResNet50Backbone(FrozenBackbone):
    """torchvision ResNet-50, IMAGENET1K_V2, global-average-pooled feature."""

    name = "resnet50"
    feature_dim = 2048

    def __init__(self, pretrained: bool = True) -> None:
        super().__init__()
        weights = models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        resnet = models.resnet50(weights=weights)
        self.body = nn.Sequential(*list(resnet.children())[:-1])
        self.freeze()

    def normalization(self) -> transforms.Normalize:
        return transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)

    def extract(self, x: torch.Tensor) -> torch.Tensor:
        return torch.flatten(self.body(x), 1)


class ViTB16Backbone(FrozenBackbone):
    """torchvision ViT-B/16, IMAGENET1K_V1, final class token."""

    name = "vit_b16"
    feature_dim = 768

    def __init__(self, pretrained: bool = True) -> None:
        super().__init__()
        weights = models.ViT_B_16_Weights.IMAGENET1K_V1 if pretrained else None
        self.vit = models.vit_b_16(weights=weights)
        self.freeze()

    def normalization(self) -> transforms.Normalize:
        return transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)

    def extract(self, x: torch.Tensor) -> torch.Tensor:
        v = self.vit
        x = v._process_input(x)
        batch_class_token = v.class_token.expand(x.shape[0], -1, -1)
        x = torch.cat([batch_class_token, x], dim=1)
        x = v.encoder(x)
        return x[:, 0]


class CLIPBackbone(FrozenBackbone):
    """OpenCLIP ViT-B-32 (pretrained='openai'), normalized image embedding."""

    name = "clip_vitb32"
    feature_dim = 512

    def __init__(self, model_name: str = "ViT-B-32", pretrained: str = "openai") -> None:
        super().__init__()
        try:
            import open_clip
        except ImportError as exc:
            raise ImportError(
                "open_clip_torch is required for the CLIP backbone. "
                "Install it with: pip install open_clip_torch"
            ) from exc

        # OpenAI's CLIP was trained with QuickGELU activations. Newer open_clip
        # versions split the architecture, so the plain "ViT-B-32" name builds
        # the non-QuickGELU variant and warns about the mismatch while still
        # loading the OpenAI weights -- a silently WRONG forward pass, because
        # the weights then run through a different activation than they were
        # trained with. Requesting quick_gelu explicitly is what makes the model
        # match the mandated pretrained='openai' checkpoint.
        kwargs = {"pretrained": pretrained}
        if pretrained == "openai":
            kwargs["quick_gelu"] = True

        try:
            self.model, _, _ = open_clip.create_model_and_transforms(
                model_name, **kwargs
            )
        except TypeError:
            # Older open_clip has no quick_gelu argument; there the plain name
            # already builds the QuickGELU variant for the openai tag.
            self.model, _, _ = open_clip.create_model_and_transforms(
                model_name, pretrained=pretrained
            )

        self.tokenizer = open_clip.get_tokenizer(model_name)
        self.model_name = model_name
        self.freeze()

    def normalization(self) -> transforms.Normalize:
        return transforms.Normalize(CLIP_MEAN, CLIP_STD)

    def extract(self, x: torch.Tensor) -> torch.Tensor:
        features = self.model.encode_image(x)
        return features / features.norm(dim=-1, keepdim=True)

    @torch.no_grad()
    def encode_text(self, prompts: list[str]) -> torch.Tensor:
        """Return normalized text embeddings for zero-shot classification."""
        device = next(self.model.parameters()).device
        tokens = self.tokenizer(prompts).to(device)
        features = self.model.encode_text(tokens)
        return features / features.norm(dim=-1, keepdim=True)

    @property
    def logit_scale(self) -> torch.Tensor:
        """The learned temperature applied to cosine similarities."""
        return self.model.logit_scale.exp()


class LinearHead(nn.Module):
    """Linear classifier trained on top of a frozen backbone's features."""

    def __init__(self, feature_dim: int, num_classes: int) -> None:
        super().__init__()
        self.fc = nn.Linear(feature_dim, num_classes)
        self.feature_dim = feature_dim
        self.num_classes = num_classes

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.fc(features)


BACKBONE_REGISTRY = {
    "resnet50": ResNet50Backbone,
    "vit_b16": ViTB16Backbone,
    "clip_vitb32": CLIPBackbone,
}


def build_backbone(name: str, **kwargs) -> FrozenBackbone:
    """Instantiate a frozen backbone by registry name."""
    if name not in BACKBONE_REGISTRY:
        raise KeyError(
            f"Unknown backbone {name!r}. Available: {sorted(BACKBONE_REGISTRY)}"
        )
    return BACKBONE_REGISTRY[name](**kwargs)
