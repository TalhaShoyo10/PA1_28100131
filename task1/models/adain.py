"""AdaIN style transfer for cue-conflict generation.

Attribution: the AdaIN operation, encoder truncation and decoder architecture
below are reimplemented from Huang and Belongie (2017). The pretrained VGG
encoder and decoder WEIGHTS are downloaded from the public PyTorch port
naoto0804/pytorch-AdaIN and are not authored here. See README.md.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from urllib.request import urlopen

import torch
import torch.nn as nn

VGG_URL = "https://github.com/naoto0804/pytorch-AdaIN/releases/download/v0.0.0/vgg_normalised.pth"
DECODER_URL = "https://github.com/naoto0804/pytorch-AdaIN/releases/download/v0.0.0/decoder.pth"

DEFAULT_WEIGHTS_DIR = Path("task1/models/weights")


def adaptive_instance_normalization(
    content: torch.Tensor, style: torch.Tensor, eps: float = 1e-5
) -> torch.Tensor:
    """Align content feature statistics to style feature statistics per channel."""
    if content.shape[:2] != style.shape[:2]:
        raise ValueError(
            f"Batch/channel mismatch: content {tuple(content.shape[:2])} "
            f"vs style {tuple(style.shape[:2])}"
        )

    b, c = content.shape[:2]
    content_flat = content.reshape(b, c, -1)
    style_flat = style.reshape(b, c, -1)

    content_mean = content_flat.mean(dim=2).reshape(b, c, 1, 1)
    content_std = content_flat.var(dim=2).add(eps).sqrt().reshape(b, c, 1, 1)
    style_mean = style_flat.mean(dim=2).reshape(b, c, 1, 1)
    style_std = style_flat.var(dim=2).add(eps).sqrt().reshape(b, c, 1, 1)

    normalized = (content - content_mean) / content_std
    return normalized * style_std + style_mean


RELU4_1_CUTOFF = 31


def build_full_vgg() -> nn.Sequential:
    """The complete normalised VGG-19, matching the published checkpoint.

    ``vgg_normalised.pth`` stores the whole network, not a truncation, so the
    checkpoint must be loaded into this full definition before slicing. The
    module indices are what the state_dict keys refer to, so the layer order
    here is fixed by the published file.
    """
    return nn.Sequential(
        nn.Conv2d(3, 3, 1),
        nn.ReflectionPad2d(1), nn.Conv2d(3, 64, 3), nn.ReLU(),
        nn.ReflectionPad2d(1), nn.Conv2d(64, 64, 3), nn.ReLU(),
        nn.MaxPool2d(2, 2, ceil_mode=True),
        nn.ReflectionPad2d(1), nn.Conv2d(64, 128, 3), nn.ReLU(),
        nn.ReflectionPad2d(1), nn.Conv2d(128, 128, 3), nn.ReLU(),
        nn.MaxPool2d(2, 2, ceil_mode=True),
        nn.ReflectionPad2d(1), nn.Conv2d(128, 256, 3), nn.ReLU(),
        nn.ReflectionPad2d(1), nn.Conv2d(256, 256, 3), nn.ReLU(),
        nn.ReflectionPad2d(1), nn.Conv2d(256, 256, 3), nn.ReLU(),
        nn.ReflectionPad2d(1), nn.Conv2d(256, 256, 3), nn.ReLU(),
        nn.MaxPool2d(2, 2, ceil_mode=True),
        nn.ReflectionPad2d(1), nn.Conv2d(256, 512, 3), nn.ReLU(),   # relu4_1
        nn.ReflectionPad2d(1), nn.Conv2d(512, 512, 3), nn.ReLU(),
        nn.ReflectionPad2d(1), nn.Conv2d(512, 512, 3), nn.ReLU(),
        nn.ReflectionPad2d(1), nn.Conv2d(512, 512, 3), nn.ReLU(),
        nn.MaxPool2d(2, 2, ceil_mode=True),
        nn.ReflectionPad2d(1), nn.Conv2d(512, 512, 3), nn.ReLU(),
        nn.ReflectionPad2d(1), nn.Conv2d(512, 512, 3), nn.ReLU(),
        nn.ReflectionPad2d(1), nn.Conv2d(512, 512, 3), nn.ReLU(),
        nn.ReflectionPad2d(1), nn.Conv2d(512, 512, 3), nn.ReLU(),
    )


def build_vgg_encoder() -> nn.Sequential:
    """VGG-19 truncated after relu4_1: the AdaIN content/style encoder."""
    return nn.Sequential(*list(build_full_vgg())[:RELU4_1_CUTOFF])


def build_decoder() -> nn.Sequential:
    """Mirror of the encoder, mapping relu4_1 features back to RGB."""
    return nn.Sequential(
        nn.ReflectionPad2d(1), nn.Conv2d(512, 256, 3), nn.ReLU(),
        nn.Upsample(scale_factor=2, mode="nearest"),
        nn.ReflectionPad2d(1), nn.Conv2d(256, 256, 3), nn.ReLU(),
        nn.ReflectionPad2d(1), nn.Conv2d(256, 256, 3), nn.ReLU(),
        nn.ReflectionPad2d(1), nn.Conv2d(256, 256, 3), nn.ReLU(),
        nn.ReflectionPad2d(1), nn.Conv2d(256, 128, 3), nn.ReLU(),
        nn.Upsample(scale_factor=2, mode="nearest"),
        nn.ReflectionPad2d(1), nn.Conv2d(128, 128, 3), nn.ReLU(),
        nn.ReflectionPad2d(1), nn.Conv2d(128, 64, 3), nn.ReLU(),
        nn.Upsample(scale_factor=2, mode="nearest"),
        nn.ReflectionPad2d(1), nn.Conv2d(64, 64, 3), nn.ReLU(),
        nn.ReflectionPad2d(1), nn.Conv2d(64, 3, 3),
    )


def download_weights(
    url: str, destination: Path, expected_sha256: str | None = None
) -> Path:
    """Download a checkpoint to ``destination`` unless it already exists."""
    destination = Path(destination)
    if destination.exists():
        return destination

    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        with urlopen(url, timeout=120) as response:
            payload = response.read()
    except Exception as exc:
        raise RuntimeError(
            f"Failed to download AdaIN weights from {url}. These are required "
            "for cue-conflict generation and are not distributed with this "
            "repository. Download manually and place at "
            f"{destination}. Original error: {exc}"
        ) from exc

    digest = hashlib.sha256(payload).hexdigest()
    if expected_sha256 is not None and digest != expected_sha256:
        raise RuntimeError(
            f"Checksum mismatch for {destination.name}: expected "
            f"{expected_sha256}, got {digest}. Refusing to use the file."
        )

    destination.write_bytes(payload)
    return destination


class AdaINStyleTransfer(nn.Module):
    """Frozen AdaIN stylizer producing cue-conflict images."""

    def __init__(
        self,
        weights_dir: Path | str = DEFAULT_WEIGHTS_DIR,
        download: bool = True,
        vgg_sha256: str | None = None,
        decoder_sha256: str | None = None,
    ) -> None:
        super().__init__()
        weights_dir = Path(weights_dir)

        self.decoder = build_decoder()

        vgg_path = weights_dir / "vgg_normalised.pth"
        decoder_path = weights_dir / "decoder.pth"

        if download:
            download_weights(VGG_URL, vgg_path, vgg_sha256)
            download_weights(DECODER_URL, decoder_path, decoder_sha256)

        for path in (vgg_path, decoder_path):
            if not path.exists():
                raise FileNotFoundError(
                    f"AdaIN weights missing: {path}. Run with download=True or "
                    "fetch them manually (see README.md)."
                )

        full_vgg = build_full_vgg()
        full_vgg.load_state_dict(torch.load(vgg_path, map_location="cpu"))
        self.encoder = nn.Sequential(*list(full_vgg)[:RELU4_1_CUTOFF])

        self.decoder.load_state_dict(torch.load(decoder_path, map_location="cpu"))

        for param in self.parameters():
            param.requires_grad = False
        self.eval()

    def train(self, mode: bool = True):
        """Override: the stylizer is frozen and stays in eval mode."""
        return super().train(False)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode an RGB batch in [0, 1] to relu4_1 features."""
        return self.encoder(x)

    @torch.no_grad()
    def forward(
        self, content: torch.Tensor, style: torch.Tensor, alpha: float = 1.0
    ) -> torch.Tensor:
        """Stylize ``content`` with ``style``; ``alpha`` interpolates the effect."""
        if not 0.0 <= alpha <= 1.0:
            raise ValueError(f"alpha must be in [0, 1], got {alpha}")

        content_features = self.encode(content)
        style_features = self.encode(style)

        target = adaptive_instance_normalization(content_features, style_features)
        blended = alpha * target + (1.0 - alpha) * content_features

        return self.decoder(blended).clamp(0.0, 1.0)
