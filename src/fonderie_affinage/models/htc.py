"""HTC base model: the DKFZ organ-segmentation U-Net as a region classifier.

HTC (IMSY-DKFZ, "hyperspectral tissue classification") publishes U-Nets with an
EfficientNet-B5 encoder that were trained to segment porcine organs in whole
TIVITA frames (480 x 640 pixels, 100 bands, each pixel L1-normalised). Here the
encoder and decoder are reused as a feature extractor: the output of every
encoder stage and every decoder block is averaged over the annotated region,
the averages are concatenated, and a new linear head classifies the region.
The source segmentation head is not used.

The published checkpoint is a pickle-based Lightning file inside a zip archive.
It is converted once to safetensors under size and SHA-256 pins. The network
itself comes from ``segmentation_models_pytorch`` (the ``htc`` extra).
"""

from __future__ import annotations

import io
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file, save_file
from torch import nn

from fonderie_affinage import record as rec

MODEL_ID = "IMSY-DKFZ/htc image@2022-02-03_22-58-44_generated_default_model_comparison"
SOURCE_URL = (
    "https://e130-hyperspectal-tissue-classification.s3.dkfz.de/models/"
    "image%402022-02-03_22-58-44_generated_default_model_comparison.zip"
)
LICENCE = "MIT (htc repository LICENSE.md; models released as accompanying material)"
CONVERTED_NAME = "htc-image-2022-backbone.safetensors"
FOLD = "fold_P044,P050,P059"
ARCHIVE = {
    "name": "image@2022-02-03_22-58-44_generated_default_model_comparison.zip",
    "bytes": 1612525259,
    "sha256": "cada3c831483c654f8b4e4998f0cb0bd71e826fa84136e8a213f6f98ae955658",
}
CHECKPOINT = {
    "name": f"image/2022-02-03_22-58-44_generated_default_model_comparison/{FOLD}/"
    "epoch=70-dice_metric=0.90.ckpt",
    "sha256": "e4802ec52f3be7bceb8d19c94a935e7466c864bd60ec26dd7ee25c3c27d415ef",
}
KEY_PREFIX = "model.architecture."
SOURCE_HEAD_PREFIX = "segmentation_head."
FRAME_MULTIPLE = 32


class BaseModelError(ValueError):
    """A base-model file does not match its pin or the expected architecture."""


@dataclass(frozen=True)
class HTCConfig:
    encoder_name: str = "efficientnet-b5"
    bands: int = 100
    source_classes: int = 19
    decoder_channels: tuple[int, ...] = (256, 128, 64, 32, 16)

    def as_dict(self) -> dict:
        return {**asdict(self), "decoder_channels": list(self.decoder_channels)}


HTC_BASE = HTCConfig()
TINY_HTC = HTCConfig(
    encoder_name="resnet18", bands=12, source_classes=4, decoder_channels=(32, 24, 16, 8, 8)
)


def build_unet(config: HTCConfig) -> nn.Module:
    """The source network, without downloading any initial weights."""
    import segmentation_models_pytorch as smp

    return smp.Unet(
        encoder_name=config.encoder_name,
        encoder_weights=None,
        classes=config.source_classes,
        in_channels=config.bands,
        decoder_channels=config.decoder_channels,
    )


class HTCRegionClassifier(nn.Module):
    """Region classifier on region-pooled encoder and decoder features of the HTC U-Net."""

    family = "htc"
    input_mode = "frame"

    def __init__(self, config: HTCConfig, num_classes: int) -> None:
        super().__init__()
        self.config = config
        self.num_classes = num_classes
        unet = build_unet(config)
        self.encoder = unet.encoder
        self.decoder = unet.decoder
        self.feature_dim = sum(self.encoder.out_channels[1:]) + sum(config.decoder_channels)
        self.head = nn.Sequential(
            nn.LayerNorm(self.feature_dim), nn.Linear(self.feature_dim, num_classes)
        )

    def train(self, mode: bool = True) -> HTCRegionClassifier:
        """Training mode, except that batch-norm statistics of the base model stay fixed."""
        super().train(mode)
        for module in (*self.encoder.modules(), *self.decoder.modules()):
            if isinstance(module, nn.modules.batchnorm._BatchNorm):
                module.eval()
        return self

    @staticmethod
    def _pool(feature_map: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        """Average a (B, C, h, w) map with the region's coverage of each cell as weight."""
        cell = nn.functional.adaptive_avg_pool2d(weights[:, None], feature_map.shape[2:])
        total = cell.sum(dim=(2, 3), keepdim=True)
        uniform = torch.full_like(cell, 1.0 / (cell.shape[2] * cell.shape[3]))
        cell = torch.where(total > 0, cell / total.clamp_min(1e-12), uniform)
        return (feature_map * cell).sum(dim=(2, 3))

    def features(self, x: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        """``x``: (B, bands, H, W), L1-normalised; ``weights``: (B, H, W) region mask."""
        if x.shape[2] % FRAME_MULTIPLE or x.shape[3] % FRAME_MULTIPLE:
            raise ValueError(f"Frame height and width must be multiples of {FRAME_MULTIPLE}.")
        stages = self.encoder(x)
        pooled = [self._pool(stage, weights) for stage in stages[1:]]
        shapes = [stage.shape[2:] for stage in stages][::-1]
        skips = stages[1:][::-1]
        y = self.decoder.center(skips[0])
        for index, block in enumerate(self.decoder.blocks):
            height, width = shapes[index + 1]
            skip = skips[index + 1] if index + 1 < len(skips) else None
            y = block(y, height, width, skip_connection=skip)
            pooled.append(self._pool(y, weights))
        return torch.cat(pooled, dim=1)

    def forward(self, x: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        return self.head(self.features(x, weights))


def base_model_entry() -> dict[str, Any]:
    """The base-model block of a run record."""
    return {
        "id": MODEL_ID,
        "source": SOURCE_URL,
        "revision": f"archive sha256 {ARCHIVE['sha256']}, {FOLD}",
        "licence": LICENCE,
        "files": [dict(ARCHIVE), dict(CHECKPOINT)],
    }


def read_checkpoint(archive_path: Path) -> dict[str, torch.Tensor]:
    """Verify archive and checkpoint against their pins and return the network tensors."""
    archive_path = Path(archive_path)
    size = archive_path.stat().st_size
    if size != ARCHIVE["bytes"]:
        raise BaseModelError(f"{archive_path}: {size} bytes, pinned {ARCHIVE['bytes']}.")
    if rec.sha256_file(archive_path) != ARCHIVE["sha256"]:
        raise BaseModelError(f"{archive_path}: SHA-256 differs from the pin.")
    with zipfile.ZipFile(archive_path) as archive:
        raw = archive.read(CHECKPOINT["name"])
    if rec.sha256_bytes(raw) != CHECKPOINT["sha256"]:
        raise BaseModelError("Checkpoint inside the archive differs from the pin.")
    payload = torch.load(io.BytesIO(raw), map_location="cpu", weights_only=True)
    source = payload["state_dict"]
    other = sorted(name for name in source if not name.startswith(KEY_PREFIX))
    if other != ["ce_loss_weighted.weight"]:
        raise BaseModelError(f"Unexpected tensors outside the network: {other[:5]}")
    return {
        name[len(KEY_PREFIX) :]: tensor
        for name, tensor in source.items()
        if name.startswith(KEY_PREFIX)
    }


def convert(archive_path: Path, out_dir: Path, command: list[str]) -> Path:
    """Convert the pinned HTC checkpoint to safetensors; return the record path."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=False)
    record = rec.new_record("conversion", config={"target": CONVERTED_NAME}, command=command)
    record["base_model"] = base_model_entry()
    state = read_checkpoint(archive_path)
    build_unet(HTC_BASE).load_state_dict(state, strict=True)
    kept = {
        name: tensor.contiguous()
        for name, tensor in state.items()
        if not name.startswith(SOURCE_HEAD_PREFIX)
    }
    dropped = sorted(name for name in state if name not in kept)
    target = out_dir / CONVERTED_NAME
    save_file(
        kept,
        str(target),
        metadata=rec.weight_metadata(base_model=MODEL_ID, archive_sha256=ARCHIVE["sha256"]),
    )
    record["data"] = {"note": "No dataset is involved in a conversion."}
    record["conversion"] = {
        "load_method": "torch.load(weights_only=True) on the pinned checkpoint from the archive",
        "strict_load": "all network tensors loaded into the source U-Net before reduction",
        "tensors": len(kept),
        "parameters": int(sum(t.numel() for t in kept.values() if t.is_floating_point())),
        "dropped": dropped,
    }
    return rec.finish_record(record, out_dir, [target])


def load_converted(model: HTCRegionClassifier, converted_dir: Path) -> dict[str, Any]:
    """Load the converted backbone into a classifier and return the parent reference."""
    converted_dir = Path(converted_dir)
    record_path = converted_dir / rec.RECORD_NAME
    problems = rec.verify_outputs(record_path)
    if problems:
        raise BaseModelError(f"Converted base model failed verification: {problems}")
    state = load_file(str(converted_dir / CONVERTED_NAME))
    result = model.load_state_dict(state, strict=False)
    not_head = [name for name in result.missing_keys if not name.startswith("head.")]
    if not_head or result.unexpected_keys:
        raise BaseModelError(f"Missing {not_head[:5]}, unexpected {result.unexpected_keys[:5]}.")
    return rec.parent_reference(record_path)
