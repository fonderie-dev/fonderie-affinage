"""HyperSIGMA base encoders: pinned source files and conversion to safetensors.

HyperSIGMA (WHU-Sigma) publishes its pretrained spatial and spectral ViT-B
encoders as pickle-based ``.pth`` training snapshots. Fonderie loads weights
from safetensors only, so the snapshots are converted once: each file is
checked against the pinned size and SHA-256, opened with
``torch.load(weights_only=True)``, reduced to the encoder tensors and written
to one safetensors file with a conversion record.

Dropped on purpose: the masked-autoencoder decoder, the mask token, the
position embedding (the pretraining encoder never adds it) and the spatial
``fpn1`` layers (never used by the encoder path).
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file, save_file

from fonderie_affinage import record as rec
from fonderie_affinage.models.encoders import (
    HYPERSIGMA_BASE,
    DualEncoderClassifier,
    SpatialEncoder,
    SpectralEncoder,
)

MODEL_ID = "WHU-Sigma/HyperSIGMA"
SOURCE_URL = "https://huggingface.co/WHU-Sigma/HyperSIGMA"
REVISION = "e0567395fbdfddbae994695baf5fc73358a1ec3c"
LICENCE = "Apache-2.0 (Hugging Face model card metadata and repository LICENSE)"
CONVERTED_NAME = "hypersigma-base-encoders.safetensors"

SOURCE_FILES = {
    "spatial": {
        "name": "spat-vit-base-ultra-checkpoint-1599.pth",
        "bytes": 1436444759,
        "sha256": "df6584e8124a9916687a21193bad0516439c97ed712ba7a854e1843d3d5fabc7",
    },
    "spectral": {
        "name": "spec-vit-base-ultra-checkpoint-1599.pth",
        "bytes": 1391770843,
        "sha256": "662845ed9669c9fc8f39b18b19a10dff61792a9c8700b6bfea439815f7c86d6f",
    },
}
ENCODER_PREFIXES = ("patch_embed.", "spat_map.", "blocks.", "norm.")


class BaseModelError(ValueError):
    """A base-model file does not match its pin or the expected architecture."""


def base_model_entry() -> dict[str, Any]:
    """The base-model block of a run record."""
    return {
        "id": MODEL_ID,
        "source": SOURCE_URL,
        "revision": REVISION,
        "licence": LICENCE,
        "files": [dict(entry) for entry in SOURCE_FILES.values()],
    }


def verify_source(path: Path, branch: str) -> None:
    pin = SOURCE_FILES[branch]
    size = Path(path).stat().st_size
    if size != pin["bytes"]:
        raise BaseModelError(f"{path}: {size} bytes, pinned {pin['bytes']}.")
    digest = rec.sha256_file(path)
    if digest != pin["sha256"]:
        raise BaseModelError(f"{path}: SHA-256 {digest} differs from the pin.")


def read_snapshot(path: Path) -> dict[str, torch.Tensor]:
    """Model tensors of a pretraining snapshot, read without executing pickled code."""
    with torch.serialization.safe_globals([argparse.Namespace]):
        snapshot = torch.load(str(path), map_location="cpu", weights_only=True, mmap=True)
    state = snapshot.get("model", snapshot)
    return {name: tensor for name, tensor in state.items() if isinstance(tensor, torch.Tensor)}


def encoder_state(state: dict[str, torch.Tensor], branch: str) -> tuple[dict, list[str]]:
    """Keep the encoder tensors, check them against the architecture, list what was dropped."""
    kept = {name: tensor for name, tensor in state.items() if name.startswith(ENCODER_PREFIXES)}
    dropped = sorted(name for name in state if name not in kept)
    module = (SpatialEncoder if branch == "spatial" else SpectralEncoder)(HYPERSIGMA_BASE)
    expected = module.state_dict()
    if set(kept) != set(expected):
        missing = sorted(set(expected) - set(kept))
        extra = sorted(set(kept) - set(expected))
        raise BaseModelError(f"{branch}: missing {missing[:5]}, unexpected {extra[:5]}.")
    for name, tensor in kept.items():
        if tensor.shape != expected[name].shape:
            raise BaseModelError(f"{branch}: {name} has shape {tuple(tensor.shape)}.")
    module.load_state_dict(kept, strict=True)
    return kept, dropped


def convert(spatial_path: Path, spectral_path: Path, out_dir: Path, command: list[str]) -> Path:
    """Convert the two pinned snapshots to one safetensors file; return the record path."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=False)
    record = rec.new_record("conversion", config={"target": CONVERTED_NAME}, command=command)
    record["base_model"] = base_model_entry()
    tensors: dict[str, torch.Tensor] = {}
    dropped: dict[str, list[str]] = {}
    for branch, path in (("spatial", spatial_path), ("spectral", spectral_path)):
        verify_source(path, branch)
        kept, dropped[branch] = encoder_state(read_snapshot(path), branch)
        for name, tensor in kept.items():
            tensors[f"{branch}.{name}"] = tensor.to(torch.float32).contiguous()
        del kept
    target = out_dir / CONVERTED_NAME
    save_file(
        tensors,
        str(target),
        metadata=rec.weight_metadata(base_model=MODEL_ID, revision=REVISION),
    )
    record["data"] = {"note": "No dataset is involved in a conversion."}
    record["conversion"] = {
        "load_method": "torch.load(weights_only=True) with argparse.Namespace allow-listed",
        "tensors": len(tensors),
        "parameters": int(sum(tensor.numel() for tensor in tensors.values())),
        "dropped": dropped,
    }
    return rec.finish_record(record, out_dir, [target])


def load_converted(model: DualEncoderClassifier, converted_dir: Path) -> dict[str, Any]:
    """Load converted encoders into a classifier and return the parent reference.

    The converted file is checked against the conversion record before use.
    """
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
