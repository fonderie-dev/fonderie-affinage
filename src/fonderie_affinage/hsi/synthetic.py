"""Synthetic hyperspectral cases for tests and dry runs. No real data is involved."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np


def write_synthetic_dataset(
    root: Path,
    *,
    groups: int = 6,
    classes: int = 3,
    bands: int = 12,
    height: int = 24,
    width: int = 32,
    seed: int = 0,
) -> Path:
    """Write one cube and mask per group and class, plus a manifest; return the manifest path.

    Each class has a smooth spectral signature; each group adds its own gain
    and offset, so a model has to learn the class rather than the group.
    """
    root = Path(root)
    (root / "cubes").mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    axis = np.linspace(0.0, 1.0, bands)
    signatures = np.stack(
        [
            0.5 + 0.3 * np.sin(2 * np.pi * (axis * (index + 1) / 2 + index / classes))
            for index in range(classes)
        ]
    )
    rows = []
    for group in range(groups):
        gain = 1.0 + 0.1 * rng.standard_normal()
        offset = 0.05 * rng.standard_normal()
        for label in range(classes):
            case = f"g{group:02d}_c{label}"
            background = 0.4 + 0.02 * rng.standard_normal((height, width, bands))
            mask = np.zeros((height, width), dtype=bool)
            top = int(rng.integers(1, height // 3))
            left = int(rng.integers(1, width // 3))
            mask[top : top + height // 2, left : left + width // 2] = True
            noise = 0.02 * rng.standard_normal((height, width, bands))
            cube = np.where(mask[..., None], gain * signatures[label] + offset + noise, background)
            np.save(root / "cubes" / f"{case}_cube.npy", cube.astype(np.float32))
            np.save(root / "cubes" / f"{case}_mask.npy", mask)
            rows.append(
                {
                    "case_id": case,
                    "label_name": f"class_{label}",
                    "subject_name": f"G{group:02d}",
                    "cube_path": f"cubes/{case}_cube.npy",
                    "mask_path": f"cubes/{case}_mask.npy",
                }
            )
    manifest = root / "manifest.csv"
    with manifest.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return manifest
