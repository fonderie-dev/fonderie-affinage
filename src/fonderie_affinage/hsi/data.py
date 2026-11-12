"""Hyperspectral cases: manifest, cube and mask readers, and the model input window.

A case is one annotated region in one cube: a label, a group (the animal or
patient it comes from) and the files that hold the cube and the region mask.
Cubes are float32 in HWC layout (rows, columns, bands).
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from fonderie_affinage.record import sha256_bytes

TIVITA_HEADER_BYTES = 12
TIVITA_STORAGE_SHAPE = (640, 480, 100)
TIVITA_WAVELENGTHS_NM = tuple(float(v) for v in range(500, 1000, 5))


class DataError(ValueError):
    """A manifest, cube or mask does not match the expected layout."""


@dataclass(frozen=True)
class Case:
    case_id: str
    label: str
    group: str
    cube_path: Path
    mask_path: Path


@dataclass(frozen=True)
class Columns:
    """Manifest column names. The defaults match a HeiPorSPECTRAL intake manifest."""

    case_id: str = "case_id"
    label: str = "label_name"
    group: str = "subject_name"
    cube: str = "cube_path"
    mask: str = "mask_path"


def read_manifest(
    path: Path, root: Path | None = None, columns: Columns | None = None
) -> list[Case]:
    """Read a CSV manifest. Relative file paths are resolved against ``root``."""
    columns = columns or Columns()
    root = Path(root) if root else Path(path).resolve().parent
    with Path(path).open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise DataError(f"Manifest has no rows: {path}")
    needed = (columns.case_id, columns.label, columns.group, columns.cube, columns.mask)
    missing = [name for name in needed if name not in rows[0]]
    if missing:
        raise DataError(f"Manifest lacks columns: {', '.join(missing)}")
    cases = []
    for row in rows:
        cases.append(
            Case(
                case_id=row[columns.case_id],
                label=row[columns.label],
                group=row[columns.group],
                cube_path=_resolve(row[columns.cube], root),
                mask_path=_resolve(row[columns.mask], root),
            )
        )
    ids = [case.case_id for case in cases]
    if len(set(ids)) != len(ids):
        raise DataError("Manifest case identifiers are not unique.")
    return sorted(cases, key=lambda case: case.case_id)


def _resolve(value: str, root: Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def decode_tivita(raw: bytes, flip_y: bool = True) -> np.ndarray:
    """Decode a TIVITA ``*_SpecCube.dat`` to float32 HWC (480, 640, 100).

    The file starts with three big-endian int32 (640, 480, 100), followed by
    big-endian float32 reflectance in (x, y, band) order. ``flip_y`` reverses
    the stored y axis, which is the orientation of the HeiPorSPECTRAL masks.
    """
    if len(raw) <= TIVITA_HEADER_BYTES:
        raise DataError("File too small to hold a SpecCube.")
    dims = tuple(int(v) for v in np.frombuffer(raw, dtype=">i4", count=3))
    if dims != TIVITA_STORAGE_SHAPE:
        raise DataError(f"Header declares {dims}, expected {TIVITA_STORAGE_SHAPE}.")
    if len(raw) != TIVITA_HEADER_BYTES + 4 * int(np.prod(dims)):
        raise DataError("Payload size does not match the header.")
    storage = np.frombuffer(raw, dtype=">f4", offset=TIVITA_HEADER_BYTES).reshape(dims)
    if flip_y:
        storage = storage[:, ::-1, :]
    return np.ascontiguousarray(np.transpose(storage, (1, 0, 2)), dtype=np.float32)


def read_cube(path: Path, cube_format: str, flip_y: bool = True) -> tuple[np.ndarray, str]:
    """Read one cube as float32 HWC and return it with the SHA-256 of the file."""
    raw = Path(path).read_bytes()
    digest = sha256_bytes(raw)
    if cube_format == "tivita":
        return decode_tivita(raw, flip_y=flip_y), digest
    if cube_format == "npy":
        cube = np.load(path, allow_pickle=False)
        if cube.ndim != 3:
            raise DataError(f"Expected an HWC cube, got shape {cube.shape}: {path}")
        return np.ascontiguousarray(cube, dtype=np.float32), digest
    raise DataError(f"Unknown cube format: {cube_format!r}")


def read_mask(path: Path) -> tuple[np.ndarray, str]:
    """Read a region mask (``.npy`` or an image; non-zero is inside) with its SHA-256."""
    path = Path(path)
    digest = sha256_bytes(path.read_bytes())
    if path.suffix == ".npy":
        mask = np.load(path, allow_pickle=False)
    else:
        from PIL import Image

        with Image.open(path) as image:
            mask = np.asarray(image)
        if mask.ndim == 3:
            mask = mask.max(axis=2)
    if mask.ndim != 2:
        raise DataError(f"Expected a 2-D mask, got shape {mask.shape}: {path}")
    return mask != 0, digest


def area_resample(values: np.ndarray, out_h: int, out_w: int) -> np.ndarray:
    """Exact box average of an (h, w) or (h, w, c) array onto an out_h by out_w grid."""
    rows = _overlap(values.shape[0], out_h)
    cols = _overlap(values.shape[1], out_w)
    if values.ndim == 2:
        return rows @ values.astype(np.float64) @ cols.T
    return np.einsum("ih,hwc,jw->ijc", rows, values.astype(np.float64), cols)


def _overlap(size_in: int, size_out: int) -> np.ndarray:
    """Row-stochastic matrix: the fraction of each output cell covered by each input cell."""
    edges = np.linspace(0.0, size_in, size_out + 1)
    lower = np.arange(size_in, dtype=np.float64)
    upper = lower + 1.0
    overlap = np.minimum(upper[None, :], edges[1:, None]) - np.maximum(
        lower[None, :], edges[:-1, None]
    )
    overlap = np.clip(overlap, 0.0, None)
    return overlap / overlap.sum(axis=1, keepdims=True)


def roi_window(mask: np.ndarray) -> tuple[int, int, int]:
    """Smallest square (top, left, side) that holds the whole region.

    The square may extend past the frame; :func:`prepare_window` pads it. No
    region pixel is ever cropped away.
    """
    rows = np.flatnonzero(mask.any(axis=1))
    cols = np.flatnonzero(mask.any(axis=0))
    if rows.size == 0:
        raise DataError("Region mask is empty.")
    height = int(rows[-1] - rows[0] + 1)
    width = int(cols[-1] - cols[0] + 1)
    side = max(height, width)
    top = int(rows[0]) - (side - height) // 2
    left = int(cols[0]) - (side - width) // 2
    return top, left, side


def prepare_window(
    cube: np.ndarray, mask: np.ndarray, size: int, normalisation: str = "none"
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Cut the region's square window and resample cube and mask to ``size``.

    Returns the window as float32 (size, size, bands), the fraction of each
    window cell covered by the region as float32 (size, size), and counts for
    the record. Parts of the square outside the frame are edge-padded and stay
    outside the mask. Non-finite values are replaced by the band median of the
    finite window pixels.
    """
    if cube.shape[:2] != mask.shape:
        raise DataError(f"Cube {cube.shape[:2]} and mask {mask.shape} differ in size.")
    top, left, side = roi_window(mask)
    height, width = mask.shape
    pad = (
        (max(0, -top), max(0, top + side - height)),
        (max(0, -left), max(0, left + side - width)),
    )
    row0, col0 = top + pad[0][0], left + pad[1][0]
    padded_cube = np.pad(cube, (*pad, (0, 0)), mode="edge")
    padded_mask = np.pad(mask, pad, mode="constant", constant_values=False)
    window = padded_cube[row0 : row0 + side, col0 : col0 + side, :].astype(np.float64)
    region = padded_mask[row0 : row0 + side, col0 : col0 + side]

    bad = ~np.isfinite(window)
    replaced = int(bad.sum())
    if replaced:
        if bad.all(axis=(0, 1)).any():
            raise DataError("A band has no finite value inside the window.")
        medians = np.nanmedian(np.where(bad, np.nan, window), axis=(0, 1))
        window = np.where(bad, medians[None, None, :], window)
    if normalisation == "l1":
        totals = np.abs(window).sum(axis=2, keepdims=True)
        window = window / np.where(totals > 0, totals, 1.0)
    elif normalisation != "none":
        raise DataError(f"Unknown normalisation: {normalisation!r}")

    info = {
        "window_top": top,
        "window_left": left,
        "window_side": side,
        "region_pixels": int(mask.sum()),
        "region_pixels_in_window": int(region.sum()),
        "padded_pixels": int(side * side - (side - sum(pad[0])) * (side - sum(pad[1]))),
        "nonfinite_replaced": replaced,
    }
    resampled = area_resample(window, size, size).astype(np.float32)
    weights = area_resample(region.astype(np.float64), size, size).astype(np.float32)
    return resampled, weights, info


def prepare_frame(
    cube: np.ndarray, mask: np.ndarray, normalisation: str = "l1"
) -> tuple[np.ndarray, np.ndarray, dict]:
    """The whole frame as model input, with the region mask as weights.

    With ``l1`` every pixel spectrum is divided by the sum of its absolute
    values and non-finite results become zero, as the HTC models expect.
    """
    if cube.shape[:2] != mask.shape:
        raise DataError(f"Cube {cube.shape[:2]} and mask {mask.shape} differ in size.")
    if not mask.any():
        raise DataError("Region mask is empty.")
    frame = cube.astype(np.float32)
    replaced = int((~np.isfinite(frame)).sum())
    if normalisation == "l1":
        with np.errstate(divide="ignore", invalid="ignore"):
            frame = frame / np.abs(frame).sum(axis=2, keepdims=True)
    elif normalisation != "none":
        raise DataError(f"Unknown normalisation: {normalisation!r}")
    frame = np.nan_to_num(frame, nan=0.0, posinf=0.0, neginf=0.0)
    info = {
        "frame_height": int(mask.shape[0]),
        "frame_width": int(mask.shape[1]),
        "region_pixels": int(mask.sum()),
        "nonfinite_replaced": replaced,
    }
    return frame, mask.astype(np.float32), info
