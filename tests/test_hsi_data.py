import numpy as np
import pytest

from fonderie_affinage.hsi import data
from fonderie_affinage.hsi.splits import grouped_split
from fonderie_affinage.hsi.synthetic import write_synthetic_dataset


def tivita_bytes(storage: np.ndarray) -> bytes:
    header = np.asarray(storage.shape, dtype=">i4").tobytes()
    return header + storage.astype(">f4").tobytes()


def test_tivita_decoding_and_orientation() -> None:
    storage = np.zeros(data.TIVITA_STORAGE_SHAPE, dtype=np.float32)
    storage[3, 5, 7] = 1.0  # x=3, y=5, band=7
    raw = tivita_bytes(storage)
    plain = data.decode_tivita(raw, flip_y=False)
    assert plain.shape == (480, 640, 100)
    assert plain[5, 3, 7] == 1.0
    flipped = data.decode_tivita(raw, flip_y=True)
    assert flipped[480 - 1 - 5, 3, 7] == 1.0


def test_tivita_rejects_wrong_header_and_size() -> None:
    with pytest.raises(data.DataError, match="Header"):
        data.decode_tivita(np.asarray((1, 2, 3), dtype=">i4").tobytes() + b"\0" * 24)
    good = tivita_bytes(np.zeros(data.TIVITA_STORAGE_SHAPE, dtype=np.float32))
    with pytest.raises(data.DataError, match="Payload"):
        data.decode_tivita(good[:-4])


def test_area_resample_preserves_means_and_coverage() -> None:
    rng = np.random.default_rng(0)
    values = rng.random((12, 20, 3))
    out = data.area_resample(values, 4, 5)
    assert out.shape == (4, 5, 3)
    np.testing.assert_allclose(out.mean(axis=(0, 1)), values.mean(axis=(0, 1)))
    mask = np.zeros((8, 8))
    mask[:4, :] = 1.0
    np.testing.assert_allclose(data.area_resample(mask, 2, 2), [[1.0, 1.0], [0.0, 0.0]])


def test_window_keeps_every_region_pixel_and_pads_outside_the_frame() -> None:
    cube = np.random.default_rng(1).random((10, 30, 4)).astype(np.float32)
    mask = np.zeros((10, 30), dtype=bool)
    mask[2:9, 1:25] = True  # 7 rows by 24 columns: the square leaves the frame
    window, weights, info = data.prepare_window(cube, mask, size=8)
    assert window.shape == (8, 8, 4)
    assert weights.shape == (8, 8)
    assert info["window_side"] == 24
    assert info["region_pixels_in_window"] == info["region_pixels"] == 7 * 24
    assert info["padded_pixels"] > 0
    np.testing.assert_allclose(weights.sum() * (24 / 8) ** 2, 7 * 24, rtol=1e-5)


def test_window_replaces_nonfinite_values_and_normalises() -> None:
    cube = np.ones((6, 6, 5), dtype=np.float32)
    cube[2, 2, 0] = np.nan
    mask = np.ones((6, 6), dtype=bool)
    window, _, info = data.prepare_window(cube, mask, size=3, normalisation="l1")
    assert info["nonfinite_replaced"] == 1
    np.testing.assert_allclose(window.sum(axis=2), 1.0, rtol=1e-6)


def test_manifest_and_grouped_split(tmp_path) -> None:
    manifest = write_synthetic_dataset(tmp_path, groups=6, classes=3)
    cases = data.read_manifest(manifest)
    assert len(cases) == 18
    split = grouped_split(cases, ["G05"], validation_fraction=0.4, seed=3)
    assert split.test_groups == ("G05",)
    assert len(split.validation_groups) == 2
    assert not set(split.train_groups) & set(split.validation_groups)
    assert not set(split.train) & set(split.validation) | set(split.train) & set(split.test)
    assert split == grouped_split(cases, ["G05"], validation_fraction=0.4, seed=3)
    with pytest.raises(data.DataError, match="not in the manifest"):
        grouped_split(cases, ["nobody"], validation_fraction=0.4, seed=3)


def test_manifest_rejects_duplicate_cases(tmp_path) -> None:
    manifest = write_synthetic_dataset(tmp_path, groups=2, classes=2)
    lines = manifest.read_text(encoding="utf-8").splitlines()
    manifest.write_text("\n".join([*lines, lines[1]]) + "\n", encoding="utf-8")
    with pytest.raises(data.DataError, match="not unique"):
        data.read_manifest(manifest)
