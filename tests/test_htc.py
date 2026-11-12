import json

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("segmentation_models_pytorch")

from safetensors.torch import load_file  # noqa: E402

from fonderie_affinage import record as rec  # noqa: E402
from fonderie_affinage.config import ConfigError, load_config  # noqa: E402
from fonderie_affinage.export import MODEL_NAME, export_run, load_run_model  # noqa: E402
from fonderie_affinage.hsi.data import prepare_frame  # noqa: E402
from fonderie_affinage.hsi.synthetic import write_synthetic_dataset  # noqa: E402
from fonderie_affinage.models.adaptation import apply_method  # noqa: E402
from fonderie_affinage.models.htc import TINY_HTC, HTCRegionClassifier  # noqa: E402
from fonderie_affinage.train import ADAPTER_NAME, run_training  # noqa: E402

CONFIG = """
[run]
name = "test-htc-{method}"
output_root = "runs"
seed = 5

[data]
manifest = "data/manifest.csv"
format = "npy"
normalisation = "{normalisation}"
test_groups = ["G05"]
validation_fraction = 0.4

[model]
base = "tiny_htc"

[method]
name = "{method}"
last_blocks = 2

[training]
epochs = {epochs}
batch_size = 6
learning_rate = 0.01
augment = {augment}
"""


def make_config(tmp_path, method="head", epochs=30, augment="false", normalisation="l1"):
    if not (tmp_path / "data").exists():
        write_synthetic_dataset(tmp_path / "data", height=64, width=96)
    path = tmp_path / f"{method}.toml"
    text = CONFIG.format(method=method, epochs=epochs, augment=augment, normalisation=normalisation)
    path.write_text(text, encoding="utf-8")
    return load_config(path)


def test_frame_preparation_normalises_each_pixel() -> None:
    import numpy as np

    cube = np.random.default_rng(0).random((32, 64, 12)).astype(np.float32) + 0.1
    cube[0, 0, :] = 0.0
    mask = np.zeros((32, 64), dtype=bool)
    mask[4:20, 8:40] = True
    frame, weights, info = prepare_frame(cube, mask, "l1")
    np.testing.assert_allclose(frame[1:, :, :].sum(axis=2), 1.0, rtol=1e-5)
    assert (frame[0, 0] == 0).all()
    assert weights.sum() == info["region_pixels"] == 16 * 32


def test_features_pool_every_stage_over_the_region() -> None:
    torch.manual_seed(0)
    model = HTCRegionClassifier(TINY_HTC, num_classes=3).eval()
    x = torch.rand(2, TINY_HTC.bands, 64, 96)
    weights = torch.zeros(2, 64, 96)
    weights[:, 8:40, 16:64] = 1.0
    features = model.features(x, weights)
    assert features.shape == (2, model.feature_dim)
    assert model.feature_dim == sum(model.encoder.out_channels[1:]) + sum(TINY_HTC.decoder_channels)
    assert model(x, weights).shape == (2, 3)

    outside = x.clone()
    outside[:, :, 48:, 80:] += 5.0  # far from the region: changes context, not the mask
    assert not torch.allclose(model.features(outside, weights)[:, -8:], features[:, -8:])
    with pytest.raises(ValueError, match="multiples of 32"):
        model.features(torch.rand(1, TINY_HTC.bands, 60, 96), torch.ones(1, 60, 96))


def test_batch_statistics_stay_fixed_in_training_mode() -> None:
    model = HTCRegionClassifier(TINY_HTC, num_classes=3)
    before = {k: v.clone() for k, v in model.state_dict().items() if "running_mean" in k}
    model.train()
    model(torch.rand(2, TINY_HTC.bands, 64, 96), torch.ones(2, 64, 96))
    after = model.state_dict()
    assert before and all(torch.equal(value, after[name]) for name, value in before.items())


def test_methods_for_the_htc_family() -> None:
    model = HTCRegionClassifier(TINY_HTC, num_classes=3)
    assert all(name.startswith("head.") for name in apply_method(model, "head"))
    names = apply_method(model, "last_blocks", last_blocks=2)
    assert any(name.startswith("decoder.blocks.4.") for name in names)
    assert any(name.startswith("decoder.blocks.3.") for name in names)
    assert not any(name.startswith(("encoder.", "decoder.blocks.0.")) for name in names)
    with pytest.raises(ValueError, match="not defined"):
        apply_method(model, "lora")


def test_head_training_caches_features_and_exports(tmp_path) -> None:
    run_dir = run_training(make_config(tmp_path), ["affinage", "train"])
    record = json.loads((run_dir / rec.RECORD_NAME).read_text(encoding="utf-8"))
    assert rec.verify_outputs(run_dir / rec.RECORD_NAME) == []
    assert record["method"]["base_features_cached"] is True
    assert record["data"]["preprocessing"]["held_in_memory"] is False
    assert record["data"]["preprocessing"]["normalisation"] == "l1"
    # The base model is random and frozen here, so only "clearly above chance" is expected.
    assert record["selection"]["best_value"] > 0.5
    assert all(name.startswith("head.") for name in load_file(str(run_dir / ADAPTER_NAME)))

    export_record = export_run(run_dir, tmp_path / "export", ["affinage", "export"])
    assert rec.verify_outputs(export_record) == []
    state = load_file(str(tmp_path / "export" / MODEL_NAME))
    merged, _, labels = load_run_model(run_dir)
    plain = HTCRegionClassifier(TINY_HTC, num_classes=len(labels)).eval()
    plain.load_state_dict(state, strict=True)
    x, weights = torch.rand(1, TINY_HTC.bands, 64, 96), torch.ones(1, 64, 96)
    torch.testing.assert_close(plain(x, weights), merged(x, weights))


def test_last_blocks_training_reads_frames_on_demand(tmp_path) -> None:
    config = make_config(tmp_path, method="last_blocks", epochs=2, augment="true")
    run_dir = run_training(config, [])
    record = json.loads((run_dir / rec.RECORD_NAME).read_text(encoding="utf-8"))
    assert record["method"]["base_features_cached"] is False
    adapter = load_file(str(run_dir / ADAPTER_NAME))
    assert any(name.startswith("decoder.blocks.4.") for name in adapter)
    assert not any("running_" in name or name.startswith("encoder.") for name in adapter)


def test_same_seed_gives_identical_weights(tmp_path) -> None:
    first = run_training(make_config(tmp_path / "a", epochs=3), [])
    second = run_training(make_config(tmp_path / "b", epochs=3), [])
    assert rec.sha256_file(first / ADAPTER_NAME) == rec.sha256_file(second / ADAPTER_NAME)


def test_config_rules_for_the_htc_base(tmp_path) -> None:
    with pytest.raises(ConfigError, match="l1"):
        make_config(tmp_path, normalisation="none")
    with pytest.raises(ConfigError, match="lora"):
        make_config(tmp_path, method="lora")
