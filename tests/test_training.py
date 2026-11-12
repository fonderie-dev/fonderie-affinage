import json

import pytest

torch = pytest.importorskip("torch")

from safetensors.torch import load_file  # noqa: E402

from fonderie_affinage import record as rec  # noqa: E402
from fonderie_affinage.config import ConfigError, load_config  # noqa: E402
from fonderie_affinage.export import MODEL_NAME, export_run, load_run_model  # noqa: E402
from fonderie_affinage.hsi.synthetic import write_synthetic_dataset  # noqa: E402
from fonderie_affinage.models.adaptation import apply_method, merge_adapters  # noqa: E402
from fonderie_affinage.models.encoders import TINY, DualEncoderClassifier  # noqa: E402
from fonderie_affinage.train import ADAPTER_NAME, run_training  # noqa: E402

CONFIG = """
[run]
name = "test-{method}"
output_root = "runs"
seed = 11

[data]
manifest = "data/manifest.csv"
format = "npy"
test_groups = ["G05"]
validation_fraction = 0.4

[model]
base = "tiny"

[method]
name = "{method}"
lora_rank = 4
lora_alpha = 8.0
last_blocks = 1

[training]
epochs = {epochs}
batch_size = 6
learning_rate = 0.01
"""


def make_config(tmp_path, method: str = "lora", epochs: int = 15):
    if not (tmp_path / "data").exists():
        write_synthetic_dataset(tmp_path / "data")
    path = tmp_path / f"{method}.toml"
    path.write_text(CONFIG.format(method=method, epochs=epochs), encoding="utf-8")
    return load_config(path)


def test_encoder_shapes_and_region_pooling() -> None:
    torch.manual_seed(0)
    model = DualEncoderClassifier(TINY, num_classes=3).eval()
    x = torch.randn(2, TINY.bands, TINY.image_size, TINY.image_size)
    weights = torch.zeros(2, TINY.image_size, TINY.image_size)
    weights[:, :8, :8] = 1.0
    assert model.spatial(x).shape == (2, TINY.grid**2, TINY.embed_dim)
    assert model.spectral(x).shape == (2, TINY.bands, TINY.embed_dim)
    assert model(x, weights).shape == (2, 3)
    assert torch.isfinite(model(x, torch.zeros_like(weights))).all()


@pytest.mark.parametrize(
    ("method", "expected"),
    [("head", "head."), ("lora", "lora_"), ("last_blocks", "blocks.1.")],
)
def test_methods_train_only_what_they_name(method: str, expected: str) -> None:
    model = DualEncoderClassifier(TINY, num_classes=3)
    names = apply_method(model, method, lora_rank=2, lora_alpha=4.0, last_blocks=1)
    assert any(expected in name for name in names)
    assert all(name.startswith("head.") or expected in name or ".norm." in name for name in names)
    assert not any("blocks.0." in name for name in names)


def test_merged_adapters_give_the_same_output() -> None:
    torch.manual_seed(1)
    model = DualEncoderClassifier(TINY, num_classes=3).eval()
    apply_method(model, "lora", lora_rank=2, lora_alpha=4.0, last_blocks=2)
    for name, parameter in model.named_parameters():
        if "lora_b" in name:
            torch.nn.init.normal_(parameter, std=0.1)
    x = torch.randn(3, TINY.bands, TINY.image_size, TINY.image_size)
    weights = torch.ones(3, TINY.image_size, TINY.image_size)
    before = model(x, weights)
    assert merge_adapters(model) == 8
    assert not any("lora_" in name for name in model.state_dict())
    torch.testing.assert_close(model(x, weights), before, atol=1e-5, rtol=1e-5)


def test_training_learns_records_and_holds_out(tmp_path) -> None:
    config = make_config(tmp_path)
    run_dir = run_training(config, ["affinage", "train", "lora.toml"])
    record = json.loads((run_dir / rec.RECORD_NAME).read_text(encoding="utf-8"))

    assert rec.verify_outputs(run_dir / rec.RECORD_NAME) == []
    assert record["kind"] == "training"
    assert record["selection"]["best_value"] >= 0.8
    assert record["selection"]["test_metric"] == "not computed in training"
    assert record["split"]["test_groups"] == ["G05"]
    assert len(record["data"]["files"]) == 2 * record["data"]["cases"]
    assert all(len(entry["sha256"]) == 64 for entry in record["data"]["files"])
    assert not any(entry["path"].startswith("/") for entry in record["data"]["files"])
    assert record["method"]["trainable_parameters"] < record["method"]["total_parameters"]

    split = json.loads((run_dir / "split.json").read_text(encoding="utf-8"))
    assert not set(split["test"]) & (set(split["train"]) | set(split["validation"]))
    held_out = (run_dir / "predictions_test.csv").read_text(encoding="utf-8").splitlines()
    assert len(held_out) == 1 + len(split["test"])
    adapter = load_file(str(run_dir / ADAPTER_NAME))
    assert all(name.startswith("head.") or "lora_" in name for name in adapter)


def test_same_seed_gives_identical_weights(tmp_path) -> None:
    first = run_training(make_config(tmp_path / "a", epochs=3), [])
    second = run_training(make_config(tmp_path / "b", epochs=3), [])
    assert rec.sha256_file(first / ADAPTER_NAME) == rec.sha256_file(second / ADAPTER_NAME)


def test_export_merges_and_reproduces_the_run(tmp_path) -> None:
    config = make_config(tmp_path, epochs=4)
    run_dir = run_training(config, [])
    export_record = export_run(run_dir, tmp_path / "export", ["affinage", "export"])
    exported = json.loads(export_record.read_text(encoding="utf-8"))
    training = json.loads((run_dir / rec.RECORD_NAME).read_text(encoding="utf-8"))

    assert exported["parents"][0]["record_id"] == training["record_id"]
    assert rec.verify_outputs(export_record) == []
    state = load_file(str(tmp_path / "export" / MODEL_NAME))
    assert not any("lora_" in name for name in state)
    card = (tmp_path / "export" / "MODEL_CARD.md").read_text(encoding="utf-8")
    assert training["record_id"] in card
    assert "not a test result" in card

    merged, _, labels = load_run_model(run_dir)
    plain = DualEncoderClassifier(TINY, num_classes=len(labels)).eval()
    plain.load_state_dict(state, strict=True)
    x = torch.randn(2, TINY.bands, TINY.image_size, TINY.image_size)
    weights = torch.ones(2, TINY.image_size, TINY.image_size)
    torch.testing.assert_close(plain(x, weights), merged(x, weights))


def test_export_refuses_a_tampered_run(tmp_path) -> None:
    run_dir = run_training(make_config(tmp_path, method="head", epochs=2), [])
    (run_dir / "labels.json").write_text("[]\n", encoding="utf-8")
    with pytest.raises(rec.RecordError, match="verification"):
        export_run(run_dir, tmp_path / "export", [])


def test_config_rejects_unknown_keys(tmp_path) -> None:
    path = tmp_path / "bad.toml"
    path.write_text('[run]\nname = "x"\ntypo = 1\n[data]\nmanifest = "m.csv"\n', encoding="utf-8")
    with pytest.raises(ConfigError, match="typo"):
        load_config(path)
