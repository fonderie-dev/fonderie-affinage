"""Run configuration: a TOML file under version control describes one run."""

from __future__ import annotations

import tomllib
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

from fonderie_affinage.hsi.data import Columns


class ConfigError(ValueError):
    """A configuration file has an unknown key or an invalid value."""


@dataclass(frozen=True)
class RunConfig:
    name: str
    output_root: str = "runs"
    seed: int = 0
    device: str = "cpu"
    amp: bool = False


@dataclass(frozen=True)
class DataConfig:
    manifest: str
    root: str = ""
    format: str = "tivita"
    flip_y: bool = True
    normalisation: str = "none"
    test_groups: tuple[str, ...] = ()
    validation_fraction: float = 0.25
    cache: str = "auto"
    columns: Columns = field(default_factory=Columns)


@dataclass(frozen=True)
class ModelConfig:
    base: str = "hypersigma"
    converted_dir: str = ""


@dataclass(frozen=True)
class MethodConfig:
    name: str = "lora"
    lora_rank: int = 8
    lora_alpha: float = 16.0
    last_blocks: int = 4


@dataclass(frozen=True)
class TrainingConfig:
    epochs: int = 30
    batch_size: int = 16
    learning_rate: float = 1e-3
    weight_decay: float = 0.01
    warmup_epochs: int = 2
    augment: bool = True
    class_weighting: str = "balanced"


@dataclass(frozen=True)
class Config:
    run: RunConfig
    data: DataConfig
    model: ModelConfig
    method: MethodConfig
    training: TrainingConfig
    path: Path

    def as_dict(self) -> dict[str, Any]:
        """The configuration as plain data for the run record (without the file path)."""
        plain = asdict(self)
        plain.pop("path")
        plain["data"]["test_groups"] = list(self.data.test_groups)
        return plain

    def resolve(self, value: str) -> Path:
        """A path from the configuration, relative to the configuration file."""
        path = Path(value).expanduser()
        return path if path.is_absolute() else self.path.parent / path


def _build(cls: type, table: dict[str, Any], section: str) -> Any:
    known = {item.name for item in fields(cls)}
    unknown = sorted(set(table) - known)
    if unknown:
        raise ConfigError(f"[{section}] has unknown keys: {', '.join(unknown)}")
    try:
        return cls(**table)
    except TypeError as error:
        raise ConfigError(f"[{section}]: {error}") from error


def load_config(path: Path) -> Config:
    path = Path(path).resolve()
    with path.open("rb") as handle:
        raw = tomllib.load(handle)
    unknown = sorted(set(raw) - {"run", "data", "model", "method", "training"})
    if unknown:
        raise ConfigError(f"Unknown sections: {', '.join(unknown)}")
    for section in ("run", "data"):
        if section not in raw:
            raise ConfigError(f"Section [{section}] is required.")
    data = dict(raw["data"])
    data["columns"] = _build(Columns, dict(data.get("columns", {})), "data.columns")
    data["test_groups"] = tuple(data.get("test_groups", ()))
    config = Config(
        run=_build(RunConfig, raw["run"], "run"),
        data=_build(DataConfig, data, "data"),
        model=_build(ModelConfig, raw.get("model", {}), "model"),
        method=_build(MethodConfig, raw.get("method", {}), "method"),
        training=_build(TrainingConfig, raw.get("training", {}), "training"),
        path=path,
    )
    _check(config)
    return config


BASES = ("hypersigma", "htc", "tiny", "tiny_htc")


def _check(config: Config) -> None:
    base = config.model.base
    if base not in BASES:
        raise ConfigError(f"model.base must be one of: {', '.join(BASES)}.")
    if base in ("hypersigma", "htc") and not config.model.converted_dir:
        raise ConfigError(f"model.converted_dir is required for the {base} base.")
    if base in ("htc", "tiny_htc"):
        if config.data.normalisation != "l1":
            raise ConfigError("The HTC base needs data.normalisation = 'l1'.")
        if config.method.name == "lora":
            raise ConfigError("method.name = 'lora' is not defined for the HTC base.")
    if config.data.cache not in ("auto", "memory", "none"):
        raise ConfigError("data.cache must be 'auto', 'memory' or 'none'.")
    if config.data.format not in ("tivita", "npy"):
        raise ConfigError("data.format must be 'tivita' or 'npy'.")
    if config.data.normalisation not in ("none", "l1"):
        raise ConfigError("data.normalisation must be 'none' or 'l1'.")
    if not 0.0 < config.data.validation_fraction < 1.0:
        raise ConfigError("data.validation_fraction must lie between 0 and 1.")
    if config.training.class_weighting not in ("balanced", "none"):
        raise ConfigError("training.class_weighting must be 'balanced' or 'none'.")
    if config.training.epochs < 1 or config.training.batch_size < 1:
        raise ConfigError("training.epochs and training.batch_size must be positive.")
    if config.run.device not in ("cpu", "cuda"):
        raise ConfigError("run.device must be 'cpu' or 'cuda'.")
