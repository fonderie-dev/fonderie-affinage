"""Customise a base model on hyperspectral region labels and record the run.

The test part of the split is never used for training or selection. Its
predictions are written out, but no test metric is computed here: judging a
model is the task of an evaluation with its own record.
"""

from __future__ import annotations

import csv
import json
import random
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import torch
from safetensors.torch import load_file, save_file

from fonderie_affinage import record as rec
from fonderie_affinage.config import Config
from fonderie_affinage.hsi.data import (
    Case,
    DataError,
    prepare_frame,
    prepare_window,
    read_cube,
    read_manifest,
    read_mask,
)
from fonderie_affinage.hsi.splits import Split, grouped_split
from fonderie_affinage.models.adaptation import apply_method

ADAPTER_NAME = "adapter.safetensors"
TINY_BASE_NAME = "base.safetensors"


def seed_everything(seed: int) -> dict[str, Any]:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.benchmark = False
    return {
        "python": seed,
        "numpy": seed,
        "torch": seed,
        "data_order": seed,
        "split": seed,
        "deterministic_algorithms": "requested, warn_only",
    }


def build_model(
    config: Config, num_classes: int, tiny_base: Path | None = None
) -> tuple[torch.nn.Module, dict[str, Any], list[dict[str, str]]]:
    """Build the classifier on its base model; return it with base entry and parent records."""
    torch.manual_seed(config.run.seed)
    base = config.model.base
    if base in ("hypersigma", "tiny"):
        from fonderie_affinage.models import hypersigma
        from fonderie_affinage.models.encoders import HYPERSIGMA_BASE, TINY, DualEncoderClassifier

        model = DualEncoderClassifier(
            HYPERSIGMA_BASE if base == "hypersigma" else TINY, num_classes
        )
        family = hypersigma
    else:
        from fonderie_affinage.models import htc
        from fonderie_affinage.models.htc import HTC_BASE, TINY_HTC, HTCRegionClassifier

        model = HTCRegionClassifier(HTC_BASE if base == "htc" else TINY_HTC, num_classes)
        family = htc
    if base.startswith("tiny"):
        if tiny_base is not None:
            model.load_state_dict(load_file(str(tiny_base)), strict=True)
        entry = {
            "id": f"synthetic-{base}",
            "source": "randomly initialised in this run; stored as " + TINY_BASE_NAME,
            "revision": None,
            "licence": "none (synthetic)",
            "files": [],
        }
        return model, entry, []
    converted = config.resolve(config.model.converted_dir)
    parent = family.load_converted(model, converted)
    entry = family.base_model_entry()
    entry["local_path"] = str(converted)
    return model, entry, [parent]


class CaseStore:
    """Model inputs of all cases, held in memory or re-read from disk on demand.

    Every cube and mask is read and checksummed once when the store is built.
    Window inputs are small and kept in memory. Whole frames are large, so by
    default they are read again whenever a batch needs them.
    """

    def __init__(self, config: Config, cases: list[Case], model: torch.nn.Module, root: Path):
        self.config = config
        self.cases = cases
        self.mode = model.input_mode
        self.bands = model.config.bands
        self.size = getattr(model.config, "image_size", None)
        cache = config.data.cache
        self.in_memory = cache == "memory" or (cache == "auto" and self.mode == "window")
        self.files: list[dict] = []
        self.summary: dict[str, Any] = {"nonfinite_replaced": 0, "padded_pixels": 0}
        self._items: list[tuple[torch.Tensor, torch.Tensor]] = []
        for case in cases:
            item, digests, info = self._read(case)
            for path, digest in zip((case.cube_path, case.mask_path), digests, strict=True):
                entry = {"path": _relative(path, root), "bytes": path.stat().st_size}
                self.files.append({**entry, "sha256": digest})
            self.summary["nonfinite_replaced"] += info["nonfinite_replaced"]
            self.summary["padded_pixels"] += info.get("padded_pixels", 0)
            if self.in_memory:
                self._items.append(item)

    def _read(self, case: Case) -> tuple[tuple[torch.Tensor, torch.Tensor], tuple[str, str], dict]:
        cube, cube_sha = read_cube(case.cube_path, self.config.data.format, self.config.data.flip_y)
        mask, mask_sha = read_mask(case.mask_path)
        if cube.shape[2] != self.bands:
            raise DataError(
                f"{case.case_id}: {cube.shape[2]} bands, the model expects {self.bands}."
            )
        if self.mode == "window":
            x, w, info = prepare_window(cube, mask, self.size, self.config.data.normalisation)
        else:
            x, w, info = prepare_frame(cube, mask, self.config.data.normalisation)
        item = (torch.from_numpy(x).permute(2, 0, 1).contiguous(), torch.from_numpy(w))
        return item, (cube_sha, mask_sha), info

    def batch(self, indices: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        items = [
            self._items[i] if self.in_memory else self._read(self.cases[i])[0]
            for i in indices.tolist()
        ]
        return torch.stack([x for x, _ in items]), torch.stack([w for _, w in items])

    def preprocessing(self) -> dict[str, Any]:
        if self.mode == "window":
            layout = {
                "input": "smallest square holding the whole region, edge-padded outside the frame",
                "resampling": f"exact area averaging to {self.size} x {self.size}",
            }
        else:
            layout = {"input": "whole frame; the region mask weights the feature pooling"}
        return {
            **layout,
            "normalisation": self.config.data.normalisation,
            "format": self.config.data.format,
            "flip_y": self.config.data.flip_y,
            "held_in_memory": self.in_memory,
            **self.summary,
        }


def _relative(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return path.name


def balanced_accuracy(labels: torch.Tensor, predictions: torch.Tensor) -> float:
    """Mean recall over the classes that occur in ``labels``."""
    recalls = []
    for label in labels.unique():
        selected = labels == label
        recalls.append((predictions[selected] == label).float().mean().item())
    return float(sum(recalls) / len(recalls))


def augment(
    x: torch.Tensor, w: torch.Tensor, generator: torch.Generator
) -> tuple[torch.Tensor, torch.Tensor]:
    """One random flip or transposition per batch, applied to inputs and weights alike."""
    choice = torch.randint(0, 8, (1,), generator=generator).item()
    if choice & 1:
        x, w = x.flip(-1), w.flip(-1)
    if choice & 2:
        x, w = x.flip(-2), w.flip(-2)
    if choice & 4:
        x, w = x.transpose(-1, -2), w.transpose(-1, -2)
    return x, w


def trainable_state(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: parameter.detach().cpu().clone().contiguous()
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }


class Runner:
    """Forward passes for one run: through the whole model, or from cached features.

    When only the head is trained and no augmentation is used, the base model's
    output for a case never changes, so it is computed once and reused.
    """

    def __init__(self, config: Config, model: torch.nn.Module, store: CaseStore, device: str):
        self.config, self.model, self.store, self.device = config, model, store, device
        self.batch_size = config.training.batch_size
        self.cached: torch.Tensor | None = None
        if config.method.name == "head" and not config.training.augment:
            self.cached = self._all_features()

    @torch.no_grad()
    def _all_features(self) -> torch.Tensor:
        self.model.eval()
        outputs = []
        for start in range(0, len(self.store.cases), self.batch_size):
            index = torch.arange(start, min(start + self.batch_size, len(self.store.cases)))
            x, w = self.store.batch(index)
            outputs.append(self.model.features(x.to(self.device), w.to(self.device)).float().cpu())
        return torch.cat(outputs)

    def logits(self, index: torch.Tensor, generator: torch.Generator | None = None) -> torch.Tensor:
        if self.cached is not None:
            return self.model.head(self.cached[index].to(self.device))
        x, w = self.store.batch(index)
        if generator is not None and self.config.training.augment:
            x, w = augment(x, w, generator)
        enabled = self.config.run.amp
        with torch.autocast(device_type=self.device, dtype=torch.bfloat16, enabled=enabled):
            return self.model(x.to(self.device), w.to(self.device))

    @torch.no_grad()
    def predict(self, index: torch.Tensor) -> torch.Tensor:
        self.model.eval()
        outputs = [
            self.logits(index[start : start + self.batch_size]).float().softmax(dim=1).cpu()
            for start in range(0, len(index), self.batch_size)
        ]
        return torch.cat(outputs) if outputs else torch.empty(0, self.model.num_classes)


def run_training(config: Config, command: list[str]) -> Path:
    """Run one configured customisation and return the run directory."""
    seeds = seed_everything(config.run.seed)
    device = config.run.device
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("run.device is 'cuda' but no CUDA device is available.")
    record = rec.new_record("training", config=config.as_dict(), command=command)
    record["config_file"] = rec.file_entry(config.path)
    record["seeds"] = seeds

    manifest = config.resolve(config.data.manifest)
    root = config.resolve(config.data.root) if config.data.root else manifest.parent
    cases = read_manifest(manifest, root, config.data.columns)
    split = grouped_split(
        cases, config.data.test_groups, config.data.validation_fraction, config.run.seed
    )
    labels = sorted({case.label for case in cases})
    label_index = {label: index for index, label in enumerate(labels)}

    run_dir = rec.new_run_dir(config.resolve(config.run.output_root), config.run.name)
    model, base_entry, parents = build_model(config, len(labels))
    tiny_base = None
    if config.model.base.startswith("tiny"):
        tiny_base = run_dir / TINY_BASE_NAME
        save_file({k: v.contiguous() for k, v in model.state_dict().items()}, str(tiny_base))
    record["base_model"] = base_entry
    record["parents"] = parents

    trainable = apply_method(
        model,
        config.method.name,
        lora_rank=config.method.lora_rank,
        lora_alpha=config.method.lora_alpha,
        last_blocks=config.method.last_blocks,
    )
    model.to(device)
    store = CaseStore(config, cases, model, root)
    y = torch.tensor([label_index[case.label] for case in cases])
    position = {case.case_id: index for index, case in enumerate(cases)}
    parts = {
        name: torch.tensor([position[i] for i in getattr(split, name)], dtype=torch.long)
        for name in ("train", "validation", "test")
    }
    record["data"] = {
        "manifest": rec.file_entry(manifest),
        "cases": len(cases),
        "classes": labels,
        "groups": sorted({case.group for case in cases}),
        "files": store.files,
        "preprocessing": store.preprocessing(),
    }

    runner = Runner(config, model, store, device)
    parameters = [p for p in model.parameters() if p.requires_grad]
    optimiser = torch.optim.AdamW(
        parameters, lr=config.training.learning_rate, weight_decay=config.training.weight_decay
    )
    class_weights = None
    if config.training.class_weighting == "balanced":
        counts = torch.bincount(y[parts["train"]], minlength=len(labels)).float()
        class_weights = (counts.sum() / (len(labels) * counts.clamp_min(1.0))).to(device)
    loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights)
    generator = torch.Generator().manual_seed(config.run.seed)

    history, best = [], None
    for epoch in range(1, config.training.epochs + 1):
        warmup = max(1, config.training.warmup_epochs)
        rate = config.training.learning_rate * min(1.0, epoch / warmup)
        for group in optimiser.param_groups:
            group["lr"] = rate
        model.train()
        order = parts["train"][torch.randperm(len(parts["train"]), generator=generator)]
        total = 0.0
        for start in range(0, len(order), config.training.batch_size):
            index = order[start : start + config.training.batch_size]
            loss = loss_fn(runner.logits(index, generator).float(), y[index].to(device))
            optimiser.zero_grad(set_to_none=True)
            loss.backward()
            optimiser.step()
            total += loss.item() * len(index)
        probabilities = runner.predict(parts["validation"])
        score = balanced_accuracy(y[parts["validation"]], probabilities.argmax(dim=1))
        history.append(
            {
                "epoch": epoch,
                "learning_rate": rate,
                "train_loss": total / len(order),
                "validation_balanced_accuracy": score,
            }
        )
        if best is None or score > best["score"]:
            best = {"epoch": epoch, "score": score, "state": trainable_state(model)}

    model.load_state_dict(best["state"], strict=False)
    outputs = [run_dir / ADAPTER_NAME]
    save_file(
        best["state"],
        str(outputs[0]),
        metadata=rec.weight_metadata(method=config.method.name, base_model=base_entry["id"]),
    )
    if tiny_base is not None:
        outputs.append(tiny_base)
    outputs.append(_write_json(run_dir / "split.json", split.as_dict()))
    outputs.append(_write_json(run_dir / "labels.json", labels))
    outputs.append(_write_json(run_dir / "history.json", history))
    outputs.append(Path(shutil.copyfile(config.path, run_dir / "config.toml")))
    for name in ("validation", "test"):
        if len(parts[name]):
            selected = [cases[i] for i in parts[name].tolist()]
            path = run_dir / f"predictions_{name}.csv"
            outputs.append(_write_predictions(path, selected, labels, runner.predict(parts[name])))

    record["method"] = {
        "name": config.method.name,
        "trainable_tensors": len(trainable),
        "trainable_parameters": int(sum(p.numel() for p in parameters)),
        "total_parameters": int(sum(p.numel() for p in model.parameters())),
        "base_features_cached": runner.cached is not None,
    }
    record["split"] = _split_summary(split)
    record["selection"] = {
        "metric": "validation_balanced_accuracy",
        "rule": "highest value; ties keep the earlier epoch",
        "best_epoch": best["epoch"],
        "best_value": best["score"],
        "test_metric": "not computed in training",
    }
    rec.finish_record(record, run_dir, outputs)
    return run_dir


def _split_summary(split: Split) -> dict[str, Any]:
    return {
        "file": "split.json",
        "train_cases": len(split.train),
        "validation_cases": len(split.validation),
        "test_cases": len(split.test),
        "train_groups": list(split.train_groups),
        "validation_groups": list(split.validation_groups),
        "test_groups": list(split.test_groups),
    }


def _write_json(path: Path, value: Any) -> Path:
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    return path


def _write_predictions(
    path: Path, cases: list[Case], labels: list[str], probabilities: torch.Tensor
) -> Path:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["case_id", "group", "label", "predicted", *[f"p_{x}" for x in labels]])
        for case, row in zip(cases, probabilities, strict=True):
            predicted = labels[int(row.argmax())]
            values = [f"{v:.6f}" for v in row.tolist()]
            writer.writerow([case.case_id, case.group, case.label, predicted, *values])
    return path
