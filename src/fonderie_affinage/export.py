"""Export a finished run as one customised local model with a model card."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file, save_file

from fonderie_affinage import record as rec
from fonderie_affinage.config import load_config
from fonderie_affinage.models.adaptation import apply_method, merge_adapters
from fonderie_affinage.train import ADAPTER_NAME, TINY_BASE_NAME, build_model

MODEL_NAME = "model.safetensors"
CARD_NAME = "MODEL_CARD.md"


def load_run_model(
    run_dir: Path, converted_dir: Path | None = None
) -> tuple[torch.nn.Module, dict[str, Any], list[str]]:
    """Rebuild the trained classifier of a run with adapters merged into the weights."""
    run_dir = Path(run_dir)
    problems = rec.verify_outputs(run_dir / rec.RECORD_NAME)
    if problems:
        raise rec.RecordError(f"Run failed verification: {problems}")
    record = json.loads((run_dir / rec.RECORD_NAME).read_text(encoding="utf-8"))
    labels = json.loads((run_dir / "labels.json").read_text(encoding="utf-8"))
    config = load_config(run_dir / "config.toml")
    if not config.model.base.startswith("tiny"):
        # Paths in the copied configuration were relative to the original file.
        local = converted_dir or record["base_model"].get("local_path")
        if not local:
            raise rec.RecordError("The run record does not say where the converted base model is.")
        config = replace(config, model=replace(config.model, converted_dir=str(local)))
    tiny_base = run_dir / TINY_BASE_NAME if config.model.base.startswith("tiny") else None
    model, _, _ = build_model(config, len(labels), tiny_base=tiny_base)
    apply_method(
        model,
        config.method.name,
        lora_rank=config.method.lora_rank,
        lora_alpha=config.method.lora_alpha,
        last_blocks=config.method.last_blocks,
    )
    result = model.load_state_dict(load_file(str(run_dir / ADAPTER_NAME)), strict=False)
    if result.unexpected_keys:
        raise rec.RecordError(f"Adapter holds unknown tensors: {result.unexpected_keys[:5]}")
    merge_adapters(model)
    model.eval()
    return model, record, labels


def export_run(
    run_dir: Path, out_dir: Path, command: list[str], converted_dir: Path | None = None
) -> Path:
    """Write merged weights, labels and a model card; return the export record path."""
    run_dir, out_dir = Path(run_dir), Path(out_dir)
    model, parent, labels = load_run_model(run_dir, converted_dir)
    out_dir.mkdir(parents=True, exist_ok=False)
    record = rec.new_record("export", config={"run": run_dir.name}, command=command)
    record["parents"] = [rec.parent_reference(run_dir / rec.RECORD_NAME)]
    record["base_model"] = parent["base_model"]
    record["data"] = {"note": "No new data; see the parent training record."}
    record["seeds"] = parent["seeds"]

    target = out_dir / MODEL_NAME
    state = {
        name: tensor.detach().cpu().contiguous() for name, tensor in model.state_dict().items()
    }
    save_file(
        state,
        str(target),
        metadata=rec.weight_metadata(
            base_model=parent["base_model"]["id"], encoder=model.config.as_dict(), labels=labels
        ),
    )
    labels_path = out_dir / "labels.json"
    labels_path.write_text(json.dumps(labels, indent=2) + "\n", encoding="utf-8")
    card = out_dir / CARD_NAME
    card.write_text(model_card(parent, record["record_id"], labels), encoding="utf-8")
    return rec.finish_record(record, out_dir, [target, labels_path, card])


def model_card(training: dict[str, Any], export_id: str, labels: list[str]) -> str:
    base, data, method = training["base_model"], training["data"], training["method"]
    selection, split = training["selection"], training["split"]
    lines = [
        f"# {training['config']['run']['name']}",
        "",
        "A hyperspectral region classifier customised with Fonderie Affinage.",
        "",
        "## Provenance",
        "",
        f"- Training record: `{training['record_id']}`",
        f"- Export record: `{export_id}`",
        f"- Code: fonderie-affinage {training['code']['version']}, "
        f"commit `{training['code']['commit']}`, uncommitted changes: {training['code']['dirty']}",
        "",
        "## Base model",
        "",
        f"- Identifier: {base['id']}",
        f"- Revision: `{base['revision']}`",
        f"- Source: {base['source']}",
        f"- Licence: {base['licence']}",
        "",
        "## Customisation",
        "",
        f"- Method: {method['name']} ({method['trainable_parameters']:,} of "
        f"{method['total_parameters']:,} parameters trained)",
        f"- Seed: {training['seeds']['torch']}",
        f"- Selected epoch: {selection['best_epoch']} by {selection['metric']} "
        f"= {selection['best_value']:.4f}",
        "",
        "## Data",
        "",
        f"- Manifest SHA-256: `{data['manifest']['sha256']}`",
        f"- Cases: {data['cases']} in {len(data['groups'])} groups, {len(labels)} classes",
        f"- Training groups: {', '.join(split['train_groups'])}",
        f"- Validation groups: {', '.join(split['validation_groups'])}",
        f"- Held-out groups: {', '.join(split['test_groups']) or 'none'}",
        f"- Classes: {', '.join(labels)}",
        "",
        "The model inherits the licence terms of its base model and of its training data.",
        "State the data source, its licence and its citation here before releasing the model.",
        "",
        "## Limitations",
        "",
        "- The validation value above was used to select the epoch; it is not a test result.",
        "- No held-out performance has been established by this export. Evaluate the model",
        "  on the held-out groups, with uncertainty, before reporting any number.",
        "- The model classifies an annotated region of one cube. It does not segment images.",
        "- Not a medical device; not validated for clinical use.",
        "",
    ]
    return "\n".join(lines)
