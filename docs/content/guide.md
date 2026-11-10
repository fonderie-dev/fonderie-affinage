---
title: "Guide"
description: "How to use Affinage: installation, run configuration, data preparation, training methods, run records and export. Outline; coming soon."
summary: "From a domain dataset and a pinned base model to a customised local model, step by step."
lede: "How a domain dataset and a pinned base model become a customised local model, with a record of every step."
outlineTitle: "Planned chapters"
outline:
  - id: installation
    title: "Installation"
    text: "Installing a released version, and the optional extras that bring in the training libraries only when they are needed."
  - id: first-run
    title: "A first run"
    text: "A complete fine-tuning run on a tiny model and synthetic data, on CPU: from configuration to exported model in a few minutes."
  - id: configuration
    title: "Run configuration"
    text: "The configuration file that describes a run: base model and revision, data, method, hyperparameters and seeds, kept under version control."
  - id: data
    title: "Preparing domain data"
    text: "Turning a domain dataset into training input with a manifest and checksums, and recording the train–test split that [Essai](https://fonderie-dev.github.io/fonderie-essai/) checks for overlap."
  - id: methods
    title: "Training methods"
    text: "Full fine-tuning and parameter-efficient methods such as LoRA: what each needs in memory and time, and when to choose which."
  - id: run-record
    title: "The run record"
    text: "What Affinage writes next to every output, following the [Fonderie run record](https://fonderie-dev.github.io/website/methode/#the-run-record)."
  - id: export
    title: "Export as a local model"
    text: "Merged `safetensors` weights, an adapter, or a quantised file for a local runtime, and how to choose between them."
  - id: model-card
    title: "The model card"
    text: "The generated model card: base model, data, record, licence obligations and known limitations."
  - id: licences
    title: "Licences and restricted data"
    text: "Checking base-model and data licences before training and before release, and why weights trained on restricted data are not published. See [licensing and memorisation](https://fonderie-dev.github.io/website/methode/#licensing-and-memorisation)."
---

## Available now: install from source

There is no release yet. The scaffold installs from a clone of the repository:

```sh
git clone https://github.com/fonderie-dev/fonderie-affinage.git
cd fonderie-affinage
python -m pip install -e ".[dev]"
python -m pytest
```

The tests use tiny synthetic models and data; they need neither a GPU nor
downloads.
