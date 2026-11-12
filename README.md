# Affinage

Training and fine-tuning open models on domain data, packaged as customised
local models. Part of [Fonderie](https://github.com/fonderie-dev).

*Affinage* is the refining of molten metal: removing impurities until it
reaches a specified composition.

Project site: <https://fonderie-dev.github.io/fonderie-affinage/>, an outline of
this repository; sections that describe functionality are marked *coming
soon* until the code exists. Its source is in [docs/](docs/).

> **Status: draft (●○○).** The parts listed under *Implemented* work and are
> tested on synthetic data. No model trained on real data has been produced or
> released, and interfaces may change.

## Implemented

For hyperspectral region classification:

- **Run record** (`fonderie_affinage.record`, schema `fonderie.run-record/1`):
  base model and revision, data manifest with SHA-256 per file, configuration,
  seeds, code commit, every installed package version, hardware, start and end
  time, parent records and checksummed outputs. `affinage verify` checks a
  record against its files. Run directories are never reused.
- **Data** (`fonderie_affinage.hsi`): CSV manifests, TIVITA `SpecCube` and
  `.npy` cubes, region masks, a window that keeps every region pixel, and
  grouped splits in which all cases of an animal or patient stay together.
- **Base models**, each converted once from its pickle-based checkpoint to
  `safetensors` under size and SHA-256 pins, with a conversion record:
  - [HyperSIGMA](https://huggingface.co/WHU-Sigma/HyperSIGMA) ViT-B spatial
    and spectral encoders, fed a 64 × 64 window around the region. The encoder
    code is written here; nothing is imported from the upstream repository.
  - [HTC](https://github.com/IMSY-DKFZ/htc), the DKFZ organ-segmentation U-Net
    (EfficientNet-B5, run `image@2022-02-03_22-58-44`), fed the whole
    L1-normalised frame. Encoder-stage and decoder-block outputs are averaged
    over the region (1,336 features). Needs the `htc` extra.
- **Customisation** from one configuration file: `head` (base model frozen;
  its features are computed once and reused), `lora` (low-rank adapters on the
  attention layers of the last transformer blocks; HyperSIGMA only) and
  `last_blocks` (last transformer blocks, or last U-Net decoder blocks,
  unfrozen). The epoch is selected on validation groups only. Predictions for
  the held-out groups are written out; no test metric is computed in training.
- **Export**: merged `safetensors` weights with a generated model card and an
  export record that points to the training record.

Verified so far, all on CPU:

- The test suite trains, records, reloads and exports a tiny model on
  synthetic cubes; two runs with the same seed give byte-identical weights.
- The HyperSIGMA conversion ran on the pinned snapshots. On random input the
  encoders written here reproduce the upstream encoder path at commit
  `07e9ea2` to within 1e-5.
- A two-epoch LoRA run on the converted encoders with synthetic 100-band cubes
  trained, verified and exported.
- The HTC conversion ran on the pinned archive; a 480 × 640 × 100 frame passes
  through the converted model, and the pooled decoder path equals the
  library's own decoder output.

Not yet done: a training run on real data. Nothing here says how well a
customised model performs.

## Use

```sh
python -m pip install -e ".[train]"

# 1. Convert the base model once (files from the pinned Hugging Face revision).
affinage convert-hypersigma \
  --spatial spat-vit-base-ultra-checkpoint-1599.pth \
  --spectral spec-vit-base-ultra-checkpoint-1599.pth \
  --out hypersigma-converted

# 1b. Optional second base model (pip install -e ".[train,htc]").
affinage convert-htc \
  --archive image@2022-02-03_22-58-44_generated_default_model_comparison.zip \
  --out htc-converted

# 2. Edit the paths in the configuration, then train.
affinage train configs/heiporspectral_hypersigma_lora.toml
affinage train configs/heiporspectral_htc_head.toml

# 3. Check the record and export the customised model.
affinage verify runs/<run>/record.json
affinage export runs/<run> --out export/<name>
```

A dry run without real data or a GPU:

```sh
affinage synthetic scratch/synthetic
affinage train configs/synthetic_tiny.toml
```

The manifest is a CSV with one row per annotated region. The default column
names are `case_id`, `label_name`, `subject_name`, `cube_path` and `mask_path`;
others can be set under `[data.columns]`.

## Intended scope

- Configuration-driven training and fine-tuning runs of open-weight models,
  full and parameter-efficient (for example LoRA), from a base model pinned to
  a commit hash.
- Data preparation with a manifest and checksums, and a recorded split that
  [Essai](https://github.com/fonderie-dev/fonderie-essai) can check for
  train–test overlap.
- A run record written next to every output (see the organisation's
  [contributing guide](https://github.com/fonderie-dev/.github/blob/main/CONTRIBUTING.md#runs-and-artefacts)).
- Export of the result as a customised local model: merged `safetensors`
  weights, an adapter, or a quantised file for a local runtime, each with a
  model card.

Not in scope: hosted training services; weight inspection and surgery
([Burin](https://github.com/fonderie-dev/fonderie-burin)); benchmark
evaluation ([Essai](https://github.com/fonderie-dev/fonderie-essai)).

## First milestones

1. Run-record schema and writer, shared in form with Essai. *Done.*
2. A fine-tuning run on a tiny model and synthetic data that completes on CPU
   inside the test suite. *Done.*
3. Export with a generated model card. *Done.*

Next: a recorded HyperSIGMA run on HeiPorSPECTRAL on a GPU; quantised export.

## Development

```sh
python -m pip install -e ".[dev,train,htc]"
ruff check . && ruff format --check .
python -m pytest
```

```
src/fonderie_affinage/   package
configs/                 run configurations
tests/                   pytest suite: no downloads, no GPU
docs/                    project site (Hugo), deployed to GitHub Pages
```

CI runs the shared
[Fonderie workflow](https://github.com/fonderie-dev/.github/blob/main/.github/workflows/python-check.yaml)
on Python 3.11–3.14. It installs the `dev` extra only, so the tests that
need PyTorch are skipped there and run locally with the `train` extra.

## Licence

Apache-2.0; see [LICENSE](LICENSE) and [NOTICE](NOTICE). Base models and
datasets used with this package keep their own licences. © 2026 R. Heller.
