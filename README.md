# Affinage

Training and fine-tuning open models on domain data, packaged as customised
local models. Part of [Fonderie](https://github.com/fonderie-dev).

*Affinage* is the refining of molten metal: removing impurities until it
reaches a specified composition.

> **Status: scaffold (○○○).** The package installs and its smoke test passes.
> None of the scope below is implemented yet.

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

1. Run-record schema and writer, shared in form with Essai.
2. A fine-tuning run on a tiny model and synthetic data that completes on CPU
   inside the test suite.
3. Export with a generated model card.

## Development

```sh
python -m pip install -e ".[dev]"
ruff check . && ruff format --check .
python -m pytest
```

```
src/fonderie_affinage/   package
tests/                   pytest suite: no downloads, no GPU
```

CI runs the shared
[Fonderie workflow](https://github.com/fonderie-dev/.github/blob/main/.github/workflows/python-check.yaml)
on Python 3.11–3.14.

## Licence

Apache-2.0; see [LICENSE](LICENSE) and [NOTICE](NOTICE). Base models and
datasets used with this package keep their own licences. © 2026 R. Heller.
