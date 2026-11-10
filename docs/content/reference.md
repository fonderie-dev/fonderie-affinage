---
title: "Référence"
description: "Reference for Affinage: Python API, configuration, data manifest, run record and export formats. Outline; coming soon."
summary: "Every public function, file format and schema, field by field."
lede: "The exact interfaces of Affinage: functions, configuration keys and file formats."
outlineTitle: "Planned sections"
outline:
  - id: python-api
    title: "Python API"
    text: "Public modules, functions and classes of `fonderie_affinage`, generated from their docstrings once the interfaces settle."
  - id: configuration
    title: "Configuration keys"
    text: "Every key of a run configuration with its type, default and effect."
  - id: data-manifest
    title: "Data manifest and split"
    text: "The manifest of input files with checksums, and the recorded train–test split that Essai reads."
  - id: run-record
    title: "Run-record schema"
    text: "The run record field by field, shared in form with Essai."
  - id: export-formats
    title: "Export formats"
    text: "The files written for merged weights, adapters and quantised models, and the model-card template."
---

## Available now

The package currently exposes its version, `fonderie_affinage.__version__`,
and nothing else.
