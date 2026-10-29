# Separator configurations

## Purpose

This directory records the source-separation settings used by the cleaned
wrappers.

## Contents

| File | Purpose |
| --- | --- |
| `htdemucs_6s.yaml` | Six-source HT-Demucs settings and dataset scheduling defaults. |

## How to use these settings

The matching command automatically loads `htdemucs_6s.yaml`. Use `--config`
for a custom YAML; explicit CLI arguments override configured values.
Review the interface before running:

    stemfusion separate htdemucs --help

These files contain reproducibility settings, not local audio,
checkpoint, or output paths. Run the corresponding commands with
`stemfusion separate htdemucs`.
