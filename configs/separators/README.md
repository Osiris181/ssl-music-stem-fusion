# Separator configurations

## Purpose

This directory records the source-separation settings used by the cleaned
wrappers.

## Contents

| File | Purpose |
| --- | --- |
| `htdemucs_6s.yaml` | Six-source HT-Demucs settings and dataset scheduling defaults. |

## How to use these settings

Review these files before running the matching commands:

    stemfusion separate htdemucs --help

These files contain reproducibility settings, not local audio,
checkpoint, or output paths. Run the corresponding commands with
`stemfusion separate htdemucs`.
