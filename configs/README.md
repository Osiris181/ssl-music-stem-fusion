# Configurations

## Purpose

This directory records reusable experiment settings separately from
machine-specific paths and executable code.

## Contents

Current groups:

- `separators/`: HT-Demucs inference settings.
- `features/`: frozen feature-extractor settings.
- `training/`: shared downstream model and optimization settings.
- `experiments/paper/`: the source combinations associated with paper tables.

## How to use these files

The YAML files document the canonical values that the command-line defaults
must match. The current CLI does not load these files directly. Local dataset,
checkpoint, and output paths must be supplied through command-line arguments
and must not be committed here.

When a default changes in Python code, update its YAML file and unit test in
the same change. Use the component README to find the corresponding command.

The historical HT-Demucs Table 3 and Table 4 matrices are in
`experiments/paper/table_03_fma_htdemucs.yaml` and
`experiments/paper/table_04_fma_htdemucs.yaml`.

These are original-paper configurations. The revised matched controls and
five-seed protocol in `../PHASE2_EXPERIMENTAL_DESIGN.md` are not implemented yet.
