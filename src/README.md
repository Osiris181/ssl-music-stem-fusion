# Source tree

## Purpose

This directory uses the standard Python `src` layout. The installable package
is `stemfusion/`; keeping it below `src/` prevents tests from accidentally
importing an uninstalled copy from the repository root.

## Code contained here

All reusable pipeline code belongs below `src/stemfusion/`. Datasets,
checkpoints, feature caches, and one-off experiment outputs do not belong in
the source tree.

## How to use the source tree

Use the package through the `stemfusion` command after installation, or during
development with:

    PYTHONPATH=src python -m stemfusion --help

Research data, checkpoints, separated audio, feature caches, and experiment
outputs must not be stored under this directory.
