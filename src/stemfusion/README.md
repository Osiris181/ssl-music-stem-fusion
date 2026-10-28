# `stemfusion` package

## Purpose

This is the reusable Python package for the paper pipeline.

## Code organization

Current subpackages:

- `separation/`: HT-Demucs dataset separation.
- `data/`: dataset manifest construction and validation.
- `features/`: frozen feature extraction.
- `models/`: reusable downstream classifier architectures.
- `training/`: downstream training and evaluation workflows.

`cli.py` assembles the command-line interface, while `__main__.py` supports
`python -m stemfusion`.

## How to use the package

Inspect the available stages and their arguments with:

    python -m stemfusion --help
    python -m stemfusion data fma-manifest --help
    python -m stemfusion separate htdemucs --help
    python -m stemfusion extract htdemucs musicfm --help
    python -m stemfusion extract htdemucs rms --help
    python -m stemfusion train fma-genre --help

New classifier, training, and evaluation code should be added as reusable
subpackages rather than dataset-specific scripts at the repository root.
