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
- `experiments/fma_medium/`: the only active dataset entry points.

`configuration.py` loads/validates YAML presets and records configuration
provenance. `cli.py` assembles the command-line interface, while `__main__.py` supports
`python -m stemfusion`.

## How to use the package

Inspect the available stages and their arguments with:

    python -m stemfusion --help
    python -m stemfusion data fma-manifest --help
    python -m stemfusion separate htdemucs --help
    python -m stemfusion extract fma-medium --help
    python -m stemfusion train fma-medium mid-fusion --help
    python -m stemfusion train fma-medium late-fusion --help

Start with [the FMA-Medium guide](experiments/fma_medium/README.md).
See [configuration usage](../../docs/configuration.md) for automatic presets,
`--config`, `--run`, and CLI overrides.
Keep reusable implementation in the shared subpackages; dataset folders
contain orchestration and experiment-specific documentation.
