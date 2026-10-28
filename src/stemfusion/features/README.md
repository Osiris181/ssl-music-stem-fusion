# Feature extraction

## Purpose

This package contains the frozen feature-extraction workflows used after music
source separation.

## Public workflows

| File | Separation model | Features written |
| --- | --- | --- |
| `htdemucs.py` | HT-Demucs | MusicFM-MSD layer-7 embeddings and separate mean-RMS energy values for the mixture and six HT-Demucs stems. |

`htdemucs.py` is the public workflow entry point. Its commands are:

    stemfusion extract htdemucs musicfm --help
    stemfusion extract htdemucs rms --help

The HT-Demucs RMS mode writes one separate scaled scalar per track and source.
It never appends energy to a MusicFM embedding.

## Private implementation helpers

`_musicfm.py` holds MusicFM model loading, layer extraction, temporal mean
pooling, cache writing, and failure reporting. `_htdemucs_rms.py` holds the
HT-Demucs RMS implementation. Access both through the public HT-Demucs module.

Both extraction modes read aligned JSONL manifests and write one cache per source and
dataset split. Cache keys are stable FMA track IDs. Complete outputs are
written atomically; failures produce a partial cache and JSON report.

## How to use it

Build a dataset manifest first, then use the HT-Demucs subcommands. Complete
commands and output filenames are in `docs/fma_htdemucs_features.md`.

The revised primary design uses the mixture and five named stems, excluding
`other`. RMS mode and `other` support remain available as historical utilities.

The upstream MusicFM repository and model files remain external dependencies
and must not be copied into this package.
