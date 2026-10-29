# Dataset manifests

## Purpose

This package converts dataset metadata and audio locations into explicit,
validated manifests shared by feature extraction and classification.

## Code

Current implementation:

| File | Purpose |
| --- | --- |
| `fma.py` | Parse FMA-Medium metadata and align mixture and HT-Demucs paths by six-digit track ID. |

The FMA builder writes JSON Lines records containing `track_id`, `split`,
`genre`, `label`, and a `paths` mapping. It supports mixture-only manifests and
complete `htdemucs_6s` tracks distributed across multiple storage roots. All six
HT-Demucs stems for a track must come from the same root.

Important functions in `fma.py` are:

- `read_fma_medium_metadata()`: parse the official multi-row FMA CSV header;
- `build_fma_manifest()`: align labels, splits, mixtures, and stems;
- `read_manifest()`: validate the JSONL records consumed downstream.

## How to use it

Inspect all arguments with:

    stemfusion data fma-manifest --help

Multi-disk HT-Demucs examples are in `docs/fma_medium_extraction.md`.
The builder inventories all six separator outputs; revised training selects
the five named stems and excludes `other`.

Manifest builders must preserve official splits, detect duplicate IDs, and
avoid silently using different track sets for directly comparable methods.
