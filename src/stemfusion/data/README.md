# Dataset manifests

## Purpose

This package converts dataset metadata and audio locations into explicit,
validated manifests shared by feature extraction and classification.

## Code

Current implementation:

| File | Purpose |
| --- | --- |
| `fma.py` | Parse FMA-Medium metadata and align mixture and HT-Demucs paths by six-digit track ID. |
| `fma_features.py` | Verify extraction reports, cache contents, exact split IDs, labels, and dimensions; load the common training track set. |
| `features.py` | Dataset-independent aligned feature containers passed to training. |

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

Alternatively, from `src/stemfusion/experiments/fma_medium/`, run
`python prepare_manifest.py --help`. This direct entry point uses the same
builder and defaults its output to
`data/manifests/fma_medium/fma_medium_htdemucs.jsonl` in the repository. The
extraction and training YAML files reference that path. The CLI command above
continues to require an explicit `--output`.

Multi-disk HT-Demucs examples are in `docs/fma_medium_extraction.md`.
The builder inventories all six separator outputs; revised training selects
the five named stems and excludes `other`.

Manifest builders must preserve official splits, detect duplicate IDs, and
avoid silently using different track sets for directly comparable methods.

`load_fma_features()` reads local cached embeddings, not audio files. All six
views must contain exactly the manifest IDs for each split, with finite
float32 vectors of the configured dimension. Zero vectors are retained and
counted. Input file checksums and extraction reports are recorded in a data
audit; no track intersection or additional normalization is applied.
