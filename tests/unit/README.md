# Unit tests

## Purpose

These tests cover deterministic behavior that can be checked without large
datasets or model checkpoints.

## Test files

| File | Coverage |
| --- | --- |
| `test_htdemucs.py` | Multi-root discovery, six-stem completion, commands, storage resume, and rescue behavior. |
| `test_fma_manifest.py` | FMA splits, distributed HT-Demucs roots, and aligned missing-file handling. |
| `test_musicfm_features.py` | HT-Demucs MusicFM cache keys, reports, dry runs, and partial failures. |
| `test_htdemucs_features.py` | HT-Demucs RMS one-value caches, settings reports, dry runs, and partial failures. |
| `test_feature_entry_points.py` | HT-Demucs CLI routing, mixture/source scopes, and MusicFM/RMS modes. |
| `test_htdemucs_late_fusion.py` | Unweighted and RMS-weighted training, separate and legacy appended RMS caches, and logit aggregation. |
| `test_htdemucs_training_entrypoints.py` | Table 3/4 command routing and HT-Demucs-specific defaults. |
| `test_genre_training.py` | Source ordering, clean and legacy cache resolution, dry-run planning, and a dependency-gated synthetic mid-fusion run. |

## How to run them

Run the unit directory with:

    PYTHONPATH=src python -m unittest discover -s tests/unit -v

Run one module while developing with:

    PYTHONPATH=src python -m unittest tests.unit.test_musicfm_features -v

New tests should avoid relying on user-specific absolute paths. Use temporary
directories and dependency injection for expensive model calls.
