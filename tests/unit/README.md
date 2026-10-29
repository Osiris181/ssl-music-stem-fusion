# Unit tests

## Purpose

These tests cover deterministic behavior that can be checked without large
datasets or model checkpoints.

## Test files

| File | Coverage |
| --- | --- |
| `test_direct_extraction_script.py` | Direct Python-file execution without PYTHONPATH, YAML source selection, and non-writing dry runs. |
| `test_yaml_configuration.py` | YAML precedence, relative paths, run selection, strict validation, safe loading, and saved resolved settings. |
| `test_multi_source_extraction.py` | YAML source lists, CLI subsets, one-time encoder loading, cache skipping, partial failures, and aggregate reports. |
| `test_htdemucs.py` | Multi-root discovery, six-stem completion, commands, storage resume, and rescue behavior. |
| `test_fma_manifest.py` | FMA splits, distributed HT-Demucs roots, and aligned missing-file handling. |
| `test_musicfm_features.py` | HT-Demucs MusicFM cache keys, reports, dry runs, and partial failures. |
| `test_feature_entry_points.py` | MusicFM routing and rejection of archived RMS commands. |
| `test_fma_late_fusion.py` | Unweighted cache planning, source-logit aggregation, and synthetic training artifacts. |
| `test_fma_experiment_entry_points.py` | Dataset-specific routing, matching training defaults, and rejection of superseded commands. |
| `test_genre_training.py` | Source ordering, clean and legacy cache resolution, dry-run planning, and a dependency-gated synthetic mid-fusion run. |

## How to run them

Run the unit directory with:

    PYTHONPATH=src python -m unittest discover -s tests/unit -v

Run one module while developing with:

    PYTHONPATH=src python -m unittest tests.unit.test_musicfm_features -v

New tests should avoid relying on user-specific absolute paths. Use temporary
directories and dependency injection for expensive model calls.
