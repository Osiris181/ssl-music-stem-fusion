# Dataset experiments

Dataset-specific entry points live here. Shared data, separation, feature,
model, and training implementations stay in their corresponding packages.

[FMA-Medium](fma_medium/README.md), [GTZAN](gtzan/README.md), and
[ISMIR04](ismir04/README.md) provide dataset entry points. ISMIR04 also
audits exact duplicates and explicitly chunks long tracks during extraction.

All three datasets provide `separation.py`, `prepare_manifest.py`, `extraction.py`, `train_mid.py`,
and `train_late.py`, each supporting direct execution. Shared orchestration is:

| File | Purpose |
| --- | --- |
| `_datasets.py` | Select dataset-specific manifest and cache adapters. |
| `_separation.py` | Common direct separation CLI and dataset YAML selection; uses `separation/htdemucs.py`. |
| `_extraction.py` | Common extraction CLI and handoff to `features/_musicfm.py`. |
| `_genre.py` | Common training CLI, seed/condition scheduling, reuse, and summaries. |
| `_stem_results.py` | Common standalone-branch validation and reporting. |

FMA's `training.py` and `stem_results.py` retain compatibility imports for
existing callers. These shared modules are implementation details; run the
dataset entry scripts or unified CLI. Inspect their options with:

    stemfusion extract fma-medium --help
    stemfusion train fma-medium --help
    stemfusion extract gtzan --help
    stemfusion train gtzan --help
    stemfusion extract ismir04 --help
    stemfusion train ismir04 --help
