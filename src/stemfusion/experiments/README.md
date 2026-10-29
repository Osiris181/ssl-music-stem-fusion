# Dataset experiments

Dataset-specific entry points live here. Shared data, separation, feature,
model, and training implementations stay in their corresponding packages.

Only [FMA-Medium](fma_medium/README.md) is currently active. Add other dataset
folders as their implementations are migrated; no placeholder datasets are
provided.

FMA-Medium task entry points are grouped as `extraction.py`, `train_mid.py`,
and `train_late.py`. Inspect their options with:

    stemfusion extract fma-medium --help
    stemfusion train fma-medium --help
