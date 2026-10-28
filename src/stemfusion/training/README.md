# Training

## Purpose

This directory contains reusable downstream training workflows. It consumes
cached features and dataset manifests; it does not run source separation or
feature extraction.

## Contents

| File | Purpose |
| --- | --- |
| `genre.py` | Shared mid-fusion FMA multiclass genre training and evaluation. |
| `htdemucs_mid_fusion.py` | Public Table 3 HT-Demucs mid-fusion workflow. |
| `htdemucs_late_fusion.py` | Public Table 4 unweighted late-fusion workflow. |
| `htdemucs_rms_weighted_late_fusion.py` | Public Table 4 RMS-weighted late-fusion workflow. |

The genre workflow supports one source or a deterministic concatenation of
several sources. `genre.py` supplies shared loading, metrics and training
helpers used by the HT-Demucs workflows and the original-mixture reference.

Inspect all options with:

    stemfusion train fma-genre --help
    stemfusion train fma-htdemucs-mid-fusion --help
    stemfusion train fma-htdemucs-late-fusion --help
    stemfusion train fma-htdemucs-rms-weighted-late-fusion --help

The defaults retain the historical Table 3 and Table 4 training
protocol: Adam at `1e-4`, batch size 128, 20 epochs, seed 42, hidden dimensions
1024 and 512, and dropout 0.2. Checkpoint selection maximizes validation
accuracy plus macro ROC-AUC plus macro PR-AUC.

The revised matched controls and five-seed protocol remain planned work.
RMS-weighted late fusion is historical functionality outside the revised
primary matrix.
