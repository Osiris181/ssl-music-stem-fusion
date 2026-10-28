# Training configurations

## Purpose

This directory records training settings shared by directly comparable
experiments.

## Contents

| File | Purpose |
| --- | --- |
| `fma_genre_mlp.yaml` | Shared MLP, optimizer, checkpoint, metric, and seed settings for FMA-Medium genre classification. |
| `fma_htdemucs_mid_fusion_mlp.yaml` | Table 3 HT-Demucs MusicFM mid-fusion settings. |
| `fma_htdemucs_late_fusion_mlp.yaml` | Table 4 HT-Demucs unweighted late-fusion settings. |
| `fma_htdemucs_rms_weighted_late_fusion_mlp.yaml` | Table 4 HT-Demucs RMS-weighted late-fusion settings. |

The command-line defaults shown by the corresponding `stemfusion train` command
must remain consistent with these files.
