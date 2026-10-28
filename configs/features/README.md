# Feature configurations

## Purpose

This directory records settings for frozen feature extractors.

## Contents

| File | Purpose |
| --- | --- |
| `musicfm_msd_layer7.yaml` | Paper-aligned MusicFM-MSD layer-7 extraction for HT-Demucs mixtures and stems. |
| `rms_mean_scaled20.yaml` | HT-Demucs mean frame-level RMS, scaling factor 20.0, and separate one-value caches for weighted fusion. |

## How to use these settings

Use these files to review the effective model, audio, layer, pooling, RMS,
runtime, and output settings. Their values correspond to the defaults
shown by:

    stemfusion extract htdemucs musicfm --help
    stemfusion extract htdemucs rms --help

Checkpoint files and feature caches do not belong in this directory. Supply
the local MusicFM repository and checkpoint paths to
`stemfusion extract htdemucs musicfm`.
