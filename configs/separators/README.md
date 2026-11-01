# Separator configurations

## Purpose

This directory records the source-separation settings used by the cleaned
wrappers.

## Contents

| File | Purpose |
| --- | --- |
| `htdemucs_6s.yaml` | Six-source HT-Demucs settings and dataset scheduling defaults. |

## How to use these settings

The matching command automatically loads `htdemucs_6s.yaml`. Use `--config`
for a custom YAML; explicit CLI arguments override configured values.
Review the interface before running:

    stemfusion separate htdemucs --help

These files contain reproducibility settings, not local audio,
checkpoint, or output paths. Run the corresponding commands with
`stemfusion separate htdemucs`.

Each dataset's `configs/experiments/<dataset>/separation.yaml` inherits this
preset and supplies local paths. FMA-Medium and GTZAN use directory discovery;
ISMIR04 uses the retained audio manifest. Model, shift, overlap, rescaling, and
MP3 settings are shared. ISMIR04 overrides process batching to 32 tracks and
accepts MP3/WAV/AU; these are input/scheduling differences.
