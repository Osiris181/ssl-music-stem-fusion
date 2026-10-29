# FMA-Medium MusicFM extraction

## Purpose and provenance

The dataset task is implemented in
[experiments/fma_medium/extraction.py](../src/stemfusion/experiments/fma_medium/extraction.py).
It owns the FMA argument parser and configuration-to-extraction handoff,
then calls the shared MusicFM implementation in `features/_musicfm.py`.
The task can be run directly as a Python file, or through
`stemfusion extract fma-medium`. Both use the same parser and configuration.

From the repository root, direct execution is:

```bash
python src/stemfusion/experiments/fma_medium/extraction.py \
  --config configs/experiments/fma_medium/extraction.yaml \
  --manifest data/manifests/fma_medium_htdemucs.jsonl \
  --musicfm-root /path/to/musicfm
```

Direct file execution does not require setting `PYTHONPATH`. From another
working directory, use absolute paths for CLI arguments. Relative paths
inside YAML continue to resolve against their configuration file.

It automatically loads `configs/experiments/fma_medium/extraction.yaml`,
which references `configs/features/musicfm_msd_layer7.yaml`. Use
`--config` for a task/alternate YAML; explicit CLI values override YAML.
See [configuration usage](configuration.md) for profile references, relative
paths, and resolved-setting reports.

This workflow cleans the reusable behavior from the legacy FMA HT-Demucs
extractor:

- `/home/osiris181/htdemucs_experiment(FMA_HTDEMUCS)/fixed_get_musicfm_htdemucs.py`

The cleaned workflow follows the submitted paper's common MusicFM
configuration: MusicFM-MSD, layer 7, 24 kHz mono audio, full-duration
inference, and temporal mean pooling. The original scripts and feature caches
remain unchanged.

A MusicFM cache contains one 1,024-dimensional embedding per track. RMS
extraction and weighted fusion have been archived outside the active project;
see [the organization record](fma_medium_organization.md). Reuse existing
features only after provenance, dimensions, and track/split alignment checks.

## Inputs

The workflow requires:

- FMA `tracks.csv`;
- the original FMA-Medium audio root for the mixture;
- one or more HT-Demucs output roots;
- a local MusicFM checkout containing `msd_stats.json` and
  `pretrained_msd.pt`.

Each HT-Demucs root is the directory passed to
`stemfusion separate htdemucs --output-root`. It must therefore contain the
model directory:

    <htdemucs_root>/htdemucs_6s/000002/
    ├── bass.mp3
    ├── drums.mp3
    ├── guitar.mp3
    ├── other.mp3
    ├── piano.mp3
    └── vocals.mp3

Completed tracks may be distributed across disks. Repeat
`--htdemucs-root` in priority order. For each track, the manifest selects the
first root containing all six non-empty stems; it never combines a partial
track across disks.

## Step 1: build one aligned manifest

    stemfusion data fma-manifest \
      --tracks-csv /path/to/fma_metadata/tracks.csv \
      --mixture-root /path/to/fma_medium \
      --htdemucs-root /path/to/current_htdemucs_outputs \
      --htdemucs-root /path/to/ssd_htdemucs_outputs \
      --htdemucs-root /path/to/hdd_htdemucs_outputs \
      --allow-missing \
      --output /path/to/manifests/fma_htdemucs.jsonl

For a bounded smoke test, add one or more IDs without changing the official
split or label mapping:

    --track-id 089843 --track-id 089846

Without `--allow-missing`, any incomplete configured track is an error. With
it, the manifest keeps the aligned intersection only, so every retained track
has its mixture and all six native stems. The revised classifiers and feature
extraction use the mixture and the five selected stems, excluding `other`.

## Step 2: extract vanilla MusicFM features

The defaults in `configs/features/musicfm_msd_layer7.yaml` are:

| Setting | Value |
| --- | --- |
| Checkpoint | MusicFM-MSD (`pretrained_msd.pt`) |
| Statistics | `msd_stats.json` |
| Input | Mono, 24 kHz, full duration |
| Conformer layer | 7 |
| Temporal pooling | Mean |
| Output dimension | 1,024 |
| Flash attention | Disabled |

Run one command for the configured sources; begin with a dry run:

    stemfusion extract fma-medium \
      --manifest /path/to/manifests/fma_htdemucs.jsonl \
      --musicfm-root /path/to/musicfm \
      --output-dir /path/to/features/fma_medium/musicfm_msd_layer7 \
      --limit-per-split 2 \
      --dry-run

Remove `--dry-run` and `--limit-per-split` for the complete extraction.
The task reads `sources` from its YAML and processes mixture, guitar, drums,
piano, bass, and vocals automatically, with one encoder shared across them.
No shell loop is needed. Use `--source piano` for one source, or repeat
`--source` for a subset. The revised experiment excludes `other`.

Each source and split displays a tqdm progress bar with the processed track
count, speed, and estimated time remaining. Failed tracks count as processed;
error messages and JSON reports retain the failure details.

Vanilla outputs follow this pattern:

    fma_musicfm_<source>_<split>.pkl

Each pickle maps the six-digit track ID to a 1,024-dimensional CPU float32
tensor.

## Alignment and failure behavior

The extractor keys caches by six-digit track IDs and split names.
A complete split cache is published only when every selected record succeeds.
If any record fails, successful values are written to a `.partial.pkl` file,
the failure is recorded in a JSON report, and the command exits nonzero.

Multi-source extraction keeps individual source reports and adds
`extraction_report_musicfm_batch.json`. It stops after a failed source,
recording unprocessed sources. Rerunning skips complete split caches; partial
splits are recomputed. Dry runs and fully cached jobs do not load the model.

Do not use `--overwrite` until the dry-run report, manifest counts, model
checkpoint, layer, and source have been checked. Keep new output directories
separate from legacy caches. See [the training guide](fma_htdemucs_training.md)
for the active FMA-Medium classifier commands.
