# FMA-Medium HT-Demucs MusicFM and RMS extraction

## Purpose and provenance

This workflow cleans the reusable behavior from the legacy FMA HT-Demucs
extractors:

- `/home/osiris181/htdemucs_experiment(FMA_HTDEMUCS)/fixed_get_musicfm_htdemucs.py`
- `/home/osiris181/htdemucs_experiment(FMA_HTDEMUCS)/get_musicfm_htdemucs_add_rms_features.py`

The cleaned workflow follows the submitted paper's common MusicFM
configuration: MusicFM-MSD, layer 7, 24 kHz mono audio, full-duration
inference, and temporal mean pooling. The original scripts and feature caches
remain unchanged.

Vanilla MusicFM and RMS are stored separately. A MusicFM cache contains one
1,024-dimensional embedding per track. An RMS cache contains one scaled scalar
per track. Weighted late fusion can combine them without changing the vanilla
feature representation.

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
has its mixture and all six stems. Both vanilla and RMS extraction must read
this same manifest.

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

Run a dry run for one source first:

    stemfusion extract htdemucs musicfm \
      --manifest /path/to/manifests/fma_htdemucs.jsonl \
      --source drums \
      --musicfm-root /path/to/musicfm \
      --output-dir /path/to/features/fma_htdemucs/musicfm \
      --limit-per-split 2 \
      --dry-run

Then remove `--dry-run` and `--limit-per-split`. Repeat for
`mixture`, `drums`, `bass`, `guitar`, `piano`, `vocals`, and
`other`. Extracting `other` preserves the complete separator output even
when a paper experiment selects only the five named stems.

Vanilla outputs follow this pattern:

    fma_musicfm_<source>_<split>.pkl

Each pickle maps the six-digit track ID to a 1,024-dimensional CPU float32
tensor.

## Step 3: extract RMS features

The RMS defaults in `configs/features/rms_mean_scaled20.yaml` reproduce the
legacy energy calculation:

| Setting | Value |
| --- | --- |
| Input | Mono, 24 kHz, full duration |
| Frame length | 2,048 samples |
| Hop length | 512 samples |
| Centered frames | Yes |
| Padding mode | Constant |
| Aggregation | Mean across RMS frames |
| Scaling factor | 20.0 |
| Output dimension | 1 |

Run:

    stemfusion extract htdemucs rms \
      --manifest /path/to/manifests/fma_htdemucs.jsonl \
      --source drums \
      --output-dir /path/to/features/fma_htdemucs/rms \
      --limit-per-split 2 \
      --dry-run

Remove the two test options for the complete extraction and repeat for every
source required by the RMS-weighted fusion experiment. Outputs follow this
pattern:

    fma_rms_<source>_<split>.pkl

Each value is a one-element float32 array containing
`mean(frame_rms) * 20.0`. RMS is deliberately not appended to the MusicFM
embedding.

## Alignment and failure behavior

Both extractors key caches by the same six-digit track IDs and split names.
A complete split cache is published only when every selected record succeeds.
If any record fails, successful values are written to a `.partial.pkl` file,
the failure is recorded in a JSON report, and the command exits nonzero.

Do not use `--overwrite` until the dry-run report, manifest counts, model
checkpoint, layer, and source have been checked. Keep MusicFM and RMS output
directories separate from legacy caches. See
`docs/fma_htdemucs_training.md` for the Table 3 and Table 4 classifiers that
consume these caches.
