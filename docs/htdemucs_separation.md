# HT-Demucs separation

## Provenance

The core legacy implementation is:

- /home/osiris181/htdemucs_separation.py

Two additional scripts continued the same separation job when the original
output disk ran out of capacity:

- /home/osiris181/htdemucs_separation_ssd.py
- /home/osiris181/htdemucs_separation_new_hdd.py

They are storage-management variants, not different methods or experiments.
The ISMIR04 script applies the same method to three input folders and additional
audio formats:

- /home/osiris181/ismir_htdemucs_separation.py

All four cases are therefore represented by one implementation in
src/stemfusion/separation/htdemucs.py. Different disks are supplied as
read-only existing-output roots, while one output root receives new files. The
original files remain unchanged until output equivalence has been checked.

## Reproduction settings

The paper used Demucs 4.0.1 with:

| Setting | Value |
| --- | --- |
| Model | htdemucs_6s |
| Device | CUDA |
| Shifts | 1 |
| Overlap | 0.25 |
| Chunked separation | enabled |
| Clip mode | rescale |
| Output format | MP3 |
| MP3 bitrate | 320 kbps |
| MP3 preset | 2 |
| Worker jobs | 0 |

All six outputs are retained: bass, drums, guitar, other, piano, and vocals.
Classifier experiment configurations will determine whether the other stem is
used as an input. Separation itself must not discard it.

The normal batch size is 128 tracks. The ISMIR04 script used 32 tracks because
it scans three input roots and accepts MP3, WAV, and AU files. Batch size affects
process scheduling rather than model inference parameters.

## Installation

Install the package with its pinned HT-Demucs dependency:

    python -m pip install -e ".[htdemucs]"

The existing local environment containing Demucs 4.0.1 can also be selected
with --python-executable.

## Usage

Run a dry run first:

    stemfusion separate htdemucs \
      --input-root /path/to/fma_medium \
      --output-root /path/to/fma_separated_htdemucs \
      --python-executable /path/to/demucs/environment/bin/python \
      --dry-run

ISMIR04 can supply all three roots and formats to the same command:

    stemfusion separate htdemucs \
      --input-root /path/to/ismir04/audio/training \
      --input-root /path/to/ismir04/audio/development \
      --input-root /path/to/ismir04/audio/evaluation \
      --output-root /path/to/ismir04_separated_htdemucs \
      --extension .mp3 \
      --extension .wav \
      --extension .au \
      --batch-size 32 \
      --dry-run

When completed outputs are distributed across disks, each old location can be
checked without writing to it:

    --existing-output-root /path/to/old_hdd_outputs \
    --existing-output-root /path/to/ssd_outputs

New stems are written only to --output-root.

## Output and resume behavior

For an input named 089843.mp3, Demucs writes:

    <output_root>/htdemucs_6s/089843/
    ├── bass.mp3
    ├── drums.mp3
    ├── guitar.mp3
    ├── other.mp3
    ├── piano.mp3
    └── vocals.mp3

A track counts as complete only when all six expected files exist and are
non-empty. This is stricter than the legacy scripts, which counted arbitrary
MP3 files.

If a batch fails or returns incomplete tracks, already completed tracks are
recognized and the remaining tracks are retried individually. Failures are
recorded in separation_report_htdemucs.json, and the command exits nonzero if
any track remains incomplete.

Demucs places tracks according to the input filename without its extension.
The wrapper therefore rejects duplicate filename stems across input roots to
prevent two different tracks from writing to the same directory.
