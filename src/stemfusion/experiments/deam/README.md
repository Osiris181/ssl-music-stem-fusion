# DEAM feature preparation

Prepare frozen MERT embeddings and BTC/music21 chord/key descriptors using
the same scientific extraction settings as Jamendo and PMEmo. These scripts
do not train classifiers, binarize emotion labels or replace old features.

## Files and inputs

| File | Purpose |
| --- | --- |
| `prepare_feature_manifest.py` | Preserve supplied split IDs and song-level valence/arousal scores; bind mixture/stem audio paths. |
| `extract_mert.py` | Save all 12 frozen MERT transformer layers for every retained 30-second segment. |
| `extract_chords_keys.py` | Extract whole-track BTC chords, generated MIDI, music21 key and normalized chords independently for each source. |

Presets are in `configs/experiments/deam/`. All relative YAML paths resolve
against the YAML file's parent directory, not the shell's working directory.
CLI arguments override YAML. For example, from a preset in
`configs/experiments/deam/`, `../../../data/manifests/deam/split/train.txt`
points to this repository's `data/manifests/deam/split/train.txt`. It does not
point directly into Music2Emotion. The workspace contains copied inputs from:

```text
Music2Emotion/dataset/deam/meta/split/train.txt
Music2Emotion/dataset/deam/meta/split/validation.txt
Music2Emotion/dataset/deam/meta/split/test.txt
Music2Emotion/dataset/deam/meta/static_annotations.csv
```

The supplied splits contain 1,261/271/270 tracks, without overlap, covering
all 1,802 supplied annotation records. IDs and ordering are preserved; no
resplitting or automatic audio-availability filtering occurs. These are the
provided Music2Emotion splits, not independently verified official DEAM splits.

The manifest preserves the supplied `valence_mean` and `arousal_mean` on
their 1-9 scale. It does not recompute means from dynamic annotations or
perform label normalization. Binary thresholds and target definitions must
be verified when implementing training. The same audio features will serve
both valence and arousal; do not extract once per target.

Default audio locations:

```text
/mnt/HDD/DEAM/MEMD_audio/<id>.mp3
/mnt/HDD/osiris181/DEAM_separated/htdemucs_6s/<id>/<source>.mp3
```

Other users must provide their own split/annotation files and edit audio/model
paths. Local metadata and generated manifests remain ignored by Git. The
manifest builder only reads metadata; full source decoding occurs during
extraction, where failures are reported instead of silently changing splits.

## Shared extraction rules

- Pinned frozen MERT-v1-95M, mono 24 kHz, processor normalization, float32
  evaluation-mode inference and temporal mean pooling.
- Non-overlapping 30-second segments, retaining all nonempty tails. Only
  encoder inputs shorter than 400 samples are right-zero-padded to 400.
- Stems aligned to decoded mixture duration; end trim/padding is allowed
  only within 0.25 seconds. Larger differences are reported as failures.
- All 12 transformer layers saved as `(1, 12, 768)` float32 arrays. Select
  the agreed saved-layer indices 5/6 during later classifier loading.
- Same BTC model/utilities, released MIDI conversion policy, music21 key
  estimation and C-major/A-minor chord normalization as Jamendo/PMEmo.
- Independent whole-track harmonic extraction for mixture and each stem;
  mixture descriptors are not copied into stem branches. Unavailable keys
  remain explicitly recorded, rather than invented or silently repaired.
- Default-on second pass for recognized tiny CQT-tail failures only; no
  hardcoded failing-track list. `--no-retry-failures` disables that retry.

`experiments/_emotion_features.py` handles CLI/preset routing.
`features/emotion_dataset.py` supplies dataset-neutral scheduling with an
explicit DEAM manifest adapter. It calls the unchanged existing MERT/BTC
runtimes and per-view cache writers. Scheduling and settings parity are tested
against the frozen PMEmo adapter (itself tested against Jamendo); no new neural
or signal-processing implementation is introduced. Existing PMEmo/Jamendo
modules are not modified while their extraction jobs run. The DEAM adapter
and manifest reader have their own recorded source checksums.

## How to run

Use the existing MERT/BTC environment, not the SheetSage2-specific environment.
Wait for the current CUDA jobs to finish before starting another GPU extraction.
Inspect options with `--help`; `--source` and `--split` can be repeated.

```bash
source /home/osiris181/.venv/bin/activate
cd /home/osiris181/stem-fusion-music-classification/src/stemfusion/experiments/deam

python prepare_feature_manifest.py --dry-run
python prepare_feature_manifest.py
python extract_mert.py --dry-run
python extract_chords_keys.py --dry-run
```

The local feature manifest has already been generated during preparation.
Skip the non-dry preparation command unless generating it for the first time.
Existing manifests are protected; use `--overwrite` explicitly only when
regenerating them.

Test one training track (six views), then run the full extraction:

```bash
python extract_mert.py --split train --limit-per-split 1 --fail-fast
python extract_chords_keys.py --split train --limit-per-split 1 --fail-fast

python extract_mert.py
python extract_chords_keys.py
```

The full commands reuse valid smoke-test outputs. Reruns without `--overwrite`
verify and reuse completed caches; incompatible settings, changed inputs or
corrupt files are reported, not silently accepted. `--fail-fast` stops after
the first failure instead of finishing the normal pass and performing retries.
Normal full runs can report genuine audio failures; metadata tests are not a
guarantee that every waveform decodes successfully.

Equivalent installed commands:

```bash
stemfusion data deam-feature-manifest --dry-run
stemfusion extract deam mert --dry-run
stemfusion extract deam chords-keys --dry-run
```

## Saved features and reports

```text
data/manifests/deam/
  deam_features.jsonl
  deam_features.report.json
data/features/deam/
  mert_v1_95m/
    mert_30s/<id>/segment_<index>.npy
    mert_30s_<source>/<id>/segment_<index>.npy
  harmony/<source>/<id>/
    chords.lab
    chords.mid
    key.lab
    normalized_chords.lab
    metadata.json
    run_manifest.json
outputs/deam/
  mert_extraction_report.json
  chords_keys_report.json
  chords_keys_report.initial.json  # only when automatic CQT retries occur
```

Extraction selects mixture, guitar, drums, piano, bass and vocals. `other`
is present in the manifest but excluded from the default feature set. Each
MERT view also saves `segments.json` and `run_manifest.json` with real segment
boundaries, minimum padding/alignment information and file checksums.

Commands display tqdm and a final console summary; reports retain per-view
outcomes and failures. Dry runs do not decode audio, initialize models or
write extraction outputs/reports. A later full-run report replaces its earlier
smoke-test report while valid per-view files are reused. Previous features in
`Music2Emotion/dataset/deam/` remain untouched.
