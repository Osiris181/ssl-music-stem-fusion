# PMEmo feature preparation

These commands prepare frozen MERT and BTC/music21 harmonic features for
PMEmo's chorus audio. They do not train classifiers, create binary labels,
resplit data, or change previous features in `Music2Emotion/dataset/pmemo/`.

## Files and configuration

| File | Purpose |
| --- | --- |
| `prepare_feature_manifest.py` | Preserve supplied train/validation/test IDs, validate metadata and bind mixture/stem paths. |
| `extract_mert.py` | Extract all 12 frozen MERT transformer layers for every retained 30-second segment. |
| `extract_chords_keys.py` | Extract whole-track BTC chords, generate intermediate MIDI, estimate music21 key and normalize chords independently per source. |
| `_extraction.py` | Route dataset defaults and command-line arguments to the feature implementations. |

Defaults are in `configs/experiments/pmemo/{feature_manifest,mert,chords_keys}.yaml`.
`mert.yaml` inherits the same `configs/features/mert_v1_95m.yaml` as Jamendo.
Paths resolve relative to their YAML file; command-line arguments override
YAML. Other users must provide their own audio/model locations and metadata.

For example, `../../../data/manifests/pmemo/split/train.txt` in
`configs/experiments/pmemo/feature_manifest.yaml` resolves to this repository's
`data/manifests/pmemo/split/train.txt`. It refers to the workspace copy, not
directly to `Music2Emotion/dataset/pmemo/meta/split/train.txt`. Changing the
original Music2Emotion file later does not automatically update that copy.

The supplied split files are from
`Music2Emotion/dataset/pmemo/meta/split/{train,validation,test}.txt`, copied to
`data/manifests/pmemo/split/`. Preserve these assignments; do not generate a
new split or describe these as an independently verified official split.
The current files contain 536/116/115 tracks (767 total), without overlap.

The manifest uses copies of the original
`PMEmo2019/annotations/static_annotations.csv` and `PMEmo2019/metadata.csv`
at `data/manifests/pmemo/`. Original valence/arousal scores remain on the
0-1 scale. The historical Music2Emotion annotation copy instead uses 1-9;
this does not affect audio extraction, but binary label thresholds and score
scaling must be checked explicitly before implementing training. Both targets
will reuse the same extracted features; do not extract once per target.
Local metadata and generated manifests remain ignored by Git.

Default input roots:

```text
/mnt/HDD/PMEmo2019/chorus/<id>.mp3
/mnt/HDD/osiris181/pmemo_separated/htdemucs_6s/<id>/<source>.mp3
```

The manifest binds all six native separator stems; extraction selects mixture,
guitar, drums, piano, bass and vocals, excluding `other`. Preparation reads
metadata only. Extraction reports missing/undecodable audio; it does not
silently remove tracks or change split membership.

## Consistency with Jamendo

- Same pinned MERT-v1-95M revision, frozen evaluation-mode float32 inference,
  mono 24 kHz, processor normalization and temporal mean pooling.
- Same 30-second non-overlapping segments, retaining every nonempty tail.
  Only tails below 400 samples are zero-padded to 400 for the encoder.
- Same mixture-duration stem alignment, with end trim/padding allowed only
  within 0.25 seconds; larger differences are reported as failures.
- Same `(1, 12, 768)` float32 arrays. Save all layers now; the eventual
  classifier selects the agreed saved-layer indices 5 and 6.
- Same verified BTC utilities/checkpoint, released MIDI policy, music21 key
  estimation and C-major/A-minor chord normalization. These descriptors use
  each source's own complete waveform, not the mixture descriptors copied
  across stems. Unavailable keys are recorded, not replaced with invented keys.
- Same default-on second pass for recognized tiny CQT-tail failures only;
  no fixed failing-track list. Use `--no-retry-failures` to disable it.

`features/emotion_extraction.py` supplies PMEmo-aware scheduling and reports.
It calls the exact existing MERT/BTC runtimes and per-view writers rather than
forking the signal-processing/model algorithms. Its schedule is regression
checked against Jamendo, apart from dataset identity and added provenance.
The fingerprinted Jamendo modules stay unchanged, preserving existing caches
and the in-progress SheetSage2 extraction. PMEmo records its adapter/manifest
reader checksums too; incompatible or corrupted caches are not silently reused.

The older extractor skips tails shorter than one second and its saved arrays
alone do not establish the new alignment/cache provenance. New extraction
therefore uses separate output directories, leaving old evidence intact.

## How to run

Use the existing MERT/BTC environment, not the separate SheetSage2 environment.
Wait for the current GPU extraction to finish before starting another CUDA job.
The project extras are `mert` and `chords-keys`; no installation is performed
by these scripts. The pinned MERT snapshot and verified BTC assets are already
available locally in this workspace. `--help` lists all overrides.

```bash
source /home/osiris181/.venv/bin/activate
cd /home/osiris181/stem-fusion-music-classification/src/stemfusion/experiments/pmemo

python prepare_feature_manifest.py --dry-run
python prepare_feature_manifest.py
python extract_mert.py --dry-run
python extract_chords_keys.py --dry-run
```

The workspace manifest has already been prepared. Existing manifests are
protected; skip the non-dry preparation command unless creating it for the
first time. Use `--overwrite` explicitly only when regenerating the manifest.

First test one training track (six views), then run the full extraction:

```bash
python extract_mert.py --split train --limit-per-split 1 --fail-fast
python extract_chords_keys.py --split train --limit-per-split 1 --fail-fast

python extract_mert.py
python extract_chords_keys.py
```

The full run reuses verified smoke-test outputs. Rerunning without `--overwrite`
verifies completed outputs and processes incomplete views. Settings, package,
model or audio-identity changes can invalidate reuse; do not bypass such errors
or change output contents to make their signatures appear compatible.
With `--fail-fast`, the chord/key run stops at its first failure instead of
completing the normal pass and running the automatic retry phase.

Equivalent installed commands:

```bash
stemfusion data pmemo-feature-manifest --dry-run
stemfusion extract pmemo mert --dry-run
stemfusion extract pmemo chords-keys --dry-run
```

## Outputs and reports

```text
data/manifests/pmemo/
  pmemo_features.jsonl
  pmemo_features.report.json
data/features/pmemo/
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
outputs/pmemo/
  mert_extraction_report.json
  chords_keys_report.json
  chords_keys_report.initial.json  # only when an automatic retry pass occurs
```

Each MERT track folder also has `segments.json` and `run_manifest.json` with
true boundaries, padding/alignment metadata, settings and file checksums.
Both commands show tqdm and a final console summary; JSON reports retain
per-view outcomes and failures. Dry runs only plan: they do not decode audio,
initialize models, write features or replace saved extraction reports. A
partial smoke-test report is replaced by the later full-run report, while
valid per-track features are reused.
