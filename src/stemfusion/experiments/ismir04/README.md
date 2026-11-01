# ISMIR04 genre classification

These scripts use the same shared genre orchestration, MLPs, trainer,
checkpoint selection, evaluation, and reporting as FMA-Medium and GTZAN.
Dataset adapters handle the six CSV labels, fixed partitions, unique IDs,
and cache filenames. All training hyperparameters come from
`configs/training/genre.yaml`.

| File | Purpose |
| --- | --- |
| `prepare_manifest.py` | Preserve provided CSV partitions and labels, audit duplicates, and bind retained tracks to audio paths. |
| `separation.py` | Generate missing stem sets with the shared HT-Demucs backend. |
| `extraction.py` | Extract shared MusicFM-MSD layer-7 embeddings with explicit long-track chunking. |
| `train_mid.py` | Train the mixture baseline, three capacity controls, and three stem-fusion conditions. |
| `train_late.py` | Train independent branches, evaluate matched ensembles, and report standalone stems. |

`prepare_manifest.py` reads the provided CSVs, checks for exact audio
duplicates, and generates an aligned six-class manifest. It never creates a
random split or changes the supplied CSV/audio files.

From this directory:

```bash
python prepare_manifest.py
```

The same command is available as `stemfusion data ismir04-manifest`.
Settings and local paths are loaded from
`configs/experiments/ismir04/manifest.yaml`. Explicit CLI options override
YAML. YAML paths resolve relative to their declaring file.

## Inputs and labels

The CSVs are `metadata/{training,development,evaluation}/tracklist.csv`.
They are interpreted as train, validation, and test, respectively. The first
CSV column supplies the six-class genre label. The sixth column supplies the
relative audio path; evaluation has an additional trailing field that is
not a filename. Directory names do not determine labels.

The fixed mapping is classical=0, electronic=1, jazz_blues=2, metal_punk=3,
rock_pop=4, world=5. Track IDs combine the provided split, a filename prefix,
and a SHA256 suffix derived from the full relative source path. Recordings
with the same basename therefore remain distinct.

## Duplicate audit and exclusions

The audit reads only paths listed by the CSVs. It records SHA256 hashes of
the files and of ffmpeg-decoded, mono 24 kHz float32 audio. Exact decoded
model-input matches are grouped. Retain training before validation before
test; ties within a partition use the relative path. Conflicting genre
labels for identical audio cause an error instead of an automatic label
choice. Retained tracks keep their original partition assignments.

This checks exact duplicates, including files whose non-audio contents may
differ. It is not a comprehensive check for re-encodings, near duplicates,
or artist overlap. Decoder messages are retained in the audit for inspection.

For the local CSVs, 2,187 records were audited. Eight duplicate records were
excluded, leaving **729 train, 726 validation, and 724 test**. All six classes
remain in each partition. Three explicit filename/directory corrections in
YAML resolve existing audio paths without dropping their records or editing
the original files.

The full audit takes several minutes on HDD storage. Progress is saved, and
unchanged files are reused on subsequent runs. `--audit-only` writes/reuses
the audit and prints the selection report without publishing a manifest.
ffmpeg must be installed. Use `--overwrite` explicitly to replace an existing
manifest and sidecars.

## Outputs and separated audio

Generated local files are in `data/manifests/ismir04/` and remain ignored:

| File | Contents |
| --- | --- |
| `audio_duplicate_audit.json` | File hashes, decoded-audio hashes, decoder messages, and completion status. |
| `ismir04_htdemucs.jsonl` | Retained IDs, six-class labels, existing split assignments, and audio paths. |
| `ismir04_htdemucs.metadata.json` | Per-record CSV identities and resolved source paths. |
| `ismir04_htdemucs.report.json` | All exclusions, corrections, counts, and required stem repairs. |
| `source_collision_audit.json` | Earlier four-record waveform verification for the colliding basenames. |

Existing separated stems are reused by their original basename. For
colliding names, `verified_legacy_stems` identifies a source whose existing
stems have been independently checked. Other colliding sources are routed
to unique folders under `repair_root` and listed in `required_stem_repairs`.
No separation is launched by manifest preparation.

`separation.py` loads `configs/experiments/ismir04/separation.yaml` and uses
the same shared HT-Demucs implementation as FMA-Medium and GTZAN:

```bash
python separation.py --dry-run
python separation.py
```

It reads the retained manifest and checks the exact referenced stem folders,
including all six native outputs. Complete folders are reused; only missing
or incomplete targets are separated. Per-record target paths must fall under
the configured output root. Temporary input symlinks give Demucs the intended
folder names, so different recordings with the same original filename stay
distinct. These links are cleaned after execution, and originals are not
copied or renamed. Dry runs do not stage files, run Demucs, or write reports.
The common inference settings are unchanged; ISMIR04 uses batches of 32.
Its report defaults to `outputs/ismir04/separation_report_htdemucs.json`.

The existing `6-fantasia_6` stems match the Jacob Heringman/Catherine recording.
The Edward Martin performance uses a separate stem set under the manifest's
unique ID. Separation creates that set when missing and reuses it when complete.
The evaluation copy of `7-stick_dance` is excluded because its audio is
identical to the retained training copy.

## MusicFM extraction

`extraction.py` uses the shared CLI and MusicFM runtime used by FMA-Medium
and GTZAN. It loads `configs/experiments/ismir04/extraction.yaml`, referencing
`configs/features/musicfm_msd_layer7.yaml` for the checkout, MSD checkpoint,
statistics, layer index 7, 24 kHz mono audio, and 1024-dimensional embeddings.

Once the referenced stem sets are complete, run from this directory:

```bash
python extraction.py --dry-run
python extraction.py
```

The unified command is `stemfusion extract ismir04`. Sources are processed
in order: mixture, guitar, drums, piano, bass, vocals, with one frozen model
shared across them. Legacy feature directories remain unchanged. New outputs:

```text
data/features/ismir04/musicfm_msd_layer7/
├── ismir04_musicfm_mix_{train,validation,test}.pkl
├── ismir04_musicfm_{stem}_{train,validation,test}.pkl
├── extraction_report_musicfm_{source}.json
└── extraction_report_musicfm_batch.json
```

Each pickle maps unique manifest IDs to CPU float32 tensors of shape `(1024,)`.
No RMS scalar is appended. Canonical splits are train/validation/test, with
729/726/724 retained tracks for every selected source.

### Long-track policy

The full recording is retained (`max_seconds: null`). ISMIR04 explicitly
sets `chunk_seconds: 30` to bound encoder memory. The shared runtime divides
long recordings into balanced, non-overlapping chunks no longer than 30 s,
without dropping a final tail. It sums latent frames and divides by the total
frame count, rather than assigning equal weight to chunks of unequal lengths.

Encoder context resets at chunk boundaries; this is not claimed to equal
one full-track forward pass numerically. Reports record the policy, which is
applied identically to every ISMIR04 source. FMA-Medium and GTZAN retain their
existing single-pass defaults (`chunk_seconds: null`); their cached features
and completed training results are unchanged.

Use `--chunk-seconds N` to change the bound, or `--no-chunking` for one forward
pass. Keep every compared source cache under the same extraction policy.
Use a different `--output-dir` when intentionally changing settings.

### Bounded checks and restart behavior

For a small real check across three splits and six sources:

```bash
python extraction.py --limit-per-split 1 --output-dir /tmp/ismir04_musicfm_smoke
```

Keep bounded output separate from full caches. Existing non-empty final cache
files are skipped unless `--overwrite` is supplied; this skip check is not a
complete cache-content audit. Failed splits produce partial caches and failure
reports instead of publishing complete split caches. tqdm displays source/split
progress; per-source/aggregate JSON reports record settings, paths, counts,
and errors.

## Training configuration and commands

| YAML | Purpose |
| --- | --- |
| `configs/training/genre.yaml` | Common architecture, optimization, checkpoint selection, evaluation, and seeds for all three genre datasets. |
| `configs/training/ismir04_genre.yaml` | Six classes and local manifest, feature, run, and summary paths. |
| `configs/training/ismir04_mid_fusion.yaml`, `ismir04_late_fusion.yaml` | The same concatenation and independent-logit conventions as the other genre datasets. |
| `configs/experiments/ismir04/mid_fusion.yaml`, `late_fusion.yaml` | The same seven mid-fusion and six late-fusion conditions. |

With a prepared manifest and complete features, run from this directory:

```bash
# Validate feature contents and extraction reports without training or writing outputs.
python train_mid.py --validate-only

# Inspect conditions, paths, settings, and branch seeds without training.
python train_mid.py --dry-run
python train_late.py --dry-run

# Full five-seed matrices, including evaluation of the selected checkpoints.
python train_mid.py
python train_late.py
```

The unified commands are `stemfusion train ismir04 mid-fusion` and
`stemfusion train ismir04 late-fusion`. Each entry point automatically loads
its experiment YAML. YAML paths resolve relative to their declaring files;
explicit CLI arguments override YAML. Use `--config PATH` for another preset,
`--seed 32` for one outer seed, or `--run NAME` for one condition.
Prepared manifests and features are reused; training never reads audio.

`data/ismir04_features.py` accepts NumPy arrays and PyTorch tensors and
aligns every source to manifest ID order. It requires exact ID sets, finite
float32 vectors of width 1024, and per-source reports describing MSD layer 7,
24 kHz input, no duration truncation, and the agreed balanced 30 s chunk policy.
Valid all-zero vectors remain in the common track set; no additional
normalization is applied. The legacy ISMIR04 caches are not imported.
Manifest, cache, and report checksums are recorded in the training data audit.

## Training protocol and outputs

All genre datasets use the same standard MLP with 1024/512 hidden units,
ReLU, dropout 0.2, cross-entropy, Adam at 1e-4, batch size 128, and 20 epochs.
There is no early stopping, scheduler, or weight decay. Checkpoint selection
maximizes validation micro accuracy + macro ROC-AUC + macro average precision
on the 0–1 scale, retaining the earliest epoch on ties. Test scores are
calculated from the selected weights and never select checkpoints.
Outer seeds are **32, 64, 128, 256, 512**; branch positions and derived seeds
use the existing shared SHA256 convention.

Mid fusion compares M+G+D+P, M+G+D+P+B, and M+G+D+P+B+V against the standard
mixture baseline and three capacity-matched mixture MLPs. Late fusion sums
independently selected branch logits equally, then applies softmax, and
compares the stem ensembles against 4/5/6-member mixture ensembles.
The mixture baseline is shared between workflows. Both full matrices need
**85 distinct fits across five seeds**. Standalone guitar, drums, piano,
bass, and vocals results reuse the selected late-fusion branch models.

Run artifacts go in `outputs/ismir04/runs/`. Selected checkpoints are:

```text
branches/seed_<seed>/<branch>/checkpoint.pt
mid_fusion/<condition>/seed_<seed>/checkpoint.pt
```

Late-fusion ensemble records reference branch checkpoints. Compatible
completed fits are verified and reused automatically, so either training
script may run first. `--overwrite` explicitly replaces existing fits.
Use a separate output directory when changing the experimental settings.

Results print to the terminal and are saved directly in `outputs/ismir04/`:

```text
summary_mid_fusion.txt / .json
summary_late_fusion.txt / .json
summary_stem_only.txt / .json
```

TXT tables report means and sample SD across the five seeds, plus paired
differences against matched controls. JSON retains detailed per-seed results.
Both summary formats can be tracked by Git; checkpoints and prediction arrays
remain ignored. The late-fusion summary includes the five standalone-stem
results and also writes their separate supporting summary. Rebuild these
from saved branches without training, feature loading, or inference with:

```bash
python train_late.py --report-stems-only
```
