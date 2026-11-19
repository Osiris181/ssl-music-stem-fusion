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
| `extraction_cnn.py` | Extract the Phase 3 frozen MSD_musicnn embeddings with the same backend and window policy as GTZAN. |
| `train_mid.py` | Train the mixture baseline, three capacity controls, and three stem-fusion conditions. |
| `train_mid_cnn.py` | Train the seven CNN mid-fusion conditions using the existing genre trainer. |
| `train_late.py` | Train independent branches, evaluate matched ensembles, and report standalone stems. |
| `train_late_cnn.py` | Reuse CNN-M, train independent CNN branches, and report matched ensembles and standalone stems using the same workflow as FMA and GTZAN. |

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

## Phase 3 CNN extraction and training

All CNN datasets share [guarded decoding and cache recovery](../../../../../docs/cnn_audio_decoding.md).
Invalid primary decodes receive a checked FFmpeg fallback. Incomplete tracks
have their reference revalidated, and wrong-length caches are repaired with
backups. Valid caches and the 0.25-second alignment tolerance remain unchanged.

The CNN experiment reuses `data/manifests/ismir04/ismir04_htdemucs.jsonl`,
including its unique IDs, existing source paths, six-class mapping and fixed
729/726/724 partitions. Reuse the prepared manifest and separated audio.

The public scripts call the same `_cnn_extraction.py` and `_cnn_genre.py`
orchestration as GTZAN. The frozen encoder and window pooling live in
`features/musicnn.py`. The ISMIR04 cache adapter preserves the CSV genre map
and feeds the existing genre models, trainer, metrics and reporting.

Install/activate the isolated TensorFlow extraction environment using the
[environment instructions](../../../../../environments/musicnn/README.md).
From the repository root, activation is `source musicnn/bin/activate`.
Then, from this ISMIR04 experiment directory:

```bash
python extraction_cnn.py --dry-run
python extraction_cnn.py
```

`configs/experiments/ismir04/extraction_cnn.yaml` defines the local manifest,
cache/report paths and execution settings. YAML paths resolve relative to the
declaring YAML, while CLI paths resolve relative to the current directory.
Compatibility setup for legacy Keras and environment-local CUDA libraries runs
automatically. No separate launcher is needed.

The extractor uses the same official standard MSD_musicnn checkpoint as GTZAN:
mono 16 kHz, official log-mel preprocessing, consecutive 187-frame windows with
no overlap, and 200-dimensional penultimate embeddings. It retains and pads
the final partial log-mel window, then averages all window embeddings equally.
This differs from the released extractor's tail-dropping behavior and is the
same adaptation in both CNN datasets. No duration truncation is applied.

Sources are trimmed/padded in memory to the mixture's resampled length within
the shared 0.25-second tolerance; larger discrepancies fail extraction. This
aligns window boundaries and does not estimate waveform latency. Silent vectors
are retained. Long recordings use a full-track spectrogram and batches of at
most 16 windows for CNN inference. The window sums are divided by the total
window count, so the smaller final batch gets the correct weight. Unlike the
MusicFM configuration, no 30-second encoder chunk policy is needed here.
CPU preprocessing still needs memory for the full waveform and spectrogram.
The YAML GPU memory cap is 4096 MB; `--gpu-memory-mb N` overrides it.

Features save as `data/features/ismir04/msd_musicnn/{split}/{source}/{track_id}.npz`.
Each file contains a float32 `(200,)` embedding and JSON provenance covering
the dataset/split/label, checkpoint and source hashes, audio hash, frame counts
and padding. The final readable counts point to
`outputs/ismir04/cnn/extraction_report.json`. A failed view produces an error
and a nonzero exit status; training requires the complete manifest cohort.
Reruns reuse matching caches. Use `--overwrite` explicitly to replace them.

For a bounded check without writing full-experiment caches:

```bash
python extraction_cnn.py --limit-per-split 1 \
  --output-dir /tmp/ismir04_cnn_check/features \
  --report /tmp/ismir04_cnn_check/extraction_report.json --fail-fast
```

After full extraction finishes, activate your main PyTorch training environment
and run from this directory:

```bash
python train_mid_cnn.py --validate-only
python train_mid_cnn.py --dry-run
python train_mid_cnn.py
```

`configs/experiments/ismir04/mid_fusion_cnn.yaml` inherits the active ISMIR04
`mid_fusion.yaml` protocol and shared `configs/training/genre.yaml` settings:
1024/512 hidden layers, ReLU, dropout 0.2, unweighted cross-entropy, Adam 1e-4,
batch 128, 20 epochs and no early stopping. The best checkpoint maximizes the
validation sum of accuracy, macro ROC-AUC and macro AP; ties retain the earlier
epoch. Test evaluation follows checkpoint selection.

The five seeds remain 32, 64, 128, 256 and 512. The seven conditions are mixture
only; repeated mixture x4/x5/x6; and M+G+D+P, then +B, then +V. Input widths are
200/800/1000/1200 and the output has six classes. Corresponding repeated-mixture
and stem-fusion MLPs have identical nominal parameter counts. The mid-fusion
command covers these seven conditions; CNN late fusion uses the separate
command below. Neither CNN experiment adds harmonic features.

Training reads caches and the manifest; it checks exact IDs, labels, finite
vectors, extraction provenance and aligned source windows. CNN checkpoints and
per-run results save under `outputs/ismir04/cnn/runs/`. The final terminal table
also saves as `outputs/ismir04/cnn/summary_mid_fusion.txt` and `.json`, with
mean/sample SD and paired fusion-minus-control differences. Selected runs or
seed subsets use distinct summary names. Reruns reuse verified complete fits;
`--overwrite` retrains them. CLI training overrides match the GTZAN CNN script.

MSD pretraining overlap with the evaluation collections remains unresolved;
the CNN comparison does not by itself isolate pretraining objective from other
representation and architecture differences.

### CNN late-fusion training

Use the main PyTorch training environment and run from this ISMIR04 directory:

```bash
python train_late_cnn.py --dry-run
python train_late_cnn.py --validate-only
python train_late_cnn.py
```

The entry point uses the same `_cnn_genre.py` controller, `_genre.py` scheduler,
and shared genre branch fitter and ensemble evaluator as FMA and GTZAN.
`configs/experiments/ismir04/late_fusion_cnn.yaml` inherits ISMIR04's
MusicFM `late_fusion.yaml` and shared genre profile. Each branch is
**200 -> 1024 -> 512 -> 6**, with biased linear layers, ReLU, dropout 0.2,
unweighted cross-entropy, Adam 1e-4, batch 128, and 20 epochs. Select each
checkpoint independently using the same validation sum of accuracy, macro
ROC-AUC, and macro AP, retaining the earliest exact tie as in the other genre experiments.
There is no early stopping. Preserve the existing six-class mapping and
729/726/724 partitions.

For outer seeds 32, 64, 128, 256, and 512, reuse CNN-M as M1 from
`outputs/ismir04/cnn/runs/branches/seed_N/mixture_1/`. Independently train
M2–M6 on the same mixture embedding and G/D/P/B/V on their respective
embeddings using the same position-paired SHA256 seeds as FMA and GTZAN.
The full matrix has **50 new branch fits**. Reuse each branch for nested
four-, five-, and six-member mixture and source ensembles, adding raw
logits equally before softmax/argmax. Full runs include all five standalone
stem results from the selected branch checkpoints without extra fitting.

New checkpoints and ensemble predictions save beneath
`outputs/ismir04/cnn/late_fusion/runs/`. The terminal and
`outputs/ismir04/cnn/summary_late_fusion.{txt,json}` include the M1 reference,
six ensembles, five standalone stems, and paired differences with sample SD.
Companion `summary_stem_only.{txt,json}` files save in the same summary
directory. Existing mid-fusion checkpoints and summaries remain unchanged.

`--validate-only` checks the cache and all requested M1 references without
training or writing outputs. `--run NAME` and repeated `--seed N` select
subsets with distinct summary suffixes. Reruns reuse verified completed
fits; `--overwrite` affects late outputs only. `--reference-output-dir`
selects a completed CNN mid-run root and `--output-dir` selects the new
late-run root; the roots must be separate and non-nested. An explicit output
root redirects summaries unless `--summary-dir` is supplied. A missing or
incompatible M1 reference stops the affected run.

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
mixture baseline and three repeated-mixture MLPs (4/5/6 identical mixture
vectors, unchanged 1024/512 hidden widths, exact fusion parameter counts).
Late fusion sums
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
