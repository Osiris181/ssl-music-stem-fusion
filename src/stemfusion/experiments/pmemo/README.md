# PMEmo audio and feature preparation

These commands prepare separation jobs, frozen MERT and BTC/music21 harmonic
features for PMEmo's chorus audio, then train separate binary valence/arousal
classifiers. Preparation/extraction does not create labels; training applies
the fixed midpoint rule below. Commands do not resplit data or change previous
features in `Music2Emotion/dataset/pmemo/`.

## Files and configuration

| File | Purpose |
| --- | --- |
| `prepare_manifest.py` | Default `--purpose features`: preserve supplied splits/scores and bind existing audio. `--purpose separation`: prepare label-independent separation jobs from the same split IDs. |
| `separation.py` | Run the shared HT-Demucs backend using the separation manifest; reuse complete existing six-stem outputs unless explicitly overwritten. |
| `extract_mert.py` | Extract all 12 frozen MERT transformer layers for every retained 30-second segment. |
| `extract_chords_keys.py` | Extract whole-track BTC chords, generate intermediate MIDI, estimate music21 key and normalize chords independently per source. |
| `train_mid.py` | Thin launcher for both binary targets, seven conditions and five seeds through the shared emotion controller and Music2Emo fitter. |
| `train_late.py` | Independent binary branches, matched mixture ensembles, nested stem fusion and stem-only results through the same fitter and binary metrics. |
| `_extraction.py` | Route dataset defaults and command-line arguments to the feature implementations. |

Defaults are in `configs/experiments/pmemo/`:

| YAML | Purpose |
| --- | --- |
| `feature_manifest.yaml` | Fixed splits, original annotations, metadata and existing mixture/stem paths. |
| `separation_manifest.yaml` | The same fixed splits and mixture root, with separation destinations. Defaults to all supplied tracks; `--track-list` selects a subset. |
| `separation.yaml` | Shared HT-Demucs settings, separation manifest and output/report paths. |
| `mert.yaml` | Frozen MERT extraction and output locations. |
| `chords_keys.yaml` | Shared BTC/music21 extraction and output locations. |
| `mid_fusion.yaml` | Training paths and seven conditions; inherits `configs/training/emotion_recognition.yaml` and `music2emo.yaml`. |
| `late_fusion.yaml` | Late-fusion paths, twelve report conditions and the configured mid-fusion mixture reference; inherits the same training profiles. |

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
this does not affect audio extraction. Training uses the manifest's original
0-1 scores directly, not the historical 1-9 copy. Both targets
will reuse the same extracted features; do not extract once per target.
Local metadata and generated manifests remain ignored by Git.

Default input roots:

```text
/mnt/HDD/PMEmo2019/chorus/<id>.mp3
/mnt/HDD/osiris181/pmemo_separated/htdemucs_6s/<id>/<source>.mp3
```

The feature manifest binds all six native separator stems; extraction selects mixture,
guitar, drums, piano, bass and vocals, excluding `other`. Preparation reads
metadata only. Extraction reports missing/undecodable audio; it does not
silently remove tracks or change split membership.

Both manifest purposes use one entry point, like Jamendo. Feature and separation
manifests remain separate files because they serve different stages. Separation
needs only supplied split IDs and audio roots, not emotion annotations. It does
not create labels or alter the feature manifest. Its destinations match the
existing `<output_root>/htdemucs_6s/<id>/<source>.mp3` layout exactly; do not add
another `htdemucs_6s` suffix to `separation.yaml`'s `output_root`.

## Consistency with Jamendo

- Same shared HT-Demucs backend and `configs/separators/htdemucs_6s.yaml`:
  `htdemucs_6s`, one shift, overlap 0.25, split inference, rescale clipping,
  and 320 kbps MP3 output. Separation is independent of the feature extractor.
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
The fingerprinted Jamendo modules stay unchanged, preserving existing caches.
PMEmo records its adapter/manifest
reader checksums too; incompatible or corrupted caches are not silently reused.

The older extractor skips tails shorter than one second and its saved arrays
alone do not establish the new alignment/cache provenance. New extraction
therefore uses separate output directories, leaving old evidence intact.

## How to run

Use the uv environment described in the [root README](../../../../README.md#environment-setup-with-uv).
The preparation/extraction extras are `htdemucs`, `mert` and `chords-keys`;
scripts do not install packages themselves. Configure local audio/model paths
before use. `--help` lists all overrides.

From the repository root, enter the experiment directory and inspect both
manifest purposes. These commands read metadata only; they do not scan audio,
initialize models or overwrite saved manifests:

```bash
cd src/stemfusion/experiments/pmemo

python prepare_manifest.py --dry-run
python prepare_manifest.py --purpose separation --dry-run
```

For a new installation, prepare separation jobs and then run separation:

```bash
python prepare_manifest.py --purpose separation
python separation.py --dry-run
python separation.py
```

The separation manifest is saved to `data/manifests/pmemo/pmemo_separation.jsonl`,
with a `.report.json` companion. The separator's report is
`outputs/pmemo/separation_report_htdemucs.json`. It follows only manifest IDs,
not an entire audio-directory scan. Separation dry runs inspect job paths but
do not decode audio, launch Demucs, create output folders or save reports.
Complete, nonempty six-stem outputs are reused by default. Leave `--overwrite`
off when using existing audio. **If the separated stems already exist, there
is no need to rerun separation.**

For a targeted job list, use a text file containing one supplied numeric song
ID per line: `python prepare_manifest.py --purpose separation --track-list PATH`.
This overrides the all-track default. No failing IDs are hardcoded. Existing
manifest/report files are protected unless `--overwrite` is explicitly supplied.

Next, prepare the feature manifest only if it does not already exist:

```bash
python prepare_manifest.py
python extract_mert.py --dry-run
python extract_chords_keys.py --dry-run
```

The workspace manifest has already been prepared. Existing manifests are
protected; skip `python prepare_manifest.py` when reusing it. The unified
feature command calls the unchanged original builder, preserves the same
manifest bytes and output path, and does not invalidate existing features.
Adding separation support does not require any feature re-extraction.

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
stemfusion data pmemo-manifest --dry-run
stemfusion data pmemo-manifest --purpose separation --dry-run
stemfusion separate htdemucs --config configs/experiments/pmemo/separation.yaml --dry-run
stemfusion extract pmemo mert --dry-run
stemfusion extract pmemo chords-keys --dry-run
```

Run these installed commands from the repository root when passing the relative
`--config` path. `data pmemo-feature-manifest` remains a compatibility alias for
the feature purpose, and `data pmemo-separation-manifest` defaults to separation.
The direct preparation script is now named `prepare_manifest.py`.

## Outputs and reports

```text
data/manifests/pmemo/
  pmemo_separation.jsonl
  pmemo_separation.report.json
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
  separation_report_htdemucs.json
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

## Mid-fusion training

Use the uv environment with the `music2emo` extra, then run from this directory:

```bash
python train_mid.py --dry-run
python train_mid.py --validate-only
python train_mid.py
```

The default trains **70 classifiers**: seven conditions x five seeds x two
independent targets. To select a target, condition or seed:

```bash
python train_mid.py --target valence
python train_mid.py --target arousal
python train_mid.py --target valence --run mixture_only --seed 32
```

Equivalent installed command: `stemfusion train pmemo mid-fusion`.

## Late-fusion training

From this directory, using the same uv environment:

```bash
python train_late.py --dry-run
python train_late.py --validate-only
python train_late.py
```

Equivalent installed command: `stemfusion train pmemo late-fusion`. Both targets
and all five outer seeds run by default. The same `--target`, `--seed`, `--run`
and optional `--no-verify-checksums` selections are available, for example:

```bash
python train_late.py --target valence --seed 32 --run mixture_ensemble_4
```

Each target/outer-seed run needs eleven independent one-input classifiers:
six mixture branches and one each for guitar, drums, piano, bass and vocals.
Mixture branch 1 reuses the completed mid-fusion `mixture_only` model, including
its selected checkpoint and prediction threshold. `reference_output_dir` in
`late_fusion.yaml` points to the mid-fusion root; no trained-model path is
hardcoded in Python. Missing references can be trained with the exact mid
recipe. Incompatible references raise an error instead of being overwritten.
Even `--overwrite` never replaces an existing mid-fusion reference.

With all ten mixture references completed, the full run trains **100 new
classifiers**: ten additional branches x five seeds x two targets. Without
those references, up to ten reference fits are also needed. All branches use
the same one-input chord Transformer and 512/256 binary classifier, optimization,
early stopping and checkpoint rules described below.

Branch seeds match Jamendo and the genre experiments: position 0 uses the
outer seed; positions 1-5 use the first 32 SHA256 bits of
`stemfusion-phase2:<outer>:<index>`. Each stem branch shares the seed of the
corresponding mixture-control position, but has its own trained checkpoint.

The twelve report conditions are mixture only; independent M x 4/5/6 ensembles;
M+G+D+P, +B, +V ensembles; and five standalone stem rows. Ensembles sum selected
raw logits and then apply sigmoid, not majority voting or average probabilities.
Each ensemble selects its own validation-macro-F1 prediction threshold and
applies it unchanged to test; no ensemble weights or checkpoints are tuned.
Standalone stem rows reuse the same trained branches without extra fits.

Outputs are separated from mid fusion:

```text
outputs/pmemo/late_fusion/<target>/
  branches/<branch_id>/seed_<actual_training_seed>/
  ensembles/<condition>/seed_<outer_seed>/
outputs/pmemo/summary_late_fusion_valence.{json,txt}
outputs/pmemo/summary_late_fusion_arousal.{json,txt}
```

Each new branch saves `best.pt`, `last.pt`, ranked checkpoints and validation/test
predictions through the existing fitter. Summary TXT files include stem-only
rows and mean +/- sample SD; JSON files retain branch details and paired
comparisons. Subset runs use distinct summary filenames. Matching completed
runs are verified/reused; interrupted compatible branches resume normally.

`--output-dir` redirects new outputs and, unless explicitly supplied,
`summary_dir` to that location and the reference root to a separate sibling
`<output_dir_name>_reference`. To reuse an existing mid run from a different
location, pass `--reference-output-dir /path/to/mid_fusion` explicitly. Reference
and late-output roots must not overlap. This protects old runs when doing an
isolated smoke test or rerun.

## Shared training protocol

### Labels, representations and mid-fusion controls

- Separate valence/arousal labels: original score **>= 0.5 = high (1)**;
  **< 0.5 = low (0)**. Keep equality in high and all supplied tracks. No fitted
  label boundary, median split or score rescaling is applied. This is a binary
  adaptation, not a claim that the supplied Music2Emotion regression loader
  implements these labels.
- Reuse existing aligned MERT/harmony caches. Saved layer indices 5/6 provide
  1,536 audio values. Each source's own chord Transformer output (8) and key
  mode (1) give 1,545 encoded values per source. Reuse the exact Jamendo sampler:
  one shared consecutive range during training; all segments for evaluation.
- Conditions match Jamendo: mixture only; M x 4, M x 5, M x 6; M+G+D+P;
  M+G+D+P+B; M+G+D+P+B+V. Controls encode one mixture once and repeat its
  complete vector, including the same chord-dropout realization. The shared
  Transformer and 512/256 classifier give exactly matching parameter counts
  for every control/fusion pair.
- One sigmoid logit per target replaces Jamendo's 56-label output. All
  conditions use the same binary architecture/loss. Positive prevalence `p`
  comes from this target's **training labels only**, not Jamendo metadata.
  The retained author BCE uses `pos_weight=1/p`, `weight=2/(1+1/p)`.

### Fitting, checkpoint selection and metrics

The same `training/music2emo.py::fit_model` loop handles PMEmo and Jamendo:
Adam `1e-4`, batch 8, at most 200 epochs, no scheduler, validation weighted-BCE
early stopping (patience 10, min delta 0.0001), and minimum validation-loss
checkpoint selection (earliest exact tie). Keep five best checkpoints plus
`last.pt`. All conditions use seeds **32, 64, 128, 256, 512**. The chord
Transformer is trainable; pretrained MERT remains frozen through cached inputs.

After checkpoint selection, choose a **prediction** threshold using validation
macro F1 on 0.00, 0.01, ..., 1.00, with ties to the lowest value. Apply it
unchanged to test. This differs from the fixed **ground-truth boundary** 0.5.

Accuracy is track-level binary accuracy. Macro precision/recall/F1 average low
and high classes. Macro ROC-AUC and macro average precision also average both
one-vs-rest classes, using `1-p` for low and `p` for high. PR-AUC means average
precision, not trapezoidal area. Every split must contain both classes; invalid
splits raise an error instead of removing tracks or inventing AUCs.

Payload checksums are enabled by default, like Jamendo. Optional
`--no-verify-checksums` trusts completion markers, but still checks IDs, splits,
source identity, shapes, finiteness, settings and alignment. Finite cache edits
may go undetected in that mode, including during reuse. `--validate-only`
loads/validates features and labels without model construction, training or
output writes; `--dry-run` only resolves the plan.

Matching completed runs are verified/reused automatically. Interrupted runs
with matching `last.pt` resume optimizer/epoch/stopping state. Changed settings,
inputs, implementations and corrupted artifacts are rejected. `--overwrite`
explicitly retrains. Use a separate `--output-dir` for alternatives; it also
redirects summaries unless `--summary-dir` is supplied. Targets/conditions do
not overwrite each other's checkpoints.

### Code flow and outputs

`train_mid.py` -> `experiments/_emotion_recognition.py` ->
`data/emotion_mid_fusion.py` + existing Jamendo encoding/sampling utilities ->
`models/music2emo_emotion.py` + `training/music2emo.py`.
`training/emotion.py` supplies binary metrics and prediction-threshold selection;
no PMEmo optimizer or early-stopping loop is copied. YAML loads through the
same `configuration.py` parser as Jamendo.

The local `chord_root.json`/`chord_attr.json` copies under
`data/manifests/pmemo/music2emotion/` match Jamendo's released chord token maps,
not its label frequencies. Provide those two files at the configured locations
in a fresh checkout; metadata remains ignored by Git.

Checkpoints/predictions live under:

```text
outputs/pmemo/mid_fusion/<target>/<condition>/seed_<seed>/
  best.pt
  last.pt
  checkpoints/epoch_*.pt
  history.json
  resolved_config.json
  predictions_validation.npz
  predictions_test.npz
  result.json
  run_manifest.json
```

Training includes test evaluation, prints tables and writes
`outputs/pmemo/summary_mid_fusion_{valence,arousal}.{json,txt}`. Reports contain
mean/sample SD (`ddof=1`) and paired fusion-minus-control differences; JSON
retains per-seed metrics and checkpoint references. A run/seed subset gets a
distinct summary name and does not replace the full summary. Summary JSON/TXT
can be committed; raw caches, predictions and checkpoints remain ignored.
This entry point does not run late fusion or stem-only training.
