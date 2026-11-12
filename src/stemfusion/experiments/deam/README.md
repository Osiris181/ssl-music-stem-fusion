# DEAM preparation and fusion experiments

Prepare separation jobs, frozen MERT embeddings and BTC/music21 chord/key
descriptors, then train separate binary valence/arousal classifiers using the
same model/fitting settings as PMEmo. Preparation/extraction does not binarize
labels or replace old features; training applies the native-score rule below.

## Files and inputs

| File | Purpose |
| --- | --- |
| `prepare_manifest.py` | Default `--purpose features`: preserve split IDs/scores and bind audio. `--purpose separation`: label-independent jobs from the same split IDs. |
| `separation.py` | Shared HT-Demucs backend, tqdm progress and readable final summary; reuse complete existing native six-stem outputs. |
| `extract_mert.py` | Save all 12 frozen MERT transformer layers for every retained 30-second segment. |
| `extract_chords_keys.py` | Extract whole-track BTC chords, generated MIDI, music21 key and normalized chords independently for each source. |
| `train_mid.py` | Seven mid-fusion conditions, separate targets and five seeds through the shared Music2Emo fitter and binary metrics. |
| `train_late.py` | Independent position-paired branches, matched mixture ensembles and five standalone stem rows through the same fitter and binary ensemble evaluator as PMEmo. |

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
perform label normalization. Training assigns **high for native score >= 5**
and **low for score < 5**, independently for each target. The same audio features serve
both valence and arousal; do not extract once per target.

Both manifest purposes use one entry point, as in PMEmo and Jamendo. They
still save separate files for separate stages. The separation purpose reads
only split IDs, not annotations, and does not need existing stems. It uses
all supplied tracks by default; `--track-list` selects an explicit subset.
It preserves the same IDs, split order, mixture paths and six stem destinations
as the feature manifest. The existing feature builder/reader remains unchanged.

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

Separation uses the exact shared `configs/separators/htdemucs_6s.yaml` profile:
`htdemucs_6s`, one shift, overlap 0.25, split inference, rescale clipping and
320 kbps MP3. Dataset paths live in `separation_manifest.yaml` and
`separation.yaml`. Set `output_root` to `/mnt/HDD/osiris181/DEAM_separated`,
the parent of `htdemucs_6s`, not the model directory itself.

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

python prepare_manifest.py --dry-run
python prepare_manifest.py
python extract_mert.py --dry-run
python extract_chords_keys.py --dry-run
```

The local feature manifest has already been generated during preparation.
Skip the non-dry preparation command unless generating it for the first time.
Existing manifests are protected; use `--overwrite` explicitly only when
regenerating them.

If separation is needed, first generate its label-independent job manifest:

```bash
python prepare_manifest.py --purpose separation --dry-run
python prepare_manifest.py --purpose separation
python separation.py --dry-run
python separation.py
```

Skip the manifest-creation command if that manifest already exists. Preparation
reads metadata only; separation checks existing stem files on the HDD. The dry
run does not launch Demucs or write stems/reports. A normal run skips tracks
with all six nonempty native stems and attempts missing/incomplete tracks.
No automatic source-quality certification is implied by that completeness
check. `--overwrite` is explicit; do not use it merely to inspect completed stems.
The new preparation/separation flow does not require re-extracting valid features.

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
stemfusion data deam-manifest --dry-run
stemfusion data deam-manifest --purpose separation --dry-run
stemfusion separate htdemucs --config configs/experiments/deam/separation.yaml --dry-run
stemfusion extract deam mert --dry-run
stemfusion extract deam chords-keys --dry-run
```

Run the unified separation command from the repository root when passing that
relative config path. `data deam-feature-manifest` remains a compatibility alias;
`data deam-separation-manifest` defaults to the separation purpose. The direct
preparation filename is now `prepare_manifest.py`.

## Saved features and reports

```text
data/manifests/deam/
  deam_separation.jsonl
  deam_separation.report.json
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
  separation_report_htdemucs.json
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
The separation command prints a readable final summary (reused/new/failed/
unprocessed tracks, retries and saved report path), retaining detailed JSON
on disk like the other datasets.

## Mid-fusion training

Use the uv environment with the `music2emo` extra, then run from this directory:

```bash
python train_mid.py --dry-run
python train_mid.py --validate-only
python train_mid.py
```

Equivalent installed command: `stemfusion train deam mid-fusion`. The default
trains **70 classifiers**: seven conditions x five seeds x two independent
targets. Select a subset with the same arguments as PMEmo:

```bash
python train_mid.py --target valence
python train_mid.py --target arousal --seed 32 --run mixture_only
```

`configs/experiments/deam/mid_fusion.yaml` inherits the same
`configs/training/emotion_recognition.yaml` and `music2emo.yaml` profiles as
PMEmo, overriding only the native annotation scale/midpoint and dataset paths.
Other users must supply the released `chord_root.json` and `chord_attr.json`
vocabularies at the configured paths (local copies here are in
`data/manifests/deam/music2emotion/`). These are the same root/quality mappings
as PMEmo/Jamendo, not dataset class-frequency files.

### Labels, features and conditions

- Keep original song-level scores on **1-9**. High is **>= 5** and low is **< 5**;
  equality is high. There is no score normalization, median fitting, dynamic
  annotation re-averaging, new split or track filtering. This is our binary
  adaptation, not a claim about the original regression loader's labels.
- Reuse the existing aligned feature caches: saved MERT indices 5/6 yield 1,536
  audio values; the shared source-specific chord Transformer output (8) and key
  mode (1) yield 1,545 encoded values per source. Use exactly the PMEmo/Jamendo
  sampler: one shared consecutive training range, all segments for evaluation.
- Conditions: mixture only; repeated-mixture M x 4/5/6; M+G+D+P; M+G+D+P+B;
  M+G+D+P+B+V. Repeated controls encode the mixture once and repeat its complete
  vector, including the same dropout realization. Each fusion/control pair has
  identical input width and trainable parameter count.
- Separate one-output models for valence and arousal; the same shared chord
  Transformer and 512/256 classifier as PMEmo. Cached MERT stays frozen.
  The author-weighted BCE uses each target's training-only positive prevalence,
  with `pos_weight=1/p` and `weight=2/(1+1/p)`.

### Fitting, evaluation and output

All conditions use seeds **32, 64, 128, 256, 512**, Adam `1e-4`, batch 8, at most
200 epochs, no scheduler, validation weighted-BCE early stopping (patience 10,
min delta `0.0001`), and minimum validation-loss checkpoint selection (earliest
exact tie). Keep five best checkpoints plus `last.pt`. The exact unchanged
`training/music2emo.py::fit_model` and `training/emotion.py::BinaryEmotionTask`
handle both datasets. The native-score loader is regression-tested against
PMEmo's loader, and equivalent synthetic inputs/seeds produce identical fits.
PMEmo's fingerprinted controller/loader remain unchanged while its jobs run.

After checkpoint selection, choose the prediction threshold on validation
macro F1 using 0.00:0.01:1.00, ties to lowest, then freeze it for test. This
probability threshold is separate from the fixed **native annotation boundary 5**.
Report track-level binary accuracy and low/high two-class macro F1, ROC-AUC and
average precision (PR-AUC). Require both classes in each split rather than
silently excluding tracks or inventing undefined AUC values.

Payload SHA256 verification defaults on. `--no-verify-checksums` trusts local
completion markers but retains ID/split/source/shape/finiteness/settings and
alignment checks; finite cache edits can go undetected in that mode.
`--validate-only` checks inputs/labels without constructing models or writing
training results; `--dry-run` resolves only the plan.

```text
outputs/deam/mid_fusion/<target>/<condition>/seed_<seed>/
  best.pt
  last.pt
  checkpoints/
  predictions_validation.npz
  predictions_test.npz
  result.json
outputs/deam/summary_mid_fusion_valence.{json,txt}
outputs/deam/summary_mid_fusion_arousal.{json,txt}
```

The terminal and TXT summaries show mean +/- sample SD across seeds; JSON keeps
full metrics, predictions/checkpoint references, label policy and paired
differences. Seed/condition subset runs use distinct summary filenames.
Completed compatible models are verified/reused automatically; interrupted
runs resume from `last.pt`. Changed settings or inputs raise an error. Use
an isolated output root for changed settings; `--overwrite` is explicit.
`--output-dir` also redirects summaries unless `--summary-dir` is supplied.

## Late-fusion training

Once mid-fusion training has finished, use the same uv environment and run
from this directory:

```bash
python train_late.py --dry-run
python train_late.py --validate-only
python train_late.py
```

Equivalent installed command: `stemfusion train deam late-fusion`. Both targets
and all five outer seeds run by default. The same subset and checksum arguments
are supported, for example:

```bash
python train_late.py --target valence --seed 32 --run mixture_ensemble_4
python train_late.py --target arousal --run guitar_only
```

The native score rule stays **high >= 5, low < 5**, without normalization.
Model architecture, sampler, optimizer, weighted BCE, training-only prevalence,
early stopping, checkpoint selection and binary metrics are the same as mid
fusion and PMEmo. New branch fitting calls the exact shared
`training/music2emo.py::fit_model`; ensemble evaluation calls the exact shared
`training/emotion_late_fusion.py::assemble_ensemble`, with no DEAM-specific
optimizer or aggregation implementation.

For each target/outer seed, use eleven independent classifiers: six mixture
branches and one per guitar/drums/piano/bass/vocals. Mixture branch 1 reuses the
selected mid-fusion `mixture_only` model and its validation threshold. Its root
is explicitly configured by `reference_output_dir` in `late_fusion.yaml`;
there is no hardcoded trained-model path in Python. Incompatible references
raise an error. Missing compatible references may be fitted/resumed with the
same mid-fusion recipe, so do not run mid and late training concurrently in
the same reference directory. Even late `--overwrite` never overwrites an
existing mid reference.

With all ten references complete, the full late run trains **100 new branch
models**: ten additional branches x five outer seeds x two targets. Reference
reuse adds no training; missing references can add up to ten fits.

Seeds match PMEmo, Jamendo and the genre experiments. Outer seeds are
**32, 64, 128, 256, 512**. Position 0 uses the outer seed; positions 1-5 use
the first 32 SHA256 bits of `stemfusion-phase2:<outer>:<index>`. Each stem is
paired with the corresponding independently trained mixture position's seed.

Report twelve conditions per target/outer seed: mixture only; independent
M x 4/5/6 ensembles; M+G+D+P, +B and +V ensembles; five standalone stem rows.
Every ensemble sums selected raw logits, then applies sigmoid, rather than
voting or averaging probabilities. Select each ensemble's prediction threshold
on validation macro F1 (the same 101-point grid, ties to lowest) and freeze it
for test. No ensemble weights or ensemble-specific checkpoints are fitted.
Stem-only rows reuse the same selected branches and their thresholds without
extra training. The mean/sample SD comes from the same five outer-seed runs.

Outputs stay separate from mid fusion:

```text
outputs/deam/late_fusion/<target>/
  branches/<branch_id>/seed_<actual_training_seed>/
  ensembles/<condition>/seed_<outer_seed>/
outputs/deam/summary_late_fusion_valence.{json,txt}
outputs/deam/summary_late_fusion_arousal.{json,txt}
```

New branches save the same best/last/ranked checkpoints and prediction files
as mid fusion. The terminal and TXT summaries include stem-only rows; JSON
retains branch provenance, per-run metrics and paired comparisons. Matching
completed runs are verified/reused and compatible interrupted branches resume.
Subset runs use distinct summary filenames, avoiding replacement of full-run
summaries.

`--output-dir` redirects late outputs and, unless explicitly supplied, summaries
to that directory and references to a separate sibling `<output_dir_name>_reference`.
Use `--reference-output-dir /path/to/mid_fusion` explicitly when reusing another
mid root. Reference and late roots must not overlap. Payload checksums default
on; `--no-verify-checksums` has the same mandatory structural checks and finite
cache-edit limitation as mid fusion. Validation-only does not fit models or
write outputs; dry-run resolves only the schedule.
