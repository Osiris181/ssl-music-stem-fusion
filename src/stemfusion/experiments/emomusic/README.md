# EmoMusic preparation and fusion experiments

Prepare separation jobs, frozen MERT and BTC/music21 features using the same implementation
and scientific settings as DEAM, PMEmo and Jamendo. Preparation/extraction
preserve stored emotion scores and historical features. `train_mid.py` adds
separate binary valence/arousal classifiers through the same shared fitter;
`train_late.py` adds the same independent branch/ensemble protocol as DEAM/PMEmo.

## Files and inputs

| File | Purpose |
| --- | --- |
| `prepare_manifest.py` | Default `--purpose features`: preserve split IDs/annotations and bind audio. `--purpose separation`: label-independent jobs from the same split IDs. |
| `separation.py` | Shared HT-Demucs backend/settings, tqdm progress and readable final summary; reuse complete existing six-stem outputs. |
| `extract_mert.py` | Save all 12 frozen MERT transformer layers for every retained 30-second segment. |
| `extract_chords_keys.py` | Extract whole-track BTC chords, generated MIDI, music21 key and normalized chords independently per source. |
| `train_mid.py` | Seven-condition mid fusion, separate binary targets, five seeds and the shared Music2Emo model/fitting/metric implementation. |
| `train_late.py` | Independent matched mixture/stem branches, protected mid references, twelve reporting conditions including five standalone stems. |

The YAML presets are in `configs/experiments/emomusic/`. Relative YAML
paths resolve against the YAML file's directory, not the current working
directory. For example, `../../../data/manifests/emomusic/split/train.txt`
points into this repository. Workspace copies of the following supplied
Music2Emotion files are under `data/manifests/emomusic/`:

```text
Music2Emotion/dataset/emomusic/meta/split/train.txt
Music2Emotion/dataset/emomusic/meta/split/validation.txt
Music2Emotion/dataset/emomusic/meta/split/test.txt
Music2Emotion/dataset/emomusic/meta/static_annotations.csv
```

The manifest preserves their IDs and ordering: 495/124/125 train/validation/test
tracks, without overlap, covering all 744 supplied annotation rows. It does
not resplit the dataset or intersect available features to remove records.
These are the supplied Music2Emotion partitions, not independently verified
official splits. Metadata and generated manifests are local ignored inputs;
other users must supply their own files and audio/model locations.

`songs_info.csv` is copied for provenance only and is not consumed by extraction.
The manifest uses the provided `valence_mean` and `arousal_mean` columns on
their 1-9 scale unchanged. No averaging, normalization, column swapping or
binary thresholds are applied. The supplied EmoMusic and DEAM files exhibit
a valence/arousal column correspondence discrepancy: many same-ID pairs
match after exchanging the columns. This does not establish which file is
correct. Label provenance/meaning remains unresolved; it does not affect
audio-only feature extraction. Training uses the supplied column names without
swapping them; interpret target-specific results subject to this caveat. Both
targets reuse the same features.

## Original mixtures and shared DEAM stems

Default paths are deliberately different:

```text
/mnt/HDD/emoMusic/clips_45sec/<id>.mp3
/mnt/HDD/osiris181/DEAM_separated/htdemucs_6s/<id>/<source>.mp3
```

The DEAM stem root is intentional, not a naming mistake. The local
`Music2Emotion/preprocess/get_emomusic_from_DEAM.py` constructs this EmoMusic
audio collection by copying same-ID DEAM mixtures. Every supplied split ID
has an audio filename and a matching DEAM stem folder. Sample mixtures from
all three splits were verified byte-identical to same-ID DEAM audio. This
sample check is not a complete waveform or separation-provenance audit.
Feature extraction checks actual decoding and reports failures explicitly.
Manifest preparation and feature extraction do not copy, rename or regenerate
stem audio.

The combined manifest entry point has the same two purposes as DEAM/PMEmo.
It still writes separate feature and separation files for their respective
stages. Separation preparation reads only the supplied split IDs, without
annotations, audio scanning or existing stems. All supplied tracks are selected
by default; `--track-list` selects an explicit subset in the requested order.
The feature builder/reader, stored annotations and cached features are unchanged.

`separation_manifest.yaml` and `separation.yaml` retain the shared DEAM output
root intentionally. Set `output_root` to `/mnt/HDD/osiris181/DEAM_separated`,
the parent of `htdemucs_6s`; it must match the feature preset's `htdemucs_root`.
The shared `configs/separators/htdemucs_6s.yaml` supplies the exact same model,
one shift, overlap 0.25, split inference, rescale clipping and 320 kbps MP3
settings as DEAM/PMEmo/Jamendo. The new separation launcher can fill missing
or incomplete destinations; it does not run during preparation or extraction.
Leave `--overwrite` off to reuse complete shared stems, and do not run two
separation processes against the same output directories concurrently.

## Shared extraction implementation

DEAM and EmoMusic use the same `experiments/_emotion_features.py` orchestration
and `features/emotion_dataset.py` scheduler. Only the dataset manifest reader,
input paths, dataset identity and output/report destinations change. All model
inference and per-view writers remain the exact existing Jamendo/PMEmo backends:

- Pinned frozen MERT-v1-95M, mono 24 kHz, processor normalization, float32
  evaluation-mode inference and temporal mean pooling.
- Non-overlapping 30-second segments, retaining every nonempty tail. Only
  encoder inputs below 400 samples are right-zero-padded to 400.
- Stem end alignment to decoded mixture duration, within 0.25 seconds only;
  larger differences are errors. No additional 45-second crop is imposed.
- All transformer layers saved as `(1, 12, 768)` float32 arrays. Later training
  selects the agreed saved-layer indices 5 and 6.
- Same verified BTC utilities/checkpoint, released MIDI policy, music21 key
  estimation and C-major/A-minor chord normalization.
- Whole-track harmonic features independently extracted from each mixture
  and stem, not mixture descriptors copied to stem branches. Unavailable
  keys are recorded rather than silently replaced with invented estimates.
- Default-on bounded second pass for recognized tiny CQT-tail failures only,
  without hardcoded track lists. Disable with `--no-retry-failures`.

Scientific settings are regression-tested across EmoMusic, DEAM, PMEmo and
Jamendo. Existing fingerprinted feature modules remain unchanged while jobs
run. EmoMusic uses its own reader/provenance and feature outputs; existing DEAM
feature folders are not imported or modified.

## How to run

Use the existing MERT/BTC environment, not the separate SheetSage2 environment.
Wait for current GPU jobs to finish before starting another CUDA extraction.

```bash
source /home/osiris181/.venv/bin/activate
cd /home/osiris181/stem-fusion-music-classification/src/stemfusion/experiments/emomusic

python prepare_manifest.py --dry-run
python prepare_manifest.py
python extract_mert.py --dry-run
python extract_chords_keys.py --dry-run
```

The local manifest has already been prepared. Skip the non-dry preparation
command unless generating it for the first time. Use `--overwrite` explicitly
only when regenerating an existing manifest.

For separation, prepare the label-independent jobs first, then inspect/run:

```bash
python prepare_manifest.py --purpose separation --dry-run
python prepare_manifest.py --purpose separation
python separation.py --dry-run
python separation.py
```

Skip the manifest-creation command if its manifest already exists. Preparation
reads metadata only; separation checks stem files on the HDD. Dry-run performs
no inference, staging or output writes. Normal separation reuses tracks with
all six nonempty native stems and attempts missing/incomplete tracks only.
That completeness check does not certify waveform quality or the historical
provenance of all shared audio. No re-extraction is needed merely to adopt
this preparation layout. Wait for other GPU work before launching separation.

Test one training track (six views), then run the full extraction:

```bash
python extract_mert.py --split train --limit-per-split 1 --fail-fast
python extract_chords_keys.py --split train --limit-per-split 1 --fail-fast

python extract_mert.py
python extract_chords_keys.py
```

Full runs reuse verified smoke-test outputs. Reruns without `--overwrite`
verify completed per-view caches; incompatible settings, changed inputs and
corrupt files are reported rather than silently accepted. `--fail-fast` stops
after the first failure instead of completing the normal/retry passes.
`--help` lists options; `--source` and `--split` accept repeated selections.

Equivalent installed commands:

```bash
stemfusion data emomusic-manifest --dry-run
stemfusion data emomusic-manifest --purpose separation --dry-run
stemfusion separate htdemucs --config configs/experiments/emomusic/separation.yaml --dry-run
stemfusion extract emomusic mert --dry-run
stemfusion extract emomusic chords-keys --dry-run
```

Use the repository root for that relative `--config` path.
`data emomusic-feature-manifest` remains a feature-default compatibility alias;
`data emomusic-separation-manifest` defaults to separation. The direct
preparation filename is now `prepare_manifest.py`.

## Outputs

```text
data/manifests/emomusic/
  emomusic_separation.jsonl
  emomusic_separation.report.json
  emomusic_features.jsonl
  emomusic_features.report.json
data/features/emomusic/
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
outputs/emomusic/
  separation_report_htdemucs.json
  mert_extraction_report.json
  chords_keys_report.json
  chords_keys_report.initial.json  # only when automatic CQT retries occur
```

The manifest binds all six native stems, but default extraction selects the
mixture and guitar/drums/piano/bass/vocals, excluding `other`. MERT view folders
also contain `segments.json` and `run_manifest.json` with true boundaries,
padding/alignment metadata and file checksums. Both commands display tqdm and
final console summaries; reports retain per-view failures and outcomes.
Dry runs do not decode audio, initialize models or write extraction reports.
A full-run report replaces its smoke-test report while verified features are
reused. Old Music2Emotion extracted features remain untouched.

Separation prints the same readable final summary as the other datasets:
reused/new/failed/unprocessed tracks, retries and report location. The detailed
JSON report is retained on disk. Preparation/separation do not apply the
training label rule or resolve the annotation-provenance caveat above.

## Mid-fusion training

Use `configs/experiments/emomusic/mid_fusion.yaml`, inheriting the same
`configs/training/emotion_recognition.yaml` and `music2emo.yaml` as DEAM/PMEmo.
Keep the supplied 495/124/125 partitions and annotation columns. Native 1-9
scores are **high >= 5, low < 5**, independently for valence and arousal.
No normalization, column exchange, resplitting or fitted median is applied.
This is our binary adaptation, not the author's original regression task.

The fixed ground-truth midpoint is separate from the model's prediction
threshold. Each fit selects that probability threshold on validation macro F1
over 0.00:0.01:1.00 (ties to lowest), then freezes it for test evaluation.
Checkpoint selection minimizes validation weighted BCE (earliest epoch on
ties). The same fitter retains Adam 1e-4, batch 8, maximum 200 epochs,
validation-loss patience 10 and minimum delta 1e-4, top-five plus last
checkpoints, no scheduler, and training-only target prevalence for the loss.

Inputs use saved MERT layers 5/6 (1536 dimensions), eight encoded chord
dimensions and one key-mode value: 1545 dimensions per source. A shared
chord Transformer feeds the same 512/256-hidden-unit binary classifier.
Training samples a shared consecutive segment range across views; evaluation
uses every retained segment. Unavailable key modes follow the existing major
fallback. Stored extraction outputs are not modified.

Seven conditions run for each target and outer seed 32, 64, 128, 256, 512:

- Mixture only.
- Exact repeated mixture M x 4, M x 5, M x 6, including repeated chord/key
  descriptors, matching the corresponding fusion input widths.
- M + G + D + P, then + B, then + V.

This is 70 fits in total. Mid fusion does not train standalone stems;
those belong to the later late-fusion branch experiments.

The released root/quality vocabulary JSON files are required inputs. Local
copies are at `data/manifests/emomusic/music2emotion/chord_{root,attr}.json`;
other users must supply the same released maps and configure their paths.
No trained-model path is hardcoded in Python.

From this experiment directory:

```bash
python train_mid.py --dry-run
python train_mid.py --validate-only
python train_mid.py
```

Equivalent installed command: `stemfusion train emomusic mid-fusion`.
The default validates payload checksums. `--no-verify-checksums` skips only
payload hashing; ID/split/settings/shape/finiteness/alignment checks remain
mandatory. `--validate-only` loads/checks inputs without fitting or writing
training outputs; `--dry-run` only prints the configured plan.

Use `--target valence` or `--target arousal`, `--seed`, and `--run` to select
a subset. Complete compatible fits are reused, incomplete fits resume by
default, and incompatible settings/inputs fail rather than being silently
reused. `--output-dir` alone also redirects summaries into that directory;
an explicit `--summary-dir` overrides it. Wait for other GPU training to finish
before launching the production run.

The default checkpoints and per-fit JSON/prediction artifacts are under
`outputs/emomusic/mid_fusion/<target>/<condition>/seed_<seed>/`, including
`best.pt` and `last.pt`. Final test metrics are calculated during training;
no separate evaluation command is needed. Per-target console summaries and
`outputs/emomusic/summary_mid_fusion_{valence,arousal}.{json,txt}` report mean
and sample SD across seeds for track-level accuracy, two-class macro F1,
macro ROC-AUC and macro average precision (PR-AUC), plus paired control
differences. Summary TXT/JSON files are eligible for version control;
checkpoints, detailed per-fit artifacts and feature caches remain ignored.

The EmoMusic controller/cache adapter is parity-tested against DEAM/PMEmo.
The target/cohort implementation, model, fitter, sampler and binary metrics
are shared directly. Existing DEAM/PMEmo fingerprinted modules are left frozen
so adding EmoMusic does not invalidate their running or completed fits.

## Late-fusion training

Use `configs/experiments/emomusic/late_fusion.yaml`, inheriting the same
binary-emotion training profile. Keep the same supplied split IDs, native
score >= 5 label rule, unchanged target columns, MERT layers 5/6 and
source-specific BTC/music21 descriptors as mid fusion. The annotation-column
provenance caveat remains unresolved; late fusion does not reinterpret labels.

Each target/outer seed uses the exact DEAM/PMEmo branch schedule: reuse the
mid-fusion mixture-only model, train five additional independent mixture
models, and train one model for each of guitar, drums, piano, bass and vocals.
Every branch has the same one-source 1545-dimensional input and 512/256-hidden-
unit binary classifier, fitted by the shared Music2Emo backend with identical
optimizer, validation-loss checkpoint selection and early stopping as mid fusion.

Outer seeds remain 32, 64, 128, 256, 512. Position zero uses the outer seed;
positions 1-5 derive the first 32 SHA256 bits of
`stemfusion-phase2:<outer>:<index>`, pairing each stem's actual training seed
with the corresponding independent mixture branch. These derived seeds are
expected in progress output; they do not replace the five outer runs.

For each target, the twelve reports are:

- Mixture-only reference.
- Independent mixture ensembles M x 4, M x 5, M x 6.
- M + G + D + P, then + B, then + V.
- Guitar, drums, piano, bass and vocals individually.

Ensembles sum selected branch **raw logits, then apply sigmoid**, not voting
or averaging probabilities. Each ensemble selects its own probability threshold
on validation macro F1 using the same fixed grid/tie rule, then freezes it for
test. Branch weights and checkpoints are not tuned on test data. Stem-only
rows reuse the selected stem branches and need no extra fitting. With all ten
mid mixture references complete, the full two-target/five-seed run requires
100 new branch fits; missing references may require up to ten further fits.

Finish mid training before starting late training; do not concurrently write
the same references. From this experiment directory:

```bash
python train_late.py --dry-run
python train_late.py --validate-only
python train_late.py
```

Equivalent installed command: `stemfusion train emomusic late-fusion`.
`--dry-run`, `--validate-only`, `--no-verify-checksums`, `--target`, `--seed`
and `--run` have the same meanings as mid fusion. Default cache checksums
are enabled; structural/alignment checks remain mandatory when hashing is off.
Complete compatible fits/ensembles are reused; interrupted fits resume and
incompatible inputs/settings fail explicitly.

The YAML explicitly sets `reference_output_dir` to `outputs/emomusic/mid_fusion`;
Python does not hardcode a trained-model location. A late `--overwrite` never
overwrites existing mid-reference fits. If only `--output-dir` is overridden,
summaries follow it and the reference root defaults to a separate sibling
`<output-name>_reference`; use `--reference-output-dir` explicitly to reuse
references from a different run. Reference and late-output roots must not be
equal or nested.

Outputs are:

```text
outputs/emomusic/
  mid_fusion/<target>/mixture_only/seed_<outer>/best.pt
  late_fusion/<target>/branches/<branch>/seed_<actual>/best.pt
  late_fusion/<target>/ensembles/<condition>/seed_<outer>/
  summary_late_fusion_valence.json
  summary_late_fusion_valence.txt
  summary_late_fusion_arousal.json
  summary_late_fusion_arousal.txt
```

Test evaluation is automatic. Each target's console/JSON/TXT summary includes
the reference, matched ensembles, stem fusions and five standalone stems,
with mean/sample SD across outer seeds and paired matched-control differences.
Ensembles save predictions/metadata rather than an additional model checkpoint.
The shared fitter/evaluator and EmoMusic mid controller remain unchanged;
the new late scheduling/branch algorithms are parity-tested against DEAM/PMEmo.
