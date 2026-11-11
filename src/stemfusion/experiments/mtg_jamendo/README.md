# MTG-Jamendo mood/theme tagging

The shared audio preparation serves the MERT plus chord/key pipeline for
mood/theme tagging. Separation uses the same HT-Demucs backend
and inference profile as the genre datasets.

MERT extraction and training commands live directly in this dataset directory;
see [MERT segment extraction](#mert-segment-extraction) and
[author baseline reproduction](#author-baseline-reproduction) below.
The controlled stem experiments use [mid-fusion training](#mid-fusion-training).
Independent branches and stem-only results use [late-fusion training](#late-fusion-training).
`extract_mert.py` saves all 12 frozen transformer layers per
30-second segment with the same retained-tail/aligned-stem temporal strategy.
The dataset-level `extract_chords_keys.py` generates BTC chords, intermediate
MIDI, and global key independently for each mixture/stem, paired with the
MERT embeddings. MIDI files are generated during preprocessing,
not required inputs. See the [shared chord/key guide](#chordkey-extraction).

| File | Purpose |
| --- | --- |
| `prepare_manifest.py` | Prepare the full feature manifest by default; use `--purpose separation` to select tracks and new stem destinations. |
| `separation.py` | Run the shared separator on the prepared manifest and reuse complete output folders. |
| `validate_separation.py` | Fully decode native outputs, check duration/finiteness, and prepare a validated audio manifest with optional float32 WAV copies. |
| `extract_mert.py` | Save all-layer frozen MERT embeddings per 30-second segment. |
| `extract_chords_keys.py` | Generate whole-track BTC chords, MIDI-derived key, and normalized chords paired with MERT. |
| `train_reproduction.py` | Run the isolated five-seed original-mixture Music2Emo baseline, with released early stopping and a 200-epoch cap, before stem experiments. |
| `reproduction.py` | Baseline fitting, checkpoint selection, evaluation, and reporting implementation. |
| `train_mid.py` | Thin launcher for the seven controlled mid-fusion conditions over the same five seeds. |
| `train_late.py` | Thin launcher for independent branches, matched ensembles, and five standalone stems. |
| `_checksum_compatibility.py` | Preserve completed scientific identities only for reviewed loading/layout changes; record actual execution identities separately. |
| `_relocation.py` | Strict layout-only fingerprint compatibility for completed pre-move runs; not a runnable experiment. |

The harmonic extractor uses the original BTC -> generated MIDI -> music21
pipeline. All files listed above support this public preparation/extraction
workflow or the mixture-only baseline. `_relocation.py` is an internal runtime
dependency for existing baseline checkpoint compatibility, not an extra
experiment. This README covers MERT extraction, chord/key extraction, and
baseline training. Private experiment history is not part of this directory.

Training follows the same layered flow as the genre experiments:

```text
train_mid.py / train_late.py
    -> shared stemfusion/configuration.py loads and validates YAML
    -> experiments/_mood_theme_tagging.py schedules the selected fusion family
    -> stemfusion/training/music2emo.py fits each required classifier
    -> evaluate predictions and save TXT/JSON summaries
```

Late ensemble assembly uses `training/music2emo_late_fusion.py`; it combines
selected branch logits without another training loop. Models, cached-input
loading, and reusable training code remain in their respective packages.
The task controller lives alongside `experiments/_genre.py`, so dataset
directories only need thin training launchers rather than `training.py` wrappers.
The separate baseline reproduction retains its existing implementation.

Shared harmonic extraction loads `configs/experiments/mtg_jamendo/chords_keys.yaml`.
Run `python extract_chords_keys.py --dry-run` from this dataset directory to
plan it, or `python extract_chords_keys.py` to extract. Existing descriptor and
report paths remain unchanged; relocating the entry script does not repair
failed views or regenerate completed outputs. The shared extraction now supports
one automatic CQT-tail retry pass, enabled by default; use
`--no-retry-failures` for strict first-pass-only behavior. Verified successful
outputs are reused without rewriting them. See the shared guide for reports,
the minimal-padding adaptation, and cache compatibility.

## Metadata and track selection

The local split TSVs in `data/manifests/mtg_jamendo/split-0/` are copies of
the provided Music2Emotion files: 9,949/3,802/4,231 train/validation/test tracks
and 56 mood/theme tags. Preparation preserves their IDs, tags, relative
paths, and assignments. It rejects duplicate IDs/paths, unsafe paths, unknown
selected IDs, and unexpected tag vocabularies. It does not resplit the dataset
or scan the audio collection.

`configs/experiments/mtg_jamendo/separation_manifest.yaml` defines the split files,
original mixture root, selected-track text file, new output root, and manifest
location. The track list contains one canonical `track_0000001`-style ID per
line. These local inputs and generated manifests remain ignored under `data/`;
the public preset contains no fixed track IDs.

From this directory:

```bash
python prepare_manifest.py --purpose separation --dry-run
python prepare_manifest.py --purpose separation
python separation.py --dry-run
python separation.py
python validate_separation.py --write-wav-fallbacks
```

Default local files:

```text
data/manifests/mtg_jamendo/repair_tracks.txt
data/manifests/mtg_jamendo/mtg_jamendo_separation.jsonl
data/manifests/mtg_jamendo/mtg_jamendo_separation.report.json
```

The JSONL selects audio for separation; it is not a complete training manifest.
Rows retain multi-label tags without inventing a single genre label.
Use `--track-list PATH` for another selection, or `--all-tracks` to explicitly
select the full provided set. Choose a separate output root and manifest
path when separating another set.

The equivalent installed preparation command is
`stemfusion data mtg-jamendo-manifest --purpose separation`.
The equivalent separation command is:

```bash
stemfusion separate htdemucs --config /path/to/configs/experiments/mtg_jamendo/separation.yaml
```

## Separation settings and outputs

`separation.yaml` inherits `configs/separators/htdemucs_6s.yaml`: Demucs 4.0.1,
`htdemucs_6s`, CUDA, shifts 1, overlap 0.25, split inference, rescale clipping,
MP3 at 320 kbps with preset 2, jobs 0. All six native sources are retained.

The new output layout is:

```text
/home/osiris181/mtg_jamendo_separated_htdemucs/repairs/
└── htdemucs_6s/
    └── <numeric_track_id>/
        ├── bass.mp3
        ├── drums.mp3
        ├── guitar.mp3
        ├── other.mp3
        ├── piano.mp3
        └── vocals.mp3
```

Both presets must agree on the output root: manifest paths determine exact
destinations, so changing only the separator's root does not redirect them.
YAML paths resolve relative to their file; CLI options override YAML.
Temporary input symlinks give Demucs the intended numeric folder names;
source recordings are not copied or renamed.

Separation reuses a target when all six native files exist and are non-empty.
`--overwrite` explicitly regenerates existing targets. Separation dry runs
check exact paths without staging audio or writing outputs/reports. Manifest
dry runs read metadata without checking audio availability.

The report is `outputs/mtg_jamendo/separation_report_htdemucs.json`; model logs
and subsequent waveform-validation reports belong in the same output folder.
File completion is followed by decoded-audio validation before repaired stems
are used for feature extraction. Existing feature arrays and stem directories
remain intact.

## Decoder-compatible audio

Some MP3 streams can be decoded by libsndfile while torchaudio's decoder
rejects them. This alone does not establish that the stream contains no audio.
The validator tries torchaudio first, then libsndfile, and requires finite,
non-empty waveforms with durations within 0.25 seconds of the TSV metadata.
Valid silence is retained.

With `--write-wav-fallbacks`, alternative-decoder waveforms are saved beside
their MP3s as float32 PCM WAV. The validator verifies an exact sample round
trip through torchaudio. It adds no further lossy encoding and preserves
the MP3 files. These WAVs preserve the decoded MP3 samples; they are not
claimed to reconstruct the pre-compression model waveforms.

The downstream mapping is published only after every native stem passes:

```text
data/manifests/mtg_jamendo/mtg_jamendo_repaired_audio.jsonl
outputs/mtg_jamendo/separation_audio_validation.json
```

Use this validated mapping for repaired feature extraction. Keep the original
`mtg_jamendo_separation.jsonl` for Demucs output routing: it continues to name
the native MP3 destinations. Validation accepts `--config`, `--manifest`,
`--output-root`, `--report`, and `--validated-manifest`; existing different
WAVs/manifests require explicit `--overwrite`. It never changes input mixtures
or other stem directories.

## Shared feature-extraction manifest

Run `python prepare_manifest.py` from this directory to build
`data/manifests/mtg_jamendo/mtg_jamendo_features.jsonl`. It loads
`configs/experiments/mtg_jamendo/feature_manifest.yaml` and binds all provided
split records to existing mixture/stem audio, including validated repair
paths. It reads metadata only, without scanning or decoding the audio corpus.
The installed command is `stemfusion data mtg-jamendo-manifest`.

The default purpose is `features`; `--purpose features` is equivalent. Use
`--dry-run` to inspect the plan without writing a manifest. The same script
uses `--purpose separation` for new separation jobs, selecting the existing
`separation_manifest.yaml` instead. `--config PATH` overrides the preset for
the selected purpose, and explicit CLI options override its YAML values.
Feature-only and separation-only options cannot be mixed. The two purposes
retain their existing manifest filenames and mappings; they do not run
separation or extraction automatically. Existing manifests need not be rebuilt
solely because the entry script changed.

The old installed commands `mtg-jamendo-feature-manifest` and
`mtg-jamendo-separation-manifest` remain compatibility aliases under
`stemfusion data`, defaulting to their original purposes. The standalone
preparation entry point is now only `prepare_manifest.py`.

This shared manifest serves [MERT extraction](#mert-segment-extraction) and
[chord/key extraction](#chordkey-extraction). Both use identical track
assignments and source mappings. The separation manifest instead specifies
which audio to separate and where new stems should be written.

The older MusicFM-named manifest can remain as a historical artifact; changing
the manifest filename does not require rerunning separation or extraction
when its contents are unchanged. Existing reports retain their original
configuration paths and hashes.

## MERT segment extraction

`extract_mert.py` prepares frozen MERT embeddings; it does not train a
classifier or modify MusicFM features. Shared harmonic descriptors are
extracted independently with [`extract_chords_keys.py`](extract_chords_keys.py).
See the [shared chord/key guide](#chordkey-extraction) for that stage.

`extract_mert.py` loads `configs/experiments/mtg_jamendo/mert.yaml`,
which inherits `configs/features/mert_v1_95m.yaml`. It uses the same shared
audio manifest as the harmonic stage. It does not require chord/key
extraction to finish first.

Settings follow the released Music2Emotion MERT encoder: `m-a-p/MERT-v1-95M`,
mono 24 kHz, torchaudio channel averaging and resampling, the checkpoint's
normalized waveform processor, and temporal mean pooling of all 12 transformer
layers. The embedding state is excluded. Each `segment_N.npy` is float32 with
shape `(1, 12, 768)`, rather than MusicFM's `(1024,)`.
The frozen model is explicitly placed in evaluation mode.

The model revision is pinned to `12af15fef9d0ac838c3f475bfbbf26d2060dd4f5`.
The checkpoint includes custom Python code: only use a trusted model snapshot.
Model weights/code/configuration hashes, package versions, preprocessing rules,
and implementation hashes are recorded and checked on resume.

From the repository root, install into your intended environment:

```bash
uv pip install -e '.[mert]'
```

The default is cache-only (`runtime.local_files_only: true`). If the pinned
snapshot is not cached, explicitly allow its download with
`--no-local-files-only`; subsequent runs can use the default. Alternatively,
set `paths.model_path` in the feature YAML or pass `--model-path /path/to/snapshot`.
No local home-directory model location is hardcoded in Python.
Torchaudio must have a working decoder backend; versions using TorchCodec need
its matching installation. Audio decoding errors are reported, not silently
replaced with another library's waveform.

From this experiment directory:

```bash
python extract_mert.py --dry-run
python extract_mert.py --split train --limit-per-split 2
python extract_mert.py
```

The first command reads only the manifest/settings; it does not decode audio,
resolve/download models, or write outputs. The second actually extracts two
training tracks across all six views. `--source mixture` selects just the
mixture; repeat `--source` or `--split` to select a subset. CLI lists replace
YAML lists. The equivalent installed command is
`stemfusion extract mtg-jamendo mert`. Use `--device cpu` when appropriate.

Non-overlapping 30-second segments retain every partial tail without padding
to 30 seconds. MERT's convolution kernels `[10,3,3,3,3,2,2]` and strides
`[5,2,2,2,2,2,2]` require at least **400 samples** for one output frame.
Only shorter tails are right-zero-padded to 400 samples (16.7 ms at 24 kHz).
This is an explicit architecture-specific adaptation; the released author
code has no tiny-tail fallback. Valid segments remain unchanged.
`--short-segment-policy error` reports unsupported tails instead.
Pooling preserves the layer axis even for a single output frame, avoiding
the released encoder's unrestricted `squeeze()` edge case.

As agreed for stem fusion, the mixture duration is preserved and each stem's
end is trimmed/padded to that decoded mixture length, with at most 0.25 seconds
of correction. Larger discrepancies fail. No time-offset alignment is inferred.
This is our stem-fusion adaptation, not part of the original mixture baseline;
`--stem-alignment none` disables it explicitly. Metadata records true segment
boundaries, minimum padding, and end corrections for auditing.

Default outputs:

```text
data/features/mtg_jamendo/mert_chord_key/mert_v1_95m/
├── mert_30s/<bucket>/<numeric_track_id>/
└── mert_30s_<source>/<bucket>/<numeric_track_id>/
    ├── segment_0.npy
    ├── segment_1.npy
    ├── segments.json
    └── run_manifest.json
```

All 12 layers are retained to support the released loader's selection later.
Its saved-array indices `[5, 6]` correspond to Hugging Face hidden-state
indices `[6, 7]`, because saved index 0 is the first transformer layer.
Concatenating these two layers gives 1,536 MERT dimensions before adding
harmonic descriptors. Extraction does not select layers or append chords/key.
The arrays require about 36 KiB per segment; 842,928 segments across six views
would use approximately 29 GiB of array contents, plus metadata and filesystem
overhead.

Completed folders are published atomically and reused only when settings,
source/reference file identities, and artifact checksums match. Old array-only
caches are not automatically accepted or copied. Choose a separate directory
or explicitly use `--overwrite` to regenerate a cache. Failed overwrites
preserve the previous cache. Run one writer per output directory.

Progress uses tqdm. The report is
`outputs/mtg_jamendo/mert_chord_key/mert_extraction_report.json`, including failed
and unprocessed views, segment-count mismatches, adjusted stems, and minimum
padded segments. Failures/mismatches return nonzero status. No tracks are
silently dropped. Test a small subset before a full extraction.

Reference: the released [MERT encoder](https://github.com/AMAAI-Lab/Music2Emotion/blob/c5f7984f78d9928b841f25bd05a47377c12c58ca/preprocess/encoder/mert.py),
[segmentation code](https://github.com/AMAAI-Lab/Music2Emotion/blob/c5f7984f78d9928b841f25bd05a47377c12c58ca/preprocess/feature_extractor.py),
and [layer configuration](https://github.com/AMAAI-Lab/Music2Emotion/blob/c5f7984f78d9928b841f25bd05a47377c12c58ca/config/base_config.yaml).

## Chord/key extraction

`extract_chords_keys.py` is dataset-level harmonic preprocessing for the MERT
experiments. It uses each source's audio, not its MERT embeddings. Match
descriptors by track ID and source; extract them once and pair the saved outputs
with the corresponding MERT representation.

### What it does

For each mixture, guitar, drums, piano, bass, and vocals recording independently:

1. Run the released frozen BTC model on log-CQT features: mono 22,050 Hz,
   144 bins, 24 bins/octave, hop 2,048, 10-second feature blocks, 108-frame
   model windows, 170 chord labels, and checkpoint mean/std normalization.
   The verified author BTC configuration supplies these hyperparameters.
2. Convert whole-track predicted chords to pitch-class MIDI notes using
   `mir_eval` and `pretty_midi`, with program 0, velocity 120, and pitch base 48.
3. Estimate global key from this generated MIDI with `music21`.
4. Normalize chord roots to C major or A minor, following the released rule.

MIDI files are generated from predicted chords; source MIDI files are not
required. These descriptors describe the whole audio view, independently of
the 30-second MERT segments. Applying the original mixture-based pipeline to
stems is our extension. A stem never receives the mixture's descriptors.
Estimates can be unreliable for drums or sparse stems; an estimated key is not
a confidence guarantee. Failed key analysis is saved as `None` and reported,
without dropping the track or fabricating a major key. Its chords stay
untransposed when key is unavailable.

The shared full audio manifest is
`data/manifests/mtg_jamendo/mtg_jamendo_features.jsonl`. It contains audio paths,
labels, and split assignments, not embeddings. MERT and harmonic extraction use it,
including validated WAV alternatives. Build it with
`python prepare_manifest.py` if needed. No new split or separation
is required.

### Dependencies and configuration

From the repository root, in the intended Python environment:

```bash
uv pip install -e '.[chords-keys]'
```

The extra pins the author's librosa, pretty_midi, music21, and mir_eval Git
revision. Use a separate environment if these conflict with another project.
Provide the trusted Music2Emotion checkout and large-vocabulary BTC weights;
the command does not download models or import their Gradio application.
The external checkpoint is loaded with `weights_only=False`: do not supply
untrusted checkpoint files.

Edit `configs/experiments/mtg_jamendo/chords_keys.yaml` to set
the audio manifest, checkout, BTC configuration/checkpoint, output, and device.
YAML paths resolve relative to their YAML file; CLI values override them.
The four used BTC utility files and configuration must match the verified
author release. Modified dependencies are rejected rather than silently used.
Checkpoint hashes, implementation identity, and package versions are saved.

### Commands

From `src/stemfusion/experiments/mtg_jamendo/`:

```bash
python extract_chords_keys.py --dry-run
python extract_chords_keys.py --split train --limit-per-split 2
python extract_chords_keys.py
```

The first command only plans jobs; it does not read audio, load models, or
write outputs. The second actually processes two training tracks across all
six views. The third processes the full provided split set. To select only
mixtures, pass `--source mixture`. Repeat `--source` or `--split` to select
multiple values. `--device cpu` is available.

Equivalent installed command: `stemfusion extract mtg-jamendo chords-keys`.

### Automatic retry pass

`execution.retry_failures: true` is the YAML default. Running the normal
command first verifies/reuses completed outputs and extracts missing views,
then retries supported CQT-tail failures **once**, after the main loop:

```bash
python extract_chords_keys.py                       # Automatic retries enabled
python extract_chords_keys.py --retry-failures      # Explicitly enable
python extract_chords_keys.py --no-retry-failures   # Strict first pass only
```

CLI options override YAML. No separate repair command or track-ID list is
needed. The same shared implementation handles all requested sources;
`--fail-fast` stops immediately and does not enter the retry pass.
The retry option does not change cache identity, so toggling it does not
invalidate otherwise matching completed outputs.

Only the known six-octave CQT error for an unsupported one-sample input is
eligible. On retry, the shared frontend reproduces the released 10-second
block computation, adding **one zero sample** only to the one-sample remainder.
The minimum of two samples was verified with the pinned 22,050 Hz, 144-bin,
24-bin/octave, hop-2,048 frontend and librosa 0.10.2.post1. Longer partial blocks
are neither padded nor discarded. Source recordings are never modified.
Original duration, true block boundaries, and padding are recorded in each
repaired output's `metadata.json` under `extraction.cqt_repair`.

This is an explicit preprocessing adaptation, separate from MusicFM's
1,025-sample and MERT's 400-sample minima. No model weights, chord vocabulary,
BTC inference settings, MIDI policy, or key-estimation rules change.
Missing/corrupt files, incompatible caches, unrelated errors, and successfully
saved `None` keys are not automatically repaired. Failed retries remain in
the final report and produce a nonzero exit status; there is no infinite loop.

### Saved files and resume

Harmonic descriptors are paired with MERT and live directly under
the dataset feature directory. Paths remain configurable in the shared YAML.
The reusable implementation stays in `features/chords_keys.py`. Cache identities
use source audio and extraction settings, not the output root; moving a complete
descriptor directory does not require re-extraction or rewriting its contents.

```text
data/features/mtg_jamendo/harmony/
└── <source>/<bucket>/<numeric_track_id>/
    ├── chords.lab              # Raw BTC intervals
    ├── normalized_chords.lab   # Key-normalized intervals
    ├── key.lab                 # e.g. D minor, Bb major, or None
    ├── chords.mid              # Generated intermediate, retained by default
    ├── metadata.json           # Input identity, settings, key status
    └── run_manifest.json       # Fingerprint and artifact checksums
```

`--no-keep-midi` still generates MIDI for analysis, then discards it. Matching
completed outputs are verified and reused. Partial, modified, or incompatible
caches fail; choose a new output directory or use explicit `--overwrite`.
Failed overwrites preserve the previous completed cache.

The one verified pre-retry implementation remains compatible for successful
normal outputs. Its original metadata and completion markers are left intact;
all input identities, settings, dependency versions, weights, and artifact
checksums must still match. Unknown older code or scientific-setting changes
are not silently accepted. The report counts these outputs in `legacy_reused`.
Do not use `--overwrite` when continuing a completed extraction merely to
recover missing views.

Progress uses tqdm track/source views. The final report is
`outputs/mtg_jamendo/mert_chord_key/chords_keys_report.json`, including extracted,
reused, repaired, still-failed, unprocessed, and unavailable-key counts.
`initial_failed`, `retry_candidates`, and `retry_attempted` describe the two
passes. Before retries start, first-pass failures are saved separately in
`chords_keys_report.initial.json` beside the final report (a custom report uses
the corresponding `<stem>.initial.json`). Final records retain `initial_error`,
attempt counts, padding provenance, and any remaining retry error.
Each selected view is counted once in the final outcome:
`extracted + reused + repaired + failed + unprocessed = planned`.
After an audited artifact relocation, `records[].output` and
`current_output_dir` identify current locations, with mappings recorded in
`artifact_relocations`. Historical `resolved_config` and YAML snapshots retain
the paths used during extraction; a move is not presented as a new extraction.
An unavailable key is
not an extraction failure; failed audio/BTC processing is. Processing failures
return a nonzero exit status. No tracks are silently excluded.

### Released edge cases

The default `midi_policy: music2emotion_released` preserves the author event
loop, including its missing final flush for a sequence with just one chord
interval. That can generate empty MIDI and no key. The explicit
`--midi-policy complete_intervals` correction handles this case and leaves
multi-interval event handling unchanged. It is a preprocessing adaptation,
not the default. Use one policy consistently across all compared views; cache
signatures prevent mixing policies.

BTC interval timing and padded-boundary behavior also remain as released,
including the `10 / 108` frame-time convention. Labels are not resliced to
MERT boundaries or clamped to decoded duration. Empty or invalid intervals
are reported as failures. This command saves symbolic descriptors; their
encoding and combination with MERT must be specified in the training stage.

The released CQT frontend can reject extremely short remainders after its
10-second blocks (confirmed with a synthetic one-sample remainder at 22,050 Hz).
The first pass still follows the release. The default-on bounded retry pass
applies the explicitly recorded minimal-padding adaptation described above;
disable retries to preserve strict failure behavior.

See the [source comparison](../../../../docs/music2emotion_chords_keys.md)
for the original release and the earlier local batch script.

## Author baseline reproduction

`train_reproduction.py` trains only the original-mixture MERT + chord/key
single-dataset baseline: at most 200 epochs, batch size 8, Adam `1e-4`, and seeds
32, 64, 128, 256, 512. It uses the author chord Transformer and classifier,
not the genre MLP, and selects by validation loss before testing. Released
early-stopping settings monitor that loss with patience 10 and `min_delta=0.0001`.
Checkpoint ranking has no minimum delta; the top five plus last are saved.
It does not train stems, MusicFM classifiers, multitask students, or teacher
distillation.

```bash
python train_reproduction.py --dry-run
python train_reproduction.py --validate-only
python train_reproduction.py
```

Settings are in `configs/experiments/mtg_jamendo/reproduction.yaml`.
YAML presets define cached feature, report and checkpoint locations. Run scripts
directly from `src/stemfusion/experiments/mtg_jamendo/`.
Verified checkpoints resume automatically. The five-seed summaries are
`outputs/mtg_jamendo/summary_author_reproduction.{json,txt}`. See the
[complete reproduction guide](../../../../docs/music2emotion_reproduction.md)
for required metadata, architecture, missing-key behavior, saved predictions,
checkpoint paths, paper/code differences, and the single-device limitation.
The checkpoint root is
`outputs/mtg_jamendo/mert_chord_key/author_reproduction_early_stopping/`.

`_relocation.py` preserves the historical baseline trainer identity only when the
training AST matches the pinned pre-move implementation after explicit relocation
substitutions. Model, optimizer, sampling, stopping, and evaluation changes still
invalidate compatibility. No saved checkpoint, result, or cache provenance is
rewritten to disguise a different run.

## Mid-fusion training

Run `train_mid.py` after MERT and the original BTC/MIDI/music21 harmony caches
are complete. It loads `configs/experiments/mtg_jamendo/mid_fusion.yaml`, which
inherits `configs/training/mood_theme_tagging.yaml` and `music2emo.yaml`.
No audio, separator, MERT weights, or BTC weights are loaded during training.

The seven conditions are ordinary mixture-only; four/five/six repeated-mixture
controls; and `M+G+D+P`, `M+G+D+P+B`, `M+G+D+P+B+V`. Each uses the fixed
seeds 32, 64, 128, 256, 512 and the provided dataset splits. Missing,
misaligned, modified, or incompatible caches fail rather than excluding tracks.
The loader verifies all requested views once and keeps selected-layer arrays
in memory; the full six-view matrix needs several GiB of RAM plus worker
overhead. `--num-workers 0` can reduce worker overhead.

### Architecture and matched controls

Each view supplies its own 1,536-dimensional MERT vector, 100 chord root/quality
tokens, and key mode. One shared trainable author chord Transformer maps each
view's tokens to eight values. Its key mode adds one more, producing 1,545
values per view. The representations are concatenated before the classifier:
`(1545 * inputs) -> 512 -> 256 -> 56`. No stem receives mixture-derived harmony.

Repeated-mixture controls encode one mixture once, then repeat its complete
representation, including the same chord-dropout realization. They do not
resample or re-encode mixture copies independently. Fusion and its control
use the same shared encoder and first-layer width, with exactly matched total
trainable parameter counts:

| Inputs | First-layer width | Trainable parameters |
| --- | ---: | ---: |
| 1 | 1545 | 940,216 |
| 4 | 6180 | 3,313,336 |
| 5 | 7725 | 4,104,376 |
| 6 | 9270 | 4,895,416 |

Training samples one conditional-uniform consecutive segment range per track
and applies that same range to all selected views. Validation/test average
all segments. Cached segment boundaries and mixture references must agree.
The shared encoder and synchronized sampling are explicit multi-source
extensions of the single-mixture reproduction, not claimed author stem models.

### Training, selection, and metrics

The reproduction protocol is retained: author frequency-weighted BCE; Adam
`1e-4`, batch size 8; at most 200 epochs; validation-loss early stopping with
patience 10 and minimum delta `0.0001`. Checkpoints rank by any strictly lower
validation loss, keeping the top five plus last. Exact ties retain the earliest
epoch. The final test is evaluated with the selected best weights.

Macro PR-AUC means average precision; macro ROC-AUC is also saved. For the
additional thresholded metrics, choose one global threshold maximizing
validation macro F1 on the fixed inclusive grid `0.00, 0.01, ..., 1.00`; ties
choose the lowest threshold. Freeze that threshold for test. Accuracy means
label-wise correctness across tracks and tags, not multiclass or exact-match
accuracy; subset accuracy is additionally stored in JSON. Macro precision,
recall, and F1 use zero for undefined per-tag precision/recall denominators.
Threshold selection never affects checkpoint ranking or the saved reproduction.

### Commands and checkpoint reuse

From this dataset directory:

```bash
python train_mid.py --dry-run
python train_mid.py --validate-only
python train_mid.py
```

`--dry-run` plans without loading features or models or writing outputs.
`--validate-only` verifies cached inputs without training or writing results.
The last command runs all seven conditions and all five seeds. Equivalent
installed command: `stemfusion train mtg-jamendo mid-fusion`.

Feature payload SHA256 checks are enabled by default through
`runtime.verify_checksums: true` in `configs/experiments/mtg_jamendo/mid_fusion.yaml`.
For trusted, unchanged local caches:

```bash
python train_mid.py --no-verify-checksums
```

`--verify-checksums` explicitly enables hashing, overriding a YAML `false`.
Both flags work with `--validate-only`. ID/split/source, shapes, finite values,
encoder/preprocessing settings, segment ordering/alignment, chord/key vocabulary,
labels and training-prevalence checks remain mandatory. Both fusion methods use
the same parity-tested optional loader in `data/jamendo_late_fusion.py`; its
historical filename does not restrict it to late fusion. No audio is decoded.
Loading feature arrays is still necessary even without hashing.

Skipping checksums trusts existing completion markers. Valid-shaped finite
changes to feature values can go undetected, including when reusing checkpoints;
keep checks enabled after copying or modifying caches. The current mode is
logged and recorded in `plan.verify_checksums` and `data_audit.cache_verification`.
New fits retain the loading audit; reused run files retain their original audit.
Saved model/prediction artifact checksums are never disabled.

Use `--run mixture_guitar_drums_piano` for one condition and repeated `--seed`
options for a seed subset. These partial runs get separate summary filenames.
All paths are configured in YAML or overridden by CLI; there is no fixed
checkpoint directory in Python. For an isolated bounded training run:

```bash
python train_mid.py --run mixture_guitar_drums_piano --seed 32 --epochs 1 \
  --output-dir /path/to/separate/smoke-test
```

An explicit `--output-dir` also redirects summaries and reference checkpoints
under that root unless `--summary-dir` or `--reference-output-dir` is supplied.
Changing a budget or device may invalidate existing checkpoint signatures;
use a separate root for such checks, not the completed reproduction root.

By default, `paths.reference_output_dir` points to the completed reproduction.
The ordinary mixture condition calls the unchanged baseline fitter: compatible
completed checkpoints are reused, missing runs can train under the same
protocol, and incompatible runs fail. Historical reference files are never
overwritten by `train_mid.py`, including with `--overwrite`. The new mixture
evaluation derives thresholded metrics from verified saved predictions and
saves them separately. Compatible incomplete reference runs may resume through
the retained baseline fitter; completed reference runs are verified and reused.
Other conditions resume interrupted optimizer states
and verify completed-run checksums. `--overwrite` applies to their new outputs
and derived reference evaluations, not the historical reference trainer.

The reviewed checksum-option and controller-consolidation transitions preserve
completed mid- and late-fusion scientific identities through
`_checksum_compatibility.py`, but only for the exact reviewed controller, entry
points, shared Jamendo configuration schema, and pinned representation-equivalent
loader. Other controller/schema/loader edits or changed model/trainer dependencies invalidate
compatibility. New summaries record `execution_implementation_sha256` for the
actual current code separately from the retained scientific identity. Existing
checkpoints, prediction files, and per-run result provenance are not rewritten;
enabling or disabling this integrity option alone does not require retraining.

### Saved results

```text
outputs/mtg_jamendo/
├── summary_mid_fusion.json
├── summary_mid_fusion.txt
└── mid_fusion/<condition>/seed_<seed>/
    ├── best.pt                       # New non-reference conditions
    ├── last.pt                       # Epoch/optimizer/patience resume state
    ├── checkpoints/epoch_NNNN.pt      # Five best validation-loss checkpoints
    ├── history.json
    ├── resolved_config.json
    ├── predictions_validation.npz
    ├── predictions_test.npz
    ├── result.json
    └── run_manifest.json
```

The derived `mixture_only` folder contains predictions, result metadata, and
completion checksums; its checkpoint path references the original reproduction.
Summaries print after completion and save mean/sample SD and paired differences
against each matched control in TXT and JSON. Full predictions, per-tag AUC/AP,
thresholds, seeds, parameter counts, input identities, and checkpoint details
remain available. SD is undefined for one seed, not reported as zero.
The full matrix has 35 fits if nothing exists; five compatible reproduced
mixture runs reduce this to 30 new fits. Late fusion and standalone stem-branch
reporting are not part of this script.

## Late-fusion training

`train_late.py` loads `configs/experiments/mtg_jamendo/late_fusion.yaml` and
inherits the same `mood_theme_tagging.yaml` -> `music2emo.yaml` recipe as
mid-fusion and the retained reproduction. Like the genre entry points, both
fusion launchers delegate directly to `experiments/_mood_theme_tagging.py` and use
the shared `configuration.py` YAML loader/validator. There is no separate
late-only configuration loader. Comparable model, fitter, and scientific
settings are shared with mid fusion; both support optional feature payload
hashing without changing the retained reproduction. CLI values override the
resolved YAML; repeated `--seed` values replace the default seed list.

### Independent branches and seed pairing

For each outer seed 32, 64, 128, 256, 512, train/select six mixture classifiers
and five stem classifiers. Every branch has its own complete chord Transformer
and `1545 -> 512 -> 256 -> 56` classifier, with 940,216 trainable parameters.
There is no shared trainable state across branches. Every mixture branch sees
the same mixture caches, but uses an independently derived training seed.
Each stem sees its own MERT, chords, and key; the `other` source is excluded.

| Position | Mixture branch | Stem branch paired at that position |
| --- | --- | --- |
| 0 | M1 / `mixture_1` | Original mixture reference |
| 1 | M2 / `mixture_2` | Guitar |
| 2 | M3 / `mixture_3` | Drums |
| 3 | M4 / `mixture_4` | Piano |
| 4 | M5 / `mixture_5` | Bass |
| 5 | M6 / `mixture_6` | Vocals |

Position zero uses the outer seed. Positions 1..5 use the existing FMA
`branch_seed()` rule: the first four SHA256 bytes of
`stemfusion-phase2:<outer_seed>:<position>`, interpreted big-endian. Matching
mixture/stem positions use the same derived seed but are trained separately.
The branch registry saves both outer and actual training seeds. Models are
not copied to create artificial independent branches.

Training remains Adam `1e-4`, batch size 8, maximum 200 epochs, author
frequency-weighted BCE, and validation-loss early stopping with patience 10
and minimum delta `0.0001`. Each branch selects its own lowest-validation-loss
checkpoint; the top five plus last are retained and interrupted optimizer
states can resume. Training pools a random consecutive segment range; each
branch's validation/test pools every segment, as in the retained reproduction.

M1 uses the unchanged reproduction fitter and configured
`paths.reference_output_dir`. Compatible completed reference runs are reused;
missing runs may train and compatible incomplete runs may resume. Incompatible
reference runs fail rather than being silently reused. `--overwrite` never
overwrites completed historical reference runs; new branches/ensemble results
have their own output root.

### Ensembles and standalone results

Each seed reports these conditions:

| Stem ensemble | Matched mixture-only ensemble |
| --- | --- |
| M+G+D+P | M1+M2+M3+M4 |
| M+G+D+P+B | M1+M2+M3+M4+M5 |
| M+G+D+P+B+V | M1+M2+M3+M4+M5+M6 |

The selected branches' **raw logits are summed without weights**, then sigmoid
produces 56 independent tag probabilities. This is not softmax, voting, or
averaging probabilities. Both kinds of ensemble use exactly this aggregation.
No ensemble checkpoint or weights are optimized against test data. Saved
prediction IDs, ordering, targets, branch completion checksums, and paired
outer seeds must agree before assembly.

Each assembled ensemble independently chooses one global threshold on
validation macro F1, using the same inclusive 0.00-to-1.00 / step-0.01 grid
and lowest-threshold tie rule as mid-fusion. Test uses that frozen threshold.
Branch thresholds are not averaged. Macro average precision and ROC-AUC are
threshold-free; label-wise accuracy, macro F1/precision/recall, and subset
accuracy follow the same definitions as mid-fusion.

The same `summary_late_fusion.{json,txt}` includes the single-mixture reference,
all six ensembles, and **Guitar Only, Drums Only, Piano Only, Bass Only, Vocals
Only**. The stem-only rows directly reuse each selected branch's predictions,
metrics, and validation threshold: no new training or checkpoint selection.
There are 12 report rows and 11 distinct branch slots per seed. The five seeds
need at most 55 branch fits, or 50 new fits when all five M1 references are
already valid. Nested ensembles reuse these branches rather than retraining
them for every combination.

### Commands

From this dataset directory, after the current training has finished:

```bash
python train_late.py --dry-run
python train_late.py --validate-only
python train_late.py
```

The dry run reads presets only and writes nothing. Validation-only reads the
required caches without fitting branches or writing results. The normal run
already includes verification, so validation-only is optional, not a separate
prerequisite. No original or separated audio is decoded; no features are
regenerated. The retained reproduction still always verifies feature checksums.
Run one GPU training process at a time unless you deliberately provision
resources for concurrent runs.

Late fusion verifies feature payload SHA256 checksums by default, controlled by
`runtime.verify_checksums: true` in `configs/experiments/mtg_jamendo/late_fusion.yaml`.
For trusted, unchanged local caches, skip only that integrity check with:

```bash
python train_late.py --no-verify-checksums
```

`--verify-checksums` explicitly enables it and overrides a YAML `false`.
Either option also works with `--validate-only`. ID/split/source, checkpoint
and preprocessing identities, segment ordering/boundaries, finite float32 shapes,
chord/key vocabulary, labels, and training prevalence checks remain mandatory.
All declared cache files must exist; only inputs used for training are read in
the trusted-cache path. Feature loading is still required, so this does not
eliminate startup I/O or guarantee a particular speedup.

Disabling checksums trusts the existing completion markers: a valid-shaped,
finite modification to cached feature values can go undetected, including when
reusing checkpoints. Keep verification enabled after copying or changing caches.
The selected mode is logged and recorded in the current summary's
`plan.verify_checksums` and `data_audit.cache_verification`; newly fitted branch
results retain their loading audit. Reused branch results preserve their original
audit rather than being rewritten. Changing this integrity option alone does
not change hyperparameters, sampling, representations, or compatible checkpoint
signatures. Saved checkpoint/prediction artifact checksums remain mandatory in
both modes. Mid fusion now uses this same optional loader; the strict
checksum-only compatibility described above preserves both completed scientific
identities. Current summaries also record the actual execution-source identity.

Equivalent installed command: `stemfusion train mtg-jamendo late-fusion`.
`--run guitar_only` evaluates/trains only the needed guitar branch, while
`--run mixture_ensemble_4` needs only the four mixture branches. `--seed 32`
selects one outer seed; `--run` and seed subsets get separate summary names.
Mixture caches are still loaded and structurally validated as the common
label/alignment reference, with the same selected checksum mode as stem caches.

An isolated bounded run is:

```bash
python train_late.py --run guitar_only --seed 32 --epochs 1 \
  --output-dir /path/to/separate/late-smoke-test
```

An explicit output root also redirects summary and reference roots unless
their CLI options are explicitly supplied. All checkpoint paths are supplied
through YAML/CLI or saved branch records, not hardcoded model paths in Python.

### Artifacts

```text
outputs/mtg_jamendo/
├── summary_late_fusion.json
├── summary_late_fusion.txt
└── late_fusion/
    ├── branches/<branch_id>/seed_<actual_training_seed>/
    │   ├── best.pt / last.pt / checkpoints/    # New non-M1 branches
    │   ├── history.json / resolved_config.json
    │   ├── predictions_validation.npz
    │   ├── predictions_test.npz
    │   └── result.json / run_manifest.json
    └── ensembles/<condition>/seed_<outer_seed>/
        ├── predictions_validation.npz
        ├── predictions_test.npz
        └── result.json / run_manifest.json
```

M1's branch folder contains derived evaluation/prediction files and points to
the verified original checkpoint; it does not duplicate or overwrite its
weights. The one late-fusion summary contains both main and supporting stem
tables, mean/sample SD over outer seeds, paired differences against the matched
mixture ensembles, actual branch seeds, checkpoint epochs/paths, thresholds,
per-tag AUC/AP, cohort/input identities, and complete branch membership.
There is no separate stem-only training pass or summary file.
