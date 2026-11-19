# GTZAN genre classification

These entry scripts use the same MusicFM extractor, genre experiment
orchestration, MLPs, trainer, checkpoint selection, evaluation, and reporting
as FMA-Medium. Dataset adapters handle GTZAN paths, IDs, ten-class labels,
and cache filenames.

| File | Purpose |
| --- | --- |
| `prepare_manifest.py` | Read provided split lists and bind retained tracks to local audio paths. |
| `separation.py` | Run the shared HT-Demucs backend with GTZAN paths and WAV input discovery. |
| `extraction.py` | Extract MusicFM embeddings with the shared encoder and audio policy. |
| `extraction_cnn.py` | Extract Phase 3 frozen official MSD_musicnn features into a separate cache. |
| `train_mid.py` | Run the standard mixture baseline, three capacity controls, and three stem-fusion conditions. |
| `train_mid_cnn.py` | Run the same seven mid-fusion conditions using frozen CNN caches and the shared genre trainer. |
| `train_late.py` | Run three mixture ensembles, three stem ensembles, and supporting standalone-stem reports. |

## Phase 3 CNN extraction

This is a separate extractor, not a replacement for MusicFM. It reuses the
existing GTZAN manifest (499/250/250), labels, split assignments and six source
paths. Configuration is `configs/experiments/gtzan/extraction_cnn.yaml`;
YAML paths resolve relative to that file and CLI paths to the working directory.
The reusable backend is `features/musicnn.py`; orchestration is shared with
ISMIR04 in `experiments/_cnn_extraction.py` and `experiments/_cnn_genre.py`.
No new stem separation is needed.

Use the isolated official TensorFlow environment, not the PyTorch training
environment. Recreate the verified Python 3.12 runtime from the repository root:

```bash
UV_PROJECT_ENVIRONMENT="$PWD/musicnn" uv sync --project environments/musicnn --inexact
uv pip install --python musicnn/bin/python --no-deps 'musicnn @ git+https://github.com/jordipons/musicnn.git@516acb2a0ff5ef73f64547898e018e793152c506'
```

The separate [environment project](../../../../../environments/musicnn/README.md)
documents setup, activation and the dependency override. `--inexact` preserves
the separately installed official package on later syncs.
Do not recreate an already installed environment. The original package declares
an obsolete NumPy `<1.17` requirement; the verified runtime deliberately uses
NumPy 1.26.4. This is a documented dependency-metadata override; model source
and weights are unchanged and checked by SHA256. `uv pip check` therefore
reports this one known conflict. Do not resolve it by downgrading NumPy.

From the repository root:

```bash
# Validate configuration and cohort without loading audio or TensorFlow.
musicnn/bin/python src/stemfusion/experiments/gtzan/extraction_cnn.py --dry-run

# Full extraction, resumable by rerunning the same command.
musicnn/bin/python src/stemfusion/experiments/gtzan/extraction_cnn.py
```

With the musicnn environment activated, you can also run `python extraction_cnn.py`
directly from this GTZAN directory. The entry point automatically supplies
environment-local CUDA library paths and legacy Keras, restarting the Python
process once before imports if necessary. No separate launcher is required;
the existing launchers remain compatible. The YAML GPU memory cap is 4096 MB;
`--gpu-memory-mb N` overrides it. CPU inference is explicitly
available with `--device /CPU:0`. GPU errors never silently switch to CPU.
`--limit-per-split 1 --output-dir /tmp/gtzan-cnn-test --report /tmp/gtzan-cnn-test/report.json`
tests one track per split without writing into the production cache.

The frozen standard `MSD_musicnn` checkpoint produces 200-dimensional
penultimate vectors. Official mono 16 kHz log-mel processing is retained
(512-point FFT, hop 256, 96 bins, float16 mel before log scaling). Consecutive
187-frame patches have no overlap. Unlike the original extractor, which drops
incomplete patches and fails on short clips, the final partial patch is retained
and right-padded with zero log-mel values (silence). Every patch, including the
final one, has equal weight in the track mean. This explicit adaptation can
change results relative to the unmodified extractor.

All sources are resampled first and aligned in memory to the original mixture's
sample count: trim excess samples or zero-pad missing samples only within the
configured 0.25-second tolerance; larger differences are reported as failures.
This aligns analysis boundaries, not a measured waveform-latency correction.
Audio files are never rewritten. Silent inputs remain valid. Spectrograms are
computed over the full track; inference is batched and pooled by window count,
not by batch means. Long-recording streaming remains future work.

Caches: `data/features/gtzan/msd_musicnn/{split}/{source}/{track_id}.npz`.

All CNN datasets share [guarded decoding and cache recovery](../../../../../docs/cnn_audio_decoding.md).
Healthy caches remain reusable; invalid decodes receive a checked FFmpeg
fallback. Genuine source corruption/mismatches remain explicit failures.
Each contains `embedding` (finite float32, shape 200) and JSON `metadata`,
readable with `numpy.load(..., allow_pickle=False)`. Provenance records the
manifest, implementation/checkpoint hashes, software versions, input file hash,
sample/frame/window counts, alignment and padding policy. Reruns reuse valid
matching caches; changed file size/mtime or settings fail compatibility checks.
The fast resume check uses size/mtime, not a fresh content hash; the original
content hash remains recorded for auditing. Use `--overwrite` only to replace
caches deliberately. Do not run concurrent writers on the same cache directory.

A tqdm bar reports all selected views. The readable final count points to
`outputs/gtzan/cnn/extraction_report.json`, containing failures and provenance.
Failures return nonzero; no track is silently excluded from the intended
cohort. CNN classifier training is a subsequent step; the Phase 2 trainers
continue to load MusicFM, not these caches. MSD pretraining overlap with the
downstream collections remains unresolved and must not be described as ruled out.

### CNN mid-fusion training

After extraction finishes, switch back to your main **PyTorch training
environment**; the isolated `(musicnn)` environment is for TensorFlow extraction
only. From this GTZAN directory:

```bash
python train_mid_cnn.py --validate-only
python train_mid_cnn.py --dry-run
python train_mid_cnn.py
```

`configs/experiments/gtzan/mid_fusion_cnn.yaml` inherits the active
`mid_fusion.yaml` protocol, including the seven named conditions and genre
training profile. Only the representation (200-dimensional MSD_musicnn) and
paths change. The same genre MLP/trainer/metrics/reporting code is used:
1024/512 hidden widths, ReLU, dropout 0.2, Adam 1e-4, batch 128, 20 epochs,
no early stopping, and best validation accuracy + macro ROC-AUC + macro AP
(earliest ties). Seeds remain 32, 64, 128, 256, 512. Test evaluation occurs
after checkpoint selection. No normalization or chord/key features are added.

The 35 fits comprise mixture only, identical mixture copies x4/x5/x6, and
M+G+D+P / M+G+D+P+B / M+G+D+P+B+V. Input widths are 200, 800, 1000,
and 1200. Repeated-mixture and corresponding fusion MLPs match exactly in
nominal parameter count. No CNN late-fusion or standalone-stem runs are added.

The cache adapter requires all manifest IDs in all six sources, finite float32
vectors, matching checkpoint/extraction provenance and aligned window counts.
It reads only cached features and the manifest, not audio or TensorFlow.
Incomplete or mixed caches fail validation rather than filtering tracks.

Checkpoints and per-run metrics save under `outputs/gtzan/cnn/runs/`.
The final results print in the terminal and save as
`outputs/gtzan/cnn/summary_mid_fusion.json` and `.txt`, including mean/sample SD
and paired fusion-minus-control differences. Subset runs use distinct summary
suffixes. Rerunning reuses verified completed fits; `--overwrite` explicitly
retrains them. MusicFM checkpoints and summaries are untouched.
Use `--run NAME`, repeated `--seed N`, or the same supported training overrides
for diagnostics; default full experiments use the inherited protocol unchanged.
An explicit `--output-dir` redirects summaries too unless `--summary-dir` is set.

## Configuration and data

| YAML | Purpose |
| --- | --- |
| `configs/experiments/gtzan/manifest.yaml` | Split files, local audio roots/extensions, availability handling, and manifest output. |
| `configs/experiments/gtzan/separation.yaml` | Original audio and separated-output roots; inherits shared HT-Demucs inference settings. |
| `configs/experiments/gtzan/extraction.yaml` | Shared MusicFM profile, selected sources, manifest and cache paths. |
| `configs/features/musicfm_msd_layer7.yaml` | MusicFM checkout, MSD checkpoint, layer 7, and shared extraction settings. |
| `configs/training/genre.yaml` | Common genre architecture, optimization, metrics, checkpoint selection, and five seeds for all genre datasets. |
| `configs/training/gtzan_genre.yaml` | Ten classes and GTZAN manifest/cache/run/summary paths. |
| `configs/experiments/gtzan/mid_fusion.yaml`, `late_fusion.yaml` | The same condition matrices as FMA-Medium. |

Each script automatically loads its YAML. Paths in YAML resolve relative to
the file declaring them; CLI paths resolve relative to the current directory.
Explicit arguments override YAML. Use `--config PATH` to supply another preset.

The provided stratified files define the split assignments. Manifest
preparation uses file availability to construct a common source-aligned
track set. The same IDs, labels, and ordering are used across every condition
and seed. The metadata pickle is a reference copy; code obtains splits from
the text files and labels from the fixed genre mapping in `data/gtzan.py`.
Generated manifests contain machine-specific paths and remain Git-ignored.

Caches are `gtzan_musicfm_{source}_{split}.pkl`, where the original mixture
uses filename alias `mix`, and splits are `train`, `validation`, and `test`.
The loader accepts NumPy arrays and CPU PyTorch tensors, validates finite
float32 vectors, retains valid zero vectors, and applies no normalization.
All source caches must contain exactly the manifest IDs.

Existing imported caches are verified against `import_audit.json`, including
checksums and the original extraction script's declared settings. Their
checkpoint/layer provenance is inferred from that script, not embedded in the
pickles. New caches use the shared extractor's per-source reports. These two
provenance routes are recorded separately; original caches are not described
as outputs of the cleaned script.

## Commands

From `src/stemfusion/experiments/gtzan/`:

When separated audio needs to be generated:

```bash
python separation.py --dry-run
python separation.py
```

Complete existing stem folders are skipped. The YAML defaults to WAV inputs
under the original `genres/` directory. Inference settings come from the same
`configs/separators/htdemucs_6s.yaml` used by FMA-Medium and ISMIR04. Dataset
path options can be overridden on the command line.

```bash
# Configure your split/audio paths in manifest.yaml first.
python prepare_manifest.py --dry-run
python prepare_manifest.py

# Reuse the existing imported caches: validate them before training.
python train_mid.py --validate-only
python train_mid.py --dry-run
python train_late.py --dry-run

# Full five-seed matrices, including evaluation after checkpoint selection.
python train_mid.py
python train_late.py
```

The local manifest is already prepared for the imported features. Do not
regenerate it just to train. Use `--overwrite` explicitly if rebuilding a
manifest. Preparation checks that configured audio files exist and are
non-empty; it does not decode waveforms. Dry runs perform these checks too.
Use `--no-allow-missing` to require every listed record. To bind paths without
availability checks, supply both `--no-check-audio` and `--no-allow-missing`.

Feature extraction is available for users who need new caches:

```bash
python extraction.py --dry-run
python extraction.py
```

The shared encoder uses MusicFM-MSD layer index 7, full-duration mono audio
at 24 kHz, temporal mean pooling, and 1024-dimensional output. New vectors
are saved as CPU float32 tensors and can be read by the same loader as the
original NumPy caches. Existing cache files are skipped unless `--overwrite`
is supplied. Use a new output directory when intentionally changing features;
every condition must use the same aligned feature set.

Unified commands are also available:

```bash
stemfusion data gtzan-manifest
stemfusion extract gtzan --dry-run
stemfusion train gtzan mid-fusion
stemfusion train gtzan late-fusion
```

## Training and outputs

All genre datasets inherit the same 1024/512 hidden widths, ReLU, dropout 0.2,
cross-entropy, Adam at 1e-4, batch size 128, and 20 epochs without early
stopping. Checkpoint selection maximizes validation micro accuracy + macro
ROC-AUC + macro average precision, retaining the earliest checkpoint on ties.
Test scores never select checkpoints. Outer seeds are 32, 64, 128, 256, 512;
branch seeds use the existing SHA256 convention and shared branch positions.

Mid fusion compares M+G+D+P, M+G+D+P+B, and M+G+D+P+B+V against the standard
mixture baseline and three repeated-mixture MLPs (4/5/6 identical mixture
vectors, unchanged 1024/512 hidden widths, exact fusion parameter counts).
Late fusion sums
independently selected logits equally, then applies softmax, with matching
4/5/6-member mixture ensembles. There are 85 distinct fits across five seeds,
including the mixture baseline shared between workflows.

Run artifacts go in `outputs/gtzan/runs/`. Selected weights are
`branches/seed_<seed>/<branch>/checkpoint.pt` for independent branches and
`mid_fusion/<condition>/seed_<seed>/checkpoint.pt` for fused/capacity-control
MLPs. Late-fusion ensemble records reference the selected branch weights.
Completed compatible runs are reused automatically; `--overwrite` explicitly
retrains. Use `--seed 32` or `--run NAME` for a subset.

Complete summaries are saved directly in `outputs/gtzan/`:

```text
summary_mid_fusion.txt / .json
summary_late_fusion.txt / .json
summary_stem_only.txt / .json
```

Results also print to the terminal. TXT and JSON summaries can be tracked by
Git; checkpoints and predictions remain ignored. Reports contain means and
sample SD across seeds and paired differences against matched controls.
Full late-fusion runs include the five standalone stem evaluations in the
late-fusion report and write their separate supporting summaries. To rebuild
these from saved branches without training or inference:

```bash
python train_late.py --report-stems-only
```
