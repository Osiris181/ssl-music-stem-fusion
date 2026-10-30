# GTZAN genre classification

These entry scripts use the same MusicFM extractor, genre experiment
orchestration, MLPs, trainer, checkpoint selection, evaluation, and reporting
as FMA-Medium. Dataset adapters handle GTZAN paths, IDs, ten-class labels,
and cache filenames.

| File | Purpose |
| --- | --- |
| `prepare_manifest.py` | Read provided split lists and bind retained tracks to local audio paths. |
| `extraction.py` | Extract MusicFM embeddings with the shared encoder and audio policy. |
| `train_mid.py` | Run the standard mixture baseline, three capacity controls, and three stem-fusion conditions. |
| `train_late.py` | Run three mixture ensembles, three stem ensembles, and supporting standalone-stem reports. |

## Configuration and data

| YAML | Purpose |
| --- | --- |
| `configs/experiments/gtzan/manifest.yaml` | Split files, local audio roots/extensions, unavailable track exclusions, and manifest output. |
| `configs/experiments/gtzan/extraction.yaml` | Shared MusicFM profile, selected sources, manifest and cache paths. |
| `configs/features/musicfm_msd_layer7.yaml` | MusicFM checkout, MSD checkpoint, layer 7, and shared extraction settings. |
| `configs/training/genre.yaml` | Common genre architecture, optimization, metrics, checkpoint selection, and five seeds for both datasets. |
| `configs/training/gtzan_genre.yaml` | Ten classes and GTZAN manifest/cache/run/summary paths. |
| `configs/experiments/gtzan/mid_fusion.yaml`, `late_fusion.yaml` | The same condition matrices as FMA-Medium. |

Each script automatically loads its YAML. Paths in YAML resolve relative to
the file declaring them; CLI paths resolve relative to the current directory.
Explicit arguments override YAML. Use `--config PATH` to supply another preset.

The provided stratified files contain 500/250/250 track IDs. Unavailable
training track `jazz.00054` is explicitly excluded, leaving **499/250/250**
tracks. The same IDs, labels, and ordering are used across every condition
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
manifest. Preparation binds audio paths without scanning audio by default;
`--check-audio` opts into validating all configured files.

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

Both datasets inherit the same 1024/512 hidden widths, ReLU, dropout 0.2,
cross-entropy, Adam at 1e-4, batch size 128, and 20 epochs without early
stopping. Checkpoint selection maximizes validation micro accuracy + macro
ROC-AUC + macro average precision, retaining the earliest checkpoint on ties.
Test scores never select checkpoints. Outer seeds are 32, 64, 128, 256, 512;
branch seeds use the existing SHA256 convention and shared branch positions.

Mid fusion compares M+G+D+P, M+G+D+P+B, and M+G+D+P+B+V against the standard
mixture baseline and three capacity-matched mixture MLPs. Late fusion sums
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
