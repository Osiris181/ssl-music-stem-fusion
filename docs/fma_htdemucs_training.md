# FMA-Medium training protocol

FMA-Medium is the first dataset implemented with the revised shared training
structure. Its genre task predicts 16 classes from frozen MusicFM-MSD layer-7
embeddings. This guide describes implemented behavior; the context design
document remains the historical planning record.

## Files and configuration

Public scripts are `src/stemfusion/experiments/fma_medium/train_mid.py`
and `train_late.py`. FMA scheduling is in the adjacent `training.py`.
Shared modules are `training/trainer.py`, `mid_fusion.py`, `late_fusion.py`,
`metrics.py`, and `reporting.py`. FMA feature loading is in
`data/fma_features.py`.

Both methods inherit `configs/training/fma_medium_genre.yaml`.
Set manifest, feature directory, and common output root there.
The mid/late experiment YAMLs define their condition matrices.
No Python settings need changing for another filesystem location.

## Running

From `src/stemfusion/experiments/fma_medium/`:

```bash
python train_mid.py --dry-run
python train_late.py --dry-run
python train_mid.py --validate-only
python train_mid.py
python train_late.py
```

Dry runs inspect the manifest, cache paths, condition definitions, parameter
counts, and seed schedule without loading feature contents or writing outputs.
Validation-only runs check all cached vectors and extraction reports without
training or writing outputs. Neither workflow accesses HDD audio.

Without `--run`, training executes its complete matrix over five seeds.
For one condition/seed, use `--run matched_mixture_4 --seed 32`.
Repeated `--seed` replaces the YAML seed list. Use a different output root
for experiments with different settings.

From the project root the package interface is equivalent:

```bash
PYTHONPATH=src python -m stemfusion train fma-medium mid-fusion
PYTHONPATH=src python -m stemfusion train fma-medium late-fusion
```

## Conditions and parameter counts

| Condition | Standard or matched architecture | Trainable parameters |
| --- | --- | ---: |
| Mixture reference | 1024 → 1024 → 512 → 16 | 1,582,608 |
| Four-source fusion | 4096 → 1024 → 512 → 16 | 4,728,336 |
| Matched mixture (4) | 1024 → 3071 → 512 → 16 | 4,728,847 |
| Five-source fusion | 5120 → 1024 → 512 → 16 | 5,776,912 |
| Matched mixture (5) | 1024 → 3753 → 512 → 16 | 5,777,081 |
| Six-source fusion | 6144 → 1024 → 512 → 16 | 6,825,488 |
| Matched mixture (6) | 1024 → 4435 → 512 → 16 | 6,825,315 |

Counts come from the actual three-biased-linear-layer model. Width matching
minimizes the absolute count difference over neighboring integer widths.
Controls receive one mixture vector, rather than repeated copies.
Sources are ordered mixture, guitar, drums, piano, bass, vocals; `other`
is excluded.

Late fusion compares each 4/5/6-source ensemble with the same number of
independently trained mixture branches. Each branch uses the standard MLP.
Branches are individually selected, and logits are summed equally in fixed
source order before softmax. There is no ensemble-specific test tuning.

## Shared training and evaluation

- Outer seeds: 32, 64, 128, 256, 512; splits remain unchanged.
- MLP: two hidden layers 1024/512, ReLU, dropout 0.2, biased linear layers.
- Loss: unweighted multiclass cross-entropy.
- Adam: lr 1e-4, betas (0.9, 0.999), eps 1e-8, weight decay 0,
  amsgrad/foreach/fused disabled.
- Batch size 128; 20 epochs; no scheduler or early-stopping patience.
- Best checkpoint: validation micro accuracy + macro one-vs-rest ROC-AUC +
  macro average precision; each component on [0, 1]. Exact ties retain the
  earliest epoch.
- Main test metric: micro accuracy. Secondary metrics: macro precision,
  recall, F1, ROC-AUC, average precision; save per-class AUCs/confusion matrix.
- A missing class in any split or nonfinite metric is an explicit error.
  Precision/recall/F1 use zero_division=0.
- No extra normalization or silent-vector exclusion. All 18 source/split
  caches must have exact manifest IDs and finite float32 (1024,) vectors.
- Training loaders shuffle; validation/test loaders preserve manifest order.
  Python, NumPy, torch, worker, and loader RNGs are seeded. Deterministic
  torch algorithms and cuDNN settings are enabled.

Branch index 0 uses the outer seed. Index j>0 uses the first 32 SHA256 bits
of `stemfusion-phase2:<outer_seed>:<j>`. Stem and mixture branches in the
same ensemble position receive paired seeds. All mixture branches within an
ensemble have distinct seeds. M1 and the ordinary mixture reference are shared.
A full mid+late run needs 17 distinct classifier fits per outer seed (85 total).

## Outputs and safe reuse

Under `outputs/fma_medium/phase2/`:

```text
branches/seed_32/mixture_1/       # also the mid mixture-only reference
branches/seed_32/mixture_2/      # further mixture voters
branches/seed_32/guitar/         # independently selected stem branch
mid_fusion/<condition>/seed_32/
late_fusion/<condition>/seed_32/
plans/
data_audit_<fingerprint>.json
```

The complete reports are saved one level higher, directly under
`outputs/fma_medium/`, as `summary_mid_fusion.txt`/`.json` and
`summary_late_fusion.txt`/`.json`. `paths.summary_dir` in the shared YAML
controls this location; `--summary-dir` overrides it. An explicit
`--output-dir` also redirects summaries there unless `--summary-dir` is given.
Single-condition or seed-subset reports retain identifying suffixes so they
do not overwrite the complete five-seed reports.

Each fit writes checkpoint.pt, resolved_config.json, environment.json,
training_history.json, validation_predictions.npz, predictions.npz,
metrics.json, and run_manifest.json. Predictions include IDs, targets,
logits, probabilities, and argmax classes. Ensemble metadata lists every
selected branch checkpoint, seed, best epoch, and validation metrics.

Automatic reuse checks input hashes, effective settings, implementation hashes,
software/device metadata, and artifact checksums. It reuses complete fits
only. Partial fits require a new output root or explicit `--overwrite`;
optimizer-state resume is not implemented. Both methods share compatible
mixture-reference checkpoints automatically, including when rerunning either
command. `--resume` is accepted for compatibility; no flag is needed.
Use `--overwrite` explicitly to retrain and replace existing outputs.

Summaries use mean ± sample SD (ddof=1), preserving seed order and reporting
paired fusion-minus-control differences. Single-seed subsets have null SD.
Formal significance tests are not performed.

After each training command finishes, the console prints a table of test
accuracy, macro F1, macro ROC-AUC, and macro average precision (PR-AUC), followed
by paired differences when matched controls are included. Values use the 0–1
scale and mean +/- sample SD; single-seed runs show metrics without an SD.
The same readable table is saved as a `.txt` companion with the same basename
as the JSON. It includes dataset/feature settings, track counts, source
abbreviations, metric definitions, and matched-control explanations so others
can interpret it independently. The console prints both
report paths. Precision, recall, per-seed values, and full-precision metrics
remain available in the detailed JSON.

The repository's `.gitignore` allows `outputs/**/summary_*.txt` and
`outputs/**/summary_*.json` to be committed. Checkpoints, prediction arrays,
per-run metadata, and other generated files remain ignored. Summary files
are not automatically staged or committed.

## Validation and boundaries

Synthetic tests exercise the complete five-seed matrix, actual parameter
matching, independent branch reuse, deterministic repeat fits, checkpoint
selection without test access, cache alignment, and paired reporting.
Full real-data training has not been launched by the implementation work.
Historical FMA test-based selection of these stem sets must still be disclosed.

The previous training code/configs/tests are preserved in
`/home/osiris181/stem-fusion-training-archive-20261029/before-phase2-training.tar.gz`.
Other datasets and non-genre task adapters remain future work.
