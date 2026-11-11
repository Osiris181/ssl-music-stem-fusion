# Training configurations

| File | Purpose |
| --- | --- |
| `genre.yaml` | Common architecture, optimization, metrics, validation selection, and five outer seeds for all three genre datasets. |
| `music2emo.yaml` | Shared MERT/chord/key feature construction, chord Transformer/MLP architecture, optimizer, loss/stopping/checkpoint recipe, runtime, and five seeds. |
| `mood_theme_tagging.yaml` | Inherits `music2emo.yaml`; supplies the 56-label head and macro AUROC/average-precision evaluation for Jamendo. |
| `emotion_recognition.yaml` | Inherits the same recipe and identifies separate binary valence/arousal tasks; a preparation-only profile, not a runnable emotion trainer yet. |
| `fma_medium_genre.yaml` | Inherits `genre.yaml`; sets 16 classes and FMA paths. |
| `fma_medium_mid_fusion.yaml` | Inherits the shared settings and selects concatenation. |
| `fma_medium_late_fusion.yaml` | Inherits the shared settings and selects independent branches with equal logit summation. |
| `gtzan_genre.yaml` | Inherits `genre.yaml`; sets ten classes and GTZAN paths. |
| `gtzan_mid_fusion.yaml`, `gtzan_late_fusion.yaml` | Select the same fusion conventions for GTZAN. |
| `ismir04_genre.yaml` | Inherits `genre.yaml`; sets six CSV genre classes and ISMIR04 paths. |
| `ismir04_mid_fusion.yaml`, `ismir04_late_fusion.yaml` | Select the same fusion conventions for ISMIR04. |

Edit [genre.yaml](genre.yaml) to change settings consistently for all genre datasets and methods.
Its settings have comments
explaining its meaning. Numerical options such as learning rate and batch size
are configurable; fixed method conventions are validated by the loader and
require implementation changes before other values are supported.

All three datasets use `early_stopping: none`: train all 20 epochs and retain the best
validation checkpoint. No patience or minimum-improvement threshold is active.

The dataset experiment YAMLs in `../experiments/{fma_medium,gtzan,ismir04}/` inherit these
profiles and define the conditions. They are loaded automatically by
`train_mid.py` and `train_late.py`. Paths resolve relative to their declaring
YAML file. Explicit CLI options override YAML and are recorded.

Outer seeds are fixed to `[32, 64, 128, 256, 512]` by default. For a bounded
check, `--seed 32` selects a subset; `--run NAME` selects a condition.
Use a separate output root when changing experimental settings. See
[the training guide](../../docs/fma_htdemucs_training.md).

## MERT plus chord/key training

Jamendo's [reproduction preset](../experiments/mtg_jamendo/reproduction.yaml)
inherits `mood_theme_tagging.yaml`, which inherits `music2emo.yaml`. The dataset
preset supplies paths and the mixture input; shared profiles supply the model
and training recipe. The completed baseline remains unchanged: Adam `1e-4`,
batch size 8, at most 200 epochs, author frequency-weighted BCE, validation-loss
early stopping with patience 10 and minimum delta `0.0001`, and lowest-loss
checkpoint selection. Five best checkpoints plus last are retained.

`emotion_recognition.yaml` shares these defaults without copying Jamendo paths,
56-tag output settings, or class-frequency arrays. Its task definition is
declarative until emotion training is implemented. Dataset-specific binary
labels, output width, prevalence weights, thresholds, and metric averaging must
be verified when adding the corresponding experiment; the preset does not
make the current Jamendo trainer a binary-emotion trainer.

Only preset locations and inheritance changed. Existing feature/checkpoint
directories and saved scientific provenance are not renamed or rewritten.
MERT extraction remains in each dataset's experiment YAML and the shared
`configs/features/mert_v1_95m.yaml`, not in training profiles.

Jamendo's `experiments/mtg_jamendo/mid_fusion.yaml` inherits the same mood/theme
recipe. Its shared chord encoder, complete-vector repeated-mixture controls,
synchronized segment sampling, and validation-only global F1 threshold are
explicit fixed conventions validated by the loader. The single-mixture
condition uses the unchanged reproduction fitter, while new multi-source
conditions use `training/music2emo.py` with the same loss/stopping rules.
