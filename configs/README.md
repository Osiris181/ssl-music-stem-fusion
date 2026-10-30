# Configurations

These YAML files now supply settings to the public CLI tasks.

| Directory | Usage |
| --- | --- |
| `separators/` | HT-Demucs inference defaults, loaded by `separate htdemucs`. |
| `features/` | Shared MusicFM defaults, loaded by FMA-Medium and GTZAN extraction. |
| `training/` | Common genre settings inherited by dataset and mid-/late-fusion profiles. |
| `experiments/fma_medium/` | Profile references, named source conditions, and optional task paths. |
| `experiments/gtzan/` | Provided split/audio paths for manifest generation, extraction settings, and the same condition matrices. |

Each task automatically loads its standard preset. Use `--config PATH`
for another YAML file; explicit CLI values override YAML. A training
experiment's `--run NAME` selects one named condition. Omit it to run the
complete matrix; `--seed` selects an outer-seed subset.

Profile references and paths inside YAML are relative to that file. Set the
MusicFM checkout location in `features/musicfm_msd_layer7.yaml` and FMA input/output
paths in `experiments/fma_medium/extraction.yaml`. No separate local-path YAML
is required. Unknown or unsupported settings fail explicitly.

See [configuration usage](../docs/configuration.md) for examples, supported
path fields, precedence, and provenance recording.

Comparable methods must retain consistent splits, preprocessing, architecture,
loss, optimization, metrics, and selection rules. The shared genre training
profile `training/genre.yaml` sets the five seeds and common hyperparameters. The experiment
matrices include matched controls and independent late-fusion branches.
