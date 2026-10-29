# Configurations

These YAML files now supply settings to the public CLI tasks.

| Directory | Usage |
| --- | --- |
| `separators/` | HT-Demucs inference defaults, loaded by `separate htdemucs`. |
| `features/` | MusicFM defaults, loaded by `extract fma-medium`. |
| `training/` | Classifier/optimization defaults, loaded by the corresponding FMA training task. |
| `experiments/fma_medium/` | Profile references, named source conditions, and optional task paths. |

Each task automatically loads its standard preset. Use `--config PATH`
for another YAML file; explicit CLI values override YAML. A training
experiment's `--run NAME` selects one named source condition.

Profile references and paths inside YAML are relative to that file. Keep
machine-specific overrides in an ignored local file, such as
`configs/paths.local.yaml`. Unknown or unsupported settings fail explicitly.

See [configuration usage](../docs/configuration.md) for examples, supported
path fields, precedence, and provenance recording.

Comparable methods must retain consistent splits, preprocessing, architecture,
loss, optimization, metrics, and selection rules. YAML loading does not
implement the pending matched controls or five-seed protocol.
