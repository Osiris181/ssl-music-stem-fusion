# Dataset experiment configurations

[FMA-Medium](fma_medium/README.md), [GTZAN](gtzan/README.md), and
[ISMIR04](ismir04/README.md) task files reference shared feature and
training presets. Dataset-specific profiles supply paths and class counts.

Pass an experiment file with `--config`. For training, `--run` selects
one named source condition. Experiment values override their referenced
preset, and explicit CLI arguments override YAML.

The configuration loader only resolves settings. The training entry point
runs the full condition matrix when `--run` is omitted, using all five seeds
unless a seed subset is supplied. Mid fusion includes capacity-matched
mixture controls; late fusion uses independently selected branches and
matched mixture ensembles. See [configuration usage](../../docs/configuration.md).
