# Dataset experiment configurations

Only [FMA-Medium](fma_medium/README.md) is currently active. Its task files
reference shared feature and training presets.

Pass an experiment file with `--config`. For training, `--run` selects
one named source condition. Experiment values override their referenced
preset, and explicit CLI arguments override YAML.

The configuration loader never launches every listed run automatically.
Matched controls, independent branch training, and five-seed evaluation remain
pending. See [configuration usage](../../docs/configuration.md).
