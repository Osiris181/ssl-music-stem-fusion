# Feature configurations

`musicfm_msd_layer7.yaml` is the shared preset referenced by the FMA task's
default `configs/experiments/fma_medium/extraction.yaml`. The FMA entry point
selects that task YAML. This feature preset supplies checkpoint/statistics names,
sample rate, duration handling, encoder layer, pooling, dimension, runtime,
and split defaults.

Set `paths.musicfm_root` in this preset to your MusicFM checkout. FMA manifest
and output paths are configured in `configs/experiments/fma_medium/extraction.yaml`.
No separate local-path YAML is required.

```bash
stemfusion extract fma-medium --help
```

Use `--config PATH` for an alternate preset or an experiment referencing
this file. Explicit CLI values override configured ones. Checkpoint/statistics
basenames are resolved under the configured MusicFM checkout's `data/`
directory unless an explicit file path overrides them.

See [configuration usage](../../docs/configuration.md). The RMS preset remains
archived outside the project.
